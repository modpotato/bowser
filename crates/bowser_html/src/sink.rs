//! Single-pass tree sink.
//!
//! [`TreeSink`] is the consumer interface for tree construction: after the
//! tokenizer and insertion-mode stages finish, the tree builder drives the
//! sink once, in document order, *moving* string payloads out of its
//! working arena instead of cloning them. A sink that stores those payloads
//! therefore performs exactly one allocation per string, with no
//! intermediate owned tree.
//!
//! [`NodeSink`] is the canonical sink: it rebuilds the owned [`Node`] tree,
//! and [`parse`](crate::parse) / [`parse_with_quirks`](crate::parse_with_quirks)
//! are thin wrappers over [`parse_into`](crate::tree::parse_into) with it.
//! Downstream crates (e.g. `bowser_dom`) implement [`TreeSink`] directly on
//! their own arenas.

use crate::atom::{AttrName, TagName};
use crate::tree::{Doctype, Element, Namespace, Node, ShadowRoot, ShadowRootMode, SourceLoc};

/// Consumer of tree-construction output.
///
/// The driver ([`parse_into`](crate::tree::parse_into)) guarantees:
/// - events arrive in document order;
/// - every `start_element` is matched by exactly one `end_element`, with
///   the element's children (and, for `template`, its bracketed contents)
///   delivered in between;
/// - `start_template_contents` / `end_template_contents` bracket a
///   template element's contents fragment, delivered after the element's
///   own children (mirroring [`Element::template_contents`]; in DOM order
///   the fragment is the template element's last child);
/// - `start_shadow_root` / `end_shadow_root` bracket a declarative shadow
///   root (WHATWG HTML 13.2.6.4.7): the driver emits them in place of a
///   `<template shadowrootmode>` element, at the template's position
///   among the host's children, with the template's contents delivered
///   inside. A start is never emitted while the innermost open element
///   already has an open shadow bracket; brackets for a *descendant* host
///   opened inside the shadow tree may still appear inside an ancestor's
///   bracket in the event stream, and pair with that descendant's own
///   element brackets.
/// - all string payloads are moved, never cloned by the driver.
///
/// The walk is iterative, so adversarially nested input cannot overflow the
/// call stack through the sink.
///
/// [`TreeSink`] is a public extension point: a third-party driver may not
/// honor the balance guarantees above. Implementations must therefore fail
/// closed on unbalanced calls (`debug_assert!` plus a graceful no-op in
/// release) rather than panicking.
pub trait TreeSink {
    /// A `<!DOCTYPE ...>` child of the document.
    fn doctype(&mut self, doctype: Doctype);
    /// A text node.
    fn text(&mut self, text: String, loc: Option<SourceLoc>);
    /// A comment node.
    fn comment(&mut self, data: String, loc: Option<SourceLoc>);
    /// A processing instruction node.
    fn processing_instruction(&mut self, target: String, data: String, loc: Option<SourceLoc>);
    /// An element open: children (then, for `template`, the bracketed
    /// contents) follow, closed by [`TreeSink::end_element`].
    ///
    /// The names arrive as borrowed strings and are interned by the sink.
    /// Drivers that already hold interned names should call
    /// [`TreeSink::start_element_atomized`] instead to skip the
    /// re-interning.
    fn start_element(
        &mut self,
        tag: String,
        ns: Namespace,
        attrs: Vec<(String, String)>,
        loc: Option<SourceLoc>,
    );
    /// An element open, with tag and attribute names already interned
    /// (the tokenizer interns names as it completes them, so well-known
    /// names never allocate).
    ///
    /// The default implementation converts back to the borrowed-string
    /// form and delegates to [`TreeSink::start_element`], so existing
    /// sinks keep working unchanged; [`NodeSink`] overrides it with a
    /// pass-through that stores the atoms directly.
    fn start_element_atomized(
        &mut self,
        tag: TagName,
        ns: Namespace,
        attrs: Vec<(AttrName, String)>,
        loc: Option<SourceLoc>,
    ) {
        self.start_element(
            tag.as_str().to_string(),
            ns,
            attrs
                .into_iter()
                .map(|(name, value)| (name.as_str().to_string(), value))
                .collect(),
            loc,
        );
    }
    /// Close the element opened by [`TreeSink::start_element`].
    fn end_element(&mut self);
    /// Begin a template element's contents fragment. Always paired with
    /// [`TreeSink::end_template_contents`] before the element's
    /// [`TreeSink::end_element`].
    fn start_template_contents(&mut self);
    /// End a template element's contents fragment.
    fn end_template_contents(&mut self);
    /// Begin a declarative shadow root on the currently open element: the
    /// driver replaced a `<template shadowrootmode>` child with this
    /// bracket; the shadow children follow, closed by
    /// [`TreeSink::end_shadow_root`].
    ///
    /// The default implementation ignores the shadow root (children
    /// delivered inside the bracket land on the host element); override
    /// to keep it.
    fn start_shadow_root(&mut self, mode: ShadowRootMode, delegates_focus: bool, clonable: bool) {
        let _ = (mode, delegates_focus, clonable);
    }
    /// End a declarative shadow root opened by
    /// [`TreeSink::start_shadow_root`].
    fn end_shadow_root(&mut self) {}
}

/// A stack frame for one open element.
#[derive(Debug)]
struct Frame {
    element: Element,
    /// Buffer collecting template contents while the
    /// `start_template_contents` / `end_template_contents` bracket is open.
    template: Option<Vec<Node>>,
    /// Buffer collecting a declarative shadow root's children while the
    /// `start_shadow_root` / `end_shadow_root` bracket is open; becomes
    /// `element.shadow_root` at the closing bracket.
    shadow: Option<ShadowInProgress>,
    /// `start_shadow_root` calls seen while this frame already has a
    /// shadow bracket open. The driver never nests brackets on the same
    /// element; a buggy third-party driver that does gets fail-closed
    /// pairing instead of clobbering the outer root: each `end_shadow_root`
    /// first cancels one nested no-op start before it may close the real
    /// bracket.
    shadow_nest: u32,
}

/// A declarative shadow root being collected by [`NodeSink`].
#[derive(Debug)]
struct ShadowInProgress {
    mode: ShadowRootMode,
    delegates_focus: bool,
    clonable: bool,
    children: Vec<Node>,
}

/// [`TreeSink`] that rebuilds the owned [`Node`] tree.
///
/// This is what [`parse`](crate::parse) collects into, so the public tree
/// API and every test written against it are unaffected by the single-pass
/// refactor.
#[derive(Debug, Default)]
pub struct NodeSink {
    document: Vec<Node>,
    stack: Vec<Frame>,
}

impl NodeSink {
    /// Create an empty sink.
    pub fn new() -> Self {
        NodeSink {
            document: Vec::new(),
            stack: Vec::new(),
        }
    }

    /// Finish the tree. The stack must be empty (the driver always
    /// balances its brackets).
    pub fn into_node(self) -> Node {
        debug_assert!(self.stack.is_empty(), "NodeSink: unbalanced sink events");
        Node::Document(self.document)
    }

    /// Append a finished node to the current target: the shadow buffer
    /// when inside a shadow-root bracket, else the template buffer when
    /// inside a template-contents bracket, else the open element's
    /// children, else the document.
    fn push_node(&mut self, node: Node) {
        match self.stack.last_mut() {
            None => self.document.push(node),
            Some(frame) => {
                if let Some(shadow) = frame.shadow.as_mut() {
                    shadow.children.push(node);
                } else {
                    match frame.template.as_mut() {
                        Some(t) => t.push(node),
                        None => frame.element.children.push(node),
                    }
                }
            }
        }
    }
}

impl TreeSink for NodeSink {
    fn doctype(&mut self, doctype: Doctype) {
        self.push_node(Node::Doctype(doctype));
    }

    fn text(&mut self, text: String, loc: Option<SourceLoc>) {
        self.push_node(Node::Text { text, loc });
    }

    fn comment(&mut self, data: String, loc: Option<SourceLoc>) {
        self.push_node(Node::Comment { data, loc });
    }

    fn processing_instruction(&mut self, target: String, data: String, loc: Option<SourceLoc>) {
        self.push_node(Node::ProcessingInstruction { target, data, loc });
    }

    fn start_element_atomized(
        &mut self,
        tag: TagName,
        ns: Namespace,
        attrs: Vec<(AttrName, String)>,
        loc: Option<SourceLoc>,
    ) {
        // Names arrive already atomized from the tokenizer: store them
        // directly, no re-interning (well-known names are pure atoms with
        // zero heap use).
        self.stack.push(Frame {
            element: Element {
                tag,
                ns,
                attrs,
                children: Vec::new(),
                template_contents: None,
                shadow_root: None,
                loc,
            },
            template: None,
            shadow: None,
            shadow_nest: 0,
        });
    }

    fn start_element(
        &mut self,
        tag: String,
        ns: Namespace,
        attrs: Vec<(String, String)>,
        loc: Option<SourceLoc>,
    ) {
        self.start_element_atomized(
            TagName::from(tag),
            ns,
            attrs
                .into_iter()
                .map(|(name, value)| (AttrName::from(name), value))
                .collect(),
            loc,
        );
    }

    fn end_element(&mut self) {
        // The driver guarantees balanced calls; a buggy third-party driver
        // fails closed here instead of panicking.
        let Some(frame) = self.stack.pop() else {
            return;
        };
        let node = Node::Element(frame.element);
        self.push_node(node);
    }

    fn start_template_contents(&mut self) {
        let Some(frame) = self.stack.last_mut() else {
            // Fail-closed no-op in all builds.
            return;
        };
        frame.template = Some(Vec::new());
    }

    fn end_template_contents(&mut self) {
        let Some(frame) = self.stack.last_mut() else {
            // Fail-closed no-op in all builds.
            return;
        };
        // The driver only brackets elements that had a contents fragment,
        // so this is always `Some` (possibly empty), exactly like
        // `Element::template_contents` from the old conversion.
        frame.element.template_contents = frame.template.take();
    }

    fn start_shadow_root(&mut self, mode: ShadowRootMode, delegates_focus: bool, clonable: bool) {
        let Some(frame) = self.stack.last_mut() else {
            // Fail-closed no-op in all builds.
            return;
        };
        // The driver never nests shadow brackets on the same element; a
        // buggy third-party driver that does gets a fail-closed no-op
        // instead of clobbering the outer shadow root. The matching end
        // bracket cancels this counter (see `end_shadow_root`) rather than
        // closing the outer root early.
        if frame.shadow.is_some() {
            frame.shadow_nest += 1;
            return;
        }
        frame.shadow = Some(ShadowInProgress {
            mode,
            delegates_focus,
            clonable,
            children: Vec::new(),
        });
    }

    fn end_shadow_root(&mut self) {
        let Some(frame) = self.stack.last_mut() else {
            // Fail-closed no-op in all builds.
            return;
        };
        if frame.shadow_nest > 0 {
            // This end pairs with a nested no-op start, not with the
            // outer bracket: cancel the counter and leave the outer
            // shadow root open.
            frame.shadow_nest -= 1;
            return;
        }
        let Some(shadow) = frame.shadow.take() else {
            // Unbalanced end bracket: fail closed.
            return;
        };
        frame.element.shadow_root = Some(ShadowRoot {
            mode: shadow.mode,
            delegates_focus: shadow.delegates_focus,
            clonable: shadow.clonable,
            children: shadow.children,
        });
    }
}
