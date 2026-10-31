//! The real DOM: an arena-based node tree.
//!
//! All nodes live in a single append-only arena inside [`Document`]. Tree
//! structure is a classic first-child / next-sibling linked list, so there
//! are no reference cycles and no garbage collector: dropping the
//! [`Document`] frees everything.
//!
//! [`NodeId`]s are plain indices into the arena. Slots are never reused,
//! so a `NodeId` stays valid for the lifetime of the `Document`, across
//! any number of appends, removals, and mutations. Removed nodes are
//! detached from the tree but remain addressable (e.g. `kind` still works
//! on them). Every `NodeId` also carries its owning document's tag:
//! every accessor asserts that the tag matches, so using a `NodeId` from
//! a *different* `Document` panics loudly at the offending call instead of
//! silently addressing the wrong node. The check is unconditional (it runs
//! in release builds too; a single `u32` compare is ~free). `NodeId`s from
//! different documents never compare equal.
//!
//! Template contents convention: when `parse_html` meets a `<template>`
//! element, the template's contents fragment becomes a [`NodeKind::DocumentFragment`]
//! node that is a *child* of the template element. Serializing a template
//! serializes the fragment's children inline, so `<template><p>x</template>`
//! round-trips.
//!
//! General descendant traversal (`children`, `descendants`,
//! `text_content`, `get_elements_by_tag_name`, `query_selector_all`,
//! and the id index) does *not* cross from a `<template>` element into
//! its contents fragment, matching the spec's out-of-reach contents.
//! Traverse from the fragment node itself to see its nodes.
//!
//! Source locations from the parser are dropped on conversion (only the
//! text/attribute/doctype payloads are kept).
//!
//! No `unsafe` is used anywhere in this crate.

#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(test)]
mod tests;

pub mod anchor;

mod event;

pub use bowser_html::{
    Doctype, Namespace, QuirksMode, ScriptElement, ScriptExecutor, ScriptHost, ShadowRootMode,
    SourceLoc,
};
// Interned name atoms (see `bowser_html::atom`): re-exported so
// downstream crates (e.g. `bowser_css`'s matcher) share the one table.
pub use bowser_html::atom::{AttrName, Name, NameAtom, TagName};
pub use event::{Event, PointerData, PointerType, Touch, TouchEvent, TouchList, TouchPhase};

use bowser_html::Element as HtmlElement;
// `TreeSink` is an implementation detail here (only `DomSink` implements
// it, below); it stays a private import so it is not part of this
// crate's public API.
use bowser_html::TreeSink;

/// FNV-1a hasher (std-only, no dependency): one XOR-multiply round per
/// input byte. Far cheaper than SipHash (`RandomState`) for the short
/// keys here, and — unlike a bare rotate-multiply FxHash — its per-byte
/// mixing keeps the table index bits well distributed for short keys
/// that differ only in high bytes (e.g. sequential ids like
/// `"n1"`..`"n1999"`; measured 2.2x faster `get` than `RandomState` on a
/// 2000-id table, vs 2x *slower* for FxHash without a costly finalizer).
/// Used for the id string interner, whose keys are short id strings
/// probed on every id lookup and attribute mutation — throughput matters
/// and the fixed FNV-1a seed is accepted deliberately: id values come from
/// document content, which *is* attacker-controlled in a browser, so a
/// hostile page could craft colliding ids and degrade these tables to
/// linear chains. That worst case is acknowledged (not prevented): the
/// seed stays fixed because per-instance random seeds would cost a
/// SipHash-strength hasher's throughput on every probe, and the tables
/// are small, per-document, and dropped with the document.
#[derive(Clone, Debug, Default)]
struct FnvBuildHasher;

#[derive(Clone, Debug)]
struct FnvHasher(u64);

impl std::hash::BuildHasher for FnvBuildHasher {
    type Hasher = FnvHasher;
    #[inline]
    fn build_hasher(&self) -> FnvHasher {
        // FNV offset basis.
        FnvHasher(0xcbf29ce484222325)
    }
}

impl std::hash::Hasher for FnvHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // FNV prime.
        const P: u64 = 0x100000001b3;
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(P);
        }
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// [`HashMap`] with the [`FnvBuildHasher`] above.
type FnvHashMap<K, V> = HashMap<K, V, FnvBuildHasher>;
/// [`HashSet`] with the [`FnvBuildHasher`] above.
type FnvHashSet<T> = HashSet<T, FnvBuildHasher>;

/// Global document-tag counter. Each [`Document`] takes a unique tag at
/// construction so [`NodeId`]s carry their owner's identity. The counter
/// saturates at `u32::MAX` instead of wrapping: handing out a tag when the
/// counter is exhausted panics (2^32 live documents in one process is
/// unreachable in practice, and failing loud beats silently aliasing two
/// documents' [`NodeId`]s). Saturation matters because a caller that
/// catches the exhaustion panic must never see the counter wrap back to
/// 0 and start handing out colliding tags.
static NEXT_DOC_TAG: AtomicU32 = AtomicU32::new(1);

// ---------------------------------------------------------------------------
// Node identity and kinds
// ---------------------------------------------------------------------------

/// Identifies a node in a [`Document`]'s arena.
///
/// The arena index plus the owning document's tag: the arena is
/// append-only and slots are never recycled, so a `NodeId` remains valid
/// across appends, removals, and attribute changes for as long as the
/// `Document` lives. The tag lets every [`Document`] accessor fail closed
/// on `NodeId`s from another document (no-ops and neutral values, never
/// panics); `NodeId`s from different documents never compare equal.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId {
    index: u32,
    doc: u32,
}

impl NodeId {
    /// The arena index. Guaranteed: unique per node, stable for the
    /// document's lifetime, and always `<` the arena length. The arena is
    /// append-only (slots are pushed and never removed, `swap_remove`d,
    /// or cleared), so a `NodeId` keeps pointing at the same node across
    /// appends, removals, and attribute changes. Callers that memo on
    /// this index (e.g. bitset-based caches) may rely on it for as long
    /// as the owning [`Document`] lives.
    pub fn index(self) -> usize {
        self.index as usize
    }
}

/// What kind of node a [`NodeId`] points at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Document,
    DocumentType(DocumentTypeData),
    Element(ElementData),
    /// A text node. The payload is a shared [`Arc<str>`]: short repeated
    /// runs (whitespace between elements, repeated labels) are interned
    /// per document (see [`Document::intern_text`]), so they share one
    /// allocation instead of one per node, and cloning a text node is a
    /// refcount bump instead of a string copy. Note there is no
    /// `PartialEq<str> for Arc<str>`: compare via `t.as_ref()`.
    Text(Arc<str>),
    Comment(String),
    ProcessingInstruction {
        target: String,
        data: String,
    },
    DocumentFragment,
}

/// Payload of a `<!DOCTYPE ...>` node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentTypeData {
    pub name: String,
    pub public_id: Option<String>,
    pub system_id: Option<String>,
}

/// Payload of an element node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElementData {
    /// Interned tag name (lowercased for HTML elements by the parser and
    /// by [`Document::create_element`]; case-adjusted for SVG/MathML,
    /// e.g. `foreignObject`; kept as written by
    /// [`Document::create_element_ns`]). Well-known names are pure atoms
    /// — no per-element heap allocation; anything else spills onto the
    /// name's heap sidecar and round-trips exactly.
    pub tag_name: TagName,
    pub namespace: Namespace,
    /// The element's attributes. [`Attributes`] keeps the empty case —
    /// the common one — to a 16-byte null with no allocation (a `Vec`
    /// paid 24 bytes inline per element even when empty).
    pub attributes: Attributes,
    /// Declarative shadow root attached by the parser from
    /// `<template shadowrootmode>` (WHATWG HTML 13.2.6.4.7). `None`
    /// otherwise. The fragment is *not* linked as a light-DOM child:
    /// shadow content is out of reach of light-DOM traversal, like
    /// template contents.
    pub shadow_root: Option<ShadowRootData>,
}

/// A declarative shadow root attached to an element by the parser
/// (WHATWG HTML 13.2.6.4.7).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShadowRootData {
    /// From `shadowrootmode="open" | "closed"`.
    pub mode: ShadowRootMode,
    /// From the `shadowrootdelegatesfocus` boolean attribute.
    pub delegates_focus: bool,
    /// From the `shadowrootclonable` boolean attribute.
    pub clonable: bool,
    /// Unlinked [`NodeKind::DocumentFragment`] holding the shadow
    /// children. Traverse from the fragment to see its nodes; see
    /// [`Document::shadow_root`].
    pub root: NodeId,
}

/// Namespace URI for `xlink:*` foreign attributes.
pub const XLINK_NAMESPACE: &str = "http://www.w3.org/1999/xlink";
/// Namespace URI for `xml:*` foreign attributes.
pub const XML_NAMESPACE: &str = "http://www.w3.org/XML/1998/namespace";
/// Namespace URI for `xmlns` / `xmlns:*` foreign attributes.
pub const XMLNS_NAMESPACE: &str = "http://www.w3.org/2000/xmlns/";

/// A single attribute on an element: the interned qualified name plus
/// its value, and — for parser-created foreign attributes — the
/// namespace URI from the spec's "adjust foreign attributes" step.
///
/// The qualified name's prefix (`xlink` in `xlink:href`) and local name
/// (`href`) are *not* stored: they are derived mechanically from
/// [`Attribute::name`] on demand (see [`Attribute::prefix`] /
/// [`Attribute::local_name`]), saving two allocations per attribute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attribute {
    /// Interned qualified name, e.g. `"xlink:href"` or `"id"`. Well-known
    /// names are pure atoms, so attribute scans compare integers.
    pub name: AttrName,
    pub value: String,
    /// Namespace URI ([`XLINK_NAMESPACE`], [`XML_NAMESPACE`], or
    /// [`XMLNS_NAMESPACE`]) for parser-created `xlink:*` / `xml:*` /
    /// `xmlns:*` attributes on foreign (SVG/MathML) elements; `None` for
    /// ordinary attributes and everything created via
    /// [`Document::set_attribute`] (which remains unnamespaced).
    pub namespace: Option<&'static str>,
}

impl Attribute {
    /// Plain unnamespaced attribute. The namespace stays `None`.
    pub fn plain(name: AttrName, value: String) -> Self {
        Attribute {
            name,
            value,
            namespace: None,
        }
    }

    /// The `prefix:` part of the qualified name, if any, derived from
    /// the interned qualified name (no stored allocation).
    pub fn prefix(&self) -> Option<&str> {
        split_attr_name(self.name.as_str()).0
    }

    /// The name after the `:`, or the whole name when unprefixed,
    /// derived from the interned qualified name (no stored allocation).
    pub fn local_name(&self) -> &str {
        split_attr_name(self.name.as_str()).1
    }
}

/// Split a qualified attribute name at the first `:` into
/// `(prefix, local_name)`, borrowed from the name.
fn split_attr_name(name: &str) -> (Option<&str>, &str) {
    match name.split_once(':') {
        Some((p, l)) => (Some(p), l),
        None => (None, name),
    }
}

/// An element's attribute list.
///
/// `None` (a null fat pointer — 16 bytes, no allocation) is the common
/// case — most elements carry no attributes — where a `Vec` paid 24 bytes
/// inline per element even when empty. A non-empty list is a single
/// exact-size boxed slice: one allocation instead of `Vec`'s possibly
/// over-allocated buffer, and no capacity bookkeeping on a list that is
/// written rarely and scanned often.
///
/// Derefs to `[Attribute]`, so every read path (`.iter()`, `.len()`,
/// indexing, `for a in &attrs`) works exactly as it did on the old
/// `Vec<Attribute>`; the only mutating operations are [`Attributes::push`]
/// and [`Attributes::retain`], which re-freeze the boxed slice (and drop
/// back to `None` when the last attribute is removed).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Attributes(Option<Box<[Attribute]>>);

impl Attributes {
    /// An empty attribute list: no allocation.
    pub fn new() -> Self {
        Self(None)
    }

    /// The attributes as a slice.
    pub fn as_slice(&self) -> &[Attribute] {
        match &self.0 {
            Some(boxed) => boxed,
            None => &[],
        }
    }

    /// Append an attribute. Cold path (attribute writes are rare next to
    /// reads): re-freezes the boxed slice.
    pub fn push(&mut self, attr: Attribute) {
        match &mut self.0 {
            Some(boxed) => {
                let mut v = std::mem::take(boxed).into_vec();
                v.push(attr);
                *boxed = v.into_boxed_slice();
            }
            None => self.0 = Some(Box::from([attr])),
        }
    }

    /// Retain only the attributes matching `predicate`, in order. Drops
    /// back to the empty (allocation-free) representation when nothing
    /// remains.
    pub fn retain(&mut self, mut predicate: impl FnMut(&Attribute) -> bool) {
        if let Some(boxed) = self.0.take() {
            let mut v = boxed.into_vec();
            v.retain(|a| predicate(a));
            if !v.is_empty() {
                self.0 = Some(v.into_boxed_slice());
            }
        }
    }
}

impl std::ops::Deref for Attributes {
    type Target = [Attribute];
    fn deref(&self) -> &[Attribute] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for Attributes {
    fn deref_mut(&mut self) -> &mut [Attribute] {
        match &mut self.0 {
            Some(boxed) => boxed,
            None => &mut [],
        }
    }
}

impl<'a> IntoIterator for &'a Attributes {
    type Item = &'a Attribute;
    type IntoIter = std::slice::Iter<'a, Attribute>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl FromIterator<Attribute> for Attributes {
    fn from_iter<I: IntoIterator<Item = Attribute>>(iter: I) -> Self {
        let v: Vec<Attribute> = iter.into_iter().collect();
        Self(if v.is_empty() {
            None
        } else {
            Some(v.into_boxed_slice())
        })
    }
}

impl From<Vec<Attribute>> for Attributes {
    fn from(v: Vec<Attribute>) -> Self {
        v.into_iter().collect()
    }
}

/// Build a DOM [`Attribute`] from a parser-produced `(name, value)` pair
/// on an element in namespace `ns`, applying the spec's "adjust foreign
/// attributes" step (WHATWG HTML 13.2.6.1): on foreign (SVG/MathML)
/// elements, the table's `xlink:*` / `xml:*` / `xmlns` / `xmlns:xlink`
/// attributes gain their namespace URI, prefix, and local name. HTML
/// elements keep plain attributes, as do foreign attributes outside the
/// table (e.g. `xmlns:foo`).
///
/// Atomized variant: the tokenizer interns names at tokenize time, so
/// well-known names arrive as pure atoms with zero heap use; this takes
/// the interned name directly instead of round-tripping through a
/// `String` and re-interning.
fn parser_attribute_atomized(ns: &Namespace, name: AttrName, value: String) -> Attribute {
    let (prefix, local_name) = split_attr_name(name.as_str());
    let namespace = if *ns != Namespace::Html {
        match (prefix, local_name) {
            (
                Some("xlink"),
                "actuate" | "arcrole" | "href" | "role" | "show" | "title" | "type",
            ) => Some(XLINK_NAMESPACE),
            (Some("xml"), "base" | "lang" | "space") => Some(XML_NAMESPACE),
            (None, "xmlns") => Some(XMLNS_NAMESPACE),
            // The table lists only `xmlns:xlink`, not `xmlns:*`.
            (Some("xmlns"), "xlink") => Some(XMLNS_NAMESPACE),
            _ => None,
        }
    } else {
        None
    };
    Attribute {
        name,
        value,
        namespace,
    }
}

/// One arena slot: the node's payload plus tree links.
#[derive(Clone, Debug)]
struct Node {
    kind: NodeKind,
    parent: Option<NodeId>,
    first_child: Option<NodeId>,
    last_child: Option<NodeId>,
    next_sibling: Option<NodeId>,
    prev_sibling: Option<NodeId>,
    /// True when the node is reachable from the document root.
    /// Maintained incrementally by [`Document::link_last`],
    /// [`Document::link_before`], and [`Document::unlink`], so
    /// [`Document::is_attached`] is an O(1) field read instead of an
    /// O(depth) parent-chain walk. `alloc` starts it `false` (detached);
    /// [`Document::new`] flips the root to `true`.
    attached: bool,
    /// True when the node's ancestor chain crosses an HTML
    /// `<template>`-to-contents edge. Maintained incrementally alongside
    /// `attached` (same writers), so
    /// [`Document::in_template_contents`] is O(1) with semantics identical
    /// to the old parent-chain walk: the flag is recomputed top-down over
    /// the linked/unlinked subtree from the parent's flag plus the
    /// template-contents edge check.
    in_template_contents: bool,
}

impl Node {
    fn new(kind: NodeKind) -> Self {
        Node {
            kind,
            parent: None,
            first_child: None,
            last_child: None,
            next_sibling: None,
            prev_sibling: None,
            attached: false,
            in_template_contents: false,
        }
    }
}

/// What DOM mutations dirtied since the last [`Document::take_dirty`].
///
/// Returned drained: each mutation is reported exactly once.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DirtyReport {
    /// Roots of the dirtied subtrees. The subtree rooted at each node
    /// needs its layout/paint output recomputed; ancestor boxes are
    /// unaffected except through normal layout of the changed subtree.
    pub subtrees: Vec<NodeId>,
    /// True when at least one recorded mutation could affect selector
    /// matching (a structural change or an attribute write — attribute
    /// selectors can match any attribute): the style stage must re-run,
    /// not just layout/paint.
    pub style: bool,
    /// Roots of subtrees whose selector matching may have changed *and*
    /// whose restyle scope is provably confined to that subtree:
    /// attribute writes on elements with no shadow-DOM involvement.
    /// An attribute write changes no node's attributes but the
    /// element's own, and changes no structure or element state, so —
    /// provided no stylesheet can observe the mutation from outside the
    /// subtree (no sibling combinators, no `:has()`; see
    /// `bowser_css::selectors_are_subtree_local`, which the embedder
    /// checks) — matching can only flip for the element and its
    /// descendants. The embedder may then service the style stage by
    /// restyling just this subtree during the partial layout pass
    /// instead of forcing a full restyle+relayout. Structural changes
    /// and text emptiness toggles are deliberately *not* recorded here:
    /// on their own they keep the old fail-closed behavior (full path),
    /// because the embedder's splice gate compares root *sets*, not dirt
    /// provenance — a structural mutation whose coalesced root coincides
    /// with a localized attribute write's root takes the partial path,
    /// which is sound because the subtree pass rebuilds from the current
    /// DOM.
    pub style_subtrees: Vec<NodeId>,
    /// True when a script-driven scroll offset changed (e.g. an
    /// `Element.scrollTop` setter): sticky constraints bake at layout
    /// time, so the embedder must run a full layout pass, not the
    /// partial subtree splice (which does not re-bake constraints at
    /// the new offsets).
    pub scroll_offset: bool,
}

impl DirtyReport {
    /// True when no mutation was recorded.
    pub fn is_empty(&self) -> bool {
        self.subtrees.is_empty() && !self.style && self.style_subtrees.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

/// An HTML document: the arena that owns every node.
#[derive(Clone, Debug)]
pub struct Document {
    nodes: Vec<Node>,
    /// This document's unique tag; stamped into every [`NodeId`] minted
    /// here and checked on every access (in all builds).
    tag: u32,
    /// `id` attribute value -> first element in document order carrying
    /// it, indexed by interned id atom ([`Document::id_atoms`]). Atoms
    /// are dense `u32`s from 0, so the index is a plain `Vec` (one
    /// bounds-checked index, no second hash probe — the
    /// [`Document::get_element_by_id`] fast path is a single FNV-1a hash of
    /// the query plus one index). Maintained on every attribute and
    /// structural mutation, so [`Document::get_element_by_id`] is O(1).
    /// Only elements in the document tree are indexed; detached subtrees
    /// are invisible, exactly as with the old linear scan. Atom keys
    /// (plus the [`FnvBuildHasher`]) replace the old `String`-keyed
    /// `RandomState` map: lookups hash the query once with the cheap
    /// FNV-1a hash instead of SipHash, and mutations never clone the id string.
    ///
    /// Invariant: `id_index.len() == next_id_atom`; every atom `<`
    /// `next_id_atom` is a valid index, and every indexed id has an
    /// entry in [`Document::id_atoms`].
    id_index: Vec<Option<NodeId>>,
    /// Interner for `id` attribute values: string -> atom. Append-only;
    /// atoms are never reused, so a `u32` stays valid for the document's
    /// lifetime. The table grows by one entry per *distinct* id, bounded
    /// in practice by the number of id'd elements.
    id_atoms: FnvHashMap<String, u32>,
    /// Next fresh id atom for [`Document::id_atoms`].
    next_id_atom: u32,
    /// Fail-closed stand-in for foreign or out-of-range [`NodeId`]s: a
    /// linkless empty comment payload. Internal accessors return this
    /// (with a `debug_assert!`) instead of panicking, so every public
    /// method is total over arbitrary `NodeId`s.
    invalid_node: Node,
    /// Roots of subtrees dirtied by DOM mutations since the last
    /// [`Document::take_dirty`]. The parser builds through the internal
    /// link primitives, so a freshly parsed document starts clean; only
    /// the public mutation methods record here.
    dirty_subtrees: HashSet<NodeId>,
    /// Whether a script-driven scroll offset changed since the last
    /// [`Document::take_dirty`] (see
    /// [`Document::mark_scroll_offset_changed`]). Drained by
    /// `take_dirty` into [`DirtyReport::scroll_offset`].
    scroll_offset_dirty: bool,
    /// Whether any recorded mutation could affect selector matching
    /// (structural changes and attribute writes — attribute selectors
    /// can match any attribute). The embedder's style stage must
    /// re-run when this is set; plain layout/paint dirt never sets it.
    dirty_style: bool,
    /// Attribute-write elements whose restyle scope is provably confined
    /// to their own subtree (see [`DirtyReport::style_subtrees`]). Drained
    /// by [`Document::take_dirty`]; a subset of [`Document::dirty_subtrees`]
    /// by construction (`mark_style_localized` runs exactly where
    /// `mark_dirty` records with `style = true` for attribute writes).
    dirty_style_subtrees: HashSet<NodeId>,
    /// Shadow-root fragment -> shadow host, for every live shadow root
    /// (parser-attached declarative ones and imperative
    /// [`Document::attach_shadow_root`] ones alike, including clones of
    /// `clonable` roots). The fragment is unlinked, so a plain
    /// parent-chain walk ends at it; this map closes the loop back to
    /// the host for [`Document::shadow_host_for`] (snap-target
    /// retargeting) and for dirtying the host when shadow content
    /// mutates. Nodes are arena-allocated and never freed, so the map
    /// only grows with the arena — bounded by the document's node count.
    shadow_hosts: HashMap<NodeId, NodeId>,
    /// Scratch walk stack for [`Document::set_subtree_flags`], cleared
    /// and reused on every call so the flag walk never allocates. A
    /// plain `Vec` keeps [`Document`] `Send + Sync`.
    flag_stack: Vec<NodeId>,
    /// Scratch work stack for [`Document::clone_subtree_into`], taken,
    /// cleared, and restored on every call (same pattern as
    /// [`Document::flag_stack`]) so the clone walk never allocates in
    /// steady state. A plain `Vec` keeps [`Document`] `Send + Sync`.
    clone_stack: Vec<(NodeId, NodeId)>,
    /// Interned short text payloads: whitespace runs between elements and
    /// other repeated short strings share one [`Arc<str>`] per distinct
    /// value instead of one heap allocation per text node (see
    /// [`Document::intern_text`]). The set itself holds the shared
    /// [`Arc`]s, so interning a miss costs exactly one allocation (the
    /// [`Arc`]); cleared past [`TEXT_CACHE_CAP`] entries, so a document
    /// with many distinct short texts cannot grow it without bound —
    /// live nodes keep their [`Arc`]s, only future sharing restarts. A
    /// plain `HashSet` keeps [`Document`] `Send + Sync`.
    text_cache: HashSet<Arc<str>>,
    /// The document's encoding (WHATWG "document encoding"), detected
    /// from the BOM / `<meta charset>` prescan at parse time, UTF-8 by
    /// default. Drives the URL Standard's encoding override for query
    /// percent-encoding (see [`crate::anchor`]).
    encoding: bowser_encoding::Encoding,
    /// The document's quirks mode (WHATWG HTML 13.2.6.4.1), computed
    /// from the doctype at parse time and stored here so layout and
    /// style can branch on it without re-deriving it from the doctype
    /// (w77 quirks plumbing). `Document::new` documents — built by
    /// script, not parsed — default to [`QuirksMode::NoQuirks`];
    /// [`Document::parse_html`] and
    /// [`Document::parse_html_with_executor`] stamp the mode they
    /// return, so the returned mode and this field always agree.
    quirks_mode: QuirksMode,
}

/// Longest text payload eligible for [`Document::intern_text`] sharing.
/// Whitespace runs between elements (the dominant repeated text) are a
/// few bytes; the cap bounds both the hash cost of a cache miss and the
/// size of every cache entry.
const MAX_INTERN_TEXT_LEN: usize = 64;
/// Bound on [`Document::text_cache`] entries; past this the cache is
/// cleared (see the field docs).
const TEXT_CACHE_CAP: usize = 1024;

/// The nodes an insertion child expands to (see
/// [`Document::expand_insert_child`]): either the child itself, carried
/// inline with no allocation, or a fragment's children, collected in
/// order. Both arms expose a `&[NodeId]` so `document_insert_valid` and
/// `insert_nodes` keep their slice APIs.
enum InsertNodes {
    /// An ordinary (non-fragment) child: inserted as-is.
    Single([NodeId; 1]),
    /// A [`NodeKind::DocumentFragment`]'s children, in order.
    Fragment(Vec<NodeId>),
}

impl InsertNodes {
    /// The expanded nodes as a slice.
    fn as_slice(&self) -> &[NodeId] {
        match self {
            InsertNodes::Single(n) => n,
            InsertNodes::Fragment(v) => v,
        }
    }
}

impl Document {
    /// Create an empty document whose root is a fresh [`NodeKind::Document`] node.
    ///
    /// # Panics
    ///
    /// Panics if the global document-tag counter is exhausted (2^32
    /// documents minted in one process); unreachable in practice, and
    /// failing loud here preserves the guarantee that `NodeId`s from
    /// different documents never compare equal.
    pub fn new() -> Self {
        // Saturating increment: when the counter sits at u32::MAX the
        // update is refused and the counter stays pinned there, so the
        // exhaustion panic below can never hand out a tag first and wrap
        // to 0 (`fetch_add` would wrap before the panic fires, letting a
        // catcher see colliding tags).
        let tag = NEXT_DOC_TAG
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                if t == u32::MAX { None } else { Some(t + 1) }
            })
            .unwrap_or(u32::MAX);
        if tag == u32::MAX {
            panic!("bowser_dom: document tag counter exhausted (2^32 documents)");
        }
        let mut doc = Document {
            nodes: Vec::new(),
            tag,
            id_index: Vec::new(),
            id_atoms: HashMap::default(),
            next_id_atom: 0,
            invalid_node: Node::new(NodeKind::Comment(String::new())),
            dirty_subtrees: HashSet::new(),
            scroll_offset_dirty: false,
            dirty_style: false,
            dirty_style_subtrees: HashSet::new(),
            shadow_hosts: HashMap::new(),
            flag_stack: Vec::new(),
            clone_stack: Vec::new(),
            text_cache: HashSet::new(),
            encoding: bowser_encoding::Encoding::Utf8,
            quirks_mode: QuirksMode::NoQuirks,
        };
        let root = doc.alloc(NodeKind::Document);
        // The root is trivially reachable from itself; everything the
        // parser or the link primitives attach under it inherits this.
        doc.nodes[root.index()].attached = true;
        doc
    }

    /// Parse HTML bytes into a document, returning the document and its
    /// quirks mode (computed from the doctype per WHATWG 13.2.6.4.1).
    ///
    /// Single pass: the parser drives a [`TreeSink`] that builds this
    /// arena directly, moving string payloads out of the parser with no
    /// intermediate owned tree. Namespaces, doctypes, comments, and
    /// processing instructions are preserved. `<template>` contents become
    /// a [`NodeKind::DocumentFragment`] child of the template element (see
    /// the module docs). Source locations are dropped.
    pub fn parse_html(html: &[u8]) -> (Document, QuirksMode) {
        let mut doc = Document::new();
        let quirks = {
            let mut sink = DomSink::new(&mut doc);
            bowser_html::parse_into(html, &mut sink)
        };
        doc.set_encoding(bowser_html::detect_encoding(html));
        doc.set_quirks_mode(quirks);
        (doc, quirks)
    }

    /// [`parse_html`] with a script engine attached: when `scripting` is
    /// true, each eligible parser-inserted classic `<script>` element is
    /// handed to `executor` as soon as its end tag is processed, before
    /// the next token is tokenized (see [`ScriptExecutor`]). The
    /// executor's [`document.write`](bowser_html::ScriptHost::document_write)
    /// calls and DOM mutations land in the tree under construction, so the
    /// returned document reflects them, exactly as in a browser.
    pub fn parse_html_with_executor(
        html: &[u8],
        scripting: bool,
        executor: &mut dyn ScriptExecutor,
    ) -> (Document, QuirksMode) {
        let mut doc = Document::new();
        let quirks = {
            let mut sink = DomSink::new(&mut doc);
            bowser_html::parse_into_with_scripting_and_executor(
                html, &mut sink, scripting, executor,
            )
        };
        doc.set_encoding(bowser_html::detect_encoding(html));
        doc.set_quirks_mode(quirks);
        (doc, quirks)
    }

    /// The document's encoding (WHATWG "document encoding"): the BOM /
    /// `<meta charset>` prescan result at parse time, UTF-8 for documents
    /// built by script. Drives the URL Standard's encoding override for
    /// query percent-encoding (see [`crate::anchor`]).
    pub fn encoding(&self) -> bowser_encoding::Encoding {
        self.encoding
    }

    /// Set the document's encoding (see [`Self::encoding`]). Called by
    /// [`Self::parse_html`] from the parse-time charset detection; the
    /// HTML Standard also lets `<meta charset>` mutations and
    /// `document.charset` setters change it, which a future pass can wire
    /// here.
    pub fn set_encoding(&mut self, encoding: bowser_encoding::Encoding) {
        self.encoding = encoding;
    }

    /// The document's quirks mode (see the [`Document::quirks_mode`]
    /// field): stamped by [`Self::parse_html`] /
    /// [`Self::parse_html_with_executor`] from the doctype, `NoQuirks`
    /// for script-built documents. Layout and style read this instead
    /// of re-deriving the mode, so the mode survives from parse to
    /// layout input through every [`Document`] carrier (loader,
    /// navigation, script executor) with no per-site threading.
    pub fn quirks_mode(&self) -> QuirksMode {
        self.quirks_mode
    }

    /// Set the document's quirks mode (see [`Self::quirks_mode`]).
    /// Parse entry points call this; embedders that build a document
    /// by other means (e.g. a re-parse that reuses a document) use it
    /// to keep the field in sync with the doctype they installed.
    pub fn set_quirks_mode(&mut self, quirks_mode: QuirksMode) {
        self.quirks_mode = quirks_mode;
    }

    /// The `Document` node itself (always arena index 0).
    pub fn root(&self) -> NodeId {
        NodeId {
            index: 0,
            doc: self.tag,
        }
    }

    /// The first element child of the document node (the `<html>` element
    /// in a parsed document), if any.
    pub fn document_element(&self) -> Option<NodeId> {
        let mut cursor = self.node(self.root()).first_child;
        while let Some(n) = cursor {
            if matches!(self.kind(n), NodeKind::Element(_)) {
                return Some(n);
            }
            cursor = self.node(n).next_sibling;
        }
        None
    }

    /// Create an HTML-namespace element. The tag name is ASCII-lowercased,
    /// matching `document.createElement` for HTML documents.
    pub fn create_element(&mut self, tag: &str) -> NodeId {
        self.alloc(NodeKind::Element(ElementData {
            tag_name: TagName::intern_lowered(tag),
            namespace: Namespace::Html,
            attributes: Attributes::new(),
            shadow_root: None,
        }))
    }

    /// Create an element in the given namespace; the tag name is kept as-is.
    pub fn create_element_ns(&mut self, ns: Namespace, tag: &str) -> NodeId {
        self.alloc(NodeKind::Element(ElementData {
            tag_name: TagName::from(tag),
            namespace: ns,
            attributes: Attributes::new(),
            shadow_root: None,
        }))
    }

    /// Create a text node. Short payloads (at most
    /// [`MAX_INTERN_TEXT_LEN`] bytes) are interned per document, so
    /// repeated runs — whitespace between elements, repeated labels —
    /// share one allocation; longer payloads get their own [`Arc`].
    pub fn create_text_node(&mut self, data: &str) -> NodeId {
        let shared = self.intern_text(data);
        self.alloc(NodeKind::Text(shared))
    }

    /// Share `text` as a text-node payload: strings of at most
    /// [`MAX_INTERN_TEXT_LEN`] bytes are interned per document (one
    /// [`Arc<str>`] per distinct value), longer strings get their own
    /// [`Arc`]. Used by [`Document::create_text_node`],
    /// [`Document::set_text_content`], and the parser sink, so every text
    /// node funnels through one sharing point.
    fn intern_text(&mut self, text: &str) -> Arc<str> {
        if text.len() <= MAX_INTERN_TEXT_LEN {
            if let Some(hit) = self.text_cache.get(text) {
                return hit.clone();
            }
            if self.text_cache.len() >= TEXT_CACHE_CAP {
                // Live nodes keep their Arcs; only future sharing
                // restarts, so this is always safe.
                self.text_cache.clear();
            }
            let shared: Arc<str> = Arc::from(text);
            self.text_cache.insert(shared.clone());
            shared
        } else {
            Arc::from(text)
        }
    }

    /// Create a comment node.
    pub fn create_comment(&mut self, data: &str) -> NodeId {
        self.alloc(NodeKind::Comment(data.to_string()))
    }

    /// Create a detached document fragment node.
    pub fn create_document_fragment(&mut self) -> NodeId {
        self.alloc(NodeKind::DocumentFragment)
    }

    // -----------------------------------------------------------------------
    // Dirty tracking
    // -----------------------------------------------------------------------

    /// Record a mutation at `node`: the subtree rooted here needs its
    /// layout/paint output recomputed. `style` marks the mutation as
    /// style-affecting (it could change selector matching), so the
    /// embedder's style stage must re-run too.
    ///
    /// Mutations on detached subtrees are not recorded: they become
    /// visible exactly when the subtree is linked into the document, and
    /// the linking mutation records the parent then.
    fn mark_dirty(&mut self, node: NodeId, style: bool) {
        if self.owns(node) && self.is_attached(node) {
            self.dirty_subtrees.insert(node);
            self.dirty_style |= style;
        }
    }

    /// A script-driven scroll offset changed on `node` (e.g. an
    /// `Element.scrollTop` setter): the sticky constraints bake at
    /// layout time, so the next [`take_dirty`] must report layout dirt.
    /// The node itself did not mutate (no style recalc needed), only
    /// its scroll offset did.
    pub fn mark_scroll_offset_changed(&mut self, node: NodeId) {
        if self.owns(node) && self.is_attached(node) {
            self.dirty_subtrees.insert(node);
            self.scroll_offset_dirty = true;
            // `dirty_style` stays false: offset changes do not affect
            // selector matching.
        }
    }

    /// Drain the mutation dirt recorded since the last call: the roots of
    /// dirtied subtrees (each needs layout/paint recomputed) and whether
    /// any of them was style-affecting (the style stage must re-run).
    ///
    /// The embedder (see `bowser_app::tab::Tab::sync_dirty_flags`) folds
    /// this into its render-stage flags after anything that can mutate
    /// the DOM — script runs, timer callbacks, event handlers — so each
    /// mutation is accounted exactly once.
    pub fn take_dirty(&mut self) -> DirtyReport {
        DirtyReport {
            subtrees: self.dirty_subtrees.drain().collect(),
            style: std::mem::take(&mut self.dirty_style),
            style_subtrees: self.dirty_style_subtrees.drain().collect(),
            scroll_offset: std::mem::take(&mut self.scroll_offset_dirty),
        }
    }

    /// Append `child` as the last child of `parent`.
    ///
    /// If `child` already has a parent it is first removed from there
    /// (like DOM `appendChild` moving a node). Appending a node to itself,
    /// to one of its own descendants, or into its own shadow tree (a
    /// host smuggled into its own shadow fragment would form a
    /// host->shadow->host cycle) is a no-op.
    ///
    /// A [`NodeKind::DocumentFragment`] child is spliced: its children are
    /// inserted in order and the fragment is left empty, per DOM
    /// `appendChild`. Inserting into the [`NodeKind::Document`] node
    /// itself applies the DOM pre-insert validity rules (no text nodes, at
    /// most one element and one doctype, doctype before element).
    ///
    /// Fail-closed (no panics): a [`NodeId`] from another document, an
    /// out-of-range index, a [`NodeKind::Document`] child, a live
    /// shadow-root fragment ([`Document::is_shadow_root`]: splicing it
    /// would empty the shadow tree while `shadow_hosts` still maps it),
    /// or a `parent`
    /// that cannot have children ([`NodeKind::Text`],
    /// [`NodeKind::Comment`], [`NodeKind::ProcessingInstruction`], or
    /// [`NodeKind::DocumentType`] nodes, the DOM `HierarchyRequestError`
    /// analogue) makes this a no-op, as does a document-validity
    /// violation.
    pub fn append_child(&mut self, parent: NodeId, child: NodeId) {
        if !self.can_have_children(parent) || !self.owns(child) || self.is_shadow_root(child) {
            return;
        }
        let Some(expanded) = self.expand_insert_child(child) else {
            return;
        };
        let nodes = expanded.as_slice();
        // Cycle guard per inserted node; an empty fragment is a no-op.
        if nodes.iter().any(|&n| self.is_ancestor_or_self(n, parent)) {
            return;
        }
        if !self.document_insert_valid(parent, nodes, None, None) {
            return;
        }
        self.insert_nodes(parent, nodes, None);
        // Structural: the new subtree needs selector matching, and the
        // parent's boxes change. A shadow-tree insertion dirties the
        // host too (its boxes render the shadow children).
        self.mark_dirty_and_shadow_host(parent, true);
    }

    /// Remove `child` from `parent`'s children. If `child` is not currently
    /// a child of `parent`, this is a no-op.
    ///
    /// Fail-closed: [`NodeId`]s from another document (or out-of-range
    /// indexes) make this a no-op instead of panicking.
    pub fn remove_child(&mut self, parent: NodeId, child: NodeId) {
        if !self.owns(parent) || !self.owns(child) {
            return;
        }
        if self.node(child).parent == Some(parent) {
            self.unlink(child);
            self.note_detach(child);
            self.mark_dirty_and_shadow_host(parent, true);
        }
    }

    /// Insert `child` into `parent`'s children before `before`, or append
    /// it when `before` is `None`.
    ///
    /// If `child` already has a parent it is first removed from there (a
    /// move, like DOM `insertBefore`). Inserting a node before itself
    /// leaves it in place (the DOM algorithm re-targets the reference to
    /// the next sibling after detaching). Inserting a node into itself or
    /// into one of its own descendants is a no-op.
    ///
    /// A [`NodeKind::DocumentFragment`] child is spliced: its children are
    /// inserted in order before `before` and the fragment is left empty.
    ///
    /// Fail-closed (no panics): [`NodeId`]s from another document,
    /// out-of-range indexes, a [`NodeKind::Document`] child, a live
    /// shadow-root fragment ([`Document::is_shadow_root`], same splicing
    /// hazard as [`Document::append_child`]), a `parent`
    /// that cannot have children (the DOM `HierarchyRequestError`
    /// analogue), a `before` that is not a child of `parent` (the DOM
    /// `NotFoundError` analogue), and document-validity violations all
    /// make this a no-op.
    pub fn insert_before(&mut self, parent: NodeId, child: NodeId, before: Option<NodeId>) {
        if !self.can_have_children(parent) || !self.owns(child) || self.is_shadow_root(child) {
            return;
        }
        // Validate the reference before touching anything.
        if let Some(r) = before
            && (!self.owns(r) || self.node(r).parent != Some(parent))
        {
            return;
        }
        let Some(expanded) = self.expand_insert_child(child) else {
            return;
        };
        let nodes = expanded.as_slice();
        // Cycle guard per inserted node, same choice as `append_child`:
        // no-op, not a throw.
        if nodes.iter().any(|&n| self.is_ancestor_or_self(n, parent)) {
            return;
        }
        if !self.document_insert_valid(parent, nodes, before, None) {
            return;
        }
        // DOM pre-insert: a reference equal to the inserted node itself
        // re-targets to its next sibling (computed before any mutation).
        let reference = match before {
            Some(r) if r == child => self.node(r).next_sibling,
            other => other,
        };
        self.insert_nodes(parent, nodes, reference);
        self.mark_dirty_and_shadow_host(parent, true);
    }

    /// Replace `old_child` with `new_child` within `parent`'s children, so
    /// that `new_child` takes `old_child`'s exact position and `old_child`
    /// is detached (but stays alive, like [`Document::remove`]).
    ///
    /// If `new_child` already has a parent it is moved. Replacing a node
    /// with itself is a no-op, as is replacing into a cycle (a node into
    /// itself or one of its descendants), matching the no-op choice of
    /// [`Document::append_child`] rather than the spec's throw.
    ///
    /// A [`NodeKind::DocumentFragment`] `new_child` is spliced: its
    /// children take `old_child`'s position in order and the fragment is
    /// left empty. An empty fragment still removes `old_child`.
    ///
    /// Fail-closed (no panics): [`NodeId`]s from another document,
    /// out-of-range indexes, a [`NodeKind::Document`] `new_child`, a live
    /// shadow-root fragment ([`Document::is_shadow_root`], same splicing
    /// hazard as [`Document::append_child`]), a
    /// `parent` that cannot have children (the DOM `HierarchyRequestError`
    /// analogue), an `old_child` that is not a child of `parent` (the DOM
    /// `NotFoundError` analogue), and document-validity violations (with
    /// `old_child` excluded from the counts, since it is removed) all make
    /// this a no-op.
    pub fn replace_child(&mut self, parent: NodeId, new_child: NodeId, old_child: NodeId) {
        if !self.can_have_children(parent)
            || !self.owns(new_child)
            || !self.owns(old_child)
            || self.is_shadow_root(new_child)
        {
            return;
        }
        if self.node(old_child).parent != Some(parent) {
            return;
        }
        if new_child == old_child {
            return;
        }
        let Some(expanded) = self.expand_insert_child(new_child) else {
            return;
        };
        let nodes = expanded.as_slice();
        if nodes.iter().any(|&n| self.is_ancestor_or_self(n, parent)) {
            return;
        }
        if !self.document_insert_valid(parent, nodes, Some(old_child), Some(old_child)) {
            return;
        }
        // The replacement takes `old_child`'s exact slot: capture its next
        // sibling as the reference, detach it, then insert. (`old_child`
        // cannot be one of the new nodes: it is parent's child, they are
        // the new child's.)
        //
        // Per the DOM replace algorithm, if the reference node is itself
        // being moved into `old_child`'s slot (replacing `old` with its
        // next sibling `new`), the insertion detaches it first, so
        // retarget past it to its own next sibling (captured before any
        // mutation); otherwise `link_before` would run on a detached
        // reference.
        let mut reference = self.node(old_child).next_sibling;
        if reference == Some(new_child) {
            reference = self.node(new_child).next_sibling;
        }
        self.unlink(old_child);
        self.note_detach(old_child);
        self.insert_nodes(parent, nodes, reference);
        self.mark_dirty_and_shadow_host(parent, true);
    }

    /// Detach `node` from its parent, if it has one.
    ///
    /// The node itself stays alive and addressable (matching DOM
    /// `ChildNode.remove()` semantics): [`Document::kind`] still works on
    /// it, its own children are untouched, and it can be re-inserted
    /// elsewhere. Detached nodes, the document root, and [`NodeId`]s from
    /// another document are no-ops.
    pub fn remove(&mut self, node: NodeId) {
        if !self.owns(node) {
            return;
        }
        let parent = self.node(node).parent;
        if parent.is_some() {
            self.unlink(node);
            self.note_detach(node);
        }
        if let Some(p) = parent {
            self.mark_dirty_and_shadow_host(p, true);
        }
    }

    /// Clone `node`. With `deep = true` the whole subtree is copied
    /// (payloads cloned, structure preserved); with `deep = false` only the
    /// node itself is copied and the clone has no children. A `clonable`
    /// shadow root is copied for the node itself and, with `deep = true`,
    /// for every descendant host too, per WHATWG DOM "clone a node" (the
    /// shadow-root copy step is not conditioned on `deep`); a non-clonable
    /// shadow root is dropped. The clone is always detached, even when
    /// `node` is attached, matching DOM `cloneNode`.
    ///
    /// Iterative, so adversarially deep subtrees cannot overflow the stack.
    ///
    /// Fail-closed: a [`NodeId`] from another document (or an out-of-range
    /// index) yields a fresh empty [`NodeKind::DocumentFragment`] instead
    /// of panicking.
    pub fn clone_node(&mut self, node: NodeId, deep: bool) -> NodeId {
        if !self.owns(node) {
            return self.alloc(NodeKind::DocumentFragment);
        }
        let mut kind = self.node(node).kind.clone();
        // The shadow-root copy step applies even to a shallow clone:
        // `cloneNode(false)` copies a `clonable` shadow tree.
        let shadow_fragment = self.maybe_clone_shadow_root(&mut kind);
        let clone = self.alloc(kind);
        if let Some(fragment) = shadow_fragment {
            self.shadow_hosts.insert(fragment, clone);
        }
        if !deep {
            return clone;
        }
        self.clone_subtree_into(node, clone);
        clone
    }

    /// Iteratively clone the children of `src` into `dst` (payloads cloned,
    /// structure preserved). Every cloned host's shadow root goes through
    /// [`Document::maybe_clone_shadow_root`], so hosts nested inside a
    /// shadow tree are copied along with it.
    ///
    /// Iterative; safe on adversarially deep trees. The work stack is the
    /// reused [`Document::clone_stack`] scratch, taken/cleared/restored on
    /// every call like [`Document::flag_stack`]; a nested call (a shadow
    /// copy inside a clone walk) finds the field taken and falls back to a
    /// fresh `Vec`, so the steady-state path never allocates.
    fn clone_subtree_into(&mut self, src: NodeId, dst: NodeId) {
        let mut stack = std::mem::take(&mut self.clone_stack);
        stack.push((src, dst));
        while let Some((s, d)) = stack.pop() {
            let mut cursor = self.node(s).first_child;
            while let Some(k) = cursor {
                let mut kind = self.node(k).kind.clone();
                let shadow_fragment = self.maybe_clone_shadow_root(&mut kind);
                let nk = self.alloc(kind);
                if let Some(fragment) = shadow_fragment {
                    self.shadow_hosts.insert(fragment, nk);
                }
                self.link_last(d, nk);
                stack.push((k, nk));
                cursor = self.node(k).next_sibling;
            }
        }
        stack.clear();
        self.clone_stack = stack;
    }

    /// If `kind` is an element hosting a `clonable` shadow root, copy the
    /// whole shadow tree into a fresh unlinked fragment and point the
    /// cloned element at it, carrying over `mode`, `delegates_focus`, and
    /// `clonable` (slot assignment is out of scope: there is no `<slot>`
    /// support). A non-clonable shadow root, like a missing one, is
    /// dropped; the clone never aliases the original's shadow fragment.
    ///
    /// Returns the fresh fragment when one was created, so the caller
    /// can register it in [`Document::shadow_hosts`] against the clone
    /// that owns it.
    ///
    /// The fragment stays unlinked and its children keep `attached =
    /// false`, so cloned shadow content is out of reach of light-DOM
    /// traversal and of the id index (attached elements only), exactly
    /// like the parse-time shadow fragment.
    fn maybe_clone_shadow_root(&mut self, kind: &mut NodeKind) -> Option<NodeId> {
        // Take the root first: the borrow of `kind` must end before the
        // allocating `clone_subtree_into` call below.
        let shadow = match kind {
            NodeKind::Element(data) => data.shadow_root.take()?,
            _ => return None,
        };
        if !shadow.clonable {
            return None;
        }
        let fragment = self.alloc(NodeKind::DocumentFragment);
        self.clone_subtree_into(shadow.root, fragment);
        if let NodeKind::Element(data) = kind {
            data.shadow_root = Some(ShadowRootData {
                mode: shadow.mode,
                delegates_focus: shadow.delegates_focus,
                clonable: shadow.clonable,
                root: fragment,
            });
        }
        Some(fragment)
    }

    /// Set the node's text content, matching DOM `textContent`.
    ///
    /// For [`NodeKind::Element`] and [`NodeKind::DocumentFragment`] nodes,
    /// all children are removed and replaced with a single new text node
    /// holding `text` (nothing is appended when `text` is empty). For
    /// [`NodeKind::Text`], [`NodeKind::Comment`], and
    /// [`NodeKind::ProcessingInstruction`] nodes, the character data is set
    /// in place. For [`NodeKind::Document`] and [`NodeKind::DocumentType`]
    /// nodes this is a deliberate no-op, matching the DOM `textContent`
    /// setter which does nothing on those node types. A [`NodeId`] from
    /// another document is likewise a no-op.
    pub fn set_text_content(&mut self, node: NodeId, text: &str) {
        if !self.owns(node) {
            return;
        }
        if matches!(
            self.kind(node),
            NodeKind::Element(_) | NodeKind::DocumentFragment
        ) {
            let mut cursor = self.node(node).first_child;
            while let Some(k) = cursor {
                cursor = self.node(k).next_sibling;
                self.unlink(k);
                self.note_detach(k);
            }
            if !text.is_empty() {
                let t = self.create_text_node(text);
                self.link_last(node, t);
            }
            // Structural (children went away; `:has()` matching on
            // ancestors can change), but the element's own matching does
            // not depend on the new text.
            self.mark_dirty_and_shadow_host(node, true);
            return;
        }
        // Text nodes share payloads via interning (see
        // `set_text_node_data`); everything else falls through to the
        // in-place match below.
        if matches!(self.kind(node), NodeKind::Text(_)) {
            self.set_text_node_data(node, text);
            return;
        }
        let changed = match &mut self.node_mut(node).kind {
            NodeKind::Text(_) => {
                // Shared payloads: check-then-intern needs the `&mut`
                // borrow released first, so text nodes are handled in the
                // branch below instead.
                unreachable!("text nodes handled above")
            }
            NodeKind::Comment(t) => {
                if *t == text {
                    false
                } else {
                    *t = text.to_string();
                    true
                }
            }
            NodeKind::ProcessingInstruction { data, .. } => {
                if *data == text {
                    false
                } else {
                    *data = text.to_string();
                    true
                }
            }
            // Document and DocumentType: the DOM textContent setter is a
            // no-op on these node types.
            NodeKind::Document | NodeKind::DocumentType(_) => false,
            NodeKind::Element(_) | NodeKind::DocumentFragment => unreachable!(),
        };
        if changed {
            // Character data only: no selector matches on text, so the
            // style stage stays clean — layout and paint re-run.
            self.mark_dirty_and_shadow_host(node, false);
        }
    }

    /// Set the character data of a [`NodeKind::Text`] node, sharing the
    /// new payload through [`Document::intern_text`]. Setting identical
    /// text is a no-op (no dirty marking), matching the old behavior.
    /// The style stage re-runs only when the write toggles the node's
    /// emptiness (`:empty` matches on exactly that transition).
    /// Split out of [`Document::set_text_content`]: interning needs `&mut
    /// self` while the payload borrow is live, so the check runs first on
    /// a shared borrow.
    fn set_text_node_data(&mut self, node: NodeId, text: &str) {
        let (same, toggles_empty) = match self.kind(node) {
            NodeKind::Text(t) => (t.as_ref() == text, t.is_empty() != text.is_empty()),
            _ => return,
        };
        if same {
            return;
        }
        let shared = self.intern_text(text);
        if let NodeKind::Text(t) = &mut self.node_mut(node).kind {
            *t = shared;
        }
        // Character data only — except that `:empty` (and through it
        // `:not(:empty)` / `:has()`) toggles on exactly the empty <->
        // non-empty transition, which can restyle boxes outside this
        // subtree. Only the toggle re-runs the style stage; a content
        // change that keeps emptiness is layout/paint only.
        self.mark_dirty_and_shadow_host(node, toggles_empty);
    }

    /// Replace the children of `el` with the parsed fragment `html` (the
    /// DOM `innerHTML` setter).
    ///
    /// The fragment is parsed with [`bowser_html::parse_fragment`] using
    /// the element itself as the context element (per the WHATWG fragment
    /// parsing algorithm): `<p>one<p>two` becomes two paragraphs,
    /// entities decode, `<tr>` children of a `<table>` stay in the table,
    /// and `<template>` contents land in a [`NodeKind::DocumentFragment`]
    /// child per the module convention. When `el` is itself a `<template>`
    /// element the parsed nodes land in its contents fragment instead
    /// (the fragment is created when absent; a pre-existing one is
    /// cleared and refilled, never detached), matching the WHATWG
    /// `innerHTML` setter. The previous children are detached (like
    /// [`Document::remove`]) and the id index is maintained across the
    /// swap.
    ///
    /// Fail-closed: a [`NodeId`] from another document, an out-of-range
    /// index, or an `el` that cannot have children (the DOM
    /// `HierarchyRequestError` analogue, same rule as
    /// [`Document::append_child`]) makes this a no-op.
    pub fn set_inner_html(&mut self, el: NodeId, html: &str) {
        if !self.can_have_children(el) {
            return;
        }
        // WHATWG `innerHTML` setter: when the context element is a
        // `<template>`, the fragment's nodes land in the template's
        // contents fragment rather than as direct children of the
        // template (this crate's template-contents convention; see the
        // module docs).
        let target = if self.is_html_template(el) {
            self.template_contents_fragment(el)
        } else {
            el
        };
        let mut cursor = self.node(target).first_child;
        while let Some(k) = cursor {
            cursor = self.node(k).next_sibling;
            self.unlink(k);
            self.note_detach(k);
        }
        // The fragment algorithm takes the element as its context (tag,
        // namespace, and attributes drive the initial insertion mode and
        // foreign-content dispatch: `<annotation-xml
        // encoding="text/html">` is only an HTML integration point when
        // its attributes are visible to the parser); for non-elements
        // there is no tag, and body is the sensible default.
        let (tag, ns, attrs) = match self.kind(el) {
            NodeKind::Element(data) => (
                data.tag_name.clone(),
                data.namespace.clone(),
                data.attributes
                    .iter()
                    .map(|a| (a.name.clone(), a.value.clone()))
                    .collect::<Vec<(AttrName, String)>>(),
            ),
            _ => (
                TagName::from("body"),
                Namespace::Html,
                Vec::<(AttrName, String)>::new(),
            ),
        };
        let context = HtmlElement {
            tag,
            ns,
            attrs,
            children: Vec::new(),
            template_contents: None,
            shadow_root: None,
            loc: None,
        };
        // Direct sink: the fragment parser drives `DomSink` rooted at
        // `target`, building the arena with no intermediate owned tree.
        // Id indexing is left to `note_attach_if_attached` below (the
        // sink's insert-if-absent would be wrong for ids colliding with
        // the rest of the document).
        self.parse_fragment_into(target, &context, html);
        self.note_attach_if_attached(target);
        // The subtree was rebuilt wholesale: matching and boxes change.
        self.mark_dirty_and_shadow_host(el, true);
    }

    /// Parse `html` as a fragment with `context` and build the resulting
    /// nodes under the (detached) `target` node, driving `DomSink`
    /// directly with no intermediate owned tree. Shared by
    /// [`Document::set_inner_html`] and [`Document::set_outer_html`].
    fn parse_fragment_into(&mut self, target: NodeId, context: &HtmlElement, html: &str) {
        let mut sink = DomSink::with_root(&mut *self, target);
        bowser_html::parse_fragment_into(html.as_bytes(), context, &mut sink);
        drop(sink);
    }

    /// The template-contents fragment of an HTML `<template>` element, if
    /// it has one. General descendant traversal skips the fragment child
    /// of a `<template>`; use this to reach the contents directly, then
    /// traverse from the returned fragment normally. Returns `None` for
    /// non-`template` elements and for foreign ids (fail-closed).
    pub fn template_contents(&self, template: NodeId) -> Option<NodeId> {
        if !self.is_html_template(template) {
            return None;
        }
        let mut cursor = self.try_node(template)?.first_child;
        while let Some(k) = cursor {
            if matches!(self.kind(k), NodeKind::DocumentFragment) {
                return Some(k);
            }
            cursor = self.node(k).next_sibling;
        }
        None
    }

    /// The declarative shadow root attached to `host` by the parser
    /// (`<template shadowrootmode>`; WHATWG HTML 13.2.6.4.7), if any.
    /// Traverse from the returned [`ShadowRootData::root`] fragment to see
    /// the shadow children; the fragment is not linked as a light-DOM
    /// child, so general traversal never crosses into it.
    pub fn shadow_root(&self, host: NodeId) -> Option<&ShadowRootData> {
        match self.kind(host) {
            NodeKind::Element(data) => data.shadow_root.as_ref(),
            _ => None,
        }
    }

    /// Imperatively attach a shadow root to `host` (the DOM
    /// `Element.attachShadow({ mode })` step). Allocates an unlinked
    /// [`NodeKind::DocumentFragment`] — shadow content stays out of
    /// reach of light-DOM traversal, exactly like the parser's
    /// declarative shadow root — records it on the host, and registers
    /// the fragment→host mapping (see [`Document::shadow_host_for`]).
    ///
    /// Returns `None` (the bindings surface this as a `TypeError`, the
    /// `InvalidStateError` analogue) when `host` is not an element or
    /// already hosts a shadow root. `delegates_focus` and `clonable`
    /// are `false`: no other init-dict members are supported in this
    /// wave.
    ///
    /// Deliberately not implemented here (documented follow-ups):
    /// slotting (`<slot>`), `:host` selectors, style scoping, and event
    /// retargeting.
    pub fn attach_shadow_root(&mut self, host: NodeId, mode: ShadowRootMode) -> Option<NodeId> {
        let already = self.element_data(host)?.shadow_root.is_some();
        if already {
            return None;
        }
        let fragment = self.alloc(NodeKind::DocumentFragment);
        // `element_data_mut` cannot fail: `element_data` just proved
        // `host` is an element, and nothing ran between.
        if let Some(data) = self.element_data_mut(host) {
            data.shadow_root = Some(ShadowRootData {
                mode,
                delegates_focus: false,
                clonable: false,
                root: fragment,
            });
        }
        self.shadow_hosts.insert(fragment, host);
        // The host's boxes now render the shadow children (see
        // `bowser_layout`'s child collection), so the host subtree is
        // style/layout dirty. (The helper also covers a host nested
        // inside another shadow tree.)
        self.mark_dirty_and_shadow_host(host, true);
        Some(fragment)
    }

    /// The shadow host when `node` is a shadow-root fragment or sits
    /// inside one, `None` otherwise. The fragment is unlinked, so a
    /// plain parent walk ends at it; the
    /// [`Document::shadow_hosts`](Self::shadow_hosts) map closes the
    /// loop back to the host. Used for snap-target retargeting (a
    /// shadow-descendant snap area reports its host). Fail-closed on
    /// foreign ids.
    pub fn shadow_host_for(&self, node: NodeId) -> Option<NodeId> {
        let mut cursor = Some(node);
        while let Some(n) = cursor {
            if let Some(&host) = self.shadow_hosts.get(&n) {
                return Some(host);
            }
            cursor = self.parent(n);
        }
        None
    }

    /// True when `node` is a live shadow-root fragment. The bindings
    /// reject inserting such a node as a child anywhere (the DOM
    /// `HierarchyRequestError` analogue), and so do
    /// [`Document::append_child`]/[`Document::insert_before`]/
    /// [`Document::replace_child`] (no-op): splicing the fragment would
    /// otherwise empty the shadow tree while `shadow_hosts` still maps
    /// it. Fail-closed on foreign ids.
    pub fn is_shadow_root(&self, node: NodeId) -> bool {
        self.shadow_hosts.contains_key(&node)
    }

    /// [`Document::mark_dirty`] plus the shadow-host rule: mutations
    /// inside a shadow tree (whose fragment is unlinked, so plain
    /// `mark_dirty` records nothing) dirty the host as well, whose
    /// boxes render the shadow children. The lookup walks up from
    /// `node` via [`Document::shadow_host_for`], so mutations on
    /// nested shadow descendants — not just the fragment itself —
    /// dirty the outermost host.
    fn mark_dirty_and_shadow_host(&mut self, node: NodeId, style: bool) {
        self.mark_dirty(node, style);
        if let Some(host) = self.shadow_host_for(node) {
            self.mark_dirty(host, style);
        }
    }

    /// Record an attribute write at `node` as style-localized: selector
    /// matching can only have changed inside the subtree rooted here,
    /// so the embedder may restyle just this subtree (see
    /// [`DirtyReport::style_subtrees`]).
    ///
    /// Fail-closed on shadow DOM: nodes inside a shadow tree, and shadow
    /// hosts themselves, are not recorded — shadow stylesheets are not
    /// part of the embedder's audited stylesheet set, so the subtree
    /// confinement argument does not apply to them. Such writes keep the
    /// old behavior (style dirt forces the full path).
    fn mark_style_localized(&mut self, node: NodeId) {
        if self.owns(node)
            && self.is_attached(node)
            && self.shadow_host_for(node).is_none()
            && self.shadow_root(node).is_none()
        {
            self.dirty_style_subtrees.insert(node);
        }
    }

    /// True when `el` is an HTML-namespace `<template>` element, the only
    /// context whose [`Document::set_inner_html`] target is the template
    /// contents fragment rather than the element itself.
    fn is_html_template(&self, el: NodeId) -> bool {
        match self.kind(el) {
            NodeKind::Element(data) => {
                data.namespace == Namespace::Html && data.tag_name == "template"
            }
            _ => false,
        }
    }

    /// True when `child` is the template-contents [`NodeKind::DocumentFragment`]
    /// of an HTML `<template>` element `parent`. General descendant
    /// traversal (children, descendants, text, selectors, id indexing)
    /// skips crossing this edge, like the spec's template contents being
    /// out of reach; traverse from the fragment itself to see its nodes.
    /// Serialization still walks the fragment inline.
    fn is_template_contents_edge(&self, parent: NodeId, child: NodeId) -> bool {
        matches!(self.kind(child), NodeKind::DocumentFragment) && self.is_html_template(parent)
    }

    /// True when `node` sits inside an HTML `<template>`'s contents
    /// fragment (the node itself, or an ancestor chain crossing a
    /// template-to-fragment edge). Id indexing ignores such nodes.
    /// O(1): the flag is maintained incrementally by `link_last` /
    /// `link_before` / `unlink`, recomputed top-down over the affected
    /// subtree with semantics identical to the old parent-chain walk.
    /// Fail-closed on foreign ids, like the old walk.
    fn in_template_contents(&self, node: NodeId) -> bool {
        self.node(node).in_template_contents
    }

    /// The template-contents fragment child of a `<template>` element,
    /// creating it when absent. Per the module's template-contents
    /// convention there is at most one; the first
    /// [`NodeKind::DocumentFragment`] child wins.
    fn template_contents_fragment(&mut self, el: NodeId) -> NodeId {
        debug_assert!(self.is_html_template(el));
        let mut cursor = self.node(el).first_child;
        while let Some(k) = cursor {
            if matches!(self.kind(k), NodeKind::DocumentFragment) {
                return k;
            }
            cursor = self.node(k).next_sibling;
        }
        let frag = self.alloc(NodeKind::DocumentFragment);
        self.link_last(el, frag);
        frag
    }

    /// The parent node, if any. `None` for a [`NodeId`] from another
    /// document (fail-closed, no panic).
    pub fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.try_node(node).and_then(|n| n.parent)
    }

    /// The child nodes, in order. Empty for a [`NodeId`] from another
    /// document (fail-closed, no panic).
    pub fn children(&self, node: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut cursor = self.try_node(node).and_then(|n| n.first_child);
        while let Some(c) = cursor {
            // Template contents are out of reach of general traversal:
            // an HTML `<template>`'s fragment child is skipped (traverse
            // from the fragment itself to see its nodes).
            if !self.is_template_contents_edge(node, c) {
                out.push(c);
            }
            cursor = self.node(c).next_sibling;
        }
        out
    }

    /// The first child node, if any, as a traversal cursor. `None` for a
    /// [`NodeId`] from another document (fail-closed, no panic).
    ///
    /// O(1), non-allocating: the counterpart of [`Document::next_sibling`].
    /// Like [`Document::children`], it skips the template-contents edge —
    /// an HTML `<template>`'s fragment child is out of reach of general
    /// traversal (traverse from the fragment itself to see its nodes) — so
    /// walking `first_child` + `next_sibling` visits exactly the nodes
    /// [`Document::children`] would return, without the per-call `Vec`.
    pub fn first_child(&self, node: NodeId) -> Option<NodeId> {
        let mut cursor = self.try_node(node).and_then(|n| n.first_child);
        while let Some(c) = cursor
            && self.is_template_contents_edge(node, c)
        {
            cursor = self.node(c).next_sibling;
        }
        cursor
    }

    /// The next sibling, if any. `None` for a [`NodeId`] from another
    /// document (fail-closed, no panic).
    ///
    /// O(1), non-allocating: a single arena index hop per step, so it is a
    /// valid traversal cursor together with [`Document::first_child`].
    /// Like [`Document::children`], it skips the template-contents edge —
    /// an HTML `<template>`'s fragment child is out of reach of general
    /// traversal — however the fragment is positioned among its siblings
    /// (the sink links it last, but the walk does not rely on that), so
    /// walking `first_child` + `next_sibling` visits exactly the nodes
    /// [`Document::children`] would return.
    pub fn next_sibling(&self, node: NodeId) -> Option<NodeId> {
        let parent = self.try_node(node).and_then(|n| n.parent);
        let mut cursor = self.try_node(node).and_then(|n| n.next_sibling);
        while let Some(c) = cursor
            && let Some(p) = parent
            && self.is_template_contents_edge(p, c)
        {
            cursor = self.node(c).next_sibling;
        }
        cursor
    }

    /// The previous sibling, if any. `None` for a [`NodeId`] from another
    /// document (fail-closed, no panic).
    pub fn previous_sibling(&self, node: NodeId) -> Option<NodeId> {
        self.try_node(node).and_then(|n| n.prev_sibling)
    }

    /// The node's kind (payload).
    ///
    /// Fail-closed: a [`NodeId`] from another document (or an out-of-range
    /// index) yields a linkless empty [`NodeKind::Comment`] instead of
    /// panicking, so structural queries on it stay neutral.
    pub fn kind(&self, node: NodeId) -> &NodeKind {
        &self.node(node).kind
    }

    /// Mutable access to the node's kind (payload).
    ///
    /// Test-only and crate-private: mutating `ElementData.attributes`
    /// directly would bypass the [`Document::get_element_by_id`] index, so
    /// outside code must go through [`Document::set_attribute`] /
    /// [`Document::remove_attribute`], which maintain it.
    #[cfg(test)]
    pub(crate) fn kind_mut(&mut self, node: NodeId) -> &mut NodeKind {
        &mut self.node_mut(node).kind
    }

    /// Get an attribute value. For HTML-namespace elements the lookup is
    /// ASCII case-insensitive (matching DOM `getAttribute` on HTML
    /// documents); for foreign elements it is exact.
    pub fn get_attribute(&self, el: NodeId, name: &str) -> Option<&str> {
        let data = self.element_data(el)?;
        find_attr(&data.attributes, &data.namespace, name).map(|a| a.value.as_str())
    }

    /// Get an attribute value by namespace URI and local name (the DOM
    /// `getAttributeNS` lookup). `namespace` is `None` for unnamespaced
    /// attributes; only parser-created foreign attributes (WHATWG HTML
    /// 13.2.6.1, "adjust foreign attributes") ever carry a namespace —
    /// everything from [`Document::set_attribute`] is unnamespaced.
    pub fn get_attribute_ns(
        &self,
        el: NodeId,
        namespace: Option<&str>,
        local_name: &str,
    ) -> Option<&str> {
        let data = self.element_data(el)?;
        data.attributes
            .iter()
            .find(|a| a.namespace == namespace && a.local_name() == local_name)
            .map(|a| a.value.as_str())
    }

    /// Set an attribute, replacing any existing one with the same name.
    /// For HTML-namespace elements the stored name is ASCII-lowercased
    /// (matching DOM `setAttribute` on HTML documents).
    ///
    /// Setting the `id` attribute updates the [`Document::get_element_by_id`]
    /// index (only when the element is in the document tree; detached
    /// elements are picked up when linked in). This is the only supported
    /// way to mutate attributes: the index is kept in sync on every call
    /// here (and on [`Document::remove_attribute`], structural ops, and
    /// parsing), so the `get_element_by_id` O(1) guarantee holds as long
    /// as attribute mutation goes through this API.
    pub fn set_attribute(&mut self, el: NodeId, name: &str, value: &str) {
        // Shared-borrow pre-pass: the namespace (for name matching) and
        // the old id's atom, resolved before the write below takes
        // `&mut self`. The atom lookup is clone-free; the old code cloned
        // the old id `String` here to escape the mutable borrow.
        let (ns, is_id, old_id_atom) = match self.element_data(el) {
            Some(d) => {
                let ns = d.namespace.clone();
                let is_id = is_id_name(&ns, name);
                let old_id_atom = if is_id {
                    find_attr(&d.attributes, &ns, "id").and_then(|a| self.id_atom(&a.value))
                } else {
                    None
                };
                (ns, is_id, old_id_atom)
            }
            None => return,
        };
        let changed: bool;
        {
            let data = match self.element_data_mut(el) {
                Some(d) => d,
                // Unreachable: the pre-pass above returned early for the
                // same `el`; kept fail-closed like the rest of this impl.
                None => return,
            };
            // Intern the stored name once: it doubles as the lookup key,
            // so the replace-or-push scan below is an integer compare on
            // the fast path (see `attr_name_matches`).
            let stored = if ns == Namespace::Html {
                // Already lowercase (or ASCII-lowercased here for HTML):
                // `intern_lowered` skips the intermediate `String` when
                // no uppercase is present.
                AttrName::intern_lowered(name)
            } else {
                // Foreign elements keep case-sensitive names as written.
                AttrName::from(name)
            };
            if let Some(a) = data
                .attributes
                .iter_mut()
                .find(|a| attr_name_matches(&ns, a, &stored, name))
            {
                if a.value == value {
                    changed = false;
                } else {
                    a.value = value.to_string();
                    changed = true;
                }
            } else {
                // `set_attribute` stays unnamespaced: even a qualified
                // name like `"xlink:href"` gets no namespace URI here.
                data.attributes
                    .push(Attribute::plain(stored, value.to_string()));
                changed = true;
            }
        }
        // The id index only covers attached, non-template elements, and
        // the `id_atoms` interner is append-only (atoms are never
        // reused), so the new value is interned *only* when the index
        // actually needs its atom: the id value changed and this element
        // is indexed. No-op rewrites, detached elements, and template
        // contents never touch the interner, so `set_attribute` on them
        // costs no permanent allocation. `changed` already encodes the
        // old-value/new-value string comparison (an unchanged id means
        // the index entry, if any, is already correct), and a missing old
        // atom means the old id was never indexed, so there is nothing
        // to drop.
        if is_id && changed && self.is_attached(el) && !self.in_template_contents(el) {
            if let Some(old) = old_id_atom {
                self.note_id_loss(old, el);
            }
            let new_id_atom = self.intern_id(value);
            self.note_id_gain(new_id_atom, el);
        }
        if changed {
            // Attribute selectors (`[data-x]`, `[data-x="v"]`, ...) can
            // match any attribute, so every attribute write can change
            // selector matching: the style stage re-runs. Boxes can
            // change too (and the element always repaints). The write's
            // restyle scope is additionally recorded as subtree-local
            // (see `mark_style_localized`); the embedder checks the
            // stylesheet precondition before using it.
            self.mark_dirty_and_shadow_host(el, true);
            self.mark_style_localized(el);
        }
    }

    /// Remove an attribute by name. No-op if the element has no such attribute.
    ///
    /// Removing the `id` attribute updates the
    /// [`Document::get_element_by_id`] index.
    pub fn remove_attribute(&mut self, el: NodeId, name: &str) {
        // Shared-borrow pre-pass (see `set_attribute`): the removed id's
        // atom, clone-free.
        let (ns, removed_id_atom) = match self.element_data(el) {
            Some(d) => {
                let ns = d.namespace.clone();
                let removed = if is_id_name(&ns, name) {
                    find_attr(&d.attributes, &ns, "id").and_then(|a| self.id_atom(&a.value))
                } else {
                    None
                };
                (ns, removed)
            }
            None => return,
        };
        let removed_any: bool;
        {
            let data = match self.element_data_mut(el) {
                Some(d) => d,
                // Unreachable: the pre-pass above returned early for the
                // same `el`; kept fail-closed like the rest of this impl.
                None => return,
            };
            let before = data.attributes.len();
            // Intern the query once so the scan is an integer compare on
            // the fast path (`attr_name_matches`).
            let query = if ns == Namespace::Html {
                AttrName::intern_lowered(name)
            } else {
                AttrName::from(name)
            };
            data.attributes
                .retain(|a| !attr_name_matches(&ns, a, &query, name));
            removed_any = data.attributes.len() != before;
        }
        if let Some(atom) = removed_id_atom
            && self.is_attached(el)
            && !self.in_template_contents(el)
        {
            self.note_id_loss(atom, el);
        }
        if removed_any {
            // As in `set_attribute`: any attribute can take part in
            // selector matching, so removal re-runs the style stage.
            // The removal's restyle scope is subtree-local like a write.
            self.mark_dirty_and_shadow_host(el, true);
            self.mark_style_localized(el);
        }
    }

    /// True when the element has an attribute with the given name. Name
    /// matching follows [`Document::get_attribute`] (ASCII
    /// case-insensitive for HTML-namespace elements, exact for foreign
    /// ones). Always false for non-elements and foreign ids
    /// (fail-closed).
    pub fn has_attribute(&self, el: NodeId, name: &str) -> bool {
        self.get_attribute(el, name).is_some()
    }

    /// First element in document order whose `id` attribute equals `id`.
    ///
    /// O(1): an index is maintained on every attribute and structural
    /// mutation (attribute set/remove, node insert/remove, and parsing),
    /// so no tree walk happens here. Only elements in the document tree
    /// are indexed; detached subtrees and HTML `<template>` contents are
    /// invisible.
    pub fn get_element_by_id(&self, id: &str) -> Option<NodeId> {
        let atom = self.id_atom(id)? as usize;
        // `atom < id_index.len()` by the field invariant; indexing (not
        // `get`) keeps this at one bounds check with no second hash.
        self.id_index[atom]
    }

    /// All elements with the given tag name, in document order, as a
    /// *live* [`HtmlCollection`]: `"*"` matches every element. Tag
    /// matching is ASCII case-insensitive for HTML-namespace elements,
    /// exact for foreign elements.
    ///
    /// The collection re-runs its query on every access (one pre-order
    /// walk per access), so it reflects later DOM mutations — the old
    /// snapshot-`Vec` behavior is gone.
    ///
    /// Elements inside an HTML `<template>`'s contents fragment are not
    /// matched (template contents are out of reach, as in the spec).
    pub fn get_elements_by_tag_name(&self, tag: &str) -> HtmlCollection<'_> {
        HtmlCollection {
            doc: self,
            query: CollectionQuery::TagName(TagNameQuery::new(tag)),
        }
    }

    /// A live view over the elements whose `class` attribute contains
    /// *every* whitespace-separated name in `class_names` (the DOM
    /// `getElementsByClassName`). An empty `class_names` matches nothing.
    /// Class matching is exact and case-sensitive, per the spec; the
    /// `class` *attribute name* lookup follows
    /// [`Document::get_attribute`] (ASCII case-insensitive for
    /// HTML-namespace elements).
    ///
    /// Like [`Document::get_elements_by_tag_name`] this is live: the
    /// query re-runs on every access.
    pub fn get_elements_by_class_name(&self, class_names: &str) -> HtmlCollection<'_> {
        HtmlCollection {
            doc: self,
            query: CollectionQuery::ClassNames(
                class_names
                    .split_ascii_whitespace()
                    .map(str::to_string)
                    .collect(),
            ),
        }
    }

    /// First element in document order matching `selectors`, or `None`.
    ///
    /// Supported selector syntax: comma-separated groups; each group is a
    /// chain of compound selectors joined by the descendant (space),
    /// child (`>`), adjacent-sibling (`+`), or general-sibling (`~`)
    /// combinator. A compound selector is an optional type selector (or
    /// `*`) followed by any number of `#id`, `.class`, and `[attr]` /
    /// `[attr<op>=value]` parts in any order, e.g.
    /// `div.main > p#intro.note[data-x^="v"] + span`.
    ///
    /// A group that fails to parse is a [`SelectorParseError`] (the DOM
    /// `SyntaxError` analogue): unlike earlier versions it never silently
    /// matches nothing, and one bad group fails the whole call.
    ///
    /// Walks in document order and stops at the first hit: each element is
    /// tested right-to-left against the groups (last compound, then
    /// ancestors/siblings), so unlike
    /// [`query_selector_all`](Self::query_selector_all) no candidate sets
    /// are materialized. Comma-group semantics match `query_selector_all`
    /// (first hit across all groups).
    pub fn query_selector(&self, selectors: &str) -> Result<Option<NodeId>, SelectorParseError> {
        let mut scratch = SelectorScratch::new();
        self.query_selector_with_scratch(selectors, &mut scratch)
    }

    /// [`Document::query_selector`], reusing the caller's
    /// [`SelectorScratch`]: repeat queries against the same scratch skip
    /// re-parsing the selector string.
    pub fn query_selector_with_scratch(
        &self,
        selectors: &str,
        scratch: &mut SelectorScratch,
    ) -> Result<Option<NodeId>, SelectorParseError> {
        let groups = scratch.parse_groups_cached(selectors)?;
        let mut found = None;
        self.walk_preorder(self.root(), |n| {
            if self.node_matches_any_group(n, &groups) {
                found = Some(n);
                false
            } else {
                true
            }
        });
        Ok(found)
    }

    /// True when the element `el` matches `selectors` (the DOM
    /// `Element.matches`). Selector syntax and syntax-error behavior are
    /// the same as [`Document::query_selector`].
    ///
    /// Fail-closed: a non-element node or a [`NodeId`] from another
    /// document yields `Ok(false)`, never a panic.
    pub fn matches(&self, el: NodeId, selectors: &str) -> Result<bool, SelectorParseError> {
        let mut scratch = SelectorScratch::new();
        self.matches_with_scratch(el, selectors, &mut scratch)
    }

    /// [`Document::matches`], reusing the caller's [`SelectorScratch`]:
    /// repeat queries against the same scratch skip re-parsing the
    /// selector string.
    pub fn matches_with_scratch(
        &self,
        el: NodeId,
        selectors: &str,
        scratch: &mut SelectorScratch,
    ) -> Result<bool, SelectorParseError> {
        let groups = scratch.parse_groups_cached(selectors)?;
        Ok(self.node_matches_any_group(el, &groups))
    }

    /// The nearest ancestor-or-self element of `el` (including `el`
    /// itself) matching `selectors`, or `None` (the DOM
    /// `Element.closest`). Selector syntax and syntax-error behavior are
    /// the same as [`Document::query_selector`].
    ///
    /// The walk never crosses from an HTML `<template>`'s contents
    /// fragment to its host element: like all other general traversal,
    /// selector matching treats template contents as out of reach.
    pub fn closest(
        &self,
        el: NodeId,
        selectors: &str,
    ) -> Result<Option<NodeId>, SelectorParseError> {
        let mut scratch = SelectorScratch::new();
        self.closest_with_scratch(el, selectors, &mut scratch)
    }

    /// [`Document::closest`], reusing the caller's [`SelectorScratch`]:
    /// repeat queries against the same scratch skip re-parsing the
    /// selector string.
    pub fn closest_with_scratch(
        &self,
        el: NodeId,
        selectors: &str,
        scratch: &mut SelectorScratch,
    ) -> Result<Option<NodeId>, SelectorParseError> {
        let groups = scratch.parse_groups_cached(selectors)?;
        let mut cursor = Some(el);
        while let Some(n) = cursor {
            if self.node_matches_any_group(n, &groups) {
                return Ok(Some(n));
            }
            cursor = self.selector_parent(n);
        }
        Ok(None)
    }

    /// Parse a comma-separated selector list into groups. Every group
    /// must parse; the first failure is returned (an empty group, e.g.
    /// from a trailing comma, fails too).
    fn parse_selector_groups(selectors: &str) -> Result<Vec<Selector>, SelectorParseError> {
        selectors
            .split(',')
            .map(|g| Selector::parse(g.trim()))
            .collect()
    }

    /// The parent of `node` for selector ancestor walks: `None` when the
    /// step would cross the template-contents boundary (from a contents
    /// fragment to its `<template>` host), keeping selector matching out
    /// of reach of template contents like all other general traversal.
    fn selector_parent(&self, node: NodeId) -> Option<NodeId> {
        let p = self.node(node).parent?;
        if self.is_template_contents_edge(p, node) {
            None
        } else {
            Some(p)
        }
    }

    /// True when `el` matches at least one selector group.
    fn node_matches_any_group(&self, el: NodeId, groups: &[Selector]) -> bool {
        groups.iter().any(|g| self.node_matches_group(el, g))
    }

    /// Right-to-left single-node match for [`query_selector`](Self::query_selector):
    /// the last compound must match `el`, then each combinator is satisfied
    /// by walking outward — the parent for `>`, any ancestor for a space,
    /// the immediately preceding element sibling for `+`, any preceding
    /// element sibling for `~`. Ancestor chains never cross into a
    /// `<template>`'s contents fragment (candidates come from
    /// [`walk_preorder`](Self::walk_preorder), which never enters template
    /// contents, so no edge check is needed for them; the check lives in
    /// [`Document::selector_parent`] for the [`Document::closest`] /
    /// [`Document::matches`] paths, whose start node may sit inside
    /// template contents).
    fn node_matches_group(&self, el: NodeId, sel: &Selector) -> bool {
        if !self.element_matches(el, sel.compounds.last().unwrap()) {
            return false;
        }
        let mut node = el;
        for (i, combinator) in sel.combinators.iter().enumerate().rev() {
            let compound = &sel.compounds[i];
            match combinator {
                Combinator::Child => {
                    let Some(parent) = self.selector_parent(node) else {
                        return false;
                    };
                    if !self.element_matches(parent, compound) {
                        return false;
                    }
                    node = parent;
                }
                Combinator::Descendant => {
                    let mut ancestor = self.selector_parent(node);
                    loop {
                        let Some(a) = ancestor else {
                            return false;
                        };
                        if self.element_matches(a, compound) {
                            node = a;
                            break;
                        }
                        ancestor = self.selector_parent(a);
                    }
                }
                Combinator::NextSibling => {
                    let Some(sib) = self.prev_element_sibling_raw(node) else {
                        return false;
                    };
                    if !self.element_matches(sib, compound) {
                        return false;
                    }
                    node = sib;
                }
                Combinator::SubsequentSibling => {
                    let mut cursor = self.node(node).prev_sibling;
                    let mut hit = None;
                    while let Some(s) = cursor {
                        if matches!(self.kind(s), NodeKind::Element(_))
                            && self.element_matches(s, compound)
                        {
                            hit = Some(s);
                            break;
                        }
                        cursor = self.node(s).prev_sibling;
                    }
                    let Some(sib) = hit else {
                        return false;
                    };
                    node = sib;
                }
            }
        }
        true
    }

    /// The immediately preceding *element* sibling of `node`, via raw
    /// sibling links (no template-edge skipping: sibling combinators
    /// relate actual siblings, and the template-contents fragment is not
    /// an element so it never interferes).
    fn prev_element_sibling_raw(&self, node: NodeId) -> Option<NodeId> {
        let mut cursor = self.node(node).prev_sibling;
        while let Some(s) = cursor {
            if matches!(self.kind(s), NodeKind::Element(_)) {
                return Some(s);
            }
            cursor = self.node(s).prev_sibling;
        }
        None
    }

    /// The immediately following *element* sibling of `node`, via raw
    /// sibling links (see [`Document::prev_element_sibling_raw`]).
    fn next_element_sibling_raw(&self, node: NodeId) -> Option<NodeId> {
        let mut cursor = self.node(node).next_sibling;
        while let Some(s) = cursor {
            if matches!(self.kind(s), NodeKind::Element(_)) {
                return Some(s);
            }
            cursor = self.node(s).next_sibling;
        }
        None
    }

    /// All elements in document order matching `selectors`.
    ///
    /// See [`Document::query_selector`] for the supported syntax; a group
    /// that fails to parse is a [`SelectorParseError`], failing the whole
    /// call.
    ///
    /// Matching runs left-to-right, one linear pass per combinator step,
    /// so cost is O(nodes x compounds) per selector group rather than
    /// O(nodes x depth) as with right-to-left ancestor walking.
    ///
    /// Elements inside an HTML `<template>`'s contents fragment are not
    /// matched (template contents are out of reach, as in the spec).
    pub fn query_selector_all(&self, selectors: &str) -> Result<Vec<NodeId>, SelectorParseError> {
        let mut scratch = SelectorScratch::new();
        self.query_selector_all_with_scratch(selectors, &mut scratch)
    }

    /// [`Document::query_selector_all`], reusing the caller's
    /// [`SelectorScratch`]: repeat queries reuse the per-step candidate
    /// buffers and skip re-parsing the selector string.
    pub fn query_selector_all_with_scratch(
        &self,
        selectors: &str,
        scratch: &mut SelectorScratch,
    ) -> Result<Vec<NodeId>, SelectorParseError> {
        let groups = scratch.parse_groups_cached(selectors)?;
        if groups.len() == 1 {
            // Single group: the match already comes out in document order.
            return Ok(self.match_selector_group(&groups[0], scratch));
        }
        // Union of the groups' matches, emitted in document order. (A
        // node matching two groups appears once, as with per-node
        // filtering.)
        let mut matched = vec![false; self.nodes.len()];
        for g in groups.iter() {
            for n in self.match_selector_group(g, scratch) {
                matched[n.index()] = true;
            }
        }
        let mut out = Vec::new();
        self.walk_preorder(self.root(), |n| {
            if matched[n.index()] {
                out.push(n);
            }
            true
        });
        Ok(out)
    }

    /// First descendant of `el` (never `el` itself) matching `selectors`,
    /// in document order, or `None` (the DOM `Element.querySelector`).
    /// Comma-group semantics match [`Document::query_selector_all`]:
    /// the earliest descendant in document order matching any group,
    /// not the first group's first match. A group that fails to parse
    /// is a [`SelectorParseError`], failing the whole call even when
    /// `el` has no descendants. A node id from another document yields
    /// `Ok(None)`: [`Document::descendants`] is fail-closed there.
    pub fn query_selector_in(
        &self,
        el: NodeId,
        selectors: &str,
    ) -> Result<Option<NodeId>, SelectorParseError> {
        Ok(self
            .query_selector_all_in(el, selectors)?
            .into_iter()
            .next())
    }

    /// Every descendant of `el` (never `el` itself) matching `selectors`,
    /// in document order, each node once (the DOM
    /// `Element.querySelectorAll`). Comma groups are unioned in document
    /// order, like [`Document::query_selector_all`]: the candidate set
    /// is the descendant walk itself, so no dedup scan is needed — each
    /// descendant is visited exactly once and tested against all groups.
    /// Combinator steps walk ancestors/siblings with the same
    /// [`Document::node_matches_any_group`] machinery as the
    /// document-wide query (the parse-tree scoped query behaves the
    /// same way), so a descendant combinator may be satisfied by an
    /// ancestor above `el`, exactly as in the document-wide match.
    pub fn query_selector_all_in(
        &self,
        el: NodeId,
        selectors: &str,
    ) -> Result<Vec<NodeId>, SelectorParseError> {
        let mut scratch = SelectorScratch::new();
        let groups = scratch.parse_groups_cached(selectors)?;
        Ok(self
            .descendants(el)
            .into_iter()
            .filter(|n| self.node_matches_any_group(*n, &groups))
            .collect())
    }

    /// Iterative pre-order walk of the subtree rooted at `node`
    /// (inclusive). The callback returns whether to keep walking; returning
    /// `false` stops the walk immediately (useful for first-hit searches).
    /// Uses the last-child/prev-sibling links so no per-node allocation
    /// occurs; safe on adversarially deep trees.
    fn walk_preorder(&self, node: NodeId, mut f: impl FnMut(NodeId) -> bool) {
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            if !f(n) {
                return;
            }
            let mut kid = self.node(n).last_child;
            while let Some(k) = kid {
                // Never cross from an HTML `<template>` into its contents
                // fragment: selectors and id repair share this walk.
                if !self.is_template_contents_edge(n, k) {
                    stack.push(k);
                }
                kid = self.node(k).prev_sibling;
            }
        }
    }

    /// Elements matching one selector group, in document order.
    ///
    /// Matches left-to-right: start with every element matching the first
    /// compound, then advance the candidate set once per combinator — to
    /// matching children for `>`, to matching proper descendants for a
    /// space, to the matching next element sibling for `+`, or to
    /// matching following element siblings for `~`. A descendant step
    /// walks only the candidate subtrees (not the whole document) and
    /// stamps visited nodes in `scratch`, so each node is visited at most
    /// once per step. Sibling steps stamp the candidate siblings first
    /// and then do one document-order walk, which keeps the output in
    /// document order and each node visited once per step.
    ///
    /// The candidate Vecs and the walk stack live in `scratch` and are
    /// reused across steps and calls — no per-step allocation.
    fn match_selector_group(&self, sel: &Selector, scratch: &mut SelectorScratch) -> Vec<NodeId> {
        scratch.current.clear();
        self.walk_preorder(self.root(), |n| {
            if self.element_matches(n, &sel.compounds[0]) {
                scratch.current.push(n);
            }
            true
        });
        for (i, combinator) in sel.combinators.iter().enumerate() {
            if scratch.current.is_empty() {
                break;
            }
            let compound = &sel.compounds[i + 1];
            match combinator {
                Combinator::Child => {
                    scratch.next.clear();
                    for &parent in scratch.current.iter() {
                        let mut kid = self.node(parent).first_child;
                        while let Some(k) = kid {
                            // Never cross into template contents.
                            if !self.is_template_contents_edge(parent, k)
                                && self.element_matches(k, compound)
                            {
                                scratch.next.push(k);
                            }
                            kid = self.node(k).next_sibling;
                        }
                    }
                }
                Combinator::Descendant => {
                    let generation = scratch.next_generation(self.nodes.len());
                    scratch.next.clear();
                    scratch.stack.clear();
                    // Sources are in document order and pairwise either
                    // disjoint or nested; the stamp makes nested sources
                    // cheap (their subtrees were already walked) and keeps
                    // `next` in document order.
                    for &src in scratch.current.iter() {
                        if scratch.mark[src.index()] == generation {
                            continue;
                        }
                        let mut kid = self.node(src).last_child;
                        while let Some(k) = kid {
                            // Never cross into template contents.
                            if !self.is_template_contents_edge(src, k) {
                                scratch.stack.push(k);
                            }
                            kid = self.node(k).prev_sibling;
                        }
                        while let Some(n) = scratch.stack.pop() {
                            if scratch.mark[n.index()] == generation {
                                // Visited before: its subtree is fully
                                // processed, so skip it without re-walking.
                                continue;
                            }
                            scratch.mark[n.index()] = generation;
                            if self.element_matches(n, compound) {
                                scratch.next.push(n);
                            }
                            let mut k2 = self.node(n).last_child;
                            while let Some(k) = k2 {
                                if !self.is_template_contents_edge(n, k) {
                                    scratch.stack.push(k);
                                }
                                k2 = self.node(k).prev_sibling;
                            }
                        }
                    }
                }
                Combinator::NextSibling | Combinator::SubsequentSibling => {
                    // Stamp the candidate siblings first: for `+` only the
                    // immediately following element sibling of each
                    // candidate, for `~` every following element sibling.
                    // One document-order walk then collects the stamped
                    // nodes matching the compound — each node visited
                    // once, output in document order, no duplicates.
                    let adjacent_only = matches!(combinator, Combinator::NextSibling);
                    let sib_gen = scratch.next_generation(self.nodes.len());
                    for &src in scratch.current.iter() {
                        let mut sib = self.next_element_sibling_raw(src);
                        while let Some(s) = sib {
                            scratch.mark[s.index()] = sib_gen;
                            if adjacent_only {
                                break;
                            }
                            sib = self.next_element_sibling_raw(s);
                        }
                    }
                    scratch.next.clear();
                    self.walk_preorder(self.root(), |n| {
                        if scratch.mark[n.index()] == sib_gen && self.element_matches(n, compound) {
                            scratch.next.push(n);
                        }
                        true
                    });
                }
            }
            std::mem::swap(&mut scratch.current, &mut scratch.next);
        }
        std::mem::take(&mut scratch.current)
    }

    /// Concatenation of all descendant text node data (including the node
    /// itself if it is a text node).
    ///
    /// Single pre-order walk appending text directly: no per-node
    /// allocation.
    ///
    /// Text inside an HTML `<template>`'s contents fragment is *not*
    /// included: template contents are out of reach of general traversal.
    /// Read the fragment's text directly from the fragment node.
    pub fn text_content(&self, node: NodeId) -> String {
        let mut out = String::new();
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            if let NodeKind::Text(t) = self.kind(n) {
                out.push_str(t);
            }
            let mut kid = self.node(n).last_child;
            while let Some(k) = kid {
                // Template contents are out of reach of text collection.
                if !self.is_template_contents_edge(n, k) {
                    stack.push(k);
                }
                kid = self.node(k).prev_sibling;
            }
        }
        out
    }

    /// All descendants in document order (iterative; safe on adversarially
    /// deep trees).
    ///
    /// Children are pushed via the last-child/prev-sibling links, so no
    /// per-node temporary list is allocated.
    ///
    /// Nodes inside an HTML `<template>`'s contents fragment are *not*
    /// included: general traversal does not cross from a `<template>`
    /// element into its fragment child. Traverse from the fragment itself
    /// to see its nodes.
    pub fn descendants(&self, node: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut stack = Vec::new();
        let mut cursor = self.node(node).last_child;
        while let Some(k) = cursor {
            // Skip the template-contents edge: general traversal does not
            // cross from an HTML `<template>` into its fragment child.
            if !self.is_template_contents_edge(node, k) {
                stack.push(k);
            }
            cursor = self.node(k).prev_sibling;
        }
        while let Some(n) = stack.pop() {
            out.push(n);
            let mut kid = self.node(n).last_child;
            while let Some(k) = kid {
                if !self.is_template_contents_edge(n, k) {
                    stack.push(k);
                }
                kid = self.node(k).prev_sibling;
            }
        }
        out
    }

    /// The ancestor chain, nearest first (parent, grandparent, ...).
    pub fn ancestors(&self, node: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut cursor = self.node(node).parent;
        while let Some(p) = cursor {
            out.push(p);
            cursor = self.node(p).parent;
        }
        out
    }

    /// The event propagation path for `node`: the node itself first, then
    /// each ancestor up to the document root (inclusive).
    ///
    /// The JS runtime uses this for dispatch: it walks the path in reverse
    /// for the capture phase (root down to the target's parent), invokes
    /// the target's listeners, then walks it forward for the bubble phase
    /// when the event bubbles. `bowser_dom` stores no listeners itself;
    /// see the [`event`] module for the full dispatch contract. A detached
    /// node yields a single-element path containing just the node.
    pub fn propagation_path(&self, node: NodeId) -> Vec<NodeId> {
        let mut path = vec![node];
        path.extend(self.ancestors(node));
        path
    }

    /// Serialize the whole document (equivalent to serializing the root).
    pub fn serialize(&self) -> String {
        self.serialize_node(self.root())
    }

    /// Serialize a node and its subtree to an HTML string. Iterative, so
    /// deeply nested subtrees cannot overflow the stack.
    pub fn serialize_node(&self, node: NodeId) -> String {
        let mut out = String::new();
        let mut stack = vec![StackItem::Node(node)];
        while let Some(item) = stack.pop() {
            match item {
                StackItem::Node(n) => self.serialize_open(n, &mut out, &mut stack),
                StackItem::EndTag(n) => {
                    // `n` was pushed as an element by `serialize_open`.
                    if let NodeKind::Element(data) = self.kind(n) {
                        out.push_str("</");
                        out.push_str(&data.tag_name);
                        out.push('>');
                    }
                }
            }
        }
        out
    }

    // ------------------------------------------------------------------
    // Web-parity DOM API: traversal, serialization, and matching
    // ------------------------------------------------------------------

    /// The last child node, if any, as a traversal cursor. `None` for a
    /// [`NodeId`] from another document (fail-closed, no panic).
    ///
    /// O(1), non-allocating: the counterpart of [`Document::first_child`].
    /// Like [`Document::first_child`], it skips the template-contents
    /// edge — an HTML `<template>`'s fragment child is out of reach of
    /// general traversal — so walking `last_child` + `previous_sibling`
    /// visits exactly the nodes [`Document::children`] would return, in
    /// reverse.
    pub fn last_child(&self, node: NodeId) -> Option<NodeId> {
        let mut cursor = self.try_node(node).and_then(|n| n.last_child);
        while let Some(c) = cursor
            && self.is_template_contents_edge(node, c)
        {
            cursor = self.node(c).prev_sibling;
        }
        cursor
    }

    /// The first child that is an element, if any (the DOM
    /// `firstElementChild`). Skips the template-contents edge like
    /// [`Document::first_child`]. `None` for foreign ids (fail-closed).
    pub fn first_element_child(&self, node: NodeId) -> Option<NodeId> {
        let mut cursor = self.first_child(node);
        while let Some(c) = cursor {
            if matches!(self.kind(c), NodeKind::Element(_)) {
                return Some(c);
            }
            cursor = self.next_sibling(c);
        }
        None
    }

    /// The next sibling that is an element, if any (the DOM
    /// `nextElementSibling`). `None` for foreign ids (fail-closed).
    pub fn next_element_sibling(&self, node: NodeId) -> Option<NodeId> {
        let mut cursor = self.next_sibling(node);
        while let Some(c) = cursor {
            if matches!(self.kind(c), NodeKind::Element(_)) {
                return Some(c);
            }
            cursor = self.next_sibling(c);
        }
        None
    }

    /// The previous sibling that is an element, if any (the DOM
    /// `previousElementSibling`). `None` for foreign ids (fail-closed).
    pub fn previous_element_sibling(&self, node: NodeId) -> Option<NodeId> {
        let mut cursor = self.previous_sibling(node);
        while let Some(c) = cursor {
            if matches!(self.kind(c), NodeKind::Element(_)) {
                return Some(c);
            }
            cursor = self.previous_sibling(c);
        }
        None
    }

    /// The number of child nodes that are elements (the DOM
    /// `childElementCount`).
    ///
    /// Allocation-free: walks the first-child/next-sibling cursors
    /// directly instead of materializing [`Document::children`]. 0 for
    /// foreign ids (fail-closed).
    pub fn child_element_count(&self, node: NodeId) -> usize {
        let mut count = 0;
        let mut cursor = self.first_child(node);
        while let Some(c) = cursor {
            if matches!(self.kind(c), NodeKind::Element(_)) {
                count += 1;
            }
            cursor = self.next_sibling(c);
        }
        count
    }

    /// Serialize the children of `node` to an HTML string (the DOM
    /// `innerHTML` getter): the fragment serialization of the node's
    /// children. For an HTML `<template>` element this is the template
    /// contents (per the module's template-contents convention), matching
    /// the WHATWG `innerHTML` getter. Fail-closed: a [`NodeId`] from
    /// another document yields an empty string.
    pub fn inner_html(&self, node: NodeId) -> String {
        if !self.owns(node) {
            return String::new();
        }
        let target = if self.is_html_template(node) {
            match self.template_contents(node) {
                Some(f) => f,
                None => return String::new(),
            }
        } else {
            node
        };
        let mut out = String::new();
        // WHATWG HTML 13.3 step 4: a shadow host's `<template
        // shadowrootmode>` serializes first, before its light-DOM
        // children.
        if let Some(shadow) = self.shadow_root(target) {
            self.serialize_shadow_template(shadow, &mut out);
        }
        for c in self.children(target) {
            out.push_str(&self.serialize_node(c));
        }
        out
    }

    /// Serialize `node` itself to an HTML string (the DOM `outerHTML`
    /// getter). Equivalent to [`Document::serialize_node`].
    pub fn outer_html(&self, node: NodeId) -> String {
        self.serialize_node(node)
    }

    /// Replace `node` with the parsed fragment `html` (the DOM `outerHTML`
    /// setter).
    ///
    /// The fragment is parsed with the node's *parent* as the context
    /// element (per the WHATWG fragment parsing algorithm, like
    /// [`Document::set_inner_html`]); the parsed nodes take `node`'s
    /// exact position in order, and `node` itself is detached but stays
    /// alive (like [`Document::remove`]). An empty `html` just removes
    /// the node. The id index is maintained across the swap, and the
    /// parent is dirtied (style-affecting, like any structural change).
    ///
    /// Fail-closed: a [`NodeId`] from another document, an out-of-range
    /// index, a node with no parent, or a parent that is the
    /// [`NodeKind::Document`] node itself (the DOM `HierarchyRequestError`
    /// analogue) makes this a no-op.
    pub fn set_outer_html(&mut self, node: NodeId, html: &str) {
        if !self.owns(node) {
            return;
        }
        let Some(parent) = self.node(node).parent else {
            return;
        };
        if matches!(self.kind(parent), NodeKind::Document) {
            return;
        }
        if !self.can_have_children(parent) {
            return;
        }
        // Context element for the fragment algorithm: the parent (its
        // tag, namespace, and attributes drive the initial insertion
        // mode), defaulting to body for non-element parents (a
        // DocumentFragment), mirroring `set_inner_html`.
        let (tag, ns, attrs) = match self.kind(parent) {
            NodeKind::Element(data) => (
                data.tag_name.clone(),
                data.namespace.clone(),
                data.attributes
                    .iter()
                    .map(|a| (a.name.clone(), a.value.clone()))
                    .collect::<Vec<(AttrName, String)>>(),
            ),
            _ => (
                TagName::from("body"),
                Namespace::Html,
                Vec::<(AttrName, String)>::new(),
            ),
        };
        let context = HtmlElement {
            tag,
            ns,
            attrs,
            children: Vec::new(),
            template_contents: None,
            shadow_root: None,
            loc: None,
        };
        let frag = self.create_document_fragment();
        self.parse_fragment_into(frag, &context, html);
        // `replace_child` splices the fragment's children into `node`'s
        // slot in order and detaches `node`; the id index, template-edge
        // handling, and dirty marking all follow from it.
        self.replace_child(parent, frag, node);
    }

    // ------------------------------------------------------------------
    // internal helpers
    // ------------------------------------------------------------------

    fn alloc(&mut self, kind: NodeKind) -> NodeId {
        // The arena index must fit in a u32; 4 billion nodes is
        // unreachable in practice, and failing loud beats silently
        // aliasing two nodes to the same NodeId. This is a hard check in
        // all builds, not a debug_assert: without it a release build
        // would wrap the index back to 0.
        let index = self.nodes.len();
        assert!(
            index <= u32::MAX as usize,
            "bowser_dom: arena exceeded 2^32 nodes"
        );
        let id = NodeId {
            index: index as u32,
            doc: self.tag,
        };
        self.nodes.push(Node::new(kind));
        id
    }

    /// Whether `id` addresses a live node in this document: the document
    /// tag matches and the index is in range. False for [`NodeId`]s minted
    /// by another [`Document`], and for out-of-range indexes (not
    /// constructible through the public API, but not assumed impossible).
    /// A single `u32` compare plus a bounds check: ~free, checked in all
    /// builds.
    #[inline]
    fn owns(&self, id: NodeId) -> bool {
        id.doc == self.tag && (id.index as usize) < self.nodes.len()
    }

    /// Total node lookup: `None` for foreign or out-of-range ids instead
    /// of panicking.
    #[inline]
    fn try_node(&self, id: NodeId) -> Option<&Node> {
        self.owns(id).then(|| &self.nodes[id.index as usize])
    }

    /// Internal fast path for ids the caller already validated (or minted
    /// in this arena): falls back to the linkless [`Document::invalid_node`]
    /// stand-in on a foreign or out-of-range id instead of panicking.
    /// Public methods pre-check with [`Document::owns`] and define their
    /// own fail-closed behavior; this is the backstop that keeps every
    /// accessor total. (Deliberately silent in all builds: cross-document
    /// use is defined fail-closed behavior, not misuse. Internal helpers
    /// with stricter contracts use `debug_assert!(self.owns(..))` at their
    /// own boundaries.)
    #[inline]
    fn node(&self, id: NodeId) -> &Node {
        match self.try_node(id) {
            Some(n) => n,
            None => &self.invalid_node,
        }
    }

    /// Mutable version of [`Document::node`]: same fail-closed contract.
    /// Note the stand-in is shared, so a `&mut` to it must never escape a
    /// no-op path; public mutators pre-check with [`Document::owns`].
    #[inline]
    fn node_mut(&mut self, id: NodeId) -> &mut Node {
        if !self.owns(id) {
            return &mut self.invalid_node;
        }
        &mut self.nodes[id.index as usize]
    }

    fn element_data(&self, el: NodeId) -> Option<&ElementData> {
        match &self.node(el).kind {
            NodeKind::Element(d) => Some(d),
            _ => None,
        }
    }

    fn element_data_mut(&mut self, el: NodeId) -> Option<&mut ElementData> {
        match &mut self.node_mut(el).kind {
            NodeKind::Element(d) => Some(d),
            _ => None,
        }
    }

    // ------------------------------------------------------------------
    // id index maintenance
    //
    // Invariant: `id_index` maps each id carried by an *attached* element
    // to the first such element in document order. Attribute changes only
    // touch the index for attached elements; structural changes call
    // `note_attach_if_attached` / `note_detach` around the link/unlink.
    // Ids are keyed by atom (`u32`) from the `id_atoms` interner, never by
    // `String`: lookups hash once with FNV-1a instead of SipHash, and
    // mutations move atoms instead of cloning strings.
    // ------------------------------------------------------------------

    /// Intern `id`, returning its atom. Inserts on first sight (one clone
    /// per *distinct* id, amortized); repeat calls are a single FNV-1a
    /// probe with no allocation. Also extends [`Document::id_index`] so
    /// the `id_index.len() == next_id_atom` invariant holds.
    fn intern_id(&mut self, id: &str) -> u32 {
        if let Some(&atom) = self.id_atoms.get(id) {
            return atom;
        }
        let atom = self.next_id_atom;
        // Unreachable in practice (one atom pins one heap `String`; the
        // process OOMs long before 2^32 distinct ids), and failing loud
        // beats silently aliasing two ids' atoms.
        self.next_id_atom = self
            .next_id_atom
            .checked_add(1)
            .expect("bowser_dom: id atom space exhausted");
        self.id_atoms.insert(id.to_string(), atom);
        self.id_index.push(None);
        atom
    }

    /// The atom for `id`, if it was interned before. Lookup-only: never
    /// grows the table. `None` means `id` cannot be in `id_index` (every
    /// indexed id is interned — see the `id_index` field docs).
    fn id_atom(&self, id: &str) -> Option<u32> {
        self.id_atoms.get(id).copied()
    }

    /// True when `node` is reachable from the document root. O(1): the
    /// flag is maintained incrementally by `link_last` / `link_before` /
    /// `unlink`. Fail-closed on foreign ids (the linkless stand-in node
    /// is never attached), matching the old parent-chain walk.
    fn is_attached(&self, node: NodeId) -> bool {
        self.node(node).attached
    }

    /// Every `(id, element)` pair in the subtree rooted at `node`, in
    /// document order. Iterative; safe on adversarially deep trees.
    fn subtree_id_elements(&self, node: NodeId, out: &mut Vec<(String, NodeId)>) {
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            if let Some(id) = self.get_attribute(n, "id") {
                out.push((id.to_string(), n));
            }
            let mut kid = self.node(n).last_child;
            while let Some(k) = kid {
                // Template contents are out of reach of id indexing.
                if !self.is_template_contents_edge(n, k) {
                    stack.push(k);
                }
                kid = self.node(k).prev_sibling;
            }
        }
    }

    /// First element in document order whose `id` interns to `atom`.
    /// Single pre-order walk stopping at the first hit; used to repair
    /// the index after removals. Atom equality is string equality
    /// (interning is a pure function), so this matches exactly the
    /// elements the old string-keyed walk matched.
    fn first_element_with_id_atom(&self, atom: u32) -> Option<NodeId> {
        let mut found = None;
        self.walk_preorder(self.root(), |n| {
            let hit = self.get_attribute(n, "id").and_then(|id| self.id_atom(id)) == Some(atom);
            if hit {
                found = Some(n);
                false
            } else {
                true
            }
        });
        found
    }

    /// Point each id in `atoms` at its first in-document-order element.
    /// One pre-order walk total (stopping once every id has a hit),
    /// replacing one walk per duplicate id. Used to resolve duplicate ids
    /// after attach/detach.
    fn resolve_duplicate_ids(&mut self, atoms: &[u32]) {
        let mut pending: FnvHashSet<u32> = atoms.iter().copied().collect();
        let mut found: Vec<(u32, NodeId)> = Vec::new();
        self.walk_preorder(self.root(), |n| {
            if pending.is_empty() {
                return false;
            }
            if let Some(id) = self.get_attribute(n, "id")
                && let Some(atom) = self.id_atom(id)
                && pending.remove(&atom)
            {
                found.push((atom, n));
            }
            true
        });
        for (atom, el) in found {
            self.id_index[atom as usize] = Some(el);
        }
    }

    /// The attached element `el` gained the id with `atom`: index it,
    /// resolving duplicates in document order.
    fn note_id_gain(&mut self, atom: u32, el: NodeId) {
        match self.id_index[atom as usize] {
            None => {
                self.id_index[atom as usize] = Some(el);
            }
            Some(cur) if cur != el => self.resolve_duplicate_ids(&[atom]),
            Some(_) => {}
        }
    }

    /// The element `el` stopped carrying the id with `atom` (attribute
    /// removed, renamed, or the element detached): drop the index entry
    /// when it pointed at `el`, repairing it with the next
    /// in-document-order element carrying the id.
    fn note_id_loss(&mut self, atom: u32, el: NodeId) {
        if self.id_index[atom as usize] == Some(el) {
            self.id_index[atom as usize] = None;
            if let Some(next) = self.first_element_with_id_atom(atom) {
                self.id_index[atom as usize] = Some(next);
            }
        }
    }

    /// The subtree `root` was just attached to the document tree: index
    /// every id'd element in it. The subtree walk is in document order, so
    /// the first occurrence of each id in the subtree wins directly; ids
    /// that collide with elements already indexed elsewhere in the
    /// document are re-resolved with a single document walk, no matter how
    /// many duplicates there are.
    fn note_attach(&mut self, root: NodeId) {
        let mut pairs = Vec::new();
        self.subtree_id_elements(root, &mut pairs);
        // `pairs` is in subtree document order.
        let mut first_in_subtree: FnvHashMap<u32, NodeId> = HashMap::default();
        let mut seen: FnvHashSet<u32> = HashSet::default();
        let mut duplicates: Vec<u32> = Vec::new();
        for (id, el) in &pairs {
            // Interning here (one clone per *distinct* id, amortized) is
            // what makes every indexed id's atom valid.
            let atom = self.intern_id(id);
            match self.id_index[atom as usize] {
                None => {
                    first_in_subtree.entry(atom).or_insert(*el);
                }
                Some(cur) if cur == *el => {}
                Some(_) => {
                    if seen.insert(atom) {
                        duplicates.push(atom);
                    }
                }
            }
        }
        for (atom, el) in first_in_subtree {
            self.id_index[atom as usize] = Some(el);
        }
        self.resolve_duplicate_ids(&duplicates);
    }

    /// `note_attach`, but only when `root` actually ended up in the
    /// document tree. Structural ops on detached subtrees must not
    /// pollute the index; their ids are picked up when the subtree is
    /// linked into the document.
    fn note_attach_if_attached(&mut self, root: NodeId) {
        // Template contents are out of reach of id indexing even though
        // they are structurally attached (e.g. `set_inner_html` on a
        // `<template>` targets its contents fragment).
        if self.is_attached(root) && !self.in_template_contents(root) {
            self.note_attach(root);
        }
    }

    /// The subtree `root` was just detached from the document tree: drop
    /// index entries pointing into it, repairing each dropped id with the
    /// next in-document-order element carrying it (or removing the id when
    /// none remains). All repairs share a single document walk, no matter
    /// how many ids were dropped. Must run *after* the unlink, so the
    /// repair walk doesn't see the detached nodes.
    fn note_detach(&mut self, root: NodeId) {
        let mut pairs = Vec::new();
        self.subtree_id_elements(root, &mut pairs);
        let mut seen: FnvHashSet<u32> = HashSet::default();
        let mut repair: Vec<u32> = Vec::new();
        for (id, el) in &pairs {
            // Lookup-only: every indexed id is interned (see the
            // `id_index` field docs), so a missing atom means "not
            // indexed" either way.
            if let Some(atom) = self.id_atom(id)
                && self.id_index[atom as usize] == Some(*el)
            {
                self.id_index[atom as usize] = None;
                if seen.insert(atom) {
                    repair.push(atom);
                }
            }
        }
        self.resolve_duplicate_ids(&repair);
    }

    /// Set `attached` on every node of the subtree rooted at `root` and
    /// recompute `in_template_contents` top-down from
    /// `root_in_template_contents`. [`Document::link_last`] /
    /// [`Document::link_before`] derive the root's values from the parent
    /// (its flags, plus the template-to-fragment edge when the link
    /// crosses it); [`Document::unlink`] passes `false` / `false`, matching
    /// the old parent-chain walks, which stopped at the severed parent
    /// link. The top-down recompute is exactly what the old walks
    /// computed, since both flags are pure functions of the ancestor
    /// chain above the subtree root.
    ///
    /// Iterative; safe on adversarially deep trees. O(subtree size).
    /// Never allocates in steady state: a childless root (the common
    /// fresh-node link) returns after touching only the root, and the
    /// general case reuses the [`Document::flag_stack`] scratch stack.
    fn set_subtree_flags(&mut self, root: NodeId, attached: bool, root_in_template_contents: bool) {
        let r = root.index();
        self.nodes[r].attached = attached;
        self.nodes[r].in_template_contents = root_in_template_contents;
        if self.nodes[r].first_child.is_none() {
            // Common fresh-node case (the parser linking a just-created
            // node): nothing below to visit, so no walk and no stack.
            return;
        }
        let mut stack = std::mem::take(&mut self.flag_stack);
        stack.push(root);
        while let Some(n) = stack.pop() {
            let ni = n.index();
            let parent_tc = self.nodes[ni].in_template_contents;
            let mut kid = self.nodes[ni].first_child;
            while let Some(k) = kid {
                let ki = k.index();
                let edge = self.is_template_contents_edge(n, k);
                let kn = &mut self.nodes[ki];
                kn.attached = attached;
                kn.in_template_contents = parent_tc || edge;
                kid = kn.next_sibling;
                stack.push(k);
            }
        }
        stack.clear();
        self.flag_stack = stack;
    }

    /// Link `child` (known detached) as the last child of `parent`.
    ///
    /// Indexes the arena directly instead of going through `node_mut`
    /// per field: this is the hottest path in tree construction.
    fn link_last(&mut self, parent: NodeId, child: NodeId) {
        debug_assert!(self.node(child).parent.is_none());
        debug_assert!(self.owns(parent));
        debug_assert!(self.owns(child));
        let p = parent.index();
        let c = child.index();
        let prev_last = self.nodes[p].last_child;
        let cn = &mut self.nodes[c];
        cn.parent = Some(parent);
        cn.prev_sibling = prev_last;
        cn.next_sibling = None;
        match prev_last {
            Some(pl) => self.nodes[pl.index()].next_sibling = Some(child),
            None => self.nodes[p].first_child = Some(child),
        }
        self.nodes[p].last_child = Some(child);
        // The child is detached (debug_assert above), so the whole
        // inserted subtree takes the parent's attachment state;
        // template-contents membership is the parent's flag plus the
        // template-to-fragment edge when this link crosses it.
        let attached = self.nodes[p].attached;
        let in_tc =
            self.nodes[p].in_template_contents || self.is_template_contents_edge(parent, child);
        self.set_subtree_flags(child, attached, in_tc);
    }

    /// Link `child` (known detached) immediately before `reference`, which
    /// must already be a child of `parent`.
    fn link_before(&mut self, parent: NodeId, child: NodeId, reference: NodeId) {
        debug_assert!(self.node(child).parent.is_none());
        debug_assert_eq!(self.node(reference).parent, Some(parent));
        debug_assert!(self.owns(parent));
        debug_assert!(self.owns(child));
        debug_assert!(self.owns(reference));
        let p = parent.index();
        let c = child.index();
        let r = reference.index();
        let prev = self.nodes[r].prev_sibling;
        let cn = &mut self.nodes[c];
        cn.parent = Some(parent);
        cn.prev_sibling = prev;
        cn.next_sibling = Some(reference);
        match prev {
            Some(pl) => self.nodes[pl.index()].next_sibling = Some(child),
            None => self.nodes[p].first_child = Some(child),
        }
        self.nodes[r].prev_sibling = Some(child);
        // Same flag derivation as `link_last`: the detached subtree takes
        // the parent's attachment state and template-contents membership.
        let attached = self.nodes[p].attached;
        let in_tc =
            self.nodes[p].in_template_contents || self.is_template_contents_edge(parent, child);
        self.set_subtree_flags(child, attached, in_tc);
    }

    /// Whether `parent` may receive children: it must belong to this
    /// document and be a [`NodeKind::Document`], [`NodeKind::Element`], or
    /// [`NodeKind::DocumentFragment`] node. Fail-closed: foreign ids and
    /// childless node kinds yield `false` (the DOM `HierarchyRequestError`
    /// analogue, without the throw).
    fn can_have_children(&self, parent: NodeId) -> bool {
        matches!(
            self.try_node(parent).map(|n| &n.kind),
            Some(NodeKind::Document | NodeKind::Element(_) | NodeKind::DocumentFragment)
        )
    }

    /// The nodes an insertion child expands to (see
    /// [`expand_insert_child`](Self::expand_insert_child)): either the
    /// child itself, carried inline with no allocation, or a fragment's
    /// children, collected in order. Both arms expose a `&[NodeId]` so
    /// [`document_insert_valid`](Self::document_insert_valid) and
    /// [`insert_nodes`](Self::insert_nodes) keep their slice APIs.
    fn expand_insert_child(&self, child: NodeId) -> Option<InsertNodes> {
        match self.try_node(child).map(|n| &n.kind) {
            None | Some(NodeKind::Document) => None,
            Some(NodeKind::DocumentFragment) => {
                let mut out = Vec::new();
                let mut cursor = self.node(child).first_child;
                while let Some(k) = cursor {
                    if matches!(self.kind(k), NodeKind::Document) {
                        return None;
                    }
                    out.push(k);
                    cursor = self.node(k).next_sibling;
                }
                Some(InsertNodes::Fragment(out))
            }
            Some(_) => Some(InsertNodes::Single([child])),
        }
    }

    /// DOM pre-insert validity for insertions into a [`NodeKind::Document`]
    /// parent, fail-closed. `nodes` are the nodes about to be inserted
    /// (fragments already expanded); `before` is the reference child for
    /// positional checks (`Some(old_child)` for replace, whose slot the
    /// nodes take). Children that will not be there after the operation —
    /// same-parent moves (nodes already parented here) and `removed` (the
    /// node a replace detaches) — are excluded from the counts.
    ///
    /// False when the insertion must not happen: a text node, more than
    /// one new element, more than one new doctype, a second element or
    /// doctype, or a doctype/element ordering violation (an element must
    /// not precede a doctype; a doctype must not follow an element).
    fn document_insert_valid(
        &self,
        parent: NodeId,
        nodes: &[NodeId],
        before: Option<NodeId>,
        removed: Option<NodeId>,
    ) -> bool {
        if !matches!(self.kind(parent), NodeKind::Document) {
            return true;
        }
        if nodes
            .iter()
            .any(|&n| matches!(self.kind(n), NodeKind::Text(_)))
        {
            return false;
        }
        let mut new_elements = 0;
        let mut new_doctypes = 0;
        for &n in nodes {
            match self.kind(n) {
                NodeKind::Element(_) => new_elements += 1,
                NodeKind::DocumentType(_) => new_doctypes += 1,
                _ => {}
            }
        }
        if new_elements > 1 || new_doctypes > 1 {
            return false;
        }
        let mut full: Vec<NodeId> = Vec::new();
        let mut cursor = self.node(parent).first_child;
        while let Some(c) = cursor {
            full.push(c);
            cursor = self.node(c).next_sibling;
        }
        let excluded = |c: NodeId| Some(c) == removed || nodes.contains(&c);
        let is_el = |c: NodeId| matches!(self.kind(c), NodeKind::Element(_));
        let is_dt = |c: NodeId| matches!(self.kind(c), NodeKind::DocumentType(_));
        if new_elements == 1 && full.iter().any(|&c| !excluded(c) && is_el(c)) {
            return false;
        }
        if new_doctypes == 1 && full.iter().any(|&c| !excluded(c) && is_dt(c)) {
            return false;
        }
        let idx = match before {
            Some(r) => full.iter().position(|&k| k == r).unwrap_or(full.len()),
            None => full.len(),
        };
        if new_elements == 1
            && full
                .iter()
                .enumerate()
                .any(|(i, &c)| i >= idx && !excluded(c) && is_dt(c))
        {
            return false;
        }
        if new_doctypes == 1
            && full
                .iter()
                .enumerate()
                .any(|(i, &c)| i < idx && !excluded(c) && is_el(c))
        {
            return false;
        }
        true
    }

    /// Link `nodes` into `parent` in order, before `before` (or at the end
    /// when `None`). Each node is detached from its old parent first (move
    /// semantics, id index maintained per subtree). Callers must have
    /// validated ownership, kinds, cycles, and document validity; `before`
    /// must be a child of `parent` or `None`.
    fn insert_nodes(&mut self, parent: NodeId, nodes: &[NodeId], before: Option<NodeId>) {
        for &n in nodes {
            if let Some(old) = self.node(n).parent {
                self.unlink(n);
                self.note_detach(n);
                // A move, not a fresh attach: the old parent loses a
                // child, so its boxes and selector matching change too.
                self.mark_dirty_and_shadow_host(old, true);
            }
        }
        match before {
            Some(r) => {
                for &n in nodes {
                    self.link_before(parent, n, r);
                    self.note_attach_if_attached(n);
                }
            }
            None => {
                for &n in nodes {
                    self.link_last(parent, n);
                    self.note_attach_if_attached(n);
                }
            }
        }
    }

    /// True when `maybe_ancestor` is `node` itself or reaches it through
    /// light-DOM child edges, shadow-root edges, or any mix of the two.
    ///
    /// Shadow edges matter: a shadow fragment is never linked as a
    /// light-DOM child, so a host can be smuggled into its own shadow
    /// tree (`append_child(shadow_root(host).root, host)`) and the plain
    /// parent-chain walk would miss the host->shadow->host cycle. That
    /// cycle hangs [`Document::clone_node`] (unbounded allocs) and
    /// overflows [`Document::serialize_node`], so the guard walks down
    /// from `maybe_ancestor` following both edge kinds instead.
    ///
    /// Iterative DFS with a visited set, so a cycle that somehow already
    /// exists cannot hang the guard itself.
    ///
    /// Fast path: a childless, shadowless node cannot be a proper
    /// ancestor of anything, so the cycle check is O(1) for fresh nodes
    /// (the common `append_child`/`insert_before` case) instead of
    /// O(subtree).
    fn is_ancestor_or_self(&self, maybe_ancestor: NodeId, node: NodeId) -> bool {
        if maybe_ancestor == node {
            return true;
        }
        if self.node(maybe_ancestor).first_child.is_none()
            && self.shadow_root_fragment(maybe_ancestor).is_none()
        {
            return false;
        }
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut stack = vec![maybe_ancestor];
        while let Some(cur) = stack.pop() {
            if cur == node {
                return true;
            }
            if !visited.insert(cur) {
                continue;
            }
            let mut cursor = self.node(cur).first_child;
            while let Some(k) = cursor {
                if k == node {
                    return true;
                }
                stack.push(k);
                cursor = self.node(k).next_sibling;
            }
            if let Some(root) = self.shadow_root_fragment(cur) {
                stack.push(root);
            }
        }
        false
    }

    /// The shadow fragment attached to `node`'s element data, if any.
    fn shadow_root_fragment(&self, node: NodeId) -> Option<NodeId> {
        match &self.node(node).kind {
            NodeKind::Element(data) => data.shadow_root.as_ref().map(|s| s.root),
            _ => None,
        }
    }

    /// Detach `child` from its parent, repairing sibling and parent links.
    fn unlink(&mut self, child: NodeId) {
        debug_assert!(self.owns(child));
        let c = child.index();
        let parent = self.nodes[c].parent.expect("unlink of detached node");
        let p = parent.index();
        let prev = self.nodes[c].prev_sibling;
        let next = self.nodes[c].next_sibling;
        match prev {
            Some(pl) => self.nodes[pl.index()].next_sibling = next,
            None => self.nodes[p].first_child = next,
        }
        match next {
            Some(nx) => self.nodes[nx.index()].prev_sibling = prev,
            None => self.nodes[p].last_child = prev,
        }
        let cn = &mut self.nodes[c];
        cn.parent = None;
        cn.prev_sibling = None;
        cn.next_sibling = None;
        // Detached: the whole subtree leaves the document, so every node
        // in it becomes unattached. With no parent chain above the root,
        // its template-contents flag recomputes from `false` — the old
        // parent-chain walk stopped at the severed parent link.
        self.set_subtree_flags(child, false, false);
    }
}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Single-pass HTML parsing: the parser drives this sink directly
// ---------------------------------------------------------------------------

/// [`TreeSink`] that builds the document arena straight from the parser,
/// with no intermediate owned tree.
///
/// The stack tracks the current parent: the document node at the bottom,
/// then open elements; a template's contents fragment is pushed while its
/// bracketed children are delivered, so it lands as the template element's
/// last child (the documented template-contents convention).
struct DomSink<'a> {
    doc: &'a mut Document,
    stack: Vec<NodeId>,
    /// Open template-contents brackets. Elements opened while this is
    /// nonzero sit inside `<template>` contents and must not feed the id
    /// index (template contents are out of reach of `get_element_by_id`).
    template_depth: u32,
    /// Whether `start_element` feeds the id index. True when parsing a
    /// fresh document (parse order is document order, so insert-if-absent
    /// lands the first occurrence); false for `set_inner_html`, where ids
    /// colliding with the rest of the document need document-order
    /// duplicate resolution via `note_attach_if_attached` instead.
    index_ids: bool,
    /// Open declarative shadow-root brackets. Shadow children are appended
    /// to the bracket's fragment (out of reach of light-DOM traversal);
    /// at the closing bracket the fragment is recorded on the host's
    /// [`ElementData::shadow_root`].
    shadow_stack: Vec<OpenShadowRoot>,
}

/// One open `start_shadow_root` / `end_shadow_root` bracket.
struct OpenShadowRoot {
    host: NodeId,
    mode: ShadowRootMode,
    delegates_focus: bool,
    clonable: bool,
    fragment: NodeId,
    /// Element-stack depth when the bracket opened. Children append to
    /// the fragment only while the stack is exactly here (i.e. the host
    /// is the current node); once elements open inside the shadow root,
    /// normal parentage applies.
    stack_len: usize,
}

impl<'a> DomSink<'a> {
    fn new(doc: &'a mut Document) -> Self {
        let root = doc.root();
        let mut sink = Self::with_root(doc, root);
        // Fresh document: parse order is document order.
        sink.index_ids = true;
        sink
    }

    /// A sink appending to `root` instead of the document root, without
    /// feeding the id index (for `set_inner_html`, which parses a
    /// fragment into an existing node; see `index_ids`).
    fn with_root(doc: &'a mut Document, root: NodeId) -> Self {
        DomSink {
            doc,
            stack: vec![root],
            template_depth: 0,
            index_ids: false,
            shadow_stack: Vec::new(),
        }
    }

    fn parent(&self) -> NodeId {
        // The driver guarantees a non-empty stack (it starts with the
        // root); a buggy third-party driver fails closed at the document
        // root instead of panicking.
        match self.stack.last() {
            Some(&p) => p,
            None => self.doc.root(),
        }
    }

    /// Where the next appended child lands: the open shadow root's
    /// fragment while the host is the current node (see
    /// [`OpenShadowRoot::stack_len`]), else the current parent.
    fn append_parent(&self) -> NodeId {
        match self.shadow_stack.last() {
            Some(open) if self.stack.len() == open.stack_len => open.fragment,
            _ => self.parent(),
        }
    }

    /// Allocate a node and link it as the last child of the current parent.
    fn append(&mut self, kind: NodeKind) {
        let id = self.doc.alloc(kind);
        let parent = self.append_parent();
        self.doc.link_last(parent, id);
    }
}

impl TreeSink for DomSink<'_> {
    fn doctype(&mut self, d: Doctype) {
        self.append(NodeKind::DocumentType(DocumentTypeData {
            name: d.name,
            public_id: d.public_id,
            system_id: d.system_id,
        }));
    }

    fn text(&mut self, text: String, _loc: Option<SourceLoc>) {
        // Funnel every parsed text run through the document's interner:
        // repeated short runs (whitespace between elements) share one
        // allocation instead of one per node.
        let text = self.doc.intern_text(&text);
        self.append(NodeKind::Text(text));
    }

    fn comment(&mut self, data: String, _loc: Option<SourceLoc>) {
        self.append(NodeKind::Comment(data));
    }

    fn processing_instruction(&mut self, target: String, data: String, _loc: Option<SourceLoc>) {
        self.append(NodeKind::ProcessingInstruction { target, data });
    }

    /// The tokenizer interns names at tokenize time, so the hot path skips
    /// the string round-trip entirely: names arrive as atoms and are
    /// stored directly, with no per-name allocation for well-known names.
    fn start_element_atomized(
        &mut self,
        tag: TagName,
        ns: Namespace,
        attrs: Vec<(AttrName, String)>,
        _loc: Option<SourceLoc>,
    ) {
        // Feed the id index while parsing: parse order is document order,
        // so insert-if-absent lands the first occurrence, matching
        // `get_element_by_id` semantics. Elements inside `<template>`
        // contents are out of reach of the index.
        // Atom fast path for the id scan: tokenizer-lowercased attribute
        // names make `name.atom() == NameAtom::id()` exactly equivalent to
        // the old `attr_name_eq(ns, name, "id")` string scan (see
        // `is_id_name` for the HTML case-insensitivity argument).
        // Interned here, while `attrs` is still borrowed: the index key
        // is the atom, so parsing never clones the id string into the map
        // (one clone per *distinct* id, amortized).
        let id_atom: Option<u32> = attrs
            .iter()
            .find(|(name, _)| name.atom() == NameAtom::id())
            .map(|(_, value)| self.doc.intern_id(value));
        let id = self.doc.alloc(NodeKind::Element(ElementData {
            tag_name: tag,
            namespace: ns.clone(),
            attributes: attrs
                .into_iter()
                .map(|(name, value)| parser_attribute_atomized(&ns, name, value))
                .collect(),
            shadow_root: None,
        }));
        let parent = self.append_parent();
        self.doc.link_last(parent, id);
        if let Some(atom) = id_atom
            && self.template_depth == 0
            && self.shadow_stack.is_empty()
            && self.index_ids
        {
            // First occurrence in parse (== document) order wins, matching
            // the old `or_insert` on the string-keyed map.
            let slot = &mut self.doc.id_index[atom as usize];
            if slot.is_none() {
                *slot = Some(id);
            }
        }
        self.stack.push(id);
    }

    fn start_element(
        &mut self,
        tag: String,
        ns: Namespace,
        attrs: Vec<(String, String)>,
        loc: Option<SourceLoc>,
    ) {
        // A driver without interned names: intern once here and funnel into
        // the atomized path, reversing the old direction (which allocated
        // Strings from atoms and re-interned them). HTML attribute names
        // are ASCII case-insensitive, so `intern_lowered` mirrors
        // `set_attribute`: a driver passing `("ID", …)` in the HTML
        // namespace feeds the id index exactly like the old
        // `attr_name_eq(ns, name, "id")` string scan did.
        // Hoisted out of the closure below: the closure borrows while `ns`
        // itself moves into `start_element_atomized`.
        let is_html = ns == Namespace::Html;
        self.start_element_atomized(
            TagName::from(tag),
            ns,
            attrs
                .into_iter()
                .map(|(name, value)| {
                    let name = if is_html {
                        AttrName::intern_lowered(&name)
                    } else {
                        AttrName::from(name)
                    };
                    (name, value)
                })
                .collect(),
            loc,
        );
    }

    fn end_element(&mut self) {
        // Fail-closed no-op on a buggy driver's unbalanced call.
        self.stack.pop();
    }

    fn start_template_contents(&mut self) {
        let frag = self.doc.alloc(NodeKind::DocumentFragment);
        let parent = self.append_parent();
        self.doc.link_last(parent, frag);
        self.stack.push(frag);
        self.template_depth = self.template_depth.saturating_add(1);
    }

    fn end_template_contents(&mut self) {
        // Fail-closed no-op on a buggy driver's unbalanced call.
        self.stack.pop();
        self.template_depth = self.template_depth.saturating_sub(1);
    }

    fn start_shadow_root(&mut self, mode: ShadowRootMode, delegates_focus: bool, clonable: bool) {
        // The driver emits the bracket in place of the
        // `<template shadowrootmode>` element, so the stack top is the
        // shadow host. Fail closed on a buggy third-party driver.
        let Some(&host) = self.stack.last() else {
            return;
        };
        // The fragment stays unlinked: shadow content is out of reach of
        // light-DOM traversal (like template contents, but without even
        // a fragment-child edge to skip).
        let fragment = self.doc.alloc(NodeKind::DocumentFragment);
        let stack_len = self.stack.len();
        self.shadow_stack.push(OpenShadowRoot {
            host,
            mode,
            delegates_focus,
            clonable,
            fragment,
            stack_len,
        });
    }

    fn end_shadow_root(&mut self) {
        // Fail-closed no-op on a buggy driver's unbalanced call.
        let Some(open) = self.shadow_stack.pop() else {
            return;
        };
        if let Some(data) = self.doc.element_data_mut(open.host) {
            data.shadow_root = Some(ShadowRootData {
                mode: open.mode,
                delegates_focus: open.delegates_focus,
                clonable: open.clonable,
                root: open.fragment,
            });
            self.doc.shadow_hosts.insert(open.fragment, open.host);
        }
    }
}

/// Attribute-name comparison honoring the HTML case-insensitivity rule.
fn attr_name_eq(ns: &Namespace, stored: &str, query: &str) -> bool {
    if *ns == Namespace::Html {
        stored.eq_ignore_ascii_case(query)
    } else {
        stored == query
    }
}

/// Whether a stored attribute's name matches a query name: an integer
/// atom compare on the fast path (both names well-known); the
/// [`attr_name_eq`] string rule when either side spilled onto the heap.
/// `query_raw` is the query as written (for the HTML case-insensitive
/// fallback); `query` is its interned form, reused for the push when the
/// attribute is new.
fn attr_name_matches(
    ns: &Namespace,
    stored: &Attribute,
    query: &AttrName,
    query_raw: &str,
) -> bool {
    let (sa, qa) = (stored.name.atom(), query.atom());
    if sa == NameAtom::UNKNOWN || qa == NameAtom::UNKNOWN {
        attr_name_eq(ns, stored.name.as_str(), query_raw)
    } else {
        sa == qa
    }
}

/// Whether `name` is the `id` attribute name under the namespace's rule
/// (ASCII case-insensitive for HTML, exact for foreign elements): an
/// atom lookup instead of a string compare.
fn is_id_name(ns: &Namespace, name: &str) -> bool {
    (if *ns == Namespace::Html {
        NameAtom::lookup_html(name)
    } else {
        NameAtom::lookup(name)
    }) == Some(NameAtom::id())
}

/// Find an attribute by name, honoring the HTML ASCII case-insensitivity
/// rule. Well-known names take the atom fast path (one table lookup,
/// then an integer scan); anything else falls back to the string scan.
/// Uses the same spill-fallback rule as [`attr_name_matches`]: a stored
/// name that spilled off the atom table (possible only via hand-built
/// `Attribute`s, never via the public DOM APIs) compares by string, so
/// get/set/remove agree on what counts as a match.
fn find_attr<'a>(attrs: &'a [Attribute], ns: &Namespace, name: &str) -> Option<&'a Attribute> {
    let atom = if *ns == Namespace::Html {
        NameAtom::lookup_html(name)
    } else {
        NameAtom::lookup(name)
    };
    match atom {
        Some(a) => attrs.iter().find(|x| {
            let xa = x.name.atom();
            if xa == NameAtom::UNKNOWN {
                attr_name_eq(ns, x.name.as_str(), name)
            } else {
                xa == a
            }
        }),
        None => attrs
            .iter()
            .find(|x| attr_name_eq(ns, x.name.as_str(), name)),
    }
}

// ---------------------------------------------------------------------------
// Selector engine (small, correct; the full CSS engine lives in bowser_css)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Combinator {
    Descendant,
    Child,
    /// `A + B`: `B`'s immediately preceding *element* sibling matches `A`.
    NextSibling,
    /// `A ~ B`: some preceding element sibling of `B` matches `A`.
    SubsequentSibling,
}

/// A selector syntax error: the offending selector string failed to
/// parse, so the query it came from reports the failure instead of
/// silently matching nothing (the DOM `SyntaxError` analogue; see
/// [`Document::query_selector`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectorParseError {
    message: String,
}

impl SelectorParseError {
    fn new(message: impl Into<String>) -> Self {
        SelectorParseError {
            message: message.into(),
        }
    }

    /// Human-readable description of what failed to parse.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for SelectorParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid selector: {}", self.message)
    }
}

impl std::error::Error for SelectorParseError {}

/// One `[attr]`, `[attr=value]`, or `[attr<op>=value]` selector.
#[derive(Debug, Clone)]
struct AttrSelector {
    name: String,
    op: AttrOp,
    value: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttrOp {
    /// `[attr]`: the attribute is present (any value).
    Exists,
    /// `[attr=value]`: exact value equality.
    Equals,
    /// `[attr~=value]`: a whitespace-separated word equals `value`.
    Includes,
    /// `[attr|=value]`: `value` exactly, or `value` followed by `-`.
    DashMatch,
    /// `[attr^=value]`: the value starts with `value`.
    Prefix,
    /// `[attr$=value]`: the value ends with `value`.
    Suffix,
    /// `[attr*=value]`: the value contains `value`.
    Substring,
}

/// Caller-owned reusable state for selector matching.
///
/// Carries the generation-stamped node marker that deduplicates
/// descendant-step visits, the per-combinator-step candidate Vecs (kept
/// alive across steps and calls instead of reallocated per step), and a
/// small cache of parsed selector groups keyed by selector string.
/// Parsing is pure, so a cached parse is always valid; the cache is the
/// bigger win for [`Document::matches`] / [`Document::closest`] /
/// [`Document::query_selector`], which otherwise parse once per call.
///
/// The parse cache is bounded: once it holds
/// [`SelectorScratch::MAX_PARSED_CACHE_ENTRIES`] entries it is cleared
/// wholesale (the cheapest correct eviction for a pure-function cache — a
/// miss simply re-parses). This caps the map plus its `Rc`-held parse
/// trees instead of letting a distinct-selector-per-query workload grow
/// the scratch without limit.
///
/// Own it in the caller (e.g. the style engine) and pass `&mut` to the
/// `*_with_scratch` query variants; the plain query methods are thin
/// wrappers that build a throwaway scratch. The parse cache lives as
/// long as the scratch does — keyed by the exact selector string, it
/// stabilizes at the distinct selector set in play.
///
/// Not shared across threads: keep one per thread or call sequence. It
/// holds no interior mutability, so [`Document`] stays `Send + Sync`
/// regardless.
#[derive(Default)]
pub struct SelectorScratch {
    mark: Vec<u32>,
    generation: u32,
    /// Reused across combinator steps and calls: the current candidate
    /// set, the next candidate set, and the descendant-step walk stack.
    current: Vec<NodeId>,
    next: Vec<NodeId>,
    stack: Vec<NodeId>,
    /// Parsed selector groups by selector string, or the parse error for
    /// selectors that failed to parse (deterministic, so cached too). The
    /// `Rc` lets a cached hit be used while the scratch is mutably
    /// borrowed for matching. Bounded: cleared when it would exceed
    /// [`SelectorScratch::MAX_PARSED_CACHE_ENTRIES`] (see
    /// [`SelectorScratch::parse_groups_cached`]).
    parsed: FnvHashMap<String, Result<Rc<Vec<Selector>>, SelectorParseError>>,
}

impl SelectorScratch {
    /// Cap on the `parsed` selector cache. Parsing is pure, so eviction
    /// is always correct — it only costs a re-parse on the next miss.
    /// 512 entries is generous for real stylesheets (hundreds of
    /// distinct selectors) while capping the map and its `Rc`-held
    /// parse trees at roughly a few hundred kilobytes worst case; a
    /// pathological distinct-selector-per-query workload thrashes and
    /// re-parses instead of growing memory without bound.
    pub const MAX_PARSED_CACHE_ENTRIES: usize = 512;

    /// An empty scratch; its buffers grow on first use.
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance to a fresh generation, growing the marker only when `len`
    /// exceeds its capacity. Returns the new generation id.
    fn next_generation(&mut self, len: usize) -> u32 {
        if self.mark.len() < len {
            self.mark.resize(len, 0);
        }
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Wraparound (practically unreachable): clear all stamps.
            self.mark.fill(0);
            self.generation = 1;
        }
        self.generation
    }

    /// Parse `selectors` into groups, reusing the cached parse when the
    /// same string was seen before.
    fn parse_groups_cached(
        &mut self,
        selectors: &str,
    ) -> Result<Rc<Vec<Selector>>, SelectorParseError> {
        if let Some(hit) = self.parsed.get(selectors) {
            return hit.clone();
        }
        let parsed: Result<Rc<Vec<Selector>>, SelectorParseError> =
            Document::parse_selector_groups(selectors).map(Rc::new);
        if self.parsed.len() >= Self::MAX_PARSED_CACHE_ENTRIES {
            // Unbounded-growth guard: parsing is pure, so dropping the
            // whole cache is always correct; the next miss re-parses.
            self.parsed.clear();
        }
        self.parsed.insert(selectors.to_string(), parsed.clone());
        parsed
    }
}

#[derive(Debug, Default, Clone)]
struct Compound {
    /// `None` means the universal selector (`*` or omitted).
    tag: Option<String>,
    id: Option<String>,
    classes: Vec<String>,
    attrs: Vec<AttrSelector>,
}

#[derive(Debug, Clone)]
struct Selector {
    compounds: Vec<Compound>,
    /// `combinators[i]` joins `compounds[i]` and `compounds[i + 1]`.
    combinators: Vec<Combinator>,
}

/// One token of a selector group: a compound slice (which may contain
/// bracketed attribute selectors) or an explicit combinator.
#[derive(Debug, Clone, Copy)]
enum SelToken<'a> {
    Compound(&'a str),
    Comb(Combinator),
}

impl Selector {
    /// Parse one comma-separated selector group, e.g.
    /// `div.main > p#x.note + [data-k="v"]`.
    ///
    /// Supported syntax per group: a chain of compound selectors joined
    /// by the descendant (whitespace), child (`>`), adjacent-sibling
    /// (`+`), or general-sibling (`~`) combinator. A compound selector is
    /// an optional type selector (or `*`) followed by any number of
    /// `#id`, `.class`, and `[attr]` / `[attr<op>=value]` parts in any
    /// order, with `<op>` one of `=` `~=` `|=` `^=` `$=` `*=`.
    ///
    /// Returns [`SelectorParseError`] when the group is empty or
    /// malformed (unlike the old never-match behavior, callers surface
    /// this as a syntax error).
    fn parse(group: &str) -> Result<Selector, SelectorParseError> {
        let tokens = tokenize_group(group)?;
        if tokens.is_empty() {
            return Err(SelectorParseError::new("empty selector"));
        }
        let mut compounds = Vec::new();
        let mut combinators = Vec::new();
        let mut expect_compound = true;
        for tok in tokens {
            match tok {
                SelToken::Comb(c) => {
                    if expect_compound {
                        return Err(SelectorParseError::new(
                            "unexpected combinator (leading or doubled)",
                        ));
                    }
                    combinators.push(c);
                    expect_compound = true;
                }
                SelToken::Compound(s) => {
                    if !expect_compound {
                        combinators.push(Combinator::Descendant);
                    }
                    compounds.push(parse_compound(s)?);
                    expect_compound = false;
                }
            }
        }
        if expect_compound {
            return Err(SelectorParseError::new("trailing combinator"));
        }
        debug_assert_eq!(combinators.len() + 1, compounds.len());
        Ok(Selector {
            compounds,
            combinators,
        })
    }
}

/// Split one selector group into compound slices and explicit
/// combinator tokens. `>`, `+`, and `~` outside `[...]` are combinators;
/// whitespace separates compounds (the descendant combinator).
/// Bracketed attribute selectors are kept intact inside compound
/// slices, including quoted strings that may contain whitespace or
/// combinator characters. Slicing is on char boundaries: the scan only
/// splits at ASCII bytes.
fn tokenize_group(group: &str) -> Result<Vec<SelToken<'_>>, SelectorParseError> {
    let err = |msg: &str| SelectorParseError::new(msg);
    let mut tokens = Vec::new();
    let bytes = group.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if b == b'>' || b == b'+' || b == b'~' {
            tokens.push(SelToken::Comb(match b {
                b'>' => Combinator::Child,
                b'+' => Combinator::NextSibling,
                _ => Combinator::SubsequentSibling,
            }));
            i += 1;
            continue;
        }
        // A compound: scan to whitespace or a combinator char at bracket
        // depth 0, tracking quotes so `]` / combinators inside a quoted
        // attribute value do not split or close the bracket early.
        let start = i;
        let mut quote: Option<u8> = None;
        let mut depth = 0u32;
        while i < bytes.len() {
            let c = bytes[i];
            if let Some(q) = quote {
                // No escape processing (v1 limitation): a backslash-quote
                // inside the value ends the string early and the group
                // then fails to parse as a syntax error.
                if c == q {
                    quote = None;
                }
                i += 1;
                continue;
            }
            match c {
                b'\'' | b'"' => {
                    quote = Some(c);
                    i += 1;
                }
                b'[' => {
                    depth += 1;
                    i += 1;
                }
                b']' => {
                    if depth == 0 {
                        return Err(err("stray ']'"));
                    }
                    depth -= 1;
                    i += 1;
                }
                b'>' | b'+' | b'~' if depth == 0 => break,
                _ if c.is_ascii_whitespace() && depth == 0 => break,
                _ => i += 1,
            }
        }
        if quote.is_some() {
            return Err(err("unterminated string in attribute selector"));
        }
        if depth != 0 {
            return Err(err("unterminated '[' in attribute selector"));
        }
        tokens.push(SelToken::Compound(&group[start..i]));
    }
    Ok(tokens)
}

/// Parse one compound selector: optional type/`*`, then `#id`, `.class`,
/// and `[attr]` / `[attr<op>=value]` parts in any order.
fn parse_compound(tok: &str) -> Result<Compound, SelectorParseError> {
    let err = |msg: &str| SelectorParseError::new(msg);
    let mut c = Compound::default();
    let mut any = false;
    let bytes = tok.as_bytes();
    let mut i = 0;
    if i < bytes.len() && !matches!(bytes[i], b'.' | b'#' | b'[') {
        let mut j = i;
        while j < bytes.len() && !matches!(bytes[j], b'.' | b'#' | b'[') {
            j += 1;
        }
        let tag = &tok[i..j];
        // A `]` here is stray: the tokenizer only lets `]` through inside
        // `[...]`, which the attr parser below consumes. Anything that is
        // not a CSS ident (e.g. `p:first-child`, `div!`) is a syntax
        // error, not a never-match.
        if tag.is_empty() || tag.bytes().any(|b| b == b']') {
            return Err(err("bad type selector"));
        }
        if tag != "*" && !tag.bytes().all(is_name_char) {
            return Err(err("invalid tag name in selector"));
        }
        c.tag = if tag == "*" {
            None
        } else {
            Some(tag.to_string())
        };
        any = true;
        i = j;
    }
    while i < bytes.len() {
        match bytes[i] {
            b'#' | b'.' => {
                let sigil = bytes[i];
                i += 1;
                let mut j = i;
                while j < bytes.len() && !matches!(bytes[j], b'.' | b'#' | b'[') {
                    j += 1;
                }
                let name = &tok[i..j];
                if name.is_empty()
                    || name.bytes().any(|b| b == b']')
                    || !name.bytes().all(is_name_char)
                {
                    return Err(err("bad id or class name in selector"));
                }
                if sigil == b'#' {
                    if c.id.is_some() {
                        return Err(err("duplicate id selector"));
                    }
                    c.id = Some(name.to_string());
                } else {
                    c.classes.push(name.to_string());
                }
                any = true;
                i = j;
            }
            b'[' => {
                let (attr, next) = parse_attr_selector(tok, i)?;
                c.attrs.push(attr);
                any = true;
                i = next;
            }
            _ => return Err(err("unexpected character in compound selector")),
        }
    }
    if !any {
        return Err(err("empty compound selector"));
    }
    Ok(c)
}

/// CSS ident character for selector names: ASCII alphanumerics, `-`,
/// `_`, and non-ASCII bytes (passed through whole; the tokenizer only
/// ever splits on ASCII, so byte boundaries stay on char boundaries).
fn is_name_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b >= 0x80
}

/// Parse the `[...]` attribute selector at `tok[start]`
/// (`start` points at `[`). Returns the selector and the index just past
/// the closing `]`.
///
/// Grammar: `[` ws* name ws* (`<op>` ws* value ws*)? `]`, where `<op>` is
/// one of `=` `~=` `|=` `^=` `$=` `*=` and `value` is a quoted string or
/// an unquoted word. Operator-specific validity follows CSS: `~=` needs a
/// non-empty word value, and `|=` `^=` `$=` `*=` need a non-empty value;
/// violations are syntax errors, not just non-matches.
fn parse_attr_selector(
    tok: &str,
    start: usize,
) -> Result<(AttrSelector, usize), SelectorParseError> {
    let err = |msg: &str| SelectorParseError::new(format!("bad attribute selector: {msg}"));
    let bytes = tok.as_bytes();
    let mut i = start + 1;
    let skip_ws = |i: &mut usize| {
        while *i < bytes.len() && bytes[*i].is_ascii_whitespace() {
            *i += 1;
        }
    };
    skip_ws(&mut i);
    let name_start = i;
    // Name: CSS ident chars only — anything else (e.g. `!`) ends the
    // name and fails below at the `]` expectation, so `[data-x!="1"]`
    // is a syntax error rather than a match on an attribute named
    // `data-x!`.
    while i < bytes.len() && is_name_char(bytes[i]) {
        i += 1;
    }
    let name = &tok[name_start..i];
    if name.is_empty() {
        return Err(err("missing attribute name"));
    }
    skip_ws(&mut i);
    if i >= bytes.len() {
        return Err(err("unterminated '['"));
    }
    if bytes[i] == b']' {
        return Ok((
            AttrSelector {
                name: name.to_string(),
                op: AttrOp::Exists,
                value: String::new(),
            },
            i + 1,
        ));
    }
    let op = match bytes[i] {
        b'=' => {
            i += 1;
            AttrOp::Equals
        }
        b'~' | b'|' | b'^' | b'$' | b'*' => {
            if bytes.get(i + 1) != Some(&b'=') {
                return Err(err("expected '=' after operator"));
            }
            let op = match bytes[i] {
                b'~' => AttrOp::Includes,
                b'|' => AttrOp::DashMatch,
                b'^' => AttrOp::Prefix,
                b'$' => AttrOp::Suffix,
                _ => AttrOp::Substring,
            };
            i += 2;
            op
        }
        _ => return Err(err("expected operator or ']'")),
    };
    skip_ws(&mut i);
    if i >= bytes.len() {
        return Err(err("unterminated '['"));
    }
    let value = if bytes[i] == b'\'' || bytes[i] == b'"' {
        let q = bytes[i];
        i += 1;
        let vstart = i;
        while i < bytes.len() && bytes[i] != q {
            i += 1;
        }
        if i >= bytes.len() {
            return Err(err("unterminated string"));
        }
        let v = tok[vstart..i].to_string();
        i += 1;
        v
    } else {
        let vstart = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b']' {
            i += 1;
        }
        tok[vstart..i].to_string()
    };
    match op {
        AttrOp::Includes if value.is_empty() || value.bytes().any(|b| b.is_ascii_whitespace()) => {
            return Err(err("~= value must be a single non-empty word"));
        }
        AttrOp::DashMatch | AttrOp::Prefix | AttrOp::Suffix | AttrOp::Substring
            if value.is_empty() =>
        {
            return Err(err("operator value must not be empty"));
        }
        _ => {}
    }
    skip_ws(&mut i);
    if bytes.get(i) != Some(&b']') {
        return Err(err("expected ']'"));
    }
    Ok((
        AttrSelector {
            name: name.to_string(),
            op,
            value,
        },
        i + 1,
    ))
}

// ---------------------------------------------------------------------------
// Live HTMLCollection
// ---------------------------------------------------------------------------

/// A live view over the elements matching
/// [`Document::get_elements_by_tag_name`] or
/// [`Document::get_elements_by_class_name`]: the DOM `HTMLCollection`
/// analogue.
///
/// The query re-runs on every access, so the view reflects DOM mutations
/// made after it was created — no snapshot staleness, and no
/// cache-invalidation bookkeeping in the [`Document`]: each access costs
/// one pre-order walk, O(nodes). Call [`to_vec`](Self::to_vec) if you
/// need to iterate repeatedly.
///
/// Elements inside an HTML `<template>`'s contents fragment are never
/// matched (template contents are out of reach, as in the spec).
pub struct HtmlCollection<'a> {
    doc: &'a Document,
    query: CollectionQuery,
}

#[derive(Clone, Debug)]
enum CollectionQuery {
    TagName(TagNameQuery),
    ClassNames(Vec<String>),
}

/// A pre-atomized `getElementsByTagName` query: the raw text (for `"*"`)
/// plus the selector-name atoms resolved once at query construction, so
/// the per-node scan is an integer compare on the fast path.
#[derive(Clone, Debug)]
struct TagNameQuery {
    text: String,
    /// Case-sensitive atom (foreign elements).
    atom: Option<NameAtom>,
    /// ASCII case-insensitive atom (HTML elements).
    atom_insensitive: Option<NameAtom>,
}

impl TagNameQuery {
    fn new(tag: &str) -> Self {
        TagNameQuery {
            text: tag.to_string(),
            atom: NameAtom::lookup(tag),
            atom_insensitive: NameAtom::lookup_html(tag),
        }
    }

    fn matches(&self, data: &ElementData) -> bool {
        if self.text == "*" {
            return true;
        }
        if data.namespace == Namespace::Html {
            // Spill-first ordering (mirrors `bowser_css::matching`): an
            // element whose tag name is not in the atom table — e.g.
            // `createElementNS(Namespace::Html, "DIV")` keeps its case
            // and spills — falls back to the ASCII case-insensitive
            // string compare instead of silently missing on
            // `UNKNOWN == atom`.
            if data.tag_name.atom() == NameAtom::UNKNOWN {
                data.tag_name.as_str().eq_ignore_ascii_case(&self.text)
            } else {
                match self.atom_insensitive {
                    Some(a) => data.tag_name.atom() == a,
                    None => data.tag_name.as_str().eq_ignore_ascii_case(&self.text),
                }
            }
        } else {
            match self.atom {
                Some(a) => data.tag_name.atom() == a,
                None => data.tag_name.as_str() == self.text,
            }
        }
    }
}

impl<'a> HtmlCollection<'a> {
    /// Number of matching elements, in document order. Re-runs the query.
    pub fn len(&self) -> usize {
        let mut n = 0;
        self.doc.walk_preorder(self.doc.root(), |c| {
            if self.matches_node(c) {
                n += 1;
            }
            true
        });
        n
    }

    /// True when no element matches. Re-runs the query (stops at the
    /// first hit).
    pub fn is_empty(&self) -> bool {
        let mut found = false;
        self.doc.walk_preorder(self.doc.root(), |c| {
            if self.matches_node(c) {
                found = true;
                false
            } else {
                true
            }
        });
        !found
    }

    /// The `index`-th matching element in document order, or `None`.
    /// Re-runs the query.
    pub fn item(&self, index: usize) -> Option<NodeId> {
        let mut seen = 0;
        let mut out = None;
        self.doc.walk_preorder(self.doc.root(), |c| {
            if self.matches_node(c) {
                if seen == index {
                    out = Some(c);
                    return false;
                }
                seen += 1;
            }
            true
        });
        out
    }

    /// The first matching element in document order, or `None`.
    /// Re-runs the query.
    pub fn first(&self) -> Option<NodeId> {
        self.item(0)
    }

    /// All matching elements in document order. Re-runs the query.
    pub fn to_vec(&self) -> Vec<NodeId> {
        let mut out = Vec::new();
        self.doc.walk_preorder(self.doc.root(), |c| {
            if self.matches_node(c) {
                out.push(c);
            }
            true
        });
        out
    }

    /// Iterate the matching elements in document order. Re-runs the query
    /// (the walk happens eagerly, when the iterator is created).
    pub fn iter(&self) -> impl Iterator<Item = NodeId> + 'a {
        self.to_vec().into_iter()
    }

    fn matches_node(&self, n: NodeId) -> bool {
        let NodeKind::Element(data) = self.doc.kind(n) else {
            return false;
        };
        match &self.query {
            CollectionQuery::TagName(q) => q.matches(data),
            CollectionQuery::ClassNames(want) => {
                // An empty class list matches nothing, per the spec.
                if want.is_empty() {
                    return false;
                }
                let class_attr = self.doc.get_attribute(n, "class").unwrap_or("");
                want.iter()
                    .all(|c| class_attr.split_ascii_whitespace().any(|have| have == *c))
            }
        }
    }
}

impl<'a> IntoIterator for HtmlCollection<'a> {
    type Item = NodeId;
    type IntoIter = std::vec::IntoIter<NodeId>;
    fn into_iter(self) -> Self::IntoIter {
        self.to_vec().into_iter()
    }
}

impl Document {
    fn element_matches(&self, el: NodeId, c: &Compound) -> bool {
        let NodeKind::Element(data) = self.kind(el) else {
            return false;
        };
        if let Some(tag) = &c.tag {
            let same = if data.namespace == Namespace::Html {
                data.tag_name.eq_ignore_ascii_case(tag)
            } else {
                data.tag_name == *tag
            };
            if !same {
                return false;
            }
        }
        if let Some(id) = &c.id
            && self.get_attribute(el, "id") != Some(id.as_str())
        {
            return false;
        }
        if !c.classes.is_empty() {
            let class_attr = self.get_attribute(el, "class").unwrap_or("");
            for want in &c.classes {
                if !class_attr
                    .split_ascii_whitespace()
                    .any(|have| have == *want)
                {
                    return false;
                }
            }
        }
        for a in &c.attrs {
            // Name lookup goes through `get_attribute`, so HTML elements
            // match attribute names ASCII case-insensitively, like the
            // DOM. Values are always case-sensitive here (the CSS `i`
            // flag is not supported).
            let have = self.get_attribute(el, &a.name);
            let ok = match a.op {
                AttrOp::Exists => have.is_some(),
                AttrOp::Equals => have == Some(a.value.as_str()),
                AttrOp::Includes => {
                    have.is_some_and(|v| v.split_ascii_whitespace().any(|w| w == a.value))
                }
                AttrOp::DashMatch => have.is_some_and(|v| {
                    v == a.value
                        || v.strip_prefix(a.value.as_str())
                            .is_some_and(|rest| rest.starts_with('-'))
                }),
                AttrOp::Prefix => have.is_some_and(|v| v.starts_with(a.value.as_str())),
                AttrOp::Suffix => have.is_some_and(|v| v.ends_with(a.value.as_str())),
                AttrOp::Substring => have.is_some_and(|v| v.contains(a.value.as_str())),
            };
            if !ok {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

/// HTML void elements: serialized without an end tag (HTML namespace only).
const VOID_ELEMENTS: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
    "track", "wbr",
];

/// HTML raw-text elements: their text children serialize literally, never
/// entity-escaped (WHATWG fragment serialization algorithm). `title` and
/// `textarea` are RCDATA, not raw text: their contents are entity-decoded
/// on parse and must be re-escaped here, so they are deliberately absent.
const RAW_TEXT_ELEMENTS: &[&str] = &["script", "style", "xmp", "iframe", "noembed", "noframes"];

/// True when `node` is a text node whose parent is an HTML-namespace
/// raw-text element.
fn is_raw_text_child(doc: &Document, node: NodeId) -> bool {
    match doc.parent(node).and_then(|p| doc.element_data(p)) {
        Some(data) => {
            data.namespace == Namespace::Html && RAW_TEXT_ELEMENTS.contains(&data.tag_name.as_str())
        }
        None => false,
    }
}

enum StackItem {
    Node(NodeId),
    /// Closing tag for `NodeId`, which must be an element; the tag name
    /// is re-read from the arena instead of cloning a `String` per
    /// element.
    EndTag(NodeId),
}

impl Document {
    fn serialize_open(&self, node: NodeId, out: &mut String, stack: &mut Vec<StackItem>) {
        match self.kind(node) {
            NodeKind::Document | NodeKind::DocumentFragment => {
                push_children_reversed(self, node, stack);
            }
            NodeKind::DocumentType(d) => {
                out.push_str("<!DOCTYPE ");
                out.push_str(&d.name);
                match (&d.public_id, &d.system_id) {
                    (Some(pid), Some(sid)) => {
                        out.push_str(" PUBLIC \"");
                        out.push_str(pid);
                        out.push_str("\" \"");
                        out.push_str(sid);
                        out.push('"');
                    }
                    (Some(pid), None) => {
                        out.push_str(" PUBLIC \"");
                        out.push_str(pid);
                        out.push('"');
                    }
                    (None, Some(sid)) => {
                        out.push_str(" SYSTEM \"");
                        out.push_str(sid);
                        out.push('"');
                    }
                    (None, None) => {}
                }
                out.push('>');
            }
            NodeKind::Text(t) => {
                // Raw-text elements (script, style, ...) serialize their
                // text children literally per the WHATWG fragment
                // serialization algorithm; everything else is escaped.
                if is_raw_text_child(self, node) {
                    out.push_str(t);
                } else {
                    escape_text(t, out);
                }
            }
            NodeKind::Comment(c) => {
                out.push_str("<!--");
                out.push_str(c);
                out.push_str("-->");
            }
            NodeKind::ProcessingInstruction { target, data } => {
                out.push_str("<?");
                out.push_str(target);
                if !data.is_empty() {
                    out.push(' ');
                    out.push_str(data);
                }
                out.push_str("?>");
            }
            NodeKind::Element(data) => {
                let void = data.namespace == Namespace::Html
                    && VOID_ELEMENTS.contains(&data.tag_name.as_str());
                out.push('<');
                out.push_str(&data.tag_name);
                for a in &data.attributes {
                    out.push(' ');
                    out.push_str(&a.name);
                    out.push_str("=\"");
                    escape_attr(&a.value, out);
                    out.push('"');
                }
                if void {
                    out.push('>');
                } else {
                    out.push('>');
                    stack.push(StackItem::EndTag(node));
                    push_children_reversed(self, node, stack);
                }
                // A declarative shadow root re-emits as
                // `<template shadowrootmode>` (WHATWG HTML 13.3 step 4),
                // first inside the host, so a parse→serialize→parse
                // round-trip reattaches it instead of silently dropping
                // it. The light-DOM children above are only queued on the
                // stack, so writing the template inline here keeps it
                // first.
                if let Some(shadow) = &data.shadow_root {
                    self.serialize_shadow_template(shadow, out);
                }
            }
        }
    }

    /// Append one declarative shadow root as `<template shadowrootmode>`
    /// (WHATWG HTML 13.3 step 4): the mode plus the reflected boolean
    /// flags, the shadow children, and the close tag. Shared by
    /// `serialize_open` (host elements) and `inner_html` (which
    /// serializes the host's children directly and would otherwise skip
    /// the root).
    fn serialize_shadow_template(&self, shadow: &ShadowRootData, out: &mut String) {
        out.push_str("<template shadowrootmode=\"");
        out.push_str(match shadow.mode {
            ShadowRootMode::Open => "open",
            ShadowRootMode::Closed => "closed",
        });
        out.push('"');
        if shadow.delegates_focus {
            out.push_str(" shadowrootdelegatesfocus=\"\"");
        }
        if shadow.clonable {
            out.push_str(" shadowrootclonable=\"\"");
        }
        out.push('>');
        out.push_str(&self.serialize_node(shadow.root));
        out.push_str("</template>");
    }
}

/// Push a node's children onto `stack` in reverse document order, using
/// the last-child/prev-sibling links so no temporary list is allocated.
fn push_children_reversed(doc: &Document, node: NodeId, stack: &mut Vec<StackItem>) {
    let mut kid = doc.node(node).last_child;
    while let Some(k) = kid {
        stack.push(StackItem::Node(k));
        kid = doc.node(k).prev_sibling;
    }
}

/// Escape text content by scanning bytes for the (ASCII) special bytes
/// and copying the runs between them, instead of pushing one `char` at a
/// time. Slicing is on char boundaries: every match position is either an
/// ASCII byte or the 0xC2 lead byte of U+00A0, both of which start a char.
fn escape_text(t: &str, out: &mut String) {
    out.reserve(t.len());
    escape_bytes(t, out, &[(b'&', "&amp;"), (b'<', "&lt;"), (b'>', "&gt;")]);
}

fn escape_attr(v: &str, out: &mut String) {
    out.reserve(v.len());
    escape_bytes(v, out, &[(b'&', "&amp;"), (b'"', "&quot;")]);
}

/// Shared byte scanner for the escapers above. `table` maps a special
/// byte to its entity; U+00A0 (bytes C2 A0) maps to `&nbsp;` in both.
fn escape_bytes(s: &str, out: &mut String, table: &[(u8, &str)]) {
    let bytes = s.as_bytes();
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        let mut rep: Option<&str> = None;
        for &(special, entity) in table {
            if b == special {
                rep = Some(entity);
                break;
            }
        }
        let mut len = 1;
        if rep.is_none() && b == 0xC2 && bytes.get(i + 1) == Some(&0xA0) {
            rep = Some("&nbsp;");
            len = 2;
        }
        match rep {
            Some(r) => {
                out.push_str(&s[start..i]);
                out.push_str(r);
                i += len;
                start = i;
            }
            None => i += 1,
        }
    }
    out.push_str(&s[start..]);
}

/// Wave 65 perf tests: the [`Attributes`] small-storage and the
/// atom-keyed id index. Behavior pins: every test below passed on the
/// old `Vec<Attribute>` / `HashMap<String, NodeId>` representation too,
/// except the two size pins, which record the measured win
/// (`ElementData` 88 -> 80 bytes).
#[cfg(test)]
mod w65_p_dom_tests {
    use super::*;

    #[test]
    fn attributes_storage_is_compact_when_empty() {
        // A boxed slice is a fat pointer: 16 bytes inline (vs 24 for
        // Vec), still with no allocation when empty.
        assert_eq!(std::mem::size_of::<Attributes>(), 16);
        assert_eq!(
            std::mem::size_of::<Attributes>(),
            std::mem::size_of::<Option<Box<[Attribute]>>>()
        );
        let a = Attributes::new();
        assert!(a.is_empty());
        assert_eq!(a.len(), 0);
        assert!(a.0.is_none(), "empty Attributes must hold no allocation");
        assert_eq!(a.as_slice(), &[] as &[Attribute]);
    }

    #[test]
    fn element_data_shrank_by_eight_bytes() {
        // 24-byte inline Vec -> 16-byte Option<Box<[Attribute]>>: 8 bytes
        // saved per element node (measured 88 -> 80), and no allocation
        // at all for the common attribute-less element.
        assert_eq!(std::mem::size_of::<ElementData>(), 80);
    }

    #[test]
    fn attributes_push_retain_roundtrip() {
        let mut a = Attributes::new();
        a.push(Attribute::plain("id".into(), "x".into()));
        a.push(Attribute::plain("class".into(), "c".into()));
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].name.as_str(), "id");
        assert_eq!(a[1].value.as_str(), "c");
        assert!(a.0.is_some());
        // Deref read paths agree with the old Vec.
        assert_eq!(a.iter().count(), 2);
        assert_eq!(a.as_slice()[1].name.as_str(), "class");
        a.retain(|attr| attr.name.as_str() != "id");
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].value.as_str(), "c");
        // Removing the last attribute frees back to the empty
        // representation.
        a.retain(|_| false);
        assert!(a.is_empty());
        assert!(a.0.is_none());
    }

    #[test]
    fn attributes_from_iter() {
        let empty: Attributes = Vec::<Attribute>::new().into_iter().collect();
        assert!(empty.0.is_none());
        let two: Attributes = vec![
            Attribute::plain("a".into(), "1".into()),
            Attribute::plain("b".into(), "2".into()),
        ]
        .into_iter()
        .collect();
        assert_eq!(two.len(), 2);
        // Clone/eq behave element-wise, like the old Vec.
        assert_eq!(two.clone(), two);
        assert_ne!(two, Attributes::new());
    }

    #[test]
    fn id_index_tracks_rename() {
        let mut doc = Document::new();
        let root = doc.root();
        let e = doc.create_element("div");
        doc.append_child(root, e);
        assert_eq!(doc.get_element_by_id("a"), None);
        doc.set_attribute(e, "id", "a");
        assert_eq!(doc.get_element_by_id("a"), Some(e));
        // Same-value rewrite: index undisturbed.
        doc.set_attribute(e, "id", "a");
        assert_eq!(doc.get_element_by_id("a"), Some(e));
        // Rename: old id gone, new id present.
        doc.set_attribute(e, "id", "b");
        assert_eq!(doc.get_element_by_id("a"), None);
        assert_eq!(doc.get_element_by_id("b"), Some(e));
        doc.remove_attribute(e, "id");
        assert_eq!(doc.get_element_by_id("b"), None);
    }

    #[test]
    fn id_index_duplicate_first_in_document_order() {
        let mut doc = Document::new();
        // A container: the document node itself takes only one element
        // child per DOM validity, so duplicates need a real parent.
        let root = doc.create_element("div");
        doc.append_child(doc.root(), root);
        let a = doc.create_element("div");
        let b = doc.create_element("span");
        doc.append_child(root, a);
        doc.append_child(root, b);
        doc.set_attribute(a, "id", "dup");
        doc.set_attribute(b, "id", "dup");
        assert_eq!(doc.get_element_by_id("dup"), Some(a));
        // Losing the winner repairs to the next in document order.
        doc.remove_attribute(a, "id");
        assert_eq!(doc.get_element_by_id("dup"), Some(b));
        // And back again on re-gain.
        doc.set_attribute(a, "id", "dup");
        assert_eq!(doc.get_element_by_id("dup"), Some(a));
    }

    #[test]
    fn id_index_ignores_detached_subtrees() {
        let mut doc = Document::new();
        let root = doc.root();
        let e = doc.create_element("div");
        doc.set_attribute(e, "id", "detached");
        assert_eq!(doc.get_element_by_id("detached"), None);
        doc.append_child(root, e);
        assert_eq!(doc.get_element_by_id("detached"), Some(e));
        doc.remove_child(root, e);
        assert_eq!(doc.get_element_by_id("detached"), None);
        // Re-attaching re-indexes.
        doc.append_child(root, e);
        assert_eq!(doc.get_element_by_id("detached"), Some(e));
    }

    #[test]
    fn id_index_parse_duplicate_first_wins() {
        let (doc, _) = Document::parse_html(b"<div id=\"d\"></div><p id=\"d\"></p>");
        let first = doc.get_elements_by_tag_name("div").item(0).unwrap();
        assert_eq!(doc.get_element_by_id("d"), Some(first));
    }

    #[test]
    fn quirks_mode_field_matches_parse_return() {
        // w77 quirks plumbing: the mode returned by `parse_html` must
        // also be the mode the document carries, for all three modes,
        // so downstream layout/style need no separate threading.
        assert_eq!(Document::new().quirks_mode(), QuirksMode::NoQuirks);
        let (doc, q) = Document::parse_html(b"<!DOCTYPE html><html></html>");
        assert_eq!(q, QuirksMode::NoQuirks);
        assert_eq!(doc.quirks_mode(), QuirksMode::NoQuirks);
        let (doc, q) = Document::parse_html(
            b"<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\"><html></html>",
        );
        assert_eq!(q, QuirksMode::LimitedQuirks);
        assert_eq!(doc.quirks_mode(), QuirksMode::LimitedQuirks);
        let (doc, q) = Document::parse_html(b"<html></html>");
        assert_eq!(q, QuirksMode::Quirks);
        assert_eq!(doc.quirks_mode(), QuirksMode::Quirks);
        let mut doc = Document::new();
        doc.set_quirks_mode(QuirksMode::Quirks);
        assert_eq!(doc.quirks_mode(), QuirksMode::Quirks);
    }

    #[test]
    fn id_index_foreign_id_is_case_sensitive() {
        // SVG: only the exact name "id" feeds the index, matching
        // `is_id_name`'s exact-match rule for foreign elements.
        let mut doc = Document::new();
        let root = doc.root();
        let e = doc.create_element_ns(Namespace::Svg, "rect");
        doc.append_child(root, e);
        doc.set_attribute(e, "id", "s");
        assert_eq!(doc.get_element_by_id("s"), Some(e));
        let f = doc.create_element_ns(Namespace::Svg, "circle");
        doc.append_child(root, f);
        doc.set_attribute(f, "ID", "t");
        assert_eq!(doc.get_element_by_id("t"), None);
        assert_eq!(doc.get_attribute(f, "ID"), Some("t"));
    }

    #[test]
    fn id_lookup_miss_allocates_nothing_observable() {
        // A never-seen id misses without growing any table: repeated
        // misses are pure lookups.
        let mut doc = Document::new();
        let root = doc.root();
        let e = doc.create_element("div");
        doc.append_child(root, e);
        doc.set_attribute(e, "id", "real");
        for _ in 0..10 {
            assert_eq!(doc.get_element_by_id("never-seen"), None);
        }
        assert_eq!(doc.get_element_by_id("real"), Some(e));
    }
}
