//! Tab state: one tab's session history, document, and render caches.
//!
//! [`Tab`] is defined here. The browser worker owns the tab vec:
//! `browser.rs`'s `Browser` holds `Vec<Tab>` (see the worker report).

use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Mutex;

use bowser_css::Stylesheet;
use bowser_dom::{Document, NodeId, NodeKind};
use bowser_js::{ElementScrollStates, LayoutEnsurer, ScrollState};
use bowser_layout::{IntrinsicDims, LayoutBox, Size};
use bowser_paint::fonts::FontCache;
use bowser_url::Url;

use crate::loader::effective_base_url;
use crate::script::{PageVm, ScriptReport};
use crate::scroll::ScrollTree;

// ---------------------------------------------------------------------------
// Session history
// ---------------------------------------------------------------------------

/// One tab's session history: an ordered list of visited URLs plus the
/// index of the current entry.
///
/// Forward entries are truncated whenever a new navigation commits via
/// [`SessionHistory::push`].
#[derive(Debug, Default, Clone)]
pub struct SessionHistory {
    entries: Vec<Url>,
    index: usize,
}

impl SessionHistory {
    /// Empty history with no current entry.
    pub fn new() -> Self {
        SessionHistory::default()
    }

    /// The current entry, if any navigation has committed.
    pub fn current(&self) -> Option<&Url> {
        self.entries.get(self.index)
    }

    /// The entry at `index`, if it exists.
    pub fn entry(&self, index: usize) -> Option<&Url> {
        self.entries.get(index)
    }

    /// Number of entries, oldest first in `0..len`.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no navigation has committed yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Index of the current entry. Only meaningful when non-empty.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Commit a new navigation: truncate any forward entries, append
    /// `url`, and make it the current entry.
    pub fn push(&mut self, url: Url) {
        self.entries.truncate(self.index + 1);
        self.entries.push(url);
        self.index = self.entries.len() - 1;
    }

    /// Rewrite the entry at `index` in place, e.g. to the final URL after
    /// a redirect: the installed document was fetched from that URL, so
    /// the entry must not keep pointing at the pre-redirect one.
    /// Out-of-range indices are ignored.
    pub fn replace(&mut self, index: usize, url: Url) {
        if let Some(entry) = self.entries.get_mut(index) {
            *entry = url;
        }
    }

    /// True when there is an older entry to go back to.
    pub fn can_go_back(&self) -> bool {
        !self.entries.is_empty() && self.index > 0
    }

    /// True when there is a newer entry to go forward to.
    pub fn can_go_forward(&self) -> bool {
        self.index + 1 < self.entries.len()
    }

    /// Move back one entry and return the new current entry.
    /// `None` (and no movement) at the oldest entry.
    pub fn back(&mut self) -> Option<&Url> {
        if self.can_go_back() {
            self.index -= 1;
            self.current()
        } else {
            None
        }
    }

    /// Move forward one entry and return the new current entry.
    /// `None` (and no movement) at the newest entry.
    pub fn forward(&mut self) -> Option<&Url> {
        if self.can_go_forward() {
            self.index += 1;
            self.current()
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Display list
// ---------------------------------------------------------------------------

/// The paint crate's display list; see
/// [`bowser_paint::display_list::DisplayList`].
pub use bowser_paint::display_list::DisplayList;

// ---------------------------------------------------------------------------
// Dirty flags
// ---------------------------------------------------------------------------

/// Which render stages need re-running for a tab.
///
/// Set by navigation (via [`mark_all_dirty`]) and by DOM/CSS mutations;
/// cleared by the window worker's `refresh` after it rebuilds the stage.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DirtyFlags {
    /// Style (selector matching + cascade) must re-run.
    pub style: bool,
    /// Layout (box tree) must re-run.
    pub layout: bool,
    /// The display list must be rebuilt.
    pub paint: bool,
}

/// Coalesced dirty-subtree count above which the incremental pipeline
/// gives up and the caller must take the full layout path.
///
/// Why 4: the partial path (wave 51 step 2) only splices a single
/// subtree, and each retained root costs a tree search plus gate checks
/// in `refresh`. Beyond a handful of disjoint dirtied subtrees the
/// bookkeeping and repeated gate walks cost more than they can save
/// (a full pass is one linear walk), so the report collapses to the
/// `full` fallback instead of pretending the dirt is localized.
pub const DIRTY_SUBTREE_FULL_THRESHOLD: usize = 4;

/// Coalesce a set of dirtied subtree roots: drop duplicates and any
/// root that is a descendant of another root in the set (the ancestor's
/// recomputation already covers it), and sort by arena index so the
/// result is deterministic (`Document::take_dirty` drains a hash set,
/// whose order is not).
///
/// Ancestry is resolved through `doc`'s parent links; a root whose
/// parent chain reaches another listed root is dropped. Roots that are
/// no longer attached simply never match an ancestor and are kept —
/// the layout stage's box search treats them as a full-path fallback.
pub fn coalesce_dirty_subtrees(doc: &Document, subtrees: &[NodeId]) -> Vec<NodeId> {
    let mut roots: Vec<NodeId> = Vec::with_capacity(subtrees.len());
    for &node in subtrees {
        if !roots.contains(&node) {
            roots.push(node);
        }
    }
    let keep: Vec<bool> = roots
        .iter()
        .map(|&node| {
            let mut cursor = doc.parent(node);
            while let Some(parent) = cursor {
                if roots.contains(&parent) {
                    return false;
                }
                cursor = doc.parent(parent);
            }
            true
        })
        .collect();
    let mut keep = keep.into_iter();
    roots.retain(|_| keep.next().unwrap_or(true));
    roots.sort_by_key(|node| node.index());
    roots
}

/// Whether a coalesced dirty report forces the full layout path: it
/// names the document node or the document element (either one roots
/// effectively the whole render tree), or it exceeds
/// [`DIRTY_SUBTREE_FULL_THRESHOLD`] disjoint subtrees.
pub fn dirty_report_forces_full(doc: &Document, coalesced: &[NodeId]) -> bool {
    coalesced.len() > DIRTY_SUBTREE_FULL_THRESHOLD
        || coalesced.contains(&doc.root())
        || doc
            .document_element()
            .is_some_and(|root| coalesced.contains(&root))
}

// ---------------------------------------------------------------------------
// Late-image load bookkeeping
// ---------------------------------------------------------------------------

/// Per-`<img>` fetch bookkeeping for late image loads.
///
/// The loader fetches each `<img>`'s `src` once per document load.
/// Images added later — or whose `src` changes after load — are fetched
/// on demand by the script-pump driver through
/// [`Tab::pending_late_images`], which records every fetch here, and
/// completed through [`Tab::note_image_loaded`].
///
/// `url` is the serialized *request* URL of the most recent fetch
/// started for this node: `src` resolved against the document base, the
/// same key the loader image cache uses. The driver detects a `src`
/// mutation by comparing the element's current resolved `src` against
/// this recorded URL, so an already-loaded `<img>` whose `src` changes
/// is refetched even though it already has an entry in
/// [`Tab::image_intrinsics`].
///
/// `generation` is bumped every time a (re)fetch starts for the node. A
/// completion carries the generation its fetch started with, and
/// [`Tab::note_image_loaded`] applies it only when it matches the
/// recorded generation: completions that arrive out of order (src
/// A→B→A with B's slow fetch landing last) are stale and dropped
/// instead of installing the wrong size. `completed_generation` records
/// the generation of the last applied completion — success or failure —
/// so a duplicate delivery of the same completion is dropped rather
/// than repainting twice, and a later retry starts from a
/// distinguishable state.
#[derive(Debug, Clone)]
pub struct ImageLoad {
    /// Serialized request URL of the most recent fetch for this node.
    pub url: String,
    /// Generation counter, bumped by [`Tab::pending_late_images`] on
    /// every (re)fetch start. Seeded as 0 for load-time images.
    pub generation: u64,
    /// Generation of the last completion applied by
    /// [`Tab::note_image_loaded`], `None` while the current generation
    /// has not completed. Set on failures too, so a retry is
    /// distinguishable from the failed attempt.
    pub completed_generation: Option<u64>,
}

/// One late-image fetch the driver must start: the unit returned by
/// [`Tab::pending_late_images`].
///
/// The driver runs
/// [`fetch_late_image`](crate::loader::fetch_late_image) with `url` on
/// a worker thread, then feeds the outcome back on the tab-owning
/// thread via [`Tab::note_image_loaded`] as `(node, generation,
/// result)`.
#[derive(Debug, Clone)]
pub struct PendingLateImage {
    /// The `<img>` element to fetch for.
    pub node: NodeId,
    /// Resolved request URL: `src` against the document base.
    pub url: Url,
    /// The generation [`Tab::note_image_loaded`] must see back with the
    /// completion.
    pub generation: u64,
}

// ---------------------------------------------------------------------------
// RenderPipeline
// ---------------------------------------------------------------------------

/// One tab's render-pipeline state: everything the style / layout /
/// scroll-tree / publish stages read and write.
///
/// Split out of [`Tab`] as a direct field so a page re-entry (eval/pump)
/// can hand script natives a disjoint `&mut RenderPipeline` while the
/// worker holds `&mut Tab::installed`'s page VM: the borrow checker
/// proves the two borrows disjoint, so a geometry getter may force a
/// synchronous reflow ([`bowser_js::LayoutEnsurer`]) without aliasing
/// the VM borrow the pump holds. [`Tab`] derefs to [`RenderPipeline`]
/// (`Deref`/`DerefMut`), so existing `tab.dirty`-style accesses keep
/// working unchanged; only code that needs the disjoint split names
/// `tab.render` explicitly.
///
/// (`LayoutBox` has no `Debug` impl, so neither does `RenderPipeline`.)
pub struct RenderPipeline {
    /// Author stylesheets (`<style>` elements and fetched `<link>` sheets)
    /// of the current document, in document order.
    pub author: Vec<Stylesheet>,
    /// Intrinsic image sizes by `<img>` node, from the loader, for
    /// `bowser_layout::layout_with_intrinsics`. Kept across
    /// same-document (fragment) navigations; replaced on full navigation.
    ///
    /// Installed once at load time, then updated as late images
    /// complete: [`Tab::note_image_loaded`] records a late `<img>`'s
    /// decoded size and invalidates paint on every new completion —
    /// layout only when the size actually changed — so an image inserted
    /// by script after the initial layout grows to its real size instead
    /// of staying a 0x0 box. A failed load leaves the recorded size
    /// untouched. Per-node fetch generations and recorded request URLs
    /// live in [`Tab::image_loads`].
    pub image_intrinsics: HashMap<NodeId, IntrinsicDims>,
    /// Cached layout tree; rebuilt by the window worker when
    /// [`DirtyFlags::layout`] is set.
    ///
    /// Shared (`Rc`) rather than owned: the page's web APIs publish
    /// the very same tree ([`bowser_js::share_layout_tree`]), so a
    /// refcount bump replaces the old per-pass deep clone. Readers
    /// that need `&LayoutBox` go through `.as_deref()`.
    pub layout_root: Option<Rc<LayoutBox>>,
    /// Layout tree with the nested scroll containers' offsets baked in
    /// (scrolled subtrees shifted, scrollports fixed): the display list
    /// is built from this copy, so nested scrolling moves content
    /// without a relayout. Rebuilt after each layout; `None` exactly
    /// when [`RenderPipeline::layout_root`] is `None`.
    pub(crate) paint_root: Option<LayoutBox>,
    /// Viewport size in CSS px.
    pub viewport: Size,
    /// The scroll containers of the current layout: the viewport
    /// scroller (index 0, aliasing [`RenderPipeline::scroll()`]) plus one entry per
    /// nested scroll container. Rebuilt by the window worker's `refresh`
    /// after each layout; viewport-only before the first layout.
    pub scroll_tree: ScrollTree,
    /// Which render stages are stale.
    pub dirty: DirtyFlags,
    /// Coalesced subtree roots of the latest unserviced
    /// [`bowser_dom::DirtyReport`], retained by
    /// [`RenderPipeline::sync_dirty_flags`] next to the collapsed
    /// [`RenderPipeline::dirty`] bools (which stay the stage contract for
    /// every other caller).
    ///
    /// This is the incremental pipeline's input: `refresh` may lay out
    /// only these subtrees and splice them into [`RenderPipeline::layout_root`]
    /// when [`RenderPipeline::dirty_full_fallback`] is clear and every conservative
    /// gate passes. Cleared by `refresh` once the layout stage has
    /// serviced the dirt (partially or via the full path), and by
    /// [`Tab::install_document`], since a new document mints new
    /// [`NodeId`]s.
    pub dirty_subtrees: Vec<NodeId>,
    /// Coalesced roots of the [`bowser_dom::DirtyReport::style_subtrees`]
    /// sets retained alongside [`RenderPipeline::dirty_subtrees`]: subtrees whose
    /// selector matching may have changed but whose restyle scope is
    /// provably confined to the subtree (attribute writes with no
    /// shadow-DOM involvement). `refresh`'s partial path may service
    /// style dirt only when this coalesces to exactly the single
    /// [`RenderPipeline::dirty_subtrees`] root and the stylesheet precondition
    /// holds (see `try_partial_layout`); anything else fails closed to
    /// the full path. Cleared with the rest of the retained report.
    pub style_subtrees: Vec<NodeId>,
    /// True when the retained dirty report cannot be served
    /// incrementally and `refresh` must take the full layout path: the
    /// report named the document/document element, exceeded
    /// [`DIRTY_SUBTREE_FULL_THRESHOLD`] disjoint subtrees, or a caller
    /// asked for every stage via [`mark_all_dirty`].
    pub dirty_full_fallback: bool,
    /// The viewport the current [`RenderPipeline::layout_root`] tree was laid
    /// out at by the last *full* layout pass (`refresh` records it on
    /// every full pass), or `None` when no full pass has run for the
    /// current tree. Belt-and-braces for the partial path (wave 51
    /// rework F1): a retained subtree report splices into a tree laid
    /// out at this viewport, so `try_partial_layout` refuses when the
    /// tab's live viewport differs — even if a non-DOM layout-dirt
    /// writer somehow reached `refresh` without forcing the full
    /// fallback through [`RenderPipeline::mark_layout_dirty_full`].
    pub last_layout_viewport: Option<Size>,
    /// Test-visible pipeline counters: full layout passes `refresh`
    /// has run for this tab.
    pub layout_full_passes: u64,
    /// Test-visible pipeline counters: partial (single-subtree splice)
    /// layout passes `refresh` has run for this tab.
    pub layout_partial_passes: u64,
    /// Test-visible pipeline counters: layout boxes spliced into the
    /// previous tree by partial passes, totalled across passes (the
    /// root box of each spliced subtree plus its descendants).
    pub boxes_spliced: u64,
    /// Scroll offset in CSS px, shared with the page's JS web APIs.
    ///
    /// An `Rc` (not a plain tuple) so the bindings hold the same state
    /// (attached at document install via
    /// [`bowser_js::share_scroll_state`]): `window.scrollTo`,
    /// `Element.scrollTop`, and friends read and write the very cells
    /// the renderer's wheel/keyboard paths mutate — script and wheel can
    /// never disagree about the offset. Every write clamps into
    /// `[0, max]`; `max` (content extent minus viewport) is published by
    /// the window worker's `refresh` after each layout.
    ///
    /// NEVER reassign this field after [`Tab::new`]: the bindings alias
    /// this exact allocation (shared at document install), so assigning
    /// a new `Rc` would silently fork script-visible scroll state from
    /// the renderer/wheel state. Privacy enforces the invariant: the
    /// only handle is [`RenderPipeline::scroll`], which hands out
    /// `&Rc<ScrollState>` — callers can clone the handle or mutate the
    /// cells in place, but nobody outside this module can replace the
    /// allocation. Reset in place ([`ScrollState::reset`]) instead of
    /// replacing.
    scroll: Rc<ScrollState>,
    /// Per-element scroll states for the nested scroll containers,
    /// shared with the page's JS web APIs (attached at document install
    /// via [`bowser_js::share_element_scroll_states`]): the same cells
    /// the wheel/keyboard paths mutate, so `Element.scrollTop` and
    /// friends read and write the live offsets. Maxima are published by
    /// the window worker's `refresh` after each layout; a fresh document
    /// starts empty.
    ///
    /// NEVER reassign this field after [`Tab::new`], for the same reason
    /// as [`RenderPipeline::scroll`]: the bindings alias this exact allocation
    /// (shared at document install), so a new `Rc` would silently fork
    /// script-visible element scroll state from the renderer's. Privacy
    /// enforces the invariant: the only handle is
    /// [`RenderPipeline::element_scroll`], which hands out `&Rc<ElementScrollStates>`
    /// — callers can clone the handle or mutate the cells in place, but
    /// nobody outside this module can replace the allocation. Reset
    /// in place instead of replacing.
    element_scroll: Rc<ElementScrollStates>,
}

impl RenderPipeline {
    /// A fresh pipeline for `viewport`: no document, no layout, no dirt.
    fn new(viewport: Size) -> Self {
        let scroll = Rc::new(ScrollState::new());
        RenderPipeline {
            author: Vec::new(),
            image_intrinsics: HashMap::new(),
            layout_root: None,
            paint_root: None,
            viewport,
            scroll_tree: ScrollTree::for_viewport(&scroll, viewport, (true, true)),
            dirty: DirtyFlags::default(),
            dirty_subtrees: Vec::new(),
            style_subtrees: Vec::new(),
            dirty_full_fallback: false,
            last_layout_viewport: None,
            layout_full_passes: 0,
            layout_partial_passes: 0,
            boxes_spliced: 0,
            scroll,
            element_scroll: Rc::new(ElementScrollStates::new()),
        }
    }

    /// The viewport scroll state (shared with the bindings; see the
    /// field docs). Hands out `&Rc<ScrollState>`: callers can clone the
    /// handle or mutate the cells in place, but not replace the
    /// allocation.
    pub fn scroll(&self) -> &Rc<ScrollState> {
        &self.scroll
    }

    /// The per-element scroll states (shared with the bindings; see the
    /// field docs). Hands out `&Rc<ElementScrollStates>`; the allocation
    /// is never replaced.
    pub fn element_scroll(&self) -> &Rc<ElementScrollStates> {
        &self.element_scroll
    }

    /// Fold the document's dirty report into the stage flags and retain
    /// the coalesced subtree roots for the incremental path. `doc` is
    /// the installed document's shared tree.
    pub fn sync_dirty_flags(&mut self, doc: &RefCell<Document>) {
        let (report, coalesced, style_coalesced, forces_full) = {
            let mut doc = doc.borrow_mut();
            let report = doc.take_dirty();
            if report.is_empty() {
                return;
            }
            // Retain the report's subtree roots next to the stage
            // flags, merged with any report an earlier sync retained
            // but `refresh` has not serviced yet: dirt accumulates
            // across syncs, then coalesces as one set (a later mutation
            // of an ancestor absorbs an earlier retained descendant).
            // (Cloned, not taken: the caller already holds the
            // document.) The style-localized roots merge and coalesce
            // identically, so the partial path's
            // `style_subtrees == dirty_subtrees` check compares like
            // with like.
            let mut merged = self.dirty_subtrees.clone();
            merged.extend_from_slice(&report.subtrees);
            let coalesced = coalesce_dirty_subtrees(&doc, &merged);
            let mut style_merged = self.style_subtrees.clone();
            style_merged.extend_from_slice(&report.style_subtrees);
            let style_coalesced = coalesce_dirty_subtrees(&doc, &style_merged);
            let forces_full = dirty_report_forces_full(&doc, &coalesced);
            (report, coalesced, style_coalesced, forces_full)
        };
        self.dirty_subtrees = coalesced;
        self.style_subtrees = style_coalesced;
        // The full fallback is sticky until `refresh` services the
        // dirt: a report that once forced a full pass must not become
        // partial again just because a later sync added a small root.
        self.dirty_full_fallback |= forces_full;
        // A script-driven scroll offset change needs a full layout: the
        // sticky constraints bake at layout time, and the partial
        // subtree splice does not re-bake them at the new offsets.
        self.dirty_full_fallback |= report.scroll_offset;
        // Any DOM mutation can move boxes: layout and paint re-run.
        // Justified direct write (not `mark_layout_dirty_full`): this
        // is the report producer — the retained subtrees set above
        // are exactly what a partial pass may service, and
        // `forces_full` above already forced the fallback when the
        // report itself demands it.
        self.dirty.layout = true;
        self.dirty.paint = true;
        self.dirty.style |= report.style;
    }

    /// Drop the retained dirty report: called by `refresh` once the
    /// layout stage has serviced the dirt, and on document install.
    pub(crate) fn clear_dirty_report(&mut self) {
        self.dirty_subtrees.clear();
        self.style_subtrees.clear();
        self.dirty_full_fallback = false;
    }

    /// Mark layout dirty for a reason that is *not* a retained DOM
    /// [`bowser_dom::DirtyReport`] subtree: a resize, a sticky
    /// re-bake after scroll, a late image's intrinsic size arriving,
    /// a full invalidation. Such dirt can move *every* box (the whole
    /// tree was laid out at other geometry), so no retained subtree
    /// report may be served incrementally against it: this sets
    /// [`DirtyFlags::layout`], forces the full fallback, and drops
    /// the retained subtrees, so a same-turn text edit cannot let a
    /// partial splice silently service and discard this dirt (wave
    /// 51 rework F1). Every direct non-report writer of
    /// `dirty.layout` in `bowser_app` routes through here; the only
    /// exceptions are [`RenderPipeline::sync_dirty_flags`] (the report producer,
    /// which sets the retained report it is about to service) and
    /// test fixtures.
    pub fn mark_layout_dirty_full(&mut self) {
        self.dirty.layout = true;
        self.dirty_full_fallback = true;
        self.dirty_subtrees.clear();
        self.style_subtrees.clear();
    }
}

// ---------------------------------------------------------------------------
// InstalledDocument
// ---------------------------------------------------------------------------

/// A tab's installed document together with its JS realm: the pair the
/// renderer and the JS bindings work through.
///
/// The renderer reads the DOM through [`Tab::document_ref`]; the
/// bindings reach it through a registry [`Weak`](std::rc::Weak) to the
/// tab's strong handle, upgraded per native call — so the registry can
/// never keep the document alive past its install or operate on a
/// different tree. Bundling the two in one struct makes the atomic swap
/// a type-level invariant: there is no way to write one without the
/// other.
/// [`Tab::install_document`] is the only installer; readers go through
/// [`InstalledDocument::document`], [`InstalledDocument::page_vm`], or
/// [`Tab::document_ref`].
pub struct InstalledDocument {
    /// The single shared DOM tree. The tab owns the sole strong `Rc`;
    /// the JS bindings hold only a `Weak` (upgraded per native call)
    /// and the renderer borrows the same allocation — script
    /// mutations land in the tree the renderer reads, with no
    /// per-consumer copies. Mutations record dirt on the document itself
    /// (see [`Document::take_dirty`]); [`Tab::sync_dirty_flags`] folds
    /// that dirt into [`Tab::dirty`].
    document: Rc<RefCell<Document>>,
    /// The current document's JS realm: a fresh [`PageVm`] installed on
    /// every full navigation (one realm per document, like a new
    /// `Window`), kept across same-document fragment navigations, dropped
    /// with the tab.
    page_vm: PageVm,
}

impl InstalledDocument {
    /// Bundle a freshly loaded document with its page realm.
    /// `page_vm` must have been built for `document` — e.g. via
    /// [`PageVm::new_for_document`](crate::script::PageVm::new_for_document).
    /// Crate-private so only [`Tab::install_document`] can build the
    /// bundle; document and realm always swap as one atomic pair.
    ///
    /// The pairing canary below is the guarded boundary of this
    /// constructor: when the realm carries web APIs, the registry
    /// `Weak` must upgrade to exactly the document being installed
    /// with it. A mispairing (realm built for document A, installed
    /// alongside document B) would otherwise diverge silently —
    /// natives would mutate A through the registry `Weak` while the
    /// renderer reads B. Panics in all builds, like the pre-Weak
    /// `assert_shared_document` this replaces. Install-time only, zero
    /// per-dispatch cost.
    pub(crate) fn new(document: Rc<RefCell<Document>>, mut page_vm: PageVm) -> Self {
        if let Some(ctx) = page_vm.web_api_context() {
            let paired = page_vm.vm_mut().web_apis_document(ctx);
            assert!(
                paired
                    .as_ref()
                    .is_some_and(|live| Rc::ptr_eq(live, &document)),
                "install_document: page realm's web APIs resolve to a document \
                 other than the one being installed"
            );
        }
        InstalledDocument { document, page_vm }
    }

    /// The shared DOM tree: the renderer's view of the document.
    ///
    /// Returns a borrow of the tab's `RefCell`, never the `Rc` itself:
    /// readers can read and mutate the tree, but nothing outside the
    /// install path can mint a second strong handle to it. The tab
    /// keeps the sole strong owner; out-of-band readers that cannot
    /// hold the tab's borrow use [`InstalledDocument::with_document`],
    /// which runs a closure against the cell without ever handing out
    /// the `Rc`.
    pub(crate) fn document(&self) -> &RefCell<Document> {
        &self.document
    }

    /// Scoped access to the shared DOM tree: the sole sanctioned way to
    /// reach the document without holding the tab's own borrow (e.g.
    /// the paint pass assigning `display_list` while reading the tree).
    ///
    /// The strong handle is upgraded, handed to `f` as a bare cell
    /// borrow, and dropped before this returns — the `Rc` can never
    /// escape into a stored field, so a second persistent owner is a
    /// compile-time impossibility rather than a convention. (The old
    /// `document_weak` escape hatch is gone for exactly this reason:
    /// any future caller could upgrade it and stash the `Rc`, smuggling
    /// a second owner past the install-time sole-ownership assert.)
    ///
    /// Returns `None` only if the document is already gone, which
    /// cannot happen while the install is live; callers that checked
    /// `has_document`/`installed` treat `None` as unreachable.
    pub(crate) fn with_document<R>(&self, f: impl FnOnce(&RefCell<Document>) -> R) -> Option<R> {
        Rc::downgrade(&self.document)
            .upgrade()
            .map(|shared| f(&shared))
    }

    /// Disjoint borrow of the bundle for event dispatch: the shared DOM
    /// handle and the mutable realm. The fields are disjoint, so a
    /// dispatch can hold both at once without cloning the `Rc`.
    ///
    /// Reentrancy invariant: never hold a `Ref`/`RefMut` of the
    /// document across `PageVm` re-entry. Dispatch holds this
    /// `&RefCell<Document>` while calling `page_vm.eval`, and natives
    /// re-borrow the same cell through the registry `Weak` on every
    /// call — a borrow alive across an eval panics at runtime on the
    /// re-borrow. Keep every borrow scoped (borrow, copy out, drop)
    /// before each eval; every dispatch eval entry point in
    /// `crate::script` and `crate::window` runs the all-builds
    /// [`InstalledDocument::assert_no_live_borrow`] gate first.
    pub(crate) fn document_and_vm_mut(&mut self) -> (&RefCell<Document>, &mut PageVm) {
        (&self.document, &mut self.page_vm)
    }

    /// All-builds reentrancy gate for eval dispatch entry points.
    ///
    /// Asserts (in every build profile, not debug-only) that `cell`
    /// — the bundle's shared document cell — currently has no live
    /// borrow of either kind. Natives re-borrow the document through
    /// the registry `Weak` on every call, so a borrow held across a
    /// page re-entry would panic on the re-borrow deep inside a
    /// native: a held `RefMut` breaks even a shared re-borrow, and a
    /// held `Ref` breaks any mutating native. The old `debug_assert!`
    /// guards at the dispatch entry points only probed
    /// `try_borrow` (shared-borrow-shaped) in debug builds; in
    /// release the backstop was the raw `RefCell` panic — a confusing
    /// crash with no invariant context. Tripping here names the
    /// violated invariant at the dispatch boundary instead.
    ///
    /// The strictness is deliberate: the gate rejects even an
    /// uncontended shared borrow, not just a conflicting one. A `Ref`
    /// alive across re-entry is a latent bug — any mutating native
    /// would panic on its re-borrow — so fail fast here with the
    /// invariant named, rather than letting a future refactor's
    /// statement-temporary grow into a confusing deep-VM panic. One
    /// probe suffices: `try_borrow_mut` fails on any live borrow,
    /// shared or exclusive, so it subsumes the old `try_borrow`
    /// probe.
    ///
    /// Cost: one `RefCell` flag read per call — noise next to an
    /// eval or a pump. Call it immediately before every
    /// `page_vm.eval`, `page_vm.pump`, or `run_inline` re-entry made
    /// while the bundle's `&RefCell<Document>` is held.
    pub(crate) fn assert_no_live_borrow(cell: &RefCell<Document>) {
        assert!(
            cell.try_borrow_mut().is_ok(),
            "document borrow held across page re-entry"
        );
    }

    /// The document's JS realm: the bindings' view of the document.
    pub fn page_vm(&self) -> &PageVm {
        &self.page_vm
    }

    /// The document's JS realm, mutably: the entry point for event
    /// dispatch and timer pumping.
    pub fn page_vm_mut(&mut self) -> &mut PageVm {
        &mut self.page_vm
    }
}

// ---------------------------------------------------------------------------
// Forced sync layout (LayoutEnsurer)
// ---------------------------------------------------------------------------

/// [`bowser_js::LayoutEnsurer`] for production tabs: a geometry getter
/// forces a synchronous style/layout/publish reflow when the tab is
/// dirty.
///
/// Holds raw handles captured at page re-entry ([`Tab::reenter_page`]):
/// the render pipeline, the installed document's cell, the web-API
/// context, and the font cache. The handles are valid exactly for the
/// re-entry's duration — the tab cannot move while the worker holds its
/// borrow chain — and the slot is cleared when the re-entry ends.
///
/// # Safety contract
///
/// - `render` points at the tab's `RenderPipeline`, which outlives the
///   re-entry (the worker's borrow chain pins the tab). The native
///   derives `&mut RenderPipeline`, disjoint from the `&mut PageVm` the
///   pump holds (disjoint fields of `Tab`; the borrow checker proves it
///   at the re-entry site). No other `&mut RenderPipeline` exists
///   during the re-entry.
/// - `document` points at the `RefCell` inside the installed bundle's
///   `Rc<RefCell<Document>>`, kept alive by the bundle for the
///   re-entry's duration. Only shared reborrows are derived.
/// - `fonts` points at the worker's `FontCache`, alive for the worker's
///   lifetime. The worker never holds the lock across a re-entry, so
///   the native's lock cannot deadlock.
/// - `ensure_layout` never runs script and never touches the page VM.
pub(crate) struct AppLayoutEnsurer {
    render: *mut RenderPipeline,
    document: *const RefCell<Document>,
    ctx: u32,
    fonts: *const Mutex<FontCache>,
    // Opt out of the auto `Send`/`Sync` raw pointers would grant: the
    // handles are valid only on the tab-owning thread for the re-entry's
    // duration, so thread transfer must be a compile error, not a
    // latent soundness hazard. `Rc` is `!Send + !Sync`, and the field is
    // zero-sized.
    _thread_pinned: PhantomData<Rc<()>>,
}

impl LayoutEnsurer for AppLayoutEnsurer {
    fn ensure_layout(&self) -> bool {
        // SAFETY: see the struct docs. The re-entry site guarantees the
        // tab is pinned and the borrows disjoint; the pointers are only
        // dereferenced here, during the re-entry.
        let render: &mut RenderPipeline = unsafe { &mut *self.render };
        let document: &RefCell<Document> = unsafe { &*self.document };
        let fonts: &Mutex<FontCache> = unsafe { &*self.fonts };
        // The getter may be the first thing to observe the task's
        // mutations: pull the document's dirty flags into the pipeline
        // before deciding whether a pass is needed
        // (`refresh_style_layout` itself does not sync, preserving the
        // base `refresh` contract for manual flag management).
        render.sync_dirty_flags(document);
        crate::window::refresh_style_layout(render, document, Some(self.ctx), fonts)
    }
}

/// Restores the web-API context's [`LayoutEnsurer`] slot on drop:
/// [`Tab::reenter_page`] installs the tab's slot for the re-entry and
/// the guard puts the previous one back (usually none), so nested
/// re-entries unwind correctly.
pub(crate) struct LayoutEnsurerGuard {
    ctx: u32,
    prev: Option<Rc<dyn LayoutEnsurer>>,
}

impl AppLayoutEnsurer {
    /// Slot for a document not yet installed in the tab (initial
    /// load): the render pipeline comes from the tab, the document
    /// cell from the fresh `Rc` the caller keeps alive, plus the
    /// web-API context and the font cache.
    ///
    /// The caller upholds the struct's safety contract with two
    /// additions:
    /// - the tab's render pipeline (and the tab itself) is not
    ///   accessed while the slot is live;
    /// - the slot is cleared (via [`install_layout_ensurer`]'s guard)
    ///   before `document`'s `Rc` is dropped or moved into the tab
    ///   (the raw pointer does not disturb `install_document`'s
    ///   sole-ownership gate).
    pub(crate) fn for_pending_install(
        render: &mut RenderPipeline,
        document: &Rc<RefCell<Document>>,
        ctx: u32,
        fonts: &Mutex<FontCache>,
    ) -> Self {
        AppLayoutEnsurer {
            render: render as *mut RenderPipeline,
            document: Rc::as_ptr(document),
            ctx,
            fonts: fonts as *const Mutex<FontCache>,
            _thread_pinned: PhantomData,
        }
    }
}

/// Install `ensurer` as the web-API context's [`LayoutEnsurer`] slot,
/// returning a guard that restores the previous slot on drop (usually
/// none). For the initial-load script phase; pump-time re-entries use
/// [`Tab::reenter_page`] instead.
pub(crate) fn install_layout_ensurer(
    ctx: u32,
    ensurer: Rc<dyn LayoutEnsurer>,
) -> LayoutEnsurerGuard {
    let prev = bowser_js::layout_ensurer(ctx);
    bowser_js::set_layout_ensurer(ctx, Some(ensurer));
    LayoutEnsurerGuard { ctx, prev }
}

impl Drop for LayoutEnsurerGuard {
    fn drop(&mut self) {
        bowser_js::set_layout_ensurer(self.ctx, self.prev.take());
    }
}

// ---------------------------------------------------------------------------
// Tab
// ---------------------------------------------------------------------------

/// One browser tab's navigational and render state.
///
/// Session history is a `Vec<Url>` plus an index into it; forward entries
/// are truncated whenever a new navigation commits. The rendered stages
/// (`layout_root`, `display_list`) are caches owned by the paint/window
/// workers and are only *invalidated* from this module.
///
/// (`LayoutBox` has no `Debug` impl, so neither does `Tab`.)
pub struct Tab {
    /// Session history, oldest first. See [`SessionHistory`].
    pub history: SessionHistory,
    /// The tab's installed document and its JS realm, if one has been
    /// loaded. The field is private and [`Tab::install_document`] (crate
    /// visible) is its only writer: it swaps the bundle atomically and
    /// panics unless the realm's web API bindings handle aliases the
    /// shared document handle, so a mis-paired bundle is rejected at
    /// install instead of diverging silently later.
    installed: Option<InstalledDocument>,
    /// The render pipeline: style/layout/scroll-tree/publish state.
    /// [`Tab`] derefs to [`RenderPipeline`] (`Deref`/`DerefMut`), so
    /// `tab.dirty`-style accesses keep working; code that needs the
    /// disjoint `&mut RenderPipeline` / `&mut installed` split for a
    /// page re-entry names `tab.render` explicitly.
    pub render: RenderPipeline,
    /// Per-`<img>` late-load fetch bookkeeping: the request URL and
    /// generation of the most recent fetch started per node, plus which
    /// generation last completed. Written by
    /// [`Tab::pending_late_images`] when a fetch starts and by
    /// [`Tab::note_image_loaded`] when it completes; cleared on every
    /// full navigation by [`Tab::install_document`], since a new
    /// document mints new [`NodeId`]s. A node with an entry in
    /// [`Tab::image_intrinsics`] but none here is a load-time image the
    /// driver has not scanned yet — the first scan seeds it as a
    /// completed generation 0 instead of refetching.
    pub image_loads: HashMap<NodeId, ImageLoad>,
    /// DOM nodes whose image load completed, shared with the page's JS
    /// web APIs (attached at document install via
    /// [`bowser_js::share_completed_image_loads`]): backs the
    /// `HTMLImageElement.complete` getter. Cleared on every full
    /// navigation by [`Tab::install_document`], since a new document
    /// mints new [`NodeId`]s.
    ///
    /// NEVER reassign this field after [`Tab::new`], for the same reason
    /// as [`Tab::scroll`]: the bindings alias this exact allocation
    /// (shared at document install).
    completed_image_loads: Rc<RefCell<HashSet<NodeId>>>,
    /// Cached display list; rebuilt by the window worker when
    /// [`DirtyFlags::paint`] is set.
    pub display_list: Option<DisplayList>,
    /// Defer the window `load` event until after the first layout
    /// publishes (headless/WPT driver only): set before
    /// [`crate::navigation::install_loaded_document`], which then
    /// records `pending_load` instead of dispatching. The headless
    /// loader fires it after the first `refresh`, so `load` handlers
    /// observe a published layout tree (WPT scrollable-overflow).
    /// Windowed installs leave this false and keep the immediate
    /// dispatch.
    pub(crate) defer_load: bool,
    /// A deferred window `load` event waiting for the first layout.
    /// Fired by the headless loader after `refresh`; cleared then.
    pub(crate) pending_load: bool,
    /// Document title (first `<title>` element's text), or empty.
    pub title: String,
}

impl std::ops::Deref for Tab {
    type Target = RenderPipeline;

    /// The render pipeline: `tab.dirty`, `tab.layout_root`,
    /// `tab.viewport`, ... resolve through this. The deref exists so
    /// the pipeline split does not churn every access site; code that
    /// needs the disjoint `&mut RenderPipeline` / `&mut installed`
    /// split for a page re-entry names `tab.render` explicitly.
    fn deref(&self) -> &RenderPipeline {
        &self.render
    }
}

impl std::ops::DerefMut for Tab {
    fn deref_mut(&mut self) -> &mut RenderPipeline {
        &mut self.render
    }
}

impl Tab {
    /// A fresh tab with no history and no document.
    pub fn new(viewport: Size) -> Self {
        Tab {
            history: SessionHistory::new(),
            installed: None,
            render: RenderPipeline::new(viewport),
            image_loads: HashMap::new(),
            completed_image_loads: Rc::new(RefCell::new(HashSet::new())),
            display_list: None,
            defer_load: false,
            pending_load: false,
            title: String::new(),
        }
    }

    /// The current session-history entry, if any navigation has committed.
    pub fn current_url(&self) -> Option<&Url> {
        self.history.current()
    }

    /// Whether a document is installed.
    pub fn has_document(&self) -> bool {
        self.installed.is_some()
    }

    /// The installed document bundle, if one has been loaded.
    pub fn installed(&self) -> Option<&InstalledDocument> {
        self.installed.as_ref()
    }

    /// The installed document bundle, mutably.
    pub fn installed_mut(&mut self) -> Option<&mut InstalledDocument> {
        self.installed.as_mut()
    }

    /// The installed document's JS realm, if one has been loaded.
    pub fn page_vm(&self) -> Option<&PageVm> {
        self.installed.as_ref().map(InstalledDocument::page_vm)
    }

    /// The installed document's JS realm, mutably.
    pub fn page_vm_mut(&mut self) -> Option<&mut PageVm> {
        self.installed.as_mut().map(InstalledDocument::page_vm_mut)
    }

    /// Borrow the shared document for reading: the renderer's view of
    /// the tree. `None` when no document is loaded.
    pub fn document_ref(&self) -> Option<Ref<'_, Document>> {
        self.installed
            .as_ref()
            .map(|installed| installed.document.borrow())
    }

    /// The layout tree with nested scroll offsets baked in, for
    /// display-list builds. `None` exactly when there is no layout yet.
    pub fn paint_root(&self) -> Option<&LayoutBox> {
        self.render.paint_root.as_ref()
    }

    /// The tab's viewport scroll state, shared with the page's JS web
    /// APIs. Returns the same `Rc<ScrollState>` allocation the bindings
    /// alias (shared at document install): callers may clone it or
    /// mutate the cells in place, but the accessor hands out `&Rc`, so
    /// the allocation itself cannot be replaced from outside this
    /// module — replacing it would silently fork script-visible scroll
    /// state from the renderer/wheel state.
    pub fn scroll(&self) -> &Rc<ScrollState> {
        self.render.scroll()
    }

    /// The per-element scroll states for the tab's nested scroll
    /// containers, shared with the page's JS web APIs. Same aliasing
    /// contract as [`Tab::scroll`]: callers may clone or mutate in
    /// place, but cannot replace the allocation.
    pub fn element_scroll(&self) -> &Rc<ElementScrollStates> {
        self.render.element_scroll()
    }

    /// The completed-image-load set, shared with the page's JS web APIs.
    /// Same aliasing contract as [`Tab::scroll`]: callers may clone or
    /// mutate in place, but cannot replace the allocation.
    pub fn completed_image_loads(&self) -> &Rc<RefCell<HashSet<NodeId>>> {
        &self.completed_image_loads
    }

    /// Fold the shared document's recorded mutation dirt into the tab's
    /// stage flags, draining the document's report so each mutation is
    /// accounted exactly once. Any dirtied subtree needs layout+paint;
    /// only style-affecting dirt (structural changes, attribute writes)
    /// additionally needs the style stage — a text-only edit leaves
    /// [`DirtyFlags::style`] clear, so selector matching is not
    /// re-run for it.
    ///
    /// Call after anything that can mutate the DOM outside navigation
    /// (script runs, timer callbacks, event handlers) and before the
    /// window worker's `refresh`, which re-runs exactly the stages whose
    /// flags are set.
    pub fn sync_dirty_flags(&mut self) {
        // Disjoint split: the render pipeline takes `&mut`, the
        // installed bundle only `&` (its document cell is borrowed
        // inside).
        let (render, installed) = (&mut self.render, &self.installed);
        let Some(installed) = installed.as_ref() else {
            return;
        };
        render.sync_dirty_flags(installed.document());
    }

    /// Drop the retained dirty report: called by `refresh` once the
    /// layout stage has serviced the dirt, and on document install.
    pub(crate) fn clear_dirty_report(&mut self) {
        self.render.clear_dirty_report();
    }

    /// Mark layout dirty for a reason that is *not* a retained DOM
    /// [`bowser_dom::DirtyReport`] subtree: a resize, a sticky
    /// re-bake after scroll, a late image's intrinsic size arriving,
    /// a full invalidation. Such dirt can move *every* box (the whole
    /// tree was laid out at other geometry), so no retained subtree
    /// report may be served incrementally against it: this sets
    /// [`DirtyFlags::layout`], forces the full fallback, and drops
    /// the retained subtrees, so a same-turn text edit cannot let a
    /// partial splice silently service and discard this dirt (wave
    /// 51 rework F1). Every direct non-report writer of
    /// `dirty.layout` in `bowser_app` routes through here; the only
    /// exceptions are [`Tab::sync_dirty_flags`] (the report producer,
    /// which sets the retained report it is about to service) and
    /// test fixtures.
    pub fn mark_layout_dirty_full(&mut self) {
        self.render.mark_layout_dirty_full();
    }

    /// Disjoint `&mut` split of the installed bundle and the render
    /// pipeline, for page re-entry (eval/pump): script natives may force
    /// a synchronous reflow through the [`RenderPipeline`] while the
    /// worker holds the installed bundle's page VM. The borrow checker
    /// proves the two borrows disjoint — the fields are split before any
    /// reborrow, so no whole-`Tab` borrow ever exists across the call.
    pub(crate) fn split_for_reentry(
        &mut self,
    ) -> (Option<&mut InstalledDocument>, &mut RenderPipeline) {
        let (installed, render) = (&mut self.installed, &mut self.render);
        (installed.as_mut(), render)
    }

    /// Begin a page re-entry (eval/pump): installs the forced-sync-layout
    /// slot ([`bowser_js::LayoutEnsurer`]) for the installed document's
    /// web-API context and hands back the page VM. The returned guard
    /// must stay live across the re-entry; dropping it restores the
    /// previous slot (usually none).
    ///
    /// Returns `None` when no document is installed: there is no script
    /// to re-enter. The slot itself is installed only when the realm
    /// has a web-API context; the VM is returned (for timer pumping)
    /// either way.
    ///
    /// Runs the [`InstalledDocument::assert_no_live_borrow`] reentrancy
    /// gate: natives re-borrow the document through the registry `Weak`,
    /// so no `Ref`/`RefMut` may be alive across the re-entry.
    pub(crate) fn reenter_page(
        &mut self,
        fonts: &Mutex<FontCache>,
    ) -> Option<(Option<LayoutEnsurerGuard>, &mut PageVm)> {
        // Disjoint split first: `&mut RenderPipeline` for the slot,
        // `&mut InstalledDocument` for the VM. The slot holds only raw
        // pointers (no borrows), so the borrows below end before the
        // page VM is reborrowed for the caller.
        let (installed, render) = self.split_for_reentry();
        let installed: &mut InstalledDocument = installed?;
        InstalledDocument::assert_no_live_borrow(installed.document());
        let ctx = installed.page_vm().web_api_context();
        let guard = ctx.map(|ctx| {
            let ensurer = Rc::new(AppLayoutEnsurer {
                render: render as *mut RenderPipeline,
                document: installed.document() as *const RefCell<Document>,
                ctx,
                fonts: fonts as *const Mutex<FontCache>,
                _thread_pinned: PhantomData,
            });
            let prev = bowser_js::layout_ensurer(ctx);
            bowser_js::set_layout_ensurer(ctx, Some(ensurer));
            LayoutEnsurerGuard { ctx, prev }
        });
        let page_vm = installed.page_vm_mut();
        Some((guard, page_vm))
    }

    /// Rebuild the scroll state after a manual layout: the [`Tab`]-shaped
    /// entry point over [`crate::scroll::rebuild_after_layout`], for
    /// tests and callers that hold `&mut Tab` (not the disjoint split).
    /// No-op without an installed document.
    #[cfg(test)]
    pub(crate) fn rebuild_scroll_after_layout(&mut self) {
        let (render, installed) = (&mut self.render, &self.installed);
        let Some(installed) = installed.as_ref() else {
            return;
        };
        crate::scroll::rebuild_after_layout(render, installed.document());
    }

    /// Driver entry point for late image loads: scan the installed
    /// document for `<img>` elements needing a (re)fetch, and record
    /// each fetch's request URL and generation in [`Tab::image_loads`].
    ///
    /// An `<img>` needs a fetch when it has no bookkeeping entry and no
    /// entry in [`Tab::image_intrinsics`] — a genuinely new late image,
    /// or a load-time image whose fetch failed (the failure is retried
    /// once here, then recorded) — or when its `src` resolved against
    /// the document base differs from the recorded request URL, i.e. the
    /// `src` attribute (or `<base href>`) changed since the last fetch.
    /// In particular an already-loaded `<img>` whose `src` changes IS
    /// refetched. An image with an intrinsics entry but no bookkeeping
    /// entry is a load-time success the driver has not scanned yet: it
    /// is seeded as a completed generation 0, not refetched.
    /// Bookkeeping for nodes no longer in the document is dropped, so
    /// the map stays proportional to the live tree.
    ///
    /// Each returned [`PendingLateImage`] carries the generation the
    /// driver must hand back to [`Tab::note_image_loaded`] with the
    /// completion. This is the whole driver trigger: no other caller
    /// starts late-image fetches.
    ///
    /// Call on the tab-owning thread only: [`Tab`] is `!Send` (it holds
    /// `Rc`-shared document and scroll state). The returned URLs are
    /// safe to fetch on worker threads via
    /// [`crate::loader::fetch_late_image`].
    pub fn pending_late_images(&mut self) -> Vec<PendingLateImage> {
        let Some(installed) = self.installed.as_ref() else {
            return Vec::new();
        };
        let document = installed.document.borrow();
        let base: Option<Url> = self
            .history
            .current()
            .map(|url| effective_base_url(&document, url));
        // Every <img> currently in the document: (node, raw src) pairs
        // for the fetch check, plus the full node set so bookkeeping
        // for detached nodes can be dropped below.
        let mut with_src: Vec<(NodeId, String)> = Vec::new();
        let mut img_nodes: HashSet<NodeId> = HashSet::new();
        for node in document.descendants(document.root()) {
            let NodeKind::Element(data) = document.kind(node) else {
                continue;
            };
            if data.tag_name != "img" {
                continue;
            }
            img_nodes.insert(node);
            if let Some(src) = document.get_attribute(node, "src") {
                with_src.push((node, src.to_owned()));
            }
        }
        drop(document);

        let mut out = Vec::new();
        for (node, src) in with_src {
            let resolved = match &base {
                Some(base) => bowser_url::parse(&src, Some(base)),
                None => bowser_url::parse(&src, None),
            };
            let Ok(url) = resolved else {
                // Unresolvable src: the loader could not have fetched it
                // either; skip until the attribute becomes valid.
                continue;
            };
            let key = url.serialize();
            match self.image_loads.get(&node) {
                Some(load) if load.url == key => {}
                Some(_) => {
                    // The src (or <base href>) changed since the last
                    // fetch: bump the generation and refetch.
                    let load = self
                        .image_loads
                        .get_mut(&node)
                        .expect("pending_late_images: entry vanished");
                    load.generation += 1;
                    load.url = key;
                    out.push(PendingLateImage {
                        node,
                        url,
                        generation: load.generation,
                    });
                }
                None if self.image_intrinsics.contains_key(&node) => {
                    // Load-time success the driver has not scanned yet:
                    // seed as completed instead of refetching.
                    self.completed_image_loads.borrow_mut().insert(node);
                    self.image_loads.insert(
                        node,
                        ImageLoad {
                            url: key,
                            generation: 0,
                            completed_generation: Some(0),
                        },
                    );
                }
                None => {
                    self.image_loads.insert(
                        node,
                        ImageLoad {
                            url: key,
                            generation: 0,
                            completed_generation: None,
                        },
                    );
                    out.push(PendingLateImage {
                        node,
                        url,
                        generation: 0,
                    });
                }
            }
        }
        // Nodes detached from the document can never paint: drop their
        // bookkeeping so the map stays proportional to the live tree. A
        // reinserted node re-seeds from `image_intrinsics` (same src) or
        // is refetched (no intrinsics entry).
        self.image_loads.retain(|node, _| img_nodes.contains(node));
        out
    }

    /// Record the outcome of a late image load for `node` — an `<img>`
    /// added to the document after its load phase completed, or one
    /// whose `src` changed after load — and invalidate the render stages
    /// the outcome actually affects, so the window worker's `refresh`
    /// re-runs them.
    ///
    /// `generation` is the generation
    /// [`Tab::pending_late_images`] returned when this fetch started.
    /// The completion is applied only when it matches the node's
    /// recorded generation: a stale completion — a newer fetch has
    /// started since, e.g. src A→B→A with B's slow fetch landing last —
    /// is dropped, as is a duplicate delivery of an already-applied
    /// completion. Completions for nodes with no bookkeeping entry (no
    /// fetch was started through [`Tab::pending_late_images`]) are
    /// dropped too.
    ///
    /// `intrinsics` is `Some` with the decoded size when the load
    /// succeeded ([`crate::loader::fetch_late_image`]), `None` when it
    /// failed (fetch error, non-2xx, decode failure). Every *new*
    /// successful completion invalidates paint — the pixels are new
    /// even when the decoded size is unchanged, since the last paint
    /// showed a placeholder or older pixels — while layout is
    /// invalidated only when the size actually changed. A failed load
    /// leaves the recorded size untouched (a previously-sized box keeps
    /// its size; an assumed-0x0 box stays 0x0, as load-time failures do)
    /// and invalidates nothing: the failure is still recorded against
    /// its generation, so a later retry starts from a distinguishable
    /// state, but there is no automatic retry.
    ///
    /// Call on the tab-owning thread only ([`Tab`] is `!Send`); the
    /// fetch itself runs on a worker thread.
    ///
    /// Returns true when the completion applied (not stale): the pump's
    /// late-image drain uses this to fire the matching `load`/`error`
    /// DOM event.
    pub fn note_image_loaded(
        &mut self,
        node: NodeId,
        generation: u64,
        intrinsics: Option<IntrinsicDims>,
    ) -> bool {
        let Some(load) = self.image_loads.get_mut(&node) else {
            return false;
        };
        if load.generation != generation {
            return false;
        }
        if load.completed_generation == Some(generation) {
            return false;
        }
        load.completed_generation = Some(generation);
        // The load completed (successfully or not):
        // `HTMLImageElement.complete` flips true either way — a broken
        // image is still "complete" (HTML §4.8.3).
        self.completed_image_loads.borrow_mut().insert(node);
        let Some(dims) = intrinsics else {
            // A failed load changes no geometry and invalidates
            // nothing; the pump fires `error` for it.
            return true;
        };
        // New pixels arrived: repaint even at an unchanged size.
        self.dirty.paint = true;
        if self.image_intrinsics.get(&node).copied() != Some(dims) {
            self.image_intrinsics.insert(node, dims);
            // A changed intrinsic size can move boxes anywhere the
            // image's size propagates: full layout, never a retained
            // subtree splice (F1).
            self.mark_layout_dirty_full();
        }
        true
    }

    /// The outcome of running the current document's `<script>` elements:
    /// counts of executed/skipped scripts plus evaluation failures.
    /// `None` before the first document installs.
    pub fn script_report(&self) -> Option<&ScriptReport> {
        self.installed.as_ref().map(|i| i.page_vm.report())
    }

    /// Install a freshly loaded document as the tab's current document,
    /// together with its page realm: the only writer of the installed
    /// bundle (the field is private, and this method is `pub(crate)`).
    /// The two are always swapped as one atomic pair, so the renderer
    /// (which reads through [`Tab::document_ref`]) and the JS bindings
    /// (which reach the document through a registry `Weak`) always
    /// agree on which document is current. A loud install-time canary
    /// in [`InstalledDocument::new`] asserts the pairing itself: the
    /// realm's registry document must be the document installed with
    /// it.
    ///
    /// The tab holds the sole strong handle to `shared`; the registry
    /// keeps only a `Weak`, so dropping this install (by reinstalling
    /// or closing the tab) cleanly tears down script access with no
    /// second owner to disagree with.
    ///
    /// Caller contract: the caller must not retain the `Rc`. `shared`
    /// is moved in and the tab becomes its sole strong owner — the
    /// hard assertion below enforces `Rc::strong_count(&shared) == 1`
    /// in all build profiles, so a retained alias panics at install
    /// instead of silently outliving the install in release builds.
    /// Drop the local handle before calling (or never create one);
    /// re-acquire out-of-band access afterwards through
    /// [`InstalledDocument::with_document`].
    pub(crate) fn install_document(&mut self, shared: Rc<RefCell<Document>>, page_vm: PageVm) {
        // Sole-ownership gate: the tab must be the only strong owner of
        // the document it installs. A caller-retained alias would
        // survive reinstall/tab close and keep the old tree alive past
        // its install — the second-owner divergence the Weak-only
        // registry exists to prevent. Hard panic in all builds, not
        // debug-only: this gate is the only thing stopping a future
        // caller from smuggling a second strong owner past the
        // Weak-only registry.
        assert_eq!(
            Rc::strong_count(&shared),
            1,
            "install_document requires sole ownership: another strong Rc<RefCell<Document>> exists; \
             drop every alias before installing so the tab is the sole strong owner"
        );
        // The pairing canary lives in [`InstalledDocument::new`], the
        // guarded boundary that builds the bundle; document and realm
        // still swap as one atomic pair here.
        //
        // The tab holds the sole strong handle to `shared`; the registry
        // keeps only a `Weak`, so dropping this install (by reinstalling
        // or closing the tab) cleanly tears down script access with no
        // second owner to disagree with.
        self.installed = Some(InstalledDocument::new(shared, page_vm));
        // A new document mints new NodeIds: drop the previous
        // document's per-image fetch bookkeeping. Load-time images
        // re-seed lazily in pending_late_images from image_intrinsics.
        self.image_loads.clear();
        // The completed-load set names nodes of the previous document
        // too; the bindings alias this exact allocation, so clear it in
        // place rather than replacing it.
        self.completed_image_loads.borrow_mut().clear();
        // Same for the retained dirty report: its roots name nodes of
        // the previous document.
        self.clear_dirty_report();
        // The previous document's layout viewport record names a tree
        // of the previous document too.
        self.last_layout_viewport = None;
    }

    /// Test-only: install a document with a bare (binding-less) realm,
    /// for render-path tests that need no JS. Like
    /// [`install_document`](Self::install_document), this is the only
    /// way to write the bundle from tests — document and realm still
    /// swap atomically.
    #[cfg(test)]
    pub(crate) fn install_test_document(&mut self, shared: Rc<RefCell<Document>>) {
        self.install_document(shared, PageVm::new());
    }

    /// Test-only: drop the installed bundle (document *and* realm
    /// together), returning the tab to the no-document state. The
    /// halves are never separated: there is no way to drop just one.
    #[cfg(test)]
    pub(crate) fn uninstall_test_document(&mut self) {
        self.installed = None;
        self.image_loads.clear();
        self.completed_image_loads.borrow_mut().clear();
    }
}

/// Mark every render stage dirty.
///
/// Refresh contract (for the window worker): the window loop calls its own
/// `refresh(tab)` — owned by the paint/window worker — which runs
/// style → layout → display-list construction whenever the corresponding
/// dirty flag is set, then clears the flags it serviced. Navigation only
/// *sets* flags through this function; it never builds styles, boxes, or
/// display lists itself.
pub fn mark_all_dirty(tab: &mut Tab) {
    tab.dirty.style = true;
    tab.dirty.paint = true;
    // Every stage re-runs from scratch: no retained subtree report can
    // be served incrementally, so the next `refresh` takes the full
    // path (and clears the report once it has). The helper sets the
    // layout flag, forces the fallback, and drops the retained
    // subtrees in one place.
    tab.mark_layout_dirty_full();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn url(path: &str) -> Url {
        bowser_url::parse(&format!("http://example.com{path}"), None).expect("test url")
    }

    #[test]
    fn history_push_truncates_forward_entries() {
        let mut h = SessionHistory::new();
        assert!(h.is_empty());
        assert!(!h.can_go_back());
        assert!(!h.can_go_forward());
        assert_eq!(h.current(), None);

        h.push(url("/a"));
        h.push(url("/b"));
        h.push(url("/c"));
        assert_eq!(h.len(), 3);
        assert_eq!(h.index(), 2);
        assert_eq!(h.current().map(Url::path), Some("/c"));
        assert!(h.can_go_back());
        assert!(!h.can_go_forward());

        h.back();
        h.back();
        assert_eq!(h.current().map(Url::path), Some("/a"));
        assert!(!h.can_go_back());
        assert!(h.can_go_forward());

        // Back at the oldest entry: no-op.
        assert_eq!(h.back(), None);
        assert_eq!(h.index(), 0);

        // A new navigation truncates the forward entries.
        h.push(url("/d"));
        assert_eq!(h.len(), 2);
        assert_eq!(h.index(), 1);
        assert_eq!(h.entry(0).map(Url::path), Some("/a"));
        assert_eq!(h.entry(1).map(Url::path), Some("/d"));
        assert!(!h.can_go_forward());
    }

    #[test]
    fn history_forward_restores_newest_entry() {
        let mut h = SessionHistory::new();
        h.push(url("/a"));
        h.push(url("/b"));
        assert_eq!(h.forward(), None, "at newest: no-op");
        assert_eq!(h.index(), 1);

        h.back();
        assert_eq!(h.forward().map(Url::path), Some("/b"));
        assert_eq!(h.forward(), None);
    }

    #[test]
    fn mark_all_dirty_sets_every_flag() {
        let mut tab = Tab::new(Size { w: 800.0, h: 600.0 });
        assert!(!tab.dirty.style && !tab.dirty.layout && !tab.dirty.paint);
        mark_all_dirty(&mut tab);
        assert!(tab.dirty.style && tab.dirty.layout && tab.dirty.paint);
    }

    fn tab_with_doc(html: &[u8]) -> Tab {
        let (doc, _) = Document::parse_html(html);
        let mut tab = Tab::new(Size { w: 800.0, h: 600.0 });
        tab.install_test_document(Rc::new(RefCell::new(doc)));
        tab.dirty = DirtyFlags::default();
        tab
    }

    #[test]
    fn document_is_shared_not_copied() {
        let tab = tab_with_doc(b"<html><body><p>hi</p></body></html>");
        // A second handle to the same allocation (what the JS bindings
        // hold): mutations through it are visible through the tab's own
        // handle, which is what the renderer reads. Test-only: access
        // goes through the scoped `with_document`, so no strong handle
        // can escape — the closure sees the tab's own cell, and the
        // pointer check proves it.
        let installed = tab.installed().expect("document");
        let p = installed
            .with_document(|cell| {
                assert!(
                    std::ptr::eq(cell, installed.document()),
                    "with_document must hand out the tab's own cell"
                );
                let p = cell
                    .borrow()
                    .get_elements_by_tag_name("p")
                    .item(0)
                    .expect("p");
                cell.borrow_mut().set_attribute(p, "id", "main");
                p
            })
            .expect("tab owns document");
        let seen = tab
            .document_ref()
            .expect("document")
            .get_element_by_id("main");
        assert_eq!(seen, Some(p), "binding mutation visible to renderer");
    }

    #[test]
    #[should_panic(expected = "install_document: page realm's web APIs resolve to a document")]
    fn installed_document_new_panics_on_mispaired_realm() {
        // The pairing canary on `InstalledDocument::new`: a realm built
        // for document A installed alongside document B must die loudly
        // at install time, never diverge silently.
        let (doc_a, _) = Document::parse_html(b"<html><body>a</body></html>");
        let doc_a = Rc::new(RefCell::new(doc_a));
        let realm_for_a =
            PageVm::new_for_document(doc_a.clone(), &url("/a"), "TestBrowser/9.9", &[])
                .expect("web API install");
        let (doc_b, _) = Document::parse_html(b"<html><body>b</body></html>");
        let doc_b = Rc::new(RefCell::new(doc_b));
        let _ = InstalledDocument::new(doc_b, realm_for_a);
    }

    #[test]
    #[should_panic(expected = "install_document requires sole ownership")]
    fn install_document_panics_on_retained_alias() {
        // The sole-ownership gate is a hard runtime check in all
        // builds, not debug-only: a caller that keeps a second strong
        // handle must die at install, never diverge silently in
        // release.
        let mut tab = Tab::new(Size { w: 800.0, h: 600.0 });
        let (doc, _) = Document::parse_html(b"<html><body><p>x</p></body></html>");
        let shared = Rc::new(RefCell::new(doc));
        let retained = Rc::clone(&shared);
        tab.install_test_document(shared);
        drop(retained);
    }

    #[test]
    #[should_panic(expected = "document borrow held across page re-entry")]
    fn reentrancy_gate_trips_on_live_borrow() {
        // The dispatch-entry reentrancy gate is a hard runtime check in
        // all builds, not debug-only: a `RefMut` held across a page
        // re-entry must die at the dispatch boundary with the invariant
        // named, never fall through to the raw `RefCell` panic deep
        // inside a native in release builds.
        let tab = tab_with_doc(b"<html><body><p>x</p></body></html>");
        let installed = tab.installed().expect("document");
        let _held = installed.document().borrow_mut();
        InstalledDocument::assert_no_live_borrow(installed.document());
    }

    #[test]
    #[should_panic(expected = "document borrow held across page re-entry")]
    fn reentrancy_gate_trips_on_live_shared_borrow() {
        // Same gate, shared-borrow shape: a `Ref` held across a page
        // re-entry breaks any mutating native's `borrow_mut`, so the
        // all-builds gate rejects it too — stricter than the old
        // `debug_assert!(try_borrow().is_ok())`, which only saw
        // exclusive borrows.
        let tab = tab_with_doc(b"<html><body><p>x</p></body></html>");
        let installed = tab.installed().expect("document");
        let _held = installed.document().borrow();
        InstalledDocument::assert_no_live_borrow(installed.document());
    }

    #[test]
    fn sync_dirty_flags_folds_document_dirt() {
        let mut tab = tab_with_doc(b"<html><body><p>hi</p></body></html>");
        // Scoped access: each closure borrows the tab's own cell and no
        // strong handle escapes, so the tab's sole ownership is never
        // contested at an install boundary.

        // Text-only edit: layout+paint re-run, style does not.
        let text = tab
            .installed()
            .expect("document")
            .with_document(|cell| {
                let doc = cell.borrow();
                let p = doc.get_elements_by_tag_name("p").item(0).expect("p");
                doc.first_child(p).expect("text child")
            })
            .expect("tab owns document");
        tab.installed()
            .expect("document")
            .with_document(|cell| cell.borrow_mut().set_text_content(text, "hello"))
            .expect("tab owns document");
        tab.sync_dirty_flags();
        assert!(!tab.dirty.style, "text edit must not re-run style");
        assert!(tab.dirty.layout && tab.dirty.paint);

        // A class change can affect matching: all three stages.
        tab.dirty = DirtyFlags::default();
        let p = tab
            .installed()
            .expect("document")
            .with_document(|cell| {
                cell.borrow()
                    .get_elements_by_tag_name("p")
                    .item(0)
                    .expect("p")
            })
            .expect("tab owns document");
        tab.installed()
            .expect("document")
            .with_document(|cell| cell.borrow_mut().set_attribute(p, "class", "big"))
            .expect("tab owns document");
        tab.sync_dirty_flags();
        assert!(tab.dirty.style && tab.dirty.layout && tab.dirty.paint);

        // No mutations: flags untouched, and the drain is one-shot.
        tab.dirty = DirtyFlags::default();
        tab.sync_dirty_flags();
        assert!(!tab.dirty.style && !tab.dirty.layout && !tab.dirty.paint);
    }

    #[test]
    fn sync_dirty_flags_without_document_is_noop() {
        let mut tab = Tab::new(Size { w: 800.0, h: 600.0 });
        tab.sync_dirty_flags();
        assert!(!tab.dirty.style && !tab.dirty.layout && !tab.dirty.paint);
    }

    fn tab_node_by_id(tab: &Tab, id: &str) -> NodeId {
        tab.installed()
            .expect("document")
            .with_document(|cell| cell.borrow().get_element_by_id(id).expect("element by id"))
            .expect("tab owns document")
    }

    fn tab_text_in(tab: &Tab, id: &str) -> NodeId {
        let el = tab_node_by_id(tab, id);
        tab.installed()
            .expect("document")
            .with_document(|cell| {
                let doc = cell.borrow();
                // Descend element -> <p> -> text (the fixture's shape).
                let p = doc.first_child(el).expect("p child");
                doc.first_child(p).expect("text child")
            })
            .expect("tab owns document")
    }

    fn tab_mutate(tab: &Tab, f: impl FnOnce(&mut Document)) {
        tab.installed()
            .expect("document")
            .with_document(|cell| f(&mut cell.borrow_mut()))
            .expect("tab owns document");
    }

    #[test]
    fn coalesce_dirty_subtrees_drops_descendants_and_duplicates() {
        let (doc, _) = Document::parse_html(
            b"<html><body><div id=\"a\"><p id=\"b\">x</p></div><p id=\"c\">y</p></body></html>",
        );
        let a = doc.get_element_by_id("a").expect("a");
        let b = doc.get_element_by_id("b").expect("b");
        let c = doc.get_element_by_id("c").expect("c");
        let text_b = doc.first_child(b).expect("text in b");
        // Parent + child + grandchild text + duplicate -> parent only,
        // disjoint sibling kept, order deterministic by arena index.
        assert_eq!(
            coalesce_dirty_subtrees(&doc, &[text_b, b, a, a, c]),
            vec![a, c]
        );
        assert_eq!(coalesce_dirty_subtrees(&doc, &[c, b]), vec![b, c]);
        assert!(coalesce_dirty_subtrees(&doc, &[]).is_empty());
    }

    #[test]
    fn sync_dirty_flags_retains_coalesced_report_across_syncs() {
        let mut tab = tab_with_doc(
            b"<html><body><div id=\"a\"><p>x</p></div><div id=\"b\"><p>y</p></div></body></html>",
        );
        // A text edit in #a: the retained report names the text node.
        let text_a = tab_text_in(&tab, "a");
        tab_mutate(&tab, |doc| doc.set_text_content(text_a, "changed"));
        tab.sync_dirty_flags();
        assert_eq!(tab.dirty_subtrees, vec![text_a]);
        assert!(!tab.dirty_full_fallback);

        // A second, disjoint edit accumulates instead of replacing.
        let b = tab_node_by_id(&tab, "b");
        tab_mutate(&tab, |doc| doc.set_attribute(b, "class", "z"));
        tab.sync_dirty_flags();
        assert_eq!(tab.dirty_subtrees, vec![text_a, b]);
        assert!(!tab.dirty_full_fallback);

        // Mutating the ancestor of the retained text root absorbs it.
        let a = tab_node_by_id(&tab, "a");
        tab_mutate(&tab, |doc| doc.set_attribute(a, "class", "z"));
        tab.sync_dirty_flags();
        assert_eq!(tab.dirty_subtrees, vec![a, b]);
        assert!(!tab.dirty_full_fallback);

        // A sync with no new mutations leaves the retained report be.
        tab.sync_dirty_flags();
        assert_eq!(tab.dirty_subtrees, vec![a, b]);
    }

    #[test]
    fn mark_layout_dirty_full_drops_retained_report() {
        // F1: non-DOM layout dirt arriving in the same turn as a
        // retained text subtree must force the full path and drop the
        // retained report, not be silently serviced by a splice.
        let mut tab = tab_with_doc(b"<html><body><div id=\"a\"><p>x</p></div></body></html>");
        let text_a = tab_text_in(&tab, "a");
        tab_mutate(&tab, |doc| doc.set_text_content(text_a, "changed"));
        tab.sync_dirty_flags();
        assert_eq!(tab.dirty_subtrees, vec![text_a]);
        assert!(!tab.dirty_full_fallback);
        tab.dirty = DirtyFlags::default();

        tab.mark_layout_dirty_full();
        assert!(tab.dirty.layout);
        assert!(tab.dirty_full_fallback, "non-report dirt forces full");
        assert!(tab.dirty_subtrees.is_empty(), "retained report dropped");
    }

    #[test]
    fn sync_dirty_flags_full_fallback_on_document_element() {
        let mut tab = tab_with_doc(b"<html><body><p>hi</p></body></html>");
        let root = tab
            .installed()
            .expect("document")
            .with_document(|cell| cell.borrow().document_element().expect("document element"))
            .expect("tab owns document");
        tab_mutate(&tab, |doc| doc.set_attribute(root, "lang", "en"));
        tab.sync_dirty_flags();
        assert_eq!(tab.dirty_subtrees, vec![root]);
        assert!(tab.dirty_full_fallback, "document element dirt is full");
    }

    #[test]
    fn sync_dirty_flags_full_fallback_over_threshold() {
        let mut tab = tab_with_doc(
            b"<html><body><p id=\"p0\">0</p><p id=\"p1\">1</p><p id=\"p2\">2</p>\
              <p id=\"p3\">3</p><p id=\"p4\">4</p></body></html>",
        );
        for i in 0..5 {
            let p = tab_node_by_id(&tab, &format!("p{i}"));
            tab_mutate(&tab, |doc| doc.set_attribute(p, "class", "z"));
        }
        tab.sync_dirty_flags();
        assert_eq!(tab.dirty_subtrees.len(), 5, "five disjoint roots");
        assert!(
            tab.dirty_full_fallback,
            "more than {DIRTY_SUBTREE_FULL_THRESHOLD} disjoint roots is full"
        );

        // Exactly at the threshold stays partial-eligible.
        let mut tab = tab_with_doc(
            b"<html><body><p id=\"p0\">0</p><p id=\"p1\">1</p><p id=\"p2\">2</p>\
              <p id=\"p3\">3</p></body></html>",
        );
        for i in 0..4 {
            let p = tab_node_by_id(&tab, &format!("p{i}"));
            tab_mutate(&tab, |doc| doc.set_attribute(p, "class", "z"));
        }
        tab.sync_dirty_flags();
        assert_eq!(tab.dirty_subtrees.len(), 4);
        assert!(!tab.dirty_full_fallback);
    }

    /// Graceful teardown when the tab drops a document: the tab owns
    /// the sole strong document handle, so reinstalling swaps the whole
    /// bundle (document + realm) at once and the old install is fully
    /// released — no surviving second owner, no stale registry handle
    /// handing script a dead tree. This test drives a real native call
    /// against the torn-down context and asserts the host error
    /// surfaces as a JS exception (no panic, no stale tree), then
    /// asserts the registry holds no entry for the old context at all
    /// (removed, not merely unreachable through a dead `Weak`).
    #[test]
    fn install_document_drop_is_graceful_not_divergent() {
        use crate::script::PageVm;
        use bowser_js::web_apis_context_registered;

        fn base() -> Url {
            bowser_url::parse("http://example.com/", None).expect("test url")
        }

        fn install(tab: &mut Tab, html: &[u8]) {
            let (doc, _) = Document::parse_html(html);
            let shared = Rc::new(RefCell::new(doc));
            let page_vm = PageVm::new_for_document(shared.clone(), &base(), "TestBrowser/9.9", &[])
                .expect("web apis");
            // The only writer of the bundle. `install_document` takes
            // sole strong ownership: move the last local handle in
            // (no clone retained), so the tab is the sole strong owner
            // at install. Out-of-band reads go through the tab's scoped
            // `with_document` — the way any out-of-band reader (like the
            // paint pass) gets it — so no strong alias ever exists
            // outside the tab, and reinstalling never trips the
            // sole-ownership gate.
            tab.install_document(shared, page_vm);
        }

        let mut tab = Tab::new(Size { w: 800.0, h: 600.0 });
        assert!(!tab.has_document());
        assert!(tab.installed().is_none());

        install(&mut tab, b"<html><body><p>one</p></body></html>");
        let old_ctx = tab
            .installed()
            .expect("installed")
            .page_vm()
            .web_api_context()
            .expect("web api ctx");
        assert!(tab.has_document());
        // The renderer's handle aliases the tab's installed handle:
        // the scoped closure sees the tab's own cell, and the pointer
        // check proves it — no second strong owner is minted.
        let cell_ptr = tab
            .installed()
            .expect("installed")
            .with_document(|cell| cell as *const RefCell<Document>)
            .expect("tab owns document");
        assert_eq!(
            cell_ptr,
            tab.installed().expect("installed").document() as *const _,
            "with_document must hand out the tab's own cell"
        );
        // The registry serves the live document for this context.
        // Upgrade the registry's own `Weak` and prove it aliases the
        // install; the `Rc` dies at the end of this block and the
        // downgraded `Weak` stays behind as the drop observer for the
        // reinstall below.
        let weak = {
            let live = tab
                .installed_mut()
                .expect("installed")
                .page_vm_mut()
                .vm_mut()
                .web_apis_document(old_ctx)
                .expect("live registry document");
            assert_eq!(
                Rc::as_ptr(&live),
                cell_ptr,
                "registry must serve the installed document"
            );
            Rc::downgrade(&live)
        };
        assert!(tab.page_vm().is_some());
        assert!(tab.page_vm_mut().is_some());
        assert!(
            tab.document_ref()
                .expect("document")
                .get_elements_by_tag_name("p")
                .item(0)
                .is_some()
        );

        // Baseline: a native call works while the install is live. The
        // degradation check below is measured against this.
        assert!(
            tab.installed_mut()
                .expect("installed")
                .page_vm_mut()
                .eval("document.createElement('div')")
                .is_ok(),
            "native call on the live install must succeed"
        );

        // Tear down the registry context while the realm stays alive —
        // the same call `PageVm::Drop` performs on reinstall/tab close —
        // then drive a real native call against the dead context. The
        // host error must surface as a JS exception: no panic, and no
        // stale tree handed to script.
        let page_vm = tab.installed_mut().expect("installed").page_vm_mut();
        page_vm.vm_mut().remove_page_web_apis(old_ctx);
        let err = page_vm
            .eval("document.createElement('div')")
            .expect_err("native call against a torn-down context must throw");
        let msg = err.to_string();
        assert!(
            msg.contains("web api context is gone"),
            "expected a clean host error, got: {msg}"
        );

        // A reinstall swaps the whole bundle at once: the old document
        // and the old realm are both gone, never half-kept. The tab's
        // strong handle was the last one — the registry keeps no second
        // strong owner — so the old tree is fully released. (No alias to
        // drop first: `with_document` never hands one out, so the
        // sole-ownership gate sees exactly the tab's handle.)
        install(&mut tab, b"<html><body><p>two</p></body></html>");
        let cell_ptr2 = tab
            .installed()
            .expect("installed")
            .with_document(|cell| cell as *const RefCell<Document>)
            .expect("tab owns document");
        // Proof of the invariant from the other side: `weak` predates
        // the reinstall and deliberately outlives it — a `Weak`, so the
        // old tree still dies on schedule and this upgrade fails.
        assert!(
            weak.upgrade().is_none(),
            "reinstall must drop the old document and its realm"
        );

        // The torn-down context resolves to nothing: lookups return
        // `None` and a repeated teardown is a harmless no-op. And the
        // registry holds no entry for the old context at all — removed,
        // not leaked (a leaked entry's document `Weak` would also fail
        // to upgrade, so `web_apis_document` alone cannot tell the two
        // apart).
        let new_ctx = tab
            .installed_mut()
            .expect("installed")
            .page_vm_mut()
            .web_api_context()
            .expect("new web api ctx");
        let vm = tab
            .installed_mut()
            .expect("installed")
            .page_vm_mut()
            .vm_mut();
        assert!(vm.web_apis_document(old_ctx).is_none());
        assert!(
            !web_apis_context_registered(old_ctx),
            "old context must be removed from the registry, not leaked"
        );
        assert!(
            web_apis_context_registered(new_ctx),
            "new context must be live in the registry"
        );
        vm.remove_page_web_apis(old_ctx);
        assert!(!web_apis_context_registered(old_ctx));
        // The new bundle is fully live: its context resolves to the new
        // document, so script dispatch on the reinstall works.
        let live2 = vm
            .web_apis_document(new_ctx)
            .expect("new registry document");
        assert_eq!(
            Rc::as_ptr(&live2),
            cell_ptr2,
            "new registry context must serve the new document"
        );
    }
}
