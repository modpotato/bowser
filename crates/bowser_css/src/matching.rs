//! Selector matching: right-to-left complex-selector matching against
//! the DOM.
//!
//! [`matches`] answers whether a parsed [`Selector`] matches a node,
//! consulting only structural information. [`matches_with_state`] does the
//! same but additionally evaluates the interactive pseudo-classes
//! (`:hover`, `:active`, `:focus`, `:link`, `:visited`) against an
//! [`ElementState`] supplied by the caller.
//!
//! Pseudo-element selectors (`div::before`) never match a real element:
//! their declarations belong to a pseudo box, not the originating
//! element's own style. Real pseudo-element box layout comes later; until
//! then the declarations apply nowhere rather than contaminating the
//! originating element's computed style.
//!
//! [`RuleIndex`] pre-filters a stylesheet's selectors by their rightmost
//! compound's tag/id/classes so the cascade tests only selectors that can
//! possibly match an element; full [`matches_with_state`] still decides
//! each candidate.
//!
//! ## `:has()` denial-of-service bounds
//!
//! The relative-selector search costs O(k x subtree^2) per anchor for k
//! compounds (see [`matches_relative`]). k is *not* left to the
//! stylesheet: each `:has()` relative selector is capped at
//! 16 compounds at parse time (see `MAX_HAS_RELATIVE_COMPOUNDS` in
//! `selector.rs`; over-cap groups are dropped as invalid groups of the
//! forgiving list), and `:has()` cannot nest inside `:has()` at all.
//! The failed-state memo is therefore intentionally *uncapped*: a memo
//! cap was tried and reverted because evicting recorded failures
//! re-expands them exponentially, breaking the polynomial guarantee.
//! With k bounded by a small constant the uncapped memo stays correct
//! and small (at most 16 x subtree entries per relative selector).
//!
//! The memo is shared across every anchor evaluated in one style pass
//! (see [`MatchScratch::has_failed`]), so one `:has()` rule over N
//! anchors costs O(16 x subtree^2 + N x subtree) total — the first
//! anchor explores, every later anchor replays memo hits — instead of
//! the Σ O(16 x subtree^2) the cascade would otherwise pay. Its
//! lifetime is scoped to one style-resolution pass: a fresh scratch per
//! `Styler`, cleared on every interactive-state epoch bump, cleared per
//! one-shot call.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::hash::{BuildHasher, Hasher};
use std::rc::Rc;

/// FxHash (the rustc-hash algorithm) implemented locally: the workspace
/// has no fxhash-style dependency for `bowser_css` and the mission is
/// std-only, so the ~40 lines live here instead of in `Cargo.toml`.
/// This is the crate's single copy: `cascade.rs` imports the
/// `pub(crate)` aliases below rather than duplicating them.
///
/// FxHash is a fast non-cryptographic hash tuned for small keys. Every
/// hot map in this file keys by integers (`NameAtom`/`u32`,
/// `NodeId`, `(u32, NodeId)`) or short strings (ids, classes,
/// [`TypeKey`]), exactly the workload FxHash beats SipHash-1-3 on —
/// SipHash's per-hash random state and multi-round mixing dominate the
/// cost of hashing a few bytes.
pub(crate) const FX_SEED: usize = 0x51_7c_c1_b7_27_22_0a_95;

#[derive(Clone, Debug)]
pub(crate) struct FxHasher {
    hash: usize,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, x: usize) {
        self.hash = (self.hash.rotate_left(5) ^ x).wrapping_mul(FX_SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // usize-at-a-time, then a zero-padded tail chunk.
        let (chunks, tail) = bytes.as_chunks::<{ std::mem::size_of::<usize>() }>();
        for c in chunks {
            self.add(usize::from_ne_bytes(*c));
        }
        if !tail.is_empty() {
            let mut buf = [0u8; std::mem::size_of::<usize>()];
            buf[..tail.len()].copy_from_slice(tail);
            self.add(usize::from_ne_bytes(buf));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as usize);
    }
    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(i as usize);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as usize);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        #[cfg(target_pointer_width = "64")]
        self.add(i as usize);
        #[cfg(target_pointer_width = "32")]
        {
            self.add(i as usize);
            self.add((i >> 32) as usize);
        }
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i);
    }
    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.add(i as usize);
    }
    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.add(i as usize);
    }
    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.add(i as usize);
    }
    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn write_isize(&mut self, i: isize) {
        self.add(i as usize);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash as u64
    }
}

/// [`BuildHasher`] for [`FxHasher`]: zero per-map state, so every map
/// lookup skips the random-state setup SipHash pays.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FxBuildHasher;

impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;

    #[inline]
    fn build_hasher(&self) -> FxHasher {
        FxHasher { hash: 0 }
    }
}

/// `HashMap` with the local FxHash: the hot maps in this file
/// ([`RuleIndex`] buckets, [`SiblingCache`], the failed-state memo), and
/// the cascade's per-element maps (imported by `crate::cascade`).
pub(crate) type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;
/// `HashSet` with the local FxHash (imported by `crate::cascade`).
pub(crate) type FxHashSet<T> = HashSet<T, FxBuildHasher>;

use bowser_dom::{Document, ElementData, NameAtom, Namespace, NodeId, NodeKind};

use crate::parse::Stylesheet;
use crate::selector::{
    AttrOp, AttrSelector, Combinator, CompoundSelector, NamespaceConstraint, PseudoClass,
    RelativeSelector, Selector, resolve_namespace_prefix,
};

/// User-interaction state of an element, consulted when matching the
/// interactive pseudo-classes (`:hover`, `:active`, `:focus`, `:link`,
/// `:visited`).
///
/// The renderer fills this in per element (hit-testing for `:hover`,
/// mouse-button capture for `:active`, focus navigation for `:focus`,
/// history lookup for `:visited`) and calls [`matches_with_state`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ElementState {
    /// Whether the pointing device is currently over the element.
    pub hover: bool,
    /// Whether the element is being activated (mouse button held down on it).
    pub active: bool,
    /// Whether the element currently has keyboard/focus navigation focus.
    pub focus: bool,
    /// Whether the link's URL has been visited (see the privacy caveat on
    /// [`matches_with_state`]).
    pub visited: bool,
}

/// Whether `selector` matches `node` in `doc`.
///
/// Matching runs right-to-left: the rightmost compound must match the node
/// itself, then each combinator walks to the matching sibling/ancestor.
///
/// Dynamic pseudo-classes (`:hover`, `:active`, `:focus`, `:link`,
/// `:visited`) never match: this is [`matches_with_state`] with
/// [`ElementState::default()`].
///
/// One-shot calls share a per-thread warm [`MatchScratch`]: the backing
/// allocations (failed-state memo, backtracking stacks) happen once per
/// thread and amortize to zero per call. Callers making many matches
/// against the same document can own the scratch explicitly with
/// [`matches_with_scratch`] instead.
///
/// Not re-entrant: calling a one-shot entry while another one-shot match
/// is already running on this thread panics. The matcher itself never
/// re-enters — nested `:has()` / `:is()` / `:not()` arguments reuse the
/// borrowed scratch through the take/restore protocol, not through this
/// cell — and the one-shot entries take no caller callbacks, so there is
/// no path back in.
pub fn matches(selector: &Selector, doc: &Document, node: NodeId) -> bool {
    matches_with_state(selector, doc, node, &ElementState::default())
}

thread_local! {
    /// Warm scratch shared by the one-shot entries on this thread. Warmed
    /// once (see [`MatchScratch::warm_up`]); each thread gets its own, so
    /// parallel test threads never share match state.
    static SHARED_SCRATCH: RefCell<MatchScratch> = RefCell::new(MatchScratch::new());
}

/// Whether `selector` matches `node` in `doc`, evaluating interactive
/// pseudo-classes against `state`.
///
/// `:hover` matches when `state.hover` is set, `:active` when
/// `state.active` is set, `:focus` when `state.focus` is set, `:visited`
/// when `state.visited` is set, and `:link` when `state.visited` is *not*
/// set. All other selector parts behave exactly as in [`matches`]; the
/// state also applies inside `:not()` (so `:not(:hover)` matches when the
/// element is not hovered).
///
/// # Privacy caveat
///
/// Per CSS privacy restrictions, `:visited` styling must never be
/// observable through `getComputedStyle` or any other script-visible
/// channel (this would leak browsing history). Callers rendering
/// `:visited` rules must compute the visited and unvisited styles in a
/// way that lies to `getComputedStyle` — typically by returning the
/// unvisited style for script queries. This function reports the match
/// truthfully; enforcing the privacy boundary is the JS engine's job.
pub fn matches_with_state(
    selector: &Selector,
    doc: &Document,
    node: NodeId,
    state: &ElementState,
) -> bool {
    let state = *state;
    // One-shot call: no shared cache, so the structural pseudo-classes
    // use the direct sibling walk (zero allocation, exactly the old
    // behavior). Repeat callers (the cascade's `Styler`) pass a shared
    // `SiblingCache` through `matches_cascade` for amortized O(1).
    //
    // The match scratch is the thread's shared warm scratch (see
    // [`matches`]): backing allocations happen once per thread and
    // amortize to zero per call. Repeat callers that want to own the
    // scratch explicitly use [`matches_with_scratch`] instead.
    SHARED_SCRATCH.with(|s| {
        let mut scratch = s.borrow_mut();
        scratch.warm_up();
        // One-shot call: a single anchor only, and the thread-local
        // scratch spans arbitrary selectors and documents across calls —
        // so the cross-anchor `:has()` memo is cleared here rather than
        // shared (see `MatchScratch::has_failed`).
        scratch.clear_has_memo();
        match_complex(selector, doc, node, &|_| state, None, &mut scratch, None)
    })
}

/// Whether `selector` matches `node` in `doc`, reusing a caller-owned
/// [`MatchScratch`].
///
/// The one-shot [`matches`] equivalent for repeat callers: pass one
/// scratch across many matches and the backing allocations (failed-state
/// memo, backtracking stacks) amortize to zero per call after the first
/// sizes them. Dynamic pseudo-classes never match, exactly like
/// [`matches`]. For the stateful variant see
/// [`matches_with_state_and_scratch`].
pub fn matches_with_scratch(
    selector: &Selector,
    doc: &Document,
    node: NodeId,
    scratch: &mut MatchScratch,
) -> bool {
    matches_with_state_and_scratch(selector, doc, node, &ElementState::default(), scratch)
}

/// Whether `selector` matches `node` in `doc`, evaluating interactive
/// pseudo-classes against `state` and reusing a caller-owned
/// [`MatchScratch`].
///
/// The [`matches_with_state`] equivalent for repeat callers: the scratch
/// is warmed on first use (see [`MatchScratch::warm_up`]) and its buffers
/// are then reused across calls, so only the first call sizes the
/// backing allocations.
///
/// The cross-anchor `:has()` failure memo is cleared on entry: unlike
/// the cascade's pass-owned scratch, a caller-owned scratch may span
/// different selectors and documents across calls, where the
/// address-keyed memo entries would be unsound. Per-call memoization
/// within the match is unaffected.
pub fn matches_with_state_and_scratch(
    selector: &Selector,
    doc: &Document,
    node: NodeId,
    state: &ElementState,
    scratch: &mut MatchScratch,
) -> bool {
    let state = *state;
    scratch.warm_up();
    scratch.clear_has_memo();
    match_complex(selector, doc, node, &|_| state, None, scratch, None)
}

/// Whether `selector` matches `node` in `doc`, resolving each tested
/// element's interactive state through `state_for`.
///
/// This is the cascade's entry point: unlike [`matches_with_state`],
/// which applies one [`ElementState`] to every element in the match, the
/// state is looked up per element, so `.parent:hover .child` tests the
/// *parent's* hover state rather than the child's. Ancestor, sibling, and
/// `:has()`-descendant compounds each see their own element's state.
///
/// `scratch` is the matcher's reusable scratch space (see
/// [`MatchScratch`]): the cascade's `Styler` owns one for the whole style
/// pass so the backing allocations amortize to zero per element. It is
/// warmed on first use here, so the pre-sized buffers are ready before
/// the first candidate match.
///
/// Unlike the one-shot entries, the cross-anchor `:has()` failure memo
/// ([`MatchScratch::has_failed`]) is *not* cleared here: it persists
/// across `matches_cascade` calls so one `:has()` rule's failures are
/// computed once per style pass and replayed for every anchor, instead
/// of re-explored per anchor. The `Styler` scopes that lifetime — a
/// fresh scratch per pass, cleared on every interactive-state epoch
/// bump — so entries never outlive the document, stylesheets, and
/// element states they were recorded under.
///
/// `pre` carries the subject element's precomputed `id`/`class` values
/// ([`SubjectAttrs`]); the cascade computes them once per element in
/// `candidates()`. Pass `None` when there is no precompute (every other
/// entry point): the id/class checks fall back to the lazy
/// attribute-list scan.
pub(crate) fn matches_cascade(
    selector: &Selector,
    doc: &Document,
    node: NodeId,
    state_for: &dyn Fn(NodeId) -> ElementState,
    sib: Option<&SiblingCache>,
    scratch: &mut MatchScratch,
    pre: Option<SubjectAttrs<'_>>,
) -> bool {
    scratch.warm_up();
    match_complex(selector, doc, node, state_for, sib, scratch, pre)
}

/// Whether `selector` targets the `pseudo` pseudo-element of `node`:
/// the selector's rightmost compound carries that pseudo-element and
/// the rest of the selector matches `node` (the originating element).
/// A bare `::pseudo` (universal originating selector) matches any
/// element. Used by the cascade to style pseudo boxes such as
/// `::marker`, which are not real DOM nodes.
pub(crate) fn matches_pseudo_element(
    selector: &Selector,
    pseudo: &str,
    doc: &Document,
    node: NodeId,
    state_for: &dyn Fn(NodeId) -> ElementState,
    sib: Option<&SiblingCache>,
    scratch: &mut MatchScratch,
) -> bool {
    let is_target = selector
        .compounds
        .last()
        .is_some_and(|c| c.pseudo_element.as_deref() == Some(pseudo));
    if !is_target {
        return false;
    }
    // The pseudo-element itself is not a real node: strip it and match
    // the originating element. The clone only happens for selectors
    // that actually target this pseudo-element.
    let mut stripped = selector.clone();
    if let Some(c) = stripped.compounds.last_mut() {
        c.pseudo_element = None;
    }
    match_complex(&stripped, doc, node, state_for, sib, scratch, None)
}

/// Reusable scratch space for selector matching: the failed-state memo
/// sets and the two backtracking choice-point stacks, hoisted out of the
/// per-call path.
///
/// `match_complex` used to allocate a fresh `Vec<Frame>` on every
/// multi-compound match and a fresh memo `HashSet` on every long-selector
/// match; `matches_relative` did the same for long relative selectors.
/// Threaded `&mut` from the top-level entry ([`matches`],
/// [`matches_with_state`], [`matches_with_scratch`], or the cascade's
/// `Styler`) through [`matches_cascade`] into `match_complex` /
/// `matches_has` / `matches_relative` / `dfs_relative`, so the backing
/// allocations happen once and are amortized over the scratch's lifetime.
///
/// The one-shot entries ([`matches`], [`matches_with_state`]) share one
/// warm scratch per thread; [`matches_with_scratch`] and
/// [`matches_with_state_and_scratch`] take a caller-owned scratch for
/// repeat callers that want to control its lifetime.
///
/// Each matcher *takes* the backtracking buffers out of the scratch on
/// entry ([`std::mem::take`]) and puts them back on exit: a nested match
/// (a `:has()` / `:is()` / `:not()` argument evaluated while an outer match
/// is mid-backtrack) takes the emptied scratch and uses it as its own, so
/// an ancestor's in-progress state is never clobbered.
///
/// The two memo sets have different lifetimes:
/// * `failed` backs `match_complex`'s failed-state memo. It is taken and
///   *cleared* on entry — its keys are `(compound index, node)` pairs
///   scoped to one complex selector, so entries from a previous match
///   are meaningless here — which keeps the capacity while dropping the
///   stale keys.
/// * `has_failed` backs the `:has()` relative-search memo and is *shared*:
///   it is never cleared by the matcher itself, so one `:has()` rule's
///   failures persist across the per-anchor evaluations of a whole style
///   pass (see the field docs for the soundness argument and the exact
///   lifetime). The one-shot entries clear it per call — a one-shot call
///   evaluates a single anchor — and the cascade clears it at style-pass
///   boundaries.
///
/// [`MatchScratch::new`] is still fully lazy (zero heap): the old
/// laziness contract the allocation tests pin. Every entry point warms
/// the buffers on first use instead (see [`warm_up`]), so the first call
/// pre-sizes them once and later calls reuse the capacity.
#[derive(Debug, Default)]
pub struct MatchScratch {
    /// Failed-state memo for `match_complex`: (compound index, witness
    /// node) pairs scoped to one complex-selector match. Taken and
    /// cleared per call; the backing allocation amortizes over the
    /// scratch's lifetime.
    failed: FxHashSet<(u32, NodeId)>,
    /// Failed-state memo for `:has()` relative searches, shared across
    /// every anchor evaluated in one style pass. Bitset-backed
    /// ([`HasFailMemo`]): one bit per (relative selector, compound
    /// level, node) instead of one 24-byte hash key — 2N bytes per
    /// selector at N nodes, and a probe is a u64 load plus a bit test.
    ///
    /// A recorded bit says: expanding `node` through
    /// `combinators[level - 1]` admits no match of the relative
    /// selector's `compounds[level..]`. Sharing it across anchors is
    /// sound because that proposition is anchor-independent:
    /// * the document tree is fixed — the cascade's `Styler` holds
    ///   `&'a Document`, and `bowser_dom` has no interior mutability on
    ///   the node tree, so the borrow checker forbids any DOM mutation
    ///   while the scratch lives (the same audit as [`SiblingCache`]);
    /// * the relative selector is fixed — the key carries the selector's
    ///   identity (its address; the cascade's stylesheets are borrowed
    ///   for the styler's whole lifetime, so addresses are stable and
    ///   unique within the pass);
    /// * the element states are fixed — the cascade resolves them from
    ///   one map, and any change bumps the style-pass epoch, which
    ///   clears this set (see [`clear_has_memo`]);
    /// * the sibling cache is fixed for the pass (same DOM-immutability
    ///   audit as the document).
    ///
    /// The anchor only determines where the level-0 search *starts*; the
    /// memoized subproblems (`level >= 1`) never mention it. So a
    /// failure recorded under anchor A is valid for anchor B, and one
    /// `:has()` rule over N anchors costs O(16·subtree² + N·subtree)
    /// total instead of Σ O(16·subtree²) — the first anchor explores,
    /// every later anchor replays memo hits.
    ///
    /// A nested `:has()` cannot occur — the parser rejects it outright
    /// (`SelCursor::in_has` in `selector.rs`) — but the memo would share
    /// safely anyway: any distinct relative selector carries a different
    /// address, so its bits live in a disjoint slot and cannot poison
    /// another search's entries. The bitsets themselves hold no
    /// in-progress iteration state (unlike the `Vec` stacks, which is
    /// why *those* are still
    /// taken/restored per call) — only monotonic inserts — so a nested
    /// search recording into it mid-outer-search is harmless, and its
    /// entries are valid under the same pass context anyway.
    ///
    /// Lifetime: scoped to one style-resolution pass. A fresh scratch
    /// starts empty (the cascade's `Styler` owns one per pass);
    /// [`clear_has_memo`] runs at pass boundaries — every interactive-
    /// state epoch bump — and the one-shot entries clear it per call.
    ///
    /// Deliberately uncapped: the wave-27 lesson stands — evicting
    /// recorded failures re-expands them exponentially, and even
    /// *pausing* inserts mid-search would reintroduce the per-anchor
    /// exponential blowup the memo exists to prevent. The size is
    /// bounded structurally instead: at most 16 levels × R distinct
    /// long relative selectors × N document nodes, *one bit each*
    /// (linear in the stylesheet's `:has()` count and the document
    /// size — both attacker-paid). The bitset drops the old key's
    /// document tag: the memo is scoped to one pass over one document
    /// (same immutability audit as above), and `clear_has_memo` wipes
    /// it at every boundary anyway.
    has_failed: HasFailMemo,
    /// Backtracking choice-point stack for `match_complex`.
    frames: Vec<Frame>,
    /// Depth-first choice-point stack for `dfs_relative`.
    rel_frames: Vec<RelFrame>,
    /// Test-only: shared-`:has()`-memo probes, so tests can assert
    /// cross-anchor reuse deterministically.
    #[cfg(test)]
    has_memo_probes: u64,
    /// Test-only: shared-`:has()`-memo hits.
    #[cfg(test)]
    has_memo_hits: u64,
    /// Test-only: `dfs_relative` search-expansion pushes, so tests can
    /// assert the second anchor re-expands nothing.
    #[cfg(test)]
    rel_expansions: u64,
}

/// Cross-anchor `:has()` failure memo, bitset-backed.
///
/// A set bit at `(rel, level, node)` says: expanding `node` through
/// `combinators[level - 1]` admits no match of the relative selector's
/// `compounds[level..]`. Anchor-independent by construction: the anchor
/// appears nowhere in the recorded state (see
/// [`MatchScratch::has_failed`] for the soundness argument).
///
/// The old `HashSet<HasFailKey>` key cost 24 bytes per entry and could
/// hold 16·N entries per distinct long relative selector (~460 MB at
/// N=10⁶, multiplied by R distinct selectors). The DOM arena is
/// append-only — `NodeId` indexes are dense in `[0, N)` and never reused
/// (`bowser_dom`'s arena slots are never recycled) — so each
/// `(rel, level)` subproblem memoizes as one bit per node: 16 × N/8 =
/// 2N bytes per selector (~2 MB at N=10⁶, ~230× smaller), and probes are
/// a u64 load plus a bit test instead of a hash lookup.
///
/// The pass-stable `rel` address namespace (the relative selector's
/// address, minted in [`matches_relative`] only for long selectors)
/// maps to a compact `u32` index by linear scan of [`rels`](Self::rels):
/// R is tiny (the distinct long relative selectors of one pass), so the
/// scan is noise. `:has()` cannot nest (the parser rejects it outright),
/// so any two relative selectors here are siblings with distinct
/// addresses, and each mints a disjoint slot by construction — no search
/// can poison another's bits.
///
/// Bitsets size lazily via [`ensure`](Self::ensure): the arena only
/// grows, so a sized bitset never needs to shrink. Deliberately
/// uncapped, same as before — eviction re-expands exponentially
/// (wave-27) and the bitset makes a cap unnecessary anyway: the size is
/// bounded structurally at 16 levels × R selectors × N nodes, one bit
/// each.
#[derive(Debug, Default)]
struct HasFailMemo {
    /// Pass-stable relative-selector addresses, in index order; index
    /// `r` owns bitset slots `r * HAS_MEMO_LEVELS .. (r + 1) *
    /// HAS_MEMO_LEVELS`.
    rels: Vec<usize>,
    /// `rels.len() * HAS_MEMO_LEVELS` bitsets; slot
    /// `r * HAS_MEMO_LEVELS + level` holds the failure bits of rel `r`
    /// at compound `level`, one bit per node arena index.
    bits: Vec<Vec<u64>>,
    /// Set bits across all slots: the exact entry count.
    len: usize,
}

/// Compound levels a `:has()` relative selector can carry: the parser
/// caps relative selectors at 16 compounds (`MAX_HAS_RELATIVE_COMPOUNDS`
/// in `selector.rs`), so levels 0..16 cover every recorded subproblem.
const HAS_MEMO_LEVELS: usize = 16;

impl HasFailMemo {
    /// Compact slot owner for the pass-stable `rel` address, minting 16
    /// fresh zeroed bitset slots on first use. Linear scan: R is tiny.
    fn rel_index(&mut self, rel: usize) -> usize {
        if let Some(i) = self.rels.iter().position(|&r| r == rel) {
            return i;
        }
        let i = self.rels.len();
        self.rels.push(rel);
        self.bits.resize_with((i + 1) * HAS_MEMO_LEVELS, Vec::new);
        i
    }

    /// Compact index for an already-known address; `None` means nothing
    /// was ever recorded under it, so no probe can hit.
    fn known_rel_index(&self, rel: usize) -> Option<usize> {
        self.rels.iter().position(|&r| r == rel)
    }

    /// Grow bitset `slot` to cover `node_idx`, zero-filling. The arena
    /// only grows, so this only ever extends.
    #[allow(clippy::manual_div_ceil)]
    fn ensure(&mut self, slot: usize, node_idx: usize) {
        // Words needed so word `node_idx / 64` exists: one more than
        // the word index. (Not `div_ceil`: bit `idx` lives in word
        // `idx / 64`, so `idx.div_ceil(64)` is one word short whenever
        // `idx` is a multiple of 64.)
        let words = node_idx / 64 + 1;
        let bits = &mut self.bits[slot];
        if bits.len() < words {
            bits.resize(words, 0);
        }
    }

    /// True when a previous search — under this or any earlier anchor
    /// of the pass — already proved that no match of the relative
    /// selector's `compounds[level..]` starts in `node`'s expansion.
    fn probe(&self, rel: usize, level: u32, node: NodeId) -> bool {
        let level = level as usize;
        if level >= HAS_MEMO_LEVELS {
            return false;
        }
        let Some(r) = self.known_rel_index(rel) else {
            return false;
        };
        let idx = node.index();
        let bits = &self.bits[r * HAS_MEMO_LEVELS + level];
        idx / 64 < bits.len() && bits[idx / 64] & (1u64 << (idx % 64)) != 0
    }

    /// Record that expanding `node` at `level` provably admits no match
    /// of the relative selector's `compounds[level..]`.
    fn record(&mut self, rel: usize, level: u32, node: NodeId) {
        let level = level as usize;
        if level >= HAS_MEMO_LEVELS {
            return;
        }
        let r = self.rel_index(rel);
        let idx = node.index();
        let slot = r * HAS_MEMO_LEVELS + level;
        self.ensure(slot, idx);
        let word = &mut self.bits[slot][idx / 64];
        let mask = 1u64 << (idx % 64);
        if *word & mask == 0 {
            *word |= mask;
            self.len += 1;
        }
    }

    /// Drop every entry, keeping the bitsets' capacity for the next
    /// pass (same capacity-retention contract as the old set's
    /// [`clear`](HashSet::clear)).
    fn clear(&mut self) {
        for bits in &mut self.bits {
            bits.fill(0);
        }
        self.rels.clear();
        self.len = 0;
    }

    /// Exact number of recorded failures.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.len
    }
}

/// Backing capacity pre-sized into the failed-state memo by
/// [`MatchScratch::warm_up`]: large enough that short-selector misses
/// never grow the set, small enough to be noise on a style pass.
const MEMO_WARM_CAPACITY: usize = 64;

/// Backing capacity pre-sized into each backtracking stack by
/// [`MatchScratch::warm_up`]: covers the choice-point depth of typical
/// selectors; deeper (hostile or pathological) selectors grow the stack
/// once, amortized over the scratch's lifetime.
const STACK_WARM_CAPACITY: usize = 8;

/// Retention ceiling for the `:has()` failure memo's bitset backing,
/// enforced by [`MatchScratch::clear_has_memo`].
///
/// The thread-local `SHARED_SCRATCH` lives indefinitely, so without a
/// ceiling a hostile one-shot call that sized the bitsets huge would pin
/// that backing forever. 8 MiB is far past warm-path sizes — one long
/// `:has()` selector over a million-node document reserves 16 x 128 KiB
/// = 2 MiB, and several such selectors still stay under — so only
/// multi-million-node hostile shapes trip the release, while normal
/// passes keep their warm capacity and the warm path stays allocation-free.
const HAS_MEMO_RETENTION_BYTES: usize = 8 << 20;

impl MatchScratch {
    /// A fully lazy scratch: zero heap allocation. Every entry point
    /// warms it on first use (see [`warm_up`]).
    pub fn new() -> Self {
        MatchScratch {
            failed: FxHashSet::default(),
            has_failed: HasFailMemo::default(),
            frames: Vec::new(),
            rel_frames: Vec::new(),
            #[cfg(test)]
            has_memo_probes: 0,
            #[cfg(test)]
            has_memo_hits: 0,
            #[cfg(test)]
            rel_expansions: 0,
        }
    }

    /// Drop the cross-anchor `:has()` failure memo, keeping its capacity.
    ///
    /// Pass-boundary invalidation: the memo's entries depend on the
    /// document, the stylesheets, and the elements' interactive states,
    /// so they are only valid within one style-resolution pass. The
    /// cascade calls this when a pass ends — a fresh [`MatchScratch`]
    /// starts empty, and
    /// [`Styler::set_element_state`](crate::cascade::Styler::set_element_state)
    /// clears it on every epoch bump (states changed, so every recorded
    /// failure may be stale). The one-shot entries clear it per call: a
    /// one-shot call evaluates a single anchor, and its thread-local or
    /// caller-owned scratch may span selectors and documents across
    /// calls, where the address-keyed entries would be unsound.
    pub(crate) fn clear_has_memo(&mut self) {
        // Retention hygiene: the thread-local `SHARED_SCRATCH` outlives
        // any single call, so a hostile one-shot that sized the memo's
        // bitsets huge would pin that backing forever. Past the ceiling
        // the backing is released instead of retained; normal sizes keep
        // their warm capacity, so the warm path is untouched.
        let reserved: usize = self.has_failed.bits.iter().map(|b| b.capacity() * 8).sum();
        if reserved > HAS_MEMO_RETENTION_BYTES {
            self.has_failed = HasFailMemo::default();
        } else {
            self.has_failed.clear();
        }
    }

    /// Test-only: number of entries in the shared `:has()` failure memo.
    #[cfg(test)]
    pub(crate) fn has_failed_len(&self) -> usize {
        self.has_failed.len()
    }

    /// Test-only: heap bytes currently used by the `:has()` memo
    /// bitsets.
    #[cfg(test)]
    pub(crate) fn has_failed_bits_bytes(&self) -> usize {
        self.has_failed.bits.iter().map(|b| b.len() * 8).sum()
    }

    /// Test-only: number of distinct relative-selector slots in the
    /// `:has()` memo.
    #[cfg(test)]
    pub(crate) fn has_failed_rel_count(&self) -> usize {
        self.has_failed.rels.len()
    }

    /// Test-only: heap bytes the `:has()` memo bitsets keep reserved
    /// after [`clear_has_memo`](Self::clear_has_memo).
    #[cfg(test)]
    pub(crate) fn has_failed_bits_capacity_bytes(&self) -> usize {
        self.has_failed.bits.iter().map(|b| b.capacity() * 8).sum()
    }

    /// Test-only: shared-memo probes, hits, and `dfs_relative` search
    /// expansions since the last reset.
    #[cfg(test)]
    pub(crate) fn memo_counters(&self) -> (u64, u64, u64) {
        (
            self.has_memo_probes,
            self.has_memo_hits,
            self.rel_expansions,
        )
    }

    /// Test-only: zero the [`memo_counters`](Self::memo_counters).
    #[cfg(test)]
    pub(crate) fn reset_memo_counters(&mut self) {
        self.has_memo_probes = 0;
        self.has_memo_hits = 0;
        self.rel_expansions = 0;
    }

    /// Pre-size the scratch's backing buffers, once. Idempotent: later
    /// calls only top the buffers back up if something shrank them
    /// (nothing in the matcher shrinks them, so in practice this runs
    /// exactly once per scratch).
    ///
    /// Called by every matcher entry point on first use — the cascade's
    /// `Styler` warms its pass-lifetime scratch on its first
    /// [`matches_cascade`] call, and the one-shot entries warm the
    /// thread's shared scratch — so the first recorded memo failure and
    /// the first backtrack push reuse warm capacity instead of
    /// allocating. `clear()`-on-entry semantics are unchanged: warming
    /// only reserves capacity, never inserts entries.
    pub(crate) fn warm_up(&mut self) {
        if self.failed.capacity() < MEMO_WARM_CAPACITY {
            self.failed.reserve(MEMO_WARM_CAPACITY);
        }
        if self.frames.capacity() < STACK_WARM_CAPACITY {
            self.frames.reserve(STACK_WARM_CAPACITY);
        }
        if self.rel_frames.capacity() < STACK_WARM_CAPACITY {
            self.rel_frames.reserve(STACK_WARM_CAPACITY);
        }
    }
}

/// Right-to-left complex-selector matching with backtracking.
///
/// The rightmost compound must match `node`; then each combinator walks
/// left to a matching sibling/ancestor. Descendant and subsequent-sibling
/// combinators try *every* candidate witness, nearest first — not just the
/// nearest: `a > b c` against nested `b`s binds the inner `b` first, and
/// on failure backtracks to the outer one whose parent is an `a`.
///
/// Iterative with an explicit choice-point stack rather than recursion:
/// recursion depth would be bounded by selector length, and stylesheets
/// are remote input.
///
/// Failed-state memoization (below) keeps hostile selectors polynomial:
/// without it, backtracking over `k` compounds costs O(size^k),
/// exponential in the selector length.
fn match_complex(
    selector: &Selector,
    doc: &Document,
    node: NodeId,
    state_for: &dyn Fn(NodeId) -> ElementState,
    sib: Option<&SiblingCache>,
    scratch: &mut MatchScratch,
    pre: Option<SubjectAttrs<'_>>,
) -> bool {
    let n = selector.compounds.len();
    if n == 0 {
        return false;
    }
    if !matches_compound(
        doc,
        node,
        &selector.compounds[n - 1],
        state_for,
        sib,
        scratch,
        pre,
    ) {
        return false;
    }
    if n == 1 {
        return true;
    }
    // Failed-state memo: (compound index, witness node) pairs whose
    // leftward prefix provably cannot match. Each pair is expanded at
    // most once, so a hostile descendant chain degrades from O(size^k)
    // to O(k * size^2). The memo is scoped to this call: `state_for`
    // and `sib` are fixed for the whole match, so a recorded failure
    // stays valid. It only engages for long selectors (see
    // `MEMO_MIN_COMPOUNDS`): short ones are already polynomial and keep
    // the zero-allocation search. The set is taken from the shared
    // scratch and cleared on entry (its keys are scoped to this call's
    // selector); the backing allocation is amortized over the scratch's
    // lifetime — every entry point warms it on first use (see
    // [`MatchScratch::warm_up`]).
    //
    // This is the *per-complex-selector* memo (`MatchScratch::failed`).
    // The `:has()` relative-search memo is a different set
    // (`MatchScratch::has_failed`): its entries are anchor-independent
    // and shared across the whole style pass instead of cleared here —
    // see `matches_relative`.
    let mut failed: Option<FxHashSet<(u32, NodeId)>> = (n >= MEMO_MIN_COMPOUNDS).then(|| {
        let mut set = std::mem::take(&mut scratch.failed);
        set.clear();
        set
    });
    // Choice-point stack: frame k chooses a node for `compounds[n-2-k]`,
    // standing in `combinators[n-2-k]` relation to the node chosen for
    // `compounds[n-1-k]`. Candidates enumerate nearest-first; an
    // exhausted frame pops and the previous frame resumes its own
    // enumeration (backtracking). Taken from the shared scratch instead
    // of allocating a fresh `Vec` per call (see `MatchScratch`); a nested
    // `match_complex` takes the emptied scratch, so the outer backtrack
    // state is never clobbered. Handed back on exit.
    let mut frames: Vec<Frame> = std::mem::take(&mut scratch.frames);
    frames.clear();
    frames.push(Frame::seed(doc, selector.combinators[n - 2], node));
    let matched = 'search: loop {
        let top = frames.len() - 1;
        let candidate = frames[top].next_candidate(doc);
        match candidate {
            Some(cand) => {
                // The compound index this frame is choosing for.
                let i = n - 2 - top;
                if matches_compound(
                    doc,
                    cand,
                    &selector.compounds[i],
                    state_for,
                    sib,
                    scratch,
                    pre,
                ) {
                    if i == 0 {
                        break 'search true;
                    }
                    // Skip witnesses already proven to admit no leftward
                    // match: re-expanding them is the exponential blowup.
                    let known_failed = failed
                        .as_ref()
                        .is_some_and(|set| set.contains(&(i as u32, cand)));
                    if !known_failed {
                        frames.push(Frame::seed(doc, selector.combinators[i - 1], cand));
                    }
                }
            }
            None => {
                let depth = frames.len() - 1;
                let frame = frames.pop().expect("frames is non-empty here");
                // The exhausted frame chose for compound `n - 2 - depth`
                // and every candidate failed, so no match of
                // compounds[0..=n-1-depth] can end at the right-neighbor
                // witness: record the failure for compound `n - 1 - depth`.
                if let Some(set) = failed.as_mut() {
                    set.insert(((n - 1 - depth) as u32, frame.right));
                }
                if frames.is_empty() {
                    break 'search false;
                }
            }
        }
    };
    // Hand the buffers back to the scratch for the next match. A nested
    // `match_complex` (via `:is()`/`:not()`/`:has()`) took the emptied
    // scratch for its own use and restored it on exit, so these locals
    // are this call's buffers again here.
    if let Some(set) = failed.take() {
        scratch.failed = set;
    }
    scratch.frames = frames;
    matched
}

/// Minimum compound count for failed-state memoization in
/// [`match_complex`] and [`dfs_relative`].
///
/// Backtracking over `k` compounds costs O(size^k) — exponential in the
/// selector length — without memoization, but is already polynomial for
/// short selectors, so the memo only engages at 5+ compounds, where the
/// exponential lives. Short selectors keep the current zero-allocation
/// search the `:has()` allocation test pins.
///
/// The memo is deliberately *uncapped*: capping the failed-state set was
/// tried and reverted — evicted failures get re-expanded exponentially,
/// which breaks the polynomial search guarantee the memo exists to
/// provide. The size bound comes from parse time instead: `:has()`
/// relative selectors are capped at 16 compounds
/// (`MAX_HAS_RELATIVE_COMPOUNDS` in `selector.rs`) and `:has()` cannot
/// nest, so the memo this set backs holds at most 16 x subtree entries.
/// Top-level complex selectors — and every `:is()`/`:not()`/`:where()`
/// argument, each of which gets its own `match_complex` call and
/// therefore its own per-call memo — are capped at 64 compounds
/// (`MAX_COMPLEX_COMPOUNDS` in `selector.rs`), so that memo holds at
/// most 64 x nodes entries per call.
const MEMO_MIN_COMPOUNDS: usize = 5;

/// One backtracking choice point: candidate nodes for a compound standing
/// in a combinator relation to the already-chosen right-neighbor node.
#[derive(Debug)]
struct Frame {
    combinator: Combinator,
    /// Next candidate to try: a single node for `>`/`+`, or the cursor of
    /// an ancestor/sibling chain for ` `/`~` (`None` = exhausted).
    cursor: Option<NodeId>,
    /// The already-chosen witness for the compound to the right, which
    /// this frame's candidates must stand in `combinator` relation to.
    /// Keys the failed-state memo when the frame exhausts.
    right: NodeId,
}

impl Frame {
    fn seed(doc: &Document, combinator: Combinator, right: NodeId) -> Self {
        let cursor = match combinator {
            Combinator::Child | Combinator::Descendant => doc.parent(right),
            Combinator::NextSibling | Combinator::SubsequentSibling => {
                prev_element_sibling(doc, right)
            }
        };
        Frame {
            combinator,
            cursor,
            right,
        }
    }

    /// The next candidate, nearest-first; `None` when exhausted.
    fn next_candidate(&mut self, doc: &Document) -> Option<NodeId> {
        let cand = self.cursor?;
        self.cursor = match self.combinator {
            Combinator::Child | Combinator::NextSibling => None,
            Combinator::Descendant => doc.parent(cand),
            Combinator::SubsequentSibling => prev_element_sibling(doc, cand),
        };
        Some(cand)
    }
}

// ---------------------------------------------------------------------------
// Rule indexing: rightmost-compound fast path
// ---------------------------------------------------------------------------

/// A `(rule, selector)` reference into a [`Stylesheet`]: the unit the
/// cascade tests against an element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct SelectorRef {
    pub(crate) rule: usize,
    pub(crate) selector: usize,
}

/// Fast-path index over one stylesheet's rules.
///
/// Selector matching runs right-to-left: the rightmost compound must
/// match the element itself. A selector whose rightmost compound names a
/// tag, id, or class can therefore only match elements carrying that
/// tag/id/class, so the candidate selectors for an element are the union
/// of its tag bucket, its id bucket, its class buckets, and the universal
/// bucket (selectors with no tag/id/class in the rightmost compound,
/// e.g. `[data-x]`, `:root`, `*`). The index is a pure pre-filter: every
/// selector it excludes is guaranteed not to match, so full
/// [`matches_with_state`] still decides each candidate and matching
/// semantics are unchanged.
#[derive(Clone, Debug, Default)]
pub(crate) struct RuleIndex {
    /// Tag buckets keyed by [`NameAtom`]: the atom is a `u32`, so the
    /// lookup hashes 4 bytes instead of the whole tag string, and the
    /// element side needs no string work at all — its atom comes
    /// straight from the interned tag name.
    ///
    /// The key is the ASCII-lowercased name's atom
    /// ([`NameAtom::lookup_html`]): HTML type selectors match ASCII
    /// case-insensitively and foreign ones match exactly, but a
    /// lowercased bucket key is a correct pre-filter either way — the
    /// full matcher re-checks the exact name, and two names share a
    /// bucket only when they lowercase to the same atom. Names outside
    /// the atom table (custom elements, mixed-case foreign names like
    /// `foreignObject`) share [`Self::by_tag_unknown`] instead.
    by_tag: FxHashMap<NameAtom, Vec<SelectorRef>>,
    /// Tag selectors naming something outside the atom table. One
    /// shared pre-filter bucket: the full matcher re-checks exact
    /// names, so this only adds false positives, never misses.
    by_tag_unknown: Vec<SelectorRef>,
    /// Every distinct id/class string named by an indexed selector,
    /// interned once at build to a dense `u32` key. The element side
    /// probes this with one FxHash of the attribute string — no
    /// SipHash over the full string per element — and indexes the
    /// bucket `Vec`s below directly. Ids and classes share one key
    /// space; the bucket vecs are separate, so a shared key never
    /// mixes an id bucket with a class bucket.
    ///
    /// `NameAtom` cannot serve here: it names only the fixed
    /// well-known table, while ids/classes are arbitrary author
    /// strings. Exact-string interning is still a sound pre-filter —
    /// id and class matching are both case-sensitive, so two names
    /// share a key exactly when they are equal, and the full matcher
    /// re-checks every candidate anyway.
    strings: FxHashMap<String, u32>,
    /// `id_buckets[k]` holds the refs whose rightmost compound names
    /// the id interned as `k` in [`Self::strings`]; empty when no
    /// selector names that id. Indexed by the interned key, so the
    /// element side pays no second hash.
    id_buckets: Vec<Vec<SelectorRef>>,
    /// `class_buckets[k]` holds the refs whose rightmost compound
    /// names the class interned as `k` in [`Self::strings`]; empty
    /// when no selector names that class.
    class_buckets: Vec<Vec<SelectorRef>>,
    universal: Vec<SelectorRef>,
}

impl RuleIndex {
    /// The interned key for `s`, interning it on first use. Keys are
    /// dense from 0, so they index [`Self::id_buckets`] /
    /// [`Self::class_buckets`] directly. Both bucket vecs grow in
    /// lockstep with the interner, so the element side indexes them
    /// without bounds checks.
    fn intern(&mut self, s: &str) -> u32 {
        // Keys are dense from 0 and never removed, so `len` is the key a
        // miss takes. The entry API hashes once instead of get+insert.
        let k = self.strings.len() as u32;
        match self.strings.entry(s.to_owned()) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                e.insert(k);
                self.id_buckets.push(Vec::new());
                self.class_buckets.push(Vec::new());
                k
            }
        }
    }
}

impl RuleIndex {
    /// Index every selector of every rule in `sheet` by its rightmost
    /// compound's tag name, ids, and classes.
    ///
    /// Every bucket is pushed in ascending `(rule, selector)` order, so
    /// each bucket is pre-sorted — [`candidates_into`] merges them
    /// instead of sorting per element.
    pub(crate) fn build(sheet: &Stylesheet) -> Self {
        let mut index = RuleIndex::default();
        for (rule_idx, rule) in sheet.rules().iter().enumerate() {
            for (sel_idx, selector) in rule.selectors.iter().enumerate() {
                let Some(rightmost) = selector.compounds.last() else {
                    // `matches_with_state` rejects empty selectors anyway.
                    continue;
                };
                let r = SelectorRef {
                    rule: rule_idx,
                    selector: sel_idx,
                };
                let mut bucketed = false;
                if let Some(name) = rightmost.tag.as_ref().and_then(|t| t.name.as_ref()) {
                    // Atom-keyed tag bucket (see `by_tag`): allocation-
                    // free for short names via `lookup_html`'s stack
                    // buffer; everything the atom table cannot name
                    // shares the unknown bucket.
                    match NameAtom::lookup_html(name) {
                        Some(atom) => index.by_tag.entry(atom).or_default().push(r),
                        None => index.by_tag_unknown.push(r),
                    }
                    bucketed = true;
                }
                for id in &rightmost.ids {
                    let k = index.intern(id) as usize;
                    index.id_buckets[k].push(r);
                    bucketed = true;
                }
                for class in &rightmost.classes {
                    let k = index.intern(class) as usize;
                    index.class_buckets[k].push(r);
                    bucketed = true;
                }
                if !bucketed {
                    index.universal.push(r);
                }
            }
        }
        index
    }

    /// Calls `f` once per candidate bucket for `data`: its tag bucket,
    /// its id bucket, one bucket per class token, and the universal
    /// bucket. Empty buckets are yielded too; callers skip those as
    /// needed. Callers must pass only elements: no selector matches a
    /// non-element, and the cascade skips those up front.
    ///
    /// The yielded slices borrow the index itself (not the call), so a
    /// caller may stash them in a longer-lived buffer for the merge.
    fn for_each_bucket<'s>(&'s self, data: &ElementData, mut f: impl FnMut(&'s [SelectorRef])) {
        // Tag bucket. The element's atom is the key directly — no
        // string hashing. Well-known names arrive lowercase from the
        // parser / `createElement`, so the atom hits `by_tag` as-is.
        // Anything else (unknown or mixed-case names) goes through
        // `lookup_html`, mirroring `build`'s keying: a hit probes the
        // atom bucket (e.g. element `RECT` finds selector `rect`), a
        // miss takes the shared unknown bucket.
        let tag_name = &data.tag_name;
        let atom = tag_name.atom();
        if atom != NameAtom::UNKNOWN && tag_name.as_str().bytes().all(|b| !b.is_ascii_uppercase()) {
            if let Some(bucket) = self.by_tag.get(&atom) {
                f(bucket);
            }
        } else if let Some(lowered) = NameAtom::lookup_html(tag_name.as_str()) {
            if let Some(bucket) = self.by_tag.get(&lowered) {
                f(bucket);
            }
        } else {
            f(&self.by_tag_unknown);
        }
        // Id bucket: one FxHash of the id string into the build-time
        // interner, then a direct `Vec` index — no per-element
        // SipHash over the full string, no string-keyed map probe.
        if let Some(id) = attr_value_by_atom(data, NameAtom::id())
            && let Some(&k) = self.strings.get(id)
        {
            f(&self.id_buckets[k as usize]);
        }
        // Class buckets: same, one FxHash per whitespace-separated
        // token.
        if let Some(class_attr) = attr_value_by_atom(data, NameAtom::class()) {
            for tok in class_attr.split_ascii_whitespace() {
                if let Some(&k) = self.strings.get(tok) {
                    f(&self.class_buckets[k as usize]);
                }
            }
        }
        f(&self.universal);
    }

    /// Candidate `(rule, selector)` refs for an element, appended to
    /// `out`: its tag, id, and class buckets plus the universal bucket,
    /// sorted by `(rule, selector)` and deduplicated (a selector lands in
    /// several buckets when its rightmost compound names several
    /// ids/classes, and `.b.b` can even push one ref twice into a single
    /// bucket). Callers must pass only elements: no selector matches a
    /// non-element, and the cascade skips those up front.
    ///
    /// Buckets arrive pre-sorted from [`build`], so this is a k-way merge
    /// over the bucket slices with on-the-fly dedup — no per-element
    /// sort. `out` is a caller-provided scratch buffer, cleared on entry
    /// and reused across queries instead of allocating a fresh `Vec` per
    /// element per stylesheet.
    pub(crate) fn candidates_into(&self, data: &ElementData, out: &mut Vec<SelectorRef>) {
        out.clear();
        // The slice list lives on the stack: tag + id + universal + up
        // to 29 class buckets. Past that, `candidates_sort_fallback`
        // runs the old extend + sort + dedup (correct, just slower).
        const MAX_BUCKETS: usize = 32;
        let mut slices: [&[SelectorRef]; MAX_BUCKETS] = [&[]; MAX_BUCKETS];
        let mut count = 0usize;
        let mut overflow = false;
        self.for_each_bucket(data, |s| {
            if s.is_empty() {
                return;
            }
            if count < MAX_BUCKETS {
                slices[count] = s;
                count += 1;
            } else {
                overflow = true;
            }
        });
        if overflow {
            return self.candidates_sort_fallback(data, out);
        }
        if count == 1 {
            // Single live bucket, pre-sorted: copy it and drop adjacent
            // duplicates (`.b.b` can push one ref twice into a bucket).
            out.extend_from_slice(slices[0]);
            out.dedup();
            return;
        }
        // K-way merge with a linear head scan: the bucket count is tiny,
        // so this beats a heap. `pos[i]` is the read head into
        // `slices[i]`; the smallest head wins each round, and equal refs
        // (across buckets or within one bucket) are emitted once.
        // Exhausted buckets are swap-removed (`live` shrinks), so the
        // scan only visits buckets that still have heads — the common
        // skewed case (one huge universal bucket plus a few tiny ones)
        // degrades to a linear pass.
        let mut pos = [0usize; MAX_BUCKETS];
        out.reserve(slices[..count].iter().map(|s| s.len()).sum());
        let mut live = count;
        let mut last: Option<SelectorRef> = None;
        loop {
            let mut best: Option<(usize, SelectorRef)> = None;
            for i in 0..live {
                let r = slices[i][pos[i]];
                if best.is_none_or(|(_, b)| r < b) {
                    best = Some((i, r));
                }
            }
            let Some((bi, r)) = best else { break };
            pos[bi] += 1;
            if pos[bi] == slices[bi].len() {
                live -= 1;
                slices.swap(bi, live);
                pos.swap(bi, live);
            }
            if last != Some(r) {
                out.push(r);
                last = Some(r);
            }
        }
    }

    /// [`candidates_into`] fallback when an element hits more buckets
    /// than the stack merge handles (30+ class tokens with live
    /// buckets): the previous extend + sort + dedup. `out` is empty on
    /// entry.
    fn candidates_sort_fallback(&self, data: &ElementData, out: &mut Vec<SelectorRef>) {
        debug_assert!(out.is_empty());
        self.for_each_bucket(data, |s| out.extend_from_slice(s));
        out.sort();
        out.dedup();
    }
}

fn is_element(doc: &Document, node: NodeId) -> bool {
    matches!(doc.kind(node), NodeKind::Element(_))
}

fn element_data(doc: &Document, node: NodeId) -> Option<&ElementData> {
    match doc.kind(node) {
        NodeKind::Element(d) => Some(d),
        _ => None,
    }
}

/// Nearest preceding sibling that is an element (CSS sibling combinators
/// only consider elements).
fn prev_element_sibling(doc: &Document, node: NodeId) -> Option<NodeId> {
    let mut s = doc.previous_sibling(node);
    while let Some(n) = s {
        if is_element(doc, n) {
            return Some(n);
        }
        s = doc.previous_sibling(n);
    }
    None
}

fn next_element_sibling(doc: &Document, node: NodeId) -> Option<NodeId> {
    let mut s = doc.next_sibling(node);
    while let Some(n) = s {
        if is_element(doc, n) {
            return Some(n);
        }
        s = doc.next_sibling(n);
    }
    None
}

fn matches_compound(
    doc: &Document,
    node: NodeId,
    c: &CompoundSelector,
    state_for: &dyn Fn(NodeId) -> ElementState,
    sib: Option<&SiblingCache>,
    scratch: &mut MatchScratch,
    pre: Option<SubjectAttrs<'_>>,
) -> bool {
    // Pseudo-element selectors never match a real element: their
    // declarations belong to a pseudo box, not the originating element's
    // own style. (Real pseudo-element boxes come later; until then the
    // declarations apply nowhere rather than contaminating the
    // originating element's computed style.)
    if c.pseudo_element.is_some() {
        return false;
    }
    let data = match element_data(doc, node) {
        Some(d) => d,
        None => return false,
    };
    let html = data.namespace == Namespace::Html;

    match &c.tag {
        None => {}
        Some(tag) => {
            if !namespace_matches(tag.namespace.as_ref(), &data.namespace) {
                return false;
            }
            if let Some(name) = &tag.name {
                // Fast path: both sides are atoms — one `u32` compare.
                // Element tag atoms are canonical (HTML tag names are
                // lowercased at intern time), so the selector name goes
                // through the case-insensitive lookup for HTML elements.
                // Either side outside the atom table (custom elements,
                // `createElementNS` without lowercasing, ...) falls back
                // to the string compare, which stays correct against
                // spilled names.
                let elem = &data.tag_name;
                let ok = if elem.atom() == NameAtom::UNKNOWN {
                    if html {
                        elem.as_str().eq_ignore_ascii_case(name)
                    } else {
                        elem.as_str() == name.as_str()
                    }
                } else if html {
                    match NameAtom::lookup_html(name) {
                        Some(sel) => elem.atom() == sel,
                        None => elem.as_str().eq_ignore_ascii_case(name),
                    }
                } else {
                    match NameAtom::lookup(name) {
                        Some(sel) => elem.atom() == sel,
                        None => elem.as_str() == name.as_str(),
                    }
                };
                if !ok {
                    return false;
                }
            }
        }
    }
    if !c.ids.is_empty() {
        // One `id` lookup per compound, not per id selector: every id
        // check compares against the same attribute value. The cascade
        // precomputes it once per element (`pre`); any other tested node
        // keeps the lazy attribute-list scan.
        let id_attr = match pre {
            Some(p) if p.node == node => p.id,
            _ => attr_value_by_atom(data, NameAtom::id()),
        };
        for id in &c.ids {
            if id_attr != Some(id.as_str()) {
                return false;
            }
        }
    }
    if !c.classes.is_empty() {
        // One `class` lookup per compound: every class check tokenizes
        // the same attribute value (precomputed per element by the
        // cascade, lazy otherwise — see the `id` block above).
        let class_attr = match pre {
            Some(p) if p.node == node => p.class,
            _ => attr_value_by_atom(data, NameAtom::class()),
        };
        let Some(class_attr) = class_attr else {
            return false;
        };
        for class in &c.classes {
            if !has_class_in(class_attr, class) {
                return false;
            }
        }
    }
    for attr in &c.attrs {
        if !matches_attr(data, attr) {
            return false;
        }
    }
    // Each element's dynamic pseudo-classes see its own state, resolved
    // through `state_for` (ancestor/sibling/`:has()`-descendant compounds
    // must not see the target node's state). The `state_for` HashMap
    // lookup is skipped entirely when the compound has no pseudos.
    if c.pseudos.is_empty() {
        return true;
    }
    let state = state_for(node);
    c.pseudos
        .iter()
        .all(|p| matches_pseudo(doc, node, p, &state, state_for, sib, scratch, pre))
}

/// Check an attribute selector's namespace constraint. Attributes in
/// this DOM carry no namespace, so `[|id]` (no namespace) and unprefixed
/// / `*|`-prefixed selectors always pass the namespace check, while a
/// named prefix (`[html|id]`, `[svg|viewBox]`) never matches — even when
/// the prefix resolves, the attribute itself is in no namespace.
fn attr_namespace_matches(constraint: Option<&NamespaceConstraint>) -> bool {
    match constraint {
        None | Some(NamespaceConstraint::Any) | Some(NamespaceConstraint::NoNamespace) => true,
        Some(NamespaceConstraint::Named(_)) => false,
    }
}

/// Check a type selector's namespace constraint against the element's
/// namespace. No prefix (`div`) and `*|` match any namespace; `|` (no
/// namespace) never matches — every element in this DOM carries a
/// namespace; a named prefix must resolve via the built-in table, so
/// unknown prefixes never match.
fn namespace_matches(constraint: Option<&NamespaceConstraint>, ns: &Namespace) -> bool {
    match constraint {
        None | Some(NamespaceConstraint::Any) => true,
        Some(NamespaceConstraint::NoNamespace) => false,
        Some(NamespaceConstraint::Named(p)) => resolve_namespace_prefix(p).as_ref() == Some(ns),
    }
}

/// Attribute lookup against already-fetched element data: the same
/// semantics as [`Document::get_attribute`] (ASCII case-insensitive names
/// for HTML elements, exact for foreign elements) without re-fetching the
/// element payload through the arena on every call. Well-known names take
/// the atom fast path (one table lookup, then an integer scan over the
/// interned attribute names); anything else falls back to the string scan.
fn attr_value<'a>(data: &'a ElementData, name: &str) -> Option<&'a str> {
    let insensitive = data.namespace == Namespace::Html;
    let atom = if insensitive {
        NameAtom::lookup_html(name)
    } else {
        NameAtom::lookup(name)
    };
    match atom {
        Some(a) => attr_value_by_atom(data, a),
        None => data
            .attributes
            .as_slice()
            .iter()
            .find(|a| {
                let s = a.name.as_str();
                if insensitive {
                    s.eq_ignore_ascii_case(name)
                } else {
                    s == name
                }
            })
            .map(|a| a.value.as_str()),
    }
}

/// Attribute lookup by precomputed atom (e.g. `id`/`class`): no table
/// lookup at all — a pure integer scan over the interned names.
fn attr_value_by_atom(data: &ElementData, atom: NameAtom) -> Option<&str> {
    data.attributes
        .as_slice()
        .iter()
        .find(|a| a.name.atom() == atom)
        .map(|a| a.value.as_str())
}

/// Precomputed `id`/`class` attribute values for the subject element of
/// a cascade style match.
///
/// The cascade's `candidates()` computes this once per element and
/// threads it through [`matches_cascade`] → `match_complex` →
/// [`matches_compound`], so the rightmost compound of every candidate
/// selector reuses the values instead of re-scanning the element's
/// attribute list once per compound. It applies only when the tested
/// node is the subject node itself (`node == pre.node`):
/// ancestor/sibling nodes in combinator walks — and every other entry
/// point, which passes `None` — keep the lazy [`attr_value_by_atom`]
/// lookup, since each of those nodes is tested rarely and precomputing
/// for the whole walk would cost more than it saves.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SubjectAttrs<'a> {
    /// The subject node these values were computed for.
    node: NodeId,
    /// Precomputed `id` attribute value (`None` = no `id` attribute).
    id: Option<&'a str>,
    /// Precomputed `class` attribute value (`None` = no `class`
    /// attribute).
    class: Option<&'a str>,
}

impl<'a> SubjectAttrs<'a> {
    /// Snapshot the `id`/`class` attribute values of element `node`
    /// from already-fetched element `data`. Same semantics as the lazy
    /// [`attr_value_by_atom`] lookups in [`matches_compound`]: `None`
    /// when the attribute is absent, so id matching stays exact and
    /// class matching stays space-separated token matching on the raw
    /// value.
    pub(crate) fn for_element(node: NodeId, data: &'a ElementData) -> Self {
        Self {
            node,
            id: attr_value_by_atom(data, NameAtom::id()),
            class: attr_value_by_atom(data, NameAtom::class()),
        }
    }
}

/// Whitespace-separated class token match against an already-fetched
/// `class` attribute value. Case-sensitive: in standards mode (the
/// default) HTML class matching is case-sensitive, unlike the old ASCII
/// case-insensitive behavior.
fn has_class_in(class_attr: &str, class: &str) -> bool {
    // `split_ascii_whitespace` splits on exactly the ASCII whitespace set
    // and skips empty tokens, matching the old explicit split + filter.
    class_attr.split_ascii_whitespace().any(|tok| tok == class)
}

/// Attribute selector match. Fully allocation-free: comparisons run
/// directly on the borrowed attribute value and the selector value.
fn matches_attr(data: &ElementData, attr: &AttrSelector) -> bool {
    // Attribute namespaces are independent of the *element's* namespace:
    // attributes in this DOM carry no namespace, so `|` (no namespace)
    // and no prefix always match, while a named prefix (`[svg|viewBox]`)
    // never does — even when the prefix resolves, the attribute is not in
    // that namespace. (Type selectors, by contrast, test the element's
    // namespace via [`namespace_matches`].)
    if !attr_namespace_matches(attr.namespace.as_ref()) {
        return false;
    }
    let actual = match attr_value(data, &attr.name) {
        Some(v) => v,
        None => return false,
    };
    if attr.insensitive {
        return matches_attr_insensitive(actual, attr);
    }
    let val = attr.value.as_str();
    match attr.op {
        AttrOp::Exists => true,
        AttrOp::Equals => actual == val,
        AttrOp::Includes => {
            // Precomputed at parse time: a whitespace-containing value can
            // never match `~=`.
            if attr.value_has_ws {
                return false;
            }
            actual.split_ascii_whitespace().any(|tok| tok == val)
        }
        AttrOp::DashMatch => {
            actual == val
                || actual
                    .strip_prefix(val)
                    .is_some_and(|rest| rest.starts_with('-'))
        }
        AttrOp::Prefix => !val.is_empty() && actual.starts_with(val),
        AttrOp::Suffix => !val.is_empty() && actual.ends_with(val),
        AttrOp::Substring => !val.is_empty() && actual.contains(val),
    }
}

/// `[attr<op>value i]`: ASCII case-insensitive attribute matching.
/// `eq_ignore_ascii_case` is exactly "lowercase both sides, then compare"
/// for ASCII (non-ASCII bytes compare unchanged), so this matches the old
/// allocate-two-lowercased-copies behavior with no allocation. Byte-level
/// prefix/suffix slicing is used instead of `str` slicing so a value
/// length that lands mid-char cannot panic.
fn matches_attr_insensitive(actual: &str, attr: &AttrSelector) -> bool {
    let val = attr.value.as_str();
    let (a, v) = (actual.as_bytes(), val.as_bytes());
    match attr.op {
        AttrOp::Exists => true,
        AttrOp::Equals => actual.eq_ignore_ascii_case(val),
        AttrOp::Includes => {
            if attr.value_has_ws {
                return false;
            }
            actual
                .split_ascii_whitespace()
                .any(|tok| tok.eq_ignore_ascii_case(val))
        }
        AttrOp::DashMatch => {
            actual.eq_ignore_ascii_case(val)
                || (a.len() > v.len() && a[..v.len()].eq_ignore_ascii_case(v) && a[v.len()] == b'-')
        }
        AttrOp::Prefix => {
            !v.is_empty() && a.len() >= v.len() && a[..v.len()].eq_ignore_ascii_case(v)
        }
        AttrOp::Suffix => {
            !v.is_empty() && a.len() >= v.len() && a[a.len() - v.len()..].eq_ignore_ascii_case(v)
        }
        AttrOp::Substring => {
            // `windows(0)` panics, so the empty value is rejected first.
            !v.is_empty() && a.windows(v.len()).any(|w| w.eq_ignore_ascii_case(v))
        }
    }
}

// `pre` (the subject's precomputed id/class pair) threads through so
// nested `:is()`/`:not()` matches against the subject reuse it; the
// other seven are the pre-existing pseudo-class context.
#[allow(clippy::too_many_arguments)]
fn matches_pseudo(
    doc: &Document,
    node: NodeId,
    p: &PseudoClass,
    state: &ElementState,
    state_for: &dyn Fn(NodeId) -> ElementState,
    sib: Option<&SiblingCache>,
    scratch: &mut MatchScratch,
    pre: Option<SubjectAttrs<'_>>,
) -> bool {
    match p {
        PseudoClass::Root => doc.document_element() == Some(node),
        PseudoClass::FirstChild => prev_element_sibling(doc, node).is_none(),
        PseudoClass::LastChild => next_element_sibling(doc, node).is_none(),
        PseudoClass::NthChild(a, b) => nth_matches(*a, *b, element_index(sib, doc, node)),
        PseudoClass::NthLastChild(a, b) => {
            nth_matches(*a, *b, element_index_from_end(sib, doc, node))
        }
        PseudoClass::NthOfType(a, b) => nth_matches(*a, *b, element_index_of_type(sib, doc, node)),
        PseudoClass::NthLastOfType(a, b) => {
            nth_matches(*a, *b, element_index_of_type_from_end(sib, doc, node))
        }
        PseudoClass::FirstOfType => element_index_of_type(sib, doc, node) == 1,
        PseudoClass::LastOfType => element_index_of_type_from_end(sib, doc, node) == 1,
        PseudoClass::OnlyChild => {
            prev_element_sibling(doc, node).is_none() && next_element_sibling(doc, node).is_none()
        }
        PseudoClass::OnlyOfType => {
            element_index_of_type(sib, doc, node) == 1
                && element_index_of_type_from_end(sib, doc, node) == 1
        }
        PseudoClass::Empty => children_cursor(doc, node).all(|ch| match doc.kind(ch) {
            NodeKind::Element(_) => false,
            // Selectors 4: any text node with non-zero-length data defeats
            // `:empty` — even whitespace-only text. Comments and PIs are
            // ignored.
            NodeKind::Text(t) => t.is_empty(),
            _ => true,
        }),
        // Interactive pseudo-classes read the caller's ElementState.
        PseudoClass::Hover => state.hover,
        PseudoClass::Active => state.active,
        PseudoClass::Focus => state.focus,
        // `:link` matches an unvisited link; `:visited` matches a visited
        // one. Only HTML `a`/`area`/`link` elements with `href` are links.
        PseudoClass::Link => is_html_link(doc, node) && !state.visited,
        PseudoClass::Visited => is_html_link(doc, node) && state.visited,
        PseudoClass::Checked => is_checked(doc, node),
        PseudoClass::Disabled => is_disabled(doc, node),
        PseudoClass::Enabled => is_enabled(doc, node),
        PseudoClass::Not(args) => !args
            .iter()
            .any(|s| matches_cascade(s, doc, node, state_for, sib, scratch, pre)),
        // `:is()` and `:where()` match identically; only specificity differs.
        PseudoClass::Is(args) | PseudoClass::Where(args) => args
            .iter()
            .any(|s| matches_cascade(s, doc, node, state_for, sib, scratch, pre)),
        PseudoClass::Has(args) => matches_has(doc, node, args, state_for, sib, scratch),
    }
}

/// Per-parent memoized sibling indexes, making the structural
/// pseudo-classes (`:nth-child`, `:nth-last-child`, `:nth-of-type`,
/// `:nth-last-of-type`, `:first-of-type`, ...) O(1) amortized per query
/// instead of O(siblings) per query.
///
/// The naive implementation walks `prev_element_sibling` from the node,
/// so matching `:nth-child(...)` against every child of a parent with N
/// children costs O(N^2). The cache builds one index per parent on first
/// use — a single cursor walk over the parent's element children, no
/// `Vec` from [`Document::children`] — and every later query against
/// that parent is a hash lookup (plus a binary search for the `-of-type`
/// family).
///
/// The cache is keyed by parent [`NodeId`] only, so one [`SiblingCache`]
/// must never be shared across documents. The one-shot
/// [`matches_with_state`] passes no cache (direct sibling walks, zero
/// allocation — exactly the old behavior); the cascade's
/// [`Styler`](crate::cascade::Styler) owns one cache for its whole
/// lifetime, borrowing a single `Document`, so every repeated
/// `:nth-child` query in a style pass shares the per-parent index.
///
/// Lifetime / mutation audit: the `Styler` is strictly per-pass, and no
/// invalidation is needed, by construction. The styler holds `&'a
/// Document` — a shared borrow — while every structural or attribute
/// edit on [`Document`] (`append_child`, `remove_child`,
/// `insert_before`, `set_attribute`, the constructors) takes `&mut
/// self`, and `bowser_dom` has no interior mutability (`RefCell`,
/// `Cell`, `UnsafeCell`) on the node tree. The borrow checker therefore
/// forbids any DOM mutation while a `Styler` (and its cache) lives: a
/// cached per-parent index cannot go stale mid-pass. Each style pass
/// builds a fresh `Styler` with a fresh cache, so edits between passes
/// start clean. (Interactive-state changes via
/// [`Styler::set_element_state`](crate::cascade::Styler::set_element_state)
/// do not touch sibling structure and need no invalidation either.)
#[derive(Debug, Default)]
pub(crate) struct SiblingCache {
    by_parent: RefCell<FxHashMap<NodeId, Rc<SiblingIndex>>>,
    /// Test-only: number of per-parent indexes built, so tests can assert
    /// memoization deterministically (one build per parent, not per
    /// query).
    #[cfg(test)]
    builds: std::cell::Cell<usize>,
}

/// Memoized sibling data for one parent node: every element child gets
/// its 1-based document-order index once, and the `-of-type` family reads
/// its rank from the same build.
#[derive(Debug)]
struct SiblingIndex {
    /// 1-based index of each element child, in document order.
    positions: FxHashMap<NodeId, u32>,
    /// 1-based positions per same-type group, ascending. The group key is
    /// a [`TypeKey`] — namespace plus tag name with exactly
    /// [`same_type`]'s equivalence (HTML ASCII-case-insensitive, exact
    /// elsewhere), keyed on the interned atom: no per-child `String`
    /// allocation for well-known tags.
    by_type: FxHashMap<TypeKey, Vec<u32>>,
}

/// Discriminant for the [`TypeKey`] namespace. [`Namespace`] has no
/// `Hash` impl. Custom namespaces (see [`Namespace::Other`]) share one
/// discriminant: type selectors can only name the built-in prefixes (see
/// `resolve_namespace_prefix`), so two different custom URIs are
/// indistinguishable to selector matching.
fn ns_key(ns: &Namespace) -> u8 {
    match ns {
        Namespace::Html => 0,
        Namespace::MathMl => 1,
        Namespace::Svg => 2,
        Namespace::Other(_) => 3,
    }
}

/// Group key for [`SiblingIndex::by_type`]: namespace discriminant plus
/// tag name, with exactly [`same_type`]'s equivalence —
/// ASCII-case-insensitive in HTML, exact elsewhere.
///
/// Well-known tags borrow their canonical atom-table string (no
/// allocation, no heap hashing). Anything needing normalization —
/// unknown/spilled names, or a mixed-case well-known name under HTML's
/// case-insensitive rule (reachable only via `createElementNS`) —
/// carries a pre-normalized owned string. That path still allocates
/// once per element, exactly like the old `(u8, String)` key, and only
/// there.
#[derive(Clone, Debug)]
struct TypeKey {
    ns: u8,
    name: TypeName,
}

/// The tag-name half of a [`TypeKey`].
#[derive(Clone, Debug)]
enum TypeName {
    /// Canonical atom-table string (`&'static str`). Already lowercase
    /// for every name the HTML parser or `createElement` can produce, so
    /// it doubles as the HTML-normalized form.
    Borrowed(&'static str),
    /// Normalized owned name: ASCII-lowercased for HTML (so an unknown
    /// `<DIV>` groups with a known `<div>`, exactly like [`same_type`]'s
    /// string fallback), verbatim otherwise.
    Owned(Box<str>),
}

impl TypeName {
    fn as_str(&self) -> &str {
        match self {
            TypeName::Borrowed(s) => s,
            TypeName::Owned(s) => s,
        }
    }
}

// Manual impls: `Borrowed("div")` and `Owned("div")` name the same
// group, so equality and hashing run on the string contents.
impl PartialEq for TypeName {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for TypeName {}

impl std::hash::Hash for TypeName {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl PartialEq for TypeKey {
    fn eq(&self, other: &Self) -> bool {
        self.ns == other.ns && self.name == other.name
    }
}

impl Eq for TypeKey {}

impl std::hash::Hash for TypeKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ns.hash(state);
        self.name.hash(state);
    }
}

/// The [`SiblingIndex::by_type`] group key for an element. HTML tag
/// names are ASCII-lowercased so the key agrees with [`same_type`]'s
/// case-insensitive comparison; the atom table already canonicalizes
/// parser/`createElement` output, so well-known tags skip the
/// allocation entirely.
fn type_key(data: &ElementData) -> TypeKey {
    let ns = ns_key(&data.namespace);
    let atom = data.tag_name.atom();
    if data.namespace == Namespace::Html {
        if atom != NameAtom::UNKNOWN {
            let canonical = atom.as_str();
            if canonical.bytes().all(|b| !b.is_ascii_uppercase()) {
                // The hot path: parser/`createElement` output is already
                // lowercase — borrow the table string, zero allocation.
                return TypeKey {
                    ns,
                    name: TypeName::Borrowed(canonical),
                };
            }
            // Mixed-case well-known name, reachable only via
            // `createElementNS`: normalize like the old code did.
            return TypeKey {
                ns,
                name: TypeName::Owned(canonical.to_ascii_lowercase().into_boxed_str()),
            };
        }
        // Unknown (spilled) name: mirror `same_type`'s string fallback.
        // Already-lowercase spills still need an owned copy (the table
        // has no entry to borrow), but skip the lowercase pass.
        let spill = data.tag_name.as_str();
        let name = if spill.bytes().any(|b| b.is_ascii_uppercase()) {
            TypeName::Owned(spill.to_ascii_lowercase().into_boxed_str())
        } else {
            TypeName::Owned(spill.into())
        };
        return TypeKey { ns, name };
    }
    // Foreign namespaces compare exactly; well-known names borrow the
    // table string, unknown names carry their verbatim spill.
    TypeKey {
        ns,
        name: if atom != NameAtom::UNKNOWN {
            TypeName::Borrowed(atom.as_str())
        } else {
            TypeName::Owned(data.tag_name.as_str().into())
        },
    }
}

impl SiblingCache {
    /// The memoized index for `parent`'s element children, building it on
    /// first use with one cursor walk (no allocation from
    /// [`Document::children`]).
    fn index_for(&self, doc: &Document, parent: NodeId) -> Rc<SiblingIndex> {
        if let Some(hit) = self.by_parent.borrow().get(&parent) {
            return Rc::clone(hit);
        }
        let mut positions = FxHashMap::default();
        let mut by_type: FxHashMap<TypeKey, Vec<u32>> = FxHashMap::default();
        let mut pos: u32 = 0;
        for child in children_cursor(doc, parent) {
            let Some(data) = element_data(doc, child) else {
                continue;
            };
            pos += 1;
            positions.insert(child, pos);
            by_type.entry(type_key(data)).or_default().push(pos);
        }
        #[cfg(test)]
        self.builds.set(self.builds.get() + 1);
        let index = Rc::new(SiblingIndex { positions, by_type });
        self.by_parent
            .borrow_mut()
            .insert(parent, Rc::clone(&index));
        index
    }

    /// 1-based index of `node` among its element siblings (counting
    /// itself), O(1) amortized.
    fn index(&self, doc: &Document, node: NodeId) -> Option<u32> {
        let parent = doc.parent(node)?;
        let index = self.index_for(doc, parent);
        // `matches_compound` rejects non-elements before any structural
        // pseudo-class runs, so `node` is always an element child of
        // `parent` here; `None` is a defensive fallback.
        index.positions.get(&node).copied()
    }
}

/// 1-based index of `node` among its element siblings (counting itself).
/// With a shared cache this is O(1) amortized; without one (`None`, the
/// one-shot public API) it is the direct sibling walk — zero allocation,
/// exactly the old behavior.
fn element_index(sib: Option<&SiblingCache>, doc: &Document, node: NodeId) -> i32 {
    if let Some(sib) = sib {
        // No parent means no element siblings are possible: the node is
        // trivially first, exactly as the sibling walk below computes.
        return sib.index(doc, node).unwrap_or(1) as i32;
    }
    let mut index = 1;
    let mut s = prev_element_sibling(doc, node);
    while let Some(n) = s {
        index += 1;
        s = prev_element_sibling(doc, n);
    }
    index
}

/// `:nth-child(an+b)`: true when `index = a*k + b` for some integer k >= 0.
///
/// All arithmetic runs in `i64`: `index - b` overflows `i32` for
/// adversarial inputs like `:nth-child(n-2147483648)` (panics in debug,
/// wraps in release).
fn nth_matches(a: i32, b: i32, index: i32) -> bool {
    if a == 0 {
        return index == b;
    }
    let (a, b, index) = (a as i64, b as i64, index as i64);
    let d = index - b;
    d.rem_euclid(a) == 0 && d / a >= 0
}

/// 1-based index of `node` among its element siblings, counting from the
/// end (the last element sibling is 1). With a shared cache this is
/// derived from the same memoized index — no second walk; without one it
/// is the direct sibling walk.
fn element_index_from_end(sib: Option<&SiblingCache>, doc: &Document, node: NodeId) -> i32 {
    if let Some(sib) = sib {
        let Some(parent) = doc.parent(node) else {
            return 1;
        };
        let index = sib.index_for(doc, parent);
        let pos = index.positions.get(&node).copied().unwrap_or(1);
        return index.positions.len() as i32 - pos as i32 + 1;
    }
    let mut index = 1;
    let mut s = next_element_sibling(doc, node);
    while let Some(n) = s {
        index += 1;
        s = next_element_sibling(doc, n);
    }
    index
}

/// Whether two elements are of the same type for `:*-of-type`: same
/// namespace and same tag name (ASCII case-insensitive in HTML, exact
/// elsewhere — the same rule as type-selector matching). Canonical tag
/// atoms compare as one `u32`; spilled (unknown) names fall back to the
/// string rule so non-canonical names (e.g. `createElementNS` without
/// lowercasing) still group correctly.
fn same_type(a: &ElementData, b: &ElementData) -> bool {
    if a.namespace != b.namespace {
        return false;
    }
    let (aa, ab) = (a.tag_name.atom(), b.tag_name.atom());
    if aa == NameAtom::UNKNOWN || ab == NameAtom::UNKNOWN {
        return if a.namespace == Namespace::Html {
            a.tag_name
                .as_str()
                .eq_ignore_ascii_case(b.tag_name.as_str())
        } else {
            a.tag_name.as_str() == b.tag_name.as_str()
        };
    }
    aa == ab
}

/// 1-based index of `node` among its same-type element siblings
/// (counting itself). With a shared cache the rank is a binary search
/// over the memoized group; without one it is the direct sibling walk.
/// Non-elements never reach here through the matcher; a defensive 0
/// keeps the callers total (and matches the old behavior, where
/// non-elements also yielded 0).
fn element_index_of_type(sib: Option<&SiblingCache>, doc: &Document, node: NodeId) -> i32 {
    let data = match element_data(doc, node) {
        Some(d) => d,
        None => return 0,
    };
    if let Some(sib) = sib {
        let Some(parent) = doc.parent(node) else {
            return 1;
        };
        let index = sib.index_for(doc, parent);
        let pos = match index.positions.get(&node).copied() {
            Some(p) => p,
            None => return 0,
        };
        // The group's positions ascend, so the rank is a binary search.
        return match index
            .by_type
            .get(&type_key(data))
            .and_then(|v| v.binary_search(&pos).ok())
        {
            Some(rank) => rank as i32 + 1,
            None => 0,
        };
    }
    let mut index = 1;
    let mut s = prev_element_sibling(doc, node);
    while let Some(n) = s {
        // `prev_element_sibling` only yields elements.
        if element_data(doc, n).is_some_and(|d| same_type(data, d)) {
            index += 1;
        }
        s = prev_element_sibling(doc, n);
    }
    index
}

/// 1-based index of `node` among its same-type element siblings,
/// counting from the end. Cached and uncached paths as above.
fn element_index_of_type_from_end(sib: Option<&SiblingCache>, doc: &Document, node: NodeId) -> i32 {
    let data = match element_data(doc, node) {
        Some(d) => d,
        None => return 0,
    };
    if let Some(sib) = sib {
        let Some(parent) = doc.parent(node) else {
            return 1;
        };
        let index = sib.index_for(doc, parent);
        let pos = match index.positions.get(&node).copied() {
            Some(p) => p,
            None => return 0,
        };
        let key = type_key(data);
        let group = match index.by_type.get(&key) {
            Some(g) => g,
            None => return 0,
        };
        return match group.binary_search(&pos) {
            // Last of the group is 1: `len - rank`.
            Ok(rank) => group.len() as i32 - rank as i32,
            Err(_) => 0,
        };
    }
    let mut index = 1;
    let mut s = next_element_sibling(doc, node);
    while let Some(n) = s {
        if element_data(doc, n).is_some_and(|d| same_type(data, d)) {
            index += 1;
        }
        s = next_element_sibling(doc, n);
    }
    index
}

/// Whether the element can take `:enabled`/`:disabled`: the HTML form
/// controls and related elements. Anything else matches neither.
fn is_disableable(data: &ElementData) -> bool {
    data.namespace == Namespace::Html
        && [
            "button", "fieldset", "input", "keygen", "menuitem", "optgroup", "option", "select",
            "textarea",
        ]
        .iter()
        .any(|t| data.tag_name.eq_ignore_ascii_case(t))
}

/// `:disabled`: a disableable element with the `disabled` attribute, or
/// disabled through an ancestor: a control inside a disabled `fieldset`
/// (outside that fieldset's first `legend` child), or an `option` inside
/// a disabled `optgroup`.
fn is_disabled(doc: &Document, node: NodeId) -> bool {
    let data = match element_data(doc, node) {
        Some(d) => d,
        None => return false,
    };
    if !is_disableable(data) {
        return false;
    }
    if attr_value(data, "disabled").is_some() {
        return true;
    }
    disabled_by_ancestor(doc, node, data)
}

/// Ancestor-driven disabledness: disabled `fieldset` / `optgroup`.
fn disabled_by_ancestor(doc: &Document, node: NodeId, data: &ElementData) -> bool {
    let is_option =
        data.namespace == Namespace::Html && data.tag_name.eq_ignore_ascii_case("option");
    let mut anc = doc.parent(node);
    while let Some(a) = anc {
        if let Some(ad) = element_data(doc, a)
            && ad.namespace == Namespace::Html
        {
            if ad.tag_name.eq_ignore_ascii_case("fieldset")
                && attr_value(ad, "disabled").is_some()
                && !in_first_legend(doc, a, node)
            {
                return true;
            }
            if is_option
                && ad.tag_name.eq_ignore_ascii_case("optgroup")
                && attr_value(ad, "disabled").is_some()
            {
                return true;
            }
        }
        anc = doc.parent(a);
    }
    false
}

/// Whether `node` is the fieldset's first `legend` element child or a
/// descendant of it; such nodes escape the fieldset's `disabled`.
fn in_first_legend(doc: &Document, fieldset: NodeId, node: NodeId) -> bool {
    let mut legend = None;
    for ch in children_cursor(doc, fieldset) {
        if let Some(d) = element_data(doc, ch)
            && d.namespace == Namespace::Html
            && d.tag_name.eq_ignore_ascii_case("legend")
        {
            legend = Some(ch);
            break;
        }
    }
    let legend = match legend {
        Some(l) => l,
        None => return false,
    };
    let mut cur = Some(node);
    while let Some(c) = cur {
        if c == legend {
            return true;
        }
        cur = doc.parent(c);
    }
    false
}

/// `:enabled`: a disableable element that is not `:disabled`.
fn is_enabled(doc: &Document, node: NodeId) -> bool {
    match element_data(doc, node) {
        Some(d) => is_disableable(d) && !is_disabled(doc, node),
        None => false,
    }
}

/// Whether `node` is a hyperlink source for `:link`/`:visited`: an `a`,
/// `area`, or `link` element in the HTML namespace with an `href`
/// attribute. Anything else (including `a` without `href`) matches
/// neither pseudo-class.
fn is_html_link(doc: &Document, node: NodeId) -> bool {
    let data = match element_data(doc, node) {
        Some(d) => d,
        None => return false,
    };
    if data.namespace != Namespace::Html {
        return false;
    }
    let tag = data.tag_name.as_str();
    (tag.eq_ignore_ascii_case("a")
        || tag.eq_ignore_ascii_case("area")
        || tag.eq_ignore_ascii_case("link"))
        && attr_value(data, "href").is_some()
}

/// `:checked`: checkbox/radio inputs with the `checked` attribute, and
/// options with the `selected` attribute. Static attribute presence is
/// the signal — there is no live checkedness state yet.
fn is_checked(doc: &Document, node: NodeId) -> bool {
    let data = match element_data(doc, node) {
        Some(d) => d,
        None => return false,
    };
    if data.namespace != Namespace::Html {
        return false;
    }
    if data.tag_name.eq_ignore_ascii_case("input") {
        let checkable = attr_value(data, "type")
            .is_some_and(|t| t.eq_ignore_ascii_case("checkbox") || t.eq_ignore_ascii_case("radio"));
        return checkable && attr_value(data, "checked").is_some();
    }
    if data.tag_name.eq_ignore_ascii_case("option") {
        return attr_value(data, "selected").is_some();
    }
    false
}

/// Whether any relative selector in a `:has()` argument list matches
/// `anchor`.
fn matches_has(
    doc: &Document,
    anchor: NodeId,
    rels: &[RelativeSelector],
    state_for: &dyn Fn(NodeId) -> ElementState,
    sib: Option<&SiblingCache>,
    scratch: &mut MatchScratch,
) -> bool {
    rels.iter()
        .any(|rel| matches_relative(doc, anchor, rel, state_for, sib, scratch))
}

/// Depth-first iterator over a node's descendants (excluding the node
/// itself), allocation-free: it walks [`Document::first_child`] /
/// Lazy pre-order descendant iterator built on [`Document::first_child`]
/// / [`Document::next_sibling`] cursors and climbs back via
/// [`Document::parent`], so per-visited-node cost is O(1) with no `Vec`.
///
/// The cursors filter the template-contents edge (`first_child` skips a
/// leading fragment child, `next_sibling` skips the fragment however it
/// is positioned among its siblings), so an HTML `<template>`'s contents
/// fragment is *not* descended into — the same view as
/// [`Document::children`]. If selector matching should ever style
/// template contents, that is a functional change to this iterator, not
/// a documentation one.
///
/// `Document::descendants` materializes the whole subtree into one `Vec`;
/// this visits nodes one at a time with O(depth) retained state, so
/// `:has()` searches stop at the first witness instead of allocating the
/// subtree up front.
struct Descendants<'a> {
    doc: &'a Document,
    anchor: NodeId,
    /// Next node to yield; `None` once the subtree is exhausted.
    cursor: Option<NodeId>,
}

impl<'a> Descendants<'a> {
    fn new(doc: &'a Document, anchor: NodeId) -> Self {
        Descendants {
            doc,
            anchor,
            cursor: doc.first_child(anchor),
        }
    }
}

impl Iterator for Descendants<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        let n = self.cursor?;
        // Advance: descend into the first child first (document order);
        // when a subtree is exhausted, climb via `parent` and step to the
        // next sibling until a new branch opens or we return to the anchor.
        self.cursor = self.doc.first_child(n).or_else(|| {
            let mut c = n;
            loop {
                if c == self.anchor {
                    return None;
                }
                if let Some(s) = self.doc.next_sibling(c) {
                    return Some(s);
                }
                c = self.doc.parent(c)?;
            }
        });
        Some(n)
    }
}

/// Cursor-based child walk: `first_child` + `next_sibling`, no `Vec`.
/// Visits exactly the nodes [`Document::children`] would return (the
/// template-contents edge is filtered by the cursors: `first_child` skips
/// a leading fragment child and `next_sibling` skips the fragment however
/// it is positioned among its siblings).
fn children_cursor<'a>(doc: &'a Document, node: NodeId) -> impl Iterator<Item = NodeId> + 'a {
    std::iter::successors(doc.first_child(node), |&n| doc.next_sibling(n))
}

/// Evaluate one relative selector against `anchor`, per the Selectors 4
/// "absolutize" rule: `(combinator, S)` behaves like the complex selector
/// `:scope <combinator> S` with `:scope` = `anchor`.
///
/// Depth-first and allocation-free on a warm scratch: for each
/// anchor-relation candidate matching `S`'s leftmost compound,
/// `dfs_relative` matches the remaining compounds and returns at the
/// first witness. Candidates walk the lazy [`Descendants`] /
/// `children_cursor` `first_child` / `next_sibling` cursors — no
/// `Document::children()` `Vec`. Short selectors
/// (`n < MEMO_MIN_COMPOUNDS`) engage no failed-state memo, so they
/// allocate nothing; long ones consult the shared cross-anchor memo in
/// the [`MatchScratch`] to stay O(compounds × subtree²) per anchor and
/// O(compounds × subtree² + anchors × subtree) across the pass. The
/// depth-first search itself is iterative over an explicit [`RelFrame`]
/// heap stack (never native recursion — selector length is
/// attacker-controlled), also taken from the scratch. Nested `:has()` /
/// `:is()` / `:not()` take the emptied scratch, so outer backtrack state
/// is never clobbered.
fn matches_relative(
    doc: &Document,
    anchor: NodeId,
    rel: &RelativeSelector,
    state_for: &dyn Fn(NodeId) -> ElementState,
    sib: Option<&SiblingCache>,
    scratch: &mut MatchScratch,
) -> bool {
    let sel = &rel.selector;
    let n = sel.compounds.len();
    if n == 0 {
        return false;
    }
    if n == 1 {
        // Fast path: a single-compound relative selector is decided by
        // the first witness, so walk lazily instead of building the
        // whole candidate list (`:has(div)`, `:has(> img)`).
        return has_anchor_match(
            doc,
            anchor,
            rel.combinator,
            &sel.compounds[0],
            state_for,
            sib,
            scratch,
        );
    }
    // Multi-compound selectors are matched depth-first, allocation-free
    // on a warm scratch: for each level-0 candidate (matching the
    // leftmost compound in the anchor relation), `dfs_relative`
    // iteratively matches the remaining compounds over an explicit
    // `RelFrame` heap stack (never native recursion: selector length is
    // attacker-controlled). Each level walks lazily, returning at the
    // first witness. No `Vec`/`HashSet` frontier: the boolean result is
    // identical to the level-by-level search (dedup only avoided
    // redundant work, never changed the outcome).
    //
    // Failed-state memoization (`MatchScratch::has_failed`): without it
    // the depth-first search costs O(subtree^compounds) — exponential in
    // the selector length — because every (level, node) subproblem is
    // re-expanded for each ancestor that reaches it. Recording each
    // failure once makes it O(compounds * subtree^2) per anchor. Only
    // engaged for long selectors (see `MEMO_MIN_COMPOUNDS`); short ones
    // are already polynomial and keep the zero-allocation path the
    // `:has()` allocation test pins.
    //
    // The memo is keyed by the relative selector's identity (`rel`'s
    // address — see [`HasFailMemo`]), so failures are shared across every
    // anchor evaluated in the style pass instead of cleared per call:
    // one `:has()` rule over N anchors costs O(16·subtree² + N·subtree)
    // total, not Σ O(16·subtree²). The entries are anchor-independent
    // (the anchor only picks the level-0 candidates; see
    // `MatchScratch::has_failed` for the soundness argument). A distinct
    // relative selector (`:has()` cannot nest — the parser rejects it)
    // keys into a disjoint slot by address, so its bits cannot poison
    // this search's entries.
    //
    // `compounds` here is capped at 16 by the parser
    // (`MAX_HAS_RELATIVE_COMPOUNDS` in `selector.rs`), so the worst case
    // per anchor is O(16 * subtree^2) and the memo holds at most
    // 16 * subtree entries per relative selector. The memo itself stays
    // uncapped — a cap would reintroduce exponential re-expansion (see
    // `MEMO_MIN_COMPOUNDS`).
    let rel_id: Option<usize> =
        (n >= MEMO_MIN_COMPOUNDS).then_some(rel as *const RelativeSelector as usize);
    let level0 = |n: NodeId, scratch: &mut MatchScratch| -> bool {
        matches_compound(doc, n, &sel.compounds[0], state_for, sib, scratch, None)
            && dfs_relative(doc, n, 1, sel, state_for, sib, rel_id, scratch)
    };
    match rel.combinator {
        Combinator::Descendant => {
            Descendants::new(doc, anchor).any(|d| is_element(doc, d) && level0(d, scratch))
        }
        Combinator::Child => {
            children_cursor(doc, anchor).any(|c| is_element(doc, c) && level0(c, scratch))
        }
        Combinator::NextSibling => {
            next_element_sibling(doc, anchor).is_some_and(|s| level0(s, scratch))
        }
        Combinator::SubsequentSibling => {
            let mut nxt = next_element_sibling(doc, anchor);
            let mut found = false;
            while let Some(s) = nxt {
                if level0(s, scratch) {
                    found = true;
                    break;
                }
                nxt = next_element_sibling(doc, s);
            }
            found
        }
    }
}

/// Probe the shared cross-anchor `:has()` failure memo
/// ([`MatchScratch::has_failed`]): true when a previous search — under
/// this or any earlier anchor of the pass — already proved that no
/// match of the relative selector's `compounds[level..]` starts in
/// `node`'s expansion.
fn has_memo_hit(scratch: &mut MatchScratch, rel: usize, level: u32, node: NodeId) -> bool {
    #[cfg(test)]
    {
        scratch.has_memo_probes += 1;
    }
    let hit = scratch.has_failed.probe(rel, level, node);
    #[cfg(test)]
    if hit {
        scratch.has_memo_hits += 1;
    }
    hit
}

/// One depth-first choice point for a relative selector's compound
/// chain: candidates for `compounds[level]` standing in `combinator`
/// relation to `node`. Mirrors [`Frame`] (the `match_complex`
/// backtracking stack) for the forward direction.
///
/// `dfs_relative` used to recurse once per compound; selector compound
/// count is attacker-controlled — the outer complex selector still has
/// no compound cap (only functional-pseudo-class *nesting* is capped,
/// by `MAX_FUNC_NESTING` in `selector.rs`), while `:has()` relative
/// selectors are capped at 16 compounds at parse time
/// (`MAX_HAS_RELATIVE_COMPOUNDS`) — so the heap stack keeps the search
/// iterative at any selector length; it is
/// taken from the [`MatchScratch`] on entry and handed back on exit, so
/// a nested `:has()` takes the emptied scratch and never clobbers an
/// outer search. (The shared failure memo needs no such take/restore:
/// it holds no in-progress state, only monotonic inserts keyed by
/// relative-selector identity — see [`MatchScratch::has_failed`].)
#[derive(Debug)]
struct RelFrame {
    /// Expansion source; candidates stand in `combinator` relation to
    /// it. Keys the failed-state memo when the frame exhausts.
    node: NodeId,
    /// Compound index this frame's candidates are tested against.
    level: usize,
    combinator: Combinator,
    /// Lazy candidate cursor.
    cursor: RelCursor,
}

/// Lazy candidate cursor for one [`RelFrame`].
#[derive(Debug)]
enum RelCursor {
    /// `first_child` / `next_sibling` walk (template-fragment edges are
    /// filtered by the cursors themselves, exactly like
    /// [`children_cursor`]).
    Children(Option<NodeId>),
    /// Pre-order subtree walk; the inline state mirrors [`Descendants`]
    /// (see [`next_descendant`]).
    Descendants {
        anchor: NodeId,
        cursor: Option<NodeId>,
    },
    /// `next_element_sibling` chain (already element-filtered).
    Siblings(Option<NodeId>),
}

impl RelFrame {
    fn new(doc: &Document, node: NodeId, level: usize, combinator: Combinator) -> Self {
        let cursor = match combinator {
            Combinator::Child => RelCursor::Children(doc.first_child(node)),
            Combinator::Descendant => RelCursor::Descendants {
                anchor: node,
                cursor: doc.first_child(node),
            },
            Combinator::NextSibling | Combinator::SubsequentSibling => {
                RelCursor::Siblings(next_element_sibling(doc, node))
            }
        };
        RelFrame {
            node,
            level,
            combinator,
            cursor,
        }
    }

    /// The next candidate, nearest-first; `None` when exhausted.
    /// Non-element children/descendants are skipped here: they could
    /// never match (candidates are always tested with
    /// [`matches_compound`], which only matches elements), so skipping
    /// is behavior-identical to the old `is_element` filter at the
    /// `.any()` site.
    fn next_candidate(&mut self, doc: &Document) -> Option<NodeId> {
        match &mut self.cursor {
            RelCursor::Children(cursor) => {
                while let Some(c) = cursor.take() {
                    *cursor = doc.next_sibling(c);
                    if is_element(doc, c) {
                        return Some(c);
                    }
                }
                None
            }
            RelCursor::Descendants { anchor, cursor } => {
                while let Some(d) = next_descendant(doc, *anchor, cursor) {
                    if is_element(doc, d) {
                        return Some(d);
                    }
                }
                None
            }
            RelCursor::Siblings(cursor) => {
                let s = cursor.take()?;
                if self.combinator == Combinator::SubsequentSibling {
                    *cursor = next_element_sibling(doc, s);
                }
                Some(s)
            }
        }
    }
}

/// One step of the pre-order subtree walk behind [`Descendants`],
/// factored out so [`RelFrame`]'s cursor can advance from `&mut self`
/// with the [`Document`] passed in: the [`RelFrame`] stack lives in the
/// [`MatchScratch`], which is not lifetime-parameterized and so cannot
/// hold the borrowing [`Descendants`] iterator itself. Behavior is
/// exactly [`Descendants`]'s: document order, stopping at the anchor,
/// template-fragment edges filtered by the cursors.
fn next_descendant(doc: &Document, anchor: NodeId, cursor: &mut Option<NodeId>) -> Option<NodeId> {
    let n = cursor.take()?;
    // Advance: descend into the first child first (document order);
    // when a subtree is exhausted, climb via `parent` and step to the
    // next sibling until a new branch opens or we return to the anchor.
    *cursor = doc.first_child(n).or_else(|| {
        let mut c = n;
        loop {
            if c == anchor {
                return None;
            }
            if let Some(s) = doc.next_sibling(c) {
                return Some(s);
            }
            c = doc.parent(c)?;
        }
    });
    Some(n)
}

/// Depth-first match of a relative selector's compounds `level..`,
/// starting from `node` (which matched `compounds[level - 1]`).
/// Expands `node` through `combinators[level - 1]`; each candidate
/// matching `compounds[level]` is a witness (last level) or descended
/// into. Every expansion walks lazily with early exit, and the search
/// is iterative over an explicit [`RelFrame`] heap stack — never native
/// recursion — so selector length cannot overflow the stack (see
/// [`RelFrame`]).
///
/// `rel_id` is the shared failure memo's key namespace for this relative
/// selector (`Some` address from [`matches_relative`], `None` keeps the
/// memo off entirely for short selectors). Failures are recorded into
/// [`MatchScratch::has_failed`] and never cleared here: they are
/// anchor-independent (see the field docs), so later anchors of the same
/// style pass reuse them instead of re-expanding.
///
/// Eight parameters is over clippy's default limit, but each has a
/// distinct role (`doc`/`sel`/`state_for`/`sib` are the fixed match
/// context, `node`/`level` the search position, `rel_id` the memo
/// namespace, `scratch` the shared buffers); bundling them would only
/// obscure the search.
#[allow(clippy::too_many_arguments)]
fn dfs_relative(
    doc: &Document,
    node: NodeId,
    level: usize,
    sel: &Selector,
    state_for: &dyn Fn(NodeId) -> ElementState,
    sib: Option<&SiblingCache>,
    rel_id: Option<usize>,
    scratch: &mut MatchScratch,
) -> bool {
    // A previous search — under this anchor or an earlier one — already
    // proved that no match of compounds[level..] starts in this
    // expansion: skip it outright. Sound across anchors: the subproblem
    // depends only on `doc`, the relative selector, `state_for`, and
    // `sib`, all fixed for the whole style pass (see
    // `MatchScratch::has_failed`); the anchor only chose the level-0
    // candidates. (The old per-call version checked this on entry to
    // every call; the iterative version checks it before pushing each
    // frame.)
    if rel_id.is_some_and(|id| has_memo_hit(scratch, id, level as u32, node)) {
        return false;
    }
    // Explicit heap stack instead of one native frame per compound
    // (see `RelFrame`): taken from the shared scratch and handed back
    // on exit, so a nested `:has()` search takes the emptied scratch
    // and never clobbers this search. The shared failure memo is keyed
    // by relative-selector identity, so the nested search's entries
    // live in a disjoint namespace.
    let mut stack: Vec<RelFrame> = std::mem::take(&mut scratch.rel_frames);
    stack.clear();
    stack.push(RelFrame::new(doc, node, level, sel.combinators[level - 1]));
    #[cfg(test)]
    {
        scratch.rel_expansions += 1;
    }
    let hit = 'search: loop {
        // Probe the top frame's cursor; the borrow ends with the call,
        // so the stack can be pushed below.
        let cand = match stack.last_mut() {
            Some(frame) => frame.next_candidate(doc),
            None => break 'search false,
        };
        match cand {
            Some(cand) => {
                // The stack is non-empty here: the candidate just came
                // from its top frame.
                let level = stack.last().expect("dfs stack non-empty").level;
                if !matches_compound(
                    doc,
                    cand,
                    &sel.compounds[level],
                    state_for,
                    sib,
                    scratch,
                    None,
                ) {
                    continue 'search;
                }
                if level == sel.compounds.len() - 1 {
                    break 'search true;
                }
                // Memo entry check for the child subproblem, mirroring
                // the entry check above.
                if rel_id.is_some_and(|id| has_memo_hit(scratch, id, (level + 1) as u32, cand)) {
                    continue 'search;
                }
                stack.push(RelFrame::new(doc, cand, level + 1, sel.combinators[level]));
                #[cfg(test)]
                {
                    scratch.rel_expansions += 1;
                }
            }
            None => {
                let frame = stack.pop().expect("dfs stack non-empty");
                // The frame's expansion was exhaustive (a candidate is
                // only skipped after failing, and the loop returns true
                // at the first witness), so recording the failure is
                // sound for the rest of the pass: an identical
                // subproblem can never succeed. Entries are never
                // evicted — eviction would re-expand them exponentially
                // (see `MEMO_MIN_COMPOUNDS`).
                if let Some(id) = rel_id {
                    scratch
                        .has_failed
                        .record(id, frame.level as u32, frame.node);
                }
            }
        }
    };
    // Hand the stack back to the scratch for the next match.
    scratch.rel_frames = stack;
    hit
}

/// Whether any element standing in the anchor relation for a relative
/// selector's leading combinator matches `compound`. Lazy: stops at the
/// first witness instead of collecting every candidate up front, so
/// `:has()` on a large subtree does not materialize it.
///
/// The anchor itself never qualifies for the descendant combinator
/// (`:scope div` needs a *descendant* div); the sibling combinators
/// start from the anchor's own siblings.
fn has_anchor_match(
    doc: &Document,
    anchor: NodeId,
    comb: Combinator,
    compound: &CompoundSelector,
    state_for: &dyn Fn(NodeId) -> ElementState,
    sib: Option<&SiblingCache>,
    scratch: &mut MatchScratch,
) -> bool {
    match comb {
        Combinator::Child => children_cursor(doc, anchor).any(|c| {
            is_element(doc, c) && matches_compound(doc, c, compound, state_for, sib, scratch, None)
        }),
        Combinator::Descendant => {
            // Iterative DFS with early exit over the lazy Descendants
            // iterator: no whole-subtree allocation.
            Descendants::new(doc, anchor).any(|n| {
                is_element(doc, n)
                    && matches_compound(doc, n, compound, state_for, sib, scratch, None)
            })
        }
        Combinator::NextSibling => next_element_sibling(doc, anchor)
            .is_some_and(|s| matches_compound(doc, s, compound, state_for, sib, scratch, None)),
        Combinator::SubsequentSibling => {
            let mut nxt = next_element_sibling(doc, anchor);
            while let Some(s) = nxt {
                if matches_compound(doc, s, compound, state_for, sib, scratch, None) {
                    return true;
                }
                nxt = next_element_sibling(doc, s);
            }
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selector::parse_selector_list;
    use bowser_dom::Attributes;

    fn sel(input: &str) -> Selector {
        let list = parse_selector_list(input).expect("selector should parse");
        assert_eq!(list.len(), 1);
        list.into_iter().next().unwrap()
    }

    fn m(input: &str, doc: &Document, node: NodeId) -> bool {
        matches(&sel(input), doc, node)
    }

    /// `<div id="d" class="a B" data-x="hello world" lang="en-US"><p id="p1">x</p><p id="p2">y</p><span id="s"></span></div>`
    fn sample() -> (Document, NodeId, NodeId, NodeId, NodeId) {
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        doc.set_attribute(div, "id", "d");
        doc.set_attribute(div, "class", "a B");
        doc.set_attribute(div, "data-x", "hello world");
        doc.set_attribute(div, "lang", "en-US");
        let p1 = doc.create_element("p");
        doc.append_child(div, p1);
        doc.set_attribute(p1, "id", "p1");
        let t = doc.create_text_node("x");
        doc.append_child(p1, t);
        let p2 = doc.create_element("p");
        doc.append_child(div, p2);
        doc.set_attribute(p2, "id", "p2");
        let span = doc.create_element("span");
        doc.append_child(div, span);
        doc.set_attribute(span, "id", "s");
        (doc, div, p1, p2, span)
    }

    /// A chain of `depth` nested divs under the document root; returns the
    /// document, the top div, and the deepest div.
    fn deep_div_chain(depth: usize) -> (Document, NodeId, NodeId) {
        let mut doc = Document::new();
        let root = doc.root();
        let top = doc.create_element("div");
        doc.append_child(root, top);
        let mut parent = top;
        for _ in 0..depth {
            let d = doc.create_element("div");
            doc.append_child(parent, d);
            parent = d;
        }
        (doc, top, parent)
    }

    /// Hostile backtracking: the rightmost `div` matches the deepest node
    /// and every middle compound matches every ancestor, but the leftmost
    /// `zzz` can never match, so the search must exhaust every witness
    /// combination — C(depth, compounds-1) without failed-state
    /// memoization. With the memo each (compound, node) pair expands once
    /// and the match finishes in milliseconds.
    #[test]
    fn descendant_backtracking_hostile_is_polynomial() {
        let (doc, _top, deepest) = deep_div_chain(150);
        let sel = "zzz div div div div div div div div div div div";
        let start = std::time::Instant::now();
        assert!(!m(sel, &doc, deepest));
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(10),
            "hostile descendant selector took {elapsed:?}"
        );
        // The same chain with a matchable leftmost compound still
        // matches (memoization must not change outcomes).
        let sel = "div div div div div div div div div div div div";
        assert!(m(sel, &doc, deepest));
    }

    /// Hostile backtracking at the parse cap: 64 compounds (the maximum
    /// the parser accepts), the leftmost unmatchable, over a 400-deep
    /// chain. Pins the capped worst case: the uncapped per-call
    /// failed-state memo holds at most 64 x nodes entries here and the
    /// search stays polynomial, instead of the pre-cap shape where a
    /// 10^5-compound selector pinned ~2 GB transient per anchor.
    #[test]
    fn capped_complex_chain_hostile_is_fast() {
        let (doc, _top, deepest) = deep_div_chain(400);
        let miss = format!("zzz {}", vec!["div"; 63].join(" "));
        let start = std::time::Instant::now();
        assert!(!m(&miss, &doc, deepest));
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(30),
            "capped hostile complex selector took {elapsed:?}"
        );
        // A matchable 64-compound tail still finds its witness quickly.
        let hit = vec!["div"; 64].join(" ");
        let start = std::time::Instant::now();
        assert!(m(&hit, &doc, deepest));
        assert!(
            start.elapsed() < std::time::Duration::from_secs(30),
            "capped complex hit took too long"
        );
    }

    /// Hostile `:has()` backtracking: 8 compounds, the last unmatchable,
    /// over a deep chain. The depth-first relative search re-expands
    /// every (level, node) subproblem once per ancestor that reaches it
    /// without the failed-state memo; with it, each is computed once.
    #[test]
    fn has_backtracking_hostile_is_polynomial() {
        let (doc, anchor, _deepest) = deep_div_chain(100);
        let start = std::time::Instant::now();
        assert!(!m("div:has(div div div div div div div zzz)", &doc, anchor));
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(10),
            "hostile :has() selector took {elapsed:?}"
        );
        // A matchable tail still finds its witness.
        assert!(m("div:has(div div div div div div div)", &doc, anchor));
    }

    /// Hostile `:has()` at the parse cap: 16 compounds (the maximum the
    /// parser accepts), the last unmatchable, over a 400-deep chain. The
    /// per-anchor worst case is O(16 x subtree^2) with the uncapped memo
    /// holding at most 16 x subtree entries; this pins that the capped
    /// worst case completes fast instead of degrading.
    #[test]
    fn has_capped_compound_chain_hostile_is_fast() {
        let (doc, anchor, _deepest) = deep_div_chain(400);
        let miss = format!("div:has({} zzz)", vec!["div"; 15].join(" "));
        let start = std::time::Instant::now();
        assert!(!m(&miss, &doc, anchor));
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(30),
            "capped hostile :has() selector took {elapsed:?}"
        );
        // A matchable 16-compound tail still finds its witness quickly.
        let hit = format!("div:has({})", vec!["div"; 16].join(" "));
        let start = std::time::Instant::now();
        assert!(m(&hit, &doc, anchor));
        assert!(
            start.elapsed() < std::time::Duration::from_secs(30),
            "capped :has() hit took too long"
        );
    }

    /// Hostile `:has()` over a huge subtree: 3000 elements under one
    /// anchor with a 16-compound miss selector. The lazy descendant walk
    /// stops at witnesses and the capped search stays polynomial; the
    /// whole match must finish well inside the time bound.
    #[test]
    fn has_huge_subtree_hostile_is_fast() {
        let mut doc = Document::new();
        let root = doc.root();
        let anchor = doc.create_element("div");
        doc.append_child(root, anchor);
        // A wide subtree: 3000 children, each with one nested div, so the
        // descendant walk has real work at two levels.
        for _ in 0..3000 {
            let c = doc.create_element("div");
            doc.append_child(anchor, c);
            let g = doc.create_element("div");
            doc.append_child(c, g);
        }
        let miss = format!("div:has({} zzz)", vec!["div"; 15].join(" "));
        let start = std::time::Instant::now();
        assert!(!m(&miss, &doc, anchor));
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(30),
            "huge-subtree :has() took {elapsed:?}"
        );
    }

    /// A chain of `depth` nested divs under the document root; returns
    /// the document and every div, top-first — every div is an anchor
    /// the cascade would evaluate a `:has()` rule against.
    fn div_chain_all(depth: usize) -> (Document, Vec<NodeId>) {
        let mut doc = Document::new();
        let root = doc.root();
        let mut divs = Vec::with_capacity(depth);
        let mut parent = root;
        for _ in 0..depth {
            let d = doc.create_element("div");
            doc.append_child(parent, d);
            divs.push(d);
            parent = d;
        }
        (doc, divs)
    }

    /// The cross-anchor `:has()` failure memo: the second anchor's
    /// evaluation reuses failures recorded by the first anchor's,
    /// instead of re-expanding them.
    ///
    /// Deterministic (no timing): on a deep div chain with a 6-compound
    /// `:has()` miss, the first anchor's search expands every (level,
    /// node) subproblem once; the second anchor — whose whole subtree
    /// was already proven failing by the first — performs zero search
    /// expansions and serves every memo probe as a hit.
    #[test]
    fn has_failure_memo_shared_across_anchors() {
        let (doc, divs) = div_chain_all(300);
        // 6 compounds: memo-engaged (`MEMO_MIN_COMPOUNDS`), last
        // compound unmatchable so every search exhausts.
        let selector = sel("div:has(div div div div div zzz)");
        let state_for = |_: NodeId| ElementState::default();
        let mut scratch = MatchScratch::new();

        // Anchor 1 (top of the chain): full search, populates the memo.
        assert!(!matches_cascade(
            &selector,
            &doc,
            divs[0],
            &state_for,
            None,
            &mut scratch,
            None
        ));
        let (_probes1, _hits1, expansions1) = scratch.memo_counters();
        assert!(expansions1 > 0, "first anchor should expand the search");
        assert!(
            scratch.has_failed_len() > 0,
            "first anchor should memoize failures"
        );

        // Anchor 2: its entire subtree was already proven failing by
        // anchor 1, so the shared memo serves every probe — no
        // re-expansion. (A per-call memo would expand ~as much as
        // anchor 1 did: the subtrees differ by one node.)
        scratch.reset_memo_counters();
        assert!(!matches_cascade(
            &selector,
            &doc,
            divs[1],
            &state_for,
            None,
            &mut scratch,
            None
        ));
        let (probes2, hits2, expansions2) = scratch.memo_counters();
        assert!(probes2 > 0, "second anchor should consult the memo");
        assert!(
            hits2 > 0,
            "second anchor should reuse the first anchor's failures"
        );
        assert_eq!(
            expansions2, 0,
            "second anchor must not re-expand memoized failures"
        );

        // Sharing must not change outcomes: a fresh scratch (no shared
        // memo) agrees on every anchor, and a *hitting* selector still
        // finds its witness through the shared memo — the memo is keyed
        // by relative-selector identity, so the miss selector's
        // failures cannot shadow the hit selector's search.
        let mut fresh = MatchScratch::new();
        for &d in &divs {
            assert!(!matches_cascade(
                &selector, &doc, d, &state_for, None, &mut fresh, None
            ));
        }
        let hit = sel("div:has(div div div div div div)");
        assert!(matches_cascade(
            &hit,
            &doc,
            divs[0],
            &state_for,
            None,
            &mut scratch,
            None
        ));
        assert!(matches_cascade(
            &hit,
            &doc,
            divs[1],
            &state_for,
            None,
            &mut scratch,
            None
        ));
    }

    /// Hostile N-anchor `:has()`: one 16-compound (parse-cap) miss rule
    /// evaluated against every div of a deep chain — the shape the
    /// cascade actually pays (one `:has()` rule x N anchors), which the
    /// earlier hostile tests never pinned (they measure single anchors).
    ///
    /// Without cross-anchor sharing this is Σ O(16·mᵢ²) — cubic in the
    /// chain depth (≈10⁹ compound checks at depth 500: tens of minutes
    /// in debug). With the shared failure memo the first anchor pays
    /// O(16·m²) once and every later anchor is O(m) memo probes, so the
    /// total is ~linear in N·m. A deep `<div>` chain is a few dozen KB
    /// of hostile HTML and the parser is iterative, so this shape is
    /// reachable.
    #[test]
    fn has_n_anchor_deep_chain_miss_is_linear() {
        const DEPTH: usize = 500;
        let (doc, divs) = div_chain_all(DEPTH);
        let miss = format!("div:has({} zzz)", vec!["div"; 15].join(" "));
        let selector = sel(&miss);
        let state_for = |_: NodeId| ElementState::default();
        // One pass-owned scratch, exactly like the cascade's `Styler`:
        // the failure memo persists across the per-anchor evaluations.
        let mut scratch = MatchScratch::new();
        let start = std::time::Instant::now();
        for &d in &divs {
            assert!(
                !matches_cascade(&selector, &doc, d, &state_for, None, &mut scratch, None),
                ":has() miss must stay a miss under the shared memo"
            );
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(90),
            "N-anchor :has() miss took {elapsed:?}: cross-anchor sharing regressed"
        );
    }

    /// `clear_has_memo` drops the shared `:has()` failure memo while
    /// keeping its capacity: the cascade calls it at style-pass
    /// boundaries (every interactive-state epoch bump), so entries
    /// recorded under one set of element states can never shadow a
    /// later pass.
    #[test]
    fn clear_has_memo_empties_shared_memo() {
        let (doc, divs) = div_chain_all(50);
        let selector = sel("div:has(div div div div div zzz)");
        let state_for = |_: NodeId| ElementState::default();
        let mut scratch = MatchScratch::new();
        assert!(!matches_cascade(
            &selector,
            &doc,
            divs[0],
            &state_for,
            None,
            &mut scratch,
            None
        ));
        assert!(scratch.has_failed_len() > 0);
        let bytes = scratch.has_failed_bits_capacity_bytes();
        scratch.clear_has_memo();
        assert_eq!(scratch.has_failed_len(), 0);
        assert!(
            scratch.has_failed_bits_capacity_bytes() >= bytes,
            "clear must keep bitset capacity for the next pass"
        );
    }

    /// Retention hygiene: past [`HAS_MEMO_RETENTION_BYTES`] of reserved
    /// bitset backing, [`clear_has_memo`](MatchScratch::clear_has_memo)
    /// releases the backing instead of pinning it in the thread-local
    /// scratch forever after a hostile one-shot call. Small memos keep
    /// their warm capacity (pinned by
    /// `clear_has_memo_empties_shared_memo` above).
    #[test]
    fn clear_has_memo_shrinks_past_retention_ceiling() {
        let mut scratch = MatchScratch::new();
        // Simulate a hostile one-shot: bitset backing past the ceiling.
        // (Pushed directly: the bitsets size by node-arena index, so a
        // real document this large is not needed to exercise the
        // release path.)
        let words = HAS_MEMO_RETENTION_BYTES / 8 + 64;
        scratch.has_failed.bits.push(vec![0u64; words]);
        assert!(scratch.has_failed_bits_capacity_bytes() > HAS_MEMO_RETENTION_BYTES);
        scratch.clear_has_memo();
        assert_eq!(scratch.has_failed_len(), 0);
        assert_eq!(
            scratch.has_failed_bits_capacity_bytes(),
            0,
            "over-ceiling backing must be released, not retained"
        );
        // The scratch is still usable afterwards.
        let (doc, divs) = div_chain_all(10);
        let selector = sel("div div");
        let state_for = |_: NodeId| ElementState::default();
        assert!(matches_cascade(
            &selector,
            &doc,
            divs[9],
            &state_for,
            None,
            &mut scratch,
            None
        ));
    }

    /// Bitset memo: a hostile 16-compound (parse-cap) `:has()` relative
    /// selector over a deep chain must (a) return exactly the same
    /// results as a fresh scratch per anchor — the bitset records and
    /// replays the same failures the old `HashSet<HasFailKey>` did — and
    /// (b) grow bounded: one bit per (level, node), never one hash key
    /// per entry. At depth 200 the old memo held ~16·200 entries at
    /// ~28.6 B amortized (~91 KB); the bitsets hold 16 · 200 bits
    /// (3.2 KB). The bound asserted here is structural: 16 levels × N
    /// nodes, one bit each, plus one word of slack for lazy sizing.
    #[test]
    fn has_bitset_memo_same_results_and_bounded_memory() {
        const DEPTH: usize = 200;
        let (doc, divs) = div_chain_all(DEPTH);
        let state_for = |_: NodeId| ElementState::default();

        // Two selectors: a miss (hostile, exhausts the search) and a
        // hit (witness at the bottom of the chain), so both the
        // failure-recording and the early-exit paths run through the
        // bitset.
        let miss = sel(&format!("div:has({} zzz)", vec!["div"; 15].join(" ")));
        let hit = sel(&format!("div:has({})", vec!["div"; 15].join(" ")));

        // Baseline: one fresh scratch per anchor — no cross-anchor
        // sharing at all.
        let mut baseline = Vec::with_capacity(divs.len() * 2);
        for selector in [&miss, &hit] {
            for &d in &divs {
                let mut fresh = MatchScratch::new();
                baseline.push(matches_cascade(
                    selector, &doc, d, &state_for, None, &mut fresh, None,
                ));
            }
        }

        // One pass-owned scratch: failures recorded by the first anchor
        // are replayed as bit tests for every later anchor.
        let mut scratch = MatchScratch::new();
        let mut results = Vec::with_capacity(divs.len() * 2);
        for selector in [&miss, &hit] {
            for &d in &divs {
                results.push(matches_cascade(
                    selector,
                    &doc,
                    d,
                    &state_for,
                    None,
                    &mut scratch,
                    None,
                ));
            }
        }
        assert_eq!(
            results, baseline,
            "bitset memo changed match results vs unshared baseline"
        );
        assert!(
            results.iter().any(|&r| r),
            "the hit selector must match somewhere in the chain"
        );

        // Bounded growth: entries ≤ 16 levels × R selectors × N nodes,
        // and the bits themselves are ≤ 16 bitsets of N bits per
        // selector (+ one word of slack per bitset for lazy sizing).
        // Two selectors ran above (miss + hit); the nested selector
        // runs after this check.
        let n = DEPTH;
        assert!(
            scratch.has_failed_len() <= 2 * 16 * n,
            "memo entries {} exceed 2·16·N = {}",
            scratch.has_failed_len(),
            2 * 16 * n
        );
        let bit_bytes = scratch.has_failed_bits_bytes();
        assert!(
            bit_bytes <= 2 * 16 * (n.div_ceil(8)) + 2 * 16 * 8,
            "memo bitsets use {bit_bytes} bytes: bit-per-entry bound violated"
        );

        // Disjoint namespaces: the miss and hit selectors are distinct
        // relative selectors, so they mint distinct rel slots (two
        // addresses → two slots) and neither shadows the other — the
        // identical-to-baseline results above already prove no
        // cross-talk, and this pins the slot accounting.
        assert_eq!(
            scratch.has_failed_rel_count(),
            2,
            "two distinct long relative selectors must mint two rel slots"
        );
    }

    #[test]
    fn type_universal_id_class() {
        let (doc, div, p1, _, _) = sample();
        assert!(m("div", &doc, div));
        assert!(m("DIV", &doc, div)); // HTML type: ASCII case-insensitive
        assert!(!m("span", &doc, div));
        assert!(m("*", &doc, div));
        assert!(m("#d", &doc, div));
        assert!(!m("#p1", &doc, div));
        assert!(m(".a", &doc, div));
        // Standards mode: class matching is case-sensitive (was ASCII
        // case-insensitive before the fix).
        assert!(!m(".b", &doc, div));
        assert!(m(".B", &doc, div));
        assert!(!m(".c", &doc, div));
        assert!(!m("div#d.a", &doc, p1));
        assert!(m("p#p1", &doc, p1));
        // Non-elements never match.
        assert!(!m("*", &doc, doc.root()));
    }

    #[test]
    fn foreign_case_sensitivity() {
        let mut doc = Document::new();
        let root = doc.root();
        let el = doc.create_element_ns(Namespace::Svg, "foreignObject");
        doc.append_child(root, el);
        doc.set_attribute(el, "class", "AbC");
        assert!(m("foreignObject", &doc, el));
        assert!(!m("foreignobject", &doc, el)); // foreign: case-sensitive
        assert!(m("*", &doc, el));
        assert!(m(".AbC", &doc, el));
        assert!(!m(".abc", &doc, el)); // foreign class: case-sensitive
    }

    #[test]
    fn class_matching_is_case_sensitive() {
        let mut doc = Document::new();
        let root = doc.root();
        let el = doc.create_element("div");
        doc.append_child(root, el);
        doc.set_attribute(el, "class", "Foo bar");
        assert!(m(".Foo", &doc, el));
        assert!(m(".bar", &doc, el));
        assert!(!m(".foo", &doc, el));
        assert!(!m(".BAR", &doc, el));
        assert!(!m(".baz", &doc, el));
    }

    #[test]
    fn nth_child_extreme_coefficients() {
        // `index - b` overflows i32 for b = i32::MIN; must not panic.
        assert!(nth_matches(1, i32::MIN, 5));
        assert!(nth_matches(1, i32::MIN, 1));
        assert!(!nth_matches(2, i32::MIN, 5));
        assert!(nth_matches(-1, i32::MAX, 1)); // -n+MAX matches everything >= 1
        assert!(nth_matches(-1, i32::MAX, 2));
        assert!(nth_matches(-1, i32::MAX, i32::MAX));
        assert!(!nth_matches(i32::MIN, 0, 1)); // 1 = MIN*k has no solution
        // End to end: `:nth-child(n-2147483648)` parses and matches.
        let (doc, _, p1, _, _) = sample();
        assert!(m(":nth-child(n-2147483648)", &doc, p1));
    }

    #[test]
    fn empty_rejects_any_text() {
        // Selectors 4: any non-empty text node defeats :empty — even
        // whitespace-only text (comments and PIs are still ignored).
        let mut doc = Document::new();
        let root = doc.root();
        for ws in [" ", "\t", "\n", "\r", "\x0C", " \t\n\r\x0C ", "\u{a0}"] {
            let el = doc.create_element("div");
            doc.append_child(root, el);
            let t = doc.create_text_node(ws);
            doc.append_child(el, t);
            assert!(!m("div:empty", &doc, el), "{ws:?} should not be :empty");
        }
        // The empty text node does not defeat :empty.
        let el = doc.create_element("div");
        doc.append_child(root, el);
        let t = doc.create_text_node("");
        doc.append_child(el, t);
        assert!(m("div:empty", &doc, el));
    }

    #[test]
    fn namespace_matching() {
        let (doc, div, _, _, _) = sample(); // HTML div
        // `svg|div` must not match an HTML div.
        assert!(!m("svg|div", &doc, div));
        assert!(!m("svg|*", &doc, div));
        assert!(!m("foo|div", &doc, div)); // unknown prefix never matches
        assert!(!m("|div", &doc, div)); // no-namespace never matches here
        // `*|div` and unprefixed forms match.
        assert!(m("*|div", &doc, div));
        assert!(m("*|*", &doc, div));
        assert!(m("html|div", &doc, div));
        assert!(m("div", &doc, div));
        assert!(!m("svg|span", &doc, div));
        // Attribute namespaces: attributes carry no namespace, so
        // `[*|id]`, `[id]`, and `[|id]` (no namespace) all match, while a
        // named prefix never does — even when it resolves (`[html|id]`)
        // and even on foreign elements (`[svg|viewBox]`).
        assert!(m("[*|id]", &doc, div));
        assert!(m("[id]", &doc, div));
        assert!(m("[|id]", &doc, div));
        assert!(!m("[html|id=d]", &doc, div));
        assert!(!m("[svg|id]", &doc, div));
        assert!(!m("[foo|id]", &doc, div)); // unknown prefix never matches
        // SVG element: type selectors still test the element's namespace,
        // but attribute selectors never match a named prefix (attributes
        // carry no namespace even on foreign elements).
        let mut doc2 = Document::new();
        let root2 = doc2.root();
        let rect = doc2.create_element_ns(Namespace::Svg, "rect");
        doc2.append_child(root2, rect);
        doc2.set_attribute(rect, "viewBox", "0 0 1 1");
        assert!(m("svg|rect", &doc2, rect));
        assert!(m("svg|*", &doc2, rect));
        assert!(m("*|rect", &doc2, rect));
        assert!(!m("html|rect", &doc2, rect));
        assert!(!m("div", &doc2, rect));
        assert!(!m("[svg|viewBox]", &doc2, rect));
        assert!(!m("[html|viewBox]", &doc2, rect));
        assert!(m("[*|viewBox]", &doc2, rect));
        assert!(m("[|viewBox]", &doc2, rect));
    }

    #[test]
    fn attribute_operators() {
        let (doc, div, _, _, _) = sample();
        assert!(m("[id]", &doc, div));
        assert!(!m("[title]", &doc, div));
        assert!(m("[id=d]", &doc, div));
        assert!(!m("[id=e]", &doc, div));
        assert!(m("[class~=a]", &doc, div));
        assert!(m("[class~=B]", &doc, div));
        assert!(!m("[class~=c]", &doc, div));
        assert!(m("[data-x~=hello]", &doc, div));
        assert!(m("[lang|=en]", &doc, div));
        assert!(!m("[lang|=US]", &doc, div));
        assert!(m("[id^=d]", &doc, div));
        assert!(!m("[id^=x]", &doc, div));
        assert!(m("[lang$=US]", &doc, div));
        assert!(m("[data-x*=\"llo wo\"]", &doc, div));
        assert!(!m("[data-x*=zzz]", &doc, div));
        // Empty substring/prefix/suffix values never match.
        assert!(!m("[id*=\"\"]", &doc, div));
        assert!(!m("[id^=\"\"]", &doc, div));
        assert!(!m("[id$=\"\"]", &doc, div));
        // Case flags.
        assert!(m("[id=D i]", &doc, div));
        assert!(!m("[id=E i]", &doc, div));
        assert!(!m("[id=D s]", &doc, div));
        assert!(m("[id=d s]", &doc, div));
        assert!(m("[lang^=EN i]", &doc, div));
    }

    #[test]
    fn combinators() {
        let (doc, _div, p1, p2, span) = sample();
        assert!(m("div p", &doc, p1));
        assert!(m("div p", &doc, p2));
        assert!(!m("span p", &doc, p1));
        assert!(m("div > p", &doc, p1));
        assert!(m("div > span", &doc, span));
        assert!(!m("div > p > span", &doc, span));
        assert!(m("p + p", &doc, p2));
        assert!(!m("p + p", &doc, p1));
        assert!(m("p + span", &doc, span));
        assert!(m("p ~ span", &doc, span));
        assert!(m("p#p1 ~ span", &doc, span));
        assert!(!m("span ~ p", &doc, p2));
        assert!(!m("div + p", &doc, p1));
        // Chains.
        assert!(m("div > p#p1 + p#p2", &doc, p2));
        assert!(m("div p#p2", &doc, p2));
        // Deep descendant: grandchild.
        let mut doc2 = Document::new();
        let root = doc2.root();
        let a = doc2.create_element("div");
        doc2.append_child(root, a);
        let b = doc2.create_element("section");
        doc2.append_child(a, b);
        let c = doc2.create_element("p");
        doc2.append_child(b, c);
        assert!(m("div p", &doc2, c));
        assert!(!m("div > p", &doc2, c));
        assert!(m("div section > p", &doc2, c));
    }

    #[test]
    fn structural_pseudo_classes() {
        let (doc, div, p1, p2, span) = sample();
        assert!(m("p:first-child", &doc, p1));
        assert!(!m("p:first-child", &doc, p2));
        assert!(m("span:last-child", &doc, span));
        assert!(!m("p:last-child", &doc, p1));
        assert!(m("p:nth-child(1)", &doc, p1));
        assert!(m("p:nth-child(2)", &doc, p2));
        assert!(m("p:nth-child(odd)", &doc, p1));
        assert!(m("p:nth-child(even)", &doc, p2));
        assert!(m(":nth-child(2n+1)", &doc, span)); // span is 3rd element child
        assert!(!m("span:nth-child(2n)", &doc, span));
        assert!(m("span:nth-child(-n+3)", &doc, span));
        assert!(!m("span:nth-child(-n+2)", &doc, span));
        assert!(!m("div:empty", &doc, div));
        assert!(m("span:empty", &doc, span));
        assert!(!m("p:empty", &doc, p1)); // has text
    }

    #[test]
    fn empty_ignores_whitespace_and_comments() {
        let mut doc = Document::new();
        let root = doc.root();
        let a = doc.create_element("div");
        doc.append_child(root, a);
        assert!(m("div:empty", &doc, a));
        let ws = doc.create_text_node("  \n  ");
        doc.append_child(a, ws);
        assert!(!m("div:empty", &doc, a)); // whitespace-only text defeats :empty
        // Comments don't affect :empty either.
        let (doc2, _) = Document::parse_html(b"<div><!-- hi --></div>");
        let div2 = doc2.get_elements_by_tag_name("div").first().unwrap();
        assert!(m("div:empty", &doc2, div2));
    }

    #[test]
    fn root_pseudo_class() {
        let (doc, html) = parsed_doc();
        assert!(m(":root", &doc, html));
        assert!(!m("div:root", &doc, html));
    }

    #[test]
    fn not_pseudo_class() {
        let (doc, div, p1, p2, _) = sample();
        assert!(m("p:not(#p2)", &doc, p1));
        assert!(!m("p:not(#p2)", &doc, p2));
        assert!(m(":not(div)", &doc, p1));
        assert!(!m(":not(div)", &doc, div));
        assert!(m("div:not(.zzz)", &doc, div));
        // :not with a combinator inside.
        assert!(m(":not(div > p)", &doc, div));
        assert!(!m(":not(div > p)", &doc, p1));
    }

    #[test]
    fn not_selector_list_matching() {
        // Selectors 4: `:not()` takes a forgiving selector list; the
        // element must match none of the valid arguments.
        let (doc, div, p1, p2, _) = sample();
        assert!(m("p:not(.zzz, #p2)", &doc, p1));
        assert!(!m("p:not(.zzz, #p2)", &doc, p2));
        assert!(m(":not(div, span)", &doc, p1));
        assert!(!m(":not(div, span)", &doc, div));
        // Forgiving: the invalid group is dropped, `:not(div)` remains.
        assert!(m(":not(div, bogus##)", &doc, p1));
        assert!(!m(":not(div, bogus##)", &doc, div));
    }

    #[test]
    fn dynamic_pseudo_classes_never_match() {
        // `matches` consults no interaction state, so the interactive
        // pseudo-classes never match through it (existing behavior).
        let (doc, div, _, _, _) = sample();
        for s in [
            "a:link",
            "div:hover",
            "div:active",
            "div:focus",
            "div:visited",
        ] {
            let list = parse_selector_list(s).unwrap();
            assert!(!matches(&list[0], &doc, div), "{s} should never match");
        }
    }

    #[test]
    fn interactive_pseudo_classes_with_state() {
        fn ms(input: &str, doc: &Document, node: NodeId, state: &ElementState) -> bool {
            matches_with_state(&sel(input), doc, node, state)
        }

        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        doc.set_attribute(div, "id", "d");
        doc.set_attribute(div, "class", "a");
        let a = doc.create_element("a");
        doc.append_child(div, a);
        doc.set_attribute(a, "id", "link");
        doc.set_attribute(a, "href", "https://example.com/");

        let none = ElementState::default();
        let hover = ElementState {
            hover: true,
            ..Default::default()
        };
        let active = ElementState {
            active: true,
            ..Default::default()
        };
        let focus = ElementState {
            focus: true,
            ..Default::default()
        };
        let visited = ElementState {
            visited: true,
            ..Default::default()
        };
        let hover_active = ElementState {
            hover: true,
            active: true,
            ..Default::default()
        };

        // Each pseudo-class matches exactly when its state flag is set.
        assert!(ms("div:hover", &doc, div, &hover));
        assert!(!ms("div:hover", &doc, div, &none));
        assert!(!ms("div:hover", &doc, div, &active));
        assert!(ms("div:active", &doc, div, &active));
        assert!(!ms("div:active", &doc, div, &hover));
        assert!(ms("div:focus", &doc, div, &focus));
        assert!(!ms("div:focus", &doc, div, &none));

        // `:link` is the negation of `:visited`.
        assert!(ms("a:link", &doc, a, &none));
        assert!(!ms("a:visited", &doc, a, &none));
        assert!(!ms("a:link", &doc, a, &visited));
        assert!(ms("a:visited", &doc, a, &visited));

        // Combined with other selector parts.
        assert!(ms("div#d.a:hover", &doc, div, &hover));
        assert!(!ms("div#d.a:hover", &doc, div, &none));
        assert!(ms("div#d.a:hover", &doc, div, &hover_active));
        assert!(!ms("div:hover:active", &doc, div, &hover));
        assert!(ms("div:hover:active", &doc, div, &hover_active));
        assert!(ms("a#link:link", &doc, a, &none));
        assert!(!ms("a#link:link", &doc, a, &visited));

        // State applies to ancestor compounds too (right-to-left matching).
        assert!(ms("div:hover a", &doc, a, &hover));
        assert!(!ms("div:hover a", &doc, a, &none));

        // State threads through `:not()`: `:not(:hover)` matches when the
        // element is not hovered.
        assert!(ms(":not(:hover)", &doc, div, &none));
        assert!(!ms(":not(:hover)", &doc, div, &hover));
        assert!(ms("a:not(:visited)", &doc, a, &none));
        assert!(!ms("a:not(:visited)", &doc, a, &visited));

        // `matches` is `matches_with_state` with a default (all-false) state.
        for s in [
            "div:hover",
            "div:active",
            "div:focus",
            "a:link",
            "a:visited",
        ] {
            let list = parse_selector_list(s).unwrap();
            assert_eq!(
                matches(&list[0], &doc, div),
                matches_with_state(&list[0], &doc, div, &ElementState::default()),
                "{s}: matches must equal matches_with_state with default state"
            );
        }
    }

    #[test]
    fn link_visited_require_href() {
        fn ms(input: &str, doc: &Document, node: NodeId, state: &ElementState) -> bool {
            matches_with_state(&sel(input), doc, node, state)
        }

        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        // `a` with `href`: a real link.
        let a = doc.create_element("a");
        doc.append_child(div, a);
        doc.set_attribute(a, "href", "https://example.com/");
        // `a` without `href`: a placeholder, not a link.
        let anchor = doc.create_element("a");
        doc.append_child(div, anchor);
        doc.set_attribute(anchor, "name", "frag");
        // `area` and `link` with `href`: links too.
        let area = doc.create_element("area");
        doc.append_child(div, area);
        doc.set_attribute(area, "href", "#map");
        let link = doc.create_element("link");
        doc.append_child(div, link);
        doc.set_attribute(link, "href", "/style.css");

        let none = ElementState::default();
        let visited = ElementState {
            visited: true,
            ..Default::default()
        };

        // Only `a`/`area`/`link` with `href` match `:link`/`:visited`.
        assert!(ms("a:link", &doc, a, &none));
        assert!(ms("a:visited", &doc, a, &visited));
        assert!(ms("area:link", &doc, area, &none));
        assert!(ms("link:link", &doc, link, &none));
        // `a` without `href` matches neither, visited or not.
        assert!(!ms("a:link", &doc, anchor, &none));
        assert!(!ms("a:visited", &doc, anchor, &none));
        assert!(!ms("a:link", &doc, anchor, &visited));
        assert!(!ms("a:visited", &doc, anchor, &visited));
        // Non-link elements never match, whatever the state.
        assert!(!ms("div:link", &doc, div, &none));
        assert!(!ms("div:visited", &doc, div, &none));
        assert!(!ms("div:link", &doc, div, &visited));
        assert!(!ms("div:visited", &doc, div, &visited));
        // Universal `:link` still only hits real links.
        assert!(ms("div :link", &doc, a, &none));
        assert!(!ms("div :link", &doc, anchor, &none));
    }

    #[test]
    fn parsed_html_document() {
        let (doc, _) = Document::parse_html(
            b"<!DOCTYPE html><html><head><title>t</title></head><body><div id=x class=\"a\">hi</div></body></html>",
        );
        let div = doc.get_element_by_id("x").unwrap();
        assert!(m("div", &doc, div));
        assert!(m("body div", &doc, div));
        assert!(m("body > div", &doc, div));
        assert!(m("html body > div.a", &doc, div));
        assert!(m("div:first-child", &doc, div));
        assert!(m("div:last-child", &doc, div)); // only element child of body
        let html = doc.document_element().unwrap();
        assert!(m(":root", &doc, html));
    }

    /// Parse a tiny document and return (doc, html element).
    fn parsed_doc() -> (Document, NodeId) {
        let (doc, _) = Document::parse_html(b"<html><body><div></div></body></html>");
        let html = doc.document_element().unwrap();
        (doc, html)
    }

    #[test]
    fn pseudo_elements_never_match_real_elements() {
        // Pseudo-element declarations must not contaminate the originating
        // element's own computed style (their pseudo boxes come later).
        let (doc, div, _, _, _) = sample();
        assert!(!m("div::before", &doc, div));
        assert!(!m("div::after", &doc, div));
        assert!(!m("div:before", &doc, div));
        assert!(!m("p::before", &doc, div));
        assert!(!m("div::first-line", &doc, div));
        assert!(!m("div::selection", &doc, div));
    }

    #[test]
    fn nth_last_child_matching() {
        let (doc, _, p1, p2, span) = sample();
        assert!(m("span:nth-last-child(1)", &doc, span));
        assert!(m("p:nth-last-child(2)", &doc, p2));
        assert!(!m("p:nth-last-child(1)", &doc, p2));
        assert!(m(":nth-last-child(odd)", &doc, p1));
        assert!(m(":nth-last-child(even)", &doc, p2));
        assert!(m("span:nth-last-child(-n+2)", &doc, span));
        assert!(!m("p:nth-last-child(-n+1)", &doc, p1));
    }

    #[test]
    fn of_type_matching() {
        let (doc, _, p1, p2, span) = sample();
        assert!(m("p:first-of-type", &doc, p1));
        assert!(!m("p:first-of-type", &doc, p2));
        assert!(m("p:last-of-type", &doc, p2));
        assert!(m("span:only-of-type", &doc, span));
        assert!(!m("p:only-of-type", &doc, p1));
        assert!(m("p:nth-of-type(2)", &doc, p2));
        assert!(m("p:nth-last-of-type(1)", &doc, p2));
        assert!(m("p:nth-last-of-type(2)", &doc, p1));
        assert!(!m("span:nth-of-type(2)", &doc, span));
    }

    #[test]
    fn of_type_counts_same_tag_only() {
        // :nth-of-type counts siblings with the same tag name, not all siblings.
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        let p1 = doc.create_element("p");
        doc.append_child(div, p1);
        let span = doc.create_element("span");
        doc.append_child(div, span);
        let p2 = doc.create_element("p");
        doc.append_child(div, p2);
        assert!(m("p:nth-of-type(1)", &doc, p1));
        assert!(m("p:nth-of-type(2)", &doc, p2));
        assert!(!m("p:nth-of-type(3)", &doc, p2));
        assert!(m("span:nth-of-type(1)", &doc, span));
        assert!(m("p:nth-child(3)", &doc, p2));
        assert!(!m("p:nth-child(2)", &doc, p2));
    }

    #[test]
    fn only_child_matching() {
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        let only = doc.create_element("p");
        doc.append_child(div, only);
        assert!(m("p:only-child", &doc, only));
        assert!(m("p:first-child:last-child", &doc, only));
        let second = doc.create_element("span");
        doc.append_child(div, second);
        assert!(!m("p:only-child", &doc, only));
        assert!(!m("span:only-child", &doc, second));
    }

    #[test]
    fn is_where_matching() {
        let (doc, div, p1, _, span) = sample();
        assert!(m(":is(p, span)", &doc, p1));
        assert!(m(":is(p, span)", &doc, span));
        assert!(!m(":is(p, span)", &doc, div));
        assert!(m(":where(p, span)", &doc, p1));
        assert!(m("div :is(p:first-child, span)", &doc, p1));
        assert!(m(":is(p, span):last-child", &doc, span));
        assert!(!m(":is(p, span):last-child", &doc, p1));
        // :not() still rejects through :is().
        assert!(!m(":not(:is(p, span))", &doc, p1));
        assert!(m(":not(:is(p, span))", &doc, div));
    }

    /// Fixture: main > article + aside; article > h + p1 + p2; p2 > a + em.
    fn has_fixture() -> (
        Document,
        NodeId,
        NodeId,
        NodeId,
        NodeId,
        NodeId,
        NodeId,
        NodeId,
    ) {
        let mut doc = Document::new();
        let root = doc.root();
        let main = doc.create_element("main");
        doc.append_child(root, main);
        let article = doc.create_element("article");
        doc.append_child(main, article);
        let aside = doc.create_element("aside");
        doc.append_child(main, aside);
        let h = doc.create_element("h1");
        doc.append_child(article, h);
        let p1 = doc.create_element("p");
        doc.append_child(article, p1);
        let p2 = doc.create_element("p");
        doc.append_child(article, p2);
        let a = doc.create_element("a");
        doc.append_child(p2, a);
        let em = doc.create_element("em");
        doc.append_child(p2, em);
        (doc, main, article, h, p1, p2, a, em)
    }

    #[test]
    fn has_descendant_anchor() {
        let (doc, main_el, a1, _, p1, _, as_, _) = has_fixture();
        // Default (descendant) anchor.
        assert!(m("main:has(p)", &doc, main_el));
        assert!(m("main:has(article p)", &doc, main_el));
        assert!(m("main:has(article > p)", &doc, main_el));
        assert!(!m("main:has(section p)", &doc, main_el));
        assert!(m("article:has(p)", &doc, a1));
        assert!(m("article:has(> p)", &doc, a1));
        assert!(!m("article:has(p)", &doc, p1));
        assert!(!m("aside:has(p)", &doc, as_));
        // :has() looks forward only, never at ancestors or self.
        assert!(!m("p:has(article)", &doc, p1));
        assert!(!m("main:has(main)", &doc, main_el));
    }

    #[test]
    fn has_child_anchor() {
        let (doc, main_el, a1, _, _, _, _, _) = has_fixture();
        assert!(m("main:has(> article)", &doc, main_el));
        assert!(!m("main:has(> p)", &doc, main_el));
        assert!(m("main:has(> article > p)", &doc, main_el));
        assert!(m("article:has(> p)", &doc, a1));
        assert!(!m("article:has(> a)", &doc, a1));
    }

    #[test]
    fn has_sibling_anchors() {
        let (doc, _, a1, h, p1, p2, _, _) = has_fixture();
        // Next-sibling anchor: h1 + p matches p1 (immediately preceded by h1).
        assert!(m("h1:has(+ p)", &doc, h));
        assert!(!m("h1:has(+ a)", &doc, h));
        // Subsequent-sibling anchor: h1 ~ p matches h (p1 and p2 follow it).
        assert!(m("h1:has(~ p)", &doc, h));
        assert!(!m("p:has(~ p)", &doc, p2));
        assert!(m("p:has(~ p)", &doc, p1));
        // :has() never looks backward.
        assert!(!m("p:has(+ h1)", &doc, p1));
        let _ = a1;
    }

    #[test]
    fn has_relative_lists_and_forgiving() {
        let (doc, main_el, a1, _, _, _, as_, _) = has_fixture();
        // Selector lists inside :has().
        assert!(m("main:has(> section, article)", &doc, main_el));
        assert!(!m("main:has(> section, nav)", &doc, main_el));
        // Forgiving: invalid entries are dropped, valid ones still match.
        assert!(m("main:has(::bogus, article)", &doc, main_el));
        // All-invalid still invalidates the whole :has().
        assert!(parse_selector_list("article:has(::bogus)").is_err());
        // Nested functional pseudos are allowed; nested :has() is not.
        assert!(m("main:has(article:not(.zzz))", &doc, main_el));
        assert!(m("main:has(:is(article, section))", &doc, main_el));
        assert!(parse_selector_list("main:has(:has(p))").is_err());
        let _ = a1;
        let _ = as_;
    }

    #[test]
    fn has_with_element_state() {
        fn ms(input: &str, doc: &Document, node: NodeId, state: &ElementState) -> bool {
            matches_with_state(&sel(input), doc, node, state)
        }
        let (doc, main_el, _, _, _, _, _, _) = has_fixture();
        let none = ElementState::default();
        let hover = ElementState {
            hover: true,
            ..Default::default()
        };
        // State on the anchor element is visible to :has().
        assert!(ms("main:has(p)", &doc, main_el, &none));
        let s = sel("main:has(p):hover");
        assert!(matches_with_state(&s, &doc, main_el, &hover));
        assert!(!matches_with_state(&s, &doc, main_el, &none));
        // State also applies inside the :has() argument.
        let s2 = sel("main:has(p:hover)");
        assert!(!matches_with_state(&s2, &doc, main_el, &none));
        assert!(matches_with_state(&s2, &doc, main_el, &hover));
    }

    /// Fixture: a form with checkbox, radio, option, and disabled controls.
    fn form_fixture() -> (
        Document,
        NodeId,
        NodeId,
        NodeId,
        NodeId,
        NodeId,
        NodeId,
        NodeId,
    ) {
        let mut doc = Document::new();
        let root = doc.root();
        let form = doc.create_element("form");
        doc.append_child(root, form);
        let checked_box = doc.create_element("input");
        doc.set_attribute(checked_box, "type", "checkbox");
        doc.set_attribute(checked_box, "checked", "");
        doc.append_child(form, checked_box);
        let unchecked_box = doc.create_element("input");
        doc.set_attribute(unchecked_box, "type", "checkbox");
        doc.append_child(form, unchecked_box);
        let radio = doc.create_element("input");
        doc.set_attribute(radio, "type", "radio");
        doc.append_child(form, radio);
        let text = doc.create_element("input");
        doc.set_attribute(text, "type", "text");
        doc.append_child(form, text);
        let select = doc.create_element("select");
        doc.append_child(form, select);
        let selected_opt = doc.create_element("option");
        doc.set_attribute(selected_opt, "selected", "");
        doc.append_child(select, selected_opt);
        let plain_opt = doc.create_element("option");
        doc.append_child(select, plain_opt);
        (
            doc,
            checked_box,
            unchecked_box,
            radio,
            text,
            select,
            selected_opt,
            plain_opt,
        )
    }

    #[test]
    fn checked_matching() {
        let (doc, yes, no, radio, text, _, sel_opt, plain_opt) = form_fixture();
        assert!(m("input:checked", &doc, yes));
        assert!(!m("input:checked", &doc, no));
        // Radio without checked attribute does not match.
        assert!(!m("input:checked", &doc, radio));
        // Non-checkable input never matches.
        assert!(!m("input:checked", &doc, text));
        // Selected options match :checked.
        assert!(m("option:checked", &doc, sel_opt));
        assert!(!m("option:checked", &doc, plain_opt));
        // Checked state composes with :not() and combinators.
        assert!(m("input:not(:checked)", &doc, no));
        assert!(!m("input:not(:checked)", &doc, yes));
        assert!(m("form input:checked", &doc, yes));
    }

    #[test]
    fn disabled_enabled_matching() {
        let mut doc = Document::new();
        let root = doc.root();
        let form = doc.create_element("form");
        doc.append_child(root, form);
        let enabled_input = doc.create_element("input");
        doc.append_child(form, enabled_input);
        let disabled_input = doc.create_element("input");
        doc.set_attribute(disabled_input, "disabled", "");
        doc.append_child(form, disabled_input);
        let button = doc.create_element("button");
        doc.append_child(form, button);
        let div = doc.create_element("div");
        doc.append_child(form, div);

        assert!(m("input:disabled", &doc, disabled_input));
        assert!(!m("input:disabled", &doc, enabled_input));
        assert!(m("input:enabled", &doc, enabled_input));
        assert!(!m("input:enabled", &doc, disabled_input));
        assert!(m("button:enabled", &doc, button));
        // Non-disableable elements never match either.
        assert!(!m("div:disabled", &doc, div));
        assert!(!m("div:enabled", &doc, div));
        assert!(!m("form:disabled", &doc, form));
        // Composes with :not().
        assert!(m("input:not(:disabled)", &doc, enabled_input));
    }

    #[test]
    fn disabled_fieldset_and_optgroup_inheritance() {
        let mut doc = Document::new();
        let root = doc.root();
        let fs = doc.create_element("fieldset");
        doc.set_attribute(fs, "disabled", "");
        doc.append_child(root, fs);
        let legend = doc.create_element("legend");
        doc.append_child(fs, legend);
        let in_legend = doc.create_element("input");
        doc.append_child(legend, in_legend);
        let in_fs = doc.create_element("input");
        doc.append_child(fs, in_fs);
        // Controls inside the first legend escape the fieldset's disabled.
        assert!(!m("input:disabled", &doc, in_legend));
        assert!(m("input:enabled", &doc, in_legend));
        // Other descendants of a disabled fieldset are disabled.
        assert!(m("input:disabled", &doc, in_fs));
        assert!(!m("input:enabled", &doc, in_fs));
        assert!(m("fieldset:disabled", &doc, fs));
        // Options inherit from a disabled optgroup.
        let select = doc.create_element("select");
        doc.append_child(root, select);
        let og = doc.create_element("optgroup");
        doc.set_attribute(og, "disabled", "");
        doc.append_child(select, og);
        let opt = doc.create_element("option");
        doc.append_child(og, opt);
        assert!(m("option:disabled", &doc, opt));
        assert!(!m("option:enabled", &doc, opt));
    }

    #[test]
    fn structural_pseudo_class_combos() {
        // New pseudo-classes compose with combinators and :not().
        let (doc, _, p1, p2, span) = sample();
        assert!(m("div > p:first-of-type", &doc, p1));
        assert!(m("div > :is(p, span):last-child", &doc, span));
        assert!(m("p:not(:first-of-type)", &doc, p2));
        assert!(!m("p:not(:first-of-type)", &doc, p1));
        assert!(m(":where(p):nth-of-type(2)", &doc, p2));
    }

    /// (rule, selector) pairs from [`RuleIndex::candidates_into`], for assertions.
    fn cand_pairs(index: &RuleIndex, doc: &Document, node: NodeId) -> Vec<(usize, usize)> {
        let data = match doc.kind(node) {
            NodeKind::Element(d) => d.clone(),
            _ => panic!("expected an element"),
        };
        let mut refs = Vec::new();
        index.candidates_into(&data, &mut refs);
        refs.into_iter().map(|r| (r.rule, r.selector)).collect()
    }

    #[test]
    fn rule_index_buckets_by_rightmost_compound() {
        use crate::parse::parse_stylesheet;
        // rule: 0:div 1:#a 2:.b 3:div#a.b 4:[data-x] 5::root 6:* 7:span.c 8:p,.d 9:div p
        let sheet = parse_stylesheet(
            "div { color: red; } #a { color: blue; } .b { color: green; } \
             div#a.b { color: yellow; } [data-x] { color: black; } :root { color: white; } \
             * { color: gray; } span.c { color: orange; } p, .d { color: pink; } \
             div p { color: brown; }",
        );
        let index = RuleIndex::build(&sheet);

        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        doc.set_attribute(div, "id", "a");
        doc.set_attribute(div, "class", "b c");
        doc.set_attribute(div, "data-x", "1");

        // tag div -> (0,0),(3,0); id a -> (1,0),(3,0); class b -> (2,0),(3,0);
        // class c -> (7,0) (tag mismatch is the full matcher's job);
        // universal -> (4,0),(5,0),(6,0). `div p` (9,0) is keyed on its
        // rightmost compound `p`, so it must NOT appear here.
        assert_eq!(
            cand_pairs(&index, &doc, div),
            vec![
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
                (7, 0)
            ]
        );

        // A `p` child sees the `p` selectors, including `div p` via its
        // rightmost compound.
        let p = doc.create_element("p");
        doc.append_child(div, p);
        assert_eq!(
            cand_pairs(&index, &doc, p),
            vec![(4, 0), (5, 0), (6, 0), (8, 0), (9, 0)]
        );

        // Id matching is exact: `id="A"` finds nothing under `#a`.
        let div2 = doc.create_element("div");
        doc.append_child(root, div2);
        doc.set_attribute(div2, "id", "A");
        assert_eq!(
            cand_pairs(&index, &doc, div2),
            vec![(0, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
        );
    }

    /// `candidates_into` fills a caller-provided scratch buffer and reuses
    /// its allocation across queries: no fresh `Vec` per element per
    /// sheet on the cascade hot path.
    #[test]
    fn candidates_into_reuses_scratch_buffer() {
        use crate::parse::parse_stylesheet;
        let sheet =
            parse_stylesheet("div#a.b.c.d { color: red; } .b { color: blue; } * { color: gray; }");
        let index = RuleIndex::build(&sheet);

        let mut doc = Document::new();
        let root = doc.root();
        // Largest candidate set first, so the second query fits the
        // existing capacity without growing.
        let big = doc.create_element("div");
        doc.append_child(root, big);
        doc.set_attribute(big, "id", "a");
        doc.set_attribute(big, "class", "b c d");
        let small = doc.create_element("span");
        doc.append_child(root, small);

        fn data_of(doc: &Document, node: NodeId) -> ElementData {
            match doc.kind(node) {
                NodeKind::Element(d) => d.clone(),
                _ => panic!("expected an element"),
            }
        }

        let mut buf = Vec::new();
        index.candidates_into(&data_of(&doc, big), &mut buf);
        assert_eq!(buf.len(), 3);
        let (ptr, cap) = (buf.as_ptr(), buf.capacity());
        // Same content as the old allocating `candidates`.
        let again: Vec<SelectorRef> = {
            let mut v = Vec::new();
            index.candidates_into(&data_of(&doc, big), &mut v);
            v
        };
        assert_eq!(buf, again);

        // Second query reuses the buffer: same pointer and capacity, no
        // reallocation.
        index.candidates_into(&data_of(&doc, small), &mut buf);
        assert_eq!(buf.as_ptr(), ptr);
        assert_eq!(buf.capacity(), cap);
        // `span` sees only the universal rule.
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn rule_index_foreign_tag_lookup() {
        use crate::parse::parse_stylesheet;
        // Foreign tag names are case-sensitive for matching, but the
        // index buckets them lowercased as a pre-filter.
        let sheet = parse_stylesheet("foreignObject { color: red; }");
        let index = RuleIndex::build(&sheet);

        let mut doc = Document::new();
        let root = doc.root();
        let el = doc.create_element_ns(Namespace::Svg, "foreignObject");
        doc.append_child(root, el);
        assert_eq!(cand_pairs(&index, &doc, el), vec![(0, 0)]);

        // An uppercase HTML tag still finds the lowercased bucket.
        let el2 = doc.create_element_ns(Namespace::Svg, "RECT");
        doc.append_child(root, el2);
        let sheet2 = parse_stylesheet("rect { color: red; }");
        let index2 = RuleIndex::build(&sheet2);
        assert_eq!(cand_pairs(&index2, &doc, el2), vec![(0, 0)]);
    }

    // ---- backtracking: descendant/~ combinators must not commit to the
    // nearest witness ----

    #[test]
    fn descendant_backtracks_past_inner_match() {
        // `a > b c`: the inner `b` is the nearest descendant-match for `c`,
        // but its parent is not an `a`; the outer `b` completes the match.
        let (doc, _) =
            Document::parse_html(b"<html><body><a><b><b><c></c></b></b></a></body></html>");
        let c = doc.get_elements_by_tag_name("c").first().unwrap();
        assert!(m("a > b c", &doc, c));
        // And the old greedy behavior still agrees on simple cases.
        assert!(m("a b c", &doc, c));
        assert!(!m("x > b c", &doc, c));
        // `a > b > c` correctly does NOT match: c's parent chain is b, b, a.
        assert!(!m("a > b > c", &doc, c));
    }

    #[test]
    fn descendant_backtracks_through_sibling_combinator() {
        // `a ~ b c`: the nearest `b` ancestor of `c` (inner) has no
        // preceding `a` sibling; the outer `b` does. Greedy nearest-
        // witness matching fails this; backtracking finds the outer `b`.
        let (doc, _) =
            Document::parse_html(b"<html><body><a></a><b><i><b><c></c></b></i></b></body></html>");
        let c = doc.get_elements_by_tag_name("c").first().unwrap();
        assert!(m("a ~ b c", &doc, c));
        // Without the `a`, nothing matches.
        let (doc2, _) =
            Document::parse_html(b"<html><body><b><i><b><c></c></b></i></b></body></html>");
        let c2 = doc2.get_elements_by_tag_name("c").first().unwrap();
        assert!(!m("a ~ b c", &doc2, c2));
    }

    #[test]
    fn subsequent_sibling_backtracks() {
        // `a ~ b + c`: the `+` binds `c` to the inner `b`, but `~` must
        // keep scanning left past the matching-less sibling for a `b`
        // with an `a` sibling... here: c's previous sibling is b2 (no `a`
        // sibling), backtrack: is there any earlier sibling matching `b`
        // preceded by `a`? b1 has a before it -> match.
        let (doc, _) =
            Document::parse_html(b"<html><body><a></a><b></b><b></b><c></c></body></html>");
        let c = doc.get_elements_by_tag_name("c").first().unwrap();
        assert!(m("a ~ b + c", &doc, c));
        let (doc2, _) = Document::parse_html(b"<html><body><b></b><b></b><c></c></body></html>");
        let c2 = doc2.get_elements_by_tag_name("c").first().unwrap();
        assert!(!m("a ~ b + c", &doc2, c2));
    }

    // ---- per-element interactive state ----

    /// `matches_cascade` with an explicit per-node state map.
    fn mc_states(
        selector: &str,
        doc: &Document,
        node: NodeId,
        states: &[(NodeId, ElementState)],
    ) -> bool {
        let s = sel(selector);
        let sib = SiblingCache::default();
        let mut scratch = MatchScratch::new();
        matches_cascade(
            &s,
            doc,
            node,
            &|n| {
                states
                    .iter()
                    .find(|(id, _)| *id == n)
                    .map(|(_, st)| *st)
                    .unwrap_or_default()
            },
            Some(&sib),
            &mut scratch,
            None,
        )
    }

    fn hovered() -> ElementState {
        ElementState {
            hover: true,
            ..ElementState::default()
        }
    }

    fn ms(input: &str, doc: &Document, node: NodeId, state: &ElementState) -> bool {
        matches_with_state(&sel(input), doc, node, state)
    }

    #[test]
    fn ancestor_hover_uses_ancestor_state() {
        // `.p:hover .c`: the PARENT is hovered, the child is not. The old
        // single-state matcher got this wrong (it tested the child's
        // state for the parent's compound).
        let (doc, _) =
            Document::parse_html(b"<html><body><div id=p><span id=c></span></div></body></html>");
        let p = doc.get_element_by_id("p").unwrap();
        let c = doc.get_element_by_id("c").unwrap();
        assert!(mc_states("#p:hover #c", &doc, c, &[(p, hovered())]));
        assert!(!mc_states("#p:hover #c", &doc, c, &[(c, hovered())]));
        assert!(!mc_states("#p:hover #c", &doc, c, &[]));
        // And the public single-state API is unchanged in behavior: the
        // one state applies everywhere (documented limitation).
        assert!(ms("#p:hover #c", &doc, c, &hovered()));
    }

    #[test]
    fn sibling_hover_uses_sibling_state() {
        let (doc, _) = Document::parse_html(b"<html><body><i></i><b></b></body></html>");
        let i = doc.get_elements_by_tag_name("i").first().unwrap();
        let b = doc.get_elements_by_tag_name("b").first().unwrap();
        assert!(mc_states("i:hover + b", &doc, b, &[(i, hovered())]));
        assert!(!mc_states("i:hover + b", &doc, b, &[(b, hovered())]));
    }

    #[test]
    fn has_witness_hover_uses_witness_state() {
        // `div:has(:hover)`: the hovered element is a DESCENDANT witness,
        // not the div itself.
        let (doc, _) = Document::parse_html(b"<html><body><div><em></em></div></body></html>");
        let div = doc.get_elements_by_tag_name("div").first().unwrap();
        let em = doc.get_elements_by_tag_name("em").first().unwrap();
        assert!(mc_states("div:has(:hover)", &doc, div, &[(em, hovered())]));
        assert!(!mc_states(
            "div:has(:hover)",
            &doc,
            div,
            &[(div, hovered())]
        ));
        // The em itself matches `:hover` through its own state.
        assert!(mc_states(":hover", &doc, em, &[(em, hovered())]));
        assert!(!mc_states(":hover", &doc, em, &[(div, hovered())]));
    }
    // -- :has() cursor traversal -------------------------------------------

    /// The allocation-free [`Descendants`] cursor must visit nodes in the
    /// same document order as [`Document::descendants`], over a mixed tree
    /// with text nodes and nested depth.
    #[test]
    fn descendants_cursor_matches_descendants_vec() {
        let (doc, main_el, _, _, _, _, _, em) = has_fixture();
        let via_cursor: Vec<NodeId> = Descendants::new(&doc, main_el).collect();
        assert_eq!(via_cursor, doc.descendants(main_el));
        // Empty subtree: no descendants, no panic.
        assert!(Descendants::new(&doc, em).next().is_none());
        // Every subtree agrees, node by node.
        for n in std::iter::once(main_el).chain(via_cursor.iter().copied()) {
            let a: Vec<NodeId> = Descendants::new(&doc, n).collect();
            assert_eq!(a, doc.descendants(n), "mismatch under {n:?}");
        }
    }

    /// The cursor does not cross the template-contents edge, exactly like
    /// [`Document::children`].
    #[test]
    fn descendants_cursor_skips_template_contents() {
        let (doc, _) = Document::parse_html(
            b"<html><body><div><template><p>in</p></template><span></span></div></body></html>",
        );
        let div = doc.get_elements_by_tag_name("div").first().unwrap();
        let via_cursor: Vec<NodeId> = Descendants::new(&doc, div).collect();
        assert_eq!(via_cursor, doc.descendants(div));
        assert!(!via_cursor.iter().any(|&n| tag_name_is(&doc, n, "p")));
        assert!(via_cursor.iter().any(|&n| tag_name_is(&doc, n, "span")));
    }

    fn tag_name_is(doc: &Document, node: NodeId, tag: &str) -> bool {
        matches!(doc.kind(node), NodeKind::Element(d) if d.tag_name == tag)
    }

    /// `children_cursor` visits exactly `children()`'s nodes, without the
    /// `Vec`.
    #[test]
    fn children_cursor_matches_children_vec() {
        let (doc, main_el, a1, _, _, _, _, _) = has_fixture();
        let a: Vec<NodeId> = children_cursor(&doc, main_el).collect();
        assert_eq!(a, doc.children(main_el));
        assert_eq!(children_cursor(&doc, a1).count(), doc.children(a1).len());
    }

    /// Multi-compound `:has()` keeps dedup + first-witness semantics on
    /// diamond-shaped frontiers (two paths reach the same node).
    #[test]
    fn has_multi_compound_dedup() {
        let (doc, main_el, a1, _, p1, p2, _, _) = has_fixture();
        // Three compounds: level-2 frontier walks from every article>p.
        assert!(m("main:has(article p em)", &doc, main_el));
        assert!(!m("main:has(article p strong)", &doc, main_el));
        assert!(!m("main:has(section p em)", &doc, main_el));
        // Dedup: `p` matches at level 1 via one path only per node; the
        // anchor still sees both witnesses without double-counting.
        assert!(m("article:has(p em)", &doc, a1));
        assert!(!m("article:has(p strong)", &doc, a1));
        let _ = (p1, p2);
    }

    // -- :nth-child memoization --------------------------------------------

    /// A wide parent: 600 element children with interleaved tags and text
    /// nodes, so the naive per-query sibling walk would be O(N^2) across
    /// the row.
    fn wide_row() -> (Document, NodeId, Vec<NodeId>) {
        let mut html = String::from("<html><body><div>");
        for i in 0..600 {
            match i % 3 {
                0 => html.push_str("<li></li>"),
                1 => html.push_str("<p></p>"),
                _ => html.push_str("<span></span>"),
            }
            if i % 7 == 0 {
                html.push_str("text");
            }
        }
        html.push_str("</div></body></html>");
        let (doc, _) = Document::parse_html(html.as_bytes());
        let div = doc.get_elements_by_tag_name("div").first().unwrap();
        let kids: Vec<NodeId> = children_cursor(&doc, div)
            .filter(|&n| is_element(&doc, n))
            .collect();
        assert_eq!(kids.len(), 600);
        (doc, div, kids)
    }

    /// Naive O(siblings) reference for the structural indexes, built on
    /// the allocating `children()` API.
    fn naive_indexes(doc: &Document, parent: NodeId, node: NodeId) -> (i32, i32, i32, i32) {
        let elems: Vec<NodeId> = doc
            .children(parent)
            .into_iter()
            .filter(|&n| is_element(doc, n))
            .collect();
        let pos = elems.iter().position(|&n| n == node).unwrap() as i32 + 1;
        let from_end = elems.len() as i32 - pos + 1;
        let data = element_data(doc, node).unwrap();
        let group: Vec<NodeId> = elems
            .iter()
            .copied()
            .filter(|&n| same_type(data, element_data(doc, n).unwrap()))
            .collect();
        let rank = group.iter().position(|&n| n == node).unwrap() as i32 + 1;
        (pos, from_end, rank, group.len() as i32 - rank + 1)
    }

    /// The memoized sibling index agrees with the naive walk on every
    /// child, and — the point of the cache — builds exactly once per
    /// parent no matter how many queries hit it.
    #[test]
    fn nth_child_index_builds_once_per_parent() {
        let (doc, div, kids) = wide_row();
        let sib = SiblingCache::default();
        let state_for = |_: NodeId| ElementState::default();
        let nth = sel(":nth-child(2n+1)");
        let mut scratch = MatchScratch::new();
        for &k in &kids {
            let (pos, from_end, rank, rank_end) = naive_indexes(&doc, div, k);
            assert_eq!(element_index(Some(&sib), &doc, k), pos, "index at {k:?}");
            assert_eq!(
                element_index_from_end(Some(&sib), &doc, k),
                from_end,
                "from_end at {k:?}"
            );
            assert_eq!(
                element_index_of_type(Some(&sib), &doc, k),
                rank,
                "of_type at {k:?}"
            );
            assert_eq!(
                element_index_of_type_from_end(Some(&sib), &doc, k),
                rank_end,
                "of_type_from_end at {k:?}"
            );
            // Exercise the full matcher path too, through the shared cache.
            let _ = match_complex(&nth, &doc, k, &state_for, Some(&sib), &mut scratch, None);
        }
        assert_eq!(
            sib.builds.get(),
            1,
            "one parent => one index build, not one per query"
        );
    }

    /// End-to-end through the public one-shot API: `:nth-child`,
    /// `:nth-last-child`, and the `-of-type` family agree with the naive
    /// reference on every child of the wide row (interleaved text nodes
    /// must not shift element indexes).
    #[test]
    fn nth_family_matches_naive_reference() {
        let (doc, div, kids) = wide_row();
        for &k in &kids {
            let (pos, from_end, rank, rank_end) = naive_indexes(&doc, div, k);
            let cases = [
                (":nth-child(2n)", nth_matches(2, 0, pos)),
                (":nth-child(odd)", nth_matches(2, 1, pos)),
                (":nth-child(3n+1)", nth_matches(3, 1, pos)),
                (":nth-child(-n+5)", nth_matches(-1, 5, pos)),
                (":nth-last-child(2)", nth_matches(0, 2, from_end)),
                (":nth-last-child(3n)", nth_matches(3, 0, from_end)),
                (":nth-of-type(2n)", nth_matches(2, 0, rank)),
                (":nth-last-of-type(1)", nth_matches(0, 1, rank_end)),
                (":first-of-type", rank == 1),
                (":last-of-type", rank_end == 1),
                (":only-of-type", rank == 1 && rank_end == 1),
            ];
            for (s, want) in cases {
                assert_eq!(m(s, &doc, k), want, "{s} mismatch at {k:?}");
            }
        }
    }

    /// `type_key` partitions elements exactly like [`same_type`]: equal
    /// keys iff same type. Guards the [`SiblingIndex::by_type`] grouping
    /// rule (HTML case-insensitivity, foreign exactness, namespace
    /// separation).
    #[test]
    fn type_key_agrees_with_same_type() {
        fn data(ns: Namespace, tag: &str) -> ElementData {
            ElementData {
                tag_name: tag.into(),
                namespace: ns,
                attributes: Attributes::new(),
                shadow_root: None,
            }
        }
        let cases = [
            (Namespace::Html, "div"),
            (Namespace::Html, "DIV"),
            (Namespace::Html, "Div"),
            (Namespace::Html, "span"),
            (Namespace::Html, "foreignObject"),
            (Namespace::Html, "FOREIGNOBJECT"),
            (Namespace::Svg, "rect"),
            (Namespace::Svg, "RECT"),
            (Namespace::Svg, "foreignObject"),
            (Namespace::Svg, "foreignobject"),
            (Namespace::MathMl, "mi"),
            (Namespace::MathMl, "MI"),
        ];
        for (i, (ns_a, tag_a)) in cases.iter().enumerate() {
            for (ns_b, tag_b) in &cases[i..] {
                let a = data(ns_a.clone(), tag_a);
                let b = data(ns_b.clone(), tag_b);
                assert_eq!(
                    same_type(&a, &b),
                    type_key(&a) == type_key(&b),
                    "key/same_type disagree for ({ns_a:?},{tag_a}) vs ({ns_b:?},{tag_b})"
                );
            }
        }
    }

    /// `:nth-of-type` grouping through the atom-keyed sibling index:
    /// mixed-case HTML tags group case-insensitively, SVG tags group
    /// case-sensitively, and unknown (custom-element / non-canonical)
    /// names group exactly like [`same_type`]. The cached index must
    /// agree with the uncached sibling walk on every node and selector.
    #[test]
    fn nth_of_type_mixed_case_cached_matches_uncached() {
        let mut doc = Document::new();
        let root = doc.root();
        let parent = doc.create_element("div");
        doc.append_child(root, parent);
        // HTML, mixed case via `create_element_ns` (kept as-is): "DIV"
        // and "Div" must group with "div".
        let d1 = doc.create_element("div");
        let d2 = doc.create_element_ns(Namespace::Html, "DIV");
        let d3 = doc.create_element_ns(Namespace::Html, "Div");
        // Unknown custom elements, mixed case: one group.
        let c1 = doc.create_element_ns(Namespace::Html, "my-widget");
        let c2 = doc.create_element_ns(Namespace::Html, "MY-WIDGET");
        // SVG: case-sensitive — "rect" and "RECT" are different types.
        let r1 = doc.create_element_ns(Namespace::Svg, "rect");
        let r2 = doc.create_element_ns(Namespace::Svg, "RECT");
        let r3 = doc.create_element_ns(Namespace::Svg, "rect");
        let nodes = [d1, d2, d3, c1, c2, r1, r2, r3];
        for &n in &nodes {
            doc.append_child(parent, n);
        }

        let selectors = [
            ":nth-of-type(1)",
            ":nth-of-type(2)",
            ":nth-of-type(3)",
            ":nth-of-type(2n)",
            ":nth-of-type(2n+1)",
            ":nth-last-of-type(1)",
            ":nth-last-of-type(2)",
            ":first-of-type",
            ":last-of-type",
            ":only-of-type",
        ];
        let sib = SiblingCache::default();
        let state_for = |_: NodeId| ElementState::default();
        for s in selectors {
            let selector = sel(s);
            for &node in &nodes {
                let mut scratch = MatchScratch::new();
                let cached = matches_cascade(
                    &selector,
                    &doc,
                    node,
                    &state_for,
                    Some(&sib),
                    &mut scratch,
                    None,
                );
                assert_eq!(
                    cached,
                    m(s, &doc, node),
                    "cached/uncached disagree for {s} on {node:?}"
                );
            }
        }

        // Pin the grouping itself, not just cached/uncached agreement.
        // div group: d1=1, d2=2, d3=3 (mixed case grouped together).
        assert!(m(":nth-of-type(1)", &doc, d1));
        assert!(m(":nth-of-type(2)", &doc, d2));
        assert!(m(":nth-of-type(3)", &doc, d3));
        assert!(m(":nth-last-of-type(1)", &doc, d3));
        assert!(!m(":only-of-type", &doc, d2));
        // Custom-element group: c1=1, c2=2.
        assert!(m(":nth-of-type(2)", &doc, c2));
        assert!(m(":last-of-type", &doc, c2));
        // SVG: "rect" (r1=1, r3=2) and "RECT" (r2 alone) differ.
        assert!(m(":nth-of-type(1)", &doc, r1));
        assert!(m(":nth-of-type(2)", &doc, r3));
        assert!(m(":last-of-type", &doc, r3));
        assert!(m(":only-of-type", &doc, r2));
        assert!(!m(":nth-of-type(2)", &doc, r2));
    }

    // -- atom interning: selector-matching equivalence -------------------
    //
    // These pin the observable matching behavior across the atom
    // boundary: well-known names (integer fast path), unknown names
    // (heap-spill fallback), and non-canonical names must all match
    // exactly as the old string compares did.

    /// Type selectors against interned tag names: ASCII
    /// case-insensitive for HTML in any selector case, exact for
    /// foreign elements, correct through the spill fallback for custom
    /// elements.
    #[test]
    fn atom_type_selector_equivalence() {
        let mut doc = Document::new();
        let root = doc.root();
        // Uppercased input is lowercased at intern time (createElement).
        let div = doc.create_element("DIV");
        doc.append_child(root, div);
        let svg = doc.create_element_ns(Namespace::Svg, "foreignObject");
        doc.append_child(root, svg);
        // Unknown name: spill path on the element side.
        let widget = doc.create_element("my-widget");
        doc.append_child(root, widget);

        for s in ["div", "DIV", "Div", "dIv"] {
            assert!(m(s, &doc, div), "{s} should match <div>");
        }
        assert!(m("foreignObject", &doc, svg));
        assert!(!m("foreignobject", &doc, svg));
        assert!(!m("FOREIGNOBJECT", &doc, svg));
        assert!(m("my-widget", &doc, widget));
        // HTML stays case-insensitive even through the spill fallback.
        assert!(m("MY-WIDGET", &doc, widget));
        assert!(!m("my-other", &doc, widget));
        assert!(!m("div", &doc, widget));
        // Unknown selector name never matches a well-known element name.
        assert!(!m("my-widget", &doc, div));
    }

    /// Attribute selectors against interned attribute names, including
    /// case-insensitive HTML names and spilled unknown names.
    #[test]
    fn atom_attr_selector_equivalence() {
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        // Uppercased name is lowercased at intern time (setAttribute).
        doc.set_attribute(div, "DATA-X", "1");
        // Unknown name: spill path.
        doc.set_attribute(div, "data-unknown-attr", "2");

        assert!(m("[data-x]", &doc, div));
        assert!(m("[DATA-X]", &doc, div));
        assert!(m("[data-x=\"1\"]", &doc, div));
        assert!(!m("[data-x=\"2\"]", &doc, div));
        assert!(m("[data-unknown-attr]", &doc, div));
        assert!(m("[DATA-UNKNOWN-ATTR]", &doc, div));
        assert!(m("[data-unknown-attr=\"2\"]", &doc, div));
        assert!(!m("[data-y]", &doc, div));
        // Unknown selector name vs well-known element attr: no match.
        assert!(!m("[data-zzz]", &doc, div));
        // `get_attribute` honors the same rule through atoms.
        assert_eq!(doc.get_attribute(div, "DATA-X"), Some("1"));
        assert_eq!(doc.get_attribute(div, "data-unknown-attr"), Some("2"));
        assert_eq!(doc.get_attribute(div, "DATA-UNKNOWN-ATTR"), Some("2"));
    }

    /// Id and class matching through the precomputed atom fast path.
    #[test]
    fn atom_id_class_matching() {
        let (doc, div, _, _, _) = sample();
        assert!(m("#d", &doc, div));
        assert!(m(".a", &doc, div));
        assert!(m("div#d.a", &doc, div));
        assert!(m("DIV#d", &doc, div));
        assert!(!m("#e", &doc, div));
        // Class *values* stay case-sensitive.
        assert!(!m(".b", &doc, div));
        assert!(m(".B", &doc, div));
    }

    /// `:nth-of-type` grouping by atom equality, including spilled
    /// custom-element names and non-canonical HTML names.
    #[test]
    fn atom_nth_of_type_grouping() {
        let mut doc = Document::new();
        let root = doc.root();
        let wrap = doc.create_element("div");
        doc.append_child(root, wrap);
        let a = doc.create_element("my-widget");
        doc.append_child(wrap, a);
        let b = doc.create_element("my-widget");
        doc.append_child(wrap, b);
        assert!(m("my-widget:nth-of-type(1)", &doc, a));
        assert!(m("my-widget:nth-of-type(2)", &doc, b));
        assert!(!m("my-widget:nth-of-type(2)", &doc, a));
        assert!(m("my-widget:first-of-type", &doc, a));
        assert!(m("my-widget:last-of-type", &doc, b));

        // Non-canonical HTML names (createElementNS keeps case) still
        // group case-insensitively via the spill fallback.
        let c = doc.create_element_ns(Namespace::Html, "DIV");
        doc.append_child(wrap, c);
        let d = doc.create_element("div");
        doc.append_child(wrap, d);
        assert!(m("div:nth-of-type(1)", &doc, c));
        assert!(m("div:nth-of-type(2)", &doc, d));
        assert!(m("DIV:nth-of-type(1)", &doc, wrap));
    }

    // ---- shared match scratch ----

    /// A div with a 6-deep span tree (4 span children per node), like the
    /// `tests/has_alloc.rs` fixture: deep enough for 5+-compound `:has()`
    /// selectors to hit and miss.
    fn deep_tree() -> (Document, Vec<NodeId>) {
        let mut doc = Document::new();
        let root = doc.root();
        let outer = doc.create_element("div");
        doc.append_child(root, outer);
        let mut all = vec![outer];
        let mut parents = vec![outer];
        for _ in 0..6 {
            let mut next = Vec::new();
            for &p in &parents {
                for _ in 0..4 {
                    let el = doc.create_element("span");
                    doc.append_child(p, el);
                    next.push(el);
                    all.push(el);
                }
            }
            parents = next;
        }
        (doc, all)
    }

    /// The shared [`MatchScratch`] amortizes the matcher scratch buffers:
    /// after one warmup pass, repeated [`matches_cascade`] calls over the
    /// whole tree reuse the buffers without growing them — including the
    /// shapes that used to allocate per call:
    /// * a 5+-compound `:has()` (failed-state memo, miss-heavy: inserts on
    ///   every element),
    /// * a multi-compound outer selector (backtracking choice-point
    ///   stack),
    /// * a `:has()` argument holding a nested complex selector
    ///   (`:has(:is(a b))`: a nested `match_complex` mid-match).
    ///
    /// White-box capacity assertions instead of a counting allocator
    /// (the crate forbids `unsafe_code`, so no custom global allocator):
    /// any regression to per-call allocation leaves the scratch buffers
    /// at capacity 0 (or shrinks them), failing the assertions. The
    /// one-shot [`matches`] API builds a fresh scratch per call, so this
    /// amortization is only observable through a shared scratch (here;
    /// the cascade's `Styler` in production).
    #[test]
    fn match_scratch_reused_across_matches() {
        let (doc, all) = deep_tree();
        // 6-compound relative selector, miss-heavy: every span exhausts
        // the search and records memo failures.
        let long_has_miss = sel("span:has(> span > span > span > span > zzz)");
        // 5-compound relative selector (memo engaged): hit on the level-0
        // spans, miss below.
        let long_has_hit = sel("span:has(> span > span > span > span > span)");
        // Multi-compound outer selector: backtracking stack per element.
        let multi_outer = sel("div span");
        // Nested complex selector inside a `:has()` argument: a nested
        // `match_complex` runs while the outer match is in progress.
        let nested_complex = sel("div:has(:is(span span))");

        // Behavior sanity through the one-shot API.
        assert!(
            all.iter()
                .all(|&n| !m("span:has(> span > span > span > span > zzz)", &doc, n))
        );
        assert!(m(
            "span:has(> span > span > span > span > span)",
            &doc,
            all[1]
        )); // level-0: hit
        assert!(!m(
            "span:has(> span > span > span > span > span)",
            &doc,
            all[5]
        )); // level-1: miss
        assert!(!m(
            "span:has(> span > span > span > span > span)",
            &doc,
            *all.last().unwrap()
        )); // leaf: miss
        assert!(m("div span", &doc, all[1]));
        assert!(!m("div span", &doc, all[0]));
        assert!(m("div:has(:is(span span))", &doc, all[0]));
        assert!(!m("div:has(:is(span span))", &doc, all[1]));

        let mut scratch = MatchScratch::new();
        let state_for = |_: NodeId| ElementState::default();
        let run_all = |scratch: &mut MatchScratch| {
            let mut evals = 0usize;
            for &n in &all {
                let _ = matches_cascade(&long_has_miss, &doc, n, &state_for, None, scratch, None);
                let _ = matches_cascade(&long_has_hit, &doc, n, &state_for, None, scratch, None);
                let _ = matches_cascade(&multi_outer, &doc, n, &state_for, None, scratch, None);
                let _ = matches_cascade(&nested_complex, &doc, n, &state_for, None, scratch, None);
                evals += 4;
            }
            evals
        };
        // Warmup: entry-point `warm_up` pre-sizes the buffers, and the
        // first miss-heavy pass grows the memo to its working size.
        // `frames`/`rel_frames` must already be warm-sized (no growth
        // from the backtrack pushes below).
        let evals = run_all(&mut scratch);
        assert!(evals > 20000, "fixture too small to be meaningful");
        let frames_cap = scratch.frames.capacity();
        let failed_cap = scratch.failed.capacity();
        let rel_cap = scratch.rel_frames.capacity();
        assert!(frames_cap > 0, "warmup should size the backtrack stack");
        assert!(failed_cap > 0, "warmup should size the failed-state memo");
        assert!(rel_cap > 0, "warmup should size the relative-search stack");

        // Measured pass: every match reuses the shared buffers — none
        // may grow (growth = a heap allocation).
        let evals = run_all(&mut scratch);
        assert!(evals > 20000, "fixture too small to be meaningful");
        assert_eq!(
            scratch.frames.capacity(),
            frames_cap,
            "backtracking stack reallocated during matching"
        );
        assert_eq!(
            scratch.failed.capacity(),
            failed_cap,
            "failed-state memo reallocated during matching"
        );
        assert_eq!(
            scratch.rel_frames.capacity(),
            rel_cap,
            "relative-search stack reallocated during matching"
        );
    }
}

/// Wave 65 perf builder: id/class rule-index buckets keyed by interned
/// `u32` (one FxHash per element/token plus a `Vec` index) instead of
/// SipHash-hashing full `String`s per element, and FxHash for the hot
/// maps in this file.
#[cfg(test)]
mod w65_p_match_tests {
    use super::*;
    use crate::parse::parse_stylesheet;

    fn data_of(doc: &Document, node: NodeId) -> ElementData {
        match doc.kind(node) {
            NodeKind::Element(d) => d.clone(),
            _ => panic!("expected an element"),
        }
    }

    fn cand_pairs(index: &RuleIndex, doc: &Document, node: NodeId) -> Vec<(usize, usize)> {
        let data = data_of(doc, node);
        let mut refs = Vec::new();
        index.candidates_into(&data, &mut refs);
        refs.into_iter().map(|r| (r.rule, r.selector)).collect()
    }

    #[test]
    fn id_bucket_exact_and_case_sensitive() {
        let sheet = parse_stylesheet("#a { color: red; } #A { color: blue; } div { color: gray; }");
        let index = RuleIndex::build(&sheet);
        let mut doc = Document::new();
        let root = doc.root();
        let lower = doc.create_element("div");
        doc.append_child(root, lower);
        doc.set_attribute(lower, "id", "a");
        assert_eq!(cand_pairs(&index, &doc, lower), vec![(0, 0), (2, 0)]);
        let upper = doc.create_element("div");
        doc.append_child(root, upper);
        doc.set_attribute(upper, "id", "A");
        assert_eq!(cand_pairs(&index, &doc, upper), vec![(1, 0), (2, 0)]);
        // An id no selector names contributes nothing.
        let other = doc.create_element("div");
        doc.append_child(root, other);
        doc.set_attribute(other, "id", "zzz");
        assert_eq!(cand_pairs(&index, &doc, other), vec![(2, 0)]);
    }

    #[test]
    fn class_buckets_case_sensitive_and_multi_token() {
        let sheet =
            parse_stylesheet(".b { color: red; } .B { color: blue; } #x.y.z { color: green; }");
        let index = RuleIndex::build(&sheet);
        let mut doc = Document::new();
        let root = doc.root();
        let el = doc.create_element("span");
        doc.append_child(root, el);
        doc.set_attribute(el, "id", "x");
        doc.set_attribute(el, "class", "b B y z");
        // `.b`, `.B`, `#x.y.z` (via id bucket and via class buckets y, z —
        // deduped to one ref).
        assert_eq!(cand_pairs(&index, &doc, el), vec![(0, 0), (1, 0), (2, 0)]);
        // Class tokens are exact: `b` does not see `.B`'s bucket.
        let el2 = doc.create_element("span");
        doc.append_child(root, el2);
        doc.set_attribute(el2, "class", "B");
        assert_eq!(cand_pairs(&index, &doc, el2), vec![(1, 0)]);
    }

    #[test]
    fn shared_id_bucket_holds_every_selector() {
        let sheet = parse_stylesheet("#same { color: red; } div#same { color: blue; }");
        let index = RuleIndex::build(&sheet);
        let mut doc = Document::new();
        let root = doc.root();
        let el = doc.create_element("div");
        doc.append_child(root, el);
        doc.set_attribute(el, "id", "same");
        // Both selectors name `#same`: one interned key, one bucket, both
        // refs (the tag-bucketed second rule also appears via `div`).
        assert_eq!(cand_pairs(&index, &doc, el), vec![(0, 0), (1, 0)]);
    }

    #[test]
    fn id_value_interned_only_as_class_is_safe() {
        // `.foo` interns "foo"; an element with `id="foo"` must probe
        // the (empty) id bucket without panicking and find nothing.
        let sheet = parse_stylesheet(".foo { color: red; }");
        let index = RuleIndex::build(&sheet);
        let mut doc = Document::new();
        let root = doc.root();
        let el = doc.create_element("div");
        doc.append_child(root, el);
        doc.set_attribute(el, "id", "foo");
        assert_eq!(cand_pairs(&index, &doc, el), vec![]);
        // And the class side still works.
        let el2 = doc.create_element("div");
        doc.append_child(root, el2);
        doc.set_attribute(el2, "class", "foo");
        assert_eq!(cand_pairs(&index, &doc, el2), vec![(0, 0)]);
    }

    #[test]
    fn fxhashmap_smoke() {
        let mut m: FxHashMap<String, u32> = FxHashMap::default();
        m.insert("alpha".to_owned(), 1);
        m.insert("beta".to_owned(), 2);
        assert_eq!(m.get("alpha"), Some(&1));
        assert_eq!(m.get("nope"), None);
        let mut n: FxHashMap<u32, &str> = FxHashMap::default();
        for i in 0..1000u32 {
            n.insert(i, "v");
        }
        assert_eq!(n.len(), 1000);
        assert_eq!(n.get(&999), Some(&"v"));
        let mut s: FxHashSet<(u32, NodeId)> = FxHashSet::default();
        let mut doc = Document::new();
        let el = doc.create_element("div");
        doc.append_child(doc.root(), el);
        s.insert((7, el));
        assert!(s.contains(&(7, el)));
    }

    /// Throughput probe for the element-side bucket lookup
    /// ([`RuleIndex::for_each_bucket`]): 2000 elements x 600 selectors.
    /// Print-only; run with `-- --nocapture` to see the numbers.
    #[test]
    fn bucket_lookup_throughput() {
        let mut css = String::new();
        for i in 0..200 {
            css.push_str(&format!("#id-{i} {{ color: red; }}\n"));
            css.push_str(&format!(".cls-{i} {{ color: red; }}\n"));
            css.push_str(&format!("div#id-{i}.cls-{i} {{ color: red; }}\n"));
        }
        let sheet = parse_stylesheet(&css);
        let index = RuleIndex::build(&sheet);
        let mut doc = Document::new();
        let root = doc.root();
        let mut datas = Vec::with_capacity(2000);
        for i in 0..2000 {
            let el = doc.create_element("div");
            doc.append_child(root, el);
            doc.set_attribute(el, "id", &format!("id-{}", i % 200));
            doc.set_attribute(el, "class", &format!("cls-{} other", i % 200));
            datas.push(data_of(&doc, el));
        }
        // Warmup.
        let mut sink = 0usize;
        for d in &datas {
            index.for_each_bucket(d, |s| sink += s.len());
        }
        let start = std::time::Instant::now();
        let rounds = 200usize;
        for _ in 0..rounds {
            for d in &datas {
                index.for_each_bucket(d, |s| sink += s.len());
            }
        }
        let dt = start.elapsed();
        let total = (datas.len() * rounds) as f64;
        eprintln!(
            "w65 for_each_bucket: {:.1} ns/element (sink {sink})",
            dt.as_nanos() as f64 / total
        );
        // End-to-end through the merge as well.
        let mut buf = Vec::new();
        let start = std::time::Instant::now();
        for _ in 0..rounds {
            for d in &datas {
                index.candidates_into(d, &mut buf);
                sink += buf.len();
            }
        }
        let dt = start.elapsed();
        eprintln!(
            "w65 candidates_into: {:.1} ns/element (sink {sink})",
            dt.as_nanos() as f64 / total
        );
    }
}
