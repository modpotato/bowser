//! Allocation accounting for `:has()` / `:empty` selector matching.
//!
//! The matcher traverses the DOM through `Document::first_child` /
//! `Document::next_sibling` cursors instead of `Document::children()`'s
//! per-call `Vec`, so evaluating `:has()` (and `:empty`) must not
//! allocate per visited node. This test enforces that with a thread-local
//! counting global allocator: parsing and fixture building happen before
//! the probe is armed, and the measured region — matching `:has()`/`:empty`
//! selectors against every element of a ~5k-node tree — must allocate zero
//! heap bytes. The measured region covers multi-compound relative
//! selectors, `+`/`~` sibling combinators (hit and miss), and a
//! 5-compound (memo-engaging) `:has()` hit; miss-heavy long selectors,
//! multi-compound outer selectors, and nested complex selectors inside
//! `:has()` arguments (e.g. `div:has(:is(span span))`) allocate on first
//! memo insert / first backtrack push per scratch by design, and their
//! amortization across a shared scratch is pinned by
//! `long_selector_matching_amortizes_to_zero_alloc` below (public
//! [`matches_with_scratch`](bowser_css::matches_with_scratch) entry with
//! a caller-owned scratch) and white-box by
//! `match_scratch_reused_across_matches` in `src/matching.rs`.
//!
//! The one-shot [`matches`](bowser_css::matches) entry shares a warm
//! per-thread scratch, so the measured region below observes the
//! amortized (warm) behavior; the behavior checks before it do the
//! warming, outside the probe window.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use bowser_css::{MatchScratch, matches, matches_with_scratch, parse_selector_list};
use bowser_dom::{Document, NodeId};

thread_local! {
    static BYTES: Cell<usize> = const { Cell::new(0) };
    static ARMED: Cell<bool> = const { Cell::new(false) };
}

/// Counting global allocator: forwards to [`System`], tallying bytes on
/// the armed thread only, so parallel test threads cannot pollute the
/// measurement.
struct Probe;

// SAFETY: `Probe` is stateless; `alloc`/`dealloc` forward to `System`,
// whose contract (valid layout, non-null on success, matching dealloc)
// is upheld verbatim. The tally touches only thread-local state.
unsafe impl GlobalAlloc for Probe {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.with(Cell::get) {
            BYTES.with(|b| b.set(b.get().saturating_add(layout.size())));
        }
        // SAFETY: same contract as `System::alloc`.
        unsafe { System.alloc(layout) }
    }

    // SAFETY: forwarded unchanged to `System::dealloc`.
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: caller upholds `GlobalAlloc::dealloc`'s contract, which
        // is exactly `System::dealloc`'s.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Probe = Probe;

/// Heap bytes allocated on this thread while `f` runs.
fn bytes_allocated_while(f: impl FnOnce()) -> usize {
    BYTES.with(|b| b.set(0));
    ARMED.with(|a| a.set(true));
    f();
    ARMED.with(|a| a.set(false));
    BYTES.with(Cell::get)
}

/// A div tree, 6 levels deep with 4 span children per node: every div is
/// a `:has(> span)` witness and no leaf matches `:has(> zzz)`.
fn fixture() -> (Document, NodeId, Vec<NodeId>) {
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
    (doc, outer, all)
}

/// Naive reference for `div:has(> span)`: does any *direct* child have
/// the `span` tag? Built with the allocating `children()` API, outside
/// the probe window.
fn reference_has_gt_span(doc: &Document, node: NodeId) -> bool {
    fn tag(doc: &Document, n: NodeId) -> Option<String> {
        match doc.kind(n) {
            bowser_dom::NodeKind::Element(d) => Some(d.tag_name.to_string()),
            _ => None,
        }
    }
    if tag(doc, node).as_deref() != Some("div") {
        return false;
    }
    doc.children(node)
        .iter()
        .any(|&c| tag(doc, c).as_deref() == Some("span"))
}

#[test]
fn has_matching_allocates_nothing_per_node() {
    let (doc, _outer, all) = fixture();
    let has_gt_span = &parse_selector_list("div:has(> span)").unwrap()[0];
    let has_desc_span = &parse_selector_list("div:has(span)").unwrap()[0];
    let has_gt_zzz = &parse_selector_list("div:has(> zzz)").unwrap()[0];
    let empty = &parse_selector_list("div:empty").unwrap()[0];
    // Multi-compound relative selectors: exercise the depth-first
    // multi-level path (`matches_relative` with 2+ compounds), which
    // must also allocate nothing.
    let has_gt_span_gt_span = &parse_selector_list("span:has(> span > span)").unwrap()[0];
    let has_span_gt_span = &parse_selector_list("div:has(span > span)").unwrap()[0];
    let has_gt_span_gt_zzz = &parse_selector_list("span:has(> span > zzz)").unwrap()[0];
    // Long relative selectors (5+ compounds): engage the failed-state
    // memo in `matches_relative`. The hit below records no failures on
    // its first-tried witness path, so it must still allocate nothing;
    // the miss records failures (memo inserts), which is pinned
    // white-box instead — see `match_scratch_reused_across_matches` in
    // `src/matching.rs`, since the one-shot API builds a fresh scratch
    // per call and cannot observe amortization.
    let has_5hit = &parse_selector_list("div:has(> span > span > span > span > span)").unwrap()[0];
    let has_5miss = &parse_selector_list("div:has(> span > span > span > span > zzz)").unwrap()[0];
    // Multi-compound outer selectors: exercise the backtracking
    // choice-point stack in `match_complex`.
    let outer_multi = &parse_selector_list("div span:has(> span)").unwrap()[0];
    // Nested complex selector inside a `:has()` argument: a nested
    // `match_complex` runs while the outer match is in progress.
    let nested_complex = &parse_selector_list("div:has(:is(span span))").unwrap()[0];
    // Sibling combinators: short selectors (no failed-state memo), both
    // hit and miss paths must allocate nothing. Spans are siblings of
    // each other in the fixture (4 per parent), so every span but the
    // last child of its parent matches `:has(+ span)` / `:has(~ span)`;
    // `:has(+ span > span)` exercises the multi-compound depth-first
    // path through a sibling step.
    let has_plus_span = &parse_selector_list("span:has(+ span)").unwrap()[0];
    let has_tilde_span = &parse_selector_list("span:has(~ span)").unwrap()[0];
    let has_plus_span_gt_span = &parse_selector_list("span:has(+ span > span)").unwrap()[0];
    let has_tilde_zzz = &parse_selector_list("span:has(~ zzz)").unwrap()[0];

    // Behavior check first (allocating reference is fine here): the
    // cursor-based matcher agrees with the naive `children()`-based
    // reference on the whole deep tree.
    for &n in &all {
        assert_eq!(
            matches(has_gt_span, &doc, n),
            reference_has_gt_span(&doc, n),
            "mismatch at {n:?}"
        );
    }
    // Sanity on the other shapes: descendant combinator hits everywhere a
    // div stands, the miss never matches, and nothing is `:empty`.
    assert!(
        all.iter()
            .filter(|&&n| matches!(doc.kind(n), bowser_dom::NodeKind::Element(_)))
            .all(|&n| !matches(empty, &doc, n) || !reference_has_gt_span(&doc, n))
    );
    assert!(matches(has_desc_span, &doc, all[0]));
    assert!(!matches(has_gt_zzz, &doc, all[0]));
    assert!(!matches(empty, &doc, all[0]));
    // Multi-compound behavior against a naive reference: a span matches
    // `span:has(> span > span)` iff it has a child span with a span child;
    // the fixture is 6 span levels deep, so levels 0..4 match. The miss
    // never matches.
    for (i, &n) in all.iter().enumerate() {
        assert_eq!(
            matches(has_gt_span_gt_span, &doc, n),
            reference_has_gt_span_gt_span(&doc, n),
            "multi-compound mismatch at {n:?} (index {i})"
        );
    }
    assert!(matches(has_span_gt_span, &doc, all[0]));
    assert!(!matches(has_gt_span_gt_zzz, &doc, all[0]));
    // Long relative selectors: the 5-compound hit matches the outer div
    // (five span generations below it); the miss never matches.
    assert!(matches(has_5hit, &doc, all[0]));
    assert!(!matches(has_5miss, &doc, all[0]));
    // Multi-compound outer: level-0 spans match `div span:has(> span)`
    // (div ancestor, span child); leaf spans have no span child.
    assert!(matches(outer_multi, &doc, all[1]));
    assert!(!matches(outer_multi, &doc, *all.last().unwrap()));
    assert!(!matches(outer_multi, &doc, all[0]));
    // Nested complex selector in the `:has()` argument: the div has
    // descendant spans that themselves have span ancestors.
    assert!(matches(nested_complex, &doc, all[0]));
    assert!(!matches(nested_complex, &doc, all[1]));
    // Sibling behavior: all[1] is the first span child of the outer
    // div, so its next sibling (all[2]) is a span; the last node has
    // no next sibling at all.
    let last = *all.last().unwrap();
    assert!(matches(has_plus_span, &doc, all[1]));
    assert!(!matches(has_plus_span, &doc, last));
    assert!(matches(has_tilde_span, &doc, all[1]));
    assert!(!matches(has_tilde_span, &doc, last));
    // `span:has(+ span > span)`: all[2] is a non-leaf span, so all[1]
    // matches; the leaf span does not.
    assert!(matches(has_plus_span_gt_span, &doc, all[1]));
    assert!(!matches(has_plus_span_gt_span, &doc, last));
    assert!(!matches(has_tilde_zzz, &doc, all[1]));

    // The measured region: match every `:has()`/`:empty` selector against
    // every element. Must allocate zero heap bytes.
    //
    // `has_5hit` is the only long (5+ compound) relative selector in the
    // measured region: its first-tried witness path succeeds, so the
    // failed-state memo records nothing and the warm scratch stays
    // heap-free. The miss-heavy long selector and the multi-compound
    // outer selector need one warmup pass to size the memo / backtrack
    // stack, so they stay outside this region by design; their
    // amortization is pinned by `long_selector_matching_amortizes_to_zero_alloc`
    // below (public scratch entry) and white-box by
    // `match_scratch_reused_across_matches` in `src/matching.rs`.
    //
    // `nested_complex` (`div:has(:is(span span))`) is likewise outside
    // the measured region by design: its nested `:is()` argument is a
    // 2-compound complex selector whose first backtrack choice-point
    // push sizes the scratch's stack on first use. The one-shot entry
    // shares a warm per-thread scratch, so that push amortizes away
    // across calls — pinned by the amortized test below.
    let bytes = bytes_allocated_while(|| {
        let mut evals = 0usize;
        for &n in &all {
            let _ = matches(has_gt_span, &doc, n);
            let _ = matches(has_desc_span, &doc, n);
            let _ = matches(has_gt_zzz, &doc, n);
            let _ = matches(empty, &doc, n);
            let _ = matches(has_gt_span_gt_span, &doc, n);
            let _ = matches(has_span_gt_span, &doc, n);
            let _ = matches(has_gt_span_gt_zzz, &doc, n);
            let _ = matches(has_5hit, &doc, n);
            let _ = matches(has_plus_span, &doc, n);
            let _ = matches(has_tilde_span, &doc, n);
            let _ = matches(has_plus_span_gt_span, &doc, n);
            let _ = matches(has_tilde_zzz, &doc, n);
            evals += 12;
        }
        assert!(evals > 20000, "fixture too small to be meaningful");
    });
    assert_eq!(
        bytes, 0,
        ":has()/:empty matching allocated {bytes} heap bytes"
    );
}

/// Naive reference for `span:has(> span > span)`: does any direct child
/// span itself have a direct child span? Built with the allocating
/// `children()` API, outside the probe window.
fn reference_has_gt_span_gt_span(doc: &Document, node: NodeId) -> bool {
    fn tag(doc: &Document, n: NodeId) -> Option<String> {
        match doc.kind(n) {
            bowser_dom::NodeKind::Element(d) => Some(d.tag_name.to_string()),
            _ => None,
        }
    }
    if tag(doc, node).as_deref() != Some("span") {
        return false;
    }
    doc.children(node).iter().any(|&c| {
        tag(doc, c).as_deref() == Some("span")
            && doc
                .children(c)
                .iter()
                .any(|&g| tag(doc, g).as_deref() == Some("span"))
    })
}

/// Amortization of the long-selector path through the public scratch
/// entry: shapes that allocate on a cold scratch by design (first memo
/// insert / first backtrack push size the buffers) must allocate zero
/// heap bytes once the caller-owned scratch is warm.
///
/// This is the integration-level pin for the allocation the one-shot
/// path amortizes away: the miss-heavy 6-compound `:has()` (memo
/// inserts), the multi-compound outer selector (backtracking stack), and
/// the nested complex selector inside a `:has()` argument (nested
/// backtracking while the outer search is in progress).
#[test]
fn long_selector_matching_amortizes_to_zero_alloc() {
    let (doc, _outer, all) = fixture();
    // Miss-heavy long relative selector: every node exhausts the search
    // and records memo failures.
    let has_6miss = &parse_selector_list("div:has(> span > span > span > span > zzz)").unwrap()[0];
    // Long relative selector, hit on the first-tried witness path.
    let has_6hit = &parse_selector_list("div:has(> span > span > span > span > span)").unwrap()[0];
    // Multi-compound outer selector: backtracking choice-point stack.
    let outer_multi = &parse_selector_list("div span:has(> span)").unwrap()[0];
    // Nested complex selector inside a `:has()` argument: a nested
    // `match_complex` runs while the outer match is in progress.
    let nested_complex = &parse_selector_list("div:has(:is(span span))").unwrap()[0];

    let mut scratch = MatchScratch::new();

    // Behavior sanity through the scratch entry (unmeasured).
    assert!(matches_with_scratch(has_6hit, &doc, all[0], &mut scratch));
    assert!(!matches_with_scratch(has_6miss, &doc, all[0], &mut scratch));
    assert!(matches_with_scratch(
        outer_multi,
        &doc,
        all[1],
        &mut scratch
    ));
    assert!(!matches_with_scratch(
        outer_multi,
        &doc,
        *all.last().unwrap(),
        &mut scratch
    ));
    assert!(!matches_with_scratch(
        outer_multi,
        &doc,
        all[0],
        &mut scratch
    ));
    assert!(matches_with_scratch(
        nested_complex,
        &doc,
        all[0],
        &mut scratch
    ));
    assert!(!matches_with_scratch(
        nested_complex,
        &doc,
        all[1],
        &mut scratch
    ));

    // Warmup pass (unmeasured): sizes the failed-state memo to its
    // working capacity and warms both choice-point stacks.
    let run_all = |scratch: &mut MatchScratch| {
        let mut evals = 0usize;
        for &n in &all {
            let _ = matches_with_scratch(has_6miss, &doc, n, scratch);
            let _ = matches_with_scratch(has_6hit, &doc, n, scratch);
            let _ = matches_with_scratch(outer_multi, &doc, n, scratch);
            let _ = matches_with_scratch(nested_complex, &doc, n, scratch);
            evals += 4;
        }
        evals
    };
    assert!(run_all(&mut scratch) > 20000, "fixture too small");

    // Measured pass: every match reuses the warm buffers — zero heap
    // bytes, including the miss-heavy long selector whose memo is
    // cleared and rebuilt per call (capacity retained, no growth).
    let bytes = bytes_allocated_while(|| {
        assert!(run_all(&mut scratch) > 20000, "fixture too small");
    });
    assert_eq!(
        bytes, 0,
        "long-selector matching allocated {bytes} heap bytes on a warm scratch"
    );
}
