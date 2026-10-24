//! Allocation micro-checks for tag/attribute name interning.
//!
//! The point of atomizing names is removing heap work from the parse
//! path: interning a well-known name must not allocate at all. These are
//! the exact interning calls the atomized `start_element_atomized` path
//! (and the `String` `start_element` fallback feeding it) make, so a
//! zero count here is the zero-alloc property the atomization refactor
//! exists for. A thread-local counting allocator observes only the test
//! thread, so parallel tests in this binary cannot pollute the counts.
//!
//! Node storage (`Document::alloc`, the attribute `Vec`) still allocates;
//! that is out of scope — this pins the *name interning* cost only.

use bowser_dom::{AttrName, NameAtom, TagName};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAlloc;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAlloc = CountingAlloc;

/// Allocations on this thread while `f` runs.
fn allocs_during(f: impl FnOnce()) -> usize {
    let before = ALLOCS.with(|c| c.get());
    f();
    ALLOCS.with(|c| c.get()) - before
}

#[test]
fn well_known_names_intern_without_allocating() {
    // Borrowed interning of already-lowercase names (the hot
    // `start_element_atomized` path after the tokenizer lowercases):
    // pure atoms, no heap.
    let n = allocs_during(|| {
        for _ in 0..10_000 {
            let t = TagName::intern_lowered("div");
            let a = AttrName::intern_lowered("id");
            let b = AttrName::intern_lowered("class");
            std::hint::black_box((t, a, b));
        }
    });
    assert_eq!(n, 0, "well-known names must intern allocation-free");
    // `intern` is case-sensitive: the uppercase form misses the table.
    let n = allocs_during(|| {
        for _ in 0..1_000 {
            std::hint::black_box(AttrName::intern("ID"));
        }
    });
    assert!(n > 0, "case-sensitive intern of an uppercase name spills");

    // `intern_lowered` on an uppercase HTML name: the ASCII-lowercasing
    // allocates one temporary `String`; interning itself adds nothing.
    let n = allocs_during(|| {
        for _ in 0..1_000 {
            std::hint::black_box(AttrName::intern_lowered("ID"));
        }
    });
    assert_eq!(
        n, 1_000,
        "only the lowering temporary may allocate, never the intern"
    );

    // Owned interning of an already-lowercase `String` (the `String`
    // `start_element` fallback path): the `String` is dropped, nothing
    // new is allocated.
    let owned: Vec<String> = (0..1_000).map(|_| "span".to_string()).collect();
    let n = allocs_during(|| {
        for s in &owned {
            std::hint::black_box(TagName::from(s.clone()));
        }
    });
    // Each `s.clone()` allocates the `String` under test; interning
    // itself must add nothing on top.
    assert_eq!(n, 1_000, "owned interning of well-known names is free");
}

#[test]
fn unknown_names_spill_onto_the_sidecar() {
    // Unknown names (custom elements, `data-*`) cannot be atoms: each
    // interning copies onto the heap sidecar exactly once.
    let n = allocs_during(|| {
        for _ in 0..1_000 {
            let t = TagName::intern_lowered("my-element");
            let a = AttrName::intern_lowered("data-custom-thing");
            assert_eq!(t.atom(), NameAtom::UNKNOWN);
            assert_eq!(a.atom(), NameAtom::UNKNOWN);
            std::hint::black_box((t, a));
        }
    });
    assert_eq!(n, 2_000, "unknown names spill exactly once each");

    // Owned interning moves the `String` into the sidecar: zero-copy,
    // no additional allocation.
    let owned: Vec<String> = (0..1_000).map(|_| "my-attr".to_string()).collect();
    let n = allocs_during(|| {
        for s in &owned {
            let a = AttrName::from(s.clone());
            assert_eq!(a.as_str(), "my-attr");
            std::hint::black_box(a);
        }
    });
    assert_eq!(n, 1_000, "owned interning must not copy the sidecar");
}
