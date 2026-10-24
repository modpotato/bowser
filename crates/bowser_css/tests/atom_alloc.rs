//! Allocation micro-checks for property-name interning.
//!
//! The point of atomizing property names is removing heap work from the
//! parse and cascade paths: interning a well-known name must not allocate
//! at all. A thread-local counting allocator observes only the test
//! thread, so parallel tests in this binary cannot pollute the counts.

use bowser_css::{PropertyAtom, PropertyName};
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
    // Borrowed interning of already-lowercase names: pure atom, no heap.
    let n = allocs_during(|| {
        for _ in 0..10_000 {
            let a = PropertyName::intern("margin-top");
            let b = PropertyName::intern_lowered("border-top-width");
            std::hint::black_box((a, b));
        }
    });
    assert_eq!(n, 0, "well-known names must intern allocation-free");

    // Owned interning of an already-lowercase `String` (the parse fast
    // path): the `String` is dropped, nothing new is allocated.
    let owned: Vec<String> = (0..1_000).map(|_| "padding-left".to_string()).collect();
    let n = allocs_during(|| {
        for s in &owned {
            std::hint::black_box(PropertyName::intern_lowered_owned(s.clone()));
        }
    });
    // Each `s.clone()` allocates the `String` under test; interning
    // itself must add nothing on top.
    assert_eq!(
        n, 1_000,
        "interning must not allocate beyond the input String"
    );
}

#[test]
fn uppercase_names_lowercase_in_place() {
    // An uppercase owned name is lowered with `make_ascii_lowercase`
    // (in place) rather than `to_ascii_lowercase` (fresh `String`): the
    // only allocation is the input `String` itself.
    let owned: Vec<String> = (0..1_000).map(|_| "MARGIN-TOP".to_string()).collect();
    let n = allocs_during(|| {
        for s in &owned {
            let p = PropertyName::intern_lowered_owned(s.clone());
            assert_eq!(p.atom(), PropertyAtom::MARGIN_TOP);
            std::hint::black_box(p);
        }
    });
    assert_eq!(n, 1_000, "lowercasing must happen in place, not via a copy");
}

#[test]
fn unknown_names_take_exactly_one_sidecar_alloc() {
    // Unknown (custom) names still need one heap sidecar, and only one.
    let names: Vec<String> = (0..1_000).map(|i| format!("--custom-{i}")).collect();
    let n = allocs_during(|| {
        for name in &names {
            let p = PropertyName::intern(name);
            assert_eq!(p.atom(), PropertyAtom::UNKNOWN);
            assert_eq!(p.as_str(), name.as_str());
            std::hint::black_box(p);
        }
    });
    assert_eq!(n, 1_000, "one sidecar alloc per unknown name");
}
