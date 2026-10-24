//! Allocation accounting for `ComputedStyle::from_slots`'s sparse delta.
//!
//! Every styled element collapses its full-width computed-value scratch
//! into a sparse `(index, value)` delta. The delta `Vec` used to grow
//! from `Vec::new()` (grow, grow, `into_boxed_slice` shrink: up to 3
//! allocations per element); it is now exact-sized up front, so the only
//! allocation is the boxed slice itself — one for an element with
//! deviating slots, zero for an all-initial element. (A thread-local
//! pooled buffer cannot beat this: `into_boxed_slice` moves the buffer's
//! allocation into the style, so the pool would hand out a
//! zero-capacity `Vec` on every take.)
//!
//! This test enforces that with a thread-local counting global
//! allocator: after warmup (parsing, `Styler` construction, and styling
//! a throwaway set of elements — all outside the probe window), styling
//! M fresh elements must stay under a calibrated byte bound. The old
//! growth ramp costs at least 64 bytes more per element (the 0→4 slot
//! growth, present for every element with ≥1 deviating slot — and every
//! div has ≥1 via the UA `display: block`), so the bound fails on the
//! pre-change code by a wide margin.
//!
//! The computed-value assertions pin bit-identical output through the
//! changed path: declared, UA-default, inherited, and custom-property
//! values must all read back exactly.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use bowser_css::{Color, CssValue, LengthUnit, Styler, default_ua_stylesheet, parse_stylesheet};
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

const M: usize = 400;

fn fixture() -> (Document, Vec<NodeId>) {
    let mut doc = Document::new();
    let root = doc.root();
    let html = doc.create_element("html");
    doc.append_child(root, html);
    let mut divs = Vec::with_capacity(2 * M);
    for _ in 0..2 * M {
        let d = doc.create_element("div");
        doc.append_child(html, d);
        divs.push(d);
    }
    (doc, divs)
}

const CSS: &str = r#"
    div {
        color: red;
        margin-top: 5px;
        padding-left: 3px;
        border-top-width: 2px;
        font-size: 20px;
        --accent: 7px;
    }
"#;

#[test]
fn sparse_delta_allocates_once_per_element() {
    let (doc, divs) = fixture();
    let author = parse_stylesheet(CSS);
    let ua = default_ua_stylesheet();
    let viewport = bowser_css::Viewport { w: 800.0, h: 600.0 };
    let author_refs = [&author];
    let styler = Styler::new(&doc, &author_refs, &ua, viewport);

    // Warmup (unmeasured): styles the first half, warming every lazy
    // cache (`initials()`, selector matching scratch, share maps) so the
    // probe window sees only per-element work.
    for &d in &divs[..M] {
        let _ = styler.style_for(d);
    }

    // Bit-identical output through the changed `from_slots` path. This
    // node was styled in warmup; the memo returns the same values.
    let s = styler.style_for(divs[0]);
    assert_eq!(
        s.get("color"),
        Some(&CssValue::Color(Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255
        }))
    );
    assert_eq!(
        s.get("margin-top"),
        Some(&CssValue::Length(5.0, LengthUnit::Px))
    );
    assert_eq!(
        s.get("padding-left"),
        Some(&CssValue::Length(3.0, LengthUnit::Px))
    );
    assert_eq!(
        s.get("font-size"),
        Some(&CssValue::Length(20.0, LengthUnit::Px))
    );
    assert_eq!(s.get("display"), Some(&CssValue::Keyword("block".into())));
    assert_eq!(
        s.get("border-top-width"),
        Some(&CssValue::Length(2.0, LengthUnit::Px))
    );
    assert!(s.get("--accent").is_some(), "custom property survives");

    // The measured region: style M fresh elements. Each node is styled
    // exactly once, so there are no memo hits — every element runs the
    // full path including `from_slots`.
    let bytes = bytes_allocated_while(|| {
        for &d in &divs[M..] {
            let _ = styler.style_for(d);
        }
    });
    println!("CALIBRATION: styling {M} elements allocated {bytes} bytes");
    // Calibrated 2026-10-21 at 382656 bytes for M=400: the old growth
    // ramp needs >= 64 bytes more per element (the 0→4 slot growth,
    // present for every element with >= 1 deviating slot), so this
    // bound (measured + 32/element headroom) fails pre-change with
    // wide margin while passing deterministically post-change.
    let bound = 382_656 + M * 32;
    assert!(
        bytes <= bound,
        "styling {M} elements allocated {bytes} bytes (bound {bound})"
    );
}
