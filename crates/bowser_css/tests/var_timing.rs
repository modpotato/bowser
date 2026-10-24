//! Timing smoke test for `var()` resolution at cascade scale.
//!
//! Hundreds of elements share long-named custom properties, so
//! `style_for` performs thousands of `var()` resolutions: the interned
//! names and the reused cycle-detection set keep this fast and
//! allocation-lean. (A counting-allocator variant is impractical here:
//! `style_for`'s fixed per-node allocations — the 80-slot computed-value
//! array above all — dwarf the per-`var()` bytes through the public API,
//! so the byte bound cannot isolate the fix. The interner dedup and the
//! scratch-set reuse are pinned white-box by
//! `var_interner_dedups_and_scratch_set_reused` in cascade.rs instead.)

use bowser_css::{Styler, default_ua_stylesheet, parse_stylesheet};
use bowser_dom::{Document, NodeId};

/// 200 divs; the stylesheet defines 20 long-named custom properties on a
/// shared ancestor and every div references all 20: 4000 `var()`
/// resolutions through one `Styler`.
fn fixture() -> (Document, Vec<NodeId>, String) {
    let mut doc = Document::new();
    let root = doc.root();
    let css = {
        let mut css = String::from(":root { ");
        for i in 0..20 {
            css.push_str(&format!(
                "--theme-variable-number-{i:02}-with-a-long-name: {i}px; "
            ));
        }
        css.push_str("} div { ");
        for i in 0..20 {
            css.push_str(&format!(
                "--ref-{i:02}: var(--theme-variable-number-{i:02}-with-a-long-name); "
            ));
        }
        css.push_str("margin-top: var(--ref-00); }");
        css
    };
    let html = doc.create_element("html");
    doc.append_child(root, html);
    let mut divs = Vec::new();
    for _ in 0..200 {
        let d = doc.create_element("div");
        doc.append_child(html, d);
        divs.push(d);
    }
    (doc, divs, css)
}

#[test]
fn many_var_references_across_elements_stay_fast() {
    let (doc, divs, css) = fixture();
    let author = parse_stylesheet(&css);
    let ua = default_ua_stylesheet();
    let viewport = bowser_css::Viewport { w: 800.0, h: 600.0 };
    let author_refs = [&author];

    let start = std::time::Instant::now();
    let styler = Styler::new(&doc, &author_refs, &ua, viewport);
    for &d in &divs {
        let s = styler.style_for(d);
        // Spot-check correctness on the first and last element.
        if d == divs[0] || d == divs[divs.len() - 1] {
            assert_eq!(
                s.get("--ref-07"),
                Some(&bowser_css::CssValue::Length(
                    7.0,
                    bowser_css::LengthUnit::Px
                ))
            );
            assert_eq!(
                s.get("margin-top"),
                Some(&bowser_css::CssValue::Length(
                    0.0,
                    bowser_css::LengthUnit::Px
                ))
            );
        }
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "4000 var() resolutions took {elapsed:?}"
    );
}
