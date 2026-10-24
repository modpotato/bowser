//! Layout/paint text metrics agreement.
//!
//! `bowser_app::refresh` lays out with [`SwashFontMetrics`] over the
//! window's [`FontCache`] — the same shaped advances (kerning, ligatures)
//! paint's backends rasterize. These tests pin the contract at the
//! granularity paint consumes it: one display-list `TextRun` per layout
//! text fragment, so every fragment's laid-out width must equal the
//! metrics' advance for that fragment's exact slice, and no laid-out row
//! may overflow the measure width it was broken at.

use bowser_dom::Document;
use bowser_layout::{FontFamilyId, FontMetrics, Size, layout};
use bowser_paint::fonts::{FontCache, SwashFontMetrics};

/// Lay out a wrapping paragraph with [`SwashFontMetrics`] and return the
/// `(fragment text, fragment width, row y)` triples in tree order.
fn laid_out_fragments() -> (FontCache, Vec<(String, f32, u32)>) {
    let cache = FontCache::new();
    let metrics = SwashFontMetrics::new(&cache);

    let words = "AV To fi fl the quick brown fox jumps over the lazy dog ".repeat(6);
    let html = format!("<html><body><p style=\"font-size: 20px\">{words}</p></body></html>");
    let (doc, _) = Document::parse_html(html.as_bytes());
    // Narrow enough to force several wrapped rows.
    let root = layout(&doc, &[], Size { w: 220.0, h: 800.0 }, &metrics);

    let mut out = Vec::new();
    let mut stack = vec![&root];
    while let Some(b) = stack.pop() {
        stack.extend(&b.children);
        let Some(text) = b.text.as_deref() else {
            continue;
        };
        for frag in &b.fragments {
            let slice = text.get(frag.text_range.clone()).unwrap_or("");
            if slice.is_empty() {
                continue;
            }
            out.push((slice.to_owned(), frag.rect.w, frag.rect.y.to_bits()));
        }
    }
    (cache, out)
}

#[test]
fn fragment_widths_match_measured_advances() {
    let (cache, frags) = laid_out_fragments();
    let metrics = SwashFontMetrics::new(&cache);
    assert!(
        frags.len() > 4,
        "paragraph must wrap into several fragments, got {}",
        frags.len()
    );
    for (slice, w, _) in &frags {
        let want = metrics.advance(slice, 20.0, FontFamilyId::DEFAULT);
        assert!(
            (w - want).abs() < 0.01,
            "fragment {slice:?}: laid-out width {w} != measured advance {want}"
        );
    }
}

#[test]
fn wrapped_rows_do_not_overflow_measure_width() {
    let (_, frags) = laid_out_fragments();
    let mut rows = std::collections::HashMap::new();
    for (_, w, y) in &frags {
        // Fragments are measured per word/space; a row's painted extent
        // is the sum of its fragments' advances.
        *rows.entry(*y).or_insert(0.0f32) += w;
    }
    assert!(rows.len() > 1, "wrapping must produce multiple rows");
    for (y, total) in &rows {
        assert!(
            *total <= 220.0 + 0.5,
            "row at y-bits {y}: painted width {total} overflows the 220px measure width"
        );
    }
}

#[test]
fn fragment_ranges_are_ordered_and_disjoint() {
    // Within each text box, fragment byte ranges must be ordered and
    // non-overlapping, so paint's one-TextRun-per-fragment slicing can
    // never double-paint or skip text.
    use std::ops::Range;

    let cache = FontCache::new();
    let metrics = SwashFontMetrics::new(&cache);
    let words = "AV To fi fl the quick brown fox jumps over the lazy dog ".repeat(6);
    let html = format!("<html><body><p style=\"font-size: 20px\">{words}</p></body></html>");
    let (doc, _) = Document::parse_html(html.as_bytes());
    let root = layout(&doc, &[], Size { w: 220.0, h: 800.0 }, &metrics);

    let mut stack = vec![&root];
    let mut boxes_seen = 0;
    while let Some(b) = stack.pop() {
        stack.extend(&b.children);
        let Some(text) = b.text.as_deref() else {
            continue;
        };
        let mut prev: Option<Range<usize>> = None;
        let mut n = 0;
        for frag in &b.fragments {
            let range = frag.text_range.clone();
            assert!(
                text.get(range.clone()).is_some(),
                "range {range:?} out of bounds"
            );
            if let Some(p) = &prev {
                assert!(
                    p.end <= range.start,
                    "overlapping/out-of-order ranges {p:?} then {range:?}"
                );
            }
            prev = Some(range);
            n += 1;
        }
        if n > 0 {
            boxes_seen += 1;
        }
    }
    assert!(boxes_seen > 0, "expected at least one text box");
}
