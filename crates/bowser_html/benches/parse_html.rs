//! Text-path allocation benchmark: parses a text-heavy document and reports
//! wall time plus heap allocation counts. Run with:
//! `cargo bench -p bowser_html --bench parse_html`
//!
//! The benchmark exists to validate the run-level text batching in
//! `tree.rs` (`push_text_char`/`flush_text`): before batching, parsing
//! paid roughly one small allocation per text character.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Counting allocator wrapper: tracks allocation count and requested bytes.
struct CountingAlloc;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: delegating to the system allocator with the same layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout come from a matching `alloc` call.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn reset_counters() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
}

/// ~1 MiB of text-heavy HTML: long paragraph runs, a big script/style
/// block (raw-text states), preformatted text, and foster-parented table
/// text, so every text insertion path in the tree builder is exercised.
fn fixture() -> Vec<u8> {
    let mut html = String::from("<!DOCTYPE html><html><head><title>bench</title><style>");
    html.push_str(&"a { color: red; } /* pad */".repeat(20_000));
    html.push_str("</style><script>");
    html.push_str(&"var x = 'hello world'; // trailing comment\n".repeat(20_000));
    html.push_str("</script></head><body>");
    for _ in 0..2000 {
        html.push_str("<p>");
        html.push_str(&"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(8));
        html.push_str("</p>");
    }
    html.push_str("<table>stray text<tr><td>cell one</td><td>cell two</td></tr></table>");
    html.push_str("<pre>");
    html.push_str(&"  indented preformatted line\n".repeat(5000));
    html.push_str("</pre></body></html>");
    html.into_bytes()
}

fn bench(name: &str, html: &[u8], iters: u32) {
    // Warm up (caches, first-parse one-time costs) without counting.
    reset_counters();
    let _ = bowser_html::parse(html);
    reset_counters();

    let start = Instant::now();
    for _ in 0..iters {
        let _ = bowser_html::parse(html);
    }
    let elapsed = start.elapsed();
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
    let bytes = ALLOC_BYTES.load(Ordering::Relaxed);

    println!(
        "{name}: {iters} iters, {:.1} MiB input total\n  \
         {:>10.2} ms/iter  {:>12} allocs/iter  {:>10.2} allocs/KiB  {:>8.1} MiB allocated/iter",
        html.len() as f64 * iters as f64 / (1 << 20) as f64,
        elapsed.as_secs_f64() * 1000.0 / iters as f64,
        allocs / iters as usize,
        allocs as f64 / iters as f64 / (html.len() as f64 / 1024.0),
        bytes as f64 / iters as f64 / (1 << 20) as f64,
    );
}

fn main() {
    let html = fixture();
    println!(
        "fixture: {} bytes ({:.2} MiB)",
        html.len(),
        html.len() as f64 / (1 << 20) as f64
    );
    bench("parse_html", &html, 5);
}
