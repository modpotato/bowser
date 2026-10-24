//! Fuzz the HTML tokenizer + tree builder (`bowser_html`).
//!
//! Three entry points are exercised per input:
//!   1. `parse_with_errors` - full document parse (WHATWG 13.2.6).
//!   2. `parse_fragment_with_errors` with a `div` context - fragment parse
//!      (WHATWG 13.4), covering the adoption agency algorithm in fragment
//!      mode.
//!   3. `parse_fragment_with_errors` with a `table` context - fragment parse
//!      covering foster parenting.
//!
//! Expected: a tree, possibly with parse errors recorded.
//! Panic / OOM / hang = bug.

#![no_main]

use libfuzzer_sys::fuzz_target;

fn ctx(tag: &str) -> bowser_html::Element {
    bowser_html::Element {
        tag: tag.to_string(),
        ns: bowser_html::Namespace::Html,
        attrs: Vec::new(),
        children: Vec::new(),
        template_contents: None,
        loc: None,
    }
}

fuzz_target!(|data: &[u8]| {
    // 1. Full document parse.
    let _ = bowser_html::parse_with_errors(data);

    // 2-3. Fragment parses with representative contexts.
    let div = ctx("div");
    let _ = bowser_html::parse_fragment_with_errors(data, &div);
    let table = ctx("table");
    let _ = bowser_html::parse_fragment_with_errors(data, &table);
});
