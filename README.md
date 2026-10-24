# Bowser

A from-scratch web browser engine in Rust. Standard-library first, minimal
dependencies, WebGPU rendering, extreme performance.

The goal: a smaller footprint and faster than every modern browser, loading
90% of real websites, aiming at the same test suites Chromium and Firefox
pass (web-platform-tests, test262).

## Layout

| Crate | Subsystem |
|---|---|
| `crates/bowser_url` | WHATWG URL parser |
| `crates/bowser_encoding` | WHATWG Encoding Standard: label table, BOM sniffing, legacy decoders, meta prescan |
| `crates/bowser_net` | TCP/TLS/HTTP networking, fetch, cookies |
| `crates/bowser_html` | HTML tokenizer + tree construction |
| `crates/bowser_dom` | DOM core, events |
| `crates/bowser_css` | CSS parsing, cascade, values |
| `crates/bowser_layout` | Layout: block, inline, flexbox |
| `crates/bowser_paint` | Display lists, WebGPU renderer |
| `crates/bowser_js` | JavaScript runtime (browser-specific, from scratch) |
| `crates/bowser_wpt` | web-platform-tests harness (scoreboard; pluggable driver; real `BowserDriver` in `bowser_app`) |
| `crates/bowser_app` | Browser binary and windowed UI |

## Building

```sh
cargo build --release
cargo test --workspace
```

See `AGENTS.md` for the engineering rules every contributor (human or
agent) follows.
