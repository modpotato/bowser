//! `bowser` binary: command-line entry point for the Bowser browser.
//!
//! ```
//! bowser [URL] [--headless] [--viewport WxH] [--dump-display-list] [--timeout SECS]
//! ```
//!
//! * Without flags, opens `URL` (default: a built-in welcome page) in a
//!   window. The windowed event loop is owned by the `window` module.
//! * `--headless` fetches `URL` synchronously, renders it offscreen, and
//!   writes raw RGBA8 pixels (row-major, `4 * w * h` bytes, no header) to
//!   stdout. Progress and errors go to stderr.
//! * `--dump-display-list` loads `URL` through the real navigation
//!   pipeline, builds its display list, and prints the list's debug
//!   representation to stdout.

use std::io::Write;
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

use bowser_app::{Browser, BrowserConfig, Tab};
use bowser_css::ComputedStyle;
use bowser_dom::Document;
use bowser_layout::{LayoutBox, Size};
use bowser_paint::WgpuBackend;
use bowser_paint::display_list::build_display_list;
use bowser_paint::fonts::{FontCache, FontKey};

/// Parsed command line.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    url: Option<String>,
    headless: bool,
    check: bool,
    session: bool,
    viewport: (u32, u32),
    dump_display_list: bool,
    timeout_secs: u64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            url: None,
            headless: false,
            check: false,
            session: false,
            viewport: (1280, 800),
            dump_display_list: false,
            timeout_secs: 30,
        }
    }
}

/// Maximum viewport dimension in CSS px. A 16384x16384 RGBA8 frame is already
/// 1 GiB; anything larger is rejected instead of attempting a multi-gigabyte
/// allocation (u32::MAX per dimension would try to allocate ~40 GiB).
const MAX_VIEWPORT_DIM: u32 = 16_384;

/// Parse `WxH` into `(width, height)`; both must be positive and at most
/// `MAX_VIEWPORT_DIM`.
fn parse_viewport(s: &str) -> Result<(u32, u32), String> {
    let (w, h) = s
        .split_once('x')
        .or_else(|| s.split_once('X'))
        .ok_or_else(|| format!("bad viewport {s:?}: expected WxH, e.g. 1280x800"))?;
    let bad = || format!("bad viewport {s:?}: expected WxH with positive integers");
    let w: u32 = w.parse().map_err(|_| bad())?;
    let h: u32 = h.parse().map_err(|_| bad())?;
    if w == 0 || h == 0 {
        return Err(bad());
    }
    if w > MAX_VIEWPORT_DIM || h > MAX_VIEWPORT_DIM {
        return Err(format!(
            "bad viewport {s:?}: dimensions must not exceed {MAX_VIEWPORT_DIM}"
        ));
    }
    Ok((w, h))
}

/// Take a flag's value, either `--flag=value` or `--flag value`.
fn take_value<'a>(
    name: &str,
    inline: Option<&'a str>,
    raw: &'a [String],
    i: &mut usize,
) -> Result<&'a str, String> {
    if let Some(v) = inline {
        return Ok(v);
    }
    *i += 1;
    raw.get(*i)
        .filter(|v| !v.starts_with('-'))
        .map(String::as_str)
        .ok_or_else(|| format!("--{name} needs a value"))
}

/// Hand-rolled argument parser (std only, no dependency).
fn parse_args(raw: &[String]) -> Result<Args, String> {
    let mut args = Args::default();
    let mut i = 0;
    while i < raw.len() {
        let a = &raw[i];
        if let Some(flag) = a.strip_prefix("--") {
            let (name, inline) = match flag.split_once('=') {
                Some((n, v)) => (n, Some(v)),
                None => (flag, None),
            };
            match name {
                "headless" => {
                    if inline.is_some() {
                        return Err("--headless takes no value".to_string());
                    }
                    args.headless = true;
                }
                "check" => {
                    if inline.is_some() {
                        return Err("--check takes no value".to_string());
                    }
                    args.check = true;
                }
                "session" => {
                    if inline.is_some() {
                        return Err("--session takes no value".to_string());
                    }
                    args.session = true;
                }
                "dump-display-list" => {
                    if inline.is_some() {
                        return Err("--dump-display-list takes no value".to_string());
                    }
                    args.dump_display_list = true;
                }
                "viewport" => {
                    args.viewport = parse_viewport(take_value(name, inline, raw, &mut i)?)?;
                }
                "timeout" => {
                    let v: u64 = take_value(name, inline, raw, &mut i)?
                        .parse()
                        .map_err(|_| {
                            "--timeout needs a positive integer number of seconds".to_string()
                        })?;
                    if v == 0 {
                        return Err(
                            "--timeout needs a positive integer number of seconds".to_string()
                        );
                    }
                    args.timeout_secs = v;
                }
                "help" => { /* handled in main before parsing */ }
                other => return Err(format!("unknown flag: --{other}")),
            }
        } else if a.starts_with('-') && a.len() > 1 {
            return Err(format!("unknown flag: {a}"));
        } else if args.url.is_some() {
            return Err("only one URL may be given".to_string());
        } else {
            args.url = Some(a.clone());
        }
        i += 1;
    }
    // Output modes are mutually exclusive: running two would silently
    // discard one (e.g. `--headless --check` prints the check verdict
    // while the user waits for pixels on stdout), so reject the
    // combination instead of silently picking a precedence.
    let modes = [
        ("--headless", args.headless),
        ("--check", args.check),
        ("--session", args.session),
        ("--dump-display-list", args.dump_display_list),
    ];
    let set: Vec<&str> = modes
        .iter()
        .filter_map(|(name, on)| on.then_some(*name))
        .collect();
    if set.len() > 1 {
        return Err(format!(
            "{} and {} are mutually exclusive; pick one output mode",
            set[0], set[1]
        ));
    }
    Ok(args)
}

fn print_usage() {
    // Usage always goes to stderr: the `--headless` contract is stdout =
    // raw RGBA only, and argument errors report on stderr too, so every
    // `print_usage` call site is an error path except `--help` itself.
    eprintln!(
        "usage: bowser [URL] [options]\n\
         \n\
         options:\n\
         \x20 --headless             render offscreen, write raw pixels to stdout\n\
         \x20 --check                load the page through the real pipeline and\n\
         \x20                         run its scripts, without rendering;\n\
         \x20                         exit 0 = clean load, no uncaught script errors,\n\
         \x20                         1 = load failure or uncaught script error,\n\
         \x20                         >=2 = bowser-internal error (WPT driver)\n\
         \x20 --viewport WxH         viewport in css pixels (default 1280x800)\n\
         \x20 --dump-display-list    print the display list debug and exit\n\
         \x20 --timeout SECS         overall load deadline, headless mode (default 30)\n\
         \x20 -h, --help             print this help\n\
         \n\
         --headless, --check and --dump-display-list are output modes:\n\
         at most one may be given.\n\
         \n\
         URL defaults to about:blank, which renders a built-in welcome page.\n\
         \n\
         headless output: raw RGBA8 pixels, row-major, 4*w*h bytes on stdout\n\
         (no header). informational messages go to stderr."
    );
}

/// `--check`: load the page through the real headless pipeline and run
/// it, without rendering, for the WPT driver (`bowser --check {url}`;
/// see `wpt_driver.rs` for the protocol).
///
/// Exit 0: the page loaded and every script ran with no uncaught errors.
/// Exit 1: page-level failure — fetch/load failure, the `--timeout`
/// deadline firing, or an uncaught script error. Exit 2:
/// bowser-internal error (the check worker died without a verdict; bad
/// arguments are rejected by the parser with exit 2 before this runs).
fn run_check(args: &Args) -> ExitCode {
    let url = args
        .url
        .clone()
        .unwrap_or_else(|| "about:blank".to_string());
    let (w, h) = args.viewport;
    let viewport = Size {
        w: w as f32,
        h: h as f32,
    };
    // The check runs on a worker thread so `--timeout` still bounds it;
    // `Tab` is not `Send`, so only the verdict crosses back over the
    // channel (the same shape as `run_headless`). A panic in the
    // pipeline kills the worker, which the driver reads as a crash
    // (BrowserGone) rather than a page-level failure.
    let (tx, rx) = std::sync::mpsc::channel();
    let check_url = url.clone();
    // 8 MiB stack: the page realm raises its native-stack guard to 7 MiB
    // (see `install_loaded_document`), so the installing thread must
    // really have that much stack. This mirrors the test262 harness
    // recipe — 8 MiB thread, 7 MiB guard, 1 MiB of headroom to unwind —
    // and is what lets testharness.js-sized script runs complete instead
    // of tripping `RangeError: stack overflow` a few dozen nested calls
    // into the parse.
    std::thread::Builder::new()
        .stack_size(8 << 20)
        .spawn(move || {
            let _ = tx.send(check_page(&check_url, viewport));
        })
        .expect("check worker thread spawns");
    match rx.recv_timeout(Duration::from_secs(args.timeout_secs)) {
        Ok(Ok(())) => {
            eprintln!("bowser: check: {url} loaded and ran cleanly");
            ExitCode::SUCCESS
        }
        Ok(Err(e)) => {
            eprintln!("bowser: check: {e}");
            ExitCode::from(1)
        }
        Err(RecvTimeoutError::Timeout) => {
            eprintln!("bowser: timed out after {}s", args.timeout_secs);
            ExitCode::from(1)
        }
        // The worker died without sending (a panic inside the pipeline).
        // Report it honestly rather than as a timeout.
        Err(RecvTimeoutError::Disconnected) => {
            eprintln!("bowser: check worker died without responding");
            ExitCode::from(2)
        }
    }
}

/// The `--check` verdict for one URL: run it through the real headless
/// pipeline ([`bowser_app::load_tab_headless`] — document, subresources,
/// cookies, and `<script>` execution, exactly like `--headless` minus the
/// rasterization) and report `Ok` only when the load succeeded and no
/// script threw an uncaught error.
fn check_page(url: &str, viewport: Size) -> Result<(), String> {
    let tab =
        bowser_app::load_tab_headless(url, viewport).map_err(|e| format!("load failed: {e}"))?;
    let errors = tab
        .script_report()
        .map(|report| report.errors.len())
        .unwrap_or(0);
    if errors == 0 {
        return Ok(());
    }
    let first = tab
        .script_report()
        .and_then(|report| report.errors.first())
        .map(|error| error.to_string())
        .unwrap_or_default();
    Err(format!(
        "{} uncaught script error{}: {first}",
        errors,
        if errors == 1 { "" } else { "s" }
    ))
}

/// `--headless`: load `URL` through the full navigation pipeline
/// ([`bowser_app::load_tab_headless`] — document, linked stylesheets,
/// images, scripts, and cookies, exactly like `--dump-display-list`),
/// rasterize it offscreen, and write raw RGBA8 pixels (row-major,
/// `4 * w * h` bytes, no header) to stdout. Progress and errors go to
/// stderr.
///
/// The load and rasterization run on a worker thread so `--timeout`
/// still bounds them; `Tab` is not `Send`, so only the pixels cross back
/// over the channel (see [`render_headless_pixels`]).
///
/// Load `url` through the full headless pipeline and rasterize it to raw
/// RGBA8 pixels (row-major, `4 * w * h` bytes).
///
/// `fonts` is the process's single font cache, built once by
/// [`run_headless`] and reused across renders: constructing a
/// [`FontCache`] walks the system font directories, so threading one
/// instance through every render (instead of one per render) keeps that
/// walk a one-time cost. It travels behind a [`Mutex`] because the
/// reftest handshake ([`bowser_app::settle_reftest_wait`]) re-lays out
/// through the windowed pipeline's shared font-cache convention; the
/// worker thread is the only locker here.
///
/// `settle_deadline` bounds the `reftest-wait`/`TestRendered` handshake;
/// the caller derives it from `--timeout`.
fn render_headless_pixels(
    url: &str,
    viewport: Size,
    fonts: &Mutex<FontCache>,
    settle_deadline: Instant,
) -> Result<Vec<u8>, String> {
    let mut tab =
        bowser_app::load_tab_headless(url, viewport).map_err(|e| format!("load failed: {e}"))?;
    // Reftest handshake: a page with `class="reftest-wait"` on `<html>`
    // must see `TestRendered` before the screenshot, or the pixels race
    // the page's own mutations (dynamic-004). `--check` and
    // `--dump-display-list` don't come through here, so the `TestRendered`
    // dispatch is untouched for them — but note the shared headless pump
    // (`load_tab_headless` above) does drive one synthetic rAF frame per
    // iteration for every mode, so `requestAnimationFrame` callbacks now
    // fire under `--check`/`--dump-display-list` too (previously they
    // never ran headless).
    bowser_app::settle_reftest_wait(&mut tab, fonts, settle_deadline);
    // Mirror the windowed paint path (`refresh` in window.rs): the
    // display list builds from the paint tree — the layout tree with
    // nested scroll offsets baked in — falling back to the layout tree
    // before the first bake. Building from the layout tree unconditionally
    // drops nested scroll offsets from `--headless`/`--screenshot` output.
    let (Some(document), Some(root)) = (tab.document_ref(), headless_render_root(&tab)) else {
        return Err("load produced no document to render".to_string());
    };
    // Rasterize the pipeline-loaded page, resolving `<img>`/CSS images
    // through the loader's decoded-image cache (same as the windowed
    // paint path): real pixels, not placeholders.
    let document_url = tab
        .current_url()
        .cloned()
        .ok_or("load produced a tab with no URL")?;
    // Same `src` -> same `ImageId` mapping the windowed paint path uses.
    let mut image_ids = bowser_app::ImageIdMap::new(&document, Some(&document_url));
    let mut resolve_images = |src: &str| -> Option<(
        bowser_paint::display_list::ImageId,
        bowser_paint::display_list::DecodedImage,
    )> { image_ids.resolve(src) };
    // The worker thread is the only locker; a poisoned mutex means this
    // thread panicked earlier holding it, which cannot happen — recover
    // like the other lock sites instead of failing the render.
    let mut fonts = fonts.lock().unwrap_or_else(|e| e.into_inner());
    // The viewport scroll offset rides into the rasterizer like the
    // windowed path's per-frame uniform: a page the headless pump
    // scrolled (e.g. via `scrollIntoView`) screenshots at its live
    // scroll position, not the top of the document.
    let scroll = tab.scroll().offset();
    Ok(render_layout_to_rgba_with_fonts(
        root,
        &document,
        viewport,
        &mut resolve_images,
        &mut fonts,
        scroll,
    ))
}

/// The layout tree a headless render builds its display list from:
/// the paint tree (nested scroll offsets baked in) once the pipeline
/// has baked one, else the layout tree.
///
/// Mirrors the windowed `refresh` root selection
/// (`tab.paint_root.as_ref().or(tab.layout_root.as_ref())` in
/// window.rs); kept in sync with it.
fn headless_render_root(tab: &Tab) -> Option<&LayoutBox> {
    tab.paint_root().or(tab.layout_root.as_deref())
}

/// Round `viewport` to whole px; `None` for a zero-area viewport (the
/// render entry points return an empty vector then).
///
/// Mirrors `bowser_paint::render`'s private helper of the same name; kept
/// in sync with it.
fn viewport_whole(viewport: Size) -> Option<(u32, u32)> {
    let w = viewport.w.max(0.0).round() as u32;
    let h = viewport.h.max(0.0).round() as u32;
    (w > 0 && h > 0).then_some((w, h))
}

/// Composite straight-alpha RGBA8 pixels over opaque white, in place.
///
/// Both paint backends clear to transparent black; browsers start the
/// canvas opaque white. Mirrors `bowser_paint::render`'s private helper
/// of the same name; kept in sync with it.
fn composite_over_white(pixels: &mut [u8]) {
    for px in pixels.as_chunks_mut::<4>().0 {
        let (r, g, b, a) = (
            u32::from(px[0]),
            u32::from(px[1]),
            u32::from(px[2]),
            u32::from(px[3]),
        );
        let inv = 255 - a;
        px[0] = ((r * a + 255 * inv + 127) / 255) as u8;
        px[1] = ((g * a + 255 * inv + 127) / 255) as u8;
        px[2] = ((b * a + 255 * inv + 127) / 255) as u8;
        px[3] = 255;
    }
}

/// Mirror of [`bowser_paint::render::render_layout_to_rgba_with_scroll`]
/// that renders through a caller-owned [`FontCache`] instead of
/// constructing one per call.
///
/// `scroll` is the viewport scroll offset (CSS px), subtracted from
/// content at draw time like the windowed path's per-frame uniform.
/// Layout is already done by the caller; this builds the display list
/// (resolving each text run's `(family, weight, style)` through `fonts`),
/// renders offscreen via WebGPU when an adapter is available, and
/// composites over opaque white. Returns RGBA8 row-major pixels at
/// `viewport` rounded to whole pixels; an empty vector when the viewport
/// has zero area.
///
/// The adapter probe comes first so the CPU fallback below never pays
/// for a display list it would throw away: on adapter-less machines the
/// original entry point runs instead (identical pixels to before this
/// change). That fallback owns the with-font-data CPU rasterizer
/// (`CpuBackend` is crate-private to `bowser_paint`), so it still builds
/// a per-call cache there; the follow-up is a cache-taking render
/// variant inside `bowser_paint::render`.
fn render_layout_to_rgba_with_fonts(
    root: &LayoutBox,
    doc: &Document,
    viewport: Size,
    image_resolver: &mut dyn FnMut(
        &str,
    ) -> Option<(
        bowser_paint::display_list::ImageId,
        bowser_paint::display_list::DecodedImage,
    )>,
    fonts: &mut FontCache,
    scroll: (f32, f32),
) -> Vec<u8> {
    let Some((w, h)) = viewport_whole(viewport) else {
        return Vec::new();
    };
    if WgpuBackend::new_headless().is_some() {
        let display_list = {
            let mut resolve_fonts =
                |style: &ComputedStyle| -> FontKey { bowser_app::font_key_for_style(fonts, style) };
            build_display_list(
                root,
                doc,
                &mut resolve_fonts,
                &mut *image_resolver,
                viewport,
            )
        };
        if let Some(mut pixels) =
            WgpuBackend::render_offscreen_with_scroll(&display_list, w, h, fonts, scroll)
        {
            composite_over_white(&mut pixels);
            return pixels;
        }
        // Adapter vanished between probe and render; fall through to the
        // CPU path below.
    }
    bowser_paint::render::render_layout_to_rgba_with_scroll(
        root,
        doc,
        viewport,
        image_resolver,
        scroll,
    )
}

fn run_headless(args: &Args) -> ExitCode {
    let url = args
        .url
        .clone()
        .unwrap_or_else(|| "about:blank".to_string());
    let (w, h) = args.viewport;
    let viewport = Size {
        w: w as f32,
        h: h as f32,
    };

    let (tx, rx) = std::sync::mpsc::channel();
    // One font cache for every render this process performs: constructing
    // it walks the system font directories once, so it is built here and
    // threaded through the worker rather than per render.
    let fonts = Mutex::new(FontCache::new());
    // The reftest-wait handshake gets its own budget inside the overall
    // `--timeout`: the worker thread (and its `recv_timeout` below) stays
    // the hard bound for load + settle + rasterize together.
    let settle_deadline = Instant::now() + bowser_app::REFTEST_SETTLE_BUDGET;
    // 8 MiB stack, like the `--check` worker above: the install path
    // (`install_loaded_document`) raises the page realm's native-stack
    // guard to 7 MiB, which is only sound on a thread that actually has
    // that much stack.
    std::thread::Builder::new()
        .stack_size(8 << 20)
        .spawn(move || {
            let _ = tx.send(render_headless_pixels(
                &url,
                viewport,
                &fonts,
                settle_deadline,
            ));
        })
        .expect("headless worker thread spawns");
    let pixels = match rx.recv_timeout(Duration::from_secs(args.timeout_secs)) {
        Ok(Ok(pixels)) => pixels,
        Ok(Err(e)) => {
            eprintln!("bowser: {e}");
            return ExitCode::from(1);
        }
        Err(RecvTimeoutError::Timeout) => {
            eprintln!("bowser: timed out after {}s", args.timeout_secs);
            return ExitCode::from(1);
        }
        // The worker died without sending (a panic inside the pipeline).
        // Report it honestly rather than as a timeout.
        Err(RecvTimeoutError::Disconnected) => {
            eprintln!("bowser: headless worker died without responding");
            return ExitCode::from(2);
        }
    };
    // The headless contract is exactly 4*w*h raw RGBA8 bytes on stdout. A
    // short or overlong buffer means the renderer is broken: fail loudly
    // (exit 2, bowser-internal error) instead of writing a corrupt frame.
    let expected = (w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(4));
    let Some(expected) = expected else {
        eprintln!("bowser: viewport {w}x{h} overflows the pixel buffer size");
        return ExitCode::from(2);
    };
    if pixels.len() != expected {
        eprintln!(
            "bowser: renderer returned {} bytes, expected {expected} (4 * {w} * {h})",
            pixels.len()
        );
        return ExitCode::from(2);
    };
    if let Err(e) = std::io::stdout().write_all(&pixels) {
        eprintln!("bowser: failed writing pixels: {e}");
        return ExitCode::from(1);
    }
    eprintln!(
        "bowser: wrote {w}x{h} RGBA8 row-major ({} bytes) to stdout",
        pixels.len()
    );
    ExitCode::SUCCESS
}

/// Build the display-list debug dump for `--dump-display-list`: load the
/// URL through the real navigation pipeline
/// ([`bowser_app::load_tab_headless`]), run style/layout/paint, and
/// return the built display list in pretty debug format.
///
/// `about:blank` (the default) renders the built-in welcome page without
/// touching the network, so this is fully deterministic offline.
fn dump_display_list_text(url: &str, viewport: (u32, u32)) -> Result<String, String> {
    let tab = bowser_app::load_tab_headless(
        url,
        Size {
            w: viewport.0 as f32,
            h: viewport.1 as f32,
        },
    )
    .map_err(|e| format!("load failed: {e}"))?;
    Ok(match tab.display_list.as_ref() {
        Some(list) => format!("{list:#?}\n"),
        None => "(no display list: nothing to paint)\n".to_string(),
    })
}

fn run_dump_display_list(args: &Args) -> ExitCode {
    let url = args
        .url
        .clone()
        .unwrap_or_else(|| "about:blank".to_string());
    match dump_display_list_text(&url, args.viewport) {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("bowser: --dump-display-list: {e}");
            ExitCode::from(1)
        }
    }
}

/// Windowed event-loop backend: `window::run_windowed` (winit 0.30 event
/// loop, wgpu presentation). Errors are reported on stderr; exit 2 marks
/// a bowser-internal failure.
fn run_windowed_backend(mut browser: Browser, tab: bowser_app::TabId) -> ExitCode {
    match bowser_app::run_windowed(&mut browser, tab) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bowser: windowed mode failed: {e}");
            ExitCode::from(2)
        }
    }
}

fn run_windowed_mode(args: &Args) -> ExitCode {
    let url = args
        .url
        .clone()
        .unwrap_or_else(|| "about:blank".to_string());
    let config = BrowserConfig {
        viewport: args.viewport,
        ..BrowserConfig::default()
    };
    let mut browser = Browser::new(config);
    let tab = browser.new_tab(&url);
    run_windowed_backend(browser, tab)
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return ExitCode::SUCCESS;
    }
    let args = match parse_args(&raw) {
        Ok(args) => args,
        Err(e) => {
            eprintln!("bowser: {e}");
            print_usage();
            return ExitCode::from(2);
        }
    };
    // `parse_args` rejects conflicting output modes; the dispatch order
    // below is a belt-and-braces fallback, not a precedence rule.
    if args.dump_display_list {
        return run_dump_display_list(&args);
    }
    if args.session {
        // WPT session child: multiplexed headless protocol over stdio.
        // The driver side lives behind the `wpt-driver` feature; without
        // it the binary rejects `--session` instead of failing the
        // handshake mid-run.
        #[cfg(feature = "wpt-driver")]
        {
            bowser_app::wpt_driver::run_session_child();
        }
        #[cfg(not(feature = "wpt-driver"))]
        {
            eprintln!("bowser: --session requires the wpt-driver feature");
            return ExitCode::from(2);
        }
    }
    if args.check {
        return run_check(&args);
    }
    if args.headless {
        return run_headless(&args);
    }
    run_windowed_mode(&args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn defaults_when_no_args() {
        assert_eq!(parse_args(&[]).unwrap(), Args::default());
    }

    #[test]
    fn positional_url() {
        let a = parse_args(&raw(&["https://example.com/"])).unwrap();
        assert_eq!(a.url.as_deref(), Some("https://example.com/"));
        assert!(!a.headless);
    }

    #[test]
    fn all_flags() {
        let a = parse_args(&raw(&[
            "--headless",
            "--viewport",
            "800x600",
            "--timeout",
            "5",
            "https://example.com/",
        ]))
        .unwrap();
        assert!(a.headless);
        assert!(!a.dump_display_list);
        assert_eq!(a.viewport, (800, 600));
        assert_eq!(a.timeout_secs, 5);
        assert_eq!(a.url.as_deref(), Some("https://example.com/"));
    }

    #[test]
    fn output_modes_are_mutually_exclusive() {
        for pair in [
            ["--headless", "--check"],
            ["--headless", "--dump-display-list"],
            ["--headless", "--session"],
            ["--check", "--dump-display-list"],
            ["--check", "--session"],
            ["--dump-display-list", "--session"],
        ] {
            let err = parse_args(&raw(&pair)).expect_err("conflicting modes must be rejected");
            assert!(
                err.contains("mutually exclusive"),
                "unexpected error message: {err}"
            );
        }
        // Three-way conflict is rejected too.
        assert!(parse_args(&raw(&["--headless", "--check", "--dump-display-list"])).is_err());
    }

    #[test]
    fn each_output_mode_alone_is_fine() {
        for flag in ["--headless", "--check", "--dump-display-list", "--session"] {
            parse_args(&raw(&[flag])).expect("a single output mode must parse");
        }
        parse_args(&[]).expect("no mode flag (windowed) must parse");
    }

    #[test]
    fn viewport_and_timeout_inline_form() {
        let a = parse_args(&raw(&["--viewport=1024x768", "--timeout=10"])).unwrap();
        assert_eq!(a.viewport, (1024, 768));
        assert_eq!(a.timeout_secs, 10);
    }

    #[test]
    fn bad_viewports_rejected() {
        for bad in [
            "800",
            "0x600",
            "800x0",
            "axb",
            "800x600x400",
            "",
            "x",
            "16385x800",
            "800x16385",
            "4294967295x4294967295",
        ] {
            let flag = format!("--viewport={bad}");
            assert!(parse_args(&raw(&[&flag])).is_err(), "{bad}");
        }
    }

    #[test]
    fn viewport_cap_boundary() {
        assert_eq!(
            parse_args(&raw(&["--viewport=16384x16384"]))
                .unwrap()
                .viewport,
            (16384, 16384)
        );
        assert!(parse_args(&raw(&["--viewport=16385x16384"])).is_err());
        assert!(parse_args(&raw(&["--viewport=16384x16385"])).is_err());
    }

    #[test]
    fn unknown_flags_rejected() {
        assert!(parse_args(&raw(&["--frobnicate"])).is_err());
        assert!(parse_args(&raw(&["-x"])).is_err());
    }

    #[test]
    fn two_positionals_rejected() {
        assert!(parse_args(&raw(&["a", "b"])).is_err());
    }

    #[test]
    fn timeout_must_be_a_positive_integer() {
        assert!(parse_args(&raw(&["--timeout=0"])).is_err());
        assert!(parse_args(&raw(&["--timeout=abc"])).is_err());
        assert!(parse_args(&raw(&["--timeout"])).is_err());
    }

    #[test]
    fn check_flag() {
        let a = parse_args(&raw(&["--check", "https://example.com/"])).unwrap();
        assert!(a.check);
        assert!(!a.headless);
        assert_eq!(a.url.as_deref(), Some("https://example.com/"));
        assert!(parse_args(&raw(&["--check=yes"])).is_err());
    }

    #[test]
    fn bool_flags_reject_values() {
        assert!(parse_args(&raw(&["--headless=true"])).is_err());
        assert!(parse_args(&raw(&["--dump-display-list=1"])).is_err());
    }

    #[test]
    fn missing_viewport_value_rejected() {
        assert!(parse_args(&raw(&["--viewport"])).is_err());
    }

    #[test]
    fn dump_display_list_renders_debug_for_blank_page() {
        // about:blank renders the built-in welcome page without network,
        // so this is deterministic offline.
        let text = dump_display_list_text("about:blank", (800, 600)).expect("dump");
        assert!(text.contains("DisplayList"), "pretty debug names the type");
        assert!(text.contains("items"), "items are listed");
        assert!(
            !text.contains("items: []"),
            "the welcome page paints something"
        );
    }

    #[test]
    fn dump_display_list_rejects_bad_url() {
        assert!(dump_display_list_text("://bad", (800, 600)).is_err());
    }

    /// Headless rendering must build its display list from the paint
    /// tree (the layout tree with nested scroll offsets baked in), not
    /// the raw layout tree: the selection mirrors the windowed
    /// `refresh` path. `about:blank` (the built-in welcome page) bakes
    /// a paint tree on load, so the helper must return exactly it —
    /// a revert to `layout_root` fails this test.
    #[test]
    fn headless_render_root_prefers_paint_root() {
        let viewport = Size { w: 800.0, h: 600.0 };
        let tab = bowser_app::load_tab_headless("about:blank", viewport).expect("blank tab loads");
        let paint = tab.paint_root().expect("headless load bakes a paint tree");
        let root = headless_render_root(&tab).expect("a root to render");
        assert!(
            std::ptr::eq(root, paint),
            "headless must render from paint_root, not layout_root"
        );
    }

    /// Before the first paint-tree bake there is no `paint_root`, so
    /// the headless root falls back to the layout tree (and to `None`
    /// when there is no layout yet).
    #[test]
    fn headless_render_root_falls_back_to_layout_root() {
        let viewport = Size { w: 800.0, h: 600.0 };
        let mut tab = Tab::new(viewport);
        assert!(tab.paint_root().is_none());
        assert!(headless_render_root(&tab).is_none());
        tab.layout_root = Some(std::rc::Rc::new(LayoutBox {
            rect: bowser_layout::Rect {
                x: 0.0,
                y: 0.0,
                w: 800.0,
                h: 600.0,
            },
            children: Vec::new(),
            box_type: bowser_layout::BoxType::Block,
            style: None,
            text: None,
            node: None,
            fragments: Vec::new(),
            z_index: None,
            collapsed_borders: None,
            phantom_w: false,
            phantom_h: false,
        }));
        let layout = tab.layout_root.as_deref().expect("layout root set");
        let root = headless_render_root(&tab).expect("falls back to layout root");
        assert!(std::ptr::eq(root, layout));
    }

    /// Canned HTTP server for the `check_page` tests: `/throw.html`
    /// throws an uncaught error from an inline script, `/clean.html` is
    /// a clean page, anything else 404s.
    fn spawn_check_server() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind check server");
        let port = listener.local_addr().expect("port").port();
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let mut buf = [0u8; 4096];
                let _ = std::io::Read::read(&mut stream, &mut buf);
                let head = std::str::from_utf8(&buf).unwrap_or("");
                let target = head
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/");
                let (status, body) = if target.ends_with("/throw.html") {
                    (
                        "200 OK",
                        "<!DOCTYPE html><html><head><title>t</title>\
                         <script>throw new Error(\"fail\");</script></head>\
                         <body></body></html>",
                    )
                } else if target.ends_with("/clean.html") {
                    (
                        "200 OK",
                        "<!DOCTYPE html><html><head><title>c</title></head>\
                         <body>clean</body></html>",
                    )
                } else {
                    ("404 Not Found", "<title>nope</title>")
                };
                let head = format!(
                    "HTTP/1.1 {status}\r\n\
                     Content-Type: text/html; charset=utf-8\r\n\
                     Content-Length: {}\r\n\
                     Connection: close\r\n\r\n",
                    body.len()
                );
                let _ = std::io::Write::write_all(&mut stream, head.as_bytes());
                let _ = std::io::Write::write_all(&mut stream, body.as_bytes());
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    #[test]
    fn check_page_runs_scripts_and_reports_uncaught_errors() {
        let base = spawn_check_server();
        let viewport = Size { w: 800.0, h: 600.0 };

        // The reviewer's regression: a throwing script must fail the check.
        let err = check_page(&format!("{base}/throw.html"), viewport)
            .expect_err("throwing script must fail --check");
        assert!(
            err.contains("uncaught script error"),
            "unexpected verdict: {err}"
        );

        check_page(&format!("{base}/clean.html"), viewport).expect("clean page checks clean");
        check_page("about:blank", viewport).expect("about:blank checks clean");

        assert!(
            check_page(&format!("{base}/missing.html"), viewport).is_err(),
            "a 404 must fail --check"
        );
    }
}
