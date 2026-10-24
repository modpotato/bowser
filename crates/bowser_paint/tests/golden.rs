//! Golden pixel tests for `bowser_paint::render`.
//!
//! Methodology: each test renders a tiny fixture at a fixed 64x64 viewport
//! and checks two things.
//!
//! 1. Probe points: exact RGBA at hand-picked pixels. These encode the
//!    CSS/geometry contract (box starts at the origin because the UA sheet
//!    sets no body margin; `border`/`background` shorthands expand per
//!    `bowser_css`) and are backend-independent.
//! 2. Golden hash: FNV-1a over the whole buffer, recorded below. If any
//!    pipeline stage changes its output the hash changes and the test
//!    fails; re-baseline deliberately by pasting the `actual` hash from the
//!    failure message into the constant after confirming the new pixels
//!    are correct.
//!
//! Determinism: every fixture renders through
//! [`render_html_to_rgba_cpu`][bowser_paint::render::render_html_to_rgba_cpu],
//! which pins the CPU backend, so the hashes are identical with and
//! without a GPU adapter.
//!
//! Text is gated: the CPU path has no font data, so text runs rasterize to
//! nothing there. `golden_text_line` therefore asserts the canvas contract
//! (opaque white, text present in the display list but unpainted) rather
//! than glyph coverage; glyph probes re-enter when font data lands in the
//! render path (see the `render` module docs).
//!
//! Canvas is opaque white; the renderer is expected to return exactly
//! `4 * 64 * 64` bytes for these fixtures.

use bowser_paint::render::render_html_to_rgba_cpu;

const W: u32 = 64;
const H: u32 = 64;

fn render_fixture(name: &str) -> Vec<u8> {
    let html = match name {
        "red_div" => include_str!("fixtures/red_div.html"),
        "bordered_box" => include_str!("fixtures/bordered_box.html"),
        "text_line" => include_str!("fixtures/text_line.html"),
        "nested_bg" => include_str!("fixtures/nested_bg.html"),
        "translated_div" => include_str!("fixtures/translated_div.html"),
        "scaled_div" => include_str!("fixtures/scaled_div.html"),
        "rotated_div" => include_str!("fixtures/rotated_div.html"),
        "shadow_box" => include_str!("fixtures/shadow_box.html"),
        _ => unreachable!("unknown fixture"),
    };
    let px = render_html_to_rgba_cpu(html, (W, H));
    assert_eq!(
        px.len(),
        (4 * W * H) as usize,
        "fixture {name}: buffer size"
    );
    px
}

fn pixel(buf: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]
}

/// FNV-1a 64-bit, the golden hash.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_0000_01b3);
    }
    h
}

fn assert_golden(buf: &[u8], expected: u64, name: &str) {
    let actual = fnv1a_64(buf);
    assert_eq!(
        actual, expected,
        "fixture {name}: golden hash changed. If the new rendering is correct, \
         re-baseline deliberately by setting the constant to 0x{actual:016x}."
    );
}

fn region_has_nonwhite(buf: &[u8], x0: u32, y0: u32, x1: u32, y1: u32) -> bool {
    for y in y0..y1 {
        for x in x0..x1 {
            if pixel(buf, x, y) != [255, 255, 255, 255] {
                return true;
            }
        }
    }
    false
}

const RED: [u8; 4] = [255, 0, 0, 255];
const BLACK: [u8; 4] = [0, 0, 0, 255];
const WHITE: [u8; 4] = [255, 255, 255, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const YELLOW: [u8; 4] = [255, 255, 0, 255];

// Baselines recorded 2026-09-26 on the deterministic CPU path
// (`render_html_to_rgba_cpu`): render verified sane (see probe points)
// and each `actual` hash from the first green run pasted below per
// `assert_golden`. Any pipeline change (layout, display-list, CPU paint,
// canvas composite, FixedFontMetrics) changes these; re-baseline
// deliberately per the instructions in `assert_golden`.
const RED_DIV_GOLDEN: u64 = 0x5d86_6447_807a_4325;
const BORDERED_BOX_GOLDEN: u64 = 0x132a_3248_54bf_9225;
const TEXT_LINE_GOLDEN: u64 = 0x7d05_b753_2d39_6325;
const NESTED_BG_GOLDEN: u64 = 0x98a8_24e3_51be_ece5;
// Baselines for the transform fixtures recorded 2026-10-11 on the
// deterministic CPU path (`render_html_to_rgba_cpu`); probe points
// verified sane before pasting each `actual` hash per `assert_golden`.
const TRANSLATED_DIV_GOLDEN: u64 = 0xa2d5_c013_2d14_0d45;
const SCALED_DIV_GOLDEN: u64 = 0x5432_186f_faa1_d9a5;
const ROTATED_DIV_GOLDEN: u64 = 0x8ace_c942_a3a3_3a11;

#[test]
fn golden_red_div() {
    // 100x100 red div at the origin covers the whole 64x64 viewport.
    let px = render_fixture("red_div");
    for (x, y) in [(0, 0), (63, 0), (0, 63), (63, 63), (32, 32)] {
        assert_eq!(pixel(&px, x, y), RED, "red_div probe ({x},{y})");
    }
    assert_golden(&px, RED_DIV_GOLDEN, "red_div");
}

#[test]
fn golden_bordered_box() {
    // 40x40 content + 4px solid black border => border box (0,0,48,48).
    let px = render_fixture("bordered_box");
    assert_eq!(pixel(&px, 2, 2), BLACK, "border top-left");
    assert_eq!(pixel(&px, 45, 2), BLACK, "border top-right");
    assert_eq!(pixel(&px, 2, 45), BLACK, "border bottom-left");
    assert_eq!(pixel(&px, 45, 45), BLACK, "border bottom-right");
    assert_eq!(pixel(&px, 24, 24), WHITE, "content area");
    assert_eq!(pixel(&px, 60, 60), WHITE, "outside box");
    assert_golden(&px, BORDERED_BOX_GOLDEN, "bordered_box");
}

#[test]
fn golden_text_line() {
    // <p> has 1em (=16px) top margin; the 16px "hi" line sits below it.
    //
    // GATED: the deterministic CPU path has no font data, so the text run
    // rasterizes to nothing (see module docs). Assert the canvas contract
    // instead: the whole viewport is opaque white.
    let px = render_fixture("text_line");
    for (x, y) in [(0, 0), (63, 0), (0, 63), (63, 63), (16, 24), (32, 32)] {
        assert_eq!(pixel(&px, x, y), WHITE, "text_line probe ({x},{y})");
    }
    assert!(
        !region_has_nonwhite(&px, 0, 0, 64, 64),
        "text_line: CPU path must not paint text without font data"
    );
    assert_golden(&px, TEXT_LINE_GOLDEN, "text_line");
}

#[test]
fn golden_nested_bg() {
    // 20x20 yellow div nested at the origin of a 60x60 blue div.
    let px = render_fixture("nested_bg");
    assert_eq!(pixel(&px, 10, 10), YELLOW, "inner div");
    assert_eq!(pixel(&px, 40, 40), BLUE, "outer div only");
    assert_eq!(pixel(&px, 63, 63), WHITE, "canvas");
    assert_golden(&px, NESTED_BG_GOLDEN, "nested_bg");
}

#[test]
fn golden_translated_div() {
    // 20x20 red div translated by (30px, 10px) => covers (30,10,20,20).
    let px = render_fixture("translated_div");
    assert_eq!(pixel(&px, 35, 15), RED, "translated div");
    assert_eq!(pixel(&px, 5, 5), WHITE, "origin vacated");
    assert_eq!(pixel(&px, 55, 15), WHITE, "right of div");
    assert_eq!(pixel(&px, 35, 35), WHITE, "below div");
    assert_golden(&px, TRANSLATED_DIV_GOLDEN, "translated_div");
}

#[test]
fn golden_scaled_div() {
    // 20x20 blue div scaled 2x about left top => covers (0,0,40,40).
    let px = render_fixture("scaled_div");
    assert_eq!(pixel(&px, 5, 5), BLUE, "scaled div top-left");
    assert_eq!(pixel(&px, 30, 30), BLUE, "scaled div bottom-right");
    assert_eq!(pixel(&px, 50, 50), WHITE, "outside div");
    assert_golden(&px, SCALED_DIV_GOLDEN, "scaled_div");
}

#[test]
fn golden_rotated_div() {
    // 40x20 red div at the origin rotated 90deg about its center
    // (20,10): maps to x in [10,30], y in [-10,30]; the visible part
    // is the (10,0,20,30) rect.
    let px = render_fixture("rotated_div");
    assert_eq!(pixel(&px, 20, 15), RED, "rotated div center");
    assert_eq!(pixel(&px, 5, 5), WHITE, "vacated corner");
    assert_eq!(pixel(&px, 35, 15), WHITE, "right of rotated div");
    assert_eq!(pixel(&px, 20, 40), WHITE, "below rotated div");
    assert_golden(&px, ROTATED_DIV_GOLDEN, "rotated_div");
}

// Baseline recorded 2026-10-11 on the deterministic CPU path
// (`render_html_to_rgba_cpu`); probe points verified sane before
// pasting the `actual` hash per `assert_golden`.
const SHADOW_BOX_GOLDEN: u64 = 0x117d_57b5_a5c5_24a5;

#[test]
fn golden_shadow_box() {
    // 40x40 red div at the origin with `box-shadow: 8px 8px 0 black`:
    // the sharp shadow rect covers (8,8,40,40), peeking out right and
    // below the div.
    let px = render_fixture("shadow_box");
    assert_eq!(pixel(&px, 4, 4), RED, "div background");
    assert_eq!(pixel(&px, 44, 20), BLACK, "shadow right of div");
    assert_eq!(pixel(&px, 20, 44), BLACK, "shadow below div");
    assert_eq!(pixel(&px, 44, 44), BLACK, "shadow bottom-right");
    assert_eq!(pixel(&px, 60, 60), WHITE, "outside shadow");
    assert_golden(&px, SHADOW_BOX_GOLDEN, "shadow_box");
}

/// A nonzero viewport scroll offset shifts the raster: scrolling down by
/// `dy` moves content up by `dy` — the scrolled frame's row `y` matches
/// the unscrolled frame's row `y + dy` over the overlap. Exercises
/// [`render_html_to_rgba_with_scroll`], the scroll-carrying entry the
/// headless screenshot paths (`main.rs`, `wpt_driver.rs`) render
/// through.
///
/// Backend note: unlike the hash fixtures above this goes through the
/// GPU-first entry (falling back to the CPU rasterizer when no adapter
/// is present), so no golden hash — the assertions are backend-agnostic
/// (relative shift plus solid-color probes), and both renders in the
/// test take the same backend.
#[test]
fn golden_viewport_scroll_shifts_raster() {
    use bowser_paint::render::render_html_to_rgba_with_scroll;

    // Red band on top, blue band below; the UA sheet sets no body
    // margin, belt-and-braces inline anyway.
    let html = "<html><body style=\"margin:0\">\
                <div style=\"width:64px;height:32px;background:red\"></div>\
                <div style=\"width:64px;height:32px;background:blue\"></div>\
                </body></html>";
    let plain = render_html_to_rgba_with_scroll(html, (W, H), (0.0, 0.0));
    let scrolled = render_html_to_rgba_with_scroll(html, (W, H), (0.0, 32.0));
    assert_eq!(plain.len(), (4 * W * H) as usize, "plain buffer size");
    assert_eq!(scrolled.len(), (4 * W * H) as usize, "scrolled buffer size");
    assert_ne!(plain, scrolled, "a scroll offset must change the raster");
    // Probes: unscrolled shows red on top and blue below; scrolled down
    // by 32 the red band is off-frame and the blue band reaches the top.
    assert_eq!(pixel(&plain, 32, 16), RED, "plain top band");
    assert_eq!(pixel(&plain, 32, 48), BLUE, "plain bottom band");
    assert_eq!(pixel(&scrolled, 32, 0), BLUE, "scrolled top row");
    assert_eq!(pixel(&scrolled, 32, 31), BLUE, "scrolled last blue row");
    assert_eq!(pixel(&scrolled, 32, 32), WHITE, "scrolled past content");
    // Shift property over the overlap: scrolled row y == plain row y+32.
    for y in 0..(H - 32) {
        for x in 0..W {
            assert_eq!(
                pixel(&scrolled, x, y),
                pixel(&plain, x, y + 32),
                "scroll(0,32): pixel ({x},{y})"
            );
        }
    }
}
