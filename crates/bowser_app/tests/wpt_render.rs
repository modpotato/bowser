//! End-to-end WPT driver check against the real `bowser` binary.
//!
//! Serves the fixture pages under `tests/fixtures/wpt` over the std-only
//! [`FixtureServer`](bowser_app::wpt_driver::FixtureServer) (no external
//! network), renders each through the real headless pipeline (navigate ->
//! layout -> `render_to_rgba` in the `bowser` binary), and asserts render
//! success plus pixel sanity: the frame is not a blank canvas and its
//! average color matches the fixture's expected color within a small
//! documented tolerance.
//!
//! Fixture groups: solid fills (`solid_*`, `split`), the box model
//! (`box_padding`, `box_margin`, `box_border`), text (`text_basic`,
//! `text_colored`), replaced images (`img_placeholder*`), form controls
//! (`form_button`, `form_input`), and tables (`table_cells`,
//! `table_bordered`).

use std::path::PathBuf;
use std::time::Duration;

use bowser_app::wpt_driver::{BowserDriver, average_color, is_blank_canvas};
use bowser_wpt::driver::{BrowserDriver, Screenshot, Viewport};

/// Viewport the fixtures are sized for (each fixture covers it exactly:
/// the UA sheet sets no body margin, so a 160x120 div starts at the
/// origin and fills the frame).
const VIEWPORT: Viewport = Viewport {
    width: 160,
    height: 120,
};

/// Per-channel slack when comparing the frame's average color against a
/// fixture's expected solid color. The pipeline works in f32 and rounds
/// when rasterizing, so off-by-a-few is normal; anything larger means the
/// wrong color was painted.
const COLOR_SLACK: u8 = 8;

/// Driver pointed at the real `bowser` binary under test, serving
/// `tests/fixtures/wpt`.
fn driver() -> BowserDriver {
    let exe = PathBuf::from(env!("CARGO_BIN_EXE_bowser"));
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wpt");
    assert!(
        fixtures.is_dir(),
        "fixture dir missing: {}",
        fixtures.display()
    );
    let mut d = BowserDriver::new(exe, &fixtures).expect("fixture server failed to start");
    // Cold GPU init (shader compile, adapter probing) can be slow; the
    // driver default is 60s, keep it generous here too.
    d.render_timeout = Duration::from_secs(120);
    d
}

/// Render `page` and assert the frame is sane: exact byte count, not a
/// blank canvas, average RGB within `COLOR_SLACK` of `expected`, opaque.
fn assert_renders_with_color(name: &str, page: &str, expected: [u8; 3]) {
    let d = driver();
    let url = format!("{}{page}", d.server_base);
    let shot = d
        .render(&url, VIEWPORT)
        .unwrap_or_else(|e| panic!("{name}: render of {url} failed: {e}"));
    assert_eq!(
        (shot.width, shot.height),
        (VIEWPORT.width, VIEWPORT.height),
        "{name}: dimensions"
    );
    assert_eq!(
        shot.rgba.len(),
        VIEWPORT.width as usize * VIEWPORT.height as usize * 4,
        "{name}: byte count"
    );
    assert!(
        !is_blank_canvas(&shot),
        "{name}: rendered a blank (all-white) canvas"
    );
    let avg = average_color(&shot);
    for (i, (got, want)) in avg[..3].iter().zip(expected.iter()).enumerate() {
        assert!(
            got.abs_diff(*want) <= COLOR_SLACK,
            "{name}: channel {i} average {got} not within {COLOR_SLACK} of {want} (avg={avg:?})"
        );
    }
    assert_eq!(avg[3], 255, "{name}: expected a fully opaque frame");
}

#[test]
fn renders_solid_red_fixture() {
    assert_renders_with_color("solid_red", "/solid_red.html", [255, 0, 0]);
}

#[test]
fn renders_solid_blue_fixture() {
    assert_renders_with_color("solid_blue", "/solid_blue.html", [0, 0, 255]);
}

#[test]
fn renders_split_fixture() {
    // Top half red, bottom half blue: the frame average is purple.
    assert_renders_with_color("split", "/split.html", [128, 0, 128]);
}

// --- Box model -----------------------------------------------------------
// Each fixture's expected color is hand-computed from the painted areas
// (frame is 160x120 = 19200 px); `average_color` rounds down, and the
// COLOR_SLACK of 8 covers rasterizer rounding.

#[test]
fn renders_box_padding() {
    // 160x120 red div, 20px padding, 120x80 blue child at the padding
    // origin: 9600 blue px, 9600 red px.
    assert_renders_with_color("box_padding", "/box_padding.html", [127, 0, 127]);
}

#[test]
fn renders_box_margin() {
    // 100x60 red div with `margin: 30px 20px` on a white body: the red
    // box sits at (20, 30), 6000 px; the other 13200 px are white.
    assert_renders_with_color("box_margin", "/box_margin.html", [255, 175, 175]);
}

#[test]
fn renders_box_border() {
    // 160x120 red div with a 20px solid blue border (content-box, so the
    // border box is 200x160 and the frame clips it): visible blue is the
    // top 160x20 strip plus the left 20x100 strip (5200 px), visible red
    // is 140x100 (14000 px).
    assert_renders_with_color("box_border", "/box_border.html", [185, 0, 69]);
}

// --- Text ----------------------------------------------------------------

/// Render `page` and return the frame; the test fails on render error.
fn render_page(name: &str, page: &str) -> Screenshot {
    let d = driver();
    let url = format!("{}{page}", d.server_base);
    let shot = d
        .render(&url, VIEWPORT)
        .unwrap_or_else(|e| panic!("{name}: render of {url} failed: {e}"));
    assert_eq!(
        (shot.width, shot.height),
        (VIEWPORT.width, VIEWPORT.height),
        "{name}: dimensions"
    );
    assert!(
        !is_blank_canvas(&shot),
        "{name}: rendered a blank (all-white) canvas"
    );
    shot
}

#[test]
fn renders_text_basic() {
    // 100px black "X" on a white 160x120 div: the glyph coverage must
    // pull every channel visibly below white. (Exact glyph pixels differ
    // between the GPU and CPU rasterizers, so this asserts darkening
    // rather than an exact average.)
    let shot = render_page("text_basic", "/text_basic.html");
    let avg = average_color(&shot);
    for (i, c) in avg[..3].iter().enumerate() {
        assert!(
            *c <= 250,
            "text_basic: channel {i} average {c} is too close to white; text did not paint (avg={avg:?})"
        );
    }
    assert_eq!(avg[3], 255, "text_basic: expected a fully opaque frame");
}

#[test]
fn renders_text_colored() {
    // 100px red "X" on white: every pixel keeps R=255 (red and white
    // both have full red, including antialiased edges), while the glyph
    // coverage pulls green and blue down.
    let shot = render_page("text_colored", "/text_colored.html");
    let avg = average_color(&shot);
    assert!(
        avg[0].abs_diff(255) <= 2,
        "text_colored: red channel average {} should stay ~255 (avg={avg:?})",
        avg[0]
    );
    for (i, c) in avg[1..3].iter().enumerate() {
        assert!(
            *c <= 250,
            "text_colored: channel {} average {c} is too close to white; text did not paint (avg={avg:?})",
            i + 1
        );
    }
    assert_eq!(avg[3], 255, "text_colored: expected a fully opaque frame");
}

// --- Images --------------------------------------------------------------
// `<img>` with an unresolvable src paints the gray (200,200,200)
// placeholder box (see `bowser_paint::display_list`); the headless
// pipeline has no image loading, so `missing.png` never resolves.

#[test]
fn renders_img_placeholder() {
    // 160x120 placeholder: the whole frame is placeholder gray.
    assert_renders_with_color("img_placeholder", "/img_placeholder.html", [200, 200, 200]);
}

#[test]
fn renders_img_placeholder_small() {
    // 80x60 placeholder at the origin (4800 px gray), white elsewhere:
    // (4800*200 + 14400*255) / 19200 = 241.25 -> 241.
    assert_renders_with_color(
        "img_placeholder_small",
        "/img_placeholder_small.html",
        [241, 241, 241],
    );
}

// --- Forms ---------------------------------------------------------------

#[test]
fn renders_form_button() {
    // An empty 160x120 green button fills the frame (CSS `green` is
    // (0,128,0)).
    assert_renders_with_color("form_button", "/form_button.html", [0, 128, 0]);
}

#[test]
fn renders_form_input() {
    // A 100x40 yellow input at the origin (4000 px), white elsewhere:
    // blue channel (15200*255)/19200 = 201.9 -> 201.
    assert_renders_with_color("form_input", "/form_input.html", [255, 255, 201]);
}

// --- Tables --------------------------------------------------------------
// The UA sheet maps table parts to block layout (no table layout yet),
// so rows/cells stack as plain blocks; these fixtures pin that the
// table markup renders its cell backgrounds in order.

#[test]
fn renders_table_cells() {
    // Two 160x60 cells, red over blue: 9600 px each.
    assert_renders_with_color("table_cells", "/table_cells.html", [127, 0, 127]);
}

#[test]
fn renders_table_bordered() {
    // 160-wide table, 10px solid black border, two 160x50 cells (red,
    // blue). Content-box: the border box is 180x120, so the content box
    // (x in [10,170)) is clipped by the 160-wide viewport to 150 px.
    // Visible: black = frame minus the 150x100 visible content box
    // (19200 - 15000 = 4200 px), red 150x50 (7500 px) at (10,10), blue
    // 150x50 (7500 px) at (10,60): 7500*255/19200 = 99.6 -> 99.
    assert_renders_with_color("table_bordered", "/table_bordered.html", [99, 0, 99]);
}
