//! One-call page-to-pixels API for headless mode, tests, and the WPT driver.
//!
//! # Sibling-subsystem contract
//!
//! This module codes against subsystems owned by sibling workers that may
//! not have landed yet. Exact API surface assumed here (per the render
//! worker's spec):
//!
//! - `crate::display_list`:
//!   - `pub struct ImageId` — atlas slot for a decoded image.
//!   - `pub struct DecodedImage` — decoded RGBA8 image bytes plus size.
//!   - `pub struct DisplayList` — the built display list.
//!   - `pub fn build_display_list(root: &bowser_layout::LayoutBox,
//!     doc: &bowser_dom::Document,
//!     font_resolver: &mut dyn FnMut(&ComputedStyle) -> FontKey,
//!     image_resolver: &mut dyn FnMut(&str) -> Option<(ImageId, DecodedImage)>)
//!     -> DisplayList`
//! - `crate::wgpu_be::WgpuBackend::render_offscreen(display_list:
//!   &crate::display_list::DisplayList, width: u32, height: u32,
//!   fonts: &crate::fonts::FontCache) -> Option<Vec<u8>>` — `None` when no
//!   GPU adapter is available (headless CI, no Vulkan), in which case the
//!   CPU fallback is used.
//! - `crate::cpu::CpuBackend::render_to_rgba_full` — the `&mut self`
//!   5-argument form (CSS-px scale, scroll); the static 3-argument
//!   `render_to_rgba` carries no font data.
//!
//! Observed divergences from the above as sibling work lands (2026-09-25/26):
//! - `crate::display_list` HAS landed with exactly the assumed API
//!   (`build_display_list`, `ImageId`, `DecodedImage`, `DisplayList`,
//!   plus a `Backend` trait). `DisplayItem::TextRun` is now verbatim
//!   text per layout fragment (not pre-shaped glyphs); see the
//!   display-list module docs.
//! - `crate::images`' `ImageId`/`DecodedImage` were deduplicated into
//!   `crate::display_list` (re-exported from `crate::images`).
//! - `crate::wgpu_be` HAS landed with the assumed `render_offscreen`
//!   associated function.
//! - `crate::cpu` HAS landed; its `render_to_rgba` is the static
//!   3-argument associated function the deterministic path below uses,
//!   and the `&mut self` 5-argument form is `render_to_rgba_full`. It
//!   needs `crate::fonts` (`FontData`, `FontKey` — `FontKey` is owned by
//!   `crate::display_list`). Both backends shape through
//!   [`crate::text::shape_text`] (real swash shaping: kerning, ligature
//!   substitution, cluster-aware runs); the interim per-char
//!   charmap+advance path survives only as the fallback inside
//!   `shape_text_with_direction` when a font cannot be shaped.
//! - `crate::fonts` HAS landed: this module builds display lists through
//!   [`build_display_list`](crate::display_list::build_display_list)
//!   with a `FontCache`-backed font resolver (family/weight/style read
//!   from each run's [`ComputedStyle`]) instead of the constant
//!   `FontKey(0)`.
//!
//! # Fonts
//!
//! Real text metrics have landed: [`FontCache`] implements
//! [`bowser_layout::FontMetrics`] with real swash glyph advances (and
//! real line metrics), so layout can measure text exactly as paint
//! rasterizes it. The convenience entry points below still measure with
//! `bowser_layout::FixedFontMetrics` (0.6em advance per char, 1.2em line
//! height) *deliberately*: the golden suite in `tests/golden.rs` needs
//! deterministic pixels, and text advances/line breaks (hence every
//! golden hash) change with real metrics. The app worker wires its own
//! per-window `FontCache` as the layout metrics (see "Window
//! integration" below).
//!
//! Paint's *internal* pipeline is already on real fonts: text runs carry
//! [`FontKey`]s resolved from their computed style through an explicit
//! [`FontCache`], and both backends rasterize real swash coverage from the
//! resolved [`FontData`]:
//!
//! - the GPU backend ([`crate::wgpu_be::WgpuBackend::render_offscreen`])
//!   takes the same `&FontCache` the display list was built with and
//!   resolves through it;
//! - the CPU fallback in [`render_layout_to_rgba`] builds a CPU backend
//!   with a `get_font_data` hook over a snapshot of the same cache, so
//!   the keys stay valid;
//! - the deterministic [`render_layout_to_rgba_cpu`] path keeps the
//!   static fontless CPU bridge, so the golden suite's pixels are
//!   unchanged (text runs rasterize to nothing there).
//!
//! There is no process-wide font cache: each entry point below constructs
//! its own [`FontCache`] and threads it explicitly through display-list
//! build and backend. [`FontKey`]s are cache-local, so every stage must
//! resolve against the *same* cache instance — the explicit parameter
//! makes that impossible to get wrong. The app worker owns its own
//! `FontCache` per window and wires it explicitly; see "Window
//! integration" below.
//!
//! # Window integration (for the app worker)
//!
//! `crates/bowser_app/src/window.rs` now resolves display-list fonts through
//! its own per-window `FontCache` (via `font_key_for_style`, which resolves
//! and falls back to `FontKey(0)` on fontless CI), used both as the
//! display-list font resolver *and* as the
//! [`bowser_layout::FontMetrics`] for layout (it implements the trait
//! with real glyph advances). The shape is:
//!
//! ```ignore
//! // Window state owns its own FontCache (one per window).
//! let display_list = build_display_list(
//!     &root, &doc,
//!     &mut |style: &ComputedStyle| {
//!         fonts.resolve(&family, weight, style).unwrap_or(FontKey(0))
//!     },
//!     &mut image_resolver,
//!     viewport,
//! );
//! let mut cpu = CpuBackend::new(Rc::new(move |key| fonts.font_data(key).cloned()));
//! gpu.set_font_data(Arc::new(move |key| fonts.font_data(key).cloned()));
//! ```
//!
//! All four pieces (`build_display_list`, `FontKey`, `CpuBackend::new`,
//! `WgpuBackend::set_font_data`) are public; the window's `FontCache`
//! comes from `bowser_paint::fonts::FontCache`. Note the `Rc`/`Arc`
//! split: the CPU hook is `Rc`, the GPU hook must be `Arc` (`Send +
//! Sync`) because a window's backend may live in shared state.
//! [`FontCache::resolver`](crate::fonts::FontCache::resolver) builds the
//! `Arc` hook directly.
//!
//! # Documented limitations of `render_html_to_rgba`
//!
//! - Test/dev convenience only: parses author CSS solely from inline
//!   `<style>` elements. `<link rel=stylesheet>` is ignored (no network in
//!   the test path); the UA stylesheet is applied inside
//!   [`bowser_layout::layout`].
//! - `box-shadow` and `text-shadow` ARE honored: parsed in
//!   `bowser_css`, emitted as display-list items by the builder, and
//!   painted by both backends (outset shadows knock out the border box
//!   per CSS Backgrounds 3 §7.2; blur is a gaussian σ = blur/2 on the
//!   CPU backend and a centered-smoothstep approximation on the GPU
//!   backend — see `SHADER_SHADOW`).
//!   `transform`/`transform-origin` ARE honored: the display-list
//!   builder wraps transformed subtrees in
//!   `PushTransform`/`PopTransform` groups (with `transform-origin`
//!   folded into the matrix) and both backends paint the subtree
//!   through the matrix. `mix-blend-mode` IS honored via
//!   `PushBlend`/`PopBlend` groups (the blend worker landed it).
//!   `text-transform` IS honored
//!   (applied during layout's whitespace collapsing) and `border-radius`
//!   IS honored (rounded border tessellation in both backends);
//!   `opacity` IS honored: the display-list builder
//!   wraps subtrees with `opacity < 1` in `PushOpacity`/`PopOpacity`
//!   groups and both backends multiply the group opacity into item alpha.
//! - No image loading: the resolver always returns `None`, so `<img>`
//!   paints its alt text / placeholder box per the display-list worker.
//! - Canvas starts opaque white: both backends clear to transparent black,
//!   so [`render_layout_to_rgba`] and [`render_layout_to_rgba_cpu`]
//!   composite the backend output over an opaque white canvas (matching
//!   browser reality) before returning. The viewport is clipped to
//!   `viewport_px` (no scrolling, no pagination).
//!
//! # Determinism
//!
//! [`render_html_to_rgba`] prefers the WebGPU backend and falls back to the
//! CPU rasterizer; both now render real glyph coverage (the GPU path and
//! the CPU fallback resolve through the entry point's `FontCache`), so
//! text pixels agree across backends up to rasterizer hinting differences.
//! Tests that need stable pixels — the golden suite in `tests/golden.rs` —
//! must use [`render_html_to_rgba_cpu`], which always renders on the
//! fontless CPU bridge: identical with and without a GPU adapter, and text
//! runs rasterize to nothing there.

use std::rc::Rc;

use bowser_css::{ComputedStyle, CssValue, Stylesheet, props};
use bowser_dom::Document;
use bowser_layout::{FontMetrics, LayoutBox, Size};

use crate::display_list::{DecodedImage, DisplayList, FontKey, ImageId};
use crate::fonts::{FontCache, FontData, FontStyle};

/// Composite straight-RGBA8 `pixels` over an opaque white canvas.
///
/// Both backends clear to transparent black; browsers start the canvas
/// opaque white, so `render_*` does the composite here. Fully transparent
/// pixels become white, fully opaque pixels are unchanged, and
/// semi-transparent pixels blend over white (straight-alpha source-over).
/// The buffer is fully opaque on return.
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

/// Round `viewport` to whole px; `None` for a zero-area viewport (the
/// entry points return an empty vector then).
fn viewport_whole(viewport: Size) -> Option<(u32, u32)> {
    let w = viewport.w.max(0.0).round() as u32;
    let h = viewport.h.max(0.0).round() as u32;
    (w > 0 && h > 0).then_some((w, h))
}

/// Parse `html`, concatenate every `<style>` element's text into one
/// author stylesheet, and lay out at `viewport_px`.
///
/// Shared by [`render_html_to_rgba`] and [`render_html_to_rgba_cpu`]
/// (see the module docs for rendering limitations).
fn parse_and_layout(html: &str, viewport_px: (u32, u32)) -> (Document, LayoutBox) {
    let (doc, _quirks) = Document::parse_html(html.as_bytes());

    let mut author_css = String::new();
    for style_el in doc.get_elements_by_tag_name("style") {
        author_css.push_str(&doc.text_content(style_el));
        author_css.push('\n');
    }
    let author: Stylesheet = bowser_css::parse_stylesheet(&author_css);

    // FixedFontMetrics deliberately: the golden suite needs
    // deterministic pixels (real metrics via FontCache change advances
    // and line breaks); see the module docs' "Fonts" section.
    let fonts: &dyn FontMetrics = &bowser_layout::FixedFontMetrics;
    let viewport = Size {
        w: viewport_px.0 as f32,
        h: viewport_px.1 as f32,
    };
    let root = bowser_layout::layout(&doc, &[&author], viewport, fonts);
    (doc, root)
}

/// Read the `(family, weight, style)` font request out of a computed
/// style, for `FontCache::resolve`.
///
/// Shared with `bowser_app`'s window font resolution: both crates call
/// these exact functions, so the extraction lives here once.
///
/// - family: the first `font-family` entry (`String`, `Keyword`, or the
///   head of a `List`); missing falls back to `"serif"`, matching the UA
///   stylesheet's default (see `bowser_css`). The returned `&str` is
///   borrowed from the style's [`CssValue`] (or the static fallback), so
///   this allocates nothing on the per-text-box hot path. Zero-copy by
///   design: call sites pass the result straight into
///   `FontCache::resolve`, whose hit path also allocates nothing.
/// - weight: numeric `font-weight` clamped to 1..=1000, `bold` -> 700,
///   anything else (`normal`, ...) -> 400.
/// - style: `italic`/`oblique` keywords, else normal.
///
pub fn css_font_family(style: &ComputedStyle) -> &str {
    match style.get_by_index(props::FONT_FAMILY) {
        Some(CssValue::List(items)) => items
            .iter()
            .find_map(|v| match v {
                CssValue::String(s) => Some(s.as_str()),
                CssValue::Keyword(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or("serif"),
        // `get_str` covers `String`/`Keyword` and yields `None` for
        // anything else, which falls back to `"serif"` below.
        _ => style
            .get_str_by_index(props::FONT_FAMILY)
            .unwrap_or("serif"),
    }
}

pub fn css_font_weight(style: &ComputedStyle) -> u16 {
    match style.get_by_index(props::FONT_WEIGHT) {
        Some(CssValue::Number(n)) => n.round().clamp(1.0, 1000.0) as u16,
        Some(CssValue::Keyword(s)) if s == "bold" => 700,
        _ => 400,
    }
}

pub fn css_font_style(style: &ComputedStyle) -> FontStyle {
    match style.get_str_by_index(props::FONT_STYLE) {
        Some("italic") => FontStyle::Italic,
        Some("oblique") => FontStyle::Oblique,
        _ => FontStyle::Normal,
    }
}

/// Build the display list with a real font resolver: every text run's
/// [`FontKey`] comes from `fonts` via `resolve` on the run's computed
/// `(family, weight, style)`, falling back to [`FontKey(0)`] when no face
/// resolves (unreachable while the embedded fallback face exists, kept
/// defensive).
fn build_display_list_with_fonts(
    root: &LayoutBox,
    doc: &Document,
    fonts: &mut FontCache,
    image_resolver: &mut dyn FnMut(&str) -> Option<(ImageId, DecodedImage)>,
    viewport: Size,
) -> DisplayList {
    crate::display_list::build_display_list(
        root,
        doc,
        &mut |style: &ComputedStyle| {
            fonts
                .resolve(
                    css_font_family(style),
                    css_font_weight(style),
                    css_font_style(style),
                )
                .unwrap_or(FontKey(0))
        },
        image_resolver,
        viewport,
    )
}

/// `get_font_data` hook over a [`FontCache`] snapshot, for the CPU backend
/// (whose hook is `Rc`, not `Send`).
fn font_data_hook(fonts: &FontCache) -> Rc<dyn Fn(FontKey) -> Option<FontData>> {
    let map = fonts.snapshot();
    Rc::new(move |key| map.get(&key).cloned())
}

/// Full pipeline: layout is already done by the caller.
///
/// Builds the display list (resolving images through `image_resolver`
/// and fonts through an explicitly owned [`FontCache`]), renders
/// offscreen via WebGPU when an adapter is available, and falls back to
/// the CPU rasterizer otherwise. Returns RGBA8 row-major pixels at
/// `viewport` rounded to whole pixels; an empty vector when the viewport
/// has zero area.
///
/// Both paths render real glyph coverage: the GPU path and the CPU
/// fallback resolve through the same cache the display list was built
/// with.
///
/// The buffer is composited over an opaque white canvas before returning
/// (see [`composite_over_white`]).
pub fn render_layout_to_rgba(
    root: &LayoutBox,
    doc: &Document,
    viewport: Size,
    image_resolver: &mut dyn FnMut(&str) -> Option<(ImageId, DecodedImage)>,
) -> Vec<u8> {
    render_layout_to_rgba_with_scroll(root, doc, viewport, image_resolver, (0.0, 0.0))
}

/// [`render_layout_to_rgba`] with an explicit viewport scroll offset
/// (`scroll.0` = x, `scroll.1` = y, CSS px): subtracted from content
/// coordinates at draw time, so a headless render shows the scrolled
/// position instead of always the top of the document. Both the GPU
/// path and the CPU fallback honor it; the plain entry keeps the
/// historical zero-scroll contract.
pub fn render_layout_to_rgba_with_scroll(
    root: &LayoutBox,
    doc: &Document,
    viewport: Size,
    image_resolver: &mut dyn FnMut(&str) -> Option<(ImageId, DecodedImage)>,
    scroll: (f32, f32),
) -> Vec<u8> {
    let Some((w, h)) = viewport_whole(viewport) else {
        return Vec::new();
    };
    let mut fonts = FontCache::new();
    let display_list =
        build_display_list_with_fonts(root, doc, &mut fonts, image_resolver, viewport);
    let mut pixels = if let Some(p) = crate::wgpu_be::WgpuBackend::render_offscreen_with_scroll(
        &display_list,
        w,
        h,
        &fonts,
        scroll,
    ) {
        p
    } else {
        // CPU fallback with real font data: the hook resolves the
        // display list's `FontKey`s against a snapshot of the same
        // cache the resolver above used, so text paints with real
        // glyphs here too (unlike the static fontless bridge).
        let mut cpu = crate::cpu::CpuBackend::new(font_data_hook(&fonts));
        cpu.render_to_rgba_full(&display_list, w, h, 1.0, scroll)
    };
    composite_over_white(&mut pixels);
    pixels
}

/// Deterministic CPU-only twin of [`render_layout_to_rgba`]: never tries
/// the WebGPU backend, so output is identical with and without a GPU
/// adapter. The golden suite in `tests/golden.rs` renders through this.
///
/// The display list carries real [`FontKey`]s (resolved through the entry
/// point's [`FontCache`] like the GPU path), but the static CPU bridge has
/// no font data, so text runs still rasterize to nothing here and the
/// golden pixels are unchanged. Same canvas contract (opaque white) and
/// zero-area behavior.
pub fn render_layout_to_rgba_cpu(
    root: &LayoutBox,
    doc: &Document,
    viewport: Size,
    image_resolver: &mut dyn FnMut(&str) -> Option<(ImageId, DecodedImage)>,
) -> Vec<u8> {
    let Some((w, h)) = viewport_whole(viewport) else {
        return Vec::new();
    };
    let mut fonts = FontCache::new();
    let display_list =
        build_display_list_with_fonts(root, doc, &mut fonts, image_resolver, viewport);
    let mut pixels = crate::cpu::CpuBackend::render_to_rgba(&display_list, w, h);
    composite_over_white(&mut pixels);
    pixels
}

/// Convenience: parse an HTML string, style it, lay it out, and render to
/// RGBA8 row-major pixels at `viewport_px` (test/dev only).
///
/// Pipeline: [`parse_and_layout`] (the UA stylesheet comes from
/// [`bowser_layout::layout`] itself) → [`render_layout_to_rgba`] with an
/// image resolver that always returns `None`.
///
/// Returns `4 * w * h` bytes, or an empty vector for a zero-area viewport.
/// See the module docs for rendering limitations.
pub fn render_html_to_rgba(html: &str, viewport_px: (u32, u32)) -> Vec<u8> {
    render_html_to_rgba_with_scroll(html, viewport_px, (0.0, 0.0))
}

/// [`render_html_to_rgba`] with an explicit viewport scroll offset
/// (`scroll.0` = x, `scroll.1` = y, CSS px). Test/dev only.
pub fn render_html_to_rgba_with_scroll(
    html: &str,
    viewport_px: (u32, u32),
    scroll: (f32, f32),
) -> Vec<u8> {
    let (doc, root) = parse_and_layout(html, viewport_px);
    let viewport = Size {
        w: viewport_px.0 as f32,
        h: viewport_px.1 as f32,
    };
    render_layout_to_rgba_with_scroll(&root, &doc, viewport, &mut |_url: &str| None, scroll)
}

/// Deterministic CPU-only twin of [`render_html_to_rgba`]: parses, lays
/// out, and renders on the CPU backend only, so the result is identical
/// with and without a GPU adapter. The golden suite in `tests/golden.rs`
/// renders through this.
///
/// Text runs carry real [`FontKey`]s but rasterize to nothing on this
/// path (no font data is available to the static CPU bridge); see the
/// module docs for the canvas contract.
pub fn render_html_to_rgba_cpu(html: &str, viewport_px: (u32, u32)) -> Vec<u8> {
    let (doc, root) = parse_and_layout(html, viewport_px);
    let viewport = Size {
        w: viewport_px.0 as f32,
        h: viewport_px.1 as f32,
    };
    render_layout_to_rgba_cpu(&root, &doc, viewport, &mut |_url: &str| None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display_list::DisplayItem;

    /// Layout `html` and return the text runs' [`FontKey`]s in document
    /// order, built through the real font resolver over `fonts`.
    fn text_run_keys(html: &str, fonts: &mut FontCache) -> Vec<FontKey> {
        let (doc, _quirks) = Document::parse_html(html.as_bytes());
        let author = bowser_css::parse_stylesheet("");
        let viewport = Size { w: 400.0, h: 400.0 };
        let root =
            bowser_layout::layout(&doc, &[&author], viewport, &bowser_layout::FixedFontMetrics);
        let list = build_display_list_with_fonts(&root, &doc, fonts, &mut |_| None, viewport);
        list.items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::TextRun { font_key, .. } => Some(*font_key),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn font_key_matches_computed_triple() {
        // The `font:` shorthand expands to the font-style/font-weight/
        // font-family longhands; the run's key must equal a direct
        // resolve of that triple.
        let mut fonts = FontCache::new();
        let keys = text_run_keys(
            r#"<p style="font: italic bold 16px serif">hi</p>"#,
            &mut fonts,
        );
        assert_eq!(keys.len(), 1, "expected one text run");
        let direct = fonts
            .resolve("serif", 700, FontStyle::Italic)
            .unwrap_or(FontKey(0));
        assert_eq!(keys[0], direct);
    }

    #[test]
    fn font_keys_distinguish_family_weight_style() {
        let mut fonts = FontCache::new();
        let keys = text_run_keys(
            concat!(
                r#"<p style="font-family: monospace">a</p>"#,
                r#"<p style="font-weight: 700">b</p>"#,
                r#"<p style="font-style: oblique">c</p>"#,
            ),
            &mut fonts,
        );
        assert_eq!(keys.len(), 3, "expected three text runs");
        let a = fonts
            .resolve("monospace", 400, FontStyle::Normal)
            .unwrap_or(FontKey(0));
        let b = fonts
            .resolve("serif", 700, FontStyle::Normal)
            .unwrap_or(FontKey(0));
        let c = fonts
            .resolve("serif", 400, FontStyle::Oblique)
            .unwrap_or(FontKey(0));
        assert_eq!(keys, [a, b, c], "keys must match the computed triples");
        // Distinct triples mint distinct keys even when the underlying
        // face file coincides.
        assert!(
            a != b && b != c && a != c,
            "distinct (family, weight, style) triples need distinct keys"
        );
    }

    #[test]
    fn full_pipeline_paints_text_with_real_fonts() {
        // Prefers the GPU backend, falls back to the CPU rasterizer;
        // both now render real glyph coverage (system fonts or the
        // embedded fallback), so text must paint non-white pixels on the
        // opaque white canvas either way.
        let px = render_html_to_rgba("<p>hi</p>", (200, 100));
        assert_eq!(px.len(), 200 * 100 * 4);
        let nonwhite = px
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[0] != 255 || p[1] != 255 || p[2] != 255)
            .count();
        assert!(nonwhite > 10, "expected painted text pixels");
    }

    #[test]
    fn deterministic_path_still_paints_no_text() {
        // Golden-stability guard: the deterministic CPU bridge has no
        // font data, so even with real FontKeys in the list, text runs
        // rasterize to nothing and the pixels stay exactly as before.
        let px = render_html_to_rgba_cpu("<p>hi</p>", (200, 100));
        assert_eq!(px.len(), 200 * 100 * 4);
        assert!(
            px.as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[0] == 255 && p[1] == 255 && p[2] == 255 && p[3] == 255),
            "deterministic path must stay all-white (no font data)"
        );
    }

    #[test]
    fn zero_area_viewport_returns_empty() {
        assert!(render_html_to_rgba("<p>hi</p>", (0, 100)).is_empty());
        assert!(render_html_to_rgba_cpu("<p>hi</p>", (100, 0)).is_empty());
        let (doc, root) = parse_and_layout("<p>hi</p>", (100, 100));
        let viewport = Size { w: 0.0, h: 100.0 };
        assert!(render_layout_to_rgba(&root, &doc, viewport, &mut |_| None).is_empty());
        assert!(render_layout_to_rgba_cpu(&root, &doc, viewport, &mut |_| None).is_empty());
    }

    #[test]
    fn parse_and_layout_finds_styled_text() {
        // The shared parse/layout helper honors author CSS: the styled
        // paragraph lays out wider than the unstyled one.
        let (_, root) = parse_and_layout(
            "<style>p.big { font-size: 64px }</style><p class=\"big\">hi</p>",
            (400, 400),
        );
        let (_, plain) = parse_and_layout("<p>hi</p>", (400, 400));
        assert!(
            layout_width(&root) > layout_width(&plain),
            "64px text must lay out wider than default-size text"
        );
    }

    /// Width of the widest text fragment in the layout tree.
    fn layout_width(root: &LayoutBox) -> f32 {
        fn walk(node: &LayoutBox, best: &mut f32) {
            for fragment in &node.fragments {
                *best = best.max(fragment.rect.w);
            }
            for child in &node.children {
                walk(child, best);
            }
        }
        let mut best = 0.0;
        walk(root, &mut best);
        best
    }
}
