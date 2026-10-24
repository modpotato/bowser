//! WebGPU paint backend (`wgpu` 30).
//!
//! Three tiny instanced WGSL pipelines share one per-frame uniform
//! (viewport size + scroll offset, so scrolling never rebuilds anything):
//!
//! 1. **solid** — per-instance rect (page px) + color;
//! 2. **glyph** — per-instance rect + atlas uv + color (grayscale coverage
//!    multiplied by the color's alpha);
//! 3. **image** — per-instance rect + uv + alpha, sampling an RGBA texture:
//!    the texture's straight alpha is multiplied by the per-instance alpha
//!    (which folds the current opacity group in); `ALPHA_BLENDING`
//!    premultiplies at blend time.
//!
//! [`DisplayList`] lowering: `FillRect` becomes solid quads; `Border` goes
//! through [`crate::borders::tessellate_border`], or
//! [`crate::borders::tessellate_border_rounded`] when any corner radius in
//! the item's `radii` is positive; `TextRun` carries verbatim text, shaped
//! here with the shared [`crate::text::shape_text`] (kerning, ligatures,
//! complex scripts — the same run the CPU backend paints, so both
//! backends' pixels agree), each shaped glyph placed from its atlas
//! [`crate::text::GlyphCell`]'s ink UV subrect and bearings;
//! `Image`/`BackgroundImage` upload [`crate::display_list::DecodedImage`]
//! bytes once and cache the texture by [`crate::display_list::ImageId`];
//! a per-id content-hash probe (full chunked FNV-1a over the decoded bytes
//! plus the dimensions, computed once at registration and cached in the
//! display list) makes the per-frame "same id, same bytes?" check O(1)
//! instead of O(bytes).
//! [`ImageId`]s are only unique within one list, and lists are routinely
//! rebuilt at reused addresses, so the probe compares the content hash
//! (plus width/height) rather than the allocation address. The current
//! opacity folds into a per-instance alpha;
//! `Decoration` and `ListMarker` go through
//! [`crate::borders::decoration_geometry`] /
//! [`crate::borders::marker_geometry`], with decimal markers rendered as
//! text via [`crate::borders::marker_text`] using the item's 1-based
//! `index`; `PushClip`/`PushRoundedClip`/`PopClip` maintain a clip
//! stack (snapshotted per change; the fragment shader evaluates the
//! stack's geometric intersection per pixel, scroll-subtracted at draw
//! time); `PushOpacity`/`PopOpacity`
//! render isolated opacity groups: a group with opacity 1.0 is a no-op, a
//! group with opacity 0.0 paints nothing, and a group holding a single
//! paint item (no nested groups) folds the opacity into the item's
//! per-instance alpha — exactly isolated compositing for one item. Any
//! other group is lowered separately, rendered into an offscreen texture
//! at its own alphas, and composited over the target with the group
//! opacity through a dedicated premultiplied composite pipeline. Each
//! pushed value is clamped to [0,1] like the CPU backend.
//!
//! # Documented approximations
//!
//! - **Opacity groups**: CSS `opacity` on a subtree composites the whole
//!   group in isolation and then blends it; this backend does exactly
//!   that (offscreen group texture + composite), with the fast paths
//!   above. The CPU backend (`crate::cpu`) implements the same semantics.
//! - **Text shaping**: `TextRun` carries verbatim text; this backend
//!   shapes it with the shared [`crate::text::shape_text`] (kerning,
//!   ligatures, complex scripts) — the same run the CPU backend paints,
//!   so both backends' pixels agree. Coverage is the real swash
//!   rasterization from [`crate::text`] over [`crate::fonts::FontData`]
//!   resolved through the backend's `get_font_data` hook (same hook shape
//!   as the CPU backend); runs whose font does not resolve paint nothing,
//!   exactly like the CPU backend.
//!
//! # SPEC-DELTA notes (wgpu worker, 2026-09-25)
//!
//! The task SPEC assumed sibling modules that landed with different
//! shapes; this file codes against the tree as it exists:
//!
//! - `borders::tessellate_border` takes per-side `widths`/`styles`/`colors`
//!   arrays, so `DisplayItem::Border` carries those arrays (the SPEC's
//!   single color/style would not compile).
//! - `borders::marker_geometry(kind, x, y, size, color)` and
//!   `borders::decoration_geometry(kind, x0, x1, y, thickness, color)`
//!   take scalars, not rects; the agreed-spec type is `DecoKind` (with
//!   wavy variants), not `DecorationKind`.
//! - `render_offscreen` is an **associated function**
//!   `WgpuBackend::render_offscreen(list, width_px, height_px, fonts) ->
//!   Option<Vec<u8>>`, exactly as the landed `crate::render` calls it;
//!   the headless device is brought up once per thread and reused across
//!   calls (thread-local, so concurrent callers never serialize on a
//!   global lock).
//!   `fonts` is the [`crate::fonts::FontCache`] the display list was built
//!   with, threaded explicitly (no global font cache): the `FontKey`s in
//!   the list are cache-local, so they must resolve against the same
//!   instance. The SPEC's `&mut self` method form with a `viewport: Size`
//!   parameter cannot coexist under the same name;
//!   `render_offscreen_with_viewport` covers that shape as a method.
//!   `None` means no Vulkan adapter was found (the `render` module falls
//!   back to the CPU rasterizer); the unit tests treat `None` as a loud
//!   failure so CI notices a missing GPU.
//! - `Backend::paint` lowers the list to validate it (no surface
//!   attached); the app worker drives `render_to_surface` per frame.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bowser_layout::{Rect, Size};

use crate::borders::{
    decoration_geometry, marker_geometry, marker_text, tessellate_border, tessellate_border_rounded,
};
use crate::clip::{ClipSnapshot, MAX_CLIP_ENTRIES, MAX_POLY_VERTS, PageClip, snapshot_clips};
use crate::cpu::{
    MAX_OPACITY_GROUPS, opacity_foldable, precompute_group_scans, scan_blend_group,
    scan_filter_group, scan_opacity_group, scan_transform_group, transform_rect_bounds,
};
use crate::display_list::{
    Backend, BackgroundRepeat, BlendMode, Color, DecoKind, DecodedImage, DisplayItem, DisplayList,
    EDGE_GLOW_COLOR, Edge4, FontKey, ImageId, MarkerKind, background_space_axis, cap_repeat_tile,
    clamp_repeat_area, edge_glow_alpha, repeat_snap_band,
};
use crate::filter::{ColorMatrix, FilterPass, blur_box_radius, filter_spill, plan_filter_passes};
use crate::fonts::{FontCache, FontData};
use crate::gradient::{Gradient, bake_gradient_rgba8};
use crate::text::{ATLAS_SIZE, Atlas, rounded_size, shape_text};

// ---------------------------------------------------------------------------
// WGSL shaders
// ---------------------------------------------------------------------------

/// Shared uniform block: viewport size and scroll offset, all in page px.
/// The vertex shader maps page px to NDC; scrolling only rewrites this.
const SHADER_COMMON: &str = r#"
struct FrameUniform {
    viewport: vec2<f32>,
    scroll: vec2<f32>,
};
@group(0) @binding(0) var<uniform> frame: FrameUniform;

fn page_to_ndc(p: vec2<f32>) -> vec2<f32> {
    let v = p - frame.scroll;
    return vec2<f32>(
        v.x / frame.viewport.x * 2.0 - 1.0,
        1.0 - v.y / frame.viewport.y * 2.0,
    );
}

// Quad corners from vertex_index for a triangle strip:
// 0 -> (0,0), 1 -> (1,0), 2 -> (0,1), 3 -> (1,1).
fn strip_corner(vi: u32) -> vec2<f32> {
    return vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
}

// Exact fractional clip, in device px of the current render target.
// Up to 8 clip-stack entries (oldest first); `count == 0` means no
// clip. The per-pixel coverage is the coverage of the entries'
// geometric intersection (not the min of per-entry coverages: those
// differ for nested fractional rects, e.g. 128 vs 64). Entry rects
// are intersected exactly; pixels touching a rounded corner square
// are 8x8-subsampled with per-sample membership in every entry,
// mirroring `clip_stack_coverage_byte` in clip.rs op-for-op (same f32
// ops in the same order, same `floor(v + 0.5)` quantization): with
// all-zero radii it is exactly the old rect formula, so both backends
// agree on clip boundaries pixel-for-pixel. Bound at group 2 on every
// pipeline; the composite pipeline leaves it disabled (its content was
// already clipped when the group rendered).
struct ClipEntryU {
    rect: vec4<f32>,
    radii: vec4<f32>,
    // 0 = rounded-rect entry, 1 = `clip-path` polygon entry
    // (even-odd rule over `poly[..poly_len].xy`; `rect` is its bbox).
    kind: u32,
    poly_len: u32,
    // Pad so `poly` lands on a 16-byte boundary: in the `uniform`
    // address space an array's stride must be a multiple of 16, so the
    // vertices are `vec4`s (`.xy` used) rather than `vec2`s.
    _pad: vec2<u32>,
    poly: array<vec4<f32>, 48>,
};
struct ClipUniform {
    count: u32,
    has_rounded: u32,
    has_poly: u32,
    _pad: u32,
    entries: array<ClipEntryU, 8>,
};
@group(2) @binding(0) var<uniform> clip: ClipUniform;

fn point_in_rounded_rect(
    x0: f32, y0: f32, x1: f32, y1: f32,
    radii: vec4<f32>, sx: f32, sy: f32,
) -> bool {
    if (sx < x0 || sx >= x1 || sy < y0 || sy >= y1) {
        return false;
    }
    let rtl = radii.x;
    let rtr = radii.y;
    let rbr = radii.z;
    let rbl = radii.w;
    if (rtl > 0.0 && sx < x0 + rtl && sy < y0 + rtl) {
        let dx = sx - (x0 + rtl);
        let dy = sy - (y0 + rtl);
        return dx * dx + dy * dy <= rtl * rtl;
    }
    if (rtr > 0.0 && sx > x1 - rtr && sy < y0 + rtr) {
        let dx = sx - (x1 - rtr);
        let dy = sy - (y0 + rtr);
        return dx * dx + dy * dy <= rtr * rtr;
    }
    if (rbr > 0.0 && sx > x1 - rbr && sy > y1 - rbr) {
        let dx = sx - (x1 - rbr);
        let dy = sy - (y1 - rbr);
        return dx * dx + dy * dy <= rbr * rbr;
    }
    if (rbl > 0.0 && sx < x0 + rbl && sy > y1 - rbl) {
        let dx = sx - (x0 + rbl);
        let dy = sy - (y1 - rbl);
        return dx * dx + dy * dy <= rbl * rbl;
    }
    return true;
}

// Point-in-polygon (even-odd rule) for one subpixel sample, using
// only exactly-rounded f32 ops (`+ - * /`, comparisons) so the CPU
// (`point_in_polygon` in clip.rs) and this mirror agree bit-for-bit.
fn point_in_poly(px: f32, py: f32, e: ClipEntryU) -> bool {
    let n = e.poly_len;
    if (n < 3u) {
        return false;
    }
    var inside = false;
    for (var i = 0u; i < n; i++) {
        let p1 = e.poly[i];
        let p2 = e.poly[(i + 1u) % n];
        if ((p1.y > py) != (p2.y > py)) {
            let xinters = (p2.x - p1.x) * (py - p1.y) / (p2.y - p1.y) + p1.x;
            if (px < xinters) {
                inside = !inside;
            }
        }
    }
    return inside;
}

// Whether any polygon edge's bounding box overlaps the pixel's unit
// square: a conservative (superset) trigger for "a polygon edge
// crosses this pixel", mirroring `polygon_touches_pixel` in clip.rs.
fn poly_touches_pixel(e: ClipEntryU, px: vec2<f32>) -> bool {
    let n = e.poly_len;
    if (n < 3u) {
        return false;
    }
    let qx = px.x + 1.0;
    let qy = px.y + 1.0;
    for (var i = 0u; i < n; i++) {
        let p1 = e.poly[i];
        let p2 = e.poly[(i + 1u) % n];
        if (min(p1.x, p2.x) <= qx && max(p1.x, p2.x) >= px.x
            && min(p1.y, p2.y) <= qy && max(p1.y, p2.y) >= px.y) {
            return true;
        }
    }
    return false;
}

// Fractional coverage of the device pixel `px` (the unit square
// `[px, px+1)`) by the clip stack's geometric intersection, quantized
// exactly like the CPU mask byte. Mirrors `clip_stack_coverage_byte`
// in clip.rs op-for-op; keep them in sync.
fn clip_coverage(px: vec2<f32>) -> f32 {
    // Intersect the entry rects.
    var l = -3.402823466e+38;
    var t = -3.402823466e+38;
    var r = 3.402823466e+38;
    var b = 3.402823466e+38;
    for (var i = 0u; i < clip.count; i++) {
        let e = clip.entries[i];
        l = max(l, e.rect.x);
        t = max(t, e.rect.y);
        r = min(r, e.rect.z);
        b = min(b, e.rect.w);
    }
    if (l >= r || t >= b) {
        return 0.0;
    }
    let cx = clamp(min(r, px.x + 1.0) - max(l, px.x), 0.0, 1.0);
    let cy = clamp(min(b, px.y + 1.0) - max(t, px.y), 0.0, 1.0);
    let rect_cov = floor(cx * cy * 255.0 + 0.5) / 255.0;
    // Fast path: no rounded corners and no polygons in the stack. (The
    // per-entry reads are avoided here; see `has_rounded`/`has_poly`.)
    if (clip.has_rounded == 0u && clip.has_poly == 0u) {
        return rect_cov;
    }
    // Does the pixel touch any positive-radius corner square?
    var in_corner = false;
    for (var i = 0u; i < clip.count; i++) {
        let e = clip.entries[i];
        let x0 = e.rect.x;
        let y0 = e.rect.y;
        let x1 = e.rect.z;
        let y1 = e.rect.w;
        let rtl = max(e.radii.x, 0.0);
        let rtr = max(e.radii.y, 0.0);
        let rbr = max(e.radii.z, 0.0);
        let rbl = max(e.radii.w, 0.0);
        if (rtl > 0.0 && px.x < x0 + rtl && px.x + 1.0 > x0 && px.y < y0 + rtl && px.y + 1.0 > y0) { in_corner = true; }
        if (rtr > 0.0 && px.x + 1.0 > x1 - rtr && px.x < x1 && px.y < y0 + rtr && px.y + 1.0 > y0) { in_corner = true; }
        if (rbr > 0.0 && px.x + 1.0 > x1 - rbr && px.x < x1 && px.y + 1.0 > y1 - rbr && px.y < y1) { in_corner = true; }
        if (rbl > 0.0 && px.x < x0 + rbl && px.x + 1.0 > x0 && px.y + 1.0 > y1 - rbl && px.y < y1) { in_corner = true; }
        if (in_corner) {
            break;
        }
    }
    var need_sample = in_corner;
    // Polygon entries: the pixel's unit square either misses the
    // entry's bounding box (coverage 0), is crossed by a polygon edge
    // (needs sampling), or is uniformly inside/outside the polygon
    // (the center sample decides).
    if (clip.has_poly != 0u && !need_sample) {
        for (var k = 0u; k < clip.count; k++) {
            let e = clip.entries[k];
            if (e.kind != 1u) {
                continue;
            }
            if (e.poly_len < 3u) {
                return 0.0;
            }
            if (px.x + 1.0 <= e.rect.x || px.x >= e.rect.z
                || px.y + 1.0 <= e.rect.y || px.y >= e.rect.w) {
                return 0.0;
            }
            if (poly_touches_pixel(e, px)) {
                need_sample = true;
                break;
            }
            if (!point_in_poly(px.x + 0.5, px.y + 0.5, e)) {
                return 0.0;
            }
        }
    }
    if (!need_sample) {
        return rect_cov;
    }
    // 8x8 subpixel sampling with per-sample all-entry membership.
    var inside = 0u;
    for (var j = 0u; j < 8u; j++) {
        let sy = px.y + (f32(j) + 0.5) / 8.0;
        for (var i = 0u; i < 8u; i++) {
            let sx = px.x + (f32(i) + 0.5) / 8.0;
            var ok = true;
            for (var k = 0u; k < clip.count; k++) {
                let e = clip.entries[k];
                var entry_ok = true;
                if (e.kind == 1u) {
                    entry_ok = point_in_poly(sx, sy, e);
                } else {
                    entry_ok = point_in_rounded_rect(e.rect.x, e.rect.y, e.rect.z, e.rect.w, e.radii, sx, sy);
                }
                if (!entry_ok) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                inside++;
            }
        }
    }
    return floor(f32(inside) * (255.0 / 64.0) + 0.5) / 255.0;
}
"#;

const SHADER_SOLID: &str = r#"
struct VsIn {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) color: vec4<f32>,
};
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_solid(in: VsIn) -> VsOut {
    let p = in.rect.xy + strip_corner(in.vi) * in.rect.zw;
    var out: VsOut;
    out.pos = vec4<f32>(page_to_ndc(p), 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_solid(in: VsOut) -> @location(0) vec4<f32> {
    let q = clip_coverage(floor(in.pos.xy));
    return vec4<f32>(in.color.rgb, in.color.a * q);
}
"#;

const SHADER_GLYPH: &str = r#"
@group(1) @binding(0) var atlas_tex: texture_2d<f32>;
@group(1) @binding(1) var atlas_smp: sampler;

struct VsIn {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) uv: vec4<f32>,
    @location(2) color: vec4<f32>,
};
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_glyph(in: VsIn) -> VsOut {
    let corner = strip_corner(in.vi);
    let p = in.rect.xy + corner * in.rect.zw;
    var out: VsOut;
    out.pos = vec4<f32>(page_to_ndc(p), 0.0, 1.0);
    out.uv = mix(in.uv.xy, in.uv.zw, corner);
    out.color = in.color;
    return out;
}

@fragment
fn fs_glyph(in: VsOut) -> @location(0) vec4<f32> {
    let coverage = textureSample(atlas_tex, atlas_smp, in.uv).r;
    let q = clip_coverage(floor(in.pos.xy));
    return vec4<f32>(in.color.rgb, in.color.a * coverage * q);
}
"#;

const SHADER_IMAGE: &str = r#"
@group(1) @binding(0) var img_tex: texture_2d<f32>;
@group(1) @binding(1) var img_smp: sampler;

struct VsIn {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) uv: vec4<f32>,
    @location(2) alpha: f32,
    // Repeat params: (mode_x, mode_y, pitch_x, pitch_y). uv is in tile
    // units (see `image_repeat_quad`); mode per axis is 0 = single copy
    // (discard outside [0,1)), 1 = wrap (fract), 2 = spaced (whole tiles
    // on `pitch`, discard the gaps). Plain image quads pass 0/0.
    @location(3) params: vec4<f32>,
    // Tile counts per axis for `spaced` mode; the wrap-mode
    // tile-boundary snap band (see `lower_repeat_image`) otherwise.
    @location(4) counts: vec2<f32>,
};
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) alpha: f32,
    @location(2) params: vec4<f32>,
    @location(3) counts: vec2<f32>,
};

@vertex
fn vs_image(in: VsIn) -> VsOut {
    let corner = strip_corner(in.vi);
    let p = in.rect.xy + corner * in.rect.zw;
    var out: VsOut;
    out.pos = vec4<f32>(page_to_ndc(p), 0.0, 1.0);
    out.uv = mix(in.uv.xy, in.uv.zw, corner);
    out.alpha = in.alpha;
    out.params = in.params;
    out.counts = in.counts;
    return out;
}

// One axis of a repeat-uv in tile units -> sampled coordinate.
//
// Residual precision (deliberate): the vertex UVs span the whole area in
// tile units, so the coordinate evaluated here carries a residual of a
// few ulps at that magnitude (magnitude = max |tile-unit uv| over the
// quad, the u0 offset included) versus the CPU backend's exact per-tile
// coordinates. The budget has three contributors, ~1 ulp of the magnitude
// each: the vertex shader's `mix(in.uv.xy, in.uv.zw, corner)` rounding,
// the rasterizer's f32 affine interpolation of the varying across the
// quad (a few ulps worst case across vendors' evaluation orders), and
// the `fract` rounding below. This is inherent to f32 area-wide affine
// interpolation and cannot be made exact without abandoning the
// single-quad design (see `lower_repeat_image`); the CPU/GPU differential
// tests budget ~4 ulps of the tile-unit magnitude for the whole chain.
// Tile-boundary fragments are additionally canonicalized to the CPU's
// left-tile convention via the wrap-mode snap band (see below), whose
// width is 4 ulps of that same magnitude (see `repeat_snap_band`).
fn repeat_axis(t: f32, mode: f32, pitch: f32, n: f32) -> f32 {
    if (mode < 0.5) {
        // Single copy: fragments outside the one tile paint nothing.
        if (t < 0.0 || t >= 1.0) { discard; }
        return t;
    }
    if (mode < 1.5) {
        // Wrap: `n` is the tile-boundary snap band (see
        // `lower_repeat_image`). A fragment center exactly on a tile
        // boundary belongs to the tile on its left: the CPU backend's
        // rect rasterization keeps a boundary pixel in the left tile,
        // whose edge taps clamp to its last texel. But the residue chain
        // (vertex `mix` + rasterizer interpolation + `fract` rounding,
        // ~4 ulps of the tile-unit magnitude — see above) can lift `t`
        // just above the integer, where fract would wrap to ~0 and
        // sample the next tile's first texel instead, or pull `t` just
        // below it, where fract reads ~1 (one ulp below the exact
        // boundary coordinate the CPU samples). Snap the residue band
        // on both sides of the boundary to 1.0: the clamp-to-edge
        // sampler then reads the last texel at full weight, exactly
        // like the CPU's edge taps.
        let u = fract(t);
        if (u < n || u > 1.0 - n) {
            return 1.0;
        }
        return u;
    }
    // Spaced: whole tiles of width 1 on a `pitch`, `n` of them; the
    // gaps between tiles (and everything past the last tile) paint
    // nothing. The epsilon on the tile index keeps a fragment sitting
    // exactly on a tile's left edge (up to float error) in that tile
    // instead of being discarded as past the previous tile's right edge.
    let k = floor(t / pitch + 1e-4);
    let w = t - k * pitch;
    if (k < 0.0 || k >= n || w < 0.0 || w >= 1.0) { discard; }
    return w;
}

@fragment
fn fs_image(in: VsOut) -> @location(0) vec4<f32> {
    // Fold the per-instance alpha (opacity group) into the texture's
    // straight alpha, mirroring `Color::with_opacity`; ALPHA_BLENDING
    // premultiplies at blend time, and the readback path unpremultiplies
    // back to straight RGBA8.
    let u = repeat_axis(in.uv.x, in.params.x, in.params.z, in.counts.x);
    let v = repeat_axis(in.uv.y, in.params.y, in.params.w, in.counts.y);
    let tex = textureSample(img_tex, img_smp, vec2<f32>(u, v));
    let q = clip_coverage(floor(in.pos.xy));
    return vec4<f32>(tex.rgb, tex.a * in.alpha * q);
}
"#;

/// Composite an isolated opacity group's offscreen texture over the
/// target. Unlike the image pipeline, the group texture already holds
/// premultiplied framebuffer values, so both rgb and alpha are scaled by
/// the per-instance group opacity and the pipeline blends premultiplied
/// (`PREMULTIPLIED_ALPHA_BLENDING`): out = tex.rgb*a + dst*(1-tex.a*a).
const SHADER_COMPOSITE: &str = r#"
@group(1) @binding(0) var grp_tex: texture_2d<f32>;
@group(1) @binding(1) var grp_smp: sampler;

struct VsIn {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) uv: vec4<f32>,
    @location(2) alpha: f32,
    // 2D transform `(a, b, c, d, e, f)` applied to the quad's page-px
    // corners (identity for opacity/blend groups): the group texture
    // holds the untransformed rendering.
    @location(4) mat_ab: vec2<f32>,
    @location(5) mat_cd: vec2<f32>,
    @location(6) mat_ef: vec2<f32>,
};
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) alpha: f32,
};

@vertex
fn vs_composite(in: VsIn) -> VsOut {
    let corner = strip_corner(in.vi);
    let local = in.rect.xy + corner * in.rect.zw;
    let p = vec2<f32>(
        in.mat_ab.x * local.x + in.mat_cd.x * local.y + in.mat_ef.x,
        in.mat_ab.y * local.x + in.mat_cd.y * local.y + in.mat_ef.y,
    );
    var out: VsOut;
    out.pos = vec4<f32>(page_to_ndc(p), 0.0, 1.0);
    out.uv = mix(in.uv.xy, in.uv.zw, corner);
    out.alpha = in.alpha;
    return out;
}

@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(grp_tex, grp_smp, in.uv);
    return vec4<f32>(tex.rgb * in.alpha, tex.a * in.alpha);
}
"#;

/// Blend-mode group compositing: samples the isolated group texture
/// (`grp_tex`, premultiplied) and a copy of the backdrop
/// (`backdrop_tex`, premultiplied) over the group's bounds, and
/// evaluates the CSS compositing formula for the instance's blend
/// mode in the fragment shader. The pipeline leaves blending disabled
/// (REPLACE): the shader outputs the fully computed premultiplied
/// result, which becomes the framebuffer value exactly.
///
/// Mirrors `blend_pixel`/`blend_channel` in cpu.rs op-for-op in f32
/// (unpremultiply as channel/alpha), so the backends agree within
/// rounding.
const SHADER_BLEND_COMPOSITE: &str = r#"
@group(1) @binding(0) var grp_tex: texture_2d<f32>;
@group(1) @binding(1) var grp_smp: sampler;
@group(1) @binding(2) var backdrop_tex: texture_2d<f32>;

struct VsIn {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) uv: vec4<f32>,
    @location(2) alpha: f32,
    @location(3) mode: u32,
    // 2D transform like `vs_composite` (identity unless a group ever
    // carries both a blend mode and a transform).
    @location(4) mat_ab: vec2<f32>,
    @location(5) mat_cd: vec2<f32>,
    @location(6) mat_ef: vec2<f32>,
};
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) mode: u32,
};

@vertex
fn vs_blend_composite(in: VsIn) -> VsOut {
    let corner = strip_corner(in.vi);
    let local = in.rect.xy + corner * in.rect.zw;
    let p = vec2<f32>(
        in.mat_ab.x * local.x + in.mat_cd.x * local.y + in.mat_ef.x,
        in.mat_ab.y * local.x + in.mat_cd.y * local.y + in.mat_ef.y,
    );
    var out: VsOut;
    out.pos = vec4<f32>(page_to_ndc(p), 0.0, 1.0);
    out.uv = mix(in.uv.xy, in.uv.zw, corner);
    out.mode = in.mode;
    return out;
}

// One channel of the blend function B(Cb, Cs), unpremultiplied (CSS
// Compositing and Blending L1). Discriminants match
// `BlendMode::as_u32`.
fn blend_channel(mode: u32, cb: f32, cs: f32) -> f32 {
    switch mode {
        case 1u: {
            return cb * cs;
        }
        case 2u: {
            return cb + cs - cb * cs;
        }
        case 3u: {
            if (cb <= 0.5) {
                return 2.0 * cb * cs;
            }
            return 1.0 - 2.0 * (1.0 - cb) * (1.0 - cs);
        }
        case 4u: {
            return min(cb, cs);
        }
        case 5u: {
            return max(cb, cs);
        }
        default: {
            return cs;
        }
    }
}

@fragment
fn fs_blend_composite(in: VsOut) -> @location(0) vec4<f32> {
    let src = textureSample(grp_tex, grp_smp, in.uv);
    let dst = textureSample(backdrop_tex, grp_smp, in.uv);
    let sa = src.a;
    let da = dst.a;
    // Unpremultiply in f32. `max(., 1e-7)`: a zero alpha yields a zero
    // color (premultiplied storage), and the composite below then
    // reduces to the backdrop for every mode.
    let cs = src.rgb / max(sa, 0.0000001);
    let cb = dst.rgb / max(da, 0.0000001);
    let b = vec3<f32>(
        blend_channel(in.mode, cb.r, cs.r),
        blend_channel(in.mode, cb.g, cs.g),
        blend_channel(in.mode, cb.b, cs.b),
    );
    // Cs' = (1 − αb)·Cs + αb·B(Cb, Cs).
    let csp = (1.0 - da) * cs + da * b;
    let ao = sa + da * (1.0 - sa);
    // Co = αs·Cs' + αb·(1 − αs)·Cb, premultiplied on output: no extra
    // `* ao` (that would double-count the output alpha).
    let co = clamp(sa * csp + da * (1.0 - sa) * cb, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(co, ao);
}
"#;

/// Box-shadow as a single quad with an analytic rounded-box SDF in the
/// fragment shader (mirrors the CPU backend's coverage math: outset =
/// blurred spread-inflated shape at the offset, knocked out by the
/// sharp border box (CSS Backgrounds 3 §7.2 — outer shadows are drawn
/// outside the border edge only); inset = blurred spread-deflated shape
/// knocked out by the sharp border box).
/// Blur model agreement: the CPU blurs coverage with a gaussian
/// σ = blur/2 centered on the shape edge; the shader centers its
/// smoothstep transition on the edge the same way
/// (`1 − smoothstep(−blur/2, blur/2, d)`). The smoothstep is a
/// single-pass analytic *approximation* of the gaussian, not an exact
/// match — see `cpu_gpu_blurred_shadow_agreement` for the measured
/// tolerance.
/// The vertex shader draws the conservative coverage `area`; the
/// fragment evaluates the shadow shape against `rect`. Straight alpha
/// output with `ALPHA_BLENDING`, like the solid pipeline.
const SHADER_SHADOW: &str = r#"
struct VsIn {
    @builtin(vertex_index) vi: u32,
    @location(0) area: vec4<f32>,
    @location(1) rect: vec4<f32>,
    @location(2) radii: vec4<f32>,
    @location(3) color: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(5) flags: vec4<f32>,
};
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) page: vec2<f32>,
    @location(1) rect: vec4<f32>,
    @location(2) radii: vec4<f32>,
    @location(3) color: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(5) flags: vec4<f32>,
};

@vertex
fn vs_shadow(in: VsIn) -> VsOut {
    let corner = strip_corner(in.vi);
    let p = in.area.xy + corner * in.area.zw;
    var out: VsOut;
    out.pos = vec4<f32>(page_to_ndc(p), 0.0, 1.0);
    out.page = p;
    out.rect = in.rect;
    out.radii = in.radii;
    out.color = in.color;
    out.params = in.params;
    out.flags = in.flags;
    return out;
}

fn shadow_corner_radius(p: vec2<f32>, r: vec4<f32>) -> f32 {
    // r = (TL, TR, BR, BL); page px are y-down.
    if (p.x < 0.0) {
        if (p.y < 0.0) { return r.x; } else { return r.w; }
    } else {
        if (p.y < 0.0) { return r.y; } else { return r.z; }
    }
}

// Signed distance to the rounded box centered at `center` with half
// extents `half` and per-corner radii `r` (negative inside).
fn sd_round_box(px: vec2<f32>, center: vec2<f32>, half: vec2<f32>, r: vec4<f32>) -> f32 {
    let p = px - center;
    let rad = shadow_corner_radius(p, r);
    let q = abs(p) - half + rad;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - rad;
}

@fragment
fn fs_shadow(in: VsOut) -> @location(0) vec4<f32> {
    // params = (offset.x, offset.y, blur, spread); flags.x = inset.
    let blur = max(in.params.z, 0.0);
    let spread = in.params.w;
    let inset = in.flags.x > 0.5;
    let center = in.rect.xy + 0.5 * in.rect.zw;
    let half = 0.5 * in.rect.zw;
    var alpha: f32;
    if (!inset) {
        // Outset: the spread-inflated shape at the shadow offset,
        // blurred with the transition centered on the shape edge
        // (matching the CPU's gaussian σ = blur/2; the smoothstep is a
        // single-pass analytic approximation — see SHADER_SHADOW).
        // CSS Backgrounds 3 §7.2: outer shadows are drawn outside the
        // border edge only, so the sharp border box is knocked out of
        // the coverage.
        let rad = max(in.radii + spread, vec4<f32>(0.0));
        let d = sd_round_box(in.page, center + in.params.xy, half + spread, rad);
        var cov = 1.0 - smoothstep(-0.5 * blur, 0.5 * blur, d);
        if (blur <= 0.0) {
            cov = select(0.0, 1.0, d <= 0.0);
        }
        let d_box = sd_round_box(in.page, center, half, in.radii);
        let sharp = select(0.0, 1.0, d_box <= 0.0);
        cov = cov * (1.0 - sharp);
        alpha = in.color.a * cov;
    } else {
        // Inset: the spread-deflated shape at the offset, blurred with
        // the transition centered on the shape edge (like outset),
        // knocked out by the sharp border box (S − B), clipped to the
        // interior. A deflated-empty shape blurs to nothing, so the
        // whole interior goes dark (matching huge inset spreads).
        let half_i = half - spread;
        let rad_i = max(in.radii - spread, vec4<f32>(0.0));
        let d_box = sd_round_box(in.page, center, half, in.radii);
        let sharp = select(0.0, 1.0, d_box <= 0.0);
        var cov_blur = 0.0;
        if (half_i.x > 0.0 && half_i.y > 0.0) {
            let d = sd_round_box(in.page, center + in.params.xy, half_i, rad_i);
            cov_blur = 1.0 - smoothstep(-0.5 * blur, 0.5 * blur, d);
            if (blur <= 0.0) {
                cov_blur = select(0.0, 1.0, d <= 0.0);
            }
        }
        alpha = in.color.a * clamp(sharp - cov_blur, 0.0, 1.0);
    }
    let q = clip_coverage(floor(in.pos.xy));
    let a = alpha * q;
    return vec4<f32>(in.color.rgb, a);
}
"#;

/// Backdrop-filter write-back: like `SHADER_COMPOSITE` (premultiplied
/// source-over of an offscreen texture quad), but the fragment
/// multiplies by the exact fractional clip coverage. The filtered
/// backdrop is part of the element's rendering, so the element's
/// clip-path (and ancestor clips) clip it exactly like content — the
/// CPU backend's masked write-back in
/// [`paint_backdrop_filter_group`](crate::cpu::Painter::paint_backdrop_filter_group)
/// computes the same `src * q + dst * (1 - src_a * q)` per pixel.
const SHADER_COMPOSITE_CLIP: &str = r#"
@group(1) @binding(0) var grp_tex: texture_2d<f32>;
@group(1) @binding(1) var grp_smp: sampler;

struct VsIn {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) uv: vec4<f32>,
    @location(2) alpha: f32,
    @location(4) mat_ab: vec2<f32>,
    @location(5) mat_cd: vec2<f32>,
    @location(6) mat_ef: vec2<f32>,
};
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) alpha: f32,
};

@vertex
fn vs_composite_clip(in: VsIn) -> VsOut {
    let corner = strip_corner(in.vi);
    let local = in.rect.xy + corner * in.rect.zw;
    let p = vec2<f32>(
        in.mat_ab.x * local.x + in.mat_cd.x * local.y + in.mat_ef.x,
        in.mat_ab.y * local.x + in.mat_cd.y * local.y + in.mat_ef.y,
    );
    var out: VsOut;
    out.pos = vec4<f32>(page_to_ndc(p), 0.0, 1.0);
    out.uv = mix(in.uv.xy, in.uv.zw, corner);
    out.alpha = in.alpha;
    return out;
}

@fragment
fn fs_composite_clip(in: VsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(grp_tex, grp_smp, in.uv);
    let q = clip_coverage(floor(in.pos.xy));
    return vec4<f32>(tex.rgb * in.alpha * q, tex.a * in.alpha * q);
}
"#;

/// Filter passes (see [`FilterPass`](crate::filter::FilterPass)):
/// fullscreen-triangle post-processing over an offscreen group texture.
/// `mode` selects the pass: 0 = color matrix, 1 = one axis of the
/// separable box blur (the executor runs H then V, three times, like
/// the CPU's [`box_blur_sigma_rgba`](crate::cpu::box_blur_sigma_rgba)).
///
/// The color pass mirrors
/// [`apply_color_matrix_rgba8`](crate::filter::apply_color_matrix_rgba8):
/// unpremultiply to straight f32, apply the 4x5 matrix, clamp to
/// [0, 1], premultiply back. `flags` bit 0 selects the
/// premultiplied-direct variant (the drop-shadow tint matrix runs on
/// the stored premultiplied bytes with no (un)premultiply round-trip).
/// Fully transparent texels stay transparent either way (0/0 is
/// guarded).
///
/// The blur pass samples with the clamp-to-edge sampler, reproducing
/// the CPU's edge clamping; each H/V pass quantizes to u8 with
/// round-to-nearest while the CPU truncates, so blurred pixels agree
/// within ~1 LSB (see `cpu_gpu_filtered_blur_agreement`).
const SHADER_FILTER: &str = r#"
@group(1) @binding(0) var filter_tex: texture_2d<f32>;
@group(1) @binding(1) var filter_smp: sampler;
struct FilterUniform {
    mat_rows: array<vec4<f32>, 4>,
    mat_offset: vec4<f32>,
    texel: vec2<f32>,
    radius: f32,
    dir: u32,
    mode: u32,
    flags: u32,
    _pad: vec2<u32>,
};
@group(2) @binding(0) var<uniform> flt: FilterUniform;

struct FilterVsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_filter(@builtin(vertex_index) vi: u32) -> FilterVsOut {
    // Fullscreen triangle; uv (0, 0) is the texture's top-left texel.
    // `p` overshoots the [0, 1] quad ((0,0),(2,0),(0,2)), so the uv
    // must overshoot in lockstep ((0,1),(2,1),(0,-1)): halving it here
    // would map the viewport onto the texture's bottom-left quadrant,
    // stretched 2x.
    let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: FilterVsOut;
    out.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(p.x, 1.0 - p.y);
    return out;
}

@fragment
fn fs_filter(in: FilterVsOut) -> @location(0) vec4<f32> {
    if (flt.mode == 0u) {
        let tex = textureSample(filter_tex, filter_smp, in.uv);
        var c: vec4<f32>;
        if ((flt.flags & 1u) != 0u) {
            c = vec4<f32>(
                dot(flt.mat_rows[0], tex),
                dot(flt.mat_rows[1], tex),
                dot(flt.mat_rows[2], tex),
                dot(flt.mat_rows[3], tex),
            ) + flt.mat_offset;
        } else {
            let a = tex.a;
            var s = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            if (a > 0.0) {
                s = vec4<f32>(tex.rgb / a, a);
            }
            var o = vec4<f32>(
                dot(flt.mat_rows[0], s),
                dot(flt.mat_rows[1], s),
                dot(flt.mat_rows[2], s),
                dot(flt.mat_rows[3], s),
            ) + flt.mat_offset;
            o = clamp(o, vec4<f32>(0.0), vec4<f32>(1.0));
            c = vec4<f32>(o.rgb * o.a, o.a);
        }
        return clamp(c, vec4<f32>(0.0), vec4<f32>(1.0));
    }
    // mode 1: one axis of the separable box blur. Accumulate a 1D
    // window of 2*radius+1 texels along `flt.dir` (0 = horizontal, 1
    // = vertical) in premultiplied space, mirroring the CPU's
    // `box_blur_sigma_rgba`; the executor runs H then V three times.
    //
    // Taps use `textureLoad` (exact texel fetch), NOT the linear
    // sampler: interpolated UVs and linear weights carry f32 noise
    // (~1e-5), and over up to 513 taps x 6 passes that noise can push
    // a truncated mean across an integer boundary by a few LSBs. The
    // taps are accumulated as integers and divided with integer
    // division, which is bit-exact with the CPU's u32 window sums
    // (`sum / width` truncates); the `t * 255 + 0.5` un-normalizes the
    // exact f32(k/255) texel values back to k. Edge taps clamp
    // manually, reproducing the CPU's clamp-to-edge window (and the
    // `filter_smp` ClampToEdge addressing is irrelevant on this path).
    // Emitting an exact multiple of 1/255 makes the Rgba8Unorm target
    // store it losslessly. (The old code divided the f32 mean and let
    // the target round-to-nearest while the CPU truncates: a 1-LSB
    // premultiplied difference that the readback unpremultiply then
    // amplified into tens of straight-RGB LSBs at tiny alpha.)
    //
    // `radius` is clamped at uniform build time to
    // `GPU_BLUR_RADIUS_CAP` (256; see `filter_blur_uniform` for the
    // deliberate CPU divergence this implies), so the worst case is 6
    // passes x 513 taps per output pixel, and the u32 accumulator tops
    // out at 513 * 255 = 130815. A sliding-window O(1)/pixel port of
    // the CPU's running sum is not possible here: every fragment
    // invocation is independent, so a running sum would need the
    // previous pixel's sum -- this pass's own output -- which is a
    // read/write feedback loop WebGPU forbids.
    if (flt.mode == 1u) {
        let r = i32(flt.radius);
        let dims = textureDimensions(filter_tex);
        // Integer texel of this fragment: uv = ((x+0.5)/w,
        // 1-(y+0.5)/h), so truncating uv*dims recovers (x, y); the
        // +-0.5 texel-center offset vanishes under truncation and the
        // interpolation noise (<< 0.5) cannot change the result.
        let base = vec2<i32>(
            i32(in.uv.x * f32(dims.x)),
            i32((1.0 - in.uv.y) * f32(dims.y)),
        );
        let lo = vec2<i32>(0, 0);
        let hi = vec2<i32>(dims) - vec2<i32>(1, 1);
        let step = select(vec2<i32>(1, 0), vec2<i32>(0, 1), flt.dir == 1u);
        var acc = vec4<u32>(0u, 0u, 0u, 0u);
        for (var i = -r; i <= r; i++) {
            let t = clamp(base + i * step, lo, hi);
            let v = textureLoad(filter_tex, t, 0);
            acc += vec4<u32>(v * 255.0 + 0.5);
        }
        let q = acc / u32(2 * r + 1);
        return vec4<f32>(q) / 255.0;
    }
    return textureSample(filter_tex, filter_smp, in.uv);
}
"#;

// ---------------------------------------------------------------------------
// Lowered frame: instance byte buffers + ordered draw/scissor ops
// ---------------------------------------------------------------------------

/// Bytes per instance for each pipeline.
const SOLID_STRIDE: usize = 32; // rect vec4 + color vec4
const GLYPH_STRIDE: usize = 48; // rect vec4 + uv vec4 + color vec4
/// rect vec4 + uv vec4 + alpha f32 + repeat params vec4 + repeat counts
/// vec2. The repeat fields drive [`DisplayItem::RepeatImage`]'s single
/// tiled quad (see `SHADER_IMAGE`); plain [`DisplayItem::Image`] quads
/// push neutral values (single-copy, no wrap).
const IMAGE_STRIDE: usize = 60;
/// Group-composite instances keep the plain image layout (rect + uv +
/// alpha) plus a `u32` blend-mode discriminant (`0` = normal) plus the
/// 2D transform matrix; they never carry repeat params.
const COMPOSITE_STRIDE: usize = 64;
/// Box-shadow instances: coverage area vec4 + border-box rect vec4 +
/// radii vec4 + color vec4 + params vec4 (offset.x, offset.y, blur,
/// spread) + flags vec4 (inset, 0, 0, 0).
const SHADOW_STRIDE: usize = 96;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PipeKind {
    Solid,
    Glyph,
    Image,
    /// Group compositing: like [`PipeKind::Image`] but the texture holds
    /// premultiplied framebuffer values (see `SHADER_COMPOSITE`).
    Composite,
    /// Box shadows: analytic rounded-box SDF in the fragment shader
    /// (see `SHADER_SHADOW`).
    Shadow,
}

impl PipeKind {
    /// Index into [`WgpuBackend::instance_bufs`].
    fn slot(self) -> usize {
        match self {
            PipeKind::Solid => 0,
            PipeKind::Glyph => 1,
            PipeKind::Image => 2,
            PipeKind::Composite => 3,
            PipeKind::Shadow => 4,
        }
    }

    /// Bytes per instance in the pipe's instance buffer.
    fn stride(self) -> usize {
        match self {
            PipeKind::Solid => SOLID_STRIDE,
            PipeKind::Glyph => GLYPH_STRIDE,
            PipeKind::Image => IMAGE_STRIDE,
            PipeKind::Composite => COMPOSITE_STRIDE,
            PipeKind::Shadow => SHADOW_STRIDE,
        }
    }

    /// Inverse of [`PipeKind::slot`].
    fn from_slot(slot: usize) -> Self {
        match slot {
            0 => PipeKind::Solid,
            1 => PipeKind::Glyph,
            2 => PipeKind::Image,
            3 => PipeKind::Composite,
            _ => PipeKind::Shadow,
        }
    }
}

enum Op {
    /// Set the clip state to `snap` (see [`ClipSnapshot`]): the
    /// executor derives the coarse integer scissor from the entries'
    /// intersected bound and uploads the entries to the group-2 clip
    /// uniform for exact per-pixel coverage. `len == 0` disables
    /// clipping (whole target). Boxed: the snapshot is 260 bytes and
    /// would bloat every `Op`.
    Scissor(Box<ClipSnapshot>),
    /// Draw `count` instances from the pipeline's buffer.
    Draw {
        pipe: PipeKind,
        image: Option<ImageId>,
        /// Atlas generation the batch's quads were rasterized into
        /// (meaningful for [`PipeKind::Glyph`] only; 0 otherwise). A
        /// mid-frame atlas rollover retires the old page, so batches must
        /// not merge across a generation boundary: each batch samples
        /// exactly one page texture.
        generation: u64,
        start: u32,
        count: u32,
    },
    /// Draw `count` image-pipe instances sampling a baked gradient
    /// texture: `tex` keys [`WgpuBackend::gradient_cache`]. The
    /// instances live in the image pipe's buffer (same layout as
    /// [`Op::Draw`] with `pipe = PipeKind::Image`), but the texture is
    /// bound from the gradient cache instead of the image cache, so
    /// gradient batches can never merge with image batches.
    DrawGradient { tex: u64, start: u32, count: u32 },
    /// Composite an isolated opacity group: render
    /// [`Lowered::groups`]`[idx]` into an offscreen texture, then draw it
    /// over the target with the group's opacity.
    Group(usize),
    /// Apply a `backdrop-filter` chain: capture the already-painted
    /// backdrop under `rect` (page px) from the current target, run the
    /// planned filter passes over the capture, and draw it back over
    /// the same region. The ops following this one (the element's own
    /// content) then paint over the filtered backdrop inline. Boxed:
    /// the pass vec would bloat every `Op`.
    BackdropFilter(Box<BackdropFilterOp>),
}

fn push_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_ne_bytes());
}

fn push_rect(buf: &mut Vec<u8>, r: Rect) {
    push_f32(buf, r.x);
    push_f32(buf, r.y);
    push_f32(buf, r.w);
    push_f32(buf, r.h);
}

fn push_color(buf: &mut Vec<u8>, c: Color) {
    // The instance layout is unchanged (four f32 words): the packed
    // bytes expand back to the exact f32 values the old 4×f32 `Color`
    // carried, so the shaders and their numeric behavior are untouched.
    push_f32(buf, c.r());
    push_f32(buf, c.g());
    push_f32(buf, c.b());
    push_f32(buf, c.a());
}

fn intersect(a: Rect, b: Rect) -> Rect {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.w).min(b.x + b.w);
    let y1 = (a.y + a.h).min(b.y + b.h);
    Rect {
        x: x0,
        y: y0,
        w: (x1 - x0).max(0.0),
        h: (y1 - y0).max(0.0),
    }
}

/// One [`ClipUniform`](`SHADER_COMMON`) entry as u32 words:
/// device-px `(x0, y0, x1, y1)` bits, the four corner-radius bits, the
/// entry kind (`0` = rounded-rect, `1` = `clip-path` polygon), the
/// polygon vertex count, two pad words, then 48 `(x, y, 0, 0)` vertex
/// words (device-px bits; unused words are zero). Vertices are `vec4`s
/// because the `uniform` address space requires array strides that are
/// multiples of 16 (see `ClipEntryU`).
const CLIP_ENTRY_WORDS: usize = 8 + 1 + 1 + 2 + MAX_POLY_VERTS * 4;
/// Bytes per clip uniform entry: `vec4 rect + vec4 radii + u32 kind +
/// u32 poly_len + vec2 pad + array<vec4<f32>, 48> poly` (matches
/// `ClipEntryU` in `SHADER_COMMON`).
const CLIP_ENTRY_BYTES: usize = 816;
/// Group-2 clip uniform size: u32 count + u32 `has_rounded` + u32
/// `has_poly` + 4 pad bytes, then [`MAX_CLIP_ENTRIES`] entries.
const CLIP_UNIFORM_BYTES: u64 = 16 + MAX_CLIP_ENTRIES as u64 * CLIP_ENTRY_BYTES as u64;

/// Group-2 filter uniform size: 4x`vec4` matrix rows + `vec4` offset +
/// `vec2` texel + f32 radius + u32 dir + u32 mode + u32 flags + `vec2`
/// pad (matches `FilterUniform` in `SHADER_FILTER`).
const FILTER_UNIFORM_BYTES: usize = 112;

/// Group-2 clip bind group cache key: entry count plus the device-px
/// bits of up to [`MAX_CLIP_ENTRIES`] entries, oldest first. `None`
/// disables clipping.
///
/// Radii note: the page-to-device mapping scales x and y by separate
/// factors (`sx`, `sy`), so a page-px circle is in general an ellipse
/// on the target. The clip model keeps one radius per corner, scaled
/// by `sx` (the x-axis factor); when `sx == sy` — every real render
/// and test — this is exact, and the CPU backend only supports a
/// uniform scale anyway.
type ClipKey = Option<(u8, [[u32; CLIP_ENTRY_WORDS]; MAX_CLIP_ENTRIES])>;

/// Intermediate lowering state; one frame's worth of GPU instance data.
struct Lowered {
    solid: Vec<u8>,
    glyph: Vec<u8>,
    image: Vec<u8>,
    shadow: Vec<u8>,
    ops: Vec<Op>,
    /// Last clip snapshot pushed to `ops` (starts as the disabled
    /// snapshot, which is also the device's initial state).
    /// PushClip/PopClip/PushRoundedClip skip emitting `Op::Scissor`
    /// when the snapshot is unchanged, so redundant clip pushes (e.g.
    /// nested clips of the same rect) cost nothing and the draws
    /// around them stay mergeable into one batch.
    scissor: Option<ClipSnapshot>,
    /// Isolated opacity groups, referenced by [`Op::Group`] index. Each
    /// group is a separately lowered item range, rendered into its own
    /// offscreen texture at paint time.
    groups: Vec<OpacityGroup>,
}

/// One isolated group: the item range lowered on its own so it can
/// render into an offscreen texture, then composite with `opacity`
/// and/or blend with `blend` against the backdrop. Opacity groups
/// (`PushOpacity`) set `opacity` and leave `blend` at `Normal`;
/// blend groups (`PushBlend`) set `blend` and leave `opacity` at 1.0;
/// transform groups (`PushTransform`) set `transform` and leave both
/// at their no-op values.
struct OpacityGroup {
    lowered: Lowered,
    opacity: f32,
    /// `mix-blend-mode` for the composite: `Normal` takes the plain
    /// premultiplied source-over composite; any other mode takes the
    /// blend composite path (backdrop copy + blend shader).
    blend: BlendMode,
    /// Exact paint bounds of `lowered` in page px (see
    /// [`WgpuBackend::lowered_bounds`]): the offscreen texture is sized to
    /// these bounds snapped outward to device px instead of the full
    /// target, and the composite quad covers exactly that snapped rect.
    bounds: Rect,
    /// 2D transform for the composite, in page px `(a, b, c, d, e, f)`
    /// (see [`DisplayItem::PushTransform`]): the group's texture holds
    /// the untransformed rendering and the composite quad's corners are
    /// mapped through this matrix. `None` draws the quad untransformed.
    /// Ancestor clips were drained when the group lowered (it lowers
    /// with empty clips), so the composite applies the outer clip state.
    transform: Option<[f32; 6]>,
    /// Planned [`FilterPass`]es for a [`DisplayItem::PushFilter`]
    /// group, applied to the group texture after
    /// [`render_group_texture`](WgpuBackend::render_group_texture) and
    /// before the composite. Empty for opacity/blend/transform groups.
    filter: Vec<FilterPass>,
}

/// One [`Op::BackdropFilter`]: the planned passes plus the page-px
/// filter region (the element's border box).
struct BackdropFilterOp {
    passes: Vec<FilterPass>,
    rect: Rect,
}

/// One frame's prepared view of a single [`Lowered`] in the opacity-group
/// tree: the per-pipe base instance indices where this lowering's bytes
/// land in the frame's combined instance buffers (see
/// [`WgpuBackend::prepare_lowering`]). Draws address
/// `base + op.start .. base + op.start + op.count`, so every lowering —
/// root and nested groups alike — draws from its own disjoint slice of
/// the shared buffers. Mirrors [`Lowered::groups`] one level at a time;
/// [`WgpuBackend::execute_ops_into`] recurses through it in lockstep
/// with the lowering tree.
struct Prepared {
    /// Base instance index per [`PipeKind`] slot into the frame's
    /// combined per-pipe buffer.
    bases: [u32; 5],
    /// One per group, in [`Lowered::groups`] order.
    groups: Vec<Prepared>,
}

/// One pooled offscreen target for isolated opacity groups: the size it
/// was created at plus the texture. `wgpu::Texture` is `Arc`-backed, so
/// checkout clones are cheap.
struct GroupTarget {
    w: u32,
    h: u32,
    texture: wgpu::Texture,
}

impl Lowered {
    /// `items` is the display-list item count. Quads-per-item estimates:
    /// solids average ~2 (borders and decorations tessellate to several
    /// quads), glyphs ~1 (text runs are a minority of items but emit one
    /// quad per char), images ~1 every other item at most. Over-reserving
    /// is cheap (transient per-frame Vecs); the goal is zero reallocs on
    /// typical pages.
    fn with_capacity(items: usize) -> Self {
        Self {
            solid: Vec::with_capacity(items * 2 * SOLID_STRIDE),
            glyph: Vec::with_capacity(items * GLYPH_STRIDE),
            image: Vec::with_capacity(items * IMAGE_STRIDE / 2),
            shadow: Vec::with_capacity(items * SHADOW_STRIDE / 4),
            ops: Vec::with_capacity(items * 2),
            scissor: Some(ClipSnapshot::none()),
            groups: Vec::new(),
        }
    }

    /// Append an instance to `pipe`'s buffer, merging with the open batch
    /// when the pipeline (and image) matches and the instance is
    /// contiguous with it.
    fn emit(&mut self, pipe: PipeKind, image: Option<ImageId>, generation: u64) {
        let len = match pipe {
            PipeKind::Solid => self.solid.len(),
            PipeKind::Glyph => self.glyph.len(),
            PipeKind::Image => self.image.len(),
            PipeKind::Shadow => self.shadow.len(),
            // Composite instances are built at execute time (see `Op::Group`
            // handling in `execute_ops_into`); they never flow through the
            // draw batcher.
            PipeKind::Composite => unreachable!("no Draw op uses the Composite pipe"),
        };
        debug_assert_eq!(len % pipe.stride(), 0);
        let start = (len / pipe.stride()) as u32;
        let merged = match self.ops.last_mut() {
            Some(Op::Draw {
                pipe: p,
                image: im,
                generation: g,
                start: s,
                count,
            }) if *p == pipe && *im == image && *g == generation && *s + *count == start => {
                *count += 1;
                true
            }
            _ => false,
        };
        if !merged {
            self.ops.push(Op::Draw {
                pipe,
                image,
                generation,
                start,
                count: 1,
            });
        }
    }

    /// Push a scissor op only when the clip snapshot actually changes.
    /// Keeps draw batches mergeable across redundant clip push/pop
    /// pairs.
    fn set_scissor(&mut self, stack: &[PageClip]) {
        let snap = snapshot_clips(stack);
        if self.scissor != Some(snap) {
            self.ops.push(Op::Scissor(Box::new(snap)));
            self.scissor = Some(snap);
        }
    }

    fn solid_quad(&mut self, rect: Rect, color: Color) {
        if !rect_is_finite(rect) || rect.w <= 0.0 || rect.h <= 0.0 || color.a() <= 0.0 {
            return;
        }
        // Emit before pushing: `emit` derives the instance index from the
        // pre-push buffer length.
        self.emit(PipeKind::Solid, None, 0);
        push_rect(&mut self.solid, rect);
        push_color(&mut self.solid, color);
    }

    fn glyph_quad(&mut self, rect: Rect, uv: [f32; 4], color: Color, generation: u64) {
        if !rect_is_finite(rect) || rect.w <= 0.0 || rect.h <= 0.0 || color.a() <= 0.0 {
            return;
        }
        // Emit before pushing: `emit` derives the instance index from the
        // pre-push buffer length. `generation` is the atlas page the cell
        // was rasterized into: batches never merge across a rollover, so
        // each draw samples exactly one page texture.
        self.emit(PipeKind::Glyph, None, generation);
        push_rect(&mut self.glyph, rect);
        for v in uv {
            push_f32(&mut self.glyph, v);
        }
        push_color(&mut self.glyph, color);
    }

    fn image_quad(&mut self, rect: Rect, image: ImageId, alpha: f32) {
        if !rect_is_finite(rect) || rect.w <= 0.0 || rect.h <= 0.0 || alpha <= 0.0 {
            return;
        }
        // Emit before pushing: `emit` derives the instance index from the
        // pre-push buffer length.
        self.emit(PipeKind::Image, Some(image), 0);
        push_rect(&mut self.image, rect);
        for v in [0.0f32, 0.0, 1.0, 1.0] {
            push_f32(&mut self.image, v);
        }
        push_f32(&mut self.image, alpha);
        // Neutral repeat params: single copy, no wrap, no spacing.
        for v in [0.0f32, 0.0, 0.0, 0.0] {
            push_f32(&mut self.image, v);
        }
        for v in [0.0f32, 0.0] {
            push_f32(&mut self.image, v);
        }
    }

    /// One [`DisplayItem::Gradient`] quad: image-pipe instance bytes
    /// identical to [`Lowered::image_quad`] (full-texture UVs, neutral
    /// repeat), but the draw op binds `tex` from the gradient texture
    /// cache instead of an [`ImageId`] from the image cache. Merges
    /// with the open batch only when the texture matches and the
    /// instance is contiguous with it (same rule as [`Lowered::emit`]).
    fn gradient_quad(&mut self, rect: Rect, tex: u64, alpha: f32) {
        if !rect_is_finite(rect) || rect.w <= 0.0 || rect.h <= 0.0 || alpha <= 0.0 {
            return;
        }
        // Emit before pushing: the instance index derives from the
        // pre-push buffer length (see `image_quad`).
        let len = self.image.len();
        debug_assert_eq!(len % IMAGE_STRIDE, 0);
        let start = (len / IMAGE_STRIDE) as u32;
        let merged = match self.ops.last_mut() {
            Some(Op::DrawGradient {
                tex: t,
                start: s,
                count,
            }) if *t == tex && *s + *count == start => {
                *count += 1;
                true
            }
            _ => false,
        };
        if !merged {
            self.ops.push(Op::DrawGradient {
                tex,
                start,
                count: 1,
            });
        }
        push_rect(&mut self.image, rect);
        for v in [0.0f32, 0.0, 1.0, 1.0] {
            push_f32(&mut self.image, v);
        }
        push_f32(&mut self.image, alpha);
        // Neutral repeat params: single copy, no wrap, no spacing.
        for v in [0.0f32, 0.0, 0.0, 0.0] {
            push_f32(&mut self.image, v);
        }
        for v in [0.0f32, 0.0] {
            push_f32(&mut self.image, v);
        }
    }

    /// One [`DisplayItem::BoxShadow`] quad: `area` is the conservative
    /// coverage rect (page px) the vertex shader draws; the fragment
    /// shader evaluates the shadow SDF against `rect`/`radii`.
    #[allow(clippy::too_many_arguments)]
    fn shadow_quad(
        &mut self,
        area: Rect,
        rect: Rect,
        radii: [f32; 4],
        color: Color,
        offset: (f32, f32),
        blur: f32,
        spread: f32,
        inset: bool,
    ) {
        if !rect_is_finite(area)
            || area.w <= 0.0
            || area.h <= 0.0
            || !rect_is_finite(rect)
            || color.a() <= 0.0
        {
            return;
        }
        // Emit before pushing: `emit` derives the instance index from the
        // pre-push buffer length.
        self.emit(PipeKind::Shadow, None, 0);
        push_rect(&mut self.shadow, area);
        push_rect(&mut self.shadow, rect);
        for v in radii {
            push_f32(&mut self.shadow, v);
        }
        push_color(&mut self.shadow, color);
        for v in [offset.0, offset.1, blur.max(0.0), spread] {
            push_f32(&mut self.shadow, v);
        }
        push_f32(&mut self.shadow, if inset { 1.0 } else { 0.0 });
        for _ in 0..3 {
            push_f32(&mut self.shadow, 0.0);
        }
    }

    /// One tiled quad for [`DisplayItem::RepeatImage`]: `uv` is
    /// (u0, v0, u1, v1) in tile units across `rect`, `modes` is
    /// (mode_x, mode_y) with 0 = single copy, 1 = wrap, 2 = spaced (see
    /// `SHADER_IMAGE`), `pitch` the spaced pitch per axis in tile units,
    /// and `counts` the spaced tile counts per axis — except in wrap
    /// mode, where the WGSL `repeat_axis` wrap branch dual-purposes it
    /// as the per-axis tile-boundary snap band in tile units
    /// (`snap(span) = min(4 * EPSILON * max(|span|, 1), 0.25)`, computed
    /// in `lower_repeat_image`).
    #[allow(clippy::too_many_arguments)]
    fn image_repeat_quad(
        &mut self,
        rect: Rect,
        uv: [f32; 4],
        modes: (f32, f32),
        pitch: (f32, f32),
        counts: (f32, f32),
        image: ImageId,
        alpha: f32,
    ) {
        if !rect_is_finite(rect) || rect.w <= 0.0 || rect.h <= 0.0 || alpha <= 0.0 {
            return;
        }
        // Emit before pushing: `emit` derives the instance index from the
        // pre-push buffer length.
        self.emit(PipeKind::Image, Some(image), 0);
        push_rect(&mut self.image, rect);
        for v in uv {
            push_f32(&mut self.image, v);
        }
        push_f32(&mut self.image, alpha);
        for v in [modes.0, modes.1, pitch.0, pitch.1] {
            push_f32(&mut self.image, v);
        }
        for v in [counts.0, counts.1] {
            push_f32(&mut self.image, v);
        }
    }
}

/// Union of every instance rect in a lowering, in page px: the exact
/// paint bounds of the lowered content. Every pipe's instance starts
/// with the rect as four little-endian `f32`s (see the `*_quad`
/// emitters above); glyph quads are ink-exact (see
/// [`emit_shaped_text`]), so no padding is needed. Nested isolated
/// groups composite into this lowering over their snapped device bounds
/// (see `Op::Group` handling in [`WgpuBackend::execute_ops_into`]), so
/// their page-px [`OpacityGroup::bounds`] union in too —
/// without this, a parent whose direct content is small but whose nested
/// group extends further would size its offscreen texture too small and
/// clip the nested composite. Returns `None` when the lowering emitted
/// no instances.
///
/// Isolated opacity groups size their offscreen textures to these bounds
/// instead of the full target (see [`WgpuBackend::render_group_texture`]).
fn lowered_bounds(lowered: &Lowered) -> Option<Rect> {
    fn union_into(acc: &mut Option<Rect>, r: Rect) {
        *acc = Some(match *acc {
            Some(a) => Rect {
                x: a.x.min(r.x),
                y: a.y.min(r.y),
                w: (a.x + a.w).max(r.x + r.w) - a.x.min(r.x),
                h: (a.y + a.h).max(r.y + r.h) - a.y.min(r.y),
            },
            None => r,
        });
    }
    fn union_pipe(bytes: &[u8], stride: usize, acc: &mut Option<Rect>) {
        let mut off = 0;
        while off + 16 <= bytes.len() {
            let r = Rect {
                x: f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]),
                y: f32::from_le_bytes([
                    bytes[off + 4],
                    bytes[off + 5],
                    bytes[off + 6],
                    bytes[off + 7],
                ]),
                w: f32::from_le_bytes([
                    bytes[off + 8],
                    bytes[off + 9],
                    bytes[off + 10],
                    bytes[off + 11],
                ]),
                h: f32::from_le_bytes([
                    bytes[off + 12],
                    bytes[off + 13],
                    bytes[off + 14],
                    bytes[off + 15],
                ]),
            };
            // Defensive: the emitters only push finite, non-degenerate
            // rects, so this never triggers on real lowerings.
            if rect_is_finite(r) && r.w > 0.0 && r.h > 0.0 {
                union_into(acc, r);
            }
            off += stride;
        }
    }
    let mut acc = None;
    union_pipe(&lowered.solid, PipeKind::Solid.stride(), &mut acc);
    union_pipe(&lowered.glyph, PipeKind::Glyph.stride(), &mut acc);
    union_pipe(&lowered.image, PipeKind::Image.stride(), &mut acc);
    union_pipe(&lowered.shadow, PipeKind::Shadow.stride(), &mut acc);
    // Nested groups: their composited output covers `group.bounds` in
    // this lowering's page-px space (the nested bounds were computed the
    // same way, recursively, when the group lowered) — transformed by
    // the group's matrix when it carries one, since the composite quad
    // draws the texture through the transform.
    for group in &lowered.groups {
        let b = match group.transform {
            Some(m) => transform_rect_bounds(m, group.bounds),
            None => group.bounds,
        };
        if rect_is_finite(b) && b.w > 0.0 && b.h > 0.0 {
            union_into(&mut acc, b);
        }
    }
    acc
}

/// `false` for rects with NaN or infinite components.
///
/// The quad emitters' `w <= 0.0` early-out does not catch NaN (`NaN <= 0.0`
/// is false), so without this guard a NaN rect would emit garbage vertices
/// into the instance buffers and poison the draw.
fn rect_is_finite(rect: Rect) -> bool {
    rect.x.is_finite() && rect.y.is_finite() && rect.w.is_finite() && rect.h.is_finite()
}

/// Thread-local font-bytes cache for [`emit_shaped_text`].
///
/// The hook clones the whole font byte buffer on every call, so bytes are
/// resolved once per [`FontKey`] and shared across text runs through an
/// `Rc` (one refcount bump per run instead of a multi-hundred-KB memcpy).
/// Thread-local: each backend thread resolves independently.
///
/// `FontKey`s are minted sequentially per [`FontCache`](crate::fonts::FontCache)
/// instance, so the same key can name different faces in different caches.
/// Entries are therefore validated against the installing hook's identity:
/// when the hook changes, every entry is flushed before resolving. The
/// installing hook is retained so its allocation cannot be recycled for a
/// different closure (which would defeat the `Arc::ptr_eq` identity check).
/// Entries are bounded by distinct `FontKey`s (one key per resolved face),
/// so the cache cannot grow without bound.
///
/// Reentrancy: the hook is caller-supplied and may re-enter text lowering
/// on this thread, so callers use the two-phase [`get`](Self::get)/
/// [`insert`](Self::insert) (hook runs outside the `RefCell` borrow);
/// never invoke the hook while the borrow is held.
struct FontDataCache {
    hook: Option<std::sync::Arc<dyn Fn(FontKey) -> Option<FontData> + Send + Sync>>,
    entries: HashMap<FontKey, std::rc::Rc<FontData>>,
}

impl FontDataCache {
    fn new() -> Self {
        Self {
            hook: None,
            entries: HashMap::new(),
        }
    }

    /// Install `hook`, flushing all entries when it changed. Factored out
    /// so [`get`](Self::get) and [`insert`](Self::insert) validate the
    /// hook identity identically.
    fn install(&mut self, hook: &Hook) {
        if !self
            .hook
            .as_ref()
            .is_some_and(|h| std::sync::Arc::ptr_eq(h, hook))
        {
            self.entries.clear();
            self.hook = Some(std::sync::Arc::clone(hook));
        }
    }

    /// Cached entry for `key` without invoking the hook. Never calls out,
    /// so it is safe to hold the cache borrow across it.
    fn get(&mut self, hook: &Hook, key: FontKey) -> Option<std::rc::Rc<FontData>> {
        self.install(hook);
        self.entries.get(&key).map(std::rc::Rc::clone)
    }

    /// Store hook-resolved `data` for `key`, re-validating the hook
    /// identity first: the hook ran outside the cache borrow and may have
    /// re-entered and installed a different hook, so a blind insert could
    /// file another hook's bytes under this one.
    fn insert(&mut self, hook: &Hook, key: FontKey, data: FontData) -> std::rc::Rc<FontData> {
        self.install(hook);
        let shared = std::rc::Rc::new(data);
        self.entries.insert(key, std::rc::Rc::clone(&shared));
        shared
    }
}

/// The font-data hook type shared by [`FontDataCache`] and
/// [`emit_shaped_text`].
type Hook = std::sync::Arc<dyn Fn(FontKey) -> Option<FontData> + Send + Sync>;

/// Text shaping for the GPU path: runs go through the shared
/// [`shape_text`](crate::text::shape_text) (kerning, ligatures, complex
/// scripts) — the same run the CPU backend paints, so both backends'
/// pixels agree. Each shaped glyph looks up its [`GlyphCell`] in the
/// atlas; the ink's real UV subrect and bearings place the quad, and the
/// pen advances by the shaped advance.
///
/// Glyphs paint nothing when no font data resolves for `font_key` (or a
/// glyph has no coverage), exactly like the CPU backend.
#[allow(clippy::too_many_arguments)]
fn emit_shaped_text(
    out: &mut Lowered,
    atlas: &mut Atlas,
    queue: &wgpu::Queue,
    get_font_data: &Option<Arc<dyn Fn(FontKey) -> Option<FontData> + Send + Sync>>,
    text: &str,
    x0: f32,
    baseline: f32,
    font_key: FontKey,
    size_px: f32,
    color: Color,
    atlas_generation: &mut u64,
    scale: (f32, f32),
) {
    let Some(hook) = get_font_data else { return };
    // Defensive: `lower` sanitizes, but a non-positive scale here would
    // divide placement by zero below.
    let sanitize = |s: f32| {
        if s.is_finite() && s > 0.0 { s } else { 1.0 }
    };
    let (sx, sy) = (sanitize(scale.0), sanitize(scale.1));
    // M5 font-data sharing: resolve once per FontKey and share the
    // bytes across text runs (see [`FontDataCache`]): the installing
    // hook's identity guards against one cache's keys resolving to
    // another cache's bytes.
    //
    // Two-phase lookup for reentrancy: the hook is caller-supplied and
    // runs OUTSIDE the cache borrow. A hook that re-enters text lowering
    // on this thread would otherwise re-borrow the `RefCell` and panic.
    // `insert` re-validates the hook identity in case the re-entrant call
    // installed a different hook.
    thread_local! {
        static FONT_DATA_CACHE: std::cell::RefCell<FontDataCache> =
            std::cell::RefCell::new(FontDataCache::new());
    }
    let cached = FONT_DATA_CACHE.with(|cache| cache.borrow_mut().get(hook, font_key));
    let Some(font_data) = cached.or_else(|| {
        let data = hook(font_key)?;
        Some(FONT_DATA_CACHE.with(|cache| cache.borrow_mut().insert(hook, font_key, data)))
    }) else {
        return;
    };
    // Shape and rasterize at device px so glyphs stay crisp when the
    // target is denser than the page grid (DPR > 1) — the CPU backend
    // shapes at `rounded_size(size_px * scale)` too, so advances agree
    // across backends (previously this backend shaped at the raw page-px
    // size, so its ink was upscaled and blurry on hidpi targets). The
    // atlas keys by `rounded_size` as well, so shaping and ink stay
    // consistent. Placement below stays in page px — the vertex shader
    // maps page px onto device px per axis — so device-px metrics convert
    // back with the raster scale.
    //
    // Non-uniform target/viewport aspect: rasterize at the denser axis.
    // The shaped metrics are device px at that raster scale, so dividing
    // by it (rather than by either axis scale) puts placement back on
    // the page-px grid exactly, and the vertex shader then stretches
    // each axis to its own device scale. Rasterizing at the horizontal
    // scale alone (the old uniform-DPR assumption) resampled the
    // vertical axis from too-coarse ink whenever the vertical device
    // scale exceeded the horizontal one.
    let raster_scale = sx.max(sy);
    let raster_size = f32::from(rounded_size(size_px * raster_scale));
    let run = shape_text(&font_data, text, raster_size);
    let to_page = 1.0 / raster_scale;
    let n = ATLAS_SIZE as f32;
    let mut pen_x = x0;
    for shaped in &run.glyphs {
        let Some(cell) =
            atlas.get_or_rasterize(queue, font_key, &font_data, shaped.glyph_id, raster_size)
        else {
            // No coverage (space, missing glyph): the pen still moves,
            // otherwise words would collapse onto each other.
            pen_x += shaped.advance * to_page;
            continue;
        };
        // The cell's owning generation: `get_or_rasterize` may have
        // rolled the page over just above, installing a fresh texture
        // that the glyph bind group must follow (see
        // `rebuild_glyph_bind_group`).
        *atlas_generation = (*atlas_generation).max(cell.generation);
        // Ink UV subrect: the ink sits zero-padded at the cell's
        // top-left with its real (ink_w, ink_h) size.
        let (cx, cy, _, _) = cell.cell;
        let uv = [
            f32::from(cx) / n,
            f32::from(cy) / n,
            f32::from(cx + cell.ink_w) / n,
            f32::from(cy + cell.ink_h) / n,
        ];
        // Placement mirrors the CPU backend: the shaped offset plus the
        // glyph's own bearings; the ink's top edge sits `top` px above
        // the baseline. Bearings and advances are device-px (shaped at
        // `raster_size`); `to_page` puts the quad on the page-px grid.
        out.glyph_quad(
            Rect {
                x: pen_x + (shaped.x_offset + cell.left as f32) * to_page,
                y: baseline + (shaped.y_offset - cell.top as f32) * to_page,
                w: f32::from(cell.ink_w) * to_page,
                h: f32::from(cell.ink_h) * to_page,
            },
            uv,
            color,
            cell.generation,
        );
        pen_x += shaped.advance * to_page;
    }
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

/// The render pipelines for one output format.
#[derive(Clone)]
struct Pipelines {
    solid: wgpu::RenderPipeline,
    glyph: wgpu::RenderPipeline,
    image: wgpu::RenderPipeline,
    /// Group compositing: premultiplied blending of an offscreen group
    /// texture with a per-instance group opacity.
    composite: wgpu::RenderPipeline,
    /// Blend-mode group compositing: the fragment shader blends the
    /// offscreen group texture against a copied backdrop with the
    /// per-instance blend mode; blending is disabled (REPLACE) since
    /// the shader outputs the final premultiplied value.
    composite_blend: wgpu::RenderPipeline,
    /// Box shadows: analytic rounded-box SDF in the fragment shader
    /// (see `SHADER_SHADOW`); straight-alpha output like `solid`.
    shadow: wgpu::RenderPipeline,
    /// Filter passes: fullscreen-triangle post-processing over an
    /// offscreen group texture (see `SHADER_FILTER`); blending disabled
    /// (REPLACE), the shader outputs the final premultiplied value.
    filter: wgpu::RenderPipeline,
    /// Backdrop-filter write-back: like `composite`, but the fragment
    /// multiplies by the exact fractional clip coverage (see
    /// `SHADER_COMPOSITE_CLIP`).
    composite_clip: wgpu::RenderPipeline,
}

// 9 params, all used; bundling would only add indirection.
#[allow(clippy::too_many_arguments)]
fn make_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    vs_entry: &str,
    fs_entry: &str,
    instance_layout: wgpu::VertexBufferLayout<'_>,
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
    label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vs_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(instance_layout)],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fs_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// Cached GPU upload of one decoded image.
struct ImageEntry {
    /// Kept alive for the texture/view lifetime.
    _texture: wgpu::Texture,
    _view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    /// FNV-1a hash of the decoded RGBA bytes plus the dimensions: a
    /// cache hit requires all three to match, so an [`ImageId`] reused
    /// with different bytes (ids are only unique within one
    /// [`DisplayList`], and lists are routinely rebuilt at reused
    /// addresses) can never serve a stale texture.
    hash: u64,
    width: u32,
    height: u32,
    /// VRAM bytes pinned by this entry's texture (`width * height * 4`,
    /// Rgba8Unorm, one mip level); counted in
    /// [`ImageCache::total_bytes`].
    bytes: usize,
    /// LRU recency: [`ImageCache::clock`] value at the last reference
    /// (cache hit or upload). Frame-boundary eviction drops the smallest
    /// values first.
    last_used: u64,
    /// Set when the entry is referenced during the current frame's
    /// lowering ([`ensure_image`], on a hit or a fresh upload). Such an
    /// entry is already in the frame's instance stream and must survive
    /// until the frame submits, so frame-boundary eviction prefers
    /// entries with this unset and mid-frame eviction never happens at
    /// all. Cleared for every entry at the start of
    /// [`WgpuBackend::lower`].
    lowered_this_frame: bool,
}

/// Cached GPU upload of one baked gradient texture.
///
/// Gradients are vector data: the backend bakes each distinct
/// [`DisplayItem::Gradient`] to a small RGBA texture at device
/// resolution (see [`bake_gradient_rgba8`]) and draws it as one
/// stretched image-pipe quad, so no gradient shader pipeline is
/// needed. The cache key is the gradient's [`Gradient::content_hash`]
/// mixed with the bake dimensions: identical gradients at identical
/// sizes share one texture across frames, and the frame-based
/// `last_frame` eviction (see [`WgpuBackend::lower`]) bounds the cache
/// at the current frame's distinct gradients.
struct GradientEntry {
    /// Kept alive for the texture/view lifetime.
    _texture: wgpu::Texture,
    _view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    /// [`WgpuBackend::gradient_frame`] value at the last lowering that
    /// referenced this entry; entries from older frames are evicted at
    /// the end of [`WgpuBackend::lower`].
    last_frame: u64,
}

/// Max bake dimension (device px, per axis) for one gradient texture,
/// mirroring the CPU backend's `MAX_GRADIENT_BAKE_DIM`: a hostile
/// page-px rect must not allocate a gigantic texture.
const MAX_GRADIENT_BAKE_DIM: u32 = 4096;

/// Upper bound on live image textures per backend. Backends are
/// per-window and long-lived; without a cap, one texture per
/// Pooled group targets kept across frames: enough to cover the common
/// case (a handful of opacity groups) without a one-frame spike pinning
/// dozens of textures forever. See `render_into`.
const MAX_POOLED_GROUP_TARGETS: usize = 8;

/// Byte budget for VRAM pinned by the image upload cache (see
/// [`ImageCache`]). Entry-count alone does not bound memory (one entry
/// can be a multi-megapixel texture), and the frame-start retain only
/// bounds the cache by the current frame's distinct referenced images —
/// a frame with many large distinct images would otherwise upload
/// unbounded VRAM. 256 MiB mirrors the PNG/JPEG post-expansion decode
/// caps ([`crate::images`]): a single full-size decoded image fits, and
/// the steady-state working set of real pages is far smaller.
///
/// Enforcement is two-tier: [`ImageCache::reserve`] runs before every
/// upload during lowering and evicts least-recently-used entries the
/// in-progress frame has not referenced (those are not in the instance
/// stream, so dropping them mid-frame is safe), and
/// [`WgpuBackend::render_into`] enforces the budget again after the
/// frame submits. A single frame's pinned working set can still exceed
/// the budget — entries already referenced by lowered instance streams
/// are never evicted mid-frame; correctness of the in-flight frame
/// always wins over the budget, and the excess is evicted at the next
/// frame boundary, LRU-first.
const IMAGE_CACHE_BUDGET_BYTES: usize = 256 * 1024 * 1024;

/// Scale reference for the image-cache regression tests below (a frame
/// with more distinct images than this exercises the no-mid-frame-eviction
/// path). The live bound is precise, not a count: [`WgpuBackend::lower`]
/// retains exactly the entries the incoming frame references and evicts
/// everything else at the frame boundary, so the cache can never hold
/// more than the current frame's distinct referenced images and
/// cross-frame accumulation is impossible.
#[cfg(test)]
const MAX_CACHED_IMAGES: usize = 512;

/// Per-backend image upload cache: GPU textures by [`ImageId`].
/// Content identity comes from [`DisplayList::image_content_hash`] (the
/// registration-cached [`DecodedImage::content_hash`]); the id alone is
/// not enough since ids are per-list and lists are rebuilt (often at
/// reused addresses) every frame.
///
/// Byte-aware: [`total_bytes`](Self::total_bytes) tracks the VRAM pinned
/// by all entries, [`reserve`](Self::reserve) enforces
/// [`budget_bytes`](Self::budget_bytes) before each upload during
/// lowering (LRU-first, never evicting entries the in-progress frame
/// referenced), and [`evict_for_budget`](Self::evict_for_budget) trims
/// the excess after the frame submits.
struct ImageCache {
    entries: HashMap<ImageId, ImageEntry>,
    /// Sum of [`ImageEntry::bytes`] over `entries`: the VRAM bytes
    /// currently pinned by cached image textures.
    total_bytes: usize,
    /// Monotonic recency sequence; bumped on every cache reference and
    /// stored into [`ImageEntry::last_used`].
    clock: u64,
    /// Byte budget enforced by [`reserve`](Self::reserve) and
    /// [`evict_for_budget`](Self::evict_for_budget).
    /// [`IMAGE_CACHE_BUDGET_BYTES`] in production; the regression tests
    /// below set a small budget to exercise the real enforcement path.
    budget_bytes: usize,
}

impl Default for ImageCache {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            total_bytes: 0,
            clock: 0,
            budget_bytes: IMAGE_CACHE_BUDGET_BYTES,
        }
    }
}

impl ImageCache {
    /// Touch an entry: refresh its LRU recency and mark it referenced by
    /// the current frame's lowering (see
    /// [`ImageEntry::lowered_this_frame`]).
    fn touch(&mut self, id: &ImageId) {
        let clock = self.clock;
        self.clock = clock.wrapping_add(1);
        if let Some(entry) = self.entries.get_mut(id) {
            entry.last_used = clock;
            entry.lowered_this_frame = true;
        }
    }

    /// Insert `entry` under `id`, replacing any previous entry and
    /// keeping [`total_bytes`](Self::total_bytes) exact.
    fn insert(&mut self, id: ImageId, entry: ImageEntry) {
        let bytes = entry.bytes;
        if let Some(old) = self.entries.insert(id, entry) {
            self.total_bytes = self.total_bytes.saturating_sub(old.bytes);
        }
        self.total_bytes = self.total_bytes.saturating_add(bytes);
    }

    /// Remove the entry for `id`, keeping
    /// [`total_bytes`](Self::total_bytes) exact.
    fn remove(&mut self, id: &ImageId) {
        if let Some(old) = self.entries.remove(id) {
            self.total_bytes = self.total_bytes.saturating_sub(old.bytes);
        }
    }

    /// Clear [`ImageEntry::lowered_this_frame`] on every entry: a new
    /// frame's lowering has not referenced anything yet.
    fn clear_frame_flags(&mut self) {
        for entry in self.entries.values_mut() {
            entry.lowered_this_frame = false;
        }
    }

    /// Evict entries for ids the incoming frame does not reference
    /// (stale: ids are only unique within one [`DisplayList`]), keeping
    /// [`total_bytes`](Self::total_bytes) exact. Hot entries for
    /// still-visible images survive (no re-upload churn).
    fn retain_referenced(&mut self, referenced: &HashSet<ImageId>) {
        let stale: Vec<ImageId> = self
            .entries
            .keys()
            .filter(|id| !referenced.contains(id))
            .copied()
            .collect();
        for id in stale {
            self.remove(&id);
        }
    }

    /// Make room for an upload of `bytes` without evicting anything the
    /// current frame's lowering has referenced. Evicts
    /// least-recently-used entries with
    /// [`ImageEntry::lowered_this_frame`] unset first: those are not in
    /// the frame's instance stream (no lowered draw can sample them), so
    /// dropping them mid-frame is safe — the next reference re-uploads on
    /// demand. Stops when the upload fits or no unpinned entry remains:
    /// an upload larger than the whole budget, or a frame whose every
    /// entry is already referenced, still uploads — correctness of the
    /// in-flight frame always wins over the budget, and the post-submit
    /// [`ImageCache::evict_for_budget`] pass trims the excess. Called by
    /// [`ensure_image`] before every texture upload, so the cache is
    /// byte-bounded *during* lowering, not just at frame boundaries.
    fn reserve(&mut self, bytes: usize) {
        while self.total_bytes.saturating_add(bytes) > self.budget_bytes {
            let victim = self
                .entries
                .iter()
                .filter(|(_, e)| !e.lowered_this_frame)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(id, _)| *id);
            let Some(id) = victim else { break };
            self.remove(&id);
        }
    }

    /// Enforce the byte budget after the frame submits, when no pending
    /// draw can reference any entry. Evicts least-recently-used entries
    /// first, preferring entries the frame that just executed did not
    /// reference ([`ImageEntry::lowered_this_frame`] unset): those are
    /// cold — e.g. survivors of an opacity-0-skipped group — and evicting
    /// them costs no re-upload. When only hot entries remain, LRU among
    /// them still applies, so a steady over-budget working set rotates
    /// fairly instead of starving one image. The budget is
    /// [`budget_bytes`](Self::budget_bytes):
    /// [`IMAGE_CACHE_BUDGET_BYTES`] in production, a small test budget in
    /// the regression tests below.
    fn evict_for_budget(&mut self) {
        let budget = self.budget_bytes;
        while self.total_bytes > budget {
            // Cold entries first, then hot ones; LRU within each tier.
            let victim = self
                .entries
                .iter()
                .filter(|(_, e)| !e.lowered_this_frame)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(id, _)| *id)
                .or_else(|| {
                    self.entries
                        .iter()
                        .min_by_key(|(_, e)| e.last_used)
                        .map(|(id, _)| *id)
                });
            let Some(id) = victim else { break };
            self.remove(&id);
        }
    }
}

/// WebGPU paint backend. See the module docs for the design.
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    atlas: Atlas,
    sampler: wgpu::Sampler,
    /// Group 0 for every pipeline: the per-frame uniform.
    frame_layout: wgpu::BindGroupLayout,
    /// Group 1 for glyph/image pipelines: sampled texture + sampler.
    tex_layout: wgpu::BindGroupLayout,
    /// Group 1 for the blend-composite pipeline: group texture +
    /// sampler + copied backdrop texture.
    blend_layout: wgpu::BindGroupLayout,
    /// Group 2 for every pipeline: the exact fractional clip uniform.
    clip_layout: wgpu::BindGroupLayout,
    /// Group 2 for the filter pipeline: the [`FilterUniform`] params
    /// (`SHADER_FILTER`). Filter passes render into offscreen targets
    /// with no clipping (the group texture was already clipped when it
    /// rendered), so the filter pipeline has its own layout without
    /// the clip group.
    filter_uniform_layout: wgpu::BindGroupLayout,
    /// Clip bind groups by clip-rect bits (`None` = clipping disabled):
    /// rects repeat across frames, so the 32-byte uniform + bind group
    /// are built once per distinct clip. Cleared at the start of every
    /// [`WgpuBackend::render_into`] so a huge one-frame clip set cannot
    /// pin buffers across frames.
    clip_bg_cache: HashMap<ClipKey, wgpu::BindGroup>,
    frame_uniform_buf: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    glyph_bind_group: wgpu::BindGroup,
    /// Atlas generation [`WgpuBackend::glyph_bind_group`] was built
    /// against. The atlas retires its page when a glyph does not fit,
    /// installing a fresh generation; the bind group is rebuilt to sample
    /// the new texture. Quads lowered *before* the rollover keep the old
    /// page's cell rects, so they must sample the retired page: draw
    /// batches carry their generation (see [`Op::Draw`]) and bind through
    /// `retired_glyph_bind_groups`. Updated by [`WgpuBackend::lower`]
    /// whenever lowering observes a newer generation.
    glyph_bind_group_generation: u64,
    /// Bind groups sampling retired atlas pages, keyed by generation.
    /// Built by [`WgpuBackend::lower`] for every retired generation a
    /// frame's glyph batches reference; cleared by `render_into` after
    /// the frame submits (with [`Atlas::end_frame`]), since the retired
    /// pages are dropped there and the bind groups would otherwise pin
    /// their textures.
    retired_glyph_bind_groups: HashMap<u64, wgpu::BindGroup>,
    pipelines: HashMap<wgpu::TextureFormat, Pipelines>,
    image_cache: ImageCache,
    /// Baked gradient textures by content hash (see [`GradientEntry`]).
    gradient_cache: HashMap<u64, GradientEntry>,
    /// Lowering generation, bumped once per [`WgpuBackend::lower`]:
    /// [`GradientEntry::last_frame`] stamps entries with this, and
    /// `lower` evicts entries stamped with an older value when it
    /// finishes, so the cache holds at most the current frame's
    /// distinct gradients.
    gradient_frame: u64,
    /// Per-pipe persistent instance buffers: (buffer, capacity in bytes).
    /// Reused across frames and reallocated (grown) only when a frame's
    /// instance data outgrows the capacity, instead of allocating a fresh
    /// GPU buffer on every frame. Slots follow [`PipeKind::slot`].
    instance_bufs: [Option<(wgpu::Buffer, u64)>; 5],
    /// Consecutive upload passes each pipe's instance bytes used at most
    /// 1/INSTANCE_SHRINK_RATIO of its buffer capacity; drives the
    /// shrink-back policy in [`WgpuBackend::instance_buffer`].
    instance_small_streak: [u32; 5],
    /// Free-list of bounds-sized offscreen textures for isolated
    /// opacity groups. Checked out by
    /// [`WgpuBackend::checkout_group_target`] per group render and
    /// returned only after the frame's final submit (see `render_into`):
    /// a checked-out target is first a render attachment and then sampled
    /// by a not-yet-submitted composite, so returning it mid-frame would
    /// let a later group overwrite pixels a pending composite still
    /// needs. Exact-size matches only; the pool itself is capped at
    /// [`MAX_POOLED_GROUP_TARGETS`] (see `render_into`).
    group_target_pool: Vec<GroupTarget>,
    /// Targets checked out this frame; drained back into
    /// `group_target_pool` by `render_into` after the frame submits.
    group_targets_in_use: Vec<GroupTarget>,
    /// Free-list of backdrop-capture textures for [`Op::BackdropFilter`]
    /// (see [`WgpuBackend::checkout_capture_target`]). Captures need
    /// `COPY_DST | TEXTURE_BINDING | RENDER_ATTACHMENT` in the current
    /// target's format, which the group pool's Rgba8Unorm targets don't
    /// provide, so they get their own pool keyed by exact (width,
    /// height, format). Checked-out captures return here only after the
    /// frame's final submit: the filter passes sample the capture from
    /// not-yet-submitted commands, so returning one mid-frame would let
    /// a later element's copy overwrite pixels a pending filter still
    /// needs. Capped at [`MAX_POOLED_GROUP_TARGETS`] like the group
    /// pool (see `render_into`).
    capture_pool: Vec<GroupTarget>,
    /// Captures checked out this frame; drained back into
    /// `capture_pool` by `render_into` after the frame submits.
    captures_in_use: Vec<GroupTarget>,
    /// Filter-pass bind groups (texture + uniform) created by
    /// [`WgpuBackend::run_filter_pass`]. They must stay alive until the
    /// frame submits: the encoder records `set_bind_group` references,
    /// but dropping the bind group (and its buffer) before submit
    /// destroys the GPU resources out from under the pending commands.
    /// Drained by `render_into` after the frame submits.
    filter_transients: Vec<wgpu::BindGroup>,
    /// Persistent filter-uniform ring for [`WgpuBackend::run_filter_pass`]:
    /// one slot per pass at `filter_uniform_stride` bytes (uniform-offset
    /// aligned), written with `queue.write_buffer` then referenced via a
    /// dynamic offset — so a filter pass allocates no GPU buffer, just
    /// claims a slot. `filter_uniform_bg` binds the whole buffer once;
    /// `filter_uniform_next` counts claimed slots this frame and resets
    /// in `render_into` after the submit (slots stay valid until then).
    /// Grows by doubling when a frame needs more slots than
    /// `filter_uniform_slots`; the retired buffer's bind group is parked
    /// in `filter_transients` so pending commands keep it alive until
    /// that same submit.
    filter_uniform_buf: wgpu::Buffer,
    filter_uniform_bg: wgpu::BindGroup,
    filter_uniform_stride: u64,
    filter_uniform_slots: u32,
    filter_uniform_next: u32,
    /// Resolves a [`FontKey`] to font bytes for text rasterization.
    /// Same hook shape as the CPU backend's (`CpuBackend::new`); `None`
    /// until set, in which case text runs paint nothing (also exactly
    /// like the CPU backend with an empty resolver). `Arc` (not `Rc`
    /// like the CPU side) because the process-wide shared backend
    /// requires `Send + Sync`.
    get_font_data: Option<Arc<dyn Fn(FontKey) -> Option<FontData> + Send + Sync>>,
}

/// Thread-local [`WgpuBackend`] cache for [`WgpuBackend::render_offscreen`].
///
/// The wrapper's `Drop` intentionally forgets the backend instead of
/// dropping it. Thread-local destructors run during TLS teardown, and by
/// then wgpu-core's own thread-locals (its debug-only lock-trace cell)
/// may already be destroyed, so dropping a `wgpu::Queue`/`Device` at that
/// point panics inside wgpu-core and aborts the process (observed in
/// `render::tests::full_pipeline_paints_text_with_real_fonts`). The leak
/// is bounded — at most one backend per thread that ever rendered — and
/// the OS reclaims it at process exit.
///
/// A render thread that is done rendering (e.g. a worker thread shutting
/// down) should call [`shutdown_headless_backend`] instead of relying on
/// the forget: that drains the slot on a live thread, running the real
/// destructor while wgpu-core's thread-locals are still alive, and frees
/// the device, queue, and caches immediately. Drops that happen outside
/// teardown (e.g. replacing the cached backend after a device loss, or
/// the explicit shutdown) still run the real destructor: only this
/// wrapper's own `Drop` forgets.
///
/// Why not an automatic hook: render threads are caller-owned (the WPT
/// driver, the app, tests), and Rust runs thread-local destructors in an
/// unspecified order with no stable "before TLS teardown" hook point —
/// another TLS destructor would face the same ordering hazard. The
/// explicit drain is the only sound place to destroy eagerly.
struct CachedBackend(Option<WgpuBackend>);

impl Drop for CachedBackend {
    fn drop(&mut self) {
        // Never run `WgpuBackend`'s destructor during TLS teardown (see
        // above); `mem::forget` skips it entirely.
        std::mem::forget(self.0.take());
    }
}

/// Drain the calling thread's cached headless backend, running its real
/// destructor now instead of leaking it at TLS teardown.
///
/// Call this on a render thread that will not call
/// [`WgpuBackend::render_offscreen`] again (e.g. a worker thread that is
/// shutting down). Dropping here is safe — wgpu-core's thread-locals are
/// alive outside teardown — and frees the device, queue, and all GPU
/// caches immediately instead of holding them until process exit.
/// Threads that never call this keep the old behavior: the
/// [`CachedBackend`] `Drop` forgets the backend during TLS teardown
/// rather than risk the wgpu-core teardown abort. Calling it on a thread
/// with no cached backend is a no-op.
pub fn shutdown_headless_backend() {
    HEADLESS_BACKEND.with(|slot| {
        // Take the backend out of the forgetting wrapper and drop it
        // directly: this runs `WgpuBackend`'s real destructor while
        // wgpu-core's thread-locals are still alive. (`CachedBackend`'s
        // own `Drop` is not involved: it only forgets.)
        let backend = slot.borrow_mut().0.take();
        drop(backend);
    });
}

thread_local! {
    /// One headless backend per thread, reused across
    /// [`WgpuBackend::render_offscreen`] calls so a WPT driver rendering
    /// hundreds of tests does not pay device bring-up per test.
    /// Thread-local rather than a global lock, so concurrent callers on
    /// different threads never serialize; `wgpu` objects are `Send`, so
    /// moving the backend into its thread's slot is sound. The slot stays
    /// `None` — re-probed on every call — while no adapter exists, which
    /// preserves `render_offscreen`'s `None`-means-no-GPU contract.
    static HEADLESS_BACKEND: RefCell<CachedBackend> = const { RefCell::new(CachedBackend(None)) };
}

/// Format a list-marker `index` as decimal into `buf`, returning the
/// written digits. `pad_two` left-pads a single digit with `'0'`
/// (`decimal-leading-zero`). `buf` needs 20 bytes (`u32::MAX` is 10
/// digits, plus the pad). Stack-allocated: the per-frame marker path
/// never touches the heap.
fn format_marker_numeral(buf: &mut [u8; 20], index: u32, pad_two: bool) -> &str {
    let mut i = buf.len();
    let mut n = index;
    loop {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    if pad_two && i == buf.len() - 1 {
        i -= 1;
        buf[i] = b'0';
    }
    // Only ASCII digits were written.
    str::from_utf8(&buf[i..]).expect("decimal digits are valid UTF-8")
}

impl WgpuBackend {
    /// Create a backend from an existing device/queue pair.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bowser-frame-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let tex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bowser-tex-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        // Blend-mode compositing (group 1 on the blend pipeline): the
        // group texture + sampler, plus the copied backdrop texture the
        // blend shader samples alongside it.
        let blend_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bowser-blend-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        // Exact fractional clip uniform (group 2 on every pipeline): the
        // fragment shader applies the per-pixel clip coverage instead of
        // relying on the outward-snapped integer scissor.
        // `CLIP_UNIFORM_BYTES` — u32 count + 12 pad bytes, then
        // `MAX_CLIP_ENTRIES` entries of (vec4 rect + vec4 radii) (see
        // `ClipUniform` in `SHADER_COMMON`).
        let clip_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bowser-clip-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(CLIP_UNIFORM_BYTES),
                },
                count: None,
            }],
        });
        // Filter-pass params (group 2 on the filter pipeline; see
        // `FilterUniform` in `SHADER_FILTER`).
        let filter_uniform_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bowser-filter-uniform-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        // Dynamic: `run_filter_pass` claims one slot per
                        // pass out of a persistent ring buffer (see the
                        // `filter_uniform_*` field docs) instead of
                        // allocating a uniform buffer per pass.
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(FILTER_UNIFORM_BYTES as u64),
                    },
                    count: None,
                }],
            });

        // Filter-uniform ring (see the `filter_uniform_*` field docs):
        // slots are uniform-offset aligned so each pass can bind its own
        // slot with a dynamic offset.
        let filter_uniform_stride = (FILTER_UNIFORM_BYTES as u64)
            .next_multiple_of(device.limits().min_uniform_buffer_offset_alignment.max(1) as u64);
        const FILTER_UNIFORM_INITIAL_SLOTS: u32 = 16;
        let (filter_uniform_buf, filter_uniform_bg) = Self::make_filter_uniform_ring(
            device,
            &filter_uniform_layout,
            filter_uniform_stride,
            FILTER_UNIFORM_INITIAL_SLOTS,
        );

        let frame_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bowser-frame-uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bowser-frame-bg"),
            layout: &frame_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_uniform_buf.as_entire_binding(),
            }],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bowser-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..wgpu::SamplerDescriptor::default()
        });

        let atlas = Atlas::new(device);
        let glyph_bind_group =
            Self::glyph_bind_group(device, &tex_layout, atlas.texture_view(), &sampler);

        Self {
            device: device.clone(),
            queue: queue.clone(),
            atlas,
            sampler,
            frame_layout,
            tex_layout,
            blend_layout,
            clip_layout,
            filter_uniform_layout,
            clip_bg_cache: HashMap::new(),
            frame_uniform_buf,
            frame_bind_group,
            glyph_bind_group,
            glyph_bind_group_generation: 0,
            retired_glyph_bind_groups: HashMap::new(),
            pipelines: HashMap::new(),
            image_cache: ImageCache::default(),
            gradient_cache: HashMap::new(),
            gradient_frame: 0,
            instance_bufs: [None, None, None, None, None],
            instance_small_streak: [0; 5],
            group_target_pool: Vec::new(),
            group_targets_in_use: Vec::new(),
            capture_pool: Vec::new(),
            captures_in_use: Vec::new(),
            filter_transients: Vec::new(),
            filter_uniform_buf,
            filter_uniform_bg,
            filter_uniform_stride,
            filter_uniform_slots: FILTER_UNIFORM_INITIAL_SLOTS,
            filter_uniform_next: 0,
            get_font_data: None,
        }
    }

    /// Attach the font-data resolver used for text rasterization.
    ///
    /// Additive (not a `new` parameter) so the frozen `new(device, queue)`
    /// surface is untouched. The app worker calls this with its per-window
    /// [`crate::fonts::FontCache`]; the associated
    /// [`WgpuBackend::render_offscreen`] takes the cache explicitly as a
    /// parameter instead. Takes `Arc` rather than the CPU backend's `Rc`:
    /// the resolver is `Send + Sync + 'static` so backends stay shareable
    /// across threads.
    pub fn set_font_data(
        &mut self,
        get_font_data: Arc<dyn Fn(FontKey) -> Option<FontData> + Send + Sync>,
    ) {
        // Same hook already installed (the common per-frame case:
        // `render_offscreen` reinstalls the cache's stable resolver every
        // call, and `FontCache::resolver` now returns one `Arc` until a
        // new face resolves): the key space is unchanged, so skip the
        // atlas reset — cached glyphs stay valid and the thread-local
        // font-data cache (keyed by hook identity) stays warm.
        if self
            .get_font_data
            .as_ref()
            .is_some_and(|installed| Arc::ptr_eq(installed, &get_font_data))
        {
            return;
        }
        // A new resolver may map the same `FontKey` to a different face's
        // bytes, but the atlas cache is keyed by `FontKey` — without a
        // reset, glyphs would reuse the old face's coverage. Drop the
        // atlas (and every bind group bound to its pages) so all glyphs
        // re-rasterize against the new faces.
        self.atlas = Atlas::new(&self.device);
        self.glyph_bind_group_generation = 0;
        self.retired_glyph_bind_groups.clear();
        self.rebuild_glyph_bind_group();
        self.get_font_data = Some(get_font_data);
    }

    /// Bind group sampling one atlas page's texture (group 1 of the
    /// glyph pipeline's layout).
    fn glyph_bind_group(
        device: &wgpu::Device,
        tex_layout: &wgpu::BindGroupLayout,
        view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bowser-glyph-bg"),
            layout: tex_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    /// Bind group sampling one atlas generation's page texture, for
    /// glyph batches lowered before a mid-frame rollover retired that
    /// page. `None` when the generation is unknown or was already
    /// dropped by [`Atlas::end_frame`].
    fn build_glyph_bind_group_for(&self, generation: u64) -> Option<wgpu::BindGroup> {
        let view = self.atlas.view_for_generation(generation)?;
        Some(Self::glyph_bind_group(
            &self.device,
            &self.tex_layout,
            view,
            &self.sampler,
        ))
    }

    /// Rebuild the glyph bind group against the atlas's current
    /// generation texture. Called from [`WgpuBackend::lower`] when
    /// lowering observed a newer generation than the bind group was
    /// built for (see `glyph_bind_group_generation`).
    fn rebuild_glyph_bind_group(&mut self) {
        self.glyph_bind_group = Self::glyph_bind_group(
            &self.device,
            &self.tex_layout,
            self.atlas.texture_view(),
            &self.sampler,
        );
    }

    /// Create a backend on a headless device.
    ///
    /// Requests a Vulkan adapter (software/lavapipe counts as a fallback
    /// adapter) via [`WgpuBackend::request_device`]. Returns `None` —
    /// instead of panicking — when no adapter is available (headless CI
    /// without Vulkan), so callers can degrade to the CPU rasterizer.
    pub fn new_headless() -> Option<Self> {
        let (_adapter, device, queue) = Self::request_device(None)?;
        Some(Self::new(&device, &queue))
    }

    /// Render one frame to a surface texture.
    ///
    /// Configures the surface from `config`, renders, and presents. Frames
    /// that fail to acquire (timeout/occluded/outdated/lost) are skipped.
    pub fn render_to_surface(
        &mut self,
        list: &DisplayList,
        viewport: Size,
        scroll: (f32, f32),
        surface: &wgpu::Surface,
        config: &wgpu::SurfaceConfiguration,
    ) {
        surface.configure(&self.device, config);
        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => return,
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.render_into(
            &view,
            config.format,
            list,
            viewport,
            scroll,
            config.width,
            config.height,
        );
        self.queue.present(frame);
    }

    /// Render offscreen at `width_px` x `height_px` and read the pixels
    /// back (RGBA8, row-major, top row first).
    ///
    /// The headless device is brought up once per thread and reused
    /// across calls ([`HEADLESS_BACKEND`]): only the first call on a
    /// thread pays adapter enumeration, device creation, and
    /// shader/pipeline compilation. The backend is shared, but every call
    /// installs its own font resolver first: [`FontKey`]s are cache-local,
    /// so they must resolve against the [`FontCache`] the display list
    /// was built with. Installing a *different* resolver resets the glyph
    /// atlas (which is keyed by `FontKey`); re-installing the same cache's
    /// resolver is a no-op — [`FontCache::resolver`] hands back one stable
    /// `Arc` until a new face resolves, so consecutive frames share the
    /// atlas and the thread-local font-data cache stays warm. Returns
    /// `None` when no Vulkan adapter is available so the caller
    /// (`crate::render`) can fall back to the CPU rasterizer.
    pub fn render_offscreen(
        list: &DisplayList,
        width_px: u32,
        height_px: u32,
        fonts: &FontCache,
    ) -> Option<Vec<u8>> {
        Self::render_offscreen_with_scroll(list, width_px, height_px, fonts, (0.0, 0.0))
    }

    /// [`render_offscreen`] with an explicit viewport scroll offset
    /// (`scroll.0` = x, `scroll.1` = y, CSS px): the offset is subtracted
    /// from content coordinates at draw time, exactly like the windowed
    /// path's per-frame scroll uniform (`render_to_surface`), so a
    /// headless render can show the scrolled position. The plain entry
    /// keeps the historical zero-scroll contract. Returns `None` when no
    /// Vulkan adapter is available, like [`render_offscreen`].
    pub fn render_offscreen_with_scroll(
        list: &DisplayList,
        width_px: u32,
        height_px: u32,
        fonts: &FontCache,
        scroll: (f32, f32),
    ) -> Option<Vec<u8>> {
        HEADLESS_BACKEND.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.0.is_none() {
                slot.0 = WgpuBackend::new_headless();
            }
            let backend = slot.0.as_mut()?;
            backend.set_font_data(fonts.resolver());
            let pixels = backend.render_offscreen_inner(list, width_px, height_px, scroll);
            if pixels.is_none() {
                // Readback failed (device lost, ...): drop the cached
                // backend — a normal drop, wgpu's thread-locals are alive
                // outside teardown — so the next call re-brings-up
                // instead of retrying on a dead device.
                slot.0 = None;
            }
            pixels
        })
    }

    /// Offscreen target dimensions must be nonzero and within the
    /// device's 2D texture limit: `create_texture` with a zero or
    /// oversized extent is a wgpu validation error, which the default
    /// uncaptured-error handler turns into a panic.
    fn target_dims_ok(&self, w: u32, h: u32) -> bool {
        if w == 0 || h == 0 {
            return false;
        }
        let max_dim = self.device.limits().max_texture_dimension_2d;
        w <= max_dim && h <= max_dim
    }

    /// Method form of offscreen rendering on an existing backend, with an
    /// explicit CSS viewport (page px) that may differ from the pixel
    /// target size. Degrades to an empty vector when the readback fails.
    pub fn render_offscreen_with_viewport(
        &mut self,
        list: &DisplayList,
        viewport: Size,
        width_px: u32,
        height_px: u32,
    ) -> Vec<u8> {
        // Zero-size targets have no pixels, and oversized targets exceed
        // the device; `make_target` would hand a bad extent to
        // `create_texture`, which is a wgpu validation error. (Matches
        // `render_to_rgba`'s empty-vector convention.)
        if !self.target_dims_ok(width_px, height_px) {
            return Vec::new();
        }
        let target = self.make_target(width_px, height_px);
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        self.render_into(
            &view,
            wgpu::TextureFormat::Rgba8Unorm,
            list,
            viewport,
            (0.0, 0.0),
            width_px,
            height_px,
        );
        Self::readback(&self.device, &self.queue, &target, width_px, height_px).unwrap_or_default()
    }

    /// Lower + render + read back on this backend's device.
    /// `None` when the readback fails (device lost mid-frame, ...).
    fn render_offscreen_inner(
        &mut self,
        list: &DisplayList,
        width_px: u32,
        height_px: u32,
        scroll: (f32, f32),
    ) -> Option<Vec<u8>> {
        self.render_to_rgba_with_scroll(list, width_px, height_px, scroll)
    }

    /// Render one frame into an existing [`wgpu::TextureView`].
    ///
    /// `size` is the target size in device px; the display list's page-px
    /// coordinates map 1:1 onto it and scrolling is zero. The pipeline set
    /// is chosen from the view's texture format, so the caller must pass a
    /// view whose (possibly reinterpreted) format matches the texture it
    /// was created from.
    ///
    /// `mix-blend-mode` groups copy the backdrop out of the target
    /// texture mid-frame: give the target
    /// [`wgpu::TextureUsages::COPY_SRC`] when the list may contain
    /// blend groups, otherwise they degrade to plain source-over
    /// (with a debug-build warning).
    pub fn render(&mut self, list: &DisplayList, view: &wgpu::TextureView, size: (u32, u32)) {
        let (w, h) = size;
        let viewport = Size {
            w: w as f32,
            h: h as f32,
        };
        self.render_into(
            view,
            view.texture().format(),
            list,
            viewport,
            (0.0, 0.0),
            w,
            h,
        );
    }

    /// Render `list` to straight RGBA8 row-major pixels (`w` x `h`,
    /// top row first) using an internal offscreen target.
    ///
    /// This is the headless/WPT readback path: it renders on this
    /// backend's device and copies the result to CPU memory. Returns
    /// `Some` with an empty vector when either dimension is zero, and
    /// `None` when the readback itself fails (so callers can degrade to
    /// the CPU rasterizer instead of panicking).
    pub fn render_to_rgba(&mut self, list: &DisplayList, w: u32, h: u32) -> Option<Vec<u8>> {
        self.render_to_rgba_with_scroll(list, w, h, (0.0, 0.0))
    }

    /// [`render_to_rgba`] with an explicit viewport scroll offset
    /// (`scroll.0` = x, `scroll.1` = y, CSS px): subtracted from content
    /// coordinates at draw time, matching the windowed
    /// `render_to_surface` uniform. The plain entry keeps the historical
    /// zero-scroll contract.
    pub fn render_to_rgba_with_scroll(
        &mut self,
        list: &DisplayList,
        w: u32,
        h: u32,
        scroll: (f32, f32),
    ) -> Option<Vec<u8>> {
        // Zero-size targets have no pixels; oversized targets exceed the
        // device. `make_target` would hand a bad extent to
        // `create_texture`, which is a wgpu validation error.
        if !self.target_dims_ok(w, h) {
            return Some(Vec::new());
        }
        let target = self.make_target(w, h);
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let viewport = Size {
            w: w as f32,
            h: h as f32,
        };
        self.render_into(&view, view.texture().format(), list, viewport, scroll, w, h);
        Self::readback(&self.device, &self.queue, &target, w, h)
    }

    /// Core frame render into `view`.
    // 8 params, all used; bundling would only add indirection.
    #[allow(clippy::too_many_arguments)]
    fn render_into(
        &mut self,
        view: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        list: &DisplayList,
        viewport: Size,
        scroll: (f32, f32),
        target_w: u32,
        target_h: u32,
    ) {
        if target_w == 0 || target_h == 0 {
            return;
        }
        // Clip bind groups are per-frame: a huge one-frame clip set must
        // not pin uniform buffers across frames (same rationale as the
        // instance-buffer shrink streak).
        self.clip_bg_cache.clear();
        // Device px per page px, per axis: geometry scales in the vertex
        // shader via the viewport uniform, but text rasterizes at device
        // px (see `emit_shaped_text`). `lower` sanitizes degenerate
        // values to 1.0.
        let scale = (
            if viewport.w > 0.0 {
                target_w as f32 / viewport.w
            } else {
                1.0
            },
            if viewport.h > 0.0 {
                target_h as f32 / viewport.h
            } else {
                1.0
            },
        );
        let lowered = self.lower(list, scale);

        // Per-frame uniform: viewport + scroll.
        let mut uniform = [0u8; 16];
        for (i, v) in [viewport.w, viewport.h, scroll.0, scroll.1]
            .iter()
            .enumerate()
        {
            uniform[i * 4..(i + 1) * 4].copy_from_slice(&v.to_ne_bytes());
        }
        self.queue
            .write_buffer(&self.frame_uniform_buf, 0, &uniform);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bowser-frame"),
            });
        // Flatten the whole lowering tree (root + nested opacity groups)
        // into combined per-pipe byte vectors and upload once: one
        // `queue.write_buffer` per pipe per frame, no per-group encoders,
        // submits, or instance-buffer re-uploads. `prepared` records each
        // lowering's base instance offsets into the combined buffers, so
        // every lowering's draws address disjoint slices.
        let mut combined: [Vec<u8>; 5] =
            [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        Self::reserve_combined(&lowered, &mut combined);
        let prepared = Self::prepare_lowering(&lowered, &mut combined);
        let bufs = self.upload_instances(&combined);
        // Clone: `execute_ops_into` takes `&mut self`, so the root bind
        // group cannot be borrowed off `self` for the call. `BindGroup` is
        // refcounted, so this is one atomic bump, not a GPU copy.
        let frame_bg = self.frame_bind_group.clone();
        self.execute_ops_into(
            &mut encoder,
            view,
            view.texture(),
            format,
            &lowered,
            &prepared,
            &bufs,
            viewport,
            scroll,
            target_w,
            target_h,
            &frame_bg,
        );
        self.queue.submit([encoder.finish()]);
        // The frame is submitted: no pending draw can reference an image
        // texture anymore, so enforce the image-cache byte budget now.
        // Mid-frame, [`ImageCache::reserve`] already kept uploads within
        // the budget by evicting only entries the lowering had not
        // referenced; this pass trims whatever the pinned working set
        // pushed over (see `IMAGE_CACHE_BUDGET_BYTES`).
        self.image_cache.evict_for_budget();
        // The frame is submitted: every checked-out group target's
        // composite was enqueued after its render, so no pending draw can
        // reference one anymore — return them to the pool instead of
        // dropping a bounds-sized texture per group per frame. Group
        // targets are bounds-sized now (see `lowered_bounds`), so the
        // pool holds many sizes; the retain below is gone — every size is
        // reusable (checkout matches exact size).
        self.group_target_pool
            .append(&mut self.group_targets_in_use);
        // Same for backdrop captures: every checked-out capture's copy
        // and filter passes were enqueued before its writeback draw, so
        // no pending draw can reference one anymore.
        self.capture_pool.append(&mut self.captures_in_use);
        // Filter-pass bind groups are no longer referenced by pending
        // commands after submit; drop them so their buffers/views can be
        // recycled.
        self.filter_transients.clear();
        // The frame's filter-uniform ring slots are free again: no
        // pending command references them after the submit.
        self.filter_uniform_next = 0;
        // Bound the pools themselves: a one-frame spike of many groups
        // or backdrop-filter elements must not pin dozens of textures
        // across all later frames. The free lists are unordered, so
        // dropping from the end is as good as any.
        while self.group_target_pool.len() > MAX_POOLED_GROUP_TARGETS {
            self.group_target_pool.pop();
        }
        while self.capture_pool.len() > MAX_POOLED_GROUP_TARGETS {
            self.capture_pool.pop();
        }
        // The frame is submitted: no new draw can reference retired atlas
        // generations anymore, so drop them instead of leaking one 1 MiB
        // texture per page overflow (see `Atlas::end_frame`). Their bind
        // groups go too, or they would pin the retired pages' textures.
        self.retired_glyph_bind_groups.clear();
        self.atlas.end_frame();
    }

    /// Begin a clear-transparent render pass into `view` on `encoder`.
    /// For a fresh (or pooled, reused) target: `LoadOp::Clear` wipes any
    /// previous contents.
    fn begin_pass<'e>(
        encoder: &'e mut wgpu::CommandEncoder,
        view: &'e wgpu::TextureView,
    ) -> wgpu::RenderPass<'e> {
        Self::begin_pass_with_load(encoder, view, wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT))
    }

    /// Re-begin a pass over a target that already holds this frame's
    /// draws (used after ending the pass around an opacity group):
    /// `LoadOp::Load` preserves them. Re-beginning with `Clear` here
    /// would erase everything drawn before the group.
    fn begin_pass_load<'e>(
        encoder: &'e mut wgpu::CommandEncoder,
        view: &'e wgpu::TextureView,
    ) -> wgpu::RenderPass<'e> {
        Self::begin_pass_with_load(encoder, view, wgpu::LoadOp::Load)
    }

    fn begin_pass_with_load<'e>(
        encoder: &'e mut wgpu::CommandEncoder,
        view: &'e wgpu::TextureView,
        load: wgpu::LoadOp<wgpu::Color>,
    ) -> wgpu::RenderPass<'e> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bowser-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    /// Execute a lowered op stream into `view`.
    ///
    /// `Op::Group` renders the group's lowering into an offscreen texture
    /// (recursively, so nested groups work the same way) and composites it
    /// over the current target with the group opacity. The pass is ended
    /// and re-begun around each group; groups are rare, so the extra passes
    /// cost nothing in the common case.
    ///
    /// One encoder, one submit for the whole frame: `render_group_texture`
    /// records into this same `encoder` and never submits mid-frame, and
    /// every lowering's instances were uploaded once up front
    /// ([`WgpuBackend::prepare_lowering`] +
    /// [`WgpuBackend::upload_instances`]). `prepared` carries each
    /// lowering's base instance offsets into the shared buffers, so draws
    /// recorded before and after a group read disjoint slices — no
    /// per-group instance-buffer re-uploads, no buffer aliasing.
    // 13 params, all used; bundling would only add indirection.
    #[allow(clippy::too_many_arguments)]
    fn execute_ops_into(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        // The texture behind `view`: the blend-composite path copies
        // the backdrop out of it (see `Op::Group` handling below).
        target_tex: &wgpu::Texture,
        format: wgpu::TextureFormat,
        lowered: &Lowered,
        prepared: &Prepared,
        bufs: &[Option<wgpu::Buffer>; 5],
        viewport: Size,
        scroll: (f32, f32),
        target_w: u32,
        target_h: u32,
        // Group 0 bind group for the draws below. The root frame uses
        // `self.frame_bind_group`; isolated groups render with a
        // bounds-mapped one built by `render_group_texture`, so the
        // vertex shader maps group page px onto the group's snapped
        // device-px texture at the true device scale.
        frame_bg: &wgpu::BindGroup,
    ) {
        // Persistent per-pipe buffers, uploaded once up front by the
        // caller: `bufs` are the current handles, `prepared` the
        // per-lowering base instance offsets into them.
        let pipes = self.pipelines_for(format);

        let mut pass = Self::begin_pass(encoder, view);
        // Last scissor as (x, y, w, h) device-px extents, re-applied when
        // the pass is re-begun around a group (`None` = whole target, the
        // pass default).
        let mut last_scissor: Option<(u32, u32, u32, u32)> = None;
        // Group-2 clip bind group for the current scissor rect (starts
        // disabled); set on every draw so the fragment shader enforces
        // the exact fractional clip edge.
        let mut clip_bg = self.clip_bind_group(None);

        {
            let mut clip_empty = false;
            for op in &lowered.ops {
                match op {
                    Op::Scissor(snap) => {
                        let entries = snap.as_slice();
                        // Intersected bound in page px (entries are
                        // oldest-first).
                        let mut bound: Option<Rect> = None;
                        for e in entries {
                            bound = Some(match bound {
                                Some(b) => intersect(b, e.rect),
                                None => e.rect,
                            });
                        }
                        let sx = target_w as f32 / viewport.w.max(f32::EPSILON);
                        let sy = target_h as f32 / viewport.h.max(f32::EPSILON);
                        // Exact device-px clip bound (f32): page px ->
                        // device px with the same mapping the vertex
                        // shader uses (see `page_to_ndc`). The integer
                        // scissor below stays as a coarse outward-snapped
                        // cull; the fragment shader's `clip_coverage`
                        // enforces the exact fractional edge per pixel,
                        // per entry.
                        let clip_f32: Option<(f32, f32, f32, f32)> = bound.map(|r| {
                            (
                                ((r.x - scroll.0) * sx).max(0.0),
                                ((r.y - scroll.1) * sy).max(0.0),
                                ((r.x + r.w - scroll.0) * sx).min(target_w as f32),
                                ((r.y + r.h - scroll.1) * sy).min(target_h as f32),
                            )
                        });
                        let (x, y, w, h) = match clip_f32 {
                            None => (0, 0, target_w, target_h),
                            Some((x0, y0, x1, y1)) => (
                                (x0.floor() as u32).min(target_w),
                                (y0.floor() as u32).min(target_h),
                                x1.ceil() as u32,
                                y1.ceil() as u32,
                            ),
                        };
                        // Clip bind group for the exact entries; an empty
                        // or absent clip disables clipping (empty clips
                        // skip their draws via `clip_empty` regardless).
                        let clip_key: ClipKey = match clip_f32 {
                            Some((x0, y0, x1, y1)) if x1 > x0 && y1 > y0 => {
                                let mut words = [[0u32; CLIP_ENTRY_WORDS]; MAX_CLIP_ENTRIES];
                                for (i, e) in entries.iter().enumerate() {
                                    // Words 0..8: device-px rect +
                                    // corner radii (see
                                    // `CLIP_ENTRY_WORDS`).
                                    let mut w = [0u32; CLIP_ENTRY_WORDS];
                                    w[0] = ((e.rect.x - scroll.0) * sx).max(0.0).to_bits();
                                    w[1] = ((e.rect.y - scroll.1) * sy).max(0.0).to_bits();
                                    w[2] = ((e.rect.x + e.rect.w - scroll.0) * sx)
                                        .min(target_w as f32)
                                        .to_bits();
                                    w[3] = ((e.rect.y + e.rect.h - scroll.1) * sy)
                                        .min(target_h as f32)
                                        .to_bits();
                                    w[4] = (e.radii[0] * sx).to_bits();
                                    w[5] = (e.radii[1] * sx).to_bits();
                                    w[6] = (e.radii[2] * sx).to_bits();
                                    w[7] = (e.radii[3] * sx).to_bits();
                                    if e.is_polygon() {
                                        // Word 8: kind, word 9: vertex
                                        // count, words 10..12: pad,
                                        // words 12..: device-px vertex
                                        // `(x, y, 0, 0)` quads.
                                        w[8] = 1;
                                        w[9] = u32::from(e.poly_len);
                                        for (j, v) in
                                            e.poly.iter().take(usize::from(e.poly_len)).enumerate()
                                        {
                                            w[12 + j * 4] = ((v[0] - scroll.0) * sx).to_bits();
                                            w[12 + j * 4 + 1] = ((v[1] - scroll.1) * sy).to_bits();
                                        }
                                    }
                                    words[i] = w;
                                }
                                Some((entries.len() as u8, words))
                            }
                            _ => None,
                        };
                        clip_bg = self.clip_bind_group(clip_key);
                        if w <= x || h <= y {
                            clip_empty = true;
                            last_scissor = None;
                        } else {
                            clip_empty = false;
                            last_scissor = Some((x, y, w - x, h - y));
                            pass.set_scissor_rect(x, y, w - x, h - y);
                        }
                    }
                    Op::Draw {
                        pipe,
                        image,
                        generation,
                        start,
                        count,
                    } => {
                        if clip_empty || *count == 0 {
                            continue;
                        }
                        let pipeline = match pipe {
                            PipeKind::Solid => &pipes.solid,
                            PipeKind::Glyph => &pipes.glyph,
                            PipeKind::Image => &pipes.image,
                            PipeKind::Shadow => &pipes.shadow,
                            // Composite instances are uploaded on demand
                            // per group op below, never batched as draws.
                            PipeKind::Composite => {
                                unreachable!("no Draw op uses the Composite pipe")
                            }
                        };
                        let buf = match &bufs[pipe.slot()] {
                            Some(b) => b,
                            None => continue,
                        };
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, frame_bg, &[]);
                        match pipe {
                            PipeKind::Solid | PipeKind::Shadow => {}
                            PipeKind::Glyph => {
                                // Each batch samples the atlas page its
                                // quads were rasterized into: after a
                                // mid-frame rollover the current page's
                                // texture holds different cells at the same
                                // rects, so pre-rollover batches must bind
                                // the retired page (built by `lower`).
                                let bind_group = if *generation == self.glyph_bind_group_generation
                                {
                                    self.glyph_bind_group.clone()
                                } else {
                                    match self.retired_glyph_bind_groups.get(generation) {
                                        Some(bg) => bg.clone(),
                                        None => {
                                            #[cfg(debug_assertions)]
                                            eprintln!(
                                                "[bowser_paint] skipping glyph draw batch: no bind group for atlas generation {generation}"
                                            );
                                            continue;
                                        }
                                    }
                                };
                                pass.set_bind_group(1, &bind_group, &[]);
                            }
                            PipeKind::Image => {
                                // Invariant (enforced by `ensure_image` at
                                // lowering): every Image-pipe draw batch
                                // carries an id and its texture is in the
                                // cache. If the invariant is ever violated,
                                // skip the batch instead of panicking.
                                let Some(id) = image else {
                                    #[cfg(debug_assertions)]
                                    eprintln!(
                                        "[bowser_paint] skipping image draw batch without image id"
                                    );
                                    continue;
                                };
                                let Some(entry) = self.image_cache.entries.get(id) else {
                                    #[cfg(debug_assertions)]
                                    eprintln!(
                                        "[bowser_paint] skipping image draw batch: texture missing for {id:?}"
                                    );
                                    continue;
                                };
                                pass.set_bind_group(1, &entry.bind_group, &[]);
                            }
                            // Handled above: no Draw op uses Composite.
                            PipeKind::Composite => {}
                        }
                        // Exact fractional clip (group 2); the integer
                        // scissor above is only a coarse cull.
                        pass.set_bind_group(2, &clip_bg, &[]);
                        pass.set_vertex_buffer(0, buf.slice(..));
                        // `start` is relative to this lowering's bytes;
                        // `prepared` shifts it into the frame's combined
                        // buffer, where this lowering owns a disjoint
                        // slice.
                        let base = prepared.bases[pipe.slot()];
                        pass.draw(0..4, base + *start..base + *start + *count);
                    }
                    Op::DrawGradient { tex, start, count } => {
                        if clip_empty || *count == 0 {
                            continue;
                        }
                        // Invariant (enforced by
                        // `ensure_gradient_texture` at lowering): every
                        // gradient batch's texture is in the gradient
                        // cache. If the invariant is ever violated,
                        // skip the batch instead of panicking (same
                        // policy as the image path above).
                        let Some(entry) = self.gradient_cache.get(tex) else {
                            #[cfg(debug_assertions)]
                            eprintln!(
                                "[bowser_paint] skipping gradient draw batch: texture missing for {tex:?}"
                            );
                            continue;
                        };
                        pass.set_pipeline(&pipes.image);
                        pass.set_bind_group(0, frame_bg, &[]);
                        pass.set_bind_group(1, &entry.bind_group, &[]);
                        // Exact fractional clip (group 2); the integer
                        // scissor above is only a coarse cull.
                        pass.set_bind_group(2, &clip_bg, &[]);
                        let buf = match &bufs[PipeKind::Image.slot()] {
                            Some(b) => b,
                            None => continue,
                        };
                        pass.set_vertex_buffer(0, buf.slice(..));
                        let base = prepared.bases[PipeKind::Image.slot()];
                        pass.draw(0..4, base + *start..base + *start + *count);
                    }
                    Op::Group(idx) => {
                        let group = &lowered.groups[*idx];
                        if clip_empty || group.opacity <= 0.0 {
                            continue;
                        }
                        let buf = match &bufs[PipeKind::Composite.slot()] {
                            Some(b) => b,
                            None => continue,
                        };
                        let base = prepared.bases[PipeKind::Composite.slot()];
                        // End the pass: the group renders into its own
                        // texture on this same encoder (no second encoder,
                        // no mid-frame submit), then a fresh pass
                        // composites it over the current target.
                        drop(pass);
                        let (bind_group, group_tex, gx0, gy0) = self.render_group_texture(
                            encoder,
                            group,
                            &prepared.groups[*idx],
                            bufs,
                            viewport,
                            scroll,
                            target_w,
                            target_h,
                        );
                        // Filter groups: run the planned passes over the
                        // group texture before compositing. The texture
                        // is spill-inflated at lowering, so blur and
                        // drop-shadow have room beyond the content
                        // bounds (mirrors the CPU backend's
                        // `paint_filter_group`). The composite samples
                        // through a bind group, so a filtered group
                        // needs a fresh one for the result texture.
                        let (bind_group, group_tex) = if group.filter.is_empty() {
                            (bind_group, group_tex)
                        } else {
                            let sx = target_w as f32 / viewport.w.max(f32::EPSILON);
                            let filtered = self.apply_filter_passes(
                                encoder,
                                group_tex,
                                &group.filter,
                                sx,
                                frame_bg,
                                &pipes,
                            );
                            let view =
                                filtered.create_view(&wgpu::TextureViewDescriptor::default());
                            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("bowser-group-bg-filtered"),
                                layout: &self.tex_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(&view),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                                    },
                                ],
                            });
                            (bg, filtered)
                        };
                        // The composite quad covers the group's snapped
                        // device bounds, converted back to page px under
                        // the current target's mapping, sampling the
                        // whole group texture (uv 0..1): group texel
                        // (0, 0) lands exactly on current-target device
                        // pixel (gx0, gy0) — the same pixel grid the CPU
                        // backend composites onto. The snapped rect is
                        // only known at execute time (it needs the
                        // current scroll and scale), so the instance is
                        // built here per group op and written into this
                        // lowering's reserved composite slot in the
                        // combined buffer — one disjoint slot per group
                        // op (see `prepare_lowering`), so groups sharing
                        // one frame never clobber each other's instance.
                        let sx = target_w as f32 / viewport.w.max(f32::EPSILON);
                        let sy = target_h as f32 / viewport.h.max(f32::EPSILON);
                        let (gw, gh) = (group_tex.width() as f32, group_tex.height() as f32);
                        let mut instance = [0u8; COMPOSITE_STRIDE];
                        for (i, v) in [
                            scroll.0 + gx0 / sx,
                            scroll.1 + gy0 / sy,
                            gw / sx,
                            gh / sy,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                            group.opacity,
                        ]
                        .iter()
                        .enumerate()
                        {
                            instance[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
                        }
                        instance[36..40].copy_from_slice(&group.blend.as_u32().to_le_bytes());
                        // 2D transform matrix (identity when the group has
                        // none): the vertex shader maps the quad's
                        // page-px corners through it.
                        let matrix = group.transform.unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
                        for (i, v) in matrix.iter().enumerate() {
                            let o = 40 + i * 4;
                            instance[o..o + 4].copy_from_slice(&v.to_le_bytes());
                        }
                        let offset = (base + *idx as u32) as u64 * COMPOSITE_STRIDE as u64;
                        self.queue.write_buffer(buf, offset, &instance);
                        let instance_range = offset..offset + COMPOSITE_STRIDE as u64;
                        // The blend path copies the backdrop out of the
                        // current target, which needs COPY_SRC. Offscreen
                        // targets (`make_target`) always have it; an
                        // externally supplied surface texture might not —
                        // degrade to plain source-over there instead of
                        // tripping wgpu validation.
                        let can_blend = !group.blend.is_normal()
                            && target_tex.usage().contains(wgpu::TextureUsages::COPY_SRC);
                        #[cfg(debug_assertions)]
                        if !group.blend.is_normal() && !can_blend {
                            eprintln!(
                                "[bowser_paint] blend target lacks COPY_SRC; \
                                 compositing as source-over"
                            );
                        }
                        if !can_blend {
                            // Load, not clear: the previous pass already drew
                            // this target's content and the composite must draw
                            // over it.
                            pass = Self::begin_pass_load(encoder, view);
                            if let Some((x, y, w, h)) = last_scissor {
                                pass.set_scissor_rect(x, y, w, h);
                            }
                            pass.set_pipeline(&pipes.composite);
                            // The current target's frame uniform: at the root
                            // this is `self.frame_bind_group`, but inside a
                            // group render it is the group's bounds-mapped
                            // uniform (`frame_bg`) — the composite quad is in
                            // page px and must map onto the *current* target,
                            // or a nested group's composite lands at the wrong
                            // offset in the parent group's texture.
                            pass.set_bind_group(0, frame_bg, &[]);
                            pass.set_bind_group(1, &bind_group, &[]);
                            // No clip on the composite: the group's content was
                            // already clipped when the group texture rendered —
                            // re-applying it here would square the fractional
                            // boundary coverage. Transform groups are the
                            // exception: their texture rendered with ancestor
                            // clips drained, so the outer clip state applies
                            // here, post-transform, to the warped quad.
                            if group.transform.is_some() {
                                pass.set_bind_group(2, &clip_bg, &[]);
                            } else {
                                pass.set_bind_group(2, &self.clip_bind_group(None), &[]);
                            }
                            pass.set_vertex_buffer(0, buf.slice(instance_range.clone()));
                            pass.draw(0..4, 0..1);
                        } else {
                            // Blend path: copy the backdrop (this target's
                            // content under the group's bounds) into a fresh
                            // texture, then composite with the blend
                            // pipeline, which blends the group texture
                            // against the copied backdrop and replaces the
                            // framebuffer (blending disabled: the shader
                            // outputs the final premultiplied value).
                            let (bw, bh) = (group_tex.width(), group_tex.height());
                            // Device-px origin of the group's snapped
                            // bounds in the current target (unclamped: may
                            // be negative or past the target edge when the
                            // group hangs off the viewport): group texel
                            // (i, j) is target texel (gx0 + i, gy0 + j).
                            let x0_raw = gx0 as i32;
                            let y0_raw = gy0 as i32;
                            // Clamp the copy to the target. Backdrop texel
                            // (i, j) corresponds to target texel
                            // (x0_raw + i, y0_raw + j), so the destination
                            // origin shifts by the clamped amount; texels
                            // outside the target stay transparent black (no
                            // fragment ever samples them — the composite
                            // quad is clipped to the target by the render
                            // pass).
                            let src_x = x0_raw.max(0) as u32;
                            let src_y = y0_raw.max(0) as u32;
                            let dst_x = (src_x as i32 - x0_raw) as u32;
                            let dst_y = (src_y as i32 - y0_raw) as u32;
                            let copy_w = ((bw as i32 - dst_x as i32)
                                .min(target_w as i32 - src_x as i32))
                            .max(0) as u32;
                            let copy_h = ((bh as i32 - dst_y as i32)
                                .min(target_h as i32 - src_y as i32))
                            .max(0) as u32;
                            let backdrop = self.device.create_texture(&wgpu::TextureDescriptor {
                                label: Some("bowser-blend-backdrop"),
                                size: wgpu::Extent3d {
                                    width: bw,
                                    height: bh,
                                    depth_or_array_layers: 1,
                                },
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D2,
                                format: target_tex.format(),
                                usage: wgpu::TextureUsages::COPY_DST
                                    | wgpu::TextureUsages::TEXTURE_BINDING
                                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                                view_formats: &[],
                            });
                            let backdrop_view =
                                backdrop.create_view(&wgpu::TextureViewDescriptor::default());
                            // Clear first: the copy below covers a clamped
                            // subregion, and untouched texels must read as
                            // transparent black.
                            drop(Self::begin_pass(encoder, &backdrop_view));
                            if copy_w > 0 && copy_h > 0 {
                                encoder.copy_texture_to_texture(
                                    wgpu::TexelCopyTextureInfo {
                                        texture: target_tex,
                                        mip_level: 0,
                                        origin: wgpu::Origin3d {
                                            x: src_x,
                                            y: src_y,
                                            z: 0,
                                        },
                                        aspect: wgpu::TextureAspect::All,
                                    },
                                    wgpu::TexelCopyTextureInfo {
                                        texture: &backdrop,
                                        mip_level: 0,
                                        origin: wgpu::Origin3d {
                                            x: dst_x,
                                            y: dst_y,
                                            z: 0,
                                        },
                                        aspect: wgpu::TextureAspect::All,
                                    },
                                    wgpu::Extent3d {
                                        width: copy_w,
                                        height: copy_h,
                                        depth_or_array_layers: 1,
                                    },
                                );
                            }
                            let group_view =
                                group_tex.create_view(&wgpu::TextureViewDescriptor::default());
                            let blend_bg =
                                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("bowser-blend-bg"),
                                    layout: &self.blend_layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: wgpu::BindingResource::TextureView(
                                                &group_view,
                                            ),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 2,
                                            resource: wgpu::BindingResource::TextureView(
                                                &backdrop_view,
                                            ),
                                        },
                                    ],
                                });
                            // Load, not clear: the previous pass already drew
                            // this target's content and the composite must
                            // draw over it.
                            pass = Self::begin_pass_load(encoder, view);
                            if let Some((x, y, w, h)) = last_scissor {
                                pass.set_scissor_rect(x, y, w, h);
                            }
                            pass.set_pipeline(&pipes.composite_blend);
                            // Current target's frame uniform (see above).
                            pass.set_bind_group(0, frame_bg, &[]);
                            pass.set_bind_group(1, &blend_bg, &[]);
                            // No clip on the composite (see above).
                            pass.set_bind_group(2, &self.clip_bind_group(None), &[]);
                            pass.set_vertex_buffer(0, buf.slice(instance_range.clone()));
                            pass.draw(0..4, 0..1);
                        }
                    }
                    Op::BackdropFilter(bf) => {
                        // `backdrop-filter`: the backdrop under the
                        // element's border box was already painted into
                        // the current target. Capture it, filter the
                        // capture, and draw it back over the same region
                        // (masked by the current clip: the filtered
                        // backdrop is part of the element's rendering, so
                        // the element's clip-path clips it like content).
                        // The ops after this one paint the element's own
                        // content over the filtered backdrop inline.
                        if clip_empty || bf.passes.is_empty() {
                            continue;
                        }
                        let sx = target_w as f32 / viewport.w.max(f32::EPSILON);
                        let sy = target_h as f32 / viewport.h.max(f32::EPSILON);
                        // Device-px border box, outward-snapped, clamped
                        // to the target (mirrors the CPU backend's
                        // `paint_backdrop_filter_group`).
                        let x0 = ((bf.rect.x - scroll.0) * sx).floor().max(0.0) as u32;
                        let y0 = ((bf.rect.y - scroll.1) * sy).floor().max(0.0) as u32;
                        let x1 = ((bf.rect.x + bf.rect.w - scroll.0) * sx)
                            .ceil()
                            .min(target_w as f32) as u32;
                        let y1 = ((bf.rect.y + bf.rect.h - scroll.1) * sy)
                            .ceil()
                            .min(target_h as f32) as u32;
                        if x1 <= x0 || y1 <= y0 {
                            continue;
                        }
                        let (w, h) = (x1 - x0, y1 - y0);
                        // End the pass: the capture/filter/draw-back run
                        // on this same encoder, then a fresh pass
                        // continues the frame.
                        drop(pass);
                        // Pooled capture texture (see
                        // `checkout_capture_target`): no fresh GPU texture
                        // per element per frame.
                        let cap = self.checkout_capture_target(w, h, target_tex.format());
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo {
                                texture: target_tex,
                                mip_level: 0,
                                origin: wgpu::Origin3d { x: x0, y: y0, z: 0 },
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::TexelCopyTextureInfo {
                                texture: &cap,
                                mip_level: 0,
                                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::Extent3d {
                                width: w,
                                height: h,
                                depth_or_array_layers: 1,
                            },
                        );
                        let filtered = self
                            .apply_filter_passes(encoder, cap, &bf.passes, sx, frame_bg, &pipes);
                        // Draw the filtered capture back over its region
                        // with the clip-aware composite pipeline.
                        let mut instance = [0u8; COMPOSITE_STRIDE];
                        Self::write_composite_instance(
                            &mut instance,
                            (
                                scroll.0 + x0 as f32 / sx,
                                scroll.1 + y0 as f32 / sy,
                                w as f32 / sx,
                                h as f32 / sy,
                            ),
                            1.0,
                            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        );
                        let ibuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("bowser-backdrop-writeback-instances"),
                            size: COMPOSITE_STRIDE as u64,
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        self.queue.write_buffer(&ibuf, 0, &instance);
                        let fview = filtered.create_view(&wgpu::TextureViewDescriptor::default());
                        let fbg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("bowser-backdrop-writeback-bg"),
                            layout: &self.tex_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(&fview),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                                },
                            ],
                        });
                        // Load, not clear: the draw-back composites over
                        // the target's current content.
                        pass = Self::begin_pass_load(encoder, view);
                        if let Some((x, y, w, h)) = last_scissor {
                            pass.set_scissor_rect(x, y, w, h);
                        }
                        pass.set_pipeline(&pipes.composite_clip);
                        pass.set_bind_group(0, frame_bg, &[]);
                        pass.set_bind_group(1, &fbg, &[]);
                        // The current clip state: the fragment multiplies
                        // by its exact coverage.
                        pass.set_bind_group(2, &clip_bg, &[]);
                        pass.set_vertex_buffer(0, ibuf.slice(..));
                        pass.draw(0..4, 0..1);
                    }
                }
            }
        }
    }

    /// Check out a bounds-sized offscreen texture for an opacity group
    /// (see [`lowered_bounds`]), reusing an exact-size pooled texture
    /// when one is free instead of allocating a fresh texture per group
    /// per frame. The texture stays checked out until the frame's final
    /// submit (see the `group_target_pool` field docs); `begin_pass`
    /// clears on load, so a reused texture always starts clean.
    fn checkout_group_target(&mut self, w: u32, h: u32) -> wgpu::Texture {
        if let Some(idx) = self
            .group_target_pool
            .iter()
            .position(|t| t.w == w && t.h == h)
        {
            let target = self.group_target_pool.swap_remove(idx);
            let texture = target.texture.clone();
            self.group_targets_in_use.push(target);
            return texture;
        }
        let texture = self.make_target(w, h);
        self.group_targets_in_use.push(GroupTarget {
            w,
            h,
            texture: texture.clone(),
        });
        texture
    }

    /// Check out a backdrop-capture texture for [`Op::BackdropFilter`],
    /// reusing an exact-size, exact-format pooled texture when one is
    /// free instead of allocating a fresh texture per element per frame.
    /// The texture stays checked out until the frame's final submit (see
    /// the `capture_pool` field docs); the capture copy overwrites the
    /// whole region, so a reused texture needs no clear on checkout.
    fn checkout_capture_target(
        &mut self,
        w: u32,
        h: u32,
        format: wgpu::TextureFormat,
    ) -> wgpu::Texture {
        if let Some(idx) = self
            .capture_pool
            .iter()
            .position(|t| t.w == w && t.h == h && t.texture.format() == format)
        {
            let target = self.capture_pool.swap_remove(idx);
            let texture = target.texture.clone();
            self.captures_in_use.push(target);
            return texture;
        }
        let texture = self.make_capture_target(w, h, format);
        self.captures_in_use.push(GroupTarget {
            w,
            h,
            texture: texture.clone(),
        });
        texture
    }

    /// Backdrop-capture texture: sampled and rendered into, so it needs
    /// TEXTURE_BINDING + RENDER_ATTACHMENT on top of COPY_DST, in the
    /// current target's format.
    fn make_capture_target(&self, w: u32, h: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bowser-backdrop-capture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    /// Render an isolated opacity group's lowering into an offscreen
    /// texture; returns a bind group sampling it (group 1 of the composite
    /// pipeline's layout), the texture itself (for the blend-composite
    /// path, which samples it alongside a copied backdrop), and the
    /// texture's device-px origin in the current target. The bind group
    /// keeps the texture and its view alive. The texture is sized to the
    /// group's paint bounds (see [`lowered_bounds`]), not the full
    /// target, and comes from the group-target pool
    /// ([`WgpuBackend::checkout_group_target`]) — one allocation per
    /// distinct group size, not one per group per frame. Nested groups
    /// recurse through [`WgpuBackend::execute_ops_into`].
    ///
    /// Records into the frame's `encoder` and never submits: the whole
    /// frame — groups included — is one encoder and one submit (see
    /// [`WgpuBackend::execute_ops_into`]). Group passes are fully
    /// recorded (and ended) before the parent pass resumes, so command
    /// order matches the old per-group-encoder behavior; `prepared`
    /// addresses this group's disjoint slice of the frame's combined
    /// instance buffers.
    ///
    /// The texture covers the group's paint bounds snapped outward to
    /// device px (floor the origin, ceil the extent), mirroring the CPU
    /// backend's [`checkout_bounded_group_pixmap`](crate::cpu::CpuBackend::checkout_bounded_group_pixmap):
    /// page px map to group device px at the true `sx`/`sy` of the
    /// current target, with the integral snapped origin subtracted, so
    /// both backends rasterize the identical pixel grid. (Mapping the
    /// fractional page-px bounds onto the whole texture instead would
    /// rasterize at `tw / bounds.w` device px per page px — wrong
    /// whenever `bounds.w * sx` is fractional.)
    // 9 params, all used; bundling would only add indirection.
    #[allow(clippy::too_many_arguments)]
    fn render_group_texture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        group: &OpacityGroup,
        prepared: &Prepared,
        bufs: &[Option<wgpu::Buffer>; 5],
        viewport: Size,
        scroll: (f32, f32),
        target_w: u32,
        target_h: u32,
    ) -> (wgpu::BindGroup, wgpu::Texture, f32, f32) {
        // True device px per page px of the current target (the guards
        // are defensive — `render_into` never passes a zero viewport).
        let sx = target_w as f32 / viewport.w.max(f32::EPSILON);
        let sy = target_h as f32 / viewport.h.max(f32::EPSILON);
        // Outward-snapped device-px bounds of the group in the current
        // target: the texture must cover every painted device pixel,
        // and its origin must be integral so the group's pixel grid
        // aligns with the current target's (and the CPU's).
        let gx0 = ((group.bounds.x - scroll.0) * sx).floor();
        let gy0 = ((group.bounds.y - scroll.1) * sy).floor();
        let gx1 = ((group.bounds.x + group.bounds.w - scroll.0) * sx).ceil();
        let gy1 = ((group.bounds.y + group.bounds.h - scroll.1) * sy).ceil();
        let max_dim = self.device.limits().max_texture_dimension_2d;
        let tw = (gx1 - gx0).max(1.0).min(max_dim as f32) as u32;
        let th = (gy1 - gy0).max(1.0).min(max_dim as f32) as u32;
        let target = self.checkout_group_target(tw, th);
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        // Group-local frame uniform: the vertex shader maps page px to
        // NDC through (viewport_size, scroll) as (w, h, x, y). A page
        // point `p` must land on group device px
        // `(p - scroll) * s - (gx0, gy0)`, i.e. the uniform carries
        // viewport `(tw / sx, th / sy)` and scroll
        // `scroll + (gx0 / sx, gy0 / sy)` — page→device at the true
        // `sx`/`sy` with the integral snapped origin. A throwaway 16-byte
        // uniform plus one bind group per group render; groups are capped
        // per frame (`MAX_OPACITY_GROUPS`) and the uniform dies with the
        // bind group after submit.
        let mut uniform = [0u8; 16];
        for (i, v) in [
            tw as f32 / sx,
            th as f32 / sy,
            scroll.0 + gx0 / sx,
            scroll.1 + gy0 / sy,
        ]
        .iter()
        .enumerate()
        {
            uniform[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bowser-group-uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&uniform_buf, 0, &uniform);
        let frame_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bowser-group-frame-bg"),
            layout: &self.frame_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        self.execute_ops_into(
            encoder,
            &view,
            &target,
            wgpu::TextureFormat::Rgba8Unorm,
            &group.lowered,
            prepared,
            bufs,
            Size {
                w: tw as f32 / sx,
                h: th as f32 / sy,
            },
            (scroll.0 + gx0 / sx, scroll.1 + gy0 / sy),
            tw,
            th,
            &frame_bg,
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bowser-group-bg"),
            layout: &self.tex_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        // The bind group keeps the texture alive; the texture itself is
        // also returned for the blend-composite path, which samples it
        // alongside a copied backdrop. `gx0`/`gy0` are the texture's
        // device-px origin in the current target: the composite quad and
        // the blend backdrop copy are positioned from them.
        (bind_group, target, gx0, gy0)
    }

    /// Build the 112-byte [`FilterUniform`] for a color-matrix pass
    /// (see `SHADER_FILTER`): the 4x5 matrix as four `vec4` rows plus a
    /// `vec4` constant offset. `premult_direct` selects the
    /// premultiplied-direct variant (the drop-shadow tint matrix runs
    /// on the stored premultiplied bytes with no (un)premultiply
    /// round-trip).
    fn filter_color_uniform(m: &ColorMatrix, premult_direct: bool) -> [u8; FILTER_UNIFORM_BYTES] {
        let mut u = [0u8; FILTER_UNIFORM_BYTES];
        for (i, row) in m.m.iter().enumerate() {
            for (j, v) in row.iter().enumerate().take(4) {
                u[(i * 4 + j) * 4..(i * 4 + j + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
            // Column 4 is the constant offset.
            u[64 + i * 4..64 + (i + 1) * 4].copy_from_slice(&row[4].to_le_bytes());
        }
        u[96..100].copy_from_slice(&0u32.to_le_bytes()); // mode = color
        u[100..104].copy_from_slice(&u32::from(premult_direct).to_le_bytes()); // flags
        u
    }

    /// Build the [`FilterUniform`] for the drop-shadow tint step: a
    /// premultiplied-direct color matrix taking the source alpha to the
    /// shadow color (`R' = Cs.r * Cs.a * A`, …), mirroring the CPU's
    /// tint loop in
    /// [`apply_drop_shadow`](crate::filter::apply_drop_shadow).
    fn filter_tint_uniform(color: Color) -> [u8; FILTER_UNIFORM_BYTES] {
        let (cr, cg, cb, ca) = (
            f32::from(color.r8()) / 255.0,
            f32::from(color.g8()) / 255.0,
            f32::from(color.b8()) / 255.0,
            f32::from(color.a8()) / 255.0,
        );
        let mut m = ColorMatrix::identity();
        m.m[0] = [0.0, 0.0, 0.0, cr * ca, 0.0];
        m.m[1] = [0.0, 0.0, 0.0, cg * ca, 0.0];
        m.m[2] = [0.0, 0.0, 0.0, cb * ca, 0.0];
        m.m[3] = [0.0, 0.0, 0.0, ca, 0.0];
        Self::filter_color_uniform(&m, true)
    }

    /// Upper bound on the GPU box-blur half-width, in texels. The
    /// fragment shader taps `2r+1` texels per pixel per pass (6 passes
    /// per blur), so an uncapped radius is a TDR/device-loss risk:
    /// `blur(1000px)` on a 1920x1080 surface would be ~2001 taps/px x
    /// 2M px x 6 passes ~= 24G taps. The CPU backend is O(1)/pixel via
    /// a sliding window and caps only at `max(w, h)` (see
    /// `clamp_blur_radius`), so this is a DELIBERATE divergence: for
    /// radii above 256 the GPU renders a narrower (sharper) blur than
    /// the CPU. 256 still covers every realistic `blur()` radius
    /// (r ~= sigma ~= radius/2 in px); beyond it both backends produce
    /// "very blurry", and the GPU stays TDR-safe (worst case 6 x 513
    /// taps/px).
    ///
    /// Rejected alternatives: (a) a sliding-window running sum in the
    /// shader is impossible in an independent per-pixel fragment
    /// shader (it needs the previous pixel's sum, i.e. this pass's own
    /// output); the exact O(1) alternatives (a compute-shader
    /// per-row/per-column sliding window, or a prefix-sum reduction)
    /// need new pipelines and storage-texture plumbing that cannot be
    /// validated without a Vulkan adapter. (b) A "constant image mean"
    /// fast path for `r >= max(w, h)` would be WRONG under
    /// clamp-to-edge: output pixel x is
    /// `((r-x)*t0 + (r-(w-1-x))*t(w-1) + row_sum) / (2r+1)`, which
    /// still varies with x (it converges to the edge-texel average,
    /// not the image mean), so it would diverge from the CPU by tens
    /// of LSBs on content with bright/dark edges.
    const GPU_BLUR_RADIUS_CAP: f32 = 256.0;

    /// Build the [`FilterUniform`] for one axis of the separable box
    /// blur: `radius_texels` is the integer box half-width in texels,
    /// `dir` 0 = horizontal, 1 = vertical.
    ///
    /// The radius is clamped to the texture's larger dimension (like
    /// the CPU's `clamp_blur_radius`) and to
    /// [`GPU_BLUR_RADIUS_CAP`](Self::GPU_BLUR_RADIUS_CAP) (unlike the
    /// CPU: a deliberate perf-vs-fidelity tradeoff, documented above).
    /// The CPU's extra u32-safety cap never binds for GPU-sized
    /// textures.
    fn filter_blur_uniform(
        w: u32,
        h: u32,
        radius_texels: f32,
        dir: u32,
    ) -> [u8; FILTER_UNIFORM_BYTES] {
        let radius_texels = radius_texels
            .min(w.max(h) as f32)
            .min(Self::GPU_BLUR_RADIUS_CAP);
        let mut u = [0u8; FILTER_UNIFORM_BYTES];
        u[80..84].copy_from_slice(&(1.0 / w as f32).to_le_bytes());
        u[84..88].copy_from_slice(&(1.0 / h as f32).to_le_bytes());
        u[88..92].copy_from_slice(&radius_texels.to_le_bytes());
        u[92..96].copy_from_slice(&dir.to_le_bytes());
        u[96..100].copy_from_slice(&1u32.to_le_bytes()); // mode = blur
        u
    }

    /// Build one filter-uniform ring buffer plus its bind group: `slots`
    /// slots of `stride` bytes each, bound once with a dynamic offset so
    /// each pass selects its slot at `set_bind_group` time.
    fn make_filter_uniform_ring(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        stride: u64,
        slots: u32,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bowser-filter-uniform-ring"),
            size: stride * u64::from(slots),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bowser-filter-uniform-ring-bg"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(FILTER_UNIFORM_BYTES as u64),
                }),
            }],
        });
        (buf, bg)
    }

    /// Double the filter-uniform ring when a frame claims more slots than
    /// it holds. The retired buffer's bind group is parked in
    /// `filter_transients`: earlier passes this frame already recorded
    /// `set_bind_group` against it, and dropping it before the submit
    /// would destroy their uniforms.
    fn grow_filter_uniform_ring(&mut self) {
        let (buf, bg) = Self::make_filter_uniform_ring(
            &self.device,
            &self.filter_uniform_layout,
            self.filter_uniform_stride,
            self.filter_uniform_slots * 2,
        );
        let old_bg = std::mem::replace(&mut self.filter_uniform_bg, bg);
        self.filter_uniform_buf = buf;
        self.filter_uniform_slots *= 2;
        self.filter_transients.push(old_bg);
    }

    /// One filter render pass: draw the fullscreen triangle sampling
    /// `src_tex` through the filter pipeline with `uniform` params into
    /// `dst_tex` (REPLACE, no blending, no clipping — the group texture
    /// was already clipped when it rendered). The uniform goes into one
    /// slot of the persistent ring (see the `filter_uniform_*` field
    /// docs) instead of a fresh buffer per pass: the write is enqueued
    /// before this pass's draw, and the dynamic offset keeps the draw
    /// reading its own slot even though later passes claim later slots
    /// before the frame submits.
    fn run_filter_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        src_tex: &wgpu::Texture,
        dst_tex: &wgpu::Texture,
        uniform: &[u8; FILTER_UNIFORM_BYTES],
        frame_bg: &wgpu::BindGroup,
        pipes: &Pipelines,
    ) {
        if self.filter_uniform_next == self.filter_uniform_slots {
            self.grow_filter_uniform_ring();
        }
        let slot = self.filter_uniform_next;
        self.filter_uniform_next += 1;
        let dyn_offset = (u64::from(slot) * self.filter_uniform_stride) as u32;
        self.queue
            .write_buffer(&self.filter_uniform_buf, u64::from(dyn_offset), uniform);
        let view = src_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let tex_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bowser-filter-tex-bg"),
            layout: &self.tex_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let dst_view = dst_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut pass = Self::begin_pass(encoder, &dst_view);
        pass.set_pipeline(&pipes.filter);
        pass.set_bind_group(0, frame_bg, &[]);
        pass.set_bind_group(1, &tex_bg, &[]);
        pass.set_bind_group(2, &self.filter_uniform_bg, &[dyn_offset]);
        pass.draw(0..3, 0..1);
        // The texture bind group (and its view) must outlive this
        // function: the encoder only records references, and dropping it
        // now would destroy the GPU resources before submit. The uniform
        // bind group is a persistent backend field, so it needs no
        // transient.
        self.filter_transients.push(tex_bg);
    }

    /// Write one composite-pipe instance (see
    /// `composite_instance_layout`): page-px `rect`, full-texture uv,
    /// `alpha`, and the 2D `(a, b, c, d, e, f)` matrix.
    fn write_composite_instance(
        out: &mut [u8],
        rect: (f32, f32, f32, f32),
        alpha: f32,
        matrix: [f32; 6],
    ) {
        debug_assert_eq!(out.len(), COMPOSITE_STRIDE);
        for (i, v) in [rect.0, rect.1, rect.2, rect.3].iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in [0.0f32, 0.0, 1.0, 1.0].iter().enumerate() {
            out[(4 + i) * 4..(5 + i) * 4].copy_from_slice(&v.to_le_bytes());
        }
        out[32..36].copy_from_slice(&alpha.to_le_bytes());
        out[36..40].copy_from_slice(&0u32.to_le_bytes()); // Normal
        for (i, v) in matrix.iter().enumerate() {
            let o = 40 + i * 4;
            out[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
    }

    /// Merge a drop-shadow pass: draw `shadow_tex` translated by
    /// (`dx_texels`, `dy_texels`) device texels, then `content_tex` at
    /// identity, into `dst_tex` (cleared first). All three are the same
    /// device-px size; a throwaway frame uniform maps texels 1:1 so the
    /// two draws align exactly, and the composite pipe source-overs the
    /// content over the shadow — the same `over()` the CPU's
    /// [`apply_drop_shadow`](crate::filter::apply_drop_shadow) computes.
    fn merge_drop_shadow(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        shadow_tex: &wgpu::Texture,
        content_tex: &wgpu::Texture,
        dst_tex: &wgpu::Texture,
        offset_texels: (f32, f32),
        pipes: &Pipelines,
    ) {
        let (w, h) = (dst_tex.width() as f32, dst_tex.height() as f32);
        let (dx_texels, dy_texels) = offset_texels;
        let mut funiform = [0u8; 16];
        for (i, v) in [w, h, 0.0f32, 0.0].iter().enumerate() {
            funiform[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        let fbuf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bowser-filter-merge-frame"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&fbuf, 0, &funiform);
        let fbg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bowser-filter-merge-frame-bg"),
            layout: &self.frame_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: fbuf.as_entire_binding(),
            }],
        });
        let mut instances = [0u8; COMPOSITE_STRIDE * 2];
        Self::write_composite_instance(
            &mut instances[..COMPOSITE_STRIDE],
            (0.0, 0.0, w, h),
            1.0,
            [1.0, 0.0, 0.0, 1.0, dx_texels, dy_texels],
        );
        Self::write_composite_instance(
            &mut instances[COMPOSITE_STRIDE..],
            (0.0, 0.0, w, h),
            1.0,
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );
        let ibuf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bowser-filter-merge-instances"),
            size: (COMPOSITE_STRIDE * 2) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&ibuf, 0, &instances);
        let tex_bg = |label: &str, tex: &wgpu::Texture| {
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.tex_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        };
        let shadow_bg = tex_bg("bowser-filter-merge-shadow-bg", shadow_tex);
        let content_bg = tex_bg("bowser-filter-merge-content-bg", content_tex);
        let dst_view = dst_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut pass = Self::begin_pass(encoder, &dst_view);
        pass.set_pipeline(&pipes.composite);
        pass.set_bind_group(0, &fbg, &[]);
        // No clip on the merge: both sources were already clipped when
        // the group texture rendered.
        pass.set_bind_group(2, &self.clip_bind_group(None), &[]);
        pass.set_bind_group(1, &shadow_bg, &[]);
        pass.set_vertex_buffer(0, ibuf.slice(..COMPOSITE_STRIDE as u64));
        pass.draw(0..4, 0..1);
        pass.set_bind_group(1, &content_bg, &[]);
        pass.set_vertex_buffer(0, ibuf.slice(COMPOSITE_STRIDE as u64..));
        pass.draw(0..4, 0..1);
    }

    /// Run `passes` over `src` (an offscreen group/capture texture),
    /// returning the texture holding the result. Ping-pongs between two
    /// group-pool targets; `src` and the unused target recycle at frame
    /// end with every other group target. `sx` is device px per page px:
    /// the passes' lengths are CSS px, scaled here exactly like the CPU
    /// backend's [`apply_filter_passes`](crate::cpu::Painter::apply_filter_passes).
    fn apply_filter_passes(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        src: wgpu::Texture,
        passes: &[FilterPass],
        sx: f32,
        frame_bg: &wgpu::BindGroup,
        pipes: &Pipelines,
    ) -> wgpu::Texture {
        debug_assert!(!passes.is_empty());
        let (w, h) = (src.width(), src.height());
        let ta = self.checkout_group_target(w, h);
        let tb = self.checkout_group_target(w, h);
        // Ping-pong between `ta`/`tb`; the source stays read-only.
        // `rotate` swaps read/write after each pass; on the very first
        // pass the old `read` is the source, which is swapped out for
        // the spare so it never becomes a render target.
        let mut read = src;
        let mut write = ta;
        let mut spare = tb;
        let mut first = true;
        // Local macro to avoid repeating the rotation at 4 sites.
        macro_rules! rotate {
            () => {
                std::mem::swap(&mut read, &mut write);
                if first {
                    std::mem::swap(&mut write, &mut spare);
                    first = false;
                }
            };
        }
        for pass in passes {
            match pass {
                FilterPass::Color(m) => {
                    let u = Self::filter_color_uniform(m, false);
                    self.run_filter_pass(encoder, &read, &write, &u, frame_bg, pipes);
                    rotate!();
                }
                FilterPass::Blur(r) => {
                    // σ = radius / 2, scaled to device px (mirrors the
                    // CPU). Three (H, V) separable box passes approximate
                    // the gaussian, matching `box_blur_sigma_rgba`; the
                    // shader implements one axis per pass (see
                    // SHADER_FILTER).
                    //
                    // LAVAPIPE BARRIER BUG (UNRESOLVED): blur was
                    // previously a passthrough because "multi-pass hits
                    // a lavapipe barrier bug". This executor is now 6
                    // dependent fullscreen render passes ping-ponging
                    // between two group-pool targets (plus tint/merge
                    // passes for drop-shadow), and the barrier concern
                    // has NOT been cleared: this container has no Vulkan
                    // adapter, so the multi-pass hazard was never
                    // reproduced or ruled out here. The CPU/GPU agreement
                    // tests below must be re-run under lavapipe
                    // (VK_ICD_FILENAMES -> lvp_icd.json) before their
                    // tolerances can be trusted; do not delete this
                    // comment on a passing run elsewhere without
                    // confirming which driver cleared it.
                    let radius = blur_box_radius(r * sx / 2.0);
                    if radius >= 1 {
                        for _ in 0..3 {
                            for dir in [0u32, 1u32] {
                                let u = Self::filter_blur_uniform(w, h, radius as f32, dir);
                                self.run_filter_pass(encoder, &read, &write, &u, frame_bg, pipes);
                                rotate!();
                            }
                        }
                    }
                }
                FilterPass::DropShadow {
                    offset,
                    blur,
                    color,
                } => {
                    let content = read.clone();
                    // 1. Tint the source alpha with the shadow color.
                    let u = Self::filter_tint_uniform(*color);
                    self.run_filter_pass(encoder, &read, &write, &u, frame_bg, pipes);
                    rotate!();
                    // 2. Blur the tinted shadow: three (H, V) separable
                    // box passes, like the CPU's `apply_drop_shadow`
                    // (see SHADER_FILTER).
                    let radius = blur_box_radius(blur * sx / 2.0);
                    if radius >= 1 {
                        for _ in 0..3 {
                            for dir in [0u32, 1u32] {
                                let u = Self::filter_blur_uniform(w, h, radius as f32, dir);
                                self.run_filter_pass(encoder, &read, &write, &u, frame_bg, pipes);
                                rotate!();
                            }
                        }
                    }
                    // 3. Merge: the shadow at the snapped offset under
                    // the content, into a fresh target (distinct from
                    // shadow, content, and the ping-pong pair). The
                    // offset snaps to whole device texels, exactly
                    // like the CPU's `apply_drop_shadow`.
                    let shadow = read;
                    let merge_dst = self.checkout_group_target(w, h);
                    self.merge_drop_shadow(
                        encoder,
                        &shadow,
                        &content,
                        &merge_dst,
                        ((offset.0 * sx).round(), (offset.1 * sx).round()),
                        pipes,
                    );
                    read = merge_dst;
                    write = shadow;
                }
            }
        }
        read
    }

    /// Reserve the frame's combined per-pipe byte vectors (see
    /// [`WgpuBackend::prepare_lowering`]) from the lowering tree's exact
    /// byte counts, so the flattening pass never reallocates: the
    /// per-lowering vectors were already sized by `Lowered::with_capacity`.
    /// Totals are summed
    /// across the whole tree first: reserving per level would not
    /// accumulate, since every level reserves while the vectors are still
    /// empty.
    fn reserve_combined(lowered: &Lowered, out: &mut [Vec<u8>; 5]) {
        let mut totals = [0usize; 5];
        Self::combined_totals(lowered, &mut totals);
        for (slot, total) in totals.iter().enumerate() {
            out[slot].reserve(*total);
        }
    }

    /// Sum one lowering tree's exact per-pipe byte counts into `totals`
    /// (indexed by [`PipeKind::slot`]). Composite instances are built at
    /// execute time (see `Op::Group` handling in `execute_ops_into`), so
    /// the composite slot always totals zero here.
    fn combined_totals(lowered: &Lowered, totals: &mut [usize; 5]) {
        totals[PipeKind::Solid.slot()] += lowered.solid.len();
        totals[PipeKind::Glyph.slot()] += lowered.glyph.len();
        totals[PipeKind::Image.slot()] += lowered.image.len();
        totals[PipeKind::Shadow.slot()] += lowered.shadow.len();
        // One composite instance slot per group op: the instance bytes
        // are built at execute time (the snapped device rect needs the
        // current scroll and scale), so `prepare_lowering` reserves
        // zeroed slots here and `Op::Group` handling fills its own
        // disjoint slot (see `prepare_lowering`).
        let groups = lowered
            .ops
            .iter()
            .filter(|op| matches!(op, Op::Group(_)))
            .count();
        totals[PipeKind::Composite.slot()] += groups * PipeKind::Composite.stride();
        for group in &lowered.groups {
            Self::combined_totals(&group.lowered, totals);
        }
    }

    /// Flatten one frame's whole lowering tree into combined per-pipe
    /// byte vectors (indexed by [`PipeKind::slot`]), recording each
    /// lowering's base instance offset per pipe in the returned
    /// [`Prepared`] tree. The root's bytes come first, then each group's
    /// in [`Lowered::groups`] order, recursing the same way — so the
    /// [`Prepared`] tree mirrors the lowering tree and every lowering's
    /// draws address disjoint buffer slices.
    ///
    /// No `&self`: pure data shuffling, callable before any upload.
    /// Each group op reserves one zeroed composite instance slot, in op
    /// order (group indices ascend in op order). The instance bytes are
    /// *not* pre-built here: each group op's composite quad covers the
    /// group's snapped device bounds, which are only known at execute
    /// time (see `Op::Group` handling in
    /// [`WgpuBackend::execute_ops_into`]), so the handler writes its own
    /// disjoint slot then.
    fn prepare_lowering(lowered: &Lowered, out: &mut [Vec<u8>; 5]) -> Prepared {
        // One zeroed composite instance slot per group op, in op order.
        let mut composite_bytes = Vec::new();
        for op in &lowered.ops {
            if let Op::Group(_) = op {
                composite_bytes.extend_from_slice(&[0u8; COMPOSITE_STRIDE]);
            }
        }
        let pipe_bytes: [&[u8]; 5] = [
            &lowered.solid,
            &lowered.glyph,
            &lowered.image,
            &composite_bytes,
            &lowered.shadow,
        ];
        let mut bases = [0u32; 5];
        for (slot, bytes) in pipe_bytes.iter().enumerate() {
            let stride = PipeKind::from_slot(slot).stride();
            debug_assert_eq!(bytes.len() % stride, 0);
            debug_assert_eq!(out[slot].len() % stride, 0);
            bases[slot] = (out[slot].len() / stride) as u32;
            out[slot].extend_from_slice(bytes);
        }
        let groups = lowered
            .groups
            .iter()
            .map(|group| Self::prepare_lowering(&group.lowered, out))
            .collect();
        Prepared { bases, groups }
    }

    /// Upload a frame's combined instance bytes (see
    /// [`WgpuBackend::prepare_lowering`]) into the persistent per-pipe
    /// buffers, growing them when a frame outgrows the capacity; returns
    /// the current buffer handles by [`PipeKind::slot`]. One
    /// `queue.write_buffer` per pipe per frame: every lowering's bytes
    /// are uploaded once up front, so group rendering never re-uploads
    /// the parent's bytes and the frame needs no mid-frame submit.
    fn upload_instances(&mut self, combined: &[Vec<u8>; 5]) -> [Option<wgpu::Buffer>; 5] {
        // `let _ =` silences the `#[must_use]` on the returned handle;
        // the handles are fetched below so a group that grew a buffer
        // cannot leave a stale one behind.
        let _ = self.instance_buffer(PipeKind::Solid, &combined[PipeKind::Solid.slot()]);
        let _ = self.instance_buffer(PipeKind::Glyph, &combined[PipeKind::Glyph.slot()]);
        let _ = self.instance_buffer(PipeKind::Image, &combined[PipeKind::Image.slot()]);
        let _ = self.instance_buffer(PipeKind::Composite, &combined[PipeKind::Composite.slot()]);
        let _ = self.instance_buffer(PipeKind::Shadow, &combined[PipeKind::Shadow.slot()]);
        let mut out: [Option<wgpu::Buffer>; 5] = [None, None, None, None, None];
        for (slot, handle) in out.iter_mut().enumerate() {
            *handle = self.instance_bufs[slot].as_ref().map(|(b, _)| b.clone());
        }
        out
    }

    /// Collect the atlas generations referenced by a lowering's glyph
    /// draw batches (recursing into isolated opacity groups). Rollover
    /// count per frame is tiny, so a deduped `Vec` beats a set.
    fn glyph_generations(lowered: &Lowered, out: &mut Vec<u64>) {
        for op in &lowered.ops {
            if let Op::Draw {
                pipe: PipeKind::Glyph,
                generation,
                ..
            } = op
                && !out.contains(generation)
            {
                out.push(*generation);
            }
        }
        for group in &lowered.groups {
            Self::glyph_generations(&group.lowered, out);
        }
    }

    /// Lower a display list to GPU instance data, honoring clip and
    /// opacity stacks.
    ///
    /// `scale` is device px per page px, per axis, for the frame being
    /// lowered. Geometry needs no scale here (the vertex shader maps page
    /// px onto the device target), but text rasterizes at device px to
    /// stay crisp — see [`emit_shaped_text`]. Non-finite or non-positive
    /// values fall back to 1.0, per axis.
    fn lower(&mut self, list: &DisplayList, scale: (f32, f32)) -> Lowered {
        let sanitize = |s: f32| {
            if s.is_finite() && s > 0.0 { s } else { 1.0 }
        };
        let scale = (sanitize(scale.0), sanitize(scale.1));
        // Bump the gradient-cache generation: entries this frame's
        // lowering references get stamped with it (see
        // `ensure_gradient_texture`), and the stale ones are evicted
        // below, bounding the cache at the frame's distinct gradients.
        self.gradient_frame = self.gradient_frame.wrapping_add(1);
        // Bound the image cache once per frame, before lowering (never
        // mid-frame). First collect every image id the frame references —
        // opacity groups lower from item ranges of this same list, so one
        // scan over `list.items` covers them. Every referenced id must
        // survive lowering AND execution (`execute_ops_into` binds the
        // texture by id at draw time), so referenced entries are pinned
        // for the whole frame and can never be evicted mid-frame. Entries
        // for ids the frame does not reference are stale (ids are only
        // unique within one DisplayList) and are evicted now, before any
        // op is lowered. After this retain the cache holds at most the
        // frame's distinct referenced images: cross-frame accumulation is
        // impossible, hot entries for still-visible images survive (no
        // re-upload churn like the old clear-at-capacity), and nothing an
        // already-lowered op needs can be evicted. A mid-frame clear here
        // used to wipe textures for image ops already lowered and panic
        // at draw time ("image texture missing for draw") once the 513th
        // distinct image uploaded.
        //
        // Byte accounting lives in [`ImageCache`]: the retain below goes
        // through [`ImageCache::retain_referenced`] so `total_bytes`
        // stays exact, and every entry starts the frame with
        // `lowered_this_frame` unset
        // ([`ImageCache::clear_frame_flags`]); [`ensure_image`] sets it
        // as lowering references entries, [`ImageCache::reserve`] keeps
        // each upload within the byte budget by evicting only unpinned
        // entries mid-frame, and `render_into` trims the excess after the
        // frame submits.
        let mut referenced: HashSet<ImageId> = HashSet::new();
        for item in &list.items {
            match item {
                DisplayItem::Image { image, .. }
                | DisplayItem::BackgroundImage { image, .. }
                | DisplayItem::RepeatImage { image, .. } => {
                    referenced.insert(*image);
                }
                _ => {}
            }
        }
        self.image_cache.clear_frame_flags();
        self.image_cache.retain_referenced(&referenced);
        // Split borrows: atlas/image-cache/sampler/device/queue are used
        // while the accumulator builds; the parts struct is threaded
        // through so opacity groups can lower recursively.
        //
        // Highest atlas generation touched while lowering: a rollover
        // installs a fresh texture, and the glyph bind group must follow
        // it (see `rebuild_glyph_bind_group`). Seeded from the stored
        // generation so a text-free frame cannot regress it; the
        // generation only increases, so the max over observed cells is
        // the current generation.
        let mut atlas_generation = self.glyph_bind_group_generation;
        let lowered = {
            let mut parts = LowerParts::of(self, &mut atlas_generation, scale);
            // Match every opacity group once up front (O(items)); group
            // lowering below is then pure lookup, never a re-scan.
            parts.group_scans = precompute_group_scans(&list.items);
            Self::lower_range(&mut parts, list, 0, list.items.len(), Vec::new(), 1.0)
        };
        if atlas_generation != self.glyph_bind_group_generation {
            self.glyph_bind_group_generation = atlas_generation;
            self.rebuild_glyph_bind_group();
        }
        // Quads lowered before a mid-frame atlas rollover reference the
        // retired page's cells: each such generation needs its own bind
        // group sampling the retired page's texture (see
        // `retired_glyph_bind_groups`). Collect every generation the
        // frame's glyph batches reference, including nested groups.
        let mut generations = Vec::new();
        Self::glyph_generations(&lowered, &mut generations);
        for generation in generations {
            if generation == self.glyph_bind_group_generation
                || self.retired_glyph_bind_groups.contains_key(&generation)
            {
                continue;
            }
            if let Some(bind_group) = self.build_glyph_bind_group_for(generation) {
                self.retired_glyph_bind_groups
                    .insert(generation, bind_group);
            }
        }
        // Evict gradient textures the frame did not reference: ids are
        // content-hashed, so unlike image ids they stay valid across
        // frames, and only the frame's live set is retained.
        let frame = self.gradient_frame;
        self.gradient_cache.retain(|_, e| e.last_frame == frame);
        lowered
    }

    /// Lower one [`DisplayItem::RepeatImage`] to a single tiled image
    /// quad.
    ///
    /// The area is clamped to [`MAX_REPEAT_AREA_DIM`] per axis (the
    /// hostile-input bound, shared with the CPU backend via
    /// [`clamp_repeat_area`]) and the tile is enlarged via
    /// [`cap_repeat_tile`] (likewise shared with the CPU backend, so a
    /// hand-built item with a hostile tiny tile renders the same coarse
    /// grid on both), and the draw is scissored to the area
    /// intersected with the active clip, so partial edge tiles trim
    /// exactly like the old per-tile expansion's `PushClip`/`PopClip`
    /// pair. UVs span the drawn quad in tile units from the positioned
    /// origin; the fragment shader wraps/discards per axis per the
    /// repeat mode (see `SHADER_IMAGE`), so one draw covers any tile
    /// count. The image is uploaded once, so the 256 MiB texture budget
    /// accounts exactly one texture per image regardless of tile count.
    #[allow(clippy::too_many_arguments)]
    fn lower_repeat_image(
        out: &mut Lowered,
        clips: &mut Vec<PageClip>,
        area: Rect,
        image: ImageId,
        tile: (f32, f32),
        repeat: BackgroundRepeat,
        offset: (f32, f32),
        opacity: f32,
    ) {
        let (tw, th) = tile;
        if tw <= 0.0
            || th <= 0.0
            || !tw.is_finite()
            || !th.is_finite()
            || !offset.0.is_finite()
            || !offset.1.is_finite()
        {
            return;
        }
        let area = clamp_repeat_area(area);
        if area.w <= 0.0 || area.h <= 0.0 {
            return;
        }
        // Enlarge a hostile tiny tile exactly like the CPU backend (see
        // [`cap_repeat_tile`]): a hand-built `RepeatImage` bypasses the
        // builder's cap, and without this the GPU renders un-enlarged
        // tiles while the CPU renders enlarged ones — a full divergence,
        // not a 1-ulp wobble. A no-op once the tile count is within
        // budget (so `NoRepeat`'s single tile is unaffected).
        let (tw, th) = cap_repeat_tile((tw, th), area);
        // Scissor to the area, restored afterwards; `set_scissor`
        // snapshots the stack (all-rect stacks collapse to their exact
        // bound), so a redundant area push emits no op and the
        // surrounding draws stay mergeable into one batch.
        clips.push(PageClip::rect(area));
        out.set_scissor(clips);

        // Axis repeat modes for the shader: 0 = single copy, 1 = wrap,
        // 2 = spaced.
        const ONCE: f32 = 0.0;
        const WRAP: f32 = 1.0;
        const SPACED: f32 = 2.0;
        let (x0, y0) = (area.x + offset.0, area.y + offset.1);
        // UV span in tile units across the drawn quad from an origin.
        // The far edge is the true far-edge coordinate
        // `(p0 + len - o) / t`, not `u0 + len / t`: the rounded addition
        // differs from the true coordinate by 1 ulp in the general case,
        // systematically tilting the whole interpolation ramp.
        let uv_span = |p0: f32, len: f32, o: f32, t: f32| [(p0 - o) / t, (p0 + len - o) / t];
        match repeat {
            BackgroundRepeat::NoRepeat => {
                // One copy at the positioned origin; the scissor trims
                // overhang past the area, exactly like the old clip pair.
                out.image_quad(
                    Rect {
                        x: x0,
                        y: y0,
                        w: tw,
                        h: th,
                    },
                    image,
                    opacity,
                );
            }
            BackgroundRepeat::Repeat
            | BackgroundRepeat::RepeatX
            | BackgroundRepeat::RepeatY
            | BackgroundRepeat::Round => {
                // The builder normalizes `Round` to `Repeat` at emit
                // time; a hand-built item carrying `Round` tiles like
                // `Repeat` here too.
                let (mx, my) = match repeat {
                    BackgroundRepeat::Repeat | BackgroundRepeat::Round => (WRAP, WRAP),
                    BackgroundRepeat::RepeatX => (WRAP, ONCE),
                    BackgroundRepeat::RepeatY => (ONCE, WRAP),
                    _ => (WRAP, WRAP),
                };
                let [u0, u1] = uv_span(area.x, area.w, x0, tw);
                let [v0, v1] = uv_span(area.y, area.h, y0, th);
                // Wrap-mode tile-boundary snap band per axis, passed via
                // `counts` (the wrap branch uses it as `n`; the spaced
                // branch still gets its tile counts below). See
                // [`repeat_snap_band`]: the band is 4 ulps of the
                // interpolated tile-unit magnitude (`max(|u0|, |u1|)`),
                // because the f32 interpolation residue scales with that
                // magnitude — the `u0` offset included — not with the
                // span. A span-based band let boundary fragments escape
                // the snap under large `background-position` offsets
                // (the GPU sampled the next tile's first texel while the
                // CPU sampled the left tile's last).
                out.image_repeat_quad(
                    area,
                    [u0, v0, u1, v1],
                    (mx, my),
                    (0.0, 0.0),
                    (repeat_snap_band(u0, u1), repeat_snap_band(v0, v1)),
                    image,
                    opacity,
                );
            }
            BackgroundRepeat::Space => {
                let (nx, sx0, px) = background_space_axis(area.x, area.w, tw, offset.0);
                let (ny, sy0, py) = background_space_axis(area.y, area.h, th, offset.1);
                let [u0, u1] = uv_span(area.x, area.w, sx0, tw);
                let [v0, v1] = uv_span(area.y, area.h, sy0, th);
                out.image_repeat_quad(
                    area,
                    [u0, v0, u1, v1],
                    (SPACED, SPACED),
                    (px / tw, py / th),
                    (nx as f32, ny as f32),
                    image,
                    opacity,
                );
            }
        }

        clips.pop();
        out.set_scissor(clips);
    }

    /// Lower items `[start, end)` to GPU instance data.
    ///
    /// `initial_clips` seeds the clip stack (isolated opacity groups
    /// inherit the clip active at the `PushOpacity`, rounded entries
    /// included); `opacity` seeds the
    /// running per-item opacity product (1.0 at the top level, and — since
    /// only foldable single-item groups without nested groups take the
    /// fold fast path — also 1.0 at every isolated group, so group items
    /// lower at their own alphas and the group opacity applies at
    /// composite time).
    fn lower_range(
        parts: &mut LowerParts,
        list: &DisplayList,
        start: usize,
        end: usize,
        initial_clips: Vec<PageClip>,
        opacity: f32,
    ) -> Lowered {
        let mut out = Lowered::with_capacity(end.saturating_sub(start));
        let mut clips = initial_clips;
        // Groups inherit the active clip: the offscreen group texture is
        // the same size as the target, so the group's own lowering must
        // start with the same scissor the main stream has here.
        out.set_scissor(&clips);
        let mut opacity = opacity;
        let mut opacity_stack: Vec<f32> = Vec::new();

        let mut i = start;
        while i < end {
            match &list.items[i] {
                DisplayItem::FillRect { rect, color } => {
                    out.solid_quad(*rect, color.with_opacity(opacity));
                }
                DisplayItem::Border {
                    rect,
                    widths,
                    styles,
                    colors,
                    radii,
                } => {
                    let quads = if radii.iter().any(|r| *r > 0.0) {
                        tessellate_border_rounded(*rect, *widths, *styles, *colors, *radii)
                    } else {
                        tessellate_border(*rect, *widths, *styles, *colors)
                    };
                    for quad in quads {
                        out.solid_quad(quad.rect, quad.color.with_opacity(opacity));
                    }
                }
                DisplayItem::TextRun {
                    text,
                    text_range,
                    rect,
                    baseline,
                    font_key,
                    size_px,
                    color,
                } => {
                    let color = color.with_opacity(opacity);
                    // `text` is the originating box's whole text, shared
                    // across its fragments: shape only this fragment's
                    // verbatim slice.
                    let slice = text.get(text_range.clone()).unwrap_or("");
                    emit_shaped_text(
                        &mut out,
                        parts.atlas,
                        parts.queue,
                        parts.get_font_data,
                        slice,
                        rect.x,
                        *baseline,
                        *font_key,
                        *size_px,
                        color,
                        parts.atlas_generation,
                        parts.scale,
                    );
                }
                DisplayItem::Image { rect, image } => {
                    if ensure_image(
                        parts.device,
                        parts.queue,
                        parts.sampler,
                        parts.tex_layout,
                        parts.image_cache,
                        list,
                        *image,
                    ) {
                        out.image_quad(*rect, *image, opacity);
                    }
                }
                DisplayItem::BackgroundImage { rect, image } => {
                    if ensure_image(
                        parts.device,
                        parts.queue,
                        parts.sampler,
                        parts.tex_layout,
                        parts.image_cache,
                        list,
                        *image,
                    ) {
                        out.image_quad(*rect, *image, opacity);
                    }
                }
                DisplayItem::RepeatImage {
                    area,
                    image,
                    tile,
                    repeat,
                    offset,
                } => {
                    if ensure_image(
                        parts.device,
                        parts.queue,
                        parts.sampler,
                        parts.tex_layout,
                        parts.image_cache,
                        list,
                        *image,
                    ) {
                        Self::lower_repeat_image(
                            &mut out, &mut clips, *area, *image, *tile, *repeat, *offset, opacity,
                        );
                    }
                }
                DisplayItem::Gradient { rect, gradient } => {
                    // Bake once per distinct (gradient, size) and draw
                    // one stretched image-pipe quad sampling the baked
                    // texture (see `ensure_gradient_texture`).
                    if let Some(tex) = ensure_gradient_texture(
                        parts.device,
                        parts.queue,
                        parts.sampler,
                        parts.tex_layout,
                        parts.gradient_cache,
                        parts.gradient_frame,
                        gradient,
                        rect,
                        parts.scale,
                    ) {
                        out.gradient_quad(*rect, tex, opacity);
                    }
                }
                DisplayItem::Decoration {
                    rect,
                    kind,
                    color,
                    thickness,
                } => {
                    let y = match kind {
                        DecoKind::Underline | DecoKind::WavyUnderline => {
                            rect.y + rect.h - thickness
                        }
                        DecoKind::Overline | DecoKind::WavyOverline => rect.y,
                        DecoKind::LineThrough | DecoKind::WavyLineThrough => {
                            rect.y + rect.h * 0.5 - thickness * 0.5
                        }
                    };
                    for quad in decoration_geometry(
                        *kind,
                        rect.x,
                        rect.x + rect.w,
                        y,
                        *thickness,
                        color.with_opacity(opacity),
                    ) {
                        out.solid_quad(quad.rect, quad.color);
                    }
                }
                DisplayItem::ListMarker {
                    rect,
                    kind,
                    color,
                    index,
                    font_key,
                } => {
                    let color = color.with_opacity(opacity);
                    // Decimal markers are text, not geometry: the numeral
                    // comes from the item's 1-based index, formatted into
                    // a stack buffer so the per-frame marker path never
                    // touches the heap.
                    let mut num_buf = [0u8; 20];
                    let owned_marker: Option<String>;
                    let numeral: Option<&str> = match kind {
                        MarkerKind::Decimal(_) => {
                            Some(format_marker_numeral(&mut num_buf, *index, false))
                        }
                        MarkerKind::DecimalLeadingZero(_) => {
                            Some(format_marker_numeral(&mut num_buf, *index, true))
                        }
                        _ => {
                            owned_marker = marker_text(*kind);
                            owned_marker.as_deref()
                        }
                    };
                    if let Some(numeral) = numeral {
                        // Decimal markers are text: shape the numeral inside
                        // the marker box with the item's font (the CPU
                        // backend uses the same key; `FontKey(0)` here
                        // would shape with the wrong face whenever the
                        // item's font is not the default).
                        let size_px = rect.w.min(rect.h).max(1.0);
                        let baseline = rect.y + 0.8 * (rect.h - size_px) + size_px;
                        emit_shaped_text(
                            &mut out,
                            parts.atlas,
                            parts.queue,
                            parts.get_font_data,
                            numeral,
                            rect.x,
                            baseline,
                            *font_key,
                            size_px,
                            color,
                            parts.atlas_generation,
                            parts.scale,
                        );
                    } else {
                        let size = rect.w.min(rect.h);
                        for quad in marker_geometry(
                            *kind,
                            rect.x + rect.w * 0.5,
                            rect.y + rect.h * 0.5,
                            size,
                            color,
                        ) {
                            out.solid_quad(quad.rect, quad.color);
                        }
                    }
                }
                DisplayItem::PushClip { rect } => {
                    clips.push(PageClip::rect(*rect));
                    // Redundant pushes (same snapshot as active) emit no
                    // op, keeping the surrounding draws mergeable.
                    out.set_scissor(&clips);
                }
                DisplayItem::PushRoundedClip { rect, radii } => {
                    clips.push(PageClip {
                        rect: *rect,
                        radii: *radii,
                        poly: [[0.0; 2]; MAX_POLY_VERTS],
                        poly_len: 0,
                    });
                    out.set_scissor(&clips);
                }
                DisplayItem::PushClipPath { points } => {
                    if points.len() >= 3 {
                        // Bounding box in page px for coarse culling
                        // (`rect`); the exact shape lives in `poly`.
                        let mut x0 = f32::INFINITY;
                        let mut y0 = f32::INFINITY;
                        let mut x1 = f32::NEG_INFINITY;
                        let mut y1 = f32::NEG_INFINITY;
                        for p in points.iter() {
                            x0 = x0.min(p[0]);
                            y0 = y0.min(p[1]);
                            x1 = x1.max(p[0]);
                            y1 = y1.max(p[1]);
                        }
                        clips.push(PageClip::polygon(
                            Rect {
                                x: x0,
                                y: y0,
                                w: (x1 - x0).max(0.0),
                                h: (y1 - y0).max(0.0),
                            },
                            points,
                        ));
                    } else {
                        // Degenerate polygon: clips everything.
                        clips.push(PageClip::rect(Rect {
                            x: 0.0,
                            y: 0.0,
                            w: 0.0,
                            h: 0.0,
                        }));
                    }
                    out.set_scissor(&clips);
                }
                DisplayItem::PopClip => {
                    clips.pop();
                    out.set_scissor(&clips);
                }
                DisplayItem::PushOpacity { opacity: o } => {
                    // Clamp each pushed value to [0,1] like the CPU
                    // backend; the running product of [0,1] factors stays
                    // in [0,1]. Non-finite opacity normalizes to 0.0:
                    // `clamp` passes NaN through, and a NaN alpha would
                    // reach the blend equation where it is
                    // driver-undefined.
                    let oc = if o.is_finite() {
                        o.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    if oc <= 0.0 {
                        // Fully transparent group: skip it entirely.
                        let (gend, _, _, _) = parts.group_scan(&list.items, i);
                        debug_assert!(
                            gend < list.items.len(),
                            "PushOpacity without matching PopOpacity"
                        );
                        i = gend.min(end) + 1;
                        continue;
                    }
                    if oc < 1.0 {
                        let (gend, paint_items, nested, single_item) =
                            parts.group_scan(&list.items, i);
                        debug_assert!(
                            gend < list.items.len(),
                            "PushOpacity without matching PopOpacity"
                        );
                        // Fold fast path: only when the single item's
                        // primitives cannot overlap (see
                        // `opacity_foldable`) — a lone TextRun's glyphs
                        // can, and folding would double-apply the opacity
                        // where they do.
                        let foldable =
                            single_item.is_none_or(|idx| opacity_foldable(&list.items[idx]));
                        // Bound isolated groups per frame (see
                        // `MAX_OPACITY_GROUPS`): each pins a bounds-sized
                        // texture until submit (see `lowered_bounds`) and
                        // nested groups recurse in `lower_range`. Past the
                        // cap the opacity folds into the running product — exact only for
                        // non-overlapping primitives, a documented
                        // degradation for pathological pages versus
                        // unbounded VRAM or stack overflow.
                        let over_cap = parts.group_count >= MAX_OPACITY_GROUPS;
                        if over_cap {
                            #[cfg(debug_assertions)]
                            eprintln!(
                                "[bowser_paint] opacity group cap hit; folding group opacity"
                            );
                        }
                        if (paint_items > 1 || nested || !foldable) && !over_cap {
                            parts.group_count += 1;
                            // Isolated group: render the range into an
                            // offscreen texture and composite it with
                            // `oc`. Items lower at their own alphas (the
                            // fold fast path never nests, so `opacity`
                            // here is the group's own product); the group
                            // opacity applies at composite time.
                            let gend = gend.min(end);
                            let group =
                                Self::lower_range(parts, list, i + 1, gend, clips.clone(), opacity);
                            // Size the offscreen texture to the group's
                            // actual paint bounds (see `lowered_bounds`)
                            // instead of the full target: a group covering
                            // a small region of a large target allocates a
                            // small texture. A group that lowered to
                            // nothing paints nothing (compositing
                            // transparent black is a no-op), so skip it
                            // entirely: no group entry, no texture, no
                            // composite quad.
                            let Some(bounds) = lowered_bounds(&group) else {
                                i = gend + 1;
                                continue;
                            };
                            out.ops.push(Op::Group(out.groups.len()));
                            out.groups.push(OpacityGroup {
                                lowered: group,
                                opacity: oc,
                                blend: BlendMode::Normal,
                                bounds,
                                transform: None,
                                filter: Vec::new(),
                            });
                            i = gend + 1;
                            continue;
                        }
                        // Else: fold fast path below — a single *foldable*
                        // paint item (non-overlapping primitives) with no
                        // nested groups: folding `oc` into the item's
                        // alpha is exactly isolated compositing (one
                        // source-over either way).
                    }
                    // Fast path (also a no-op for oc == 1.0): fold into the
                    // running opacity; the matching PopOpacity pops for
                    // balance.
                    opacity_stack.push(opacity);
                    opacity *= oc;
                }
                DisplayItem::PopOpacity => {
                    debug_assert!(
                        !opacity_stack.is_empty(),
                        "unbalanced PopOpacity without PushOpacity"
                    );
                    opacity = opacity_stack.pop().unwrap_or(1.0);
                }
                DisplayItem::PushBlend { mode } => {
                    if mode.is_normal() {
                        // No-op fast path: no isolated group, no
                        // composite; the matching PopBlend is a no-op
                        // below. (A hand-built list may still contain
                        // `Normal` groups; the builder never emits them.)
                        i += 1;
                        continue;
                    }
                    // Blend groups always isolate (unlike opacity, there
                    // is no fold fast path: the mode blends the isolated
                    // group against the backdrop, so per-item blending
                    // would be wrong for multi-item groups — and for a
                    // single item it is identical anyway, so one path
                    // covers both).
                    let gend = scan_blend_group(&list.items, i);
                    debug_assert!(
                        gend < list.items.len(),
                        "PushBlend without matching PopBlend"
                    );
                    let over_cap = parts.group_count >= MAX_OPACITY_GROUPS;
                    if over_cap {
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "[bowser_paint] blend group cap hit; compositing inline as source-over"
                        );
                        // Degradation mirrors the CPU backend: past the
                        // cap the group's items lower inline with plain
                        // source-over.
                        i += 1;
                        continue;
                    }
                    parts.group_count += 1;
                    let gend = gend.min(end);
                    let group = Self::lower_range(parts, list, i + 1, gend, clips.clone(), opacity);
                    // Like opacity groups: a group that lowered to
                    // nothing paints nothing, so skip it entirely.
                    let Some(bounds) = lowered_bounds(&group) else {
                        i = gend + 1;
                        continue;
                    };
                    out.ops.push(Op::Group(out.groups.len()));
                    out.groups.push(OpacityGroup {
                        lowered: group,
                        opacity: 1.0,
                        blend: *mode,
                        bounds,
                        transform: None,
                        filter: Vec::new(),
                    });
                    i = gend + 1;
                    continue;
                }
                // No blend stack: blend groups consume their whole
                // range, and `Normal` groups paint inline, so the pop is
                // always a no-op.
                DisplayItem::PopBlend => {}
                DisplayItem::PushTransform { matrix } => {
                    let gend = scan_transform_group(&list.items, i);
                    debug_assert!(
                        gend < list.items.len(),
                        "PushTransform without matching PopTransform"
                    );
                    if *matrix == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] {
                        // Identity: no-op fast path, items lower inline;
                        // the matching PopTransform is a no-op below.
                        i += 1;
                        continue;
                    }
                    // Non-invertible: per spec the element renders
                    // nothing (mirrors the CPU backend). The test is
                    // exact-zero so tiny-but-valid transforms (e.g.
                    // `scale(1e-7)`) still lower.
                    let det = matrix[0] * matrix[3] - matrix[1] * matrix[2];
                    if !matrix.iter().all(|v| v.is_finite()) || det == 0.0 {
                        i = gend.min(end) + 1;
                        continue;
                    }
                    let over_cap = parts.group_count >= MAX_OPACITY_GROUPS;
                    if over_cap {
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "[bowser_paint] transform group cap hit; lowering inline untransformed"
                        );
                        // Degradation mirrors the CPU backend: past the
                        // cap the group's items lower inline,
                        // untransformed, with plain source-over.
                        i += 1;
                        continue;
                    }
                    parts.group_count += 1;
                    let gend = gend.min(end);
                    // Lower with empty clips: ancestor clips apply
                    // post-transform at composite time (the executor
                    // sets the outer clip state for the composite
                    // draw); clips pushed inside the range lower
                    // normally and apply to the untransformed content.
                    let group = Self::lower_range(parts, list, i + 1, gend, Vec::new(), opacity);
                    // A group that lowered to nothing paints nothing.
                    let Some(bounds) = lowered_bounds(&group) else {
                        i = gend + 1;
                        continue;
                    };
                    out.ops.push(Op::Group(out.groups.len()));
                    out.groups.push(OpacityGroup {
                        lowered: group,
                        opacity: 1.0,
                        blend: BlendMode::Normal,
                        bounds,
                        transform: Some(*matrix),
                        filter: Vec::new(),
                    });
                    i = gend + 1;
                    continue;
                }
                // No transform stack: transform groups consume their
                // whole range (like blend groups), so the pop is always
                // a no-op.
                DisplayItem::PopTransform => {}
                DisplayItem::PushFilter { ops } => {
                    if ops.is_empty() {
                        // No-op fast path: the builder never emits
                        // empty filter chains; the matching PopFilter is
                        // a no-op below. (A hand-built list may still
                        // contain them.)
                        i += 1;
                        continue;
                    }
                    // Filter groups always isolate (like blend groups):
                    // the chain post-processes the isolated rendering.
                    let gend = scan_filter_group(&list.items, i);
                    debug_assert!(
                        gend < list.items.len(),
                        "PushFilter without matching PopFilter"
                    );
                    let over_cap = parts.group_count >= MAX_OPACITY_GROUPS;
                    if over_cap {
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "[bowser_paint] filter group cap hit; lowering inline unfiltered"
                        );
                        // Degradation mirrors the CPU backend: past the
                        // cap the group's items lower inline, unfiltered,
                        // with plain source-over.
                        i += 1;
                        continue;
                    }
                    parts.group_count += 1;
                    let gend = gend.min(end);
                    let group = Self::lower_range(parts, list, i + 1, gend, clips.clone(), opacity);
                    // Like the other isolated groups: a group that
                    // lowered to nothing paints nothing, so skip it
                    // entirely.
                    let Some(bounds) = lowered_bounds(&group) else {
                        i = gend + 1;
                        continue;
                    };
                    // The filter can spill outside the content bounds
                    // (blur, drop-shadow): inflate the offscreen texture
                    // by the chain's conservative page-px spill so
                    // filtered pixels are not culled (mirrors the CPU
                    // backend's inflated group pixmap).
                    let spill = filter_spill(ops);
                    let bounds = Rect {
                        x: bounds.x - spill,
                        y: bounds.y - spill,
                        w: bounds.w + 2.0 * spill,
                        h: bounds.h + 2.0 * spill,
                    };
                    out.ops.push(Op::Group(out.groups.len()));
                    out.groups.push(OpacityGroup {
                        lowered: group,
                        opacity: 1.0,
                        blend: BlendMode::Normal,
                        bounds,
                        transform: None,
                        filter: plan_filter_passes(ops),
                    });
                    i = gend + 1;
                    continue;
                }
                // No filter stack: filter groups consume their whole
                // range, so the pop is always a no-op.
                DisplayItem::PopFilter => {}
                DisplayItem::PushBackdropFilter { ops, rect } => {
                    if ops.is_empty() {
                        // No-op fast path (like PushFilter).
                        i += 1;
                        continue;
                    }
                    // Backdrop-filter is not an isolated group: the
                    // executor captures the already-painted backdrop
                    // under `rect`, filters the capture, writes it back,
                    // and the following ops (the element's content) paint
                    // over it inline. The lowering just records the
                    // planned passes and the page-px filter region.
                    out.ops.push(Op::BackdropFilter(Box::new(BackdropFilterOp {
                        passes: plan_filter_passes(ops),
                        rect: *rect,
                    })));
                    i += 1;
                    continue;
                }
                // Like PopFilter: the group consumes its whole range.
                DisplayItem::PopBackdropFilter => {}
                DisplayItem::BoxShadow {
                    rect,
                    radii,
                    color,
                    offset,
                    blur,
                    spread,
                    inset,
                } => {
                    let color = color.with_opacity(opacity);
                    // Conservative coverage area (page px): outset needs
                    // the spread-inflated shape ± blur at the offset;
                    // inset needs the border box ± blur around the
                    // offset deflated shape. Over-cover is harmless
                    // (the fragment shader evaluates the exact SDF).
                    let pad = blur.max(0.0) + spread.abs() + offset.0.abs().max(offset.1.abs());
                    let area = if *inset {
                        Rect {
                            x: rect.x - pad,
                            y: rect.y - pad,
                            w: rect.w + 2.0 * pad,
                            h: rect.h + 2.0 * pad,
                        }
                    } else {
                        Rect {
                            x: rect.x - pad + offset.0.min(0.0),
                            y: rect.y - pad + offset.1.min(0.0),
                            w: rect.w + 2.0 * pad + offset.0.abs(),
                            h: rect.h + 2.0 * pad + offset.1.abs(),
                        }
                    };
                    out.shadow_quad(area, *rect, *radii, color, *offset, *blur, *spread, *inset);
                }
                DisplayItem::TextShadow {
                    text,
                    text_range,
                    rect,
                    baseline,
                    font_key,
                    size_px,
                    color,
                    offset,
                    blur,
                } => {
                    let slice = text.get(text_range.clone()).unwrap_or("");
                    if *blur <= 0.0 {
                        // Sharp shadow: one text draw at the offset.
                        emit_shaped_text(
                            &mut out,
                            parts.atlas,
                            parts.queue,
                            parts.get_font_data,
                            slice,
                            rect.x + offset.0,
                            baseline + offset.1,
                            *font_key,
                            *size_px,
                            color.with_opacity(opacity),
                            parts.atlas_generation,
                            parts.scale,
                        );
                    } else {
                        // Blurred shadow: 3×3 taps with alpha/9 each, a
                        // single-pass box-blur approximation of the CPU
                        // backend's gaussian σ = blur/2. A 3-tap box has
                        // per-axis variance 2r²/3; matching the gaussian
                        // variance σ² gives r = σ√(3/2) = blur·√6/4, so
                        // the tap spread carries the same blur energy as
                        // the CPU's multi-pass blur. (A single shaped run
                        // re-emitted per tap would avoid the 9× shaping
                        // cost; text shadows are rare enough that the
                        // simple path wins.)
                        let r = blur * 6.0f32.sqrt() / 4.0;
                        let tap = color.with_opacity(opacity / 9.0);
                        for jy in -1..=1 {
                            for jx in -1..=1 {
                                emit_shaped_text(
                                    &mut out,
                                    parts.atlas,
                                    parts.queue,
                                    parts.get_font_data,
                                    slice,
                                    rect.x + offset.0 + jx as f32 * r,
                                    baseline + offset.1 + jy as f32 * r,
                                    *font_key,
                                    *size_px,
                                    tap,
                                    parts.atlas_generation,
                                    parts.scale,
                                );
                            }
                        }
                    }
                }
                DisplayItem::EdgeGlow {
                    edge,
                    rect,
                    strength,
                } => {
                    // Gradient through the solid-quad pipe: one quad per
                    // page-px row with the shared `edge_glow_alpha`
                    // profile sampled at the row center. No dedicated
                    // gradient pipe — up to ~96 instances per glow is
                    // noise next to a frame's quad count, and the clip
                    // scissor from the emitter's PushClip/PopClip pair
                    // applies to every row exactly like a FillRect, so
                    // the glow paints above scrolled content but inside
                    // the scrollport. Documented choice: analytic
                    // per-row stepping beats baking stacked FillRect
                    // bands into the display list (fewer items, exact
                    // profile shared with the CPU backend).
                    let strength = strength.clamp(0.0, 1.0);
                    if strength > 0.0 && rect.w > 0.0 && rect.h > 0.0 {
                        let horizontal = matches!(edge, Edge4::Left | Edge4::Right);
                        let depth = if horizontal { rect.w } else { rect.h };
                        if depth > 0.0 {
                            // Clamp: a hand-built item with a huge rect
                            // must not blow up the instance buffer (the
                            // emitter caps depth at 96).
                            let steps = depth.ceil().clamp(1.0, 4096.0) as usize;
                            let step = depth / steps as f32;
                            for i in 0..steps {
                                let d0 = i as f32 * step;
                                let d1 = d0 + step;
                                let t = (d0 + d1) / (2.0 * depth);
                                let a = edge_glow_alpha(strength, t) * opacity;
                                if a <= 0.0 {
                                    continue;
                                }
                                let row = match edge {
                                    Edge4::Top => Rect {
                                        x: rect.x,
                                        y: rect.y + d0,
                                        w: rect.w,
                                        h: step,
                                    },
                                    Edge4::Bottom => Rect {
                                        x: rect.x,
                                        y: rect.y + rect.h - d1,
                                        w: rect.w,
                                        h: step,
                                    },
                                    Edge4::Left => Rect {
                                        x: rect.x + d0,
                                        y: rect.y,
                                        w: step,
                                        h: rect.h,
                                    },
                                    Edge4::Right => Rect {
                                        x: rect.x + rect.w - d1,
                                        y: rect.y,
                                        w: step,
                                        h: rect.h,
                                    },
                                };
                                out.solid_quad(row, EDGE_GLOW_COLOR.with_opacity(a));
                            }
                        }
                    }
                }
            }
            i += 1;
        }
        // Push/PopOpacity items are balanced by construction; a missed
        // pop must not leak folded opacity into following items.
        debug_assert!(
            opacity_stack.is_empty(),
            "unbalanced PushOpacity/PopOpacity: opacity stack not drained"
        );
        out
    }

    /// Get (building on first use) the three pipelines for `format`.
    /// Returns an owned clone so the render pass can borrow other
    /// backend state freely (`RenderPipeline` is `Arc`-backed; cheap).
    fn pipelines_for(&mut self, format: wgpu::TextureFormat) -> Pipelines {
        if !self.pipelines.contains_key(&format) {
            let pipes = build_pipelines(
                &self.device,
                &self.frame_layout,
                &self.tex_layout,
                &self.blend_layout,
                &self.clip_layout,
                &self.filter_uniform_layout,
                format,
            );
            self.pipelines.insert(format, pipes);
        }
        self.pipelines[&format].clone()
    }

    /// Get (building on first use) the group-2 clip bind group for a
    /// clip state (`None` = clipping disabled). The uniform is
    /// `CLIP_UNIFORM_BYTES`: u32 count, u32 `has_rounded`, u32
    /// `has_poly`, 4 pad bytes, then 8 entries of (vec4 rect + vec4
    /// radii + u32 kind + u32 poly_len + vec2 pad + 48 vec4 poly) as le
    /// f32s/u32s (matches `ClipUniform` in `SHADER_COMMON`).
    fn clip_bind_group(&mut self, key: ClipKey) -> wgpu::BindGroup {
        if let Some(bg) = self.clip_bg_cache.get(&key) {
            return bg.clone();
        }
        let mut uniform = [0u8; CLIP_UNIFORM_BYTES as usize];
        if let Some((count, entries)) = key {
            uniform[0] = count;
            // `has_rounded`: 1 if any entry has a positive corner radius.
            // Lets the shader skip the per-entry radii scan (and the
            // 8x8 sampling decision) for all-rect stacks.
            let mut has_rounded = 0u32;
            // `has_poly`: 1 if any entry is a `clip-path` polygon.
            let mut has_poly = 0u32;
            for (i, words) in entries.iter().enumerate() {
                let base = 16 + i * CLIP_ENTRY_BYTES;
                for (j, w) in words.iter().enumerate() {
                    uniform[base + j * 4..base + (j + 1) * 4].copy_from_slice(&w.to_le_bytes());
                }
                if i < count as usize {
                    // Radii are words 4..8 of each entry.
                    if words[4..8]
                        .iter()
                        .any(|&w| f32::from_le_bytes(w.to_le_bytes()) > 0.0)
                    {
                        has_rounded = 1;
                    }
                    // Kind is word 8 of each entry.
                    if words[8] == 1 {
                        has_poly = 1;
                    }
                }
            }
            uniform[4..8].copy_from_slice(&has_rounded.to_le_bytes());
            uniform[8..12].copy_from_slice(&has_poly.to_le_bytes());
        }
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bowser-clip-uniform"),
            size: CLIP_UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, &uniform);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bowser-clip-bg"),
            layout: &self.clip_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });
        self.clip_bg_cache.insert(key, bg.clone());
        bg
    }

    /// Upload passes of sustained underuse before a persistent instance
    /// buffer shrinks back: one huge frame must not pin a huge buffer
    /// forever, but the streak keeps jittery frame sizes from thrashing
    /// between grow and shrink. Counted in [`WgpuBackend::upload_instances`]
    /// calls — exactly one per frame, since group rendering no longer
    /// re-uploads the parent's bytes.
    const INSTANCE_SHRINK_STREAK: u32 = 120;
    /// A buffer is "underused" when the pass's bytes fit in 1/this of its
    /// capacity.
    const INSTANCE_SHRINK_RATIO: u64 = 4;

    /// Floor capacity for instance buffers: avoids tiny buffers for
    /// near-empty frames.
    const INSTANCE_BUF_FLOOR: u64 = 4096;

    /// Pure capacity decision behind [`WgpuBackend::instance_buffer`]:
    /// `(target_capacity, new_streak)` for the pass, or `None` when a
    /// pipe that never draws gets no bytes. Split out so the shrink
    /// policy is unit-testable without a GPU.
    fn instance_target_cap(cap: Option<u64>, streak: u32, need: u64) -> Option<(u64, u32)> {
        match cap {
            // A pipe that never draws never allocates.
            None if need == 0 => None,
            None => Some((need.next_power_of_two().max(Self::INSTANCE_BUF_FLOOR), 0)),
            Some(cap) if need > cap => {
                // Power-of-two growth amortizes reallocations across
                // passes whose instance counts jitter.
                Some((need.next_power_of_two().max(Self::INSTANCE_BUF_FLOOR), 0))
            }
            Some(cap) => {
                if need <= cap / Self::INSTANCE_SHRINK_RATIO {
                    let streak = streak.saturating_add(1);
                    if streak >= Self::INSTANCE_SHRINK_STREAK {
                        let shrunk = need.next_power_of_two().max(Self::INSTANCE_BUF_FLOOR);
                        return Some((if shrunk < cap { shrunk } else { cap }, 0));
                    }
                    return Some((cap, streak));
                }
                Some((cap, 0))
            }
        }
    }

    /// Upload one frame's instance bytes into the pipe's persistent
    /// buffer, growing it (power-of-two) only when the frame outgrows
    /// the current capacity, and shrinking it back after
    /// [`INSTANCE_SHRINK_STREAK`] consecutive passes of using at most
    /// 1/[`INSTANCE_SHRINK_RATIO`] of the capacity. `None` when there is
    /// nothing to draw. `wgpu::Buffer` is `Arc`-backed, so the returned
    /// clone is cheap.
    fn instance_buffer(&mut self, pipe: PipeKind, bytes: &[u8]) -> Option<wgpu::Buffer> {
        let need = bytes.len() as u64;
        let slot = pipe.slot();
        // Copy the capacity out first so the shrink/grow decisions below
        // can mutate disjoint fields without borrow conflicts.
        let cap = self.instance_bufs[slot].as_ref().map(|(_, cap)| *cap);
        let (target_cap, streak) =
            Self::instance_target_cap(cap, self.instance_small_streak[slot], need)?;
        self.instance_small_streak[slot] = streak;
        if Some(target_cap) != cap {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bowser-instances"),
                size: target_cap,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_bufs[slot] = Some((buf, target_cap));
        }
        let (buf, _) = self.instance_bufs[slot]
            .as_ref()
            .expect("slot filled above");
        if bytes.is_empty() {
            return None;
        }
        self.queue.write_buffer(buf, 0, bytes);
        Some(buf.clone())
    }

    /// Offscreen render target: RGBA8 + COPY_SRC for readback.
    fn make_target(&self, w: u32, h: u32) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bowser-offscreen"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            // TEXTURE_BINDING: isolated opacity groups sample the target
            // back when compositing.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    /// Copy a texture to CPU memory (RGBA8 row-major, top row first).
    ///
    /// Returns `None` — instead of panicking — when the device poll or
    /// the buffer mapping fails (device lost mid-frame, ...), or when
    /// the dimensions overflow the staging/output sizing, so callers can
    /// degrade gracefully.
    fn readback(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        w: u32,
        h: u32,
    ) -> Option<Vec<u8>> {
        // All sizing in u64 with checked arithmetic: `w`/`h` are u32, so
        // `w * h * 4` overflows u32 above ~1 gigapixel, and a wrapped
        // size would under-allocate the staging buffer or the output and
        // panic (debug) or go out of bounds (release) in the row loop
        // below. Every fallible conversion and the output length is
        // computed BEFORE any GPU allocation, so an oversized target
        // fails without allocating a staging buffer or issuing a copy.
        let w64 = u64::from(w);
        let h64 = u64::from(h);
        let row_bytes = w64 * 4;
        let padded_bpr = row_bytes.div_ceil(256) * 256;
        let bytes_per_row = u32::try_from(padded_bpr).ok()?;
        let out_len = usize::try_from(w64.checked_mul(h64)?.checked_mul(4)?).ok()?;
        let row_len = usize::try_from(row_bytes).ok()?;
        let buf_size = padded_bpr.checked_mul(h64)?;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bowser-readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bowser-readback"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let (tx, rx) = std::sync::mpsc::channel();
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok()?;
        // The callback result, not just the poll: a failed mapping must
        // not reach `get_mapped_range`.
        rx.recv().ok()?.ok()?;
        let mut out = {
            let mapped = slice.get_mapped_range().ok()?;
            let mut pixels = vec![0u8; out_len];
            for row in 0..h64 {
                // Checked offsets plus `get` (not indexing): no panic or
                // OOB even if the mapped range is shorter than computed.
                let src_off = usize::try_from(row.checked_mul(padded_bpr)?).ok()?;
                let dst_off = usize::try_from(row.checked_mul(row_bytes)?).ok()?;
                let src = mapped.get(src_off..src_off.checked_add(row_len)?)?;
                let dst = pixels.get_mut(dst_off..dst_off.checked_add(row_len)?)?;
                dst.copy_from_slice(src);
            }
            pixels
        };
        buf.unmap();
        // The framebuffer holds premultiplied composites (standard
        // ALPHA_BLENDING); the crate's pixel convention is straight
        // RGBA8, matching the CPU backend's `unpremultiply`.
        unpremultiply(&mut out);
        Some(out)
    }
}

/// Divide premultiplied rgb by alpha, in place (straight RGBA8 output).
/// Fully transparent pixels map to (0, 0, 0, 0).
fn unpremultiply(pixels: &mut [u8]) {
    let (chunks, _) = pixels.as_chunks_mut::<4>();
    for px in chunks {
        let a = px[3];
        if a == 0 {
            px[0] = 0;
            px[1] = 0;
            px[2] = 0;
        } else if a < 255 {
            let inv = 255.0 / f32::from(a);
            for c in &mut px[0..3] {
                *c = ((f32::from(*c) * inv).round().clamp(0.0, 255.0)) as u8;
            }
        }
    }
}

/// Reborrowed backend parts used during lowering (avoids borrowck
/// conflicts between the accumulator and the backend's caches).
struct LowerParts<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    atlas: &'a mut Atlas,
    sampler: &'a wgpu::Sampler,
    tex_layout: &'a wgpu::BindGroupLayout,
    image_cache: &'a mut ImageCache,
    get_font_data: &'a Option<Arc<dyn Fn(FontKey) -> Option<FontData> + Send + Sync>>,
    /// Baked gradient textures (see [`GradientEntry`]); entries are
    /// stamped with `gradient_frame` on reference.
    gradient_cache: &'a mut HashMap<u64, GradientEntry>,
    /// Current [`WgpuBackend::gradient_frame`]: stamps
    /// [`GradientEntry::last_frame`] on every bake or cache hit.
    gradient_frame: u64,
    /// Highest atlas generation observed while lowering this frame.
    /// [`emit_shaped_text`] bumps it from each rasterized glyph's owning
    /// generation; [`WgpuBackend::lower`] rebuilds the glyph bind group
    /// when it exceeds `glyph_bind_group_generation`.
    atlas_generation: &'a mut u64,
    /// Isolated opacity groups created during this lowering; capped at
    /// [`MAX_OPACITY_GROUPS`]. Each one pins a full-target texture until
    /// submit and nested groups recurse in `lower_range`, so the cap
    /// bounds both transient VRAM and stack depth. Fresh per `lower()`.
    group_count: u32,
    /// `PushOpacity`-index -> [`scan_opacity_group`] tuple, built once per
    /// [`WgpuBackend::lower`] by [`precompute_group_scans`]. Nested groups
    /// would otherwise re-walk the same items (O(items x depth)); the
    /// single pass keeps group discovery O(items).
    group_scans: HashMap<usize, (usize, usize, bool, Option<usize>)>,
    /// Device px per page px, per axis, for this frame (the target size
    /// over the CSS viewport size; 1.0 when they match). Geometry scales
    /// in the vertex shader via the viewport uniform, but text must
    /// rasterize at device px to stay crisp — see [`emit_shaped_text`].
    scale: (f32, f32),
}

impl<'a> LowerParts<'a> {
    fn of(backend: &'a mut WgpuBackend, atlas_generation: &'a mut u64, scale: (f32, f32)) -> Self {
        Self {
            device: &backend.device,
            queue: &backend.queue,
            atlas: &mut backend.atlas,
            sampler: &backend.sampler,
            tex_layout: &backend.tex_layout,
            image_cache: &mut backend.image_cache,
            get_font_data: &backend.get_font_data,
            gradient_cache: &mut backend.gradient_cache,
            gradient_frame: backend.gradient_frame,
            atlas_generation,
            group_count: 0,
            group_scans: HashMap::new(),
            scale,
        }
    }

    /// Precomputed [`scan_opacity_group`] tuple for the `PushOpacity` at
    /// `idx` (see [`precompute_group_scans`]). The on-the-fly scan is a
    /// defensive fallback for a list that changed after the precompute.
    fn group_scan(&self, items: &[DisplayItem], idx: usize) -> (usize, usize, bool, Option<usize>) {
        self.group_scans
            .get(&idx)
            .copied()
            .unwrap_or_else(|| scan_opacity_group(items, idx))
    }
}

/// Upload `id`'s decoded bytes to a GPU texture on first use and cache the
/// bind group. Returns false when the display list has no bytes for `id`.
///
/// Content identity comes from [`DisplayList::image_content_hash`], computed
/// once at registration ([`DisplayList::add_image`]) instead of rehashing
/// the full pixel buffer every frame: registered bytes are immutable
/// through every list API, so the cached hash always matches the decoded
/// bytes. The hash (not the buffer address) is the content identity, so
/// display lists rebuilt at reused addresses can never serve a stale
/// texture.
#[allow(clippy::too_many_arguments)]
fn ensure_image(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sampler: &wgpu::Sampler,
    tex_layout: &wgpu::BindGroupLayout,
    cache: &mut ImageCache,
    list: &DisplayList,
    id: ImageId,
) -> bool {
    let decoded: &DecodedImage = match list.image(id) {
        Some(d) => d,
        None => return false,
    };
    // Mirror the CPU backend's guard (`CpuBackend::image_pixmap`): the
    // `DecodedImage` fields are public, so a caller can construct one
    // that bypasses `DecodedImage::from_rgba`'s length assert. Uploading
    // short bytes via `queue.write_texture` below is a wgpu validation
    // error, which the default uncaptured-error handler turns into a
    // panic. Checked arithmetic also guards against `width * height * 4`
    // overflowing on 32-bit targets.
    let expected_len = (decoded.width as usize)
        .checked_mul(decoded.height as usize)
        .and_then(|pixels| pixels.checked_mul(4));
    if decoded.width == 0 || decoded.height == 0 || expected_len != Some(decoded.rgba.len()) {
        return false;
    }
    // The decoder caps (16384px) can exceed what the device supports:
    // `create_texture` with oversized dimensions is a wgpu validation
    // error, which the default uncaptured-error handler turns into a
    // panic. Validate against the device's limit first and skip the image
    // instead — the caller already treats `false` as "skip this draw".
    let max_dim = device.limits().max_texture_dimension_2d;
    if decoded.width > max_dim || decoded.height > max_dim {
        return false;
    }
    // The registration-cached hash: O(1) per reference, no pixel-buffer
    // rehash. The on-demand fallback only covers a hypothetical map
    // divergence (`add_image` populates both maps together, so for a
    // registered id it never triggers); it keeps this total.
    let hash = list
        .image_content_hash(id)
        .unwrap_or_else(|| decoded.content_hash());
    // Cache hit only when the content hash and dimensions match: the id
    // alone is not enough, since ids are per-list and lists are rebuilt
    // (often at reused addresses) every frame.
    if let Some(entry) = cache.entries.get(&id) {
        if entry.hash == hash && entry.width == decoded.width && entry.height == decoded.height {
            // Referenced by this frame's lowering: refresh LRU recency
            // and pin for the frame (see
            // `ImageEntry::lowered_this_frame`).
            cache.touch(&id);
            return true;
        }
        cache.remove(&id);
    }
    // Byte-budget capacity before the upload: evict cold entries (never
    // ones this frame's lowering already referenced) so the new texture
    // fits. Every entry referenced by the in-progress lowering must
    // survive until execution — a mid-frame clear used to wipe textures
    // for image ops already lowered, panicking at draw time once the
    // 513th distinct image uploaded — and `reserve` only evicts entries
    // with `lowered_this_frame` unset, which are not in the instance
    // stream (see `ImageCache::reserve`).
    cache.reserve(decoded.rgba.len());
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bowser-image"),
        size: wgpu::Extent3d {
            width: decoded.width,
            height: decoded.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        &decoded.rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(decoded.width * 4),
            rows_per_image: Some(decoded.height),
        },
        wgpu::Extent3d {
            width: decoded.width,
            height: decoded.height,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bowser-image-bg"),
        layout: tex_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    cache.insert(
        id,
        ImageEntry {
            _texture: texture,
            _view: view,
            bind_group,
            hash,
            width: decoded.width,
            height: decoded.height,
            // `expected_len` above already proved
            // `width * height * 4 == rgba.len()`: the buffer length is
            // the exact VRAM byte count, with no overflow arithmetic.
            bytes: decoded.rgba.len(),
            last_used: cache.clock,
            // Uploaded by this frame's lowering: in the instance stream
            // as of the `image_quad` the caller emits next.
            lowered_this_frame: true,
        },
    );
    cache.clock = cache.clock.wrapping_add(1);
    true
}

/// Bake a [`DisplayItem::Gradient`] to a GPU texture on first use and
/// cache the bind group, returning the [`GradientEntry`] cache key.
/// Returns `None` when the item's device-px size is degenerate.
///
/// The bake runs through the shared [`bake_gradient_rgba8`]
/// rasterizer, so GPU gradient pixels are identical to the CPU
/// backend's. The cache key mixes the gradient's
/// [`Gradient::content_hash`] with the bake dimensions, so identical
/// gradients at identical sizes share one texture across frames;
/// [`WgpuBackend::lower`] evicts entries its frame did not reference.
#[allow(clippy::too_many_arguments)]
fn ensure_gradient_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sampler: &wgpu::Sampler,
    tex_layout: &wgpu::BindGroupLayout,
    cache: &mut HashMap<u64, GradientEntry>,
    frame: u64,
    gradient: &Gradient,
    rect: &Rect,
    scale: (f32, f32),
) -> Option<u64> {
    // Device-px bake size, snapped outward; the shader stretches the
    // texture over the page-px quad exactly like an image.
    let w = ((rect.w * scale.0).ceil() as u32).clamp(1, MAX_GRADIENT_BAKE_DIM);
    let h = ((rect.h * scale.1).ceil() as u32).clamp(1, MAX_GRADIENT_BAKE_DIM);
    // The device limit is the hard ceiling: `create_texture` with an
    // oversized extent is a wgpu validation error, which the default
    // uncaptured-error handler turns into a panic (same guard as
    // `ensure_image`).
    let max_dim = device
        .limits()
        .max_texture_dimension_2d
        .min(MAX_GRADIENT_BAKE_DIM);
    let (w, h) = (w.min(max_dim).max(1), h.min(max_dim).max(1));
    // Mix the bake dimensions into the key: one gradient at two sizes
    // is two textures.
    let mut key = gradient.content_hash();
    for b in w.to_le_bytes().into_iter().chain(h.to_le_bytes()) {
        key ^= u64::from(b);
        key = key.wrapping_mul(0x0100_0000_0000_01b3);
    }
    if let Some(entry) = cache.get_mut(&key) {
        entry.last_frame = frame;
        return Some(key);
    }
    let rgba = bake_gradient_rgba8(gradient, rect, w, h);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bowser-gradient"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bowser-gradient-bg"),
        layout: tex_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    cache.insert(
        key,
        GradientEntry {
            _texture: texture,
            _view: view,
            bind_group,
            last_frame: frame,
        },
    );
    Some(key)
}

/// Build all three pipelines for one output format.
#[allow(clippy::too_many_arguments)]
fn build_pipelines(
    device: &wgpu::Device,
    frame_layout: &wgpu::BindGroupLayout,
    tex_layout: &wgpu::BindGroupLayout,
    blend_layout: &wgpu::BindGroupLayout,
    clip_layout: &wgpu::BindGroupLayout,
    filter_uniform_layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
) -> Pipelines {
    let solid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bowser-solid"),
        source: wgpu::ShaderSource::Wgsl([SHADER_COMMON, SHADER_SOLID].concat().into()),
    });
    let glyph_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bowser-glyph"),
        source: wgpu::ShaderSource::Wgsl([SHADER_COMMON, SHADER_GLYPH].concat().into()),
    });
    let image_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bowser-image"),
        source: wgpu::ShaderSource::Wgsl([SHADER_COMMON, SHADER_IMAGE].concat().into()),
    });
    let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bowser-composite"),
        source: wgpu::ShaderSource::Wgsl([SHADER_COMMON, SHADER_COMPOSITE].concat().into()),
    });
    let blend_composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bowser-blend-composite"),
        source: wgpu::ShaderSource::Wgsl([SHADER_COMMON, SHADER_BLEND_COMPOSITE].concat().into()),
    });
    let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bowser-shadow"),
        source: wgpu::ShaderSource::Wgsl([SHADER_COMMON, SHADER_SHADOW].concat().into()),
    });
    let filter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bowser-filter"),
        source: wgpu::ShaderSource::Wgsl([SHADER_COMMON, SHADER_FILTER].concat().into()),
    });
    let composite_clip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bowser-composite-clip"),
        source: wgpu::ShaderSource::Wgsl([SHADER_COMMON, SHADER_COMPOSITE_CLIP].concat().into()),
    });

    let solid_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bowser-solid-pl"),
        bind_group_layouts: &[Some(frame_layout), None, Some(clip_layout)],
        immediate_size: 0,
    });
    let tex_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bowser-tex-pl"),
        bind_group_layouts: &[Some(frame_layout), Some(tex_layout), Some(clip_layout)],
        immediate_size: 0,
    });
    // Blend-mode compositing: group 1 carries the group texture, the
    // sampler, and the copied backdrop texture.
    let blend_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bowser-blend-pl"),
        bind_group_layouts: &[Some(frame_layout), Some(blend_layout), Some(clip_layout)],
        immediate_size: 0,
    });

    let solid = make_pipeline(
        device,
        &solid_layout,
        &solid_shader,
        "vs_solid",
        "fs_solid",
        wgpu::VertexBufferLayout {
            array_stride: SOLID_STRIDE as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 1,
                },
            ],
        },
        format,
        Some(wgpu::BlendState::ALPHA_BLENDING),
        "bowser-solid-pipeline",
    );
    let glyph = make_pipeline(
        device,
        &tex_pipeline_layout,
        &glyph_shader,
        "vs_glyph",
        "fs_glyph",
        wgpu::VertexBufferLayout {
            array_stride: GLYPH_STRIDE as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 2,
                },
            ],
        },
        format,
        Some(wgpu::BlendState::ALPHA_BLENDING),
        "bowser-glyph-pipeline",
    );
    let image = make_pipeline(
        device,
        &tex_pipeline_layout,
        &image_shader,
        "vs_image",
        "fs_image",
        wgpu::VertexBufferLayout {
            array_stride: IMAGE_STRIDE as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 32,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 36,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 52,
                    shader_location: 4,
                },
            ],
        },
        format,
        Some(wgpu::BlendState::ALPHA_BLENDING),
        "bowser-image-pipeline",
    );
    // Group compositing: the offscreen group texture already holds
    // premultiplied values, so the blend must not premultiply again.
    // The instance layout carries the blend-mode discriminant at
    // offset 36 (ignored by `fs_composite`, consumed by the blend
    // pipeline below) and the 2D transform matrix at offsets 40/48/56
    // (identity for opacity/blend groups); both composite pipelines
    // share it.
    let composite_instance_layout = || wgpu::VertexBufferLayout {
        array_stride: COMPOSITE_STRIDE as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32,
                offset: 32,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 36,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 40,
                shader_location: 4,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 48,
                shader_location: 5,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 56,
                shader_location: 6,
            },
        ],
    };
    let composite = make_pipeline(
        device,
        &tex_pipeline_layout,
        &composite_shader,
        "vs_composite",
        "fs_composite",
        composite_instance_layout(),
        format,
        Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
        "bowser-composite-pipeline",
    );
    // Blend-mode compositing: same instance layout, but the fragment
    // shader blends against a copied backdrop and blending is disabled
    // (the shader outputs the final premultiplied value).
    let composite_blend = make_pipeline(
        device,
        &blend_pipeline_layout,
        &blend_composite_shader,
        "vs_blend_composite",
        "fs_blend_composite",
        composite_instance_layout(),
        format,
        None,
        "bowser-blend-composite-pipeline",
    );
    // Box shadows: no texture (group 1 unused, like the solid
    // pipeline); the fragment shader evaluates the SDF from the
    // instance params.
    let shadow = make_pipeline(
        device,
        &solid_layout,
        &shadow_shader,
        "vs_shadow",
        "fs_shadow",
        wgpu::VertexBufferLayout {
            array_stride: SHADOW_STRIDE as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 48,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 64,
                    shader_location: 4,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 80,
                    shader_location: 5,
                },
            ],
        },
        format,
        Some(wgpu::BlendState::ALPHA_BLENDING),
        "bowser-shadow-pipeline",
    );
    // Filter passes: fullscreen triangle (no vertex buffer), group 0 =
    // frame (unused by the shader but required by the layout
    // convention), group 1 = source texture + sampler, group 2 =
    // filter params. Blending disabled (REPLACE): the shader outputs
    // the final premultiplied value.
    let filter_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bowser-filter-pl"),
        bind_group_layouts: &[
            Some(frame_layout),
            Some(tex_layout),
            Some(filter_uniform_layout),
        ],
        immediate_size: 0,
    });
    let filter = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("bowser-filter-pipeline"),
        layout: Some(&filter_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &filter_shader,
            entry_point: Some("vs_filter"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &filter_shader,
            entry_point: Some("fs_filter"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    // Backdrop-filter write-back: same instance layout and pipeline
    // layout as the composite pipeline, but the fragment shader
    // multiplies by the exact fractional clip coverage.
    let composite_clip = make_pipeline(
        device,
        &tex_pipeline_layout,
        &composite_clip_shader,
        "vs_composite_clip",
        "fs_composite_clip",
        composite_instance_layout(),
        format,
        Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
        "bowser-composite-clip-pipeline",
    );
    Pipelines {
        solid,
        glyph,
        image,
        composite,
        composite_blend,
        shadow,
        filter,
        composite_clip,
    }
}

impl Backend for WgpuBackend {
    /// Lower `list` to validate it; with no surface attached there is no
    /// render target, so the lowered batches are dropped. Pixel-producing
    /// entry points are [`WgpuBackend::render_to_surface`] (live surface)
    /// and [`WgpuBackend::render_offscreen_with_viewport`] (tests).
    fn paint(&mut self, list: &DisplayList, _viewport: Size, _scroll: (f32, f32)) {
        // Validation-only path: no target size is known, so there is no
        // device scale to apply; 1.0 keeps text at page px.
        let _ = self.lower(list, (1.0, 1.0));
        // The lowered batches are dropped without submitting: no draw can
        // reference retired atlas generations afterwards, so reclaim them
        // here (and their bind groups) instead of leaking one 1 MiB page
        // texture per overflow across `paint` calls. `render_into` does
        // the same after submit.
        self.retired_glyph_bind_groups.clear();
        self.atlas.end_frame();
    }
}

// ---------------------------------------------------------------------------
// GPU bring-up (single choke point)
// ---------------------------------------------------------------------------

impl WgpuBackend {
    /// Request a Vulkan adapter + device + queue.
    ///
    /// Single choke point for GPU bring-up: the windowed path
    /// (`bowser_app`'s `init_gpu`) and every headless path (tests,
    /// [`WgpuBackend::new_headless`]) come through here instead of
    /// hand-rolling adapter/device requests with divergent options.
    ///
    /// The request policy follows `compatible_surface`:
    /// - `Some(surface)` (live window): `PowerPreference::HighPerformance`
    ///   with no software fallback; only adapters able to present to the
    ///   surface are considered.
    /// - `None` (headless): `PowerPreference::LowPower` with
    ///   `force_fallback_adapter`, so software adapters (lavapipe) work in
    ///   CI.
    ///
    /// The adapter is returned alongside the device/queue because the
    /// windowed caller needs it for `Surface::get_default_config`.
    /// `block_on` is used for the adapter/device requests (init only,
    /// never per-frame). Returns `None` when no adapter matches or device
    /// creation fails, so callers can degrade (e.g. fall back to the CPU
    /// rasterizer) instead of panicking.
    pub fn request_device(
        compatible_surface: Option<&wgpu::Surface<'static>>,
    ) -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
        let (power_preference, force_fallback_adapter, label) = if compatible_surface.is_some() {
            (
                wgpu::PowerPreference::HighPerformance,
                false,
                "bowser-window",
            )
        } else {
            (wgpu::PowerPreference::LowPower, true, "bowser-headless")
        };
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference,
            force_fallback_adapter,
            compatible_surface,
            ..wgpu::RequestAdapterOptions::default()
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some(label),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..wgpu::DeviceDescriptor::default()
        }))
        .ok()?;
        Some((adapter, device, queue))
    }
}

/// Headless device for tests that need a real GPU object (atlas upload
/// round-trips). `None` where no Vulkan adapter exists; tests skip
/// gracefully instead of failing (unlike `render_or_fail`, which must
/// stay loud).
#[cfg(test)]
pub(crate) fn test_headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    WgpuBackend::request_device(None).map(|(_, device, queue)| (device, queue))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display_list::{BorderStyle, FontKey};
    use crate::filter::FilterOp;

    /// The instance-buffer shrink policy: 120 consecutive passes of
    /// underuse (need <= cap/4) shrink back to the need's power of two
    /// (floored at 4096), and anything else holds or resets the streak.
    #[test]
    fn instance_buffer_shrink_policy() {
        let floor = WgpuBackend::INSTANCE_BUF_FLOOR;
        let streak = WgpuBackend::INSTANCE_SHRINK_STREAK;

        // A pipe that never draws never allocates.
        assert_eq!(WgpuBackend::instance_target_cap(None, 0, 0), None);
        // First draw allocates the power of two, floored.
        assert_eq!(
            WgpuBackend::instance_target_cap(None, 0, 100),
            Some((floor, 0))
        );
        assert_eq!(
            WgpuBackend::instance_target_cap(None, 0, 5000),
            Some((8192, 0))
        );
        // Growth on overflow resets the streak.
        assert_eq!(
            WgpuBackend::instance_target_cap(Some(4096), 77, 5000),
            Some((8192, 0))
        );
        // Underuse advances the streak without shrinking early.
        assert_eq!(
            WgpuBackend::instance_target_cap(Some(16384), 0, 100),
            Some((16384, 1))
        );
        assert_eq!(
            WgpuBackend::instance_target_cap(Some(16384), streak - 2, 100),
            Some((16384, streak - 1))
        );
        // The 120th consecutive underused pass shrinks to the floor.
        assert_eq!(
            WgpuBackend::instance_target_cap(Some(16384), streak - 1, 100),
            Some((floor, 0))
        );
        // A merely-small need still shrinks, to its own power of two.
        assert_eq!(
            WgpuBackend::instance_target_cap(Some(1 << 20), streak - 1, 5000),
            Some((8192, 0))
        );
        // No shrink when the shrunk size would not beat the capacity.
        assert_eq!(
            WgpuBackend::instance_target_cap(Some(floor), streak - 1, 1000),
            Some((floor, 0))
        );
        // Normal use (over a quarter of capacity) resets the streak.
        assert_eq!(
            WgpuBackend::instance_target_cap(Some(16384), 119, 5000),
            Some((16384, 0))
        );
        // A streak that would overflow u32 still shrinks instead.
        assert_eq!(
            WgpuBackend::instance_target_cap(Some(16384), u32::MAX, 100),
            Some((floor, 0))
        );
    }

    /// `prepare_lowering` flattens the whole group tree into combined
    /// per-pipe buffers with disjoint per-lowering slices: the root's
    /// bytes come first, then each group's, and every `Prepared::bases`
    /// entry points at its own lowering's instances — so one upload and
    /// one encoder serve the frame with no per-group re-uploads.
    #[test]
    fn prepare_lowering_assigns_disjoint_instance_ranges() {
        let mut root = Lowered::with_capacity(8);
        // Emit before pushing: `emit` derives the instance index from the
        // pre-push buffer length (see `solid_quad`).
        root.emit(PipeKind::Solid, None, 0);
        root.solid.extend_from_slice(&[1u8; SOLID_STRIDE]);
        root.emit(PipeKind::Solid, None, 0); // merges: one Draw, count 2
        root.solid.extend_from_slice(&[2u8; SOLID_STRIDE]);

        let mut group_low = Lowered::with_capacity(8);
        for _ in 0..3 {
            group_low.emit(PipeKind::Solid, None, 0); // one Draw, count 3
            group_low.solid.extend_from_slice(&[3u8; SOLID_STRIDE]);
        }
        root.groups.push(OpacityGroup {
            lowered: group_low,
            opacity: 0.5,
            blend: BlendMode::Normal,
            bounds: Rect {
                x: 0.0,
                y: 0.0,
                w: 10.0,
                h: 10.0,
            },
            transform: None,
            filter: Vec::new(),
        });
        root.ops.push(Op::Group(0));

        let mut combined: [Vec<u8>; 5] =
            [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        WgpuBackend::reserve_combined(&root, &mut combined);
        // Exact reservation: no realloc while flattening. The composite
        // slot reserves one zeroed instance per group op; the bytes are
        // filled at execute time (see `Op::Group` handling in
        // `execute_ops_into`).
        for (slot, bytes) in combined.iter().enumerate() {
            let want = match slot {
                0 => 5 * SOLID_STRIDE,
                3 => COMPOSITE_STRIDE,
                _ => 0,
            };
            assert!(
                bytes.capacity() >= want,
                "slot {slot}: reserved {} < needed {want}",
                bytes.capacity()
            );
        }
        let prepared = WgpuBackend::prepare_lowering(&root, &mut combined);

        let solid = PipeKind::Solid.slot();
        assert_eq!(prepared.bases[solid], 0, "root solid base");
        assert_eq!(
            prepared.groups[0].bases[solid], 2,
            "group solid base follows the root's 2 instances"
        );
        assert_eq!(combined[solid].len(), 5 * SOLID_STRIDE);

        let composite = PipeKind::Composite.slot();
        assert_eq!(prepared.bases[composite], 0, "root composite base");
        assert_eq!(
            combined[composite].len(),
            COMPOSITE_STRIDE,
            "one zeroed composite slot for the single group op"
        );
        assert!(
            combined[composite].iter().all(|b| *b == 0),
            "reserved composite slots start zeroed; the Op::Group handler fills its own slot at execute time"
        );
        assert_eq!(
            prepared.groups[0].bases[composite], 1,
            "group composite base follows the root's reserved slot"
        );

        // Draws address their own slices via base + start: the recorded
        // offsets stay valid without any re-upload.
        match &root.ops[0] {
            Op::Draw {
                pipe, start, count, ..
            } => {
                assert_eq!(*pipe, PipeKind::Solid);
                assert_eq!((*start, *count), (0, 2));
                let base = prepared.bases[pipe.slot()];
                assert_eq!((base + *start, *count), (0, 2));
            }
            _ => panic!("expected root Draw"),
        }
        match &root.groups[0].lowered.ops[0] {
            Op::Draw {
                pipe, start, count, ..
            } => {
                let base = prepared.groups[0].bases[pipe.slot()];
                assert_eq!((base + *start, *count), (2, 3));
            }
            _ => panic!("expected group Draw"),
        }
    }

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect { x, y, w, h }
    }

    /// Render `list` offscreen, failing LOUDLY when no GPU adapter is
    /// available: these tests must never silently skip. Under lavapipe
    /// (see the panic message) they run headless in CI.
    ///
    /// Builds its own [`FontCache`]: `FontKey(0)` in the test display
    /// lists is that cache's default face (the first key minted, resolved
    /// eagerly in `FontCache::new`).
    fn render_or_fail(list: &DisplayList, w: u32, h: u32) -> Vec<u8> {
        let fonts = crate::fonts::FontCache::new();
        WgpuBackend::render_offscreen(list, w, h, &fonts).unwrap_or_else(|| {
            panic!(
                "no Vulkan adapter available for wgpu backend test \
                 (set VK_ICD_FILENAMES to a lavapipe icd.json, e.g. \
                 /usr/share/vulkan/icd.d/lvp_icd.json \
                 with LD_LIBRARY_PATH pointing at its lib dir)"
            )
        })
    }

    fn nonzero_alpha_count(px: &[u8]) -> usize {
        let (chunks, _) = px.as_chunks::<4>();
        chunks.iter().filter(|p| p[3] > 0).count()
    }

    /// The font resolver hook currently installed on this thread's
    /// headless backend (`None` when the backend is not up yet), for
    /// tests that assert hook identity across `render_offscreen` calls.
    fn headless_installed_hook() -> Option<Hook> {
        super::HEADLESS_BACKEND.with(|cell| {
            cell.borrow()
                .0
                .as_ref()
                .and_then(|b| b.get_font_data.clone())
        })
    }

    /// The single bring-up choke point must yield all three GPU objects
    /// on the headless (fallback-tolerant) policy; loud failure, like
    /// `render_or_fail`, so CI notices a missing adapter.
    #[test]
    fn request_device_headless_brings_up_adapter() {
        let (adapter, _device, _queue) = WgpuBackend::request_device(None).unwrap_or_else(|| {
            panic!(
                "no Vulkan adapter available for wgpu backend test \
                 (set VK_ICD_FILENAMES to a lavapipe icd.json, e.g. \
                 /usr/share/vulkan/icd.d/lvp_icd.json \
                 with LD_LIBRARY_PATH pointing at its lib dir)"
            )
        });
        // The adapter is returned because the windowed caller needs it
        // for `Surface::get_default_config`; it must be a real,
        // queryable device.
        assert!(
            !adapter.get_info().name.is_empty(),
            "request_device must return a queryable adapter"
        );
    }

    #[test]
    fn solid_fill_covers_expected_pixels() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            rect: rect(100.0, 100.0, 200.0, 150.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        let px = render_or_fail(&list, 800, 600);
        assert_eq!(px.len(), 800 * 600 * 4);
        // Inside the rect: opaque red.
        let inside = |x: u32, y: u32| &px[(y as usize * 800 + x as usize) * 4..][..4];
        assert_eq!(inside(150, 150), &[255, 0, 0, 255]);
        assert_eq!(inside(299, 249), &[255, 0, 0, 255]);
        // Outside: transparent black (clear color).
        assert_eq!(inside(50, 50), &[0, 0, 0, 0]);
        assert_eq!(inside(799, 599), &[0, 0, 0, 0]);
    }

    /// Structural: an [`DisplayItem::EdgeGlow`] lowers through the
    /// solid-quad pipe and paints pixels; degenerate inputs (zero-area
    /// band, zero strength) lower to nothing without panicking.
    #[test]
    fn edge_glow_lowers_without_crash() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(0.0, 0.0, 200.0, 200.0),
        });
        list.push(DisplayItem::EdgeGlow {
            edge: Edge4::Top,
            rect: rect(0.0, 0.0, 200.0, 96.0),
            strength: 1.0,
        });
        list.push(DisplayItem::PopClip);
        list.push(DisplayItem::EdgeGlow {
            edge: Edge4::Left,
            rect: rect(0.0, 0.0, 0.0, 50.0),
            strength: 1.0,
        });
        list.push(DisplayItem::EdgeGlow {
            edge: Edge4::Right,
            rect: rect(150.0, 0.0, 50.0, 200.0),
            strength: 0.0,
        });
        let px = render_or_fail(&list, 200, 200);
        assert_eq!(px.len(), 200 * 200 * 4);
        // The glow painted over the clear color at the top edge...
        let top = &px[(10 * 200 + 100) * 4..][..4];
        assert!(top[3] > 0, "glow paints pixels, got {top:?}");
        // ...and nothing below the band.
        let below = &px[(150 * 200 + 100) * 4..][..4];
        assert_eq!(below, &[0, 0, 0, 0]);
    }

    /// CPU and GPU backends agree pixel-for-pixel (within rounding) on a
    /// mixed scene: solid fills, an isolated opacity group, and a clip.
    #[test]
    fn cpu_gpu_pixel_agreement() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            rect: rect(10.0, 10.0, 60.0, 60.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(40.0, 40.0, 60.0, 60.0),
            color: Color::rgba8(0, 0, 255, 255),
        });
        list.push(DisplayItem::PopOpacity);
        list.push(DisplayItem::PushClip {
            rect: rect(120.0, 10.0, 40.0, 40.0),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(100.0, 0.0, 100.0, 100.0),
            color: Color::rgba8(0, 255, 0, 255),
        });
        list.push(DisplayItem::PopClip);

        let gpu = render_or_fail(&list, 200, 100);
        let cpu = crate::cpu::CpuBackend::render_to_rgba(&list, 200, 100);
        assert_eq!(gpu.len(), cpu.len());
        for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert!(
                (g as i16 - c as i16).abs() <= 2,
                "pixel byte {i} disagrees: gpu={g} cpu={c}"
            );
        }
    }

    /// CPU and GPU backends agree (within tolerance) on a blurred
    /// outset box-shadow: both center the blur transition on the shape
    /// edge (CPU: gaussian σ = blur/2; GPU: centered smoothstep — an
    /// approximation, so the tolerance is wider than the pixel-exact
    /// agreement test above) and both knock the sharp border box out
    /// of outset coverage (CSS Backgrounds 3 §7.2).
    #[test]
    fn cpu_gpu_blurred_shadow_agreement() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::BoxShadow {
            rect: rect(50.0, 50.0, 40.0, 40.0),
            radii: [0.0; 4],
            color: Color::rgba8(255, 0, 0, 255),
            offset: (6.0, 4.0),
            blur: 8.0,
            spread: 6.0,
            inset: false,
        });
        let gpu = render_or_fail(&list, 200, 200);
        let cpu = crate::cpu::CpuBackend::render_to_rgba(&list, 200, 200);
        assert_eq!(gpu.len(), cpu.len());
        // Per-pixel agreement within the blur-model tolerance: the GPU's
        // centered smoothstep only approximates the CPU's gaussian (its
        // tail falls to zero faster, and the SDF evaluation is not
        // separable like the CPU's H/V passes, so corners differ most —
        // measured max diff 94). RGB is compared premultiplied: where
        // alpha is ~0 the CPU keeps the unmultiplied color while the
        // GPU's blend folds it to black, so straight RGB is meaningless
        // there.
        let mut max_diff = 0i16;
        let (gchunks, _) = gpu.as_chunks::<4>();
        let (cchunks, _) = cpu.as_chunks::<4>();
        for (p, (g, c)) in gchunks.iter().zip(cchunks.iter()).enumerate() {
            let da = (g[3] as i16 - c[3] as i16).abs();
            max_diff = max_diff.max(da);
            assert!(
                da <= 96,
                "pixel {p} alpha disagrees: gpu={} cpu={}",
                g[3],
                c[3]
            );
            for b in 0..3 {
                let gp = g[b] as u16 * g[3] as u16 / 255;
                let cp = c[b] as u16 * c[3] as u16 / 255;
                let d = gp.abs_diff(cp) as i16;
                max_diff = max_diff.max(d);
                assert!(
                    d <= 96,
                    "pixel {p} channel {b} disagrees: gpu={gp} cpu={cp}"
                );
            }
        }
        assert!(max_diff > 0, "expected the fixture to paint a shadow");
        // Blur-centering probe: 1.5px outside the shape's straight right
        // edge the centered model is well into the falloff (measured
        // gpu=59, cpu=95). The old edge-outside model still had ~90%
        // coverage here (alpha 230), so this pins the centering.
        let alpha_at = |buf: &[u8], x: u32, y: u32| buf[(y as usize * 200 + x as usize) * 4 + 3];
        let g = alpha_at(&gpu, 103, 74);
        assert!(
            (20..=140).contains(&g),
            "gpu blur falloff misplaced at (103,74): alpha={g}"
        );
        // The border-box interior stays hollow on both backends (the
        // outset knockout).
        assert_eq!(alpha_at(&gpu, 70, 70), 0);
        assert_eq!(alpha_at(&cpu, 70, 70), 0);
    }

    /// An isolated opacity group with fractional paint bounds composites
    /// onto the same device pixels as the CPU: the group texture
    /// rasterizes at the true device scale on the outward-snapped pixel
    /// grid — the same grid as the CPU's group pixmap — instead of
    /// mapping the fractional page-px bounds onto the whole texture
    /// (which rasterized at `tw / bounds.w` device px per page px and
    /// resampled on composite, shifting interior content by up to half a
    /// pixel). Two paint items force a real isolated group (no fold fast
    /// path); the 1 px white line has integral edges so its interior
    /// pixels must agree byte-for-byte, while the red rect's fractional
    /// edges use the touched-set convention (the backends rasterize
    /// fractional rect edges differently: pixel-center on GPU, analytic
    /// on CPU).
    #[test]
    fn fractional_group_bounds_agree_with_cpu() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 200.0, 100.0),
            color: Color::rgba8(32, 64, 128, 255),
        });
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        // Fractional group bounds (10.25, 10.5, 50.5, 30.25).
        list.push(DisplayItem::FillRect {
            rect: rect(10.25, 10.5, 50.5, 30.25),
            color: Color::rgba8(200, 40, 40, 255),
        });
        // Sharp integral feature inside the group: a half-pixel scale
        // error in the group texture moves it off its device pixel.
        list.push(DisplayItem::FillRect {
            rect: rect(40.0, 11.0, 1.0, 29.0),
            color: Color::rgba8(255, 255, 255, 255),
        });
        list.push(DisplayItem::PopOpacity);

        let gpu = render_or_fail(&list, 200, 100);
        let cpu = crate::cpu::CpuBackend::render_to_rgba(&list, 200, 100);
        assert_eq!(gpu.len(), cpu.len());
        // Touched set: the same pixels are painted (no leak or shift).
        for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            if i % 4 == 3 {
                assert_eq!(
                    (g == 0),
                    (c == 0),
                    "alpha touched-set disagrees at pixel {}: gpu={g} cpu={c}",
                    i / 4
                );
            }
        }
        // Interior pixels (away from the red rect's fractional edges at
        // x=10/60, y=10/40) agree byte-for-byte within rounding. The
        // white line at x=40 is the scale-error detector: the old
        // whole-texture mapping rasterized it half a texel off and the
        // composite's linear resample turned it pink.
        for y in 20..35 {
            for x in 20..50 {
                let i = (y * 200 + x) * 4;
                for b in 0..4 {
                    assert!(
                        (gpu[i + b] as i16 - cpu[i + b] as i16).abs() <= 2,
                        "pixel ({x},{y}) byte {b} disagrees: gpu={} cpu={}",
                        gpu[i + b],
                        cpu[i + b]
                    );
                }
            }
        }
        // Spot check: the line pixel is white at 0.5 opacity over the
        // blue backdrop on both backends: (255*.5+32*.5, 255*.5+64*.5,
        // 255*.5+128*.5) = (144, 160, 192).
        for px in [&gpu[..], &cpu[..]] {
            let p = &px[(25 * 200 + 40) * 4..][..4];
            assert!(
                (p[0] as i16 - 144).abs() <= 2
                    && (p[1] as i16 - 160).abs() <= 2
                    && (p[2] as i16 - 192).abs() <= 2,
                "line pixel not white-at-half-opacity: {p:?}"
            );
        }
    }

    #[test]
    fn border_tessellation_reaches_gpu() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::Border {
            rect: rect(10.0, 10.0, 100.0, 60.0),
            widths: [4.0, 4.0, 4.0, 4.0],
            styles: [BorderStyle::Solid; 4],
            colors: [Color::rgba8(0, 0, 255, 255); 4],
            radii: [0.0; 4],
        });
        let px = render_or_fail(&list, 800, 600);
        // Top border strip should be blue; the interior stays clear.
        let at = |x: u32, y: u32| &px[(y as usize * 800 + x as usize) * 4..][..4];
        assert_eq!(at(50, 11), &[0, 0, 255, 255]);
        assert_eq!(at(50, 40), &[0, 0, 0, 0]);
    }

    #[test]
    fn glyph_quads_sample_atlas_coverage() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::TextRun {
            text: Arc::from("A"),
            text_range: 0..1,
            rect: rect(20.0, 20.0, 100.0, 20.0),
            baseline: 32.0,
            font_key: FontKey(0),
            size_px: 16.0,
            color: Color::rgba8(0, 255, 0, 255),
        });
        let px = render_or_fail(&list, 800, 600);
        // The 16x16 cell at (20,20) must contain covered (green) pixels
        // and clear pixels: a real 'A' rasterized from the cache's default
        // face (FontKey(0)) has ink and empty cell area.
        let mut green = 0;
        let mut clear = 0;
        for y in 20..36u32 {
            for x in 20..36u32 {
                let p = &px[(y as usize * 800 + x as usize) * 4..][..4];
                if p[1] > 0 && p[3] > 0 {
                    green += 1;
                } else if p[3] == 0 {
                    clear += 1;
                }
            }
        }
        assert!(green > 50, "expected glyph coverage, got {green}");
        assert!(clear > 0, "expected clear pixels in glyph cell");
    }

    #[test]
    fn image_upload_and_sampling() {
        // 2x2 test image: red, green / blue, white.
        let rgba: Vec<u8> = vec![
            255, 0, 0, 255, 0, 255, 0, 255, //
            0, 0, 255, 255, 255, 255, 255, 255,
        ];
        let mut list = DisplayList::new();
        let id = ImageId(1);
        list.add_image(id, DecodedImage::from_rgba(2, 2, rgba));
        list.push(DisplayItem::Image {
            rect: rect(0.0, 0.0, 2.0, 2.0),
            image: id,
        });
        let px = render_or_fail(&list, 800, 600);
        let at = |x: u32, y: u32| &px[(y as usize * 800 + x as usize) * 4..][..4];
        // Nearest-ish sampling of a 2x2 image stretched over 2x2 px: each
        // output pixel should be close to its texel (linear filtering may
        // blend edges slightly; check dominant channels).
        assert!(
            at(0, 0)[0] > 200,
            "top-left should be red-ish: {:?}",
            at(0, 0)
        );
        assert!(
            at(1, 1)[0] > 200 && at(1, 1)[1] > 200 && at(1, 1)[2] > 200,
            "bottom-right should be white-ish: {:?}",
            at(1, 1)
        );
    }

    /// Build a display list holding `MAX_CACHED_IMAGES + 1` distinct 1x1
    /// images, mirroring the adversarial repro harness.
    fn many_distinct_images_list() -> DisplayList {
        let mut list = DisplayList::new();
        for i in 0..=MAX_CACHED_IMAGES as u32 {
            let px = vec![(i % 256) as u8, 0, 0, 255];
            let id = ImageId(i);
            list.add_image(id, DecodedImage::from_rgba(1, 1, px));
            list.push(DisplayItem::Image {
                rect: rect((i % 64) as f32, (i / 64) as f32, 1.0, 1.0),
                image: id,
            });
        }
        list
    }

    /// Regression: lowering more than `MAX_CACHED_IMAGES` distinct
    /// images in one frame must not evict textures for ops already
    /// lowered. The old mid-frame `entries.clear()` wiped the first 512
    /// textures when the 513th image uploaded, and execution then
    /// panicked on "image texture missing for draw".
    #[test]
    fn image_cache_survives_past_capacity_in_one_frame() {
        let list = many_distinct_images_list();
        let px = render_or_fail(&list, 64, 64);
        assert_eq!(px.len(), 64 * 64 * 4);
        // The old mid-frame clear panicked instead of rendering; every
        // 1x1 image is opaque, so nonzero alpha must cover the grid.
        assert!(
            nonzero_alpha_count(&px) > MAX_CACHED_IMAGES,
            "expected all 513 image pixels painted"
        );
    }

    /// Lowering-level half of the 513-image regression: after lowering
    /// the list, the cache still holds every entry uploaded during that
    /// lowering (no mid-frame eviction).
    #[test]
    fn lowering_keeps_cache_entries_past_capacity() {
        let list = many_distinct_images_list();
        let mut backend = WgpuBackend::new_headless().unwrap_or_else(|| {
            panic!(
                "no Vulkan adapter available for wgpu backend test \
                 (set VK_ICD_FILENAMES to a lavapipe icd.json)"
            )
        });
        let _ = backend.lower(&list, (1.0, 1.0));
        assert_eq!(
            backend.image_cache.entries.len(),
            MAX_CACHED_IMAGES + 1,
            "lowering must not evict entries uploaded earlier in the frame"
        );
        for i in 0..=MAX_CACHED_IMAGES as u32 {
            assert!(
                backend.image_cache.entries.contains_key(&ImageId(i)),
                "cache lost image {i} uploaded earlier in the frame"
            );
        }
    }

    /// Lowering more isolated opacity groups than `MAX_OPACITY_GROUPS`
    /// caps the group count instead of pinning a full-target texture per
    /// group: the 32 outermost levels stay isolated, deeper ones fold.
    /// Group discovery is O(items) via the precompute, so even 1000-deep
    /// nesting lowers without re-scanning.
    #[test]
    fn opacity_group_count_capped_during_lower() {
        let mut list = DisplayList::new();
        for _ in 0..1000 {
            list.push(DisplayItem::PushOpacity { opacity: 0.5 });
            // Two rects per level: every level is isolated (never the
            // single-item fold fast path).
            list.push(DisplayItem::FillRect {
                rect: rect(0.0, 0.0, 1.0, 1.0),
                color: Color::rgba8(255, 0, 0, 255),
            });
            list.push(DisplayItem::FillRect {
                rect: rect(2.0, 2.0, 1.0, 1.0),
                color: Color::rgba8(0, 255, 0, 255),
            });
        }
        for _ in 0..1000 {
            list.push(DisplayItem::PopOpacity);
        }
        let mut backend = WgpuBackend::new_headless().unwrap_or_else(|| {
            panic!(
                "no Vulkan adapter available for wgpu backend test \
                 (set VK_ICD_FILENAMES to a lavapipe icd.json)"
            )
        });
        let lowered = backend.lower(&list, (1.0, 1.0));
        // Nested groups live inside child `Lowered` values, so count the
        // whole tree: the cap is global (`parts.group_count` threads
        // through the recursive `lower_range`), not per level.
        fn count_groups(lowered: &Lowered) -> usize {
            lowered
                .groups
                .iter()
                .map(|g| 1 + count_groups(&g.lowered))
                .sum()
        }
        let total = count_groups(&lowered);
        assert!(
            total <= MAX_OPACITY_GROUPS as usize,
            "isolated groups not capped: {total}"
        );
        assert!(
            total >= 1,
            "expected at least one isolated group from the nesting"
        );
    }

    /// One-line text list + headless backend with the default test face,
    /// for lowering-level text assertions.
    fn text_backend() -> (DisplayList, WgpuBackend) {
        let mut list = DisplayList::new();
        list.push(DisplayItem::TextRun {
            text: Arc::from("hello"),
            text_range: 0..5,
            rect: rect(20.0, 20.0, 200.0, 24.0),
            baseline: 36.0,
            font_key: FontKey(0),
            size_px: 16.0,
            color: Color::rgba8(0, 255, 0, 255),
        });
        let fonts = crate::fonts::FontCache::new();
        let mut backend = WgpuBackend::new_headless().unwrap_or_else(|| {
            panic!(
                "no Vulkan adapter available for wgpu backend test \
                 (set VK_ICD_FILENAMES to a lavapipe icd.json)"
            )
        });
        backend.set_font_data(fonts.resolver());
        (list, backend)
    }

    /// First 8 floats (rect vec4 + uv vec4) of every glyph instance.
    fn glyph_rect_uvs(lowered: &Lowered) -> Vec<[f32; 8]> {
        let (chunks, _) = lowered.glyph.as_chunks::<GLYPH_STRIDE>();
        chunks
            .iter()
            .map(|c| {
                let mut out = [0.0f32; 8];
                for (i, slot) in out.iter_mut().enumerate() {
                    let bytes: [u8; 4] = c[i * 4..(i + 1) * 4].try_into().unwrap();
                    *slot = f32::from_ne_bytes(bytes);
                }
                out
            })
            .collect()
    }

    /// Text at device scale 2.0 rasterizes at 2x (crisp on hidpi) but lays
    /// out on the same page-px grid: the glyph quads keep their page-px
    /// sizes while their atlas UVs point at the larger ink.
    #[test]
    fn dpr_text_rasterizes_at_device_px() {
        let (list, mut backend) = text_backend();
        let at_1x = glyph_rect_uvs(&backend.lower(&list, (1.0, 1.0)));
        let at_2x = glyph_rect_uvs(&backend.lower(&list, (2.0, 2.0)));
        assert!(!at_1x.is_empty(), "expected glyph quads");
        assert_eq!(
            at_1x.len(),
            at_2x.len(),
            "glyph count must not change with scale"
        );
        for (a, b) in at_1x.iter().zip(at_2x.iter()) {
            // Same page-px ink size (32px ink halved back to the page
            // grid). Tolerance is a full page px: each ink edge rounds
            // to a whole device px independently at each raster size, so
            // e.g. a 9px-wide 16px raster can become an 8px-wide page
            // quad from the 32px raster. A missing device->page
            // conversion would double sizes instead (diff == size).
            for i in 2..4 {
                assert!(
                    (a[i] - b[i]).abs() <= 1.0,
                    "glyph size diverged at scale 2: {a:?} vs {b:?}"
                );
            }
            // ...but different atlas cells: the ink was re-rasterized at
            // 32px, not reused from the 16px cells.
            assert_ne!(
                &a[4..8],
                &b[4..8],
                "glyph uv unchanged at scale 2: ink was not re-rasterized"
            );
        }
        // The first glyph's page-px position is also stable within the
        // same raster rounding (later glyphs accumulate per-advance
        // rounding, so only the first is pinned).
        for i in 0..2 {
            assert!(
                (at_1x[0][i] - at_2x[0][i]).abs() <= 1.0,
                "glyph position diverged at scale 2"
            );
        }
    }

    /// Degenerate scales (zero, negative, NaN, infinite) behave like 1.0
    /// instead of dividing glyph placement by zero or emitting NaN quads.
    /// Degenerate on one axis only sanitizes that axis; the other axis
    /// keeps its scale.
    #[test]
    fn lower_sanitizes_degenerate_scale() {
        let (list, mut backend) = text_backend();
        let at_1x = glyph_rect_uvs(&backend.lower(&list, (1.0, 1.0)));
        assert!(!at_1x.is_empty(), "expected glyph quads");
        for scale in [0.0, -2.0, f32::NAN, f32::INFINITY] {
            let rects = glyph_rect_uvs(&backend.lower(&list, (scale, scale)));
            assert_eq!(rects.len(), at_1x.len(), "scale {scale}");
            for (a, b) in rects.iter().zip(at_1x.iter()) {
                for i in 0..8 {
                    assert!(
                        (a[i] - b[i]).abs() < 1e-3,
                        "scale {scale} diverged from 1.0: {a:?} vs {b:?}"
                    );
                }
            }
        }
        // One degenerate axis: placement must match the uniform frame at
        // the surviving axis's scale (2x raster on both axes here).
        let at_2x = glyph_rect_uvs(&backend.lower(&list, (2.0, 2.0)));
        for degenerate in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            for pair in [(degenerate, 2.0), (2.0, degenerate)] {
                let rects = glyph_rect_uvs(&backend.lower(&list, pair));
                assert_eq!(rects.len(), at_2x.len(), "scale {pair:?}");
                for (a, b) in rects.iter().zip(at_2x.iter()) {
                    for i in 0..8 {
                        assert!(
                            (a[i] - b[i]).abs() < 1e-3,
                            "scale {pair:?} diverged from 2x: {a:?} vs {b:?}"
                        );
                    }
                }
            }
        }
    }

    /// Non-uniform target/viewport aspect: glyphs rasterize at the denser
    /// axis and placement divides by that raster scale, so a (1x, 2x)
    /// frame lays out exactly like the uniform (2x, 2x) frame — the old
    /// width-ratio-only scale shaped vertical ink at the horizontal
    /// density instead.
    #[test]
    fn non_uniform_scale_rasterizes_at_denser_axis() {
        let (list, mut backend) = text_backend();
        let non_uniform = glyph_rect_uvs(&backend.lower(&list, (1.0, 2.0)));
        let uniform = glyph_rect_uvs(&backend.lower(&list, (2.0, 2.0)));
        assert!(!non_uniform.is_empty(), "expected glyph quads");
        assert_eq!(non_uniform.len(), uniform.len());
        // Same raster size on both runs, so the atlas serves identical
        // cells: rects and UVs must match exactly.
        for (a, b) in non_uniform.iter().zip(uniform.iter()) {
            for i in 0..8 {
                assert!(
                    (a[i] - b[i]).abs() < 1e-6,
                    "non-uniform (1, 2) diverged from uniform 2x: {a:?} vs {b:?}"
                );
            }
        }
    }

    /// `render_offscreen` reuses one headless backend per thread instead
    /// of bringing up a fresh device per call: after a successful render
    /// the thread-local slot stays populated, and a second render on the
    /// same thread reuses it (identical pixels, no re-bring-up).
    #[test]
    fn render_offscreen_reuses_headless_device() {
        fn cached() -> bool {
            HEADLESS_BACKEND.with(|slot| slot.borrow().0.is_some())
        }
        // Clear any backend cached by an earlier test on this thread, so
        // the assertions below observe this test's own bring-up.
        HEADLESS_BACKEND.with(|slot| slot.borrow_mut().0 = None);
        assert!(!cached());
        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 4.0, 4.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        let fonts = crate::fonts::FontCache::new();
        let first = WgpuBackend::render_offscreen(&list, 4, 4, &fonts)
            .expect("no Vulkan adapter available for wgpu backend test");
        assert!(
            cached(),
            "render_offscreen must leave the headless backend cached"
        );
        let fonts2 = crate::fonts::FontCache::new();
        let second = WgpuBackend::render_offscreen(&list, 4, 4, &fonts2)
            .expect("cached headless backend must render");
        assert!(
            cached(),
            "second render_offscreen must reuse the cached backend"
        );
        assert_eq!(first, second, "reused device must render identical pixels");
    }

    /// `shutdown_headless_backend` drains the calling thread's cached
    /// backend through the real destructor (not the TLS-teardown
    /// `mem::forget`), leaving the slot empty so the next
    /// `render_offscreen` re-brings-up cleanly. Populating the slot
    /// directly exercises the drop path without depending on a render.
    #[test]
    fn shutdown_headless_backend_drains_slot() {
        fn cached() -> bool {
            HEADLESS_BACKEND.with(|slot| slot.borrow().0.is_some())
        }
        // Start clean: an earlier test on this thread may have cached.
        HEADLESS_BACKEND.with(|slot| slot.borrow_mut().0 = None);
        shutdown_headless_backend();
        assert!(!cached(), "shutdown on an empty slot is a no-op");
        // Populate the slot if an adapter exists (stays empty on
        // adapter-less machines, where the shutdown below is still a
        // safe no-op).
        HEADLESS_BACKEND.with(|slot| {
            slot.borrow_mut().0 = WgpuBackend::new_headless();
        });
        shutdown_headless_backend();
        assert!(
            !cached(),
            "shutdown must drain the cached backend via its real destructor"
        );
        // The slot is usable again afterwards.
        HEADLESS_BACKEND.with(|slot| {
            slot.borrow_mut().0 = WgpuBackend::new_headless();
        });
        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 4.0, 4.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        let fonts = crate::fonts::FontCache::new();
        if WgpuBackend::render_offscreen(&list, 4, 4, &fonts).is_some() {
            assert!(cached(), "render_offscreen re-caches after shutdown");
        }
        shutdown_headless_backend();
        assert!(!cached());
    }

    /// Zero-size offscreen targets return empty pixels instead of handing
    /// a zero extent to `create_texture` (a wgpu validation error).
    #[test]
    fn offscreen_with_viewport_zero_size_returns_empty() {
        let (list, mut backend) = text_backend();
        let viewport = Size { w: 800.0, h: 600.0 };
        assert!(
            backend
                .render_offscreen_with_viewport(&list, viewport, 0, 600)
                .is_empty()
        );
        assert!(
            backend
                .render_offscreen_with_viewport(&list, viewport, 800, 0)
                .is_empty()
        );
    }

    /// Frame-boundary eviction is precise, not a blind clear: lowering a
    /// second frame that references only one image evicts the other 512
    /// stale entries but keeps the still-referenced one hot (no
    /// re-upload).
    #[test]
    fn image_cache_evicts_only_unreferenced_entries_at_frame_start() {
        let list = many_distinct_images_list();
        let mut backend = WgpuBackend::new_headless().unwrap_or_else(|| {
            panic!(
                "no Vulkan adapter available for wgpu backend test \
                 (set VK_ICD_FILENAMES to a lavapipe icd.json)"
            )
        });
        let _ = backend.lower(&list, (1.0, 1.0));
        assert_eq!(backend.image_cache.entries.len(), MAX_CACHED_IMAGES + 1);

        let mut second = DisplayList::new();
        second.add_image(
            ImageId(0),
            DecodedImage::from_rgba(1, 1, vec![7, 0, 0, 255]),
        );
        second.push(DisplayItem::Image {
            rect: rect(0.0, 0.0, 1.0, 1.0),
            image: ImageId(0),
        });
        let _ = backend.lower(&second, (1.0, 1.0));
        // ImageId(0) was re-registered with different bytes, so its entry
        // was replaced on the hash mismatch; every other id is stale and
        // evicted at the frame boundary.
        assert_eq!(backend.image_cache.entries.len(), 1);
        assert!(backend.image_cache.entries.contains_key(&ImageId(0)));
    }

    /// Byte-budget eviction is LRU-first: six 128x128 images (64 KiB
    /// each) over a tiny 192 KiB test budget. The budget is enforced on
    /// the real path — [`ImageCache::reserve`] during lowering and
    /// [`ImageCache::evict_for_budget`] after submit — never by evicting
    /// textures the in-flight frame referenced: every emitted draw still
    /// resolves its texture.
    #[test]
    fn image_cache_evicts_lru_first_within_byte_budget() {
        let mut backend = WgpuBackend::new_headless().unwrap_or_else(|| {
            panic!(
                "no Vulkan adapter available for wgpu backend test \
                 (set VK_ICD_FILENAMES to a lavapipe icd.json)"
            )
        });
        // 64 KiB per image; the test budget holds exactly three. Set on
        // the cache so lowering and the post-submit pass enforce it for
        // real — no manual eviction call below.
        const BUDGET: usize = 192 * 1024;
        backend.image_cache.budget_bytes = BUDGET;
        let mut list = DisplayList::new();
        for i in 0..6u32 {
            let px = vec![(i * 37 % 256) as u8; 128 * 128 * 4];
            list.add_image(ImageId(i), DecodedImage::from_rgba(128, 128, px));
            list.push(DisplayItem::Image {
                rect: rect((i % 3) as f32 * 128.0, (i / 3) as f32 * 128.0, 128.0, 128.0),
                image: ImageId(i),
            });
        }
        assert!(
            backend.render_to_rgba(&list, 384, 256).is_some(),
            "frame with 6 images must render"
        );
        // The real path held the budget: six 64 KiB uploads went in, the
        // three least-recently-used came back out at submit.
        assert!(
            backend.image_cache.total_bytes <= BUDGET,
            "cache holds {} bytes over a {BUDGET}-byte budget",
            backend.image_cache.total_bytes
        );
        // LRU-first: images 0..2 uploaded first, so they go and the
        // three most recent survive.
        for i in 0..3u32 {
            assert!(
                !backend.image_cache.entries.contains_key(&ImageId(i)),
                "image {i} should have been evicted LRU-first"
            );
        }
        for i in 3..6u32 {
            assert!(
                backend.image_cache.entries.contains_key(&ImageId(i)),
                "image {i} should survive as most-recently-used"
            );
        }
        // A second frame referencing only survivors must not hit the
        // "texture missing for draw" skip: every draw resolves and
        // paints.
        let mut second = DisplayList::new();
        for i in 3..6u32 {
            second.add_image(
                ImageId(i),
                DecodedImage::from_rgba(128, 128, vec![(i * 37 % 256) as u8; 128 * 128 * 4]),
            );
            second.push(DisplayItem::Image {
                rect: rect((i % 3) as f32 * 128.0, 0.0, 128.0, 128.0),
                image: ImageId(i),
            });
        }
        let px = backend
            .render_to_rgba(&second, 384, 128)
            .expect("second frame renders");
        assert!(
            nonzero_alpha_count(&px) > 1000,
            "survivor images must paint"
        );
    }

    /// [`ImageCache::reserve`] evicts unpinned entries mid-frame when an
    /// upload would exceed the byte budget — never entries the
    /// in-progress lowering referenced. Evicted entries re-upload on
    /// demand when referenced later in the same frame: the cache stays
    /// bounded and every draw still resolves its texture (no "texture
    /// missing for draw" skip).
    #[test]
    fn image_cache_reserve_evicts_unpinned_mid_frame() {
        let mut backend = WgpuBackend::new_headless().unwrap_or_else(|| {
            panic!(
                "no Vulkan adapter available for wgpu backend test \
                 (set VK_ICD_FILENAMES to a lavapipe icd.json)"
            )
        });
        // Frame 1: three 64 KiB images under the live budget.
        let mut first = DisplayList::new();
        for i in 0..3u32 {
            first.add_image(
                ImageId(i),
                DecodedImage::from_rgba(128, 128, vec![(i * 37 + 11) as u8; 128 * 128 * 4]),
            );
            first.push(DisplayItem::Image {
                rect: rect(i as f32 * 128.0, 0.0, 128.0, 128.0),
                image: ImageId(i),
            });
        }
        assert!(
            backend.render_to_rgba(&first, 384, 128).is_some(),
            "first frame renders"
        );
        assert_eq!(backend.image_cache.total_bytes, 3 * 64 * 1024);

        // Frame 2 under a one-image budget, referencing two *new* images
        // before the three old ones: the new uploads must evict the cold
        // (not-yet-referenced) entries mid-frame to fit, and the old ones
        // re-upload on demand when their items lower.
        backend.image_cache.budget_bytes = 64 * 1024;
        let mut second = DisplayList::new();
        for (k, i) in [3u32, 4, 0, 1, 2].iter().enumerate() {
            second.add_image(
                ImageId(*i),
                DecodedImage::from_rgba(128, 128, vec![(i * 37 + 11) as u8; 128 * 128 * 4]),
            );
            second.push(DisplayItem::Image {
                rect: rect(k as f32 * 128.0, 0.0, 128.0, 128.0),
                image: ImageId(*i),
            });
        }
        let px = backend
            .render_to_rgba(&second, 640, 128)
            .expect("second frame renders");
        // Bounded: only the most-recently-used single image survives.
        assert_eq!(backend.image_cache.total_bytes, 64 * 1024);
        assert_eq!(backend.image_cache.entries.len(), 1);
        assert!(
            backend.image_cache.entries.contains_key(&ImageId(2)),
            "the last-referenced image must survive"
        );
        // Correct: all five images painted, including the ones evicted
        // mid-frame and re-uploaded later in the same frame.
        assert!(
            nonzero_alpha_count(&px) > 1000,
            "evicted-then-reuploaded images must paint"
        );
    }

    /// Isolated opacity groups render into bounds-sized textures, not
    /// full-target ones: a small two-item group on a large target pools
    /// a small texture, and the group region still paints.
    #[test]
    fn opacity_group_allocates_bounded_target() {
        let mut backend = WgpuBackend::new_headless().unwrap_or_else(|| {
            panic!(
                "no Vulkan adapter available for wgpu backend test \
                 (set VK_ICD_FILENAMES to a lavapipe icd.json)"
            )
        });
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(100.0, 100.0, 100.0, 50.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(120.0, 110.0, 100.0, 50.0),
            color: Color::rgba8(0, 0, 255, 255),
        });
        list.push(DisplayItem::PopOpacity);
        let px = backend
            .render_to_rgba(&list, 800, 600)
            .expect("group frame renders");
        // The pooled group target must hug the group bounds
        // (100,100)-(220,160) at scale 1, not the 800x600 full target.
        assert!(
            !backend.group_target_pool.is_empty(),
            "group target should return to the pool after submit"
        );
        for t in &backend.group_target_pool {
            assert!(
                t.w < 800 && t.h < 600,
                "group target must not be full-target sized: {}x{}",
                t.w,
                t.h
            );
            assert!(
                t.w <= 130 && t.h <= 70,
                "group target should hug the group bounds, got {}x{}",
                t.w,
                t.h
            );
        }
        // Pixel sanity: the group region painted, the rest is empty.
        let at = |x: u32, y: u32| &px[(y as usize * 800 + x as usize) * 4..][..4];
        assert!(at(150, 125)[3] > 0, "group region must paint");
        assert_eq!(at(10, 10), &[0, 0, 0, 0], "outside the group stays empty");
    }

    /// A parent group's paint bounds include nested isolated groups: the
    /// parent's offscreen texture must cover the nested group's
    /// composited output, not just the parent's direct instances.
    /// Regression: `lowered_bounds` used to scan only the direct
    /// solid/glyph/image buffers, so a parent with small direct content
    /// but a far-reaching nested group sized its texture too small and
    /// clipped the nested composite.
    #[test]
    fn opacity_group_bounds_include_nested_groups() {
        let mut backend = WgpuBackend::new_headless().unwrap_or_else(|| {
            panic!(
                "no Vulkan adapter available for wgpu backend test \
                 (set VK_ICD_FILENAMES to a lavapipe icd.json)"
            )
        });
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        // Small direct content at (10,10)-(20,20)...
        list.push(DisplayItem::FillRect {
            rect: rect(10.0, 10.0, 10.0, 10.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        // ...plus a nested isolated group (two items: not foldable)
        // painting far away at (500,400)-(560,440).
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(500.0, 400.0, 60.0, 40.0),
            color: Color::rgba8(0, 255, 0, 255),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(510.0, 410.0, 40.0, 30.0),
            color: Color::rgba8(0, 0, 255, 255),
        });
        list.push(DisplayItem::PopOpacity);
        list.push(DisplayItem::PopOpacity);
        let lowered = backend.lower(&list, (1.0, 1.0));
        assert_eq!(lowered.groups.len(), 1, "outer group must lower isolated");
        assert_eq!(
            lowered.groups[0].lowered.groups.len(),
            1,
            "inner group must lower isolated, not fold"
        );
        let b = lowered.groups[0].bounds;
        assert!(
            b.x <= 10.0 && b.y <= 10.0,
            "parent bounds must start at the direct rect, got ({}, {})",
            b.x,
            b.y
        );
        assert!(
            b.x + b.w >= 560.0 && b.y + b.h >= 440.0,
            "parent bounds must cover the nested group, got ({}, {}) + ({}, {})",
            b.x,
            b.y,
            b.w,
            b.h
        );
        // The pooled texture hugs the union bounds (10,10)-(560,440),
        // not just the direct rect — and both regions paint.
        let px = backend
            .render_to_rgba(&list, 800, 600)
            .expect("nested-group frame renders");
        assert!(
            backend
                .group_target_pool
                .iter()
                .any(|t| t.w == 550 && t.h == 430),
            "outer group texture must be union-bounds sized, pool: {:?}",
            backend
                .group_target_pool
                .iter()
                .map(|t| (t.w, t.h))
                .collect::<Vec<_>>()
        );
        let at = |x: u32, y: u32| &px[(y as usize * 800 + x as usize) * 4..][..4];
        assert!(at(15, 15)[3] > 0, "direct rect must paint");
        assert!(at(530, 420)[3] > 0, "nested group region must paint");
    }

    /// Zero-size and device-oversized offscreen targets degrade to an
    /// empty vector instead of hitting `create_texture` validation (which
    /// the default uncaptured-error handler turns into a panic).
    #[test]
    fn offscreen_target_dims_are_validated() {
        let mut backend = WgpuBackend::new_headless().unwrap_or_else(|| {
            panic!(
                "no Vulkan adapter available for wgpu backend test \
                 (set VK_ICD_FILENAMES to a lavapipe icd.json)"
            )
        });
        let max_dim = backend.device.limits().max_texture_dimension_2d;
        let list = DisplayList::new();
        assert_eq!(backend.render_to_rgba(&list, 0, 16), Some(Vec::new()));
        assert_eq!(
            backend.render_to_rgba(&list, max_dim + 1, 16),
            Some(Vec::new())
        );
        assert_eq!(
            backend.render_offscreen_with_viewport(
                &list,
                Size { w: 1.0, h: 1.0 },
                max_dim + 1,
                max_dim + 1,
            ),
            Vec::new()
        );
    }

    #[test]
    fn clip_restricts_drawing() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(0.0, 0.0, 50.0, 50.0),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 200.0, 200.0),
            color: Color::rgba8(255, 255, 0, 255),
        });
        list.push(DisplayItem::PopClip);
        list.push(DisplayItem::FillRect {
            rect: rect(100.0, 100.0, 50.0, 50.0),
            color: Color::rgba8(255, 0, 255, 255),
        });
        let px = render_or_fail(&list, 800, 600);
        let at = |x: u32, y: u32| &px[(y as usize * 800 + x as usize) * 4..][..4];
        assert_eq!(at(25, 25), &[255, 255, 0, 255], "inside clip");
        assert_eq!(at(100, 25), &[0, 0, 0, 0], "clipped away");
        assert_eq!(at(125, 125), &[255, 0, 255, 255], "after pop");
    }

    #[test]
    fn opacity_multiplies_alpha() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: Color::rgba8(255, 255, 255, 255),
        });
        list.push(DisplayItem::PopOpacity);
        let px = render_or_fail(&list, 800, 600);
        let p = &px[(50 * 800 + 50) * 4..][..4];
        // White at 50% over transparent black: rgb stays 255 (src-over with
        // dst alpha 0 keeps src rgb), alpha ~128.
        assert_eq!(&p[0..3], &[255, 255, 255]);
        assert!(
            (p[3] as i32 - 128).abs() <= 1,
            "expected alpha ~128, got {}",
            p[3]
        );
    }

    #[test]
    fn decoration_and_marker_emit_quads() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::Decoration {
            rect: rect(10.0, 10.0, 100.0, 20.0),
            kind: DecoKind::Underline,
            color: Color::rgba8(0, 0, 0, 255),
            thickness: 2.0,
        });
        list.push(DisplayItem::ListMarker {
            rect: rect(200.0, 200.0, 12.0, 12.0),
            kind: crate::display_list::MarkerKind::Disc,
            color: Color::rgba8(0, 0, 0, 255),
            index: 1,
            font_key: FontKey(0),
        });
        let px = render_or_fail(&list, 800, 600);
        assert!(
            nonzero_alpha_count(&px) > 100,
            "decoration + marker should paint pixels"
        );
    }

    #[test]
    fn paint_trait_validates_list() {
        let (device, queue) = WgpuBackend::request_device(None)
            .map(|(_, device, queue)| (device, queue))
            .unwrap_or_else(|| panic!("no Vulkan adapter available for wgpu backend test"));
        let mut backend = WgpuBackend::new(&device, &queue);
        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 10.0, 10.0),
            color: Color::rgba8(1, 2, 3, 4),
        });
        // With no surface attached, paint lowers the list for validation
        // and drops the batches; it must not panic.
        backend.paint(&list, Size { w: 800.0, h: 600.0 }, (5.0, 5.0));
    }

    #[test]
    fn backend_new_builds_all_pipelines() {
        let (device, queue) = WgpuBackend::request_device(None)
            .map(|(_, device, queue)| (device, queue))
            .unwrap_or_else(|| panic!("no Vulkan adapter available for wgpu backend test"));
        let mut backend = WgpuBackend::new(&device, &queue);
        // Force pipeline creation for two formats without a surface.
        for format in [
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Bgra8Unorm,
        ] {
            let _ = backend.pipelines_for(format);
        }
        assert_eq!(backend.pipelines.len(), 2);
    }

    #[test]
    fn rounded_border_radii_reach_gpu() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::Border {
            rect: rect(10.0, 10.0, 100.0, 60.0),
            widths: [8.0, 8.0, 8.0, 8.0],
            styles: [BorderStyle::Solid; 4],
            colors: [Color::rgba8(255, 0, 0, 255); 4],
            radii: [20.0; 4],
        });
        let px = render_or_fail(&list, 800, 600);
        let at = |x: u32, y: u32| &px[(y as usize * 800 + x as usize) * 4..][..4];
        // Top-edge midpoint sits in the straight bar: red.
        assert_eq!(at(60, 12), &[255, 0, 0, 255]);
        // Interior stays clear.
        assert_eq!(at(60, 40), &[0, 0, 0, 0]);
    }

    #[test]
    fn decimal_marker_renders_numeral_from_index() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::ListMarker {
            rect: rect(200.0, 200.0, 12.0, 12.0),
            kind: MarkerKind::Decimal(7),
            color: Color::rgba8(0, 0, 0, 255),
            index: 3,
            font_key: FontKey(0),
        });
        let px = render_or_fail(&list, 800, 600);
        // The numeral comes from the 1-based `index` ("3") and paints glyph
        // pixels inside the marker box via the text path.
        let mut painted = 0;
        for y in 200..212u32 {
            for x in 200..224u32 {
                if px[(y as usize * 800 + x as usize) * 4 + 3] > 0 {
                    painted += 1;
                }
            }
        }
        assert!(painted > 10, "expected numeral text pixels, got {painted}");
    }

    /// The GPU list-marker path shapes decimal numerals with the
    /// item's `font_key`, not the fallback face: lower a marker whose
    /// font is `FontKey(7)` with a recording font hook and assert the
    /// hook was asked for key 7 (previously the path hardcoded
    /// `FontKey(0)` here while the CPU backend used the item's key).
    #[test]
    fn decimal_marker_shapes_with_item_font_key() {
        use std::sync::Mutex;
        let (device, queue) = WgpuBackend::request_device(None)
            .map(|(_, device, queue)| (device, queue))
            .unwrap_or_else(|| panic!("no Vulkan adapter available for wgpu backend test"));
        let mut backend = WgpuBackend::new(&device, &queue);
        let seen = std::sync::Arc::new(Mutex::new(Vec::new()));
        let seen_hook = std::sync::Arc::clone(&seen);
        backend.set_font_data(std::sync::Arc::new(move |key: FontKey| {
            seen_hook.lock().expect("lock").push(key);
            None
        }));
        let mut list = DisplayList::new();
        list.push(DisplayItem::ListMarker {
            rect: rect(200.0, 200.0, 12.0, 12.0),
            kind: MarkerKind::Decimal(7),
            color: Color::rgba8(0, 0, 0, 255),
            index: 3,
            font_key: FontKey(7),
        });
        let _ = backend.lower(&list, (1.0, 1.0));
        let seen = seen.lock().expect("lock").clone();
        assert_eq!(
            seen,
            vec![FontKey(7)],
            "marker numeral must resolve the item's font key, got {seen:?}"
        );
    }

    #[test]
    fn image_cache_is_per_list() {
        // Same ImageId, different lists: the second list's bytes must win.
        // (Regression: the texture cache was keyed by ImageId alone and
        // served the first list's texture for every later list.)
        fn list_with(id: ImageId, rgba: [u8; 16]) -> DisplayList {
            let mut list = DisplayList::new();
            list.add_image(id, DecodedImage::from_rgba(2, 2, rgba.to_vec()));
            list.push(DisplayItem::Image {
                rect: rect(0.0, 0.0, 2.0, 2.0),
                image: id,
            });
            list
        }
        let red = [
            255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
        ];
        let blue = [
            0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255,
        ];
        let px_red = render_or_fail(&list_with(ImageId(1), red), 8, 8);
        assert!(px_red[0] > 200 && px_red[2] < 50, "first list is red");
        let px_blue = render_or_fail(&list_with(ImageId(1), blue), 8, 8);
        assert!(
            px_blue[2] > 200 && px_blue[0] < 50,
            "second list must be blue, got {:?}",
            &px_blue[..4]
        );
    }

    #[test]
    fn malformed_image_bytes_are_skipped_not_panicked() {
        // Regression: `DecodedImage`'s fields are public, so a caller can
        // build one whose `rgba` length disagrees with the dimensions,
        // bypassing `DecodedImage::from_rgba`'s assert. The CPU backend
        // (`CpuBackend::image_pixmap`) skips such images; the GPU backend
        // used to reach `queue.write_texture` with 3 bytes where 16 were
        // required, a wgpu validation error that the default
        // uncaptured-error handler turns into a panic.
        let mut list = DisplayList::new();
        list.add_image(
            ImageId(7),
            DecodedImage {
                width: 2,
                height: 2,
                rgba: vec![0u8; 3], // 16 bytes required
            },
        );
        list.push(DisplayItem::Image {
            rect: rect(0.0, 0.0, 16.0, 16.0),
            image: ImageId(7),
        });
        // A known-good rect proves the rest of the pipeline survives the
        // skipped image.
        list.push(DisplayItem::FillRect {
            rect: rect(20.0, 20.0, 10.0, 10.0),
            color: Color::rgba8(0, 255, 0, 255),
        });
        // Must not panic: `render_or_fail` already fails loudly without an
        // adapter, so this exercises the real `ensure_image` path.
        let px = render_or_fail(&list, 64, 64);
        let at = |x: u32, y: u32| &px[(y as usize * 64 + x as usize) * 4..][..4];
        assert_eq!(at(8, 8), &[0, 0, 0, 0], "malformed image paints nothing");
        assert_eq!(at(25, 25), &[0, 255, 0, 255], "rest of scene paints");
    }

    #[test]
    fn scissor_subtracts_scroll() {
        // Clip rects are page px like everything else: with scroll (100,0)
        // the (100,100,50,50) clip lands at device (0,100,50,50).
        let (device, queue) = WgpuBackend::request_device(None)
            .map(|(_, device, queue)| (device, queue))
            .unwrap_or_else(|| panic!("no Vulkan adapter available for wgpu backend test"));
        let mut backend = WgpuBackend::new(&device, &queue);
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(100.0, 100.0, 50.0, 50.0),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 200.0, 200.0),
            color: Color::rgba8(255, 255, 0, 255),
        });
        let target = backend.make_target(800, 600);
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        backend.render_into(
            &view,
            wgpu::TextureFormat::Rgba8Unorm,
            &list,
            Size { w: 800.0, h: 600.0 },
            (100.0, 0.0),
            800,
            600,
        );
        let px =
            WgpuBackend::readback(&device, &queue, &target, 800, 600).expect("readback failed");
        let at = |x: u32, y: u32| &px[(y as usize * 800 + x as usize) * 4..][..4];
        // Device (25,125) == page (125,125): inside clip and fill.
        assert_eq!(at(25, 125), &[255, 255, 0, 255]);
        // Device (150,125) == page (250,125): outside the clip.
        assert_eq!(at(150, 125), &[0, 0, 0, 0]);
    }

    #[test]
    fn opacity_folds_into_image_alpha() {
        let rgba = vec![
            255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
        ];
        let mut list = DisplayList::new();
        list.add_image(ImageId(1), DecodedImage::from_rgba(2, 2, rgba));
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::Image {
            rect: rect(0.0, 0.0, 2.0, 2.0),
            image: ImageId(1),
        });
        list.push(DisplayItem::PopOpacity);
        let px = render_or_fail(&list, 8, 8);
        let p = &px[..4];
        // Red texel at 50% over transparent: rgb restored by the readback
        // unpremultiply, alpha ~128.
        assert!(p[0] > 200, "red kept, got {p:?}");
        assert!((p[3] as i32 - 128).abs() <= 1, "alpha ~128, got {}", p[3]);
    }

    #[test]
    fn push_opacity_clamps_like_cpu() {
        let fill = |opacity: f32| {
            let mut list = DisplayList::new();
            list.push(DisplayItem::PushOpacity { opacity });
            list.push(DisplayItem::FillRect {
                rect: rect(0.0, 0.0, 10.0, 10.0),
                color: Color::rgba8(255, 255, 255, 255),
            });
            list.push(DisplayItem::PopOpacity);
            render_or_fail(&list, 800, 600)
        };
        // 2.0 clamps to 1.0: fully opaque.
        assert_eq!(fill(2.0)[3], 255);
        // -0.5 clamps to 0.0: nothing painted.
        assert_eq!(fill(-0.5)[3], 0);
    }

    /// NaN opacity normalizes to 0.0 at group entry (like the CPU
    /// backend): the group paints nothing and no NaN alpha ever reaches
    /// the blend equation.
    #[test]
    fn push_opacity_nan_paints_nothing() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushOpacity { opacity: f32::NAN });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 10.0, 10.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(2.0, 2.0, 6.0, 6.0),
            color: Color::rgba8(0, 255, 0, 255),
        });
        list.push(DisplayItem::PopOpacity);
        let px = render_or_fail(&list, 20, 10);
        assert!(
            px.iter().all(|&b| b == 0),
            "NaN opacity group must paint nothing"
        );
    }

    /// Opacity group textures are pooled, not reallocated per frame: the
    /// first frame with an isolated group checks a target out and returns
    /// it to the pool after the frame's submit; the second identical frame
    /// reuses it (pool size stays 1) and renders identical pixels.
    #[test]
    fn group_target_pool_reuses_textures_across_frames() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 40.0, 40.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        // Two paint items: the isolated-group path (offscreen texture),
        // not the single-item fold fast path.
        list.push(DisplayItem::FillRect {
            rect: rect(20.0, 20.0, 40.0, 40.0),
            color: Color::rgba8(0, 255, 0, 255),
        });
        list.push(DisplayItem::PopOpacity);

        let (_adapter, device, queue) = WgpuBackend::request_device(None).unwrap_or_else(|| {
            panic!(
                "no Vulkan adapter available for wgpu backend test \
                 (set VK_ICD_FILENAMES to a lavapipe icd.json)"
            )
        });
        let mut backend = WgpuBackend::new(&device, &queue);

        let px1 = backend
            .render_to_rgba(&list, 64, 64)
            .expect("readback failed");
        assert!(
            backend.group_targets_in_use.is_empty(),
            "checked-out group targets must return to the pool after the frame submits"
        );
        assert_eq!(
            backend.group_target_pool.len(),
            1,
            "one isolated group should leave exactly one pooled target"
        );

        let px2 = backend
            .render_to_rgba(&list, 64, 64)
            .expect("readback failed");
        assert_eq!(
            backend.group_target_pool.len(),
            1,
            "second frame must reuse the pooled target, not allocate"
        );
        assert!(backend.group_targets_in_use.is_empty());
        assert_eq!(
            px1, px2,
            "pooled group texture must render identical pixels"
        );
        assert!(
            nonzero_alpha_count(&px1) > 0,
            "the opacity group should paint something"
        );
    }

    /// Backdrop-filter capture textures are pooled, not reallocated per
    /// frame: the first frame with a backdrop-filter element checks a
    /// capture out and returns it to the pool after the frame's submit;
    /// the second identical frame reuses it (pool size stays 1) and
    /// renders identical pixels.
    #[test]
    fn backdrop_capture_pool_reuses_textures_across_frames() {
        let mut list = DisplayList::new();
        // Backdrop content under the filter region: something to capture.
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 64.0, 64.0),
            color: Color::rgba8(0, 0, 255, 255),
        });
        list.push(DisplayItem::PushBackdropFilter {
            ops: vec![FilterOp::Grayscale(1.0)].into(),
            rect: rect(8.0, 8.0, 32.0, 32.0),
        });
        // The element's own content paints over the filtered backdrop.
        list.push(DisplayItem::FillRect {
            rect: rect(8.0, 8.0, 32.0, 32.0),
            color: Color::rgba8(255, 255, 255, 128),
        });
        list.push(DisplayItem::PopBackdropFilter);

        let (_adapter, device, queue) = WgpuBackend::request_device(None).unwrap_or_else(|| {
            panic!(
                "no Vulkan adapter available for wgpu backend test \
                 (set VK_ICD_FILENAMES to a lavapipe icd.json)"
            )
        });
        let mut backend = WgpuBackend::new(&device, &queue);

        let px1 = backend
            .render_to_rgba(&list, 64, 64)
            .expect("readback failed");
        assert!(
            backend.captures_in_use.is_empty(),
            "checked-out captures must return to the pool after the frame submits"
        );
        assert_eq!(
            backend.capture_pool.len(),
            1,
            "one backdrop-filter element should leave exactly one pooled capture"
        );
        // The capture hugs the filter rect: (8,8)-(40,40) at scale 1.
        let cap = &backend.capture_pool[0];
        assert_eq!(
            (cap.w, cap.h),
            (32, 32),
            "capture should hug the filter rect, got {}x{}",
            cap.w,
            cap.h
        );

        let px2 = backend
            .render_to_rgba(&list, 64, 64)
            .expect("readback failed");
        assert_eq!(
            backend.capture_pool.len(),
            1,
            "second frame must reuse the pooled capture, not allocate"
        );
        assert!(backend.captures_in_use.is_empty());
        assert_eq!(px1, px2, "pooled capture must render identical pixels");
        assert!(
            nonzero_alpha_count(&px1) > 0,
            "the backdrop-filter element should paint something"
        );
    }

    #[test]
    fn opacity_group_composites_isolated() {
        // Overlapping translucent red and blue rects inside an opacity
        // 0.5 group must composite like an isolated group (CPU backend
        // reference: overlap ~[85, 0, 170, 96]), not like folded per-item
        // alpha (which would give ~[109, 0, 146, 112]).
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 10.0, 10.0),
            color: Color::rgba8(255, 0, 0, 128),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(5.0, 0.0, 10.0, 10.0),
            color: Color::rgba8(0, 0, 255, 128),
        });
        list.push(DisplayItem::PopOpacity);
        let px = render_or_fail(&list, 20, 10);
        let at = |x: u32, y: u32| &px[(y as usize * 20 + x as usize) * 4..][..4];
        // Overlap column: red under blue, composited at group opacity.
        let p = at(7, 5);
        for (got, want) in p.iter().zip([85u8, 0, 170, 96]) {
            assert!(
                (*got as i32 - want as i32).abs() <= 2,
                "overlap pixel {p:?}, want ~[85, 0, 170, 96]"
            );
        }
        // Red-only column: the readback unpremultiplies, restoring full
        // red with the halved alpha.
        let r = at(2, 5);
        assert!((r[0] as i32 - 255).abs() <= 2, "red-only {r:?}");
        assert!((r[3] as i32 - 64).abs() <= 2, "red-only alpha {r:?}");
    }

    #[test]
    fn opacity_group_nested() {
        // A group inside a group: inner opacity applies when the inner
        // group composites, outer when the outer group does.
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 10.0, 10.0),
            color: Color::rgba8(0, 255, 0, 255),
        });
        list.push(DisplayItem::PopOpacity);
        list.push(DisplayItem::PopOpacity);
        let px = render_or_fail(&list, 20, 10);
        let p = &px[(5 * 20 + 5) * 4..][..4];
        // 0.5 * 0.5 = 0.25: full green restored by the readback
        // unpremultiply, alpha ~64.
        assert!((p[1] as i32 - 255).abs() <= 2, "nested green {p:?}");
        assert!((p[3] as i32 - 64).abs() <= 2, "nested alpha {p:?}");
    }

    #[test]
    fn opacity_group_inside_clip() {
        // The group inherits the active clip: content outside the clip
        // must not leak through the offscreen composite.
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(0.0, 0.0, 5.0, 10.0),
        });
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 10.0, 10.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::PopOpacity);
        list.push(DisplayItem::PopClip);
        let px = render_or_fail(&list, 20, 10);
        let at = |x: u32, y: u32| &px[(y as usize * 20 + x as usize) * 4..][..4];
        let inside = at(2, 5);
        // Full red restored by the readback unpremultiply, alpha ~128.
        assert!(
            (inside[0] as i32 - 255).abs() <= 2,
            "clipped group {inside:?}"
        );
        assert!(
            (inside[3] as i32 - 128).abs() <= 2,
            "clipped alpha {inside:?}"
        );
        // Outside the clip: the composite must not paint either.
        assert_eq!(at(7, 5), &[0, 0, 0, 0]);
    }

    /// Fractional clip on the GPU: the fragment shader enforces the
    /// exact clip edge per pixel — the outward-snapped integer scissor
    /// is only a coarse cull, so no half-outside pixel leaks.
    #[test]
    fn fractional_clip_solid_fill_boundary_gpu() {
        let mut list = DisplayList::new();
        // x=10.5, width=100.25.
        list.push(DisplayItem::PushClip {
            rect: rect(10.5, 0.0, 100.25, 100.0),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 200.0, 100.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::PopClip);
        let px = render_or_fail(&list, 200, 100);
        let at = |x: u32, y: u32| &px[(y as usize * 200 + x as usize) * 4..][..4];
        // No leak past either edge.
        assert_eq!(at(9, 50), &[0, 0, 0, 0]);
        assert_eq!(at(111, 50), &[0, 0, 0, 0]);
        // Boundary coverage bytes match the shared quantized formula
        // (half -> 128, 0.75 -> 191).
        let b10 = at(10, 50);
        assert!((b10[3] as i32 - 128).abs() <= 1, "left boundary {b10:?}");
        assert!((b10[0] as i32 - 255).abs() <= 1, "left boundary {b10:?}");
        let b110 = at(110, 50);
        assert!((b110[3] as i32 - 191).abs() <= 1, "right boundary {b110:?}");
        assert_eq!(at(11, 50), &[255, 0, 0, 255]);
    }

    /// CPU and GPU agree on fractional clip boundaries pixel-for-pixel:
    /// the same pixels are touched (no leak on either backend), and the
    /// boundary coverage bytes match. Uses the task's x=10.5 /
    /// width=100.25 example, plus a nested fractional clip.
    #[test]
    fn fractional_clip_cpu_gpu_agree() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(10.5, 10.5, 100.25, 80.5),
        });
        list.push(DisplayItem::PushClip {
            rect: rect(20.25, 20.75, 50.5, 40.25),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 200.0, 100.0),
            color: Color::rgba8(0, 255, 0, 255),
        });
        list.push(DisplayItem::PopClip);
        list.push(DisplayItem::PopClip);

        let gpu = render_or_fail(&list, 200, 100);
        let cpu = crate::cpu::CpuBackend::render_to_rgba(&list, 200, 100);
        assert_eq!(gpu.len(), cpu.len());
        // The clip decision is shared exactly: a pixel the CPU leaves
        // transparent must be transparent on the GPU too, and vice
        // versa. Anything else is a boundary leak on one backend.
        for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            if i % 4 == 3 {
                assert_eq!(
                    (g == 0),
                    (c == 0),
                    "alpha touched-set disagrees at pixel {}: gpu={g} cpu={c}",
                    i / 4
                );
            }
        }
        // Blended values: the backends use different blend pipelines
        // (tiny-skia integer vs wgpu float), so allow the established
        // cross-backend tolerance; the clip boundary itself is exact
        // per the touched-set check above.
        for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert!(
                (g as i16 - c as i16).abs() <= 2,
                "pixel byte {i} disagrees: gpu={g} cpu={c}"
            );
        }
    }

    /// Rounded clips cut corners on the GPU exactly like the CPU: the
    /// corner pixels are empty, the center and edge midpoints are
    /// solid, and arc pixels are partial.
    #[test]
    fn rounded_clip_cuts_corners_gpu() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushRoundedClip {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            radii: [20.0, 20.0, 20.0, 20.0],
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::PopClip);
        let px = render_or_fail(&list, 100, 100);
        let at = |x: u32, y: u32| &px[(y as usize * 100 + x as usize) * 4..][..4];
        assert_eq!(at(0, 0), &[0, 0, 0, 0]);
        assert_eq!(at(99, 0), &[0, 0, 0, 0]);
        assert_eq!(at(0, 99), &[0, 0, 0, 0]);
        assert_eq!(at(99, 99), &[0, 0, 0, 0]);
        assert_eq!(at(50, 50), &[255, 0, 0, 255]);
        assert_eq!(at(50, 0), &[255, 0, 0, 255]);
        assert_eq!(at(0, 50), &[255, 0, 0, 255]);
        let arc = at(5, 5);
        assert!(
            arc[3] > 0 && arc[3] < 255,
            "arc pixel must be partial, got {arc:?}"
        );
    }

    /// A circular clip (`border-radius: 50%` on a square) on the GPU:
    /// the center is solid, the corners are empty, and edge midpoints
    /// sit on the disc.
    #[test]
    fn rounded_clip_circle_gpu() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushRoundedClip {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            radii: [50.0, 50.0, 50.0, 50.0],
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::PopClip);
        let px = render_or_fail(&list, 100, 100);
        let at = |x: u32, y: u32| &px[(y as usize * 100 + x as usize) * 4..][..4];
        assert_eq!(at(50, 50), &[255, 0, 0, 255]);
        assert_eq!(at(0, 0), &[0, 0, 0, 0]);
        assert_eq!(at(99, 99), &[0, 0, 0, 0]);
        assert_eq!(at(50, 0), &[255, 0, 0, 255]);
        assert_eq!(at(99, 50), &[255, 0, 0, 255]);
    }

    /// Nested rounded + rect clips on the GPU: the inner rounded clip
    /// cuts corners inside the outer rect clip, and the outer rect
    /// still bounds the rounded one.
    #[test]
    fn rounded_clip_nested_rect_gpu() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(0.0, 0.0, 60.0, 100.0),
        });
        list.push(DisplayItem::PushRoundedClip {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            radii: [20.0, 20.0, 20.0, 20.0],
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::PopClip);
        list.push(DisplayItem::PopClip);
        let px = render_or_fail(&list, 100, 100);
        let at = |x: u32, y: u32| &px[(y as usize * 100 + x as usize) * 4..][..4];
        assert_eq!(at(50, 50), &[255, 0, 0, 255]);
        assert_eq!(at(0, 0), &[0, 0, 0, 0]);
        assert_eq!(at(80, 50), &[0, 0, 0, 0]);
        assert_eq!(at(99, 99), &[0, 0, 0, 0]);
    }

    /// CPU and GPU agree on rounded clips pixel-for-pixel: the same
    /// pixels are touched (the 8x8 subpixel sampling is bit-identical
    /// in Rust f32 and WGSL), and blended values stay within the
    /// established cross-backend tolerance. Exercises fractional radii
    /// and rounded/rect nesting.
    #[test]
    fn rounded_clip_cpu_gpu_agree() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushRoundedClip {
            rect: rect(10.5, 10.5, 100.25, 80.5),
            radii: [15.5, 0.0, 10.25, 0.0],
        });
        list.push(DisplayItem::PushClip {
            rect: rect(0.0, 0.0, 60.0, 100.0),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 200.0, 100.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::PopClip);
        list.push(DisplayItem::PopClip);

        let gpu = render_or_fail(&list, 200, 100);
        let cpu = crate::cpu::CpuBackend::render_to_rgba(&list, 200, 100);
        assert_eq!(gpu.len(), cpu.len());
        // Touched set: exact — the clip boundary itself is bit-exact,
        // only the blend pipelines differ.
        for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            if i % 4 == 3 {
                assert_eq!(
                    (g > 0),
                    (c > 0),
                    "alpha touched-set disagrees at pixel {}: gpu={g} cpu={c}",
                    i / 4
                );
            }
        }
        // Blended values: the backends use different blend pipelines
        // (tiny-skia integer vs wgpu float), so allow the established
        // cross-backend tolerance.
        for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert!(
                (g as i16 - c as i16).abs() <= 2,
                "pixel byte {i} disagrees: gpu={g} cpu={c}"
            );
        }
    }

    /// A rounded clip around an isolated opacity group on the GPU: the
    /// group inherits the rounded entries while rendering, and the
    /// composite does not clip twice (which would square the corner
    /// coverage).
    #[test]
    fn rounded_clip_opacity_group_gpu() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushRoundedClip {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            radii: [20.0, 20.0, 20.0, 20.0],
        });
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::PopOpacity);
        list.push(DisplayItem::PopClip);
        let px = render_or_fail(&list, 100, 100);
        let at = |x: u32, y: u32| &px[(y as usize * 100 + x as usize) * 4..][..4];
        // Rounded corner still cut through the group composite.
        assert_eq!(at(0, 0), &[0, 0, 0, 0]);
        // Center: half opacity, full red restored by the readback
        // unpremultiply.
        let c = at(50, 50);
        assert!((c[0] as i32 - 255).abs() <= 2, "center red {c:?}");
        assert!((c[3] as i32 - 128).abs() <= 2, "center alpha {c:?}");
    }

    /// A fractional clip inside an isolated opacity group on the GPU:
    /// the group inherits the clip while rendering, and the composite
    /// does not clip twice (which would square the boundary coverage).
    #[test]
    fn fractional_clip_opacity_group_gpu() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(10.5, 0.0, 100.25, 100.0),
        });
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 200.0, 100.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::PopOpacity);
        list.push(DisplayItem::PopClip);
        let px = render_or_fail(&list, 200, 100);
        let at = |x: u32, y: u32| &px[(y as usize * 200 + x as usize) * 4..][..4];
        assert_eq!(at(9, 50), &[0, 0, 0, 0]);
        assert_eq!(at(111, 50), &[0, 0, 0, 0]);
        // Boundary: clip coverage 128, group opacity 0.5 -> alpha ~64.
        // Double-clipping would give ~32.
        let b10 = at(10, 50);
        assert!(
            (b10[3] as i32 - 64).abs() <= 2,
            "group boundary double-clipped? {b10:?}"
        );
        let inside = at(50, 50);
        assert!(
            (inside[3] as i32 - 128).abs() <= 2,
            "group interior {inside:?}"
        );
    }

    #[test]
    fn text_run_honors_baseline() {
        // Two runs with the same rect but different baselines must paint
        // at different heights: the baseline, not rect.y, positions text.
        let run_at = |baseline: f32| {
            let mut list = DisplayList::new();
            list.push(DisplayItem::TextRun {
                text: Arc::from("A"),
                text_range: 0..1,
                rect: rect(20.0, 20.0, 100.0, 20.0),
                baseline,
                font_key: FontKey(0),
                size_px: 16.0,
                color: Color::rgba8(0, 255, 0, 255),
            });
            render_or_fail(&list, 800, 600)
        };
        let top_row = |px: &[u8]| {
            (0..600u32)
                .find(|&y| (20..36u32).any(|x| px[(y as usize * 800 + x as usize) * 4 + 3] > 0))
                .unwrap_or(600)
        };
        let hi = top_row(&run_at(200.0));
        let lo = top_row(&run_at(400.0));
        assert!(
            lo > hi + 100,
            "baseline must move the text: tops {hi} vs {lo}"
        );
    }

    /// Regression (GPU path): same key-aliasing scenario as the CPU
    /// backend's `font_data_cache_does_not_alias_keys_across_caches`.
    /// Two `FontCache`s with swapped resolution orders collide on one
    /// `FontKey`; the second `render_offscreen` on the thread must use
    /// its own cache's bytes, not the first render's.
    #[test]
    fn font_data_cache_does_not_alias_keys_across_caches() {
        use crate::fonts::FontStyle;

        let cache_a = FontCache::new();
        let key_a = cache_a
            .resolve("sans-serif", 400, FontStyle::Normal)
            .expect("sans-serif resolves");
        let cache_b = FontCache::new();
        let key_b = cache_b
            .resolve("monospace", 400, FontStyle::Normal)
            .expect("monospace resolves");
        // Swapped resolution orders collide on one FontKey: the premise
        // of the bug. Deterministic: each cache mints FontKey(0) for its
        // eager default face, then FontKey(1) for the first resolve.
        assert_eq!(
            key_a, key_b,
            "swapped resolution orders must collide on one FontKey"
        );

        let data_a = cache_a.font_data(key_a).expect("key_a resolves");
        let data_b = cache_b.font_data(key_b).expect("key_b resolves");
        if data_a.bytes() == data_b.bytes() {
            // Same face both times here: nothing to distinguish.
            return;
        }

        let mk_list = |key: FontKey| {
            let mut list = DisplayList::new();
            list.push(DisplayItem::FillRect {
                rect: rect(0.0, 0.0, 200.0, 60.0),
                color: Color::rgba8(255, 255, 255, 255),
            });
            list.push(DisplayItem::TextRun {
                text: Arc::from("Hello"),
                text_range: 0..5,
                rect: rect(10.0, 10.0, 180.0, 40.0),
                baseline: 40.0,
                font_key: key,
                size_px: 32.0,
                color: Color::rgba8(0, 0, 0, 255),
            });
            list
        };
        // Same-thread sequential renders, like `render_or_fail` but with
        // an explicit cache per render: the thread-local must not serve
        // the first render's bytes to the second.
        let render = |list: &DisplayList, fonts: &FontCache| {
            WgpuBackend::render_offscreen(list, 200, 60, fonts).unwrap_or_else(|| {
                panic!(
                    "no Vulkan adapter available for wgpu backend test \
                     (set VK_ICD_FILENAMES to a lavapipe icd.json)"
                )
            })
        };
        let px_a = render(&mk_list(key_a), &cache_a);
        let px_b = render(&mk_list(key_b), &cache_b);
        assert_ne!(
            px_a, px_b,
            "same FontKey from another cache rendered with stale bytes"
        );
    }

    /// Two consecutive `render_offscreen` calls against the same
    /// `FontCache` share one resolver hook: the per-call `set_font_data`
    /// reinstall is a no-op, so the glyph atlas is not dropped and
    /// re-rasterized every frame.
    #[test]
    fn render_offscreen_reuses_resolver_across_frames() {
        let fonts = FontCache::new();
        let key = fonts.default_key().expect("default face resolves");
        let mut list = DisplayList::new();
        list.push(DisplayItem::TextRun {
            text: Arc::from("Hello"),
            text_range: 0..5,
            rect: rect(10.0, 10.0, 180.0, 40.0),
            baseline: 40.0,
            font_key: key,
            size_px: 32.0,
            color: Color::rgba8(0, 0, 0, 255),
        });
        let render = || {
            WgpuBackend::render_offscreen(&list, 200, 60, &fonts).unwrap_or_else(|| {
                panic!(
                    "no Vulkan adapter available for wgpu backend test \
                     (set VK_ICD_FILENAMES to a lavapipe icd.json)"
                )
            })
        };
        let px_a = render();
        let hook_a = headless_installed_hook().expect("hook installed");
        let px_b = render();
        let hook_b = headless_installed_hook().expect("hook installed");
        assert_eq!(px_a, px_b, "frames must paint identically");
        assert!(
            Arc::ptr_eq(&hook_a, &hook_b),
            "consecutive render_offscreen calls must share the resolver hook"
        );
        // A genuinely different cache still installs a different hook, so
        // hook identity keeps guarding the thread-local cache against key
        // aliasing.
        let other = FontCache::new();
        let _ =
            WgpuBackend::render_offscreen(&list, 200, 60, &other).expect("render with other cache");
        let hook_c = headless_installed_hook().expect("hook installed");
        assert!(
            !Arc::ptr_eq(&hook_a, &hook_c),
            "a different FontCache must install a different hook"
        );
    }

    // -- RepeatImage --------------------------------------------------------

    /// Pixel accessor for RGBA8 buffers (mirrors the CPU backend's test
    /// helper).
    fn rgba_at(rgba: &[u8], stride: u32, x: u32, y: u32) -> [u8; 4] {
        let o = (y as usize * stride as usize + x as usize) * 4;
        [rgba[o], rgba[o + 1], rgba[o + 2], rgba[o + 3]]
    }

    /// 4x4 test tile with distinct pixels per quadrant: TL red, TR
    /// green, BL blue, BR white.
    fn repeat_checker_tile() -> DecodedImage {
        let mut px = vec![0u8; 4 * 4 * 4];
        for y in 0..4 {
            for x in 0..4 {
                let c = match (x < 2, y < 2) {
                    (true, true) => [255, 0, 0, 255],
                    (false, true) => [0, 255, 0, 255],
                    (true, false) => [0, 0, 255, 255],
                    (false, false) => [255, 255, 255, 255],
                };
                let o = (y * 4 + x) * 4;
                px[o..o + 4].copy_from_slice(&c);
            }
        }
        DecodedImage::from_rgba(4, 4, px)
    }

    /// One `RepeatImage` item over a 12x12 area with a 4x4 tile tiles 3x3
    /// on the GPU, with the positioned origin respected.
    #[test]
    fn gpu_repeat_image_tiles() {
        let mut list = DisplayList::new();
        let id = ImageId(1);
        list.add_image(id, repeat_checker_tile());
        list.push(DisplayItem::RepeatImage {
            area: Rect {
                x: 0.0,
                y: 0.0,
                w: 12.0,
                h: 12.0,
            },
            image: id,
            tile: (4.0, 4.0),
            repeat: BackgroundRepeat::Repeat,
            offset: (0.0, 0.0),
        });
        let rgba = render_or_fail(&list, 12, 12);
        // Tile grid 3x3: each tile's TL quadrant is red, BR white.
        assert_eq!(rgba_at(&rgba, 12, 0, 0), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 12, 4, 0), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 12, 8, 8), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 12, 2, 2), [255, 255, 255, 255]);
        assert_eq!(rgba_at(&rgba, 12, 6, 6), [255, 255, 255, 255]);
    }

    /// `repeat-x` on the GPU: one row, nothing above or below it.
    #[test]
    fn gpu_repeat_x_paints_single_row() {
        let mut list = DisplayList::new();
        let id = ImageId(1);
        list.add_image(id, repeat_checker_tile());
        list.push(DisplayItem::RepeatImage {
            area: Rect {
                x: 0.0,
                y: 0.0,
                w: 12.0,
                h: 12.0,
            },
            image: id,
            tile: (4.0, 4.0),
            repeat: BackgroundRepeat::RepeatX,
            offset: (0.0, 4.0),
        });
        let rgba = render_or_fail(&list, 12, 12);
        assert_eq!(rgba_at(&rgba, 12, 0, 4), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 12, 8, 4), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 12, 6, 6), [255, 255, 255, 255]);
        assert_eq!(rgba_at(&rgba, 12, 6, 0), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 12, 6, 11), [0, 0, 0, 0]);
    }

    /// `repeat-y` on the GPU: one column, nothing left or right of it.
    #[test]
    fn gpu_repeat_y_paints_single_column() {
        let mut list = DisplayList::new();
        let id = ImageId(1);
        list.add_image(id, repeat_checker_tile());
        list.push(DisplayItem::RepeatImage {
            area: Rect {
                x: 0.0,
                y: 0.0,
                w: 12.0,
                h: 12.0,
            },
            image: id,
            tile: (4.0, 4.0),
            repeat: BackgroundRepeat::RepeatY,
            offset: (4.0, 0.0),
        });
        let rgba = render_or_fail(&list, 12, 12);
        assert_eq!(rgba_at(&rgba, 12, 4, 0), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 12, 4, 8), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 12, 6, 6), [255, 255, 255, 255]);
        assert_eq!(rgba_at(&rgba, 12, 0, 6), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 12, 11, 6), [0, 0, 0, 0]);
    }

    /// `no-repeat` on the GPU: exactly one positioned tile.
    #[test]
    fn gpu_no_repeat_paints_single_tile() {
        let mut list = DisplayList::new();
        let id = ImageId(1);
        list.add_image(id, repeat_checker_tile());
        list.push(DisplayItem::RepeatImage {
            area: Rect {
                x: 0.0,
                y: 0.0,
                w: 12.0,
                h: 12.0,
            },
            image: id,
            tile: (4.0, 4.0),
            repeat: BackgroundRepeat::NoRepeat,
            offset: (5.0, 3.0),
        });
        let rgba = render_or_fail(&list, 12, 12);
        assert_eq!(rgba_at(&rgba, 12, 5, 3), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 12, 7, 5), [255, 255, 255, 255]);
        assert_eq!(rgba_at(&rgba, 12, 4, 3), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 12, 9, 3), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 12, 5, 7), [0, 0, 0, 0]);
    }

    /// `space` on the GPU: 3 whole 4px tiles across a 14px area with
    /// 1px gaps between them.
    #[test]
    fn gpu_space_leaves_gaps() {
        let mut list = DisplayList::new();
        let id = ImageId(1);
        list.add_image(id, repeat_checker_tile());
        list.push(DisplayItem::RepeatImage {
            area: Rect {
                x: 0.0,
                y: 0.0,
                w: 14.0,
                h: 14.0,
            },
            image: id,
            tile: (4.0, 4.0),
            repeat: BackgroundRepeat::Space,
            offset: (0.0, 0.0),
        });
        let rgba = render_or_fail(&list, 14, 14);
        // Tiles at x/y = 0, 5, 10; gaps at 4 and 9.
        assert_eq!(rgba_at(&rgba, 14, 0, 0), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 14, 5, 0), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 14, 10, 10), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 14, 2, 2), [255, 255, 255, 255]);
        assert_eq!(rgba_at(&rgba, 14, 4, 2), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 14, 9, 9), [0, 0, 0, 0]);
    }

    /// f32 ulp at `x`'s magnitude (`x` must be positive and finite).
    fn ulp_at(x: f32) -> f32 {
        2f32.powi(x.log2().floor() as i32 - 23)
    }

    /// Byte bar for repeat-image CPU/GPU differential tests: the
    /// single-quad affine UV ramp can sit ~1 ulp of the tile-unit
    /// magnitude off the CPU backend's exact per-tile coordinates (see
    /// `repeat_axis`), so the sampled texel can shift by ~4 ulps of the
    /// magnitude; scaled by the test texture's texel density and the
    /// worst-case byte step between adjacent texels, rounded up, at
    /// least 1 byte. The below-side snap band can canonicalize a
    /// fragment genuinely up to the full band (4 ulps of the magnitude)
    /// below a boundary, hence the doubled headroom. Sampled colors
    /// agree to <1 LSB at modest tile counts; adversarial ≥4096-tile
    /// spans can straddle a texel boundary and are out of scope for
    /// this bar.
    fn repeat_byte_tolerance(tile_mag: f32, texels_per_tile: f32) -> u8 {
        (4.0 * ulp_at(tile_mag) * texels_per_tile * 255.0)
            .ceil()
            .max(1.0) as u8
    }

    /// CPU and GPU agree on the new repeat primitive across repeat
    /// modes (same bar as the existing `cpu_gpu_pixel_agreement`).
    #[test]
    fn gpu_repeat_matches_cpu() {
        // Tile-unit magnitude of the repeat axes; the differential bar is
        // ~4 ulps of it (see `repeat_byte_tolerance`).
        let tile_mag = (100.0f32 / 30.0).max(60.0 / 30.0);
        let tol = repeat_byte_tolerance(tile_mag, 4.0);
        for repeat in [
            BackgroundRepeat::Repeat,
            BackgroundRepeat::RepeatX,
            BackgroundRepeat::RepeatY,
            BackgroundRepeat::NoRepeat,
            BackgroundRepeat::Space,
        ] {
            let mut list = DisplayList::new();
            let id = ImageId(1);
            list.add_image(id, repeat_checker_tile());
            list.push(DisplayItem::RepeatImage {
                area: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 100.0,
                    h: 60.0,
                },
                image: id,
                tile: (30.0, 30.0),
                repeat,
                offset: (10.0, 7.0),
            });
            let cpu = crate::cpu::CpuBackend::render_to_rgba(&list, 100, 60);
            let gpu = render_or_fail(&list, 100, 60);
            assert_eq!(cpu.len(), gpu.len());
            let max_diff = cpu
                .iter()
                .zip(gpu.iter())
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            assert!(
                max_diff <= tol,
                "CPU/GPU repeat disagree by {max_diff} (bar {tol}) for {repeat:?}"
            );
        }
    }

    /// The CPU backend's tile-boundary convention, pinned: a pixel
    /// whose center sits exactly on a tile boundary is painted by the
    /// tile on its left (rect rasterization keeps the boundary pixel in
    /// the left tile, whose edge taps clamp to its last texel). With a
    /// half-pixel offset the tile edges at 10.5, 40.5, ... fall on
    /// pixel centers, so column x=10 samples the previous tile's right
    /// edge: the 4x4 checker's last column/row, i.e. white. The GPU
    /// wrap branch (`repeat_axis`) canonicalizes to this convention via
    /// the snap band; without it the ±1 ulp interpolation residue would
    /// flip these pixels to the next tile's first texel.
    #[test]
    fn cpu_repeat_boundary_pixel_paints_left_tile() {
        let mut list = DisplayList::new();
        let id = ImageId(1);
        list.add_image(id, repeat_checker_tile());
        list.push(DisplayItem::RepeatImage {
            area: Rect {
                x: 0.0,
                y: 0.0,
                w: 100.0,
                h: 60.0,
            },
            image: id,
            tile: (30.0, 30.0),
            repeat: BackgroundRepeat::Repeat,
            offset: (10.5, 7.5),
        });
        let cpu = crate::cpu::CpuBackend::render_to_rgba(&list, 100, 60);
        // Pixel centers on the x tile boundaries (10.5, 40.5, 70.5)
        // sample the left tile's last texel column (white).
        assert_eq!(rgba_at(&cpu, 100, 10, 0), [255, 255, 255, 255]);
        assert_eq!(rgba_at(&cpu, 100, 40, 0), [255, 255, 255, 255]);
        // A pixel center on a y tile boundary (7.5) samples the tile
        // above's last texel row.
        assert_eq!(rgba_at(&cpu, 100, 0, 7)[3], 255);
    }

    /// CPU/GPU agreement when pixel centers fall exactly on tile
    /// boundaries: the half-pixel offset puts tile edges at 10.5, 40.5,
    /// ... / 7.5, 37.5, so every pixel in columns 10/40/70 and rows
    /// 7/37 is a boundary pixel. The GPU wrap branch snaps its
    /// interpolation residue to the CPU's left-tile convention (see
    /// `repeat_axis`), so these agree within the usual bar instead of
    /// flipping to the next tile's first texel.
    #[test]
    fn gpu_repeat_tile_boundary_matches_cpu() {
        let tile_mag = (100.0f32 / 30.0).max(60.0 / 30.0);
        let tol = repeat_byte_tolerance(tile_mag, 4.0);
        let mut list = DisplayList::new();
        let id = ImageId(1);
        list.add_image(id, repeat_checker_tile());
        list.push(DisplayItem::RepeatImage {
            area: Rect {
                x: 0.0,
                y: 0.0,
                w: 100.0,
                h: 60.0,
            },
            image: id,
            tile: (30.0, 30.0),
            repeat: BackgroundRepeat::Repeat,
            offset: (10.5, 7.5),
        });
        let cpu = crate::cpu::CpuBackend::render_to_rgba(&list, 100, 60);
        let gpu = render_or_fail(&list, 100, 60);
        assert_eq!(cpu.len(), gpu.len());
        let max_diff = cpu
            .iter()
            .zip(gpu.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(
            max_diff <= tol,
            "CPU/GPU repeat disagree by {max_diff} (bar {tol}) on tile boundaries"
        );
    }

    /// CPU/GPU agreement on tile boundaries with a large
    /// `background-position` offset: the offset shifts the tile-unit
    /// vertex UVs (`u0 = -33.35` here) far from the span, so the f32
    /// interpolation residue (~1 ulp of that magnitude) exceeds the old
    /// span-based snap band. [`repeat_snap_band`] sizes the band from
    /// the magnitude, keeping boundary fragments canonicalized to the
    /// CPU's left-tile convention; with the old band these pixels
    /// flipped to the next tile's first texel on the GPU.
    #[test]
    fn gpu_repeat_tile_boundary_large_offset_matches_cpu() {
        // u0 = (0 - 1000.5)/30, u1 = (100 - 1000.5)/30.
        let tile_mag = (1000.5f32 / 30.0).max(900.5 / 30.0);
        let tol = repeat_byte_tolerance(tile_mag, 4.0);
        let mut list = DisplayList::new();
        let id = ImageId(1);
        list.add_image(id, repeat_checker_tile());
        list.push(DisplayItem::RepeatImage {
            area: Rect {
                x: 0.0,
                y: 0.0,
                w: 100.0,
                h: 60.0,
            },
            image: id,
            tile: (30.0, 30.0),
            repeat: BackgroundRepeat::Repeat,
            // 1000.5 mod 30 == 10.5: same tile phase as
            // `gpu_repeat_tile_boundary_matches_cpu`, so the CPU pixels
            // are identical and only the GPU's UV magnitude changes.
            offset: (1000.5, 7.5),
        });
        let cpu = crate::cpu::CpuBackend::render_to_rgba(&list, 100, 60);
        let gpu = render_or_fail(&list, 100, 60);
        assert_eq!(cpu.len(), gpu.len());
        let max_diff = cpu
            .iter()
            .zip(gpu.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(
            max_diff <= tol,
            "CPU/GPU repeat disagree by {max_diff} (bar {tol}) on tile boundaries with large offset"
        );
    }

    /// A hand-built `RepeatImage` with a hostile tiny tile (bypassing
    /// the builder's [`cap_repeat_tile`]) renders the same enlarged
    /// tiles on the GPU as on the CPU: `lower_repeat_image` applies the
    /// cap just like `CpuBackend::paint_repeat_image`. Without the cap
    /// the GPU would paint un-enlarged sub-pixel tiles — a full
    /// divergence, not a 1-ulp wobble.
    #[test]
    fn gpu_repeat_hostile_tiny_tile_matches_cpu() {
        let mut list = DisplayList::new();
        let id = ImageId(1);
        list.add_image(id, repeat_checker_tile());
        // 1e-10 px tile over a 64x64 area: the cap enlarges it to
        // exactly 0.25 px (256 tiles per axis, the 65536-tile budget).
        // The 0.1 px offset puts pixel centers at fract 0.6 of a tile —
        // mid-texel, far from any texel boundary, so the ±1 ulp
        // interpolation residual cannot flip a sampled texel.
        list.push(DisplayItem::RepeatImage {
            area: Rect {
                x: 0.0,
                y: 0.0,
                w: 64.0,
                h: 64.0,
            },
            image: id,
            tile: (1e-10, 1e-10),
            repeat: BackgroundRepeat::Repeat,
            offset: (0.1, 0.1),
        });
        let cpu = crate::cpu::CpuBackend::render_to_rgba(&list, 64, 64);
        let gpu = render_or_fail(&list, 64, 64);
        assert_eq!(cpu.len(), gpu.len());
        let tol = repeat_byte_tolerance(256.0, 4.0);
        let max_diff = cpu
            .iter()
            .zip(gpu.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(
            max_diff <= tol,
            "CPU/GPU hostile repeat disagree by {max_diff} (bar {tol})"
        );
        // Sanity: the capped grid is really there. At fract 0.6 of the
        // tile the linear sampler blends texels (1,1)..(2,2), giving
        // R ≈ 0.81*255 + 0.01*255 ≈ 209 on both backends; an uncapped
        // GPU would sample fract(huge) = 0 everywhere and land near
        // R ≈ 127 instead.
        let r = rgba_at(&gpu, 64, 0, 0)[0];
        assert!(
            (200..=220).contains(&r),
            "capped hostile tile should sample mid-tile, got R={r}"
        );
    }

    /// A hostile repeat on the GPU (1 px tile over a 10M-px area)
    /// completes and tiles the viewport: the area is clamped at
    /// lowering time and one draw covers it.
    #[test]
    fn gpu_repeat_huge_area_stays_bounded() {
        let mut list = DisplayList::new();
        let id = ImageId(1);
        list.add_image(id, DecodedImage::from_rgba(1, 1, vec![200, 10, 10, 255]));
        list.push(DisplayItem::RepeatImage {
            area: Rect {
                x: 0.0,
                y: 0.0,
                w: 10_000_000.0,
                h: 10_000_000.0,
            },
            image: id,
            tile: (1.0, 1.0),
            repeat: BackgroundRepeat::Repeat,
            offset: (0.0, 0.0),
        });
        assert_eq!(list.items.len(), 1, "huge repeat is one item");
        let rgba = render_or_fail(&list, 64, 64);
        assert_eq!(rgba_at(&rgba, 64, 0, 0), [200, 10, 10, 255]);
        assert_eq!(rgba_at(&rgba, 64, 63, 63), [200, 10, 10, 255]);
    }

    /// `mix-blend-mode`: six side-by-side swatches, one per mode, over
    /// a shared backdrop. The GPU blend-composite path (backdrop copy
    /// + blend shader, blending disabled) must agree with the CPU
    ///   backend byte-for-byte within 2.
    #[test]
    fn blend_modes_agree_with_cpu() {
        let modes = [
            BlendMode::Normal,
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Overlay,
            BlendMode::Darken,
            BlendMode::Lighten,
        ];
        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 120.0, 20.0),
            color: Color::rgba8(64, 128, 192, 255),
        });
        for (k, mode) in modes.iter().enumerate() {
            let x = k as f32 * 20.0;
            list.push(DisplayItem::PushBlend { mode: *mode });
            list.push(DisplayItem::FillRect {
                rect: rect(x, 0.0, 20.0, 20.0),
                color: Color::rgba8(192, 64, 128, 255),
            });
            list.push(DisplayItem::PopBlend);
        }
        let gpu = render_or_fail(&list, 120, 20);
        let cpu = crate::cpu::CpuBackend::render_to_rgba(&list, 120, 20);
        assert_eq!(gpu.len(), cpu.len());
        for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert!(
                (g as i16 - c as i16).abs() <= 2,
                "pixel byte {i} disagrees: gpu={g} cpu={c}"
            );
        }
        // Spot-check the hand-computed values at each swatch's center.
        let expected = [
            [192, 64, 128, 255],
            [48, 32, 96, 255],
            [208, 160, 224, 255],
            [96, 65, 192, 255],
            [64, 64, 128, 255],
            [192, 128, 192, 255],
        ];
        for (k, &px) in expected.iter().enumerate() {
            assert_eq!(rgba_at(&gpu, 120, k as u32 * 20 + 10, 10), px, "swatch {k}");
        }
    }

    /// `Normal` blend groups are the no-op fast path on the GPU too:
    /// lowering emits no group op and isolates no group texture.
    #[test]
    fn normal_blend_group_lowers_inline() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: Color::rgba8(64, 128, 192, 255),
        });
        list.push(DisplayItem::PushBlend {
            mode: BlendMode::Normal,
        });
        list.push(DisplayItem::FillRect {
            rect: rect(10.0, 10.0, 80.0, 80.0),
            color: Color::rgba8(192, 64, 128, 255),
        });
        list.push(DisplayItem::PopBlend);
        let mut backend = WgpuBackend::new_headless().unwrap_or_else(|| {
            panic!(
                "no Vulkan adapter available for wgpu backend test \
                 (set VK_ICD_FILENAMES to a lavapipe icd.json)"
            )
        });
        let lowered = backend.lower(&list, (1.0, 1.0));
        assert!(
            lowered.groups.is_empty(),
            "normal blend must not isolate a group"
        );
        assert!(
            lowered.ops.iter().all(|op| !matches!(op, Op::Group(_))),
            "normal blend must not emit a group op"
        );
    }

    /// `gradient_quad` batches contiguous same-texture quads into one
    /// `Op::DrawGradient` (same rule as `emit`): a different texture
    /// or an intervening op starts a new draw; degenerate rects and
    /// zero alpha push nothing. Needs no GPU device.
    #[test]
    fn gradient_quad_batches_same_texture() {
        let mut low = Lowered::with_capacity(8);
        let r = Rect {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
        };
        low.gradient_quad(r, 7, 1.0);
        low.gradient_quad(r, 7, 1.0); // merges: one op, count 2
        low.gradient_quad(r, 9, 1.0); // new texture: new op
        assert_eq!(low.ops.len(), 2);
        match low.ops[0] {
            Op::DrawGradient { tex, start, count } => {
                assert_eq!(tex, 7);
                assert_eq!(start, 0);
                assert_eq!(count, 2);
            }
            _ => panic!("expected DrawGradient op"),
        }
        match low.ops[1] {
            Op::DrawGradient { tex, start, count } => {
                assert_eq!(tex, 9);
                assert_eq!(start, 2);
                assert_eq!(count, 1);
            }
            _ => panic!("expected DrawGradient op"),
        }
        // An intervening op breaks contiguity: no merge.
        low.emit(PipeKind::Solid, None, 0);
        low.gradient_quad(r, 7, 1.0);
        assert_eq!(low.ops.len(), 4);
        match low.ops[3] {
            Op::DrawGradient { tex, start, count } => {
                assert_eq!(tex, 7);
                assert_eq!(start, 3);
                assert_eq!(count, 1);
            }
            _ => panic!("expected DrawGradient op"),
        }
        // Degenerate inputs push nothing.
        let n = low.ops.len();
        let flat = Rect {
            x: 0.0,
            y: 0.0,
            w: 0.0,
            h: 10.0,
        };
        low.gradient_quad(flat, 7, 1.0);
        low.gradient_quad(r, 7, 0.0);
        assert_eq!(low.ops.len(), n);
    }

    // ------------------------------------------------------------------
    // filter / clip-path / backdrop-filter: CPU/GPU agreement
    // ------------------------------------------------------------------

    /// Assert the GPU rendering matches the CPU within `tol` per byte.
    fn assert_cpu_gpu_agree(list: &DisplayList, w: u32, h: u32, tol: u8, what: &str) {
        let gpu = render_or_fail(list, w, h);
        let cpu = crate::cpu::CpuBackend::render_to_rgba(list, w, h);
        assert_eq!(gpu.len(), cpu.len());
        let mut worst = 0u8;
        for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let d = g.abs_diff(c);
            worst = worst.max(d);
            assert!(
                d <= tol,
                "{what}: byte {i} disagrees: gpu={g} cpu={c} (tol {tol})"
            );
        }
        let _ = worst;
    }

    fn filter_group_list(ops: Vec<FilterOp>, content: DisplayItem) -> DisplayList {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushFilter { ops: ops.into() });
        list.push(content);
        list.push(DisplayItem::PopFilter);
        list
    }

    /// `grayscale(1)` on red: exact color math, tight tolerance.
    #[test]
    fn filter_grayscale_cpu_gpu_agree() {
        let list = filter_group_list(
            vec![FilterOp::Grayscale(1.0)],
            DisplayItem::FillRect {
                rect: rect(10.0, 10.0, 80.0, 80.0),
                color: Color::rgba8(255, 0, 0, 255),
            },
        );
        assert_cpu_gpu_agree(&list, 100, 100, 2, "grayscale");
    }

    /// `grayscale(1)` over NON-UNIFORM content: the filter pass must
    /// sample the whole group texture 1:1. Catches the `vs_filter` UV
    /// bug, which mapped the viewport onto the texture's bottom-left
    /// quadrant stretched 2x — invisible on the uniform-red test above
    /// (a 2x zoom of uniform content is uniform).
    #[test]
    fn filter_grayscale_nonuniform_cpu_gpu_agree() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushFilter {
            ops: vec![FilterOp::Grayscale(1.0)].into(),
        });
        // Sharp vertical edge: left red, right blue.
        list.push(DisplayItem::FillRect {
            rect: rect(10.0, 10.0, 40.0, 80.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(50.0, 10.0, 40.0, 80.0),
            color: Color::rgba8(0, 0, 255, 255),
        });
        list.push(DisplayItem::PopFilter);
        assert_cpu_gpu_agree(&list, 100, 100, 2, "grayscale non-uniform");
    }

    /// `backdrop-filter: grayscale(1)` over NON-UNIFORM backdrop: the
    /// capture + filter pass must reproduce the backdrop 1:1. Same
    /// `vs_filter` UV coverage as the test above, through the
    /// capture/write-back path. The white 50%-alpha overlay exercises
    /// the write-back blend too; tolerance 4 covers its extra
    /// quantization on top of the filter pass (the capture itself is
    /// an exact copy).
    #[test]
    fn backdrop_filter_grayscale_nonuniform_cpu_gpu_agree() {
        let mut list = DisplayList::new();
        // Non-uniform backdrop: left red, right blue.
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 50.0, 100.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(50.0, 0.0, 50.0, 100.0),
            color: Color::rgba8(0, 0, 255, 255),
        });
        list.push(DisplayItem::PushBackdropFilter {
            ops: vec![FilterOp::Grayscale(1.0)].into(),
            rect: rect(10.0, 10.0, 80.0, 80.0),
        });
        // The element's own content paints over the filtered backdrop.
        list.push(DisplayItem::FillRect {
            rect: rect(10.0, 10.0, 80.0, 80.0),
            color: Color::rgba8(255, 255, 255, 128),
        });
        list.push(DisplayItem::PopBackdropFilter);
        assert_cpu_gpu_agree(&list, 100, 100, 4, "backdrop-filter non-uniform");
    }

    /// `filter_blur_uniform` clamps the radius to the texture's larger
    /// dimension (mirrors the CPU's `clamp_blur_radius`) and to
    /// `GPU_BLUR_RADIUS_CAP` (a deliberate, documented GPU-only
    /// perf-vs-fidelity tradeoff: the shader taps `2r+1` texels per
    /// pixel per pass, so an uncapped radius is a TDR risk). Pure
    /// CPU-side check, no adapter needed.
    #[test]
    fn filter_blur_uniform_clamps_radius() {
        let u = WgpuBackend::filter_blur_uniform(100, 80, 500.0, 0);
        let radius = f32::from_le_bytes(u[88..92].try_into().unwrap());
        assert_eq!(radius, 100.0, "radius must clamp to max(w, h)");
        assert_eq!(u32::from_le_bytes(u[92..96].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(u[96..100].try_into().unwrap()), 1);
        // On a large texture the texture-size clamp does not bind, so
        // the GPU perf cap must: 500 -> 256.
        let u = WgpuBackend::filter_blur_uniform(2000, 1000, 500.0, 1);
        let radius = f32::from_le_bytes(u[88..92].try_into().unwrap());
        assert_eq!(
            radius,
            WgpuBackend::GPU_BLUR_RADIUS_CAP,
            "radius must clamp to the GPU perf cap"
        );
        assert_eq!(u32::from_le_bytes(u[92..96].try_into().unwrap()), 1);
        // Small radii pass through untouched.
        let u = WgpuBackend::filter_blur_uniform(100, 80, 7.0, 1);
        let radius = f32::from_le_bytes(u[88..92].try_into().unwrap());
        assert_eq!(radius, 7.0);
        assert_eq!(u32::from_le_bytes(u[92..96].try_into().unwrap()), 1);
    }

    /// `filter: blur()` over NON-UNIFORM content must match the CPU's
    /// 3-fold separable box blur. Catches the old WGPU no-op
    /// passthrough and the `vs_filter` UV bug alike: either defect
    /// shifts pixel values by tens of LSBs at a sharp edge.
    ///
    /// Tolerance derivation: the kernel accumulates exact texel values
    /// as integers and divides with integer division, bit-exact with
    /// the CPU's u32 window sums in `box_blur_sigma_rgba`, so the
    /// premultiplied bytes agree exactly. The only remaining
    /// divergence is the readback unpremultiply rounding (<= 1 LSB per
    /// channel: the CPU's rounds half-up, the GPU's rounds to nearest).
    /// Measured worst case under SwiftShader is 0 LSB over the whole
    /// image; the tolerance of 8 is generous slack for driver
    /// differences in sampling the group texture's edge texels.
    #[test]
    fn cpu_gpu_filtered_blur_agreement() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushFilter {
            ops: vec![FilterOp::Blur(4.0)].into(),
        });
        // Sharp vertical edge: left black, right white.
        list.push(DisplayItem::FillRect {
            rect: rect(10.0, 10.0, 40.0, 80.0),
            color: Color::rgba8(0, 0, 0, 255),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(50.0, 10.0, 40.0, 80.0),
            color: Color::rgba8(255, 255, 255, 255),
        });
        list.push(DisplayItem::PopFilter);
        assert_cpu_gpu_agree(&list, 100, 100, 8, "blur");
    }

    /// `filter: drop-shadow()` must match the CPU's
    /// [`apply_drop_shadow`](crate::cpu::Painter::apply_drop_shadow).
    /// Exercises the drop-shadow internal blur path, which runs the
    /// same mode-1 box-blur kernel as `filter: blur()` (tint, then 3 x
    /// (H, V) passes, then the offset merge), so the truncation fix in
    /// `SHADER_FILTER` covers it too. The offset (5, 7) keeps the
    /// shadow's spill fringe over the background, where the old
    /// round-vs-truncate divergence was amplified by the unpremultiply.
    #[test]
    fn cpu_gpu_filtered_drop_shadow_agreement() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushFilter {
            ops: vec![FilterOp::DropShadow {
                offset: (5.0, 7.0),
                blur: 4.0,
                color: Color::rgba8(255, 0, 0, 255),
            }]
            .into(),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(20.0, 20.0, 40.0, 40.0),
            color: Color::rgba8(0, 0, 255, 255),
        });
        list.push(DisplayItem::PopFilter);
        assert_cpu_gpu_agree(&list, 100, 100, 8, "drop-shadow");
    }

    /// `clip-path: polygon()` triangle: exact clip math, tight tolerance.
    #[test]
    fn clip_path_polygon_cpu_gpu_agree() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClipPath {
            points: vec![[10.0, 10.0], [90.0, 10.0], [50.0, 90.0]].into(),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::PopClip);
        assert_cpu_gpu_agree(&list, 100, 100, 2, "clip-path polygon");
    }
}
