//! Display list: the paint crate's backend-neutral scene description.
//!
//! [`build_display_list`] walks a [`LayoutBox`] tree and emits one
//! [`DisplayItem`] per paint primitive, in painter's-algorithm order
//! (later items paint over earlier ones). Every coordinate is absolute
//! page CSS px, y-down, so scrolling is a backend-side translate and
//! never requires rebuilding the list.
//!
//! Text is stored verbatim: one [`DisplayItem::TextRun`] per layout text
//! fragment. The run shares the fragment's originating box text via
//! `Arc<str>` (one allocation per text box per rebuild, not one per
//! fragment) and records the fragment's byte range into it. Layout
//! has already applied whitespace collapsing and `text-transform`, so
//! backends must render the slice as-is. Shaping (glyph ids, advances)
//! is the backend's job; the builder resolves each run's [`FontKey`]
//! through a caller-supplied resolver, defaulting to [`FontKey(0)`] in
//! [`build_display_list_with_images`].
//!
//! # Interface history (2026-09-25)
//!
//! This module replaces the wgpu worker's stand-in. The visible changes:
//!
//! - `DisplayItem::TextRun` is now `{ text, text_range, rect, baseline,
//!   font_key, size_px, color }`: `text` is the originating box's whole
//!   [`LayoutBox::text`] shared across the box's fragments via `Arc<str>`
//!   and `text_range` the fragment's byte range into it (verbatim per
//!   [`bowser_layout::TextFragment`], not pre-shaped glyphs). Backends
//!   shape at paint time (the fonts/text workers own shaping and
//!   rasterization).
//! - `ShapedGlyph` is removed; nothing consumes it anymore.
//! - `FontData` is defined once in the (crate-private) `fonts` module
//!   (it owns the font bytes, face index, and `font_ref()`) and
//!   re-exported here, so there is exactly one definition; backends
//!   import it from either module.
//! - New [`DisplayListBuilder`] with `push_*` methods and `build()`;
//!   [`build_display_list`] is written against it.
//! - The agreed-spec stub (`Quad`, `Color`, `BorderStyle`, `DecoKind`,
//!   `MarkerKind`) is unchanged, except that `Color` is now a packed
//!   RGBA8 `u32` (`0xRRGGBBAA`) instead of 4×f32: `Color::rgba8` and
//!   `Color::TRANSPARENT` keep their signatures, `with_opacity` keeps
//!   its semantics, and `r()`/`g()`/`b()`/`a()` (0..1 floats) plus
//!   `r8()`/`g8()`/`b8()`/`a8()` (bytes) replace field access.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use bowser_css::{
    ComputedStyle, CssValue, Filter, TransformContext, TransformOrigin, evaluate_transform,
    length_to_px, props, resolve_absolute_length,
};
use bowser_dom::{Document, NodeId, NodeKind};
use bowser_layout::{
    BoxType, CollapsedBorderStyle, ColumnRuleSpec, ColumnRuleStyle, LayoutBox, Rect, Size,
    TextFragment, resolve_font_size, sane_len, take_column_rules,
};

use crate::filter::{
    FilterOp, LengthCtx, ResolvedClip, filter_memo_key, resolve_clip_path, resolve_filter_chain,
};
use crate::fonts::FnvHashMap;
use crate::gradient::{Gradient, ParsedGradient, parse_gradient};

// ---------------------------------------------------------------------------
// Agreed-spec stub (borders worker; do not change without that worker).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quad {
    pub rect: Rect,
    pub color: Color,
}

/// Packed RGBA8 color: `0xRRGGBBAA`, red in the most significant byte.
///
/// Display lists are rebuilt every frame, so this is hot: 4 bytes
/// instead of 16 shrinks every [`DisplayItem`] carrying a color
/// (`FillRect`, `TextRun`, `Decoration`, `ListMarker`, and the four
/// colors of `Border`, whose payload drops from 64 to 16 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color(u32);

impl Color {
    /// Build from 8-bit channels.
    pub fn rgba8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self((u32::from(r) << 24) | (u32::from(g) << 16) | (u32::from(b) << 8) | u32::from(a))
    }

    /// Build from 0..1 f32 channels, rounding each to the nearest byte
    /// exactly the way the backends' f32→u8 conversion always did, so a
    /// color computed in f32 (border shading in [`crate::borders`])
    /// packs to the same bytes the old 4×f32 pipeline rasterized.
    pub fn from_f32(r: f32, g: f32, b: f32, a: f32) -> Self {
        fn byte(v: f32) -> u32 {
            (v.clamp(0.0, 1.0) * 255.0).round() as u32
        }
        Self((byte(r) << 24) | (byte(g) << 16) | (byte(b) << 8) | byte(a))
    }

    /// Transparent black.
    pub const TRANSPARENT: Self = Self(0);

    /// Multiply straight alpha by `opacity` (used for opacity groups).
    ///
    /// Rounds the scaled alpha byte. The old pipeline multiplied the
    /// f32 alpha and rounded once at backend conversion; the
    /// multiply-then-round commutes with the pack/unpack roundtrip, so
    /// this is byte-identical for every input.
    #[must_use]
    pub fn with_opacity(self, opacity: f32) -> Self {
        // `as u8` saturates, matching the old clamp-then-round.
        let a = (f32::from(self.a8()) * opacity).round() as u8;
        Self((self.0 & 0xFF_FF_FF_00) | u32::from(a))
    }

    /// 8-bit channels.
    pub fn r8(self) -> u8 {
        (self.0 >> 24) as u8
    }

    /// 8-bit channels.
    pub fn g8(self) -> u8 {
        (self.0 >> 16) as u8
    }

    /// 8-bit channels.
    pub fn b8(self) -> u8 {
        (self.0 >> 8) as u8
    }

    /// 8-bit channels.
    pub fn a8(self) -> u8 {
        self.0 as u8
    }

    /// Red as a 0..1 float (`r8 as f32 / 255.0`): bit-identical to the
    /// f32 the old 4×f32 `Color` stored for `rgba8`-built colors.
    pub fn r(self) -> f32 {
        f32::from(self.r8()) / 255.0
    }

    /// Green as a 0..1 float; see [`Color::r`].
    pub fn g(self) -> f32 {
        f32::from(self.g8()) / 255.0
    }

    /// Blue as a 0..1 float; see [`Color::r`].
    pub fn b(self) -> f32 {
        f32::from(self.b8()) / 255.0
    }

    /// Alpha as a 0..1 float; see [`Color::r`].
    pub fn a(self) -> f32 {
        f32::from(self.a8()) / 255.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BorderStyle {
    None,
    Hidden,
    Dotted,
    Dashed,
    Solid,
    Double,
    Groove,
    Ridge,
    Inset,
    Outset,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoKind {
    Underline,
    Overline,
    LineThrough,
    WavyUnderline,
    WavyOverline,
    WavyLineThrough,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkerKind {
    Disc,
    Circle,
    Square,
    Decimal(u32),
    /// `list-style-type: decimal-leading-zero`; the numeral is zero-padded
    /// to at least two digits. Added by the borders worker alongside
    /// `borders::marker_text`; like `Decimal`, it renders as text, not
    /// geometry.
    DecimalLeadingZero(u32),
}

// ---------------------------------------------------------------------------
// Types owned by this module.
// ---------------------------------------------------------------------------

// `FontData` is defined once in the (crate-private) `fonts` module; it
// owns the font bytes, face index, and `font_ref()`. Re-exported here so
// display-list consumers resolve `FontKey` and `FontData` from one place.
// (`fonts` re-exports `FontKey` from here in return; the re-exports are
// acyclic.)
//
// The doc comment below shadows the target's docs so public
// documentation never links into the private `fonts` module.

/// Raw font face data for a [`FontKey`].
///
/// Owns the raw font file bytes (SFNT: TTF/OTF) and the face index within
/// the file; typographic access goes through `font_ref()` on the defining
/// type in the `fonts` module.
pub use crate::fonts::FontData;

/// Opaque handle to a decoded image cached by the backend.
///
/// The [`DisplayList`] owns the decoded bytes; backends upload them to GPU
/// textures on first use and cache by this id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageId(pub u32);

/// Opaque handle to a font face known to the text subsystem.
///
/// Key space is defined here; backends resolve keys through a
/// `&dyn Fn(FontKey) -> Option<FontData>` supplied by the text subsystem.
/// [`FontKey(0)`] is the default/fallback face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FontKey(pub u32);

/// A decoded image: straight (non-premultiplied) RGBA8, row-major, top row
/// first.
#[derive(Clone, Debug)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl DecodedImage {
    /// Build from raw RGBA8 bytes; panics if the length mismatches.
    pub fn from_rgba(width: u32, height: u32, rgba: Vec<u8>) -> Self {
        // Checked: `width as usize * height as usize * 4` overflows even
        // 64-bit usize above ~4 gigapixels; a wrapped product would let a
        // mismatched buffer through (release) or panic with the wrong
        // message (debug).
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4));
        assert_eq!(
            expected,
            Some(rgba.len()),
            "rgba length must match dimensions"
        );
        Self {
            width,
            height,
            rgba,
        }
    }

    /// Full-content identity hash: FNV-1a 64 over the width, height, and
    /// every RGBA byte.
    ///
    /// Texture caches key uploads by [`ImageId`], but ids are only unique
    /// within one [`DisplayList`]: lists are rebuilt every frame (often at
    /// reused addresses) and [`DisplayList::add_image`] can re-register an
    /// id with different bytes. [`DisplayList`] computes this once per
    /// registration and caches it ([`DisplayList::image_content_hash`]) —
    /// registered bytes are immutable through every list API, so the
    /// cached hash always matches the stored bytes and backends never
    /// rehash the pixel buffer per frame. A standalone `DecodedImage`
    /// (never registered) can still have its `pub` buffer mutated in
    /// place; this method then hashes the *current* bytes on demand, so
    /// the mutation is overwhelmingly likely to change the hash (FNV-1a
    /// 64 is not collision-free, so this is probabilistic, not a
    /// guarantee).
    ///
    /// The hash alone is never trusted as content identity: cache hits
    /// additionally verify the stored dimensions and byte length against
    /// the current source (see `CpuImageEntry::matches`; the GPU path
    /// checks `entry.hash == hash && entry.width == decoded.width &&
    /// entry.height == decoded.height` in `ensure_image`), and a hit
    /// with mismatched shape or length is treated as a miss and decoded
    /// fresh. Not security against a dedicated collision search — FNV-1a
    /// is not collision-resistant — but a wrong-shape/wrong-length
    /// alias can never serve stale pixels.
    pub fn content_hash(&self) -> u64 {
        const PRIME: u64 = 0x1000_0000_01b3;
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |word: u64| {
            h ^= word;
            h = h.wrapping_mul(PRIME);
        };
        // Dimensions first so same-bytes/different-shape images differ.
        mix(u64::from(self.width));
        mix(u64::from(self.height));
        // Chunked over 8-byte words: one xor+multiply per word instead of
        // per byte. Compared only within one process, so the
        // platform-dependent word chunking is fine.
        let (words, tail) = self.rgba.as_chunks::<8>();
        for w in words {
            mix(u64::from_ne_bytes(*w));
        }
        for &b in tail {
            h ^= u64::from(b);
            h = h.wrapping_mul(PRIME);
        }
        h
    }
}

/// How many [`FilterOp`]s a [`FilterOps`] holds inline before spilling
/// to the heap. 1-2 covers the typical `filter`/`backdrop-filter`
/// chain (`blur(8px)`, `grayscale(50%) brightness(1.2)`) with zero
/// allocation; the inline case keeps [`DisplayItem`] at 88 B (see the
/// size assertion in the tests below).
const FILTER_OPS_INLINE: usize = 2;

/// How many vertices a [`ClipPoints`] holds inline before spilling to
/// the heap. 8 covers typical authored `polygon()` shapes with zero
/// allocation; tessellated `circle()`/`ellipse()` shapes spill.
/// Inline keeps [`DisplayItem`] at 88 B (see the size assertion below).
const CLIP_POINTS_INLINE: usize = 8;

/// Inline-first storage for a resolved `filter`/`backdrop-filter` op
/// chain.
///
/// A fresh `Vec<FilterOp>` per filtered display item per frame is
/// allocator churn on the paint hot path; typical chains are 1-2 ops,
/// so the first [`FILTER_OPS_INLINE`] live directly in the
/// [`DisplayItem`] and only longer chains spill to a heap [`Vec`].
/// [`Deref`](std::ops::Deref) exposes the ops as a `[FilterOp]` slice,
/// so all consumers (the backends' `&[FilterOp]` entry points, the
/// emptiness checks) read them exactly like the old `Vec`.
///
/// `PartialEq` compares the live op slice (see
/// [`as_slice`](Self::as_slice)), not the storage representation: an
/// `Inline` and a `Spill` holding the same ops compare equal, so a
/// future non-canonical constructor cannot silently break equality.
#[derive(Clone, Debug)]
pub enum FilterOps {
    /// `len` live ops at the front of `ops`; the remainder are
    /// `Blur(0.0)` fillers, never observed (every accessor goes
    /// through the [`Deref`](std::ops::Deref) slice, which covers only
    /// the live prefix).
    Inline {
        len: u8,
        ops: [FilterOp; FILTER_OPS_INLINE],
    },
    /// More than [`FILTER_OPS_INLINE`] ops.
    Spill(Vec<FilterOp>),
}

impl FilterOps {
    /// Empty op chain.
    pub fn new() -> Self {
        FilterOps::Inline {
            len: 0,
            ops: [FilterOp::Blur(0.0), FilterOp::Blur(0.0)],
        }
    }

    /// The ops as a slice.
    pub fn as_slice(&self) -> &[FilterOp] {
        self
    }
}

impl Default for FilterOps {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Vec<FilterOp>> for FilterOps {
    /// Canonical construction: `<= INLINE` ops go inline (the source
    /// `Vec`'s buffer is dropped — the win is that the *display item*
    /// no longer owns a heap allocation).
    fn from(ops: Vec<FilterOp>) -> Self {
        if ops.len() <= FILTER_OPS_INLINE {
            let mut inline = [FilterOp::Blur(0.0), FilterOp::Blur(0.0)];
            let mut len = 0u8;
            for op in ops {
                inline[len as usize] = op;
                len += 1;
            }
            FilterOps::Inline { len, ops: inline }
        } else {
            FilterOps::Spill(ops)
        }
    }
}

impl std::ops::Deref for FilterOps {
    type Target = [FilterOp];

    fn deref(&self) -> &[FilterOp] {
        match self {
            FilterOps::Inline { len, ops } => &ops[..*len as usize],
            FilterOps::Spill(ops) => ops,
        }
    }
}

impl PartialEq for FilterOps {
    /// Content equality: compare the live op slices, ignoring whether
    /// the ops live inline or spilled.
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

/// Inline-first storage for a `clip-path` polygon's vertices.
///
/// Same rationale as [`FilterOps`]: one `Vec<[f32; 2]>` per
/// clip-path display item per frame, with typical authored polygons
/// at 3-8 vertices going inline. [`Deref`](std::ops::Deref) exposes
/// the points as a `[[f32; 2]]` slice, so the backends' polygon entry
/// points are unchanged.
/// `PartialEq` compares the live vertex slice (see
/// [`as_slice`](Self::as_slice)), not the storage representation: an
/// `Inline` and a `Spill` holding the same points compare equal, so a
/// future non-canonical constructor cannot silently break equality.
#[derive(Clone, Debug)]
pub enum ClipPoints {
    /// `len` live vertices at the front of `points`; the remainder
    /// are `[0.0, 0.0]` fillers, never observed.
    Inline {
        len: u8,
        points: [[f32; 2]; CLIP_POINTS_INLINE],
    },
    /// More than [`CLIP_POINTS_INLINE`] vertices.
    Spill(Vec<[f32; 2]>),
}

impl ClipPoints {
    /// Empty point list.
    pub fn new() -> Self {
        ClipPoints::Inline {
            len: 0,
            points: [[0.0, 0.0]; CLIP_POINTS_INLINE],
        }
    }

    /// The points as a slice.
    pub fn as_slice(&self) -> &[[f32; 2]] {
        self
    }
}

impl Default for ClipPoints {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Vec<[f32; 2]>> for ClipPoints {
    /// Canonical construction: `<= INLINE` points go inline.
    fn from(points: Vec<[f32; 2]>) -> Self {
        if points.len() <= CLIP_POINTS_INLINE {
            let mut inline = [[0.0, 0.0]; CLIP_POINTS_INLINE];
            let mut len = 0u8;
            for p in points {
                inline[len as usize] = p;
                len += 1;
            }
            ClipPoints::Inline {
                len,
                points: inline,
            }
        } else {
            ClipPoints::Spill(points)
        }
    }
}

impl std::ops::Deref for ClipPoints {
    type Target = [[f32; 2]];

    fn deref(&self) -> &[[f32; 2]] {
        match self {
            ClipPoints::Inline { len, points } => &points[..*len as usize],
            ClipPoints::Spill(points) => points,
        }
    }
}

impl PartialEq for ClipPoints {
    /// Content equality: compare the live vertex slices, ignoring
    /// whether the points live inline or spilled.
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

/// One paint command. Items paint in order; later items cover earlier ones
/// (painter's algorithm). All rects are absolute page CSS px.
#[derive(Clone, Debug)]
pub enum DisplayItem {
    /// Solid rectangle.
    FillRect { rect: Rect, color: Color },
    /// Box border. `widths`/`styles`/`colors` are [top, right, bottom, left]
    /// and match [`crate::borders::tessellate_border`]'s signature
    /// directly. `radii` is the outer border radius in px per corner,
    /// ordered [top-left, top-right, bottom-right, bottom-left]; backends
    /// lower through [`crate::borders::tessellate_border_rounded`] when
    /// any radius is positive.
    Border {
        rect: Rect,
        widths: [f32; 4],
        styles: [BorderStyle; 4],
        colors: [Color; 4],
        radii: [f32; 4],
    },
    /// Verbatim text run: one per layout [`TextFragment`].
    ///
    /// `text` is the originating box's whole [`LayoutBox::text`], shared
    /// across the box's fragments via `Arc` (one allocation per text box
    /// per rebuild, not one per fragment); `text_range` is the
    /// fragment's byte range into it, already whitespace-collapsed and
    /// text-transformed by layout. Backends render
    /// `text.get(text_range).unwrap_or("")` as-is and shape it
    /// themselves. `rect` is the fragment's placed
    /// rect and `baseline` the line's baseline, both absolute page px.
    TextRun {
        text: Arc<str>,
        text_range: Range<usize>,
        rect: Rect,
        baseline: f32,
        font_key: FontKey,
        size_px: f32,
        color: Color,
    },
    /// Replaced-element image, stretched to `rect`.
    Image { rect: Rect, image: ImageId },
    /// One tile of a background image. `rect` is the tile's placed rect at
    /// the image's native size, so a backend's stretch-to-`rect` is the
    /// identity. The builder emits one item per visible tile inside a
    /// `PushClip`/`PopClip` pair around the background painting area, so
    /// partial edge tiles clip correctly and backends need no tiling
    /// support of their own; see
    /// [`DisplayListBuilder::push_background_image_tiled`].
    BackgroundImage { rect: Rect, image: ImageId },
    /// A CSS gradient (`linear-gradient()`, `radial-gradient()`,
    /// `conic-gradient()`, and the `repeating-` variants) painted over
    /// `rect` (the background painting area, i.e. the border box).
    /// `gradient` is already resolved against `rect` (see
    /// [`ParsedGradient::resolve`]): backends rasterize it directly
    /// with no CSS unit handling. The CPU backend evaluates it
    /// per-pixel; the GPU backend bakes it to a texture at device
    /// resolution and draws one stretched quad.
    Gradient { rect: Rect, gradient: Gradient },
    /// Tiled background image, expanded natively by the backends.
    ///
    /// `area` is the background painting area (the border box) in page
    /// px; backends clip tile output to it. `tile` is the tile size in px
    /// (`background-repeat: round` is pre-applied by the builder, which
    /// rescales the tile and emits [`BackgroundRepeat::Repeat`]).
    /// `offset` is the `background-position` offset of the tile grid's
    /// origin within `area`.
    ///
    /// Unlike the per-tile [`DisplayItem::BackgroundImage`] expansion
    /// this replaces, one item covers an arbitrarily large area: the
    /// CPU backend tiles inside the rasterizer and the GPU backend
    /// draws a single quad with UV wrapping, so there is no tile-count
    /// cap on the item list. Backends clamp `area` to
    /// [`MAX_REPEAT_AREA_DIM`] per axis before expanding and enlarge
    /// `tile` via [`cap_repeat_tile`] so the expansion stays under
    /// [`MAX_REPEAT_TILES`] tiles, which keeps hostile inputs bounded.
    RepeatImage {
        area: Rect,
        image: ImageId,
        tile: (f32, f32),
        repeat: BackgroundRepeat,
        offset: (f32, f32),
    },
    /// Text decoration. `rect` is the text's ink box; the backend positions
    /// the line per `kind` with the given `thickness`.
    Decoration {
        rect: Rect,
        kind: DecoKind,
        color: Color,
        thickness: f32,
    },
    /// List item marker drawn centered in `rect`. `index` is the item's
    /// 1-based position among its `<li>` siblings; `Decimal`/
    /// `DecimalLeadingZero` markers render as text, with the numeral
    /// taken from `index` and formatted per `kind` (`"3"`, `"03"`) in
    /// the item's `font_key`. Backends share that formula (see the GPU
    /// backend's marker lowering).
    ListMarker {
        rect: Rect,
        kind: MarkerKind,
        color: Color,
        index: u32,
        font_key: FontKey,
    },
    /// Intersect `rect` with the current clip; backend keeps a stack.
    PushClip { rect: Rect },
    /// Intersect a rounded rect with the current clip; backend keeps a
    /// stack. `radii` is the outer border radius in px per corner,
    /// `[top-left, top-right, bottom-right, bottom-left]` (the same
    /// order [`DisplayItem::Border`] uses), already CSS-clamped to
    /// `rect` by the builder. Backends evaluate the stack's geometric
    /// intersection per pixel, so rounded and rect entries nest
    /// correctly in any order. All-zero radii behave exactly like
    /// [`DisplayItem::PushClip`].
    PushRoundedClip { rect: Rect, radii: [f32; 4] },
    /// Pop the innermost clip (pops [`DisplayItem::PushClip`],
    /// [`DisplayItem::PushRoundedClip`], and
    /// [`DisplayItem::PushClipPath`] alike; backends keep one clip
    /// stack).
    PopClip,
    /// Intersect a `clip-path` polygon with the current clip; backend
    /// keeps a stack. `points` are absolute page-px vertices (a
    /// `circle()`/`ellipse()` tessellated by the builder, a `polygon()`
    /// as authored), evaluated with the even-odd rule. The builder
    /// emits this only for shapes with at least 3 finite vertices;
    /// `inset()` without `round` becomes [`DisplayItem::PushClip`] and
    /// with `round` becomes [`DisplayItem::PushRoundedClip`] instead.
    /// Inline-first storage ([`ClipPoints`]): typical authored
    /// polygons allocate nothing.
    PushClipPath { points: ClipPoints },
    /// Multiply subsequent item colors' alpha by `opacity` until the
    /// matching pop.
    PushOpacity { opacity: f32 },
    /// Pop the innermost opacity group.
    PopOpacity,
    /// Blend the following isolated group's rendering against the
    /// backdrop with `mode` until the matching pop. Backends render the
    /// group's items into an offscreen surface at their own alphas,
    /// then composite that surface over the current target with the
    /// blend formula (CSS compositing: the blend applies per stacking
    /// context against the backdrop). `Normal` is a no-op: backends
    /// paint the range inline with plain source-over.
    PushBlend { mode: BlendMode },
    /// Pop the innermost blend group.
    PopBlend,
    /// Apply a resolved CSS `filter` chain to the following isolated
    /// group's rendering (Filter Effects 1 §8) until the matching pop.
    /// Backends render the group's items into an offscreen surface at
    /// their own alphas, run the [`FilterOp`](crate::filter::FilterOp)
    /// chain over it (see
    /// [`plan_filter_passes`](crate::filter::plan_filter_passes)), then
    /// composite the filtered surface over the current target with
    /// plain source-over. An empty `ops` is a no-op: the builder never
    /// emits it (zero-cost no-filter fast path). Inline-first storage
    /// ([`FilterOps`]): typical 1-2 op chains allocate nothing.
    PushFilter { ops: FilterOps },
    /// Pop the innermost filter group.
    PopFilter,
    /// Apply a resolved CSS `backdrop-filter` chain to the
    /// already-painted backdrop (Filter Effects 2 §2) until the
    /// matching pop. `rect` is the element's border box in page px: the
    /// backdrop region that gets filtered. Backends copy the target's
    /// content under `rect`, run the
    /// [`FilterOp`](crate::filter::FilterOp) chain over the copy, write
    /// the filtered backdrop back, then composite the group's own items
    /// (rendered isolated, like [`DisplayItem::PushFilter`]) over it
    /// with plain source-over. An empty `ops` is a no-op: the builder
    /// never emits it. Inline-first storage ([`FilterOps`]), like
    /// [`DisplayItem::PushFilter`].
    PushBackdropFilter { ops: FilterOps, rect: Rect },
    /// Pop the innermost backdrop-filter group.
    PopBackdropFilter,
    /// Transform the following group's rendering by a 2D matrix until
    /// the matching pop (CSS Transforms 1). `matrix` is
    /// `(a, b, c, d, e, f)`: a page-px point `(x, y)` paints at
    /// `(a*x + c*y + e, b*x + d*y + f)` — the same component order as
    /// [`bowser_css::TransformMatrix`]. The builder folds
    /// `transform-origin` into the matrix
    /// (`T(origin) · M · T(−origin)`), so backends apply it as-is.
    /// Backends render the group's items into an offscreen surface
    /// untransformed, then composite the surface through the matrix
    /// (bilinear); clips pushed outside the group apply to the
    /// transformed result, clips inside to the untransformed content.
    /// An identity matrix is a no-op: backends paint the range inline.
    /// A non-invertible matrix paints nothing (per spec, the element
    /// renders nothing).
    PushTransform { matrix: [f32; 6] },
    /// Pop the innermost transform group.
    PopTransform,
    /// A `box-shadow` (CSS Backgrounds 3 §7.2). `rect` is the border
    /// box in page px, `radii` its border radii
    /// `[top-left, top-right, bottom-right, bottom-left]`. The shadow
    /// shape is `rect` inflated (outset) or deflated (inset) by
    /// `spread`, shifted by `offset`, blurred with radius `blur`
    /// (gaussian, σ ≈ `blur` / 2), and drawn in `color`. Outset
    /// shadows paint under the element's background (the builder emits
    /// this before the background fill); inset shadows paint over the
    /// background but under content. `blur`/`spread`/`offset` are page
    /// px; negative `blur` is treated as zero.
    BoxShadow {
        rect: Rect,
        radii: [f32; 4],
        color: Color,
        offset: (f32, f32),
        blur: f32,
        spread: f32,
        inset: bool,
    },
    /// A `text-shadow` (CSS Text Decoration 3 §4.3). The text fields
    /// mirror [`DisplayItem::TextRun`] (same fragment, same shaping);
    /// backends paint the run a second time in `color`, shifted by
    /// `offset` and blurred with radius `blur` (gaussian,
    /// σ ≈ `blur` / 2), underneath the real run. `offset`/`blur` are
    /// page px; negative `blur` is treated as zero.
    TextShadow {
        text: Arc<str>,
        text_range: Range<usize>,
        rect: Rect,
        baseline: f32,
        font_key: FontKey,
        size_px: f32,
        color: Color,
        offset: (f32, f32),
        blur: f32,
    },
    /// Overscroll edge glow (phase 2 of the overscroll affordance):
    /// the gradient band painted along one scrollport edge while the
    /// transient rubber-band shift (phase 1) is in flight.
    /// `rect` is the glow *band* rect — the edge strip, not the whole
    /// scrollport — and `strength` is the peak intensity in `[0, 1]`,
    /// proportional to the rubber-band shift magnitude. Backends fade the fixed
    /// [`EDGE_GLOW_COLOR`] from full `strength` at the edge to
    /// transparent at the band's inner end (see
    /// [`edge_glow_alpha`]); the emitter wraps the item in a
    /// `PushClip`/`PopClip` pair at the scrollport so the glow never
    /// bleeds outside it. A dedicated item (rather than stacked
    /// `FillRect` alpha bands baked by the emitter) keeps the gradient
    /// analytic per backend: the CPU backend interpolates per
    /// scanline and the GPU backend reuses its solid-quad pipe with
    /// per-row alpha steps, so neither backend pays a pile of
    /// display-list items per frame of glow.
    EdgeGlow {
        edge: Edge4,
        rect: Rect,
        strength: f32,
    },
}

/// One physical edge of a box: the side a
/// [`DisplayItem::EdgeGlow`] hugs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge4 {
    Top,
    Right,
    Bottom,
    Left,
}

/// Depth of the overscroll glow band in page px: the gradient fades
/// from the scrollport edge to transparent over this distance. The
/// emitter clamps it to the scrollport's cross dimension, so a tiny
/// scrollport gets a proportionally thinner band rather than a band
/// that covers it whole.
pub const EDGE_GLOW_DEPTH: f32 = 96.0;

/// Fixed tint of the overscroll edge glow. Alpha comes from
/// `strength` × [`edge_glow_alpha`]; the color itself is a cool
/// white-blue in the platform-glow tradition. Built from the packed
/// word directly (`from_f32` is not `const`): `0x9EC7FFFF` is exactly
/// `from_f32(0.62, 0.78, 1.0, 1.0)` — `(0.62*255).round() = 158`,
/// `(0.78*255).round() = 199`.
pub const EDGE_GLOW_COLOR: Color = Color(0x9E_C7_FF_FF);

/// Peak alpha multiplier of the overscroll glow at full strength:
/// `strength == 1.0` (shift at the overscroll cap) paints the edge row
/// at this alpha.
pub const EDGE_GLOW_PEAK_ALPHA: f32 = 0.55;

/// Alpha profile of the overscroll glow: `t` is the fractional
/// distance from the glowing edge (`0.0` = at the edge, `1.0` = the
/// band's inner end), `strength` the item's peak intensity in
/// `[0, 1]`. Quadratic falloff: bright at the edge, gone before the
/// band ends, no visible step at the inner end. Both backends share
/// this so their pixels agree.
pub fn edge_glow_alpha(strength: f32, t: f32) -> f32 {
    let falloff = (1.0 - t.clamp(0.0, 1.0)).powi(2);
    strength.clamp(0.0, 1.0) * EDGE_GLOW_PEAK_ALPHA * falloff
}

/// The glow band rect for `edge` of `scrollport`: the edge strip
/// [`DisplayItem::EdgeGlow`] paints its gradient into. `None` when
/// the scrollport is degenerate (zero/negative cross dimension), so
/// the emitter skips the item instead of pushing an empty band.
pub fn edge_glow_band(scrollport: Rect, edge: Edge4) -> Option<Rect> {
    if !scrollport.w.is_finite() || !scrollport.h.is_finite() {
        return None;
    }
    let depth = match edge {
        Edge4::Top | Edge4::Bottom => EDGE_GLOW_DEPTH.min(scrollport.h),
        Edge4::Left | Edge4::Right => EDGE_GLOW_DEPTH.min(scrollport.w),
    };
    if depth <= 0.0 {
        return None;
    }
    Some(match edge {
        Edge4::Top => Rect {
            x: scrollport.x,
            y: scrollport.y,
            w: scrollport.w,
            h: depth,
        },
        Edge4::Bottom => Rect {
            x: scrollport.x,
            y: scrollport.y + scrollport.h - depth,
            w: scrollport.w,
            h: depth,
        },
        Edge4::Left => Rect {
            x: scrollport.x,
            y: scrollport.y,
            w: depth,
            h: scrollport.h,
        },
        Edge4::Right => Rect {
            x: scrollport.x + scrollport.w - depth,
            y: scrollport.y,
            w: depth,
            h: scrollport.h,
        },
    })
}

/// A `mix-blend-mode` value: how an isolated group's rendering blends
/// against its backdrop (CSS Compositing and Blending Level 1). Only
/// the most-used modes are supported; the cascade rejects the rest, so
/// the builder only ever sees these.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlendMode {
    /// Plain source-over (the initial value); backends paint inline.
    #[default]
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
}

impl BlendMode {
    /// Parse a `mix-blend-mode` keyword. `None` for anything else (the
    /// cascade's validator already rejects those, so this is a second
    /// line of defense for hand-built lists).
    pub fn from_css_keyword(s: &str) -> Option<Self> {
        Some(match s {
            "normal" => Self::Normal,
            "multiply" => Self::Multiply,
            "screen" => Self::Screen,
            "overlay" => Self::Overlay,
            "darken" => Self::Darken,
            "lighten" => Self::Lighten,
            _ => return None,
        })
    }

    /// Whether this mode needs no blending work at all.
    pub fn is_normal(self) -> bool {
        matches!(self, Self::Normal)
    }

    /// Discriminant for the GPU composite instance (`0` = normal).
    pub fn as_u32(self) -> u32 {
        match self {
            Self::Normal => 0,
            Self::Multiply => 1,
            Self::Screen => 2,
            Self::Overlay => 3,
            Self::Darken => 4,
            Self::Lighten => 5,
        }
    }
}

/// An ordered list of paint commands for one frame, plus the decoded
/// images its [`DisplayItem::Image`]/`BackgroundImage` items reference.
///
/// Backends resolve image bytes through [`DisplayList::image`]; this keeps
/// image upload lazy and backend-local (GPU texture cache keyed by
/// [`ImageId`]).
#[derive(Clone, Debug, Default)]
pub struct DisplayList {
    pub items: Vec<DisplayItem>,
    pub(crate) images: HashMap<ImageId, DecodedImage>,
    /// [`DecodedImage::content_hash`] per registered image, computed once
    /// in [`DisplayList::add_image`] — the only mutation point
    /// ([`DisplayList::image`] hands out `&`, so registered bytes are
    /// immutable afterward). Backends key their upload caches on this
    /// instead of rehashing the full pixel buffer every frame; a
    /// re-registered id overwrites the entry, so the hash always matches
    /// the stored bytes.
    pub(crate) image_hashes: HashMap<ImageId, u64>,
}

impl DisplayList {
    /// Empty display list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one item.
    pub fn push(&mut self, item: DisplayItem) {
        self.items.push(item);
    }

    /// Register decoded bytes for `id` (overwrites any previous entry).
    pub fn add_image(&mut self, id: ImageId, image: DecodedImage) {
        // Hash once here: the bytes cannot change after registration
        // (see the `image_hashes` field docs), so backends never rehash
        // them per frame.
        let hash = image.content_hash();
        self.image_hashes.insert(id, hash);
        self.images.insert(id, image);
    }

    /// Look up the decoded bytes for `id`, if registered.
    pub fn image(&self, id: ImageId) -> Option<&DecodedImage> {
        self.images.get(&id)
    }

    /// The content hash of the image registered under `id`, computed once
    /// at registration: backends use this for upload-cache validation
    /// instead of rehashing the pixel buffer every frame. `None` when `id`
    /// is not registered.
    pub(crate) fn image_content_hash(&self, id: ImageId) -> Option<u64> {
        self.image_hashes.get(&id).copied()
    }

    /// Number of registered images (useful for tests).
    pub fn image_count(&self) -> usize {
        self.images.len()
    }
}

// ---------------------------------------------------------------------------
// Scrollbars
// ---------------------------------------------------------------------------

/// Scrollbar track color: light neutral, opaque.
pub fn scrollbar_track_color() -> Color {
    Color::rgba8(234, 234, 234, 255)
}

/// Scrollbar thumb color: mid gray, opaque.
pub fn scrollbar_thumb_color() -> Color {
    Color::rgba8(158, 158, 158, 255)
}

/// Append one scrollbar axis' track + thumb as solid rects, above
/// everything already in `list`. Both rects are in page coordinates
/// (the display-list frame); the backend applies the viewport scroll
/// to them like any other item. No fancy styling: solid fills only.
pub fn push_scrollbar(list: &mut DisplayList, track: Rect, thumb: Rect) {
    list.push(DisplayItem::FillRect {
        rect: track,
        color: scrollbar_track_color(),
    });
    list.push(DisplayItem::FillRect {
        rect: thumb,
        color: scrollbar_thumb_color(),
    });
}

/// A paint backend: consumes a [`DisplayList`] and produces pixels.
///
/// `viewport` is the visible page area in CSS px; `scroll` is the page-px
/// offset of the viewport's top-left corner. Backends apply scroll as a
/// translate without rebuilding cached GPU resources.
pub trait Backend {
    fn paint(&mut self, list: &DisplayList, viewport: Size, scroll: (f32, f32));
}

/// Sane bound on a [`DisplayItem::RepeatImage`] destination rect: backends
/// clamp each axis of the repeat area to this many px before expanding
/// tiles. 8192 px per axis caps the expanded region at 256 MiB of
/// pixels — the same budget as the GPU image-upload cache — so a hostile
/// `background-repeat` over a gigantic area cannot blow up rasterizer
/// work or offscreen group textures, while every realistic page is
/// unaffected.
pub const MAX_REPEAT_AREA_DIM: f32 = 8192.0;

/// Upper bound on tiles one [`DisplayItem::RepeatImage`] may expand to.
/// Past the cap the tile is enlarged uniformly (like the old per-tile
/// emitter's `MAX_BACKGROUND_TILES` cap did), so the grid still covers
/// the whole area with no holes, while an absurdly small but finite tile
/// (e.g. `1e-30px` from a hostile stylesheet) cannot expand to trillions
/// of tiles in the backends.
pub(crate) const MAX_REPEAT_TILES: f64 = 65_536.0;

/// Clamp a repeat destination rect to [`MAX_REPEAT_AREA_DIM`] per axis,
/// anchored at the rect's top-left corner. Non-finite coordinates or
/// dimensions collapse to an empty rect at the origin; non-positive
/// dimensions clamp to zero.
pub(crate) fn clamp_repeat_area(area: Rect) -> Rect {
    let (x, y) = (area.x, area.y);
    if !x.is_finite() || !y.is_finite() || !area.w.is_finite() || !area.h.is_finite() {
        return Rect {
            x: 0.0,
            y: 0.0,
            w: 0.0,
            h: 0.0,
        };
    }
    Rect {
        x,
        y,
        w: area.w.clamp(0.0, MAX_REPEAT_AREA_DIM),
        h: area.h.clamp(0.0, MAX_REPEAT_AREA_DIM),
    }
}

/// Enlarge `tile` uniformly so a [`DisplayItem::RepeatImage`] over
/// `area` expands to at most [`MAX_REPEAT_TILES`] tiles. The enlarged
/// tile still covers the whole area with no holes; past the cap the grid
/// is coarser, which is the documented hostile-input degradation (see
/// [`MAX_REPEAT_TILES`]).
///
/// `tile` must be positive and finite on both axes (callers filter
/// degenerate tiles first); `area` should already be clamped with
/// [`clamp_repeat_area`]. f64 keeps the tile-count quotient finite even
/// for extreme inputs (a `1e-30` px tile over an 8192 px area counts
/// `~6.7e67` tiles without overflowing); the enlarged tile saturates at
/// `f32::MAX`, which the backends still bound via the area clamp.
pub(crate) fn cap_repeat_tile(tile: (f32, f32), area: Rect) -> (f32, f32) {
    let (tw, th) = tile;
    debug_assert!(tw > 0.0 && th > 0.0 && tw.is_finite() && th.is_finite());
    let total = (area.w as f64 / tw as f64).ceil() * (area.h as f64 / th as f64).ceil();
    if total > MAX_REPEAT_TILES {
        let s = (total / MAX_REPEAT_TILES).sqrt();
        (
            ((tw as f64 * s) as f32).min(f32::MAX),
            ((th as f64 * s) as f32).min(f32::MAX),
        )
    } else {
        (tw, th)
    }
}

/// Tile-boundary snap band for the GPU backend's single-quad repeat
/// lowering (the wrap axes of [`BackgroundRepeat::Repeat`]/`RepeatX`/
/// `RepeatY`): the f32 value the rasterizer interpolates across the
/// whole area in tile units carries a residual of ~1 ulp of the
/// interpolated value's magnitude (a few ulps worst case across
/// vendors' evaluation orders), so a fragment mathematically on a tile
/// boundary can land a hair above the integer, where `fract` wraps to
/// ~0 and samples the *next* tile's first texel while the CPU backend's
/// per-tile draws sample the *left* tile's last texel. The wrap branch
/// of the image shader snaps a wrapped coordinate inside this band to
/// 1.0 (clamp-to-edge then reads the last texel at full weight),
/// canonicalizing both backends to the CPU's left-tile convention.
///
/// `u0`/`u1` are the quad's vertex UVs in tile units on the axis. The
/// residue scales with the interpolated *magnitude* — the `u0` offset
/// included, not just the `u1 - u0` span — so the band is 4 ulps of
/// `max(|u0|, |u1|)`, floored at 1.0 so tiny spans still get a nonzero
/// band. A span-based band (the pre-fix formula) lets boundary
/// fragments escape the snap once `|u0|` exceeds ~2x the span, i.e.
/// with a large `background-position` offset: the GPU then samples the
/// wrong tile's texel while the CPU does not. The 0.25 cap keeps
/// adjacent boundary bands from ever meeting; past it (`|t|` above
/// ~524k tiles, reachable only via hostile hand-built offsets) the band
/// no longer covers the residue and boundary agreement degrades — the
/// documented degenerate case.
pub(crate) fn repeat_snap_band(u0: f32, u1: f32) -> f32 {
    (4.0 * f32::EPSILON * u0.abs().max(u1.abs()).max(1.0)).min(0.25)
}

/// Tile grid covering `cover` for a repeating background axis pair.
///
/// `origin` is the tile grid's origin (the positioned origin:
/// `area.x + offset.0`, `area.y + offset.1`), `tile` the tile size.
/// Returns `(i_start, j_start, nx, ny)`: the index ranges whose tiles
/// cover `cover`. On repeating axes the start indices step back so
/// tiles cover `cover`'s top-left corner even when the origin is
/// positive; on non-repeating axes exactly one tile is placed at the
/// origin. `cover` may be smaller than the full background area (e.g.
/// intersected with the visible region): the grid stays anchored at
/// `origin`, so the phase matches a full-area expansion exactly.
///
/// No epsilon is subtracted before the end-index ceil: a quotient that
/// lands a hair above an exact integer yields one phantom tile, which
/// is harmless (the area clip trims the overhang); but an epsilon here
/// would drop a genuine sub-epsilon sliver at the far edge (area width
/// 2.0000005 with tile 1.0 would leave the last 5e-7 px unpainted). The
/// +1e-6 on the start index only suppresses phantom *leading* tiles at
/// exact boundaries, never coverage.
pub(crate) fn background_tile_grid(
    cover: Rect,
    origin: (f32, f32),
    tile: (f32, f32),
    repeat_x: bool,
    repeat_y: bool,
) -> (i64, i64, i64, i64) {
    let (tw, th) = tile;
    let (x0, y0) = origin;
    let i_start = if repeat_x {
        ((cover.x - x0) / tw + 1e-6).floor() as i64
    } else {
        0
    };
    let j_start = if repeat_y {
        ((cover.y - y0) / th + 1e-6).floor() as i64
    } else {
        0
    };
    // Exclusive end indices covering the cover rect's bottom-right
    // corner (one tile when the axis does not repeat).
    let i_end = if repeat_x {
        ((cover.x + cover.w - x0) / tw).ceil() as i64
    } else {
        i_start + 1
    };
    let j_end = if repeat_y {
        ((cover.y + cover.h - y0) / th).ceil() as i64
    } else {
        j_start + 1
    };
    let nx = (i_end - i_start).max(1);
    let ny = (j_end - j_start).max(1);
    (i_start, j_start, nx, ny)
}

/// One axis of `background-repeat: space` placement: whole tiles only,
/// never clipped. Returns `(tile count, first-tile origin, pitch)`. A
/// multi-tile axis sits flush with both edges; the single-tile case
/// uses the positioned origin (pitch unused).
pub(crate) fn background_space_axis(
    start: f32,
    len: f32,
    tile_d: f32,
    off: f32,
) -> (i64, f32, f32) {
    let n = (len / tile_d).floor().max(1.0) as i64;
    if n == 1 {
        (1, start + off, tile_d)
    } else {
        let gap = (len - n as f32 * tile_d) / (n - 1) as f32;
        (n, start, tile_d + gap)
    }
}

/// Tiling mode for a CSS background image (`background-repeat`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackgroundRepeat {
    /// Tile in both axes (`repeat`).
    Repeat,
    /// Tile horizontally only; the single row sits at the vertical
    /// `background-position` offset (`repeat-x`).
    RepeatX,
    /// Tile vertically only; the single column sits at the horizontal
    /// `background-position` offset (`repeat-y`).
    RepeatY,
    /// One copy at the `background-position` offset (`no-repeat`).
    NoRepeat,
    /// Rescale each tile so a whole number of tiles fits exactly along
    /// each axis, then tile like [`BackgroundRepeat::Repeat`]
    /// (`round`).
    Round,
    /// Whole tiles only, never clipped: the first and last tiles sit
    /// flush with the area's edges and leftover space is distributed
    /// evenly between tiles (`space`). When only one tile fits on an
    /// axis, `background-position` places it, like `no-repeat`.
    Space,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Incremental [`DisplayList`] construction: `push_*` one item at a time,
/// then [`build`](Self::build) the finished list.
#[derive(Debug)]
pub struct DisplayListBuilder {
    list: DisplayList,
    next_image_id: u32,
    /// `column-rule` specs drained from layout by [`build_display_list`]
    /// (see [`take_column_rules`]); the box walk claims each spec when it
    /// visits the spec's container (see `paint_column_rules`).
    column_rules: ColumnRuleClaims,
    /// Memoized [`parse_gradient`] results by verbatim `background-image`
    /// source string: a page with N boxes sharing one gradient class
    /// parses the source once per display-list build instead of N times.
    /// `None` records a parse failure so failed sources are not re-parsed
    /// either (a failed gradient emits nothing, matching the rendering
    /// of a dropped declaration).
    ///
    /// Retained across builds on a reused builder (see
    /// [`begin_reused_build`](Self::begin_reused_build)): parsing is a
    /// pure function of the source text, so an entry can never go stale
    /// — the same source always parses the same way, whatever the
    /// styles around it do. Invalidation is still needed as a *memory*
    /// bound (a page that rewrites gradient sources every frame would
    /// otherwise grow this map without bound); that is what the
    /// [`style_epoch`](Self::style_epoch) scheme below provides.
    ///
    /// Values are shared by [`Arc`](std::sync::Arc): a cache hit hands
    /// out a refcount bump instead of cloning the stop list, so N boxes
    /// sharing one gradient class cost one parse and zero per-box
    /// copies per build. Callers only ever read through the `Arc`
    /// (see [`parsed_gradient`](Self::parsed_gradient)).
    ///
    /// FNV-1a-hashed (see [`FnvBuildHasher`](crate::fonts::FnvBuildHasher)):
    /// every lookup hashes the full gradient source string. Hash-only
    /// change; values are untouched.
    gradient_cache: FnvHashMap<Box<str>, Option<Arc<ParsedGradient>>>,
    /// Memoized [`resolve_filter_chain`](crate::filter::resolve_filter_chain)
    /// results by canonical filter text plus length context and
    /// `currentcolor` (see
    /// [`filter_memo_key`](crate::filter::filter_memo_key)): a page with
    /// N boxes sharing one filter class resolves the argument strings
    /// once per display-list build instead of N times. Empty chains are
    /// memoized too (no group is emitted for them either way).
    ///
    /// Retained across builds like `gradient_cache`. The memo key
    /// already captures every resolution input (function names, raw
    /// argument strings, the em/root/viewport length context, and the
    /// `currentcolor`), so an entry can never resolve *wrong* for a new
    /// frame — but per-frame-varying keys (e.g. an animated
    /// `filter: blur(<n>px)` rewritten each frame) would grow the map
    /// without bound. The [`style_epoch`](Self::style_epoch) scheme
    /// bounds it: any style recalculation clears both caches, so they
    /// hold at most one style generation's distinct sources.
    filter_cache: HashMap<Box<str>, FilterOps>,
    /// Style generation the caches were last validated against.
    ///
    /// Epoch/invalidation scheme: the builder itself owns no notion of
    /// "the document changed" — instead, each
    /// [`begin_reused_build`](Self::begin_reused_build) takes a
    /// `style_changed` flag from the caller (window.rs captures
    /// `tab.dirty.style` before the style stage runs; every restyle in
    /// this pipeline also rebuilds the display list in the same tick,
    /// so the flag is exact). On `true` the epoch bumps and both caches
    /// are cleared; on `false` they are kept, so scroll-only frames —
    /// the common case — hit warm caches instead of re-parsing.
    ///
    /// Correctness does not depend on the clear: both caches are keyed
    /// by their complete inputs (see the field docs), so a stale entry
    /// is impossible — the epoch exists to bound memory, not to fix
    /// staleness.
    style_epoch: u64,
}

impl DisplayListBuilder {
    /// Empty builder.
    pub fn new() -> Self {
        Self {
            list: DisplayList::new(),
            next_image_id: 1,
            column_rules: ColumnRuleClaims::default(),
            gradient_cache: FnvHashMap::default(),
            filter_cache: HashMap::new(),
            style_epoch: 0,
        }
    }

    /// Begin a new build reusing this builder's allocations.
    ///
    /// `out` is the previous frame's finished list (or an empty one):
    /// its item and image storage is cleared — keeping capacity — and
    /// adopted as this build's working storage, so the per-frame
    /// `Vec<DisplayItem>` regrow ramp is gone after the first frame.
    /// `finish_reused_build` hands the finished list back.
    ///
    /// `style_changed` must be true when styles were recalculated since
    /// the last build (see the [`style_epoch`](Self::style_epoch) field
    /// docs for the invalidation scheme). The image-id sequence
    /// restarts at 1 exactly like a fresh builder: backends key their
    /// texture upload caches on [`ImageId`], so cross-frame id
    /// stability for identical content is what keeps those caches
    /// hitting.
    pub fn begin_reused_build(&mut self, out: &mut DisplayList, style_changed: bool) {
        out.items.clear();
        out.images.clear();
        out.image_hashes.clear();
        // Adopt the recycled storage as the working list.
        std::mem::swap(&mut self.list, out);
        self.next_image_id = 1;
        self.column_rules.clear();
        if style_changed {
            self.style_epoch = self.style_epoch.wrapping_add(1);
            self.gradient_cache.clear();
            self.filter_cache.clear();
        }
    }

    /// Finish a [`begin_reused_build`](Self::begin_reused_build) build:
    /// move the working list into `out`, leaving an empty working list
    /// (whose storage the *next* `begin_reused_build` will adopt back)
    /// in the builder.
    pub fn finish_reused_build(&mut self, out: &mut DisplayList) {
        std::mem::swap(&mut self.list, out);
    }

    /// Stash `column-rule` specs drained from layout for this build.
    /// Called once by [`build_display_list`]; the box walk claims each
    /// spec when it visits the spec's container.
    pub(crate) fn set_column_rules(&mut self, specs: Vec<ColumnRuleSpec>) -> &mut Self {
        self.column_rules.insert_all(specs);
        self
    }

    /// Append a solid rectangle.
    pub fn push_fill_rect(&mut self, rect: Rect, color: Color) -> &mut Self {
        self.list.push(DisplayItem::FillRect { rect, color });
        self
    }

    /// Append a box border; `widths`/`styles`/`colors` are
    /// [top, right, bottom, left], `radii` is [top-left, top-right,
    /// bottom-right, bottom-left] in px.
    pub fn push_border(
        &mut self,
        rect: Rect,
        widths: [f32; 4],
        styles: [BorderStyle; 4],
        colors: [Color; 4],
        radii: [f32; 4],
    ) -> &mut Self {
        self.list.push(DisplayItem::Border {
            rect,
            widths,
            styles,
            colors,
            radii,
        });
        self
    }

    /// Append a verbatim text run (one per layout text fragment).
    ///
    /// `text` is the fragment's originating box text, shared across the
    /// box's fragments via `Arc` (wrap once per box, `Arc::clone` per
    /// fragment — no per-fragment allocation); `text_range` is the
    /// fragment's byte range into it, already whitespace-collapsed and
    /// text-transformed by layout. Backends render
    /// `text.get(text_range).unwrap_or("")` as-is and shape it
    /// themselves.
    #[allow(clippy::too_many_arguments)]
    pub fn push_text_run(
        &mut self,
        text: Arc<str>,
        text_range: Range<usize>,
        rect: Rect,
        baseline: f32,
        font_key: FontKey,
        size_px: f32,
        color: Color,
    ) -> &mut Self {
        self.list.push(DisplayItem::TextRun {
            text,
            text_range,
            rect,
            baseline,
            font_key,
            size_px,
            color,
        });
        self
    }

    /// Append a replaced-element image.
    pub fn push_image(&mut self, rect: Rect, image: ImageId) -> &mut Self {
        self.list.push(DisplayItem::Image { rect, image });
        self
    }

    /// Append a background image.
    pub fn push_background_image(&mut self, rect: Rect, image: ImageId) -> &mut Self {
        self.list.push(DisplayItem::BackgroundImage { rect, image });
        self
    }

    /// Append a tiled CSS background image over `area` (the background
    /// painting area, i.e. the border box).
    ///
    /// `tile` is the image's native size in px and `offset` the
    /// `background-position` offset of the tile grid's origin within
    /// `area`. Emits a single [`DisplayItem::RepeatImage`]: backends
    /// expand the tiling natively (the CPU backend tiles in the
    /// rasterizer, the GPU backend draws one quad with UV wrapping), so
    /// arbitrarily large areas need no per-tile items and no tile-count
    /// cap. Backends clamp `area` to [`MAX_REPEAT_AREA_DIM`] per axis
    /// before expanding, which bounds hostile inputs.
    ///
    /// `round` rescales the tile so a whole number of tiles spans each
    /// axis exactly, then tiles like `repeat`. `space` places whole
    /// unclipped tiles per [`background_space_axis`]; backends expand
    /// that placement natively too.
    ///
    /// Degenerate input (non-positive or non-finite tile, area, or
    /// offset) emits nothing. The expanded tile count is capped at
    /// 65,536: past the cap the tile is enlarged uniformly so the grid
    /// still covers the area, bounding hostile tiny tiles.
    pub fn push_background_image_tiled(
        &mut self,
        area: Rect,
        image: ImageId,
        tile: (f32, f32),
        repeat: BackgroundRepeat,
        offset: (f32, f32),
    ) -> &mut Self {
        let (tw, th) = tile;
        if tw <= 0.0
            || th <= 0.0
            || area.w <= 0.0
            || area.h <= 0.0
            || !tw.is_finite()
            || !th.is_finite()
            || !area.x.is_finite()
            || !area.y.is_finite()
            || !area.w.is_finite()
            || !area.h.is_finite()
            || !offset.0.is_finite()
            || !offset.1.is_finite()
        {
            return self;
        }
        // `round` rescales the tile so a whole number of tiles spans
        // each axis exactly, then tiles like `repeat`.
        let (tw, th, repeat) = match repeat {
            BackgroundRepeat::Round => (
                round_tile_len(area.w, tw),
                round_tile_len(area.h, th),
                BackgroundRepeat::Repeat,
            ),
            r => (tw, th, r),
        };
        // Bound the expanded tile count (see [`cap_repeat_tile`]): an
        // absurdly small tile over a large area would otherwise expand
        // to trillions of tiles in the backends.
        let (tw, th) = cap_repeat_tile((tw, th), area);
        self.list.push(DisplayItem::RepeatImage {
            area,
            image,
            tile: (tw, th),
            repeat,
            offset,
        });
        self
    }

    /// Append a CSS gradient over `rect` (the background painting area,
    /// i.e. the border box).
    ///
    /// `parsed` is resolved against `rect` via
    /// [`ParsedGradient::resolve`]; `current` resolves `currentcolor`
    /// stops. Degenerate input (empty/non-finite `rect`, or a gradient
    /// that resolves to nothing) emits nothing.
    pub fn push_gradient(
        &mut self,
        rect: Rect,
        parsed: &ParsedGradient,
        current: Color,
    ) -> &mut Self {
        if let Some(gradient) = parsed.resolve(&rect, current) {
            self.list.push(DisplayItem::Gradient { rect, gradient });
        }
        self
    }

    /// Parsed gradient for `source`, memoized by verbatim source string.
    ///
    /// Display-list builds call this once per box carrying a gradient
    /// `background-image`; a page with N boxes sharing one gradient
    /// class parses the source once per build instead of N times. Parse
    /// failures are memoized too, so a dropped declaration is not
    /// re-parsed per box either (it emits nothing either way).
    ///
    /// The result is [`Arc`](std::sync::Arc)-shared: a cache hit clones
    /// the `Arc` (a refcount bump), never the stop list. The returned
    /// value is never mutated by callers — they only resolve it against
    /// a rect via [`ParsedGradient::resolve`] — so sharing is sound.
    pub(crate) fn parsed_gradient(&mut self, source: &str) -> Option<Arc<ParsedGradient>> {
        if let Some(cached) = self.gradient_cache.get(source) {
            return cached.clone();
        }
        let parsed = parse_gradient(source).ok().map(Arc::new);
        self.gradient_cache.insert(source.into(), parsed.clone());
        parsed
    }

    /// Number of memoized gradient sources (test introspection).
    #[cfg(test)]
    pub(crate) fn gradient_cache_len(&self) -> usize {
        self.gradient_cache.len()
    }

    /// Resolved [`FilterOp`] chain for a `filter`/`backdrop-filter`
    /// value, memoized by canonical filter text plus the length context
    /// and `currentcolor` (see
    /// [`filter_memo_key`](crate::filter::filter_memo_key)).
    ///
    /// Display-list builds call this once per box carrying a filter; a
    /// page with N boxes sharing one filter class resolves the argument
    /// strings once per build instead of N times. `Filter::None` (and
    /// values whose key is `None`) resolve to empty without touching
    /// the cache.
    pub(crate) fn resolved_filter(
        &mut self,
        filter: &Filter,
        ctx: &LengthCtx,
        current: Color,
    ) -> FilterOps {
        let Some(key) = filter_memo_key(filter, ctx, current) else {
            return FilterOps::new();
        };
        if let Some(cached) = self.filter_cache.get(key.as_str()) {
            return cached.clone();
        }
        let ops = FilterOps::from(resolve_filter_chain(filter, ctx, current));
        self.filter_cache.insert(key.into(), ops.clone());
        ops
    }

    /// Append a text decoration over an ink box.
    pub fn push_decoration(
        &mut self,
        rect: Rect,
        kind: DecoKind,
        color: Color,
        thickness: f32,
    ) -> &mut Self {
        self.list.push(DisplayItem::Decoration {
            rect,
            kind,
            color,
            thickness,
        });
        self
    }

    /// Append a list item marker; `index` is the 1-based item number.
    /// `font_key` is the item's font, used for `Decimal`/
    /// `DecimalLeadingZero` numerals.
    pub fn push_list_marker(
        &mut self,
        rect: Rect,
        kind: MarkerKind,
        color: Color,
        index: u32,
        font_key: FontKey,
    ) -> &mut Self {
        self.list.push(DisplayItem::ListMarker {
            rect,
            kind,
            color,
            index,
            font_key,
        });
        self
    }

    /// Intersect `rect` with the current clip.
    pub fn push_clip(&mut self, rect: Rect) -> &mut Self {
        self.list.push(DisplayItem::PushClip { rect });
        self
    }

    /// Intersect a rounded rect with the current clip. `radii` is the
    /// outer border radius in px per corner, `[top-left, top-right,
    /// bottom-right, bottom-left]`, CSS-clamped to `rect` (see
    /// [`resolve_clip_radii`]).
    pub fn push_rounded_clip(&mut self, rect: Rect, radii: [f32; 4]) -> &mut Self {
        self.list.push(DisplayItem::PushRoundedClip { rect, radii });
        self
    }

    /// Pop the innermost clip.
    pub fn pop_clip(&mut self) -> &mut Self {
        self.list.push(DisplayItem::PopClip);
        self
    }

    /// Intersect a `clip-path` polygon with the current clip. `points`
    /// are absolute page-px vertices (see
    /// [`DisplayItem::PushClipPath`]); the caller resolved them via
    /// [`resolve_clip_path`](crate::filter::resolve_clip_path).
    pub fn push_clip_path(&mut self, points: ClipPoints) -> &mut Self {
        self.list.push(DisplayItem::PushClipPath { points });
        self
    }

    /// Multiply subsequent item alpha by `opacity` until [`pop_opacity`](Self::pop_opacity).
    pub fn push_opacity(&mut self, opacity: f32) -> &mut Self {
        self.list.push(DisplayItem::PushOpacity { opacity });
        self
    }

    /// Pop the innermost opacity group.
    pub fn pop_opacity(&mut self) -> &mut Self {
        self.list.push(DisplayItem::PopOpacity);
        self
    }

    /// Blend the following isolated group against the backdrop with
    /// `mode` until [`pop_blend`](Self::pop_blend). `Normal` is a no-op
    /// (backends paint the range inline), so callers should skip the
    /// push/pop pair entirely for it — see `css_blend_mode` in the box
    /// painter.
    pub fn push_blend(&mut self, mode: BlendMode) -> &mut Self {
        self.list.push(DisplayItem::PushBlend { mode });
        self
    }

    /// Pop the innermost blend group.
    pub fn pop_blend(&mut self) -> &mut Self {
        self.list.push(DisplayItem::PopBlend);
        self
    }

    /// Apply a resolved CSS `filter` chain to the following isolated
    /// group until [`pop_filter`](Self::pop_filter). Callers skip the
    /// push/pop pair entirely for empty chains (the no-filter fast
    /// path).
    pub fn push_filter(&mut self, ops: FilterOps) -> &mut Self {
        self.list.push(DisplayItem::PushFilter { ops });
        self
    }

    /// Pop the innermost filter group.
    pub fn pop_filter(&mut self) -> &mut Self {
        self.list.push(DisplayItem::PopFilter);
        self
    }

    /// Apply a resolved CSS `backdrop-filter` chain to the
    /// already-painted backdrop under `rect` (the element's border box)
    /// until [`pop_backdrop_filter`](Self::pop_backdrop_filter).
    /// Callers skip the push/pop pair entirely for empty chains.
    pub fn push_backdrop_filter(&mut self, ops: FilterOps, rect: Rect) -> &mut Self {
        self.list
            .push(DisplayItem::PushBackdropFilter { ops, rect });
        self
    }

    /// Pop the innermost backdrop-filter group.
    pub fn pop_backdrop_filter(&mut self) -> &mut Self {
        self.list.push(DisplayItem::PopBackdropFilter);
        self
    }

    /// Transform the following group's rendering by `matrix`
    /// (`(a, b, c, d, e, f)`, see [`DisplayItem::PushTransform`]) until
    /// [`pop_transform`](Self::pop_transform). Callers skip the
    /// push/pop pair for the identity matrix (backends paint it
    /// inline); see `css_transform_matrix` in the box painter.
    pub fn push_transform(&mut self, matrix: [f32; 6]) -> &mut Self {
        self.list.push(DisplayItem::PushTransform { matrix });
        self
    }

    /// Pop the innermost transform group.
    pub fn pop_transform(&mut self) -> &mut Self {
        self.list.push(DisplayItem::PopTransform);
        self
    }

    /// Paint one `box-shadow` (see [`DisplayItem::BoxShadow`]). The
    /// caller orders it: outset shadows before the background fill,
    /// inset shadows after the background but before borders/content.
    /// [`build_box`] emits these from the parsed `box-shadow` computed
    /// value ([`CssValue::Shadows`]); backends paint the items.
    #[allow(clippy::too_many_arguments)]
    pub fn push_box_shadow(
        &mut self,
        rect: Rect,
        radii: [f32; 4],
        color: Color,
        offset: (f32, f32),
        blur: f32,
        spread: f32,
        inset: bool,
    ) -> &mut Self {
        self.list.push(DisplayItem::BoxShadow {
            rect,
            radii,
            color,
            offset,
            blur,
            spread,
            inset,
        });
        self
    }

    /// Paint one `text-shadow` (see [`DisplayItem::TextShadow`]),
    /// underneath the text run it shadows. [`build_box`] emits these
    /// from the parsed `text-shadow` computed value
    /// ([`CssValue::Shadows`]).
    #[allow(clippy::too_many_arguments)]
    pub fn push_text_shadow(
        &mut self,
        text: Arc<str>,
        text_range: Range<usize>,
        rect: Rect,
        baseline: f32,
        font_key: FontKey,
        size_px: f32,
        color: Color,
        offset: (f32, f32),
        blur: f32,
    ) -> &mut Self {
        self.list.push(DisplayItem::TextShadow {
            text,
            text_range,
            rect,
            baseline,
            font_key,
            size_px,
            color,
            offset,
            blur,
        });
        self
    }

    /// Register a backend-generated image, returning its fresh id.
    pub fn add_image(&mut self, image: DecodedImage) -> ImageId {
        let id = ImageId(self.next_image_id);
        // Wrapping add keeps the id space valid forever; 0 stays reserved
        // for "no image".
        self.next_image_id = self.next_image_id.wrapping_add(1).max(1);
        self.list.add_image(id, image);
        id
    }

    /// Register externally-resolved image bytes under a caller-chosen id
    /// (e.g. from an image resolver).
    pub fn register_image(&mut self, id: ImageId, image: DecodedImage) -> &mut Self {
        self.list.add_image(id, image);
        self
    }

    /// Finish and return the display list.
    pub fn build(self) -> DisplayList {
        self.list
    }
}

impl Default for DisplayListBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Layout -> display list
// ---------------------------------------------------------------------------

fn css_color(style: &ComputedStyle, property: u8) -> Option<Color> {
    match style.get_by_index(property) {
        Some(CssValue::Color(c)) => Some(Color::rgba8(c.r, c.g, c.b, c.a)),
        _ => None,
    }
}

/// Resolve one shadow length (`box-shadow`/`text-shadow` offset, blur,
/// spread) to px. Shadows take no percentages; a deferred `calc()`
/// contributes only its absolute `px` coefficient when the percentage
/// coefficient is zero.
fn resolve_shadow_length(
    value: &CssValue,
    em_basis: f32,
    root_fs: f32,
    viewport: (f32, f32),
) -> f32 {
    match value {
        CssValue::Length(v, unit) => sane_len(length_to_px(
            *v, unit, em_basis, root_fs, viewport.0, viewport.1,
        )),
        CssValue::Number(v) => sane_len(*v),
        CssValue::MathDeferred { pct, px, .. } if *pct == 0.0 => sane_len(*px),
        _ => 0.0,
    }
}

/// Resolve a shadow color: `Color(..)` as-is; `currentcolor` (the CSS
/// initial when the color is omitted) against the element's `color`.
fn shadow_color(value: &CssValue, current: Color) -> Color {
    match value {
        CssValue::Color(c) => Color::rgba8(c.r, c.g, c.b, c.a),
        CssValue::Keyword(k) if k.eq_ignore_ascii_case("currentcolor") => current,
        _ => current,
    }
}

/// Outer corner radii in px, [top-left, top-right, bottom-right,
/// bottom-left]; 0 when unset (the CSS worker owns radius
/// parsing/shorthand expansion). Shared by the border painter and the
/// `box-shadow` emitter: the shadow follows the border box's curve.
fn css_border_radii(style: &ComputedStyle, font_size: f32, uctx: UnitCtx) -> [f32; 4] {
    [
        resolve_length(
            style.get_by_index(props::BORDER_TOP_LEFT_RADIUS),
            font_size,
            uctx.root_fs,
            uctx.viewport,
        ),
        resolve_length(
            style.get_by_index(props::BORDER_TOP_RIGHT_RADIUS),
            font_size,
            uctx.root_fs,
            uctx.viewport,
        ),
        resolve_length(
            style.get_by_index(props::BORDER_BOTTOM_RIGHT_RADIUS),
            font_size,
            uctx.root_fs,
            uctx.viewport,
        ),
        resolve_length(
            style.get_by_index(props::BORDER_BOTTOM_LEFT_RADIUS),
            font_size,
            uctx.root_fs,
            uctx.viewport,
        ),
    ]
}

/// Emit one `box-shadow` layer set from the box's own style (the
/// property is not inherited, so the caller passes the box's own
/// style). `inset` selects which layers to emit: the caller emits
/// outset layers below the background and inset layers above the
/// background but below the border (CSS Backgrounds 3 §7.2). `color`
/// is the element's `color`, the `currentcolor` basis. The shadow
/// follows the border box's corner radii. Layers list front-to-back;
/// display items paint back-to-front, so this emits in reverse.
#[allow(clippy::too_many_arguments)]
fn emit_box_shadows(
    b: &mut DisplayListBuilder,
    rect: Rect,
    own: &ComputedStyle,
    color: Color,
    font_size: f32,
    uctx: UnitCtx,
    inset: bool,
) {
    let Some(CssValue::Shadows(layers)) = own.get_by_index(props::BOX_SHADOW) else {
        return;
    };
    // Radii resolve lazily: most boxes have no shadow layers at all.
    let mut radii: Option<[f32; 4]> = None;
    for s in layers.iter().rev() {
        if s.inset != inset {
            continue;
        }
        let radii = *radii.get_or_insert_with(|| css_border_radii(own, font_size, uctx));
        b.push_box_shadow(
            rect,
            radii,
            shadow_color(&s.color, color),
            (
                resolve_shadow_length(&s.offset_x, font_size, uctx.root_fs, uctx.viewport),
                resolve_shadow_length(&s.offset_y, font_size, uctx.root_fs, uctx.viewport),
            ),
            resolve_shadow_length(&s.blur, font_size, uctx.root_fs, uctx.viewport).max(0.0),
            resolve_shadow_length(&s.spread, font_size, uctx.root_fs, uctx.viewport),
            s.inset,
        );
    }
}

/// Inherited context for resolving relative CSS units at paint time.
///
/// Layout resolves relative units while building boxes, but the
/// display-list builder re-reads a few raw computed values (border
/// widths, radii, `font-size`); those need the same bases layout used.
/// `font-size` resolves through [`bowser_layout::resolve_font_size`] and
/// lengths through [`bowser_css::length_to_px`], the canonical
/// implementations shared with layout and the CSS math evaluator.
#[derive(Clone, Copy)]
struct UnitCtx {
    /// Parent box's resolved font size in px: the basis for this box's
    /// own `font-size` when it uses `em`/`%`/`larger`/`smaller`.
    parent_fs: f32,
    /// Root element's resolved font size in px (`rem` basis).
    root_fs: f32,
    /// Viewport size in px (`vw`/`vh` basis).
    viewport: (f32, f32),
}

/// Resolve a general length property (border widths, radii) to px.
/// `em` resolves against the element's own computed font size per CSS;
/// `rem` against the root; `vw`/`vh` against the viewport. Unitless
/// numbers are treated as px, matching layout's leniency for them;
/// anything else resolves to 0.0.
///
/// The `LengthUnit -> px` factor table lives in
/// [`bowser_css::length_to_px`]; this only supplies paint's bases and
/// the number/zero fallbacks. The result goes through
/// [`bowser_layout::sane_len`], the same finite clamp layout applies, so
/// hostile values (`1e30px`) cannot paint at infinities layout never
/// produced.
///
/// Deferred math (`MathDeferred`) with a zero percentage coefficient —
/// the frozen base-0 form of non-affine `min()`/`max()`/`clamp()` with
/// `%` (see [`CssValue::MathDeferred`]) — contributes its constant,
/// mirroring [`resolve_shadow_length`]. A nonzero percentage
/// coefficient has no base in this path and still falls back to 0.0;
/// the basis-aware [`resolve_length_pct`] handles those where a
/// percentage base exists.
fn resolve_length(
    value: Option<&CssValue>,
    em_basis: f32,
    root_fs: f32,
    viewport: (f32, f32),
) -> f32 {
    match value {
        Some(CssValue::Length(v, unit)) => sane_len(length_to_px(
            *v, unit, em_basis, root_fs, viewport.0, viewport.1,
        )),
        Some(CssValue::Number(v)) => sane_len(*v),
        Some(CssValue::MathDeferred { pct, px, .. }) if *pct == 0.0 => sane_len(*px),
        _ => 0.0,
    }
}

/// Resolve one corner radius to px. Lengths and numbers go through
/// [`resolve_length`]; percentages resolve against `percent_basis`
/// (the border box's width — CSS resolves the horizontal radius
/// against the width, and the clip keeps one circular radius per
/// corner). Deferred math resolves its coefficients against the same
/// basis: the affine form keeps its percentage live, and the frozen
/// base-0 form (`pct == 0.0`) contributes its constant — so
/// `border-top-left-radius: max(10px, 5%)` paints at 10px, not 0.
/// Elliptical two-value radii (`CssValue::List`) are not
/// modeled: 0, i.e. a square corner, exactly like the border path.
fn resolve_radius(
    value: Option<&CssValue>,
    em_basis: f32,
    root_fs: f32,
    viewport: (f32, f32),
    percent_basis: f32,
) -> f32 {
    match value {
        Some(CssValue::Percentage(p)) => sane_len(p / 100.0 * percent_basis),
        Some(CssValue::MathDeferred { pct, px, .. }) => sane_len(pct / 100.0 * percent_basis + px),
        _ => resolve_length(value, em_basis, root_fs, viewport),
    }
}

/// Resolve the box's outer border radii (`border-top-left-radius` and
/// friends) to px, `[top-left, top-right, bottom-right, bottom-left]`,
/// with the CSS adjacent-sum clamp applied
/// ([`crate::borders::clamp_radii`]). The overflow clip uses exactly
/// these radii, so a rounded clip and the box's rounded border curve
/// agree. Percentages resolve against the border box's width (see
/// [`resolve_radius`]); note the border tessellator currently resolves
/// percentage radii to 0 via `resolve_length`, so for percentage radii
/// the clip is ahead of the border until that path learns percentages.
fn resolve_clip_radii(
    style: &ComputedStyle,
    font_size: f32,
    uctx: UnitCtx,
    rect: Rect,
) -> [f32; 4] {
    let radii = [
        resolve_radius(
            style.get_by_index(props::BORDER_TOP_LEFT_RADIUS),
            font_size,
            uctx.root_fs,
            uctx.viewport,
            rect.w,
        ),
        resolve_radius(
            style.get_by_index(props::BORDER_TOP_RIGHT_RADIUS),
            font_size,
            uctx.root_fs,
            uctx.viewport,
            rect.w,
        ),
        resolve_radius(
            style.get_by_index(props::BORDER_BOTTOM_RIGHT_RADIUS),
            font_size,
            uctx.root_fs,
            uctx.viewport,
            rect.w,
        ),
        resolve_radius(
            style.get_by_index(props::BORDER_BOTTOM_LEFT_RADIUS),
            font_size,
            uctx.root_fs,
            uctx.viewport,
            rect.w,
        ),
    ];
    crate::borders::clamp_radii(rect, radii)
}

fn css_opacity(style: &ComputedStyle) -> f32 {
    match style.get_by_index(props::OPACITY) {
        Some(CssValue::Number(v)) => *v,
        Some(CssValue::Percentage(v)) => *v / 100.0,
        _ => 1.0,
    }
    .clamp(0.0, 1.0)
}

/// The `mix-blend-mode` of a box: `Normal` when the property is absent
/// or holds an unsupported keyword (the cascade rejects those, so the
/// builder only sees `Normal` there).
fn css_blend_mode(style: &ComputedStyle) -> BlendMode {
    style
        .get_str_by_index(props::MIX_BLEND_MODE)
        .and_then(BlendMode::from_css_keyword)
        .unwrap_or(BlendMode::Normal)
}

/// Whether the box's `transform` is active (any value other than
/// `none`): per CSS Transforms 1 §6 such boxes create a stacking
/// context, even when the value evaluates to the identity matrix
/// (e.g. `translate(0)`).
fn css_transform_active(style: &ComputedStyle) -> bool {
    match style.get_by_index(props::TRANSFORM) {
        None => false,
        Some(CssValue::Keyword(k)) => !k.eq_ignore_ascii_case("none"),
        // `CssValue::Transform` (possibly an empty list) or anything
        // else the cascade kept: active unless it evaluates to `none`.
        // An empty transform list is the identity but still counts
        // (the specified value is not `none`).
        Some(_) => true,
    }
}

/// Evaluate the box's `transform` to a 2D matrix in page px, pivoted
/// around `transform-origin` (see
/// [`bowser_css::evaluate_transform`]). The reference box is the
/// border box (`transform-box` defaults to `view-box`, which is the
/// border box for HTML elements). Returns `None` when the property is
/// absent/`none`/unevaluable — the caller then emits no transform
/// group. Identity matrices are returned (not filtered): the caller
/// skips the push/pop pair for those, but they still count for
/// [`creates_stacking_context`].
fn css_transform_matrix(style: &ComputedStyle, border_box: Rect) -> Option<[f32; 6]> {
    let transform = style.get_by_index(props::TRANSFORM)?;
    if matches!(transform, CssValue::Keyword(k) if k.eq_ignore_ascii_case("none")) {
        return None;
    }
    let origin = style
        .get_by_index(props::TRANSFORM_ORIGIN)
        .and_then(TransformOrigin::parse)
        .unwrap_or_else(TransformOrigin::initial);
    let ctx = TransformContext {
        reference_width: border_box.w.max(0.0),
        reference_height: border_box.h.max(0.0),
        resolve_length: &resolve_absolute_length,
    };
    let m = evaluate_transform(transform, &origin, &ctx)?;
    if !m.a.is_finite()
        || !m.b.is_finite()
        || !m.c.is_finite()
        || !m.d.is_finite()
        || !m.e.is_finite()
        || !m.f.is_finite()
    {
        return None;
    }
    Some([m.a, m.b, m.c, m.d, m.e, m.f])
}

fn css_border_style(style: &ComputedStyle, property: u8) -> BorderStyle {
    match style.get_str_by_index(property) {
        Some("dotted") => BorderStyle::Dotted,
        Some("dashed") => BorderStyle::Dashed,
        Some("solid") => BorderStyle::Solid,
        Some("double") => BorderStyle::Double,
        Some("groove") => BorderStyle::Groove,
        Some("ridge") => BorderStyle::Ridge,
        Some("inset") => BorderStyle::Inset,
        Some("outset") => BorderStyle::Outset,
        Some("hidden") => BorderStyle::Hidden,
        _ => BorderStyle::None,
    }
}

// ---------------------------------------------------------------------------
// Column rules (`column-rule` on multicol containers)
// ---------------------------------------------------------------------------

/// Tolerance (px) for matching a drained [`ColumnRuleSpec`] against a
/// container's recomputed geometry: covers f32 summation dust from
/// re-resolving border/padding insets and column metrics at paint time.
const COLUMN_RULE_MATCH_EPS: f32 = 0.01;

/// Hard cap on the column count, mirroring `bowser_layout`'s
/// `MAX_COLUMNS` in `layout_multicol.rs`.
const MAX_COLUMN_RULE_COLUMNS: usize = 1024;

/// `column-rule` specs drained from layout, awaiting their container box.
///
/// [`build_display_list`] drains the layout-side table once per layout
/// pass ([`take_column_rules`]) and stashes the specs here; the box walk
/// claims each spec when it visits the spec's container (see
/// [`paint_column_rules`]).
#[derive(Default, Debug)]
struct ColumnRuleClaims {
    /// Specs naming a DOM node, keyed by it. The block-children
    /// distribution path records `dom_node`, so the claim is exact: the
    /// box generated for that node.
    by_node: HashMap<NodeId, ColumnRuleSpec>,
    /// Specs from the inline distribution path, which records no DOM
    /// node. Matched by geometry (see [`claim_orphan_column_rule`]).
    orphans: Vec<ColumnRuleSpec>,
}

impl ColumnRuleClaims {
    fn is_empty(&self) -> bool {
        self.by_node.is_empty() && self.orphans.is_empty()
    }

    /// Drop all specs, keeping the map/vec capacities: the per-build
    /// claims never carry over to the next build on a reused builder.
    fn clear(&mut self) {
        self.by_node.clear();
        self.orphans.clear();
    }

    fn insert_all(&mut self, specs: Vec<ColumnRuleSpec>) {
        for spec in specs {
            match spec.dom_node {
                Some(node) => {
                    self.by_node.insert(node, spec);
                }
                None => self.orphans.push(spec),
            }
        }
    }
}

/// `column-rule-style` keyword to [`ColumnRuleStyle`], mirroring
/// `bowser_layout`'s `resolve_column_rule`: `none`/`hidden`/unrecognized
/// draw nothing.
fn css_column_rule_style(style: &ComputedStyle) -> Option<ColumnRuleStyle> {
    // `get_str` preserves case; CSS keywords are ASCII case-insensitive,
    // so `eq_ignore_ascii_case` avoids the `to_ascii_lowercase` allocation.
    match style.get_str_by_index(props::COLUMN_RULE_STYLE)? {
        s if s.eq_ignore_ascii_case("solid") => Some(ColumnRuleStyle::Solid),
        s if s.eq_ignore_ascii_case("dashed") => Some(ColumnRuleStyle::Dashed),
        s if s.eq_ignore_ascii_case("dotted") => Some(ColumnRuleStyle::Dotted),
        s if s.eq_ignore_ascii_case("double") => Some(ColumnRuleStyle::Double),
        s if s.eq_ignore_ascii_case("groove") => Some(ColumnRuleStyle::Groove),
        s if s.eq_ignore_ascii_case("ridge") => Some(ColumnRuleStyle::Ridge),
        s if s.eq_ignore_ascii_case("inset") => Some(ColumnRuleStyle::Inset),
        s if s.eq_ignore_ascii_case("outset") => Some(ColumnRuleStyle::Outset),
        _ => None,
    }
}

/// [`resolve_length`] with percentage support: percentages resolve
/// against `pct_basis`; everything else behaves exactly like
/// [`resolve_length`] (including the [`sane_len`] clamp).
///
/// Deferred math evaluates its coefficients against the same basis —
/// the affine form keeps its percentage live, and the frozen base-0
/// form (`pct == 0.0`, non-affine `min()`/`max()`/`clamp()` with `%`)
/// contributes its constant — so `padding-top: max(10px, 0%)` paints
/// at 10px, not 0. Layout resolves the same value through its own
/// `SpecLen`; the paint-side basis (e.g. the border-box width for
/// padding) is the documented approximation.
fn resolve_length_pct(
    value: Option<&CssValue>,
    em_basis: f32,
    root_fs: f32,
    viewport: (f32, f32),
    pct_basis: f32,
) -> f32 {
    match value {
        Some(CssValue::Percentage(p)) => sane_len(p / 100.0 * pct_basis),
        Some(CssValue::MathDeferred { pct, px, .. }) => sane_len(pct / 100.0 * pct_basis + px),
        _ => resolve_length(value, em_basis, root_fs, viewport),
    }
}

/// This box's content box in absolute page px: the border box
/// (`box_.rect` — the origin `build_box` positions border items from)
/// inset by the used border widths and padding.
///
/// Border widths resolve exactly the way the border phase resolves them
/// ([`resolve_length`] on the same [`BORDER_WIDTH_PROPS`] longhands), so
/// the inset stays consistent with border positioning. Padding
/// percentages resolve against the border-box width as an approximation
/// of the containing-block width (exact for `px`/`em`/unitless padding).
fn content_box(box_: &LayoutBox, font_size: f32, uctx: UnitCtx) -> Rect {
    let (bl, br, bt, bb, pl, pr, pt, pb) = match box_.style.as_ref() {
        Some(s) => {
            let border = |prop: u8| {
                resolve_length(s.get_by_index(prop), font_size, uctx.root_fs, uctx.viewport)
            };
            // Percentages resolve against the containing-block width;
            // the border-box width is the paint-side approximation.
            let pad = |prop: u8| {
                resolve_length_pct(
                    s.get_by_index(prop),
                    font_size,
                    uctx.root_fs,
                    uctx.viewport,
                    box_.rect.w,
                )
            };
            (
                border(BORDER_WIDTH_PROPS[3]),
                border(BORDER_WIDTH_PROPS[1]),
                border(BORDER_WIDTH_PROPS[0]),
                border(BORDER_WIDTH_PROPS[2]),
                pad(props::PADDING_LEFT),
                pad(props::PADDING_RIGHT),
                pad(props::PADDING_TOP),
                pad(props::PADDING_BOTTOM),
            )
        }
        None => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    };
    let r = box_.rect;
    Rect {
        x: r.x + bl + pl,
        y: r.y + bt + pt,
        w: (r.w - bl - br - pl - pr).max(0.0),
        h: (r.h - bt - bb - pt - pb).max(0.0),
    }
}

/// `column-rule-color` resolution, mirroring `bowser_layout`'s
/// `resolve_rule_color`: a concrete color is used directly;
/// `currentcolor`/missing resolves to the used `color`; anything else
/// falls back to opaque black.
fn column_rule_color(style: &ComputedStyle) -> bowser_css::Color {
    const BLACK: bowser_css::Color = bowser_css::Color {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };
    let text_color = match style.get_by_index(props::COLOR) {
        Some(CssValue::Color(c)) => *c,
        _ => BLACK,
    };
    match style.get_by_index(props::COLUMN_RULE_COLOR) {
        Some(CssValue::Color(c)) => *c,
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("currentcolor") => text_color,
        // Missing → initial `currentcolor`.
        None => text_color,
        _ => BLACK,
    }
}

/// Resolved `column-rule` appearance for a candidate container —
/// `(style, width_px, color)` — or `None` when nothing draws. Mirrors
/// `bowser_layout`'s `resolve_column_rule`: width keywords follow the
/// CSS 2.1 §8.5.3 UA convention (`thin`/`medium`/`thick` → 1/3/5px).
fn css_column_rule(
    style: &ComputedStyle,
    font_size: f32,
    uctx: UnitCtx,
    content_w: f32,
) -> Option<(ColumnRuleStyle, f32, bowser_css::Color)> {
    let rule_style = css_column_rule_style(style)?;
    let width = match style.get_by_index(props::COLUMN_RULE_WIDTH) {
        Some(CssValue::Keyword(k)) => {
            if k.eq_ignore_ascii_case("thin") {
                1.0
            } else if k.eq_ignore_ascii_case("thick") {
                5.0
            } else {
                // "medium" (the initial) and unrecognized keywords.
                3.0
            }
        }
        Some(v) => {
            resolve_length_pct(Some(v), font_size, uctx.root_fs, uctx.viewport, content_w).max(0.0)
        }
        // Missing → initial `medium`.
        None => 3.0,
    };
    if width <= 0.0 {
        return None;
    }
    Some((rule_style, width, column_rule_color(style)))
}

/// Recompute a multicol container's `(count, gap, col_w)` from its style
/// and content width, mirroring `bowser_layout`'s `multicol_params`
/// (CSS Multicol 1 §3.4). Returns `None` when the box is not a multicol
/// container. `column-gap: normal` is 1em; a non-positive `column-width`
/// counts as `auto`.
fn multicol_column_geometry(
    style: &ComputedStyle,
    font_size: f32,
    uctx: UnitCtx,
    content_w: f32,
) -> Option<(usize, f32, f32)> {
    // `column-count: <integer> | auto`.
    let count = match style.get_by_index(props::COLUMN_COUNT) {
        Some(CssValue::Number(n)) if n.is_finite() && *n >= 1.0 => {
            Some((*n as usize).clamp(1, MAX_COLUMN_RULE_COLUMNS))
        }
        _ => None,
    };
    // `column-width: <length> | auto`.
    let width = match style.get_by_index(props::COLUMN_WIDTH) {
        Some(v) => {
            let w = resolve_length_pct(Some(v), font_size, uctx.root_fs, uctx.viewport, content_w);
            if w > 0.0 { Some(w) } else { None }
        }
        None => None,
    };
    // `column-gap: <length-percentage> | normal`.
    let gap = match style.get_by_index(props::COLUMN_GAP) {
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("normal") => font_size,
        // Unrecognized keywords and missing → initial `normal` → 1em.
        Some(CssValue::Keyword(_)) | None => font_size,
        Some(v) => {
            resolve_length_pct(Some(v), font_size, uctx.root_fs, uctx.viewport, content_w).max(0.0)
        }
    };
    // CSS Multicol 1 §3.4.
    let derived = |w: f32| ((content_w + gap) / (w + gap)).floor().max(1.0) as usize;
    let mut n = match (count, width) {
        (Some(c), None) => c,
        (None, Some(w)) => derived(w),
        (Some(c), Some(w)) => c.min(derived(w)),
        (None, None) => return None,
    };
    n = n.clamp(1, MAX_COLUMN_RULE_COLUMNS);
    if n < 2 {
        return None;
    }
    let col_w = ((content_w - (n as f32 - 1.0) * gap) / n as f32).max(0.0);
    Some((n, gap, col_w))
}

/// Remove and return the orphan spec matching `box_`'s recomputed rule
/// geometry, if any: the documented fallback for specs recorded by the
/// inline distribution path, which carries no DOM node. Recomputes the
/// expected rule offsets/height for this rule-bearing multicol
/// container (mirroring layout's `multicol_params` /
/// `resolve_column_rule`) and takes the spec with equal `x_offsets` /
/// `height` — plus an equal resolved rule, so two same-geometry
/// containers with different rules cannot cross-claim. Specs with
/// identical geometry are interchangeable; the first match wins.
fn claim_orphan_column_rule(
    claims: &mut ColumnRuleClaims,
    box_: &LayoutBox,
    font_size: f32,
    uctx: UnitCtx,
) -> Option<ColumnRuleSpec> {
    if claims.orphans.is_empty() {
        return None;
    }
    let own = box_.style.as_ref()?;
    let content = content_box(box_, font_size, uctx);
    let (rule_style, rule_width, rule_color) = css_column_rule(own, font_size, uctx, content.w)?;
    let (count, gap, col_w) = multicol_column_geometry(own, font_size, uctx, content.w)?;
    // Gap centers, relative to the content-box left edge — the same
    // formula layout's `rule_x_offsets` uses.
    let expected: Vec<f32> = (0..count - 1)
        .map(|k| col_w + gap / 2.0 + k as f32 * (col_w + gap))
        .collect();
    let pos = claims.orphans.iter().position(|s| {
        s.style == rule_style
            && (s.width - rule_width).abs() <= COLUMN_RULE_MATCH_EPS
            && s.color == rule_color
            && (s.height - content.h).abs() <= COLUMN_RULE_MATCH_EPS
            && s.x_offsets.len() == expected.len()
            && s.x_offsets
                .iter()
                .zip(expected.iter())
                .all(|(a, b)| (a - b).abs() <= COLUMN_RULE_MATCH_EPS)
    });
    pos.map(|i| claims.orphans.remove(i))
}

/// Emit `column-rule` display items for `box_` when a drained
/// [`ColumnRuleSpec`] names it as its container.
///
/// Called from `build_box` in the container's border phase (right after
/// its own border), so rules paint above the background like borders do.
/// Each rule becomes one [`DisplayItem::Border`] carrying only a left
/// side: the border tessellator's vertical bar is exactly the rule's
/// line/rect, so every [`ColumnRuleStyle`] reuses the existing
/// dash/dot/double/groove machinery in [`crate::borders`] with no new
/// tessellation code.
///
/// The spec's `x_offsets` are relative to the content-box left edge;
/// they are translated by the content-box origin (the border-box origin
/// border items are positioned from, inset by border and padding — see
/// [`content_box`]) and each rule is centered on its gap center.
///
/// Defensive checks: layout already filters `none`/`hidden`/zero-width
/// rules, but a stale or foreign spec with non-finite or non-positive
/// `width`/`height` paints nothing here.
fn paint_column_rules(b: &mut DisplayListBuilder, box_: &LayoutBox, font_size: f32, uctx: UnitCtx) {
    if b.column_rules.is_empty() {
        return;
    }
    let spec = match box_.node.and_then(|n| b.column_rules.by_node.remove(&n)) {
        Some(s) => s,
        None => match claim_orphan_column_rule(&mut b.column_rules, box_, font_size, uctx) {
            Some(s) => s,
            None => return,
        },
    };
    if !spec.width.is_finite() || spec.width <= 0.0 {
        return;
    }
    if !spec.height.is_finite() || spec.height <= 0.0 {
        return;
    }
    if spec.x_offsets.is_empty() {
        return;
    }
    let border_style = match spec.style {
        ColumnRuleStyle::Solid => BorderStyle::Solid,
        ColumnRuleStyle::Dashed => BorderStyle::Dashed,
        ColumnRuleStyle::Dotted => BorderStyle::Dotted,
        ColumnRuleStyle::Double => BorderStyle::Double,
        ColumnRuleStyle::Groove => BorderStyle::Groove,
        ColumnRuleStyle::Ridge => BorderStyle::Ridge,
        ColumnRuleStyle::Inset => BorderStyle::Inset,
        ColumnRuleStyle::Outset => BorderStyle::Outset,
    };
    let rule_color = Color::rgba8(spec.color.r, spec.color.g, spec.color.b, spec.color.a);
    let content = content_box(box_, font_size, uctx);
    let half = spec.width / 2.0;
    for &x in &spec.x_offsets {
        if !x.is_finite() {
            continue;
        }
        // Center the rule on the gap center: the tessellator draws the
        // left side's bar exactly over `rect`.
        let rect = Rect {
            x: content.x + x - half,
            y: content.y,
            w: spec.width,
            h: spec.height,
        };
        b.push_border(
            rect,
            [0.0, 0.0, 0.0, spec.width],
            [
                BorderStyle::None,
                BorderStyle::None,
                BorderStyle::None,
                border_style,
            ],
            [
                Color::TRANSPARENT,
                Color::TRANSPARENT,
                Color::TRANSPARENT,
                rule_color,
            ],
            [0.0; 4],
        );
    }
}

/// `background-repeat` as a [`BackgroundRepeat`]. Two-keyword forms
/// (`repeat space`, etc.) are not representable in the single-axis enum
/// and fall back to [`BackgroundRepeat::Repeat`], as does anything
/// unrecognized.
fn css_background_repeat(style: &ComputedStyle) -> BackgroundRepeat {
    match style.get_str_by_index(props::BACKGROUND_REPEAT) {
        Some("no-repeat") => BackgroundRepeat::NoRepeat,
        Some("repeat-x") => BackgroundRepeat::RepeatX,
        Some("repeat-y") => BackgroundRepeat::RepeatY,
        Some("space") => BackgroundRepeat::Space,
        Some("round") => BackgroundRepeat::Round,
        _ => BackgroundRepeat::Repeat,
    }
}

/// Rescaled tile length for `background-repeat: round`: the length that
/// lets a whole number of tiles span the axis exactly
/// (`area_d / max(1, round(area_d / tile_d))`).
pub(crate) fn round_tile_len(area_d: f32, tile_d: f32) -> f32 {
    area_d / (area_d / tile_d).round().max(1.0)
}

/// Resolve one `background-position` component to a px offset of the
/// tile-grid origin within the area.
///
/// Percentages resolve against the slack `(area - tile)` per CSS
/// (`p%` of the leftover space); lengths resolve through
/// [`bowser_css::length_to_px`] against the same bases
/// [`resolve_length`] uses (`em` against the element's own font size,
/// `rem` against root, `vw`/`vh` against the viewport), clamped by
/// [`bowser_layout::sane_len`] like every other resolved length;
/// unitless numbers are px, matching layout's leniency; keywords map to
/// 0/50/100%.
fn background_position_component(
    value: &CssValue,
    area: f32,
    tile: f32,
    is_x: bool,
    em_basis: f32,
    root_fs: f32,
    viewport: (f32, f32),
) -> f32 {
    match value {
        CssValue::Percentage(p) => (area - tile) * p / 100.0,
        CssValue::Length(v, unit) => sane_len(length_to_px(
            *v, unit, em_basis, root_fs, viewport.0, viewport.1,
        )),
        CssValue::Number(v) => sane_len(*v),
        CssValue::Keyword(k) => {
            let pct = match (k.as_str(), is_x) {
                ("left", true) | ("top", false) => 0.0,
                ("center", _) => 50.0,
                ("right", true) | ("bottom", false) => 100.0,
                _ => 0.0,
            };
            (area - tile) * pct / 100.0
        }
        _ => 0.0,
    }
}

/// `background-position` as the (x, y) px offset of the tile-grid origin
/// within `area`, for a tile of native size `tile`.
///
/// Handles the shapes the CSS parser produces: `List([x, y])`, a single
/// value (y defaults to `center`; a lone `top`/`bottom` is vertical with
/// x defaulting to `center`), and anything else as `0 0`.
fn css_background_position(
    style: &ComputedStyle,
    area: Rect,
    tile: (f32, f32),
    em_basis: f32,
    root_fs: f32,
    viewport: (f32, f32),
) -> (f32, f32) {
    let center = CssValue::Percentage(50.0);
    // Unset properties read as None on this path (no initial-value
    // substitution); the CSS initial for background-position is 0% 0%.
    let zero = CssValue::Percentage(0.0);
    let (x, y) = match style.get_by_index(props::BACKGROUND_POSITION) {
        Some(CssValue::List(items)) => {
            let x = items.first().unwrap_or(&zero);
            // A lone x value leaves y at `center` per CSS.
            let y = items.get(1).unwrap_or(&center);
            (x, y)
        }
        // Single keyword: `center` is horizontal (y then defaults to
        // center); a lone `top`/`bottom` is vertical per CSS.
        Some(v @ CssValue::Keyword(k)) if k == "top" || k == "bottom" => (&center, v),
        Some(v) => (v, &center),
        None => (&zero, &zero),
    };
    (
        background_position_component(x, area.w, tile.0, true, em_basis, root_fs, viewport),
        background_position_component(y, area.h, tile.1, false, em_basis, root_fs, viewport),
    )
}

/// True when the box clips its children: any non-`visible` overflow
/// (`hidden`, `clip`, `scroll`, `auto`) clips per CSS.
///
/// Checks the `overflow` shorthand first, then the per-axis longhands; any
/// clipping axis clips. (Scrolling itself is backend-side and
/// unimplemented here, but the clip still applies: scrolled-out content
/// must not paint.)
fn overflow_clips(style: &ComputedStyle) -> bool {
    let clips = |v: Option<&str>| {
        matches!(
            v,
            Some("hidden") | Some("clip") | Some("scroll") | Some("auto")
        )
    };
    clips(style.get_str_by_index(props::OVERFLOW))
        || clips(style.get_str_by_index(props::OVERFLOW_X))
        || clips(style.get_str_by_index(props::OVERFLOW_Y))
}

/// The computed `position` keyword as an atom: one `&str` -> enum
/// conversion per box per display-list build, after which the call sites
/// below compare variants (a single integer compare) instead of
/// string-matching the keyword per box.
///
/// Preserves the exact case semantics of the old string checks:
/// `is_out_of_flow` and the `== "static"` / `== "fixed"` call sites match
/// the computed keyword case-sensitively, while `is_positioned` treats
/// `"static"` ASCII-case-insensitively (and a missing `position` reads as
/// `static`).
///
/// The cascade ASCII-lowercases idents when it parses values into
/// `CssValue::Keyword`, so any `position` keyword that went through
/// `compute_style` is already lowercase: `StaticCaseVariant` is
/// unreachable through the real pipeline and exists only as defense for
/// non-cascade-constructed styles (e.g. a `ComputedStyle` built directly
/// in a test).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PositionAtom {
    /// Missing `position`, or the exact keyword `static`.
    Static,
    /// A case-variant spelling of `static` (`STATIC`, `Static`): not
    /// positioned, but not byte-equal to `"static"`.
    StaticCaseVariant,
    /// Exact keyword `absolute`.
    Absolute,
    /// Exact keyword `fixed`.
    Fixed,
    /// Any other keyword (`relative`, `sticky`, unrecognized) or a
    /// non-string value.
    Other,
}

/// This box's own `position` value as an atom, `Static` when it has no
/// style. The `get_str_by_index(props::POSITION)` slot read is the wave-38
/// atom lookup (no property-name hashing); the match here converts the
/// keyword once so the six call sites below pay only integer compares.
fn box_position(box_: &LayoutBox) -> PositionAtom {
    match box_
        .style
        .as_ref()
        .and_then(|s| s.get_str_by_index(props::POSITION))
    {
        None | Some("static") => PositionAtom::Static,
        Some("absolute") => PositionAtom::Absolute,
        Some("fixed") => PositionAtom::Fixed,
        Some(kw) if kw.eq_ignore_ascii_case("static") => PositionAtom::StaticCaseVariant,
        _ => PositionAtom::Other,
    }
}

/// Whether the box is out of flow (`position: absolute | fixed`).
fn is_out_of_flow(box_: &LayoutBox) -> bool {
    matches!(
        box_position(box_),
        PositionAtom::Absolute | PositionAtom::Fixed
    )
}

/// Whether a table cell is "empty" for CSS 2.1 §17.5.1 `empty-cells`: it
/// has no in-flow content. Whitespace-only text is empty (matching
/// browsers — layout already drops most of it, but `white-space: pre`
/// can preserve it); anonymous wrappers are box-construction artifacts,
/// not content; absolutely/fixed positioned subtrees are out of flow
/// and don't count; any other in-flow box — an element (even an empty
/// `<div></div>`), a replaced element (`<img>`), a float, a nested
/// table — is content. Iterative: the box tree can nest arbitrarily
/// deep.
fn table_cell_is_empty(cell: &LayoutBox) -> bool {
    let mut stack: Vec<&LayoutBox> = cell.children.iter().collect();
    while let Some(b) = stack.pop() {
        // Out-of-flow subtrees are not in-flow content.
        if is_out_of_flow(b) {
            continue;
        }
        match b.box_type {
            BoxType::Text => {
                if b.text
                    .as_deref()
                    .is_some_and(|t| t.chars().any(|c| !c.is_whitespace()))
                {
                    return false;
                }
            }
            // Transparent wrapper: look through it.
            BoxType::AnonymousBlock => stack.extend(b.children.iter()),
            // Any other in-flow box is content.
            _ => return false,
        }
    }
    true
}

/// Whether the box is positioned for paint order (CSS 2.1 Appendix E):
/// anything but `static`. Missing/unrecognized computes to `static`,
/// matching layout's `position_of`.
fn is_positioned(box_: &LayoutBox) -> bool {
    !matches!(
        box_position(box_),
        PositionAtom::Static | PositionAtom::StaticCaseVariant
    )
}

/// Whether the box is a non-positioned float (Appendix E step 4).
/// `position: absolute | fixed` forces `float: none` (CSS 2.1 §9.7), and
/// relative/sticky boxes are positioned, so only static boxes float.
/// `float` is ignored on flex/grid items (flexbox §4, css-grid-1 §5.4 —
/// layout positions them as items regardless of the property), so a
/// static item of a flex/grid container never joins the float bucket;
/// it paints at step 3/5 by its block/inline level like any other
/// in-flow child.
fn is_float(box_: &LayoutBox, parent: &LayoutBox) -> bool {
    if is_positioned(box_) || is_flex_or_grid_container(parent) {
        return false;
    }
    matches!(
        box_.style.as_ref().and_then(|s| s.get_str_by_index(props::FLOAT)),
        Some(f) if f.eq_ignore_ascii_case("left") || f.eq_ignore_ascii_case("right")
    )
}

/// Whether the box is block-level for Appendix E steps 3 vs 5.
/// Table-internal boxes and grid containers are block-level;
/// `Inline`, `InlineBlock`, and `Text` are inline-level.
fn is_block_level(box_: &LayoutBox) -> bool {
    matches!(
        box_.box_type,
        BoxType::Block
            | BoxType::AnonymousBlock
            | BoxType::Table
            | BoxType::TableRowGroup
            | BoxType::TableRow
            | BoxType::TableCell
            | BoxType::TableColumn
            | BoxType::TableColumnGroup
            | BoxType::Grid
    )
}

/// Resolved `order` (`0` when unset): flexbox §5.4 / css-grid-1 §4.2.
/// Layout sorts flex/grid items by this for layout; paint re-sorts by it
/// for paint order ("as if the document were reordered"). Read from the
/// box's own computed style — layout does not carry `order` on
/// `LayoutBox` — with the same saturating float→int conversion layout
/// uses.
fn order_of(box_: &LayoutBox) -> i32 {
    match box_
        .style
        .as_ref()
        .and_then(|s| s.get_by_index(props::ORDER))
    {
        Some(CssValue::Number(n)) => {
            let v = *n as i64;
            i32::try_from(v).unwrap_or(if v < 0 { i32::MIN } else { i32::MAX })
        }
        _ => 0,
    }
}

/// Whether `order` re-sorts this box's children's paint order: a flex
/// (`flex`/`inline-flex`) or grid container. Layout buckets `flex` into
/// `BoxType::Block`, so the raw `display` value is consulted, mirroring
/// layout's `flex_kind`; `BoxType::Grid` covers block-level grids.
fn is_flex_or_grid_container(box_: &LayoutBox) -> bool {
    if box_.box_type == BoxType::Grid {
        return true;
    }
    matches!(
        box_.style.as_ref().and_then(|s| s.get_str_by_index(props::DISPLAY)),
        Some(d)
            if d.eq_ignore_ascii_case("flex")
                || d.eq_ignore_ascii_case("inline-flex")
                || d.eq_ignore_ascii_case("grid")
                || d.eq_ignore_ascii_case("inline-grid")
    )
}

/// Whether the box creates a stacking context *without* being positioned:
/// `opacity < 1` (css-color-3 §3.2), a non-`normal` `mix-blend-mode`
/// (css-compositing-1 §4), a non-`none` `transform` (css-transforms-1
/// §6), or a non-`none` `filter` / `backdrop-filter` (filter-effects-1
/// §8 / filter-effects-2 §2). Such boxes paint at Appendix E step 6
/// (stack level 0, tree order), not at steps 3/5. `isolation`,
/// `perspective`, `mask`, and `will-change` also create stacking
/// contexts per spec but are not supported by this codebase, so these
/// five are the complete set here. Uses the box's own style only: none
/// of the five properties is inherited (the opacity/blend/transform/
/// filter groups in `build_box` already isolate the whole subtree).
fn creates_stacking_context(box_: &LayoutBox) -> bool {
    match box_.style.as_ref() {
        Some(s) => {
            css_opacity(s) < 1.0
                || css_blend_mode(s) != BlendMode::Normal
                || css_transform_active(s)
                || css_filter_active(s)
        }
        None => false,
    }
}

/// Whether the style carries a non-`none` `filter` or `backdrop-filter`
/// (see [`creates_stacking_context`]). A `Filter::List` is always
/// non-empty (the parser rejects empty function lists), but an empty
/// list counts as `none` defensively.
fn css_filter_active(style: &ComputedStyle) -> bool {
    for prop in [props::FILTER, props::BACKDROP_FILTER] {
        if let Some(CssValue::Filter(Filter::List(fns))) = style.get_by_index(prop)
            && !fns.is_empty()
        {
            return true;
        }
    }
    false
}

/// Whether the box is a stacking context for Appendix E paint ordering:
/// [`creates_stacking_context`] (opacity/blend/transform), or any non-`auto`
/// `z-index`. `LayoutBox::z_index` is `Some` exactly for positioned boxes
/// with non-`auto` z-index and for static flex/grid items with non-`auto`
/// z-index (layout preserves it for those) — i.e. exactly the boxes that
/// form child stacking contexts in their parent's z-index steps.
///
/// Shared with `bowser_js`'s hit-testing (`elements_at_point`), which must
/// order negative-z-index children against the parent's own background the
/// same way the painter does.
pub fn is_stacking_context(box_: &LayoutBox) -> bool {
    creates_stacking_context(box_) || box_.z_index.is_some()
}

/// Whether the child joins the Appendix E z-index steps (ranks 0/4/5):
/// positioned boxes, non-positioned stacking-context creators (see
/// [`creates_stacking_context`]), and static flex/grid items with
/// non-`auto` `z-index` (layout preserves `z_index` for those; it is
/// `None` for every other static box).
fn joins_z_steps(box_: &LayoutBox) -> bool {
    is_positioned(box_) || box_.z_index.is_some() || creates_stacking_context(box_)
}

/// Document-order rank of a box that joins the z-index steps, from the
/// up-front discovery pass. Z-index ties paint in tree order (CSS 2.1
/// Appendix E); only step-joining boxes are ranked (they are the only
/// ones that merge with drained out-of-flow runs in z-index steps), so
/// the map stays empty (zero allocation) for ordinary trees. A missing
/// entry means "does not join the steps" and sorts last — harmless,
/// because every other entry in the same step is then also unranked and
/// the stable sort keeps their insertion (document) order.
fn doc_seq(box_: &LayoutBox, doc_order: &FnvHashMap<*const LayoutBox, usize>) -> usize {
    doc_order
        .get(&(box_ as *const LayoutBox))
        .copied()
        .unwrap_or(usize::MAX)
}

/// One child subtree scheduled at a CSS 2.1 Appendix E paint step.
#[derive(Clone)]
enum StepChild<'a> {
    /// In-flow child, painted with the parent's inheritance context.
    InFlow { box_: &'a LayoutBox, uctx: UnitCtx },
    /// Drained out-of-flow descendant: the up-front discovery pass fixed
    /// its inheritance context and containing-block clip anchoring.
    Oof(PendingOof<'a>),
}

impl<'a> StepChild<'a> {
    fn box_(&self) -> &'a LayoutBox {
        match self {
            StepChild::InFlow { box_, .. } => box_,
            StepChild::Oof(p) => p.box_,
        }
    }

    /// Ascending paint key within a step: z-index (`auto` sorts as 0),
    /// then `order` (flex/grid in-flow items only — out-of-flow boxes are
    /// not flex/grid items), then document order.
    fn sort_key(
        &self,
        order_items: bool,
        doc_order: &FnvHashMap<*const LayoutBox, usize>,
    ) -> (i32, i32, usize) {
        let b = self.box_();
        let z = b.z_index.unwrap_or(0);
        let ord = if order_items && matches!(self, StepChild::InFlow { .. }) {
            order_of(b)
        } else {
            0
        };
        (z, ord, doc_seq(b, doc_order))
    }
}

/// Per-build scratch for Appendix E step partitioning in [`build_box`]
/// (mirrors `LayoutScratch` in `bowser_layout`, threaded the same
/// way): one `(rank, child)` Vec reused by every stacking context in
/// the build. A context tags its in-flow children with their Appendix E
/// paint rank on classification, the pending out-of-flow drains append
/// directly, and the new tail is sorted in place — replacing the eight
/// per-context Vecs (`blocks`/`floats`/`inlines`/`positioned`,
/// `neg`/`zero`/`pos`, and the merged `paint_order` second copy), plus
/// the six per-context drain Vecs. Nested stacking contexts push their
/// own tail after the parent's range and pop it back on return, so
/// ranges never overlap; after emission the Vec is empty and its
/// capacity serves the next context for free.
struct StepScratch<'a> {
    steps: Vec<(u8, StepChild<'a>)>,
}

impl<'a> StepScratch<'a> {
    fn new() -> Self {
        StepScratch { steps: Vec::new() }
    }
}

/// Paint one [`StepChild`]: in-flow children inherit this box's context;
/// drained out-of-flow boxes use their deferral-site context and paint
/// with themselves as the nearest positioned ancestor (they are
/// positioned — same as the old drain call).
#[allow(clippy::too_many_arguments)]
fn paint_step_child<'a>(
    b: &mut DisplayListBuilder,
    scratch: &mut StepScratch<'a>,
    sc: StepChild<'a>,
    doc: &Document,
    font_resolver: &mut dyn FnMut(&ComputedStyle) -> FontKey,
    image_resolver: &mut dyn FnMut(&str) -> Option<(ImageId, DecodedImage)>,
    parent_style: Option<&'a ComputedStyle>,
    in_collapse: bool,
    child_npa: Option<*const LayoutBox>,
    pending: &mut PendingOofList<'a>,
    doc_order: &FnvHashMap<*const LayoutBox, usize>,
) {
    match sc {
        StepChild::InFlow { box_, uctx } => build_box(
            b,
            scratch,
            box_,
            doc,
            font_resolver,
            image_resolver,
            parent_style,
            uctx,
            in_collapse,
            child_npa,
            pending,
            false,
            doc_order,
        ),
        StepChild::Oof(p) => build_box(
            b,
            scratch,
            p.box_,
            doc,
            font_resolver,
            image_resolver,
            p.parent_style,
            p.uctx,
            in_collapse,
            Some(p.box_ as *const LayoutBox),
            pending,
            false,
            doc_order,
        ),
    }
}

/// Up-front out-of-flow discovery (CSS 2.1 §11.1.1 anchoring).
///
/// Runs once over the whole tree before any paint item is emitted,
/// deferring every absolutely/fixed-positioned box into `pending`.
/// Paint time only *drains* `pending` into Appendix E z-index steps, so
/// every drain sees the complete set — including negative-z boxes
/// nested deep inside in-flow subtrees, which must paint at step 2,
/// before in-flow content, and therefore cannot be discovered lazily
/// while painting.
///
/// Drain anchoring: each entry anchors at its containing block, except
/// when a stacking context (see [`is_stacking_context`]) sits below
/// (deeper than) the containing block — then it anchors at the nearest
/// such context. CSS 2.1 Appendix E paints an out-of-flow box inside its
/// nearest ancestor stacking context, so an `absolute` box inside a
/// static `opacity < 1` / non-`normal` blend box drains inside that box's
/// group instead of escaping to an outer containing block's drain, where
/// it would paint unfaded/unblended. A z-indexed out-of-flow box whose
/// containing block is not a stacking context is hoisted the rest of the
/// way: it anchors at the nearest ancestor stacking context even when
/// that sits at or above the containing block (Appendix E — a positioned
/// box with non-`auto` z-index forms a stacking context painted in its
/// nearest ancestor stacking context, not in its containing block's
/// scope). The containing block's own overflow clip travels with the
/// entry ([`PendingOof::outer_clips`]).
///
/// In-flow positioned boxes with non-`auto` z-index are hoisted the same
/// way: when their parent box is not a stacking context they are
/// deferred here (anchored at the nearest ancestor stacking context, or
/// the root when there is none) instead of joining the parent's z-index
/// steps, and skipped by [`build_box`]'s child classification. Their
/// ancestor overflow clips below the drain anchor travel in
/// [`PendingOof::outer_clips`]; in-flow content stays clipped by every
/// ancestor's overflow, anchor included (`escape_anchor_clip` is false
/// for these).
///
/// The traversal mirrors [`build_box`]'s context computation exactly
/// (inherited style, `visibility: hidden` subtree skipping, `UnitCtx`,
/// nearest positioned ancestor): all are pure functions of the ancestor
/// chain, so deferral-site values match what the paint walk would have
/// captured. `nsc` is the nearest ancestor stacking context strictly
/// below the nearest positioned ancestor (`npa`) — crossing the
/// containing block resets it; `nsc_vp` is the nearest ancestor stacking
/// context with no reset, serving `position: fixed` boxes whose
/// containing block is the viewport (every ancestor stacking context
/// sits below it). `nsc_full` is the nearest [`is_stacking_context`]
/// ancestor with no reset — the hoist target for z-indexed boxes,
/// which, unlike `nsc`/`nsc_vp`, also sees positioned boxes with
/// non-`auto` z-index as stacking contexts. `clip_stack` accumulates the
/// overflow-clip frames of the current ancestor chain (outermost first;
/// the root's frame is never pushed — see [`ClipFrame`]) so hoisted
/// entries can capture the clips they are lifted out of. `ancestors`
/// links every non-root ancestor frame (innermost first — see
/// [`AncestorFrame`]) so an escaping entry can record the
/// overflow-clipping boxes strictly between its drain anchor and its
/// containing block (CSS 2.1 §11.1.1 — see [`PendingOofList`]). The
/// root is excluded, like `clip_stack`: its overflow is the viewport's,
/// i.e. the containing block's own for viewport-anchored boxes.
/// `doc_order`
/// records document-order ranks, but only for boxes that join the z-index
/// steps (see [`joins_z_steps`]): they are the only ones that merge with
/// drained out-of-flow runs inside z-index steps, so the map stays empty
/// (zero allocation) for ordinary trees.
// Thirteen parameters: the discovery walk mirrors `build_box`'s context,
// so its ancestor state travels as arguments (precedent: `#[allow]`s in
// `bowser_net` and `bowser_layout` for the same reason).
#[allow(clippy::too_many_arguments)]
fn discover_oof<'a, 'c>(
    box_: &'a LayoutBox,
    parent_style: Option<&'a ComputedStyle>,
    uctx: UnitCtx,
    npa: Option<*const LayoutBox>,
    nsc: Option<*const LayoutBox>,
    nsc_vp: Option<*const LayoutBox>,
    pending: &mut PendingOofList<'a>,
    doc_order: &mut FnvHashMap<*const LayoutBox, usize>,
    seq: &mut usize,
    nsc_full: Option<*const LayoutBox>,
    clip_stack: &mut Vec<ClipFrame>,
    is_root: bool,
    ancestors: Option<&'c AncestorFrame<'c>>,
) {
    let self_ptr = box_ as *const LayoutBox;
    // One `creates_stacking_context` read serves both the z-step
    // ranking and the drain-anchor tracking below.
    let self_sc = creates_stacking_context(box_);
    if self_sc || is_positioned(box_) || box_.z_index.is_some() {
        doc_order.insert(self_ptr, *seq);
        *seq += 1;
    }
    let style = box_.style.as_deref().or(parent_style);
    // `visibility: hidden` skips the whole subtree in paint (groups must
    // stay balanced): its out-of-flow descendants are never deferred —
    // return before linking the ancestor frame below. `visibility:
    // collapse` paints the same as `hidden` (CSS 2.1 §17.5: on elements
    // other than rows, row groups, columns, and column groups it *means*
    // `hidden`); rows/columns collapsed this way are additionally
    // zero-sized by layout, so skipping their subtree removes them from
    // the display entirely.
    let vis = style.and_then(|s| s.get_str_by_index(props::VISIBILITY));
    if matches!(vis, Some("hidden") | Some("collapse")) {
        return;
    }
    // Ancestor frame for §11.1.1 escape-path capture (see the doc
    // comment): linked innermost-first through the call stack, so no
    // heap allocation. Linked after the `visibility: hidden` early
    // return so skipped subtrees leave no trace. The root is excluded:
    // its overflow is the viewport's.
    let frame = AncestorFrame {
        ptr: self_ptr,
        clips: box_.style.as_deref().is_some_and(overflow_clips),
        parent: ancestors,
    };
    let ancestors = if is_root { ancestors } else { Some(&frame) };
    let font_size = match box_.style.as_ref() {
        Some(s) => resolve_font_size(
            s.get_by_index(props::FONT_SIZE),
            uctx.parent_fs,
            uctx.root_fs,
            uctx.viewport.0,
            uctx.viewport.1,
        ),
        None => uctx.parent_fs,
    };
    let child_npa = if box_position(box_) == PositionAtom::Static {
        npa
    } else {
        Some(self_ptr)
    };
    // Nearest ancestor stacking context for drain anchoring. `child_nsc`
    // is the nearest one strictly below the children's containing block:
    // a static stacking-context box extends the accumulation, while a
    // positioned box restarts it (stacking contexts at or above the
    // containing block keep the containing-block anchor).
    // `child_nsc_vp` never restarts: `position: fixed` boxes are
    // viewport-anchored, so every ancestor stacking context sits below
    // their containing block.
    let child_nsc = if box_position(box_) == PositionAtom::Static {
        if self_sc { Some(self_ptr) } else { nsc }
    } else if self_sc {
        Some(self_ptr)
    } else {
        None
    };
    let child_nsc_vp = if self_sc { Some(self_ptr) } else { nsc_vp };
    // Widened version of the above, with no containing-block reset: the
    // nearest `is_stacking_context` ancestor, for hoisting z-indexed
    // boxes (positioned non-`auto` z-index counts, unlike above).
    let child_nsc_full = if is_stacking_context(box_) {
        Some(self_ptr)
    } else {
        nsc_full
    };
    // Overflow-clip frame for hoisted entries (see `ClipFrame`): the
    // root's frame is never pushed — no drain anchor sits strictly below
    // the root, so it is never part of a captured clip list.
    let pushed_clip = if !is_root && box_.style.as_deref().is_some_and(overflow_clips) {
        let radii = box_
            .style
            .as_ref()
            .map(|own| resolve_clip_radii(own, font_size, uctx, box_.rect))
            .unwrap_or([0.0; 4]);
        clip_stack.push(ClipFrame {
            ptr: self_ptr,
            rect: box_.rect,
            radii,
        });
        true
    } else {
        false
    };
    for child in &box_.children {
        let child_ctx = UnitCtx {
            parent_fs: font_size,
            ..uctx
        };
        if is_out_of_flow(child) {
            let child_ptr = child as *const LayoutBox;
            let fixed = box_position(child) == PositionAtom::Fixed;
            // `position: fixed` is viewport-anchored; an `absolute` box
            // with no positioned ancestor is too.
            let cb = if fixed { None } else { child_npa };
            // A z-indexed box forms a stacking context painted in its
            // nearest ancestor stacking context (Appendix E) — hoisted
            // past a non-stacking-context containing block. A z-indexed
            // `position: fixed` box hoists the same way: the viewport
            // anchor sees every ancestor stacking context below it. Any
            // other out-of-flow box keeps the old rule: the stacking
            // context below the containing block wins the drain anchor,
            // else the containing-block anchor is unchanged.
            let z_indexed = child.z_index.is_some();
            let anchor = if z_indexed {
                child_nsc_full.or(cb)
            } else if fixed {
                child_nsc_vp
            } else {
                child_nsc.or(cb)
            };
            // The anchor differs from the containing block: the drain
            // must paint outside the anchor's own overflow clip
            // (CSS 2.1 §11.1.1 — only the containing block's overflow
            // may clip an out-of-flow descendant).
            let escape_anchor_clip = if fixed {
                anchor.is_some()
            } else {
                anchor != cb
            };
            // Hoisted at/above the containing block: the drain sits
            // outside the containing block's subtree, so its overflow
            // clip no longer applies via nesting — capture and re-apply
            // it at the drain site. (Anchored below the containing
            // block, the drain stays inside its subtree and the captured
            // push is exactly redundant with the nested clip — except
            // under another entry's escape toggle, where it is
            // load-bearing.)
            let outer_clips = if escape_anchor_clip {
                cb.and_then(|cb_ptr| {
                    clip_stack
                        .iter()
                        .find(|f| f.ptr == cb_ptr)
                        .map(|f| vec![(f.rect, f.radii)])
                })
                .unwrap_or_default()
            } else {
                Vec::new()
            };
            // CSS 2.1 §11.1.1 escape path: the overflow-clipping boxes
            // strictly between the drain anchor and the containing block.
            // The drain paints inside the anchor's subtree, so each such
            // box's own clip would wrap it via nesting — record them
            // (each with the anchor-child it must wrap) so `build_box`
            // toggles the clip off around that subtree's steps. Only
            // when the anchor sits strictly below the containing block:
            // a hoisted entry (anchor at/above the containing block)
            // drains outside the containing block's subtree, so no
            // intermediate clip wraps it via nesting.
            if escape_anchor_clip && let Some(anchor_ptr) = anchor {
                // Phase A: walk the ancestor chain innermost-first to the
                // anchor. Passing the containing block first means the
                // anchor sits at/above it (hoisted) — nothing to record.
                // (`cb == None` is the viewport: never passed. A
                // containing block missing from the chain is the root,
                // which is never linked: everything above the anchor is
                // strictly below it.)
                let mut chain = ancestors;
                let mut at_anchor = false;
                while let Some(f) = chain {
                    if f.ptr == anchor_ptr {
                        at_anchor = true;
                        chain = f.parent;
                        break;
                    }
                    if Some(f.ptr) == cb {
                        // Anchor sits at/above the containing block
                        // (hoisted): no intermediate clip wraps the drain
                        // via nesting — nothing to record.
                        break;
                    }
                    chain = f.parent;
                }
                // Phase B: from just above the anchor outward, record
                // each clipping frame strictly below the containing
                // block, paired with the child on the path to the anchor
                // (starting at the anchor itself).
                if at_anchor {
                    let mut child_ptr = anchor_ptr;
                    while let Some(f) = chain {
                        if Some(f.ptr) == cb {
                            break;
                        }
                        if f.clips {
                            pending.record_escape_wrap(f.ptr, child_ptr);
                        }
                        child_ptr = f.ptr;
                        chain = f.parent;
                    }
                }
            }
            pending.push(PendingOof {
                box_: child,
                anchor,
                escape_anchor_clip,
                outer_clips,
                parent_style: style,
                uctx: child_ctx,
            });
            // The deferred box's own subtree still needs discovery: its
            // out-of-flow descendants anchor at it (it is positioned),
            // so the below-containing-block accumulation restarts here.
            // The widened accumulation sees the box itself when it is a
            // stacking context (e.g. a z-indexed deferred box).
            let child_sc = creates_stacking_context(child).then_some(child_ptr);
            let child_full_sc = is_stacking_context(child).then_some(child_ptr);
            discover_oof(
                child,
                style,
                child_ctx,
                Some(child_ptr),
                child_sc,
                child_sc.or(child_nsc_vp),
                pending,
                doc_order,
                seq,
                child_full_sc.or(nsc_full),
                clip_stack,
                false,
                ancestors,
            );
        } else if is_positioned(child) && child.z_index.is_some() && !is_stacking_context(box_) {
            // Appendix E hoist for in-flow content: a positioned box with
            // non-`auto` z-index forms a stacking context painted in the
            // nearest ancestor stacking context — not in this box's
            // steps — when this box is not itself a stacking context.
            // `build_box` skips the child in its classification; it
            // drains at the anchor below. Every ancestor overflow clip
            // travels with the entry (in-flow content stays clipped by
            // every ancestor's overflow): re-pushing a clip that still
            // applies via nesting is a visual no-op, and the full stack
            // restores correctness where another entry's escape toggle
            // popped a clip the drain nests inside.
            let child_ptr = child as *const LayoutBox;
            let anchor = nsc_full;
            let outer_clips: Vec<(Rect, [f32; 4])> =
                clip_stack.iter().map(|f| (f.rect, f.radii)).collect();
            pending.push(PendingOof {
                box_: child,
                anchor,
                escape_anchor_clip: false,
                outer_clips,
                parent_style: style,
                uctx: child_ctx,
            });
            // The hoisted box's own subtree still needs discovery (it is
            // positioned, so its out-of-flow descendants anchor at it);
            // it is itself a stacking context for the widened
            // accumulation below.
            let child_sc = creates_stacking_context(child).then_some(child_ptr);
            discover_oof(
                child,
                style,
                child_ctx,
                Some(child_ptr),
                child_sc,
                child_sc.or(child_nsc_vp),
                pending,
                doc_order,
                seq,
                Some(child_ptr),
                clip_stack,
                false,
                ancestors,
            );
        } else {
            discover_oof(
                child,
                style,
                child_ctx,
                child_npa,
                child_nsc,
                child_nsc_vp,
                pending,
                doc_order,
                seq,
                child_nsc_full,
                clip_stack,
                false,
                ancestors,
            );
        }
    }
    if pushed_clip {
        clip_stack.pop();
    }
}

/// An out-of-flow descendant deferred for painting at its drain anchor
/// (fixes 3/4): the nearest ancestor stacking context when one sits
/// below (deeper than) the containing block, else the containing
/// block's identity (`None` for the viewport — `position: fixed`, or
/// `absolute` with no positioned ancestor and no stacking-context
/// ancestor). Deferred boxes paint at their anchor's Appendix E z-index
/// step (see [`build_box`]), inside the anchor's opacity/blend group.
///
/// Clip placement (CSS 2.1 §11.1.1 — only the containing block's
/// overflow may clip an out-of-flow descendant):
/// - `escape_anchor_clip == false` (anchor IS the containing block, or
///   the viewport): the drain paints inside the anchor's own overflow
///   clip, exactly as before.
/// - `escape_anchor_clip == true` (anchor is a stacking context below
///   the containing block): the drain paints inside the anchor's
///   opacity/blend group but OUTSIDE the anchor's own overflow clip —
///   the anchor's `overflow: hidden` must not clip a box whose
///   containing block sits above it (fade masks, rounded cards). The
///   same holds for every overflow-clipping box strictly between the
///   anchor and the containing block: each is recorded in
///   [`PendingOofList::escape_wraps`] (with the direct child whose
///   subtree contains the anchor) so [`build_box`] toggles its own
///   clip off around that child's steps. Only the containing block's
///   own overflow still clips the drained box.
///
/// Entries are created by the up-front [`discover_oof`] pass, which runs
/// once over the whole tree before any paint item is emitted, so every
/// drain sees the complete set — including negative-z boxes nested deep
/// inside in-flow subtrees, which must paint at step 2, before in-flow
/// content, and therefore cannot be discovered lazily while painting.
///
/// Inherited properties and relative units resolve against the DOM
/// parent, not the containing block, so the deferral-site `parent_style`
/// and `uctx` are captured here at discovery time; the drain loops must
/// use these rather than the containing block's context (`font-size: 2em`
/// on a deferred box inside a 10px DOM parent but a 30px containing
/// block is 20px, not 60px).
///
/// Hoisted entries (see [`discover_oof`]) drain outside the subtree they
/// were lifted out of, so the overflow clips that would have applied via
/// nesting are captured in `outer_clips` and re-applied around the entry
/// at the drain site (each push balanced by a pop in the drain loop):
/// - a z-indexed out-of-flow box hoisted at/above its containing block
///   keeps the containing block's own overflow clip (CSS 2.1 §11.1.1);
/// - a hoisted in-flow positioned box keeps every ancestor overflow clip
///   (re-pushing a clip that still applies via nesting is a visual
///   no-op; it is load-bearing where another entry's escape toggle
///   popped it).
///
/// The containing block's clip is captured whenever the entry escapes
/// the anchor's clip — below-containing-block drains included: the drain
/// stays inside the clip's subtree, so the re-push is redundant with
/// nesting except under escape toggles, where it is load-bearing.
/// Not `Copy` (the clip list is a heap `Vec`, allocated only for hoisted
/// entries).
#[derive(Clone)]
struct PendingOof<'a> {
    box_: &'a LayoutBox,
    anchor: Option<*const LayoutBox>,
    /// True when the anchor differs from the entry's containing block:
    /// the drain must paint outside the anchor's own overflow clip
    /// (CSS 2.1 §11.1.1 — only the containing block's overflow may clip
    /// an out-of-flow descendant). False when the anchor IS the
    /// containing block (or the viewport), where the anchor's clip still
    /// applies. Always false for hoisted in-flow boxes: an ancestor's
    /// overflow clips in-flow content, anchor included.
    escape_anchor_clip: bool,
    /// Overflow clips to re-apply around the drained entry, outermost
    /// first (see above). Empty unless the entry was hoisted out of the
    /// clips' subtrees.
    outer_clips: Vec<(Rect, [f32; 4])>,
    parent_style: Option<&'a ComputedStyle>,
    uctx: UnitCtx,
}

/// One overflow-clipping ancestor frame captured during [`discover_oof`]:
/// the rect/radii the drain loop re-pushes for hoisted entries (see
/// [`PendingOof::outer_clips`]). The tree root's frame is never pushed —
/// no drain anchor sits strictly below the root, and the root-as-anchor
/// case is covered by the drain loop's own clip toggle — so every frame
/// here belongs to a proper descendant.
struct ClipFrame {
    ptr: *const LayoutBox,
    rect: Rect,
    radii: [f32; 4],
}

/// One ancestor frame for §11.1.1 escape-path capture (see
/// [`discover_oof`]): the ancestor's identity and whether it clips its
/// children, linked innermost-first through the `discover_oof` call
/// stack. Zero heap allocation — the frames live in the recursion's own
/// stack frames, so the discovery walk stays allocation-free for
/// ordinary trees (see `deep_styled_tree_build_allocations_bounded`).
struct AncestorFrame<'a> {
    ptr: *const LayoutBox,
    clips: bool,
    parent: Option<&'a AncestorFrame<'a>>,
}

/// Out-of-flow boxes deferred by the up-front [`discover_oof`] pass,
/// indexed by drain anchor (`None` = the viewport) so each box's
/// Appendix E drain touches only that anchor's entries — O(entries for
/// this anchor) instead of scanning every deferred box (quadratic on
/// trees with many absolutely-positioned descendants).
///
/// The map is built lazily on the first deferred box: ordinary trees
/// (no out-of-flow boxes) allocate nothing.
#[derive(Default)]
struct PendingOofList<'a> {
    by_anchor: Option<HashMap<Option<*const LayoutBox>, Vec<PendingOof<'a>>>>,
    /// CSS 2.1 §11.1.1 escape-path index, built by [`discover_oof`]:
    /// maps an overflow-clipping box strictly between a drain anchor
    /// and its containing block to that box's direct children whose
    /// subtree contains the anchor. [`build_box`] toggles its own
    /// overflow clip off around exactly those children's steps, so the
    /// drained box paints unclipped by the intermediate box (only the
    /// containing block's overflow may clip it). `None` until the first
    /// escaping entry with a non-empty path — ordinary trees allocate
    /// nothing.
    escape_wraps: Option<HashMap<*const LayoutBox, Vec<*const LayoutBox>>>,
}

impl<'a> PendingOofList<'a> {
    fn push(&mut self, entry: PendingOof<'a>) {
        self.by_anchor
            .get_or_insert_with(HashMap::new)
            .entry(entry.anchor)
            .or_default()
            .push(entry);
    }

    /// Record that `clip_box` (overflow-clipping, strictly between a
    /// drain anchor and its containing block) must not clip the subtree
    /// of its direct child `anchor_child`, which contains the anchor.
    fn record_escape_wrap(&mut self, clip_box: *const LayoutBox, anchor_child: *const LayoutBox) {
        let v = self
            .escape_wraps
            .get_or_insert_with(HashMap::new)
            .entry(clip_box)
            .or_default();
        if !v.contains(&anchor_child) {
            v.push(anchor_child);
        }
    }

    /// This box's direct children whose subtree contains a drain anchor
    /// with this box on its §11.1.1 escape path (`None` when there are
    /// none).
    fn escape_wraps_for(&self, box_: *const LayoutBox) -> Option<&Vec<*const LayoutBox>> {
        self.escape_wraps.as_ref().and_then(|m| m.get(&box_))
    }

    /// Whether any deferred box is anchored at `anchor` (`None` = viewport).
    fn has_entries_for(&self, anchor: Option<*const LayoutBox>) -> bool {
        self.by_anchor
            .as_ref()
            .is_some_and(|m| m.get(&anchor).is_some_and(|v| !v.is_empty()))
    }

    /// Remove the entries anchored at `anchor` whose z-index satisfies
    /// `pred`, in discovery (document) order, pushing them tagged with
    /// `rank` into the build-wide step scratch (see [`StepScratch`])
    /// instead of returning a fresh Vec — the Appendix E steps in
    /// [`build_box`] drain straight into the scratch. Every entry is
    /// removed exactly once — at its anchor's steps — so the list is
    /// empty when the paint walk finishes.
    fn drain_into(
        &mut self,
        anchor: Option<*const LayoutBox>,
        pred: impl Fn(Option<i32>) -> bool,
        rank: u8,
        out: &mut Vec<(u8, StepChild<'a>)>,
    ) {
        let Some(m) = self.by_anchor.as_mut() else {
            return;
        };
        // Take ownership of the whole vector and partition it in one
        // pass: repeated `Vec::remove` shifts the tail on every match,
        // which is O(n*k) when k entries drain at one anchor
        // (quadratic for the common all-z-auto case).
        let Some(v) = m.remove(&anchor) else {
            return;
        };
        let mut kept = Vec::with_capacity(v.len());
        for entry in v {
            if pred(entry.box_.z_index) {
                out.push((rank, StepChild::Oof(entry)));
            } else {
                kept.push(entry);
            }
        }
        if !kept.is_empty() {
            m.insert(anchor, kept);
        }
    }

    /// True when no entry remains anywhere.
    fn is_empty(&self) -> bool {
        self.by_anchor
            .as_ref()
            .is_none_or(|m| m.values().all(Vec::is_empty))
    }

    /// Remove and return every remaining entry (safety-net drain only).
    fn take_all(&mut self) -> Vec<PendingOof<'a>> {
        self.by_anchor
            .take()
            .map(|m| m.into_values().flatten().collect())
            .unwrap_or_default()
    }
}

/// Per-side `border-*-width` / `border-*-style` / `border-*-color`
/// property names, in [top, right, bottom, left] order.
///
/// Static strings: the border path runs for every styled box, and
/// `format!("border-{side}-width")` here cost 12 heap allocations per
/// box per frame even when no border is set at all.
const BORDER_WIDTH_PROPS: [u8; 4] = [
    props::BORDER_TOP_WIDTH,
    props::BORDER_RIGHT_WIDTH,
    props::BORDER_BOTTOM_WIDTH,
    props::BORDER_LEFT_WIDTH,
];
const BORDER_STYLE_PROPS: [u8; 4] = [
    props::BORDER_TOP_STYLE,
    props::BORDER_RIGHT_STYLE,
    props::BORDER_BOTTOM_STYLE,
    props::BORDER_LEFT_STYLE,
];
const BORDER_COLOR_PROPS: [u8; 4] = [
    props::BORDER_TOP_COLOR,
    props::BORDER_RIGHT_COLOR,
    props::BORDER_BOTTOM_COLOR,
    props::BORDER_LEFT_COLOR,
];

/// Text-decoration line kinds from the computed `text-decoration` value.
/// Wavy variants come from `text-decoration-style: wavy`, or from `wavy`
/// written inline in the `text-decoration` shorthand (the parser keeps
/// the shorthand's keywords together, so the style keyword can appear
/// in the same value list as the line keywords).
fn deco_kinds(style: &ComputedStyle) -> Vec<DecoKind> {
    let wavy = style.get_str_by_index(props::TEXT_DECORATION_STYLE) == Some("wavy") || {
        let is_wavy = |v: &CssValue| matches!(v, CssValue::Keyword(s) if s == "wavy");
        match style.get_by_index(props::TEXT_DECORATION) {
            Some(CssValue::List(items)) => items.iter().any(is_wavy),
            Some(v) => is_wavy(v),
            None => false,
        }
    };
    let mut kinds = Vec::new();
    let mut push = |word: &str| match word {
        "underline" => kinds.push(if wavy {
            DecoKind::WavyUnderline
        } else {
            DecoKind::Underline
        }),
        "overline" => kinds.push(if wavy {
            DecoKind::WavyOverline
        } else {
            DecoKind::Overline
        }),
        "line-through" => kinds.push(if wavy {
            DecoKind::WavyLineThrough
        } else {
            DecoKind::LineThrough
        }),
        _ => {}
    };
    match style.get_by_index(props::TEXT_DECORATION) {
        Some(CssValue::Keyword(s)) => push(s),
        Some(CssValue::List(items)) => {
            for item in items {
                if let CssValue::Keyword(s) = item {
                    push(s);
                }
            }
        }
        _ => {}
    }
    kinds
}

/// The verbatim slice of `text` a fragment renders: the fragment's
/// `text_range` into the box's (already `text-transform`ed) text.
///
/// Slices are exact even with proportional fonts, where
/// width-proportional cuts land mid-word or on the wrong line. An
/// out-of-bounds range (never produced by layout) yields an empty slice
/// rather than panicking.
///
/// Returns a borrow, never an owned `String`: the text-run loop calls
/// this per fragment directly instead of collecting a `Vec<String>` per
/// text box. The loop wraps the box's text in one shared `Arc<str>` and
/// `Arc::clone`s it per fragment, so the per-fragment slice copy this
/// borrow used to feed (via `push_text_run`'s old `to_string`) is gone.
fn verbatim_slice<'t>(text: &'t str, frag: &TextFragment) -> &'t str {
    text.get(frag.text_range.clone()).unwrap_or("")
}

/// 1-based index of an `<li>` among its `<li>` siblings.
///
/// Layout does not number list items, so the display-list builder
/// counts preceding `<li>` siblings straight from the DOM.
fn list_item_index(doc: &Document, node: NodeId) -> u32 {
    let mut index = 1u32;
    let mut next = doc.previous_sibling(node);
    while let Some(sib) = next {
        if let NodeKind::Element(el) = doc.kind(sib)
            && el.tag_name == "li"
        {
            index += 1;
        }
        next = doc.previous_sibling(sib);
    }
    index
}

/// Build a display list from a laid-out tree.
///
/// `font_resolver` maps each text run's computed style to a [`FontKey`];
/// `image_resolver` maps an `<img src>` / `background-image` URL to
/// decoded bytes (returning `None` paints the image placeholder box).
/// Id-stability contract: the resolver must return ids that are stable
/// per distinct image content within one list. `build_box` calls the
/// resolver per element and registering an id overwrites the previous
/// bytes for that id, so a resolver returning one constant id for
/// different URLs silently renders every image as the last-registered
/// bytes. (`DisplayListBuilder::add_image` mints fresh unique ids when
/// the caller does not manage the id space itself.)
/// `viewport` is the layout viewport in CSS px: the basis for `vw`/`vh`
/// units (the root box's rect cannot supply it — its height is the
/// document height, not the viewport height).
/// All emitted coordinates are absolute page px: scrolling never requires
/// a rebuild.
///
/// Per box, in CSS painting order: background, background image, border,
/// replaced-element image / list marker, text runs (+ decorations over
/// them), then children. Boxes with `opacity < 1` wrap their whole
/// subtree in an opacity group. Boxes with `overflow: hidden` (or `clip`)
/// wrap their children in a clip to the border box; boxes with
/// `visibility: hidden` (inherited) are skipped entirely.
///
/// Out-of-flow (`position: absolute | fixed`) descendants paint at their
/// containing block's Appendix E z-index step (CSS 2.1 §11.1.1), and
/// only the containing block's overflow clips them.
///
/// Paint order within each box follows CSS 2.1 Appendix E: the box's own
/// background/borders, child stacking contexts with negative z-index,
/// in-flow non-positioned block backgrounds, floats, in-flow inline
/// content, positioned descendants with z-index `auto`/`0`, then child
/// stacking contexts with positive z-index. A stacking context is the
/// root, a positioned box with non-`auto` z-index, a box with
/// `opacity < 1` or non-`normal` `mix-blend-mode`, or a flex/grid item
/// with non-`auto` z-index even when static. (`LayoutBox::z_index` is
/// `None` for `auto` except on static flex/grid items, where layout
/// preserves it.) Flex/grid items additionally
/// re-sort by `order`, as if the document were reordered.
///
/// Known simplifications: each child paints atomically (a block's inline
/// descendants do not interleave with later floats).
///
/// A positioned box with non-`auto` z-index whose parent (or containing
/// block, for out-of-flow boxes) is not a stacking context is hoisted to
/// the nearest ancestor stacking context for paint order (CSS 2.1
/// Appendix E). Hoisted flex/grid items lose the `order` tie-break
/// against their new siblings.
pub fn build_display_list(
    root: &LayoutBox,
    doc: &Document,
    font_resolver: &mut dyn FnMut(&ComputedStyle) -> FontKey,
    image_resolver: &mut dyn FnMut(&str) -> Option<(ImageId, DecodedImage)>,
    viewport: Size,
) -> DisplayList {
    let mut scratch = StepScratch::new();
    build_display_list_with_scratch(
        root,
        doc,
        font_resolver,
        image_resolver,
        viewport,
        &mut scratch,
    )
}

/// [`build_display_list`] with a caller-owned [`StepScratch`]: the
/// Appendix E step scratch lives for the whole build and is reused by
/// every stacking context (see [`StepScratch`]). Production calls use
/// a fresh scratch per build via [`build_display_list`]; tests pass
/// their own scratch to observe the reuse.
fn build_display_list_with_scratch<'a>(
    root: &'a LayoutBox,
    doc: &Document,
    font_resolver: &mut dyn FnMut(&ComputedStyle) -> FontKey,
    image_resolver: &mut dyn FnMut(&str) -> Option<(ImageId, DecodedImage)>,
    viewport: Size,
    scratch: &mut StepScratch<'a>,
) -> DisplayList {
    let mut b = DisplayListBuilder::new();
    build_display_list_inner(
        &mut b,
        root,
        doc,
        font_resolver,
        image_resolver,
        viewport,
        scratch,
    );
    b.build()
}

/// Build a display list reusing a retained [`DisplayListBuilder`]
/// across frames.
///
/// This is the production frame path (window.rs keeps one builder in
/// a thread-local): `out` is the previous frame's finished list (or an
/// empty one) and its item/image storage is recycled, so after the
/// first frame the build performs no `Vec<DisplayItem>` regrow ramp and
/// no per-frame gradient/filter cache-map regrow. `style_changed` must
/// be true when styles were recalculated since the last build — it
/// drives the builder's cache epoch (see
/// [`DisplayListBuilder::begin_reused_build`]).
///
/// Behavior is identical to [`build_display_list`]: same items, same
/// image ids, same pixels.
#[allow(clippy::too_many_arguments)] // mirrors `build_display_list`'s shape plus the reuse params
pub fn build_display_list_reused(
    builder: &mut DisplayListBuilder,
    out: &mut DisplayList,
    root: &LayoutBox,
    doc: &Document,
    font_resolver: &mut dyn FnMut(&ComputedStyle) -> FontKey,
    image_resolver: &mut dyn FnMut(&str) -> Option<(ImageId, DecodedImage)>,
    viewport: Size,
    style_changed: bool,
) {
    builder.begin_reused_build(out, style_changed);
    let mut scratch = StepScratch::new();
    build_display_list_inner(
        builder,
        root,
        doc,
        font_resolver,
        image_resolver,
        viewport,
        &mut scratch,
    );
    builder.finish_reused_build(out);
}

/// The box walk shared by [`build_display_list_with_scratch`] and
/// [`build_display_list_reused`]: drains layout's `column-rule` specs
/// for this pass, discovers out-of-flow boxes up front, and paints the
/// tree per CSS 2.1 Appendix E into `b`.
fn build_display_list_inner<'a>(
    b: &mut DisplayListBuilder,
    root: &'a LayoutBox,
    doc: &Document,
    font_resolver: &mut dyn FnMut(&ComputedStyle) -> FontKey,
    image_resolver: &mut dyn FnMut(&str) -> Option<(ImageId, DecodedImage)>,
    viewport: Size,
    scratch: &mut StepScratch<'a>,
) {
    // Drain the `column-rule` geometry layout recorded during this pass,
    // exactly once: the box walk below claims each spec when it visits
    // the spec's container (see `paint_column_rules`). The table is
    // thread-local and per-layout-pass, so draining here pairs this
    // build with the layout that preceded it.
    b.set_column_rules(take_column_rules());
    // Viewport basis for `vw`/`vh`: the layout viewport passed in, not
    // the root box's rect — the root's height is its content height
    // (document height), not the viewport height.
    let viewport = (viewport.w, viewport.h);
    // Root font size, mirroring bowser_layout: the root element's
    // computed font-size resolved against the 16px initial value
    // (`rem` on the root itself means 16px there).
    let root_fs = resolve_font_size(
        root.style
            .as_ref()
            .and_then(|s| s.get_by_index(props::FONT_SIZE)),
        16.0,
        16.0,
        viewport.0,
        viewport.1,
    );
    let uctx = UnitCtx {
        parent_fs: 16.0,
        root_fs,
        viewport,
    };
    let mut pending = PendingOofList::default();
    // Up-front out-of-flow discovery: anchor every absolutely/fixed-
    // positioned box at its containing block before any paint item is
    // emitted, so the Appendix E drains in `build_box` see the complete
    // set (negative-z boxes nested in in-flow subtrees must paint at
    // step 2, before in-flow content). Also ranks boxes that join the
    // z-index steps for tie-breaking (`doc_order` stays empty for
    // ordinary trees).
    let mut doc_order: FnvHashMap<*const LayoutBox, usize> = FnvHashMap::default();
    // Overflow-clip frames of the current ancestor chain, captured for
    // hoisted entries (see `discover_oof`); empty (zero allocation) when
    // nothing clips.
    let mut clip_stack: Vec<ClipFrame> = Vec::new();
    {
        let mut seq = 0usize;
        discover_oof(
            root,
            None,
            uctx,
            None,
            None,
            None,
            &mut pending,
            &mut doc_order,
            &mut seq,
            // The root is the root stacking context: seed the widened
            // accumulation so z-indexed boxes with no other
            // stacking-context ancestor hoist to it (Appendix E).
            Some(root as *const LayoutBox),
            &mut clip_stack,
            true,
            // The root is excluded from the ancestor chain (see
            // `discover_oof`).
            None,
        );
    }
    build_box(
        b,
        scratch,
        root,
        doc,
        font_resolver,
        image_resolver,
        None,
        uctx,
        false,
        None,
        &mut pending,
        true,
        &doc_order,
    );
    // Up-front discovery plus per-containing-block drains provably
    // empty the list: every deferred box is removed exactly once, at its
    // own containing block's steps (viewport-anchored boxes at the
    // root's). If a future `build_box` early-return beyond
    // `visibility: hidden` ever strands entries here, this assert fails
    // loudly instead of silently painting them last. The drain below is
    // a belt-and-braces fallback, not a live path.
    debug_assert!(
        pending.is_empty(),
        "pending out-of-flow list not fully drained"
    );
    // Safety net: drain anything left in z-index, document order rather
    // than dropping it. Negative-z viewport boxes already painted at the
    // root's step 2, so this only sees z-index >= auto. `doc_order`
    // ranks step-joining boxes and `sort_by_key` is stable, so ties stay
    // in document order.
    let mut rest = pending.take_all();
    rest.sort_by_key(|p| (p.box_.z_index.unwrap_or(0), doc_seq(p.box_, &doc_order)));
    for p in rest {
        // Deferral-site context again: viewport-anchored boxes still
        // inherit from their DOM parent, not from the root.
        build_box(
            b,
            scratch,
            p.box_,
            doc,
            font_resolver,
            image_resolver,
            p.parent_style,
            p.uctx,
            false,
            None,
            &mut pending,
            false,
            &doc_order,
        );
    }
}

/// [`build_display_list`] with a constant [`FontKey(0)`] font resolver.
///
/// Convenience for callers without a font cache; `crate::render` builds
/// its display lists with a `FontCache`-backed resolver (family/weight/
/// style read from each run's [`ComputedStyle`]) through
/// [`build_display_list`].
pub fn build_display_list_with_images(
    root: &LayoutBox,
    doc: &Document,
    image_resolver: &mut dyn FnMut(&str) -> Option<(ImageId, DecodedImage)>,
    viewport: Size,
) -> DisplayList {
    build_display_list(root, doc, &mut |_| FontKey(0), image_resolver, viewport)
}

/// Push a box's own overflow clip: with `border-radius` the clip follows
/// the rounded border box, otherwise it is the plain rect clip.
/// Extracted so the Appendix E step loop in [`build_box`] can toggle the
/// clip around out-of-clip drain runs (CSS 2.1 §11.1.1) instead of
/// holding one eager push/pop pair.
fn push_own_clip(b: &mut DisplayListBuilder, rect: Rect, radii: [f32; 4]) {
    // Rounded corners cut the children exactly where the border curve
    // runs. Without radii this is the plain rect clip as before.
    if radii.iter().any(|r| *r > 0.0) {
        b.push_rounded_clip(rect, radii);
    } else {
        b.push_clip(rect);
    }
}

// Thirteen parameters: the display-list builder walks the whole box tree,
// so its context travels as arguments (precedent: `#[allow]`s in
// `bowser_net` and `bowser_layout` for the same reason).
//
// `npa` is the nearest positioned ancestor's identity (`None` when there
// is none above this box); `pending` holds the up-front out-of-flow
// discovery (see [`build_display_list`]); `scratch` is the build-wide
// Appendix E step scratch (see [`StepScratch`]); `is_root` marks the outermost
// call — the root stacking context, which also drains viewport-anchored
// boxes into its own Appendix E steps; `doc_order` ranks positioned
// boxes in document order for z-index tie-breaking (empty for ordinary
// trees — no allocation in the hot path).
#[allow(clippy::too_many_arguments)]
fn build_box<'a>(
    b: &mut DisplayListBuilder,
    scratch: &mut StepScratch<'a>,
    box_: &'a LayoutBox,
    doc: &Document,
    font_resolver: &mut dyn FnMut(&ComputedStyle) -> FontKey,
    image_resolver: &mut dyn FnMut(&str) -> Option<(ImageId, DecodedImage)>,
    parent_style: Option<&'a ComputedStyle>,
    uctx: UnitCtx,
    in_collapse: bool,
    npa: Option<*const LayoutBox>,
    pending: &mut PendingOofList<'a>,
    is_root: bool,
    doc_order: &FnvHashMap<*const LayoutBox, usize>,
) {
    // Nearest style for inherited text properties (text boxes have none).
    let style = box_.style.as_deref().or(parent_style);

    // Whether this box is inside a `border-collapse: collapse` table's
    // border model. Recomputed at each `display: table` box (a nested
    // table resets it); otherwise inherited from the parent box.
    // Case-insensitive, matching the layout side that computed the
    // collapsed winners.
    let in_collapse = if box_.box_type == BoxType::Table {
        box_.style.as_ref().is_some_and(|s| {
            s.get_str_by_index(props::BORDER_COLLAPSE)
                .is_some_and(|v| v.eq_ignore_ascii_case("collapse"))
        })
    } else {
        in_collapse
    };

    // This box's resolved font size: the `em` basis for the lengths read
    // below, and the size stamped on text runs. A box with its own style
    // resolves its `font-size` against the parent's via
    // `bowser_layout::resolve_font_size` (including a resolved zero
    // staying zero); style-less (anonymous/text) boxes inherit the
    // parent size outright.
    let font_size = match box_.style.as_ref() {
        Some(s) => resolve_font_size(
            s.get_by_index(props::FONT_SIZE),
            uctx.parent_fs,
            uctx.root_fs,
            uctx.viewport.0,
            uctx.viewport.1,
        ),
        None => uctx.parent_fs,
    };

    // `visibility: hidden` (inherited): skip the box and its whole subtree
    // before any group is opened, so push/pop items stay balanced.
    // `visibility: collapse` paints the same as `hidden` (CSS 2.1 §17.5:
    // on elements other than rows, row groups, columns, and column
    // groups it *means* `hidden`); rows/columns collapsed this way are
    // additionally zero-sized by layout, so skipping their subtree
    // removes them from the display entirely.
    let vis = style.and_then(|s| s.get_str_by_index(props::VISIBILITY));
    if matches!(vis, Some("hidden") | Some("collapse")) {
        return;
    }

    // Blend group around the whole subtree (background included),
    // opened *outside* the opacity group: the blend applies against
    // the real page backdrop first, then the opacity fades the blended
    // result (an inner blend would blend against the transparent
    // opacity-group surface instead of the page). `Normal` emits
    // nothing (cheap no-op path). Uses the box's *own* style only:
    // unlike `opacity`, `mix-blend-mode` is not inherited, so
    // anonymous text boxes must not re-blend inside a blended parent
    // (the parent's group already isolates the whole subtree).
    let blend = box_
        .style
        .as_deref()
        .map(css_blend_mode)
        .unwrap_or(BlendMode::Normal);
    let blended = !blend.is_normal();
    if blended {
        b.push_blend(blend);
    }

    // Opacity group inside the blend group. Uses the box's *own*
    // style only: unlike the inherited text properties read through
    // `style` above, `opacity` is not inherited, so anonymous text
    // boxes (style None) must not re-open the parent's opacity group
    // around their own text runs — that would fade the text twice
    // (0.5 inside 0.5 renders at 0.25). Mirrors the blend-group code,
    // which already reads the box's own style for the same reason.
    let opacity = box_.style.as_deref().map(css_opacity).unwrap_or(1.0);
    let grouped = opacity < 1.0 - f32::EPSILON;
    if grouped {
        b.push_opacity(opacity);
    }

    // Transform group inside the opacity group: the transform applies
    // to the element's whole rendering (background, borders, content,
    // shadows); blend and opacity then apply to the transformed result
    // (a transformed blend must blend the transformed pixels, so the
    // transform group nests inside the blend group too). Identity
    // matrices emit nothing (backends paint them inline); anything
    // else — including non-invertible matrices, which paint nothing
    // per spec — goes through the isolated group path. Uses the box's
    // *own* style only: `transform` is not inherited, so anonymous
    // text boxes must not re-transform inside a transformed parent
    // (the parent's group already isolates the whole subtree).
    let transform = box_
        .style
        .as_deref()
        .and_then(|s| css_transform_matrix(s, box_.rect))
        .filter(|m| *m != [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    if let Some(m) = transform {
        b.push_transform(m);
    }

    // `clip-path` (css-clip-path-1 §4): intersects the element's whole
    // rendering — own items, filter output, and descendants — with the
    // resolved shape. `inset()` without `round` lowers to the plain rect
    // clip, with `round` to the rounded-rect clip; every other shape
    // lowers to a polygon clip. `none` (and unresolvable shapes) emit
    // nothing: the zero-cost fast path. Uses the box's *own* style only:
    // `clip-path` is not inherited, so anonymous text boxes must not
    // re-clip inside a clipped parent (the parent's clip already covers
    // the whole subtree).
    let length_ctx = LengthCtx {
        em: font_size,
        root: uctx.root_fs,
        viewport: uctx.viewport,
    };
    let clip_path = box_
        .style
        .as_deref()
        .and_then(|s| match s.get_by_index(props::CLIP_PATH) {
            Some(CssValue::ClipPath(cp)) => resolve_clip_path(cp, &box_.rect, &length_ctx),
            _ => None,
        });
    let clipped = clip_path.is_some();
    // By value: the polygon's `Vec` moves into the inline-first
    // `ClipPoints` (no clone); the rect cases are `Copy`.
    match clip_path {
        Some(ResolvedClip::Rect(rect)) => {
            b.push_clip(rect);
        }
        Some(ResolvedClip::RoundedRect { rect, radii }) => {
            b.push_rounded_clip(rect, radii);
        }
        Some(ResolvedClip::Polygon { points }) => {
            b.push_clip_path(points.into());
        }
        None => {}
    }

    // `currentcolor` for `drop-shadow()`'s default color.
    let current_color = box_
        .style
        .as_deref()
        .and_then(|s| css_color(s, props::COLOR))
        .unwrap_or(Color::rgba8(0, 0, 0, 255));

    // `backdrop-filter` (Filter Effects 2 §2): filters the
    // already-painted backdrop under the element's border box, then
    // paints the element's content over it. `none`/empty chains emit
    // nothing. Like `filter`, uses the box's *own* style only.
    let backdrop_filter_ops = box_
        .style
        .as_deref()
        .and_then(|s| match s.get_by_index(props::BACKDROP_FILTER) {
            Some(CssValue::Filter(f)) => Some(b.resolved_filter(f, &length_ctx, current_color)),
            _ => None,
        })
        .filter(|ops| !ops.is_empty());
    let backdrop_filtered = backdrop_filter_ops.is_some();
    if let Some(ops) = backdrop_filter_ops {
        b.push_backdrop_filter(ops, box_.rect);
    }

    // `filter` (Filter Effects 1 §8): filters the element's whole
    // rendering (background, borders, content, descendants). `none`/
    // empty chains emit nothing: the zero-cost fast path. Uses the
    // box's *own* style only, like the opacity/blend/transform groups
    // above.
    let filter_ops = box_
        .style
        .as_deref()
        .and_then(|s| match s.get_by_index(props::FILTER) {
            Some(CssValue::Filter(f)) => Some(b.resolved_filter(f, &length_ctx, current_color)),
            _ => None,
        })
        .filter(|ops| !ops.is_empty());
    let filtered = filter_ops.is_some();
    if let Some(ops) = filter_ops {
        b.push_filter(ops);
    }

    if let Some(style) = style {
        // CSS 2.1 §17.5.1 `empty-cells: hide`: in the separated borders
        // model, an empty cell paints no background and no border — as
        // if the cell alone were `visibility: hidden` (the row's and
        // table's own backgrounds still show through behind it).
        // `empty-cells` is inherited, so the cell's own computed style
        // carries the table's value; the collapsing border model
        // ignores the property (`in_collapse` is true there).
        let hide_empty_cell = !in_collapse
            && box_.box_type == BoxType::TableCell
            && box_.style.as_ref().is_some_and(|s| {
                s.get_str_by_index(props::EMPTY_CELLS)
                    .is_some_and(|v| v.eq_ignore_ascii_case("hide"))
            })
            && table_cell_is_empty(box_);
        // Background fill.
        // `box-shadow`: outset layers paint below the background (CSS
        // Backgrounds 3 §7.2). Not inherited, so anonymous boxes must
        // not re-emit the parent's shadow: the box's own style only.
        // `empty-cells: hide` suppresses the shadow like the background.
        let text_color = css_color(style, props::COLOR).unwrap_or(Color::rgba8(0, 0, 0, 255));
        if !hide_empty_cell && let Some(own) = box_.style.as_ref() {
            emit_box_shadows(b, box_.rect, own, text_color, font_size, uctx, false);
        }
        if !hide_empty_cell
            && let Some(color) = css_color(style, props::BACKGROUND_COLOR)
            && color.a() > 0.0
        {
            b.push_fill_rect(box_.rect, color);
        }
        // Background image: tiled per `background-repeat` /
        // `background-position`; the builder emits one `RepeatImage`
        // item, which backends expand natively (no per-tile items).
        if !hide_empty_cell
            && let Some(CssValue::Url(url)) = style.get_by_index(props::BACKGROUND_IMAGE)
            && let Some((id, decoded)) = image_resolver(url)
        {
            let tile = (decoded.width as f32, decoded.height as f32);
            b.register_image(id, decoded);
            let repeat = css_background_repeat(style);
            let offset = css_background_position(
                style,
                box_.rect,
                tile,
                font_size,
                uctx.root_fs,
                uctx.viewport,
            );
            b.push_background_image_tiled(box_.rect, id, tile, repeat, offset);
        }
        // Background gradient: gradients are `<image>`s (CSS Images 3
        // §3.1), so `background-image: linear-gradient(...)` reaches
        // here as `CssValue::Gradient`. They paint once over the border
        // box: `background-repeat` / `background-position` tiling does
        // not apply. A gradient that fails to parse emits nothing,
        // matching the rendering of a dropped declaration.
        // `background-image` is not inherited, so this reads the box's
        // *own* style only: an anonymous/text child (style `None`,
        // computing to `none`) must not re-emit the parent's gradient
        // over its own smaller rect — the parse would resolve against
        // the wrong rect, a visible seam (unlike `background-color`,
        // re-painting a gradient is not idempotent). Parsed sources are
        // memoized on the builder (see `parsed_gradient`); `text_color`
        // still comes from the inherited `style` above (`color` *is*
        // inherited, which is what `currentcolor` stops resolve to).
        if !hide_empty_cell
            && let Some(own) = box_.style.as_ref()
            && let Some(CssValue::Gradient { source, .. }) =
                own.get_by_index(props::BACKGROUND_IMAGE)
            && let Some(parsed) = b.parsed_gradient(source)
        {
            b.push_gradient(box_.rect, &parsed, text_color);
        }
        // `box-shadow`: inset layers paint above the background but
        // below the border (CSS Backgrounds 3 §7.2). Same
        // non-inherited, `empty-cells`-gated emission as the outset
        // layers above.
        if !hide_empty_cell && let Some(own) = box_.style.as_ref() {
            emit_box_shadows(b, box_.rect, own, text_color, font_size, uctx, true);
        }
        // Border.
        //
        // In the collapsing border model (`border-collapse: collapse`),
        // table/row/row-group/column/column-group boxes paint no border
        // of their own: their claims were folded into the per-cell
        // winners during layout.
        // Each cell instead paints its [`LayoutBox::collapsed_borders`]
        // winners — the single shared border per grid line, owned by
        // exactly one cell, centered on the grid line (CSS 2.1 §17.6.2.2).
        let collapsed = if box_.box_type == BoxType::TableCell {
            box_.collapsed_borders.as_deref().copied()
        } else {
            None
        };
        // `empty-cells: hide` (see above): an empty cell in the
        // separated model paints no border either. Unreachable in the
        // collapsing model (`hide_empty_cell` is false there), but
        // stated explicitly so the two paths can't diverge.
        if !hide_empty_cell && let Some(edges) = collapsed {
            let mut widths = [0.0f32; 4];
            let mut styles = [BorderStyle::None; 4];
            let mut colors = [Color::TRANSPARENT; 4];
            let mut any = false;
            for (i, e) in edges.iter().enumerate() {
                let s = match e.style {
                    CollapsedBorderStyle::Hidden | CollapsedBorderStyle::None => BorderStyle::None,
                    CollapsedBorderStyle::Double => BorderStyle::Double,
                    CollapsedBorderStyle::Solid => BorderStyle::Solid,
                    CollapsedBorderStyle::Dashed => BorderStyle::Dashed,
                    CollapsedBorderStyle::Dotted => BorderStyle::Dotted,
                    CollapsedBorderStyle::Ridge => BorderStyle::Ridge,
                    CollapsedBorderStyle::Outset => BorderStyle::Outset,
                    CollapsedBorderStyle::Groove => BorderStyle::Groove,
                    CollapsedBorderStyle::Inset => BorderStyle::Inset,
                };
                if e.width > 0.0 && !matches!(s, BorderStyle::None) {
                    widths[i] = e.width;
                    styles[i] = s;
                    colors[i] = Color::rgba8(e.color.r, e.color.g, e.color.b, e.color.a);
                    any = true;
                }
            }
            if any {
                // Center each side on its grid line: the tessellator
                // draws borders inside `rect`, so grow the rect by half
                // the winning width on every painted side.
                let r = box_.rect;
                let rect = Rect {
                    x: r.x - widths[3] / 2.0,
                    y: r.y - widths[0] / 2.0,
                    w: r.w + (widths[3] + widths[1]) / 2.0,
                    h: r.h + (widths[0] + widths[2]) / 2.0,
                };
                // `border-radius` has no effect in the collapsing border
                // model; the tessellator still takes radii, all zero.
                b.push_border(rect, widths, styles, colors, [0.0; 4]);
            }
        // Borders are not inherited: the plain CSS border path reads
        // only the box's own style. Anonymous/text boxes borrow the
        // parent's style for inherited text properties above, so they
        // must not re-emit the parent's border as extra fragments.
        // `empty-cells: hide` (see above): an empty cell in the
        // separated model paints no border.
        } else if !hide_empty_cell
            && let Some(own) = box_.style.as_ref()
            && (!in_collapse
                || !matches!(
                    box_.box_type,
                    BoxType::Table
                        | BoxType::TableRowGroup
                        | BoxType::TableRow
                        | BoxType::TableColumn
                        | BoxType::TableColumnGroup
                ))
        {
            let mut widths = [0.0f32; 4];
            let mut styles = [BorderStyle::None; 4];
            let mut colors = [Color::TRANSPARENT; 4];
            let mut any = false;
            for i in 0..4 {
                let w = resolve_length(
                    own.get_by_index(BORDER_WIDTH_PROPS[i]),
                    font_size,
                    uctx.root_fs,
                    uctx.viewport,
                );
                let s = css_border_style(own, BORDER_STYLE_PROPS[i]);
                if w > 0.0 && !matches!(s, BorderStyle::None | BorderStyle::Hidden) {
                    widths[i] = w;
                    styles[i] = s;
                    colors[i] =
                        css_color(own, BORDER_COLOR_PROPS[i]).unwrap_or(Color::rgba8(0, 0, 0, 255));
                    any = true;
                }
            }
            if any {
                // Outer corner radii in px; the shadow emitter shares
                // this helper (the shadow follows the border box curve).
                let radii = css_border_radii(own, font_size, uctx);
                b.push_border(box_.rect, widths, styles, colors, radii);
            }
        }
    }

    // Column rules (`column-rule` on multicol containers): one vertical
    // rule item per column gap, painted in the container's border phase.
    paint_column_rules(b, box_, font_size, uctx);

    // `<img>` replaced element, and `<li>` markers.
    if let Some(node) = box_.node
        && let NodeKind::Element(el) = doc.kind(node)
    {
        if el.tag_name == "img" {
            let mut painted = false;
            if let Some(src) = doc.get_attribute(node, "src")
                && let Some((id, decoded)) = image_resolver(src)
            {
                b.register_image(id, decoded);
                b.push_image(box_.rect, id);
                painted = true;
            }
            if !painted {
                // Placeholder box for missing/unloadable images.
                b.push_fill_rect(box_.rect, Color::rgba8(200, 200, 200, 255));
            }
        } else if el.tag_name == "li"
            && let Some(style) = style
        {
            // 1-based item number among `<li>` siblings; layout
            // does not number items, so count from the DOM here.
            let index = list_item_index(doc, node);
            let kind = match style.get_str_by_index(props::LIST_STYLE_TYPE) {
                Some("circle") => Some(MarkerKind::Circle),
                Some("square") => Some(MarkerKind::Square),
                Some("decimal") => Some(MarkerKind::Decimal(index)),
                Some("decimal-leading-zero") => Some(MarkerKind::DecimalLeadingZero(index)),
                Some("disc") | None => Some(MarkerKind::Disc),
                _ => None,
            };
            if let Some(kind) = kind {
                let size = font_size;
                let m = (0.35 * size).max(2.0);
                let color = css_color(style, props::COLOR).unwrap_or(Color::rgba8(0, 0, 0, 255));
                // Marker box just outside the border box's left
                // edge, near the top (approximation: no line-box
                // info at this level).
                b.push_list_marker(
                    Rect {
                        x: box_.rect.x - m - 0.5 * size,
                        y: box_.rect.y + 0.3 * size,
                        w: m,
                        h: m,
                    },
                    kind,
                    color,
                    index,
                    font_resolver(style),
                );
            }
        }
    }

    // Text runs: one TextRun per fragment, verbatim slice of the box's
    // (already transformed) text; decorations paint over each run.
    if box_.box_type == BoxType::Text
        && let (Some(text), Some(style)) = (box_.text.as_ref(), style)
    {
        let size = font_size;
        let color = css_color(style, props::COLOR).unwrap_or(Color::rgba8(0, 0, 0, 255));
        let font_key = font_resolver(style);
        let kinds = deco_kinds(style);
        let deco_color = match style.get("text-decoration-color") {
            Some(CssValue::Color(c)) => Color::rgba8(c.r, c.g, c.b, c.a),
            _ => color,
        };
        let thickness = (size * 0.08).max(1.0);
        // One shared allocation for the whole box text; each fragment's
        // TextRun clones the `Arc` (no per-fragment copy) and records its
        // byte range into it.
        let shared: Arc<str> = Arc::clone(text);
        for frag in &box_.fragments {
            let slice = verbatim_slice(text, frag);
            if slice.is_empty() {
                continue;
            }
            // `text-shadow` (inherited): each layer paints under the run.
            // Layers list front-to-back, so emit back-to-front.
            if let Some(CssValue::Shadows(layers)) = style.get_by_index(props::TEXT_SHADOW) {
                for s in layers.iter().rev() {
                    b.push_text_shadow(
                        Arc::clone(&shared),
                        frag.text_range.clone(),
                        frag.rect,
                        frag.baseline,
                        font_key,
                        size,
                        shadow_color(&s.color, color),
                        (
                            resolve_shadow_length(&s.offset_x, size, uctx.root_fs, uctx.viewport),
                            resolve_shadow_length(&s.offset_y, size, uctx.root_fs, uctx.viewport),
                        ),
                        resolve_shadow_length(&s.blur, size, uctx.root_fs, uctx.viewport).max(0.0),
                    );
                }
            }
            b.push_text_run(
                Arc::clone(&shared),
                frag.text_range.clone(),
                frag.rect,
                frag.baseline,
                font_key,
                size,
                color,
            );
            for kind in &kinds {
                b.push_decoration(frag.rect, *kind, deco_color, thickness);
            }
        }
    }

    // Non-visible `overflow` (not inherited): clip the children to the
    // box's border box. Uses the box's own style, not the inherited one,
    // so anonymous text boxes don't re-clip themselves.
    //
    // The clip is pushed lazily around the in-clip runs of the Appendix E
    // steps below (see the drain emission loop): drained out-of-flow boxes
    // anchored at a stacking context below their containing block paint
    // inside this box's opacity/blend group but outside this clip
    // (CSS 2.1 §11.1.1), while in-flow children — and boxes drained at
    // their containing block — paint inside it. The fast path has no
    // drains, so it keeps the single eager push/pop pair.
    let clips_children = box_.style.as_deref().is_some_and(overflow_clips);
    let clip_radii = if clips_children {
        box_.style
            .as_ref()
            .map(|own| resolve_clip_radii(own, font_size, uctx, box_.rect))
            .unwrap_or([0.0; 4])
    } else {
        [0.0; 4]
    };

    // The containing block for the out-of-flow descendants drained
    // below: this box when it is positioned, else the nearest positioned
    // ancestor above it (`None` = the viewport).
    let self_ptr = box_ as *const LayoutBox;
    let child_npa = if box_position(box_) == PositionAtom::Static {
        npa
    } else {
        Some(self_ptr)
    };

    // CSS 2.1 Appendix E paint order for this box's children. Step 1
    // (this box's own background/borders) was emitted above. The merged
    // step sequence below preserves Appendix E order exactly; the box's
    // own overflow clip toggles around it (see the drain emission loop):
    // in-flow children — and drained boxes whose anchor IS the containing
    // block — paint inside the clip, while drained boxes anchored at a
    // stacking context below their containing block paint inside this
    // box's opacity/blend group but outside the clip (CSS 2.1 §11.1.1:
    // only the containing block's overflow clips an out-of-flow
    // descendant).
    //
    // Out-of-flow descendants were deferred up front (see
    // `build_display_list`); the walk below only classifies in-flow
    // children — positioned ones (relative/sticky) join steps 2/6/7, the
    // rest split into block backgrounds (3), floats (4), and inline
    // content (5). Each child paints atomically: its whole subtree at
    // its slot.
    //
    // Known simplification, documented for the next pass: steps 3–5 do
    // not interleave a block's inline descendants with later floats
    // (each in-flow subtree paints atomically).
    //
    // A positioned child with non-`auto` z-index whose parent is not a
    // stacking context does not join the steps below: `discover_oof`
    // hoisted it to the nearest ancestor stacking context (Appendix E),
    // so it is skipped here and drains there.
    //
    // Fast path: ordinary children — no floats, no in-flow positioned
    // boxes, no flex/grid `order` reordering, and no deferred boxes
    // anchored at this containing block — paint in Appendix E step
    // order: first all in-flow block-level children (step 3), then all
    // in-flow inline-level children (step 5), each pass in DOM order,
    // with zero allocation. This is the hot path.
    let order_items = is_flex_or_grid_container(box_);
    let mut needs_steps = order_items;
    if !needs_steps {
        needs_steps = box_.children.iter().any(|c| {
            // Out-of-flow children were deferred up front and drain at
            // their containing block; they never affect in-flow order.
            // Opacity/blend stacking-context creators need the slow path
            // too: they paint at step 6, after floats and inlines.
            !is_out_of_flow(c)
                && (is_positioned(c) || is_float(c, box_) || creates_stacking_context(c))
        });
    }
    if !needs_steps {
        needs_steps =
            pending.has_entries_for(Some(self_ptr)) || (is_root && pending.has_entries_for(None));
    }
    if !needs_steps {
        // Two cheap in-order passes (no allocation): Appendix E paints
        // all in-flow block-level backgrounds (step 3) before all
        // in-flow inline content (step 5), even when block and inline
        // siblings interleave in DOM order. No drains anchor here, so the
        // clip stays a single eager push/pop pair around both passes —
        // except around an anchor-child subtree (see `escape_wraps_for`):
        // this box sits strictly between a drain anchor in that child's
        // subtree and the drained box's containing block, so its overflow
        // must not clip the drained box (CSS 2.1 §11.1.1).
        // Cloned: the loops below take `pending` mutably, so the borrow
        // cannot live across them (escape paths are rare; the vec is tiny).
        let escape_wraps: Option<Vec<*const LayoutBox>> = if clips_children {
            pending.escape_wraps_for(self_ptr).cloned()
        } else {
            None
        };
        if clips_children {
            push_own_clip(b, box_.rect, clip_radii);
        }
        let mut clip_open = clips_children;
        for pass in 0..2 {
            for child in &box_.children {
                if is_out_of_flow(child) {
                    // Deferred up front; drains at its containing block.
                    continue;
                }
                // Pass 0 paints step-3 (block-level) children, pass 1
                // paints step-5 (inline-level) children.
                if (pass == 0) != is_block_level(child) {
                    continue;
                }
                // §11.1.1: this child roots a drain anchor's escape path —
                // its subtree paints outside this box's own clip. Steps
                // merge atomically per child, so the wrap covers the
                // whole subtree (known simplification, as in the slow
                // path).
                let wraps_escape = escape_wraps
                    .as_ref()
                    .is_some_and(|w| w.contains(&(child as *const LayoutBox)));
                if clips_children {
                    if wraps_escape && clip_open {
                        b.pop_clip();
                        clip_open = false;
                    } else if !wraps_escape && !clip_open {
                        push_own_clip(b, box_.rect, clip_radii);
                        clip_open = true;
                    }
                }
                build_box(
                    b,
                    scratch,
                    child,
                    doc,
                    font_resolver,
                    image_resolver,
                    style,
                    UnitCtx {
                        parent_fs: font_size,
                        ..uctx
                    },
                    in_collapse,
                    child_npa,
                    pending,
                    false,
                    doc_order,
                );
            }
        }
        if clip_open {
            b.pop_clip();
        }
    } else {
        // Appendix E step partitioning into the build-wide step scratch
        // (see [`StepScratch`]): each in-flow child is classified and
        // tagged with its paint rank in one pass, the pending
        // out-of-flow drains append directly, and the new tail is
        // sorted in place below — no per-step Vecs, and no second copy
        // into a merged `paint_order`.
        //
        // Step ranks: 0 = negative-z stacking contexts (above this box's own
        // background, below in-flow content) · 1 = block backgrounds ·
        // 2 = floats · 3 = inline content · 4 = positioned z-index auto/0 ·
        // 5 = positive-z stacking contexts.
        let base = scratch.steps.len();
        for child in &box_.children {
            if is_out_of_flow(child) {
                // Already deferred by the up-front discovery pass; drained
                // into the z-index steps below.
                continue;
            }
            if is_positioned(child) && child.z_index.is_some() && !is_stacking_context(box_) {
                // Hoisted by the up-front discovery pass (Appendix E):
                // a positioned box with non-`auto` z-index paints in the
                // nearest ancestor stacking context, not in this box's
                // steps, when this box is not itself a stacking context.
                // It drains at the anchor's steps below. (A positioned
                // child always trips `needs_steps`, so the fast path
                // above never has to skip hoisted children.)
                continue;
            }
            let child_ctx = UnitCtx {
                parent_fs: font_size,
                ..uctx
            };
            // `z_index` is `Some` exactly for positioned boxes with
            // non-`auto` z-index and for static flex/grid items with
            // non-`auto` z-index (layout preserves it for those) — i.e.
            // exactly the child stacking contexts, which sort into the
            // negative/zero/positive runs. Opacity/blend
            // stacking-context creators have `z_index: None` and land in
            // the zero run (step 6, tree order).
            let rank = if joins_z_steps(child) {
                match child.z_index {
                    Some(z) if z < 0 => 0,
                    Some(z) if z > 0 => 5,
                    _ => 4,
                }
            } else if is_float(child, box_) {
                2
            } else if is_block_level(child) {
                1
            } else {
                3
            };
            scratch.steps.push((
                rank,
                StepChild::InFlow {
                    box_: child,
                    uctx: child_ctx,
                },
            ));
        }
        pending.drain_into(
            Some(self_ptr),
            |z| z.is_some_and(|z| z < 0),
            0,
            &mut scratch.steps,
        );
        pending.drain_into(
            Some(self_ptr),
            |z| z.is_none_or(|z| z == 0),
            4,
            &mut scratch.steps,
        );
        pending.drain_into(
            Some(self_ptr),
            |z| z.is_some_and(|z| z > 0),
            5,
            &mut scratch.steps,
        );
        if is_root {
            // Viewport-anchored boxes (`position: fixed`, or `absolute` with
            // no positioned ancestor) belong to the root stacking context's
            // own steps — negative-z above the root background but below
            // in-flow content — not to the paint-last drain below.
            pending.drain_into(None, |z| z.is_some_and(|z| z < 0), 0, &mut scratch.steps);
            pending.drain_into(None, |z| z.is_none_or(|z| z == 0), 4, &mut scratch.steps);
            pending.drain_into(None, |z| z.is_some_and(|z| z > 0), 5, &mut scratch.steps);
        }
        // One merged paint sequence: (Appendix E step, z-index, `order`,
        // document order). `sort_by_key` is stable, and equal-key entries
        // keep the classification order of the old per-step Vecs —
        // in-flow children in child order, drained out-of-flow boxes
        // after them — so the sequence is identical to the old merged
        // `paint_order`.
        scratch.steps[base..]
            .sort_by_key(|(rank, sc)| (*rank, sc.sort_key(order_items, doc_order)));
        // Drain emission with §11.1.1 clip placement. The merged order
        // above is emitted verbatim — Appendix E rank, then z-index,
        // `order`, document order — so a drained box keeps its exact slot
        // relative to same-rank in-flow steps; only the clip state moves:
        // drained boxes anchored at a stacking context below their
        // containing block (`escape_anchor_clip`) paint inside this box's
        // opacity/blend group but OUTSIDE this box's own overflow clip,
        // while in-flow children — and boxes drained at their containing
        // block — paint INSIDE it. An in-flow child whose subtree
        // contains such a drain's anchor (see `escape_wraps_for`)
        // likewise paints outside this box's clip: this box sits
        // strictly between the anchor and the containing block, so its
        // overflow must not clip the drained box (CSS 2.1 §11.1.1).
        // Hoisted entries additionally re-apply their captured
        // `outer_clips` (the clips they were lifted out of). The clip
        // toggles around the out-of-clip runs; every push is balanced by
        // a pop.
        //
        // §11.1.1 intermediate wraps for this box (`None` when it clips
        // nothing or anchors no escape path): the direct children whose
        // subtree contains a drain anchor with this box strictly between
        // the anchor and the containing block.
        // Cloned: the loops below take `pending` mutably, so the borrow
        // cannot live across them (escape paths are rare; the vec is tiny).
        let escape_wraps: Option<Vec<*const LayoutBox>> = if clips_children {
            pending.escape_wraps_for(self_ptr).cloned()
        } else {
            None
        };
        let mut clip_open = false;
        // The sorted tail is reversed in place and popped: each child
        // paints from an owned `StepChild` with no borrow of the scratch
        // held across the recursive `paint_step_child` call, so nested
        // stacking contexts can push their own tail after ours and pop
        // it back on return. The loop drains exactly to `base`, leaving
        // the scratch empty for the next context.
        scratch.steps[base..].reverse();
        while scratch.steps.len() > base {
            let (_, sc) = scratch.steps.pop().expect("loop bound guarantees an entry");
            let (escapes_clip, outer_clips) = match &sc {
                StepChild::Oof(p) => (p.escape_anchor_clip, p.outer_clips.as_slice()),
                // Steps merge atomically per child, so the wrap covers
                // the whole anchor-child subtree — including its in-flow
                // content, which strictly speaking stays clipped by this
                // box's overflow (known simplification).
                StepChild::InFlow { box_, .. } => (
                    escape_wraps
                        .as_ref()
                        .is_some_and(|w| w.contains(&(*box_ as *const LayoutBox))),
                    &[] as &[(Rect, [f32; 4])],
                ),
            };
            if clips_children {
                if escapes_clip && clip_open {
                    b.pop_clip();
                    clip_open = false;
                } else if !escapes_clip && !clip_open {
                    push_own_clip(b, box_.rect, clip_radii);
                    clip_open = true;
                }
            }
            // Re-apply the overflow clips a hoisted entry was lifted out
            // of (see `PendingOof::outer_clips`); each push is balanced
            // after the paint below. Only the count crosses the
            // `paint_step_child` move — the borrow of `sc` must end
            // before it.
            for &(rect, radii) in outer_clips {
                push_own_clip(b, rect, radii);
            }
            let outer_clip_count = outer_clips.len();
            // Inheritance context: the DOM parent's style/units, captured at
            // the classification site above (or the deferral site for
            // out-of-flow boxes) — not this containing block's.
            paint_step_child(
                b,
                scratch,
                sc,
                doc,
                font_resolver,
                image_resolver,
                style,
                in_collapse,
                child_npa,
                pending,
                doc_order,
            );
            for _ in 0..outer_clip_count {
                b.pop_clip();
            }
        }
        if clip_open {
            b.pop_clip();
        }
    }

    if filtered {
        b.pop_filter();
    }

    if backdrop_filtered {
        b.pop_backdrop_filter();
    }

    if clipped {
        b.pop_clip();
    }

    if transform.is_some() {
        b.pop_transform();
    }

    if grouped {
        b.pop_opacity();
    }

    if blended {
        b.pop_blend();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradient::GradientStops;
    use bowser_layout::BoxType;

    /// Builds a one-element box whose computed style comes from `decl`
    /// (a `position: <v>` declaration), or from no rule at all when
    /// `decl` is `None`.
    fn position_test_box(decl: Option<&str>) -> LayoutBox {
        use bowser_css::{Viewport, compute_style, default_ua_stylesheet, parse_stylesheet};

        let mut doc = Document::new();
        let root = doc.root();
        let el = doc.create_element("div");
        doc.append_child(root, el);
        let css = decl.map(|v| format!("div {{ position: {v}; }}"));
        let author = parse_stylesheet(css.as_deref().unwrap_or(""));
        let vp = Viewport { w: 800.0, h: 600.0 };
        let style = compute_style(&doc, el, &[&author], &default_ua_stylesheet(), &vp);
        LayoutBox {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w: 0.0,
                h: 0.0,
            },
            children: Vec::new(),
            box_type: BoxType::Block,
            style: Some(std::rc::Rc::new(style)),
            text: None,
            node: Some(el),
            fragments: Vec::new(),
            z_index: None,
            collapsed_borders: None,
            phantom_w: false,
            phantom_h: false,
        }
    }

    #[test]
    fn box_position_atom_matches_string_semantics() {
        // (declaration, atom, out-of-flow, positioned): pins the atom
        // conversion against the old string-match behavior for every
        // representative `position` value, including case-variant
        // spellings and an unrecognized keyword. Case-variants reach the
        // computed style ASCII-lowercased by the cascade (the value
        // parser lowercases idents into `CssValue::Keyword`), so
        // `"STATIC"` behaves exactly like `"static"` and `"ABSOLUTE"`
        // like `"absolute"`.
        let cases: &[(&str, PositionAtom, bool, bool)] = &[
            ("static", PositionAtom::Static, false, false),
            ("STATIC", PositionAtom::Static, false, false),
            ("relative", PositionAtom::Other, false, true),
            ("absolute", PositionAtom::Absolute, true, true),
            ("ABSOLUTE", PositionAtom::Absolute, true, true),
            ("fixed", PositionAtom::Fixed, true, true),
            ("sticky", PositionAtom::Other, false, true),
            // Invalid keyword: the declaration fails validation and is
            // dropped, so the computed value is the initial `static`.
            ("bogus", PositionAtom::Static, false, false),
        ];
        for (decl, atom, oof, positioned) in cases {
            let b = position_test_box(Some(decl));
            assert_eq!(box_position(&b), *atom, "atom for position: {decl}");
            assert_eq!(is_out_of_flow(&b), *oof, "out-of-flow for position: {decl}");
            assert_eq!(
                is_positioned(&b),
                *positioned,
                "positioned for position: {decl}"
            );
        }
        // Missing `position` reads as `static`, like a style-less box.
        let missing = position_test_box(None);
        assert_eq!(box_position(&missing), PositionAtom::Static);
        assert!(!is_out_of_flow(&missing));
        assert!(!is_positioned(&missing));
    }

    #[test]
    fn color_rgba8_scales() {
        let c = Color::rgba8(255, 128, 0, 255);
        assert_eq!((c.r8(), c.g8(), c.b8(), c.a8()), (255, 128, 0, 255));
        assert_eq!((c.r(), c.g(), c.b(), c.a()), (1.0, 128.0 / 255.0, 0.0, 1.0));
    }

    #[test]
    fn opacity_multiplies_alpha_only() {
        let c = Color::rgba8(10, 20, 30, 200).with_opacity(0.5);
        assert_eq!(c.a8(), 100);
        assert!((c.a() - 100.0 / 255.0).abs() < 1e-6);
        assert_eq!(
            (c.r(), c.g(), c.b()),
            (10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0)
        );
    }

    #[test]
    fn display_list_image_roundtrip() {
        let mut dl = DisplayList::new();
        let id = ImageId(7);
        dl.add_image(id, DecodedImage::from_rgba(1, 1, vec![1, 2, 3, 4]));
        assert_eq!(dl.image_count(), 1);
        let img = dl.image(id).expect("registered");
        assert_eq!((img.width, img.height), (1, 1));
        assert!(dl.image(ImageId(8)).is_none());
    }

    #[test]
    fn decoded_image_validates_length() {
        let ok = DecodedImage::from_rgba(2, 1, vec![0u8; 8]);
        assert_eq!((ok.width, ok.height), (2, 1));
    }

    #[test]
    #[should_panic]
    fn decoded_image_rejects_bad_length() {
        let _ = DecodedImage::from_rgba(2, 1, vec![0u8; 7]);
    }

    #[test]
    fn border_item_matches_tessellator_signature() {
        // The Border item carries per-side arrays so it feeds
        // borders::tessellate_border without adaptation.
        let item = DisplayItem::Border {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w: 10.0,
                h: 10.0,
            },
            widths: [2.0, 2.0, 2.0, 2.0],
            styles: [BorderStyle::Solid; 4],
            colors: [Color::rgba8(255, 0, 0, 255); 4],
            radii: [3.0, 0.0, 0.0, 0.0],
        };
        if let DisplayItem::Border {
            rect,
            widths,
            styles,
            colors,
            radii,
        } = item
        {
            let quads = crate::borders::tessellate_border(rect, widths, styles, colors);
            assert!(!quads.is_empty());
            // The radii ride along for the rounded tessellator.
            let rounded =
                crate::borders::tessellate_border_rounded(rect, widths, styles, colors, radii);
            assert!(!rounded.is_empty());
        } else {
            panic!("expected Border");
        }
    }

    #[test]
    fn builder_push_build_keeps_sequence() {
        let r = Rect {
            x: 1.0,
            y: 2.0,
            w: 3.0,
            h: 4.0,
        };
        let mut b = DisplayListBuilder::new();
        b.push_fill_rect(r, Color::rgba8(255, 0, 0, 255))
            .push_text_run(
                Arc::from("hi"),
                0..2,
                r,
                10.0,
                FontKey(3),
                16.0,
                Color::rgba8(0, 0, 0, 255),
            )
            .push_clip(r);
        b.pop_clip();
        let img_id = b.add_image(DecodedImage::from_rgba(1, 1, vec![9, 9, 9, 9]));
        b.push_image(r, img_id).push_opacity(0.5);
        b.pop_opacity();
        let list = b.build();
        // FillRect, TextRun, PushClip, PopClip, Image, PushOpacity,
        // PopOpacity.
        assert_eq!(list.items.len(), 7);
        assert!(matches!(list.items[0], DisplayItem::FillRect { .. }));
        match &list.items[1] {
            DisplayItem::TextRun {
                text,
                text_range,
                baseline,
                font_key,
                size_px,
                ..
            } => {
                assert_eq!(text.get(text_range.clone()), Some("hi"));
                assert_eq!(text_range, &(0..2));
                assert_eq!(*baseline, 10.0);
                assert_eq!(*font_key, FontKey(3));
                assert_eq!(*size_px, 16.0);
            }
            other => panic!("expected TextRun, got {other:?}"),
        }
        assert!(matches!(list.items[2], DisplayItem::PushClip { .. }));
        assert!(matches!(list.items[3], DisplayItem::PopClip));
        assert!(matches!(
            list.items[4],
            DisplayItem::Image { image, .. } if image == img_id
        ));
        assert!(matches!(
            list.items[5],
            DisplayItem::PushOpacity { .. } | DisplayItem::PopOpacity
        ));
        assert!(list.image(img_id).is_some());
    }

    // -- layout-driven tests: parse, style, lay out, build, assert ------

    fn layout_and_build(html: &str, vw: f32, vh: f32) -> (Document, LayoutBox, DisplayList) {
        let (doc, _quirks) = Document::parse_html(html.as_bytes());
        let author = bowser_css::parse_stylesheet("");
        let viewport = Size { w: vw, h: vh };
        let root =
            bowser_layout::layout(&doc, &[&author], viewport, &bowser_layout::FixedFontMetrics);
        let list = build_display_list_with_images(&root, &doc, &mut |_| None, viewport);
        (doc, root, list)
    }

    fn text_runs(list: &DisplayList) -> Vec<(String, Rect, f32)> {
        list.items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::TextRun {
                    text,
                    text_range,
                    rect,
                    baseline,
                    ..
                } => Some((
                    text.get(text_range.clone()).unwrap_or("").to_owned(),
                    *rect,
                    *baseline,
                )),
                _ => None,
            })
            .collect()
    }

    /// Find the layout box generated for `node`, if any.
    fn find_box_by_node(b: &LayoutBox, node: NodeId) -> Option<&LayoutBox> {
        if b.node == Some(node) {
            return Some(b);
        }
        b.children.iter().find_map(|c| find_box_by_node(c, node))
    }

    fn collect_text_boxes(b: &LayoutBox, out: &mut Vec<(String, Vec<TextFragment>)>) {
        if b.box_type == BoxType::Text
            && let Some(t) = &b.text
        {
            out.push((t.to_string(), b.fragments.clone()));
        }
        for c in &b.children {
            collect_text_boxes(c, out);
        }
    }

    // -- out-of-flow paint order and clipping (fixes 3/4) ----------------

    fn layout_and_build_css(html: &str, css: &str, vw: f32, vh: f32) -> DisplayList {
        let (doc, _quirks) = Document::parse_html(html.as_bytes());
        let author = bowser_css::parse_stylesheet(css);
        let viewport = Size { w: vw, h: vh };
        let root =
            bowser_layout::layout(&doc, &[&author], viewport, &bowser_layout::FixedFontMetrics);
        build_display_list_with_images(&root, &doc, &mut |_| None, viewport)
    }

    // -- column-rule rendering --------------------------------------------

    /// The column-rule emission shape (see `paint_column_rules`): border
    /// items carrying only a left side, as `(rect, width, style, color)`.
    fn column_rule_items(list: &DisplayList) -> Vec<(Rect, f32, BorderStyle, Color)> {
        list.items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::Border {
                    rect,
                    widths,
                    styles,
                    colors,
                    ..
                } if widths[0] == 0.0
                    && widths[1] == 0.0
                    && widths[2] == 0.0
                    && widths[3] > 0.0
                    && styles[3] != BorderStyle::None =>
                {
                    Some((*rect, widths[3], styles[3], colors[3]))
                }
                _ => None,
            })
            .collect()
    }

    /// `column-rule: 2px solid red` on a 3-column container with block
    /// children (the block distribution path, matched by DOM node)
    /// emits one rule item per column gap: 2px wide, solid red, centered
    /// on each gap center, spanning the container's content height.
    #[test]
    fn column_rule_block_children_emit_two_solid_rules() {
        let _ = take_column_rules(); // drain stragglers from other tests
        let (doc, _quirks) = Document::parse_html(
            r#"<div class="mc"><div class="a"></div><div class="b"></div><div class="c"></div></div>"#
                .as_bytes(),
        );
        let author = bowser_css::parse_stylesheet(
            ".mc { width: 630px; column-count: 3; column-gap: 30px; column-rule: 2px solid red; } \
             .a, .b, .c { height: 40px; }",
        );
        let viewport = Size { w: 800.0, h: 600.0 };
        let root =
            bowser_layout::layout(&doc, &[&author], viewport, &bowser_layout::FixedFontMetrics);
        let list = build_display_list_with_images(&root, &doc, &mut |_| None, viewport);

        let mc = doc.get_elements_by_tag_name("div").item(0).expect("mc div");
        let mc_box = find_box_by_node(&root, mc).expect("mc layout box");

        let rules = column_rule_items(&list);
        assert_eq!(rules.len(), 2, "one rule per column gap, got {rules:?}");
        // col_w = (630 - 2*30) / 3 = 190; gap centers at 190 + 15 = 205
        // and 205 + 190 + 30 = 425, relative to the content-box left
        // edge. The container has no border/padding, so the content box
        // is the border box; each rule is centered on its gap center.
        let red = Color::rgba8(255, 0, 0, 255);
        for (i, want_center) in [205.0f32, 425.0].iter().enumerate() {
            let (rect, width, style, color) = &rules[i];
            assert_eq!(*width, 2.0, "rule {i} width");
            assert_eq!(*style, BorderStyle::Solid, "rule {i} style");
            assert_eq!(*color, red, "rule {i} color");
            assert_eq!(rect.w, 2.0, "rule {i} rect width");
            assert!(
                (rect.x - (mc_box.rect.x + want_center - 1.0)).abs() < 0.01,
                "rule {i} centered on gap center {want_center}, got {rect:?}"
            );
            assert!(
                (rect.y - mc_box.rect.y).abs() < 0.01,
                "rule {i} starts at the content-box top, got {rect:?}"
            );
            // Three 40px children balance one per column: 40px tall.
            assert!(
                (rect.h - 40.0).abs() < 0.01,
                "rule {i} spans the content height, got {rect:?}"
            );
        }
    }

    /// The same rule on a container with inline content (the inline
    /// distribution path, which records no DOM node): the orphan spec is
    /// matched by geometry and still paints both rules.
    #[test]
    fn column_rule_inline_content_matches_by_geometry() {
        let _ = take_column_rules(); // drain stragglers from other tests
        let (doc, _quirks) = Document::parse_html(
            r#"<div class="mc">lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua</div>"#.as_bytes(),
        );
        let author = bowser_css::parse_stylesheet(
            ".mc { width: 630px; column-count: 3; column-gap: 30px; column-rule: 2px solid red; }",
        );
        let viewport = Size { w: 800.0, h: 600.0 };
        let root =
            bowser_layout::layout(&doc, &[&author], viewport, &bowser_layout::FixedFontMetrics);
        let list = build_display_list_with_images(&root, &doc, &mut |_| None, viewport);

        let mc = doc.get_elements_by_tag_name("div").item(0).expect("mc div");
        let mc_box = find_box_by_node(&root, mc).expect("mc layout box");

        let rules = column_rule_items(&list);
        assert_eq!(rules.len(), 2, "one rule per column gap, got {rules:?}");
        let red = Color::rgba8(255, 0, 0, 255);
        for (i, want_center) in [205.0f32, 425.0].iter().enumerate() {
            let (rect, width, style, color) = &rules[i];
            assert_eq!(*width, 2.0, "rule {i} width");
            assert_eq!(*style, BorderStyle::Solid, "rule {i} style");
            assert_eq!(*color, red, "rule {i} color");
            assert!(
                (rect.x - (mc_box.rect.x + want_center - 1.0)).abs() < 0.01,
                "rule {i} centered on gap center {want_center}, got {rect:?}"
            );
            assert!(
                (rect.y - mc_box.rect.y).abs() < 0.01,
                "rule {i} starts at the content-box top, got {rect:?}"
            );
            assert!(
                (rect.h - mc_box.rect.h).abs() < 0.05,
                "rule {i} spans the content height, got {rect:?}"
            );
        }
    }

    /// `column-rule-style: none` records no spec, so nothing paints.
    #[test]
    fn column_rule_none_style_paints_nothing() {
        let _ = take_column_rules(); // drain stragglers from other tests
        let list = layout_and_build_css(
            r#"<div class="mc"><div class="a"></div><div class="b"></div></div>"#,
            ".mc { width: 420px; column-count: 2; column-gap: 20px; column-rule: 2px none red; } \
             .a, .b { height: 30px; }",
            800.0,
            600.0,
        );
        assert!(
            column_rule_items(&list).is_empty(),
            "no rule items for `none` style"
        );
    }

    /// Mixed-case `column-rule` keywords resolve the same as lowercase:
    /// `THICK DASHED` paints a 5px dashed rule.
    #[test]
    fn column_rule_keywords_are_ascii_case_insensitive() {
        let _ = take_column_rules(); // drain stragglers from other tests
        let list = layout_and_build_css(
            r#"<div class="mc"><div class="a"></div><div class="b"></div></div>"#,
            ".mc { width: 420px; column-count: 2; column-gap: 20px; column-rule: THICK DASHED red; } \
             .a, .b { height: 30px; }",
            800.0,
            600.0,
        );
        let rules = column_rule_items(&list);
        assert_eq!(rules.len(), 1, "one rule for the single gap, got {rules:?}");
        let (_, width, style, _) = &rules[0];
        assert_eq!(*width, 5.0, "THICK -> 5px");
        assert_eq!(*style, BorderStyle::Dashed, "DASHED style");
    }

    /// `background-color` fills as `(r, g, b)` bytes, in paint order.
    fn fill_colors(list: &DisplayList) -> Vec<(u8, u8, u8)> {
        list.items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::FillRect { color, .. } => Some((color.r8(), color.g8(), color.b8())),
                _ => None,
            })
            .collect()
    }

    /// Whether the blue (`#abs`) fill is painted under any clip.
    fn blue_is_clipped(list: &DisplayList) -> bool {
        let mut clip_depth = 0;
        let mut clipped = false;
        for it in &list.items {
            match it {
                DisplayItem::PushClip { .. } | DisplayItem::PushRoundedClip { .. } => {
                    clip_depth += 1
                }
                DisplayItem::PopClip => clip_depth -= 1,
                DisplayItem::FillRect { color, .. } if color.b8() == 255 => {
                    clipped = clip_depth > 0;
                }
                _ => {}
            }
        }
        clipped
    }

    /// CSS 2.1 Appendix E: positioned descendants paint above in-flow
    /// content, whatever the DOM order.
    #[test]
    fn abspos_paints_after_flow_content() {
        let html = r#"<div id="abs"></div><div id="flow"></div>"#;
        let css = "#abs { position: absolute; top: 0; left: 0; width: 10px; height: 10px; background-color: blue; } \
             #flow { width: 10px; height: 10px; background-color: green; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let flow = colors
            .iter()
            .position(|&c| c == (0, 128, 0))
            .expect("flow fill");
        let abs = colors
            .iter()
            .position(|&c| c == (0, 0, 255))
            .expect("abs fill");
        assert!(abs > flow, "abs {abs} must paint after flow {flow}");
    }

    /// `position: fixed` is viewport-anchored: it paints above page
    /// content even when it comes first in the DOM.
    #[test]
    fn fixed_paints_above_page_content() {
        let html = r#"<div id="fix"></div><div id="flow"></div>"#;
        let css = "#fix { position: fixed; top: 0; left: 0; width: 10px; height: 10px; background-color: blue; } \
             #flow { width: 10px; height: 10px; background-color: green; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let flow = colors
            .iter()
            .position(|&c| c == (0, 128, 0))
            .expect("flow fill");
        let fix = colors
            .iter()
            .position(|&c| c == (0, 0, 255))
            .expect("fixed fill");
        assert!(fix > flow, "fixed {fix} must paint after flow {flow}");
    }

    // -- CSS 2.1 Appendix E: z-index and `order` paint order ----------

    /// Sibling stacking contexts paint in ascending z-index order, not
    /// DOM order (Appendix E step 7).
    #[test]
    fn z_index_orders_overlapping_abspos() {
        let html = r#"<div id="a"></div><div id="b"></div>"#;
        let css = "#a { position: absolute; z-index: 2; top: 0; left: 0; width: 10px; height: 10px; background-color: red; } \
             #b { position: absolute; z-index: 1; top: 0; left: 0; width: 10px; height: 10px; background-color: blue; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let a = colors
            .iter()
            .position(|&c| c == (255, 0, 0))
            .expect("a fill");
        let b = colors
            .iter()
            .position(|&c| c == (0, 0, 255))
            .expect("b fill");
        assert!(a > b, "z-index 2 ({a}) must paint after z-index 1 ({b})");
    }

    /// A negative z-index child hoists to its nearest ancestor stacking
    /// context (Appendix E step 2): `#wrap` is `position: relative` with
    /// `z-index: auto`, so it is NOT a stacking context, and `#neg`
    /// (z-index -1) paints in the root stacking context — above the root
    /// background but below `#wrap`'s own background and in-flow content.
    /// (The classic "negative z-index goes behind its non-stacking-context
    /// parent" case.)
    #[test]
    fn negative_z_index_paints_behind_flow_content() {
        let html = r#"<div id="wrap"><div id="neg"></div><div id="flow"></div></div>"#;
        let css = "#wrap { position: relative; background-color: white; } \
             #neg { position: absolute; z-index: -1; top: 0; left: 0; width: 10px; height: 10px; background-color: red; } \
             #flow { width: 10px; height: 10px; background-color: green; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let wrap = colors
            .iter()
            .position(|&c| c == (255, 255, 255))
            .expect("wrap fill");
        let neg = colors
            .iter()
            .position(|&c| c == (255, 0, 0))
            .expect("neg fill");
        let flow = colors
            .iter()
            .position(|&c| c == (0, 128, 0))
            .expect("flow fill");
        assert!(
            neg < wrap,
            "z-index -1 ({neg}) hoists past the non-stacking-context parent, behind its background ({wrap})"
        );
        assert!(
            neg < flow,
            "z-index -1 ({neg}) must paint before in-flow content ({flow})"
        );
    }

    /// A positioned descendant with z-index `auto` paints above
    /// non-positioned content even when earlier in the DOM (Appendix E
    /// step 6 vs steps 3-5).
    #[test]
    fn relative_z_auto_paints_above_static_content() {
        let html = r#"<div id="rel"></div><div id="flow"></div>"#;
        let css = "#rel { position: relative; width: 10px; height: 10px; background-color: red; } \
             #flow { width: 10px; height: 10px; background-color: green; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let rel = colors
            .iter()
            .position(|&c| c == (255, 0, 0))
            .expect("rel fill");
        let flow = colors
            .iter()
            .position(|&c| c == (0, 128, 0))
            .expect("flow fill");
        assert!(
            rel > flow,
            "positioned z-index auto ({rel}) must paint after in-flow content ({flow})"
        );
    }

    /// `opacity < 1` creates a stacking context (css-color-3 §3.2): `#a`
    /// paints at Appendix E step 6 (stack level 0, tree order) above the
    /// positioned `#b`, not at step 3 with the block backgrounds.
    #[test]
    fn opacity_creates_stacking_context_above_positioned_sibling() {
        let html = r#"<div id="b"></div><div id="a"></div>"#;
        let css = "#b { position: relative; width: 10px; height: 10px; background-color: blue; } \
             #a { opacity: 0.5; width: 10px; height: 10px; background-color: red; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let a = colors
            .iter()
            .position(|&c| c == (255, 0, 0))
            .expect("a fill");
        let b = colors
            .iter()
            .position(|&c| c == (0, 0, 255))
            .expect("b fill");
        assert!(
            a > b,
            "opacity stacking context ({a}) must paint after positioned sibling ({b})"
        );
    }

    /// A non-`normal` `mix-blend-mode` creates a stacking context
    /// (css-compositing-1 §4): same step-6 placement as `opacity < 1`.
    #[test]
    fn blend_mode_creates_stacking_context_above_positioned_sibling() {
        let html = r#"<div id="b"></div><div id="a"></div>"#;
        let css = "#b { position: relative; width: 10px; height: 10px; background-color: blue; } \
             #a { mix-blend-mode: multiply; width: 10px; height: 10px; background-color: red; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let a = colors
            .iter()
            .position(|&c| c == (255, 0, 0))
            .expect("a fill");
        let b = colors
            .iter()
            .position(|&c| c == (0, 0, 255))
            .expect("b fill");
        assert!(
            a > b,
            "blend-mode stacking context ({a}) must paint after positioned sibling ({b})"
        );
    }

    /// CSS Transforms 1 §6: a non-`none` transform creates a stacking
    /// context, painting above the positioned sibling at Appendix E
    /// step 6 — even when the transform evaluates to the identity
    /// matrix (`translate(0px, 0px)`), which emits no `PushTransform`
    /// group but still counts for stacking.
    #[test]
    fn transform_creates_stacking_context_above_positioned_sibling() {
        let html = r#"<div id="b"></div><div id="a"></div>"#;
        let css = "#b { position: relative; width: 10px; height: 10px; background-color: blue; } \
             #a { transform: translate(0px, 0px); width: 10px; height: 10px; background-color: red; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let a = colors
            .iter()
            .position(|&c| c == (255, 0, 0))
            .expect("a fill");
        let b = colors
            .iter()
            .position(|&c| c == (0, 0, 255))
            .expect("b fill");
        assert!(
            a > b,
            "transform stacking context ({a}) must paint after positioned sibling ({b})"
        );
        assert!(
            !list
                .items
                .iter()
                .any(|it| matches!(it, DisplayItem::PushTransform { .. })),
            "identity transform must not emit a PushTransform group"
        );
    }

    /// A real transform wraps the box's items in a
    /// `PushTransform`/`PopTransform` pair carrying the evaluated
    /// matrix, inside the opacity/blend groups.
    #[test]
    fn transform_emits_push_transform_group() {
        let html = r#"<div id="a"></div>"#;
        let css = "#a { transform: translate(10px, 20px); width: 10px; height: 10px; background-color: red; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let push = list
            .items
            .iter()
            .position(|it| matches!(it, DisplayItem::PushTransform { .. }))
            .expect("PushTransform");
        let pop = list
            .items
            .iter()
            .position(|it| matches!(it, DisplayItem::PopTransform))
            .expect("PopTransform");
        assert!(pop > push, "PopTransform must close PushTransform");
        match list.items[push] {
            DisplayItem::PushTransform { matrix } => {
                assert_eq!(matrix, [1.0, 0.0, 0.0, 1.0, 10.0, 20.0]);
            }
            _ => unreachable!(),
        }
        // The red fill paints inside the transform group.
        let fill = list
            .items
            .iter()
            .position(|it| matches!(it, DisplayItem::FillRect { color, .. } if color.r8() == 255))
            .expect("red fill");
        assert!(
            push < fill && fill < pop,
            "fill must paint inside the transform group"
        );
    }

    /// `transform: none` (and an absent transform) emits no group.
    #[test]
    fn transform_none_emits_no_group() {
        for css in [
            "#a { transform: none; width: 10px; height: 10px; background-color: red; }",
            "#a { width: 10px; height: 10px; background-color: red; }",
        ] {
            let list = layout_and_build_css(r#"<div id="a"></div>"#, css, 800.0, 600.0);
            assert!(
                !list.items.iter().any(|it| matches!(
                    it,
                    DisplayItem::PushTransform { .. } | DisplayItem::PopTransform
                )),
                "no transform group expected for {css}"
            );
        }
    }

    /// `transform-origin` pivots the matrix: `scale(2)` about `left
    /// top` is the pure scale about the border box's top-left corner.
    #[test]
    fn transform_origin_pivots_matrix() {
        let html = r#"<div id="a"></div>"#;
        let css = "#a { transform: scale(2); transform-origin: left top; width: 10px; height: 10px; background-color: red; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let matrix = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::PushTransform { matrix } => Some(*matrix),
                _ => None,
            })
            .next()
            .expect("PushTransform");
        let rect = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::FillRect { rect, .. } => Some(*rect),
                _ => None,
            })
            .next()
            .expect("red fill rect");
        // T(origin) · S(2) · T(−origin), origin = the fill's top-left:
        // (x, y) -> (2x − ox, 2y − oy).
        assert_eq!(matrix[0], 2.0);
        assert_eq!(matrix[1], 0.0);
        assert_eq!(matrix[2], 0.0);
        assert_eq!(matrix[3], 2.0);
        assert!(
            (matrix[4] + rect.x).abs() < 1e-4,
            "e = {}, rect.x = {}",
            matrix[4],
            rect.x
        );
        assert!(
            (matrix[5] + rect.y).abs() < 1e-4,
            "f = {}, rect.y = {}",
            matrix[5],
            rect.y
        );
    }

    /// `rotate(90deg)` about the default origin (center): the matrix
    /// pivots around the border box's center, not the page origin.
    #[test]
    fn transform_rotate_about_center() {
        let html = r#"<div id="a"></div>"#;
        let css =
            "#a { transform: rotate(90deg); width: 20px; height: 10px; background-color: red; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let matrix = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::PushTransform { matrix } => Some(*matrix),
                _ => None,
            })
            .next()
            .expect("PushTransform");
        // cos90 ≈ 0, sin90 = 1: (x, y) -> (cx - (y - cy), cy + (x - cx)).
        assert!(matrix[0].abs() < 1e-5, "a ≈ 0, got {}", matrix[0]);
        assert!((matrix[1] - 1.0).abs() < 1e-5, "b ≈ 1, got {}", matrix[1]);
        assert!((matrix[2] + 1.0).abs() < 1e-5, "c ≈ -1, got {}", matrix[2]);
        assert!(matrix[3].abs() < 1e-5, "d ≈ 0, got {}", matrix[3]);
        // The center maps to itself: M(cx, cy) == (cx, cy).
        let rect = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::FillRect { rect, .. } => Some(*rect),
                _ => None,
            })
            .next()
            .expect("fill rect");
        let (cx, cy) = (rect.x + rect.w / 2.0, rect.y + rect.h / 2.0);
        let (px, py) = (
            matrix[0] * cx + matrix[2] * cy + matrix[4],
            matrix[1] * cx + matrix[3] * cy + matrix[5],
        );
        assert!(
            (px - cx).abs() < 1e-3 && (py - cy).abs() < 1e-3,
            "center ({cx}, {cy}) must map to itself, got ({px}, {py})"
        );
    }

    /// A static flex item with non-`auto` `z-index` forms a stacking
    /// context (css-flexbox-1) painting at Appendix E step 7, above the
    /// positioned `z-index: auto` sibling.
    #[test]
    fn static_flex_item_z_index_paints_above_positioned_sibling() {
        let html = r#"<div id="c"><div id="a"></div><div id="b"></div></div>"#;
        let css = "#c { display: flex; width: 100px; height: 10px; } \
             #a { z-index: 1; width: 10px; height: 10px; background-color: red; } \
             #b { position: relative; width: 10px; height: 10px; background-color: blue; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let a = colors
            .iter()
            .position(|&c| c == (255, 0, 0))
            .expect("a fill");
        let b = colors
            .iter()
            .position(|&c| c == (0, 0, 255))
            .expect("b fill");
        assert!(
            a > b,
            "static flex item z-index 1 ({a}) must paint after positioned z-index auto sibling ({b})"
        );
    }

    /// Same for grid items (css-grid-1 §4.2).
    #[test]
    fn static_grid_item_z_index_paints_above_positioned_sibling() {
        let html = r#"<div id="c"><div id="a"></div><div id="b"></div></div>"#;
        let css = "#c { display: grid; width: 100px; height: 10px; } \
             #a { z-index: 1; width: 10px; height: 10px; background-color: red; } \
             #b { position: relative; width: 10px; height: 10px; background-color: blue; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let a = colors
            .iter()
            .position(|&c| c == (255, 0, 0))
            .expect("a fill");
        let b = colors
            .iter()
            .position(|&c| c == (0, 0, 255))
            .expect("b fill");
        assert!(
            a > b,
            "static grid item z-index 1 ({a}) must paint after positioned z-index auto sibling ({b})"
        );
    }

    /// A nested stacking context paints atomically at its own z-index:
    /// `#inner` (z-index 100 inside `#outer` at z-index 1) still paints
    /// below the z-index 2 sibling.
    #[test]
    fn nested_stacking_context_paints_atomically() {
        let html = r#"<div id="sib"></div><div id="outer"><div id="inner"></div></div>"#;
        let css = "#sib { position: absolute; z-index: 2; top: 0; left: 0; width: 10px; height: 10px; background-color: green; } \
             #outer { position: absolute; z-index: 1; top: 0; left: 0; width: 10px; height: 10px; background-color: red; } \
             #inner { position: absolute; z-index: 100; top: 0; left: 0; width: 10px; height: 10px; background-color: blue; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let outer = colors
            .iter()
            .position(|&c| c == (255, 0, 0))
            .expect("outer fill");
        let inner = colors
            .iter()
            .position(|&c| c == (0, 0, 255))
            .expect("inner fill");
        let sib = colors
            .iter()
            .position(|&c| c == (0, 128, 0))
            .expect("sib fill");
        assert!(
            inner > outer,
            "inner paints after its stacking context root"
        );
        assert!(
            inner < sib,
            "inner z-index 100 ({inner}) is confined to outer's context, below sibling z-index 2 ({sib})"
        );
    }

    /// Stacking-context-heavy fixture: every Appendix E step bucket,
    /// z-index ties, hoisted in-flow and out-of-flow drains, and a
    /// nested stacking context — the exact fill order is pinned so the
    /// display-list builder's step partitioning (scratch Vec, sorted in
    /// place) cannot silently reorder paints. Each box gets a distinct
    /// red channel (see the CSS) so the sequence identifies the box.
    fn stacking_fixture() -> (&'static str, &'static str) {
        let html = r#"<div id="root">
            <div id="p2"></div><div id="n1"></div><div id="blk"></div>
            <div id="flt"></div><div id="inl"></div><div id="zro"></div>
            <div id="p1"></div><div id="n2"></div>
            <div id="outer"><div id="on5"></div><div id="oblk"></div><div id="op3"></div></div>
            <div id="a0"></div><div id="an"></div>
            <div id="t1"></div><div id="t2"></div>
        </div>"#;
        let css = "#root { background-color: rgb(170, 0, 0); } \
             div, #inl { width: 10px; height: 10px; } \
             #p2 { position: absolute; z-index: 2; background-color: rgb(10, 0, 0); } \
             #n1 { position: relative; z-index: -1; background-color: rgb(20, 0, 0); } \
             #blk { background-color: rgb(30, 0, 0); } \
             #flt { float: left; background-color: rgb(40, 0, 0); } \
             #inl { display: inline-block; background-color: rgb(50, 0, 0); } \
             #zro { opacity: 0.5; background-color: rgb(60, 0, 0); } \
             #p1 { position: relative; z-index: 1; background-color: rgb(70, 0, 0); } \
             #n2 { position: relative; z-index: -2; background-color: rgb(80, 0, 0); } \
             #outer { opacity: 0.9; background-color: rgb(90, 0, 0); } \
             #on5 { position: relative; z-index: -5; background-color: rgb(100, 0, 0); } \
             #oblk { background-color: rgb(110, 0, 0); } \
             #op3 { position: relative; z-index: 3; background-color: rgb(120, 0, 0); } \
             #a0 { position: absolute; background-color: rgb(130, 0, 0); } \
             #an { position: absolute; z-index: -3; background-color: rgb(140, 0, 0); } \
             #t1 { position: relative; z-index: 5; background-color: rgb(150, 0, 0); } \
             #t2 { position: relative; z-index: 5; background-color: rgb(160, 0, 0); }";
        (html, css)
    }

    #[test]
    fn stacking_context_steps_paint_in_exact_appendix_e_order() {
        let (html, css) = stacking_fixture();
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        // Recorded from the pre-scratch implementation; the old and new
        // step partitioning must agree exactly. Reading the sequence:
        // negative z-index ascending (`#an` -3, `#n2` -2, `#n1` -1),
        // then `#root`'s block-level content (`#blk`; the repeated 170
        // fills are `#root`'s own background, re-painted by the
        // fragmented block boxes layout splits it into — pre-existing
        // behavior, pinned as-is), the inline-level `#inl`, the float
        // `#flt`, the stack-level-0 contexts in tree order (`#zro`, then
        // `#outer` with its own nested steps: `#on5` -5, `#oblk`,
        // `#op3` 3), the z-auto abspos `#a0`, and positive z-index
        // ascending (`#p1` 1, `#p2` 2, then the z-5 tie `#t1`, `#t2` in
        // document order).
        let expected: Vec<(u8, u8, u8)> = [
            140, 80, 20, //
            170, 170, 30, 170, 170, 50, //
            170, 170, 170, 170, //
            40, 60, //
            90, 100, 110, 120, //
            130, 70, 10, 150, 160,
        ]
        .into_iter()
        .map(|r| (r, 0, 0))
        .collect();
        assert_eq!(colors, expected);
    }

    /// The step scratch is allocated once and reused by every stacking
    /// context in the build: after a build it is empty again, and a
    /// second build of the same fixture does not grow its capacity.
    #[test]
    fn step_scratch_capacity_reused_across_builds() {
        let (html, css) = stacking_fixture();
        let (doc, _quirks) = Document::parse_html(html.as_bytes());
        let author = bowser_css::parse_stylesheet(css);
        let viewport = Size { w: 800.0, h: 600.0 };
        let root =
            bowser_layout::layout(&doc, &[&author], viewport, &bowser_layout::FixedFontMetrics);
        let mut scratch = StepScratch::new();
        let list1 = build_display_list_with_scratch(
            &root,
            &doc,
            &mut |_| FontKey(0),
            &mut |_| None,
            viewport,
            &mut scratch,
        );
        let cap = scratch.steps.capacity();
        assert!(cap > 0, "the scratch grew during the first build");
        assert!(
            scratch.steps.is_empty(),
            "every context popped its tail back"
        );
        let list2 = build_display_list_with_scratch(
            &root,
            &doc,
            &mut |_| FontKey(0),
            &mut |_| None,
            viewport,
            &mut scratch,
        );
        eprintln!(
            "[measure] step scratch capacity after two builds: {}",
            scratch.steps.capacity()
        );
        assert_eq!(
            scratch.steps.capacity(),
            cap,
            "the second build reuses the first build's allocation"
        );
        assert_eq!(fill_colors(&list1), fill_colors(&list2));
    }

    /// A z-indexed abspos box inside a positioned non-stacking-context
    /// containing block hoists to the nearest ancestor stacking context
    /// (CSS 2.1 Appendix E): `#z` (z-index 15) paints after `#sib`
    /// (z-index 10), a sibling of its containing block, instead of being
    /// confined to the containing block's scope.
    #[test]
    fn z_indexed_abspos_hoists_past_non_stacking_context_cb() {
        let html = r#"<div id="cb"><div id="z"></div></div><div id="sib"></div>"#;
        let css = "#cb { position: relative; width: 10px; height: 10px; background-color: green; } \
             #z { position: absolute; z-index: 15; top: 0; left: 0; width: 10px; height: 10px; background-color: blue; } \
             #sib { position: absolute; z-index: 10; top: 0; left: 0; width: 10px; height: 10px; background-color: red; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let z = colors
            .iter()
            .position(|&c| c == (0, 0, 255))
            .expect("z fill");
        let sib = colors
            .iter()
            .position(|&c| c == (255, 0, 0))
            .expect("sib fill");
        assert!(
            z > sib,
            "hoisted z-index 15 ({z}) must paint after sibling z-index 10 ({sib})"
        );
    }

    /// Same hoist for in-flow positioned boxes: `#z`
    /// (`position: relative`, z-index 5) inside non-stacking-context
    /// `#cb` paints in the root stacking context's order — after `#sib`
    /// (z-index 1), a sibling of `#cb` — not confined to `#cb`'s scope.
    #[test]
    fn z_indexed_relative_hoists_past_non_stacking_context_parent() {
        let html = r#"<div id="cb"><div id="z"></div></div><div id="sib"></div>"#;
        let css = "#cb { position: relative; width: 10px; height: 10px; background-color: green; } \
             #z { position: relative; z-index: 5; width: 10px; height: 10px; background-color: blue; } \
             #sib { position: relative; z-index: 1; width: 10px; height: 10px; background-color: red; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let z = colors
            .iter()
            .position(|&c| c == (0, 0, 255))
            .expect("z fill");
        let sib = colors
            .iter()
            .position(|&c| c == (255, 0, 0))
            .expect("sib fill");
        assert!(
            z > sib,
            "hoisted z-index 5 ({z}) must paint after sibling z-index 1 ({sib})"
        );
    }

    /// A z-indexed `position: fixed` box hoists to its nearest ancestor
    /// stacking context (CSS 2.1 Appendix E), like a z-indexed abspos
    /// box: `#f` (fixed, z-index 5) drains into `#sc`'s steps and paints
    /// before `#q` (z-index 7, in-flow inside `#sc`) instead of draining
    /// at the viewport above `#sc`'s whole subtree.
    #[test]
    fn z_indexed_fixed_hoists_to_ancestor_stacking_context() {
        let html = r#"<div id="sc"><div id="f"></div><div id="q"></div></div>"#;
        let css = "#sc { position: relative; z-index: 3; width: 10px; height: 10px; background-color: green; } \
             #f { position: fixed; z-index: 5; top: 0; left: 0; width: 10px; height: 10px; background-color: blue; } \
             #q { position: relative; z-index: 7; width: 10px; height: 10px; background-color: yellow; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let f = colors
            .iter()
            .position(|&c| c == (0, 0, 255))
            .expect("fixed fill");
        let q = colors
            .iter()
            .position(|&c| c == (255, 255, 0))
            .expect("sibling fill");
        assert!(
            f < q,
            "hoisted fixed z-index 5 ({f}) must paint before sibling z-index 7 ({q})"
        );
    }

    /// Flex items re-sort by `order` for paint (flexbox §5.4), not just
    /// for layout.
    #[test]
    fn flex_order_reorders_paint() {
        let html = r#"<div id="c"><div id="a"></div><div id="b"></div></div>"#;
        let css = "#c { display: flex; width: 100px; height: 10px; } \
             #a { width: 10px; height: 10px; background-color: red; } \
             #b { width: 10px; height: 10px; background-color: blue; order: -1; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let a = colors
            .iter()
            .position(|&c| c == (255, 0, 0))
            .expect("a fill");
        let b = colors
            .iter()
            .position(|&c| c == (0, 0, 255))
            .expect("b fill");
        assert!(b < a, "order -1 ({b}) must paint before order 0 ({a})");
    }

    /// Grid items re-sort by `order` for paint (css-grid-1 §4.2).
    #[test]
    fn grid_order_reorders_paint() {
        let html = r#"<div id="c"><div id="a"></div><div id="b"></div></div>"#;
        let css = "#c { display: grid; width: 100px; height: 10px; } \
             #a { width: 10px; height: 10px; background-color: red; } \
             #b { width: 10px; height: 10px; background-color: blue; order: -1; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let a = colors
            .iter()
            .position(|&c| c == (255, 0, 0))
            .expect("a fill");
        let b = colors
            .iter()
            .position(|&c| c == (0, 0, 255))
            .expect("b fill");
        assert!(b < a, "order -1 ({b}) must paint before order 0 ({a})");
    }

    /// Floats paint above in-flow block backgrounds (Appendix E step 4
    /// vs step 3), whatever the DOM order.
    #[test]
    fn float_paints_above_block_background() {
        let html = r#"<div id="f"></div><div id="b"></div>"#;
        let css = "#f { float: left; width: 10px; height: 10px; background-color: blue; } \
             #b { width: 10px; height: 10px; background-color: red; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let f = colors
            .iter()
            .position(|&c| c == (0, 0, 255))
            .expect("float fill");
        let b = colors
            .iter()
            .position(|&c| c == (255, 0, 0))
            .expect("block fill");
        assert!(f > b, "float ({f}) must paint after block background ({b})");
    }

    /// Appendix E steps 4/5: every non-positioned float paints below all
    /// in-flow inline content, whatever the DOM order — a float placed
    /// after inline text in the DOM still paints under the text.
    ///
    /// The tree is built by hand: layout wraps inline runs among blocks
    /// in anonymous block boxes (CSS 2.1 §9.2.1.1), so a layout-produced
    /// tree never presents a raw float-after-inline sibling pair; paint's
    /// Appendix E logic must still order it correctly for trees that do.
    #[test]
    fn float_after_inline_text_paints_below_text() {
        use bowser_css::{Viewport, compute_style, default_ua_stylesheet, parse_stylesheet};

        let mut doc = Document::new();
        let mut styled = |style_attr: &str| {
            let el = doc.create_element("div");
            doc.set_attribute(el, "style", style_attr);
            let ua = default_ua_stylesheet();
            let vp = Viewport { w: 800.0, h: 600.0 };
            let author = parse_stylesheet("");
            compute_style(&doc, el, &[&author], &ua, &vp)
        };
        let plain = styled("color: black;");
        let blue_float = styled("float: left; background-color: blue; color: black;");
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
        };
        let mk = |box_type: BoxType, style: bowser_css::ComputedStyle| LayoutBox {
            rect,
            children: vec![],
            box_type,
            style: Some(std::rc::Rc::new(style)),
            text: None,
            node: None,
            fragments: vec![],
            z_index: None,
            collapsed_borders: None,
            phantom_w: false,
            phantom_h: false,
        };
        // Inline text first in DOM order, float second: the float's blue
        // background must still paint below the text (step 4 before
        // step 5).
        let mut text_box = mk(BoxType::Text, plain.clone());
        text_box.text = Some(Arc::from("hello"));
        text_box.fragments = vec![TextFragment {
            rect,
            baseline: 8.0,
            text_range: 0..5,
        }];
        let float_box = mk(BoxType::Block, blue_float);
        let root_box = LayoutBox {
            children: vec![text_box, float_box],
            ..mk(BoxType::Block, plain)
        };
        let list = build_display_list(
            &root_box,
            &doc,
            &mut |_| FontKey(0),
            &mut |_| None,
            Size { w: 800.0, h: 600.0 },
        );
        let items = &list.items;
        let text_pos = items
            .iter()
            .position(|it| {
                matches!(it, DisplayItem::TextRun { text, text_range, .. }
                    if text.get(text_range.clone()) == Some("hello"))
            })
            .expect("text run");
        let float_pos = items
            .iter()
            .position(|it| {
                matches!(it, DisplayItem::FillRect { color, .. }
                    if (color.r8(), color.g8(), color.b8()) == (0, 0, 255))
            })
            .expect("float fill");
        assert!(
            float_pos < text_pos,
            "float ({float_pos}) must paint before inline text ({text_pos})"
        );
    }

    /// Appendix E steps 3 vs 5 on the fast path: all in-flow block
    /// backgrounds paint before all in-flow inline content, even when
    /// block and inline siblings interleave in DOM order.
    ///
    /// The tree is built by hand: layout wraps inline runs among blocks
    /// in anonymous block boxes (CSS 2.1 §9.2.1.1), so a layout-produced
    /// tree never presents raw block/inline interleave; paint's Appendix
    /// E logic must still order it correctly for trees that do.
    #[test]
    fn interleaved_blocks_and_inlines_step_3_before_step_5() {
        use bowser_css::{Viewport, compute_style, default_ua_stylesheet, parse_stylesheet};

        let mut doc = Document::new();
        let mut styled = |style_attr: &str| {
            let el = doc.create_element("div");
            doc.set_attribute(el, "style", style_attr);
            let ua = default_ua_stylesheet();
            let vp = Viewport { w: 800.0, h: 600.0 };
            let author = parse_stylesheet("");
            compute_style(&doc, el, &[&author], &ua, &vp)
        };
        let plain = styled("color: black;");
        let red = styled("background-color: red; color: black;");
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
        };
        let mk = |box_type: BoxType, style: bowser_css::ComputedStyle| LayoutBox {
            rect,
            children: vec![],
            box_type,
            style: Some(std::rc::Rc::new(style)),
            text: None,
            node: None,
            fragments: vec![],
            z_index: None,
            collapsed_borders: None,
            phantom_w: false,
            phantom_h: false,
        };
        // `<div>a</div><span>b</span><div>c</div>` shape: block, inline
        // text, block — the later block's red background must paint
        // below the earlier inline's text (step 3 before step 5).
        let mut text_box = mk(BoxType::Text, plain.clone());
        text_box.text = Some(Arc::from("b"));
        text_box.fragments = vec![TextFragment {
            rect,
            baseline: 8.0,
            text_range: 0..1,
        }];
        let root_box = LayoutBox {
            children: vec![
                mk(BoxType::Block, plain.clone()),
                text_box,
                mk(BoxType::Block, red),
            ],
            ..mk(BoxType::Block, plain)
        };
        let list = build_display_list(
            &root_box,
            &doc,
            &mut |_| FontKey(0),
            &mut |_| None,
            Size { w: 800.0, h: 600.0 },
        );
        let items = &list.items;
        let red_bg = items
            .iter()
            .position(|it| {
                matches!(it, DisplayItem::FillRect { color, .. }
                    if (color.r8(), color.g8(), color.b8()) == (255, 0, 0))
            })
            .expect("red block background");
        let b_text = items
            .iter()
            .position(|it| {
                matches!(it, DisplayItem::TextRun { text, text_range, .. }
                    if text.get(text_range.clone()) == Some("b"))
            })
            .expect("b text run");
        assert!(
            red_bg < b_text,
            "later block background ({red_bg}) must paint before earlier inline text ({b_text})"
        );
    }

    /// flexbox §4 / css-grid-1 §5.4: `float` is ignored on flex/grid
    /// items (layout positions them as items regardless), so a static
    /// flex item with `float: left` paints at the in-flow step (3),
    /// not the float step (2) — in DOM order with the other item.
    #[test]
    fn float_ignored_on_flex_item() {
        let html = r#"<div id="f"><div id="i"></div><div id="j"></div></div>"#;
        let css = "#f { display: flex; } \
             #i { float: left; width: 10px; height: 10px; background-color: blue; } \
             #j { width: 10px; height: 10px; background-color: red; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        let colors = fill_colors(&list);
        let i = colors
            .iter()
            .position(|&c| c == (0, 0, 255))
            .expect("floated item fill");
        let j = colors
            .iter()
            .position(|&c| c == (255, 0, 0))
            .expect("in-flow item fill");
        assert!(
            i < j,
            "floated flex item ({i}) must paint in flow order, before ({j})"
        );
    }

    /// CSS 2.1 §11.1.1: an absolutely positioned descendant is clipped
    /// only by its containing block's overflow. The non-positioned
    /// `#clipper` between it and `#outer` must not clip it.
    #[test]
    fn abspos_escapes_non_containing_block_overflow_clip() {
        let html = r#"<div id="outer"><div id="clipper"><div id="abs"></div></div></div>"#;
        let css = "#outer { position: relative; } \
             #clipper { overflow: hidden; width: 200px; height: 100px; } \
             #abs { position: absolute; top: 0; left: 0; width: 10px; height: 10px; background-color: blue; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        assert!(
            !blue_is_clipped(&list),
            "abs fill must paint outside the clipper's clip"
        );
    }

    /// Positive control for the above: when the overflow box IS the
    /// containing block, the clip still applies.
    #[test]
    fn abspos_still_clipped_by_its_containing_block() {
        let html = r#"<div id="cb"><div id="abs"></div></div>"#;
        let css = "#cb { position: relative; overflow: hidden; width: 200px; height: 100px; } \
             #abs { position: absolute; top: 0; left: 0; width: 10px; height: 10px; background-color: blue; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        assert!(
            blue_is_clipped(&list),
            "abs fill must paint inside its containing block's clip"
        );
    }

    /// CSS Overflow 3: `overflow: clip` clips content (a `PushClip` wraps
    /// the children) without establishing a scroll container.
    #[test]
    fn overflow_clip_clips_content() {
        let html = r#"<div id="clip"><div id="child"></div></div>"#;
        let css = "#clip { overflow: clip; width: 200px; height: 100px; } \
             #child { width: 400px; height: 400px; background-color: blue; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        assert!(
            blue_is_clipped(&list),
            "overflow: clip must clip the overflowing child"
        );
        // Control: `visible` does not clip.
        let css = "#clip { overflow: visible; width: 200px; height: 100px; } \
             #child { width: 400px; height: 400px; background-color: blue; }";
        let list = layout_and_build_css(html, css, 800.0, 600.0);
        assert!(
            !blue_is_clipped(&list),
            "overflow: visible must not clip the child"
        );
    }

    // -- out-of-flow drain anchoring at stacking contexts --------------

    /// Synthetic `root > wrap > oof` tree: `wrap` carries `wrap_style`,
    /// `oof` carries `oof_style` (`position: absolute` or `fixed` with a
    /// blue background). Styles run through the real cascade; every box
    /// paints exactly one `FillRect` background.
    fn oof_anchor_tree(wrap_style: &str, oof_style: &str) -> DisplayList {
        use bowser_css::{Viewport, compute_style, default_ua_stylesheet, parse_stylesheet};

        let mut doc = Document::new();
        let mut styled = |style_attr: &str| {
            let el = doc.create_element("div");
            doc.set_attribute(el, "style", style_attr);
            let ua = default_ua_stylesheet();
            let vp = Viewport { w: 800.0, h: 600.0 };
            let author = parse_stylesheet("");
            compute_style(&doc, el, &[&author], &ua, &vp)
        };
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
        };
        let mk = |box_type: BoxType, style: bowser_css::ComputedStyle| LayoutBox {
            rect,
            children: vec![],
            box_type,
            style: Some(std::rc::Rc::new(style)),
            text: None,
            node: None,
            fragments: vec![],
            z_index: None,
            collapsed_borders: None,
            phantom_w: false,
            phantom_h: false,
        };
        let oof = mk(BoxType::Block, styled(oof_style));
        let wrap = LayoutBox {
            children: vec![oof],
            ..mk(BoxType::Block, styled(wrap_style))
        };
        let root = LayoutBox {
            children: vec![wrap],
            ..mk(BoxType::Block, styled("color: black;"))
        };
        build_display_list(
            &root,
            &doc,
            &mut |_| FontKey(0),
            &mut |_| None,
            Size { w: 800.0, h: 600.0 },
        )
    }

    fn blue_fill(list: &DisplayList) -> usize {
        list.items
            .iter()
            .position(|it| {
                matches!(it, DisplayItem::FillRect { color, .. }
                    if *color == Color::rgba8(0, 0, 255, 255))
            })
            .expect("oof blue fill paints")
    }

    /// CSS 2.1 Appendix E: an `absolute` box inside a static
    /// `opacity < 1` box paints inside that box's stacking context —
    /// the opacity group must bracket its fill. Before the anchor fix
    /// it drained at the viewport-anchored root steps and painted
    /// unfaded, outside the group.
    #[test]
    fn oof_absolute_inside_opacity_group() {
        let list = oof_anchor_tree(
            "opacity: 0.5; background-color: red;",
            "position: absolute; background-color: blue;",
        );
        let items = &list.items;
        let push = items
            .iter()
            .position(|it| {
                matches!(it, DisplayItem::PushOpacity { opacity } if (*opacity - 0.5).abs() < 1e-6)
            })
            .expect("opacity group opens");
        let pop = items
            .iter()
            .position(|it| matches!(it, DisplayItem::PopOpacity))
            .expect("opacity group closes");
        let blue = blue_fill(&list);
        assert!(
            push < blue && blue < pop,
            "absolute fill ({blue}) must paint inside the opacity group ({push}..{pop})"
        );
    }

    /// Same as above for the other non-positioned stacking-context
    /// creator: `mix-blend-mode: multiply`.
    #[test]
    fn oof_absolute_inside_blend_group() {
        let list = oof_anchor_tree(
            "mix-blend-mode: multiply; background-color: red;",
            "position: absolute; background-color: blue;",
        );
        let items = &list.items;
        let push = items
            .iter()
            .position(|it| matches!(it, DisplayItem::PushBlend { .. }))
            .expect("blend group opens");
        let pop = items
            .iter()
            .position(|it| matches!(it, DisplayItem::PopBlend))
            .expect("blend group closes");
        let blue = blue_fill(&list);
        assert!(
            push < blue && blue < pop,
            "absolute fill ({blue}) must paint inside the blend group ({push}..{pop})"
        );
    }

    /// No-regression: an `absolute` child of a *positioned* stacking
    /// context drains at its containing block's scope exactly as before
    /// (anchor == containing block), inside that box's own group.
    #[test]
    fn oof_absolute_of_positioned_sc_drains_at_cb() {
        let list = oof_anchor_tree(
            "position: relative; opacity: 0.5; background-color: red;",
            "position: absolute; background-color: blue;",
        );
        let items = &list.items;
        let push = items
            .iter()
            .position(|it| {
                matches!(it, DisplayItem::PushOpacity { opacity } if (*opacity - 0.5).abs() < 1e-6)
            })
            .expect("opacity group opens");
        let pop = items
            .iter()
            .position(|it| matches!(it, DisplayItem::PopOpacity))
            .expect("opacity group closes");
        let blue = blue_fill(&list);
        assert!(
            push < blue && blue < pop,
            "absolute fill ({blue}) must stay inside its positioned SC's group ({push}..{pop})"
        );
    }

    /// No-regression: a plain `absolute` box with no stacking-context
    /// ancestor anchors at the viewport drain exactly as before — no
    /// group opens anywhere and the fill still paints.
    #[test]
    fn oof_plain_absolute_unchanged() {
        let list = oof_anchor_tree(
            "background-color: red;",
            "position: absolute; background-color: blue;",
        );
        let _ = blue_fill(&list);
        assert!(
            !list.items.iter().any(|it| matches!(
                it,
                DisplayItem::PushOpacity { .. }
                    | DisplayItem::PopOpacity
                    | DisplayItem::PushBlend { .. }
                    | DisplayItem::PopBlend
            )),
            "no stacking-context group may open for a plain absolute box"
        );
    }

    /// `position: fixed` is viewport-anchored but still paints inside
    /// its nearest ancestor stacking context: a fixed box inside a
    /// static `opacity < 1` box must fade with it.
    #[test]
    fn oof_fixed_inside_opacity_group() {
        let list = oof_anchor_tree(
            "opacity: 0.5; background-color: red;",
            "position: fixed; background-color: blue;",
        );
        let items = &list.items;
        let push = items
            .iter()
            .position(|it| {
                matches!(it, DisplayItem::PushOpacity { opacity } if (*opacity - 0.5).abs() < 1e-6)
            })
            .expect("opacity group opens");
        let pop = items
            .iter()
            .position(|it| matches!(it, DisplayItem::PopOpacity))
            .expect("opacity group closes");
        let blue = blue_fill(&list);
        assert!(
            push < blue && blue < pop,
            "fixed fill ({blue}) must paint inside the opacity group ({push}..{pop})"
        );
    }

    /// CSS 2.1 §11.1.1 clip placement for anchored drains: a static
    /// `opacity < 1` + `overflow: hidden` box `S` containing an
    /// `overflow: hidden` wrapper `W` containing an `absolute` box `A`
    /// whose containing block is the outer positioned box `P`. `A`
    /// anchors at `S` (nearest stacking context below its containing
    /// block), so it must paint inside `S`'s opacity group but OUTSIDE
    /// `S`'s own overflow clip — `S`'s overflow is not `A`'s containing
    /// block's. The in-flow positioned sibling `B` (same Appendix E
    /// rank) keeps its merged z-order slot above `A` and stays inside
    /// the clip.
    #[test]
    fn oof_drain_escapes_anchor_clip() {
        use bowser_css::{Viewport, compute_style, default_ua_stylesheet, parse_stylesheet};

        let mut doc = Document::new();
        let mut styled = |style_attr: &str| {
            let el = doc.create_element("div");
            doc.set_attribute(el, "style", style_attr);
            let ua = default_ua_stylesheet();
            let vp = Viewport { w: 800.0, h: 600.0 };
            let author = parse_stylesheet("");
            compute_style(&doc, el, &[&author], &ua, &vp)
        };
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
        };
        let mk = |box_type: BoxType, style: bowser_css::ComputedStyle| LayoutBox {
            rect,
            children: vec![],
            box_type,
            style: Some(std::rc::Rc::new(style)),
            text: None,
            node: None,
            fragments: vec![],
            z_index: None,
            collapsed_borders: None,
            phantom_w: false,
            phantom_h: false,
        };
        // root > P (positioned: the containing block) > S (static
        // opacity stacking context + overflow hidden) > W (overflow
        // hidden) > A (absolute). B is S's in-flow positioned z-0
        // sibling, after W in DOM order.
        let abs = mk(
            BoxType::Block,
            styled("position: absolute; background-color: blue;"),
        );
        let wrap = LayoutBox {
            children: vec![abs],
            ..mk(
                BoxType::Block,
                styled("overflow: hidden; background-color: green;"),
            )
        };
        let sib = LayoutBox {
            z_index: Some(0),
            ..mk(
                BoxType::Block,
                styled("position: relative; z-index: 0; background-color: yellow;"),
            )
        };
        let sc = LayoutBox {
            children: vec![wrap, sib],
            ..mk(
                BoxType::Block,
                styled("opacity: 0.5; overflow: hidden; background-color: red;"),
            )
        };
        let outer = LayoutBox {
            children: vec![sc],
            ..mk(BoxType::Block, styled("position: relative;"))
        };
        let root = LayoutBox {
            children: vec![outer],
            ..mk(BoxType::Block, styled("color: black;"))
        };
        let list = build_display_list(
            &root,
            &doc,
            &mut |_| FontKey(0),
            &mut |_| None,
            Size { w: 800.0, h: 600.0 },
        );
        let items = &list.items;
        let push = items
            .iter()
            .position(|it| {
                matches!(it, DisplayItem::PushOpacity { opacity } if (*opacity - 0.5).abs() < 1e-6)
            })
            .expect("opacity group opens");
        let pop = items
            .iter()
            .position(|it| matches!(it, DisplayItem::PopOpacity))
            .expect("opacity group closes");
        let blue = blue_fill(&list);
        let yellow = items
            .iter()
            .position(|it| {
                matches!(it, DisplayItem::FillRect { color, .. }
                    if *color == Color::rgba8(255, 255, 0, 255))
            })
            .expect("in-flow sibling yellow fill paints");
        assert!(
            push < blue && blue < pop,
            "absolute fill ({blue}) must paint inside the anchor's opacity group ({push}..{pop})"
        );
        // Pair every clip push with its pop: the drained fill must sit
        // outside all of them (§11.1.1 — only the containing block's
        // overflow may clip it, and P has none).
        let mut stack: Vec<usize> = Vec::new();
        for (i, it) in items.iter().enumerate() {
            match it {
                DisplayItem::PushClip { .. } | DisplayItem::PushRoundedClip { .. } => stack.push(i),
                DisplayItem::PopClip => {
                    let open = stack.pop().expect("balanced clip pops");
                    assert!(
                        !(open < blue && blue < i),
                        "absolute fill ({blue}) must not sit inside a clip ({open}..{i})"
                    );
                }
                _ => {}
            }
        }
        assert!(stack.is_empty(), "balanced clip pushes");
        // Same-rank z-interleaving is preserved: the drained box (earlier
        // in document order) keeps its merged slot below the sibling.
        assert!(
            blue < yellow,
            "drained fill ({blue}) must keep its z-order slot below the sibling ({yellow})"
        );
        // Positive control: the in-flow sibling still paints clipped.
        let mut depth = 0;
        let mut yellow_depth = 0;
        for (i, it) in items.iter().enumerate() {
            match it {
                DisplayItem::PushClip { .. } | DisplayItem::PushRoundedClip { .. } => depth += 1,
                DisplayItem::PopClip => depth -= 1,
                _ => {}
            }
            if i == yellow {
                yellow_depth = depth;
            }
        }
        assert!(
            yellow_depth > 0,
            "in-flow sibling ({yellow}) must stay inside the anchor's clip"
        );
    }

    /// CSS 2.1 §11.1.1 clip placement for intermediate boxes: `X`
    /// (`overflow: hidden`, static, not a stacking context) sits strictly
    /// between the `absolute` box `A`'s containing block `P` and its
    /// drain anchor `S` (static `opacity < 1` stacking context). Only the
    /// containing block's overflow may clip `A`, so `A` must paint
    /// outside `X`'s clip — inside `S`'s opacity group (which `X`'s clip
    /// would otherwise wrap via nesting). Mirrors
    /// `oof_drain_escapes_anchor_clip` with `X` inserted between `P` and
    /// `S`.
    #[test]
    fn oof_drain_escapes_intermediate_clip() {
        use bowser_css::{Viewport, compute_style, default_ua_stylesheet, parse_stylesheet};

        let mut doc = Document::new();
        let mut styled = |style_attr: &str| {
            let el = doc.create_element("div");
            doc.set_attribute(el, "style", style_attr);
            let ua = default_ua_stylesheet();
            let vp = Viewport { w: 800.0, h: 600.0 };
            let author = parse_stylesheet("");
            compute_style(&doc, el, &[&author], &ua, &vp)
        };
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
        };
        let mk = |box_type: BoxType, style: bowser_css::ComputedStyle| LayoutBox {
            rect,
            children: vec![],
            box_type,
            style: Some(std::rc::Rc::new(style)),
            text: None,
            node: None,
            fragments: vec![],
            z_index: None,
            collapsed_borders: None,
            phantom_w: false,
            phantom_h: false,
        };
        // root > P (positioned: the containing block) > X (overflow
        // hidden, static, not a stacking context) > S (static opacity
        // stacking context + overflow hidden: the drain anchor) > W
        // (overflow hidden) > A (absolute). B is S's in-flow positioned
        // z-0 sibling, after W in DOM order.
        let abs = mk(
            BoxType::Block,
            styled("position: absolute; background-color: blue;"),
        );
        let wrap = LayoutBox {
            children: vec![abs],
            ..mk(
                BoxType::Block,
                styled("overflow: hidden; background-color: green;"),
            )
        };
        let sib = LayoutBox {
            z_index: Some(0),
            ..mk(
                BoxType::Block,
                styled("position: relative; z-index: 0; background-color: yellow;"),
            )
        };
        let sc = LayoutBox {
            children: vec![wrap, sib],
            ..mk(
                BoxType::Block,
                styled("opacity: 0.5; overflow: hidden; background-color: red;"),
            )
        };
        let mid = LayoutBox {
            children: vec![sc],
            ..mk(BoxType::Block, styled("overflow: hidden;"))
        };
        let outer = LayoutBox {
            children: vec![mid],
            ..mk(BoxType::Block, styled("position: relative;"))
        };
        let root = LayoutBox {
            children: vec![outer],
            ..mk(BoxType::Block, styled("color: black;"))
        };
        let list = build_display_list(
            &root,
            &doc,
            &mut |_| FontKey(0),
            &mut |_| None,
            Size { w: 800.0, h: 600.0 },
        );
        let items = &list.items;
        let push = items
            .iter()
            .position(|it| {
                matches!(it, DisplayItem::PushOpacity { opacity } if (*opacity - 0.5).abs() < 1e-6)
            })
            .expect("opacity group opens");
        let pop = items
            .iter()
            .position(|it| matches!(it, DisplayItem::PopOpacity))
            .expect("opacity group closes");
        let blue = blue_fill(&list);
        let yellow = items
            .iter()
            .position(|it| {
                matches!(it, DisplayItem::FillRect { color, .. }
                    if *color == Color::rgba8(255, 255, 0, 255))
            })
            .expect("in-flow sibling yellow fill paints");
        assert!(
            push < blue && blue < pop,
            "absolute fill ({blue}) must paint inside the anchor's opacity group ({push}..{pop})"
        );
        // Pair every clip push with its pop: the drained fill must sit
        // outside all of them — `X`'s included (§11.1.1: `X` sits
        // strictly between the anchor and the containing block, and `P`
        // has no overflow to clip with).
        let mut stack: Vec<usize> = Vec::new();
        for (i, it) in items.iter().enumerate() {
            match it {
                DisplayItem::PushClip { .. } | DisplayItem::PushRoundedClip { .. } => stack.push(i),
                DisplayItem::PopClip => {
                    let open = stack.pop().expect("balanced clip pops");
                    assert!(
                        !(open < blue && blue < i),
                        "absolute fill ({blue}) must not sit inside a clip ({open}..{i})"
                    );
                }
                _ => {}
            }
        }
        assert!(stack.is_empty(), "balanced clip pushes");
        // Same-rank z-interleaving is preserved: the drained box (earlier
        // in document order) keeps its merged slot below the sibling.
        assert!(
            blue < yellow,
            "drained fill ({blue}) must keep its z-order slot below the sibling ({yellow})"
        );
        // Positive control: the in-flow sibling still paints clipped
        // (by the anchor's own clip, which the intermediate wrap leaves
        // alone).
        let mut depth = 0;
        let mut yellow_depth = 0;
        for (i, it) in items.iter().enumerate() {
            match it {
                DisplayItem::PushClip { .. } | DisplayItem::PushRoundedClip { .. } => depth += 1,
                DisplayItem::PopClip => depth -= 1,
                _ => {}
            }
            if i == yellow {
                yellow_depth = depth;
            }
        }
        assert!(
            yellow_depth > 0,
            "in-flow sibling ({yellow}) must stay inside the anchor's clip"
        );
    }

    /// Every layout text fragment becomes exactly one verbatim TextRun,
    /// in order, with the fragment's rect and baseline.
    #[test]
    fn one_verbatim_run_per_fragment() {
        let (_doc, root, list) =
            layout_and_build("<div>hello <b>bold</b> world</div>", 400.0, 400.0);
        let mut boxes = Vec::new();
        collect_text_boxes(&root, &mut boxes);
        let expected: Vec<(String, Rect, f32)> = boxes
            .iter()
            .flat_map(|(t, frags)| {
                frags
                    .iter()
                    .map(|f| verbatim_slice(t, f))
                    .zip(frags.iter())
                    .filter(|(s, _)| !s.is_empty())
                    .map(|(s, f)| (s.to_string(), f.rect, f.baseline))
            })
            .collect();
        let runs = text_runs(&list);
        assert!(!runs.is_empty(), "expected text runs");
        assert_eq!(runs.len(), expected.len(), "one run per fragment");
        for (run, exp) in runs.iter().zip(expected.iter()) {
            assert_eq!(run.0, exp.0, "verbatim text");
            assert_eq!(run.1, exp.1, "fragment rect");
            assert_eq!(run.2, exp.2, "fragment baseline");
        }
        // Concatenation reproduces the layout text exactly.
        let want: String = boxes.iter().map(|(t, _)| t.as_str()).collect();
        let got: String = runs.iter().map(|r| r.0.as_str()).collect();
        assert_eq!(got, want);
    }

    /// Regression test for the per-element allocations in `build_box`'s
    /// hot path.
    ///
    /// Builds a deep tree of styled boxes and measures heap bytes
    /// allocated *during* [`build_display_list`] via the shared
    /// counting allocator
    /// ([`crate::images::test_alloc::ALLOCATED_BYTES`]). The old code
    /// `format!`ed 12 border property-name `String`s per styled box
    /// (even with no border set) and collected a `Vec<String>` per text
    /// box; the new code allocates nothing per box on this path. The
    /// boxes' style sets only the inherited `color`, so no items are
    /// emitted and the byte count isolates per-box overhead: the bound
    /// fails on the old code (~130 KiB for this tree) and passes with
    /// headroom on the new code (~0).
    #[test]
    fn deep_styled_tree_build_allocations_bounded() {
        use crate::images::test_alloc::ALLOCATED_BYTES;
        use bowser_css::{Viewport, compute_style, default_ua_stylesheet, parse_stylesheet};

        const DEPTH: usize = 200;
        let mut doc = Document::new();
        let root = doc.root();
        let styled = doc.create_element("div");
        doc.append_child(root, styled);
        doc.set_attribute(styled, "style", "color: red;");
        let ua = default_ua_stylesheet();
        let vp = Viewport { w: 800.0, h: 600.0 };
        let author = parse_stylesheet("");
        let style = compute_style(&doc, styled, &[&author], &ua, &vp);

        // Nest DEPTH boxes, each holding one child.
        let mut inner: Option<LayoutBox> = None;
        for _ in 0..DEPTH {
            inner = Some(LayoutBox {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 10.0,
                    h: 10.0,
                },
                children: inner.map(|b| vec![b]).unwrap_or_default(),
                box_type: BoxType::Block,
                style: Some(std::rc::Rc::new(style.clone())),
                text: None,
                node: None,
                fragments: vec![],
                z_index: None,
                collapsed_borders: None,
                phantom_w: false,
                phantom_h: false,
            });
        }
        let root_box = inner.expect("non-empty tree");
        assert_eq!(root_box.children.len(), 1, "tree must be nested");

        // Warm the CSS property-name index (`OnceLock<HashMap>` built on
        // first `ComputedStyle::get`): a one-time per-process cost, not
        // per-box, so it must not count toward the per-build budget.
        let _ = style.get_by_index(props::COLOR);

        ALLOCATED_BYTES.with(|c| c.set(0));
        let list = build_display_list(
            &root_box,
            &doc,
            &mut |_| FontKey(0),
            &mut |_| None,
            Size { w: 800.0, h: 600.0 },
        );
        let allocated = ALLOCATED_BYTES.with(|c| c.get());
        assert!(
            allocated < 4096,
            "deep build allocated {allocated} bytes; per-box allocations regressed"
        );
        assert!(list.items.is_empty(), "color-only boxes emit no items");
    }

    /// `verbatim_slice` consumes each fragment's `text_range` instead of
    /// estimating from widths: the narrow "iii" fragment (10px of 100px)
    /// still gets all three chars, where the old width-proportional cut
    /// gave it one ("i") and pushed "iWWW" onto the wrong line.
    #[test]
    fn verbatim_slice_uses_text_range_not_widths() {
        let frags = [
            TextFragment {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 10.0,
                    h: 10.0,
                },
                baseline: 8.0,
                text_range: 0..3,
            },
            TextFragment {
                rect: Rect {
                    x: 10.0,
                    y: 0.0,
                    w: 90.0,
                    h: 10.0,
                },
                baseline: 8.0,
                text_range: 3..6,
            },
        ];
        let slices: Vec<&str> = frags.iter().map(|f| verbatim_slice("iiiWWW", f)).collect();
        assert_eq!(slices, vec!["iii", "WWW"]);
        // Out-of-bounds ranges degrade to empty slices, never a panic.
        let bad = [TextFragment {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w: 10.0,
                h: 10.0,
            },
            baseline: 8.0,
            text_range: 0..100,
        }];
        assert_eq!(verbatim_slice("hi", &bad[0]), "");
    }

    /// Narrow viewport wraps the line: still one run per fragment, text
    /// preserved across the wrap.
    /// Wrapping drops the wrap-opportunity spaces (CSS whitespace
    /// collapsing: the fragments' `text_range`s skip them), so runs
    /// concatenate to the box text minus those spaces; every other char
    /// is kept exactly once, in order, and each run is exactly its
    /// fragment's range (no mid-word cuts).
    #[test]
    fn wrapped_text_keeps_every_char() {
        let (_doc, root, list) = layout_and_build("<p>aaaa bbbb cccc dddd eeee</p>", 50.0, 400.0);
        let runs = text_runs(&list);
        assert!(runs.len() >= 2, "expected wrapped fragments, got {runs:?}");
        let mut boxes = Vec::new();
        collect_text_boxes(&root, &mut boxes);
        let want: String = boxes
            .iter()
            .flat_map(|(t, _)| t.chars().filter(|c| !c.is_whitespace()))
            .collect();
        let got: String = runs
            .iter()
            .flat_map(|r| r.0.chars().filter(|c| !c.is_whitespace()))
            .collect();
        assert_eq!(got, want, "no char lost or duplicated across fragments");
        // Each run is exactly its fragment's text_range.
        let expected: Vec<String> = boxes
            .iter()
            .flat_map(|(t, frags)| frags.iter().map(|f| t[f.text_range.clone()].to_string()))
            .filter(|s| !s.is_empty())
            .collect();
        let got_runs: Vec<String> = runs.iter().map(|r| r.0.clone()).collect();
        assert_eq!(got_runs, expected, "runs match fragment ranges");
    }

    /// Background paints before the text that sits on it.
    #[test]
    fn background_before_text_sequence() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"background: red; width: 50px; height: 50px\">x</div>",
            200.0,
            200.0,
        );
        let bg = list.items.iter().position(|it| {
            matches!(it, DisplayItem::FillRect { color, .. } if *color == Color::rgba8(255, 0, 0, 255))
        });
        let tx = list
            .items
            .iter()
            .position(|it| matches!(it, DisplayItem::TextRun { .. }));
        match (bg, tx) {
            (Some(b), Some(t)) => assert!(b < t, "background paints before text"),
            _ => panic!("expected background fill and text run"),
        }
    }

    /// Borders become one Border item with per-side widths/styles.
    #[test]
    fn border_item_from_style() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"border: 2px solid black; width: 10px; height: 10px\"></div>",
            200.0,
            200.0,
        );
        let border = list.items.iter().find_map(|it| match it {
            DisplayItem::Border {
                widths,
                styles,
                radii,
                ..
            } => Some((*widths, *styles, *radii)),
            _ => None,
        });
        let (widths, styles, radii) = border.expect("expected a Border item");
        assert_eq!(widths, [2.0, 2.0, 2.0, 2.0]);
        assert_eq!(styles, [BorderStyle::Solid; 4]);
        // No radius declarations: all zero.
        assert_eq!(radii, [0.0; 4]);
    }

    /// `border-collapse: collapse`: adjacent cells with different borders
    /// paint one shared winning border per grid line instead of each
    /// cell's own. The wider (6px blue) border wins the shared edge; the
    /// table's own 10px green border wins the outer edges; the table and
    /// row boxes emit no Border items of their own (their claims are
    /// folded into the cell winners).
    ///
    /// (Cell text is wrapped in borderless `<span>`s; text boxes never
    /// paint borders, so bare text paints the same winners — see
    /// [`collapse_table_bare_text_paints_no_extra_borders`].)
    #[test]
    fn collapse_table_paints_single_shared_border() {
        let (_doc, _root, list) = layout_and_build(
            "<table style=\"border-collapse: collapse; border: 10px solid green\">\
             <tr><td style=\"border: 2px solid red\"><span>a</span></td>\
             <td style=\"border: 6px solid blue\"><span>b</span></td></tr></table>",
            400.0,
            200.0,
        );
        let borders: Vec<([f32; 4], [BorderStyle; 4], [Color; 4])> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::Border {
                    widths,
                    styles,
                    colors,
                    ..
                } => Some((*widths, *styles, *colors)),
                _ => None,
            })
            .collect();
        // One Border item per cell; the table/row boxes paint nothing.
        assert_eq!(borders.len(), 2, "expected one Border item per cell");
        let green = Color::rgba8(0, 128, 0, 255);
        let blue = Color::rgba8(0, 0, 255, 255);
        // The shared edge is painted exactly once: the right cell's left
        // edge carries the 6px blue winner ...
        let shared: Vec<_> = borders.iter().filter(|(w, _, _)| w[3] == 6.0).collect();
        assert_eq!(shared.len(), 1, "shared edge painted exactly once");
        assert_eq!(shared[0].1[3], BorderStyle::Solid);
        assert_eq!(shared[0].2[3], blue);
        // ... and the left cell paints nothing on its (unowned) right edge.
        assert!(
            borders.iter().any(|(w, _, _)| w[1] == 0.0),
            "left cell must not double-paint the shared edge"
        );
        // Outer edges resolve against the table's 10px green border.
        assert!(
            borders
                .iter()
                .any(|(w, s, c)| w[0] == 10.0 && s[0] == BorderStyle::Solid && c[0] == green),
            "table border wins the outer top edge"
        );
    }

    /// `border-collapse: separate` (the default) is untouched: cells keep
    /// painting their own CSS borders, one Border item per cell (text is
    /// wrapped as in [`collapse_table_paints_single_shared_border`];
    /// bare text paints the same since text boxes never emit borders).
    #[test]
    fn separate_table_paints_each_cell_own_border() {
        let (_doc, _root, list) = layout_and_build(
            "<table><tr><td style=\"border: 2px solid red\"><span>a</span></td>\
             <td style=\"border: 6px solid blue\"><span>b</span></td></tr></table>",
            400.0,
            200.0,
        );
        let borders: Vec<[f32; 4]> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::Border { widths, .. } => Some(*widths),
                _ => None,
            })
            .collect();
        assert_eq!(borders.len(), 2);
        assert!(borders.contains(&[2.0, 2.0, 2.0, 2.0]));
        assert!(borders.contains(&[6.0, 6.0, 6.0, 6.0]));
    }

    // -- `visibility: collapse` (CSS 2.1 §17.5) --------------------------

    /// `visibility: collapse` on a row: the row's whole subtree paints
    /// nothing (the keyword inherits onto its cells), while the visible
    /// row paints normally. Layout additionally zeroes the row's height.
    #[test]
    fn visibility_collapse_row_paints_nothing() {
        let (_doc, _root, list) = layout_and_build(
            "<table><tr><td><span>a</span></td></tr>\
             <tr style=\"visibility: collapse\"><td><span>b</span></td></tr></table>",
            400.0,
            200.0,
        );
        let texts: Vec<String> = text_runs(&list).into_iter().map(|(t, _, _)| t).collect();
        assert!(
            texts.iter().any(|t| t == "a"),
            "visible row paints: {texts:?}"
        );
        assert!(
            !texts.iter().any(|t| t == "b"),
            "collapsed row must not paint: {texts:?}"
        );
    }

    /// `visibility: collapse` on a column: the column's cells paint
    /// nothing. Columns are not DOM ancestors of their cells, so the
    /// keyword cannot inherit onto them — instead layout skips those
    /// cells' tasks outright (zero rect, no shaped text), and there is
    /// nothing for paint to emit.
    #[test]
    fn visibility_collapse_column_paints_nothing() {
        let (_doc, _root, list) = layout_and_build(
            "<table><col><col style=\"visibility: collapse\">\
             <tr><td><span>a</span></td><td><span>b</span></td></tr></table>",
            400.0,
            200.0,
        );
        let texts: Vec<String> = text_runs(&list).into_iter().map(|(t, _, _)| t).collect();
        assert!(
            texts.iter().any(|t| t == "a"),
            "visible column paints: {texts:?}"
        );
        assert!(
            !texts.iter().any(|t| t == "b"),
            "collapsed column must not paint: {texts:?}"
        );
    }

    /// `visibility: collapse` on an element that is not a row, row
    /// group, column, or column group means `hidden` (CSS 2.1 §17.5):
    /// the box and its subtree paint nothing.
    #[test]
    fn visibility_collapse_on_plain_box_paints_like_hidden() {
        let list = layout_and_build_css(
            "<div><p class=\"gone\">x</p><p>y</p></div>",
            "p.gone { visibility: collapse; }",
            400.0,
            200.0,
        );
        let texts: Vec<String> = text_runs(&list).into_iter().map(|(t, _, _)| t).collect();
        assert!(texts.iter().any(|t| t == "y"), "sibling paints: {texts:?}");
        assert!(
            !texts.iter().any(|t| t == "x"),
            "collapsed box must not paint: {texts:?}"
        );
    }

    // -- `empty-cells` --------------------------------------------------

    /// Table cells in layout order (document order).
    fn table_cells(root: &LayoutBox) -> Vec<&LayoutBox> {
        let mut out = Vec::new();
        fn go<'a>(b: &'a LayoutBox, out: &mut Vec<&'a LayoutBox>) {
            if b.box_type == BoxType::TableCell {
                out.push(b);
            }
            for c in &b.children {
                go(c, out);
            }
        }
        go(root, &mut out);
        out
    }

    fn rect_approx(a: Rect, b: Rect) -> bool {
        const E: f32 = 0.01;
        (a.x - b.x).abs() < E
            && (a.y - b.y).abs() < E
            && (a.w - b.w).abs() < E
            && (a.h - b.h).abs() < E
    }

    fn fill_rects(list: &DisplayList) -> Vec<Rect> {
        list.items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::FillRect { rect, .. } => Some(*rect),
                _ => None,
            })
            .collect()
    }

    fn border_rects(list: &DisplayList) -> Vec<Rect> {
        list.items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::Border { rect, .. } => Some(*rect),
                _ => None,
            })
            .collect()
    }

    /// CSS 2.1 §17.5.1: in the separated borders model,
    /// `empty-cells: hide` suppresses the background and border of an
    /// empty cell, while a non-empty cell paints both. The property is
    /// set on the table and inherited by the cells.
    #[test]
    fn empty_cells_hide_skips_empty_cell_background_and_border() {
        let (_doc, root, list) = layout_and_build(
            "<table style=\"border-collapse: separate; empty-cells: hide\">\
             <tr><td style=\"background: red; border: 2px solid red\"></td>\
             <td style=\"background: red; border: 2px solid red\">x</td></tr></table>",
            400.0,
            200.0,
        );
        let cells = table_cells(&root);
        assert_eq!(cells.len(), 2);
        let (empty, full) = (cells[0].rect, cells[1].rect);
        // The empty cell paints no background and no border of its own;
        // the non-empty cell paints both. (Note: the text box repaints
        // the cell's background at its own rect through the inherited
        // style — a pre-existing quirk unrelated to `empty-cells` — so
        // the assertions are rect-based, not count-based.)
        let fills = fill_rects(&list);
        assert!(
            !fills.iter().any(|r| rect_approx(*r, empty)),
            "empty cell must not paint a background"
        );
        assert!(
            fills.iter().any(|r| rect_approx(*r, full)),
            "non-empty cell must paint its background"
        );
        // Borders read the box's own style only, so exactly one Border
        // item exists: the non-empty cell's.
        let borders = border_rects(&list);
        assert_eq!(borders.len(), 1, "empty cell must not paint a border");
        assert!(rect_approx(borders[0], full));
    }

    /// Whitespace-only content counts as empty (matching browsers), and
    /// so does a cell whose only in-flow child is out of flow
    /// (`position: absolute`): neither paints under `empty-cells: hide`.
    #[test]
    fn empty_cells_hide_treats_whitespace_and_oof_only_as_empty() {
        let (_doc, root, list) = layout_and_build(
            "<table style=\"border-collapse: separate; empty-cells: hide\">\
             <tr><td style=\"background: red; border: 2px solid red\">   </td>\
             <td style=\"background: red; border: 2px solid red\">\
             <div style=\"position: absolute\">x</div></td></tr></table>",
            400.0,
            200.0,
        );
        let cells = table_cells(&root);
        assert_eq!(cells.len(), 2);
        // Neither cell paints a background or border of its own. (The
        // absolutely positioned text still paints its text run — only
        // the cell's background/border are suppressed.)
        assert!(
            fill_rects(&list).is_empty(),
            "no cell background should paint"
        );
        assert!(
            border_rects(&list).is_empty(),
            "no cell border should paint"
        );
        assert!(
            list.items
                .iter()
                .any(|it| matches!(it, DisplayItem::TextRun { .. })),
            "out-of-flow text still paints"
        );
    }

    /// Any in-flow element — even an empty `<div></div>` or a replaced
    /// element like `<img>` — is content: the cell is not empty and
    /// paints under `empty-cells: hide`.
    #[test]
    fn empty_cells_hide_keeps_cells_with_element_content() {
        let (_doc, root, list) = layout_and_build(
            "<table style=\"border-collapse: separate; empty-cells: hide\">\
             <tr><td style=\"background: red; border: 2px solid red\"><div></div></td>\
             <td style=\"background: red; border: 2px solid red\"><img></td></tr></table>",
            400.0,
            200.0,
        );
        let cells = table_cells(&root);
        assert_eq!(cells.len(), 2);
        // Both cells paint their own background and border (the `<img>`
        // also paints its missing-image placeholder, which is not the
        // cell's background).
        for (i, cell) in cells.iter().enumerate() {
            assert!(
                fill_rects(&list).iter().any(|r| rect_approx(*r, cell.rect)),
                "cell {i} should paint its background"
            );
            assert!(
                border_rects(&list)
                    .iter()
                    .any(|r| rect_approx(*r, cell.rect)),
                "cell {i} should paint its border"
            );
        }
    }

    /// `empty-cells: show` (and the default, which is `show`) paints
    /// empty cells normally in the separated model.
    #[test]
    fn empty_cells_show_paints_empty_cells() {
        for css in ["empty-cells: show", ""] {
            let (_doc, root, list) = layout_and_build(
                &format!(
                    "<table style=\"border-collapse: separate; {css}\">\
                     <tr><td style=\"background: red; border: 2px solid red\"></td>\
                     <td style=\"background: red; border: 2px solid red\">x</td></tr></table>"
                ),
                400.0,
                200.0,
            );
            let cells = table_cells(&root);
            assert_eq!(cells.len(), 2);
            // Both cells paint their own background and border (rect-
            // based: see the note in
            // `empty_cells_hide_skips_empty_cell_background_and_border`).
            let fills = fill_rects(&list);
            for cell in &cells {
                assert!(
                    fills.iter().any(|r| rect_approx(*r, cell.rect)),
                    "show: {css}: cell should paint its background"
                );
            }
            assert_eq!(border_rects(&list).len(), 2, "show: {css}");
        }
    }

    /// The collapsing border model ignores `empty-cells`: even empty
    /// cells paint their (collapsed) borders.
    #[test]
    fn empty_cells_ignored_in_collapse_model() {
        let (_doc, root, list) = layout_and_build(
            "<table style=\"border-collapse: collapse; empty-cells: hide\">\
             <tr><td style=\"border: 2px solid red\"></td>\
             <td style=\"border: 2px solid red\">x</td></tr></table>",
            400.0,
            200.0,
        );
        let cells = table_cells(&root);
        assert_eq!(cells.len(), 2);
        // One collapsed Border item per cell, including the empty one.
        assert_eq!(border_rects(&list).len(), 2);
    }

    /// Collapsed-border table with *bare* text children (no wrapper
    /// spans): text fragments must not emit borders of their own, so
    /// exactly the per-cell collapsed winners paint — no extra
    /// fragments from the text boxes inheriting the cell/table style.
    #[test]
    fn collapse_table_bare_text_paints_no_extra_borders() {
        let (_doc, _root, list) = layout_and_build(
            "<table style=\"border-collapse: collapse; border: 10px solid green\">\
             <tr><td style=\"border: 2px solid red\">a</td>\
             <td style=\"border: 6px solid blue\">b</td></tr></table>",
            400.0,
            200.0,
        );
        let borders: Vec<([f32; 4], [BorderStyle; 4], [Color; 4])> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::Border {
                    widths,
                    styles,
                    colors,
                    ..
                } => Some((*widths, *styles, *colors)),
                _ => None,
            })
            .collect();
        // One Border item per cell; bare text children add none.
        assert_eq!(borders.len(), 2, "expected one Border item per cell");
        let green = Color::rgba8(0, 128, 0, 255);
        let blue = Color::rgba8(0, 0, 255, 255);
        // The shared edge is painted exactly once: the right cell's left
        // edge carries the 6px blue winner ...
        let shared: Vec<_> = borders.iter().filter(|(w, _, _)| w[3] == 6.0).collect();
        assert_eq!(shared.len(), 1, "shared edge painted exactly once");
        assert_eq!(shared[0].1[3], BorderStyle::Solid);
        assert_eq!(shared[0].2[3], blue);
        // ... and the left cell paints nothing on its (unowned) right edge.
        assert!(
            borders.iter().any(|(w, _, _)| w[1] == 0.0),
            "left cell must not double-paint the shared edge"
        );
        // Outer edges resolve against the table's 10px green border.
        assert!(
            borders
                .iter()
                .any(|(w, s, c)| w[0] == 10.0 && s[0] == BorderStyle::Solid && c[0] == green),
            "table border wins the outer top edge"
        );
        // No Border item is sized like a text fragment: every border's
        // rect must span a full cell or grid line, not a text run.
        for it in &list.items {
            if let DisplayItem::Border { rect, widths, .. } = it {
                let any_side = widths.iter().any(|&w| w > 0.0);
                assert!(any_side, "Border item with no painted side");
                assert!(
                    rect.w > 20.0 || rect.h > 20.0,
                    "no fragment-sized borders, got rect {rect:?}"
                );
            }
        }
    }

    /// General case: a bordered `<div>` with raw text children paints
    /// exactly one Border item. Borders are not inherited, so the
    /// text boxes (which borrow the parent style for text properties)
    /// must not re-emit the parent's border as extra fragments.
    #[test]
    fn bordered_div_with_raw_text_paints_single_border() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"border: 2px solid black; width: 100px; height: 50px\">hello world</div>",
            200.0,
            200.0,
        );
        let borders: Vec<[f32; 4]> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::Border { widths, .. } => Some(*widths),
                _ => None,
            })
            .collect();
        assert_eq!(borders.len(), 1, "expected exactly one Border item");
        assert_eq!(borders[0], [2.0, 2.0, 2.0, 2.0]);
    }

    /// `border-width: 2em` resolves against the element's own font size:
    /// with `font-size: 32px` the painted border is 64px, matching the
    /// border box layout computed (previously painted as 2px).
    #[test]
    fn border_width_em_uses_element_font_size() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"font-size: 32px; border: 2em solid black; width: 10px; height: 10px\"></div>",
            200.0,
            200.0,
        );
        let widths = list.items.iter().find_map(|it| match it {
            DisplayItem::Border { widths, .. } => Some(*widths),
            _ => None,
        });
        assert_eq!(widths.expect("expected a Border item"), [64.0; 4]);
    }

    /// `border-width: 2rem` resolves against the root font size, not the
    /// element's: with the 16px initial root size the border is 32px
    /// even though the element itself is `font-size: 10px`.
    #[test]
    fn border_width_rem_uses_root_font_size() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"font-size: 10px; border: 2rem solid black; width: 10px; height: 10px\"></div>",
            200.0,
            200.0,
        );
        let widths = list.items.iter().find_map(|it| match it {
            DisplayItem::Border { widths, .. } => Some(*widths),
            _ => None,
        });
        assert_eq!(widths.expect("expected a Border item"), [32.0; 4]);
    }

    /// `border-width: 2vw` resolves against the viewport width: 2% of
    /// the 200px test viewport is 4px.
    #[test]
    fn border_width_vw_uses_viewport() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"border: 2vw solid black; width: 10px; height: 10px\"></div>",
            200.0,
            200.0,
        );
        let widths = list.items.iter().find_map(|it| match it {
            DisplayItem::Border { widths, .. } => Some(*widths),
            _ => None,
        });
        assert_eq!(widths.expect("expected a Border item"), [4.0; 4]);
    }

    /// `vh` resolves against the real viewport height, not the document
    /// height: here the document is 10px tall but the viewport is 300px,
    /// so `10vh` is 30px (the old root-rect basis gave 1px).
    #[test]
    fn border_width_vh_uses_viewport_not_document_height() {
        let (_doc, root, list) = layout_and_build(
            "<div style=\"border: 10vh solid black; width: 10px; height: 10px\"></div>",
            200.0,
            300.0,
        );
        assert!(
            root.rect.h < 300.0,
            "test needs document height != viewport height, got {}",
            root.rect.h
        );
        let widths = list.items.iter().find_map(|it| match it {
            DisplayItem::Border { widths, .. } => Some(*widths),
            _ => None,
        });
        assert_eq!(widths.expect("expected a Border item"), [30.0; 4]);
    }

    /// A resolved zero font size stays zero (mirroring layout) instead
    /// of falling back to 16px.
    #[test]
    fn font_size_zero_stays_zero() {
        let (_doc, _root, list) =
            layout_and_build("<div style=\"font-size: 0px\">hi</div>", 200.0, 200.0);
        let sizes: Vec<f32> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::TextRun { size_px, .. } => Some(*size_px),
                _ => None,
            })
            .collect();
        assert!(!sizes.is_empty(), "expected text runs");
        for s in sizes {
            assert_eq!(s, 0.0, "text run size_px = {s}");
        }
    }

    /// `font-size: large` paints text runs at 18px, the same size layout
    /// used for the line boxes (previously fell back to 16px).
    #[test]
    fn font_size_large_keyword_matches_layout() {
        let (_doc, _root, list) =
            layout_and_build("<div style=\"font-size: large\">hi</div>", 200.0, 200.0);
        let sizes: Vec<f32> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::TextRun { size_px, .. } => Some(*size_px),
                _ => None,
            })
            .collect();
        assert!(!sizes.is_empty(), "expected text runs");
        for s in sizes {
            assert!((s - 18.0).abs() < 1e-6, "text run size_px = {s}");
        }
    }

    /// Percentage font sizes inherit: 50% of the parent's 20px paints
    /// the child's text at 10px.
    #[test]
    fn font_size_percentage_inherits_parent_size() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"font-size: 20px\"><span style=\"font-size: 50%\">hi</span></div>",
            200.0,
            200.0,
        );
        let sizes: Vec<f32> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::TextRun { size_px, .. } => Some(*size_px),
                _ => None,
            })
            .collect();
        assert!(!sizes.is_empty(), "expected text runs");
        for s in sizes {
            assert!((s - 10.0).abs() < 1e-6, "text run size_px = {s}");
        }
    }

    /// A deferred abspos box resolves `font-size` against its DOM parent,
    /// not its containing block: `2em` under a 10px DOM parent but a
    /// 30px containing block paints at 20px (the containing-block drain
    /// used to rebuild the unit context from the CB's font size, giving
    /// 60px).
    #[test]
    fn abspos_em_resolves_against_dom_parent_not_containing_block() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"position: relative; font-size: 30px\">\
             <div style=\"font-size: 10px\">\
             <div style=\"position: absolute; font-size: 2em\">hi</div>\
             </div></div>",
            200.0,
            200.0,
        );
        let sizes: Vec<f32> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::TextRun { size_px, .. } => Some(*size_px),
                _ => None,
            })
            .collect();
        assert!(!sizes.is_empty(), "expected text runs");
        for s in sizes {
            assert!((s - 20.0).abs() < 1e-6, "text run size_px = {s}");
        }
    }

    /// Same through the viewport-anchored drain: a `position: fixed`
    /// box's `2em` resolves against its 10px DOM parent (20px), not the
    /// 16px root context the top-level drain used to pass.
    #[test]
    fn fixed_em_resolves_against_dom_parent_not_root() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"font-size: 10px\">\
             <div style=\"position: fixed; font-size: 2em\">hi</div>\
             </div>",
            200.0,
            200.0,
        );
        let sizes: Vec<f32> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::TextRun { size_px, .. } => Some(*size_px),
                _ => None,
            })
            .collect();
        assert!(!sizes.is_empty(), "expected text runs");
        for s in sizes {
            assert!((s - 20.0).abs() < 1e-6, "text run size_px = {s}");
        }
    }

    /// `push_border` stores the corner radii on the item.
    #[test]
    fn push_border_stores_radii() {
        let mut b = DisplayListBuilder::new();
        let r = Rect {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
        };
        b.push_border(
            r,
            [2.0; 4],
            [BorderStyle::Solid; 4],
            [Color::rgba8(255, 0, 0, 255); 4],
            [9.0, 8.0, 7.0, 6.0],
        );
        let list = b.build();
        match &list.items[0] {
            DisplayItem::Border { radii, .. } => {
                assert_eq!(*radii, [9.0, 8.0, 7.0, 6.0]);
            }
            other => panic!("expected Border, got {other:?}"),
        }
    }

    /// `decimal` markers are numbered 1-based across `<li>` siblings.
    ///
    /// (`list-style-type` is not inherited by the CSS cascade yet, so the
    /// test sets it inline per item; inheritance is the CSS worker's.)
    #[test]
    fn decimal_markers_numbered() {
        let (_doc, _root, list) = layout_and_build(
            "<ol><li style=\"list-style-type: decimal\">a</li><li style=\"list-style-type: decimal\">b</li><li style=\"list-style-type: decimal\">c</li></ol>",
            200.0,
            400.0,
        );
        let markers: Vec<(MarkerKind, u32)> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::ListMarker { kind, index, .. } => Some((*kind, *index)),
                _ => None,
            })
            .collect();
        assert_eq!(markers.len(), 3, "one marker per item, got {markers:?}");
        for (i, (kind, index)) in markers.iter().enumerate() {
            let n = (i + 1) as u32;
            assert_eq!(*index, n, "1-based index");
            assert_eq!(*kind, MarkerKind::Decimal(n), "kind carries the number");
        }
    }

    /// `decimal-leading-zero` pads the marker numeral.
    #[test]
    fn decimal_leading_zero_marker() {
        let (_doc, _root, list) = layout_and_build(
            "<ol><li style=\"list-style-type: decimal-leading-zero\">a</li></ol>",
            200.0,
            400.0,
        );
        let marker = list.items.iter().find_map(|it| match it {
            DisplayItem::ListMarker { kind, index, .. } => Some((*kind, *index)),
            _ => None,
        });
        assert_eq!(
            marker.expect("expected a marker"),
            (MarkerKind::DecimalLeadingZero(1), 1)
        );
        // The numeral text both backends render.
        assert_eq!(
            crate::borders::marker_text(MarkerKind::DecimalLeadingZero(1)),
            Some("01".to_string())
        );
    }

    /// `list_item_index` counts preceding `<li>` siblings (DOM-only).
    #[test]
    fn list_item_index_counts_li_siblings() {
        let (doc, _quirks) =
            Document::parse_html("<ol><li>a</li><li>b</li><li>c</li></ol>".as_bytes());
        let items = doc.get_elements_by_tag_name("li");
        assert_eq!(items.len(), 3);
        let got: Vec<u32> = items
            .to_vec()
            .iter()
            .map(|n| list_item_index(&doc, *n))
            .collect();
        assert_eq!(got, vec![1, 2, 3]);
    }

    /// `opacity < 1` wraps the subtree in an opacity group.
    #[test]
    fn opacity_wraps_subtree() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"opacity: 0.5; background: red; width: 10px; height: 10px\"></div>",
            200.0,
            200.0,
        );
        let push = list.items.iter().position(|it| {
            matches!(it, DisplayItem::PushOpacity { opacity } if (*opacity - 0.5).abs() < 1e-6)
        });
        let pop = list
            .items
            .iter()
            .position(|it| matches!(it, DisplayItem::PopOpacity));
        let bg = list.items.iter().position(|it| {
            matches!(it, DisplayItem::FillRect { color, .. } if *color == Color::rgba8(255, 0, 0, 255))
        });
        match (push, bg, pop) {
            (Some(pu), Some(b), Some(po)) => assert!(pu < b && b < po, "fill inside group"),
            _ => panic!("expected opacity group around background"),
        }
    }

    /// Regression: anonymous text boxes inherit their parent's style
    /// reference for inherited text properties, but `opacity` is not
    /// inherited — a text box inside `opacity: 0.5` must not open a
    /// second opacity group (the text would fade twice, 0.25).
    #[test]
    fn opacity_group_opens_once_per_box_not_per_text_run() {
        let (_doc, _root, list) =
            layout_and_build("<div style=\"opacity: 0.5\">hi</div>", 200.0, 200.0);
        let pushes: Vec<f32> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::PushOpacity { opacity } => Some(*opacity),
                _ => None,
            })
            .collect();
        assert_eq!(pushes.len(), 1, "one opacity group, got {pushes:?}");
        assert!((pushes[0] - 0.5).abs() < 1e-6);
        assert!(
            text_runs(&list).iter().any(|(t, _, _)| t.contains("hi")),
            "text still paints"
        );
    }

    /// Unresolvable `<img>` paints the gray placeholder box.
    #[test]
    fn img_placeholder_when_unresolvable() {
        let (_doc, _root, list) =
            layout_and_build("<img src=\"missing.png\" alt=\"x\">", 200.0, 200.0);
        assert!(
            !list
                .items
                .iter()
                .any(|it| matches!(it, DisplayItem::Image { .. })),
            "no Image item without resolved bytes"
        );
        assert!(
            list.items.iter().any(|it| matches!(
                it,
                DisplayItem::FillRect { color, .. } if *color == Color::rgba8(200, 200, 200, 255)
            )),
            "expected gray placeholder"
        );
    }

    /// Resolved `<img>` becomes an Image item with registered bytes.
    #[test]
    fn img_resolves_to_image_item() {
        let (doc, _quirks) = Document::parse_html("<img src=\"a.png\">".as_bytes());
        let author = bowser_css::parse_stylesheet("");
        let root = bowser_layout::layout(
            &doc,
            &[&author],
            Size { w: 200.0, h: 200.0 },
            &bowser_layout::FixedFontMetrics,
        );
        let decoded = DecodedImage::from_rgba(1, 1, vec![1, 2, 3, 255]);
        let list = build_display_list_with_images(
            &root,
            &doc,
            &mut |url| {
                assert_eq!(url, "a.png");
                Some((ImageId(9), decoded.clone()))
            },
            Size { w: 200.0, h: 200.0 },
        );
        assert!(
            list.items
                .iter()
                .any(|it| matches!(it, DisplayItem::Image { image, .. } if *image == ImageId(9))),
            "expected Image item"
        );
        assert!(list.image(ImageId(9)).is_some());
    }

    /// Underlines become Decoration items over the text run.
    #[test]
    fn underline_emits_decoration() {
        let (_doc, _root, list) = layout_and_build(
            "<p style=\"text-decoration: underline\">hi</p>",
            200.0,
            200.0,
        );
        let kinds: Vec<DecoKind> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::Decoration { kind, .. } => Some(*kind),
                _ => None,
            })
            .collect();
        assert!(
            kinds.contains(&DecoKind::Underline),
            "expected underline decoration, got {kinds:?}"
        );
    }

    /// `text-decoration-style: wavy` maps each line to its wavy variant,
    /// and so does `wavy` written inline in the `text-decoration`
    /// shorthand.
    #[test]
    fn wavy_text_decoration_style_emits_wavy_kinds() {
        for style in [
            "text-decoration: underline; text-decoration-style: wavy",
            "text-decoration: underline wavy",
        ] {
            let html = format!("<p style=\"{style}\">hi</p>");
            let (_doc, _root, list) = layout_and_build(&html, 200.0, 200.0);
            let kinds: Vec<DecoKind> = list
                .items
                .iter()
                .filter_map(|it| match it {
                    DisplayItem::Decoration { kind, .. } => Some(*kind),
                    _ => None,
                })
                .collect();
            assert!(
                kinds.contains(&DecoKind::WavyUnderline),
                "style {style:?}: expected wavy underline, got {kinds:?}"
            );
            assert!(
                !kinds.contains(&DecoKind::Underline),
                "style {style:?}: no straight underline when wavy, got {kinds:?}"
            );
        }
    }

    /// Overline and line-through map to their wavy variants too.
    #[test]
    fn wavy_overline_and_line_through() {
        let (_doc, _root, list) = layout_and_build(
            "<p style=\"text-decoration: overline line-through; \
              text-decoration-style: wavy\">hi</p>",
            200.0,
            200.0,
        );
        let kinds: Vec<DecoKind> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::Decoration { kind, .. } => Some(*kind),
                _ => None,
            })
            .collect();
        assert!(
            kinds.contains(&DecoKind::WavyOverline),
            "expected wavy overline, got {kinds:?}"
        );
        assert!(
            kinds.contains(&DecoKind::WavyLineThrough),
            "expected wavy line-through, got {kinds:?}"
        );
    }

    /// The font resolver feeds each text run's key.
    #[test]
    fn font_resolver_feeds_text_runs() {
        let (doc, _quirks) = Document::parse_html("<p>hi</p>".as_bytes());
        let author = bowser_css::parse_stylesheet("");
        let root = bowser_layout::layout(
            &doc,
            &[&author],
            Size { w: 200.0, h: 200.0 },
            &bowser_layout::FixedFontMetrics,
        );
        let list = build_display_list(
            &root,
            &doc,
            &mut |_| FontKey(42),
            &mut |_| None,
            Size { w: 200.0, h: 200.0 },
        );
        for it in &list.items {
            if let DisplayItem::TextRun { font_key, .. } = it {
                assert_eq!(*font_key, FontKey(42));
            }
        }
        assert!(
            list.items
                .iter()
                .any(|it| matches!(it, DisplayItem::TextRun { .. })),
            "expected at least one text run"
        );
    }

    /// The font resolver feeds list markers too, not just text runs:
    /// `Decimal`/`DecimalLeadingZero` numerals render in the item's
    /// font instead of the fallback face.
    #[test]
    fn font_resolver_feeds_list_markers() {
        let (doc, _quirks) = Document::parse_html("<ol><li>hi</li></ol>".as_bytes());
        let author = bowser_css::parse_stylesheet("");
        let root = bowser_layout::layout(
            &doc,
            &[&author],
            Size { w: 200.0, h: 200.0 },
            &bowser_layout::FixedFontMetrics,
        );
        let list = build_display_list(
            &root,
            &doc,
            &mut |_| FontKey(42),
            &mut |_| None,
            Size { w: 200.0, h: 200.0 },
        );
        let mut saw = false;
        for it in &list.items {
            if let DisplayItem::ListMarker { font_key, .. } = it {
                assert_eq!(*font_key, FontKey(42), "marker uses the item's font");
                saw = true;
            }
        }
        assert!(saw, "expected at least one list marker");
    }

    /// `overflow: hidden` wraps the children in a clip to the border box.
    #[test]
    fn overflow_hidden_clips_children() {
        let (doc, root, list) = layout_and_build(
            "<div style=\"overflow: hidden; width: 50px; height: 50px\">\
             <div style=\"width: 200px; height: 200px; background: blue\"></div></div>",
            400.0,
            400.0,
        );
        let divs = doc.get_elements_by_tag_name("div");
        let clip_box = find_box_by_node(&root, divs.item(0).expect("outer div"))
            .expect("outer div layout box");
        let push = list
            .items
            .iter()
            .position(|it| matches!(it, DisplayItem::PushClip { rect } if *rect == clip_box.rect))
            .expect("PushClip with the border box");
        let pop = list
            .items
            .iter()
            .skip(push)
            .position(|it| matches!(it, DisplayItem::PopClip))
            .map(|i| i + push)
            .expect("matching PopClip");
        let blue = list
            .items
            .iter()
            .position(|it| {
                matches!(it, DisplayItem::FillRect { color, .. } if *color == Color::rgba8(0, 0, 255, 255))
            })
            .expect("child background");
        assert!(
            push < blue && blue < pop,
            "child paints between PushClip and PopClip"
        );
    }

    /// `overflow-x: hidden` alone also clips (per-axis longhand).
    #[test]
    fn overflow_x_hidden_clips() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"overflow-x: hidden; width: 50px; height: 50px\">x</div>",
            400.0,
            400.0,
        );
        let pushes = list
            .items
            .iter()
            .filter(|it| matches!(it, DisplayItem::PushClip { .. }))
            .count();
        let pops = list
            .items
            .iter()
            .filter(|it| matches!(it, DisplayItem::PopClip))
            .count();
        assert_eq!((pushes, pops), (1, 1), "one balanced clip pair");
    }

    /// Without `overflow: hidden` no clip items are emitted at all.
    #[test]
    fn overflow_visible_emits_no_clip() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"width: 50px; height: 50px\">\
             <div style=\"width: 200px; height: 200px; background: blue\"></div></div>",
            400.0,
            400.0,
        );
        assert!(
            !list
                .items
                .iter()
                .any(|it| matches!(it, DisplayItem::PushClip { .. } | DisplayItem::PopClip)),
            "no clip items without overflow: hidden"
        );
    }

    /// `overflow: hidden` with `border-radius` emits a rounded clip
    /// whose radii are the resolved border radii (px, CSS-clamped).
    #[test]
    fn overflow_hidden_with_radius_emits_rounded_clip() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"overflow: hidden; border-radius: 10px; width: 50px; height: 50px\">\
             <div style=\"width: 200px; height: 200px; background: blue\"></div></div>",
            400.0,
            400.0,
        );
        let mut saw = false;
        for it in &list.items {
            match it {
                DisplayItem::PushRoundedClip { rect, radii } => {
                    assert_eq!(*radii, [10.0, 10.0, 10.0, 10.0]);
                    assert_eq!((rect.w, rect.h), (50.0, 50.0));
                    saw = true;
                }
                DisplayItem::PushClip { .. } => panic!("expected PushRoundedClip, got PushClip"),
                _ => {}
            }
        }
        assert!(saw, "expected a PushRoundedClip");
    }

    /// `overflow: hidden` without `border-radius` keeps emitting the
    /// plain rect clip (no rounded items).
    #[test]
    fn overflow_hidden_without_radius_emits_plain_clip() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"overflow: hidden; width: 50px; height: 50px\">\
             <div style=\"width: 200px; height: 200px; background: blue\"></div></div>",
            400.0,
            400.0,
        );
        assert!(
            list.items
                .iter()
                .any(|it| matches!(it, DisplayItem::PushClip { .. })),
            "expected a plain PushClip"
        );
        assert!(
            !list
                .items
                .iter()
                .any(|it| matches!(it, DisplayItem::PushRoundedClip { .. })),
            "no rounded clip without border-radius"
        );
    }

    /// `overflow: scroll` and `overflow: auto` clip like `hidden`
    /// (CSS: every non-visible value clips); with `border-radius` the
    /// clip is rounded.
    #[test]
    fn overflow_scroll_and_auto_clip_rounded() {
        for overflow in ["scroll", "auto"] {
            let (_doc, _root, list) = layout_and_build(
                &format!(
                    "<div style=\"overflow: {overflow}; border-radius: 8px; \
                     width: 50px; height: 50px\">\
                     <div style=\"width: 200px; height: 200px; background: blue\"></div></div>"
                ),
                400.0,
                400.0,
            );
            let mut saw = false;
            for it in &list.items {
                if let DisplayItem::PushRoundedClip { radii, .. } = it {
                    assert_eq!(*radii, [8.0; 4]);
                    saw = true;
                }
            }
            assert!(saw, "overflow: {overflow} must emit a rounded clip");
        }
    }

    /// `border-radius: 50%` on a square box resolves to a circular clip
    /// (radius = half the border-box width).
    #[test]
    fn border_radius_50_percent_is_circular() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"overflow: hidden; border-radius: 50%; width: 100px; height: 100px\">\
             <div style=\"width: 200px; height: 200px; background: blue\"></div></div>",
            400.0,
            400.0,
        );
        let mut saw = false;
        for it in &list.items {
            if let DisplayItem::PushRoundedClip { radii, .. } = it {
                assert_eq!(*radii, [50.0, 50.0, 50.0, 50.0]);
                saw = true;
            }
        }
        assert!(saw, "expected a circular PushRoundedClip");
    }

    /// Percentage radii resolve against the border-box width; lengths
    /// that would overflow the box are CSS-clamped (adjacent sums
    /// scale down to the side length).
    #[test]
    fn border_radius_percent_and_clamp() {
        // 25% of 100px = 25px per corner.
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"overflow: hidden; border-radius: 25%; width: 100px; height: 100px\">\
             <div style=\"width: 200px; height: 200px; background: blue\"></div></div>",
            400.0,
            400.0,
        );
        let radii = list
            .items
            .iter()
            .find_map(|it| match it {
                DisplayItem::PushRoundedClip { radii, .. } => Some(*radii),
                _ => None,
            })
            .expect("rounded clip");
        assert_eq!(radii, [25.0, 25.0, 25.0, 25.0]);

        // 100px radii on a 50px box: adjacent sums (200) scale by
        // 50/200, so every corner becomes 25px.
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"overflow: hidden; border-radius: 100px; width: 50px; height: 50px\">\
             <div style=\"width: 200px; height: 200px; background: blue\"></div></div>",
            400.0,
            400.0,
        );
        let radii = list
            .items
            .iter()
            .find_map(|it| match it {
                DisplayItem::PushRoundedClip { radii, .. } => Some(*radii),
                _ => None,
            })
            .expect("rounded clip");
        assert_eq!(radii, [25.0, 25.0, 25.0, 25.0]);
    }

    /// Nested `overflow: hidden` boxes nest their clips, innermost last.
    #[test]
    fn nested_overflow_hidden_balances_clips() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"overflow: hidden; width: 100px; height: 100px\">\
             <div style=\"overflow: hidden; width: 80px; height: 80px\"><p>deep</p></div></div>",
            400.0,
            400.0,
        );
        let pushes: Vec<usize> = list
            .items
            .iter()
            .enumerate()
            .filter(|(_, it)| matches!(it, DisplayItem::PushClip { .. }))
            .map(|(i, _)| i)
            .collect();
        let pops: Vec<usize> = list
            .items
            .iter()
            .enumerate()
            .filter(|(_, it)| matches!(it, DisplayItem::PopClip))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(pushes.len(), 2, "one clip per clipping ancestor");
        assert_eq!(pops.len(), 2, "every clip popped");
        assert!(
            pushes[0] < pushes[1] && pushes[1] < pops[0] && pops[0] < pops[1],
            "clips nest: {pushes:?} / {pops:?}"
        );
        let text = list
            .items
            .iter()
            .position(|it| matches!(it, DisplayItem::TextRun { .. }))
            .expect("text run");
        assert!(
            pushes[0] < text && text < pops[1],
            "innermost content inside both clips"
        );
    }

    /// `visibility: hidden` skips the box, its background, and its whole
    /// subtree; visible siblings still paint. Opacity groups stay balanced.
    #[test]
    fn visibility_hidden_skips_subtree() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"visibility: hidden; opacity: 0.5; background: red; \
             width: 50px; height: 50px\">secret</div><p>shown</p>",
            400.0,
            400.0,
        );
        let runs = text_runs(&list);
        assert!(
            runs.iter().all(|(t, _, _)| !t.contains("secret")),
            "hidden text not painted: {runs:?}"
        );
        assert!(
            runs.iter().any(|(t, _, _)| t.contains("shown")),
            "visible sibling still paints"
        );
        assert!(
            !list.items.iter().any(|it| matches!(
                it,
                DisplayItem::FillRect { color, .. } if *color == Color::rgba8(255, 0, 0, 255)
            )),
            "hidden background not painted"
        );
        let push_op = list
            .items
            .iter()
            .filter(|it| matches!(it, DisplayItem::PushOpacity { .. }))
            .count();
        let pop_op = list
            .items
            .iter()
            .filter(|it| matches!(it, DisplayItem::PopOpacity))
            .count();
        assert_eq!(
            (push_op, pop_op),
            (0, 0),
            "hidden subtree opens no opacity group"
        );
    }

    /// `visibility: hidden` inherits: a child of a hidden box paints
    /// nothing even without the property set on it directly.
    #[test]
    fn visibility_hidden_inherits_to_children() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"visibility: hidden\"><p>buried</p></div>",
            400.0,
            400.0,
        );
        assert!(
            !list
                .items
                .iter()
                .any(|it| matches!(it, DisplayItem::TextRun { .. })),
            "no text runs from a hidden subtree"
        );
    }

    // -- background tiling ------------------------------------------------

    /// One collected [`DisplayItem::RepeatImage`]: (area, image, tile,
    /// repeat, offset).
    type BgRepeat = (Rect, ImageId, (f32, f32), BackgroundRepeat, (f32, f32));

    /// Collect the `RepeatImage` items in a list.
    fn bg_repeats(list: &DisplayList) -> Vec<BgRepeat> {
        list.items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::RepeatImage {
                    area,
                    image,
                    tile,
                    repeat,
                    offset,
                } => Some((*area, *image, *tile, *repeat, *offset)),
                _ => None,
            })
            .collect()
    }

    fn area(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect { x, y, w, h }
    }

    /// Expand a [`background_tile_grid`] spec into the tile rects a
    /// backend paints for one `RepeatImage` item.
    fn grid_tiles(
        cover: Rect,
        origin: (f32, f32),
        tile: (f32, f32),
        repeat_x: bool,
        repeat_y: bool,
    ) -> Vec<Rect> {
        let (i0, j0, nx, ny) = background_tile_grid(cover, origin, tile, repeat_x, repeat_y);
        let mut out = Vec::new();
        for j in j0..j0 + ny {
            for i in i0..i0 + nx {
                out.push(Rect {
                    x: origin.0 + i as f32 * tile.0,
                    y: origin.1 + j as f32 * tile.1,
                    w: tile.0,
                    h: tile.1,
                });
            }
        }
        out
    }

    /// Expand a `space` placement into the tile rects a backend paints.
    fn spaced_tiles(area_: Rect, tile: (f32, f32), offset: (f32, f32)) -> Vec<Rect> {
        let (nx, x0, px) = background_space_axis(area_.x, area_.w, tile.0, offset.0);
        let (ny, y0, py) = background_space_axis(area_.y, area_.h, tile.1, offset.1);
        let mut out = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                out.push(Rect {
                    x: x0 + i as f32 * px,
                    y: y0 + j as f32 * py,
                    w: tile.0,
                    h: tile.1,
                });
            }
        }
        out
    }

    /// `repeat` emits exactly one `RepeatImage` item (no per-tile items,
    /// no clip pair); the backend grid for it covers 100x100 with 30px
    /// tiles in a 4x4 layout.
    #[test]
    fn background_repeat_emits_single_item() {
        let mut b = DisplayListBuilder::new();
        let id = b.add_image(DecodedImage::from_rgba(1, 1, vec![0, 0, 0, 255]));
        b.push_background_image_tiled(
            area(0.0, 0.0, 100.0, 100.0),
            id,
            (30.0, 30.0),
            BackgroundRepeat::Repeat,
            (0.0, 0.0),
        );
        let list = b.build();
        assert_eq!(
            list.items.len(),
            1,
            "one RepeatImage item, no clip pair, no per-tile items"
        );
        let reps = bg_repeats(&list);
        assert_eq!(reps.len(), 1);
        let (a, img, tile, repeat, offset) = reps[0];
        assert_eq!(a, area(0.0, 0.0, 100.0, 100.0));
        assert_eq!(img, id);
        assert_eq!(tile, (30.0, 30.0));
        assert_eq!(repeat, BackgroundRepeat::Repeat);
        assert_eq!(offset, (0.0, 0.0));
        // The grid the backends expand for this item: 4x4 tiles; the
        // edge tile overhangs the area and is trimmed by the area clip
        // the backends apply.
        let tiles = grid_tiles(a, (a.x + offset.0, a.y + offset.1), tile, true, true);
        assert_eq!(tiles.len(), 16);
        let mut want = Vec::new();
        for y in [0.0f32, 30.0, 60.0, 90.0] {
            for x in [0.0f32, 30.0, 60.0, 90.0] {
                want.push(area(x, y, 30.0, 30.0));
            }
        }
        assert_eq!(tiles, want);
        assert!(tiles.iter().any(|r| r.x + r.w > 100.0));
    }

    /// An absurdly small tile is enlarged so the expanded tile count
    /// stays under the cap: the grid still covers the area, and the
    /// backends can never loop over trillions of tiles.
    #[test]
    fn background_repeat_tiny_tile_is_enlarged_to_cap() {
        let mut b = DisplayListBuilder::new();
        let id = b.add_image(DecodedImage::from_rgba(1, 1, vec![0, 0, 0, 255]));
        b.push_background_image_tiled(
            area(0.0, 0.0, 100.0, 100.0),
            id,
            (1e-30, 1e-30),
            BackgroundRepeat::Repeat,
            (0.0, 0.0),
        );
        let list = b.build();
        assert_eq!(list.items.len(), 1);
        let reps = bg_repeats(&list);
        assert_eq!(reps.len(), 1);
        let tile = reps[0].2;
        // 100px / tile per axis, squared, must stay under 65_536 tiles.
        let per_axis = (100.0 / tile.0).ceil();
        assert!(per_axis * per_axis <= 65_536.0, "tile count capped");
        assert!(tile.0 > 1e-30 && tile.1 > 1e-30, "tile enlarged");
        // The enlarged grid still covers the area with no holes.
        assert!(per_axis * tile.0 >= 100.0);
        assert!(per_axis * tile.1 >= 100.0);
    }

    /// [`cap_repeat_tile`] directly: a subnormal tile over the clamped
    /// area is enlarged so the expansion stays under
    /// [`MAX_REPEAT_TILES`] tiles; sane tiles pass through untouched.
    #[test]
    fn cap_repeat_tile_bounds_hostile_tiles() {
        // The exact hostile case from the CPU hang: the smallest
        // positive subnormal tile over an 8192 px area counts ~5.7e96
        // tiles uncapped (i64::MAX loop iterations in the backends).
        let big = clamp_repeat_area(area(0.0, 0.0, 10_000_000.0, 10_000_000.0));
        let sub = f32::from_bits(1);
        let (tw, th) = cap_repeat_tile((sub, sub), big);
        assert!(tw > sub && th > sub, "tile enlarged");
        let total = (big.w as f64 / tw as f64).ceil() * (big.h as f64 / th as f64).ceil();
        assert!(
            total <= MAX_REPEAT_TILES,
            "expanded tile count capped, got {total}"
        );
        // The enlarged grid still covers the area with no holes.
        assert!((big.w as f64 / tw as f64).ceil() * tw as f64 >= big.w as f64);
        assert!((big.h as f64 / th as f64).ceil() * th as f64 >= big.h as f64);
        // Extreme aspect ratios saturate at f32::MAX instead of
        // overflowing to infinity.
        let (tw2, _) = cap_repeat_tile((f32::MAX, sub), big);
        assert!(tw2.is_finite() && tw2 > 0.0);
        // Sane tiles are untouched.
        assert_eq!(cap_repeat_tile((256.0, 256.0), big), (256.0, 256.0));
        assert_eq!(
            cap_repeat_tile((30.0, 30.0), area(0.0, 0.0, 100.0, 100.0)),
            (30.0, 30.0)
        );
    }

    /// [`repeat_snap_band`] scales with the interpolated tile-unit
    /// magnitude (`max(|u0|, |u1|)`), not the span: the f32
    /// interpolation residue the wrap branch must absorb grows with
    /// the `u0` offset included.
    #[test]
    fn repeat_snap_band_scales_with_uv_magnitude() {
        let e = f32::EPSILON;
        // Small span, small magnitude: 4 ulps of the larger endpoint.
        assert_eq!(repeat_snap_band(-0.35, 2.99), 4.0 * e * 2.99);
        // Same span, large offset: the band grows with |u0|.
        let small = repeat_snap_band(-0.35, 2.99);
        let large = repeat_snap_band(-33.35, -30.02);
        assert_eq!(large, 4.0 * e * 33.35);
        assert!(large > 10.0 * small);
        // Floor at 1.0 keeps a nonzero band for tiny spans.
        assert_eq!(repeat_snap_band(0.1, 0.2), 4.0 * e);
        assert_eq!(repeat_snap_band(-0.05, 0.05), 4.0 * e);
        // 0.25 cap keeps adjacent boundary bands from meeting.
        assert_eq!(repeat_snap_band(0.0, 1e7), 0.25);
        assert_eq!(repeat_snap_band(-1e7, 1e7), 0.25);
        // Symmetric endpoints.
        assert_eq!(repeat_snap_band(100.0, -100.0), 4.0 * e * 100.0);
    }

    /// `repeat-x` emits one item; the backend grid is a single row at
    /// the vertical position offset.
    #[test]
    fn background_repeat_x_emits_row_item() {
        let mut b = DisplayListBuilder::new();
        let id = b.add_image(DecodedImage::from_rgba(1, 1, vec![0, 0, 0, 255]));
        b.push_background_image_tiled(
            area(0.0, 0.0, 100.0, 100.0),
            id,
            (30.0, 30.0),
            BackgroundRepeat::RepeatX,
            (0.0, 7.0),
        );
        let list = b.build();
        assert_eq!(list.items.len(), 1);
        let reps = bg_repeats(&list);
        assert_eq!(reps.len(), 1);
        let (a, _, tile, repeat, offset) = reps[0];
        assert_eq!(repeat, BackgroundRepeat::RepeatX);
        assert_eq!(offset, (0.0, 7.0));
        let tiles = grid_tiles(a, (a.x + offset.0, a.y + offset.1), tile, true, false);
        assert_eq!(tiles.len(), 4, "one row of four tiles");
        for (i, r) in tiles.iter().enumerate() {
            assert_eq!((r.x, r.y, r.w, r.h), (i as f32 * 30.0, 7.0, 30.0, 30.0));
        }
    }

    /// `repeat-y` emits one item; the backend grid is a single column at
    /// the horizontal position offset.
    #[test]
    fn background_repeat_y_emits_column_item() {
        let mut b = DisplayListBuilder::new();
        let id = b.add_image(DecodedImage::from_rgba(1, 1, vec![0, 0, 0, 255]));
        b.push_background_image_tiled(
            area(0.0, 0.0, 100.0, 100.0),
            id,
            (30.0, 30.0),
            BackgroundRepeat::RepeatY,
            (11.0, 0.0),
        );
        let list = b.build();
        assert_eq!(list.items.len(), 1);
        let reps = bg_repeats(&list);
        assert_eq!(reps.len(), 1);
        let (a, _, tile, repeat, offset) = reps[0];
        assert_eq!(repeat, BackgroundRepeat::RepeatY);
        let tiles = grid_tiles(a, (a.x + offset.0, a.y + offset.1), tile, false, true);
        assert_eq!(tiles.len(), 4, "one column of four tiles");
        for (j, r) in tiles.iter().enumerate() {
            assert_eq!((r.x, r.y, r.w, r.h), (11.0, j as f32 * 30.0, 30.0, 30.0));
        }
    }

    /// `no-repeat` emits one item; the backend paints exactly one tile
    /// at the position offset.
    #[test]
    fn background_no_repeat_positions_single_tile() {
        let mut b = DisplayListBuilder::new();
        let id = b.add_image(DecodedImage::from_rgba(1, 1, vec![0, 0, 0, 255]));
        b.push_background_image_tiled(
            area(0.0, 0.0, 100.0, 100.0),
            id,
            (30.0, 30.0),
            BackgroundRepeat::NoRepeat,
            (12.0, 34.0),
        );
        let list = b.build();
        assert_eq!(list.items.len(), 1);
        let reps = bg_repeats(&list);
        assert_eq!(reps.len(), 1);
        let (a, _, tile, repeat, offset) = reps[0];
        assert_eq!(repeat, BackgroundRepeat::NoRepeat);
        assert_eq!(offset, (12.0, 34.0));
        let tiles = grid_tiles(a, (a.x + offset.0, a.y + offset.1), tile, false, false);
        assert_eq!(tiles, vec![area(12.0, 34.0, 30.0, 30.0)]);
    }

    /// A positive position offset shifts the whole tile grid: the
    /// backend grid steps back so the area's top-left corner stays
    /// covered.
    #[test]
    fn background_position_offset_shifts_tile_grid() {
        let mut b = DisplayListBuilder::new();
        let id = b.add_image(DecodedImage::from_rgba(1, 1, vec![0, 0, 0, 255]));
        b.push_background_image_tiled(
            area(0.0, 0.0, 100.0, 100.0),
            id,
            (30.0, 30.0),
            BackgroundRepeat::Repeat,
            (10.0, 0.0),
        );
        let reps = bg_repeats(&b.build());
        assert_eq!(reps.len(), 1);
        let (a, _, tile, _, offset) = reps[0];
        assert_eq!(offset, (10.0, 0.0));
        let tiles = grid_tiles(a, (a.x + offset.0, a.y + offset.1), tile, true, true);
        let xs: Vec<f32> = tiles.iter().map(|r| r.x).collect();
        // Grid origin at x=10: first column steps back to x=-20 so the
        // area's left edge is covered.
        assert!(xs.contains(&-20.0), "grid steps back, got {xs:?}");
        assert!(xs.contains(&70.0), "grid covers the right edge, got {xs:?}");
        assert_eq!(tiles.len(), 16);
    }

    /// A fractional area slightly wider than an exact tile multiple is
    /// fully covered: no epsilon is subtracted before the end-index ceil,
    /// so the sub-epsilon sliver at the far edge still gets a tile (the
    /// overhang is trimmed by the area clip the backends apply).
    #[test]
    fn background_tiled_fractional_area_fully_covered() {
        let mut b = DisplayListBuilder::new();
        let id = b.add_image(DecodedImage::from_rgba(1, 1, vec![0, 0, 0, 255]));
        let w = 2.0000005f32;
        b.push_background_image_tiled(
            area(0.0, 0.0, w, 1.0),
            id,
            (1.0, 1.0),
            BackgroundRepeat::Repeat,
            (0.0, 0.0),
        );
        let reps = bg_repeats(&b.build());
        assert_eq!(reps.len(), 1);
        let (a, _, tile, _, offset) = reps[0];
        let tiles = grid_tiles(a, (a.x + offset.0, a.y + offset.1), tile, true, true);
        // Tiles at x=0,1,2: the last one covers the 5e-7 px sliver past
        // x=2 (with the old `- 1e-6` epsilon only 2 tiles were emitted).
        assert_eq!(tiles.len(), 3, "got {tiles:?}");
        let xs: Vec<f32> = tiles.iter().map(|r| r.x).collect();
        assert_eq!(xs, vec![0.0, 1.0, 2.0]);
        let far_edge = tiles.iter().map(|r| r.x + r.w).fold(f32::MIN, f32::max);
        assert!(far_edge >= w, "tile grid must reach the area's right edge");
    }

    /// Degenerate tile or area sizes emit nothing (and never divide by
    /// zero).
    #[test]
    fn background_tiled_degenerate_input_emits_nothing() {
        for (a, tile) in [
            (area(0.0, 0.0, 100.0, 100.0), (0.0, 30.0)),
            (area(0.0, 0.0, 100.0, 100.0), (30.0, 0.0)),
            (area(0.0, 0.0, 100.0, 100.0), (-5.0, 30.0)),
            (area(0.0, 0.0, 0.0, 100.0), (30.0, 30.0)),
        ] {
            let mut b = DisplayListBuilder::new();
            let id = b.add_image(DecodedImage::from_rgba(1, 1, vec![0, 0, 0, 255]));
            b.push_background_image_tiled(a, id, tile, BackgroundRepeat::Repeat, (0.0, 0.0));
            let list = b.build();
            assert!(
                list.items.is_empty(),
                "nothing emitted for {a:?} / {tile:?}"
            );
        }
    }

    /// A 1 px tile over a huge area emits a single `RepeatImage` — the
    /// old 4096 tile-item cap is gone at the display-list level. The
    /// hostile bound now lives in the backends, which clamp the repeat
    /// area to [`MAX_REPEAT_AREA_DIM`] per axis before expanding.
    #[test]
    fn background_tiled_huge_area_is_one_item() {
        let mut b = DisplayListBuilder::new();
        let id = b.add_image(DecodedImage::from_rgba(1, 1, vec![0, 0, 0, 255]));
        let big = area(0.0, 0.0, 100_000.0, 100_000.0);
        b.push_background_image_tiled(big, id, (1.0, 1.0), BackgroundRepeat::Repeat, (0.0, 0.0));
        let list = b.build();
        assert_eq!(
            list.items.len(),
            1,
            "huge repeat is one item, not a capped tile list"
        );
        let reps = bg_repeats(&list);
        assert_eq!(reps.len(), 1);
        assert_eq!(reps[0].0, big, "the item keeps the full area");
        // The backend clamp bounds the expansion: anchored at the
        // origin, each axis capped at MAX_REPEAT_AREA_DIM.
        let clamped = clamp_repeat_area(big);
        assert_eq!(
            clamped,
            area(0.0, 0.0, MAX_REPEAT_AREA_DIM, MAX_REPEAT_AREA_DIM)
        );
        let shifted = clamp_repeat_area(area(50.0, 70.0, 100_000.0, 100_000.0));
        assert_eq!(
            shifted,
            area(50.0, 70.0, MAX_REPEAT_AREA_DIM, MAX_REPEAT_AREA_DIM),
            "clamp anchors at the area origin"
        );
        // Small areas pass through untouched.
        assert_eq!(
            clamp_repeat_area(area(1.0, 2.0, 100.0, 200.0)),
            area(1.0, 2.0, 100.0, 200.0)
        );
        // Degenerate input collapses instead of producing NaN tiles.
        assert_eq!(
            clamp_repeat_area(area(0.0, 0.0, f32::INFINITY, 100.0)).w,
            0.0
        );
    }

    /// A 0.1px tile over a 0.2px-wide area: 2 real columns + 1 harmless
    /// phantom column (float error makes 0.3/0.1 read 2.0000001 in f32).
    /// No epsilon is subtracted before the end-index ceil — a genuine
    /// sub-epsilon sliver must stay covered — so the phantom column is
    /// accepted and the area clip the backends apply trims its overhang.
    #[test]
    fn background_tiled_phantom_tile_trimmed_by_clip_on_float_error() {
        let mut b = DisplayListBuilder::new();
        let id = b.add_image(DecodedImage::from_rgba(1, 1, vec![0, 0, 0, 255]));
        b.push_background_image_tiled(
            area(0.1, 0.0, 0.2, 50.0),
            id,
            (0.1, 10.0),
            BackgroundRepeat::Repeat,
            (0.0, 0.0),
        );
        let reps = bg_repeats(&b.build());
        assert_eq!(reps.len(), 1);
        let (a, _, tile, _, offset) = reps[0];
        let tiles = grid_tiles(a, (a.x + offset.0, a.y + offset.1), tile, true, true);
        // 2 real columns + 1 harmless phantom column, x 5 rows.
        assert_eq!(tiles.len(), 15, "got {tiles:?}");
        // The area stays fully covered edge to edge.
        let left = tiles.iter().map(|r| r.x).fold(f32::MAX, f32::min);
        let right = tiles.iter().map(|r| r.x + r.w).fold(f32::MIN, f32::max);
        assert!(left <= 0.1, "grid covers the area's left edge");
        assert!(right >= 0.1 + 0.2, "grid covers the area's right edge");
    }

    /// `space` emits one item with the native tile size; the backend
    /// placement puts whole tiles flush with both edges, leftover space
    /// spread evenly between them.
    #[test]
    fn background_space_distributes_tiles_evenly() {
        let mut b = DisplayListBuilder::new();
        let id = b.add_image(DecodedImage::from_rgba(1, 1, vec![0, 0, 0, 255]));
        b.push_background_image_tiled(
            area(0.0, 0.0, 100.0, 100.0),
            id,
            (30.0, 30.0),
            BackgroundRepeat::Space,
            (0.0, 0.0),
        );
        let list = b.build();
        assert_eq!(list.items.len(), 1, "one RepeatImage item");
        let reps = bg_repeats(&list);
        assert_eq!(reps.len(), 1);
        let (a, _, tile, repeat, offset) = reps[0];
        assert_eq!(repeat, BackgroundRepeat::Space);
        assert_eq!(tile, (30.0, 30.0), "space keeps the native tile size");
        let tiles = spaced_tiles(a, tile, offset);
        assert_eq!(tiles.len(), 9);
        // 3 tiles of 30px across 100px: 10px of slack, 5px gaps, so the
        // pitch is 35px and the first/last tiles touch the area's edges.
        let mut want = Vec::new();
        for y in [0.0f32, 35.0, 70.0] {
            for x in [0.0f32, 35.0, 70.0] {
                want.push(area(x, y, 30.0, 30.0));
            }
        }
        assert_eq!(tiles, want);
        for r in &tiles {
            assert!(
                r.x >= 0.0 && r.x + r.w <= 100.0,
                "space never clips a tile, got {r:?}"
            );
        }
        assert!(tiles.iter().any(|r| (r.x + r.w - 100.0).abs() < 1e-5));
    }

    /// `space` with room for a single tile keeps the
    /// `background-position` offset, like `no-repeat`.
    #[test]
    fn background_space_single_tile_uses_position_offset() {
        let mut b = DisplayListBuilder::new();
        let id = b.add_image(DecodedImage::from_rgba(1, 1, vec![0, 0, 0, 255]));
        b.push_background_image_tiled(
            area(0.0, 0.0, 100.0, 100.0),
            id,
            (60.0, 60.0),
            BackgroundRepeat::Space,
            (10.0, 20.0),
        );
        let reps = bg_repeats(&b.build());
        assert_eq!(reps.len(), 1);
        let (a, _, tile, repeat, offset) = reps[0];
        assert_eq!(repeat, BackgroundRepeat::Space);
        let tiles = spaced_tiles(a, tile, offset);
        assert_eq!(tiles, vec![area(10.0, 20.0, 60.0, 60.0)]);
    }

    /// `round` rescales the tile so a whole number of tiles spans each
    /// axis exactly, then tiles like `repeat` (normalized at emit time).
    #[test]
    fn background_round_rescales_to_whole_tiles() {
        let mut b = DisplayListBuilder::new();
        let id = b.add_image(DecodedImage::from_rgba(1, 1, vec![0, 0, 0, 255]));
        b.push_background_image_tiled(
            area(0.0, 0.0, 100.0, 100.0),
            id,
            (24.0, 24.0),
            BackgroundRepeat::Round,
            (0.0, 0.0),
        );
        let list = b.build();
        assert_eq!(list.items.len(), 1);
        let reps = bg_repeats(&list);
        assert_eq!(reps.len(), 1);
        let (a, _, tile, repeat, offset) = reps[0];
        // round(100/24) = 4 tiles of exactly 25px each per axis.
        assert!(
            (tile.0 - 25.0).abs() < 1e-4,
            "tile width rescaled, got {tile:?}"
        );
        assert!(
            (tile.1 - 25.0).abs() < 1e-4,
            "tile height rescaled, got {tile:?}"
        );
        assert_eq!(
            repeat,
            BackgroundRepeat::Repeat,
            "round normalizes to repeat"
        );
        let tiles = grid_tiles(a, (a.x + offset.0, a.y + offset.1), tile, true, true);
        assert_eq!(tiles.len(), 16, "4x4 rescaled tiles, got {tiles:?}");
        for (i, r) in tiles.iter().enumerate() {
            let (col, row) = (i % 4, i / 4);
            assert!((r.x - col as f32 * 25.0).abs() < 1e-3, "got {r:?}");
            assert!((r.y - row as f32 * 25.0).abs() < 1e-3, "got {r:?}");
        }
        let right = tiles.iter().map(|r| r.x + r.w).fold(f32::MIN, f32::max);
        assert!(
            (right - 100.0).abs() < 1e-3,
            "last tile ends exactly at the area edge"
        );
    }

    /// `round` picks the whole tile count nearest the natural count and
    /// stretches/shrinks the tile to span the axis exactly.
    #[test]
    fn round_tile_len_snaps_to_whole_counts() {
        assert!((round_tile_len(100.0, 24.0) - 25.0).abs() < 1e-5);
        assert!((round_tile_len(100.0, 30.0) - 100.0 / 3.0).abs() < 1e-5);
        // Fewer than one natural tile: stretch to fill the axis.
        assert!((round_tile_len(100.0, 90.0) - 100.0).abs() < 1e-5);
        assert!((round_tile_len(100.0, 60.0) - 50.0).abs() < 1e-5);
    }

    /// End to end: `background-repeat: repeat-x` from a real stylesheet
    /// emits one row item with native-size tiles across the box.
    #[test]
    fn background_repeat_x_from_style() {
        let (_doc, _quirks) = Document::parse_html(
            "<div style=\"background-image: url(a.png); background-repeat: repeat-x; \
             width: 100px; height: 60px\"></div>"
                .as_bytes(),
        );
        let author = bowser_css::parse_stylesheet("");
        let root = bowser_layout::layout(
            &_doc,
            &[&author],
            Size { w: 400.0, h: 400.0 },
            &bowser_layout::FixedFontMetrics,
        );
        let div = _doc.get_elements_by_tag_name("div").item(0).expect("div");
        let div_box = find_box_by_node(&root, div).expect("div layout box");
        let decoded = DecodedImage::from_rgba(10, 10, vec![9u8; 10 * 10 * 4]);
        let list = build_display_list_with_images(
            &root,
            &_doc,
            &mut |url| {
                assert_eq!(url, "a.png");
                Some((ImageId(5), decoded.clone()))
            },
            Size { w: 400.0, h: 400.0 },
        );
        let reps = bg_repeats(&list);
        assert_eq!(reps.len(), 1, "one repeat item for the background");
        let (a, _, tile, repeat, offset) = reps[0];
        assert_eq!(a, div_box.rect, "area is the border box");
        assert_eq!(tile, (10.0, 10.0), "native tile size");
        assert_eq!(repeat, BackgroundRepeat::RepeatX);
        assert_eq!(offset, (0.0, 0.0));
        // The backend grid for this item: 100px / 10px tile = 10 tiles
        // in one row at the box top.
        let tiles = grid_tiles(a, (a.x + offset.0, a.y + offset.1), tile, true, false);
        assert_eq!(tiles.len(), 10, "got {tiles:?}");
        for (i, r) in tiles.iter().enumerate() {
            assert_eq!(r.y, div_box.rect.y, "single row at the box top");
            assert!(
                (r.x - (div_box.rect.x + i as f32 * 10.0)).abs() < 1e-3,
                "tiles step by 10px, got {r:?}"
            );
        }
    }

    /// End to end: `background-image: linear-gradient(...)` emits one
    /// `DisplayItem::Gradient` over the border box, with no tiling
    /// (`background-repeat` / `background-position` do not apply to
    /// gradients) and no `RepeatImage` item.
    #[test]
    fn background_linear_gradient_from_style() {
        let (_doc, _quirks) = Document::parse_html(
            "<div style=\"background-image: linear-gradient(to right, red, blue); \
             width: 100px; height: 60px\"></div>"
                .as_bytes(),
        );
        let author = bowser_css::parse_stylesheet("");
        let root = bowser_layout::layout(
            &_doc,
            &[&author],
            Size { w: 400.0, h: 400.0 },
            &bowser_layout::FixedFontMetrics,
        );
        let div = _doc.get_elements_by_tag_name("div").item(0).expect("div");
        let div_box = find_box_by_node(&root, div).expect("div layout box");
        let list = build_display_list_with_images(
            &root,
            &_doc,
            &mut |_url| None,
            Size { w: 400.0, h: 400.0 },
        );
        let grads: Vec<(Rect, Gradient)> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::Gradient { rect, gradient } => Some((*rect, gradient.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(grads.len(), 1, "one gradient item for the background");
        assert_eq!(grads[0].0, div_box.rect, "rect is the border box");
        assert!(
            grads[0].1.stops.len() >= 2,
            "gradient has the two declared stops"
        );
        assert!(
            bg_repeats(&list).is_empty(),
            "gradients never emit RepeatImage tiles"
        );
    }

    /// End to end: a `background-image` gradient the paint parser
    /// rejects (`linear-gradient(25%, red, blue)` — a bare percentage
    /// first arg is not a valid gradient line) emits nothing, matching
    /// the rendering of a dropped declaration.
    #[test]
    fn background_invalid_gradient_emits_nothing() {
        let (_doc, _quirks) = Document::parse_html(
            "<div style=\"background-image: linear-gradient(25%, red, blue); \
             width: 100px; height: 60px\"></div>"
                .as_bytes(),
        );
        let author = bowser_css::parse_stylesheet("");
        let root = bowser_layout::layout(
            &_doc,
            &[&author],
            Size { w: 400.0, h: 400.0 },
            &bowser_layout::FixedFontMetrics,
        );
        let list = build_display_list_with_images(
            &root,
            &_doc,
            &mut |_url| None,
            Size { w: 400.0, h: 400.0 },
        );
        assert!(
            !list.items.iter().any(|it| matches!(
                it,
                DisplayItem::Gradient { .. } | DisplayItem::RepeatImage { .. }
            )),
            "invalid gradient paints no background item"
        );
    }

    /// Regression (harsh-review finding): a gradient `background-image`
    /// on an element with a text child emits exactly ONE
    /// `DisplayItem::Gradient`, over the element's border box.
    /// `background-image` is not inherited, so the text child (own style
    /// `None`, computing to `none`) must not re-emit the parent's
    /// gradient over its own smaller rect — the old code resolved that
    /// second gradient against the smaller rect, a hard visible seam.
    #[test]
    fn background_gradient_text_child_emits_single_item() {
        let (_doc, _quirks) = Document::parse_html(
            "<div style=\"background-image: linear-gradient(to right, red, blue); \
             width: 100px; height: 60px\">hello</div>"
                .as_bytes(),
        );
        let author = bowser_css::parse_stylesheet("");
        let root = bowser_layout::layout(
            &_doc,
            &[&author],
            Size { w: 400.0, h: 400.0 },
            &bowser_layout::FixedFontMetrics,
        );
        let div = _doc.get_elements_by_tag_name("div").item(0).expect("div");
        let div_box = find_box_by_node(&root, div).expect("div layout box");
        let list = build_display_list_with_images(
            &root,
            &_doc,
            &mut |_url| None,
            Size { w: 400.0, h: 400.0 },
        );
        let grads: Vec<Rect> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::Gradient { rect, .. } => Some(*rect),
                _ => None,
            })
            .collect();
        assert_eq!(
            grads.len(),
            1,
            "one gradient item: the text child must not re-emit the parent's gradient"
        );
        assert_eq!(grads[0], div_box.rect, "rect is the div's border box");
    }

    /// The builder memoizes parsed gradients by verbatim source string:
    /// N boxes sharing one gradient class parse the source once per
    /// build, and failed parses are memoized too (no repeated doomed
    /// parses — a failed gradient emits nothing either way).
    #[test]
    fn builder_memoizes_parsed_gradients() {
        let mut b = DisplayListBuilder::new();
        let src = "linear-gradient(to right, red, blue)";
        let first = b.parsed_gradient(src);
        assert!(first.is_some(), "valid source parses");
        let second = b.parsed_gradient(src);
        assert_eq!(first, second, "memoized result is identical");
        assert_eq!(b.gradient_cache_len(), 1, "one cache entry per source");
        assert!(b.parsed_gradient("radial-gradient(red, blue)").is_some());
        assert!(
            b.parsed_gradient("linear-gradient(25%, red, blue)")
                .is_none()
        );
        assert!(
            b.parsed_gradient("linear-gradient(25%, red, blue)")
                .is_none()
        );
        assert_eq!(b.gradient_cache_len(), 3, "misses are memoized too");
    }

    /// w58: the gradient memo shares one allocation across boxes.
    /// `parsed_gradient` returns `Arc`s now: repeated lookups must hand
    /// out the *same* allocation (`Arc::ptr_eq`), proving the per-hit
    /// deep clone of the stop list is gone, while the values stay equal
    /// (bit-identical rendering).
    #[test]
    fn parsed_gradient_cache_shares_one_allocation() {
        let mut b = DisplayListBuilder::new();
        let src = "linear-gradient(to right, red, blue)";
        let first = b.parsed_gradient(src).expect("valid source parses");
        let second = b.parsed_gradient(src).expect("cache hit parses");
        assert!(
            Arc::ptr_eq(&first, &second),
            "cache hit shares the allocation instead of cloning the stop list"
        );
        assert_eq!(first, second, "shared value is identical");
        // A failed parse memoizes `None` without allocating either.
        assert!(
            b.parsed_gradient("linear-gradient(25%, red, blue)")
                .is_none()
        );
        assert!(
            b.parsed_gradient("linear-gradient(25%, red, blue)")
                .is_none()
        );
    }

    /// The filter cache keys on the length context and `currentcolor`,
    /// not just the function text: two boxes sharing `blur(1em)` with
    /// different font sizes resolve different px radii, and a
    /// currentcolor `drop-shadow()` resolves the shadow color per box.
    /// Same context still hits the cache (F7).
    #[test]
    fn resolved_filter_cache_keys_on_ctx_and_currentcolor() {
        use bowser_css::FilterFunction;
        let blur_em = Filter::List(vec![FilterFunction {
            name: "blur".to_string(),
            raw_args: "1em".to_string(),
        }]);
        let black = Color::rgba8(0, 0, 0, 255);
        let red = Color::rgba8(255, 0, 0, 255);
        let ctx16 = LengthCtx {
            em: 16.0,
            root: 16.0,
            viewport: (800.0, 600.0),
        };
        let ctx32 = LengthCtx { em: 32.0, ..ctx16 };
        let mut b = DisplayListBuilder::new();
        // First box populates the cache…
        let ops16 = b.resolved_filter(&blur_em, &ctx16, black);
        assert_eq!(ops16.as_slice(), &[FilterOp::Blur(16.0)]);
        // …but a box with a different font size must not reuse it.
        let ops32 = b.resolved_filter(&blur_em, &ctx32, black);
        assert_eq!(ops32.as_slice(), &[FilterOp::Blur(32.0)]);
        // Same font size reuses the cached entry.
        assert_eq!(b.resolved_filter(&blur_em, &ctx16, black), ops16);

        // `drop-shadow()` with no explicit color defaults to
        // currentcolor: different `color`s resolve different shadow
        // colors.
        let ds = Filter::List(vec![FilterFunction {
            name: "drop-shadow".to_string(),
            raw_args: "2px 2px".to_string(),
        }]);
        let black_ops = b.resolved_filter(&ds, &ctx16, black);
        let red_ops = b.resolved_filter(&ds, &ctx16, red);
        assert_ne!(black_ops, red_ops, "shadow color follows currentcolor");
        assert!(matches!(
            black_ops.as_slice(),
            [FilterOp::DropShadow { color, .. }] if *color == black
        ));
        assert!(matches!(
            red_ops.as_slice(),
            [FilterOp::DropShadow { color, .. }] if *color == red
        ));
    }

    /// `DisplayItem` size budget (wave-32 shrink: 120 B -> ~72 B; the
    /// inline-first stop storage moves it to 88 B — still well under
    /// the pre-shrink 120 B). If this fails, a variant grew past the
    /// budget: shrink the new field or spill it to the heap before
    /// accepting the regression.
    #[test]
    fn display_item_stays_within_size_budget() {
        let size = std::mem::size_of::<DisplayItem>();
        assert!(
            size <= 88,
            "DisplayItem is {size} B, over the 88 B budget (was 80 B before inline stops)"
        );
    }

    /// `FilterOps` goes inline for the typical 1-2 op chain and spills
    /// for longer ones; the deref'd slice covers exactly the live ops.
    #[test]
    fn filter_ops_inline_and_spill() {
        let one = FilterOps::from(vec![FilterOp::Blur(4.0)]);
        assert!(matches!(one, FilterOps::Inline { len: 1, .. }));
        assert_eq!(one.as_slice(), &[FilterOp::Blur(4.0)]);

        let two = FilterOps::from(vec![FilterOp::Blur(4.0), FilterOp::Brightness(1.5)]);
        assert!(matches!(two, FilterOps::Inline { len: 2, .. }));
        assert_eq!(two.len(), 2);

        let three = FilterOps::from(vec![
            FilterOp::Blur(4.0),
            FilterOp::Brightness(1.5),
            FilterOp::Contrast(0.5),
        ]);
        assert!(matches!(three, FilterOps::Spill(_)));
        assert_eq!(three.len(), 3);
        assert_eq!(three[2], FilterOp::Contrast(0.5));

        assert!(FilterOps::new().is_empty());
    }

    /// `ClipPoints` goes inline for typical authored polygons (<= 8
    /// vertices) and spills for tessellated circles and the like.
    #[test]
    fn clip_points_inline_and_spill() {
        let tri = ClipPoints::from(vec![[10.0, 10.0], [90.0, 10.0], [50.0, 90.0]]);
        assert!(matches!(tri, ClipPoints::Inline { len: 3, .. }));
        assert_eq!(tri.as_slice().len(), 3);
        assert_eq!(tri[2], [50.0, 90.0]);

        let many: Vec<[f32; 2]> = (0..32).map(|i| [i as f32, 0.0]).collect();
        let spilled = ClipPoints::from(many);
        assert!(matches!(spilled, ClipPoints::Spill(_)));
        assert_eq!(spilled.len(), 32);
        assert_eq!(spilled[31], [31.0, 0.0]);
    }

    /// `build_display_list_reused` emits exactly the same items as
    /// `build_display_list`: reuse changes allocation behavior, never
    /// pixels. Covers gradients (inline stops), filters (inline ops),
    /// and clip-paths (inline points) on one page.
    #[test]
    fn reused_build_matches_fresh_build() {
        let html = concat!(
            "<div style=\"width: 100px; height: 50px; ",
            "background: linear-gradient(red, blue); filter: blur(2px);\">",
            "<div style=\"width: 20px; height: 20px; ",
            "clip-path: polygon(0 0, 10px 0, 10px 10px);\"></div>",
            "</div>",
        );
        let (doc, _quirks) = Document::parse_html(html.as_bytes());
        let author = bowser_css::parse_stylesheet("");
        let viewport = Size { w: 400.0, h: 400.0 };
        let root =
            bowser_layout::layout(&doc, &[&author], viewport, &bowser_layout::FixedFontMetrics);
        let mut fonts = |_: &ComputedStyle| FontKey(0);
        let mut images = |_: &str| None;
        let fresh = build_display_list(&root, &doc, &mut fonts, &mut images, viewport);

        let mut builder = DisplayListBuilder::new();
        let mut out = DisplayList::new();
        // First reused build: cold caches (style changed).
        build_display_list_reused(
            &mut builder,
            &mut out,
            &root,
            &doc,
            &mut fonts,
            &mut images,
            viewport,
            true,
        );
        assert_eq!(format!("{:?}", fresh.items), format!("{:?}", out.items));
        // Second reused build: warm caches, recycled storage.
        build_display_list_reused(
            &mut builder,
            &mut out,
            &root,
            &doc,
            &mut fonts,
            &mut images,
            viewport,
            false,
        );
        assert_eq!(format!("{:?}", fresh.items), format!("{:?}", out.items));
        // The two-stop gradient's stops went inline, not to the heap.
        assert!(
            out.items.iter().any(|it| matches!(
                it,
                DisplayItem::Gradient { gradient, .. }
                if matches!(gradient.stops, GradientStops::Inline { .. })
            )),
            "expected an inline-stops gradient item"
        );
    }

    /// Reusing the builder recycles the previous frame's item storage:
    /// the second identical build reallocates nothing (same backing
    /// buffer, same capacity).
    #[test]
    fn reused_builder_recycles_item_storage() {
        let (doc, _quirks) =
            Document::parse_html("<div style=\"width: 100px; height: 50px;\">hi</div>".as_bytes());
        let author = bowser_css::parse_stylesheet("");
        let viewport = Size { w: 400.0, h: 400.0 };
        let root =
            bowser_layout::layout(&doc, &[&author], viewport, &bowser_layout::FixedFontMetrics);
        let mut fonts = |_: &ComputedStyle| FontKey(0);
        let mut images = |_: &str| None;
        let mut build = |builder: &mut DisplayListBuilder, out: &mut DisplayList| {
            build_display_list_reused(
                builder,
                out,
                &root,
                &doc,
                &mut fonts,
                &mut images,
                viewport,
                false,
            );
        };

        let mut builder = DisplayListBuilder::new();
        let mut out = DisplayList::new();
        build(&mut builder, &mut out);
        assert!(!out.items.is_empty(), "fixture must produce items");
        let (ptr, cap) = (out.items.as_ptr(), out.items.capacity());
        build(&mut builder, &mut out);
        assert_eq!(
            out.items.as_ptr(),
            ptr,
            "second build reused the item buffer instead of reallocating"
        );
        assert_eq!(out.items.capacity(), cap);
    }

    /// The style epoch: scroll-only frames keep warm caches, a restyle
    /// clears them (memory bound — the caches are keyed by complete
    /// inputs, so entries can never go stale).
    #[test]
    fn builder_epoch_clears_caches_on_style_change() {
        let mut b = DisplayListBuilder::new();
        let mut out = DisplayList::new();
        b.parsed_gradient("linear-gradient(red, blue)");
        assert_eq!(b.gradient_cache_len(), 1);
        // Scroll-only frame: caches survive.
        b.begin_reused_build(&mut out, false);
        b.finish_reused_build(&mut out);
        assert_eq!(b.gradient_cache_len(), 1, "warm cache survives");
        assert_eq!(b.style_epoch, 0, "epoch untouched without a restyle");
        // Restyle: epoch bumps, caches clear.
        b.begin_reused_build(&mut out, true);
        b.finish_reused_build(&mut out);
        assert_eq!(b.gradient_cache_len(), 0, "restyle invalidates the epoch");
        assert_eq!(b.style_epoch, 1);
    }

    /// End to end: `background-repeat: round` rescales the native tile
    /// so a whole number of tiles fits each axis exactly.
    #[test]
    fn background_round_from_style() {
        let (_doc, _quirks) = Document::parse_html(
            "<div style=\"background-image: url(a.png); background-repeat: round; \
             width: 100px; height: 100px\"></div>"
                .as_bytes(),
        );
        let author = bowser_css::parse_stylesheet("");
        let root = bowser_layout::layout(
            &_doc,
            &[&author],
            Size { w: 400.0, h: 400.0 },
            &bowser_layout::FixedFontMetrics,
        );
        let div = _doc.get_elements_by_tag_name("div").item(0).expect("div");
        let div_box = find_box_by_node(&root, div).expect("div layout box");
        let decoded = DecodedImage::from_rgba(24, 24, vec![9u8; 24 * 24 * 4]);
        let list = build_display_list_with_images(
            &root,
            &_doc,
            &mut |url| {
                assert_eq!(url, "a.png");
                Some((ImageId(5), decoded.clone()))
            },
            Size { w: 400.0, h: 400.0 },
        );
        let reps = bg_repeats(&list);
        assert_eq!(reps.len(), 1);
        let (a, _, tile, repeat, _) = reps[0];
        assert_eq!(a, div_box.rect);
        // round(100/24) = 4 tiles per axis, rescaled 24px -> 25px.
        assert!(
            (tile.0 - 25.0).abs() < 1e-4,
            "tile width rescaled, got {tile:?}"
        );
        assert!(
            (tile.1 - 25.0).abs() < 1e-4,
            "tile height rescaled, got {tile:?}"
        );
        assert_eq!(repeat, BackgroundRepeat::Repeat);
    }

    /// End to end: `background-repeat: space` places whole unclipped
    /// tiles with the leftover space spread evenly between them.
    #[test]
    fn background_space_from_style() {
        // 100x60 box, 30px tiles: 3 columns (5px gaps) x 2 rows (no gap).
        let (_doc, _quirks) = Document::parse_html(
            "<div style=\"background-image: url(a.png); background-repeat: space; \
             width: 100px; height: 60px\"></div>"
                .as_bytes(),
        );
        let author = bowser_css::parse_stylesheet("");
        let root = bowser_layout::layout(
            &_doc,
            &[&author],
            Size { w: 400.0, h: 400.0 },
            &bowser_layout::FixedFontMetrics,
        );
        let div = _doc.get_elements_by_tag_name("div").item(0).expect("div");
        let div_box = find_box_by_node(&root, div).expect("div layout box");
        let decoded = DecodedImage::from_rgba(30, 30, vec![9u8; 30 * 30 * 4]);
        let list = build_display_list_with_images(
            &root,
            &_doc,
            &mut |url| {
                assert_eq!(url, "a.png");
                Some((ImageId(5), decoded.clone()))
            },
            Size { w: 400.0, h: 400.0 },
        );
        let reps = bg_repeats(&list);
        assert_eq!(reps.len(), 1);
        let (a, _, tile, repeat, offset) = reps[0];
        assert_eq!(repeat, BackgroundRepeat::Space);
        assert_eq!(tile, (30.0, 30.0), "tiles keep native size");
        let tiles = spaced_tiles(a, tile, offset);
        assert_eq!(tiles.len(), 6, "3x2 spaced tiles, got {tiles:?}");
        // First column at the box's left edge, last column flush with the
        // right edge; the 10px of slack is split into two 5px gaps.
        let mut xs: Vec<f32> = tiles.iter().map(|r| r.x - div_box.rect.x).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs.dedup();
        assert_eq!(xs.len(), 3, "three columns, got {xs:?}");
        assert!((xs[0] - 0.0).abs() < 1e-3, "first tile at the left edge");
        assert!((xs[1] - 35.0).abs() < 1e-3, "even 5px gap, got {xs:?}");
        assert!(
            (xs[2] - 70.0).abs() < 1e-3,
            "last tile flush right, got {xs:?}"
        );
        for r in &tiles {
            assert!(
                r.x >= div_box.rect.x && r.x + r.w <= div_box.rect.x + div_box.rect.w,
                "space never clips a tile, got {r:?}"
            );
        }
    }

    /// End to end: `background-position: 25% 10px` with `no-repeat`
    /// positions the single tile per CSS (percentage of the slack).
    #[test]
    fn background_position_from_style() {
        let (_doc, _quirks) = Document::parse_html(
            "<div style=\"background-image: url(a.png); background-repeat: no-repeat; \
             background-position: 25% 10px; width: 100px; height: 60px\"></div>"
                .as_bytes(),
        );
        let author = bowser_css::parse_stylesheet("");
        let root = bowser_layout::layout(
            &_doc,
            &[&author],
            Size { w: 400.0, h: 400.0 },
            &bowser_layout::FixedFontMetrics,
        );
        let div = _doc.get_elements_by_tag_name("div").item(0).expect("div");
        let div_box = find_box_by_node(&root, div).expect("div layout box");
        let decoded = DecodedImage::from_rgba(10, 10, vec![9u8; 10 * 10 * 4]);
        let list = build_display_list_with_images(
            &root,
            &_doc,
            &mut |url| {
                assert_eq!(url, "a.png");
                Some((ImageId(5), decoded.clone()))
            },
            Size { w: 400.0, h: 400.0 },
        );
        let reps = bg_repeats(&list);
        assert_eq!(reps.len(), 1, "no-repeat paints one item");
        let (a, _, tile, repeat, offset) = reps[0];
        assert_eq!(repeat, BackgroundRepeat::NoRepeat);
        // x = 25% of (100 - 10) = 22.5; y = 10px.
        assert!((offset.0 - 22.5).abs() < 1e-3, "got {offset:?}");
        assert!((offset.1 - 10.0).abs() < 1e-3, "got {offset:?}");
        let tiles = grid_tiles(a, (a.x + offset.0, a.y + offset.1), tile, false, false);
        assert_eq!(tiles.len(), 1);
        let r = tiles[0];
        assert!((r.x - (div_box.rect.x + 22.5)).abs() < 1e-3, "got {r:?}");
        assert!((r.y - (div_box.rect.y + 10.0)).abs() < 1e-3, "got {r:?}");
    }

    /// End to end: the `background` shorthand's single `center` keyword
    /// centers the tile on both axes.
    #[test]
    fn background_position_center_keyword_from_style() {
        let (_doc, _quirks) = Document::parse_html(
            "<div style=\"background: url(a.png) no-repeat center; \
             width: 100px; height: 60px\"></div>"
                .as_bytes(),
        );
        let author = bowser_css::parse_stylesheet("");
        let root = bowser_layout::layout(
            &_doc,
            &[&author],
            Size { w: 400.0, h: 400.0 },
            &bowser_layout::FixedFontMetrics,
        );
        let div = _doc.get_elements_by_tag_name("div").item(0).expect("div");
        let div_box = find_box_by_node(&root, div).expect("div layout box");
        let reps = bg_repeats(&build_display_list_with_images(
            &root,
            &_doc,
            &mut |url| {
                assert_eq!(url, "a.png");
                let decoded = DecodedImage::from_rgba(10, 10, vec![9u8; 10 * 10 * 4]);
                Some((ImageId(5), decoded))
            },
            Size { w: 400.0, h: 400.0 },
        ));
        assert_eq!(reps.len(), 1);
        let (a, _, tile, repeat, offset) = reps[0];
        assert_eq!(repeat, BackgroundRepeat::NoRepeat);
        assert!((offset.0 - 45.0).abs() < 1e-3, "got {offset:?}");
        assert!((offset.1 - 25.0).abs() < 1e-3, "got {offset:?}");
        let tiles = grid_tiles(a, (a.x + offset.0, a.y + offset.1), tile, false, false);
        assert_eq!(tiles.len(), 1);
        let r = tiles[0];
        assert!((r.x - (div_box.rect.x + 45.0)).abs() < 1e-3, "got {r:?}");
        assert!((r.y - (div_box.rect.y + 25.0)).abs() < 1e-3, "got {r:?}");
    }

    /// End to end: `background-position: 2em 0` with `font-size: 16px`
    /// resolves the x offset through the element's font size (2em = 32px),
    /// not as a raw px value. Regression: non-px length units used to be
    /// passed through as-is (2px).
    #[test]
    fn background_position_em_unit_from_style() {
        let (_doc, _quirks) = Document::parse_html(
            "<div style=\"background-image: url(a.png); background-repeat: no-repeat; \
             background-position: 2em 0; font-size: 16px; width: 100px; height: 60px\"></div>"
                .as_bytes(),
        );
        let author = bowser_css::parse_stylesheet("");
        let root = bowser_layout::layout(
            &_doc,
            &[&author],
            Size { w: 400.0, h: 400.0 },
            &bowser_layout::FixedFontMetrics,
        );
        let div = _doc.get_elements_by_tag_name("div").item(0).expect("div");
        let div_box = find_box_by_node(&root, div).expect("div layout box");
        let decoded = DecodedImage::from_rgba(10, 10, vec![9u8; 10 * 10 * 4]);
        let list = build_display_list_with_images(
            &root,
            &_doc,
            &mut |url| {
                assert_eq!(url, "a.png");
                Some((ImageId(5), decoded.clone()))
            },
            Size { w: 400.0, h: 400.0 },
        );
        let reps = bg_repeats(&list);
        assert_eq!(reps.len(), 1, "no-repeat paints one item");
        let (a, _, tile, _, offset) = reps[0];
        // x = 2em at 16px font = 32px; y = 0% of the slack = 0.
        assert!((offset.0 - 32.0).abs() < 1e-3, "got {offset:?}");
        assert!((offset.1 - 0.0).abs() < 1e-3, "got {offset:?}");
        let tiles = grid_tiles(a, (a.x + offset.0, a.y + offset.1), tile, false, false);
        assert_eq!(tiles.len(), 1);
        let r = tiles[0];
        assert!((r.x - (div_box.rect.x + 32.0)).abs() < 1e-3, "got {r:?}");
        assert!((r.y - div_box.rect.y).abs() < 1e-3, "got {r:?}");
    }

    // -- texture-cache validation -----------------------------------------

    /// Identical bytes hash identically (deterministic cache key).
    #[test]
    fn content_hash_stable_for_identical_bytes() {
        let a = DecodedImage::from_rgba(2, 2, vec![7u8; 16]);
        let b = DecodedImage::from_rgba(2, 2, vec![7u8; 16]);
        assert_eq!(a.content_hash(), b.content_hash());
    }

    /// A single flipped byte in the middle of the buffer changes the
    /// hash. This is the adversarial case for the old sparse head/tail
    /// sample: a mutation it could never see.
    #[test]
    fn content_hash_changes_on_mid_buffer_mutation() {
        let mut bytes = vec![7u8; 64];
        let before = DecodedImage::from_rgba(4, 4, bytes.clone()).content_hash();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        let after = DecodedImage::from_rgba(4, 4, bytes).content_hash();
        assert_ne!(before, after, "mid-buffer mutation must change the hash");
    }

    /// Same bytes under different dimensions hash differently.
    #[test]
    fn content_hash_covers_dimensions() {
        let bytes = vec![7u8; 16];
        let a = DecodedImage::from_rgba(2, 2, bytes.clone()).content_hash();
        let b = DecodedImage::from_rgba(4, 1, bytes).content_hash();
        assert_ne!(a, b, "dimensions are part of the identity");
    }

    /// Re-registering an [`ImageId`] with different bytes (the aliasing
    /// scenario: ids are per-list, lists are rebuilt) changes the hash a
    /// texture cache validates against, so the stale entry can never hit.
    #[test]
    fn reregistered_image_changes_hash() {
        let mut list = DisplayList::new();
        let id = ImageId(3);
        list.add_image(id, DecodedImage::from_rgba(2, 2, vec![1u8; 16]));
        let stale = list.image(id).expect("registered").content_hash();
        // Same id, mutated bytes (e.g. the image re-resolved on rebuild).
        list.add_image(id, DecodedImage::from_rgba(2, 2, vec![2u8; 16]));
        let fresh = list.image(id).expect("re-registered").content_hash();
        assert_ne!(
            stale, fresh,
            "a cache keyed on (id, hash) must miss after re-registration"
        );
    }

    /// In-place mutation of a standalone `DecodedImage` (never registered:
    /// the list caches the hash at registration, where the bytes are
    /// immutable afterward) is covered because `content_hash()` hashes the
    /// current bytes on demand.
    #[test]
    fn content_hash_sees_in_place_mutation() {
        let mut img = DecodedImage::from_rgba(2, 2, vec![1u8; 16]);
        let before = img.content_hash();
        img.rgba[8] = 42;
        assert_ne!(before, img.content_hash());
    }

    /// `add_image` computes the content hash once at registration and
    /// caches it, so backends never rehash the pixel buffer per frame; a
    /// re-registered id refreshes the cached hash with the new bytes.
    #[test]
    fn add_image_caches_content_hash_at_registration() {
        let mut list = DisplayList::new();
        let id = ImageId(7);
        assert_eq!(list.image_content_hash(id), None);
        let img = DecodedImage::from_rgba(2, 2, vec![1u8; 16]);
        let expected = img.content_hash();
        list.add_image(id, img);
        assert_eq!(list.image_content_hash(id), Some(expected));
        assert_eq!(list.image_content_hash(ImageId(999)), None);
        // Same id, different bytes: the cached hash tracks the newest
        // registration, so a cache keyed on (id, hash) misses.
        let img2 = DecodedImage::from_rgba(2, 2, vec![2u8; 16]);
        let expected2 = img2.content_hash();
        assert_ne!(expected, expected2);
        list.add_image(id, img2);
        assert_eq!(list.image_content_hash(id), Some(expected2));
    }

    /// Hostile border widths are sanitized through `sane_len` before
    /// painting: `border: 1e30px solid black` paints at the finite
    /// clamp, not at infinity (issue 9).
    #[test]
    fn huge_border_width_is_sanitized() {
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"border: 1e30px solid black; width: 10px; height: 10px\"></div>",
            200.0,
            200.0,
        );
        let widths = list.items.iter().find_map(|it| match it {
            DisplayItem::Border { widths, .. } => Some(*widths),
            _ => None,
        });
        let widths = widths.expect("expected a Border item");
        assert!(
            widths.iter().all(|w| w.is_finite()),
            "border widths must be finite, got {:?}",
            widths
        );
        assert_eq!(widths, [bowser_layout::sane_len(1e30); 4]);
    }

    /// Hostile `background-position` lengths go through `sane_len` like
    /// every other resolved paint length: `1e30px` resolves to the
    /// finite clamp, not to infinity (paint must never lay out at a
    /// value layout could not have produced).
    #[test]
    fn huge_background_position_length_is_sanitized() {
        use bowser_css::LengthUnit;
        let hostile = CssValue::Length(1e30, LengthUnit::Px);
        let x =
            background_position_component(&hostile, 200.0, 50.0, true, 16.0, 16.0, (800.0, 600.0));
        assert!(
            x.is_finite(),
            "background-position x must be finite, got {x}"
        );
        assert_eq!(x, bowser_layout::sane_len(1e30));
        // Unitless numbers take the same path.
        let y = background_position_component(
            &CssValue::Number(-1e30),
            200.0,
            50.0,
            false,
            16.0,
            16.0,
            (800.0, 600.0),
        );
        assert!(
            y.is_finite(),
            "background-position y must be finite, got {y}"
        );
        assert_eq!(y, bowser_layout::sane_len(-1e30));
    }

    /// `mix-blend-mode` display-list wrapping: a non-normal mode wraps
    /// the whole subtree in `PushBlend`/`PopBlend`, with the blend
    /// group *outside* the opacity group (blend against the real page
    /// backdrop, then fade); `normal` (and an absent property) emits
    /// no blend items at all.
    #[test]
    fn blend_mode_wraps_subtree() {
        use bowser_css::{Viewport, compute_style, default_ua_stylesheet, parse_stylesheet};

        fn list_for(style_attr: &str) -> DisplayList {
            let mut doc = Document::new();
            let root = doc.root();
            let div = doc.create_element("div");
            doc.append_child(root, div);
            doc.set_attribute(div, "style", style_attr);
            let ua = default_ua_stylesheet();
            let vp = Viewport { w: 800.0, h: 600.0 };
            let author = parse_stylesheet("");
            let style = compute_style(&doc, div, &[&author], &ua, &vp);
            let layout_box = LayoutBox {
                rect: Rect {
                    x: 10.0,
                    y: 10.0,
                    w: 100.0,
                    h: 100.0,
                },
                children: vec![],
                box_type: BoxType::Block,
                style: Some(std::rc::Rc::new(style)),
                text: None,
                node: Some(div),
                fragments: vec![],
                z_index: None,
                collapsed_borders: None,
                phantom_w: false,
                phantom_h: false,
            };
            build_display_list(
                &layout_box,
                &doc,
                &mut |_| FontKey(0),
                &mut |_| None,
                Size { w: 800.0, h: 600.0 },
            )
        }

        fn kinds(list: &DisplayList) -> Vec<&'static str> {
            list.items
                .iter()
                .map(|it| match it {
                    DisplayItem::PushBlend { .. } => "push_blend",
                    DisplayItem::PopBlend => "pop_blend",
                    DisplayItem::PushOpacity { .. } => "push_opacity",
                    DisplayItem::PopOpacity => "pop_opacity",
                    DisplayItem::FillRect { .. } => "fill",
                    _ => "other",
                })
                .collect()
        }

        // Blend outer, opacity inner, background fill inside both.
        let list = list_for("mix-blend-mode: multiply; opacity: 0.5; background-color: red;");
        assert_eq!(
            kinds(&list),
            [
                "push_blend",
                "push_opacity",
                "fill",
                "pop_opacity",
                "pop_blend"
            ]
        );
        assert!(
            matches!(
                list.items[0],
                DisplayItem::PushBlend {
                    mode: BlendMode::Multiply
                }
            ),
            "expected PushBlend Multiply, got {:?}",
            list.items[0]
        );

        // `normal` and an absent property emit no blend items.
        for css in ["", "mix-blend-mode: normal;"] {
            let list = list_for(&format!("{css} background-color: red;"));
            assert!(
                list.items
                    .iter()
                    .all(|it| !matches!(it, DisplayItem::PushBlend { .. } | DisplayItem::PopBlend)),
                "{css:?} must not emit blend items: {:?}",
                kinds(&list)
            );
        }
    }

    /// `BlendMode::from_css_keyword` accepts exactly the six supported
    /// keywords (the cascade already rejects the rest; this is the
    /// second line of defense for hand-built lists).
    #[test]
    fn blend_mode_keywords_parse() {
        let cases = [
            ("normal", BlendMode::Normal),
            ("multiply", BlendMode::Multiply),
            ("screen", BlendMode::Screen),
            ("overlay", BlendMode::Overlay),
            ("darken", BlendMode::Darken),
            ("lighten", BlendMode::Lighten),
        ];
        for (keyword, mode) in cases {
            assert_eq!(BlendMode::from_css_keyword(keyword), Some(mode));
        }
        assert_eq!(BlendMode::from_css_keyword("difference"), None);
        assert_eq!(BlendMode::from_css_keyword(""), None);
        assert!(BlendMode::Normal.is_normal());
        assert!(!BlendMode::Multiply.is_normal());
    }

    // -- box-shadow / text-shadow emission ------------------------------

    /// One emitted `BoxShadow`: (rect, radii, color, offset, blur,
    /// spread, inset).
    type BoxShadowSpec = (Rect, [f32; 4], Color, (f32, f32), f32, f32, bool);

    /// The `BoxShadow` items in list order.
    fn box_shadow_items(list: &DisplayList) -> Vec<BoxShadowSpec> {
        list.items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::BoxShadow {
                    rect,
                    radii,
                    color,
                    offset,
                    blur,
                    spread,
                    inset,
                } => Some((*rect, *radii, *color, *offset, *blur, *spread, *inset)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn box_shadow_outset_below_background_inset_above() {
        let (doc, root, list) = layout_and_build(
            r#"<div style="width:40px;height:40px;background:red;box-shadow:5px 6px 7px 8px blue, inset 1px 2px 3px green"></div>"#,
            200.0,
            200.0,
        );
        let shadows = box_shadow_items(&list);
        assert_eq!(
            shadows.len(),
            2,
            "one outset + one inset layer, got {shadows:?}"
        );
        // Outset layers emit before the background, inset layers after
        // it; within each group layers list front-to-back in CSS and
        // emit back-to-front here.
        let outset = &shadows[0];
        assert!(!outset.6, "first emitted layer is the outset one");
        assert_eq!(outset.3, (5.0, 6.0), "outset offset");
        assert_eq!(outset.2, Color::rgba8(0, 0, 255, 255), "outset color");
        assert_eq!((outset.4, outset.5), (7.0, 8.0), "outset blur/spread");
        let inset = &shadows[1];
        assert!(inset.6, "second emitted layer is the inset one");
        assert_eq!(inset.3, (1.0, 2.0), "inset offset");
        assert_eq!(inset.2, Color::rgba8(0, 128, 0, 255), "inset color");
        assert_eq!((inset.4, inset.5), (3.0, 0.0), "inset blur/spread");
        for s in &shadows {
            assert_eq!(s.1, [0.0; 4], "no border-radius set");
        }
        // Paint order: outset shadow, background fill, inset shadow.
        let mut outset_idx = None;
        let mut fill_idx = None;
        let mut inset_idx = None;
        for (i, it) in list.items.iter().enumerate() {
            match it {
                DisplayItem::BoxShadow { inset: false, .. } => outset_idx = Some(i),
                DisplayItem::BoxShadow { inset: true, .. } => inset_idx = Some(i),
                DisplayItem::FillRect { .. } => {
                    fill_idx.get_or_insert(i);
                }
                _ => {}
            }
        }
        let (o, f, n) = (
            outset_idx.expect("outset shadow"),
            fill_idx.expect("background fill"),
            inset_idx.expect("inset shadow"),
        );
        assert!(
            o < f && f < n,
            "outset < background < inset, got {o} < {f} < {n}"
        );
        // The shadow rect is the div's border box.
        let div_node = doc.get_elements_by_tag_name("div").item(0).expect("div");
        let div_box = find_box_by_node(&root, div_node).expect("div box");
        assert_eq!(shadows[0].0, div_box.rect);
        assert_eq!(shadows[1].0, div_box.rect);
    }

    #[test]
    fn box_shadow_layers_emit_back_to_front() {
        let (_doc, _root, list) = layout_and_build(
            r#"<div style="width:40px;height:40px;box-shadow:1px 1px red, 2px 2px blue"></div>"#,
            200.0,
            200.0,
        );
        let shadows = box_shadow_items(&list);
        assert_eq!(shadows.len(), 2, "two outset layers, got {shadows:?}");
        // CSS lists front-to-back (red on top); the display list paints
        // back-to-front, so the blue layer emits first.
        assert_eq!(shadows[0].3, (2.0, 2.0), "back layer first");
        assert_eq!(shadows[0].2, Color::rgba8(0, 0, 255, 255));
        assert_eq!(shadows[1].3, (1.0, 1.0), "front layer second");
        assert_eq!(shadows[1].2, Color::rgba8(255, 0, 0, 255));
    }

    #[test]
    fn box_shadow_currentcolor_and_em_lengths() {
        let (_doc, _root, list) = layout_and_build(
            r#"<div style="width:40px;height:40px;font-size:20px;color:#00ff00;box-shadow:1em 2px"></div>"#,
            200.0,
            200.0,
        );
        let shadows = box_shadow_items(&list);
        assert_eq!(shadows.len(), 1, "one layer, got {shadows:?}");
        let s = &shadows[0];
        assert_eq!(s.3, (20.0, 2.0), "1em resolves against font-size");
        // Omitted shadow color defaults to `currentcolor`.
        assert_eq!(s.2, Color::rgba8(0, 255, 0, 255), "currentcolor");
        assert!(!s.6);
    }

    #[test]
    fn box_shadow_absent_means_no_items() {
        let (_doc, _root, list) = layout_and_build(
            r#"<div style="width:40px;height:40px;background:red"></div>"#,
            200.0,
            200.0,
        );
        assert!(
            box_shadow_items(&list).is_empty(),
            "no shadow property, no shadow items"
        );
    }

    #[test]
    fn text_shadow_emits_under_each_run() {
        let (_doc, _root, list) = layout_and_build(
            r#"<p style="font-size:16px;text-shadow:2px 3px 4px red">hi</p>"#,
            200.0,
            200.0,
        );
        // Each TextRun is immediately preceded by its TextShadow.
        let mut run_count = 0;
        for (i, it) in list.items.iter().enumerate() {
            if let DisplayItem::TextRun { .. } = it {
                run_count += 1;
                match &list.items[i - 1] {
                    DisplayItem::TextShadow {
                        offset,
                        blur,
                        color,
                        size_px,
                        ..
                    } => {
                        assert_eq!(*offset, (2.0, 3.0), "shadow offset");
                        assert_eq!(*blur, 4.0, "shadow blur");
                        assert_eq!(*color, Color::rgba8(255, 0, 0, 255));
                        assert_eq!(*size_px, 16.0);
                    }
                    other => panic!("TextRun not preceded by TextShadow: {other:?}"),
                }
            }
        }
        assert!(run_count > 0, "expected at least one text run");
    }

    #[test]
    fn text_shadow_layers_emit_back_to_front() {
        let (_doc, _root, list) = layout_and_build(
            r#"<p style="font-size:16px;text-shadow:1px 1px red, 2px 2px blue">hi</p>"#,
            200.0,
            200.0,
        );
        let offsets: Vec<(f32, f32)> = list
            .items
            .iter()
            .filter_map(|it| match it {
                DisplayItem::TextShadow { offset, .. } => Some(*offset),
                _ => None,
            })
            .collect();
        assert!(!offsets.is_empty(), "expected text-shadow items");
        // Front-to-back in CSS (red on top of blue); back-to-front in
        // the display list, so the blue layer emits first per run.
        for pair in offsets.chunks(2) {
            assert_eq!(
                pair,
                &[(2.0, 2.0), (1.0, 1.0)],
                "back layer first, got {pair:?}"
            );
        }
    }

    #[test]
    fn shadow_length_resolves_deferred_calc() {
        let v = CssValue::MathDeferred {
            pct: 0.0,
            px: 5.0,
            had_pct: false,
            affine: true,
        };
        assert_eq!(resolve_shadow_length(&v, 16.0, 16.0, (800.0, 600.0)), 5.0);
        // Shadows have no percentage basis: a nonzero percentage
        // coefficient cannot resolve.
        let v = CssValue::MathDeferred {
            pct: 50.0,
            px: 5.0,
            had_pct: true,
            affine: true,
        };
        assert_eq!(resolve_shadow_length(&v, 16.0, 16.0, (800.0, 600.0)), 0.0);
        // Blur clamps at the emission site, not here.
        assert_eq!(
            resolve_shadow_length(&CssValue::Number(0.0), 16.0, 16.0, (800.0, 600.0)),
            0.0
        );
    }

    #[test]
    fn resolve_length_handles_deferred_calc() {
        // Frozen base-0 form (`pct == 0.0`): non-affine `max(10px, 0%)`
        // contributes its constant instead of painting at 0.
        let v = CssValue::MathDeferred {
            pct: 0.0,
            px: 10.0,
            had_pct: true,
            affine: false,
        };
        assert_eq!(resolve_length(Some(&v), 16.0, 16.0, (800.0, 600.0)), 10.0);
        // A nonzero percentage coefficient has no base in this path:
        // it still falls back to 0.0 (the basis-aware
        // `resolve_length_pct` handles those).
        let v = CssValue::MathDeferred {
            pct: 50.0,
            px: 10.0,
            had_pct: true,
            affine: true,
        };
        assert_eq!(resolve_length(Some(&v), 16.0, 16.0, (800.0, 600.0)), 0.0);
    }

    #[test]
    fn resolve_length_pct_handles_deferred_calc() {
        // Frozen base-0 form: `padding-top: max(10px, 0%)` paints at the
        // base-0 value (10px), not 0.
        let v = CssValue::MathDeferred {
            pct: 0.0,
            px: 10.0,
            had_pct: true,
            affine: false,
        };
        assert_eq!(
            resolve_length_pct(Some(&v), 16.0, 16.0, (800.0, 600.0), 200.0),
            10.0
        );
        // Affine form: the percentage coefficient resolves against the
        // real basis (50% of 200px + 4px).
        let v = CssValue::MathDeferred {
            pct: 50.0,
            px: 4.0,
            had_pct: true,
            affine: true,
        };
        assert_eq!(
            resolve_length_pct(Some(&v), 16.0, 16.0, (800.0, 600.0), 200.0),
            104.0
        );
    }

    #[test]
    fn resolve_radius_handles_deferred_calc() {
        // Frozen base-0 form: `border-top-left-radius: max(10px, 5%)`
        // paints at the base-0 value (10px), not 0.
        let v = CssValue::MathDeferred {
            pct: 0.0,
            px: 10.0,
            had_pct: true,
            affine: false,
        };
        assert_eq!(
            resolve_radius(Some(&v), 16.0, 16.0, (800.0, 600.0), 100.0),
            10.0
        );
        // Affine form: the percentage coefficient resolves against the
        // border-box width (50% of 100px).
        let v = CssValue::MathDeferred {
            pct: 50.0,
            px: 0.0,
            had_pct: true,
            affine: true,
        };
        assert_eq!(
            resolve_radius(Some(&v), 16.0, 16.0, (800.0, 600.0), 100.0),
            50.0
        );
    }

    /// End to end: non-affine percentage math in `border-radius`
    /// reaches the rounded clip at its base-0 value, not 0.
    #[test]
    fn border_radius_non_affine_math_paints_base0_value() {
        // `max(10px, 5%)` freezes as `MathDeferred { pct: 0.0, px: 10.0,
        // had_pct: true }`; the clip must use 10px per corner.
        let (_doc, _root, list) = layout_and_build(
            "<div style=\"overflow: hidden; border-radius: max(10px, 5%); width: 100px; height: 100px\">\
             <div style=\"width: 200px; height: 200px; background: blue\"></div></div>",
            400.0,
            400.0,
        );
        let radii = list
            .items
            .iter()
            .find_map(|it| match it {
                DisplayItem::PushRoundedClip { radii, .. } => Some(*radii),
                _ => None,
            })
            .expect("rounded clip");
        assert_eq!(radii, [10.0, 10.0, 10.0, 10.0]);
    }
}

/// Wave 66 perf tests: `gradient_cache` moved from `RandomState`
/// (SipHash) to the local FNV-1a hasher. Behavior pins: the cache
/// semantics are unchanged — only the hash function changed, and the
/// borrow through `Box<str>: Borrow<str>` still allocates nothing on hit.
#[cfg(test)]
mod w66_p_paint_hash_tests {
    use crate::fonts::FnvHashMap;
    use std::collections::HashMap;
    use std::time::Instant;

    /// FNV-1a and SipHash maps agree on identical gradient-source
    /// insert sequences, and `get(&str)` probes hit through `Borrow<str>`.
    #[test]
    fn gradient_cache_hash_agreement() {
        let sources: Vec<Box<str>> = (0..300)
            .map(|i: usize| {
                format!("linear-gradient({i}deg, rgba(255,0,0,0.5) 0%, #00ff00 {i}px, blue 100%)")
                    .into_boxed_str()
            })
            .collect();
        let sip: HashMap<Box<str>, usize> = sources
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        let fnv: FnvHashMap<Box<str>, usize> = sources
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        for s in &sources {
            let probe: &str = s;
            assert_eq!(sip.get(probe), fnv.get(probe), "source must agree");
        }
        assert!(!fnv.contains_key("linear-gradient(1deg, #fff, #000)"));
    }

    /// Micro-benchmark: FNV-1a vs SipHash `get` on full gradient-source
    /// strings. Prints ns/get; asserts nothing (shared CI boxes are too
    /// noisy for timing asserts).
    #[test]
    fn gradient_source_get_timing() {
        let sources: Vec<Box<str>> = (0..300)
            .map(|i: usize| {
                format!("linear-gradient({i}deg, rgba(255,0,0,0.5) 0%, #00ff00 {i}px, blue 100%)")
                    .into_boxed_str()
            })
            .collect();
        let sip: HashMap<Box<str>, usize> = sources
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        let fnv: FnvHashMap<Box<str>, usize> = sources
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        let time_get = |get: &dyn Fn(&str) -> usize| {
            let start = Instant::now();
            let mut acc = 0u64;
            let n = 200 * sources.len();
            for i in 0..n {
                let key: &str = &sources[i % sources.len()];
                acc += get(key) as u64;
            }
            (start.elapsed().as_nanos() as f64 / n as f64, acc)
        };
        let (sip_ns, sip_acc) = time_get(&|k| *sip.get(k).unwrap_or(&0));
        let (fnv_ns, fnv_acc) = time_get(&|k| *fnv.get(k).unwrap_or(&0));
        assert_eq!(sip_acc, fnv_acc, "both maps return the same values");
        println!(
            "w66 gradient-source keys: SipHash {sip_ns:.1}ns/get vs FNV-1a {fnv_ns:.1}ns/get ({:.2}x)",
            sip_ns / fnv_ns.max(0.01)
        );
    }
}
