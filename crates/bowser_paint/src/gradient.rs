//! CSS gradient parsing, resolution, and rasterization.
//!
//! Covers `linear-gradient()`, `radial-gradient()`, `conic-gradient()`
//! and their `repeating-` variants (CSS Images 3/4).
//!
//! `bowser_css` has no gradient value representation (a gradient in
//! `background-image` currently fails value parsing there), so this
//! module parses the CSS function text itself. The pipeline is:
//!
//! 1. [`parse_gradient`] turns the function text into a
//!    [`ParsedGradient`] (colors, unresolved stop positions, geometry
//!    with lengths/percentages as written).
//! 2. [`ParsedGradient::resolve`] binds it to a concrete box
//!    ([`Rect`]), applying the CSS color-stop fixup rules and resolving
//!    percentages/lengths to gradient-relative fractions, producing a
//!    backend-neutral [`Gradient`].
//! 3. Backends rasterize the resolved gradient: [`sample_gradient`]
//!    evaluates one page-px point, [`bake_gradient_rgba8`] fills a
//!    straight-RGBA8 buffer at device resolution. Both backends share
//!    this code, so CPU and GPU pixels agree by construction.
//!
//! Color interpolation follows CSS Color 4: premultiplied Oklab, the
//! default interpolation space for gradients when no explicit
//! interpolation method is given.
//!
//! Deliberate limitations (documented, not silent):
//!
//! - Stop positions, explicit radial sizes, and `at` positions accept
//!   percentages, `px` lengths, and unitless zero only. Other length
//!   units (`em`, `rem`, `vw`, ...) are a parse error; they need font
//!   and viewport context the display-list builder does not thread
//!   through yet.
//! - `currentcolor` stops are parsed and resolved against a caller
//!   supplied color (see [`ParsedGradient::resolve`]).
//! - The CSS Images 4 `in <color-space>` interpolation-method prelude
//!   is not accepted; interpolation is always the default Oklab.

use crate::display_list::Color;
use bowser_layout::Rect;
use std::fmt;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Why [`parse_gradient`] rejected its input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GradientParseError {
    message: String,
}

impl GradientParseError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Human-readable reason for the rejection.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for GradientParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid gradient: {}", self.message)
    }
}

impl std::error::Error for GradientParseError {}

// ---------------------------------------------------------------------------
// Parsed (unresolved) representation
// ---------------------------------------------------------------------------

/// Which gradient function was parsed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradientFunc {
    Linear,
    RepeatingLinear,
    Radial,
    RepeatingRadial,
    Conic,
    RepeatingConic,
}

impl GradientFunc {
    /// Whether this is one of the `repeating-` variants.
    pub fn is_repeating(self) -> bool {
        matches!(
            self,
            GradientFunc::RepeatingLinear
                | GradientFunc::RepeatingRadial
                | GradientFunc::RepeatingConic
        )
    }
}

/// A color stop's color: a concrete color or `currentcolor` (resolved
/// against the element's `color` at [`ParsedGradient::resolve`] time).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopColor {
    Rgba(Color),
    CurrentColor,
}

/// A stop/hint position as written: a percentage of the gradient's
/// reference length, or an absolute px length along it.
///
/// The reference length is the gradient-line length for linear
/// gradients, the ending-shape radius (horizontal gradient ray) for
/// radial gradients, and 360 degrees for conic gradients (which only
/// accept angles and percentages).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StopPos {
    Pct(f32),
    Px(f32),
}

/// One component of an `at <position>`: percentage of the box
/// dimension, an absolute px offset from the box origin, or an offset
/// inward from the far edge (`right 10px`, `bottom 5%`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PosComp {
    Pct(f32),
    Px(f32),
    /// `100% - v` (from `right v%` / `bottom v%`).
    FarPct(f32),
    /// `dim - v` px (from `right vpx` / `bottom vpx`).
    FarPx(f32),
}

impl PosComp {
    /// Resolve against a box dimension in px, returning a 0..1 fraction.
    fn fraction(self, dim: f32) -> f32 {
        match self {
            PosComp::Pct(v) => v / 100.0,
            PosComp::Px(v) => v / dim,
            PosComp::FarPct(v) => 1.0 - v / 100.0,
            PosComp::FarPx(v) => (dim - v) / dim,
        }
    }
}

/// An `at <position>` value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientPosition {
    pub x: PosComp,
    pub y: PosComp,
}

/// Explicit radial ending-shape size as written.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RadialSize {
    /// One of the `closest-side`/`farthest-side`/`closest-corner`/
    /// `farthest-corner` extent keywords.
    Extent(RadialExtent),
    /// A single `<length>`: a circle with that radius.
    CirclePx(f32),
    /// Two `<length-percentage>`s: an ellipse with those horizontal and
    /// vertical radii (percentages resolve against the box width and
    /// height respectively).
    Ellipse(PosComp, PosComp),
}

/// Radial ending-shape extent keywords.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadialExtent {
    ClosestSide,
    FarthestSide,
    ClosestCorner,
    FarthestCorner,
}

/// One parsed color stop: a color with zero, one, or two positions
/// (double-position syntax, e.g. `red 10% 20%`, is two stops sharing a
/// color and is expanded during [`ParsedGradient::resolve`]).
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedStop {
    pub color: StopColor,
    /// `positions[0]` is the stop's position; `positions[1]` is `Some`
    /// only for double-position syntax.
    pub positions: [Option<StopPos>; 2],
}

/// One entry of a parsed color-stop list: either a color stop or a
/// transition hint (a bare position between two stops, e.g.
/// `linear-gradient(red, 30%, blue)`).
#[derive(Clone, Debug, PartialEq)]
pub enum ParsedStopItem {
    Stop(ParsedStop),
    Hint(StopPos),
}

/// A linear gradient's direction as written: either an explicit
/// angle, or a `to <corner>` keyword pair. Corners cannot be angles:
/// per CSS Images 4 the gradient line runs corner-to-corner, so the
/// angle depends on the box's aspect ratio and is only known at
/// [`ParsedGradient::resolve`] time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LinearDir {
    /// CSS angle in radians: 0 points up (`to top`), clockwise.
    Angle(f32),
    /// `to <corner>` (e.g. `to top right`): resolved from the box in
    /// [`Self::angle_for`].
    Corner { to_top: bool, to_right: bool },
}

impl LinearDir {
    /// Resolve to a CSS angle in radians for a `w` x `h` box. A corner
    /// direction is the box diagonal: `to top right` is
    /// `atan2(w, h)` (CSS convention: 0 = up, clockwise), so a wide
    /// box gets a shallower angle than a tall one.
    pub fn angle_for(self, w: f32, h: f32) -> f32 {
        match self {
            LinearDir::Angle(a) => a,
            LinearDir::Corner { to_top, to_right } => {
                let dx = if to_right { w } else { -w };
                let dy = if to_top { -h } else { h };
                dx.atan2(-dy)
            }
        }
    }
}

/// A parsed CSS gradient, not yet bound to a box. Build with
/// [`parse_gradient`], then call [`ParsedGradient::resolve`].
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedGradient {
    /// Which function was parsed (encodes the repeating variants).
    pub func: GradientFunc,
    /// Linear-gradient direction. `None` for other functions.
    pub linear_dir: Option<LinearDir>,
    /// Radial center/shape/size. `None` for other functions.
    pub radial: Option<ParsedRadial>,
    /// Conic `from` angle (radians, CSS convention) and center.
    /// `None` for other functions.
    pub conic: Option<ParsedConic>,
    /// The color-stop list: at least two stops, hints only between
    /// stops.
    pub stops: Vec<ParsedStopItem>,
}

/// Parsed radial-gradient geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParsedRadial {
    /// `circle` or `ellipse` (default `ellipse`).
    pub circle: bool,
    pub center: GradientPosition,
    pub size: RadialSize,
}

/// Parsed conic-gradient geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParsedConic {
    /// `from` angle in radians (CSS convention: 0 = up, clockwise).
    pub from: f32,
    pub center: GradientPosition,
}

// ---------------------------------------------------------------------------
// Resolved (display-ready) representation
// ---------------------------------------------------------------------------

/// One resolved color stop: a concrete color at an absolute offset in
/// `[0, 1]` of the gradient line (post-fixup; offsets are ascending).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorStop {
    pub color: Color,
    pub offset: f32,
    /// Transition hint for the segment starting at this stop, as a
    /// fraction of the segment (`None` = linear interpolation). The
    /// hint marks where the 50% blend between this stop and the next
    /// should land.
    pub hint_after: Option<f32>,
}

/// Resolved gradient geometry, all in fractions of the box (so the
/// same [`Gradient`] rasterizes at any device scale).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GradientKind {
    /// Gradient-line start point `(x0, y0)` and span `(dx, dy)` in box
    /// fractions; `t = dot(p - start, span) / dot(span, span)`.
    Linear { x0: f32, y0: f32, dx: f32, dy: f32 },
    /// Center and radii in box fractions. A non-positive radius is
    /// degenerate: the whole box paints the last stop's color (CSS
    /// Images 4 degenerate-radial behavior).
    Radial { cx: f32, cy: f32, rx: f32, ry: f32 },
    /// Center in box fractions; `from` is the CSS start angle in
    /// radians (0 = up, clockwise positive).
    Conic { cx: f32, cy: f32, from: f32 },
}

/// A gradient bound to a box: what the backends rasterize. Build via
/// [`ParsedGradient::resolve`].
#[derive(Clone, Debug, PartialEq)]
pub struct Gradient {
    pub kind: GradientKind,
    /// Post-fixup stops, ascending offsets. Inline-first storage
    /// ([`GradientStops`]): the common 1-2 stop case allocates nothing.
    pub stops: GradientStops,
    /// Whether the gradient repeats past `[0, 1]`.
    pub repeating: bool,
}

impl Gradient {
    /// Content identity for backend texture caches: FNV-1a over the
    /// kind discriminant, geometry bits, stop colors/offsets/hints,
    /// and the repeating flag.
    ///
    /// The hash feeds the FNV-1a state byte-by-byte instead of staging
    /// the byte stream in a `Vec<u8>` first: the byte order is identical
    /// to the old staging version, so cache keys are unchanged, but a
    /// hash costs zero allocations (the GPU backend re-hashes every
    /// gradient item once per frame for its texture-cache key).
    pub fn content_hash(&self) -> u64 {
        // Incremental FNV-1a byte sink; `byte` order below reproduces
        // the old staged byte stream exactly.
        struct Hasher(u64);
        impl Hasher {
            #[inline]
            fn byte(&mut self, b: u8) {
                self.0 ^= u64::from(b);
                self.0 = self.0.wrapping_mul(0x0100_0000_0000_01b3);
            }
            #[inline]
            fn bytes(&mut self, bs: &[u8]) {
                for &b in bs {
                    self.byte(b);
                }
            }
            #[inline]
            fn f32_canon(&mut self, f: f32) {
                // Canonicalize so bit-level noise never splits cache
                // entries: `-0.0 + 0.0 == 0.0`, and every NaN payload maps
                // to one canonical pattern (resolve never emits NaN, but
                // the hash must not depend on payloads if one sneaks in).
                let f = f + 0.0;
                let bits = if f.is_nan() { 0x7fc0_0000 } else { f.to_bits() };
                self.bytes(&bits.to_le_bytes());
            }
        }
        let mut h = Hasher(0xcbf2_9ce4_8422_2325);
        h.byte(self.repeating as u8);
        match self.kind {
            GradientKind::Linear { x0, y0, dx, dy } => {
                h.byte(0);
                for f in [x0, y0, dx, dy] {
                    h.f32_canon(f);
                }
            }
            GradientKind::Radial { cx, cy, rx, ry } => {
                h.byte(1);
                for f in [cx, cy, rx, ry] {
                    h.f32_canon(f);
                }
            }
            GradientKind::Conic { cx, cy, from } => {
                h.byte(2);
                for f in [cx, cy, from] {
                    h.f32_canon(f);
                }
            }
        }
        for stop in self.stops.iter() {
            h.bytes(&stop.color.to_bits().to_le_bytes());
            h.f32_canon(stop.offset);
            match stop.hint_after {
                Some(hint) => {
                    h.byte(1);
                    h.f32_canon(hint);
                }
                None => h.byte(0),
            }
        }
        h.0
    }
}

/// How many [`ColorStop`]s a [`GradientStops`] holds inline before
/// spilling to the heap. 2 covers the overwhelmingly common two-stop
/// gradient (`linear-gradient(red, blue)`) with zero allocation; the
/// inline case keeps [`DisplayItem`](crate::display_list::DisplayItem)
/// at 88 B (see the size assertion in `display_list`'s tests).
const GRADIENT_STOPS_INLINE: usize = 2;

/// Inline-first storage for a resolved gradient's color stops.
///
/// A fresh `Vec<ColorStop>` per gradient display item per frame is
/// allocator churn on the paint hot path; typical gradients carry 1-2
/// stops, so the first [`GRADIENT_STOPS_INLINE`] live directly in the
/// [`DisplayItem`](crate::display_list::DisplayItem) and only longer
/// lists (a hostile stylesheet can name thousands of stops) spill to a
/// heap [`Vec`]. [`Deref`](std::ops::Deref) exposes the stops as a
/// `[ColorStop]` slice, so all consumers (the sampler, the content
/// hash, the backends) read them exactly like the old `Vec`.
///
/// `PartialEq` compares the live stop slice (see
/// [`as_slice`](Self::as_slice)), not the storage representation: an
/// `Inline` and a `Spill` holding the same stops compare equal, so a
/// future non-canonical constructor cannot silently break equality.
/// Content identity for backend caches is
/// [`Gradient::content_hash`], which hashes the deref'd slice and is
/// representation-blind.
#[derive(Clone, Debug)]
pub enum GradientStops {
    /// `len` live stops at the front of `stops`; the remainder are
    /// zero stops, never observed (every accessor goes through the
    /// [`Deref`](std::ops::Deref) slice, which covers only the live
    /// prefix).
    Inline {
        len: u8,
        stops: [ColorStop; GRADIENT_STOPS_INLINE],
    },
    /// More than [`GRADIENT_STOPS_INLINE`] stops.
    Spill(Vec<ColorStop>),
}

impl GradientStops {
    /// Empty stop list.
    pub fn new() -> Self {
        Self::from_inline(0, &[])
    }

    /// The stops as a slice.
    pub fn as_slice(&self) -> &[ColorStop] {
        self
    }

    /// Build the `Inline` case from a prefix: `stops` must hold at
    /// most [`GRADIENT_STOPS_INLINE`] entries.
    fn from_inline(len: u8, stops: &[ColorStop]) -> Self {
        debug_assert!((len as usize) <= GRADIENT_STOPS_INLINE);
        debug_assert_eq!(len as usize, stops.len());
        // `ColorStop: Copy`, so the repeat expression is legal; the
        // filler is never observed (see the `Inline` docs).
        let mut inline = [ColorStop {
            color: Color::rgba8(0, 0, 0, 0),
            offset: 0.0,
            hint_after: None,
        }; GRADIENT_STOPS_INLINE];
        inline[..stops.len()].copy_from_slice(stops);
        GradientStops::Inline { len, stops: inline }
    }

    /// Append one stop, spilling to the heap past
    /// [`GRADIENT_STOPS_INLINE`].
    pub fn push(&mut self, stop: ColorStop) {
        match self {
            GradientStops::Inline { len, stops } if (*len as usize) < GRADIENT_STOPS_INLINE => {
                stops[*len as usize] = stop;
                *len += 1;
            }
            this => {
                // Spill once: move the inline prefix into a `Vec`,
                // then push there for all subsequent stops.
                let mut spilled = Vec::with_capacity(this.len() + 1);
                spilled.extend_from_slice(this);
                spilled.push(stop);
                *this = GradientStops::Spill(spilled);
            }
        }
    }
}

impl From<Vec<ColorStop>> for GradientStops {
    /// Canonical construction: `<= INLINE` stops go inline (the source
    /// `Vec`'s buffer is dropped without reuse — the win is that the
    /// *display item* no longer owns a heap allocation).
    fn from(stops: Vec<ColorStop>) -> Self {
        if stops.len() <= GRADIENT_STOPS_INLINE {
            Self::from_inline(stops.len() as u8, &stops)
        } else {
            GradientStops::Spill(stops)
        }
    }
}

impl Default for GradientStops {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for GradientStops {
    type Target = [ColorStop];

    fn deref(&self) -> &[ColorStop] {
        match self {
            GradientStops::Inline { len, stops } => &stops[..*len as usize],
            GradientStops::Spill(stops) => stops,
        }
    }
}

impl PartialEq for GradientStops {
    /// Content equality: compare the live stop slices, ignoring whether
    /// the stops live inline or spilled.
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

// `Color` needs a bit accessor for hashing.
trait ColorBits {
    fn to_bits(self) -> u32;
}

impl ColorBits for Color {
    fn to_bits(self) -> u32 {
        (u32::from(self.r8()) << 24)
            | (u32::from(self.g8()) << 16)
            | (u32::from(self.b8()) << 8)
            | u32::from(self.a8())
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Byte cursor over a gradient function string.
struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            b: s.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r' | 0x0c)) {
            self.pos += 1;
        }
    }

    fn at_end(&mut self) -> bool {
        self.skip_ws();
        self.pos >= self.b.len()
    }

    /// CSS identifier: `-?[_a-zA-Z][_a-zA-Z0-9-]*` (ASCII subset; enough
    /// for gradient keywords, color names, and units).
    fn parse_ident(&mut self) -> Option<&'a str> {
        self.skip_ws();
        self.parse_ident_raw()
    }

    /// CSS identifier without skipping leading whitespace. CSS forbids
    /// whitespace between a number and its unit (`10px`, not `10 px`)
    /// and between a function name and `(` (`rgb(`, not `rgb (`), so
    /// the dimension/angle/color paths use this instead of
    /// [`Self::parse_ident`].
    fn parse_ident_raw(&mut self) -> Option<&'a str> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'a'..=b'z' | b'A'..=b'Z' | b'_') => self.pos += 1,
            _ => {
                self.pos = start;
                return None;
            }
        }
        while matches!(
            self.peek(),
            Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_')
        ) {
            self.pos += 1;
        }
        std::str::from_utf8(&self.b[start..self.pos]).ok()
    }

    /// CSS number: `[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?`.
    fn parse_number(&mut self) -> Option<f32> {
        self.skip_ws();
        let start = self.pos;
        if matches!(self.peek(), Some(b'+' | b'-')) {
            self.pos += 1;
        }
        let mut digits = 0;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
            digits += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
                digits += 1;
            }
        }
        if digits == 0 {
            return None;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            let save = self.pos;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let mut exp_digits = 0;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
                exp_digits += 1;
            }
            if exp_digits == 0 {
                self.pos = save;
            }
        }
        std::str::from_utf8(&self.b[start..self.pos])
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
    }

    /// `<number><unit-ident>` (e.g. `45deg`, `10px`): no whitespace
    /// is allowed between the number and the unit.
    fn parse_dimension(&mut self) -> Option<(f32, &'a str)> {
        let n = self.parse_number()?;
        let unit = self.parse_ident_raw()?;
        Some((n, unit))
    }

    /// `<number>%`.
    fn parse_percentage(&mut self) -> Option<f32> {
        let n = self.parse_number()?;
        self.skip_ws();
        if self.peek() == Some(b'%') {
            self.pos += 1;
            Some(n)
        } else {
            None
        }
    }

    /// `#` followed by hex digits.
    fn parse_hash(&mut self) -> Option<&'a str> {
        self.skip_ws();
        if self.peek() != Some(b'#') {
            return None;
        }
        let start = self.pos + 1;
        self.pos += 1;
        while matches!(self.peek(), Some(b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F')) {
            self.pos += 1;
        }
        if self.pos == start {
            return None;
        }
        std::str::from_utf8(&self.b[start..self.pos]).ok()
    }
}

/// Split a function body's text into top-level comma-separated
/// arguments, respecting nested parens and quoted strings.
fn split_args(inner: &str) -> Result<Vec<&str>, GradientParseError> {
    let mut args = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut quote = None::<u8>;
    let b = inner.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if let Some(q) = quote {
            if c == q {
                quote = None;
            } else if c == b'\\' {
                i += 1;
            }
        } else {
            match c {
                b'\'' | b'"' => quote = Some(c),
                b'(' => depth += 1,
                b')' => {
                    depth = depth.checked_sub(1).ok_or_else(|| {
                        GradientParseError::new("unbalanced parentheses in gradient")
                    })?;
                }
                b',' if depth == 0 => {
                    args.push(inner[start..i].trim());
                    start = i + 1;
                }
                _ => {}
            }
        }
        i += 1;
    }
    if quote.is_some() || depth != 0 {
        return Err(GradientParseError::new(
            "unbalanced parentheses or quotes in gradient",
        ));
    }
    args.push(inner[start..].trim());
    Ok(args)
}

/// Split `name(...)` into the function name and its inner text.
fn split_function(s: &str) -> Result<(&str, &str), GradientParseError> {
    let mut c = Cursor::new(s);
    let name = c
        .parse_ident()
        .ok_or_else(|| GradientParseError::new("expected a gradient function name"))?;
    c.skip_ws();
    if c.peek() != Some(b'(') {
        return Err(GradientParseError::new("expected '(' after gradient name"));
    }
    // Find the matching close paren from the raw text.
    let b = s.as_bytes();
    let mut i = c.pos + 1;
    let mut depth = 1usize;
    let mut quote = None::<u8>;
    while i < b.len() {
        let ch = b[i];
        if let Some(q) = quote {
            if ch == q {
                quote = None;
            } else if ch == b'\\' {
                i += 1;
            }
        } else {
            match ch {
                b'\'' | b'"' => quote = Some(ch),
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        let inner = &s[c.pos + 1..i];
                        let rest = s[i + 1..].trim();
                        if !rest.is_empty() {
                            return Err(GradientParseError::new(
                                "unexpected text after gradient function",
                            ));
                        }
                        return Ok((name, inner));
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    Err(GradientParseError::new(
        "unbalanced parentheses in gradient function",
    ))
}

// ---------------------------------------------------------------------------
// Color parsing
// ---------------------------------------------------------------------------

/// CSS named colors, sorted for binary search: `(name, 0xRRGGBB)`.
const NAMED_COLORS: &[(&str, u32)] = &[
    ("aliceblue", 0xF0F8FF),
    ("antiquewhite", 0xFAEBD7),
    ("aqua", 0x00FFFF),
    ("aquamarine", 0x7FFFD4),
    ("azure", 0xF0FFFF),
    ("beige", 0xF5F5DC),
    ("bisque", 0xFFE4C4),
    ("black", 0x000000),
    ("blanchedalmond", 0xFFEBCD),
    ("blue", 0x0000FF),
    ("blueviolet", 0x8A2BE2),
    ("brown", 0xA52A2A),
    ("burlywood", 0xDEB887),
    ("cadetblue", 0x5F9EA0),
    ("chartreuse", 0x7FFF00),
    ("chocolate", 0xD2691E),
    ("coral", 0xFF7F50),
    ("cornflowerblue", 0x6495ED),
    ("cornsilk", 0xFFF8DC),
    ("crimson", 0xDC143C),
    ("cyan", 0x00FFFF),
    ("darkblue", 0x00008B),
    ("darkcyan", 0x008B8B),
    ("darkgoldenrod", 0xB8860B),
    ("darkgray", 0xA9A9A9),
    ("darkgreen", 0x006400),
    ("darkgrey", 0xA9A9A9),
    ("darkkhaki", 0xBDB76B),
    ("darkmagenta", 0x8B008B),
    ("darkolivegreen", 0x556B2F),
    ("darkorange", 0xFF8C00),
    ("darkorchid", 0x9932CC),
    ("darkred", 0x8B0000),
    ("darksalmon", 0xE9967A),
    ("darkseagreen", 0x8FBC8F),
    ("darkslateblue", 0x483D8B),
    ("darkslategray", 0x2F4F4F),
    ("darkslategrey", 0x2F4F4F),
    ("darkturquoise", 0x00CED1),
    ("darkviolet", 0x9400D3),
    ("deeppink", 0xFF1493),
    ("deepskyblue", 0x00BFFF),
    ("dimgray", 0x696969),
    ("dimgrey", 0x696969),
    ("dodgerblue", 0x1E90FF),
    ("firebrick", 0xB22222),
    ("floralwhite", 0xFFFAF0),
    ("forestgreen", 0x228B22),
    ("fuchsia", 0xFF00FF),
    ("gainsboro", 0xDCDCDC),
    ("ghostwhite", 0xF8F8FF),
    ("gold", 0xFFD700),
    ("goldenrod", 0xDAA520),
    ("gray", 0x808080),
    ("green", 0x008000),
    ("greenyellow", 0xADFF2F),
    ("grey", 0x808080),
    ("honeydew", 0xF0FFF0),
    ("hotpink", 0xFF69B4),
    ("indianred", 0xCD5C5C),
    ("indigo", 0x4B0082),
    ("ivory", 0xFFFFF0),
    ("khaki", 0xF0E68C),
    ("lavender", 0xE6E6FA),
    ("lavenderblush", 0xFFF0F5),
    ("lawngreen", 0x7CFC00),
    ("lemonchiffon", 0xFFFACD),
    ("lightblue", 0xADD8E6),
    ("lightcoral", 0xF08080),
    ("lightcyan", 0xE0FFFF),
    ("lightgoldenrodyellow", 0xFAFAD2),
    ("lightgray", 0xD3D3D3),
    ("lightgreen", 0x90EE90),
    ("lightgrey", 0xD3D3D3),
    ("lightpink", 0xFFB6C1),
    ("lightsalmon", 0xFFA07A),
    ("lightseagreen", 0x20B2AA),
    ("lightskyblue", 0x87CEFA),
    ("lightslategray", 0x778899),
    ("lightslategrey", 0x778899),
    ("lightsteelblue", 0xB0C4DE),
    ("lightyellow", 0xFFFFE0),
    ("lime", 0x00FF00),
    ("limegreen", 0x32CD32),
    ("linen", 0xFAF0E6),
    ("magenta", 0xFF00FF),
    ("maroon", 0x800000),
    ("mediumaquamarine", 0x66CDAA),
    ("mediumblue", 0x0000CD),
    ("mediumorchid", 0xBA55D3),
    ("mediumpurple", 0x9370DB),
    ("mediumseagreen", 0x3CB371),
    ("mediumslateblue", 0x7B68EE),
    ("mediumspringgreen", 0x00FA9A),
    ("mediumturquoise", 0x48D1CC),
    ("mediumvioletred", 0xC71585),
    ("midnightblue", 0x191970),
    ("mintcream", 0xF5FFFA),
    ("mistyrose", 0xFFE4E1),
    ("moccasin", 0xFFE4B5),
    ("navajowhite", 0xFFDEAD),
    ("navy", 0x000080),
    ("oldlace", 0xFDF5E6),
    ("olive", 0x808000),
    ("olivedrab", 0x6B8E23),
    ("orange", 0xFFA500),
    ("orangered", 0xFF4500),
    ("orchid", 0xDA70D6),
    ("palegoldenrod", 0xEEE8AA),
    ("palegreen", 0x98FB98),
    ("paleturquoise", 0xAFEEEE),
    ("palevioletred", 0xDB7093),
    ("papayawhip", 0xFFEFD5),
    ("peachpuff", 0xFFDAB9),
    ("peru", 0xCD853F),
    ("pink", 0xFFC0CB),
    ("plum", 0xDDA0DD),
    ("powderblue", 0xB0E0E6),
    ("purple", 0x800080),
    ("rebeccapurple", 0x663399),
    ("red", 0xFF0000),
    ("rosybrown", 0xBC8F8F),
    ("royalblue", 0x4169E1),
    ("saddlebrown", 0x8B4513),
    ("salmon", 0xFA8072),
    ("sandybrown", 0xF4A460),
    ("seagreen", 0x2E8B57),
    ("seashell", 0xFFF5EE),
    ("sienna", 0xA0522D),
    ("silver", 0xC0C0C0),
    ("skyblue", 0x87CEEB),
    ("slateblue", 0x6A5ACD),
    ("slategray", 0x708090),
    ("slategrey", 0x708090),
    ("snow", 0xFFFAFA),
    ("springgreen", 0x00FF7F),
    ("steelblue", 0x4682B4),
    ("tan", 0xD2B48C),
    ("teal", 0x008080),
    ("thistle", 0xD8BFD8),
    ("tomato", 0xFF6347),
    ("turquoise", 0x40E0D0),
    ("violet", 0xEE82EE),
    ("wheat", 0xF5DEB3),
    ("white", 0xFFFFFF),
    ("whitesmoke", 0xF5F5F5),
    ("yellow", 0xFFFF00),
    ("yellowgreen", 0x9ACD32),
];

fn named_color(name: &str) -> Option<Color> {
    let mut lower = [0u8; 32];
    let bytes = name.as_bytes();
    if bytes.len() > lower.len() {
        return None;
    }
    for (i, b) in bytes.iter().enumerate() {
        lower[i] = b.to_ascii_lowercase();
    }
    let key = std::str::from_utf8(&lower[..bytes.len()]).ok()?;
    NAMED_COLORS
        .binary_search_by(|(n, _)| n.cmp(&key))
        .ok()
        .map(|i| {
            let rgb = NAMED_COLORS[i].1;
            Color::rgba8((rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8, 255)
        })
}

fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Parse `#rgb`, `#rgba`, `#rrggbb`, or `#rrggbbaa`.
fn parse_hex_color(hex: &str) -> Option<Color> {
    let b = hex.as_bytes();
    let channel = |i: usize| -> Option<u8> {
        let hi = hex_digit(*b.get(i)?)?;
        let lo = hex_digit(*b.get(i + 1)?)?;
        Some(hi * 16 + lo)
    };
    let dup = |i: usize| -> Option<u8> {
        let d = hex_digit(*b.get(i)?)?;
        Some(d * 16 + d)
    };
    let (r, g, bl, a) = match b.len() {
        3 => (dup(0)?, dup(1)?, dup(2)?, 255),
        4 => (dup(0)?, dup(1)?, dup(2)?, dup(3)?),
        6 => (channel(0)?, channel(2)?, channel(4)?, 255),
        8 => (channel(0)?, channel(2)?, channel(4)?, channel(6)?),
        _ => return None,
    };
    Some(Color::rgba8(r, g, bl, a))
}

/// Split a functional color's argument text into components, handling
/// both comma syntax (`rgb(255, 0, 0)`) and space/slash syntax
/// (`rgb(255 0 0 / 50%)`).
fn split_color_components(
    inner: &str,
) -> Result<(Vec<String>, Option<String>), GradientParseError> {
    // Normalize: turn "/" into its own token, then split whitespace;
    // commas split too (legacy syntax never mixes the two).
    let mut parts: Vec<String> = Vec::new();
    if inner.contains(',') {
        if inner.contains('/') {
            return Err(GradientParseError::new("color function mixes ',' and '/'"));
        }
        for p in inner.split(',') {
            let t = p.trim();
            if t.is_empty() {
                return Err(GradientParseError::new("empty color component"));
            }
            parts.push(t.to_string());
        }
    } else {
        for p in inner.split_whitespace() {
            if p == "/" {
                parts.push("/".to_string());
            } else if let Some((head, tail)) = p.split_once('/') {
                // `0/50%` without spaces: rare, but accept.
                if !head.is_empty() {
                    parts.push(head.to_string());
                }
                parts.push("/".to_string());
                if !tail.is_empty() {
                    parts.push(tail.to_string());
                }
            } else {
                parts.push(p.to_string());
            }
        }
    }
    // Pull off an optional `/ alpha`.
    let mut alpha = None;
    if let Some(pos) = parts.iter().position(|p| p == "/") {
        if pos + 2 != parts.len() {
            return Err(GradientParseError::new(
                "expected a single alpha value after '/'",
            ));
        }
        alpha = Some(parts.remove(pos + 1));
        parts.remove(pos);
    }
    // Legacy comma syntax: 3 components + optional alpha (4th).
    if alpha.is_none() && parts.len() == 4 && inner.contains(',') {
        alpha = Some(parts.remove(3));
    }
    Ok((parts, alpha))
}

/// Parse one `rgb()` channel: `<number>` (0-255) or `<percentage>`.
fn parse_rgb_channel(s: &str) -> Option<u8> {
    let s = s.trim();
    if let Some(num) = s.strip_suffix('%') {
        let v: f32 = num.trim().parse().ok()?;
        Some(((v / 100.0 * 255.0).round().clamp(0.0, 255.0)) as u8)
    } else {
        let v: f32 = s.parse().ok()?;
        Some(v.round().clamp(0.0, 255.0) as u8)
    }
}

/// Parse an alpha value: `<number>` (0-1) or `<percentage>`.
fn parse_alpha_value(s: &str) -> Option<u8> {
    let s = s.trim();
    let a = if let Some(num) = s.strip_suffix('%') {
        num.trim().parse::<f32>().ok()? / 100.0
    } else {
        s.parse::<f32>().ok()?
    };
    Some((a.clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn parse_rgb_function(inner: &str) -> Result<Color, GradientParseError> {
    let bad = || GradientParseError::new("invalid rgb()/rgba() color");
    let (parts, alpha) = split_color_components(inner)?;
    if parts.len() != 3 {
        return Err(bad());
    }
    let r = parse_rgb_channel(&parts[0]).ok_or_else(bad)?;
    let g = parse_rgb_channel(&parts[1]).ok_or_else(bad)?;
    let b = parse_rgb_channel(&parts[2]).ok_or_else(bad)?;
    let a = match alpha {
        Some(a) => parse_alpha_value(&a).ok_or_else(bad)?,
        None => 255,
    };
    Ok(Color::rgba8(r, g, b, a))
}

/// Parse an `<angle>` or `<number>` hue: plain numbers are degrees.
fn parse_hue(s: &str) -> Option<f32> {
    let s = s.trim();
    let mut c = Cursor::new(s);
    let n = c.parse_number()?;
    if c.at_end() {
        return Some(n);
    }
    let unit = c.parse_ident()?;
    if !c.at_end() {
        return None;
    }
    match unit.to_ascii_lowercase().as_str() {
        "deg" => Some(n),
        "grad" => Some(n * 360.0 / 400.0),
        "rad" => Some(n * 180.0 / std::f32::consts::PI),
        "turn" => Some(n * 360.0),
        _ => None,
    }
}

fn parse_hsl_function(inner: &str) -> Result<Color, GradientParseError> {
    let bad = || GradientParseError::new("invalid hsl()/hsla() color");
    let (parts, alpha) = split_color_components(inner)?;
    if parts.len() != 3 {
        return Err(bad());
    }
    let h = parse_hue(&parts[0]).ok_or_else(bad)?;
    let pct = |s: &str| -> Option<f32> {
        let num = s.strip_suffix('%')?;
        num.trim().parse::<f32>().ok().map(|v| v / 100.0)
    };
    let s = pct(&parts[1]).ok_or_else(bad)?.clamp(0.0, 1.0);
    let l = pct(&parts[2]).ok_or_else(bad)?.clamp(0.0, 1.0);
    let a = match alpha {
        Some(a) => parse_alpha_value(&a).ok_or_else(bad)?,
        None => 255,
    };
    let (r, g, b) = hsl_to_rgb(h, s, l);
    Ok(Color::rgba8(r, g, b, a))
}

/// Standard HSL (h in degrees) to 8-bit sRGB.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let h = ((h % 360.0) + 360.0) % 360.0 / 360.0;
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let channel = |t: f32| -> u8 {
        let mut t = t;
        if t < 0.0 {
            t += 1.0;
        }
        if t > 1.0 {
            t -= 1.0;
        }
        let v = if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        };
        (v.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    (channel(h + 1.0 / 3.0), channel(h), channel(h - 1.0 / 3.0))
}

/// Parse one `<color>`: named color, `#hex`, `rgb()`/`rgba()`,
/// `hsl()`/`hsla()`, `transparent`, or `currentcolor`.
fn parse_stop_color(c: &mut Cursor<'_>) -> Result<StopColor, GradientParseError> {
    let bad = || GradientParseError::new("expected a color for the color stop");
    if c.peek() == Some(b'#') {
        let hex = c.parse_hash().ok_or_else(bad)?;
        return parse_hex_color(hex).map(StopColor::Rgba).ok_or_else(bad);
    }
    let ident = c.parse_ident().ok_or_else(bad)?;
    let lower = ident.to_ascii_lowercase();
    // Functional color: the ident is immediately followed by `(`
    // (CSS forbids whitespace between a function name and `(`).
    if c.peek() == Some(b'(') {
        c.pos += 1;
        // Find the matching close paren in the remaining text.
        let rest = &c.b[c.pos..];
        let mut depth = 1usize;
        let mut i = 0;
        while i < rest.len() {
            match rest[i] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        if depth != 0 || i >= rest.len() {
            return Err(bad());
        }
        let inner = std::str::from_utf8(&rest[..i]).map_err(|_| bad())?;
        c.pos += i + 1;
        let color = match lower.as_str() {
            "rgb" | "rgba" => parse_rgb_function(inner)?,
            "hsl" | "hsla" => parse_hsl_function(inner)?,
            _ => return Err(bad()),
        };
        return Ok(StopColor::Rgba(color));
    }
    if lower == "currentcolor" {
        return Ok(StopColor::CurrentColor);
    }
    if lower == "transparent" {
        return Ok(StopColor::Rgba(Color::rgba8(0, 0, 0, 0)));
    }
    named_color(ident).map(StopColor::Rgba).ok_or_else(bad)
}

/// Parse one stop/hint position: `<percentage>`, `<length>` in px, or
/// unitless zero. Other units are rejected (see the module docs).
/// When `conic`, lengths are rejected and CSS angles are accepted
/// instead (conic stops are `<angle> | <percentage>`), converted to
/// the equivalent percentage of a full turn.
fn parse_stop_pos(c: &mut Cursor<'_>, conic: bool) -> Option<StopPos> {
    let save = c.pos;
    if let Some(p) = c.parse_percentage() {
        return Some(StopPos::Pct(p));
    }
    c.pos = save;
    if conic {
        if let Some(rad) = parse_angle_at(c) {
            // Angles as a percentage of the 360-degree gradient line.
            return Some(StopPos::Pct(rad * 180.0 / std::f32::consts::PI / 3.6));
        }
        c.pos = save;
        return None;
    }
    if let Some((n, unit)) = c.parse_dimension() {
        if unit.eq_ignore_ascii_case("px") {
            return Some(StopPos::Px(n));
        }
        c.pos = save;
        return None;
    }
    c.pos = save;
    if let Some(n) = c.parse_number()
        && n == 0.0
    {
        return Some(StopPos::Px(0.0));
    }
    c.pos = save;
    None
}

/// Parse a CSS `<angle>` at the cursor, returning radians. Accepts
/// `deg`/`grad`/`rad`/`turn` and unitless zero. Restores the cursor
/// and returns `None` when the text is not an angle. No whitespace is
/// allowed between the number and the unit (`90deg`, not `90 deg`).
fn parse_angle_at(c: &mut Cursor<'_>) -> Option<f32> {
    let save = c.pos;
    let n = c.parse_number()?;
    // Unitless zero: the number must end the component.
    if c.peek().is_none() {
        return if n == 0.0 { Some(0.0) } else { None };
    }
    // Whitespace between the number and the unit is a parse error.
    if matches!(c.peek(), Some(b' ' | b'\t' | b'\n' | b'\r' | 0x0c)) {
        c.pos = save;
        return None;
    }
    let unit = c.parse_ident_raw()?;
    let deg = match unit.to_ascii_lowercase().as_str() {
        "deg" => n,
        "grad" => n * 360.0 / 400.0,
        "rad" => n * 180.0 / std::f32::consts::PI,
        "turn" => n * 360.0,
        _ => {
            c.pos = save;
            return None;
        }
    };
    Some(deg * std::f32::consts::PI / 180.0)
}

/// Parse a CSS `<angle>` (radians out): `deg`/`grad`/`rad`/`turn`, or
/// unitless zero. No whitespace between the number and the unit.
fn parse_angle_text(s: &str) -> Option<f32> {
    let mut c = Cursor::new(s);
    let n = c.parse_number()?;
    // No whitespace between the number and the unit (`at_end` skips
    // whitespace, so check the raw next byte first).
    if matches!(c.peek(), Some(b' ' | b'\t' | b'\n' | b'\r' | 0x0c)) {
        return None;
    }
    if c.at_end() {
        return if n == 0.0 { Some(0.0) } else { None };
    }
    let unit = c.parse_ident_raw()?;
    if !c.at_end() {
        return None;
    }
    let deg = match unit.to_ascii_lowercase().as_str() {
        "deg" => n,
        "grad" => n * 360.0 / 400.0,
        "rad" => n * 180.0 / std::f32::consts::PI,
        "turn" => n * 360.0,
        _ => return None,
    };
    Some(deg * std::f32::consts::PI / 180.0)
}

/// Parse `to <side-or-corner>` into a [`LinearDir`]. Sides become
/// explicit angles; corners stay corner markers (the angle is
/// box-dependent, resolved in [`LinearDir::angle_for`]).
fn parse_to_keyword(words: &[&str]) -> Option<LinearDir> {
    if words.is_empty() || !words[0].eq_ignore_ascii_case("to") {
        return None;
    }
    let rad = |deg: f32| deg * std::f32::consts::PI / 180.0;
    let rest = &words[1..];
    let dir = match rest {
        [a] if a.eq_ignore_ascii_case("top") => LinearDir::Angle(rad(0.0)),
        [a] if a.eq_ignore_ascii_case("right") => LinearDir::Angle(rad(90.0)),
        [a] if a.eq_ignore_ascii_case("bottom") => LinearDir::Angle(rad(180.0)),
        [a] if a.eq_ignore_ascii_case("left") => LinearDir::Angle(rad(270.0)),
        [a, b] => {
            let vertical = if a.eq_ignore_ascii_case("top") || b.eq_ignore_ascii_case("top") {
                "top"
            } else if a.eq_ignore_ascii_case("bottom") || b.eq_ignore_ascii_case("bottom") {
                "bottom"
            } else {
                return None;
            };
            let horizontal = if a.eq_ignore_ascii_case("left") || b.eq_ignore_ascii_case("left") {
                "left"
            } else if a.eq_ignore_ascii_case("right") || b.eq_ignore_ascii_case("right") {
                "right"
            } else {
                return None;
            };
            // The two keywords must be one vertical and one horizontal.
            let kinds = [a, b].map(|w| {
                if w.eq_ignore_ascii_case("top") || w.eq_ignore_ascii_case("bottom") {
                    0
                } else {
                    1
                }
            });
            if kinds[0] == kinds[1] {
                return None;
            }
            LinearDir::Corner {
                to_top: vertical == "top",
                to_right: horizontal == "right",
            }
        }
        _ => return None,
    };
    Some(dir)
}

/// Parse one `at <position>` component: keyword or length-percentage.
fn parse_pos_comp(word: &str) -> Option<PosComp> {
    let mut c = Cursor::new(word);
    if let Some(p) = c.parse_percentage()
        && c.at_end()
    {
        return Some(PosComp::Pct(p));
    }
    let mut c = Cursor::new(word);
    if let Some((n, unit)) = c.parse_dimension() {
        if c.at_end() && unit.eq_ignore_ascii_case("px") {
            return Some(PosComp::Px(n));
        }
        return None;
    }
    let mut c = Cursor::new(word);
    if let Some(n) = c.parse_number()
        && c.at_end()
        && n == 0.0
    {
        return Some(PosComp::Px(0.0));
    }
    None
}

/// Parse an `at <position>` value (the words after `at`): 1-4
/// components per CSS Values 4 `<position>`.
fn parse_position(words: &[&str]) -> Option<GradientPosition> {
    // Keyword to (axis, fraction): axis 0 = horizontal, 1 = vertical.
    fn kw(word: &str) -> Option<(u8, f32)> {
        match word.to_ascii_lowercase().as_str() {
            "left" => Some((0, 0.0)),
            "right" => Some((0, 100.0)),
            "top" => Some((1, 0.0)),
            "bottom" => Some((1, 100.0)),
            "center" => None,
            _ => None,
        }
    }
    let center = GradientPosition {
        x: PosComp::Pct(50.0),
        y: PosComp::Pct(50.0),
    };
    match words {
        [] => None,
        [a] => {
            if let Some((axis, v)) = kw(a) {
                let mut p = center;
                if axis == 0 {
                    p.x = PosComp::Pct(v);
                } else {
                    p.y = PosComp::Pct(v);
                }
                Some(p)
            } else if a.eq_ignore_ascii_case("center") {
                Some(center)
            } else {
                parse_pos_comp(a).map(|x| GradientPosition {
                    x,
                    y: PosComp::Pct(50.0),
                })
            }
        }
        [a, b] => {
            // Each word is horizontal (`left`/`right`/length),
            // vertical (`top`/`bottom`/length), or `center` (either
            // axis). Two non-center words must be on different axes;
            // a length takes the axis the other word does not use.
            fn classify(word: &str) -> Option<(u8, PosComp)> {
                // Axis: 0 = horizontal, 1 = vertical, 2 = center
                // (either), 3 = length (either).
                match word.to_ascii_lowercase().as_str() {
                    "left" => Some((0, PosComp::Pct(0.0))),
                    "right" => Some((0, PosComp::Pct(100.0))),
                    "top" => Some((1, PosComp::Pct(0.0))),
                    "bottom" => Some((1, PosComp::Pct(100.0))),
                    "center" => Some((2, PosComp::Pct(50.0))),
                    _ => parse_pos_comp(word).map(|v| (3, v)),
                }
            }
            let (aa, av) = classify(a)?;
            let (ba, bv) = classify(b)?;
            let (x, y) = match (aa, ba) {
                (0, 1) | (0, 3) | (3, 1) | (3, 3) | (0, 2) | (3, 2) | (2, 2) => (av, bv),
                (1, 0) | (1, 3) | (3, 0) | (2, 0) | (2, 1) | (1, 2) => (bv, av),
                (2, 3) => (av, bv),
                _ => return None,
            };
            Some(GradientPosition { x, y })
        }
        _ if words.len() <= 4 => {
            // 3-4 component syntax: `[left|right|center] <offset>?` and
            // `[top|bottom|center] <offset>?`, each axis at most once,
            // offsets default to zero (`right 10px bottom 20px`).
            let mut hx: Option<PosComp> = None;
            let mut vx: Option<PosComp> = None;
            let mut i = 0;
            while i < words.len() {
                let w = words[i];
                let lower = w.to_ascii_lowercase();
                let (axis, base): (u8, PosComp) = match lower.as_str() {
                    "left" => (0, PosComp::Pct(0.0)),
                    "right" => (0, PosComp::Pct(100.0)),
                    "top" => (1, PosComp::Pct(0.0)),
                    "bottom" => (1, PosComp::Pct(100.0)),
                    "center" => {
                        // `center` alone fills the first free axis.
                        if hx.is_none() {
                            hx = Some(PosComp::Pct(50.0));
                        } else if vx.is_none() {
                            vx = Some(PosComp::Pct(50.0));
                        } else {
                            return None;
                        }
                        i += 1;
                        continue;
                    }
                    _ => return None,
                };
                let slot = if axis == 0 { &mut hx } else { &mut vx };
                if slot.is_some() {
                    return None;
                }
                // Optional offset after the keyword: measured inward
                // from the named edge (`right 10px`).
                if i + 1 < words.len()
                    && let Some(off) = parse_pos_comp(words[i + 1])
                {
                    let v = match (base, off) {
                        (PosComp::Pct(0.0), PosComp::Pct(o)) => PosComp::Pct(o),
                        (PosComp::Pct(100.0), PosComp::Pct(o)) => PosComp::FarPct(o),
                        (PosComp::Pct(0.0), PosComp::Px(o)) => PosComp::Px(o),
                        (PosComp::Pct(100.0), PosComp::Px(o)) => PosComp::FarPx(o),
                        _ => return None,
                    };
                    *slot = Some(v);
                    i += 2;
                    continue;
                }
                *slot = Some(base);
                i += 1;
            }
            Some(GradientPosition {
                x: hx.unwrap_or(PosComp::Pct(50.0)),
                y: vx.unwrap_or(PosComp::Pct(50.0)),
            })
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Top-level parse
// ---------------------------------------------------------------------------

/// Parse one color-stop list argument: `<color> <position>{0,2}`.
/// When `conic`, positions are `<angle> | <percentage>` (lengths are
/// rejected); otherwise they are `<length> | <percentage>`.
fn parse_stop_arg(arg: &str, conic: bool) -> Result<ParsedStopItem, GradientParseError> {
    let bad = || GradientParseError::new(format!("invalid color stop '{arg}'"));
    let mut c = Cursor::new(arg);
    // A bare position with no color is a transition hint.
    let save = c.pos;
    if parse_stop_pos(&mut c, conic).is_some() && c.at_end() {
        c.pos = save;
        let pos = parse_stop_pos(&mut c, conic).ok_or_else(bad)?;
        return Ok(ParsedStopItem::Hint(pos));
    }
    c.pos = save;
    let color = parse_stop_color(&mut c).map_err(|_| bad())?;
    let mut positions = [None, None];
    for slot in positions.iter_mut() {
        let save = c.pos;
        if let Some(p) = parse_stop_pos(&mut c, conic) {
            *slot = Some(p);
        } else {
            c.pos = save;
            break;
        }
    }
    if !c.at_end() {
        return Err(bad());
    }
    Ok(ParsedStopItem::Stop(ParsedStop { color, positions }))
}

/// Parse the `linear-gradient()` prelude: `[<angle> | to <side-or-corner>]?`.
fn parse_linear_prelude(arg: &str) -> Option<LinearDir> {
    if let Some(angle) = parse_angle_text(arg.trim()) {
        return Some(LinearDir::Angle(angle));
    }
    let words: Vec<&str> = arg.split_whitespace().collect();
    parse_to_keyword(&words)
}

/// Parse the `radial-gradient()` prelude:
/// `[<radial-shape> || <radial-size>]? [at <position>]?`.
fn parse_radial_prelude(arg: &str) -> Result<ParsedRadial, GradientParseError> {
    let bad = || GradientParseError::new(format!("invalid radial-gradient prelude '{arg}'"));
    let words: Vec<&str> = arg.split_whitespace().collect();
    let mut circle: Option<bool> = None;
    let mut size: Option<RadialSize> = None;
    let mut center: Option<GradientPosition> = None;
    let mut i = 0;
    while i < words.len() {
        let w = words[i];
        let lower = w.to_ascii_lowercase();
        match lower.as_str() {
            "circle" | "ellipse" => {
                if circle.is_some() {
                    return Err(bad());
                }
                circle = Some(lower == "circle");
                i += 1;
            }
            "closest-side" | "farthest-side" | "closest-corner" | "farthest-corner" => {
                if size.is_some() {
                    return Err(bad());
                }
                size = Some(RadialSize::Extent(match lower.as_str() {
                    "closest-side" => RadialExtent::ClosestSide,
                    "farthest-side" => RadialExtent::FarthestSide,
                    "closest-corner" => RadialExtent::ClosestCorner,
                    _ => RadialExtent::FarthestCorner,
                }));
                i += 1;
            }
            "at" => {
                if center.is_some() {
                    return Err(bad());
                }
                let pos = parse_position(&words[i + 1..]).ok_or_else(bad)?;
                center = Some(pos);
                break;
            }
            _ => {
                // An explicit size: one or two length-percentages.
                if size.is_some() {
                    return Err(bad());
                }
                let mut comps = Vec::new();
                while i < words.len() && !words[i].eq_ignore_ascii_case("at") && comps.len() < 2 {
                    comps.push(parse_pos_comp(words[i]).ok_or_else(bad)?);
                    i += 1;
                }
                if comps.is_empty() {
                    return Err(bad());
                }
                // Explicit sizes must be non-negative (CSS Images 4:
                // a negative size is an invalid declaration).
                for comp in &comps {
                    let negative = match comp {
                        PosComp::Px(v) => *v < 0.0,
                        PosComp::Pct(p) => *p < 0.0,
                        // `parse_pos_comp` never produces these, but
                        // keep the check total.
                        PosComp::FarPct(p) => *p < 0.0,
                        PosComp::FarPx(v) => *v < 0.0,
                    };
                    if negative {
                        return Err(bad());
                    }
                }
                size = Some(if comps.len() == 1 {
                    match comps[0] {
                        PosComp::Px(v) => RadialSize::CirclePx(v),
                        // A single percentage is not a valid circle
                        // radius (percentages only size ellipses).
                        _ => return Err(bad()),
                    }
                } else {
                    RadialSize::Ellipse(comps[0], comps[1])
                });
            }
        }
    }
    // Shape/size consistency: a single length is a circle radius (an
    // explicit `ellipse` keyword with it is invalid); two radii are an
    // ellipse (an explicit `circle` keyword with them is invalid).
    let circle = match (&size, circle) {
        (Some(RadialSize::CirclePx(_)), None) => true,
        (Some(RadialSize::CirclePx(_)), Some(true)) => true,
        (Some(RadialSize::Ellipse(..)), None) => false,
        (Some(RadialSize::Ellipse(..)), Some(false)) => false,
        (Some(RadialSize::Extent(_)), c) => c.unwrap_or(false),
        (None, c) => c.unwrap_or(false),
        _ => return Err(bad()),
    };
    let size = size.unwrap_or(RadialSize::Extent(RadialExtent::FarthestCorner));
    Ok(ParsedRadial {
        circle,
        center: center.unwrap_or(GradientPosition {
            x: PosComp::Pct(50.0),
            y: PosComp::Pct(50.0),
        }),
        size,
    })
}

/// Parse the `conic-gradient()` prelude:
/// `[from <angle>]? [at <position>]?` in either order.
fn parse_conic_prelude(arg: &str) -> Result<ParsedConic, GradientParseError> {
    let bad = || GradientParseError::new(format!("invalid conic-gradient prelude '{arg}'"));
    let words: Vec<&str> = arg.split_whitespace().collect();
    let mut from: Option<f32> = None;
    let mut center: Option<GradientPosition> = None;
    let mut i = 0;
    while i < words.len() {
        let w = words[i];
        if w.eq_ignore_ascii_case("from") {
            if from.is_some() || i + 1 >= words.len() {
                return Err(bad());
            }
            from = Some(parse_angle_text(words[i + 1]).ok_or_else(bad)?);
            i += 2;
        } else if w.eq_ignore_ascii_case("at") {
            if center.is_some() {
                return Err(bad());
            }
            center = Some(parse_position(&words[i + 1..]).ok_or_else(bad)?);
            break;
        } else {
            return Err(bad());
        }
    }
    Ok(ParsedConic {
        from: from.unwrap_or(0.0),
        center: center.unwrap_or(GradientPosition {
            x: PosComp::Pct(50.0),
            y: PosComp::Pct(50.0),
        }),
    })
}

/// Parse a CSS gradient function:
/// `linear-gradient()` / `radial-gradient()` / `conic-gradient()` and
/// their `repeating-` variants.
pub fn parse_gradient(css: &str) -> Result<ParsedGradient, GradientParseError> {
    let (name, inner) = split_function(css)?;
    let func = match name.to_ascii_lowercase().as_str() {
        "linear-gradient" => GradientFunc::Linear,
        "repeating-linear-gradient" => GradientFunc::RepeatingLinear,
        "radial-gradient" => GradientFunc::Radial,
        "repeating-radial-gradient" => GradientFunc::RepeatingRadial,
        "conic-gradient" => GradientFunc::Conic,
        "repeating-conic-gradient" => GradientFunc::RepeatingConic,
        _ => {
            return Err(GradientParseError::new(format!(
                "not a gradient function: '{name}'"
            )));
        }
    };
    let args = split_args(inner)?;
    if args.iter().any(|a| a.is_empty()) {
        return Err(GradientParseError::new("empty argument in gradient"));
    }
    // Peel off the geometry prelude when the first argument parses as
    // one (a color stop never parses as a prelude: preludes start with
    // an angle, `to`, a shape/size keyword, `at`, or `from`).
    let mut rest: &[&str] = &args;
    let (linear_dir, radial, conic) = match func {
        GradientFunc::Linear | GradientFunc::RepeatingLinear => {
            let dir = parse_linear_prelude(args[0]);
            if dir.is_some() {
                rest = &args[1..];
            }
            // `to bottom` is 180 degrees: the default when no prelude.
            (
                Some(dir.unwrap_or(LinearDir::Angle(std::f32::consts::PI))),
                None,
                None,
            )
        }
        GradientFunc::Radial | GradientFunc::RepeatingRadial => {
            // A radial prelude is present when the first argument is
            // not a color stop: try parsing it as a stop, and treat
            // failure as a prelude.
            let is_stop = matches!(parse_stop_arg(args[0], false), Ok(ParsedStopItem::Stop(_)));
            if is_stop {
                (
                    None,
                    Some(ParsedRadial {
                        circle: false,
                        center: GradientPosition {
                            x: PosComp::Pct(50.0),
                            y: PosComp::Pct(50.0),
                        },
                        size: RadialSize::Extent(RadialExtent::FarthestCorner),
                    }),
                    None,
                )
            } else {
                rest = &args[1..];
                (None, Some(parse_radial_prelude(args[0])?), None)
            }
        }
        GradientFunc::Conic | GradientFunc::RepeatingConic => {
            let is_stop = matches!(parse_stop_arg(args[0], true), Ok(ParsedStopItem::Stop(_)));
            if is_stop {
                (
                    None,
                    None,
                    Some(ParsedConic {
                        from: 0.0,
                        center: GradientPosition {
                            x: PosComp::Pct(50.0),
                            y: PosComp::Pct(50.0),
                        },
                    }),
                )
            } else {
                rest = &args[1..];
                (None, None, Some(parse_conic_prelude(args[0])?))
            }
        }
    };
    if rest.len() < 2 {
        return Err(GradientParseError::new(
            "a gradient needs at least two color stops",
        ));
    }
    let conic_stops = matches!(func, GradientFunc::Conic | GradientFunc::RepeatingConic);
    let mut stops = Vec::with_capacity(rest.len());
    for arg in rest {
        stops.push(parse_stop_arg(arg, conic_stops)?);
    }
    // Hints are only valid between two stops.
    for (i, item) in stops.iter().enumerate() {
        if matches!(item, ParsedStopItem::Hint(_)) {
            let before_ok = i > 0 && matches!(stops[i - 1], ParsedStopItem::Stop(_));
            let after_ok = i + 1 < stops.len() && matches!(stops[i + 1], ParsedStopItem::Stop(_));
            if !(before_ok && after_ok) {
                return Err(GradientParseError::new(
                    "a transition hint must sit between two color stops",
                ));
            }
        }
    }
    let first_is_stop = matches!(stops.first(), Some(ParsedStopItem::Stop(_)));
    let last_is_stop = matches!(stops.last(), Some(ParsedStopItem::Stop(_)));
    if !(first_is_stop && last_is_stop) {
        return Err(GradientParseError::new(
            "a gradient must start and end with a color stop",
        ));
    }
    Ok(ParsedGradient {
        func,
        linear_dir,
        radial,
        conic,
        stops,
    })
}

// ---------------------------------------------------------------------------
// Resolution: ParsedGradient -> Gradient
// ---------------------------------------------------------------------------

impl ParsedGradient {
    /// Bind the parsed gradient to a concrete box, producing the
    /// backend-neutral [`Gradient`].
    ///
    /// `current_color` resolves `currentcolor` stops. Returns `None`
    /// for degenerate boxes (zero/negative/non-finite size) or a
    /// zero-length linear gradient line.
    pub fn resolve(&self, rect: &Rect, current_color: Color) -> Option<Gradient> {
        if rect.w <= 0.0 || rect.h <= 0.0 || !rect.w.is_finite() || !rect.h.is_finite() {
            return None;
        }
        let kind = match self.func {
            GradientFunc::Linear | GradientFunc::RepeatingLinear => {
                let linear_dir = self
                    .linear_dir
                    .unwrap_or(LinearDir::Angle(std::f32::consts::PI));
                // `to <corner>` resolves from the box: the gradient
                // line runs corner-to-corner (CSS Images 4).
                let angle = linear_dir.angle_for(rect.w, rect.h);
                let dir = (angle.sin(), -angle.cos());
                // Project the box corners onto the gradient direction;
                // t = 0 at the smallest projection, t = 1 at the
                // largest (CSS Images 4 linear-gradient rendering).
                let (hw, hh) = (rect.w / 2.0, rect.h / 2.0);
                let mut min = f32::INFINITY;
                let mut max = f32::NEG_INFINITY;
                for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                    let p = sx * hw * dir.0 + sy * hh * dir.1;
                    min = min.min(p);
                    max = max.max(p);
                }
                let len = max - min;
                if len <= 0.0 || !len.is_finite() {
                    return None;
                }
                // Start point and span in box fractions.
                let x0 = 0.5 - dir.0 * (len / 2.0) / rect.w;
                let y0 = 0.5 - dir.1 * (len / 2.0) / rect.h;
                GradientKind::Linear {
                    x0,
                    y0,
                    dx: dir.0 * len / rect.w,
                    dy: dir.1 * len / rect.h,
                }
            }
            GradientFunc::Radial | GradientFunc::RepeatingRadial => {
                let radial = self.radial.as_ref()?;
                let cx = radial.center.x.fraction(rect.w);
                let cy = radial.center.y.fraction(rect.h);
                let (cx_px, cy_px) = (cx * rect.w, cy * rect.h);
                let (rx, ry) = match radial.size {
                    RadialSize::Extent(extent) => {
                        let dl = cx_px;
                        let dr = rect.w - cx_px;
                        let dt = cy_px;
                        let db = rect.h - cy_px;
                        match extent {
                            RadialExtent::ClosestSide => {
                                if radial.circle {
                                    let r = dl.min(dr).min(dt).min(db);
                                    (r, r)
                                } else {
                                    (dl.min(dr), dt.min(db))
                                }
                            }
                            RadialExtent::FarthestSide => {
                                if radial.circle {
                                    let r = dl.max(dr).max(dt).max(db);
                                    (r, r)
                                } else {
                                    (dl.max(dr), dt.max(db))
                                }
                            }
                            RadialExtent::ClosestCorner | RadialExtent::FarthestCorner => {
                                // Corner offsets from the center. Circles
                                // use the corner distance directly;
                                // ellipses keep the side-based aspect
                                // ratio (closest-side for closest-corner,
                                // farthest-side for farthest-corner) and
                                // scale uniformly so the ellipse passes
                                // through the target corner (CSS Images 3
                                // radial-gradient sizing).
                                let corners = [(-dl, -dt), (dr, -dt), (-dl, db), (dr, db)];
                                let &(ccx, ccy) = if extent == RadialExtent::ClosestCorner {
                                    corners
                                        .iter()
                                        .min_by(|a, b| a.0.hypot(a.1).total_cmp(&b.0.hypot(b.1)))
                                        .unwrap()
                                } else {
                                    corners
                                        .iter()
                                        .max_by(|a, b| a.0.hypot(a.1).total_cmp(&b.0.hypot(b.1)))
                                        .unwrap()
                                };
                                if radial.circle {
                                    let r = ccx.hypot(ccy);
                                    (r, r)
                                } else {
                                    let (bx, by) = if extent == RadialExtent::ClosestCorner {
                                        (dl.min(dr), dt.min(db))
                                    } else {
                                        (dl.max(dr), dt.max(db))
                                    };
                                    if bx <= 0.0 || by <= 0.0 {
                                        (0.0, 0.0)
                                    } else {
                                        let k = (ccx / bx).hypot(ccy / by);
                                        (bx * k, by * k)
                                    }
                                }
                            }
                        }
                    }
                    RadialSize::CirclePx(r) => (r, r),
                    RadialSize::Ellipse(x, y) => (
                        match x {
                            PosComp::Pct(p) => p / 100.0 * rect.w,
                            PosComp::Px(v) => v,
                            PosComp::FarPct(p) => (100.0 - p) / 100.0 * rect.w,
                            PosComp::FarPx(v) => rect.w - v,
                        },
                        match y {
                            PosComp::Pct(p) => p / 100.0 * rect.h,
                            PosComp::Px(v) => v,
                            PosComp::FarPct(p) => (100.0 - p) / 100.0 * rect.h,
                            PosComp::FarPx(v) => rect.h - v,
                        },
                    ),
                };
                GradientKind::Radial {
                    cx,
                    cy,
                    rx: rx / rect.w,
                    ry: ry / rect.h,
                }
            }
            GradientFunc::Conic | GradientFunc::RepeatingConic => {
                let conic = self.conic.as_ref()?;
                GradientKind::Conic {
                    cx: conic.center.x.fraction(rect.w),
                    cy: conic.center.y.fraction(rect.h),
                    from: conic.from,
                }
            }
        };
        // Reference length for stop positions: gradient-line length
        // (linear), ending-shape horizontal radius in px (radial), or
        // 360 degrees (conic; conic stops are angles/percentages only).
        let ref_len = match kind {
            GradientKind::Linear { dx, dy, .. } => (dx * rect.w).hypot(dy * rect.h),
            GradientKind::Radial { rx, .. } => rx * rect.w,
            GradientKind::Conic { .. } => 360.0,
        };
        let to_offset = |pos: StopPos| -> f32 {
            match pos {
                StopPos::Pct(p) => p / 100.0,
                // Conic stops never carry lengths (rejected at parse
                // time); linear/radial lengths are absolute px along
                // the gradient line / horizontal ray.
                StopPos::Px(v) => v / ref_len,
            }
        };
        let stops = resolve_stops(&self.stops, to_offset, current_color)?;
        Some(Gradient {
            kind,
            stops,
            repeating: self.func.is_repeating(),
        })
    }
}

/// Expand double positions, apply the CSS Images 4 color-stop fixup,
/// and produce resolved stops with absolute `[0, 1]` offsets.
fn resolve_stops(
    items: &[ParsedStopItem],
    to_offset: impl Fn(StopPos) -> f32,
    current_color: Color,
) -> Option<GradientStops> {
    // Expanded entries: (color, Option<offset>, hint-follows flag is
    // tracked separately below).
    struct Entry {
        color: Color,
        offset: Option<f32>,
    }
    // Hints as (segment index, Option<offset>): the segment index
    // counts expanded stops before the hint.
    let mut entries: Vec<Entry> = Vec::new();
    let mut hints: Vec<(usize, Option<f32>)> = Vec::new();
    for item in items {
        match item {
            ParsedStopItem::Stop(stop) => {
                let color = match stop.color {
                    StopColor::Rgba(c) => c,
                    StopColor::CurrentColor => current_color,
                };
                // Double-position syntax expands to two stops sharing
                // the color.
                let mut pushed = false;
                for pos in stop.positions.iter().flatten() {
                    entries.push(Entry {
                        color,
                        offset: Some(to_offset(*pos)),
                    });
                    pushed = true;
                }
                if !pushed {
                    entries.push(Entry {
                        color,
                        offset: None,
                    });
                }
            }
            ParsedStopItem::Hint(pos) => {
                hints.push((entries.len().saturating_sub(1), Some(to_offset(*pos))));
            }
        }
    }
    if entries.len() < 2 {
        return None;
    }
    // Fixup step 1+2: first -> 0, last -> 1 when unpositioned.
    if entries.first().unwrap().offset.is_none() {
        entries.first_mut().unwrap().offset = Some(0.0);
    }
    if entries.last().unwrap().offset.is_none() {
        entries.last_mut().unwrap().offset = Some(1.0);
    }
    // Fixup step 3: clamp every positioned stop up to the largest
    // positioned *stop* offset before it. Transition hints clamp
    // against that floor but never raise it (CSS Images 4: hints do
    // not affect color stops), so a hint past a later stop stays
    // where it is relative to the hinted segment. List order is stop,
    // hint, stop, so a hint attached to segment `i` is processed
    // after entries[i].
    let mut max_stop = f32::NEG_INFINITY;
    // Merge hints into the ordering pass: hint (seg, off) sits between
    // entries[seg] and entries[seg + 1].
    let mut hint_idx = 0;
    for (i, entry) in entries.iter_mut().enumerate() {
        if let Some(o) = entry.offset {
            if o < max_stop {
                entry.offset = Some(max_stop);
            } else {
                max_stop = o;
            }
        }
        while hint_idx < hints.len() && hints[hint_idx].0 == i {
            if let Some(o) = hints[hint_idx].1
                && o < max_stop
            {
                hints[hint_idx].1 = Some(max_stop);
            }
            hint_idx += 1;
        }
    }
    // Fixup step 4: evenly space runs of unpositioned stops.
    let mut i = 0;
    while i < entries.len() {
        if entries[i].offset.is_some() {
            i += 1;
            continue;
        }
        let run_start = i;
        while i < entries.len() && entries[i].offset.is_none() {
            i += 1;
        }
        let run_end = i; // first positioned stop after the run
        let before = entries[run_start - 1].offset.unwrap_or(0.0);
        let after = entries[run_end].offset.unwrap_or(1.0);
        let n = run_end - run_start;
        for (k, entry) in entries[run_start..run_end].iter_mut().enumerate() {
            entry.offset = Some(before + (after - before) * (k + 1) as f32 / (n + 1) as f32);
        }
    }
    // Build the resolved stops, attaching hints as segment fractions.
    // Non-finite offsets cannot come from written percentages, only
    // from px positions divided by a zero reference length (degenerate
    // geometry, which renders last-stop-wins regardless); pin them to
    // 0.0 so NaN can never reach the rasterizer or the content hash.
    let offsets: Vec<f32> = entries
        .iter()
        .map(|e| {
            let o = e.offset.unwrap_or(0.0);
            if o.is_finite() { o } else { 0.0 }
        })
        .collect();
    let mut hint_after: Vec<Option<f32>> = vec![None; entries.len()];
    for (seg, hint) in hints {
        if let Some(h) = hint
            && seg + 1 < offsets.len()
        {
            let (o0, o1) = (offsets[seg], offsets[seg + 1]);
            if o1 > o0 {
                let frac = ((h - o0) / (o1 - o0)).clamp(0.001, 0.999);
                hint_after[seg] = Some(frac);
            }
        }
    }
    // Build the resolved stops, attaching hints as segment fractions.
    // Non-finite offsets cannot come from written percentages, only
    // from px positions divided by a zero reference length (degenerate
    // geometry, which renders last-stop-wins regardless); pin them to
    // 0.0 so NaN can never reach the rasterizer or the content hash.
    // Pushed one at a time into the inline-first `GradientStops`: the
    // common 1-2 stop case never touches the allocator.
    let mut stops = GradientStops::new();
    for (i, (e, offset)) in entries.into_iter().zip(offsets).enumerate() {
        stops.push(ColorStop {
            color: e.color,
            offset,
            hint_after: hint_after[i],
        });
    }
    Some(stops)
}

// ---------------------------------------------------------------------------
// Rasterization
// ---------------------------------------------------------------------------

/// sRGB (0..1) to linear-light.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear-light (0..1) to sRGB.
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// sRGB (0..1) to Oklab.
fn srgb_to_oklab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let r = srgb_to_linear(r);
    let g = srgb_to_linear(g);
    let b = srgb_to_linear(b);
    let l = 0.41222146 * r + 0.53633255 * g + 0.051445995 * b;
    let m = 0.2119035 * r + 0.6806995 * g + 0.10739696 * b;
    let s = 0.08830246 * r + 0.28171885 * g + 0.6299787 * b;
    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();
    (
        0.21045426 * l_ + 0.7936178 * m_ - 0.004072047 * s_,
        1.9779985 * l_ - 2.4285922 * m_ + 0.4505937 * s_,
        0.025904037 * l_ + 0.78277177 * m_ - 0.80867577 * s_,
    )
}

/// Oklab to sRGB (0..1).
fn oklab_to_srgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let l_ = l + 0.39633778 * a + 0.21580376 * b;
    let m_ = l - 0.105561346 * a - 0.06385417 * b;
    let s_ = l - 0.08948418 * a - 1.2914855 * b;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    (
        linear_to_srgb(4.0767417 * l - 3.3077116 * m + 0.23096994 * s),
        linear_to_srgb(-1.268438 * l + 2.6097574 * m - 0.34131938 * s),
        linear_to_srgb(-0.0041960864 * l - 0.7034186 * m + 1.7076147 * s),
    )
}

/// A stop color as premultiplied Oklab plus alpha (CSS Color 4
/// interpolation: premultiplied alpha in the interpolation space).
fn to_premul_oklab(c: Color) -> (f32, f32, f32, f32) {
    let alpha = c.a();
    let (l, a, b) = srgb_to_oklab(c.r(), c.g(), c.b());
    (l * alpha, a * alpha, b * alpha, alpha)
}

/// Interpolate two premultiplied-Oklab stop colors
/// (`(l, a, b, alpha)` tuples from [`to_premul_oklab`]; CSS Color 4
/// default for gradients). Returns straight RGBA8.
fn lerp_color_oklab(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32), t: f32) -> [u8; 4] {
    let (la, aa, ba, alpha_a) = a;
    let (lb, ab, bb, alpha_b) = b;
    let alpha = alpha_a + (alpha_b - alpha_a) * t;
    // Quantize the alpha first: an alpha that rounds to 0 is
    // transparent, and unpremultiplying it would amplify float noise
    // into garbage channels (e.g. t = 0.9999999 on a red->transparent
    // gradient must be transparent black, not [214, 0, 0, 0]).
    if (alpha.clamp(0.0, 1.0) * 255.0).round() == 0.0 {
        return [0, 0, 0, 0];
    }
    let l = (la + (lb - la) * t) / alpha;
    let ac = (aa + (ab - aa) * t) / alpha;
    let bc = (ba + (bb - ba) * t) / alpha;
    let (r, g, bl) = oklab_to_srgb(l, ac, bc);
    let c = Color::from_f32(r, g, bl, alpha);
    [c.r8(), c.g8(), c.b8(), c.a8()]
}

/// Apply a transition hint: `local` is the point's fraction within its
/// segment, `hint` the hint's fraction (where the 50% blend lands).
/// CSS Images 4: `C = P ^ log_H(0.5)`.
fn apply_hint(local: f32, hint: f32) -> f32 {
    let h = hint.clamp(0.001, 0.999);
    local.powf((0.5f32).ln() / h.ln())
}

/// The gradient-line parameter `t` for a page-px point: 0 at the start
/// of the gradient line, 1 at its end.
fn gradient_t(g: &Gradient, rect: &Rect, x: f32, y: f32) -> f32 {
    match g.kind {
        GradientKind::Linear { x0, y0, dx, dy } => {
            let px = (x - rect.x) / rect.w;
            let py = (y - rect.y) / rect.h;
            ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)
        }
        GradientKind::Radial { cx, cy, rx, ry } => {
            // Degenerate (zero radius) gradients paint the last stop's
            // color everywhere; signal with +infinity so the sampler
            // clamps to the last stop.
            if rx <= 0.0 || ry <= 0.0 {
                return f32::INFINITY;
            }
            let dx = ((x - rect.x) / rect.w - cx) / rx;
            let dy = ((y - rect.y) / rect.h - cy) / ry;
            dx.hypot(dy)
        }
        GradientKind::Conic { cx, cy, from } => {
            let dx = (x - rect.x) / rect.w - cx;
            let dy = (y - rect.y) / rect.h - cy;
            // 0 = up, increasing clockwise (CSS angle convention).
            let mut a = dx.atan2(-dy) - from;
            a %= std::f32::consts::TAU;
            if a < 0.0 {
                a += std::f32::consts::TAU;
            }
            a / std::f32::consts::TAU
        }
    }
}

/// A resolved gradient with stop colors preconverted to
/// premultiplied Oklab: one sRGB->Oklab conversion per stop per bake
/// instead of per pixel (each conversion costs ~9 `powf` + 3 `cbrt`).
struct PreparedGradient<'a> {
    gradient: &'a Gradient,
    /// Premultiplied Oklab `(l, a, b, alpha)` per stop, parallel to
    /// [`Gradient::stops`].
    oklab: Vec<(f32, f32, f32, f32)>,
}

impl<'a> PreparedGradient<'a> {
    fn prepare(g: &'a Gradient) -> Self {
        Self {
            gradient: g,
            oklab: g.stops.iter().map(|s| to_premul_oklab(s.color)).collect(),
        }
    }

    /// Evaluate at one page-px point. Returns straight RGBA8.
    fn sample(&self, rect: &Rect, x: f32, y: f32) -> [u8; 4] {
        let g = self.gradient;
        let stops = &g.stops;
        // Empty/single-stop gradients are degenerate (`resolve`
        // never produces them); fall back gracefully instead of
        // panicking.
        if stops.is_empty() {
            return [0, 0, 0, 0];
        }
        if stops.len() == 1 {
            let c = stops[0].color;
            return [c.r8(), c.g8(), c.b8(), c.a8()];
        }
        let mut t = gradient_t(g, rect, x, y);
        if g.repeating {
            // Repeating gradients tile the first-to-last-stop interval
            // indefinitely in both directions.
            let first = stops[0].offset;
            let span = stops[stops.len() - 1].offset - first;
            t = if span > 0.0 {
                first + (t - first - ((t - first) / span).floor() * span)
            } else {
                first
            };
        } else {
            t = t.clamp(0.0, 1.0);
        }
        if t < stops[0].offset {
            let c = stops[0].color;
            return [c.r8(), c.g8(), c.b8(), c.a8()];
        }
        // Binary search the first stop with offset > t: the sample
        // sits in the segment before it (offsets are ascending
        // post-fixup). A linear scan here is a CPU DoS on hostile
        // many-stop gradients.
        let mut lo = 1usize;
        let mut hi = stops.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if stops[mid].offset <= t {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo >= stops.len() {
            let c = stops[stops.len() - 1].color;
            return [c.r8(), c.g8(), c.b8(), c.a8()];
        }
        let i = lo - 1;
        let (o0, o1) = (stops[i].offset, stops[i + 1].offset);
        if o1 <= o0 {
            // Hard stop: the later stop wins at the shared offset.
            let c = stops[i + 1].color;
            return [c.r8(), c.g8(), c.b8(), c.a8()];
        }
        let mut local = (t - o0) / (o1 - o0);
        if let Some(hint) = stops[i].hint_after {
            local = apply_hint(local, hint);
        }
        lerp_color_oklab(self.oklab[i], self.oklab[i + 1], local)
    }
}

/// Evaluate the resolved gradient at one page-px point. Returns
/// straight RGBA8.
pub fn sample_gradient(g: &Gradient, rect: &Rect, x: f32, y: f32) -> [u8; 4] {
    PreparedGradient::prepare(g).sample(rect, x, y)
}

/// Rasterize the resolved gradient over `rect` into a straight-RGBA8
/// buffer of `width` x `height` pixels, sampling pixel centers.
/// Shared by the CPU backend and the GPU backend's texture bake, so
/// both backends produce identical pixels. A zero width or height
/// yields an empty buffer (no panic).
pub fn bake_gradient_rgba8(g: &Gradient, rect: &Rect, width: u32, height: u32) -> Vec<u8> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u8; w * h * 4];
    // Stop colors are converted to premultiplied Oklab once here, not
    // per pixel.
    let pg = PreparedGradient::prepare(g);
    let sx = rect.w / width as f32;
    let sy = rect.h / height as f32;
    for (y, row) in out.chunks_exact_mut(w * 4).enumerate() {
        let py = rect.y + (y as f32 + 0.5) * sy;
        for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let x = rect.x + (x as f32 + 0.5) * sx;
            px.copy_from_slice(&pg.sample(rect, x, py));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(w: f32, h: f32) -> Rect {
        Rect {
            x: 0.0,
            y: 0.0,
            w,
            h,
        }
    }

    fn resolve(css: &str, w: f32, h: f32) -> Gradient {
        let parsed = parse_gradient(css).expect("parse");
        parsed
            .resolve(&rect(w, h), Color::rgba8(1, 2, 3, 255))
            .expect("resolve")
    }

    fn offsets(g: &Gradient) -> Vec<f32> {
        g.stops.iter().map(|s| s.offset).collect()
    }

    /// `GradientStops` goes inline for the typical 1-2 stop case and
    /// spills for longer lists; the deref'd slice covers exactly the
    /// live stops either way.
    #[test]
    fn gradient_stops_inline_and_spill() {
        let stop = |offset: f32| ColorStop {
            color: Color::rgba8(255, 0, 0, 255),
            offset,
            hint_after: None,
        };
        let two: GradientStops = vec![stop(0.0), stop(1.0)].into();
        assert!(matches!(two, GradientStops::Inline { len: 2, .. }));
        assert_eq!(two.len(), 2);
        assert_eq!(two[1].offset, 1.0);
        assert_eq!(two.as_slice().len(), 2);

        let three: GradientStops = vec![stop(0.0), stop(0.5), stop(1.0)].into();
        assert!(matches!(three, GradientStops::Spill(_)));
        assert_eq!(three.len(), 3);
        assert_eq!(three[2].offset, 1.0);

        // The `push` path spills exactly once, at the third stop.
        let mut pushed = GradientStops::new();
        assert!(pushed.is_empty());
        pushed.push(stop(0.0));
        pushed.push(stop(0.5));
        assert!(matches!(pushed, GradientStops::Inline { len: 2, .. }));
        pushed.push(stop(1.0));
        assert!(matches!(pushed, GradientStops::Spill(_)));
        assert_eq!(pushed.len(), 3);
        assert_eq!(pushed.as_slice()[0].offset, 0.0);

        // Canonical construction: equal stop lists compare equal
        // regardless of the construction path.
        assert_eq!(pushed, three);
    }

    /// Offsets carry float error from the gradient-line math (e.g.
    /// `sin(pi)` is not exactly 0), so compare approximately.
    fn assert_offsets_approx(g: &Gradient, want: &[f32]) {
        let got = offsets(g);
        assert_eq!(got.len(), want.len(), "stop count");
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).abs() < 1e-4, "offset {a} != {b}");
        }
    }

    #[test]
    fn named_color_table_is_sorted_for_binary_search() {
        for pair in NAMED_COLORS.windows(2) {
            assert!(
                pair[0].0 < pair[1].0,
                "named color table out of order: {} >= {}",
                pair[0].0,
                pair[1].0
            );
        }
        assert_eq!(NAMED_COLORS.len(), 148);
    }

    #[test]
    fn parse_linear_defaults_to_bottom() {
        let p = parse_gradient("linear-gradient(red, blue)").unwrap();
        assert_eq!(p.func, GradientFunc::Linear);
        assert!(!p.func.is_repeating());
        // 180 degrees = pi radians.
        assert!(matches!(
            p.linear_dir,
            Some(LinearDir::Angle(a)) if (a - std::f32::consts::PI).abs() < 1e-6
        ));
        assert_eq!(p.stops.len(), 2);
    }

    #[test]
    fn parse_linear_angles_and_keywords() {
        let p = parse_gradient("linear-gradient(45deg, red, blue)").unwrap();
        assert!(matches!(
            p.linear_dir,
            Some(LinearDir::Angle(a)) if (a - std::f32::consts::FRAC_PI_4).abs() < 1e-6
        ));
        let p = parse_gradient("linear-gradient(to right, red, blue)").unwrap();
        assert!(matches!(
            p.linear_dir,
            Some(LinearDir::Angle(a)) if (a - std::f32::consts::FRAC_PI_2).abs() < 1e-6
        ));
        // `to <corner>` stays a corner marker (no fixed 315deg): the
        // angle is box-dependent (CSS Images 4).
        let p = parse_gradient("linear-gradient(to top left, red, blue)").unwrap();
        assert_eq!(
            p.linear_dir,
            Some(LinearDir::Corner {
                to_top: true,
                to_right: false
            })
        );
        let p = parse_gradient("linear-gradient(to bottom right, red, blue)").unwrap();
        assert_eq!(
            p.linear_dir,
            Some(LinearDir::Corner {
                to_top: false,
                to_right: true
            })
        );
        // Keyword order does not matter.
        let p = parse_gradient("linear-gradient(to right top, red, blue)").unwrap();
        assert_eq!(
            p.linear_dir,
            Some(LinearDir::Corner {
                to_top: true,
                to_right: true
            })
        );
        let p = parse_gradient("linear-gradient(0.5turn, red, blue)").unwrap();
        assert!(matches!(
            p.linear_dir,
            Some(LinearDir::Angle(a)) if (a - std::f32::consts::PI).abs() < 1e-6
        ));
        let p = parse_gradient("repeating-linear-gradient(red, blue)").unwrap();
        assert_eq!(p.func, GradientFunc::RepeatingLinear);
        assert!(p.func.is_repeating());
    }

    #[test]
    fn corner_direction_uses_box_diagonal() {
        // `to top right` runs corner-to-corner: atan2(w, h) in CSS
        // convention (0 = up, clockwise), not a fixed 45 degrees.
        let corner = LinearDir::Corner {
            to_top: true,
            to_right: true,
        };
        // Square box: the diagonal is 45 degrees (the old fixed value).
        let a = corner.angle_for(100.0, 100.0);
        assert!((a - std::f32::consts::FRAC_PI_4).abs() < 1e-6, "a={a}");
        // 200x100 box: atan2(200, 100) ~= 63.43 degrees.
        let a = corner.angle_for(200.0, 100.0);
        let want = 200.0f32.atan2(100.0);
        assert!((a - want).abs() < 1e-6, "a={a} want={want}");
        // `to bottom left` is the opposite diagonal.
        let corner = LinearDir::Corner {
            to_top: false,
            to_right: false,
        };
        let a = corner.angle_for(200.0, 100.0);
        let want = (-200.0f32).atan2(-100.0);
        assert!((a - want).abs() < 1e-6, "a={a} want={want}");
        // End-to-end: `to top right` on a 200x100 box runs the
        // gradient line exactly corner to corner in box fractions
        // (start = bottom-left, end = top-right). A fixed 45-degree
        // angle would start at x0 = 0.125, y0 = 1.25 instead.
        let g = resolve("linear-gradient(to top right, red, blue)", 200.0, 100.0);
        match g.kind {
            GradientKind::Linear { x0, y0, dx, dy } => {
                assert!(x0.abs() < 1e-4, "x0={x0}");
                assert!((y0 - 1.0).abs() < 1e-4, "y0={y0}");
                assert!((dx - 1.0).abs() < 1e-4, "dx={dx}");
                assert!((dy + 1.0).abs() < 1e-4, "dy={dy}");
            }
            _ => panic!("expected linear"),
        }
        let r = rect(200.0, 100.0);
        // Near the top-right corner the sample is nearly all blue;
        // near the bottom-left it is nearly all red.
        let tr = sample_gradient(&g, &r, 195.0, 5.0);
        assert!(
            tr[2] > 200 && tr[0] < 60,
            "top-right should be blue: {tr:?}"
        );
        let bl = sample_gradient(&g, &r, 5.0, 95.0);
        assert!(
            bl[0] > 200 && bl[2] < 60,
            "bottom-left should be red: {bl:?}"
        );
    }

    #[test]
    fn parse_rejects_bad_input() {
        for bad in [
            "radial-gradient(red, blue)", // sanity: this one is fine, skip
            "linear-gradient(red)",
            "linear-gradient()",
            "linear-gradient(red, blue",
            "linear-gradient(red blue)",
            "linear-gradient(to top top, red, blue)",
            "linear-gradient(banana, blue)",
            "linear-gradient(red, blue, 30%)",
            "linear-gradient(30%, red, blue)",
            "linear-gradient(red 10px 20px 30px, blue)",
            "conic-gradient(from banana, red, blue)",
            "not-a-gradient(red, blue)",
            "",
        ] {
            if bad == "radial-gradient(red, blue)" {
                assert!(parse_gradient(bad).is_ok());
                continue;
            }
            assert!(parse_gradient(bad).is_err(), "should reject: {bad}");
        }
    }

    #[test]
    fn parse_radial_preludes() {
        let p =
            parse_gradient("radial-gradient(circle closest-side at 10px 20px, red, blue)").unwrap();
        let r = p.radial.unwrap();
        assert!(r.circle);
        assert_eq!(r.size, RadialSize::Extent(RadialExtent::ClosestSide));
        assert_eq!(r.center.x, PosComp::Px(10.0));
        assert_eq!(r.center.y, PosComp::Px(20.0));

        let p = parse_gradient("radial-gradient(ellipse 50px 25% at left top, red, blue)").unwrap();
        let r = p.radial.unwrap();
        assert!(!r.circle);
        assert_eq!(
            r.size,
            RadialSize::Ellipse(PosComp::Px(50.0), PosComp::Pct(25.0))
        );
        assert_eq!(r.center.x, PosComp::Pct(0.0));
        assert_eq!(r.center.y, PosComp::Pct(0.0));

        let p = parse_gradient("radial-gradient(30px, red, blue)").unwrap();
        let r = p.radial.unwrap();
        assert_eq!(r.size, RadialSize::CirclePx(30.0));

        // Single percentage is not a valid circle radius.
        assert!(parse_gradient("radial-gradient(30%, red, blue)").is_err());
        // Explicit circle radius with `ellipse` is invalid.
        assert!(parse_gradient("radial-gradient(ellipse 30px, red, blue)").is_err());
    }

    #[test]
    fn parse_conic_preludes() {
        let p = parse_gradient("conic-gradient(from 90deg at 25% 75%, red, blue)").unwrap();
        let c = p.conic.unwrap();
        assert!((c.from - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert_eq!(c.center.x, PosComp::Pct(25.0));
        assert_eq!(c.center.y, PosComp::Pct(75.0));

        let p = parse_gradient("repeating-conic-gradient(red, blue)").unwrap();
        assert_eq!(p.func, GradientFunc::RepeatingConic);
        let c = p.conic.unwrap();
        assert_eq!(c.from, 0.0);
        assert_eq!(c.center.x, PosComp::Pct(50.0));
    }

    #[test]
    fn parse_colors() {
        let p = parse_gradient("linear-gradient(#f00, #00ff0080, rgb(0,0,255), rgba(0,255,0,0.5), hsl(60, 100%, 50%), rebeccapurple, transparent, blue)").unwrap();
        assert_eq!(p.stops.len(), 8);
        let colors: Vec<Color> = p
            .stops
            .iter()
            .map(|s| match s {
                ParsedStopItem::Stop(st) => match st.color {
                    StopColor::Rgba(c) => c,
                    StopColor::CurrentColor => panic!("unexpected currentcolor"),
                },
                ParsedStopItem::Hint(_) => panic!("unexpected hint"),
            })
            .collect();
        assert_eq!(colors[0], Color::rgba8(255, 0, 0, 255));
        assert_eq!(colors[1], Color::rgba8(0, 255, 0, 128));
        assert_eq!(colors[2], Color::rgba8(0, 0, 255, 255));
        assert_eq!(colors[3], Color::rgba8(0, 255, 0, 128));
        assert_eq!(colors[4], Color::rgba8(255, 255, 0, 255));
        assert_eq!(colors[5], Color::rgba8(102, 51, 153, 255));
        assert_eq!(colors[6], Color::rgba8(0, 0, 0, 0));
        assert_eq!(colors[7], Color::rgba8(0, 0, 255, 255));
    }

    #[test]
    fn parse_currentcolor_and_space_syntax() {
        let p = parse_gradient("linear-gradient(currentcolor, rgb(255 0 0 / 50%))").unwrap();
        assert!(matches!(
            p.stops[0],
            ParsedStopItem::Stop(ParsedStop {
                color: StopColor::CurrentColor,
                ..
            })
        ));
        // `currentcolor` resolves against the caller-supplied color.
        let g = parse_gradient("linear-gradient(currentcolor, blue)")
            .unwrap()
            .resolve(&rect(10.0, 10.0), Color::rgba8(9, 9, 9, 255))
            .unwrap();
        assert_eq!(g.stops[0].color, Color::rgba8(9, 9, 9, 255));
    }

    #[test]
    fn stop_fixup_positions() {
        // Spec example: unpositioned stops space out evenly.
        let g = resolve("linear-gradient(red, white 20%, blue)", 100.0, 100.0);
        assert_offsets_approx(&g, &[0.0, 0.2, 1.0]);
        // First/last default to 0%/100%.
        let g = resolve("linear-gradient(red 40%, white, black, blue)", 100.0, 100.0);
        assert_offsets_approx(&g, &[0.4, 0.6, 0.8, 1.0]);
        // Decreasing positions clamp up (spec fixup step 3).
        let g = resolve(
            "linear-gradient(red 80px, white 0px, black, blue 100px)",
            100.0,
            100.0,
        );
        assert_offsets_approx(&g, &[0.8, 0.8, 0.9, 1.0]);
        // Spec example 6: the last stop's step-2 position is itself
        // clamped by step 3.
        let g = resolve(
            "linear-gradient(red, white -50%, black 150%, blue)",
            100.0,
            100.0,
        );
        assert_offsets_approx(&g, &[0.0, 0.0, 1.5, 1.5]);
    }

    #[test]
    fn double_position_expands_to_two_stops() {
        let g = resolve("linear-gradient(red 10% 20%, blue)", 100.0, 100.0);
        assert_eq!(g.stops.len(), 3);
        assert_offsets_approx(&g, &[0.1, 0.2, 1.0]);
        assert_eq!(g.stops[0].color, g.stops[1].color);
    }

    #[test]
    fn transition_hint_parses_and_attaches() {
        let g = resolve("linear-gradient(red, 25%, blue)", 100.0, 100.0);
        assert_eq!(g.stops.len(), 2);
        // The hint sits 25% along the single segment.
        assert_eq!(g.stops[0].hint_after, Some(0.25));
        assert_eq!(g.stops[1].hint_after, None);
        // Misplaced hints are rejected.
        assert!(parse_gradient("linear-gradient(25%, red, blue)").is_err());
        assert!(parse_gradient("linear-gradient(red, blue, 25%)").is_err());
    }

    #[test]
    fn linear_geometry_to_bottom() {
        // `to bottom` on a 100x100 box: t = y fraction.
        let g = resolve("linear-gradient(to bottom, red, blue)", 100.0, 100.0);
        match g.kind {
            GradientKind::Linear { x0, y0, dx, dy } => {
                assert!((x0 - 0.5).abs() < 1e-6);
                assert!(y0.abs() < 1e-6);
                assert!(dx.abs() < 1e-6);
                assert!((dy - 1.0).abs() < 1e-6);
            }
            _ => panic!("expected linear"),
        }
        let r = rect(100.0, 100.0);
        assert_eq!(sample_gradient(&g, &r, 50.0, 0.0), [255, 0, 0, 255]);
        assert_eq!(sample_gradient(&g, &r, 50.0, 100.0), [0, 0, 255, 255]);
    }

    #[test]
    fn linear_lengths_resolve_against_gradient_line() {
        // 200px-wide box, `to right`: 100px = 50%.
        let g = resolve("linear-gradient(to right, red 100px, blue)", 200.0, 100.0);
        assert_offsets_approx(&g, &[0.5, 1.0]);
    }

    #[test]
    fn radial_default_is_ellipse_farthest_corner() {
        // 100x100 box, center: farthest-corner ellipse through the
        // corners => t = 1 exactly at each corner, t = 0 at center.
        let g = resolve("radial-gradient(red, blue)", 100.0, 100.0);
        match g.kind {
            GradientKind::Radial { cx, cy, rx, ry } => {
                assert!((cx - 0.5).abs() < 1e-6);
                assert!((cy - 0.5).abs() < 1e-6);
                let want = 50.0f32.hypot(50.0) / 100.0;
                assert!((rx - want).abs() < 1e-5, "rx={rx} want={want}");
                assert!((ry - want).abs() < 1e-5, "ry={ry} want={want}");
            }
            _ => panic!("expected radial"),
        }
        let r = rect(100.0, 100.0);
        assert_eq!(sample_gradient(&g, &r, 50.0, 50.0), [255, 0, 0, 255]);
        assert_eq!(sample_gradient(&g, &r, 100.0, 100.0), [0, 0, 255, 255]);
        assert_eq!(sample_gradient(&g, &r, 0.0, 0.0), [0, 0, 255, 255]);
    }

    #[test]
    fn radial_circle_closest_side() {
        // 200x100 box, center: closest side is 50px => r = 50.
        let g = resolve(
            "radial-gradient(circle closest-side, red, blue)",
            200.0,
            100.0,
        );
        match g.kind {
            GradientKind::Radial { rx, ry, .. } => {
                assert!((rx * 200.0 - 50.0).abs() < 1e-4);
                assert!((ry * 100.0 - 50.0).abs() < 1e-4);
            }
            _ => panic!("expected radial"),
        }
        let r = rect(200.0, 100.0);
        // Top-middle is on the ending circle: blue.
        assert_eq!(sample_gradient(&g, &r, 100.0, 0.0), [0, 0, 255, 255]);
    }

    #[test]
    fn radial_explicit_size_and_position() {
        let g = resolve(
            "radial-gradient(ellipse 40px 20px at 10px 10px, red, blue)",
            100.0,
            100.0,
        );
        match g.kind {
            GradientKind::Radial { cx, cy, rx, ry } => {
                assert!((cx - 0.1).abs() < 1e-6);
                assert!((cy - 0.1).abs() < 1e-6);
                assert!((rx - 0.4).abs() < 1e-6);
                assert!((ry - 0.2).abs() < 1e-6);
            }
            _ => panic!("expected radial"),
        }
        // Percentages resolve against the horizontal gradient ray:
        // 50% = half the horizontal radius.
        let g = resolve("radial-gradient(circle 40px, red 50%, blue)", 100.0, 100.0);
        assert_offsets_approx(&g, &[0.5, 1.0]);
    }

    #[test]
    fn radial_degenerate_paints_last_stop() {
        // Center on the top edge with closest-side: zero radius.
        let g = resolve(
            "radial-gradient(circle closest-side at center top, red, blue)",
            100.0,
            100.0,
        );
        let r = rect(100.0, 100.0);
        assert_eq!(sample_gradient(&g, &r, 50.0, 50.0), [0, 0, 255, 255]);
        assert_eq!(sample_gradient(&g, &r, 10.0, 10.0), [0, 0, 255, 255]);
    }

    #[test]
    fn conic_angles() {
        // `from 0deg`: 0% at the top, sweeping clockwise.
        let g = resolve("conic-gradient(red, blue)", 100.0, 100.0);
        let r = rect(100.0, 100.0);
        assert_eq!(sample_gradient(&g, &r, 50.0, 0.0), [255, 0, 0, 255]);
        // 90deg (right side) is 25% into the red->blue blend.
        let quarter = sample_gradient(&g, &r, 100.0, 50.0);
        assert!(
            quarter[0] > 0 && quarter[2] > 0,
            "quarter blend should mix red and blue: {quarter:?}"
        );
        assert_ne!(quarter, [255, 0, 0, 255]);
        assert_ne!(quarter, [0, 0, 255, 255]);
        // `from 90deg` rotates the whole gradient: red moves to the right.
        let g = resolve("conic-gradient(from 90deg, red, blue)", 100.0, 100.0);
        assert_eq!(sample_gradient(&g, &r, 100.0, 50.0), [255, 0, 0, 255]);
    }

    #[test]
    fn conic_angle_stops() {
        // Conic stops accept angles: 90deg = 25% of the turn.
        let g = resolve(
            "conic-gradient(red 0deg, blue 90deg, red 180deg)",
            100.0,
            100.0,
        );
        assert_offsets_approx(&g, &[0.0, 0.25, 0.5]);
        // Lengths are rejected for conic stops.
        assert!(parse_gradient("conic-gradient(red 10px, blue)").is_err());
    }

    #[test]
    fn repeating_wraps() {
        let g = resolve(
            "repeating-linear-gradient(to right, red 0px, blue 50px)",
            200.0,
            100.0,
        );
        assert!(g.repeating);
        let r = rect(200.0, 100.0);
        // t at x=0 is 0 (red); x=50 -> t=0.25... check the wrap: x=100
        // is t=0.5 of the second repeat, same as x=50.
        let a = sample_gradient(&g, &r, 50.0, 50.0);
        let b = sample_gradient(&g, &r, 150.0, 50.0);
        assert_eq!(a, b);
        assert_eq!(sample_gradient(&g, &r, 0.0, 50.0), [255, 0, 0, 255]);
        assert_eq!(sample_gradient(&g, &r, 100.0, 50.0), [255, 0, 0, 255]);
    }

    #[test]
    fn oklab_midpoint_is_not_srgb_midpoint() {
        // CSS Color 4: gradients interpolate in Oklab, not sRGB. The
        // Oklab midpoint of red/blue differs from the naive sRGB
        // midpoint (127, 0, 127).
        let g = resolve("linear-gradient(to right, red, blue)", 100.0, 100.0);
        let r = rect(100.0, 100.0);
        let mid = sample_gradient(&g, &r, 50.0, 50.0);
        assert_ne!(mid, [127, 0, 127, 255]);
        // Sanity: fully red at one end, fully blue at the other, and
        // the midpoint is a purple (r and b both well above 0).
        assert!(mid[0] > 100 && mid[2] > 100, "mid={mid:?}");
        assert_eq!(mid[3], 255);
    }

    #[test]
    fn alpha_interpolates_premultiplied() {
        // `red -> transparent`: premultiplied interpolation keeps the
        // hue (no gray fringe): halfway is half-alpha red.
        let g = resolve("linear-gradient(to right, red, transparent)", 100.0, 100.0);
        let r = rect(100.0, 100.0);
        let mid = sample_gradient(&g, &r, 50.0, 50.0);
        assert_eq!(mid, [255, 0, 0, 128]);
        assert_eq!(sample_gradient(&g, &r, 100.0, 50.0), [0, 0, 0, 0]);
    }

    #[test]
    fn hard_stops_render_solid_bands() {
        // Two stops at the same offset: an infinitesimal transition
        // (the later stop wins at the shared offset).
        let g = resolve("linear-gradient(to right, red 50%, blue 50%)", 100.0, 100.0);
        let r = rect(100.0, 100.0);
        assert_eq!(sample_gradient(&g, &r, 0.0, 50.0), [255, 0, 0, 255]);
        assert_eq!(sample_gradient(&g, &r, 49.0, 50.0), [255, 0, 0, 255]);
        assert_eq!(sample_gradient(&g, &r, 50.0, 50.0), [0, 0, 255, 255]);
        assert_eq!(sample_gradient(&g, &r, 99.0, 50.0), [0, 0, 255, 255]);
    }

    #[test]
    fn content_hash_is_stable_and_sensitive() {
        let a = resolve("linear-gradient(red, blue)", 100.0, 100.0);
        let b = resolve("linear-gradient(red, blue)", 100.0, 100.0);
        let c = resolve("linear-gradient(red, green)", 100.0, 100.0);
        let d = resolve("radial-gradient(red, blue)", 100.0, 100.0);
        assert_eq!(a.content_hash(), b.content_hash());
        assert_ne!(a.content_hash(), c.content_hash());
        assert_ne!(a.content_hash(), d.content_hash());
    }

    #[test]
    fn bake_matches_sampler() {
        let g = resolve(
            "radial-gradient(circle at 30% 40%, red, 25%, blue)",
            64.0,
            48.0,
        );
        let r = rect(64.0, 48.0);
        let px = bake_gradient_rgba8(&g, &r, 64, 48);
        assert_eq!(px.len(), 64 * 48 * 4);
        // Every baked pixel equals the sampler at the pixel center.
        for (x, y) in [(0, 0), (32, 24), (63, 47), (10, 40), (63, 0)] {
            let want = sample_gradient(&g, &r, x as f32 + 0.5, y as f32 + 0.5);
            let i = (y * 64 + x) as usize * 4;
            assert_eq!(&px[i..i + 4], &want, "pixel ({x},{y})");
        }
        // Exact bands for a hard-stop linear gradient: rows 0-1
        // (centers y=0.5,1.5, t<0.5) are red, rows 2-3 are blue.
        let g = resolve("linear-gradient(to bottom, red 50%, blue 50%)", 4.0, 4.0);
        let r = rect(4.0, 4.0);
        let px = bake_gradient_rgba8(&g, &r, 4, 4);
        assert_eq!(&px[0..4], &[255, 0, 0, 255]);
        assert_eq!(&px[4 * 4..4 * 4 + 4], &[255, 0, 0, 255]);
        assert_eq!(&px[2 * 4 * 4..2 * 4 * 4 + 4], &[0, 0, 255, 255]);
        assert_eq!(&px[3 * 4 * 4..3 * 4 * 4 + 4], &[0, 0, 255, 255]);
    }

    #[test]
    fn degenerate_rect_resolves_to_none() {
        let p = parse_gradient("linear-gradient(red, blue)").unwrap();
        assert!(
            p.resolve(
                &Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 0.0,
                    h: 10.0
                },
                Color::rgba8(0, 0, 0, 255)
            )
            .is_none()
        );
    }

    #[test]
    fn transition_hint_does_not_raise_stop_clamp_floor() {
        // CSS Images 4: stops clamp against the largest *specified
        // color stop* before them; transition hints never affect stops.
        // `red 0%, 90%, blue 50%` must resolve to [0.0, 0.5], not
        // [0.0, 0.9].
        let g = resolve("linear-gradient(red 0%, 90%, blue 50%)", 100.0, 100.0);
        assert_offsets_approx(&g, &[0.0, 0.5]);
        // A hint before the stop floor still clamps up to the floor.
        let g = resolve("linear-gradient(red 50%, 30%, blue 70%)", 100.0, 100.0);
        assert_offsets_approx(&g, &[0.5, 0.7]);
        assert_eq!(g.stops[0].hint_after, Some(0.001));
    }

    #[test]
    fn interior_coincident_stops_later_wins() {
        // At a shared offset the later stop wins (spec hard-stop
        // rule): t = 0.5 is blue, not red.
        let g = resolve(
            "linear-gradient(to right, green 0%, red 50%, blue 50%, yellow 100%)",
            100.0,
            100.0,
        );
        let r = rect(100.0, 100.0);
        assert_eq!(sample_gradient(&g, &r, 50.0, 50.0), [0, 0, 255, 255]);
        // Just before the shared offset the earlier segment still
        // blends toward red.
        let before = sample_gradient(&g, &r, 49.0, 50.0);
        assert!(before[0] > 200, "should be reddish: {before:?}");
        assert_eq!(sample_gradient(&g, &r, 100.0, 50.0), [255, 255, 0, 255]);
    }

    #[test]
    fn parse_rejects_negative_explicit_radial_sizes() {
        for bad in [
            "radial-gradient(circle -10px, red, blue)",
            "radial-gradient(-10px, red, blue)",
            "radial-gradient(ellipse -10px 20px, red, blue)",
            "radial-gradient(ellipse 10px -20px, red, blue)",
            "radial-gradient(ellipse -10% 20%, red, blue)",
            "radial-gradient(ellipse 10% -20%, red, blue)",
        ] {
            assert!(parse_gradient(bad).is_err(), "should reject: {bad}");
        }
        // Zero and positive explicit sizes still parse.
        assert!(parse_gradient("radial-gradient(circle 0px, red, blue)").is_ok());
        assert!(parse_gradient("radial-gradient(ellipse 10px 20%, red, blue)").is_ok());
        assert!(parse_gradient("radial-gradient(circle 10px, red, blue)").is_ok());
    }

    #[test]
    fn bake_handles_zero_size_and_empty_stops_gracefully() {
        let g = resolve("linear-gradient(red, blue)", 100.0, 100.0);
        let r = rect(100.0, 100.0);
        // Zero width/height: empty buffer, no panic (a zero chunk size
        // would panic in `chunks_exact_mut`).
        assert!(bake_gradient_rgba8(&g, &r, 0, 10).is_empty());
        assert!(bake_gradient_rgba8(&g, &r, 10, 0).is_empty());
        assert!(bake_gradient_rgba8(&g, &r, 0, 0).is_empty());
        // Empty stops (`pub` struct, `pub` fields): graceful fallback.
        let empty = Gradient {
            kind: g.kind,
            stops: GradientStops::new(),
            repeating: false,
        };
        assert_eq!(sample_gradient(&empty, &r, 50.0, 50.0), [0, 0, 0, 0]);
        let px = bake_gradient_rgba8(&empty, &r, 4, 4);
        assert_eq!(px.len(), 4 * 4 * 4);
        assert!(px.iter().all(|&b| b == 0));
        // A single stop paints its color everywhere.
        let single = Gradient {
            kind: g.kind,
            stops: vec![ColorStop {
                color: Color::rgba8(1, 2, 3, 255),
                offset: 0.0,
                hint_after: None,
            }]
            .into(),
            repeating: false,
        };
        assert_eq!(sample_gradient(&single, &r, 50.0, 50.0), [1, 2, 3, 255]);
    }

    #[test]
    fn parse_rejects_whitespace_inside_dimensions_angles_and_functions() {
        for bad in [
            "linear-gradient(red 10 px, blue)",
            "linear-gradient(90 deg, red, blue)",
            "linear-gradient(rgb (255, 0, 0), blue)",
            "linear-gradient(rgba (255, 0, 0, 1), blue)",
            "linear-gradient(hsl (0, 100%, 50%), blue)",
            "conic-gradient(from 90 deg, red, blue)",
            "conic-gradient(red 90 deg, blue)",
            "radial-gradient(circle 10 px, red, blue)",
        ] {
            assert!(parse_gradient(bad).is_err(), "should reject: {bad}");
        }
        // The no-whitespace forms still parse.
        for ok in [
            "linear-gradient(red 10px, blue)",
            "linear-gradient(90deg, red, blue)",
            "linear-gradient(rgb(255, 0, 0), blue)",
            "conic-gradient(from 90deg, red, blue)",
            "conic-gradient(red 90deg, blue)",
            "radial-gradient(circle 10px, red, blue)",
        ] {
            assert!(parse_gradient(ok).is_ok(), "should accept: {ok}");
        }
    }

    #[test]
    fn many_stop_gradient_samples_correctly() {
        // 10001 stops: the sampler binary-searches ascending offsets
        // (a linear per-pixel scan would be a CPU DoS here).
        let mut css = String::from("linear-gradient(to right");
        for i in 0..=10_000 {
            let p = i as f32 / 100.0;
            let v = (i % 256) as u8;
            css.push_str(&format!(", rgb({v},{v},{v}) {p}%"));
        }
        css.push(')');
        let g = parse_gradient(&css)
            .expect("parse")
            .resolve(&rect(1000.0, 10.0), Color::rgba8(0, 0, 0, 255))
            .expect("resolve");
        assert_eq!(g.stops.len(), 10_001);
        let r = rect(1000.0, 10.0);
        // Stop i sits exactly at x = i / 10 px.
        assert_eq!(sample_gradient(&g, &r, 0.0, 5.0), [0, 0, 0, 255]);
        assert_eq!(sample_gradient(&g, &r, 500.0, 5.0), [136, 136, 136, 255]);
        assert_eq!(sample_gradient(&g, &r, 1000.0, 5.0), [16, 16, 16, 255]);
        // Between stops 2500 and 2501 the blend is between their grays.
        let mid = sample_gradient(&g, &r, 250.05, 5.0);
        let (v0, v1) = (2500 % 256, 2501 % 256);
        assert!(
            mid[0] == v0 as u8 || mid[0] == v1 as u8,
            "mid-blend should be near a neighbor gray: {mid:?}"
        );
    }

    #[test]
    fn content_hash_canonicalizes_float_noise() {
        let mk = |offset: f32| Gradient {
            kind: GradientKind::Linear {
                x0: 0.5,
                y0: 0.0,
                dx: 0.0,
                dy: 1.0,
            },
            stops: vec![
                ColorStop {
                    color: Color::rgba8(255, 0, 0, 255),
                    offset,
                    hint_after: None,
                },
                ColorStop {
                    color: Color::rgba8(0, 0, 255, 255),
                    offset: 1.0,
                    hint_after: None,
                },
            ]
            .into(),
            repeating: false,
        };
        // -0.0 and +0.0 are the same gradient: one cache entry.
        assert_eq!(mk(0.0).content_hash(), mk(-0.0).content_hash());
        // NaN payloads all map to one canonical pattern.
        assert_eq!(mk(f32::NAN).content_hash(), mk(-f32::NAN).content_hash());
        assert_eq!(
            mk(f32::NAN).content_hash(),
            mk(f32::from_bits(0x7f80_0001)).content_hash()
        );
        // Distinct gradients still hash distinctly.
        assert_ne!(mk(0.0).content_hash(), mk(0.5).content_hash());
    }

    /// `content_hash` performs zero heap allocation: the GPU backend
    /// re-hashes every gradient item once per frame for its
    /// texture-cache key, so the old per-call `Vec<u8>` byte-stream
    /// staging was allocator churn on the paint hot path. Measured with
    /// the crate's shared per-thread counting allocator
    /// (`images::test_alloc`).
    #[test]
    fn content_hash_allocates_nothing() {
        use crate::images::test_alloc::ALLOCATED_BYTES;
        let g = resolve("linear-gradient(red, blue)", 100.0, 100.0);
        // Warm: `resolve` itself allocates; only the hash calls below
        // are measured.
        let _ = g.content_hash();
        let before = ALLOCATED_BYTES.with(|c| c.get());
        let h1 = g.content_hash();
        let h2 = g.content_hash();
        let after = ALLOCATED_BYTES.with(|c| c.get());
        assert_eq!(h1, h2);
        assert_eq!(
            after,
            before,
            "content_hash allocated {} bytes",
            after.saturating_sub(before)
        );
    }
}
