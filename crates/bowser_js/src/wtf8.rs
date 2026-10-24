//! WTF-8 strings: the storage encoding for JavaScript strings.
//!
//! A JS string is a sequence of UTF-16 code units and may contain lone
//! surrogates, which are not Unicode scalar values and therefore cannot
//! live in Rust's `str`/`String`. WTF-8 extends UTF-8 by encoding each
//! surrogate code point `U+D800..=U+DFFF` as the 3-byte CESU-8 sequence
//! `0xED 0xA0..=0xBF 0x80..=0xBF`. Every valid UTF-8 string is valid WTF-8
//! with byte-identical content, so ASCII and BMP text is untouched and all
//! byte-level operations (hashing, equality, prefix tests, the ASCII fast
//! paths) keep working unchanged.
//!
//! [`Wtf8`] is the borrowed view (the analogue of `str`); [`Wtf8Buf`] is
//! the owned buffer (the analogue of `String`). Both are `#[repr(transparent)]`
//! over byte storage so conversions are free.

use std::borrow::Cow;
use std::fmt;

/// A borrowed WTF-8 string slice. Like `str`, but the bytes may contain
/// CESU-8 surrogate sequences (`0xED 0xA0..=0xBF 0x80..=0xBF`).
///
/// The type is unsized; use `&Wtf8` exactly as you would `&str`.
#[repr(transparent)]
pub struct Wtf8([u8]);

impl Wtf8 {
    /// Reinterpret a `&str` as WTF-8. Infallible: every UTF-8 string is
    /// valid WTF-8 with identical bytes. Zero cost.
    #[inline]
    /// Borrow a `&str` as WTF-8 (valid UTF-8 is valid WTF-8). Named
    /// `from_str` for the `Wtf8Buf::from(&str)` symmetry; it does not
    /// implement [`std::str::FromStr`].
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> &Wtf8 {
        // SAFETY: UTF-8 is a subset of WTF-8; `Wtf8` is repr(transparent)
        // over `[u8]`.
        unsafe { &*(s.as_bytes() as *const [u8] as *const Wtf8) }
    }

    /// Validate `b` as WTF-8 and reinterpret it, or `None` when invalid.
    pub fn from_bytes(b: &[u8]) -> Option<&Wtf8> {
        if validate_wtf8(b) {
            // SAFETY: just validated; layout is repr(transparent).
            Some(unsafe { Self::from_bytes_unchecked(b) })
        } else {
            None
        }
    }

    /// Reinterpret bytes as WTF-8 without validation.
    ///
    /// # Safety
    /// `b` must be valid WTF-8.
    #[inline]
    pub unsafe fn from_bytes_unchecked(b: &[u8]) -> &Wtf8 {
        // SAFETY: caller guarantees validity; repr(transparent) over [u8].
        unsafe { &*(b as *const [u8] as *const Wtf8) }
    }

    /// The raw WTF-8 bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Byte length (not the JS `.length`, which counts UTF-16 code units;
    /// see [`Wtf8::units_len`]).
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True when there are no bytes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Decode the sequence starting at `pos` as `(code point, byte length)`.
    /// `pos` must be a valid WTF-8 boundary (the returned length keeps it
    /// one); out-of-range yields `None`.
    pub fn code_point_at(&self, pos: usize) -> Option<(u32, usize)> {
        let b = &self.0;
        if pos >= b.len() {
            return None;
        }
        let w = utf8_seq_len(b[pos]);
        Some((decode_at(b, pos, w), w))
    }

    /// Decode the first WTF-8 sequence as `(code point, byte length)`.
    /// Surrogates decode to `0xD800..=0xDFFF`. `None` when empty.
    pub fn first_code_point(&self) -> Option<(u32, usize)> {
        self.code_point_at(0)
    }

    /// Strip an ASCII `prefix` (e.g. `b"-"`). The remainder always starts
    /// on a valid WTF-8 boundary.
    pub fn strip_ascii_prefix(&self, prefix: &[u8]) -> Option<&Wtf8> {
        let rest = self.0.strip_prefix(prefix)?;
        // SAFETY: stripping ASCII bytes cannot split a sequence.
        Some(unsafe { Wtf8::from_bytes_unchecked(rest) })
    }

    /// True when every byte is ASCII (implies well-formed and one byte per
    /// UTF-16 code unit).
    #[inline]
    pub fn is_ascii(&self) -> bool {
        self.0.is_ascii()
    }

    /// The string as valid UTF-8, or `None` when it holds lone surrogates.
    /// Well-formed WTF-8 *is* valid UTF-8, so this is free when it succeeds.
    pub fn as_utf8(&self) -> Option<&str> {
        if self.is_well_formed() {
            // SAFETY: well-formed WTF-8 contains no surrogate sequences,
            // hence is valid UTF-8.
            Some(unsafe { std::str::from_utf8_unchecked(&self.0) })
        } else {
            None
        }
    }

    /// The bytes as `&str` when the string is ASCII. ASCII bytes are always
    /// valid UTF-8, so this is free; debug-asserts the precondition. For
    /// the ASCII fast paths in the string builtins, where byte indices
    /// are code-unit indices.
    pub fn as_ascii_str(&self) -> &str {
        debug_assert!(self.is_ascii(), "as_ascii_str on non-ASCII");
        // SAFETY: ASCII bytes are valid UTF-8 by construction.
        unsafe { std::str::from_utf8_unchecked(&self.0) }
    }

    /// The string as valid UTF-8, replacing each lone surrogate with one
    /// U+FFFD. Borrowed (zero-copy) when already well-formed.
    pub fn to_utf8_lossy(&self) -> Cow<'_, str> {
        match self.as_utf8() {
            Some(s) => Cow::Borrowed(s),
            None => {
                // Walk sequences: with maximal pairing each 3-byte
                // surrogate sequence is exactly one lone surrogate.
                let b = &self.0;
                let mut out = String::with_capacity(b.len());
                let mut i = 0;
                while i < b.len() {
                    let w = utf8_seq_len(b[i]);
                    if b[i] == 0xED && (0xA0..=0xBF).contains(&b[i + 1]) {
                        out.push('\u{FFFD}');
                    } else {
                        // SAFETY: non-surrogate sequences of valid WTF-8
                        // are valid UTF-8.
                        out.push_str(unsafe { std::str::from_utf8_unchecked(&b[i..i + w]) });
                    }
                    i += w;
                }
                Cow::Owned(out)
            }
        }
    }

    /// Iterate the UTF-16 code units (what JS indexes and `.length`
    /// count). Lone surrogates surface as their unit values.
    #[inline]
    pub fn code_units(&self) -> CodeUnits<'_> {
        CodeUnits {
            bytes: &self.0,
            pos: 0,
            pending: None,
        }
    }

    /// Number of UTF-16 code units. ASCII fast path: bytes == units.
    pub fn units_len(&self) -> usize {
        if self.is_ascii() {
            return self.0.len();
        }
        self.code_units().count()
    }

    /// The `i`-th UTF-16 code unit, or `None` when out of range. ASCII fast
    /// path avoids decoding.
    pub fn nth_unit(&self, i: usize) -> Option<u16> {
        if self.is_ascii() {
            return self.0.get(i).map(|&b| u16::from(b));
        }
        self.code_units().nth(i)
    }

    /// JS `s[start..end]` by UTF-16 code-unit indices (clamped; empty when
    /// `start >= end`). The ASCII fast path slices bytes directly, and
    /// boundaries that align to code-point boundaries borrow. A boundary
    /// that splits a surrogate pair (e.g. `"\u{1F600}".slice(1, 2)` — the
    /// lone trail surrogate) materializes the unit window instead.
    pub fn slice_units(&self, start: usize, end: usize) -> Cow<'_, Wtf8> {
        if start >= end {
            return Cow::Borrowed(Wtf8::from_str(""));
        }
        if self.is_ascii() {
            let n = self.0.len();
            let (a, b) = (start.min(n), end.min(n));
            if a >= b {
                return Cow::Borrowed(Wtf8::from_str(""));
            }
            // SAFETY: ASCII subslice at byte boundaries is valid WTF-8.
            return Cow::Borrowed(unsafe { Wtf8::from_bytes_unchecked(&self.0[a..b]) });
        }
        let bytes = &self.0;
        let mut pos = 0usize;
        let mut units = 0usize;
        let mut bs: Option<usize> = None;
        let mut be: Option<usize> = None;
        while pos < bytes.len() {
            if units == start {
                bs = Some(pos);
            }
            if units == end {
                be = Some(pos);
                break;
            }
            let w = utf8_seq_len(bytes[pos]);
            units += if w == 4 { 2 } else { 1 };
            pos += w;
        }
        if units == start && bs.is_none() {
            bs = Some(pos);
        }
        if units == end && be.is_none() {
            be = Some(pos);
        }
        if let (Some(a), Some(b)) = (bs, be)
            && a <= b
        {
            // SAFETY: `a`/`b` are code-point boundaries of valid WTF-8.
            return Cow::Borrowed(unsafe { Wtf8::from_bytes_unchecked(&bytes[a..b]) });
        }
        // A boundary split a surrogate pair (or lies past the end):
        // materialize the unit window and re-encode.
        let units: Vec<u16> = self.code_units().skip(start).take(end - start).collect();
        if units.is_empty() {
            return Cow::Borrowed(Wtf8::from_str(""));
        }
        Cow::Owned(Wtf8Buf::from_units(&units))
    }

    /// Byte-level prefix test (exact for WTF-8, as for UTF-8).
    pub fn starts_with(&self, pat: &Wtf8) -> bool {
        self.0.starts_with(&pat.0)
    }

    /// Byte-level suffix test.
    pub fn ends_with(&self, pat: &Wtf8) -> bool {
        self.0.ends_with(&pat.0)
    }

    /// True when the string contains no lone surrogates (spec
    /// `IsStringWellFormedUnicode`). ASCII implies well-formed.
    pub fn is_well_formed(&self) -> bool {
        if self.is_ascii() {
            return true;
        }
        let b = &self.0;
        let mut i = 0;
        while i < b.len() {
            // Only 0xED can start a surrogate sequence; check the second
            // byte for the 0xA0..=0xBF surrogate marker.
            if b[i] == 0xED && i + 1 < b.len() && (0xA0..=0xBF).contains(&b[i + 1]) {
                return false;
            }
            i += utf8_seq_len(b[i]);
        }
        true
    }

    /// Spec `String.prototype.toWellFormed`: lone surrogates become U+FFFD.
    /// Returns a borrowed view when already well-formed.
    pub fn to_well_formed(&self) -> Cow<'_, Wtf8> {
        if self.is_well_formed() {
            return Cow::Borrowed(self);
        }
        let b = &self.0;
        let mut out = Vec::with_capacity(b.len());
        let mut i = 0;
        while i < b.len() {
            let w = utf8_seq_len(b[i]);
            if b[i] == 0xED && w == 3 && (0xA0..=0xBF).contains(&b[i + 1]) {
                out.extend_from_slice(&[0xEF, 0xBF, 0xBD]); // U+FFFD
            } else {
                out.extend_from_slice(&b[i..i + w]);
            }
            i += w;
        }
        // SAFETY: replacing surrogate sequences with U+FFFD yields valid
        // WTF-8 (in fact valid UTF-8).
        Cow::Owned(unsafe { Wtf8Buf::from_vec_unchecked(out) })
    }

    /// Iterate Unicode code points as `u32`; lone surrogates surface as
    /// `0xD800..=0xDFFF` (not valid `char`s, hence `u32`).
    pub fn chars_wtf8(&self) -> impl Iterator<Item = u32> + '_ {
        let mut pos = 0usize;
        let bytes = &self.0;
        std::iter::from_fn(move || {
            if pos >= bytes.len() {
                return None;
            }
            let w = utf8_seq_len(bytes[pos]);
            let cp = decode_at(bytes, pos, w);
            pos += w;
            Some(cp)
        })
    }

    /// Append the UTF-16 code units to `out`.
    pub fn encode_units_into(&self, out: &mut Vec<u16>) {
        out.extend(self.code_units());
    }

    /// Collect the UTF-16 code units.
    pub fn to_units(&self) -> Vec<u16> {
        self.code_units().collect()
    }
}

/// Length in bytes of the UTF-8/WTF-8 sequence starting with lead byte `b`.
/// `b` must be a valid lead byte (callers walk validated WTF-8).
#[inline]
fn utf8_seq_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}

/// Decode the `w`-byte sequence at `pos` to a code point (`u32`), with lone
/// surrogates decoding to `0xD800..=0xDFFF`.
fn decode_at(bytes: &[u8], pos: usize, w: usize) -> u32 {
    let b0 = bytes[pos] as u32;
    match w {
        1 => b0,
        2 => ((b0 & 0x1F) << 6) | (bytes[pos + 1] as u32 & 0x3F),
        3 => {
            let b1 = bytes[pos + 1];
            if b0 == 0xED && (0xA0..=0xBF).contains(&b1) {
                0xD800 | ((b1 as u32 & 0x3F) << 6) | (bytes[pos + 2] as u32 & 0x3F)
            } else {
                ((b0 & 0x0F) << 12) | ((b1 as u32 & 0x3F) << 6) | (bytes[pos + 2] as u32 & 0x3F)
            }
        }
        _ => {
            ((b0 & 0x07) << 18)
                | ((bytes[pos + 1] as u32 & 0x3F) << 12)
                | ((bytes[pos + 2] as u32 & 0x3F) << 6)
                | (bytes[pos + 3] as u32 & 0x3F)
        }
    }
}

/// Validate `b` as WTF-8: UTF-8, plus surrogate code points encoded as
/// `0xED 0xA0..=0xBF 0x80..=0xBF`.
fn validate_wtf8(b: &[u8]) -> bool {
    let mut i = 0;
    while i < b.len() {
        let b0 = b[i];
        let (w, lo, hi) = if b0 < 0x80 {
            i += 1;
            continue;
        } else if (0xC2..0xE0).contains(&b0) {
            (2, 0x80, 0xBF)
        } else if b0 == 0xE0 {
            (3, 0xA0, 0xBF)
        } else if (0xE1..0xED).contains(&b0) {
            (3, 0x80, 0xBF)
        } else if b0 == 0xED {
            // 0x80..=0x9F: U+D000..=U+D7FF; 0xA0..=0xBF: surrogates
            // (allowed in WTF-8).
            (3, 0x80, 0xBF)
        } else if (0xEE..0xF0).contains(&b0) {
            (3, 0x80, 0xBF)
        } else if b0 == 0xF0 {
            (4, 0x90, 0xBF)
        } else if (0xF1..0xF4).contains(&b0) {
            (4, 0x80, 0xBF)
        } else if b0 == 0xF4 {
            (4, 0x80, 0x8F)
        } else {
            return false;
        };
        if i + w > b.len() {
            return false;
        }
        if !(lo..=hi).contains(&b[i + 1]) {
            return false;
        }
        let mut j = 2;
        while j < w {
            if !(0x80..=0xBF).contains(&b[i + j]) {
                return false;
            }
            j += 1;
        }
        i += w;
    }
    true
}

/// Iterator over the UTF-16 code units of a [`Wtf8`] string.
#[derive(Clone)]
pub struct CodeUnits<'a> {
    bytes: &'a [u8],
    pos: usize,
    /// Trail surrogate of an astral char whose lead was already emitted.
    pending: Option<u16>,
}

impl Iterator for CodeUnits<'_> {
    type Item = u16;

    fn next(&mut self) -> Option<u16> {
        if let Some(trail) = self.pending.take() {
            return Some(trail);
        }
        if self.pos >= self.bytes.len() {
            return None;
        }
        let b = self.bytes;
        let p = self.pos;
        let w = utf8_seq_len(b[p]);
        let unit = match w {
            1 => u16::from(b[p]),
            2 => {
                let cp = ((u32::from(b[p]) & 0x1F) << 6) | (u32::from(b[p + 1]) & 0x3F);
                cp as u16
            }
            3 => {
                if b[p] == 0xED && (0xA0..=0xBF).contains(&b[p + 1]) {
                    // Lone surrogate: one code unit.
                    0xD800 | ((u16::from(b[p + 1]) & 0x3F) << 6) | (u16::from(b[p + 2]) & 0x3F)
                } else {
                    let cp = ((u32::from(b[p]) & 0x0F) << 12)
                        | ((u32::from(b[p + 1]) & 0x3F) << 6)
                        | (u32::from(b[p + 2]) & 0x3F);
                    cp as u16
                }
            }
            _ => {
                // Astral: surrogate pair; stash the trail for the next call.
                let cp = ((u32::from(b[p]) & 0x07) << 18)
                    | ((u32::from(b[p + 1]) & 0x3F) << 12)
                    | ((u32::from(b[p + 2]) & 0x3F) << 6)
                    | (u32::from(b[p + 3]) & 0x3F);
                let v = cp - 0x10000;
                let lead = 0xD800 | ((v >> 10) as u16);
                let trail = 0xDC00 | ((v & 0x3FF) as u16);
                // Emit lead now; rewind so the next call emits the trail.
                // Instead of rewinding, store pending trail in `pos` via a
                // sentinel: use the high bit trick — simpler to keep a
                // small state field.
                self.pos = p + w;
                self.pending = Some(trail);
                return Some(lead);
            }
        };
        self.pos = p + w;
        Some(unit)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.bytes.len() - self.pos;
        // Each unit takes at most 3 bytes (lone surrogates, BMP); astral
        // chars take 4 bytes for 2 units.
        (
            remaining / 4,
            Some(remaining + self.pending.map_or(0, |_| 1)),
        )
    }
}

/// Encode one UTF-16 code unit as WTF-8 bytes (surrogates become the
/// 3-byte CESU-8 form).
///
/// Note: this never pairs surrogates; the pairing entry points
/// ([`Wtf8Buf::from_units`], [`Wtf8Buf::push_unit`], [`Wtf8Buf::push_wtf8`])
/// combine an adjacent lead+trail into the astral 4-byte form so the
/// buffer always holds the maximally-paired encoding (which is what makes
/// "well-formed ⟺ valid UTF-8" hold).
pub fn encode_unit(unit: u16, out: &mut Vec<u8>) {
    if unit < 0x80 {
        out.push(unit as u8);
    } else if unit < 0x800 {
        out.push(0xC0 | (unit >> 6) as u8);
        out.push(0x80 | (unit & 0x3F) as u8);
    } else if (0xD800..0xE000).contains(&unit) {
        let v = unit - 0xD800;
        out.push(0xED);
        out.push(0xA0 | (v >> 6) as u8);
        out.push(0x80 | (v & 0x3F) as u8);
    } else {
        out.push(0xE0 | (unit >> 12) as u8);
        out.push(0x80 | ((unit >> 6) & 0x3F) as u8);
        out.push(0x80 | (unit & 0x3F) as u8);
    }
}

/// Combine a lead+trail surrogate pair into the astral code point.
#[inline]
fn pair_astral(lead: u16, trail: u16) -> u32 {
    0x1_0000 + (((lead - 0xD800) as u32) << 10) + ((trail - 0xDC00) as u32)
}

/// Encode an astral code point as 4-byte UTF-8.
fn push_astral(cp: u32, out: &mut Vec<u8>) {
    debug_assert!((0x1_0000..=0x10_FFFF).contains(&cp));
    out.push((0xF0 | (cp >> 18)) as u8);
    out.push((0x80 | ((cp >> 12) & 0x3F)) as u8);
    out.push((0x80 | ((cp >> 6) & 0x3F)) as u8);
    out.push((0x80 | (cp & 0x3F)) as u8);
}

/// Decode the lead surrogate unit from a trailing 3-byte WTF-8 surrogate
/// sequence (`ED A0..AF 80..BF`), or `None` when `buf` does not end with one.
pub(crate) fn trailing_lead_unit(buf: &[u8]) -> Option<u16> {
    let n = buf.len();
    if n >= 3 && buf[n - 3] == 0xED && (0xA0..=0xAF).contains(&buf[n - 2]) {
        let v = (((buf[n - 2] & 0x0F) as u16) << 6) | ((buf[n - 1] & 0x3F) as u16);
        Some(0xD800 + v)
    } else {
        None
    }
}

/// Decode the trail surrogate unit from a leading 3-byte WTF-8 surrogate
/// sequence (`ED B0..BF 80..BF`), or `None` when `bytes` does not start
/// with one.
pub(crate) fn leading_trail_unit(bytes: &[u8]) -> Option<u16> {
    if bytes.len() >= 3 && bytes[0] == 0xED && (0xB0..=0xBF).contains(&bytes[1]) {
        let v = (((bytes[1] & 0x0F) as u16) << 6) | ((bytes[2] & 0x3F) as u16);
        Some(0xDC00 + v)
    } else {
        None
    }
}

/// Concatenate two WTF-8 strings with maximal pairing at the boundary
/// (for the [`Str`] inline path, which cannot go through [`Wtf8Buf`]).
pub(crate) fn concat_bytes(a: &Wtf8, b: &Wtf8) -> Vec<u8> {
    let (ab, bb) = (a.as_bytes(), b.as_bytes());
    let mut out = Vec::with_capacity(ab.len() + bb.len());
    if let (Some(lead), Some(trail)) = (trailing_lead_unit(ab), leading_trail_unit(bb)) {
        out.extend_from_slice(&ab[..ab.len() - 3]);
        push_astral(pair_astral(lead, trail), &mut out);
        out.extend_from_slice(&bb[3..]);
    } else {
        out.extend_from_slice(ab);
        out.extend_from_slice(bb);
    }
    out
}

/// Concatenate two WTF-8 strings with maximal pairing at the boundary,
/// writing the result directly into `dst`; returns the bytes written
/// (`a.len() + b.len()`, or 2 fewer when the boundary folds a
/// lead+trail pair into the astral 4-byte form). The direct-to-arena
/// concat fast path: one copy of the operands, no intermediate buffer.
///
/// # Safety
///
/// * `dst` must be valid for writes of `a.len() + b.len()` bytes minus 2
///   if the boundary folds a lead+trail pair into the astral 4-byte form
///   (i.e., `merged_len` as computed by the caller).
/// * `dst` must not overlap `a` or `b`.
pub(crate) unsafe fn concat_into(a: &Wtf8, b: &Wtf8, dst: *mut u8) -> usize {
    let (ab, bb) = (a.as_bytes(), b.as_bytes());
    // SAFETY: the caller guarantees `dst` has room for both slices and no
    // overlap; the fold below only shrinks the output, so every write
    // stays in-bounds.
    unsafe {
        if let (Some(lead), Some(trail)) = (trailing_lead_unit(ab), leading_trail_unit(bb)) {
            let prefix = ab.len() - 3;
            std::ptr::copy_nonoverlapping(ab.as_ptr(), dst, prefix);
            // `push_astral`'s 4-byte encoding, inlined for the raw
            // destination (`pair_astral` always yields an astral code
            // point for valid surrogate units).
            let cp = pair_astral(lead, trail);
            dst.add(prefix).write((0xF0 | (cp >> 18)) as u8);
            dst.add(prefix + 1)
                .write((0x80 | ((cp >> 12) & 0x3F)) as u8);
            dst.add(prefix + 2).write((0x80 | ((cp >> 6) & 0x3F)) as u8);
            dst.add(prefix + 3).write((0x80 | (cp & 0x3F)) as u8);
            std::ptr::copy_nonoverlapping(bb.as_ptr().add(3), dst.add(prefix + 4), bb.len() - 3);
            ab.len() + bb.len() - 2
        } else {
            std::ptr::copy_nonoverlapping(ab.as_ptr(), dst, ab.len());
            std::ptr::copy_nonoverlapping(bb.as_ptr(), dst.add(ab.len()), bb.len());
            ab.len() + bb.len()
        }
    }
}

/// An owned WTF-8 string buffer. The analogue of `String`.
#[derive(Clone, Default)]
pub struct Wtf8Buf(Vec<u8>);

impl Wtf8Buf {
    /// Empty buffer.
    pub fn new() -> Wtf8Buf {
        Wtf8Buf(Vec::new())
    }

    /// Empty buffer with capacity for `cap` bytes.
    pub fn with_capacity(cap: usize) -> Wtf8Buf {
        Wtf8Buf(Vec::with_capacity(cap))
    }

    /// Wrap bytes known to be valid WTF-8.
    ///
    /// # Safety
    /// `v` must be valid WTF-8.
    pub unsafe fn from_vec_unchecked(v: Vec<u8>) -> Wtf8Buf {
        Wtf8Buf(v)
    }

    /// Validate and wrap, or `None` when invalid.
    pub fn from_vec(v: Vec<u8>) -> Option<Wtf8Buf> {
        if validate_wtf8(&v) {
            Some(Wtf8Buf(v))
        } else {
            None
        }
    }

    /// Build from UTF-16 code units, preserving lone surrogates. The
    /// lossless replacement for `String::from_utf16_lossy`.
    ///
    /// Adjacent lead+trail pairs are encoded as the astral 4-byte UTF-8
    /// sequence (maximal pairing): the buffer always holds the canonical
    /// encoding of its unit sequence, which is what makes "well-formed
    /// ⟺ valid UTF-8" hold.
    pub fn from_units(units: &[u16]) -> Wtf8Buf {
        let mut out = Vec::with_capacity(units.len());
        let mut i = 0;
        while i < units.len() {
            let u = units[i];
            if (0xD800..0xDC00).contains(&u)
                && i + 1 < units.len()
                && (0xDC00..0xE000).contains(&units[i + 1])
            {
                push_astral(pair_astral(u, units[i + 1]), &mut out);
                i += 2;
            } else {
                encode_unit(u, &mut out);
                i += 1;
            }
        }
        // SAFETY: `encode_unit`/`push_astral` emit valid WTF-8 by construction.
        unsafe { Wtf8Buf::from_vec_unchecked(out) }
    }

    /// The borrowed view.
    pub fn as_wtf8(&self) -> &Wtf8 {
        // SAFETY: invariant holds by construction.
        unsafe { Wtf8::from_bytes_unchecked(&self.0) }
    }

    /// Push raw bytes known to be valid WTF-8 (e.g. a subslice of a `&Wtf8`
    /// cut at ASCII boundaries, where multi-byte sequences cannot start).
    pub fn push_wtf8_bytes(&mut self, bytes: &[u8]) {
        debug_assert!(
            Wtf8::from_bytes(bytes).is_some(),
            "push_wtf8_bytes requires valid WTF-8"
        );
        self.0.extend_from_slice(bytes);
    }

    /// Push a `char` (never a lone surrogate: `char` cannot hold one).
    pub fn push_char(&mut self, c: char) {
        let mut tmp = [0u8; 4];
        self.0.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
    }

    /// Push one UTF-16 code unit (lone surrogates preserved). When the
    /// unit is a trail surrogate and the buffer ends with a lead-surrogate
    /// encoding, the two are folded into the astral 4-byte form, keeping
    /// the maximal-pairing invariant.
    pub fn push_unit(&mut self, unit: u16) {
        if (0xDC00..0xE000).contains(&unit)
            && let Some(lead) = trailing_lead_unit(&self.0)
        {
            self.0.truncate(self.0.len() - 3);
            push_astral(pair_astral(lead, unit), &mut self.0);
            return;
        }
        encode_unit(unit, &mut self.0);
    }

    /// Push code units (maximal pairing, as in [`Wtf8Buf::from_units`]).
    pub fn extend_units(&mut self, units: &[u16]) {
        self.0.reserve(units.len());
        let mut i = 0;
        // A trail first: it may pair with a lead already in the buffer.
        if !units.is_empty()
            && (0xDC00..0xE000).contains(&units[0])
            && let Some(lead) = trailing_lead_unit(&self.0)
        {
            self.0.truncate(self.0.len() - 3);
            push_astral(pair_astral(lead, units[0]), &mut self.0);
            i = 1;
        }
        while i < units.len() {
            let u = units[i];
            if (0xD800..0xDC00).contains(&u)
                && i + 1 < units.len()
                && (0xDC00..0xE000).contains(&units[i + 1])
            {
                push_astral(pair_astral(u, units[i + 1]), &mut self.0);
                i += 2;
            } else {
                encode_unit(u, &mut self.0);
                i += 1;
            }
        }
    }

    /// Push WTF-8 content. When the buffer ends with a lead-surrogate
    /// encoding and `s` starts with a trail-surrogate encoding, the pair
    /// is folded into the astral 4-byte form (maximal pairing).
    pub fn push_wtf8(&mut self, s: &Wtf8) {
        let sb = s.as_bytes();
        if let (Some(lead), Some(trail)) = (trailing_lead_unit(&self.0), leading_trail_unit(sb)) {
            self.0.truncate(self.0.len() - 3);
            push_astral(pair_astral(lead, trail), &mut self.0);
            self.0.extend_from_slice(&sb[3..]);
        } else {
            self.0.extend_from_slice(sb);
        }
    }

    /// Push a `&str` (valid UTF-8 is valid WTF-8).
    pub fn push_str(&mut self, s: &str) {
        self.0.extend_from_slice(s.as_bytes());
    }

    /// Byte length.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Consume into the byte vector.
    pub fn into_vec(self) -> Vec<u8> {
        self.0
    }

    /// Raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl From<&Wtf8> for Wtf8Buf {
    fn from(s: &Wtf8) -> Wtf8Buf {
        Wtf8Buf(s.as_bytes().to_vec())
    }
}

impl From<&str> for Wtf8Buf {
    fn from(s: &str) -> Wtf8Buf {
        Wtf8Buf(s.as_bytes().to_vec())
    }
}

impl From<&String> for Wtf8Buf {
    fn from(s: &String) -> Wtf8Buf {
        Wtf8Buf(s.as_bytes().to_vec())
    }
}

impl std::ops::Deref for Wtf8Buf {
    type Target = Wtf8;
    fn deref(&self) -> &Wtf8 {
        self.as_wtf8()
    }
}

impl std::borrow::Borrow<Wtf8> for Wtf8Buf {
    fn borrow(&self) -> &Wtf8 {
        self.as_wtf8()
    }
}

impl std::borrow::ToOwned for Wtf8 {
    type Owned = Wtf8Buf;
    fn to_owned(&self) -> Wtf8Buf {
        Wtf8Buf(self.0.to_vec())
    }
}

impl PartialEq for Wtf8Buf {
    fn eq(&self, other: &Wtf8Buf) -> bool {
        self.as_wtf8() == other.as_wtf8()
    }
}

impl Eq for Wtf8Buf {}

impl std::hash::Hash for Wtf8Buf {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_wtf8().hash(state);
    }
}

impl PartialEq for Wtf8 {
    fn eq(&self, other: &Wtf8) -> bool {
        self.0 == other.0
    }
}

impl Eq for Wtf8 {}

// Byte-wise lexicographic order coincides with UTF-16 code-unit order
// (the JS `<` semantics): ASCII sorts first; the 3-byte surrogate
// sequences `0xED 0xA0..=0xBF …` sit between `U+D7FF` (`0xED 0x9F 0xBF`)
// and `U+E000` (`0xEE 0x80 0x80`), exactly where the code units
// `0xD800..=0xDFFF` sit; astral 4-byte sequences sort after all BMP
// content. A lone surrogate is a proper prefix of any pair starting
// with the same half, matching the shorter-is-less rule.
impl PartialOrd for Wtf8 {
    fn partial_cmp(&self, other: &Wtf8) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Wtf8 {
    fn cmp(&self, other: &Wtf8) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for Wtf8Buf {
    fn partial_cmp(&self, other: &Wtf8Buf) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Wtf8Buf {
    fn cmp(&self, other: &Wtf8Buf) -> std::cmp::Ordering {
        self.as_wtf8().cmp(other.as_wtf8())
    }
}

impl std::hash::Hash for Wtf8 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// Byte equality against a `&str`: UTF-8 is a subset of WTF-8, so this is
/// exact (a WTF-8 string with lone surrogates never equals a `&str`).
impl PartialEq<str> for Wtf8 {
    fn eq(&self, other: &str) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl<'a> PartialEq<&'a str> for Wtf8 {
    fn eq(&self, other: &&'a str) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

// Covered by the blanket `impl<A: PartialEq<B>> PartialEq<&B> for &A`:
// `&Wtf8 == &str` works through `Wtf8: PartialEq<str>`.

impl PartialEq<Wtf8> for str {
    fn eq(&self, other: &Wtf8) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl PartialEq<Wtf8Buf> for Wtf8 {
    fn eq(&self, other: &Wtf8Buf) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl PartialEq<String> for Wtf8 {
    fn eq(&self, other: &String) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl PartialEq<Wtf8> for String {
    fn eq(&self, other: &Wtf8) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl PartialEq<Wtf8> for Wtf8Buf {
    fn eq(&self, other: &Wtf8) -> bool {
        self.as_wtf8() == other
    }
}

impl<'a> PartialEq<&'a str> for Wtf8Buf {
    fn eq(&self, other: &&'a str) -> bool {
        self.as_wtf8() == *other
    }
}

impl PartialEq<str> for Wtf8Buf {
    fn eq(&self, other: &str) -> bool {
        self.as_wtf8() == other
    }
}

impl fmt::Debug for Wtf8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Quoted like `str`'s Debug; lone surrogates escape as `\u{D800}`.
        f.write_str("\"")?;
        for cp in self.chars_wtf8() {
            if (0xD800..0xE000).contains(&cp) {
                write!(f, "\\u{{{:X}}}", cp)?;
            } else {
                // SAFETY: non-surrogate code points from validated WTF-8
                // are valid chars.
                let c = unsafe { char::from_u32_unchecked(cp) };
                // Quoted like `str`'s Debug: `escape_debug` also escapes
                // `'`, which `str` leaves alone.
                if c == '\'' {
                    f.write_str("'")?;
                } else {
                    for e in c.escape_debug() {
                        f.write_str(e.encode_utf8(&mut [0u8; 4]))?;
                    }
                }
            }
        }
        f.write_str("\"")
    }
}

impl fmt::Debug for Wtf8Buf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_wtf8(), f)
    }
}

impl fmt::Display for Wtf8 {
    /// Lossy: lone surrogates render as U+FFFD.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_utf8_lossy())
    }
}

impl fmt::Display for Wtf8Buf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_wtf8(), f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(s: &str) -> &Wtf8 {
        Wtf8::from_str(s)
    }

    #[test]
    fn ascii_round_trip() {
        let s = w("hello");
        assert_eq!(s.units_len(), 5);
        assert_eq!(s.to_units(), vec![104, 101, 108, 108, 111]);
        assert!(s.is_well_formed());
        assert_eq!(s.as_utf8(), Some("hello"));
        assert_eq!(&*s.slice_units(1, 4), w("ell"));
        assert_eq!(&*s.slice_units(0, 99), w("hello"));
        assert_eq!(s.slice_units(4, 2).units_len(), 0);
    }

    #[test]
    fn lone_surrogate_storage() {
        let buf = Wtf8Buf::from_units(&[0xD800]);
        let s = buf.as_wtf8();
        assert_eq!(s.as_bytes(), &[0xED, 0xA0, 0x80]);
        assert_eq!(s.units_len(), 1);
        assert_eq!(s.to_units(), vec![0xD800]);
        assert!(!s.is_well_formed());
        assert_eq!(s.as_utf8(), None);
        assert_eq!(s.to_utf8_lossy(), "\u{FFFD}");
        // Round-trips through units.
        let back = Wtf8Buf::from_units(&s.to_units());
        assert_eq!(back.as_wtf8(), s);
    }

    #[test]
    fn surrogate_pair_units() {
        // U+1F600 = D83D DE00.
        let buf = Wtf8Buf::from_units(&[0xD83D, 0xDE00]);
        let s = buf.as_wtf8();
        assert_eq!(s.units_len(), 2);
        assert_eq!(s.to_units(), vec![0xD83D, 0xDE00]);
        assert!(s.is_well_formed());
        assert_eq!(s.as_utf8(), Some("\u{1F600}"));
        assert_eq!(s.slice_units(0, 1).to_units(), vec![0xD83D]);
        assert_eq!(s.slice_units(1, 2).to_units(), vec![0xDE00]);
        assert_eq!(&*s.slice_units(0, 2), w("\u{1F600}"));
        // Splitting the pair materializes (owned); aligned borrows.
        assert!(matches!(s.slice_units(1, 2), Cow::Owned(_)));
        assert!(matches!(s.slice_units(0, 2), Cow::Borrowed(_)));
    }

    #[test]
    fn mixed_content() {
        let buf = Wtf8Buf::from_units(&[0x61, 0xD800, 0x62, 0xD83D, 0xDE00, 0x63]);
        let s = buf.as_wtf8();
        assert_eq!(s.units_len(), 6);
        assert_eq!(s.to_units(), vec![0x61, 0xD800, 0x62, 0xD83D, 0xDE00, 0x63]);
        assert!(!s.is_well_formed());
        assert_eq!(s.nth_unit(0), Some(0x61));
        assert_eq!(s.nth_unit(1), Some(0xD800));
        assert_eq!(s.nth_unit(5), Some(0x63));
        assert_eq!(s.nth_unit(6), None);
        assert_eq!(s.slice_units(1, 3).to_units(), vec![0xD800, 0x62]);
        assert_eq!(s.slice_units(0, 0).units_len(), 0);
    }

    #[test]
    fn to_well_formed_replaces_lone_surrogates() {
        // 0xD800 and 0xDC00 separated: both lone, both replaced. (Adjacent
        // D800 DC00 would be a valid pair and survive.)
        let buf = Wtf8Buf::from_units(&[0x61, 0xD800, 0x62, 0xDC00]);
        let s = buf.as_wtf8();
        let fixed = s.to_well_formed();
        assert_eq!(fixed.to_units(), vec![0x61, 0xFFFD, 0x62, 0xFFFD]);
        assert!(fixed.is_well_formed());
        // Already well-formed borrows.
        let ok = w("abc");
        assert!(matches!(ok.to_well_formed(), Cow::Borrowed(_)));
    }

    #[test]
    fn validation() {
        assert!(Wtf8::from_bytes(b"abc").is_some());
        assert!(Wtf8::from_bytes(&[0xED, 0xA0, 0x80]).is_some()); // lone surrogate: ok
        assert!(Wtf8::from_bytes(&[0xED, 0x9F, 0xBF]).is_some()); // U+D7FF: ok
        assert!(Wtf8::from_bytes(&[0xC0, 0x80]).is_none()); // overlong
        assert!(Wtf8::from_bytes(&[0xED, 0xA0]).is_none()); // truncated
        assert!(Wtf8::from_bytes(&[0x80]).is_none()); // stray continuation
        assert!(Wtf8::from_bytes(&[0xF5, 0x80, 0x80, 0x80]).is_none()); // > U+10FFFF
        assert!(Wtf8::from_bytes("😀".as_bytes()).is_some());
    }

    #[test]
    fn str_equality() {
        assert_eq!(w("abc"), "abc");
        assert_ne!(w("abc"), "abd");
        let lone = Wtf8Buf::from_units(&[0xD800]);
        assert_ne!(lone.as_wtf8(), "\u{FFFD}");
    }

    #[test]
    fn debug_format() {
        let buf = Wtf8Buf::from_units(&[0x61, 0xD800]);
        assert_eq!(format!("{:?}", buf.as_wtf8()), "\"a\\u{D800}\"");
        assert_eq!(format!("{:?}", w("a\"b")), "\"a\\\"b\"");
    }
}
