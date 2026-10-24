//! Image decoding: PNG, JPEG, and GIF to RGBA8.
//!
//! [`decode_image`] dispatches on magic bytes and always returns pixels as
//! RGBA8, row-major, top-to-bottom.
//!
//! Format notes:
//! - PNG: decoded with `EXPAND | STRIP_16 | ALPHA`, so palette, low-bit
//!   gray, 16-bit, and tRNS images all normalize to 8-bit with alpha.
//!   Interlaced (Adam7) PNGs are deinterlaced by the `png` crate.
//! - Animated PNG: only the first frame is decoded; later frames are
//!   ignored. Animation support is a future subsystem.
//! - JPEG: baseline and progressive decode via `zune-jpeg`. CMYK/YCCK
//!   JPEGs are rejected with a clean [`ImageError::DecodeFailed`]; the
//!   decoder has no CMYK path and silently producing wrong colors would be
//!   worse than an error.
//! - GIF: GIF87a/GIF89a via a hand-rolled std-only decoder (no new
//!   dependency): Logical Screen Descriptor, global/local color tables,
//!   Graphic Control Extension transparency, GIF-variant LZW, and 4-pass
//!   deinterlacing. Only the first frame is decoded; animation support is
//!   a future subsystem. Frames are clipped to the logical screen.
//! - WebP: NOT supported ([`ImageError::UnknownFormat`]). Rationale:
//!   lossless WebP (VP8L) needs Huffman + LZ77 + color-cache + spatial
//!   transforms, and lossy WebP needs a full VP8 bitstream decoder (DCT,
//!   loop filter, boolean entropy coder) — thousands of lines with real
//!   misdecode risk, not a std-trivial codec. Depending on a WebP crate
//!   would violate the std-first rule for a format that is nice-to-have
//!   in v1; deferred until animated-image support is designed, at which
//!   point the dependency question can be weighed with orchestrator
//!   approval.
//! - Images with either dimension above [`MAX_IMAGE_DIMENSION`] are
//!   rejected with [`ImageError::TooLarge`] before any large allocation.
//!   PNG and JPEG are additionally capped at 256 MiB of *post-expansion
//!   RGBA* (not raw decoder output), and GIF output at 64 MiB, so a header
//!   claiming 16384x16384 is rejected instead of triggering a ~1 GiB
//!   allocation before decode fails. Decodes expand to RGBA in place in
//!   the single capped buffer, so peak memory stays at one capped buffer
//!   rather than a raw + RGBA double buffer.

use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

/// Maximum allowed image width or height, in pixels.
///
/// Checked from the image header before decoding, so no large buffer is
/// ever allocated for an oversized image.
pub const MAX_IMAGE_DIMENSION: u32 = 16384;

/// Maximum bytes a PNG decode may allocate, covering both the `png`
/// crate's internal buffers and the RGBA output buffer we allocate.
///
/// `png::Limits` only bounds the crate's *internal* allocations; the
/// output buffer is allocated by us, so its *post-expansion RGBA* size
/// (`w*h*4`) is checked against this cap *before* allocating — the raw
/// frame then decodes into the head of that same buffer and is expanded
/// in place, so the peak stays at one capped buffer. A header claiming
/// 16384x16384 (1 GiB of RGBA) passes the dimension check but is rejected
/// at that check instead of triggering a 1 GiB `vec` before the (absent)
/// image data fails to decode. 256 MiB covers 8192x8192 RGBA, far above
/// any image this renderer will realistically decode.
const PNG_MAX_OUTPUT_BYTES: usize = 256 * 1024 * 1024;

/// Maximum bytes a JPEG decode may allocate for its post-expansion RGBA
/// image: the same 256 MiB cap as PNG. The decoder allocates `w*h*3` for
/// RGB output inside `decode()`, but the caller ends up holding `w*h*4`
/// after RGBA expansion, so the cap is enforced against `w*h*4` — a header
/// claiming 16384x16384 (1 GiB of RGBA) passes the dimension check but is
/// rejected at the checked multiplication below instead of triggering the
/// allocation. Expansion happens in place in the decoded buffer, so the
/// peak stays at one capped buffer.
const JPEG_MAX_OUTPUT_BYTES: usize = 256 * 1024 * 1024;

/// A decoded image: RGBA8 pixels, row-major, top-to-bottom.
///
/// NOTE (deduplicated 2026-09-25 by the display-list worker):
/// `DecodedImage` and `ImageId` are canonically defined in
/// `crate::display_list` (this module's original NOTE anticipated exactly
/// this); they are re-exported here so `decode_image` callers keep a
/// single path to the types.
pub use crate::display_list::{DecodedImage, ImageId};

/// Errors from [`decode_image`].
#[derive(Debug)]
pub enum ImageError {
    /// The input did not match any supported image signature.
    UnknownFormat,
    /// The input looked like a supported format but could not be decoded
    /// (truncated data, corrupt chunks, unsupported color space, ...).
    DecodeFailed(String),
    /// Either dimension exceeds [`MAX_IMAGE_DIMENSION`], or the
    /// decoded output would exceed the allocation cap (256 MiB for
    /// PNG/JPEG, 64 MiB for GIF).
    TooLarge,
}

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageError::UnknownFormat => write!(f, "unknown image format"),
            ImageError::DecodeFailed(msg) => write!(f, "image decode failed: {msg}"),
            ImageError::TooLarge => {
                write!(f, "image dimensions exceed {MAX_IMAGE_DIMENSION}px")
            }
        }
    }
}

impl std::error::Error for ImageError {}

const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
const JPEG_SOI: [u8; 2] = [0xFF, 0xD8];
const GIF_MAGIC_87A: [u8; 6] = *b"GIF87a";
const GIF_MAGIC_89A: [u8; 6] = *b"GIF89a";

/// Decode `bytes` to RGBA8, dispatching on magic bytes.
///
/// Returns [`ImageError::UnknownFormat`] when no known signature matches;
/// never panics on malformed input.
pub fn decode_image(bytes: &[u8]) -> Result<DecodedImage, ImageError> {
    if bytes.len() >= PNG_SIGNATURE.len() && bytes[..8] == PNG_SIGNATURE {
        decode_png(bytes)
    } else if bytes.len() >= JPEG_SOI.len() && bytes[..2] == JPEG_SOI {
        decode_jpeg(bytes)
    } else if bytes.len() >= GIF_MAGIC_89A.len()
        && (bytes[..6] == GIF_MAGIC_87A || bytes[..6] == GIF_MAGIC_89A)
    {
        decode_gif(bytes)
    } else {
        Err(ImageError::UnknownFormat)
    }
}

fn decode_failed<E: std::fmt::Display>(e: E) -> ImageError {
    ImageError::DecodeFailed(e.to_string())
}

fn decode_png(bytes: &[u8]) -> Result<DecodedImage, ImageError> {
    // The `png` decoder needs `Read`; Cursor<&[u8]> provides it with no copy.
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    // Restore a sane allocation limit instead of disabling the crate's
    // 64 MiB bomb protection: this bounds the decoder's internal buffers
    // (IDAT accumulation, iCCP inflation, ...). The output buffer we
    // allocate below is checked against the same cap before allocating.
    decoder.set_limits(png::Limits {
        bytes: PNG_MAX_OUTPUT_BYTES,
    });
    decoder.set_transformations(
        png::Transformations::normalize_to_color8() | png::Transformations::ALPHA,
    );
    let mut reader = decoder.read_info().map_err(decode_failed)?;

    let (ihdr_w, ihdr_h) = (reader.info().width, reader.info().height);
    if ihdr_w > MAX_IMAGE_DIMENSION || ihdr_h > MAX_IMAGE_DIMENSION {
        return Err(ImageError::TooLarge);
    }

    // Post-expansion cap, enforced before any allocation: the final RGBA
    // image is `w*h*4` bytes, and the raw frame decodes into the head of
    // that same buffer and is expanded in place, so this one check bounds
    // both the allocation and the peak memory — there is no transient
    // raw+RGBA double buffer. (Capping the raw output alone would not do:
    // a 16384x16384 RGB PNG's raw output is 768 MiB but its RGBA expansion
    // is 1 GiB.)
    let rgba_len = (ihdr_w as usize)
        .checked_mul(ihdr_h as usize)
        .and_then(|n| n.checked_mul(4))
        .filter(|&n| n <= PNG_MAX_OUTPUT_BYTES)
        .ok_or(ImageError::TooLarge)?;
    // png 0.17's `output_buffer_size` is infallible (`usize`); the
    // post-expansion cap above bounds it too, since post-transformation
    // output is at most 4 bytes/px — re-check defensively so a decoder
    // that disagrees with the header about the output shape cannot make
    // the in-place expansion overrun.
    let size = reader.output_buffer_size();
    if size > rgba_len {
        return Err(ImageError::DecodeFailed(
            "PNG output size exceeds RGBA bound".into(),
        ));
    }

    // Single buffer: decode the raw frame into the head, then expand to
    // RGBA in place (`next_frame` accepts a larger-than-needed buffer).
    let mut buf = vec![0u8; rgba_len];
    // First frame only: for APNG this decodes frame 0 and stops.
    let out = reader.next_frame(&mut buf[..size]).map_err(decode_failed)?;
    // Authoritative decoded byte count (equals `size` for well-formed
    // input; `next_frame` guarantees the frame fits).
    let decoded = out.buffer_size().min(buf.len());

    // APNG frame 0 is IHDR-sized in practice; if a lying fcTL disagrees,
    // expand in the same buffer rather than allocating a second RGBA
    // vector (the raw frame sits at the head from `next_frame`).
    let rgba = if out.width == ihdr_w && out.height == ihdr_h {
        expand_png_to_rgba_in_place(&mut buf, decoded, out.color_type, out.width, out.height)?;
        buf
    } else {
        // Cap the fallback's post-expansion size like the common path.
        let fallback_len = (out.width as usize)
            .checked_mul(out.height as usize)
            .and_then(|n| n.checked_mul(4))
            .filter(|&n| n <= PNG_MAX_OUTPUT_BYTES)
            .ok_or(ImageError::TooLarge)?;
        if buf.len() != fallback_len {
            // `resize` may reallocate (transient old+new), but the result
            // is still one capped buffer — never a separate raw+RGBA pair.
            buf.resize(fallback_len, 0);
        }
        expand_png_to_rgba_in_place(&mut buf, decoded, out.color_type, out.width, out.height)?;
        buf
    };
    Ok(DecodedImage {
        width: out.width,
        height: out.height,
        rgba,
    })
}

/// Expand an 8-bit PNG frame (post-transformations) to RGBA8 in place.
///
/// `buf` is a `width*height*4` buffer whose first `raw_size` bytes hold
/// the decoded frame; expansion runs back-to-front — each destination
/// pixel sits at a higher address than its source (`i*4 > i*channels`),
/// so no source pixel is overwritten before it is read. RGBA frames are
/// already expanded and need no work, but their `raw_size` is still
/// validated against `width*height*4`.
fn expand_png_to_rgba_in_place(
    buf: &mut [u8],
    raw_size: usize,
    color_type: png::ColorType,
    width: u32,
    height: u32,
) -> Result<(), ImageError> {
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| ImageError::DecodeFailed("PNG dimensions overflow".into()))?;
    let rgba_len = pixel_count
        .checked_mul(4)
        .ok_or_else(|| ImageError::DecodeFailed("PNG dimensions overflow".into()))?;
    // The RGBA path validates too: the decoder's byte count is
    // authoritative, and a mismatch would hand a mis-sized buffer
    // downstream as "RGBA".
    if color_type == png::ColorType::Rgba {
        if raw_size != rgba_len || buf.len() < rgba_len {
            return Err(ImageError::DecodeFailed(format!(
                "PNG data size mismatch: got {raw_size}, expected {rgba_len}"
            )));
        }
        return Ok(());
    }
    let channels: usize = match color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Indexed => {
            // Unreachable: EXPAND converts indexed to RGB/RGBA.
            return Err(ImageError::DecodeFailed(
                "unexpected indexed PNG after expansion".into(),
            ));
        }
        png::ColorType::Rgba => unreachable!("handled above"),
    };
    let expected = pixel_count
        .checked_mul(channels)
        .ok_or_else(|| ImageError::DecodeFailed("PNG dimensions overflow".into()))?;
    if raw_size != expected || buf.len() < rgba_len {
        return Err(ImageError::DecodeFailed(format!(
            "PNG data size mismatch: got {raw_size}, expected {expected}"
        )));
    }
    for i in (0..pixel_count).rev() {
        let (s, d) = (i * channels, i * 4);
        // Copy the source bytes out first: `u8` is `Copy`, so the
        // borrow ends before the destination slice is written.
        match channels {
            1 => {
                let g = buf[s];
                buf[d..d + 4].copy_from_slice(&[g, g, g, 255]);
            }
            2 => {
                let (g, a) = (buf[s], buf[s + 1]);
                buf[d..d + 4].copy_from_slice(&[g, g, g, a]);
            }
            _ => {
                let (r, g, b) = (buf[s], buf[s + 1], buf[s + 2]);
                buf[d..d + 4].copy_from_slice(&[r, g, b, 255]);
            }
        }
    }
    Ok(())
}

/// Scan JPEG markers for the first Start-Of-Frame segment and return its
/// `(width, height)`, or `None` if no SOF appears before SOS / EOI /
/// truncation.
///
/// This is a header-only pre-check so oversized images are rejected with
/// [`ImageError::TooLarge`] deterministically, without depending on
/// `zune-jpeg`'s error wording for its own dimension limit.
fn jpeg_sof_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    // Caller guarantees the SOI signature; start past it.
    let mut pos = 2usize;
    while pos < bytes.len() {
        if bytes[pos] != 0xFF {
            return None;
        }
        // Skip fill bytes; the marker is the first non-0xFF byte.
        let mut m = pos + 1;
        while m < bytes.len() && bytes[m] == 0xFF {
            m += 1;
        }
        if m >= bytes.len() {
            return None;
        }
        let marker = bytes[m];
        pos = m + 1;
        // Standalone markers carry no length field.
        if marker == 0xD8 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            continue;
        }
        if marker == 0xD9 {
            return None; // EOI: no SOF found.
        }
        if pos + 2 > bytes.len() {
            return None;
        }
        // Segment length includes its own 2 bytes.
        let seg_len = u16::from_be_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        if seg_len < 2 || pos + seg_len > bytes.len() {
            return None;
        }
        // SOF0-SOF15, excluding DHT (0xC4), JPG (0xC8), DAC (0xCC).
        let is_sof = (0xC0..=0xCF).contains(&marker) && !matches!(marker, 0xC4 | 0xC8 | 0xCC);
        if is_sof {
            // SOF body: precision:1, height:2, width:2, components:1, ...
            if seg_len < 8 {
                return None;
            }
            let height = u16::from_be_bytes([bytes[pos + 3], bytes[pos + 4]]) as u32;
            let width = u16::from_be_bytes([bytes[pos + 5], bytes[pos + 6]]) as u32;
            return Some((width, height));
        }
        if marker == 0xDA {
            return None; // SOS: entropy-coded data follows, no more headers.
        }
        pos += seg_len;
    }
    None
}

/// True when a `w`x`h` JPEG's post-expansion RGBA image (`w*h*4`, the
/// buffer the caller ends up holding) would exceed the allocation cap.
/// The decoder's own raw output (`w*h*3`) is smaller, so capping the raw
/// output alone would still allow a 341 MiB RGBA image plus a 256 MiB
/// transient. Checked arithmetic: a product that overflows `usize` counts
/// as exceeding the cap.
fn jpeg_rgba_exceeds_cap(w: u32, h: u32) -> bool {
    (w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(4))
        .is_none_or(|n| n > JPEG_MAX_OUTPUT_BYTES)
}

fn decode_jpeg(bytes: &[u8]) -> Result<DecodedImage, ImageError> {
    // Reject oversized headers with our own TooLarge before the decoder
    // runs, so the rejection never depends on zune-jpeg's error wording.
    if let Some((w, h)) = jpeg_sof_dimensions(bytes)
        && (w > MAX_IMAGE_DIMENSION || h > MAX_IMAGE_DIMENSION || jpeg_rgba_exceeds_cap(w, h))
    {
        return Err(ImageError::TooLarge);
    }

    let options = DecoderOptions::default()
        .set_max_width(MAX_IMAGE_DIMENSION as usize)
        .set_max_height(MAX_IMAGE_DIMENSION as usize);
    let mut dec = JpegDecoder::new_with_options(bytes, options);
    dec.decode_headers().map_err(decode_failed)?;

    let info = dec
        .info()
        .ok_or_else(|| ImageError::DecodeFailed("JPEG headers missing after decode".into()))?;
    let (w, h) = (u32::from(info.width), u32::from(info.height));
    if w == 0 || h == 0 {
        return Err(ImageError::DecodeFailed("JPEG has zero dimension".into()));
    }
    if w > MAX_IMAGE_DIMENSION || h > MAX_IMAGE_DIMENSION {
        return Err(ImageError::TooLarge);
    }
    // The header pre-scan may miss the SOF (e.g. truncated input), so
    // check the post-expansion cap here too, before `decode()` allocates.
    if jpeg_rgba_exceeds_cap(w, h) {
        return Err(ImageError::TooLarge);
    }

    // CMYK/YCCK have no correct decode path in zune-jpeg; reject cleanly
    // instead of producing wrong colors.
    if matches!(
        dec.get_input_colorspace(),
        Some(ColorSpace::CMYK) | Some(ColorSpace::YCCK)
    ) {
        return Err(ImageError::DecodeFailed(
            "CMYK JPEG images are not supported".into(),
        ));
    }

    let raw = dec.decode().map_err(decode_failed)?;
    // Expand to RGBA in the decoded buffer itself: `w*h*4` already passed
    // the cap above, so growing this one buffer avoids a second permanent
    // RGBA allocation. (`Vec::resize` may reallocate and copy, so the
    // transient peak can briefly hold old+new — the guarantee is one
    // capped buffer retained, never a raw + RGBA double buffer.)
    let pixel_count = (w as usize)
        .checked_mul(h as usize)
        .ok_or_else(|| ImageError::DecodeFailed("JPEG dimensions overflow".into()))?;
    let rgba_len = pixel_count
        .checked_mul(4)
        .filter(|&n| n <= JPEG_MAX_OUTPUT_BYTES)
        .ok_or(ImageError::TooLarge)?;
    let mut rgba = raw;
    let rgba = match dec.get_output_colorspace() {
        Some(ColorSpace::RGB) => {
            let expected = pixel_count
                .checked_mul(3)
                .ok_or_else(|| ImageError::DecodeFailed("JPEG dimensions overflow".into()))?;
            if rgba.len() != expected {
                return Err(ImageError::DecodeFailed(format!(
                    "JPEG RGB size mismatch: got {}, expected {expected}",
                    rgba.len()
                )));
            }
            rgba.resize(rgba_len, 0);
            expand_rgb_to_rgba_in_place(&mut rgba, pixel_count);
            rgba
        }
        Some(ColorSpace::RGBA) => {
            if rgba.len() != rgba_len {
                return Err(ImageError::DecodeFailed(format!(
                    "JPEG RGBA size mismatch: got {}, expected {rgba_len}",
                    rgba.len()
                )));
            }
            rgba
        }
        Some(ColorSpace::Luma) => {
            if rgba.len() != pixel_count {
                return Err(ImageError::DecodeFailed(format!(
                    "JPEG gray size mismatch: got {}, expected {pixel_count}",
                    rgba.len()
                )));
            }
            rgba.resize(rgba_len, 0);
            expand_gray_to_rgba_in_place(&mut rgba, pixel_count);
            rgba
        }
        other => {
            return Err(ImageError::DecodeFailed(format!(
                "unsupported JPEG output colorspace: {other:?}"
            )));
        }
    };
    Ok(DecodedImage {
        width: w,
        height: h,
        rgba,
    })
}

/// Expand `pixel_count` RGB triples to RGBA in place: `buf` holds the RGB
/// bytes followed by zero padding out to `pixel_count*4`. Back-to-front,
/// so no source pixel is overwritten before it is read.
fn expand_rgb_to_rgba_in_place(buf: &mut [u8], pixel_count: usize) {
    debug_assert!(buf.len() >= pixel_count * 4);
    for i in (0..pixel_count).rev() {
        let (s, d) = (i * 3, i * 4);
        let (r, g, b) = (buf[s], buf[s + 1], buf[s + 2]);
        buf[d..d + 4].copy_from_slice(&[r, g, b, 255]);
    }
}

/// Expand `pixel_count` gray bytes to RGBA in place: `buf` holds the gray
/// bytes followed by zero padding out to `pixel_count*4`. Back-to-front.
fn expand_gray_to_rgba_in_place(buf: &mut [u8], pixel_count: usize) {
    debug_assert!(buf.len() >= pixel_count * 4);
    for i in (0..pixel_count).rev() {
        let g = buf[i];
        let d = i * 4;
        buf[d..d + 4].copy_from_slice(&[g, g, g, 255]);
    }
}

// ---------------------------------------------------------------------------
// GIF
// ---------------------------------------------------------------------------

/// Maximum bytes a GIF decode may allocate for its RGBA canvas: 64 MiB,
/// i.e. 16_777_216 pixels at 4 bytes per pixel. Checked from the Logical
/// Screen Descriptor before the canvas is allocated, so a header claiming
/// 16384x16384 (1 GiB of RGBA) is rejected here instead of triggering a
/// 1 GiB `vec`. Per-frame checks reuse this bound in pixel terms as
/// `GIF_MAX_OUTPUT_BYTES / 4`.
const GIF_MAX_OUTPUT_BYTES: usize = 64 * 1024 * 1024;

/// Cap on buffered GIF sub-block data (LZW image data). Pathological
/// input cannot force a larger buffer before the pixel-count cap stops
/// the decode; 64 MiB of compressed data is far beyond any legitimate
/// image that fits the output cap.
const GIF_MAX_SUB_BLOCK_BYTES: usize = 64 * 1024 * 1024;

/// Decode the first frame of a GIF87a/GIF89a image to RGBA8.
///
/// Hand-rolled std-only decoder: Logical Screen Descriptor, global/local
/// color tables, Graphic Control Extension transparency, GIF-variant LZW,
/// and 4-pass deinterlacing. Later frames are ignored (animation is a
/// future subsystem), so disposal methods are irrelevant; the frame is
/// composited onto the screen-sized canvas, clipped to the logical
/// screen. Never panics on malformed input.
fn decode_gif(bytes: &[u8]) -> Result<DecodedImage, ImageError> {
    // The dispatcher matched the 6-byte signature; the Logical Screen
    // Descriptor needs 7 more bytes.
    if bytes.len() < 13 {
        return Err(ImageError::DecodeFailed("truncated GIF header".into()));
    }
    let width = u16::from_le_bytes([bytes[6], bytes[7]]) as u32;
    let height = u16::from_le_bytes([bytes[8], bytes[9]]) as u32;
    if width == 0 || height == 0 {
        return Err(ImageError::DecodeFailed("GIF has zero dimension".into()));
    }
    if width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION {
        return Err(ImageError::TooLarge);
    }
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| ImageError::DecodeFailed("GIF dimensions overflow".into()))?;
    if pixel_count
        .checked_mul(4)
        .is_none_or(|n| n > GIF_MAX_OUTPUT_BYTES)
    {
        return Err(ImageError::TooLarge);
    }

    let packed = bytes[10];
    let bg_index = bytes[11] as usize;
    let mut pos = 13usize;

    let mut gct: &[u8] = &[];
    if packed & 0x80 != 0 {
        // 2^((packed & 7) + 1) entries of 3 bytes; at most 768 bytes.
        let table_len = 3usize << ((packed & 0x07) + 1);
        let end = pos + table_len;
        if end > bytes.len() {
            return Err(ImageError::DecodeFailed(
                "truncated GIF global color table".into(),
            ));
        }
        gct = &bytes[pos..end];
        pos = end;
    }

    // Canvas, pre-filled with the background color when the GCT names a
    // valid one; transparent black otherwise.
    let mut canvas = vec![0u8; pixel_count * 4];
    if bg_index * 3 + 2 < gct.len() {
        let (r, g, b) = (
            gct[bg_index * 3],
            gct[bg_index * 3 + 1],
            gct[bg_index * 3 + 2],
        );
        for px in canvas.as_chunks_mut::<4>().0 {
            px[0] = r;
            px[1] = g;
            px[2] = b;
            px[3] = 255;
        }
    }

    // Graphic Control Extension state applying to the next image block.
    let mut transparent: Option<u8> = None;
    loop {
        let block = bytes
            .get(pos)
            .copied()
            .ok_or_else(|| ImageError::DecodeFailed("truncated GIF block".into()))?;
        pos += 1;
        match block {
            // Extension block.
            0x21 => {
                let label = bytes
                    .get(pos)
                    .copied()
                    .ok_or_else(|| ImageError::DecodeFailed("truncated GIF extension".into()))?;
                pos += 1;
                if label == 0xF9 {
                    transparent = parse_gif_gce(bytes, &mut pos)?;
                } else if label == 0x01 {
                    // Plain Text Extension: 12-byte fixed header, then
                    // data sub-blocks (both skipped by
                    // `skip_gif_fixed_extension`). A GCE preceding a
                    // plain-text extension applies to that text block
                    // (GIF89a: the GCE prefixes the graphic rendering
                    // block that follows it), so the pending transparency
                    // is consumed here and must not leak into a later
                    // image block.
                    skip_gif_fixed_extension(bytes, &mut pos, 12, "plain text")?;
                    transparent = None;
                } else if label == 0xFF {
                    // Application Extension: 11-byte fixed header, then
                    // data sub-blocks.
                    skip_gif_fixed_extension(bytes, &mut pos, 11, "application")?;
                } else {
                    // Comment and unknown extensions are pure sub-block
                    // chains.
                    skip_gif_sub_blocks(bytes, &mut pos)?;
                }
            }
            // Image descriptor: decode the first frame and stop.
            0x2C => {
                return decode_gif_frame(bytes, &mut pos, width, height, gct, transparent, canvas);
            }
            0x3B => {
                return Err(ImageError::DecodeFailed(
                    "GIF contains no image data".into(),
                ));
            }
            other => {
                return Err(ImageError::DecodeFailed(format!(
                    "unexpected GIF block 0x{other:02X}"
                )));
            }
        }
    }
}

/// Parse a Graphic Control Extension at `*pos` (past the 0xF9 label),
/// advancing `*pos` past the extension. Returns the transparent color
/// index when the transparency flag is set.
fn parse_gif_gce(bytes: &[u8], pos: &mut usize) -> Result<Option<u8>, ImageError> {
    let fail = || ImageError::DecodeFailed("truncated GIF graphic control extension".into());
    let size = bytes.get(*pos).copied().ok_or_else(fail)? as usize;
    *pos += 1;
    if size != 4 {
        // Malformed GCE: skip the declared field bytes and the
        // terminator as fixed-size data. They are NOT a sub-block
        // chain — interpreting them as one would desync the stream and
        // swallow the blocks that follow (the real GCE, the image
        // descriptor, ...).
        let end = pos
            .checked_add(size)
            .ok_or_else(|| ImageError::DecodeFailed("GIF extension offset overflow".into()))?;
        if bytes.get(*pos..end).is_none() {
            return Err(fail());
        }
        *pos = end;
        if bytes.get(*pos).copied() != Some(0) {
            return Err(fail());
        }
        *pos += 1;
        return Ok(None);
    }
    let end = pos
        .checked_add(4)
        .ok_or_else(|| ImageError::DecodeFailed("GIF extension offset overflow".into()))?;
    let fields = bytes.get(*pos..end).ok_or_else(fail)?;
    *pos = end;
    if bytes.get(*pos).copied() != Some(0) {
        return Err(fail());
    }
    *pos += 1;
    Ok((fields[0] & 0x01 != 0).then_some(fields[3]))
}

/// Skip a fixed-size extension header followed by a sub-block chain.
///
/// `expected` is the spec'd header size (12 for the Plain Text Extension,
/// 11 for the Application Extension), declared by the size byte past
/// `*pos`. The size is validated: on mismatch the stream is malformed,
/// and erroring out beats interpreting header bytes as sub-block lengths
/// — that desync would swallow the blocks that follow, e.g. a Graphic
/// Control Extension carrying the frame's transparency.
fn skip_gif_fixed_extension(
    bytes: &[u8],
    pos: &mut usize,
    expected: usize,
    name: &str,
) -> Result<(), ImageError> {
    let fail = || ImageError::DecodeFailed(format!("truncated GIF {name} extension"));
    let size = bytes.get(*pos).copied().ok_or_else(fail)? as usize;
    *pos += 1;
    if size != expected {
        return Err(ImageError::DecodeFailed(format!(
            "bad GIF {name} extension header size: {size}"
        )));
    }
    let end = pos
        .checked_add(size)
        .ok_or_else(|| ImageError::DecodeFailed("GIF extension offset overflow".into()))?;
    if end > bytes.len() {
        return Err(fail());
    }
    *pos = end;
    skip_gif_sub_blocks(bytes, pos)
}

/// Skip a sub-block chain (extension payloads): length-prefixed blocks
/// terminated by a zero length. Advances `*pos` past the terminator. No
/// allocation happens, so no cap is needed: the chain is bounded by the
/// input length.
fn skip_gif_sub_blocks(bytes: &[u8], pos: &mut usize) -> Result<(), ImageError> {
    loop {
        let size = bytes
            .get(*pos)
            .copied()
            .ok_or_else(|| ImageError::DecodeFailed("truncated GIF sub-block".into()))?
            as usize;
        *pos += 1;
        if size == 0 {
            return Ok(());
        }
        *pos = pos
            .checked_add(size)
            .ok_or_else(|| ImageError::DecodeFailed("GIF sub-block offset overflow".into()))?;
        if *pos > bytes.len() {
            return Err(ImageError::DecodeFailed(
                "truncated GIF sub-block data".into(),
            ));
        }
    }
}

/// Read a sub-block chain (LZW image data) into one buffer, capped at
/// [`GIF_MAX_SUB_BLOCK_BYTES`]. Advances `*pos` past the terminator.
fn read_gif_sub_blocks(bytes: &[u8], pos: &mut usize) -> Result<Vec<u8>, ImageError> {
    let mut out = Vec::new();
    loop {
        let size = bytes
            .get(*pos)
            .copied()
            .ok_or_else(|| ImageError::DecodeFailed("truncated GIF data sub-block".into()))?
            as usize;
        *pos += 1;
        if size == 0 {
            return Ok(out);
        }
        let end = pos
            .checked_add(size)
            .ok_or_else(|| ImageError::DecodeFailed("GIF sub-block offset overflow".into()))?;
        if end > bytes.len() {
            return Err(ImageError::DecodeFailed(
                "truncated GIF sub-block data".into(),
            ));
        }
        if out.len() + size > GIF_MAX_SUB_BLOCK_BYTES {
            return Err(ImageError::TooLarge);
        }
        out.extend_from_slice(&bytes[*pos..end]);
        *pos = end;
    }
}

/// Decode the image descriptor at `*pos` (past the 0x2C separator),
/// compositing its pixels onto `canvas` clipped to the logical screen,
/// and return the finished image. Advances `*pos` past the image data.
fn decode_gif_frame(
    bytes: &[u8],
    pos: &mut usize,
    screen_w: u32,
    screen_h: u32,
    gct: &[u8],
    transparent: Option<u8>,
    mut canvas: Vec<u8>,
) -> Result<DecodedImage, ImageError> {
    let end = pos
        .checked_add(9)
        .ok_or_else(|| ImageError::DecodeFailed("GIF image descriptor offset overflow".into()))?;
    let d = bytes
        .get(*pos..end)
        .ok_or_else(|| ImageError::DecodeFailed("truncated GIF image descriptor".into()))?;
    *pos = end;
    let left = u16::from_le_bytes([d[0], d[1]]) as usize;
    let top = u16::from_le_bytes([d[2], d[3]]) as usize;
    let fw = u16::from_le_bytes([d[4], d[5]]) as usize;
    let fh = u16::from_le_bytes([d[6], d[7]]) as usize;
    let flags = d[8];
    if fw == 0 || fh == 0 {
        return Err(ImageError::DecodeFailed(
            "GIF frame has zero dimension".into(),
        ));
    }
    if fw > MAX_IMAGE_DIMENSION as usize || fh > MAX_IMAGE_DIMENSION as usize {
        return Err(ImageError::TooLarge);
    }

    let mut palette: &[u8] = gct;
    if flags & 0x80 != 0 {
        // At most 768 bytes; no overflow possible.
        let table_len = 3usize << ((flags & 0x07) + 1);
        let tend = pos.checked_add(table_len).ok_or_else(|| {
            ImageError::DecodeFailed("GIF local color table offset overflow".into())
        })?;
        if tend > bytes.len() {
            return Err(ImageError::DecodeFailed(
                "truncated GIF local color table".into(),
            ));
        }
        palette = &bytes[*pos..tend];
        *pos = tend;
    }
    if palette.is_empty() {
        return Err(ImageError::DecodeFailed(
            "GIF frame has no color table".into(),
        ));
    }

    let interlaced = flags & 0x40 != 0;
    let min_code_size = bytes
        .get(*pos)
        .copied()
        .ok_or_else(|| ImageError::DecodeFailed("truncated GIF LZW minimum code size".into()))?;
    *pos += 1;
    if !matches!(min_code_size, 2..=8) {
        return Err(ImageError::DecodeFailed(
            "bad GIF LZW minimum code size".into(),
        ));
    }

    let frame_pixels = fw
        .checked_mul(fh)
        .ok_or_else(|| ImageError::DecodeFailed("GIF frame dimensions overflow".into()))?;
    // Indices are one byte each; the working-buffer cap matches the
    // canvas cap in pixel terms: GIF_MAX_OUTPUT_BYTES / 4 pixels.
    if frame_pixels > GIF_MAX_OUTPUT_BYTES / 4 {
        return Err(ImageError::TooLarge);
    }
    let compressed = read_gif_sub_blocks(bytes, pos)?;
    let mut indices = lzw_decompress_gif(&compressed, min_code_size, frame_pixels)?;
    if interlaced {
        indices = deinterlace_gif(&indices, fw, fh);
    }

    // Composite onto the canvas, clipping the frame to the screen. A pixel
    // index past the end of a short color table wraps (lenient: such files
    // are malformed but common in the wild); it can never index out of
    // bounds.
    let colors = palette.len() / 3;
    let sw = screen_w as usize;
    let sh = screen_h as usize;
    for (i, &index) in indices.iter().enumerate() {
        let cx = left + (i % fw);
        let cy = top + (i / fw);
        if cx >= sw || cy >= sh {
            continue;
        }
        if transparent == Some(index) {
            continue;
        }
        let p = (index as usize % colors) * 3;
        let o = (cy * sw + cx) * 4;
        canvas[o] = palette[p];
        canvas[o + 1] = palette[p + 1];
        canvas[o + 2] = palette[p + 2];
        canvas[o + 3] = 255;
    }

    Ok(DecodedImage {
        width: screen_w,
        height: screen_h,
        rgba: canvas,
    })
}

/// GIF-variant LZW decompression to exactly `pixel_count` color indices.
///
/// The dictionary holds prefix/suffix chains; codes grow from
/// `min_code_size + 1` to 12 bits, resetting on the clear code. Malformed
/// input (truncated stream, out-of-range code) is
/// [`ImageError::DecodeFailed`], never a panic. A stream yielding more
/// than `pixel_count` indices is truncated instead of failing: corrupt
/// input must not grow the buffer past its pre-checked size.
fn lzw_decompress_gif(
    data: &[u8],
    min_code_size: u8,
    pixel_count: usize,
) -> Result<Vec<u8>, ImageError> {
    let clear = 1u32 << min_code_size;
    let eoi = clear + 1;

    // prefix[i]/suffix[i] is the byte string for code i; codes below
    // `clear` are single bytes and need no table entry. Every stored
    // prefix is a smaller code, so chains strictly shrink and always
    // terminate; `scratch` (4096 bytes) always fits the longest chain.
    let mut prefix = [0u16; 4096];
    let mut suffix = [0u8; 4096];
    let mut out = Vec::with_capacity(pixel_count.min(1024 * 1024));
    let mut bit_pos = 0usize;
    let mut code_size = u32::from(min_code_size) + 1;
    // The code size grows once the next free code no longer fits.
    let mut max_code = 1u32 << code_size;
    let mut next_code = eoi + 1;
    let mut prev: Option<u32> = None;
    // One decoded string, stored reversed (the prefix chain walks tail-first).
    let mut scratch = [0u8; 4096];

    loop {
        let code = read_lzw_bits(data, &mut bit_pos, code_size)
            .ok_or_else(|| ImageError::DecodeFailed("truncated GIF LZW stream".into()))?;
        if code == clear {
            code_size = u32::from(min_code_size) + 1;
            max_code = 1u32 << code_size;
            next_code = eoi + 1;
            prev = None;
            continue;
        }
        if code == eoi {
            break;
        }

        // Resolve the code into `scratch` (reversed) and take its first byte.
        let (first, len) = if code == next_code {
            // KwKwK case: the code is not in the table yet; it stands for
            // the previous string plus its own first byte.
            let p = prev.ok_or_else(|| ImageError::DecodeFailed("bad GIF LZW code".into()))?;
            let (first, len) = write_lzw_chain(p, clear, &prefix, &suffix, &mut scratch);
            scratch[len] = first;
            (first, len + 1)
        } else if code < next_code {
            write_lzw_chain(code, clear, &prefix, &suffix, &mut scratch)
        } else {
            return Err(ImageError::DecodeFailed("bad GIF LZW code".into()));
        };

        for i in (0..len).rev() {
            // Truncate (not fail) corrupt overlong streams.
            if out.len() >= pixel_count {
                break;
            }
            out.push(scratch[i]);
        }
        if let Some(p) = prev
            && next_code < 4096
        {
            prefix[next_code as usize] = p as u16;
            suffix[next_code as usize] = first;
            next_code += 1;
            // The encoder widens the code the moment the next free code no
            // longer fits the current width, so the following code must be
            // read with the new width. `max_code` is the first code that
            // does not fit (1 << code_size), not the largest fitting code.
            if next_code >= max_code && code_size < 12 {
                code_size += 1;
                max_code = 1u32 << code_size;
            }
        }
        prev = Some(code);
    }

    if out.len() != pixel_count {
        return Err(ImageError::DecodeFailed("truncated GIF image data".into()));
    }
    Ok(out)
}

/// Write the dictionary string for `code` into `scratch`, reversed (tail
/// byte first); returns `(first_byte, len)`. Callers guarantee `code` is a
/// valid table entry, so the prefix chain strictly shrinks and the loop
/// terminates with `len <= 4096`.
fn write_lzw_chain(
    mut code: u32,
    clear: u32,
    prefix: &[u16; 4096],
    suffix: &[u8; 4096],
    scratch: &mut [u8; 4096],
) -> (u8, usize) {
    let mut len = 0usize;
    while code >= clear {
        scratch[len] = suffix[code as usize];
        len += 1;
        code = u32::from(prefix[code as usize]);
    }
    scratch[len] = code as u8;
    len += 1;
    (code as u8, len)
}

/// Read `n` bits (`n <= 12`), LSB-first, advancing `*pos`. Returns `None`
/// when the input ends mid-code.
fn read_lzw_bits(data: &[u8], pos: &mut usize, n: u32) -> Option<u32> {
    let byte = *pos / 8;
    let shift = *pos % 8;
    // The bits span ceil((shift + n) / 8) bytes: at most 3 for n <= 12.
    let mut v = 0u32;
    for i in 0..(shift + n as usize).div_ceil(8) {
        v |= u32::from(*data.get(byte + i)?) << (8 * i);
    }
    *pos += n as usize;
    Some((v >> shift) & ((1u32 << n) - 1))
}

/// Undo GIF interlacing: four passes starting at rows 0, 4, 2, 1 with
/// strides 8, 8, 4, 2. `pixels` must hold exactly `w * h` indices.
fn deinterlace_gif(pixels: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; pixels.len()];
    let mut src_row = 0usize;
    for (start, step) in [(0usize, 8usize), (4, 8), (2, 4), (1, 2)] {
        let mut y = start;
        while y < h {
            out[y * w..(y + 1) * w].copy_from_slice(&pixels[src_row * w..(src_row + 1) * w]);
            src_row += 1;
            y += step;
        }
    }
    out
}

/// Counting global allocator for tests: measures heap bytes allocated
/// through the system allocator, per thread, so tests can assert an
/// operation stays within an allocation budget without interference
/// from other test threads.
///
/// There can be only one `#[global_allocator]` per test binary, so it
/// lives here (outside `mod tests`) and every test module in the crate
/// shares it via [`ALLOCATED_BYTES`].
#[cfg(test)]
pub(crate) mod test_alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    // Per-thread bytes allocated through the global allocator.
    thread_local! {
        pub(crate) static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
    }

    struct CountingAlloc;

    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: delegates to System; only adds a thread-local
            // counter bump, which performs no allocation itself (the
            // Cell is const-initialized TLS storage).
            let ptr = unsafe { System.alloc(layout) };
            if !ptr.is_null() {
                ALLOCATED_BYTES.with(|c| c.set(c.get().saturating_add(layout.size())));
            }
            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: delegates to System with the layout from alloc.
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static COUNTING_ALLOC: CountingAlloc = CountingAlloc;
}

#[cfg(test)]
mod tests {
    use super::test_alloc::ALLOCATED_BYTES;
    use super::*;

    /// Build a tiny PNG in memory with the `png` encoder.
    fn make_png(
        width: u32,
        height: u32,
        color: png::ColorType,
        bit_depth: png::BitDepth,
        pixels: &[u8],
        interlaced: bool,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, width, height);
            enc.set_color(color);
            enc.set_depth(bit_depth);
            if color == png::ColorType::Indexed {
                // The encoder writes no palette by default, and encoding
                // an indexed image without one fails; give index 0 an
                // explicit red entry so the palette-expansion path is
                // exercised.
                enc.set_palette(vec![255, 0, 0]);
            }
            let mut writer = enc.write_header().unwrap();
            writer.write_image_data(pixels).unwrap();
        }
        if interlaced {
            // The encoder exposes no interlace API, so stamp the
            // Adam7 interlace byte into the IHDR by hand. Only valid for
            // 1x1: there the Adam7 pass layout is byte-identical to a
            // non-interlaced IDAT (only pass 1 carries data).
            debug_assert_eq!((width, height), (1, 1));
            out[28] = 1; // IHDR interlace method: 0 = none, 1 = Adam7
            // IHDR CRC covers the chunk type ("IHDR", bytes 12..16) plus
            // the 13 data bytes; hashing the data alone yields a CRC the
            // decoder rejects.
            let crc = crc32(&out[12..29]);
            out[29..33].copy_from_slice(&crc.to_be_bytes());
        }
        out
    }

    /// Minimal valid baseline 1-component JPEG, `width` x `height`, all
    /// pixels mid-gray. Hand-built: SOI, DQT, SOF0, DHT (DC+AC luminance),
    /// SOS, one flat MCU, EOI.
    fn make_jpeg(width: u16, height: u16) -> Vec<u8> {
        let mut j = vec![0xFF, 0xD8]; // SOI
        // DQT, table 0, all quantizers = 8.
        j.extend([0xFF, 0xDB, 0x00, 0x43, 0x00]);
        j.extend([8u8; 64]);
        // SOF0: 8-bit, h x w, 1 component, no subsampling.
        j.extend([0xFF, 0xC0, 0x00, 0x0B, 0x08]);
        j.extend(height.to_be_bytes());
        j.extend(width.to_be_bytes());
        j.extend([0x01, 0x01, 0x11, 0x00]);
        // DHT: standard luminance DC table 0.
        j.extend([0xFF, 0xC4, 0x00, 0x1F, 0x00]);
        j.extend([0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]);
        j.extend(0u8..12);
        // DHT: standard luminance AC table 0.
        j.extend([0xFF, 0xC4, 0x00, 0xB5, 0x10]);
        j.extend([0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D]);
        j.extend([
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51,
            0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1,
            0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18,
            0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
            0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57,
            0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
            0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92,
            0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
            0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
            0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8,
            0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2,
            0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
        ]);
        // SOS.
        j.extend([0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00]);
        // One flat block: DC category 0 ("00") + EOB ("1010"), padded with 1s.
        let blocks = (width as usize).div_ceil(8) * (height as usize).div_ceil(8);
        j.extend(std::iter::repeat_n(0x2B, blocks));
        j.extend([0xFF, 0xD9]); // EOI
        j
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }

    /// PNG with a lying IHDR (correct CRC) but no real image data.
    fn make_oversized_png_header(width: u32, height: u32) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend(PNG_SIGNATURE);
        let mut ihdr = Vec::new();
        ihdr.extend(width.to_be_bytes());
        ihdr.extend(height.to_be_bytes());
        ihdr.extend([8u8, 2, 0, 0, 0]); // 8-bit RGB
        p.extend(13u32.to_be_bytes());
        p.extend(b"IHDR");
        p.extend(&ihdr);
        let mut ihdr_tag = b"IHDR".to_vec();
        ihdr_tag.extend_from_slice(&ihdr);
        p.extend(crc32(&ihdr_tag).to_be_bytes());
        // Empty IDAT so read_info succeeds.
        p.extend(0u32.to_be_bytes());
        p.extend(b"IDAT");
        p.extend(crc32(b"IDAT").to_be_bytes());
        p
    }

    #[test]
    fn png_1x1_red() {
        let bytes = make_png(
            1,
            1,
            png::ColorType::Rgba,
            png::BitDepth::Eight,
            &[255, 0, 0, 255],
            false,
        );
        let img = decode_image(&bytes).unwrap();
        assert_eq!(img.width, 1);
        assert_eq!(img.height, 1);
        assert_eq!(img.rgba, vec![255, 0, 0, 255]);
    }

    #[test]
    fn png_interlaced_1x1_red() {
        let bytes = make_png(
            1,
            1,
            png::ColorType::Rgba,
            png::BitDepth::Eight,
            &[255, 0, 0, 255],
            true,
        );
        let img = decode_image(&bytes).unwrap();
        assert_eq!((img.width, img.height), (1, 1));
        assert_eq!(img.rgba, vec![255, 0, 0, 255]);
    }

    #[test]
    fn png_grayscale_expands() {
        let bytes = make_png(
            2,
            1,
            png::ColorType::Grayscale,
            png::BitDepth::Eight,
            &[0x10, 0x20],
            false,
        );
        let img = decode_image(&bytes).unwrap();
        assert_eq!((img.width, img.height), (2, 1));
        assert_eq!(img.rgba, vec![0x10, 0x10, 0x10, 255, 0x20, 0x20, 0x20, 255]);
    }

    #[test]
    fn png_palette_expands() {
        let bytes = make_png(
            1,
            1,
            png::ColorType::Indexed,
            png::BitDepth::Eight,
            &[0],
            false,
        );
        // Index 0 maps to the fixture's red palette entry; the EXPAND
        // transformation turns it into RGB, which becomes opaque red.
        let img = decode_image(&bytes).unwrap();
        assert_eq!((img.width, img.height), (1, 1));
        assert_eq!(img.rgba, vec![255, 0, 0, 255]);
    }

    #[test]
    fn jpeg_1x1_decodes() {
        let bytes = make_jpeg(1, 1);
        let img = decode_image(&bytes).unwrap();
        assert_eq!((img.width, img.height), (1, 1));
        assert_eq!(img.rgba.len(), 4);
        // Flat mid-gray block: r == g == b, fully opaque.
        assert_eq!(img.rgba[0], img.rgba[1]);
        assert_eq!(img.rgba[1], img.rgba[2]);
        assert_eq!(img.rgba[3], 255);
    }

    #[test]
    fn jpeg_8x8_decodes() {
        let bytes = make_jpeg(8, 8);
        let img = decode_image(&bytes).unwrap();
        assert_eq!((img.width, img.height), (8, 8));
        assert_eq!(img.rgba.len(), 8 * 8 * 4);
    }

    #[test]
    fn truncated_png_is_decode_failed_not_panic() {
        let bytes = make_png(
            4,
            4,
            png::ColorType::Rgba,
            png::BitDepth::Eight,
            &[9; 4 * 4 * 4],
            false,
        );
        let cut = bytes.len() / 2;
        match decode_image(&bytes[..cut]) {
            Err(ImageError::DecodeFailed(_)) => {}
            other => panic!("expected DecodeFailed, got {other:?}"),
        }
    }

    #[test]
    fn truncated_jpeg_is_decode_failed_not_panic() {
        let bytes = make_jpeg(8, 8);
        let cut = bytes.len() * 3 / 4;
        match decode_image(&bytes[..cut]) {
            Err(ImageError::DecodeFailed(_)) => {}
            other => panic!("expected DecodeFailed, got {other:?}"),
        }
    }

    #[test]
    fn unknown_bytes_are_unknown_format() {
        for input in [
            &[][..],
            b"hello world",
            b"BM",
            b"\x89PN", // partial PNG signature
            b"\xFF",   // partial JPEG SOI
            b"GIF8",   // partial GIF signature
        ] {
            match decode_image(input) {
                Err(ImageError::UnknownFormat) => {}
                other => panic!("expected UnknownFormat, got {other:?}"),
            }
        }
    }

    /// Minimal GIF89a header builder: signature + LSD + GCT, no blocks.
    fn make_gif_header(width: u16, height: u16) -> Vec<u8> {
        let mut g = Vec::new();
        g.extend_from_slice(b"GIF89a");
        g.extend_from_slice(&width.to_le_bytes());
        g.extend_from_slice(&height.to_le_bytes());
        g.extend_from_slice(&[0xF0, 0, 0]); // GCT flag, 2 entries
        g.extend_from_slice(&[255, 0, 0, 0, 0, 255]); // red, blue
        g
    }

    #[test]
    fn truncated_gif_is_decode_failed_not_panic() {
        // Signature matched, so these are DecodeFailed (not UnknownFormat).
        for cut in [6usize, 7, 12, 13, 20] {
            let full = make_gif_header(4, 2);
            // A header alone has no image data; append a truncated
            // trailer-less tail for the longer cuts.
            let mut bytes = full;
            bytes.extend_from_slice(&[0x21, 0xFE]); // comment extension, cut mid-chain
            match decode_image(&bytes[..cut.min(bytes.len())]) {
                Err(ImageError::DecodeFailed(_)) => {}
                other => panic!("expected DecodeFailed, got {other:?}"),
            }
        }
    }

    #[test]
    fn gif_without_image_data_is_decode_failed() {
        // Well-formed header, but the trailer arrives with no image block.
        let mut bytes = make_gif_header(4, 2);
        bytes.push(0x3B);
        match decode_image(&bytes) {
            Err(ImageError::DecodeFailed(_)) => {}
            other => panic!("expected DecodeFailed, got {other:?}"),
        }
    }

    #[test]
    fn oversized_gif_is_too_large() {
        for (w, h) in [(20000u16, 100u16), (100, 20000), (16384, 16384)] {
            // 16384x16384 passes the dimension check but needs a 1 GiB
            // canvas; the 64 MiB output cap rejects it before allocating.
            let bytes = make_gif_header(w, h);
            match decode_image(&bytes) {
                Err(ImageError::TooLarge) => {}
                other => panic!("expected TooLarge for {w}x{h}, got {other:?}"),
            }
        }
    }

    #[test]
    fn oversized_gif_frame_is_too_large() {
        // Tiny screen so the canvas cap passes, but the frame descriptor
        // claims 16384x4096 = 67_108_864 pixels: more than the 64 MiB / 4
        // = 16_777_216 pixel frame cap. The old check compared the pixel
        // count against the byte constant and let this through toward a
        // 64 MiB index buffer; it must be TooLarge.
        let mut bytes = make_gif_header(4, 2);
        bytes.push(0x2C); // image descriptor
        bytes.extend_from_slice(&0u16.to_le_bytes()); // left
        bytes.extend_from_slice(&0u16.to_le_bytes()); // top
        bytes.extend_from_slice(&16384u16.to_le_bytes()); // width
        bytes.extend_from_slice(&4096u16.to_le_bytes()); // height
        bytes.push(0x00); // flags: no local color table, no interlace
        bytes.push(0x02); // LZW minimum code size
        match decode_image(&bytes) {
            Err(ImageError::TooLarge) => {}
            other => panic!("expected TooLarge for oversized frame, got {other:?}"),
        }
    }

    #[test]
    fn oversized_png_is_too_large() {
        let bytes = make_oversized_png_header(20000, 100);
        match decode_image(&bytes) {
            Err(ImageError::TooLarge) => {}
            other => panic!("expected TooLarge, got {other:?}"),
        }
    }

    #[test]
    fn oversized_jpeg_is_rejected() {
        let bytes = make_jpeg(20000, 20000);
        match decode_image(&bytes) {
            // The SOF pre-scan rejects oversized headers with our own
            // TooLarge variant, so this assertion does not depend on
            // zune-jpeg's error wording.
            Err(ImageError::TooLarge) => {}
            other => panic!("expected TooLarge, got {other:?}"),
        }
    }

    /// JPEG with a hand-built SOI + SOF0 header claiming `width`x`height`
    /// and nothing else: just enough for the header-level checks to run,
    /// without depending on the decoder's truncation error wording.
    fn make_oversized_jpeg_header(width: u16, height: u16) -> Vec<u8> {
        let mut j = vec![0xFF, 0xD8]; // SOI
        // SOF0: 8-bit, height x width, 1 component.
        j.extend([0xFF, 0xC0, 0x00, 0x0B, 0x08]);
        j.extend(height.to_be_bytes());
        j.extend(width.to_be_bytes());
        j.extend([0x01, 0x01, 0x11, 0x00]);
        j
    }

    #[test]
    fn jpeg_max_dimensions_header_is_too_large() {
        // 16384x16384 passes the dimension check but its RGBA
        // post-expansion output (~1 GiB) exceeds the 256 MiB cap; it must
        // be rejected before the decoder allocates it.
        let bytes = make_oversized_jpeg_header(16384, 16384);
        let before = ALLOCATED_BYTES.with(|c| c.get());
        let result = decode_image(&bytes);
        let allocated = ALLOCATED_BYTES.with(|c| c.get()).saturating_sub(before);
        match result {
            Err(ImageError::TooLarge) => {}
            other => panic!("expected TooLarge, got {other:?}"),
        }
        assert!(
            allocated < 64 * 1024 * 1024,
            "bomb header allocated {allocated} bytes before rejection"
        );
    }

    #[test]
    fn png_bomb_header_is_rejected_without_huge_allocation() {
        // 16384x16384 passes the dimension check but needs a 1 GiB RGBA
        // output buffer; it must be rejected before that allocation.
        let bytes = make_oversized_png_header(16384, 16384);
        let before = ALLOCATED_BYTES.with(|c| c.get());
        let result = decode_image(&bytes);
        let allocated = ALLOCATED_BYTES.with(|c| c.get()).saturating_sub(before);
        match result {
            Err(ImageError::TooLarge) => {}
            other => panic!("expected TooLarge, got {other:?}"),
        }
        assert!(
            allocated < 64 * 1024 * 1024,
            "bomb header allocated {allocated} bytes before rejection"
        );
    }

    #[test]
    fn error_display_is_sane() {
        assert!(!format!("{}", ImageError::UnknownFormat).is_empty());
        assert!(!format!("{}", ImageError::TooLarge).is_empty());
        assert!(format!("{}", ImageError::DecodeFailed("x".into())).contains('x'));
    }

    // GIF decoding against binary fixtures in `tests/fixtures/`. The
    // fixtures are generated by
    // `~/workspace/scratch/bowser-orch4/build-images/mk_gif_fixtures.py` and
    // independently verified with PIL; the assertions below pin the exact
    // output pixels. `include_bytes!` keeps the tests inside this module so
    // no extra test target is needed.

    #[test]
    fn gif_4x2_fixture_decodes_exact_pixels() {
        let img = decode_image(include_bytes!("../tests/fixtures/gif_4x2.gif"));
        let img = img.expect("decode gif_4x2.gif");
        assert_eq!((img.width, img.height), (4, 2));
        // Row 0: red, green, blue, white. Row 1: white, blue, green, red.
        assert_eq!(
            img.rgba,
            vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255, //
                255, 255, 255, 255, 0, 0, 255, 255, 0, 255, 0, 255, 255, 0, 0, 255,
            ]
        );
    }

    #[test]
    fn gif_interlaced_fixture_deinterlaces() {
        let img = decode_image(include_bytes!("../tests/fixtures/gif_4x4_interlaced.gif"));
        let img = img.expect("decode gif_4x4_interlaced.gif");
        assert_eq!((img.width, img.height), (4, 4));
        // One flat color per row: row y is palette color y.
        let mut expected = Vec::with_capacity(4 * 4 * 4);
        for &(r, g, b) in &[(255u8, 0u8, 0u8), (0, 255, 0), (0, 0, 255), (255, 255, 255)] {
            for _ in 0..4 {
                expected.extend_from_slice(&[r, g, b, 255]);
            }
        }
        assert_eq!(img.rgba, expected);
    }

    #[test]
    fn gif_transparency_fixture_composites_onto_background() {
        let img = decode_image(include_bytes!("../tests/fixtures/gif_2x2_trans.gif"));
        let img = img.expect("decode gif_2x2_trans.gif");
        assert_eq!((img.width, img.height), (2, 2));
        // Pixels: red, green / transparent (index 2), white. The transparent
        // pixel shows the canvas background (background index 0 = red).
        assert_eq!(
            img.rgba,
            vec![
                255, 0, 0, 255, 0, 255, 0, 255, //
                255, 0, 0, 255, 255, 255, 255, 255,
            ]
        );
    }

    /// Minimal 1x1 GIF builder: red/blue palette, background index 1
    /// (blue), one red pixel, `extensions` spliced between the global
    /// color table and the image descriptor.
    fn gif_1x1_with_extensions(extensions: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GIF89a");
        b.extend_from_slice(&1u16.to_le_bytes()); // width
        b.extend_from_slice(&1u16.to_le_bytes()); // height
        b.extend_from_slice(&[0x80, 0x01, 0x00]); // GCT follows, bg = 1
        b.extend_from_slice(&[0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF]); // red, blue
        b.extend_from_slice(extensions);
        b.push(0x2C); // image descriptor
        b.extend_from_slice(&0u16.to_le_bytes()); // left
        b.extend_from_slice(&0u16.to_le_bytes()); // top
        b.extend_from_slice(&1u16.to_le_bytes()); // width
        b.extend_from_slice(&1u16.to_le_bytes()); // height
        b.push(0x00); // no local table, not interlaced
        // LZW: min code size 2, one pixel of index 0 (clear, 0, eoi).
        b.extend_from_slice(&[0x02, 0x02, 0x44, 0x01, 0x00]);
        b.push(0x3B); // trailer
        b
    }

    /// A Graphic Control Extension flagging palette index 0 transparent.
    fn gif_gce_transparent_index0() -> Vec<u8> {
        vec![0x21, 0xF9, 0x04, 0x01, 0x00, 0x00, 0x00, 0x00]
    }

    /// A well-formed Plain Text Extension (`0x21 0x01`): 12-byte fixed
    /// header, one data sub-block, terminator.
    fn gif_plain_text_extension() -> Vec<u8> {
        let mut b = vec![0x21, 0x01, 0x0C];
        b.extend_from_slice(&[0u8; 12]);
        b.extend_from_slice(&[0x01, 0x41, 0x00]);
        b
    }

    /// A GCE must still be consumed (parsed) when a plain-text extension
    /// is skipped before it: the transparent pixel shows the blue canvas
    /// background.
    #[test]
    fn gif_gce_after_plain_text_extension_applies() {
        let mut ext = gif_plain_text_extension();
        ext.extend_from_slice(&gif_gce_transparent_index0());
        let img = decode_image(&gif_1x1_with_extensions(&ext)).expect("decode");
        assert_eq!((img.width, img.height), (1, 1));
        assert_eq!(img.rgba, vec![0, 0, 255, 255]);
    }

    /// A GCE before a plain-text extension is consumed by that text block
    /// (GIF89a): the following image renders opaque, so the red pixel
    /// shows instead of the blue canvas background.
    #[test]
    fn gif_gce_before_plain_text_extension_applies() {
        let mut ext = gif_gce_transparent_index0();
        ext.extend_from_slice(&gif_plain_text_extension());
        let img = decode_image(&gif_1x1_with_extensions(&ext)).expect("decode");
        assert_eq!((img.width, img.height), (1, 1));
        assert_eq!(img.rgba, vec![255, 0, 0, 255]);
    }

    /// A plain-text extension with a lying header size must fail loudly,
    /// not desync the block parser and silently swallow the GCE that
    /// follows it.
    #[test]
    fn gif_plain_text_bad_header_size_is_decode_failed() {
        let mut ext = vec![0x21, 0x01, 0x05]; // not the spec'd 12
        ext.extend_from_slice(&[0u8; 12]);
        ext.extend_from_slice(&gif_gce_transparent_index0());
        match decode_image(&gif_1x1_with_extensions(&ext)) {
            Err(ImageError::DecodeFailed(msg)) => assert!(msg.contains("plain text")),
            other => panic!("expected DecodeFailed, got {other:?}"),
        }
    }

    /// A malformed GCE (block size != 4) is skipped as fixed-size data;
    /// the real GCE that follows it is still consumed.
    #[test]
    fn gif_malformed_gce_does_not_swallow_next_gce() {
        let mut ext = vec![0x21, 0xF9, 0x02, 0xAA, 0xBB, 0x00];
        ext.extend_from_slice(&gif_gce_transparent_index0());
        let img = decode_image(&gif_1x1_with_extensions(&ext)).expect("decode");
        assert_eq!((img.width, img.height), (1, 1));
        assert_eq!(img.rgba, vec![0, 0, 255, 255]);
    }

    /// In-place PNG expansion: every color type expands back-to-front to
    /// the right RGBA bytes.
    #[test]
    fn png_in_place_expansion_per_color_type() {
        // 2x1 RGB: red, green.
        let mut buf = vec![255u8, 0, 0, 0, 255, 0, 0, 0];
        expand_png_to_rgba_in_place(&mut buf, 6, png::ColorType::Rgb, 2, 1).unwrap();
        assert_eq!(buf, vec![255, 0, 0, 255, 0, 255, 0, 255]);
        // 2x1 grayscale.
        let mut buf = vec![10u8, 200, 0, 0, 0, 0, 0, 0];
        expand_png_to_rgba_in_place(&mut buf, 2, png::ColorType::Grayscale, 2, 1).unwrap();
        assert_eq!(buf, vec![10, 10, 10, 255, 200, 200, 200, 255]);
        // 1x1 grayscale+alpha.
        let mut buf = vec![10u8, 20, 0, 0];
        expand_png_to_rgba_in_place(&mut buf, 2, png::ColorType::GrayscaleAlpha, 1, 1).unwrap();
        assert_eq!(buf, vec![10, 10, 10, 20]);
        // RGBA is a validated no-op.
        let mut buf = vec![1u8, 2, 3, 4];
        expand_png_to_rgba_in_place(&mut buf, 4, png::ColorType::Rgba, 1, 1).unwrap();
        assert_eq!(buf, vec![1, 2, 3, 4]);
    }

    /// Size mismatches fail instead of mis-expanding: RGBA with a short
    /// byte count must not pass through as "RGBA", and short raw data or
    /// a short buffer must not expand either.
    #[test]
    fn png_in_place_expansion_rejects_size_mismatch() {
        let mut buf = vec![1u8, 2, 3, 4, 0, 0, 0, 0];
        assert!(expand_png_to_rgba_in_place(&mut buf, 4, png::ColorType::Rgba, 2, 1).is_err());
        let mut buf = vec![0u8; 8];
        assert!(expand_png_to_rgba_in_place(&mut buf, 5, png::ColorType::Rgb, 2, 1).is_err());
        let mut buf = vec![255u8, 0, 0, 0, 255, 0];
        assert!(expand_png_to_rgba_in_place(&mut buf, 6, png::ColorType::Rgb, 2, 1).is_err());
    }

    /// JPEG in-place expanders produce exact RGBA bytes.
    #[test]
    fn jpeg_in_place_expanders() {
        let mut buf = vec![255u8, 0, 0, 0, 255, 0, 0, 0];
        expand_rgb_to_rgba_in_place(&mut buf, 2);
        assert_eq!(buf, vec![255, 0, 0, 255, 0, 255, 0, 255]);
        let mut buf = vec![10u8, 200, 0, 0, 0, 0, 0, 0];
        expand_gray_to_rgba_in_place(&mut buf, 2);
        assert_eq!(buf, vec![10, 10, 10, 255, 200, 200, 200, 255]);
    }

    /// Cap-helper boundaries: exactly at the 256 MiB cap passes, one
    /// pixel over fails, and huge dimensions fail via checked arithmetic
    /// instead of wrapping.
    #[test]
    fn jpeg_rgba_cap_helper_boundaries() {
        // 8192x8192x4 = exactly 256 MiB.
        assert!(!jpeg_rgba_exceeds_cap(8192, 8192));
        assert!(jpeg_rgba_exceeds_cap(8193, 8192));
        assert!(jpeg_rgba_exceeds_cap(u32::MAX, u32::MAX));
        assert!(!jpeg_rgba_exceeds_cap(1, 1));
    }
}
