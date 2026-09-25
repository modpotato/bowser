//! Decoder implementations per the WHATWG Encoding Standard.
//!
//! Each decoder appends scalar values to `out` and returns whether any
//! U+FFFD substitution ("error" in spec terms) happened.

use super::data;
use super::{DecodeError, Encoding};

/// Decode `bytes` with `encoding`, appending to `out`.
///
/// Returns `Ok(had_errors)`, or `Err(Unsupported)` for encodings without a
/// decoder in this crate.
pub(crate) fn decode_into(
    encoding: Encoding,
    bytes: &[u8],
    out: &mut String,
) -> Result<bool, DecodeError> {
    if let Some(table) = encoding.single_byte_table() {
        return Ok(decode_single_byte(table, bytes, out));
    }
    match encoding {
        Encoding::Utf8 => Ok(decode_utf8(bytes, out)),
        Encoding::Big5 => Ok(decode_big5(bytes, out)),
        Encoding::Gbk | Encoding::Gb18030 => Ok(decode_gb18030(bytes, out)),
        Encoding::Utf16Be => Ok(decode_utf16(bytes, true, out)),
        Encoding::Utf16Le => Ok(decode_utf16(bytes, false, out)),
        Encoding::Replacement => Ok(decode_replacement(bytes, out)),
        Encoding::XUserDefined => Ok(decode_x_user_defined(bytes, out)),
        // Single-byte encodings are handled above; the rest have no decoder.
        _ => Err(DecodeError::Unsupported(encoding)),
    }
}

fn push_cp(out: &mut String, cp: u32) {
    out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
}

fn push_error(out: &mut String, had_errors: &mut bool) {
    out.push('\u{FFFD}');
    *had_errors = true;
}

/// UTF-8 decoder. Implemented on top of `core::str::from_utf8`, which reports
/// the same maximal-subpart errors the spec's algorithm produces ("Best
/// Practices for Using U+FFFD").
fn decode_utf8(bytes: &[u8], out: &mut String) -> bool {
    let mut had_errors = false;
    let mut rest = bytes;
    loop {
        match core::str::from_utf8(rest) {
            Ok(valid) => {
                out.push_str(valid);
                break;
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                out.push_str(core::str::from_utf8(&rest[..valid_up_to]).unwrap_or(""));
                push_error(out, &mut had_errors);
                match e.error_len() {
                    // A truncated sequence at the end is one error total.
                    None => break,
                    Some(len) => rest = &rest[valid_up_to + len..],
                }
            }
        }
    }
    had_errors
}

/// Legacy single-byte decoder: 0x00-0x7F map to themselves, the rest via the
/// index table (0xFFFF = unmapped = error).
fn decode_single_byte(table: &[u16; 128], bytes: &[u8], out: &mut String) -> bool {
    let mut had_errors = false;
    for &b in bytes {
        if b < 0x80 {
            out.push(b as char);
        } else {
            let cp = table[(b - 0x80) as usize];
            if cp == 0xFFFF {
                push_error(out, &mut had_errors);
            } else {
                push_cp(out, u32::from(cp));
            }
        }
    }
    had_errors
}

/// Big5 decoder. `leading` holds a pending lead byte (0x81-0xFE).
fn decode_big5(bytes: &[u8], out: &mut String) -> bool {
    let mut had_errors = false;
    let mut leading: Option<u8> = None;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        i += 1;
        match leading.take() {
            None => {
                if b < 0x80 {
                    out.push(b as char);
                } else if (0x81..=0xFE).contains(&b) {
                    leading = Some(b);
                } else {
                    push_error(out, &mut had_errors);
                }
            }
            Some(l) => {
                let offset = if b < 0x7F { 0x40 } else { 0x62 };
                let pointer = if (0x40..=0x7E).contains(&b) || (0xA1..=0xFE).contains(&b) {
                    Some(u32::from(l - 0x81) * 157 + u32::from(b - offset))
                } else {
                    None
                };
                match pointer {
                    // Pointers the index cannot express: two code points each.
                    Some(1133) => {
                        out.push('\u{CA}');
                        out.push('\u{304}');
                    }
                    Some(1135) => {
                        out.push('\u{CA}');
                        out.push('\u{30C}');
                    }
                    Some(1164) => {
                        out.push('\u{EA}');
                        out.push('\u{304}');
                    }
                    Some(1166) => {
                        out.push('\u{EA}');
                        out.push('\u{30C}');
                    }
                    Some(p) if (p as usize) < data::BIG5.len() => {
                        let cp = data::BIG5[p as usize];
                        if cp == 0xFFFF_FFFF {
                            push_error(out, &mut had_errors);
                            if b < 0x80 {
                                i -= 1; // restore ASCII byte to the queue
                            }
                        } else {
                            push_cp(out, cp);
                        }
                    }
                    _ => {
                        push_error(out, &mut had_errors);
                        if b < 0x80 {
                            i -= 1; // restore ASCII byte to the queue
                        }
                    }
                }
            }
        }
    }
    if leading.is_some() {
        push_error(out, &mut had_errors);
    }
    had_errors
}

/// gb18030 decoder, which is also GBK's decoder per the standard.
fn decode_gb18030(bytes: &[u8], out: &mut String) -> bool {
    let mut had_errors = false;
    // Decoder state: 0 = ground, 1 = have first, 2 = have first+second,
    // 3 = have first+second+third.
    let mut state: u8 = 0;
    let (mut first, mut second, mut third) = (0u8, 0u8, 0u8);
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        i += 1;
        match state {
            0 => match b {
                0x00..=0x7F => out.push(b as char),
                0x80 => out.push('\u{20AC}'),
                0x81..=0xFE => {
                    state = 1;
                    first = b;
                }
                0xFF => push_error(out, &mut had_errors),
            },
            1 => match b {
                0x30..=0x39 => {
                    state = 2;
                    second = b;
                }
                _ => {
                    let pointer = if (0x40..=0x7E).contains(&b) || (0x80..=0xFE).contains(&b) {
                        let offset = if b < 0x7F { 0x40 } else { 0x41 };
                        Some(u32::from(first - 0x81) * 190 + u32::from(b - offset))
                    } else {
                        None
                    };
                    match pointer.and_then(gb18030_two_byte) {
                        Some(cp) => {
                            push_cp(out, cp);
                        }
                        None => {
                            // Prepend byte to the stream and error.
                            push_error(out, &mut had_errors);
                            i -= 1;
                        }
                    }
                    state = 0;
                }
            },
            2 => match b {
                0x81..=0xFE => {
                    state = 3;
                    third = b;
                }
                // Restore «second, byte» to the queue.
                _ => {
                    push_error(out, &mut had_errors);
                    i -= 2;
                    state = 0;
                }
            },
            _ => match b {
                0x30..=0x39 => {
                    let pointer = u32::from(first - 0x81) * 12600
                        + u32::from(second - 0x30) * 1260
                        + u32::from(third - 0x81) * 10
                        + u32::from(b - 0x30);
                    match gb18030_ranges(pointer) {
                        Some(cp) => push_cp(out, cp),
                        None => push_error(out, &mut had_errors),
                    }
                    state = 0;
                }
                // Restore «second, third, byte» to the queue.
                _ => {
                    push_error(out, &mut had_errors);
                    i -= 3;
                    state = 0;
                }
            },
        }
    }
    if state != 0 {
        push_error(out, &mut had_errors);
    }
    had_errors
}

/// "index code point for pointer in index gb18030": exact match, else null.
fn gb18030_two_byte(pointer: u32) -> Option<u32> {
    let cp = *data::GB18030.get(pointer as usize)?;
    (cp != 0xFFFF_FFFF).then_some(cp)
}

/// "index gb18030 ranges code point for pointer".
fn gb18030_ranges(pointer: u32) -> Option<u32> {
    // The two dead zones of the four-byte space.
    if (pointer > 39419 && pointer < 189_000) || pointer > 1_237_575 {
        return None;
    }
    if pointer == 7457 {
        return Some(0xE7C7);
    }
    if (pointer > 39419 && pointer < 189000) || pointer > 1237575 {
        return None;
    }
    // Last range with start <= pointer.
    let idx = data::GB18030_RANGES.partition_point(|&(start, _)| start <= pointer);
    if idx == 0 {
        return None;
    }
    let (start, base) = data::GB18030_RANGES[idx - 1];
    Some(base + (pointer - start))
}

/// Shared UTF-16 decoder (BE or LE).
fn decode_utf16(bytes: &[u8], big_endian: bool, out: &mut String) -> bool {
    fn is_lead(u: u16) -> bool {
        (0xD800..=0xDBFF).contains(&u)
    }
    fn is_trail(u: u16) -> bool {
        (0xDC00..=0xDFFF).contains(&u)
    }

    let mut had_errors = false;
    let mut lead: Option<u16> = None;
    let (chunks, remainder) = bytes.as_chunks::<2>();
    let mut chunks = chunks.iter();
    // Process units; a `pending` unit is re-examined after an error restore.
    let mut pending: Option<u16> = None;
    loop {
        let unit = match pending.take() {
            Some(u) => Some(u),
            None => chunks.next().map(|c| {
                if big_endian {
                    u16::from(c[0]) << 8 | u16::from(c[1])
                } else {
                    u16::from(c[1]) << 8 | u16::from(c[0])
                }
            }),
        };
        let Some(unit) = unit else { break };
        match lead.take() {
            Some(l) => {
                if is_trail(unit) {
                    let hi = u32::from(l - 0xD800);
                    let lo = u32::from(unit - 0xDC00);
                    push_cp(out, 0x10000 + (hi << 10) + lo);
                } else {
                    // Error on the lead; the spec restores this unit's bytes.
                    push_error(out, &mut had_errors);
                    pending = Some(unit);
                }
            }
            None => {
                if is_lead(unit) {
                    lead = Some(unit);
                } else if is_trail(unit) {
                    push_error(out, &mut had_errors);
                } else {
                    push_cp(out, u32::from(unit));
                }
            }
        }
    }
    if remainder.len() == 1 {
        push_error(out, &mut had_errors);
    }
    if lead.is_some() {
        push_error(out, &mut had_errors);
    }
    had_errors
}

/// replacement decoder: one error for any non-empty input, then finished.
fn decode_replacement(bytes: &[u8], out: &mut String) -> bool {
    if bytes.is_empty() {
        false
    } else {
        out.push('\u{FFFD}');
        true
    }
}

/// x-user-defined decoder: ASCII as-is, 0x80-0xFF -> U+F780-U+F7FF.
fn decode_x_user_defined(bytes: &[u8], out: &mut String) -> bool {
    for &b in bytes {
        if b < 0x80 {
            out.push(b as char);
        } else {
            push_cp(out, 0xF780 + u32::from(b - 0x80));
        }
    }
    false
}
