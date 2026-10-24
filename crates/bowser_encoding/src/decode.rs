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
        Encoding::ShiftJis => Ok(decode_shift_jis(bytes, out)),
        Encoding::EucJp => Ok(decode_euc_jp(bytes, out)),
        Encoding::EucKr => Ok(decode_euc_kr(bytes, out)),
        Encoding::Utf16Be => Ok(decode_utf16(bytes, true, out)),
        Encoding::Utf16Le => Ok(decode_utf16(bytes, false, out)),
        Encoding::Replacement => Ok(decode_replacement(bytes, out)),
        Encoding::XUserDefined => Ok(decode_x_user_defined(bytes, out)),
        Encoding::Iso2022Jp => Ok(decode_iso_2022_jp(bytes, out)),
        // Fail closed for any future variant added without a decoder.
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

/// Shift_JIS decoder. `leading` holds a pending lead byte
/// (0x81-0x9F or 0xE0-0xFC).
fn decode_shift_jis(bytes: &[u8], out: &mut String) -> bool {
    let mut had_errors = false;
    let mut leading: Option<u8> = None;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        i += 1;
        match leading.take() {
            None => {
                if b <= 0x80 {
                    // ASCII and U+0080 map to themselves.
                    out.push(b as char);
                } else if (0xA1..=0xDF).contains(&b) {
                    push_cp(out, 0xFF61 - 0xA1 + u32::from(b));
                } else if (0x81..=0x9F).contains(&b) || (0xE0..=0xFC).contains(&b) {
                    leading = Some(b);
                } else {
                    push_error(out, &mut had_errors);
                }
            }
            Some(l) => {
                let offset = if b < 0x7F { 0x40 } else { 0x41 };
                let leading_offset = if l < 0xA0 { 0x81 } else { 0xC1 };
                let pointer = if (0x40..=0x7E).contains(&b) || (0x80..=0xFC).contains(&b) {
                    Some(u32::from(l - leading_offset) * 188 + u32::from(b - offset))
                } else {
                    None
                };
                match pointer {
                    // Windows EUDC (U+E000..=U+E757) is consulted before the index.
                    Some(p) if (8836..=10715).contains(&p) => {
                        push_cp(out, 0xE000 - 8836 + p);
                    }
                    _ => {
                        let cp = pointer
                            .and_then(|p| data::JIS0208.get(p as usize).copied())
                            .filter(|&cp| cp != 0xFFFF);
                        match cp {
                            Some(cp) => push_cp(out, u32::from(cp)),
                            None => {
                                push_error(out, &mut had_errors);
                                if b < 0x80 {
                                    i -= 1; // restore ASCII byte to the queue
                                }
                            }
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

/// EUC-JP decoder. `leading` holds a pending lead byte (0x8E, 0x8F, or
/// 0xA1-0xFE); `jis0212` marks a 0x8F-introduced JIS X 0212 pair.
fn decode_euc_jp(bytes: &[u8], out: &mut String) -> bool {
    let mut had_errors = false;
    let mut leading: Option<u8> = None;
    let mut jis0212 = false;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        i += 1;
        match leading.take() {
            // 0x8E trail: halfwidth katakana.
            Some(0x8E) if (0xA1..=0xDF).contains(&b) => {
                push_cp(out, 0xFF61 - 0xA1 + u32::from(b));
            }
            // 0x8F trail: first half of a JIS X 0212 pair.
            Some(0x8F) if (0xA1..=0xFE).contains(&b) => {
                jis0212 = true;
                leading = Some(b);
            }
            Some(l) => {
                let pointer = if (0xA1..=0xFE).contains(&l) && (0xA1..=0xFE).contains(&b) {
                    Some(u32::from(l - 0xA1) * 94 + u32::from(b - 0xA1))
                } else {
                    None
                };
                let table: &[u16] = if jis0212 {
                    &data::JIS0212
                } else {
                    &data::JIS0208
                };
                jis0212 = false;
                let cp = pointer.and_then(|p| table.get(p as usize).copied());
                match cp {
                    Some(cp) if cp != 0xFFFF => push_cp(out, u32::from(cp)),
                    _ => {
                        push_error(out, &mut had_errors);
                        if b < 0x80 {
                            i -= 1; // restore ASCII byte to the queue
                        }
                    }
                }
            }
            None => {
                if b < 0x80 {
                    out.push(b as char);
                } else if b == 0x8E || b == 0x8F || (0xA1..=0xFE).contains(&b) {
                    leading = Some(b);
                } else {
                    push_error(out, &mut had_errors);
                }
            }
        }
    }
    if leading.is_some() {
        push_error(out, &mut had_errors);
    }
    had_errors
}

/// EUC-KR decoder. `leading` holds a pending lead byte (0x81-0xFE).
fn decode_euc_kr(bytes: &[u8], out: &mut String) -> bool {
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
                let pointer = if (0x41..=0xFE).contains(&b) {
                    Some(u32::from(l - 0x81) * 190 + u32::from(b - 0x41))
                } else {
                    None
                };
                let cp = pointer.and_then(|p| data::EUCKR.get(p as usize).copied());
                match cp {
                    Some(cp) if cp != 0xFFFF => push_cp(out, u32::from(cp)),
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
                            push_error(out, &mut had_errors);
                            if b < 0x80 {
                                i -= 1; // restore ASCII byte to the queue
                            }
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

/// ISO-2022-JP decoder: the one stateful decoder in the standard.
///
/// Seven states (ASCII, Roman, Katakana, LeadByte, TrailByte, EscapeStart,
/// Escape), an `output_state` to fall back to after a bad escape sequence,
/// and an `output` flag: a recognized escape sequence sets it, any emitted
/// code point clears it, and a second identical escape sequence while it is
/// set becomes an error. The spec's queue-restore steps in EscapeStart and
/// Escape become read-index rewinds here.
fn decode_iso_2022_jp(bytes: &[u8], out: &mut String) -> bool {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum State {
        Ascii,
        Roman,
        Katakana,
        LeadByte,
        TrailByte,
        EscapeStart,
        Escape,
    }

    let mut had_errors = false;
    let mut state = State::Ascii;
    let mut output_state = State::Ascii;
    let mut leading: u8 = 0;
    let mut output = false;

    let mut i = 0;
    loop {
        let byte: Option<u8> = if i < bytes.len() {
            let b = bytes[i];
            i += 1;
            Some(b)
        } else {
            None
        };
        match state {
            State::Ascii => match byte {
                Some(0x1B) => state = State::EscapeStart,
                Some(b) if b < 0x80 && b != 0x0E && b != 0x0F => {
                    output = false;
                    out.push(b as char);
                }
                Some(_) => {
                    output = false;
                    push_error(out, &mut had_errors);
                }
                None => break, // finished
            },
            State::Roman => match byte {
                Some(0x1B) => state = State::EscapeStart,
                Some(0x5C) => {
                    output = false;
                    out.push('\u{A5}');
                }
                Some(0x7E) => {
                    output = false;
                    out.push('\u{203E}');
                }
                Some(b) if b < 0x80 && b != 0x0E && b != 0x0F => {
                    output = false;
                    out.push(b as char);
                }
                Some(_) => {
                    output = false;
                    push_error(out, &mut had_errors);
                }
                None => break, // finished
            },
            State::Katakana => match byte {
                Some(0x1B) => state = State::EscapeStart,
                Some(b) if (0x21..=0x5F).contains(&b) => {
                    output = false;
                    push_cp(out, 0xFF61 - 0x21 + u32::from(b));
                }
                Some(_) => {
                    output = false;
                    push_error(out, &mut had_errors);
                }
                None => break, // finished
            },
            State::LeadByte => match byte {
                Some(0x1B) => state = State::EscapeStart,
                Some(b) if (0x21..=0x7E).contains(&b) => {
                    output = false;
                    leading = b;
                    state = State::TrailByte;
                }
                Some(_) => {
                    output = false;
                    push_error(out, &mut had_errors);
                }
                None => break, // finished
            },
            State::TrailByte => match byte {
                // The 0x1B is consumed, not restored.
                Some(0x1B) => {
                    state = State::EscapeStart;
                    push_error(out, &mut had_errors);
                }
                Some(b) if (0x21..=0x7E).contains(&b) => {
                    state = State::LeadByte;
                    let pointer = u32::from(leading - 0x21) * 94 + u32::from(b - 0x21);
                    match data::JIS0208.get(pointer as usize).copied() {
                        Some(cp) if cp != 0xFFFF => push_cp(out, u32::from(cp)),
                        _ => push_error(out, &mut had_errors),
                    }
                }
                _ => {
                    state = State::LeadByte;
                    push_error(out, &mut had_errors);
                }
            },
            State::EscapeStart => match byte {
                Some(b @ (0x24 | 0x28)) => {
                    leading = b;
                    state = State::Escape;
                }
                Some(_) => {
                    i -= 1; // restore byte
                    output = false;
                    state = output_state;
                    push_error(out, &mut had_errors);
                }
                None => {
                    output = false;
                    state = output_state;
                    push_error(out, &mut had_errors);
                }
            },
            State::Escape => {
                let lead = leading;
                leading = 0;
                let target = match (lead, byte) {
                    (0x28, Some(0x42)) => Some(State::Ascii),
                    (0x28, Some(0x4A)) => Some(State::Roman),
                    (0x28, Some(0x49)) => Some(State::Katakana),
                    (0x24, Some(0x40)) | (0x24, Some(0x42)) => Some(State::LeadByte),
                    _ => None,
                };
                match target {
                    Some(next) => {
                        state = next;
                        output_state = next;
                        let was_output = output;
                        output = true;
                        if was_output {
                            push_error(out, &mut had_errors);
                        }
                    }
                    None => {
                        match byte {
                            Some(_) => i -= 2, // restore «leading, byte»
                            None => i -= 1,    // restore leading
                        }
                        output = false;
                        state = output_state;
                        push_error(out, &mut had_errors);
                    }
                }
            }
        }
    }
    had_errors
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode `bytes` with one of the new CJK decoders directly.
    fn run(decoder: fn(&[u8], &mut String) -> bool, bytes: &[u8]) -> (String, bool) {
        let mut out = String::new();
        let had_errors = decoder(bytes, &mut out);
        (out, had_errors)
    }

    #[test]
    fn shift_jis_vectors() {
        // ASCII and U+0080 pass through.
        assert_eq!(run(decode_shift_jis, b"hello"), ("hello".into(), false));
        assert_eq!(run(decode_shift_jis, b"\x80"), ("\u{80}".into(), false));
        // Halfwidth katakana: 0xA6 -> U+FF66.
        assert_eq!(run(decode_shift_jis, b"\xa6"), ("\u{FF66}".into(), false));
        // 0x82 0xA0 -> pointer 283 -> U+3042 (HIRAGANA LETTER A).
        assert_eq!(run(decode_shift_jis, b"\x82\xa0"), ("あ".into(), false));
        // 0x82 0xA4 -> pointer 287 -> U+3046 (HIRAGANA LETTER U).
        assert_eq!(run(decode_shift_jis, b"\x82\xa4"), ("う".into(), false));
        // High lead range: 0xE0 0x40 -> pointer 5828 -> U+6F3E.
        assert_eq!(run(decode_shift_jis, b"\xe0\x40"), ("漾".into(), false));
        // Windows EUDC: pointers 8836..=10715 map to U+E000..=U+E757.
        assert_eq!(
            run(decode_shift_jis, b"\xf0\x40"),
            ("\u{E000}".into(), false)
        );
        assert_eq!(
            run(decode_shift_jis, b"\xf9\xfc"),
            ("\u{E757}".into(), false)
        );
        // Unmapped pointer (108): one error, no byte restored (0xAD > 0x7F).
        assert_eq!(
            run(decode_shift_jis, b"\x81\xad"),
            ("\u{FFFD}".into(), true)
        );
        // Invalid trail byte: error, then the ASCII byte is restored.
        assert_eq!(
            run(decode_shift_jis, b"\x82\x20"),
            ("\u{FFFD} ".into(), true)
        );
        // Lone lead at end of input: one error.
        assert_eq!(run(decode_shift_jis, b"\x82"), ("\u{FFFD}".into(), true));
        // 0xFD is not a lead byte.
        assert_eq!(run(decode_shift_jis, b"\xfd"), ("\u{FFFD}".into(), true));
        // Lead, error, then valid pair: decoding resumes cleanly.
        assert_eq!(
            run(decode_shift_jis, b"\x82\x20\x82\xa0"),
            ("\u{FFFD} あ".into(), true)
        );
    }

    #[test]
    fn euc_jp_vectors() {
        assert_eq!(run(decode_euc_jp, b"hello"), ("hello".into(), false));
        // 0x8E 0xA6 -> U+FF66 (halfwidth katakana).
        assert_eq!(run(decode_euc_jp, b"\x8e\xa6"), ("\u{FF66}".into(), false));
        // 0xA4 0xA2 -> pointer 283 -> U+3042 (HIRAGANA LETTER A).
        assert_eq!(run(decode_euc_jp, b"\xa4\xa2"), ("あ".into(), false));
        // 0x8F 0xA2 0xAF -> JIS X 0212 pointer 108 -> U+02D8 (MODIFIER LETTER BREVE).
        assert_eq!(
            run(decode_euc_jp, b"\x8f\xa2\xaf"),
            ("\u{2D8}".into(), false)
        );
        // JIS X 0212 pointer 0 is unmapped.
        assert_eq!(
            run(decode_euc_jp, b"\x8f\xa1\xa1"),
            ("\u{FFFD}".into(), true)
        );
        // jis0212 flag resets: 0212 pair followed by a 0208 pair.
        assert_eq!(
            run(decode_euc_jp, b"\x8f\xa2\xaf\xa4\xa2"),
            ("\u{2D8}あ".into(), false)
        );
        // Invalid trail: error, ASCII byte restored.
        assert_eq!(run(decode_euc_jp, b"\xa4\x20"), ("\u{FFFD} ".into(), true));
        // 0x8E with a bad trail consumes both bytes (0xE0 is not restored).
        assert_eq!(run(decode_euc_jp, b"\x8e\xe0"), ("\u{FFFD}".into(), true));
        // Dangling leads at end of input.
        assert_eq!(run(decode_euc_jp, b"\x8e"), ("\u{FFFD}".into(), true));
        assert_eq!(run(decode_euc_jp, b"\x8f\xa1"), ("\u{FFFD}".into(), true));
        // 0xA0 is not a lead byte.
        assert_eq!(run(decode_euc_jp, b"\xa0"), ("\u{FFFD}".into(), true));
    }

    #[test]
    fn euc_kr_vectors() {
        assert_eq!(run(decode_euc_kr, b"hello"), ("hello".into(), false));
        // 0xB0 0xA1 -> pointer 9026 -> U+AC00 (HANGUL SYLLABLE GA).
        assert_eq!(run(decode_euc_kr, b"\xb0\xa1"), ("가".into(), false));
        // 0xB0 0xA2 -> pointer 9027 -> U+AC01.
        assert_eq!(run(decode_euc_kr, b"\xb0\xa2"), ("\u{AC01}".into(), false));
        // Unmapped pointer (26): error, then the ASCII trail is restored.
        assert_eq!(run(decode_euc_kr, b"\x81\x5b"), ("\u{FFFD}[".into(), true));
        // Invalid trail: error, ASCII byte restored.
        assert_eq!(run(decode_euc_kr, b"\xb0\x20"), ("\u{FFFD} ".into(), true));
        // Dangling lead at end of input.
        assert_eq!(run(decode_euc_kr, b"\xb0"), ("\u{FFFD}".into(), true));
        // 0xFF is not a lead byte; 0x80 is not ASCII and not a lead either.
        assert_eq!(run(decode_euc_kr, b"\xff"), ("\u{FFFD}".into(), true));
        assert_eq!(run(decode_euc_kr, b"\x80"), ("\u{FFFD}".into(), true));
    }

    #[test]
    fn decode_into_routes_cjk_encodings() {
        let mut out = String::new();
        // Shift_JIS 0x82 0xA0, EUC-JP 0xA4 0xA2, EUC-KR 0xB0 0xA1 and
        // ISO-2022-JP ESC $ B 0x24 0x22 all decode without errors now
        // instead of returning Unsupported.
        for (enc, bytes, want) in [
            (Encoding::ShiftJis, b"\x82\xa0".as_slice(), "あ"),
            (Encoding::EucJp, b"\xa4\xa2".as_slice(), "あ"),
            (Encoding::EucKr, b"\xb0\xa1".as_slice(), "가"),
            (Encoding::Iso2022Jp, b"\x1b$B\x24\x22".as_slice(), "あ"),
        ] {
            out.clear();
            let had_errors = decode_into(enc, bytes, &mut out).expect("supported");
            assert!(!had_errors);
            assert_eq!(out, want);
        }
    }

    #[test]
    fn iso_2022_jp_vectors() {
        // ASCII passes through.
        assert_eq!(run(decode_iso_2022_jp, b"hello"), ("hello".into(), false));
        // ESC $ B enters JIS X 0208: lead 0x24 trail 0x22 is pointer 283 -> あ.
        assert_eq!(
            run(decode_iso_2022_jp, b"\x1b$B\x24\x22"),
            ("あ".into(), false)
        );
        // ESC $ @ also enters JIS X 0208: 0x21 0x21 is pointer 0 -> U+3000.
        assert_eq!(
            run(decode_iso_2022_jp, b"\x1b$@\x21\x21"),
            ("\u{3000}".into(), false)
        );
        // Escape back to ASCII mid-stream.
        assert_eq!(
            run(decode_iso_2022_jp, "a\x1b$B\x24\x22\x1b(Bb".as_bytes()),
            ("aあb".into(), false)
        );
        // ESC ( I enters Katakana: 0x21 -> U+FF61, 0x5F -> U+FF9F.
        assert_eq!(
            run(decode_iso_2022_jp, b"\x1b(I\x21\x5f"),
            ("｡ﾟ".into(), false)
        );
        // ESC ( J enters Roman: 0x5C -> U+00A5, 0x7E -> U+203E.
        assert_eq!(
            run(decode_iso_2022_jp, b"\x1b(J\x5c\x7e"),
            ("¥\u{203E}".into(), false)
        );
        // A repeated escape sequence is an error while the output flag is set.
        assert_eq!(
            run(decode_iso_2022_jp, b"\x1b$B\x1b$B"),
            ("\u{FFFD}".into(), true)
        );
        // The flag clears once a code point is emitted, so this is clean.
        assert_eq!(
            run(decode_iso_2022_jp, b"\x1b$B\x24\x22\x1b$B"),
            ("あ".into(), false)
        );
        // Spec 12.2.2's example: U+00A5 encoded twice decodes as ¥ FFFD ¥.
        assert_eq!(
            run(decode_iso_2022_jp, b"\x1b(J\x5c\x1b(B\x1b(J\x5c\x1b(B"),
            ("¥\u{FFFD}¥".into(), true)
        );
        // Bad escape: the byte is restored to the output state.
        assert_eq!(
            run(decode_iso_2022_jp, b"\x1bX"),
            ("\u{FFFD}X".into(), true)
        );
        // Bad escape sequence: «leading, byte» restored.
        assert_eq!(
            run(decode_iso_2022_jp, b"\x1b$)"),
            ("\u{FFFD}$)".into(), true)
        );
        // ESC at end of input: one error, then finished in the output state.
        assert_eq!(run(decode_iso_2022_jp, b"\x1b"), ("\u{FFFD}".into(), true));
        // ESC $ at end: the lead byte is restored and decoded as ASCII.
        assert_eq!(
            run(decode_iso_2022_jp, b"\x1b$"),
            ("\u{FFFD}$".into(), true)
        );
        // Dangling JIS X 0208 lead at end: one error.
        assert_eq!(
            run(decode_iso_2022_jp, b"\x1b$B\x24"),
            ("\u{FFFD}".into(), true)
        );
        // 0x1B in TrailByte is swallowed: error, then ESC ( B works.
        assert_eq!(
            run(decode_iso_2022_jp, b"\x1b$B\x24\x1b(B"),
            ("\u{FFFD}".into(), true)
        );
        // Unmapped pointer (108): error, both bytes consumed.
        assert_eq!(
            run(decode_iso_2022_jp, b"\x1b$B\x22\x2f"),
            ("\u{FFFD}".into(), true)
        );
        // 0x0E/0x0F and non-ASCII bytes are errors in the ASCII state.
        assert_eq!(run(decode_iso_2022_jp, b"\x0e"), ("\u{FFFD}".into(), true));
        assert_eq!(run(decode_iso_2022_jp, b"\x0f"), ("\u{FFFD}".into(), true));
        assert_eq!(run(decode_iso_2022_jp, b"\x80"), ("\u{FFFD}".into(), true));
        // 0x20 is out of the Katakana range.
        assert_eq!(
            run(decode_iso_2022_jp, b"\x1b(I\x20"),
            ("\u{FFFD}".into(), true)
        );
    }
}
