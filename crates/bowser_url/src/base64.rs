//! Base64 decode (standard alphabet), std only.
//!
//! Shared by the `atob` web API (`bowser_js`) and `data:` URL decoding
//! (image/font loads). One implementation so the two cannot disagree.

const B64_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_value(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Forgiving base64 decode: strip ASCII whitespace, reject a length of
/// `1 mod 4` (before and after padding removal), strip up to two `=`
/// padding characters, then decode 4-char quanta.
///
/// This matches the web `atob` semantics, which is also what `data:`
/// URLs require (RFC 2397 via the Fetch spec: ASCII whitespace is
/// stripped before decoding).
#[allow(clippy::result_unit_err)]
pub fn base64_decode(input: &str) -> Result<Vec<u8>, ()> {
    let mut s: Vec<u8> = input
        .bytes()
        .filter(|b| !matches!(b, b'\t' | b'\n' | 0x0C | b'\r' | b' '))
        .collect();
    if s.len() % 4 == 1 {
        return Err(());
    }
    if s.ends_with(b"==") {
        s.truncate(s.len() - 2);
    } else if s.ends_with(b"=") {
        s.truncate(s.len() - 1);
    }
    if s.len() % 4 == 1 {
        return Err(());
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3 + 3);
    for quad in s.chunks(4) {
        let k = quad.len();
        let mut n: u32 = 0;
        for &c in quad {
            n = (n << 6) | u32::from(b64_value(c).ok_or(())?);
        }
        n <<= 6 * (4 - k);
        out.push((n >> 16) as u8);
        if k > 2 {
            out.push((n >> 8) as u8);
        }
        if k > 3 {
            out.push(n as u8);
        }
    }
    Ok(out)
}

/// Standard-alphabet base64 encode (the `btoa` counterpart).
pub fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(B64_ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            B64_ALPHABET[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64_ALPHABET[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        for data in [&b""[..], b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar"] {
            assert_eq!(base64_decode(&base64_encode(data)).unwrap(), data);
        }
    }

    #[test]
    fn forgiving_whitespace_and_padding() {
        assert_eq!(base64_decode("Zm9v").unwrap(), b"foo");
        assert_eq!(base64_decode(" Zm9v\n").unwrap(), b"foo");
        assert_eq!(base64_decode("Zg==").unwrap(), b"f");
        assert_eq!(base64_decode("Zm8=").unwrap(), b"fo");
        assert!(base64_decode("Z").is_err());
        assert!(base64_decode("Zm9v!").is_err());
    }
}
