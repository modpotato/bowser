//! Symmetric cryptography for Bowser's hand-rolled TLS 1.2/1.3 stack.
//!
//! All primitives are implemented from scratch using only `std` — no
//! dependencies. This is security-critical code; it is written for clarity
//! and correctness first, with constant-time practices where they are
//! straightforward:
//!
//! - ChaCha20 and Poly1305 use only additions, XORs, multiplications and
//!   fixed rotations: no secret-dependent branches or memory accesses.
//! - GHASH field multiplication is branch-free (mask-based conditional XOR).
//! - Tag comparisons are constant-time (XOR accumulation).
//! - AES is table-driven (S-box lookups). Table lookups can leak through
//!   cache timing on shared hardware; a bitsliced AES is future work if the
//!   TLS reviewer demands it.
//!
//! Every primitive is covered by known-answer tests taken from its defining
//! specification (FIPS-197, NIST SP 800-38D, RFC 8439, FIPS-180-4,
//! RFC 5869, RFC 4231).

pub mod aes;
pub mod chacha20poly1305;
pub mod gcm;
pub mod hkdf;
pub mod sha2;

pub use aes::{Aes128, Aes256};
pub use chacha20poly1305::ChaCha20Poly1305;
pub use gcm::{Aes128Gcm, Aes256Gcm};
pub use hkdf::{HashAlg, hkdf_expand, hkdf_extract};
pub use sha2::{
    Sha256, Sha384, Sha512, hmac_sha256, hmac_sha384, hmac_sha512, sha256, sha384, sha512,
};

/// Errors returned by the symmetric crypto primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// AEAD authentication tag did not verify.
    AuthenticationFailed,
    /// An input had an invalid length (key, nonce, tag, or output length).
    InvalidLength,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::AuthenticationFailed => write!(f, "authentication tag mismatch"),
            Error::InvalidLength => write!(f, "invalid input length"),
        }
    }
}

impl std::error::Error for Error {}

/// Constant-time equality for secret byte strings (e.g. authentication tags).
///
/// Accumulates XOR differences so the runtime does not depend on where the
/// first difference is. Lengths are expected to match; a length mismatch
/// returns false.
pub(crate) fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

#[cfg(test)]
pub(crate) fn hex(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "odd-length hex string");
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for i in (0..bytes.len()).step_by(2) {
        let hi = hex_val(bytes[i]);
        let lo = hex_val(bytes[i + 1]);
        out.push(hi << 4 | lo);
    }
    out
}

#[cfg(test)]
fn hex_val(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => panic!("invalid hex digit"),
    }
}
