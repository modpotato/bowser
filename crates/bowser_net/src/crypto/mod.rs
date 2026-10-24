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
//! - AES uses AES-NI via `core::arch` on x86-64 CPUs that report the
//!   feature at runtime: the instructions run entirely on registers, so
//!   there are no secret-dependent branches or memory accesses. Other
//!   targets (and x86-64 without AES-NI) use a constant-time scalar
//!   fallback whose S-box lookups scan the full table behind a
//!   branch-free mask.
//!
//! Every primitive is covered by known-answer tests taken from its defining
//! specification (FIPS-197, NIST SP 800-38D, RFC 8439, FIPS-180-4,
//! RFC 5869, RFC 4231).

pub mod aes;
pub mod chacha20poly1305;
pub mod gcm;
pub mod hkdf;
pub mod sha2;

pub(crate) mod bignum;
pub(crate) mod ec;
pub(crate) mod rsa;
pub(crate) mod x509;

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

/// Overwrite `buf` with zeros through volatile writes, so the compiler
/// cannot elide the wipe as a dead store. Used to scrub key material on
/// drop. std-only: no external zeroize crate.
pub(crate) fn zeroize(buf: &mut [u8]) {
    for b in buf.iter_mut() {
        // SAFETY: `b` is a `&mut u8` into a live allocation, so the
        // pointer is valid, properly aligned, and exclusively borrowed
        // for the duration of the write; storing 0 is always a valid
        // `u8` value.
        unsafe {
            core::ptr::write_volatile(b, 0);
        }
    }
}

mod seal {
    /// Seal for [`Zeroizable`](super::Zeroizable): only this module can
    /// name it, so the trait cannot be implemented outside this file.
    pub trait Sealed {}
}

/// Marker for the plain-old-data types [`zeroize_val`] may wipe.
///
/// # Safety
///
/// Implementing types must be plain old data for which the all-zero bit
/// pattern is a valid value: integers and fixed-size arrays of them.
/// Anything with padding-sensitive invariants, pointers, references, or
/// a `Drop` impl must never implement this trait — `zeroize_val` writes
/// zeros over the whole value, which would corrupt such a type. The
/// trait is sealed, so the impls below are the complete list; adding one
/// requires editing this file and re-checking the contract.
pub(crate) unsafe trait Zeroizable: seal::Sealed {}

impl seal::Sealed for u64 {}
// SAFETY: `u64` is plain old data; all-zero is a valid value.
unsafe impl Zeroizable for u64 {}

impl seal::Sealed for u128 {}
// SAFETY: `u128` is plain old data; all-zero is a valid value.
unsafe impl Zeroizable for u128 {}

impl seal::Sealed for [u64; 2] {}
// SAFETY: a fixed-size array of `u64` is plain old data; all-zero is a
// valid value.
unsafe impl Zeroizable for [u64; 2] {}

impl seal::Sealed for [u64; 3] {}
// SAFETY: a fixed-size array of `u64` is plain old data; all-zero is a
// valid value.
unsafe impl Zeroizable for [u64; 3] {}

/// Volatile-wipe every byte of `val` (e.g. an integer or array holding key
/// material). The [`Zeroizable`] bound restricts this to the sealed list of
/// plain-old-data types above, where all-zero is a valid bit pattern, so
/// passing any other type fails to compile instead of corrupting it.
pub(crate) fn zeroize_val<T: Zeroizable>(val: &mut T) {
    // SAFETY: `val` is exclusively borrowed for the call, so the
    // `size_of::<T>()` bytes at its address are valid, aligned and
    // unaliased for the duration; `zeroize` writes only zeros through
    // them, and all-zero is a valid pattern for the covered types.
    let bytes = unsafe {
        core::slice::from_raw_parts_mut(val as *mut T as *mut u8, core::mem::size_of::<T>())
    };
    zeroize(bytes);
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

mod keccak;
pub mod ml_kem_768;
mod rand;
pub mod x25519;

pub(crate) use rand::fill_random;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroize_wipes_bytes() {
        let mut buf = [0xabu8; 32];
        zeroize(&mut buf);
        assert_eq!(buf, [0u8; 32]);
    }

    #[test]
    fn zeroize_val_wipes_integers() {
        let mut v = 0xdead_beef_cafe_f00du64;
        zeroize_val(&mut v);
        assert_eq!(v, 0);
        let mut limbs = [0xffff_ffff_ffff_ffffu64; 3];
        zeroize_val(&mut limbs);
        assert_eq!(limbs, [0u64; 3]);
    }
}
