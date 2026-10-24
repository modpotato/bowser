//! HKDF: HMAC-based Extract-and-Expand key derivation (RFC 5869).
//!
//! Used by TLS 1.3 for its entire key schedule.

use super::Error;
use super::sha2::{hmac_sha256, hmac_sha384, hmac_sha512};

/// Hash function selecting the HKDF/HMAC instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlg {
    /// HKDF-SHA-256 (TLS 1.3, RFC 8446 Section 7.1).
    Sha256,
    /// HKDF-SHA-384 (TLS 1.3).
    Sha384,
    /// HKDF-SHA-512.
    Sha512,
}

impl HashAlg {
    fn hash_len(self) -> usize {
        match self {
            HashAlg::Sha256 => 32,
            HashAlg::Sha384 => 48,
            HashAlg::Sha512 => 64,
        }
    }

    fn hmac(self, key: &[u8], msg: &[u8]) -> Vec<u8> {
        match self {
            HashAlg::Sha256 => hmac_sha256(key, msg).to_vec(),
            HashAlg::Sha384 => hmac_sha384(key, msg).to_vec(),
            HashAlg::Sha512 => hmac_sha512(key, msg).to_vec(),
        }
    }
}

/// HKDF-Extract: `PRK = HMAC-Hash(salt, IKM)`.
///
/// A missing salt is replaced by a string of `HashLen` zeros, per RFC 5869.
pub fn hkdf_extract(alg: HashAlg, salt: &[u8], ikm: &[u8]) -> Vec<u8> {
    let zeros;
    let salt = if salt.is_empty() {
        zeros = vec![0u8; alg.hash_len()];
        &zeros
    } else {
        salt
    };
    alg.hmac(salt, ikm)
}

/// HKDF-Expand: derive `len` bytes of keying material from `prk`.
///
/// Returns [`Error::InvalidLength`] if `prk` is shorter than `HashLen` or
/// `len` exceeds `255 * HashLen` (RFC 5869, Section 2.3).
pub fn hkdf_expand(alg: HashAlg, prk: &[u8], info: &[u8], len: usize) -> Result<Vec<u8>, Error> {
    let hash_len = alg.hash_len();
    if prk.len() < hash_len {
        return Err(Error::InvalidLength);
    }
    let n = len.div_ceil(hash_len);
    if n > 255 {
        return Err(Error::InvalidLength);
    }
    let mut okm = Vec::with_capacity(len);
    let mut t = Vec::new();
    for i in 1..=n {
        let mut msg = Vec::with_capacity(t.len() + info.len() + 1);
        msg.extend_from_slice(&t);
        msg.extend_from_slice(info);
        msg.push(i as u8);
        t = alg.hmac(prk, &msg);
        okm.extend_from_slice(&t);
    }
    okm.truncate(len);
    Ok(okm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::hex;

    fn hx(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn hkdf_rfc5869_case1() {
        // RFC 5869, Appendix A.1.
        let ikm = hex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
        let salt = hex("000102030405060708090a0b0c");
        let info = hex("f0f1f2f3f4f5f6f7f8f9");
        let prk = hkdf_extract(HashAlg::Sha256, &salt, &ikm);
        assert_eq!(
            hx(&prk),
            "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"
        );
        let okm = hkdf_expand(HashAlg::Sha256, &prk, &info, 42).unwrap();
        assert_eq!(
            hx(&okm),
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
        );
    }

    #[test]
    fn hkdf_rfc5869_case2() {
        // RFC 5869, Appendix A.2: longer inputs/outputs.
        let ikm = hex(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f",
        );
        let salt = hex(
            "606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeaf",
        );
        let info = hex(
            "b0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff",
        );
        let prk = hkdf_extract(HashAlg::Sha256, &salt, &ikm);
        assert_eq!(
            hx(&prk),
            "06a6b88c5853361a06104c9ceb35b45cef760014904671014a193f40c15fc244"
        );
        let okm = hkdf_expand(HashAlg::Sha256, &prk, &info, 82).unwrap();
        assert_eq!(
            hx(&okm),
            "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c59045a99cac7827271cb41c65e590e09da3275600c2f09b8367793a9aca3db71cc30c58179ec3e87c14c01d5c1f3434f1d87"
        );
    }

    #[test]
    fn hkdf_empty_salt_uses_zeros() {
        // Empty salt must equal an explicit zero salt (RFC 5869, Section 2.2).
        let ikm = hex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
        let a = hkdf_extract(HashAlg::Sha256, &[], &ikm);
        let b = hkdf_extract(HashAlg::Sha256, &[0u8; 32], &ikm);
        assert_eq!(a, b);
    }

    #[test]
    fn hkdf_expand_limits() {
        let prk = hkdf_extract(HashAlg::Sha256, b"salt", b"ikm");
        assert!(hkdf_expand(HashAlg::Sha256, &prk, b"info", 255 * 32).is_ok());
        assert_eq!(
            hkdf_expand(HashAlg::Sha256, &prk, b"info", 255 * 32 + 1),
            Err(Error::InvalidLength)
        );
        assert_eq!(
            hkdf_expand(HashAlg::Sha256, &prk[..16], b"info", 32),
            Err(Error::InvalidLength)
        );
        assert_eq!(
            hkdf_expand(HashAlg::Sha256, &prk, b"info", 0).unwrap(),
            Vec::<u8>::new()
        );
    }

    #[test]
    fn hkdf_expand_prefix_property() {
        // Shorter OKMs are prefixes of longer ones with the same inputs.
        let prk = hkdf_extract(HashAlg::Sha384, b"salt", b"ikm");
        let long = hkdf_expand(HashAlg::Sha384, &prk, b"info", 100).unwrap();
        let short = hkdf_expand(HashAlg::Sha384, &prk, b"info", 37).unwrap();
        assert_eq!(&long[..37], &short[..]);
    }

    #[test]
    fn hkdf_sha512_roundtrip() {
        let prk = hkdf_extract(HashAlg::Sha512, b"s", b"i");
        assert_eq!(prk.len(), 64);
        let okm = hkdf_expand(HashAlg::Sha512, &prk, b"", 64).unwrap();
        assert_eq!(okm.len(), 64);
    }
}
