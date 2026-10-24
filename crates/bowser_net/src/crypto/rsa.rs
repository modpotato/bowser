//! RSA PKCS#1 v1.5 signature verification (RFC 8017 §8.2.2).
//!
//! Verification only — no signing, no decryption. Built on
//! [`crate::crypto::bignum`]'s Montgomery exponentiation.
//!
//! # Constant-time posture
//!
//! All inputs to verification (modulus, exponent, message, signature) are
//! public, so the variable-time square-and-multiply in `bignum` leaks
//! nothing secret. The final encoded-message comparison uses
//! constant-time equality as defense-in-depth.

use core::fmt;

use super::bignum::{BigUint, Montgomery};
use super::ct_eq;
use super::sha2::{sha256, sha384, sha512};

/// Hash functions supported for PKCS#1 v1.5 verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RsaHash {
    Sha256,
    Sha384,
    Sha512,
}

/// Maximum RSA modulus size accepted by verification, in bits.
///
/// Deployed RSA keys are at most 4096 bits; this bound is deliberately
/// generous to avoid rejecting legitimate keys. It exists to cap the
/// CPU cost of [`Montgomery`] exponentiation against a malicious peer
/// presenting an absurdly large (tens of KB) modulus.
pub(crate) const MAX_RSA_MODULUS_BITS: usize = 16384;

/// DER `DigestInfo` prefixes for EMSA-PKCS1-v1_5 (RFC 8017 §9.2, notes).
/// Each is followed directly by the hash output `H`.
const DIGESTINFO_SHA256: &[u8] = &[
    0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05,
    0x00, 0x04, 0x20,
];
const DIGESTINFO_SHA384: &[u8] = &[
    0x30, 0x41, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02, 0x05,
    0x00, 0x04, 0x30,
];
const DIGESTINFO_SHA512: &[u8] = &[
    0x30, 0x51, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03, 0x05,
    0x00, 0x04, 0x40,
];

/// Why RSA verification rejected its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RsaError {
    /// The public key is malformed (bad modulus/exponent).
    BadKey(&'static str),
    /// The signature is malformed or does not verify.
    BadSignature(&'static str),
}

impl fmt::Display for RsaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RsaError::BadKey(s) => write!(f, "bad RSA key: {s}"),
            RsaError::BadSignature(s) => write!(f, "bad RSA signature: {s}"),
        }
    }
}

impl std::error::Error for RsaError {}

fn hash_message(hash: RsaHash, msg: &[u8]) -> (Vec<u8>, &'static [u8]) {
    match hash {
        RsaHash::Sha256 => (sha256(msg).to_vec(), DIGESTINFO_SHA256),
        RsaHash::Sha384 => (sha384(msg).to_vec(), DIGESTINFO_SHA384),
        RsaHash::Sha512 => (sha512(msg).to_vec(), DIGESTINFO_SHA512),
    }
}

/// Verify an RSASSA-PKCS1-v1_5 signature: `sig` over `msg` with public key
/// `(n, e)`, all big-endian.
///
/// `sig` must be exactly the modulus length in bytes; a shorter value is
/// left-padded (some encoders drop leading zeros), a longer one is
/// rejected. The public exponent must satisfy `3 <= e < n`: `e = 1`
/// would make verification trivially forgeable and is rejected outright.
pub(crate) fn verify_pkcs1_v15(
    n_be: &[u8],
    e_be: &[u8],
    msg: &[u8],
    sig_be: &[u8],
    hash: RsaHash,
) -> Result<(), RsaError> {
    let n = BigUint::from_be_bytes(n_be);
    let e = BigUint::from_be_bytes(e_be);
    let k = n_be.len();
    if n.bit_len() > MAX_RSA_MODULUS_BITS {
        return Err(RsaError::BadKey("modulus too large"));
    }
    let mont = Montgomery::new(&n).ok_or(RsaError::BadKey("modulus must be odd and > 1"))?;
    let three = BigUint::from_u64(3);
    if e.cmp(&three) == core::cmp::Ordering::Less || e.cmp(&n) != core::cmp::Ordering::Less {
        return Err(RsaError::BadKey("exponent out of range"));
    }

    let (digest, prefix) = hash_message(hash, msg);
    let t_len = prefix.len() + digest.len();
    // RFC 8017 §9.2: emLen >= tLen + 11 (8 octets of 0xFF padding minimum).
    if k < t_len + 11 {
        return Err(RsaError::BadKey("modulus too short for hash"));
    }
    if sig_be.len() > k {
        return Err(RsaError::BadSignature("signature longer than modulus"));
    }
    let mut sig_padded = vec![0u8; k - sig_be.len()];
    sig_padded.extend_from_slice(sig_be);

    // RFC 8017 §8.2.2 step 1: the signature representative must be in
    // [0, n-1]. RSAVP1 would silently reduce an out-of-range s mod n.
    let s = BigUint::from_be_bytes(&sig_padded);
    if s.cmp(&n) != core::cmp::Ordering::Less {
        return Err(RsaError::BadSignature(
            "signature representative out of range",
        ));
    }

    // RSAVP1: m = s^e mod n.
    let m = mont.pow(&s, &e);
    let em = m
        .to_be_bytes_padded(k)
        .ok_or(RsaError::BadSignature("decrypted block too long"))?;

    // EMSA-PKCS1-v1_5 encoding: 0x00 || 0x01 || PS || 0x00 || DigestInfo.
    let mut expected = vec![0u8; k];
    expected[0] = 0x00;
    expected[1] = 0x01;
    let ps_end = k - t_len - 1;
    expected[2..ps_end].fill(0xff);
    expected[ps_end] = 0x00;
    expected[ps_end + 1..ps_end + 1 + prefix.len()].copy_from_slice(prefix);
    expected[ps_end + 1 + prefix.len()..].copy_from_slice(&digest);

    if ct_eq(&em, &expected) {
        Ok(())
    } else {
        Err(RsaError::BadSignature("encoding mismatch"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::hex;

    /// openssl-generated RSA KAT row: `hash / n / e / msg / sig`.
    type RsaKat = (RsaHash, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

    /// Parse the openssl-generated KAT fixture: blocks of
    /// `name: <hash>\nn: <hex>\ne: <hex>\nmsg: <hex>\nsig: <hex>`.
    fn load_kats() -> Vec<RsaKat> {
        let text = include_str!("../../tests/fixtures/tls/rsa_kat.txt");
        let mut out = Vec::new();
        let (mut hash, mut n, mut e, mut msg, mut sig) =
            (RsaHash::Sha256, vec![], vec![], vec![], vec![]);
        let mut have = false;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (k, v) = line.split_once(": ").expect("malformed kat line");
            match k {
                "hash" => {
                    if have {
                        out.push((hash, n, e, msg, sig));
                    }
                    hash = match v {
                        "sha256" => RsaHash::Sha256,
                        "sha384" => RsaHash::Sha384,
                        "sha512" => RsaHash::Sha512,
                        _ => panic!("bad hash {v}"),
                    };
                    (n, e, msg, sig) = (vec![], vec![], vec![], vec![]);
                    have = true;
                }
                "n" => n = hex(v),
                "e" => e = hex(v),
                "msg" => msg = hex(v),
                "sig" => sig = hex(v),
                _ => panic!("bad kat key {k}"),
            }
        }
        if have {
            out.push((hash, n, e, msg, sig));
        }
        assert!(!out.is_empty(), "no KATs loaded");
        out
    }

    #[test]
    fn openssl_kat_vectors_verify() {
        for (hash, n, e, msg, sig) in load_kats() {
            verify_pkcs1_v15(&n, &e, &msg, &sig, hash)
                .expect("openssl-generated signature must verify");
        }
    }

    #[test]
    fn tampered_signature_rejected() {
        let (hash, n, e, msg, mut sig) = load_kats().remove(0);
        // Flip a bit in the middle of the signature.
        let i = sig.len() / 2;
        sig[i] ^= 0x01;
        assert!(matches!(
            verify_pkcs1_v15(&n, &e, &msg, &sig, hash),
            Err(RsaError::BadSignature(_))
        ));
        // Flip a bit in the message instead.
        let (hash, n, e, mut msg, sig) = load_kats().remove(0);
        msg[0] ^= 0x01;
        assert!(matches!(
            verify_pkcs1_v15(&n, &e, &msg, &sig, hash),
            Err(RsaError::BadSignature(_))
        ));
    }

    #[test]
    fn bad_keys_rejected() {
        let (hash, n, e, msg, sig) = load_kats().remove(0);
        // Exponent 1 would make verification trivially forgeable.
        assert!(matches!(
            verify_pkcs1_v15(&n, &[0x01], &msg, &sig, hash),
            Err(RsaError::BadKey(_))
        ));
        // Even modulus.
        let mut n_even = n.clone();
        let l = n_even.len() - 1;
        n_even[l] &= 0xfe;
        assert!(matches!(
            verify_pkcs1_v15(&n_even, &e, &msg, &sig, hash),
            Err(RsaError::BadKey(_))
        ));
        // Signature longer than the modulus.
        let mut long_sig = vec![0u8];
        long_sig.extend_from_slice(&sig);
        assert!(matches!(
            verify_pkcs1_v15(&n, &e, &msg, &long_sig, hash),
            Err(RsaError::BadSignature(_))
        ));
        // Modulus too short for the hash (64-byte toy modulus, SHA-512).
        let tiny_n = vec![0xabu8; 64];
        assert!(matches!(
            verify_pkcs1_v15(&tiny_n, &e, &msg, &sig, RsaHash::Sha512),
            Err(RsaError::BadKey(_))
        ));
    }

    #[test]
    fn oversized_modulus_rejected_fast() {
        let (hash, _, e, msg, sig) = load_kats().remove(0);
        // 16400-bit modulus: must be rejected by the bit bound before
        // any Montgomery setup, not after minutes of exponentiation.
        let big_n = vec![0x7f; 2050];
        let start = std::time::Instant::now();
        assert!(matches!(
            verify_pkcs1_v15(&big_n, &e, &msg, &sig, hash),
            Err(RsaError::BadKey(_))
        ));
        assert!(
            start.elapsed().as_secs() < 5,
            "oversized modulus must fail fast"
        );

        // Exactly 16384 bits is still accepted by the bound (fails
        // later, for a non-size reason, on this synthetic key).
        let boundary_n = vec![0xabu8; 2048];
        assert!(matches!(
            verify_pkcs1_v15(&boundary_n, &e, &msg, &sig, hash),
            Err(RsaError::BadSignature(_))
        ));
    }

    #[test]
    fn signature_at_or_above_modulus_rejected() {
        let (hash, n, e, msg, _) = load_kats().remove(0);
        let k = n.len();
        // s == n: RFC 8017 §8.2.2 step 1 rejects, not reduces.
        assert!(matches!(
            verify_pkcs1_v15(&n, &e, &msg, &n, hash),
            Err(RsaError::BadSignature(_))
        ));
        // s > n but still k bytes (KAT moduli carry a leading zero byte).
        let mut big_s = n.clone();
        big_s[0] = 0x7f;
        assert_eq!(big_s.len(), k);
        assert!(matches!(
            verify_pkcs1_v15(&n, &e, &msg, &big_s, hash),
            Err(RsaError::BadSignature(_))
        ));
        // s == 0 is in range and proceeds to the encoding check.
        assert!(matches!(
            verify_pkcs1_v15(&n, &e, &msg, &[0u8], hash),
            Err(RsaError::BadSignature(_))
        ));
    }

    #[test]
    fn short_signature_left_padded() {
        // A signature with a dropped leading zero byte still verifies.
        let (hash, n, e, msg, sig) = load_kats().remove(0);
        assert!(
            sig[0] == 0,
            "KAT fixture must have a leading-zero signature"
        );
        let short = &sig[1..];
        verify_pkcs1_v15(&n, &e, msg.as_slice(), short, hash)
            .expect("short signature must be left-padded");
    }

    #[test]
    fn digestinfo_prefixes() {
        // RFC 8017 §9.2 note vectors (AlgorithmIdentifier encodings).
        assert_eq!(
            DIGESTINFO_SHA256,
            &hex("3031300d060960864801650304020105000420")
        );
        assert_eq!(
            DIGESTINFO_SHA384,
            &hex("3041300d060960864801650304020205000430")
        );
        assert_eq!(
            DIGESTINFO_SHA512,
            &hex("3051300d060960864801650304020305000440")
        );
    }
}
