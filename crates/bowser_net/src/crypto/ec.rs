//! ECDSA verification over NIST P-256 and P-384 (FIPS 186-4 §6.4.1).
//!
//! Verification only — no signing. Field arithmetic runs through
//! [`crate::crypto::bignum`]'s Montgomery multiplication; points use
//! affine coordinates (one Fermat inverse per add/double).
//!
//! # Constant-time posture
//!
//! Same as [`crate::crypto::bignum`]: verification inputs are public, so
//! variable-time double-and-add leaks nothing secret. Do not reuse this
//! for signing (which would need a constant-time, RFC 6979-style nonce).

use core::fmt;
use std::sync::OnceLock;

use super::bignum::{BigUint, Montgomery};
use super::sha2::{sha256, sha384};

/// Curves supported for ECDSA verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Curve {
    P256,
    P384,
}

/// Hash functions supported for ECDSA verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EcdsaHash {
    Sha256,
    Sha384,
}

/// Why ECDSA verification rejected its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EcdsaError {
    /// The public key is malformed or not on the curve.
    BadKey(&'static str),
    /// The signature is malformed or does not verify.
    BadSignature(&'static str),
}

impl fmt::Display for EcdsaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EcdsaError::BadKey(s) => write!(f, "bad EC key: {s}"),
            EcdsaError::BadSignature(s) => write!(f, "bad ECDSA signature: {s}"),
        }
    }
}

impl std::error::Error for EcdsaError {}

fn hex(s: &str) -> BigUint {
    assert!(s.len().is_multiple_of(2));
    let mut out = Vec::with_capacity(s.len() / 2);
    let b = s.as_bytes();
    for i in (0..b.len()).step_by(2) {
        let hv = |c: u8| match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => panic!("bad hex"),
        };
        out.push(hv(b[i]) << 4 | hv(b[i + 1]));
    }
    BigUint::from_be_bytes(&out)
}

struct CurveParams {
    /// Field modulus p (regular representation, for comparisons).
    p: BigUint,
    /// Montgomery context mod p.
    field: Montgomery,
    /// Montgomery context mod n (curve order).
    order: Montgomery,
    /// Order n (regular, for comparisons).
    n: BigUint,
    /// Generator, Montgomery form.
    gx: BigUint,
    gy: BigUint,
    /// Curve coefficient b, Montgomery form (a = -3 for both curves).
    b: BigUint,
    /// Field element size in bytes.
    field_len: usize,
}

fn params_p256() -> &'static CurveParams {
    static P: OnceLock<CurveParams> = OnceLock::new();
    P.get_or_init(|| {
        let p = hex("ffffffff00000001000000000000000000000000ffffffffffffffffffffffff");
        let field = Montgomery::new(&p).expect("p odd");
        let n = hex("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551");
        let order = Montgomery::new(&n).expect("n odd");
        CurveParams {
            gx: field.to_mont(&hex(
                "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296",
            )),
            gy: field.to_mont(&hex(
                "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5",
            )),
            b: field.to_mont(&hex(
                "5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b",
            )),
            p,
            field,
            order,
            n,
            field_len: 32,
        }
    })
}

fn params_p384() -> &'static CurveParams {
    static P: OnceLock<CurveParams> = OnceLock::new();
    P.get_or_init(|| {
        let p = hex("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffff0000000000000000ffffffff");
        let field = Montgomery::new(&p).expect("p odd");
        let n = hex("ffffffffffffffffffffffffffffffffffffffffffffffffc7634d81f4372ddf581a0db248b0a77aecec196accc52973");
        let order = Montgomery::new(&n).expect("n odd");
        CurveParams {
            gx: field.to_mont(&hex(
                "aa87ca22be8b05378eb1c71ef320ad746e1d3b628ba79b9859f741e082542a385502f25dbf55296c3a545e3872760ab7",
            )),
            gy: field.to_mont(&hex(
                "3617de4a96262c6f5d9e98bf9292dc29f8f41dbd289a147ce9da3113b5f0b8c00a60b1ce1d7e819d7a431d7c90ea0e5f",
            )),
            b: field.to_mont(&hex(
                "b3312fa7e23ee7e4988e056be3f82d19181d9c6efe8141120314088f5013875ac656398d8a2ed19d2a85c8edd3ec2aef",
            )),
            p,
            field,
            order,
            n,
            field_len: 48,
        }
    })
}

fn params(curve: Curve) -> &'static CurveParams {
    match curve {
        Curve::P256 => params_p256(),
        Curve::P384 => params_p384(),
    }
}

/// Affine point with coordinates in Montgomery form.
#[derive(Clone)]
struct Point {
    x: BigUint,
    y: BigUint,
    inf: bool,
}

impl Point {
    fn inf() -> Self {
        Point {
            x: BigUint::zero(),
            y: BigUint::zero(),
            inf: true,
        }
    }
}

struct FieldOps<'a> {
    cp: &'a CurveParams,
}

impl<'a> FieldOps<'a> {
    fn add(&self, a: &BigUint, b: &BigUint) -> BigUint {
        let mut c = a.add(b);
        if c.cmp(&self.cp.p) != core::cmp::Ordering::Less {
            c = c.sub(&self.cp.p).expect("c >= p");
        }
        c
    }

    fn sub(&self, a: &BigUint, b: &BigUint) -> BigUint {
        match a.cmp(b) {
            core::cmp::Ordering::Less => a.add(&self.cp.p).sub(b).expect("a + p >= b"),
            _ => a.sub(b).expect("a >= b"),
        }
    }

    fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        self.cp.field.mul(a, b)
    }

    fn inv(&self, a: &BigUint) -> BigUint {
        let reg = self.cp.field.demont(a);
        let inv = self.cp.field.inv(&reg);
        self.cp.field.to_mont(&inv)
    }

    /// Double `p` (affine). `y == 0` would need `inv(0)`; on these
    /// prime-order curves that only happens for the point at infinity,
    /// which is handled by the caller — treat defensively as infinity.
    fn double(&self, p: &Point) -> Point {
        if p.inf || p.y.is_zero() {
            return Point::inf();
        }
        // s = 3(x^2 - 1) / (2y), since a = -3.
        let x2 = self.mul(&p.x, &p.x);
        let three_x2m3 = {
            let t = self.sub(&x2, &self.cp.field.to_mont(&BigUint::one()));
            let two_t = self.add(&t, &t);
            self.add(&two_t, &t)
        };
        let two_y = self.add(&p.y, &p.y);
        let s = self.mul(&three_x2m3, &self.inv(&two_y));
        let s2 = self.mul(&s, &s);
        let two_x = self.add(&p.x, &p.x);
        let x3 = self.sub(&s2, &two_x);
        let y3 = self.sub(&self.mul(&s, &self.sub(&p.x, &x3)), &p.y);
        Point {
            x: x3,
            y: y3,
            inf: false,
        }
    }

    fn add_points(&self, p: &Point, q: &Point) -> Point {
        if p.inf {
            return q.clone();
        }
        if q.inf {
            return p.clone();
        }
        if p.x == q.x {
            if p.y == q.y {
                return self.double(p);
            }
            return Point::inf(); // p == -q
        }
        // s = (y2 - y1) / (x2 - x1)
        let s = self.mul(&self.sub(&q.y, &p.y), &self.inv(&self.sub(&q.x, &p.x)));
        let s2 = self.mul(&s, &s);
        let x3 = self.sub(&self.sub(&s2, &p.x), &q.x);
        let y3 = self.sub(&self.mul(&s, &self.sub(&p.x, &x3)), &p.y);
        Point {
            x: x3,
            y: y3,
            inf: false,
        }
    }

    /// Double-and-add scalar multiplication, MSB first.
    fn scalar_mul(&self, p: &Point, scalar: &BigUint) -> Point {
        let mut acc = Point::inf();
        for i in (0..scalar.bit_len()).rev() {
            acc = self.double(&acc);
            if scalar.get_bit(i) {
                acc = self.add_points(&acc, p);
            }
        }
        acc
    }
}

/// Verify an ECDSA signature `(r, s)` over `msg`.
///
/// `qx`/`qy` are the uncompressed public-key coordinates (fixed field-size
/// bytes each); `r`/`s` are big-endian integers that must satisfy
/// `1 <= r, s < n`. The public key is validated to lie on the curve.
pub(crate) fn verify_ecdsa(
    curve: Curve,
    qx_be: &[u8],
    qy_be: &[u8],
    msg: &[u8],
    r_be: &[u8],
    s_be: &[u8],
    hash: EcdsaHash,
) -> Result<(), EcdsaError> {
    let cp = params(curve);
    let f = FieldOps { cp };
    if qx_be.len() != cp.field_len || qy_be.len() != cp.field_len {
        return Err(EcdsaError::BadKey("bad coordinate length"));
    }
    let x = BigUint::from_be_bytes(qx_be);
    let y = BigUint::from_be_bytes(qy_be);
    if x.cmp(&cp.p) != core::cmp::Ordering::Less || y.cmp(&cp.p) != core::cmp::Ordering::Less {
        return Err(EcdsaError::BadKey("coordinate out of range"));
    }
    // Q must satisfy y^2 = x^3 - 3x + b.
    let xm = cp.field.to_mont(&x);
    let ym = cp.field.to_mont(&y);
    let lhs = f.mul(&ym, &ym);
    let x2 = f.mul(&xm, &xm);
    let x3 = f.mul(&x2, &xm);
    let three_x = {
        let t = f.add(&xm, &xm);
        f.add(&t, &xm)
    };
    let rhs = f.add(&f.sub(&x3, &three_x), &cp.b);
    if lhs != rhs {
        return Err(EcdsaError::BadKey("point not on curve"));
    }

    let r = BigUint::from_be_bytes(r_be);
    let s = BigUint::from_be_bytes(s_be);
    let one = BigUint::one();
    for v in [&r, &s] {
        if v.cmp(&one) == core::cmp::Ordering::Less || v.cmp(&cp.n) != core::cmp::Ordering::Less {
            return Err(EcdsaError::BadSignature("r/s out of range"));
        }
    }

    // e = leftmost min(N, hashBits) bits of H(m).
    let digest: Vec<u8> = match hash {
        EcdsaHash::Sha256 => sha256(msg).to_vec(),
        EcdsaHash::Sha384 => sha384(msg).to_vec(),
    };
    let e = truncate_to_bits(&digest, cp.n.bit_len()).rem(&cp.n);

    // w = s^-1 mod n; u1 = e*w mod n; u2 = r*w mod n.
    let w = cp.order.inv(&s);
    let mul_n = |a: &BigUint, b: &BigUint| {
        cp.order
            .demont(&cp.order.mul(&cp.order.to_mont(a), &cp.order.to_mont(b)))
    };
    let u1 = mul_n(&e, &w);
    let u2 = mul_n(&r, &w);

    let g = Point {
        x: cp.gx.clone(),
        y: cp.gy.clone(),
        inf: false,
    };
    let q = Point {
        x: xm,
        y: ym,
        inf: false,
    };
    let p1 = f.scalar_mul(&g, &u1);
    let p2 = f.scalar_mul(&q, &u2);
    let pt = f.add_points(&p1, &p2);
    if pt.inf {
        return Err(EcdsaError::BadSignature("point at infinity"));
    }
    let x1 = cp.field.demont(&pt.x).rem(&cp.n);
    // Both x1 and r are public; a direct comparison is fine.
    if x1 == r {
        Ok(())
    } else {
        Err(EcdsaError::BadSignature("x1 != r"))
    }
}

/// Keep the leftmost `nbits` of `digest` (FIPS 186-4 §6.4.1: bits2int).
fn truncate_to_bits(digest: &[u8], nbits: usize) -> BigUint {
    let keep = nbits.div_ceil(8).min(digest.len());
    let mut b = digest[..keep].to_vec();
    // Short digest (outlen < N): bits2int uses the whole digest, so
    // drop nothing. The subtraction must not underflow: e.g. P-384
    // (nbits=384) with a SHA-256 digest (keep=32) gives
    // 32*8-384 < 0, which panicked in debug — remotely triggerable
    // via a crafted chain (leaf ecdsa-with-SHA256 under a P-384 key).
    let excess = (keep * 8).saturating_sub(nbits); // 0..8 low bits to drop
    if excess > 0 {
        // Big-endian right shift by `excess`: the low `excess` bits
        // of each byte fall into the top of the next (less
        // significant) byte. (Iterating from the most significant
        // byte; the reverse direction would shift the wrong way.)
        let mut carry = 0u8;
        for byte in b.iter_mut() {
            let new_carry = (*byte & ((1u8 << excess) - 1)) << (8 - excess);
            *byte = (*byte >> excess) | carry;
            carry = new_carry;
        }
    }
    BigUint::from_be_bytes(&b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::hex as test_hex;

    /// openssl-generated ECDSA KAT row: `hash / curve / qx / qy / msg / r / s`.
    type EcdsaKat = (
        EcdsaHash,
        Curve,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
    );

    fn unhex(s: &str) -> Vec<u8> {
        test_hex(s)
    }

    /// RFC 6979 §A.2.5: P-256/SHA-256, message "sample". Q is derived as x*G
    /// with the module's own scalar multiplication.
    #[test]
    fn rfc6979_p256_vector() {
        let cp = params(Curve::P256);
        let f = FieldOps { cp };
        let x = super::hex("c9afa9d845ba75166b5c215767b1d6934e50c3db36e89b127b8a622b120f6721");
        let g = Point {
            x: cp.gx.clone(),
            y: cp.gy.clone(),
            inf: false,
        };
        let q = f.scalar_mul(&g, &x);
        assert!(!q.inf);
        let qx = cp.field.demont(&q.x).to_be_bytes_padded(32).unwrap();
        let qy = cp.field.demont(&q.y).to_be_bytes_padded(32).unwrap();
        verify_ecdsa(
            Curve::P256,
            &qx,
            &qy,
            b"sample",
            &unhex("efd48b2aacb6a8fd1140dd9cd45e81d69d2c877b56aaf991c34d0ea84eaf3716"),
            &unhex("f7cb1c942d657c41d436c7a1b6e29f65f3e900dbb9aff4064dc4ab2f843acda8"),
            EcdsaHash::Sha256,
        )
        .expect("RFC 6979 A.1 signature must verify");

        // Wrong message fails.
        assert!(matches!(
            verify_ecdsa(
                Curve::P256,
                &qx,
                &qy,
                b"sample!",
                &unhex("efd48b2aacb6a8fd1140dd9cd45e81d69d2c877b56aaf991c34d0ea84eaf3716"),
                &unhex("f7cb1c942d657c41d436c7a1b6e29f65f3e900dbb9aff4064dc4ab2f843acda8"),
                EcdsaHash::Sha256,
            ),
            Err(EcdsaError::BadSignature(_))
        ));
    }

    /// openssl-generated KATs: `hash / curve / qx / qy / msg / r / s`.
    fn load_kats() -> Vec<EcdsaKat> {
        let text = include_str!("../../tests/fixtures/tls/ecdsa_kat.txt");
        let mut out = Vec::new();
        let (mut hash, mut curve) = (EcdsaHash::Sha256, Curve::P256);
        let (mut qx, mut qy, mut msg, mut r, mut s) = (vec![], vec![], vec![], vec![], vec![]);
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
                        out.push((hash, curve, qx, qy, msg, r, s));
                    }
                    hash = match v {
                        "sha256" => EcdsaHash::Sha256,
                        "sha384" => EcdsaHash::Sha384,
                        _ => panic!("bad hash {v}"),
                    };
                    (qx, qy, msg, r, s) = (vec![], vec![], vec![], vec![], vec![]);
                    have = true;
                }
                "curve" => {
                    curve = match v {
                        "p256" => Curve::P256,
                        "p384" => Curve::P384,
                        _ => panic!("bad curve {v}"),
                    }
                }
                "qx" => qx = unhex(v),
                "qy" => qy = unhex(v),
                "msg" => msg = unhex(v),
                "r" => r = unhex(v),
                "s" => s = unhex(v),
                _ => panic!("bad kat key {k}"),
            }
        }
        if have {
            out.push((hash, curve, qx, qy, msg, r, s));
        }
        assert!(!out.is_empty(), "no KATs loaded");
        out
    }

    #[test]
    fn openssl_kat_vectors_verify() {
        for (hash, curve, qx, qy, msg, r, s) in load_kats() {
            verify_ecdsa(curve, &qx, &qy, &msg, &r, &s, hash)
                .expect("openssl-generated signature must verify");
        }
    }

    #[test]
    fn tampered_openssl_signature_rejected() {
        let (hash, curve, qx, qy, msg, r, mut s) = load_kats().remove(0);
        let i = s.len() / 2;
        s[i] ^= 0x01;
        assert!(matches!(
            verify_ecdsa(curve, &qx, &qy, &msg, &r, &s, hash),
            Err(EcdsaError::BadSignature(_))
        ));
    }

    #[test]
    fn bad_keys_rejected() {
        let (hash, curve, qx, qy, msg, r, s) = load_kats().remove(0);
        // Point not on the curve (flip qy).
        let mut qy_bad = qy.clone();
        qy_bad[0] ^= 0x01;
        assert!(matches!(
            verify_ecdsa(curve, &qx, &qy_bad, &msg, &r, &s, hash),
            Err(EcdsaError::BadKey(_))
        ));
        // Coordinate out of range.
        let cp = params(curve);
        let p_bytes = cp.p.to_be_bytes_padded(cp.field_len).unwrap();
        assert!(matches!(
            verify_ecdsa(curve, &p_bytes, &qy, &msg, &r, &s, hash),
            Err(EcdsaError::BadKey(_))
        ));
        // r = 0, s = n.
        let zero = vec![0u8; r.len()];
        assert!(matches!(
            verify_ecdsa(curve, &qx, &qy, &msg, &zero, &s, hash),
            Err(EcdsaError::BadSignature(_))
        ));
        let n_bytes = cp.n.to_be_bytes_padded(cp.field_len).unwrap();
        assert!(matches!(
            verify_ecdsa(curve, &qx, &qy, &msg, &r, &n_bytes, hash),
            Err(EcdsaError::BadSignature(_))
        ));
    }

    #[test]
    fn truncate_to_bits_short_digest() {
        // C1 regression: a digest shorter than nbits must use the whole
        // digest (FIPS 186-4 §6.4.1 bits2int). The old code computed
        // `keep * 8 - nbits` unconditionally, which underflows usize
        // (e.g. P-384 nbits=384 with a 32-byte SHA-256 digest) and
        // panics in debug — remotely triggerable via a crafted chain
        // (leaf ecdsa-with-SHA256 under a P-384 issuer key).
        let ff32 = [0xFFu8; 32];
        // Short digest: whole digest used, nothing dropped.
        assert_eq!(truncate_to_bits(&ff32, 384).to_be_bytes(), ff32);
        // Exact fit.
        assert_eq!(truncate_to_bits(&ff32, 256).to_be_bytes(), ff32);
        // nbits not a multiple of 8: drop the low 6 bits → 2^250 - 1.
        let mut want = vec![0x03u8];
        want.extend_from_slice(&[0xFFu8; 31]);
        assert_eq!(truncate_to_bits(&ff32, 250).to_be_bytes(), want);
        // Empty digest → zero (also underflowed before the fix).
        assert!(truncate_to_bits(&[], 256).is_zero());
    }

    #[test]
    fn p384_sha256_kat_verifies() {
        // C1 cross-combo: openssl-generated P-384 / ecdsa-with-SHA256
        // vector. No existing test covered a hash shorter than the
        // curve order; this exercises truncate_to_bits end to end.
        let qx = unhex(
            "66d24ca055ec668af5f20ea734b9f9895d26308649317ebd75e199f9bda6d641f03d7a4727a8e4561e1dc46d4ab3ddee",
        );
        let qy = unhex(
            "5fdd56ced32bc55c6cbef58ae2350c1cefeb78a5f2a4fdeb35dfce77ffe33b3d02b4abd08df1b93fd4a423b330b09bc5",
        );
        let msg = unhex("703338342f7368613235362063726f73732d636f6d626f206b6174");
        let r = unhex(
            "31a0c86a9358feb6f25da5781d5c7cbae9947ccef3f82a29e79af09516436c97a74fdb0d737b7e12171485c6da87dc7a",
        );
        let s = unhex(
            "0081e4a6f3af053065fb2153c42281b9cd8fb1f81cf6268604e4d718b43c81e1c18c4228258d1d8590090d86ee376da805",
        );
        verify_ecdsa(Curve::P384, &qx, &qy, &msg, &r, &s, EcdsaHash::Sha256)
            .expect("P-384/SHA-256 KAT signature must verify");
        // Wrong message fails.
        let mut bad_msg = msg.clone();
        bad_msg[0] ^= 0x01;
        assert!(matches!(
            verify_ecdsa(Curve::P384, &qx, &qy, &bad_msg, &r, &s, EcdsaHash::Sha256),
            Err(EcdsaError::BadSignature(_))
        ));
    }

    #[test]
    fn scalar_mul_generator_order_is_infinity() {
        // n*G must be the point at infinity on both curves.
        for curve in [Curve::P256, Curve::P384] {
            let cp = params(curve);
            let f = FieldOps { cp };
            let g = Point {
                x: cp.gx.clone(),
                y: cp.gy.clone(),
                inf: false,
            };
            let inf = f.scalar_mul(&g, &cp.n);
            assert!(inf.inf, "{curve:?}: n*G should be infinity");
            // 1*G == G.
            let one_g = f.scalar_mul(&g, &BigUint::one());
            assert!(!one_g.inf && one_g.x == g.x && one_g.y == g.y);
        }
    }
}
