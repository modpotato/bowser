//! X25519 Diffie-Hellman key exchange ([RFC 7748](https://www.rfc-editor.org/rfc/rfc7748)).
//!
//! Field arithmetic uses five 51-bit limbs (radix 2^51). The Montgomery
//! ladder runs in constant time: no secret-dependent branches and no
//! secret-dependent memory accesses.

use std::io;

use super::rand::fill_random;

/// Number of bytes in an X25519 secret, public key, or shared secret.
pub const KEY_LEN: usize = 32;

const MASK51: u64 = (1u64 << 51) - 1;
const MASK51_128: u128 = MASK51 as u128;

/// `(p-2)` as `(2^255 - 21)`: the exponent used for inversion.
/// Bits 254..5 are 1; bits 4..0 are `0b01011` (`0xEB`).
const INV_LOW_BITS: u8 = 0xEB;

/// X25519 base point: `u = 9`.
const BASEPOINT: [u8; 32] = [
    9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Field element modulo `p = 2^255 - 19`, five radix-2^51 limbs.
///
/// After every operation the limbs are normalized to `< 2^51 + 2^9`, which
/// keeps all intermediate products inside `u128`.
#[derive(Clone, Copy)]
struct Fe([u64; 5]);

/// Limbs of `p = 2^255 - 19` in radix 2^51.
const P_LIMBS: [u64; 5] = [MASK51 - 18, MASK51, MASK51, MASK51, MASK51];

/// Normalize limbs: carry base-2^51 with the fold `2^255 -> 19`.
///
/// Preserves the value modulo `p`. Eight passes take any entry with limbs
/// `< 2^110` down to limbs `< 2^51 + 2^9`, which is the invariant every
/// [`Fe`] operation restores.
fn normalize(r: &mut [u128; 5]) {
    for _ in 0..8 {
        let mut c = r[0] >> 51;
        r[0] &= MASK51_128;
        r[1] += c;
        c = r[1] >> 51;
        r[1] &= MASK51_128;
        r[2] += c;
        c = r[2] >> 51;
        r[2] &= MASK51_128;
        r[3] += c;
        c = r[3] >> 51;
        r[3] &= MASK51_128;
        r[4] += c;
        c = r[4] >> 51;
        r[4] &= MASK51_128;
        r[0] += 19 * c;
    }
}

impl Fe {
    fn zero() -> Self {
        Self([0; 5])
    }

    fn one() -> Self {
        Self([1, 0, 0, 0, 0])
    }

    fn load8(s: &[u8; 32], i: usize) -> u64 {
        (s[i] as u64)
            | ((s[i + 1] as u64) << 8)
            | ((s[i + 2] as u64) << 16)
            | ((s[i + 3] as u64) << 24)
            | ((s[i + 4] as u64) << 32)
            | ((s[i + 5] as u64) << 40)
            | ((s[i + 6] as u64) << 48)
            | ((s[i + 7] as u64) << 56)
    }

    fn from_bytes(s: &[u8; 32]) -> Self {
        Self([
            Self::load8(s, 0) & MASK51,
            (Self::load8(s, 6) >> 3) & MASK51,
            (Self::load8(s, 12) >> 6) & MASK51,
            (Self::load8(s, 19) >> 1) & MASK51,
            (Self::load8(s, 24) >> 12) & MASK51,
        ])
    }

    /// Canonical little-endian encoding in `[0, p)`.
    fn to_bytes(self) -> [u8; 32] {
        let mut r = [
            self.0[0] as u128,
            self.0[1] as u128,
            self.0[2] as u128,
            self.0[3] as u128,
            self.0[4] as u128,
        ];
        normalize(&mut r);
        // Exact base-2^51 digits plus a possible overflow digit. The value
        // is below `2*p`, so one conditional subtraction of `p` suffices.
        let mut d = [0u64; 6];
        let mut c: u64 = 0;
        for i in 0..5 {
            let t = r[i] as u64 + c;
            d[i] = t & MASK51;
            c = t >> 51;
        }
        d[5] = c;
        // `sub = d - p` with borrow; `borrow_out == 1` iff `d < p`.
        let mut sub = [0u64; 6];
        let mut borrow: i64 = 0;
        for i in 0..6 {
            let t = d[i] as i64 - P_LIMBS.get(i).copied().unwrap_or(0) as i64 - borrow;
            sub[i] = (t as u64) & MASK51;
            borrow = i64::from(t < 0);
        }
        // The value is below `2*p`, so one conditional subtraction of `p`
        // suffices; `borrow == 1` (i.e. `d < p`) is fine and common.
        let mask = 0u64.wrapping_sub(borrow as u64);
        let h = [
            (d[0] & mask) | (sub[0] & !mask),
            (d[1] & mask) | (sub[1] & !mask),
            (d[2] & mask) | (sub[2] & !mask),
            (d[3] & mask) | (sub[3] & !mask),
            (d[4] & mask) | (sub[4] & !mask),
        ];
        let mut s = [0u8; 32];
        s[0] = h[0] as u8;
        s[1] = (h[0] >> 8) as u8;
        s[2] = (h[0] >> 16) as u8;
        s[3] = (h[0] >> 24) as u8;
        s[4] = (h[0] >> 32) as u8;
        s[5] = (h[0] >> 40) as u8;
        s[6] = ((h[0] >> 48) | (h[1] << 3)) as u8;
        s[7] = (h[1] >> 5) as u8;
        s[8] = (h[1] >> 13) as u8;
        s[9] = (h[1] >> 21) as u8;
        s[10] = (h[1] >> 29) as u8;
        s[11] = (h[1] >> 37) as u8;
        s[12] = ((h[1] >> 45) | (h[2] << 6)) as u8;
        s[13] = (h[2] >> 2) as u8;
        s[14] = (h[2] >> 10) as u8;
        s[15] = (h[2] >> 18) as u8;
        s[16] = (h[2] >> 26) as u8;
        s[17] = (h[2] >> 34) as u8;
        s[18] = (h[2] >> 42) as u8;
        s[19] = ((h[2] >> 50) | (h[3] << 1)) as u8;
        s[20] = (h[3] >> 7) as u8;
        s[21] = (h[3] >> 15) as u8;
        s[22] = (h[3] >> 23) as u8;
        s[23] = (h[3] >> 31) as u8;
        s[24] = (h[3] >> 39) as u8;
        s[25] = ((h[3] >> 47) | (h[4] << 4)) as u8;
        s[26] = (h[4] >> 4) as u8;
        s[27] = (h[4] >> 12) as u8;
        s[28] = (h[4] >> 20) as u8;
        s[29] = (h[4] >> 28) as u8;
        s[30] = (h[4] >> 36) as u8;
        s[31] = (h[4] >> 44) as u8;
        s
    }

    fn add(&self, other: &Self) -> Self {
        let mut r = [0u128; 5];
        for ((r, a), b) in r.iter_mut().zip(self.0.iter()).zip(other.0.iter()) {
            *r = *a as u128 + *b as u128;
        }
        normalize(&mut r);
        Self([
            r[0] as u64,
            r[1] as u64,
            r[2] as u64,
            r[3] as u64,
            r[4] as u64,
        ])
    }

    fn sub(&self, other: &Self) -> Self {
        // Add `2*p` first so every limb stays non-negative: inputs are
        // normalized (`< 2^51 + 2^9`), so `2*p[i] - other[i] > 0`.
        let mut r = [0u128; 5];
        for i in 0..5 {
            r[i] = self.0[i] as u128 + 2 * P_LIMBS[i] as u128 - other.0[i] as u128;
        }
        normalize(&mut r);
        Self([
            r[0] as u64,
            r[1] as u64,
            r[2] as u64,
            r[3] as u64,
            r[4] as u64,
        ])
    }

    fn mul(&self, other: &Self) -> Self {
        let a = self.0;
        let b = other.0;
        let a = [
            a[0] as u128,
            a[1] as u128,
            a[2] as u128,
            a[3] as u128,
            a[4] as u128,
        ];
        let b = [
            b[0] as u128,
            b[1] as u128,
            b[2] as u128,
            b[3] as u128,
            b[4] as u128,
        ];
        let mut r = [
            a[0] * b[0] + 19 * (a[1] * b[4] + a[2] * b[3] + a[3] * b[2] + a[4] * b[1]),
            a[0] * b[1] + a[1] * b[0] + 19 * (a[2] * b[4] + a[3] * b[3] + a[4] * b[2]),
            a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + 19 * (a[3] * b[4] + a[4] * b[3]),
            a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0] + 19 * (a[4] * b[4]),
            a[0] * b[4] + a[1] * b[3] + a[2] * b[2] + a[3] * b[1] + a[4] * b[0],
        ];
        normalize(&mut r);
        Self([
            r[0] as u64,
            r[1] as u64,
            r[2] as u64,
            r[3] as u64,
            r[4] as u64,
        ])
    }

    fn sq(&self) -> Self {
        self.mul(self)
    }

    /// Multiply by a small (17-bit) constant.
    fn mul_small(&self, n: u64) -> Self {
        let mut r = [0u128; 5];
        for (r, a) in r.iter_mut().zip(self.0.iter()) {
            *r = *a as u128 * n as u128;
        }
        normalize(&mut r);
        Self([
            r[0] as u64,
            r[1] as u64,
            r[2] as u64,
            r[3] as u64,
            r[4] as u64,
        ])
    }

    /// Constant-time select: `a` when `bit == 1`, `b` when `bit == 0`.
    fn select(bit: u8, a: &Self, b: &Self) -> Self {
        debug_assert!(bit <= 1);
        let mask = 0u64.wrapping_sub(u64::from(bit));
        let mut r = [0u64; 5];
        for ((r, a), b) in r.iter_mut().zip(a.0.iter()).zip(b.0.iter()) {
            *r = (*a & mask) | (*b & !mask);
        }
        Self(r)
    }

    /// `z^(p-2) = z^(2^255 - 21)` by binary square-and-multiply.
    fn pow_p_minus_2(&self) -> Self {
        let mut r = Self::one();
        for i in (0..255).rev() {
            r = r.sq();
            let bit = if i >= 5 { 1 } else { (INV_LOW_BITS >> i) & 1 };
            let rz = r.mul(self);
            r = Self::select(bit, &rz, &r);
        }
        r
    }
}

/// Constant-time conditional swap of two field elements.
fn cswap(a: &mut Fe, b: &mut Fe, swap: u8) {
    debug_assert!(swap <= 1);
    let mask = 0u64.wrapping_sub(u64::from(swap));
    for (a, b) in a.0.iter_mut().zip(b.0.iter_mut()) {
        let t = mask & (*a ^ *b);
        *a ^= t;
        *b ^= t;
    }
}

/// Clamp a scalar per RFC 7748 section 5.
fn clamp(k: &mut [u8; 32]) {
    k[0] &= 248;
    k[31] &= 127;
    k[31] |= 64;
}

/// The Montgomery ladder: `scalar * u` on Curve25519.
///
/// `scalar` is clamped and the top bit of `u` is masked, per RFC 7748.
/// Constant time in `scalar`.
fn scalarmult(scalar: &[u8; 32], u: &[u8; 32]) -> [u8; 32] {
    let mut k = *scalar;
    clamp(&mut k);
    let mut ub = *u;
    ub[31] &= 127;

    let x1 = Fe::from_bytes(&ub);
    let mut x2 = Fe::one();
    let mut z2 = Fe::zero();
    let mut x3 = x1;
    let mut z3 = Fe::one();
    let mut swap: u8 = 0;

    for bit in (0..=254).rev() {
        let kt = (k[bit >> 3] >> (bit & 7)) & 1;
        swap ^= kt;
        cswap(&mut x2, &mut x3, swap);
        cswap(&mut z2, &mut z3, swap);
        swap = kt;

        let a = x2.add(&z2);
        let aa = a.sq();
        let b = x2.sub(&z2);
        let bb = b.sq();
        let e = aa.sub(&bb);
        let c = x3.add(&z3);
        let d = x3.sub(&z3);
        let da = d.mul(&a);
        let cb = c.mul(&b);
        x3 = da.add(&cb).sq();
        z3 = x1.mul(&da.sub(&cb).sq());
        x2 = aa.mul(&bb);
        // a24 = 121665
        z2 = e.mul(&aa.add(&e.mul_small(121_665)));
    }
    cswap(&mut x2, &mut x3, swap);
    cswap(&mut z2, &mut z3, swap);

    x2.mul(&z2.pow_p_minus_2()).to_bytes()
}

/// Generate a fresh X25519 key pair.
///
/// Returns `(secret, public)`; the public key is the clamped secret times
/// the base point. The secret is drawn from the OS entropy source.
///
/// # Errors
///
/// Returns an error if OS randomness is unavailable.
pub fn keypair() -> io::Result<([u8; 32], [u8; 32])> {
    let mut secret = [0u8; 32];
    fill_random(&mut secret)?;
    let public = scalarmult(&secret, &BASEPOINT);
    Ok((secret, public))
}

/// X25519 Diffie-Hellman: `clamp(secret) * peer_public`.
///
/// Both parties must pass their own secret and the peer's public key; the
/// outputs agree. There is no validation of `peer_public` beyond masking
/// its top bit, per RFC 7748.
#[must_use]
pub fn shared_secret(secret: &[u8; 32], peer_public: &[u8; 32]) -> [u8; 32] {
    scalarmult(secret, peer_public)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_hex(s: &str) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            out[i] = u8::from_str_radix(std::str::from_utf8(chunk).unwrap(), 16).unwrap();
        }
        out
    }

    fn hex(bytes: &[u8; 32]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// RFC 7748 section 5.2, first test vector.
    #[test]
    fn rfc7748_vector_1() {
        let scalar = from_hex("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4");
        let u = from_hex("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c");
        assert_eq!(
            hex(&scalarmult(&scalar, &u)),
            "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552"
        );
    }

    /// RFC 7748 section 5.2, second test vector.
    #[test]
    fn rfc7748_vector_2() {
        let scalar = from_hex("4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d");
        let u = from_hex("e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493");
        assert_eq!(
            hex(&scalarmult(&scalar, &u)),
            "95cbde9476e8907d7aade45cb4b873f88b595a68799fa152e6f8f7647aac7957"
        );
    }

    /// RFC 7748 section 5.2, iteration test vector. Per the RFC, each
    /// iteration sets `k` to the function result and `u` to the old `k`.
    #[test]
    fn rfc7748_iteration_vector() {
        let init = from_hex("0900000000000000000000000000000000000000000000000000000000000000");
        let (mut k, mut u) = (init, init);
        for i in 1..=1000 {
            let next = scalarmult(&k, &u);
            u = k;
            k = next;
            if i == 1 {
                assert_eq!(
                    hex(&k),
                    "422c8e7a6227d7bca1350b3e2bb7279f7897b87bb6854b783c60e80311ae3079"
                );
            }
        }
        assert_eq!(
            hex(&k),
            "684cf59ba83309552800ef566f2f4d3c1c3887c49360e3875f2eb94d99532c51"
        );
    }

    /// RFC 7748 section 6.1 Diffie-Hellman exchange.
    #[test]
    fn rfc7748_dh_exchange() {
        let alice_secret =
            from_hex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
        let bob_secret =
            from_hex("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb");
        let alice_public = scalarmult(&alice_secret, &BASEPOINT);
        let bob_public = scalarmult(&bob_secret, &BASEPOINT);
        assert_eq!(
            hex(&alice_public),
            "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a"
        );
        assert_eq!(
            hex(&bob_public),
            "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f"
        );
        let s1 = shared_secret(&alice_secret, &bob_public);
        let s2 = shared_secret(&bob_secret, &alice_public);
        assert_eq!(s1, s2);
        assert_eq!(
            hex(&s1),
            "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742"
        );
    }

    /// Field arithmetic self-consistency: `(a*b)*c == a*(b*c)`,
    /// `(a+b)-b == a`, and `a * a^(p-2) == 1`.
    #[test]
    fn field_arithmetic_identities() {
        let a = Fe::from_bytes(&from_hex(
            "a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4",
        ));
        let b = Fe::from_bytes(&from_hex(
            "e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c",
        ));
        let c = Fe::from_bytes(&from_hex(
            "4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d",
        ));
        assert_eq!(a.mul(&b).mul(&c).to_bytes(), a.mul(&b.mul(&c)).to_bytes());
        assert_eq!(a.add(&b).sub(&b).to_bytes(), a.to_bytes());
        assert_eq!(a.mul(&a.pow_p_minus_2()).to_bytes(), Fe::one().to_bytes());
        assert_eq!(Fe::zero().add(&a).to_bytes(), a.to_bytes());
    }
}
