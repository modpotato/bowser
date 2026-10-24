//! Unsigned big integers for public-key cryptography (RSA, ECDSA).
//!
//! This is the arithmetic core underneath [`crate::crypto::rsa`] and
//! [`crate::crypto::ec`]. It is deliberately small: schoolbook
//! multiplication, binary long division for the one-off reductions, and
//! Montgomery multiplication / exponentiation for everything hot.
//!
//! # Constant-time posture
//!
//! This module is **not** constant-time, and that is acceptable for its
//! only uses: RSA and ECDSA *verification*. Every input to verification
//! (signature, modulus, public key, message hash) is public — there is no
//! secret to leak through timing or cache behavior. The final signature
//! comparisons in `rsa`/`ec` still use constant-time equality as
//! defense-in-depth. Do **not** reuse this module for private-key
//! operations (signing, decryption) without a constant-time audit.

use core::cmp::Ordering;

/// Unsigned big integer, little-endian `u64` limbs, no trailing zero limbs
/// (the value zero is the empty vector). All operations panic-free:
/// out-of-range or malformed inputs produce `None`/zero rather than panics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BigUint {
    limbs: Vec<u64>,
}

impl BigUint {
    pub(crate) fn zero() -> Self {
        BigUint { limbs: Vec::new() }
    }

    pub(crate) fn one() -> Self {
        BigUint { limbs: vec![1] }
    }

    fn from_limbs(mut limbs: Vec<u64>) -> Self {
        while limbs.last() == Some(&0) {
            limbs.pop();
        }
        BigUint { limbs }
    }

    pub(crate) fn from_u64(v: u64) -> Self {
        Self::from_limbs(vec![v])
    }

    /// Parse big-endian bytes. Leading zero bytes are allowed (DER INTEGERs
    /// carry one). Empty input is zero.
    pub(crate) fn from_be_bytes(b: &[u8]) -> Self {
        let mut limbs = vec![0u64; b.len().div_ceil(8)];
        for (i, &byte) in b.iter().rev().enumerate() {
            limbs[i / 8] |= (byte as u64) << ((i % 8) * 8);
        }
        Self::from_limbs(limbs)
    }

    /// Minimal big-endian encoding (empty for zero).
    pub(crate) fn to_be_bytes(&self) -> Vec<u8> {
        if self.limbs.is_empty() {
            return Vec::new();
        }
        let mut out = vec![0u8; self.limbs.len() * 8];
        for (i, &limb) in self.limbs.iter().enumerate() {
            out[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        out.reverse();
        let skip = out.iter().take_while(|&&b| b == 0).count();
        out[skip..].to_vec()
    }

    /// Big-endian encoding left-padded to exactly `len` bytes.
    /// Returns `None` when the value does not fit.
    pub(crate) fn to_be_bytes_padded(&self, len: usize) -> Option<Vec<u8>> {
        let raw = self.to_be_bytes();
        if raw.len() > len {
            return None;
        }
        let mut out = vec![0u8; len];
        out[len - raw.len()..].copy_from_slice(&raw);
        Some(out)
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    pub(crate) fn is_odd(&self) -> bool {
        self.limbs.first().is_some_and(|&l| l & 1 == 1)
    }

    /// Number of significant bits (0 for zero).
    pub(crate) fn bit_len(&self) -> usize {
        match self.limbs.last() {
            None => 0,
            Some(&top) => (self.limbs.len() - 1) * 64 + (64 - top.leading_zeros() as usize),
        }
    }

    pub(crate) fn get_bit(&self, i: usize) -> bool {
        let (w, b) = (i / 64, i % 64);
        self.limbs.get(w).is_some_and(|&l| (l >> b) & 1 == 1)
    }

    pub(crate) fn cmp(&self, other: &Self) -> Ordering {
        match self.limbs.len().cmp(&other.limbs.len()) {
            Ordering::Equal => self.limbs.iter().rev().cmp(other.limbs.iter().rev()),
            ord => ord,
        }
    }

    fn ge(&self, other: &Self) -> bool {
        self.cmp(other) != Ordering::Less
    }

    /// `self + other`.
    pub(crate) fn add(&self, other: &Self) -> Self {
        let n = self.limbs.len().max(other.limbs.len());
        let mut out = Vec::with_capacity(n + 1);
        let mut carry = 0u64;
        for i in 0..n {
            let a = self.limbs.get(i).copied().unwrap_or(0);
            let b = other.limbs.get(i).copied().unwrap_or(0);
            let (s1, c1) = a.overflowing_add(b);
            let (s2, c2) = s1.overflowing_add(carry);
            out.push(s2);
            carry = (c1 as u64) + (c2 as u64);
        }
        if carry != 0 {
            out.push(carry);
        }
        Self::from_limbs(out)
    }

    /// `self - other`, or `None` on underflow.
    pub(crate) fn sub(&self, other: &Self) -> Option<Self> {
        if !self.ge(other) {
            return None;
        }
        let mut out = Vec::with_capacity(self.limbs.len());
        let mut borrow = 0u64;
        for i in 0..self.limbs.len() {
            let a = self.limbs[i];
            let b = other.limbs.get(i).copied().unwrap_or(0);
            let (d1, br1) = a.overflowing_sub(b);
            let (d2, br2) = d1.overflowing_sub(borrow);
            out.push(d2);
            borrow = (br1 as u64) + (br2 as u64);
        }
        debug_assert_eq!(borrow, 0);
        Some(Self::from_limbs(out))
    }

    /// Schoolbook multiplication.
    pub(crate) fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let (a, b) = (&self.limbs, &other.limbs);
        let mut out = vec![0u64; a.len() + b.len()];
        for (i, &x) in a.iter().enumerate() {
            let mut carry = 0u64;
            for (j, &y) in b.iter().enumerate() {
                let t = out[i + j] as u128 + x as u128 * y as u128 + carry as u128;
                out[i + j] = t as u64;
                carry = (t >> 64) as u64;
            }
            // `out[i + b.len()]` may already hold a partial sum from the
            // previous row's carry; propagate.
            let mut k = i + b.len();
            let mut c = carry;
            while c != 0 {
                let t = out[k] as u128 + c as u128;
                out[k] = t as u64;
                c = (t >> 64) as u64;
                k += 1;
            }
        }
        Self::from_limbs(out)
    }

    /// `self << 1`.
    fn shl1(&self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let mut out = Vec::with_capacity(self.limbs.len() + 1);
        let mut carry = 0u64;
        for &l in &self.limbs {
            out.push((l << 1) | carry);
            carry = l >> 63;
        }
        if carry != 0 {
            out.push(carry);
        }
        Self::from_limbs(out)
    }

    /// `self mod n` by binary long division. Simple and obviously correct;
    /// used only for one-off reductions (converting into Montgomery form),
    /// never in hot loops.
    pub(crate) fn rem(&self, n: &Self) -> Self {
        if n.is_zero() || self.cmp(n) == Ordering::Less {
            return self.clone();
        }
        let mut r = Self::zero();
        for i in (0..self.bit_len()).rev() {
            r = r.shl1();
            if self.get_bit(i) {
                r = r.add(&Self::one());
            }
            if r.ge(n) {
                r = r.sub(n).expect("r >= n");
            }
        }
        r
    }
}

/// Montgomery arithmetic modulo an odd `n > 1`.
///
/// Values are converted once with [`Montgomery::to_mont`], multiplied with
/// [`Montgomery::mul`] (no division anywhere), and converted back with
/// [`Montgomery::demont`].
pub(crate) struct Montgomery {
    n: BigUint,
    /// `-n^{-1} mod 2^64`, Newton iteration.
    n0: u64,
    /// `R^2 mod n` in regular representation, where `R = 2^(64k)`.
    r2: BigUint,
    k: usize,
}

impl Montgomery {
    /// `None` when `n` is even or `n < 2`.
    pub(crate) fn new(n: &BigUint) -> Option<Self> {
        if !n.is_odd() || n.cmp(&BigUint::one()) != Ordering::Greater {
            return None;
        }
        let k = n.limbs.len();
        // Newton iteration for n^{-1} mod 2^64 (doubles precision per step).
        let n0limb = n.limbs[0];
        let mut inv = 1u64;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(n0limb.wrapping_mul(inv)));
        }
        let n0 = inv.wrapping_neg();
        // R^2 mod n via repeated doubling from 1 (2*64k doublings).
        let mut r2 = BigUint::one();
        for _ in 0..2 * 64 * k {
            r2 = r2.shl1();
            if r2.ge(n) {
                r2 = r2.sub(n).expect("r2 >= n");
            }
        }
        Some(Montgomery {
            n: n.clone(),
            n0,
            r2,
            k,
        })
    }

    /// `REDC`: reduce `t` (at most `2k+1` limbs, `t < R*n`) to `t*R^{-1} mod n`.
    fn redc(&self, t: &mut Vec<u64>) -> BigUint {
        debug_assert!(t.len() <= 2 * self.k + 1);
        t.resize(2 * self.k + 1, 0);
        let nlimbs = &self.n.limbs;
        for i in 0..self.k {
            let m = t[i].wrapping_mul(self.n0);
            let mut carry = 0u64;
            for (j, &nl) in nlimbs.iter().enumerate() {
                let prod = (m as u128) * (nl as u128) + (t[i + j] as u128) + (carry as u128);
                t[i + j] = prod as u64;
                carry = (prod >> 64) as u64;
            }
            // Propagate the carry past t[i+k].
            let mut p = i + self.k;
            let mut c = carry;
            while c != 0 {
                let s = (t[p] as u128) + (c as u128);
                t[p] = s as u64;
                c = (s >> 64) as u64;
                p += 1;
            }
        }
        let mut out = BigUint::from_limbs(t[self.k..].to_vec());
        if out.ge(&self.n) {
            out = out.sub(&self.n).expect("out >= n");
        }
        out
    }

    /// Montgomery multiplication: `a*b*R^{-1} mod n`, `a` and `b` in
    /// Montgomery form.
    pub(crate) fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        let t = a.mul(b);
        let mut limbs = t.limbs;
        self.redc(&mut limbs)
    }

    /// Convert `a` (any size) into Montgomery form: `a*R mod n`.
    pub(crate) fn to_mont(&self, a: &BigUint) -> BigUint {
        let ar = a.rem(&self.n);
        // a*R2 has < 2k+1 limbs and a*R2 < n^2 < R*n: REDC applies.
        let t = ar.mul(&self.r2);
        let mut limbs = t.limbs;
        self.redc(&mut limbs)
    }

    /// Convert out of Montgomery form (`a*R^{-1} mod n`). Named `demont`
    /// rather than `from_mont` so it reads as a conversion method on `self`,
    /// not a `Self` constructor (clippy `wrong_self_convention`).
    pub(crate) fn demont(&self, a: &BigUint) -> BigUint {
        let mut limbs = a.limbs.clone();
        self.redc(&mut limbs)
    }

    /// `base^exp mod n`, both arbitrary size. `base` is reduced mod `n`
    /// first, so oversized signatures/inputs fail at comparison time,
    /// not here.
    pub(crate) fn pow(&self, base: &BigUint, exp: &BigUint) -> BigUint {
        let mut acc = self.to_mont(&BigUint::one());
        let b = self.to_mont(base);
        for i in (0..exp.bit_len()).rev() {
            acc = self.mul(&acc, &acc);
            if exp.get_bit(i) {
                acc = self.mul(&acc, &b);
            }
        }
        self.demont(&acc)
    }

    /// Modular inverse via Fermat: `a^{n-2} mod n`. The caller must ensure
    /// `gcd(a, n) = 1` (true for prime `n` and `0 < a < n`); otherwise the
    /// result is meaningless but the function still terminates.
    pub(crate) fn inv(&self, a: &BigUint) -> BigUint {
        let two = BigUint::from_u64(2);
        let exp = self.n.sub(&two).expect("n >= 2");
        self.pow(a, &exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::hex;

    fn bi(s: &str) -> BigUint {
        BigUint::from_be_bytes(&hex(s))
    }

    #[test]
    fn roundtrip_bytes() {
        for s in ["", "00", "01", "ff", "0100", "123456789abcdef0", "00ff00"] {
            let v = bi(s);
            let back = v.to_be_bytes();
            let want = hex(s);
            let want = want
                .iter()
                .skip_while(|&&b| b == 0)
                .copied()
                .collect::<Vec<_>>();
            assert_eq!(back, want, "roundtrip {s}");
        }
        assert_eq!(bi("1234").to_be_bytes_padded(4), Some(hex("00001234")));
        assert_eq!(bi("1234").to_be_bytes_padded(1), None);
    }

    #[test]
    fn add_sub_mul() {
        let a = bi("ffffffffffffffff");
        let b = bi("01");
        assert_eq!(a.add(&b).to_be_bytes(), hex("010000000000000000"));
        let c = bi("010000000000000000");
        assert_eq!(c.sub(&b).unwrap().to_be_bytes(), hex("ffffffffffffffff"));
        assert!(b.sub(&c).is_none());
        // 0x1234 * 0x5678 = 0x06260060
        assert_eq!(bi("1234").mul(&bi("5678")).to_be_bytes(), hex("06260060"));
        // (2^128 - 1)^2 = 2^256 - 2^129 + 1 = 1 mod 2^128.
        let m = bi("ffffffffffffffffffffffffffffffff");
        let sq = m.mul(&m);
        assert_eq!(sq.bit_len(), 256);
        assert_eq!(
            sq.rem(&bi("0100000000000000000000000000000000"))
                .to_be_bytes(),
            hex("01")
        );
    }

    #[test]
    fn rem_known_values() {
        // 1000 mod 7 = 6
        assert_eq!(bi("03e8").rem(&bi("07")).to_be_bytes(), hex("06"));
        // 2^200 mod (2^128 + 1): since 2^128 = -1 mod m,
        // 2^200 = 2^72 * 2^128 = -2^72 mod m.
        let mut p = BigUint::one();
        for _ in 0..200 {
            p = p.shl1();
        }
        let mut modu = BigUint::one();
        for _ in 0..128 {
            modu = modu.shl1();
        }
        modu = modu.add(&BigUint::one());
        let mut p72 = BigUint::one();
        for _ in 0..72 {
            p72 = p72.shl1();
        }
        assert_eq!(p.rem(&modu), modu.sub(&p72).unwrap());
        assert_eq!(bi("ff").rem(&bi("0100")).to_be_bytes(), hex("ff"));
    }

    #[test]
    fn montgomery_new_rejects_bad_moduli() {
        assert!(Montgomery::new(&bi("00")).is_none());
        assert!(Montgomery::new(&bi("01")).is_none());
        assert!(Montgomery::new(&bi("10")).is_none()); // even
        assert!(Montgomery::new(&bi("03")).is_some());
    }

    /// pow cross-checked against a naive square-and-multiply with `rem`.
    fn naive_pow(base: &BigUint, exp: &BigUint, n: &BigUint) -> BigUint {
        let mut acc = BigUint::one().rem(n);
        let b = base.rem(n);
        for i in (0..exp.bit_len()).rev() {
            acc = acc.mul(&acc).rem(n);
            if exp.get_bit(i) {
                acc = acc.mul(&b).rem(n);
            }
        }
        acc
    }

    #[test]
    fn montgomery_pow_matches_naive() {
        // A 256-bit prime-ish modulus and a 2048-bit RSA-like modulus.
        let p = bi("ffffffff00000001000000000000000000000000ffffffffffffffffffffffff");
        let mont = Montgomery::new(&p).unwrap();
        let base = bi("6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296");
        let exp = bi("0a6e3c57dd01abe90086538398355dd4c3b17aa873382f26612f66a72a07");
        assert_eq!(mont.pow(&base, &exp), naive_pow(&base, &exp, &p));

        let n = bi(&"ab".repeat(256));
        let mont = Montgomery::new(&n).unwrap();
        let base = bi(&"cd".repeat(200));
        let exp = bi("010001");
        assert_eq!(mont.pow(&base, &exp), naive_pow(&base, &exp, &n));
    }

    #[test]
    fn montgomery_roundtrip_and_mul() {
        let p = bi("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551");
        let mont = Montgomery::new(&p).unwrap();
        let a = bi("0123456789abcdef");
        let am = mont.to_mont(&a);
        assert_eq!(mont.demont(&am), a);
        let b = bi("0fedcba987654321");
        let ab = mont.demont(&mont.mul(&mont.to_mont(&a), &mont.to_mont(&b)));
        assert_eq!(ab, a.mul(&b).rem(&p));
    }

    #[test]
    fn fermat_inverse() {
        let p = bi("ffffffff00000001000000000000000000000000ffffffffffffffffffffffff");
        let mont = Montgomery::new(&p).unwrap();
        let a = bi("deadbeefcafe");
        let inv = mont.inv(&a);
        let one = mont.demont(&mont.mul(&mont.to_mont(&a), &mont.to_mont(&inv)));
        assert_eq!(one, BigUint::one());
    }

    #[test]
    fn pow_oversized_base_reduced() {
        // base > n must behave as base mod n.
        let n = bi("fffd");
        let mont = Montgomery::new(&n).unwrap();
        let base = bi("010000"); // 65536
        let got = mont.pow(&base, &bi("03"));
        let want = naive_pow(&base, &bi("03"), &n);
        assert_eq!(got, want);
    }
}
