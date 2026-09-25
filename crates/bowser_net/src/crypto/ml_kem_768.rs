//! ML-KEM-768 key encapsulation (final FIPS 203).
//!
//! Implements K-PKE key generation, encryption, and decryption
//! (Algorithms 13-15) and the ML-KEM internal key generation,
//! encapsulation, and decapsulation (Algorithms 16-18) for the
//! ML-KEM-768 parameter set (`k = 3`, `eta1 = eta2 = 2`, `du = 10`,
//! `dv = 4`).
//!
//! [`keygen`] and [`encaps`] draw randomness from the OS entropy source.

use std::io;

use super::keccak::{Shake128, sha3_256, sha3_512, shake256};
use super::rand::fill_random;

/// Length in bytes of an ML-KEM-768 encapsulation key.
pub const ENCAPSULATION_KEY_LEN: usize = 1184;
/// Length in bytes of an ML-KEM-768 decapsulation key.
pub const DECAPSULATION_KEY_LEN: usize = 2400;
/// Length in bytes of an ML-KEM-768 ciphertext.
pub const CIPHERTEXT_LEN: usize = 1088;
/// Length in bytes of an ML-KEM-768 shared secret.
pub const SHARED_SECRET_LEN: usize = 32;

const N: usize = 256;
const Q: i32 = 3329;
const K: usize = 3;
/// `eta1 == eta2 == 2` for ML-KEM-768.
const ETA: usize = 2;
const DU: u32 = 10;
const DV: u32 = 4;
/// `384 * k`: byte length of the encoded `t` vector and `s` vector.
const VEC12_LEN: usize = 1152;
/// `32 * du * k`: byte length of the first ciphertext part.
const C1_LEN: usize = 960;

/// A polynomial: 256 coefficients in `[0, Q)`.
type Poly = [i16; N];

const fn bit_rev7(x: u8) -> u8 {
    let mut r = 0u8;
    let mut i = 0u32;
    while i < 7 {
        r |= ((x >> i) & 1) << (6 - i);
        i += 1;
    }
    r
}

const fn mod_pow(mut base: i32, mut exp: u32) -> i16 {
    let mut result: i32 = 1;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % Q;
        }
        base = (base * base) % Q;
        exp >>= 1;
    }
    result as i16
}

/// NTT twiddle factors `zeta[i] = 17^{BitRev7(i+1)}` (FIPS 203, Alg. 9/10).
const ZETAS: [i16; 127] = {
    let mut z = [0i16; 127];
    let mut i = 0usize;
    while i < 127 {
        z[i] = mod_pow(17, bit_rev7(i as u8 + 1) as u32);
        i += 1;
    }
    z
};

/// Base-case multiplication factors `gamma[i] = 17^{2*BitRev7(i)+1}`
/// (FIPS 203, Alg. 11/12).
const GAMMAS: [i16; 128] = {
    let mut g = [0i16; 128];
    let mut i = 0usize;
    while i < 128 {
        g[i] = mod_pow(17, 2 * bit_rev7(i as u8) as u32 + 1);
        i += 1;
    }
    g
};

/// Reduce `x` into `[0, Q)`.
fn red(x: i32) -> i16 {
    let r = x % Q;
    (if r < 0 { r + Q } else { r }) as i16
}

/// In-place NTT (FIPS 203, Algorithm 9).
fn ntt(f: &mut Poly) {
    let mut k = 0usize;
    let mut len = 128usize;
    while len >= 2 {
        let mut start = 0usize;
        while start < N {
            let zeta = i32::from(ZETAS[k]);
            k += 1;
            for j in start..start + len {
                let t = red(zeta * i32::from(f[j + len]));
                f[j + len] = red(i32::from(f[j]) - i32::from(t));
                f[j] = red(i32::from(f[j]) + i32::from(t));
            }
            start += 2 * len;
        }
        len /= 2;
    }
}

/// In-place inverse NTT (FIPS 203, Algorithm 10).
fn inv_ntt(f: &mut Poly) {
    let mut k = 127usize;
    let mut len = 2usize;
    while len <= 128 {
        let mut start = 0usize;
        while start < N {
            let zeta = i32::from(ZETAS[k - 1]);
            k -= 1;
            for j in start..start + len {
                let t = f[j];
                f[j] = red(i32::from(t) + i32::from(f[j + len]));
                f[j + len] = red(zeta * (i32::from(f[j + len]) - i32::from(t)));
            }
            start += 2 * len;
        }
        len *= 2;
    }
    // Multiply by `128^{-1} mod q = 3303`.
    for coeff in f.iter_mut() {
        *coeff = red(3303 * i32::from(*coeff));
    }
}

/// Pointwise multiplication of NTT-domain polynomials (FIPS 203, Alg. 12).
fn basemul(a: &Poly, b: &Poly) -> Poly {
    let mut h = [0i16; N];
    for i in 0..128 {
        let gamma = i32::from(GAMMAS[i]);
        let (a0, a1) = (i32::from(a[2 * i]), i32::from(a[2 * i + 1]));
        let (b0, b1) = (i32::from(b[2 * i]), i32::from(b[2 * i + 1]));
        h[2 * i] = red(a0 * b0 + red(a1 * b1) as i32 * gamma);
        h[2 * i + 1] = red(a0 * b1 + a1 * b0);
    }
    h
}

fn poly_add(a: &Poly, b: &Poly) -> Poly {
    let mut r = [0i16; N];
    for i in 0..N {
        r[i] = red(i32::from(a[i]) + i32::from(b[i]));
    }
    r
}

/// `SampleNTT` (FIPS 203, Algorithm 7): SHAKE-128 over `seed || b1 || b2`.
fn sample_ntt(seed: &[u8; 32], b1: u8, b2: u8) -> Poly {
    let mut xof = Shake128::new();
    xof.absorb(seed);
    xof.absorb(&[b1, b2]);
    let mut f = [0i16; N];
    let mut j = 0usize;
    let mut buf = [0u8; 3];
    while j < N {
        xof.squeeze(&mut buf);
        let d1 = u16::from(buf[0]) + 256 * u16::from(buf[1] & 0x0f);
        let d2 = u16::from(buf[1] >> 4) + 16 * u16::from(buf[2]);
        if d1 < Q as u16 {
            f[j] = d1 as i16;
            j += 1;
        }
        if d2 < Q as u16 && j < N {
            f[j] = d2 as i16;
            j += 1;
        }
    }
    f
}

/// `PRF_eta(sigma, n) = SHAKE-256(sigma || n, 64*eta bytes)` (FIPS 203, 4.1).
fn prf(s: &[u8; 32], n: u8) -> [u8; 64 * ETA] {
    let mut input = [0u8; 33];
    input[..32].copy_from_slice(s);
    input[32] = n;
    let mut out = [0u8; 64 * ETA];
    shake256(&input, &mut out);
    out
}

/// `SamplePolyCBD_eta` for `eta = 2` (FIPS 203, Algorithm 8).
fn sample_cbd(buf: &[u8; 64 * ETA]) -> Poly {
    let bit = |k: usize| (buf[k >> 3] >> (k & 7)) & 1;
    let mut f = [0i16; N];
    for (i, coeff) in f.iter_mut().enumerate() {
        let mut x = 0u32;
        let mut y = 0u32;
        for j in 0..ETA {
            x += u32::from(bit(2 * ETA * i + j));
            y += u32::from(bit(2 * ETA * i + ETA + j));
        }
        *coeff = red(x as i32 - y as i32);
    }
    f
}

/// `ByteEncode_d` (FIPS 203, Algorithm 5); `d` in `{1, 4, 10, 12}`.
fn byte_encode(f: &Poly, d: u32, out: &mut [u8]) {
    debug_assert_eq!(out.len(), 32 * d as usize);
    out.fill(0);
    let mut bit = 0usize;
    for &coeff in f.iter() {
        let mut x = coeff as u32;
        for _ in 0..d {
            if x & 1 == 1 {
                out[bit >> 3] |= 1 << (bit & 7);
            }
            x >>= 1;
            bit += 1;
        }
    }
}

/// `ByteDecode_d` (FIPS 203, Algorithm 6); `d` in `{1, 4, 10, 12}`.
fn byte_decode(bytes: &[u8], d: u32) -> Poly {
    debug_assert_eq!(bytes.len(), 32 * d as usize);
    let mut f = [0i16; N];
    let mut bit = 0usize;
    for coeff in f.iter_mut() {
        let mut x = 0u32;
        for j in 0..d {
            if (bytes[bit >> 3] >> (bit & 7)) & 1 == 1 {
                x |= 1 << j;
            }
            bit += 1;
        }
        *coeff = if d == 12 {
            (x % Q as u32) as i16
        } else {
            x as i16
        };
    }
    f
}

/// `Compress_d(x) = round(2^d / q * x) mod 2^d` (FIPS 203, 4.2.1).
fn compress(x: i16, d: u32) -> u16 {
    let t = (u32::from(x as u16) << d) + Q as u32 / 2;
    ((t / Q as u32) & ((1 << d) - 1)) as u16
}

/// `Decompress_d(y) = round(q / 2^d * y)` (FIPS 203, 4.2.1).
fn decompress(y: u16, d: u32) -> i16 {
    (((Q as u32 * u32::from(y)) + (1 << (d - 1))) >> d) as i16
}

/// K-PKE key generation (FIPS 203, Algorithm 13).
fn kpke_keygen(d: &[u8; 32]) -> ([u8; ENCAPSULATION_KEY_LEN], [u8; VEC12_LEN]) {
    let mut g_in = [0u8; 33];
    g_in[..32].copy_from_slice(d);
    g_in[32] = K as u8;
    let g = sha3_512(&g_in);
    let mut rho = [0u8; 32];
    let mut sigma = [0u8; 32];
    rho.copy_from_slice(&g[..32]);
    sigma.copy_from_slice(&g[32..]);

    let mut a_hat = [[[0i16; N]; K]; K];
    for (i, row) in a_hat.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            *entry = sample_ntt(&rho, j as u8, i as u8);
        }
    }

    let mut n: u8 = 0;
    let mut s_hat = [[0i16; N]; K];
    let mut e_hat = [[0i16; N]; K];
    for v in s_hat.iter_mut() {
        *v = sample_cbd(&prf(&sigma, n));
        n += 1;
        ntt(v);
    }
    for v in e_hat.iter_mut() {
        *v = sample_cbd(&prf(&sigma, n));
        n += 1;
        ntt(v);
    }

    let mut t_hat = [[0i16; N]; K];
    for (i, t) in t_hat.iter_mut().enumerate() {
        let mut acc = [0i16; N];
        for j in 0..K {
            acc = poly_add(&acc, &basemul(&a_hat[i][j], &s_hat[j]));
        }
        *t = poly_add(&acc, &e_hat[i]);
    }

    let mut ek = [0u8; ENCAPSULATION_KEY_LEN];
    let mut dk = [0u8; VEC12_LEN];
    for i in 0..K {
        byte_encode(&t_hat[i], 12, &mut ek[i * 384..(i + 1) * 384]);
        byte_encode(&s_hat[i], 12, &mut dk[i * 384..(i + 1) * 384]);
    }
    ek[VEC12_LEN..].copy_from_slice(&rho);
    (ek, dk)
}

/// K-PKE encryption (FIPS 203, Algorithm 14).
fn kpke_encrypt(
    ek: &[u8; ENCAPSULATION_KEY_LEN],
    m: &[u8; 32],
    r: &[u8; 32],
) -> [u8; CIPHERTEXT_LEN] {
    let mut t_hat = [[0i16; N]; K];
    for (i, t) in t_hat.iter_mut().enumerate() {
        *t = byte_decode(&ek[i * 384..(i + 1) * 384], 12);
    }
    let mut rho = [0u8; 32];
    rho.copy_from_slice(&ek[VEC12_LEN..]);

    let mut a_hat = [[[0i16; N]; K]; K];
    for (i, row) in a_hat.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            *entry = sample_ntt(&rho, j as u8, i as u8);
        }
    }

    let mut n: u8 = 0;
    let mut y_hat = [[0i16; N]; K];
    let mut e1 = [[0i16; N]; K];
    for v in y_hat.iter_mut() {
        *v = sample_cbd(&prf(r, n));
        n += 1;
        ntt(v);
    }
    for v in e1.iter_mut() {
        *v = sample_cbd(&prf(r, n));
        n += 1;
    }
    let e2 = sample_cbd(&prf(r, n));

    let mut u = [[0i16; N]; K];
    for (i, ui) in u.iter_mut().enumerate() {
        let mut acc = [0i16; N];
        for j in 0..K {
            acc = poly_add(&acc, &basemul(&a_hat[j][i], &y_hat[j]));
        }
        inv_ntt(&mut acc);
        *ui = poly_add(&acc, &e1[i]);
    }

    let mut acc = [0i16; N];
    for j in 0..K {
        acc = poly_add(&acc, &basemul(&t_hat[j], &y_hat[j]));
    }
    inv_ntt(&mut acc);
    let m_poly = byte_decode(m, 1);
    let mut v = [0i16; N];
    for i in 0..N {
        v[i] =
            red(i32::from(acc[i]) + i32::from(e2[i]) + i32::from(decompress(m_poly[i] as u16, 1)));
    }

    let mut c = [0u8; CIPHERTEXT_LEN];
    for (i, ui) in u.iter().enumerate() {
        let mut cu = [0i16; N];
        for j in 0..N {
            cu[j] = compress(ui[j], DU) as i16;
        }
        byte_encode(&cu, DU, &mut c[i * 320..(i + 1) * 320]);
    }
    let mut cv = [0i16; N];
    for j in 0..N {
        cv[j] = compress(v[j], DV) as i16;
    }
    byte_encode(&cv, DV, &mut c[C1_LEN..]);
    c
}

/// K-PKE decryption (FIPS 203, Algorithm 15).
///
/// Used only by the test-only [`decaps_internal`].
#[cfg(test)]
fn kpke_decrypt(dk: &[u8; VEC12_LEN], c: &[u8; CIPHERTEXT_LEN]) -> [u8; 32] {
    let mut u = [[0i16; N]; K];
    for (i, ui) in u.iter_mut().enumerate() {
        let cu = byte_decode(&c[i * 320..(i + 1) * 320], DU);
        for j in 0..N {
            ui[j] = decompress(cu[j] as u16, DU);
        }
    }
    let cv = byte_decode(&c[C1_LEN..], DV);
    let mut v = [0i16; N];
    for j in 0..N {
        v[j] = decompress(cv[j] as u16, DV);
    }

    let mut s_hat = [[0i16; N]; K];
    for (i, s) in s_hat.iter_mut().enumerate() {
        *s = byte_decode(&dk[i * 384..(i + 1) * 384], 12);
    }

    let mut acc = [0i16; N];
    for j in 0..K {
        let mut uj = u[j];
        ntt(&mut uj);
        acc = poly_add(&acc, &basemul(&s_hat[j], &uj));
    }
    inv_ntt(&mut acc);

    let mut cw = [0i16; N];
    for j in 0..N {
        cw[j] = compress(red(i32::from(v[j]) - i32::from(acc[j])), 1) as i16;
    }
    let mut m = [0u8; 32];
    byte_encode(&cw, 1, &mut m);
    m
}

/// ML-KEM internal key generation (FIPS 203, Algorithm 16).
fn keygen_internal(
    d: &[u8; 32],
    z: &[u8; 32],
) -> ([u8; ENCAPSULATION_KEY_LEN], [u8; DECAPSULATION_KEY_LEN]) {
    let (ek, dk_pke) = kpke_keygen(d);
    let h = sha3_256(&ek);
    let mut dk = [0u8; DECAPSULATION_KEY_LEN];
    dk[..VEC12_LEN].copy_from_slice(&dk_pke);
    dk[VEC12_LEN..VEC12_LEN + ENCAPSULATION_KEY_LEN].copy_from_slice(&ek);
    dk[VEC12_LEN + ENCAPSULATION_KEY_LEN..VEC12_LEN + ENCAPSULATION_KEY_LEN + 32]
        .copy_from_slice(&h);
    dk[VEC12_LEN + ENCAPSULATION_KEY_LEN + 32..].copy_from_slice(z);
    (ek, dk)
}

/// ML-KEM internal encapsulation (FIPS 203, Algorithm 17).
fn encaps_internal(
    ek: &[u8; ENCAPSULATION_KEY_LEN],
    m: &[u8; 32],
) -> ([u8; CIPHERTEXT_LEN], [u8; SHARED_SECRET_LEN]) {
    let h = sha3_256(ek);
    let mut g_in = [0u8; 64];
    g_in[..32].copy_from_slice(m);
    g_in[32..].copy_from_slice(&h);
    let g = sha3_512(&g_in);
    let mut k = [0u8; SHARED_SECRET_LEN];
    let mut r = [0u8; 32];
    k.copy_from_slice(&g[..32]);
    r.copy_from_slice(&g[32..]);
    let c = kpke_encrypt(ek, m, &r);
    (c, k)
}

/// ML-KEM internal decapsulation (FIPS 203, Algorithm 18).
///
/// Test-only: validates the KEM round trip against the Wycheproof vectors.
#[cfg(test)]
fn decaps_internal(dk: &[u8; DECAPSULATION_KEY_LEN], c: &[u8; CIPHERTEXT_LEN]) -> [u8; 32] {
    let mut dk_pke = [0u8; VEC12_LEN];
    let mut ek = [0u8; ENCAPSULATION_KEY_LEN];
    let mut h = [0u8; 32];
    let mut z = [0u8; 32];
    dk_pke.copy_from_slice(&dk[..VEC12_LEN]);
    ek.copy_from_slice(&dk[VEC12_LEN..VEC12_LEN + ENCAPSULATION_KEY_LEN]);
    h.copy_from_slice(
        &dk[VEC12_LEN + ENCAPSULATION_KEY_LEN..VEC12_LEN + ENCAPSULATION_KEY_LEN + 32],
    );
    z.copy_from_slice(&dk[VEC12_LEN + ENCAPSULATION_KEY_LEN + 32..]);

    let m2 = kpke_decrypt(&dk_pke, c);
    let mut g_in = [0u8; 64];
    g_in[..32].copy_from_slice(&m2);
    g_in[32..].copy_from_slice(&h);
    let g = sha3_512(&g_in);
    let mut k2 = [0u8; 32];
    let mut r2 = [0u8; 32];
    k2.copy_from_slice(&g[..32]);
    r2.copy_from_slice(&g[32..]);

    let mut j_in = [0u8; 32 + CIPHERTEXT_LEN];
    j_in[..32].copy_from_slice(&z);
    j_in[32..].copy_from_slice(c);
    let mut kbar = [0u8; 32];
    shake256(&j_in, &mut kbar);

    let c2 = kpke_encrypt(&ek, &m2, &r2);
    // Constant-time implicit-rejection select: `k2` when `c == c2`.
    let mut diff = 0u8;
    for i in 0..CIPHERTEXT_LEN {
        diff |= c[i] ^ c2[i];
    }
    let nonzero = ((u16::from(diff) | u16::from(diff.wrapping_neg())) >> 8) as u8 & 1;
    let mask = 0u8.wrapping_sub(1 - nonzero);
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = (k2[i] & mask) | (kbar[i] & !mask);
    }
    out
}

/// Encapsulation key check (FIPS 203, section 7.2): the type check is
/// enforced by the array length; this is the modulus check.
fn ek_is_valid(ek: &[u8; ENCAPSULATION_KEY_LEN]) -> bool {
    let mut reencoded = [0u8; VEC12_LEN];
    for i in 0..K {
        let f = byte_decode(&ek[i * 384..(i + 1) * 384], 12);
        byte_encode(&f, 12, &mut reencoded[i * 384..(i + 1) * 384]);
    }
    let mut diff = 0u8;
    for i in 0..VEC12_LEN {
        diff |= ek[i] ^ reencoded[i];
    }
    diff == 0
}

/// Generate a fresh ML-KEM-768 key pair.
///
/// Returns `(encapsulation_key, decapsulation_key)`. The 64 bytes of
/// randomness come from the OS entropy source.
///
/// # Errors
///
/// Returns an error if OS randomness is unavailable.
pub fn keygen() -> io::Result<([u8; ENCAPSULATION_KEY_LEN], [u8; DECAPSULATION_KEY_LEN])> {
    let mut d = [0u8; 32];
    let mut z = [0u8; 32];
    fill_random(&mut d)?;
    fill_random(&mut z)?;
    Ok(keygen_internal(&d, &z))
}

/// Encapsulate a fresh shared secret under `ek`.
///
/// Returns `(ciphertext, shared_secret)`. The encapsulation key must pass
/// the FIPS 203 section 7.2 checks; an unchecked or malformed key is
/// rejected with [`io::ErrorKind::InvalidInput`].
///
/// # Errors
///
/// Returns an error if `ek` fails validation or OS randomness is
/// unavailable.
pub fn encaps(
    ek: &[u8; ENCAPSULATION_KEY_LEN],
) -> io::Result<([u8; CIPHERTEXT_LEN], [u8; SHARED_SECRET_LEN])> {
    if !ek_is_valid(ek) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid ML-KEM-768 encapsulation key",
        ));
    }
    let mut m = [0u8; 32];
    fill_random(&mut m)?;
    Ok(encaps_internal(ek, &m))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_fixture(text: &str) -> Vec<Vec<(String, String)>> {
        let mut vecs = Vec::new();
        let mut cur = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                if !cur.is_empty() {
                    vecs.push(std::mem::take(&mut cur));
                }
                continue;
            }
            if line.starts_with('#') {
                continue;
            }
            let (k, v) = line.split_once(' ').expect("malformed fixture line");
            cur.push((k.to_string(), v.to_string()));
        }
        if !cur.is_empty() {
            vecs.push(cur);
        }
        vecs
    }

    fn get(vec: &[(String, String)], key: &str) -> Vec<u8> {
        let hex = vec
            .iter()
            .find(|(k, _)| k == key)
            .unwrap_or_else(|| panic!("missing key {key}"))
            .1
            .clone();
        assert_eq!(hex.len() % 2, 0);
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("bad hex"))
            .collect()
    }

    fn as_array<const N: usize>(v: &[u8]) -> [u8; N] {
        v.try_into().expect("fixture length mismatch")
    }

    /// Wycheproof MLKEMTest KATs: `keygen_internal(seed) -> ek`.
    #[test]
    fn kat_keygen() {
        let text = include_str!("../../tests/fixtures/mlkem768_keygen.txt");
        let vecs = parse_fixture(text);
        assert!(!vecs.is_empty());
        for v in &vecs {
            let seed: [u8; 64] = as_array(&get(v, "seed"));
            let expected_ek: [u8; ENCAPSULATION_KEY_LEN] = as_array(&get(v, "ek"));
            let (ek, dk) = keygen_internal(
                &seed[..32].try_into().unwrap(),
                &seed[32..].try_into().unwrap(),
            );
            assert_eq!(ek, expected_ek, "ek mismatch");
            assert_eq!(dk.len(), DECAPSULATION_KEY_LEN);
            // `dk` embeds `ek` and `H(ek)` per Algorithm 16.
            assert_eq!(&dk[VEC12_LEN..VEC12_LEN + ENCAPSULATION_KEY_LEN], &ek);
            assert_eq!(
                &dk[VEC12_LEN + ENCAPSULATION_KEY_LEN..VEC12_LEN + ENCAPSULATION_KEY_LEN + 32],
                &sha3_256(&ek)
            );
        }
    }

    /// Wycheproof MLKEMEncapsTest KATs: `encaps_internal(ek, m) -> (c, K)`.
    #[test]
    fn kat_encaps() {
        let text = include_str!("../../tests/fixtures/mlkem768_encaps.txt");
        let vecs = parse_fixture(text);
        assert!(!vecs.is_empty());
        for v in &vecs {
            let ek: [u8; ENCAPSULATION_KEY_LEN] = as_array(&get(v, "ek"));
            let m: [u8; 32] = as_array(&get(v, "m"));
            let expected_c: [u8; CIPHERTEXT_LEN] = as_array(&get(v, "c"));
            let expected_k: [u8; 32] = as_array(&get(v, "K"));
            let (c, k) = encaps_internal(&ek, &m);
            assert_eq!(c, expected_c, "ciphertext mismatch");
            assert_eq!(k, expected_k, "shared secret mismatch");
        }
    }

    /// Wycheproof MLKEMTest KATs: `decaps_internal(dk, c) -> K`.
    #[test]
    fn kat_decaps() {
        let text = include_str!("../../tests/fixtures/mlkem768_decaps.txt");
        let vecs = parse_fixture(text);
        assert!(!vecs.is_empty());
        for v in &vecs {
            let seed: [u8; 64] = as_array(&get(v, "seed"));
            let c: [u8; CIPHERTEXT_LEN] = as_array(&get(v, "c"));
            let expected_k: [u8; 32] = as_array(&get(v, "K"));
            let (_, dk) = keygen_internal(
                &seed[..32].try_into().unwrap(),
                &seed[32..].try_into().unwrap(),
            );
            assert_eq!(decaps_internal(&dk, &c), expected_k, "decaps mismatch");
        }
    }

    /// Full round trip with real OS randomness.
    #[test]
    fn roundtrip() {
        let (ek, dk) = keygen().expect("keygen");
        let (c, k1) = encaps(&ek).expect("encaps");
        let k2 = decaps_internal(&dk, &c);
        assert_eq!(k1, k2);
    }

    /// `encaps` rejects keys that fail the FIPS 203 modulus check.
    #[test]
    fn encaps_rejects_bad_key() {
        let (ek, _) = keygen().expect("keygen");
        // Encode the out-of-range value 3330 in the first coefficient.
        let mut bad = ek;
        bad[0] = 0x02;
        bad[1] = 0x0d;
        assert!(!ek_is_valid(&bad));
        assert!(encaps(&bad).is_err());
        assert!(ek_is_valid(&ek));
    }

    /// `NTT^-1(NTT(f)) == f`.
    #[test]
    fn ntt_roundtrip() {
        let mut f = [0i16; N];
        for (i, c) in f.iter_mut().enumerate() {
            *c = ((i * 7919 + 13) % Q as usize) as i16;
        }
        let orig = f;
        ntt(&mut f);
        assert_ne!(f, orig);
        inv_ntt(&mut f);
        assert_eq!(f, orig);
    }

    /// `ByteDecode_d(ByteEncode_d(f)) == f`, exact for `d = 12` and for
    /// coefficients below `2^d` otherwise.
    #[test]
    fn byte_codec_roundtrip() {
        let mut f = [0i16; N];
        for (i, c) in f.iter_mut().enumerate() {
            *c = ((i * 104729 + 7) % Q as usize) as i16;
        }
        // d = 12 is exact on canonical coefficients.
        let mut buf = vec![0u8; 32 * 12];
        byte_encode(&f, 12, &mut buf);
        assert_eq!(byte_decode(&buf, 12), f);
        // For smaller widths the codec is exact below 2^d.
        for d in [1u32, 4, 10] {
            let mut g = [0i16; N];
            for (i, c) in g.iter_mut().enumerate() {
                *c = f[i] & ((1i16 << d) - 1);
            }
            let mut buf = vec![0u8; 32 * d as usize];
            byte_encode(&g, d, &mut buf);
            assert_eq!(byte_decode(&buf, d), g);
        }
    }

    /// `Compress_d(Decompress_d(y)) == y` for the used bit widths.
    #[test]
    fn compress_roundtrip() {
        for d in [1u32, 4, 10] {
            for y in 0..(1u16 << d) {
                assert_eq!(compress(decompress(y, d), d), y, "d={d} y={y}");
            }
        }
    }
}
