//! AES-128 and AES-256 block cipher (FIPS-197).
//!
//! Table-driven implementation: S-box lookups are the only
//! secret-dependent memory accesses. All arithmetic is branch-free.

/// AES substitution box (FIPS-197, Section 5.1.1).
#[rustfmt::skip]
const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

/// Inverse AES substitution box (FIPS-197, Section 5.3.2).
#[rustfmt::skip]
const INV_SBOX: [u8; 256] = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
];

/// Round constants for key expansion (FIPS-197, Section 5.2).
const RCON: [u32; 10] = [
    0x0100_0000,
    0x0200_0000,
    0x0400_0000,
    0x0800_0000,
    0x1000_0000,
    0x2000_0000,
    0x4000_0000,
    0x8000_0000,
    0x1b00_0000,
    0x3600_0000,
];

/// Expanded AES key schedule shared by AES-128 and AES-256.
///
/// The state is stored column-major: byte `state[row + 4 * col]`.
#[derive(Clone)]
pub(crate) struct ExpandedKey {
    /// Round keys; only the first `rounds + 1` entries are used.
    rk: [[u8; 16]; 15],
    rounds: usize,
}

impl ExpandedKey {
    fn expand(key_words: &[u32]) -> Self {
        let nk = key_words.len();
        debug_assert!(nk == 4 || nk == 8);
        let rounds = nk + 6;
        let total_words = 4 * (rounds + 1);

        let mut w = [0u32; 60];
        w[..nk].copy_from_slice(key_words);
        for i in nk..total_words {
            let mut temp = w[i - 1];
            if i % nk == 0 {
                temp = sub_word(temp.rotate_left(8)) ^ RCON[i / nk - 1];
            } else if nk > 6 && i % nk == 4 {
                temp = sub_word(temp);
            }
            w[i] = w[i - nk] ^ temp;
        }

        let mut rk = [[0u8; 16]; 15];
        for (r, key) in rk.iter_mut().enumerate().take(rounds + 1) {
            for c in 0..4 {
                key[4 * c..4 * c + 4].copy_from_slice(&w[4 * r + c].to_be_bytes());
            }
        }
        Self { rk, rounds }
    }

    pub(crate) fn expand_128(key: &[u8; 16]) -> Self {
        let mut words = [0u32; 4];
        for (i, word) in words.iter_mut().enumerate() {
            *word = u32::from_be_bytes(key[4 * i..4 * i + 4].try_into().unwrap());
        }
        Self::expand(&words)
    }

    pub(crate) fn expand_256(key: &[u8; 32]) -> Self {
        let mut words = [0u32; 8];
        for (i, word) in words.iter_mut().enumerate() {
            *word = u32::from_be_bytes(key[4 * i..4 * i + 4].try_into().unwrap());
        }
        Self::expand(&words)
    }

    /// Encrypt one 16-byte block in place.
    pub(crate) fn encrypt_block(&self, block: &mut [u8; 16]) {
        let mut s = *block;
        add_round_key(&mut s, &self.rk[0]);
        for round in 1..self.rounds {
            sub_bytes(&mut s);
            shift_rows(&mut s);
            mix_columns(&mut s);
            add_round_key(&mut s, &self.rk[round]);
        }
        sub_bytes(&mut s);
        shift_rows(&mut s);
        add_round_key(&mut s, &self.rk[self.rounds]);
        *block = s;
    }

    /// Decrypt one 16-byte block in place (equivalent inverse cipher).
    pub(crate) fn decrypt_block(&self, block: &mut [u8; 16]) {
        let mut s = *block;
        add_round_key(&mut s, &self.rk[self.rounds]);
        for round in (1..self.rounds).rev() {
            inv_shift_rows(&mut s);
            inv_sub_bytes(&mut s);
            add_round_key(&mut s, &self.rk[round]);
            inv_mix_columns(&mut s);
        }
        inv_shift_rows(&mut s);
        inv_sub_bytes(&mut s);
        add_round_key(&mut s, &self.rk[0]);
        *block = s;
    }
}

fn sub_word(w: u32) -> u32 {
    let b = w.to_be_bytes();
    u32::from_be_bytes([
        SBOX[b[0] as usize],
        SBOX[b[1] as usize],
        SBOX[b[2] as usize],
        SBOX[b[3] as usize],
    ])
}

fn add_round_key(s: &mut [u8; 16], rk: &[u8; 16]) {
    for i in 0..16 {
        s[i] ^= rk[i];
    }
}

fn sub_bytes(s: &mut [u8; 16]) {
    for b in s.iter_mut() {
        *b = SBOX[*b as usize];
    }
}

fn inv_sub_bytes(s: &mut [u8; 16]) {
    for b in s.iter_mut() {
        *b = INV_SBOX[*b as usize];
    }
}

/// Row `r` rotates left by `r` positions.
fn shift_rows(s: &mut [u8; 16]) {
    let t = *s;
    for r in 0..4 {
        for c in 0..4 {
            s[r + 4 * c] = t[r + 4 * ((c + r) % 4)];
        }
    }
}

fn inv_shift_rows(s: &mut [u8; 16]) {
    let t = *s;
    for r in 0..4 {
        for c in 0..4 {
            s[r + 4 * c] = t[r + 4 * ((c + 4 - r) % 4)];
        }
    }
}

/// Multiply by x in GF(2^8), branch-free.
fn xtime(x: u8) -> u8 {
    (x << 1) ^ (((x >> 7) & 1).wrapping_neg() & 0x1b)
}

fn mix_columns(s: &mut [u8; 16]) {
    for c in 0..4 {
        let a0 = s[4 * c];
        let a1 = s[4 * c + 1];
        let a2 = s[4 * c + 2];
        let a3 = s[4 * c + 3];
        s[4 * c] = xtime(a0) ^ xtime(a1) ^ a1 ^ a2 ^ a3;
        s[4 * c + 1] = a0 ^ xtime(a1) ^ xtime(a2) ^ a2 ^ a3;
        s[4 * c + 2] = a0 ^ a1 ^ xtime(a2) ^ xtime(a3) ^ a3;
        s[4 * c + 3] = xtime(a0) ^ a0 ^ a1 ^ a2 ^ xtime(a3);
    }
}

fn mul09(x: u8) -> u8 {
    xtime(xtime(xtime(x))) ^ x
}
fn mul0b(x: u8) -> u8 {
    xtime(xtime(xtime(x))) ^ xtime(x) ^ x
}
fn mul0d(x: u8) -> u8 {
    xtime(xtime(xtime(x))) ^ xtime(xtime(x)) ^ x
}
fn mul0e(x: u8) -> u8 {
    xtime(xtime(xtime(x))) ^ xtime(xtime(x)) ^ xtime(x)
}

fn inv_mix_columns(s: &mut [u8; 16]) {
    for c in 0..4 {
        let a0 = s[4 * c];
        let a1 = s[4 * c + 1];
        let a2 = s[4 * c + 2];
        let a3 = s[4 * c + 3];
        s[4 * c] = mul0e(a0) ^ mul0b(a1) ^ mul0d(a2) ^ mul09(a3);
        s[4 * c + 1] = mul09(a0) ^ mul0e(a1) ^ mul0b(a2) ^ mul0d(a3);
        s[4 * c + 2] = mul0d(a0) ^ mul09(a1) ^ mul0e(a2) ^ mul0b(a3);
        s[4 * c + 3] = mul0b(a0) ^ mul0d(a1) ^ mul09(a2) ^ mul0e(a3);
    }
}

/// AES-128 block cipher (FIPS-197).
#[derive(Clone)]
pub struct Aes128 {
    inner: ExpandedKey,
}

impl Aes128 {
    /// Expand a 128-bit key. Panics are impossible: the key length is fixed
    /// by the type.
    pub fn new(key: &[u8; 16]) -> Self {
        Self {
            inner: ExpandedKey::expand_128(key),
        }
    }

    /// Encrypt one 16-byte block in place.
    pub fn encrypt_block(&self, block: &mut [u8; 16]) {
        self.inner.encrypt_block(block);
    }

    /// Decrypt one 16-byte block in place.
    pub fn decrypt_block(&self, block: &mut [u8; 16]) {
        self.inner.decrypt_block(block);
    }
}

/// AES-256 block cipher (FIPS-197).
#[derive(Clone)]
pub struct Aes256 {
    inner: ExpandedKey,
}

impl Aes256 {
    /// Expand a 256-bit key.
    pub fn new(key: &[u8; 32]) -> Self {
        Self {
            inner: ExpandedKey::expand_256(key),
        }
    }

    /// Encrypt one 16-byte block in place.
    pub fn encrypt_block(&self, block: &mut [u8; 16]) {
        self.inner.encrypt_block(block);
    }

    /// Decrypt one 16-byte block in place.
    pub fn decrypt_block(&self, block: &mut [u8; 16]) {
        self.inner.decrypt_block(block);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::hex;

    fn e128(key: &str, pt: &str) -> String {
        let k = hex(key);
        let mut b = [0u8; 16];
        b.copy_from_slice(&hex(pt));
        Aes128::new(k.as_slice().try_into().unwrap()).encrypt_block(&mut b);
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    fn d128(key: &str, ct: &str) -> String {
        let k = hex(key);
        let mut b = [0u8; 16];
        b.copy_from_slice(&hex(ct));
        Aes128::new(k.as_slice().try_into().unwrap()).decrypt_block(&mut b);
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    fn e256(key: &str, pt: &str) -> String {
        let k = hex(key);
        let mut b = [0u8; 16];
        b.copy_from_slice(&hex(pt));
        Aes256::new(k.as_slice().try_into().unwrap()).encrypt_block(&mut b);
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn aes128_fips197_c1_encrypt() {
        // FIPS-197 Appendix B / C.1.
        assert_eq!(
            e128(
                "000102030405060708090a0b0c0d0e0f",
                "00112233445566778899aabbccddeeff"
            ),
            "69c4e0d86a7b0430d8cdb78070b4c55a"
        );
    }

    #[test]
    fn aes128_fips197_c1_decrypt() {
        assert_eq!(
            d128(
                "000102030405060708090a0b0c0d0e0f",
                "69c4e0d86a7b0430d8cdb78070b4c55a"
            ),
            "00112233445566778899aabbccddeeff"
        );
    }

    #[test]
    fn aes128_zero_vector() {
        assert_eq!(
            e128(
                "00000000000000000000000000000000",
                "00000000000000000000000000000000"
            ),
            "66e94bd4ef8a2c3b884cfa59ca342b2e"
        );
    }

    #[test]
    fn aes256_fips197_c3_encrypt() {
        // FIPS-197 Appendix C.3.
        assert_eq!(
            e256(
                "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
                "00112233445566778899aabbccddeeff"
            ),
            "8ea2b7ca516745bfeafc49904b496089"
        );
    }

    #[test]
    fn aes256_zero_vector() {
        assert_eq!(
            e256(
                "0000000000000000000000000000000000000000000000000000000000000000",
                "00000000000000000000000000000000"
            ),
            "dc95c078a2408989ad48a21492842087"
        );
    }

    #[test]
    fn aes128_roundtrip() {
        let key = [0x42u8; 16];
        let aes = Aes128::new(&key);
        for i in 0..32u8 {
            let mut b = [i; 16];
            let orig = b;
            aes.encrypt_block(&mut b);
            assert_ne!(b, orig);
            aes.decrypt_block(&mut b);
            assert_eq!(b, orig);
        }
    }

    #[test]
    fn aes256_roundtrip() {
        let key = [0x24u8; 32];
        let aes = Aes256::new(&key);
        for i in 0..32u8 {
            let mut b = [i; 16];
            let orig = b;
            aes.encrypt_block(&mut b);
            assert_ne!(b, orig);
            aes.decrypt_block(&mut b);
            assert_eq!(b, orig);
        }
    }

    #[test]
    fn aes128_avalanche_key() {
        // Flipping one key bit must change the ciphertext.
        let k1 = [0u8; 16];
        let mut k2 = [0u8; 16];
        k2[0] = 1;
        let mut b1 = [0u8; 16];
        let mut b2 = [0u8; 16];
        Aes128::new(&k1).encrypt_block(&mut b1);
        Aes128::new(&k2).encrypt_block(&mut b2);
        assert_ne!(b1, b2);
    }
}
