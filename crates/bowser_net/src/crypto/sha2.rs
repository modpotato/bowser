//! SHA-256, SHA-384, SHA-512 (FIPS-180-4) and HMAC (RFC 2104).
//!
//! Straightforward Merkle-Damgard implementations. All operations are
//! additions, bitwise ops and fixed rotations: constant-time by
//! construction.

/// SHA-256 round constants (FIPS-180-4, Section 4.2.2).
#[rustfmt::skip]
const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-512 round constants (FIPS-180-4, Section 4.2.3).
#[rustfmt::skip]
const K512: [u64; 80] = [
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
];

const IV256: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const IV512: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const IV384: [u64; 8] = [
    0xcbbb9d5dc1059ed8,
    0x629a292a367cd507,
    0x9159015a3070dd17,
    0x152fecd8f70e5939,
    0x67332667ffc00b31,
    0x8eb44a8768581511,
    0xdb0c2e0d64f98fa7,
    0x47b5481dbefa4fa4,
];

fn compress256(h: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().unwrap());
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K256[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
}

fn compress512(h: &mut [u64; 8], block: &[u8; 128]) {
    let mut w = [0u64; 80];
    for i in 0..16 {
        w[i] = u64::from_be_bytes(block[8 * i..8 * i + 8].try_into().unwrap());
    }
    for i in 16..80 {
        let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
        let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
    for i in 0..80 {
        let s1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K512[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
}

/// Incremental SHA-256 hasher (for transcript hashing in TLS).
#[derive(Clone)]
pub struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total: u64,
}

impl Sha256 {
    /// Fresh hasher.
    pub fn new() -> Self {
        Self {
            h: IV256,
            buf: [0; 64],
            buf_len: 0,
            total: 0,
        }
    }

    /// Absorb more input.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let buf = self.buf;
                compress256(&mut self.h, &buf);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let block: &[u8; 64] = data[..64].try_into().unwrap();
            compress256(&mut self.h, block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    /// Pad, append the bit length, and output the 32-byte digest.
    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total.wrapping_mul(8);
        self.update(&[0x80]);
        let zeros = [0u8; 64];
        self.update(&zeros[..(56 + 64 - self.buf_len) % 64]);
        self.update(&bit_len.to_be_bytes());
        debug_assert_eq!(self.buf_len, 0);
        let mut out = [0u8; 32];
        for i in 0..8 {
            out[4 * i..4 * i + 4].copy_from_slice(&self.h[i].to_be_bytes());
        }
        out
    }
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot SHA-256.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

struct Engine512 {
    h: [u64; 8],
    buf: [u8; 128],
    buf_len: usize,
    total: u128,
}

impl Engine512 {
    fn new(iv: [u64; 8]) -> Self {
        Self {
            h: iv,
            buf: [0; 128],
            buf_len: 0,
            total: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u128;
        if self.buf_len > 0 {
            let take = (128 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 128 {
                let buf = self.buf;
                compress512(&mut self.h, &buf);
                self.buf_len = 0;
            }
        }
        while data.len() >= 128 {
            let block: &[u8; 128] = data[..128].try_into().unwrap();
            compress512(&mut self.h, block);
            data = &data[128..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn finalize(mut self) -> [u64; 8] {
        let bit_len = self.total.wrapping_mul(8);
        self.update(&[0x80]);
        let zeros = [0u8; 128];
        self.update(&zeros[..(112 + 128 - self.buf_len) % 128]);
        self.update(&bit_len.to_be_bytes());
        debug_assert_eq!(self.buf_len, 0);
        self.h
    }
}

/// Incremental SHA-512 hasher.
#[derive(Clone)]
pub struct Sha512 {
    inner: Engine512,
}

impl Sha512 {
    /// Fresh hasher.
    pub fn new() -> Self {
        Self {
            inner: Engine512::new(IV512),
        }
    }

    /// Absorb more input.
    pub fn update(&mut self, data: &[u8]) {
        self.inner.update(data);
    }

    /// Pad and output the 64-byte digest.
    pub fn finalize(self) -> [u8; 64] {
        let h = self.inner.finalize();
        let mut out = [0u8; 64];
        for i in 0..8 {
            out[8 * i..8 * i + 8].copy_from_slice(&h[i].to_be_bytes());
        }
        out
    }
}

impl Default for Sha512 {
    fn default() -> Self {
        Self::new()
    }
}

/// Incremental SHA-384 hasher (SHA-512 with different IV, truncated output).
#[derive(Clone)]
pub struct Sha384 {
    inner: Engine512,
}

impl Sha384 {
    /// Fresh hasher.
    pub fn new() -> Self {
        Self {
            inner: Engine512::new(IV384),
        }
    }

    /// Absorb more input.
    pub fn update(&mut self, data: &[u8]) {
        self.inner.update(data);
    }

    /// Pad and output the 48-byte digest.
    pub fn finalize(self) -> [u8; 48] {
        let h = self.inner.finalize();
        let mut out = [0u8; 48];
        for i in 0..6 {
            out[8 * i..8 * i + 8].copy_from_slice(&h[i].to_be_bytes());
        }
        out
    }
}

impl Default for Sha384 {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot SHA-512.
pub fn sha512(data: &[u8]) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(data);
    h.finalize()
}

/// One-shot SHA-384.
pub fn sha384(data: &[u8]) -> [u8; 48] {
    let mut h = Sha384::new();
    h.update(data);
    h.finalize()
}

// `Engine512` needs `Clone` for the public hashers' derives.
impl Clone for Engine512 {
    fn clone(&self) -> Self {
        Self {
            h: self.h,
            buf: self.buf,
            buf_len: self.buf_len,
            total: self.total,
        }
    }
}

fn hmac<const BLOCK: usize, const OUT: usize>(
    key: &[u8],
    msg: &[u8],
    hash: fn(&[u8]) -> [u8; OUT],
) -> [u8; OUT] {
    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK {
        k[..OUT].copy_from_slice(&hash(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; BLOCK];
    let mut opad = [0x5cu8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let mut inner = Vec::with_capacity(BLOCK + msg.len());
    inner.extend_from_slice(&ipad);
    inner.extend_from_slice(msg);
    let inner_hash = hash(&inner);
    let mut outer = Vec::with_capacity(BLOCK + OUT);
    outer.extend_from_slice(&opad);
    outer.extend_from_slice(&inner_hash);
    hash(&outer)
}

/// HMAC-SHA-256 (RFC 2104).
pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    hmac::<64, 32>(key, msg, sha256)
}

/// HMAC-SHA-384 (RFC 2104).
pub fn hmac_sha384(key: &[u8], msg: &[u8]) -> [u8; 48] {
    hmac::<128, 48>(key, msg, sha384)
}

/// HMAC-SHA-512 (RFC 2104).
pub fn hmac_sha512(key: &[u8], msg: &[u8]) -> [u8; 64] {
    hmac::<128, 64>(key, msg, sha512)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hx(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn sha256_empty() {
        assert_eq!(
            hx(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_abc() {
        assert_eq!(
            hx(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_multiblock() {
        // 56-byte message: exercises the two-block padding path.
        let msg = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(
            hx(&sha256(msg)),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn sha256_incremental_matches_oneshot() {
        let msg: Vec<u8> = (0..300u32).map(|i| (i * 7) as u8).collect();
        let mut h = Sha256::new();
        for chunk in msg.chunks(13) {
            h.update(chunk);
        }
        assert_eq!(h.finalize(), sha256(&msg));
    }

    #[test]
    fn sha384_empty() {
        assert_eq!(
            hx(&sha384(b"")),
            "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b"
        );
    }

    #[test]
    fn sha384_abc() {
        assert_eq!(
            hx(&sha384(b"abc")),
            "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7"
        );
    }

    #[test]
    fn sha512_empty() {
        assert_eq!(
            hx(&sha512(b"")),
            "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
        );
    }

    #[test]
    fn sha512_abc() {
        assert_eq!(
            hx(&sha512(b"abc")),
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
        );
    }

    #[test]
    fn sha512_incremental_matches_oneshot() {
        let msg: Vec<u8> = (0..500u32).map(|i| (i * 13) as u8).collect();
        let mut h = Sha512::new();
        for chunk in msg.chunks(7) {
            h.update(chunk);
        }
        assert_eq!(h.finalize(), sha512(&msg));
        let mut h = Sha384::new();
        for chunk in msg.chunks(29) {
            h.update(chunk);
        }
        assert_eq!(h.finalize(), sha384(&msg));
    }

    #[test]
    fn hmac_sha256_rfc4231_case1() {
        // RFC 4231, test case 1.
        let mac = hmac_sha256(&[0x0bu8; 20], b"Hi There");
        assert_eq!(
            hx(&mac),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn hmac_sha256_long_key() {
        // RFC 4231, test case 6: key longer than the block size is hashed first.
        let mac = hmac_sha256(
            &[0xaau8; 131],
            b"Test Using Larger Than Block-Size Key - Hash Key First",
        );
        assert_eq!(
            hx(&mac),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn hmac_sha512_rfc4231_case1() {
        // RFC 4231, test case 1.
        let mac = hmac_sha512(&[0x0bu8; 20], b"Hi There");
        assert_eq!(
            hx(&mac),
            "87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cdedaa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854"
        );
    }

    #[test]
    fn hmac_sha384_self_consistent() {
        // Long key must behave as hash(key).
        let long: Vec<u8> = (0..200u8).collect();
        let hashed = sha384(&long);
        assert_eq!(hmac_sha384(&long, b"msg"), hmac_sha384(&hashed, b"msg"));
        assert_ne!(hmac_sha384(b"k1", b"msg"), hmac_sha384(b"k2", b"msg"));
    }
}
