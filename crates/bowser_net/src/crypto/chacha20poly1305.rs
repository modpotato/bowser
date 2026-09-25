//! ChaCha20 stream cipher and Poly1305 authenticator, plus their
//! AEAD composition (RFC 8439).
//!
//! Both primitives are constant-time by construction: only additions,
//! XORs, multiplications and fixed rotations, with no secret-dependent
//! branches or table lookups.

use super::{Error, ct_eq};

/// One ChaCha quarter round on state words (a, b, c, d).
fn quarter_round(x: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    x[a] = x[a].wrapping_add(x[b]);
    x[d] ^= x[a];
    x[d] = x[d].rotate_left(16);
    x[c] = x[c].wrapping_add(x[d]);
    x[b] ^= x[c];
    x[b] = x[b].rotate_left(12);
    x[a] = x[a].wrapping_add(x[b]);
    x[d] ^= x[a];
    x[d] = x[d].rotate_left(8);
    x[c] = x[c].wrapping_add(x[d]);
    x[b] ^= x[c];
    x[b] = x[b].rotate_left(7);
}

/// The ChaCha20 block function (RFC 8439, Section 2.3).
pub(crate) fn chacha20_block(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u8; 64] {
    let mut x = [0u32; 16];
    x[0] = 0x6170_7865;
    x[1] = 0x3320_646e;
    x[2] = 0x7962_2d32;
    x[3] = 0x6b20_6574;
    for i in 0..8 {
        x[4 + i] = u32::from_le_bytes(key[4 * i..4 * i + 4].try_into().unwrap());
    }
    x[12] = counter;
    for i in 0..3 {
        x[13 + i] = u32::from_le_bytes(nonce[4 * i..4 * i + 4].try_into().unwrap());
    }

    let mut w = x;
    for _ in 0..10 {
        // Column rounds.
        quarter_round(&mut w, 0, 4, 8, 12);
        quarter_round(&mut w, 1, 5, 9, 13);
        quarter_round(&mut w, 2, 6, 10, 14);
        quarter_round(&mut w, 3, 7, 11, 15);
        // Diagonal rounds.
        quarter_round(&mut w, 0, 5, 10, 15);
        quarter_round(&mut w, 1, 6, 11, 12);
        quarter_round(&mut w, 2, 7, 8, 13);
        quarter_round(&mut w, 3, 4, 9, 14);
    }
    for i in 0..16 {
        w[i] = w[i].wrapping_add(x[i]);
    }

    let mut out = [0u8; 64];
    for i in 0..16 {
        out[4 * i..4 * i + 4].copy_from_slice(&w[i].to_le_bytes());
    }
    out
}

/// XOR `input` with the ChaCha20 keystream starting at `counter`.
fn chacha20_xor(key: &[u8; 32], counter: u32, nonce: &[u8; 12], input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    let mut ctr = counter;
    for chunk in input.chunks(64) {
        let ks = chacha20_block(key, ctr, nonce);
        ctr = ctr.wrapping_add(1);
        out.extend(chunk.iter().zip(ks.iter()).map(|(&a, &b)| a ^ b));
    }
    out
}

/// Poly1305 one-time authenticator (RFC 8439, Section 2.5).
///
/// The accumulator, `r` and the message blocks are held in radix 2^44
/// (three 44-bit limbs), so every intermediate product fits in `u128`.
/// Reduction uses 2^130 = 5 (mod p): B^3 = 2^132 = 20 and B^4 = 20*B.
struct Poly1305 {
    h: [u64; 3],
    r: [u64; 3],
    s: [u64; 2],
}

/// 44 low bits set.
const MASK44: u64 = 0x0fff_ffff_ffff;
/// 2^130 - 5 in radix 2^44.
const P: [u64; 3] = [0x0fff_ffff_fffb, 0x0fff_ffff_ffff, 0x03ff_ffff_ffff];
/// Poly1305 `r` clamping mask (RFC 8439, Section 2.5).
const CLAMP: u128 = 0x0fff_fffc_0fff_fffc_0fff_fffc_0fff_ffff;

impl Poly1305 {
    fn new(key: &[u8; 32]) -> Self {
        let r = u128::from_le_bytes(key[0..16].try_into().unwrap()) & CLAMP;
        Self {
            h: [0, 0, 0],
            r: [
                (r & MASK44 as u128) as u64,
                ((r >> 44) & MASK44 as u128) as u64,
                (r >> 88) as u64,
            ],
            s: [
                u64::from_le_bytes(key[16..24].try_into().unwrap()),
                u64::from_le_bytes(key[24..32].try_into().unwrap()),
            ],
        }
    }

    /// Absorb whole message: for each 16-byte chunk, h = (h + n) * r mod p,
    /// where n is the chunk read little-endian with a 1 bit appended.
    fn absorb(&mut self, msg: &[u8]) {
        for chunk in msg.chunks(16) {
            let mut n = [0u64; 3];
            for (i, &b) in chunk.iter().enumerate() {
                let bit = (i * 8) as u32;
                n[(bit / 44) as usize] |= (b as u64) << (bit % 44);
            }
            // Append the 0x01 byte beyond the chunk.
            let bit = (chunk.len() * 8) as u32;
            n[(bit / 44) as usize] |= 1u64 << (bit % 44);

            // h += n (limbs stay small; products below stay in u128).
            let mut c = 0u64;
            let mut h = self.h;
            for i in 0..3 {
                let v = h[i] + n[i] + c;
                c = v >> 44;
                h[i] = v & MASK44;
            }

            // h *= r, folding B^3 -> 20 and B^4 -> 20*B (since B^3 = 2^132
            // = 4 * 2^130 and 2^130 = 5 mod p, so B^3 = 20 mod p).
            let (h0, h1, h2) = (h[0] as u128, h[1] as u128, h[2] as u128);
            let (r0, r1, r2) = (self.r[0] as u128, self.r[1] as u128, self.r[2] as u128);
            let mut d0 = h0 * r0 + 20 * (h1 * r2 + h2 * r1);
            let mut d1 = h0 * r1 + h1 * r0 + 20 * (h2 * r2);
            let mut d2 = h0 * r2 + h1 * r1 + h2 * r0;
            Self::carry128(&mut d0, &mut d1, &mut d2);
            self.h = [d0 as u64, d1 as u64, d2 as u64];
        }
    }

    /// Absorb `data` padded with zeros to a multiple of 16 bytes.
    fn absorb_padded(&mut self, data: &[u8]) {
        let rem = data.len() % 16;
        let (full, tail) = data.split_at(data.len() - rem);
        self.absorb(full);
        if !tail.is_empty() {
            let mut last = [0u8; 16];
            last[..tail.len()].copy_from_slice(tail);
            self.absorb(&last);
        }
    }

    /// Carry the (possibly large) product limbs d0 + d1*B + d2*B^2,
    /// reducing mod p = 2^130 - 5 via B^3 = 20. On return each limb
    /// is below 2^44 (d1 may exceed it by a tiny amount, harmless for
    /// the next multiply whose products still fit comfortably in u128).
    fn carry128(d0: &mut u128, d1: &mut u128, d2: &mut u128) {
        *d1 += *d0 >> 44;
        *d0 &= MASK44 as u128;
        *d2 += *d1 >> 44;
        *d1 &= MASK44 as u128;
        *d0 += (*d2 >> 44) * 20;
        *d2 &= MASK44 as u128;
        *d1 += *d0 >> 44;
        *d0 &= MASK44 as u128;
    }

    /// Carry the accumulator limbs (u64 version of [`Self::carry128`]).
    fn carry(h: &mut [u64; 3]) {
        let mut d0 = h[0] as u128;
        let mut d1 = h[1] as u128;
        let mut d2 = h[2] as u128;
        Self::carry128(&mut d0, &mut d1, &mut d2);
        // A second pass squeezes out the small residue left in d1, so all
        // limbs end up strictly below 2^44.
        Self::carry128(&mut d0, &mut d1, &mut d2);
        h[0] = d0 as u64;
        h[1] = d1 as u64;
        h[2] = d2 as u64;
    }

    /// h = h - p if h >= p, else h. Branch-free constant-time select;
    /// the caller must ensure h < 2^132 with limbs below 2^44, and calls
    /// this enough times to drive h below p.
    fn csub_p(h: [u64; 3]) -> [u64; 3] {
        let mut t = [0u64; 3];
        let mut borrow = 0u64;
        for i in 0..3 {
            let (v1, b1) = h[i].overflowing_sub(P[i]);
            let (v2, b2) = v1.overflowing_sub(borrow);
            // Keep only the low 44 bits: the borrow is already propagated
            // to the next limb, so the limb value is v2 mod 2^44.
            t[i] = v2 & MASK44;
            borrow = u64::from(b1 || b2);
        }
        // mask is all-ones exactly when borrow == 0, i.e. h >= p.
        let mask = 0u64.wrapping_sub(1u64.wrapping_sub(borrow));
        [0, 1, 2].map(|i| (h[i] & !mask) | (t[i] & mask))
    }

    /// Finalize: reduce the accumulator mod p, add `s` mod 2^128, and
    /// serialize little-endian.
    fn finalize(mut self) -> [u8; 16] {
        Self::carry(&mut self.h);
        // Now h < 2^44 + 2^88 + 2^132 < 5p, so five conditional
        // subtractions drive it below p.
        let mut h = self.h;
        for _ in 0..5 {
            h = Self::csub_p(h);
        }
        let acc = h[0] as u128 | (h[1] as u128) << 44 | (h[2] as u128) << 88;
        // (h + s) mod 2^128: the sum can exceed u128, so wrap.
        let tag = acc
            .wrapping_add(self.s[0] as u128)
            .wrapping_add((self.s[1] as u128) << 64);
        tag.to_le_bytes()
    }
}

/// ChaCha20-Poly1305 AEAD (RFC 8439, Section 2.8).
#[derive(Clone)]
pub struct ChaCha20Poly1305 {
    key: [u8; 32],
}

impl ChaCha20Poly1305 {
    /// Build from a 256-bit key.
    pub fn new(key: &[u8; 32]) -> Self {
        Self { key: *key }
    }

    /// Encrypt and authenticate. Returns `(ciphertext, tag)`.
    ///
    /// The nonce must be unique per key; it is 96 bits as in TLS 1.3.
    pub fn seal(&self, nonce: &[u8; 12], aad: &[u8], plaintext: &[u8]) -> (Vec<u8>, [u8; 16]) {
        let block0 = chacha20_block(&self.key, 0, nonce);
        let mut mac = Poly1305::new(&block0[..32].try_into().unwrap());
        let ct = chacha20_xor(&self.key, 1, nonce, plaintext);
        mac.absorb_padded(aad);
        mac.absorb_padded(&ct);
        let mut lens = [0u8; 16];
        lens[..8].copy_from_slice(&(aad.len() as u64).to_le_bytes());
        lens[8..].copy_from_slice(&(ct.len() as u64).to_le_bytes());
        mac.absorb(&lens);
        (ct, mac.finalize())
    }

    /// Verify the tag and decrypt. Fails closed on any mismatch.
    pub fn open(
        &self,
        nonce: &[u8; 12],
        aad: &[u8],
        ciphertext: &[u8],
        tag: &[u8; 16],
    ) -> Result<Vec<u8>, Error> {
        let block0 = chacha20_block(&self.key, 0, nonce);
        let mut mac = Poly1305::new(&block0[..32].try_into().unwrap());
        mac.absorb_padded(aad);
        mac.absorb_padded(ciphertext);
        let mut lens = [0u8; 16];
        lens[..8].copy_from_slice(&(aad.len() as u64).to_le_bytes());
        lens[8..].copy_from_slice(&(ciphertext.len() as u64).to_le_bytes());
        mac.absorb(&lens);
        if !ct_eq(&mac.finalize(), tag) {
            return Err(Error::AuthenticationFailed);
        }
        Ok(chacha20_xor(&self.key, 1, nonce, ciphertext))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::hex;

    fn h(s: &str) -> Vec<u8> {
        hex(s)
    }

    fn hx(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    fn key_00_1f() -> [u8; 32] {
        let v = h("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f");
        v.try_into().unwrap()
    }

    #[test]
    fn chacha20_block_rfc8439_232() {
        // RFC 8439, Section 2.3.2.
        let nonce: [u8; 12] = h("000000090000004a00000000").try_into().unwrap();
        let out = chacha20_block(&key_00_1f(), 1, &nonce);
        assert_eq!(
            hx(&out),
            "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4ed2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e"
        );
    }

    #[test]
    fn chacha20_encrypt_rfc8439_242() {
        // RFC 8439, Section 2.4.2.
        let nonce: [u8; 12] = h("000000000000004a00000000").try_into().unwrap();
        let pt = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
        let ct = chacha20_xor(&key_00_1f(), 1, &nonce, pt);
        assert_eq!(
            hx(&ct),
            "6e2e359a2568f98041ba0728dd0d6981e97e7aec1d4360c20a27afccfd9fae0bf91b65c5524733ab8f593dabcd62b3571639d624e65152ab8f530c359f0861d807ca0dbf500d6a6156a38e088a22b65e52bc514d16ccf806818ce91ab77937365af90bbf74a35be6b40b8eedf2785e42874d"
        );
        // Decryption is the same operation.
        assert_eq!(chacha20_xor(&key_00_1f(), 1, &nonce, &ct), pt);
    }

    #[test]
    fn poly1305_rfc8439_252() {
        // RFC 8439, Section 2.5.2.
        let key = h("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b");
        let mut mac = Poly1305::new(&key.try_into().unwrap());
        mac.absorb(b"Cryptographic Forum Research Group");
        assert_eq!(hx(&mac.finalize()), "a8061dc1305136c6c22b8baf0c0127a9");
    }

    #[test]
    fn poly1305_empty_message() {
        // MAC of the empty message is just `s` serialized.
        let key = h("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b");
        let mac = Poly1305::new(&key.try_into().unwrap());
        assert_eq!(hx(&mac.finalize()), "0103808afb0db2fd4abff6af4149f51b");
    }

    #[test]
    fn poly1305_key_gen_rfc8439_262() {
        // RFC 8439, Section 2.6.2.
        let key: [u8; 32] = h("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f")
            .try_into()
            .unwrap();
        let nonce: [u8; 12] = h("000000000001020304050607").try_into().unwrap();
        let block = chacha20_block(&key, 0, &nonce);
        assert_eq!(
            hx(&block[..32]),
            "8ad5a08b905f81cc815040274ab29471a833b637e3fd0da508dbb8e2fdd1a646"
        );
    }

    #[test]
    fn aead_rfc8439_282() {
        // RFC 8439, Section 2.8.2.
        let key: [u8; 32] = h("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f")
            .try_into()
            .unwrap();
        let nonce: [u8; 12] = h("070000004041424344454647").try_into().unwrap();
        let aad = h("50515253c0c1c2c3c4c5c6c7");
        let pt = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
        let aead = ChaCha20Poly1305::new(&key);
        let (ct, tag) = aead.seal(&nonce, &aad, pt);
        assert_eq!(
            hx(&ct),
            "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d63dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b3692ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc3ff4def08e4b7a9de576d26586cec64b6116"
        );
        assert_eq!(hx(&tag), "1ae10b594f09e26a7e902ecbd0600691");
        assert_eq!(aead.open(&nonce, &aad, &ct, &tag).unwrap(), pt);
    }

    #[test]
    fn aead_rfc8439_a5_decrypt_vectors() {
        // RFC 8439, Appendix A.5: decrypt the given ciphertexts.
        let key: [u8; 32] = h("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f")
            .try_into()
            .unwrap();
        let nonce: [u8; 12] = h("070000004041424344454647").try_into().unwrap();
        let aad = h("50515253c0c1c2c3c4c5c6c7");
        let ct = h(
            "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d63dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b3692ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc3ff4def08e4b7a9de576d26586cec64b6116",
        );
        let tag: [u8; 16] = h("1ae10b594f09e26a7e902ecbd0600691").try_into().unwrap();
        let pt = ChaCha20Poly1305::new(&key)
            .open(&nonce, &aad, &ct, &tag)
            .unwrap();
        assert_eq!(pt, b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.");
    }

    #[test]
    fn aead_tamper_rejected() {
        let aead = ChaCha20Poly1305::new(&[0x33u8; 32]);
        let nonce = [0x44u8; 12];
        let (ct, tag) = aead.seal(&nonce, b"aad", b"plaintext");
        let mut bad_tag = tag;
        bad_tag[15] ^= 0x80;
        assert_eq!(
            aead.open(&nonce, b"aad", &ct, &bad_tag),
            Err(Error::AuthenticationFailed)
        );
        let mut bad_ct = ct.clone();
        bad_ct[0] ^= 1;
        assert_eq!(
            aead.open(&nonce, b"aad", &bad_ct, &tag),
            Err(Error::AuthenticationFailed)
        );
        assert_eq!(
            aead.open(&nonce, b"other aad", &ct, &tag),
            Err(Error::AuthenticationFailed)
        );
    }

    #[test]
    fn aead_length_sweep() {
        // Padding paths: every plaintext/AAD length mod 16.
        let aead = ChaCha20Poly1305::new(&[0x5au8; 32]);
        for pt_len in 0..40usize {
            for aad_len in [0, 1, 15, 16, 17, 31] {
                let pt: Vec<u8> = (0..pt_len as u8).collect();
                let aad: Vec<u8> = (0..aad_len as u8).map(|x| x.wrapping_add(3)).collect();
                let nonce = [pt_len as u8; 12];
                let (ct, tag) = aead.seal(&nonce, &aad, &pt);
                assert_eq!(aead.open(&nonce, &aad, &ct, &tag).unwrap(), pt);
            }
        }
    }
}
