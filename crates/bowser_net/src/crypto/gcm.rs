//! AES-GCM authenticated encryption (NIST SP 800-38D).
//!
//! GHASH is implemented with a branch-free shift-and-conditional-XOR field
//! multiplier, so the hash subkey `H` (derived from the AES key) never
//! influences control flow.

use super::aes::ExpandedKey;
use super::{Error, ct_eq};

/// GHASH reduction polynomial: x^128 + x^7 + x^2 + x + 1, represented with
/// the x^127 coefficient in the most significant bit.
const R: u128 = 0xe100_0000_0000_0000_0000_0000_0000_0000;

/// Multiply two GF(2^128) elements, MSB-first (SP 800-38D, Algorithm 1).
///
/// Branch-free: the conditional XORs use all-ones/all-zero masks derived
/// from the tested bits.
fn gfmul(mut x: u128, mut v: u128) -> u128 {
    let mut z = 0u128;
    for _ in 0..128 {
        let x_msb = (x >> 127) & 1;
        z ^= v & x_msb.wrapping_neg();
        let v_lsb = v & 1;
        v >>= 1;
        v ^= R & v_lsb.wrapping_neg();
        x <<= 1;
    }
    z
}

fn block_to_u128(block: &[u8]) -> u128 {
    let mut b = [0u8; 16];
    b[..block.len()].copy_from_slice(block);
    u128::from_be_bytes(b)
}

/// GHASH_H over `data`, zero-padding the final partial block.
fn ghash(h: u128, data: &[u8]) -> u128 {
    let mut y = 0u128;
    for chunk in data.chunks(16) {
        y = gfmul(y ^ block_to_u128(chunk), h);
    }
    y
}

/// GHASH_H(A || 0^pad || C || 0^pad || [len(A)]_64 || [len(C)]_64).
/// Lengths are bit lengths encoded as 64-bit big-endian values.
fn ghash_aad_ct(h: u128, aad: &[u8], ct: &[u8]) -> u128 {
    let mut y = ghash(h, aad);
    for chunk in ct.chunks(16) {
        y = gfmul(y ^ block_to_u128(chunk), h);
    }
    let len_block = (((aad.len() as u128) * 8) & 0xffff_ffff_ffff_ffff) << 64
        | (((ct.len() as u128) * 8) & 0xffff_ffff_ffff_ffff);
    gfmul(y ^ len_block, h)
}

/// Derive J0 from the nonce. 12-byte nonces use the fast path
/// (IV || 0^31 || 1); all other lengths go through GHASH_H.
fn derive_j0(h: u128, nonce: &[u8]) -> u128 {
    if nonce.len() == 12 {
        let mut b = [0u8; 16];
        b[..12].copy_from_slice(nonce);
        b[15] = 1;
        u128::from_be_bytes(b)
    } else {
        let mut y = ghash(h, nonce);
        let len_bits = ((nonce.len() as u128) * 8) & 0xffff_ffff_ffff_ffff;
        y = gfmul(y ^ len_bits, h);
        y
    }
}

/// Increment the low 32 bits of the counter block (SP 800-38D `inc_32`).
fn inc32(cb: u128, i: u32) -> u128 {
    (cb & !0xffff_ffff) | u128::from((cb as u32).wrapping_add(i))
}

#[derive(Clone)]
struct Gcm {
    aes: ExpandedKey,
}

impl Gcm {
    fn hash_subkey(&self) -> u128 {
        let mut h = [0u8; 16];
        self.aes.encrypt_block(&mut h);
        u128::from_be_bytes(h)
    }

    fn encrypt_block(&self, block: [u8; 16]) -> [u8; 16] {
        let mut b = block;
        self.aes.encrypt_block(&mut b);
        b
    }

    /// Raw CTR transform (encrypt and decrypt are identical).
    fn ctr(&self, j0: u128, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(data.len());
        for (i, chunk) in data.chunks(16).enumerate() {
            let ks = self.encrypt_block(inc32(j0, (i as u32) + 1).to_be_bytes());
            out.extend(chunk.iter().zip(ks.iter()).map(|(&p, &k)| p ^ k));
        }
        out
    }

    fn seal(&self, nonce: &[u8], aad: &[u8], plaintext: &[u8]) -> (Vec<u8>, [u8; 16]) {
        let h = self.hash_subkey();
        let j0 = derive_j0(h, nonce);
        let ct = self.ctr(j0, plaintext);
        let s = ghash_aad_ct(h, aad, &ct);
        let tag_block = u128::from_be_bytes(self.encrypt_block(j0.to_be_bytes())) ^ s;
        (ct, tag_block.to_be_bytes())
    }

    fn open(
        &self,
        nonce: &[u8],
        aad: &[u8],
        ciphertext: &[u8],
        tag: &[u8; 16],
    ) -> Result<Vec<u8>, Error> {
        let h = self.hash_subkey();
        let j0 = derive_j0(h, nonce);
        let s = ghash_aad_ct(h, aad, ciphertext);
        let expected =
            (u128::from_be_bytes(self.encrypt_block(j0.to_be_bytes())) ^ s).to_be_bytes();
        if !ct_eq(&expected, tag) {
            return Err(Error::AuthenticationFailed);
        }
        Ok(self.ctr(j0, ciphertext))
    }
}

/// AES-128-GCM AEAD.
#[derive(Clone)]
pub struct Aes128Gcm {
    inner: Gcm,
}

impl Aes128Gcm {
    /// Build from a 128-bit key.
    pub fn new(key: &[u8; 16]) -> Self {
        Self {
            inner: Gcm {
                aes: ExpandedKey::expand_128(key),
            },
        }
    }

    /// Encrypt and authenticate. Returns `(ciphertext, tag)`.
    pub fn seal(&self, nonce: &[u8], aad: &[u8], plaintext: &[u8]) -> (Vec<u8>, [u8; 16]) {
        self.inner.seal(nonce, aad, plaintext)
    }

    /// Verify the tag and decrypt. Fails closed on any mismatch.
    pub fn open(
        &self,
        nonce: &[u8],
        aad: &[u8],
        ciphertext: &[u8],
        tag: &[u8; 16],
    ) -> Result<Vec<u8>, Error> {
        self.inner.open(nonce, aad, ciphertext, tag)
    }
}

/// AES-256-GCM AEAD.
#[derive(Clone)]
pub struct Aes256Gcm {
    inner: Gcm,
}

impl Aes256Gcm {
    /// Build from a 256-bit key.
    pub fn new(key: &[u8; 32]) -> Self {
        Self {
            inner: Gcm {
                aes: ExpandedKey::expand_256(key),
            },
        }
    }

    /// Encrypt and authenticate. Returns `(ciphertext, tag)`.
    pub fn seal(&self, nonce: &[u8], aad: &[u8], plaintext: &[u8]) -> (Vec<u8>, [u8; 16]) {
        self.inner.seal(nonce, aad, plaintext)
    }

    /// Verify the tag and decrypt. Fails closed on any mismatch.
    pub fn open(
        &self,
        nonce: &[u8],
        aad: &[u8],
        ciphertext: &[u8],
        tag: &[u8; 16],
    ) -> Result<Vec<u8>, Error> {
        self.inner.open(nonce, aad, ciphertext, tag)
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

    /// McGrew-Viega GCM test case 1: empty plaintext, empty AAD.
    #[test]
    fn gcm_case1_empty() {
        let g = Aes128Gcm::new(&[0u8; 16]);
        let (ct, tag) = g.seal(&[0u8; 12], &[], &[]);
        assert!(ct.is_empty());
        assert_eq!(hx(&tag), "58e2fccefa7e3061367f1d57a4e7455a");
        assert_eq!(
            g.open(&[0u8; 12], &[], &ct, &tag).unwrap(),
            Vec::<u8>::new()
        );
    }

    /// McGrew-Viega GCM test case 2: one zero block.
    #[test]
    fn gcm_case2_one_block() {
        let g = Aes128Gcm::new(&[0u8; 16]);
        let (ct, tag) = g.seal(&[0u8; 12], &[], &[0u8; 16]);
        assert_eq!(hx(&ct), "0388dace60b6a392f328c2b971b2fe78");
        assert_eq!(hx(&tag), "ab6e47d42cec13bdf53a67b21257bddf");
        assert_eq!(g.open(&[0u8; 12], &[], &ct, &tag).unwrap(), vec![0u8; 16]);
    }

    /// McGrew-Viega GCM test case 3.
    #[test]
    fn gcm_case3() {
        let g = Aes128Gcm::new(&h("feffe9928665731c6d6a8f9467308308").try_into().unwrap());
        let (ct, tag) = g.seal(
            &h("cafebabefacedbaddecaf888"),
            &[],
            &h("d9313225f88406e5a559a55b5d05df"),
        );
        assert_eq!(hx(&ct), "42831ec2217774244b728d2976202d");
        assert_eq!(hx(&tag), "d33ebf862b06bdc2727f09eb0c026b12");
        assert_eq!(
            g.open(&h("cafebabefacedbaddecaf888"), &[], &ct, &tag)
                .unwrap(),
            h("d9313225f88406e5a559a55b5d05df")
        );
    }

    /// McGrew-Viega GCM test case 4: 60-byte plaintext, 20-byte AAD.
    /// Only the tag is asserted (it covers the ciphertext through GHASH).
    #[test]
    fn gcm_case4_tag() {
        let g = Aes128Gcm::new(&h("feffe9928665731c6d6a8f9467308308").try_into().unwrap());
        let pt = h(
            "d9313225f88406e5a559a55b5d05dfd9313225f88406e5a559a55b5d05dfd9313225f88406e5a559a55b5d05dfd9313225f88406e5a559a55b5d05df",
        );
        let aad = h("feedfacedeadbeeffeedfacedeadbeefabaddad2");
        let (ct, tag) = g.seal(&h("cafebabefacedbaddecaf888"), &aad, &pt);
        assert_eq!(ct.len(), 60);
        assert_eq!(hx(&tag), "195fc60690036333e8a4a7d8f8b1232c");
        assert_eq!(
            g.open(&h("cafebabefacedbaddecaf888"), &aad, &ct, &tag)
                .unwrap(),
            pt
        );
    }

    /// McGrew-Viega GCM test case 6: 60-byte IV exercises the
    /// GHASH-based J0 derivation. 60-byte plaintext, 20-byte AAD.
    #[test]
    fn gcm_case6_long_nonce() {
        let g = Aes128Gcm::new(&h("feffe9928665731c6d6a8f9467308308").try_into().unwrap());
        let iv = h(
            "9313225df88406e555909c5aff5269aa6a7a9538534f7da1e4c303d2a318a728c3c0c95156809539fcf0e2429a6b525416aedbf5a0de6a57a637b39b",
        );
        assert_eq!(iv.len(), 60);
        let pt = h(
            "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39",
        );
        assert_eq!(pt.len(), 60);
        let aad = h("feedfacedeadbeeffeedfacedeadbeefabaddad2");
        assert_eq!(aad.len(), 20);
        let (ct, tag) = g.seal(&iv, &aad, &pt);
        assert_eq!(
            hx(&ct),
            "8ce24998625615b603a033aca13fb894be9112a5c3a211a8ba262a3cca7e2ca701e4a9a4fba43c90ccdcb281d48c7c6fd62875d2aca417034c34aee5"
        );
        assert_eq!(hx(&tag), "619cc5aefffe0bfa462af43c1699d050");
        assert_eq!(g.open(&iv, &aad, &ct, &tag).unwrap(), pt);
    }

    #[test]
    fn gcm256_roundtrip() {
        let g = Aes256Gcm::new(&[0x77u8; 32]);
        let nonce = [0x99u8; 12];
        let aad = b"associated data";
        let pt = b"the quick brown fox jumps over the lazy dog, twice!";
        let (ct, tag) = g.seal(&nonce, aad, pt);
        assert_ne!(ct, pt);
        assert_eq!(g.open(&nonce, aad, &ct, &tag).unwrap(), pt);
    }

    #[test]
    fn gcm_tampered_tag_rejected() {
        let g = Aes128Gcm::new(&[0x11u8; 16]);
        let (ct, mut tag) = g.seal(&[0x22u8; 12], b"aad", b"secret");
        tag[0] ^= 1;
        assert_eq!(
            g.open(&[0x22u8; 12], b"aad", &ct, &tag),
            Err(Error::AuthenticationFailed)
        );
    }

    #[test]
    fn gcm_tampered_ciphertext_rejected() {
        let g = Aes128Gcm::new(&[0x11u8; 16]);
        let (mut ct, tag) = g.seal(&[0x22u8; 12], b"aad", b"secret");
        ct[0] ^= 1;
        assert_eq!(
            g.open(&[0x22u8; 12], b"aad", &ct, &tag),
            Err(Error::AuthenticationFailed)
        );
    }

    #[test]
    fn gcm_tampered_aad_rejected() {
        let g = Aes128Gcm::new(&[0x11u8; 16]);
        let (ct, tag) = g.seal(&[0x22u8; 12], b"aad", b"secret");
        assert_eq!(
            g.open(&[0x22u8; 12], b"bad", &ct, &tag),
            Err(Error::AuthenticationFailed)
        );
    }

    #[test]
    fn gcm_partial_block_lengths() {
        // Exercise every plaintext/AAD length mod 16 for padding paths.
        let g = Aes128Gcm::new(&[0x5au8; 16]);
        for pt_len in 0..40usize {
            for aad_len in [0, 1, 15, 16, 17, 31] {
                let pt: Vec<u8> = (0..pt_len as u8).collect();
                let aad: Vec<u8> = (0..aad_len as u8).map(|x| x.wrapping_add(7)).collect();
                let nonce = [pt_len as u8; 12];
                let (ct, tag) = g.seal(&nonce, &aad, &pt);
                assert_eq!(g.open(&nonce, &aad, &ct, &tag).unwrap(), pt);
            }
        }
    }
}
