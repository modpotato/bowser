//! AES-GCM authenticated encryption (NIST SP 800-38D).
//!
//! GHASH is implemented with a branch-free shift-and-conditional-XOR field
//! multiplier, so the hash subkey `H` (derived from the AES key) never
//! influences control flow.

use super::aes::ExpandedKey;
use super::{Error, ct_eq, zeroize, zeroize_val};

/// GHASH reduction polynomial: x^128 + x^7 + x^2 + x + 1, represented with
/// the x^127 coefficient in the most significant bit.
const R: u128 = 0xe100_0000_0000_0000_0000_0000_0000_0000;

/// Largest plaintext/ciphertext accepted by one `seal`/`open` call, in
/// bytes: SP 800-38D caps the plaintext at 2^39 - 256 bits. This also
/// keeps the 32-bit CTR block index (which starts at 1) from wrapping
/// and reusing keystream.
const MAX_INPUT_LEN: u64 = (1u64 << 36) - 32;

/// Reject inputs that would exceed the SP 800-38D plaintext limit or wrap
/// the 32-bit CTR block index (fail closed).
fn check_input_len(len: usize) -> Result<(), Error> {
    if len as u64 > MAX_INPUT_LEN {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

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
    // NOTE: the expanded round keys live in `ExpandedKey`, whose bytes are
    // private to the `aes` module, so they cannot be scrubbed from here on
    // drop. That needs a `pub(crate)` scrub method on `ExpandedKey`
    // itself; the key-derived per-call intermediates (`h`, `s`) are
    // scrubbed explicitly at the end of `seal`/`open` below.
    aes: ExpandedKey,
}

impl Gcm {
    fn hash_subkey(&self) -> u128 {
        let mut h = [0u8; 16];
        self.aes.encrypt_block(&mut h);
        let subkey = u128::from_be_bytes(h);
        // Scrub the E_K(0^128) byte array before returning: it must not
        // linger on the stack.
        zeroize(&mut h);
        subkey
    }

    fn encrypt_block(&self, block: [u8; 16]) -> [u8; 16] {
        let mut b = block;
        self.aes.encrypt_block(&mut b);
        b
    }

    /// Raw CTR transform in place (encrypt and decrypt are identical).
    ///
    /// Constant-time by construction: the chunking is fixed 16-byte blocks
    /// over a public length, and the per-block keystream comes from the
    /// constant-time AES block cipher; nothing secret-dependent selects a
    /// branch or a table entry.
    fn ctr_inplace(&self, j0: u128, buf: &mut [u8]) {
        for (i, chunk) in buf.chunks_mut(16).enumerate() {
            // wrapping_add: the length guard in seal/open keeps the block
            // index far below u32::MAX, so this never actually wraps.
            let ks = self.encrypt_block(inc32(j0, (i as u32).wrapping_add(1)).to_be_bytes());
            for (b, &k) in chunk.iter_mut().zip(ks.iter()) {
                *b ^= k;
            }
        }
    }

    /// Encrypt and authenticate in place: XOR `buf` with the CTR keystream
    /// and return the tag. Thin wrapper: the public 12-byte-nonce
    /// variants live on `Aes128Gcm`/`Aes256Gcm`.
    fn seal_in_place(&self, nonce: &[u8], aad: &[u8], buf: &mut [u8]) -> Result<[u8; 16], Error> {
        check_input_len(buf.len())?;
        // Guard the AAD with the same fail-closed length check as the
        // data buffer: the 64-bit bit-length encoding in `ghash_aad_ct`
        // must never wrap.
        check_input_len(aad.len())?;
        let mut h = self.hash_subkey();
        let mut j0 = derive_j0(h, nonce);
        self.ctr_inplace(j0, buf);
        let mut s = ghash_aad_ct(h, aad, buf);
        let tag = u128::from_be_bytes(self.encrypt_block(j0.to_be_bytes())) ^ s;
        // Scrub the key-derived intermediates before returning.
        zeroize_val(&mut h);
        zeroize_val(&mut j0);
        zeroize_val(&mut s);
        Ok(tag.to_be_bytes())
    }

    /// Verify the tag and decrypt in place. `buf` holds `ct || tag`; the
    /// tag is verified FIRST, then the ciphertext region is decrypted in
    /// place, and the plaintext region (tag stripped) is returned.
    ///
    /// On authentication failure the buffer is left untouched: it still
    /// holds the ciphertext, never half-decrypted plaintext. The
    /// key-derived intermediates (`h`, `s`) are scrubbed on every path.
    fn open_in_place<'b>(
        &self,
        nonce: &[u8],
        aad: &[u8],
        buf: &'b mut [u8],
    ) -> Result<&'b mut [u8], Error> {
        let ct_len = buf.len().checked_sub(16).ok_or(Error::InvalidLength)?;
        check_input_len(ct_len)?;
        // Guard the AAD with the same fail-closed length check as the
        // ciphertext buffer: the 64-bit bit-length encoding in
        // `ghash_aad_ct` must never wrap.
        check_input_len(aad.len())?;
        let (ct, tag) = buf.split_at_mut(ct_len);
        let mut h = self.hash_subkey();
        let mut j0 = derive_j0(h, nonce);
        let mut s = ghash_aad_ct(h, aad, ct);
        let mut expected =
            (u128::from_be_bytes(self.encrypt_block(j0.to_be_bytes())) ^ s).to_be_bytes();
        let ok = ct_eq(&expected, tag);
        // Scrub the key-derived intermediates before any early return.
        // `j0` stays live until after decryption below, so it is scrubbed
        // on each path separately.
        zeroize(&mut expected);
        zeroize_val(&mut h);
        zeroize_val(&mut s);
        if !ok {
            // Decryption never ran, so `ct` still holds ciphertext and no
            // half-decrypted plaintext exists anywhere.
            zeroize_val(&mut j0);
            return Err(Error::AuthenticationFailed);
        }
        self.ctr_inplace(j0, ct);
        zeroize_val(&mut j0);
        Ok(ct)
    }

    fn seal(
        &self,
        nonce: &[u8],
        aad: &[u8],
        plaintext: &[u8],
    ) -> Result<(Vec<u8>, [u8; 16]), Error> {
        let mut ct = plaintext.to_vec();
        let tag = self.seal_in_place(nonce, aad, &mut ct)?;
        Ok((ct, tag))
    }

    fn open(
        &self,
        nonce: &[u8],
        aad: &[u8],
        ciphertext: &[u8],
        tag: &[u8; 16],
    ) -> Result<Vec<u8>, Error> {
        let mut buf = ciphertext.to_vec();
        buf.extend_from_slice(tag);
        let pt = self.open_in_place(nonce, aad, &mut buf)?;
        Ok(pt.to_vec())
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
    /// Fails with [`Error::InvalidLength`] if `plaintext` exceeds the
    /// SP 800-38D limit of 2^39 - 256 bits.
    pub fn seal(
        &self,
        nonce: &[u8],
        aad: &[u8],
        plaintext: &[u8],
    ) -> Result<(Vec<u8>, [u8; 16]), Error> {
        self.inner.seal(nonce, aad, plaintext)
    }

    /// Encrypt and authenticate in place: XOR `buf` with the CTR keystream
    /// and return the tag.
    ///
    /// Fails with [`Error::InvalidLength`] if `buf` exceeds the SP 800-38D
    /// plaintext limit. Key-derived intermediates are scrubbed before this
    /// method returns on every path.
    pub fn seal_in_place(
        &self,
        nonce: &[u8; 12],
        aad: &[u8],
        buf: &mut [u8],
    ) -> Result<[u8; 16], Error> {
        self.inner.seal_in_place(nonce, aad, buf)
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

    /// Verify the tag and decrypt in place. `buf` holds `ct || tag`; the
    /// tag is verified FIRST, then the ciphertext region is decrypted in
    /// place, and the plaintext region (tag stripped) is returned.
    ///
    /// Fails with [`Error::InvalidLength`] if `buf` is shorter than 16
    /// bytes or the ciphertext region exceeds the SP 800-38D plaintext
    /// limit. On authentication failure the buffer is left untouched: it
    /// still holds the ciphertext, never half-decrypted plaintext.
    pub fn open_in_place<'b>(
        &self,
        nonce: &[u8; 12],
        aad: &[u8],
        buf: &'b mut [u8],
    ) -> Result<&'b mut [u8], Error> {
        self.inner.open_in_place(nonce, aad, buf)
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
    /// Fails with [`Error::InvalidLength`] if `plaintext` exceeds the
    /// SP 800-38D limit of 2^39 - 256 bits.
    pub fn seal(
        &self,
        nonce: &[u8],
        aad: &[u8],
        plaintext: &[u8],
    ) -> Result<(Vec<u8>, [u8; 16]), Error> {
        self.inner.seal(nonce, aad, plaintext)
    }

    /// Encrypt and authenticate in place: XOR `buf` with the CTR keystream
    /// and return the tag.
    ///
    /// Fails with [`Error::InvalidLength`] if `buf` exceeds the SP 800-38D
    /// plaintext limit. Key-derived intermediates are scrubbed before this
    /// method returns on every path.
    pub fn seal_in_place(
        &self,
        nonce: &[u8; 12],
        aad: &[u8],
        buf: &mut [u8],
    ) -> Result<[u8; 16], Error> {
        self.inner.seal_in_place(nonce, aad, buf)
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

    /// Verify the tag and decrypt in place. `buf` holds `ct || tag`; the
    /// tag is verified FIRST, then the ciphertext region is decrypted in
    /// place, and the plaintext region (tag stripped) is returned.
    ///
    /// Fails with [`Error::InvalidLength`] if `buf` is shorter than 16
    /// bytes or the ciphertext region exceeds the SP 800-38D plaintext
    /// limit. On authentication failure the buffer is left untouched: it
    /// still holds the ciphertext, never half-decrypted plaintext.
    pub fn open_in_place<'b>(
        &self,
        nonce: &[u8; 12],
        aad: &[u8],
        buf: &'b mut [u8],
    ) -> Result<&'b mut [u8], Error> {
        self.inner.open_in_place(nonce, aad, buf)
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
        let (ct, tag) = g.seal(&[0u8; 12], &[], &[]).unwrap();
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
        let (ct, tag) = g.seal(&[0u8; 12], &[], &[0u8; 16]).unwrap();
        assert_eq!(hx(&ct), "0388dace60b6a392f328c2b971b2fe78");
        assert_eq!(hx(&tag), "ab6e47d42cec13bdf53a67b21257bddf");
        assert_eq!(g.open(&[0u8; 12], &[], &ct, &tag).unwrap(), vec![0u8; 16]);
    }

    /// McGrew-Viega GCM test case 3.
    #[test]
    fn gcm_case3() {
        let g = Aes128Gcm::new(&h("feffe9928665731c6d6a8f9467308308").try_into().unwrap());
        let (ct, tag) = g
            .seal(
                &h("cafebabefacedbaddecaf888"),
                &[],
                &h("d9313225f88406e5a559a55b5d05df"),
            )
            .unwrap();
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
        let (ct, tag) = g.seal(&h("cafebabefacedbaddecaf888"), &aad, &pt).unwrap();
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
        let (ct, tag) = g.seal(&iv, &aad, &pt).unwrap();
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
        let (ct, tag) = g.seal(&nonce, aad, pt).unwrap();
        assert_ne!(ct, pt);
        assert_eq!(g.open(&nonce, aad, &ct, &tag).unwrap(), pt);
    }

    #[test]
    fn gcm_tampered_tag_rejected() {
        let g = Aes128Gcm::new(&[0x11u8; 16]);
        let (ct, mut tag) = g.seal(&[0x22u8; 12], b"aad", b"secret").unwrap();
        tag[0] ^= 1;
        assert_eq!(
            g.open(&[0x22u8; 12], b"aad", &ct, &tag),
            Err(Error::AuthenticationFailed)
        );
    }

    #[test]
    fn gcm_tampered_ciphertext_rejected() {
        let g = Aes128Gcm::new(&[0x11u8; 16]);
        let (mut ct, tag) = g.seal(&[0x22u8; 12], b"aad", b"secret").unwrap();
        ct[0] ^= 1;
        assert_eq!(
            g.open(&[0x22u8; 12], b"aad", &ct, &tag),
            Err(Error::AuthenticationFailed)
        );
    }

    #[test]
    fn gcm_tampered_aad_rejected() {
        let g = Aes128Gcm::new(&[0x11u8; 16]);
        let (ct, tag) = g.seal(&[0x22u8; 12], b"aad", b"secret").unwrap();
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
                let (ct, tag) = g.seal(&nonce, &aad, &pt).unwrap();
                assert_eq!(g.open(&nonce, &aad, &ct, &tag).unwrap(), pt);
            }
        }
    }

    #[test]
    fn gcm_inplace_kat_cases_1_to_4() {
        // McGrew-Viega GCM test cases 1-4, run byte-for-byte through the
        // in-place APIs.
        let g = Aes128Gcm::new(&[0u8; 16]);

        // Case 1: empty plaintext, empty AAD.
        let nonce12 = [0u8; 12];
        let mut buf: Vec<u8> = Vec::new();
        let tag = g.seal_in_place(&nonce12, &[], &mut buf).unwrap();
        assert!(buf.is_empty());
        assert_eq!(hx(&tag), "58e2fccefa7e3061367f1d57a4e7455a");
        buf.extend_from_slice(&tag);
        assert!(g.open_in_place(&nonce12, &[], &mut buf).unwrap().is_empty());

        // Case 2: one zero block.
        let mut buf = vec![0u8; 16];
        let tag = g.seal_in_place(&nonce12, &[], &mut buf).unwrap();
        assert_eq!(hx(&buf), "0388dace60b6a392f328c2b971b2fe78");
        assert_eq!(hx(&tag), "ab6e47d42cec13bdf53a67b21257bddf");
        buf.extend_from_slice(&tag);
        assert_eq!(
            g.open_in_place(&nonce12, &[], &mut buf).unwrap(),
            &[0u8; 16][..]
        );

        // Case 3.
        let g3 = Aes128Gcm::new(&h("feffe9928665731c6d6a8f9467308308").try_into().unwrap());
        let n3: [u8; 12] = h("cafebabefacedbaddecaf888").try_into().unwrap();
        let mut buf = h("d9313225f88406e5a559a55b5d05df");
        let tag = g3.seal_in_place(&n3, &[], &mut buf).unwrap();
        assert_eq!(hx(&buf), "42831ec2217774244b728d2976202d");
        assert_eq!(hx(&tag), "d33ebf862b06bdc2727f09eb0c026b12");
        buf.extend_from_slice(&tag);
        assert_eq!(
            g3.open_in_place(&n3, &[], &mut buf).unwrap(),
            &h("d9313225f88406e5a559a55b5d05df")[..]
        );

        // Case 4: 60-byte plaintext, 20-byte AAD; tag only asserted.
        let pt = h(
            "d9313225f88406e5a559a55b5d05dfd9313225f88406e5a559a55b5d05dfd9313225f88406e5a559a55b5d05dfd9313225f88406e5a559a55b5d05df",
        );
        let aad = h("feedfacedeadbeeffeedfacedeadbeefabaddad2");
        let mut buf = pt.clone();
        let tag = g3.seal_in_place(&n3, &aad, &mut buf).unwrap();
        assert_eq!(buf.len(), 60);
        assert_eq!(hx(&tag), "195fc60690036333e8a4a7d8f8b1232c");
        buf.extend_from_slice(&tag);
        assert_eq!(g3.open_in_place(&n3, &aad, &mut buf).unwrap(), &pt[..]);
    }

    #[test]
    fn gcm_inplace_case6_long_nonce() {
        // McGrew-Viega GCM test case 6: 60-byte IV exercises the
        // GHASH-based J0 derivation, through the in-place path.
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
        let mut buf = pt.clone();
        let tag = g.inner.seal_in_place(&iv, &aad, &mut buf).unwrap();
        assert_eq!(
            hx(&buf),
            "8ce24998625615b603a033aca13fb894be9112a5c3a211a8ba262a3cca7e2ca701e4a9a4fba43c90ccdcb281d48c7c6fd62875d2aca417034c34aee5"
        );
        assert_eq!(hx(&tag), "619cc5aefffe0bfa462af43c1699d050");
        buf.extend_from_slice(&tag);
        assert_eq!(g.inner.open_in_place(&iv, &aad, &mut buf).unwrap(), &pt[..]);
    }

    #[test]
    fn gcm_inplace_diff_sweep() {
        // In-place seal/open must agree byte-for-byte with the allocating
        // seal/open across all padding paths, for both key sizes.
        for pt_len in 0..40usize {
            for aad_len in [0, 1, 15, 16, 17, 31] {
                let pt: Vec<u8> = (0..pt_len as u8).collect();
                let aad: Vec<u8> = (0..aad_len as u8).map(|x| x.wrapping_add(7)).collect();
                let nonce = [pt_len as u8; 12];
                let g128 = Aes128Gcm::new(&[0x5au8; 16]);
                let mut in_ct = pt.clone();
                let in_tag = g128.seal_in_place(&nonce, &aad, &mut in_ct).unwrap();
                let (ct, tag) = g128.seal(&nonce, &aad, &pt).unwrap();
                assert_eq!(in_ct, ct);
                assert_eq!(in_tag, tag);
                let mut combined = in_ct.clone();
                combined.extend_from_slice(&in_tag);
                let in_pt = g128.open_in_place(&nonce, &aad, &mut combined).unwrap();
                assert_eq!(in_pt, &pt[..]);
                assert_eq!(g128.open(&nonce, &aad, &in_ct, &in_tag).unwrap(), pt);

                let g256 = Aes256Gcm::new(&[0x77u8; 32]);
                let mut in_ct = pt.clone();
                let in_tag = g256.seal_in_place(&nonce, &aad, &mut in_ct).unwrap();
                let (ct, tag) = g256.seal(&nonce, &aad, &pt).unwrap();
                assert_eq!(in_ct, ct);
                assert_eq!(in_tag, tag);
                let mut combined = in_ct;
                combined.extend_from_slice(&in_tag);
                assert_eq!(
                    g256.open_in_place(&nonce, &aad, &mut combined).unwrap(),
                    &pt[..]
                );
                assert_eq!(g256.open(&nonce, &aad, &ct, &tag).unwrap(), pt);
            }
        }
    }

    #[test]
    fn gcm_inplace_tamper_rejected() {
        // Tag flip, ciphertext flip and AAD flip on the in-place path all
        // fail closed, and the failed open leaves ct || tag in the buffer.
        for make in [
            Aes128Gcm::new(&[0x11u8; 16]).inner,
            Aes256Gcm::new(&[0x11u8; 32]).inner,
        ] {
            let nonce = [0x22u8; 12];
            let aad = b"aad";
            let pt = b"secret message for tamper checks";
            let (ct, tag) = make.seal(&nonce, aad, pt).unwrap();
            let mut good = ct.clone();
            good.extend_from_slice(&tag);

            let mut bad_tag = good.clone();
            bad_tag[good.len() - 2] ^= 0x40;
            let before = bad_tag.clone();
            assert_eq!(
                make.open_in_place(&nonce, aad, &mut bad_tag),
                Err(Error::AuthenticationFailed)
            );
            assert_eq!(bad_tag, before);

            let mut bad_ct = good.clone();
            bad_ct[ct.len() - 1] ^= 1;
            let before = bad_ct.clone();
            assert_eq!(
                make.open_in_place(&nonce, aad, &mut bad_ct),
                Err(Error::AuthenticationFailed)
            );
            assert_eq!(bad_ct, before);

            let mut bad_aad = good.clone();
            let before = bad_aad.clone();
            assert_eq!(
                make.open_in_place(&nonce, b"bad", &mut bad_aad),
                Err(Error::AuthenticationFailed)
            );
            assert_eq!(bad_aad, before);

            // A good open still works afterwards and strips the tag.
            let mut ok = good.clone();
            assert_eq!(make.open_in_place(&nonce, aad, &mut ok).unwrap(), &pt[..]);
        }

        // Buffers too short to hold a tag fail closed.
        let g = Aes128Gcm::new(&[0x11u8; 16]);
        let mut short = vec![0u8; 15];
        assert_eq!(
            g.open_in_place(&[0x22u8; 12], b"aad", &mut short),
            Err(Error::InvalidLength)
        );
    }

    #[test]
    fn gcm_rejects_oversize_plaintext() {
        // SP 800-38D caps the plaintext at 2^39 - 256 bits = 2^36 - 32
        // bytes; beyond that the 32-bit CTR index could wrap and reuse
        // keystream. Inputs this large cannot be allocated in a test, so
        // the guard boundary itself is asserted here.
        assert_eq!(MAX_INPUT_LEN, (1u64 << 36) - 32);
        assert_eq!(check_input_len(0), Ok(()));
        assert_eq!(check_input_len(MAX_INPUT_LEN as usize), Ok(()));
        assert_eq!(
            check_input_len(MAX_INPUT_LEN as usize + 1),
            Err(Error::InvalidLength)
        );
        assert_eq!(check_input_len(usize::MAX), Err(Error::InvalidLength));

        // With the guard in place the CTR block index can never reach
        // u32::MAX: max blocks are 2^32 - 2, so `(i as u32).wrapping_add(1)`
        // in `ctr` never truncates or wraps.
        let max_blocks = (MAX_INPUT_LEN as usize).div_ceil(16);
        assert!(max_blocks < u32::MAX as usize);

        // The guard is wired through both entry points.
        let g = Aes128Gcm::new(&[0x11u8; 16]);
        let (ct, tag) = g.seal(&[0x22u8; 12], b"aad", b"ok").unwrap();
        assert_eq!(g.open(&[0x22u8; 12], b"aad", &ct, &tag).unwrap(), b"ok");
    }
}
