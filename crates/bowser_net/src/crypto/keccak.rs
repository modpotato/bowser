//! Keccak-f\[1600\] and the SHA-3 / SHAKE functions built on it.
//!
//! Implements FIPS 202: `SHA3-256`, `SHA3-512`, `SHAKE-128` (streaming, for
//! ML-KEM's `SampleNTT`) and `SHAKE-256` (one-shot, for `PRF` and `J`).
//! Standard library only; no external crates.

/// Keccak-f\[1600\] round constants (FIPS 202, section 3.2.5).
const ROUND_CONSTANTS: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808a,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808b,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008a,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000a,
    0x0000_0000_8000_808b,
    0x8000_0000_0000_008b,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800a,
    0x8000_0000_8000_000a,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

/// Keccak-f\[1600\] rotation offsets `r[x][y]` (FIPS 202, section 3.2.2).
const ROTATION_OFFSETS: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

/// The Keccak-f\[1600\] permutation on a 25-lane state.
fn keccak_f1600(state: &mut [u64; 25]) {
    for round_constant in ROUND_CONSTANTS {
        // Theta.
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                state[x + 5 * y] ^= d;
            }
        }
        // Rho and pi combined.
        let mut b = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                b[y + 5 * ((2 * x + 3 * y) % 5)] =
                    state[x + 5 * y].rotate_left(ROTATION_OFFSETS[x][y]);
            }
        }
        // Chi.
        for x in 0..5 {
            for y in 0..5 {
                state[x + 5 * y] =
                    b[x + 5 * y] ^ ((!b[(x + 1) % 5 + 5 * y]) & b[(x + 2) % 5 + 5 * y]);
            }
        }
        // Iota.
        state[0] ^= round_constant;
    }
}

/// XOR `data` into the state starting at byte `offset` (little-endian lanes).
fn xor_bytes(state: &mut [u64; 25], offset: usize, data: &[u8]) {
    for (i, byte) in data.iter().enumerate() {
        let lane = (offset + i) / 8;
        let shift = 8 * ((offset + i) % 8);
        state[lane] ^= (*byte as u64) << shift;
    }
}

/// Read bytes out of the state starting at byte `offset` (little-endian lanes).
fn read_bytes(state: &[u64; 25], offset: usize, out: &mut [u8]) {
    for (i, byte) in out.iter_mut().enumerate() {
        let lane = (offset + i) / 8;
        let shift = 8 * ((offset + i) % 8);
        *byte = (state[lane] >> shift) as u8;
    }
}

/// A byte-oriented sponge over Keccak-f\[1600\] with a compile-time rate.
struct Sponge<const RATE: usize> {
    state: [u64; 25],
    /// Bytes of the current rate block already absorbed or squeezed.
    pos: usize,
    finalized: bool,
}

impl<const RATE: usize> Sponge<RATE> {
    fn new() -> Self {
        Self {
            state: [0u64; 25],
            pos: 0,
            finalized: false,
        }
    }

    fn absorb(&mut self, mut data: &[u8]) {
        debug_assert!(!self.finalized);
        while !data.is_empty() {
            let n = (RATE - self.pos).min(data.len());
            xor_bytes(&mut self.state, self.pos, &data[..n]);
            self.pos += n;
            data = &data[n..];
            if self.pos == RATE {
                keccak_f1600(&mut self.state);
                self.pos = 0;
            }
        }
    }

    /// Apply the multi-rate padding with the given domain-separation byte.
    fn finalize(&mut self, domain: u8) {
        if self.finalized {
            return;
        }
        self.finalized = true;
        let mut pad = [0u8; RATE];
        pad[0] = domain;
        pad[RATE - 1 - self.pos] |= 0x80;
        let tail = RATE - self.pos;
        xor_bytes(&mut self.state, self.pos, &pad[..tail]);
        keccak_f1600(&mut self.state);
        self.pos = 0;
    }

    fn squeeze(&mut self, domain: u8, mut out: &mut [u8]) {
        self.finalize(domain);
        while !out.is_empty() {
            if self.pos == RATE {
                keccak_f1600(&mut self.state);
                self.pos = 0;
            }
            let n = (RATE - self.pos).min(out.len());
            read_bytes(&self.state, self.pos, &mut out[..n]);
            self.pos += n;
            out = &mut out[n..];
        }
    }
}

/// SHA3-256 (FIPS 202).
pub(crate) fn sha3_256(msg: &[u8]) -> [u8; 32] {
    let mut sponge = Sponge::<136>::new();
    sponge.absorb(msg);
    let mut out = [0u8; 32];
    sponge.squeeze(0x06, &mut out);
    out
}

/// SHA3-512 (FIPS 202).
pub(crate) fn sha3_512(msg: &[u8]) -> [u8; 64] {
    let mut sponge = Sponge::<72>::new();
    sponge.absorb(msg);
    let mut out = [0u8; 64];
    sponge.squeeze(0x06, &mut out);
    out
}

/// SHAKE-256 with arbitrary output length (FIPS 202).
pub(crate) fn shake256(msg: &[u8], out: &mut [u8]) {
    let mut sponge = Sponge::<136>::new();
    sponge.absorb(msg);
    sponge.squeeze(0x1f, out);
}

/// Streaming SHAKE-128 (rate 168) used by ML-KEM's `SampleNTT`.
pub(crate) struct Shake128 {
    sponge: Sponge<168>,
}

impl Shake128 {
    pub(crate) fn new() -> Self {
        Self {
            sponge: Sponge::new(),
        }
    }

    pub(crate) fn absorb(&mut self, data: &[u8]) {
        self.sponge.absorb(data);
    }

    pub(crate) fn squeeze(&mut self, out: &mut [u8]) {
        self.sponge.squeeze(0x1f, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn sha3_256_empty() {
        assert_eq!(
            hex(&sha3_256(b"")),
            "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
        );
    }

    #[test]
    fn sha3_256_abc() {
        assert_eq!(
            hex(&sha3_256(b"abc")),
            "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
        );
    }

    #[test]
    fn sha3_512_abc() {
        assert_eq!(
            hex(&sha3_512(b"abc")),
            concat!(
                "b751850b1a57168a5693cd924b6b096e08f621827444f70d884f5d0240d271",
                "2e10e116e9192af3c91a7ec57647e3934057340b4cf408d5a56592f8274eec",
                "53f0",
            )
        );
    }

    #[test]
    fn shake128_abc() {
        let mut xof = Shake128::new();
        xof.absorb(b"abc");
        let mut out = [0u8; 32];
        xof.squeeze(&mut out);
        assert_eq!(
            hex(&out),
            "5881092dd818bf5cf8a3ddb793fbcba74097d5c526a6d35f97b83351940f2cc8"
        );
    }

    #[test]
    fn shake256_abc() {
        let mut out = [0u8; 64];
        shake256(b"abc", &mut out);
        assert_eq!(
            hex(&out),
            concat!(
                "483366601360a8771c6863080cc4114d8db44530f8f1e1ee4f94ea37e78b57",
                "39d5a15bef186a5386c75744c0527e1faa9f8726e462a12a4feb06bd8801e7",
                "51e4",
            )
        );
    }

    #[test]
    fn shake128_incremental_squeeze_matches() {
        // Absorbing in two chunks and squeezing in small pieces must match
        // the one-shot output.
        let mut a = Shake128::new();
        a.absorb(b"abc");
        let mut out_a = [0u8; 200];
        for chunk in out_a.chunks_mut(3) {
            a.squeeze(chunk);
        }

        let mut b = Shake128::new();
        b.absorb(b"a");
        b.absorb(b"bc");
        let mut out_b = [0u8; 200];
        b.squeeze(&mut out_b);

        assert_eq!(out_a, out_b);
    }
}
