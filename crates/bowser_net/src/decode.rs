//! std-only `Content-Encoding` decoding: gzip and deflate (RFC 1951/1950/1952).
//!
//! A from-scratch DEFLATE inflater plus the gzip and zlib wrappers, so
//! `fetch` can advertise and decode `Accept-Encoding: gzip, deflate`
//! without any dependency. Two APIs:
//!
//! * One-shot ([`gunzip`], [`inflate_zlib`], [`inflate_raw`],
//!   [`decode_content_encoding`]) for fully-framed bodies.
//! * Streaming ([`Decoder`], [`DecodeChain`]): feed input chunks as they
//!   arrive; decoded bytes are emitted incrementally with bounded memory
//!   (a 32 KiB LZ77 window plus a small input staging buffer). The fetch
//!   layer uses the streaming API so a large compressed body never sits
//!   whole in memory in encoded form.
//!
//! Both APIs cap decompressed output at a caller-supplied limit
//! (zip-bomb protection).

use std::fmt;

/// Failure to decode a `Content-Encoding` body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// The input ended mid-symbol, mid-block, or mid-wrapper.
    Truncated,
    /// The input is not valid deflate/gzip/zlib (bad magic, bad header
    /// bits, over-subscribed Huffman codes, bad distance, CRC/Adler
    /// mismatch, ...).
    InvalidData,
    /// Decompressed output exceeded the caller's `max_output` cap.
    OutputTooLarge,
    /// A `Content-Encoding` token this client cannot decode.
    UnsupportedEncoding,
    /// More than [`MAX_CODING_DEPTH`] stacked `Content-Encoding` layers.
    CodingDepthExceeded,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("truncated compressed data"),
            Self::InvalidData => f.write_str("invalid compressed data"),
            Self::OutputTooLarge => f.write_str("decompressed output exceeds the size cap"),
            Self::UnsupportedEncoding => f.write_str("unsupported content-encoding"),
            Self::CodingDepthExceeded => f.write_str("too many stacked content codings"),
        }
    }
}

impl std::error::Error for DecodeError {}

// ---------------------------------------------------------------------------
// Bit reader (LSB-first, RFC 1951 §3.1)
// ---------------------------------------------------------------------------

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bitbuf: u64,
    bits: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bitbuf: 0,
            bits: 0,
        }
    }

    fn fill(&mut self) {
        while self.bits <= 56 && self.pos < self.data.len() {
            self.bitbuf |= (self.data[self.pos] as u64) << self.bits;
            self.bits += 8;
            self.pos += 1;
        }
    }

    fn read_bits(&mut self, n: u32) -> Result<u32, DecodeError> {
        debug_assert!(n <= 16);
        self.fill();
        if self.bits < n {
            return Err(DecodeError::Truncated);
        }
        let v = (self.bitbuf & ((1u64 << n) - 1)) as u32;
        self.bitbuf >>= n;
        self.bits -= n;
        Ok(v)
    }

    fn align_to_byte(&mut self) {
        let skip = self.bits % 8;
        self.bitbuf >>= skip;
        self.bits -= skip;
    }

    /// Copy `n` whole bytes (after [`align_to_byte`][Self::align_to_byte])
    /// into `out`, enforcing the output cap first.
    fn take_bytes(&mut self, n: usize, out: &mut Vec<u8>, max: usize) -> Result<(), DecodeError> {
        self.align_to_byte();
        if out.len().checked_add(n).is_none_or(|end| end > max) {
            return Err(DecodeError::OutputTooLarge);
        }
        let from_buf = ((self.bits / 8) as usize).min(n);
        for _ in 0..from_buf {
            out.push(self.bitbuf as u8);
            self.bitbuf >>= 8;
            self.bits -= 8;
        }
        let rest = n - from_buf;
        if self.data.len() - self.pos < rest {
            return Err(DecodeError::Truncated);
        }
        out.extend_from_slice(&self.data[self.pos..self.pos + rest]);
        self.pos += rest;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Canonical Huffman codes (the "puff" construction)
// ---------------------------------------------------------------------------

struct Huffman {
    counts: [u16; 16],
    /// Symbols sorted by (code length, symbol); `decode` indexes in.
    symbols: Vec<u16>,
    max_len: u32,
    /// Primary 9-bit lookup table, built once per code in
    /// [`build_huffman`]. `lut[peek].len == 0` means "no code of length
    /// `<= LUT_BITS` starts with these bits": fall back to the
    /// bit-at-a-time walk (codes longer than 9 bits, or input too short
    /// to fill the window).
    lut: [LutEntry; LUT_SIZE],
}

/// Primary lookup width for [`Huffman`] fast decode: 9 bits covers every
/// fixed-block code (7/8/9-bit literal/length, 5-bit distance) in a single
/// table lookup; longer dynamic codes take the slow path.
const LUT_BITS: u32 = 9;
const LUT_SIZE: usize = 1 << LUT_BITS; // 512
const LUT_MASK: u64 = (LUT_SIZE as u64) - 1;

/// One primary-table entry. `len == 0` marks an empty slot (slow path).
#[derive(Clone, Copy)]
struct LutEntry {
    sym: u16,
    len: u8,
}

/// Reverse the low `len` bits of `v`. DEFLATE packs Huffman code bits
/// MSB-first (RFC 1951 §3.1.1) while the bit buffer stages bits LSB-first,
/// so the on-wire order of a code is the bit-reverse of the integer code
/// the canonical walk accumulates.
fn reverse_bits(mut v: u32, mut len: u32) -> u32 {
    let mut r = 0u32;
    while len > 0 {
        r = (r << 1) | (v & 1);
        v >>= 1;
        len -= 1;
    }
    r
}

fn build_huffman(lengths: &[u8]) -> Result<Huffman, DecodeError> {
    let mut counts = [0u16; 16];
    for &l in lengths {
        if l >= 16 {
            return Err(DecodeError::InvalidData);
        }
        counts[l as usize] += 1;
    }
    counts[0] = 0;
    // Reject over-subscribed codes; incomplete ones are legal (a lone
    // distance code is a single 1-bit code with one unused slot).
    let mut left: i32 = 1;
    let mut max_len = 0u32;
    for (len, &count) in counts.iter().enumerate().skip(1) {
        left <<= 1;
        left -= count as i32;
        if left < 0 {
            return Err(DecodeError::InvalidData);
        }
        if count > 0 {
            max_len = len as u32;
        }
    }
    if max_len == 0 {
        return Err(DecodeError::InvalidData);
    }
    let mut offs = [0usize; 16];
    for len in 1..16 {
        offs[len] = offs[len - 1] + counts[len - 1] as usize;
    }
    let mut symbols = vec![0u16; lengths.len()];
    for (sym, &l) in lengths.iter().enumerate() {
        if l != 0 {
            symbols[offs[l as usize]] = sym as u16;
            offs[l as usize] += 1;
        }
    }
    // Primary lookup table. Canonical codes are assigned in the standard
    // way (`next_code[len]` is the first code of that length); a code of
    // length `len <= LUT_BITS` fills every slot whose low `len` bits are
    // the bit-reversed code, i.e. every 9-bit window starting with it.
    // Codes longer than `LUT_BITS` leave their slots empty, and so does
    // any prefix of an incomplete code: both cases resolve identically
    // through the slow bit-walk, so output is unchanged.
    let mut lut = [LutEntry { sym: 0, len: 0 }; LUT_SIZE];
    let mut code = 0u32;
    let mut next_code = [0u32; 16];
    for len in 1..16 {
        code = (code + counts[len - 1] as u32) << 1;
        next_code[len] = code;
    }
    for (sym, &l) in lengths.iter().enumerate() {
        let len = l as usize;
        if len == 0 || len > LUT_BITS as usize {
            continue;
        }
        let c = next_code[len];
        next_code[len] += 1;
        let base = reverse_bits(c, len as u32) as usize;
        let reps = 1usize << (LUT_BITS as usize - len);
        for r in 0..reps {
            lut[base | (r << len)] = LutEntry {
                sym: sym as u16,
                len: len as u8,
            };
        }
    }
    Ok(Huffman {
        counts,
        symbols,
        max_len,
        lut,
    })
}

impl Huffman {
    fn decode(&self, br: &mut BitReader<'_>) -> Result<u16, DecodeError> {
        // Fast path: one 9-bit table lookup against the buffered u64
        // bit accumulator. Short input falls through to the slow walk,
        // which reports `Truncated` exactly as the pre-fast-path code
        // did; a table miss (code longer than 9 bits, or an unused slot
        // of an incomplete code) does the same.
        br.fill();
        if br.bits >= LUT_BITS {
            let e = self.lut[(br.bitbuf & LUT_MASK) as usize];
            if e.len != 0 {
                let len = e.len as u32;
                br.bitbuf >>= len;
                br.bits -= len;
                return Ok(e.sym);
            }
        }
        self.decode_slow(br)
    }

    /// Bit-at-a-time canonical walk (the original decoder): slow path
    /// for codes longer than [`LUT_BITS`] and for input too short to
    /// fill the lookup window.
    fn decode_slow(&self, br: &mut BitReader<'_>) -> Result<u16, DecodeError> {
        let mut code = 0u32;
        let mut first = 0u32;
        let mut index = 0u32;
        for len in 1..=self.max_len {
            code |= br.read_bits(1)?;
            let count = self.counts[len as usize] as u32;
            if code < first + count {
                return Ok(self.symbols[(index + code - first) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(DecodeError::InvalidData)
    }
}

// ---------------------------------------------------------------------------
// DEFLATE (RFC 1951)
// ---------------------------------------------------------------------------

/// (base length, extra bits) for literal/length symbols 257..=285.
fn length_base_extra(sym: u16) -> (usize, u32) {
    match sym {
        257..=264 => (3 + (sym - 257) as usize, 0),
        265..=268 => (11 + ((sym - 265) << 1) as usize, 1),
        269..=272 => (19 + ((sym - 269) << 2) as usize, 2),
        273..=276 => (35 + ((sym - 273) << 3) as usize, 3),
        277..=280 => (67 + ((sym - 277) << 4) as usize, 4),
        281..=284 => (131 + ((sym - 281) << 5) as usize, 5),
        _ => (258, 0), // 285
    }
}

/// (base distance, extra bits) for distance symbols 0..=29.
fn dist_base_extra(sym: u16) -> (usize, u32) {
    match sym {
        0..=3 => (1 + sym as usize, 0),
        _ => {
            let extra = (sym - 2) / 2;
            let base = (1usize << (extra + 1)) + 1 + (((sym & 1) as usize) << extra);
            (base, extra as u32)
        }
    }
}

fn fixed_litlen() -> Huffman {
    let mut lengths = [0u8; 288];
    lengths[0..144].fill(8);
    lengths[144..256].fill(9);
    lengths[256..280].fill(7);
    lengths[280..288].fill(8);
    // Valid by construction.
    build_huffman(&lengths).expect("fixed literal/length table is valid")
}

fn fixed_dist() -> Huffman {
    // Valid by construction.
    build_huffman(&[5u8; 32]).expect("fixed distance table is valid")
}

fn push_byte(out: &mut Vec<u8>, b: u8, max: usize) -> Result<(), DecodeError> {
    if out.len() >= max {
        return Err(DecodeError::OutputTooLarge);
    }
    out.push(b);
    Ok(())
}

/// Copy `len` bytes from `dist` bytes back in the 32 KiB window.
fn push_match(out: &mut Vec<u8>, dist: usize, len: usize, max: usize) -> Result<(), DecodeError> {
    if dist == 0 || dist > out.len() {
        return Err(DecodeError::InvalidData);
    }
    if out.len().checked_add(len).is_none_or(|end| end > max) {
        return Err(DecodeError::OutputTooLarge);
    }
    out.reserve(len);
    let mut src = out.len() - dist;
    let mut remaining = len;
    while remaining > 0 {
        // `chunk` bytes at `src` are already decoded, so the source range
        // is fully within the current output; the copy may overlap the
        // destination (distance < length), which `ptr::copy` permits.
        let chunk = remaining.min(out.len() - src);
        let dst = out.len();
        // SAFETY: `dst..dst + chunk` is inside the reserved capacity, and
        // `src..src + chunk` is inside the initialized prefix; `ptr::copy`
        // handles overlapping ranges.
        unsafe {
            let p = out.as_mut_ptr();
            std::ptr::copy(p.add(src), p.add(dst), chunk);
            out.set_len(dst + chunk);
        }
        src += chunk;
        remaining -= chunk;
    }
    Ok(())
}

fn decode_huffman_block(
    br: &mut BitReader<'_>,
    out: &mut Vec<u8>,
    max: usize,
    litlen: &Huffman,
    dist: &Huffman,
) -> Result<(), DecodeError> {
    loop {
        let sym = litlen.decode(br)?;
        if sym < 256 {
            push_byte(out, sym as u8, max)?;
        } else if sym == 256 {
            return Ok(());
        } else if sym <= 285 {
            let (base, extra) = length_base_extra(sym);
            let len = base + br.read_bits(extra)? as usize;
            let dsym = dist.decode(br)?;
            if dsym > 29 {
                return Err(DecodeError::InvalidData);
            }
            let (dbase, dextra) = dist_base_extra(dsym);
            let d = dbase + br.read_bits(dextra)? as usize;
            push_match(out, d, len, max)?;
        } else {
            return Err(DecodeError::InvalidData);
        }
    }
}

/// Order in which code-length code lengths are stored (RFC 1951 §3.2.7).
const CL_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

fn decode_dynamic_block(
    br: &mut BitReader<'_>,
    out: &mut Vec<u8>,
    max: usize,
) -> Result<(), DecodeError> {
    let hlit = br.read_bits(5)? as usize + 257;
    let hdist = br.read_bits(5)? as usize + 1;
    let hclen = br.read_bits(4)? as usize + 4;
    if hlit > 288 || hdist > 32 {
        return Err(DecodeError::InvalidData);
    }
    let mut cl_lengths = [0u8; 19];
    for i in 0..hclen {
        cl_lengths[CL_ORDER[i]] = br.read_bits(3)? as u8;
    }
    let cl = build_huffman(&cl_lengths)?;
    let total = hlit + hdist;
    let mut lengths = vec![0u8; total];
    let mut i = 0;
    while i < total {
        let sym = cl.decode(br)?;
        match sym {
            0..=15 => {
                lengths[i] = sym as u8;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(DecodeError::InvalidData);
                }
                let rep = br.read_bits(2)? as usize + 3;
                if i + rep > total {
                    return Err(DecodeError::InvalidData);
                }
                let v = lengths[i - 1];
                lengths[i..i + rep].fill(v);
                i += rep;
            }
            17 => {
                let rep = br.read_bits(3)? as usize + 3;
                if i + rep > total {
                    return Err(DecodeError::InvalidData);
                }
                i += rep; // zeros already in place
            }
            18 => {
                let rep = br.read_bits(7)? as usize + 11;
                if i + rep > total {
                    return Err(DecodeError::InvalidData);
                }
                i += rep;
            }
            _ => return Err(DecodeError::InvalidData),
        }
    }
    let litlen = build_huffman(&lengths[..hlit])?;
    let dist = build_huffman(&lengths[hlit..])?;
    decode_huffman_block(br, out, max, &litlen, &dist)
}

/// Inflate a raw DEFLATE stream, returning the decoded bytes and the
/// number of input bytes the stream consumed. Trailing bytes (a wrapper
/// trailer, the next concatenated member) are left for the caller:
/// after the final block the stream is byte-aligned and any whole bytes
/// still staged in the bit buffer belong to what follows.
fn inflate_raw_span(data: &[u8], max_output: usize) -> Result<(Vec<u8>, usize), DecodeError> {
    let mut out = Vec::new();
    let used = inflate_raw_span_into(data, &mut out, max_output)?;
    Ok((out, used))
}

/// Inflate a raw DEFLATE stream, appending the decoded bytes to `out`
/// and returning the number of input bytes consumed. `max_output` caps
/// the *total* `out` length, so callers decoding concatenated members
/// into one buffer pass the same cap for every member and the output
/// never needs a per-member temporary: peak allocation stays at the
/// caller's buffer instead of transiently doubling it.
fn inflate_raw_span_into(
    data: &[u8],
    out: &mut Vec<u8>,
    max_output: usize,
) -> Result<usize, DecodeError> {
    let mut br = BitReader::new(data);
    loop {
        let bfinal = br.read_bits(1)?;
        let btype = br.read_bits(2)?;
        match btype {
            0 => {
                // Stored block: skip to a byte boundary, then LEN/NLEN.
                br.align_to_byte();
                let len = br.read_bits(16)? as usize;
                let nlen = br.read_bits(16)?;
                if (len as u32) ^ 0xffff != nlen {
                    return Err(DecodeError::InvalidData);
                }
                br.take_bytes(len, out, max_output)?;
            }
            1 => decode_huffman_block(&mut br, out, max_output, &fixed_litlen(), &fixed_dist())?,
            2 => decode_dynamic_block(&mut br, out, max_output)?,
            _ => return Err(DecodeError::InvalidData),
        }
        if bfinal == 1 {
            br.align_to_byte();
            let used = br.pos - (br.bits / 8) as usize;
            return Ok(used);
        }
    }
}

/// Inflate a raw DEFLATE stream, capping decompressed output at
/// `max_output` bytes.
pub fn inflate_raw(data: &[u8], max_output: usize) -> Result<Vec<u8>, DecodeError> {
    Ok(inflate_raw_span(data, max_output)?.0)
}

// ---------------------------------------------------------------------------
// Checksums
// ---------------------------------------------------------------------------

/// Slicing-by-4 tables for [`crc32`]: 4 tables of 256 `u32`s.
///
/// `tables[0]` is the standard byte table. `tables[j][i]` for `j > 0`
/// is the CRC state after feeding byte `i` followed by `j` zero bytes
/// (i.e. one more [`crc32`] byte-step with `b = 0` applied to
/// `tables[j - 1][i]`), which lets the 4-byte update below fold four
/// byte-steps into four table lookups (zlib's `BYFOUR`).
const fn crc32_tables() -> [[u32; 256]; 4] {
    let mut t = [[0u32; 256]; 4];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 == 1 {
                0xedb88320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    // Derived tables need the complete `t[0]`: a second pass, not folded
    // into the loop above (which would read not-yet-computed entries).
    let mut j = 1;
    while j < 4 {
        let mut i = 0;
        while i < 256 {
            let prev = t[j - 1][i];
            t[j][i] = t[0][(prev & 0xff) as usize] ^ (prev >> 8);
            i += 1;
        }
        j += 1;
    }
    t
}

static CRC32_TABLES: [[u32; 256]; 4] = crc32_tables();

/// Fold `data` into a running CRC32 state: the slicing-by-4 core of
/// [`crc32`], exposed so the streaming decoder can checksum whole
/// decoded chunks without reimplementing the loop. `crc` is the
/// `0xFFFF_FFFF`-initialized, not-yet-final-XOR form.
fn crc32_update(mut c: u32, data: &[u8]) -> u32 {
    let t = &CRC32_TABLES;
    let (chunks, tail) = data.as_chunks::<4>();
    for ch in chunks {
        // Fold four bytes into the running CRC, then resolve four
        // byte-steps with four table lookups.
        c ^= u32::from_le_bytes(*ch);
        c = t[3][(c & 0xff) as usize]
            ^ t[2][((c >> 8) & 0xff) as usize]
            ^ t[1][((c >> 16) & 0xff) as usize]
            ^ t[0][(c >> 24) as usize];
    }
    for &b in tail {
        c = t[0][((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    c
}

fn crc32(data: &[u8]) -> u32 {
    crc32_update(0xffff_ffffu32, data) ^ 0xffff_ffff
}

fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let mut a = 1u32;
    let mut b = 0u32;
    for chunk in data.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= MOD;
        b %= MOD;
    }
    (b << 16) | a
}

// ---------------------------------------------------------------------------
// gzip (RFC 1952) and zlib (RFC 1950) wrappers
// ---------------------------------------------------------------------------

fn le_u32(data: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
}

/// Decompress a single gzip member, appending its payload to `out` and
/// returning the member's total byte length (header + deflate body +
/// trailer). The FHCRC header CRC16 is verified when the flag is set,
/// and the CRC32 / ISIZE trailer always is, over exactly the bytes this
/// member appended (`out[start..]` at entry).
fn gunzip_member_into(
    data: &[u8],
    out: &mut Vec<u8>,
    max_output: usize,
) -> Result<usize, DecodeError> {
    if data.len() < 10 {
        return Err(DecodeError::Truncated);
    }
    if data[0] != 0x1f || data[1] != 0x8b {
        return Err(DecodeError::InvalidData);
    }
    if data[2] != 8 {
        // Only the deflate compression method is defined.
        return Err(DecodeError::InvalidData);
    }
    let flg = data[3];
    if flg & 0xe0 != 0 {
        return Err(DecodeError::InvalidData);
    }
    let mut pos = 10; // magic(2) + method + flags + mtime(4) + xfl + os
    macro_rules! need {
        ($n:expr) => {
            if data.len() < pos + $n {
                return Err(DecodeError::Truncated);
            }
        };
    }
    if flg & 0x04 != 0 {
        need!(2);
        let xlen = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        need!(xlen);
        pos += xlen;
    }
    if flg & 0x08 != 0 {
        // FNAME: NUL-terminated original file name.
        loop {
            need!(1);
            let b = data[pos];
            pos += 1;
            if b == 0 {
                break;
            }
        }
    }
    if flg & 0x10 != 0 {
        // FCOMMENT: NUL-terminated comment.
        loop {
            need!(1);
            let b = data[pos];
            pos += 1;
            if b == 0 {
                break;
            }
        }
    }
    if flg & 0x02 != 0 {
        // FHCRC: the low 16 bits of the CRC-32 over every header byte
        // up to (not including) this field (RFC 1952 §2.3.1).
        need!(2);
        let stored = u16::from_le_bytes([data[pos], data[pos + 1]]);
        if crc32(&data[..pos]) as u16 != stored {
            return Err(DecodeError::InvalidData);
        }
        pos += 2;
    }
    let start = out.len();
    let used = inflate_raw_span_into(&data[pos..], out, max_output)?;
    pos += used;
    need!(8);
    let payload = &out[start..];
    if crc32(payload) != le_u32(data, pos) || payload.len() as u32 != le_u32(data, pos + 4) {
        return Err(DecodeError::InvalidData);
    }
    Ok(pos + 8)
}

/// Decompress a gzip stream: one or more concatenated members
/// (RFC 1952 §2.2 — a gzip file is a series of members, each with its
/// own header and trailer), verifying every member's trailer and
/// concatenating the payloads.
///
/// Each member decodes directly into the shared output buffer, capped at
/// `max_output` total: there is no per-member temporary, so a hostile
/// stream of many members cannot transiently double the allocation the
/// way a decode-then-copy loop would.
pub fn gunzip(data: &[u8], max_output: usize) -> Result<Vec<u8>, DecodeError> {
    if data.is_empty() {
        return Err(DecodeError::Truncated);
    }
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        // A member is at least 18 bytes (10 header + 8 trailer), so the
        // loop always makes progress.
        let used = gunzip_member_into(&data[pos..], &mut out, max_output)?;
        debug_assert!(used > 0);
        pos += used;
    }
    Ok(out)
}

/// Decompress a zlib stream, verifying the Adler-32 trailer.
pub fn inflate_zlib(data: &[u8], max_output: usize) -> Result<Vec<u8>, DecodeError> {
    if data.len() < 6 {
        // 2-byte header + at least one deflate byte + 4-byte trailer.
        return Err(DecodeError::Truncated);
    }
    let cmf = data[0];
    let flg = data[1];
    if !u16::from_be_bytes([cmf, flg]).is_multiple_of(31) {
        return Err(DecodeError::InvalidData);
    }
    if cmf & 0x0f != 8 {
        return Err(DecodeError::InvalidData);
    }
    if cmf >> 4 > 7 {
        return Err(DecodeError::InvalidData);
    }
    if flg & 0x20 != 0 {
        // Preset dictionaries are not supported.
        return Err(DecodeError::InvalidData);
    }
    let out = inflate_raw(&data[2..data.len() - 4], max_output)?;
    let adler = u32::from_be_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);
    if adler32(&out) != adler {
        return Err(DecodeError::InvalidData);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Content-Encoding dispatch
// ---------------------------------------------------------------------------

/// Maximum stacked `Content-Encoding` layers decoded. Decoding is
/// per-byte work at every layer, so an unbounded stack (an 8 KiB header
/// line holds ~1300 `gzip` tokens) is a CPU-amplification vector.
/// `identity`/empty tokens are not layers and do not count.
const MAX_CODING_DEPTH: usize = 4;

/// Decode a response body per its `Content-Encoding` header value.
///
/// Tokens are listed in the order the encodings were applied, so they are
/// decoded in reverse (e.g. `gzip, deflate` means gzip was applied last).
/// `identity` (and empty tokens) are no-ops. Decompressed output across
/// all layers is capped at `max_output` bytes. At most
/// [`MAX_CODING_DEPTH`] layers are decoded: each layer reprocesses every
/// byte, so deeper stacks are a CPU-amplification vector.
pub fn decode_content_encoding(
    body: &[u8],
    value: &str,
    max_output: usize,
) -> Result<Vec<u8>, DecodeError> {
    let mut owned: Option<Vec<u8>> = None;
    let mut depth = 0;
    for token in value.rsplit(',') {
        let token = token.split(';').next().unwrap_or("").trim();
        if token.is_empty() || token.eq_ignore_ascii_case("identity") {
            continue;
        }
        depth += 1;
        if depth > MAX_CODING_DEPTH {
            return Err(DecodeError::CodingDepthExceeded);
        }
        let input: &[u8] = owned.as_deref().unwrap_or(body);
        if token.eq_ignore_ascii_case("gzip") || token.eq_ignore_ascii_case("x-gzip") {
            owned = Some(gunzip(input, max_output)?);
        } else if token.eq_ignore_ascii_case("deflate") {
            // RFC 9110 says `deflate` is the zlib wrapper, but a large
            // share of servers send raw deflate instead; try zlib first,
            // then fall back to raw. An output-cap breach is not retried:
            // raw would blow past it too.
            owned = Some(match inflate_zlib(input, max_output) {
                Ok(v) => v,
                Err(DecodeError::OutputTooLarge) => return Err(DecodeError::OutputTooLarge),
                Err(_) => inflate_raw(input, max_output)?,
            });
        } else {
            return Err(DecodeError::UnsupportedEncoding);
        }
    }
    Ok(owned.unwrap_or_else(|| body.to_vec()))
}

// ---------------------------------------------------------------------------
// Streaming decoder
// ---------------------------------------------------------------------------

/// Which wrapper the streaming [`Decoder`] expects around the deflate
/// stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coding {
    /// RFC 1952 gzip member: header, deflate body, CRC32 + ISIZE trailer.
    Gzip,
    /// RFC 1950 zlib stream: 2-byte header, deflate body, Adler-32 trailer.
    Zlib,
    /// Raw RFC 1951 deflate stream: no wrapper, no trailer.
    Raw,
    /// `Content-Encoding: deflate`: sniff the first two bytes — a valid
    /// zlib header selects [`Coding::Zlib`], anything else [`Coding::Raw`]
    /// (many servers send raw deflate mislabeled as `deflate`).
    Auto,
}

/// Staged input for the streaming decoder: bytes not yet consumed into
/// the bit buffer. A read that cannot be satisfied consumes nothing, so
/// suspending mid-symbol never loses input.
struct BitState {
    bitbuf: u64,
    bits: u32,
    pending: Vec<u8>,
    ppos: usize,
}

impl BitState {
    fn new() -> Self {
        Self {
            bitbuf: 0,
            bits: 0,
            pending: Vec::new(),
            ppos: 0,
        }
    }

    fn push(&mut self, input: &[u8]) {
        self.pending.extend_from_slice(input);
    }

    fn fill(&mut self) {
        while self.bits <= 56 && self.ppos < self.pending.len() {
            self.bitbuf |= (self.pending[self.ppos] as u64) << self.bits;
            self.bits += 8;
            self.ppos += 1;
        }
    }

    /// Read `n` bits, or `Ok(None)` when the input is exhausted. A short
    /// read consumes nothing.
    fn read_bits(&mut self, n: u32) -> Result<Option<u32>, DecodeError> {
        debug_assert!(n <= 16);
        self.fill();
        if self.bits < n {
            return Ok(None);
        }
        let v = (self.bitbuf & ((1u64 << n) - 1)) as u32;
        self.bitbuf >>= n;
        self.bits -= n;
        Ok(Some(v))
    }

    fn align(&mut self) {
        let skip = self.bits % 8;
        self.bitbuf >>= skip;
        self.bits -= skip;
    }

    /// Read one whole byte; the caller must have [`align`](Self::align)ed
    /// first. Bytes staged in the bit buffer come before pending input.
    fn read_byte_aligned(&mut self) -> Result<Option<u8>, DecodeError> {
        debug_assert!(self.bits.is_multiple_of(8));
        if self.bits >= 8 {
            let b = self.bitbuf as u8;
            self.bitbuf >>= 8;
            self.bits -= 8;
            Ok(Some(b))
        } else if self.ppos < self.pending.len() {
            let b = self.pending[self.ppos];
            self.ppos += 1;
            Ok(Some(b))
        } else {
            Ok(None)
        }
    }

    /// Drop consumed input so the staging buffer stays small.
    fn compact(&mut self) {
        if self.ppos > 0 && (self.ppos >= 4096 || self.ppos == self.pending.len()) {
            self.pending.drain(..self.ppos);
            self.ppos = 0;
        }
    }
}

/// In-progress Huffman decode, carried across `Ok(None)` suspensions.
#[derive(Clone, Copy)]
struct HuffProg {
    code: u32,
    first: u32,
    index: u32,
    /// Next code length to try (1-based).
    len: u32,
}

impl HuffProg {
    fn fresh() -> Self {
        Self {
            code: 0,
            first: 0,
            index: 0,
            len: 1,
        }
    }
}

impl Huffman {
    /// Resumable [`decode`](Self::decode): `prog` holds the in-progress
    /// state, so a `Ok(None)` (need more input) can be retried later.
    /// `prog` is reset on a completed symbol.
    fn decode_resume(
        &self,
        prog: &mut HuffProg,
        bit: &mut BitState,
    ) -> Result<Option<u16>, DecodeError> {
        // Fast path: only valid at a symbol boundary (`len == 1`) with a
        // full lookup window buffered. It consumes only already-staged
        // bits, so it can neither suspend nor disturb `prog`; on a table
        // miss the slow walk below resumes exactly as before.
        if prog.len == 1 {
            bit.fill();
            if bit.bits >= LUT_BITS {
                let e = self.lut[(bit.bitbuf & LUT_MASK) as usize];
                if e.len != 0 {
                    let len = e.len as u32;
                    bit.bitbuf >>= len;
                    bit.bits -= len;
                    *prog = HuffProg::fresh();
                    return Ok(Some(e.sym));
                }
            }
        }
        let mut code = prog.code;
        let mut first = prog.first;
        let mut index = prog.index;
        let mut len = prog.len;
        while len <= self.max_len {
            let b = match bit.read_bits(1)? {
                None => {
                    prog.code = code;
                    prog.first = first;
                    prog.index = index;
                    prog.len = len;
                    return Ok(None);
                }
                Some(b) => b,
            };
            code |= b;
            let count = self.counts[len as usize] as u32;
            if code < first + count {
                *prog = HuffProg::fresh();
                return Ok(Some(self.symbols[(index + code - first) as usize]));
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
            len += 1;
        }
        Err(DecodeError::InvalidData)
    }
}

/// Decoded-output state: running checksums, the LZ77 window, and the
/// output cap.
struct OutState {
    /// Last decoded bytes (at least a full 32 KiB window) for
    /// back-references. Drained down as output is emitted, so memory
    /// stays bounded no matter how large the stream is.
    history: Vec<u8>,
    /// Total bytes decoded across all members (output cap accounting).
    total_out: u64,
    /// Bytes decoded for the current member (the ISIZE trailer is
    /// per-member; reset by [`Decoder::reset_member`]).
    member_out: u64,
    max_out: usize,
    crc: u32,
    adler_a: u32,
    adler_b: u32,
    adler_n: u32,
}

impl OutState {
    fn new(max_out: usize) -> Self {
        Self {
            history: Vec::new(),
            total_out: 0,
            member_out: 0,
            max_out,
            crc: 0xffff_ffff,
            adler_a: 1,
            adler_b: 0,
            adler_n: 0,
        }
    }

    fn emit_chunk(&mut self, chunk: &[u8], out: &mut Vec<u8>) -> Result<(), DecodeError> {
        if self.total_out + chunk.len() as u64 > self.max_out as u64 {
            return Err(DecodeError::OutputTooLarge);
        }
        self.crc = crc32_update(self.crc, chunk);
        for &b in chunk {
            // Adler-32 deferred-modulo: 5552 is the largest run that
            // cannot overflow u32 (same bound zlib uses).
            self.adler_a += b as u32;
            self.adler_b += self.adler_a;
            self.adler_n += 1;
            if self.adler_n == 5552 {
                self.adler_a %= 65521;
                self.adler_b %= 65521;
                self.adler_n = 0;
            }
        }
        self.total_out += chunk.len() as u64;
        self.member_out += chunk.len() as u64;
        self.history.extend_from_slice(chunk);
        out.extend_from_slice(chunk);
        Ok(())
    }

    fn emit_byte(&mut self, b: u8, out: &mut Vec<u8>) -> Result<(), DecodeError> {
        self.emit_chunk(&[b], out)
    }

    /// Copy `len` bytes from `dist` bytes back in the window. `len` is at
    /// most 258 (the longest RFC 1951 match), so a stack buffer holds one
    /// overlap-safe chunk. The distance is checked against the retained
    /// window, not the total output: concatenated gzip members reset the
    /// window, and a back-reference must never reach into an earlier
    /// member (each member is an independent deflate stream).
    fn emit_match(
        &mut self,
        dist: usize,
        len: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), DecodeError> {
        if dist == 0 || dist > self.history.len() {
            return Err(DecodeError::InvalidData);
        }
        let mut tmp = [0u8; 258];
        let mut src = self.history.len() - dist;
        let mut remaining = len;
        while remaining > 0 {
            let chunk = remaining.min(self.history.len() - src);
            debug_assert!(chunk <= tmp.len());
            tmp[..chunk].copy_from_slice(&self.history[src..src + chunk]);
            self.emit_chunk(&tmp[..chunk], out)?;
            src += chunk;
            remaining -= chunk;
        }
        Ok(())
    }

    /// Keep at least a full 32 KiB window for back-references; drop the
    /// rest. Called between [`Decoder::feed`] calls, never mid-match, so
    /// no in-progress copy can observe the drain.
    fn drain_history(&mut self) {
        if self.history.len() > 65536 {
            let drop = self.history.len() - 32768;
            self.history.drain(..drop);
        }
    }
}

/// Where the decoder is in the gzip optional header fields.
#[derive(Clone, Copy, Debug)]
enum GzNext {
    Name,
    Comment,
    Hcrc,
    Blocks,
}

/// Fixed 10-byte gzip header parse state.
struct GzHead {
    pos: u8,
    flags: u8,
    xlen: u16,
}

/// Dynamic-block header parse state (RFC 1951 §3.2.7).
struct DynState {
    stage: DynStage,
    hlit: usize,
    hdist: usize,
    hclen: usize,
    cl_lens: [u8; 19],
    cl_read: usize,
    cl: Option<Huffman>,
    clp: HuffProg,
    lens: Vec<u8>,
    /// Pending repeat code (16/17/18) whose extra bits are not yet read.
    rep: u8,
}

#[derive(Clone, Copy, Debug)]
enum DynStage {
    Hlit,
    Hdist,
    Hclen,
    ClLens,
    Lengths,
}

impl DynState {
    fn new() -> Self {
        Self {
            stage: DynStage::Hlit,
            hlit: 0,
            hdist: 0,
            hclen: 0,
            cl_lens: [0; 19],
            cl_read: 0,
            cl: None,
            clp: HuffProg::fresh(),
            lens: Vec::new(),
            rep: 0,
        }
    }
}

/// Suspended symbol-decode state within a Huffman-coded block.
#[derive(Clone, Copy, Debug)]
enum SymKind {
    LitLen,
    LenExtra { extra: u32 },
    Dist,
    DistExtra { extra: u32 },
}

#[derive(Clone, Copy, Debug)]
enum Phase {
    /// [`Coding::Auto`]: peek at the first two bytes.
    Sniff,
    GzHead,
    GzXlen,
    GzSkip(usize, GzNext),
    GzSkipNul(GzNext),
    GzHcrc,
    ZlHead,
    BlockHead,
    StoredLen,
    StoredCopy,
    DynHead,
    Symbols,
    Trailer,
    Done,
}

/// Incremental gzip/zlib/deflate decoder.
///
/// Feed compressed input with [`feed`](Self::feed) as it arrives;
/// decoded bytes are appended to the caller's `out` buffer. When the
/// input ends, call [`finish`](Self::finish) to verify the wrapper
/// trailer (CRC32/Adler-32). Total decoded output is capped at
/// `max_output` (zip-bomb protection). Memory stays bounded: besides the
/// caller's output, only a 32 KiB LZ77 window and a small input staging
/// buffer are retained.
///
/// After an error the decoder is poisoned and must not be fed further.
pub struct Decoder {
    coding: Coding,
    phase: Phase,
    bit: BitState,
    out: OutState,
    // Deflate block state.
    bfinal: bool,
    litlen: Option<Huffman>,
    dist: Option<Huffman>,
    sym: SymKind,
    hprog: HuffProg,
    sym_len: usize,
    sym_dist: usize,
    dyns: DynState,
    stored_left: usize,
    sl_bytes: [u8; 4],
    sl_have: u8,
    // Wrapper state.
    gz: GzHead,
    gx_have: u8,
    zl_have: u8,
    zl_cmf: u8,
    trailer: [u8; 8],
    trailer_have: u8,
    /// Running CRC-32 over the gzip header bytes (for FHCRC), and the
    /// staged FHCRC field bytes.
    gz_hcrc: u32,
    hcrc: [u8; 2],
    hcrc_have: u8,
    finished: bool,
}

impl Decoder {
    /// New decoder for `coding`, capping decoded output at `max_output`
    /// bytes.
    pub fn new(coding: Coding, max_output: usize) -> Self {
        let phase = match coding {
            Coding::Gzip => Phase::GzHead,
            Coding::Zlib => Phase::ZlHead,
            Coding::Raw => Phase::BlockHead,
            Coding::Auto => Phase::Sniff,
        };
        Self {
            coding,
            phase,
            bit: BitState::new(),
            out: OutState::new(max_output),
            bfinal: false,
            litlen: None,
            dist: None,
            sym: SymKind::LitLen,
            hprog: HuffProg::fresh(),
            sym_len: 0,
            sym_dist: 0,
            dyns: DynState::new(),
            stored_left: 0,
            sl_bytes: [0; 4],
            sl_have: 0,
            gz: GzHead {
                pos: 0,
                flags: 0,
                xlen: 0,
            },
            gx_have: 0,
            zl_have: 0,
            zl_cmf: 0,
            trailer: [0; 8],
            trailer_have: 0,
            gz_hcrc: 0xffff_ffff,
            hcrc: [0; 2],
            hcrc_have: 0,
            finished: false,
        }
    }

    /// Feed the next input chunk; decoded bytes are appended to `out`.
    /// Returns `Ok(())` both when input was consumed and when more input
    /// is needed to make progress. Feeding empty input after the stream
    /// completed is a no-op (layers of a [`DecodeChain`] finish at
    /// different times). Non-empty input after completion starts the
    /// next concatenated gzip member; for other codings it is an error.
    pub fn feed(&mut self, input: &[u8], out: &mut Vec<u8>) -> Result<(), DecodeError> {
        if self.finished {
            if input.is_empty() {
                return Ok(());
            }
            return Err(DecodeError::InvalidData);
        }
        if matches!(self.phase, Phase::Done) {
            // A member completed in an earlier feed; further input
            // starts the next concatenated member (gzip only).
            if input.is_empty() {
                return Ok(());
            }
            if self.coding != Coding::Gzip {
                return Err(DecodeError::InvalidData);
            }
            self.reset_member();
        }
        self.bit.push(input);
        let r = self.drive(out);
        self.bit.compact();
        self.out.drain_history();
        r
    }

    /// Feed the final input chunk and verify the wrapper trailer. Fails
    /// with [`DecodeError::Truncated`] when the stream is incomplete and
    /// [`DecodeError::InvalidData`] on a bad trailer or trailing garbage
    /// (concatenated gzip members are decoded, not treated as garbage).
    /// Idempotent: calling it again after success is a no-op.
    pub fn finish(&mut self, input: &[u8], out: &mut Vec<u8>) -> Result<(), DecodeError> {
        if self.finished {
            return Ok(());
        }
        self.bit.push(input);
        let r = self.drive(out);
        self.bit.compact();
        self.out.drain_history();
        r?;
        if !matches!(self.phase, Phase::Done) {
            return Err(DecodeError::Truncated);
        }
        // Bytes past a complete member are corrupt input, not padding
        // (gzip members chain inside drive(), so none can remain for
        // gzip here).
        if self.bit.ppos < self.bit.pending.len() || self.bit.bits >= 8 {
            return Err(DecodeError::InvalidData);
        }
        self.finished = true;
        Ok(())
    }

    /// Drive the state machine until it suspends (needs more input),
    /// errors, or reaches [`Phase::Done`] with no further input. Every
    /// iteration that does not suspend makes progress (consumes input,
    /// emits output, or advances the phase), so the loop always
    /// terminates. A complete gzip member with bytes still staged starts
    /// the next concatenated member (RFC 1952 §2.2) instead of stopping.
    fn drive(&mut self, out: &mut Vec<u8>) -> Result<(), DecodeError> {
        loop {
            if matches!(self.phase, Phase::Done) {
                // Bytes staged past a complete member belong to the next
                // member. Only gzip has member framing; other codings
                // leave them for finish() to reject as trailing garbage.
                let more = self.bit.bits >= 8 || self.bit.ppos < self.bit.pending.len();
                if more && self.coding == Coding::Gzip {
                    self.reset_member();
                    continue;
                }
                return Ok(());
            }
            let suspend = match self.phase {
                Phase::Sniff => self.sniff()?,
                Phase::GzHead => self.gz_head()?,
                Phase::GzXlen => self.gz_xlen()?,
                Phase::GzSkip(n, next) => self.gz_skip(n, next)?,
                Phase::GzSkipNul(next) => self.gz_skip_nul(next)?,
                Phase::GzHcrc => self.gz_hcrc()?,
                Phase::ZlHead => self.zl_head()?,
                Phase::BlockHead => self.block_head()?,
                Phase::StoredLen => self.stored_len()?,
                Phase::StoredCopy => self.stored_copy(out)?,
                Phase::DynHead => self.dyn_head()?,
                Phase::Symbols => self.symbols(out)?,
                Phase::Trailer => self.trailer()?,
                Phase::Done => unreachable!("handled above"),
            };
            if suspend {
                return Ok(());
            }
        }
    }

    /// [`Coding::Auto`]: the first two bytes decide zlib vs raw. They are
    /// peeked, not consumed — the chosen path re-reads them.
    fn sniff(&mut self) -> Result<bool, DecodeError> {
        debug_assert!(self.bit.bits == 0);
        if self.bit.pending.len() - self.bit.ppos < 2 {
            return Ok(true);
        }
        let cmf = self.bit.pending[self.bit.ppos];
        let flg = self.bit.pending[self.bit.ppos + 1];
        let zlib = u16::from_be_bytes([cmf, flg]).is_multiple_of(31)
            && cmf & 0x0f == 8
            && cmf >> 4 <= 7
            && flg & 0x20 == 0;
        self.coding = if zlib { Coding::Zlib } else { Coding::Raw };
        self.phase = if zlib {
            self.zl_have = 0;
            Phase::ZlHead
        } else {
            Phase::BlockHead
        };
        Ok(false)
    }

    /// Reset per-member state to parse the next concatenated gzip member
    /// (RFC 1952 §2.2). Staged input, decoded-output totals, and the
    /// output cap carry over; checksums, the LZ77 window, and all
    /// header/block state start fresh, since each member is an
    /// independent deflate stream.
    fn reset_member(&mut self) {
        debug_assert_eq!(self.coding, Coding::Gzip);
        self.phase = Phase::GzHead;
        self.bfinal = false;
        self.litlen = None;
        self.dist = None;
        self.sym = SymKind::LitLen;
        self.hprog = HuffProg::fresh();
        self.sym_len = 0;
        self.sym_dist = 0;
        self.dyns = DynState::new();
        self.stored_left = 0;
        self.sl_bytes = [0; 4];
        self.sl_have = 0;
        self.gz = GzHead {
            pos: 0,
            flags: 0,
            xlen: 0,
        };
        self.gx_have = 0;
        self.zl_have = 0;
        self.zl_cmf = 0;
        self.trailer = [0; 8];
        self.trailer_have = 0;
        self.gz_hcrc = 0xffff_ffff;
        self.hcrc = [0; 2];
        self.hcrc_have = 0;
        self.out.crc = 0xffff_ffff;
        self.out.adler_a = 1;
        self.out.adler_b = 0;
        self.out.adler_n = 0;
        self.out.member_out = 0;
        self.out.history.clear();
    }

    /// Fold one gzip header byte into the running FHCRC checksum.
    fn gz_crc_byte(&mut self, b: u8) {
        self.gz_hcrc =
            CRC32_TABLES[0][((self.gz_hcrc ^ b as u32) & 0xff) as usize] ^ (self.gz_hcrc >> 8);
    }

    fn gz_head(&mut self) -> Result<bool, DecodeError> {
        loop {
            let b = match self.bit.read_byte_aligned()? {
                None => return Ok(true),
                Some(b) => b,
            };
            match self.gz.pos {
                0 => {
                    if b != 0x1f {
                        return Err(DecodeError::InvalidData);
                    }
                }
                1 => {
                    if b != 0x8b {
                        return Err(DecodeError::InvalidData);
                    }
                }
                2 => {
                    if b != 8 {
                        return Err(DecodeError::InvalidData);
                    }
                }
                3 => {
                    if b & 0xe0 != 0 {
                        return Err(DecodeError::InvalidData);
                    }
                    self.gz.flags = b;
                }
                4..=9 => {} // MTIME, XFL, OS: parsed but unused.
                _ => return Err(DecodeError::InvalidData),
            }
            // Every header byte feeds the FHCRC checksum (verified later
            // when the flag is set).
            self.gz_crc_byte(b);
            self.gz.pos += 1;
            if self.gz.pos == 10 {
                self.phase = if self.gz.flags & 0x04 != 0 {
                    self.gx_have = 0;
                    Phase::GzXlen
                } else {
                    self.gz_continue(GzNext::Name)
                };
                return Ok(false);
            }
        }
    }

    /// Continue past the fixed gzip header through the optional fields
    /// in flag order: FEXTRA, FNAME, FCOMMENT, FHCRC.
    fn gz_continue(&self, next: GzNext) -> Phase {
        let f = self.gz.flags;
        match next {
            GzNext::Name if f & 0x08 != 0 => Phase::GzSkipNul(GzNext::Comment),
            GzNext::Name => self.gz_continue(GzNext::Comment),
            GzNext::Comment if f & 0x10 != 0 => Phase::GzSkipNul(GzNext::Hcrc),
            GzNext::Comment => self.gz_continue(GzNext::Hcrc),
            GzNext::Hcrc if f & 0x02 != 0 => Phase::GzHcrc,
            GzNext::Hcrc => Phase::BlockHead,
            GzNext::Blocks => Phase::BlockHead,
        }
    }

    fn gz_xlen(&mut self) -> Result<bool, DecodeError> {
        while self.gx_have < 2 {
            match self.bit.read_byte_aligned()? {
                None => return Ok(true),
                Some(b) => {
                    self.gz_crc_byte(b);
                    if self.gx_have == 0 {
                        self.gz.xlen = b as u16;
                    } else {
                        self.gz.xlen |= (b as u16) << 8;
                    }
                    self.gx_have += 1;
                }
            }
        }
        let skip = self.gz.xlen as usize;
        self.phase = Phase::GzSkip(skip, GzNext::Name);
        Ok(false)
    }

    /// Skip FEXTRA bytes; all of them are header bytes for FHCRC.
    fn gz_skip(&mut self, mut n: usize, next: GzNext) -> Result<bool, DecodeError> {
        while n > 0 {
            match self.bit.read_byte_aligned()? {
                None => {
                    self.phase = Phase::GzSkip(n, next);
                    return Ok(true);
                }
                Some(b) => {
                    self.gz_crc_byte(b);
                    n -= 1;
                }
            }
        }
        self.phase = self.gz_continue(next);
        Ok(false)
    }

    /// Skip a NUL-terminated header string (FNAME/FCOMMENT); the bytes,
    /// including the terminator, are header bytes for FHCRC.
    fn gz_skip_nul(&mut self, next: GzNext) -> Result<bool, DecodeError> {
        loop {
            match self.bit.read_byte_aligned()? {
                None => return Ok(true),
                Some(b) => {
                    self.gz_crc_byte(b);
                    if b == 0 {
                        self.phase = self.gz_continue(next);
                        return Ok(false);
                    }
                }
            }
        }
    }

    /// Verify the FHCRC field: the low 16 bits of the CRC-32 over every
    /// header byte up to (not including) this field (RFC 1952 §2.3.1).
    fn gz_hcrc(&mut self) -> Result<bool, DecodeError> {
        while self.hcrc_have < 2 {
            match self.bit.read_byte_aligned()? {
                None => return Ok(true),
                Some(b) => {
                    self.hcrc[self.hcrc_have as usize] = b;
                    self.hcrc_have += 1;
                }
            }
        }
        let stored = u16::from_le_bytes(self.hcrc);
        if (self.gz_hcrc ^ 0xffff_ffff) as u16 != stored {
            return Err(DecodeError::InvalidData);
        }
        self.phase = self.gz_continue(GzNext::Blocks);
        Ok(false)
    }

    fn zl_head(&mut self) -> Result<bool, DecodeError> {
        while self.zl_have < 2 {
            match self.bit.read_byte_aligned()? {
                None => return Ok(true),
                Some(b) => {
                    if self.zl_have == 0 {
                        self.zl_cmf = b;
                    } else {
                        let cmf = self.zl_cmf;
                        if !u16::from_be_bytes([cmf, b]).is_multiple_of(31)
                            || cmf & 0x0f != 8
                            || cmf >> 4 > 7
                            || b & 0x20 != 0
                        {
                            return Err(DecodeError::InvalidData);
                        }
                    }
                    self.zl_have += 1;
                }
            }
        }
        self.phase = Phase::BlockHead;
        Ok(false)
    }

    fn block_head(&mut self) -> Result<bool, DecodeError> {
        // BFINAL(1) + BTYPE(2) in a single read: the three bits are
        // consumed atomically, so a suspension never strands BFINAL.
        let v = match self.bit.read_bits(3)? {
            None => return Ok(true),
            Some(v) => v,
        };
        self.bfinal = v & 1 == 1;
        match (v >> 1) & 3 {
            0 => {
                self.bit.align();
                self.sl_have = 0;
                self.phase = Phase::StoredLen;
            }
            1 => {
                // Valid by construction, as in the one-shot path.
                self.litlen = Some(fixed_litlen());
                self.dist = Some(fixed_dist());
                self.sym = SymKind::LitLen;
                self.phase = Phase::Symbols;
            }
            2 => {
                self.dyns = DynState::new();
                self.phase = Phase::DynHead;
            }
            _ => return Err(DecodeError::InvalidData),
        }
        Ok(false)
    }

    fn stored_len(&mut self) -> Result<bool, DecodeError> {
        while self.sl_have < 4 {
            match self.bit.read_byte_aligned()? {
                None => return Ok(true),
                Some(b) => {
                    self.sl_bytes[self.sl_have as usize] = b;
                    self.sl_have += 1;
                }
            }
        }
        let len = u16::from_le_bytes([self.sl_bytes[0], self.sl_bytes[1]]) as usize;
        let nlen = u16::from_le_bytes([self.sl_bytes[2], self.sl_bytes[3]]);
        if len as u16 ^ 0xffff != nlen {
            return Err(DecodeError::InvalidData);
        }
        self.stored_left = len;
        self.phase = Phase::StoredCopy;
        Ok(false)
    }

    fn stored_copy(&mut self, out: &mut Vec<u8>) -> Result<bool, DecodeError> {
        loop {
            if self.stored_left == 0 {
                self.end_of_block()?;
                return Ok(false);
            }
            // Whole bytes already staged in the bit buffer come first.
            while self.bit.bits >= 8 && self.stored_left > 0 {
                let b = (self.bit.bitbuf & 0xff) as u8;
                self.bit.bitbuf >>= 8;
                self.bit.bits -= 8;
                self.out.emit_byte(b, out)?;
                self.stored_left -= 1;
            }
            if self.stored_left == 0 {
                continue;
            }
            let avail = self.bit.pending.len() - self.bit.ppos;
            if avail == 0 {
                return Ok(true);
            }
            let take = avail.min(self.stored_left);
            let ppos = self.bit.ppos;
            self.out
                .emit_chunk(&self.bit.pending[ppos..ppos + take], out)?;
            self.bit.ppos += take;
            self.stored_left -= take;
        }
    }

    /// Transition at the end of a block (end code or stored data
    /// exhausted): when BFINAL the wrapper trailer follows — it starts
    /// on a byte boundary, so padding bits are dropped — otherwise the
    /// next block header starts at the very next bit.
    fn end_of_block(&mut self) -> Result<(), DecodeError> {
        if self.bfinal {
            self.bit.align();
            self.phase = match self.coding {
                Coding::Raw => Phase::Done,
                Coding::Gzip | Coding::Zlib => Phase::Trailer,
                // Unreachable: `sniff` resolves `Auto` before any block.
                Coding::Auto => return Err(DecodeError::InvalidData),
            };
        } else {
            self.phase = Phase::BlockHead;
        }
        Ok(())
    }

    fn dyn_head(&mut self) -> Result<bool, DecodeError> {
        loop {
            match self.dyns.stage {
                DynStage::Hlit => match self.bit.read_bits(5)? {
                    None => return Ok(true),
                    Some(v) => {
                        self.dyns.hlit = v as usize + 257;
                        self.dyns.stage = DynStage::Hdist;
                    }
                },
                DynStage::Hdist => match self.bit.read_bits(5)? {
                    None => return Ok(true),
                    Some(v) => {
                        self.dyns.hdist = v as usize + 1;
                        self.dyns.stage = DynStage::Hclen;
                    }
                },
                DynStage::Hclen => match self.bit.read_bits(4)? {
                    None => return Ok(true),
                    Some(v) => {
                        self.dyns.hclen = v as usize + 4;
                        self.dyns.stage = DynStage::ClLens;
                    }
                },
                DynStage::ClLens => {
                    if self.dyns.cl_read >= self.dyns.hclen {
                        self.dyns.cl = Some(build_huffman(&self.dyns.cl_lens)?);
                        self.dyns.stage = DynStage::Lengths;
                        continue;
                    }
                    match self.bit.read_bits(3)? {
                        None => return Ok(true),
                        Some(v) => {
                            let i = self.dyns.cl_read;
                            self.dyns.cl_lens[CL_ORDER[i]] = v as u8;
                            self.dyns.cl_read += 1;
                        }
                    }
                }
                DynStage::Lengths => {
                    let total = self.dyns.hlit + self.dyns.hdist;
                    if self.dyns.lens.len() >= total {
                        let hlit = self.dyns.hlit;
                        let litlen = build_huffman(&self.dyns.lens[..hlit])?;
                        let dist = build_huffman(&self.dyns.lens[hlit..])?;
                        self.litlen = Some(litlen);
                        self.dist = Some(dist);
                        self.dyns = DynState::new();
                        self.sym = SymKind::LitLen;
                        self.phase = Phase::Symbols;
                        return Ok(false);
                    }
                    // A repeat code whose extra bits are still unread.
                    if self.dyns.rep != 0 {
                        let rep = self.dyns.rep;
                        let (extra, base) = match rep {
                            16 => (2u32, 3usize),
                            17 => (3, 3),
                            _ => (7, 11),
                        };
                        match self.bit.read_bits(extra)? {
                            None => return Ok(true),
                            Some(e) => {
                                let count = e as usize + base;
                                if self.dyns.lens.len() + count > total {
                                    return Err(DecodeError::InvalidData);
                                }
                                let fill = if rep == 16 {
                                    *self.dyns.lens.last().ok_or(DecodeError::InvalidData)?
                                } else {
                                    0
                                };
                                self.dyns.lens.extend(std::iter::repeat_n(fill, count));
                                self.dyns.rep = 0;
                            }
                        }
                        continue;
                    }
                    let sym = {
                        let d = &mut self.dyns;
                        let cl = d.cl.as_ref().ok_or(DecodeError::InvalidData)?;
                        cl.decode_resume(&mut d.clp, &mut self.bit)?
                    };
                    match sym {
                        None => return Ok(true),
                        Some(s) if s <= 15 => self.dyns.lens.push(s as u8),
                        Some(16) => {
                            if self.dyns.lens.is_empty() {
                                return Err(DecodeError::InvalidData);
                            }
                            self.dyns.rep = 16;
                        }
                        Some(17) => self.dyns.rep = 17,
                        Some(18) => self.dyns.rep = 18,
                        _ => return Err(DecodeError::InvalidData),
                    }
                }
            }
        }
    }

    fn symbols(&mut self, out: &mut Vec<u8>) -> Result<bool, DecodeError> {
        loop {
            match self.sym {
                SymKind::LitLen => {
                    let sym = {
                        let litlen = self.litlen.as_ref().ok_or(DecodeError::InvalidData)?;
                        litlen.decode_resume(&mut self.hprog, &mut self.bit)?
                    };
                    match sym {
                        None => return Ok(true),
                        Some(s) if s < 256 => {
                            self.out.emit_byte(s as u8, out)?;
                        }
                        Some(256) => {
                            self.end_of_block()?;
                            self.sym = SymKind::LitLen;
                            return Ok(false);
                        }
                        Some(s) if s <= 285 => {
                            let (base, extra) = length_base_extra(s);
                            self.sym_len = base;
                            self.sym = SymKind::LenExtra { extra };
                        }
                        _ => return Err(DecodeError::InvalidData),
                    }
                }
                SymKind::LenExtra { extra } => match self.bit.read_bits(extra)? {
                    None => return Ok(true),
                    Some(e) => {
                        self.sym_len += e as usize;
                        self.sym = SymKind::Dist;
                    }
                },
                SymKind::Dist => {
                    let sym = {
                        let dist = self.dist.as_ref().ok_or(DecodeError::InvalidData)?;
                        dist.decode_resume(&mut self.hprog, &mut self.bit)?
                    };
                    match sym {
                        None => return Ok(true),
                        Some(d) => {
                            if d > 29 {
                                return Err(DecodeError::InvalidData);
                            }
                            let (base, extra) = dist_base_extra(d);
                            self.sym_dist = base;
                            self.sym = SymKind::DistExtra { extra };
                        }
                    }
                }
                SymKind::DistExtra { extra } => match self.bit.read_bits(extra)? {
                    None => return Ok(true),
                    Some(e) => {
                        let dist = self.sym_dist + e as usize;
                        let len = self.sym_len;
                        self.out.emit_match(dist, len, out)?;
                        self.sym = SymKind::LitLen;
                    }
                },
            }
        }
    }

    fn trailer(&mut self) -> Result<bool, DecodeError> {
        let need = match self.coding {
            Coding::Gzip => 8,
            Coding::Zlib => 4,
            Coding::Raw => 0,
            Coding::Auto => return Err(DecodeError::InvalidData),
        };
        while self.trailer_have < need {
            match self.bit.read_byte_aligned()? {
                None => return Ok(true),
                Some(b) => {
                    self.trailer[self.trailer_have as usize] = b;
                    self.trailer_have += 1;
                }
            }
        }
        match self.coding {
            Coding::Gzip => {
                let crc = u32::from_le_bytes(self.trailer[0..4].try_into().unwrap());
                let isize = u32::from_le_bytes(self.trailer[4..8].try_into().unwrap());
                if self.out.crc ^ 0xffff_ffff != crc
                    || self.out.member_out & 0xffff_ffff != isize as u64
                {
                    return Err(DecodeError::InvalidData);
                }
            }
            Coding::Zlib => {
                let adler = u32::from_be_bytes(self.trailer[0..4].try_into().unwrap());
                let expect = (self.out.adler_b % 65521) << 16 | (self.out.adler_a % 65521);
                if expect != adler {
                    return Err(DecodeError::InvalidData);
                }
            }
            Coding::Raw => {}
            Coding::Auto => return Err(DecodeError::InvalidData),
        }
        self.phase = Phase::Done;
        Ok(false)
    }
}

// ---------------------------------------------------------------------------
// Stacked content codings
// ---------------------------------------------------------------------------

/// An incremental decode pipeline for stacked `Content-Encoding` layers,
/// outermost first (e.g. `gzip, gzip`). Built by
/// [`DecodeChain::new`]; feed input with [`feed`](Self::feed) and verify
/// trailers with [`finish`](Self::finish).
pub struct DecodeChain {
    decoders: Vec<Decoder>,
    /// Scratch buffers between layers; `bufs[i]` carries decoder `i`'s
    /// output into decoder `i + 1`.
    bufs: Vec<Vec<u8>>,
    fed_input: u64,
}

impl DecodeChain {
    /// Build the pipeline for a `Content-Encoding` header value. Returns
    /// `Ok(None)` when the value needs no decoding (empty or `identity`
    /// only). Tokens decode in reverse application order, mirroring
    /// [`decode_content_encoding`]. At most [`MAX_CODING_DEPTH`] layers
    /// are stacked: each layer reprocesses every byte, so deeper stacks
    /// are a CPU-amplification vector.
    pub fn new(value: &str, max_output: usize) -> Result<Option<Self>, DecodeError> {
        let mut codings = Vec::new();
        for token in value.rsplit(',') {
            let token = token.split(';').next().unwrap_or("").trim();
            if token.is_empty() || token.eq_ignore_ascii_case("identity") {
                continue;
            } else if token.eq_ignore_ascii_case("gzip") || token.eq_ignore_ascii_case("x-gzip") {
                codings.push(Coding::Gzip);
            } else if token.eq_ignore_ascii_case("deflate") {
                codings.push(Coding::Auto);
            } else {
                return Err(DecodeError::UnsupportedEncoding);
            }
        }
        if codings.len() > MAX_CODING_DEPTH {
            return Err(DecodeError::CodingDepthExceeded);
        }
        if codings.is_empty() {
            return Ok(None);
        }
        let bufs = (0..codings.len().saturating_sub(1))
            .map(|_| Vec::new())
            .collect();
        Ok(Some(Self {
            decoders: codings
                .into_iter()
                .map(|c| Decoder::new(c, max_output))
                .collect(),
            bufs,
            fed_input: 0,
        }))
    }

    /// Feed the next input chunk through every layer; final decoded bytes
    /// are appended to `out`.
    pub fn feed(&mut self, input: &[u8], out: &mut Vec<u8>) -> Result<(), DecodeError> {
        self.fed_input += input.len() as u64;
        let n = self.decoders.len();
        for i in 0..n {
            if i + 1 == n {
                let src: &[u8] = if i == 0 { input } else { &self.bufs[i - 1] };
                self.decoders[i].feed(src, out)?;
            } else {
                // Disjoint split: `src` borrows `bufs[..i]` while `dst`
                // mutably borrows `bufs[i]`.
                let (head, tail) = self.bufs.split_at_mut(i);
                let src: &[u8] = if i == 0 { input } else { &head[i - 1] };
                let dst = &mut tail[0];
                dst.clear();
                self.decoders[i].feed(src, dst)?;
            }
        }
        Ok(())
    }

    /// Feed the final input (usually empty: all input went through
    /// [`feed`](Self::feed)) and verify every layer's trailer.
    pub fn finish(&mut self, out: &mut Vec<u8>) -> Result<(), DecodeError> {
        let n = self.decoders.len();
        for i in 0..n {
            if i + 1 == n {
                let src: &[u8] = if i == 0 { &[] } else { &self.bufs[i - 1] };
                self.decoders[i].finish(src, out)?;
            } else {
                let (head, tail) = self.bufs.split_at_mut(i);
                let src: &[u8] = if i == 0 { &[] } else { &head[i - 1] };
                let dst = &mut tail[0];
                dst.clear();
                self.decoders[i].finish(src, dst)?;
            }
        }
        Ok(())
    }

    /// Total encoded bytes fed through the outermost layer so far.
    pub fn fed_input(&self) -> u64 {
        self.fed_input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! fixture {
        ($name:expr) => {
            include_bytes!(concat!("../tests/fixtures/decode/", $name))
        };
    }

    const PLAIN: &[u8] = fixture!("hello.txt");
    // 512 KiB keeps the cap tests fast while exercising the window.
    const CAP: usize = 512 * 1024;

    #[test]
    fn gzip_roundtrip() {
        assert_eq!(gunzip(fixture!("hello.gz"), CAP).unwrap(), PLAIN);
    }

    #[test]
    fn gzip_empty() {
        assert_eq!(gunzip(fixture!("empty.gz"), CAP).unwrap(), b"");
    }

    #[test]
    fn gzip_binary() {
        assert_eq!(
            gunzip(fixture!("binary.gz"), 2 * 1024 * 1024).unwrap(),
            fixture!("binary.bin")
        );
    }

    #[test]
    fn gzip_with_header_flags() {
        // FEXTRA + FNAME + FCOMMENT + FHCRC all present.
        assert_eq!(gunzip(fixture!("hello.flags.gz"), CAP).unwrap(), PLAIN);
    }

    #[test]
    fn gzip_stored_blocks() {
        assert_eq!(gunzip(fixture!("hello.stored.gz"), CAP).unwrap(), PLAIN);
    }

    #[test]
    fn gzip_truncated_fails() {
        assert_eq!(
            gunzip(fixture!("truncated.gz"), CAP).unwrap_err(),
            DecodeError::Truncated
        );
    }

    #[test]
    fn gzip_bad_trailer_fails() {
        assert_eq!(
            gunzip(fixture!("badcrc.gz"), CAP).unwrap_err(),
            DecodeError::InvalidData
        );
    }

    #[test]
    fn gzip_bad_payload_fails() {
        assert_eq!(
            gunzip(fixture!("baddata.gz"), CAP).unwrap_err(),
            DecodeError::InvalidData
        );
    }

    #[test]
    fn gzip_not_gzip_fails() {
        assert_eq!(
            gunzip(b"definitely not gzip", CAP).unwrap_err(),
            DecodeError::InvalidData
        );
    }

    /// The slicing-by-4 [`crc32`] matches the classic byte-at-a-time
    /// table drive on every length class (including lengths that are
    /// not multiples of 4, exercising the tail loop) and on
    /// adversarial byte patterns.
    #[test]
    fn crc32_slicing_matches_byte_at_a_time() {
        fn reference(data: &[u8]) -> u32 {
            let mut c = 0xffff_ffffu32;
            for &b in data {
                c = CRC32_TABLES[0][((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
            }
            c ^ 0xffff_ffff
        }
        // Known answer: CRC32("123456789") = 0xCBF43926.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(reference(b"123456789"), 0xCBF4_3926);
        // Length sweep across the 4-byte chunk boundary, on
        // pseudorandom data and on hostile patterns (runs, alternating
        // bits, all-zeros, all-ones).
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut next_byte = || {
            rng = rng
                .wrapping_mul(0x5851_f42d_4c95_7f2d)
                .wrapping_add(0x1405_7b7e_f767_814f);
            (rng >> 33) as u8
        };
        for len in 0..=65usize {
            let mut data: Vec<u8> = (0..len).map(|_| next_byte()).collect();
            assert_eq!(crc32(&data), reference(&data), "random len {len}");
            data.iter_mut().for_each(|b| *b = 0x00);
            assert_eq!(crc32(&data), reference(&data), "zeros len {len}");
            data.iter_mut().for_each(|b| *b = 0xff);
            assert_eq!(crc32(&data), reference(&data), "ones len {len}");
            for (i, b) in data.iter_mut().enumerate() {
                *b = if i % 2 == 0 { 0x55 } else { 0xaa };
            }
            assert_eq!(crc32(&data), reference(&data), "alternating len {len}");
        }
        // `crc32_update` composes: chunked updates equal one shot.
        let big: Vec<u8> = (0..1000).map(|_| next_byte()).collect();
        let mut c = 0xffff_ffffu32;
        for chunk in big.chunks(7) {
            c = crc32_update(c, chunk);
        }
        assert_eq!(c ^ 0xffff_ffff, crc32(&big));
    }

    #[test]
    fn zlib_roundtrip() {
        assert_eq!(inflate_zlib(fixture!("hello.zlib"), CAP).unwrap(), PLAIN);
    }

    #[test]
    fn raw_deflate_roundtrip() {
        assert_eq!(inflate_raw(fixture!("hello.raw"), CAP).unwrap(), PLAIN);
        assert_eq!(
            inflate_raw(fixture!("hello.stored.raw"), CAP).unwrap(),
            PLAIN
        );
        assert_eq!(
            inflate_raw(fixture!("hello.fixed.raw"), CAP).unwrap(),
            PLAIN
        );
    }

    #[test]
    fn output_cap_enforced() {
        assert_eq!(
            gunzip(fixture!("hello.gz"), 16).unwrap_err(),
            DecodeError::OutputTooLarge
        );
        assert_eq!(
            inflate_zlib(fixture!("hello.zlib"), 16).unwrap_err(),
            DecodeError::OutputTooLarge
        );
        assert_eq!(
            inflate_raw(fixture!("hello.stored.raw"), 16).unwrap_err(),
            DecodeError::OutputTooLarge
        );
    }

    #[test]
    fn content_encoding_dispatch() {
        let gz = fixture!("hello.gz");
        assert_eq!(decode_content_encoding(gz, "gzip", CAP).unwrap(), PLAIN);
        assert_eq!(decode_content_encoding(gz, "x-gzip", CAP).unwrap(), PLAIN);
        assert_eq!(decode_content_encoding(gz, "GZip", CAP).unwrap(), PLAIN);
        let zl = fixture!("hello.zlib");
        assert_eq!(decode_content_encoding(zl, "deflate", CAP).unwrap(), PLAIN);
        // Raw deflate mislabeled as `deflate` (common server bug).
        let raw = fixture!("hello.raw");
        assert_eq!(decode_content_encoding(raw, "deflate", CAP).unwrap(), PLAIN);
        // Stacked encodings decode in reverse application order.
        let dbl = fixture!("double.gz");
        assert_eq!(
            decode_content_encoding(dbl, "gzip, gzip", CAP).unwrap(),
            PLAIN
        );
        // Identity and empty values pass through untouched.
        assert_eq!(
            decode_content_encoding(PLAIN, "identity", CAP).unwrap(),
            PLAIN
        );
        assert_eq!(decode_content_encoding(PLAIN, "", CAP).unwrap(), PLAIN);
        assert_eq!(decode_content_encoding(PLAIN, "  ", CAP).unwrap(), PLAIN);
    }

    #[test]
    fn unsupported_encoding_fails() {
        assert_eq!(
            decode_content_encoding(PLAIN, "br", CAP).unwrap_err(),
            DecodeError::UnsupportedEncoding
        );
    }

    #[test]
    fn deflate_cap_not_retried_as_raw() {
        // The cap breach surfaces even though raw fallback exists.
        assert_eq!(
            decode_content_encoding(fixture!("hello.zlib"), "deflate", 16).unwrap_err(),
            DecodeError::OutputTooLarge
        );
    }

    // ------------------------------------------------------------------
    // Test helpers
    // ------------------------------------------------------------------

    /// Build a minimal gzip member around `payload` with a stored
    /// (uncompressed) deflate block. When `with_fhcrc`, an FHCRC field is
    /// appended; `fhcrc_ok` selects a correct vs. corrupted header CRC16.
    fn make_member(payload: &[u8], with_fhcrc: bool, fhcrc_ok: bool) -> Vec<u8> {
        assert!(payload.len() < 65536);
        let mut m = vec![0x1fu8, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff];
        if with_fhcrc {
            m[3] |= 0x02;
            let crc = crc32(&m) as u16;
            let stored = if fhcrc_ok { crc } else { crc ^ 0xffff };
            m.extend_from_slice(&stored.to_le_bytes());
        }
        // Stored deflate block: BFINAL=1, BTYPE=00, then LEN/NLEN.
        let len = payload.len() as u16;
        m.push(0x01);
        m.extend_from_slice(&len.to_le_bytes());
        m.extend_from_slice(&(len ^ 0xffff).to_le_bytes());
        m.extend_from_slice(payload);
        m.extend_from_slice(&crc32(payload).to_le_bytes());
        m.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        m
    }

    #[test]
    fn gzip_concatenated_members_oneshot() {
        let a = make_member(b"hello ", false, true);
        let b = make_member(b"world", true, true);
        let mut data = a.clone();
        data.extend_from_slice(&b);
        data.extend_from_slice(&a);
        assert_eq!(gunzip(&data, CAP).unwrap(), b"hello worldhello ");
        // Single members and the empty-input edge are unchanged.
        assert_eq!(gunzip(&a, CAP).unwrap(), b"hello ");
        assert_eq!(gunzip(&[], CAP).unwrap_err(), DecodeError::Truncated);
        // The cap applies across the concatenated output.
        assert_eq!(gunzip(&data, 8).unwrap_err(), DecodeError::OutputTooLarge);
    }

    #[test]
    fn gzip_concatenated_member_bad_second_trailer() {
        let mut data = make_member(b"ab", false, true);
        let mut bad = make_member(b"cd", false, true);
        let last = bad.len() - 1;
        bad[last] ^= 0xff; // corrupt ISIZE of the 2nd member
        data.extend_from_slice(&bad);
        assert_eq!(gunzip(&data, CAP).unwrap_err(), DecodeError::InvalidData);
    }

    #[test]
    fn gzip_truncated_second_member_fails() {
        let mut data = make_member(b"ab", false, true);
        let second = make_member(b"cd", false, true);
        data.extend_from_slice(&second[..second.len() - 3]);
        assert_eq!(gunzip(&data, CAP).unwrap_err(), DecodeError::Truncated);
    }

    /// Build a hostile gzip member: `blocks` stored deflate blocks of
    /// `block_len` zero bytes each, so the decoded payload is
    /// `blocks * block_len` bytes. Stored blocks grow the output `Vec`
    /// by exact powers of two when `block_len` is one, so capacity lands
    /// exactly on the decoded length with no allocator slack to hide
    /// behind: any per-member temporary shows up in the peak at full
    /// size.
    fn make_hostile_member(blocks: usize, block_len: usize) -> Vec<u8> {
        assert!(block_len < 65536, "stored blocks carry a u16 length");
        let mut m = vec![0x1fu8, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0];
        for b in 0..blocks {
            let last = b + 1 == blocks;
            // 3-bit block header: BFINAL | BTYPE = 00 (stored).
            m.push(if last { 0x01 } else { 0x00 });
            let len = block_len as u16;
            m.extend_from_slice(&len.to_le_bytes());
            m.extend_from_slice(&(len ^ 0xffff).to_le_bytes());
            m.extend(std::iter::repeat_n(0, block_len));
        }
        let payload = vec![0u8; blocks * block_len];
        m.extend_from_slice(&crc32(&payload).to_le_bytes());
        m.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        m
    }

    /// Members decode straight into the caller's capped buffer: on a
    /// hostile 1 MiB member the decode itself must not allocate a
    /// per-member temporary, so peak heap beyond the pre-sized output
    /// buffer stays within small slack instead of ~1x the cap extra.
    /// Uses the counting global allocator shared with the TLS benches
    /// (`crate::tls::tests::bench`), whose thread-local counters keep
    /// parallel test threads from polluting the measurement.
    #[test]
    fn gzip_member_decode_peak_stays_within_cap() {
        use crate::tls::tests::bench::{peak_live_bytes, reset_counters};

        const CAP: usize = 1024 * 1024;
        // 64 KiB of slack for decoder tables and bit-level staging; far
        // below the extra CAP a per-member temporary would cost.
        const SLACK: u64 = 64 * 1024;

        let data = make_hostile_member(32, 32 * 1024);
        // The capped buffer is sized up front, outside the probe window,
        // so its own growth cannot hide (or fake) the decoder's peak.
        let mut out = Vec::with_capacity(CAP);
        reset_counters();
        let used = gunzip_member_into(&data, &mut out, CAP).unwrap();
        let peak = peak_live_bytes();
        assert_eq!(used, data.len());
        assert_eq!(out.len(), CAP);
        assert!(out.iter().all(|&b| b == 0));
        assert!(
            peak <= SLACK,
            "hostile gzip member allocated {peak} heap bytes beyond the capped buffer"
        );
    }

    #[test]
    fn gzip_fhcrc_verified_oneshot() {
        // Correct header CRC passes (also exercised by hello.flags.gz).
        assert_eq!(
            gunzip(&make_member(b"data", true, true), CAP).unwrap(),
            b"data"
        );
        // Corrupt CRC field fails ...
        assert_eq!(
            gunzip(&make_member(b"data", true, false), CAP).unwrap_err(),
            DecodeError::InvalidData
        );
        // ... as does a flipped header byte under a correct field.
        let mut m = make_member(b"data", true, true);
        m[5] ^= 0x01;
        assert_eq!(gunzip(&m, CAP).unwrap_err(), DecodeError::InvalidData);
    }

    // ------------------------------------------------------------------
    // Streaming decoder
    // ------------------------------------------------------------------

    /// Drive `coding` over `data` in `chunk`-sized feeds, then finish.
    fn feed_all(
        coding: Coding,
        data: &[u8],
        chunk: usize,
        max: usize,
    ) -> Result<Vec<u8>, DecodeError> {
        let mut d = Decoder::new(coding, max);
        let mut out = Vec::new();
        for piece in data.chunks(chunk) {
            d.feed(piece, &mut out)?;
        }
        d.finish(&[], &mut out)?;
        Ok(out)
    }

    #[test]
    fn streaming_matches_oneshot_at_any_chunk_size() {
        // (compressed, expected plain, coding). Odd chunk sizes suspend
        // the inflater mid-symbol, mid-block, and mid-header.
        let cases: &[(&[u8], &[u8], Coding)] = &[
            (fixture!("hello.gz"), PLAIN, Coding::Gzip),
            (fixture!("empty.gz"), b"", Coding::Gzip),
            (fixture!("binary.gz"), fixture!("binary.bin"), Coding::Gzip),
            (fixture!("hello.flags.gz"), PLAIN, Coding::Gzip),
            (fixture!("hello.stored.gz"), PLAIN, Coding::Gzip),
            (fixture!("hello.zlib"), PLAIN, Coding::Zlib),
            (fixture!("hello.raw"), PLAIN, Coding::Raw),
            (fixture!("hello.fixed.raw"), PLAIN, Coding::Raw),
            (fixture!("hello.stored.raw"), PLAIN, Coding::Raw),
        ];
        for &(data, expected, coding) in cases {
            for &chunk in &[1usize, 2, 3, 5, 7, 13, 64, 1024, 1 << 20] {
                let got = feed_all(coding, data, chunk, 4 * 1024 * 1024).unwrap();
                assert_eq!(got, expected, "coding={coding:?} chunk={chunk}");
            }
        }
    }

    #[test]
    fn streaming_large_output_stays_correct_across_window() {
        // 64 KiB of output forces the 32 KiB history window to drain
        // mid-stream; back-references must still resolve.
        let got = feed_all(Coding::Gzip, fixture!("binary.gz"), 700, 2 * 1024 * 1024).unwrap();
        assert_eq!(got, fixture!("binary.bin"));
    }

    #[test]
    fn streaming_auto_sniffs_zlib_and_raw() {
        assert_eq!(
            feed_all(Coding::Auto, fixture!("hello.zlib"), 3, CAP).unwrap(),
            PLAIN
        );
        assert_eq!(
            feed_all(Coding::Auto, fixture!("hello.raw"), 3, CAP).unwrap(),
            PLAIN
        );
        // The two sniff bytes may arrive in separate feeds.
        let mut d = Decoder::new(Coding::Auto, CAP);
        let mut out = Vec::new();
        let z = fixture!("hello.zlib");
        d.feed(&z[..1], &mut out).unwrap();
        assert!(out.is_empty());
        d.feed(&z[1..], &mut out).unwrap();
        d.finish(&[], &mut out).unwrap();
        assert_eq!(out, PLAIN);
    }

    #[test]
    fn streaming_chain_decodes_stacked_gzip() {
        let mut chain = DecodeChain::new("gzip, gzip", CAP).unwrap().unwrap();
        let mut out = Vec::new();
        for piece in fixture!("double.gz").chunks(5) {
            chain.feed(piece, &mut out).unwrap();
        }
        chain.finish(&mut out).unwrap();
        assert_eq!(out, PLAIN);
    }

    #[test]
    fn streaming_chain_construction() {
        assert!(DecodeChain::new("", CAP).unwrap().is_none());
        assert!(DecodeChain::new("identity", CAP).unwrap().is_none());
        assert!(DecodeChain::new(" , identity, ", CAP).unwrap().is_none());
        assert!(DecodeChain::new("gzip", CAP).unwrap().is_some());
        assert!(matches!(
            DecodeChain::new("br", CAP),
            Err(DecodeError::UnsupportedEncoding)
        ));
        assert!(matches!(
            DecodeChain::new("gzip, br", CAP),
            Err(DecodeError::UnsupportedEncoding)
        ));
    }

    #[test]
    fn streaming_truncated_input_fails_at_finish() {
        let err = feed_all(Coding::Gzip, fixture!("truncated.gz"), 7, CAP).unwrap_err();
        assert_eq!(err, DecodeError::Truncated);
        // So does finishing without feeding anything.
        let mut d = Decoder::new(Coding::Gzip, CAP);
        let mut out = Vec::new();
        assert_eq!(d.finish(&[], &mut out).unwrap_err(), DecodeError::Truncated);
    }

    #[test]
    fn streaming_bad_trailer_fails() {
        assert_eq!(
            feed_all(Coding::Gzip, fixture!("badcrc.gz"), 11, CAP).unwrap_err(),
            DecodeError::InvalidData
        );
    }

    #[test]
    fn streaming_bad_payload_fails() {
        assert_eq!(
            feed_all(Coding::Gzip, fixture!("baddata.gz"), 11, CAP).unwrap_err(),
            DecodeError::InvalidData
        );
    }

    #[test]
    fn streaming_trailing_garbage_rejected() {
        let mut data = fixture!("hello.gz").to_vec();
        data.extend_from_slice(b"junk");
        assert_eq!(
            feed_all(Coding::Gzip, &data, 7, CAP).unwrap_err(),
            DecodeError::InvalidData
        );
        // ... but a *valid* second member is accepted (RFC 1952), not
        // treated as garbage.
        let mut cat = fixture!("hello.gz").to_vec();
        cat.extend_from_slice(fixture!("empty.gz"));
        cat.extend_from_slice(fixture!("hello.gz"));
        let mut expected = PLAIN.to_vec();
        expected.extend_from_slice(PLAIN);
        assert_eq!(feed_all(Coding::Gzip, &cat, 7, CAP).unwrap(), expected);
    }

    #[test]
    fn streaming_concatenated_members_at_any_chunk_size() {
        let mut data = make_member(b"one ", false, true);
        data.extend_from_slice(&make_member(b"two ", true, true));
        data.extend_from_slice(&make_member(b"three", false, true));
        // Odd chunk sizes split members mid-header, mid-block, and
        // mid-trailer.
        for &chunk in &[1usize, 2, 3, 5, 7, 13, 64, 1024, 1 << 20] {
            let got = feed_all(Coding::Gzip, &data, chunk, CAP).unwrap();
            assert_eq!(got, b"one two three", "chunk={chunk}");
        }
    }

    #[test]
    fn streaming_concatenated_members_split_across_feeds() {
        // The first member completes in one feed; the second arrives in a
        // later feed (the fetch layer feeds per read).
        let a = make_member(b"one ", false, true);
        let b = make_member(b"two", false, true);
        let mut d = Decoder::new(Coding::Gzip, CAP);
        let mut out = Vec::new();
        d.feed(&a, &mut out).unwrap();
        assert_eq!(out, b"one ");
        d.feed(&b, &mut out).unwrap();
        d.finish(&[], &mut out).unwrap();
        assert_eq!(out, b"one two");
    }

    #[test]
    fn streaming_concatenated_truncated_second_member_fails() {
        let mut data = make_member(b"ab", false, true);
        let second = make_member(b"cd", false, true);
        data.extend_from_slice(&second[..second.len() - 3]);
        assert_eq!(
            feed_all(Coding::Gzip, &data, 5, CAP).unwrap_err(),
            DecodeError::Truncated
        );
    }

    #[test]
    fn streaming_fhcrc_verified() {
        assert_eq!(
            feed_all(Coding::Gzip, &make_member(b"data", true, true), 3, CAP).unwrap(),
            b"data"
        );
        assert_eq!(
            feed_all(Coding::Gzip, &make_member(b"data", true, false), 3, CAP).unwrap_err(),
            DecodeError::InvalidData
        );
        let mut m = make_member(b"data", true, true);
        m[7] ^= 0x80; // OS byte under a correct FHCRC field
        assert_eq!(
            feed_all(Coding::Gzip, &m, 3, CAP).unwrap_err(),
            DecodeError::InvalidData
        );
    }

    /// Build a gzip member whose deflate body is hand-rolled fixed
    /// Huffman: BFINAL=1, literal 'x', one match, end-of-block.
    /// `len_rev`/`dist_rev` are the length/distance codes in transmission
    /// order (Huffman codes transmit MSB-first per RFC 1951 §3.2.6, so
    /// callers pass the bit-reversed code); `dist_extra` carries
    /// `dist_extra_bits` LSB-first extra distance bits.
    fn match_member(
        len_rev: u64,
        dist_rev: u64,
        dist_extra: u64,
        dist_extra_bits: u32,
        payload: &[u8],
    ) -> Vec<u8> {
        let mut deflate = Vec::new();
        let mut bits: u64 = 0;
        let mut nbits = 0u32;
        let mut emit = |v: u64, n: u32| {
            bits |= v << nbits;
            nbits += n;
            while nbits >= 8 {
                deflate.push(bits as u8);
                bits >>= 8;
                nbits -= 8;
            }
        };
        emit(0b011, 3); // BFINAL=1, BTYPE=01 (fixed Huffman)
        emit(0x15, 8); // 'x' = 0xA8 MSB-first
        emit(len_rev, 7);
        emit(dist_rev, 5);
        emit(dist_extra, dist_extra_bits);
        emit(0, 7); // end of block (sym 256)
        if nbits > 0 {
            deflate.push(bits as u8);
        }
        let mut m = vec![0x1fu8, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff];
        m.extend_from_slice(&deflate);
        m.extend_from_slice(&crc32(payload).to_le_bytes());
        m.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        m
    }

    #[test]
    fn streaming_concatenated_members_reset_window() {
        // Control: an in-member back-reference (dist=1) decodes fine in
        // a later member, proving the hand-rolled bits are well-formed:
        // len 3 is sym 257 (code 0000001), dist 1 is sym 0 (code 00000).
        let mut data = make_member(b"AAAA", false, true);
        data.extend_from_slice(&match_member(0x40, 0, 0, 0, b"xxxx"));
        assert_eq!(feed_all(Coding::Gzip, &data, 7, CAP).unwrap(), b"AAAAxxxx");
        // A back-reference in the second member must not reach into the
        // first member's output: each member is an independent deflate
        // stream with a fresh LZ77 window. dist=5 (sym 4, code 00100,
        // one zero extra bit) with only "x" decoded is invalid; a stale
        // window would wrongly resolve it into the first member's "AAAA".
        let mut data = make_member(b"AAAA", false, true);
        data.extend_from_slice(&match_member(0x20, 4, 0, 1, b"xxxx"));
        assert_eq!(
            feed_all(Coding::Gzip, &data, 7, CAP).unwrap_err(),
            DecodeError::InvalidData
        );
        // One-shot path agrees.
        assert_eq!(gunzip(&data, CAP).unwrap_err(), DecodeError::InvalidData);
    }

    #[test]
    fn coding_depth_capped() {
        let four = "gzip, gzip, gzip, gzip";
        let five = "gzip, gzip, gzip, gzip, gzip";
        assert!(matches!(DecodeChain::new(four, CAP), Ok(Some(_))));
        assert!(matches!(
            DecodeChain::new(five, CAP),
            Err(DecodeError::CodingDepthExceeded)
        ));
        // Five genuinely nested layers: the fifth is rejected even
        // though every layer decodes cleanly.
        let mut nested = b"deep".to_vec();
        for _ in 0..5 {
            nested = make_member(&nested, false, true);
        }
        assert_eq!(
            decode_content_encoding(&nested, five, CAP).unwrap_err(),
            DecodeError::CodingDepthExceeded
        );
        // Garbage still fails fast as invalid data, before any depth
        // accounting matters.
        assert_eq!(
            decode_content_encoding(PLAIN, five, CAP).unwrap_err(),
            DecodeError::InvalidData
        );
        // `identity` and empty tokens are not layers.
        assert!(matches!(
            DecodeChain::new("gzip, identity, , gzip, gzip, gzip", CAP),
            Ok(Some(_))
        ));
    }

    #[test]
    fn coding_depth_four_decodes() {
        let mut body = b"deep".to_vec();
        for _ in 0..4 {
            body = make_member(&body, false, true);
        }
        assert_eq!(
            decode_content_encoding(&body, "gzip, gzip, gzip, gzip", CAP).unwrap(),
            b"deep"
        );
        let mut chain = DecodeChain::new("gzip, gzip, gzip, gzip", CAP)
            .unwrap()
            .unwrap();
        let mut out = Vec::new();
        for piece in body.chunks(11) {
            chain.feed(piece, &mut out).unwrap();
        }
        chain.finish(&mut out).unwrap();
        assert_eq!(out, b"deep");
    }

    #[test]
    fn streaming_output_cap_enforced_incrementally() {
        assert_eq!(
            feed_all(Coding::Gzip, fixture!("hello.gz"), 3, 16).unwrap_err(),
            DecodeError::OutputTooLarge
        );
        assert_eq!(
            feed_all(Coding::Zlib, fixture!("hello.zlib"), 3, 16).unwrap_err(),
            DecodeError::OutputTooLarge
        );
        // The error also surfaces mid-feed, not only at finish.
        let mut d = Decoder::new(Coding::Gzip, 16);
        let mut out = Vec::new();
        let gz = fixture!("hello.gz");
        let mut saw_err = false;
        for piece in gz.chunks(9) {
            if d.feed(piece, &mut out).is_err() {
                saw_err = true;
                break;
            }
        }
        assert!(saw_err, "cap must trip while feeding");
    }

    #[test]
    fn streaming_decoder_rejects_use_after_finish() {
        let mut d = Decoder::new(Coding::Raw, CAP);
        let mut out = Vec::new();
        d.feed(fixture!("hello.raw"), &mut out).unwrap();
        d.finish(&[], &mut out).unwrap();
        assert_eq!(out, PLAIN);
        // finish is idempotent, but feeding again is an error.
        let mut out2 = Vec::new();
        d.finish(&[], &mut out2).unwrap();
        assert!(d.feed(b"x", &mut out2).is_err());
    }

    #[test]
    fn streaming_emits_output_before_input_ends() {
        // Incremental means decoded bytes surface before the whole
        // input arrives: after the first half, output is non-empty.
        let gz = fixture!("binary.gz");
        let mut d = Decoder::new(Coding::Gzip, 2 * 1024 * 1024);
        let mut out = Vec::new();
        d.feed(&gz[..gz.len() / 2], &mut out).unwrap();
        assert!(!out.is_empty(), "no incremental output produced");
        d.feed(&gz[gz.len() / 2..], &mut out).unwrap();
        d.finish(&[], &mut out).unwrap();
        assert_eq!(out, fixture!("binary.bin"));
    }

    // ------------------------------------------------------------------
    // Wave 50: Huffman fast path (9-bit LUT + slow bit-walk)
    // ------------------------------------------------------------------

    /// LSB-first bit writer for hand-crafted deflate streams. Huffman
    /// codes go out MSB-first ([`push_code`](Self::push_code)); header
    /// fields and extra bits go out LSB-first
    /// ([`push_bits`](Self::push_bits)), matching RFC 1951 §3.1.1.
    struct BitWriter {
        bytes: Vec<u8>,
        cur: u32,
        n: u32,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                cur: 0,
                n: 0,
            }
        }

        /// Append the low `n` bits of `v`, LSB-first.
        fn push_bits(&mut self, v: u32, n: u32) {
            self.cur |= v << self.n;
            self.n += n;
            while self.n >= 8 {
                self.bytes.push(self.cur as u8);
                self.cur >>= 8;
                self.n -= 8;
            }
        }

        /// Append a canonical Huffman code, MSB-first.
        fn push_code(&mut self, code: u32, len: u32) {
            for i in (0..len).rev() {
                self.push_bits((code >> i) & 1, 1);
            }
        }

        fn finish(mut self) -> Vec<u8> {
            if self.n > 0 {
                self.bytes.push(self.cur as u8);
            }
            self.bytes
        }
    }

    /// A raw deflate stream whose single dynamic block uses 12-bit
    /// literal codes (longer than the 9-bit LUT, so every literal takes
    /// the slow bit-walk) plus a lone 1-bit distance code. Decodes to
    /// `[0]`.
    ///
    /// Layout: HLIT=257, HDIST=1. Code-length code: sym18 len 1, sym1
    /// and sym12 len 2 (canonical: 18 -> `0`, 1 -> `10`, 12 -> `11`).
    /// Lengths: litlen[0] = litlen[1] = litlen[256] = 12 (canonical
    /// codes 0, 1, 2), everything else 0, dist[0] = 1. Body: literal
    /// `0x00`, then end-of-block.
    fn long_code_stream() -> Vec<u8> {
        let mut w = BitWriter::new();
        w.push_bits(1, 1); // BFINAL
        w.push_bits(2, 2); // BTYPE = dynamic
        w.push_bits(0, 5); // HLIT = 257
        w.push_bits(0, 5); // HDIST = 1
        w.push_bits(14, 4); // HCLEN = 18
        // CL lengths in CL_ORDER: index 2 (sym18) = 1, index 12 (sym12)
        // = 2, index 17 (sym1) = 2, the rest 0.
        for &v in &[0u32, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2] {
            w.push_bits(v, 3);
        }
        w.push_code(0b11, 2); // litlen[0] = 12
        w.push_code(0b11, 2); // litlen[1] = 12
        w.push_code(0, 1);
        w.push_bits(127, 7); // sym18: 138 zeros (litlen[2..140])
        w.push_code(0, 1);
        w.push_bits(105, 7); // sym18: 116 zeros (litlen[140..256])
        w.push_code(0b11, 2); // litlen[256] = 12
        w.push_code(0b10, 2); // dist[0] = 1
        w.push_code(0, 12); // literal 0x00
        w.push_code(2, 12); // end of block
        w.finish()
    }

    #[test]
    fn dynamic_block_long_codes_roundtrip() {
        let data = long_code_stream();
        assert_eq!(inflate_raw(&data, CAP).unwrap(), [0u8]);
        // The streaming path (decode_resume slow walk) agrees at any
        // chunk size, including splits mid-code.
        for &chunk in &[1usize, 2, 3, 5, 64] {
            assert_eq!(
                feed_all(Coding::Raw, &data, chunk, CAP).unwrap(),
                [0u8],
                "chunk={chunk}"
            );
        }
    }

    /// Tiny deterministic PRNG (xorshift64) for the differential test.
    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Differential check: for random valid length assignments (many
    /// with codes longer than the 9-bit LUT), encode symbols with an
    /// independently computed canonical code and decode them back. The
    /// LUT fast path and the slow walk must both reproduce the input.
    #[test]
    fn huffman_lut_and_slow_path_agree_with_canonical_codes() {
        let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
        for trial in 0..200 {
            let n = 2 + rng.below(60);
            let mut lengths = vec![0u8; n];
            for l in lengths.iter_mut() {
                *l = 1 + rng.below(13) as u8;
            }
            // Repair over-subscription by lengthening codes until the
            // Kraft sum fits (incomplete codes are legal).
            loop {
                let kraft: u32 = lengths.iter().map(|&l| 1u32 << (15 - l)).sum();
                if kraft <= 1 << 15 {
                    break;
                }
                let i = rng.below(n);
                if lengths[i] < 15 {
                    lengths[i] += 1;
                }
            }
            let huff = build_huffman(&lengths).unwrap();
            // Independent canonical-code computation.
            let mut counts = [0u32; 16];
            for &l in &lengths {
                counts[l as usize] += 1;
            }
            counts[0] = 0;
            let mut code = 0u32;
            let mut next = [0u32; 16];
            for len in 1..16 {
                code = (code + counts[len - 1]) << 1;
                next[len] = code;
            }
            let mut codes = vec![0u32; n];
            for (sym, &l) in lengths.iter().enumerate() {
                codes[sym] = next[l as usize];
                next[l as usize] += 1;
            }
            let want: Vec<u16> = (0..64).map(|_| rng.below(n) as u16).collect();
            let mut w = BitWriter::new();
            for &s in &want {
                w.push_code(codes[s as usize], lengths[s as usize] as u32);
            }
            let bytes = w.finish();
            let mut br = BitReader::new(&bytes);
            for &s in &want {
                assert_eq!(huff.decode(&mut br).unwrap(), s, "trial {trial}");
            }
        }
    }

    #[test]
    fn degenerate_huffman_codes_rejected() {
        // Empty, all-zero, over-subscribed, and out-of-range length sets
        // are all invalid; none may panic.
        assert_eq!(build_huffman(&[]).err(), Some(DecodeError::InvalidData));
        assert_eq!(
            build_huffman(&[0, 0, 0]).err(),
            Some(DecodeError::InvalidData)
        );
        assert_eq!(
            build_huffman(&[1, 1, 1]).err(),
            Some(DecodeError::InvalidData)
        );
        assert_eq!(build_huffman(&[16]).err(), Some(DecodeError::InvalidData));
        // Decoding from empty input reports Truncated, never panics.
        let h = build_huffman(&[1, 1]).unwrap();
        let mut br = BitReader::new(&[]);
        assert_eq!(h.decode(&mut br).unwrap_err(), DecodeError::Truncated);
        // A lone 1-bit code (incomplete) decodes its one symbol...
        let h = build_huffman(&[1]).unwrap();
        let mut br = BitReader::new(&[0]);
        assert_eq!(h.decode(&mut br).unwrap(), 0);
        // ...and its unused slot is invalid data, not a hang.
        let mut br = BitReader::new(&[0xff]);
        assert_eq!(h.decode(&mut br).unwrap_err(), DecodeError::InvalidData);
    }

    #[test]
    fn truncated_inputs_error_not_panic() {
        // Every proper prefix of a valid long-code stream must error
        // cleanly (Truncated or InvalidData), never panic or hang.
        let data = long_code_stream();
        for cut in 0..data.len() {
            assert!(inflate_raw(&data[..cut], CAP).is_err(), "cut={cut}");
        }
        // Same for a gzip member, one-shot.
        let gz = fixture!("hello.gz");
        for cut in 0..gz.len() {
            assert!(gunzip(&gz[..cut], CAP).is_err(), "cut={cut}");
        }
    }
}
