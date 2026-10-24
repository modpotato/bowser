//! HPACK header compression (RFC 7541) for HTTP/2.
//!
//! Standard-library only. The Huffman table below is the code table from
//! RFC 7541 Appendix B; it is validated against the Appendix C wire
//! vectors by the `huffman_rfc_vectors` test.
//!
//! The [`Encoder`] never inserts into the dynamic table: every field is
//! emitted as indexed (static table) or literal-without-indexing. That is
//! always legal HPACK and keeps per-connection encoder state trivial; a
//! single-request-per-connection client gains nothing from dynamic
//! indexing. The [`Decoder`] implements the full dynamic table with
//! eviction, since the peer may index freely.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Every way HPACK decoding can fail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HpackError {
    /// Truncated integer, string, or header block.
    Truncated,
    /// An integer continuation would overflow 32 bits.
    BadInteger,
    /// A Huffman-coded string is not a valid Huffman sequence (bad code,
    /// an EOS symbol on the wire, or non-`1` padding).
    BadHuffman,
    /// An indexed representation names an entry that does not exist.
    BadIndex(u32),
    /// A dynamic-table-size update exceeds the agreed maximum.
    BadTableSize(u32),
    /// A dynamic-table-size update did not open the header block.
    MisplacedSizeUpdate,
    /// A header field name is empty.
    EmptyName,
    /// A header field name contains an uppercase ASCII letter. Field
    /// names are lowercase on the wire (RFC 9113 §8.2.2).
    UppercaseName,
    /// The header list exceeds the configured field limit.
    TooManyFields,
    /// The decoded block would grow the slab past [`MAX_HEADER_BLOCK`].
    BlockTooLarge,
}

impl std::fmt::Display for HpackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => f.write_str("truncated HPACK data"),
            Self::BadInteger => f.write_str("HPACK integer overflow"),
            Self::BadHuffman => f.write_str("invalid Huffman coding"),
            Self::BadIndex(i) => write!(f, "HPACK index out of range: {i}"),
            Self::BadTableSize(s) => write!(f, "HPACK table size update too large: {s}"),
            Self::MisplacedSizeUpdate => f.write_str("HPACK table size update not at block start"),
            Self::EmptyName => f.write_str("empty HPACK header name"),
            Self::UppercaseName => f.write_str("uppercase HPACK header name"),
            Self::TooManyFields => f.write_str("too many HPACK header fields"),
            Self::BlockTooLarge => f.write_str("HPACK header block exceeds 64 KiB decoded"),
        }
    }
}

// ---------------------------------------------------------------------------
// Huffman coding (RFC 7541 Appendix B)
// ---------------------------------------------------------------------------

/// `(bit length, code bits)` per symbol: symbols `0..=255`, then the EOS
/// padding symbol `256` (30 one-bits). RFC 7541 Appendix B.
const HUFFMAN: [(u8, u32); 257] = [
    (13, 0x00001ff8),
    (23, 0x007fffd8),
    (28, 0x0fffffe2),
    (28, 0x0fffffe3), // 0..3
    (28, 0x0fffffe4),
    (28, 0x0fffffe5),
    (28, 0x0fffffe6),
    (28, 0x0fffffe7), // 4..7
    (28, 0x0fffffe8),
    (24, 0x00ffffea),
    (30, 0x3ffffffc),
    (28, 0x0fffffe9), // 8..11
    (28, 0x0fffffea),
    (30, 0x3ffffffd),
    (28, 0x0fffffeb),
    (28, 0x0fffffec), // 12..15
    (28, 0x0fffffed),
    (28, 0x0fffffee),
    (28, 0x0fffffef),
    (28, 0x0ffffff0), // 16..19
    (28, 0x0ffffff1),
    (28, 0x0ffffff2),
    (30, 0x3ffffffe),
    (28, 0x0ffffff3), // 20..23
    (28, 0x0ffffff4),
    (28, 0x0ffffff5),
    (28, 0x0ffffff6),
    (28, 0x0ffffff7), // 24..27
    (28, 0x0ffffff8),
    (28, 0x0ffffff9),
    (28, 0x0ffffffa),
    (28, 0x0ffffffb), // 28..31
    (6, 0x00000014),
    (10, 0x000003f8),
    (10, 0x000003f9),
    (12, 0x00000ffa), // 32..35
    (13, 0x00001ff9),
    (6, 0x00000015),
    (8, 0x000000f8),
    (11, 0x000007fa), // 36..39
    (10, 0x000003fa),
    (10, 0x000003fb),
    (8, 0x000000f9),
    (11, 0x000007fb), // 40..43
    (8, 0x000000fa),
    (6, 0x00000016),
    (6, 0x00000017),
    (6, 0x00000018), // 44..47
    (5, 0x00000000),
    (5, 0x00000001),
    (5, 0x00000002),
    (6, 0x00000019), // 48..51
    (6, 0x0000001a),
    (6, 0x0000001b),
    (6, 0x0000001c),
    (6, 0x0000001d), // 52..55
    (6, 0x0000001e),
    (6, 0x0000001f),
    (7, 0x0000005c),
    (8, 0x000000fb), // 56..59
    (15, 0x00007ffc),
    (6, 0x00000020),
    (12, 0x00000ffb),
    (10, 0x000003fc), // 60..63
    (13, 0x00001ffa),
    (6, 0x00000021),
    (7, 0x0000005d),
    (7, 0x0000005e), // 64..67
    (7, 0x0000005f),
    (7, 0x00000060),
    (7, 0x00000061),
    (7, 0x00000062), // 68..71
    (7, 0x00000063),
    (7, 0x00000064),
    (7, 0x00000065),
    (7, 0x00000066), // 72..75
    (7, 0x00000067),
    (7, 0x00000068),
    (7, 0x00000069),
    (7, 0x0000006a), // 76..79
    (7, 0x0000006b),
    (7, 0x0000006c),
    (7, 0x0000006d),
    (7, 0x0000006e), // 80..83
    (7, 0x0000006f),
    (7, 0x00000070),
    (7, 0x00000071),
    (7, 0x00000072), // 84..87
    (8, 0x000000fc),
    (7, 0x00000073),
    (8, 0x000000fd),
    (13, 0x00001ffb), // 88..91
    (19, 0x0007fff0),
    (13, 0x00001ffc),
    (14, 0x00003ffc),
    (6, 0x00000022), // 92..95
    (15, 0x00007ffd),
    (5, 0x00000003),
    (6, 0x00000023),
    (5, 0x00000004), // 96..99
    (6, 0x00000024),
    (5, 0x00000005),
    (6, 0x00000025),
    (6, 0x00000026), // 100..103
    (6, 0x00000027),
    (5, 0x00000006),
    (7, 0x00000074),
    (7, 0x00000075), // 104..107
    (6, 0x00000028),
    (6, 0x00000029),
    (6, 0x0000002a),
    (5, 0x00000007), // 108..111
    (6, 0x0000002b),
    (7, 0x00000076),
    (6, 0x0000002c),
    (5, 0x00000008), // 112..115
    (5, 0x00000009),
    (6, 0x0000002d),
    (7, 0x00000077),
    (7, 0x00000078), // 116..119
    (7, 0x00000079),
    (7, 0x0000007a),
    (7, 0x0000007b),
    (15, 0x00007ffe), // 120..123
    (11, 0x000007fc),
    (14, 0x00003ffd),
    (13, 0x00001ffd),
    (28, 0x0ffffffc), // 124..127
    (20, 0x000fffe6),
    (22, 0x003fffd2),
    (20, 0x000fffe7),
    (20, 0x000fffe8), // 128..131
    (22, 0x003fffd3),
    (22, 0x003fffd4),
    (22, 0x003fffd5),
    (23, 0x007fffd9), // 132..135
    (22, 0x003fffd6),
    (23, 0x007fffda),
    (23, 0x007fffdb),
    (23, 0x007fffdc), // 136..139
    (23, 0x007fffdd),
    (23, 0x007fffde),
    (24, 0x00ffffeb),
    (23, 0x007fffdf), // 140..143
    (24, 0x00ffffec),
    (24, 0x00ffffed),
    (22, 0x003fffd7),
    (23, 0x007fffe0), // 144..147
    (24, 0x00ffffee),
    (23, 0x007fffe1),
    (23, 0x007fffe2),
    (23, 0x007fffe3), // 148..151
    (23, 0x007fffe4),
    (21, 0x001fffdc),
    (22, 0x003fffd8),
    (23, 0x007fffe5), // 152..155
    (22, 0x003fffd9),
    (23, 0x007fffe6),
    (23, 0x007fffe7),
    (24, 0x00ffffef), // 156..159
    (22, 0x003fffda),
    (21, 0x001fffdd),
    (20, 0x000fffe9),
    (22, 0x003fffdb), // 160..163
    (22, 0x003fffdc),
    (23, 0x007fffe8),
    (23, 0x007fffe9),
    (21, 0x001fffde), // 164..167
    (23, 0x007fffea),
    (22, 0x003fffdd),
    (22, 0x003fffde),
    (24, 0x00fffff0), // 168..171
    (21, 0x001fffdf),
    (22, 0x003fffdf),
    (23, 0x007fffeb),
    (23, 0x007fffec), // 172..175
    (21, 0x001fffe0),
    (21, 0x001fffe1),
    (22, 0x003fffe0),
    (21, 0x001fffe2), // 176..179
    (23, 0x007fffed),
    (22, 0x003fffe1),
    (23, 0x007fffee),
    (23, 0x007fffef), // 180..183
    (20, 0x000fffea),
    (22, 0x003fffe2),
    (22, 0x003fffe3),
    (22, 0x003fffe4), // 184..187
    (23, 0x007ffff0),
    (22, 0x003fffe5),
    (22, 0x003fffe6),
    (23, 0x007ffff1), // 188..191
    (26, 0x03ffffe0),
    (26, 0x03ffffe1),
    (20, 0x000fffeb),
    (19, 0x0007fff1), // 192..195
    (22, 0x003fffe7),
    (23, 0x007ffff2),
    (22, 0x003fffe8),
    (25, 0x01ffffec), // 196..199
    (26, 0x03ffffe2),
    (26, 0x03ffffe3),
    (26, 0x03ffffe4),
    (27, 0x07ffffde), // 200..203
    (27, 0x07ffffdf),
    (26, 0x03ffffe5),
    (24, 0x00fffff1),
    (25, 0x01ffffed), // 204..207
    (19, 0x0007fff2),
    (21, 0x001fffe3),
    (26, 0x03ffffe6),
    (27, 0x07ffffe0), // 208..211
    (27, 0x07ffffe1),
    (26, 0x03ffffe7),
    (27, 0x07ffffe2),
    (24, 0x00fffff2), // 212..215
    (21, 0x001fffe4),
    (21, 0x001fffe5),
    (26, 0x03ffffe8),
    (26, 0x03ffffe9), // 216..219
    (28, 0x0ffffffd),
    (27, 0x07ffffe3),
    (27, 0x07ffffe4),
    (27, 0x07ffffe5), // 220..223
    (20, 0x000fffec),
    (24, 0x00fffff3),
    (20, 0x000fffed),
    (21, 0x001fffe6), // 224..227
    (22, 0x003fffe9),
    (21, 0x001fffe7),
    (21, 0x001fffe8),
    (23, 0x007ffff3), // 228..231
    (22, 0x003fffea),
    (22, 0x003fffeb),
    (25, 0x01ffffee),
    (25, 0x01ffffef), // 232..235
    (24, 0x00fffff4),
    (24, 0x00fffff5),
    (26, 0x03ffffea),
    (23, 0x007ffff4), // 236..239
    (26, 0x03ffffeb),
    (27, 0x07ffffe6),
    (26, 0x03ffffec),
    (26, 0x03ffffed), // 240..243
    (27, 0x07ffffe7),
    (27, 0x07ffffe8),
    (27, 0x07ffffe9),
    (27, 0x07ffffea), // 244..247
    (27, 0x07ffffeb),
    (28, 0x0ffffffe),
    (27, 0x07ffffec),
    (27, 0x07ffffed), // 248..251
    (27, 0x07ffffee),
    (27, 0x07ffffef),
    (27, 0x07fffff0),
    (26, 0x03ffffee), // 252..255
    (30, 0x3fffffff), // 256..256
];

/// Encoded length in bytes of `data` under Huffman coding.
pub(crate) fn huffman_encoded_len(data: &[u8]) -> usize {
    let mut bits: u64 = 0;
    for &b in data {
        bits += HUFFMAN[b as usize].0 as u64;
    }
    (bits as usize).div_ceil(8)
}

/// Append the Huffman coding of `data` to `out`, padding the final byte
/// with the EOS prefix (all ones) per RFC 7541 §5.2.
pub(crate) fn huffman_encode(data: &[u8], out: &mut Vec<u8>) {
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    for &b in data {
        let (len, code) = HUFFMAN[b as usize];
        acc = (acc << len) | code as u64;
        nbits += len as u32;
        while nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
            // Drop the consumed bits: without this the accumulator
            // keeps every code ever shifted in and overflows u64 on
            // long strings. `nbits` stays below 38 here, so the shift
            // is always in range.
            acc &= (1u64 << nbits) - 1;
        }
    }
    if nbits > 0 {
        let pad = 8 - nbits;
        out.push(((acc << pad) | ((1 << pad) - 1)) as u8);
    }
}

struct HuffNode {
    child: [u32; 2],
    /// `u16::MAX` for internal nodes, the symbol otherwise.
    symbol: u16,
}

/// Binary decode tree built once from [`HUFFMAN`]. The table is a
/// complete prefix code (Kraft sum is exactly 1), so every bit walk from
/// the root ends at exactly one leaf.
fn huffman_tree() -> &'static Vec<HuffNode> {
    static TREE: OnceLock<Vec<HuffNode>> = OnceLock::new();
    TREE.get_or_init(|| {
        let mut nodes = vec![HuffNode {
            child: [u32::MAX; 2],
            symbol: u16::MAX,
        }];
        for (sym, &(len, code)) in HUFFMAN.iter().enumerate() {
            let mut cur = 0u32;
            for i in (0..len).rev() {
                let bit = ((code >> i) & 1) as usize;
                let next = nodes[cur as usize].child[bit];
                let next = if next == u32::MAX {
                    let id = nodes.len() as u32;
                    nodes.push(HuffNode {
                        child: [u32::MAX; 2],
                        symbol: u16::MAX,
                    });
                    nodes[cur as usize].child[bit] = id;
                    id
                } else {
                    next
                };
                cur = next;
            }
            // Prefix-free: no code is a prefix of another, so the landing
            // node is always fresh.
            debug_assert_eq!(nodes[cur as usize].symbol, u16::MAX);
            nodes[cur as usize].symbol = sym as u16;
        }
        nodes
    })
}

/// Decode a Huffman-coded string, appending the bytes to `out`. The EOS
/// symbol on the wire is rejected, and trailing padding must be all
/// ones (RFC 7541 §5.2).
pub(crate) fn huffman_decode(data: &[u8], out: &mut Vec<u8>) -> Result<(), HpackError> {
    let tree = huffman_tree();
    let mut node = 0u32;
    // Bits consumed since the last completed symbol; the padding check
    // below only ever inspects the final byte.
    let mut tail_bits: u32 = 0;
    for &byte in data {
        for i in (0..8).rev() {
            let bit = ((byte >> i) & 1) as usize;
            let next = tree[node as usize].child[bit];
            if next == u32::MAX {
                return Err(HpackError::BadHuffman);
            }
            node = next;
            tail_bits += 1;
            let sym = tree[node as usize].symbol;
            if sym != u16::MAX {
                if sym == 256 {
                    // EOS must only pad, never appear as a symbol.
                    return Err(HpackError::BadHuffman);
                }
                out.push(sym as u8);
                node = 0;
                tail_bits = 0;
            }
        }
    }
    if node != 0 {
        // Ended mid-code: the leftover must be a short all-ones padding.
        // (`tail_bits < 8` always holds here: the input is whole bytes and
        // a symbol completed at or after the last byte boundary... except
        // when 8+ bits passed without completing a symbol, which is a
        // truncated code and fails the `< 8` check.)
        if tail_bits == 0 || tail_bits >= 8 {
            return Err(HpackError::BadHuffman);
        }
        let mask = ((1u16 << tail_bits) - 1) as u8;
        if data[data.len() - 1] & mask != mask {
            return Err(HpackError::BadHuffman);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Integer and string primitives (RFC 7541 §5.1, §5.2)
// ---------------------------------------------------------------------------

/// Encode `value` with an N-bit prefix; `first` carries the already-set
/// high bits of the first byte (representation marker).
pub(crate) fn encode_int(mut value: u32, prefix: u8, first: u8, out: &mut Vec<u8>) {
    debug_assert!((1..=8).contains(&prefix));
    let max = (1u32 << prefix) - 1;
    if value < max {
        out.push(first | value as u8);
        return;
    }
    out.push(first | max as u8);
    value -= max;
    while value >= 128 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Decode an N-bit-prefixed integer; returns the value and bytes consumed.
pub(crate) fn decode_int(data: &[u8], prefix: u8) -> Result<(u32, usize), HpackError> {
    debug_assert!((1..=8).contains(&prefix));
    if data.is_empty() {
        return Err(HpackError::Truncated);
    }
    let max = (1u32 << prefix) - 1;
    let mut value = (data[0] & max as u8) as u32;
    if value < max {
        return Ok((value, 1));
    }
    let mut shift = 0u32;
    let mut pos = 1;
    loop {
        let b = *data.get(pos).ok_or(HpackError::Truncated)?;
        pos += 1;
        // The accumulator must stay within 32 bits: at most five
        // continuation bytes may follow the prefix.
        if shift >= 32 || (shift == 28 && b & 0x7f > 0x0f) {
            return Err(HpackError::BadInteger);
        }
        value += ((b & 0x7f) as u32) << shift;
        shift += 7;
        if b & 0x80 == 0 {
            break;
        }
    }
    Ok((value, pos))
}

/// Encode a string, using Huffman coding when it shortens the value.
fn encode_string(data: &[u8], out: &mut Vec<u8>) {
    if huffman_encoded_len(data) < data.len() {
        let mut h = Vec::with_capacity(huffman_encoded_len(data));
        huffman_encode(data, &mut h);
        encode_int(h.len() as u32, 7, 0x80, out);
        out.extend_from_slice(&h);
    } else {
        encode_int(data.len() as u32, 7, 0x00, out);
        out.extend_from_slice(data);
    }
}

/// Maximum decoded header-block size accepted into the slab. Matches the
/// HTTP/2 layer's `MAX_HEADER_BLOCK` (64 KiB); a block that would decode
/// past it fails with [`HpackError::BlockTooLarge`] instead of growing
/// the slab without bound. `u32` field offsets are ample at this cap.
pub(crate) const MAX_HEADER_BLOCK: usize = 64 * 1024;

/// Decode a string, appending its bytes to `out`; returns the wire bytes
/// consumed. Huffman strings decode in place via [`huffman_decode`]. The
/// slab is rolled back and [`HpackError::BlockTooLarge`] returned if the
/// append would grow past [`MAX_HEADER_BLOCK`].
fn decode_string_into(data: &[u8], out: &mut Vec<u8>) -> Result<usize, HpackError> {
    if data.is_empty() {
        return Err(HpackError::Truncated);
    }
    let huffman = data[0] & 0x80 != 0;
    let (len, mut pos) = decode_int(data, 7)?;
    let len = len as usize;
    let end = pos.checked_add(len).ok_or(HpackError::Truncated)?;
    if data.len() < end {
        return Err(HpackError::Truncated);
    }
    let raw = &data[pos..end];
    pos = end;
    let base = out.len();
    if huffman {
        if let Err(e) = huffman_decode(raw, out) {
            out.truncate(base);
            return Err(e);
        }
    } else if out.len() + raw.len() > MAX_HEADER_BLOCK {
        return Err(HpackError::BlockTooLarge);
    } else {
        out.extend_from_slice(raw);
    }
    if out.len() > MAX_HEADER_BLOCK {
        // Huffman output can exceed its wire length.
        out.truncate(base);
        return Err(HpackError::BlockTooLarge);
    }
    Ok(pos)
}

/// memcpy `data` into the slab, enforcing the block cap.
fn append_capped(out: &mut Vec<u8>, data: &[u8]) -> Result<(), HpackError> {
    if out.len() + data.len() > MAX_HEADER_BLOCK {
        return Err(HpackError::BlockTooLarge);
    }
    out.extend_from_slice(data);
    Ok(())
}

// ---------------------------------------------------------------------------
// Static table (RFC 7541 Appendix A)
// ---------------------------------------------------------------------------

/// The 61-entry static table, 1-based on the wire.
const STATIC_TABLE: [(&str, &str); 61] = [
    (":authority", ""),
    (":method", "GET"),
    (":method", "POST"),
    (":path", "/"),
    (":path", "/index.html"),
    (":scheme", "http"),
    (":scheme", "https"),
    (":status", "200"),
    (":status", "204"),
    (":status", "206"),
    (":status", "304"),
    (":status", "400"),
    (":status", "404"),
    (":status", "500"),
    ("accept-charset", ""),
    ("accept-encoding", "gzip, deflate"),
    ("accept-language", ""),
    ("accept-ranges", ""),
    ("accept", ""),
    ("access-control-allow-origin", ""),
    ("age", ""),
    ("allow", ""),
    ("authorization", ""),
    ("cache-control", ""),
    ("content-disposition", ""),
    ("content-encoding", ""),
    ("content-language", ""),
    ("content-length", ""),
    ("content-location", ""),
    ("content-range", ""),
    ("content-type", ""),
    ("cookie", ""),
    ("date", ""),
    ("etag", ""),
    ("expect", ""),
    ("expires", ""),
    ("from", ""),
    ("host", ""),
    ("if-match", ""),
    ("if-modified-since", ""),
    ("if-none-match", ""),
    ("if-range", ""),
    ("if-unmodified-since", ""),
    ("last-modified", ""),
    ("link", ""),
    ("location", ""),
    ("max-forwards", ""),
    ("proxy-authenticate", ""),
    ("proxy-authorization", ""),
    ("range", ""),
    ("referer", ""),
    ("refresh", ""),
    ("retry-after", ""),
    ("server", ""),
    ("set-cookie", ""),
    ("strict-transport-security", ""),
    ("transfer-encoding", ""),
    ("user-agent", ""),
    ("vary", ""),
    ("via", ""),
    ("www-authenticate", ""),
];

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// HPACK encoder. Emits indexed representations for static-table hits
/// and literal-without-indexing otherwise; it never touches the dynamic
/// table.
#[derive(Default)]
pub struct Encoder;

impl Encoder {
    pub fn new() -> Self {
        Encoder
    }

    /// Encode a header list. Pseudo-headers (`:`-prefixed) must precede
    /// regular fields; names and values are raw bytes.
    pub fn encode(&mut self, headers: &[(&[u8], &[u8])], out: &mut Vec<u8>) {
        for &(name, value) in headers {
            if let Some(i) = STATIC_TABLE
                .iter()
                .position(|&(n, v)| n.as_bytes() == name && v.as_bytes() == value)
            {
                encode_int(i as u32 + 1, 7, 0x80, out);
                continue;
            }
            if let Some(i) = STATIC_TABLE.iter().position(|&(n, _)| n.as_bytes() == name) {
                // Literal without indexing, indexed name.
                encode_int(i as u32 + 1, 4, 0x00, out);
            } else {
                // Literal without indexing, new name.
                encode_int(0, 4, 0x00, out);
                encode_string(name, out);
            }
            encode_string(value, out);
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

struct DynEntry {
    name: Vec<u8>,
    value: Vec<u8>,
}

/// One header field as offsets into [`DecodedHeaders::bytes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    /// Byte range of the field name in [`DecodedHeaders::bytes`].
    pub name: Range<u32>,
    /// Byte range of the field value in [`DecodedHeaders::bytes`].
    pub value: Range<u32>,
}

/// A decoded header block: every name and value concatenated into one
/// bump-allocated byte slab, with one [`Field`] of `u32` offsets per
/// header in wire order.
///
/// Decoding a block costs one slab plus one small [`Field`] per header —
/// no per-string allocation — and indexed static/dynamic references are
/// memcpy'd into the slab instead of `.to_vec()`'d. Only
/// incremental-indexing inserts copy out of the slab into owned
/// dynamic-table entries (the table outlives the slab contents).
///
/// Fully owned and `Send`: safe to move across the `mpsc` event
/// boundary. Reuse one instance across blocks; [`Decoder::decode_block`]
/// resets it on entry.
#[derive(Debug, Clone, Default)]
pub struct DecodedHeaders {
    /// All field names and values, concatenated back-to-back.
    pub bytes: Vec<u8>,
    /// One entry per decoded field, in wire order.
    pub fields: Vec<Field>,
}

impl DecodedHeaders {
    /// Empty block; the slab keeps its allocation across [`clear`][Self::clear].
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset for the next block, keeping the slab allocation.
    pub fn clear(&mut self) {
        self.bytes.clear();
        self.fields.clear();
    }

    /// Number of decoded fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether any field was decoded.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Name and value byte slices for field `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of range.
    pub fn field(&self, i: usize) -> (&[u8], &[u8]) {
        let f = &self.fields[i];
        (
            &self.bytes[f.name.start as usize..f.name.end as usize],
            &self.bytes[f.value.start as usize..f.value.end as usize],
        )
    }
}

/// HPACK decoder with the dynamic table (RFC 7541 §2.3, §4).
pub struct Decoder {
    /// Front is the lowest dynamic index (62 on the wire).
    table: VecDeque<DynEntry>,
    /// Maximum table size we advertise (our SETTINGS_HEADER_TABLE_SIZE).
    max_size: usize,
    /// Current table size: sum of `32 + name.len() + value.len()`.
    size: usize,
}

impl Decoder {
    pub fn new(max_size: usize) -> Self {
        Decoder {
            table: VecDeque::new(),
            max_size,
            size: 0,
        }
    }

    /// Apply a peer dynamic-table-size update: it may only shrink.
    fn set_max_size(&mut self, max: usize) -> Result<(), HpackError> {
        if max > self.max_size {
            return Err(HpackError::BadTableSize(max as u32));
        }
        self.max_size = max;
        self.evict();
        Ok(())
    }

    fn evict(&mut self) {
        while self.size > self.max_size {
            if let Some(e) = self.table.pop_back() {
                self.size -= 32 + e.name.len() + e.value.len();
            } else {
                break;
            }
        }
    }

    fn insert(&mut self, name: Vec<u8>, value: Vec<u8>) {
        let entry_size = 32 + name.len() + value.len();
        if entry_size > self.max_size {
            // An entry larger than the table drains it (RFC 7541 §4.2).
            self.table.clear();
            self.size = 0;
            return;
        }
        self.size += entry_size;
        self.table.push_front(DynEntry { name, value });
        self.evict();
    }

    /// Resolve a 1-based index into static or dynamic table.
    fn get(&self, index: u32) -> Result<(&[u8], &[u8]), HpackError> {
        if index == 0 {
            return Err(HpackError::BadIndex(0));
        }
        if index <= STATIC_TABLE.len() as u32 {
            let (n, v) = STATIC_TABLE[index as usize - 1];
            return Ok((n.as_bytes(), v.as_bytes()));
        }
        self.table
            .get(index as usize - STATIC_TABLE.len() - 1)
            .map(|e| (e.name.as_slice(), e.value.as_slice()))
            .ok_or(HpackError::BadIndex(index))
    }

    /// Decode one header block into the bump slab `out`, resetting it
    /// first. Strings decode directly into [`DecodedHeaders::bytes`];
    /// indexed static/dynamic references are memcpy'd in; only
    /// incremental-indexing inserts copy name+value out of the slab into
    /// owned dynamic-table entries. At most `max_fields` fields are
    /// accepted, and the slab never grows past [`MAX_HEADER_BLOCK`]
    /// bytes. All wave-9 hardening applies: [`HpackError::TooManyFields`],
    /// [`HpackError::BadHuffman`], [`HpackError::BadInteger`], the
    /// size-update rules, and dynamic-table eviction/size accounting.
    pub fn decode_block(
        &mut self,
        block: &[u8],
        max_fields: usize,
        out: &mut DecodedHeaders,
    ) -> Result<(), HpackError> {
        out.clear();
        let mut pos = 0;
        // A size update must open the block (RFC 7541 §4.2).
        let mut at_start = true;
        while pos < block.len() {
            if out.fields.len() >= max_fields {
                return Err(HpackError::TooManyFields);
            }
            let n = self.decode_field(
                &block[pos..],
                &mut out.bytes,
                &mut out.fields,
                &mut at_start,
            )?;
            pos += n;
        }
        Ok(())
    }

    /// Decode one field representation at the head of `data`: append the
    /// name and value to `slab`, push the [`Field`], and return the wire
    /// bytes consumed. On error the slab is rolled back to its entry
    /// length and the dynamic table is left untouched (validation always
    /// precedes insertion).
    fn decode_field(
        &mut self,
        data: &[u8],
        slab: &mut Vec<u8>,
        fields: &mut Vec<Field>,
        at_start: &mut bool,
    ) -> Result<usize, HpackError> {
        let base = slab.len();
        let fields_base = fields.len();
        let r = self.decode_field_inner(data, slab, fields, at_start);
        if r.is_err() {
            slab.truncate(base);
            fields.truncate(fields_base);
        }
        r
    }

    /// The representation dispatch behind [`Decoder::decode_field`].
    /// `data` is non-empty (the caller only invokes it with `pos < len`).
    fn decode_field_inner(
        &mut self,
        data: &[u8],
        slab: &mut Vec<u8>,
        fields: &mut Vec<Field>,
        at_start: &mut bool,
    ) -> Result<usize, HpackError> {
        let mut pos = 0;
        let b = data[0];
        if b & 0x80 != 0 {
            // Indexed header field.
            let (idx, n) = decode_int(data, 7)?;
            pos += n;
            let (name, value) = self.get(idx)?;
            check_name(name)?;
            let name_start = slab.len();
            append_capped(slab, name)?;
            let value_start = slab.len();
            append_capped(slab, value)?;
            fields.push(Field {
                name: name_start as u32..value_start as u32,
                value: value_start as u32..slab.len() as u32,
            });
            *at_start = false;
        } else if b & 0x40 != 0 {
            // Literal with incremental indexing.
            let (idx, n) = decode_int(data, 6)?;
            pos += n;
            let name_start = slab.len();
            if idx == 0 {
                let n2 = decode_string_into(&data[pos..], slab)?;
                pos += n2;
            } else {
                let (nm, _) = self.get(idx)?;
                append_capped(slab, nm)?;
            }
            let name_end = slab.len();
            let value_start = name_end;
            let n2 = decode_string_into(&data[pos..], slab)?;
            pos += n2;
            let value_end = slab.len();
            // Re-slice after the appends: the slab may have reallocated
            // between the name and the value copies.
            let name = slab[name_start..name_end].to_vec();
            let value = slab[value_start..value_end].to_vec();
            check_name(&name)?;
            self.insert(name, value);
            fields.push(Field {
                name: name_start as u32..name_end as u32,
                value: value_start as u32..value_end as u32,
            });
            *at_start = false;
        } else if b & 0x20 != 0 {
            // Dynamic table size update.
            if !*at_start {
                return Err(HpackError::MisplacedSizeUpdate);
            }
            let (new_max, n) = decode_int(data, 5)?;
            pos += n;
            self.set_max_size(new_max as usize)?;
        } else {
            // Literal without indexing (`0000`) or never-indexed
            // (`0001`); both behave the same here.
            let (idx, n) = decode_int(data, 4)?;
            pos += n;
            let name_start = slab.len();
            if idx == 0 {
                let n2 = decode_string_into(&data[pos..], slab)?;
                pos += n2;
            } else {
                let (nm, _) = self.get(idx)?;
                append_capped(slab, nm)?;
            }
            let name_end = slab.len();
            let n2 = decode_string_into(&data[pos..], slab)?;
            pos += n2;
            let value_end = slab.len();
            check_name(&slab[name_start..name_end])?;
            fields.push(Field {
                name: name_start as u32..name_end as u32,
                value: name_end as u32..value_end as u32,
            });
            *at_start = false;
        }
        Ok(pos)
    }

    /// Current dynamic table size in bytes (for tests).
    #[cfg(test)]
    fn table_size(&self) -> usize {
        self.size
    }

    /// Dynamic table entries oldest-first (for tests).
    #[cfg(test)]
    fn table_entries(&self) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.table
            .iter()
            .rev()
            .map(|e| (e.name.clone(), e.value.clone()))
            .collect()
    }
}

/// Header names must be non-empty and lowercase on the wire.
fn check_name(name: &[u8]) -> Result<(), HpackError> {
    if name.is_empty() {
        return Err(HpackError::EmptyName);
    }
    if name.iter().any(|b| b.is_ascii_uppercase()) {
        return Err(HpackError::UppercaseName);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn int_rfc_example() {
        // RFC 7541 §5.1: 1337 with a 5-bit prefix.
        let mut out = Vec::new();
        encode_int(1337, 5, 0x00, &mut out);
        assert_eq!(out, vec![0x1f, 0x9a, 0x0a]);
        assert_eq!(decode_int(&out, 5).unwrap(), (1337, 3));
    }

    #[test]
    fn int_roundtrip() {
        for prefix in 1..=8u8 {
            // `first` carries only the opcode bits above the prefix.
            let pfx_mask: u8 = if prefix == 8 {
                0xff
            } else {
                (1u8 << prefix) - 1
            };
            let first = 0xa0 & !pfx_mask;
            for &v in &[0u32, 1, 30, 31, 127, 128, 255, 1337, 65535, u32::MAX] {
                let mut out = Vec::new();
                encode_int(v, prefix, first, &mut out);
                let (got, _) = decode_int(&out, prefix).unwrap();
                assert_eq!(got, v, "prefix {prefix} value {v}");
            }
        }
    }

    #[test]
    fn int_overflow_rejected() {
        // Six continuation bytes: would exceed 32 bits.
        let bad = vec![0xff, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00];
        assert_eq!(decode_int(&bad, 8), Err(HpackError::BadInteger));
        assert_eq!(decode_int(&[0xff], 8), Err(HpackError::Truncated));
    }

    #[test]
    fn huffman_rfc_vectors() {
        // RFC 7541 Appendix C wire vectors (validated against the RFC).
        let cases = [
            ("www.example.com", "f1e3c2e5f23a6ba0ab90f4ff"),
            ("no-cache", "a8eb10649cbf"),
            ("custom-key", "25a849e95ba97d7f"),
            ("private", "aec3771a4b"),
            ("date", "90692f"),
            (
                "Mon, 21 Oct 2013 20:13:21 GMT",
                "d07abe941054d444a8200595040b8166e082a62d1bff",
            ),
        ];
        for (plain, wire) in cases {
            let mut enc = Vec::new();
            huffman_encode(plain.as_bytes(), &mut enc);
            assert_eq!(to_hex(&enc), wire, "{plain}");
            let mut dec = Vec::new();
            huffman_decode(&hex(wire), &mut dec).unwrap();
            assert_eq!(dec, plain.as_bytes(), "{plain}");
        }
    }

    #[test]
    fn huffman_roundtrip_all_bytes() {
        for b in 0..=255u8 {
            let mut enc = Vec::new();
            huffman_encode(&[b], &mut enc);
            let mut dec = Vec::new();
            huffman_decode(&enc, &mut dec).unwrap();
            assert_eq!(dec, vec![b]);
        }
    }

    #[test]
    fn huffman_bad_inputs_rejected() {
        let mut out = Vec::new();
        // Eight zero bits: no code starts with 00000.
        assert_eq!(
            huffman_decode(&[0x00], &mut out),
            Err(HpackError::BadHuffman)
        );
        // EOS (30 one-bits) must never appear as a symbol.
        assert_eq!(
            huffman_decode(&[0xff, 0xff, 0xff, 0xff], &mut out),
            Err(HpackError::BadHuffman)
        );
        // Valid encoding of "a" with a corrupted padding bit.
        let mut enc = Vec::new();
        huffman_encode(b"a", &mut enc);
        assert_eq!(enc.len(), 1);
        let bad = vec![enc[0] & 0xf0];
        assert_eq!(huffman_decode(&bad, &mut out), Err(HpackError::BadHuffman));
        // Empty input decodes to empty.
        let mut fresh = Vec::new();
        huffman_decode(&[], &mut fresh).unwrap();
        assert!(fresh.is_empty());
    }

    /// Decode a block through the slab form and return the header list
    /// as strings.
    fn decode_all(block: &[u8], dec: &mut Decoder) -> Vec<(String, String)> {
        let mut slab = DecodedHeaders::new();
        dec.decode_block(block, 256, &mut slab).unwrap();
        (0..slab.len())
            .map(|i| {
                let (n, v) = slab.field(i);
                (
                    String::from_utf8(n.to_vec()).unwrap(),
                    String::from_utf8(v.to_vec()).unwrap(),
                )
            })
            .collect()
    }

    /// Decode a block into a fresh slab (for shape assertions).
    fn decode_slab(block: &[u8], dec: &mut Decoder) -> DecodedHeaders {
        let mut slab = DecodedHeaders::new();
        dec.decode_block(block, 256, &mut slab).unwrap();
        slab
    }

    #[test]
    fn rfc_c4_request_sequence_no_huffman() {
        // RFC 7541 §C.4: three requests, decoder state evolves.
        let mut dec = Decoder::new(4096);
        let h1 = decode_all(
            &hex("8286 8441 0f 7777772e 6578616d706c652e636f6d"),
            &mut dec,
        );
        assert_eq!(
            h1,
            [
                (":method", "GET"),
                (":scheme", "http"),
                (":path", "/"),
                (":authority", "www.example.com"),
            ]
            .into_iter()
            .map(|(a, b)| (a.to_owned(), b.to_owned()))
            .collect::<Vec<_>>()
        );
        assert_eq!(
            dec.table_entries(),
            vec![(b":authority".to_vec(), b"www.example.com".to_vec())]
        );

        let h2 = decode_all(&hex("8286 84be 5808 6e6f2d6361636865"), &mut dec);
        assert_eq!(
            h2[3],
            (":authority".to_owned(), "www.example.com".to_owned())
        );
        assert_eq!(h2[4], ("cache-control".to_owned(), "no-cache".to_owned()));
        assert_eq!(
            dec.table_entries(),
            vec![
                (b":authority".to_vec(), b"www.example.com".to_vec()),
                (b"cache-control".to_vec(), b"no-cache".to_vec()),
            ]
        );

        let h3 = decode_all(
            &hex("8287 85bf 400a 637573746f6d2d6b6579 0d 637573746f6d2d686561646572"),
            &mut dec,
        );
        assert_eq!(h3[0], (":method".to_owned(), "GET".to_owned()));
        assert_eq!(h3[1], (":scheme".to_owned(), "https".to_owned()));
        assert_eq!(h3[2], (":path".to_owned(), "/index.html".to_owned()));
        assert_eq!(
            h3[3],
            (":authority".to_owned(), "www.example.com".to_owned())
        );
        assert_eq!(h3[4], ("custom-key".to_owned(), "custom-header".to_owned()));
        assert_eq!(dec.table_size(), 55 + 53 + 57);
    }

    #[test]
    fn rfc_c5_request_sequence_huffman() {
        // RFC 7541 §C.5: same three requests, Huffman-coded.
        let mut dec = Decoder::new(4096);
        let h1 = decode_all(&hex("8286 8441 8c f1e3c2e5f23a6ba0ab90f4ff"), &mut dec);
        assert_eq!(
            h1[3],
            (":authority".to_owned(), "www.example.com".to_owned())
        );

        let h2 = decode_all(&hex("8286 84be 5886 a8eb10649cbf"), &mut dec);
        assert_eq!(h2[4], ("cache-control".to_owned(), "no-cache".to_owned()));

        let h3 = decode_all(
            &hex("8287 85bf 4088 25a849e95ba97d7f 89 25a849e95a728e42d9"),
            &mut dec,
        );
        assert_eq!(h3[4], ("custom-key".to_owned(), "custom-header".to_owned()));
    }

    #[test]
    fn encoder_matches_rfc_c5_1() {
        // Same headers as RFC 7541 C.5.1. This encoder deliberately
        // uses literal-without-indexing (`01`) where the RFC example
        // uses incremental indexing (`41`): no dynamic-table state is
        // ever created. The indexed fields and the Huffman bytes match
        // the RFC exactly.
        let mut enc = Encoder::new();
        let headers: Vec<(&[u8], &[u8])> = vec![
            (b":method", b"GET"),
            (b":scheme", b"http"),
            (b":path", b"/"),
            (b":authority", b"www.example.com"),
        ];
        let mut out = Vec::new();
        enc.encode(&headers, &mut out);
        assert_eq!(out, hex("8286 8401 8c f1e3c2e5f23a6ba0ab90f4ff"));
        // And the block decodes back to the same headers.
        let mut dec = Decoder::new(4096);
        let mut slab = DecodedHeaders::new();
        dec.decode_block(&out, 100, &mut slab).unwrap();
        let fields: Vec<(Vec<u8>, Vec<u8>)> = (0..slab.len())
            .map(|i| {
                let (n, v) = slab.field(i);
                (n.to_vec(), v.to_vec())
            })
            .collect();
        let want: Vec<(Vec<u8>, Vec<u8>)> = headers
            .iter()
            .map(|(n, v)| (n.to_vec(), v.to_vec()))
            .collect();
        assert_eq!(fields, want);
    }

    #[test]
    fn encoder_decoder_roundtrip() {
        let mut enc = Encoder::new();
        let mut dec = Decoder::new(4096);
        let headers: Vec<(&[u8], &[u8])> = vec![
            (b":method", b"POST"),
            (b":scheme", b"https"),
            (b":path", b"/submit?x=1&y=2"),
            (b":authority", b"example.com:8443"),
            (b"content-type", b"application/json"),
            (b"x-custom-thing", b"some fairly long value with spaces"),
            (b"empty", b""),
        ];
        let mut wire = Vec::new();
        enc.encode(&headers, &mut wire);
        let mut slab = DecodedHeaders::new();
        dec.decode_block(&wire, 64, &mut slab).unwrap();
        let out: Vec<(Vec<u8>, Vec<u8>)> = (0..slab.len())
            .map(|i| {
                let (n, v) = slab.field(i);
                (n.to_vec(), v.to_vec())
            })
            .collect();
        let expect: Vec<(Vec<u8>, Vec<u8>)> = headers
            .into_iter()
            .map(|(n, v)| (n.to_vec(), v.to_vec()))
            .collect();
        assert_eq!(out, expect);
    }

    #[test]
    fn dynamic_table_evicts_oldest() {
        let mut dec = Decoder::new(128);
        // Each entry costs 32 + 1 + 2 = 35 bytes; the fourth insert
        // (140 > 128) evicts the oldest.
        for i in 0..4u8 {
            // Literal with incremental indexing, new name "k", value "v{i}".
            let block = [0x40, 0x01, b'k', 0x02, b'v', b'0' + i];
            let mut slab = DecodedHeaders::new();
            dec.decode_block(&block, 16, &mut slab).unwrap();
        }
        let entries = dec.table_entries();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].1, b"v1".to_vec());
        assert_eq!(entries[2].1, b"v3".to_vec());
        assert!(dec.table_size() <= 128);
    }

    #[test]
    fn table_size_update_shrinks() {
        let mut dec = Decoder::new(4096);
        // Insert one entry, then shrink the table to 0 mid-... at block start.
        let mut slab = DecodedHeaders::new();
        dec.decode_block(&[0x40, 0x01, b'k', 0x01, b'v'], 16, &mut slab)
            .unwrap();
        assert_eq!(dec.table_size(), 32 + 1 + 1);
        dec.decode_block(&[0x20], 16, &mut slab).unwrap(); // size update to 0
        assert_eq!(dec.table_size(), 0);
        // Growing beyond the agreed max is an error (max is 0 now).
        assert_eq!(
            dec.decode_block(&[0x3f, 0xe1, 0x1f], 16, &mut slab),
            Err(HpackError::BadTableSize(4096))
        );
        // A size update after a field is misplaced.
        assert_eq!(
            dec.decode_block(&[0x82, 0x20], 16, &mut slab),
            Err(HpackError::MisplacedSizeUpdate)
        );
    }

    #[test]
    fn bad_indexes_rejected() {
        let mut dec = Decoder::new(4096);
        let mut slab = DecodedHeaders::new();
        // Indexed 0 is never valid.
        assert_eq!(
            dec.decode_block(&[0x80], 16, &mut slab),
            Err(HpackError::BadIndex(0))
        );
        // Dynamic index past the end.
        assert_eq!(
            dec.decode_block(&[0xbe], 16, &mut slab),
            Err(HpackError::BadIndex(62))
        );
        // Uppercase names are a protocol violation.
        assert_eq!(
            dec.decode_block(&[0x40, 0x01, b'K', 0x01, b'v'], 16, &mut slab),
            Err(HpackError::UppercaseName)
        );
        // A bad name must not pollute the dynamic table.
        assert_eq!(dec.table_size(), 0);
        // Field limit enforced.
        assert_eq!(
            dec.decode_block(&[0x82, 0x82], 1, &mut slab),
            Err(HpackError::TooManyFields)
        );
        // Bad Huffman in a literal value is rejected.
        assert_eq!(
            dec.decode_block(&[0x00, 0x01, b'k', 0x81, 0x00], 16, &mut slab),
            Err(HpackError::BadHuffman)
        );
    }

    #[test]
    fn slab_shape_is_packed() {
        // Field ranges must tile the slab back-to-back with no gaps:
        // the sum of all name+value lengths equals bytes.len().
        let mut dec = Decoder::new(4096);
        let slab = decode_slab(
            &hex(
                "8286 8441 8c f1e3c2e5f23a6ba0ab90f4ff 40 0a 637573746f6d2d6b6579 0d 637573746f6d2d686561646572",
            ),
            &mut dec,
        );
        assert_eq!(slab.len(), 5);
        assert!(!slab.is_empty());
        let mut total = 0usize;
        let mut cursor = 0u32;
        for f in &slab.fields {
            assert_eq!(
                f.name.start, cursor,
                "name must start where the previous value ended"
            );
            assert!(f.name.end > f.name.start, "empty name");
            assert_eq!(f.value.start, f.name.end);
            assert!(f.value.end >= f.value.start);
            total += (f.name.end - f.name.start) as usize + (f.value.end - f.value.start) as usize;
            cursor = f.value.end;
        }
        assert_eq!(total, slab.bytes.len());
        assert_eq!(cursor as usize, slab.bytes.len());
        // Spot-check contents through the accessor.
        assert_eq!(slab.field(0), (b":method".as_slice(), b"GET".as_slice()));
        assert_eq!(
            slab.field(4),
            (b"custom-key".as_slice(), b"custom-header".as_slice())
        );
    }

    #[test]
    fn slab_reused_across_blocks() {
        // One DecodedHeaders reused across blocks: each decode resets
        // it, so only the latest block's bytes are present, while the
        // dynamic table keeps evolving across the reuse.
        let mut dec = Decoder::new(4096);
        let mut slab = DecodedHeaders::new();
        dec.decode_block(&hex("8286 8441 8c f1e3c2e5f23a6ba0ab90f4ff"), 64, &mut slab)
            .unwrap();
        let first_len = slab.bytes.len();
        assert_eq!(slab.len(), 4);
        dec.decode_block(&hex("8286 84be 5886 a8eb10649cbf"), 64, &mut slab)
            .unwrap();
        assert_eq!(slab.len(), 5);
        assert_eq!(
            slab.field(4),
            (b"cache-control".as_slice(), b"no-cache".as_slice())
        );
        // The slab holds exactly this block: sum of lengths == bytes.len().
        let total: usize = slab
            .fields
            .iter()
            .map(|f| (f.name.end - f.name.start + f.value.end - f.value.start) as usize)
            .sum();
        assert_eq!(total, slab.bytes.len());
        assert_ne!(slab.bytes.len(), first_len);
        // Dynamic-table state survived the slab reuse: index 63 names
        // the entry inserted by the first block (62 is the newer
        // cache-control entry from the second block).
        let mut probe = DecodedHeaders::new();
        dec.decode_block(&[0xbf], 64, &mut probe).unwrap();
        assert_eq!(
            probe.field(0),
            (b":authority".as_slice(), b"www.example.com".as_slice())
        );
    }

    #[test]
    fn slab_capped_at_64kib() {
        // A literal string that would decode past 64 KiB fails instead
        // of growing the slab without bound.
        let mut dec = Decoder::new(4096);
        let mut slab = DecodedHeaders::new();
        let mut block = vec![0x00, 0x01, b'k'];
        encode_int(70_000, 7, 0x00, &mut block);
        block.extend(vec![b'x'; 70_000]);
        assert_eq!(
            dec.decode_block(&block, 64, &mut slab),
            Err(HpackError::BlockTooLarge)
        );
        assert!(slab.fields.is_empty());
        // Just under the cap still decodes.
        let mut block = vec![0x00, 0x01, b'k'];
        encode_int(60_000, 7, 0x00, &mut block);
        block.extend(vec![b'x'; 60_000]);
        dec.decode_block(&block, 64, &mut slab).unwrap();
        assert_eq!(slab.len(), 1);
        assert_eq!(slab.field(0).1.len(), 60_000);
    }

    fn to_hex(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
            s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
        }
        s
    }
}
