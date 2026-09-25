//! Huffman coding for HPACK string literals (RFC 7541, Section 5.2).
//!
//! The code table below is transcribed from RFC 7541 Appendix B. Each entry
//! is `(code, len)` where `code` is the Huffman code as an LSB-aligned
//! integer and `len` is its bit length. Symbol 256 is EOS (end of string),
//! used only for padding.
//!
//! Decoding uses canonical-code tables built once from `CODES`
//! (`first_code`/`count` per length plus symbols ordered by `(len, code)`),
//! which is valid because the RFC's code is canonical. Encoding is a direct
//! table lookup per byte.

use std::sync::OnceLock;

// Huffman codes from RFC 7541 Appendix B.
// (code as LSB-aligned integer, bit length), indexed by symbol; 256 = EOS.
const CODES: [(u32, u8); 257] = [
    (0x1ff8, 13),     // 0
    (0x7fffd8, 23),   // 1
    (0xfffffe2, 28),  // 2
    (0xfffffe3, 28),  // 3
    (0xfffffe4, 28),  // 4
    (0xfffffe5, 28),  // 5
    (0xfffffe6, 28),  // 6
    (0xfffffe7, 28),  // 7
    (0xfffffe8, 28),  // 8
    (0xffffea, 24),   // 9
    (0x3ffffffc, 30), // 10
    (0xfffffe9, 28),  // 11
    (0xfffffea, 28),  // 12
    (0x3ffffffd, 30), // 13
    (0xfffffeb, 28),  // 14
    (0xfffffec, 28),  // 15
    (0xfffffed, 28),  // 16
    (0xfffffee, 28),  // 17
    (0xfffffef, 28),  // 18
    (0xffffff0, 28),  // 19
    (0xffffff1, 28),  // 20
    (0xffffff2, 28),  // 21
    (0x3ffffffe, 30), // 22
    (0xffffff3, 28),  // 23
    (0xffffff4, 28),  // 24
    (0xffffff5, 28),  // 25
    (0xffffff6, 28),  // 26
    (0xffffff7, 28),  // 27
    (0xffffff8, 28),  // 28
    (0xffffff9, 28),  // 29
    (0xffffffa, 28),  // 30
    (0xffffffb, 28),  // 31
    (0x14, 6),        // 32
    (0x3f8, 10),      // 33
    (0x3f9, 10),      // 34
    (0xffa, 12),      // 35
    (0x1ff9, 13),     // 36
    (0x15, 6),        // 37
    (0xf8, 8),        // 38
    (0x7fa, 11),      // 39
    (0x3fa, 10),      // 40
    (0x3fb, 10),      // 41
    (0xf9, 8),        // 42
    (0x7fb, 11),      // 43
    (0xfa, 8),        // 44
    (0x16, 6),        // 45
    (0x17, 6),        // 46
    (0x18, 6),        // 47
    (0x0, 5),         // 48
    (0x1, 5),         // 49
    (0x2, 5),         // 50
    (0x19, 6),        // 51
    (0x1a, 6),        // 52
    (0x1b, 6),        // 53
    (0x1c, 6),        // 54
    (0x1d, 6),        // 55
    (0x1e, 6),        // 56
    (0x1f, 6),        // 57
    (0x5c, 7),        // 58
    (0xfb, 8),        // 59
    (0x7ffc, 15),     // 60
    (0x20, 6),        // 61
    (0xffb, 12),      // 62
    (0x3fc, 10),      // 63
    (0x1ffa, 13),     // 64
    (0x21, 6),        // 65
    (0x5d, 7),        // 66
    (0x5e, 7),        // 67
    (0x5f, 7),        // 68
    (0x60, 7),        // 69
    (0x61, 7),        // 70
    (0x62, 7),        // 71
    (0x63, 7),        // 72
    (0x64, 7),        // 73
    (0x65, 7),        // 74
    (0x66, 7),        // 75
    (0x67, 7),        // 76
    (0x68, 7),        // 77
    (0x69, 7),        // 78
    (0x6a, 7),        // 79
    (0x6b, 7),        // 80
    (0x6c, 7),        // 81
    (0x6d, 7),        // 82
    (0x6e, 7),        // 83
    (0x6f, 7),        // 84
    (0x70, 7),        // 85
    (0x71, 7),        // 86
    (0x72, 7),        // 87
    (0xfc, 8),        // 88
    (0x73, 7),        // 89
    (0xfd, 8),        // 90
    (0x1ffb, 13),     // 91
    (0x7fff0, 19),    // 92
    (0x1ffc, 13),     // 93
    (0x3ffc, 14),     // 94
    (0x22, 6),        // 95
    (0x7ffd, 15),     // 96
    (0x3, 5),         // 97
    (0x23, 6),        // 98
    (0x4, 5),         // 99
    (0x24, 6),        // 100
    (0x5, 5),         // 101
    (0x25, 6),        // 102
    (0x26, 6),        // 103
    (0x27, 6),        // 104
    (0x6, 5),         // 105
    (0x74, 7),        // 106
    (0x75, 7),        // 107
    (0x28, 6),        // 108
    (0x29, 6),        // 109
    (0x2a, 6),        // 110
    (0x7, 5),         // 111
    (0x2b, 6),        // 112
    (0x76, 7),        // 113
    (0x2c, 6),        // 114
    (0x8, 5),         // 115
    (0x9, 5),         // 116
    (0x2d, 6),        // 117
    (0x77, 7),        // 118
    (0x78, 7),        // 119
    (0x79, 7),        // 120
    (0x7a, 7),        // 121
    (0x7b, 7),        // 122
    (0x7ffe, 15),     // 123
    (0x7fc, 11),      // 124
    (0x3ffd, 14),     // 125
    (0x1ffd, 13),     // 126
    (0xffffffc, 28),  // 127
    (0xfffe6, 20),    // 128
    (0x3fffd2, 22),   // 129
    (0xfffe7, 20),    // 130
    (0xfffe8, 20),    // 131
    (0x3fffd3, 22),   // 132
    (0x3fffd4, 22),   // 133
    (0x3fffd5, 22),   // 134
    (0x7fffd9, 23),   // 135
    (0x3fffd6, 22),   // 136
    (0x7fffda, 23),   // 137
    (0x7fffdb, 23),   // 138
    (0x7fffdc, 23),   // 139
    (0x7fffdd, 23),   // 140
    (0x7fffde, 23),   // 141
    (0xffffeb, 24),   // 142
    (0x7fffdf, 23),   // 143
    (0xffffec, 24),   // 144
    (0xffffed, 24),   // 145
    (0x3fffd7, 22),   // 146
    (0x7fffe0, 23),   // 147
    (0xffffee, 24),   // 148
    (0x7fffe1, 23),   // 149
    (0x7fffe2, 23),   // 150
    (0x7fffe3, 23),   // 151
    (0x7fffe4, 23),   // 152
    (0x1fffdc, 21),   // 153
    (0x3fffd8, 22),   // 154
    (0x7fffe5, 23),   // 155
    (0x3fffd9, 22),   // 156
    (0x7fffe6, 23),   // 157
    (0x7fffe7, 23),   // 158
    (0xffffef, 24),   // 159
    (0x3fffda, 22),   // 160
    (0x1fffdd, 21),   // 161
    (0xfffe9, 20),    // 162
    (0x3fffdb, 22),   // 163
    (0x3fffdc, 22),   // 164
    (0x7fffe8, 23),   // 165
    (0x7fffe9, 23),   // 166
    (0x1fffde, 21),   // 167
    (0x7fffea, 23),   // 168
    (0x3fffdd, 22),   // 169
    (0x3fffde, 22),   // 170
    (0xfffff0, 24),   // 171
    (0x1fffdf, 21),   // 172
    (0x3fffdf, 22),   // 173
    (0x7fffeb, 23),   // 174
    (0x7fffec, 23),   // 175
    (0x1fffe0, 21),   // 176
    (0x1fffe1, 21),   // 177
    (0x3fffe0, 22),   // 178
    (0x1fffe2, 21),   // 179
    (0x7fffed, 23),   // 180
    (0x3fffe1, 22),   // 181
    (0x7fffee, 23),   // 182
    (0x7fffef, 23),   // 183
    (0xfffea, 20),    // 184
    (0x3fffe2, 22),   // 185
    (0x3fffe3, 22),   // 186
    (0x3fffe4, 22),   // 187
    (0x7ffff0, 23),   // 188
    (0x3fffe5, 22),   // 189
    (0x3fffe6, 22),   // 190
    (0x7ffff1, 23),   // 191
    (0x3ffffe0, 26),  // 192
    (0x3ffffe1, 26),  // 193
    (0xfffeb, 20),    // 194
    (0x7fff1, 19),    // 195
    (0x3fffe7, 22),   // 196
    (0x7ffff2, 23),   // 197
    (0x3fffe8, 22),   // 198
    (0x1ffffec, 25),  // 199
    (0x3ffffe2, 26),  // 200
    (0x3ffffe3, 26),  // 201
    (0x3ffffe4, 26),  // 202
    (0x7ffffde, 27),  // 203
    (0x7ffffdf, 27),  // 204
    (0x3ffffe5, 26),  // 205
    (0xfffff1, 24),   // 206
    (0x1ffffed, 25),  // 207
    (0x7fff2, 19),    // 208
    (0x1fffe3, 21),   // 209
    (0x3ffffe6, 26),  // 210
    (0x7ffffe0, 27),  // 211
    (0x7ffffe1, 27),  // 212
    (0x3ffffe7, 26),  // 213
    (0x7ffffe2, 27),  // 214
    (0xfffff2, 24),   // 215
    (0x1fffe4, 21),   // 216
    (0x1fffe5, 21),   // 217
    (0x3ffffe8, 26),  // 218
    (0x3ffffe9, 26),  // 219
    (0xffffffd, 28),  // 220
    (0x7ffffe3, 27),  // 221
    (0x7ffffe4, 27),  // 222
    (0x7ffffe5, 27),  // 223
    (0xfffec, 20),    // 224
    (0xfffff3, 24),   // 225
    (0xfffed, 20),    // 226
    (0x1fffe6, 21),   // 227
    (0x3fffe9, 22),   // 228
    (0x1fffe7, 21),   // 229
    (0x1fffe8, 21),   // 230
    (0x7ffff3, 23),   // 231
    (0x3fffea, 22),   // 232
    (0x3fffeb, 22),   // 233
    (0x1ffffee, 25),  // 234
    (0x1ffffef, 25),  // 235
    (0xfffff4, 24),   // 236
    (0xfffff5, 24),   // 237
    (0x3ffffea, 26),  // 238
    (0x7ffff4, 23),   // 239
    (0x3ffffeb, 26),  // 240
    (0x7ffffe6, 27),  // 241
    (0x3ffffec, 26),  // 242
    (0x3ffffed, 26),  // 243
    (0x7ffffe7, 27),  // 244
    (0x7ffffe8, 27),  // 245
    (0x7ffffe9, 27),  // 246
    (0x7ffffea, 27),  // 247
    (0x7ffffeb, 27),  // 248
    (0xffffffe, 28),  // 249
    (0x7ffffec, 27),  // 250
    (0x7ffffed, 27),  // 251
    (0x7ffffee, 27),  // 252
    (0x7ffffef, 27),  // 253
    (0x7fffff0, 27),  // 254
    (0x3ffffee, 26),  // 255
    (0x3fffffff, 30), // 256
];

/// End-of-string symbol: only valid as padding, never in data.
pub const EOS: u16 = 256;
/// Longest Huffman code in bits.
const MAX_LEN: usize = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Input ended mid-code and the leftover bits are not a valid
    /// (<= 7 bit, all-ones) EOS padding.
    BadPadding,
    /// The EOS symbol appeared in the data (forbidden by RFC 7541, 5.2).
    EosInData,
}

/// Canonical decode tables derived from [`CODES`].
struct Tables {
    /// First code of each bit length (index = length); `u32::MAX` if none.
    first: [u32; MAX_LEN + 1],
    /// Number of codes of each bit length.
    count: [u16; MAX_LEN + 1],
    /// Offset into `symbols` for each bit length.
    start: [u16; MAX_LEN + 1],
    /// Symbols ordered by (length, code).
    symbols: [u16; 257],
}

static TABLES: OnceLock<Tables> = OnceLock::new();

fn tables() -> &'static Tables {
    TABLES.get_or_init(|| {
        let mut t = Tables {
            first: [u32::MAX; MAX_LEN + 1],
            count: [0; MAX_LEN + 1],
            start: [0; MAX_LEN + 1],
            symbols: [0; 257],
        };
        let mut pos = 0usize;
        for len in 1..=MAX_LEN {
            // Collect (code, symbol) pairs of this length, sorted by code.
            // n is tiny (<= ~40), so insertion sort is plenty.
            let mut entries: [(u32, u16); 64] = [(0, 0); 64];
            let mut n = 0usize;
            for (sym, &(code, l)) in CODES.iter().enumerate() {
                if l as usize == len {
                    // insertion sort by code
                    let mut i = n;
                    while i > 0 && entries[i - 1].0 > code {
                        entries[i] = entries[i - 1];
                        i -= 1;
                    }
                    entries[i] = (code, sym as u16);
                    n += 1;
                }
            }
            if n == 0 {
                continue;
            }
            debug_assert!(n <= 64, "more codes of one length than expected");
            // The RFC table is canonical: codes of a length are consecutive.
            debug_assert!(
                entries[n - 1].0 - entries[0].0 == (n - 1) as u32,
                "non-canonical Huffman table"
            );
            t.first[len] = entries[0].0;
            t.count[len] = n as u16;
            t.start[len] = pos as u16;
            for entry in entries.iter().take(n) {
                t.symbols[pos] = entry.1;
                pos += 1;
            }
        }
        debug_assert_eq!(pos, 257);
        t
    })
}

/// Huffman-encode `data`, appending to `out`. The final octet is padded with
/// ones (the MSB prefix of EOS), per RFC 7541 Section 5.2.
pub fn encode(data: &[u8], out: &mut Vec<u8>) {
    out.reserve(data.len());
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    for &b in data {
        let (code, len) = CODES[b as usize];
        acc = (acc << len) | u64::from(code);
        nbits += u32::from(len);
        while nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
        }
        acc &= (1u64 << nbits) - 1;
    }
    if nbits > 0 {
        let pad = (1u64 << (8 - nbits)) - 1;
        out.push(((acc << (8 - nbits)) | pad) as u8);
    }
}

/// Huffman-decode `data`, appending to `out`.
pub fn decode(data: &[u8], out: &mut Vec<u8>) -> Result<(), Error> {
    let t = tables();
    let total_bits = data.len() * 8;
    let mut bitpos = 0usize;
    out.reserve(data.len());

    while bitpos < total_bits {
        let mut val: u32 = 0;
        let mut matched = false;
        for len in 1..=MAX_LEN {
            if bitpos >= total_bits {
                // End of data mid-code: the leftover bits are padding and
                // must be the all-ones MSB prefix of EOS, at most 7 bits.
                let pad = len - 1;
                if pad == 0 || pad > 7 || val != (1u32 << pad) - 1 {
                    return Err(Error::BadPadding);
                }
                return Ok(());
            }
            let byte = data[bitpos / 8];
            let bit = u32::from((byte >> (7 - (bitpos % 8))) & 1);
            bitpos += 1;
            val = (val << 1) | bit;
            let first = t.first[len];
            if first != u32::MAX && val >= first {
                let k = val - first;
                if (k as usize) < t.count[len] as usize {
                    let sym = t.symbols[t.start[len] as usize + k as usize];
                    if sym == EOS {
                        return Err(Error::EosInData);
                    }
                    out.push(sym as u8);
                    matched = true;
                    break;
                }
                // Otherwise the code is longer; keep consuming bits.
            }
        }
        if !matched {
            // 30 bits without matching a code: invalid.
            return Err(Error::BadPadding);
        }
    }
    Ok(())
}

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
    fn table_covers_all_symbols() {
        assert_eq!(CODES.len(), 257);
        for (i, &(_, len)) in CODES.iter().enumerate() {
            assert!((5..=30).contains(&len), "sym {i} len {len}");
        }
        assert_eq!(CODES[256], (0x3fff_ffff, 30)); // EOS
    }

    #[test]
    fn table_is_a_valid_prefix_code() {
        // No code is a prefix of another: walk a bit trie.
        #[derive(Default)]
        struct Node {
            kid: [Option<Box<Node>>; 2],
            leaf: bool,
        }
        let mut root = Node::default();
        for &(code, len) in &CODES {
            let mut node = &mut root;
            for i in (0..len).rev() {
                let bit = ((code >> i) & 1) as usize;
                assert!(!node.leaf, "code extends a shorter code");
                node = node.kid[bit].get_or_insert_with(|| Box::new(Node::default()));
            }
            assert!(
                node.kid.iter().all(|k| k.is_none()) && !node.leaf,
                "duplicate code"
            );
            node.leaf = true;
        }
    }

    #[test]
    fn rfc7541_c41_www_example_com() {
        // RFC 7541 C.4.1: "www.example.com" Huffman-encodes to 12 bytes.
        let mut out = Vec::new();
        encode(b"www.example.com", &mut out);
        assert_eq!(out, hex("f1e3 c2e5 f23a 6ba0 ab90 f4ff"));

        let mut back = Vec::new();
        decode(&hex("f1e3 c2e5 f23a 6ba0 ab90 f4ff"), &mut back).unwrap();
        assert_eq!(back, b"www.example.com");
    }

    #[test]
    fn rfc7541_spot_vectors() {
        let cases: &[(&[u8], &str)] = &[
            (b"no-cache", "a8eb10649cbf"),       // C.4.2 (hyphenated)
            (b"custom-key", "25a849e95ba97d7f"), // C.4.3
            (b"custom-value", "25a849e95bb8e8b4bf"),
            (b"302", "6402"),   // C.6.1
            (b"307", "640eff"), // C.6.2
            (b"private", "aec3771a4b"),
            (b"gzip", "9bd9ab"), // C.6.3
        ];
        for (plain, hx) in cases {
            let mut enc = Vec::new();
            encode(plain, &mut enc);
            assert_eq!(enc, hex(hx), "encode {plain:?}");
            let mut dec = Vec::new();
            decode(&hex(hx), &mut dec).unwrap();
            assert_eq!(dec, *plain, "decode {hx}");
        }
    }

    #[test]
    fn rfc7541_c61_date_and_location() {
        let date = b"Mon, 21 Oct 2013 20:13:21 GMT";
        let mut enc = Vec::new();
        encode(date, &mut enc);
        assert_eq!(enc.len(), 22); // RFC C.6.1
        let mut dec = Vec::new();
        decode(
            &hex("d07a be94 1054 d444 a820 0595 040b 8166 e082 a62d 1bff"),
            &mut dec,
        )
        .unwrap();
        assert_eq!(dec, date);

        let loc = b"https://www.example.com";
        let mut enc = Vec::new();
        encode(loc, &mut enc);
        assert_eq!(enc.len(), 17);
        let mut dec = Vec::new();
        decode(&hex("9d29 ad17 1863 c78f 0b97 c8e9 ae82 ae43 d3"), &mut dec).unwrap();
        assert_eq!(dec, loc);
    }

    #[test]
    fn rfc7541_c63_set_cookie() {
        // 56-byte value -> 45-byte Huffman blob (RFC 7541 C.6.3).
        let value = b"foo=ASDJKHQKBZXOQWEOPIUAXQWEOIU; max-age=3600; version=1";
        let mut enc = Vec::new();
        encode(value, &mut enc);
        assert_eq!(enc.len(), 45);
        let mut dec = Vec::new();
        decode(
            &hex("94e7 821d d7f2 e6c7 b335 dfdf cd5b 3960 d5af 2708 7f36 72c1 ab27 0fb5 291f 9587 3160 65c0 03ed 4ee5 b106 3d50 07"),
            &mut dec,
        )
        .unwrap();
        assert_eq!(dec, value);
    }

    #[test]
    fn roundtrip_all_byte_values() {
        let all: Vec<u8> = (0..=255u8).collect();
        let mut enc = Vec::new();
        encode(&all, &mut enc);
        let mut dec = Vec::new();
        decode(&enc, &mut dec).unwrap();
        assert_eq!(dec, all);
    }

    #[test]
    fn roundtrip_empty() {
        let mut enc = Vec::new();
        encode(b"", &mut enc);
        assert!(enc.is_empty());
        let mut dec = Vec::new();
        decode(b"", &mut dec).unwrap();
        assert!(dec.is_empty());
    }

    #[test]
    fn roundtrip_typical_headers() {
        for s in [
            "GET",
            "/index.html",
            "Mozilla/5.0 (X11; Linux x86_64)",
            "text/html,application/xhtml+xml",
            "Tue, 21 Oct 2026 07:28:00 GMT",
            "sessionid=abc123; theme=dark",
        ] {
            let mut enc = Vec::new();
            encode(s.as_bytes(), &mut enc);
            let mut dec = Vec::new();
            decode(&enc, &mut dec).unwrap();
            assert_eq!(dec, s.as_bytes());
        }
    }

    #[test]
    fn explicit_eos_rejected() {
        // 30 one-bits = the EOS code itself.
        assert_eq!(
            decode(&hex("ffffff ff"), &mut Vec::new()),
            Err(Error::EosInData)
        );
    }

    #[test]
    fn bad_padding_rejected() {
        // 0x00: '0' (00000) then leftover 000, not all-ones.
        assert_eq!(decode(&[0x00], &mut Vec::new()), Err(Error::BadPadding));
        // 8 one-bits: valid 7-bit pad would stop earlier; 8 ones can only be
        // a longer-than-7 padding -> error. (0xff = 7 ones would decode as
        // padding after a symbol; here there is no symbol.)
        assert_eq!(
            decode(&[0xff, 0xff], &mut Vec::new()),
            Err(Error::BadPadding)
        );
    }

    #[test]
    fn padding_exactly_seven_ones_ok() {
        // 'a' = 00011 (5 bits) + 3 one-bits padding = 0x1f.
        let mut dec = Vec::new();
        decode(&[0x1f], &mut dec).unwrap();
        assert_eq!(dec, b"a");
    }
}
