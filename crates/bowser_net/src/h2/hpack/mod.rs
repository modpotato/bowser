//! HPACK header compression (RFC 7541).
//!
//! [`Encoder`] turns a header list into an HPACK header block;
//! [`Decoder`] turns a header block back into headers. Both share the
//! connection's dynamic table discipline: entries are `(name, value)` pairs
//! with size `32 + name.len() + value.len()`, newest first.
//!
//! [`encode_int`]/[`decode_int`] implement the Section 5.1 integer
//! representation; [`encode_string`]/[`decode_string`] the Section 5.2 string
//! literals (plain or Huffman-coded via [`huffman`]).

pub mod huffman;

use huffman::Error as HuffmanError;

/// Errors from HPACK decoding (and table misuse).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HpackError {
    /// Input ended before a complete representation.
    Truncated,
    /// Integer continuation would exceed `u32`.
    IntegerOverflow,
    /// Indexed representation used index 0.
    ZeroIndex,
    /// Index is outside the static + dynamic tables.
    BadIndex(u32),
    /// Bad Huffman data in a string literal.
    Huffman(HuffmanError),
    /// A dynamic table size update appeared after a header field.
    SizeUpdateNotAtStart,
    /// A dynamic table size update exceeds the protocol maximum.
    TableSizeTooLarge,
}

impl From<HuffmanError> for HpackError {
    fn from(e: HuffmanError) -> Self {
        HpackError::Huffman(e)
    }
}

/// A decoded header field. Names and values are opaque octets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderField {
    pub name: Vec<u8>,
    pub value: Vec<u8>,
}

impl HeaderField {
    pub fn new(name: Vec<u8>, value: Vec<u8>) -> Self {
        HeaderField { name, value }
    }
}

// ---------------------------------------------------------------------------
// Integer representation (RFC 7541, Section 5.1)
// ---------------------------------------------------------------------------

/// Encode `value` with an N-bit prefix (`1..=8`). `prefix_mask` carries the
/// already-set high bits of the first octet (e.g. `0x80` for an indexed
/// representation).
pub fn encode_int(value: u32, prefix_bits: u8, prefix_mask: u8, out: &mut Vec<u8>) {
    debug_assert!((1..=8).contains(&prefix_bits));
    let max = (1u32 << prefix_bits) - 1;
    debug_assert!(u32::from(prefix_mask) & max == 0);
    if value < max {
        out.push(prefix_mask | value as u8);
        return;
    }
    out.push(prefix_mask | max as u8);
    let mut rest = value - max;
    while rest >= 128 {
        out.push((rest % 128) as u8 | 0x80);
        rest /= 128;
    }
    out.push(rest as u8);
}

/// Decode an integer with an N-bit prefix. Returns `(value, bytes consumed)`.
pub fn decode_int(data: &[u8], prefix_bits: u8) -> Result<(u32, usize), HpackError> {
    debug_assert!((1..=8).contains(&prefix_bits));
    let max = (1u32 << prefix_bits) - 1;
    let first = *data.first().ok_or(HpackError::Truncated)?;
    let lo = u32::from(first) & max;
    if lo < max {
        return Ok((lo, 1));
    }
    let mut acc: u64 = u64::from(max);
    let mut shift = 0u32;
    let mut consumed = 1usize;
    loop {
        let b = *data.get(consumed).ok_or(HpackError::Truncated)?;
        consumed += 1;
        if shift >= 32 {
            return Err(HpackError::IntegerOverflow);
        }
        acc += (u64::from(b & 0x7f)) << shift;
        if acc > u64::from(u32::MAX) {
            return Err(HpackError::IntegerOverflow);
        }
        shift += 7;
        if b & 0x80 == 0 {
            break;
        }
    }
    Ok((acc as u32, consumed))
}

// ---------------------------------------------------------------------------
// String literals (RFC 7541, Section 5.2)
// ---------------------------------------------------------------------------

/// Encode a string literal; `huffman` selects Huffman coding.
pub fn encode_string(data: &[u8], huffman: bool, out: &mut Vec<u8>) {
    if huffman {
        let start = out.len();
        out.push(0); // placeholder for length octet(s)
        let body = out.len();
        huffman::encode(data, out);
        let len = (out.len() - body) as u32;
        // Splice the length prefix in front of the Huffman bytes.
        let mut prefix = Vec::new();
        encode_int(len, 7, 0x80, &mut prefix);
        out.splice(start..body, prefix);
    } else {
        encode_int(data.len() as u32, 7, 0x00, out);
        out.extend_from_slice(data);
    }
}

/// Decode a string literal. Returns `(bytes, bytes consumed)`.
pub fn decode_string(data: &[u8]) -> Result<(Vec<u8>, usize), HpackError> {
    let first = *data.first().ok_or(HpackError::Truncated)?;
    let huffman = first & 0x80 != 0;
    let (len, mut consumed) = decode_int(data, 7)?;
    let len = len as usize;
    let end = consumed.checked_add(len).ok_or(HpackError::Truncated)?;
    let bytes = data.get(consumed..end).ok_or(HpackError::Truncated)?;
    consumed = end;
    let mut out = Vec::with_capacity(bytes.len());
    if huffman {
        huffman::decode(bytes, &mut out)?;
    } else {
        out.extend_from_slice(bytes);
    }
    Ok((out, consumed))
}

// ---------------------------------------------------------------------------
// Static table (RFC 7541, Appendix A)
// ---------------------------------------------------------------------------

/// The 61-entry static table: `(name, value)`, 1-based on the wire.
pub const STATIC_TABLE: [(&str, &str); 61] = [
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

pub const STATIC_TABLE_LEN: u32 = STATIC_TABLE.len() as u32; // 61

// ---------------------------------------------------------------------------
// Dynamic table (RFC 7541, Sections 2.3.2, 4)
// ---------------------------------------------------------------------------

/// Overhead bytes per dynamic table entry (RFC 7541, Section 4.1).
pub const ENTRY_OVERHEAD: usize = 32;
/// Default maximum dynamic table size (SETTINGS_HEADER_TABLE_SIZE default).
pub const DEFAULT_MAX_TABLE_SIZE: usize = 4096;

/// The HPACK dynamic table. Index 1 is the newest entry.
#[derive(Debug, Clone, Default)]
pub struct DynamicTable {
    entries: Vec<(Vec<u8>, Vec<u8>)>,
    size: usize,
    max_size: usize,
    /// Protocol limit from SETTINGS_HEADER_TABLE_SIZE; size updates from the
    /// peer may not exceed it.
    protocol_max: usize,
}

impl DynamicTable {
    pub fn new() -> Self {
        DynamicTable {
            entries: Vec::new(),
            size: 0,
            max_size: DEFAULT_MAX_TABLE_SIZE,
            protocol_max: DEFAULT_MAX_TABLE_SIZE,
        }
    }

    /// Current table size in bytes (sum of entry sizes).
    pub fn size(&self) -> usize {
        self.size
    }

    /// Current maximum table size.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Protocol maximum from the peer's SETTINGS_HEADER_TABLE_SIZE.
    pub fn protocol_max(&self) -> usize {
        self.protocol_max
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Set the protocol maximum (from SETTINGS). Evicts if now too large.
    pub fn set_protocol_max(&mut self, max: usize) {
        self.protocol_max = max;
        if self.max_size > max {
            self.set_max_size(max);
        }
    }

    /// Change the maximum table size, evicting oldest entries as needed.
    pub fn set_max_size(&mut self, max: usize) {
        self.max_size = max.min(self.protocol_max);
        self.evict_to_fit(0);
    }

    /// 1-based lookup, 1 = newest.
    pub fn get(&self, index: usize) -> Option<(&[u8], &[u8])> {
        self.entries
            .get(index.checked_sub(1)?)
            .map(|(n, v)| (n.as_slice(), v.as_slice()))
    }

    /// Insert a new entry as the newest, evicting oldest-first. If the entry
    /// alone exceeds the maximum, the table is emptied and nothing is added
    /// (RFC 7541, Section 4.4).
    pub fn insert(&mut self, name: Vec<u8>, value: Vec<u8>) {
        let entry_size = ENTRY_OVERHEAD + name.len() + value.len();
        if entry_size > self.max_size {
            self.entries.clear();
            self.size = 0;
            return;
        }
        self.evict_to_fit(entry_size);
        self.entries.insert(0, (name, value));
        self.size += entry_size;
    }

    fn evict_to_fit(&mut self, need: usize) {
        while self.size + need > self.max_size {
            let (n, v) = self.entries.pop().expect("size accounting bug");
            self.size -= ENTRY_OVERHEAD + n.len() + v.len();
        }
    }

    /// Absolute index (static 1..=61, dynamic 62..) to a header field.
    pub fn indexed(&self, index: u32) -> Result<HeaderField, HpackError> {
        if index == 0 {
            return Err(HpackError::ZeroIndex);
        }
        if index <= STATIC_TABLE_LEN {
            let (n, v) = STATIC_TABLE[(index - 1) as usize];
            return Ok(HeaderField::new(
                n.as_bytes().to_vec(),
                v.as_bytes().to_vec(),
            ));
        }
        let dyn_index = (index - STATIC_TABLE_LEN) as usize;
        match self.get(dyn_index) {
            Some((n, v)) => Ok(HeaderField::new(n.to_vec(), v.to_vec())),
            None => Err(HpackError::BadIndex(index)),
        }
    }

    /// Absolute index to a name only (for literal representations).
    /// Index 0 means "literal name follows" and is handled by the caller.
    pub fn indexed_name(&self, index: u32) -> Result<Vec<u8>, HpackError> {
        Ok(self.indexed(index)?.name)
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// HPACK encoder. Holds the dynamic table and emits header blocks.
///
/// Strategy: indexed representation on any full (name, value) match in the
/// static or dynamic table; otherwise a literal with incremental indexing,
/// reusing an indexed name when one matches. The encoder always uses
/// incremental indexing (like the RFC 7541 Appendix C examples), so the
/// dynamic table grows on both sides in lockstep.
#[derive(Debug)]
pub struct Encoder {
    table: DynamicTable,
    huffman: bool,
    pending_max: Option<usize>,
}

impl Encoder {
    pub fn new() -> Self {
        Encoder {
            table: DynamicTable::new(),
            huffman: true,
            pending_max: None,
        }
    }

    /// Use Huffman coding for string literals (default: on).
    pub fn set_huffman(&mut self, huffman: bool) {
        self.huffman = huffman;
    }

    /// Set the protocol maximum dynamic table size from the peer's
    /// SETTINGS_HEADER_TABLE_SIZE. Size updates are clamped to it, and the
    /// table evicts if it is now too large.
    pub fn set_protocol_max(&mut self, max: usize) {
        self.table.set_protocol_max(max);
    }

    /// Queue a dynamic table size update, emitted at the start of the next
    /// header block (RFC 7541, Section 4.2). Clamped to the protocol maximum
    /// (and to `u32::MAX`, the largest encodable value).
    pub fn set_max_table_size(&mut self, max: usize) {
        self.pending_max = Some(max.min(self.table.protocol_max()).min(u32::MAX as usize));
    }

    pub fn table(&self) -> &DynamicTable {
        &self.table
    }

    /// Encode a header list into `out`.
    pub fn encode(&mut self, headers: &[HeaderField], out: &mut Vec<u8>) {
        if let Some(max) = self.pending_max.take() {
            encode_int(max as u32, 5, 0x20, out);
            self.table.set_max_size(max);
        }
        for h in headers {
            self.encode_one(h, out);
        }
    }

    fn encode_one(&mut self, h: &HeaderField, out: &mut Vec<u8>) {
        // Full match in the static table?
        if let Some(i) = static_full_match(&h.name, &h.value) {
            encode_int(i, 7, 0x80, out);
            return;
        }
        // Full match in the dynamic table?
        if let Some(i) = dynamic_full_match(&self.table, &h.name, &h.value) {
            encode_int(STATIC_TABLE_LEN + i as u32, 7, 0x80, out);
            return;
        }
        // Name match -> literal with incremental indexing, indexed name.
        let name_index = static_name_match(&h.name).or_else(|| {
            dynamic_name_match(&self.table, &h.name).map(|i| STATIC_TABLE_LEN + i as u32)
        });
        match name_index {
            Some(idx) => {
                encode_int(idx, 6, 0x40, out);
            }
            None => {
                encode_int(0, 6, 0x40, out);
                encode_string(&h.name, self.huffman, out);
            }
        }
        encode_string(&h.value, self.huffman, out);
        self.table.insert(h.name.clone(), h.value.clone());
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Encoder::new()
    }
}

fn static_full_match(name: &[u8], value: &[u8]) -> Option<u32> {
    STATIC_TABLE
        .iter()
        .position(|(n, v)| n.as_bytes() == name && v.as_bytes() == value)
        .map(|i| i as u32 + 1)
}

fn static_name_match(name: &[u8]) -> Option<u32> {
    STATIC_TABLE
        .iter()
        .position(|(n, _)| n.as_bytes() == name)
        .map(|i| i as u32 + 1)
}

fn dynamic_full_match(table: &DynamicTable, name: &[u8], value: &[u8]) -> Option<usize> {
    (1..=table.len()).find(|&i| table.get(i).is_some_and(|(n, v)| n == name && v == value))
}

fn dynamic_name_match(table: &DynamicTable, name: &[u8]) -> Option<usize> {
    (1..=table.len()).find(|&i| table.get(i).is_some_and(|(n, _)| n == name))
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// HPACK decoder. Holds the dynamic table; feed it header blocks in order.
#[derive(Debug, Default)]
pub struct Decoder {
    table: DynamicTable,
}

impl Decoder {
    pub fn new() -> Self {
        Decoder {
            table: DynamicTable::new(),
        }
    }

    /// Apply the peer's SETTINGS_HEADER_TABLE_SIZE (protocol limit).
    pub fn set_max_table_size(&mut self, max: usize) {
        self.table.set_protocol_max(max);
    }

    pub fn table(&self) -> &DynamicTable {
        &self.table
    }

    /// Decode one header block into a header list, updating the dynamic
    /// table. See RFC 7541, Section 3.
    pub fn decode(&mut self, data: &[u8]) -> Result<Vec<HeaderField>, HpackError> {
        let mut out = Vec::new();
        let mut pos = 0;
        let mut size_update_allowed = true;
        while pos < data.len() {
            let b = data[pos];
            if b & 0x80 != 0 {
                // 6.1 Indexed header field.
                let (index, n) = decode_int(&data[pos..], 7)?;
                pos += n;
                out.push(self.table.indexed(index)?);
                size_update_allowed = false;
            } else if b & 0x40 != 0 {
                // 6.2.1 Literal with incremental indexing.
                let (index, n) = decode_int(&data[pos..], 6)?;
                pos += n;
                let (name, value, m) = self.literal(index, &data[pos..])?;
                pos += m;
                self.table.insert(name.clone(), value.clone());
                out.push(HeaderField::new(name, value));
                size_update_allowed = false;
            } else if b & 0x20 != 0 {
                // 6.3 Dynamic table size update: only at the start.
                if !size_update_allowed {
                    return Err(HpackError::SizeUpdateNotAtStart);
                }
                let (max, n) = decode_int(&data[pos..], 5)?;
                pos += n;
                if max as usize > self.table.protocol_max() {
                    return Err(HpackError::TableSizeTooLarge);
                }
                self.table.set_max_size(max as usize);
            } else {
                // 6.2.2 Literal without indexing / 6.2.3 never-indexed.
                // Both use a 4-bit prefix; the decoder treats them the same
                // (it never re-emits sensitive data into the table anyway).
                let (index, n) = decode_int(&data[pos..], 4)?;
                pos += n;
                let (name, value, m) = self.literal(index, &data[pos..])?;
                pos += m;
                out.push(HeaderField::new(name, value));
                size_update_allowed = false;
            }
        }
        Ok(out)
    }

    /// Decode the (name, value) of a literal representation given the name
    /// index (0 = literal name follows).
    fn literal(&self, index: u32, data: &[u8]) -> Result<(Vec<u8>, Vec<u8>, usize), HpackError> {
        let (name, mut consumed) = if index == 0 {
            decode_string(data)?
        } else {
            (self.table.indexed_name(index)?, 0)
        };
        let (value, n) = decode_string(&data[consumed..])?;
        consumed += n;
        Ok((name, value, consumed))
    }
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

    fn field(name: &str, value: &str) -> HeaderField {
        HeaderField::new(name.as_bytes().to_vec(), value.as_bytes().to_vec())
    }

    fn fields(pairs: &[(&str, &str)]) -> Vec<HeaderField> {
        pairs.iter().map(|(n, v)| field(n, v)).collect()
    }

    // --- Integer representation (RFC 7541 C.1) ---

    #[test]
    fn int_c11_encode_10_with_5_bit_prefix() {
        let mut out = Vec::new();
        encode_int(10, 5, 0x00, &mut out);
        assert_eq!(out, vec![0x0a]);
        assert_eq!(decode_int(&out, 5).unwrap(), (10, 1));
    }

    #[test]
    fn int_c12_encode_1337_with_5_bit_prefix() {
        let mut out = Vec::new();
        encode_int(1337, 5, 0x00, &mut out);
        assert_eq!(out, vec![0x1f, 0x9a, 0x0a]);
        assert_eq!(decode_int(&out, 5).unwrap(), (1337, 3));
    }

    #[test]
    fn int_c13_encode_42_at_octet_boundary() {
        let mut out = Vec::new();
        encode_int(42, 8, 0x00, &mut out);
        assert_eq!(out, vec![0x2a]);
        assert_eq!(decode_int(&out, 8).unwrap(), (42, 1));
    }

    #[test]
    fn int_roundtrip_boundaries() {
        for (prefix, mask, values) in [
            (1u8, 0xa0u8, vec![0u32, 1, 2]),
            (
                5u8,
                0xa0u8,
                vec![0, 30, 31, 32, 127, 128, 255, 1337, 16_383],
            ),
            (8u8, 0x00u8, vec![0, 254, 255, 256, 65_535, u32::MAX]),
        ] {
            for v in values {
                let mut out = Vec::new();
                encode_int(v, prefix, mask, &mut out);
                // High bits of first octet preserved.
                let keep = if prefix == 8 {
                    0u8
                } else {
                    !((1u8 << prefix) - 1)
                };
                assert_eq!(out[0] & keep, mask);
                assert_eq!(decode_int(&out, prefix).unwrap().0, v, "v={v}");
            }
        }
    }

    #[test]
    fn int_rejects_overflow_and_truncation() {
        // Six continuation bytes cannot fit in a u32.
        assert_eq!(
            decode_int(&hex("ff 80 80 80 80 80 80"), 8),
            Err(HpackError::IntegerOverflow)
        );
        // u32::MAX with an 8-bit prefix is ff 80 fe ff ff 0f; bumping the
        // last digit from 15 to 16 overflows.
        let mut over = hex("ff 80 fe ff ff 0f");
        *over.last_mut().unwrap() = 0x10;
        assert_eq!(decode_int(&over, 8), Err(HpackError::IntegerOverflow));
        // u32::MAX itself round-trips.
        let mut enc = Vec::new();
        encode_int(u32::MAX, 8, 0x00, &mut enc);
        assert_eq!(enc, hex("ff 80 fe ff ff 0f"));
        assert_eq!(decode_int(&enc, 8).unwrap().0, u32::MAX);
        assert_eq!(decode_int(&hex("1f 9a"), 5), Err(HpackError::Truncated));
        assert_eq!(decode_int(&[], 5), Err(HpackError::Truncated));
    }

    // --- String literals ---

    #[test]
    fn string_roundtrip_plain_and_huffman() {
        for huffman in [false, true] {
            for s in [
                "",
                "hello",
                "www.example.com",
                ":authority",
                "a".repeat(300).as_str(),
            ] {
                let mut out = Vec::new();
                encode_string(s.as_bytes(), huffman, &mut out);
                let (back, n) = decode_string(&out).unwrap();
                assert_eq!(n, out.len());
                assert_eq!(back, s.as_bytes());
            }
        }
    }

    #[test]
    fn string_huffman_flag_on_wire() {
        let mut out = Vec::new();
        encode_string(b"hi", true, &mut out);
        assert!(out[0] & 0x80 != 0);
        let mut out = Vec::new();
        encode_string(b"hi", false, &mut out);
        assert!(out[0] & 0x80 == 0);
        assert_eq!(out, b"\x02hi");
    }

    // --- Header field representations (RFC 7541 C.2) ---

    #[test]
    fn c21_literal_with_indexing() {
        let mut d = Decoder::new();
        let headers = d
            .decode(&hex(
                "400a 63757374 6f6d2d6b 6579 0d63 7573 746f6d2d 68656164 6572",
            ))
            .unwrap();
        assert_eq!(headers, fields(&[("custom-key", "custom-header")]));
        assert_eq!(d.table().len(), 1);
        assert_eq!(d.table().size(), 32 + 10 + 13);
        assert_eq!(
            d.table().get(1),
            Some((b"custom-key".as_slice(), b"custom-header".as_slice()))
        );
    }

    #[test]
    fn c22_literal_without_indexing() {
        let mut d = Decoder::new();
        let headers = d.decode(&hex("040c 2f73 616d70 6c652f 70617468")).unwrap();
        assert_eq!(headers, fields(&[(":path", "/sample/path")]));
        assert!(d.table().is_empty());
    }

    #[test]
    fn c23_literal_never_indexed() {
        let mut d = Decoder::new();
        let headers = d
            .decode(&hex("1008 70617373 776f72 6406 73656372 6574"))
            .unwrap();
        assert_eq!(headers, fields(&[("password", "secret")]));
        assert!(d.table().is_empty());
    }

    #[test]
    fn c24_indexed_header_field() {
        let mut d = Decoder::new();
        let headers = d.decode(&hex("82")).unwrap();
        assert_eq!(headers, fields(&[(":method", "GET")]));
        assert!(d.table().is_empty());
    }

    #[test]
    fn decode_rejects_index_zero() {
        let mut d = Decoder::new();
        assert_eq!(d.decode(&hex("80")), Err(HpackError::ZeroIndex));
    }

    #[test]
    fn decode_rejects_bad_index() {
        let mut d = Decoder::new();
        assert_eq!(d.decode(&hex("ff01")), Err(HpackError::BadIndex(128)));
    }

    // --- Request examples without Huffman (RFC 7541 C.3) ---

    #[test]
    fn c31_first_request() {
        let mut d = Decoder::new();
        let headers = d
            .decode(&hex("8286 8441 0f77 7777 2e65 7861 6d70 6c65 2e63 6f6d"))
            .unwrap();
        assert_eq!(
            headers,
            fields(&[
                (":method", "GET"),
                (":scheme", "http"),
                (":path", "/"),
                (":authority", "www.example.com"),
            ])
        );
        assert_eq!(d.table().len(), 1);
        assert_eq!(d.table().size(), 57);
    }

    #[test]
    fn c32_second_request() {
        let mut d = Decoder::new();
        d.decode(&hex("8286 8441 0f77 7777 2e65 7861 6d70 6c65 2e63 6f6d"))
            .unwrap();
        let headers = d.decode(&hex("8286 84be 5808 6e6f2d63 61636865")).unwrap();
        assert_eq!(
            headers,
            fields(&[
                (":method", "GET"),
                (":scheme", "http"),
                (":path", "/"),
                (":authority", "www.example.com"),
                ("cache-control", "no-cache"),
            ])
        );
        assert_eq!(d.table().len(), 2);
        assert_eq!(d.table().size(), 110);
        // Newest entry first.
        assert_eq!(
            d.table().get(1),
            Some((b"cache-control".as_slice(), b"no-cache".as_slice()))
        );
        assert_eq!(
            d.table().get(2),
            Some((b":authority".as_slice(), b"www.example.com".as_slice()))
        );
    }

    #[test]
    fn c33_third_request() {
        let mut d = Decoder::new();
        d.decode(&hex("8286 8441 0f77 7777 2e65 7861 6d70 6c65 2e63 6f6d"))
            .unwrap();
        d.decode(&hex("8286 84be 5808 6e6f2d63 61636865")).unwrap();
        let headers = d
            .decode(&hex(
                "8287 85bf 400a 6375 7374 6f6d 2d6b 6579 0c63 7573 746f6d 2d76 616c 7565",
            ))
            .unwrap();
        assert_eq!(
            headers,
            fields(&[
                (":method", "GET"),
                (":scheme", "https"),
                (":path", "/index.html"),
                (":authority", "www.example.com"),
                ("custom-key", "custom-value"),
            ])
        );
        assert_eq!(d.table().len(), 3);
        assert_eq!(d.table().size(), 164);
    }

    // --- Request examples with Huffman (RFC 7541 C.4) ---

    #[test]
    fn c41_first_request_huffman() {
        let mut d = Decoder::new();
        let headers = d
            .decode(&hex("8286 8441 8cf1 e3c2 e5f2 3a6b a0ab 90f4 ff"))
            .unwrap();
        assert_eq!(
            headers,
            fields(&[
                (":method", "GET"),
                (":scheme", "http"),
                (":path", "/"),
                (":authority", "www.example.com"),
            ])
        );
        assert_eq!(d.table().size(), 57);
    }

    #[test]
    fn c42_second_request_huffman() {
        let mut d = Decoder::new();
        d.decode(&hex("8286 8441 8cf1 e3c2 e5f2 3a6b a0ab 90f4 ff"))
            .unwrap();
        let headers = d.decode(&hex("8286 84be 5886 a8eb 1064 9cbf")).unwrap();
        assert_eq!(
            headers,
            fields(&[
                (":method", "GET"),
                (":scheme", "http"),
                (":path", "/"),
                (":authority", "www.example.com"),
                ("cache-control", "no-cache"),
            ])
        );
        assert_eq!(d.table().size(), 110);
    }

    #[test]
    fn c43_third_request_huffman() {
        let mut d = Decoder::new();
        d.decode(&hex("8286 8441 8cf1 e3c2 e5f2 3a6b a0ab 90f4 ff"))
            .unwrap();
        d.decode(&hex("8286 84be 5886 a8eb 1064 9cbf")).unwrap();
        let headers = d
            .decode(&hex(
                "8287 85bf 4088 25a8 49e9 5ba9 7d7f 8925 a849 e95b b8e8 b4bf",
            ))
            .unwrap();
        assert_eq!(
            headers,
            fields(&[
                (":method", "GET"),
                (":scheme", "https"),
                (":path", "/index.html"),
                (":authority", "www.example.com"),
                ("custom-key", "custom-value"),
            ])
        );
        assert_eq!(d.table().size(), 164);
    }

    // --- Response examples without Huffman (RFC 7541 C.5, table size 256) ---

    fn response_decoder() -> Decoder {
        let mut d = Decoder::new();
        d.set_max_table_size(256);
        d
    }

    #[test]
    fn c51_first_response() {
        let mut d = response_decoder();
        let headers = d
            .decode(&hex(
                "4803 3330 3258 0770 7269 7661 7465 611d 4d6f6e 2c20 3231 204f 6374 2032 3031 3320 3230 3a31 333a 3231 2047 4d54 6e17 6874 7470 733a 2f2f 7777 772e 6578 616d 706c 652e 636f 6d",
            ))
            .unwrap();
        assert_eq!(
            headers,
            fields(&[
                (":status", "302"),
                ("cache-control", "private"),
                ("date", "Mon, 21 Oct 2013 20:13:21 GMT"),
                ("location", "https://www.example.com"),
            ])
        );
        assert_eq!(d.table().size(), 222);
        assert_eq!(d.table().len(), 4);
    }

    #[test]
    fn c52_second_response_evicts() {
        let mut d = response_decoder();
        d.decode(&hex(
            "4803 3330 3258 0770 7269 7661 7465 611d 4d6f6e 2c20 3231 204f 6374 2032 3031 3320 3230 3a31 333a 3231 2047 4d54 6e17 6874 7470 733a 2f2f 7777 772e 6578 616d 706c 652e 636f 6d",
        ))
        .unwrap();
        let headers = d.decode(&hex("4803 3330 37c1 c0bf")).unwrap();
        assert_eq!(
            headers,
            fields(&[
                (":status", "307"),
                ("cache-control", "private"),
                ("date", "Mon, 21 Oct 2013 20:13:21 GMT"),
                ("location", "https://www.example.com"),
            ])
        );
        // (:status, 302) evicted to make room for (:status, 307).
        assert_eq!(d.table().size(), 222);
        assert_eq!(
            d.table().get(1),
            Some((b":status".as_slice(), b"307".as_slice()))
        );
        assert_eq!(d.table().len(), 4);
    }

    #[test]
    fn c53_third_response_evicts_several() {
        let mut d = response_decoder();
        d.decode(&hex(
            "4803 3330 3258 0770 7269 7661 7465 611d 4d6f6e 2c20 3231 204f 6374 2032 3031 3320 3230 3a31 333a 3231 2047 4d54 6e17 6874 7470 733a 2f2f 7777 772e 6578 616d 706c 652e 636f 6d",
        ))
        .unwrap();
        d.decode(&hex("4803 3330 37c1 c0bf")).unwrap();
        let headers = d
            .decode(&hex(
                "88c1 611d 4d6f6e 2c20 3231 204f 6374 2032 3031 3320 3230 3a31 333a 3232 2047 4d54 c05a 0467 7a69 7077 3866 6f6f 3d41 5344 4a4b 4851 4b42 5a58 4f51 5745 4f50 4955 4158 5157 454f 4955 3b20 6d61 782d 6167 653d 3336 3030 3b20 7665 7273 696f 6e3d 31",
            ))
            .unwrap();
        assert_eq!(
            headers,
            fields(&[
                (":status", "200"),
                ("cache-control", "private"),
                ("date", "Mon, 21 Oct 2013 20:13:22 GMT"),
                ("location", "https://www.example.com"),
                ("content-encoding", "gzip"),
                (
                    "set-cookie",
                    "foo=ASDJKHQKBZXOQWEOPIUAXQWEOIU; max-age=3600; version=1"
                ),
            ])
        );
        assert_eq!(d.table().size(), 215);
        assert_eq!(d.table().len(), 3);
        assert_eq!(d.table().get(1).unwrap().0, b"set-cookie".as_slice());
    }

    // --- Response examples with Huffman (RFC 7541 C.6) ---

    #[test]
    fn c61_first_response_huffman() {
        let mut d = response_decoder();
        let headers = d
            .decode(&hex(
                "4882 6402 5885 aec3 771a 4b61 96d0 7abe 9410 54d4 44a8 2005 9504 0b81 66e0 82a6 2d1b ff6e 919d 29ad 1718 63c7 8f0b 97c8 e9ae 82ae 43d3",
            ))
            .unwrap();
        assert_eq!(
            headers,
            fields(&[
                (":status", "302"),
                ("cache-control", "private"),
                ("date", "Mon, 21 Oct 2013 20:13:21 GMT"),
                ("location", "https://www.example.com"),
            ])
        );
        assert_eq!(d.table().size(), 222);
    }

    #[test]
    fn c62_second_response_huffman() {
        let mut d = response_decoder();
        d.decode(&hex(
            "4882 6402 5885 aec3 771a 4b61 96d0 7abe 9410 54d4 44a8 2005 9504 0b81 66e0 82a6 2d1b ff6e 919d 29ad 1718 63c7 8f0b 97c8 e9ae 82ae 43d3",
        ))
        .unwrap();
        let headers = d.decode(&hex("4883 640e ffc1 c0bf")).unwrap();
        assert_eq!(
            headers,
            fields(&[
                (":status", "307"),
                ("cache-control", "private"),
                ("date", "Mon, 21 Oct 2013 20:13:21 GMT"),
                ("location", "https://www.example.com"),
            ])
        );
        assert_eq!(d.table().size(), 222);
    }

    #[test]
    fn c63_third_response_huffman() {
        let mut d = response_decoder();
        d.decode(&hex(
            "4882 6402 5885 aec3 771a 4b61 96d0 7abe 9410 54d4 44a8 2005 9504 0b81 66e0 82a6 2d1b ff6e 919d 29ad 1718 63c7 8f0b 97c8 e9ae 82ae 43d3",
        ))
        .unwrap();
        d.decode(&hex("4883 640e ffc1 c0bf")).unwrap();
        let headers = d
            .decode(&hex(
                "88c1 6196 d07a be94 1054 d444 a820 0595 040b 8166 e084 a62d 1bff c05a 839b d9ab 77ad 94e7 821d d7f2 e6c7 b335 dfdf cd5b 3960 d5af 2708 7f36 72c1 ab27 0fb5 291f 9587 3160 65c0 03ed 4ee5 b106 3d50 07",
            ))
            .unwrap();
        assert_eq!(
            headers,
            fields(&[
                (":status", "200"),
                ("cache-control", "private"),
                ("date", "Mon, 21 Oct 2013 20:13:22 GMT"),
                ("location", "https://www.example.com"),
                ("content-encoding", "gzip"),
                (
                    "set-cookie",
                    "foo=ASDJKHQKBZXOQWEOPIUAXQWEOIU; max-age=3600; version=1"
                ),
            ])
        );
        assert_eq!(d.table().size(), 215);
        assert_eq!(d.table().len(), 3);
    }

    // --- Table size update rules ---

    #[test]
    fn size_update_must_lead_the_block() {
        let mut d = Decoder::new();
        // Indexed field first, then a size update.
        assert_eq!(
            d.decode(&hex("82 3f00")),
            Err(HpackError::SizeUpdateNotAtStart)
        );
    }

    #[test]
    fn size_update_at_start_ok() {
        let mut d = Decoder::new();
        // Max size 0, then :method: GET.
        let headers = d.decode(&hex("20 82")).unwrap();
        assert_eq!(headers, fields(&[(":method", "GET")]));
        assert_eq!(d.table().max_size(), 0);
    }

    #[test]
    fn size_update_above_protocol_max_rejected() {
        let mut d = Decoder::new();
        d.set_max_table_size(256);
        // 0x3f1f = 31 + 256*... : 0x3f then continuation 0x01 -> 31+128=159? compute: prefix 5 max 31, then 0x01 -> 159. Use a bigger one:
        assert_eq!(
            d.decode(&hex("3fe1 0f")),
            Err(HpackError::TableSizeTooLarge)
        );
    }

    #[test]
    fn dynamic_table_evicts_oldest_first() {
        let mut t = DynamicTable::new();
        t.set_max_size(100);
        // entry size = 32 + 1 + 1 = 34.
        t.insert(b"a".to_vec(), b"1".to_vec());
        t.insert(b"b".to_vec(), b"2".to_vec());
        assert_eq!(t.len(), 2);
        t.insert(b"c".to_vec(), b"3".to_vec()); // 102 > 100: evict "a".
        assert_eq!(t.len(), 2);
        assert_eq!(t.get(2), Some((b"b".as_slice(), b"2".as_slice())));
        assert_eq!(t.get(1), Some((b"c".as_slice(), b"3".as_slice())));
    }

    #[test]
    fn oversized_entry_empties_table() {
        let mut t = DynamicTable::new();
        t.set_max_size(64);
        t.insert(b"a".to_vec(), b"1".to_vec());
        assert_eq!(t.len(), 1);
        // 32 + 40 + 40 = 112 > 64: table emptied, nothing added.
        t.insert(vec![b'x'; 40], vec![b'y'; 40]);
        assert!(t.is_empty());
        assert_eq!(t.size(), 0);
    }

    // --- Encoder ---

    #[test]
    fn encoder_matches_rfc_c31_and_c41_byte_for_byte() {
        // RFC 7541 C.3.1 (no Huffman) and C.4.1 (Huffman): the encoder's
        // strategy (indexed on full match, literal with incremental indexing
        // otherwise) reproduces the RFC's bytes exactly.
        let headers = fields(&[
            (":method", "GET"),
            (":scheme", "http"),
            (":path", "/"),
            (":authority", "www.example.com"),
        ]);
        let mut e = Encoder::new();
        e.set_huffman(false);
        let mut out = Vec::new();
        e.encode(&headers, &mut out);
        assert_eq!(
            out,
            hex("8286 8441 0f77 7777 2e65 7861 6d70 6c65 2e63 6f6d")
        );

        let mut e = Encoder::new();
        e.set_huffman(true);
        let mut out = Vec::new();
        e.encode(&headers, &mut out);
        assert_eq!(out, hex("8286 8441 8cf1 e3c2 e5f2 3a6b a0ab 90f4 ff"));
    }

    #[test]
    fn encoder_uses_indexed_for_static_hits() {
        let mut e = Encoder::new();
        e.set_huffman(false);
        let mut out = Vec::new();
        e.encode(&fields(&[(":method", "GET"), (":path", "/")]), &mut out);
        assert_eq!(out, vec![0x82, 0x84]);
    }

    #[test]
    fn encoder_literal_with_indexed_name() {
        // :authority is static index 1 with empty value; new value ->
        // literal with incremental indexing, like RFC C.3.1.
        let mut e = Encoder::new();
        e.set_huffman(false);
        let mut out = Vec::new();
        e.encode(&fields(&[(":authority", "www.example.com")]), &mut out);
        let mut expect = vec![0x41, 0x0f];
        expect.extend_from_slice(b"www.example.com");
        assert_eq!(out, expect);
        // And the decoder agrees, with the dynamic table updated.
        let mut d = Decoder::new();
        let back = d.decode(&out).unwrap();
        assert_eq!(back, fields(&[(":authority", "www.example.com")]));
        assert_eq!(d.table().size(), 57);
    }

    #[test]
    fn encoder_decoder_roundtrip() {
        let headers = fields(&[
            (":method", "POST"),
            (":scheme", "https"),
            (":path", "/submit?x=1&y=2"),
            (":authority", "example.org"),
            ("content-type", "application/json"),
            ("x-custom", "some value here"),
            ("cookie", "a=b; c=d"),
        ]);
        for huffman in [false, true] {
            let mut e = Encoder::new();
            e.set_huffman(huffman);
            let mut d = Decoder::new();
            for _ in 0..3 {
                let mut out = Vec::new();
                e.encode(&headers, &mut out);
                let back = d.decode(&out).unwrap();
                assert_eq!(back, headers, "huffman={huffman}");
            }
            // Second and third blocks reuse dynamic table entries (indexed).
            assert!(!d.table().is_empty());
        }
    }

    #[test]
    fn encoder_emits_size_update_first() {
        let mut e = Encoder::new();
        e.set_huffman(false);
        e.set_max_table_size(256);
        let mut out = Vec::new();
        e.encode(&fields(&[(":method", "GET")]), &mut out);
        // 256 = 31 + 225, and 225 = 97 + 1*128.
        assert_eq!(&out[..3], &[0x3f, 0xe1, 0x01]);
        assert_eq!(&out[3..], &[0x82]);
        assert_eq!(e.table().max_size(), 256);
    }

    #[test]
    fn encoder_size_update_clamped_to_protocol_max() {
        let mut e = Encoder::new();
        e.set_huffman(false);
        e.set_protocol_max(128);
        e.set_max_table_size(4096); // clamped to 128
        let mut out = Vec::new();
        e.encode(&fields(&[(":method", "GET")]), &mut out);
        // 128 = 31 + 97.
        assert_eq!(&out[..2], &[0x3f, 0x61]);
        assert_eq!(e.table().max_size(), 128);
    }

    #[test]
    fn static_table_has_61_entries() {
        assert_eq!(STATIC_TABLE.len(), 61);
        assert_eq!(STATIC_TABLE[0], (":authority", ""));
        assert_eq!(STATIC_TABLE[1], (":method", "GET"));
        assert_eq!(STATIC_TABLE[60], ("www-authenticate", ""));
    }
}
