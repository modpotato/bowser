//! HTTP/2 frame codec: all ten frame types (RFC 9113, Section 6).
//!
//! [`Frame::decode`] parses one frame from the front of a byte buffer and
//! reports how many bytes were consumed, so a driver can loop over a read
//! buffer. [`Frame::encode`] serializes a frame. Unknown frame types are
//! preserved as [`Payload::Unknown`] so the driver can ignore them, as the
//! RFC requires for extensibility.

use crate::h2::settings::Setting;

/// Length of a frame header in bytes.
pub const HEADER_LEN: usize = 9;
/// Largest value expressible in the 24-bit length field.
pub const MAX_PAYLOAD_LEN: u32 = 0x00ff_ffff;

// Flag bits. Names are per-frame-type aliases of the same bit positions.
pub const FLAG_END_STREAM: u8 = 0x1;
pub const FLAG_ACK: u8 = 0x1;
pub const FLAG_END_HEADERS: u8 = 0x4;
pub const FLAG_PADDED: u8 = 0x8;
pub const FLAG_PRIORITY: u8 = 0x20;

/// Frame type codes (RFC 9113, Section 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FrameType {
    Data = 0x0,
    Headers = 0x1,
    Priority = 0x2,
    RstStream = 0x3,
    Settings = 0x4,
    PushPromise = 0x5,
    Ping = 0x6,
    GoAway = 0x7,
    WindowUpdate = 0x8,
    Continuation = 0x9,
}

impl FrameType {
    /// Map a type byte to a [`FrameType`]; unknown types yield `None`.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x0 => Some(FrameType::Data),
            0x1 => Some(FrameType::Headers),
            0x2 => Some(FrameType::Priority),
            0x3 => Some(FrameType::RstStream),
            0x4 => Some(FrameType::Settings),
            0x5 => Some(FrameType::PushPromise),
            0x6 => Some(FrameType::Ping),
            0x7 => Some(FrameType::GoAway),
            0x8 => Some(FrameType::WindowUpdate),
            0x9 => Some(FrameType::Continuation),
            _ => None,
        }
    }

    /// Frames of this type are only valid on stream zero (connection-level).
    fn connection_level(self) -> bool {
        matches!(
            self,
            FrameType::Settings | FrameType::Ping | FrameType::GoAway
        )
    }

    /// Frames of this type are only valid on non-zero streams.
    fn stream_level(self) -> bool {
        matches!(
            self,
            FrameType::Data
                | FrameType::Headers
                | FrameType::Priority
                | FrameType::RstStream
                | FrameType::PushPromise
                | FrameType::Continuation
        )
    }
}

/// HTTP/2 error codes (RFC 9113, Section 7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ErrorCode {
    NoError = 0x0,
    ProtocolError = 0x1,
    InternalError = 0x2,
    FlowControlError = 0x3,
    SettingsTimeout = 0x4,
    StreamClosed = 0x5,
    FrameSizeError = 0x6,
    RefusedStream = 0x7,
    Cancel = 0x8,
    CompressionError = 0x9,
    ConnectError = 0xa,
    EnhanceYourCalm = 0xb,
    InadequateSecurity = 0xc,
    Http11Required = 0xd,
}

impl ErrorCode {
    /// Map a code to [`ErrorCode`]; unknown codes become
    /// [`ErrorCode::InternalError`], per RFC 9113 Section 7.
    pub fn from_u32(v: u32) -> Self {
        match v {
            0x0 => ErrorCode::NoError,
            0x1 => ErrorCode::ProtocolError,
            0x2 => ErrorCode::InternalError,
            0x3 => ErrorCode::FlowControlError,
            0x4 => ErrorCode::SettingsTimeout,
            0x5 => ErrorCode::StreamClosed,
            0x6 => ErrorCode::FrameSizeError,
            0x7 => ErrorCode::RefusedStream,
            0x8 => ErrorCode::Cancel,
            0x9 => ErrorCode::CompressionError,
            0xa => ErrorCode::ConnectError,
            0xb => ErrorCode::EnhanceYourCalm,
            0xc => ErrorCode::InadequateSecurity,
            0xd => ErrorCode::Http11Required,
            _ => ErrorCode::InternalError,
        }
    }

    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Errors from frame decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameError {
    /// Fewer than 9 header bytes, or the payload is shorter than declared.
    Truncated,
    /// The reserved high bit of a stream identifier was set.
    ReservedBitSet,
    /// Frame on the wrong stream (e.g. SETTINGS on a non-zero stream).
    BadStreamId,
    /// Payload length is illegal for this frame type.
    BadLength,
    /// PADDED length exceeds the payload.
    BadPadding,
    /// WINDOW_UPDATE with a zero increment.
    ZeroWindowIncrement,
    /// Priority weight out of range 1..=256.
    BadWeight,
}

/// A decoded 9-byte frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Payload length (24-bit).
    pub length: u32,
    pub ty: FrameType,
    pub flags: u8,
    /// Stream identifier (31-bit; reserved bit must be zero).
    pub stream_id: u32,
}

impl Header {
    pub fn encode(&self, out: &mut Vec<u8>) {
        debug_assert!(self.length <= MAX_PAYLOAD_LEN);
        debug_assert!(self.stream_id < 0x8000_0000);
        out.push((self.length >> 16) as u8);
        out.push((self.length >> 8) as u8);
        out.push(self.length as u8);
        out.push(self.ty as u8);
        out.push(self.flags);
        out.extend_from_slice(&self.stream_id.to_be_bytes());
    }

    /// Decode a header from exactly the first 9 bytes of `buf`.
    pub fn decode(buf: &[u8]) -> Result<Self, FrameError> {
        let b: &[u8; HEADER_LEN] = buf
            .get(..HEADER_LEN)
            .and_then(|s| s.try_into().ok())
            .ok_or(FrameError::Truncated)?;
        let length = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        let ty = FrameType::from_u8(b[3]).ok_or(FrameError::BadLength)?;
        // Unknown frame types are handled by Frame::decode, which keeps the
        // raw type byte; Header::decode rejects them so callers that only
        // accept known types get an error. Frame::decode never calls this
        // with an unknown type byte.
        let stream_id = u32::from_be_bytes([b[5], b[6], b[7], b[8]]);
        if stream_id & 0x8000_0000 != 0 {
            return Err(FrameError::ReservedBitSet);
        }
        Ok(Header {
            length,
            ty,
            flags: b[4],
            stream_id,
        })
    }
}

/// Stream dependency and weight from HEADERS/PRIORITY (RFC 9113, 6.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrioritySpec {
    pub exclusive: bool,
    /// 31-bit stream identifier this stream depends on.
    pub stream_dependency: u32,
    /// Weight in 1..=256.
    pub weight: u16,
}

impl PrioritySpec {
    pub fn new(exclusive: bool, stream_dependency: u32, weight: u16) -> Result<Self, FrameError> {
        if stream_dependency & 0x8000_0000 != 0 {
            return Err(FrameError::ReservedBitSet);
        }
        if !(1..=256).contains(&weight) {
            return Err(FrameError::BadWeight);
        }
        Ok(PrioritySpec {
            exclusive,
            stream_dependency,
            weight,
        })
    }

    fn decode(buf: &[u8]) -> Result<Self, FrameError> {
        let b: &[u8; 5] = buf
            .get(..5)
            .and_then(|s| s.try_into().ok())
            .ok_or(FrameError::Truncated)?;
        let raw_dep = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        PrioritySpec::new(
            raw_dep & 0x8000_0000 != 0,
            raw_dep & 0x7fff_ffff,
            u16::from(b[4]) + 1,
        )
    }

    fn encode(&self, out: &mut Vec<u8>) {
        let mut dep = self.stream_dependency;
        if self.exclusive {
            dep |= 0x8000_0000;
        }
        out.extend_from_slice(&dep.to_be_bytes());
        out.push((self.weight - 1) as u8);
    }
}

/// A frame payload. Lifetimes borrow the decode buffer (zero-copy).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Payload<'a> {
    Data {
        data: &'a [u8],
        end_stream: bool,
    },
    Headers {
        fragment: &'a [u8],
        end_stream: bool,
        end_headers: bool,
        priority: Option<PrioritySpec>,
    },
    Priority(PrioritySpec),
    RstStream {
        error: ErrorCode,
    },
    Settings {
        ack: bool,
        params: Vec<Setting>,
    },
    PushPromise {
        promised_id: u32,
        end_headers: bool,
        fragment: &'a [u8],
    },
    Ping {
        ack: bool,
        opaque: [u8; 8],
    },
    GoAway {
        last_stream_id: u32,
        error: ErrorCode,
        debug_data: &'a [u8],
    },
    WindowUpdate {
        /// 1..=2^31-1.
        increment: u32,
    },
    Continuation {
        end_headers: bool,
        fragment: &'a [u8],
    },
    /// An extension frame type we do not implement; carried so the
    /// connection can ignore it per RFC 9113 Section 5.5.
    Unknown {
        ty: u8,
        flags: u8,
        data: &'a [u8],
    },
}

/// A complete frame: stream id plus payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame<'a> {
    pub stream_id: u32,
    pub payload: Payload<'a>,
}

impl<'a> Frame<'a> {
    /// The frame type, or `None` for an unknown extension type.
    pub fn ty(&self) -> Option<FrameType> {
        match &self.payload {
            Payload::Data { .. } => Some(FrameType::Data),
            Payload::Headers { .. } => Some(FrameType::Headers),
            Payload::Priority(_) => Some(FrameType::Priority),
            Payload::RstStream { .. } => Some(FrameType::RstStream),
            Payload::Settings { .. } => Some(FrameType::Settings),
            Payload::PushPromise { .. } => Some(FrameType::PushPromise),
            Payload::Ping { .. } => Some(FrameType::Ping),
            Payload::GoAway { .. } => Some(FrameType::GoAway),
            Payload::WindowUpdate { .. } => Some(FrameType::WindowUpdate),
            Payload::Continuation { .. } => Some(FrameType::Continuation),
            Payload::Unknown { .. } => None,
        }
    }

    /// The type byte as it appears on the wire.
    pub fn raw_ty(&self) -> u8 {
        match &self.payload {
            Payload::Unknown { ty, .. } => *ty,
            _ => self.ty().unwrap() as u8,
        }
    }

    /// Flags implied by the payload.
    pub fn flags(&self) -> u8 {
        match &self.payload {
            Payload::Data { end_stream, .. } => {
                if *end_stream {
                    FLAG_END_STREAM
                } else {
                    0
                }
            }
            Payload::Headers {
                end_stream,
                end_headers,
                priority,
                ..
            } => {
                (if *end_stream { FLAG_END_STREAM } else { 0 })
                    | (if *end_headers { FLAG_END_HEADERS } else { 0 })
                    | (if priority.is_some() { FLAG_PRIORITY } else { 0 })
            }
            Payload::Priority(_) => 0,
            Payload::RstStream { .. } => 0,
            Payload::Settings { ack, .. } => {
                if *ack {
                    FLAG_ACK
                } else {
                    0
                }
            }
            Payload::PushPromise { end_headers, .. } => {
                if *end_headers {
                    FLAG_END_HEADERS
                } else {
                    0
                }
            }
            Payload::Ping { ack, .. } => {
                if *ack {
                    FLAG_ACK
                } else {
                    0
                }
            }
            Payload::GoAway { .. } => 0,
            Payload::WindowUpdate { .. } => 0,
            Payload::Continuation { end_headers, .. } => {
                if *end_headers {
                    FLAG_END_HEADERS
                } else {
                    0
                }
            }
            Payload::Unknown { flags, .. } => *flags,
        }
    }

    /// Serialize the frame (header + payload) into `out`. Padding is never
    /// emitted; it is optional on the wire.
    pub fn encode(&self, out: &mut Vec<u8>) {
        let mut payload = Vec::new();
        match &self.payload {
            Payload::Data { data, .. } => payload.extend_from_slice(data),
            Payload::Headers {
                fragment, priority, ..
            } => {
                if let Some(p) = priority {
                    p.encode(&mut payload);
                }
                payload.extend_from_slice(fragment);
            }
            Payload::Priority(p) => p.encode(&mut payload),
            Payload::RstStream { error } => {
                payload.extend_from_slice(&error.as_u32().to_be_bytes())
            }
            Payload::Settings { ack, params } => {
                debug_assert!(!ack || params.is_empty());
                for s in params {
                    payload.extend_from_slice(&s.id.to_be_bytes());
                    payload.extend_from_slice(&s.value.to_be_bytes());
                }
            }
            Payload::PushPromise {
                promised_id,
                fragment,
                ..
            } => {
                debug_assert!(*promised_id < 0x8000_0000);
                payload.extend_from_slice(&promised_id.to_be_bytes());
                payload.extend_from_slice(fragment);
            }
            Payload::Ping { opaque, .. } => payload.extend_from_slice(opaque),
            Payload::GoAway {
                last_stream_id,
                error,
                debug_data,
            } => {
                debug_assert!(*last_stream_id < 0x8000_0000);
                payload.extend_from_slice(&last_stream_id.to_be_bytes());
                payload.extend_from_slice(&error.as_u32().to_be_bytes());
                payload.extend_from_slice(debug_data);
            }
            Payload::WindowUpdate { increment } => {
                debug_assert!(*increment >= 1 && *increment <= 0x7fff_ffff);
                payload.extend_from_slice(&increment.to_be_bytes());
            }
            Payload::Continuation { fragment, .. } => payload.extend_from_slice(fragment),
            Payload::Unknown {
                ty, flags, data, ..
            } => {
                // Re-emit verbatim: bypass the typed header path.
                debug_assert!(data.len() <= MAX_PAYLOAD_LEN as usize);
                debug_assert!(self.stream_id < 0x8000_0000);
                out.push((data.len() >> 16) as u8);
                out.push((data.len() >> 8) as u8);
                out.push(data.len() as u8);
                out.push(*ty);
                out.push(*flags);
                out.extend_from_slice(&self.stream_id.to_be_bytes());
                out.extend_from_slice(data);
                return;
            }
        }
        debug_assert!(payload.len() <= MAX_PAYLOAD_LEN as usize);
        Header {
            length: payload.len() as u32,
            ty: self.ty().expect("known frame type"),
            flags: self.flags(),
            stream_id: self.stream_id,
        }
        .encode(out);
        out.extend_from_slice(&payload);
    }

    /// Decode one frame from the front of `buf`.
    ///
    /// Returns the frame and the total bytes consumed (header + payload).
    /// Returns [`FrameError::Truncated`] if `buf` does not yet hold the full
    /// frame; the caller should read more and retry.
    pub fn decode(buf: &'a [u8]) -> Result<(Self, usize), FrameError> {
        let raw = RawHeader::decode(buf)?;
        let total = HEADER_LEN + raw.length as usize;
        let payload = buf.get(HEADER_LEN..total).ok_or(FrameError::Truncated)?;

        // Stream-id placement rules (RFC 9113, Section 6).
        if let Some(ty) = FrameType::from_u8(raw.ty) {
            if ty.connection_level() && raw.stream_id != 0 {
                return Err(FrameError::BadStreamId);
            }
            if ty.stream_level() && raw.stream_id == 0 {
                return Err(FrameError::BadStreamId);
            }
        }

        let frame_payload = match FrameType::from_u8(raw.ty) {
            None => Payload::Unknown {
                ty: raw.ty,
                flags: raw.flags,
                data: payload,
            },
            Some(FrameType::Data) => {
                let data = depad(payload, raw.flags)?;
                Payload::Data {
                    data,
                    end_stream: raw.flags & FLAG_END_STREAM != 0,
                }
            }
            Some(FrameType::Headers) => {
                let mut rest = depad(payload, raw.flags)?;
                let priority = if raw.flags & FLAG_PRIORITY != 0 {
                    let p = PrioritySpec::decode(rest)?;
                    rest = &rest[5..];
                    Some(p)
                } else {
                    None
                };
                Payload::Headers {
                    fragment: rest,
                    end_stream: raw.flags & FLAG_END_STREAM != 0,
                    end_headers: raw.flags & FLAG_END_HEADERS != 0,
                    priority,
                }
            }
            Some(FrameType::Priority) => {
                if payload.len() != 5 {
                    return Err(FrameError::BadLength);
                }
                Payload::Priority(PrioritySpec::decode(payload)?)
            }
            Some(FrameType::RstStream) => {
                if payload.len() != 4 {
                    return Err(FrameError::BadLength);
                }
                Payload::RstStream {
                    error: ErrorCode::from_u32(u32::from_be_bytes(payload.try_into().unwrap())),
                }
            }
            Some(FrameType::Settings) => {
                let ack = raw.flags & FLAG_ACK != 0;
                if ack && !payload.is_empty() {
                    return Err(FrameError::BadLength);
                }
                if !payload.len().is_multiple_of(6) {
                    return Err(FrameError::BadLength);
                }
                let mut params = Vec::with_capacity(payload.len() / 6);
                let (chunks, _) = payload.as_chunks::<6>();
                for chunk in chunks {
                    params.push(Setting {
                        id: u16::from_be_bytes([chunk[0], chunk[1]]),
                        value: u32::from_be_bytes([chunk[2], chunk[3], chunk[4], chunk[5]]),
                    });
                }
                Payload::Settings { ack, params }
            }
            Some(FrameType::PushPromise) => {
                let rest = depad(payload, raw.flags)?;
                let raw_id: &[u8; 4] = rest
                    .get(..4)
                    .and_then(|s| s.try_into().ok())
                    .ok_or(FrameError::Truncated)?;
                let promised_id = u32::from_be_bytes(*raw_id);
                if promised_id & 0x8000_0000 != 0 {
                    return Err(FrameError::ReservedBitSet);
                }
                Payload::PushPromise {
                    promised_id,
                    end_headers: raw.flags & FLAG_END_HEADERS != 0,
                    fragment: &rest[4..],
                }
            }
            Some(FrameType::Ping) => {
                if payload.len() != 8 {
                    return Err(FrameError::BadLength);
                }
                Payload::Ping {
                    ack: raw.flags & FLAG_ACK != 0,
                    opaque: payload.try_into().unwrap(),
                }
            }
            Some(FrameType::GoAway) => {
                if payload.len() < 8 {
                    return Err(FrameError::BadLength);
                }
                let last = u32::from_be_bytes(payload[..4].try_into().unwrap());
                if last & 0x8000_0000 != 0 {
                    return Err(FrameError::ReservedBitSet);
                }
                Payload::GoAway {
                    last_stream_id: last,
                    error: ErrorCode::from_u32(u32::from_be_bytes(
                        payload[4..8].try_into().unwrap(),
                    )),
                    debug_data: &payload[8..],
                }
            }
            Some(FrameType::WindowUpdate) => {
                if payload.len() != 4 {
                    return Err(FrameError::BadLength);
                }
                let increment = u32::from_be_bytes(payload.try_into().unwrap()) & 0x7fff_ffff;
                if increment == 0 {
                    return Err(FrameError::ZeroWindowIncrement);
                }
                Payload::WindowUpdate { increment }
            }
            Some(FrameType::Continuation) => Payload::Continuation {
                end_headers: raw.flags & FLAG_END_HEADERS != 0,
                fragment: payload,
            },
        };

        Ok((
            Frame {
                stream_id: raw.stream_id,
                payload: frame_payload,
            },
            total,
        ))
    }

    /// Encode a frame of an unknown (extension) type verbatim.
    pub fn encode_raw(ty: u8, flags: u8, stream_id: u32, payload: &[u8], out: &mut Vec<u8>) {
        debug_assert!(FrameType::from_u8(ty).is_none());
        debug_assert!(payload.len() <= MAX_PAYLOAD_LEN as usize);
        debug_assert!(stream_id < 0x8000_0000);
        out.push((payload.len() >> 16) as u8);
        out.push((payload.len() >> 8) as u8);
        out.push(payload.len() as u8);
        out.push(ty);
        out.push(flags);
        out.extend_from_slice(&stream_id.to_be_bytes());
        out.extend_from_slice(payload);
    }
}

/// Header decode that tolerates unknown frame types (for [`Frame::decode`]).
struct RawHeader {
    length: u32,
    ty: u8,
    flags: u8,
    stream_id: u32,
}

impl RawHeader {
    fn decode(buf: &[u8]) -> Result<Self, FrameError> {
        let b: &[u8; HEADER_LEN] = buf
            .get(..HEADER_LEN)
            .and_then(|s| s.try_into().ok())
            .ok_or(FrameError::Truncated)?;
        let stream_id = u32::from_be_bytes([b[5], b[6], b[7], b[8]]);
        if stream_id & 0x8000_0000 != 0 {
            return Err(FrameError::ReservedBitSet);
        }
        Ok(RawHeader {
            length: (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]),
            ty: b[3],
            flags: b[4],
            stream_id,
        })
    }
}

/// Strip padding (RFC 9113, 6.1). Returns the data between the pad-length
/// octet and the trailing padding.
fn depad(payload: &[u8], flags: u8) -> Result<&[u8], FrameError> {
    if flags & FLAG_PADDED == 0 {
        return Ok(payload);
    }
    let pad_len = *payload.first().ok_or(FrameError::Truncated)? as usize;
    // The pad-length octet plus pad_len padding octets must leave data.
    if pad_len >= payload.len() {
        return Err(FrameError::BadPadding);
    }
    Ok(&payload[1..payload.len() - pad_len])
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

    fn assert_roundtrip(frame: &Frame) {
        let mut buf = Vec::new();
        frame.encode(&mut buf);
        let (back, consumed) = Frame::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(back.stream_id, frame.stream_id);
        assert_eq!(back.raw_ty(), frame.raw_ty());
        assert_eq!(back.flags(), frame.flags());
        assert_eq!(back.payload, frame.payload);
    }

    #[test]
    fn header_roundtrip() {
        let h = Header {
            length: 0x123456,
            ty: FrameType::Headers,
            flags: FLAG_END_HEADERS | FLAG_END_STREAM,
            stream_id: 7,
        };
        let mut buf = Vec::new();
        h.encode(&mut buf);
        assert_eq!(buf, hex("123456 01 05 00000007"));
        assert_eq!(Header::decode(&buf).unwrap(), h);
    }

    #[test]
    fn header_rejects_short_buffer() {
        assert_eq!(
            Header::decode(&hex("123456 01")),
            Err(FrameError::Truncated)
        );
    }

    #[test]
    fn header_rejects_reserved_bit() {
        assert_eq!(
            Header::decode(&hex("000000 06 00 80000000")),
            Err(FrameError::ReservedBitSet)
        );
    }

    #[test]
    fn header_rejects_unknown_type() {
        // Header::decode is for known types only; Frame::decode keeps unknown.
        assert_eq!(
            Header::decode(&hex("000000 ff 00 00000000")),
            Err(FrameError::BadLength)
        );
    }

    #[test]
    fn data_roundtrip() {
        let f = Frame {
            stream_id: 1,
            payload: Payload::Data {
                data: b"hello",
                end_stream: true,
            },
        };
        assert_roundtrip(&f);
    }

    #[test]
    fn data_with_padding_decodes() {
        // len=7, PADDED: padlen=3, "abc", 3 zero pad bytes.
        let buf = hex("000007 00 08 00000001 03 616263 000000");
        let (f, consumed) = Frame::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(
            f.payload,
            Payload::Data {
                data: b"abc",
                end_stream: false
            }
        );
    }

    #[test]
    fn data_bad_padding_rejected() {
        // pad length 8 >= payload length 8.
        let buf = hex("000008 00 08 00000001 08 6162636465 0000");
        assert_eq!(Frame::decode(&buf), Err(FrameError::BadPadding));
        // empty payload with PADDED: no pad-length octet at all.
        let buf = hex("000000 00 08 00000001");
        assert_eq!(Frame::decode(&buf), Err(FrameError::Truncated));
    }

    #[test]
    fn data_on_stream_zero_rejected() {
        let buf = hex("000005 00 00 00000000 68656c6c6f");
        assert_eq!(Frame::decode(&buf), Err(FrameError::BadStreamId));
    }

    #[test]
    fn headers_with_priority_roundtrip() {
        let f = Frame {
            stream_id: 3,
            payload: Payload::Headers {
                fragment: b"\x88",
                end_stream: true,
                end_headers: true,
                priority: Some(PrioritySpec::new(true, 5, 256).unwrap()),
            },
        };
        assert_roundtrip(&f);
        // wire: priority spec first (exclusive dep 5, weight 256 -> 0xff).
        let mut buf = Vec::new();
        f.encode(&mut buf);
        assert_eq!(&buf[9..14], &hex("80000005 ff"));
        assert_eq!(buf[14], 0x88);
    }

    #[test]
    fn headers_on_stream_zero_rejected() {
        let buf = hex("000001 01 04 00000000 88");
        assert_eq!(Frame::decode(&buf), Err(FrameError::BadStreamId));
    }

    #[test]
    fn priority_roundtrip() {
        let f = Frame {
            stream_id: 9,
            payload: Payload::Priority(PrioritySpec::new(false, 0, 16).unwrap()),
        };
        assert_roundtrip(&f);
    }

    #[test]
    fn priority_bad_length_rejected() {
        let buf = hex("000004 02 00 00000001 80000005");
        assert_eq!(Frame::decode(&buf), Err(FrameError::BadLength));
    }

    #[test]
    fn priority_spec_validation() {
        assert_eq!(PrioritySpec::new(false, 0, 0), Err(FrameError::BadWeight));
        assert_eq!(
            PrioritySpec::new(false, 0x8000_0001, 1),
            Err(FrameError::ReservedBitSet)
        );
    }

    #[test]
    fn rst_stream_roundtrip() {
        let f = Frame {
            stream_id: 5,
            payload: Payload::RstStream {
                error: ErrorCode::Cancel,
            },
        };
        assert_roundtrip(&f);
        let mut buf = Vec::new();
        f.encode(&mut buf);
        assert_eq!(buf, hex("000004 03 00 00000005 00000008"));
    }

    #[test]
    fn rst_stream_bad_length_rejected() {
        let buf = hex("000003 03 00 00000005 000008");
        assert_eq!(Frame::decode(&buf), Err(FrameError::BadLength));
    }

    #[test]
    fn unknown_error_code_maps_to_internal() {
        assert_eq!(ErrorCode::from_u32(0xdead), ErrorCode::InternalError);
        assert_eq!(ErrorCode::from_u32(0x8).as_u32(), 0x8);
    }

    #[test]
    fn settings_roundtrip() {
        let f = Frame {
            stream_id: 0,
            payload: Payload::Settings {
                ack: false,
                params: vec![
                    Setting {
                        id: 0x4,
                        value: 65535,
                    },
                    Setting {
                        id: 0x5,
                        value: 16384,
                    },
                    Setting { id: 0x99, value: 1 },
                ],
            },
        };
        assert_roundtrip(&f);
    }

    #[test]
    fn settings_ack_roundtrip() {
        let f = Frame {
            stream_id: 0,
            payload: Payload::Settings {
                ack: true,
                params: vec![],
            },
        };
        let mut buf = Vec::new();
        f.encode(&mut buf);
        assert_eq!(buf, hex("000000 04 01 00000000"));
        assert_roundtrip(&f);
    }

    #[test]
    fn settings_ack_with_payload_rejected() {
        let buf = hex("000006 04 01 00000000 00040000ffff");
        assert_eq!(Frame::decode(&buf), Err(FrameError::BadLength));
    }

    #[test]
    fn settings_bad_length_rejected() {
        let buf = hex("000005 04 00 00000000 00040000ff");
        assert_eq!(Frame::decode(&buf), Err(FrameError::BadLength));
    }

    #[test]
    fn settings_on_stream_rejected() {
        let buf = hex("000000 04 00 00000001");
        assert_eq!(Frame::decode(&buf), Err(FrameError::BadStreamId));
    }

    #[test]
    fn push_promise_roundtrip() {
        let f = Frame {
            stream_id: 1,
            payload: Payload::PushPromise {
                promised_id: 2,
                end_headers: true,
                fragment: b"\x88",
            },
        };
        assert_roundtrip(&f);
    }

    #[test]
    fn push_promise_reserved_bit_rejected() {
        let buf = hex("000005 05 04 00000001 80000002 88");
        assert_eq!(Frame::decode(&buf), Err(FrameError::ReservedBitSet));
    }

    #[test]
    fn ping_roundtrip() {
        let f = Frame {
            stream_id: 0,
            payload: Payload::Ping {
                ack: false,
                opaque: *b"12345678",
            },
        };
        let mut buf = Vec::new();
        f.encode(&mut buf);
        assert_eq!(buf, hex("000008 06 00 00000000 3132333435363738"));
        assert_roundtrip(&f);
    }

    #[test]
    fn ping_ack_flag() {
        let f = Frame {
            stream_id: 0,
            payload: Payload::Ping {
                ack: true,
                opaque: [0; 8],
            },
        };
        let mut buf = Vec::new();
        f.encode(&mut buf);
        assert_eq!(buf[4], FLAG_ACK);
    }

    #[test]
    fn ping_bad_length_rejected() {
        let buf = hex("000007 06 00 00000000 31323334353637");
        assert_eq!(Frame::decode(&buf), Err(FrameError::BadLength));
    }

    #[test]
    fn ping_on_stream_rejected() {
        let buf = hex("000008 06 00 00000001 3132333435363738");
        assert_eq!(Frame::decode(&buf), Err(FrameError::BadStreamId));
    }

    #[test]
    fn goaway_roundtrip() {
        let f = Frame {
            stream_id: 0,
            payload: Payload::GoAway {
                last_stream_id: 11,
                error: ErrorCode::NoError,
                debug_data: b"bye",
            },
        };
        assert_roundtrip(&f);
    }

    #[test]
    fn goaway_short_rejected() {
        let buf = hex("000007 07 00 00000000 0000000b000000");
        assert_eq!(Frame::decode(&buf), Err(FrameError::BadLength));
    }

    #[test]
    fn window_update_roundtrip() {
        for sid in [0u32, 3] {
            let f = Frame {
                stream_id: sid,
                payload: Payload::WindowUpdate { increment: 1024 },
            };
            assert_roundtrip(&f);
        }
    }

    #[test]
    fn window_update_zero_rejected() {
        let buf = hex("000004 08 00 00000001 00000000");
        assert_eq!(Frame::decode(&buf), Err(FrameError::ZeroWindowIncrement));
        // reserved bit is masked off, not an error: 0x80000001 -> 1.
        let buf = hex("000004 08 00 00000001 80000001");
        let (f, _) = Frame::decode(&buf).unwrap();
        assert_eq!(f.payload, Payload::WindowUpdate { increment: 1 });
    }

    #[test]
    fn continuation_roundtrip() {
        let f = Frame {
            stream_id: 1,
            payload: Payload::Continuation {
                end_headers: true,
                fragment: b"\xbe\x58",
            },
        };
        assert_roundtrip(&f);
    }

    #[test]
    fn continuation_on_stream_zero_rejected() {
        let buf = hex("000002 09 04 00000000 be58");
        assert_eq!(Frame::decode(&buf), Err(FrameError::BadStreamId));
    }

    #[test]
    fn unknown_frame_type_preserved() {
        let mut buf = Vec::new();
        Frame::encode_raw(0xcc, 0x03, 7, b"ext", &mut buf);
        let (f, consumed) = Frame::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(f.stream_id, 7);
        assert_eq!(f.ty(), None);
        assert_eq!(f.raw_ty(), 0xcc);
        assert_eq!(
            f.payload,
            Payload::Unknown {
                ty: 0xcc,
                flags: 0x03,
                data: b"ext"
            }
        );
        // Re-encodes verbatim.
        let mut back = Vec::new();
        f.encode(&mut back);
        assert_eq!(back, buf);
    }

    #[test]
    fn truncated_payload_reports_truncated() {
        // header claims 10 payload bytes, only 3 present.
        let buf = hex("00000a 00 01 00000001 616263");
        assert_eq!(Frame::decode(&buf), Err(FrameError::Truncated));
    }

    #[test]
    fn frame_type_from_u8() {
        assert_eq!(FrameType::from_u8(0x9), Some(FrameType::Continuation));
        assert_eq!(FrameType::from_u8(0xa), None);
    }
}
