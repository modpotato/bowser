//! HTTP/2 client (RFC 9113): frame codec, stream multiplexing, and flow
//! control, wired into the fetch pipeline.
//!
//! Negotiation is by ALPN `"h2"` on the TLS handshake (see
//! [`is_h2_conn`]); the TLS layer already offers and records ALPN, so no
//! new handshake machinery was needed. There is deliberately no
//! cleartext `"h2c"` path: no prior-knowledge or `Upgrade` mechanism
//! exists in the engine to extend, and plaintext HTTP/2 would weaken
//! the transport guarantees the fetch layer relies on. A server that
//! does not negotiate `"h2"` falls back to the HTTP/1.1 path, which is
//! untouched.
//!
//! H2 connections are pooled per origin next to the HTTP/1.1
//! keep-alive pool: a pooled [`Session`] carries its HPACK dynamic
//! tables and flow-control windows, so a resumed connection skips the
//! TCP+TLS handshake and the H2 SETTINGS handshake and simply opens a
//! new stream on the existing connection. Closed stream state is reaped
//! before a session goes idle so a long-lived session cannot
//! accumulate it.
//!
//! Scope is client-side only: server push is answered with
//! `RST_STREAM(CANCEL)`, and there is no server implementation.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::time::{Duration, Instant};

use crate::cookies::{CookieJar, RequestContext};
use crate::fetch::{
    FetchError, Response, check_fetch_deadline, content_decoder, finish_chain, header_string,
    map_decode_error, map_io_error,
};
use crate::hpack::{self, HpackError};
use crate::http1::Limits;
use crate::pool::Conn;
use bowser_url::Url;

// ---------------------------------------------------------------------------
// Protocol constants
// ---------------------------------------------------------------------------

/// Client connection preface (RFC 9113 §3.4).
const CLIENT_PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

// Frame types (RFC 9113 §6).
const F_DATA: u8 = 0x0;
const F_HEADERS: u8 = 0x1;
const F_PRIORITY: u8 = 0x2;
const F_RST_STREAM: u8 = 0x3;
const F_SETTINGS: u8 = 0x4;
const F_PUSH_PROMISE: u8 = 0x5;
const F_PING: u8 = 0x6;
const F_GOAWAY: u8 = 0x7;
const F_WINDOW_UPDATE: u8 = 0x8;
const F_CONTINUATION: u8 = 0x9;

// Flags.
const FLAG_END_STREAM: u8 = 0x1;
const FLAG_END_HEADERS: u8 = 0x4;
const FLAG_PADDED: u8 = 0x8;
const FLAG_PRIORITY: u8 = 0x20;
const FLAG_ACK: u8 = 0x1;

// Settings parameters (RFC 9113 §6.5.2).
const S_HEADER_TABLE_SIZE: u16 = 0x1;
const S_ENABLE_PUSH: u16 = 0x2;
const S_MAX_CONCURRENT_STREAMS: u16 = 0x3;
const S_INITIAL_WINDOW_SIZE: u16 = 0x4;
const S_MAX_FRAME_SIZE: u16 = 0x5;
#[allow(dead_code)]
const S_MAX_HEADER_LIST_SIZE: u16 = 0x6;

// Error codes (RFC 9113 §7).
const E_PROTOCOL_ERROR: u32 = 0x1;
const E_FLOW_CONTROL_ERROR: u32 = 0x3;
pub(crate) const E_STREAM_CLOSED: u32 = 0x5;
const E_FRAME_SIZE_ERROR: u32 = 0x6;
const E_COMPRESSION_ERROR: u32 = 0x9;
pub(crate) const E_CANCEL: u32 = 0x8;

/// Default flow-control window (RFC 9113 §6.9.2).
const DEFAULT_WINDOW: i64 = 65_535;
/// Windows are 31-bit.
const MAX_WINDOW: i64 = (1 << 31) - 1;
/// Largest frame we accept (our SETTINGS_MAX_FRAME_SIZE; we advertise
/// the default and never raise it).
const MAX_FRAME_SIZE: usize = 16_384;
/// Cap on one encoded header block (HEADERS + CONTINUATIONs).
const MAX_HEADER_BLOCK: usize = 64 * 1024;
/// Replenish a receive window once it drops to half or below.
const WINDOW_REPLENISH_AT: i64 = 32_768;
/// Bound on interim (1xx) responses, mirroring the HTTP/1.1 path.
const MAX_INTERIM_RESPONSES: u32 = 8;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Every way an HTTP/2 session can fail.
#[derive(Debug)]
pub(crate) enum H2Error {
    /// Transport I/O failed.
    Io(io::Error),
    /// Connection-level protocol violation; carries the RFC 9113 error
    /// code sent (best effort) in GOAWAY.
    Connection(u32, &'static str),
    /// HPACK failure; always a connection error of type
    /// COMPRESSION_ERROR on the wire.
    Hpack(HpackError),
    /// Response `:status` missing, duplicated, or malformed.
    BadStatus(&'static str),
    /// A response header broke an RFC 9113 §8.2.2 rule.
    BadHeader(&'static str),
    /// The response body exceeded the byte cap.
    BodyTooLarge,
    /// The server sent GOAWAY before the response completed; carries
    /// the GOAWAY error code.
    GoAway(u32),
    /// A WINDOW_UPDATE would push a window past 2^31 - 1.
    WindowOverflow,
    /// Client-initiated stream ids are exhausted (2^31 - 1 open).
    StreamsExhausted,
    /// Opening another stream would exceed the peer's
    /// SETTINGS_MAX_CONCURRENT_STREAMS.
    ConcurrentStreamsExceeded,
}

impl std::fmt::Display for H2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "transport: {e}"),
            Self::Connection(code, msg) => write!(f, "connection error {code}: {msg}"),
            Self::Hpack(e) => write!(f, "HPACK: {e}"),
            Self::BadStatus(msg) => write!(f, "bad :status: {msg}"),
            Self::BadHeader(msg) => write!(f, "bad response header: {msg}"),
            Self::BodyTooLarge => f.write_str("response body too large"),
            Self::GoAway(code) => write!(f, "server GOAWAY ({code}) mid-response"),
            Self::WindowOverflow => f.write_str("flow-control window overflow"),
            Self::StreamsExhausted => f.write_str("stream ids exhausted"),
            Self::ConcurrentStreamsExceeded => {
                f.write_str("peer's SETTINGS_MAX_CONCURRENT_STREAMS reached")
            }
        }
    }
}

impl From<io::Error> for H2Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<HpackError> for H2Error {
    fn from(e: HpackError) -> Self {
        Self::Hpack(e)
    }
}

/// Map a session error onto the fetch error taxonomy: transport and
/// timeout failures keep their [`FetchError`] kinds, an oversized body
/// is [`FetchError::BodyTooLarge`], and everything else becomes the
/// dedicated HTTP/2 variant.
impl From<H2Error> for FetchError {
    fn from(e: H2Error) -> Self {
        match e {
            H2Error::Io(io) => map_io_error(io),
            H2Error::BodyTooLarge => FetchError::BodyTooLarge,
            other => FetchError::Http2(other.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// `Conn` as a byte stream
// ---------------------------------------------------------------------------

// `pool::Conn` exposes inherent `read`/`write_all`; the session needs
// the `io` traits, so they are implemented here (same crate, no orphan
// issue, no changes to `pool.rs`).
impl io::Read for Conn {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Conn::Plain(s) => s.read(buf),
            Conn::Tls(s) => s.read(buf),
        }
    }
}

impl io::Write for Conn {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Conn::Plain(s) => s.write(buf),
            // `TlsConnection::write` always consumes the whole slice.
            Conn::Tls(s) => s.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Whether this connection negotiated HTTP/2 via ALPN. Plain TCP is
/// never H2 (no h2c); a TLS connection without an ALPN result falls
/// back to HTTP/1.1.
pub(crate) fn is_h2_conn(conn: &Conn) -> bool {
    alpn_selects_h2(conn.alpn_protocol())
}

/// Select the exchange from the negotiated ALPN protocol: only an
/// explicit `"h2"` takes the HTTP/2 path. `"http/1.1"` and no ALPN at
/// all both mean the HTTP/1.1 path runs unchanged.
pub(crate) fn alpn_selects_h2(alpn: Option<&str>) -> bool {
    alpn == Some("h2")
}

// ---------------------------------------------------------------------------
// Frame codec (RFC 9113 §4, §6)
// ---------------------------------------------------------------------------

/// A decoded frame: 9-byte header plus payload.
#[derive(Debug, PartialEq, Eq)]
struct Frame {
    ty: u8,
    flags: u8,
    /// 31-bit stream identifier (the reserved bit is stripped).
    stream: u32,
    payload: Vec<u8>,
}

/// Read one frame. Payloads larger than our max frame size are a
/// connection error before any allocation beyond the cap.
fn read_frame<T: io::Read>(io: &mut T) -> Result<Frame, H2Error> {
    let mut hdr = [0u8; 9];
    io.read_exact(&mut hdr)?;
    let len = u32::from_be_bytes([0, hdr[0], hdr[1], hdr[2]]) as usize;
    if len > MAX_FRAME_SIZE {
        return Err(H2Error::Connection(
            E_FRAME_SIZE_ERROR,
            "frame exceeds max frame size",
        ));
    }
    let ty = hdr[3];
    let flags = hdr[4];
    let stream = u32::from_be_bytes([hdr[5], hdr[6], hdr[7], hdr[8]]) & 0x7fff_ffff;
    let mut payload = vec![0u8; len];
    io.read_exact(&mut payload)?;
    Ok(Frame {
        ty,
        flags,
        stream,
        payload,
    })
}

/// Serialize one frame.
fn write_frame<T: io::Write>(
    io: &mut T,
    ty: u8,
    flags: u8,
    stream: u32,
    payload: &[u8],
) -> Result<(), H2Error> {
    debug_assert!(stream <= 0x7fff_ffff);
    debug_assert!(payload.len() < (1 << 24));
    let len = payload.len() as u32;
    let mut hdr = [0u8; 9];
    hdr[0] = (len >> 16) as u8;
    hdr[1] = (len >> 8) as u8;
    hdr[2] = len as u8;
    hdr[3] = ty;
    hdr[4] = flags;
    hdr[5..9].copy_from_slice(&(stream & 0x7fff_ffff).to_be_bytes());
    io.write_all(&hdr)?;
    io.write_all(payload)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// Client-side stream state (RFC 9113 §5.1). Only the states a client
/// that always ends its request can reach are exercised, but the full
/// small machine is tracked for the multiplexed case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamState {
    Open,
    HalfClosedLocal,
    HalfClosedRemote,
    Closed,
}

struct Stream {
    state: StreamState,
    send_window: i64,
    recv_window: i64,
    /// Header-block fragments received so far (HEADERS without
    /// END_HEADERS, plus CONTINUATIONs).
    frag: Vec<u8>,
    /// Whether a header block is being received on this stream.
    frag_expected: bool,
    /// The fragments belong to a PUSH_PROMISE: discard, never decode.
    frag_discard: bool,
    /// END_STREAM arrived with the HEADERS that opened `frag`.
    frag_end_stream: bool,
}

/// Stream-level events delivered to the driver. Connection-level frames
/// (PING, SETTINGS, WINDOW_UPDATE, GOAWAY, PUSH_PROMISE) are absorbed
/// inside the session.
#[derive(Debug)]
pub(crate) enum Event {
    /// A complete header block (response headers or trailers), decoded
    /// into the bump slab: names and values live in one allocation,
    /// indexed by [`hpack::Field`] offsets.
    Headers {
        stream: u32,
        fields: hpack::DecodedHeaders,
        end_stream: bool,
    },
    /// A DATA payload chunk.
    Data {
        stream: u32,
        chunk: Vec<u8>,
        end_stream: bool,
    },
    /// The peer reset the stream; carries the error code.
    Reset { stream: u32, code: u32 },
}

impl Event {
    /// The stream this event belongs to. The pool's cooperative pump
    /// uses it to forward events to the thread owning each stream.
    pub(crate) fn stream_id(&self) -> u32 {
        match *self {
            Event::Headers { stream, .. }
            | Event::Data { stream, .. }
            | Event::Reset { stream, .. } => stream,
        }
    }
}

// Manual `PartialEq`/`Eq`: [`hpack::DecodedHeaders`] does not derive
// them, so the slab is compared by contents (bytes plus field ranges),
// which is exactly what the old `Vec<(Vec<u8>, Vec<u8>)>` comparison
// established.
impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Event::Headers {
                    stream: s1,
                    fields: f1,
                    end_stream: e1,
                },
                Event::Headers {
                    stream: s2,
                    fields: f2,
                    end_stream: e2,
                },
            ) => s1 == s2 && e1 == e2 && f1.bytes == f2.bytes && f1.fields == f2.fields,
            (
                Event::Data {
                    stream: s1,
                    chunk: c1,
                    end_stream: e1,
                },
                Event::Data {
                    stream: s2,
                    chunk: c2,
                    end_stream: e2,
                },
            ) => s1 == s2 && c1 == c2 && e1 == e2,
            (
                Event::Reset {
                    stream: s1,
                    code: c1,
                },
                Event::Reset {
                    stream: s2,
                    code: c2,
                },
            ) => s1 == s2 && c1 == c2,
            _ => false,
        }
    }
}

impl Eq for Event {}

/// A client HTTP/2 session over any byte stream: preface, SETTINGS,
/// multiplexed streams, HPACK, and flow control.
pub(crate) struct Session<T: io::Read + io::Write> {
    io: T,
    enc: hpack::Encoder,
    dec: hpack::Decoder,
    /// Next client-initiated stream id (odd: 1, 3, 5, ...).
    next_stream: u32,
    conn_send_window: i64,
    conn_recv_window: i64,
    /// Peer's SETTINGS_MAX_FRAME_SIZE: our send chunk size.
    peer_max_frame: usize,
    /// Peer's SETTINGS_INITIAL_WINDOW_SIZE: new streams' send window.
    peer_init_window: u32,
    /// Peer's SETTINGS_MAX_CONCURRENT_STREAMS: cap on streams that are
    /// not yet fully closed. `u32::MAX` until the peer says otherwise
    /// (RFC 9113 §6.5.2: the default is unlimited).
    peer_max_concurrent: u32,
    streams: HashMap<u32, Stream>,
    pending: VecDeque<Event>,
    /// `(last_stream_id, error_code)` once GOAWAY arrives.
    goaway: Option<(u32, u32)>,
    max_fields: usize,
    /// Scratch bump slab for header-block decoding. `queue_header_block`
    /// takes it into the [`Event::Headers`] it queues, leaving a fresh
    /// one behind; the next decode reuses whatever allocation a
    /// recycled slab hands back, so steady state is one slab plus one
    /// index vec per response head with no per-field allocation.
    header_scratch: hpack::DecodedHeaders,
    /// Opaque PING payload counter; each outbound PING carries a fresh
    /// value so its ACK can be matched unambiguously.
    ping_counter: u64,
}

impl<T: io::Read + io::Write> Session<T> {
    pub(crate) fn new(io: T, max_fields: usize) -> Self {
        Session {
            io,
            enc: hpack::Encoder::new(),
            dec: hpack::Decoder::new(4096),
            next_stream: 1,
            conn_send_window: DEFAULT_WINDOW,
            conn_recv_window: DEFAULT_WINDOW,
            peer_max_frame: MAX_FRAME_SIZE,
            peer_init_window: DEFAULT_WINDOW as u32,
            peer_max_concurrent: u32::MAX,
            streams: HashMap::new(),
            pending: VecDeque::new(),
            goaway: None,
            max_fields,
            header_scratch: hpack::DecodedHeaders::new(),
            ping_counter: 0,
        }
    }

    /// `(last_stream_id, error_code)` after GOAWAY, `None` before.
    pub(crate) fn goaway(&self) -> Option<(u32, u32)> {
        self.goaway
    }

    /// The peer's `SETTINGS_MAX_CONCURRENT_STREAMS`: the cap on streams
    /// that are not yet fully closed. `u32::MAX` until the peer's
    /// SETTINGS arrive. The pool's shared-session layer reads this to
    /// size its stream-slot reservations.
    pub(crate) fn peer_max_concurrent(&self) -> u32 {
        self.peer_max_concurrent
    }

    /// Recover the transport, e.g. so the pool can harvest TLS session
    /// tickets from a discarded session.
    pub(crate) fn into_inner(self) -> T {
        self.io
    }

    /// The transport, for liveness probing and ticket harvesting by the
    /// pool. Frame-level state is untouched.
    pub(crate) fn conn_mut(&mut self) -> &mut T {
        &mut self.io
    }

    /// Drop terminal stream state no future frame can need: fully closed
    /// streams with no header-block fragments in flight. Discard-tracked
    /// push promises (closed but `frag_expected`) are kept so their
    /// CONTINUATIONs are still absorbed; they become reapable once the
    /// block completes. Called before a session goes idle in the pool so
    /// a long-lived session cannot accumulate one entry per fetch.
    pub(crate) fn reap_closed_streams(&mut self) {
        self.streams
            .retain(|_, s| s.state != StreamState::Closed || s.frag_expected);
    }

    /// Connection send window (for tests).
    #[cfg(test)]
    fn send_window(&self) -> i64 {
        self.conn_send_window
    }

    /// Whether the stream is fully closed — or its state was already
    /// reaped. A missing entry means the same thing as Closed here:
    /// `reap_closed_streams` only removes terminal state. (For tests.)
    #[cfg(test)]
    pub(crate) fn is_stream_closed(&self, stream_id: u32) -> bool {
        self.streams
            .get(&stream_id)
            .is_none_or(|s| s.state == StreamState::Closed)
    }

    /// Connection receive window (for tests).
    #[cfg(test)]
    fn recv_window(&self) -> i64 {
        self.conn_recv_window
    }

    // -- handshake ------------------------------------------------------

    /// Write the client preface and exchange SETTINGS with the server.
    /// The server's first frame MUST be its (non-ACK) SETTINGS; it is
    /// applied and acknowledged.
    pub(crate) fn handshake(&mut self, deadline: Instant) -> Result<(), H2Error> {
        self.io.write_all(CLIENT_PREFACE)?;
        // This client never consumes server push: say so up front.
        let mut settings = Vec::with_capacity(6);
        settings.extend_from_slice(&S_ENABLE_PUSH.to_be_bytes());
        settings.extend_from_slice(&0u32.to_be_bytes());
        write_frame(&mut self.io, F_SETTINGS, 0, 0, &settings)?;
        loop {
            check_fetch_deadline(deadline)?;
            let f = read_frame(&mut self.io)?;
            if f.ty == F_SETTINGS && f.flags & FLAG_ACK == 0 {
                if f.stream != 0 {
                    return Err(H2Error::Connection(
                        E_PROTOCOL_ERROR,
                        "SETTINGS on non-zero stream",
                    ));
                }
                self.on_settings(&f.payload)?;
                write_frame(&mut self.io, F_SETTINGS, FLAG_ACK, 0, &[])?;
                return Ok(());
            }
            // Be liberal before the preface completes: answer PINGs, note
            // GOAWAY, ignore the rest; anything malformed still errors.
            self.on_frame(f)?;
        }
    }

    /// Apply a SETTINGS payload (without ACK).
    fn on_settings(&mut self, payload: &[u8]) -> Result<(), H2Error> {
        if !payload.len().is_multiple_of(6) {
            return Err(H2Error::Connection(
                E_PROTOCOL_ERROR,
                "SETTINGS length not a multiple of 6",
            ));
        }
        for chunk in payload.as_chunks::<6>().0 {
            let id = u16::from_be_bytes([chunk[0], chunk[1]]);
            let val = u32::from_be_bytes([chunk[2], chunk[3], chunk[4], chunk[5]]);
            match id {
                S_HEADER_TABLE_SIZE => {
                    // Caps our encoder's dynamic table; this encoder never
                    // inserts, so there is nothing to shrink.
                }
                S_ENABLE_PUSH => {
                    if val > 1 {
                        return Err(H2Error::Connection(
                            E_PROTOCOL_ERROR,
                            "bad ENABLE_PUSH value",
                        ));
                    }
                }
                S_MAX_CONCURRENT_STREAMS => {
                    // A hard cap, not advisory: the peer will treat a
                    // stream past this limit as a connection error
                    // (RFC 9113 §5.1.2), so `open_stream` enforces it
                    // against streams that are not fully closed.
                    self.peer_max_concurrent = val;
                }
                S_INITIAL_WINDOW_SIZE => {
                    if val as i64 > MAX_WINDOW {
                        return Err(H2Error::Connection(
                            E_FLOW_CONTROL_ERROR,
                            "INITIAL_WINDOW_SIZE too large",
                        ));
                    }
                    let delta = val as i64 - self.peer_init_window as i64;
                    self.peer_init_window = val;
                    // RFC 9113 §6.9.2: a SETTINGS change adjusts all open
                    // streams' send windows by the delta.
                    for st in self.streams.values_mut() {
                        st.send_window += delta;
                        if st.send_window > MAX_WINDOW {
                            return Err(H2Error::Connection(
                                E_FLOW_CONTROL_ERROR,
                                "stream window overflow",
                            ));
                        }
                    }
                }
                S_MAX_FRAME_SIZE => {
                    if !(16_384..=16_777_215).contains(&val) {
                        return Err(H2Error::Connection(
                            E_PROTOCOL_ERROR,
                            "bad MAX_FRAME_SIZE value",
                        ));
                    }
                    self.peer_max_frame = val as usize;
                }
                _ => {
                    // Unknown settings are ignored (RFC 9113 §6.5.2).
                }
            }
        }
        Ok(())
    }

    // -- sending --------------------------------------------------------

    /// Open a stream with a header block. Returns the stream id.
    /// `end_stream` sets END_STREAM on the final HEADERS/CONTINUATION.
    pub(crate) fn open_stream(
        &mut self,
        headers: &[(&[u8], &[u8])],
        end_stream: bool,
        deadline: Instant,
    ) -> Result<u32, H2Error> {
        if let Some((_, code)) = self.goaway {
            return Err(H2Error::GoAway(code));
        }
        let id = self.next_stream;
        if id > 0x7fff_ffff {
            return Err(H2Error::StreamsExhausted);
        }
        // RFC 9113 §5.1.2: streams in any state but closed count
        // against the peer's SETTINGS_MAX_CONCURRENT_STREAMS. Checked
        // before the id is consumed so a refused open wastes nothing.
        let open = self
            .streams
            .values()
            .filter(|s| s.state != StreamState::Closed)
            .count() as u64;
        if open >= self.peer_max_concurrent as u64 {
            return Err(H2Error::ConcurrentStreamsExceeded);
        }
        self.next_stream += 2;
        check_fetch_deadline(deadline)?;
        let mut block = Vec::new();
        self.enc.encode(headers, &mut block);
        // Fragment across HEADERS + CONTINUATIONs at the peer's max
        // frame size; the (never empty: pseudo-headers) block always
        // emits at least one frame.
        let mut chunks = block.chunks(self.peer_max_frame).peekable();
        let mut first = true;
        while chunks.peek().is_some() {
            let chunk = chunks.next().unwrap_or(&[]);
            let last = chunks.peek().is_none();
            let ty = if first { F_HEADERS } else { F_CONTINUATION };
            first = false;
            let mut flags = 0u8;
            if last {
                flags |= FLAG_END_HEADERS;
                if end_stream {
                    flags |= FLAG_END_STREAM;
                }
            }
            write_frame(&mut self.io, ty, flags, id, chunk)?;
        }
        // An empty header block cannot happen (pseudo-headers always
        // encode to something), but stay total rather than emitting no
        // frame at all.
        if block.is_empty() {
            let mut flags = FLAG_END_HEADERS;
            if end_stream {
                flags |= FLAG_END_STREAM;
            }
            write_frame(&mut self.io, F_HEADERS, flags, id, &[])?;
        }
        self.streams.insert(
            id,
            Stream {
                state: if end_stream {
                    StreamState::HalfClosedLocal
                } else {
                    StreamState::Open
                },
                send_window: self.peer_init_window as i64,
                recv_window: DEFAULT_WINDOW,
                frag: Vec::new(),
                frag_expected: false,
                frag_discard: false,
                frag_end_stream: false,
            },
        );
        Ok(id)
    }

    /// Send a request body with END_STREAM, honoring connection- and
    /// stream-level flow control. Blocks (deadline-bounded) for
    /// WINDOW_UPDATEs when the windows run dry.
    pub(crate) fn send_data(
        &mut self,
        stream_id: u32,
        mut data: &[u8],
        deadline: Instant,
    ) -> Result<(), H2Error> {
        while !data.is_empty() {
            let stream_window = self
                .streams
                .get(&stream_id)
                .map(|s| s.send_window)
                .unwrap_or(0);
            let avail = self
                .conn_send_window
                .min(stream_window)
                .min(self.peer_max_frame as i64);
            if avail <= 0 {
                // Windows exhausted: pump inbound frames until the peer
                // opens them (or the deadline / a reset / GOAWAY ends it).
                check_fetch_deadline(deadline)?;
                self.pump(deadline)?;
                let state = self
                    .streams
                    .get(&stream_id)
                    .map(|s| s.state)
                    .unwrap_or(StreamState::Closed);
                if state == StreamState::Closed {
                    return Err(H2Error::Connection(
                        E_STREAM_CLOSED,
                        "stream closed while sending",
                    ));
                }
                continue;
            }
            let n = (avail as usize).min(data.len());
            let (chunk, rest) = data.split_at(n);
            data = rest;
            let flags = if data.is_empty() { FLAG_END_STREAM } else { 0 };
            write_frame(&mut self.io, F_DATA, flags, stream_id, chunk)?;
            self.conn_send_window -= n as i64;
            if let Some(st) = self.streams.get_mut(&stream_id) {
                st.send_window -= n as i64;
                if flags & FLAG_END_STREAM != 0 {
                    st.state = match st.state {
                        StreamState::Open => StreamState::HalfClosedLocal,
                        StreamState::HalfClosedRemote => StreamState::Closed,
                        s => s,
                    };
                }
            }
        }
        Ok(())
    }

    /// Reset a stream.
    pub(crate) fn send_rst(&mut self, stream_id: u32, code: u32) -> Result<(), H2Error> {
        write_frame(
            &mut self.io,
            F_RST_STREAM,
            0,
            stream_id,
            &code.to_be_bytes(),
        )?;
        if let Some(st) = self.streams.get_mut(&stream_id) {
            st.state = StreamState::Closed;
            st.frag.clear();
            st.frag_expected = false;
        }
        Ok(())
    }

    // -- receiving ------------------------------------------------------

    /// Read inbound frames until a stream-level event is ready.
    /// Connection-level frames are absorbed along the way.
    pub(crate) fn next_event(&mut self, deadline: Instant) -> Result<Event, H2Error> {
        loop {
            if let Some(ev) = self.pending.pop_front() {
                return Ok(ev);
            }
            self.pump(deadline)?;
        }
    }

    /// Read and dispatch exactly one frame.
    fn pump(&mut self, deadline: Instant) -> Result<(), H2Error> {
        check_fetch_deadline(deadline)?;
        let f = read_frame(&mut self.io)?;
        self.on_frame(f)
    }

    fn on_frame(&mut self, f: Frame) -> Result<(), H2Error> {
        // While a header block is incomplete, only CONTINUATION frames
        // for its stream may intervene (RFC 9113 §6.10); anything else
        // is a connection error.
        if f.ty != F_CONTINUATION && self.streams.values().any(|s| s.frag_expected) {
            return Err(H2Error::Connection(
                E_PROTOCOL_ERROR,
                "frame interleaved in header block",
            ));
        }
        match f.ty {
            F_DATA => self.on_data(f),
            F_HEADERS => self.on_headers(f),
            F_PRIORITY => {
                // Deprecated by RFC 9113, but still validated.
                if f.stream == 0 {
                    return Err(H2Error::Connection(
                        E_PROTOCOL_ERROR,
                        "PRIORITY on stream 0",
                    ));
                }
                if f.payload.len() != 5 {
                    return Err(H2Error::Connection(
                        E_FRAME_SIZE_ERROR,
                        "bad PRIORITY length",
                    ));
                }
                Ok(())
            }
            F_RST_STREAM => self.on_rst_stream(f),
            F_SETTINGS => self.on_settings_frame(f),
            F_PUSH_PROMISE => self.on_push_promise(f),
            F_PING => self.on_ping(f),
            F_GOAWAY => self.on_goaway(f),
            F_WINDOW_UPDATE => self.on_window_update(f),
            F_CONTINUATION => self.on_continuation(f),
            _ => Ok(()), // Unknown frame types are ignored (RFC 9113 §5.5).
        }
    }

    fn on_data(&mut self, f: Frame) -> Result<(), H2Error> {
        if f.stream == 0 {
            return Err(H2Error::Connection(E_PROTOCOL_ERROR, "DATA on stream 0"));
        }
        // Take ownership of the frame payload up front: the delivered
        // chunk moves it into the event instead of copying it
        // (`to_vec`) per DATA frame on the hot path.
        let wire_len = f.payload.len() as i64;
        let mut payload = f.payload;
        if f.flags & FLAG_PADDED != 0 {
            if payload.is_empty() {
                return Err(H2Error::Connection(E_PROTOCOL_ERROR, "bad DATA padding"));
            }
            let pad = payload[0] as usize;
            if payload.len() - 1 < pad {
                return Err(H2Error::Connection(E_PROTOCOL_ERROR, "bad DATA padding"));
            }
            // Strip the pad-length byte and the trailing padding in
            // place: `drain` shifts left, `truncate` is free — no
            // allocation, unlike slicing plus `to_vec`.
            payload.drain(..1);
            payload.truncate(payload.len() - pad);
        }
        // Flow-control accounting counts the whole frame payload on the
        // wire, padding included (RFC 9113 §6.9); only the depadded
        // bytes are delivered.
        let state = match self.streams.get(&f.stream) {
            None => return Ok(()), // Unknown/retired stream: racing data, ignore.
            Some(st) => st.state,
        };
        match state {
            StreamState::Closed => return Ok(()), // Already done; ignore.
            StreamState::HalfClosedRemote => {
                // DATA after END_STREAM: stream error (RFC 9113 §5.1).
                self.send_rst(f.stream, E_STREAM_CLOSED)?;
                return Ok(());
            }
            StreamState::Open | StreamState::HalfClosedLocal => {}
        }
        // Flow-control accounting before touching state further.
        {
            let st = self.streams.get(&f.stream).unwrap();
            if wire_len > st.recv_window || wire_len > self.conn_recv_window {
                return Err(H2Error::Connection(
                    E_FLOW_CONTROL_ERROR,
                    "flow-control window exceeded",
                ));
            }
        }
        // Replenish windows that dropped to half or below, keeping the
        // peer's upload unthrottled.
        let stream_inc = {
            let st = self.streams.get_mut(&f.stream).unwrap();
            st.recv_window -= wire_len;
            if st.recv_window <= WINDOW_REPLENISH_AT {
                let inc = (DEFAULT_WINDOW - st.recv_window) as u32;
                st.recv_window += inc as i64;
                Some(inc)
            } else {
                None
            }
        };
        if let Some(inc) = stream_inc {
            write_frame(
                &mut self.io,
                F_WINDOW_UPDATE,
                0,
                f.stream,
                &inc.to_be_bytes(),
            )?;
        }
        self.conn_recv_window -= wire_len;
        if self.conn_recv_window <= WINDOW_REPLENISH_AT {
            let inc = (DEFAULT_WINDOW - self.conn_recv_window) as u32;
            self.conn_recv_window += inc as i64;
            write_frame(&mut self.io, F_WINDOW_UPDATE, 0, 0, &inc.to_be_bytes())?;
        }
        let end_stream = f.flags & FLAG_END_STREAM != 0;
        if end_stream && let Some(st) = self.streams.get_mut(&f.stream) {
            st.state = match st.state {
                StreamState::HalfClosedLocal => StreamState::Closed,
                _ => StreamState::HalfClosedRemote,
            };
        }
        self.pending.push_back(Event::Data {
            stream: f.stream,
            chunk: payload,
            end_stream,
        });
        Ok(())
    }

    fn on_headers(&mut self, f: Frame) -> Result<(), H2Error> {
        if f.stream == 0 {
            return Err(H2Error::Connection(E_PROTOCOL_ERROR, "HEADERS on stream 0"));
        }
        let mut payload = f.payload.as_slice();
        if f.flags & FLAG_PADDED != 0 {
            if payload.is_empty() {
                return Err(H2Error::Connection(E_PROTOCOL_ERROR, "bad HEADERS padding"));
            }
            let pad = payload[0] as usize;
            if payload.len() - 1 < pad {
                return Err(H2Error::Connection(E_PROTOCOL_ERROR, "bad HEADERS padding"));
            }
            payload = &payload[1..payload.len() - pad];
        }
        if f.flags & FLAG_PRIORITY != 0 {
            // RFC 9113 removed HEADERS priority, but a peer may still
            // set the flag: skip the 5-byte priority prefix.
            if payload.len() < 5 {
                return Err(H2Error::Connection(
                    E_PROTOCOL_ERROR,
                    "short HEADERS priority prefix",
                ));
            }
            payload = &payload[5..];
        }
        let known = self.streams.contains_key(&f.stream);
        if !known {
            // Idle stream: only even (server-push) ids are legal here, and
            // we RST every push, so this is racing header bytes for a
            // reset push — unless CONTINUATIONs follow, which
            // `on_continuation` ignores for unknown streams.
            if f.stream % 2 == 1 {
                // Odd ids below `next_stream` were opened by us and have
                // since been reaped as closed (see
                // `reap_closed_streams`): tolerate racing bytes for them
                // exactly as when the entry was still present.
                if f.stream < self.next_stream {
                    return Ok(());
                }
                return Err(H2Error::Connection(
                    E_PROTOCOL_ERROR,
                    "HEADERS on idle stream",
                ));
            }
            return Ok(());
        }
        // From here the stream is known; split borrows via a scope.
        let end_stream = f.flags & FLAG_END_STREAM != 0;
        let end_headers = f.flags & FLAG_END_HEADERS != 0;
        // Fast path: a complete header block in one HEADERS frame with
        // nothing staged decodes straight from the frame payload,
        // skipping the `frag` staging buffer (one alloc + memcpy per
        // response head).
        let staged = {
            let st = self.streams.get_mut(&f.stream).unwrap();
            if st.frag_expected {
                return Err(H2Error::Connection(
                    E_PROTOCOL_ERROR,
                    "HEADERS while CONTINUATION pending",
                ));
            }
            match st.state {
                // Response headers after the stream closed (e.g. trailers
                // past END_STREAM): nothing left to attach them to.
                StreamState::Closed | StreamState::HalfClosedRemote => return Ok(()),
                StreamState::Open | StreamState::HalfClosedLocal => {}
            }
            if st.frag.len() + payload.len() > MAX_HEADER_BLOCK {
                return Err(H2Error::Connection(
                    E_COMPRESSION_ERROR,
                    "header block too large",
                ));
            }
            !(end_headers && st.frag.is_empty())
        };
        if !staged {
            return self.queue_header_block(f.stream, payload, end_stream);
        }
        {
            let st = self.streams.get_mut(&f.stream).unwrap();
            st.frag.extend_from_slice(payload);
            st.frag_end_stream = end_stream;
            if !end_headers {
                st.frag_expected = true;
                return Ok(());
            }
        }
        self.finish_header_block(f.stream)
    }

    /// Decode one header block and queue the HEADERS event, updating
    /// the stream state for END_STREAM. Shared by the single-fragment
    /// fast path (decoded straight from the frame payload) and the
    /// reassembled multi-fragment path.
    ///
    /// The block decodes into the session's scratch bump slab, which
    /// moves into the queued event (a fresh slab stays behind for the
    /// next block): one slab plus one index vec per response head, no
    /// per-field allocation.
    fn queue_header_block(
        &mut self,
        stream: u32,
        block: &[u8],
        end_stream: bool,
    ) -> Result<(), H2Error> {
        let mut fields = std::mem::take(&mut self.header_scratch);
        self.dec
            .decode_block(block, self.max_fields, &mut fields)
            .map_err(H2Error::from)?;
        if end_stream && let Some(st) = self.streams.get_mut(&stream) {
            st.state = match st.state {
                StreamState::HalfClosedLocal => StreamState::Closed,
                _ => StreamState::HalfClosedRemote,
            };
        }
        self.pending.push_back(Event::Headers {
            stream,
            fields,
            end_stream,
        });
        Ok(())
    }

    /// Decode the accumulated header block for `stream` and queue it.
    fn finish_header_block(&mut self, stream: u32) -> Result<(), H2Error> {
        let (frag, end_stream) = {
            let st = self.streams.get_mut(&stream).ok_or(H2Error::Connection(
                E_PROTOCOL_ERROR,
                "header block for unknown stream",
            ))?;
            st.frag_expected = false;
            (std::mem::take(&mut st.frag), st.frag_end_stream)
        };
        self.queue_header_block(stream, &frag, end_stream)
    }

    fn on_continuation(&mut self, f: Frame) -> Result<(), H2Error> {
        match self.streams.get_mut(&f.stream) {
            Some(st) if st.frag_expected => {
                if st.frag_discard {
                    // A pushed header block we are skipping.
                    if f.flags & FLAG_END_HEADERS != 0 {
                        st.frag.clear();
                        st.frag_expected = false;
                        st.frag_discard = false;
                    }
                    return Ok(());
                }
                if st.frag.len() + f.payload.len() > MAX_HEADER_BLOCK {
                    return Err(H2Error::Connection(
                        E_COMPRESSION_ERROR,
                        "header block too large",
                    ));
                }
                st.frag.extend_from_slice(&f.payload);
                if f.flags & FLAG_END_HEADERS != 0 {
                    return self.finish_header_block(f.stream);
                }
                Ok(())
            }
            _ => {
                // No header block in flight for this stream. A
                // CONTINUATION must always continue a HEADERS /
                // PUSH_PROMISE / CONTINUATION without END_HEADERS; a
                // stray one is a connection error (RFC 9113 §6.10).
                Err(H2Error::Connection(E_PROTOCOL_ERROR, "stray CONTINUATION"))
            }
        }
    }

    fn on_rst_stream(&mut self, f: Frame) -> Result<(), H2Error> {
        if f.stream == 0 {
            return Err(H2Error::Connection(
                E_PROTOCOL_ERROR,
                "RST_STREAM on stream 0",
            ));
        }
        if f.payload.len() != 4 {
            return Err(H2Error::Connection(
                E_FRAME_SIZE_ERROR,
                "bad RST_STREAM length",
            ));
        }
        let code = u32::from_be_bytes([f.payload[0], f.payload[1], f.payload[2], f.payload[3]]);
        if let Some(st) = self.streams.get_mut(&f.stream) {
            st.state = StreamState::Closed;
            st.frag.clear();
            st.frag_expected = false;
            // A reset only means something to an exchange on a live
            // stream. A reset for an unknown id — never opened, already
            // reaped, or a future id the peer invented — queues no
            // event: without this, a stray RST_STREAM absorbed while
            // the pool's take-time PING round trip is waiting for its
            // ACK would sit in `pending` and be misattributed to the
            // next exchange's fresh stream (client ids are sequential
            // odd numbers, so the next id is predictable).
            self.pending.push_back(Event::Reset {
                stream: f.stream,
                code,
            });
        }
        Ok(())
    }

    fn on_settings_frame(&mut self, f: Frame) -> Result<(), H2Error> {
        if f.stream != 0 {
            return Err(H2Error::Connection(
                E_PROTOCOL_ERROR,
                "SETTINGS on non-zero stream",
            ));
        }
        if f.flags & FLAG_ACK != 0 {
            if !f.payload.is_empty() {
                return Err(H2Error::Connection(
                    E_FRAME_SIZE_ERROR,
                    "SETTINGS ack with payload",
                ));
            }
            return Ok(());
        }
        self.on_settings(&f.payload)?;
        write_frame(&mut self.io, F_SETTINGS, FLAG_ACK, 0, &[])?;
        Ok(())
    }

    fn on_ping(&mut self, f: Frame) -> Result<(), H2Error> {
        if f.stream != 0 {
            return Err(H2Error::Connection(
                E_PROTOCOL_ERROR,
                "PING on non-zero stream",
            ));
        }
        if f.payload.len() != 8 {
            return Err(H2Error::Connection(E_FRAME_SIZE_ERROR, "bad PING length"));
        }
        if f.flags & FLAG_ACK == 0 {
            let payload = f.payload.clone();
            write_frame(&mut self.io, F_PING, FLAG_ACK, 0, &payload)?;
        }
        Ok(())
    }

    fn on_goaway(&mut self, f: Frame) -> Result<(), H2Error> {
        if f.stream != 0 {
            return Err(H2Error::Connection(
                E_PROTOCOL_ERROR,
                "GOAWAY on non-zero stream",
            ));
        }
        if f.payload.len() < 8 {
            return Err(H2Error::Connection(E_FRAME_SIZE_ERROR, "bad GOAWAY length"));
        }
        let last = u32::from_be_bytes([f.payload[0], f.payload[1], f.payload[2], f.payload[3]])
            & 0x7fff_ffff;
        let code = u32::from_be_bytes([f.payload[4], f.payload[5], f.payload[6], f.payload[7]]);
        self.goaway = Some((last, code));
        Ok(())
    }

    fn on_window_update(&mut self, f: Frame) -> Result<(), H2Error> {
        if f.payload.len() != 4 {
            return Err(H2Error::Connection(
                E_FRAME_SIZE_ERROR,
                "bad WINDOW_UPDATE length",
            ));
        }
        let inc = u32::from_be_bytes([f.payload[0], f.payload[1], f.payload[2], f.payload[3]])
            & 0x7fff_ffff;
        if inc == 0 {
            return Err(H2Error::Connection(
                E_PROTOCOL_ERROR,
                "zero WINDOW_UPDATE increment",
            ));
        }
        if f.stream == 0 {
            self.conn_send_window = self
                .conn_send_window
                .checked_add(inc as i64)
                .filter(|&w| w <= MAX_WINDOW)
                .ok_or(H2Error::WindowOverflow)?;
        } else if let Some(st) = self.streams.get_mut(&f.stream) {
            st.send_window = st
                .send_window
                .checked_add(inc as i64)
                .filter(|&w| w <= MAX_WINDOW)
                .ok_or(H2Error::Connection(
                    E_FLOW_CONTROL_ERROR,
                    "stream window overflow",
                ))?;
        }
        // WINDOW_UPDATE for an unknown/closed stream: ignored.
        Ok(())
    }

    fn on_push_promise(&mut self, f: Frame) -> Result<(), H2Error> {
        if f.stream == 0 {
            return Err(H2Error::Connection(
                E_PROTOCOL_ERROR,
                "PUSH_PROMISE on stream 0",
            ));
        }
        if f.payload.len() < 4 {
            return Err(H2Error::Connection(
                E_FRAME_SIZE_ERROR,
                "short PUSH_PROMISE",
            ));
        }
        let mut payload = f.payload.as_slice();
        if f.flags & FLAG_PADDED != 0 {
            if payload.is_empty() {
                return Err(H2Error::Connection(
                    E_PROTOCOL_ERROR,
                    "bad PUSH_PROMISE padding",
                ));
            }
            let pad = payload[0] as usize;
            if payload.len() - 1 < pad {
                return Err(H2Error::Connection(
                    E_PROTOCOL_ERROR,
                    "bad PUSH_PROMISE padding",
                ));
            }
            payload = &payload[1..payload.len() - pad];
        }
        let promised =
            u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]) & 0x7fff_ffff;
        // This client never consumes server push: cancel it outright.
        // A malformed promise (stream 0, odd id, or a duplicate) is a
        // connection error instead.
        if promised == 0 || promised % 2 == 1 || self.streams.contains_key(&promised) {
            return Err(H2Error::Connection(
                E_PROTOCOL_ERROR,
                "bad promised stream id",
            ));
        }
        let parent_ok = self
            .streams
            .get(&f.stream)
            .is_some_and(|s| matches!(s.state, StreamState::Open | StreamState::HalfClosedLocal));
        if !parent_ok {
            // Racing a closed parent: still track the promised stream
            // as discard so any following CONTINUATIONs are absorbed
            // instead of tripping the sequence rule.
            self.track_discarded_push(promised, f.flags & FLAG_END_HEADERS == 0);
            return Ok(());
        }
        self.send_rst(promised, E_CANCEL)?;
        self.track_discarded_push(promised, f.flags & FLAG_END_HEADERS == 0);
        Ok(())
    }

    /// Track a promised stream we will never consume: closed, with its
    /// header-block fragments (if any) marked for discard. With
    /// `expect_continuation` false the whole block arrived already and
    /// nothing needs tracking.
    fn track_discarded_push(&mut self, promised: u32, expect_continuation: bool) {
        if !expect_continuation {
            return;
        }
        self.streams.insert(
            promised,
            Stream {
                state: StreamState::Closed,
                send_window: self.peer_init_window as i64,
                recv_window: DEFAULT_WINDOW,
                frag: Vec::new(),
                frag_expected: true,
                frag_discard: true,
                frag_end_stream: false,
            },
        );
    }

    // -- keepalive --------------------------------------------------------

    /// Send a PING carrying a fresh opaque payload. Fire-and-forget
    /// keepalive: the send itself resets the server's idle timer, and a
    /// write failure is an early, cheap signal that the peer is gone.
    /// The payload is returned for convenience/debugging only; it is
    /// never matched by the pool (`on_ping` ignores ACKs with zero
    /// bookkeeping), so no caller should wait for a response to it.
    pub(crate) fn send_ping(&mut self) -> Result<[u8; 8], H2Error> {
        let payload = self.ping_counter.to_be_bytes();
        self.ping_counter = self.ping_counter.wrapping_add(1);
        write_frame(&mut self.io, F_PING, 0, 0, &payload)?;
        Ok(payload)
    }
}

// ---------------------------------------------------------------------------
// Fetch wiring
// ---------------------------------------------------------------------------

/// The response side of one H2 stream: everything
/// [`h2_recv_response`] needs from the session. `Session` implements it
/// directly for the exclusive fresh-dial path; the pool's shared
/// sessions implement it over a cooperative pump that forwards other
/// streams' events to their owners, so the response loop below runs
/// unchanged on a multiplexed connection.
pub(crate) trait H2EventSource {
    /// Next event for the caller's stream, like [`Session::next_event`].
    fn next_event(&mut self, deadline: Instant) -> Result<Event, H2Error>;
    /// Like [`Session::goaway`].
    fn goaway(&self) -> Option<(u32, u32)>;
}

impl<T: io::Read + io::Write> H2EventSource for Session<T> {
    fn next_event(&mut self, deadline: Instant) -> Result<Event, H2Error> {
        Session::next_event(self, deadline)
    }

    fn goaway(&self) -> Option<(u32, u32)> {
        Session::goaway(self)
    }
}

/// Run one request/response exchange over an `"h2"`-negotiated
/// connection, returning the session alongside the result.
///
/// The session always comes back: on success the caller may hand it to
/// the pool when no GOAWAY was observed, resuming it — HPACK tables,
/// flow-control windows, and stream counter intact — for the next fetch
/// instead of dialing again. On error, or when the server sent GOAWAY,
/// the caller must discard it; the transport is recovered via
/// [`Session::into_inner`] so TLS session tickets are still harvested.
///
/// Mirrors the HTTP/1.1 exchange's observable behavior: same header
/// selection, same cookie handling, same content decoding, same body
/// cap, same total deadline (measured from exchange start — the H2
/// handshake's round trip is inside the budget, unlike the H1 deadline
/// which starts at request-send).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fetch_h2(
    conn: Conn,
    method: &str,
    url: &Url,
    extra: &[(String, String)],
    body: Option<&[u8]>,
    cookie: Option<&str>,
    strip_auth: bool,
    proxy_auth: Option<&str>,
    origin_form: bool,
    fetch_total: Duration,
    limits: &Limits,
    jar: &mut CookieJar,
    ctx: Option<&RequestContext>,
    now: u64,
) -> (Result<Response, FetchError>, Session<Conn>) {
    let start = Instant::now();
    let deadline = start + fetch_total;
    let mut session = Session::new(conn, limits.max_headers);
    let result = match session.handshake(deadline).map_err(FetchError::from) {
        Ok(()) => fetch_h2_exchange(
            &mut session,
            method,
            url,
            extra,
            body,
            cookie,
            strip_auth,
            proxy_auth,
            origin_form,
            deadline,
            fetch_total,
            limits,
            jar,
            ctx,
            now,
        ),
        Err(e) => (Err(e), false),
    };
    // A fresh connection is never retried, so the response-head flag is
    // irrelevant here; only the resumed-session path in `fetch.rs` uses
    // it.
    let (result, _got_head) = result;
    (result, session)
}

/// Run one request/response exchange on an established H2 session (the
/// SETTINGS handshake is already done). `deadline` bounds the response
/// phase; `fetch_total` feeds the content-length slack rule. Generic
/// over the transport so fresh and pooled sessions share the exact same
/// exchange logic.
///
/// Returns the exchange outcome together with whether the response
/// HEADERS arrived: a failed exchange may only be replayed when no
/// response head was received, because once the head arrived the server
/// may have acted on the request.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fetch_h2_exchange<T: io::Read + io::Write>(
    session: &mut Session<T>,
    method: &str,
    url: &Url,
    extra: &[(String, String)],
    body: Option<&[u8]>,
    cookie: Option<&str>,
    strip_auth: bool,
    proxy_auth: Option<&str>,
    origin_form: bool,
    deadline: Instant,
    fetch_total: Duration,
    limits: &Limits,
    jar: &mut CookieJar,
    ctx: Option<&RequestContext>,
    now: u64,
) -> (Result<Response, FetchError>, bool) {
    // Header selection lives in the fetch layer: the pool's shared-H2
    // fast path builds the same headers to open a stream without
    // running a whole exchange.
    let headers = match crate::fetch::h2_request_parts(
        method,
        url,
        extra,
        body,
        cookie,
        strip_auth,
        proxy_auth,
        origin_form,
    ) {
        Ok(h) => h,
        Err(e) => return (Err(e), false),
    };
    let header_refs: Vec<(&[u8], &[u8])> = headers
        .iter()
        .map(|(n, v)| (n.as_slice(), v.as_slice()))
        .collect();
    let body_bytes = body.unwrap_or(&[]);
    let stream = match session
        .open_stream(&header_refs, body_bytes.is_empty(), deadline)
        .map_err(FetchError::from)
    {
        Ok(s) => s,
        Err(e) => return (Err(e), false),
    };
    if !body_bytes.is_empty()
        && let Err(e) = session
            .send_data(stream, body_bytes, deadline)
            .map_err(FetchError::from)
    {
        return (Err(e), false);
    }
    // The request is fully sent: from here the deadline mirrors the
    // HTTP/1.1 exchange, including the per-size slack below.
    let sent_at = Instant::now();
    h2_recv_response(
        session,
        stream,
        method,
        url,
        deadline,
        sent_at,
        fetch_total,
        limits,
        jar,
        ctx,
        now,
    )
}

/// Assemble one response from a stream's events: status, fields,
/// trailers, and body, with the same validation, cookie handling, and
/// content decoding as the HTTP/1.1 path. The request is already on the
/// wire (`stream` was opened by the caller); `sent_at` marks when it
/// finished sending, mirroring the HTTP/1.1 deadline behavior.
///
/// Runs on any [`H2EventSource`]: the exclusive fresh-dial session or
/// one multiplexed stream of a shared pooled session.
///
/// Returns the exchange outcome together with whether the final
/// response HEADERS arrived: a failed exchange may only be replayed
/// when they did not — once the head arrived the server may have acted
/// on the request. Interim 1xx blocks do not set the flag.
#[allow(clippy::too_many_arguments)]
pub(crate) fn h2_recv_response(
    events: &mut dyn H2EventSource,
    stream: u32,
    method: &str,
    url: &Url,
    deadline: Instant,
    sent_at: Instant,
    fetch_total: Duration,
    limits: &Limits,
    jar: &mut CookieJar,
    ctx: Option<&RequestContext>,
    now: u64,
) -> (Result<Response, FetchError>, bool) {
    let mut got_head = false;
    let result = h2_recv_response_inner(
        events,
        stream,
        method,
        url,
        deadline,
        sent_at,
        fetch_total,
        limits,
        jar,
        ctx,
        now,
        &mut got_head,
    );
    (result, got_head)
}

/// The response loop itself: same arguments as [`h2_recv_response`],
/// plus the out-flag set the moment the final response HEADERS parse.
#[allow(clippy::too_many_arguments)]
fn h2_recv_response_inner(
    events: &mut dyn H2EventSource,
    stream: u32,
    method: &str,
    url: &Url,
    mut deadline: Instant,
    sent_at: Instant,
    fetch_total: Duration,
    limits: &Limits,
    jar: &mut CookieJar,
    ctx: Option<&RequestContext>,
    now: u64,
    got_head: &mut bool,
) -> Result<Response, FetchError> {
    let mut status: Option<u16> = None;
    let mut fields: Vec<(String, Vec<u8>)> = Vec::new();
    let mut raw_body: Vec<u8> = Vec::new();
    let mut interim_seen = 0u32;
    let is_head = method.eq_ignore_ascii_case("head");
    let mut done = false;
    while !done {
        // A GOAWAY that arrived before the response completed ends the
        // exchange; the session absorbed the frame already.
        if let Some((_, code)) = events.goaway() {
            return Err(FetchError::from(H2Error::GoAway(code)));
        }
        match events.next_event(deadline).map_err(FetchError::from)? {
            Event::Headers {
                stream: s,
                fields: hf,
                end_stream,
            } if s == stream => {
                let (st, mut fds) = parse_response_headers(&hf)?;
                if (100..200).contains(&st) && !end_stream {
                    // Interim 1xx: discarded, like the HTTP/1.1 path.
                    interim_seen += 1;
                    if interim_seen > MAX_INTERIM_RESPONSES {
                        return Err(FetchError::Http2("too many interim responses".into()));
                    }
                    continue;
                }
                if status.is_none() {
                    status = Some(st);
                    // The final response head arrived: from here the
                    // server may have acted on the request, so the
                    // exchange must not be replayed. Interim 1xx blocks
                    // `continue` above without reaching this point.
                    *got_head = true;
                    // Same slack rule as the HTTP/1.1 body deadline: one
                    // extra second per 64 KiB of declared body.
                    if let Some(n) = declared_content_length(&fds)? {
                        deadline =
                            sent_at + fetch_total + Duration::from_secs(n.div_ceil(64 * 1024));
                    }
                    // Store Set-Cookie before reading the body, exactly
                    // like the HTTP/1.1 path.
                    if let Some(ctx) = ctx {
                        for (name, value) in &fds {
                            if name.eq_ignore_ascii_case("set-cookie") {
                                jar.set_cookie(&String::from_utf8_lossy(value), ctx, now);
                            }
                        }
                    }
                    fields = fds;
                } else {
                    // HEADERS after the final head are trailers; the
                    // HTTP/1.1 path appends them after the head fields.
                    fields.append(&mut fds);
                }
                if end_stream {
                    done = true;
                }
            }
            Event::Data {
                stream: s,
                chunk,
                end_stream,
            } if s == stream => {
                let st = status
                    .ok_or_else(|| FetchError::Http2("DATA before response headers".into()))?;
                // 1xx/204/304 and HEAD responses carry no body; any DATA
                // is dropped (flow control was already accounted).
                if !is_head && !matches!(st, 100..=199 | 204 | 304) {
                    if raw_body.len() + chunk.len() > limits.max_body_bytes {
                        return Err(H2Error::BodyTooLarge.into());
                    }
                    raw_body.extend_from_slice(&chunk);
                }
                if end_stream {
                    done = true;
                }
            }
            Event::Reset { stream: s, code } if s == stream => {
                return Err(FetchError::Http2(format!(
                    "server reset stream {s}: error code {code}"
                )));
            }
            _ => {
                // Events for other streams never reach this loop: the
                // shared-session pump forwards them to their owners.
                // Defensive all the same.
            }
        }
    }
    let status = status.ok_or_else(|| FetchError::Http2("response had no headers".into()))?;

    // `Content-Length` must match the DATA total (RFC 9113 §8.6),
    // counting transfer bytes like the HTTP/1.1 path does.
    if !is_head
        && !matches!(status, 100..=199 | 204 | 304)
        && let Some(declared) = declared_content_length(&fields)?
        && declared != raw_body.len() as u64
    {
        return Err(FetchError::Http2(format!(
            "content-length {declared} but received {} body bytes",
            raw_body.len()
        )));
    }

    // Same content decoding as HTTP/1.1 (gzip/deflate), same cap.
    let mut chain = content_decoder(&fields, limits)?;
    let mut body_out = Vec::new();
    if let Some(chain) = chain.as_mut() {
        chain
            .feed(&raw_body, &mut body_out)
            .map_err(map_decode_error)?;
        finish_chain(status, chain, &mut body_out)?;
        raw_body = body_out;
    }

    Ok(Response {
        status,
        headers: fields
            .into_iter()
            .map(|(n, v)| (n, header_string(v)))
            .collect(),
        body: raw_body,
        url: url.clone(),
    })
}

/// Declared `Content-Length` of a response: every value must parse and
/// all values must agree (mirroring the HTTP/1.1 conflicting-length
/// rejection).
fn declared_content_length(fields: &[(String, Vec<u8>)]) -> Result<Option<u64>, FetchError> {
    let bad = || FetchError::Http2("bad content-length in response".into());
    let mut declared: Option<u64> = None;
    for (name, value) in fields {
        if !name.eq_ignore_ascii_case("content-length") {
            continue;
        }
        let n: u64 = String::from_utf8_lossy(value)
            .trim()
            .parse()
            .map_err(|_| bad())?;
        if declared.is_some_and(|d| d != n) {
            return Err(bad());
        }
        declared = Some(n);
    }
    Ok(declared)
}

/// A validated response head: status plus regular fields.
type ResponseHead = (u16, Vec<(String, Vec<u8>)>);

/// Validate response pseudo-headers (RFC 9113 §8.2.2) and split the
/// header block into status + regular fields.
///
/// Reads names and values as slices of the decode slab; only the
/// validated [`ResponseHead`] at the API edge owns its strings.
fn parse_response_headers(hf: &hpack::DecodedHeaders) -> Result<ResponseHead, FetchError> {
    let bad = |e: H2Error| FetchError::from(e);
    let mut status: Option<u16> = None;
    let mut seen_regular = false;
    let mut fields = Vec::with_capacity(hf.len());
    for i in 0..hf.len() {
        let (name, value) = hf.field(i);
        if name.first() == Some(&b':') {
            if seen_regular {
                return Err(bad(H2Error::BadHeader(
                    "pseudo-header after regular header",
                )));
            }
            if name != b":status" {
                return Err(bad(H2Error::BadHeader(
                    "unexpected pseudo-header in response",
                )));
            }
            if status.is_some() {
                return Err(bad(H2Error::BadStatus("duplicate :status")));
            }
            if value.len() != 3 || !value.iter().all(|b| b.is_ascii_digit()) {
                return Err(bad(H2Error::BadStatus("malformed :status")));
            }
            status = Some(
                std::str::from_utf8(value)
                    .map_err(|_| bad(H2Error::BadStatus("malformed :status")))?
                    .parse()
                    .map_err(|_| bad(H2Error::BadStatus("malformed :status")))?,
            );
        } else {
            seen_regular = true;
            let name_str = std::str::from_utf8(name)
                .map_err(|_| bad(H2Error::BadHeader("non-UTF-8 header name")))?;
            // Connection-specific fields are never valid in HTTP/2.
            // `te: trailers` is the one legal exception.
            if name_str.eq_ignore_ascii_case("connection")
                || name_str.eq_ignore_ascii_case("keep-alive")
                || name_str.eq_ignore_ascii_case("proxy-connection")
                || name_str.eq_ignore_ascii_case("upgrade")
                || name_str.eq_ignore_ascii_case("transfer-encoding")
            {
                return Err(bad(H2Error::BadHeader(
                    "connection-specific header in response",
                )));
            }
            if name_str.eq_ignore_ascii_case("te") && !value.eq_ignore_ascii_case(b"trailers") {
                return Err(bad(H2Error::BadHeader("non-trailers te in response")));
            }
            fields.push((name_str.to_owned(), value.to_vec()));
        }
    }
    let status = status.ok_or(H2Error::BadStatus("missing :status"))?;
    Ok((status, fields))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// In-memory transport: pre-loaded server bytes flow in, everything
    /// the client writes lands in `outbound`.
    struct FakeIo {
        inbound: VecDeque<u8>,
        outbound: Vec<u8>,
    }

    impl FakeIo {
        fn new(inbound: Vec<u8>) -> Self {
            Self {
                inbound: inbound.into(),
                outbound: Vec::new(),
            }
        }
    }

    impl io::Read for FakeIo {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.inbound.is_empty() {
                // Script exhausted: behave like a non-blocking socket
                // with nothing to deliver.
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "no more scripted bytes",
                ));
            }
            let n = buf.len().min(self.inbound.len());
            for slot in buf.iter_mut().take(n) {
                *slot = self.inbound.pop_front().unwrap();
            }
            Ok(n)
        }
    }

    impl io::Write for FakeIo {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.outbound.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(5)
    }

    fn frame_bytes(ty: u8, flags: u8, stream: u32, payload: &[u8]) -> Vec<u8> {
        let mut v = Vec::with_capacity(9 + payload.len());
        let len = payload.len() as u32;
        v.extend_from_slice(&[(len >> 16) as u8, (len >> 8) as u8, len as u8, ty, flags]);
        v.extend_from_slice(&(stream & 0x7fff_ffff).to_be_bytes());
        v.extend_from_slice(payload);
        v
    }

    fn settings_frame(payload: &[u8]) -> Vec<u8> {
        frame_bytes(F_SETTINGS, 0, 0, payload)
    }

    fn setting(id: u16, val: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity(6);
        v.extend_from_slice(&id.to_be_bytes());
        v.extend_from_slice(&val.to_be_bytes());
        v
    }

    /// Decode every frame in `bytes` (used to inspect client output).
    fn parse_frames(mut bytes: &[u8]) -> Vec<Frame> {
        let mut out = Vec::new();
        while !bytes.is_empty() {
            out.push(read_frame(&mut bytes).unwrap());
        }
        out
    }

    /// HPACK-encode several header blocks with one encoder, preserving
    /// dynamic-table state like a real peer.
    fn encode_blocks(blocks: &[&[(&[u8], &[u8])]]) -> Vec<Vec<u8>> {
        let mut enc = hpack::Encoder::new();
        blocks
            .iter()
            .map(|fields| {
                let mut out = Vec::new();
                enc.encode(fields, &mut out);
                out
            })
            .collect()
    }

    fn get(path: &[u8]) -> Vec<(&[u8], &[u8])> {
        vec![
            (b":method".as_slice(), b"GET".as_slice()),
            (b":scheme".as_slice(), b"https".as_slice()),
            (b":path".as_slice(), path),
            (b":authority".as_slice(), b"x.test".as_slice()),
        ]
    }

    /// Whether a decoded header block contains `name: value`.
    fn has_field(fields: &hpack::DecodedHeaders, name: &[u8], value: &[u8]) -> bool {
        (0..fields.len()).any(|i| {
            let (n, v) = fields.field(i);
            n == name && v == value
        })
    }

    /// Build a [`hpack::DecodedHeaders`] slab directly from name/value
    /// pairs, for tests that exercise header validation without going
    /// through HPACK encoding.
    fn decoded(pairs: &[(&[u8], &[u8])]) -> hpack::DecodedHeaders {
        let mut out = hpack::DecodedHeaders::new();
        for (n, v) in pairs {
            let name_start = out.bytes.len() as u32;
            out.bytes.extend_from_slice(n);
            let value_start = out.bytes.len() as u32;
            out.bytes.extend_from_slice(v);
            let value_end = out.bytes.len() as u32;
            out.fields.push(hpack::Field {
                name: name_start..value_start,
                value: value_start..value_end,
            });
        }
        out
    }

    #[test]
    fn alpn_selects_h2_cases() {
        assert!(alpn_selects_h2(Some("h2")));
        assert!(!alpn_selects_h2(Some("http/1.1")));
        assert!(!alpn_selects_h2(None));
        assert!(!alpn_selects_h2(Some("h3")));
    }

    #[test]
    fn frame_codec_round_trip() {
        let frames = vec![
            Frame {
                ty: F_PING,
                flags: FLAG_ACK,
                stream: 0,
                payload: b"12345678".to_vec(),
            },
            Frame {
                ty: F_DATA,
                flags: FLAG_END_STREAM,
                stream: 13,
                payload: vec![9u8; 300],
            },
            Frame {
                ty: F_WINDOW_UPDATE,
                flags: 0,
                stream: 0,
                payload: 32768u32.to_be_bytes().to_vec(),
            },
        ];
        let mut bytes = Vec::new();
        for f in &frames {
            write_frame(&mut bytes, f.ty, f.flags, f.stream, &f.payload).unwrap();
        }
        let mut slice = bytes.as_slice();
        for want in &frames {
            assert_eq!(&read_frame(&mut slice).unwrap(), want);
        }
        assert!(slice.is_empty());
    }

    #[test]
    fn handshake_exchanges_preface_and_settings() {
        let io = FakeIo::new(settings_frame(&[]));
        let mut s = Session::new(io, 100);
        s.handshake(deadline()).unwrap();

        let out = &s.io.outbound;
        assert_eq!(&out[..CLIENT_PREFACE.len()], CLIENT_PREFACE);
        let frames = parse_frames(&out[CLIENT_PREFACE.len()..]);
        assert_eq!(frames.len(), 2);
        // Client SETTINGS: disables server push.
        assert_eq!(frames[0].ty, F_SETTINGS);
        assert_eq!(frames[0].flags, 0);
        assert_eq!(frames[0].stream, 0);
        assert_eq!(frames[0].payload, setting(S_ENABLE_PUSH, 0));
        // ACK of the server SETTINGS.
        assert_eq!(frames[1].ty, F_SETTINGS);
        assert_eq!(frames[1].flags, FLAG_ACK);
        assert!(frames[1].payload.is_empty());
    }

    #[test]
    fn multiplexed_interleaved_streams() {
        let blocks = encode_blocks(&[
            &[(b":status".as_slice(), b"200".as_slice())],
            &[(b":status".as_slice(), b"404".as_slice())],
        ]);
        let mut inbound = settings_frame(&[]);
        inbound.extend(frame_bytes(F_HEADERS, FLAG_END_HEADERS, 1, &blocks[0]));
        inbound.extend(frame_bytes(F_HEADERS, FLAG_END_HEADERS, 3, &blocks[1]));
        inbound.extend(frame_bytes(F_DATA, 0, 1, b"hello "));
        inbound.extend(frame_bytes(F_DATA, FLAG_END_STREAM, 3, b"oops"));
        inbound.extend(frame_bytes(F_DATA, FLAG_END_STREAM, 1, b"world"));

        let mut s = Session::new(FakeIo::new(inbound), 100);
        s.handshake(deadline()).unwrap();
        assert_eq!(s.open_stream(&get(b"/a"), true, deadline()).unwrap(), 1);
        assert_eq!(s.open_stream(&get(b"/b"), true, deadline()).unwrap(), 3);

        match s.next_event(deadline()).unwrap() {
            Event::Headers {
                stream,
                fields,
                end_stream,
            } => {
                assert_eq!(stream, 1);
                assert!(!end_stream);
                assert!(has_field(&fields, b":status", b"200"));
            }
            e => panic!("unexpected {e:?}"),
        }
        match s.next_event(deadline()).unwrap() {
            Event::Headers {
                stream, end_stream, ..
            } => {
                assert_eq!(stream, 3);
                assert!(!end_stream);
            }
            e => panic!("unexpected {e:?}"),
        }
        assert_eq!(
            s.next_event(deadline()).unwrap(),
            Event::Data {
                stream: 1,
                chunk: b"hello ".to_vec(),
                end_stream: false
            }
        );
        assert_eq!(
            s.next_event(deadline()).unwrap(),
            Event::Data {
                stream: 3,
                chunk: b"oops".to_vec(),
                end_stream: true
            }
        );
        assert_eq!(
            s.next_event(deadline()).unwrap(),
            Event::Data {
                stream: 1,
                chunk: b"world".to_vec(),
                end_stream: true
            }
        );
    }

    #[test]
    fn reap_closed_streams_drops_terminal_state() {
        let blocks = encode_blocks(&[&[(b":status".as_slice(), b"200".as_slice())]]);
        let mut inbound = settings_frame(&[]);
        inbound.extend(frame_bytes(
            F_HEADERS,
            FLAG_END_HEADERS | FLAG_END_STREAM,
            1,
            &blocks[0],
        ));
        let mut s = Session::new(FakeIo::new(inbound), 100);
        s.handshake(deadline()).unwrap();
        s.open_stream(&get(b"/a"), true, deadline()).unwrap();
        match s.next_event(deadline()).unwrap() {
            Event::Headers { end_stream, .. } => assert!(end_stream),
            e => panic!("unexpected {e:?}"),
        }
        // The completed stream sits in the map as Closed...
        assert_eq!(s.streams.len(), 1);
        s.reap_closed_streams();
        // ...and reaping drops it, so a pooled session cannot
        // accumulate one entry per fetch.
        assert!(s.streams.is_empty());
    }

    #[test]
    fn headers_on_retired_stream_are_tolerated_after_reap() {
        let blocks = encode_blocks(&[&[(b":status".as_slice(), b"200".as_slice())]]);
        let mut inbound = settings_frame(&[]);
        inbound.extend(frame_bytes(
            F_HEADERS,
            FLAG_END_HEADERS | FLAG_END_STREAM,
            1,
            &blocks[0],
        ));
        let mut s = Session::new(FakeIo::new(inbound), 100);
        s.handshake(deadline()).unwrap();
        s.open_stream(&get(b"/a"), true, deadline()).unwrap();
        let _ = s.next_event(deadline()).unwrap();
        s.reap_closed_streams();
        assert!(s.streams.is_empty());

        // A late HEADERS for the retired stream (id 1 < next_stream 3)
        // is ignored, exactly as when the Closed entry was present.
        let late = frame_bytes(F_HEADERS, FLAG_END_HEADERS | FLAG_END_STREAM, 1, &blocks[0]);
        s.on_frame(read_frame(&mut &late[..]).unwrap()).unwrap();

        // A HEADERS for a never-opened odd stream is still a connection
        // error.
        let idle = frame_bytes(F_HEADERS, FLAG_END_HEADERS | FLAG_END_STREAM, 5, &blocks[0]);
        assert!(matches!(
            s.on_frame(read_frame(&mut &idle[..]).unwrap()),
            Err(H2Error::Connection(E_PROTOCOL_ERROR, _))
        ));
    }

    #[test]
    fn send_blocks_on_window_until_update() {
        let mut inbound = settings_frame(&setting(S_INITIAL_WINDOW_SIZE, 100));
        // The peer opens the stream window mid-send.
        inbound.extend(frame_bytes(F_WINDOW_UPDATE, 0, 1, &200u32.to_be_bytes()));

        let mut s = Session::new(FakeIo::new(inbound), 100);
        s.handshake(deadline()).unwrap();
        assert_eq!(s.open_stream(&get(b"/post"), false, deadline()).unwrap(), 1);
        s.send_data(1, &[7u8; 250], deadline()).unwrap();

        let out = &s.io.outbound;
        let frames = parse_frames(&out[CLIENT_PREFACE.len()..]);
        let data: Vec<&Frame> = frames.iter().filter(|f| f.ty == F_DATA).collect();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].stream, 1);
        assert_eq!(data[0].payload.len(), 100);
        assert_eq!(data[0].flags, 0);
        assert_eq!(data[1].payload.len(), 150);
        assert_eq!(data[1].flags, FLAG_END_STREAM);
        // 100 + 150 left the connection window; the stream window is
        // exactly exhausted: 100 - 100 + 200 - 150 = 50.
        assert_eq!(s.send_window(), DEFAULT_WINDOW - 250);
    }

    #[test]
    fn recv_replenishes_windows() {
        let mut inbound = settings_frame(&[]);
        // 40000 bytes in three frames; the second drops the connection
        // window to half, triggering a WINDOW_UPDATE.
        inbound.extend(frame_bytes(F_DATA, 0, 1, &vec![1u8; 16384]));
        inbound.extend(frame_bytes(F_DATA, 0, 1, &vec![2u8; 16384]));
        inbound.extend(frame_bytes(F_DATA, FLAG_END_STREAM, 1, &vec![3u8; 7232]));

        let mut s = Session::new(FakeIo::new(inbound), 100);
        s.handshake(deadline()).unwrap();
        s.open_stream(&get(b"/"), true, deadline()).unwrap();
        for _ in 0..3 {
            assert!(matches!(
                s.next_event(deadline()).unwrap(),
                Event::Data { .. }
            ));
        }
        // 65535 - 32768 + 32768(replenish) - 7232.
        assert_eq!(s.recv_window(), 58_303);

        let frames = parse_frames(&s.io.outbound[CLIENT_PREFACE.len()..]);
        let wus: Vec<&Frame> = frames
            .iter()
            .filter(|f| f.ty == F_WINDOW_UPDATE && f.stream == 0)
            .collect();
        assert_eq!(wus.len(), 1);
        assert_eq!(wus[0].payload, 32768u32.to_be_bytes());
    }

    #[test]
    fn recv_window_violation_is_connection_error() {
        let mut s = Session::new(FakeIo::new(settings_frame(&[])), 100);
        s.handshake(deadline()).unwrap();
        s.open_stream(&get(b"/"), true, deadline()).unwrap();
        // Shrink the stream window below the incoming frame.
        s.streams.get_mut(&1).unwrap().recv_window = 50;
        s.io.inbound.extend(frame_bytes(F_DATA, 0, 1, &[0u8; 100]));
        match s.next_event(deadline()) {
            Err(H2Error::Connection(code, _)) => assert_eq!(code, E_FLOW_CONTROL_ERROR),
            r => panic!("unexpected {r:?}"),
        }
    }

    #[test]
    fn padded_data_counts_full_wire_length() {
        // PADDED frame: pad length 10, 5 data bytes, 10 padding bytes =
        // 16 wire bytes against the window, 5 delivered.
        let mut payload = vec![10u8];
        payload.extend_from_slice(b"hello");
        payload.extend(vec![0u8; 10]);
        let mut inbound = settings_frame(&[]);
        inbound.extend(frame_bytes(F_DATA, FLAG_PADDED, 1, &payload));

        let mut s = Session::new(FakeIo::new(inbound), 100);
        s.handshake(deadline()).unwrap();
        s.open_stream(&get(b"/"), true, deadline()).unwrap();
        match s.next_event(deadline()).unwrap() {
            Event::Data { chunk, .. } => assert_eq!(chunk, b"hello"),
            e => panic!("unexpected {e:?}"),
        }
        assert_eq!(s.recv_window(), DEFAULT_WINDOW - 16);
    }

    #[test]
    fn single_fragment_headers_skip_staging_buffer() {
        // The single-fragment fast path decodes straight from the frame
        // payload: the stream's `frag` staging buffer must stay
        // untouched (no alloc, no copy) instead of round-tripping the
        // header block through it.
        let blocks = encode_blocks(&[&[(b":status".as_slice(), b"200".as_slice())]]);
        let mut inbound = settings_frame(&[]);
        inbound.extend(frame_bytes(F_HEADERS, FLAG_END_HEADERS, 1, &blocks[0]));
        let mut s = Session::new(FakeIo::new(inbound), 100);
        s.handshake(deadline()).unwrap();
        s.open_stream(&get(b"/"), true, deadline()).unwrap();
        match s.next_event(deadline()).unwrap() {
            Event::Headers { fields, .. } => {
                assert!(has_field(&fields, b":status", b"200"))
            }
            e => panic!("unexpected {e:?}"),
        }
        let st = s.streams.get(&1).unwrap();
        assert!(st.frag.is_empty());
        assert_eq!(st.frag.capacity(), 0);
    }

    #[test]
    fn fragmented_headers_still_use_staging_buffer() {
        // A header block split across CONTINUATION must still
        // reassemble through the staging path: the fast path fires only
        // when nothing is staged and the block completes in one frame.
        let blocks = encode_blocks(&[&[
            (b":status".as_slice(), b"200".as_slice()),
            (b"content-type".as_slice(), b"text/html".as_slice()),
        ]]);
        assert!(blocks[0].len() >= 2);
        let (first, rest) = blocks[0].split_at(blocks[0].len() / 2);
        let mut inbound = settings_frame(&[]);
        inbound.extend(frame_bytes(F_HEADERS, 0, 1, first));
        inbound.extend(frame_bytes(F_CONTINUATION, FLAG_END_HEADERS, 1, rest));
        let mut s = Session::new(FakeIo::new(inbound), 100);
        s.handshake(deadline()).unwrap();
        s.open_stream(&get(b"/"), true, deadline()).unwrap();
        match s.next_event(deadline()).unwrap() {
            Event::Headers { fields, .. } => {
                assert!(has_field(&fields, b":status", b"200"));
                assert!(has_field(&fields, b"content-type", b"text/html"))
            }
            e => panic!("unexpected {e:?}"),
        }
        // The staging buffer was consumed by the decode (taken, empty).
        assert!(s.streams.get(&1).unwrap().frag.is_empty());
    }

    #[test]
    fn header_scratch_reused_across_response_heads() {
        // Two identical response heads. Each decode takes the session's
        // scratch slab into the queued event, leaving a fresh one
        // behind; handing a consumed slab back lets the next decode
        // reuse it in place with no growth.
        let head: &[(&[u8], &[u8])] = &[
            (b":status".as_slice(), b"200".as_slice()),
            (b"content-type".as_slice(), b"text/html".as_slice()),
        ];
        let blocks = encode_blocks(&[head, head]);
        let mut inbound = settings_frame(&[]);
        inbound.extend(frame_bytes(F_HEADERS, FLAG_END_HEADERS, 1, &blocks[0]));
        let mut s = Session::new(FakeIo::new(inbound), 100);
        s.handshake(deadline()).unwrap();
        s.open_stream(&get(b"/a"), true, deadline()).unwrap();
        s.open_stream(&get(b"/b"), true, deadline()).unwrap();

        // First head: the scratch moves into the event...
        let caps = match s.next_event(deadline()).unwrap() {
            Event::Headers { fields, .. } => {
                assert!(has_field(&fields, b":status", b"200"));
                assert!(has_field(&fields, b"content-type", b"text/html"));
                // ...leaving a fresh slab behind in the session.
                assert_eq!(s.header_scratch.bytes.capacity(), 0);
                assert_eq!(s.header_scratch.fields.capacity(), 0);
                let caps = (fields.bytes.capacity(), fields.fields.capacity());
                // Steady-state recycle: hand the consumed slab back.
                s.header_scratch = fields;
                caps
            }
            e => panic!("unexpected {e:?}"),
        };
        assert!(caps.0 > 0 && caps.1 > 0);

        // Second head decodes into the recycled slab.
        s.io.inbound
            .extend(frame_bytes(F_HEADERS, FLAG_END_HEADERS, 3, &blocks[1]));
        match s.next_event(deadline()).unwrap() {
            Event::Headers { fields, .. } => {
                assert!(has_field(&fields, b":status", b"200"));
                assert!(has_field(&fields, b"content-type", b"text/html"));
                // Capacity preserved: the identical block fit the reused
                // slab with no growth.
                assert_eq!(fields.bytes.capacity(), caps.0);
                assert_eq!(fields.fields.capacity(), caps.1);
            }
            e => panic!("unexpected {e:?}"),
        }
    }

    #[test]
    fn ping_is_acked() {
        let mut inbound = settings_frame(&[]);
        inbound.extend(frame_bytes(F_PING, 0, 0, b"abcdefgh"));
        let mut s = Session::new(FakeIo::new(inbound), 100);
        s.handshake(deadline()).unwrap();
        // The PING is absorbed; the script then runs dry.
        let _ = s.next_event(deadline());
        let frames = parse_frames(&s.io.outbound[CLIENT_PREFACE.len()..]);
        let acks: Vec<&Frame> = frames
            .iter()
            .filter(|f| f.ty == F_PING && f.flags & FLAG_ACK != 0)
            .collect();
        assert_eq!(acks.len(), 1);
        assert_eq!(acks[0].payload, b"abcdefgh");
    }

    #[test]
    fn push_promise_is_cancelled() {
        let blocks = encode_blocks(&[&[
            (b":method".as_slice(), b"GET".as_slice()),
            (b":scheme".as_slice(), b"https".as_slice()),
            (b":path".as_slice(), b"/p".as_slice()),
            (b":authority".as_slice(), b"x.test".as_slice()),
        ]]);
        let mut promise = 2u32.to_be_bytes().to_vec();
        promise.extend_from_slice(&blocks[0]);

        // Promise with END_HEADERS: single RST, no tracking needed.
        let mut inbound = settings_frame(&[]);
        inbound.extend(frame_bytes(F_PUSH_PROMISE, FLAG_END_HEADERS, 1, &promise));
        let mut s = Session::new(FakeIo::new(inbound), 100);
        s.handshake(deadline()).unwrap();
        s.open_stream(&get(b"/"), true, deadline()).unwrap();
        let _ = s.next_event(deadline());
        let frames = parse_frames(&s.io.outbound[CLIENT_PREFACE.len()..]);
        let rsts: Vec<&Frame> = frames.iter().filter(|f| f.ty == F_RST_STREAM).collect();
        assert_eq!(rsts.len(), 1);
        assert_eq!(rsts[0].stream, 2);
        assert_eq!(rsts[0].payload, E_CANCEL.to_be_bytes());
    }

    #[test]
    fn push_promise_continuations_are_discarded() {
        let blocks = encode_blocks(&[&[
            (b":method".as_slice(), b"GET".as_slice()),
            (b":scheme".as_slice(), b"https".as_slice()),
            (b":path".as_slice(), b"/p".as_slice()),
            (b":authority".as_slice(), b"x.test".as_slice()),
        ]]);
        let (first, rest) = blocks[0].split_at(blocks[0].len() / 2);
        let mut promise = 2u32.to_be_bytes().to_vec();
        promise.extend_from_slice(first);

        let mut inbound = settings_frame(&[]);
        inbound.extend(frame_bytes(F_PUSH_PROMISE, 0, 1, &promise));
        inbound.extend(frame_bytes(F_CONTINUATION, FLAG_END_HEADERS, 2, rest));
        let mut s = Session::new(FakeIo::new(inbound), 100);
        s.handshake(deadline()).unwrap();
        s.open_stream(&get(b"/"), true, deadline()).unwrap();
        // Both frames absorbed (RST + discarded CONTINUATION), then dry.
        let _ = s.next_event(deadline());
        let _ = s.next_event(deadline());
        let frames = parse_frames(&s.io.outbound[CLIENT_PREFACE.len()..]);
        assert_eq!(frames.iter().filter(|f| f.ty == F_RST_STREAM).count(), 1);
    }

    #[test]
    fn goaway_surfaces_and_blocks_new_streams() {
        let mut inbound = settings_frame(&[]);
        let mut payload = 1u32.to_be_bytes().to_vec();
        payload.extend_from_slice(&0u32.to_be_bytes());
        inbound.extend(frame_bytes(F_GOAWAY, 0, 0, &payload));

        let mut s = Session::new(FakeIo::new(inbound), 100);
        s.handshake(deadline()).unwrap();
        s.open_stream(&get(b"/"), true, deadline()).unwrap();
        let _ = s.next_event(deadline());
        assert_eq!(s.goaway(), Some((1, 0)));
        match s.open_stream(&get(b"/"), true, deadline()) {
            Err(H2Error::GoAway(0)) => {}
            r => panic!("unexpected {r:?}"),
        }
    }

    #[test]
    fn interleaved_frame_in_header_block_is_connection_error() {
        let mut inbound = settings_frame(&[]);
        // HEADERS without END_HEADERS, then a PING before the
        // CONTINUATION: a sequencing violation.
        inbound.extend(frame_bytes(F_HEADERS, 0, 1, b"partial"));
        inbound.extend(frame_bytes(F_PING, 0, 0, b"12345678"));
        let mut s = Session::new(FakeIo::new(inbound), 100);
        s.handshake(deadline()).unwrap();
        s.open_stream(&get(b"/"), true, deadline()).unwrap();
        match s.next_event(deadline()) {
            Err(H2Error::Connection(code, _)) => assert_eq!(code, E_PROTOCOL_ERROR),
            r => panic!("unexpected {r:?}"),
        }
    }

    #[test]
    fn stray_continuation_is_connection_error() {
        let mut inbound = settings_frame(&[]);
        inbound.extend(frame_bytes(F_CONTINUATION, FLAG_END_HEADERS, 1, b"nope"));
        let mut s = Session::new(FakeIo::new(inbound), 100);
        s.handshake(deadline()).unwrap();
        s.open_stream(&get(b"/"), true, deadline()).unwrap();
        match s.next_event(deadline()) {
            Err(H2Error::Connection(code, _)) => assert_eq!(code, E_PROTOCOL_ERROR),
            r => panic!("unexpected {r:?}"),
        }
    }

    #[test]
    fn response_header_validation() {
        let ok = decoded(&[
            (b":status".as_slice(), b"200".as_slice()),
            (b"content-type".as_slice(), b"text/html".as_slice()),
            (b"te".as_slice(), b"trailers".as_slice()),
        ]);
        let (status, fields) = parse_response_headers(&ok).unwrap();
        assert_eq!(status, 200);
        assert_eq!(fields.len(), 2);

        // Missing :status.
        assert!(parse_response_headers(&decoded(&[(b"x".as_slice(), b"y".as_slice())])).is_err());
        // Duplicate :status.
        assert!(
            parse_response_headers(&decoded(&[
                (b":status".as_slice(), b"200".as_slice()),
                (b":status".as_slice(), b"200".as_slice()),
            ]))
            .is_err()
        );
        // Malformed :status.
        assert!(
            parse_response_headers(&decoded(&[(b":status".as_slice(), b"99".as_slice())])).is_err()
        );
        // Connection-specific header.
        assert!(
            parse_response_headers(&decoded(&[
                (b":status".as_slice(), b"200".as_slice()),
                (b"transfer-encoding".as_slice(), b"chunked".as_slice()),
            ]))
            .is_err()
        );
        // Non-trailers TE.
        assert!(
            parse_response_headers(&decoded(&[
                (b":status".as_slice(), b"200".as_slice()),
                (b"te".as_slice(), b"gzip".as_slice()),
            ]))
            .is_err()
        );
        // Pseudo-header after a regular one.
        assert!(
            parse_response_headers(&decoded(&[
                (b"x".as_slice(), b"y".as_slice()),
                (b":status".as_slice(), b"200".as_slice()),
            ]))
            .is_err()
        );
    }

    #[test]
    fn content_length_rules() {
        let one = |v: &str| ("content-length".to_owned(), v.as_bytes().to_vec());
        assert_eq!(declared_content_length(&[one("5")]).unwrap(), Some(5));
        assert_eq!(
            declared_content_length(&[one("5"), one("5")]).unwrap(),
            Some(5)
        );
        assert!(declared_content_length(&[one("5"), one("6")]).is_err());
        assert!(declared_content_length(&[one("nope")]).is_err());
        assert_eq!(declared_content_length(&[]).unwrap(), None);
    }

    // -- SETTINGS_MAX_CONCURRENT_STREAMS ----------------------------------

    fn settings_with_max_concurrent(n: u32) -> Vec<u8> {
        settings_frame(&setting(S_MAX_CONCURRENT_STREAMS, n))
    }

    #[test]
    fn max_concurrent_defaults_to_unlimited() {
        let mut s = Session::new(FakeIo::new(settings_frame(&[])), 100);
        s.handshake(deadline()).unwrap();
        assert_eq!(s.peer_max_concurrent, u32::MAX);
        for _ in 0..8 {
            s.open_stream(&get(b"/"), true, deadline()).unwrap();
        }
    }

    #[test]
    fn max_concurrent_streams_enforced() {
        let mut s = Session::new(FakeIo::new(settings_with_max_concurrent(1)), 100);
        s.handshake(deadline()).unwrap();
        assert_eq!(s.peer_max_concurrent, 1);
        s.open_stream(&get(b"/a"), true, deadline()).unwrap();
        match s.open_stream(&get(b"/b"), true, deadline()) {
            Err(H2Error::ConcurrentStreamsExceeded) => {}
            r => panic!("expected ConcurrentStreamsExceeded, got {r:?}"),
        }
        // A zero limit refuses every new stream.
        let mut s = Session::new(FakeIo::new(settings_with_max_concurrent(0)), 100);
        s.handshake(deadline()).unwrap();
        assert!(matches!(
            s.open_stream(&get(b"/"), true, deadline()),
            Err(H2Error::ConcurrentStreamsExceeded)
        ));
    }

    #[test]
    fn closed_streams_do_not_count_against_limit() {
        let blocks = encode_blocks(&[&[(b":status".as_slice(), b"200".as_slice())]]);
        let mut inbound = settings_with_max_concurrent(1);
        inbound.extend(frame_bytes(
            F_HEADERS,
            FLAG_END_HEADERS | FLAG_END_STREAM,
            1,
            &blocks[0],
        ));
        let mut s = Session::new(FakeIo::new(inbound), 100);
        s.handshake(deadline()).unwrap();
        s.open_stream(&get(b"/a"), true, deadline()).unwrap();
        // The response HEADERS with END_STREAM closes stream 1...
        match s.next_event(deadline()).unwrap() {
            Event::Headers {
                stream: 1,
                end_stream: true,
                ..
            } => {}
            ev => panic!("unexpected {ev:?}"),
        }
        // ...so it no longer counts and a new stream may open.
        assert_eq!(s.open_stream(&get(b"/b"), true, deadline()).unwrap(), 3);
    }

    #[test]
    fn max_concurrent_streams_updated_by_later_settings() {
        let mut s = Session::new(FakeIo::new(settings_with_max_concurrent(100)), 100);
        s.handshake(deadline()).unwrap();
        s.open_stream(&get(b"/a"), true, deadline()).unwrap();
        s.open_stream(&get(b"/b"), true, deadline()).unwrap();
        // The peer lowers the limit mid-session: the two open streams
        // are unaffected, but new opens are refused.
        s.io.inbound
            .extend(settings_frame(&setting(S_MAX_CONCURRENT_STREAMS, 2)));
        s.pump(deadline()).unwrap();
        assert_eq!(s.peer_max_concurrent, 2);
        match s.open_stream(&get(b"/c"), true, deadline()) {
            Err(H2Error::ConcurrentStreamsExceeded) => {}
            r => panic!("expected ConcurrentStreamsExceeded, got {r:?}"),
        }
    }

    // -- PING ------------------------------------------------------------------

    #[test]
    fn send_ping_uses_fresh_payloads() {
        let mut s = Session::new(FakeIo::new(settings_frame(&[])), 100);
        s.handshake(deadline()).unwrap();
        let p0 = s.send_ping().unwrap();
        let p1 = s.send_ping().unwrap();
        assert_eq!(p0, 0u64.to_be_bytes());
        assert_eq!(p1, 1u64.to_be_bytes());
        let frames = parse_frames(&s.io.outbound[CLIENT_PREFACE.len()..]);
        let pings: Vec<&Frame> = frames.iter().filter(|f| f.ty == F_PING).collect();
        assert_eq!(pings.len(), 2);
        for p in &pings {
            assert_eq!(p.flags, 0);
            assert_eq!(p.stream, 0);
        }
        assert_eq!(pings[0].payload, p0);
        assert_eq!(pings[1].payload, p1);
    }
}
