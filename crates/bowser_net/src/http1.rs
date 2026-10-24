//! HTTP/1.1 message codec (RFC 9110, RFC 9112).
//!
//! Parses and serializes HTTP/1.x request/response heads, implements
//! chunked transfer-coding (one-shot and streaming), and answers the
//! framing questions every connection needs: how long is the body, and
//! does the connection persist?
//!
//! The design is zero-copy where it matters: [`parse_request_head`] and
//! [`parse_response_head`] borrow the caller's buffer, [`decode_chunked`]
//! hands back slices of the input for chunk data, and the streaming
//! [`ChunkedDecoder`] emits [`ChunkEvent::Data`] slices that borrow the
//! fed input. Only trailer fields in the streaming decoder are copied
//! (they are small and rare).
//!
//! Strictness notes (deliberate, RFC 9112 §5):
//! * Line endings: CRLF is the wire format; a lone LF is tolerated
//!   everywhere a line ending is expected (start-line, header lines,
//!   chunk-size lines, chunk-data terminators, trailers). A bare CR
//!   (a CR not followed by LF) is always rejected.
//! * `obs-fold` (a header line starting with SP/HTAB) is rejected, not
//!   unfolded.
//! * Whitespace between a field name and the colon is rejected.
//! * `Transfer-Encoding: chunked` must be the final coding for framing to
//!   apply. On requests, anything else is an error; on responses, any
//!   `Transfer-Encoding` without `chunked` final means the body runs until
//!   the connection closes (RFC 9112 §6.3 rules 2, 4).

use std::error::Error;
use std::fmt;
use std::str;

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/// Resource limits applied while parsing a message head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Maximum number of header fields in one head (headers + trailers).
    pub max_headers: usize,
    /// Maximum total bytes of one head, start-line included.
    pub max_head_bytes: usize,
    /// Maximum bytes of a single line (start-line, header, chunk-size...).
    pub max_line_bytes: usize,
    /// Maximum bytes of a message body: either the `Content-Length` value,
    /// or the total decoded payload of a chunked body.
    ///
    /// The default is 256 MiB. A larger `Content-Length` is rejected before
    /// any downstream reader can size an allocation by it; a chunked body
    /// is rejected once its cumulative payload exceeds the cap. The fetch
    /// layer may tighten this per request or response.
    pub max_body_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_headers: 100,
            max_head_bytes: 32 * 1024,
            max_line_bytes: 8 * 1024,
            max_body_bytes: 256 * 1024 * 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Every way HTTP/1.1 message parsing or serialization can fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpError {
    /// More input is needed before a decision can be made.
    Incomplete,
    /// A single line exceeded [`Limits::max_line_bytes`].
    LineTooLong,
    /// The head exceeded [`Limits::max_head_bytes`].
    HeadersTooLarge,
    /// More than [`Limits::max_headers`] header fields were seen.
    TooManyHeaders,
    /// The request-line is not `method SP target SP version`.
    InvalidRequestLine,
    /// The status-line is not `version SP status SP reason`.
    InvalidStatusLine,
    /// The version is not `HTTP/DIGIT.DIGIT`.
    InvalidVersion,
    /// The method is not a valid token.
    InvalidMethod,
    /// The request-target is empty or contains illegal bytes.
    InvalidTarget,
    /// The status code is not exactly three ASCII digits.
    InvalidStatus,
    /// A header field-name is empty or not a valid token.
    InvalidHeaderName,
    /// A header line has no colon, or has whitespace before the colon.
    MissingHeaderColon,
    /// A header field-value contains illegal bytes (CTL, DEL, bare CR).
    InvalidHeaderValue,
    /// A line starting with SP/HTAB (`obs-fold`) was found; rejected.
    ObsFoldRejected,
    /// A CR not followed by LF was found.
    BareCr,
    /// A line was terminated by a bare LF instead of CRLF (RFC 9112
    /// §2.2: recipients MUST NOT tolerate bare line terminators).
    BareLf,
    /// A `Content-Length` value is not a valid non-negative integer.
    InvalidContentLength,
    /// A `Content-Length` value exceeds [`Limits::max_body_bytes`], or a
    /// chunked body's cumulative payload exceeded it.
    BodyTooLarge,
    /// Multiple `Content-Length` values disagree with each other.
    ConflictingContentLength,
    /// A `Transfer-Encoding` list is syntactically invalid.
    InvalidTransferEncoding,
    /// A `Transfer-Encoding` coding other than `chunked` was seen.
    UnsupportedTransferEncoding,
    /// A `Transfer-Encoding` coding was `identity`, which is not a valid
    /// transfer coding (RFC 9112 §6.1). It must not be filtered as a
    /// no-op: that would let the message fall through to
    /// `Content-Length` framing.
    IdentityTransferEncoding,
    /// `chunked` appears but is not the final transfer coding.
    ChunkedNotFinal,
    /// A `Transfer-Encoding` field was present on an HTTP/1.0 (or earlier)
    /// message; the framing must be treated as faulty (RFC 9112 §6.1).
    TransferEncodingOnHttp10,
    /// A chunk-size line has no valid hexadecimal size.
    InvalidChunkSize,
    /// A chunk-size exceeds [`MAX_CHUNK_SIZE`]. Chunk sizes parse into a
    /// u64, so without a cap a hostile size line could overflow the
    /// `usize` arithmetic used while slicing chunk data.
    ChunkTooLarge,
    /// A chunk extension is malformed.
    InvalidChunkExtension,
    /// Chunk data is not followed by CRLF.
    InvalidChunkTerminator,
    /// A trailer field is one a trailer must not contain
    /// (`Content-Length`, `Transfer-Encoding`, `Trailer`, `TE`).
    ForbiddenTrailer,
}

impl fmt::Display for HttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::Incomplete => "incomplete HTTP message",
            Self::LineTooLong => "HTTP line exceeds size limit",
            Self::HeadersTooLarge => "HTTP head exceeds size limit",
            Self::TooManyHeaders => "too many HTTP header fields",
            Self::InvalidRequestLine => "malformed HTTP request-line",
            Self::InvalidStatusLine => "malformed HTTP status-line",
            Self::InvalidVersion => "malformed HTTP version",
            Self::InvalidMethod => "invalid HTTP method",
            Self::InvalidTarget => "invalid request-target",
            Self::InvalidStatus => "invalid HTTP status code",
            Self::InvalidHeaderName => "invalid HTTP header field-name",
            Self::MissingHeaderColon => "HTTP header line without colon",
            Self::InvalidHeaderValue => "invalid HTTP header field-value",
            Self::ObsFoldRejected => "obsolete line folding (obs-fold) is not accepted",
            Self::BareCr => "bare CR in HTTP message",
            Self::BareLf => "bare LF in HTTP message",
            Self::InvalidContentLength => "invalid Content-Length value",
            Self::BodyTooLarge => "HTTP message body exceeds size limit",
            Self::ConflictingContentLength => "conflicting Content-Length values",
            Self::InvalidTransferEncoding => "invalid Transfer-Encoding value",
            Self::UnsupportedTransferEncoding => "unsupported transfer coding",
            Self::IdentityTransferEncoding => {
                "Transfer-Encoding: identity is not a valid transfer coding"
            }
            Self::ChunkedNotFinal => "chunked is not the final transfer coding",
            Self::TransferEncodingOnHttp10 => {
                "Transfer-Encoding is not allowed in HTTP/1.0 messages"
            }
            Self::InvalidChunkSize => "invalid chunk-size",
            Self::ChunkTooLarge => "chunk size exceeds limit",
            Self::InvalidChunkExtension => "invalid chunk extension",
            Self::InvalidChunkTerminator => "chunk data not followed by CRLF or LF",
            Self::ForbiddenTrailer => "forbidden field in chunked trailer",
        };
        f.write_str(msg)
    }
}

impl Error for HttpError {}

// ---------------------------------------------------------------------------
// HTTP version
// ---------------------------------------------------------------------------

/// An HTTP version, e.g. `HTTP/1.1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HttpVersion {
    pub major: u8,
    pub minor: u8,
}

impl HttpVersion {
    pub const HTTP_10: Self = Self { major: 1, minor: 0 };
    pub const HTTP_11: Self = Self { major: 1, minor: 1 };
}

impl fmt::Display for HttpVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HTTP/{}.{}", self.major, self.minor)
    }
}

fn parse_version(bytes: &[u8]) -> Result<HttpVersion, HttpError> {
    // Must be exactly `HTTP/` DIGIT `.` DIGIT.
    if bytes.len() != 8
        || &bytes[0..5] != b"HTTP/"
        || !bytes[5].is_ascii_digit()
        || bytes[6] != b'.'
        || !bytes[7].is_ascii_digit()
    {
        return Err(HttpError::InvalidVersion);
    }
    Ok(HttpVersion {
        major: bytes[5] - b'0',
        minor: bytes[7] - b'0',
    })
}

// ---------------------------------------------------------------------------
// Byte-level helpers
// ---------------------------------------------------------------------------

/// `tchar` from RFC 9110 §5.6.2.
fn is_tchar(b: u8) -> bool {
    b.is_ascii_alphanumeric()
        || matches!(
            b,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

/// Non-empty run of `tchar`s.
pub(crate) fn is_token(bytes: &[u8]) -> bool {
    !bytes.is_empty() && bytes.iter().all(|&b| is_tchar(b))
}

/// Strip optional whitespace (SP / HTAB) from both ends.
fn trim_ows(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|&b| b != b' ' && b != b'\t')
        .unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|&b| b != b' ' && b != b'\t')
        .map_or(0, |i| i + 1);
    // On all-whitespace input `start == len` and `end == 0`; clamp so the
    // slice is empty instead of `start > end` (which would panic).
    &bytes[start..end.max(start)]
}

/// A field-value may hold HTAB, VCHAR, and obs-text — nothing else.
fn validate_field_value(value: &[u8]) -> Result<(), HttpError> {
    for &b in value {
        let ok = b == b'\t' || (0x20..=0x7e).contains(&b) || b >= 0x80;
        if !ok {
            return Err(HttpError::InvalidHeaderValue);
        }
    }
    Ok(())
}

fn validate_field_name(name: &[u8]) -> Result<(), HttpError> {
    if !is_token(name) {
        return Err(HttpError::InvalidHeaderName);
    }
    Ok(())
}

/// Request-target: visible ASCII or obs-text, never whitespace or CTLs.
///
/// Leniency (deliberate): raw non-ASCII bytes (≥ 0x80) are accepted even
/// though RFC 3986 forbids them in a request-target — clients in the wild
/// send raw UTF-8 without percent-encoding it. Callers using this parser
/// on the server side should pre-validate the target if strict RFC 3986
/// conformance matters to them.
fn valid_target(target: &[u8]) -> bool {
    !target.is_empty()
        && target
            .iter()
            .all(|&b| (0x21..=0x7e).contains(&b) || b >= 0x80)
}

/// Reason-phrase: HTAB / SP / VCHAR / obs-text.
fn valid_reason(reason: &[u8]) -> bool {
    reason
        .iter()
        .all(|&b| b == b'\t' || (0x20..=0x7e).contains(&b) || b >= 0x80)
}

/// A physical line: its bytes (terminator stripped) and bytes consumed.
struct Line<'a> {
    bytes: &'a [u8],
    consumed: usize,
}

/// Split one line off the front of `input`.
///
/// Requires CRLF (RFC 9112 §2.2: a recipient MUST NOT tolerate a bare LF
/// or a bare CR as a line terminator). A CR at the very end of the
/// buffer may be a split CRLF — that yields [`HttpError::Incomplete`] so
/// the caller can feed more.
fn next_line(input: &[u8], max_line_bytes: usize) -> Result<Line<'_>, HttpError> {
    match input.iter().position(|&b| b == b'\n') {
        Some(lf) => {
            if lf == 0 || input[lf - 1] != b'\r' {
                return Err(HttpError::BareLf);
            }
            let line = &input[..lf - 1];
            if line.contains(&b'\r') {
                return Err(HttpError::BareCr);
            }
            if line.len() > max_line_bytes {
                return Err(HttpError::LineTooLong);
            }
            Ok(Line {
                bytes: line,
                consumed: lf + 1,
            })
        }
        None => {
            if input.len() > max_line_bytes {
                return Err(HttpError::LineTooLong);
            }
            for (i, &b) in input.iter().enumerate() {
                if b == b'\r' && i + 1 < input.len() {
                    return Err(HttpError::BareCr);
                }
            }
            Err(HttpError::Incomplete)
        }
    }
}

// ---------------------------------------------------------------------------
// Headers
// ---------------------------------------------------------------------------

/// One header field, borrowed from the parsed buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header<'a> {
    /// Field-name, exactly as sent (compare case-insensitively).
    pub name: &'a str,
    /// Field-value with surrounding OWS stripped.
    pub value: &'a [u8],
}

/// An ordered list of header fields. Duplicates are preserved; lookups are
/// case-insensitive and [`get`](Headers::get) returns the first match.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Headers<'a> {
    headers: Vec<Header<'a>>,
}

impl<'a> Headers<'a> {
    pub fn new() -> Self {
        Self {
            headers: Vec::new(),
        }
    }

    /// Push a field, validating name and value first.
    pub fn push(&mut self, name: &'a str, value: &'a [u8]) -> Result<(), HttpError> {
        validate_field_name(name.as_bytes())?;
        validate_field_value(value)?;
        self.headers.push(Header { name, value });
        Ok(())
    }

    /// First field-value for `name` (case-insensitive), if any.
    pub fn get(&self, name: &str) -> Option<&'a [u8]> {
        self.get_all(name).next()
    }

    /// All field-values for `name` (case-insensitive), in order.
    ///
    /// The returned iterator borrows only from `self`; the `name` borrow
    /// ends when this function returns, so the iterator may be held past
    /// `name`'s lifetime.
    pub fn get_all<'s, 'n>(&'s self, name: &'n str) -> impl Iterator<Item = &'a [u8]> + 's {
        // Own a normalized copy of `name` so the iterator does not capture
        // the `&'n str` borrow; items borrow from `self` alone.
        let name_lower = name.to_ascii_lowercase();
        self.headers
            .iter()
            .filter(move |h| h.name.eq_ignore_ascii_case(&name_lower))
            .map(|h| h.value)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Header<'a>> {
        self.headers.iter()
    }

    pub fn len(&self) -> usize {
        self.headers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.headers.is_empty()
    }
}

/// Parse one header line into a field. Rejects `obs-fold`, missing colons,
/// whitespace before the colon, bad names, and bad values.
fn parse_header_line(line: &[u8]) -> Result<Header<'_>, HttpError> {
    if line.first().is_some_and(|b| *b == b' ' || *b == b'\t') {
        return Err(HttpError::ObsFoldRejected);
    }
    let colon = line
        .iter()
        .position(|&b| b == b':')
        .ok_or(HttpError::MissingHeaderColon)?;
    let (name, rest) = line.split_at(colon);
    validate_field_name(name)?;
    // A token is pure ASCII, so this cannot fail; keep the error regardless.
    let name = str::from_utf8(name).map_err(|_| HttpError::InvalidHeaderName)?;
    let value = trim_ows(&rest[1..]);
    validate_field_value(value)?;
    Ok(Header { name, value })
}

/// Parse header fields up to and including the blank line. `base` is the
/// number of head bytes already consumed (the start-line), counted against
/// [`Limits::max_head_bytes`]. Returns the fields and bytes consumed.
fn parse_headers<'a>(
    input: &'a [u8],
    limits: &Limits,
    base: usize,
) -> Result<(Headers<'a>, usize), HttpError> {
    let mut headers = Headers::new();
    let mut pos = 0usize;
    loop {
        let line = next_line(&input[pos..], limits.max_line_bytes)?;
        pos += line.consumed;
        if base + pos > limits.max_head_bytes {
            return Err(HttpError::HeadersTooLarge);
        }
        if line.bytes.is_empty() {
            break;
        }
        if headers.len() >= limits.max_headers {
            return Err(HttpError::TooManyHeaders);
        }
        let field = parse_header_line(line.bytes)?;
        headers.headers.push(field);
    }
    Ok((headers, pos))
}

// ---------------------------------------------------------------------------
// Request / response heads
// ---------------------------------------------------------------------------

/// A parsed request head: request-line plus header fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestHead<'a> {
    pub method: &'a str,
    pub target: &'a str,
    pub version: HttpVersion,
    pub headers: Headers<'a>,
}

/// A parsed response head: status-line plus header fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseHead<'a> {
    pub version: HttpVersion,
    pub status: u16,
    pub reason: &'a str,
    pub headers: Headers<'a>,
}

fn parse_request_line(line: &[u8]) -> Result<(&str, &str, HttpVersion), HttpError> {
    let mut parts = line.split(|&b| b == b' ');
    let method = parts.next().ok_or(HttpError::InvalidRequestLine)?;
    let target = parts.next().ok_or(HttpError::InvalidRequestLine)?;
    let version = parts.next().ok_or(HttpError::InvalidRequestLine)?;
    if parts.next().is_some() {
        return Err(HttpError::InvalidRequestLine);
    }
    if !is_token(method) {
        return Err(HttpError::InvalidMethod);
    }
    if !valid_target(target) {
        return Err(HttpError::InvalidTarget);
    }
    let version = parse_version(version)?;
    let method = str::from_utf8(method).map_err(|_| HttpError::InvalidMethod)?;
    let target = str::from_utf8(target).map_err(|_| HttpError::InvalidTarget)?;
    Ok((method, target, version))
}

fn parse_status_line(line: &[u8]) -> Result<(HttpVersion, u16, &str), HttpError> {
    let space = line
        .iter()
        .position(|&b| b == b' ')
        .ok_or(HttpError::InvalidStatusLine)?;
    let version = parse_version(&line[..space])?;
    let rest = &line[space + 1..];
    if rest.len() < 3 || !rest[..3].iter().all(|b| b.is_ascii_digit()) {
        return Err(HttpError::InvalidStatus);
    }
    let status = u16::from(rest[0] - b'0') * 100
        + u16::from(rest[1] - b'0') * 10
        + u16::from(rest[2] - b'0');
    let reason = match &rest[3..] {
        [] => "",
        [b' ', phrase @ ..] => {
            if !valid_reason(phrase) {
                return Err(HttpError::InvalidStatusLine);
            }
            str::from_utf8(phrase).map_err(|_| HttpError::InvalidStatusLine)?
        }
        _ => return Err(HttpError::InvalidStatusLine),
    };
    Ok((version, status, reason))
}

/// Parse a request head. Returns the head and the number of bytes consumed
/// (the body, if any, starts there). [`HttpError::Incomplete`] means: feed
/// more bytes and retry.
///
/// The request-target accepts raw non-ASCII bytes; see [`valid_target`]
/// for the deliberate leniency rationale.
pub fn parse_request_head<'a>(
    input: &'a [u8],
    limits: &Limits,
) -> Result<(RequestHead<'a>, usize), HttpError> {
    let line = next_line(input, limits.max_line_bytes)?;
    if line.consumed > limits.max_head_bytes {
        return Err(HttpError::HeadersTooLarge);
    }
    let (method, target, version) = parse_request_line(line.bytes)?;
    let (headers, hlen) = parse_headers(&input[line.consumed..], limits, line.consumed)?;
    Ok((
        RequestHead {
            method,
            target,
            version,
            headers,
        },
        line.consumed + hlen,
    ))
}

/// Parse a response head. Returns the head and the number of bytes consumed.
/// [`HttpError::Incomplete`] means: feed more bytes and retry.
pub fn parse_response_head<'a>(
    input: &'a [u8],
    limits: &Limits,
) -> Result<(ResponseHead<'a>, usize), HttpError> {
    let line = next_line(input, limits.max_line_bytes)?;
    if line.consumed > limits.max_head_bytes {
        return Err(HttpError::HeadersTooLarge);
    }
    let (version, status, reason) = parse_status_line(line.bytes)?;
    let (headers, hlen) = parse_headers(&input[line.consumed..], limits, line.consumed)?;
    Ok((
        ResponseHead {
            version,
            status,
            reason,
            headers,
        },
        line.consumed + hlen,
    ))
}

// ---------------------------------------------------------------------------
// Message framing: Content-Length, Transfer-Encoding, body length
// ---------------------------------------------------------------------------

/// How the body of a message is delimited on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodySpec {
    /// No body bytes follow the head.
    Empty,
    /// Exactly this many bytes follow the head.
    ContentLength(u64),
    /// The body uses chunked transfer-coding.
    Chunked,
    /// The body runs until the connection closes.
    UntilClose,
    /// The connection becomes an opaque tunnel after the head
    /// (2xx response to CONNECT, RFC 9112 §6.3 rule 2).
    /// Framing headers are ignored; the bytes after the head are
    /// not an HTTP/1.x body.
    Tunnel,
}

fn parse_decimal(digits: &[u8]) -> Option<u64> {
    let mut n: u64 = 0;
    for &b in digits {
        n = n.checked_mul(10)?.checked_add(u64::from(b - b'0'))?;
    }
    Some(n)
}

/// The effective `Content-Length`: every value, across every field, must be
/// a valid decimal integer and they must all agree (RFC 9110 §8.6).
/// Returns `None` when no `Content-Length` is present.
///
/// A value larger than [`Limits::max_body_bytes`] is rejected with
/// [`HttpError::BodyTooLarge`] before any downstream reader can size an
/// allocation by it.
pub fn content_length(headers: &Headers, limits: &Limits) -> Result<Option<u64>, HttpError> {
    let mut result: Option<u64> = None;
    for value in headers.get_all("content-length") {
        for part in value.split(|&b| b == b',') {
            let part = trim_ows(part);
            if part.is_empty() || !part.iter().all(|b| b.is_ascii_digit()) {
                return Err(HttpError::InvalidContentLength);
            }
            let n = parse_decimal(part).ok_or(HttpError::InvalidContentLength)?;
            if n > limits.max_body_bytes as u64 {
                return Err(HttpError::BodyTooLarge);
            }
            match result {
                Some(prev) if prev != n => {
                    return Err(HttpError::ConflictingContentLength);
                }
                Some(_) => {}
                None => result = Some(n),
            }
        }
    }
    Ok(result)
}

enum TransferOutcome {
    /// No `Transfer-Encoding` field is present. (Any present field now
    /// always resolves to a coding outcome or an error — `identity` is
    /// rejected, never filtered — so TE presence can never fall through
    /// to `Content-Length`.)
    None,
    Chunked,
    /// `Transfer-Encoding` is present but `chunked` is not the final coding
    /// (possibly absent entirely). On responses this means the body runs
    /// until the connection closes (RFC 9112 §6.3 rule 4).
    UntilClose,
}

/// Which side sent the message; RFC 9112 §6.3 frames Transfer-Encoding
/// differently for requests (rule 2) and responses (rule 4).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MessageDirection {
    Request,
    Response,
}

/// The effective transfer coding. `chunked` must appear exactly once and,
/// when present, be the final coding; it is the only framing this layer
/// applies (any other content coding is a decoding concern above it).
/// When `chunked` is not the final coding (possibly absent entirely),
/// requests are rejected (RFC 9112 §6.3 rule 2) while responses read until
/// the connection closes (rule 4).
/// When `Transfer-Encoding` is present it overrides `Content-Length`
/// (RFC 9112 §6.3); callers implement that by checking this first.
///
/// Leniency (deliberate): transfer parameters after a `;` (e.g. the
/// `q=0.8` in `gzip; q=0.8`) are accepted and ignored — only the coding
/// token itself is validated. Real-world senders attach malformed or
/// unknown parameters often enough that rejecting them breaks real
/// traffic while buying little security.
///
/// A `Transfer-Encoding` field on an HTTP/1.0 (or earlier) message is faulty
/// framing (RFC 9112 §6.1).
fn transfer_encoding(
    version: HttpVersion,
    headers: &Headers,
    direction: MessageDirection,
) -> Result<TransferOutcome, HttpError> {
    if version == HttpVersion::HTTP_10 && headers.get_all("transfer-encoding").next().is_some() {
        return Err(HttpError::TransferEncodingOnHttp10);
    }
    let mut codings: Vec<&[u8]> = Vec::new();
    for value in headers.get_all("transfer-encoding") {
        for part in value.split(|&b| b == b',') {
            // A coding may carry parameters (`gzip; q=0.8`); only the token matters.
            let coding = trim_ows(part.split(|&b| b == b';').next().unwrap_or(&[]));
            if coding.is_empty() || !is_token(coding) {
                return Err(HttpError::InvalidTransferEncoding);
            }
            // `identity` is not a transfer coding (RFC 9112 §6.1): it is
            // only meaningful in Content-Encoding/TE negotiation.
            // Filtering it as a no-op would let `Transfer-Encoding:
            // identity` fall through to Content-Length framing, so it is
            // rejected outright instead.
            if coding.eq_ignore_ascii_case(b"identity") {
                return Err(HttpError::IdentityTransferEncoding);
            }
            codings.push(coding);
        }
    }
    if codings.is_empty() {
        return Ok(TransferOutcome::None);
    }
    // Framing validity: chunked must be the final coding (RFC 9112 §6.3).
    let chunked_count = codings
        .iter()
        .filter(|c| c.eq_ignore_ascii_case(b"chunked"))
        .count();
    let chunked_is_final = codings
        .last()
        .is_some_and(|c| c.eq_ignore_ascii_case(b"chunked"));
    if !chunked_is_final {
        // Chunked is not the final coding, or is absent entirely. Requests
        // cannot be reliably framed: `TE: chunked, <other>` is
        // ChunkedNotFinal, and any other unsupported coding is
        // UnsupportedTransferEncoding (rule 2). Responses are read until
        // the connection closes whenever Transfer-Encoding is present
        // without chunked as the final coding (rule 4).
        return match direction {
            MessageDirection::Request => {
                if chunked_count > 0 {
                    Err(HttpError::ChunkedNotFinal)
                } else {
                    Err(HttpError::UnsupportedTransferEncoding)
                }
            }
            MessageDirection::Response => Ok(TransferOutcome::UntilClose),
        };
    }
    if chunked_count > 1 {
        return Err(HttpError::InvalidTransferEncoding);
    }
    Ok(TransferOutcome::Chunked)
}

/// How a request's body is framed. With neither `Transfer-Encoding` nor
/// `Content-Length` the body length is zero (RFC 9112 §6.3).
/// A `Content-Length` above [`Limits::max_body_bytes`] is rejected.
pub fn request_body_spec(head: &RequestHead, limits: &Limits) -> Result<BodySpec, HttpError> {
    match transfer_encoding(head.version, &head.headers, MessageDirection::Request)? {
        TransferOutcome::Chunked => Ok(BodySpec::Chunked),
        TransferOutcome::UntilClose => Ok(BodySpec::UntilClose),
        TransferOutcome::None => Ok(match content_length(&head.headers, limits)? {
            Some(n) => BodySpec::ContentLength(n),
            None => BodySpec::Empty,
        }),
    }
}

/// How a response's body is framed (RFC 9112 §6.3, RFC 9110 §8.6, §15).
///
/// `request_method` is the method of the request this responds to: responses
/// to `HEAD`, and responses with status 1xx/204/304, never carry a body.
/// A 2xx response to `CONNECT` turns the connection into a tunnel: the client
/// must ignore `Content-Length`/`Transfer-Encoding` (`BodySpec::Tunnel`).
/// A `Content-Length` above [`Limits::max_body_bytes`] is rejected.
pub fn response_body_spec(
    head: &ResponseHead,
    request_method: &[u8],
    limits: &Limits,
) -> Result<BodySpec, HttpError> {
    // 1xx / 204 / 304 responses, and responses to HEAD, never carry a body,
    // but their framing headers must still be well-formed: garbage
    // Transfer-Encoding / Content-Length is rejected rather than silently
    // swallowed.
    if request_method == b"HEAD" || matches!(head.status, 100..=199 | 204 | 304) {
        transfer_encoding(head.version, &head.headers, MessageDirection::Response)?;
        content_length(&head.headers, limits)?;
        return Ok(BodySpec::Empty);
    }
    if request_method == b"CONNECT" && (200..=299).contains(&head.status) {
        return Ok(BodySpec::Tunnel);
    }
    match transfer_encoding(head.version, &head.headers, MessageDirection::Response)? {
        TransferOutcome::Chunked => Ok(BodySpec::Chunked),
        TransferOutcome::UntilClose => Ok(BodySpec::UntilClose),
        TransferOutcome::None => Ok(match content_length(&head.headers, limits)? {
            Some(n) => BodySpec::ContentLength(n),
            None => BodySpec::UntilClose,
        }),
    }
}

// ---------------------------------------------------------------------------
// Connection persistence
// ---------------------------------------------------------------------------

/// The tokens of every `Connection` header field, in order, OWS-trimmed.
///
/// Non-UTF-8 tokens are skipped rather than failing the whole parse: a
/// garbage byte sequence can never equal an ASCII token such as `close`
/// or `keep-alive`, so dropping it cannot change the persistence decision —
/// it only keeps one malformed field from tearing down the connection.
pub fn connection_tokens<'a>(headers: &'a Headers<'a>) -> impl Iterator<Item = &'a str> + 'a {
    headers
        .iter()
        .filter(|h| h.name.eq_ignore_ascii_case("connection"))
        .flat_map(|h| h.value.split(|&b| b == b','))
        .map(trim_ows)
        .filter(|t| !t.is_empty())
        .filter_map(|t| str::from_utf8(t).ok())
        .filter(|t| !t.is_empty())
}

/// Whether the connection stays open after this message (RFC 9112 §9.3).
///
/// HTTP/1.1 defaults to persistent unless `Connection: close`; HTTP/1.0
/// defaults to close unless `Connection: keep-alive`. `close` always wins.
pub fn is_persistent(version: HttpVersion, headers: &Headers) -> bool {
    let mut close = false;
    let mut keep_alive = false;
    for token in connection_tokens(headers) {
        if token.eq_ignore_ascii_case("close") {
            close = true;
        } else if token.eq_ignore_ascii_case("keep-alive") {
            keep_alive = true;
        }
    }
    if close {
        return false;
    }
    match (version.major, version.minor) {
        (1, 1) => true,
        (1, 0) => keep_alive,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Chunked transfer-coding
// ---------------------------------------------------------------------------

/// Trailer fields that must never appear in a trailer section
/// (RFC 9112 §7.1.2).
///
/// Trailers arrive after the body, so a sender MUST NOT generate a trailer
/// section containing fields that are needed *before* the content is fully
/// received. The RFC forbids fields for message framing, routing, request
/// modifiers, authentication, response control data, and payload processing;
/// the list below covers those categories explicitly. Comparison is
/// case-insensitive (field names are case-insensitive per RFC 9110 §5.1).
const FORBIDDEN_TRAILERS: &[&str] = &[
    // Framing: must be known before the body is read.
    "content-length",
    "transfer-encoding",
    "trailer",
    "te",
    // Routing.
    "host",
    // Request modifiers: controls and conditionals (affect how the
    // request is dispatched or evaluated, before the body completes).
    "expect",
    "max-forwards",
    "range",
    "if-match",
    "if-none-match",
    "if-modified-since",
    "if-unmodified-since",
    "if-range",
    // Authentication (RFC 9112 §7.1.2 cites RFC 7616 and RFC 8174).
    "authorization",
    "proxy-authenticate",
    "proxy-authorization",
    "www-authenticate",
    // Response control data (RFC 9110 §7.1): caching and freshness
    // metadata that a recipient must see before consuming the content.
    "age",
    "cache-control",
    "expires",
    "date",
    "location",
    "retry-after",
    "vary",
    "warning",
    // Payload processing: how to interpret the content itself.
    "content-encoding",
    "content-type",
    "content-range",
    "content-language",
    "content-location",
    "digest",
];

fn is_forbidden_trailer(name: &str) -> bool {
    FORBIDDEN_TRAILERS
        .iter()
        .any(|f| name.eq_ignore_ascii_case(f))
}

/// Encode a body with chunked transfer-coding, optionally with trailers.
///
/// Empty chunks are skipped (a zero-size chunk terminates the body).
pub fn encode_chunked(chunks: &[&[u8]], trailers: &[(&str, &[u8])]) -> Result<Vec<u8>, HttpError> {
    for (name, value) in trailers {
        validate_field_name(name.as_bytes())?;
        validate_field_value(value)?;
        if is_forbidden_trailer(name) {
            return Err(HttpError::ForbiddenTrailer);
        }
    }
    let mut out = Vec::new();
    for chunk in chunks {
        if chunk.is_empty() {
            continue;
        }
        push_hex(&mut out, chunk.len() as u64);
        out.extend_from_slice(b"\r\n");
        out.extend_from_slice(chunk);
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(b"0\r\n");
    for (name, value) in trailers {
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(b": ");
        out.extend_from_slice(value);
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(b"\r\n");
    Ok(out)
}

fn push_hex(out: &mut Vec<u8>, mut n: u64) {
    let mut buf = [0u8; 16];
    let mut i = buf.len();
    loop {
        i -= 1;
        let digit = (n & 0xf) as u8;
        buf[i] = if digit < 10 {
            b'0' + digit
        } else {
            b'A' + digit - 10
        };
        n >>= 4;
        if n == 0 {
            break;
        }
    }
    out.extend_from_slice(&buf[i..]);
}

/// Largest single chunk the chunked decoders accept. Chunk sizes parse into
/// a u64 (up to 16 hex digits), but chunk payloads are sliced with `usize`
/// arithmetic in [`decode_chunked`], so an uncapped size like
/// `FFFFFFFFFFFFFFFF` would overflow. 16 MiB is far above any sane single
/// chunk and fits comfortably in `usize` on 32- and 64-bit targets.
pub const MAX_CHUNK_SIZE: u64 = 16 * 1024 * 1024;

/// Parse a chunk-size line: `1*HEXDIG [ chunk-ext ]`.
/// Returns the size and whether this is the terminating zero-size chunk.
fn parse_chunk_size(line: &[u8]) -> Result<(u64, bool), HttpError> {
    let semi = line.iter().position(|&b| b == b';').unwrap_or(line.len());
    let digits = &line[..semi];
    if digits.is_empty() || digits.len() > 16 || !digits.iter().all(|b| b.is_ascii_hexdigit()) {
        return Err(HttpError::InvalidChunkSize);
    }
    let mut size: u64 = 0;
    for &b in digits {
        let d = match b {
            b'0'..=b'9' => u64::from(b - b'0'),
            b'a'..=b'f' => u64::from(b - b'a') + 10,
            b'A'..=b'F' => u64::from(b - b'A') + 10,
            // Checked by `is_ascii_hexdigit` above.
            _ => return Err(HttpError::InvalidChunkSize),
        };
        size = size * 16 + d;
    }
    if size > MAX_CHUNK_SIZE {
        return Err(HttpError::ChunkTooLarge);
    }
    validate_chunk_ext(&line[semi..])?;
    Ok((size, size == 0))
}

/// Validate `chunk-ext`: `*( ";" chunk-ext-name [ "=" chunk-ext-val ] )`.
fn validate_chunk_ext(mut ext: &[u8]) -> Result<(), HttpError> {
    while !ext.is_empty() {
        if ext[0] != b';' {
            return Err(HttpError::InvalidChunkExtension);
        }
        ext = &ext[1..];
        let name_end = ext
            .iter()
            .position(|&b| b == b'=' || b == b';')
            .unwrap_or(ext.len());
        if !is_token(&ext[..name_end]) {
            return Err(HttpError::InvalidChunkExtension);
        }
        ext = &ext[name_end..];
        if ext.first() == Some(&b'=') {
            ext = &ext[1..];
            if ext.first() == Some(&b'"') {
                ext = skip_quoted_string(ext)?;
            } else {
                let val_end = ext.iter().position(|&b| b == b';').unwrap_or(ext.len());
                if !is_token(&ext[..val_end]) {
                    return Err(HttpError::InvalidChunkExtension);
                }
                ext = &ext[val_end..];
            }
        }
    }
    Ok(())
}

/// Skip over a quoted-string; `s` must start with `"`. Returns the rest.
fn skip_quoted_string(s: &[u8]) -> Result<&[u8], HttpError> {
    let mut i = 1; // consume the opening quote
    while i < s.len() {
        match s[i] {
            b'"' => return Ok(&s[i + 1..]),
            b'\\' => {
                i += 1;
                if i >= s.len() {
                    return Err(HttpError::InvalidChunkExtension);
                }
                let c = s[i];
                let ok = c == b'\t' || c == b' ' || (0x21..=0x7e).contains(&c) || c >= 0x80;
                if !ok {
                    return Err(HttpError::InvalidChunkExtension);
                }
                i += 1;
            }
            b'\t' | b' ' | 0x21 | 0x23..=0x5b | 0x5d..=0x7e | 0x80..=0xff => i += 1,
            _ => return Err(HttpError::InvalidChunkExtension),
        }
    }
    Err(HttpError::InvalidChunkExtension)
}

/// Parse the trailer section after the zero-size chunk. Rejects forbidden
/// trailer fields. `headers_seen` is the number of header fields already
/// parsed in the message head; trailers share the remaining
/// [`Limits::max_headers`] budget with them.
fn parse_trailers<'a>(
    input: &'a [u8],
    limits: &Limits,
    headers_seen: usize,
) -> Result<(Headers<'a>, usize), HttpError> {
    let trailer_limits = Limits {
        max_headers: limits.max_headers.saturating_sub(headers_seen),
        ..*limits
    };
    let (headers, len) = parse_headers(input, &trailer_limits, 0)?;
    for h in headers.iter() {
        if is_forbidden_trailer(h.name) {
            return Err(HttpError::ForbiddenTrailer);
        }
    }
    Ok((headers, len))
}

/// A decoded chunked body. Chunk data borrows the input buffer; `rest` is
/// whatever followed the trailer section (e.g. the next pipelined message).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkedMessage<'a> {
    pub chunks: Vec<&'a [u8]>,
    pub trailers: Headers<'a>,
    pub rest: &'a [u8],
}

/// Decode a complete chunked body in one shot. Chunk payloads are slices of
/// `input` — no copying. `headers_seen` is the number of header fields
/// already parsed in the message head; trailer fields share the remaining
/// [`Limits::max_headers`] budget with them (headers + trailers). Returns
/// [`HttpError::Incomplete`] if the buffer does not yet hold the whole body,
/// and [`HttpError::BodyTooLarge`] if the cumulative chunk payload exceeds
/// [`Limits::max_body_bytes`].
///
/// Line endings: chunk-size lines (and trailers) go through [`next_line`]
/// and require strict CRLF; only the CRLF after each chunk's data keeps a
/// lenient path that also accepts a lone LF. A bare CR is rejected.
pub fn decode_chunked<'a>(
    input: &'a [u8],
    limits: &Limits,
    headers_seen: usize,
) -> Result<ChunkedMessage<'a>, HttpError> {
    let mut chunks = Vec::new();
    let mut pos = 0usize;
    let mut total: u64 = 0;
    loop {
        let line = next_line(&input[pos..], limits.max_line_bytes)?;
        pos += line.consumed;
        let (size, is_last) = parse_chunk_size(line.bytes)?;
        if is_last {
            let (trailers, tlen) = parse_trailers(&input[pos..], limits, headers_seen)?;
            pos += tlen;
            return Ok(ChunkedMessage {
                chunks,
                trailers,
                rest: &input[pos..],
            });
        }
        // Cumulative payload cap: check before borrowing the slice.
        total = total.saturating_add(size);
        if total > limits.max_body_bytes as u64 {
            return Err(HttpError::BodyTooLarge);
        }
        // Checked conversion: `parse_chunk_size` already caps at
        // MAX_CHUNK_SIZE, but the explicit check keeps this sound even if
        // the cap ever changes.
        let size = usize::try_from(size).map_err(|_| HttpError::ChunkTooLarge)?;
        if input.len() - pos < size {
            return Err(HttpError::Incomplete);
        }
        chunks.push(&input[pos..pos + size]);
        let term = &input[pos + size..];
        // The terminator accepts CRLF or a lone LF. A lone trailing CR may be
        // the first half of a split CRLF, so that stays Incomplete.
        let term_len = if term.starts_with(b"\r\n") {
            2
        } else if term.starts_with(b"\n") {
            1
        } else if term.is_empty() || term == b"\r" {
            // No terminator bytes yet, or a lone CR that may be the first
            // half of a split CRLF.
            return Err(HttpError::Incomplete);
        } else {
            return Err(HttpError::InvalidChunkTerminator);
        };
        pos += size + term_len;
    }
}

// ---------------------------------------------------------------------------
// Streaming chunked decoder
// ---------------------------------------------------------------------------

/// One parsed trailer field from the streaming decoder (owned: trailer
/// lines may span `feed` calls).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrailerField {
    pub name: String,
    pub value: Vec<u8>,
}

/// Events produced by [`ChunkedDecoder::feed`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkEvent<'a> {
    /// Chunk payload bytes, borrowed from the fed input (zero-copy).
    Data(&'a [u8]),
    /// One parsed trailer field.
    Trailer(TrailerField),
    /// The terminating zero-size chunk and trailer section were consumed.
    End,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecoderState {
    ChunkSize,
    ChunkData { remaining: u64 },
    ChunkDataCr,
    ChunkDataLf,
    TrailerLine,
    Done,
}

/// Incremental chunked-body decoder.
///
/// Feed it arbitrary slices, passing a reusable events buffer; it pushes
/// [`ChunkEvent::Data`] slices that borrow the fed input, so the data path
/// never copies or allocates a fresh event list. Only trailer
/// fields are copied into owned values.
///
/// Trailer fields share the [`Limits::max_headers`] budget with the header
/// fields already parsed in the message head; tell the decoder about them
/// with [`ChunkedDecoder::new_with_headers_seen`].
///
/// Poison-on-error contract: after [`feed`](Self::feed) returns an
/// error the decoder is in an indeterminate state and must not be fed
/// further input. Call [`reset`](Self::reset) to restore the initial
/// state before decoding another message; dropping the decoder and
/// creating a new one works too.
///
/// Payload bytes are counted cumulatively across feeds: once they exceed
/// [`Limits::max_body_bytes`], [`feed`](Self::feed) fails with
/// [`HttpError::BodyTooLarge`].
pub struct ChunkedDecoder {
    state: DecoderState,
    line: Vec<u8>,
    limits: Limits,
    headers_seen: usize,
    trailer_count: usize,
    body_bytes: u64,
}

impl ChunkedDecoder {
    pub fn new(limits: Limits) -> Self {
        Self::new_with_headers_seen(limits, 0)
    }

    /// Create a decoder for a body whose message head already yielded
    /// `headers_seen` header fields. Trailer fields share the remaining
    /// [`Limits::max_headers`] budget with them (headers + trailers).
    pub fn new_with_headers_seen(limits: Limits, headers_seen: usize) -> Self {
        Self {
            state: DecoderState::ChunkSize,
            line: Vec::new(),
            limits,
            headers_seen,
            trailer_count: 0,
            body_bytes: 0,
        }
    }

    /// True once the terminating chunk and trailers were fully consumed.
    pub fn is_done(&self) -> bool {
        self.state == DecoderState::Done
    }

    /// Restore the decoder to its initial state, as after
    /// [`new`](Self::new). Buffered partial line data and any decoded
    /// state are discarded; the configured [`Limits`] are kept.
    ///
    /// This is the way to recover after [`feed`](Self::feed) returned an
    /// error: an errored decoder is poisoned, and this resets it so it can
    /// decode a fresh message.
    pub fn reset(&mut self) {
        self.state = DecoderState::ChunkSize;
        self.line.clear();
        self.trailer_count = 0;
        self.body_bytes = 0;
    }

    /// Feed input bytes, pushing the completed events into the caller-provided
    /// `events` buffer. Reusing the buffer across calls avoids allocating a
    /// `Vec` per call. Returns the number of input bytes consumed. After
    /// [`is_done`](Self::is_done) further input is ignored, reported as zero
    /// consumed bytes. If an error is returned the decoder is poisoned (see
    /// the struct docs) — call [`reset`](Self::reset) before feeding more.
    ///
    /// Pipelined messages: when [`ChunkEvent::End`] is emitted, any bytes
    /// after the terminator belong to the next message on the connection.
    /// They are deliberately not consumed: `&input[consumed..]` is the
    /// start of the next message, and the caller must retain it.
    pub fn feed<'a>(
        &mut self,
        mut input: &'a [u8],
        events: &mut Vec<ChunkEvent<'a>>,
    ) -> Result<usize, HttpError> {
        let start_len = input.len();
        while !input.is_empty() {
            match self.state {
                DecoderState::Done => break,
                DecoderState::ChunkData { remaining } => {
                    let take = (remaining.min(input.len() as u64)) as usize;
                    let (data, rest) = input.split_at(take);
                    if !data.is_empty() {
                        self.body_bytes = self.body_bytes.saturating_add(take as u64);
                        if self.body_bytes > self.limits.max_body_bytes as u64 {
                            return Err(HttpError::BodyTooLarge);
                        }
                        events.push(ChunkEvent::Data(data));
                    }
                    input = rest;
                    let remaining = remaining - take as u64;
                    self.state = if remaining == 0 {
                        DecoderState::ChunkDataCr
                    } else {
                        DecoderState::ChunkData { remaining }
                    };
                }
                DecoderState::ChunkDataCr => {
                    // The chunk-data terminator has its own lenient path
                    // (not `next_line`): CRLF, or a lone LF. A CR followed
                    // by anything else is still an error.
                    if input[0] == b'\n' {
                        input = &input[1..];
                        self.state = DecoderState::ChunkSize;
                    } else if input[0] == b'\r' {
                        input = &input[1..];
                        self.state = DecoderState::ChunkDataLf;
                    } else {
                        return Err(HttpError::InvalidChunkTerminator);
                    }
                }
                DecoderState::ChunkDataLf => {
                    if input[0] != b'\n' {
                        return Err(HttpError::BareCr);
                    }
                    input = &input[1..];
                    self.state = DecoderState::ChunkSize;
                }
                DecoderState::ChunkSize | DecoderState::TrailerLine => {
                    match input.iter().position(|&b| b == b'\n') {
                        Some(lf) => {
                            // Strict line endings, matching `next_line`
                            // (RFC 9112 §2.2): chunk-size and trailer lines
                            // must end in CRLF; a bare LF is rejected with
                            // `HttpError::BareLf`. Checked before the length
                            // cap, as `next_line` does. The chunk-*data*
                            // terminator keeps its own lenient path
                            // (`DecoderState::ChunkDataCr`) and is unaffected.
                            let has_cr = if lf > 0 {
                                input[lf - 1] == b'\r'
                            } else {
                                // The LF is the first byte of this feed, so
                                // the line is fully buffered; a trailing CR
                                // there is a split-CRLF candidate (interior
                                // CRs were already rejected when buffered).
                                self.line.last() == Some(&b'\r')
                            };
                            if !has_cr {
                                return Err(HttpError::BareLf);
                            }
                            if self.line.len() + lf > self.limits.max_line_bytes {
                                return Err(HttpError::LineTooLong);
                            }
                            self.line.extend_from_slice(&input[..lf]);
                            input = &input[lf + 1..];
                            let is_trailer = self.state == DecoderState::TrailerLine;
                            self.handle_line(events, is_trailer)?;
                        }
                        None => {
                            // Length first, matching `next_line`: an over-limit line
                            // reports LineTooLong even if it also holds a bare CR.
                            if self.line.len() + input.len() > self.limits.max_line_bytes {
                                return Err(HttpError::LineTooLong);
                            }
                            for (i, &b) in input.iter().enumerate() {
                                if b == b'\r' && i + 1 < input.len() {
                                    return Err(HttpError::BareCr);
                                }
                            }
                            self.line.extend_from_slice(input);
                            input = &[];
                        }
                    }
                }
            }
        }
        Ok(start_len - input.len())
    }

    fn handle_line(
        &mut self,
        events: &mut Vec<ChunkEvent<'_>>,
        is_trailer: bool,
    ) -> Result<(), HttpError> {
        let mut line = std::mem::take(&mut self.line);
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        if line.contains(&b'\r') {
            return Err(HttpError::BareCr);
        }
        if line.len() > self.limits.max_line_bytes {
            return Err(HttpError::LineTooLong);
        }
        if is_trailer {
            if line.is_empty() {
                self.state = DecoderState::Done;
                events.push(ChunkEvent::End);
                return Ok(());
            }
            if self.headers_seen.saturating_add(self.trailer_count) >= self.limits.max_headers {
                return Err(HttpError::TooManyHeaders);
            }
            let field = parse_header_line(&line)?;
            if is_forbidden_trailer(field.name) {
                return Err(HttpError::ForbiddenTrailer);
            }
            self.trailer_count += 1;
            events.push(ChunkEvent::Trailer(TrailerField {
                name: field.name.to_owned(),
                value: field.value.to_vec(),
            }));
        } else {
            let (size, is_last) = parse_chunk_size(&line)?;
            self.state = if is_last {
                DecoderState::TrailerLine
            } else {
                DecoderState::ChunkData { remaining: size }
            };
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

fn push_decimal_u8(out: &mut Vec<u8>, mut n: u8) {
    let mut buf = [0u8; 3];
    let mut i = buf.len();
    loop {
        i -= 1;
        buf[i] = b'0' + n % 10;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    out.extend_from_slice(&buf[i..]);
}

fn push_version(out: &mut Vec<u8>, version: HttpVersion) {
    out.extend_from_slice(b"HTTP/");
    push_decimal_u8(out, version.major);
    out.push(b'.');
    push_decimal_u8(out, version.minor);
}

fn serialize_headers(out: &mut Vec<u8>, headers: &[(&str, &[u8])]) -> Result<(), HttpError> {
    for (name, value) in headers {
        validate_field_name(name.as_bytes())?;
        validate_field_value(value)?;
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(b": ");
        out.extend_from_slice(value);
        out.extend_from_slice(b"\r\n");
    }
    Ok(())
}

/// Serialize a request head. Names and values are validated; a bad one is
/// an error, never silently emitted.
pub fn serialize_request_head(
    method: &str,
    target: &str,
    version: HttpVersion,
    headers: &[(&str, &[u8])],
) -> Result<Vec<u8>, HttpError> {
    if !is_token(method.as_bytes()) {
        return Err(HttpError::InvalidMethod);
    }
    if !valid_target(target.as_bytes()) {
        return Err(HttpError::InvalidTarget);
    }
    let mut out = Vec::new();
    out.extend_from_slice(method.as_bytes());
    out.push(b' ');
    out.extend_from_slice(target.as_bytes());
    out.push(b' ');
    push_version(&mut out, version);
    out.extend_from_slice(b"\r\n");
    serialize_headers(&mut out, headers)?;
    out.extend_from_slice(b"\r\n");
    Ok(out)
}

/// Serialize a response head. The status must be a 3-digit code.
pub fn serialize_response_head(
    version: HttpVersion,
    status: u16,
    reason: &str,
    headers: &[(&str, &[u8])],
) -> Result<Vec<u8>, HttpError> {
    if !(100..=999).contains(&status) {
        return Err(HttpError::InvalidStatus);
    }
    if !valid_reason(reason.as_bytes()) {
        return Err(HttpError::InvalidStatusLine);
    }
    let mut out = Vec::new();
    push_version(&mut out, version);
    out.push(b' ');
    out.extend_from_slice(&[
        b'0' + (status / 100) as u8,
        b'0' + ((status / 10) % 10) as u8,
        b'0' + (status % 10) as u8,
    ]);
    out.push(b' ');
    out.extend_from_slice(reason.as_bytes());
    out.extend_from_slice(b"\r\n");
    serialize_headers(&mut out, headers)?;
    out.extend_from_slice(b"\r\n");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> Limits {
        Limits::default()
    }

    fn req(input: &[u8]) -> Result<(RequestHead<'_>, usize), HttpError> {
        parse_request_head(input, &limits())
    }

    fn res(input: &[u8]) -> Result<(ResponseHead<'_>, usize), HttpError> {
        parse_response_head(input, &limits())
    }

    // -- request line -------------------------------------------------------

    #[test]
    fn parse_simple_get() {
        let input = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let (head, consumed) = req(input).unwrap();
        assert_eq!(consumed, input.len());
        assert_eq!(head.method, "GET");
        assert_eq!(head.target, "/index.html");
        assert_eq!(head.version, HttpVersion::HTTP_11);
        assert_eq!(head.headers.get("host"), Some(b"example.com".as_slice()));
    }

    #[test]
    fn parse_request_http10_and_absolute_form() {
        let input = b"POST http://example.com/submit HTTP/1.0\r\n\r\n";
        let (head, consumed) = req(input).unwrap();
        assert_eq!(consumed, input.len());
        assert_eq!(head.method, "POST");
        assert_eq!(head.target, "http://example.com/submit");
        assert_eq!(head.version, HttpVersion::HTTP_10);
        assert!(head.headers.is_empty());
    }

    #[test]
    fn parse_request_options_asterisk() {
        let (head, _) = req(b"OPTIONS * HTTP/1.1\r\n\r\n").unwrap();
        assert_eq!(head.target, "*");
    }

    #[test]
    fn request_head_consumed_stops_at_blank_line() {
        let input = b"GET / HTTP/1.1\r\nHost: x\r\n\r\nBODYPAD";
        let (head, consumed) = req(input).unwrap();
        assert_eq!(consumed, input.len() - 7);
        assert_eq!(&input[consumed..], b"BODYPAD");
        assert_eq!(head.method, "GET");
    }

    #[test]
    fn request_line_rejects_malformed() {
        // Missing / extra parts.
        assert_eq!(
            req(b"GET /HTTP/1.1\r\n\r\n"),
            Err(HttpError::InvalidRequestLine)
        );
        assert_eq!(
            req(b"GET / HTTP/1.1 extra\r\n\r\n"),
            Err(HttpError::InvalidRequestLine)
        );
        assert_eq!(
            req(b"GET  / HTTP/1.1\r\n\r\n"),
            Err(HttpError::InvalidRequestLine)
        );
        // Empty method.
        assert_eq!(req(b" / HTTP/1.1\r\n\r\n"), Err(HttpError::InvalidMethod));
        // Empty target.
        assert_eq!(req(b"GET  HTTP/1.1\r\n\r\n"), Err(HttpError::InvalidTarget));
        // CTL in target.
        assert_eq!(
            req(b"GET /\x01 HTTP/1.1\r\n\r\n"),
            Err(HttpError::InvalidTarget)
        );
        // Non-token method.
        assert_eq!(
            req(b"G@T / HTTP/1.1\r\n\r\n"),
            Err(HttpError::InvalidMethod)
        );
    }

    #[test]
    fn request_version_rejects_malformed() {
        for bad in [
            "FOO".as_bytes(),
            b"HTTP/2".as_slice(),
            b"http/1.1",
            b"HTTP/1.10",
            b"HTTP/10.1",
            b"",
        ] {
            let mut input = b"GET / ".to_vec();
            input.extend_from_slice(bad);
            input.extend_from_slice(b"\r\n\r\n");
            assert_eq!(
                req(&input),
                Err(HttpError::InvalidVersion),
                "input: {bad:?}"
            );
        }
    }

    #[test]
    fn request_incomplete_inputs() {
        assert_eq!(req(b""), Err(HttpError::Incomplete));
        assert_eq!(req(b"GET"), Err(HttpError::Incomplete));
        assert_eq!(req(b"GET / HTTP/1.1\r\n"), Err(HttpError::Incomplete));
        assert_eq!(
            req(b"GET / HTTP/1.1\r\nHost: x\r\n"),
            Err(HttpError::Incomplete)
        );
        // Trailing CR may be a split CRLF: incomplete, not bare-CR.
        assert_eq!(req(b"GET / HTTP/1.1\r"), Err(HttpError::Incomplete));
        assert_eq!(res(b"HTTP/1.1 200 OK\r\n"), Err(HttpError::Incomplete));
    }

    #[test]
    fn lone_lf_is_rejected() {
        // RFC 9112 §2.2: bare LF is not a valid line terminator.
        assert_eq!(req(b"GET / HTTP/1.1\nHost: x\n\n"), Err(HttpError::BareLf));
        assert_eq!(
            res(b"HTTP/1.1 200 OK\nContent-Length: 0\n\n"),
            Err(HttpError::BareLf)
        );
        // A lone LF after valid CRLF lines is still rejected.
        assert_eq!(
            req(b"GET / HTTP/1.1\r\nHost: x\n\r\n"),
            Err(HttpError::BareLf)
        );
    }

    #[test]
    fn bare_cr_is_rejected() {
        assert_eq!(req(b"GET / HTTP/1.1\rX\r\n\r\n"), Err(HttpError::BareCr));
        assert_eq!(
            req(b"GET / HTTP/1.1\r\nHost: a\rb\r\n\r\n"),
            Err(HttpError::BareCr)
        );
        assert_eq!(res(b"HTTP/1.1 200\rOK\r\n\r\n"), Err(HttpError::BareCr));
    }

    #[test]
    fn limits_are_enforced() {
        let tiny_line = Limits {
            max_line_bytes: 10,
            ..Default::default()
        };
        assert_eq!(
            parse_request_head(b"GET / HTTP/1.1\r\n\r\n", &tiny_line),
            Err(HttpError::LineTooLong)
        );

        let tiny_head = Limits {
            max_head_bytes: 30,
            ..Default::default()
        };
        assert_eq!(
            parse_request_head(b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n", &tiny_head),
            Err(HttpError::HeadersTooLarge)
        );

        let few_headers = Limits {
            max_headers: 2,
            ..Default::default()
        };
        assert_eq!(
            parse_request_head(
                b"GET / HTTP/1.1\r\nA: 1\r\nB: 2\r\nC: 3\r\n\r\n",
                &few_headers
            ),
            Err(HttpError::TooManyHeaders)
        );
        // Exactly at the limit is fine.
        assert!(
            parse_request_head(b"GET / HTTP/1.1\r\nA: 1\r\nB: 2\r\n\r\n", &few_headers).is_ok()
        );
    }

    // -- header fields ------------------------------------------------------

    #[test]
    fn header_ows_is_trimmed_and_empty_value_allowed() {
        let (head, _) =
            req(b"GET / HTTP/1.1\r\nHost: \t example.com \t\r\nX-Empty:\r\n\r\n").unwrap();
        assert_eq!(head.headers.get("host"), Some(b"example.com".as_slice()));
        assert_eq!(head.headers.get("x-empty"), Some(b"".as_slice()));
    }

    // Regression tests: `trim_ows` used to panic on all-whitespace input
    // (`&bytes[len..0]`). Every shape below is reachable from the network
    // and must parse (or fail cleanly) instead of panicking.

    #[test]
    fn trim_ows_all_whitespace_is_empty() {
        assert_eq!(trim_ows(b" "), b"");
        assert_eq!(trim_ows(b"   "), b"");
        assert_eq!(trim_ows(b"\t \t"), b"");
        assert_eq!(trim_ows(b""), b"");
        assert_eq!(trim_ows(b"  x  "), b"x");
        assert_eq!(trim_ows(b"x"), b"x");
    }

    #[test]
    fn header_single_space_value_parses_as_empty() {
        // `X: \r\n`: OWS-only value is the empty value (RFC 9110 §5.5).
        let (head, _) = req(b"GET / HTTP/1.1\r\nX: \r\n\r\n").unwrap();
        assert_eq!(head.headers.get("x"), Some(b"".as_slice()));
    }

    #[test]
    fn content_length_ows_only_value_does_not_panic() {
        let (head, _) = req(b"POST / HTTP/1.1\r\nContent-Length:    \r\n\r\n").unwrap();
        assert_eq!(head.headers.get("content-length"), Some(b"".as_slice()));
        // Empty is not a valid decimal; that is an error, not a panic.
        assert_eq!(
            content_length(&head.headers, &limits()),
            Err(HttpError::InvalidContentLength)
        );
    }

    #[test]
    fn content_length_ows_only_list_element_does_not_panic() {
        // `Content-Length: 5,   ,6`: the empty element rejects cleanly.
        let (head, _) = req(b"POST / HTTP/1.1\r\nContent-Length: 5,   ,6\r\n\r\n").unwrap();
        assert_eq!(
            content_length(&head.headers, &limits()),
            Err(HttpError::InvalidContentLength)
        );
    }

    #[test]
    fn transfer_encoding_ows_only_value_rejected_without_panic() {
        // An empty coding is invalid; that is an error, not a panic.
        let (head, _) = req(b"POST / HTTP/1.1\r\nTransfer-Encoding:   \r\n\r\n").unwrap();
        assert_eq!(head.headers.get("transfer-encoding"), Some(b"".as_slice()));
        assert_eq!(
            request_body_spec(&head, &limits()),
            Err(HttpError::InvalidTransferEncoding)
        );
    }

    #[test]
    fn transfer_encoding_ows_only_list_element_rejected_without_panic() {
        // `Transfer-Encoding: chunked,   ,identity`: empty coding rejects.
        let (head, _) =
            req(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked,   ,identity\r\n\r\n").unwrap();
        assert_eq!(
            request_body_spec(&head, &limits()),
            Err(HttpError::InvalidTransferEncoding)
        );
    }

    #[test]
    fn te_header_with_ows_only_element_parses() {
        // `TE: chunked,   ,identity`: header value survives intact, no panic.
        let (head, _) = req(b"GET / HTTP/1.1\r\nTE: chunked,   ,identity\r\n\r\n").unwrap();
        assert_eq!(
            head.headers.get("te"),
            Some(b"chunked,   ,identity".as_slice())
        );
    }

    #[test]
    fn decode_chunked_ows_only_trailer_value_parses() {
        // `0\r\nX:  \r\n\r\n`: trailer with OWS-only value is empty.
        let msg = decode_chunked(b"0\r\nX:  \r\n\r\n", &limits(), 0).unwrap();
        assert_eq!(msg.trailers.len(), 1);
        assert_eq!(msg.trailers.get("x"), Some(b"".as_slice()));
    }

    #[test]
    fn header_value_allows_htab_and_obs_text() {
        let (head, _) = req("GET / HTTP/1.1\r\nX: a\tb caf\u{e9}\r\n\r\n".as_bytes()).unwrap();
        assert_eq!(head.headers.get("x"), Some("a\tb caf\u{e9}".as_bytes()));
    }

    #[test]
    fn header_malformed_lines_rejected() {
        // No colon.
        assert_eq!(
            req(b"GET / HTTP/1.1\r\nNoColonHere\r\n\r\n"),
            Err(HttpError::MissingHeaderColon)
        );
        // Whitespace before the colon.
        assert_eq!(
            req(b"GET / HTTP/1.1\r\nHost : x\r\n\r\n"),
            Err(HttpError::InvalidHeaderName)
        );
        assert_eq!(
            req(b"GET / HTTP/1.1\r\nHost\t: x\r\n\r\n"),
            Err(HttpError::InvalidHeaderName)
        );
        // Empty name.
        assert_eq!(
            req(b"GET / HTTP/1.1\r\n: x\r\n\r\n"),
            Err(HttpError::InvalidHeaderName)
        );
        // Non-token characters in the name.
        assert_eq!(
            req(b"GET / HTTP/1.1\r\nHo st: x\r\n\r\n"),
            Err(HttpError::InvalidHeaderName)
        );
        assert_eq!(
            req(b"GET / HTTP/1.1\r\nHo@st: x\r\n\r\n"),
            Err(HttpError::InvalidHeaderName)
        );
        let non_ascii = "GET / HTTP/1.1\r\nH\u{f6}st: x\r\n\r\n";
        assert_eq!(req(non_ascii.as_bytes()), Err(HttpError::InvalidHeaderName));
        // CTL / DEL in the value.
        assert_eq!(
            req(b"GET / HTTP/1.1\r\nX: a\x01b\r\n\r\n"),
            Err(HttpError::InvalidHeaderValue)
        );
        assert_eq!(
            req(b"GET / HTTP/1.1\r\nX: a\x7fb\r\n\r\n"),
            Err(HttpError::InvalidHeaderValue)
        );
    }

    #[test]
    fn obs_fold_is_rejected() {
        // Classic obs-fold: continuation line.
        assert_eq!(
            req(b"GET / HTTP/1.1\r\nHost: x\r\n continued\r\n\r\n"),
            Err(HttpError::ObsFoldRejected)
        );
        // A whitespace-only line is also obs-fold, not a blank line.
        assert_eq!(
            req(b"GET / HTTP/1.1\r\nHost: x\r\n \r\n\r\n"),
            Err(HttpError::ObsFoldRejected)
        );
        assert_eq!(
            req(b"GET / HTTP/1.1\r\nHost: x\r\n\t\r\n\r\n"),
            Err(HttpError::ObsFoldRejected)
        );
    }

    #[test]
    fn headers_case_insensitive_and_duplicates_kept() {
        let (head, _) = req(b"GET / HTTP/1.1\r\nX-Dup: 1\r\nx-dup: 2\r\nHOST: y\r\n\r\n").unwrap();
        // First match wins for `get`.
        assert_eq!(head.headers.get("X-DUP"), Some(b"1".as_slice()));
        assert_eq!(
            head.headers.get_all("x-dup").collect::<Vec<_>>(),
            vec![b"1".as_slice(), b"2".as_slice()]
        );
        assert_eq!(head.headers.len(), 3);
    }

    /// Compile-test: the `get_all` iterator must not borrow `name`.
    /// `name` is dropped at the end of the inner block; the iterator
    /// remains usable afterwards. This fails to compile with the old
    /// unified `<'s>` signature.
    #[test]
    fn get_all_iterator_outlives_name_borrow() {
        let (head, _) = req(b"GET / HTTP/1.1\r\nX-Dup: 1\r\nx-dup: 2\r\n\r\n").unwrap();
        let values: Vec<&[u8]> = {
            let name = String::from("X-DUP");
            let iter = head.headers.get_all(&name);
            drop(name);
            iter.collect()
        };
        assert_eq!(values, vec![b"1".as_slice(), b"2".as_slice()]);
    }

    #[test]
    fn headers_push_validates() {
        let mut h = Headers::new();
        assert_eq!(h.push("Bad Name", b"v"), Err(HttpError::InvalidHeaderName));
        assert_eq!(h.push("X", b"a\rb"), Err(HttpError::InvalidHeaderValue));
        h.push("X-Ok", b"v").unwrap();
        assert_eq!(h.get("x-ok"), Some(b"v".as_slice()));
    }

    // -- status line --------------------------------------------------------

    #[test]
    fn parse_simple_response() {
        let input = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
        let (head, consumed) = res(input).unwrap();
        assert_eq!(consumed, input.len());
        assert_eq!(head.version, HttpVersion::HTTP_11);
        assert_eq!(head.status, 200);
        assert_eq!(head.reason, "OK");
        assert_eq!(head.headers.get("content-length"), Some(b"0".as_slice()));
    }

    #[test]
    fn parse_response_empty_and_multiword_reason() {
        let (head, _) = res(b"HTTP/1.0 204\r\n\r\n").unwrap();
        assert_eq!(head.status, 204);
        assert_eq!(head.reason, "");
        let (head, _) = res(b"HTTP/1.1 404 Not Found\r\n\r\n").unwrap();
        assert_eq!(head.reason, "Not Found");
    }

    #[test]
    fn status_line_rejects_malformed() {
        assert_eq!(
            res(b"HTTP/1.1 20 OK\r\n\r\n"),
            Err(HttpError::InvalidStatus)
        );
        assert_eq!(
            res(b"HTTP/1.1 2X0 OK\r\n\r\n"),
            Err(HttpError::InvalidStatus)
        );
        assert_eq!(res(b"HTTP/1.1\r\n\r\n"), Err(HttpError::InvalidStatusLine));
        // Four digits: not a 3-digit code followed by SP.
        assert_eq!(
            res(b"HTTP/1.1 2000 OK\r\n\r\n"),
            Err(HttpError::InvalidStatusLine)
        );
        // CTL in the reason.
        assert_eq!(
            res(b"HTTP/1.1 200 O\x01K\r\n\r\n"),
            Err(HttpError::InvalidStatusLine)
        );
        // Bad version.
        assert_eq!(
            res(b"HTP/1.1 200 OK\r\n\r\n"),
            Err(HttpError::InvalidVersion)
        );
    }

    // -- Content-Length -----------------------------------------------------

    fn cl(raw: &[u8]) -> Result<Option<u64>, HttpError> {
        let input = [b"GET / HTTP/1.1\r\n", raw, b"\r\n\r\n"].concat();
        let (head, _) = req(&input).unwrap();
        content_length(&head.headers, &limits())
    }

    #[test]
    fn content_length_values() {
        assert_eq!(cl(b"Content-Length: 5"), Ok(Some(5)));
        assert_eq!(cl(b"Content-Length: 0"), Ok(Some(0)));
        assert_eq!(cl(b"Content-Length: 007"), Ok(Some(7)));
        // Exactly at the body-size cap passes...
        assert_eq!(
            cl(b"Content-Length: 268435456"),
            Ok(Some(256 * 1024 * 1024))
        );
        // ...but anything above it is rejected before a downstream reader
        // can size an allocation by it.
        assert_eq!(
            cl(b"Content-Length: 268435457"),
            Err(HttpError::BodyTooLarge)
        );
        assert_eq!(
            cl(b"Content-Length: 18446744073709551615"),
            Err(HttpError::BodyTooLarge)
        );
        // Equal values agree: comma list and separate fields.
        assert_eq!(cl(b"Content-Length: 5, 5"), Ok(Some(5)));
        assert_eq!(cl(b"Content-Length: 5\r\nContent-Length: 5"), Ok(Some(5)));
        // Absent entirely.
        assert_eq!(cl(b"Host: x"), Ok(None));
    }

    #[test]
    fn content_length_respects_tightened_limit() {
        // The fetch layer may tighten the cap per request/response.
        let tight = Limits {
            max_body_bytes: 100,
            ..Default::default()
        };
        let input = b"POST / HTTP/1.1\r\nContent-Length: 101\r\n\r\n";
        let (head, _) = parse_request_head(input, &tight).unwrap();
        assert_eq!(
            request_body_spec(&head, &tight),
            Err(HttpError::BodyTooLarge)
        );
        let input = b"POST / HTTP/1.1\r\nContent-Length: 100\r\n\r\n";
        let (head, _) = parse_request_head(input, &tight).unwrap();
        assert_eq!(
            request_body_spec(&head, &tight),
            Ok(BodySpec::ContentLength(100))
        );
    }

    #[test]
    fn content_length_conflicts_rejected() {
        assert_eq!(
            cl(b"Content-Length: 5\r\nContent-Length: 6"),
            Err(HttpError::ConflictingContentLength)
        );
        assert_eq!(
            cl(b"Content-Length: 5, 6"),
            Err(HttpError::ConflictingContentLength)
        );
    }

    #[test]
    fn content_length_invalid_rejected() {
        for bad in [
            "abc",
            "-5",
            "0x10",
            "5.0",
            "99999999999999999999999",
            "1_000",
        ] {
            let raw = format!("Content-Length: {bad}");
            assert_eq!(
                cl(raw.as_bytes()),
                Err(HttpError::InvalidContentLength),
                "input: {bad:?}"
            );
        }
        assert_eq!(cl(b"Content-Length:"), Err(HttpError::InvalidContentLength));
    }

    // -- Transfer-Encoding --------------------------------------------------

    fn te_spec(raw: &[u8]) -> Result<BodySpec, HttpError> {
        let input = [b"POST / HTTP/1.1\r\n", raw, b"\r\n\r\n"].concat();
        let (head, _) = req(&input).unwrap();
        request_body_spec(&head, &limits())
    }

    #[test]
    fn transfer_encoding_chunked_variants() {
        assert_eq!(
            te_spec(b"Transfer-Encoding: chunked"),
            Ok(BodySpec::Chunked)
        );
        assert_eq!(
            te_spec(b"Transfer-Encoding: Chunked"),
            Ok(BodySpec::Chunked)
        );
        assert_eq!(
            te_spec(b"Transfer-Encoding: \t chunked "),
            Ok(BodySpec::Chunked)
        );
        // Multiple TE fields combine.
        assert_eq!(
            te_spec(b"Transfer-Encoding: chunked\r\nTransfer-Encoding: chunked"),
            Err(HttpError::InvalidTransferEncoding)
        );
    }

    // Transfer parameters are accepted and ignored (deliberate leniency:
    // only the coding token is validated). Pinned so the behavior cannot
    // change silently.
    #[test]
    fn transfer_encoding_parameters_accepted_and_ignored() {
        assert_eq!(
            te_spec(b"Transfer-Encoding: chunked;"),
            Ok(BodySpec::Chunked)
        );
        assert_eq!(
            te_spec(b"Transfer-Encoding: chunked; garbage"),
            Ok(BodySpec::Chunked)
        );
        assert_eq!(
            te_spec(b"Transfer-Encoding: chunked; q=0.8; foo"),
            Ok(BodySpec::Chunked)
        );
        // Parameters do not make the token matching any less strict, but a
        // final `chunked` still frames: content codings are a decoding
        // concern above this layer (m1b / RFC 9112 §6.3 rule 2).
        assert_eq!(
            te_spec(b"Transfer-Encoding: gzip; q=0.8, chunked"),
            Ok(BodySpec::Chunked)
        );
    }

    #[test]
    fn transfer_encoding_identity_is_rejected() {
        // `identity` is not a transfer coding (RFC 9112 §6.1): it is
        // rejected outright, never filtered as a no-op.
        assert_eq!(
            te_spec(b"Transfer-Encoding: identity"),
            Err(HttpError::IdentityTransferEncoding)
        );
        assert_eq!(
            te_spec(b"Transfer-Encoding: identity, identity"),
            Err(HttpError::IdentityTransferEncoding)
        );
        assert_eq!(
            te_spec(b"Transfer-Encoding: chunked, identity"),
            Err(HttpError::IdentityTransferEncoding)
        );
        assert_eq!(
            te_spec(b"Transfer-Encoding: identity, chunked"),
            Err(HttpError::IdentityTransferEncoding)
        );
        // TE presence must never fall through to Content-Length.
        assert_eq!(
            te_spec(b"Transfer-Encoding: identity\r\nContent-Length: 5"),
            Err(HttpError::IdentityTransferEncoding)
        );
        // Responses reject it too (no silent until-close fallback).
        assert_eq!(
            res_spec(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: identity", b"GET"),
            Err(HttpError::IdentityTransferEncoding)
        );
    }

    #[test]
    fn transfer_encoding_bad_codings_rejected() {
        // Unknown coding alongside chunked: chunked is the final coding, so
        // framing is valid chunked; the content coding is a decoding concern
        // above this layer (RFC 9112 §6.3 rule 2 only rejects chunked when
        // it is not final).
        assert_eq!(
            te_spec(b"Transfer-Encoding: gzip, chunked"),
            Ok(BodySpec::Chunked)
        );
        // Chunked but not final.
        assert_eq!(
            te_spec(b"Transfer-Encoding: chunked, gzip"),
            Err(HttpError::ChunkedNotFinal)
        );
        // Chunked applied twice.
        assert_eq!(
            te_spec(b"Transfer-Encoding: chunked, chunked"),
            Err(HttpError::InvalidTransferEncoding)
        );
        // Empty list element.
        assert_eq!(
            te_spec(b"Transfer-Encoding: ,chunked"),
            Err(HttpError::InvalidTransferEncoding)
        );
        assert_eq!(
            te_spec(b"Transfer-Encoding:"),
            Err(HttpError::InvalidTransferEncoding)
        );
    }

    #[test]
    fn request_chunked_not_final_still_rejected() {
        // RFC 9112 §6.3 rule 2: a request with chunked not final is an error.
        assert_eq!(
            te_spec(b"Transfer-Encoding: chunked, gzip"),
            Err(HttpError::ChunkedNotFinal)
        );
        // A request with a non-chunked coding alone cannot be framed.
        assert_eq!(
            te_spec(b"Transfer-Encoding: gzip"),
            Err(HttpError::UnsupportedTransferEncoding)
        );
        // Plain `TE: chunked` framing is unchanged on requests.
        assert_eq!(
            te_spec(b"Transfer-Encoding: chunked"),
            Ok(BodySpec::Chunked)
        );
    }

    #[test]
    fn response_chunked_not_final_is_until_close() {
        // RFC 9112 §6.3 rule 4: a response with Transfer-Encoding present but
        // chunked not final is read until the connection closes.
        assert_eq!(
            res_spec(
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked, gzip",
                b"GET"
            ),
            Ok(BodySpec::UntilClose)
        );
        // Case-insensitivity of the codings.
        assert_eq!(
            res_spec(
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: Chunked, GZIP",
                b"GET"
            ),
            Ok(BodySpec::UntilClose)
        );
        // Transfer-Encoding still overrides Content-Length.
        assert_eq!(
            res_spec(
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked, gzip\r\nContent-Length: 5",
                b"GET"
            ),
            Ok(BodySpec::UntilClose)
        );
        // Plain `TE: chunked` framing is unchanged on responses.
        assert_eq!(
            res_spec(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked", b"GET"),
            Ok(BodySpec::Chunked)
        );
    }

    #[test]
    fn response_te_without_final_chunked_is_until_close() {
        // RFC 9112 §6.3 rule 4: a response with Transfer-Encoding present
        // but chunked not the final coding is read until the connection
        // closes, even when chunked is absent entirely.
        assert_eq!(
            res_spec(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip", b"GET"),
            Ok(BodySpec::UntilClose)
        );
        // Transfer-Encoding still overrides Content-Length.
        assert_eq!(
            res_spec(
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip\r\nContent-Length: 5",
                b"GET"
            ),
            Ok(BodySpec::UntilClose)
        );
        // Chunked as the final coding still frames as chunked.
        assert_eq!(
            res_spec(
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip, chunked",
                b"GET"
            ),
            Ok(BodySpec::Chunked)
        );
        // Plain `TE: chunked` framing is unchanged on responses.
        assert_eq!(
            res_spec(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked", b"GET"),
            Ok(BodySpec::Chunked)
        );
    }

    #[test]
    fn transfer_encoding_overrides_content_length() {
        let (head, _) =
            req(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\nContent-Length: 5\r\n\r\n")
                .unwrap();
        assert_eq!(request_body_spec(&head, &limits()), Ok(BodySpec::Chunked));
    }

    // -- body framing -------------------------------------------------------

    #[test]
    fn request_body_spec_cases() {
        let (head, _) = req(b"GET / HTTP/1.1\r\n\r\n").unwrap();
        assert_eq!(request_body_spec(&head, &limits()), Ok(BodySpec::Empty));
        let (head, _) = req(b"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\n").unwrap();
        assert_eq!(
            request_body_spec(&head, &limits()),
            Ok(BodySpec::ContentLength(5))
        );
        let (head, _) = req(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n").unwrap();
        assert_eq!(request_body_spec(&head, &limits()), Ok(BodySpec::Chunked));
    }

    fn res_spec(status_line: &[u8], method: &[u8]) -> Result<BodySpec, HttpError> {
        let input = [status_line, b"\r\n\r\n"].concat();
        let (head, _) = parse_response_head(&input, &limits()).unwrap();
        response_body_spec(&head, method, &limits())
    }

    #[test]
    fn response_body_spec_cases() {
        // Normal framing.
        assert_eq!(
            res_spec(b"HTTP/1.1 200 OK\r\nContent-Length: 5", b"GET"),
            Ok(BodySpec::ContentLength(5))
        );
        assert_eq!(
            res_spec(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked", b"GET"),
            Ok(BodySpec::Chunked)
        );
        assert_eq!(
            res_spec(b"HTTP/1.1 200 OK", b"GET"),
            Ok(BodySpec::UntilClose)
        );
        // 1xx / 204 / 304 never have a body, headers or not.
        for status in [100, 101, 103, 199, 204, 304] {
            let line = format!("HTTP/1.1 {status} X\r\nContent-Length: 5");
            assert_eq!(
                res_spec(line.as_bytes(), b"GET"),
                Ok(BodySpec::Empty),
                "status {status}"
            );
        }
        // HEAD responses never have a body.
        assert_eq!(
            res_spec(b"HTTP/1.1 200 OK\r\nContent-Length: 5", b"HEAD"),
            Ok(BodySpec::Empty)
        );
        assert_eq!(
            res_spec(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked", b"HEAD"),
            Ok(BodySpec::Empty)
        );
    }

    #[test]
    fn bodiless_response_framing_still_validated() {
        // Malformed framing headers are rejected even though the body is empty.
        // (m1b: on responses a non-chunked-final Transfer-Encoding is valid
        // until-close framing per RFC 9112 §6.3 rule 4, so `bogus` alone no
        // longer errors on a bodiless response — but syntactically invalid
        // TE still does.)
        assert_eq!(
            res_spec(b"HTTP/1.1 204 X\r\nTransfer-Encoding: bogus", b"GET"),
            Ok(BodySpec::Empty)
        );
        assert_eq!(
            res_spec(
                b"HTTP/1.1 204 X\r\nTransfer-Encoding: chunked, chunked",
                b"GET"
            ),
            Err(HttpError::InvalidTransferEncoding)
        );
        assert_eq!(
            res_spec(b"HTTP/1.1 304 X\r\nContent-Length: abc", b"GET"),
            Err(HttpError::InvalidContentLength)
        );
        assert_eq!(
            res_spec(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: bogus", b"HEAD"),
            Ok(BodySpec::Empty)
        );
        // Well-formed framing headers on bodiless responses are fine.
        assert_eq!(
            res_spec(b"HTTP/1.1 204 X\r\nContent-Length: 0", b"GET"),
            Ok(BodySpec::Empty)
        );
        // `Transfer-Encoding: identity` is rejected even on a 204: framing
        // headers must be well-formed, never silently swallowed.
        assert_eq!(
            res_spec(b"HTTP/1.1 204 X\r\nTransfer-Encoding: identity", b"GET"),
            Err(HttpError::IdentityTransferEncoding)
        );
    }

    #[test]
    fn connect_2xx_becomes_tunnel() {
        // RFC 9112 §6.3 rule 2: any 2xx response to CONNECT turns the
        // connection into a tunnel; framing headers must be ignored.
        assert_eq!(
            res_spec(b"HTTP/1.1 200 OK\r\nContent-Length: 5", b"CONNECT"),
            Ok(BodySpec::Tunnel)
        );
        assert_eq!(
            res_spec(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked", b"CONNECT"),
            Ok(BodySpec::Tunnel)
        );
        assert_eq!(
            res_spec(b"HTTP/1.1 200 Connection Established", b"CONNECT"),
            Ok(BodySpec::Tunnel)
        );
        // Other 2xx codes tunnel as well.
        assert_eq!(
            res_spec(b"HTTP/1.1 299 X\r\nContent-Length: 5", b"CONNECT"),
            Ok(BodySpec::Tunnel)
        );
        // Non-2xx responses to CONNECT use normal framing.
        assert_eq!(
            res_spec(
                b"HTTP/1.1 407 Proxy Auth Required\r\nContent-Length: 5",
                b"CONNECT"
            ),
            Ok(BodySpec::ContentLength(5))
        );
        assert_eq!(
            res_spec(b"HTTP/1.1 502 Bad Gateway", b"CONNECT"),
            Ok(BodySpec::UntilClose)
        );
        // 204 to CONNECT stays bodiless (rule 1 wins over rule 2).
        assert_eq!(
            res_spec(b"HTTP/1.1 204 No Content\r\nContent-Length: 5", b"CONNECT"),
            Ok(BodySpec::Empty)
        );
    }

    #[test]
    fn transfer_encoding_rejected_on_http10_request() {
        // RFC 9112 §6.1: TE on an HTTP/1.0 message is faulty framing,
        // even `identity`.
        let (head, _) = req(b"POST / HTTP/1.0\r\nTransfer-Encoding: chunked\r\n\r\n").unwrap();
        assert_eq!(
            request_body_spec(&head, &limits()),
            Err(HttpError::TransferEncodingOnHttp10)
        );
        let (head, _) = req(b"POST / HTTP/1.0\r\nTransfer-Encoding: identity\r\n\r\n").unwrap();
        assert_eq!(
            request_body_spec(&head, &limits()),
            Err(HttpError::TransferEncodingOnHttp10)
        );
        // Content-Length framing on HTTP/1.0 is unaffected.
        let (head, _) = req(b"POST / HTTP/1.0\r\nContent-Length: 5\r\n\r\n").unwrap();
        assert_eq!(
            request_body_spec(&head, &limits()),
            Ok(BodySpec::ContentLength(5))
        );
        // HTTP/1.1 with TE: chunked still works.
        let (head, _) = req(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n").unwrap();
        assert_eq!(request_body_spec(&head, &limits()), Ok(BodySpec::Chunked));
    }

    #[test]
    fn transfer_encoding_rejected_on_http10_response() {
        // RFC 9112 §6.1: TE on an HTTP/1.0 message is faulty framing.
        assert_eq!(
            res_spec(b"HTTP/1.0 200 OK\r\nTransfer-Encoding: chunked", b"GET"),
            Err(HttpError::TransferEncodingOnHttp10)
        );
        assert_eq!(
            res_spec(b"HTTP/1.0 200 OK\r\nTransfer-Encoding: identity", b"GET"),
            Err(HttpError::TransferEncodingOnHttp10)
        );
        // Content-Length framing on HTTP/1.0 is unaffected.
        assert_eq!(
            res_spec(b"HTTP/1.0 200 OK\r\nContent-Length: 5", b"GET"),
            Ok(BodySpec::ContentLength(5))
        );
        // HTTP/1.1 with TE: chunked still works.
        assert_eq!(
            res_spec(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked", b"GET"),
            Ok(BodySpec::Chunked)
        );
    }

    // -- connection persistence ----------------------------------------------

    fn persistent(raw: &[u8]) -> bool {
        let (head, _) = req(raw).unwrap();
        is_persistent(head.version, &head.headers)
    }

    #[test]
    fn connection_persistence_cases() {
        // HTTP/1.1 defaults to persistent.
        assert!(persistent(b"GET / HTTP/1.1\r\n\r\n"));
        // ... unless close is asked for.
        assert!(!persistent(b"GET / HTTP/1.1\r\nConnection: close\r\n\r\n"));
        assert!(!persistent(
            b"GET / HTTP/1.1\r\nConnection: Keep-Alive, close\r\n\r\n"
        ));
        // HTTP/1.0 defaults to close...
        assert!(!persistent(b"GET / HTTP/1.0\r\n\r\n"));
        // ... unless keep-alive is asked for.
        assert!(persistent(
            b"GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n"
        ));
        assert!(persistent(
            b"GET / HTTP/1.0\r\nConnection: Keep-Alive\r\n\r\n"
        ));
        // close wins over keep-alive.
        assert!(!persistent(
            b"GET / HTTP/1.0\r\nConnection: keep-alive\r\nConnection: close\r\n\r\n"
        ));
        // Unrelated tokens do not matter.
        assert!(persistent(b"GET / HTTP/1.1\r\nConnection: Upgrade\r\n\r\n"));
    }

    #[test]
    fn connection_tokens_parsed() {
        let (head, _) =
            req(b"GET / HTTP/1.1\r\nConnection: keep-alive, Upgrade\r\nConnection: close\r\n\r\n")
                .unwrap();
        let tokens: Vec<&str> = connection_tokens(&head.headers).collect();
        assert_eq!(tokens, vec!["keep-alive", "Upgrade", "close"]);
    }

    // -- chunked encode ------------------------------------------------------

    #[test]
    fn encode_chunked_basic() {
        let out = encode_chunked(&[b"hello".as_slice(), b" world".as_slice()], &[]).unwrap();
        assert_eq!(out, b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n");
    }

    #[test]
    fn encode_chunked_skips_empty_chunks() {
        let out = encode_chunked(&[b"".as_slice(), b"hi".as_slice(), b"".as_slice()], &[]).unwrap();
        assert_eq!(out, b"2\r\nhi\r\n0\r\n\r\n");
    }

    #[test]
    fn encode_chunked_with_trailers() {
        let out = encode_chunked(&[b"hi".as_slice()], &[("X-Sum", b"abc".as_slice())]).unwrap();
        assert_eq!(out, b"2\r\nhi\r\n0\r\nX-Sum: abc\r\n\r\n");
    }

    #[test]
    fn encode_chunked_rejects_bad_trailers() {
        assert_eq!(
            encode_chunked(&[b"hi".as_slice()], &[("Content-Length", b"5".as_slice())]),
            Err(HttpError::ForbiddenTrailer)
        );
        assert_eq!(
            encode_chunked(
                &[b"hi".as_slice()],
                &[("Transfer-Encoding", b"chunked".as_slice())]
            ),
            Err(HttpError::ForbiddenTrailer)
        );
        assert_eq!(
            encode_chunked(&[b"hi".as_slice()], &[("Bad Name", b"v".as_slice())]),
            Err(HttpError::InvalidHeaderName)
        );
        assert_eq!(
            encode_chunked(&[b"hi".as_slice()], &[("X", b"a\rb".as_slice())]),
            Err(HttpError::InvalidHeaderValue)
        );
    }

    #[test]
    fn forbidden_trailers_cover_rfc9112_categories() {
        // Every field newly forbidden by the RFC 9112 §7.1.2 categories
        // (routing, request modifiers, authentication, response control
        // data, payload processing) must be rejected on every path that
        // accepts trailers.
        let forbidden = [
            // Routing.
            "Host",
            // Request modifiers: controls and conditionals.
            "Expect",
            "Max-Forwards",
            "Range",
            "If-Match",
            "If-None-Match",
            "If-Modified-Since",
            "If-Unmodified-Since",
            "If-Range",
            // Authentication.
            "Authorization",
            "Proxy-Authenticate",
            "Proxy-Authorization",
            "WWW-Authenticate",
            // Response control data (RFC 9110 §7.1).
            "Age",
            "Cache-Control",
            "Expires",
            "Date",
            "Location",
            "Retry-After",
            "Vary",
            "Warning",
            // Payload processing.
            "Content-Encoding",
            "Content-Type",
            "Content-Range",
            "Content-Language",
            "Content-Location",
            "Digest",
        ];
        for name in forbidden {
            // Encoder.
            assert_eq!(
                encode_chunked(&[b"hi".as_slice()], &[(name, b"v".as_slice())]),
                Err(HttpError::ForbiddenTrailer),
                "encode: {name} must be a forbidden trailer"
            );
            // One-shot decoder, both canonical and lowercase spellings
            // (field names are case-insensitive).
            let raw = format!("0\r\n{name}: v\r\n\r\n");
            assert_eq!(
                decode_chunked(raw.as_bytes(), &limits(), 0),
                Err(HttpError::ForbiddenTrailer),
                "decode: {name} must be a forbidden trailer"
            );
            let lower = format!("0\r\n{}: v\r\n\r\n", name.to_ascii_lowercase());
            assert_eq!(
                decode_chunked(lower.as_bytes(), &limits(), 0),
                Err(HttpError::ForbiddenTrailer),
                "decode (lowercase): {name} must be a forbidden trailer"
            );
            // Streaming decoder.
            let mut dec = ChunkedDecoder::new(Limits::default());
            let mut events = Vec::new();
            assert_eq!(
                dec.feed(raw.as_bytes(), &mut events),
                Err(HttpError::ForbiddenTrailer),
                "streaming decode: {name} must be a forbidden trailer"
            );
        }
    }

    #[test]
    fn legitimate_trailers_still_accepted() {
        // X-Checksum is an ordinary integrity trailer: it fits none of the
        // RFC 9112 §7.1.2 forbidden categories and must keep working.
        let out =
            encode_chunked(&[b"hi".as_slice()], &[("X-Checksum", b"abc".as_slice())]).unwrap();
        assert_eq!(out, b"2\r\nhi\r\n0\r\nX-Checksum: abc\r\n\r\n");
        let msg = decode_chunked(b"0\r\nX-Checksum: abc\r\n\r\n", &limits(), 0).unwrap();
        assert_eq!(msg.trailers.get("x-checksum"), Some(b"abc".as_slice()));
        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut events = Vec::new();
        dec.feed(b"0\r\nX-Checksum: abc\r\n\r\n", &mut events)
            .unwrap();
        assert!(events.iter().any(|e| matches!(
            e,
            ChunkEvent::Trailer(t) if t.name == "X-Checksum" && t.value == b"abc"
        )));
        assert!(dec.is_done());
    }

    #[test]
    fn encode_chunked_hex_is_uppercase() {
        let data = vec![0u8; 255];
        let out = encode_chunked(&[data.as_slice()], &[]).unwrap();
        assert!(out.starts_with(b"FF\r\n"));
    }

    // -- chunked decode (one-shot) --------------------------------------------

    #[test]
    fn decode_chunked_basic() {
        let msg = decode_chunked(b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n", &limits(), 0).unwrap();
        assert_eq!(msg.chunks, vec![b"hello".as_slice(), b" world".as_slice()]);
        assert!(msg.trailers.is_empty());
        assert_eq!(msg.rest, b"");
    }

    #[test]
    fn decode_chunked_keeps_rest() {
        let msg =
            decode_chunked(b"2\r\nhi\r\n0\r\n\r\nGET /next HTTP/1.1\r\n", &limits(), 0).unwrap();
        assert_eq!(msg.chunks, vec![b"hi".as_slice()]);
        assert_eq!(msg.rest, b"GET /next HTTP/1.1\r\n");
    }

    #[test]
    fn decode_chunked_trailers() {
        let msg =
            decode_chunked(b"3\r\nfoo\r\n0\r\nX-A: 1\r\nX-B: two\r\n\r\n", &limits(), 0).unwrap();
        assert_eq!(msg.trailers.len(), 2);
        assert_eq!(msg.trailers.get("x-a"), Some(b"1".as_slice()));
        assert_eq!(msg.trailers.get("x-b"), Some(b"two".as_slice()));
    }

    #[test]
    fn decode_chunked_trailers_share_max_headers_with_head() {
        // Limits::max_headers covers headers + trailers: one head header
        // leaves room for exactly one trailer field.
        let limits = Limits {
            max_headers: 2,
            ..Default::default()
        };
        assert_eq!(
            decode_chunked(b"0\r\nX-A: 1\r\nX-B: 2\r\n\r\n", &limits, 1),
            Err(HttpError::TooManyHeaders)
        );
        let msg = decode_chunked(b"0\r\nX-A: 1\r\n\r\n", &limits, 1).unwrap();
        assert_eq!(msg.trailers.len(), 1);
        // A head that already exhausted the budget rejects any trailer.
        assert_eq!(
            decode_chunked(b"0\r\nX-A: 1\r\n\r\n", &limits, 2),
            Err(HttpError::TooManyHeaders)
        );
    }

    #[test]
    fn decode_chunked_extensions_accepted() {
        let msg = decode_chunked(
            b"5;ext=1;foo=\"bar\\\"baz\";flag\r\nhello\r\n0\r\n\r\n",
            &limits(),
            0,
        )
        .unwrap();
        assert_eq!(msg.chunks, vec![b"hello".as_slice()]);
    }

    #[test]
    fn decode_chunked_hex_variants() {
        let msg = decode_chunked(b"A\r\n0123456789\r\n0\r\n\r\n", &limits(), 0).unwrap();
        assert_eq!(msg.chunks, vec![b"0123456789".as_slice()]);
        let msg = decode_chunked(b"005\r\nhello\r\n0\r\n\r\n", &limits(), 0).unwrap();
        assert_eq!(msg.chunks, vec![b"hello".as_slice()]);
    }

    #[test]
    fn decode_chunked_empty_body() {
        let msg = decode_chunked(b"0\r\n\r\n", &limits(), 0).unwrap();
        assert!(msg.chunks.is_empty());
        assert!(msg.trailers.is_empty());
    }

    #[test]
    fn decode_chunked_lone_lf_terminator() {
        // Chunk-size lines go through `next_line`: a bare LF is rejected
        // (RFC 9112 §2.2), not treated as a size-line terminator.
        assert_eq!(
            decode_chunked(b"5\nhello\n0\r\n\r\n", &limits(), 0),
            Err(HttpError::BareLf)
        );
        // Trailer lines go through `next_line` too.
        assert_eq!(
            decode_chunked(b"5\r\nhello\r\n0\r\nX-A: 1\n\n", &limits(), 0),
            Err(HttpError::BareLf)
        );
        // The chunk-*data* terminator has its own lenient path (not
        // `next_line`) and still accepts a lone LF.
        let msg = decode_chunked(b"5\r\nhello\n0\r\n\r\n", &limits(), 0).unwrap();
        assert_eq!(msg.chunks, vec![b"hello".as_slice()]);
        // A trailing lone CR may be the first half of a split CRLF: incomplete.
        assert_eq!(
            decode_chunked(b"5\r\nhello\r", &limits(), 0),
            Err(HttpError::Incomplete)
        );
        // A byte that can never start a line ending is a terminator error.
        assert_eq!(
            decode_chunked(b"5\r\nhelloX", &limits(), 0),
            Err(HttpError::InvalidChunkTerminator)
        );
    }

    #[test]
    fn bare_lf_rejected_uniformly_by_both_chunked_decoders() {
        // Chunk-size and trailer lines must end in CRLF in both the
        // one-shot and the streaming decoder (RFC 9112 §2.2); both report
        // `HttpError::BareLf`. The chunk-*data* terminator stays lenient in
        // both (see `decode_chunked_lone_lf_terminator` and
        // `streaming_lone_lf_terminator`).
        let cases: &[&[u8]] = &[
            b"5\nhello\r\n0\r\n\r\n",           // bare LF ends a size line
            b"5\r\nhello\r\n0\n\r\n",           // bare LF ends the last-chunk line
            b"5\r\nhello\r\n0\r\nX-A: 1\n\r\n", // bare LF ends a trailer line
            b"5\r\nhello\r\n0\r\nX-A: 1\r\n\n", // bare LF ends the trailer section
        ];
        for input in cases {
            assert_eq!(
                decode_chunked(input, &limits(), 0).map(|m| m.chunks.len()),
                Err(HttpError::BareLf),
                "one-shot disagrees: {input:?}"
            );
            let mut dec = ChunkedDecoder::new(limits());
            let mut events = Vec::new();
            assert_eq!(
                dec.feed(input, &mut events).map(|_| events.len()),
                Err(HttpError::BareLf),
                "streaming disagrees: {input:?}"
            );
        }
        // Lenient in both: a lone LF as the chunk-data terminator decodes.
        let input = b"5\r\nhello\n0\r\n\r\n";
        let msg = decode_chunked(input, &limits(), 0).unwrap();
        assert_eq!(msg.chunks, vec![b"hello".as_slice()]);
        let mut dec = ChunkedDecoder::new(limits());
        let mut events = Vec::new();
        dec.feed(input, &mut events).unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ChunkEvent::Data(b"hello")))
        );
        assert!(events.iter().any(|e| matches!(e, ChunkEvent::End)));
    }

    #[test]
    fn decode_chunked_malformed_rejected() {
        // Bad hex.
        assert_eq!(
            decode_chunked(b"zz\r\nhello\r\n0\r\n\r\n", &limits(), 0),
            Err(HttpError::InvalidChunkSize)
        );
        // Empty size line.
        assert_eq!(
            decode_chunked(b"\r\n0\r\n\r\n", &limits(), 0),
            Err(HttpError::InvalidChunkSize)
        );
        // 17 hex digits: cannot fit in u64.
        assert_eq!(
            decode_chunked(b"10000000000000000\r\n0\r\n\r\n", &limits(), 0),
            Err(HttpError::InvalidChunkSize)
        );
        // Space inside the size.
        assert_eq!(
            decode_chunked(b"5 \r\nhello\r\n0\r\n\r\n", &limits(), 0),
            Err(HttpError::InvalidChunkSize)
        );
        // Bad chunk extension.
        assert_eq!(
            decode_chunked(b"5;ext=\r\nhello\r\n0\r\n\r\n", &limits(), 0),
            Err(HttpError::InvalidChunkExtension)
        );
        assert_eq!(
            decode_chunked(b"5;\r\nhello\r\n0\r\n\r\n", &limits(), 0),
            Err(HttpError::InvalidChunkExtension)
        );
        assert_eq!(
            decode_chunked(b"5;\"x\"\r\nhello\r\n0\r\n\r\n", &limits(), 0),
            Err(HttpError::InvalidChunkExtension)
        );
        // Chunk data not followed by CRLF.
        assert_eq!(
            decode_chunked(b"5\r\nhelloXX0\r\n\r\n", &limits(), 0),
            Err(HttpError::InvalidChunkTerminator)
        );
        // Forbidden trailer fields.
        assert_eq!(
            decode_chunked(b"0\r\nContent-Length: 5\r\n\r\n", &limits(), 0),
            Err(HttpError::ForbiddenTrailer)
        );
        assert_eq!(
            decode_chunked(b"0\r\nTE: trailers\r\n\r\n", &limits(), 0),
            Err(HttpError::ForbiddenTrailer)
        );
        // Bare CR inside the chunked body.
        assert_eq!(
            decode_chunked(b"5\rhello\r\n0\r\n\r\n", &limits(), 0),
            Err(HttpError::BareCr)
        );
    }

    #[test]
    fn decode_chunked_huge_size_rejected_not_panicking() {
        // B1: 16 hex F's (u64::MAX) must be a clean error, never a panic.
        // This exact 18-byte message used to kill the process.
        assert_eq!(
            decode_chunked(b"FFFFFFFFFFFFFFFF\r\n", &limits(), 0),
            Err(HttpError::ChunkTooLarge)
        );
        // Lowercase and near-max variants.
        assert_eq!(
            decode_chunked(b"ffffffffffffffff\r\n", &limits(), 0),
            Err(HttpError::ChunkTooLarge)
        );
        assert_eq!(
            decode_chunked(b"FFFFFFFFFFFFFFFE\r\n", &limits(), 0),
            Err(HttpError::ChunkTooLarge)
        );
        // Just over the 16 MiB cap (0x1000001).
        assert_eq!(
            decode_chunked(b"1000001\r\n", &limits(), 0),
            Err(HttpError::ChunkTooLarge)
        );
        // A chunk of exactly MAX_CHUNK_SIZE (0x1000000) is accepted
        // size-wise: with a short buffer the decoder reports Incomplete.
        assert_eq!(
            decode_chunked(b"1000000\r\n", &limits(), 0),
            Err(HttpError::Incomplete)
        );
        // And a full 16 MiB chunk decodes.
        let mut body = Vec::from(b"1000000\r\n".as_slice());
        body.extend_from_slice(&vec![b'x'; MAX_CHUNK_SIZE as usize]);
        body.extend_from_slice(b"\r\n0\r\n\r\n");
        let msg = decode_chunked(&body, &limits(), 0).unwrap();
        assert_eq!(msg.chunks.len(), 1);
        assert_eq!(msg.chunks[0].len(), MAX_CHUNK_SIZE as usize);
        assert_eq!(msg.rest, b"");
    }

    #[test]
    fn decode_chunked_incomplete() {
        assert_eq!(
            decode_chunked(b"", &limits(), 0),
            Err(HttpError::Incomplete)
        );
        assert_eq!(
            decode_chunked(b"5\r\nhel", &limits(), 0),
            Err(HttpError::Incomplete)
        );
        assert_eq!(
            decode_chunked(b"5\r\nhello\r\n", &limits(), 0),
            Err(HttpError::Incomplete)
        );
        assert_eq!(
            decode_chunked(b"5\r\nhello\r\n0\r\n", &limits(), 0),
            Err(HttpError::Incomplete)
        );
        assert_eq!(
            decode_chunked(b"5\r\nhello\r\n0\r\nX-A: 1\r\n", &limits(), 0),
            Err(HttpError::Incomplete)
        );
    }

    #[test]
    fn chunked_roundtrip() {
        let chunks: Vec<&[u8]> = vec![b"hello".as_slice(), b", ".as_slice(), b"world".as_slice()];
        let encoded = encode_chunked(&chunks, &[("X-Check", b"ok".as_slice())]).unwrap();
        let msg = decode_chunked(&encoded, &limits(), 0).unwrap();
        assert_eq!(msg.chunks, chunks);
        assert_eq!(msg.trailers.get("x-check"), Some(b"ok".as_slice()));
        assert_eq!(msg.rest, b"");
    }

    #[test]
    fn decode_chunked_enforces_body_limit_cumulatively() {
        let capped = Limits {
            max_body_bytes: 10,
            ..Default::default()
        };
        // Each chunk fits alone, but the cumulative payload exceeds the cap.
        let body = b"6\r\nhello!\r\n6\r\n world\r\n0\r\n\r\n";
        assert_eq!(
            decode_chunked(body, &capped, 0),
            Err(HttpError::BodyTooLarge)
        );
        // A single oversized chunk is caught too.
        let body = b"B\r\n0123456789A\r\n0\r\n\r\n";
        assert_eq!(
            decode_chunked(body, &capped, 0),
            Err(HttpError::BodyTooLarge)
        );
        // Exactly at the cap passes.
        let body = b"5\r\nhello\r\n5\r\nworld\r\n0\r\n\r\n";
        let msg = decode_chunked(body, &capped, 0).unwrap();
        assert_eq!(msg.chunks, vec![b"hello".as_slice(), b"world".as_slice()]);
    }

    // -- streaming decoder -----------------------------------------------------

    fn collect_stream(feeds: &[&[u8]]) -> (Vec<u8>, Vec<TrailerField>, bool, usize) {
        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut data = Vec::new();
        let mut trailers = Vec::new();
        let mut events = Vec::new();
        let mut consumed = 0usize;
        for feed in feeds {
            events.clear();
            consumed += dec.feed(feed, &mut events).unwrap();
            for ev in events.drain(..) {
                match ev {
                    ChunkEvent::Data(d) => data.extend_from_slice(d),
                    ChunkEvent::Trailer(t) => trailers.push(t),
                    ChunkEvent::End => {}
                }
            }
        }
        (data, trailers, dec.is_done(), consumed)
    }

    #[test]
    fn streaming_byte_by_byte() {
        let input = b"5\r\nhello\r\n6\r\n world\r\n0\r\nX-T: 1\r\n\r\n";
        let feeds: Vec<&[u8]> = input.chunks(1).map(|c| c as &[u8]).collect();
        let (data, trailers, done, consumed) = collect_stream(&feeds);
        assert_eq!(data, b"hello world");
        assert_eq!(trailers.len(), 1);
        assert_eq!(trailers[0].name, "X-T");
        assert_eq!(trailers[0].value, b"1");
        assert!(done);
        assert_eq!(consumed, input.len());
    }

    #[test]
    fn streaming_awkward_splits() {
        // Splits inside CRLF pairs, inside hex digits, inside data.
        let feeds: &[&[u8]] = &[
            b"5", b"\r", b"\nhel", b"lo\r", b"\n0\r", b"\n", b"\r", b"\n",
        ];
        let (data, trailers, done, consumed) = collect_stream(feeds);
        assert_eq!(data, b"hello");
        assert!(trailers.is_empty());
        assert!(done);
        assert_eq!(consumed, feeds.iter().map(|f| f.len()).sum::<usize>());
    }

    #[test]
    fn streaming_one_shot_matches_incremental() {
        let input = b"3\r\nfoo\r\n0\r\n\r\n";
        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut events = Vec::new();
        let consumed = dec.feed(input, &mut events).unwrap();
        assert_eq!(consumed, input.len());
        assert!(matches!(events[0], ChunkEvent::Data(b"foo")));
        assert!(matches!(events[1], ChunkEvent::End));
        assert_eq!(consumed, input.len());
        assert!(dec.is_done());
        // Input after End is ignored: no events, nothing consumed.
        events.clear();
        let consumed = dec.feed(b"junk", &mut events).unwrap();
        assert!(events.is_empty());
        assert_eq!(consumed, 0);
    }

    #[test]
    fn streaming_consumed_stops_at_pipelined_next_message() {
        // The terminator is 5 bytes; the pipelined next request must be
        // recoverable via &input[consumed..], not swallowed.
        let input = b"0\r\n\r\nGET /next HTTP/1.1\r\n";
        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut events = Vec::new();
        let consumed = dec.feed(input, &mut events).unwrap();
        assert!(matches!(events.as_slice(), [ChunkEvent::End]));
        assert_eq!(consumed, 5);
        assert_eq!(&input[consumed..], b"GET /next HTTP/1.1\r\n");
        assert!(dec.is_done());
    }

    #[test]
    fn streaming_consumed_sums_across_byte_at_a_time_feeds() {
        let input = b"3\r\nfoo\r\n0\r\n\r\n";
        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut total = 0usize;
        let mut data = Vec::new();
        let mut events = Vec::new();
        for byte in input.chunks(1) {
            events.clear();
            let consumed = dec.feed(byte, &mut events).unwrap();
            assert_eq!(consumed, byte.len());
            total += consumed;
            for ev in events.drain(..) {
                if let ChunkEvent::Data(d) = ev {
                    data.extend_from_slice(d);
                }
            }
        }
        assert_eq!(data, b"foo");
        assert_eq!(total, input.len());
        assert!(dec.is_done());
        // Input after End is ignored.
        events.clear();
        assert_eq!(dec.feed(b"junk", &mut events).unwrap(), 0);
        assert!(events.is_empty());
    }

    #[test]
    fn streaming_errors() {
        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut events = Vec::new();
        assert_eq!(
            dec.feed(b"zz\r\n", &mut events),
            Err(HttpError::InvalidChunkSize)
        );

        let mut dec = ChunkedDecoder::new(Limits::default());
        events.clear();
        dec.feed(b"5\r\nhello", &mut events).unwrap();
        assert!(events.iter().any(|e| matches!(e, ChunkEvent::Data(_))));
        assert_eq!(
            dec.feed(b"XX", &mut events),
            Err(HttpError::InvalidChunkTerminator)
        );

        let mut dec = ChunkedDecoder::new(Limits::default());
        assert_eq!(
            dec.feed(b"0\r\nContent-Length: 1\r\n\r\n", &mut events),
            Err(HttpError::ForbiddenTrailer)
        );
    }

    #[test]
    fn streaming_huge_chunk_size_rejected() {
        // Same B1 fix as the one-shot path: an absurd chunk-size line is a
        // clean error through the streaming decoder too.
        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut events = Vec::new();
        assert_eq!(
            dec.feed(b"FFFFFFFFFFFFFFFF\r\n", &mut events),
            Err(HttpError::ChunkTooLarge)
        );
        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut events = Vec::new();
        assert_eq!(
            dec.feed(b"1000001\r\n", &mut events),
            Err(HttpError::ChunkTooLarge)
        );
    }

    #[test]
    fn streaming_respects_line_limit() {
        let tiny = Limits {
            max_line_bytes: 4,
            ..Default::default()
        };
        let mut dec = ChunkedDecoder::new(tiny);
        let mut events = Vec::new();
        assert_eq!(
            dec.feed(b"12345\r\n", &mut events),
            Err(HttpError::LineTooLong)
        );
    }

    #[test]
    fn streaming_trailers_share_max_headers_with_head() {
        // Limits::max_headers covers headers + trailers: one head header
        // leaves room for exactly one trailer field.
        let limits = Limits {
            max_headers: 2,
            ..Default::default()
        };
        let mut dec = ChunkedDecoder::new_with_headers_seen(limits, 1);
        let mut events = Vec::new();
        dec.feed(b"0\r\nX-A: 1\r\n\r\n", &mut events).unwrap();
        assert!(events.iter().any(|e| matches!(e, ChunkEvent::Trailer(_))));
        assert!(dec.is_done());

        let mut dec = ChunkedDecoder::new_with_headers_seen(limits, 1);
        let mut events = Vec::new();
        assert_eq!(
            dec.feed(b"0\r\nX-A: 1\r\nX-B: 2\r\n\r\n", &mut events),
            Err(HttpError::TooManyHeaders)
        );
        // A head that already exhausted the budget rejects any trailer.
        let mut dec = ChunkedDecoder::new_with_headers_seen(limits, 2);
        let mut events = Vec::new();
        assert_eq!(
            dec.feed(b"0\r\nX-A: 1\r\n\r\n", &mut events),
            Err(HttpError::TooManyHeaders)
        );
    }

    #[test]
    fn streaming_lone_lf_terminator() {
        // The streaming decoder tolerates a lone LF after chunk data,
        // matching `decode_chunked` (m4); message head lines still
        // require strict CRLF.
        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut events = Vec::new();
        dec.feed(b"5\r\nhello\n0\r\n\r\n", &mut events).unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ChunkEvent::Data(b"hello")))
        );
        assert!(events.iter().any(|e| matches!(e, ChunkEvent::End)));
        assert!(dec.is_done());
        // Lone LF arriving in its own feed.
        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut events = Vec::new();
        dec.feed(b"5\r\nhello", &mut events).unwrap();
        assert!(events.iter().any(|e| matches!(e, ChunkEvent::Data(_))));
        events.clear();
        dec.feed(b"\n0\r\n\r\n", &mut events).unwrap();
        assert!(events.iter().any(|e| matches!(e, ChunkEvent::End)));
        assert!(dec.is_done());
        // A CR followed by a non-LF byte is still rejected, not tolerated.
        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut events = Vec::new();
        dec.feed(b"5\r\nhello", &mut events).unwrap();
        assert_eq!(dec.feed(b"\rX", &mut events), Err(HttpError::BareCr));
    }

    #[test]
    fn streaming_bare_lf_line_rejected() {
        // Same strict CRLF as the one-shot decoder (RFC 9112 §2.2): a bare
        // LF ending a chunk-size or trailer line is `HttpError::BareLf`,
        // even when the line was buffered across feeds and the LF arrives
        // at the start of the next one.
        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut events = Vec::new();
        assert_eq!(
            dec.feed(b"5\nhello\r\n0\r\n\r\n", &mut events),
            Err(HttpError::BareLf)
        );

        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut events = Vec::new();
        dec.feed(b"5", &mut events).unwrap();
        assert_eq!(
            dec.feed(b"\nhello\r\n0\r\n\r\n", &mut events),
            Err(HttpError::BareLf)
        );

        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut events = Vec::new();
        dec.feed(b"5\r\nhello\r\n0\r\nX-A: 1", &mut events).unwrap();
        assert_eq!(dec.feed(b"\n\r\n", &mut events), Err(HttpError::BareLf));

        // A split CRLF across feeds still decodes: only the *bare* LF is
        // rejected, not every multi-feed line ending.
        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut events = Vec::new();
        dec.feed(b"5\r", &mut events).unwrap();
        events.clear();
        dec.feed(b"\nhello\r\n0\r\n\r\n", &mut events).unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ChunkEvent::Data(b"hello")))
        );
        assert!(events.iter().any(|e| matches!(e, ChunkEvent::End)));
        assert!(dec.is_done());
    }

    #[test]
    fn over_limit_line_with_bare_cr_reports_line_too_long_in_both_paths() {
        // Error precedence is length-first in both the one-shot line parser
        // and the streaming decoder: an over-limit line holding an interior
        // CR (and no LF yet) is LineTooLong in both (m5).
        let tiny = Limits {
            max_line_bytes: 8,
            ..Default::default()
        };
        let input = b"AAAAAAAAA\rBB"; // 12 bytes, interior CR, no LF.
        assert_eq!(decode_chunked(input, &tiny, 0), Err(HttpError::LineTooLong));
        let mut dec = ChunkedDecoder::new(tiny);
        let mut events = Vec::new();
        assert_eq!(dec.feed(input, &mut events), Err(HttpError::LineTooLong));
    }

    // Erroring poisons the decoder; reset() restores the initial state so
    // a fresh message decodes successfully afterwards.
    #[test]
    fn streaming_reset_recovers_from_error() {
        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut events = Vec::new();
        assert_eq!(
            dec.feed(b"zz\r\n", &mut events),
            Err(HttpError::InvalidChunkSize)
        );
        assert!(!dec.is_done());
        dec.reset();
        let mut data = Vec::new();
        events.clear();
        dec.feed(b"5\r\nhello\r\n0\r\n\r\n", &mut events).unwrap();
        for ev in events.drain(..) {
            match ev {
                ChunkEvent::Data(d) => data.extend_from_slice(d),
                ChunkEvent::End => {}
                ChunkEvent::Trailer(_) => panic!("no trailers expected"),
            }
        }
        assert_eq!(data, b"hello");
        assert!(dec.is_done());
    }

    #[test]
    fn streaming_enforces_body_limit_cumulatively() {
        let capped = Limits {
            max_body_bytes: 10,
            ..Default::default()
        };
        // Each chunk fits alone, but the cumulative payload across feeds
        // exceeds the cap.
        let mut dec = ChunkedDecoder::new(capped);
        let mut events = Vec::new();
        dec.feed(b"6\r\nhello!\r\n", &mut events).unwrap();
        assert!(events.iter().any(|e| matches!(
            e,
            ChunkEvent::Data(d) if *d == b"hello!"
        )));
        events.clear();
        assert_eq!(
            dec.feed(b"6\r\n world\r\n", &mut events),
            Err(HttpError::BodyTooLarge)
        );
        // Exactly at the cap passes, even split across feeds.
        let mut dec = ChunkedDecoder::new(capped);
        let feeds: &[&[u8]] = &[b"5\r\nhel", b"lo\r\n5\r\nwo", b"rld\r\n0\r\n\r\n"];
        let mut data = Vec::new();
        let mut events = Vec::new();
        for feed in feeds {
            events.clear();
            dec.feed(feed, &mut events).unwrap();
            for ev in events.drain(..) {
                if let ChunkEvent::Data(d) = ev {
                    data.extend_from_slice(d);
                }
            }
        }
        assert_eq!(data, b"helloworld");
        assert!(dec.is_done());
    }

    // -- serialization --------------------------------------------------------

    #[test]
    fn serialize_request_roundtrip() {
        let bytes = serialize_request_head(
            "GET",
            "/index?q=1",
            HttpVersion::HTTP_11,
            &[
                ("Host", b"example.com".as_slice()),
                ("X-A", b"1".as_slice()),
            ],
        )
        .unwrap();
        assert_eq!(
            bytes,
            b"GET /index?q=1 HTTP/1.1\r\nHost: example.com\r\nX-A: 1\r\n\r\n"
        );
        let (head, consumed) = req(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(head.method, "GET");
        assert_eq!(head.target, "/index?q=1");
        assert_eq!(head.version, HttpVersion::HTTP_11);
        assert_eq!(head.headers.len(), 2);
    }

    #[test]
    fn serialize_response_roundtrip() {
        let bytes = serialize_response_head(
            HttpVersion::HTTP_10,
            404,
            "Not Found",
            &[("Content-Length", b"0".as_slice())],
        )
        .unwrap();
        assert_eq!(
            bytes,
            b"HTTP/1.0 404 Not Found\r\nContent-Length: 0\r\n\r\n"
        );
        let (head, consumed) = res(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(head.status, 404);
        assert_eq!(head.reason, "Not Found");
    }

    #[test]
    fn serialize_rejects_invalid_input() {
        let v = HttpVersion::HTTP_11;
        assert_eq!(
            serialize_request_head("GE T", "/", v, &[]),
            Err(HttpError::InvalidMethod)
        );
        assert_eq!(
            serialize_request_head("GET", "/a b", v, &[]),
            Err(HttpError::InvalidTarget)
        );
        assert_eq!(
            serialize_request_head("GET", "/", v, &[("Bad Name", b"v".as_slice())]),
            Err(HttpError::InvalidHeaderName)
        );
        assert_eq!(
            serialize_request_head("GET", "/", v, &[("X", b"a\r\nb".as_slice())]),
            Err(HttpError::InvalidHeaderValue)
        );
        assert_eq!(
            serialize_response_head(v, 99, "OK", &[]),
            Err(HttpError::InvalidStatus)
        );
        assert_eq!(
            serialize_response_head(v, 1000, "OK", &[]),
            Err(HttpError::InvalidStatus)
        );
        assert_eq!(
            serialize_response_head(v, 200, "O\x01K", &[]),
            Err(HttpError::InvalidStatusLine)
        );
    }

    // -- misc ------------------------------------------------------------------

    #[test]
    fn version_display() {
        assert_eq!(HttpVersion::HTTP_11.to_string(), "HTTP/1.1");
        assert_eq!(HttpVersion::HTTP_10.to_string(), "HTTP/1.0");
    }

    #[test]
    fn error_display_is_human_readable() {
        assert!(HttpError::ObsFoldRejected.to_string().contains("obs-fold"));
        assert!(HttpError::Incomplete.to_string().contains("incomplete"));
        // HttpError is usable as a std error.
        fn assert_error<E: std::error::Error>() {}
        assert_error::<HttpError>();
    }

    #[test]
    fn limits_default_values() {
        let l = Limits::default();
        assert_eq!(l.max_headers, 100);
        assert_eq!(l.max_head_bytes, 32 * 1024);
        assert_eq!(l.max_line_bytes, 8 * 1024);
        assert_eq!(l.max_body_bytes, 256 * 1024 * 1024);
    }
}
