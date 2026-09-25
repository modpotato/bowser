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
//! * Header lines must end with CRLF; a lone LF is tolerated, a bare CR
//!   is rejected.
//! * `obs-fold` (a header line starting with SP/HTAB) is rejected, not
//!   unfolded.
//! * Whitespace between a field name and the colon is rejected.
//! * `Transfer-Encoding: chunked` must be the final coding; anything else
//!   is an error.

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
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_headers: 100,
            max_head_bytes: 32 * 1024,
            max_line_bytes: 8 * 1024,
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
    /// A `Content-Length` value is not a valid non-negative integer.
    InvalidContentLength,
    /// Multiple `Content-Length` values disagree with each other.
    ConflictingContentLength,
    /// A `Transfer-Encoding` list is syntactically invalid.
    InvalidTransferEncoding,
    /// A `Transfer-Encoding` coding other than `chunked`/`identity` was seen.
    UnsupportedTransferEncoding,
    /// `chunked` appears but is not the final transfer coding.
    ChunkedNotFinal,
    /// A chunk-size line has no valid hexadecimal size.
    InvalidChunkSize,
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
            Self::InvalidContentLength => "invalid Content-Length value",
            Self::ConflictingContentLength => "conflicting Content-Length values",
            Self::InvalidTransferEncoding => "invalid Transfer-Encoding value",
            Self::UnsupportedTransferEncoding => "unsupported transfer coding",
            Self::ChunkedNotFinal => "chunked is not the final transfer coding",
            Self::InvalidChunkSize => "invalid chunk-size",
            Self::InvalidChunkExtension => "invalid chunk extension",
            Self::InvalidChunkTerminator => "chunk data not followed by CRLF",
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
fn is_token(bytes: &[u8]) -> bool {
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
    &bytes[start..end]
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
/// Accepts CRLF; a lone LF is tolerated. A CR not followed by LF is an
/// error, except a CR at the very end of the buffer, which may be a split
/// CRLF — that yields [`HttpError::Incomplete`] so the caller can feed more.
fn next_line(input: &[u8], max_line_bytes: usize) -> Result<Line<'_>, HttpError> {
    match input.iter().position(|&b| b == b'\n') {
        Some(lf) => {
            let mut line = &input[..lf];
            if line.last() == Some(&b'\r') {
                line = &line[..line.len() - 1];
            }
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
    pub fn get_all<'s>(&'s self, name: &'s str) -> impl Iterator<Item = &'a [u8]> + 's {
        self.headers
            .iter()
            .filter(move |h| h.name.eq_ignore_ascii_case(name))
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
pub fn content_length(headers: &Headers) -> Result<Option<u64>, HttpError> {
    let mut result: Option<u64> = None;
    for value in headers.get_all("content-length") {
        for part in value.split(|&b| b == b',') {
            let part = trim_ows(part);
            if part.is_empty() || !part.iter().all(|b| b.is_ascii_digit()) {
                return Err(HttpError::InvalidContentLength);
            }
            let n = parse_decimal(part).ok_or(HttpError::InvalidContentLength)?;
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
    /// No transfer coding applies (absent, or only `identity`).
    None,
    Chunked,
}

/// The effective transfer coding. `chunked` must appear exactly once and be
/// the final coding; any other non-`identity` coding is unsupported.
/// When `Transfer-Encoding` is present it overrides `Content-Length`
/// (RFC 9112 §6.3); callers implement that by checking this first.
fn transfer_encoding(headers: &Headers) -> Result<TransferOutcome, HttpError> {
    let mut codings: Vec<&[u8]> = Vec::new();
    for value in headers.get_all("transfer-encoding") {
        for part in value.split(|&b| b == b',') {
            // A coding may carry parameters (`gzip; q=0.8`); only the token matters.
            let coding = trim_ows(part.split(|&b| b == b';').next().unwrap_or(&[]));
            if coding.is_empty() || !is_token(coding) {
                return Err(HttpError::InvalidTransferEncoding);
            }
            codings.push(coding);
        }
    }
    let meaningful: Vec<&[u8]> = codings
        .into_iter()
        .filter(|c| !c.eq_ignore_ascii_case(b"identity"))
        .collect();
    if meaningful.is_empty() {
        return Ok(TransferOutcome::None);
    }
    // Framing validity first: chunked must be the final coding.
    let chunked_count = meaningful
        .iter()
        .filter(|c| c.eq_ignore_ascii_case(b"chunked"))
        .count();
    if chunked_count > 0
        && !meaningful
            .last()
            .is_some_and(|c| c.eq_ignore_ascii_case(b"chunked"))
    {
        return Err(HttpError::ChunkedNotFinal);
    }
    if chunked_count == 0
        || meaningful
            .iter()
            .any(|c| !c.eq_ignore_ascii_case(b"chunked"))
    {
        return Err(HttpError::UnsupportedTransferEncoding);
    }
    if chunked_count > 1 {
        return Err(HttpError::InvalidTransferEncoding);
    }
    Ok(TransferOutcome::Chunked)
}

/// How a request's body is framed. With neither `Transfer-Encoding` nor
/// `Content-Length` the body length is zero (RFC 9112 §6.3).
pub fn request_body_spec(head: &RequestHead) -> Result<BodySpec, HttpError> {
    match transfer_encoding(&head.headers)? {
        TransferOutcome::Chunked => Ok(BodySpec::Chunked),
        TransferOutcome::None => Ok(match content_length(&head.headers)? {
            Some(n) => BodySpec::ContentLength(n),
            None => BodySpec::Empty,
        }),
    }
}

/// How a response's body is framed (RFC 9112 §6.3, RFC 9110 §8.6, §15).
///
/// `request_method` is the method of the request this responds to: responses
/// to `HEAD`, and responses with status 1xx/204/304, never carry a body.
pub fn response_body_spec(
    head: &ResponseHead,
    request_method: &[u8],
) -> Result<BodySpec, HttpError> {
    if request_method == b"HEAD" {
        return Ok(BodySpec::Empty);
    }
    match head.status {
        100..=199 | 204 | 304 => return Ok(BodySpec::Empty),
        _ => {}
    }
    match transfer_encoding(&head.headers)? {
        TransferOutcome::Chunked => Ok(BodySpec::Chunked),
        TransferOutcome::None => Ok(match content_length(&head.headers)? {
            Some(n) => BodySpec::ContentLength(n),
            None => BodySpec::UntilClose,
        }),
    }
}

// ---------------------------------------------------------------------------
// Connection persistence
// ---------------------------------------------------------------------------

/// The tokens of every `Connection` header field, in order, OWS-trimmed.
/// Non-UTF-8 tokens are skipped.
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
/// trailer fields.
fn parse_trailers<'a>(input: &'a [u8], limits: &Limits) -> Result<(Headers<'a>, usize), HttpError> {
    let (headers, len) = parse_headers(input, limits, 0)?;
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
/// `input` — no copying. Returns [`HttpError::Incomplete`] if the buffer
/// does not yet hold the whole body.
pub fn decode_chunked<'a>(
    input: &'a [u8],
    limits: &Limits,
) -> Result<ChunkedMessage<'a>, HttpError> {
    let mut chunks = Vec::new();
    let mut pos = 0usize;
    loop {
        let line = next_line(&input[pos..], limits.max_line_bytes)?;
        pos += line.consumed;
        let (size, is_last) = parse_chunk_size(line.bytes)?;
        if is_last {
            let (trailers, tlen) = parse_trailers(&input[pos..], limits)?;
            pos += tlen;
            return Ok(ChunkedMessage {
                chunks,
                trailers,
                rest: &input[pos..],
            });
        }
        let size = size as usize;
        if input.len() - pos < size + 2 {
            return Err(HttpError::Incomplete);
        }
        chunks.push(&input[pos..pos + size]);
        pos += size;
        if &input[pos..pos + 2] != b"\r\n" {
            return Err(HttpError::InvalidChunkTerminator);
        }
        pos += 2;
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
/// Feed it arbitrary slices; it emits [`ChunkEvent::Data`] slices that
/// borrow the fed input, so the data path never copies. Only trailer
/// fields are copied into owned values.
pub struct ChunkedDecoder {
    state: DecoderState,
    line: Vec<u8>,
    limits: Limits,
    trailer_count: usize,
}

impl ChunkedDecoder {
    pub fn new(limits: Limits) -> Self {
        Self {
            state: DecoderState::ChunkSize,
            line: Vec::new(),
            limits,
            trailer_count: 0,
        }
    }

    /// True once the terminating chunk and trailers were fully consumed.
    pub fn is_done(&self) -> bool {
        self.state == DecoderState::Done
    }

    /// Feed input bytes; returns the events they complete. After
    /// [`is_done`](Self::is_done) further input is ignored.
    pub fn feed<'a>(&mut self, mut input: &'a [u8]) -> Result<Vec<ChunkEvent<'a>>, HttpError> {
        let mut events = Vec::new();
        while !input.is_empty() {
            match self.state {
                DecoderState::Done => break,
                DecoderState::ChunkData { remaining } => {
                    let take = (remaining.min(input.len() as u64)) as usize;
                    let (data, rest) = input.split_at(take);
                    if !data.is_empty() {
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
                    if input[0] != b'\r' {
                        return Err(HttpError::InvalidChunkTerminator);
                    }
                    input = &input[1..];
                    self.state = DecoderState::ChunkDataLf;
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
                            if self.line.len() + lf > self.limits.max_line_bytes {
                                return Err(HttpError::LineTooLong);
                            }
                            self.line.extend_from_slice(&input[..lf]);
                            input = &input[lf + 1..];
                            let is_trailer = self.state == DecoderState::TrailerLine;
                            self.handle_line(&mut events, is_trailer)?;
                        }
                        None => {
                            for (i, &b) in input.iter().enumerate() {
                                if b == b'\r' && i + 1 < input.len() {
                                    return Err(HttpError::BareCr);
                                }
                            }
                            if self.line.len() + input.len() > self.limits.max_line_bytes {
                                return Err(HttpError::LineTooLong);
                            }
                            self.line.extend_from_slice(input);
                            input = &[];
                        }
                    }
                }
            }
        }
        Ok(events)
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
            if self.trailer_count >= self.limits.max_headers {
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
    fn lone_lf_is_tolerated() {
        let (head, consumed) = req(b"GET / HTTP/1.1\nHost: x\n\n").unwrap();
        assert_eq!(head.method, "GET");
        assert_eq!(head.headers.get("host"), Some(b"x".as_slice()));
        assert_eq!(consumed, 24);
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
        content_length(&head.headers)
    }

    #[test]
    fn content_length_values() {
        assert_eq!(cl(b"Content-Length: 5"), Ok(Some(5)));
        assert_eq!(cl(b"Content-Length: 0"), Ok(Some(0)));
        assert_eq!(cl(b"Content-Length: 007"), Ok(Some(7)));
        assert_eq!(
            cl(b"Content-Length: 18446744073709551615"),
            Ok(Some(u64::MAX))
        );
        // Equal values agree: comma list and separate fields.
        assert_eq!(cl(b"Content-Length: 5, 5"), Ok(Some(5)));
        assert_eq!(cl(b"Content-Length: 5\r\nContent-Length: 5"), Ok(Some(5)));
        // Absent entirely.
        assert_eq!(cl(b"Host: x"), Ok(None));
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
        request_body_spec(&head)
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

    #[test]
    fn transfer_encoding_identity_is_no_coding() {
        assert_eq!(te_spec(b"Transfer-Encoding: identity"), Ok(BodySpec::Empty));
        assert_eq!(
            te_spec(b"Transfer-Encoding: identity, identity"),
            Ok(BodySpec::Empty)
        );
    }

    #[test]
    fn transfer_encoding_bad_codings_rejected() {
        // Unknown coding alongside chunked.
        assert_eq!(
            te_spec(b"Transfer-Encoding: gzip, chunked"),
            Err(HttpError::UnsupportedTransferEncoding)
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
    fn transfer_encoding_overrides_content_length() {
        let (head, _) =
            req(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\nContent-Length: 5\r\n\r\n")
                .unwrap();
        assert_eq!(request_body_spec(&head), Ok(BodySpec::Chunked));
    }

    // -- body framing -------------------------------------------------------

    #[test]
    fn request_body_spec_cases() {
        let (head, _) = req(b"GET / HTTP/1.1\r\n\r\n").unwrap();
        assert_eq!(request_body_spec(&head), Ok(BodySpec::Empty));
        let (head, _) = req(b"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\n").unwrap();
        assert_eq!(request_body_spec(&head), Ok(BodySpec::ContentLength(5)));
        let (head, _) = req(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n").unwrap();
        assert_eq!(request_body_spec(&head), Ok(BodySpec::Chunked));
    }

    fn res_spec(status_line: &[u8], method: &[u8]) -> Result<BodySpec, HttpError> {
        let input = [status_line, b"\r\n\r\n"].concat();
        let (head, _) = parse_response_head(&input, &limits()).unwrap();
        response_body_spec(&head, method)
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
                decode_chunked(raw.as_bytes(), &limits()),
                Err(HttpError::ForbiddenTrailer),
                "decode: {name} must be a forbidden trailer"
            );
            let lower = format!("0\r\n{}: v\r\n\r\n", name.to_ascii_lowercase());
            assert_eq!(
                decode_chunked(lower.as_bytes(), &limits()),
                Err(HttpError::ForbiddenTrailer),
                "decode (lowercase): {name} must be a forbidden trailer"
            );
            // Streaming decoder.
            let mut dec = ChunkedDecoder::new(Limits::default());
            assert_eq!(
                dec.feed(raw.as_bytes()),
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
        let msg = decode_chunked(b"0\r\nX-Checksum: abc\r\n\r\n", &limits()).unwrap();
        assert_eq!(msg.trailers.get("x-checksum"), Some(b"abc".as_slice()));
        let mut dec = ChunkedDecoder::new(Limits::default());
        let events = dec.feed(b"0\r\nX-Checksum: abc\r\n\r\n").unwrap();
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
        let msg = decode_chunked(b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n", &limits()).unwrap();
        assert_eq!(msg.chunks, vec![b"hello".as_slice(), b" world".as_slice()]);
        assert!(msg.trailers.is_empty());
        assert_eq!(msg.rest, b"");
    }

    #[test]
    fn decode_chunked_keeps_rest() {
        let msg = decode_chunked(b"2\r\nhi\r\n0\r\n\r\nGET /next HTTP/1.1\r\n", &limits()).unwrap();
        assert_eq!(msg.chunks, vec![b"hi".as_slice()]);
        assert_eq!(msg.rest, b"GET /next HTTP/1.1\r\n");
    }

    #[test]
    fn decode_chunked_trailers() {
        let msg =
            decode_chunked(b"3\r\nfoo\r\n0\r\nX-A: 1\r\nX-B: two\r\n\r\n", &limits()).unwrap();
        assert_eq!(msg.trailers.len(), 2);
        assert_eq!(msg.trailers.get("x-a"), Some(b"1".as_slice()));
        assert_eq!(msg.trailers.get("x-b"), Some(b"two".as_slice()));
    }

    #[test]
    fn decode_chunked_extensions_accepted() {
        let msg = decode_chunked(
            b"5;ext=1;foo=\"bar\\\"baz\";flag\r\nhello\r\n0\r\n\r\n",
            &limits(),
        )
        .unwrap();
        assert_eq!(msg.chunks, vec![b"hello".as_slice()]);
    }

    #[test]
    fn decode_chunked_hex_variants() {
        let msg = decode_chunked(b"A\r\n0123456789\r\n0\r\n\r\n", &limits()).unwrap();
        assert_eq!(msg.chunks, vec![b"0123456789".as_slice()]);
        let msg = decode_chunked(b"005\r\nhello\r\n0\r\n\r\n", &limits()).unwrap();
        assert_eq!(msg.chunks, vec![b"hello".as_slice()]);
    }

    #[test]
    fn decode_chunked_empty_body() {
        let msg = decode_chunked(b"0\r\n\r\n", &limits()).unwrap();
        assert!(msg.chunks.is_empty());
        assert!(msg.trailers.is_empty());
    }

    #[test]
    fn decode_chunked_malformed_rejected() {
        // Bad hex.
        assert_eq!(
            decode_chunked(b"zz\r\nhello\r\n0\r\n\r\n", &limits()),
            Err(HttpError::InvalidChunkSize)
        );
        // Empty size line.
        assert_eq!(
            decode_chunked(b"\r\n0\r\n\r\n", &limits()),
            Err(HttpError::InvalidChunkSize)
        );
        // 17 hex digits: cannot fit in u64.
        assert_eq!(
            decode_chunked(b"10000000000000000\r\n0\r\n\r\n", &limits()),
            Err(HttpError::InvalidChunkSize)
        );
        // Space inside the size.
        assert_eq!(
            decode_chunked(b"5 \r\nhello\r\n0\r\n\r\n", &limits()),
            Err(HttpError::InvalidChunkSize)
        );
        // Bad chunk extension.
        assert_eq!(
            decode_chunked(b"5;ext=\r\nhello\r\n0\r\n\r\n", &limits()),
            Err(HttpError::InvalidChunkExtension)
        );
        assert_eq!(
            decode_chunked(b"5;\r\nhello\r\n0\r\n\r\n", &limits()),
            Err(HttpError::InvalidChunkExtension)
        );
        assert_eq!(
            decode_chunked(b"5;\"x\"\r\nhello\r\n0\r\n\r\n", &limits()),
            Err(HttpError::InvalidChunkExtension)
        );
        // Chunk data not followed by CRLF.
        assert_eq!(
            decode_chunked(b"5\r\nhelloXX0\r\n\r\n", &limits()),
            Err(HttpError::InvalidChunkTerminator)
        );
        // Forbidden trailer fields.
        assert_eq!(
            decode_chunked(b"0\r\nContent-Length: 5\r\n\r\n", &limits()),
            Err(HttpError::ForbiddenTrailer)
        );
        assert_eq!(
            decode_chunked(b"0\r\nTE: trailers\r\n\r\n", &limits()),
            Err(HttpError::ForbiddenTrailer)
        );
        // Bare CR inside the chunked body.
        assert_eq!(
            decode_chunked(b"5\rhello\r\n0\r\n\r\n", &limits()),
            Err(HttpError::BareCr)
        );
    }

    #[test]
    fn decode_chunked_incomplete() {
        assert_eq!(decode_chunked(b"", &limits()), Err(HttpError::Incomplete));
        assert_eq!(
            decode_chunked(b"5\r\nhel", &limits()),
            Err(HttpError::Incomplete)
        );
        assert_eq!(
            decode_chunked(b"5\r\nhello\r\n", &limits()),
            Err(HttpError::Incomplete)
        );
        assert_eq!(
            decode_chunked(b"5\r\nhello\r\n0\r\n", &limits()),
            Err(HttpError::Incomplete)
        );
        assert_eq!(
            decode_chunked(b"5\r\nhello\r\n0\r\nX-A: 1\r\n", &limits()),
            Err(HttpError::Incomplete)
        );
    }

    #[test]
    fn chunked_roundtrip() {
        let chunks: Vec<&[u8]> = vec![b"hello".as_slice(), b", ".as_slice(), b"world".as_slice()];
        let encoded = encode_chunked(&chunks, &[("X-Check", b"ok".as_slice())]).unwrap();
        let msg = decode_chunked(&encoded, &limits()).unwrap();
        assert_eq!(msg.chunks, chunks);
        assert_eq!(msg.trailers.get("x-check"), Some(b"ok".as_slice()));
        assert_eq!(msg.rest, b"");
    }

    // -- streaming decoder -----------------------------------------------------

    fn collect_stream(feeds: &[&[u8]]) -> (Vec<u8>, Vec<TrailerField>, bool) {
        let mut dec = ChunkedDecoder::new(Limits::default());
        let mut data = Vec::new();
        let mut trailers = Vec::new();
        for feed in feeds {
            for ev in dec.feed(feed).unwrap() {
                match ev {
                    ChunkEvent::Data(d) => data.extend_from_slice(d),
                    ChunkEvent::Trailer(t) => trailers.push(t),
                    ChunkEvent::End => {}
                }
            }
        }
        (data, trailers, dec.is_done())
    }

    #[test]
    fn streaming_byte_by_byte() {
        let input = b"5\r\nhello\r\n6\r\n world\r\n0\r\nX-T: 1\r\n\r\n";
        let feeds: Vec<&[u8]> = input.chunks(1).map(|c| c as &[u8]).collect();
        let (data, trailers, done) = collect_stream(&feeds);
        assert_eq!(data, b"hello world");
        assert_eq!(trailers.len(), 1);
        assert_eq!(trailers[0].name, "X-T");
        assert_eq!(trailers[0].value, b"1");
        assert!(done);
    }

    #[test]
    fn streaming_awkward_splits() {
        // Splits inside CRLF pairs, inside hex digits, inside data.
        let feeds: &[&[u8]] = &[
            b"5", b"\r", b"\nhel", b"lo\r", b"\n0\r", b"\n", b"\r", b"\n",
        ];
        let (data, trailers, done) = collect_stream(feeds);
        assert_eq!(data, b"hello");
        assert!(trailers.is_empty());
        assert!(done);
    }

    #[test]
    fn streaming_one_shot_matches_incremental() {
        let input = b"3\r\nfoo\r\n0\r\n\r\n";
        let mut dec = ChunkedDecoder::new(Limits::default());
        let events = dec.feed(input).unwrap();
        assert!(matches!(events[0], ChunkEvent::Data(b"foo")));
        assert!(matches!(events[1], ChunkEvent::End));
        assert!(dec.is_done());
        // Input after End is ignored.
        assert!(dec.feed(b"junk").unwrap().is_empty());
    }

    #[test]
    fn streaming_errors() {
        let mut dec = ChunkedDecoder::new(Limits::default());
        assert_eq!(dec.feed(b"zz\r\n"), Err(HttpError::InvalidChunkSize));

        let mut dec = ChunkedDecoder::new(Limits::default());
        assert!(
            dec.feed(b"5\r\nhello")
                .unwrap()
                .iter()
                .any(|e| matches!(e, ChunkEvent::Data(_)))
        );
        assert_eq!(dec.feed(b"XX"), Err(HttpError::InvalidChunkTerminator));

        let mut dec = ChunkedDecoder::new(Limits::default());
        assert_eq!(
            dec.feed(b"0\r\nContent-Length: 1\r\n\r\n"),
            Err(HttpError::ForbiddenTrailer)
        );
    }

    #[test]
    fn streaming_respects_line_limit() {
        let tiny = Limits {
            max_line_bytes: 4,
            ..Default::default()
        };
        let mut dec = ChunkedDecoder::new(tiny);
        assert_eq!(dec.feed(b"12345\r\n"), Err(HttpError::LineTooLong));
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
    }
}
