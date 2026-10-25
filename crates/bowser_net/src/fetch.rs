//! WHATWG fetch (basic, no streams) over the hand-rolled HTTP/1.1 and
//! TLS 1.3 stack.
//!
//! Standard-library Rust only. See /AGENTS.md.
//!
//! What this implements:
//!
//! * `http` via plain TCP and `https` via [`crate::tls::TlsConnection`],
//!   speaking HTTP/1.1 through the [`crate::http1`] codec. Anything else
//!   is [`FetchError::UnsupportedScheme`].
//! * Redirects: 301/302/303/307/308 with WHATWG URL resolution of
//!   `Location`, at most 20 hops. 303 always rewrites the request to GET
//!   and drops the body; 301/302 do the same for any non-GET/HEAD request,
//!   per WHATWG fetch. 307/308 preserve the method and body.
//! * Redirect chains: [`fetch_with_trace`] reports every URL actually
//!   fetched in order ([`FetchTrace`], initial URL plus each redirect
//!   target); [`fetch`] and [`fetch_with_jar`] follow the same chain and
//!   discard the trace.
//! * Credentials in the URL (`user:pass@host`) are never sent on the wire:
//!   the request-target, the `Host` header, and the dial address are all
//!   built without userinfo (RFC 9110 §4.2.4).
//! * Interim 1xx responses (e.g. 103 Early Hints) are skipped; reading
//!   continues until a final status. At most 8 interim responses are
//!   tolerated before the exchange fails.
//! * Timeouts: each TCP connection attempt gets 30 s
//!   ([`CONNECT_TIMEOUT`]), but resolved addresses race Happy-Eyeballs
//!   style: a new attempt starts every 250 ms ([`EYEBALL_DELAY`]) and the
//!   first success wins, so a dead first address cannot stall fallback
//!   behind the full per-attempt budget. At most eight addresses race at
//!   once ([`MAX_RACE_ATTEMPTS`]); further addresses are tried as earlier
//!   attempts fail, so a huge DNS answer cannot spawn hundreds of
//!   threads. 60 s per socket read/write ([`SOCKET_TIMEOUT`]); either
//!   surfaces as [`FetchError::Timeout`]. The timeouts are set on the
//!   underlying `TcpStream`, so they cover the TLS handshake and record
//!   layer too. On top of the per-read budget, each request/response
//!   exchange has an absolute wall-clock deadline
//!   ([`TOTAL_FETCH_TIMEOUT`]) from request-send to body-complete, so a
//!   peer drip-feeding bytes can never hold a fetch open past it; it
//!   surfaces as [`FetchError::Timeout`] too.
//! * Cookie jar integration: matching cookies are sent, `Set-Cookie` is
//!   stored. The request is treated as a top-level navigation for
//!   `SameSite` purposes by default; a caller fetching subresources should
//!   use [`fetch_with_jar_in_context`] (or [`fetch_with_trace_in_context`])
//!   with the document's host as the top-level context, so `Lax` cookies
//!   are not over-sent on cross-site redirect chains.
//! * Body decoding: [`Response::text`] decodes per the `Content-Type`
//!   charset via `bowser_encoding` (UTF-8 default, BOM wins). Chunked
//!   bodies are decoded incrementally and the 32 MiB cap is enforced on
//!   decoded bytes as they arrive; chunked trailers are appended to
//!   [`Response::headers`].
//!
//! Deliberate limitations, documented so callers are not surprised:
//!
//! * Keep-alive connection pooling: connections are reused across
//!   redirect hops and across standalone `fetch` calls, keyed by
//!   `(scheme, host, port)` through one process-wide pool. A connection
//!   is pooled only when the response allows persistence (no
//!   `Connection: close`, HTTP/1.1 or `keep-alive` 1.0) *and* the
//!   response framing consumed exactly the message, so the next
//!   response starts at a clean boundary. Close-delimited bodies are
//!   never pooled. A pooled connection that dies before delivering a
//!   response is retried once on a fresh connection for safe methods
//!   only; non-idempotent request bodies are never replayed.
//! * TLS session resumption: post-handshake session tickets are cached
//!   per origin and offered on later connections, so repeat visits can
//!   skip the certificate flight (TLS 1.3 abbreviated handshake). A
//!   rejected ticket falls back to a full handshake transparently.
//! * TLS is authenticated: the server certificate chain is validated
//!   against the vendored Mozilla root store (see [`crate::crypto::x509`]),
//!   the hostname is checked per RFC 6125, and the `CertificateVerify`
//!   signature is verified before the handshake completes. A chain that
//!   does not verify fails closed with [`FetchError::Tls`]. Plain `http`
//!   works end to end.
//! * No request or response streaming: bodies are buffered, capped at
//!   32 MiB ([`FetchError::BodyTooLarge`]).
//! * `Authorization` is stripped when a redirect leaves the current
//!   origin; cookies are re-evaluated from the jar per hop.
//! * HSTS (RFC 6797): `Strict-Transport-Security` on `https` responses is
//!   recorded in a caller-owned [`HstsStore`], and `http` URLs whose host
//!   (or a parent domain with `includeSubDomains`) has a live policy are
//!   upgraded to `https` before fetching. Expired policies are evicted
//!   and never honored.
//! * Downgrade protection: a redirect from an `https` URL to an `http`
//!   URL is not followed — it fails closed with
//!   [`FetchError::InsecureRedirect`]. A redirect target naming an
//!   HSTS-known host is upgraded to `https` first, so only a genuine
//!   plaintext downgrade trips the block.
//! * HTTP proxy support ([`ProxyConfig`]): `http` URLs are fetched through
//!   the proxy with absolute-form request-targets; `https` URLs tunnel
//!   via `CONNECT`, with the TLS handshake running over the established
//!   tunnel through the existing TLS client. Proxy credentials come from
//!   the proxy URL's userinfo (`Proxy-Authorization: Basic`);
//!   [`ProxyConfig::no_proxy`] bypasses the proxy per target host.
//!   Proxied connections are never pooled: a pooled entry is keyed by
//!   origin, but a proxied socket is bound to a (proxy, origin) pair.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::io;
use std::net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bowser_encoding::{Encoding, decode, encoding_from_label};
use bowser_url::{ParseError, Url, parse as parse_url, percent_decode};

use crate::cookies::{CookieJar, RequestContext};
use crate::decode::{DecodeChain, DecodeError};
use crate::http1::{self, BodySpec, HttpError, HttpVersion, Limits, is_token};
use crate::pool::{Conn, OpenRequestError, Pool, PoolKey, SharedStream, global_pool};
use crate::tls::{PskOffer, TlsConfig, TlsConnection};

/// Default `User-Agent` sent when the caller does not supply one.
pub const USER_AGENT: &str = "Bowser/0.1";

/// Maximum redirects followed before [`FetchError::TooManyRedirects`].
pub const MAX_REDIRECTS: usize = 20;

/// Maximum interim 1xx responses skipped while waiting for a final status.
/// Bounds the head loop so a peer streaming endless 1xx cannot spin it
/// without doing I/O (socket read timeouts only fire when a read blocks).
const MAX_INTERIM_RESPONSES: usize = 8;

/// How long a fetch waits for the TCP connection to establish before
/// failing with [`FetchError::Timeout`].
///
/// This is a per-attempt budget, not a total: resolved addresses race
/// (see [`EYEBALL_DELAY`]), so this only fires when every address stalls
/// or fails.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Stagger between connection-attempt starts when racing resolved
/// addresses ([`race_connect`]). 250 ms is the RFC 8305 recommended
/// connection-attempt delay: long enough to prefer the first (usually
/// IPv6) address when it is healthy, short enough that a dead first
/// address does not stall fallback.
pub const EYEBALL_DELAY: Duration = Duration::from_millis(250);

/// Maximum Happy Eyeballs racer threads. A large DNS answer must not
/// spawn hundreds of threads: the first this many addresses race with
/// [`EYEBALL_DELAY`] stagger, and the rest are claimed by racers as
/// earlier attempts fail, so every address is still tried. Eight is
/// generous — answers past the first few addresses are already deep
/// fallback — and keeps thread churn bounded on hostile or gigantic
/// answers.
pub const MAX_RACE_ATTEMPTS: usize = 8;

/// How long a fetch waits for a single socket read or write before failing
/// with [`FetchError::Timeout`]. Set on the underlying `TcpStream`, so it
/// covers both plain HTTP and the TLS handshake/record layer.
pub const SOCKET_TIMEOUT: Duration = Duration::from_secs(60);

/// Absolute wall-clock bound on one request/response exchange, measured
/// from the moment the request is fully sent until the body completes.
/// Per-read timeouts reset on every byte received, so a malicious or
/// broken peer can drip-feed one byte at a time and keep a fetch alive
/// indefinitely; this deadline fires regardless of read progress and
/// surfaces as [`FetchError::Timeout`].
///
/// It is deliberately generous (10 minutes) so normal downloads are
/// unaffected: even a full 32 MiB ([`MAX_BODY_BYTES`]) body at a sluggish
/// 100 KiB/s finishes in under six minutes, and known-`Content-Length`
/// bodies get an additional loose per-size extension (see
/// `fetch_body_deadline`). Chunked and close-delimited bodies of unknown
/// size get only this fixed bound.
///
/// The deadline is per redirect hop, not per `fetch` call: each hop sends
/// a new request and gets its own budget.
pub const TOTAL_FETCH_TIMEOUT: Duration = Duration::from_secs(600);

/// Maximum response body accepted, in bytes. Larger `Content-Length`
/// values are rejected before reading; chunked and close-delimited bodies
/// are rejected once they exceed the cap.
pub const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// An outgoing fetch request.
#[derive(Debug, Clone)]
pub struct Request {
    /// HTTP method, e.g. `"GET"`. Known methods are uppercased per the
    /// fetch standard; anything else is sent verbatim (it must be a valid
    /// token). `"CONNECT"` is rejected: it is a forbidden method.
    pub method: String,
    /// Request URL. Only `http` and `https` are supported.
    pub url: Url,
    /// Extra header fields. `Host`, `Content-Length`, `Transfer-Encoding`,
    /// `Cookie`, and `Connection` are managed by fetch and ignored here;
    /// `User-Agent` overrides the default only.
    pub headers: Vec<(String, String)>,
    /// Request body. `Some` (even empty) sends `Content-Length`.
    pub body: Option<Vec<u8>>,
}

/// A fetched response.
#[derive(Debug, Clone)]
pub struct Response {
    /// HTTP status code.
    pub status: u16,
    /// Response header fields, in received order, names as sent. Chunked
    /// trailers, when present, are appended after the head fields.
    pub headers: Vec<(String, String)>,
    /// Raw response body bytes.
    pub body: Vec<u8>,
    /// Final URL after following redirects.
    pub url: Url,
}

impl Response {
    /// Decode the body to text per the `Content-Type` charset parameter,
    /// defaulting to UTF-8. A byte-order mark overrides the charset, per
    /// the Encoding Standard. Undecodable bytes become U+FFFD; charsets
    /// this build cannot decode fall back to lossy UTF-8.
    pub fn text(&self) -> String {
        let mut fallback = Encoding::Utf8;
        if let Some(ct) = self.header("content-type")
            && let Some(label) = charset_from_content_type(ct)
            && let Some(enc) = encoding_from_label(label)
        {
            fallback = enc;
        }
        match decode(&self.body, fallback) {
            Ok(result) => result.text,
            Err(_) => String::from_utf8_lossy(&self.body).into_owned(),
        }
    }

    /// First header value for `name` (case-insensitive), if any.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// The redirect chain a fetch followed: every URL actually fetched, in
/// order, starting with the initial request URL and then each redirect
/// target. A fetch with no redirects reports exactly one hop.
#[derive(Debug, Clone)]
pub struct FetchTrace {
    pub hops: Vec<Url>,
}

/// Every way a fetch can fail.
#[derive(Debug)]
pub enum FetchError {
    /// Redirect `Location` (or the request URL's authority) failed to parse.
    Url(ParseError),
    /// Transport I/O failed: DNS, connect, read, or a truncated body.
    Io(io::Error),
    /// The TLS handshake or record layer failed: malformed messages, an
    /// untrusted or expired certificate chain, a hostname mismatch, or a
    /// bad `CertificateVerify` signature. Certificate problems fail closed
    /// with [`crate::tls::TlsError::BadCertificate`] or
    /// [`crate::tls::TlsError::CertificateVerificationFailed`].
    Tls(crate::tls::TlsError),
    /// The HTTP/1.1 exchange failed: malformed head, bad framing, etc.
    Http(HttpError),
    /// The HTTP/2 exchange failed: framing, HPACK, flow-control, or a
    /// protocol violation. Carries a human-readable reason.
    Http2(String),
    /// More than [`MAX_REDIRECTS`] redirects.
    TooManyRedirects,
    /// A 3xx redirect had no usable `Location` header.
    RedirectLocationMissing,
    /// The request could not be built: a forbidden method (`CONNECT`),
    /// or a caller header with an invalid name or value.
    InvalidHeader,
    /// The server answered `101 Switching Protocols`. This client never
    /// sends `Upgrade`, so a 101 means the server is switching to a
    /// protocol it does not speak; the bytes after the head are not HTTP
    /// and must not be parsed as a response.
    SwitchingProtocols,
    /// The response body exceeded [`MAX_BODY_BYTES`].
    BodyTooLarge,
    /// The URL scheme is not `http` or `https`; carries the scheme.
    UnsupportedScheme(String),
    /// A redirect would have downgraded the connection from `https` to
    /// `http`, so it was not followed.
    ///
    /// Security rationale: an https-to-http redirect hands every later
    /// byte of the exchange — cookies, `Authorization` headers, request
    /// and response bodies — to the network in cleartext. A network
    /// attacker who can inject or rewrite a redirect (SSL stripping)
    /// depends on the client obediently following the downgrade; failing
    /// closed removes that primitive. This is stricter than historical
    /// browser behavior and matches the HSTS threat model (RFC 6797):
    /// once a host is known-secure, plaintext is never acceptable.
    /// Redirect targets naming an HSTS-known host are upgraded to
    /// `https` before this check runs, so only a genuine plaintext
    /// downgrade trips it. Carries the serialized redirect source and
    /// target (strings, not URLs, to keep the error type small).
    InsecureRedirect { from: String, to: String },
    /// Using the configured proxy failed: the proxy URL is not usable
    /// (a non-`http` scheme, a missing host), the proxy refused the
    /// request (e.g. `407` to a `CONNECT`), or the tunnel could not be
    /// established. Carries a human-readable reason.
    Proxy(String),
    /// A connect, read, or write exceeded its timeout
    /// ([`CONNECT_TIMEOUT`], [`SOCKET_TIMEOUT`]); carries the underlying
    /// I/O error.
    Timeout(io::Error),
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Url(e) => write!(f, "fetch URL error: {e}"),
            Self::Io(e) => write!(f, "fetch I/O error: {e}"),
            Self::Tls(e) => write!(f, "fetch TLS error: {e}"),
            Self::Http(e) => write!(f, "fetch HTTP error: {e}"),
            Self::Http2(e) => write!(f, "fetch HTTP/2 error: {e}"),
            Self::TooManyRedirects => {
                write!(f, "too many redirects (>{MAX_REDIRECTS})")
            }
            Self::RedirectLocationMissing => {
                f.write_str("redirect response has no usable Location header")
            }
            Self::InvalidHeader => f.write_str("invalid request header or method"),
            Self::SwitchingProtocols => f.write_str("unexpected 101 Switching Protocols"),
            Self::BodyTooLarge => {
                write!(f, "response body exceeds {MAX_BODY_BYTES} bytes")
            }
            Self::UnsupportedScheme(s) => write!(f, "unsupported URL scheme: {s}"),
            Self::InsecureRedirect { from, to } => write!(
                f,
                "blocked insecure redirect from {from} to {to}: https downgraded to http"
            ),
            Self::Proxy(reason) => write!(f, "proxy error: {reason}"),
            Self::Timeout(e) => write!(f, "fetch timed out: {e}"),
        }
    }
}

impl std::error::Error for FetchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Url(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::Tls(e) => Some(e),
            Self::Http(e) => Some(e),
            Self::Timeout(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for FetchError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ParseError> for FetchError {
    fn from(e: ParseError) -> Self {
        Self::Url(e)
    }
}

impl From<crate::tls::TlsError> for FetchError {
    fn from(e: crate::tls::TlsError) -> Self {
        Self::Tls(e)
    }
}

/// Whether two URLs are same-origin: equal [`Url::origin`] tuples.
pub fn is_same_origin(a: &Url, b: &Url) -> bool {
    a.origin() == b.origin()
}

/// `SameSite` navigation context for one fetch's cookie handling.
///
/// The default (all fields unset) treats the fetch as a top-level
/// navigation: every redirect hop uses the hop's own host as the top-level
/// host, and `top_level_safe_navigation` follows the request method. That
/// is what document loads want.
///
/// Subresource fetches must pass the document's host instead, with
/// `top_level_safe_navigation: false`: otherwise each hop looks like its
/// own top-level navigation and `Lax` cookies ride along cross-site
/// redirect chains that a real browser would treat as subresource
/// requests.
#[derive(Debug, Clone, Default)]
pub struct NavigationContext {
    /// Host of the top-level document, for `SameSite` decisions. `None`
    /// means "this fetch is itself the top-level navigation".
    pub top_host: Option<String>,
    /// Whether the top-level document was loaded over a secure
    /// transport, for schemeful same-site decisions. `None` (unknown)
    /// is treated as cross-site.
    pub top_secure: Option<bool>,
    /// True only for a top-level navigation with a safe method; lets `Lax`
    /// cookies ride cross-site top-level navigations. Must be `false` for
    /// subresource fetches.
    pub top_level_safe_navigation: bool,
}

// ---------------------------------------------------------------------------
// HTTP proxy support
// ---------------------------------------------------------------------------

/// How one fetch reaches the network through an HTTP proxy.
///
/// * `http` target URLs are sent to the proxy with absolute-form
///   request-targets (`GET http://host/path HTTP/1.1`); the proxy
///   forwards them.
/// * `https` target URLs tunnel: fetch sends `CONNECT host:port HTTP/1.1`
///   to the proxy and, once the proxy answers 2xx, runs the TLS
///   handshake to the origin over the established tunnel through the
///   existing TLS client.
/// * Credentials in the proxy URL's userinfo (`http://user:pass@proxy/`)
///   are sent preemptively as `Proxy-Authorization: Basic`; the userinfo
///   is percent-decoded before encoding. A caller-supplied
///   `Proxy-Authorization` header is dropped when the config carries
///   credentials (the config is authoritative, like the other managed
///   headers).
/// * [`ProxyConfig::no_proxy`] bypasses the proxy per target host.
///
/// Only `http` proxy URLs are supported: the proxy itself is always
/// contacted over plaintext. The default proxy port is 8080.
///
/// Deliberate limitations: proxied connections are never pooled (a
/// pooled entry is keyed by origin, but a proxied socket is bound to a
/// (proxy, origin) pair), so each redirect hop through a proxy dials
/// fresh, and TLS session tickets are neither offered nor harvested for
/// proxied connections. There is no reactive `407` handling: a proxy
/// that answers `407` anyway fails the fetch with [`FetchError::Proxy`].
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Proxy URL. The scheme must be `http`; the host and optional port
    /// name the proxy; optional `user:password` userinfo becomes
    /// `Proxy-Authorization: Basic`.
    pub url: Url,
    /// `NO_PROXY` bypass entries: hostnames matched case-insensitively,
    /// exactly or as a parent domain (`example.com` also bypasses
    /// `a.example.com`), optionally with `:port` to bypass only that
    /// port; `*` bypasses everything. See [`ProxyConfig::bypasses`].
    pub no_proxy: Vec<String>,
}

impl ProxyConfig {
    /// A config for `url` with an empty bypass list.
    pub fn new(url: Url) -> Self {
        Self {
            url,
            no_proxy: Vec::new(),
        }
    }

    /// Build from the standard environment: `HTTP_PROXY` (falling back
    /// to `http_proxy`) for the proxy URL, `NO_PROXY` (falling back to
    /// `no_proxy`, comma-separated) for the bypass list. A schemeless
    /// value like `proxy:8080` is read as `http://proxy:8080`. Returns
    /// `None` when no proxy URL is set or it fails to parse.
    pub fn from_env() -> Option<Self> {
        Self::from_env_with(|key| std::env::var(key).ok())
    }

    /// [`ProxyConfig::from_env`] with the environment lookup injected,
    /// so the parsing rules are unit-testable without touching process
    /// state.
    fn from_env_with(get: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let raw = get("HTTP_PROXY").or_else(|| get("http_proxy"))?;
        let raw = raw.trim();
        if raw.is_empty() {
            return None;
        }
        // A schemeless value like `proxy:8080` is not a valid URL on its
        // own (it would parse as scheme `proxy`); read it as http.
        let url = if raw.contains("://") {
            parse_url(raw, None).ok()?
        } else {
            parse_url(&format!("http://{raw}"), None).ok()?
        };
        let no_proxy = get("NO_PROXY")
            .or_else(|| get("no_proxy"))
            .map(|list| {
                list.split(',')
                    .map(|entry| entry.trim().to_owned())
                    .filter(|entry| !entry.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        Some(Self { url, no_proxy })
    }

    /// Whether the proxy is bypassed for `host:port`. `host` is the
    /// serialized target host (`[::1]` for IPv6); matching is
    /// case-insensitive.
    pub fn bypasses(&self, host: &str, port: u16) -> bool {
        self.no_proxy
            .iter()
            .any(|entry| no_proxy_match(entry, host, port))
    }

    /// `(host, port)` of the proxy itself. Fails with
    /// [`FetchError::Proxy`] for a non-`http` scheme or a missing host.
    fn proxy_authority(&self) -> Result<(&str, u16), FetchError> {
        if self.url.scheme() != "http" {
            return Err(FetchError::Proxy(format!(
                "unsupported proxy scheme {:?}: only http proxies are supported",
                self.url.scheme()
            )));
        }
        let host = self
            .url
            .host()
            .ok_or_else(|| FetchError::Proxy("proxy URL has no host".to_owned()))?;
        Ok((host, self.url.port().unwrap_or(8080)))
    }

    /// The `Proxy-Authorization` header value from the proxy URL's
    /// userinfo: `Basic` over base64 of the percent-decoded
    /// `user:password`. `None` when the proxy URL carries no userinfo.
    fn proxy_authorization_value(&self) -> Option<String> {
        let user = self.url.username();
        let password = self.url.password();
        if user.is_empty() && password.is_empty() {
            return None;
        }
        let mut credentials = percent_decode(user);
        credentials.push(b':');
        credentials.extend_from_slice(&percent_decode(password));
        Some(format!("Basic {}", base64_encode(&credentials)))
    }
}

/// Strip one pair of IPv6 brackets: `[::1]` becomes `::1`. Anything else
/// is returned unchanged.
fn strip_brackets(host: &str) -> &str {
    host.strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host)
}

/// Split a lowercased `NO_PROXY` entry into (host, port). `[v6]` and
/// `[v6]:port` split at the bracket; otherwise the last colon splits
/// `host:port` only when it is the *only* colon, so a bare IPv6 literal
/// like `::1` is kept whole. Returns `None` for malformed entries (a
/// colon with a non-numeric port): a malformed entry never matches, so
/// the proxy stays in use (fail closed).
fn split_no_proxy_entry(entry: &str) -> Option<(&str, Option<u16>)> {
    if let Some(inner) = entry.strip_prefix('[') {
        let end = inner.find(']')?;
        let rest = &inner[end + 1..];
        let port = match rest.strip_prefix(':') {
            Some(port) => Some(port.parse::<u16>().ok()?),
            None if rest.is_empty() => None,
            _ => return None,
        };
        // `end` indexes `inner`; add one for the stripped `[`.
        return Some((&entry[..end + 2], port));
    }
    match entry.rsplit_once(':') {
        Some((host, port)) if !host.contains(':') => Some((host, Some(port.parse::<u16>().ok()?))),
        _ => Some((entry, None)),
    }
}

/// One `NO_PROXY` entry against a target `host:port`. A leading dot
/// (`.example.com`) means the same as the bare domain: the domain and
/// its subdomains match, with a label boundary required (so
/// `example.com` does not match `badexample.com`). Comparison is
/// case-insensitive; a trailing dot on either side (FQDN root) is
/// ignored. `host` may be bracketed (`[::1]`) or bare.
fn no_proxy_match(entry: &str, host: &str, port: u16) -> bool {
    let entry = entry.trim().to_ascii_lowercase();
    if entry.is_empty() {
        return false;
    }
    if entry == "*" {
        return true;
    }
    let Some((entry_host, entry_port)) = split_no_proxy_entry(&entry) else {
        return false;
    };
    let entry_host = entry_host.strip_prefix('.').unwrap_or(entry_host);
    let entry_host = strip_brackets(entry_host);
    let entry_host = entry_host.strip_suffix('.').unwrap_or(entry_host);
    if entry_host.is_empty() {
        return false;
    }
    let host = strip_brackets(host).to_ascii_lowercase();
    let host = host.strip_suffix('.').unwrap_or(host.as_str());
    let host_matches = host == entry_host
        || host
            .strip_suffix(entry_host)
            .is_some_and(|rest| rest.ends_with('.'));
    let port_matches = entry_port.is_none_or(|p| p == port);
    host_matches && port_matches
}

/// Standard base64 (RFC 4648 §4), std-only. Used for
/// `Proxy-Authorization: Basic`.
fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let mut n = 0u32;
        for (i, &b) in chunk.iter().enumerate() {
            n |= (b as u32) << (16 - 8 * i);
        }
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Origin-form request-target for a request sent through a `CONNECT`
/// tunnel (`/path?query`, no fragment): the tunnel already names the
/// authority, so the tunneled request looks like a direct one.
fn origin_form_target(url: &Url) -> String {
    match url.query() {
        Some(query) => format!("{}?{query}", url.path()),
        None => url.path().to_owned(),
    }
}

// ---------------------------------------------------------------------------
// HSTS (RFC 6797)
// ---------------------------------------------------------------------------

/// One cached HSTS policy: the host must only be contacted over HTTPS
/// until `expires_at` (unix seconds), optionally including subdomains.
#[derive(Debug, Clone, Copy)]
struct HstsPolicy {
    expires_at: u64,
    include_subdomains: bool,
}

/// In-memory HSTS policy store: host -> expiry (RFC 6797's "HSTS Host"
/// cache).
///
/// The store is caller-owned, like [`CookieJar`]: keep one for the life
/// of the process (or profile) and hand it to the `*_hsts` entry points
/// so a policy learned from one fetch upgrades later fetches. The plain
/// entry points ([`fetch`], [`fetch_with_trace`], ...) use a throwaway
/// store instead: policies learned during one fetch still apply to its
/// own redirect chain, but are not retained afterwards.
///
/// Policies are learned only from `https` responses (RFC 6797 §8.1: a UA
/// MUST NOT honor `Strict-Transport-Security` over an insecure
/// transport) and are keyed by the normalized host. A parent domain's
/// policy covers subdomains only with `includeSubDomains`. Expired
/// entries are evicted lazily on every query and update, so dead
/// policies neither linger nor are honored.
#[derive(Debug, Clone, Default)]
pub struct HstsStore {
    policies: HashMap<String, HstsPolicy>,
}

impl HstsStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop expired policies. Runs on every query and update, so dead
    /// entries never linger and are never honored.
    fn evict_expired(&mut self, now: u64) {
        self.policies.retain(|_, p| p.expires_at > now);
    }

    /// Canonical store key: [`psl::normalize_host`], so `A.COM`,
    /// `a.com`, and `a.com.` share one policy regardless of how the
    /// URL parser normalizes case or trailing dots.
    ///
    /// Returned borrowed when the host is already normalized (the
    /// common case: lowercase, no trailing dot), so hot-path queries
    /// allocate nothing; only genuinely unnormalized hosts pay for a
    /// copy, and only inserts keep one.
    fn key(host: &str) -> Cow<'_, str> {
        crate::psl::normalize_host(host)
    }

    /// Record the HSTS policy from an `https` response's
    /// `Strict-Transport-Security` headers. No-op for non-`https` URLs.
    ///
    /// The first header value that parses wins; `max-age=0` deletes any
    /// existing policy for the host (RFC 6797 §6.1.1). A malformed value
    /// leaves any existing policy untouched: failing to parse must never
    /// weaken (or plant) a policy.
    pub fn note_response(&mut self, url: &Url, headers: &[(String, String)], now: u64) {
        self.evict_expired(now);
        if url.scheme() != "https" {
            return;
        }
        let Some(host) = url.host() else { return };
        let parsed = headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("strict-transport-security"))
            .find_map(|(_, value)| parse_sts(value));
        let Some((max_age, include_subdomains)) = parsed else {
            return;
        };
        let key = Self::key(host);
        if max_age == 0 {
            self.policies.remove(key.as_ref());
        } else {
            self.policies.insert(
                key.into_owned(),
                HstsPolicy {
                    expires_at: now.saturating_add(max_age),
                    include_subdomains,
                },
            );
        }
    }

    /// Whether a live policy covers `host`: an exact entry, or a parent
    /// domain's entry with `includeSubDomains`. `host` must already be
    /// key-normalized.
    fn covers(&self, host: &str) -> bool {
        if self.policies.contains_key(host) {
            return true;
        }
        let mut parent = host;
        while let Some(dot) = parent.find('.') {
            parent = &parent[dot + 1..];
            if self
                .policies
                .get(parent)
                .is_some_and(|p| p.include_subdomains)
            {
                return true;
            }
        }
        false
    }
    /// If `url` is `http` and a live HSTS policy covers its host, return
    /// the `https` version. Returns `None` otherwise — including for
    /// non-`http` URLs and expired policies.
    pub fn upgrade_url(&mut self, url: &Url, now: u64) -> Option<Url> {
        self.evict_expired(now);
        if url.scheme() != "http" {
            return None;
        }
        let host = url.host()?;
        if !self.covers(Self::key(host).as_ref()) {
            return None;
        }
        let mut upgraded = url.clone();
        // `http` -> `https` always re-parses; the only failure mode is a
        // serializer/parser disagreement on an already-parsed URL.
        upgraded.set_protocol("https:").ok()?;
        (upgraded.scheme() == "https").then_some(upgraded)
    }
}

/// The process-shared default [`HstsStore`] backing the fetchers that
/// do not take a caller-owned store: policies learned by one fetch
/// protect later fetches process-wide, instead of dying in a per-call
/// scratch store. Callers that need isolation (tests, separate
/// profiles) pass their own store via `fetch_with_hsts`.
fn default_hsts() -> &'static Mutex<HstsStore> {
    static STORE: OnceLock<Mutex<HstsStore>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HstsStore::new()))
}

/// Parse one `Strict-Transport-Security` header value into `(max-age,
/// include_subdomains)`. Returns `None` when the value is malformed —
/// in particular when no valid `max-age` directive is present.
/// Unknown directives (e.g. `preload`) are ignored per RFC 6797; a
/// repeated `max-age` takes the last value.
fn parse_sts(value: &str) -> Option<(u64, bool)> {
    let mut max_age: Option<u64> = None;
    let mut include_subdomains = false;
    for directive in value.split(';') {
        let directive = directive.trim();
        if directive.is_empty() {
            continue;
        }
        let (name, val) = match directive.split_once('=') {
            Some((n, v)) => (n.trim(), Some(v.trim())),
            None => (directive, None),
        };
        if name.eq_ignore_ascii_case("max-age") {
            // `delta-seconds`: ASCII digits only. Anything else —
            // negative, quoted, empty, overflowing — makes the whole
            // value malformed, so any existing policy is left untouched.
            let digits = val?;
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            // All-digit input only fails to parse on overflow, which is
            // treated as malformed (fail closed: no policy change).
            max_age = Some(digits.parse().ok()?);
        } else if name.eq_ignore_ascii_case("includesubdomains") {
            // The directive takes no value; `includeSubDomains=x` is
            // ignored without poisoning the rest of the header.
            if val.is_none() {
                include_subdomains = true;
            }
        }
        // Unknown directives are ignored.
    }
    max_age.map(|age| (age, include_subdomains))
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Classify a transport I/O error: socket timeouts surface as
/// [`io::ErrorKind::TimedOut`] (e.g. `connect_timeout`) or
/// [`io::ErrorKind::WouldBlock`] (read/write timeouts via
/// `SO_RCVTIMEO`/`SO_SNDTIMEO` on Unix); everything else stays
/// [`FetchError::Io`]. Sockets here are always in blocking mode, so a
/// `WouldBlock` can only mean a timeout fired.
pub(crate) fn map_io_error(e: io::Error) -> FetchError {
    match e.kind() {
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock => FetchError::Timeout(e),
        _ => FetchError::Io(e),
    }
}

/// Timeouts applied to one fetch. The public entry points use the
/// defaults ([`CONNECT_TIMEOUT`], [`SOCKET_TIMEOUT`],
/// [`TOTAL_FETCH_TIMEOUT`]).
#[derive(Debug, Clone, Copy)]
struct Timeouts {
    connect: Duration,
    rw: Duration,
    /// Absolute wall-clock budget for one request/response exchange, from
    /// request-send to body-complete ([`TOTAL_FETCH_TIMEOUT`]). Fires even
    /// when bytes are trickling in, so a drip-feeding peer cannot hold a
    /// fetch open indefinitely.
    fetch_total: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            connect: CONNECT_TIMEOUT,
            rw: SOCKET_TIMEOUT,
            fetch_total: TOTAL_FETCH_TIMEOUT,
        }
    }
}

/// Fetch `req` with no cookie persistence. Redirects share a throwaway
/// jar for the duration of this call only. Equivalent to
/// [`fetch_with_trace`] with the trace discarded.
pub fn fetch(req: &Request) -> Result<Response, FetchError> {
    fetch_with_trace(req, None).map(|(response, _)| response)
}

/// Fetch `req`, sending cookies from `jar` and storing `Set-Cookie`
/// responses into it. Pass `None` for a throwaway jar.
///
/// The fetch runs in the default [`NavigationContext`]: a top-level
/// navigation. Its initial URL's host and scheme anchor the `SameSite`
/// top context for every redirect hop, so `Strict` cookies for a
/// redirect target are withheld on a cross-site navigation, while `Lax`
/// cookies still ride safe top-level navigations. A caller fetching
/// subresources must use [`fetch_with_jar_in_context`] with the
/// document's host instead, or it over-sends `Lax` cookies on such chains.
///
/// Equivalent to [`fetch_with_trace`] with the trace discarded.
pub fn fetch_with_jar(req: &Request, jar: Option<&mut CookieJar>) -> Result<Response, FetchError> {
    fetch_with_jar_in_context(req, jar, &NavigationContext::default())
}

/// Fetch `req` with cookies, as [`fetch_with_jar`], but in the given
/// `SameSite` navigation context. Subresource fetches pass the document's
/// host with `top_level_safe_navigation: false`; the context applies to
/// every redirect hop of this fetch.
///
/// Equivalent to [`fetch_with_trace_in_context`] with the trace discarded.
pub fn fetch_with_jar_in_context(
    req: &Request,
    jar: Option<&mut CookieJar>,
    nav: &NavigationContext,
) -> Result<Response, FetchError> {
    fetch_with_trace_in_context(req, jar, nav).map(|(response, _)| response)
}

/// Fetch `req`, sending cookies from `jar` and storing `Set-Cookie`
/// responses into it (pass `None` for a throwaway jar), and report the
/// redirect chain. The returned [`FetchTrace`] lists every URL actually
/// fetched, in order: the initial request URL followed by each redirect
/// target. A fetch with no redirects reports exactly one hop. SameSite
/// behavior matches [`fetch_with_jar`].
pub fn fetch_with_trace(
    req: &Request,
    jar: Option<&mut CookieJar>,
) -> Result<(Response, FetchTrace), FetchError> {
    fetch_with_trace_in_context(req, jar, &NavigationContext::default())
}

/// Fetch `req` with cookies and report the redirect chain, as
/// [`fetch_with_trace`], but in the given `SameSite` navigation context.
/// Subresource fetches pass the document's host with
/// `top_level_safe_navigation: false`; the context applies to every
/// redirect hop of this fetch.
pub fn fetch_with_trace_in_context(
    req: &Request,
    jar: Option<&mut CookieJar>,
    nav: &NavigationContext,
) -> Result<(Response, FetchTrace), FetchError> {
    fetch_impl(req, jar, Timeouts::default(), nav, None, None)
        .map(|(response, trace, _)| (response, trace))
}

/// Fetch `req` with cookies and report the redirect chain plus
/// whether any hop actually sent a credential: a `Cookie` header
/// from the jar (including a cookie a redirect hop's `Set-Cookie`
/// stored into a scratch jar and a later hop then sent) or an
/// `Authorization` header that survived the per-hop strip (see
/// [`fetch_once`]). Callers that gate on credentials — CORS, which
/// accepts `*` only when no credentials were sent — must use this
/// variant, not infer credentials from the initial URL: a chain's
/// first hop can be credentialless while a later hop is not, and
/// vice versa.
///
/// `FetchTrace` itself carries no such flag: other crates build it
/// with struct literals, so adding a field there would break their
/// construction sites; the flag travels alongside the trace instead.
pub fn fetch_with_trace_credentials(
    req: &Request,
    jar: Option<&mut CookieJar>,
) -> Result<(Response, FetchTrace, bool), FetchError> {
    fetch_with_trace_credentials_in_context(req, jar, &NavigationContext::default())
}

/// [`fetch_with_trace_credentials`], but in the given `SameSite`
/// navigation context. Subresource fetches pass the document's host
/// with `top_level_safe_navigation: false`; the context applies to
/// every redirect hop of this fetch.
pub fn fetch_with_trace_credentials_in_context(
    req: &Request,
    jar: Option<&mut CookieJar>,
    nav: &NavigationContext,
) -> Result<(Response, FetchTrace, bool), FetchError> {
    fetch_impl(req, jar, Timeouts::default(), nav, None, None)
}

/// Fetch `req` with HSTS state in `hsts`: `Strict-Transport-Security`
/// policies are learned from `https` responses, and `http` URLs whose
/// host has a live policy are upgraded to `https` before fetching.
/// Keep one store for the life of the process so policies learned by
/// one fetch protect later ones. Downgrade redirects (`https` -> `http`)
/// fail with [`FetchError::InsecureRedirect`].
///
/// Equivalent to [`fetch_with_trace_hsts`] with the trace discarded.
pub fn fetch_with_hsts(req: &Request, hsts: &mut HstsStore) -> Result<Response, FetchError> {
    fetch_with_trace_hsts(req, None, hsts).map(|(response, _)| response)
}

/// Fetch `req` with cookies and HSTS, and report the redirect chain.
/// `hsts` is the caller-owned [`HstsStore`]; pass `None` for the jar to
/// use a throwaway one. An HSTS-upgraded initial URL appears in the
/// trace as the upgraded URL: it is the URL actually fetched. SameSite
/// behavior matches [`fetch_with_jar`].
pub fn fetch_with_trace_hsts(
    req: &Request,
    jar: Option<&mut CookieJar>,
    hsts: &mut HstsStore,
) -> Result<(Response, FetchTrace), FetchError> {
    fetch_with_trace_hsts_in_context(req, jar, hsts, &NavigationContext::default())
}

/// Fetch `req` with cookies and HSTS and report the redirect chain, as
/// [`fetch_with_trace_hsts`], but in the given `SameSite` navigation
/// context. Subresource fetches pass the document's host with
/// `top_level_safe_navigation: false`; the context applies to every
/// redirect hop of this fetch.
pub fn fetch_with_trace_hsts_in_context(
    req: &Request,
    jar: Option<&mut CookieJar>,
    hsts: &mut HstsStore,
    nav: &NavigationContext,
) -> Result<(Response, FetchTrace), FetchError> {
    fetch_impl(req, jar, Timeouts::default(), nav, Some(hsts), None)
        .map(|(response, trace, _)| (response, trace))
}

/// Fetch `req` through `proxy` (see [`ProxyConfig`]): `http` URLs go to
/// the proxy with absolute-form request-targets, `https` URLs tunnel via
/// `CONNECT`, and hosts in the proxy's `no_proxy` list are fetched
/// direct. Redirects are followed as usual. Equivalent to
/// [`fetch_with_trace_proxy`] with the trace discarded.
pub fn fetch_with_proxy(req: &Request, proxy: &ProxyConfig) -> Result<Response, FetchError> {
    fetch_with_trace_proxy(req, None, proxy).map(|(response, _)| response)
}

/// Fetch `req` through `proxy` with cookies, and report the redirect
/// chain. The proxy applies per redirect hop: each hop re-evaluates
/// `no_proxy` against its own host, while HSTS upgrades and downgrade
/// protection work as usual. SameSite behavior matches
/// [`fetch_with_jar`].
pub fn fetch_with_trace_proxy(
    req: &Request,
    jar: Option<&mut CookieJar>,
    proxy: &ProxyConfig,
) -> Result<(Response, FetchTrace), FetchError> {
    fetch_with_trace_proxy_in_context(req, jar, proxy, &NavigationContext::default())
}

/// Fetch `req` through `proxy` with cookies and report the redirect
/// chain, as [`fetch_with_trace_proxy`], but in the given `SameSite`
/// navigation context. Subresource fetches pass the document's host with
/// `top_level_safe_navigation: false`; the context applies to every
/// redirect hop of this fetch.
pub fn fetch_with_trace_proxy_in_context(
    req: &Request,
    jar: Option<&mut CookieJar>,
    proxy: &ProxyConfig,
    nav: &NavigationContext,
) -> Result<(Response, FetchTrace), FetchError> {
    fetch_impl(req, jar, Timeouts::default(), nav, None, Some(proxy))
        .map(|(response, trace, _)| (response, trace))
}

/// Which HSTS store one fetch consults: a caller-provided store, or the
/// process-shared default.
///
/// The default's mutex is locked only for the duration of a single store
/// operation (upgrade check, policy learn) — never across network I/O. A
/// stalled or abandoned fetch must not serialize every other fetch in the
/// process behind the lock: navigation deadlines abandon their worker
/// thread, and a worker stuck in a slow body read would otherwise hold
/// the global lock until its socket I/O finished.
enum HstsAccess<'a> {
    Owned(&'a mut HstsStore),
    Shared,
}

impl HstsAccess<'_> {
    fn with<R>(&mut self, op: impl FnOnce(&mut HstsStore) -> R) -> R {
        match self {
            HstsAccess::Owned(store) => op(store),
            HstsAccess::Shared => {
                // A poisoned lock is recovered rather than panicking: the
                // store may be stale but is never unsafe to keep using.
                let mut guard = default_hsts()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                op(&mut guard)
            }
        }
    }
}

fn fetch_impl(
    req: &Request,
    jar: Option<&mut CookieJar>,
    timeouts: Timeouts,
    nav: &NavigationContext,
    hsts: Option<&mut HstsStore>,
    proxy: Option<&ProxyConfig>,
) -> Result<(Response, FetchTrace, bool), FetchError> {
    let mut scratch = CookieJar::new();
    let jar: &mut CookieJar = match jar {
        Some(j) => j,
        None => &mut scratch,
    };
    // Without a caller-provided store, learn and honor policies against
    // the process-shared default store, so a Strict-Transport-Security
    // policy one fetch learns protects later fetches; a caller-owned
    // store still wins when given. See [`HstsAccess`]: the default's
    // mutex is locked per store operation, never across network I/O.
    let mut hsts = match hsts {
        Some(h) => HstsAccess::Owned(h),
        None => HstsAccess::Shared,
    };

    let limits = Limits {
        max_body_bytes: MAX_BODY_BYTES,
        ..Limits::default()
    };
    let mut method = normalize_method(&req.method);
    let mut current = req.url.clone();
    let mut body: Option<Vec<u8>> = req.body.clone();
    // `Authorization` is dropped once a redirect leaves the origin, so a
    // credential never leaks cross-origin.
    let mut strip_auth = false;
    // One pool for the whole fetch: redirect hops reuse connections with
    // each other and with every other fetch in the process.
    let pool = global_pool();

    // `0..=MAX_REDIRECTS` attempts admit at most MAX_REDIRECTS redirects.
    // Every hop is recorded in order: the initial URL first, then each
    // redirect target. Error paths drop the partial trace along with the
    // error; the caller only sees the trace on success.
    //
    // The initial navigation's site anchors SameSite for the whole
    // redirect chain: re-rooting the top context at each hop would let
    // Strict cookies for a redirect target attach to a cross-site
    // navigation (e.g. a top-level navigation from A to B must not send
    // B's Strict cookies).
    let mut anchored_nav: Option<NavigationContext> = None;
    let mut trace = FetchTrace { hops: Vec::new() };
    // OR over every hop: one credentialed hop taints the whole chain
    // for CORS purposes, whatever the initial hop did.
    let mut credentials_sent = false;
    for _ in 0..=MAX_REDIRECTS {
        let now = now_secs();
        // HSTS (RFC 6797): upgrade before fetching. The upgraded URL is
        // what actually goes on the wire, so it is what the trace
        // records.
        current = hsts.with(|s| prepare_hop_url(s, &current, now));
        // Anchor once, at the first hop: the top-level document is the
        // URL actually fetched, after HSTS upgrade.
        let nav: &NavigationContext = if nav.top_host.is_none() {
            anchored_nav.get_or_insert_with(|| initial_nav(&current, &method))
        } else {
            nav
        };
        trace.hops.push(current.clone());
        let (response, hop_credentials) = fetch_once(
            &method,
            &current,
            &req.headers,
            body.as_deref(),
            jar,
            strip_auth,
            &limits,
            timeouts,
            nav,
            pool,
            proxy,
        )?;
        credentials_sent |= hop_credentials;
        // Learn HSTS policy from the response before following any
        // redirect: a later hop to the same host upgrades immediately.
        // `note_response` ignores non-https responses itself.
        hsts.with(|s| s.note_response(&current, &response.headers, now));
        if !matches!(response.status, 301 | 302 | 303 | 307 | 308) {
            return Ok((response, trace, credentials_sent));
        }
        let location = response
            .header("location")
            .ok_or(FetchError::RedirectLocationMissing)?;
        let next = hsts.with(|s| redirect_target(s, &current, location, now))?;
        if !is_same_origin(&current, &next) {
            strip_auth = true;
        }
        // WHATWG fetch: 303 always rewrites to GET with no body; 301/302
        // do the same for any non-GET/HEAD request. 307/308 preserve the
        // method and body.
        let rewrite_to_get = response.status == 303
            || (matches!(response.status, 301 | 302) && !matches!(method.as_str(), "GET" | "HEAD"));
        if rewrite_to_get {
            method = "GET".to_string();
            body = None;
        }
        current = next;
    }
    Err(FetchError::TooManyRedirects)
}

/// Apply HSTS upgrade to the URL about to be fetched. The returned URL
/// is what goes on the wire and what the trace records.
fn prepare_hop_url(hsts: &mut HstsStore, url: &Url, now: u64) -> Url {
    hsts.upgrade_url(url, now).unwrap_or_else(|| url.clone())
}

/// Resolve one redirect hop: WHATWG URL resolution of `Location` against
/// `current`, HSTS upgrade of the target, then downgrade protection.
///
/// A target naming an HSTS-known host is upgraded to `https` first: the
/// host previously proved (over TLS) that it must only be contacted
/// securely, so honoring the policy fixes the redirect instead of
/// blocking it. Anything still `http` after that, reached from an
/// `https` URL, is a plaintext downgrade and fails closed with
/// [`FetchError::InsecureRedirect`]: following it would hand cookies,
/// credentials, and bodies to the network in cleartext (SSL stripping).
fn redirect_target(
    hsts: &mut HstsStore,
    current: &Url,
    location: &str,
    now: u64,
) -> Result<Url, FetchError> {
    let mut next = parse_url(location, Some(current)).map_err(FetchError::Url)?;
    if let Some(upgraded) = hsts.upgrade_url(&next, now) {
        next = upgraded;
    }
    if current.scheme() == "https" && next.scheme() == "http" {
        return Err(FetchError::InsecureRedirect {
            from: current.serialize(),
            to: next.serialize(),
        });
    }
    Ok(next)
}

/// Methods the fetch standard normalizes to uppercase.
fn normalize_method(method: &str) -> String {
    const KNOWN: &[&str] = &["DELETE", "GET", "HEAD", "OPTIONS", "POST", "PUT"];
    let upper = method.to_ascii_uppercase();
    if KNOWN.contains(&upper.as_str()) {
        upper
    } else {
        method.to_string()
    }
}

// ---------------------------------------------------------------------------
// One request/response exchange (no redirects)
// ---------------------------------------------------------------------------

/// Scheme, host, port, and TLS flag for a fetchable URL.
fn authority(url: &Url) -> Result<(&str, u16, bool), FetchError> {
    let (default_port, secure) = match url.scheme() {
        "http" => (80u16, false),
        "https" => (443u16, true),
        other => return Err(FetchError::UnsupportedScheme(other.to_owned())),
    };
    let host = url.host().ok_or(ParseError::MissingHost)?;
    Ok((host, url.port().unwrap_or(default_port), secure))
}

/// Happy Eyeballs connection racing (RFC 8305, simplified): attempts
/// to every resolved address start in DNS order, staggered by
/// [`EYEBALL_DELAY`], and the first success wins. A dead IPv6 address no
/// longer stalls IPv4 fallback behind a full [`CONNECT_TIMEOUT`]: the
/// next attempt starts 250 ms later and wins as soon as it connects.
/// Attempts still queued when a winner arrives are abandoned before they
/// dial, so dead addresses do not pile up long connects. Racers already
/// dialing when the winner lands abandon within one poll slice
/// ([`interruptible_dial`]) instead of holding their thread for the
/// full timeout. Every attempt keeps the full `timeout` budget; the last
/// error surfaces if all fail.
///
/// Fast path: a single address needs no race and no interruptible
/// wrapper. The helper thread inside [`interruptible_dial`] exists only
/// so a Happy Eyeballs loser abandons its dial promptly; with one
/// address there is nothing to lose to, and `connect_timeout` already
/// enforces the full budget on the calling thread. This is the common
/// case (numeric IPs, single-A/AAAA answers), and one thread spawn plus
/// a channel plus the 50 ms poll loop is the most expensive part of a
/// localhost dial, so skipping it is the single biggest fetch-latency
/// win available here.
fn race_connect(addrs: &[SocketAddr], timeout: Duration) -> io::Result<TcpStream> {
    if let &[addr] = addrs {
        if timeout.is_zero() {
            // `connect_timeout` with a zero budget is a platform quirk
            // (EINVAL on Linux); the interruptible path reports a clean
            // timeout without dialing, so preserve exactly that.
            return Err(io::Error::new(io::ErrorKind::TimedOut, "connect timed out"));
        }
        return TcpStream::connect_timeout(&addr, timeout);
    }
    race_with(addrs, move |addr, cancel: &AtomicBool| {
        interruptible_connect(&addr, timeout, cancel)
    })
}

/// Poll slice while waiting for the connect helper thread: short enough
/// that a racer that lost the Happy Eyeballs race abandons within a
/// blink, long enough not to spin the polling thread.
const CANCEL_POLL: Duration = Duration::from_millis(50);

/// One real connect attempt with a shared cancel flag.
///
/// std exposes no nonblocking `connect`, so the attempt runs on a helper
/// thread with the full `timeout` budget while the caller polls the
/// cancel flag in short slices. One socket, one SYN: a handshake slower
/// than a poll slice is never restarted, so intercontinental RTTs, mobile
/// networks, and proxies all work — the old slice loop re-dialed every
/// 50 ms and could never complete a slow handshake. Pure std: no raw
/// sockets, no unsafe.
///
/// If the flag fires or the budget expires this returns promptly; the
/// helper thread finishes its own attempt (bounded by `timeout`) and
/// exits, so at most one stray thread lingers per abandoned attempt.
/// The `dial` seam runs the real attempt; tests inject slow or hanging
/// dials to pin the interrupt and timeout behavior deterministically.
fn interruptible_dial(
    timeout: Duration,
    cancel: &AtomicBool,
    dial: impl FnOnce() -> io::Result<TcpStream> + Send + 'static,
) -> io::Result<TcpStream> {
    if cancel.load(Ordering::Relaxed) {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "happy eyeballs race lost",
        ));
    }
    let (tx, rx) = mpsc::channel::<io::Result<TcpStream>>();
    thread::spawn(move || {
        // The receiver is gone exactly when the caller already gave up;
        // dropping that result is correct.
        let _ = tx.send(dial());
    });
    let start = Instant::now();
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "happy eyeballs race lost",
            ));
        }
        let remaining = timeout.saturating_sub(start.elapsed());
        if remaining.is_zero() {
            return Err(io::Error::new(io::ErrorKind::TimedOut, "connect timed out"));
        }
        match rx.recv_timeout(CANCEL_POLL.min(remaining)) {
            Ok(result) => return result,
            // Slice expired without an answer: loop back and re-check the
            // flag; the full `timeout` still applies to the one attempt.
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            // The helper died without reporting. `connect_timeout` cannot
            // do this in practice; surface it instead of hanging.
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(io::Error::new(
                    io::ErrorKind::ConnectionAborted,
                    "connect helper thread died",
                ));
            }
        }
    }
}

/// Interruptible TCP connect: one [`interruptible_dial`] attempt running
/// `TcpStream::connect_timeout` with the whole budget.
fn interruptible_connect(
    addr: &SocketAddr,
    timeout: Duration,
    cancel: &AtomicBool,
) -> io::Result<TcpStream> {
    let addr = *addr;
    interruptible_dial(timeout, cancel, move || {
        TcpStream::connect_timeout(&addr, timeout)
    })
}

/// Test seam for [`race_connect`]: race the injected `dial` against every
/// address with [`EYEBALL_DELAY`] stagger between attempt starts. The
/// first success wins; if every attempt fails, the last error is
/// returned. An empty address list fails with `AddrNotAvailable`.
///
/// At most [`MAX_RACE_ATTEMPTS`] racer threads ever exist: the first
/// that many addresses start staggered, and the rest are claimed by
/// racers as earlier attempts fail, so every address is still tried but
/// a gigantic DNS answer cannot spawn hundreds of threads.
///
/// `dial` receives a shared cancel flag next to the address: it must
/// poll the flag and abandon promptly when set, because a losing racer
/// cannot otherwise be reclaimed — its thread would sit inside the dial
/// for the full timeout. The production dial ([`interruptible_dial`])
/// polls the flag while a single attempt runs on a helper thread.
///
/// Fast path: a single address needs no race. It is dialed directly on
/// the calling thread — no thread spawn, no channel, no stagger sleep.
/// This is the common case (numeric IPs, single-A/AAAA names), and thread
/// startup plus scheduling is the most expensive part of connecting, so
/// skipping it is the single biggest fetch-latency win available here.
/// Timeout semantics are unchanged: the caller's `dial` still enforces
/// the full per-attempt budget.
fn race_with(
    addrs: &[SocketAddr],
    dial: impl Fn(SocketAddr, &AtomicBool) -> io::Result<TcpStream> + Send + Sync + 'static,
) -> io::Result<TcpStream> {
    if let &[addr] = addrs {
        return dial(addr, &AtomicBool::new(false));
    }
    let addrs: std::sync::Arc<[SocketAddr]> = addrs.into();
    let dial = std::sync::Arc::new(dial);
    let done = std::sync::Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<io::Result<TcpStream>>();
    // Racers beyond the staggered wave claim queued addresses as earlier
    // attempts fail; every address is tried exactly once.
    let next = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(
        addrs.len().min(MAX_RACE_ATTEMPTS),
    ));
    let workers = addrs.len().min(MAX_RACE_ATTEMPTS);
    for w in 0..workers {
        let (tx, dial, done, next, addrs) = (
            tx.clone(),
            dial.clone(),
            done.clone(),
            next.clone(),
            addrs.clone(),
        );
        thread::spawn(move || {
            thread::sleep(EYEBALL_DELAY * w as u32);
            // Worker `w` owns address `w` first, preserving DNS order and
            // the stagger schedule; afterwards it steals queued addresses
            // as earlier attempts fail.
            let mut idx = w;
            loop {
                // A winner may have arrived while this attempt slept or
                // dialed; skip the dial so abandoned attempts never open
                // sockets.
                if done.load(Ordering::Relaxed) {
                    return;
                }
                let Some(&addr) = addrs.get(idx) else {
                    return;
                };
                let result = dial(addr, &done);
                let won = result.is_ok();
                if won {
                    done.store(true, Ordering::Relaxed);
                }
                // The receiver is gone exactly when a winner already
                // returned; dropping that result is correct.
                let _ = tx.send(result);
                if won || done.load(Ordering::Relaxed) {
                    return;
                }
                idx = next.fetch_add(1, Ordering::SeqCst);
            }
        });
    }
    drop(tx);
    let mut pending = addrs.len();
    let mut last_err = None;
    while pending > 0 {
        let Ok(result) = rx.recv() else { break };
        pending -= 1;
        match result {
            Ok(stream) => return Ok(stream),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| {
        io::Error::new(
            io::ErrorKind::AddrNotAvailable,
            "name resolved to no addresses",
        )
    }))
}

fn connect(url: &Url, timeouts: Timeouts, pool: &Pool, key: &PoolKey) -> Result<Conn, FetchError> {
    let (host, port, secure) = authority(url)?;
    let stream = dial_host(host, port, timeouts)?;
    if !secure {
        return Ok(Conn::Plain(stream));
    }
    // Offer a cached session ticket when resuming: the server may accept
    // it and skip the certificate flight (TLS 1.3 abbreviated handshake).
    // A rejection is not an error — the handshake just falls back to full.
    tls_over(stream, host, pool.ticket(key))
}

/// Apply the request's port to cached bare IPs. The DNS cache stores
/// port-free [`IpAddr`]s keyed by host, so one cached answer serves every
/// scheme: the port is a property of the request, not of the answer.
fn sock_addrs(ips: &[IpAddr], port: u16) -> Vec<SocketAddr> {
    ips.iter().map(|ip| SocketAddr::new(*ip, port)).collect()
}

/// TCP dial with Happy-Eyeballs racing ([`race_connect`]) and socket
/// timeouts. Shared by direct connects and proxy dials. DNS answers are
/// resolved through the process-wide [`crate::dns`] cache, with the
/// lookup itself bounded by [`crate::dns::DNS_TIMEOUT`].
fn dial_host(host: &str, port: u16, timeouts: Timeouts) -> Result<TcpStream, FetchError> {
    // `host` is already serialized (`[::1]` for IPv6), so `host:port`
    // resolves as a socket address without ambiguity. The cache holds
    // bare IPs; the request's port is applied at dial time, so an answer
    // cached for `http` can never leak port 80 into an `https` dial (or
    // vice versa).
    let ips = crate::dns::global_dns_cache()
        .resolve(host, move |h| {
            format!("{h}:{port}")
                .to_socket_addrs()
                .map(|it| it.map(|sa| sa.ip()).collect())
        })
        .map_err(map_io_error)?;
    // `resolve` returns the cached answer in a shared `Arc`: on a cache
    // hit no address list is reallocated, only the refcount is bumped.
    let addrs = sock_addrs(&ips, port);
    let stream = race_connect(&addrs, timeouts.connect).map_err(map_io_error)?;
    // Bound every later read and write on this socket, including the TLS
    // handshake below: the same `TcpStream` carries the TLS records, and
    // `tls.rs` is out of scope for this change, so configuring the socket
    // up front is how the timeouts reach the TLS path.
    stream
        .set_read_timeout(Some(timeouts.rw))
        .map_err(map_io_error)?;
    stream
        .set_write_timeout(Some(timeouts.rw))
        .map_err(map_io_error)?;
    Ok(stream)
}

/// ALPN protocols offered on every TLS handshake, in preference order.
/// `"h2"` first: servers that speak HTTP/2 negotiate it and the fetch
/// runs the HTTP/2 exchange; anything else (including no ALPN at all)
/// falls back to the HTTP/1.1 path byte-for-byte as before.
fn tls_alpn_protos() -> Vec<String> {
    vec!["h2".to_owned(), "http/1.1".to_owned()]
}

/// Run the TLS handshake for `host` over an established TCP `stream`.
/// `psk` offers a cached session ticket for resumption; `None` does a
/// full handshake.
fn tls_over(stream: TcpStream, host: &str, psk: Option<PskOffer>) -> Result<Conn, FetchError> {
    tls_over_with_prefix(stream, Vec::new(), host, psk)
}

/// [`tls_over`], with bytes queued ahead of the socket: for CONNECT
/// tunnels the proxy may coalesce the 2xx response with the origin's
/// TLS flight, and `read_head` over-reads those bytes past the response
/// head. Feeding them to the handshake instead of dropping them keeps a
/// coalesced ServerHello from desyncing it.
fn tls_over_with_prefix(
    stream: TcpStream,
    prefix: Vec<u8>,
    host: &str,
    psk: Option<PskOffer>,
) -> Result<Conn, FetchError> {
    // SNI takes the bare IP literal: the handshake omits the extension
    // for IP literals itself (`server_name.parse::<IpAddr>()`).
    let server_name = strip_brackets(host).to_owned();
    let config = TlsConfig {
        server_name,
        alpn: tls_alpn_protos(),
        fingerprint_profile: None,
        psk,
    };
    TlsConnection::connect_with_prefix(stream, prefix, &config)
        .map_err(FetchError::Tls)
        .map(|c| Conn::Tls(Box::new(c)))
}

/// Open a connection to `url` through `proxy`: dial the proxy, and for
/// `https` targets establish a `CONNECT` tunnel and run the TLS
/// handshake to the origin over it via the existing TLS client. `http`
/// targets are spoken to the proxy directly — the request head already
/// uses the absolute-form target the proxy forwards.
fn proxy_connect(
    url: &Url,
    proxy: &ProxyConfig,
    timeouts: Timeouts,
    limits: &Limits,
    pool: &Pool,
    key: &PoolKey,
) -> Result<Conn, FetchError> {
    let (proxy_host, proxy_port) = proxy.proxy_authority()?;
    let stream = dial_host(proxy_host, proxy_port, timeouts)?;
    let (host, port, secure) = authority(url)?;
    if !secure {
        return Ok(Conn::Plain(stream));
    }
    let mut conn = Conn::Plain(stream);
    // Authority-form CONNECT target; the Host header names the same
    // authority (RFC 9110 §9.3.6).
    let target = format!("{host}:{port}");
    let mut headers: Vec<(&str, &[u8])> = Vec::with_capacity(2);
    headers.push(("Host", target.as_bytes()));
    let auth;
    if let Some(value) = proxy.proxy_authorization_value() {
        auth = value;
        headers.push(("Proxy-Authorization", auth.as_bytes()));
    }
    let connect_req =
        http1::serialize_request_head("CONNECT", &target, HttpVersion::HTTP_11, &headers).map_err(
            |e| match e {
                HttpError::InvalidMethod
                | HttpError::InvalidTarget
                | HttpError::InvalidHeaderName
                | HttpError::InvalidHeaderValue => FetchError::InvalidHeader,
                _ => FetchError::Http(e),
            },
        )?;
    conn.write_all(&connect_req).map_err(map_io_error)?;
    // The tunnel setup is not part of the fetch's total deadline (that
    // starts when the request is sent), but a drip-feeding proxy must not
    // hold the CONNECT head open forever either: reuse the same generous
    // bound, measured from the tunnel request.
    let parsed = read_head(
        &mut conn,
        "CONNECT",
        limits,
        Instant::now() + timeouts.fetch_total,
    )?;
    if !(200..300).contains(&parsed.status) {
        // A 407 names the schemes the proxy accepts via
        // `Proxy-Authenticate`; surface it so the caller can see why the
        // tunnel was refused.
        let authenticate = parsed
            .fields
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("proxy-authenticate"))
            .map(|(_, value)| format!(" ({})", String::from_utf8_lossy(value)))
            .unwrap_or_default();
        return Err(FetchError::Proxy(format!(
            "proxy refused CONNECT {target}: HTTP {}{authenticate}",
            parsed.status
        )));
    }
    // A 2xx response to CONNECT has no body (BodySpec::Tunnel), so there
    // is nothing to drain: the bytes after the head are the origin's TLS
    // stream. The proxy may have coalesced the response with the TLS
    // flight, and `read_head` over-read those bytes into `parsed.buf` —
    // feed them to the handshake instead of dropping them, or a
    // coalesced ServerHello is lost and the handshake desyncs.
    let prefix = parsed.buf[parsed.head_len..].to_vec();
    let Conn::Plain(stream) = conn else {
        unreachable!("a proxy CONNECT starts on a plain TCP connection")
    };
    // Offer the origin's cached session ticket, exactly like a direct
    // connection does: the ticket is bound to the origin, not the path,
    // so a fresh tunnel can still resume with an abbreviated handshake.
    // The tunnel itself is never pooled (see `perform_exchange`); its
    // tickets are harvested on discard instead.
    tls_over_with_prefix(stream, prefix, host, pool.ticket(key))
}

fn request_context<'a>(
    url: &'a Url,
    method: &str,
    nav: &'a NavigationContext,
) -> Option<RequestContext<'a>> {
    let host = url.host()?;
    let safe_method = matches!(
        method,
        "GET" | "HEAD" | "OPTIONS" | "TRACE" | "get" | "head" | "options" | "trace"
    );
    // A bare fetch is its own top-level navigation; a subresource fetch
    // keeps the document's top context for every hop of its chain.
    // (fetch_impl anchors a top-level navigation's context at its first
    // hop instead, so redirect chains are never re-rooted per hop.)
    let (top_host, top_secure, top_level_safe_navigation) = match nav.top_host.as_deref() {
        Some(top) => (top, nav.top_secure, nav.top_level_safe_navigation),
        None => (host, Some(url.scheme() == "https"), safe_method),
    };
    Some(RequestContext {
        host,
        path: url.path(),
        secure: url.scheme() == "https",
        http_api: true,
        top_host: Some(top_host),
        top_secure,
        top_level_safe_navigation,
    })
}

/// Anchor a top-level navigation's SameSite context at its first hop:
/// the initial URL's host and scheme stay the top context for every
/// redirect hop of the chain.
fn initial_nav(url: &Url, method: &str) -> NavigationContext {
    let safe_method = matches!(
        method,
        "GET" | "HEAD" | "OPTIONS" | "TRACE" | "get" | "head" | "options" | "trace"
    );
    NavigationContext {
        top_host: url.host().map(str::to_string),
        top_secure: Some(url.scheme() == "https"),
        top_level_safe_navigation: safe_method,
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Absolute-form request-target for the request line (RFC 9110 §4.2.4),
/// without the fragment. The URL serializer emits `username:password@`
/// into the authority; userinfo must never reach the wire, so it is
/// stripped here. The `Host` header and the dial address never contain
/// userinfo: they are built from [`Url::host`] and the port, which exclude
/// credentials by construction.
fn request_target(url: &Url) -> String {
    let full = url.serialize_without_fragment();
    if url.username().is_empty() && url.password().is_empty() {
        return full;
    }
    // The serializer emits `scheme://[userinfo@]host...` whenever a
    // username or password is present, and the host cannot contain `@`,
    // so the first `@` after `://` ends the userinfo.
    let auth = full.find("://").map(|i| i + 3).unwrap_or(0);
    match full[auth..].find('@') {
        Some(i) => {
            let mut out = String::with_capacity(full.len() - i - 1);
            out.push_str(&full[..auth]);
            out.push_str(&full[auth + i + 1..]);
            out
        }
        // Unreachable when credentials are present: the serializer emits
        // them exactly when the username or password is non-empty.
        None => full,
    }
}

/// Header names fetch manages itself; a caller copy would collide with the
/// generated one, so it is dropped. Compared case-insensitively without
/// allocating a lowercased copy per header.
fn is_managed_request_header(name: &str) -> bool {
    name.eq_ignore_ascii_case("host")
        || name.eq_ignore_ascii_case("content-length")
        || name.eq_ignore_ascii_case("transfer-encoding")
        || name.eq_ignore_ascii_case("cookie")
        || name.eq_ignore_ascii_case("connection")
}

/// Serialize the request head. `Host`, `Content-Length`, `Cookie`, a
/// default `User-Agent`, and a default `Accept-Encoding: gzip, deflate`
/// are managed here; caller headers pass through except the managed set
/// (`Host`, `Content-Length`, `Transfer-Encoding`, `Cookie`,
/// `Connection`). A caller-supplied `User-Agent` or `Accept-Encoding`
/// overrides the default. Any userinfo in the URL is stripped from the
/// request-target (see [`request_target`]).
///
/// `proxy_auth` carries the `Proxy-Authorization` value from the
/// [`ProxyConfig`]: when `Some`, it is sent and any caller-supplied
/// `Proxy-Authorization` is dropped (the config is authoritative);
/// when `None`, a caller header passes through unless `strip_auth` drops
/// it. The header is only ever addressed to the proxy: when `origin_form`
/// is set (inside a `CONNECT` tunnel, where the request reaches the
/// origin) it is dropped entirely, caller-supplied or not, because the
/// `CONNECT` request already carried it. `origin_form` selects the
/// origin-form request-target (`/path?query`) used inside a `CONNECT`
/// tunnel; otherwise the absolute-form target is used, which is what a
/// plain-HTTP proxy forwards.
/// The effective request headers for one fetch: the request target plus
/// the full ordered header list after managed-header filtering,
/// defaults, cookies, and content-length. `request_head_bytes`
/// serializes this for HTTP/1.1; the HTTP/2 exchange consumes the same
/// fields directly (minus connection-specific headers).
#[derive(Debug, Clone)]
pub(crate) struct RequestHeaders {
    pub(crate) target: String,
    pub(crate) fields: Vec<(String, Vec<u8>)>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn request_headers(
    method: &str,
    url: &Url,
    extra: &[(String, String)],
    body: Option<&[u8]>,
    cookie: Option<&str>,
    strip_auth: bool,
    proxy_auth: Option<&str>,
    origin_form: bool,
) -> Result<RequestHeaders, FetchError> {
    if method.eq_ignore_ascii_case("connect") {
        // CONNECT is a forbidden method in fetch: the response would turn
        // the connection into an opaque tunnel, which has no body semantics.
        return Err(FetchError::InvalidHeader);
    }
    // Absolute-form request-target for proxies and direct origin
    // connections alike (RFC 9110 §4.2.4); origin-form inside a CONNECT
    // tunnel, where the tunnel already names the authority. Either way
    // without the fragment and without any userinfo (never sent on the
    // wire).
    let target = if origin_form {
        origin_form_target(url)
    } else {
        request_target(url)
    };
    let (host, _, _) = authority(url)?;
    // `Url::port` is `None` when absent or when it is the scheme default,
    // so `Some` always means "include it".
    let host_value = match url.port() {
        Some(p) => format!("{host}:{p}"),
        _ => host.to_owned(),
    };

    let mut fields: Vec<(String, Vec<u8>)> = Vec::with_capacity(extra.len() + 4);
    fields.push(("Host".to_owned(), host_value.into_bytes()));
    let mut has_user_agent = false;
    let mut has_accept_encoding = false;
    for (name, value) in extra {
        // Managed by fetch; a caller copy would collide.
        if is_managed_request_header(name) {
            continue;
        }
        if name.eq_ignore_ascii_case("proxy-authorization") {
            if origin_form {
                // Inside a CONNECT tunnel the request goes to the origin:
                // proxy credentials are never sent past the tunnel. The
                // CONNECT request already carried them.
                continue;
            }
            if proxy_auth.is_some() {
                // The proxy config's credentials are authoritative.
                continue;
            }
            if strip_auth {
                continue;
            }
        } else if strip_auth && name.eq_ignore_ascii_case("authorization") {
            continue;
        }
        if name.eq_ignore_ascii_case("user-agent") {
            has_user_agent = true;
        }
        if name.eq_ignore_ascii_case("accept-encoding") {
            has_accept_encoding = true;
        }
        fields.push((name.clone(), value.as_bytes().to_vec()));
    }
    if !has_user_agent {
        fields.push(("User-Agent".to_owned(), USER_AGENT.as_bytes().to_vec()));
    }
    // Only what this client can decode is advertised; `br` stays out
    // until brotli exists. A caller-supplied value wins, as with
    // `User-Agent`.
    if !has_accept_encoding {
        fields.push(("Accept-Encoding".to_owned(), b"gzip, deflate".to_vec()));
    }
    // Proxy-Authorization is addressed to the proxy only. Inside a
    // CONNECT tunnel the CONNECT request already authenticated; sending
    // it again would hand the proxy's credentials to the origin server.
    if let Some(auth) = proxy_auth.filter(|_| !origin_form) {
        fields.push(("Proxy-Authorization".to_owned(), auth.as_bytes().to_vec()));
    }
    if let Some(cookie) = cookie {
        fields.push(("Cookie".to_owned(), cookie.as_bytes().to_vec()));
    }
    let content_length;
    if let Some(body) = body {
        content_length = body.len().to_string();
        fields.push(("Content-Length".to_owned(), content_length.into_bytes()));
    }
    Ok(RequestHeaders { target, fields })
}

/// The HTTP/2 request header block for one exchange: pseudo-headers
/// first, then regular fields lowercased (RFC 9113 §8.2.2).
/// Connection-specific fields have no meaning in HTTP/2; the fetch
/// Owned H2 header pairs: `(name, value)` byte strings.
type HeaderPairs = Vec<(Vec<u8>, Vec<u8>)>;

/// layer already filters the managed set, and any stragglers are
/// dropped rather than sent.
///
/// Same header selection as the HTTP/1.1 path, byte-identical set.
/// Both the fresh-dial H2 exchange and the pool's shared-H2 fast path
/// build their streams from this, so a multiplexed request is
/// indistinguishable on the wire from an exclusive one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn h2_request_parts(
    method: &str,
    url: &Url,
    extra: &[(String, String)],
    body: Option<&[u8]>,
    cookie: Option<&str>,
    strip_auth: bool,
    proxy_auth: Option<&str>,
    origin_form: bool,
) -> Result<HeaderPairs, FetchError> {
    let parts: RequestHeaders = request_headers(
        method,
        url,
        extra,
        body,
        cookie,
        strip_auth,
        proxy_auth,
        origin_form,
    )?;
    if !is_token(method.as_bytes()) {
        return Err(FetchError::InvalidHeader);
    }
    let authority = parts
        .fields
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case("host"))
        .map(|(_, v)| v.clone())
        .ok_or(FetchError::InvalidHeader)?;
    let mut headers: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(parts.fields.len() + 4);
    headers.push((b":method".to_vec(), method.as_bytes().to_vec()));
    headers.push((b":scheme".to_vec(), url.scheme().as_bytes().to_vec()));
    let path = if parts.target.is_empty() {
        "/".to_owned()
    } else {
        parts.target.clone()
    };
    headers.push((b":path".to_vec(), path.into_bytes()));
    headers.push((b":authority".to_vec(), authority));
    for (name, value) in &parts.fields {
        if name.eq_ignore_ascii_case("host") {
            continue;
        }
        let lower = name.to_ascii_lowercase();
        if matches!(
            lower.as_str(),
            "connection" | "keep-alive" | "proxy-connection" | "transfer-encoding" | "upgrade"
        ) {
            continue;
        }
        headers.push((lower.into_bytes(), value.clone()));
    }
    Ok(headers)
}

#[allow(clippy::too_many_arguments)]
fn request_head_bytes(
    method: &str,
    url: &Url,
    extra: &[(String, String)],
    body: Option<&[u8]>,
    cookie: Option<&str>,
    strip_auth: bool,
    proxy_auth: Option<&str>,
    origin_form: bool,
) -> Result<Vec<u8>, FetchError> {
    let parts = request_headers(
        method,
        url,
        extra,
        body,
        cookie,
        strip_auth,
        proxy_auth,
        origin_form,
    )?;
    let borrowed: Vec<(&str, &[u8])> = parts
        .fields
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect();
    http1::serialize_request_head(method, &parts.target, HttpVersion::HTTP_11, &borrowed).map_err(
        |e| match e {
            HttpError::InvalidMethod
            | HttpError::InvalidTarget
            | HttpError::InvalidHeaderName
            | HttpError::InvalidHeaderValue => FetchError::InvalidHeader,
            _ => FetchError::Http(e),
        },
    )
}

fn map_http_error(e: HttpError) -> FetchError {
    match e {
        HttpError::BodyTooLarge => FetchError::BodyTooLarge,
        _ => FetchError::Http(e),
    }
}

/// Received header fields: (name as sent, raw value bytes).
type HeaderFields = Vec<(String, Vec<u8>)>;

/// A parsed response head with its raw bytes retained for body framing.
struct ParsedHead {
    status: u16,
    /// Header fields in received order: (name as sent, raw value).
    fields: HeaderFields,
    /// Bytes consumed by the head; the body starts here in `buf`.
    head_len: usize,
    headers_seen: usize,
    spec: BodySpec,
    buf: Vec<u8>,
    /// Whether the connection may be reused after this response
    /// (RFC 9112 §9.3), from the *final* response's version and
    /// `Connection` tokens. Interim 1xx responses do not affect it.
    persistent: bool,
}

/// Fail with a [`io::ErrorKind::TimedOut`] I/O error once the total fetch
/// deadline has passed. Called before each socket read in the response
/// phase: unlike the per-read socket timeout, which resets on every byte
/// received, this is an absolute wall-clock bound, so a peer dripping one
/// byte at a time cannot keep a fetch alive. The error flows through
/// [`map_io_error`] into [`FetchError::Timeout`].
pub(crate) fn check_fetch_deadline(deadline: Instant) -> io::Result<()> {
    if Instant::now() >= deadline {
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "total fetch deadline exceeded",
        ))
    } else {
        Ok(())
    }
}

/// Wall-clock deadline for the body phase. The base is the request-send
/// instant plus `fetch_total`; a known `Content-Length` additionally gets
/// a loose per-size extension — one extra second per 64 KiB, as if the
/// peer sustained at least 64 KiB/s — so a large-but-legitimate download
/// at a modest rate is not cut off while a drip-feeding peer still hits a
/// hard wall. Bodies of unknown size (chunked, close-delimited) get only
/// the fixed base bound.
fn fetch_body_deadline(sent_at: Instant, fetch_total: Duration, spec: &BodySpec) -> Instant {
    let slack = match spec {
        BodySpec::ContentLength(n) => Duration::from_secs(n.div_ceil(64 * 1024)),
        _ => Duration::ZERO,
    };
    sent_at + fetch_total + slack
}

/// Read until the response head parses. Returns the head plus the receive
/// buffer, which may already hold the start of the body. Interim 1xx
/// responses (e.g. 103 Early Hints) are discarded and reading continues
/// until a final status; at most [`MAX_INTERIM_RESPONSES`] are tolerated.
fn read_head(
    conn: &mut Conn,
    method: &str,
    limits: &Limits,
    deadline: Instant,
) -> Result<ParsedHead, FetchError> {
    // Heads are small; one 8 KiB chunk almost always holds the whole
    // head, so pre-size for it and skip the first growth steps.
    let mut buf = Vec::with_capacity(8192);
    let mut interim_seen = 0usize;
    loop {
        match http1::parse_response_head(&buf, limits) {
            Ok((head, consumed)) => {
                // This client never sends `Upgrade`, so a 101 means the
                // server is switching to a protocol we do not speak: the
                // bytes after the head are not HTTP and must not be
                // parsed as a response. Fail closed before the generic
                // interim handling below.
                if head.status == 101 {
                    return Err(FetchError::SwitchingProtocols);
                }
                if (100..200).contains(&head.status) {
                    interim_seen += 1;
                    if interim_seen > MAX_INTERIM_RESPONSES {
                        return Err(FetchError::Io(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "too many interim 1xx responses",
                        )));
                    }
                    // `head` borrows `buf`; its last use is the status
                    // check above, so the borrow has ended and the head
                    // bytes can be discarded.
                    buf.drain(..consumed);
                    continue;
                }
                let spec = http1::response_body_spec(&head, method.as_bytes(), limits)
                    .map_err(map_http_error)?;
                let headers_seen = head.headers.len();
                let persistent = http1::is_persistent(head.version, &head.headers);
                let fields = head
                    .headers
                    .iter()
                    .map(|h| (h.name.to_owned(), h.value.to_vec()))
                    .collect();
                let status = head.status;
                // `head` borrows `buf`; the borrow ends here.
                return Ok(ParsedHead {
                    status,
                    fields,
                    head_len: consumed,
                    headers_seen,
                    spec,
                    buf,
                    persistent,
                });
            }
            Err(HttpError::Incomplete) => {
                // The total fetch deadline bounds even a dribbled head:
                // per-read timeouts alone cannot, since they reset on
                // every byte.
                check_fetch_deadline(deadline).map_err(map_io_error)?;
                let mut chunk = [0u8; 8192];
                let n = conn.read(&mut chunk).map_err(map_io_error)?;
                if n == 0 {
                    return Err(FetchError::Io(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "connection closed before the response head completed",
                    )));
                }
                buf.extend_from_slice(&chunk[..n]);
            }
            Err(e) => return Err(map_http_error(e)),
        }
    }
}

/// Read one chunk of body bytes; EOF mid-body is an error (close-delimited
/// bodies use their own loop). `deadline` is the total fetch deadline: it
/// is checked before every read so a drip-feeding peer cannot keep the
/// body phase alive past it.
fn read_more(conn: &mut Conn, buf: &mut Vec<u8>, deadline: Instant) -> Result<(), FetchError> {
    // 64 KiB reads: bulk bodies arrive in a quarter of the syscalls of
    // the old 16 KiB chunk; the stack cost is negligible and short reads
    // still return as soon as data is available.
    check_fetch_deadline(deadline).map_err(map_io_error)?;
    let mut chunk = [0u8; 65536];
    let n = conn.read(&mut chunk).map_err(map_io_error)?;
    if n == 0 {
        return Err(FetchError::Io(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "connection closed mid-body",
        )));
    }
    buf.extend_from_slice(&chunk[..n]);
    Ok(())
}

/// Read the body per its framing, enforcing [`MAX_BODY_BYTES`]. When the
/// response carries a decodable `Content-Encoding` (`gzip`/`deflate`),
/// body bytes stream through the incremental decoder as they arrive:
/// the encoded side never sits whole in memory, and the decoded output
/// is capped at the same limit as it is produced. Returns
/// the head fields with any chunked trailers appended, the body, and
/// whether the framing consumed exactly the response: no bytes beyond
/// the message boundary, so the connection sits at the start of the next
/// response and may be pooled. Close-delimited bodies are never exactly
/// framed — the message ends only when the peer closes.
fn read_body(
    conn: &mut Conn,
    parsed: ParsedHead,
    limits: &Limits,
    deadline: Instant,
) -> Result<(HeaderFields, Vec<u8>, bool), FetchError> {
    let ParsedHead {
        status,
        mut fields,
        head_len,
        headers_seen,
        spec,
        mut buf,
        ..
    } = parsed;
    let mut chain = content_decoder(&fields, limits)?;
    let (body, framed_exact) = match spec {
        // Any bytes past the head belong to a message we must not
        // swallow: only an exactly-consumed head is reusable.
        BodySpec::Empty => (Vec::new(), buf.len() == head_len),
        // `fetch` never issues CONNECT itself, so tunnel framing here
        // would mean the proxy path evolved without updating this
        // match: fail closed rather than misparse the tunnel stream
        // as a message body.
        BodySpec::Tunnel => {
            return Err(FetchError::Proxy(
                "tunnel framing reached the response body reader".to_string(),
            ));
        }
        BodySpec::ContentLength(n) => {
            // `response_body_spec` already enforced n <= max_body_bytes.
            let n = n as usize;
            match chain.as_mut() {
                // No decoding: reuse the receive buffer for the body.
                None => {
                    // Reserve the whole body up front: `buf` currently holds
                    // little more than the head plus the first body bytes, so
                    // without this the read loop grows it by repeated doubling,
                    // paying ~2x the body size in reallocation copies.
                    let have = buf.len().saturating_sub(head_len);
                    buf.reserve(n.saturating_sub(have));
                    while buf.len() - head_len < n {
                        read_more(conn, &mut buf, deadline)?;
                    }
                    // Exact framing means the peer sent precisely n body bytes;
                    // anything more belongs to the next response.
                    let framed_exact = buf.len() - head_len == n;
                    // Drop the head bytes in place instead of allocating a
                    // second copy of up to `max_body_bytes`.
                    buf.drain(..head_len);
                    buf.truncate(n);
                    (buf, framed_exact)
                }
                // Decoding: feed each chunk to the incremental decoder as
                // it arrives; only the decoded output accumulates.
                Some(chain) => {
                    let mut out = Vec::new();
                    let mut remaining = n;
                    // Offset into `buf` of the first not-yet-fed byte.
                    let mut start = head_len;
                    while remaining > 0 {
                        if start == buf.len() {
                            // Drop consumed bytes (the head is fully
                            // parsed); only unread input is retained.
                            buf.drain(..start);
                            start = 0;
                            read_more(conn, &mut buf, deadline)?;
                        }
                        let take = (buf.len() - start).min(remaining);
                        chain
                            .feed(&buf[start..start + take], &mut out)
                            .map_err(map_decode_error)?;
                        start += take;
                        remaining -= take;
                    }
                    finish_chain(status, chain, &mut out)?;
                    // As above: leftover bytes past the n encoded bytes
                    // belong to the next response.
                    let framed_exact = start == buf.len();
                    (out, framed_exact)
                }
            }
        }
        BodySpec::Chunked => {
            // Stream through the incremental decoder: each input byte is
            // fed once (no O(n^2) re-parse of the whole buffer), and the
            // decoder counts decoded payload bytes against the cap as they
            // arrive, so framing bytes can never blow past it either.
            let mut decoder = http1::ChunkedDecoder::new_with_headers_seen(*limits, headers_seen);
            let mut out = Vec::new();
            // Offset into `buf` of the first not-yet-fed byte.
            let mut start = head_len;
            loop {
                // `events` is per-iteration: its `Data` slices borrow
                // `buf`, so the borrow must end before `buf` is mutated
                // below; a `Vec` reused across iterations (even with
                // `clear()`, which counts as a use of the borrow) would
                // keep the borrow live for the whole loop.
                let mut events = Vec::new();
                let consumed = decoder
                    .feed(&buf[start..], &mut events)
                    .map_err(map_http_error)?;
                for event in events.drain(..) {
                    match event {
                        http1::ChunkEvent::Data(data) => match chain.as_mut() {
                            // Content-decoding streams too: chunk payload
                            // feeds the decoder instead of accumulating
                            // encoded.
                            Some(chain) => chain.feed(data, &mut out).map_err(map_decode_error)?,
                            None => out.extend_from_slice(data),
                        },
                        // Trailers are surfaced as response headers,
                        // appended after the head fields.
                        http1::ChunkEvent::Trailer(t) => {
                            fields.push((t.name, t.value));
                        }
                        http1::ChunkEvent::End => {}
                    }
                }
                start += consumed;
                if decoder.is_done() {
                    break;
                }
                // Compact the buffer so a slow peer cannot grow it without
                // bound; only undecoded framing bytes are retained.
                if start > 0 {
                    buf.drain(..start);
                    start = 0;
                }
                read_more(conn, &mut buf, deadline)?;
            }
            if let Some(chain) = chain.as_mut() {
                finish_chain(status, chain, &mut out)?;
            }
            // The decoder stops exactly at the message boundary; any
            // unfed bytes past `start` belong to the next response.
            let framed_exact = start == buf.len();
            (out, framed_exact)
        }
        BodySpec::UntilClose => match chain.as_mut() {
            // No decoding: the body moves out of the receive buffer
            // instead of being copied into a fresh one.
            None => {
                loop {
                    // 64 KiB reads, as in `read_more`: fewer syscalls for
                    // bulk close-delimited bodies.
                    check_fetch_deadline(deadline).map_err(map_io_error)?;
                    let mut chunk = [0u8; 65536];
                    let n = conn.read(&mut chunk).map_err(map_io_error)?;
                    if n == 0 {
                        break;
                    }
                    if buf.len() - head_len + n > limits.max_body_bytes {
                        return Err(FetchError::BodyTooLarge);
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
                buf.drain(..head_len);
                (buf, false)
            }
            // Decoding: feed each read to the incremental decoder; the
            // encoded side stays bounded while decoded output grows
            // against the decoder's own cap.
            Some(chain) => {
                let mut out = Vec::new();
                let mut start = head_len;
                loop {
                    if start < buf.len() {
                        let end = buf.len();
                        chain
                            .feed(&buf[start..end], &mut out)
                            .map_err(map_decode_error)?;
                        start = end;
                    }
                    let mut chunk = [0u8; 65536];
                    check_fetch_deadline(deadline).map_err(map_io_error)?;
                    let n = conn.read(&mut chunk).map_err(map_io_error)?;
                    if n == 0 {
                        break;
                    }
                    // Compact: only unread input is retained.
                    buf.drain(..start);
                    start = 0;
                    buf.extend_from_slice(&chunk[..n]);
                }
                finish_chain(status, chain, &mut out)?;
                // Never exactly framed: the message ends only when the
                // peer closes.
                (out, false)
            }
        },
    };
    Ok((fields, body, framed_exact))
}

/// Build the incremental `Content-Encoding` pipeline for a response, if
/// its headers name a decodable coding. Returns `None` when the body
/// passes through untouched (no header, empty, or `identity` only).
/// Unknown codings fail closed.
pub(crate) fn content_decoder(
    fields: &HeaderFields,
    limits: &Limits,
) -> Result<Option<DecodeChain>, FetchError> {
    let Some(value) = fields
        .iter()
        .rev()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-encoding"))
        .map(|(_, v)| header_string(v.clone()))
    else {
        return Ok(None);
    };
    if value.trim().is_empty() || value.trim().eq_ignore_ascii_case("identity") {
        return Ok(None);
    }
    DecodeChain::new(&value, limits.max_body_bytes).map_err(|e| match e {
        // Only construction can see an unknown token.
        DecodeError::UnsupportedEncoding => FetchError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported content-encoding: {value}"),
        )),
        e => map_decode_error(e),
    })
}

/// Map a [`DecodeError`] from the streaming decoder to the fetch-layer
/// error: cap breaches are [`FetchError::BodyTooLarge`], corrupt data is
/// an I/O error. Never panics; unknown codings never reach here (see
/// [`content_decoder`]).
pub(crate) fn map_decode_error(e: DecodeError) -> FetchError {
    match e {
        DecodeError::OutputTooLarge => FetchError::BodyTooLarge,
        DecodeError::UnsupportedEncoding => FetchError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported content-encoding",
        )),
        DecodeError::CodingDepthExceeded => FetchError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "too many stacked content codings",
        )),
        DecodeError::Truncated | DecodeError::InvalidData => FetchError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "content-encoding body failed to decode",
        )),
    }
}

/// Run the decoder's end-of-stream trailer checks. Skipped only when the
/// response status forbids a body (1xx, 204, 304): those responses may
/// carry a `Content-Encoding` header with no payload. Any other status
/// with an empty encoded body is corrupt input, so the trailer check
/// runs and the decode error surfaces instead of passing the empty body
/// through untouched.
pub(crate) fn finish_chain(
    status: u16,
    chain: &mut DecodeChain,
    out: &mut Vec<u8>,
) -> Result<(), FetchError> {
    if !matches!(status, 100..=199 | 204 | 304) || chain.fed_input() > 0 {
        chain.finish(out).map_err(map_decode_error)?;
    }
    Ok(())
}

/// Header value bytes as a `String`: moved, not copied, when they are
/// valid UTF-8 (the common case); lossy-decoded only otherwise.
pub(crate) fn header_string(value: Vec<u8>) -> String {
    match String::from_utf8(value) {
        Ok(s) => s,
        Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
    }
}

/// Whether a failed exchange on a pooled connection may be retried on a
/// fresh connection: only for safe methods (replaying them cannot
/// duplicate a side effect), only when no response head arrived, and
/// only once. A non-idempotent request body is never replayed.
fn may_retry_stale(was_pooled: bool, got_head: bool, method: &str, retried: bool) -> bool {
    was_pooled && !got_head && !retried && matches!(method, "GET" | "HEAD" | "OPTIONS" | "TRACE")
}

/// One exchange's connection: an HTTP/1.1 connection (fresh or pooled)
/// or a resumed pooled HTTP/2 session.
enum FetchConn {
    H1(Conn),
    H2Shared(SharedStream),
}

#[allow(clippy::too_many_arguments)]
fn fetch_once(
    method: &str,
    url: &Url,
    extra_headers: &[(String, String)],
    body: Option<&[u8]>,
    jar: &mut CookieJar,
    strip_auth: bool,
    limits: &Limits,
    timeouts: Timeouts,
    nav: &NavigationContext,
    pool: &Pool,
    proxy: Option<&ProxyConfig>,
) -> Result<(Response, bool), FetchError> {
    let (host, port, secure) = authority(url)?;
    // The proxy applies per hop: each redirect target re-evaluates
    // `no_proxy` against its own host.
    let via_proxy = proxy.filter(|p| !p.bypasses(host, port));
    let proxy_auth = via_proxy.and_then(|p| p.proxy_authorization_value());
    // Inside a CONNECT tunnel the request looks like a direct one:
    // origin-form target. Plain http through a proxy keeps the
    // absolute-form target the proxy forwards.
    let origin_form = via_proxy.is_some() && secure;
    // Build the head before connecting: a malformed caller request must
    // fail as InvalidHeader, not as a connection error.
    let now = now_secs();
    let ctx = request_context(url, method, nav);
    let cookie: Option<String> = ctx.as_ref().and_then(|ctx| jar.cookie_header(ctx, now));
    // What actually went on the wire for this hop: a `Cookie` header
    // the jar produced, or a caller `Authorization` header that
    // `request_headers` did not strip (a stripped one — cross-origin
    // redirect, `strip_auth` — counts for nothing: it never left).
    // `Proxy-Authorization` is deliberately excluded: it is addressed
    // to the proxy, not the origin, so it is not an origin credential
    // for CORS purposes. URL userinfo is likewise excluded: it is
    // stripped from the request-target and never sent (see
    // `request_target`).
    let credentials_sent = cookie.is_some()
        || (!strip_auth
            && extra_headers
                .iter()
                .any(|(n, _)| n.eq_ignore_ascii_case("authorization")));
    let head = request_head_bytes(
        method,
        url,
        extra_headers,
        body,
        cookie.as_deref(),
        strip_auth,
        proxy_auth.as_deref(),
        origin_form,
    )?;
    let key = PoolKey::new(secure, host, port);

    let mut retried = false;
    // A pooled connection that fails is dropped; a fresh one is dialed
    // only when the pool has nothing usable. Proxied connections bypass
    // the pool entirely: a pooled entry is keyed by origin, but a
    // proxied socket is bound to a (proxy, origin) pair, so pooling it
    // under the origin key would hand one origin's tunnel to another.
    // Each proxied exchange dials fresh.
    //
    // The shared-H2 fast path opens its stream before the retry loop;
    // its deadline and start time are captured there so the response
    // below reuses them instead of starting a second full budget.
    let mut h2_shared_timing: Option<(Instant, Instant)> = None;
    let mut next: Option<(FetchConn, bool)> = if via_proxy.is_none() {
        // Prefer a shared H2 session: concurrent fetches to one origin
        // multiplex streams on the same connection instead of racing to
        // dial their own.
        let mut shared: Option<(FetchConn, bool)> = None;
        if let Some(pooled) = pool.take_shared_h2(&key, timeouts.fetch_total) {
            // A malformed request fails here, before any I/O: the
            // shared path needs the headers to open its stream.
            let headers = h2_request_parts(
                method,
                url,
                extra_headers,
                body,
                cookie.as_deref(),
                strip_auth,
                proxy_auth.as_deref(),
                origin_form,
            )?;
            let header_refs: Vec<(&[u8], &[u8])> = headers
                .iter()
                .map(|(n, v)| (n.as_slice(), v.as_slice()))
                .collect();
            // One budget for the whole exchange: the stream open and
            // the response share this deadline and start time.
            let sent_at = Instant::now();
            let deadline = sent_at + timeouts.fetch_total;
            match pooled.open_request(&header_refs, body.unwrap_or(&[]), deadline) {
                Ok(stream) => {
                    shared = Some((FetchConn::H2Shared(stream), true));
                    h2_shared_timing = Some((deadline, sent_at));
                }
                // Nothing reached the peer — saturated, or the peer
                // went away between take and open: fall through to the
                // H1 pool, then a fresh dial. The session stays pooled
                // for other streams.
                Err(OpenRequestError::PreSend(_reason)) => {}
                // Request bytes may have reached the peer: never
                // silently re-send. An idempotent method may still be
                // retried on another connection (pooled, no response
                // head — the same discipline as a stale H1
                // connection); anything else fails the fetch here so a
                // non-idempotent request is never executed twice.
                Err(OpenRequestError::PostSend(e)) => {
                    if may_retry_stale(true, false, method, retried) {
                        retried = true;
                    } else {
                        return Err(e.into());
                    }
                }
            }
        }
        shared.or_else(|| pool.take(&key).map(|conn| (FetchConn::H1(conn), true)))
    } else {
        None
    };
    loop {
        let (fetch_conn, was_pooled) = match next.take() {
            Some(pair) => pair,
            None => (
                match via_proxy {
                    Some(p) => FetchConn::H1(proxy_connect(url, p, timeouts, limits, pool, &key)?),
                    None => FetchConn::H1(connect(url, timeouts, pool, &key)?),
                },
                false,
            ),
        };
        // A shared H2 stream: the request is already on the wire —
        // `open_request` sent it when the stream was opened above.
        // The deadline and start time are the exchange's originals,
        // not a fresh budget.
        if let FetchConn::H2Shared(mut stream) = fetch_conn {
            let stream_id = stream.stream_id();
            let (deadline, sent_at) =
                h2_shared_timing.expect("an H2Shared stream always carries its open timing");
            let (result, got_head) = crate::http2::h2_recv_response(
                &mut stream,
                stream_id,
                method,
                url,
                deadline,
                sent_at,
                timeouts.fetch_total,
                limits,
                &mut *jar,
                ctx.as_ref(),
                now,
            );
            match result {
                Ok(response) => return Ok((response, credentials_sent)),
                Err(e) => {
                    // The stream handle's `Drop` releases its slot; a
                    // dead session was already flagged inside, so the
                    // pool will not hand it out again. A stale pooled
                    // session gets one retry on a fresh connection,
                    // like HTTP/1.1 — but only when the failure came
                    // before the response head.
                    if may_retry_stale(was_pooled, got_head, method, retried) {
                        retried = true;
                        continue;
                    }
                    return Err(e);
                }
            }
        }
        // A resumed H2 session: the handshake is already done, so this
        // exchange just opens a fresh stream on the existing connection.
        // (Pooled connections never come from a proxy, so there is no
        // proxy case here.)
        let FetchConn::H1(mut conn) = fetch_conn else {
            unreachable!("shared H2 streams are handled above")
        };
        let mut got_head = false;
        // A pooled H1 connection always takes the HTTP/1.1 path (pooled
        // H2 sessions are handled above); a fresh connection takes the
        // H2 path when ALPN negotiated it.
        let use_h2 = !was_pooled && crate::http2::is_h2_conn(&conn);
        if use_h2 {
            let (result, session) = crate::http2::fetch_h2(
                conn,
                method,
                url,
                extra_headers,
                body,
                cookie.as_deref(),
                strip_auth,
                proxy_auth.as_deref(),
                origin_form,
                timeouts.fetch_total,
                limits,
                &mut *jar,
                ctx.as_ref(),
                now,
            );
            match result {
                Ok(response) => {
                    // Pool the fresh H2 session: the next fetch to this
                    // origin multiplexes streams on it instead of dialing
                    // again.
                    if via_proxy.is_none() && session.goaway().is_none() {
                        pool.put_shared_h2(&key, session, timeouts.rw);
                    } else {
                        // Proxied connections are dropped, never pooled
                        // (see above) — but their tickets are still
                        // harvested. So is a session the server is
                        // shutting down.
                        pool.discard(&key, session.into_inner());
                    }
                    return Ok((response, credentials_sent));
                }
                Err(e) => {
                    // Harvest session tickets even from a failed
                    // session: a response may have carried them before
                    // the failure.
                    pool.discard(&key, session.into_inner());
                    return Err(e);
                }
            }
        }
        let outcome = (|| -> Result<(Response, bool), FetchError> {
            conn.write_all(&head).map_err(map_io_error)?;
            if let Some(body) = body {
                conn.write_all(body).map_err(map_io_error)?;
            }
            // The request is fully sent: the total fetch deadline starts
            // here. It is an absolute wall-clock bound on the response
            // phase, so a peer drip-feeding bytes can never hold the
            // fetch open past it even though per-read timeouts reset on
            // every byte received.
            let sent_at = Instant::now();
            let parsed = read_head(&mut conn, method, limits, sent_at + timeouts.fetch_total)?;
            got_head = true;
            let status = parsed.status;
            // Store Set-Cookie before reading the body: the jar is per-request
            // context, and a huge body must not delay cookie storage.
            if let Some(ctx) = ctx.as_ref() {
                for (name, value) in &parsed.fields {
                    if name.eq_ignore_ascii_case("set-cookie") {
                        jar.set_cookie(&String::from_utf8_lossy(value), ctx, now);
                    }
                }
            }
            let persistent = parsed.persistent;
            let body_deadline = fetch_body_deadline(sent_at, timeouts.fetch_total, &parsed.spec);
            let (fields, body, framed_exact) = read_body(&mut conn, parsed, limits, body_deadline)?;
            // Reuse needs both: the peer must allow persistence *and* the
            // framing must have consumed exactly this response, leaving
            // the connection positioned at the start of the next one.
            let reusable = framed_exact && persistent;
            Ok((
                Response {
                    status,
                    headers: fields
                        .into_iter()
                        .map(|(n, v)| (n, header_string(v)))
                        .collect(),
                    body,
                    url: url.clone(),
                },
                reusable,
            ))
        })();
        match outcome {
            Ok((response, true)) => {
                if via_proxy.is_none() {
                    pool.put(&key, conn);
                } else {
                    // A proxied connection is dropped, never pooled (see
                    // above) — but its tickets are still harvested, so a
                    // later tunnel can resume.
                    pool.discard(&key, conn);
                }
                return Ok((response, credentials_sent));
            }
            Ok((response, false)) => {
                // `discard` harvests tickets without pooling, for direct
                // and proxied connections alike.
                pool.discard(&key, conn);
                return Ok((response, credentials_sent));
            }
            Err(e) => {
                // Harvest session tickets even from a failed connection:
                // a response may have carried them before the failure.
                // Proxied connections are dropped without pooling, but
                // their tickets are still harvested.
                pool.discard(&key, conn);
                if may_retry_stale(was_pooled, got_head, method, retried) {
                    // The pooled connection died before delivering a
                    // response; retry once on a fresh connection.
                    retried = true;
                    continue;
                }
                return Err(e);
            }
        }
    }
}

/// The `charset` parameter of a `Content-Type` value, if present.
fn charset_from_content_type(content_type: &str) -> Option<&str> {
    let mut parts = content_type.split(';');
    parts.next()?; // media type
    for param in parts {
        let (name, value) = param.split_once('=')?;
        if name.trim().eq_ignore_ascii_case("charset") {
            let value = value.trim().trim_matches('"').trim();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http2::Session;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    fn url(s: &str) -> Url {
        bowser_url::parse(s, None).unwrap()
    }

    fn get_request(url_str: &str) -> Request {
        Request {
            method: "GET".to_string(),
            url: url(url_str),
            headers: Vec::new(),
            body: None,
        }
    }

    /// A generous total-fetch deadline for tests that read heads/bodies
    /// directly: it only needs to not fire during a fast local exchange.
    fn test_deadline() -> Instant {
        Instant::now() + Duration::from_secs(60)
    }

    // -- pure unit tests ----------------------------------------------------

    #[test]
    fn same_origin_cases() {
        assert!(is_same_origin(
            &url("http://a.com/"),
            &url("http://a.com/x")
        ));
        assert!(is_same_origin(
            &url("http://a.com/"),
            &url("http://a.com:80/")
        ));
        assert!(is_same_origin(
            &url("https://a.com:443/"),
            &url("https://a.com/")
        ));
        assert!(!is_same_origin(
            &url("http://a.com/"),
            &url("https://a.com/")
        ));
        assert!(!is_same_origin(
            &url("http://a.com/"),
            &url("http://a.com:81/")
        ));
        assert!(!is_same_origin(
            &url("http://a.com/"),
            &url("http://b.com/")
        ));
        assert!(!is_same_origin(
            &url("http://a.com/"),
            &url("data:text/plain,hi")
        ));
    }

    #[test]
    fn charset_param_parsing() {
        assert_eq!(
            charset_from_content_type("text/html; charset=windows-1252"),
            Some("windows-1252")
        );
        assert_eq!(
            charset_from_content_type("text/html; Charset=\"UTF-8\""),
            Some("UTF-8")
        );
        assert_eq!(
            charset_from_content_type("text/html;charset = iso-8859-1 ; foo=bar"),
            Some("iso-8859-1")
        );
        assert_eq!(charset_from_content_type("text/html"), None);
        assert_eq!(charset_from_content_type("text/html; charset="), None);
        assert_eq!(charset_from_content_type("text/html; foo=bar"), None);
    }

    #[test]
    fn text_decoding() {
        let mut r = Response {
            status: 200,
            headers: vec![(
                "Content-Type".to_string(),
                "text/plain; charset=windows-1252".to_string(),
            )],
            body: b"caf\xe9".to_vec(),
            url: url("http://a.com/"),
        };
        assert_eq!(r.text(), "café");
        // No charset: UTF-8 default, invalid bytes become U+FFFD.
        r.headers.clear();
        r.body = b"a\xffb".to_vec();
        assert_eq!(r.text(), "a\u{fffd}b");
        // BOM wins over the charset parameter.
        r.headers = vec![(
            "Content-Type".to_string(),
            "text/plain; charset=windows-1252".to_string(),
        )];
        r.body = b"\xff\xfeh\x00i\x00".to_vec();
        assert_eq!(r.text(), "hi");
        // Unknown charset label: fall back to UTF-8 behavior.
        r.headers = vec![(
            "Content-Type".to_string(),
            "text/plain; charset=bogus-99".to_string(),
        )];
        r.body = "héllo".as_bytes().to_vec();
        assert_eq!(r.text(), "héllo");
    }

    #[test]
    fn unsupported_scheme() {
        for s in ["ftp://a.com/x", "data:text/plain,hi", "file:///etc/passwd"] {
            let req = get_request(s);
            match fetch(&req) {
                Err(FetchError::UnsupportedScheme(scheme)) => {
                    assert!(s.starts_with(&scheme), "{s} -> {scheme}");
                }
                other => panic!("expected UnsupportedScheme for {s}, got {other:?}"),
            }
        }
    }

    #[test]
    fn invalid_header_value_rejected() {
        let mut req = get_request("http://127.0.0.1/");
        req.headers.push(("X-Bad".to_string(), "a\nb".to_string()));
        assert!(matches!(fetch(&req), Err(FetchError::InvalidHeader)));
    }

    #[test]
    fn connect_method_rejected() {
        let mut req = get_request("http://127.0.0.1/");
        req.method = "CONNECT".to_string();
        assert!(matches!(fetch(&req), Err(FetchError::InvalidHeader)));
    }

    #[test]
    fn method_normalization() {
        assert_eq!(normalize_method("get"), "GET");
        assert_eq!(normalize_method("Post"), "POST");
        assert_eq!(normalize_method("PURGE"), "PURGE");
    }

    // -- Happy Eyeballs racing (no external network) -----------------------

    fn race_addrs(base_port: u16, n: u16) -> Vec<SocketAddr> {
        (0..n)
            .map(|i| SocketAddr::from(([127, 0, 0, 1], base_port + i)))
            .collect()
    }

    #[test]
    fn sock_addrs_applies_request_port_to_cached_ips() {
        // The DNS cache holds bare IPs; the request's port is applied at
        // dial time. One cached answer must serve every scheme: http gets
        // 80, https gets 443, and an explicit port gets exactly that —
        // never the port of whichever scheme resolved first.
        let ips = vec![
            IpAddr::from([93, 184, 216, 34]),
            IpAddr::from([0x2606, 0x2800, 0x220, 1, 0x248, 0x1893, 0x25c8, 0x1946]),
        ];
        assert_eq!(
            sock_addrs(&ips, 80),
            vec![SocketAddr::new(ips[0], 80), SocketAddr::new(ips[1], 80)]
        );
        assert_eq!(
            sock_addrs(&ips, 443),
            vec![SocketAddr::new(ips[0], 443), SocketAddr::new(ips[1], 443)]
        );
        assert_eq!(sock_addrs(&ips, 8080)[0].port(), 8080);
        assert_eq!(sock_addrs(&ips, 8080)[1].port(), 8080);
    }

    #[test]
    fn race_single_address_dials_on_calling_thread() {
        // The fast path: one address is dialed directly, without spawning
        // a racer thread.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            for _ in 0..2 {
                if listener.accept().is_err() {
                    break;
                }
            }
        });
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        let caller = thread::current().id();
        let stream = race_with(&[addr], move |a, _cancel: &AtomicBool| {
            assert_eq!(a, addr);
            assert_eq!(
                thread::current().id(),
                caller,
                "a single address must be dialed on the calling thread, not a spawned racer"
            );
            TcpStream::connect(a)
        })
        .unwrap();
        drop(stream);
    }

    #[test]
    fn race_first_success_wins_without_waiting_for_dead_address() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        // A real (loopback-only) listener so the winning dial can return
        // a genuine TcpStream; nothing leaves the machine.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            for _ in 0..8 {
                if listener.accept().is_err() {
                    break;
                }
            }
        });
        let winner = std::sync::Arc::new(AtomicUsize::new(usize::MAX));
        let addrs = race_addrs(40000, 3);
        let start = std::time::Instant::now();
        let stream = race_with(&addrs, {
            let winner = winner.clone();
            move |addr, _cancel: &AtomicBool| {
                let idx = (addr.port() - 40000) as usize;
                if idx == 0 {
                    // Dead first address: hangs far past the stagger, then
                    // fails. The old sequential code would have stalled
                    // here for the full connect timeout.
                    thread::sleep(Duration::from_secs(10));
                    Err(io::Error::new(io::ErrorKind::TimedOut, "dead"))
                } else {
                    winner.store(idx, Ordering::SeqCst);
                    TcpStream::connect(("127.0.0.1", port))
                }
            }
        })
        .unwrap();
        drop(stream);
        let elapsed = start.elapsed();
        assert_eq!(
            winner.load(Ordering::SeqCst),
            1,
            "the second address should win the race"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "racing must not wait for the dead address; took {elapsed:?}"
        );
    }

    #[test]
    fn race_all_fail_returns_last_error() {
        let addrs = race_addrs(41000, 2);
        let err = race_with(&addrs, |addr, _cancel: &AtomicBool| {
            let idx = addr.port() - 41000;
            if idx == 1 {
                // Fail after the stagger so this error is deterministically
                // the last one reported.
                thread::sleep(Duration::from_millis(50));
            }
            Err(io::Error::new(
                io::ErrorKind::ConnectionRefused,
                format!("dead-{idx}"),
            ))
        })
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::ConnectionRefused);
        assert!(err.to_string().contains("dead-1"), "{err}");
    }

    #[test]
    fn race_no_addresses_fails() {
        let err = race_connect(&[], Duration::from_secs(1)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AddrNotAvailable);
    }

    #[test]
    fn race_connect_single_address_dials_directly() {
        // One address takes the fast path: a plain `connect_timeout`
        // on the calling thread — no helper thread, no channel, no
        // poll loop.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let _ = listener.accept();
        });
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        let stream = race_connect(&[addr], Duration::from_secs(30)).unwrap();
        assert_eq!(stream.peer_addr().unwrap().port(), port);
    }

    #[test]
    fn race_connect_single_address_reports_refusal() {
        // A refused single-address dial surfaces the refusal itself,
        // not a timeout and not an interrupt.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        let err = race_connect(&[addr], Duration::from_secs(30)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::ConnectionRefused);
    }

    #[test]
    fn race_connect_single_address_zero_timeout_is_timed_out() {
        // A zero budget never dials: the old interruptible path
        // reported `TimedOut` without touching the network, and the
        // direct fast path preserves exactly that (a plain
        // `connect_timeout` with a zero duration is EINVAL on Linux).
        let addr = SocketAddr::from(([127, 0, 0, 1], 9));
        let err = race_connect(&[addr], Duration::ZERO).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
    }

    #[test]
    fn race_attempts_start_staggered() {
        let starts = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let addrs = race_addrs(42000, 3);
        let _ = race_with(&addrs, {
            let starts = starts.clone();
            move |addr, _cancel: &AtomicBool| {
                let idx = (addr.port() - 42000) as usize;
                starts
                    .lock()
                    .unwrap()
                    .push((idx, std::time::Instant::now()));
                thread::sleep(Duration::from_millis(100));
                Err(io::Error::new(io::ErrorKind::ConnectionRefused, "dead"))
            }
        });
        let mut starts = starts.lock().unwrap();
        starts.sort_by_key(|&(idx, _)| idx);
        assert_eq!(starts.len(), 3);
        for w in starts.windows(2) {
            let gap = w[1].1.duration_since(w[0].1);
            assert!(
                gap >= Duration::from_millis(200),
                "attempts must start ~250ms apart; gap was {gap:?}"
            );
        }
    }

    #[test]
    fn race_loser_dial_abandons_on_cancel() {
        // The loser hangs the way a blackholed connect would, but
        // watches the cancel flag the way interruptible_connect does.
        // The race must finish promptly after the winner lands instead
        // of waiting out the loser's full timeout.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            for _ in 0..4 {
                if listener.accept().is_err() {
                    break;
                }
            }
        });
        let addrs = race_addrs(43000, 2);
        let start = std::time::Instant::now();
        let stream = race_with(&addrs, move |addr, cancel: &AtomicBool| {
            let idx = (addr.port() - 43000) as usize;
            if idx == 0 {
                let wait_start = std::time::Instant::now();
                while !cancel.load(Ordering::SeqCst) {
                    if wait_start.elapsed() > Duration::from_secs(10) {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "cancel flag never arrived",
                        ));
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                return Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"));
            }
            TcpStream::connect(("127.0.0.1", port))
        })
        .unwrap();
        drop(stream);
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(5),
            "loser must abandon on cancel, not hold its thread; took {elapsed:?}"
        );
    }

    #[test]
    fn race_caps_concurrent_attempts() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        // 20 addresses, far past the racer cap: at most
        // MAX_RACE_ATTEMPTS dials may be in flight at once, but every
        // address is still tried exactly once.
        let addrs = race_addrs(44000, 20);
        let in_flight = std::sync::Arc::new(AtomicUsize::new(0));
        let max_in_flight = std::sync::Arc::new(AtomicUsize::new(0));
        let started = std::sync::Arc::new(AtomicUsize::new(0));
        let release = std::sync::Arc::new(AtomicBool::new(false));
        // While the first wave of racers is still dialing, no queued
        // address may start; once proven, release them all so the race
        // drains.
        let watcher = {
            let (started, release) = (started.clone(), release.clone());
            thread::spawn(move || {
                while started.load(Ordering::SeqCst) < MAX_RACE_ATTEMPTS {
                    thread::yield_now();
                }
                thread::sleep(Duration::from_millis(300));
                assert_eq!(
                    started.load(Ordering::SeqCst),
                    MAX_RACE_ATTEMPTS,
                    "queued addresses must wait for a racer to free up"
                );
                release.store(true, Ordering::SeqCst);
            })
        };
        let err = race_with(&addrs, {
            let (in_flight, max_in_flight, started, release) = (
                in_flight.clone(),
                max_in_flight.clone(),
                started.clone(),
                release.clone(),
            );
            move |_addr, _cancel: &AtomicBool| {
                started.fetch_add(1, Ordering::SeqCst);
                let cur = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                max_in_flight.fetch_max(cur, Ordering::SeqCst);
                while !release.load(Ordering::SeqCst) {
                    thread::sleep(Duration::from_millis(1));
                }
                in_flight.fetch_sub(1, Ordering::SeqCst);
                Err(io::Error::new(io::ErrorKind::ConnectionRefused, "dead"))
            }
        })
        .unwrap_err();
        watcher.join().expect("watcher panicked");
        assert_eq!(err.kind(), io::ErrorKind::ConnectionRefused);
        assert_eq!(
            started.load(Ordering::SeqCst),
            20,
            "every address must eventually be tried"
        );
        assert!(
            max_in_flight.load(Ordering::SeqCst) <= MAX_RACE_ATTEMPTS,
            "at most MAX_RACE_ATTEMPTS dials may run concurrently"
        );
    }

    #[test]
    fn interruptible_connect_abandons_when_already_cancelled() {
        // A pre-set cancel flag must fail fast without dialing at all.
        let cancel = AtomicBool::new(true);
        let addr = SocketAddr::from(([127, 0, 0, 1], 9));
        let start = std::time::Instant::now();
        let err = interruptible_connect(&addr, Duration::from_secs(30), &cancel).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Interrupted);
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "cancelled connect must not block"
        );
    }

    #[test]
    fn interruptible_connect_reports_refusal_fast() {
        // A refused connect surfaces immediately, not after the budget
        // is burned.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        let err = interruptible_connect(&addr, Duration::from_secs(30), &AtomicBool::new(false))
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::ConnectionRefused);
    }

    #[test]
    fn interruptible_connect_succeeds_on_loopback() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            for _ in 0..2 {
                if listener.accept().is_err() {
                    break;
                }
            }
        });
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        let mut stream =
            interruptible_connect(&addr, Duration::from_secs(30), &AtomicBool::new(false))
                .expect("loopback connect must succeed");
        // The returned stream is a real, usable connection.
        stream.write_all(b"ping").unwrap();
        drop(stream);
    }

    #[test]
    fn interruptible_dial_completes_a_slow_attempt() {
        // Regression: the old slice loop re-dialed every 50 ms, so an
        // attempt slower than one slice could never succeed. One attempt
        // runs with the whole budget now.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            for _ in 0..2 {
                if listener.accept().is_err() {
                    break;
                }
            }
        });
        // A dial slower than the old 50 ms slice: 300 ms of "handshake".
        let stream =
            interruptible_dial(Duration::from_secs(5), &AtomicBool::new(false), move || {
                thread::sleep(Duration::from_millis(300));
                TcpStream::connect(("127.0.0.1", port))
            })
            .expect("a slow attempt within budget must succeed");
        drop(stream);
    }

    #[test]
    fn interruptible_dial_abandons_on_cancel() {
        // A dial that would hang for the whole budget must still notice
        // the cancel flag within one poll slice, the way a Happy Eyeballs
        // loser does when the winner lands.
        let cancel = AtomicBool::new(false);
        thread::scope(|s| {
            s.spawn(|| {
                thread::sleep(Duration::from_millis(100));
                cancel.store(true, Ordering::SeqCst);
            });
            let start = Instant::now();
            let err = interruptible_dial(Duration::from_secs(30), &cancel, || {
                // Hangs like a blackholed connect; the return value is
                // discarded once the caller abandons it.
                thread::sleep(Duration::from_secs(10));
                Err(io::Error::other("unreached"))
            })
            .unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::Interrupted);
            let elapsed = start.elapsed();
            assert!(
                elapsed < Duration::from_secs(5),
                "cancelled dial must abandon promptly; took {elapsed:?}"
            );
        });
    }

    #[test]
    fn interruptible_dial_uses_the_full_budget() {
        // One attempt gets the whole timeout: a dial that never answers
        // must fail with TimedOut after ~budget, not after one slice.
        let start = Instant::now();
        let err = interruptible_dial(Duration::from_millis(300), &AtomicBool::new(false), || {
            thread::sleep(Duration::from_secs(30));
            Err(io::Error::other("unreached"))
        })
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(250),
            "must wait out the budget; returned after {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "must not overshoot the budget; took {elapsed:?}"
        );
    }

    #[test]
    fn interruptible_dial_propagates_dial_errors() {
        // A refused dial surfaces immediately, not after the budget.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let start = Instant::now();
        let err = interruptible_dial(
            Duration::from_secs(30),
            &AtomicBool::new(false),
            move || TcpStream::connect(("127.0.0.1", port)),
        )
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::ConnectionRefused);
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "refusal must surface fast"
        );
    }

    // -- local test server --------------------------------------------------

    /// Read one request (head plus any framed body) from `stream`. Test
    /// servers never need bodies, but the client may still be writing one
    /// when the response is ready; draining it avoids a RST race that
    /// would otherwise surface as ECONNRESET on the client.
    fn read_request(stream: &mut TcpStream) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        let head_end = loop {
            match stream.read(&mut tmp) {
                Ok(0) | Err(_) => break buf.len(),
                Ok(n) => {
                    buf.extend_from_slice(&tmp[..n]);
                    if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        break i + 4;
                    }
                }
            }
        };
        let content_length = String::from_utf8_lossy(&buf[..head_end])
            .lines()
            .filter_map(|l| l.split_once(':'))
            .find(|(name, _)| name.trim().eq_ignore_ascii_case("content-length"))
            .and_then(|(_, v)| v.trim().parse::<usize>().ok())
            .unwrap_or(0);
        let mut remaining = content_length.saturating_sub(buf.len() - head_end);
        while remaining > 0 {
            let want = remaining.min(tmp.len());
            let n = match stream.read(&mut tmp[..want]) {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            remaining -= n;
        }
        buf
    }

    /// The request-target of the raw request bytes (absolute-form).
    fn raw_request_target(raw: &[u8]) -> &str {
        let line_end = raw.iter().position(|&b| b == b'\r').unwrap_or(raw.len());
        let line = &raw[..line_end];
        let mut parts = line.split(|&b| b == b' ');
        parts.next();
        let target = parts.next().unwrap_or(b"/");
        std::str::from_utf8(target).unwrap_or("/")
    }

    /// Path of an absolute-form target, e.g. `/r1` of
    /// `http://127.0.0.1:PORT/r1`. Panic-free: a malformed request line
    /// yields `/` rather than underflowing the index arithmetic.
    fn request_path(raw: &[u8]) -> &str {
        let target = raw_request_target(raw);
        let after_scheme = target.split("://").nth(1).unwrap_or(target);
        let path = after_scheme
            .split('/')
            .skip(1)
            .collect::<Vec<_>>()
            .join("/");
        // `path` is derived from `after_scheme`, which is a suffix of
        // `target`: the path starts `path.len() + 1` bytes before the end
        // of `target` (the +1 is the `/` separator). Checked arithmetic so
        // a degenerate target can never panic.
        match target
            .len()
            .checked_sub(path.len())
            .and_then(|n| n.checked_sub(1))
        {
            Some(idx) => &target[idx..],
            None => "/",
        }
    }

    /// Spawn a server handling `n` connections; `handler` maps raw request
    /// bytes to raw response bytes. Returns the base URL.
    fn spawn_server<F>(n: usize, handler: F) -> String
    where
        F: Fn(&[u8]) -> Vec<u8> + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            tx.send(()).unwrap();
            for _ in 0..n {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let req = read_request(&mut stream);
                let resp = handler(&req);
                let _ = stream.write_all(&resp);
                // Drop: closes the connection (servers here never reuse).
            }
        });
        rx.recv().unwrap();
        format!("http://127.0.0.1:{port}")
    }

    fn head(status: u16, headers: &[(&str, &str)]) -> Vec<u8> {
        let pairs: Vec<(&str, &[u8])> = headers.iter().map(|(n, v)| (*n, v.as_bytes())).collect();
        http1::serialize_response_head(HttpVersion::HTTP_11, status, "OK", &pairs).unwrap()
    }

    fn has_header(raw: &[u8], name: &str, value: &str) -> bool {
        let raw = String::from_utf8_lossy(raw).to_ascii_lowercase();
        raw.lines().any(|l| {
            l.starts_with(&format!("{}:", name.to_ascii_lowercase()))
                && l.contains(&value.to_ascii_lowercase())
        })
    }

    #[test]
    fn basic_get_sends_expected_head() {
        let base = spawn_server(1, |req| {
            // Echo the request head back as the body for assertions.
            let end = req
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|i| i + 4)
                .unwrap_or(req.len());
            let mut resp = head(200, &[("Content-Length", &req[..end].len().to_string())]);
            resp.extend_from_slice(&req[..end]);
            resp
        });
        let port = base.rsplit(':').next().unwrap();
        let resp = fetch(&get_request(&format!("{base}/hello?q=1"))).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.url.serialize(), format!("{base}/hello?q=1"));
        let echo = String::from_utf8_lossy(&resp.body);
        assert!(
            echo.starts_with(&format!("GET {base}/hello?q=1 HTTP/1.1")),
            "{echo}"
        );
        assert!(echo.contains(&format!("Host: 127.0.0.1:{port}")), "{echo}");
        assert!(echo.contains("User-Agent: Bowser/0.1"), "{echo}");
    }

    #[test]
    fn custom_user_agent_not_duplicated() {
        let base = spawn_server(1, |req| {
            // Count User-Agent lines in the request and report via a header.
            let count = String::from_utf8_lossy(req)
                .lines()
                .filter(|l| l.to_ascii_lowercase().starts_with("user-agent:"))
                .count();
            let mut resp = head(
                200,
                &[("Content-Length", "1"), ("X-UA-Count", &count.to_string())],
            );
            resp.extend_from_slice(b"x");
            resp
        });
        let mut req = get_request(&format!("{base}/"));
        req.headers
            .push(("User-Agent".to_string(), "TestAgent/9".to_string()));
        let resp = fetch(&req).unwrap();
        assert_eq!(resp.header("x-ua-count"), Some("1"));
    }

    #[test]
    fn redirect_chain() {
        let base = spawn_server(3, move |req| match request_path(req) {
            "/r1" => head(302, &[("Location", "/r2")]),
            "/r2" => head(301, &[("Location", "/r3")]),
            _ => {
                let mut r = head(200, &[("Content-Length", "5")]);
                r.extend_from_slice(b"final");
                r
            }
        });
        let resp = fetch(&get_request(&format!("{base}/r1"))).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, b"final");
        assert_eq!(resp.url.serialize(), format!("{base}/r3"));
    }

    #[test]
    fn trace_records_redirect_chain_in_order() {
        let base = spawn_server(3, move |req| match request_path(req) {
            "/r1" => head(302, &[("Location", "/r2")]),
            "/r2" => head(301, &[("Location", "/r3")]),
            _ => {
                let mut r = head(200, &[("Content-Length", "5")]);
                r.extend_from_slice(b"final");
                r
            }
        });
        let (resp, trace) = fetch_with_trace(&get_request(&format!("{base}/r1")), None).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, b"final");
        let expected: Vec<Url> = ["/r1", "/r2", "/r3"]
            .iter()
            .map(|p| url(&format!("{base}{p}")))
            .collect();
        assert_eq!(trace.hops, expected);
        // The final URL matches the last hop, as before.
        assert_eq!(&resp.url, trace.hops.last().unwrap());
    }

    #[test]
    fn trace_single_hop_without_redirects() {
        let base = spawn_server(1, |_| {
            let mut r = head(200, &[("Content-Length", "2")]);
            r.extend_from_slice(b"ok");
            r
        });
        let (resp, trace) = fetch_with_trace(&get_request(&format!("{base}/plain")), None).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(trace.hops, vec![url(&format!("{base}/plain"))]);
        assert_eq!(&resp.url, trace.hops.last().unwrap());
    }

    #[test]
    fn trace_records_method_rewrite_hops() {
        // The trace still lists every fetched URL even when the method
        // was rewritten along the way.
        let base = spawn_server(2, |req| {
            if request_path(req) == "/post" {
                head(303, &[("Location", "/landing")])
            } else {
                let mut r = head(200, &[("Content-Length", "2")]);
                r.extend_from_slice(b"ok");
                r
            }
        });
        let mut req = get_request(&format!("{base}/post"));
        req.method = "POST".to_string();
        req.body = Some(b"data=1".to_vec());
        let (resp, trace) = fetch_with_trace(&req, None).unwrap();
        assert_eq!(resp.status, 200);
        let expected: Vec<Url> = ["/post", "/landing"]
            .iter()
            .map(|p| url(&format!("{base}{p}")))
            .collect();
        assert_eq!(trace.hops, expected);
    }

    /// Base URL of the request's own absolute-form target, with the
    /// host swapped to `localhost`: the test server listens on
    /// `127.0.0.1`, so a redirect to the `localhost` spelling is a
    /// cross-origin hop that still lands on the same server.
    fn req_target_host_local(raw: &[u8]) -> String {
        let target = raw_request_target(raw).replacen("127.0.0.1", "localhost", 1);
        if !target.starts_with("http://") {
            return target;
        }
        let end = target[7..].find('/').map(|i| i + 7).unwrap_or(target.len());
        target[..end].to_string()
    }

    /// w77: per-hop credentials on redirect chains, as reported by
    /// `fetch_with_trace_credentials`. The flag is an OR over every
    /// hop — one credentialed hop taints the chain — never a
    /// property of the initial URL alone.
    #[test]
    fn trace_credentials_reports_any_hop() {
        // (a) Scratch-jar ride: no jar was passed, yet the redirect
        // hop's `Set-Cookie` is sent on the final hop, so the chain
        // is credentialed even though its first hop was not.
        let base = spawn_server(8, |req| match request_path(req) {
            "/set" => head(
                302,
                &[("Location", "/echo"), ("Set-Cookie", "session=abc; Path=/")],
            ),
            _ => {
                let s = String::from_utf8_lossy(req);
                let cookie = s
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("cookie:"))
                    .map(|l| l["cookie:".len()..].trim().to_owned())
                    .unwrap_or_else(|| "none".to_owned());
                let mut r = head(200, &[("Content-Length", &cookie.len().to_string())]);
                r.extend_from_slice(cookie.as_bytes());
                r
            }
        });
        let (_, _, creds) =
            fetch_with_trace_credentials(&get_request(&format!("{base}/set")), None).unwrap();
        assert!(creds, "scratch-jar cookie riding the final hop must count");
        let (resp, _, creds) =
            fetch_with_trace_credentials(&get_request(&format!("{base}/set")), None).unwrap();
        assert_eq!(String::from_utf8_lossy(&resp.body), "session=abc");
        assert!(creds);

        // (b) Seeded jar cookie sent on a single hop, and (c) no
        // credentials at all on a plain hop.
        let base = spawn_server(8, |req| match request_path(req) {
            "/set" => head(200, &[("Set-Cookie", "k=v; Path=/")]),
            _ => {
                let mut r = head(200, &[("Content-Length", "2")]);
                r.extend_from_slice(b"ok");
                r
            }
        });
        let mut jar = CookieJar::new();
        fetch_with_jar(&get_request(&format!("{base}/set")), Some(&mut jar)).unwrap();
        let (_, _, creds) =
            fetch_with_trace_credentials(&get_request(&format!("{base}/plain")), Some(&mut jar))
                .unwrap();
        assert!(creds, "seeded jar cookie sent on the hop must count");
        let mut empty = CookieJar::new();
        let (_, _, creds) =
            fetch_with_trace_credentials(&get_request(&format!("{base}/plain")), Some(&mut empty))
                .unwrap();
        assert!(!creds, "no cookie, no auth on any hop must not count");

        // (d) `Authorization` on a single hop counts.
        let mut req = get_request(&format!("{base}/plain"));
        req.headers
            .push(("Authorization".to_string(), "Bearer x".to_string()));
        let (_, _, creds) = fetch_with_trace_credentials(&req, None).unwrap();
        assert!(creds, "an Authorization header sent on a hop must count");

        // (e) `Authorization` stripped on a cross-origin redirect
        // still counts: the FIRST hop sent it, even though the final
        // hop (which echoes auth presence) did not receive it.
        let base = spawn_server(8, |req| match request_path(req) {
            "/redir" => {
                let local = req_target_host_local(req);
                head(302, &[("Location", &format!("{local}/final"))])
            }
            _ => {
                let s = String::from_utf8_lossy(req);
                let seen = if s.to_ascii_lowercase().contains("authorization:") {
                    "auth"
                } else {
                    "noauth"
                };
                let mut r = head(200, &[("Content-Length", &seen.len().to_string())]);
                r.extend_from_slice(seen.as_bytes());
                r
            }
        });
        let mut req = get_request(&format!("{base}/redir"));
        req.headers
            .push(("Authorization".to_string(), "Bearer x".to_string()));
        let (resp, _, creds) = fetch_with_trace_credentials(&req, None).unwrap();
        assert_eq!(String::from_utf8_lossy(&resp.body), "noauth");
        assert!(
            creds,
            "auth sent on the initial hop must taint the chain even after a cross-origin strip"
        );

        // (f) Cross-origin start, same-origin finish, seeded cookie
        // for the final host only: the initial hop sent nothing, the
        // final hop sent the seeded cookie, so the chain counts.
        let base = spawn_server(8, |req| match request_path(req) {
            "/set" => head(200, &[("Set-Cookie", "fin=1; Path=/")]),
            "/redir" => {
                let local = req_target_host_local(req);
                head(302, &[("Location", &format!("{local}/echo"))])
            }
            _ => {
                let s = String::from_utf8_lossy(req);
                let cookie = s
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("cookie:"))
                    .map(|l| l["cookie:".len()..].trim().to_owned())
                    .unwrap_or_else(|| "none".to_owned());
                let mut r = head(200, &[("Content-Length", &cookie.len().to_string())]);
                r.extend_from_slice(cookie.as_bytes());
                r
            }
        });
        let base_local = base.replacen("127.0.0.1", "localhost", 1);
        let mut jar = CookieJar::new();
        fetch_with_jar(&get_request(&format!("{base_local}/set")), Some(&mut jar)).unwrap();
        let (resp, _, creds) =
            fetch_with_trace_credentials(&get_request(&format!("{base}/redir")), Some(&mut jar))
                .unwrap();
        assert_eq!(String::from_utf8_lossy(&resp.body), "fin=1");
        assert!(
            creds,
            "a cookie sent only on the final hop must count, whatever the initial hop did"
        );
    }

    #[test]
    fn redirect_303_rewrites_post_to_get() {
        let base = spawn_server(2, |req| {
            // Record the method of each request in the response body.
            let line_end = req.iter().position(|&b| b == b'\r').unwrap_or(0);
            let request_line = String::from_utf8_lossy(&req[..line_end]).into_owned();
            if request_path(req) == "/post" {
                head(303, &[("Location", "/landing")])
            } else {
                let mut r = head(200, &[("Content-Length", &request_line.len().to_string())]);
                r.extend_from_slice(request_line.as_bytes());
                // Also reflect whether a body was framed.
                r
            }
        });
        let mut req = get_request(&format!("{base}/post"));
        req.method = "POST".to_string();
        req.body = Some(b"data=1".to_vec());
        let resp = fetch(&req).unwrap();
        assert_eq!(resp.status, 200);
        let seen = String::from_utf8_lossy(&resp.body);
        assert!(seen.starts_with("GET "), "{seen}");
        assert!(seen.contains("/landing"), "{seen}");
        assert_eq!(resp.url.serialize(), format!("{base}/landing"));
    }

    /// Drive a POST through a 301/302 redirect; the follow-up request must
    /// be a bodyless GET, per WHATWG fetch.
    fn post_redirect_rewrite(status: u16) {
        let base = spawn_server(2, move |req| {
            if request_path(req) == "/post" {
                head(status, &[("Location", "/landing")])
            } else {
                // Reflect the method and whether a body was framed.
                let s = String::from_utf8_lossy(req);
                let method = s.split_whitespace().next().unwrap_or("").to_owned();
                let cl = s
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                    .map(|l| l.to_owned())
                    .unwrap_or_else(|| "no-content-length".to_owned());
                let seen = format!("{method} {cl}");
                let mut r = head(200, &[("Content-Length", &seen.len().to_string())]);
                r.extend_from_slice(seen.as_bytes());
                r
            }
        });
        let mut req = get_request(&format!("{base}/post"));
        req.method = "POST".to_string();
        req.body = Some(b"data=1".to_vec());
        let resp = fetch(&req).unwrap();
        assert_eq!(resp.status, 200);
        let seen = String::from_utf8_lossy(&resp.body);
        assert!(
            seen.starts_with("GET no-content-length"),
            "301/302 must rewrite POST to a bodyless GET, saw: {seen}"
        );
        assert_eq!(resp.url.serialize(), format!("{base}/landing"));
    }

    #[test]
    fn redirect_301_rewrites_post_to_get() {
        post_redirect_rewrite(301);
    }

    #[test]
    fn redirect_302_rewrites_post_to_get() {
        post_redirect_rewrite(302);
    }

    #[test]
    fn redirect_301_keeps_get() {
        // GET stays GET through 301/302 (no rewrite needed, no body).
        let base = spawn_server(2, |req| match request_path(req) {
            "/r1" => head(302, &[("Location", "/r2")]),
            _ => {
                let s = String::from_utf8_lossy(req);
                let method = s.split_whitespace().next().unwrap_or("").to_owned();
                let mut r = head(200, &[("Content-Length", &method.len().to_string())]);
                r.extend_from_slice(method.as_bytes());
                r
            }
        });
        let resp = fetch(&get_request(&format!("{base}/r1"))).unwrap();
        assert_eq!(String::from_utf8_lossy(&resp.body), "GET");
    }

    #[test]
    fn redirect_307_keeps_post_and_body() {
        let base = spawn_server(2, |req| {
            if request_path(req) == "/post" {
                head(307, &[("Location", "/echo")])
            } else {
                // Echo back method + content-length as seen.
                let s = String::from_utf8_lossy(req);
                let method = s.split_whitespace().next().unwrap_or("").to_owned();
                let cl = s
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                    .unwrap_or("content-length: missing")
                    .to_owned();
                let body = format!("{method} {cl}");
                let mut r = head(200, &[("Content-Length", &body.len().to_string())]);
                r.extend_from_slice(body.as_bytes());
                r
            }
        });
        let mut req = get_request(&format!("{base}/post"));
        req.method = "POST".to_string();
        req.body = Some(b"data=1".to_vec());
        let resp = fetch(&req).unwrap();
        let seen = String::from_utf8_lossy(&resp.body);
        assert!(
            seen.starts_with("POST ") && seen.to_ascii_lowercase().contains("content-length: 6"),
            "{seen}"
        );
    }

    #[test]
    fn too_many_redirects() {
        let base = spawn_server(MAX_REDIRECTS + 1, |_| head(302, &[("Location", "/loop")]));
        match fetch(&get_request(&format!("{base}/loop"))) {
            Err(FetchError::TooManyRedirects) => {}
            other => panic!("expected TooManyRedirects, got {other:?}"),
        }
    }

    #[test]
    fn redirect_without_location() {
        let base = spawn_server(1, |_| head(301, &[("Content-Length", "0")]));
        assert!(matches!(
            fetch(&get_request(&format!("{base}/x"))),
            Err(FetchError::RedirectLocationMissing)
        ));
    }

    #[test]
    fn cookies_roundtrip() {
        let base = spawn_server(3, |req| {
            if request_path(req) == "/set" {
                head(
                    302,
                    &[("Location", "/echo"), ("Set-Cookie", "session=abc; Path=/")],
                )
            } else {
                let s = String::from_utf8_lossy(req);
                let cookie = s
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("cookie:"))
                    .map(|l| l["cookie:".len()..].trim().to_owned())
                    .unwrap_or_else(|| "none".to_owned());
                let mut r = head(200, &[("Content-Length", &cookie.len().to_string())]);
                r.extend_from_slice(cookie.as_bytes());
                r
            }
        });
        let mut jar = CookieJar::new();
        let resp = fetch_with_jar(&get_request(&format!("{base}/set")), Some(&mut jar)).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(String::from_utf8_lossy(&resp.body), "session=abc");
        assert_eq!(jar.len(), 1);
        // The jar persists: a second fetch sends the cookie without a redirect.
        let base2 = base.clone();
        let resp2 = fetch_with_jar(&get_request(&format!("{base2}/echo")), Some(&mut jar)).unwrap();
        assert_eq!(String::from_utf8_lossy(&resp2.body), "session=abc");
    }

    #[test]
    fn subresource_context_withholds_lax_cookies_cross_site() {
        let base = spawn_server(4, |req| {
            if request_path(req) == "/set" {
                head(200, &[("Set-Cookie", "l=2; SameSite=Lax; Path=/")])
            } else {
                let s = String::from_utf8_lossy(req);
                let cookie = s
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("cookie:"))
                    .map(|l| l["cookie:".len()..].trim().to_owned())
                    .unwrap_or_else(|| "none".to_owned());
                let mut r = head(200, &[("Content-Length", &cookie.len().to_string())]);
                r.extend_from_slice(cookie.as_bytes());
                r
            }
        });
        let mut jar = CookieJar::new();
        fetch_with_jar(&get_request(&format!("{base}/set")), Some(&mut jar)).unwrap();
        assert_eq!(jar.len(), 1);

        // Default (top-level navigation) context: Lax rides along.
        let resp = fetch_with_jar(&get_request(&format!("{base}/echo")), Some(&mut jar)).unwrap();
        assert_eq!(String::from_utf8_lossy(&resp.body), "l=2");

        // Subresource context with a foreign top host: Lax is withheld.
        let cross = NavigationContext {
            top_host: Some("other.com".to_string()),
            top_secure: Some(false),
            top_level_safe_navigation: false,
        };
        let resp = fetch_with_jar_in_context(
            &get_request(&format!("{base}/echo")),
            Some(&mut jar),
            &cross,
        )
        .unwrap();
        assert_eq!(String::from_utf8_lossy(&resp.body), "none");

        // Subresource context with the same top host: Lax is sent.
        let same = NavigationContext {
            top_host: Some("127.0.0.1".to_string()),
            top_secure: Some(false),
            top_level_safe_navigation: false,
        };
        let resp =
            fetch_with_jar_in_context(&get_request(&format!("{base}/echo")), Some(&mut jar), &same)
                .unwrap();
        assert_eq!(String::from_utf8_lossy(&resp.body), "l=2");
    }

    #[test]
    fn redirect_chain_keeps_initial_top_for_samesite() {
        // A top-level navigation from A to B must not re-root the
        // SameSite top context at B: B's Strict cookies stay withheld
        // on the cross-site redirect hop.
        let base_b = spawn_server(2, |req| {
            if request_path(req) == "/set" {
                head(200, &[("Set-Cookie", "s=1; SameSite=Strict; Path=/")])
            } else {
                let s = String::from_utf8_lossy(req);
                let cookie = s
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("cookie:"))
                    .map(|l| l["cookie:".len()..].trim().to_owned())
                    .unwrap_or_else(|| "none".to_owned());
                let mut r = head(200, &[("Content-Length", &cookie.len().to_string())]);
                r.extend_from_slice(cookie.as_bytes());
                r
            }
        });
        // Same server, different hostname: localhost vs 127.0.0.1 is
        // cross-site.
        let base_b_local = base_b.replacen("127.0.0.1", "localhost", 1);
        let location = format!("{base_b_local}/echo");
        let base_a = spawn_server(1, move |_req| head(302, &[("Location", &location)]));

        let mut jar = CookieJar::new();
        // Seed B's Strict cookie with a direct top-level visit to B.
        let resp =
            fetch_with_jar(&get_request(&format!("{base_b_local}/set")), Some(&mut jar)).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(jar.len(), 1);

        // Top-level navigation to A, redirected to B: the top context
        // stays A, so B's Strict cookie must not ride the redirect hop.
        let resp = fetch_with_jar(&get_request(&format!("{base_a}/")), Some(&mut jar)).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(String::from_utf8_lossy(&resp.body), "none");
    }

    #[test]
    fn chunked_body_decoded() {
        let base = spawn_server(1, |_| {
            let mut r = head(200, &[("Transfer-Encoding", "chunked")]);
            r.extend_from_slice(b"5\r\nhello\r\n8\r\n chunked\r\n0\r\n\r\n");
            r
        });
        let resp = fetch(&get_request(&format!("{base}/"))).unwrap();
        assert_eq!(resp.body, b"hello chunked");
    }

    #[test]
    fn chunked_trailers_surfaced_in_headers() {
        let base = spawn_server(1, |_| {
            let mut r = head(
                200,
                &[("Transfer-Encoding", "chunked"), ("Trailer", "X-Sum")],
            );
            r.extend_from_slice(b"5\r\nhello\r\n0\r\nX-Sum: abc\r\n\r\n");
            r
        });
        let resp = fetch(&get_request(&format!("{base}/"))).unwrap();
        assert_eq!(resp.body, b"hello");
        // The trailer arrives as a response header, after the head fields.
        assert_eq!(resp.header("x-sum"), Some("abc"));
        let names: Vec<&str> = resp.headers.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["Transfer-Encoding", "Trailer", "X-Sum"]);
    }

    #[test]
    fn chunked_body_cap_enforced_while_streaming() {
        // Drip-feed a chunked body whose decoded payload (32 bytes)
        // exceeds a 16-byte cap: the fetch layer must fail with
        // BodyTooLarge incrementally, not after buffering everything.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let _ = read_request(&mut s);
            let _ = s.write_all(&head(200, &[("Transfer-Encoding", "chunked")]));
            for chunk in ["01234567", "89abcdef", "01234567", "89abcdef"] {
                let _ = s.write_all(format!("8\r\n{chunk}\r\n").as_bytes());
                std::thread::sleep(Duration::from_millis(5));
            }
            let _ = s.write_all(b"0\r\n\r\n");
        });
        let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        // The test server waits for a request head before responding.
        stream
            .write_all(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
            .unwrap();
        let mut conn = Conn::Plain(stream);
        let limits = Limits {
            max_body_bytes: 16,
            ..Limits::default()
        };
        let parsed = read_head(&mut conn, "GET", &limits, test_deadline()).unwrap();
        match read_body(&mut conn, parsed, &limits, test_deadline()) {
            Err(FetchError::BodyTooLarge) => {}
            other => panic!("expected BodyTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn tunnel_framing_in_body_reader_fails_closed() {
        // `fetch` never issues CONNECT, but if the proxy path ever
        // evolves to construct BodySpec::Tunnel here, the body reader
        // must return an error — never panic, and never misparse the
        // tunnel stream as a message body.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let mut conn = Conn::Plain(TcpStream::connect(("127.0.0.1", port)).unwrap());
        let parsed = ParsedHead {
            status: 200,
            fields: Vec::new(),
            head_len: 0,
            headers_seen: 0,
            spec: BodySpec::Tunnel,
            buf: Vec::new(),
            persistent: false,
        };
        match read_body(&mut conn, parsed, &Limits::default(), test_deadline()) {
            Err(FetchError::Proxy(_)) => {}
            other => panic!("expected Proxy error, got {other:?}"),
        }
    }

    #[test]
    fn interim_1xx_skipped() {
        let base = spawn_server(1, |_| {
            let mut r = head(103, &[("Link", "</style.css>; rel=preload")]);
            r.extend_from_slice(&head(200, &[("Content-Length", "2")]));
            r.extend_from_slice(b"ok");
            r
        });
        let resp = fetch(&get_request(&format!("{base}/"))).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, b"ok");
    }

    #[test]
    fn switching_protocols_101_is_rejected() {
        // This client never sends `Upgrade`: a 101 means the server
        // switched to a protocol we do not speak, so the bytes after
        // the head must not be parsed as a response — fail closed.
        let base = spawn_server(1, |_| {
            let mut r = head(101, &[("Upgrade", "websocket")]);
            r.extend_from_slice(b"switched-protocol bytes, not HTTP\r\n\r\n");
            r
        });
        let err = fetch(&get_request(&format!("{base}/"))).unwrap_err();
        assert!(
            matches!(err, FetchError::SwitchingProtocols),
            "101 must fail closed, got {err:?}"
        );
    }

    #[test]
    fn stalled_server_times_out() {
        // Accept the connection and read the request, then never respond.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_request(&mut stream);
            std::thread::park(); // never respond
        });
        let req = get_request(&format!("http://127.0.0.1:{port}/"));
        let timeouts = Timeouts {
            connect: Duration::from_secs(5),
            rw: Duration::from_millis(300),
            fetch_total: Duration::from_secs(60),
        };
        match fetch_impl(
            &req,
            None,
            timeouts,
            &NavigationContext::default(),
            None,
            None,
        )
        .map(|(response, _, _)| response)
        {
            Err(FetchError::Timeout(_)) => {}
            other => panic!("expected FetchError::Timeout, got {other:?}"),
        }
    }

    #[test]
    fn drip_feeding_peer_is_cut_off_by_total_deadline() {
        // A peer that drips one byte every 200 ms keeps the per-read
        // socket timeout from ever firing (each read succeeds well within
        // it), but the total fetch deadline is an absolute wall-clock
        // bound from request-send to body-complete: the fetch must fail
        // with FetchError::Timeout instead of hanging for the drip's full
        // duration.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_request(&mut stream);
            let head = head(200, &[("Content-Length", "16")]);
            let _ = stream.write_all(&head);
            // Drip one byte per 200 ms: without a total deadline the 16
            // announced bytes would take 3.2 s; the deadline fires first.
            for _ in 0..16 {
                std::thread::sleep(Duration::from_millis(200));
                if stream.write_all(b"x").is_err() {
                    break;
                }
            }
        });
        let req = get_request(&format!("http://127.0.0.1:{port}/"));
        // Test-only override: a short total deadline. The production
        // constant (TOTAL_FETCH_TIMEOUT) stays generous.
        let timeouts = Timeouts {
            connect: Duration::from_secs(5),
            rw: Duration::from_secs(60),
            fetch_total: Duration::from_millis(500),
        };
        let start = std::time::Instant::now();
        match fetch_impl(
            &req,
            None,
            timeouts,
            &NavigationContext::default(),
            None,
            None,
        )
        .map(|(response, _, _)| response)
        {
            Err(FetchError::Timeout(_)) => {}
            other => panic!("expected FetchError::Timeout, got {other:?}"),
        }
        // Content-Length 16 adds one second of size slack, so the fetch
        // must die around 1.5 s in — far short of the 3.2 s drip.
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(3),
            "deadline should have fired well before the drip finished, took {elapsed:?}"
        );
    }

    #[test]
    fn request_path_never_panics() {
        // Degenerate request bytes must not underflow the index math.
        assert_eq!(request_path(b""), "/");
        assert_eq!(request_path(b"G"), "/");
        assert_eq!(request_path(b"GET"), "/");
        assert_eq!(request_path(b"GET /\r\n\r\n"), "/");
        assert_eq!(
            request_path(b"GET http://127.0.0.1:1/r1 HTTP/1.1\r\n\r\n"),
            "/r1"
        );
        assert_eq!(
            request_path(b"GET /rel/path?q=1 HTTP/1.1\r\n\r\n"),
            "/rel/path?q=1"
        );
    }

    #[test]
    fn close_delimited_body() {
        // No Content-Length / Transfer-Encoding: read until close.
        let base = spawn_server(1, |_| {
            let mut r = head(200, &[("Content-Type", "text/plain")]);
            r.extend_from_slice(b"streamed");
            r
        });
        let resp = fetch(&get_request(&format!("{base}/"))).unwrap();
        assert_eq!(resp.body, b"streamed");
    }

    #[test]
    fn head_request_reads_no_body() {
        let base = spawn_server(1, |_| {
            // Claim a body but send none, as a real HEAD response does.
            head(200, &[("Content-Length", "5")])
        });
        let mut req = get_request(&format!("{base}/"));
        req.method = "HEAD".to_string();
        let resp = fetch(&req).unwrap();
        assert_eq!(resp.status, 200);
        assert!(resp.body.is_empty());
    }

    #[test]
    fn charset_decoding_over_wire() {
        let base = spawn_server(1, |_| {
            let mut r = head(
                200,
                &[
                    ("Content-Type", "text/plain; charset=windows-1252"),
                    ("Content-Length", "4"),
                ],
            );
            r.extend_from_slice(b"caf\xe9");
            r
        });
        let resp = fetch(&get_request(&format!("{base}/"))).unwrap();
        assert_eq!(resp.body, b"caf\xe9");
        assert_eq!(resp.text(), "café");
    }

    #[test]
    fn body_too_large_rejected_early() {
        let huge = (64 * 1024 * 1024).to_string();
        let base = spawn_server(1, move |_| {
            // The client must reject this from the head alone.
            head(200, &[("Content-Length", &huge)])
        });
        assert!(matches!(
            fetch(&get_request(&format!("{base}/"))),
            Err(FetchError::BodyTooLarge)
        ));
    }

    #[test]
    fn request_head_bytes_managed_headers() {
        let req_url = url("http://example.com:8080/a?b=c#frag");
        let extra = vec![
            ("X-Custom".to_string(), "v".to_string()),
            ("Host".to_string(), "evil.com".to_string()),
            ("Content-Length".to_string(), "999".to_string()),
            ("Connection".to_string(), "keep-alive".to_string()),
        ];
        let bytes = request_head_bytes(
            "POST",
            &req_url,
            &extra,
            Some(b"hi"),
            Some("a=b"),
            false,
            None,
            false,
        )
        .unwrap();
        let s = String::from_utf8(bytes).unwrap();
        // Absolute-form target, no fragment.
        assert!(
            s.starts_with("POST http://example.com:8080/a?b=c HTTP/1.1\r\n"),
            "{s}"
        );
        assert!(s.contains("Host: example.com:8080\r\n"), "{s}");
        assert!(!s.contains("evil.com"), "{s}");
        assert!(s.contains("Content-Length: 2\r\n"), "{s}");
        assert!(!s.contains("999"), "{s}");
        assert!(!s.contains("keep-alive"), "{s}");
        assert!(s.contains("Cookie: a=b\r\n"), "{s}");
        assert!(s.contains("X-Custom: v\r\n"), "{s}");
    }

    #[test]
    fn request_head_advertises_gzip_deflate() {
        let bytes = request_head_bytes(
            "GET",
            &url("http://example.com/"),
            &[],
            None,
            None,
            false,
            None,
            false,
        )
        .unwrap();
        let s = String::from_utf8(bytes).unwrap();
        assert!(s.contains("Accept-Encoding: gzip, deflate\r\n"), "{s}");
    }

    #[test]
    fn request_head_caller_accept_encoding_wins() {
        let extra = vec![("Accept-Encoding".to_string(), "identity".to_string())];
        let bytes = request_head_bytes(
            "GET",
            &url("http://example.com/"),
            &extra,
            None,
            None,
            false,
            None,
            false,
        )
        .unwrap();
        let s = String::from_utf8(bytes).unwrap();
        assert!(s.contains("Accept-Encoding: identity\r\n"), "{s}");
        assert!(!s.contains("gzip, deflate"), "{s}");
    }

    #[test]
    fn request_head_default_port_omitted() {
        let bytes = request_head_bytes(
            "GET",
            &url("https://example.com/"),
            &[],
            None,
            None,
            false,
            None,
            false,
        )
        .unwrap();
        let s = String::from_utf8(bytes).unwrap();
        assert!(s.contains("Host: example.com\r\n"), "{s}");
    }

    #[test]
    fn request_head_strips_userinfo() {
        let bytes = request_head_bytes(
            "GET",
            &url("http://zzuser:zzpass@example.com:8080/a?x=1#frag"),
            &[],
            None,
            None,
            false,
            None,
            false,
        )
        .unwrap();
        let s = String::from_utf8(bytes).unwrap();
        assert!(
            s.starts_with("GET http://example.com:8080/a?x=1 HTTP/1.1\r\n"),
            "{s}"
        );
        assert!(!s.contains("zzuser"), "{s}");
        assert!(!s.contains("zzpass"), "{s}");
        // Username-only URLs are stripped too.
        let bytes = request_head_bytes(
            "GET",
            &url("http://zzuser@example.com/"),
            &[],
            None,
            None,
            false,
            None,
            false,
        )
        .unwrap();
        let s = String::from_utf8(bytes).unwrap();
        assert!(s.starts_with("GET http://example.com/ HTTP/1.1\r\n"), "{s}");
        assert!(!s.contains("zzuser"), "{s}");
    }

    #[test]
    fn request_target_without_userinfo_unchanged() {
        assert_eq!(
            request_target(&url("http://example.com:8080/a?x=1#frag")),
            "http://example.com:8080/a?x=1"
        );
        assert_eq!(
            request_target(&url("http://user:pw@example.com/a")),
            "http://example.com/a"
        );
    }

    #[test]
    fn userinfo_never_reaches_the_wire() {
        // Capture the raw bytes the client actually sends.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let raw = read_request(&mut stream);
            tx.send(raw).unwrap();
            let _ = stream.write_all(&head(200, &[("Content-Length", "0")]));
        });
        let req = get_request(&format!("http://al1ce:s3cret!@127.0.0.1:{port}/p?q=1"));
        let resp = fetch(&req).unwrap();
        assert_eq!(resp.status, 200);

        let raw = rx.recv().unwrap();
        let wire = String::from_utf8_lossy(&raw);
        assert!(!wire.contains("al1ce"), "username leaked: {wire}");
        assert!(!wire.contains("s3cret"), "password leaked: {wire}");
        assert!(!wire.contains('@'), "userinfo separator leaked: {wire}");
        let line_end = raw.iter().position(|&b| b == b'\r').unwrap();
        assert_eq!(
            &raw[..line_end],
            format!("GET http://127.0.0.1:{port}/p?q=1 HTTP/1.1").as_bytes()
        );
        // The Host header and therefore the dial target carry no userinfo.
        assert!(
            has_header(&raw, "host", &format!("127.0.0.1:{port}")),
            "{wire}"
        );
    }

    #[test]
    fn auth_stripped_on_cross_origin_redirect() {
        // Two servers: a.com redirects to b.com; the second request must
        // not carry Authorization.
        let second = spawn_server(1, |req| {
            let s = String::from_utf8_lossy(req);
            let auth = s
                .lines()
                .find(|l| l.to_ascii_lowercase().starts_with("authorization:"))
                .map(|l| l.to_owned())
                .unwrap_or_else(|| "absent".to_owned());
            let mut r = head(200, &[("Content-Length", &auth.len().to_string())]);
            r.extend_from_slice(auth.as_bytes());
            r
        });
        let first = spawn_server(1, {
            let second = second.clone();
            move |_| head(302, &[("Location", &format!("{second}/"))])
        });
        let mut req = get_request(&format!("{first}/"));
        req.headers
            .push(("Authorization".to_string(), "Bearer secret".to_string()));
        let resp = fetch(&req).unwrap();
        assert_eq!(String::from_utf8_lossy(&resp.body), "absent");
    }

    #[test]
    fn has_header_helper_sanity() {
        assert!(has_header(
            b"GET / HTTP/1.1\r\nHost: x\r\n\r\n",
            "host",
            "x"
        ));
        assert!(!has_header(
            b"GET / HTTP/1.1\r\nHost: x\r\n\r\n",
            "host",
            "y"
        ));
    }

    // -- HSTS ---------------------------------------------------------------

    fn hsts_headers(value: &str) -> Vec<(String, String)> {
        vec![("Strict-Transport-Security".to_string(), value.to_string())]
    }

    /// A store holding the policy from one `https` response's STS header.
    fn noted(url_str: &str, sts: &str, now: u64) -> HstsStore {
        let mut store = HstsStore::new();
        store.note_response(&url(url_str), &hsts_headers(sts), now);
        store
    }

    #[test]
    fn hsts_parse_valid() {
        assert_eq!(parse_sts("max-age=31536000"), Some((31536000, false)));
        assert_eq!(parse_sts("  max-age=100  "), Some((100, false)));
        assert_eq!(parse_sts("max-age=0"), Some((0, false)));
    }

    #[test]
    fn hsts_parse_include_subdomains() {
        assert_eq!(
            parse_sts("max-age=100; includeSubDomains"),
            Some((100, true))
        );
        assert_eq!(
            parse_sts("max-age=100;INCLUDEsubDOMAINS"),
            Some((100, true))
        );
        // Unknown directives (preload) are ignored.
        assert_eq!(
            parse_sts("max-age=100; preload; includeSubDomains"),
            Some((100, true))
        );
        // A repeated max-age takes the last value.
        assert_eq!(parse_sts("max-age=1; max-age=99"), Some((99, false)));
    }

    #[test]
    fn hsts_parse_malformed() {
        for bad in [
            "",
            "includeSubDomains",
            "max-age",
            "max-age=",
            "max-age=abc",
            "max-age=-5",
            "max-age=12x",
            "max-age=\"100\"",
            "max-age=99999999999999999999999999", // overflows u64
            "preload",
        ] {
            assert_eq!(parse_sts(bad), None, "{bad}");
        }
    }

    #[test]
    fn hsts_learned_only_over_https() {
        let mut store = HstsStore::new();
        // Plain-HTTP responses must not install policy (RFC 6797 §8.1):
        // an attacker injecting headers on the wire could otherwise
        // plant or erase policy.
        store.note_response(&url("http://a.com/"), &hsts_headers("max-age=100"), 1000);
        assert!(store.policies.is_empty());
        store.note_response(&url("https://a.com/"), &hsts_headers("max-age=100"), 1000);
        assert!(store.policies.contains_key("a.com"));
    }

    #[test]
    fn hsts_max_age_zero_removes_policy() {
        let mut store = noted("https://a.com/", "max-age=100", 1000);
        assert!(store.policies.contains_key("a.com"));
        store.note_response(&url("https://a.com/"), &hsts_headers("max-age=0"), 1001);
        assert!(store.policies.is_empty());
        assert_eq!(store.upgrade_url(&url("http://a.com/"), 1001), None);
    }

    #[test]
    fn hsts_key_borrows_normalized_hosts() {
        // The per-request `upgrade_url` hot path must not allocate a
        // key when the host is already normalized (lowercase, no
        // trailing dot): `key` hands back the `Cow` borrowed.
        assert!(matches!(HstsStore::key("a.com"), Cow::Borrowed("a.com")));
        // Unnormalized hosts take the owned path, and still
        // canonicalize to the same key.
        assert!(matches!(HstsStore::key("A.COM"), Cow::Owned(_)));
        assert_eq!(HstsStore::key("A.COM."), "a.com");
        // The borrowed query path upgrades end to end.
        let mut store = noted("https://A.COM./", "max-age=100", 1000);
        let upgraded = store.upgrade_url(&url("http://a.com/"), 1001);
        assert_eq!(
            upgraded.map(|u| u.scheme().to_string()),
            Some("https".to_string())
        );
    }

    #[test]
    fn hsts_malformed_header_leaves_policy_untouched() {
        let mut store = noted("https://a.com/", "max-age=100", 1000);
        store.note_response(&url("https://a.com/"), &hsts_headers("max-age=bogus"), 1001);
        // The original policy survives and is still enforced.
        assert_eq!(
            store
                .upgrade_url(&url("http://a.com/"), 1001)
                .map(|u| u.serialize()),
            Some("https://a.com/".to_string())
        );
    }

    #[test]
    fn hsts_subdomain_matching() {
        let mut store = noted(
            "https://example.com/",
            "max-age=100; includeSubDomains",
            1000,
        );
        assert!(
            store
                .upgrade_url(&url("http://example.com/"), 1000)
                .is_some()
        );
        assert!(
            store
                .upgrade_url(&url("http://sub.example.com/"), 1000)
                .is_some()
        );
        assert!(
            store
                .upgrade_url(&url("http://deep.sub.example.com/"), 1000)
                .is_some()
        );
        // Unrelated hosts are not covered, including suffix tricks.
        assert_eq!(store.upgrade_url(&url("http://other.com/"), 1000), None);
        assert_eq!(
            store.upgrade_url(&url("http://example.com.evil.com/"), 1000),
            None
        );

        // Without includeSubDomains, subdomains are not covered.
        let mut narrow = noted("https://example.com/", "max-age=100", 1000);
        assert!(
            narrow
                .upgrade_url(&url("http://example.com/"), 1000)
                .is_some()
        );
        assert_eq!(
            narrow.upgrade_url(&url("http://sub.example.com/"), 1000),
            None
        );
    }

    #[test]
    fn hsts_expiry_evicts() {
        let mut store = noted("https://a.com/", "max-age=10", 1000);
        assert!(store.upgrade_url(&url("http://a.com/"), 1005).is_some());
        // At exactly the expiry instant the policy is dead...
        assert_eq!(store.upgrade_url(&url("http://a.com/"), 1010), None);
        // ...and evicted rather than left to accumulate.
        assert!(store.policies.is_empty());
    }

    #[test]
    fn hsts_upgrade_only_applies_to_http() {
        let mut store = noted("https://a.com/", "max-age=100", 1000);
        assert_eq!(store.upgrade_url(&url("https://a.com/"), 1000), None);
        let upgraded = store
            .upgrade_url(&url("http://a.com:8080/p?q=1"), 1000)
            .unwrap();
        // Scheme flips; explicit port, path, and query are preserved.
        assert_eq!(upgraded.serialize(), "https://a.com:8080/p?q=1");
    }

    #[test]
    fn hsts_host_keys_are_case_insensitive() {
        let mut store = noted("https://A.COM/", "max-age=100", 1000);
        assert!(store.upgrade_url(&url("http://a.com/"), 1000).is_some());
        assert!(store.upgrade_url(&url("http://A.COM/"), 1000).is_some());
    }

    #[test]
    fn hsts_host_keys_ignore_trailing_dot() {
        // A policy learned for `example.com` must cover `example.com.`
        // and vice versa: HSTS keys go through psl::normalize_host like
        // cookie domains do.
        let mut store = noted("https://example.com/", "max-age=100", 1000);
        assert!(
            store
                .upgrade_url(&url("http://example.com./"), 1000)
                .is_some()
        );
        let mut store = noted("https://example.com./", "max-age=100", 1000);
        assert!(
            store
                .upgrade_url(&url("http://example.com/"), 1000)
                .is_some()
        );
    }

    #[test]
    fn default_hsts_store_persists_across_calls() {
        // The process-shared default must not be a per-call scratch
        // store: a policy noted under one lock acquisition is visible
        // under the next, the way two fetch_impl calls would see it.
        let https = url("https://hsts-default-persist.example/");
        {
            let mut store = default_hsts()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            store.note_response(&https, &hsts_headers("max-age=31536000"), 1000);
        }
        {
            let mut store = default_hsts()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            assert!(
                store
                    .upgrade_url(&url("http://hsts-default-persist.example/"), 1000)
                    .is_some()
            );
            // A caller-owned store is unaffected by the shared default.
            let mut own = HstsStore::new();
            assert!(
                own.upgrade_url(&url("http://hsts-default-persist.example/"), 1000)
                    .is_none()
            );
        }
    }

    #[test]
    fn hsts_lock_not_held_across_slow_body() {
        // Regression test: the process-shared HSTS store's mutex used to
        // be held for the whole of `fetch_impl`, including a slow body
        // read. Navigation deadlines abandon their worker thread, so a
        // worker stuck in a drip-feed read held the global lock until its
        // socket I/O finished — serializing every later fetch in the
        // process behind it until their own deadlines fired.
        //
        // Drip a 5 s body on one connection; a second fetch against a fast
        // server must complete well before the drip ends.
        let drip = TcpListener::bind("127.0.0.1:0").unwrap();
        let drip_port = drip.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = drip.accept() else {
                return;
            };
            let _ = read_request(&mut stream);
            let head = head(200, &[("Content-Length", "50")]);
            let _ = stream.write_all(&head);
            for _ in 0..50 {
                std::thread::sleep(Duration::from_millis(100));
                if stream.write_all(b"x").is_err() {
                    return;
                }
            }
        });
        let fast = spawn_server(1, |_| {
            let mut resp = head(200, &[("Content-Length", "2")]);
            resp.extend_from_slice(b"ok");
            resp
        });

        // Start the slow fetch on a worker thread and wait until it is
        // inside the body read.
        let slow = std::thread::spawn(move || {
            fetch(&get_request(&format!("http://127.0.0.1:{drip_port}/drip")))
        });
        std::thread::sleep(Duration::from_millis(500));

        let start = std::time::Instant::now();
        let response = fetch(&get_request(&format!("{fast}/"))).expect("fast fetch");
        assert_eq!(response.status, 200);
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "fast fetch blocked behind the slow body read: {:?}",
            start.elapsed()
        );
        slow.join().unwrap().expect("slow fetch");
    }

    #[test]
    fn redirect_downgrade_blocked() {
        let now = 1000;
        let mut hsts = HstsStore::new();
        let err =
            redirect_target(&mut hsts, &url("https://a.com/x"), "http://b.com/y", now).unwrap_err();
        match err {
            FetchError::InsecureRedirect { from, to } => {
                assert_eq!(from, "https://a.com/x");
                assert_eq!(to, "http://b.com/y");
            }
            other => panic!("expected InsecureRedirect, got {other:?}"),
        }
        // A relative Location on an https page resolves within https.
        let next = redirect_target(&mut hsts, &url("https://a.com/x"), "/y", now).unwrap();
        assert_eq!(next.serialize(), "https://a.com/y");
    }

    #[test]
    fn redirect_downgrade_allowed_schemes() {
        let now = 1000;
        let mut hsts = HstsStore::new();
        for (cur, loc) in [
            ("http://a.com/x", "http://b.com/y"),
            ("https://a.com/x", "https://b.com/y"),
            ("http://a.com/x", "https://b.com/y"),
        ] {
            redirect_target(&mut hsts, &url(cur), loc, now)
                .unwrap_or_else(|e| panic!("{cur} -> {loc} must be allowed, got {e:?}"));
        }
    }

    #[test]
    fn redirect_to_hsts_host_upgrades_instead_of_blocking() {
        let now = 1000;
        let mut hsts = noted("https://b.com/", "max-age=100", now);
        // The https page redirects to plaintext b.com, but b.com is
        // HSTS-known: the target upgrades, so there is no downgrade.
        let next =
            redirect_target(&mut hsts, &url("https://a.com/x"), "http://b.com/y", now).unwrap();
        assert_eq!(next.serialize(), "https://b.com/y");
    }

    #[test]
    fn upgrade_then_redirect_trace_composition() {
        // The fetch loop's hop bookkeeping, in order: HSTS-upgrade the
        // hop URL (the trace records the upgraded URL), then resolve the
        // redirect with downgrade protection.
        let now = 1_700_000_000;
        let mut hsts = noted("https://a.com/", "max-age=3600", now);

        // Hop 1: the initial http URL upgrades; the trace records the
        // upgraded URL because it is the URL actually fetched.
        let current = prepare_hop_url(&mut hsts, &url("http://a.com/start"), now);
        let hops = [current.clone()];
        assert_eq!(hops[0].serialize(), "https://a.com/start");

        // The https page redirects to plaintext elsewhere: the chain
        // terminates with InsecureRedirect, and the trace still holds
        // the upgraded initial URL.
        let err = redirect_target(&mut hsts, &current, "http://b.com/next", now).unwrap_err();
        assert!(
            matches!(err, FetchError::InsecureRedirect { .. }),
            "{err:?}"
        );
        assert_eq!(
            hops.iter().map(|u| u.serialize()).collect::<Vec<_>>(),
            ["https://a.com/start"]
        );
    }

    #[test]
    fn hsts_upgrades_http_to_https_on_fetch() {
        // The server tells a plaintext GET from a TLS ClientHello by the
        // first byte: it answers 200 to the former and hangs up on the
        // latter. An HSTS-upgraded fetch must therefore fail with a TLS
        // error (it attempted https); without the policy the same fetch
        // succeeds over plaintext.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            for _ in 0..2 {
                let Ok((mut s, _)) = listener.accept() else {
                    break;
                };
                let mut first = [0u8; 1];
                s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
                let n = s.read(&mut first).unwrap_or(0);
                if n == 1 && first[0] == 0x16 {
                    // TLS handshake record: a plaintext server cannot
                    // continue; hang up so the client fails fast.
                    continue;
                }
                let _ = read_request(&mut s);
                let mut r = head(200, &[("Content-Length", "2")]);
                r.extend_from_slice(b"ok");
                let _ = s.write_all(&r);
            }
        });

        let base = format!("http://127.0.0.1:{port}");
        // Control: no HSTS policy, the plaintext fetch succeeds.
        let resp = fetch(&get_request(&format!("{base}/"))).unwrap();
        assert_eq!(resp.body, b"ok");

        // With a live policy for the host, the client upgrades to https
        // before connecting: the TLS handshake against the plaintext
        // server fails.
        let mut hsts = HstsStore::new();
        hsts.note_response(
            &url("https://127.0.0.1/"),
            &hsts_headers("max-age=3600"),
            now_secs(),
        );
        match fetch_with_hsts(&get_request(&format!("{base}/")), &mut hsts) {
            Err(FetchError::Tls(_)) => {}
            other => panic!("expected TLS failure after HSTS upgrade, got {other:?}"),
        }
    }

    /// Spawn a keep-alive server: counts accepted connections and serves
    /// `handler` per request on each connection. The handler gets the
    /// per-connection request index and returns `(response, keep_serving)`:
    /// an empty response with `keep_serving = false` closes the connection
    /// without answering, which simulates a server that goes away
    /// mid-keep-alive. Returns the base URL and the accepted count.
    fn spawn_keepalive_server<F>(
        handler: F,
    ) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>)
    where
        F: Fn(usize) -> (Vec<u8>, bool) + Send + Sync + 'static,
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let accepted = std::sync::Arc::new(AtomicUsize::new(0));
        let (tx, rx) = mpsc::channel();
        let handler = std::sync::Arc::new(handler);
        std::thread::spawn({
            let accepted = accepted.clone();
            move || {
                tx.send(()).unwrap();
                for stream in listener.incoming() {
                    let Ok(mut stream) = stream else { break };
                    accepted.fetch_add(1, Ordering::SeqCst);
                    let handler = handler.clone();
                    std::thread::spawn(move || {
                        let mut idx = 0usize;
                        loop {
                            let req = read_request(&mut stream);
                            if req.is_empty() {
                                break; // Peer went away.
                            }
                            let (resp, keep) = handler(idx);
                            idx += 1;
                            if stream.write_all(&resp).is_err() {
                                break;
                            }
                            if !keep {
                                break;
                            }
                        }
                    });
                }
            }
        });
        rx.recv().unwrap();
        (format!("http://127.0.0.1:{port}"), accepted)
    }

    fn ok_body(body: &[u8]) -> Vec<u8> {
        let mut r = head(200, &[("Content-Length", &body.len().to_string())]);
        r.extend_from_slice(body);
        r
    }

    #[test]
    fn keepalive_connection_reused_across_fetches() {
        let (base, accepted) = spawn_keepalive_server(|_| (ok_body(b"hi"), true));
        let r1 = fetch(&get_request(&format!("{base}/a"))).unwrap();
        let r2 = fetch(&get_request(&format!("{base}/b"))).unwrap();
        assert_eq!(r1.body, b"hi");
        assert_eq!(r2.body, b"hi");
        assert_eq!(accepted.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn connection_close_response_not_reused() {
        let (base, accepted) = spawn_keepalive_server(|_| {
            let mut r = head(200, &[("Content-Length", "2"), ("Connection", "close")]);
            r.extend_from_slice(b"hi");
            (r, false)
        });
        let r1 = fetch(&get_request(&format!("{base}/a"))).unwrap();
        let r2 = fetch(&get_request(&format!("{base}/b"))).unwrap();
        assert_eq!(r1.body, b"hi");
        assert_eq!(r2.body, b"hi");
        assert_eq!(accepted.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn pool_keys_isolate_origins() {
        let (base1, acc1) = spawn_keepalive_server(|_| (ok_body(b"one"), true));
        let (base2, acc2) = spawn_keepalive_server(|_| (ok_body(b"two"), true));
        assert_eq!(
            fetch(&get_request(&format!("{base1}/"))).unwrap().body,
            b"one"
        );
        assert_eq!(
            fetch(&get_request(&format!("{base2}/"))).unwrap().body,
            b"two"
        );
        // Second round trips reuse each origin's own connection.
        assert_eq!(
            fetch(&get_request(&format!("{base1}/"))).unwrap().body,
            b"one"
        );
        assert_eq!(
            fetch(&get_request(&format!("{base2}/"))).unwrap().body,
            b"two"
        );
        assert_eq!(acc1.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(acc2.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn slow_h1_response_does_not_block_concurrent_fetch_to_same_host() {
        // H1 head-of-line regression: the stack never pipelines and never
        // queues requests on one connection — each fetch checks out (or
        // dials) its own connection, so a stalled response on one must
        // not delay a concurrent fetch to the same host. The /slow
        // handler blocks until the test releases it, so the /fast fetch
        // completing at all proves it never queued behind the in-flight
        // exchange; no timing assertions needed.
        use std::sync::atomic::{AtomicBool, Ordering};
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let base = format!("http://127.0.0.1:{port}");
        let slow_seen = std::sync::Arc::new(AtomicBool::new(false));
        let released = std::sync::Arc::new(AtomicBool::new(false));
        let accepted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        thread::spawn({
            let slow_seen = slow_seen.clone();
            let released = released.clone();
            let accepted = accepted.clone();
            move || {
                for stream in listener.incoming() {
                    let Ok(mut stream) = stream else { break };
                    accepted.fetch_add(1, Ordering::SeqCst);
                    let slow_seen = slow_seen.clone();
                    let released = released.clone();
                    thread::spawn(move || {
                        let req = read_request(&mut stream);
                        // Direct fetches use absolute-form request
                        // targets (`GET http://host:port/slow ...`), so
                        // match the path inside the request line.
                        let slow = req.windows(6).any(|w| w == b"/slow ");
                        if slow {
                            // Hold this connection open until the test
                            // releases it: the response cannot be sent
                            // early, whatever the client does.
                            slow_seen.store(true, Ordering::SeqCst);
                            while !released.load(Ordering::SeqCst) {
                                thread::sleep(Duration::from_millis(5));
                            }
                            let _ = stream.write_all(&ok_body(b"slow"));
                        } else {
                            let _ = stream.write_all(&ok_body(b"fast"));
                        }
                    });
                }
            }
        });

        // The slow fetch runs on its own thread and holds its connection
        // for the whole exchange.
        let slow_base = base.clone();
        let slow = thread::spawn(move || fetch(&get_request(&format!("{slow_base}/slow"))));
        // Wait for the /slow request to arrive server-side (bounded, so
        // a failure cannot hang the suite on the fetch timeout).
        let deadline = Instant::now() + Duration::from_secs(30);
        while !slow_seen.load(Ordering::SeqCst) {
            if Instant::now() > deadline {
                released.store(true, Ordering::SeqCst);
                panic!("the /slow request never reached the server");
            }
            thread::sleep(Duration::from_millis(5));
        }
        // The concurrent fetch to the same host must complete while the
        // slow response is still blocked server-side.
        let fast = fetch(&get_request(&format!("{base}/fast"))).unwrap();
        assert_eq!(fast.body, b"fast");
        assert!(
            !released.load(Ordering::SeqCst),
            "the slow response must still be held"
        );
        released.store(true, Ordering::SeqCst);
        assert_eq!(slow.join().unwrap().unwrap().body, b"slow");
        // Two distinct connections served the two fetches.
        assert_eq!(accepted.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn stale_pooled_connection_retried_once_on_fresh() {
        // The server answers the first request on a connection, then
        // closes without responding to the second: the client's pooled
        // connection dies after its probe, forcing the once-only retry.
        let (base, accepted) = spawn_keepalive_server(|idx| {
            if idx == 1 {
                (Vec::new(), false)
            } else {
                (ok_body(b"hi"), true)
            }
        });
        let r1 = fetch(&get_request(&format!("{base}/a"))).unwrap();
        assert_eq!(r1.body, b"hi");
        // Deterministic: the pooled connection is verified alive by its
        // probe *before* the client writes the second request, and the
        // server only closes after reading that request.
        let r2 = fetch(&get_request(&format!("{base}/b"))).unwrap();
        assert_eq!(r2.body, b"hi");
        assert_eq!(accepted.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn chunked_response_reused() {
        let (base, accepted) = spawn_keepalive_server(|_| {
            let mut r = head(200, &[("Transfer-Encoding", "chunked")]);
            r.extend_from_slice(b"2\r\nhi\r\n0\r\n\r\n");
            (r, true)
        });
        let r1 = fetch(&get_request(&format!("{base}/a"))).unwrap();
        let r2 = fetch(&get_request(&format!("{base}/b"))).unwrap();
        assert_eq!(r1.body, b"hi");
        assert_eq!(r2.body, b"hi");
        assert_eq!(accepted.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn close_delimited_response_not_reused() {
        // No Content-Length and no Transfer-Encoding: the message ends
        // at close, so the connection can never be pooled.
        let (base, accepted) = spawn_keepalive_server(|_| {
            let mut r = head(200, &[]);
            r.extend_from_slice(b"hi");
            (r, false)
        });
        let r1 = fetch(&get_request(&format!("{base}/a"))).unwrap();
        let r2 = fetch(&get_request(&format!("{base}/b"))).unwrap();
        assert_eq!(r1.body, b"hi");
        assert_eq!(r2.body, b"hi");
        assert_eq!(accepted.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    /// A raw socketpair: write `response` from one end, parse and read the
    /// body from the other, and report whether the framing consumed
    /// exactly the response.
    fn framed_exact(response: &[u8], shutdown_write: bool) -> bool {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut writer = std::net::TcpStream::connect(addr).unwrap();
        let (reader, _) = listener.accept().unwrap();
        writer.write_all(response).unwrap();
        if shutdown_write {
            writer.shutdown(std::net::Shutdown::Write).unwrap();
        }
        let mut conn = Conn::Plain(reader);
        let limits = Limits::default();
        let parsed = read_head(&mut conn, "GET", &limits, test_deadline()).unwrap();
        let (_, _, exact) = read_body(&mut conn, parsed, &limits, test_deadline()).unwrap();
        exact
    }

    #[test]
    fn read_body_reports_exact_framing() {
        // Exact Content-Length: reusable.
        assert!(framed_exact(
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi",
            false
        ));
        // Trailing bytes belong to the next message: not reusable.
        assert!(!framed_exact(
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhiEXTRA",
            false
        ));
        // Exact chunked: reusable.
        assert!(framed_exact(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nhi\r\n0\r\n\r\n",
            false
        ));
        // Bytes past the chunked terminator: not reusable.
        assert!(!framed_exact(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nhi\r\n0\r\n\r\nZZ",
            false
        ));
        // Empty body, exact: reusable.
        assert!(framed_exact(
            b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n",
            false
        ));
        // Empty body with trailing garbage: not reusable.
        assert!(!framed_exact(
            b"HTTP/1.1 204 No Content\r\n\r\nGARBAGE",
            false
        ));
        // Close-delimited: never reusable, even when the peer shuts down
        // cleanly right at the boundary.
        assert!(!framed_exact(b"HTTP/1.1 200 OK\r\n\r\nhi", true));
    }

    /// Serve `body` with a `Content-Encoding` header and check `fetch`
    /// returns the decoded bytes.
    fn fetch_encoded(body: &'static [u8], encoding: &'static str) -> Vec<u8> {
        let base = spawn_server(1, move |_| {
            let mut r = head(
                200,
                &[
                    ("Content-Length", &body.len().to_string()),
                    ("Content-Encoding", encoding),
                ],
            );
            r.extend_from_slice(body);
            r
        });
        fetch(&get_request(&format!("{base}/"))).unwrap().body
    }

    #[test]
    fn fetch_decodes_gzip_body() {
        let gz: &[u8] = include_bytes!("../tests/fixtures/decode/hello.gz");
        let plain: &[u8] = include_bytes!("../tests/fixtures/decode/hello.txt");
        assert_eq!(fetch_encoded(gz, "gzip"), plain);
    }

    #[test]
    fn fetch_decodes_zlib_body() {
        let zl: &[u8] = include_bytes!("../tests/fixtures/decode/hello.zlib");
        let plain: &[u8] = include_bytes!("../tests/fixtures/decode/hello.txt");
        assert_eq!(fetch_encoded(zl, "deflate"), plain);
    }

    #[test]
    fn fetch_decodes_raw_deflate_body() {
        // Many servers mislabel raw deflate as `deflate`; the zlib-first,
        // raw-fallback handling covers them.
        let raw: &[u8] = include_bytes!("../tests/fixtures/decode/hello.raw");
        let plain: &[u8] = include_bytes!("../tests/fixtures/decode/hello.txt");
        assert_eq!(fetch_encoded(raw, "deflate"), plain);
    }

    #[test]
    fn fetch_rejects_unknown_content_encoding() {
        let base = spawn_server(1, |_| {
            let mut r = head(200, &[("Content-Length", "2"), ("Content-Encoding", "br")]);
            r.extend_from_slice(b"hi");
            r
        });
        match fetch(&get_request(&format!("{base}/"))) {
            Err(FetchError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::InvalidData),
            other => panic!("expected InvalidData Io error, got {other:?}"),
        }
    }

    #[test]
    fn fetch_empty_gzip_body_on_200_fails_decode() {
        // A 200 with `Content-Encoding: gzip` and an empty body is
        // corrupt input, not a pass-through: the trailer check runs.
        let base = spawn_server(1, |_| {
            head(
                200,
                &[("Content-Length", "0"), ("Content-Encoding", "gzip")],
            )
        });
        match fetch(&get_request(&format!("{base}/"))) {
            Err(FetchError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::InvalidData),
            other => panic!("expected InvalidData Io error, got {other:?}"),
        }
    }

    #[test]
    fn fetch_empty_gzip_body_on_204_passes_through() {
        // 204 forbids a body, so the `Content-Encoding` header with no
        // payload skips trailer validation.
        let base = spawn_server(1, |_| head(204, &[("Content-Encoding", "gzip")]));
        let resp = fetch(&get_request(&format!("{base}/"))).unwrap();
        assert!(resp.body.is_empty());
    }

    #[test]
    fn fetch_empty_gzip_body_on_304_passes_through() {
        let base = spawn_server(1, |_| head(304, &[("Content-Encoding", "gzip")]));
        let resp = fetch(&get_request(&format!("{base}/"))).unwrap();
        assert!(resp.body.is_empty());
    }

    #[test]
    fn read_body_enforces_decode_cap() {
        // A gzip body whose *encoded* form fits `max_body_bytes` but
        // whose decoded form does not: the streaming decoder must fail
        // with BodyTooLarge instead of expanding in memory (zip bomb).
        let gz: &[u8] = include_bytes!("../tests/fixtures/decode/tiny_bomb.gz");
        assert!(gz.len() < 48, "fixture must fit the encoded cap");
        let mut response = head(
            200,
            &[
                ("Content-Length", &gz.len().to_string()),
                ("Content-Encoding", "gzip"),
            ],
        );
        response.extend_from_slice(gz);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut writer = std::net::TcpStream::connect(addr).unwrap();
        let (reader, _) = listener.accept().unwrap();
        writer.write_all(&response).unwrap();
        writer.shutdown(std::net::Shutdown::Write).unwrap();
        let mut conn = Conn::Plain(reader);
        let limits = Limits {
            max_body_bytes: 48,
            ..Limits::default()
        };
        let parsed = read_head(&mut conn, "GET", &limits, test_deadline()).unwrap();
        match read_body(&mut conn, parsed, &limits, test_deadline()) {
            Err(FetchError::BodyTooLarge) => {}
            other => panic!("expected BodyTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn read_body_streams_gzip_without_buffering_encoded() {
        // A chunked gzip body decodes to the plain bytes; with a tight
        // encoded-side trickle the decoder must still produce output
        // incrementally (no whole-body buffering).
        let gz: &[u8] = include_bytes!("../tests/fixtures/decode/hello.gz");
        let plain: &[u8] = include_bytes!("../tests/fixtures/decode/hello.txt");
        let base = spawn_server(1, move |_| {
            let mut r = head(
                200,
                &[
                    ("Transfer-Encoding", "chunked"),
                    ("Content-Encoding", "gzip"),
                ],
            );
            for piece in gz.chunks(7) {
                r.extend_from_slice(format!("{:x}\r\n", piece.len()).as_bytes());
                r.extend_from_slice(piece);
                r.extend_from_slice(b"\r\n");
            }
            r.extend_from_slice(b"0\r\n\r\n");
            r
        });
        let resp = fetch(&get_request(&format!("{base}/"))).unwrap();
        assert_eq!(resp.body, plain);
    }

    #[test]
    fn stale_retry_only_when_safe_early_and_once() {
        assert!(may_retry_stale(true, false, "GET", false));
        assert!(may_retry_stale(true, false, "HEAD", false));
        assert!(may_retry_stale(true, false, "OPTIONS", false));
        assert!(may_retry_stale(true, false, "TRACE", false));
        // A non-idempotent method is never replayed.
        assert!(!may_retry_stale(true, false, "POST", false));
        assert!(!may_retry_stale(true, false, "PUT", false));
        // Only once, only before a response head arrived, only for pooled
        // connections.
        assert!(!may_retry_stale(true, false, "GET", true));
        assert!(!may_retry_stale(true, true, "GET", false));
        assert!(!may_retry_stale(false, false, "GET", false));
    }

    // -- resumed-H2 stale retry --------------------------------------------

    /// In-memory H2 transport: pre-loaded server bytes flow in,
    /// everything the client writes is discarded.
    struct ScriptIo {
        inbound: std::collections::VecDeque<u8>,
    }

    impl ScriptIo {
        fn new(inbound: Vec<u8>) -> Self {
            Self {
                inbound: inbound.into(),
            }
        }
    }

    impl Read for ScriptIo {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.inbound.is_empty() {
                // The scenarios below never read past their scripted
                // frames; surfacing WouldBlock (like a non-blocking
                // socket) turns a scripting bug into a fast timeout
                // instead of a hang on the deadline.
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "script exhausted",
                ));
            }
            let n = buf.len().min(self.inbound.len());
            for slot in buf.iter_mut().take(n) {
                *slot = self.inbound.pop_front().unwrap();
            }
            Ok(n)
        }
    }

    impl Write for ScriptIo {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// One raw HTTP/2 frame. Types: 0x0 DATA, 0x1 HEADERS, 0x4
    /// SETTINGS. Flags: 0x1 END_STREAM, 0x4 END_HEADERS.
    fn h2_frame(ty: u8, flags: u8, stream: u32, payload: &[u8]) -> Vec<u8> {
        let mut v = Vec::with_capacity(9 + payload.len());
        let len = payload.len() as u32;
        v.extend_from_slice(&[(len >> 16) as u8, (len >> 8) as u8, len as u8, ty, flags]);
        v.extend_from_slice(&(stream & 0x7fff_ffff).to_be_bytes());
        v.extend_from_slice(payload);
        v
    }

    fn h2_hpack(fields: &[(&[u8], &[u8])]) -> Vec<u8> {
        let mut enc = crate::hpack::Encoder::new();
        let mut out = Vec::new();
        enc.encode(fields, &mut out);
        out
    }

    /// Run `fetch_h2_exchange` against a scripted session. The script
    /// starts with the server's SETTINGS (consumed by the handshake);
    /// every scenario below fails the exchange. Returns the error
    /// together with whether the response HEADERS arrived before it.
    fn h2_exchange_err(script: Vec<u8>, limits: &Limits) -> (FetchError, bool) {
        let mut session = Session::new(ScriptIo::new(script), limits.max_headers);
        assert!(session.handshake(test_deadline()).is_ok());
        let mut jar = CookieJar::new();
        let (result, got_head) = crate::http2::fetch_h2_exchange(
            &mut session,
            "GET",
            &url("https://x.test/"),
            &[],
            None,
            None,
            false,
            None,
            false,
            test_deadline(),
            Duration::from_secs(60),
            limits,
            &mut jar,
            None,
            0,
        );
        (
            result.expect_err("scripted H2 exchange must fail"),
            got_head,
        )
    }

    #[test]
    fn resumed_h2_no_retry_after_response_head() {
        // 200 with `content-length: 4`, then only 2 body bytes with
        // END_STREAM: the exchange fails after the head fully arrived.
        let head = h2_hpack(&[
            (b":status".as_slice(), b"200".as_slice()),
            (b"content-length".as_slice(), b"4".as_slice()),
        ]);
        let mut script = h2_frame(0x4, 0x0, 0, &[]);
        script.extend_from_slice(&h2_frame(0x1, 0x4, 1, &head));
        script.extend_from_slice(&h2_frame(0x0, 0x1, 1, b"hi"));
        let (err, got_head) = h2_exchange_err(script, &Limits::default());
        assert!(got_head, "expected the response head, got {err:?}");
        // Like HTTP/1.1: once the head arrived the server may have
        // acted, so no replay on a fresh connection — even for a safe
        // method on a pooled session.
        assert!(!may_retry_stale(true, got_head, "GET", false));
    }

    #[test]
    fn resumed_h2_no_retry_when_body_too_large() {
        // The body cap is hit while accumulating DATA, which the
        // exchange only reaches after the final HEADERS parsed.
        let limits = Limits {
            max_body_bytes: 8,
            ..Limits::default()
        };
        let head = h2_hpack(&[(b":status".as_slice(), b"200".as_slice())]);
        let mut script = h2_frame(0x4, 0x0, 0, &[]);
        script.extend_from_slice(&h2_frame(0x1, 0x4, 1, &head));
        script.extend_from_slice(&h2_frame(0x0, 0x1, 1, &[7u8; 16]));
        let (err, got_head) = h2_exchange_err(script, &limits);
        assert!(matches!(err, FetchError::BodyTooLarge));
        assert!(got_head);
        assert!(!may_retry_stale(true, got_head, "GET", false));
    }

    #[test]
    fn resumed_h2_retry_before_response_head() {
        // DATA before any response HEADERS: the head never arrived, so
        // the stale pooled session keeps its one retry.
        let mut script = h2_frame(0x4, 0x0, 0, &[]);
        script.extend_from_slice(&h2_frame(0x0, 0x1, 1, b"hi"));
        let (err, got_head) = h2_exchange_err(script, &Limits::default());
        assert!(!got_head, "head must not have arrived, got {err:?}");
        assert!(may_retry_stale(true, got_head, "GET", false));
        // ...but still only once, never for unsafe methods, never for
        // a fresh connection.
        assert!(!may_retry_stale(true, got_head, "GET", true));
        assert!(!may_retry_stale(true, got_head, "POST", false));
        assert!(!may_retry_stale(false, got_head, "GET", false));
    }

    #[test]
    fn resumed_h2_retry_on_pre_head_goaway() {
        // GOAWAY before the server SETTINGS: the handshake absorbs it
        // and `open_stream` fails, so no HEADERS ever arrived. The
        // pooled session keeps its one retry.
        let mut goaway = 0u32.to_be_bytes().to_vec();
        goaway.extend_from_slice(&0u32.to_be_bytes());
        let mut script = h2_frame(0x7, 0x0, 0, &goaway);
        script.extend_from_slice(&h2_frame(0x4, 0x0, 0, &[]));
        let (err, got_head) = h2_exchange_err(script, &Limits::default());
        assert!(
            matches!(&err, FetchError::Http2(m) if m.contains("GOAWAY")),
            "expected a GOAWAY failure, got {err:?}"
        );
        assert!(!got_head);
        assert!(may_retry_stale(true, got_head, "GET", false));
        // Method safety still applies: an unsafe method is never
        // replayed, even pre-head.
        assert!(!may_retry_stale(true, got_head, "POST", false));
    }

    #[test]
    fn resumed_h2_no_retry_reset_after_partial_body() {
        // 200 with `content-length: 4`, 2 body bytes, then a stream
        // reset: mid-body failure after the head — no replay.
        let head = h2_hpack(&[
            (b":status".as_slice(), b"200".as_slice()),
            (b"content-length".as_slice(), b"4".as_slice()),
        ]);
        let mut script = h2_frame(0x4, 0x0, 0, &[]);
        script.extend_from_slice(&h2_frame(0x1, 0x4, 1, &head));
        script.extend_from_slice(&h2_frame(0x0, 0x0, 1, b"hi"));
        script.extend_from_slice(&h2_frame(0x3, 0x0, 1, &8u32.to_be_bytes()));
        let (err, got_head) = h2_exchange_err(script, &Limits::default());
        assert!(
            matches!(&err, FetchError::Http2(m) if m.contains("reset stream")),
            "expected a reset failure, got {err:?}"
        );
        assert!(got_head);
        assert!(!may_retry_stale(true, got_head, "GET", false));
    }

    #[test]
    fn resumed_h2_no_retry_goaway_after_partial_body() {
        // 200, 2 body bytes, then GOAWAY racing more DATA: the GOAWAY is
        // absorbed while a DATA event is still delivered, so the
        // exchange's loop-top check fires after the head arrived — no
        // replay.
        let head = h2_hpack(&[(b":status".as_slice(), b"200".as_slice())]);
        let mut goaway = 1u32.to_be_bytes().to_vec();
        goaway.extend_from_slice(&0u32.to_be_bytes());
        let mut script = h2_frame(0x4, 0x0, 0, &[]);
        script.extend_from_slice(&h2_frame(0x1, 0x4, 1, &head));
        script.extend_from_slice(&h2_frame(0x0, 0x0, 1, b"hi"));
        script.extend_from_slice(&h2_frame(0x7, 0x0, 0, &goaway));
        script.extend_from_slice(&h2_frame(0x0, 0x0, 1, b"yo"));
        let (err, got_head) = h2_exchange_err(script, &Limits::default());
        assert!(
            matches!(&err, FetchError::Http2(m) if m.contains("GOAWAY")),
            "expected a GOAWAY failure, got {err:?}"
        );
        assert!(got_head);
        assert!(!may_retry_stale(true, got_head, "GET", false));
    }

    #[test]
    fn resumed_h2_no_retry_reset_right_after_head() {
        // Boundary: the final HEADERS parsed (no END_STREAM, no body
        // bytes yet), then an immediate stream reset. The head arrived,
        // so no replay even though no body was seen.
        let head = h2_hpack(&[(b":status".as_slice(), b"200".as_slice())]);
        let mut script = h2_frame(0x4, 0x0, 0, &[]);
        script.extend_from_slice(&h2_frame(0x1, 0x4, 1, &head));
        script.extend_from_slice(&h2_frame(0x3, 0x0, 1, &8u32.to_be_bytes()));
        let (err, got_head) = h2_exchange_err(script, &Limits::default());
        assert!(
            matches!(&err, FetchError::Http2(m) if m.contains("reset stream")),
            "expected a reset failure, got {err:?}"
        );
        assert!(got_head, "head arrived before the reset");
        assert!(!may_retry_stale(true, got_head, "GET", false));
    }

    // -- proxy ---------------------------------------------------------------

    fn proxy(url_str: &str, no_proxy: &[&str]) -> ProxyConfig {
        ProxyConfig {
            url: url(url_str),
            no_proxy: no_proxy.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn base64_encode_vectors() {
        // RFC 4648 §6 test vectors, plus the empty input.
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64_encode(b"user:pass"), "dXNlcjpwYXNz");
    }

    #[test]
    fn no_proxy_entry_matching() {
        let cases: &[(&str, &str, u16, bool)] = &[
            // Exact and subdomain matches, case-insensitive.
            ("example.com", "example.com", 80, true),
            ("example.com", "a.example.com", 80, true),
            ("example.com", "deep.a.example.com", 443, true),
            ("EXAMPLE.COM", "a.example.com", 80, true),
            ("example.com", "A.EXAMPLE.COM", 80, true),
            // A leading dot means the same as the bare domain.
            (".example.com", "example.com", 80, true),
            (".example.com", "a.example.com", 80, true),
            // A label boundary is required: no suffix tricks.
            ("example.com", "badexample.com", 80, false),
            ("example.com", "example.com.evil.com", 80, false),
            ("ample.com", "example.com", 80, false),
            // A `:port` suffix restricts the bypass to that port.
            ("example.com:8080", "example.com", 8080, true),
            ("example.com:8080", "a.example.com", 8080, true),
            ("example.com:8080", "example.com", 80, false),
            ("example.com:8080", "example.com", 443, false),
            // Wildcard and degenerate entries.
            ("*", "anything.example.org", 1234, true),
            ("", "example.com", 80, false),
            ("   ", "example.com", 80, false),
            ("  example.com  ", "example.com", 80, true),
            // A trailing dot (FQDN root) is ignored on either side.
            ("example.com.", "example.com", 80, true),
            ("example.com", "example.com.", 80, true),
            // Malformed entries never match: the proxy stays in use.
            ("example.com:http", "example.com", 80, false),
            ("example.com:", "example.com", 80, false),
            ("[::1", "::1", 80, false),
            ("[::1]:bogus", "::1", 80, false),
        ];
        for &(entry, host, port, expected) in cases {
            assert_eq!(
                no_proxy_match(entry, host, port),
                expected,
                "entry={entry:?} host={host:?}:{port}"
            );
        }
    }

    #[test]
    fn no_proxy_ip_forms() {
        let cases: &[(&str, &str, u16, bool)] = &[
            ("127.0.0.1", "127.0.0.1", 80, true),
            ("127.0.0.1", "127.0.0.2", 80, false),
            ("localhost", "localhost", 3000, true),
            ("localhost", "LOCALHOST", 3000, true),
            // IPv6: bracketed or bare on either side.
            ("::1", "::1", 80, true),
            ("::1", "[::1]", 80, true),
            ("[::1]", "::1", 80, true),
            ("[::1]:8080", "::1", 8080, true),
            ("[::1]:8080", "[::1]", 8080, true),
            ("[::1]:8080", "::1", 80, false),
            ("::1", "::2", 80, false),
        ];
        for &(entry, host, port, expected) in cases {
            assert_eq!(
                no_proxy_match(entry, host, port),
                expected,
                "entry={entry:?} host={host:?}:{port}"
            );
        }
    }

    #[test]
    fn proxy_bypasses_uses_serialized_hosts() {
        let p = proxy("http://127.0.0.1:8080/", &["example.com", "[::1]"]);
        assert!(p.bypasses("example.com", 80));
        assert!(p.bypasses("a.example.com", 443));
        assert!(!p.bypasses("other.com", 80));
        // `authority()` hands fetch bracketed IPv6 hosts.
        assert!(p.bypasses("[::1]", 80));
        assert!(!p.bypasses("[::2]", 80));
    }

    #[test]
    fn proxy_authorization_from_userinfo() {
        let p = proxy("http://proxyuser:proxypass@127.0.0.1:8080/", &[]);
        assert_eq!(
            p.proxy_authorization_value().as_deref(),
            Some("Basic cHJveHl1c2VyOnByb3h5cGFzcw==")
        );
        // No userinfo: no header.
        let p = proxy("http://127.0.0.1:8080/", &[]);
        assert!(p.proxy_authorization_value().is_none());
        // Username only: the empty password still sends `user:`.
        let p = proxy("http://proxyuser@127.0.0.1:8080/", &[]);
        assert_eq!(
            p.proxy_authorization_value().as_deref(),
            Some("Basic cHJveHl1c2VyOg==")
        );
        // Userinfo is percent-decoded before encoding.
        let p = proxy("http://u%40x:p%3Ass@127.0.0.1:8080/", &[]);
        assert_eq!(
            p.proxy_authorization_value().as_deref(),
            Some("Basic dUB4OnA6c3M=")
        );
    }

    #[test]
    fn proxy_authority_validation() {
        let p = proxy("http://proxy.internal:3128/", &[]);
        assert_eq!(p.proxy_authority().unwrap(), ("proxy.internal", 3128));
        // The default proxy port is 8080.
        let p = proxy("http://proxy.internal/", &[]);
        assert_eq!(p.proxy_authority().unwrap(), ("proxy.internal", 8080));
        // Non-http schemes are rejected before any dial.
        let p = proxy("socks5://127.0.0.1:1080/", &[]);
        match p.proxy_authority() {
            Err(FetchError::Proxy(msg)) => assert!(msg.contains("http"), "{msg}"),
            other => panic!("expected FetchError::Proxy, got {other:?}"),
        }
    }

    #[test]
    fn proxy_from_env_parsing() {
        use std::collections::HashMap;
        let env: HashMap<&str, &str> = [
            ("HTTP_PROXY", "http://envuser:envpass@proxy.internal:3128/"),
            ("NO_PROXY", "localhost, 127.0.0.1 ,example.com:8080"),
        ]
        .into_iter()
        .collect();
        let cfg =
            ProxyConfig::from_env_with(|key| env.get(key).map(|s| s.to_string())).expect("set");
        assert_eq!(
            cfg.url.serialize(),
            "http://envuser:envpass@proxy.internal:3128/"
        );
        assert_eq!(cfg.no_proxy, ["localhost", "127.0.0.1", "example.com:8080"]);
        assert!(cfg.bypasses("127.0.0.1", 80));
        assert!(cfg.bypasses("example.com", 8080));
        assert!(!cfg.bypasses("example.com", 80));
        assert!(!cfg.bypasses("other.com", 80));

        // Lowercase names work as fallbacks.
        let env: HashMap<&str, &str> = [
            ("http_proxy", "http://proxy:8080/"),
            ("no_proxy", "*.internal"),
        ]
        .into_iter()
        .collect();
        let cfg =
            ProxyConfig::from_env_with(|key| env.get(key).map(|s| s.to_string())).expect("set");
        assert_eq!(cfg.url.host(), Some("proxy"));
        assert_eq!(cfg.no_proxy, ["*.internal"]);

        // Schemeless values default to http.
        let env: HashMap<&str, &str> = [("HTTP_PROXY", "proxy.internal:3128")]
            .into_iter()
            .collect();
        let cfg =
            ProxyConfig::from_env_with(|key| env.get(key).map(|s| s.to_string())).expect("set");
        assert_eq!(cfg.url.scheme(), "http");
        assert_eq!(cfg.url.host(), Some("proxy.internal"));
        assert_eq!(cfg.url.port(), Some(3128));

        // Empty or absent: None.
        let env: HashMap<&str, &str> = [("HTTP_PROXY", "  ")].into_iter().collect();
        assert!(ProxyConfig::from_env_with(|key| env.get(key).map(|s| s.to_string())).is_none());
        let env: HashMap<&str, &str> = HashMap::new();
        assert!(ProxyConfig::from_env_with(|key| env.get(key).map(|s| s.to_string())).is_none());
    }

    #[test]
    fn request_head_origin_form_for_tunnel() {
        // Inside a CONNECT tunnel the request looks like a direct one:
        // origin-form target, origin Host.
        let bytes = request_head_bytes(
            "GET",
            &url("https://example.com:8443/a?x=1#frag"),
            &[],
            None,
            None,
            false,
            None,
            true,
        )
        .unwrap();
        let s = String::from_utf8(bytes).unwrap();
        assert!(s.starts_with("GET /a?x=1 HTTP/1.1\r\n"), "{s}");
        assert!(s.contains("Host: example.com:8443\r\n"), "{s}");
        assert!(!s.contains("example.com:8443/a"), "{s}");
    }

    #[test]
    fn proxy_authorization_header_rules() {
        let extra = vec![(
            "Proxy-Authorization".to_string(),
            "Basic Y2FsbGVy".to_string(),
        )];
        // The config's credentials are authoritative: the caller's header
        // is dropped and exactly one header is sent.
        let bytes = request_head_bytes(
            "GET",
            &url("http://example.com/"),
            &extra,
            None,
            None,
            false,
            Some("Basic Y29uZmln"),
            false,
        )
        .unwrap();
        let s = String::from_utf8(bytes).unwrap();
        assert_eq!(s.matches("Proxy-Authorization:").count(), 1, "{s}");
        assert!(s.contains("Proxy-Authorization: Basic Y29uZmln\r\n"), "{s}");
        // Without config credentials the caller's header passes through...
        let bytes = request_head_bytes(
            "GET",
            &url("http://example.com/"),
            &extra,
            None,
            None,
            false,
            None,
            false,
        )
        .unwrap();
        let s = String::from_utf8(bytes).unwrap();
        assert!(s.contains("Proxy-Authorization: Basic Y2FsbGVy\r\n"), "{s}");
        // ...but is stripped on a cross-origin redirect, as before.
        let bytes = request_head_bytes(
            "GET",
            &url("http://example.com/"),
            &extra,
            None,
            None,
            true,
            None,
            false,
        )
        .unwrap();
        let s = String::from_utf8(bytes).unwrap();
        assert!(!s.contains("Proxy-Authorization"), "{s}");
    }

    #[test]
    fn proxy_authorization_never_sent_inside_connect_tunnel() {
        // Inside a CONNECT tunnel the request reaches the origin: neither
        // the config's credentials nor a caller-supplied header may go
        // with it. The CONNECT request itself carries Proxy-Authorization.
        let extra = vec![(
            "Proxy-Authorization".to_string(),
            "Basic Y2FsbGVy".to_string(),
        )];
        for proxy_auth in [Some("Basic Y29uZmln"), None] {
            let bytes = request_head_bytes(
                "GET",
                &url("https://example.com/a"),
                &extra,
                None,
                None,
                false,
                proxy_auth,
                true,
            )
            .unwrap();
            let s = String::from_utf8(bytes).unwrap();
            assert!(s.starts_with("GET /a HTTP/1.1\r\n"), "{s}");
            assert!(!s.contains("Proxy-Authorization"), "{s}");
            assert!(!s.contains("Y29uZmln"), "{s}");
            assert!(!s.contains("Y2FsbGVy"), "{s}");
        }
    }

    #[test]
    fn plain_http_proxy_absolute_form_and_auth() {
        // Fake proxy: records the raw request, then answers as the origin
        // would. The client must dial the proxy (not the origin), send an
        // absolute-form target, the origin's Host, and Basic credentials
        // from the proxy URL's userinfo.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let req = read_request(&mut s);
            tx.send(String::from_utf8_lossy(&req).into_owned()).unwrap();
            let mut r = head(200, &[("Content-Length", "5")]);
            r.extend_from_slice(b"proxy");
            let _ = s.write_all(&r);
        });
        let p = proxy(
            &format!("http://proxyuser:proxypass@127.0.0.1:{port}/"),
            &[],
        );
        let resp = fetch_with_proxy(&get_request("http://example.com/some/path?q=1"), &p).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, b"proxy");
        let wire = rx.recv().unwrap();
        assert!(
            wire.starts_with("GET http://example.com/some/path?q=1 HTTP/1.1\r\n"),
            "{wire}"
        );
        assert!(
            has_header(
                wire.as_bytes(),
                "proxy-authorization",
                "Basic cHJveHl1c2VyOnByb3h5cGFzcw=="
            ),
            "{wire}"
        );
        assert!(has_header(wire.as_bytes(), "host", "example.com"), "{wire}");
    }

    #[test]
    fn connect_tunnel_carries_tls() {
        // Fake proxy: answers 200 to the CONNECT, then expects the TLS
        // ClientHello (0x16) as the next bytes — proving the client runs
        // the real TLS handshake to the origin over the tunnel.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let req = read_request(&mut s);
            let wire = String::from_utf8_lossy(&req).into_owned();
            let _ = s.write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n");
            let mut first = [0u8; 1];
            s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
            let n = s.read(&mut first).unwrap_or(0);
            let _ = tx.send((wire, n == 1 && first[0] == 0x16));
        });
        let p = proxy(&format!("http://u:p@127.0.0.1:{port}/"), &[]);
        // No real TLS server answers, so the handshake fails — expected.
        // The assertions below are on what the proxy observed.
        match fetch_with_proxy(&get_request("https://example.com/"), &p) {
            Err(FetchError::Tls(_)) | Err(FetchError::Io(_)) => {}
            other => panic!("expected a TLS/IO failure after CONNECT, got {other:?}"),
        }
        let (wire, saw_client_hello) = rx.recv().unwrap();
        assert!(
            wire.starts_with("CONNECT example.com:443 HTTP/1.1\r\n"),
            "{wire}"
        );
        assert!(
            has_header(wire.as_bytes(), "host", "example.com:443"),
            "{wire}"
        );
        assert!(
            has_header(wire.as_bytes(), "proxy-authorization", "Basic dTpw"),
            "{wire}"
        );
        assert!(
            saw_client_hello,
            "the client must start the TLS handshake inside the tunnel"
        );
    }

    /// The PSK identity the ClientHello offers, if any: parses one TLS
    /// handshake record fragment holding a ClientHello and returns the
    /// first `pre_shared_key` identity (extension 41).
    fn client_hello_psk_identity(fragment: &[u8]) -> Option<Vec<u8>> {
        let b = fragment;
        if b.len() < 4 || b[0] != 0x01 {
            return None; // not a ClientHello handshake message
        }
        let hs_len = u32::from_be_bytes([0, b[1], b[2], b[3]]) as usize;
        if b.len() < 4 + hs_len {
            return None;
        }
        let mut p = 4 + 2 + 32; // handshake header, client_version, random
        p += 1 + b[p] as usize; // session_id
        let cs_len = u16::from_be_bytes([b[p], b[p + 1]]) as usize;
        p += 2 + cs_len; // cipher_suites
        p += 1 + b[p] as usize; // compression_methods
        let ext_len = u16::from_be_bytes([b[p], b[p + 1]]) as usize;
        p += 2;
        let ext_end = p + ext_len;
        while p + 4 <= ext_end && p + 4 <= b.len() {
            let etype = u16::from_be_bytes([b[p], b[p + 1]]);
            let elen = u16::from_be_bytes([b[p + 2], b[p + 3]]) as usize;
            p += 4;
            if etype == 41 && p + 2 <= b.len() {
                let q = p + 2; // skip identities_len
                let id_len = u16::from_be_bytes([b[q], b[q + 1]]) as usize;
                return Some(b[q + 2..q + 2 + id_len].to_vec());
            }
            p += elen;
        }
        None
    }

    #[test]
    fn proxied_https_resumes_with_cached_ticket() {
        // Fake proxy: answers 200 to two sequential CONNECTs and captures
        // each ClientHello, proving the second tunnel reuses the first
        // connection's session state via a cached ticket. No real TLS
        // server answers, so both handshakes fail — expected; the
        // assertions are on what the proxy observed.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for _ in 0..2 {
                let (mut s, _) = listener.accept().unwrap();
                let _ = read_request(&mut s);
                let _ = s.write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n");
                // The client speaks its ClientHello as one TLS record
                // next; capture it whole, then hang up so the handshake
                // fails fast instead of idling on a read timeout.
                s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
                let mut hdr = [0u8; 5];
                s.read_exact(&mut hdr).unwrap();
                assert_eq!(hdr[0], 0x16, "expected a TLS handshake record");
                let len = u16::from_be_bytes([hdr[3], hdr[4]]) as usize;
                let mut rec = vec![0u8; len];
                s.read_exact(&mut rec).unwrap();
                let _ = tx.send(rec);
            }
        });
        let p = proxy(&format!("http://127.0.0.1:{port}/"), &[]);
        // A host no other test uses, so the process-wide ticket cache
        // starts empty for it and nothing else can plant or consume its
        // tickets.
        let host = "resumption-test.invalid";
        let req_url = format!("https://{host}/");
        let key = PoolKey::new(true, host, 443);

        // First tunnel: nothing cached, so the ClientHello offers no PSK.
        match fetch_with_proxy(&get_request(&req_url), &p) {
            Err(FetchError::Tls(_)) | Err(FetchError::Io(_)) => {}
            other => panic!("expected a TLS/IO failure after CONNECT, got {other:?}"),
        }
        let first = rx.recv().unwrap();
        assert!(
            client_hello_psk_identity(&first).is_none(),
            "first tunnel must not offer a PSK"
        );

        // As if the first connection's NewSessionTickets had been
        // harvested on discard (no real server here to issue them).
        global_pool().test_seed_tickets(&key, vec![crate::tls::CachedTicket::test_ticket(3600)]);

        // Second tunnel: the cached ticket is offered back for resumption.
        match fetch_with_proxy(&get_request(&req_url), &p) {
            Err(FetchError::Tls(_)) | Err(FetchError::Io(_)) => {}
            other => panic!("expected a TLS/IO failure after CONNECT, got {other:?}"),
        }
        let second = rx.recv().unwrap();
        assert_eq!(
            client_hello_psk_identity(&second).as_deref(),
            Some(b"tick".as_slice()),
            "second tunnel must resume with the cached ticket"
        );
        // Offering consumed the ticket: it is never offered twice.
        assert!(global_pool().ticket(&key).is_none());
        // And the tunnel was never pooled: the pool holds no idle
        // connection for a proxied origin.
        assert!(global_pool().take(&key).is_none());
    }

    #[test]
    fn connect_refused_407_surfaces_proxy_error() {
        // The proxy demands authentication: the CONNECT fails with 407
        // and the client surfaces a Proxy error naming the status and
        // the Proxy-Authenticate scheme.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let _ = read_request(&mut s);
            let _ = s.write_all(
                b"HTTP/1.1 407 Proxy Authentication Required\r\nProxy-Authenticate: Basic realm=\"p\"\r\nContent-Length: 0\r\n\r\n",
            );
        });
        // No userinfo: no credentials are sent.
        let p = proxy(&format!("http://127.0.0.1:{port}/"), &[]);
        match fetch_with_proxy(&get_request("https://example.com/"), &p) {
            Err(FetchError::Proxy(msg)) => {
                assert!(msg.contains("407"), "{msg}");
                assert!(msg.contains("Basic"), "{msg}");
                assert!(msg.contains("example.com:443"), "{msg}");
            }
            other => panic!("expected FetchError::Proxy, got {other:?}"),
        }
    }

    #[test]
    fn no_proxy_bypass_goes_direct() {
        // The proxy points at a closed port: any dial to it fails. With
        // 127.0.0.1 bypassed, the fetch must go direct.
        let base = spawn_server(1, |_| ok_body(b"direct"));
        let p = proxy("http://127.0.0.1:9/", &["127.0.0.1"]);
        let resp = fetch_with_proxy(&get_request(&format!("{base}/")), &p).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, b"direct");
        // Sanity: without the bypass entry the same fetch fails, because
        // the proxy is unreachable.
        let p = proxy("http://127.0.0.1:9/", &[]);
        assert!(fetch_with_proxy(&get_request(&format!("{base}/")), &p).is_err());
    }

    #[test]
    fn proxy_applies_per_redirect_hop() {
        // S1 redirects to S2. The proxy is a closed port and no_proxy
        // bypasses only S1's port: hop 1 goes direct, hop 2 must go
        // through the dead proxy and fail — proving the proxy decision is
        // re-evaluated per hop, not once per fetch.
        let s2 = spawn_server(1, |_| ok_body(b"final"));
        let s1 = spawn_server(2, {
            let s2 = s2.clone();
            move |_| head(302, &[("Location", &format!("{s2}/final"))])
        });
        let s1_port = s1.rsplit(':').next().unwrap().to_owned();
        let bypass_s1 = format!("127.0.0.1:{s1_port}");
        let p = proxy("http://127.0.0.1:9/", &[bypass_s1.as_str()]);
        match fetch_with_proxy(&get_request(&format!("{s1}/")), &p) {
            Err(FetchError::Io(_)) => {}
            other => panic!("hop 2 must fail through the dead proxy, got {other:?}"),
        }
        // With the whole loopback bypassed, the chain completes direct.
        let p = proxy("http://127.0.0.1:9/", &["127.0.0.1"]);
        let resp = fetch_with_proxy(&get_request(&format!("{s1}/")), &p).unwrap();
        assert_eq!(resp.body, b"final");
    }

    #[test]
    fn non_http_proxy_scheme_rejected_before_dial() {
        // Validation happens before any dial: no network is touched, so
        // example.com is safe to name here.
        let p = proxy("socks5://127.0.0.1:1080/", &[]);
        match fetch_with_proxy(&get_request("http://example.com/"), &p) {
            Err(FetchError::Proxy(msg)) => assert!(msg.contains("http"), "{msg}"),
            other => panic!("expected FetchError::Proxy, got {other:?}"),
        }
    }

    // -- shared-H2 open failures: pre-send vs post-send ---------------------

    /// A loopback listener that counts fresh dials: any silent replay
    /// of a request after a pooled H2 open fails shows up here.
    struct DialCounter {
        dials: Arc<AtomicUsize>,
        handle: std::thread::JoinHandle<()>,
    }

    fn count_dials() -> (DialCounter, u16) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let dials = Arc::new(AtomicUsize::new(0));
        let dials_clone = dials.clone();
        let handle = std::thread::spawn(move || {
            let start = Instant::now();
            while start.elapsed() < Duration::from_secs(2) {
                match listener.accept() {
                    Ok((mut s, _)) => {
                        dials_clone.fetch_add(1, Ordering::SeqCst);
                        // Read the request so the client is never left
                        // hanging on a full buffer; then close.
                        s.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
                        let mut buf = [0u8; 4096];
                        let _ = s.read(&mut buf);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Err(e) => panic!("accept: {e}"),
                }
            }
        });
        (DialCounter { dials, handle }, port)
    }

    /// Pool a shared H2 session whose next frame write blocks: the
    /// peer never reads and the send buffer is full, so `open_request`
    /// fails mid-send. Returns the peer half, which the test holds so
    /// the take-time probe sees a live socket.
    fn pool_blocked_h2_session(pool: &Pool, key: &PoolKey) -> std::net::TcpStream {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = std::net::TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        client
            .set_write_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        client.set_nonblocking(true).unwrap();
        let chunk = [0xABu8; 65536];
        loop {
            match client.write(&chunk) {
                Ok(_) => continue,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) => panic!("fill: {e}"),
            }
        }
        client.set_nonblocking(false).unwrap();
        pool.put_shared_h2(key, Session::new(Conn::Plain(client), 100), SOCKET_TIMEOUT);
        server
    }

    #[test]
    fn h2_open_post_send_failure_does_not_replay_post() {
        // A POST whose body hits the wire mid-`open_request` and then
        // fails must surface the error, never silently re-send on a
        // fresh connection: the request may already have executed.
        let (counter, port) = count_dials();
        let pool = Pool::new();
        let key = PoolKey::new(false, "127.0.0.1", port);
        let _peer = pool_blocked_h2_session(&pool, &key);

        let url = url(&format!("http://127.0.0.1:{port}/"));
        let mut jar = CookieJar::new();
        let result = fetch_once(
            "POST",
            &url,
            &[],
            Some(b"request-body"),
            &mut jar,
            false,
            &Limits::default(),
            Timeouts::default(),
            &NavigationContext::default(),
            &pool,
            None,
        );
        assert!(
            matches!(result, Err(FetchError::Timeout(_))),
            "mid-send POST failure must surface, got {result:?}"
        );
        counter.handle.join().unwrap();
        assert_eq!(
            counter.dials.load(Ordering::SeqCst),
            0,
            "a failed mid-body POST must never be replayed on a fresh connection"
        );
    }

    #[test]
    fn h2_open_post_send_failure_may_retry_idempotent_method() {
        // The same mid-send failure for an idempotent method *is*
        // retried on a fresh connection: replaying it cannot duplicate
        // a side effect.
        let (counter, port) = count_dials();
        let pool = Pool::new();
        let key = PoolKey::new(false, "127.0.0.1", port);
        let _peer = pool_blocked_h2_session(&pool, &key);

        let url = url(&format!("http://127.0.0.1:{port}/"));
        let mut jar = CookieJar::new();
        let _ = fetch_once(
            "GET",
            &url,
            &[],
            None,
            &mut jar,
            false,
            &Limits::default(),
            Timeouts::default(),
            &NavigationContext::default(),
            &pool,
            None,
        );
        counter.handle.join().unwrap();
        assert_eq!(
            counter.dials.load(Ordering::SeqCst),
            1,
            "a mid-send failure for an idempotent method must be retried once on a fresh connection"
        );
    }

    #[test]
    fn h2_open_pre_send_refusal_is_retried_on_fresh_connection() {
        // A pre-send refusal (here: the fetch deadline already passed
        // before the first frame write, so provably nothing was sent)
        // falls through to a fresh dial exactly like before.
        let (counter, port) = count_dials();
        let pool = Pool::new();
        let key = PoolKey::new(false, "127.0.0.1", port);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = std::net::TcpStream::connect(addr).unwrap();
        let (_server, _) = listener.accept().unwrap();
        pool.put_shared_h2(&key, Session::new(Conn::Plain(client), 100), SOCKET_TIMEOUT);

        let url = url(&format!("http://127.0.0.1:{port}/"));
        let mut jar = CookieJar::new();
        let timeouts = Timeouts {
            fetch_total: Duration::ZERO,
            ..Timeouts::default()
        };
        let result = fetch_once(
            "GET",
            &url,
            &[],
            None,
            &mut jar,
            false,
            &Limits::default(),
            timeouts,
            &NavigationContext::default(),
            &pool,
            None,
        );
        // The exchange itself cannot succeed with a zero fetch budget,
        // but the pre-send refusal must have been retried: the
        // listener saw the fresh dial.
        assert!(result.is_err(), "expected an error, got {result:?}");
        counter.handle.join().unwrap();
        assert_eq!(
            counter.dials.load(Ordering::SeqCst),
            1,
            "a pre-send H2 open refusal must fall through to a fresh dial"
        );
    }
}
