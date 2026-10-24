//! A WHATWG URL parser, from scratch, using only the standard library.
//!
//! This crate implements the [WHATWG URL Standard]'s basic URL parser,
//! URL serializer, percent encoding, `URLSearchParams`, and a pragmatic
//! subset of IDNA (UTS #46) domain processing. There are no dependencies
//! beyond `std`.
//!
//! [WHATWG URL Standard]: https://url.spec.whatwg.org/
//!
//! # Example
//!
//! ```
//! use bowser_url::{parse, Url};
//!
//! let base: Url = parse("https://example.org/dir/page", None).unwrap();
//! let url = parse("../other?q=1#frag", Some(&base)).unwrap();
//! assert_eq!(url.serialize(), "https://example.org/other?q=1#frag");
//! assert_eq!(url.scheme(), "https");
//! assert_eq!(url.host(), Some("example.org"));
//! assert_eq!(url.port(), None);
//! assert_eq!(url.path(), "/other");
//! assert_eq!(url.query(), Some("q=1"));
//! assert_eq!(url.fragment(), Some("frag"));
//! ```
//!
//! # IDNA conformance
//!
//! Full UTS #46 requires the Unicode Character Database. This crate
//! implements the pragmatic subset documented in [`idna`], which is exact
//! for every IDNA case in the web-platform-tests URL suite. Deviations from
//! full UTS #46 conformance:
//!
//! * Only the mappings listed in [`idna`] are applied; unlisted mappings
//!   (e.g. most CJK compatibility ideographs) keep the code point as-is.
//! * NFC is limited to a small composition table (Latin-1 precompositions
//!   and the `=`/`<`/`>` + U+0338 cases); canonical decomposition and
//!   mark reordering are not performed.
//! * Unassigned code points are accepted unless explicitly denylisted.
//! * Contextual joiner and bidi rules are simplified (see [`idna`]).
//! * DNS length limits are not enforced for *acceptance*, per the URL
//!   standard itself (`VerifyDnsLength=false`); they are enforced before
//!   Punycode *decoding* as a DoS bound (see [`idna`]).

mod base64;
mod host;
mod idna;
mod percent;

pub use base64::{base64_decode, base64_encode};
pub use host::Host;
pub use idna::domain_to_ascii;
pub use percent::{EncodeSet, percent_decode, percent_encode};

use crate::percent::{hex_digit, percent_encode_bytes_into, percent_encode_into};
use std::borrow::Cow;
use std::cmp::Ordering;

/// Error type for URL parsing failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The input has no scheme and no base URL was supplied (or the base
    /// URL cannot be used for the reference).
    MissingScheme,
    /// Credentials were supplied without a host, or a host was required
    /// but empty (e.g. `http://`, `http://@/`, `http://:`).
    MissingHost,
    /// The host failed to parse as a domain, IPv4 address, IPv6 address,
    /// or opaque host.
    InvalidHost,
    /// The port is not all ASCII digits or is greater than 65535.
    InvalidPort,
    /// A relative reference cannot be resolved against the base URL
    /// (e.g. a non-fragment reference against a cannot-be-a-base base).
    UnresolvableReference,
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            ParseError::MissingScheme => "missing scheme and no usable base URL",
            ParseError::MissingHost => "missing host",
            ParseError::InvalidHost => "invalid host",
            ParseError::InvalidPort => "invalid port",
            ParseError::UnresolvableReference => "cannot resolve reference against base URL",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for ParseError {}

/// A URL's origin per the WHATWG URL Standard.
///
/// See <https://url.spec.whatwg.org/#concept-url-origin>.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Origin {
    /// A tuple origin: (scheme, host, port), where port is the URL's port
    /// or the scheme's default port.
    Tuple(String, String, u16),
    /// An opaque origin: the scheme has no tuple origin (anything but
    /// `ftp`, `http`, `https`, `ws`, `wss`), or the URL has no host.
    Opaque,
}

impl Origin {
    /// Serialize per the WHATWG "serialized origin":
    /// `scheme://host`, with `:<port>` appended only when the port is
    /// not the scheme's default (a default port in the serialization
    /// would break exact-match checks such as CORS
    /// `Access-Control-Allow-Origin` comparisons). An opaque origin
    /// serializes as `"null"`. This is the canonical serializer for
    /// the crate, and this crate is the shared home for it: both
    /// `bowser_js` and `bowser_app` depend on `bowser_url`, so
    /// neither needs its own copy of the default-port table.
    pub fn serialize(&self) -> String {
        match self {
            Origin::Tuple(scheme, host, port) => {
                if default_port(scheme) == Some(*port) {
                    format!("{scheme}://{host}")
                } else {
                    format!("{scheme}://{host}:{port}")
                }
            }
            Origin::Opaque => "null".to_string(),
        }
    }

    /// Whether two origins are the same origin. Tuple origins compare
    /// by value; opaque origins are never same-origin with anything
    /// (each opaque origin is unique, so `Opaque == Opaque` must not
    /// read as same-origin even though the enum values are equal).
    pub fn is_same_origin(&self, other: &Origin) -> bool {
        matches!((self, other), (Origin::Tuple(..), Origin::Tuple(..))) && self == other
    }
}

/// CORS response check for a font (or any simple-request) fetch, per
/// the fetch standard's CORS protocol, without preflight: the request
/// is a GET carrying only an `Origin` header, so no preflight applies.
///
/// - Same-origin (`document_origin` same-origin with
///   `response_origin`) passes unconditionally, whatever `acao` is.
/// - Cross-origin passes iff `acao` is `*` (valid only when no
///   credentials were sent, i.e. `credentials_sent` is false) or
///   exactly the serialized document origin (the only form valid with
///   credentials). Anything else — missing, wrong, or `*` with
///   credentials — fails.
/// - An opaque document origin can never match by name: `acao: null`
///   is deliberately not accepted (browsers reject it for the same
///   reason), so only credentialless `*` passes for it.
/// - `acao` carries every `Access-Control-Allow-Origin` value the
///   response sent, in order. Fetch requires exactly one value: a
///   response with two (even `*, <origin>`) fails, because the
///   wildcard must not be smuggled past the check behind a first
///   match — callers must collect all values, not `Response::header`
///   (first match only). Zero values is the missing-header case.
/// - The single value is trimmed of surrounding ASCII whitespace
///   (`trim_ascii`: space, tab, LF, CR, FF — slightly wider than
///   HTTP field-value trimming, which is space/tab only, but
///   transport parsing strips line breaks before this point)
///   before comparison; Unicode whitespace is not trimmed.
pub fn cors_check(
    document_origin: &Origin,
    response_origin: &Origin,
    acao: &[&str],
    credentials_sent: bool,
) -> bool {
    if document_origin.is_same_origin(response_origin) {
        return true;
    }
    let [acao] = acao else {
        return false;
    };
    let acao = acao.trim_ascii();
    if acao == "*" {
        return !credentials_sent;
    }
    match document_origin {
        Origin::Tuple(..) => acao == document_origin.serialize(),
        Origin::Opaque => false,
    }
}

/// A parsed URL: scheme, credentials, host, port, path, query, fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Url {
    scheme: String,
    username: String,
    password: String,
    host: Option<Host>,
    /// Cached serialization of `host` (`None` when `host` is `None`).
    host_serialized: Option<String>,
    port: Option<u16>,
    path: Path,
    /// Cached serialization of `path`.
    pathname: String,
    query: Option<String>,
    fragment: Option<String>,
}

/// A URL path: either hierarchical segments or a cannot-be-a-base opaque string.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Path {
    /// Cannot-be-a-base path (e.g. `mailto:foo@bar`), stored encoded.
    Opaque(String),
    /// Hierarchical path segments, stored percent-encoded.
    Segments(Vec<String>),
}

impl Url {
    /// The scheme, lowercased and without the trailing `:`.
    pub fn scheme(&self) -> &str {
        &self.scheme
    }

    /// The percent-encoded username (empty string when absent).
    pub fn username(&self) -> &str {
        &self.username
    }

    /// The percent-encoded password (empty string when absent).
    pub fn password(&self) -> &str {
        &self.password
    }

    /// The serialized host: a domain, an IPv4 address, an IPv6 address in
    /// brackets (e.g. `[::1]`), or an opaque host. `None` when the URL has
    /// no host.
    pub fn host(&self) -> Option<&str> {
        self.host_serialized.as_deref()
    }

    /// The parsed host as a typed [`Host`]: domain, IPv4/IPv6 address, or
    /// opaque host. `None` when the URL has no host.
    pub fn host_enum(&self) -> Option<&Host> {
        self.host.as_ref()
    }

    /// The port, or `None` when absent (default ports are omitted).
    pub fn port(&self) -> Option<u16> {
        self.port
    }

    /// The serialized path: `/a/b` for hierarchical URLs, the opaque string
    /// for cannot-be-a-base URLs.
    pub fn path(&self) -> &str {
        &self.pathname
    }

    /// The percent-encoded query without the leading `?`, or `None`.
    pub fn query(&self) -> Option<&str> {
        self.query.as_deref()
    }

    /// The percent-encoded fragment without the leading `#`, or `None`.
    pub fn fragment(&self) -> Option<&str> {
        self.fragment.as_deref()
    }

    /// Whether the URL cannot be a base URL (has an opaque path).
    pub fn cannot_be_a_base(&self) -> bool {
        matches!(self.path, Path::Opaque(_))
    }

    /// Whether the scheme is special (`ftp`, `file`, `http`, `https`,
    /// `ws`, `wss`).
    pub fn is_special(&self) -> bool {
        is_special_scheme(&self.scheme)
    }

    /// The URL's origin per the WHATWG URL Standard: a
    /// `(scheme, host, port)` tuple for the `ftp`, `http`, `https`, `ws`,
    /// and `wss` schemes when the URL has a host (port is the URL's port,
    /// or the scheme's default when absent), and an opaque origin
    /// otherwise.
    pub fn origin(&self) -> Origin {
        let tuple_scheme = matches!(
            self.scheme.as_str(),
            "ftp" | "http" | "https" | "ws" | "wss"
        );
        if tuple_scheme && let Some(host) = &self.host_serialized {
            let port = self
                .port
                .or_else(|| default_port(&self.scheme))
                .unwrap_or(0);
            return Origin::Tuple(self.scheme.clone(), host.clone(), port);
        }
        Origin::Opaque
    }

    /// The serialized origin of this URL (see [`Origin::serialize`]),
    /// e.g. for an `Origin` request header.
    pub fn origin_serialized(&self) -> String {
        self.origin().serialize()
    }

    /// Serialize the URL per the WHATWG URL serializer.
    pub fn serialize(&self) -> String {
        self.serialize_inner(true)
    }

    /// The serialized URL: the `href` getter of the WHATWG `URL`
    /// interface (<https://url.spec.whatwg.org/#dom-url-href>).
    pub fn href(&self) -> String {
        self.serialize()
    }

    /// Serialize the URL without its fragment, e.g. for HTTP request
    /// lines and origin computations.
    pub fn serialize_without_fragment(&self) -> String {
        self.serialize_inner(false)
    }

    fn serialize_inner(&self, with_fragment: bool) -> String {
        let mut out = String::with_capacity(self.pathname.len() + 32);
        out.push_str(&self.scheme);
        out.push(':');
        if self.host_serialized.is_some() {
            out.push_str("//");
            if !self.username.is_empty() || !self.password.is_empty() {
                out.push_str(&self.username);
                if !self.password.is_empty() {
                    out.push(':');
                    out.push_str(&self.password);
                }
                out.push('@');
            }
            out.push_str(self.host_serialized.as_deref().unwrap_or(""));
            if let Some(port) = self.port {
                out.push(':');
                out.push_str(&port.to_string());
            }
        }
        // Prevent "/.//"-style paths from reparsing as an authority.
        if self.host_serialized.is_none()
            && let Path::Segments(segs) = &self.path
            && segs.len() > 1
            && segs[0].is_empty()
        {
            out.push_str("/.");
        }
        out.push_str(&self.pathname);
        if let Some(query) = &self.query {
            out.push('?');
            out.push_str(query);
        }
        if with_fragment && let Some(fragment) = &self.fragment {
            out.push('#');
            out.push_str(fragment);
        }
        out
    }

    fn set_parsed_host(&mut self, h: Host) {
        self.host_serialized = Some(h.serialize());
        self.host = Some(h);
    }

    // ------------------------------------------------------------------
    // Public URL component setters (added for the `bowser_js` Web API
    // bindings: the `URL` constructor and `location`).
    //
    // `set_host` / `set_hostname` / `set_pathname` / `set_search` /
    // `set_hash` are the WHATWG state-override algorithms, run directly on
    // the parsed URL. The rest are pragmatic re-parse-based
    // implementations. Documented divergences:
    //
    // * `set_href` parses with no base URL, exactly like the spec's `href`
    //   setter (which likewise parses without a base), so relative inputs
    //   are rejected with an error.
    // * `set_protocol` re-parses `"<scheme>:<rest>"`. The spec's scheme
    //   state override forbids some transitions v1 allows (e.g.
    //   non-special `javascript:` -> special `http:` keeps an opaque path
    //   here instead of being rejected) and preserves the
    //   cannot-be-a-base flag where v1 re-derives it from the new scheme.
    // * `set_host` / `set_hostname` run the spec's host/hostname state
    //   override directly on the parsed URL (no splice + reparse): `/`,
    //   `?`, `#` (and `\` for special schemes) terminate the host and are
    //   ignored along with everything after them, so the old
    //   path/query/fragment survive; the host is committed before the
    //   port state runs, so an invalid port keeps the new host and the
    //   old port. Only opaque-path URLs return early (`foo:/x` gains a
    //   host). Like the spec, these setters never fail.
    // * `set_port` splices the authority and re-parses the whole URL. A
    //   rejected re-parse leaves the URL unchanged, matching the spec's
    //   "terminate these steps"; like the spec, it never fails.
    // * `set_pathname` / `set_search` / `set_hash` run the spec's path
    //   start / query / fragment state override directly on the parsed URL
    //   (no probe re-parse): `Parser::parse_with_override` seeds the parser
    //   from `self` with `state_override` set, so `?`/`#` are
    //   percent-encoded instead of terminating the path or query state,
    //   and `\` splits segments only for special schemes. An earlier
    //   probe-URL version got this wrong (e.g. `set_pathname("/p?qq")`
    //   dropped the `?qq`); the override version matches the spec and the
    //   Node v24 oracle.
    // ------------------------------------------------------------------

    /// Replace the whole URL by parsing `s` with no base URL.
    ///
    /// Mirrors the spec's `href` setter: relative inputs fail and the URL
    /// is left unchanged on parse failure.
    pub fn set_href(&mut self, s: &str) -> Result<(), ParseError> {
        match parse(s, None) {
            Ok(u) => {
                *self = u;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Replace the scheme; a trailing `:` is stripped, e.g. `"https:"`.
    ///
    /// An empty scheme leaves the URL unchanged. See the divergence notes
    /// above for transitions the spec would forbid.
    pub fn set_protocol(&mut self, s: &str) -> Result<(), ParseError> {
        let scheme = s.trim_end_matches(':').to_ascii_lowercase();
        if scheme.is_empty() {
            return Ok(());
        }
        let serialized = self.serialize();
        let rest = serialized
            .split_once(':')
            .map(|(_, r)| r)
            .unwrap_or_default();
        match parse(&format!("{scheme}:{rest}"), None) {
            Ok(u) => {
                *self = u;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Replace the host, preserving username/password. `s` may include a
    /// port (`"example.com:8080"`), like the spec's `host` setter.
    ///
    /// A `/`, `?`, `#` (or `\` for special schemes) in `s` ends the host
    /// and is ignored along with everything after it; the URL's
    /// path/query/fragment are left untouched. A host the host parser
    /// rejects leaves the URL unchanged (spec: terminate these steps);
    /// the host is committed before the port state runs, so an invalid
    /// port keeps the new host and the old port.
    ///
    /// Only cannot-be-a-base URLs (opaque path) return early: a URL
    /// without a host but with a hierarchical path (e.g. `foo:/x`) gains
    /// one (`foo://h/x`).
    pub fn set_host(&mut self, s: &str) {
        self.set_host_or_hostname(s, true);
    }

    /// Replace the host without touching the port.
    ///
    /// Input containing `:` outside an IPv6 literal leaves the URL
    /// unchanged (the spec's hostname-state parse terminates there).
    pub fn set_hostname(&mut self, s: &str) {
        self.set_host_or_hostname(s, false);
    }

    /// Replace the port. An empty string removes it; a non-numeric port
    /// leaves the URL unchanged (spec: terminate).
    ///
    /// Default-port normalization (`:80` on `http:`) is applied by the
    /// re-parse, matching the spec.
    pub fn set_port(&mut self, s: &str) {
        if self.cannot_be_a_base() || self.host_serialized.is_none() {
            return;
        }
        let port = if s.is_empty() {
            None
        } else {
            // The port state only accumulates ASCII digits; any other
            // input (including a leading `+`, which Rust's `u16` FromStr
            // would accept) terminates with an empty buffer, leaving the
            // port unchanged.
            if !s.bytes().all(|b| b.is_ascii_digit()) {
                return;
            }
            match s.parse::<u16>() {
                Ok(p) => Some(p),
                Err(_) => return,
            }
        };
        let mut authority = self.userinfo_prefix();
        authority.push_str(self.host_serialized.as_deref().unwrap_or_default());
        if let Some(p) = port {
            authority.push(':');
            authority.push_str(&p.to_string());
        }
        if let Ok(u) = self.reparse_with_authority(&authority) {
            *self = u;
        }
    }

    /// Replace the path. Cannot-be-a-base URLs are left unchanged.
    ///
    /// Mirrors the spec's `pathname` setter: the path is emptied and the
    /// input is parsed in the path start state with a state override, so
    /// `?` and `#` are percent-encoded (`%3F`/`%23`) instead of starting
    /// the query or fragment, and `\` is a segment separator only for
    /// special schemes. Dot segments are resolved by the parser.
    pub fn set_pathname(&mut self, s: &str) {
        if self.cannot_be_a_base() {
            return;
        }
        if let Ok(u) = Parser::parse_with_override(s, self, State::PathStart, None) {
            self.path = u.path;
            self.pathname = u.pathname;
        }
    }

    /// Replace the query. A leading `?` is stripped; an empty value
    /// removes the query. Works on cannot-be-a-base URLs too, per spec.
    ///
    /// Mirrors the spec's `search` setter: the query is reset to `""` and
    /// the input is parsed in the query state with a state override, so
    /// `#` is percent-encoded (`%23`) instead of starting the fragment.
    /// `set_search("?")` therefore yields an empty (but present) query.
    pub fn set_search(&mut self, s: &str) {
        // UTF-8 fast path: no encoding override, so the query buffer is
        // percent-encoded directly with no intermediate byte copy (see
        // `set_search_with_encoding` for the override path).
        if s.is_empty() {
            self.query = None;
            return;
        }
        let input = s.strip_prefix('?').unwrap_or(s);
        self.query = Some(String::new());
        if let Ok(u) = Parser::parse_with_override(input, self, State::Query, None) {
            self.query = u.query;
        }
    }

    /// [`Self::set_search`] with a legacy encoding override for query
    /// percent-encoding (see [`parse_with_query_encoding`]):
    /// `encode_query` converts the query text to bytes before
    /// percent-encoding (the URL Standard's "percent-encode after
    /// encoding"). HTML's hyperlink element utils pass the document
    /// encoding's encoder here.
    pub fn set_search_with_encoding(&mut self, s: &str, encode_query: QueryEncoder<'_>) {
        if s.is_empty() {
            self.query = None;
            return;
        }
        let input = s.strip_prefix('?').unwrap_or(s);
        self.query = Some(String::new());
        if let Ok(u) = Parser::parse_with_override(input, self, State::Query, Some(encode_query)) {
            self.query = u.query;
        }
    }

    /// Replace the fragment. A leading `#` is stripped; an empty value
    /// removes the fragment. Works on cannot-be-a-base URLs too, per spec.
    ///
    /// Mirrors the spec's `hash` setter: the fragment is reset to `""` and
    /// the input is parsed in the fragment state with a state override.
    /// `set_hash("#")` therefore yields an empty (but present) fragment.
    pub fn set_hash(&mut self, s: &str) {
        if s.is_empty() {
            self.fragment = None;
            return;
        }
        let input = s.strip_prefix('#').unwrap_or(s);
        self.fragment = Some(String::new());
        if let Ok(u) = Parser::parse_with_override(input, self, State::Fragment, None) {
            self.fragment = u.fragment;
        }
    }

    /// The URL's query as [`UrlSearchParams`]: the `searchParams` getter
    /// of the WHATWG `URL` interface.
    ///
    /// A missing query parses as the empty parameter list, per spec. The
    /// returned params are a disconnected snapshot: mutating them does not
    /// touch the URL. For the spec's live, bidirectionally linked
    /// `URLSearchParams` (mutations write back into the URL's query), use
    /// [`Url::search_params_mut`]; to write a snapshot back explicitly,
    /// use [`Url::set_search_params`].
    pub fn search_params(&self) -> UrlSearchParams {
        UrlSearchParams::parse(self.query.as_deref().unwrap_or(""))
    }

    /// The URL's query as a live [`UrlSearchParamsMut`]: the WHATWG
    /// `URL`'s `searchParams` object, which the standard links
    /// bidirectionally to the URL's query.
    ///
    /// Every mutation (`append`, `delete`, `set`, `sort`) immediately
    /// re-serializes the pair list into the URL's query, exactly like the
    /// spec's "update URL's query" step: an emptied list writes
    /// `Some("")` (the `?` stays in the serialization), never `None`.
    pub fn search_params_mut(&mut self) -> UrlSearchParamsMut<'_> {
        UrlSearchParamsMut {
            params: UrlSearchParams::parse(self.query.as_deref().unwrap_or("")),
            url: self,
        }
    }

    /// Replace the URL's query with the serialization of `params`.
    ///
    /// An empty parameter list sets the query to the empty string
    /// (`Some("")`), mirroring the spec's `URLSearchParams` write-back;
    /// use [`Url::set_search`] with `""` to remove the query instead.
    pub fn set_search_params(&mut self, params: &UrlSearchParams) {
        self.query = Some(params.serialize());
    }

    /// Shared implementation for [`Url::set_host`] and
    /// [`Url::set_hostname`]: the spec's host/hostname state override run
    /// directly on the parsed URL, never a splice + reparse. When
    /// `allow_port` is false (the `hostname` setter), a `:` outside an
    /// IPv6 literal terminates the setter and leaves the URL unchanged.
    fn set_host_or_hostname(&mut self, s: &str, allow_port: bool) {
        // The setters return when the URL has an opaque path.
        if self.cannot_be_a_base() {
            return;
        }
        // Basic URL parser preprocessing for state overrides: the
        // leading/trailing C0-or-space trim applies only to fresh parses,
        // but ASCII tab/newline removal always applies.
        let input: Cow<str> = if s.bytes().any(|b| matches!(b, b'\t' | b'\n' | b'\r')) {
            Cow::Owned(
                s.chars()
                    .filter(|&c| !matches!(c, '\t' | '\n' | '\r'))
                    .collect(),
            )
        } else {
            Cow::Borrowed(s)
        };
        if self.scheme == "file" {
            // Spec: host/hostname state step 1 redirects `file:` URLs to
            // the file host state.
            self.set_file_host(&input);
            return;
        }
        let special = self.is_special();
        let bytes = input.as_bytes();
        // Scan the host: `/`, `?`, `#` (and `\` for special schemes) end
        // the host and the setter with it; everything after is ignored.
        // `:` outside `[...]` switches to the port state (`host` only).
        let mut i = 0;
        let mut inside_brackets = false;
        let mut host_end = bytes.len();
        let mut port_at = None;
        while i < bytes.len() {
            match bytes[i] {
                b':' if !inside_brackets => {
                    port_at = Some(i);
                    break;
                }
                b'[' => inside_brackets = true,
                b']' => inside_brackets = false,
                b'/' | b'?' | b'#' => {
                    host_end = i;
                    break;
                }
                b'\\' if special => {
                    host_end = i;
                    break;
                }
                _ => {}
            }
            i += 1;
        }
        if let Some(colon) = port_at {
            // Host state on `:`: an empty host, or the `hostname` setter,
            // terminates; otherwise the host is committed immediately and
            // the port state runs on the remainder.
            if colon == 0 || !allow_port {
                return;
            }
            let host = match host::parse_host(&input[..colon], special) {
                Some(h) => h,
                None => return,
            };
            self.set_parsed_host(host);
            // Port state with a state override: the first non-digit ends
            // the setter. Digits parsed so far are applied when valid (an
            // out-of-range value terminates instead); an empty port
            // leaves the previous port untouched.
            let mut j = colon + 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j == colon + 1 {
                return;
            }
            let port = match parse_port(&input[colon + 1..j]) {
                Some(p) => p,
                None => return,
            };
            self.port = if default_port(&self.scheme) == Some(port) {
                None
            } else {
                Some(port)
            };
            return;
        }
        // Terminator/EOF: commit the host and return (the state override
        // never continues into the path). An empty host is rejected for
        // special schemes, and when the URL carries credentials or a port;
        // a non-special scheme takes the empty opaque host.
        let buf = &input[..host_end];
        if buf.is_empty()
            && (special
                || !self.username.is_empty()
                || !self.password.is_empty()
                || self.port.is_some())
        {
            return;
        }
        if let Some(h) = host::parse_host(buf, special) {
            self.set_parsed_host(h);
        }
    }

    /// File-host-state override for [`Url::set_host`] and
    /// [`Url::set_hostname`] on `file:` URLs (spec: host state step 1).
    /// Mirrors the parser's file host state with the state-override
    /// differences: no Windows-drive-letter branch, and the setter ends
    /// after the host instead of continuing into the path.
    fn set_file_host(&mut self, input: &str) {
        let end = input
            .bytes()
            .position(|b| matches!(b, b'/' | b'\\' | b'?' | b'#'))
            .unwrap_or(input.len());
        let buf = &input[..end];
        if buf.is_empty() {
            self.set_parsed_host(Host::Opaque(String::new()));
            return;
        }
        let mut h = match host::parse_host(buf, true) {
            Some(h) => h,
            None => return,
        };
        if matches!(h, Host::Domain(ref d) if d == "localhost") {
            h = Host::Opaque(String::new());
        }
        self.set_parsed_host(h);
    }

    /// The `username[:password]@` prefix for authority rebuilds, or the
    /// empty string when no credentials are present.
    fn userinfo_prefix(&self) -> String {
        if self.username.is_empty() && self.password.is_empty() {
            return String::new();
        }
        let mut out = String::from(&self.username);
        if !self.password.is_empty() {
            out.push(':');
            out.push_str(&self.password);
        }
        out.push('@');
        out
    }

    /// Re-parse the URL with its authority replaced by `authority`
    /// (which must already include any userinfo prefix).
    fn reparse_with_authority(&self, authority: &str) -> Result<Url, ParseError> {
        let serialized = self.serialize();
        let start = match serialized.find("://") {
            Some(i) => i + 3,
            None => return Err(ParseError::MissingHost),
        };
        let end = serialized[start..]
            .find(['/', '?', '#'])
            .map(|i| start + i)
            .unwrap_or(serialized.len());
        let mut rebuilt = String::with_capacity(serialized.len() + authority.len());
        rebuilt.push_str(&serialized[..start]);
        rebuilt.push_str(authority);
        rebuilt.push_str(&serialized[end..]);
        parse(&rebuilt, None)
    }
}

// ---------------------------------------------------------------------------
// domain_to_unicode
// ---------------------------------------------------------------------------

/// Convert `domain` to Unicode per the URL standard's [domain to Unicode]
/// concept, i.e. UTS #46 ToUnicode processing with this crate's IDNA
/// subset.
///
/// Each label is run through the same mapping as [`domain_to_ascii`];
/// labels starting with `xn--` are then Punycode-decoded and the decoded
/// form is validated, and other labels are returned in mapped form (e.g.
/// lowercased, fullwidth folded). Returns `None` when
/// [`domain_to_ascii`] rejects the domain or when the domain exceeds 253
/// octets; an `xn--` label that fails to decode or decodes to a
/// disallowed code point — or to a code point with IDNA status "ignored"
/// (e.g. U+00AD SOFT HYPHEN, per UTS #46 §4.1) — is kept verbatim in
/// ASCII form (e.g. `xn--a`
/// yields U+0080, so [`domain_to_unicode("xn--a")`] is `Some("xn--a")`),
/// matching the spec's "if an error was recorded, return domain" step so
/// that the ASCII form round-trips.
///
/// Deviations from full UTS #46 ToUnicode, inherited from the
/// [`domain_to_ascii`] subset: code points the mapping ignores (e.g.
/// U+00AD) are removed rather than kept, and the verbatim-ASCII fallback
/// for undecodable `xn--` labels is applied per label rather than to the
/// whole domain.
///
/// DoS hardening: domains longer than 253 octets (IDNA2008 / UTS #46
/// domain cap) are rejected before any label is Punycode-decoded, and
/// `xn--` labels longer than 63 octets never reach the decoder (see
/// [`idna`]).
///
/// [domain to Unicode]: https://url.spec.whatwg.org/#concept-domain-to-unicode
pub fn domain_to_unicode(domain: &str) -> Option<String> {
    let ascii = domain_to_ascii(domain)?;
    if ascii.len() > 253 {
        return None;
    }
    let mut out = String::with_capacity(ascii.len());
    for (i, label) in ascii.split('.').enumerate() {
        if i > 0 {
            out.push('.');
        }
        match label.strip_prefix("xn--") {
            // Validate the decoded form: UTS #46 ToUnicode fails when the
            // decoded label contains disallowed code points. Per the spec
            // ("if an error was recorded, return domain"), keep the label
            // in ASCII form so the domain round-trips.
            Some(rest) => match crate::idna::decode_xn_label_tounicode(rest) {
                Some(decoded) => out.push_str(&decoded),
                None => out.push_str(label),
            },
            None => out.push_str(label),
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// UrlSearchParams
// ---------------------------------------------------------------------------

/// The `application/x-www-form-urlencoded` name-value list behind the
/// WHATWG `URLSearchParams` interface
/// (<https://url.spec.whatwg.org/#interface-urlsearchparams>).
///
/// Pairs are stored percent-decoded; [`UrlSearchParams::serialize`]
/// re-encodes them with the urlencoded percent-encode set (ASCII space as
/// `+`). Use [`Url::search_params`] to read a URL's query and
/// [`Url::set_search_params`] to write the params back; the query
/// round-trips through the pair list.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UrlSearchParams {
    params: Vec<(String, String)>,
}

impl UrlSearchParams {
    /// An empty parameter list.
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    /// Parse a query string into name-value pairs, per the
    /// `URLSearchParams` constructor: one leading `?` is stripped, the
    /// string is split on `&` (empty tokens are skipped), each token is
    /// split on the first `=`, `+` decodes to space, and percent-escapes
    /// decode as UTF-8 with U+FFFD replacement.
    pub fn parse(input: &str) -> Self {
        let input = input.strip_prefix('?').unwrap_or(input);
        let mut params = Vec::new();
        for token in input.split('&') {
            if token.is_empty() {
                continue;
            }
            let (name, value) = match token.split_once('=') {
                Some((n, v)) => (n, v),
                None => (token, ""),
            };
            params.push((decode_form_component(name), decode_form_component(value)));
        }
        Self { params }
    }

    /// Append a `(name, value)` pair.
    pub fn append(&mut self, name: &str, value: &str) {
        self.params.push((name.to_owned(), value.to_owned()));
    }

    /// Remove all pairs named `name`.
    pub fn delete(&mut self, name: &str) {
        self.params.retain(|(n, _)| n.as_str() != name);
    }

    /// The value of the first pair named `name`, if any.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.params
            .iter()
            .find(|(n, _)| n.as_str() == name)
            .map(|(_, v)| v.as_str())
    }

    /// The values of all pairs named `name`, in order.
    pub fn get_all(&self, name: &str) -> Vec<&str> {
        self.params
            .iter()
            .filter(|(n, _)| n.as_str() == name)
            .map(|(_, v)| v.as_str())
            .collect()
    }

    /// Whether any pair is named `name`.
    pub fn has(&self, name: &str) -> bool {
        self.params.iter().any(|(n, _)| n.as_str() == name)
    }

    /// Set the value of the first pair named `name` to `value`, dropping
    /// any later pairs with the same name; append when `name` is absent.
    pub fn set(&mut self, name: &str, value: &str) {
        if let Some(slot) = self.params.iter_mut().find(|(n, _)| n.as_str() == name) {
            slot.1 = value.to_owned();
        } else {
            self.append(name, value);
            return;
        }
        let mut kept_first = false;
        self.params.retain(|(n, _)| {
            if n.as_str() != name {
                return true;
            }
            if kept_first {
                false
            } else {
                kept_first = true;
                true
            }
        });
    }

    /// Sort pairs by name, comparing UTF-16 code units per the spec; the
    /// relative order of pairs with equal names is preserved (stable).
    pub fn sort(&mut self) {
        self.params.sort_by(|a, b| cmp_utf16(&a.0, &b.0));
    }

    /// Serialize to a query string: `name=value` pairs joined by `&`, with
    /// names and values percent-encoded per
    /// `application/x-www-form-urlencoded` (space as `+`).
    pub fn serialize(&self) -> String {
        let mut out = String::new();
        for (i, (name, value)) in self.params.iter().enumerate() {
            if i > 0 {
                out.push('&');
            }
            urlencoded_push(&mut out, name);
            out.push('=');
            urlencoded_push(&mut out, value);
        }
        out
    }

    /// The number of pairs.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether there are no pairs.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Iterate over `(name, value)` pairs in order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.params.iter().map(|(n, v)| (n.as_str(), v.as_str()))
    }
}

impl core::fmt::Display for UrlSearchParams {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.serialize())
    }
}

/// A live `URLSearchParams` view borrowed mutably from a [`Url`]:
/// the WHATWG `URL` interface's `searchParams` object, bidirectionally
/// linked to the URL's query.
///
/// Obtained from [`Url::search_params_mut`]. Read methods behave exactly
/// like [`UrlSearchParams`]'s; every mutation (`append`, `delete`, `set`,
/// `sort`) immediately re-serializes the pair list back into the URL's
/// query, so the view and the URL can never disagree. An emptied list
/// writes `Some("")` (the trailing `?` survives in `href()`), never
/// `None`, per the spec's query-update step.
#[derive(Debug)]
pub struct UrlSearchParamsMut<'a> {
    params: UrlSearchParams,
    url: &'a mut Url,
}

impl UrlSearchParamsMut<'_> {
    /// Re-serialize the pair list into the URL's query.
    fn sync(&mut self) {
        self.url.query = Some(self.params.serialize());
    }

    /// Append a `(name, value)` pair, updating the URL's query.
    pub fn append(&mut self, name: &str, value: &str) {
        self.params.append(name, value);
        self.sync();
    }

    /// Remove all pairs named `name`, updating the URL's query.
    pub fn delete(&mut self, name: &str) {
        self.params.delete(name);
        self.sync();
    }

    /// Set the value of the first pair named `name` to `value`, dropping
    /// any later pairs with the same name; append when `name` is absent.
    /// Updates the URL's query.
    pub fn set(&mut self, name: &str, value: &str) {
        self.params.set(name, value);
        self.sync();
    }

    /// Sort pairs by name (UTF-16 code units, stable), updating the URL's
    /// query.
    pub fn sort(&mut self) {
        self.params.sort();
        self.sync();
    }

    /// The value of the first pair named `name`, if any.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.params.get(name)
    }

    /// The values of all pairs named `name`, in order.
    pub fn get_all(&self, name: &str) -> Vec<&str> {
        self.params.get_all(name)
    }

    /// Whether any pair is named `name`.
    pub fn has(&self, name: &str) -> bool {
        self.params.has(name)
    }

    /// Serialize to a query string, as [`UrlSearchParams::serialize`].
    pub fn serialize(&self) -> String {
        self.params.serialize()
    }

    /// The number of pairs.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether there are no pairs.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Iterate over `(name, value)` pairs in order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.params.iter()
    }
}

impl core::fmt::Display for UrlSearchParamsMut<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.serialize())
    }
}

/// Decode one `application/x-www-form-urlencoded` name or value: `+`
/// becomes space, then percent-decode, then UTF-8 decode with U+FFFD
/// replacement and no BOM stripping (the Encoding Standard's "UTF-8
/// decode without BOM").
fn decode_form_component(s: &str) -> String {
    let plus_to_space = s.replace('+', " ");
    String::from_utf8_lossy(&percent_decode(&plus_to_space)).into_owned()
}

/// Whether an ASCII byte is in the `application/x-www-form-urlencoded`
/// percent-encode set: the component percent-encode set (userinfo set
/// plus `$`, `%`, `&`, `+`, `,`) plus `!`, `'`, `(`, `)`, `~`.
///
/// Callers handle ASCII space (serialized as `+`) and bytes `>= 0x7F`
/// separately.
fn urlencoded_needs_encoding(b: u8) -> bool {
    debug_assert!(b < 0x80);
    matches!(
        b,
        0x00..=0x1F
            | 0x7F
            | b'"'
            | b'#'
            | b'<'
            | b'>'
            | b'?'
            | b'^'
            | b'`'
            | b'{'
            | b'}'
            | b'/'
            | b':'
            | b';'
            | b'='
            | b'@'
            | b'['
            | b'\\'
            | b']'
            | b'|'
            | b'$'
            | b'%'
            | b'&'
            | b'+'
            | b','
            | b'!'
            | b'\''
            | b'('
            | b')'
            | b'~'
    )
}

/// Append `input` to `out` percent-encoded for
/// `application/x-www-form-urlencoded` serialization: each UTF-8 byte in
/// the urlencoded percent-encode set becomes `%XX` (uppercase hex), ASCII
/// space becomes `+`, and everything else passes through unchanged.
fn urlencoded_push(out: &mut String, input: &str) {
    for b in input.bytes() {
        if b == b' ' {
            out.push('+');
        } else if b >= 0x7F || urlencoded_needs_encoding(b) {
            out.push('%');
            out.push(hex_digit(b >> 4));
            out.push(hex_digit(b & 0x0F));
        } else {
            out.push(b as char);
        }
    }
}

/// Compare two strings by UTF-16 code units, for
/// [`UrlSearchParams::sort`]. (Byte-wise UTF-8 comparison would order
/// astral code points differently from the spec's code-unit comparison.)
fn cmp_utf16(a: &str, b: &str) -> Ordering {
    let mut a = a.encode_utf16();
    let mut b = b.encode_utf16();
    loop {
        match (a.next(), b.next()) {
            (Some(x), Some(y)) => {
                let ord = x.cmp(&y);
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
        }
    }
}

/// Parse `input` as a URL, resolving against `base` when supplied.
///
/// Implements the WHATWG basic URL parser (without state override).
pub fn parse(input: &str, base: Option<&Url>) -> Result<Url, ParseError> {
    Parser::new(input, base).run()
}

/// Converts query text to bytes before percent-encoding: the URL
/// Standard's "percent-encode after encoding". HTML's hyperlink element
/// utils pass the document encoding's encoder
/// (`bowser_encoding::encode`); a UTF-8 byte copy
/// (`|s, out| out.extend_from_slice(s.as_bytes())`) reproduces the
/// default behavior.
pub type QueryEncoder<'a> = &'a dyn Fn(&str, &mut Vec<u8>);

/// Parse `input` as a URL with a legacy encoding override for query
/// percent-encoding (the URL Standard's "encoding override").
///
/// `encode_query` converts query text to bytes before percent-encoding
/// (the URL Standard's "percent-encode after encoding"); the bytes are
/// then percent-encoded opaquely with the query/special-query set.
/// HTML's hyperlink element utils pass the document encoding's encoder
/// here (`bowser_encoding::encode`); when the document encoding has no
/// encoder, UTF-8 bytes (`|s, out| out.extend_from_slice(s.as_bytes())`)
/// reproduce the default behavior.
///
/// Only the query is affected: fragment percent-encoding is always
/// UTF-8, per spec.
pub fn parse_with_query_encoding(
    input: &str,
    base: Option<&Url>,
    encode_query: QueryEncoder<'_>,
) -> Result<Url, ParseError> {
    Parser::new_with_query_encoding(input, base, encode_query).run()
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Special schemes per the URL standard.
pub fn is_special_scheme(scheme: &str) -> bool {
    matches!(scheme, "ftp" | "file" | "http" | "https" | "ws" | "wss")
}

/// Default port for a special scheme, if any.
fn default_port(scheme: &str) -> Option<u16> {
    match scheme {
        "ftp" => Some(21),
        "http" | "ws" => Some(80),
        "https" => Some(443),
        "wss" => Some(443),
        _ => None,
    }
}

/// A Windows drive letter: an ASCII alpha followed by `:` or `|`.
fn is_windows_drive_letter(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic())
        && matches!(chars.next(), Some(':' | '|'))
        && chars.next().is_none()
}

/// A normalized Windows drive letter: an ASCII alpha followed by `:`.
fn is_normalized_windows_drive_letter(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic())
        && matches!(chars.next(), Some(':'))
        && chars.next().is_none()
}

/// Whether `s` starts with a Windows drive letter: length >= 2, first two
/// are a drive letter, and length == 2 or the third code point is one of
/// `/`, `\`, `?`, `#`.
fn starts_with_windows_drive_letter(s: &str) -> bool {
    let mut chars = s.chars();
    let first = chars.next();
    let second = chars.next();
    let ok =
        matches!(first, Some(c) if c.is_ascii_alphabetic()) && matches!(second, Some(':' | '|'));
    if !ok {
        return false;
    }
    match chars.next() {
        None => true,
        Some(c) => matches!(c, '/' | '\\' | '?' | '#'),
    }
}

/// A single-dot path segment: `.` or `%2e` (ASCII case-insensitive).
fn is_single_dot_segment(s: &str) -> bool {
    s == "." || s.eq_ignore_ascii_case("%2e")
}

/// A double-dot path segment: `..`, `.%2e`, `%2e.`, `%2e%2e`.
fn is_double_dot_segment(s: &str) -> bool {
    s == ".."
        || s.eq_ignore_ascii_case(".%2e")
        || s.eq_ignore_ascii_case("%2e.")
        || s.eq_ignore_ascii_case("%2e%2e")
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    SchemeStart,
    Scheme,
    NoScheme,
    SpecialRelativeOrAuthority,
    PathOrAuthority,
    Relative,
    RelativeSlash,
    SpecialAuthoritySlashes,
    SpecialAuthorityIgnoreSlashes,
    Authority,
    Host,
    Port,
    File,
    FileSlash,
    FileHost,
    PathStart,
    Path,
    OpaquePath,
    Query,
    Fragment,
    Done,
}

struct Parser<'a> {
    /// The preprocessed input. Walked as bytes: the driver decodes one code
    /// point per iteration, ASCII on a fast path, so the common case never
    /// pays for a `Vec<char>` copy of the input.
    input: Cow<'a, str>,
    base: Option<&'a Url>,
    url: Url,
    wpath: Path,
    state: State,
    /// Whether basic URL parse was invoked with a state override (the
    /// component setters). Under an override, `?`/`#` do not terminate the
    /// path or query states; they are percent-encoded instead, per
    /// <https://url.spec.whatwg.org/#path-state> and
    /// <https://url.spec.whatwg.org/#query-state>.
    state_override: bool,
    /// Byte offset into `input`.
    pointer: isize,
    /// Byte length of the code point at `pointer` (0 at EOF); used to rewind
    /// by exactly one code point.
    cur_len: usize,
    /// Set by a state to process the current code point again in the next
    /// state instead of advancing (and to bypass the EOF break check once,
    /// which matters when `cur_len` is 0).
    reprocess: bool,
    buffer: String,
    at_sign_seen: bool,
    inside_brackets: bool,
    password_token_seen: bool,
    /// Legacy encoding override for query percent-encoding (the URL
    /// Standard's "encoding override", e.g. the document encoding for
    /// HTML's hyperlink element utils). `None` means UTF-8: the query
    /// buffer is percent-encoded directly. `Some` means "percent-encode
    /// after encoding": the buffer is converted to bytes with the
    /// callback first, then the bytes are percent-encoded opaquely.
    /// The fragment state always uses UTF-8, per spec.
    query_encoder: Option<QueryEncoder<'a>>,
    /// Scratch for the encoder's output, reused across query flushes.
    query_bytes: Vec<u8>,
}

impl<'a> Parser<'a> {
    /// The basic URL parser's input preprocessing for a fresh parse:
    /// remove leading/trailing C0 controls and space, and remove all
    /// ASCII tab and newline. State-override parses (the component
    /// setters) use [`Self::preprocess_override_input`] instead: the spec
    /// only strips tab/newline there, so leading/trailing spaces survive
    /// to be percent-encoded (`set_pathname(" ")` -> `/%20`, per Node).
    fn preprocess_input(input: &'a str) -> Cow<'a, str> {
        // Remove leading/trailing C0 controls and space; remove all
        // ASCII tab and newline. The tab/newline filter only allocates when
        // the input actually contains one.
        let trimmed = input.trim_matches(|c: char| (c as u32) <= 0x20);
        if trimmed.bytes().any(|b| matches!(b, b'\t' | b'\n' | b'\r')) {
            trimmed
                .chars()
                .filter(|&c| c != '\t' && c != '\n' && c != '\r')
                .collect::<String>()
                .into()
        } else {
            trimmed.into()
        }
    }

    /// Preprocessing for state-override parses (the `pathname`/`search`/
    /// `hash` setters): only ASCII tab and newline are removed. Unlike a
    /// fresh parse, leading/trailing C0 controls and space are kept, so
    /// the states percent-encode them.
    fn preprocess_override_input(input: &'a str) -> Cow<'a, str> {
        // The tab/newline filter only allocates when the input actually
        // contains one.
        if input.bytes().any(|b| matches!(b, b'\t' | b'\n' | b'\r')) {
            input
                .chars()
                .filter(|&c| !matches!(c, '\t' | '\n' | '\r'))
                .collect::<String>()
                .into()
        } else {
            input.into()
        }
    }

    fn new(input: &'a str, base: Option<&'a Url>) -> Self {
        Parser {
            input: Self::preprocess_input(input),
            base,
            url: Url {
                scheme: String::new(),
                username: String::new(),
                password: String::new(),
                host: None,
                host_serialized: None,
                port: None,
                path: Path::Segments(Vec::new()),
                pathname: String::new(),
                query: None,
                fragment: None,
            },
            wpath: Path::Segments(Vec::new()),
            state: State::SchemeStart,
            state_override: false,
            pointer: 0,
            cur_len: 0,
            reprocess: false,
            buffer: String::new(),
            at_sign_seen: false,
            inside_brackets: false,
            password_token_seen: false,
            query_encoder: None,
            query_bytes: Vec::new(),
        }
    }

    /// [`Self::new`] with a legacy encoding override for query
    /// percent-encoding (see [`parse_with_query_encoding`]).
    fn new_with_query_encoding(
        input: &'a str,
        base: Option<&'a Url>,
        encode_query: QueryEncoder<'a>,
    ) -> Self {
        let mut p = Self::new(input, base);
        p.query_encoder = Some(encode_query);
        p
    }

    /// Run basic URL parse on `input` with `url` as the URL and `state` as
    /// the state override, returning the mutated URL. This is how the
    /// WHATWG `pathname`/`search`/`hash` setters are specified: the states
    /// run with `state_override` set, so `?`/`#` are percent-encoded rather
    /// than terminating the path or query state. `state` must be one of
    /// [`State::PathStart`], [`State::Query`], or [`State::Fragment`]; the
    /// other states are not meaningful overrides and would misbehave.
    ///
    /// `encode_query` is the encoding override for the query state
    /// ([`parse_with_query_encoding`]); `None` keeps UTF-8.
    fn parse_with_override(
        input: &'a str,
        url: &Url,
        state: State,
        encode_query: Option<QueryEncoder<'a>>,
    ) -> Result<Url, ParseError> {
        debug_assert!(matches!(
            state,
            State::PathStart | State::Query | State::Fragment
        ));
        Parser {
            input: Self::preprocess_override_input(input),
            base: None,
            url: url.clone(),
            // The pathname setter empties the path before parsing; the
            // query/fragment setters seed their component to "" instead and
            // leave the path alone.
            wpath: match state {
                State::PathStart => Path::Segments(Vec::new()),
                _ => url.path.clone(),
            },
            state,
            state_override: true,
            pointer: 0,
            cur_len: 0,
            reprocess: false,
            buffer: String::new(),
            at_sign_seen: false,
            inside_brackets: false,
            password_token_seen: false,
            query_encoder: encode_query,
            query_bytes: Vec::new(),
        }
        .run()
    }

    /// Process the current code point again in the next state: the driver
    /// loop skips its advance (and its EOF break check) for one iteration.
    /// This replaces the old `pointer -= 1` idiom, which was net-zero
    /// against the loop's unconditional `pointer += 1`.
    fn reprocess(&mut self) {
        self.reprocess = true;
    }

    /// Whether the code point after the pointer is `c`.
    ///
    /// This implements the spec's "remaining starts with" checks. Every
    /// call site tests an ASCII code point immediately after an ASCII code
    /// point, so this is a single byte comparison.
    fn remaining_starts_with(&self, c: char) -> bool {
        debug_assert!(c.is_ascii());
        let next = self
            .pointer
            .checked_add(1)
            .and_then(|p| self.input.as_bytes().get(p as usize));
        next == Some(&(c as u8))
    }

    /// The substring from the pointer to the end of input (borrowed).
    ///
    /// The driver keeps the pointer on a char boundary, but this is
    /// attacker-reachable code: degrade to the empty string rather than
    /// panic if the invariant ever breaks.
    fn substring_from_pointer(&self) -> &str {
        let idx = self.pointer.max(0) as usize;
        self.input.get(idx..).unwrap_or("")
    }

    fn is_special(&self) -> bool {
        is_special_scheme(&self.url.scheme)
    }

    fn run(mut self) -> Result<Url, ParseError> {
        loop {
            // Decode the code point at the byte pointer: ASCII is a cast,
            // multibyte decodes from the (valid UTF-8) input slice.
            let (c, len) = match self.input.as_bytes().get(self.pointer as usize) {
                None => (None, 0),
                Some(&b) if b < 0x80 => (Some(b as char), 1),
                Some(_) => {
                    // The byte at the pointer is non-ASCII, so the pointer
                    // must be on a char boundary of the valid-UTF-8 input.
                    // Degrade to end-of-input instead of panicking if the
                    // invariant ever breaks.
                    let ch = self
                        .input
                        .get(self.pointer as usize..)
                        .and_then(|s| s.chars().next());
                    debug_assert!(ch.is_some(), "parser pointer is on a char boundary");
                    match ch {
                        Some(ch) => (Some(ch), ch.len_utf8()),
                        None => (None, 0),
                    }
                }
            };
            self.cur_len = len;
            match self.state {
                State::SchemeStart => self.scheme_start(c),
                State::Scheme => self.scheme(c)?,
                State::NoScheme => self.no_scheme(c)?,
                State::SpecialRelativeOrAuthority => self.special_relative_or_authority(c),
                State::PathOrAuthority => self.path_or_authority(c),
                State::Relative => self.relative(c)?,
                State::RelativeSlash => self.relative_slash(c)?,
                State::SpecialAuthoritySlashes => self.special_authority_slashes(c),
                State::SpecialAuthorityIgnoreSlashes => self.special_authority_ignore_slashes(c),
                State::Authority => self.authority(c)?,
                State::Host => self.host_state(c)?,
                State::Port => self.port_state(c)?,
                State::File => self.file_state(c)?,
                State::FileSlash => self.file_slash(c),
                State::FileHost => self.file_host(c)?,
                State::PathStart => self.path_start(c),
                State::Path => self.path_state(c),
                State::OpaquePath => self.opaque_path(c),
                State::Query => self.query_state(c),
                State::Fragment => self.fragment_state(c),
                State::Done => break,
            }
            if self.reprocess {
                self.reprocess = false;
                continue;
            }
            if self.pointer >= 0 && self.pointer as usize >= self.input.len() {
                break;
            }
            self.pointer += len as isize;
        }
        self.finish()
    }

    /// Compute the serialized pathname and finalize the URL.
    fn finish(mut self) -> Result<Url, ParseError> {
        let pathname = match &self.wpath {
            Path::Opaque(s) => s.clone(),
            Path::Segments(segs) => {
                let mut out = String::new();
                for seg in segs {
                    out.push('/');
                    out.push_str(seg);
                }
                out
            }
        };
        self.url.pathname = pathname;
        self.url.path = self.wpath;
        Ok(self.url)
    }

    // -- states ----------------------------------------------------------

    fn scheme_start(&mut self, c: Option<char>) {
        match c {
            Some(ch) if ch.is_ascii_alphabetic() => {
                self.buffer.push(ch.to_ascii_lowercase());
                self.state = State::Scheme;
            }
            _ => {
                self.state = State::NoScheme;
                self.reprocess();
            }
        }
    }

    fn scheme(&mut self, c: Option<char>) -> Result<(), ParseError> {
        match c {
            Some(ch) if ch.is_ascii_alphanumeric() || matches!(ch, '+' | '-' | '.') => {
                self.buffer.push(ch.to_ascii_lowercase());
            }
            Some(':') => {
                self.url.scheme = core::mem::take(&mut self.buffer);
                if self.url.scheme == "file" {
                    // Non-fatal if remaining does not start with "//".
                    self.state = State::File;
                } else if self.is_special()
                    && self.base.is_some_and(|b| b.scheme == self.url.scheme)
                {
                    self.state = State::SpecialRelativeOrAuthority;
                } else if self.is_special() {
                    self.state = State::SpecialAuthoritySlashes;
                } else if self.remaining_starts_with('/') {
                    self.state = State::PathOrAuthority;
                    self.pointer += 1;
                } else {
                    self.wpath = Path::Opaque(String::new());
                    self.state = State::OpaquePath;
                }
            }
            _ => {
                self.buffer.clear();
                self.state = State::NoScheme;
                // "Start over (from the first code point in input)."
                self.pointer = 0;
                self.reprocess = true;
            }
        }
        Ok(())
    }

    fn no_scheme(&mut self, c: Option<char>) -> Result<(), ParseError> {
        let Some(base) = self.base else {
            return Err(ParseError::MissingScheme);
        };
        let base_opaque = matches!(base.path, Path::Opaque(_));
        if base_opaque && c != Some('#') {
            return Err(ParseError::UnresolvableReference);
        }
        if base_opaque && c == Some('#') {
            self.url.scheme = base.scheme.clone();
            self.wpath = base.path.clone();
            self.url.query = base.query.clone();
            self.url.fragment = Some(String::new());
            self.state = State::Fragment;
        } else if base.scheme != "file" {
            self.state = State::Relative;
            self.reprocess();
        } else {
            self.state = State::File;
            self.reprocess();
        }
        Ok(())
    }

    fn special_relative_or_authority(&mut self, c: Option<char>) {
        if c == Some('/') && self.remaining_starts_with('/') {
            self.state = State::SpecialAuthorityIgnoreSlashes;
            // Skip the second '/'; the driver also advances past the first.
            self.pointer += 1;
        } else {
            // Non-fatal special-scheme-missing-following-solidus.
            self.state = State::Relative;
            self.reprocess();
        }
    }

    fn path_or_authority(&mut self, c: Option<char>) {
        if c == Some('/') {
            self.state = State::Authority;
        } else {
            self.state = State::Path;
            self.reprocess();
        }
    }

    fn relative(&mut self, c: Option<char>) -> Result<(), ParseError> {
        // base is non-null and base.scheme != "file" here: the only
        // transitions into this state are from `no_scheme` (which returns
        // MissingScheme when base is absent) and from
        // `special_relative_or_authority` (which requires a base whose
        // scheme matches). Degrade to a parse error rather than panic if
        // the invariant ever breaks.
        let Some(base) = self.base else {
            return Err(ParseError::UnresolvableReference);
        };
        self.url.scheme = base.scheme.clone();
        if c == Some('/') {
            self.state = State::RelativeSlash;
        } else if self.is_special() && c == Some('\\') {
            // Non-fatal invalid-reverse-solidus.
            self.state = State::RelativeSlash;
        } else {
            self.url.username = base.username.clone();
            self.url.password = base.password.clone();
            if let Some(h) = &base.host {
                self.url.set_parsed_host(h.clone());
            }
            self.url.port = base.port;
            self.wpath = base.path.clone();
            self.url.query = base.query.clone();
            if c == Some('?') {
                self.url.query = Some(String::new());
                self.state = State::Query;
            } else if c == Some('#') {
                self.url.fragment = Some(String::new());
                self.state = State::Fragment;
            } else if c.is_some() {
                self.url.query = None;
                Self::shorten(&mut self.wpath, &self.url.scheme);
                self.state = State::Path;
                self.reprocess();
            } else {
                self.state = State::Done;
            }
        }
        Ok(())
    }

    fn relative_slash(&mut self, c: Option<char>) -> Result<(), ParseError> {
        // This state is only entered from `relative`, which already
        // requires a base URL; degrade to a parse error rather than
        // panic if the invariant ever breaks.
        let Some(base) = self.base else {
            return Err(ParseError::UnresolvableReference);
        };
        if self.is_special() && matches!(c, Some('/' | '\\')) {
            // Non-fatal invalid-reverse-solidus for '\'.
            self.state = State::SpecialAuthorityIgnoreSlashes;
        } else if c == Some('/') {
            self.state = State::Authority;
        } else {
            self.url.username = base.username.clone();
            self.url.password = base.password.clone();
            if let Some(h) = &base.host {
                self.url.set_parsed_host(h.clone());
            }
            self.url.port = base.port;
            self.state = State::Path;
            self.reprocess();
        }
        Ok(())
    }

    fn special_authority_slashes(&mut self, c: Option<char>) {
        if c == Some('/') && self.remaining_starts_with('/') {
            self.state = State::SpecialAuthorityIgnoreSlashes;
            // Skip the second '/'; the driver also advances past the first.
            self.pointer += 1;
        } else {
            // Non-fatal special-scheme-missing-following-solidus.
            self.state = State::SpecialAuthorityIgnoreSlashes;
            self.reprocess();
        }
    }

    fn special_authority_ignore_slashes(&mut self, c: Option<char>) {
        if !matches!(c, Some('/' | '\\')) {
            self.state = State::Authority;
            self.reprocess();
        }
        // Otherwise consume the slash and stay (non-fatal validation for '\').
    }

    fn authority(&mut self, c: Option<char>) -> Result<(), ParseError> {
        if c == Some('@') {
            // Non-fatal invalid-credentials.
            if self.at_sign_seen {
                self.buffer.insert_str(0, "%40");
            }
            // Batch the percent-encoding: encode the username and password
            // slices once, appending directly to their destinations instead
            // of allocating per code point or per slice. `clear` (not
            // `take`) keeps the buffer's allocation for the next credentials
            // run or host segment.
            if self.password_token_seen {
                percent_encode_into(&mut self.url.password, &self.buffer, EncodeSet::Userinfo);
            } else {
                match self.buffer.find(':') {
                    Some(i) => {
                        self.password_token_seen = true;
                        let (user, pass) = self.buffer.split_at(i);
                        percent_encode_into(&mut self.url.username, user, EncodeSet::Userinfo);
                        percent_encode_into(
                            &mut self.url.password,
                            &pass[1..],
                            EncodeSet::Userinfo,
                        );
                    }
                    None => {
                        percent_encode_into(
                            &mut self.url.username,
                            &self.buffer,
                            EncodeSet::Userinfo,
                        );
                    }
                }
            }
            self.buffer.clear();
            self.at_sign_seen = true;
            return Ok(());
        }
        let special = self.is_special();
        let terminator = matches!(c, None | Some('/' | '?' | '#')) || (special && c == Some('\\'));
        if terminator {
            if self.at_sign_seen && self.buffer.is_empty() {
                return Err(ParseError::MissingHost);
            }
            // Rewind to the start of the buffered host: the buffer holds
            // exactly the bytes consumed since authority entry (or the last
            // '@'). The terminator itself is reprocessed by the host state
            // via its own rewind. Saturate instead of wrapping if the
            // invariant ever breaks on hostile input.
            self.pointer = self.pointer.saturating_sub(self.buffer.len() as isize);
            debug_assert!(self.pointer >= 0);
            self.reprocess = true;
            self.buffer.clear();
            self.state = State::Host;
        } else if let Some(ch) = c {
            // `terminator` is false, and it matches `None`, so `c` is a
            // code point here by construction. Never panic on hostile
            // input: if the invariant ever broke, dropping the code point
            // is the graceful degradation.
            debug_assert!(c.is_some(), "non-terminator is a code point");
            self.buffer.push(ch);
        }
        Ok(())
    }

    fn host_state(&mut self, c: Option<char>) -> Result<(), ParseError> {
        let special = self.is_special();
        if c == Some(':') && !self.inside_brackets {
            if self.buffer.is_empty() {
                return Err(ParseError::MissingHost);
            }
            let h = host::parse_host(&self.buffer, special).ok_or(ParseError::InvalidHost)?;
            self.url.set_parsed_host(h);
            self.buffer.clear();
            self.state = State::Port;
            return Ok(());
        }
        let terminator = matches!(c, None | Some('/' | '?' | '#')) || (special && c == Some('\\'));
        if terminator {
            self.reprocess();
            if special && self.buffer.is_empty() {
                return Err(ParseError::MissingHost);
            }
            let h = host::parse_host(&self.buffer, special).ok_or(ParseError::InvalidHost)?;
            self.url.set_parsed_host(h);
            self.buffer.clear();
            self.state = State::PathStart;
            return Ok(());
        }
        match c {
            Some('[') => self.inside_brackets = true,
            Some(']') => self.inside_brackets = false,
            _ => {}
        }
        // `terminator` is false, and it matches `None`, so `c` is a code
        // point here by construction. Never panic on hostile input: if the
        // invariant ever broke, dropping the code point is the graceful
        // degradation.
        debug_assert!(c.is_some(), "non-terminator is a code point");
        if let Some(ch) = c {
            self.buffer.push(ch);
        }
        Ok(())
    }

    fn port_state(&mut self, c: Option<char>) -> Result<(), ParseError> {
        let special = self.is_special();
        match c {
            Some(ch) if ch.is_ascii_digit() => {
                self.buffer.push(ch);
            }
            _ if matches!(c, None | Some('/' | '?' | '#')) || (special && c == Some('\\')) => {
                if !self.buffer.is_empty() {
                    let port = parse_port(&self.buffer).ok_or(ParseError::InvalidPort)?;
                    self.url.port = Some(port);
                    if default_port(&self.url.scheme) == Some(port) {
                        self.url.port = None;
                    }
                    self.buffer.clear();
                }
                self.state = State::PathStart;
                self.reprocess();
            }
            _ => return Err(ParseError::InvalidPort),
        }
        Ok(())
    }

    fn file_state(&mut self, c: Option<char>) -> Result<(), ParseError> {
        self.url.scheme = "file".to_string();
        self.url.set_parsed_host(Host::Opaque(String::new()));
        if matches!(c, Some('/' | '\\')) {
            // Non-fatal invalid-reverse-solidus for '\'.
            self.state = State::FileSlash;
            return Ok(());
        }
        // Only a `file:` base is usable here (`no_scheme` only enters this
        // state with one; `scheme` enters it with no usable base, handled
        // below). Degrade to path parsing rather than panic if the
        // invariant ever breaks.
        let Some(base) = self.base.filter(|b| b.scheme == "file") else {
            // No usable file base: parse the remainder as a path.
            self.state = State::Path;
            self.reprocess();
            return Ok(());
        };
        if let Some(h) = &base.host {
            self.url.set_parsed_host(h.clone());
        } else {
            self.url.host_serialized = None;
            self.url.host = None;
        }
        self.wpath = base.path.clone();
        self.url.query = base.query.clone();
        match c {
            Some('?') => {
                self.url.query = Some(String::new());
                self.state = State::Query;
            }
            Some('#') => {
                self.url.fragment = Some(String::new());
                self.state = State::Fragment;
            }
            Some(_) => {
                self.url.query = None;
                if !starts_with_windows_drive_letter(self.substring_from_pointer()) {
                    Self::shorten(&mut self.wpath, &self.url.scheme);
                } else {
                    // Non-fatal file-invalid-Windows-drive-letter.
                    self.wpath = Path::Segments(Vec::new());
                }
                self.state = State::Path;
                self.reprocess();
            }
            None => {
                // EOF with a file base: host, path, and query are already
                // copied from base; remain in file state so the driver
                // loop terminates without appending a path segment.
            }
        }
        Ok(())
    }

    fn file_slash(&mut self, c: Option<char>) {
        if matches!(c, Some('/' | '\\')) {
            // Non-fatal invalid-reverse-solidus for '\'.
            self.state = State::FileHost;
            return;
        }
        // The `is_some_and` guard proves `base` is `Some("file")` inside
        // this block; `if let` keeps even a broken invariant from
        // panicking on hostile input.
        if let Some(base) = self.base
            && base.scheme == "file"
        {
            if let Some(h) = &base.host {
                self.url.set_parsed_host(h.clone());
            }
            // Windows drive letter quirk: preserve base's drive letter.
            if !starts_with_windows_drive_letter(self.substring_from_pointer())
                && let Path::Segments(segs) = &base.path
                && let Some(first) = segs.first()
                && is_normalized_windows_drive_letter(first)
                && let Path::Segments(wsegs) = &mut self.wpath
            {
                wsegs.push(first.clone());
            }
        }
        self.state = State::Path;
        self.reprocess();
    }

    fn file_host(&mut self, c: Option<char>) -> Result<(), ParseError> {
        if !matches!(c, None | Some('/' | '\\' | '?' | '#')) {
            // `c` is not a terminator (and not EOF), so it is a code point
            // here by construction. Never panic on hostile input: if the
            // invariant ever broke, dropping the code point is the graceful
            // degradation.
            debug_assert!(c.is_some(), "non-terminator is a code point");
            if let Some(ch) = c {
                self.buffer.push(ch);
            }
            return Ok(());
        }
        self.reprocess();
        if is_windows_drive_letter(&self.buffer) {
            // Non-fatal file-invalid-Windows-drive-letter-host; buffer is
            // kept for the path state.
            self.state = State::Path;
            return Ok(());
        }
        if self.buffer.is_empty() {
            self.url.set_parsed_host(Host::Opaque(String::new()));
            self.state = State::PathStart;
            return Ok(());
        }
        let mut h = host::parse_host(&self.buffer, true).ok_or(ParseError::InvalidHost)?;
        if matches!(h, Host::Domain(ref d) if d == "localhost") {
            h = Host::Opaque(String::new());
        }
        self.url.set_parsed_host(h);
        self.buffer.clear();
        self.state = State::PathStart;
        Ok(())
    }

    fn path_start(&mut self, c: Option<char>) {
        if self.is_special() {
            // Non-fatal invalid-reverse-solidus for '\'.
            self.state = State::Path;
            if !matches!(c, Some('/' | '\\')) {
                self.reprocess();
            }
            return;
        }
        // Non-special schemes: under a state override `?`/`#` are ordinary
        // path characters (percent-encoded below in the path state), not
        // query/fragment delimiters.
        match c {
            Some('?') if !self.state_override => {
                self.url.query = Some(String::new());
                self.state = State::Query;
            }
            Some('#') if !self.state_override => {
                self.url.fragment = Some(String::new());
                self.state = State::Fragment;
            }
            Some(_) => {
                self.state = State::Path;
                if c != Some('/') {
                    self.reprocess();
                }
            }
            // Spec path start state, last step: with a state override and
            // no host, an empty input still yields one empty segment, so
            // path-only URLs keep their slash (`foo:/`, not `foo:`).
            // Special schemes never reach here (first branch above always
            // moves to the path state, which appends the segment at EOF).
            None => {
                if self.state_override && self.url.host.is_none() {
                    Self::push_segment(&mut self.wpath, String::new());
                }
                self.state = State::Done;
            }
        }
    }

    fn path_state(&mut self, c: Option<char>) {
        let special = self.is_special();
        // Under a state override, `?`/`#` are not terminators: they fall
        // through to the percent-encoding branch below (both are in the
        // path encode set).
        let terminator = matches!(c, None | Some('/'))
            || (special && c == Some('\\'))
            || (!self.state_override && matches!(c, Some('?' | '#')));
        if !terminator {
            // Non-fatal invalid-URL-unit checks skipped. Buffer the raw
            // code point; the segment is percent-encoded once, in full, at
            // the terminator below. `terminator` is false, and it matches
            // `None`, so `c` is a code point here by construction. Never
            // panic on hostile input: if the invariant ever broke, dropping
            // the code point is the graceful degradation.
            debug_assert!(c.is_some(), "non-terminator is a code point");
            if let Some(ch) = c {
                self.buffer.push(ch);
            }
            return;
        }
        // Non-fatal invalid-reverse-solidus for '\'.
        if is_double_dot_segment(&self.buffer) {
            Self::shorten(&mut self.wpath, &self.url.scheme);
            if !matches!(c, Some('/')) && !(special && c == Some('\\')) {
                Self::push_segment(&mut self.wpath, String::new());
            }
        } else if is_single_dot_segment(&self.buffer)
            && !matches!(c, Some('/'))
            && !(special && c == Some('\\'))
        {
            Self::push_segment(&mut self.wpath, String::new());
        } else if !is_single_dot_segment(&self.buffer) {
            if self.url.scheme == "file"
                && let Path::Segments(segs) = &self.wpath
                && segs.is_empty()
                && is_windows_drive_letter(&self.buffer)
            {
                self.buffer.replace_range(1..2, ":");
            }
            // The buffer holds raw code points ('.' and '%' are untouched
            // by the path encode set, so the dot checks above are exact);
            // encode the whole segment in one pass, straight into the
            // segment string. `clear` keeps the buffer's allocation for the
            // next segment.
            let mut segment = String::with_capacity(self.buffer.len());
            percent_encode_into(&mut segment, &self.buffer, EncodeSet::Path);
            Self::push_segment(&mut self.wpath, segment);
        }
        // Clear for dot segments too: the checks above consumed the buffer.
        self.buffer.clear();
        match c {
            Some('?') => {
                self.url.query = Some(String::new());
                self.state = State::Query;
            }
            Some('#') => {
                self.url.fragment = Some(String::new());
                self.state = State::Fragment;
            }
            _ => {}
        }
    }

    fn opaque_path(&mut self, c: Option<char>) {
        match c {
            Some('?') => {
                self.flush_opaque_buffer();
                self.url.query = Some(String::new());
                self.state = State::Query;
            }
            Some('#') => {
                self.flush_opaque_buffer();
                self.url.fragment = Some(String::new());
                self.state = State::Fragment;
            }
            Some(' ') => {
                // Non-fatal invalid-URL-unit. A space followed by '?' or
                // '#' is encoded up front ('%' is not in the C0 set, so the
                // pre-encoded form survives the batch encode below).
                if self.remaining_starts_with('?') || self.remaining_starts_with('#') {
                    self.buffer.push_str("%20");
                } else {
                    self.buffer.push(' ');
                }
            }
            // Buffer the raw code point; the run is percent-encoded once,
            // in full, when the path ends.
            Some(ch) => {
                self.buffer.push(ch);
            }
            None => {
                self.flush_opaque_buffer();
                self.state = State::Done;
            }
        }
    }

    /// Percent-encode the code points buffered by [`Self::opaque_path`]
    /// (C0 set) and append them to the opaque path, without an intermediate
    /// allocation. `clear` keeps the buffer's allocation for a later run.
    fn flush_opaque_buffer(&mut self) {
        if let Path::Opaque(o) = &mut self.wpath {
            percent_encode_into(o, &self.buffer, EncodeSet::C0);
        }
        self.buffer.clear();
    }

    fn query_state(&mut self, c: Option<char>) {
        let set = if self.is_special() {
            EncodeSet::SpecialQuery
        } else {
            EncodeSet::Query
        };
        match c {
            // Under a state override, `#` does not end the query: it is
            // buffered and percent-encoded at EOF, like any other query
            // character (`#` is in the query encode set).
            Some('#') if !self.state_override => {
                self.flush_query_buffer(set);
                self.url.fragment = Some(String::new());
                self.state = State::Fragment;
            }
            None => {
                self.flush_query_buffer(set);
                self.state = State::Done;
            }
            Some(ch) => {
                // Non-fatal invalid-URL-unit checks skipped.
                self.buffer.push(ch);
            }
        }
    }

    /// Percent-encode the code points buffered by [`Self::query_state`]
    /// (query/special-query set) and append them to the query, without an
    /// intermediate allocation. `clear` keeps the buffer's allocation.
    ///
    /// With an encoding override ([`Self::query_encoder`]), the URL
    /// Standard's "percent-encode after encoding": the buffer is
    /// converted to bytes with the override encoding first, then the
    /// bytes are percent-encoded opaquely.
    fn flush_query_buffer(&mut self, set: EncodeSet) {
        if let Some(encode) = self.query_encoder {
            self.query_bytes.clear();
            encode(&self.buffer, &mut self.query_bytes);
            let query = self.url.query.get_or_insert_with(String::new);
            percent_encode_bytes_into(query, &self.query_bytes, set);
        } else {
            // Percent-encode after encoding: buffer the raw chars, then
            // encode the accumulated buffer now, straight into the query.
            let query = self.url.query.get_or_insert_with(String::new);
            percent_encode_into(query, &self.buffer, set);
        }
        self.buffer.clear();
    }

    fn fragment_state(&mut self, c: Option<char>) {
        match c {
            // Non-fatal invalid-URL-unit checks skipped. Buffer the raw
            // code point; the fragment is percent-encoded once, in full, at
            // EOF, mirroring the query state.
            Some(ch) => {
                self.buffer.push(ch);
            }
            None => {
                let fragment = self.url.fragment.get_or_insert_with(String::new);
                percent_encode_into(fragment, &self.buffer, EncodeSet::Fragment);
                self.buffer.clear();
                self.state = State::Done;
            }
        }
    }

    // -- path helpers ----------------------------------------------------

    fn push_segment(wpath: &mut Path, seg: String) {
        match wpath {
            Path::Segments(segs) => segs.push(seg),
            Path::Opaque(_) => {}
        }
    }

    /// Shorten a path: pop the last segment unless it is a lone normalized
    /// Windows drive letter in a `file:` URL.
    fn shorten(wpath: &mut Path, scheme: &str) {
        let Path::Segments(segs) = wpath else { return };
        if scheme == "file" && segs.len() == 1 && is_normalized_windows_drive_letter(&segs[0]) {
            return;
        }
        segs.pop();
    }
}

/// Parse a port buffer: all ASCII digits, value <= 65535.
fn parse_port(buffer: &str) -> Option<u16> {
    if buffer.is_empty() || !buffer.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut value: u32 = 0;
    for b in buffer.bytes() {
        value = value.checked_mul(10)?.checked_add((b - b'0') as u32)?;
        if value > 65535 {
            return None;
        }
    }
    u16::try_from(value).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn must_parse(input: &str) -> Url {
        parse(input, None).unwrap_or_else(|e| panic!("failed to parse {input:?}: {e:?}"))
    }

    #[test]
    fn basic_components() {
        let u = must_parse("https://user:pass@example.com:8080/a/b?x=1#y");
        assert_eq!(u.scheme(), "https");
        assert_eq!(u.username(), "user");
        assert_eq!(u.password(), "pass");
        assert_eq!(u.host(), Some("example.com"));
        assert_eq!(u.port(), Some(8080));
        assert_eq!(u.path(), "/a/b");
        assert_eq!(u.query(), Some("x=1"));
        assert_eq!(u.fragment(), Some("y"));
        assert_eq!(
            u.serialize(),
            "https://user:pass@example.com:8080/a/b?x=1#y"
        );
    }

    #[test]
    fn default_ports_omitted() {
        assert_eq!(must_parse("http://h:80/").port(), None);
        assert_eq!(must_parse("http://h:80/").serialize(), "http://h/");
        assert_eq!(must_parse("https://h:443/").port(), None);
        assert_eq!(must_parse("ftp://h:21/").port(), None);
        assert_eq!(must_parse("ws://h:80/").port(), None);
        assert_eq!(must_parse("wss://h:443/").port(), None);
        assert_eq!(must_parse("http://h:8080/").port(), Some(8080));
        // Non-special schemes have no default port.
        assert_eq!(must_parse("foo://h:80/").port(), Some(80));
    }

    #[test]
    fn invalid_ports() {
        assert!(parse("http://h:65536/", None).is_err());
        assert!(parse("http://h:9999999999999999999999/", None).is_err());
        assert!(parse("http://h:abc/", None).is_err());
        // Empty port is fine.
        assert_eq!(must_parse("http://h:/").serialize(), "http://h/");
    }

    #[test]
    fn preprocessing() {
        assert_eq!(
            must_parse("  \thttp://example.com/\n ").serialize(),
            "http://example.com/"
        );
        assert_eq!(
            must_parse("http://example\t.com/").host(),
            Some("example.com")
        );
    }

    #[test]
    fn dot_segments() {
        assert_eq!(
            must_parse("http://h/a/./b/../c").serialize(),
            "http://h/a/c"
        );
        assert_eq!(must_parse("http://h/a/b/../../c").serialize(), "http://h/c");
        assert_eq!(must_parse("http://h/a/..").serialize(), "http://h/");
        assert_eq!(must_parse("http://h/a/.").serialize(), "http://h/a/");
        // Encoded dots count too.
        assert_eq!(must_parse("http://h/a/%2e/b").serialize(), "http://h/a/b");
        assert_eq!(must_parse("http://h/a/%2E%2E/b").serialize(), "http://h/b");
    }

    #[test]
    fn opaque_paths() {
        let u = must_parse("mailto:foo@bar.example");
        assert!(u.cannot_be_a_base());
        assert_eq!(u.host(), None);
        assert_eq!(u.path(), "foo@bar.example");
        assert_eq!(u.serialize(), "mailto:foo@bar.example");
        // Spaces in opaque paths.
        assert_eq!(
            must_parse("non-special:opaque #hi").serialize(),
            "non-special:opaque%20#hi"
        );
        assert_eq!(
            must_parse("lolscheme:x x#x x").serialize(),
            "lolscheme:x x#x%20x"
        );
    }

    #[test]
    fn non_special_authority() {
        // Single slash after a non-special scheme is a path, not authority.
        let u = must_parse("foo:/bar");
        assert_eq!(u.host(), None);
        assert_eq!(u.path(), "/bar");
        assert_eq!(u.serialize(), "foo:/bar");
        // The "/." guard keeps the serialization stable.
        assert_eq!(
            must_parse("non-spec:/.//path").serialize(),
            "non-spec:/.//path"
        );
        assert_eq!(must_parse("non-spec:/.//path").path(), "//path");
    }

    #[test]
    fn relative_resolution() {
        let base = must_parse("http://example.org/foo/bar");
        let u = parse("baz", Some(&base)).unwrap();
        assert_eq!(u.serialize(), "http://example.org/foo/baz");
        assert_eq!(
            parse("/abs", Some(&base)).unwrap().serialize(),
            "http://example.org/abs"
        );
        assert_eq!(
            parse("?q", Some(&base)).unwrap().serialize(),
            "http://example.org/foo/bar?q"
        );
        assert_eq!(
            parse("#f", Some(&base)).unwrap().serialize(),
            "http://example.org/foo/bar#f"
        );
        assert_eq!(
            parse("//other.com/x", Some(&base)).unwrap().serialize(),
            "http://other.com/x"
        );
        // Fragment-only works against an opaque base.
        let obase = must_parse("foo:bar");
        assert_eq!(parse("#f", Some(&obase)).unwrap().serialize(), "foo:bar#f");
        assert!(parse("baz", Some(&obase)).is_err());
    }

    #[test]
    fn file_urls() {
        assert_eq!(must_parse("file:///C|/demo").serialize(), "file:///C:/demo");
        assert_eq!(
            must_parse("file://localhost/test").serialize(),
            "file:///test"
        );
        assert_eq!(
            must_parse("file:/example.com/").serialize(),
            "file:///example.com/"
        );
        assert_eq!(must_parse("file:").serialize(), "file:///");
        let base = must_parse("file:///tmp/mock/path");
        assert_eq!(
            parse("test", Some(&base)).unwrap().serialize(),
            "file:///tmp/mock/test"
        );
        let cbase = must_parse("file:///C:/a/b");
        assert_eq!(
            parse("..", Some(&cbase)).unwrap().serialize(),
            "file:///C:/"
        );
        assert_eq!(parse("/", Some(&cbase)).unwrap().serialize(), "file:///C:/");
    }

    #[test]
    fn ipv6_round_trip() {
        let u = must_parse("http://[1:2:0:0:5:0:0:0]/");
        assert_eq!(u.host(), Some("[1:2:0:0:5::]"));
        assert_eq!(u.serialize(), "http://[1:2:0:0:5::]/");
        let u = must_parse("non-special://[::1]:8080/");
        assert_eq!(u.host(), Some("[::1]"));
        assert_eq!(u.port(), Some(8080));
    }

    #[test]
    fn idna_hosts() {
        assert_eq!(must_parse("http://é").host(), Some("xn--9ca"));
        assert_eq!(must_parse("http://你好你好").host(), Some("xn--6qqa088eba"));
        assert_eq!(
            must_parse("https://faß.ExAmPlE/").serialize(),
            "https://xn--fa-hia.example/"
        );
        assert!(parse("http://GOO 　goo.com", None).is_err());
    }

    #[test]
    fn all_ascii_xn_labels_accepted_per_spec() {
        // WHATWG URL §3.3 domain parser step 5: when beStrict is false and
        // the domain is ASCII, the lowercased domain is returned regardless
        // of ToASCII outcome. `xn--a` decodes to the disallowed U+0080, but
        // it is still accepted (the spec's example `xn--8i7caa`; WPT
        // toascii.json expects `"xn--a" -> "xn--a"`).
        assert_eq!(must_parse("http://xn--a/").host(), Some("xn--a"));
        assert_eq!(must_parse("http://XN--A/").host(), Some("xn--a"));
        assert_eq!(must_parse("http://a.xn--a/").host(), Some("a.xn--a"));
        assert_eq!(must_parse("http://xn--8i7caa/").host(), Some("xn--8i7caa"));
        // The non-ASCII path still verifies A-labels: `xn--a.é` goes
        // through `domain_to_ascii` and is rejected.
        assert_eq!(
            parse("http://xn--a.é/", None).unwrap_err(),
            ParseError::InvalidHost
        );
        // Valid `xn--` labels and hyphenated labels still parse.
        assert_eq!(must_parse("http://xn--9ca/").host(), Some("xn--9ca"));
        assert_eq!(must_parse("http://-foo.com/").host(), Some("-foo.com"));
        // `xn--0` is accepted too: no A-label verification on the
        // all-ASCII path (matches the percent-encoded `xn--%30`).
        assert_eq!(must_parse("http://xn--0/").host(), Some("xn--0"));
        assert_eq!(must_parse("http://xn--%30/").host(), Some("xn--0"));
    }

    #[test]
    fn userinfo_encoding() {
        let u = must_parse("http://user:pass@host/");
        assert_eq!(u.username(), "user");
        assert_eq!(u.password(), "pass");
        let u = must_parse("foo://%41:%42@host/");
        assert_eq!(u.username(), "%41");
        assert_eq!(u.password(), "%42");
        // Empty credentials drop the '@'.
        assert_eq!(
            must_parse("non-special://:@test/x").serialize(),
            "non-special://test/x"
        );
        // Password without username keeps ':'.
        assert_eq!(
            must_parse("foo://:pass@host/").serialize(),
            "foo://:pass@host/"
        );
    }

    #[test]
    fn failures() {
        assert_eq!(parse("http://", None).unwrap_err(), ParseError::MissingHost);
        assert_eq!(
            parse("http://[::1", None).unwrap_err(),
            ParseError::InvalidHost
        );
        assert_eq!(
            parse("relative", None).unwrap_err(),
            ParseError::MissingScheme
        );
        assert_eq!(
            parse("http://256.256.256.256/", None).unwrap_err(),
            ParseError::InvalidHost
        );
    }

    #[test]
    fn opaque_path_long_space_run_is_fast() {
        // Regression test: the opaque-path "remaining starts with" check
        // must not allocate per code point. It used to collect the entire
        // remaining input into a String twice per space, making 32 KiB of
        // spaces take ~89s in debug builds (quadratic DoS).
        let spaces = " ".repeat(32_768);
        let input = format!("x:{spaces}y");
        let start = std::time::Instant::now();
        let u = must_parse(&input);
        let elapsed = start.elapsed();
        assert_eq!(u.path(), format!("{spaces}y"));
        assert_eq!(u.serialize(), input);
        assert!(
            elapsed.as_secs() < 5,
            "opaque-path parse took too long: {elapsed:?}"
        );
    }

    #[test]
    fn origin_cases() {
        assert_eq!(
            must_parse("http://example.com:8080/").origin(),
            Origin::Tuple("http".into(), "example.com".into(), 8080)
        );
        // Default port fills in when the URL omits it.
        assert_eq!(
            must_parse("https://example.com/").origin(),
            Origin::Tuple("https".into(), "example.com".into(), 443)
        );
        assert_eq!(
            must_parse("ftp://example.com/").origin(),
            Origin::Tuple("ftp".into(), "example.com".into(), 21)
        );
        assert_eq!(
            must_parse("ws://example.com/chat").origin(),
            Origin::Tuple("ws".into(), "example.com".into(), 80)
        );
        // Host serializes the same way it does in the URL string.
        assert_eq!(
            must_parse("http://\u{2460}\u{2461}/").origin(),
            Origin::Tuple("http".into(), "0.0.0.12".into(), 80)
        );
        // file:, non-special schemes, and opaque paths are opaque origins.
        assert_eq!(must_parse("file:///x").origin(), Origin::Opaque);
        assert_eq!(must_parse("foo://h/").origin(), Origin::Opaque);
        assert_eq!(must_parse("non-special:opaque").origin(), Origin::Opaque);
    }

    #[test]
    fn origin_serialize_and_cors_check() {
        // Default ports are omitted; non-default ports are kept; the
        // host serializes as in the URL (IPv6 keeps its brackets).
        assert_eq!(
            must_parse("https://example.com:443/").origin_serialized(),
            "https://example.com"
        );
        assert_eq!(
            must_parse("http://example.com:8080/").origin_serialized(),
            "http://example.com:8080"
        );
        assert_eq!(
            must_parse("http://[::1]:8080/").origin_serialized(),
            "http://[::1]:8080"
        );
        assert_eq!(must_parse("file:///x").origin_serialized(), "null");
        // Opaque origins are never same-origin, even with each other.
        assert!(!Origin::Opaque.is_same_origin(&Origin::Opaque));
        assert!(
            must_parse("https://a.com/")
                .origin()
                .is_same_origin(&must_parse("https://a.com:443/").origin())
        );
        assert!(
            !must_parse("https://a.com/")
                .origin()
                .is_same_origin(&must_parse("https://a.com:8443/").origin())
        );

        let doc = must_parse("https://a.com/").origin();
        let same = must_parse("https://a.com/x").origin();
        let other = must_parse("https://b.com/").origin();
        // Same-origin passes unconditionally, even with no ACAO.
        assert!(cors_check(&doc, &same, &[], false));
        assert!(cors_check(&doc, &same, &["https://evil.com"], true));
        // Cross-origin: missing or wrong ACAO fails.
        assert!(!cors_check(&doc, &other, &[], false));
        assert!(!cors_check(&doc, &other, &["https://evil.com"], false));
        // `*` passes only without credentials.
        assert!(cors_check(&doc, &other, &["*"], false));
        assert!(!cors_check(&doc, &other, &["*"], true));
        // The exact serialized document origin passes with or without
        // credentials; surrounding ASCII whitespace is tolerated.
        assert!(cors_check(&doc, &other, &["https://a.com"], true));
        assert!(cors_check(&doc, &other, &["  https://a.com\t"], false));
        // Two ACAO values always fail, even when the first is `*`
        // and the second is the document origin (fetch requires
        // exactly one value).
        assert!(!cors_check(&doc, &other, &["*", "https://a.com"], false));
        assert!(!cors_check(
            &doc,
            &other,
            &["https://a.com", "https://a.com"],
            true
        ));
        // Unicode whitespace is NOT trimmed: HTTP field-value
        // trimming is space/tab only.
        assert!(!cors_check(
            &doc,
            &other,
            &["\u{a0}https://a.com\u{a0}"],
            false
        ));
        // An opaque document origin never matches by name: `null` is
        // not accepted, only credentialless `*`.
        assert!(!cors_check(&Origin::Opaque, &other, &["null"], false));
        assert!(cors_check(&Origin::Opaque, &other, &["*"], false));
        assert!(!cors_check(&Origin::Opaque, &other, &["*"], true));
    }

    #[test]
    fn host_enum_cases() {
        assert_eq!(
            must_parse("http://example.com/").host_enum(),
            Some(&Host::Domain("example.com".into()))
        );
        assert_eq!(
            must_parse("http://1.2.3.4/").host_enum(),
            Some(&Host::Ipv4(0x01020304))
        );
        assert_eq!(
            must_parse("http://[::1]/").host_enum(),
            Some(&Host::Ipv6([0, 0, 0, 0, 0, 0, 0, 1]))
        );
        assert_eq!(
            must_parse("foo://h%41/").host_enum(),
            Some(&Host::Opaque("h%41".into()))
        );
        assert_eq!(must_parse("mailto:foo@bar").host_enum(), None);
    }

    #[test]
    fn serialize_without_fragment_cases() {
        let u = must_parse("https://example.com/a?x=1#frag");
        assert_eq!(u.serialize_without_fragment(), "https://example.com/a?x=1");
        assert_eq!(u.serialize(), "https://example.com/a?x=1#frag");
        // No fragment: identical to serialize().
        let u = must_parse("https://example.com/a?x=1");
        assert_eq!(u.serialize_without_fragment(), "https://example.com/a?x=1");
        // Empty fragment still drops the '#'.
        let u = must_parse("https://example.com/a#");
        assert_eq!(u.serialize_without_fragment(), "https://example.com/a");
        assert_eq!(u.serialize(), "https://example.com/a#");
    }
}

#[cfg(test)]
mod setter_tests {
    use super::*;

    fn must_parse(s: &str) -> Url {
        parse(s, None).expect("test URL parses")
    }

    #[test]
    fn set_href_replaces_and_rejects_relative() {
        let mut u = must_parse("http://example.com/a");
        assert!(u.set_href("https://other.org/b?c#d").is_ok());
        assert_eq!(u.serialize(), "https://other.org/b?c#d");
        assert!(u.set_href("/relative").is_err());
        assert_eq!(u.serialize(), "https://other.org/b?c#d");
    }

    #[test]
    fn set_protocol_swaps_scheme() {
        let mut u = must_parse("http://example.com/a");
        assert!(u.set_protocol("https:").is_ok());
        assert_eq!(u.serialize(), "https://example.com/a");
        assert_eq!(u.scheme(), "https");
        // Empty scheme: unchanged.
        assert!(u.set_protocol(":").is_ok());
        assert_eq!(u.scheme(), "https");
        // Invalid scheme: error, unchanged.
        assert!(u.set_protocol("9bad:").is_err());
        assert_eq!(u.scheme(), "https");
    }

    #[test]
    fn set_host_keeps_userinfo_and_takes_port() {
        let mut u = must_parse("http://user:pass@example.com:8080/a");
        u.set_host("other.org:9090");
        assert_eq!(u.serialize(), "http://user:pass@other.org:9090/a");
        assert_eq!(u.host(), Some("other.org"));
        assert_eq!(u.port(), Some(9090));
    }

    #[test]
    fn set_host_rejected_input_leaves_url_unchanged() {
        let mut u = must_parse("http://example.com/a");
        u.set_host("not a host!!");
        assert_eq!(u.serialize(), "http://example.com/a");
    }

    #[test]
    fn set_hostname_preserves_port() {
        let mut u = must_parse("http://example.com:8080/a");
        u.set_hostname("other.org");
        assert_eq!(u.serialize(), "http://other.org:8080/a");
        // Smuggled port: unchanged.
        u.set_hostname("evil.org:1");
        assert_eq!(u.serialize(), "http://other.org:8080/a");
    }

    #[test]
    fn set_host_rejects_credential_injection() {
        // U2: `user@h` is not parsed as credentials; `@` fails host
        // parsing and the URL is left unchanged.
        let mut u = must_parse("http://x/a");
        u.set_host("user@h");
        assert_eq!(u.serialize(), "http://x/a");
        u.set_hostname("user@h");
        assert_eq!(u.serialize(), "http://x/a");
        let mut u = must_parse("foo:/x");
        u.set_host("user@h");
        assert_eq!(u.serialize(), "foo:/x");
    }

    #[test]
    fn set_host_empty_input() {
        // U3: an empty host is rejected for special schemes, and when the
        // URL carries credentials or a port.
        let mut u = must_parse("http://h/old");
        u.set_host("");
        assert_eq!(u.serialize(), "http://h/old");
        let mut u = must_parse("http://h:1/old");
        u.set_host("");
        assert_eq!(u.serialize(), "http://h:1/old");
        let mut u = must_parse("http://h/old");
        u.set_hostname("");
        assert_eq!(u.serialize(), "http://h/old");
        let mut u = must_parse("http://u@h/old");
        u.set_host("");
        assert_eq!(u.serialize(), "http://u@h/old");
        // A non-special scheme takes the empty opaque host.
        let mut u = must_parse("foo:/x");
        u.set_host("");
        assert_eq!(u.serialize(), "foo:///x");
    }

    #[test]
    fn set_host_terminators_are_ignored() {
        // U4: `/`, `?`, `#` (and `\` for special schemes) end the host;
        // the rest is ignored and the old path/query/fragment survive.
        let mut u = must_parse("http://h/old");
        u.set_host("h2/p");
        assert_eq!(u.serialize(), "http://h2/old");
        let mut u = must_parse("http://h/old?q=1#f");
        u.set_host("h2?x");
        assert_eq!(u.serialize(), "http://h2/old?q=1#f");
        let mut u = must_parse("http://h/old?q=1#f");
        u.set_host("h2#x");
        assert_eq!(u.serialize(), "http://h2/old?q=1#f");
        let mut u = must_parse("http://h/old?q=1#f");
        u.set_hostname("a/b");
        assert_eq!(u.serialize(), "http://a/old?q=1#f");
        let mut u = must_parse("http://h/old");
        u.set_host("h2\\x");
        assert_eq!(u.serialize(), "http://h2/old");
        let mut u = must_parse("http://h/old");
        u.set_host("h2/");
        assert_eq!(u.serialize(), "http://h2/old");
    }

    #[test]
    fn set_host_on_hostless_non_special_url() {
        // U5: only opaque-path URLs return early; `foo:/x` gains a host.
        let mut u = must_parse("foo:/x");
        u.set_host("h");
        assert_eq!(u.serialize(), "foo://h/x");
        let mut u = must_parse("foo:/x");
        u.set_hostname("h");
        assert_eq!(u.serialize(), "foo://h/x");
        let mut u = must_parse("foo:/x");
        u.set_host("h:8080");
        assert_eq!(u.serialize(), "foo://h:8080/x");
        // Opaque paths still return early.
        let mut u = must_parse("mailto:foo@bar");
        u.set_host("h");
        u.set_hostname("h");
        assert_eq!(u.serialize(), "mailto:foo@bar");
    }

    #[test]
    fn set_host_port_semantics() {
        // The host is committed before the port state runs, so an invalid
        // port keeps the new host and the old port (spec + Node v24).
        let mut u = must_parse("http://h/old");
        u.set_host("h2:bad");
        assert_eq!(u.serialize(), "http://h2/old");
        // A valid digit prefix is applied before the invalid char ends it.
        let mut u = must_parse("http://h/old");
        u.set_host("h2:8b");
        assert_eq!(u.serialize(), "http://h2:8/old");
        // Out-of-range port: host kept, port untouched.
        let mut u = must_parse("http://h:1/old");
        u.set_host("h2:65536");
        assert_eq!(u.serialize(), "http://h2:1/old");
        // An empty port, or no colon at all, keeps the old port.
        let mut u = must_parse("http://x:1/a");
        u.set_host("h:");
        assert_eq!(u.serialize(), "http://h:1/a");
        let mut u = must_parse("http://x:8080/a");
        u.set_host("h2");
        assert_eq!(u.serialize(), "http://h2:8080/a");
        // A port with an empty host is rejected outright.
        let mut u = must_parse("http://x:7/old");
        u.set_host(":80");
        assert_eq!(u.serialize(), "http://x:7/old");
        // Default ports normalize away.
        let mut u = must_parse("http://x/a");
        u.set_host("h:80");
        assert_eq!(u.serialize(), "http://h/a");
        // Smuggled ports in the hostname setter: unchanged.
        let mut u = must_parse("http://h/old");
        u.set_hostname("h2:8080");
        assert_eq!(u.serialize(), "http://h/old");
        let mut u = must_parse("http://x/");
        u.set_hostname("[::1]:8080");
        assert_eq!(u.serialize(), "http://x/");
    }

    #[test]
    fn set_host_whitespace_and_case() {
        // No leading/trailing space trim for state overrides (spec), but
        // ASCII tab/newline are removed; hosts lowercase.
        let mut u = must_parse("http://x/a");
        u.set_host("  h2  ");
        assert_eq!(u.serialize(), "http://x/a");
        let mut u = must_parse("http://x/a");
        u.set_host("h2\t");
        assert_eq!(u.serialize(), "http://h2/a");
        let mut u = must_parse("http://x/a");
        u.set_host("H2");
        assert_eq!(u.serialize(), "http://h2/a");
    }

    #[test]
    fn set_host_file_scheme() {
        // `file:` URLs use the file host state.
        let mut u = must_parse("file:///a");
        u.set_host("h");
        assert_eq!(u.serialize(), "file://h/a");
        let mut u = must_parse("file:///a");
        u.set_hostname("h");
        assert_eq!(u.serialize(), "file://h/a");
        let mut u = must_parse("file:///a");
        u.set_hostname("h/p");
        assert_eq!(u.serialize(), "file://h/a");
        let mut u = must_parse("file:///x");
        u.set_host("localhost");
        assert_eq!(u.serialize(), "file:///x");
    }

    #[test]
    fn set_host_ipv6() {
        let mut u = must_parse("http://x/");
        u.set_host("[::1]:8080");
        assert_eq!(u.serialize(), "http://[::1]:8080/");
        let mut u = must_parse("http://x/");
        u.set_hostname("[::1]");
        assert_eq!(u.serialize(), "http://[::1]/");
        let mut u = must_parse("http://x/a");
        u.set_host("[::1");
        assert_eq!(u.serialize(), "http://x/a");
    }

    #[test]
    fn set_port_sets_clears_and_rejects() {
        let mut u = must_parse("http://example.com/a");
        u.set_port("8080");
        assert_eq!(u.serialize(), "http://example.com:8080/a");
        u.set_port("");
        assert_eq!(u.serialize(), "http://example.com/a");
        u.set_port("abc");
        assert_eq!(u.port(), None);
        // A leading `+` is not a digit: the port state terminates with an
        // empty buffer, leaving the port unchanged (Rust's `u16` FromStr
        // would accept it).
        let mut u = must_parse("http://example.com:8080/a");
        u.set_port("+80");
        assert_eq!(u.serialize(), "http://example.com:8080/a");
        u.set_port("-80");
        assert_eq!(u.serialize(), "http://example.com:8080/a");
        u.set_port(" 80");
        assert_eq!(u.serialize(), "http://example.com:8080/a");
        // Default port normalizes away.
        u.set_port("80");
        assert_eq!(u.serialize(), "http://example.com/a");
    }

    #[test]
    fn set_pathname_encodes_and_resets() {
        let mut u = must_parse("http://example.com/old?q=1");
        u.set_pathname("/a b/c");
        assert_eq!(u.path(), "/a%20b/c");
        assert_eq!(u.query(), Some("q=1"));
        u.set_pathname("");
        assert_eq!(u.path(), "/");
    }

    #[test]
    fn set_search_and_hash() {
        let mut u = must_parse("http://example.com/a?old=1#oldfrag");
        u.set_search("?a=1&b=2");
        assert_eq!(u.query(), Some("a=1&b=2"));
        u.set_search("");
        assert_eq!(u.query(), None);
        u.set_hash("#frag ment");
        assert_eq!(u.fragment(), Some("frag%20ment"));
        u.set_hash("");
        assert_eq!(u.fragment(), None);
    }

    // U6: `?`/`#` in the pathname input are percent-encoded, not treated
    // as query/fragment delimiters (spec path start state with state
    // override).
    #[test]
    fn set_pathname_encodes_delimiters() {
        let mut u = must_parse("http://x/a?old=1#oldfrag");
        u.set_pathname("/p?qq");
        assert_eq!(u.serialize(), "http://x/p%3Fqq?old=1#oldfrag");
        u.set_pathname("/p#f2");
        assert_eq!(u.serialize(), "http://x/p%23f2?old=1#oldfrag");
        u.set_pathname("?");
        assert_eq!(u.serialize(), "http://x/%3F?old=1#oldfrag");
        u.set_pathname("#");
        assert_eq!(u.serialize(), "http://x/%23?old=1#oldfrag");
    }

    // U7: `\` splits path segments only for special schemes; for
    // non-special schemes it is a literal path character.
    #[test]
    fn set_pathname_backslash_scheme_aware() {
        let mut u = must_parse("foo:/old");
        u.set_pathname("a\\b");
        assert_eq!(u.serialize(), "foo:/a\\b");
        let mut u = must_parse("http://x/old");
        u.set_pathname("a\\b");
        assert_eq!(u.serialize(), "http://x/a/b");
    }

    // U8: `#` in the search input is percent-encoded, not treated as a
    // fragment delimiter (spec query state with state override).
    #[test]
    fn set_search_encodes_hash() {
        let mut u = must_parse("http://x/a?old=1#oldfrag");
        u.set_search("q#newfrag");
        assert_eq!(u.serialize(), "http://x/a?q%23newfrag#oldfrag");
        u.set_search("q?");
        assert_eq!(u.serialize(), "http://x/a?q?#oldfrag");
    }

    // U9: a lone `?`/`#` yields an empty (but present) query/fragment;
    // only the truly empty string removes the component.
    /// A legacy query encoder for tests: Big5-ish two-byte encoding for
    /// a handful of code points (mirrors `bowser_encoding::encode`
    /// without the cross-crate dependency). Unmapped code points emit
    /// the percent-encoded NCR `%26%23<dec>%3B`, per the URL Standard's
    /// "percent-encode after encoding".
    fn big5ish(s: &str, out: &mut Vec<u8>) {
        for ch in s.chars() {
            match ch {
                // U+3000 -> A1 40, U+20AC -> A3 E1, unmapped -> NCR.
                '\u{3000}' => out.extend_from_slice(&[0xA1, 0x40]),
                '\u{20AC}' => out.extend_from_slice(&[0xA3, 0xE1]),
                c if (c as u32) < 0x80 => out.push(c as u8),
                c => out.extend_from_slice(format!("%26%23{}%3B", c as u32).as_bytes()),
            }
        }
    }

    #[test]
    fn query_encoding_override_applies_to_parse() {
        // U+3000 encodes as A1 40 under the override (0x40 '@' is
        // query-allowed, so it survives unencoded); U+2603 is unmapped
        // and falls back to a numeric character reference.
        let u = parse_with_query_encoding("https://example.com/?\u{3000}\u{2603}", None, &big5ish)
            .unwrap();
        assert_eq!(u.query(), Some("%A1@%26%239731%3B"));
        // Without the override the same input is UTF-8 percent-encoded.
        let u = must_parse("https://example.com/?\u{3000}\u{2603}");
        assert_eq!(u.query(), Some("%E3%80%80%E2%98%83"));
    }

    #[test]
    fn query_encoding_override_applies_to_search_setter() {
        let mut u = must_parse("https://example.com/");
        u.set_search_with_encoding("?\u{3000}x", &big5ish);
        // 0x40 ('@') is query-allowed, so it survives unencoded.
        assert_eq!(u.query(), Some("%A1@x"));
        // The plain setter stays UTF-8.
        let mut u = must_parse("https://example.com/");
        u.set_search("?\u{3000}x");
        assert_eq!(u.query(), Some("%E3%80%80x"));
    }

    #[test]
    fn query_encoding_override_does_not_touch_fragment() {
        // Fragments always percent-encode as UTF-8, per spec.
        let u = parse_with_query_encoding("https://example.com/#\u{3000}", None, &big5ish).unwrap();
        assert_eq!(u.fragment(), Some("%E3%80%80"));
    }

    #[test]
    fn set_search_set_hash_lone_delimiter_keeps_component() {
        let mut u = must_parse("http://x/a?old=1#oldfrag");
        u.set_search("?");
        assert_eq!(u.query(), Some(""));
        assert_eq!(u.serialize(), "http://x/a?#oldfrag");
        u.set_search("");
        assert_eq!(u.query(), None);
        assert_eq!(u.serialize(), "http://x/a#oldfrag");

        let mut u = must_parse("http://x/a?old=1#oldfrag");
        u.set_hash("#");
        assert_eq!(u.fragment(), Some(""));
        assert_eq!(u.serialize(), "http://x/a?old=1#");
        u.set_hash("");
        assert_eq!(u.fragment(), None);
        assert_eq!(u.serialize(), "http://x/a?old=1");
    }

    // The override parse still resolves dot segments and keeps the
    // cannot-be-a-base early return.
    #[test]
    fn set_pathname_dot_segments_and_opaque() {
        let mut u = must_parse("http://x/old");
        u.set_pathname("/a/../b");
        assert_eq!(u.path(), "/b");
        u.set_pathname("..");
        assert_eq!(u.path(), "/");
        u.set_pathname("rel");
        assert_eq!(u.path(), "/rel");
        let mut u = must_parse("mailto:foo@bar");
        u.set_pathname("/x");
        assert_eq!(u.serialize(), "mailto:foo@bar");
    }

    // WPT setters_tests.json: empty input cannot erase the path of
    // special URLs or host-less non-special URLs, but can erase it for
    // non-special URLs with a host. Percent-encoded dot segments
    // (`%2e`, case-insensitive) resolve like literal ones.
    #[test]
    fn set_pathname_empty_and_encoded_dots() {
        let mut u = must_parse("file:///some/path");
        u.set_pathname("");
        assert_eq!(u.serialize(), "file:///");

        let mut u = must_parse("foo://somehost/some/path");
        u.set_pathname("");
        assert_eq!(u.serialize(), "foo://somehost");
        assert_eq!(u.path(), "");

        let mut u = must_parse("foo:/some/path");
        u.set_pathname("");
        assert_eq!(u.serialize(), "foo:/");

        let mut u = must_parse("http://x/old");
        u.set_pathname("/%2e%2E/y");
        assert_eq!(u.path(), "/y");
        u.set_pathname("a/./b");
        assert_eq!(u.path(), "/a/b");
        u.set_pathname("a/../b");
        assert_eq!(u.path(), "/b");
        u.set_pathname("//a");
        assert_eq!(u.path(), "//a");
    }

    #[test]
    fn setters_on_cannot_be_a_base() {
        let mut u = must_parse("mailto:foo@bar");
        u.set_host("example.com");
        u.set_pathname("/x");
        assert_eq!(u.serialize(), "mailto:foo@bar");
        u.set_search("subject=hi");
        assert_eq!(u.query(), Some("subject=hi"));
        u.set_hash("frag");
        assert_eq!(u.fragment(), Some("frag"));
    }
}

#[cfg(test)]
mod search_params_tests {
    use super::*;

    fn must_parse(s: &str) -> Url {
        parse(s, None).expect("test URL parses")
    }

    fn pairs(p: &UrlSearchParams) -> Vec<(&str, &str)> {
        p.iter().collect()
    }

    #[test]
    fn href_getter_matches_serialize() {
        let u = must_parse("https://user:pass@example.com:8080/a?x=1#f");
        assert_eq!(u.href(), u.serialize());
        assert_eq!(u.href(), "https://user:pass@example.com:8080/a?x=1#f");
        let mut u = must_parse("http://example.com/");
        u.set_href("https://other.org/").unwrap();
        assert_eq!(u.href(), "https://other.org/");
    }

    #[test]
    fn parse_vectors() {
        // WPT urlsearchparams-constructor vectors (string init).
        let cases: &[(&str, &[(&str, &str)])] = &[
            ("a=b&c=d", &[("a", "b"), ("c", "d")]),
            ("?a=b", &[("a", "b")]), // one leading '?' stripped
            ("a=", &[("a", "")]),
            ("=b", &[("", "b")]),
            ("&", &[]),
            ("&&", &[]),
            ("", &[]),
            ("a", &[("a", "")]),
            ("a=b=c", &[("a", "b=c")]), // split on first '='
            ("a=b+c", &[("a", "b c")]), // '+' decodes to space
            ("a=%2B", &[("a", "+")]),   // '%2B' stays '+'
            ("a=%20", &[("a", " ")]),
            ("a=%C3%A9", &[("a", "é")]),
            ("a=%ff", &[("a", "\u{fffd}")]), // invalid UTF-8 -> U+FFFD
            ("a=%EF%BB%BF", &[("a", "\u{feff}")]), // BOM kept, not stripped
            ("a=1&&b=2", &[("a", "1"), ("b", "2")]), // empty tokens skipped
            ("%41=%42", &[("A", "B")]),
            ("a=%2", &[("a", "%2")]), // lone '%' kept literally
        ];
        for (input, expected) in cases {
            assert_eq!(
                pairs(&UrlSearchParams::parse(input)),
                *expected,
                "parse({input:?})"
            );
        }
    }

    #[test]
    fn serialize_vectors() {
        // WPT urlsearchparams-stringifier vectors.
        let cases: &[(&[(&str, &str)], &str)] = &[
            (&[], ""),
            (&[("a", "b")], "a=b"),
            (&[("a", "b c")], "a=b+c"),
            (&[("a", "~")], "a=%7E"),
            (&[("a", "!")], "a=%21"),
            (&[("'", "(")], "%27=%28"),
            (&[(")", "~")], "%29=%7E"),
            (&[("a", "b&c=d")], "a=b%26c%3Dd"),
            (&[("a", "é")], "a=%C3%A9"),
            (&[("a", "+")], "a=%2B"),
            (&[("a", "%")], "a=%25"),
            (&[("a", "*-._")], "a=*-._"), // unreserved ASCII passes through
            (&[("a", "b"), ("c", "d")], "a=b&c=d"),
            (&[("", "")], "="),
            (&[("a", "\u{7f}")], "a=%7F"), // DEL is encoded
        ];
        for (input, expected) in cases {
            let mut p = UrlSearchParams::new();
            for (n, v) in *input {
                p.append(n, v);
            }
            assert_eq!(p.serialize(), *expected, "serialize({input:?})");
            assert_eq!(p.to_string(), *expected);
        }
    }

    #[test]
    fn accessors() {
        let mut p = UrlSearchParams::parse("a=1&a=2&b=3");
        assert_eq!(p.get("a"), Some("1"));
        assert_eq!(p.get_all("a"), vec!["1", "2"]);
        assert_eq!(p.get("missing"), None);
        assert!(p.get_all("missing").is_empty());
        assert!(p.has("b"));
        assert!(!p.has("c"));
        assert_eq!(p.len(), 3);
        assert!(!p.is_empty());

        // set() replaces the first and drops later duplicates.
        p.set("a", "9");
        assert_eq!(p.serialize(), "a=9&b=3");
        // set() appends when absent.
        p.set("c", "4");
        assert_eq!(p.serialize(), "a=9&b=3&c=4");
        // delete() removes all matches; unknown names are a no-op.
        p.delete("a");
        assert_eq!(p.serialize(), "b=3&c=4");
        p.delete("nope");
        assert_eq!(p.serialize(), "b=3&c=4");
        // append() keeps duplicates in order.
        p.append("b", "5");
        assert_eq!(p.get_all("b"), vec!["3", "5"]);

        let empty = UrlSearchParams::new();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.serialize(), "");
    }

    #[test]
    fn sort_vectors() {
        let mut p = UrlSearchParams::parse("c=1&a=2&b=3");
        p.sort();
        assert_eq!(p.serialize(), "a=2&b=3&c=1");
        // Stable for equal names.
        let mut p = UrlSearchParams::parse("b=2&a=1&b=1");
        p.sort();
        assert_eq!(p.serialize(), "a=1&b=2&b=1");
        // Empty list: no-op.
        let mut p = UrlSearchParams::new();
        p.sort();
        assert_eq!(p.serialize(), "");
        // Spec sorts by UTF-16 code units, not UTF-8 bytes: U+10000
        // ([D800 DC00]) sorts before U+FFFF, the reverse of byte order.
        let mut p = UrlSearchParams::new();
        p.append("\u{ffff}", "1");
        p.append("\u{10000}", "2");
        p.sort();
        assert_eq!(
            p.iter().map(|(n, _)| n).collect::<Vec<_>>(),
            ["\u{10000}", "\u{ffff}"]
        );
    }

    #[test]
    fn url_round_trip() {
        // Read a URL's query as params.
        let u = must_parse("https://example.com/?b=2&a=1&a=3");
        let p = u.search_params();
        assert_eq!(p.get("a"), Some("1"));
        assert_eq!(p.get_all("a"), vec!["1", "3"]);

        // A missing query is the empty parameter list.
        let mut u = must_parse("https://example.com/path");
        assert!(u.search_params().is_empty());

        // Write params back; the query round-trips through the pair list.
        let mut p = UrlSearchParams::new();
        p.append("q", "hello world");
        p.append("lang", "fr");
        u.set_search_params(&p);
        assert_eq!(u.query(), Some("q=hello+world&lang=fr"));
        assert_eq!(
            u.serialize(),
            "https://example.com/path?q=hello+world&lang=fr"
        );
        let back = u.search_params();
        assert_eq!(back.get("q"), Some("hello world"));
        assert_eq!(back.get("lang"), Some("fr"));

        // Empty params set the query to the empty string, per the spec's
        // URLSearchParams write-back (the `?` survives in the
        // serialization); `set_search("")` is how the query is removed.
        u.set_search_params(&UrlSearchParams::new());
        assert_eq!(u.query(), Some(""));
        assert_eq!(u.serialize(), "https://example.com/path?");

        // Pair-level round trip through serialize/parse, incl. metacharacters.
        let mut p = UrlSearchParams::new();
        p.append("a", "b'c\"d&e=f+g");
        p.append("ünïcode", "✓");
        let reparsed = UrlSearchParams::parse(&p.serialize());
        assert_eq!(reparsed, p);
    }

    #[test]
    fn search_params_mut_is_live_linked_view() {
        // Mutations through the live view write back into the URL's query.
        let mut u = must_parse("https://example.com/?b=2&a=1");
        {
            let mut p = u.search_params_mut();
            assert_eq!(p.get("a"), Some("1"));
            assert_eq!(p.get_all("a"), vec!["1"]);
            assert!(p.has("b"));
            p.append("c", "3");
            assert_eq!(p.serialize(), "b=2&a=1&c=3");
            assert_eq!(p.to_string(), "b=2&a=1&c=3");
        }
        assert_eq!(u.query(), Some("b=2&a=1&c=3"));
        assert_eq!(u.href(), "https://example.com/?b=2&a=1&c=3");

        // set() replaces in place; delete() drops every match. Emptying
        // the list writes Some(""), never None: the `?` stays in href().
        {
            let mut p = u.search_params_mut();
            p.set("a", "9");
            assert_eq!(p.serialize(), "b=2&a=9&c=3");
            p.delete("b");
            p.delete("c");
            p.delete("a");
            assert!(p.is_empty());
        }
        assert_eq!(u.query(), Some(""));
        assert_eq!(u.href(), "https://example.com/?");

        // Borrowing the view without mutating leaves the URL untouched.
        let mut u = must_parse("https://example.com/path");
        assert_eq!(u.query(), None);
        {
            let p = u.search_params_mut();
            assert!(p.is_empty());
            assert_eq!(p.len(), 0);
        }
        assert_eq!(u.query(), None);
        assert_eq!(u.href(), "https://example.com/path");

        // Appending to a query-less URL creates the query; sort() writes
        // back too, and the snapshot API still sees the same pairs.
        {
            let mut p = u.search_params_mut();
            p.append("q", "hello world");
            p.append("a", "1");
            p.sort();
        }
        assert_eq!(u.query(), Some("a=1&q=hello+world"));
        assert_eq!(u.href(), "https://example.com/path?a=1&q=hello+world");
        let snap = u.search_params();
        assert_eq!(snap.serialize(), "a=1&q=hello+world");
    }
}

#[cfg(test)]
mod domain_to_unicode_tests {
    use super::*;

    #[test]
    fn to_unicode_vectors() {
        // Punycode labels decode; other labels come back mapped.
        assert_eq!(
            domain_to_unicode("xn--nxasmq6b.example"),
            Some("βόλοσ.example".into())
        );
        assert_eq!(domain_to_unicode("Bücher.de"), Some("bücher.de".into()));
        assert_eq!(domain_to_unicode("faß.de"), Some("faß.de".into()));
        assert_eq!(domain_to_unicode("EXAMPLE.COM"), Some("example.com".into()));
        assert_eq!(domain_to_unicode("Ｇｏ.com"), Some("go.com".into()));
        assert_eq!(
            domain_to_unicode("www.foo。bar.com"),
            Some("www.foo.bar.com".into())
        );
        assert_eq!(domain_to_unicode("xn--6qqa088eba"), Some("你好你好".into()));
        // `xn--a` Punycode-decodes to disallowed U+0080, so UTS #46
        // ToUnicode errors; per the spec ("if an error was recorded,
        // return domain") the label is kept verbatim so the ASCII form
        // round-trips.
        assert_eq!(domain_to_unicode("xn--a"), Some("xn--a".into()));
        assert_eq!(domain_to_unicode("XN--A"), Some("xn--a".into()));
        assert_eq!(domain_to_unicode("xn--8i7caa"), Some("xn--8i7caa".into()));
        assert_eq!(domain_to_unicode("xn--9ca"), Some("é".into()));
        // `xn--kba` Punycode-decodes to U+00AD SOFT HYPHEN, which has IDNA
        // status "ignored": UTS #46 §4.1 requires a ToUnicode error there,
        // so per the spec's "if an error was recorded, return domain" step
        // the label is kept verbatim in ASCII form.
        assert_eq!(domain_to_unicode("xn--kba"), Some("xn--kba".into()));
        assert_eq!(domain_to_unicode("a..b"), Some("a..b".into()));
        assert_eq!(
            domain_to_unicode("example.com."),
            Some("example.com.".into())
        );
        assert_eq!(domain_to_unicode(""), None);
        // Rejected by domain_to_ascii: None rather than best-effort.
        assert_eq!(domain_to_unicode("xn--tešla.ß"), None);
        assert_eq!(domain_to_unicode("\u{fdd0}zyx.com"), None);
    }

    #[test]
    fn to_unicode_enforces_length_caps() {
        // 253 octets is the longest domain decoded; 254 is rejected
        // before any Punycode work (IDNA2008 / UTS #46 domain cap).
        assert_eq!(
            domain_to_unicode(&"a".repeat(253)).as_deref(),
            Some("a".repeat(253).as_str())
        );
        assert_eq!(domain_to_unicode(&"a".repeat(254)), None);
        assert_eq!(domain_to_unicode(&"ab.".repeat(100)), None);
        // An over-long `xn--` label fails verification, but per the spec
        // error fallback it is still returned verbatim in ASCII form
        // (the domain itself is under the 253-octet cap).
        assert_eq!(
            domain_to_unicode(&format!("xn--{}", "b".repeat(60))),
            Some(format!("xn--{}", "b".repeat(60)))
        );
        // A 63-octet `xn--` label still attempts verification (it fails
        // here on content — `"a"*59` decodes to disallowed U+0080 runs —
        // not on length), and falls back to the verbatim ASCII label.
        assert_eq!(
            domain_to_unicode(&format!("xn--{}", "a".repeat(59))),
            Some(format!("xn--{}", "a".repeat(59)))
        );
    }

    #[test]
    fn to_unicode_round_trip() {
        // domain_to_unicode inverts domain_to_ascii on the mapped form.
        for d in ["Bücher.de", "faß.de", "你好你好", "☕.example", "a.b"] {
            let ascii = domain_to_ascii(d).expect("to_ascii works");
            let back = domain_to_unicode(&ascii).expect("to_unicode works");
            assert_eq!(
                domain_to_ascii(&back).as_deref(),
                Some(ascii.as_str()),
                "round trip {d}"
            );
        }
    }

    #[test]
    fn to_unicode_of_url_host() {
        let u = parse("https://faß.ExAmPlE/", None).unwrap();
        assert_eq!(u.host(), Some("xn--fa-hia.example"));
        assert_eq!(
            domain_to_unicode(u.host().unwrap()),
            Some("faß.example".into())
        );
    }
}

#[cfg(test)]
mod adversarial_tests {
    //! Hostile-input sweep: URL parsing is fully attacker-controlled in a
    //! browser, so every entry point must degrade to `Err`/`None` instead
    //! of panicking. These tests assert exactly that: results are
    //! discarded, the only assertion is "did not unwind".
    //!
    //! Notes on coverage:
    //! - `&str` is always valid UTF-8, so truly invalid byte sequences
    //!   cannot reach `parse` directly; they are exercised through
    //!   `utf8_decode_lossy` (which takes `&[u8]`) and through multibyte
    //!   characters placed at every template position.
    //! - Lone surrogates cannot exist in a Rust `&str` (neither as WTF-8
    //!   nor otherwise); their CESU-8 byte encodings are fed to
    //!   `utf8_decode_lossy` below, which must replace them with U+FFFD
    //!   rather than panic.
    use super::*;
    use crate::percent::utf8_decode_lossy;

    /// Every byte value 0x00-0xFF as a `char` (all are valid scalar values;
    /// the surrogate range does not intersect 0x00-0xFF).
    fn byte_soup() -> String {
        (0u32..256).filter_map(char::from_u32).collect()
    }

    fn eat<T>(_: T) {}

    #[test]
    fn parse_never_panics() {
        let soup = byte_soup();
        let long = "a".repeat(100_000);
        let long_pct = "%".repeat(50_000);
        let long_digits = "9".repeat(100_000);
        let long_unicode = "é".repeat(20_000);

        // Every byte value embedded in URL-significant positions, plus
        // long and structurally hostile strings.
        let inputs = [
            soup.clone(),
            long.clone(),
            long_pct.clone(),
            long_digits.clone(),
            long_unicode.clone(),
            ":".to_string(),
            "".to_string(),
            "/".to_string(),
            "http:".to_string(),
            "http://".to_string(),
            "http://[".to_string(),
            "http://]".to_string(),
            format!("http://[{long}]"),
            format!("http://{long}/"),
            format!("http://user:pass@{long}:{long}/{long}?{long}#{long}"),
            format!("{long}:"),
            format!("{}://x", "a".repeat(10_000)),
            "1http://x/".to_string(),
            "-://x/".to_string(),
            "hₜₜₚ://x/".to_string(),
            format!("xn--{long}"),
            format!("xn--{long_digits}"),
            format!("http://xn--{long_digits}/"),
            "http://%ff%fe/".to_string(),
            "http://%c0%af/".to_string(),
            "file:///".to_string(),
            "file://".to_string(),
            "file:/".to_string(),
            "file:\\\\".to_string(),
            ":::::::".to_string(),
            "@@@@@@@@".to_string(),
            "????????".to_string(),
            "########".to_string(),
            "\u{feff}http://example.com/".to_string(),
            "http://example.com/\u{feff}".to_string(),
        ];
        let base_inputs = ["https://example.org/dir/page", "file:///C:/", "foo:bar"];
        let bases: Vec<Url> = base_inputs
            .iter()
            .filter_map(|b| parse(b, None).ok())
            .collect();

        for input in &inputs {
            eat(parse(input, None));
            for base in &bases {
                eat(parse(input, Some(base)));
            }
            // Byte soup spliced into every structural slot.
            for template in [
                "http://{s}/",
                "http://u@{s}/",
                "http://[{s}]/",
                "{s}://x/",
                "http://x/{s}",
                "http://x/?{s}",
                "http://x/#{s}",
                "file://{s}/",
                "foo:{s}",
            ] {
                eat(parse(&template.replace("{s}", &soup), None));
            }
        }

        // Every single byte value as the whole input, with and without base.
        for c in soup.chars() {
            let s = c.to_string();
            eat(parse(&s, None));
            for base in &bases {
                eat(parse(&s, Some(base)));
            }
        }
    }

    #[test]
    fn setters_never_panic() {
        let soup = byte_soup();
        let long = "x".repeat(50_000);
        let mut url = parse("https://example.com/", None).expect("test base parses");
        for s in [&soup, &long, ":", "", "http://[", "%ff", "\u{feff}"] {
            eat(url.set_href(s));
            eat(url.set_protocol(s));
            url.set_host(s);
            url.set_hostname(s);
            url.set_port(s);
            url.set_pathname(s);
            url.set_search(s);
            url.set_hash(s);
            eat(UrlSearchParams::parse(s));
            let mut params = UrlSearchParams::new();
            params.append(s, s);
            params.set(s, s);
            eat(params.get(s));
            params.delete(s);
        }
        eat(url.serialize());
    }

    #[test]
    fn host_idna_percent_never_panic() {
        let soup = byte_soup();
        let long = "a".repeat(20_000);
        let long_digits = "9".repeat(100_000);

        for input in [
            soup.as_str(),
            long.as_str(),
            "[",
            "]",
            "[]",
            "[::1",
            "[::1]]",
            "1.2.3.4",
            "99999999999999999999999999",
            "0x",
            "0x123456789abcdef123456789",
            "1.2.3.4.5.6",
            "....",
            "xn--",
            "xn--a",
            "%",
            "%zz",
            "\u{feff}",
        ] {
            eat(host::parse_host(input, true));
            eat(host::parse_host(input, false));
            eat(host::ipv4_parse(input));
            eat(host::ipv6_parse(input));
            eat(host::opaque_host_parse(input));
            eat(idna::domain_to_ascii(input));
            eat(domain_to_unicode(input));
            eat(idna::punycode_encode(input));
            eat(idna::punycode_decode(input));
            eat(percent_decode(input));
            eat(percent_encode(input, EncodeSet::C0));
            eat(percent_encode(input, EncodeSet::Userinfo));
        }
        // Punycode overflow drills: extreme deltas must return `None`,
        // never panic on arithmetic overflow (debug builds check).
        for evil in [
            format!("a-{long_digits}"),
            format!("xn--{long_digits}"),
            long_digits.clone(),
            format!("{}-{}", "z".repeat(1000), long_digits),
        ] {
            eat(idna::punycode_decode(&evil));
            eat(idna::punycode_encode(&evil));
            eat(idna::domain_to_ascii(&evil));
        }
        // Invalid UTF-8 byte sequences and CESU-8 surrogate encodings
        // (WTF-8 lone surrogates have no `&str` representation).
        let mut bad_bytes: Vec<Vec<u8>> = vec![
            vec![0xff, 0xfe],
            vec![0x80],
            vec![0xc3],
            vec![0xe2, 0x82],
            vec![0xf0, 0x28, 0x8c, 0xbc], // overlong
            vec![0xed, 0xa0, 0x80],       // CESU-8 U+D800
            vec![0xed, 0xbf, 0xbf],       // CESU-8 U+DFFF
            vec![0xf4, 0x90, 0x80, 0x80], // > U+10FFFF
            vec![0xef, 0xbb, 0xbf],       // lone BOM
            b"http://\xff\xfe/x".to_vec(),
        ];
        bad_bytes.push((0u8..=255).collect());
        for i in 0..256 {
            for j in 0..256 {
                bad_bytes.push(vec![i as u8, j as u8]);
            }
        }
        for bytes in &bad_bytes {
            let s = utf8_decode_lossy(bytes);
            eat(parse(&s, None));
            eat(host::parse_host(&s, true));
            eat(idna::domain_to_ascii(&s));
        }
    }

    #[test]
    fn char_boundary_positions_never_panic() {
        // Multibyte characters abutting every ASCII structural
        // character, in both orders, so no byte-boundary assumption in
        // the scanner can split a code point.
        let multis = ["é", "€", "💩", "ß", "\u{feff}", "\u{200c}"];
        let structurals = ["/", "?", "#", "@", ":", "[", "]", "%", ".", "\\"];
        for m in multis {
            for s in structurals {
                for pair in [format!("{m}{s}"), format!("{s}{m}")] {
                    eat(parse(&format!("http://x/{pair}"), None));
                    eat(parse(&format!("http://{pair}/"), None));
                    eat(host::parse_host(&pair, true));
                    eat(percent_decode(&pair));
                }
            }
        }
    }
}
