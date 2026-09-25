//! RFC 6265 / RFC 6265bis cookie jar: `Set-Cookie` parsing, storage with
//! eviction, and `Cookie` header serialization.
//!
//! * Parsing follows RFC 6265 §5.2, plus real-world quirks: lenient
//!   `Expires` date formats (RFC 1123 / RFC 850 / ANSI C `asctime`, two-digit
//!   years), unquoted values, and surrounding `DQUOTE`s.
//! * Domain handling uses the vendored Public Suffix List ([`crate::psl`]):
//!   cookies are never set on public suffixes, and `Domain` attributes must
//!   domain-match the request host.
//! * `SameSite` defaults to `Lax`; `SameSite=None` without `Secure` is
//!   rejected; `__Secure-` / `__Host-` name prefixes are enforced
//!   (case-sensitive, per RFC 6265bis §4.1.3).
//! * Eviction: 4096 cookies total, 150 per effective domain, 4096 bytes of
//!   name+value per cookie; least-recently-used cookies go first.
//!
//! Times are seconds since the Unix epoch (`u64`); the caller owns the clock.
//! Hosts are plain strings — no URL-crate types anywhere in this API.

use std::borrow::Cow;

use crate::psl;

/// Total cookies the jar holds before LRU eviction kicks in.
const MAX_COOKIES: usize = 4096;
/// Cookies per effective domain before LRU eviction kicks in.
const MAX_COOKIES_PER_HOST: usize = 150;
/// Maximum `name.len() + value.len()` for a single cookie.
const MAX_COOKIE_SIZE: usize = 4096;

/// Cookie `SameSite` enforcement. Cookies without the attribute are treated
/// as `Lax` ("Lax-by-default"); unrecognized attribute values likewise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SameSite {
    Lax,
    Strict,
    None,
}

/// Everything the jar needs to know about a request, for both storing
/// (`Set-Cookie`) and retrieving (`Cookie`) cookies.
#[derive(Debug, Clone, Copy)]
pub struct RequestContext<'a> {
    /// Request host, e.g. `"www.example.com"`. Plain string; normalized
    /// internally (lowercased, one trailing dot stripped, any `:port`
    /// suffix dropped). Callers should pass bare hosts without a port.
    ///
    /// Hosts must be IDNA-ToASCII encoded (punycode) by the caller.
    /// `set_cookie` rejects non-ASCII hosts fail-closed — nothing is
    /// stored — because the public-suffix list only understands ASCII
    /// labels. UTS-46 encoding will move into `bowser_url` later; until
    /// then the caller owns it.
    pub host: &'a str,
    /// Request path, e.g. `"/a/b"`. Empty is treated as `"/"`.
    pub path: &'a str,
    /// Whether the request uses a secure transport (https/wss).
    pub secure: bool,
    /// True when the call comes from the network stack rather than
    /// `document.cookie`. `HttpOnly` cookies are invisible to non-HTTP APIs,
    /// and non-HTTP APIs cannot create, overwrite, or delete them.
    pub http_api: bool,
    /// Host of the top-level document, for `SameSite` decisions.
    /// `None` (unknown) is treated as cross-site. Like `host`, prefer a
    /// bare host without a port.
    ///
    /// Known phase-1 limitation: [`is_same_site`] compares only registrable
    /// domains, ignoring scheme — an `http` frame embedding an `https`
    /// resource (or vice versa) counts as same-site. Schemeful same-site
    /// is future work.
    pub top_host: Option<&'a str>,
    /// True for a top-level navigation with a "safe" method (GET/HEAD/...).
    /// Lets `Lax` cookies ride cross-site top-level navigations.
    pub top_level_safe_navigation: bool,
}

#[derive(Debug, Clone)]
struct Cookie {
    name: String,
    value: String,
    /// Effective domain, normalized (lowercase, no leading dot).
    domain: String,
    /// True when no `Domain` attribute was present.
    host_only: bool,
    path: String,
    secure: bool,
    http_only: bool,
    same_site: SameSite,
    /// Expiry as epoch seconds; `u64::MAX` for session cookies.
    expires_at: u64,
    /// Monotonic insertion order: earlier sorts first in `Cookie` headers.
    creation_seq: u64,
    /// Monotonic access order for LRU eviction.
    last_access: u64,
}

/// The cookie jar.
#[derive(Debug, Default)]
pub struct CookieJar {
    cookies: Vec<Cookie>,
    next_seq: u64,
}

impl CookieJar {
    /// An empty jar.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of cookies currently stored (including session cookies).
    pub fn len(&self) -> usize {
        self.cookies.len()
    }

    /// True when no cookies are stored.
    pub fn is_empty(&self) -> bool {
        self.cookies.is_empty()
    }

    /// Drop all cookies.
    pub fn clear(&mut self) {
        self.cookies.clear();
    }

    /// Process one `Set-Cookie` header value received in a response from
    /// `ctx.host` for path `ctx.path`, at time `now` (epoch seconds), or one
    /// `document.cookie` assignment from script (`ctx.http_api == false`).
    /// Attribute parsing is the same for both; only `HttpOnly` is treated
    /// specially for non-HTTP APIs (RFC 6265bis §5.3). Invalid or rejected
    /// cookies are silently ignored, per spec.
    ///
    /// `ctx.host` and any `Domain` attribute must be IDNA-ToASCII encoded
    /// (punycode) by the caller; non-ASCII hosts or domains are rejected
    /// fail-closed (nothing is stored), because the public-suffix list
    /// only understands ASCII labels. See the IDN gate below.
    pub fn set_cookie(&mut self, header: &str, ctx: &RequestContext<'_>, now: u64) {
        let normalized = psl::normalize_host(ctx.host);
        let host = strip_port(&normalized);
        if host.is_empty() {
            return;
        }
        let host_is_ip = is_ip_literal(host);

        // --- name=value pair (RFC 6265 §5.2 steps 1-4) ---
        let pair_end = header.find(';').unwrap_or(header.len());
        let (name, value) = match split_name_value(&header[..pair_end]) {
            Some(p) => p,
            None => return,
        };

        // --- attributes ---
        let mut domain_attr: Option<String> = None;
        let mut path_attr: Option<String> = None;
        let mut secure = false;
        let mut http_only = false;
        let mut same_site = SameSite::Lax;
        let mut samesite_seen = false;
        let mut max_age: Option<i64> = None;
        let mut expires: Option<u64> = None;

        // --- attributes (RFC 6265 §5.2 steps 5+, RFC 6265bis §5.3) ---
        // Attributes parse identically for HTTP and non-HTTP APIs; only
        // HttpOnly is special-cased below.
        for av in header[pair_end..].split(';').skip(1) {
            let (attr_name, attr_value) = split_attribute(av);
            match attr_name.as_str() {
                "expires" => {
                    if expires.is_none() {
                        expires = parse_cookie_date(attr_value);
                    }
                }
                "max-age" => {
                    if max_age.is_none() {
                        max_age = parse_max_age(attr_value);
                    }
                }
                "domain" => {
                    if domain_attr.is_none() {
                        domain_attr = parse_domain_attribute(attr_value);
                    }
                }
                "path" => {
                    if path_attr.is_none() {
                        path_attr = Some(attr_value.to_string());
                    }
                }
                "secure" => secure = true,
                "httponly" => http_only = true,
                "samesite" if !samesite_seen => {
                    samesite_seen = true;
                    same_site = parse_samesite(attr_value);
                }
                _ => {} // unknown attributes are ignored
            }
        }

        // RFC 6265bis §5.6: SameSite=None requires Secure.
        if same_site == SameSite::None && !secure {
            return;
        }
        // RFC 6265 §5.3 step 4 / 6265bis §4.1.3: a cookie with the
        // secure-only flag (the Secure attribute) cannot be stored when the
        // request arrived over an insecure scheme. Without this, an
        // attacker on the network path of an http:// response could plant
        // a Secure-marked cookie that is later sent over https://.
        if secure && !ctx.secure {
            return;
        }
        // RFC 6265bis §4.1.3: name prefixes (case-sensitive). The prefixes
        // also require a secure request scheme, not just the attribute.
        if name.starts_with("__Secure-") && !(secure && ctx.secure) {
            return;
        }
        if name.starts_with("__Host-")
            && !(secure && ctx.secure && path_attr.as_deref() == Some("/") && domain_attr.is_none())
        {
            return;
        }

        // --- IDN stopgap: fail closed on non-ASCII hosts and domains ---
        // The PSL trie stores ASCII (punycode) labels only, so a non-ASCII
        // label can never match; `psl::walk` then falls back to the `*`
        // rule and an IDN public suffix such as 公司.cn would be judged a
        // registrable domain — silently bypassing the public-suffix gate
        // (fail open) on both the Domain-attribute and host-only paths.
        // Until callers pass IDNA-ToASCII (punycode) hosts (planned for
        // `bowser_url`, UTS-46), reject non-ASCII request hosts and Domain
        // attributes outright, before either PSL gate below.
        if !host.is_ascii() {
            return;
        }
        if let Some(d) = &domain_attr
            && !d.is_ascii()
        {
            return;
        }

        // --- effective domain (RFC 6265 §5.3 steps 5-7) ---
        let (domain, host_only) = match domain_attr {
            Some(d) => {
                if host_is_ip {
                    return; // no Domain cookies on IP literals
                }
                if !domain_match(host, &d) {
                    return;
                }
                if psl::is_public_suffix(&d) && !is_localhost(&d) {
                    return; // no cookies on public suffixes
                }
                (d, false)
            }
            None => {
                if psl::is_public_suffix(host) && !host_is_ip && !is_localhost(host) {
                    return; // e.g. a bare public suffix cannot set cookies
                }
                (host.to_owned(), true)
            }
        };

        // --- expiry: Max-Age wins over Expires (RFC 6265 §5.3) ---
        // Note: a cookie whose expiry equals `now` is already expired —
        // expiry comparisons are strict (`expires_at > now`) everywhere.
        let expires_at = match (max_age, expires) {
            (Some(delta), _) => {
                if delta <= 0 {
                    0
                } else {
                    now.saturating_add(delta as u64)
                }
            }
            (None, Some(t)) => t,
            (None, None) => u64::MAX,
        };

        // --- path (RFC 6265 §5.3 step 10) ---
        let path = match path_attr {
            Some(p) if p.starts_with('/') => p,
            _ => default_path(ctx.path),
        };

        if name.len() + value.len() > MAX_COOKIE_SIZE {
            return;
        }

        // RFC 6265bis §5.3 step 10: a non-HTTP API must not create a cookie
        // with the http-only flag set, nor overwrite or delete a pre-existing
        // HttpOnly cookie.
        if !ctx.http_api
            && (http_only
                || self
                    .cookies
                    .iter()
                    .find(|c| c.name == name && c.domain == domain && c.path == path)
                    .is_some_and(|c| c.http_only))
        {
            return;
        }

        // An already-expired cookie deletes any existing cookie with the
        // same (name, domain, path) and is not stored.
        if expires_at <= now {
            self.cookies
                .retain(|c| !(c.name == name && c.domain == domain && c.path == path));
            return;
        }

        let seq = self.next_seq;
        self.next_seq += 1;

        if let Some(old) = self
            .cookies
            .iter_mut()
            .find(|c| c.name == name && c.domain == domain && c.path == path)
        {
            // Replace in place, keeping the original creation order.
            old.value = value;
            old.secure = secure;
            old.http_only = http_only;
            old.same_site = same_site;
            old.expires_at = expires_at;
            old.host_only = host_only;
            old.last_access = seq;
            return;
        }

        self.evict_if_needed(&domain);
        self.cookies.push(Cookie {
            name,
            value,
            domain,
            host_only,
            path,
            secure,
            http_only,
            same_site,
            expires_at,
            creation_seq: seq,
            last_access: seq,
        });
    }

    /// Build the value for an outgoing `Cookie` header for `ctx`, at time
    /// `now`. Returns `None` when no stored cookie applies. Expired cookies
    /// are purged. Matches are sorted by longest path first, then earliest
    /// creation time (RFC 6265 §5.4).
    pub fn cookie_header(&mut self, ctx: &RequestContext<'_>, now: u64) -> Option<String> {
        let normalized = psl::normalize_host(ctx.host);
        let host = strip_port(&normalized);
        if host.is_empty() {
            return None;
        }
        let top_normalized: Option<Cow<'_, str>> = ctx.top_host.map(psl::normalize_host);
        let top = top_normalized.as_deref().map(strip_port);
        let req_path = if ctx.path.is_empty() { "/" } else { ctx.path };

        self.cookies.retain(|c| c.expires_at > now);

        let mut hits: Vec<usize> = Vec::new();
        for (i, c) in self.cookies.iter().enumerate() {
            if c.host_only {
                if c.domain != host {
                    continue;
                }
            } else if !domain_match(host, &c.domain) {
                continue;
            }
            if !path_match(req_path, &c.path) {
                continue;
            }
            if c.secure && !ctx.secure {
                continue;
            }
            if c.http_only && !ctx.http_api {
                continue;
            }
            if !samesite_allows(c, host, top, ctx) {
                continue;
            }
            hits.push(i);
        }
        if hits.is_empty() {
            return None;
        }

        // LRU accounting, then RFC 6265 §5.4 ordering.
        let seq = self.next_seq;
        self.next_seq += 1;
        for &i in &hits {
            self.cookies[i].last_access = seq;
        }
        hits.sort_by(|&a, &b| {
            self.cookies[b]
                .path
                .len()
                .cmp(&self.cookies[a].path.len())
                .then_with(|| {
                    self.cookies[a]
                        .creation_seq
                        .cmp(&self.cookies[b].creation_seq)
                })
        });

        let mut out = String::new();
        for (k, &i) in hits.iter().enumerate() {
            if k > 0 {
                out.push_str("; ");
            }
            let c = &self.cookies[i];
            out.push_str(&c.name);
            out.push('=');
            out.push_str(&c.value);
        }
        Some(out)
    }

    /// Remove expired cookies; returns the number removed.
    pub fn remove_expired(&mut self, now: u64) -> usize {
        let before = self.cookies.len();
        self.cookies.retain(|c| c.expires_at > now);
        before - self.cookies.len()
    }

    /// Evict least-recently-used cookies while over either cap.
    fn evict_if_needed(&mut self, domain: &str) {
        while self.cookies.len() >= MAX_COOKIES && self.evict_one(None) {}
        while self.cookies.iter().filter(|c| c.domain == domain).count() >= MAX_COOKIES_PER_HOST
            && self.evict_one(Some(domain))
        {}
    }

    /// Remove the LRU cookie, optionally restricted to one domain.
    fn evict_one(&mut self, domain: Option<&str>) -> bool {
        let victim = self
            .cookies
            .iter()
            .enumerate()
            .filter(|(_, c)| domain.is_none_or(|d| c.domain == d))
            .min_by_key(|(_, c)| (c.last_access, c.creation_seq))
            .map(|(i, _)| i);
        match victim {
            Some(i) => {
                self.cookies.remove(i);
                true
            }
            None => false,
        }
    }
}

/// Split `name=value` (or a bare `name`); returns `None` for empty/invalid names.
fn split_name_value(pair: &str) -> Option<(String, String)> {
    let pair = pair.trim_matches(|c| c == ' ' || c == '\t');
    let (name, value) = match pair.find('=') {
        Some(i) => (&pair[..i], &pair[i + 1..]),
        None => (pair, ""),
    };
    let name = name.trim_matches(|c| c == ' ' || c == '\t');
    let value = value.trim_matches(|c| c == ' ' || c == '\t');
    if name.is_empty() || name.bytes().any(|b| b < 0x20 || b == 0x7f) {
        return None;
    }
    // RFC 6265 cookie-octet excludes CTLs; a value carrying them would be
    // emitted verbatim into the Cookie header, enabling header injection.
    if value.bytes().any(|b| b < 0x20 || b == 0x7f) {
        return None;
    }
    // Real-world quirk: strip one pair of surrounding DQUOTEs.
    let value = value
        .strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .unwrap_or(value);
    Some((name.to_string(), value.to_string()))
}

/// Split one `attribute` into lowercased name and trimmed value.
fn split_attribute(av: &str) -> (String, &str) {
    let av = av.trim_matches(|c| c == ' ' || c == '\t');
    match av.find('=') {
        Some(i) => (
            av[..i]
                .trim_matches(|c| c == ' ' || c == '\t')
                .to_ascii_lowercase(),
            av[i + 1..].trim_matches(|c| c == ' ' || c == '\t'),
        ),
        None => (av.to_ascii_lowercase(), ""),
    }
}

/// Normalize a `Domain` attribute value: trim, strip one leading dot and any
/// trailing dots, ASCII-lowercase. `None` for empty values.
fn parse_domain_attribute(value: &str) -> Option<String> {
    let v = value.trim_matches(|c| c == ' ' || c == '\t');
    let v = v.strip_prefix('.').unwrap_or(v).trim_end_matches('.');
    if v.is_empty() {
        return None;
    }
    Some(v.to_ascii_lowercase())
}

/// Parse `SameSite`; unknown values behave as `Lax` (lax-by-default).
fn parse_samesite(value: &str) -> SameSite {
    match value.to_ascii_lowercase().as_str() {
        "strict" => SameSite::Strict,
        "none" => SameSite::None,
        _ => SameSite::Lax,
    }
}

/// Parse `Max-Age`; `None` when the value is not a valid delta.
fn parse_max_age(value: &str) -> Option<i64> {
    let s = value.trim_matches(|c| c == ' ' || c == '\t');
    if s.is_empty() {
        return None;
    }
    let (neg, digits) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut v: i64 = 0;
    for b in digits.bytes() {
        v = v.saturating_mul(10).saturating_add((b - b'0') as i64);
    }
    Some(if neg { v.saturating_neg() } else { v })
}

/// Whether `host` domain-matches `domain` (RFC 6265 §5.1.3). Both must be
/// normalized (lowercase, no leading dot). IP literals only match identically.
fn domain_match(host: &str, domain: &str) -> bool {
    if host == domain {
        return true;
    }
    if host.len() <= domain.len() || is_ip_literal(host) {
        return false;
    }
    host.as_bytes()[host.len() - domain.len() - 1] == b'.' && host.ends_with(domain)
}

/// Whether `request_path` path-matches `cookie_path` (RFC 6265 §5.1.4).
fn path_match(request_path: &str, cookie_path: &str) -> bool {
    if request_path == cookie_path {
        return true;
    }
    if !request_path.starts_with(cookie_path) {
        return false;
    }
    cookie_path.ends_with('/') || request_path.as_bytes().get(cookie_path.len()) == Some(&b'/')
}

/// Default cookie path from the request path (RFC 6265 §5.1.4).
fn default_path(request_path: &str) -> String {
    if !request_path.starts_with('/') {
        return "/".to_string();
    }
    match request_path.rfind('/') {
        Some(0) | None => "/".to_string(),
        Some(i) => request_path[..i].to_string(),
    }
}

/// Lenient IP-literal check: IPv4 in any of the WHATWG URL Standard's
/// notations (decimal/octal/hex dotted parts, or a single integer), or
/// anything with a colon (IPv6, optionally bracketed). A `host:port`
/// string is NOT an IP literal — strip the port with [`strip_port`] first.
fn is_ip_literal(host: &str) -> bool {
    let h = host
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(host);
    if h.contains(':') {
        return true;
    }
    parse_ipv4(h).is_some()
}

/// Parse an IPv4 address per the WHATWG URL Standard's IPv4 parser: 1–4
/// dot-separated parts, each decimal, octal (`0…`), or hex (`0x…`); all but
/// the last part must fit in a byte, and the last must fit in the remaining
/// bytes (`256^(5 - parts)`). Returns the 32-bit value on success.
fn parse_ipv4(input: &str) -> Option<u32> {
    let parts: Vec<&str> = input.split('.').collect();
    if parts.is_empty() || parts.len() > 4 || parts.iter().any(|p| p.is_empty()) {
        return None;
    }
    let mut value: u64 = 0;
    let n = parts.len();
    for (i, part) in parts.iter().enumerate() {
        let num = parse_ipv4_part(part)?;
        let max = if i == n - 1 {
            256u64.pow(5 - n as u32)
        } else {
            256
        };
        if num >= max {
            return None;
        }
        value += num << (8 * (n - 1 - i));
    }
    u32::try_from(value).ok()
}

/// Parse one IPv4 part: hex with a `0x`/`0X` prefix, octal with a leading
/// `0`, decimal otherwise. Overflow or invalid digits are an error.
fn parse_ipv4_part(s: &str) -> Option<u64> {
    let (digits, radix): (&str, u32) = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => (hex, 16),
        None if s.len() > 1 && s.starts_with('0') => (s, 8),
        None => (s, 10),
    };
    if digits.is_empty() || !digits.bytes().all(|b| (b as char).is_digit(radix)) {
        return None;
    }
    let mut v: u64 = 0;
    for b in digits.bytes() {
        v = v
            .checked_mul(radix as u64)?
            .checked_add((b as char).to_digit(radix)? as u64)?;
    }
    Some(v)
}

/// Strip a `:port` suffix from a normalized host. Bracketed IPv6 keeps its
/// brackets: `[::1]:80` becomes `[::1]`. An unbracketed host with more than
/// one colon (a bare IPv6 literal) is left alone.
fn strip_port(host: &str) -> &str {
    if let Some(inner) = host.strip_prefix('[') {
        return match inner.find(']') {
            Some(i) => &host[..i + 2],
            None => host,
        };
    }
    match host.rfind(':') {
        Some(i) if !host[..i].contains(':') => &host[..i],
        _ => host,
    }
}

fn is_localhost(host: &str) -> bool {
    host == "localhost"
}

/// Same-site per registrable domains; unknown or IP top contexts are
/// conservative. Hosts must be normalized.
///
/// Known phase-1 limitation: scheme is ignored, so an `http` page embedding
/// an `https` resource (or vice versa) counts as same-site. Schemeful
/// same-site is future work.
fn is_same_site(host: &str, top: &str) -> bool {
    if host == top {
        return true;
    }
    if is_ip_literal(host) || is_ip_literal(top) {
        return false;
    }
    match (psl::registrable_domain(host), psl::registrable_domain(top)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

fn samesite_allows(
    cookie: &Cookie,
    host: &str,
    top_host: Option<&str>,
    ctx: &RequestContext<'_>,
) -> bool {
    let same_site = top_host.is_some_and(|t| is_same_site(host, t));
    match cookie.same_site {
        SameSite::None => true,
        SameSite::Strict => same_site,
        // A top-level safe navigation lets Lax cookies ride along whether
        // the read comes from the network stack or `document.cookie`.
        SameSite::Lax => same_site || ctx.top_level_safe_navigation,
    }
}

// ---------------------------------------------------------------------------
// cookie-date parsing (RFC 6265 §5.1.1 + real-world quirks)
// ---------------------------------------------------------------------------

/// Parse an `Expires` attribute value into epoch seconds. Accepts the RFC
/// 6265 cookie-date grammar regardless of field order, plus the formats
/// servers actually send: IMF-fixdate (`Sun, 06 Nov 1994 08:49:37 GMT`),
/// RFC 850 (`Sunday, 06-Nov-94 08:49:37 GMT`), ANSI C `asctime`
/// (`Sun Nov  6 08:49:37 1994`), two-digit years, and stray tokens.
/// Returns `None` when the value is not a usable date.
fn parse_cookie_date(s: &str) -> Option<u64> {
    let mut time: Option<(u32, u32, u32)> = None;
    let mut month: Option<u32> = None;
    let mut nums: Vec<&str> = Vec::new();

    for tok in cookie_date_tokens(s) {
        if tok.contains(':') {
            if time.is_none() {
                time = parse_time(tok);
            }
        } else if tok.bytes().all(|b| b.is_ascii_digit()) {
            nums.push(tok);
        } else if month.is_none() {
            month = parse_month(tok);
        }
    }
    let (h, m, sec) = time?;
    let month = month?;

    // The year is the first number with 3+ digits; otherwise, with two or
    // more numbers, the last is the year and the first is the day.
    let (day_tok, year_tok) = match nums.iter().position(|n| n.len() >= 3) {
        Some(yi) => {
            let day = nums
                .iter()
                .enumerate()
                .find(|(i, _)| *i != yi)
                .map(|(_, n)| *n);
            (day, Some(nums[yi]))
        }
        None if nums.len() >= 2 => (Some(nums[0]), Some(nums[nums.len() - 1])),
        None => (None, None),
    };
    let (day_tok, year_tok) = (day_tok?, year_tok?);
    if year_tok.len() != 2 && year_tok.len() != 4 {
        return None;
    }
    let day: u32 = day_tok.parse().ok()?;
    let mut year: i32 = year_tok.parse().ok()?;
    if year_tok.len() == 2 {
        year += if year >= 70 { 1900 } else { 2000 };
    }
    if !(1601..=9999).contains(&year) || day == 0 || day > days_in_month(year, month) {
        return None;
    }
    if h > 23 || m > 59 || sec > 59 {
        return None;
    }
    Some(epoch_seconds(year, month, day, h, m, sec))
}

/// Maximal runs of non-delimiter characters, where delimiters are the
/// RFC 6265 cookie-date delimiter set. `:` is not a delimiter, so times
/// stay in one token.
fn cookie_date_tokens(s: &str) -> impl Iterator<Item = &str> {
    fn is_delim(c: char) -> bool {
        matches!(c,
            '\u{09}' | '\u{20}'..='\u{2F}'
            | '\u{3B}'..='\u{40}' | '\u{5B}'..='\u{60}' | '\u{7B}'..='\u{7E}')
    }
    let mut rest = s;
    std::iter::from_fn(move || {
        let start = rest.find(|c: char| !is_delim(c))?;
        rest = &rest[start..];
        let end = rest.find(is_delim).unwrap_or(rest.len());
        let tok = &rest[..end];
        rest = &rest[end..];
        Some(tok)
    })
}

fn parse_time(tok: &str) -> Option<(u32, u32, u32)> {
    let mut parts = tok.split(':');
    let h: u32 = parts.next()?.parse().ok()?;
    let m: u32 = parts.next()?.parse().ok()?;
    let s: u32 = match parts.next() {
        Some(p) => p.parse().ok()?,
        None => 0,
    };
    if parts.next().is_some() {
        return None;
    }
    Some((h, m, s))
}

fn parse_month(tok: &str) -> Option<u32> {
    let abbrev = tok.get(..3)?.to_ascii_lowercase();
    Some(match abbrev.as_str() {
        "jan" => 1,
        "feb" => 2,
        "mar" => 3,
        "apr" => 4,
        "may" => 5,
        "jun" => 6,
        "jul" => 7,
        "aug" => 8,
        "sep" => 9,
        "oct" => 10,
        "nov" => 11,
        "dec" => 12,
        _ => return None,
    })
}

fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Days-from-civil (Howard Hinnant's algorithm); pre-1970 dates saturate to 0.
fn epoch_seconds(year: i32, month: u32, day: u32, h: u32, m: u32, s: u32) -> u64 {
    let y = if month <= 2 {
        year as i64 - 1
    } else {
        year as i64
    };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    (days * 86_400 + h as i64 * 3600 + m as i64 * 60 + s as i64).max(0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_790_000_000; // ~2026-09-25

    fn ctx<'a>(host: &'a str, path: &'a str) -> RequestContext<'a> {
        RequestContext {
            host,
            path,
            secure: true,
            http_api: true,
            top_host: Some(host),
            top_level_safe_navigation: false,
        }
    }

    fn get(jar: &mut CookieJar, host: &str, path: &str) -> Option<String> {
        jar.cookie_header(&ctx(host, path), NOW)
    }

    #[test]
    fn basic_set_and_get() {
        let mut jar = CookieJar::new();
        assert!(jar.is_empty());
        jar.set_cookie("a=b", &ctx("www.example.com", "/"), NOW);
        assert_eq!(jar.len(), 1);
        assert_eq!(get(&mut jar, "www.example.com", "/"), Some("a=b".into()));
        assert_eq!(
            get(&mut jar, "www.example.com", "/deep/path"),
            Some("a=b".into())
        );
    }

    #[test]
    fn host_only_vs_domain_cookies() {
        let mut jar = CookieJar::new();
        jar.set_cookie("h=1", &ctx("www.example.com", "/"), NOW);
        // Host-only: not sent to subdomains.
        assert_eq!(get(&mut jar, "sub.example.com", "/"), None);
        assert_eq!(get(&mut jar, "www.example.com", "/"), Some("h=1".into()));

        jar.set_cookie("d=2; Domain=example.com", &ctx("www.example.com", "/"), NOW);
        assert_eq!(get(&mut jar, "sub.example.com", "/"), Some("d=2".into()));
        assert_eq!(
            get(&mut jar, "deep.sub.example.com", "/"),
            Some("d=2".into())
        );
        // Not sent to unrelated domains or suffix-lookalikes.
        assert_eq!(get(&mut jar, "other.com", "/"), None);
        assert_eq!(get(&mut jar, "badexample.com", "/"), None);
        assert_eq!(get(&mut jar, "example.com", "/"), Some("d=2".into()));
    }

    #[test]
    fn domain_mismatch_rejected() {
        let mut jar = CookieJar::new();
        jar.set_cookie("x=1; Domain=other.com", &ctx("www.example.com", "/"), NOW);
        assert!(jar.is_empty());
        jar.set_cookie("x=1; Domain=com", &ctx("www.example.com", "/"), NOW);
        assert!(jar.is_empty());
    }

    #[test]
    fn public_suffix_rejected() {
        let mut jar = CookieJar::new();
        // Domain attribute on a public suffix.
        jar.set_cookie("x=1; Domain=com", &ctx("www.example.com", "/"), NOW);
        jar.set_cookie("x=1; Domain=co.uk", &ctx("www.example.co.uk", "/"), NOW);
        jar.set_cookie("x=1; Domain=b.ck", &ctx("a.b.ck", "/"), NOW);
        assert!(jar.is_empty());
        // Host-only cookie set from the public suffix itself.
        jar.set_cookie("x=1", &ctx("co.uk", "/"), NOW);
        jar.set_cookie("x=1", &ctx("com", "/"), NOW);
        assert!(jar.is_empty());
    }

    #[test]
    fn idn_hosts_and_domains_rejected_fail_closed() {
        let mut jar = CookieJar::new();
        // 公司.cn is a public suffix, but the PSL trie stores ASCII
        // (punycode) labels only: without the fail-closed gate the
        // non-ASCII spelling would never match, `psl::walk` would fall back
        // to the `*` rule, and the Domain cookie below would be stored as
        // readable by every other registrant under 公司.cn.
        jar.set_cookie("s=1; Domain=公司.cn", &ctx("a.公司.cn", "/"), NOW);
        assert!(jar.is_empty());
        // Same for the host-only path: a bare Set-Cookie from the public
        // suffix itself must not be accepted either.
        jar.set_cookie("s=1", &ctx("公司.cn", "/"), NOW);
        assert!(jar.is_empty());
        // Non-ASCII in the Domain attribute is rejected even when the
        // request host is plain ASCII.
        jar.set_cookie("s=1; Domain=公司.cn", &ctx("a.example.com", "/"), NOW);
        assert!(jar.is_empty());
        // Nothing stored means nothing readable by sibling hosts.
        assert_eq!(get(&mut jar, "b.公司.cn", "/"), None);
    }

    #[test]
    fn punycode_spellings_behave_as_before() {
        let mut jar = CookieJar::new();
        // Pure-ASCII (punycode) inputs are untouched by the fail-closed
        // gate: they are judged by the PSL exactly as before. So the
        // punycode spelling of 公司.cn is rejected or accepted precisely
        // when `psl::is_public_suffix` says so — never by the ASCII gate.
        jar.set_cookie(
            "s=1; Domain=xn--55qx5d.cn",
            &ctx("a.xn--55qx5d.cn", "/"),
            NOW,
        );
        assert_eq!(jar.is_empty(), psl::is_public_suffix("xn--55qx5d.cn"));
        // A normal ASCII public suffix is still rejected by the PSL gate.
        // (Bin-independent: "com" is a PSL rule, and even the "*" fallback
        // reports a single label as a public suffix.)
        jar.clear();
        jar.set_cookie("s=1; Domain=com", &ctx("www.example.com", "/"), NOW);
        assert!(jar.is_empty());
        // A host-only cookie from a normal punycode host is accepted.
        jar.set_cookie("p=2", &ctx("www.xn--55qx5d.cn", "/"), NOW);
        assert_eq!(get(&mut jar, "www.xn--55qx5d.cn", "/"), Some("p=2".into()));
    }

    #[test]
    fn domain_attribute_normalization() {
        let mut jar = CookieJar::new();
        jar.set_cookie(
            "x=1; Domain=.Example.COM",
            &ctx("www.example.com", "/"),
            NOW,
        );
        assert_eq!(get(&mut jar, "sub.example.com", "/"), Some("x=1".into()));
        // Empty Domain behaves as absent (host-only).
        jar.set_cookie("y=2; Domain=", &ctx("www.example.com", "/"), NOW);
        assert_eq!(get(&mut jar, "sub.example.com", "/"), Some("x=1".into()));
    }

    #[test]
    fn expires_absolute_dates() {
        let mut jar = CookieJar::new();
        // Long past -> never stored.
        jar.set_cookie(
            "a=1; Expires=Sun, 06 Nov 1994 08:49:37 GMT",
            &ctx("h.test", "/"),
            NOW,
        );
        assert!(jar.is_empty());
        // Far future -> stored.
        jar.set_cookie(
            "b=2; Expires=Tue, 19 Jan 2038 03:14:07 GMT",
            &ctx("h.test", "/"),
            NOW,
        );
        assert_eq!(jar.len(), 1);
        assert_eq!(get(&mut jar, "h.test", "/"), Some("b=2".into()));
        // Setting it again with a past date deletes it.
        jar.set_cookie(
            "b=gone; Expires=Thu, 01 Jan 1970 00:00:00 GMT",
            &ctx("h.test", "/"),
            NOW,
        );
        assert!(jar.is_empty());
    }

    #[test]
    fn expires_quirks() {
        let mut jar = CookieJar::new();
        // RFC 850, ANSI C asctime, and two-digit years all parse.
        for hdr in [
            "a=1; Expires=Sunday, 06-Nov-94 08:49:37 GMT",
            "a=1; Expires=Sun Nov  6 08:49:37 1994",
            "a=1; expires=thu, 01-jan-1970 00:00:10 gmt",
        ] {
            jar.set_cookie(hdr, &ctx("h.test", "/"), NOW);
        }
        assert!(jar.is_empty(), "all of these dates are in the past");
        // Unparseable Expires -> session cookie.
        jar.set_cookie("s=9; Expires=not a date", &ctx("h.test", "/"), NOW);
        assert_eq!(get(&mut jar, "h.test", "/"), Some("s=9".into()));
        assert_eq!(jar.remove_expired(u64::MAX), 1); // even session cookies go at the end of time
    }

    #[test]
    fn max_age() {
        let mut jar = CookieJar::new();
        jar.set_cookie("a=1; Max-Age=3600", &ctx("h.test", "/"), NOW);
        assert_eq!(get(&mut jar, "h.test", "/"), Some("a=1".into()));
        assert!(jar.cookie_header(&ctx("h.test", "/"), NOW + 3599).is_some());
        assert!(jar.cookie_header(&ctx("h.test", "/"), NOW + 3600).is_none());

        // Max-Age=0 deletes.
        jar.set_cookie("b=2; Max-Age=100", &ctx("h.test", "/"), NOW);
        jar.set_cookie("b=3; Max-Age=0", &ctx("h.test", "/"), NOW);
        assert!(jar.cookies.iter().all(|c| c.name != "b"));

        // Negative deletes too.
        jar.set_cookie("c=4; Max-Age=100", &ctx("h.test", "/"), NOW);
        jar.set_cookie("c=5; Max-Age=-20", &ctx("h.test", "/"), NOW);
        assert!(jar.cookies.iter().all(|c| c.name != "c"));

        // Invalid Max-Age -> session cookie.
        jar.set_cookie("d=6; Max-Age=soon", &ctx("h.test", "/"), NOW);
        assert!(
            jar.cookie_header(&ctx("h.test", "/"), NOW + 10_000_000)
                .is_some()
        );

        // Max-Age wins over Expires.
        jar.set_cookie(
            "e=7; Expires=Tue, 19 Jan 2038 03:14:07 GMT; Max-Age=10",
            &ctx("h.test", "/"),
            NOW,
        );
        let h = jar.cookie_header(&ctx("h.test", "/"), NOW + 11).unwrap();
        assert!(!h.contains("e=7"), "{h}");
        assert!(h.contains("d=6"), "{h}"); // the session cookie survives
    }

    #[test]
    fn secure_flag() {
        let mut jar = CookieJar::new();
        let https = ctx("www.example.com", "/");
        let http = RequestContext {
            secure: false,
            ..https
        };
        jar.set_cookie("s=1; Secure", &https, NOW);
        assert_eq!(jar.cookie_header(&http, NOW), None);
        assert_eq!(jar.cookie_header(&https, NOW), Some("s=1".into()));
        // Non-secure cookies go everywhere.
        jar.set_cookie("p=2", &https, NOW);
        assert_eq!(jar.cookie_header(&http, NOW), Some("p=2".into()));
    }

    #[test]
    fn http_only() {
        let mut jar = CookieJar::new();
        let https = ctx("www.example.com", "/");
        let script = RequestContext {
            http_api: false,
            ..https
        };
        jar.set_cookie("h=1; HttpOnly", &https, NOW);
        assert_eq!(jar.cookie_header(&script, NOW), None);
        assert_eq!(jar.cookie_header(&https, NOW), Some("h=1".into()));
        // document.cookie cannot overwrite an HttpOnly cookie.
        jar.set_cookie("h=evil", &script, NOW);
        assert_eq!(jar.cookie_header(&https, NOW), Some("h=1".into()));
        // But the network stack can.
        jar.set_cookie("h=2; HttpOnly", &https, NOW);
        assert_eq!(jar.cookie_header(&https, NOW), Some("h=2".into()));
    }

    #[test]
    fn document_cookie_parses_attributes() {
        let mut jar = CookieJar::new();
        let script = RequestContext {
            http_api: false,
            ..ctx("www.example.com", "/")
        };
        // Domain attribute: a script set creates a real domain cookie,
        // visible to subdomains (RFC 6265bis §5.3, non-HTTP APIs).
        jar.set_cookie("x=1; Domain=example.com", &script, NOW);
        let script_sub = RequestContext {
            http_api: false,
            ..ctx("sub.example.com", "/")
        };
        assert_eq!(jar.cookie_header(&script_sub, NOW), Some("x=1".into()));
        // A mismatched Domain is still rejected for script sets.
        jar.set_cookie("bad=1; Domain=evil.com", &script, NOW);
        assert_eq!(jar.cookie_header(&script_sub, NOW), Some("x=1".into()));

        // Path attribute honored.
        jar.set_cookie("p=1; Path=/sub", &script, NOW);
        let script_deep = RequestContext {
            http_api: false,
            ..ctx("www.example.com", "/sub/deep")
        };
        assert_eq!(
            jar.cookie_header(&script_deep, NOW),
            Some("p=1; x=1".into())
        );
        let script_root = RequestContext {
            http_api: false,
            ..ctx("www.example.com", "/")
        };
        assert_eq!(jar.cookie_header(&script_root, NOW), Some("x=1".into()));

        // Max-Age honored.
        jar.set_cookie("m=1; Max-Age=60", &script, NOW);
        assert!(
            jar.cookie_header(&script_root, NOW + 59)
                .unwrap()
                .contains("m=1")
        );
        assert!(
            !jar.cookie_header(&script_root, NOW + 60)
                .unwrap()
                .contains("m=1")
        );

        // HttpOnly: a non-HTTP API must reject the cookie entirely
        // (RFC 6265bis §5.3 step 10).
        jar.set_cookie("ho=1; HttpOnly", &script, NOW);
        assert!(!jar.cookies.iter().any(|c| c.name == "ho"));

        // Scripts can still overwrite and delete non-HttpOnly cookies.
        jar.set_cookie("x=2; Domain=example.com", &script, NOW);
        assert_eq!(jar.cookie_header(&script_sub, NOW), Some("x=2".into()));
        jar.set_cookie("x=gone; Domain=example.com; Max-Age=0", &script, NOW);
        assert_eq!(jar.cookie_header(&script_sub, NOW), None);
    }

    #[test]
    fn document_cookie_cannot_touch_httponly() {
        let mut jar = CookieJar::new();
        let https = ctx("www.example.com", "/");
        let script = RequestContext {
            http_api: false,
            ..https
        };
        jar.set_cookie("k=1; HttpOnly", &https, NOW);
        // Scripts cannot overwrite an HttpOnly cookie...
        jar.set_cookie("k=evil", &script, NOW);
        assert_eq!(jar.cookie_header(&https, NOW), Some("k=1".into()));
        // ...nor delete one via an expired set.
        jar.set_cookie("k=gone; Max-Age=0", &script, NOW);
        assert_eq!(jar.cookie_header(&https, NOW), Some("k=1".into()));
        // The network stack still can.
        jar.set_cookie("k=2; HttpOnly", &https, NOW);
        assert_eq!(jar.cookie_header(&https, NOW), Some("k=2".into()));
    }

    #[test]
    fn document_cookie_parses_secure_and_samesite() {
        let mut jar = CookieJar::new();
        let script = RequestContext {
            http_api: false,
            ..ctx("www.example.com", "/")
        };
        // Secure attribute parsed: not sent over an insecure transport.
        jar.set_cookie("s=1; Secure", &script, NOW);
        let script_http = RequestContext {
            secure: false,
            ..script
        };
        assert_eq!(jar.cookie_header(&script, NOW), Some("s=1".into()));
        assert_eq!(jar.cookie_header(&script_http, NOW), None);

        // SameSite attribute parsed: Strict stays same-site-only.
        let mut jar = CookieJar::new();
        jar.set_cookie("t=1; SameSite=Strict", &script, NOW);
        let script_cross = RequestContext {
            top_host: Some("other.com"),
            ..script
        };
        assert_eq!(jar.cookie_header(&script_cross, NOW), None);
        assert_eq!(jar.cookie_header(&script, NOW), Some("t=1".into()));
    }

    #[test]
    fn path_matching() {
        let mut jar = CookieJar::new();
        jar.set_cookie("a=1; Path=/sub", &ctx("www.example.com", "/"), NOW);
        assert_eq!(get(&mut jar, "www.example.com", "/sub"), Some("a=1".into()));
        assert_eq!(
            get(&mut jar, "www.example.com", "/sub/deep"),
            Some("a=1".into())
        );
        assert_eq!(get(&mut jar, "www.example.com", "/"), None);
        assert_eq!(get(&mut jar, "www.example.com", "/other"), None);
        assert_eq!(get(&mut jar, "www.example.com", "/subother"), None);

        // Default path: /a/b -> /a.
        jar.set_cookie("b=2", &ctx("www.example.com", "/a/b"), NOW);
        assert_eq!(get(&mut jar, "www.example.com", "/a"), Some("b=2".into()));
        assert_eq!(get(&mut jar, "www.example.com", "/a/c"), Some("b=2".into()));
        assert_eq!(get(&mut jar, "www.example.com", "/"), None);

        // Path without a leading slash falls back to the default path.
        jar.set_cookie("c=3; Path=sub", &ctx("www.example.com", "/x/y"), NOW);
        assert_eq!(get(&mut jar, "www.example.com", "/x"), Some("c=3".into()));
        assert_eq!(get(&mut jar, "www.example.com", "/x/z"), Some("c=3".into()));
    }

    #[test]
    fn default_path_cases() {
        assert_eq!(default_path(""), "/");
        assert_eq!(default_path("noslash"), "/");
        assert_eq!(default_path("/"), "/");
        assert_eq!(default_path("/a"), "/");
        assert_eq!(default_path("/a/"), "/a");
        assert_eq!(default_path("/a/b"), "/a");
        assert_eq!(default_path("/a/b/"), "/a/b");
    }

    #[test]
    fn path_match_cases() {
        assert!(path_match("/a", "/a"));
        assert!(path_match("/a/b", "/a"));
        assert!(path_match("/a/b", "/a/"));
        assert!(!path_match("/a", "/a/"));
        assert!(!path_match("/ab", "/a"));
        assert!(path_match("/", "/"));
    }

    #[test]
    fn domain_match_cases() {
        assert!(domain_match("example.com", "example.com"));
        assert!(domain_match("www.example.com", "example.com"));
        assert!(!domain_match("badexample.com", "example.com"));
        assert!(!domain_match("example.com", "www.example.com"));
        assert!(domain_match("1.2.3.4", "1.2.3.4"));
        assert!(!domain_match("5.2.3.4", "2.3.4"));
    }

    #[test]
    fn samesite_strict() {
        let mut jar = CookieJar::new();
        jar.set_cookie("s=1; SameSite=Strict", &ctx("id.example.com", "/"), NOW);
        let cross = RequestContext {
            host: "id.example.com",
            path: "/",
            secure: true,
            http_api: true,
            top_host: Some("other.com"),
            top_level_safe_navigation: true,
        };
        assert_eq!(jar.cookie_header(&cross, NOW), None);
        // Same-site (sibling subdomain) is fine.
        let same = RequestContext {
            top_host: Some("www.example.com"),
            ..cross
        };
        assert_eq!(jar.cookie_header(&same, NOW), Some("s=1".into()));
        // Unknown top context is treated as cross-site.
        let unknown = RequestContext {
            top_host: None,
            ..cross
        };
        assert_eq!(jar.cookie_header(&unknown, NOW), None);
    }

    #[test]
    fn samesite_lax() {
        let mut jar = CookieJar::new();
        jar.set_cookie("s=1; SameSite=Strict", &ctx("id.example.com", "/"), NOW);
        jar.set_cookie("l=2; SameSite=Lax", &ctx("id.example.com", "/"), NOW);
        let base = RequestContext {
            host: "id.example.com",
            path: "/",
            secure: true,
            http_api: true,
            top_host: Some("other.com"),
            top_level_safe_navigation: false,
        };
        // Cross-site subrequest: neither Strict nor Lax is sent.
        assert_eq!(jar.cookie_header(&base, NOW), None);
        // Cross-site top-level navigation: Lax rides along, Strict does not.
        let nav = RequestContext {
            top_level_safe_navigation: true,
            ..base
        };
        assert_eq!(jar.cookie_header(&nav, NOW), Some("l=2".into()));
        // Unset SameSite defaults to Lax.
        jar.set_cookie("d=3", &ctx("id.example.com", "/"), NOW);
        assert_eq!(jar.cookie_header(&nav, NOW), Some("l=2; d=3".into()));
        assert_eq!(jar.cookie_header(&base, NOW), None);
    }

    #[test]
    fn samesite_none_requires_secure() {
        let mut jar = CookieJar::new();
        // Rejected outright without Secure (RFC 6265bis §5.6).
        jar.set_cookie("n=1; SameSite=None", &ctx("id.example.com", "/"), NOW);
        assert!(jar.is_empty());
        // With Secure it is sent cross-site.
        jar.set_cookie(
            "n=2; SameSite=None; Secure",
            &ctx("id.example.com", "/"),
            NOW,
        );
        let cross = RequestContext {
            host: "id.example.com",
            path: "/",
            secure: true,
            http_api: true,
            top_host: Some("other.com"),
            top_level_safe_navigation: false,
        };
        assert_eq!(jar.cookie_header(&cross, NOW), Some("n=2".into()));
        // Unknown values behave as Lax.
        jar.set_cookie("u=3; SameSite=Bogus", &ctx("id.example.com", "/"), NOW);
        assert_eq!(jar.cookie_header(&cross, NOW), Some("n=2".into()));
        let nav = RequestContext {
            top_level_safe_navigation: true,
            ..cross
        };
        assert_eq!(jar.cookie_header(&nav, NOW), Some("n=2; u=3".into()));
    }

    #[test]
    fn name_prefixes() {
        let mut jar = CookieJar::new();
        let https = ctx("www.example.com", "/");
        // __Secure- requires Secure.
        jar.set_cookie("__Secure-s=1", &https, NOW);
        assert!(jar.is_empty());
        jar.set_cookie("__Secure-s=1; Secure", &https, NOW);
        assert_eq!(jar.len(), 1);
        // __Host- requires Secure + Path=/ + no Domain.
        jar.set_cookie("__Host-a=1; Secure", &https, NOW);
        assert_eq!(jar.len(), 1);
        jar.set_cookie("__Host-a=1; Secure; Path=/", &https, NOW);
        assert_eq!(jar.len(), 2);
        jar.set_cookie(
            "__Host-b=2; Secure; Path=/; Domain=example.com",
            &https,
            NOW,
        );
        assert_eq!(jar.len(), 2);
        // Prefixes are case-sensitive: lowercase is not special.
        jar.set_cookie("__secure-c=3", &https, NOW);
        jar.set_cookie("__host-d=4; Path=/", &https, NOW);
        assert_eq!(jar.len(), 4);
    }

    #[test]
    fn secure_prefix_requires_secure_scheme() {
        let mut jar = CookieJar::new();
        let https = ctx("www.example.com", "/");
        let http = RequestContext {
            secure: false,
            ..https
        };
        // __Secure- over an insecure scheme is rejected, even with Secure.
        jar.set_cookie("__Secure-s=1; Secure", &http, NOW);
        assert!(jar.is_empty());
        // Control: same cookie over https:// is accepted.
        jar.set_cookie("__Secure-s=1; Secure", &https, NOW);
        assert_eq!(jar.len(), 1);
        // Without the Secure attribute it is rejected on both schemes.
        jar.clear();
        jar.set_cookie("__Secure-s=1", &https, NOW);
        assert!(jar.is_empty());
        jar.set_cookie("__Secure-s=1", &http, NOW);
        assert!(jar.is_empty());
    }

    #[test]
    fn host_prefix_requires_secure_scheme() {
        let mut jar = CookieJar::new();
        let https = ctx("www.example.com", "/");
        let http = RequestContext {
            secure: false,
            ..https
        };
        // __Host- over an insecure scheme is rejected, even with Secure + Path=/.
        jar.set_cookie("__Host-h=1; Secure; Path=/", &http, NOW);
        assert!(jar.is_empty());
        // Control: same cookie over https:// is accepted.
        jar.set_cookie("__Host-h=1; Secure; Path=/", &https, NOW);
        assert_eq!(jar.len(), 1);
    }

    #[test]
    fn secure_cookies_require_secure_scheme() {
        let mut jar = CookieJar::new();
        let https = ctx("www.example.com", "/");
        let http = RequestContext {
            secure: false,
            ..https
        };
        // A plain Secure cookie set over http:// is rejected: the jar stays
        // empty, so nothing planted over the network rides later https://
        // requests.
        jar.set_cookie("s=1; Secure", &http, NOW);
        assert!(jar.is_empty());
        // Control: same cookie over https:// is accepted and stays https-only.
        jar.set_cookie("s=1; Secure", &https, NOW);
        assert_eq!(jar.len(), 1);
        assert_eq!(jar.cookie_header(&http, NOW), None);
        assert_eq!(jar.cookie_header(&https, NOW), Some("s=1".into()));
        // Non-Secure cookies are still storable over http://.
        jar.set_cookie("p=2", &http, NOW);
        assert_eq!(jar.len(), 2);
        assert_eq!(jar.cookie_header(&http, NOW), Some("p=2".into()));
    }

    #[test]
    fn replacement_keeps_creation_order() {
        let mut jar = CookieJar::new();
        jar.set_cookie("a=1; Path=/", &ctx("h.test", "/"), NOW);
        jar.set_cookie("b=2; Path=/sub", &ctx("h.test", "/"), NOW);
        jar.set_cookie("a=3; Path=/", &ctx("h.test", "/"), NOW);
        // Longer paths first, then earlier creation time.
        assert_eq!(
            jar.cookie_header(&ctx("h.test", "/sub/deep"), NOW),
            Some("b=2; a=3".into())
        );
        assert_eq!(
            jar.cookie_header(&ctx("h.test", "/"), NOW),
            Some("a=3".into())
        );
    }

    #[test]
    fn header_sorting() {
        let mut jar = CookieJar::new();
        jar.set_cookie("z=1; Path=/", &ctx("h.test", "/"), NOW);
        jar.set_cookie("a=2; Path=/deep/path", &ctx("h.test", "/"), NOW);
        jar.set_cookie("m=3; Path=/deep", &ctx("h.test", "/"), NOW);
        assert_eq!(
            jar.cookie_header(&ctx("h.test", "/deep/path/x"), NOW),
            Some("a=2; m=3; z=1".into())
        );
    }

    #[test]
    fn per_host_eviction_is_lru() {
        let mut jar = CookieJar::new();
        for i in 0..160 {
            jar.set_cookie(&format!("c{i}=v"), &ctx("www.example.com", "/"), NOW);
        }
        assert_eq!(jar.len(), MAX_COOKIES_PER_HOST);
        assert!(!jar.cookies.iter().any(|c| c.name == "c0"));
        assert!(!jar.cookies.iter().any(|c| c.name == "c9"));
        assert!(jar.cookies.iter().any(|c| c.name == "c10"));
        assert!(jar.cookies.iter().any(|c| c.name == "c159"));
        // Other hosts are unaffected.
        jar.set_cookie("other=1", &ctx("www.other.com", "/"), NOW);
        assert_eq!(jar.len(), MAX_COOKIES_PER_HOST + 1);
    }

    #[test]
    fn cookie_size_cap() {
        let mut jar = CookieJar::new();
        let big = "v".repeat(4097);
        jar.set_cookie(&format!("a={big}"), &ctx("h.test", "/"), NOW);
        assert!(jar.is_empty());
        let ok = "v".repeat(4095); // 1 + 4095 = 4096 bytes: allowed
        jar.set_cookie(&format!("a={ok}"), &ctx("h.test", "/"), NOW);
        assert_eq!(jar.len(), 1);
    }

    #[test]
    fn parse_edge_cases() {
        let mut jar = CookieJar::new();
        jar.set_cookie("", &ctx("h.test", "/"), NOW);
        jar.set_cookie("=b", &ctx("h.test", "/"), NOW);
        jar.set_cookie(";", &ctx("h.test", "/"), NOW);
        assert!(jar.is_empty());

        jar.set_cookie("a", &ctx("h.test", "/"), NOW);
        assert_eq!(get(&mut jar, "h.test", "/"), Some("a=".into()));

        jar.set_cookie("b=x=y", &ctx("h.test", "/"), NOW);
        jar.set_cookie("c=\"quoted value\"", &ctx("h.test", "/"), NOW);
        jar.set_cookie("d=1; Unknown-Attr=zzz; Another", &ctx("h.test", "/"), NOW);
        jar.set_cookie(
            "e=2; SECURE; HTTPONLY; SAMESITE=STRICT",
            &ctx("h.test", "/"),
            NOW,
        );
        let h = get(&mut jar, "h.test", "/").unwrap();
        assert!(h.contains("b=x=y"), "{h}");
        assert!(h.contains("c=quoted value"), "{h}");
        assert!(h.contains("d=1"), "{h}");
        assert!(h.contains("e=2"), "{h}");
    }

    #[test]
    fn control_bytes_in_value_rejected() {
        let mut jar = CookieJar::new();
        // CRLF header-injection attempt.
        jar.set_cookie("a=b\r\nX-Injected: 1", &ctx("h.test", "/"), NOW);
        assert!(jar.is_empty());
        // C0 control bytes.
        jar.set_cookie("a=\u{1}\u{2}", &ctx("h.test", "/"), NOW);
        assert!(jar.is_empty());
        // DEL (0x7F).
        jar.set_cookie("a=\u{7f}", &ctx("h.test", "/"), NOW);
        assert!(jar.is_empty());
        // Interior TAB is a CTL too (RFC 6265 cookie-octet excludes CTLs).
        jar.set_cookie("a=b\tc", &ctx("h.test", "/"), NOW);
        assert!(jar.is_empty());
    }

    #[test]
    fn printable_value_chars_still_accepted() {
        let mut jar = CookieJar::new();
        // SP inside a value is fine (0x20 is not a CTL).
        jar.set_cookie("a=b c", &ctx("h.test", "/"), NOW);
        // Leading/trailing SP/TAB are trimmed, not rejected.
        jar.set_cookie("b=\tquoted\t", &ctx("h.test", "/"), NOW);
        // DQUOTE-strip quirk keeps working: surrounding DQUOTEs are removed.
        jar.set_cookie("c=\"quoted value\"", &ctx("h.test", "/"), NOW);
        jar.set_cookie("d=\"bare\"", &ctx("h.test", "/"), NOW);
        let h = get(&mut jar, "h.test", "/").unwrap();
        assert!(h.contains("a=b c"), "{h}");
        assert!(h.contains("b=quoted"), "{h}");
        assert!(h.contains("c=quoted value"), "{h}");
        assert!(h.contains("d=bare"), "{h}");
    }

    #[test]
    fn localhost_and_ip() {
        let mut jar = CookieJar::new();
        // localhost is exempt from the public-suffix block.
        jar.set_cookie("a=1", &ctx("localhost", "/"), NOW);
        assert_eq!(get(&mut jar, "localhost", "/"), Some("a=1".into()));
        jar.set_cookie("b=2; Domain=localhost", &ctx("localhost", "/"), NOW);
        assert_eq!(get(&mut jar, "localhost", "/"), Some("a=1; b=2".into()));

        // IP literals: host-only cookies work, Domain attributes are rejected.
        jar.set_cookie("c=3", &ctx("127.0.0.1", "/"), NOW);
        jar.set_cookie("d=4; Domain=127.0.0.1", &ctx("127.0.0.1", "/"), NOW);
        assert_eq!(get(&mut jar, "127.0.0.1", "/"), Some("c=3".into()));
        // No subdomain leakage for IPs.
        assert_eq!(get(&mut jar, "1.0.0.127", "/"), None);
    }

    #[test]
    fn remove_expired() {
        let mut jar = CookieJar::new();
        jar.set_cookie("a=1; Max-Age=10", &ctx("h.test", "/"), NOW);
        jar.set_cookie("b=2", &ctx("h.test", "/"), NOW);
        assert_eq!(jar.remove_expired(NOW + 5), 0);
        assert_eq!(jar.remove_expired(NOW + 11), 1);
        assert_eq!(jar.len(), 1);
    }

    #[test]
    fn clear() {
        let mut jar = CookieJar::new();
        jar.set_cookie("a=1", &ctx("h.test", "/"), NOW);
        jar.clear();
        assert!(jar.is_empty());
        assert_eq!(get(&mut jar, "h.test", "/"), None);
    }

    #[test]
    fn cookie_date_vectors() {
        assert_eq!(
            parse_cookie_date("Sun, 06 Nov 1994 08:49:37 GMT"),
            Some(784_111_777)
        );
        assert_eq!(
            parse_cookie_date("Sunday, 06-Nov-94 08:49:37 GMT"),
            Some(784_111_777)
        );
        assert_eq!(
            parse_cookie_date("Sun Nov  6 08:49:37 1994"),
            Some(784_111_777)
        );
        assert_eq!(parse_cookie_date("Thu, 01 Jan 1970 00:00:00 GMT"), Some(0));
        assert_eq!(parse_cookie_date("Thu, 01-Jan-1970 00:00:10 GMT"), Some(10));
        assert_eq!(
            parse_cookie_date("Wed, 21 Oct 2015 07:28:00 GMT"),
            Some(1_445_412_480)
        );
        assert_eq!(
            parse_cookie_date("Tue, 19 Jan 2038 03:14:07 GMT"),
            Some(2_147_483_647)
        );
        assert_eq!(
            parse_cookie_date("Sat, 29 Feb 2020 00:00:00 GMT"),
            Some(1_582_934_400)
        );
        // Two-digit years.
        assert_eq!(parse_cookie_date("Thu, 01 Jan 70 00:00:00 GMT"), Some(0));
        assert_eq!(
            parse_cookie_date("Thu, 01 Jan 69 00:00:00 GMT"),
            parse_cookie_date("Thu, 01 Jan 2069 00:00:00 GMT")
        );
        // Invalid dates.
        assert_eq!(parse_cookie_date(""), None);
        assert_eq!(parse_cookie_date("not a date"), None);
        assert_eq!(parse_cookie_date("32 Jan 2020 00:00:00 GMT"), None);
        assert_eq!(parse_cookie_date("Thu, 01 Jan 1970 25:00:00 GMT"), None);
        assert_eq!(parse_cookie_date("Mon, 29 Feb 2021 00:00:00 GMT"), None);
        assert_eq!(parse_cookie_date("Thu, 01 Foo 2020 00:00:00 GMT"), None);
    }

    #[test]
    fn max_age_vectors() {
        assert_eq!(parse_max_age("3600"), Some(3600));
        assert_eq!(parse_max_age("0"), Some(0));
        assert_eq!(parse_max_age("-1"), Some(-1));
        assert_eq!(parse_max_age("  42  "), Some(42));
        assert_eq!(parse_max_age(""), None);
        assert_eq!(parse_max_age("abc"), None);
        assert_eq!(parse_max_age("12x"), None);
        assert_eq!(parse_max_age("+12"), None);
        assert_eq!(parse_max_age("--5"), None);
        // Saturates instead of overflowing.
        assert_eq!(parse_max_age("99999999999999999999999"), Some(i64::MAX));
    }

    #[test]
    fn ip_detection() {
        assert!(is_ip_literal("127.0.0.1"));
        assert!(is_ip_literal("1.2.3.4"));
        assert!(is_ip_literal("::1"));
        assert!(is_ip_literal("[::1]"));
        assert!(is_ip_literal("2001:db8::1"));
        // WHATWG URL Standard IPv4 notations: hex/octal parts, 1–3 part
        // forms, and single integers.
        assert!(is_ip_literal("0x7f.0.0.1"));
        assert!(is_ip_literal("0177.0.0.1"));
        assert!(is_ip_literal("2130706433"));
        assert!(is_ip_literal("1.2.3"));
        // Octets must fit in a byte.
        assert!(!is_ip_literal("999.999.999.999"));
        assert!(!is_ip_literal("0x100.0.0.1"));
        assert!(!is_ip_literal("1.2.3.4.5"));
        assert!(!is_ip_literal("example.com"));
        assert!(!is_ip_literal("localhost"));
        // A port suffix is not part of the host: strip it before checking.
        assert!(!is_ip_literal(strip_port("example.com:80")));
        assert!(is_ip_literal(strip_port("1.2.3.4:8080")));
        assert!(is_ip_literal(strip_port("[::1]:80")));
    }

    #[test]
    fn strip_port_cases() {
        assert_eq!(strip_port("example.com:80"), "example.com");
        assert_eq!(strip_port("example.com"), "example.com");
        assert_eq!(strip_port("[::1]:80"), "[::1]");
        assert_eq!(strip_port("[::1]"), "[::1]");
        // A bare (unbracketed) IPv6 literal has no port to strip.
        assert_eq!(strip_port("::1"), "::1");
        assert_eq!(strip_port("1.2.3.4:8080"), "1.2.3.4");
    }

    #[test]
    fn samesite_first_wins() {
        let mut jar = CookieJar::new();
        // The first SameSite attribute wins; the later `None` is ignored.
        jar.set_cookie(
            "a=1; SameSite=Strict; SameSite=None; Secure",
            &ctx("h.test", "/"),
            NOW,
        );
        assert_eq!(jar.len(), 1);
        // Stored as Strict: not sent on a cross-site top-level navigation,
        // even though a last-wins parse would have made it `None`.
        let nav = RequestContext {
            host: "h.test",
            path: "/",
            secure: true,
            http_api: true,
            top_host: Some("other.com"),
            top_level_safe_navigation: true,
        };
        assert_eq!(jar.cookie_header(&nav, NOW), None);
        // Same-site is fine.
        let same = RequestContext {
            top_host: Some("h.test"),
            ..nav
        };
        assert_eq!(jar.cookie_header(&same, NOW), Some("a=1".into()));
    }

    #[test]
    fn lax_navigation_not_coupled_to_http_api() {
        let mut jar = CookieJar::new();
        jar.set_cookie("l=1; SameSite=Lax", &ctx("id.example.com", "/"), NOW);
        // A `document.cookie` (non-HTTP API) read during a cross-site
        // top-level safe navigation: Lax rides along regardless of
        // `http_api`.
        let nav = RequestContext {
            host: "id.example.com",
            path: "/",
            secure: true,
            http_api: false,
            top_host: Some("other.com"),
            top_level_safe_navigation: true,
        };
        assert_eq!(jar.cookie_header(&nav, NOW), Some("l=1".into()));
    }

    #[test]
    fn host_normalization() {
        let mut jar = CookieJar::new();
        jar.set_cookie("a=1", &ctx("WWW.Example.COM.", "/"), NOW);
        // Stored under the normalized host; retrievable via the bare form.
        assert_eq!(get(&mut jar, "www.example.com", "/"), Some("a=1".into()));
        assert_eq!(get(&mut jar, "WWW.EXAMPLE.COM.", "/"), Some("a=1".into()));
        // A `:port` suffix is stripped before matching, in both directions.
        jar.set_cookie("b=2", &ctx("www.other.com:8443", "/"), NOW);
        assert_eq!(get(&mut jar, "www.other.com", "/"), Some("b=2".into()));
        assert_eq!(get(&mut jar, "www.other.com:8443", "/"), Some("b=2".into()));
    }

    #[test]
    fn global_eviction_is_lru() {
        let mut jar = CookieJar::new();
        for i in 0..=MAX_COOKIES {
            let host = format!("h{i}.test");
            jar.set_cookie(&format!("c{i}=v"), &ctx(&host, "/"), NOW);
        }
        assert_eq!(jar.len(), MAX_COOKIES);
        assert!(
            !jar.cookies.iter().any(|c| c.name == "c0"),
            "the oldest cookie is evicted first"
        );
        assert!(
            jar.cookies
                .iter()
                .any(|c| c.name == format!("c{MAX_COOKIES}"))
        );
    }

    #[test]
    fn expiry_boundary_is_strict() {
        let mut jar = CookieJar::new();
        jar.set_cookie("a=1; Max-Age=10", &ctx("h.test", "/"), NOW);
        assert!(jar.cookie_header(&ctx("h.test", "/"), NOW + 9).is_some());
        // `expires_at == now` counts as expired (strict `>`), and the read
        // purges the cookie.
        assert!(jar.cookie_header(&ctx("h.test", "/"), NOW + 10).is_none());
        assert!(jar.is_empty());
    }
}
