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
//!   rejected; a `Secure` cookie received over an insecure transport is
//!   ignored (RFC 6265bis §5.3); `__Secure-` / `__Host-` name prefixes are
//!   enforced (case-sensitive, per RFC 6265bis §4.1.3).
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
    /// Request host, e.g. `"www.example.com"`. Plain string; normalized internally.
    pub host: &'a str,
    /// Request path, e.g. `"/a/b"`. Empty is treated as `"/"`.
    pub path: &'a str,
    /// Whether the request uses a secure transport (https/wss).
    pub secure: bool,
    /// True when the call comes from the network stack rather than
    /// `document.cookie`. `HttpOnly` cookies are invisible to non-HTTP APIs,
    /// and non-HTTP APIs cannot overwrite them.
    pub http_api: bool,
    /// Host of the top-level document, for `SameSite` decisions.
    /// `None` (unknown) is treated as cross-site.
    pub top_host: Option<&'a str>,
    /// Whether the top-level document was loaded over a secure
    /// transport, for schemeful same-site decisions (`http` vs `https`
    /// to the same host are cross-site). `None` (unknown) is treated
    /// as cross-site.
    pub top_secure: Option<bool>,
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
    persistent: bool,
    /// Monotonic insertion order: earlier sorts first in `Cookie` headers.
    creation_seq: u64,
    /// Monotonic access order for LRU eviction.
    last_access: u64,
}

/// The cookie jar.
#[derive(Debug, Default, Clone)]
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
    /// `ctx.host` for path `ctx.path`, at time `now` (epoch seconds).
    /// Invalid or rejected cookies are silently ignored, per spec.
    pub fn set_cookie(&mut self, header: &str, ctx: &RequestContext<'_>, now: u64) {
        let host = psl::normalize_host(ctx.host);
        if host.is_empty() {
            return;
        }
        let host_is_ip = is_ip_literal(&host);

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
        let mut max_age: Option<i64> = None;
        let mut expires: Option<u64> = None;

        // Attributes are parsed for both the network stack and
        // `document.cookie` (RFC 6265 §5.3); only HttpOnly is special:
        // a script cannot make a cookie HttpOnly, and cannot overwrite
        // one (the clobber check below enforces the latter).
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
                "httponly" => {
                    if ctx.http_api {
                        http_only = true;
                    }
                }
                "samesite" => {
                    same_site = parse_samesite(attr_value);
                }
                _ => {} // unknown attributes are ignored
            }
        }

        // RFC 6265bis §5.3 step 5: a Secure cookie received over an
        // insecure transport is ignored, not stored.
        if secure && !ctx.secure {
            return;
        }
        // RFC 6265bis §5.6: SameSite=None requires Secure.
        if same_site == SameSite::None && !secure {
            return;
        }
        // RFC 6265bis §4.1.3: name prefixes (case-sensitive).
        if name.starts_with("__Secure-") && !secure {
            return;
        }
        if name.starts_with("__Host-")
            && !(secure && path_attr.as_deref() == Some("/") && domain_attr.is_none())
        {
            return;
        }

        // --- effective domain (RFC 6265 §5.3 steps 5-7) ---
        let (domain, host_only) = match domain_attr {
            Some(d) => {
                if host_is_ip {
                    return; // no Domain cookies on IP literals
                }
                if !domain_match(&host, &d) {
                    return;
                }
                if psl::is_public_suffix(&d) && !is_localhost(&d) {
                    return; // no cookies on public suffixes
                }
                (d, false)
            }
            None => {
                if psl::is_public_suffix(&host) && !host_is_ip && !is_localhost(&host) {
                    return; // e.g. a bare public suffix cannot set cookies
                }
                (host.into_owned(), true)
            }
        };

        // --- expiry: Max-Age wins over Expires (RFC 6265 §5.3) ---
        let (expires_at, persistent) = match (max_age, expires) {
            (Some(delta), _) => {
                let at = if delta <= 0 {
                    0
                } else {
                    now.saturating_add(delta as u64)
                };
                (at, true)
            }
            (None, Some(t)) => (t, true),
            (None, None) => (u64::MAX, false),
        };

        // --- path (RFC 6265 §5.3 step 10) ---
        let path = match path_attr {
            Some(p) if p.starts_with('/') => p,
            _ => default_path(ctx.path),
        };

        if name.len() + value.len() > MAX_COOKIE_SIZE {
            return;
        }

        // RFC 6265bis: a Secure cookie may only be overwritten by a
        // response received over a secure transport. An insecure
        // response must neither replace the cookie's value nor delete
        // it via an already-expired cookie — both go through this one
        // guard, placed before the expiry-deletion below.
        if !ctx.secure
            && let Some(old) = self
                .cookies
                .iter()
                .find(|c| c.name == name && c.domain == domain && c.path == path)
            && old.secure
        {
            return;
        }

        // Non-HTTP APIs cannot clobber HttpOnly cookies (RFC 6265 §5.3):
        // the check runs before the expiry-deletion below, so a script
        // can neither overwrite nor remove an HttpOnly cookie — an
        // already-expired cookie from `document.cookie` is ignored
        // entirely, exactly like a live one.
        if !ctx.http_api
            && let Some(old) = self
                .cookies
                .iter()
                .find(|c| c.name == name && c.domain == domain && c.path == path)
            && old.http_only
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
            old.persistent = persistent;
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
            persistent,
            creation_seq: seq,
            last_access: seq,
        });
    }

    /// Build the value for an outgoing `Cookie` header for `ctx`, at time
    /// `now`. Returns `None` when no stored cookie applies. Expired cookies
    /// are purged. Matches are sorted by longest path first, then earliest
    /// creation time (RFC 6265 §5.4).
    pub fn cookie_header(&mut self, ctx: &RequestContext<'_>, now: u64) -> Option<String> {
        let host = psl::normalize_host(ctx.host);
        if host.is_empty() {
            return None;
        }
        let top: Option<Cow<'_, str>> = ctx.top_host.map(psl::normalize_host);
        let req_path = if ctx.path.is_empty() { "/" } else { ctx.path };

        self.cookies.retain(|c| c.expires_at > now);

        let mut hits: Vec<usize> = Vec::new();
        for (i, c) in self.cookies.iter().enumerate() {
            if c.host_only {
                if c.domain != host {
                    continue;
                }
            } else if !domain_match(&host, &c.domain) {
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
            if !samesite_allows(c, &host, top.as_deref(), ctx) {
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

/// Lenient IP-literal check: dotted-decimal IPv4 or anything with a colon
/// (IPv6, optionally bracketed).
fn is_ip_literal(host: &str) -> bool {
    let h = host
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(host);
    if h.contains(':') {
        return true;
    }
    let mut parts = h.split('.');
    for _ in 0..4 {
        match parts.next() {
            Some(p) if !p.is_empty() && p.len() <= 3 && p.bytes().all(|b| b.is_ascii_digit()) => {}
            _ => return false,
        }
    }
    parts.next().is_none()
}

fn is_localhost(host: &str) -> bool {
    host == "localhost"
}

/// Same-site per scheme + registrable domain (schemeful same-site):
/// `http://example.com` and `https://example.com` are cross-site.
/// Unknown or IP top contexts are conservative. Hosts must be normalized.
fn is_same_site(host: &str, host_secure: bool, top: &str, top_secure: bool) -> bool {
    if host_secure != top_secure {
        return false;
    }
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
    let same_site = match (top_host, ctx.top_secure) {
        (Some(t), Some(ts)) => is_same_site(host, ctx.secure, t, ts),
        _ => false,
    };
    match cookie.same_site {
        SameSite::None => true,
        SameSite::Strict => same_site,
        SameSite::Lax => same_site || (ctx.top_level_safe_navigation && ctx.http_api),
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
            top_secure: Some(true),
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
        // Scheme-consistent http context: an http request under an http
        // top document (schemeful same-site treats a mixed http-request /
        // https-top pair as cross-site, which would withhold Lax `p=2`
        // for the wrong reason).
        let http = RequestContext {
            secure: false,
            top_secure: Some(false),
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
    fn secure_cookie_rejected_over_http() {
        // RFC 6265bis §5.3: a Secure cookie received over an insecure
        // transport is ignored, not stored.
        let mut jar = CookieJar::new();
        let http = RequestContext {
            secure: false,
            ..ctx("www.example.com", "/")
        };
        jar.set_cookie("s=1; Secure", &http, NOW);
        assert!(jar.is_empty());
        // A plain cookie over http is still stored.
        jar.set_cookie("p=2", &http, NOW);
        assert_eq!(jar.len(), 1);
        // And a Secure cookie over https is stored.
        jar.set_cookie("s=1; Secure", &ctx("www.example.com", "/"), NOW);
        assert_eq!(jar.len(), 2);
        // SameSite=None implies Secure, so it is rejected over http too.
        jar.set_cookie("n=3; SameSite=None; Secure", &http, NOW);
        assert_eq!(jar.len(), 2);
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
    fn document_cookie_cannot_delete_httponly_cookie() {
        // Regression: the HttpOnly clobber guard must run before the
        // expiry-deletion branch, or `document.cookie` could remove an
        // HttpOnly cookie by setting an already-expired one. Browsers
        // ignore such writes entirely (RFC 6265 §5.3).
        let mut jar = CookieJar::new();
        let https = ctx("www.example.com", "/");
        let script = RequestContext {
            http_api: false,
            ..https
        };
        jar.set_cookie("h=1; HttpOnly", &https, NOW);
        assert_eq!(jar.cookie_header(&https, NOW), Some("h=1".into()));
        // Expired deletion via document.cookie: ignored.
        jar.set_cookie("h=gone; Max-Age=0", &script, NOW);
        assert_eq!(jar.len(), 1);
        assert_eq!(jar.cookie_header(&https, NOW), Some("h=1".into()));
        // Same via an Expires date in the past: also ignored.
        jar.set_cookie(
            "h=gone; Expires=Thu, 01 Jan 1970 00:00:00 GMT",
            &script,
            NOW,
        );
        assert_eq!(jar.len(), 1);
        assert_eq!(jar.cookie_header(&https, NOW), Some("h=1".into()));
    }

    #[test]
    fn network_stack_can_delete_httponly_cookie() {
        // The guard only blocks non-HTTP APIs: a Set-Cookie from the
        // network stack still deletes an HttpOnly cookie.
        let mut jar = CookieJar::new();
        let https = ctx("www.example.com", "/");
        jar.set_cookie("h=1; HttpOnly", &https, NOW);
        jar.set_cookie("h=gone; Max-Age=0", &https, NOW);
        assert!(jar.is_empty());
    }

    #[test]
    fn document_cookie_honors_attributes_except_httponly() {
        let mut jar = CookieJar::new();
        let script = RequestContext {
            http_api: false,
            ..ctx("www.example.com", "/")
        };
        // Expiry and Path are honored from document.cookie.
        jar.set_cookie("d=1; Max-Age=3600; Path=/sub", &script, NOW);
        let sub = RequestContext {
            http_api: false,
            ..ctx("www.example.com", "/sub/x")
        };
        assert_eq!(jar.cookie_header(&sub, NOW), Some("d=1".into()));
        assert_eq!(jar.cookie_header(&sub, NOW + 3600), None); // expired
        let root = RequestContext {
            http_api: false,
            ..ctx("www.example.com", "/")
        };
        assert_eq!(jar.cookie_header(&root, NOW), None); // Path mismatch

        // A matching Domain attribute is honored (not clobbered).
        jar.set_cookie("w=2; Domain=example.com", &script, NOW);
        let script_sub = RequestContext {
            http_api: false,
            ..ctx("sub.example.com", "/")
        };
        assert_eq!(jar.cookie_header(&script_sub, NOW), Some("w=2".into()));
        // A non-matching Domain is rejected outright.
        jar.set_cookie("e=3; Domain=evil.com", &script, NOW);
        assert_eq!(jar.cookies.iter().filter(|c| c.name == "e").count(), 0);

        // HttpOnly is ignored: the cookie is stored and visible to scripts.
        jar.set_cookie("h=4; HttpOnly", &script, NOW);
        assert_eq!(jar.cookie_header(&script, NOW), Some("w=2; h=4".into()));
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
            top_secure: Some(true),
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
            top_secure: Some(true),
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
            top_secure: Some(true),
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
    fn schemeful_same_site() {
        let mut jar = CookieJar::new();
        jar.set_cookie("s=1; SameSite=Strict", &ctx("example.com", "/"), NOW);
        // https top -> https request: same-site, sent.
        assert_eq!(
            jar.cookie_header(&ctx("example.com", "/"), NOW),
            Some("s=1".into())
        );
        // http top -> https request: cross-site by scheme, withheld.
        let http_top = RequestContext {
            top_secure: Some(false),
            ..ctx("example.com", "/")
        };
        assert_eq!(jar.cookie_header(&http_top, NOW), None);
        // Unknown top scheme is conservative: withheld.
        let unknown_top = RequestContext {
            top_secure: None,
            ..ctx("example.com", "/")
        };
        assert_eq!(jar.cookie_header(&unknown_top, NOW), None);
    }

    #[test]
    fn insecure_response_cannot_clobber_secure_cookie() {
        let mut jar = CookieJar::new();
        let https = ctx("example.com", "/");
        let http = RequestContext {
            secure: false,
            ..ctx("example.com", "/")
        };
        // Stored over https with the Secure flag.
        jar.set_cookie("s=1; Secure", &https, NOW);
        assert_eq!(jar.cookie_header(&https, NOW), Some("s=1".into()));
        // An insecure response must not overwrite it...
        jar.set_cookie("s=2", &http, NOW);
        assert_eq!(jar.cookie_header(&https, NOW), Some("s=1".into()));
        // ...nor delete it via an already-expired cookie.
        jar.set_cookie("s=gone; Max-Age=0", &http, NOW);
        assert_eq!(jar.cookie_header(&https, NOW), Some("s=1".into()));
        // A non-Secure cookie is still deletable over http (the guard
        // is targeted, not a blanket block on insecure writes).
        jar.set_cookie("p=1", &https, NOW);
        jar.set_cookie("p=gone; Max-Age=0", &http, NOW);
        assert_eq!(jar.cookie_header(&https, NOW), Some("s=1".into()));
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
        assert!(!is_ip_literal("example.com"));
        assert!(!is_ip_literal("localhost"));
        assert!(!is_ip_literal("1.2.3"));
        assert!(!is_ip_literal("1.2.3.4.5"));
    }
}
