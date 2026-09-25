//! Vendored browser fingerprint profiles.
//!
//! Data transcribed from the profile constants in
//! [wreq-util](https://github.com/0x676e67/wreq-util) (`src/emulate/profile/`,
//! Apache-2.0; see [`ATTRIBUTION.md`](ATTRIBUTION.md)). Only *data* was taken:
//! ordered TLS cipher/curve/signature-scheme lists, extension-behavior flags,
//! HTTP/2 settings values and ordering, default header sets, and User-Agent
//! strings. No wreq code is vendored. Applying a profile (ClientHello
//! byte layout, GREASE, permutation, HTTP/2 wire order) is bowser_net's own
//! TLS/HTTP implementation's job.
//!
//! String storage: the UA and `sec-ch-ua` strings are numerous and highly
//! redundant, so they live prefix-compressed in [`data::UA_POOL`]; use
//! [`Profile::user_agent`] / [`Profile::sec_ch_ua`] to decode them. The
//! shorter TLS list strings are plain `&'static str`.

mod data;

#[cfg(test)]
mod tests;

/// Client platform a profile emulates, in canonical slot order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Platform {
    Windows = 0,
    MacOS = 1,
    Linux = 2,
    Android = 3,
    Ios = 4,
}

impl Platform {
    /// All platforms, in canonical slot order.
    pub const ALL: [Platform; 5] = [
        Platform::Windows,
        Platform::MacOS,
        Platform::Linux,
        Platform::Android,
        Platform::Ios,
    ];

    /// Value for the `sec-ch-ua-platform` header.
    pub fn sec_ch_ua_platform(self) -> &'static str {
        match self {
            Platform::MacOS => "\"macOS\"",
            Platform::Linux => "\"Linux\"",
            Platform::Windows => "\"Windows\"",
            Platform::Android => "\"Android\"",
            Platform::Ios => "\"iOS\"",
        }
    }

    /// Value for the `sec-ch-ua-mobile` header.
    pub fn sec_ch_ua_mobile(self) -> &'static str {
        match self {
            Platform::Android | Platform::Ios => "?1",
            _ => "?0",
        }
    }
}

/// TLS version bound used by a profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TlsVersion {
    Tls10,
    Tls11,
    Tls12,
    Tls13,
}

/// TLS fingerprint of one profile: ordered lists plus behavior flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TlsConfig {
    /// `:`-joined cipher suite names, ClientHello order.
    pub ciphers: &'static str,
    /// `:`-joined supported-group names, ClientHello order.
    pub curves: &'static str,
    /// `:`-joined signature-scheme names, ClientHello order.
    pub sigalgs: &'static str,
    /// ALPN protocols to offer, in order.
    pub alpn: &'static [&'static str],
    /// Insert GREASE values.
    pub grease: bool,
    /// Permute ClientHello extensions (BoringSSL order).
    pub permute_extensions: bool,
    /// Send an ECH GREASE extension.
    pub ech_grease: bool,
    /// Offer pre-shared keys.
    pub pre_shared_key: bool,
    /// Skip session tickets when offering PSK.
    pub psk_skip_session_tickets: bool,
    /// Allow (EC)DHE key exchange with PSK.
    pub psk_dhe_ke: bool,
    /// Offer a session-ticket extension.
    pub session_ticket: bool,
    /// Request OCSP stapling.
    pub ocsp_stapling: bool,
    /// Request signed certificate timestamps.
    pub signed_cert_timestamps: bool,
    /// Use the new ALPS extension codepoint.
    pub alps_new_codepoint: bool,
    /// Insert GREASE into the signature-schemes list.
    pub sigalg_grease: bool,
    /// Advertise Chromium trust-anchor IDs.
    pub trust_anchors: bool,
    /// ALPS application protocol, if any (e.g. `"h2"`).
    pub alps: &'static str,
    /// `,`-joined certificate compressors to offer (e.g. `"brotli"`).
    pub cert_compressors: &'static str,
    /// Minimum TLS version.
    pub min_tls: TlsVersion,
    /// Maximum TLS version.
    pub max_tls: TlsVersion,
    /// Key shares to send in the key_share extension, in order.
    pub key_shares: &'static [&'static str],
    /// `:`-joined delegated-credential signature schemes.
    pub delegated_credentials: &'static str,
    /// record_size_limit extension value; 0 = absent.
    pub record_size_limit: u16,
    /// Numeric TLS extension codes giving the explicit extension order
    /// (Firefox-style); empty when extensions are permuted or default.
    pub extension_order: &'static [u16],
}

/// HTTP/2 stream dependency of a profile's header stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamDependency {
    pub parent: u32,
    pub weight: u8,
    pub exclusive: bool,
}

/// One HTTP/2 priority-tree entry (Firefox profiles).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Priority {
    pub stream: u32,
    pub parent: u32,
    pub weight: u8,
    pub exclusive: bool,
}

/// HTTP/2 fingerprint of one profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Http2Config {
    pub initial_window_size: Option<u32>,
    pub initial_connection_window_size: Option<u32>,
    pub max_header_list_size: Option<u32>,
    pub header_table_size: Option<u32>,
    pub max_concurrent_streams: Option<u32>,
    pub max_frame_size: Option<u32>,
    pub enable_push: Option<bool>,
    pub enable_connect_protocol: bool,
    pub no_rfc7540_priorities: bool,
    pub initial_stream_id: Option<u32>,
    pub stream_dependency: Option<StreamDependency>,
    pub priorities: &'static [Priority],
    /// SETTINGS parameter names in the order the SETTINGS frame sends them.
    pub settings_order: &'static [&'static str],
    /// Pseudo-header names in wire order.
    pub pseudo_order: &'static [&'static str],
}

/// Which default header set a profile sends, in wire order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeaderSet {
    Chrome,
    ChromeZstd,
    ChromeZstdPriority,
    Chrome152,
    Firefox,
    FirefoxZstd,
    Safari15,
    Safari1617,
    Safari18,
    OkHttp,
}

/// One header of a [`HeaderSet`] template.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeaderField {
    pub name: &'static str,
    pub value: HeaderValue,
}

/// A header value: fixed, or resolved per profile/platform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeaderValue {
    Static(&'static str),
    UserAgent,
    SecChUa,
    SecChUaMobile,
    SecChUaPlatform,
}

/// Ordered header template for a [`HeaderSet`].
///
/// `accept-encoding` reflects the compression-enabled build of the source
/// data; bowser_net implements compression, so it always applies.
pub fn header_template(set: HeaderSet) -> &'static [HeaderField] {
    use HeaderSet as H;
    use HeaderValue as V;
    const CHROME_ACCEPT: &str = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7";
    const FF_ACCEPT: &str = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8";
    const ENC: &str = "gzip, deflate, br";
    const ENC_ZSTD: &str = "gzip, deflate, br, zstd";
    match set {
        H::Chrome => &[
            HeaderField {
                name: "sec-ch-ua",
                value: V::SecChUa,
            },
            HeaderField {
                name: "sec-ch-ua-mobile",
                value: V::SecChUaMobile,
            },
            HeaderField {
                name: "sec-ch-ua-platform",
                value: V::SecChUaPlatform,
            },
            HeaderField {
                name: "upgrade-insecure-requests",
                value: V::Static("1"),
            },
            HeaderField {
                name: "user-agent",
                value: V::UserAgent,
            },
            HeaderField {
                name: "accept",
                value: V::Static(CHROME_ACCEPT),
            },
            HeaderField {
                name: "sec-fetch-site",
                value: V::Static("none"),
            },
            HeaderField {
                name: "sec-fetch-mode",
                value: V::Static("navigate"),
            },
            HeaderField {
                name: "sec-fetch-user",
                value: V::Static("?1"),
            },
            HeaderField {
                name: "sec-fetch-dest",
                value: V::Static("document"),
            },
            HeaderField {
                name: "accept-encoding",
                value: V::Static(ENC),
            },
            HeaderField {
                name: "accept-language",
                value: V::Static("en-US,en;q=0.9"),
            },
        ],
        H::ChromeZstd => &[
            HeaderField {
                name: "sec-ch-ua",
                value: V::SecChUa,
            },
            HeaderField {
                name: "sec-ch-ua-mobile",
                value: V::SecChUaMobile,
            },
            HeaderField {
                name: "sec-ch-ua-platform",
                value: V::SecChUaPlatform,
            },
            HeaderField {
                name: "upgrade-insecure-requests",
                value: V::Static("1"),
            },
            HeaderField {
                name: "user-agent",
                value: V::UserAgent,
            },
            HeaderField {
                name: "accept",
                value: V::Static(CHROME_ACCEPT),
            },
            HeaderField {
                name: "sec-fetch-site",
                value: V::Static("none"),
            },
            HeaderField {
                name: "sec-fetch-mode",
                value: V::Static("navigate"),
            },
            HeaderField {
                name: "sec-fetch-user",
                value: V::Static("?1"),
            },
            HeaderField {
                name: "sec-fetch-dest",
                value: V::Static("document"),
            },
            HeaderField {
                name: "accept-encoding",
                value: V::Static(ENC_ZSTD),
            },
            HeaderField {
                name: "accept-language",
                value: V::Static("en-US,en;q=0.9"),
            },
        ],
        H::ChromeZstdPriority => &[
            HeaderField {
                name: "sec-ch-ua",
                value: V::SecChUa,
            },
            HeaderField {
                name: "sec-ch-ua-mobile",
                value: V::SecChUaMobile,
            },
            HeaderField {
                name: "sec-ch-ua-platform",
                value: V::SecChUaPlatform,
            },
            HeaderField {
                name: "upgrade-insecure-requests",
                value: V::Static("1"),
            },
            HeaderField {
                name: "user-agent",
                value: V::UserAgent,
            },
            HeaderField {
                name: "accept",
                value: V::Static(CHROME_ACCEPT),
            },
            HeaderField {
                name: "sec-fetch-site",
                value: V::Static("none"),
            },
            HeaderField {
                name: "sec-fetch-mode",
                value: V::Static("navigate"),
            },
            HeaderField {
                name: "sec-fetch-user",
                value: V::Static("?1"),
            },
            HeaderField {
                name: "sec-fetch-dest",
                value: V::Static("document"),
            },
            HeaderField {
                name: "accept-encoding",
                value: V::Static(ENC_ZSTD),
            },
            HeaderField {
                name: "priority",
                value: V::Static("u=0, i"),
            },
        ],
        H::Chrome152 => &[
            HeaderField {
                name: "sec-ch-ua",
                value: V::SecChUa,
            },
            HeaderField {
                name: "sec-ch-ua-mobile",
                value: V::SecChUaMobile,
            },
            HeaderField {
                name: "sec-ch-ua-platform",
                value: V::SecChUaPlatform,
            },
            HeaderField {
                name: "accept-language",
                value: V::Static("en-US,en;q=0.9"),
            },
            HeaderField {
                name: "upgrade-insecure-requests",
                value: V::Static("1"),
            },
            HeaderField {
                name: "user-agent",
                value: V::UserAgent,
            },
            HeaderField {
                name: "accept",
                value: V::Static(CHROME_ACCEPT),
            },
            HeaderField {
                name: "sec-fetch-site",
                value: V::Static("none"),
            },
            HeaderField {
                name: "sec-fetch-mode",
                value: V::Static("navigate"),
            },
            HeaderField {
                name: "sec-fetch-user",
                value: V::Static("?1"),
            },
            HeaderField {
                name: "sec-fetch-dest",
                value: V::Static("document"),
            },
            HeaderField {
                name: "accept-encoding",
                value: V::Static(ENC_ZSTD),
            },
            HeaderField {
                name: "priority",
                value: V::Static("u=0, i"),
            },
        ],
        H::Firefox => &[
            HeaderField {
                name: "user-agent",
                value: V::UserAgent,
            },
            HeaderField {
                name: "accept",
                value: V::Static(FF_ACCEPT),
            },
            HeaderField {
                name: "accept-language",
                value: V::Static("en-US,en;q=0.5"),
            },
            HeaderField {
                name: "accept-encoding",
                value: V::Static(ENC),
            },
            HeaderField {
                name: "upgrade-insecure-requests",
                value: V::Static("1"),
            },
            HeaderField {
                name: "sec-fetch-dest",
                value: V::Static("document"),
            },
            HeaderField {
                name: "sec-fetch-mode",
                value: V::Static("navigate"),
            },
            HeaderField {
                name: "sec-fetch-site",
                value: V::Static("none"),
            },
            HeaderField {
                name: "sec-fetch-user",
                value: V::Static("?1"),
            },
            HeaderField {
                name: "te",
                value: V::Static("trailers"),
            },
        ],
        H::FirefoxZstd => &[
            HeaderField {
                name: "user-agent",
                value: V::UserAgent,
            },
            HeaderField {
                name: "accept",
                value: V::Static(FF_ACCEPT),
            },
            HeaderField {
                name: "accept-language",
                value: V::Static("en-US,en;q=0.5"),
            },
            HeaderField {
                name: "accept-encoding",
                value: V::Static(ENC_ZSTD),
            },
            HeaderField {
                name: "upgrade-insecure-requests",
                value: V::Static("1"),
            },
            HeaderField {
                name: "sec-fetch-dest",
                value: V::Static("document"),
            },
            HeaderField {
                name: "sec-fetch-mode",
                value: V::Static("navigate"),
            },
            HeaderField {
                name: "sec-fetch-site",
                value: V::Static("none"),
            },
            HeaderField {
                name: "sec-fetch-user",
                value: V::Static("?1"),
            },
            HeaderField {
                name: "priority",
                value: V::Static("u=0, i"),
            },
            HeaderField {
                name: "te",
                value: V::Static("trailers"),
            },
        ],
        H::Safari15 => &[
            HeaderField {
                name: "user-agent",
                value: V::UserAgent,
            },
            HeaderField {
                name: "accept",
                value: V::Static(FF_ACCEPT),
            },
            HeaderField {
                name: "accept-language",
                value: V::Static("en-US,en;q=0.9"),
            },
            HeaderField {
                name: "accept-encoding",
                value: V::Static(ENC),
            },
        ],
        H::Safari1617 => &[
            HeaderField {
                name: "accept",
                value: V::Static(FF_ACCEPT),
            },
            HeaderField {
                name: "sec-fetch-site",
                value: V::Static("none"),
            },
            HeaderField {
                name: "accept-encoding",
                value: V::Static(ENC),
            },
            HeaderField {
                name: "sec-fetch-mode",
                value: V::Static("navigate"),
            },
            HeaderField {
                name: "user-agent",
                value: V::UserAgent,
            },
            HeaderField {
                name: "accept-language",
                value: V::Static("en-US,en;q=0.9"),
            },
            HeaderField {
                name: "sec-fetch-dest",
                value: V::Static("document"),
            },
        ],
        H::Safari18 => &[
            HeaderField {
                name: "sec-fetch-dest",
                value: V::Static("document"),
            },
            HeaderField {
                name: "user-agent",
                value: V::UserAgent,
            },
            HeaderField {
                name: "accept",
                value: V::Static(FF_ACCEPT),
            },
            HeaderField {
                name: "sec-fetch-site",
                value: V::Static("none"),
            },
            HeaderField {
                name: "sec-fetch-mode",
                value: V::Static("navigate"),
            },
            HeaderField {
                name: "accept-language",
                value: V::Static("en-US,en;q=0.9"),
            },
            HeaderField {
                name: "priority",
                value: V::Static("u=0, i"),
            },
            HeaderField {
                name: "accept-encoding",
                value: V::Static(ENC),
            },
        ],
        H::OkHttp => &[
            HeaderField {
                name: "accept",
                value: V::Static("*/*"),
            },
            HeaderField {
                name: "accept-language",
                value: V::Static("en-US,en;q=0.9"),
            },
            HeaderField {
                name: "user-agent",
                value: V::UserAgent,
            },
            HeaderField {
                name: "accept-encoding",
                value: V::Static("gzip"),
            },
        ],
    }
}

/// One vendored fingerprint profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    /// Public profile name, e.g. `"chrome_131"`.
    pub name: &'static str,
    pub tls: &'static TlsConfig,
    pub http2: &'static Http2Config,
    pub header_set: HeaderSet,
    sec_ch_ua: Option<u16>,
    user_agents: [Option<u16>; 5],
}

impl Profile {
    /// `sec-ch-ua` header value for this profile, if it has one.
    pub fn sec_ch_ua(&self) -> Option<String> {
        self.sec_ch_ua.map(ua_pool_get)
    }

    /// User-Agent string for the given platform, if this profile defines one.
    pub fn user_agent(&self, platform: Platform) -> Option<String> {
        self.user_agents[platform as usize].map(ua_pool_get)
    }

    /// Resolve a [`HeaderField`] value for the given platform.
    ///
    /// Returns `None` when the field needs a per-profile string the profile
    /// does not define (e.g. `sec-ch-ua` on a Firefox profile).
    pub fn header_value(&self, platform: Platform, field: &HeaderField) -> Option<String> {
        match field.value {
            HeaderValue::Static(s) => Some(s.to_string()),
            HeaderValue::UserAgent => self.user_agent(platform),
            HeaderValue::SecChUa => self.sec_ch_ua(),
            HeaderValue::SecChUaMobile => Some(platform.sec_ch_ua_mobile().to_string()),
            HeaderValue::SecChUaPlatform => Some(platform.sec_ch_ua_platform().to_string()),
        }
    }
}

/// Decode one entry of the prefix-compressed [`data::UA_POOL`].
///
/// Entry `i` stores the byte length of the prefix it shares with entry
/// `i-1`'s full string, plus the remaining suffix; walking back to an entry
/// with length 0 and re-applying the prefixes reconstructs the string.
fn ua_pool_get(index: u16) -> String {
    let pool = data::UA_POOL;
    let mut chain: Vec<(u16, &str)> = Vec::new();
    let mut i = index as usize;
    loop {
        let (plen, suffix) = pool[i];
        chain.push((plen, suffix));
        if plen == 0 {
            break;
        }
        i -= 1;
    }
    let mut out = String::from(chain[chain.len() - 1].1);
    for (plen, suffix) in chain.iter().rev().skip(1) {
        out.truncate(*plen as usize);
        out.push_str(suffix);
    }
    out
}

/// Look up a profile by its public name (`"chrome_131"`, `"firefox_135"`,
/// `"safari_18"`, `"okhttp_5"`, ...). Returns `None` for unknown names.
pub fn profile(name: &str) -> Option<&'static Profile> {
    data::PROFILES.iter().find(|p| p.name == name)
}

/// All vendored profiles, sorted by name.
pub fn profiles() -> &'static [Profile] {
    data::PROFILES
}
