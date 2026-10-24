//! Hand-rolled TLS 1.2/1.3 client handshake (RFC 8446, RFC 5246).
//!
//! Standard-library Rust only, built exclusively on [`crate::crypto`]
//! primitives. The whole point of hand-rolling TLS here is byte-level
//! control of the ClientHello: [`FingerprintProfile`] dictates cipher suite
//! order, extension order (including GREASE insertion), supported groups,
//! and key-share selection, so the client can mimic real browsers on the
//! wire.
//!
//! What is implemented:
//!
//! * TLS 1.3 full handshake: ClientHello (with HelloRetryRequest handling),
//!   X25519 key exchange, the RFC 8446 §7.1 key schedule over HKDF-SHA-256 /
//!   HKDF-SHA-384, EncryptedExtensions / Certificate / CertificateVerify /
//!   Finished processing, and application-data record protection.
//! * Hybrid key exchange: `X25519MLKEM768` (0x11EC,
//!   draft-ietf-tls-hybrid-design) via [`crate::crypto::ml_kem_768`]. The
//!   client offers `x25519_pub(32) || mlkem_ek(1184)`, decapsulates the
//!   server's 1088-byte ciphertext, and feeds the 64-byte concatenated
//!   shared secret into the key schedule.
//! * TLS 1.2 negotiation: the ClientHello offers 1.2, a 1.2 ServerHello is
//!   parsed, and then the handshake fails closed — see below.
//! * Record layer: AEAD sealing/opening with per-record nonces, alert
//!   handling, handshake fragmentation/reassembly.
//! * Certificate chain verification: `crate::crypto::x509` validates the
//!   server's certificate path against the vendored Mozilla root store
//!   (RFC 5280), checks the hostname against the leaf (RFC 6125), and
//!   verifies the CertificateVerify signature over the transcript.
//!   Supported signature schemes: RSA PKCS#1 v1.5 with SHA-256/384/512
//!   and ECDSA over P-256/P-384 with SHA-256/384 (any hash/curve
//!   combination in chains; P-256/SHA-256 and P-384/SHA-384 for
//!   CertificateVerify). A flight that omits
//!   the Certificate / CertificateVerify messages is rejected at
//!   Finished instead of completing unauthenticated.
//!
//! * Session resumption (RFC 8446 §4.2.11, §4.6.1): `NewSessionTicket`
//!   messages received post-handshake are retained as [`CachedTicket`]s;
//!   a later connection to the same host can offer one back via
//!   [`TlsConfig::psk`], turning the handshake into a PSK resumption
//!   (no certificate flight). Tickets are single-use and expire per
//!   their `ticket_lifetime`.
//!
//! What is deliberately NOT implemented (fails closed, never silently
//! skipped):
//!
//! * TLS 1.2 past ServerHello: the 1.2 state machine is not implemented,
//!   so a 1.2 ServerHello fails closed with `TlsError::UnsupportedVersion`.
//! * Client certificates, KeyUpdate, post-handshake authentication.

use std::fmt;
use std::io::{self, Read, Write};
use std::net::TcpStream;

use crate::crypto::ml_kem_768;
use crate::crypto::x509;
use crate::crypto::{
    Aes128Gcm, Aes256Gcm, ChaCha20Poly1305, HashAlg, Sha256, Sha384, Sha512, ct_eq, hkdf_expand,
    hkdf_extract, hmac_sha256, hmac_sha384, hmac_sha512, sha256, sha384, sha512,
};
use crate::crypto::{Error as CryptoError, fill_random as random_bytes, x25519};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Wire constants
// ---------------------------------------------------------------------------

const RECORD_CHANGE_CIPHER_SPEC: u8 = 20;
const RECORD_ALERT: u8 = 21;
const RECORD_HANDSHAKE: u8 = 22;
const RECORD_APP_DATA: u8 = 23;

/// Legacy record version sent on the wire (RFC 8446 §5.1).
const LEGACY_RECORD_VERSION: u16 = 0x0301;
/// Legacy handshake version inside ClientHello/ServerHello.
const LEGACY_HANDSHAKE_VERSION: u16 = 0x0303;
const TLS13_VERSION: u16 = 0x0304;

const HS_CLIENT_HELLO: u8 = 1;
const HS_SERVER_HELLO: u8 = 2;
const HS_NEW_SESSION_TICKET: u8 = 4;
const HS_ENCRYPTED_EXTENSIONS: u8 = 8;
const HS_CERTIFICATE: u8 = 11;
const HS_CERTIFICATE_VERIFY: u8 = 15;
const HS_FINISHED: u8 = 20;

/// Special handshake type used in the HRR transcript hash (RFC 8446 §4.4.1).
const HS_MESSAGE_HASH: u8 = 0xFE;

// Cipher suites we can actually complete (TLS 1.3).
const TLS_AES_128_GCM_SHA256: u16 = 0x1301;
const TLS_AES_256_GCM_SHA384: u16 = 0x1302;
const TLS_CHACHA20_POLY1305_SHA256: u16 = 0x1303;

// Extensions.
const EXT_SERVER_NAME: u16 = 0;
const EXT_SUPPORTED_GROUPS: u16 = 10;
const EXT_EC_POINT_FORMATS: u16 = 11;
const EXT_SIGNATURE_ALGORITHMS: u16 = 13;
const EXT_ALPN: u16 = 16;
const EXT_EXTENDED_MASTER_SECRET: u16 = 23;
const EXT_SESSION_TICKET: u16 = 35;
const EXT_SUPPORTED_VERSIONS: u16 = 43;
const EXT_PSK_KEY_EXCHANGE_MODES: u16 = 45;
const EXT_PRE_SHARED_KEY: u16 = 41;
const EXT_KEY_SHARE: u16 = 51;
const EXT_COOKIE: u16 = 44;
const EXT_RENEGOTIATION_INFO: u16 = 65281;

// Named groups.
const GROUP_X25519: u16 = 0x001D;
/// X25519 + ML-KEM-768 hybrid (draft-ietf-tls-hybrid-design).
const GROUP_X25519MLKEM768: u16 = 0x11EC;

// Alert descriptions we send/receive.
const ALERT_CLOSE_NOTIFY: u8 = 0;
const ALERT_UNEXPECTED_MESSAGE: u8 = 10;
const ALERT_BAD_RECORD_MAC: u8 = 20;
const ALERT_HANDSHAKE_FAILURE: u8 = 40;
const ALERT_DECODE_ERROR: u8 = 50;
const ALERT_PROTOCOL_VERSION: u8 = 70;
const ALERT_INTERNAL_ERROR: u8 = 80;

/// ServerHello.random value marking a HelloRetryRequest (RFC 8446 §4.1.3).
const HRR_MAGIC: [u8; 32] = [
    0xCF, 0x21, 0xAD, 0x74, 0xE5, 0x9A, 0x61, 0x11, 0xBE, 0x1D, 0x8C, 0x02, 0x1E, 0x65, 0xB8, 0x91,
    0xC2, 0xA2, 0x11, 0x16, 0x7A, 0xBB, 0x8C, 0x5E, 0x07, 0x9E, 0x09, 0xE2, 0xC8, 0xA8, 0x33, 0x9C,
];

/// X25519 base point (u = 9), RFC 7746 §4.1 — public constant, safe to hardcode.
const X25519_BASEPOINT: [u8; 32] = {
    let mut b = [0u8; 32];
    b[0] = 9;
    b
};

/// Maximum TLSPlaintext fragment (2^14) and TLSCiphertext expansion allowance.
const MAX_PLAINTEXT_LEN: usize = 16384;
const MAX_RECORD_LEN: usize = MAX_PLAINTEXT_LEN + 256;
/// Cap on a single reassembled handshake message (certificates can be large).
const MAX_HANDSHAKE_MSG: usize = 65536;
/// Cap on consecutive ignored change_cipher_spec records (m6): a peer
/// flooding CCS must not keep the handshake loops alive forever.
const MAX_IGNORED_CCS: u8 = 8;
/// Cap on consecutive handshake records that add no bytes to the framer:
/// a peer drip-feeding empty handshake records would otherwise spin the
/// handshake loops forever without tripping any size cap. Records that
/// do carry fragment bytes reset the count.
const MAX_STALLED_HANDSHAKE_RECORDS: u8 = 32;
const TAG_LEN: usize = 16;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Failures from the TLS handshake and record layer.
#[derive(Debug)]
pub enum TlsError {
    /// Underlying transport I/O failed.
    Io(io::Error),
    /// A crypto primitive rejected its input.
    Crypto(CryptoError),
    /// The peer closed the transport mid-handshake.
    Eof,
    /// Malformed TLS record.
    BadRecord(&'static str),
    /// Malformed handshake message.
    BadHandshake(&'static str),
    /// AEAD authentication failed (bad tag / wrong key / wrong sequence).
    DecryptFailed,
    /// A message arrived that the state machine did not expect here.
    UnexpectedMessage(&'static str),
    /// No mutually supported TLS version.
    UnsupportedVersion,
    /// The server selected a cipher suite we did not offer or cannot do.
    UnsupportedCipherSuite(u16),
    /// The server selected a named group we cannot do a key exchange for.
    UnsupportedGroup(u16),
    /// The peer sent a fatal alert; carries the alert description byte.
    AlertReceived(u8),
    /// A certificate message was structurally invalid.
    BadCertificate(&'static str),
    /// Certificate path validation or the CertificateVerify signature
    /// check failed; the peer is unauthenticated. Always fails closed.
    /// Covers malformed chains, untrusted roots, expired certificates,
    /// hostname mismatches, bad signatures, and unsupported algorithms.
    CertificateVerificationFailed(String),
    /// The server's HelloRetryRequest carried a cookie, which this client
    /// does not echo (no cookie support yet).
    HrrCookieUnsupported,
    /// An absolute read deadline set via
    /// [`TlsConnection::set_read_deadline`] expired while assembling a
    /// record: a peer drip-feeding record bytes must not stretch a
    /// single `read` past the H2 keepalive's wall-clock budget.
    ReadDeadline,
    /// `server_name` was empty or otherwise unusable.
    InvalidServerName,
    /// Internal invariant violation (a bug, not a peer error).
    Internal(&'static str),
}

impl From<io::Error> for TlsError {
    fn from(e: io::Error) -> Self {
        TlsError::Io(e)
    }
}

impl From<CryptoError> for TlsError {
    fn from(e: CryptoError) -> Self {
        TlsError::Crypto(e)
    }
}

impl fmt::Display for TlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TlsError::Io(e) => write!(f, "tls i/o error: {e}"),
            TlsError::Crypto(e) => write!(f, "tls crypto error: {e}"),
            TlsError::Eof => write!(f, "unexpected EOF during TLS handshake"),
            TlsError::BadRecord(s) => write!(f, "bad TLS record: {s}"),
            TlsError::BadHandshake(s) => write!(f, "bad TLS handshake message: {s}"),
            TlsError::DecryptFailed => write!(f, "record authentication failed"),
            TlsError::UnexpectedMessage(s) => write!(f, "unexpected TLS message: {s}"),
            TlsError::UnsupportedVersion => write!(f, "no mutually supported TLS version"),
            TlsError::UnsupportedCipherSuite(id) => {
                write!(f, "unsupported cipher suite: 0x{id:04x}")
            }
            TlsError::UnsupportedGroup(id) => write!(f, "unsupported named group: 0x{id:04x}"),
            TlsError::AlertReceived(d) => write!(f, "peer sent fatal alert {d}"),
            TlsError::BadCertificate(s) => write!(f, "bad certificate message: {s}"),
            TlsError::CertificateVerificationFailed(reason) => {
                write!(f, "certificate verification failed: {reason}")
            }
            TlsError::HrrCookieUnsupported => {
                write!(
                    f,
                    "server sent HelloRetryRequest with a cookie, not supported"
                )
            }
            TlsError::InvalidServerName => write!(f, "invalid server name"),
            TlsError::Internal(s) => write!(f, "internal TLS error: {s}"),
            TlsError::ReadDeadline => write!(f, "TLS read deadline exceeded"),
        }
    }
}

impl std::error::Error for TlsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TlsError::Io(e) => Some(e),
            TlsError::Crypto(e) => Some(e),
            _ => None,
        }
    }
}

/// Map a `TlsError` onto an `io::Error` for the `read`/`write` API.
fn tls_to_io(e: TlsError) -> io::Error {
    match e {
        TlsError::Io(ioe) => ioe,
        TlsError::Eof => io::Error::new(io::ErrorKind::UnexpectedEof, "tls: unexpected EOF"),
        TlsError::AlertReceived(_) => io::Error::new(
            io::ErrorKind::ConnectionAborted,
            "tls: peer sent fatal alert",
        ),
        TlsError::DecryptFailed => io::Error::new(
            io::ErrorKind::InvalidData,
            "tls: record authentication failed",
        ),
        TlsError::ReadDeadline => {
            io::Error::new(io::ErrorKind::TimedOut, "tls: read deadline exceeded")
        }
        other => io::Error::new(io::ErrorKind::InvalidData, format!("tls: {other}")),
    }
}

/// Pick a fatal alert description for a handshake failure, if one should be sent.
fn alert_for(e: &TlsError) -> Option<u8> {
    match e {
        TlsError::DecryptFailed => Some(ALERT_BAD_RECORD_MAC),
        TlsError::BadRecord(_) => Some(ALERT_DECODE_ERROR),
        TlsError::BadHandshake(_) => Some(ALERT_DECODE_ERROR),
        TlsError::UnexpectedMessage(_) => Some(ALERT_UNEXPECTED_MESSAGE),
        TlsError::UnsupportedVersion => Some(ALERT_PROTOCOL_VERSION),
        TlsError::UnsupportedCipherSuite(_) | TlsError::UnsupportedGroup(_) => {
            Some(ALERT_HANDSHAKE_FAILURE)
        }
        TlsError::BadCertificate(_) | TlsError::CertificateVerificationFailed(_) => {
            Some(ALERT_HANDSHAKE_FAILURE)
        }
        TlsError::HrrCookieUnsupported => Some(ALERT_HANDSHAKE_FAILURE),
        TlsError::InvalidServerName | TlsError::Internal(_) => Some(ALERT_INTERNAL_ERROR),
        // Io / Eof / Crypto / AlertReceived: nothing useful to send.
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Configuration: TlsConfig + FingerprintProfile
// ---------------------------------------------------------------------------

/// One extension slot in the ClientHello, in exact wire order.
///
/// The profile's `extensions` vector is emitted verbatim, so byte-level
/// fingerprint control (order, presence, GREASE slots) is just a matter of
/// arranging this list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtensionSpec {
    /// server_name (SNI). Omitted when `server_name` is an IP literal.
    ServerName,
    /// extended_master_secret (TLS 1.2).
    ExtendedMasterSecret,
    /// renegotiation_info (TLS 1.2).
    RenegotiationInfo,
    /// supported_groups; GREASE value first when `grease` is set.
    SupportedGroups,
    /// ec_point_formats (TLS 1.2).
    EcPointFormats,
    /// session_ticket (TLS 1.2).
    SessionTicket,
    /// application_layer_protocol_negotiation, from `TlsConfig::alpn`.
    Alpn,
    /// signature_algorithms.
    SignatureAlgorithms,
    /// key_share, one entry per `key_share_groups`.
    KeyShare,
    /// psk_key_exchange_modes.
    PskKeyExchangeModes,
    /// supported_versions.
    SupportedVersions,
    /// A GREASE extension id (the connection's GREASE value) with an empty body.
    Grease,
    /// Raw extension: exact id and body bytes. Escape hatch for future knobs.
    Raw(u16, Vec<u8>),
}

/// Byte-level knobs for the ClientHello. This is the fingerprint.
///
/// Full per-browser profile tables (exact Chrome/Firefox/Safari extension
/// payloads) are future work; this struct already captures everything that
/// varies on the wire: order, values, and GREASE placement.
#[derive(Debug, Clone)]
pub struct FingerprintProfile {
    /// Cipher suites in wire order (GREASE is prepended when `grease` is set).
    pub cipher_suites: Vec<u16>,
    /// Extensions in exact wire order.
    pub extensions: Vec<ExtensionSpec>,
    /// Insert GREASE values (cipher list, supported_groups, extension slot).
    pub grease: bool,
    /// Named groups for supported_groups, in wire order.
    pub groups: Vec<u16>,
    /// Signature schemes for signature_algorithms, in wire order.
    pub signature_schemes: Vec<u16>,
    /// Groups that get a key_share entry, in wire order. X25519 (0x001D)
    /// and X25519MLKEM768 (0x11EC) are supported; anything else fails
    /// the handshake with [`TlsError::UnsupportedGroup`].
    pub key_share_groups: Vec<u16>,
    /// PSK key exchange modes, in wire order.
    pub psk_key_exchange_modes: Vec<u8>,
    /// Versions for supported_versions, in wire order.
    pub tls_versions: Vec<u16>,
    /// ClientHello session id length: 0 or 32.
    pub session_id_len: u8,
    /// Record-layer version marker on the ClientHello record.
    pub record_version: u16,
}

impl Default for FingerprintProfile {
    /// Approximate Chrome fingerprint: cipher order, extension order, and
    /// GREASE placement follow current Chrome. Payload details beyond order
    /// are standard values from the RFCs.
    fn default() -> Self {
        FingerprintProfile {
            cipher_suites: vec![
                0x1301, 0x1302, 0x1303, // TLS 1.3
                0xc02b, 0xc02f, 0xc02c, 0xc030, // ECDHE-{ECDSA,RSA}-AES-GCM
                0xcca9, 0xcca8, // ECDHE-{ECDSA,RSA}-CHACHA20-POLY1305
                0xc013, 0xc014, // ECDHE-RSA-AES-CBC-SHA{1,256}
                0x009c, 0x009d, // RSA-AES-GCM
                0x002f, 0x0035, // RSA-AES-CBC
            ],
            extensions: vec![
                ExtensionSpec::ServerName,
                ExtensionSpec::ExtendedMasterSecret,
                ExtensionSpec::RenegotiationInfo,
                ExtensionSpec::SupportedGroups,
                ExtensionSpec::EcPointFormats,
                ExtensionSpec::SessionTicket,
                ExtensionSpec::Alpn,
                ExtensionSpec::SignatureAlgorithms,
                ExtensionSpec::KeyShare,
                ExtensionSpec::PskKeyExchangeModes,
                ExtensionSpec::SupportedVersions,
                ExtensionSpec::Grease,
            ],
            grease: true,
            groups: vec![
                0x001d, // X25519
                0x0017, // secp256r1
                0x0018, // secp384r1
            ],
            signature_schemes: vec![
                // Only schemes crate::crypto::x509 can verify (no PSS).
                0x0403, // ecdsa_secp256r1_sha256
                0x0401, // rsa_pkcs1_sha256
                0x0503, // ecdsa_secp384r1_sha384
                0x0501, // rsa_pkcs1_sha384
                0x0601, // rsa_pkcs1_sha512
            ],
            key_share_groups: vec![0x001d],
            psk_key_exchange_modes: vec![1], // psk_dhe_ke
            tls_versions: vec![TLS13_VERSION, 0x0303],
            session_id_len: 32,
            record_version: LEGACY_RECORD_VERSION,
        }
    }
}

/// TLS client configuration.
#[derive(Clone)]
pub struct TlsConfig {
    /// Hostname for SNI (and, once verification lands, certificate checks).
    /// IP literals are sent without an SNI extension.
    pub server_name: String,
    /// ALPN protocols to offer, in preference order (e.g. `["h2", "http/1.1"]`).
    pub alpn: Vec<String>,
    /// ClientHello fingerprint knobs. `None` selects the default
    /// (Chrome-like) profile.
    pub fingerprint_profile: Option<FingerprintProfile>,
    /// Offer this pre-shared key for session resumption (RFC 8446
    /// §4.2.11). `None` does a full handshake. When the server accepts
    /// the offer, the handshake completes without a certificate flight:
    /// the server's Finished MAC over PSK-derived keys is the
    /// authentication. When the server declines, the handshake falls back
    /// to certificates transparently.
    pub psk: Option<PskOffer>,
}

impl core::fmt::Debug for TlsConfig {
    /// Never print the offered PSK: `{:?}` on a config must not leak
    /// key material into logs or error paths.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TlsConfig")
            .field("server_name", &self.server_name)
            .field("alpn", &self.alpn)
            .field("fingerprint_profile", &self.fingerprint_profile)
            .field("psk", &self.psk.as_ref().map(|_| "[redacted]"))
            .finish()
    }
}

/// One pre-shared key offered in the ClientHello for session resumption.
///
/// Built from a [`CachedTicket`] via [`CachedTicket::offer`]. Tickets are
/// single-use: pop the ticket from the cache when offering it, and do not
/// offer the same ticket twice (RFC 8446 §8.1).
#[derive(Clone)]
pub struct PskOffer {
    /// Ticket bytes: the PSK identity sent in the ClientHello.
    pub identity: Vec<u8>,
    /// `obfuscated_ticket_age` for this offer:
    /// `(ticket_age_ms + ticket_age_add) mod 2^32`, computed when the
    /// offer is built (RFC 8446 §4.2.11.1).
    pub obfuscated_ticket_age: u32,
    /// The PSK itself:
    /// `HKDF-Expand-Label(resumption_master_secret, "resumption",
    /// ticket_nonce, Hash.length)` (RFC 8446 §4.6.1).
    pub psk: Vec<u8>,
    /// Cipher suite the ticket was issued under. The server must select
    /// this suite (hence the same hash) for the offer to be accepted.
    pub cipher_suite: u16,
}

impl core::fmt::Debug for PskOffer {
    /// The PSK must never appear in `{:?}` output: redact it so a stray
    /// debug print in a log or error path cannot leak key material. The
    /// ticket identity is sent on the wire in the ClientHello, so its
    /// length is not sensitive.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PskOffer")
            .field("identity_len", &self.identity.len())
            .field("obfuscated_ticket_age", &self.obfuscated_ticket_age)
            .field("psk", &"[redacted]")
            .field("cipher_suite", &self.cipher_suite)
            .finish()
    }
}

impl Drop for PskOffer {
    /// Scrub the PSK: the caller's copy (built by
    /// [`CachedTicket::offer`], carried in [`TlsConfig`], cloned into
    /// [`Handshake`]) is dropped unscrubbed after every resumption
    /// handshake otherwise. Each holder owns its own clone, so every
    /// `Drop` scrubs exactly its own copy — no double-scrub of a moved
    /// value.
    fn drop(&mut self) {
        zeroize(self.psk.as_mut_slice());
    }
}

/// A session ticket received in a `NewSessionTicket` post-handshake
/// message, with everything needed to offer it back for resumption.
///
/// Tickets never cross hosts: the cache is keyed by server name, and a
/// ticket is only ever offered to the host that issued it.
#[derive(Clone)]
pub struct CachedTicket {
    ticket: Vec<u8>,
    psk: Vec<u8>,
    cipher_suite: u16,
    received_at_ms: u64,
    lifetime_secs: u32,
    age_add: u32,
}

impl Drop for CachedTicket {
    /// Scrub the ticket PSK on eviction: it is long-term key material,
    /// and an expired or superseded ticket must not linger in memory.
    fn drop(&mut self) {
        zeroize(self.psk.as_mut_slice());
    }
}

impl core::fmt::Debug for CachedTicket {
    /// Redact the PSK (and ticket bytes) for the same reason as
    /// [`PskOffer`]: `{:?}` must never dump key material.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CachedTicket")
            .field("ticket", &"[redacted]")
            .field("psk", &"[redacted]")
            .field("cipher_suite", &self.cipher_suite)
            .field("received_at_ms", &self.received_at_ms)
            .field("lifetime_secs", &self.lifetime_secs)
            .field("age_add", &self.age_add)
            .finish()
    }
}

/// Maximum ticket lifetime honored, in seconds (RFC 8446 §4.6.1: servers
/// MUST NOT issue lifetimes above 7 days; anything larger is clamped, not
/// trusted).
const MAX_TICKET_LIFETIME_SECS: u32 = 7 * 24 * 3600;

/// Cap on tickets retained per established connection: servers typically
/// send two; an unbounded queue would let a chatty peer grow memory.
const MAX_CACHED_TICKETS: usize = 8;

#[cfg(test)]
impl CachedTicket {
    pub(crate) fn test_ticket(lifetime_secs: u32) -> Self {
        Self {
            ticket: b"tick".to_vec(),
            psk: vec![0xab; 32],
            cipher_suite: TLS_AES_128_GCM_SHA256,
            received_at_ms: now_ms(),
            lifetime_secs,
            age_add: 7,
        }
    }
}

impl CachedTicket {
    /// Whether the ticket is still live at `now_ms` (unix millis):
    /// non-empty and younger than its `ticket_lifetime` (clamped to the
    /// 7-day RFC 8446 §4.6.1 maximum). A cheap peek for the pool's offer
    /// path, which must not build a [`PskOffer`] just to test liveness.
    pub fn is_live(&self, now_ms: u64) -> bool {
        if self.psk.is_empty() || self.ticket.is_empty() {
            return false;
        }
        let lifetime_ms = self.lifetime_secs.min(MAX_TICKET_LIFETIME_SECS) as u64 * 1000;
        now_ms < self.received_at_ms.saturating_add(lifetime_ms)
    }

    /// Build a [`PskOffer`] if the ticket is still live at `now_ms`
    /// (unix millis). Expired tickets yield `None` and must be dropped —
    /// clients MUST NOT offer tickets older than their `ticket_lifetime`
    /// (RFC 8446 §4.2.11.1).
    pub fn offer(&self, now_ms: u64) -> Option<PskOffer> {
        if !self.is_live(now_ms) {
            return None;
        }
        let age_ms = now_ms.saturating_sub(self.received_at_ms);
        Some(PskOffer {
            identity: self.ticket.clone(),
            obfuscated_ticket_age: age_ms.wrapping_add(self.age_add as u64) as u32,
            psk: self.psk.clone(),
            cipher_suite: self.cipher_suite,
        })
    }
}

/// Unix time in milliseconds. Falls back to 0 (epoch) when the clock is
/// unavailable; ticket ages then skew old, which only makes tickets
/// expire sooner — never a security issue.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Wire encoding helpers
// ---------------------------------------------------------------------------

fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_be_bytes());
}

/// Big-endian u24 length prefix + bytes.
fn put_handshake_header(out: &mut Vec<u8>, typ: u8, body: &[u8]) {
    out.push(typ);
    out.push((body.len() >> 16) as u8);
    out.push((body.len() >> 8) as u8);
    out.push(body.len() as u8);
    out.extend_from_slice(body);
}

/// Minimal fallible reader over a byte slice.
struct Reader<'a> {
    b: &'a [u8],
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Reader { b }
    }

    fn is_empty(&self) -> bool {
        self.b.is_empty()
    }

    fn u8(&mut self, ctx: &'static str) -> Result<u8, TlsError> {
        let (v, rest) = self.b.split_first().ok_or(TlsError::BadHandshake(ctx))?;
        self.b = rest;
        Ok(*v)
    }

    fn u16(&mut self, ctx: &'static str) -> Result<u16, TlsError> {
        if self.b.len() < 2 {
            return Err(TlsError::BadHandshake(ctx));
        }
        let v = u16::from_be_bytes([self.b[0], self.b[1]]);
        self.b = &self.b[2..];
        Ok(v)
    }

    fn u24(&mut self, ctx: &'static str) -> Result<usize, TlsError> {
        if self.b.len() < 3 {
            return Err(TlsError::BadHandshake(ctx));
        }
        let v = ((self.b[0] as usize) << 16) | ((self.b[1] as usize) << 8) | (self.b[2] as usize);
        self.b = &self.b[3..];
        Ok(v)
    }

    fn bytes(&mut self, n: usize, ctx: &'static str) -> Result<&'a [u8], TlsError> {
        if self.b.len() < n {
            return Err(TlsError::BadHandshake(ctx));
        }
        let (v, rest) = self.b.split_at(n);
        self.b = rest;
        Ok(v)
    }

    /// `opaque vec<u8>`: 1-byte length prefix.
    fn vec_u8(&mut self, ctx: &'static str) -> Result<&'a [u8], TlsError> {
        let n = self.u8(ctx)? as usize;
        self.bytes(n, ctx)
    }

    /// `opaque vec<u16>`: 2-byte length prefix.
    fn vec_u16(&mut self, ctx: &'static str) -> Result<&'a [u8], TlsError> {
        let n = self.u16(ctx)? as usize;
        self.bytes(n, ctx)
    }

    /// 3-byte length prefix.
    fn vec_u24(&mut self, ctx: &'static str) -> Result<&'a [u8], TlsError> {
        let n = self.u24(ctx)?;
        self.bytes(n, ctx)
    }
}

// ---------------------------------------------------------------------------
// Randomness
// ---------------------------------------------------------------------------
// `random_bytes` is [`crate::crypto::fill_random`], imported at the top of
// this module: the `/dev/urandom` fd is cached for the life of the
// process instead of being opened per draw, and non-Unix platforms fail
// gracefully at runtime (`io::ErrorKind::Unsupported`), not at compile
// time.

/// Pick a GREASE value (RFC 8701): 0x0A0A, 0x1A1A, ..., 0xFAFA.
fn grease_value(seed: u8) -> u16 {
    let b = 0x0A | (seed & 0xF0);
    ((b as u16) << 8) | b as u16
}

// ---------------------------------------------------------------------------
// ClientHello construction (byte-level fingerprint control)
// ---------------------------------------------------------------------------

fn sni_extension_body(server_name: &str) -> Vec<u8> {
    let mut list = Vec::new();
    list.push(0u8); // host_name
    put_u16(&mut list, server_name.len() as u16);
    list.extend_from_slice(server_name.as_bytes());
    let mut body = Vec::new();
    put_u16(&mut body, list.len() as u16);
    body.extend_from_slice(&list);
    body
}

fn alpn_extension_body(protos: &[String]) -> Vec<u8> {
    let mut list = Vec::new();
    for p in protos {
        list.push(p.len() as u8);
        list.extend_from_slice(p.as_bytes());
    }
    let mut body = Vec::new();
    put_u16(&mut body, list.len() as u16);
    body.extend_from_slice(&list);
    body
}

fn vec_u16_body(items: &[u16], grease_first: Option<u16>) -> Vec<u8> {
    let mut inner = Vec::new();
    if let Some(g) = grease_first {
        put_u16(&mut inner, g);
    }
    for i in items {
        put_u16(&mut inner, *i);
    }
    let mut body = Vec::new();
    put_u16(&mut body, inner.len() as u16);
    body.extend_from_slice(&inner);
    body
}

/// X25519 public key for a secret: `clamp(secret) * basepoint`.
/// `crypto::x25519` keeps its basepoint private, so use the RFC 7748
/// constant (u = 9) with the public `shared_secret` entry point.
fn x25519_public(secret: &[u8; 32]) -> [u8; 32] {
    x25519::shared_secret(secret, &X25519_BASEPOINT)
}

/// Key exchange bytes to offer for a named group in key_share. A hybrid
/// share also produces the ML-KEM decapsulation key, which the caller
/// must retain on the `Handshake` to decapsulate the server's reply.
fn key_share_bytes(
    group: u16,
    x25519_secret: &[u8; 32],
) -> Result<(Vec<u8>, Option<[u8; ml_kem_768::DECAPSULATION_KEY_LEN]>), TlsError> {
    match group {
        GROUP_X25519 => Ok((x25519_public(x25519_secret).to_vec(), None)),
        // draft-ietf-tls-hybrid-design: the key exchange value is the
        // 32-byte X25519 public key concatenated with the 1184-byte
        // ML-KEM-768 encapsulation key.
        GROUP_X25519MLKEM768 => {
            let (ek, dk) = ml_kem_768::keygen()?;
            let mut ke = Vec::with_capacity(32 + ml_kem_768::ENCAPSULATION_KEY_LEN);
            ke.extend_from_slice(&x25519_public(x25519_secret));
            ke.extend_from_slice(&ek);
            Ok((ke, Some(dk)))
        }
        other => Err(TlsError::UnsupportedGroup(other)),
    }
}

/// Append one `KeyShareEntry` for `group` to `entries`, retaining a fresh
/// hybrid decapsulation key (wiping any stale one first) so the
/// ServerHello's ciphertext can be decapsulated.
fn build_key_share_entry(
    entries: &mut Vec<u8>,
    group: u16,
    x25519_secret: &[u8; 32],
    mlkem_dk: &mut Option<[u8; ml_kem_768::DECAPSULATION_KEY_LEN]>,
) -> Result<(), TlsError> {
    let (ke, dk) = key_share_bytes(group, x25519_secret)?;
    if let Some(dk) = dk {
        if let Some(old) = mlkem_dk.as_mut() {
            zeroize(old);
        }
        *mlkem_dk = Some(dk);
    }
    put_u16(entries, group);
    put_u16(entries, ke.len() as u16);
    entries.extend_from_slice(&ke);
    Ok(())
}

/// Build one extension's body. `Ok(None)` omits the extension.
#[allow(clippy::too_many_arguments)]
fn build_extension(
    spec: &ExtensionSpec,
    profile: &FingerprintProfile,
    server_name: &str,
    alpn: &[String],
    grease: u16,
    x25519_secret: &[u8; 32],
    only_group: Option<u16>,
    mlkem_dk: &mut Option<[u8; ml_kem_768::DECAPSULATION_KEY_LEN]>,
) -> Result<Option<(u16, Vec<u8>)>, TlsError> {
    let ext = match spec {
        ExtensionSpec::ServerName => {
            if server_name.parse::<std::net::IpAddr>().is_ok() {
                return Ok(None);
            }
            (EXT_SERVER_NAME, sni_extension_body(server_name))
        }
        ExtensionSpec::ExtendedMasterSecret => (EXT_EXTENDED_MASTER_SECRET, Vec::new()),
        ExtensionSpec::RenegotiationInfo => (EXT_RENEGOTIATION_INFO, vec![0u8]),
        ExtensionSpec::SupportedGroups => (
            EXT_SUPPORTED_GROUPS,
            vec_u16_body(
                &profile.groups,
                if profile.grease { Some(grease) } else { None },
            ),
        ),
        ExtensionSpec::EcPointFormats => (EXT_EC_POINT_FORMATS, vec![1u8, 0u8]),
        ExtensionSpec::SessionTicket => (EXT_SESSION_TICKET, Vec::new()),
        ExtensionSpec::Alpn => {
            if alpn.is_empty() {
                return Ok(None);
            }
            (EXT_ALPN, alpn_extension_body(alpn))
        }
        ExtensionSpec::SignatureAlgorithms => (
            EXT_SIGNATURE_ALGORITHMS,
            vec_u16_body(&profile.signature_schemes, None),
        ),
        ExtensionSpec::KeyShare => {
            let mut entries = Vec::new();
            // On an HRR retry the server selects a group this client did
            // not offer; the retry's key_share MUST still carry a share
            // for exactly that group — RFC 8446 §4.2.8 requires
            // non-empty client_shares, and a server aborts on an empty
            // one. Filtering the profile list by `only_group` would emit
            // an empty share list here, so generate for the selected
            // group directly instead.
            if let Some(g) = only_group {
                build_key_share_entry(&mut entries, g, x25519_secret, mlkem_dk)?;
            } else {
                for g in &profile.key_share_groups {
                    build_key_share_entry(&mut entries, *g, x25519_secret, mlkem_dk)?;
                }
            }
            let mut body = Vec::new();
            put_u16(&mut body, entries.len() as u16);
            body.extend_from_slice(&entries);
            (EXT_KEY_SHARE, body)
        }
        ExtensionSpec::PskKeyExchangeModes => {
            let mut body = Vec::with_capacity(1 + profile.psk_key_exchange_modes.len());
            body.push(profile.psk_key_exchange_modes.len() as u8);
            body.extend_from_slice(&profile.psk_key_exchange_modes);
            (EXT_PSK_KEY_EXCHANGE_MODES, body)
        }
        ExtensionSpec::SupportedVersions => {
            let mut body = Vec::with_capacity(1 + 2 * profile.tls_versions.len());
            body.push((2 * profile.tls_versions.len()) as u8);
            for v in &profile.tls_versions {
                put_u16(&mut body, *v);
            }
            (EXT_SUPPORTED_VERSIONS, body)
        }
        ExtensionSpec::Grease => {
            if !profile.grease {
                return Ok(None);
            }
            (grease, Vec::new())
        }
        ExtensionSpec::Raw(id, data) => (*id, data.clone()),
    };
    Ok(Some(ext))
}

/// Build a full ClientHello handshake message (4-byte header included).
///
/// When `psk` is `Some`, a `pre_shared_key` extension (RFC 8446 §4.2.11)
/// is appended as the last extension, and its binder is computed over
/// the transcript through the truncated ClientHello: the transcript
/// state in `psk.1` plus the message up to — but not including — the
/// binders list (`Truncate()` removes the whole binders list, length
/// prefix included). All length fields are set as if the binders were
/// present.
#[allow(clippy::too_many_arguments)]
fn build_client_hello_msg(
    profile: &FingerprintProfile,
    server_name: &str,
    alpn: &[String],
    random: &[u8; 32],
    session_id: &[u8],
    grease: u16,
    x25519_secret: &[u8; 32],
    only_group: Option<u16>,
    psk: Option<(&PskOffer, &TranscriptHasher)>,
    // Receives the ML-KEM decapsulation key when a hybrid key share is
    // built; the caller (`Handshake`) retains it.
    mlkem_dk: &mut Option<[u8; ml_kem_768::DECAPSULATION_KEY_LEN]>,
) -> Result<Vec<u8>, TlsError> {
    if server_name.is_empty() {
        return Err(TlsError::InvalidServerName);
    }
    for p in alpn {
        if p.is_empty() || p.len() > 255 {
            return Err(TlsError::BadHandshake("bad ALPN protocol"));
        }
    }

    let mut body = Vec::new();
    put_u16(&mut body, LEGACY_HANDSHAKE_VERSION);
    body.extend_from_slice(random);
    body.push(session_id.len() as u8);
    body.extend_from_slice(session_id);

    let mut suites = Vec::new();
    if profile.grease {
        put_u16(&mut suites, grease);
    }
    for s in &profile.cipher_suites {
        put_u16(&mut suites, *s);
    }
    put_u16(&mut body, suites.len() as u16);
    body.extend_from_slice(&suites);

    body.push(1u8); // compression_methods length
    body.push(0u8); // null compression

    let mut exts = Vec::new();
    for spec in &profile.extensions {
        if let Some((id, data)) = build_extension(
            spec,
            profile,
            server_name,
            alpn,
            grease,
            x25519_secret,
            only_group,
            mlkem_dk,
        )? {
            put_u16(&mut exts, id);
            put_u16(&mut exts, data.len() as u16);
            exts.extend_from_slice(&data);
        }
    }
    // RFC 8446 §4.2.11: the pre_shared_key extension MUST be the last
    // extension in the ClientHello. The binder is a placeholder here and
    // is patched in after the message is assembled (see below).
    if let Some((offer, _)) = psk {
        let spec = cipher_spec(offer.cipher_suite)
            .ok_or(TlsError::UnsupportedCipherSuite(offer.cipher_suite))?;
        let binder_len = hash_len(spec.hash);
        let mut identities = Vec::new();
        put_u16(&mut identities, offer.identity.len() as u16);
        identities.extend_from_slice(&offer.identity);
        identities.extend_from_slice(&offer.obfuscated_ticket_age.to_be_bytes());
        let mut binders = Vec::new();
        binders.push(binder_len as u8);
        binders.extend(std::iter::repeat_n(0u8, binder_len));
        let mut ext_body = Vec::new();
        put_u16(&mut ext_body, identities.len() as u16);
        ext_body.extend_from_slice(&identities);
        put_u16(&mut ext_body, binders.len() as u16);
        ext_body.extend_from_slice(&binders);
        put_u16(&mut exts, EXT_PRE_SHARED_KEY);
        put_u16(&mut exts, ext_body.len() as u16);
        exts.extend_from_slice(&ext_body);
    }
    put_u16(&mut body, exts.len() as u16);
    body.extend_from_slice(&exts);

    let mut msg = Vec::with_capacity(4 + body.len());
    put_handshake_header(&mut msg, HS_CLIENT_HELLO, &body);

    if let Some((offer, transcript)) = psk {
        // The binder value is the last `binder_len` bytes of the message;
        // the binders list (u16 length + u8 entry length + value) is the
        // last thing in the extension, and the extension is last.
        let spec = cipher_spec(offer.cipher_suite)
            .ok_or(TlsError::UnsupportedCipherSuite(offer.cipher_suite))?;
        let hash = spec.hash;
        let binder_len = hash_len(hash);
        let value_off = msg
            .len()
            .checked_sub(binder_len)
            .ok_or(TlsError::Internal("psk binder underflow"))?;
        // Truncate() drops the binders list: u16 list length + u8 entry
        // length + the value itself.
        let trunc_off = value_off
            .checked_sub(3)
            .ok_or(TlsError::Internal("psk binder underflow"))?;
        let truncated = msg
            .get(..trunc_off)
            .ok_or(TlsError::Internal("psk binder underflow"))?;
        let thash = transcript.digest_with_suffix(hash, truncated);
        // binder_key = Derive-Secret(early_secret, "res binder", ""),
        // with early_secret = HKDF-Extract(zeros, psk). Resumption PSKs
        // use the "res binder" label; "ext binder" is for out-of-band
        // PSKs (RFC 8446 §7.1).
        let zeros = vec![0u8; binder_len];
        let mut early = hkdf_extract(hash, &zeros, &offer.psk);
        let mut binder_key =
            derive_secret(hash, &early, b"res binder", &transcript_hash(hash, b""))?;
        zeroize(early.as_mut_slice());
        let mut fkey = finished_key(hash, &binder_key)?;
        zeroize(binder_key.as_mut_slice());
        let binder = hmac(hash, &fkey, &thash);
        zeroize(fkey.as_mut_slice());
        debug_assert_eq!(binder.len(), binder_len);
        msg[value_off..].copy_from_slice(&binder);
    }
    Ok(msg)
}

// ---------------------------------------------------------------------------
// Record layer
// ---------------------------------------------------------------------------

/// AEAD algorithm backing a set of traffic keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AeadKind {
    Aes128Gcm,
    Aes256Gcm,
    ChaCha20Poly1305,
}

/// Traffic keys for one direction: AEAD key + IV. The per-record nonce is
/// `iv XOR seq` with the 64-bit sequence number in the low bytes (RFC 8446 §5.3).
#[derive(Clone)]
struct TrafficKeys {
    aead: AeadKind,
    key: Vec<u8>,
    iv: [u8; 12],
}

impl core::fmt::Debug for TrafficKeys {
    /// Redact key and IV: both are secret key material and must never
    /// appear in `{:?}` output.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TrafficKeys")
            .field("aead", &self.aead)
            .field("key", &"[redacted]")
            .field("iv", &"[redacted]")
            .finish()
    }
}

/// Overwrite `buf` with zeros through volatile writes, so the compiler
/// cannot elide the wipe as a dead store. Used to scrub key material on
/// drop. std-only: no external zeroize crate.
fn zeroize(buf: &mut [u8]) {
    for b in buf.iter_mut() {
        // SAFETY: `b` is a `&mut u8` into a live allocation, so the
        // pointer is valid, properly aligned, and exclusively borrowed
        // for the duration of the write; storing 0 is always a valid
        // `u8` value.
        unsafe {
            core::ptr::write_volatile(b, 0);
        }
    }
}

impl Drop for TrafficKeys {
    /// Scrub the AEAD key and IV: dropped key material must not linger
    /// in freed (and potentially reused) memory.
    fn drop(&mut self) {
        zeroize(self.key.as_mut_slice());
        zeroize(&mut self.iv);
    }
}

fn record_nonce(iv: &[u8; 12], seq: u64) -> [u8; 12] {
    let mut n = *iv;
    for (i, b) in seq.to_be_bytes().iter().enumerate() {
        n[4 + i] ^= b;
    }
    n
}

/// Verify the tag and decrypt in place for the connection's AEAD:
/// `buf` holds `ct || tag`; the tag is verified first (fail closed) and
/// the decrypted plaintext region is returned on success. No allocation.
fn open_aead_in_place<'b>(
    kind: AeadKind,
    key: &[u8],
    nonce: &[u8; 12],
    aad: &[u8],
    buf: &'b mut [u8],
) -> Result<&'b mut [u8], TlsError> {
    let bad = || TlsError::Internal("bad traffic key length");
    let open = match kind {
        AeadKind::Aes128Gcm => {
            let k: &[u8; 16] = key.try_into().map_err(|_| bad())?;
            Aes128Gcm::new(k).open_in_place(nonce, aad, buf)
        }
        AeadKind::Aes256Gcm => {
            let k: &[u8; 32] = key.try_into().map_err(|_| bad())?;
            Aes256Gcm::new(k).open_in_place(nonce, aad, buf)
        }
        AeadKind::ChaCha20Poly1305 => {
            let k: &[u8; 32] = key.try_into().map_err(|_| bad())?;
            ChaCha20Poly1305::new(k).open_in_place(nonce, aad, buf)
        }
    };
    // Only authentication failures map to DecryptFailed; other crypto
    // errors (InvalidLength, and anything added later) pass through as
    // TlsError::Crypto so their provenance is not flattened.
    open.map_err(|e| match e {
        CryptoError::AuthenticationFailed => TlsError::DecryptFailed,
        other => TlsError::Crypto(other),
    })
}

/// Encrypt in place for the connection's AEAD: encrypts `buf` (the record
/// fragment: plaintext + inner-type byte) and returns the tag, which the
/// caller appends. No allocation.
fn seal_aead_in_place(
    kind: AeadKind,
    key: &[u8],
    nonce: &[u8; 12],
    aad: &[u8],
    buf: &mut [u8],
) -> Result<[u8; 16], TlsError> {
    let bad = || TlsError::Internal("bad traffic key length");
    match kind {
        AeadKind::Aes128Gcm => {
            let k: &[u8; 16] = key.try_into().map_err(|_| bad())?;
            Ok(Aes128Gcm::new(k).seal_in_place(nonce, aad, buf)?)
        }
        AeadKind::Aes256Gcm => {
            let k: &[u8; 32] = key.try_into().map_err(|_| bad())?;
            Ok(Aes256Gcm::new(k).seal_in_place(nonce, aad, buf)?)
        }
        AeadKind::ChaCha20Poly1305 => {
            let k: &[u8; 32] = key.try_into().map_err(|_| bad())?;
            Ok(ChaCha20Poly1305::new(k).seal_in_place(nonce, aad, buf)?)
        }
    }
}

/// Seal one TLS 1.3 record: encrypt `plaintext || inner_type`, outer type 23.
/// Returns the full record bytes (5-byte header + fragment).
///
/// Single allocation: the record buffer is sized up front for
/// `5 + plaintext + 1 + 16`, the header and plaintext are written into it,
/// and the fragment region is sealed in place — the AAD is exactly the
/// record header, whose fragment length is known before encryption.
fn seal_record(
    keys: &TrafficKeys,
    seq: u64,
    inner_type: u8,
    plaintext: &[u8],
) -> Result<Vec<u8>, TlsError> {
    if plaintext.len() > MAX_PLAINTEXT_LEN {
        return Err(TlsError::Internal("record plaintext too large"));
    }
    let frag_len = plaintext.len() + 1 + TAG_LEN;
    let nonce = record_nonce(&keys.iv, seq);
    let mut rec = Vec::with_capacity(5 + frag_len);
    rec.push(RECORD_APP_DATA);
    rec.extend_from_slice(&LEGACY_RECORD_VERSION.to_be_bytes());
    rec.extend_from_slice(&(frag_len as u16).to_be_bytes());
    rec.extend_from_slice(plaintext);
    rec.push(inner_type);
    // Seal the fragment region in place, then append the tag. Capacity was
    // sized for header + fragment + tag, so nothing reallocates.
    let (hdr, frag) = rec.split_at_mut(5);
    let tag = seal_aead_in_place(keys.aead, &keys.key, &nonce, hdr, frag)?;
    rec.extend_from_slice(&tag);
    debug_assert_eq!(rec.len(), 5 + frag_len);
    Ok(rec)
}

/// Open one TLS 1.3 record in place. Returns `(inner_content_type, plaintext)`
/// with padding stripped; the plaintext is a reborrow of the record buffer,
/// not a new allocation.
///
/// Fail-closed ordering is preserved: `open_in_place` verifies the tag
/// before any decryption runs, so on authentication failure the buffer
/// still holds ciphertext — never half-decrypted plaintext.
fn open_record<'r>(
    keys: &TrafficKeys,
    seq: u64,
    record: &'r mut [u8],
) -> Result<(u8, &'r mut [u8]), TlsError> {
    if record.len() < 5 {
        return Err(TlsError::BadRecord("record shorter than header"));
    }
    if record[0] != RECORD_APP_DATA {
        return Err(TlsError::BadRecord("expected encrypted record"));
    }
    let declared = u16::from_be_bytes([record[3], record[4]]) as usize;
    let (hdr, frag) = record.split_at_mut(5);
    if declared != frag.len() {
        return Err(TlsError::BadRecord("record length mismatch"));
    }
    if frag.len() < TAG_LEN + 1 {
        return Err(TlsError::BadRecord("record fragment too short"));
    }

    let nonce = record_nonce(&keys.iv, seq);
    // The AAD is the 5-byte record header. `frag` is `ct || tag`, exactly
    // what `open_in_place` expects; it verifies the tag first, then
    // decrypts the fragment region in place.
    let pt = open_aead_in_place(keys.aead, &keys.key, &nonce, hdr, frag)?;

    // Strip padding: plaintext is `content || inner_type || zeros*`.
    // The scan is data-dependent but only reveals padding length, which is
    // not secret in TLS 1.3 (the peer chose it).
    let mut idx = pt.len();
    let inner = loop {
        if idx == 0 {
            return Err(TlsError::BadRecord("record with no content type"));
        }
        idx -= 1;
        if pt[idx] != 0 {
            break pt[idx];
        }
    };
    Ok((inner, &mut pt[..idx]))
}

/// Read one full record (header + fragment) from the transport.
fn read_record<S: io::Read>(stream: &mut S) -> Result<Vec<u8>, TlsError> {
    let mut hdr = [0u8; 5];
    stream.read_exact(&mut hdr).map_err(map_eof)?;
    let len = u16::from_be_bytes([hdr[3], hdr[4]]) as usize;
    if len > MAX_RECORD_LEN {
        return Err(TlsError::BadRecord("record too large"));
    }
    let mut rec = vec![0u8; 5 + len];
    rec[..5].copy_from_slice(&hdr);
    stream.read_exact(&mut rec[5..]).map_err(map_eof)?;
    Ok(rec)
}

fn map_eof(e: io::Error) -> TlsError {
    if e.kind() == io::ErrorKind::UnexpectedEof {
        TlsError::Eof
    } else {
        TlsError::Io(e)
    }
}

/// Move one complete record of `len` bytes out of the read stash
/// without copying it: `split_off` copies only the (usually empty)
/// remainder into a fresh buffer, and the swap hands back the stash's
/// own buffer as the record. The previous `to_vec` + `drain` paid one
/// full copy of the record plus a memmove of the remainder per record
/// on the data path.
///
/// `len` must not exceed `stash.len()`.
fn take_stash_record(stash: &mut Vec<u8>, len: usize) -> Vec<u8> {
    let mut rec = stash.split_off(len);
    std::mem::swap(&mut rec, stash);
    rec
}

/// Write one plaintext record (handshake / alert / change_cipher_spec).
fn write_record<S: io::Write>(
    stream: &mut S,
    ctype: u8,
    version: u16,
    fragment: &[u8],
) -> io::Result<()> {
    let mut rec = Vec::with_capacity(5 + fragment.len());
    rec.push(ctype);
    rec.extend_from_slice(&version.to_be_bytes());
    rec.extend_from_slice(&(fragment.len() as u16).to_be_bytes());
    rec.extend_from_slice(fragment);
    stream.write_all(&rec)
}

/// Reassembles handshake messages, which may be fragmented across records
/// or coalesced several per record.
struct HandshakeFramer {
    buf: Vec<u8>,
}

impl HandshakeFramer {
    fn new() -> Self {
        HandshakeFramer { buf: Vec::new() }
    }

    fn push(&mut self, fragment: &[u8]) -> Result<(), TlsError> {
        self.buf.extend_from_slice(fragment);
        // Cap buffered bytes even before a 4-byte header is available: a
        // peer drip-feeding short fragments must not grow this without
        // bound (m7).
        if self.buf.len() > MAX_HANDSHAKE_MSG + 4 {
            return Err(TlsError::BadHandshake("handshake message too large"));
        }
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Returns the next complete message as `(type, full_message)` where
    /// `full_message` includes the 4-byte handshake header (needed for the
    /// transcript hash).
    fn next_message(&mut self) -> Result<Option<(u8, Vec<u8>)>, TlsError> {
        const CTX: &str = "handshake framing";
        if self.buf.len() < 4 {
            return Ok(None);
        }
        let len =
            ((self.buf[1] as usize) << 16) | ((self.buf[2] as usize) << 8) | self.buf[3] as usize;
        if len > MAX_HANDSHAKE_MSG {
            return Err(TlsError::BadHandshake(CTX));
        }
        if self.buf.len() < 4 + len {
            return Ok(None);
        }
        let typ = self.buf[0];
        let msg = self.buf[..4 + len].to_vec();
        self.buf.drain(..4 + len);
        Ok(Some((typ, msg)))
    }
}

/// Account for one handshake record that completed no message: a record
/// carrying fragment bytes resets the stall count (reassembly is making
/// progress), while a zero-byte record increments it — an empty-record
/// flood fails closed past [`MAX_STALLED_HANDSHAKE_RECORDS`]. Records
/// that complete a message reset the counter at their call site instead:
/// a consumed message is progress even though it drains the framer,
/// which a before/after buffered-length comparison would misread as a
/// stall.
fn note_fragment_progress(had_bytes: bool, stalled: &mut u8) -> Result<(), TlsError> {
    if had_bytes {
        *stalled = 0;
    } else {
        *stalled += 1;
        if *stalled > MAX_STALLED_HANDSHAKE_RECORDS {
            return Err(TlsError::UnexpectedMessage(
                "too many handshake records without progress",
            ));
        }
    }
    Ok(())
}

/// Validate a pre-handshake dummy ChangeCipherSpec per the TLS 1.3
/// middlebox-compatibility rules (RFC 8446 §4): exactly one payload
/// byte of value 1, and legacy_record_version exactly 0x0303 (all
/// post-ClientHello records use 0x0303 per RFC 8446 §5.1). Anything else
/// is malformed: counted CCS records are validated, not just counted,
/// so a nonconforming record cannot smuggle data through the CCS path.
fn valid_dummy_ccs(rec: &[u8]) -> bool {
    if rec.len() != 6 || rec[5] != 0x01 {
        return false;
    }
    u16::from_be_bytes([rec[1], rec[2]]) == 0x0303
}

/// Incremental transcript hash (RFC 8446 §4.4).
///
/// The cipher suite — and therefore the hash algorithm — is unknown until
/// the ServerHello arrives, so handshake bytes are buffered raw only up to
/// that point (the ClientHello, under 2 KiB). Once the suite is negotiated
/// every message is fed into the running hash state and the raw bytes are
/// dropped: the transcript is hashed incrementally, never re-scanned, and
/// each digest is a cheap state clone + finalize.
#[derive(Clone)]
enum TranscriptHasher {
    /// Cipher not yet negotiated: raw handshake bytes.
    Pending(Vec<u8>),
    Sha256(Sha256),
    Sha384(Sha384),
    Sha512(Sha512),
}

impl TranscriptHasher {
    fn new() -> Self {
        // A ClientHello is well under 1 KiB; one buffer, no regrow.
        TranscriptHasher::Pending(Vec::with_capacity(1024))
    }

    /// Feed one handshake message, including its 4-byte header.
    fn push(&mut self, msg: &[u8]) {
        match self {
            TranscriptHasher::Pending(b) => b.extend_from_slice(msg),
            TranscriptHasher::Sha256(h) => h.update(msg),
            TranscriptHasher::Sha384(h) => h.update(msg),
            TranscriptHasher::Sha512(h) => h.update(msg),
        }
    }

    /// Lock in the hash algorithm once the cipher suite is negotiated,
    /// hashing any buffered pre-cipher bytes. No-op if already sealed
    /// (the post-HRR ServerHello must select the same suite).
    fn seal(&mut self, hash: HashAlg) {
        let pending = match self {
            TranscriptHasher::Pending(b) => std::mem::take(b),
            _ => return,
        };
        let mut sealed = match hash {
            HashAlg::Sha256 => TranscriptHasher::Sha256(Sha256::new()),
            HashAlg::Sha384 => TranscriptHasher::Sha384(Sha384::new()),
            HashAlg::Sha512 => TranscriptHasher::Sha512(Sha512::new()),
        };
        sealed.push(&pending);
        *self = sealed;
    }

    /// HelloRetryRequest transcript rewrite (RFC 8446 §4.4.1): the new
    /// transcript is `message_hash(CH1) || HRR || ...`, hashed from
    /// scratch under the HRR-selected suite's hash.
    fn reset_for_hrr(&mut self, hash: HashAlg, ch1_hash: &[u8]) {
        debug_assert!(matches!(self, TranscriptHasher::Pending(_)));
        let mut hdr = [0u8; 4];
        hdr[0] = HS_MESSAGE_HASH;
        // Hash outputs (32/48/64) always fit in the one-byte length field.
        hdr[3] = ch1_hash.len() as u8;
        let mut sealed = match hash {
            HashAlg::Sha256 => TranscriptHasher::Sha256(Sha256::new()),
            HashAlg::Sha384 => TranscriptHasher::Sha384(Sha384::new()),
            HashAlg::Sha512 => TranscriptHasher::Sha512(Sha512::new()),
        };
        sealed.push(&hdr);
        sealed.push(ch1_hash);
        *self = sealed;
    }

    /// The transcript digest so far. Clones the running state, so the
    /// transcript keeps accumulating afterwards.
    fn digest(&self, hash: HashAlg) -> Vec<u8> {
        match self {
            TranscriptHasher::Pending(b) => transcript_hash(hash, b),
            TranscriptHasher::Sha256(h) => {
                debug_assert!(matches!(hash, HashAlg::Sha256));
                h.clone().finalize().to_vec()
            }
            TranscriptHasher::Sha384(h) => {
                debug_assert!(matches!(hash, HashAlg::Sha384));
                h.clone().finalize().to_vec()
            }
            TranscriptHasher::Sha512(h) => {
                debug_assert!(matches!(hash, HashAlg::Sha512));
                h.clone().finalize().to_vec()
            }
        }
    }

    /// Digest of the transcript so far plus `suffix`, without disturbing
    /// the running state. Used for the PSK binder, which covers the
    /// transcript through a *partial* ClientHello that must not enter the
    /// real transcript yet (RFC 8446 §4.2.11.2).
    fn digest_with_suffix(&self, hash: HashAlg, suffix: &[u8]) -> Vec<u8> {
        match self {
            TranscriptHasher::Pending(b) => {
                let mut all = Vec::with_capacity(b.len() + suffix.len());
                all.extend_from_slice(b);
                all.extend_from_slice(suffix);
                transcript_hash(hash, &all)
            }
            TranscriptHasher::Sha256(h) => {
                debug_assert!(matches!(hash, HashAlg::Sha256));
                let mut c = h.clone();
                c.update(suffix);
                c.finalize().to_vec()
            }
            TranscriptHasher::Sha384(h) => {
                debug_assert!(matches!(hash, HashAlg::Sha384));
                let mut c = h.clone();
                c.update(suffix);
                c.finalize().to_vec()
            }
            TranscriptHasher::Sha512(h) => {
                debug_assert!(matches!(hash, HashAlg::Sha512));
                let mut c = h.clone();
                c.update(suffix);
                c.finalize().to_vec()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TLS 1.3 key schedule (RFC 8446 §7.1)
// ---------------------------------------------------------------------------

/// A TLS 1.3 cipher suite we can complete.
#[derive(Debug, Clone, Copy)]
struct CipherSpec {
    id: u16,
    hash: HashAlg,
    aead: AeadKind,
    key_len: usize,
}

fn cipher_spec(id: u16) -> Option<CipherSpec> {
    match id {
        TLS_AES_128_GCM_SHA256 => Some(CipherSpec {
            id,
            hash: HashAlg::Sha256,
            aead: AeadKind::Aes128Gcm,
            key_len: 16,
        }),
        TLS_AES_256_GCM_SHA384 => Some(CipherSpec {
            id,
            hash: HashAlg::Sha384,
            aead: AeadKind::Aes256Gcm,
            key_len: 32,
        }),
        TLS_CHACHA20_POLY1305_SHA256 => Some(CipherSpec {
            id,
            hash: HashAlg::Sha256,
            aead: AeadKind::ChaCha20Poly1305,
            key_len: 32,
        }),
        _ => None,
    }
}

fn hash_len(hash: HashAlg) -> usize {
    match hash {
        HashAlg::Sha256 => 32,
        HashAlg::Sha384 => 48,
        HashAlg::Sha512 => 64,
    }
}

fn transcript_hash(hash: HashAlg, transcript: &[u8]) -> Vec<u8> {
    match hash {
        HashAlg::Sha256 => sha256(transcript).to_vec(),
        HashAlg::Sha384 => sha384(transcript).to_vec(),
        HashAlg::Sha512 => sha512(transcript).to_vec(),
    }
}

fn hmac(hash: HashAlg, key: &[u8], msg: &[u8]) -> Vec<u8> {
    match hash {
        HashAlg::Sha256 => hmac_sha256(key, msg).to_vec(),
        HashAlg::Sha384 => hmac_sha384(key, msg).to_vec(),
        HashAlg::Sha512 => hmac_sha512(key, msg).to_vec(),
    }
}

/// HKDF-Expand-Label(Secret, Label, Context, Length), RFC 8446 §7.1.
///
/// The info struct is built on the stack. Labels used here are at most
/// 32 bytes and the context is a transcript hash (at most 64 bytes) --
/// except the NewSessionTicket nonce, which is attacker-controlled up
/// to 255 bytes (RFC 8446 §4.6.1) and flows in as the context. The
/// buffer is sized for that worst case (2 + 1 + 6 + 32 + 1 + 255 = 297
/// bytes), so a long legal nonce derives a ticket PSK instead of
/// failing the whole connection; anything beyond the RFC maxima is
/// still rejected below, fail closed, never panic.
fn hkdf_expand_label(
    hash: HashAlg,
    secret: &[u8],
    label: &[u8],
    context: &[u8],
    len: usize,
) -> Result<Vec<u8>, TlsError> {
    // info = u16 length || u8 label_len || "tls13 " + label ||
    //        u8 context_len || context: 2 + 1 + 6 + label + 1 + context.
    // The length check below is the real guard, not a debug assertion:
    // NewSessionTicket's nonce is attacker-controlled, so over-long
    // inputs must return an error in every build, never panic.
    if label.len() > 32 || context.len() > 255 || 10 + label.len() + context.len() > 297 {
        return Err(TlsError::Crypto(CryptoError::InvalidLength));
    }
    let mut info = [0u8; 297];
    let mut n = 0;
    info[n..n + 2].copy_from_slice(&(len as u16).to_be_bytes());
    n += 2;
    info[n] = (b"tls13 ".len() + label.len()) as u8;
    n += 1;
    info[n..n + 6].copy_from_slice(b"tls13 ");
    n += 6;
    info[n..n + label.len()].copy_from_slice(label);
    n += label.len();
    info[n] = context.len() as u8;
    n += 1;
    info[n..n + context.len()].copy_from_slice(context);
    n += context.len();
    Ok(hkdf_expand(hash, secret, &info[..n], len)?)
}

/// Derive-Secret(Secret, Label, Messages) = HKDF-Expand-Label with the
/// transcript hash as context.
fn derive_secret(
    hash: HashAlg,
    secret: &[u8],
    label: &[u8],
    transcript_hash: &[u8],
) -> Result<Vec<u8>, TlsError> {
    hkdf_expand_label(hash, secret, label, transcript_hash, hash_len(hash))
}

/// RFC 8446 §4.4: `finished_key = HKDF-Expand-Label(traffic_secret,
/// "finished", "", Hash.length)`. Unlike [`derive_secret`], the context
/// here is the empty string, not the transcript hash; using
/// `derive_secret` for this step is an interop-breaking bug (Finished
/// verify_data would never match a real peer).
fn finished_key(hash: HashAlg, traffic_secret: &[u8]) -> Result<Vec<u8>, TlsError> {
    hkdf_expand_label(hash, traffic_secret, b"finished", b"", hash_len(hash))
}

fn traffic_keys(
    hash: HashAlg,
    traffic_secret: &[u8],
    spec: &CipherSpec,
) -> Result<TrafficKeys, TlsError> {
    let key = hkdf_expand_label(hash, traffic_secret, b"key", b"", spec.key_len)?;
    let iv_bytes = hkdf_expand_label(hash, traffic_secret, b"iv", b"", 12)?;
    let mut iv = [0u8; 12];
    iv.copy_from_slice(&iv_bytes);
    Ok(TrafficKeys {
        aead: spec.aead,
        key,
        iv,
    })
}

// ---------------------------------------------------------------------------
// Handshake state machine (transport-independent; driven by byte messages)
// ---------------------------------------------------------------------------

/// Outcome of processing a ServerHello.
#[derive(Debug)]
enum HelloAction {
    /// Continue with the encrypted server flight.
    Proceed,
    /// The server sent HelloRetryRequest; send these ClientHello bytes next.
    RetryHello(Vec<u8>),
}

struct Handshake {
    profile: FingerprintProfile,
    server_name: String,
    alpn: Vec<String>,
    cipher: Option<CipherSpec>,
    client_random: [u8; 32],
    x25519_secret: [u8; 32],
    /// ML-KEM-768 decapsulation key for an offered X25519MLKEM768 key
    /// share, retained from ClientHello construction so the ServerHello
    /// ciphertext can be decapsulated. `None` unless a hybrid share was
    /// offered.
    mlkem_dk: Option<[u8; ml_kem_768::DECAPSULATION_KEY_LEN]>,
    session_id: Vec<u8>,
    grease: u16,
    /// Raw handshake messages (with 4-byte headers), HRR-adjusted.
    transcript: TranscriptHasher,
    client_hs_traffic: Vec<u8>,
    server_hs_traffic: Vec<u8>,
    handshake_secret: Vec<u8>,
    client_hs: Option<TrafficKeys>,
    server_hs: Option<TrafficKeys>,
    server_seq: u64,
    client_seq: u64,
    client_app: Option<TrafficKeys>,
    server_app: Option<TrafficKeys>,
    alpn_selected: Option<String>,
    hrr_done: bool,
    /// Encrypted-flight message order (RFC 8446 §4.8), enforced by the
    /// `handle_*` methods so every driver gets the checks: EE first, then
    /// Certificate, then CertificateVerify, then Finished.
    saw_ee: bool,
    saw_cert: bool,
    saw_cv: bool,
    /// Set only after the Certificate chain validates AND the
    /// CertificateVerify signature verifies. `handle_finished` fails
    /// closed unless this is set.
    cert_verified: bool,
    /// The leaf certificate's public key, retained from the Certificate
    /// message so `handle_certificate_verify` can check the
    /// CertificateVerify signature against it.
    leaf_spki: Option<x509::Spki>,
    /// Trust anchors for chain validation. `None` (the default) uses the
    /// vendored Mozilla root store; tests override this with fixture CAs.
    trust_roots: Option<Vec<x509::ParsedCert>>,
    /// PSK offer for session resumption, from [`TlsConfig::psk`]. `None`
    /// does a full handshake.
    psk: Option<PskOffer>,
    /// The server selected our PSK identity: the handshake completes
    /// without Certificate/CertificateVerify, authenticated by the
    /// Finished MAC over PSK-derived keys instead.
    psk_accepted: bool,
    /// `Derive-Secret(master_secret, "res master", "")` (RFC 8446 §7.1),
    /// retained so post-handshake `NewSessionTicket` messages can derive
    /// their PSKs.
    resumption_master_secret: Vec<u8>,
}

impl Drop for Handshake {
    /// Scrub retained key material: the X25519 secret, the handshake
    /// secret, the resumption master secret, the handshake traffic
    /// secrets, and the ML-KEM decapsulation key. The offered PSK is
    /// scrubbed by [`PskOffer`]'s own `Drop` impl — no manual wipe here,
    /// so the scrubbing lives in exactly one place.
    fn drop(&mut self) {
        zeroize(&mut self.x25519_secret);
        zeroize(self.handshake_secret.as_mut_slice());
        zeroize(self.resumption_master_secret.as_mut_slice());
        zeroize(self.client_hs_traffic.as_mut_slice());
        zeroize(self.server_hs_traffic.as_mut_slice());
        if let Some(dk) = self.mlkem_dk.as_mut() {
            zeroize(dk);
        }
    }
}

impl Handshake {
    fn new_with_secrets(
        config: &TlsConfig,
        client_random: [u8; 32],
        x25519_secret: [u8; 32],
        session_id: Vec<u8>,
    ) -> Result<Self, TlsError> {
        // A trailing dot is the FQDN root marker (RFC 1034), not part of
        // the name: strip it so SNI and hostname verification use the
        // same canonical form (RFC 6066 carries the bare hostname).
        let server_name = config.server_name.trim_end_matches('.');
        if server_name.is_empty() {
            return Err(TlsError::InvalidServerName);
        }
        let profile = config.fingerprint_profile.clone().unwrap_or_default();
        if profile.key_share_groups.is_empty() {
            return Err(TlsError::Internal("no key share groups in profile"));
        }
        for g in &profile.key_share_groups {
            if *g != GROUP_X25519 && *g != GROUP_X25519MLKEM768 {
                return Err(TlsError::UnsupportedGroup(*g));
            }
        }
        let mut session_id = session_id;
        session_id.truncate(profile.session_id_len as usize);
        Ok(Handshake {
            profile,
            server_name: server_name.to_owned(),
            alpn: config.alpn.clone(),
            cipher: None,
            client_random,
            x25519_secret,
            mlkem_dk: None,
            session_id,
            grease: grease_value(client_random[0]),
            transcript: TranscriptHasher::new(),
            client_hs_traffic: Vec::new(),
            server_hs_traffic: Vec::new(),
            handshake_secret: Vec::new(),
            client_hs: None,
            server_hs: None,
            server_seq: 0,
            client_seq: 0,
            client_app: None,
            server_app: None,
            alpn_selected: None,
            hrr_done: false,
            saw_ee: false,
            saw_cert: false,
            saw_cv: false,
            cert_verified: false,
            leaf_spki: None,
            trust_roots: None,
            psk: config.psk.clone(),
            psk_accepted: false,
            resumption_master_secret: Vec::new(),
        })
    }

    fn new(config: &TlsConfig) -> Result<Self, TlsError> {
        let mut client_random = [0u8; 32];
        random_bytes(&mut client_random)?;
        let (secret, _) = x25519::keypair()?;
        let mut session_id = [0u8; 32];
        random_bytes(&mut session_id)?;
        Self::new_with_secrets(config, client_random, secret, session_id.to_vec())
    }

    /// Build the ClientHello, append it to the transcript, return the bytes.
    fn client_hello(&mut self) -> Result<Vec<u8>, TlsError> {
        let psk = self.psk.as_ref().map(|p| (p, &self.transcript));
        let msg = build_client_hello_msg(
            &self.profile,
            &self.server_name,
            &self.alpn,
            &self.client_random,
            &self.session_id,
            self.grease,
            &self.x25519_secret,
            None,
            psk,
            &mut self.mlkem_dk,
        )?;
        self.transcript.push(&msg);
        Ok(msg)
    }

    /// Process a ServerHello (or HelloRetryRequest) handshake message,
    /// including its 4-byte header.
    fn handle_server_hello(&mut self, msg: &[u8]) -> Result<HelloAction, TlsError> {
        const CTX: &str = "ServerHello";
        let body = msg.get(4..).ok_or(TlsError::BadHandshake(CTX))?;
        let mut r = Reader::new(body);
        let legacy_version = r.u16(CTX)?;
        if legacy_version != LEGACY_HANDSHAKE_VERSION {
            return Err(TlsError::BadHandshake(CTX));
        }
        let random: &[u8; 32] = r
            .bytes(32, CTX)?
            .try_into()
            .map_err(|_| TlsError::BadHandshake(CTX))?;
        // Note (m3): the TLS 1.3 downgrade sentinel (last 8 bytes of
        // ServerHello.random == "DOWNGRD\x01") is deliberately not checked.
        // The sentinel only protects negotiating TLS 1.2 with a server that
        // also supports 1.3, and this client fails closed on every 1.2
        // ServerHello (UnsupportedVersion), so there is no downgrade path
        // to protect. No behavior change.
        let session_echo = r.vec_u8(CTX)?;
        let cipher = r.u16(CTX)?;
        let compression = r.u8(CTX)?;
        if compression != 0 {
            return Err(TlsError::BadHandshake(CTX));
        }
        let ext_bytes = r.vec_u16(CTX)?;
        if !r.is_empty() {
            return Err(TlsError::BadHandshake(CTX));
        }
        if session_echo != self.session_id.as_slice() {
            return Err(TlsError::BadHandshake("session id echo mismatch"));
        }

        if random == &HRR_MAGIC {
            return self.handle_hello_retry(msg, cipher, ext_bytes);
        }

        if !self.profile.cipher_suites.contains(&cipher) {
            return Err(TlsError::UnsupportedCipherSuite(cipher));
        }
        let exts = parse_sh_extensions(ext_bytes)?;
        match exts.version {
            Some(TLS13_VERSION) => {}
            // The TLS 1.2 state machine is not implemented; fail closed.
            _ => return Err(TlsError::UnsupportedVersion),
        }
        if self.hrr_done {
            match self.cipher {
                Some(spec) if spec.id == cipher => {}
                _ => return Err(TlsError::BadHandshake("cipher suite changed after HRR")),
            }
        }
        let spec = cipher_spec(cipher).ok_or(TlsError::UnsupportedCipherSuite(cipher))?;
        let (group, ke) = exts
            .key_share
            .ok_or(TlsError::BadHandshake("ServerHello missing key_share"))?;
        // Key exchange per named group: X25519 carries a 32-byte public
        // key; X25519MLKEM768 (draft-ietf-tls-hybrid-design) carries the
        // 32-byte X25519 public key followed by the 1088-byte ML-KEM
        // ciphertext, and the shared secret is the 64-byte concatenation
        // of the two per-KEM shared secrets.
        let mut shared: Vec<u8> = match group {
            GROUP_X25519 => {
                if ke.len() != 32 {
                    return Err(TlsError::BadHandshake("bad key_share length"));
                }
                let mut peer = [0u8; 32];
                peer.copy_from_slice(&ke);
                let mut shared = x25519::shared_secret(&self.x25519_secret, &peer);
                // RFC 8446 §7.4.1: an all-zero (EC)DHE shared secret means
                // the key exchange failed (e.g. a low-order or zero peer
                // point). Abort instead of deriving traffic keys from a
                // known value.
                if shared == [0u8; 32] {
                    return Err(TlsError::BadHandshake("all-zero key_share secret"));
                }
                let out = shared.to_vec();
                zeroize(&mut shared);
                out
            }
            GROUP_X25519MLKEM768 => {
                if ke.len() != 32 + ml_kem_768::CIPHERTEXT_LEN {
                    return Err(TlsError::BadHandshake("bad hybrid key_share length"));
                }
                let mut peer = [0u8; 32];
                peer.copy_from_slice(&ke[..32]);
                let mut x_shared = x25519::shared_secret(&self.x25519_secret, &peer);
                if x_shared == [0u8; 32] {
                    return Err(TlsError::BadHandshake("all-zero key_share secret"));
                }
                let mut ct = [0u8; ml_kem_768::CIPHERTEXT_LEN];
                ct.copy_from_slice(&ke[32..]);
                let dk = self.mlkem_dk.as_ref().ok_or(TlsError::BadHandshake(
                    "server selected hybrid group with no hybrid key share offered",
                ))?;
                let mut m_shared = ml_kem_768::decaps(dk, &ct);
                let mut shared = Vec::with_capacity(64);
                shared.extend_from_slice(&x_shared);
                shared.extend_from_slice(&m_shared);
                zeroize(&mut x_shared);
                zeroize(&mut m_shared);
                shared
            }
            other => return Err(TlsError::UnsupportedGroup(other)),
        };
        // RFC 8446 §4.2.11: the server accepts our PSK by echoing
        // pre_shared_key with the selected identity. Only identity 0
        // exists (we offer exactly one), and the server must not select a
        // suite with a different hash than the ticket's.
        let offered_psk: Option<PskOffer> = match exts.selected_psk {
            Some(i) => {
                // Cloned so the flag assignment below does not fight the
                // borrow.
                let offer = self.psk.clone().ok_or(TlsError::BadHandshake(
                    "server selected a PSK we did not offer",
                ))?;
                if i != 0 {
                    return Err(TlsError::BadHandshake(
                        "server selected unknown PSK identity",
                    ));
                }
                if offer.cipher_suite != cipher {
                    return Err(TlsError::BadHandshake(
                        "server accepted PSK with a different cipher suite",
                    ));
                }
                self.psk_accepted = true;
                Some(offer)
            }
            None => None,
        };

        self.cipher = Some(spec);
        self.transcript.seal(spec.hash);
        self.transcript.push(msg);
        self.derive_handshake_keys(
            &shared,
            offered_psk.as_ref().map(|offer| offer.psk.as_slice()),
        )?;
        // The (EC)DHE shared secret is consumed by the key schedule;
        // scrub it so it does not linger. The offered PSK is scrubbed by
        // `PskOffer`'s own `Drop` when `offered_psk` goes out of scope.
        zeroize(shared.as_mut_slice());
        drop(offered_psk);
        Ok(HelloAction::Proceed)
    }

    /// Process a HelloRetryRequest (a ServerHello with the magic random).
    fn handle_hello_retry(
        &mut self,
        msg: &[u8],
        cipher: u16,
        ext_bytes: &[u8],
    ) -> Result<HelloAction, TlsError> {
        const CTX: &str = "HelloRetryRequest";
        if self.hrr_done {
            return Err(TlsError::UnexpectedMessage("second HelloRetryRequest"));
        }
        if !self.profile.cipher_suites.contains(&cipher) {
            return Err(TlsError::UnsupportedCipherSuite(cipher));
        }
        let spec = cipher_spec(cipher).ok_or(TlsError::UnsupportedCipherSuite(cipher))?;
        let hash = spec.hash;

        // Only supported_versions, key_share and cookie may appear (RFC 8446 §4.1.4).
        let mut saw_sv = false;
        let mut selected_group = None;
        let mut seen = Vec::new();
        let mut er = Reader::new(ext_bytes);
        while !er.is_empty() {
            let id = er.u16(CTX)?;
            if seen.contains(&id) {
                return Err(TlsError::BadHandshake("duplicate HRR extension"));
            }
            seen.push(id);
            let data = er.vec_u16(CTX)?;
            match id {
                EXT_SUPPORTED_VERSIONS => {
                    if data != [0x03, 0x04] {
                        return Err(TlsError::UnsupportedVersion);
                    }
                    saw_sv = true;
                }
                EXT_KEY_SHARE => {
                    // Note: in HRR this is just the selected group, no vector.
                    let mut kr = Reader::new(data);
                    let g = kr.u16(CTX)?;
                    if !kr.is_empty() {
                        return Err(TlsError::BadHandshake(CTX));
                    }
                    selected_group = Some(g);
                }
                // The client never echoes cookies (no cookie support yet),
                // so a cookie-carrying HRR is rejected outright.
                EXT_COOKIE => return Err(TlsError::HrrCookieUnsupported),
                _ => return Err(TlsError::UnexpectedMessage("forbidden HRR extension")),
            }
        }
        if !saw_sv {
            return Err(TlsError::BadHandshake("HRR without supported_versions"));
        }
        let group = selected_group.ok_or(TlsError::BadHandshake("HRR without key_share"))?;
        // RFC 8446 §4.1.4: the server must not select a group the client
        // already offered a key share for (absent a cookie, which we
        // reject above).
        if self.profile.key_share_groups.contains(&group) {
            return Err(TlsError::BadHandshake(
                "HRR selected an already-offered group",
            ));
        }
        if group != GROUP_X25519 && group != GROUP_X25519MLKEM768 {
            return Err(TlsError::UnsupportedGroup(group));
        }

        // Transcript becomes message_hash || HRR (RFC 8446 §4.4.1),
        // hashed incrementally from here on.
        let ch1_hash = self.transcript.digest(hash);
        self.transcript.reset_for_hrr(hash, &ch1_hash);
        self.transcript.push(msg);
        self.cipher = Some(spec);
        self.hrr_done = true;

        // Fresh key share for the selected group, then a second ClientHello
        // whose key_share holds only that group. A PSK offer rides along
        // with a fresh binder over the HRR-adjusted transcript.
        let (secret, _) = x25519::keypair()?;
        self.x25519_secret = secret;
        // A stale hybrid decapsulation key must not linger: wipe it
        // before the second ClientHello (which installs a fresh one
        // when the selected group is hybrid).
        if let Some(dk) = self.mlkem_dk.as_mut() {
            zeroize(dk);
        }
        self.mlkem_dk = None;
        let psk = self.psk.as_ref().map(|p| (p, &self.transcript));
        let ch2 = build_client_hello_msg(
            &self.profile,
            &self.server_name,
            &self.alpn,
            &self.client_random,
            &self.session_id,
            self.grease,
            &self.x25519_secret,
            Some(group),
            psk,
            &mut self.mlkem_dk,
        )?;
        self.transcript.push(&ch2);
        Ok(HelloAction::RetryHello(ch2))
    }

    fn derive_handshake_keys(
        &mut self,
        shared_secret: &[u8],
        psk: Option<&[u8]>,
    ) -> Result<(), TlsError> {
        let spec = self
            .cipher
            .ok_or(TlsError::Internal("no negotiated cipher"))?;
        let hash = spec.hash;
        let zeros = [0u8; 64];
        let zeros = &zeros[..hash_len(hash)];
        // RFC 8446 §7.1: the early secret mixes in the PSK when resuming;
        // a full handshake extracts from zeros.
        let mut early = match psk {
            Some(p) => {
                if p.len() != hash_len(hash) {
                    return Err(TlsError::Internal("psk has wrong length for hash"));
                }
                hkdf_extract(hash, zeros, p)
            }
            None => hkdf_extract(hash, zeros, zeros),
        };
        let mut derived = derive_secret(hash, &early, b"derived", &transcript_hash(hash, b""))?;
        zeroize(early.as_mut_slice());
        let hs_secret = hkdf_extract(hash, &derived, shared_secret);
        zeroize(derived.as_mut_slice());
        let thash = self.transcript.digest(hash);
        self.client_hs_traffic = derive_secret(hash, &hs_secret, b"c hs traffic", &thash)?;
        self.server_hs_traffic = derive_secret(hash, &hs_secret, b"s hs traffic", &thash)?;
        self.client_hs = Some(traffic_keys(hash, &self.client_hs_traffic, &spec)?);
        self.server_hs = Some(traffic_keys(hash, &self.server_hs_traffic, &spec)?);
        self.handshake_secret = hs_secret;
        Ok(())
    }

    /// Process EncryptedExtensions; records the negotiated ALPN protocol.
    /// Must be the first message of the encrypted flight (m4).
    fn handle_encrypted_extensions(&mut self, msg: &[u8]) -> Result<(), TlsError> {
        const CTX: &str = "EncryptedExtensions";
        if self.saw_ee {
            return Err(TlsError::UnexpectedMessage("duplicate EncryptedExtensions"));
        }
        self.saw_ee = true;
        self.transcript.push(msg);
        let body = msg.get(4..).ok_or(TlsError::BadHandshake(CTX))?;
        let mut r = Reader::new(body);
        let ext_bytes = r.vec_u16(CTX)?;
        if !r.is_empty() {
            return Err(TlsError::BadHandshake(CTX));
        }
        let mut seen = Vec::new();
        let mut er = Reader::new(ext_bytes);
        while !er.is_empty() {
            let id = er.u16(CTX)?;
            if seen.contains(&id) {
                return Err(TlsError::BadHandshake("duplicate extension"));
            }
            seen.push(id);
            let data = er.vec_u16(CTX)?;
            if id == EXT_ALPN {
                let mut ar = Reader::new(data);
                let protos = ar.vec_u16(CTX)?;
                let mut pr = Reader::new(protos);
                let sel = pr.vec_u8(CTX)?;
                if !pr.is_empty() || !ar.is_empty() {
                    return Err(TlsError::BadHandshake(CTX));
                }
                let s = std::str::from_utf8(sel).map_err(|_| TlsError::BadHandshake(CTX))?;
                if !self.alpn.iter().any(|p| p == s) {
                    return Err(TlsError::BadHandshake("server picked unoffered ALPN"));
                }
                self.alpn_selected = Some(s.to_string());
            }
        }
        Ok(())
    }

    /// Process the Certificate message: parse the chain, validate it
    /// against the trust roots (RFC 5280) and the expected hostname
    /// (RFC 6125), and retain the leaf public key for CertificateVerify.
    /// Must follow EncryptedExtensions and precede CertificateVerify.
    /// A chain that does not verify fails closed here, before any
    /// CertificateVerify is accepted.
    fn handle_certificate(&mut self, msg: &[u8]) -> Result<(), TlsError> {
        const CTX: &str = "Certificate";
        if !self.saw_ee || self.saw_cert || self.saw_cv {
            return Err(TlsError::UnexpectedMessage(
                "Certificate out of order in server flight",
            ));
        }
        self.saw_cert = true;
        self.transcript.push(msg);
        let body = msg.get(4..).ok_or(TlsError::BadHandshake(CTX))?;
        let mut r = Reader::new(body);
        // RFC 8446 §4.4.2: request_context MUST be zero length in a
        // server Certificate; a non-empty value is a protocol violation.
        let request_context = r.vec_u8(CTX)?;
        if !request_context.is_empty() {
            return Err(TlsError::BadHandshake(
                "non-empty request_context in server Certificate",
            ));
        }
        let list = r.vec_u24(CTX)?;
        // Retain every certificate's DER; the chain is validated below.
        let mut lr = Reader::new(list);
        let mut chain: Vec<&[u8]> = Vec::new();
        while !lr.is_empty() {
            let cert = lr.vec_u24(CTX)?;
            let _exts = lr.vec_u16(CTX)?;
            chain.push(cert);
        }
        if !r.is_empty() {
            return Err(TlsError::BadHandshake(CTX));
        }
        if chain.is_empty() {
            return Err(TlsError::CertificateVerificationFailed(
                "server sent an empty certificate list".to_string(),
            ));
        }
        let default_roots;
        let roots: &[x509::ParsedCert] = match &self.trust_roots {
            Some(r) => r,
            None => {
                default_roots = x509::root_store();
                default_roots
            }
        };
        match x509::verify_server_chain(&chain, &self.server_name, SystemTime::now(), roots) {
            Ok(leaf) => {
                self.leaf_spki = Some(leaf.spki);
            }
            Err(e) => {
                return Err(TlsError::CertificateVerificationFailed(e.to_string()));
            }
        }
        Ok(())
    }

    /// Process CertificateVerify: check the server's signature over the
    /// transcript (RFC 8446 §4.4.3) against the leaf public key retained
    /// from the Certificate message. The signature covers the transcript
    /// *before* this message, so verification happens before the message
    /// is appended. Must follow Certificate and precede Finished (m4).
    fn handle_certificate_verify(&mut self, msg: &[u8]) -> Result<(), TlsError> {
        const CTX: &str = "CertificateVerify";
        if !self.saw_cert || self.saw_cv {
            return Err(TlsError::UnexpectedMessage(
                "CertificateVerify out of order in server flight",
            ));
        }
        let body = msg.get(4..).ok_or(TlsError::BadHandshake(CTX))?;
        let mut r = Reader::new(body);
        let scheme = r.u16(CTX)?;
        let sig = r.vec_u16(CTX)?;
        if !r.is_empty() {
            return Err(TlsError::BadHandshake(CTX));
        }
        let spki = self
            .leaf_spki
            .as_ref()
            .ok_or(TlsError::Internal("no leaf key for CertificateVerify"))?;
        // The signed content: 64 spaces, the ASCII label, a zero byte,
        // then the transcript hash *without* this message.
        let spec = self
            .cipher
            .ok_or(TlsError::Internal("no negotiated cipher"))?;
        let mut content = vec![b' '; 64];
        content.extend_from_slice(b"TLS 1.3, server CertificateVerify");
        content.push(0x00);
        content.extend_from_slice(&self.transcript.digest(spec.hash));
        if let Err(e) = x509::verify_tls_scheme(spki, scheme, &content, sig) {
            return Err(TlsError::CertificateVerificationFailed(e.to_string()));
        }
        self.saw_cv = true;
        self.transcript.push(msg);
        // Both halves of server authentication are now done: the chain
        // validated in `handle_certificate`, the signature here.
        self.cert_verified = true;
        Ok(())
    }

    /// Verify the server Finished, derive application traffic keys, and
    /// build our Finished message (returned with its 4-byte header).
    ///
    /// Must be the last message of the encrypted flight (m4), and the
    /// flight must have carried Certificate + CertificateVerify (C1).
    fn handle_finished(&mut self, msg: &[u8]) -> Result<Vec<u8>, TlsError> {
        const CTX: &str = "Finished";
        let spec = self
            .cipher
            .ok_or(TlsError::Internal("no negotiated cipher"))?;
        let hash = spec.hash;
        if !self.saw_ee {
            return Err(TlsError::UnexpectedMessage(
                "Finished before EncryptedExtensions",
            ));
        }
        let body = msg.get(4..).ok_or(TlsError::BadHandshake(CTX))?;

        // verify_data covers the transcript *before* this Finished message.
        let thash = self.transcript.digest(hash);
        let mut fkey = finished_key(hash, &self.server_hs_traffic)?;
        let expected = hmac(hash, &fkey, &thash);
        zeroize(fkey.as_mut_slice());
        if body.len() != expected.len() || !ct_eq(&expected, body) {
            return Err(TlsError::BadHandshake("Finished verify_data mismatch"));
        }

        // Server authentication (C1): the flight must have presented
        // Certificate + CertificateVerify, the chain must have validated
        // against the trust roots and the hostname, and the
        // CertificateVerify signature must have verified. A flight that
        // skips authentication is rejected instead of completing
        // unauthenticated — unless the server accepted our PSK offer, in
        // which case the Finished MAC over PSK-derived keys *is* the
        // authentication: only the ticket issuer knows the PSK.
        let authenticated =
            (self.saw_cert && self.saw_cv && self.cert_verified) || self.psk_accepted;
        if !authenticated {
            return Err(TlsError::CertificateVerificationFailed(
                "server flight was not authenticated".to_string(),
            ));
        }

        self.transcript.push(msg);

        // Application traffic secrets (transcript through server Finished).
        let thash2 = self.transcript.digest(hash);
        let mut derived = derive_secret(
            hash,
            &self.handshake_secret,
            b"derived",
            &transcript_hash(hash, b""),
        )?;
        let zeros = [0u8; 64];
        let mut master = hkdf_extract(hash, &derived, &zeros[..hash_len(hash)]);
        zeroize(derived.as_mut_slice());
        let mut client_app_traffic = derive_secret(hash, &master, b"c ap traffic", &thash2)?;
        let mut server_app_traffic = derive_secret(hash, &master, b"s ap traffic", &thash2)?;
        self.client_app = Some(traffic_keys(hash, &client_app_traffic, &spec)?);
        self.server_app = Some(traffic_keys(hash, &server_app_traffic, &spec)?);
        zeroize(client_app_traffic.as_mut_slice());
        zeroize(server_app_traffic.as_mut_slice());

        let mut cfin_key = finished_key(hash, &self.client_hs_traffic)?;
        let verify_data = hmac(hash, &cfin_key, &thash2);
        zeroize(cfin_key.as_mut_slice());
        let mut out = Vec::with_capacity(4 + verify_data.len());
        put_handshake_header(&mut out, HS_FINISHED, &verify_data);
        self.transcript.push(&out);
        // Retained for post-handshake NewSessionTicket PSK derivation.
        // RFC 8446 §7.1: the transcript runs through client Finished, so
        // the derivation happens *after* the Finished above is pushed.
        let thash3 = self.transcript.digest(hash);
        self.resumption_master_secret = derive_secret(hash, &master, b"res master", &thash3)?;
        zeroize(master.as_mut_slice());
        Ok(out)
    }

    /// Seal the client Finished and reserve the next record sequence
    /// number — *before* the write that follows. Reusing a (key, nonce)
    /// pair in AES-GCM breaks the AEAD (keystream reuse, GHASH key
    /// recovery), so `client_seq` must advance the moment the Finished is
    /// sealed: if `write_all` (or any later step) fails,
    /// `send_failure_alert` seals its alert under the next nonce instead
    /// of repeating the Finished's.
    fn seal_client_finished(&mut self, finished: &[u8]) -> Result<Vec<u8>, TlsError> {
        let keys = self
            .client_hs
            .as_ref()
            .ok_or(TlsError::Internal("no keys"))?;
        let enc = seal_record(keys, self.client_seq, RECORD_HANDSHAKE, finished)?;
        // RFC 8446 §5.3: the sequence number must never wrap — reusing a
        // nonce would break the AEAD. Fail closed on exhaustion
        // (unreachable in practice, cheap to assert).
        self.client_seq = self
            .client_seq
            .checked_add(1)
            .ok_or(TlsError::BadRecord("record sequence exhausted"))?;
        Ok(enc)
    }
}

/// Parsed ServerHello extensions we care about.
struct ServerHelloExts {
    version: Option<u16>,
    key_share: Option<(u16, Vec<u8>)>,
    /// The server's accepted PSK identity (RFC 8446 §4.2.11), if it
    /// selected one of our offered identities.
    selected_psk: Option<u16>,
}

/// Parse ServerHello extensions: supported_versions, key_share, and
/// pre_shared_key. Duplicate extensions are rejected (RFC 8446 §4.2: an
/// extension must not appear more than once).
fn parse_sh_extensions(ext_bytes: &[u8]) -> Result<ServerHelloExts, TlsError> {
    const CTX: &str = "ServerHello extensions";
    let mut version = None;
    let mut key_share = None;
    let mut selected_psk = None;
    let mut seen = Vec::new();
    let mut er = Reader::new(ext_bytes);
    while !er.is_empty() {
        let id = er.u16(CTX)?;
        if seen.contains(&id) {
            return Err(TlsError::BadHandshake("duplicate ServerHello extension"));
        }
        seen.push(id);
        let data = er.vec_u16(CTX)?;
        match id {
            EXT_SUPPORTED_VERSIONS => {
                let mut vr = Reader::new(data);
                version = Some(vr.u16(CTX)?);
                if !vr.is_empty() {
                    return Err(TlsError::BadHandshake(CTX));
                }
            }
            EXT_KEY_SHARE => {
                let mut kr = Reader::new(data);
                let group = kr.u16(CTX)?;
                let ke = kr.vec_u16(CTX)?.to_vec();
                if !kr.is_empty() {
                    return Err(TlsError::BadHandshake(CTX));
                }
                key_share = Some((group, ke));
            }
            EXT_PRE_SHARED_KEY => {
                let mut pr = Reader::new(data);
                let selected = pr.u16(CTX)?;
                if !pr.is_empty() {
                    return Err(TlsError::BadHandshake(CTX));
                }
                selected_psk = Some(selected);
            }
            _ => {}
        }
    }
    Ok(ServerHelloExts {
        version,
        key_share,
        selected_psk,
    })
}

fn parse_alert(fragment: &[u8]) -> TlsError {
    if fragment.len() == 2 {
        TlsError::AlertReceived(fragment[1])
    } else {
        TlsError::BadRecord("malformed alert")
    }
}

/// A parsed `NewSessionTicket` message (RFC 8446 §4.6.1).
struct NewSessionTicketMsg {
    /// Ticket lifetime, seconds.
    lifetime: u32,
    /// `ticket_age_add` for obfuscating the ticket age.
    age_add: u32,
    /// Per-ticket nonce, mixed into the PSK derivation.
    nonce: Vec<u8>,
    /// The opaque ticket: the PSK identity offered back on resumption.
    ticket: Vec<u8>,
}

/// Parse a NewSessionTicket message, retaining the ticket and the
/// metadata needed to derive its PSK.
fn parse_new_session_ticket(msg: &[u8]) -> Result<NewSessionTicketMsg, TlsError> {
    const CTX: &str = "NewSessionTicket";
    let body = msg.get(4..).ok_or(TlsError::BadHandshake(CTX))?;
    let mut r = Reader::new(body);
    let lifetime = u32::from_be_bytes(
        r.bytes(4, CTX)?
            .try_into()
            .map_err(|_| TlsError::BadHandshake(CTX))?,
    );
    let age_add = u32::from_be_bytes(
        r.bytes(4, CTX)?
            .try_into()
            .map_err(|_| TlsError::BadHandshake(CTX))?,
    );
    let nonce = r.vec_u8(CTX)?.to_vec();
    let ticket = r.vec_u16(CTX)?.to_vec();
    let _exts = r.vec_u16(CTX)?;
    if !r.is_empty() {
        return Err(TlsError::BadHandshake(CTX));
    }
    if ticket.is_empty() {
        return Err(TlsError::BadHandshake("empty ticket in NewSessionTicket"));
    }
    Ok(NewSessionTicketMsg {
        lifetime,
        age_add,
        nonce,
        ticket,
    })
}

// ---------------------------------------------------------------------------
// TlsConnection: established channel over TcpStream
// ---------------------------------------------------------------------------

/// Move buffered decrypted bytes into `buf`; returns bytes moved.
fn drain_pending(buf: &mut [u8], pending: &mut Vec<u8>) -> usize {
    let n = buf.len().min(pending.len());
    buf[..n].copy_from_slice(&pending[..n]);
    pending.drain(..n);
    n
}

/// Classify an alert fragment for [`TlsConnection::read`]: `Ok(true)` is a
/// clean shutdown — warning-level (1) close_notify (0), RFC 8446 §6.1 —
/// `Ok(false)` is any other alert (fatal to the connection), `Err` is a
/// malformed fragment. A fatal alert with description 0 is malformed, not
/// a clean shutdown (m8).
fn is_clean_close_notify(fragment: &[u8]) -> io::Result<bool> {
    if fragment.len() != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "tls: malformed alert",
        ));
    }
    Ok(fragment[0] == 1 && fragment[1] == ALERT_CLOSE_NOTIFY)
}

/// An established TLS connection. Application data is encrypted/decrypted
/// with the negotiated traffic keys; handshake state is gone.
pub struct TlsConnection {
    stream: TcpStream,
    read_keys: TrafficKeys,
    read_seq: u64,
    write_keys: TrafficKeys,
    write_seq: u64,
    /// Decrypted application bytes not yet consumed by `read`.
    pending: Vec<u8>,
    framer: HandshakeFramer,
    alpn: Option<String>,
    cipher_suite: u16,
    closed: bool,
    /// `Derive-Secret(master_secret, "res master", "")`, for deriving
    /// the PSK of each post-handshake `NewSessionTicket`.
    resumption_master_secret: Vec<u8>,
    resumption_hash: HashAlg,
    /// Tickets received post-handshake, not yet handed to the caller via
    /// [`TlsConnection::take_received_tickets`].
    received_tickets: Vec<CachedTicket>,
    /// Raw record bytes read ahead by [`TlsConnection::probe`] but not
    /// yet parsed. Drained by the record reader before touching the
    /// socket, so a probe can never corrupt the stream.
    stash: Vec<u8>,
    /// Absolute wall-clock bound for blocking reads, set only by the
    /// H2 pool's take-time PING keepalive. One [`TlsConnection::read`]
    /// loops internally until a full TLS record arrives, so the
    /// per-read socket timeout the caller sets is not enough: a peer
    /// drip-feeding one record byte at a time would stretch a single
    /// read to (record bytes) x the socket timeout. The deadline is
    /// checked before every blocking socket read in
    /// [`TlsConnection::read_record_buffered`]; `None` (the default)
    /// leaves reads bounded only by the socket timeout.
    read_deadline: Option<Instant>,
}

/// Keys and metadata produced by a completed handshake.
struct Established {
    read_keys: TrafficKeys,
    write_keys: TrafficKeys,
    alpn: Option<String>,
    cipher_suite: u16,
    resumption_master_secret: Vec<u8>,
    resumption_hash: HashAlg,
}

/// A `TcpStream` with bytes queued ahead of the socket: `read` drains the
/// queue first, writes go straight to the socket. Lets bytes over-read
/// past a CONNECT tunnel's 2xx response feed the TLS handshake instead of
/// being dropped — dropping them would desync the handshake when the
/// proxy coalesced the response with the origin's ServerHello flight.
struct PrefixStream {
    prefix: Vec<u8>,
    pos: usize,
    stream: TcpStream,
}

impl PrefixStream {
    /// Split back into the socket and any prefix bytes the handshake did
    /// not consume (they belong to the post-handshake stream).
    fn into_parts(mut self) -> (TcpStream, Vec<u8>) {
        self.prefix.drain(..self.pos);
        (self.stream, self.prefix)
    }
}

impl io::Read for PrefixStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos < self.prefix.len() {
            let n = (self.prefix.len() - self.pos).min(buf.len());
            buf[..n].copy_from_slice(&self.prefix[self.pos..self.pos + n]);
            self.pos += n;
            return Ok(n);
        }
        self.stream.read(buf)
    }
}

impl io::Write for PrefixStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.stream.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stream.flush()
    }
}

impl TlsConnection {
    /// Run the TLS handshake over `stream` and return the established channel.
    ///
    /// # Security
    ///
    /// The peer is authenticated: the server's certificate chain is
    /// validated against the vendored Mozilla root store (RFC 5280), the
    /// leaf is checked against `config.server_name` (RFC 6125), and the
    /// CertificateVerify signature is verified over the transcript.
    /// Supported signature schemes are RSA PKCS#1 v1.5 (SHA-256/384/512)
    /// and ECDSA P-256/SHA-256, P-384/SHA-384. A handshake that fails any
    /// of these checks aborts with
    /// [`TlsError::CertificateVerificationFailed`].
    ///
    /// # Errors
    ///
    /// Returns [`TlsError`] if the handshake fails; a fatal alert is sent on
    /// a best-effort basis before returning.
    pub fn connect(stream: TcpStream, config: &TlsConfig) -> Result<TlsConnection, TlsError> {
        let mut stream = stream;
        match Self::run_handshake(&mut stream, config) {
            Ok(est) => Ok(TlsConnection {
                stream,
                read_keys: est.read_keys,
                read_seq: 0,
                write_keys: est.write_keys,
                write_seq: 0,
                pending: Vec::new(),
                framer: HandshakeFramer::new(),
                alpn: est.alpn,
                cipher_suite: est.cipher_suite,
                closed: false,
                resumption_master_secret: est.resumption_master_secret,
                resumption_hash: est.resumption_hash,
                received_tickets: Vec::new(),
                stash: Vec::new(),
                read_deadline: None,
            }),
            Err(e) => Err(e),
        }
    }

    /// Run the TLS handshake over `stream` with `prefix` bytes queued ahead
    /// of the socket, and return the established channel.
    ///
    /// This is [`connect`](Self::connect) for the case where the caller
    /// already read past the point the handshake starts — e.g. a CONNECT
    /// tunnel whose 2xx response was coalesced with the origin's TLS
    /// flight. Prefix bytes the handshake does not consume are stashed
    /// for the record reader, so nothing is lost or duplicated.
    ///
    /// # Errors
    ///
    /// Same as [`connect`](Self::connect).
    pub fn connect_with_prefix(
        stream: TcpStream,
        prefix: Vec<u8>,
        config: &TlsConfig,
    ) -> Result<TlsConnection, TlsError> {
        let mut transport = PrefixStream {
            prefix,
            pos: 0,
            stream,
        };
        match Self::run_handshake(&mut transport, config) {
            Ok(est) => {
                let (stream, stash) = transport.into_parts();
                Ok(TlsConnection {
                    stream,
                    read_keys: est.read_keys,
                    read_seq: 0,
                    write_keys: est.write_keys,
                    write_seq: 0,
                    pending: Vec::new(),
                    framer: HandshakeFramer::new(),
                    alpn: est.alpn,
                    cipher_suite: est.cipher_suite,
                    closed: false,
                    resumption_master_secret: est.resumption_master_secret,
                    resumption_hash: est.resumption_hash,
                    received_tickets: Vec::new(),
                    stash,
                    read_deadline: None,
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Handshake driver with failure-alert handling: the alert is sent
    /// encrypted once handshake keys are established (m9), plaintext before.
    fn run_handshake<S: io::Read + io::Write>(
        stream: &mut S,
        config: &TlsConfig,
    ) -> Result<Established, TlsError> {
        let mut hs = match Handshake::new(config) {
            Ok(hs) => hs,
            Err(e) => {
                Self::send_failure_alert(stream, None, &e);
                return Err(e);
            }
        };
        let result = Self::run_handshake_inner(stream, &mut hs);
        if let Err(ref e) = result {
            Self::send_failure_alert(stream, Some(&hs), e);
        }
        result
    }

    /// Best-effort fatal alert for a failed handshake. Once handshake keys
    /// are established, RFC 8446 §6 requires the alert to go out encrypted
    /// under the client handshake traffic keys; before that point only a
    /// plaintext alert is possible.
    fn send_failure_alert<S: io::Write>(stream: &mut S, hs: Option<&Handshake>, e: &TlsError) {
        let desc = match alert_for(e) {
            Some(d) => d,
            None => return,
        };
        if let Some(hs) = hs
            && let Some(keys) = hs.client_hs.as_ref()
        {
            if let Ok(rec) = seal_record(keys, hs.client_seq, RECORD_ALERT, &[2u8, desc]) {
                let _ = stream.write_all(&rec);
            }
            return;
        }
        let _ = write_record(stream, RECORD_ALERT, LEGACY_RECORD_VERSION, &[2u8, desc]);
    }

    fn run_handshake_inner<S: io::Read + io::Write>(
        stream: &mut S,
        hs: &mut Handshake,
    ) -> Result<Established, TlsError> {
        let record_version = hs.profile.record_version;

        let ch = hs.client_hello()?;
        write_record(stream, RECORD_HANDSHAKE, record_version, &ch)?;

        let mut framer = HandshakeFramer::new();
        let mut action = Self::read_server_hello(stream, &mut *hs, &mut framer)?;
        if let HelloAction::RetryHello(ch2) = action {
            write_record(stream, RECORD_HANDSHAKE, record_version, &ch2)?;
            action = Self::read_server_hello(stream, &mut *hs, &mut framer)?;
            if matches!(action, HelloAction::RetryHello(_)) {
                return Err(TlsError::Internal("unexpected HRR after retry"));
            }
        }

        // Encrypted server flight: EncryptedExtensions, then Certificate,
        // CertificateVerify, Finished — in that order (enforced by the
        // handle_* methods). When the server accepted our PSK offer the
        // flight is abbreviated to EncryptedExtensions, Finished: the
        // Finished MAC over PSK-derived keys is the authentication.
        // Authentication is enforced: a full flight must carry
        // Certificate + CertificateVerify, the chain must validate
        // against the trust roots and the hostname, and the
        // CertificateVerify signature must verify. handle_finished rejects
        // a flight that skips any of this, so the handshake never
        // completes unauthenticated.
        let mut ccs_seen = 0u8;
        let mut stalled = 0u8;
        let client_finished = loop {
            let mut rec = read_record(stream)?;
            match rec[0] {
                RECORD_APP_DATA => {
                    let keys = hs.server_hs.as_ref().ok_or(TlsError::Internal("no keys"))?;
                    let (inner, pt) = open_record(keys, hs.server_seq, &mut rec)?;
                    // RFC 8446 §5.3: the sequence number must never wrap —
                    // reusing a nonce would break the AEAD. Fail closed on
                    // exhaustion (unreachable in practice, cheap to assert).
                    hs.server_seq = hs
                        .server_seq
                        .checked_add(1)
                        .ok_or(TlsError::BadRecord("record sequence exhausted"))?;
                    match inner {
                        RECORD_HANDSHAKE => {
                            let had_bytes = !pt.is_empty();
                            framer.push(pt)?;
                            let mut done = None;
                            let mut message_completed = false;
                            while let Some((t, m)) = framer.next_message()? {
                                // A consumed message is progress even
                                // though it drains the framer.
                                message_completed = true;
                                match t {
                                    HS_ENCRYPTED_EXTENSIONS => {
                                        hs.handle_encrypted_extensions(&m)?;
                                    }
                                    HS_CERTIFICATE => hs.handle_certificate(&m)?,
                                    HS_CERTIFICATE_VERIFY => hs.handle_certificate_verify(&m)?,
                                    HS_FINISHED => {
                                        done = Some(hs.handle_finished(&m)?);
                                        break;
                                    }
                                    _ => {
                                        return Err(TlsError::UnexpectedMessage(
                                            "unexpected handshake message",
                                        ));
                                    }
                                }
                            }
                            if let Some(cf) = done {
                                // RFC 8446 §4: no handshake message may
                                // follow Finished in the same flight — a
                                // server that sends trailing handshake
                                // bytes is a protocol violation, so fail
                                // closed instead of silently dropping them.
                                if !framer.is_empty() {
                                    return Err(TlsError::UnexpectedMessage(
                                        "trailing handshake data after Finished",
                                    ));
                                }
                                break cf;
                            }
                            // A peer flooding empty handshake records
                            // would otherwise spin this loop forever.
                            // Records that completed a message reset the
                            // count instead: consumed bytes are progress.
                            if message_completed {
                                stalled = 0;
                            } else {
                                note_fragment_progress(had_bytes, &mut stalled)?;
                            }
                        }
                        RECORD_ALERT => return Err(parse_alert(pt)),
                        _ => return Err(TlsError::UnexpectedMessage("unexpected content type")),
                    }
                }
                RECORD_ALERT => return Err(parse_alert(&rec[5..])),
                RECORD_CHANGE_CIPHER_SPEC => {
                    if !valid_dummy_ccs(&rec) {
                        return Err(TlsError::BadRecord("malformed change_cipher_spec"));
                    }
                    // A peer flooding CCS records would otherwise keep this
                    // loop alive forever (m6).
                    ccs_seen += 1;
                    if ccs_seen > MAX_IGNORED_CCS {
                        return Err(TlsError::UnexpectedMessage(
                            "too many change_cipher_spec records",
                        ));
                    }
                }
                _ => return Err(TlsError::UnexpectedMessage("unexpected record type")),
            }
        };

        let enc = hs.seal_client_finished(&client_finished)?;
        stream.write_all(&enc).map_err(TlsError::Io)?;

        let spec = hs.cipher.ok_or(TlsError::Internal("no cipher"))?;
        Ok(Established {
            read_keys: hs
                .server_app
                .take()
                .ok_or(TlsError::Internal("no app keys"))?,
            write_keys: hs
                .client_app
                .take()
                .ok_or(TlsError::Internal("no app keys"))?,
            alpn: hs.alpn_selected.take(),
            cipher_suite: spec.id,
            resumption_master_secret: std::mem::take(&mut hs.resumption_master_secret),
            resumption_hash: spec.hash,
        })
    }

    /// Read the (plaintext) ServerHello / HelloRetryRequest record(s).
    fn read_server_hello<S: io::Read + io::Write>(
        stream: &mut S,
        hs: &mut Handshake,
        framer: &mut HandshakeFramer,
    ) -> Result<HelloAction, TlsError> {
        let mut ccs_seen = 0u8;
        let mut stalled = 0u8;
        loop {
            let rec = read_record(stream)?;
            match rec[0] {
                RECORD_HANDSHAKE => {
                    let had_bytes = !rec[5..].is_empty();
                    framer.push(&rec[5..])?;
                    if let Some((t, m)) = framer.next_message()? {
                        if t != HS_SERVER_HELLO {
                            return Err(TlsError::UnexpectedMessage("expected ServerHello"));
                        }
                        // The same framer is reused for the encrypted
                        // flight: leftover plaintext must never be fed to
                        // the encrypted parser (m5).
                        if !framer.is_empty() {
                            return Err(TlsError::BadHandshake("trailing bytes after ServerHello"));
                        }
                        return hs.handle_server_hello(&m);
                    }
                    // No complete message yet: bound records that add no
                    // reassembly progress so an empty-record flood cannot
                    // spin this loop forever.
                    note_fragment_progress(had_bytes, &mut stalled)?;
                }
                RECORD_ALERT => return Err(parse_alert(&rec[5..])),
                RECORD_CHANGE_CIPHER_SPEC => {
                    if !valid_dummy_ccs(&rec) {
                        return Err(TlsError::BadRecord("malformed change_cipher_spec"));
                    }
                    ccs_seen += 1;
                    if ccs_seen > MAX_IGNORED_CCS {
                        return Err(TlsError::UnexpectedMessage(
                            "too many change_cipher_spec records",
                        ));
                    }
                }
                _ => return Err(TlsError::UnexpectedMessage("unexpected record type")),
            }
        }
    }

    /// Read decrypted application data. Returns `Ok(0)` on clean shutdown.
    ///
    /// Decrypted bytes are returned immediately: this never blocks with
    /// data already in hand, and never reports EOF while bytes are still
    /// buffered (C2).
    pub fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if !self.pending.is_empty() {
            return Ok(drain_pending(buf, &mut self.pending));
        }
        if self.closed {
            return Ok(0);
        }
        loop {
            let mut rec = self.read_record_buffered().map_err(tls_to_io)?;
            match rec[0] {
                RECORD_APP_DATA => {
                    let (inner, pt) =
                        open_record(&self.read_keys, self.read_seq, &mut rec).map_err(tls_to_io)?;
                    // RFC 8446 §5.3: the 64-bit sequence number must never
                    // wrap — reusing a nonce would break the AEAD. On
                    // exhaustion the connection fails closed instead.
                    self.read_seq = self.read_seq.checked_add(1).ok_or_else(|| {
                        self.closed = true;
                        io::Error::new(io::ErrorKind::InvalidData, "tls: record sequence exhausted")
                    })?;
                    match inner {
                        // Fast path: `pending` is empty here (drained at
                        // the top of read), so when the record fits it goes
                        // straight into the caller's buffer instead of
                        // bouncing through `pending` (one less copy and no
                        // buffer growth). Buffer, then return at once:
                        // looping back into read_record here would block
                        // instead of delivering the data (C2).
                        RECORD_APP_DATA => {
                            // RFC 8446 §5.1: an empty ApplicationData
                            // fragment is legal padding, not EOF. Skip it
                            // and keep reading; Ok(0) would misreport as
                            // EOF to callers (fetch UntilClose loop,
                            // Content-Length bodies).
                            if pt.is_empty() {
                                continue;
                            }
                            if pt.len() <= buf.len() {
                                buf[..pt.len()].copy_from_slice(pt);
                                return Ok(pt.len());
                            }
                            self.pending.extend_from_slice(pt);
                            return Ok(drain_pending(buf, &mut self.pending));
                        }
                        RECORD_HANDSHAKE => self.handle_post_handshake(pt).map_err(tls_to_io)?,
                        RECORD_ALERT => {
                            // `pending` is empty here (drained above), so
                            // Ok(0) cannot drop data (C2).
                            if is_clean_close_notify(pt)? {
                                self.closed = true;
                                return Ok(0);
                            }
                            return Err(io::Error::new(
                                io::ErrorKind::ConnectionAborted,
                                format!("tls: peer sent fatal alert {}", pt[1]),
                            ));
                        }
                        _ => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "tls: unexpected content type",
                            ));
                        }
                    }
                }
                RECORD_ALERT => {
                    // RFC 8446 §5.2/§6.2: after handshake completion every
                    // record MUST be AEAD-protected (outer type always
                    // application_data). An outer plaintext alert is an
                    // unexpected_message, fatal — honoring a plaintext
                    // close_notify as clean EOF enables a truncation
                    // attack, so the connection fails closed instead.
                    self.closed = true;
                    return Err(tls_to_io(TlsError::UnexpectedMessage(
                        "plaintext alert after handshake",
                    )));
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "tls: unexpected record type",
                    ));
                }
            }
        }
    }

    /// Read one full record, draining `stash` (bytes read ahead by
    /// [`TlsConnection::probe`]) before touching the socket.
    fn read_record_buffered(&mut self) -> Result<Vec<u8>, TlsError> {
        loop {
            if self.stash.len() >= 5 {
                let len = u16::from_be_bytes([self.stash[3], self.stash[4]]) as usize;
                if len > MAX_RECORD_LEN {
                    return Err(TlsError::BadRecord("record too large"));
                }
                if self.stash.len() >= 5 + len {
                    return Ok(take_stash_record(&mut self.stash, 5 + len));
                }
                // A peer drip-feeding a partial record must not grow the
                // stash without bound.
                if self.stash.len() >= 5 + MAX_RECORD_LEN {
                    return Err(TlsError::BadRecord("record too large"));
                }
            }
            // The deadline is checked only before a blocking socket
            // read: bytes already in the stash are returned without
            // blocking, and the outer `read` loop reaches this check on
            // every record it assembles (empty records, post-handshake
            // records), so no iteration can wait past the deadline plus
            // one per-read socket timeout.
            if let Some(deadline) = self.read_deadline
                && Instant::now() >= deadline
            {
                return Err(TlsError::ReadDeadline);
            }
            let mut tmp = [0u8; 8192];
            let n = self.stream.read(&mut tmp).map_err(map_eof)?;
            if n == 0 {
                return Err(TlsError::Eof);
            }
            self.stash.extend_from_slice(&tmp[..n]);
        }
    }

    /// Bound blocking reads on the underlying socket. Used by the H2
    /// pool's take-time PING keepalive, which shrinks the timeout for
    /// the round trip and restores it afterwards.
    pub fn set_read_timeout(&mut self, timeout: Option<Duration>) -> io::Result<()> {
        self.stream.set_read_timeout(timeout)
    }

    /// Bound blocking writes on the underlying socket. Used by the H2
    /// pool's take-time PING keepalive, which shrinks the timeout for
    /// the round trip and restores it afterwards.
    pub fn set_write_timeout(&mut self, timeout: Option<Duration>) -> io::Result<()> {
        self.stream.set_write_timeout(timeout)
    }

    /// Set an absolute wall-clock deadline for blocking reads. Checked
    /// before every blocking socket read in
    /// [`TlsConnection::read_record_buffered`]: the check runs before
    /// the read, not after, so one drip-fed read can still block up to
    /// the (shrunk) per-read socket timeout past the deadline. A peer
    /// drip-feeding one record byte at a time cannot stretch a single
    /// `read` past the deadline plus one per-read timeout, the way the
    /// resetting per-read socket timeout alone would allow. Used by the
    /// H2 pool's take-time PING keepalive alongside the timeout shrink,
    /// and cleared (`None`) on restore; no other caller should leave a
    /// deadline behind.
    pub fn set_read_deadline(&mut self, deadline: Option<Instant>) {
        self.read_deadline = deadline;
    }

    /// Best-effort liveness probe for pooled connection reuse. Never
    /// blocks: at most one non-blocking fill is attempted.
    ///
    /// Post-handshake messages already waiting on the socket (e.g.
    /// session tickets) are absorbed into the ticket queue. A
    /// `close_notify`, any other alert, any I/O or decryption failure, or
    /// unsolicited application data reports the connection dead; no
    /// waiting data reports it alive. Partial records stay in `stash` for
    /// the real read path, so probing can never corrupt the stream.
    pub fn probe(&mut self) -> bool {
        if self.closed || !self.pending.is_empty() {
            return false;
        }
        // Zero-duration read timeouts are rejected by the OS (EINVAL),
        // so toggle non-blocking mode instead: one read reports
        // WouldBlock on an idle socket. O_NONBLOCK and SO_RCVTIMEO are
        // independent flags, so the configured read timeout survives.
        if self.stream.set_nonblocking(true).is_err() {
            return false;
        }
        let alive = self.probe_once();
        // A stream left non-blocking would break the blocking read path;
        // if it cannot be restored, the connection is unusable.
        self.stream.set_nonblocking(false).is_ok() && alive
    }

    fn probe_once(&mut self) -> bool {
        // One non-blocking fill; WouldBlock means idle-but-alive.
        let mut tmp = [0u8; 8192];
        match self.stream.read(&mut tmp) {
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                return true;
            }
            Err(_) => return false,
            Ok(0) => return false,
            Ok(n) => self.stash.extend_from_slice(&tmp[..n]),
        }
        // Drain complete records; a partial tail waits for the real read.
        loop {
            if self.stash.len() < 5 {
                return true;
            }
            let len = u16::from_be_bytes([self.stash[3], self.stash[4]]) as usize;
            if len > MAX_RECORD_LEN {
                return false;
            }
            if self.stash.len() < 5 + len {
                return true;
            }
            let mut rec = take_stash_record(&mut self.stash, 5 + len);
            if !self.probe_record(&mut rec) {
                return false;
            }
        }
    }

    /// Classify one probed record: `true` if the connection is still
    /// usable. A dead connection is dropped by the caller either way, so
    /// consuming a read sequence number here is harmless.
    fn probe_record(&mut self, rec: &mut [u8]) -> bool {
        if rec[0] != RECORD_APP_DATA {
            // Plaintext records after the handshake are bogus.
            return false;
        }
        let (inner, pt) = match open_record(&self.read_keys, self.read_seq, rec) {
            Ok(x) => x,
            Err(_) => return false,
        };
        self.read_seq = match self.read_seq.checked_add(1) {
            Some(s) => s,
            None => return false,
        };
        match inner {
            // Session tickets are absorbed into the ticket queue; the
            // connection stays alive.
            RECORD_HANDSHAKE => self.handle_post_handshake(pt).is_ok(),
            // Any alert on an idle connection — even a clean
            // close_notify — means it is going away.
            _ => false,
        }
    }

    /// Encrypt and send application data. Returns the number of plaintext
    /// bytes consumed (always `buf.len()` on success).
    pub fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "tls: connection closed",
            ));
        }
        for chunk in buf.chunks(MAX_PLAINTEXT_LEN) {
            let rec = seal_record(&self.write_keys, self.write_seq, RECORD_APP_DATA, chunk)
                .map_err(tls_to_io)?;
            // RFC 8446 §5.3: the 64-bit sequence number must never
            // wrap — reusing a nonce would break the AEAD. On
            // exhaustion the connection fails closed instead.
            self.write_seq = self.write_seq.checked_add(1).ok_or_else(|| {
                self.closed = true;
                io::Error::new(io::ErrorKind::InvalidData, "tls: record sequence exhausted")
            })?;
            self.stream.write_all(&rec)?;
        }
        Ok(buf.len())
    }

    /// Send close_notify (best effort) and mark the connection closed.
    pub fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        if let Ok(rec) = seal_record(
            &self.write_keys,
            self.write_seq,
            RECORD_ALERT,
            &[1u8, ALERT_CLOSE_NOTIFY],
        ) {
            let _ = self.stream.write_all(&rec);
        }
    }

    /// Whether the connection is closed (peer sent close_notify or `close` was called).
    #[cfg(test)]
    fn is_closed(&self) -> bool {
        self.closed
    }

    /// The ALPN protocol the server selected, if any.
    #[must_use]
    pub fn alpn_protocol(&self) -> Option<&str> {
        self.alpn.as_deref()
    }

    /// The negotiated cipher suite id.
    #[must_use]
    pub fn cipher_suite(&self) -> u16 {
        self.cipher_suite
    }

    fn handle_post_handshake(&mut self, pt: &[u8]) -> Result<(), TlsError> {
        self.framer.push(pt)?;
        while let Some((t, m)) = self.framer.next_message()? {
            match t {
                HS_NEW_SESSION_TICKET => {
                    let nst = parse_new_session_ticket(&m)?;
                    let hash = self.resumption_hash;
                    // RFC 8446 §4.6.1: the PSK for this ticket.
                    let psk = hkdf_expand_label(
                        hash,
                        &self.resumption_master_secret,
                        b"resumption",
                        &nst.nonce,
                        hash_len(hash),
                    )?;
                    if self.received_tickets.len() >= MAX_CACHED_TICKETS {
                        self.received_tickets.remove(0);
                    }
                    self.received_tickets.push(CachedTicket {
                        ticket: nst.ticket,
                        psk,
                        cipher_suite: self.cipher_suite,
                        received_at_ms: now_ms(),
                        lifetime_secs: nst.lifetime,
                        age_add: nst.age_add,
                    });
                }
                _ => {
                    return Err(TlsError::UnexpectedMessage(
                        "unexpected post-handshake message",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Take the session tickets received since the last call. A
    /// connection pool drains these into its ticket cache before reusing
    /// the connection, so the *next* connection to the same host can
    /// resume instead of doing a full handshake.
    pub fn take_received_tickets(&mut self) -> Vec<CachedTicket> {
        std::mem::take(&mut self.received_tickets)
    }
}

impl Drop for TlsConnection {
    /// Best-effort close_notify on drop (`close` is idempotent), then
    /// scrub the resumption master secret: post-handshake ticket PSKs
    /// derive from it, so it must not linger. Traffic keys and cached
    /// tickets scrub themselves on drop.
    fn drop(&mut self) {
        self.close();
        zeroize(self.resumption_master_secret.as_mut_slice());
    }
}

// ---------------------------------------------------------------------------
// Tests: canned server flights drive the state machine; no network needed.
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn is_grease(v: u16) -> bool {
        v & 0x0F0F == 0x0A0A
    }

    fn test_config() -> TlsConfig {
        TlsConfig {
            server_name: "example.com".to_string(),
            alpn: vec!["h2".to_string(), "http/1.1".to_string()],
            fingerprint_profile: None,
            psk: None,
        }
    }

    fn test_handshake() -> Handshake {
        Handshake::new_with_secrets(&test_config(), [0x11; 32], [0x33; 32], vec![0x55; 32])
            .expect("test handshake")
    }

    /// Split a ClientHello handshake message into `(extension_id, body)` pairs.
    fn ch_extensions(msg: &[u8]) -> Vec<(u16, Vec<u8>)> {
        assert_eq!(msg[0], HS_CLIENT_HELLO);
        let body_len = ((msg[1] as usize) << 16) | ((msg[2] as usize) << 8) | msg[3] as usize;
        let body = &msg[4..4 + body_len];
        let mut r = Reader::new(body);
        let _ = r.u16("v").unwrap();
        let _ = r.bytes(32, "r").unwrap();
        let _ = r.vec_u8("sid").unwrap();
        let _ = r.vec_u16("suites").unwrap();
        let _ = r.vec_u8("comp").unwrap();
        let exts = r.vec_u16("exts").unwrap();
        assert!(r.is_empty(), "trailing ClientHello bytes");
        let mut er = Reader::new(exts);
        let mut out = Vec::new();
        while !er.is_empty() {
            let id = er.u16("id").unwrap();
            let data = er.vec_u16("data").unwrap().to_vec();
            out.push((id, data));
        }
        out
    }

    fn default_extension_ids(grease: u16) -> Vec<u16> {
        let profile = FingerprintProfile::default();
        profile
            .extensions
            .iter()
            .map(|spec| match spec {
                ExtensionSpec::ServerName => EXT_SERVER_NAME,
                ExtensionSpec::ExtendedMasterSecret => EXT_EXTENDED_MASTER_SECRET,
                ExtensionSpec::RenegotiationInfo => EXT_RENEGOTIATION_INFO,
                ExtensionSpec::SupportedGroups => EXT_SUPPORTED_GROUPS,
                ExtensionSpec::EcPointFormats => EXT_EC_POINT_FORMATS,
                ExtensionSpec::SessionTicket => EXT_SESSION_TICKET,
                ExtensionSpec::Alpn => EXT_ALPN,
                ExtensionSpec::SignatureAlgorithms => EXT_SIGNATURE_ALGORITHMS,
                ExtensionSpec::KeyShare => EXT_KEY_SHARE,
                ExtensionSpec::PskKeyExchangeModes => EXT_PSK_KEY_EXCHANGE_MODES,
                ExtensionSpec::SupportedVersions => EXT_SUPPORTED_VERSIONS,
                ExtensionSpec::Grease => grease,
                ExtensionSpec::Raw(id, _) => *id,
            })
            .collect()
    }

    // -- ClientHello -------------------------------------------------------

    #[test]
    fn client_hello_byte_layout() {
        let mut hs = test_handshake();
        let msg = hs.client_hello().unwrap();
        assert_eq!(msg[0], HS_CLIENT_HELLO);
        assert_eq!(&msg[4..6], &[0x03, 0x03], "legacy_version");
        assert_eq!(&msg[6..38], &[0x11; 32], "random");
        assert_eq!(msg[38], 32, "session id length");
        assert_eq!(&msg[39..71], &[0x55; 32], "session id");

        // Cipher suites: GREASE first, then the profile order.
        let suites_off = 71;
        let suites_len = u16::from_be_bytes([msg[suites_off], msg[suites_off + 1]]) as usize;
        assert_eq!(suites_len % 2, 0);
        let first = u16::from_be_bytes([msg[suites_off + 2], msg[suites_off + 3]]);
        assert!(is_grease(first), "expected GREASE first, got 0x{first:04x}");
        let second = u16::from_be_bytes([msg[suites_off + 4], msg[suites_off + 5]]);
        assert_eq!(second, 0x1301);

        // Extension order is exactly the profile order.
        let exts = ch_extensions(&msg);
        let ids: Vec<u16> = exts.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, default_extension_ids(grease_value(0x11)));

        // GREASE first in supported_groups too.
        let groups = exts
            .iter()
            .find(|(id, _)| *id == EXT_SUPPORTED_GROUPS)
            .unwrap();
        let g0 = u16::from_be_bytes([groups.1[2], groups.1[3]]);
        assert!(is_grease(g0));

        // SNI carries the hostname.
        let sni = exts.iter().find(|(id, _)| *id == EXT_SERVER_NAME).unwrap();
        assert!(
            sni.1
                .windows(b"example.com".len())
                .any(|w| w == b"example.com")
        );

        // The transcript captured the message (pre-cipher: digest hashes
        // the buffered bytes).
        assert_eq!(hs.transcript.digest(HashAlg::Sha256), sha256(&msg).to_vec());
    }

    #[test]
    fn client_hello_no_grease() {
        let profile = FingerprintProfile {
            grease: false,
            ..FingerprintProfile::default()
        };
        let config = TlsConfig {
            server_name: "example.com".to_string(),
            alpn: vec![],
            fingerprint_profile: Some(profile),
            psk: None,
        };
        let mut hs =
            Handshake::new_with_secrets(&config, [0x11; 32], [0x33; 32], vec![0x55; 32]).unwrap();
        let msg = hs.client_hello().unwrap();

        let suites_off = 71;
        let first = u16::from_be_bytes([msg[suites_off + 2], msg[suites_off + 3]]);
        assert_eq!(first, 0x1301, "no GREASE cipher when disabled");

        let exts = ch_extensions(&msg);
        assert!(
            !exts.iter().any(|(id, _)| is_grease(*id)),
            "no GREASE extension"
        );
        let groups = exts
            .iter()
            .find(|(id, _)| *id == EXT_SUPPORTED_GROUPS)
            .unwrap();
        let g0 = u16::from_be_bytes([groups.1[2], groups.1[3]]);
        assert_eq!(g0, 0x001d, "no GREASE group");
    }

    #[test]
    fn client_hello_ip_literal_omits_sni() {
        let config = TlsConfig {
            server_name: "93.184.216.34".to_string(),
            alpn: vec![],
            fingerprint_profile: None,
            psk: None,
        };
        let mut hs = Handshake::new_with_secrets(&config, [0x11; 32], [0x33; 32], vec![]).unwrap();
        let msg = hs.client_hello().unwrap();
        let exts = ch_extensions(&msg);
        assert!(!exts.iter().any(|(id, _)| *id == EXT_SERVER_NAME));
    }

    #[test]
    fn client_hello_raw_extension_order() {
        let profile = FingerprintProfile {
            extensions: vec![
                ExtensionSpec::Raw(0x1234, vec![0xAA]),
                ExtensionSpec::ServerName,
                ExtensionSpec::Raw(0x5678, vec![]),
            ],
            ..FingerprintProfile::default()
        };
        let config = TlsConfig {
            server_name: "example.com".to_string(),
            alpn: vec![],
            fingerprint_profile: Some(profile),
            psk: None,
        };
        let mut hs = Handshake::new_with_secrets(&config, [0x11; 32], [0x33; 32], vec![]).unwrap();
        let msg = hs.client_hello().unwrap();
        let ids: Vec<u16> = ch_extensions(&msg).iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![0x1234, EXT_SERVER_NAME, 0x5678]);
        let raw = ch_extensions(&msg);
        assert_eq!(raw[0].1, vec![0xAA]);
    }

    #[test]
    fn empty_server_name_rejected() {
        let config = TlsConfig {
            server_name: String::new(),
            alpn: vec![],
            fingerprint_profile: None,
            psk: None,
        };
        let r = Handshake::new_with_secrets(&config, [0x11; 32], [0x33; 32], vec![]);
        assert!(matches!(r, Err(TlsError::InvalidServerName)));
    }

    #[test]
    fn trailing_dot_server_name_sni_is_canonical() {
        // A trailing dot is the FQDN root marker, not part of the name:
        // SNI must carry the canonical form so it matches the
        // certificate hostname verification checks.
        let config = TlsConfig {
            server_name: "example.com.".to_string(),
            alpn: vec![],
            fingerprint_profile: None,
            psk: None,
        };
        let mut hs = Handshake::new_with_secrets(&config, [0x11; 32], [0x33; 32], vec![]).unwrap();
        assert_eq!(hs.server_name, "example.com");
        let msg = hs.client_hello().unwrap();
        let exts = ch_extensions(&msg);
        let sni = exts
            .iter()
            .find(|(id, _)| *id == EXT_SERVER_NAME)
            .expect("SNI present");
        let body = &sni.1;
        let name_len = u16::from_be_bytes([body[3], body[4]]) as usize;
        assert_eq!(&body[5..5 + name_len], b"example.com");
    }

    #[test]
    fn x25519mlkem768_hybrid_handshake_loopback() {
        // Positive hybrid test: the client offers an X25519MLKEM768
        // (0x11EC) key share (`x25519_pub || mlkem_ek`), the loopback
        // server encapsulates under the client's ek, and the client
        // must derive exactly the handshake secret the server computes
        // from the concatenated 64-byte shared secret
        // (draft-ietf-tls-hybrid-design).
        let profile = FingerprintProfile {
            groups: vec![GROUP_X25519MLKEM768],
            key_share_groups: vec![GROUP_X25519MLKEM768],
            ..FingerprintProfile::default()
        };
        let config = TlsConfig {
            server_name: "example.com".to_string(),
            alpn: vec![],
            fingerprint_profile: Some(profile),
            psk: None,
        };
        let (client_stream, mut server_stream) = socketpair();
        let handle = std::thread::spawn(move || {
            let mut c = client_stream;
            let mut hs = Handshake::new(&config).unwrap();
            let ch = hs.client_hello().unwrap();
            write_record(&mut c, RECORD_HANDSHAKE, LEGACY_RECORD_VERSION, &ch).unwrap();
            let mut framer = HandshakeFramer::new();
            let action = TlsConnection::read_server_hello(&mut c, &mut hs, &mut framer).unwrap();
            assert!(matches!(action, HelloAction::Proceed));
            hs
        });

        // Server side: read the ClientHello and pull the hybrid key
        // share out of its key_share extension.
        let ch_rec = read_one_record(&mut server_stream).unwrap();
        let sid = parse_client_session_id(&ch_rec);
        let ch = &ch_rec[5..];
        let kse = ch_extensions(ch)
            .into_iter()
            .find(|(id, _)| *id == EXT_KEY_SHARE)
            .map(|(_, data)| data)
            .expect("ClientHello without key_share");
        let mut kr = Reader::new(kse.as_slice());
        let entries = kr.vec_u16("kse").unwrap();
        let mut er = Reader::new(entries);
        let (group, ke) = loop {
            if er.is_empty() {
                panic!("no hybrid key share in ClientHello");
            }
            let g = er.u16("g").unwrap();
            let ke = er.vec_u16("ke").unwrap().to_vec();
            if g == GROUP_X25519MLKEM768 {
                break (g, ke);
            }
        };
        assert_eq!(group, GROUP_X25519MLKEM768);
        assert_eq!(ke.len(), 32 + ml_kem_768::ENCAPSULATION_KEY_LEN);

        // Encapsulate under the client's ek; the concatenated
        // x25519 || mlkem shared secret is what both sides feed the
        // key schedule.
        let (s_secret, s_pub) = x25519::keypair().unwrap();
        let mut ek = [0u8; ml_kem_768::ENCAPSULATION_KEY_LEN];
        ek.copy_from_slice(&ke[32..]);
        let (ct, m_shared) = ml_kem_768::encaps(&ek).unwrap();
        let mut client_x = [0u8; 32];
        client_x.copy_from_slice(&ke[..32]);
        let x_shared = x25519::shared_secret(&s_secret, &client_x);
        let mut shared = Vec::with_capacity(64);
        shared.extend_from_slice(&x_shared);
        shared.extend_from_slice(&m_shared);

        let mut server_ke = Vec::with_capacity(32 + ml_kem_768::CIPHERTEXT_LEN);
        server_ke.extend_from_slice(&s_pub);
        server_ke.extend_from_slice(&ct);
        let sh = server_hello_msg(
            &sid,
            TLS_AES_128_GCM_SHA256,
            GROUP_X25519MLKEM768,
            &server_ke,
            None,
        );
        write_record(
            &mut server_stream,
            RECORD_HANDSHAKE,
            LEGACY_RECORD_VERSION,
            &sh,
        )
        .unwrap();

        // What the server derives from the same transcript: CH || SH
        // with their 4-byte headers, exactly as fed to the transcript
        // hasher on both sides.
        let mut transcript = Vec::new();
        transcript.extend_from_slice(ch);
        transcript.extend_from_slice(&sh);
        let thash = sha256(&transcript);
        let (hs_secret, s_hs, c_hs) = server_schedule(HashAlg::Sha256, &shared, &thash);

        let hs = handle.join().unwrap();
        assert_eq!(hs.handshake_secret, hs_secret);
        assert_eq!(hs.server_hs_traffic, s_hs);
        assert_eq!(hs.client_hs_traffic, c_hs);
    }

    /// Seal one handshake message into an encrypted (ApplicationData-outer)
    /// record and send it, appending the message to the server's
    /// transcript. Test helper for the trailing-data loopback test.
    fn send_encrypted(
        server_stream: &mut TcpStream,
        transcript: &mut Vec<u8>,
        s_keys: &TrafficKeys,
        seq: &mut u64,
        msg: &[u8],
    ) {
        transcript.extend_from_slice(msg);
        let rec = seal_record(s_keys, *seq, RECORD_HANDSHAKE, msg).unwrap();
        *seq += 1;
        server_stream.write_all(&rec).unwrap();
    }

    #[test]
    fn trailing_handshake_data_after_finished_fails_closed() {
        // End-to-end through run_handshake_inner: the fake server sends a
        // fully valid encrypted flight (EE, Certificate,
        // CertificateVerify, Finished) with a second handshake message
        // coalesced into the Finished's record. The client must fail
        // closed instead of silently dropping the trailing bytes.
        let profile = FingerprintProfile {
            groups: vec![GROUP_X25519],
            key_share_groups: vec![GROUP_X25519],
            ..FingerprintProfile::default()
        };
        let config = TlsConfig {
            server_name: "bowser.test".to_string(),
            alpn: vec![],
            fingerprint_profile: Some(profile),
            psk: None,
        };
        let (client_stream, mut server_stream) = socketpair();
        let handle = std::thread::spawn(move || {
            let mut c = client_stream;
            let mut hs =
                Handshake::new_with_secrets(&config, [0x11; 32], [0x33; 32], vec![]).unwrap();
            hs.trust_roots = Some(test_roots());
            let r = TlsConnection::run_handshake_inner(&mut c, &mut hs);
            match r {
                Err(TlsError::UnexpectedMessage(_)) => {}
                Err(e) => {
                    panic!("trailing handshake data after Finished must fail closed, got {e:?}")
                }
                Ok(_) => panic!(
                    "trailing handshake data after Finished must fail closed, handshake succeeded"
                ),
            }
        });

        // Fake server: read the ClientHello and pull out the X25519
        // share, exactly like the hybrid loopback test above.
        let ch_rec = read_one_record(&mut server_stream).unwrap();
        let sid = parse_client_session_id(&ch_rec);
        let ch = &ch_rec[5..];
        let kse = ch_extensions(ch)
            .into_iter()
            .find(|(id, _)| *id == EXT_KEY_SHARE)
            .map(|(_, data)| data)
            .expect("ClientHello without key_share");
        let mut kr = Reader::new(kse.as_slice());
        let entries = kr.vec_u16("kse").unwrap();
        let mut er = Reader::new(entries);
        let client_x = loop {
            if er.is_empty() {
                panic!("ClientHello without X25519 share");
            }
            let g = er.u16("g").unwrap();
            let ke = er.vec_u16("ke").unwrap().to_vec();
            if g == GROUP_X25519 {
                break ke;
            }
        };
        assert_eq!(client_x.len(), 32);
        let mut client_x_arr = [0u8; 32];
        client_x_arr.copy_from_slice(&client_x);

        let (s_secret, s_pub) = x25519::keypair().unwrap();
        let shared = x25519::shared_secret(&s_secret, &client_x_arr);
        let sh = server_hello_msg(&sid, TLS_AES_128_GCM_SHA256, GROUP_X25519, &s_pub, None);
        write_record(
            &mut server_stream,
            RECORD_HANDSHAKE,
            LEGACY_RECORD_VERSION,
            &sh,
        )
        .unwrap();

        let spec = cipher_spec(TLS_AES_128_GCM_SHA256).expect("known suite");
        let mut transcript = Vec::new();
        transcript.extend_from_slice(ch);
        transcript.extend_from_slice(&sh);
        let thash = sha256(&transcript);
        let (_, s_hs, _) = server_schedule(spec.hash, &shared, &thash);
        let s_keys = traffic_keys(spec.hash, &s_hs, &spec).unwrap();

        let mut seq = 0u64;
        send_encrypted(
            &mut server_stream,
            &mut transcript,
            &s_keys,
            &mut seq,
            &encrypted_extensions_msg(None),
        );

        let leaf = tls_fixture("leaf-rsa.der");
        let inter = tls_fixture("test-inter.der");
        send_encrypted(
            &mut server_stream,
            &mut transcript,
            &s_keys,
            &mut seq,
            &certificate_msg(&[&leaf, &inter]),
        );

        // CertificateVerify over the transcript so far, signed by the
        // fixture leaf key (RFC 8446 §4.4.3).
        let mut content = vec![b' '; 64];
        content.extend_from_slice(b"TLS 1.3, server CertificateVerify");
        content.push(0x00);
        content.extend_from_slice(&sha256(&transcript));
        let (n, d) = leaf_rsa_key();
        let sig = rsa_test_sign(&n, &d, &content);
        send_encrypted(
            &mut server_stream,
            &mut transcript,
            &s_keys,
            &mut seq,
            &certificate_verify_msg(0x0401, &sig),
        );

        // Finished, with a second handshake message coalesced into the
        // same encrypted record: the trailing bytes are already in the
        // framer when Finished completes, so the client must reject
        // them instead of dropping them.
        let fin = server_finished_msg(spec.hash, &s_hs, &sha256(&transcript));
        let mut junk = Vec::new();
        put_handshake_header(&mut junk, HS_ENCRYPTED_EXTENSIONS, b"trailing");
        let mut both = fin;
        both.extend_from_slice(&junk);
        let rec = seal_record(&s_keys, seq, RECORD_HANDSHAKE, &both).unwrap();
        server_stream.write_all(&rec).unwrap();

        handle.join().unwrap();
    }

    #[test]
    fn new_generates_random_material() {
        let h1 = Handshake::new(&test_config()).unwrap();
        let h2 = Handshake::new(&test_config()).unwrap();
        assert_ne!(h1.client_random, h2.client_random);
        assert_ne!(h1.x25519_secret, h2.x25519_secret);
    }

    // -- Canned server flights ---------------------------------------------

    fn server_hello_msg(
        session_echo: &[u8],
        cipher: u16,
        group: u16,
        ke: &[u8],
        selected_psk: Option<u16>,
    ) -> Vec<u8> {
        let mut body = Vec::new();
        put_u16(&mut body, LEGACY_HANDSHAKE_VERSION);
        body.extend_from_slice(&[0x22; 32]);
        body.push(session_echo.len() as u8);
        body.extend_from_slice(session_echo);
        put_u16(&mut body, cipher);
        body.push(0u8);
        let mut exts = Vec::new();
        put_u16(&mut exts, EXT_SUPPORTED_VERSIONS);
        put_u16(&mut exts, 2);
        exts.extend_from_slice(&[0x03, 0x04]);
        put_u16(&mut exts, EXT_KEY_SHARE);
        let mut kse = Vec::new();
        put_u16(&mut kse, group);
        put_u16(&mut kse, ke.len() as u16);
        kse.extend_from_slice(ke);
        put_u16(&mut exts, kse.len() as u16);
        exts.extend_from_slice(&kse);
        if let Some(sel) = selected_psk {
            put_u16(&mut exts, EXT_PRE_SHARED_KEY);
            put_u16(&mut exts, 2);
            put_u16(&mut exts, sel);
        }
        put_u16(&mut body, exts.len() as u16);
        body.extend_from_slice(&exts);
        let mut msg = Vec::new();
        put_handshake_header(&mut msg, HS_SERVER_HELLO, &body);
        msg
    }

    fn encrypted_extensions_msg(alpn: Option<&str>) -> Vec<u8> {
        let mut exts = Vec::new();
        if let Some(p) = alpn {
            put_u16(&mut exts, EXT_ALPN);
            let mut list = Vec::new();
            list.push(p.len() as u8);
            list.extend_from_slice(p.as_bytes());
            let mut ab = Vec::new();
            put_u16(&mut ab, list.len() as u16);
            ab.extend_from_slice(&list);
            put_u16(&mut exts, ab.len() as u16);
            exts.extend_from_slice(&ab);
        }
        let mut body = Vec::new();
        put_u16(&mut body, exts.len() as u16);
        body.extend_from_slice(&exts);
        let mut msg = Vec::new();
        put_handshake_header(&mut msg, HS_ENCRYPTED_EXTENSIONS, &body);
        msg
    }

    /// What the fake server derives: (handshake_secret, server_hs_traffic, client_hs_traffic).
    /// Takes the transcript hash through ClientHello..ServerHello directly so
    /// callers can feed a streaming or a one-shot transcript hash.
    fn server_schedule(
        hash: HashAlg,
        shared: &[u8],
        thash_ch_sh: &[u8],
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let zeros = vec![0u8; hash_len(hash)];
        let early = hkdf_extract(hash, &zeros, &zeros);
        let derived = derive_secret(hash, &early, b"derived", &transcript_hash(hash, b"")).unwrap();
        let hs_secret = hkdf_extract(hash, &derived, shared);
        let s_hs = derive_secret(hash, &hs_secret, b"s hs traffic", thash_ch_sh).unwrap();
        let c_hs = derive_secret(hash, &hs_secret, b"c hs traffic", thash_ch_sh).unwrap();
        (hs_secret, s_hs, c_hs)
    }

    fn server_finished_msg(hash: HashAlg, s_hs_traffic: &[u8], thash: &[u8]) -> Vec<u8> {
        let fk = finished_key(hash, s_hs_traffic).unwrap();
        let vd = hmac(hash, &fk, thash);
        let mut msg = Vec::new();
        put_handshake_header(&mut msg, HS_FINISHED, &vd);
        msg
    }

    /// Fixed-secrets fake server: drives the client through ClientHello,
    /// ServerHello and EncryptedExtensions, and returns everything a test
    /// needs to finish (or break) the encrypted flight.
    struct CannedFlight {
        hs: Handshake,
        spec: CipherSpec,
        s_hs_traffic: Vec<u8>,
        /// Transcript hash through ServerHello, for divergent-transcript tests.
        thash_ch_sh: Vec<u8>,
    }

    fn canned_flight() -> CannedFlight {
        let mut hs = test_handshake();
        let ch = hs.client_hello().unwrap();
        assert_eq!(ch[0], HS_CLIENT_HELLO);

        // Fake server with a fixed X25519 keypair.
        let server_secret = [0x44; 32];
        let server_pub = x25519_public(&server_secret);
        let shared = x25519::shared_secret(&server_secret, &x25519_public(&[0x33; 32]));

        let sh = server_hello_msg(
            &[0x55; 32],
            TLS_AES_128_GCM_SHA256,
            GROUP_X25519,
            &server_pub,
            None,
        );
        assert!(matches!(
            hs.handle_server_hello(&sh).unwrap(),
            HelloAction::Proceed
        ));
        assert_eq!(hs.cipher.unwrap().id, TLS_AES_128_GCM_SHA256);

        let spec = hs.cipher.unwrap();
        let thash_ch_sh = hs.transcript.digest(spec.hash);
        let (_, s_hs_traffic, _) = server_schedule(spec.hash, &shared, &thash_ch_sh);

        let ee = encrypted_extensions_msg(Some("h2"));
        hs.handle_encrypted_extensions(&ee).unwrap();
        assert_eq!(hs.alpn_selected.as_deref(), Some("h2"));

        CannedFlight {
            hs,
            spec,
            s_hs_traffic,
            thash_ch_sh,
        }
    }

    /// The fake server's Finished, computed over the client's real
    /// transcript (CH..SH..EE) — i.e. a well-formed Finished.
    fn canned_server_finished(f: &CannedFlight) -> Vec<u8> {
        let thash = f.hs.transcript.digest(f.spec.hash);
        server_finished_msg(f.spec.hash, &f.s_hs_traffic, &thash)
    }

    // ---------- TLS 1.3 session resumption (PSK) ----------

    const TEST_PSK: [u8; 32] = [0xab; 32];
    const TEST_TICKET: &[u8] = b"test-ticket";
    const TEST_TICKET_AGE: u32 = 1234;

    fn psk_config() -> TlsConfig {
        TlsConfig {
            server_name: "example.com".to_string(),
            alpn: vec![],
            fingerprint_profile: None,
            psk: Some(PskOffer {
                identity: TEST_TICKET.to_vec(),
                obfuscated_ticket_age: TEST_TICKET_AGE,
                psk: TEST_PSK.to_vec(),
                cipher_suite: TLS_AES_128_GCM_SHA256,
            }),
        }
    }

    fn psk_handshake() -> Handshake {
        Handshake::new_with_secrets(&psk_config(), [0x11; 32], [0x33; 32], vec![0x55; 32])
            .expect("psk test handshake")
    }

    /// What the fake server derives for an abbreviated handshake: like
    /// `server_schedule`, but the early secret mixes in the PSK
    /// (RFC 8446 §7.1).
    fn server_schedule_psk(
        hash: HashAlg,
        psk: &[u8],
        shared: &[u8; 32],
        thash_ch_sh: &[u8],
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let zeros = vec![0u8; hash_len(hash)];
        let early = hkdf_extract(hash, &zeros, psk);
        let derived = derive_secret(hash, &early, b"derived", &transcript_hash(hash, b"")).unwrap();
        let hs_secret = hkdf_extract(hash, &derived, shared);
        let s_hs = derive_secret(hash, &hs_secret, b"s hs traffic", thash_ch_sh).unwrap();
        let c_hs = derive_secret(hash, &hs_secret, b"c hs traffic", thash_ch_sh).unwrap();
        (hs_secret, s_hs, c_hs)
    }

    /// Server-side binder verification for the *first* ClientHello: parse
    /// the offered identity and binder, then recompute the expected
    /// binder (RFC 8446 §4.2.11.2) and require an exact match.
    fn check_client_binder(ch: &[u8], psk: &[u8], hash: HashAlg) {
        let exts = ch_extensions(ch);
        let (id, data) = exts
            .iter()
            .find(|(id, _)| *id == EXT_PRE_SHARED_KEY)
            .expect("ClientHello carries pre_shared_key");
        assert_eq!(*id, EXT_PRE_SHARED_KEY);
        // pre_shared_key must be the last extension (RFC 8446 §4.2.11).
        assert_eq!(exts.last().map(|(id, _)| *id), Some(EXT_PRE_SHARED_KEY));
        let mut r = Reader::new(data);
        let ids = r.vec_u16("identities").unwrap();
        let mut ir = Reader::new(ids);
        let identity = ir.vec_u16("identity").unwrap();
        assert_eq!(identity, TEST_TICKET);
        let age = u32::from_be_bytes(ir.bytes(4, "age").unwrap().try_into().unwrap());
        assert_eq!(age, TEST_TICKET_AGE);
        assert!(ir.is_empty(), "exactly one identity is offered");
        let binders = r.vec_u16("binders").unwrap();
        assert!(r.is_empty(), "no trailing bytes in pre_shared_key");
        let mut br = Reader::new(binders);
        let binder = br.vec_u8("binder").unwrap();
        assert!(br.is_empty(), "exactly one binder");
        assert_eq!(binder.len(), hash_len(hash));
        // Truncate() drops the binders list, as in build_client_hello_msg.
        let trunc = &ch[..ch.len() - 3 - binder.len()];
        let zeros = vec![0u8; hash_len(hash)];
        let early = hkdf_extract(hash, &zeros, psk);
        let binder_key =
            derive_secret(hash, &early, b"res binder", &transcript_hash(hash, b"")).unwrap();
        let fk = finished_key(hash, &binder_key).unwrap();
        let mut th = TranscriptHasher::new();
        th.seal(hash);
        th.push(trunc);
        let expect = hmac(hash, &fk, &th.digest(hash));
        assert_eq!(binder, expect.as_slice(), "binder must verify");
    }

    /// Fixed-secrets fake server that *accepts* the client's PSK offer:
    /// drives the client through ClientHello (binder verified
    /// server-side), ServerHello (with the selected identity) and
    /// EncryptedExtensions; the test then supplies Finished directly —
    /// no Certificate/CertificateVerify.
    struct CannedPskFlight {
        hs: Handshake,
        spec: CipherSpec,
        s_hs_traffic: Vec<u8>,
        hs_secret: Vec<u8>,
        /// Transcript hash through ServerHello, for key derivation.
        thash_ch_sh: Vec<u8>,
    }

    fn canned_psk_flight() -> CannedPskFlight {
        let mut hs = psk_handshake();
        let ch = hs.client_hello().unwrap();
        assert_eq!(ch[0], HS_CLIENT_HELLO);
        check_client_binder(&ch, &TEST_PSK, HashAlg::Sha256);

        let server_secret = [0x44; 32];
        let server_pub = x25519_public(&server_secret);
        let shared = x25519::shared_secret(&server_secret, &x25519_public(&[0x33; 32]));

        let sh = server_hello_msg(
            &[0x55; 32],
            TLS_AES_128_GCM_SHA256,
            GROUP_X25519,
            &server_pub,
            Some(0),
        );
        assert!(matches!(
            hs.handle_server_hello(&sh).unwrap(),
            HelloAction::Proceed
        ));
        assert!(hs.psk_accepted, "server selected our PSK identity");

        let spec = hs.cipher.unwrap();
        let thash_ch_sh = hs.transcript.digest(spec.hash);
        let (hs_secret, s_hs_traffic, _) =
            server_schedule_psk(spec.hash, &TEST_PSK, &shared, &thash_ch_sh);

        let ee = encrypted_extensions_msg(None);
        hs.handle_encrypted_extensions(&ee).unwrap();

        CannedPskFlight {
            hs,
            spec,
            s_hs_traffic,
            hs_secret,
            thash_ch_sh,
        }
    }

    #[test]
    fn psk_binder_offered_last_and_verifies() {
        let mut hs = psk_handshake();
        let ch = hs.client_hello().unwrap();
        check_client_binder(&ch, &TEST_PSK, HashAlg::Sha256);
    }

    #[test]
    fn psk_handshake_completes_without_certificate() {
        let mut f = canned_psk_flight();
        // Abbreviated flight: EncryptedExtensions -> Finished directly.
        // Authentication comes from the Finished MAC over PSK-derived
        // keys; no Certificate/CertificateVerify is sent or required.
        let thash = f.hs.transcript.digest(f.spec.hash);
        let fin = server_finished_msg(f.spec.hash, &f.s_hs_traffic, &thash);
        // Transcript through the server Finished (handle_finished also
        // appends the client Finished itself, so hash a clone with the
        // server Finished pushed but before the call).
        let thash2 = {
            let mut th = f.hs.transcript.clone();
            th.push(&fin);
            th.digest(f.spec.hash)
        };
        let client_finished = f.hs.handle_finished(&fin).unwrap();
        assert!(f.hs.client_app.is_some());
        assert!(f.hs.server_app.is_some());
        // The client's Finished must verify against the fake server's
        // view: c_hs_traffic derives from the transcript through
        // ServerHello; the Finished MAC covers the transcript through
        // the server Finished.
        let c_hs =
            derive_secret(f.spec.hash, &f.hs_secret, b"c hs traffic", &f.thash_ch_sh).unwrap();
        let fk = finished_key(f.spec.hash, &c_hs).unwrap();
        let expect_vd = hmac(f.spec.hash, &fk, &thash2);
        let mut expect = Vec::new();
        put_handshake_header(&mut expect, HS_FINISHED, &expect_vd);
        assert_eq!(client_finished, expect);
        // Both sides derive the same resumption master secret; RFC 8446
        // §7.1 runs the transcript through client Finished, and
        // `handle_finished` already pushed it, so the live transcript is
        // the expected context.
        let hash = f.spec.hash;
        let zeros = vec![0u8; hash_len(hash)];
        let derived =
            derive_secret(hash, &f.hs_secret, b"derived", &transcript_hash(hash, b"")).unwrap();
        let master = hkdf_extract(hash, &derived, &zeros);
        let thash3 = f.hs.transcript.digest(hash);
        let expect_rms = derive_secret(hash, &master, b"res master", &thash3).unwrap();
        assert_eq!(f.hs.resumption_master_secret, expect_rms);
        assert!(!f.hs.resumption_master_secret.is_empty());
    }

    #[test]
    fn server_selecting_unknown_psk_identity_rejected() {
        let mut hs = psk_handshake();
        let _ = hs.client_hello().unwrap();
        let server_pub = x25519_public(&[0x44; 32]);
        let sh = server_hello_msg(
            &[0x55; 32],
            TLS_AES_128_GCM_SHA256,
            GROUP_X25519,
            &server_pub,
            Some(1),
        );
        let r = hs.handle_server_hello(&sh);
        assert!(
            matches!(r, Err(TlsError::BadHandshake(_))),
            "unknown PSK identity must be rejected, got {r:?}"
        );
    }

    #[test]
    fn server_selecting_psk_without_offer_rejected() {
        let mut hs = test_handshake();
        let _ = hs.client_hello().unwrap();
        let server_pub = x25519_public(&[0x44; 32]);
        let sh = server_hello_msg(
            &[0x55; 32],
            TLS_AES_128_GCM_SHA256,
            GROUP_X25519,
            &server_pub,
            Some(0),
        );
        let r = hs.handle_server_hello(&sh);
        assert!(
            matches!(r, Err(TlsError::BadHandshake(_))),
            "unsolicited PSK selection must be rejected, got {r:?}"
        );
    }

    #[test]
    fn server_selecting_psk_with_different_cipher_rejected() {
        // The ticket was issued for 0x1301 (SHA-256); the server must not
        // accept it under 0x1303 (SHA-384) — the hash would not match.
        let mut hs = psk_handshake();
        let _ = hs.client_hello().unwrap();
        let server_pub = x25519_public(&[0x44; 32]);
        let sh = server_hello_msg(
            &[0x55; 32],
            TLS_CHACHA20_POLY1305_SHA256,
            GROUP_X25519,
            &server_pub,
            Some(0),
        );
        let r = hs.handle_server_hello(&sh);
        assert!(
            matches!(r, Err(TlsError::BadHandshake(_))),
            "PSK accepted under a different cipher must be rejected, got {r:?}"
        );
    }

    #[test]
    fn cached_ticket_offer_expiry() {
        let fresh = CachedTicket {
            ticket: b"tick".to_vec(),
            psk: vec![0xab; 32],
            cipher_suite: TLS_AES_128_GCM_SHA256,
            received_at_ms: now_ms(),
            lifetime_secs: 3600,
            age_add: 42,
        };
        let offer = fresh.offer(now_ms()).expect("fresh ticket is offerable");
        assert_eq!(offer.identity, b"tick");
        assert_eq!(offer.psk, vec![0xab; 32]);
        assert_eq!(offer.cipher_suite, TLS_AES_128_GCM_SHA256);
        // Obfuscated age = real age + age_add; the real age here is only
        // the milliseconds between constructing the ticket and offering it.
        assert!(
            offer.obfuscated_ticket_age.wrapping_sub(42) < 10_000,
            "obfuscated age skews too far: {}",
            offer.obfuscated_ticket_age
        );
        // A ticket past its lifetime is never offered.
        let mut stale = fresh.clone();
        stale.received_at_ms = now_ms().saturating_sub(7_200_000);
        assert!(stale.offer(now_ms()).is_none());
        // Zero-length PSKs are never offered either.
        let mut bad = fresh.clone();
        bad.psk = Vec::new();
        assert!(bad.offer(now_ms()).is_none());
    }

    #[test]
    fn post_handshake_tickets_queued_capped_and_drainable() {
        let (client, _server) = socketpair();
        let mut conn = test_connection(client, test_keys(), 0);
        // The ticket PSK derives from the resumption master secret.
        conn.resumption_master_secret = vec![0xcc; 32];
        let mut flight = Vec::new();
        for i in 0..(MAX_CACHED_TICKETS + 3) {
            let mut body = Vec::new();
            body.extend_from_slice(&3600u32.to_be_bytes());
            body.extend_from_slice(&1u32.to_be_bytes());
            body.push(1u8);
            body.push(i as u8); // nonce
            let ticket = format!("t{i}");
            put_u16(&mut body, ticket.len() as u16);
            body.extend_from_slice(ticket.as_bytes());
            put_u16(&mut body, 0u16); // empty extensions
            put_handshake_header(&mut flight, HS_NEW_SESSION_TICKET, &body);
        }
        conn.handle_post_handshake(&flight).unwrap();
        let tickets = conn.take_received_tickets();
        assert_eq!(tickets.len(), MAX_CACHED_TICKETS, "ticket queue is capped");
        // Oldest evicted: the first retained ticket is t3.
        assert_eq!(tickets[0].ticket, b"t3");
        let offer = tickets[0].offer(now_ms()).expect("ticket is offerable");
        assert_eq!(offer.identity, b"t3");
        assert_eq!(offer.cipher_suite, TLS_AES_128_GCM_SHA256);
        // RFC 8446 §4.6.1: PSK = HKDF-Expand-Label(resumption_master_secret,
        // "resumption", nonce, Hash.length).
        let expect_psk = hkdf_expand_label(
            HashAlg::Sha256,
            &[0xcc; 32],
            b"resumption",
            &[3u8],
            hash_len(HashAlg::Sha256),
        )
        .unwrap();
        assert_eq!(offer.psk, expect_psk);
        // Draining is one-shot.
        assert!(conn.take_received_tickets().is_empty());
    }

    #[test]
    fn flight_without_certificates_fails_closed() {
        // C1: a flight of EncryptedExtensions -> Finished with no
        // Certificate message must not complete unauthenticated.
        let mut f = canned_flight();
        let fin = canned_server_finished(&f);
        let r = f.hs.handle_finished(&fin);
        assert!(matches!(r, Err(TlsError::CertificateVerificationFailed(_))));
    }

    #[test]
    fn finished_over_divergent_transcript_rejected() {
        // M1b: the server's Finished is computed over a transcript that is
        // missing EncryptedExtensions; the client must reject it even
        // though the flight itself delivered EE first.
        let mut f = canned_flight();
        let fin = server_finished_msg(f.spec.hash, &f.s_hs_traffic, &f.thash_ch_sh);
        let r = f.hs.handle_finished(&fin);
        assert!(
            matches!(r, Err(TlsError::BadHandshake(_))),
            "divergent-transcript Finished must be rejected, got {r:?}"
        );
    }

    #[test]
    fn finished_tamper_rejected() {
        let mut f = canned_flight();
        let mut fin = canned_server_finished(&f);
        let last = fin.len() - 1;
        fin[last] ^= 0x01;
        let r = f.hs.handle_finished(&fin);
        assert!(matches!(r, Err(TlsError::BadHandshake(_))));
    }

    #[test]
    fn tls12_server_hello_fails_closed() {
        let mut hs = test_handshake();
        let _ = hs.client_hello().unwrap();
        // A TLS 1.2 ServerHello: no supported_versions extension.
        let mut body = Vec::new();
        put_u16(&mut body, LEGACY_HANDSHAKE_VERSION);
        body.extend_from_slice(&[0x22; 32]);
        body.push(32u8);
        body.extend_from_slice(&[0x55; 32]);
        put_u16(&mut body, 0xc02f); // offered 1.2 suite
        body.push(0u8);
        put_u16(&mut body, 0u16); // empty extensions
        let mut sh = Vec::new();
        put_handshake_header(&mut sh, HS_SERVER_HELLO, &body);
        let r = hs.handle_server_hello(&sh);
        // TLS 1.2 has no state machine here; the version is unsupported.
        assert!(matches!(r, Err(TlsError::UnsupportedVersion)));
    }

    #[test]
    fn all_zero_key_share_rejected() {
        // RFC 8446 §7.4.1: an all-zero (EC)DHE shared secret must abort
        // the handshake, never feed the key schedule.
        let mut hs = test_handshake();
        let _ = hs.client_hello().unwrap();
        let sh = server_hello_msg(
            &[0x55; 32],
            TLS_AES_128_GCM_SHA256,
            GROUP_X25519,
            &[0u8; 32],
            None,
        );
        let r = hs.handle_server_hello(&sh);
        assert!(
            matches!(r, Err(TlsError::BadHandshake(_))),
            "all-zero key_share must be rejected, got {r:?}"
        );
    }

    /// Build a Certificate handshake message from DER certificates.
    fn certificate_msg(certs: &[&[u8]]) -> Vec<u8> {
        let mut body = Vec::new();
        body.push(0u8); // request_context
        let mut list = Vec::new();
        for cert in certs {
            let len = cert.len();
            list.push((len >> 16) as u8);
            list.push((len >> 8) as u8);
            list.push(len as u8);
            list.extend_from_slice(cert);
            list.extend_from_slice(&[0u8, 0u8]); // empty entry extensions
        }
        let list_len = list.len();
        body.push((list_len >> 16) as u8);
        body.push((list_len >> 8) as u8);
        body.push(list_len as u8);
        body.extend_from_slice(&list);
        let mut msg = Vec::new();
        put_handshake_header(&mut msg, HS_CERTIFICATE, &body);
        msg
    }

    /// Read a fixture from crates/bowser_net/tests/fixtures/tls/.
    fn tls_fixture(name: &str) -> Vec<u8> {
        std::fs::read(format!(
            "{}/tests/fixtures/tls/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("tls fixture")
    }

    /// Trust anchors for the handshake tests: the fixture CAs.
    fn test_roots() -> Vec<x509::ParsedCert> {
        ["test-ca.der", "ec-ca.der"]
            .iter()
            .map(|n| x509::parse_cert(&tls_fixture(n)).expect("fixture CA"))
            .collect()
    }

    /// A canned flight whose server_name matches the fixture leaf
    /// (`bowser.test`) and whose trust store is the fixture CAs.
    fn canned_flight_for(server_name: &str) -> CannedFlight {
        let mut f = canned_flight();
        f.hs.server_name = server_name.to_string();
        f.hs.trust_roots = Some(test_roots());
        f
    }

    #[test]
    fn certificate_message_with_real_chain_verifies() {
        let mut f = canned_flight_for("bowser.test");
        let leaf = tls_fixture("leaf-rsa.der");
        let inter = tls_fixture("test-inter.der");
        let msg = certificate_msg(&[&leaf, &inter]);
        f.hs.handle_certificate(&msg).unwrap();
        assert!(f.hs.leaf_spki.is_some());
        assert!(!f.hs.cert_verified, "CertificateVerify still pending");
    }

    #[test]
    fn certificate_message_with_garbage_cert_fails() {
        let mut f = canned_flight_for("bowser.test");
        let msg = certificate_msg(&[&[0xAA; 10]]);
        let r = f.hs.handle_certificate(&msg);
        assert!(
            matches!(r, Err(TlsError::CertificateVerificationFailed(_))),
            "garbage cert must fail, got {r:?}"
        );
    }

    #[test]
    fn certificate_message_with_untrusted_chain_fails() {
        // Signed by the evil CA, which is not in the test trust roots.
        let mut f = canned_flight_for("bowser.test");
        let leaf = tls_fixture("evil-leaf.der");
        let msg = certificate_msg(&[&leaf]);
        let r = f.hs.handle_certificate(&msg);
        assert!(
            matches!(r, Err(TlsError::CertificateVerificationFailed(_))),
            "untrusted chain must fail, got {r:?}"
        );
    }

    #[test]
    fn certificate_message_with_expired_chain_fails() {
        let mut f = canned_flight_for("bowser.test");
        let leaf = tls_fixture("leaf-rsa-expired.der");
        let inter = tls_fixture("test-inter.der");
        let msg = certificate_msg(&[&leaf, &inter]);
        let r = f.hs.handle_certificate(&msg);
        assert!(
            matches!(r, Err(TlsError::CertificateVerificationFailed(_))),
            "expired chain must fail, got {r:?}"
        );
    }

    #[test]
    fn certificate_message_with_wrong_host_fails() {
        let mut f = canned_flight_for("bowser.test");
        let leaf = tls_fixture("leaf-rsa-wronghost.der");
        let inter = tls_fixture("test-inter.der");
        let msg = certificate_msg(&[&leaf, &inter]);
        let r = f.hs.handle_certificate(&msg);
        assert!(
            matches!(r, Err(TlsError::CertificateVerificationFailed(_))),
            "wrong host must fail, got {r:?}"
        );
    }

    #[test]
    fn certificate_message_with_self_signed_fails() {
        let mut f = canned_flight_for("selfsigned.test");
        let leaf = tls_fixture("leaf-selfsigned.der");
        let msg = certificate_msg(&[&leaf]);
        let r = f.hs.handle_certificate(&msg);
        assert!(
            matches!(r, Err(TlsError::CertificateVerificationFailed(_))),
            "self-signed must fail, got {r:?}"
        );
    }

    #[test]
    fn certificate_message_with_empty_list_fails() {
        let mut f = canned_flight_for("bowser.test");
        let msg = certificate_msg(&[]);
        let r = f.hs.handle_certificate(&msg);
        assert!(
            matches!(r, Err(TlsError::CertificateVerificationFailed(_))),
            "empty list must fail, got {r:?}"
        );
    }

    /// Build a CertificateVerify handshake message.
    fn certificate_verify_msg(scheme: u16, sig: &[u8]) -> Vec<u8> {
        let mut body = Vec::new();
        body.push((scheme >> 8) as u8);
        body.push(scheme as u8);
        let len = sig.len();
        body.push((len >> 8) as u8);
        body.push(len as u8);
        body.extend_from_slice(sig);
        let mut msg = Vec::new();
        put_handshake_header(&mut msg, HS_CERTIFICATE_VERIFY, &body);
        msg
    }

    fn dehex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// RSA PKCS#1 v1.5 sign (SHA-256), test-only, using the fixture key.
    /// Signing lives here in the tests; production code only verifies.
    fn rsa_test_sign(n_hex: &str, d_hex: &str, msg: &[u8]) -> Vec<u8> {
        use crate::crypto::bignum::{BigUint, Montgomery};
        let n = BigUint::from_be_bytes(&dehex(n_hex));
        let d = BigUint::from_be_bytes(&dehex(d_hex));
        let k = n.bit_len().div_ceil(8);
        // DigestInfo for SHA-256.
        let prefix: [u8; 19] = [
            0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02,
            0x01, 0x05, 0x00, 0x04, 0x20,
        ];
        let digest = sha256(msg);
        let mut em = vec![0xff; k];
        em[0] = 0x00;
        em[1] = 0x01;
        let t_len = prefix.len() + digest.len();
        let ps_len = k - 3 - t_len;
        for b in em[2..2 + ps_len].iter_mut() {
            *b = 0xff;
        }
        em[2 + ps_len] = 0x00;
        em[2 + ps_len + 1..2 + ps_len + 1 + prefix.len()].copy_from_slice(&prefix);
        em[2 + ps_len + 1 + prefix.len()..].copy_from_slice(&digest);
        let mont = Montgomery::new(&n).unwrap();
        let m = BigUint::from_be_bytes(&em);
        // pow takes a plain integer and returns a plain integer.
        let s = mont.pow(&m, &d);
        s.to_be_bytes_padded(k).unwrap()
    }

    /// The fixture leaf's private key (n, d) in hex.
    fn leaf_rsa_key() -> (String, String) {
        let txt = String::from_utf8(tls_fixture("leaf-rsa-priv.txt")).unwrap();
        let mut n = String::new();
        let mut d = String::new();
        for line in txt.lines() {
            if let Some(v) = line.strip_prefix("n: ") {
                n = v.trim().to_string();
            } else if let Some(v) = line.strip_prefix("d: ") {
                d = v.trim().to_string();
            }
        }
        assert!(!n.is_empty() && !d.is_empty(), "fixture key must parse");
        (n, d)
    }

    #[test]
    fn full_handshake_with_real_chain_and_cv_succeeds() {
        // End-to-end: Certificate (real chain) -> CertificateVerify
        // (RSA-signed by the fixture leaf key) -> Finished. The client
        // must complete the handshake with cert_verified set.
        let mut f = canned_flight_for("bowser.test");
        let leaf = tls_fixture("leaf-rsa.der");
        let inter = tls_fixture("test-inter.der");
        f.hs.handle_certificate(&certificate_msg(&[&leaf, &inter]))
            .unwrap();

        // The CertificateVerify content per RFC 8446 §4.4.3, over the
        // transcript *before* the CertificateVerify message.
        let mut content = vec![b' '; 64];
        content.extend_from_slice(b"TLS 1.3, server CertificateVerify");
        content.push(0x00);
        content.extend_from_slice(&f.hs.transcript.digest(f.spec.hash));
        let (n, d) = leaf_rsa_key();
        let sig = rsa_test_sign(&n, &d, &content);
        f.hs.handle_certificate_verify(&certificate_verify_msg(0x0401, &sig))
            .unwrap();
        assert!(f.hs.cert_verified);

        let fin = canned_server_finished(&f);
        let client_fin = f.hs.handle_finished(&fin).unwrap();
        assert_eq!(client_fin[0], HS_FINISHED);
    }

    #[test]
    fn certificate_verify_with_bad_signature_fails() {
        let mut f = canned_flight_for("bowser.test");
        let leaf = tls_fixture("leaf-rsa.der");
        let inter = tls_fixture("test-inter.der");
        f.hs.handle_certificate(&certificate_msg(&[&leaf, &inter]))
            .unwrap();
        let (n, d) = leaf_rsa_key();
        let mut sig = rsa_test_sign(&n, &d, b"wrong content");
        // Corrupt one byte so the signature cannot verify.
        let last = sig.len() - 1;
        sig[last] ^= 0x01;
        let r =
            f.hs.handle_certificate_verify(&certificate_verify_msg(0x0401, &sig));
        assert!(
            matches!(r, Err(TlsError::CertificateVerificationFailed(_))),
            "bad CV signature must fail, got {r:?}"
        );
        assert!(!f.hs.cert_verified);
    }

    #[test]
    fn certificate_verify_with_unsupported_scheme_fails() {
        let mut f = canned_flight_for("bowser.test");
        let leaf = tls_fixture("leaf-rsa.der");
        let inter = tls_fixture("test-inter.der");
        f.hs.handle_certificate(&certificate_msg(&[&leaf, &inter]))
            .unwrap();
        // rsa_pss_rsae_sha256: not implemented, fails closed.
        let r =
            f.hs.handle_certificate_verify(&certificate_verify_msg(0x0804, &[0u8; 256]));
        assert!(
            matches!(r, Err(TlsError::CertificateVerificationFailed(_))),
            "unsupported scheme must fail, got {r:?}"
        );
    }

    #[test]
    fn unsupported_cipher_suite_rejected() {
        let mut hs = test_handshake();
        let _ = hs.client_hello().unwrap();
        let sh = server_hello_msg(&[0x55; 32], 0x1304, GROUP_X25519, &[0u8; 32], None);
        let r = hs.handle_server_hello(&sh);
        assert!(matches!(r, Err(TlsError::UnsupportedCipherSuite(0x1304))));
    }

    #[test]
    fn unoffered_alpn_rejected() {
        let mut hs = test_handshake();
        let ee = encrypted_extensions_msg(Some("bogus"));
        let r = hs.handle_encrypted_extensions(&ee);
        assert!(matches!(r, Err(TlsError::BadHandshake(_))));
    }

    // -- HelloRetryRequest -------------------------------------------------

    fn hrr_msg(extras: &[(u16, Vec<u8>)], cipher: u16) -> Vec<u8> {
        let mut body = Vec::new();
        put_u16(&mut body, LEGACY_HANDSHAKE_VERSION);
        body.extend_from_slice(&HRR_MAGIC);
        body.push(32u8);
        body.extend_from_slice(&[0x55; 32]);
        put_u16(&mut body, cipher);
        body.push(0u8);
        let mut exts = Vec::new();
        put_u16(&mut exts, EXT_SUPPORTED_VERSIONS);
        put_u16(&mut exts, 2);
        exts.extend_from_slice(&[0x03, 0x04]);
        put_u16(&mut exts, EXT_KEY_SHARE);
        put_u16(&mut exts, 2);
        put_u16(&mut exts, GROUP_X25519);
        for (id, data) in extras {
            put_u16(&mut exts, *id);
            put_u16(&mut exts, data.len() as u16);
            exts.extend_from_slice(data);
        }
        put_u16(&mut body, exts.len() as u16);
        body.extend_from_slice(&exts);
        let mut msg = Vec::new();
        put_handshake_header(&mut msg, HS_SERVER_HELLO, &body);
        msg
    }

    #[test]
    fn hrr_selecting_offered_group_rejected() {
        // m2: the HRR selects X25519, for which a key share was already
        // offered — abort per RFC 8446 §4.1.4 (no cookie to save it).
        let mut hs = test_handshake();
        let _ = hs.client_hello().unwrap();
        let hrr = hrr_msg(&[], TLS_AES_128_GCM_SHA256);
        let r = hs.handle_server_hello(&hrr);
        assert!(
            matches!(r, Err(TlsError::BadHandshake(_))),
            "HRR re-selecting the offered group must abort, got {r:?}"
        );
    }

    #[test]
    fn hrr_cookie_rejected() {
        // m1: a cookie-carrying HRR fails with the typed error — the
        // client never echoes cookies.
        let mut hs = test_handshake();
        let _ = hs.client_hello().unwrap();
        let hrr = hrr_msg(
            &[(EXT_COOKIE, vec![0x01, 0x02, 0x03])],
            TLS_AES_128_GCM_SHA256,
        );
        let r = hs.handle_server_hello(&hrr);
        assert!(matches!(r, Err(TlsError::HrrCookieUnsupported)));
    }

    #[test]
    fn hrr_duplicate_extension_rejected() {
        // m10: duplicated extensions abort instead of last-wins.
        // hrr_msg already emits supported_versions; appending another one
        // via extras makes it duplicated.
        let mut hs = test_handshake();
        let _ = hs.client_hello().unwrap();
        let hrr = hrr_msg(
            &[(EXT_SUPPORTED_VERSIONS, vec![0x03, 0x04])],
            TLS_AES_128_GCM_SHA256,
        );
        let r = hs.handle_server_hello(&hrr);
        assert!(matches!(r, Err(TlsError::BadHandshake(_))));
    }

    #[test]
    fn hrr_cipher_suite_change_rejected() {
        // The ServerHello after an HRR must not change the cipher suite.
        // (Driven white-box: hrr_done/cipher are set directly because a
        // real HRR cannot complete for this client — see m2.)
        let mut hs = test_handshake();
        let _ = hs.client_hello().unwrap();
        hs.cipher = cipher_spec(TLS_AES_128_GCM_SHA256);
        hs.hrr_done = true;
        let server_secret = [0x44; 32];
        let server_pub = x25519_public(&server_secret);
        let sh = server_hello_msg(
            &[0x55; 32],
            TLS_CHACHA20_POLY1305_SHA256,
            GROUP_X25519,
            &server_pub,
            None,
        );
        let r = hs.handle_server_hello(&sh);
        assert!(
            matches!(r, Err(TlsError::BadHandshake(_))),
            "cipher suite change across HRR must abort, got {r:?}"
        );
    }

    #[test]
    fn hrr_forbidden_extension_rejected() {
        let mut hs = test_handshake();
        let _ = hs.client_hello().unwrap();
        let hrr = hrr_msg(&[(EXT_SERVER_NAME, vec![])], TLS_AES_128_GCM_SHA256);
        let r = hs.handle_server_hello(&hrr);
        assert!(matches!(r, Err(TlsError::UnexpectedMessage(_))));
    }

    #[test]
    fn hrr_magic_value() {
        assert_eq!(HRR_MAGIC.len(), 32);
        assert_eq!(&HRR_MAGIC[..4], &[0xCF, 0x21, 0xAD, 0x74]);
        assert_eq!(&HRR_MAGIC[28..], &[0xC8, 0xA8, 0x33, 0x9C]);
    }

    // -- Record layer ------------------------------------------------------

    #[test]
    fn record_seal_open_roundtrip() {
        let keys = TrafficKeys {
            aead: AeadKind::Aes128Gcm,
            key: vec![0x42; 16],
            iv: [0x24; 12],
        };
        let mut rec = seal_record(&keys, 7, RECORD_APP_DATA, b"hello world").unwrap();
        assert_eq!(rec[0], RECORD_APP_DATA);
        assert_eq!(u16::from_be_bytes([rec[1], rec[2]]), LEGACY_RECORD_VERSION);
        // Keep a pristine sealed copy: `open_record` decrypts in place.
        let sealed = rec.clone();
        let (inner, pt) = open_record(&keys, 7, &mut rec).unwrap();
        assert_eq!(inner, RECORD_APP_DATA);
        assert_eq!(pt, b"hello world");

        // Wrong sequence number fails.
        assert!(matches!(
            open_record(&keys, 8, &mut rec),
            Err(TlsError::DecryptFailed)
        ));
        // Tampered ciphertext fails.
        let mut bad = sealed;
        let i = bad.len() - 20;
        bad[i] ^= 0x01;
        assert!(matches!(
            open_record(&keys, 7, &mut bad),
            Err(TlsError::DecryptFailed)
        ));

        // All three AEADs round-trip.
        for (aead, key) in [
            (AeadKind::Aes256Gcm, vec![0x11; 32]),
            (AeadKind::ChaCha20Poly1305, vec![0x33; 32]),
        ] {
            let k = TrafficKeys {
                aead,
                key,
                iv: [0x77; 12],
            };
            let mut r = seal_record(&k, 0, 22, b"data").unwrap();
            let (t, p) = open_record(&k, 0, &mut r).unwrap();
            assert_eq!(t, 22);
            assert_eq!(p, b"data");
        }
    }

    #[test]
    fn seal_record_single_allocation() {
        let keys = TrafficKeys {
            aead: AeadKind::Aes128Gcm,
            key: vec![0x42; 16],
            iv: [0x24; 12],
        };
        let plaintext = b"single allocation check";
        let rec = seal_record(&keys, 3, RECORD_APP_DATA, plaintext).unwrap();
        // header + plaintext + inner-type byte + tag, with no spare
        // capacity: the buffer was sized once up front and never reallocated.
        assert_eq!(rec.len(), 5 + plaintext.len() + 1 + TAG_LEN);
        assert_eq!(rec.capacity(), rec.len());
    }

    #[test]
    fn open_record_flipped_tag_fails_closed() {
        let keys = TrafficKeys {
            aead: AeadKind::ChaCha20Poly1305,
            key: vec![0x33; 32],
            iv: [0x77; 12],
        };
        let plaintext = b"top secret payload";
        let mut rec = seal_record(&keys, 9, RECORD_APP_DATA, plaintext).unwrap();
        // Flip a tag byte: authentication must fail, and no plaintext may
        // come out — the buffer must still hold ciphertext, untouched.
        let tag_idx = rec.len() - 1;
        rec[tag_idx] ^= 0x01;
        let before = rec.clone();
        assert!(matches!(
            open_record(&keys, 9, &mut rec),
            Err(TlsError::DecryptFailed)
        ));
        assert_eq!(rec, before, "failed open must leave the buffer untouched");
        assert_ne!(
            &rec[5..5 + plaintext.len()],
            plaintext,
            "failed open must not reveal plaintext"
        );
    }

    #[test]
    fn alert_parsing() {
        assert!(matches!(parse_alert(&[2, 40]), TlsError::AlertReceived(40)));
        assert!(matches!(parse_alert(&[2]), TlsError::BadRecord(_)));
        assert!(matches!(parse_alert(&[1, 0]), TlsError::AlertReceived(0)));
    }

    #[test]
    fn framer_reassembly() {
        let mut m1 = Vec::new();
        put_handshake_header(&mut m1, HS_ENCRYPTED_EXTENSIONS, &[1, 2, 3]);
        let mut m2 = Vec::new();
        put_handshake_header(&mut m2, HS_FINISHED, &[4, 5]);

        // Coalesced messages in one fragment.
        let mut f = HandshakeFramer::new();
        f.push(&[m1.clone(), m2.clone()].concat()).unwrap();
        let (t1, b1) = f.next_message().unwrap().unwrap();
        assert_eq!((t1, b1), (HS_ENCRYPTED_EXTENSIONS, m1.clone()));
        let (t2, b2) = f.next_message().unwrap().unwrap();
        assert_eq!((t2, b2), (HS_FINISHED, m2));
        assert!(f.next_message().unwrap().is_none());

        // Fragmented message across pushes.
        f.push(&m1[..3]).unwrap();
        assert!(f.next_message().unwrap().is_none());
        f.push(&m1[3..]).unwrap();
        let (t3, b3) = f.next_message().unwrap().unwrap();
        assert_eq!((t3, b3), (HS_ENCRYPTED_EXTENSIONS, m1));
    }

    #[test]
    fn framer_rejects_unbounded_growth() {
        // m7: drip-fed fragments that never form a header must not grow
        // the buffer without bound.
        let mut f = HandshakeFramer::new();
        let mut pushes = 0u32;
        let r = loop {
            pushes += 1;
            assert!(pushes < 200_000, "framer grew without bound");
            match f.push(&[0u8; 3]) {
                Ok(()) => continue,
                Err(e) => break e,
            }
        };
        assert!(
            matches!(r, TlsError::BadHandshake(_)),
            "expected BadHandshake, got {r:?}"
        );
        // A header declaring an oversize message is rejected too.
        let mut f2 = HandshakeFramer::new();
        let mut big = Vec::new();
        put_handshake_header(&mut big, HS_CERTIFICATE, &vec![0u8; MAX_HANDSHAKE_MSG + 1]);
        f2.push(&big).unwrap_err();
    }

    #[test]
    fn new_session_ticket_parses() {
        let mut body = Vec::new();
        body.extend_from_slice(&3600u32.to_be_bytes());
        body.extend_from_slice(&1234u32.to_be_bytes());
        body.push(1u8);
        body.push(0x99);
        body.extend_from_slice(&[0u8, 4u8]);
        body.extend_from_slice(b"tick");
        body.extend_from_slice(&[0u8, 0u8]);
        let mut msg = Vec::new();
        put_handshake_header(&mut msg, HS_NEW_SESSION_TICKET, &body);
        let nst = parse_new_session_ticket(&msg).unwrap();
        assert_eq!(nst.lifetime, 3600);
        assert_eq!(nst.age_add, 1234);
        assert_eq!(nst.nonce, vec![0x99]);
        assert_eq!(nst.ticket, b"tick");
        assert!(parse_new_session_ticket(&msg[..msg.len() - 1]).is_err());
    }

    #[test]
    fn hkdf_expand_label_matches_manual_info() {
        let secret = vec![0x0b; 32];
        let got = hkdf_expand_label(HashAlg::Sha256, &secret, b"key", b"ctx", 16).unwrap();
        let mut info = Vec::new();
        info.extend_from_slice(&16u16.to_be_bytes());
        info.push(9u8);
        info.extend_from_slice(b"tls13 key");
        info.push(3u8);
        info.extend_from_slice(b"ctx");
        let want = hkdf_expand(HashAlg::Sha256, &secret, &info, 16).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn hkdf_expand_label_overlong_inputs_error_instead_of_panicking() {
        // The NewSessionTicket nonce is attacker-controlled and feeds the
        // label/context path: over-long inputs must return an error in
        // every build, never panic on the fixed info buffer.
        let secret = vec![0x0b; 32];
        let r = hkdf_expand_label(HashAlg::Sha256, &secret, &[b'x'; 33], b"ctx", 16);
        assert!(
            matches!(r, Err(TlsError::Crypto(CryptoError::InvalidLength))),
            "over-long label must error, got {r:?}"
        );
        let r = hkdf_expand_label(HashAlg::Sha256, &secret, b"key", &[b'y'; 256], 16);
        assert!(
            matches!(r, Err(TlsError::Crypto(CryptoError::InvalidLength))),
            "over-long context must error, got {r:?}"
        );
        // Boundary inputs still work.
        hkdf_expand_label(HashAlg::Sha256, &secret, &[b'x'; 32], b"ctx", 16).unwrap();
        // A 255-byte nonce is the RFC 8446 §4.6.1 maximum and must
        // derive a ticket PSK, not fail the connection.
        hkdf_expand_label(HashAlg::Sha256, &secret, b"resumption", &[b'z'; 255], 32).unwrap();
    }

    #[test]
    fn dummy_ccs_records_are_validated() {
        // Well-formed middlebox-compatibility CCS: type 20, one payload
        // byte of value 1, version exactly 0x0303.
        assert!(valid_dummy_ccs(&[20, 3, 3, 0, 1, 1]));
        // Wrong version (0x0301 is not accepted), wrong payload, wrong
        // length, empty.
        assert!(!valid_dummy_ccs(&[20, 3, 1, 0, 1, 1]));
        assert!(!valid_dummy_ccs(&[20, 3, 2, 0, 1, 1]));
        assert!(!valid_dummy_ccs(&[20, 3, 3, 0, 1, 2]));
        assert!(!valid_dummy_ccs(&[20, 3, 3, 0, 2, 1, 1]));
        assert!(!valid_dummy_ccs(&[]));
    }

    #[test]
    fn stalled_handshake_records_are_bounded() {
        let mut stalled = 0u8;
        // Zero-byte records are tolerated up to the cap...
        for _ in 0..MAX_STALLED_HANDSHAKE_RECORDS {
            note_fragment_progress(false, &mut stalled).unwrap();
        }
        // ...then fail closed, so an empty-record flood cannot spin the
        // handshake loops forever.
        let r = note_fragment_progress(false, &mut stalled);
        assert!(
            matches!(r, Err(TlsError::UnexpectedMessage(_))),
            "stall flood must fail, got {r:?}"
        );
        // A record carrying fragment bytes resets the count...
        note_fragment_progress(true, &mut stalled).unwrap();
        assert_eq!(stalled, 0);
        // ...so the flood must start over to trip the bound.
        for _ in 0..MAX_STALLED_HANDSHAKE_RECORDS {
            note_fragment_progress(false, &mut stalled).unwrap();
        }
        assert!(note_fragment_progress(false, &mut stalled).is_err());
    }

    #[test]
    fn prefix_stream_feeds_queued_bytes_first() {
        // Bytes over-read past a CONNECT response head must reach the
        // handshake before the socket's own bytes.
        let (client, mut server) = socketpair();
        server.write_all(b"SOCKET").unwrap();
        let mut ps = PrefixStream {
            prefix: b"PREF".to_vec(),
            pos: 0,
            stream: client,
        };
        let mut buf = [0u8; 8];
        ps.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"PREFSOCK");
        // Prefix fully drained: nothing is stashed, nothing is lost.
        let (_stream, stash) = ps.into_parts();
        assert!(stash.is_empty());
    }

    #[test]
    fn prefix_stream_keeps_unconsumed_bytes() {
        let (client, _server) = socketpair();
        let ps = PrefixStream {
            prefix: b"ABCDEF".to_vec(),
            pos: 4,
            stream: client,
        };
        let (_stream, stash) = ps.into_parts();
        assert_eq!(stash, b"EF");
    }

    #[test]
    fn grease_values_wellformed() {
        for seed in [0x00u8, 0x0F, 0x10, 0xAB, 0xFF] {
            let g = grease_value(seed);
            assert!(is_grease(g), "0x{g:04x}");
            assert!(!is_grease(0x1301));
        }
        assert_eq!(grease_value(0x00), 0x0A0A);
        assert_eq!(grease_value(0xFF), 0xFAFA);
    }

    // ---------- message flight order (m4) ----------

    #[test]
    fn finished_before_ee_rejected() {
        // Finished before EncryptedExtensions is out of order.
        let f = canned_flight();
        let mut hs2 = test_handshake();
        let _ = hs2.client_hello().unwrap();
        let server_secret = [0x44; 32];
        let server_pub = x25519_public(&server_secret);
        let shared = x25519::shared_secret(&server_secret, &x25519_public(&[0x33; 32]));
        let sh = server_hello_msg(
            &[0x55; 32],
            TLS_AES_128_GCM_SHA256,
            GROUP_X25519,
            &server_pub,
            None,
        );
        assert!(matches!(
            hs2.handle_server_hello(&sh).unwrap(),
            HelloAction::Proceed
        ));
        let hash = HashAlg::Sha256;
        let thash_ch_sh = hs2.transcript.digest(hash);
        let (_, s_hs_traffic, _) = server_schedule(hash, &shared, &thash_ch_sh);
        // Well-formed Finished for CH..SH, but no EE seen yet.
        let fin = server_finished_msg(hash, &s_hs_traffic, &thash_ch_sh);
        let r = hs2.handle_finished(&fin);
        assert!(
            matches!(r, Err(TlsError::UnexpectedMessage(_))),
            "Finished before EE must be rejected as out-of-order, got {r:?}"
        );
        let _ = f;
    }

    #[test]
    fn certificate_verify_without_certificate_rejected() {
        let mut f = canned_flight();
        let msg = {
            let mut m = Vec::new();
            put_handshake_header(&mut m, HS_CERTIFICATE_VERIFY, &[0u8; 8]);
            m
        };
        let r = f.hs.handle_certificate_verify(&msg);
        assert!(matches!(r, Err(TlsError::UnexpectedMessage(_))));
    }

    #[test]
    fn duplicate_ee_rejected() {
        let mut f = canned_flight();
        let ee = encrypted_extensions_msg(None);
        let r = f.hs.handle_encrypted_extensions(&ee);
        assert!(matches!(r, Err(TlsError::UnexpectedMessage(_))));
    }

    #[test]
    fn certificate_after_cv_rejected() {
        // A Certificate arriving after CertificateVerify is out of order.
        let mut f = canned_flight();
        f.hs.saw_cert = true;
        f.hs.saw_cv = true;
        let mut body = Vec::new();
        body.push(0u8);
        body.extend_from_slice(&[0u8, 0u8, 0u8]);
        let mut msg = Vec::new();
        put_handshake_header(&mut msg, HS_CERTIFICATE, &body);
        let r = f.hs.handle_certificate(&msg);
        assert!(matches!(r, Err(TlsError::UnexpectedMessage(_))));
    }

    // ---------- duplicate extensions (m10) ----------

    fn sh_with_extra_extensions(extra: &[u8]) -> Vec<u8> {
        let server_secret = [0x44; 32];
        let server_pub = x25519_public(&server_secret);
        let mut sh = server_hello_msg(
            &[0x55; 32],
            TLS_AES_128_GCM_SHA256,
            GROUP_X25519,
            &server_pub,
            None,
        );
        // ServerHello layout: 4 hdr + 2 ver + 32 rand + 1 sidlen + 32 sid +
        // 2 cipher + 1 comp = 74; extension block (u16 len) follows.
        let ext_off = 4 + 2 + 32 + 1 + 32 + 2 + 1;
        let old_len = u16::from_be_bytes([sh[ext_off], sh[ext_off + 1]]) as usize;
        let add = extra.len();
        sh[ext_off] = ((old_len + add) >> 8) as u8;
        sh[ext_off + 1] = (old_len + add) as u8;
        sh.splice(
            ext_off + 2 + old_len..ext_off + 2 + old_len,
            extra.iter().copied(),
        );
        // Fix the handshake header length.
        let total = sh.len() - 4;
        sh[1] = (total >> 16) as u8;
        sh[2] = (total >> 8) as u8;
        sh[3] = total as u8;
        sh
    }

    #[test]
    fn duplicate_sh_extension_rejected() {
        // m10: a ServerHello that repeats supported_versions aborts.
        let mut hs = test_handshake();
        let _ = hs.client_hello().unwrap();
        let mut dup = Vec::new();
        put_u16(&mut dup, EXT_SUPPORTED_VERSIONS);
        put_u16(&mut dup, 2);
        dup.extend_from_slice(&[0x03, 0x04]);
        let sh = sh_with_extra_extensions(&dup);
        let r = hs.handle_server_hello(&sh);
        assert!(matches!(r, Err(TlsError::BadHandshake(_))));
    }

    #[test]
    fn duplicate_ee_extension_rejected() {
        // m10: a duplicated server_name in EE aborts. The second EE
        // instance itself is already rejected by ordering; drive the
        // parser directly on a second flight via a fresh handshake.
        let mut hs = test_handshake();
        let _ = hs.client_hello().unwrap();
        let server_secret = [0x44; 32];
        let server_pub = x25519_public(&server_secret);
        let sh = server_hello_msg(
            &[0x55; 32],
            TLS_AES_128_GCM_SHA256,
            GROUP_X25519,
            &server_pub,
            None,
        );
        assert!(matches!(
            hs.handle_server_hello(&sh).unwrap(),
            HelloAction::Proceed
        ));
        let mut ee = Vec::new();
        let mut exts = Vec::new();
        put_u16(&mut exts, EXT_SERVER_NAME);
        put_u16(&mut exts, 0);
        put_u16(&mut exts, EXT_SERVER_NAME);
        put_u16(&mut exts, 0);
        put_u16(&mut ee, exts.len() as u16);
        ee.extend_from_slice(&exts);
        let mut msg = Vec::new();
        put_handshake_header(&mut msg, HS_ENCRYPTED_EXTENSIONS, &ee);
        let r = hs.handle_encrypted_extensions(&msg);
        assert!(matches!(r, Err(TlsError::BadHandshake(_))));
    }

    // ---------- plaintext flight hygiene (m5/m6) ----------

    fn read_one_record(stream: &mut TcpStream) -> io::Result<Vec<u8>> {
        let mut hdr = [0u8; 5];
        stream.read_exact(&mut hdr)?;
        let len = u16::from_be_bytes([hdr[3], hdr[4]]) as usize;
        let mut body = vec![0u8; len];
        stream.read_exact(&mut body)?;
        let mut rec = hdr.to_vec();
        rec.extend_from_slice(&body);
        Ok(rec)
    }

    /// Append a plaintext record header (version marker + length).
    fn write_record_header(out: &mut Vec<u8>, ctype: u8, len: u16) {
        out.push(ctype);
        out.extend_from_slice(&LEGACY_RECORD_VERSION.to_be_bytes());
        out.extend_from_slice(&len.to_be_bytes());
    }

    /// Drive `TlsConnection::read_server_hello` against a canned plaintext
    /// flight built from the client's real session id, and return the
    /// client's verdict.
    fn client_read_sh(
        build: impl FnOnce(&[u8]) -> Vec<u8> + Send + 'static,
    ) -> Result<HelloAction, TlsError> {
        let (client_stream, mut server_stream) = socketpair();
        let config = test_config();
        let handle = std::thread::spawn(move || {
            let mut c = client_stream;
            let mut hs = Handshake::new(&config).unwrap();
            let ch = hs.client_hello().unwrap();
            write_record(&mut c, RECORD_HANDSHAKE, LEGACY_RECORD_VERSION, &ch).unwrap();
            let mut framer = HandshakeFramer::new();
            TlsConnection::read_server_hello(&mut c, &mut hs, &mut framer)
        });
        let ch_rec = read_one_record(&mut server_stream).unwrap();
        let sid = parse_client_session_id(&ch_rec);
        server_stream.write_all(&build(&sid)).unwrap();
        handle.join().unwrap()
    }

    #[test]
    fn take_stash_record_moves_without_copying() {
        // The per-record data path must hand the record out of the
        // stash without copying it: the returned buffer keeps the
        // stash's own allocation, and only the remainder moves.
        let mut stash = Vec::with_capacity(64);
        stash.extend_from_slice(b"RECORD1234remainder");
        let rec = take_stash_record(&mut stash, 10);
        assert_eq!(rec, b"RECORD1234");
        assert_eq!(stash, b"remainder");
        assert!(
            rec.capacity() >= 64,
            "record must reuse the stash buffer, cap={}",
            rec.capacity()
        );
        // Exact-length take leaves an empty stash with no allocation.
        let mut stash = b"12345".to_vec();
        let rec = take_stash_record(&mut stash, 5);
        assert_eq!(rec, b"12345");
        assert!(stash.is_empty());
    }

    #[test]
    fn trailing_plaintext_after_server_hello_rejected() {
        // m5: bytes left in the plaintext framer after ServerHello are
        // rejected instead of being decrypted as handshake data.
        let r = client_read_sh(|sid| {
            let server_secret = [0x44; 32];
            let server_pub = x25519_public(&server_secret);
            let sh = server_hello_msg(sid, TLS_AES_128_GCM_SHA256, GROUP_X25519, &server_pub, None);
            // One record carrying ServerHello followed by a stray
            // handshake fragment in the same plaintext flight.
            let mut stray = Vec::new();
            put_handshake_header(&mut stray, HS_ENCRYPTED_EXTENSIONS, &[1, 2, 3]);
            let mut rec = Vec::new();
            write_record_header(&mut rec, RECORD_HANDSHAKE, (sh.len() + stray.len()) as u16);
            rec.extend_from_slice(&sh);
            rec.extend_from_slice(&stray);
            rec
        });
        assert!(
            matches!(r, Err(TlsError::BadHandshake(_))),
            "trailing plaintext must be rejected, got {r:?}"
        );
    }

    #[test]
    fn ccs_flood_aborts() {
        // m6: more than MAX_IGNORED_CCS ChangeCipherSpec records aborts
        // the ServerHello flight.
        let r = client_read_sh(|sid| {
            let server_secret = [0x44; 32];
            let server_pub = x25519_public(&server_secret);
            let sh = server_hello_msg(sid, TLS_AES_128_GCM_SHA256, GROUP_X25519, &server_pub, None);
            let mut flight = Vec::new();
            for _ in 0..=MAX_IGNORED_CCS {
                // Well-formed dummy CCS with the RFC-mandated 0x0303
                // version, built manually: the test helper writes 0x0301.
                flight.extend_from_slice(&[20, 3, 3, 0, 1, 1]);
            }
            write_record_header(&mut flight, RECORD_HANDSHAKE, sh.len() as u16);
            flight.extend_from_slice(&sh);
            flight
        });
        assert!(
            matches!(r, Err(TlsError::UnexpectedMessage(_))),
            "CCS flood must abort, got {r:?}"
        );
    }

    // ---------- alert handling (m8) ----------

    #[test]
    fn fatal_close_notify_is_not_clean_shutdown() {
        // m8: close_notify with fatal level (2) is not a clean shutdown;
        // only warning(1)/close_notify(0) is.
        assert!(is_clean_close_notify(&[1u8, 0u8]).unwrap());
        assert!(!is_clean_close_notify(&[2u8, 0u8]).unwrap());
        assert!(!is_clean_close_notify(&[1u8, 10u8]).unwrap());
        assert!(is_clean_close_notify(&[1u8]).is_err());
        assert!(is_clean_close_notify(&[1u8, 0u8, 0u8]).is_err());
    }

    // ---------- RFC 8448 key-schedule vectors (M1a) ----------

    fn hex(s: &str) -> Vec<u8> {
        assert!(s.len().is_multiple_of(2));
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn key_schedule_matches_rfc8448_vectors() {
        // RFC 8448 §3 "Simple 1-RTT Handshake", TLS_AES_128_GCM_SHA256.
        let hash = HashAlg::Sha256;
        let zeros = vec![0u8; hash_len(hash)];

        // Early secret: HKDF-Extract(zeros, zeros).
        let early = hkdf_extract(hash, &zeros, &zeros);
        assert_eq!(
            early,
            hex("33ad0a1c607ec03b09e6cd9893680ce210adf300aa1f2660e1b22e10f170f92a").as_slice()
        );

        // Handshake secret from the RFC's (EC)DHE shared secret.
        let shared = hex("8bd4054fb55b9d63fdfbacf9f04b9f0d35e6d63f537563efd46272900f89492d");
        let derived = derive_secret(hash, &early, b"derived", &transcript_hash(hash, b"")).unwrap();
        let hs_secret = hkdf_extract(hash, &derived, &shared);
        assert_eq!(
            hs_secret,
            hex("1dc826e93606aa6fdc0aadc12f741b01046aa6b99f691ed221a9f0ca043fbeac").as_slice()
        );

        // Handshake traffic secrets from the RFC's CH..SH transcript hash.
        let ch_sh_th = hex("860c06edc07858ee8e78f0e7428c58edd6b43f2ca3e6e95f02ed063cf0e1cad8");
        let c_hs = derive_secret(hash, &hs_secret, b"c hs traffic", &ch_sh_th).unwrap();
        let s_hs = derive_secret(hash, &hs_secret, b"s hs traffic", &ch_sh_th).unwrap();
        assert_eq!(
            c_hs,
            hex("b3eddb126e067f35a780b3abf45e2d8f3b1a950738f52e9600746a0e27a55a21").as_slice()
        );
        assert_eq!(
            s_hs,
            hex("b67b7d690cc16c4e75e54213cb2d37b4e9c912bcded9105d42befd59d391ad38").as_slice()
        );

        // Server handshake key/iv.
        let spec = cipher_spec(TLS_AES_128_GCM_SHA256).unwrap();
        let s_keys = traffic_keys(hash, &s_hs, &spec).unwrap();
        assert_eq!(
            s_keys.key,
            hex("3fce516009c21727d0f2e4e86ee403bc").as_slice()
        );
        assert_eq!(s_keys.iv, hex("5d313eb2671276ee13000b30").as_slice());

        // Server Finished key and verify_data. RFC 8448 §3.5: the finished
        // key's context is empty, not the transcript hash.
        let fk = finished_key(hash, &s_hs).unwrap();
        assert_eq!(
            fk,
            hex("008d3b66f816ea559f96b537e885c31fc068bf492c652f01f288a1d8cdc19fc8").as_slice()
        );
        // verify_data is HMAC(fk, transcript-hash through CertificateVerify).
        // RFC 8448 does not print this transcript hash, but it is pinned by
        // the RFC's verify_data: HMAC(fk, th) must equal the RFC's
        // 9b9b141d... value, which holds for the hash below.
        let cv_th = hex("edb7725fa7a3473b031ec8ef65a2485493900138a2b91291407d7951a06110ed");
        let vd = hmac(hash, &fk, &cv_th);
        assert_eq!(
            vd,
            hex("9b9b141d906337fbd2cbdce71df4deda4ab42c309572cb7fffee5454b78f0718").as_slice()
        );

        // Application traffic secrets from the transcript through the
        // server Finished.
        let sfin_th = hex("9608102a0f1ccc6db6250b7b7e417b1a000eaada3daae4777a7686c9ff83df13");
        let master = {
            let d =
                derive_secret(hash, &hs_secret, b"derived", &transcript_hash(hash, b"")).unwrap();
            hkdf_extract(hash, &d, &zeros)
        };
        assert_eq!(
            master,
            hex("18df06843d13a08bf2a449844c5f8a478001bc4d4c627984d5a41da8d0402919").as_slice()
        );
        let c_ap = derive_secret(hash, &master, b"c ap traffic", &sfin_th).unwrap();
        let s_ap = derive_secret(hash, &master, b"s ap traffic", &sfin_th).unwrap();
        assert_eq!(
            c_ap,
            hex("9e40646ce79a7f9dc05af8889bce6552875afa0b06df0087f792ebb7c17504a5").as_slice()
        );
        assert_eq!(
            s_ap,
            hex("a11af9f05531f856ad47116b45a950328204b4f44bfb6b3a4b4f1f3fcb631643").as_slice()
        );
    }

    // ---------- established-connection read/close (C2, M1c, m11) ----------

    fn socketpair() -> (TcpStream, TcpStream) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        client
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        server
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        (client, server)
    }

    fn test_connection(stream: TcpStream, keys: TrafficKeys, seq: u64) -> TlsConnection {
        TlsConnection {
            stream,
            read_keys: keys.clone(),
            read_seq: seq,
            write_keys: keys,
            write_seq: 0,
            pending: Vec::new(),
            framer: HandshakeFramer::new(),
            alpn: None,
            cipher_suite: TLS_AES_128_GCM_SHA256,
            closed: false,
            resumption_master_secret: Vec::new(),
            resumption_hash: HashAlg::Sha256,
            received_tickets: Vec::new(),
            stash: Vec::new(),
            read_deadline: None,
        }
    }

    fn test_keys() -> TrafficKeys {
        traffic_keys(
            HashAlg::Sha256,
            &hex("a11af9f05531f856ad47116b45a950328204b4f44bfb6b3a4b4f1f3fcb631643"),
            &cipher_spec(TLS_AES_128_GCM_SHA256).unwrap(),
        )
        .unwrap()
    }

    fn send_app_record(peer: &mut TcpStream, keys: &TrafficKeys, seq: u64, pt: &[u8]) {
        let rec = seal_record(keys, seq, RECORD_APP_DATA, pt).unwrap();
        peer.write_all(&rec).unwrap();
    }

    fn send_alert(peer: &mut TcpStream, keys: &TrafficKeys, seq: u64, alert: &[u8; 2]) {
        let rec = seal_record(keys, seq, RECORD_ALERT, alert).unwrap();
        peer.write_all(&rec).unwrap();
    }

    #[test]
    fn read_returns_buffered_app_data() {
        // C2: read() returns decrypted application data immediately and
        // does not block waiting for more.
        let (client_stream, mut server_stream) = socketpair();
        let keys = test_keys();
        let mut conn = test_connection(client_stream, keys.clone(), 0);

        send_app_record(&mut server_stream, &keys, 0, b"hello");
        let mut buf = [0u8; 64];
        let n = conn.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"hello");
    }

    #[test]
    fn read_skips_empty_application_data() {
        // Critical net issue #3: RFC 8446 §5.1 permits empty
        // ApplicationData fragments (traffic-analysis padding). open_record
        // returns an empty plaintext for those, and read() used to surface
        // that as Ok(0), which callers misreport as EOF (truncating
        // Content-Length/chunked bodies in fetch's read_more and breaking
        // UntilClose early). Empty fragments must be skipped instead.
        let (client_stream, mut server_stream) = socketpair();
        let keys = test_keys();
        let mut conn = test_connection(client_stream, keys.clone(), 0);

        // Empty record first (seq 0), then real data (seq 1). The first
        // read() must not return Ok(0) for the empty fragment.
        send_app_record(&mut server_stream, &keys, 0, b"");
        send_app_record(&mut server_stream, &keys, 1, b"hello");
        let mut buf = [0u8; 64];
        let n = conn.read(&mut buf).unwrap();
        assert_eq!(n, 5, "empty app-data fragment must not report EOF");
        assert_eq!(&buf[..n], b"hello");

        // Multiple consecutive empty fragments are skipped as well.
        send_app_record(&mut server_stream, &keys, 2, b"");
        send_app_record(&mut server_stream, &keys, 3, b"");
        send_app_record(&mut server_stream, &keys, 4, b"bye");
        let n = conn.read(&mut buf).unwrap();
        assert_eq!(n, 3);
        assert_eq!(&buf[..n], b"bye");
    }

    #[test]
    fn write_seq_exhaustion_fails_closed() {
        // m5: the 64-bit record sequence must never wrap (RFC 8446
        // §5.3 — nonce reuse would break the AEAD). On exhaustion the
        // connection fails closed: write() errors and the connection is
        // marked closed instead of sending with sequence 0 again.
        let (client_stream, _server_stream) = socketpair();
        let keys = test_keys();
        let mut conn = test_connection(client_stream, keys, 0);
        conn.write_seq = u64::MAX;
        let r = conn.write(b"hello");
        assert!(r.is_err(), "exhausted sequence must fail, got {r:?}");
        assert!(conn.is_closed(), "connection must close on exhaustion");
        assert_eq!(conn.write_seq, u64::MAX, "sequence must not wrap");
        // Further writes fail as closed.
        assert!(conn.write(b"hello").is_err());
    }

    #[test]
    fn read_seq_exhaustion_fails_closed() {
        // m5b: the receive-side 64-bit record sequence must never wrap
        // either (RFC 8446 §5.3 — nonce reuse would break the AEAD). On
        // exhaustion read() fails and the connection closes.
        let (client_stream, mut server_stream) = socketpair();
        let keys = test_keys();
        let mut conn = test_connection(client_stream, keys.clone(), u64::MAX);
        send_app_record(&mut server_stream, &keys, u64::MAX, b"hi");
        let mut buf = [0u8; 8];
        let r = conn.read(&mut buf);
        assert!(r.is_err(), "exhausted read sequence must fail, got {r:?}");
        assert!(conn.is_closed(), "connection must close on exhaustion");
        assert_eq!(conn.read_seq, u64::MAX, "sequence must not wrap");
    }

    #[test]
    fn read_drains_pending_before_close_notify() {
        // C2: buffered plaintext is returned before the close_notify is
        // honored.
        let (client_stream, mut server_stream) = socketpair();
        let keys = test_keys();
        let mut conn = test_connection(client_stream, keys.clone(), 0);

        send_app_record(&mut server_stream, &keys, 0, b"abc");
        send_alert(&mut server_stream, &keys, 1, &[1u8, 0u8]);

        let mut buf = [0u8; 2];
        assert_eq!(conn.read(&mut buf).unwrap(), 2);
        assert_eq!(&buf, b"ab");
        assert_eq!(conn.read(&mut buf).unwrap(), 1);
        assert_eq!(&buf[..1], b"c");
        // Pending drained; now the close_notify surfaces as EOF.
        let mut big = [0u8; 64];
        assert_eq!(conn.read(&mut big).unwrap(), 0);
        assert!(conn.is_closed());
    }

    #[test]
    fn plaintext_alert_after_handshake_is_fatal() {
        // Critical net issue #1: after handshake completion RFC 8446
        // requires every record AEAD-protected (outer type always
        // application_data). An outer plaintext alert record must be
        // rejected as unexpected_message (fatal), never honored as a
        // clean close — a plaintext close_notify is a truncation
        // attack vector.
        let (client_stream, mut server_stream) = socketpair();
        let keys = test_keys();
        let mut conn = test_connection(client_stream, keys, 0);

        // Plaintext alert record: warning-level close_notify, sent
        // unprotected (outer type 21).
        let mut rec = Vec::new();
        write_record_header(&mut rec, RECORD_ALERT, 2);
        rec.extend_from_slice(&[1u8, ALERT_CLOSE_NOTIFY]);
        server_stream.write_all(&rec).unwrap();

        let mut buf = [0u8; 64];
        let r = conn.read(&mut buf);
        assert!(r.is_err(), "plaintext alert must be fatal, got {r:?}");
        assert_eq!(r.unwrap_err().kind(), io::ErrorKind::InvalidData);
        assert!(conn.is_closed(), "connection must fail closed");
        // Fail-closed: no further reads or writes are honored.
        assert_eq!(conn.read(&mut buf).unwrap(), 0);
        assert!(conn.write(b"hi").is_err());
    }

    #[test]
    fn tls_read_deadline_stops_drip_peer() {
        // The H2 pool arms an absolute read deadline on the TLS half
        // while a session is pumped under a tightened timeout slice:
        // one `TlsConnection::read` loops internally in
        // `read_record_buffered` until a full record arrives, so the
        // per-read socket timeout alone would let a peer drip-feed a
        // record byte at a time and stretch a single read to (record
        // bytes) x timeout. The deadline is checked before every
        // blocking socket read, so the read ends at the deadline plus
        // at most one per-read socket timeout.
        const BUDGET: Duration = Duration::from_millis(400);
        let (client_stream, server_stream) = socketpair();
        let keys = test_keys();
        let mut conn = test_connection(client_stream, keys.clone(), 0);
        let peer = std::thread::spawn(move || {
            let mut server = server_stream;
            // Drip one application-data record back one byte at a time.
            let rec = seal_record(&keys, 0, RECORD_APP_DATA, b"drip").expect("seal record");
            for b in rec {
                // The client stops reading at the deadline and the test
                // drops the connection below: later writes fail and end
                // the drip instead of running it to completion.
                if server.write_all(&[b]).is_err() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });
        conn.set_read_deadline(Some(Instant::now() + BUDGET));
        let start = Instant::now();
        let mut buf = [0u8; 64];
        let r = conn.read(&mut buf);
        let elapsed = start.elapsed();
        assert!(
            r.is_err(),
            "dripping TLS peer must not stretch a read past the deadline"
        );
        assert_eq!(
            r.unwrap_err().kind(),
            io::ErrorKind::TimedOut,
            "read-deadline expiry must surface as a timeout"
        );
        assert!(
            elapsed < 3 * BUDGET,
            "dripping TLS peer stretched the read to {elapsed:?} against a {BUDGET:?} budget",
        );
        drop(conn);
        peer.join().unwrap();
    }

    #[test]
    fn drop_sends_close_notify() {
        // m11: dropping a live connection sends an encrypted close_notify
        // (idempotent with close()).
        let (client_stream, mut server_stream) = socketpair();
        let keys = test_keys();
        let conn = test_connection(client_stream, keys.clone(), 0);
        drop(conn);
        let mut hdr = [0u8; 5];
        server_stream.read_exact(&mut hdr).unwrap();
        assert_eq!(hdr[0], RECORD_APP_DATA);
        let len = u16::from_be_bytes([hdr[3], hdr[4]]) as usize;
        let mut ct = vec![0u8; len];
        server_stream.read_exact(&mut ct).unwrap();
        let mut rec = hdr.to_vec();
        rec.extend_from_slice(&ct);
        let (inner, pt) = open_record(&keys, 0, &mut rec).unwrap();
        assert_eq!(inner, RECORD_ALERT);
        assert_eq!(pt, [1u8, 0u8]);
    }

    // ---------- failure alert encryption (m9) ----------

    fn parse_client_session_id(ch_rec: &[u8]) -> Vec<u8> {
        // ch_rec: record header + handshake message.
        let msg = &ch_rec[5..];
        assert_eq!(msg[0], HS_CLIENT_HELLO);
        let sid_len = msg[38] as usize;
        msg[39..39 + sid_len].to_vec()
    }

    /// Extract the client's X25519 key share from the ClientHello record.
    /// `run_handshake` uses fresh random secrets, so the fake server must
    /// read the client's actual ephemeral public key off the wire.
    fn parse_client_key_share(ch_rec: &[u8]) -> [u8; 32] {
        let msg = &ch_rec[5..];
        assert_eq!(msg[0], HS_CLIENT_HELLO);
        let sid_len = msg[38] as usize;
        let mut off = 39 + sid_len;
        let suites_len = u16::from_be_bytes([msg[off], msg[off + 1]]) as usize;
        off += 2 + suites_len;
        let comp_len = msg[off] as usize;
        off += 1 + comp_len;
        let exts_len = u16::from_be_bytes([msg[off], msg[off + 1]]) as usize;
        off += 2;
        let exts_end = off + exts_len;
        while off < exts_end {
            let id = u16::from_be_bytes([msg[off], msg[off + 1]]);
            let len = u16::from_be_bytes([msg[off + 2], msg[off + 3]]) as usize;
            off += 4;
            if id == EXT_KEY_SHARE {
                // client_shares: 2-byte length, then (group, ke) entries.
                let mut koff = off + 2;
                let shares_end = off + len;
                while koff < shares_end {
                    let group = u16::from_be_bytes([msg[koff], msg[koff + 1]]);
                    let kelen = u16::from_be_bytes([msg[koff + 2], msg[koff + 3]]) as usize;
                    koff += 4;
                    if group == GROUP_X25519 {
                        assert_eq!(kelen, 32, "bad X25519 key share length");
                        let mut ke = [0u8; 32];
                        ke.copy_from_slice(&msg[koff..koff + 32]);
                        return ke;
                    }
                    koff += kelen;
                }
                panic!("no X25519 key share in ClientHello");
            }
            off += len;
        }
        panic!("no key_share extension in ClientHello");
    }

    #[test]
    fn failure_alert_is_encrypted_once_handshake_keys_exist() {
        // m9: a handshake failure after client_hs exists (tampered
        // Finished) is answered with an encrypted alert, not plaintext.
        let (client_stream, mut server_stream) = socketpair();
        let config = test_config();
        let handle = std::thread::spawn(move || {
            let mut c = client_stream;
            TlsConnection::run_handshake(&mut c, &config)
        });

        // Act as the server: read ClientHello, answer with ServerHello +
        // a flight whose Finished is tampered.
        let ch_rec = read_one_record(&mut server_stream).unwrap();
        let sid = parse_client_session_id(&ch_rec);

        let server_secret = [0x44; 32];
        let server_pub = x25519_public(&server_secret);
        let client_pub = parse_client_key_share(&ch_rec);
        let shared = x25519::shared_secret(&server_secret, &client_pub);
        let sh = server_hello_msg(
            &sid,
            TLS_AES_128_GCM_SHA256,
            GROUP_X25519,
            &server_pub,
            None,
        );
        write_record(
            &mut server_stream,
            RECORD_HANDSHAKE,
            LEGACY_RECORD_VERSION,
            &sh,
        )
        .unwrap();

        let hash = HashAlg::Sha256;
        let spec = cipher_spec(TLS_AES_128_GCM_SHA256).unwrap();
        let transcript_ch_sh = [&ch_rec[5..], &sh[..]].concat();
        let thash_ch_sh = transcript_hash(hash, &transcript_ch_sh);
        let (_, s_hs_traffic, c_hs_traffic) = server_schedule(hash, &shared, &thash_ch_sh);
        let s_keys = traffic_keys(hash, &s_hs_traffic, &spec).unwrap();
        let ee = encrypted_extensions_msg(None);
        let thash_ee = transcript_hash(hash, &[transcript_ch_sh, ee.clone()].concat());
        let mut fin = server_finished_msg(hash, &s_hs_traffic, &thash_ee);
        let last = fin.len() - 1;
        fin[last] ^= 0x01;
        server_stream
            .write_all(&seal_record(&s_keys, 0, RECORD_HANDSHAKE, &ee).unwrap())
            .unwrap();
        server_stream
            .write_all(&seal_record(&s_keys, 1, RECORD_HANDSHAKE, &fin).unwrap())
            .unwrap();

        // The client's alert must be an encrypted record (type 23).
        let mut arec = read_one_record(&mut server_stream).unwrap();
        assert_eq!(
            arec[0], RECORD_APP_DATA,
            "failure alert after handshake keys must be encrypted"
        );
        let c_keys = traffic_keys(hash, &c_hs_traffic, &spec).unwrap();
        let (inner, pt) = open_record(&c_keys, 0, &mut arec).unwrap();
        assert_eq!(inner, RECORD_ALERT);
        assert_eq!(pt, [2u8, ALERT_DECODE_ERROR]);

        let r = handle.join().unwrap();
        assert!(matches!(r, Err(TlsError::BadHandshake(_))));
    }

    #[test]
    fn failure_alert_after_sealed_finished_uses_next_nonce() {
        // Critical net issue #2: the client Finished is sealed under
        // (client_hs keys, nonce iv^client_seq). If the Finished's
        // write_all fails (or a later step errors), the failure alert
        // must be sealed under the *next* nonce. Reusing one (key,
        // nonce) pair in AES-GCM leaks the keystream and breaks GHASH.
        let (mut client_stream, mut server_stream) = socketpair();
        let mut hs = test_handshake();
        hs.client_hs = Some(test_keys());

        // Exactly what run_handshake_inner does: seal, then write.
        let mut enc = hs.seal_client_finished(b"finished plaintext").unwrap();
        assert_eq!(
            hs.client_seq, 1,
            "sealing the Finished must reserve the next nonce"
        );
        let (inner, pt) = open_record(&test_keys(), 0, &mut enc).unwrap();
        assert_eq!(inner, RECORD_HANDSHAKE);
        assert_eq!(pt, b"finished plaintext");

        // The write failed; the alert must be encrypted under the next
        // nonce, never the Finished's.
        TlsConnection::send_failure_alert(
            &mut client_stream,
            Some(&hs),
            &TlsError::Internal("no app keys"),
        );
        let mut arec = read_one_record(&mut server_stream).unwrap();
        assert_eq!(arec[0], RECORD_APP_DATA, "alert must be encrypted");
        let (ainner, apt) = open_record(&test_keys(), 1, &mut arec).unwrap();
        assert_eq!(ainner, RECORD_ALERT);
        assert_eq!(apt, [2u8, ALERT_INTERNAL_ERROR]);
        assert!(
            open_record(&test_keys(), 0, &mut arec).is_err(),
            "alert must not decrypt under the Finished's nonce"
        );
    }

    // ---------- benchmarks (ignored by default) ----------
    //
    // Run with:
    //   cargo test -p bowser_net --release --lib -- --ignored --nocapture \
    //       --test-threads=1 tls::tests::bench
    //
    // Measures the handshake hot paths named in the optimization task:
    // record processing (seal/open) and transcript hashing, plus peak
    // heap usage via a counting global allocator. The certificate path
    // (x509/RSA in crypto/) is out of scope and bypassed with the state
    // flags directly.

    #[cfg(test)]
    pub(crate) mod bench {
        use super::*;
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;
        use std::time::Instant;

        // Thread-local so parallel test threads cannot pollute each
        // other's measurements; the benches below run single-threaded,
        // so their reports are unaffected.
        thread_local! {
            static ALLOC_BYTES: Cell<u64> = const { Cell::new(0) };
            static PEAK_BYTES: Cell<u64> = const { Cell::new(0) };
            static ALLOC_COUNT: Cell<u64> = const { Cell::new(0) };
        }

        struct CountingAlloc;

        // SAFETY: forwards to System; only adds thread-local accounting.
        unsafe impl GlobalAlloc for CountingAlloc {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                // SAFETY: layout comes from the caller, System handles it.
                let p = unsafe { System.alloc(layout) };
                if !p.is_null() {
                    let size = layout.size() as u64;
                    ALLOC_BYTES.with(|b| {
                        let cur = b.get() + size;
                        b.set(cur);
                        PEAK_BYTES.with(|p| p.set(p.get().max(cur)));
                    });
                    ALLOC_COUNT.with(|c| c.set(c.get() + 1));
                }
                p
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                // SAFETY: ptr/layout pair came from alloc above.
                unsafe { System.dealloc(ptr, layout) };
                ALLOC_BYTES.with(|b| b.set(b.get().saturating_sub(layout.size() as u64)));
            }
        }

        #[global_allocator]
        static COUNTING: CountingAlloc = CountingAlloc;

        pub(crate) fn reset_counters() {
            ALLOC_BYTES.with(|b| b.set(0));
            PEAK_BYTES.with(|p| p.set(0));
            ALLOC_COUNT.with(|c| c.set(0));
        }

        /// Peak live heap bytes on the calling thread since the last
        /// [`reset_counters`].
        pub(crate) fn peak_live_bytes() -> u64 {
            PEAK_BYTES.with(|p| p.get())
        }

        fn report(name: &str, iters: u64, dt: std::time::Duration) {
            println!(
                "BENCH {name}: {iters} iters, {:.3} ms total, {:.2} us/iter, peak {} bytes, {} allocs",
                dt.as_secs_f64() * 1000.0,
                dt.as_secs_f64() * 1e6 / iters as f64,
                PEAK_BYTES.with(|p| p.get()),
                ALLOC_COUNT.with(|c| c.get()),
            );
        }

        /// Full client handshake state machine: ClientHello build,
        /// ServerHello parse, key schedule, transcript hashing,
        /// EncryptedExtensions, and Finished verify + client Finished.
        #[test]
        #[ignore]
        fn bench_handshake() {
            let iters = 200u64;
            let server_secret = [0x44u8; 32];
            let server_pub = x25519_public(&server_secret);
            let client_pub = x25519_public(&[0x33u8; 32]);
            let shared = x25519::shared_secret(&server_secret, &client_pub);
            let hash = HashAlg::Sha256;
            reset_counters();
            let t0 = Instant::now();
            for _ in 0..iters {
                let mut hs = Handshake::new_with_secrets(
                    &test_config(),
                    [0x11; 32],
                    [0x33; 32],
                    vec![0x55; 32],
                )
                .unwrap();
                let _ch = hs.client_hello().unwrap();
                let sh = server_hello_msg(
                    &[0x55; 32],
                    TLS_AES_128_GCM_SHA256,
                    GROUP_X25519,
                    &server_pub,
                    None,
                );
                assert!(matches!(
                    hs.handle_server_hello(&sh).unwrap(),
                    HelloAction::Proceed
                ));
                let thash_ch_sh = hs.transcript.digest(hash);
                let (_, s_hs_traffic, _) = server_schedule(hash, &shared, &thash_ch_sh);
                hs.handle_encrypted_extensions(&encrypted_extensions_msg(None))
                    .unwrap();
                // Bypass the x509/RSA certificate path (out of scope).
                hs.saw_cert = true;
                hs.saw_cv = true;
                hs.cert_verified = true;
                let thash_ee = hs.transcript.digest(hash);
                let fin = server_finished_msg(hash, &s_hs_traffic, &thash_ee);
                let _cf = hs.handle_finished(&fin).unwrap();
                std::hint::black_box(&hs);
            }
            report("bench_handshake", iters, t0.elapsed());
        }

        /// Record-layer seal + open round trip on full-size (16 KiB) records.
        #[test]
        #[ignore]
        fn bench_record_16k() {
            let iters = 2000u64;
            let keys = test_keys();
            let pt = vec![0xABu8; MAX_PLAINTEXT_LEN];
            reset_counters();
            let t0 = Instant::now();
            let mut sink = 0u64;
            for seq in 0..iters {
                let mut rec = seal_record(&keys, seq, RECORD_APP_DATA, &pt).unwrap();
                let (inner, out) = open_record(&keys, seq, &mut rec).unwrap();
                sink += inner as u64 + out.len() as u64;
            }
            let dt = t0.elapsed();
            std::hint::black_box(sink);
            report("bench_record_16k", iters, dt);
        }
    }
}
