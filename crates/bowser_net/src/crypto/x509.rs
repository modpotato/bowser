//! X.509 certificates: DER parsing, RFC 5280 path validation, RFC 6125
//! hostname verification, and the vendored Mozilla root store.
//!
//! This module is what makes the TLS client authenticate its peer.
//! [`verify_server_chain`] takes the DER certificates from the TLS
//! Certificate message, builds a path to a trust anchor in
//! [`root_store`], and checks every link: name chaining, validity
//! periods, basic constraints, key usage, extended key usage, name
//! constraints, and signatures (RSA PKCS#1 v1.5 via
//! [`crate::crypto::rsa`], ECDSA via [`crate::crypto::ec`]).
//! [`verify_hostname`] then checks the leaf against the expected
//! hostname per RFC 6125.
//!
//! ## Root store
//!
//! The trust anchors are vendored as *data* (not code) in
//! `tests/fixtures/ca_bundle.bin`, loaded with `include_bytes!` exactly
//! like `psl.rs` loads `psl_data.bin`. The bundle is the curl project's
//! conversion of Mozilla's CA Certificate store
//! (<https://curl.se/ca/cacert.pem>, "Certificate data from Mozilla as
//! of: Fri Sep 25 03:12:01 2026 GMT"), regenerated with
//! `tools/gen_ca_bundle.py`. Mozilla's CA data is offered under the
//! Mozilla Public License 2.0
//! (<https://www.mozilla.org/en-US/MPL/>); see
//! <https://www.mozilla.org/en-US/about/governance/policies/security-group/certs/>.
//! Layout: a sequence of records, each `u32` big-endian DER length
//! followed by the DER bytes.
//!
//! ## Scope notes (fail closed outside these)
//!
//! * Signature algorithms: RSA PKCS#1 v1.5 with SHA-256/384/512 and
//!   ECDSA with P-256/SHA-256, P-384/SHA-384. RSASSA-PSS, EdDSA, DSA
//!   and everything else is rejected.
//! * Name chaining compares the DER encoding of the Name verbatim.
//! * Policy constraints, CRLs and OCSP are not implemented. Revocation
//!   checking is future work.
//! * Hostname verification converts internationalized names to A-label
//!   (punycode) form via `bowser_url`'s IDNA path before matching, so
//!   IDN sites verify against their A-label dNSName SANs.
//!
//! ## Root-store update story
//!
//! There is no automated one: the bundle is a build-time snapshot baked
//! in with `include_bytes!`, and it goes stale as Mozilla adds and
//! removes roots. In particular, a root Mozilla distrusts stays trusted
//! here until the bundle is regenerated with `tools/gen_ca_bundle.py`
//! and Bowser is rebuilt. The snapshot date quoted above is the
//! freshness signal; regenerating on a schedule (or fetching Mozilla's
//! bundle at first run into a local cache) is future work.

use core::fmt;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use super::ec::{self, Curve, EcdsaHash};
use super::rsa::{self, RsaHash};

/// What went wrong during certificate parsing or path validation.
/// Carries only static detail strings; the caller turns these into the
/// owned reason for [`crate::tls::TlsError::CertificateVerificationFailed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VerifyError {
    /// The certificate DER could not be parsed.
    Malformed(&'static str),
    /// No path could be built to a trusted root, or a link is broken.
    Untrusted(&'static str),
    /// A certificate is outside its validity period (or time is unusable).
    TimeInvalid(&'static str),
    /// A signature in the path did not verify.
    BadSignature(&'static str),
    /// The leaf name did not match the expected hostname.
    HostnameMismatch(&'static str),
    /// An issuer is not a CA or violates a constraint.
    NotACa(&'static str),
    /// Unsupported algorithm or key type (fails closed).
    Unsupported(&'static str),
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, detail) = match self {
            VerifyError::Malformed(s) => ("malformed certificate", *s),
            VerifyError::Untrusted(s) => ("untrusted chain", *s),
            VerifyError::TimeInvalid(s) => ("certificate time invalid", *s),
            VerifyError::BadSignature(s) => ("bad signature", *s),
            VerifyError::HostnameMismatch(s) => ("hostname mismatch", *s),
            VerifyError::NotACa(s) => ("issuer is not a CA", *s),
            VerifyError::Unsupported(s) => ("unsupported algorithm", *s),
        };
        write!(f, "{kind}: {detail}")
    }
}

impl std::error::Error for VerifyError {}

// ---------------------------------------------------------------------------
// Minimal DER reader
// ---------------------------------------------------------------------------

/// Bounds and structural errors share one message family; every path
/// fails closed on truncated or nonsensical input.
const E_TRUNC: &str = "truncated DER";
const E_FORM: &str = "bad DER form";

struct Der<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Der<'a> {
    fn new(b: &'a [u8]) -> Self {
        Der { b, pos: 0 }
    }

    fn rest(&self) -> usize {
        self.b.len() - self.pos
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.b.len()
    }

    fn peek_tag(&self) -> Result<u8, VerifyError> {
        self.b
            .get(self.pos)
            .copied()
            .ok_or(VerifyError::Malformed(E_TRUNC))
    }

    /// Read one TLV; returns `(tag, contents)`.
    fn read_tlv(&mut self) -> Result<(u8, &'a [u8]), VerifyError> {
        let tag = self.peek_tag()?;
        let bad = || VerifyError::Malformed(E_FORM);
        if self.rest() < 2 {
            return Err(VerifyError::Malformed(E_TRUNC));
        }
        let mut pos = self.pos + 1;
        let first = self.b[pos];
        pos += 1;
        let len = if first & 0x80 == 0 {
            first as usize
        } else {
            // Long form; DER forbids indefinite lengths.
            let n = (first & 0x7f) as usize;
            if n == 0 || n > 4 || self.b.len() - pos < n {
                return Err(bad());
            }
            let mut len = 0usize;
            for i in 0..n {
                len = (len << 8) | self.b[pos + i] as usize;
            }
            pos += n;
            // DER requires the shortest possible length encoding.
            if len < 128 {
                return Err(bad());
            }
            len
        };
        if self.b.len() - pos < len {
            return Err(VerifyError::Malformed(E_TRUNC));
        }
        self.pos = pos + len;
        Ok((tag, &self.b[pos..pos + len]))
    }

    /// Expect a SEQUENCE and return a cursor over its contents.
    fn read_seq(&mut self, ctx: &'static str) -> Result<Der<'a>, VerifyError> {
        let (tag, v) = self.read_tlv()?;
        if tag != 0x30 {
            return Err(VerifyError::Malformed(ctx));
        }
        Ok(Der::new(v))
    }

    /// Expect tag `t` and return the contents.
    fn read_tagged(&mut self, t: u8, ctx: &'static str) -> Result<&'a [u8], VerifyError> {
        let (tag, v) = self.read_tlv()?;
        if tag != t {
            return Err(VerifyError::Malformed(ctx));
        }
        Ok(v)
    }

    /// INTEGER as minimal big-endian bytes. Rejects negative values and
    /// non-minimal encodings (DER 8.3).
    fn read_integer(&mut self) -> Result<&'a [u8], VerifyError> {
        let v = self.read_tagged(0x02, "bad INTEGER")?;
        if v.is_empty() {
            return Err(VerifyError::Malformed("empty INTEGER"));
        }
        if v[0] & 0x80 != 0 {
            return Err(VerifyError::Malformed("negative INTEGER"));
        }
        let mut v = v;
        // A leading zero byte is only legal when the next byte needs it.
        if v.len() > 1 && v[0] == 0 && v[1] & 0x80 == 0 {
            return Err(VerifyError::Malformed("non-minimal INTEGER"));
        }
        if v.len() > 1 && v[0] == 0 {
            v = &v[1..];
        }
        Ok(v)
    }

    fn read_oid(&mut self) -> Result<Vec<u64>, VerifyError> {
        let v = self.read_tagged(0x06, "bad OID")?;
        if v.is_empty() {
            return Err(VerifyError::Malformed("empty OID"));
        }
        // The first subidentifier packs the first two arcs as
        // 40*arc0 + arc1 in a single base-128 value (possibly multi-byte
        // when arc0 == 2 and arc1 is large).
        let mut i = 0;
        let mut first: u64 = 0;
        loop {
            if i >= v.len() {
                return Err(VerifyError::Malformed("truncated OID"));
            }
            let b = v[i];
            i += 1;
            first = first
                .checked_shl(7)
                .and_then(|a| a.checked_add((b & 0x7f) as u64))
                .ok_or(VerifyError::Malformed("OID component overflow"))?;
            if b & 0x80 == 0 {
                break;
            }
        }
        let (arc0, arc1) = if first < 40 {
            (0, first)
        } else if first < 80 {
            (1, first - 40)
        } else {
            (2, first - 80)
        };
        let mut out = vec![arc0, arc1];
        while i < v.len() {
            let mut acc = 0u64;
            loop {
                if i >= v.len() {
                    return Err(VerifyError::Malformed("truncated OID"));
                }
                let b = v[i];
                i += 1;
                acc = acc
                    .checked_shl(7)
                    .and_then(|a| a.checked_add((b & 0x7f) as u64))
                    .ok_or(VerifyError::Malformed("OID component overflow"))?;
                if b & 0x80 == 0 {
                    break;
                }
            }
            out.push(acc);
        }
        Ok(out)
    }

    /// BIT STRING; requires the unused-bits count to be zero and returns
    /// the payload bytes.
    fn read_bit_string(&mut self) -> Result<&'a [u8], VerifyError> {
        let v = self.read_tagged(0x03, "bad BIT STRING")?;
        if v.is_empty() || v[0] != 0 {
            return Err(VerifyError::Malformed("bad BIT STRING padding"));
        }
        Ok(&v[1..])
    }

    /// `[n]` EXPLICIT contents.
    fn read_explicit(&mut self, n: u8, ctx: &'static str) -> Result<Der<'a>, VerifyError> {
        let (tag, v) = self.read_tlv()?;
        if tag != (0xa0 | n) {
            return Err(VerifyError::Malformed(ctx));
        }
        Ok(Der::new(v))
    }

    fn read_bool(&mut self) -> Result<bool, VerifyError> {
        let v = self.read_tagged(0x01, "bad BOOLEAN")?;
        if v.len() != 1 {
            return Err(VerifyError::Malformed("bad BOOLEAN length"));
        }
        // DER canonicality: TRUE is exactly 0xFF.
        match v[0] {
            0x00 => Ok(false),
            0xFF => Ok(true),
            _ => Err(VerifyError::Malformed("non-canonical BOOLEAN")),
        }
    }
}

// ---------------------------------------------------------------------------
// Algorithm identifiers
// ---------------------------------------------------------------------------

/// Signature algorithms this module verifies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SigAlg {
    RsaPkcs1(RsaHash),
    Ecdsa(EcdsaHash),
    /// A recognized signature algorithm this module cannot verify
    /// (SHA-1 RSA, RSASSA-PSS, Ed25519, ECDSA-SHA512). Trust anchors
    /// may carry these since their self-signature is never checked,
    /// but any attempt to verify a signature with `Other` fails.
    Other,
}

fn parse_sig_alg(d: &mut Der<'_>) -> Result<SigAlg, VerifyError> {
    let mut seq = d.read_seq("bad AlgorithmIdentifier")?;
    let oid = seq.read_oid()?;
    // Parameters (NULL or absent) are ignored; the OID selects everything.
    match oid.as_slice() {
        [1, 2, 840, 113549, 1, 1, 11] => Ok(SigAlg::RsaPkcs1(RsaHash::Sha256)),
        [1, 2, 840, 113549, 1, 1, 12] => Ok(SigAlg::RsaPkcs1(RsaHash::Sha384)),
        [1, 2, 840, 113549, 1, 1, 13] => Ok(SigAlg::RsaPkcs1(RsaHash::Sha512)),
        [1, 2, 840, 10045, 4, 3, 2] => Ok(SigAlg::Ecdsa(EcdsaHash::Sha256)),
        [1, 2, 840, 10045, 4, 3, 3] => Ok(SigAlg::Ecdsa(EcdsaHash::Sha384)),
        // Recognized but unverifiable: trust anchors may carry these
        // (their self-signature is never checked); verification with
        // them always fails closed.
        [1, 2, 840, 113549, 1, 1, 5] => Ok(SigAlg::Other), // sha1WithRSAEncryption
        [1, 2, 840, 113549, 1, 1, 10] => Ok(SigAlg::Other), // RSASSA-PSS
        [1, 2, 840, 10045, 4, 3, 4] => Ok(SigAlg::Other),  // ecdsa-with-SHA512
        [1, 3, 101, 112] => Ok(SigAlg::Other),             // Ed25519
        _ => Err(VerifyError::Unsupported("unknown signature OID")),
    }
}

/// A parsed subject public key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Spki {
    Rsa {
        n: Vec<u8>,
        e: Vec<u8>,
    },
    Ec {
        curve: Curve,
        x: Vec<u8>,
        y: Vec<u8>,
    },
    /// A public key of a type this module cannot use (unknown key OID,
    /// unknown named curve, explicit EC parameters). Trust anchors may
    /// carry these; any verification attempt with `Other` fails closed.
    Other,
}

fn parse_spki(d: &mut Der<'_>) -> Result<Spki, VerifyError> {
    let mut seq = d.read_seq("bad SubjectPublicKeyInfo")?;
    let mut alg = seq.read_seq("bad SPKI algorithm")?;
    let oid = alg.read_oid()?;
    let key = seq.read_bit_string()?;
    if !seq.is_empty() {
        return Err(VerifyError::Malformed("trailing SPKI bytes"));
    }
    match oid.as_slice() {
        [1, 2, 840, 113549, 1, 1, 1] => {
            let mut kd = Der::new(key);
            let mut ks = kd.read_seq("bad RSAPublicKey")?;
            let n = ks.read_integer()?.to_vec();
            let e = ks.read_integer()?.to_vec();
            if !ks.is_empty() || !kd.is_empty() {
                return Err(VerifyError::Malformed("trailing RSAPublicKey bytes"));
            }
            if n.is_empty() || e.is_empty() {
                return Err(VerifyError::Malformed("empty RSA key component"));
            }
            // Same cap as verification: a tens-of-KB modulus would turn
            // signature checks into a minutes-long CPU burn.
            if n.len() * 8 > rsa::MAX_RSA_MODULUS_BITS {
                return Err(VerifyError::Malformed("RSA modulus too large"));
            }
            Ok(Spki::Rsa { n, e })
        }
        [1, 2, 840, 10045, 2, 1] => {
            // EC parameters are normally a named-curve OID; anything
            // else (explicit parameters, unknown curve) parses as
            // Other and fails closed if ever used for verification.
            let curve = alg
                .read_oid()
                .ok()
                .and_then(|curve_oid| match curve_oid.as_slice() {
                    [1, 2, 840, 10045, 3, 1, 7] => Some(Curve::P256),
                    [1, 3, 132, 0, 34] => Some(Curve::P384),
                    _ => None,
                });
            let Some(curve) = curve else {
                return Ok(Spki::Other);
            };
            let flen = match curve {
                Curve::P256 => 32,
                Curve::P384 => 48,
            };
            if key.len() != 1 + 2 * flen || key[0] != 0x04 {
                return Err(VerifyError::Malformed("bad EC uncompressed point"));
            }
            Ok(Spki::Ec {
                curve,
                x: key[1..1 + flen].to_vec(),
                y: key[1 + flen..].to_vec(),
            })
        }
        // Unknown key type: parse as Other so trust anchors carrying it
        // still load; verification with it always fails closed.
        _ => Ok(Spki::Other),
    }
}

// ---------------------------------------------------------------------------
// Parsed certificate
// ---------------------------------------------------------------------------

/// A name (dNSName) or IP address from the subjectAltName extension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum San {
    Dns(String),
    Ip(Vec<u8>), // 4 or 16 bytes
}

/// One subtree from a nameConstraints extension (RFC 5280 §4.2.1.10).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NameConstraint {
    /// dNSName base, lowercased. Matches the base name itself and any
    /// name built by adding labels on the left (RFC 5280 §4.2.1.10).
    Dns(String),
    /// iPAddress base: network address and mask, 4 or 16 bytes each.
    Ip { addr: Vec<u8>, mask: Vec<u8> },
    /// A GeneralName form this module cannot evaluate (anything other
    /// than dNSName/iPAddress), or a subtree with a non-default
    /// minimum/maximum. A CA carrying one fails closed: no path
    /// through it validates.
    Unsupported,
}

/// Parsed nameConstraints extension (OID 2.5.29.30).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct NameConstraints {
    pub(crate) permitted: Vec<NameConstraint>,
    pub(crate) excluded: Vec<NameConstraint>,
}

/// Parse the value (OCTET STRING contents) of a nameConstraints
/// extension.
///
/// ```text
/// NameConstraints ::= SEQUENCE {
///     permittedSubtrees [0] GeneralSubtrees OPTIONAL,
///     excludedSubtrees  [1] GeneralSubtrees OPTIONAL }
/// GeneralSubtree ::= SEQUENCE {
///     base GeneralName, minimum [0] BaseDistance DEFAULT 0,
///     maximum [1] BaseDistance OPTIONAL }
/// ```
/// The `[0]`/`[1]` tags are IMPLICIT, so they wrap the SEQUENCE OF
/// GeneralSubtree bodies directly. At least one of the two lists must
/// be present (RFC 5280 §4.2.1.10).
fn parse_name_constraints(value: &[u8]) -> Result<NameConstraints, VerifyError> {
    let bad = |m: &'static str| VerifyError::Malformed(m);
    let mut d = Der::new(value);
    let mut seq = d.read_seq("bad nameConstraints")?;
    if !d.is_empty() {
        return Err(bad("trailing nameConstraints bytes"));
    }
    let mut nc = NameConstraints::default();
    let mut seen = false;
    while !seq.is_empty() {
        let (tag, val) = seq.read_tlv()?;
        let permitted = match tag {
            0xa0 => true,  // [0] permittedSubtrees
            0xa1 => false, // [1] excludedSubtrees
            _ => return Err(bad("bad nameConstraints tag")),
        };
        seen = true;
        let mut subtrees = Der::new(val);
        let mut count = 0;
        while !subtrees.is_empty() {
            count += 1;
            let mut st = subtrees.read_seq("bad GeneralSubtree")?;
            let (base_tag, base_val) = st.read_tlv()?;
            let mut constraint = match base_tag {
                // dNSName [2] IMPLICIT IA5String.
                0x82 => {
                    let s = core::str::from_utf8(base_val)
                        .map_err(|_| bad("bad dNSName name constraint"))?;
                    // A leading dot is openssl's "subdomains only"
                    // spelling; RFC 5280 semantics include the base
                    // name itself, so normalize it away.
                    let s = s.strip_prefix('.').unwrap_or(s).to_ascii_lowercase();
                    NameConstraint::Dns(s)
                }
                // iPAddress [7] IMPLICIT OCTET STRING: the address
                // followed by the mask.
                0x87 => {
                    if base_val.len() != 8 && base_val.len() != 32 {
                        return Err(bad("bad iPAddress name constraint"));
                    }
                    let half = base_val.len() / 2;
                    NameConstraint::Ip {
                        addr: base_val[..half].to_vec(),
                        mask: base_val[half..].to_vec(),
                    }
                }
                // Any other GeneralName form cannot be evaluated.
                _ => NameConstraint::Unsupported,
            };
            // minimum [0] / maximum [1]: only the defaults (absent
            // minimum, no maximum) are supported; anything else makes
            // the subtree unevaluable, which fails closed.
            while !st.is_empty() {
                let (t, mm) = st.read_tlv()?;
                let mut md = Der::new(mm);
                let n = md.read_integer().map_err(|_| bad("bad min/max"))?;
                if !md.is_empty() {
                    return Err(bad("bad min/max"));
                }
                match t {
                    0x80 if n == [0u8].as_slice() => {} // minimum 0: default
                    0x80 | 0x81 => constraint = NameConstraint::Unsupported,
                    _ => return Err(bad("bad GeneralSubtree")),
                }
            }
            if permitted {
                nc.permitted.push(constraint);
            } else {
                nc.excluded.push(constraint);
            }
        }
        if count == 0 {
            return Err(bad("empty GeneralSubtrees"));
        }
    }
    if !seen {
        return Err(bad("empty nameConstraints"));
    }
    Ok(nc)
}

/// An X.509 certificate with the fields path validation needs, all owned
/// (no lifetimes into the original DER).
#[derive(Debug, Clone)]
pub(crate) struct ParsedCert {
    /// Full DER of the certificate, for byte-identity comparison
    /// against trust anchors (a directly-presented anchor).
    pub(crate) der: Vec<u8>,
    /// Exact DER of tbsCertificate (the signature input).
    pub(crate) tbs_der: Vec<u8>,
    /// Raw signatureValue bytes.
    pub(crate) signature: Vec<u8>,
    pub(crate) sig_alg: SigAlg,
    /// DER of the issuer and subject Names (compared verbatim).
    pub(crate) issuer_der: Vec<u8>,
    pub(crate) subject_der: Vec<u8>,
    pub(crate) spki: Spki,
    pub(crate) not_before: i64,
    pub(crate) not_after: i64,
    pub(crate) is_ca: bool,
    pub(crate) path_len: Option<u32>,
    /// keyUsage bits as a u16 (bit i = 1 << i); None when absent.
    pub(crate) key_usage: Option<u16>,
    /// extendedKeyUsage OIDs; None when the extension is absent (which
    /// permits any use).
    pub(crate) eku: Option<Vec<Vec<u64>>>,
    /// nameConstraints; None when the extension is absent.
    pub(crate) name_constraints: Option<NameConstraints>,
    pub(crate) sans: Vec<San>,
    /// First CN, for the RFC 6125 fallback when the certificate carries
    /// no SANs at all.
    pub(crate) common_name: Option<String>,
}

/// Cap on a single certificate's DER size (DoS bound).
const MAX_CERT_LEN: usize = 65536;

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    // Howard Hinnant's algorithm.
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Parse UTCTime (`YYMMDDHHMMSSZ`) or GeneralizedTime
/// (`YYYYMMDDHHMMSSZ`) to unix seconds. Only the `Z` form is accepted.
fn parse_time(tag: u8, v: &[u8]) -> Result<i64, VerifyError> {
    let bad = || VerifyError::Malformed("bad Time");
    let s = core::str::from_utf8(v).map_err(|_| bad())?;
    let (ys, rest) = match tag {
        0x17 => {
            if s.len() != 13 {
                return Err(bad());
            }
            let yy: i64 = s[0..2].parse().map_err(|_| bad())?;
            ((if yy >= 50 { 1900 + yy } else { 2000 + yy }), &s[2..])
        }
        0x18 => {
            if s.len() != 15 {
                return Err(bad());
            }
            (s[0..4].parse::<i64>().map_err(|_| bad())?, &s[4..])
        }
        _ => return Err(bad()),
    };
    if !rest.ends_with('Z') || rest.len() != 11 {
        return Err(bad());
    }
    let num = |r: core::ops::Range<usize>| rest[r].parse::<i64>().map_err(|_| bad());
    let (mo, d, h, mi, se) = (num(0..2)?, num(2..4)?, num(4..6)?, num(6..8)?, num(8..10)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || se > 60 {
        return Err(bad());
    }
    Ok(days_from_civil(ys, mo, d) * 86400 + h * 3600 + mi * 60 + se)
}

/// Extract the raw DER slice of the next TLV (tag through value end).
fn tlv_slice<'a>(d: &mut Der<'a>) -> Result<&'a [u8], VerifyError> {
    let start = d.pos;
    d.read_tlv()?;
    Ok(&d.b[start..d.pos])
}

/// First CN value in a Name, as a String. Accepts the string types CAs
/// actually use; anything else yields None (not an error).
fn first_common_name(name_der: &[u8]) -> Option<String> {
    let mut name = Der::new(name_der);
    let mut rdn_seq = name.read_seq("").ok()?;
    while !rdn_seq.is_empty() {
        // RDN is a SET OF AttributeTypeAndValue.
        let (rdn_tag, rdn_val) = rdn_seq.read_tlv().ok()?;
        if rdn_tag != 0x31 {
            return None;
        }
        let mut rdn = Der::new(rdn_val);
        while !rdn.is_empty() {
            let mut atv = rdn.read_seq("").ok()?;
            let oid = atv.read_oid().ok()?;
            let (tag, val) = atv.read_tlv().ok()?;
            if oid.as_slice() == [2, 5, 4, 3] {
                let s = match tag {
                    0x0c | 0x13 | 0x16 => core::str::from_utf8(val).ok()?.to_string(),
                    0x1e => {
                        // BMPString: UTF-16BE, ASCII subset only.
                        if val.len() % 2 != 0 {
                            continue;
                        }
                        let mut out = String::new();
                        for ch in val.as_chunks::<2>().0 {
                            let c = u16::from_be_bytes([ch[0], ch[1]]);
                            if c > 0x7f {
                                continue;
                            }
                            out.push(c as u8 as char);
                        }
                        out
                    }
                    _ => continue,
                };
                if !s.is_empty() {
                    return Some(s);
                }
            }
        }
    }
    None
}

/// Parse the extensions block `[3] EXPLICIT SEQUENCE OF Extension`.
fn parse_extensions(d: &mut Der<'_>, cert: &mut ParsedCert) -> Result<(), VerifyError> {
    if d.is_empty() {
        return Ok(());
    }
    let mut block = d.read_explicit(3, "bad extensions block")?;
    let mut seq = block.read_seq("bad extensions")?;
    if !block.is_empty() {
        return Err(VerifyError::Malformed("trailing extensions block bytes"));
    }
    let mut seen_oids: Vec<Vec<u64>> = Vec::new();
    while !seq.is_empty() {
        let mut ext = seq.read_seq("bad extension")?;
        let oid = ext.read_oid()?;
        // RFC 5280 §4.2: each extension OID must appear at most once.
        // A duplicate is malformed — fail closed rather than last-win,
        // which could let an attacker override a restrictive value.
        if seen_oids.contains(&oid) {
            return Err(VerifyError::Malformed("duplicate extension"));
        }
        seen_oids.push(oid.clone());
        let critical = if ext.peek_tag()? == 0x01 {
            ext.read_bool()?
        } else {
            false
        };
        let value = ext.read_tagged(0x04, "bad extension value")?;
        if !ext.is_empty() {
            return Err(VerifyError::Malformed("trailing extension bytes"));
        }
        let mut v = Der::new(value);
        match oid.as_slice() {
            // subjectAltName
            [2, 5, 29, 17] => {
                let mut names = v.read_seq("bad subjectAltName")?;
                if !v.is_empty() {
                    return Err(VerifyError::Malformed("trailing SAN bytes"));
                }
                while !names.is_empty() {
                    let (tag, val) = names.read_tlv()?;
                    match tag {
                        0x82 => {
                            // dNSName IA5String
                            let s = core::str::from_utf8(val)
                                .map_err(|_| VerifyError::Malformed("bad dNSName"))?;
                            cert.sans.push(San::Dns(s.to_string()));
                        }
                        0x87 => {
                            // iPAddress OCTET STRING
                            if val.len() != 4 && val.len() != 16 {
                                return Err(VerifyError::Malformed("bad iPAddress"));
                            }
                            cert.sans.push(San::Ip(val.to_vec()));
                        }
                        _ => {} // other GeneralName forms are ignored
                    }
                }
            }
            // basicConstraints
            [2, 5, 29, 19] => {
                let mut bc = v.read_seq("bad basicConstraints")?;
                if !v.is_empty() {
                    return Err(VerifyError::Malformed("trailing basicConstraints bytes"));
                }
                if !bc.is_empty() && bc.peek_tag()? == 0x01 {
                    cert.is_ca = bc.read_bool()?;
                }
                if !bc.is_empty() {
                    let pl = bc.read_integer()?;
                    if pl.len() > 4 {
                        return Err(VerifyError::Malformed("bad pathLenConstraint"));
                    }
                    let mut n = 0u32;
                    for &b in pl {
                        n = (n << 8) | b as u32;
                    }
                    cert.path_len = Some(n);
                }
                if !bc.is_empty() {
                    return Err(VerifyError::Malformed("trailing basicConstraints bytes"));
                }
            }
            // keyUsage: a BIT STRING whose unused-bits count is usually
            // non-zero, so it gets its own reader (the strict
            // read_bit_string is only for key/signature BIT STRINGs,
            // where RFC 5280 requires zero unused bits).
            [2, 5, 29, 15] => {
                let raw = v.read_tagged(0x03, "bad keyUsage")?;
                if raw.is_empty() || raw[0] > 7 {
                    return Err(VerifyError::Malformed("bad keyUsage"));
                }
                let unused = raw[0] as usize;
                let bits = &raw[1..];
                if bits.is_empty() && unused != 0 {
                    return Err(VerifyError::Malformed("bad keyUsage"));
                }
                let significant = bits.len() * 8 - unused;
                // Bits are numbered from the MSB of the first byte
                // (RFC 5280 §4.2.1.3); only the first 16 are tracked.
                let mut ku = 0u16;
                for bit in 0..significant.min(16) {
                    if bits[bit / 8] & (0x80 >> (bit % 8)) != 0 {
                        ku |= 1 << bit;
                    }
                }
                cert.key_usage = Some(ku);
            }
            // extendedKeyUsage (2.5.29.37): a SEQUENCE of purpose OIDs.
            // Absent means any use; when present, path validation
            // requires serverAuth or anyExtendedKeyUsage (fail closed).
            [2, 5, 29, 37] => {
                let mut d = Der::new(value);
                let mut purposes = d.read_seq("bad extendedKeyUsage")?;
                if !d.is_empty() {
                    return Err(VerifyError::Malformed("trailing extendedKeyUsage bytes"));
                }
                let mut oids = Vec::new();
                while !purposes.is_empty() {
                    oids.push(purposes.read_oid()?);
                }
                if oids.is_empty() {
                    return Err(VerifyError::Malformed("empty extendedKeyUsage"));
                }
                cert.eku = Some(oids);
            }
            // nameConstraints (2.5.29.30).
            [2, 5, 29, 30] => {
                cert.name_constraints = Some(parse_name_constraints(value)?);
            }
            _ => {
                // Unknown extensions are ignored, except unknown *critical*
                // ones, which fail closed per RFC 5280 §4.2.
                if critical {
                    return Err(VerifyError::Unsupported("unknown critical extension"));
                }
            }
        }
    }
    Ok(())
}

/// Parse one DER certificate.
pub(crate) fn parse_cert(der: &[u8]) -> Result<ParsedCert, VerifyError> {
    if der.len() > MAX_CERT_LEN {
        return Err(VerifyError::Malformed("certificate too large"));
    }
    let mut d = Der::new(der);
    let mut cert_seq = d.read_seq("bad Certificate")?;
    if !d.is_empty() {
        return Err(VerifyError::Malformed("trailing Certificate bytes"));
    }
    let tbs_der = tlv_slice(&mut cert_seq)?.to_vec();
    let mut tbs = Der::new(&tbs_der);
    let mut tbs_seq = tbs.read_seq("bad tbsCertificate")?;

    // [0] EXPLICIT version (optional, DEFAULT v1). v1/v2 certificates
    // carry no extensions, so is_ca stays false and they can never act
    // as a CA in path validation; only the version number itself is
    // validated here.
    if tbs_seq.peek_tag()? == 0xa0 {
        let mut v = tbs_seq.read_explicit(0, "bad version")?;
        let vv = v.read_integer()?;
        if vv != [0] && vv != [1] && vv != [2] {
            return Err(VerifyError::Malformed("bad certificate version"));
        }
        if !v.is_empty() {
            return Err(VerifyError::Malformed("trailing version bytes"));
        }
    }
    let _serial = tbs_seq.read_integer()?;
    let sig_alg_inner = parse_sig_alg(&mut tbs_seq)?;
    let issuer_der = tlv_slice(&mut tbs_seq)?.to_vec();
    // Validity.
    let mut validity = tbs_seq.read_seq("bad validity")?;
    let (nb_tag, nb_val) = validity.read_tlv()?;
    let (na_tag, na_val) = validity.read_tlv()?;
    if !validity.is_empty() {
        return Err(VerifyError::Malformed("trailing validity bytes"));
    }
    let not_before = parse_time(nb_tag, nb_val)?;
    let not_after = parse_time(na_tag, na_val)?;
    if not_after < not_before {
        return Err(VerifyError::Malformed("notAfter before notBefore"));
    }
    let subject_der = tlv_slice(&mut tbs_seq)?.to_vec();
    let spki = parse_spki(&mut tbs_seq)?;
    // subjectUniqueID / issuerUniqueID and extensions: skip the unique IDs,
    // parse extensions if present.
    while !tbs_seq.is_empty() && tbs_seq.peek_tag()? != 0xa3 {
        let (tag, _) = tbs_seq.read_tlv()?;
        if tag != 0x81 && tag != 0x82 {
            return Err(VerifyError::Malformed("unexpected tbs field"));
        }
    }
    let mut cert = ParsedCert {
        der: der.to_vec(),
        tbs_der: tbs_der.clone(),
        signature: Vec::new(),
        sig_alg: sig_alg_inner,
        issuer_der,
        subject_der: subject_der.clone(),
        spki,
        not_before,
        not_after,
        is_ca: false,
        path_len: None,
        key_usage: None,
        eku: None,
        name_constraints: None,
        sans: Vec::new(),
        common_name: first_common_name(&subject_der),
    };
    parse_extensions(&mut tbs_seq, &mut cert)?;
    if !tbs_seq.is_empty() {
        return Err(VerifyError::Malformed("trailing tbsCertificate bytes"));
    }

    // Outer signatureAlgorithm should match the inner one; a mismatch is
    // a malformed certificate (RFC 5280 §4.1.1.2).
    let outer_alg = parse_sig_alg(&mut cert_seq)?;
    if outer_alg != sig_alg_inner {
        return Err(VerifyError::Malformed("signatureAlgorithm mismatch"));
    }
    cert.signature = cert_seq.read_bit_string()?.to_vec();
    if cert.signature.is_empty() {
        return Err(VerifyError::Malformed("empty signatureValue"));
    }
    if !cert_seq.is_empty() {
        return Err(VerifyError::Malformed("trailing Certificate bytes"));
    }
    Ok(cert)
}

// ---------------------------------------------------------------------------
// Signature verification over a path link
// ---------------------------------------------------------------------------

/// Parse a DER ECDSA signature (SEQUENCE of two INTEGERs) into (r, s).
fn parse_ecdsa_sig(sig: &[u8]) -> Result<(Vec<u8>, Vec<u8>), VerifyError> {
    let mut d = Der::new(sig);
    let mut seq = d.read_seq("bad ECDSA signature")?;
    if !d.is_empty() {
        return Err(VerifyError::Malformed("trailing ECDSA signature bytes"));
    }
    let r = seq.read_integer()?.to_vec();
    let s = seq.read_integer()?.to_vec();
    if !seq.is_empty() {
        return Err(VerifyError::Malformed("trailing ECDSA signature bytes"));
    }
    Ok((r, s))
}

/// Verify `child`'s signature over its tbsCertificate with the issuer's key.
fn verify_link(child: &ParsedCert, issuer_spki: &Spki) -> Result<(), VerifyError> {
    let bad = || VerifyError::BadSignature("signature did not verify");
    let ok = match (&child.sig_alg, issuer_spki) {
        (SigAlg::RsaPkcs1(h), Spki::Rsa { n, e }) => {
            rsa::verify_pkcs1_v15(n, e, &child.tbs_der, &child.signature, *h).is_ok()
        }
        (SigAlg::Ecdsa(h), Spki::Ec { curve, x, y }) => {
            let (r, s) = parse_ecdsa_sig(&child.signature)?;
            ec::verify_ecdsa(*curve, x, y, &child.tbs_der, &r, &s, *h).is_ok()
        }
        // Unverifiable algorithm: fail closed, never accept.
        (SigAlg::Other, _) => {
            return Err(VerifyError::Unsupported(
                "certificate signature algorithm not supported",
            ));
        }
        // Unusable issuer key: fail closed, never accept.
        (_, Spki::Other) => {
            return Err(VerifyError::Unsupported(
                "issuer public key type not supported",
            ));
        }
        _ => return Err(VerifyError::BadSignature("sig alg / issuer key mismatch")),
    };
    if ok { Ok(()) } else { Err(bad()) }
}

/// keyUsage bit positions (RFC 5280 §4.2.1.3).
const KU_DIGITAL_SIGNATURE: u16 = 1 << 0;
const KU_KEY_CERT_SIGN: u16 = 1 << 5;

/// Whether `cert` may be used for TLS server authentication.
///
/// An absent extendedKeyUsage permits any use; when present, the
/// certificate must list serverAuth (1.3.6.1.5.5.7.3.1) or
/// anyExtendedKeyUsage (2.5.29.37.0), else the path fails closed
/// (RFC 5280 §4.2.1.12).
fn eku_permits_server_auth(cert: &ParsedCert) -> bool {
    match &cert.eku {
        None => true,
        Some(oids) => oids.iter().any(|o| {
            matches!(
                o.as_slice(),
                [1, 3, 6, 1, 5, 5, 7, 3, 1] | [2, 5, 29, 37, 0]
            )
        }),
    }
}

/// A dNSName `name` (lowercase) is inside the constraint subtree
/// `base` when it equals the base or is built by adding labels on the
/// left (RFC 5280 §4.2.1.10).
fn dns_in_subtree(name: &str, base: &str) -> bool {
    if base.is_empty() {
        return false;
    }
    name == base
        || name
            .strip_suffix(base)
            .is_some_and(|prefix| prefix.ends_with('.'))
}

/// An IP address is inside an iPAddress constraint when it matches the
/// network address under the mask. v4 and v6 never match each other.
fn ip_in_range(ip: &[u8], addr: &[u8], mask: &[u8]) -> bool {
    if ip.len() != addr.len() || addr.len() != mask.len() {
        return false;
    }
    ip.iter()
        .zip(addr.iter())
        .zip(mask.iter())
        .all(|((&b, &a), &m)| b & m == a & m)
}

/// Name-constraint state accumulated down a path (RFC 5280 §6.1.5):
/// the union of excluded subtrees, plus each constraining CA's
/// permitted set (a name must satisfy *every* CA's permitted set for
/// the name forms that set constrains).
#[derive(Debug, Default)]
struct NcState {
    permitted: Vec<Vec<NameConstraint>>,
    excluded: Vec<NameConstraint>,
}

impl NcState {
    /// Fold in one CA's nameConstraints. Any unevaluable subtree
    /// fails the whole path closed.
    fn apply(&mut self, nc: &Option<NameConstraints>) -> Result<(), VerifyError> {
        let Some(nc) = nc else { return Ok(()) };
        if nc
            .permitted
            .iter()
            .chain(nc.excluded.iter())
            .any(|c| matches!(c, NameConstraint::Unsupported))
        {
            return Err(VerifyError::Untrusted("unsupported name constraint"));
        }
        if !nc.permitted.is_empty() {
            self.permitted.push(nc.permitted.clone());
        }
        self.excluded.extend(nc.excluded.iter().cloned());
        Ok(())
    }

    /// Check one certificate's names against the accumulated state.
    /// dNSName constraints apply to dNSName SANs and the subject CN;
    /// iPAddress constraints to iPAddress SANs (RFC 5280 §6.1.5).
    fn check_cert(&self, cert: &ParsedCert) -> Result<(), VerifyError> {
        let mut dns_names: Vec<&str> = Vec::new();
        for san in &cert.sans {
            if let San::Dns(d) = san {
                dns_names.push(d);
            }
        }
        if let Some(cn) = &cert.common_name {
            dns_names.push(cn);
        }
        for name in dns_names {
            let name = name.to_ascii_lowercase();
            for set in &self.permitted {
                let constrains_dns = set.iter().any(|c| matches!(c, NameConstraint::Dns(_)));
                let allowed = set.iter().any(|c| match c {
                    NameConstraint::Dns(base) => dns_in_subtree(&name, base),
                    _ => false,
                });
                if constrains_dns && !allowed {
                    return Err(VerifyError::Untrusted("name outside permitted subtrees"));
                }
            }
            let excluded = self.excluded.iter().any(|c| match c {
                NameConstraint::Dns(base) => dns_in_subtree(&name, base),
                _ => false,
            });
            if excluded {
                return Err(VerifyError::Untrusted("name inside excluded subtree"));
            }
        }
        for san in &cert.sans {
            let San::Ip(ip) = san else { continue };
            for set in &self.permitted {
                let constrains_ip = set.iter().any(|c| matches!(c, NameConstraint::Ip { .. }));
                let allowed = set.iter().any(|c| match c {
                    NameConstraint::Ip { addr, mask } => ip_in_range(ip, addr, mask),
                    _ => false,
                });
                if constrains_ip && !allowed {
                    return Err(VerifyError::Untrusted("name outside permitted subtrees"));
                }
            }
            let excluded = self.excluded.iter().any(|c| match c {
                NameConstraint::Ip { addr, mask } => ip_in_range(ip, addr, mask),
                _ => false,
            });
            if excluded {
                return Err(VerifyError::Untrusted("name inside excluded subtree"));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RFC 5280 path validation
// ---------------------------------------------------------------------------

/// Longest chain we will attempt to build (leaf + intermediates).
const MAX_CHAIN_LEN: usize = 8;

fn unix_now(now: SystemTime) -> Result<i64, VerifyError> {
    now.duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .map_err(|_| VerifyError::TimeInvalid("system clock before unix epoch"))
}

/// Validate a server certificate chain.
///
/// `chain_der` is the TLS Certificate message's list, leaf first.
/// `roots` are the trust anchors (see [`root_store`]). Returns the
/// validated leaf certificate on success.
pub(crate) fn verify_server_chain(
    chain_der: &[&[u8]],
    hostname: &str,
    now: SystemTime,
    roots: &[ParsedCert],
) -> Result<ParsedCert, VerifyError> {
    if chain_der.is_empty() {
        return Err(VerifyError::Untrusted("server sent no certificates"));
    }
    if chain_der.len() > MAX_CHAIN_LEN {
        return Err(VerifyError::Untrusted("chain too long"));
    }
    let now = unix_now(now)?;

    let mut certs = Vec::with_capacity(chain_der.len());
    for der in chain_der {
        certs.push(parse_cert(der)?);
    }

    // The leaf's name is checked first so hostname errors are reported
    // even when the chain is also broken.
    verify_hostname(&certs[0], hostname)?;

    // Name chaining between presented certificates.
    for pair in certs.windows(2) {
        if pair[0].issuer_der != pair[1].subject_der {
            return Err(VerifyError::Untrusted("issuer/subject name mismatch"));
        }
    }

    // Trust anchor: a root whose subject matches the last issuer.
    // Several roots may share a subject (cross-signing); try each in
    // turn and accept the first that validates the whole path.
    let last = &certs[certs.len() - 1];
    let candidates: Vec<&ParsedCert> = roots
        .iter()
        .filter(|r| r.subject_der == last.issuer_der)
        .collect();
    if candidates.is_empty() {
        return Err(VerifyError::Untrusted("no trusted root for issuer"));
    }
    let mut first_err = VerifyError::Untrusted("no trusted root for issuer");
    for root in candidates {
        match validate_with_root(&certs, root, now) {
            Ok(leaf) => return Ok(leaf),
            Err(e) => first_err = e,
        }
    }
    Err(first_err)
}

/// Validate `certs` (leaf first) against one trust anchor.
fn validate_with_root(
    certs: &[ParsedCert],
    root: &ParsedCert,
    now: i64,
) -> Result<ParsedCert, VerifyError> {
    if !root.is_ca {
        return Err(VerifyError::NotACa("trust anchor is not a CA"));
    }
    // The trust anchor is trusted by configuration, but an EKU on it
    // still limits what it may be used for: a root restricted to
    // clientAuth cannot anchor TLS server authentication.
    if !eku_permits_server_auth(root) {
        return Err(VerifyError::Untrusted(
            "trust anchor EKU forbids server authentication",
        ));
    }

    // A directly-presented trust anchor: the peer sent a single
    // certificate byte-identical to a configured anchor. Accept it as
    // the anchor itself instead of validating it as a leaf (which
    // would reject it for asserting CA:TRUE) — but still honor the
    // anchor's own validity window (RFC 5280 §6.1.3), the same check
    // the chained path applies to the root below.
    if certs.len() == 1 && certs[0].der == root.der {
        if root.not_before > now || now > root.not_after {
            return Err(VerifyError::TimeInvalid("root not valid at this time"));
        }
        return Ok(certs[0].clone());
    }

    // Name constraints accumulate from the trust anchor down: each
    // CA's subtrees apply to every certificate below it (RFC 5280
    // §6.1.5).
    let mut nc = NcState::default();
    nc.apply(&root.name_constraints)?;

    // Validate from the trust anchor down to the leaf.
    let mut issuer_spki = &root.spki;
    for (i, cert) in certs.iter().enumerate().rev() {
        verify_link(cert, issuer_spki)?;
        if cert.not_before > now || now > cert.not_after {
            return Err(VerifyError::TimeInvalid(
                "certificate not valid at this time",
            ));
        }
        if !eku_permits_server_auth(cert) {
            return Err(VerifyError::Untrusted(
                "extended key usage forbids server authentication",
            ));
        }
        nc.check_cert(cert)?;
        if i > 0 {
            // CA checks for every intermediate.
            if !cert.is_ca {
                return Err(VerifyError::NotACa("intermediate is not a CA"));
            }
            if let Some(ku) = cert.key_usage
                && ku & KU_KEY_CERT_SIGN == 0
            {
                return Err(VerifyError::NotACa("intermediate lacks keyCertSign"));
            }
            if let Some(max) = cert.path_len {
                // RFC 5280 §4.2.1.9: at most `max` CA certificates may
                // appear below this one (toward the leaf, i.e. at lower
                // indices in this leaf-first chain).
                let ca_below = certs[..i].iter().filter(|c| c.is_ca).count();
                if ca_below as u32 > max {
                    return Err(VerifyError::NotACa("pathLenConstraint violated"));
                }
            }
            // This CA's name constraints govern the certificates it
            // issued (checked on later iterations, closer to the leaf).
            nc.apply(&cert.name_constraints)?;
            issuer_spki = &cert.spki;
        } else {
            // End-entity checks (i == 0): a CA certificate must never
            // validate as a leaf (RFC 5280 §4.2.1.9), and a leaf that
            // carries keyUsage must assert digitalSignature for TLS
            // server authentication (RFC 5280 §4.2.1.3).
            if cert.is_ca {
                return Err(VerifyError::Untrusted("leaf certificate asserts CA:TRUE"));
            }
            if let Some(ku) = cert.key_usage
                && ku & KU_DIGITAL_SIGNATURE == 0
            {
                return Err(VerifyError::Untrusted(
                    "leaf lacks digitalSignature key usage",
                ));
            }
        }
    }
    // The root's own validity is checked too (RFC 5280 §6.1.3).
    if root.not_before > now || now > root.not_after {
        return Err(VerifyError::TimeInvalid("root not valid at this time"));
    }

    Ok(certs[0].clone())
}

// ---------------------------------------------------------------------------
// RFC 6125 hostname verification
// ---------------------------------------------------------------------------

/// Check `hostname` against the leaf certificate's identity.
///
/// IP literals match only against iPAddress SANs. DNS names match
/// dNSName SANs when the certificate carries any SANs at all, and fall
/// back to the subject CN only when it carries none (RFC 6125 §6.4.4,
/// legacy behavior kept for compatibility).
pub(crate) fn verify_hostname(cert: &ParsedCert, hostname: &str) -> Result<(), VerifyError> {
    // A single trailing dot is insignificant in DNS; strip it before
    // anything else (RFC 6125 §6.4.1).
    let host = hostname.strip_suffix('.').unwrap_or(hostname);
    if host.is_empty() {
        return Err(VerifyError::HostnameMismatch("empty hostname"));
    }
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        return verify_ip(cert, ip);
    }
    // Internationalized names are converted to A-label (punycode) form
    // through the same IDNA path the URL parser uses, then matched
    // against the certificate's dNSName SANs — CAs issue those as
    // A-labels, so comparing in A-label form is the correct
    // normalization (RFC 6125 §6.4, UTS #46). A name that does not
    // convert fails closed.
    let host = bowser_url::domain_to_ascii(host)
        .ok_or(VerifyError::HostnameMismatch("invalid hostname"))?;

    let dns_sans: Vec<&str> = cert
        .sans
        .iter()
        .filter_map(|s| match s {
            San::Dns(d) => Some(d.as_str()),
            _ => None,
        })
        .collect();
    if !cert.sans.is_empty() {
        // SANs exist: dNSName entries are authoritative. The CN
        // fallback does NOT apply when SANs are present but none is a
        // dNSName (RFC 6125 §6.4.4) — e.g. an iPAddress-only SAN never
        // falls back to the CN for a DNS hostname.
        if dns_sans.iter().any(|p| dns_match(p, &host)) {
            return Ok(());
        }
        return Err(VerifyError::HostnameMismatch("no dNSName SAN matched"));
    }
    match &cert.common_name {
        Some(cn) if dns_match(cn, &host) => Ok(()),
        Some(_) => Err(VerifyError::HostnameMismatch("CN did not match")),
        None => Err(VerifyError::HostnameMismatch("no SAN or CN to match")),
    }
}

fn verify_ip(cert: &ParsedCert, ip: std::net::IpAddr) -> Result<(), VerifyError> {
    for san in &cert.sans {
        if let San::Ip(bytes) = san {
            let san_ip = match bytes.len() {
                4 => std::net::IpAddr::from(
                    <[u8; 4]>::try_from(bytes.as_slice()).expect("len checked"),
                ),
                16 => std::net::IpAddr::from(
                    <[u8; 16]>::try_from(bytes.as_slice()).expect("len checked"),
                ),
                _ => continue,
            };
            if san_ip == ip {
                return Ok(());
            }
        }
    }
    Err(VerifyError::HostnameMismatch("no iPAddress SAN matched"))
}

/// Match a dNSName/CN pattern against a lowercase ASCII hostname.
fn dns_match(pattern: &str, host: &str) -> bool {
    if pattern.is_empty() || pattern.as_bytes().contains(&0) {
        return false;
    }
    // Belt-and-braces: a host starting with '.' is never valid, and
    // without this `*.example.com` would match `.example.com` (the
    // single-label check passes with left = ".").
    if host.starts_with('.') {
        return false;
    }
    let pattern = pattern.to_ascii_lowercase();
    if let Some(suffix) = pattern.strip_prefix("*.") {
        // Wildcard: only as the entire leftmost label (RFC 6125 §6.4.3).
        if suffix.is_empty() || suffix.contains('*') {
            return false;
        }
        let Some(left) = host.strip_suffix(suffix) else {
            return false;
        };
        // `left` must be exactly one non-empty label plus the dot.
        !left.is_empty() && left.ends_with('.') && !left[..left.len() - 1].contains('.')
    } else {
        if pattern.contains('*') {
            return false;
        }
        pattern == host
    }
}

// ---------------------------------------------------------------------------
// Root store
// ---------------------------------------------------------------------------

/// Vendored Mozilla root bundle; see the module docs for the format.
const CA_BUNDLE: &[u8] = include_bytes!("../../tests/fixtures/ca_bundle.bin");

/// The trust anchors: parsed once, shared by all handshakes.
pub(crate) fn root_store() -> &'static [ParsedCert] {
    static ROOTS: OnceLock<Vec<ParsedCert>> = OnceLock::new();
    ROOTS.get_or_init(|| {
        let mut roots = Vec::new();
        let mut pos = 0;
        while pos + 4 <= CA_BUNDLE.len() {
            let len = u32::from_be_bytes(CA_BUNDLE[pos..pos + 4].try_into().expect("len checked"))
                as usize;
            pos += 4;
            let end = pos.saturating_add(len);
            if end > CA_BUNDLE.len() {
                break;
            }
            // A single unparseable bundled cert must not take down the
            // whole store; it is skipped (and covered by a test asserting
            // the skip count is zero for the vendored bundle).
            if let Ok(cert) = parse_cert(&CA_BUNDLE[pos..end]) {
                roots.push(cert);
            }
            pos = end;
        }
        roots
    })
}

// ---------------------------------------------------------------------------
// TLS signature schemes (for CertificateVerify)
// ---------------------------------------------------------------------------

/// Verify `signature` over `message` with `spki` under the given TLS
/// signature scheme id (RFC 8446 §4.2.3).
pub(crate) fn verify_tls_scheme(
    spki: &Spki,
    scheme: u16,
    message: &[u8],
    signature: &[u8],
) -> Result<(), VerifyError> {
    let bad = || VerifyError::BadSignature("signature did not verify");
    let ok = match scheme {
        0x0401 => match spki {
            Spki::Rsa { n, e } => {
                rsa::verify_pkcs1_v15(n, e, message, signature, RsaHash::Sha256).is_ok()
            }
            _ => return Err(VerifyError::BadSignature("scheme/key type mismatch")),
        },
        0x0501 => match spki {
            Spki::Rsa { n, e } => {
                rsa::verify_pkcs1_v15(n, e, message, signature, RsaHash::Sha384).is_ok()
            }
            _ => return Err(VerifyError::BadSignature("scheme/key type mismatch")),
        },
        0x0601 => match spki {
            Spki::Rsa { n, e } => {
                rsa::verify_pkcs1_v15(n, e, message, signature, RsaHash::Sha512).is_ok()
            }
            _ => return Err(VerifyError::BadSignature("scheme/key type mismatch")),
        },
        0x0403 => match spki {
            Spki::Ec {
                curve: Curve::P256,
                x,
                y,
            } => {
                let (r, s) = parse_ecdsa_sig(signature)?;
                ec::verify_ecdsa(Curve::P256, x, y, message, &r, &s, EcdsaHash::Sha256).is_ok()
            }
            _ => return Err(VerifyError::BadSignature("scheme/key type mismatch")),
        },
        0x0503 => match spki {
            Spki::Ec {
                curve: Curve::P384,
                x,
                y,
            } => {
                let (r, s) = parse_ecdsa_sig(signature)?;
                ec::verify_ecdsa(Curve::P384, x, y, message, &r, &s, EcdsaHash::Sha384).is_ok()
            }
            _ => return Err(VerifyError::BadSignature("scheme/key type mismatch")),
        },
        _ => return Err(VerifyError::Unsupported("signature scheme not supported")),
    };
    if ok { Ok(()) } else { Err(bad()) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tls/");

    fn der(name: &str) -> Vec<u8> {
        std::fs::read(format!("{FIX}{name}")).expect("fixture missing")
    }

    /// The fixed "now" the fixtures were built for: 2026-06-01T00:00:00Z.
    const NOW: i64 = 1_780_358_400;

    fn now() -> SystemTime {
        UNIX_EPOCH + std::time::Duration::from_secs(NOW as u64)
    }

    fn test_roots() -> Vec<ParsedCert> {
        for name in ["test-ca.der", "ec-ca.der"] {
            let c = parse_cert(&der(name)).expect("fixture CA must parse");
            assert!(c.is_ca, "{name} must be a CA");
        }
        vec![
            parse_cert(&der("test-ca.der")).unwrap(),
            parse_cert(&der("ec-ca.der")).unwrap(),
        ]
    }

    #[test]
    fn root_bundle_parses_fully() {
        let roots = root_store();
        // The vendored bundle must parse completely; a skip means the
        // bundle data is corrupt, so fail loudly instead of shrinking
        // the trust store silently.
        assert!(
            roots.len() >= 100,
            "expected >=100 roots, got {}",
            roots.len()
        );
        let mut records = 0usize;
        let mut pos = 0;
        while pos + 4 <= CA_BUNDLE.len() {
            let len = u32::from_be_bytes(CA_BUNDLE[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4 + len;
            records += 1;
        }
        assert_eq!(roots.len(), records, "every bundle record must parse");
        assert!(roots.iter().all(|r| r.is_ca), "all roots must be CAs");
    }

    #[test]
    fn valid_rsa_chain_accepts() {
        let leaf = der("leaf-rsa.der");
        let inter = der("test-inter.der");
        let chain: Vec<&[u8]> = vec![&leaf, &inter];
        let leaf = verify_server_chain(&chain, "bowser.test", now(), &test_roots()).unwrap();
        assert!(matches!(leaf.spki, Spki::Rsa { .. }));
        assert_eq!(leaf.common_name.as_deref(), Some("bowser.test"));
    }

    #[test]
    fn valid_ec_chain_accepts() {
        let leaf = der("ec-leaf.der");
        let chain: Vec<&[u8]> = vec![&leaf];
        let leaf = verify_server_chain(&chain, "bowser.test", now(), &test_roots()).unwrap();
        assert!(matches!(leaf.spki, Spki::Ec { .. }));
    }

    #[test]
    fn expired_leaf_rejected() {
        let leaf = der("leaf-rsa-expired.der");
        let inter = der("test-inter.der");
        let chain: Vec<&[u8]> = vec![&leaf, &inter];
        let err = verify_server_chain(&chain, "bowser.test", now(), &test_roots()).unwrap_err();
        assert!(matches!(err, VerifyError::TimeInvalid(_)), "got {err}");
    }

    #[test]
    fn not_yet_valid_leaf_rejected() {
        // The fixture leaf is valid 2020-01-01..2040-01-01; check "now"
        // after notAfter too.
        let leaf = der("leaf-rsa.der");
        let inter = der("test-inter.der");
        let chain: Vec<&[u8]> = vec![&leaf, &inter];
        let future = UNIX_EPOCH + std::time::Duration::from_secs(2_300_000_000); // 2042-10
        let err = verify_server_chain(&chain, "bowser.test", future, &test_roots()).unwrap_err();
        assert!(matches!(err, VerifyError::TimeInvalid(_)), "got {err}");
    }

    #[test]
    fn wrong_host_rejected() {
        let leaf = der("leaf-rsa-wronghost.der");
        let inter = der("test-inter.der");
        let chain: Vec<&[u8]> = vec![&leaf, &inter];
        let err = verify_server_chain(&chain, "bowser.test", now(), &test_roots()).unwrap_err();
        assert!(matches!(err, VerifyError::HostnameMismatch(_)), "got {err}");
    }

    #[test]
    fn self_signed_leaf_rejected() {
        let leaf = der("leaf-selfsigned.der");
        let chain: Vec<&[u8]> = vec![&leaf];
        // The fixture's SAN is DNS:bowser.test, so the hostname matches;
        // rejection must come from path building, not name checking.
        let err = verify_server_chain(&chain, "bowser.test", now(), &test_roots()).unwrap_err();
        // The self-signed cert is not in the test roots, so path building
        // fails before signature checking matters.
        assert!(matches!(err, VerifyError::Untrusted(_)), "got {err}");
    }

    #[test]
    fn untrusted_chain_rejected() {
        // Signed by the evil CA, which is not in the trust store.
        let leaf = der("evil-leaf.der");
        let chain: Vec<&[u8]> = vec![&leaf];
        let err = verify_server_chain(&chain, "bowser.test", now(), &test_roots()).unwrap_err();
        assert!(matches!(err, VerifyError::Untrusted(_)), "got {err}");
    }

    #[test]
    fn intermediate_without_ca_flag_rejected() {
        let leaf = der("leaf-under-noca.der");
        let inter = der("inter-noca.der");
        let chain: Vec<&[u8]> = vec![&leaf, &inter];
        let err = verify_server_chain(&chain, "bowser.test", now(), &test_roots()).unwrap_err();
        assert!(matches!(err, VerifyError::NotACa(_)), "got {err}");
    }

    #[test]
    fn unsupported_algorithms_fail_closed() {
        // A recognized-but-unverifiable signature algorithm or key
        // type parses (so trust anchors carrying them load) but can
        // never validate a signature.
        let mut leaf = parse_cert(&der("leaf-rsa.der")).unwrap();
        let inter = parse_cert(&der("test-inter.der")).unwrap();

        leaf.sig_alg = SigAlg::Other;
        let err = verify_link(&leaf, &inter.spki).unwrap_err();
        assert!(matches!(err, VerifyError::Unsupported(_)), "got {err}");

        leaf.sig_alg = SigAlg::RsaPkcs1(RsaHash::Sha256);
        let mut inter_other = inter.clone();
        inter_other.spki = Spki::Other;
        let err = verify_link(&leaf, &inter_other.spki).unwrap_err();
        assert!(matches!(err, VerifyError::Unsupported(_)), "got {err}");
    }

    #[test]
    fn pathlen_constraint_violated() {
        // chain-pl-inter has pathlen:0 but a CA (chain-pl-sub) sits
        // below it; openssl agrees this chain is invalid.
        let leaf = der("chain-pl-leaf.der");
        let sub = der("chain-pl-sub.der");
        let inter = der("chain-pl-inter.der");
        let root = parse_cert(&der("chain-pl-root.der")).unwrap();
        let chain: Vec<&[u8]> = vec![&leaf, &sub, &inter];
        let err = verify_server_chain(&chain, "bowser.test", now(), &[root]).unwrap_err();
        assert!(matches!(err, VerifyError::NotACa(_)), "got {err}");
    }

    #[test]
    fn hostname_cases() {
        let leaf = parse_cert(&der("leaf-rsa.der")).unwrap();
        // SAN dNSName bowser.test
        verify_hostname(&leaf, "bowser.test").unwrap();
        verify_hostname(&leaf, "BOWSER.TEST").unwrap(); // case-insensitive
        verify_hostname(&leaf, "bowser.test.").unwrap(); // trailing dot
        assert!(matches!(
            verify_hostname(&leaf, "www.bowser.test"),
            Err(VerifyError::HostnameMismatch(_))
        ));
        assert!(matches!(
            verify_hostname(&leaf, "evilbowser.test"),
            Err(VerifyError::HostnameMismatch(_))
        ));

        let wild = parse_cert(&der("leaf-rsa-wildcard.der")).unwrap();
        verify_hostname(&wild, "www.bowser.test").unwrap();
        // Wildcards never cross label boundaries.
        assert!(matches!(
            verify_hostname(&wild, "a.b.bowser.test"),
            Err(VerifyError::HostnameMismatch(_))
        ));
        assert!(matches!(
            verify_hostname(&wild, "bowser.test"),
            Err(VerifyError::HostnameMismatch(_))
        ));

        let cnonly = parse_cert(&der("leaf-rsa-cnonly.der")).unwrap();
        // No SAN: falls back to CN.
        verify_hostname(&cnonly, "bowser.test").unwrap();
        assert!(matches!(
            verify_hostname(&cnonly, "other.test"),
            Err(VerifyError::HostnameMismatch(_))
        ));
    }

    #[test]
    fn idn_hostname_matches_a_label_san() {
        let mut leaf = parse_cert(&der("leaf-rsa.der")).unwrap();
        // Real IDN certificates carry the A-label (punycode) form in
        // their dNSName SANs; the U-label hostname must verify against
        // it after IDNA conversion.
        leaf.sans = vec![San::Dns("xn--bcher-kva.example".to_string())];
        verify_hostname(&leaf, "b\u{fc}cher.example").unwrap();
        verify_hostname(&leaf, "B\u{dc}CHER.EXAMPLE.").unwrap(); // case + trailing dot
        assert!(matches!(
            verify_hostname(&leaf, "other.example"),
            Err(VerifyError::HostnameMismatch(_))
        ));
        // A U-label SAN is not standard (CAs must issue A-labels) and
        // must not match: fail closed.
        leaf.sans = vec![San::Dns("b\u{fc}cher.example".to_string())];
        assert!(matches!(
            verify_hostname(&leaf, "b\u{fc}cher.example"),
            Err(VerifyError::HostnameMismatch(_))
        ));
    }

    #[test]
    fn ip_hostname_cases() {
        let leaf = parse_cert(&der("leaf-rsa-ip.der")).unwrap();
        verify_hostname(&leaf, "127.0.0.1").unwrap();
        assert!(matches!(
            verify_hostname(&leaf, "127.0.0.2"),
            Err(VerifyError::HostnameMismatch(_))
        ));
        // DNS names must not match an iPAddress-only SAN.
        assert!(matches!(
            verify_hostname(&leaf, "localhost"),
            Err(VerifyError::HostnameMismatch(_))
        ));
    }

    #[test]
    fn tampered_signature_rejected() {
        let mut leaf_der = der("leaf-rsa.der");
        // Flip a byte in the trailing signatureValue (last 16 bytes are
        // inside the signature for these fixtures).
        let last = leaf_der.len() - 1;
        leaf_der[last] ^= 0xff;
        let inter = der("test-inter.der");
        let chain: Vec<&[u8]> = vec![&leaf_der, &inter];
        let err = verify_server_chain(&chain, "bowser.test", now(), &test_roots()).unwrap_err();
        assert!(matches!(err, VerifyError::BadSignature(_)), "got {err}");
    }

    #[test]
    fn tampered_tbs_rejected() {
        let mut leaf_der = der("leaf-rsa.der");
        // Flip a byte inside the subject CN string: parsing still
        // succeeds, but the signature over the original tbs must fail.
        let cn = b"bowser.test";
        let pos = leaf_der
            .windows(cn.len())
            .position(|w| w == cn)
            .expect("fixture CN present");
        leaf_der[pos + 1] ^= 0x01;
        let inter = der("test-inter.der");
        let chain: Vec<&[u8]> = vec![&leaf_der, &inter];
        let err = verify_server_chain(&chain, "bowser.test", now(), &test_roots()).unwrap_err();
        assert!(
            matches!(
                err,
                VerifyError::BadSignature(_) | VerifyError::Malformed(_)
            ),
            "got {err}"
        );
    }

    #[test]
    fn empty_chain_rejected() {
        let err = verify_server_chain(&[], "bowser.test", now(), &test_roots()).unwrap_err();
        assert!(matches!(err, VerifyError::Untrusted(_)), "got {err}");
    }

    #[test]
    fn garbage_der_rejected() {
        let garbage = [0x30u8, 0x03, 0x01, 0x02, 0x03];
        let chain: Vec<&[u8]> = vec![&garbage];
        let err = verify_server_chain(&chain, "bowser.test", now(), &test_roots()).unwrap_err();
        assert!(matches!(err, VerifyError::Malformed(_)), "got {err}");
    }

    #[test]
    fn parse_time_vectors() {
        let t = |tag: u8, s: &str| parse_time(tag, s.as_bytes()).unwrap();
        // 2020-01-01T00:00:00Z
        assert_eq!(t(0x17, "200101000000Z"), 1_577_836_800);
        assert_eq!(t(0x18, "20200101000000Z"), 1_577_836_800);
        // UTCTime pivot: 50 -> 1950, 49 -> 2049.
        assert_eq!(t(0x17, "500101000000Z"), -631_152_000);
        assert_eq!(t(0x17, "490101000000Z"), 2_493_072_000);
        assert!(parse_time(0x17, "200101000000+0000".as_bytes()).is_err());
        assert!(parse_time(0x17, "200101000000".as_bytes()).is_err());
        assert!(parse_time(0x19, "200101000000Z".as_bytes()).is_err());
    }

    #[test]
    fn sig_alg_parsing() {
        let rsa_leaf = parse_cert(&der("leaf-rsa.der")).unwrap();
        assert_eq!(rsa_leaf.sig_alg, SigAlg::RsaPkcs1(RsaHash::Sha256));
        let ec_leaf = parse_cert(&der("ec-leaf.der")).unwrap();
        assert_eq!(ec_leaf.sig_alg, SigAlg::Ecdsa(EcdsaHash::Sha256));
        // RSA leaf signed by RSA intermediate; EC leaf by EC root.
        assert!(matches!(rsa_leaf.spki, Spki::Rsa { .. }));
        assert!(matches!(
            ec_leaf.spki,
            Spki::Ec {
                curve: Curve::P256,
                ..
            }
        ));
    }

    #[test]
    fn key_usage_parsed() {
        // openssl genrsa/x509: digitalSignature(0), keyEncipherment(2).
        let leaf = parse_cert(&der("leaf-rsa.der")).unwrap();
        assert_eq!(leaf.key_usage, Some(0b101));
        let inter = parse_cert(&der("test-inter.der")).unwrap();
        assert!(inter.is_ca);
        assert_eq!(inter.path_len, Some(0));
        assert!(inter.key_usage.unwrap() & KU_KEY_CERT_SIGN != 0);
    }

    #[test]
    fn tls_scheme_vectors() {
        // Cross-check verify_tls_scheme against the openssl KATs used for
        // rsa.rs/ec.rs: sign nothing new, just re-verify a signature from
        // the KAT files over the same message bytes.
        let kat = std::fs::read_to_string(format!("{FIX}rsa_kat.txt")).unwrap();
        let mut n = String::new();
        let mut e = String::new();
        let mut msg = String::new();
        let mut sig = String::new();
        let mut hash = String::new();
        for line in kat.lines() {
            if let Some(v) = line.strip_prefix("hash: ") {
                hash = v.to_string();
            } else if let Some(v) = line.strip_prefix("n: ") {
                n = v.to_string();
            } else if let Some(v) = line.strip_prefix("e: ") {
                e = v.to_string();
            } else if let Some(v) = line.strip_prefix("msg: ") {
                msg = v.to_string();
            } else if let Some(v) = line.strip_prefix("sig: ") {
                sig = v.to_string();
                break; // first vector only
            }
        }
        let dehex = |s: &str| {
            (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                .collect::<Vec<_>>()
        };
        let spki = Spki::Rsa {
            n: dehex(&n),
            e: dehex(&e),
        };
        let scheme = match hash.as_str() {
            "sha256" => 0x0401,
            "sha384" => 0x0501,
            "sha512" => 0x0601,
            _ => panic!("unexpected hash {hash}"),
        };
        verify_tls_scheme(&spki, scheme, &dehex(&msg), &dehex(&sig)).unwrap();
        assert!(verify_tls_scheme(&spki, 0x0804, &dehex(&msg), &dehex(&sig)).is_err());
    }

    #[test]
    fn eku_parsed() {
        // eku-ca carries EKU=clientAuth only (critical); the leaf has
        // EKU=serverAuth; ordinary CAs have no EKU at all.
        let ca = parse_cert(&der("eku-ca.der")).unwrap();
        assert_eq!(ca.eku, Some(vec![vec![1, 3, 6, 1, 5, 5, 7, 3, 2]]));
        assert!(!eku_permits_server_auth(&ca));
        let leaf = parse_cert(&der("eku-leaf.der")).unwrap();
        assert!(eku_permits_server_auth(&leaf));
        let inter = parse_cert(&der("test-inter.der")).unwrap();
        assert_eq!(inter.eku, None);
        assert!(eku_permits_server_auth(&inter));
    }

    #[test]
    fn eku_clientauth_only_chain_rejected() {
        // A chain to a root whose EKU is restricted to clientAuth (no
        // serverAuth) must fail closed for TLS server authentication,
        // even though signatures, names and dates are all valid.
        let ca = parse_cert(&der("eku-ca.der")).unwrap();
        let leaf = der("eku-leaf.der");
        let chain: Vec<&[u8]> = vec![&leaf];
        let err = verify_server_chain(&chain, "bowser.test", now(), &[ca]).unwrap_err();
        assert!(matches!(err, VerifyError::Untrusted(_)), "got {err}");
    }

    #[test]
    fn eku_serverauth_chain_accepted() {
        // The normal fixtures (leaf EKU=serverAuth, CAs without EKU)
        // keep validating; covered end-to-end by valid_rsa_chain_accepts.
        let leaf = parse_cert(&der("leaf-rsa.der")).unwrap();
        assert_eq!(leaf.eku, Some(vec![vec![1, 3, 6, 1, 5, 5, 7, 3, 1]]));
        assert!(eku_permits_server_auth(&leaf));
    }

    #[test]
    fn name_constraints_parsed() {
        let ca = parse_cert(&der("nc-ca.der")).unwrap();
        let nc = ca
            .name_constraints
            .expect("nc-ca must carry nameConstraints");
        assert_eq!(
            nc.permitted,
            vec![NameConstraint::Dns("example.com".into())]
        );
        assert!(nc.excluded.is_empty());
        let xc = parse_cert(&der("nc-exc-ca.der")).unwrap();
        let xnc = xc
            .name_constraints
            .expect("nc-exc-ca must carry nameConstraints");
        assert!(xnc.permitted.is_empty());
        assert_eq!(xnc.excluded, vec![NameConstraint::Dns("evil.com".into())]);
    }

    #[test]
    fn name_constraints_permitted_enforced() {
        // nc-ca permits only DNS:example.com.
        let ca = parse_cert(&der("nc-ca.der")).unwrap();
        let roots = [ca];
        let ok = der("nc-leaf-ok.der");
        let chain: Vec<&[u8]> = vec![&ok];
        verify_server_chain(&chain, "www.example.com", now(), &roots).unwrap();
        let bad = der("nc-leaf-bad.der");
        let chain: Vec<&[u8]> = vec![&bad];
        let err = verify_server_chain(&chain, "www.evil.com", now(), &roots).unwrap_err();
        assert!(matches!(err, VerifyError::Untrusted(_)), "got {err}");
    }

    #[test]
    fn name_constraints_excluded_enforced() {
        // nc-exc-ca excludes DNS:evil.com.
        let ca = parse_cert(&der("nc-exc-ca.der")).unwrap();
        let leaf = der("nc-exc-leaf.der");
        let chain: Vec<&[u8]> = vec![&leaf];
        let err = verify_server_chain(&chain, "www.evil.com", now(), &[ca]).unwrap_err();
        assert!(matches!(err, VerifyError::Untrusted(_)), "got {err}");
    }

    #[test]
    fn negative_pathlen_rejected() {
        // negpathlen.der is test-inter.der with the pathLenConstraint
        // INTEGER rewritten to -1 (0xFF). RFC 5280 §4.2.1.9 requires
        // pathLenConstraint to be non-negative; a negative value is
        // malformed, not a large positive count.
        let err = parse_cert(&der("negpathlen.der")).unwrap_err();
        assert!(matches!(err, VerifyError::Malformed(_)), "got {err}");
    }

    #[test]
    fn leaf_without_digital_signature_rejected() {
        // M2: keyUsage was previously enforced only on intermediates
        // (keyCertSign). A leaf carrying keyUsage without
        // digitalSignature must fail closed for server authentication.
        let mut leaf = parse_cert(&der("ec-leaf.der")).unwrap();
        assert!(leaf.key_usage.unwrap() & KU_DIGITAL_SIGNATURE != 0);
        leaf.key_usage = Some(0b100); // keyEncipherment only
        let root = parse_cert(&der("ec-ca.der")).unwrap();
        let err = validate_with_root(&[leaf], &root, NOW).unwrap_err();
        assert!(matches!(err, VerifyError::Untrusted(_)), "got {err}");
    }

    #[test]
    fn ca_true_leaf_rejected() {
        // M3: a certificate asserting CA:TRUE must never validate as an
        // end entity, even when the rest of the chain is valid.
        let mut leaf = parse_cert(&der("ec-leaf.der")).unwrap();
        leaf.is_ca = true;
        let root = parse_cert(&der("ec-ca.der")).unwrap();
        let err = validate_with_root(&[leaf], &root, NOW).unwrap_err();
        assert!(matches!(err, VerifyError::Untrusted(_)), "got {err}");
    }

    #[test]
    fn directly_presented_trust_anchor_accepted() {
        // A single presented certificate byte-identical to a trust
        // anchor is accepted as the anchor itself, not rejected as a
        // "leaf asserting CA:TRUE".
        let root = parse_cert(&der("ec-ca.der")).unwrap();
        let presented = parse_cert(&der("ec-ca.der")).unwrap();
        let validated = validate_with_root(&[presented], &root, NOW).unwrap();
        assert_eq!(validated.der, root.der);
        // A different CA certificate (not byte-identical to the anchor)
        // presented alone is still rejected as a leaf.
        let other_ca = parse_cert(&der("test-inter.der")).unwrap();
        let err = validate_with_root(&[other_ca], &root, NOW).unwrap_err();
        assert!(
            matches!(
                err,
                VerifyError::BadSignature(_) | VerifyError::Untrusted(_)
            ),
            "got {err}"
        );
    }

    #[test]
    fn expired_directly_presented_trust_anchor_rejected() {
        // An expired trust anchor must not validate by being presented
        // directly: the anchor's own validity window applies on the
        // direct path just as it does on the chained path (RFC 5280
        // §6.1.3).
        let mut root = parse_cert(&der("ec-ca.der")).unwrap();
        root.not_after = NOW - 1;
        let presented = parse_cert(&der("ec-ca.der")).unwrap();
        let err = validate_with_root(&[presented], &root, NOW).unwrap_err();
        assert!(matches!(err, VerifyError::TimeInvalid(_)), "got {err}");
    }

    /// Minimal DER TLV encoder for building adversarial extension
    /// blocks in tests (short lengths only).
    fn der_tlv(tag: u8, val: &[u8]) -> Vec<u8> {
        assert!(val.len() < 128, "test helper encodes short lengths only");
        let mut out = vec![tag, val.len() as u8];
        out.extend_from_slice(val);
        out
    }

    #[test]
    fn duplicate_extension_rejected() {
        // m6 (adversarial): RFC 5280 §4.2 — a duplicated extension OID
        // is malformed and must fail closed, not resolve last-win
        // (which could let an attacker override a restrictive value).
        let oid_bc = [0x06u8, 0x03, 0x55, 0x1D, 0x13]; // 2.5.29.19
        let bc_val = der_tlv(0x30, &[]); // empty basicConstraints
        let bc_oct = der_tlv(0x04, &bc_val);
        let mut one = oid_bc.to_vec();
        one.extend_from_slice(&bc_oct);
        let one = der_tlv(0x30, &one);
        let mut two = one.clone();
        two.extend_from_slice(&one);
        let block = der_tlv(0xA3, &der_tlv(0x30, &two)); // [3] { SEQ }

        let mut cert = parse_cert(&der("leaf-rsa.der")).unwrap();
        let mut d = Der::new(&block);
        let err = parse_extensions(&mut d, &mut cert).unwrap_err();
        assert!(matches!(err, VerifyError::Malformed(_)), "got {err}");

        // A single copy of the same extension still parses.
        let block = der_tlv(0xA3, &der_tlv(0x30, &one));
        let mut d = Der::new(&block);
        parse_extensions(&mut d, &mut cert).expect("single extension must parse");
    }

    #[test]
    fn sigalg_keytype_mismatch_rejected() {
        // Adversarial DER-level mismatch: the tbs signatureAlgorithm
        // names ECDSA but the issuer key is RSA (and vice versa).
        // verify_link must fail closed rather than coerce or skip.
        let mut leaf = parse_cert(&der("leaf-rsa.der")).unwrap();
        let inter = parse_cert(&der("test-inter.der")).unwrap();
        leaf.sig_alg = SigAlg::Ecdsa(EcdsaHash::Sha256);
        let err = verify_link(&leaf, &inter.spki).unwrap_err();
        assert!(matches!(err, VerifyError::BadSignature(_)), "got {err}");

        let mut ec_leaf = parse_cert(&der("ec-leaf.der")).unwrap();
        let ec_ca = parse_cert(&der("ec-ca.der")).unwrap();
        ec_leaf.sig_alg = SigAlg::RsaPkcs1(RsaHash::Sha256);
        let err = verify_link(&ec_leaf, &ec_ca.spki).unwrap_err();
        assert!(matches!(err, VerifyError::BadSignature(_)), "got {err}");
    }

    #[test]
    fn cn_fallback_skipped_when_non_dns_sans_exist() {
        // m7 (RFC 6125 §6.4.4): the CN fallback applies only when the
        // certificate carries no SANs at all. A cert whose SANs exist
        // but contain no dNSName (here: iPAddress only) must not fall
        // back to a matching CN.
        let mut cert = parse_cert(&der("leaf-rsa-ip.der")).unwrap();
        assert!(!cert.sans.is_empty());
        assert!(!cert.sans.iter().any(|s| matches!(s, San::Dns(_))));
        cert.common_name = Some("localhost".to_string());
        assert!(matches!(
            verify_hostname(&cert, "localhost"),
            Err(VerifyError::HostnameMismatch(_))
        ));
        // The no-SAN fixture still falls back to CN.
        let cnonly = parse_cert(&der("leaf-rsa-cnonly.der")).unwrap();
        assert!(cnonly.sans.is_empty());
        verify_hostname(&cnonly, "bowser.test").unwrap();
    }

    #[test]
    fn leading_dot_host_rejected() {
        // m8: `*.example.com` must never match `.example.com`.
        assert!(!dns_match("*.example.com", ".example.com"));
        assert!(!dns_match("example.com", ".example.com"));
        assert!(!dns_match("*.example.com", "..example.com"));
        // Sanity: legitimate matches still work.
        assert!(dns_match("*.example.com", "www.example.com"));
        assert!(dns_match("example.com", "example.com"));
        // End-to-end through the fixture wildcard cert.
        let wild = parse_cert(&der("leaf-rsa-wildcard.der")).unwrap();
        assert!(matches!(
            verify_hostname(&wild, ".bowser.test"),
            Err(VerifyError::HostnameMismatch(_))
        ));
    }
}
