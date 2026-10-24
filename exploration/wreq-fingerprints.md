# wreq fingerprint data survey (project Bowser)

Date: 2026-09-25. Explorer: read-only survey. The local clone at
`~/workspace/bowser/vendor-sources/wreq` did not exist at survey time
(re-checked over ~8 min), so findings are from upstream via the GitHub API
and raw file reads (repo state: commit `e3922a2`).

## Critical correction: the profiles are NOT in the wreq repo

The fingerprint profiles live in the **separate `wreq-util` repo**
(`github.com/0x676e67/wreq-util`, "Common utilities for wreq"). The `wreq`
repo consumes `wreq-util`; cloning `wreq` alone will not yield the profiles.
Recommend the parent also fetch `wreq-util` if local sources are wanted.

## Where the profiles live

All under `src/emulate/` of `wreq-util` (main branch):

- `src/emulate.rs` (14,881 B) - `Emulation`/`Profile` enums (via `define_enum!`
  macro), `Platform` enum (Windows/MacOS/Linux/Android/iOS), weighted-random
  picker (StatCounter June 2026 weights).
- `src/emulate/macros.rs` (11,718 B) - `mod_generator!` macro (generates one
  module per browser version: `emulation()` fn returning TLS+HTTP/2+headers),
  header-builder macros.
- `src/emulate/profile.rs` (1,255 B) - `build_standard_emulation()` glue.
- `src/emulate/profile/chrome.rs` (88,725 B) + `chrome/{tls,http2,header}.rs`
  (5,815 / 1,767 / 2,793 B) - Chrome + Edge + Opera version modules.
- `src/emulate/profile/firefox.rs` (14,678 B) + `firefox/{tls,http2,header}.rs`
  (8,359 / 3,754 / 1,082 B).
- `src/emulate/profile/safari.rs` (7,106 B) + `safari/{tls,http2,header}.rs`
  (5,798 / 3,687 / 2,246 B).
- `src/emulate/profile/okhttp.rs` (8,133 B).
- `src/emulate/profile/opera.rs` (10,336 B) + `opera/{tls,http2,header}.rs`
  (2,899 / 1,382 / 766 B).

Profile data files total: **169,326 bytes (~165 KiB)** of Rust source.
Full tree including the generator machinery: **~196 KiB**.

## Profile count: 137

Counted from the `Profile` enum in `src/emulate.rs`:

| Family | Variants |
|---|---|
| Chrome | 45 (v100-v153, non-contiguous; e.g. 100,101,104-110,114,116-120,123,124,126-153) |
| Edge | 19 (101,122,127,131,134-148) |
| Opera | 16 (116-131) |
| Firefox | 20 (109,117,128,133,135,136,139,142-151 + private_135/136 + android_135) |
| Safari | 29 (macOS 15.x-18.x/26.x, iPadOS 18/26, iOS 16.5/17.x/18.1.1/26.x) |
| OkHttp | 8 (3.9, 3.11, 3.13, 3.14, 4.9, 4.10, 4.12, 5) |

Each family file is macro-generated: one `mod_generator!(vNNN, <tls cfg>, <http2 cfg>, <header fn>, [(Platform, sec-ch-ua, user-agent), ...])`
per version. Newer versions mostly reuse an older version's TLS+HTTP/2 config
(e.g. `v101` reuses `v100::build_emulation`) and differ only in UA/sec-ch-ua strings.

## What one profile contains (Chrome 131-era example)

**TLS** (`tls.rs` consts + `ChromeTlsConfig` builder):
- Cipher list (ordered, `:`-joined names): `TLS_AES_128_GCM_SHA256`,
  `TLS_AES_256_GCM_SHA384`, `TLS_CHACHA20_POLY1305_SHA256`, then 12 legacy
  ECDHE/RSA suites (15 total).
- Curves: `X25519MLKEM768:X25519:P-256:P-384` (post-quantum hybrid first).
- SigAlgs: `ecdsa_secp256r1_sha256:rsa_pss_rsae_sha256:rsa_pkcs1_sha256:...`
  (8 entries; Chrome 150+ adds `mldsa44:mldsa65:mldsa87` first, + sigalg GREASE).
- Flags: GREASE enabled, `permute_extensions(true)`, `enable_ech_grease(true)`,
  `pre_shared_key(true)`, OCSP stapling + SCT enabled, min/max TLS 1.2/1.3,
  ALPN `h2,http/1.1`, ALPS for HTTP/2 (new codepoint on 152+), cert compression
  = Brotli, `trust_anchors` = Chromium encoded Trust Anchor IDs (optional
  `emulation-chromium-pki` feature).
- Firefox instead pins an explicit extension permutation order
  (`EXTENSION_PERMUTATION_INDICES`: server_name, extended_master_secret,
  renegotiate, supported_groups, ec_point_formats, session_ticket, alpn,
  status_request, delegated_credentials, cert_timestamp, key_share,
  supported_versions, sigalgs, psk_kex_modes, record_size_limit,
  compress_certificate, ech), plus `key_shares: [X25519_MLKEM768, X25519, P256]`,
  delegated-credentials sigalgs, `record_size_limit = 0x4001`, cert compressors
  zlib/brotli/zstd.

**HTTP/2** (`http2.rs`):
- SETTINGS values: initial_window_size 6291456, initial_connection_window_size
  15728640, max_header_list_size 262144, header_table_size 65536,
  max_concurrent_streams 1000, enable_push false (variant-dependent).
- SETTINGS frame order: HEADER_TABLE_SIZE, ENABLE_PUSH, MAX_CONCURRENT_STREAMS,
  INITIAL_WINDOW_SIZE, MAX_FRAME_SIZE, MAX_HEADER_LIST_SIZE,
  ENABLE_CONNECT_PROTOCOL, NO_RFC7540_PRIORITIES.
- Pseudo-header order: `:method :authority :scheme :path`.
- Stream dependency: parent stream 0, weight 255, exclusive (Chrome);
  Firefox/Safari define their own priority trees.

**Headers** (`header.rs`): ordered default header set -
`sec-ch-ua`, `sec-ch-ua-mobile`, `sec-ch-ua-platform`, `upgrade-insecure-requests`,
`user-agent`, `accept`, `sec-fetch-site/mode/dest/user`, `accept-encoding`
(gzip, deflate, br; zstd variants on newer), `accept-language`, and `priority: u=0, i`
on Chrome 152+. Each version x platform (MacOS/Linux/Android/Windows/iOS)
carries its own UA and `sec-ch-ua` strings.

## Vendoring assessment: data yes, behavior no

**The DATA is vendored easily.** Every fingerprint value is a plain string
constant (cipher/curve/sigalg/extension/ALPN names), integer (SETTINGS values,
stream weights), or ordered list (extension permutation, SETTINGS order,
pseudo-header order, header order) plus UA strings. Mapping names to numeric
TLS codepoints is public IANA registry knowledge. A std-only Rust crate can
hold all 137 profiles as a compact data table (est. well under 100 KiB in a
binary-friendly encoding, since most versions share TLS/HTTP2 configs and
differ only in UA strings).

**The profiles are code, not data files.** There is no JSON/TOML to copy; the
values are embedded in Rust macros that call `wreq`'s BoringSSL-backed
`TlsOptions`/`Http2Options`/header builders. We must transcribe the constants,
not depend on the crate.

**Applying a profile needs real work in Bowser's own stacks:**
- TLS: Bowser's hand-rolled TLS must assemble the ClientHello byte-for-byte:
  exact extension order, GREASE insertion points, extension permutation
  (BoringSSL's algorithm), ECH GREASE, ALPS extension, cert-compression
  negotiation, and **post-quantum key shares** (X25519MLKEM768 = ML-KEM-768
  keygen, ~1124-byte share - real crypto, not just bytes to copy). rustls
  explicitly rejects ClientHello customization (rustls#1932), so the stock
  ecosystem cannot do this; it has to be Bowser's own TLS.
- HTTP/2: Bowser's HTTP/2 must emit SETTINGS in profile order with exact
  values, honor pseudo-header order and stream priorities/dependencies.
- HTTP/1: header order + case must be preserved on the wire.

Net: wreq-util gives us the *what* (all 137 profiles' constants, kept current
by an active maintainer). Bowser must build the *how* (ClientHello/HTTP2/HTTP1
wire-format control) itself. The behavioral logic (permutation, GREASE, PQ
keygen) lives in BoringSSL + wreq's glue, not in the profile constants.

## License

`wreq-util` is **Apache-2.0**. Vendoring its constants into Bowser requires
preserving the Apache-2.0 license notice/attribution for the copied material
(keep a `NOTICE`/attribution file referencing `github.com/0x676e67/wreq-util`).

## Recommendations for parent

1. Fetch `wreq-util` (not just `wreq`) into `vendor-sources/` if local
   reference sources are desired - that is where all profile data lives.
2. Task a worker to transcribe the profile constants from
   `wreq-util/src/emulate/profile/**` into a std-only data module/table
   (no wreq dependency), preserving Apache-2.0 attribution.
3. The harder dependency is behavioral: Bowser's TLS stack needs ClientHello
   template control (extension order/GREASE/permutation/ECH-grease/ALPS/PQ
   key shares) - plan this as a TLS-stack feature, not a data import.
