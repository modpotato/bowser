//! Fuzz the X.509 DER parser and chain validator.
//!
//! `crypto::x509` is `pub(crate)` inside `bowser_net`, and the task forbids
//! modifying crate sources to expose it. Instead this target recompiles the
//! *actual* parser sources (`x509.rs` plus its `ec` / `rsa` / `bignum`
//! dependencies) into the fuzz crate via the `shim::crypto` module, which
//! `#[path]`-includes the real files. No crate source is modified or
//! copied: the files on disk under `crates/bowser_net/src/crypto/` are the
//! ones being compiled, so findings apply to the real code.
//!
//! Two entry points are exercised per input:
//!   1. `parse_cert` - raw DER deserialization.
//!   2. `verify_server_chain` - full RFC 5280 path building + signature
//!      verification (RSA PKCS#1 v1.5 / ECDSA) with the input as a
//!      single-certificate chain against the vendored Mozilla roots.
//!
//! Expected: `Err` on malformed input. Panic / OOM / hang = bug.

#![no_main]

use std::time::SystemTime;

use libfuzzer_sys::fuzz_target;

/// Mirror of `bowser_net`'s `crypto` module layout (see `shim/crypto.rs`) so
/// the `super::` paths inside the included sources keep resolving.
#[path = "../shim/crypto.rs"]
mod crypto;

fuzz_target!(|data: &[u8]| {
    // 1. Raw DER parse.
    let _ = crypto::x509::parse_cert(data);

    // 2. Chain validation against the vendored Mozilla root store
    //    (parsed once, cached in a `OnceLock` inside the parser).
    let roots = crypto::x509::root_store();
    let _ = crypto::x509::verify_server_chain(&[data], "fuzz.invalid", SystemTime::now(), roots);
});
