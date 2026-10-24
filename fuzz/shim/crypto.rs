//! Shim that recompiles `bowser_net`'s real `crypto::{bignum, ec, rsa, x509}`
//! sources into the fuzz crate.
//!
//! `crypto::x509` is `pub(crate)` upstream and the task forbids modifying
//! crate sources to expose it, so this module includes the actual files from
//! `crates/bowser_net/src/crypto/` via `#[path]`. Nothing is copied: the
//! files on disk are the ones being compiled, so any finding applies
//! directly to the real code.
//!
//! The module layout mirrors `bowser_net::crypto` so the `super::` paths
//! inside the included sources keep resolving:
//! * `sha2` is re-exported from the real crate (it is already `pub` there).
//! * `ct_eq` is duplicated from `crypto/mod.rs` (8 lines, `pub(crate)`
//!   upstream, invisible outside the crate).

pub mod sha2 {
    pub use bowser_net::crypto::sha2::*;
}

/// Duplicated from `crates/bowser_net/src/crypto/mod.rs`; the original is
/// `pub(crate)` and not visible outside `bowser_net`.
pub(crate) fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

#[path = "../../crates/bowser_net/src/crypto/bignum.rs"]
pub mod bignum;
#[path = "../../crates/bowser_net/src/crypto/ec.rs"]
pub mod ec;
#[path = "../../crates/bowser_net/src/crypto/rsa.rs"]
pub mod rsa;
#[path = "../../crates/bowser_net/src/crypto/x509.rs"]
pub mod x509;
