//! Networking stack for Bowser: hand-rolled TCP/TLS/HTTP, fetch, cookies.
//!
//! Standard-library Rust only. See /AGENTS.md.

pub mod cookies;
pub mod crypto;
pub mod decode;
mod dns;
pub mod fetch;
pub mod hpack;
pub mod http1;
pub mod http2;
mod pool;
pub mod psl;
pub mod tls;
