//! Operating-system entropy for key generation, standard library only.
//!
//! Rust's `std` has no portable cryptographic RNG, so this module reads the
//! OS entropy source directly. On Unix that is `/dev/urandom`, opened once
//! and cached for the life of the process; on other platforms key generation
//! returns [`io::ErrorKind::Unsupported`].
//!
//! Platform note: non-Unix support (e.g. BCryptGenRandom on Windows,
//! `getentropy` on macOS is still `/dev/urandom`) is deliberately left as
//! `Unsupported` here. Adding it is a separate platform-support decision,
//! not something to sneak in as part of a performance fix.

use std::io;

/// Fill `buf` with bytes from the operating system's entropy source.
///
/// # Errors
///
/// Returns an error if the OS entropy source cannot be opened or read, or if
/// the platform has no supported entropy source.
pub(crate) fn fill_random(buf: &mut [u8]) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::fs::File;
        use std::io::Read as _;
        use std::sync::OnceLock;

        // Cached for the process lifetime: avoids an open(2) per call in the
        // TLS handshake hot path. `/dev/urandom` is offset-independent, so
        // concurrent reads through a shared `&File` are safe without a lock
        // (std implements `Read for &File`). `get_or_try_init` would be
        // ideal here but is still unstable, so the `Result` is cached.
        static URANDOM: OnceLock<io::Result<File>> = OnceLock::new();
        let f = URANDOM
            .get_or_init(|| File::open("/dev/urandom"))
            .as_ref()
            .map_err(|e| io::Error::new(e.kind(), e.to_string()))?;
        let mut reader: &File = f;
        reader.read_exact(buf)
    }
    #[cfg(not(unix))]
    {
        let _ = buf;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "no OS entropy source on this platform",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::fill_random;

    #[test]
    fn repeated_calls_succeed() {
        // Exercise the cached-fd path repeatedly; every call must succeed.
        for _ in 0..64 {
            let mut buf = [0u8; 48];
            fill_random(&mut buf).expect("fill_random failed");
        }
    }

    #[test]
    fn repeated_calls_return_distinct_bytes() {
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        fill_random(&mut a).expect("fill_random failed");
        fill_random(&mut b).expect("fill_random failed");
        // Two independent 256-bit draws colliding has probability 2^-256;
        // treat equality as failure.
        assert_ne!(a, b, "two random fills returned identical bytes");
        // Also sanity-check the stream is not degenerate (e.g. all zeros).
        assert!(a.iter().any(|&x| x != 0), "random fill was all zeros");
    }

    #[test]
    fn empty_and_odd_lengths_ok() {
        fill_random(&mut []).expect("empty fill failed");
        let mut one = [0u8; 1];
        fill_random(&mut one).expect("1-byte fill failed");
        let mut odd = [0u8; 7];
        fill_random(&mut odd).expect("7-byte fill failed");
    }
}
