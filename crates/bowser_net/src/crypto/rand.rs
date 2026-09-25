//! Operating-system entropy for key generation, standard library only.
//!
//! Rust's `std` has no portable cryptographic RNG, so this module reads the
//! OS entropy source directly. On Unix that is `/dev/urandom`; on other
//! platforms key generation returns [`io::ErrorKind::Unsupported`].

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
        let mut f = File::open("/dev/urandom")?;
        f.read_exact(buf)
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
