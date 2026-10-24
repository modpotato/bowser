//! Fuzz the TLS 1.3 client handshake parser: the record layer
//! (`read_record`), the `HandshakeFramer` reassembly, ServerHello parsing
//! (`read_server_hello` / `parse_sh_extensions`), alert parsing
//! (`parse_alert`), and the encrypted-flight loop (incl. the change_cipher_spec
//! flood guard).
//!
//! Strategy: libfuzzer's bytes are replayed as the *server* flight over a
//! loopback TCP pair, then the real `TlsConnection::connect` runs the client
//! handshake against them. The server side half-closes (Shutdown::Write)
//! immediately after writing, so client reads hit EOF instead of blocking;
//! a read timeout is also set so a parser that spins waiting for more input
//! degrades to an error, while libfuzzer's `-timeout` flag turns a true hang
//! into a finding.
//!
//! The handshake is *expected* to fail on random bytes. Only panics, aborts,
//! OOMs, or hangs are findings.

#![no_main]

use std::io::Write;
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::OnceLock;
use std::thread;
use std::time::Duration;

use bowser_net::tls::{TlsConfig, TlsConnection};
use libfuzzer_sys::fuzz_target;

/// One loopback listener shared by all fuzz iterations in this process.
fn listener() -> &'static TcpListener {
    static L: OnceLock<TcpListener> = OnceLock::new();
    L.get_or_init(|| TcpListener::bind("127.0.0.1:0").expect("bind loopback listener"))
}

fn config() -> TlsConfig {
    TlsConfig {
        server_name: "fuzz.invalid".to_string(),
        alpn: vec!["http/1.1".to_string()],
        fingerprint_profile: None,
    }
}

fuzz_target!(|data: &[u8]| {
    // The record layer rejects fragments > 2^14 + 256 bytes; bounding the
    // write keeps each iteration cheap.
    let buf: Vec<u8> = data.iter().take(1 << 20).copied().collect();

    let addr = listener().local_addr().expect("listener addr");

    // Fake server: accept one connection, dump the fuzzed bytes as the
    // server flight, then half-close so the client sees EOF.
    let server = thread::spawn(move || {
        let Ok((mut s, _)) = listener().accept() else {
            return;
        };
        let _ = s.write_all(&buf);
        let _ = s.shutdown(Shutdown::Write);
        // Socket drops here; the client's in-flight reads get data-then-EOF.
    });

    let stream = TcpStream::connect(addr).expect("connect to loopback listener");
    // Safety net: a parser that blocks waiting for more input becomes an
    // error after this instead of stalling the fuzzer worker.
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));

    // Expected to fail on random bytes; a panic/hang/abort is the finding.
    let _ = TlsConnection::connect(stream, &config());

    let _ = server.join();
});
