//! Benchmarks for the connection-pool reuse policy and the cold-start
//! dial path, through the public [`fetch`] API against local servers.
//!
//! Standard-library only: no benchmark dependencies (see /AGENTS.md).
//! Run with:
//!
//! ```sh
//! cargo bench -p bowser_net --bench pool_bench
//! ```
//!
//! `cargo bench` passes `--bench` to this binary; without that flag it
//! exits immediately so `cargo test` stays fast even if the target is
//! ever executed there.
//!
//! Scenarios:
//!
//! * `reuse`: sequential fetches to one origin. Every fetch after the
//!   first must hit the pool (`take` liveness probe + `put`); the
//!   server must see exactly one TCP connection. This is the pool hot
//!   path end to end.
//! * `dial`: each fetch goes to a fresh port, so every fetch pays a
//!   full cold-start dial (DNS + TCP connect + socket options). This
//!   isolates dial-path overhead such as the Happy Eyeballs thread
//!   spawn.
//! * `burst`: T threads x M sequential fetches each against one
//!   origin. Documents the per-origin concurrency policy: how many
//!   connections a racing burst opens before the pool starts
//!   absorbing it.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use bowser_net::fetch::{Request, fetch};
use bowser_url::Url;

fn url(s: &str) -> Url {
    bowser_url::parse(s, None).unwrap()
}

fn get_request(url_str: &str) -> Request {
    Request {
        method: "GET".to_string(),
        url: url(url_str),
        headers: Vec::new(),
        body: None,
    }
}

/// Read one request head from `stream`; empty vec on EOF/error.
fn read_request(stream: &mut TcpStream) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        match stream.read(&mut tmp) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
        }
    }
    buf
}

fn ok_response() -> Vec<u8> {
    b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nok".to_vec()
}

/// A keep-alive HTTP/1.1 server: each accepted connection is served on
/// its own thread until the peer goes away. Returns the base URL and a
/// counter of accepted TCP connections.
fn spawn_keepalive_server() -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let conns = Arc::new(AtomicUsize::new(0));
    let count = conns.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            count.fetch_add(1, Ordering::Relaxed);
            std::thread::spawn(move || {
                let resp = ok_response();
                loop {
                    if read_request(&mut stream).is_empty() {
                        break;
                    }
                    if stream.write_all(&resp).is_err() {
                        break;
                    }
                }
            });
        }
    });
    (format!("http://127.0.0.1:{port}"), conns)
}

/// Median of `samples` (sorted in place).
fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn bench_h1_sequential_reuse() {
    // 200 sequential fetches to one origin: the pool must serve 199 of
    // them from the idle entry (take + probe + put). The server must
    // see exactly one connection.
    const ITERS: usize = 200;
    let (base, conns) = spawn_keepalive_server();
    let url = format!("{base}/r");
    // One primer so the measured loop never includes the cold dial.
    fetch(&get_request(&url)).unwrap();
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let resp = fetch(&get_request(&url)).unwrap();
        assert_eq!(resp.body, b"ok");
        times.push(start.elapsed().as_secs_f64());
    }
    let t = median(&mut times);
    let n = conns.load(Ordering::Relaxed);
    println!(
        "h1_sequential_reuse: {:>8.1} fetch/s  ({:.3} ms/op, {} iters, {} tcp conns)",
        1.0 / t,
        t * 1000.0,
        ITERS,
        n,
    );
    assert_eq!(n, 1, "sequential fetches must reuse one pooled connection");
}

fn bench_h1_cold_dial() {
    // Every fetch targets a fresh port (a fresh pool key), so each one
    // pays the full cold-start dial: DNS, TCP connect, socket options.
    const ITERS: usize = 40;
    let servers: Vec<(String, Arc<AtomicUsize>)> =
        (0..ITERS).map(|_| spawn_keepalive_server()).collect();
    // Let the listeners settle before timing.
    std::thread::sleep(Duration::from_millis(100));
    let mut times = Vec::with_capacity(ITERS);
    for (base, _) in &servers {
        let start = Instant::now();
        let resp = fetch(&get_request(&format!("{base}/c"))).unwrap();
        assert_eq!(resp.body, b"ok");
        times.push(start.elapsed().as_secs_f64());
    }
    let t = median(&mut times);
    println!(
        "h1_cold_dial:        {:>8.1} fetch/s  ({:.3} ms/op, {} iters)",
        1.0 / t,
        t * 1000.0,
        ITERS,
    );
}

fn bench_h1_concurrent_burst() {
    // 8 threads x 25 sequential fetches against one origin: measures
    // how many connections a racing burst opens before pool reuse
    // absorbs it, and the resulting fetch throughput.
    const THREADS: usize = 8;
    const PER_THREAD: usize = 25;
    let (base, conns) = spawn_keepalive_server();
    std::thread::sleep(Duration::from_millis(100));
    let start = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..THREADS {
            s.spawn(|| {
                for _ in 0..PER_THREAD {
                    let resp = fetch(&get_request(&format!("{base}/b"))).unwrap();
                    assert_eq!(resp.body, b"ok");
                }
            });
        }
    });
    let t = start.elapsed().as_secs_f64();
    let total = (THREADS * PER_THREAD) as f64;
    println!(
        "h1_burst_8x25:       {:>8.1} fetch/s  ({:.3} ms/op, {} tcp conns)",
        total / t,
        t * 1000.0 / total,
        conns.load(Ordering::Relaxed),
    );
}

fn main() {
    if !std::env::args().any(|a| a == "--bench") {
        return;
    }
    // POOL_BENCH_ONLY=reuse|dial|burst runs a single scenario.
    let only = std::env::var("POOL_BENCH_ONLY").ok();
    let run = |name: &str| only.as_deref().is_none_or(|o| o == name);

    println!("pool benchmarks (median of iters, localhost):");
    if run("reuse") {
        bench_h1_sequential_reuse();
    }
    if run("dial") {
        bench_h1_cold_dial();
    }
    if run("burst") {
        bench_h1_concurrent_burst();
    }
}
