//! Manual benchmark harness for the fetch hot paths.
//!
//! Standard-library only: no benchmark dependencies (see /AGENTS.md).
//! Run with:
//!
//! ```sh
//! cargo bench -p bowser_net --bench fetch_bench
//! ```
//!
//! `cargo bench` passes `--bench` to this binary; without that flag it
//! exits immediately so `cargo test` stays fast even if the target is
//! ever executed there.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
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

/// Read one request (head plus any framed body) from `stream`.
fn read_request(stream: &mut TcpStream) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let head_end = loop {
        match stream.read(&mut tmp) {
            Ok(0) | Err(_) => break buf.len(),
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    break i + 4;
                }
            }
        }
    };
    let content_length = String::from_utf8_lossy(&buf[..head_end])
        .lines()
        .filter_map(|l| l.split_once(':'))
        .find(|(name, _)| name.trim().eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let mut remaining = content_length.saturating_sub(buf.len() - head_end);
    while remaining > 0 {
        let want = remaining.min(tmp.len());
        let n = match stream.read(&mut tmp[..want]) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        remaining -= n;
    }
    buf
}

/// Spawn a server handling `n` connections; `handler` maps raw request
/// bytes to raw response bytes. Returns the base URL.
fn spawn_server<F>(n: usize, handler: F) -> String
where
    F: Fn(&[u8]) -> Vec<u8> + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for _ in 0..n {
            let Ok((mut stream, _)) = listener.accept() else {
                break;
            };
            let req = read_request(&mut stream);
            let resp = handler(&req);
            let _ = stream.write_all(&resp);
        }
    });
    format!("http://127.0.0.1:{port}")
}

/// Path of an absolute-form request target, e.g. `/r1`.
fn request_path(raw: &[u8]) -> String {
    let line_end = raw.iter().position(|&b| b == b'\r').unwrap_or(raw.len());
    let line = &raw[..line_end];
    let mut parts = line.split(|&b| b == b' ');
    parts.next();
    let target = parts.next().unwrap_or(b"/");
    let target = std::str::from_utf8(target).unwrap_or("/");
    let after_scheme = target.split("://").nth(1).unwrap_or(target);
    let path: String = after_scheme
        .split('/')
        .skip(1)
        .collect::<Vec<_>>()
        .join("/");
    let idx = target
        .len()
        .checked_sub(path.len())
        .and_then(|n| n.checked_sub(1))
        .unwrap_or(0);
    target[idx..].to_string()
}

fn head(status: u16, headers: &[(&str, &str)]) -> Vec<u8> {
    let mut out = format!("HTTP/1.1 {status} OK\r\n").into_bytes();
    for (n, v) in headers {
        out.extend_from_slice(n.as_bytes());
        out.extend_from_slice(b": ");
        out.extend_from_slice(v.as_bytes());
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(b"\r\n");
    out
}

/// Median of `samples` (sorted in place).
fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn bench_bulk_content_length() {
    // 16 MiB body with Content-Length framing: exercises the read loop
    // and the final body copy.
    const BODY: usize = 16 * 1024 * 1024;
    const ITERS: usize = 5;
    let base = spawn_server(ITERS, |_| {
        let mut r = head(200, &[("Content-Length", &BODY.to_string())]);
        r.resize(r.len() + BODY, b'x');
        r
    });
    let url = format!("{base}/big");
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let resp = fetch(&get_request(&url)).unwrap();
        assert_eq!(resp.body.len(), BODY);
        times.push(start.elapsed().as_secs_f64());
    }
    let t = median(&mut times);
    println!(
        "bulk_content_length: {:>8.1} MiB/s  ({:.1} ms/op, {} iters)",
        BODY as f64 / t / 1048576.0,
        t * 1000.0,
        ITERS
    );
}

fn bench_chunked() {
    // 4 MiB as 4096 1-KiB chunks plus one trailer: exercises the
    // streaming chunked decoder.
    const CHUNKS: usize = 4096;
    const CHUNK: usize = 1024;
    const ITERS: usize = 5;
    let base = spawn_server(ITERS, |_| {
        let mut r = head(200, &[("Transfer-Encoding", "chunked")]);
        let payload = vec![b'y'; CHUNK];
        for _ in 0..CHUNKS {
            r.extend_from_slice(format!("{CHUNK:x}\r\n").as_bytes());
            r.extend_from_slice(&payload);
            r.extend_from_slice(b"\r\n");
        }
        r.extend_from_slice(b"0\r\nX-Sum: abc\r\n\r\n");
        r
    });
    let url = format!("{base}/chunked");
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let resp = fetch(&get_request(&url)).unwrap();
        assert_eq!(resp.body.len(), CHUNKS * CHUNK);
        assert_eq!(resp.header("x-sum"), Some("abc"));
        times.push(start.elapsed().as_secs_f64());
    }
    let t = median(&mut times);
    println!(
        "chunked_4k_chunks:   {:>8.1} MiB/s  ({:.1} ms/op, {} iters)",
        (CHUNKS * CHUNK) as f64 / t / 1048576.0,
        t * 1000.0,
        ITERS
    );
}

fn bench_header_heavy() {
    // 90 response headers, tiny body: exercises the header copy path.
    // (http1's default `Limits` caps headers at 100.)
    const HEADERS: usize = 90;
    const ITERS: usize = 30;
    let header_block: String = (0..HEADERS)
        .map(|i| format!("X-Header-{i:03}: value-{i:03}-padding-padding\r\n"))
        .collect();
    let base = spawn_server(ITERS, move |_| {
        let mut r =
            format!("HTTP/1.1 200 OK\r\n{header_block}Content-Length: 2\r\n\r\n").into_bytes();
        r.extend_from_slice(b"ok");
        r
    });
    let url = format!("{base}/headers");
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let resp = fetch(&get_request(&url)).unwrap();
        assert_eq!(resp.headers.len(), HEADERS + 1);
        times.push(start.elapsed().as_secs_f64());
    }
    let t = median(&mut times);
    println!(
        "header_heavy_90:     {:>8.1} fetch/s  ({:.2} ms/op, {} iters)",
        1.0 / t,
        t * 1000.0,
        ITERS
    );
}

fn bench_redirect_chain() {
    // 5-hop redirect chain: exercises per-hop connect + head overhead.
    const HOPS: usize = 5;
    const ITERS: usize = 20;
    let base = spawn_server(ITERS * HOPS, |req| {
        let path = request_path(req);
        if let Some(n) = path.strip_prefix("/r")
            && let Ok(i) = n.parse::<usize>()
            && i + 1 < HOPS
        {
            return head(302, &[("Location", &format!("/r{}", i + 1))]);
        }
        let mut r = head(200, &[("Content-Length", "5")]);
        r.extend_from_slice(b"final");
        r
    });
    let url = format!("{base}/r0");
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let resp = fetch(&get_request(&url)).unwrap();
        assert_eq!(resp.body, b"final");
        times.push(start.elapsed().as_secs_f64());
    }
    let t = median(&mut times);
    println!(
        "redirect_5_hops:     {:>8.1} fetch/s  ({:.2} ms/op, {} iters)",
        1.0 / t,
        t * 1000.0,
        ITERS
    );
}

fn main() {
    if !std::env::args().any(|a| a == "--bench") {
        return;
    }
    // FETCH_BENCH_ONLY=bulk|chunked|headers|redirects runs a single
    // scenario (used for syscall counting under strace).
    let only = std::env::var("FETCH_BENCH_ONLY").ok();
    let run = |name: &str| only.as_deref().is_none_or(|o| o == name);
    // Warm up: one throwaway fetch so thread stacks, DNS, and the
    // allocator are past first-touch costs before timing starts.
    let warm = spawn_server(1, |_| {
        let mut r = head(200, &[("Content-Length", "1")]);
        r.extend_from_slice(b"w");
        r
    });
    let _ = fetch(&get_request(&format!("{warm}/")));
    std::thread::sleep(Duration::from_millis(100));

    println!("fetch benchmarks (median of iters, localhost):");
    if run("bulk") {
        bench_bulk_content_length();
    }
    if run("chunked") {
        bench_chunked();
    }
    if run("headers") {
        bench_header_heavy();
    }
    if run("redirects") {
        bench_redirect_chain();
    }
}
