//! Benchmarks for HPACK header-block decoding through the bump-slab
//! [`Decoder::decode_block`] path.
//!
//! Standard-library only: no benchmark dependencies (see /AGENTS.md).
//! Run with:
//!
//! ```sh
//! cargo bench -p bowser_net --bench hpack_bench
//! ```
//!
//! `cargo bench` passes `--bench` to this binary; without that flag it
//! exits immediately so `cargo test` stays fast even if the target is
//! ever executed there.
//!
//! Scenarios:
//!
//! * `response`: decode one realistic ~15-header HTTP/2 response block
//!   (indexed static-table hits, Huffman-coded literals, one literal
//!   with incremental indexing, and one dynamic-table indexed read)
//!   into a reused [`DecodedHeaders`] slab, reporting ns/op.

use std::time::Instant;

use bowser_net::hpack::{DecodedHeaders, Decoder, Encoder};

/// Append a plain (non-Huffman) HPACK string: 7-bit prefix length +
/// raw bytes.
fn push_plain_string(out: &mut Vec<u8>, s: &[u8]) {
    let mut v = s.len() as u32;
    if v < 127 {
        out.push(v as u8);
    } else {
        out.push(127);
        v -= 127;
        while v >= 128 {
            out.push((v as u8 & 0x7f) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }
    out.extend_from_slice(s);
}

/// A realistic response header block: mostly static-indexed fields,
/// several Huffman-coded literals (the encoder picks Huffman when it
/// shortens), one literal with incremental indexing (exercises the
/// dynamic-table insert path), and one indexed read of dynamic index 62
/// (exercises the dynamic-table read path; it names the entry the
/// incremental field just inserted).
fn response_block() -> Vec<u8> {
    let mut enc = Encoder::new();
    let headers: Vec<(&[u8], &[u8])> = vec![
        (b":status", b"200"),
        (b"content-type", b"text/html; charset=utf-8"),
        (b"content-length", b"12345"),
        (b"date", b"Mon, 12 Oct 2026 07:53:09 GMT"),
        (b"server", b"bowser-test/0.1"),
        (b"cache-control", b"max-age=3600, public"),
        (b"etag", b"\"abc123def456\""),
        (b"set-cookie", b"session=abcdef1234567890; Path=/; HttpOnly"),
        (b"set-cookie", b"theme=dark; Path=/; Max-Age=31536000"),
        (b"x-request-id", b"req-9f8e7d6c5b4a"),
        (
            b"strict-transport-security",
            b"max-age=31536000; includeSubDomains",
        ),
        (b"vary", b"accept-encoding"),
        (
            b"x-custom-long-value",
            b"this is a fairly long header value that huffman-encodes shorter",
        ),
        (b"via", b"1.1 example-proxy"),
    ];
    let mut block = Vec::new();
    enc.encode(&headers, &mut block);
    // Literal with incremental indexing, new name.
    block.push(0x40);
    push_plain_string(&mut block, b"x-session-hint");
    push_plain_string(&mut block, b"dynamic-table-probe");
    // Indexed field naming dynamic index 62 (the entry just inserted).
    block.push(0x80 | 62);
    block
}

fn bench_response_decode() {
    const ITERS: usize = 200_000;
    let block = response_block();
    let mut dec = Decoder::new(4096);
    let mut slab = DecodedHeaders::new();

    // Correctness check outside the timed loop.
    dec.decode_block(&block, 128, &mut slab).unwrap();
    assert_eq!(slab.len(), 16, "expected 16 decoded fields");
    assert_eq!(slab.field(0), (b":status".as_slice(), b"200".as_slice()));
    let (n, v) = slab.field(14);
    assert_eq!(n, b"x-session-hint");
    assert_eq!(v, b"dynamic-table-probe");
    // Field 15 is the dynamic-indexed re-read of field 14.
    assert_eq!(
        slab.field(15),
        (
            b"x-session-hint".as_slice(),
            b"dynamic-table-probe".as_slice()
        )
    );
    // Slab shape: ranges tile the slab with no gaps.
    let total: usize = slab
        .fields
        .iter()
        .map(|f| (f.name.end - f.name.start + f.value.end - f.value.start) as usize)
        .sum();
    assert_eq!(total, slab.bytes.len());

    // Warmup so the slab and dynamic table reach steady state.
    for _ in 0..1_000 {
        dec.decode_block(&block, 128, &mut slab).unwrap();
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        dec.decode_block(&block, 128, &mut slab).unwrap();
    }
    let ns = start.elapsed().as_secs_f64() / ITERS as f64 * 1e9;
    println!(
        "hpack_response_decode: {:>10.1} ns/block  ({:>8.1} blocks/s, {} fields/block)",
        ns,
        1e9 / ns,
        slab.len(),
    );
}

fn main() {
    if !std::env::args().any(|a| a == "--bench") {
        return;
    }
    // HPACK_BENCH_ONLY=response runs a single scenario.
    let only = std::env::var("HPACK_BENCH_ONLY").ok();
    let run = |name: &str| only.as_deref().is_none_or(|o| o == name);

    println!("hpack benchmarks (steady state, reused slab):");
    if run("response") {
        bench_response_decode();
    }
}
