//! Benchmark harness for text decoding (std only, no deps).
//!
//! Times the `decode` / `decode_with` hot paths on deterministic
//! synthetic inputs: UTF-8 (ASCII-heavy and multibyte), UTF-16LE/BE,
//! and a legacy single-byte encoding (windows-1252), plus label lookup.
//!
//! Run with: `cargo bench -p bowser_encoding` (uses the workspace bench
//! profile). This is a `harness = false` bench so it works on stable
//! Rust with zero dependencies.

use bowser_encoding::{Encoding, decode, decode_with, encoding_from_label};
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Time `f` over `runs` iterations, return median seconds.
fn bench<R>(
    name: &str,
    bytes_len: usize,
    runs: usize,
    warmup: usize,
    mut f: impl FnMut() -> R,
) -> f64
where
    R: Sized,
{
    for _ in 0..warmup {
        std::hint::black_box(f());
    }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        std::hint::black_box(f());
        times.push(start.elapsed().as_secs_f64());
    }
    let med = median(times);
    let mib = bytes_len as f64 / (1024.0 * 1024.0);
    println!(
        "  {name:<34} median: {:>9.3} ms ({:>8.1} MiB/s)",
        med * 1000.0,
        mib / med.max(f64::MIN_POSITIVE)
    );
    med
}

fn ascii_html(size: usize) -> Vec<u8> {
    let chunk = b"<div class=\"x\">lorem ipsum dolor sit amet <b>bold</b></div>\n";
    let mut v = Vec::with_capacity(size);
    while v.len() < size {
        v.extend_from_slice(chunk);
    }
    v.truncate(size);
    v
}

fn multibyte_utf8(size: usize) -> Vec<u8> {
    // Mix of 2- and 3-byte sequences: enough to exercise the multibyte path.
    let words = ["héllo", "wörld", "日本語", "текст", "😀", "café", "naïve"];
    let mut v = Vec::with_capacity(size);
    let mut i = 0;
    while v.len() < size {
        v.extend_from_slice(words[i % words.len()].as_bytes());
        v.push(b' ');
        i += 1;
    }
    v.truncate(size);
    v
}

fn utf16le(bytes: &[u8]) -> Vec<u8> {
    let s = String::from_utf8_lossy(bytes);
    let mut v = Vec::with_capacity(s.len() * 2 + 2);
    v.extend_from_slice(&[0xFF, 0xFE]); // BOM
    for u in s.encode_utf16() {
        v.extend_from_slice(&u.to_le_bytes());
    }
    v
}

fn main() {
    println!("encoding decode benches:");
    let ascii = ascii_html(1 << 20);
    bench("decode utf-8 ascii 1MiB", ascii.len(), 11, 2, || {
        decode(&ascii, Encoding::Utf8).unwrap()
    });
    let mb = multibyte_utf8(1 << 20);
    bench("decode utf-8 multibyte 1MiB", mb.len(), 11, 2, || {
        decode(&mb, Encoding::Utf8).unwrap()
    });
    let u16 = utf16le(&ascii);
    bench("decode utf-16le+bom 2MiB", u16.len(), 11, 2, || {
        decode(&u16, Encoding::Utf8).unwrap()
    });
    bench("decode_with windows-1252 1MiB", ascii.len(), 11, 2, || {
        decode_with(&ascii, Encoding::Windows1252).unwrap()
    });
    bench("label lookup x1000", 0, 101, 5, || {
        let mut n = 0;
        for _ in 0..1000 {
            n += encoding_from_label("windows-1252").is_some() as usize;
            n += encoding_from_label("  UTF-16LE ").is_some() as usize;
            n += encoding_from_label("bogus-label").is_none() as usize;
        }
        n
    });
}
