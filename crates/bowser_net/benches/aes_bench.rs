//! Manual benchmark harness for the AES block cipher.
//!
//! Standard-library only: no benchmark dependencies (see AGENTS.md).
//! Run with:
//!
//! ```sh
//! cargo bench -p bowser_net --bench aes_bench
//! ```
//!
//! `cargo bench` passes `--bench` to this binary; without that flag it
//! exits immediately so `cargo test` stays fast even if the target is
//! ever executed there.

use std::time::Instant;

use bowser_net::crypto::{Aes128, Aes256};

/// Iterations per scenario.
const ITERS: u64 = 1_000_000;

/// Time `f` over `ITERS` iterations (after warmup) and print ns/block.
fn bench(name: &str, mut f: impl FnMut()) {
    for _ in 0..10_000 {
        f();
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    let ns = start.elapsed().as_secs_f64() / ITERS as f64 * 1e9;
    println!(
        "{name:<24} {:>10.1} ns/block  ({:>8.1} MB/s)",
        ns,
        16.0 / ns * 1e3
    );
}

fn main() {
    if !std::env::args().any(|a| a == "--bench") {
        return;
    }
    println!("aes benchmarks ({ITERS} iters each, single block, chained):");

    let aes128 = Aes128::new(&[0x2bu8; 16]);
    let mut b = [0u8; 16];
    let mut i = 0u64;
    bench("aes128 encrypt_block", || {
        i += 1;
        b[0..8].copy_from_slice(&i.to_le_bytes());
        aes128.encrypt_block(&mut b);
        std::hint::black_box(&b);
    });
    bench("aes128 decrypt_block", || {
        i += 1;
        b[0..8].copy_from_slice(&i.to_le_bytes());
        aes128.decrypt_block(&mut b);
        std::hint::black_box(&b);
    });

    let aes256 = Aes256::new(&[0x2bu8; 32]);
    bench("aes256 encrypt_block", || {
        i += 1;
        b[0..8].copy_from_slice(&i.to_le_bytes());
        aes256.encrypt_block(&mut b);
        std::hint::black_box(&b);
    });
    bench("aes256 decrypt_block", || {
        i += 1;
        b[0..8].copy_from_slice(&i.to_le_bytes());
        aes256.decrypt_block(&mut b);
        std::hint::black_box(&b);
    });
}
