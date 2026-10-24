//! Manual benchmark harness for URL parsing (std only, no dependencies).
//!
//! Run with `cargo bench`. Prints ns/iter per case (minimum of 3 trials,
//! each targeting ~0.5 s of work).

use bowser_url::{Url, parse};
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Time `f` adaptively: scale iterations until a trial takes >= 0.25 s,
/// then take the best ns/iter of 3 trials.
fn bench(mut f: impl FnMut()) -> f64 {
    // Warm up.
    for _ in 0..1_000 {
        f();
    }
    let mut iters = 1_000u64;
    loop {
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        let elapsed = start.elapsed();
        if elapsed >= Duration::from_millis(250) || iters >= 10_000_000 {
            let mut best = elapsed.as_secs_f64() / iters as f64;
            for _ in 0..2 {
                let start = Instant::now();
                for _ in 0..iters {
                    f();
                }
                let ns = start.elapsed().as_secs_f64() / iters as f64;
                if ns < best {
                    best = ns;
                }
            }
            return best * 1e9;
        }
        iters *= 4;
    }
}

fn case(name: &str, input: &str) {
    let ns = bench(|| {
        black_box(parse(black_box(input), None).unwrap());
    });
    println!("{name:<28} {ns:>12.1} ns/iter");
}

fn main() {
    println!("=== parse() ===");
    case("simple", "https://example.org/dir/page");
    case("full", "https://user:pass@example.com:8080/a/b?x=1#y");
    case("query_frag", "https://example.org/dir/page?q=1#frag");
    case("idna", "https://faß.ExAmPlE/a/b?x=1#f");
    case("ipv4", "http://192.168.0.1:8080/a/b?x=1");
    case("ipv6", "http://[2001:db8::1]/a/b?x=1#f");
    case("opaque", "mailto:foo@bar.example");
    case("file", "file:///C|/demo/test");
    case("nonspecial_auth", "foo://user@host:1234/a/b");
    case("uppercase_host", "HTTPS://EXAMPLE.ORG/A/B?X=1");
    case(
        "long_path",
        "https://example.com/seg0/seg1/seg2/seg3/seg4/seg5/seg6/seg7/seg8/seg9/",
    );
    case(
        "long_query",
        "https://example.com/s?a=1&b=2&c=3&d=4&e=5&f=6&g=7&h=8#frag",
    );
    case("dot_segments", "http://h/a/./b/../c/./d/../e");
    case("pct_path", "https://example.com/a%20b/c%C3%A9/d");

    println!("=== parse() with base ===");
    let base: Url = parse("https://example.org/dir/page", None).unwrap();
    let ns = bench(|| {
        black_box(parse(black_box("../other?q=1#frag"), Some(black_box(&base))).unwrap());
    });
    println!("{:<28} {:>12.1} ns/iter", "relative", ns);

    println!("=== serialize() ===");
    let u: Url = parse("https://user:pass@example.com:8080/a/b?x=1#y", None).unwrap();
    let ns = bench(|| {
        black_box(black_box(&u).serialize());
    });
    println!("{:<28} {:>12.1} ns/iter", "serialize", ns);
}
