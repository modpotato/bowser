//! Benchmarks for the wgpu paint backend lowering path (`wgpu_be.rs`).
//!
//! std-only manual timing harness (`harness = false`): no new
//! dependencies, per the workspace's std-first rule.
//!
//! - `lower_synthetic`: lowers one representative display list via
//!   `Backend::paint` (CPU lowering only: batching, instance emission,
//!   image-hash validation, scissor dedup). This is the headline number
//!   for the batching/hash-cache work.
//! - `frame_synthetic`: full frame on a persistent headless backend via
//!   `render_offscreen_with_viewport` (lower + encode + submit +
//!   readback under lavapipe).
//! - `hash_old_vs_new`: `image_hash` microbenchmark; `hash_old` is the
//!   pre-optimization byte-at-a-time FNV-1a, kept here verbatim as the
//!   reference, `hash_new` is a copy of the chunked version that ships
//!   in `wgpu_be.rs` (the real one is private to the module).
//!
//! Run: `cargo bench -p bowser_paint --bench wgpu_be_lower`
//! (needs the lavapipe env from the wgpu_be test docs for the GPU legs).

use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bowser_layout::{Rect, Size};
use bowser_paint::WgpuBackend;
use bowser_paint::display_list::{
    Backend, BorderStyle, Color, DecoKind, DecodedImage, DisplayItem, DisplayList, FontKey,
    ImageId, MarkerKind,
};
use bowser_paint::fonts::FontCache;

fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
    Rect { x, y, w, h }
}

/// Deterministic synthetic page: solids, borders, images (few unique ids,
/// many references), text, decorations, markers, nested clips and opacity
/// groups (including redundant clip pushes).
fn synthetic_list() -> (DisplayList, usize) {
    let mut list = DisplayList::new();
    let mut items = 0usize;

    // 4000 solid fills: the batching sweet spot (one draw call expected).
    for i in 0..4000u32 {
        let x = ((i * 37) % 780) as f32;
        let y = ((i * 53) % 580) as f32;
        list.push(DisplayItem::FillRect {
            rect: rect(x, y, 20.0, 20.0),
            color: Color::rgba8((i % 256) as u8, 100, 200, 255),
        });
        items += 1;
    }

    // 400 borders: each tessellates to several solid quads.
    for i in 0..400u32 {
        let x = ((i * 91) % 700) as f32;
        let y = ((i * 67) % 540) as f32;
        list.push(DisplayItem::Border {
            rect: rect(x, y, 100.0, 60.0),
            widths: [2.0, 2.0, 2.0, 2.0],
            styles: [BorderStyle::Solid; 4],
            colors: [Color::rgba8(0, 0, 255, 255); 4],
            radii: [0.0; 4],
        });
        items += 1;
    }

    // 6 unique images, each referenced 50 times: 300 ensure_image calls
    // but only 6 distinct content hashes per frame.
    for id in 0..6u32 {
        let w = 32u32;
        let h = 32u32;
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for (p, px) in rgba.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            px[0] = (id * 40 + p as u32 % 256) as u8;
            px[1] = (p as u32 % 128) as u8;
            px[2] = 200;
            px[3] = 255;
        }
        list.add_image(ImageId(id), DecodedImage::from_rgba(w, h, rgba));
    }
    for i in 0..300u32 {
        let id = ImageId(i % 6);
        let x = ((i * 41) % 740) as f32;
        let y = ((i * 29) % 560) as f32;
        list.push(DisplayItem::Image {
            rect: rect(x, y, 32.0, 32.0),
            image: id,
        });
        items += 1;
    }

    // Text: 60 runs x 40 chars through the glyph path.
    for i in 0..60u32 {
        let y = ((i * 97) % 580) as f32;
        list.push(DisplayItem::TextRun {
            text: Arc::from("The quick brown fox jumps over 0123456789"),
            text_range: 0..41,
            rect: rect(10.0, y, 400.0, 20.0),
            baseline: y + 16.0,
            font_key: FontKey(0),
            size_px: 16.0,
            color: Color::rgba8(10, 10, 10, 255),
        });
        items += 1;
    }

    // Nested clips, every other push redundant (same rect twice).
    for i in 0..100u32 {
        let r = rect(50.0, 50.0, 700.0, 500.0);
        list.push(DisplayItem::PushClip { rect: r });
        list.push(DisplayItem::PushClip { rect: r });
        list.push(DisplayItem::FillRect {
            rect: rect(60.0 + (i % 10) as f32, 60.0, 600.0 - (i % 10) as f32, 400.0),
            color: Color::rgba8(255, 255, 0, 255),
        });
        list.push(DisplayItem::PopClip);
        list.push(DisplayItem::PopClip);
        items += 5;
    }

    // Opacity groups.
    for i in 0..100u32 {
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, (i % 60) as f32 * 10.0, 800.0, 10.0),
            color: Color::rgba8(255, 255, 255, 255),
        });
        list.push(DisplayItem::PopOpacity);
        items += 3;
    }

    // Decorations and markers.
    for i in 0..200u32 {
        let y = ((i * 31) % 580) as f32;
        list.push(DisplayItem::Decoration {
            rect: rect(10.0, y, 200.0, 20.0),
            kind: DecoKind::Underline,
            color: Color::rgba8(0, 0, 0, 255),
            thickness: 2.0,
        });
        items += 1;
    }
    for i in 0..100u32 {
        let y = ((i * 43) % 580) as f32;
        let kind = if i % 2 == 0 {
            MarkerKind::Disc
        } else {
            MarkerKind::Decimal(7)
        };
        list.push(DisplayItem::ListMarker {
            rect: rect(220.0, y, 12.0, 12.0),
            kind,
            color: Color::rgba8(0, 0, 0, 255),
            index: i + 1,
            font_key: FontKey(0),
        });
        items += 1;
    }

    (list, items)
}

fn bench_wall(name: &str, iters: usize, mut f: impl FnMut()) -> Duration {
    for _ in 0..3 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let total = start.elapsed();
    let per = total / iters as u32;
    println!("bench {name}: {per:?}/iter ({iters} iters, {total:?} total)");
    per
}

/// Pre-optimization reference: byte-at-a-time FNV-1a (verbatim copy of
/// the old `image_hash`).
fn hash_old(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Copy of the chunked `image_hash` that ships in `wgpu_be.rs`
/// (the real one is private to the module).
fn hash_new(bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x100000001b3;
    let mut h: u64 = 0xcbf29ce484222325;
    let (words, tail) = bytes.as_chunks::<8>();
    for w in words {
        h ^= u64::from_ne_bytes(*w);
        h = h.wrapping_mul(PRIME);
    }
    for &b in tail {
        h ^= u64::from(b);
        h = h.wrapping_mul(PRIME);
    }
    h
}

fn main() {
    let (list, items) = synthetic_list();
    println!("synthetic list: {items} display items");

    let fonts = FontCache::new();
    let mut backend = WgpuBackend::new_headless().expect("no Vulkan adapter for bench");
    backend.set_font_data(fonts.resolver());
    let viewport = Size { w: 800.0, h: 600.0 };

    // Headline: pure lowering cost on a warm backend.
    let per = bench_wall("lower_synthetic", 50, || {
        backend.paint(black_box(&list), viewport, (0.0, 0.0));
    });
    println!(
        "  = {:.1} ns/display-item",
        per.as_nanos() as f64 / items as f64
    );

    // Full frame: lower + encode + submit + readback.
    let per = bench_wall("frame_synthetic_800x600", 10, || {
        let px = backend.render_offscreen_with_viewport(black_box(&list), viewport, 800, 600);
        black_box(px.len());
    });
    println!("  = {per:?}/frame");

    // Hash microbenchmark on an 800x600 RGBA image.
    let img = vec![0xABu8; 800 * 600 * 4];
    assert_ne!(hash_old(&img), 0);
    assert_ne!(hash_new(&img), 0);
    let per_old = bench_wall("hash_old_800x600", 200, || {
        black_box(hash_old(black_box(&img)));
    });
    let per_new = bench_wall("hash_new_800x600", 200, || {
        black_box(hash_new(black_box(&img)));
    });
    let mb = img.len() as f64 / 1e6;
    println!(
        "  old: {:.0} MB/s, new: {:.0} MB/s ({:.2}x)",
        mb / per_old.as_secs_f64(),
        mb / per_new.as_secs_f64(),
        per_old.as_secs_f64() / per_new.as_secs_f64()
    );
}
