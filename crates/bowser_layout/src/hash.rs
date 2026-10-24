//! Local FxHash (the rustc-hash algorithm): a fast non-cryptographic hash
//! tuned for small integer/pointer keys, used by the multicol pass's
//! per-split `HashMap`s. The standard [`std::collections::HashMap`]
//! defaults to SipHash-1-3 with a random per-map state; on hot
//! per-layout maps that is pure overhead. Mirrors the hasher in
//! `bowser_css::matching`, kept local so this crate stays dependency-free.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};

/// FxHasher: `hash = hash.rotate_left(5) ^ word`, seeded at 0.
#[derive(Clone, Debug)]
pub(crate) struct FxHasher {
    hash: usize,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, i: usize) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // `Hasher`'s default `write_usize` funnels through here for our
        // node-index keys; chunk the bytes into `usize` words.
        for chunk in bytes.chunks(std::mem::size_of::<usize>()) {
            let mut word = 0usize;
            for (i, &b) in chunk.iter().enumerate() {
                word |= (b as usize) << (8 * i);
            }
            self.add(word);
        }
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i as usize);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash as u64
    }
}

/// [`BuildHasher`] for [`FxHasher`]: zero per-map state, so every map
/// lookup skips the random-state setup SipHash pays.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FxBuildHasher;

impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;

    #[inline]
    fn build_hasher(&self) -> FxHasher {
        FxHasher { hash: 0 }
    }
}

/// `HashMap` with the local FxHash: the multicol pass's hot per-split
/// maps (node -> entry index, per-node band decisions, line-map cache).
pub(crate) type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;
/// `HashSet` with the local FxHash (kept for symmetry; no users yet).
#[allow(dead_code)]
pub(crate) type FxHashSet<T> = HashSet<T, FxBuildHasher>;
