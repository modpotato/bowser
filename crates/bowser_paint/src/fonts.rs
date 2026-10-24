//! System font discovery, caching, and real text measurement.
//!
//! [`FontCache`] discovers system fonts with pure-std directory walking
//! plus `read-fonts` SFNT parsing (no fontconfig/fontique C FFI), matches
//! CSS font requests `(family, weight, style)` to concrete faces, and
//! caches them behind [`crate::display_list::FontKey`] handles.
//! [`FontData`] owns the raw bytes of a resolved face; the glyph/shaping
//! worker gets a [`swash::FontRef`] from it via [`FontData::font_ref`].
//!
//! [`FontCache`] also implements [`bowser_layout::FontMetrics`] so layout
//! measures text with real glyph advances. [`SwashFontMetrics`] is the
//! named production implementation: it wraps the window's cache and
//! measures with paint's own [`crate::text::shape_text`] (kerning,
//! ligature substitution), so laid-out line breaks and fragment widths
//! agree with the painted glyphs. `FontMetrics` is family-aware: layout
//! interns each text run's family once ([`bowser_layout::FontFamilyId`])
//! and every measurement resolves that family's face, so runs styled
//! with a non-default family (e.g. the Ahem test font) measure with the
//! right face's advances and ascent/descent. `line_height` uses the
//! face's typographic ascent/descent/leading, scaled to the requested
//! size. (Weight/style still ride the default 400/normal: that is the
//! remaining known residual mismatch.)
//!
//! ## Test fonts
//!
//! [`AHEM_FONT_BYTES`] embeds the Ahem test font so WPT runs can resolve
//! the `@font-face` family `/fonts/ahem.css` declares without network
//! font fetching (see [`FontCache::install_test_fonts`]).
//!
//! - Source: web-platform-tests `fonts/Ahem.ttf` (created by Tavmjong
//!   Bah for CSS test suites).
//! - Stored per AGENTS.md under
//!   `crates/bowser_paint/tests/fixtures/Ahem.ttf` (never inline in
//!   source).
//!
//! `FontKey` is imported from [`crate::display_list`], never redefined here
//! (the display-list worker owns it).
//!
//! ## Embedded fallback font
//!
//! [`FALLBACK_FONT_BYTES`] embeds a last-resort face so text renders on
//! fontless systems (CI): Cousine Regular subset to Basic Latin + Latin-1
//! Supplement (U+0020-007E, U+00A0-00FF), 14,364 bytes, stored per
//! AGENTS.md under `crates/bowser_paint/tests/fixtures/fallback-sans.ttf`
//! (never inline in source).
//!
//! - License: SIL Open Font License 1.1 (OFL-1.1).
//! - Source: <https://github.com/google/fonts>,
//!   `ofl/cousine/Cousine-Regular.ttf` (The Cousine Project Authors).
//! - Subset command (fonttools 4.46.0):
//!   `python3 -m fontTools.subset cousine.ttf --output-file=fallback-sans.ttf
//!   --unicodes="U+0020-007E,U+00A0-00FF" --no-hinting --desubroutinize
//!   --name-IDs="0,1,2,16,17"` (name ID 0 keeps the OFL copyright notice).
//!
//! Resolution always ends at the fallback face, so [`FontCache::resolve`]
//! effectively never returns `None`: unknown families degrade through the
//! generic-family tables below, then to the embedded face.
//!
//! ## Discovery cost
//!
//! [`FontCache::new`] reads every font file under the platform font
//! directories once to index family/weight/style. Face *bytes* are loaded
//! lazily on first resolution ([`FontCache::font_data`] reads the file
//! then), but the index pass itself is O(total font bytes). Callers that
//! render repeatedly (the WPT driver) should build one `FontCache` and
//! reuse it rather than constructing one per render: reuse never re-walks
//! the directories (the walk happens only in `new`). [`SharedFontCache`]
//! is the cheaply clonable, thread-safe handle for sharing one instance
//! across renders and threads.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use bowser_layout::{FontFamilyId, FontMetrics};
use read_fonts::tables::name::NameId;
use read_fonts::{FontRef, TableProvider};
use swash::proxy::MetricsProxy;

pub use crate::display_list::FontKey;
use crate::text::shape_text;

/// FNV-1a hasher (std-only, no dependency): one XOR-multiply round per
/// input byte. Far cheaper than SipHash (`RandomState`) for the short
/// keys here, and — unlike a bare rotate-multiply FxHash — its per-byte
/// mixing keeps the table index bits well distributed for short keys
/// that differ only in high bytes (measured 2.2x faster `get` than
/// `RandomState` on a 2000-entry string table). Used for the
/// query/face/measurement caches below, whose keys are short family-name
/// strings, `FontKey` integers, and small tuples probed on every font
/// resolution and measurement — throughput matters and the fixed FNV-1a
/// seed is accepted deliberately: family names arrive via document
/// content (stylesheets, `font-family` attributes), which *is*
/// attacker-controlled in a browser, so a hostile page could craft
/// colliding keys and degrade these caches to linear chains. That worst
/// case is acknowledged (not prevented): the seed stays fixed because a
/// SipHash-strength hasher would cost throughput on every font probe,
/// and each table is small and per-cache.
#[derive(Clone, Debug, Default)]
pub(crate) struct FnvBuildHasher;

#[derive(Clone, Debug)]
pub(crate) struct FnvHasher(u64);

impl std::hash::BuildHasher for FnvBuildHasher {
    type Hasher = FnvHasher;
    #[inline]
    fn build_hasher(&self) -> FnvHasher {
        // FNV offset basis.
        FnvHasher(0xcbf29ce484222325)
    }
}

impl std::hash::Hasher for FnvHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // FNV prime.
        const P: u64 = 0x100000001b3;
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(P);
        }
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// [`HashMap`] with the [`FnvBuildHasher`] above.
pub(crate) type FnvHashMap<K, V> = HashMap<K, V, FnvBuildHasher>;
/// [`HashSet`] with the [`FnvBuildHasher`] above.
pub(crate) type FnvHashSet<T> = HashSet<T, FnvBuildHasher>;

/// Resolved `(family, weight, style)` query cache: normalized family
/// name -> `(weight, style)` -> face key.
type ByQuery = FnvHashMap<String, FnvHashMap<(u16, FontStyle), FontKey>>;

/// CSS font style for cache lookups.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FontStyle {
    #[default]
    Normal,
    Italic,
    Oblique,
}

impl FontStyle {
    /// Parse a CSS `font-style` descriptor keyword (`normal` /
    /// `italic` / `oblique`, case-insensitive); anything else —
    /// including `oblique <angle>` — degrades to `Normal`, matching
    /// the rule-descriptor handling in [`FontCache::install_font_owned_as`]
    /// callers that pass the descriptor string through.
    pub fn from_descriptor(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "italic" => FontStyle::Italic,
            "oblique" => FontStyle::Oblique,
            _ => FontStyle::Normal,
        }
    }
}

/// Embedded last-resort face (see the module docs for license/source).
const FALLBACK_FONT_BYTES: &[u8] = include_bytes!("../tests/fixtures/fallback-sans.ttf");

/// The Ahem test font, vendored for WPT runs (see the module docs for
/// provenance). Installed via [`FontCache::install_test_fonts`] so
/// `@font-face` families from `/fonts/*.css` resolve without network
/// font fetching.
const AHEM_FONT_BYTES: &[u8] = include_bytes!("../tests/fixtures/Ahem.ttf");

/// WOFF1 repack of [`AHEM_FONT_BYTES`] (zlib-compressed tables, no
/// metadata/private data), used by the [`decode_webfont`] tests.
/// Generated by `tools/make_woff.py` from `Ahem.ttf`.
#[cfg(test)]
const WOFF_AHEM_BYTES: &[u8] = include_bytes!("../tests/fixtures/Ahem.woff");

/// Where a discovered face's bytes come from.
enum FaceSource {
    /// A system font file; `index` is the face index for collections.
    File { path: PathBuf, index: u32 },
    /// An in-memory font (the embedded fallback, or a face installed
    /// via [`FontCache::install_font`]); `index` is the face index for
    /// collections.
    Embedded { bytes: &'static [u8], index: u32 },
    /// A fetched font's owned bytes (see
    /// [`FontCache::install_font_owned`]); `index` is the face index
    /// for collections. Stored as `Arc<Vec<u8>>` so [`face_data`] can
    /// hand [`FontData`] its `Arc` without copying.
    Owned { bytes: Arc<Vec<u8>>, index: u32 },
}

/// One discovered face: its source plus the indexed metadata used for
/// matching. Bytes are loaded lazily (see [`FontCache::face_data`]).
struct Face {
    source: FaceSource,
    /// Normalized (lowercase, whitespace-collapsed) family names from the
    /// name table (typographic family 16 preferred, legacy family 1 as
    /// well so style-linked names like "Foo Bold" still match).
    families: Vec<String>,
    /// OS/2 `usWeightClass` (1..=1000), else head `macStyle` Bold -> 700.
    weight: u16,
    /// OS/2 `usWidthClass` (1..=9, 5 is normal), else 5.
    width: u16,
    /// OS/2 `fsSelection` italic/oblique bits, else head `macStyle`.
    style: FontStyle,
}

/// System font registry: discovery, matching, caching, measurement.
///
/// Discovery happens in [`FontCache::new`] (pure-std walk of the platform
/// font directories, `read-fonts` parsing); matching is lazy per
/// `(family, weight, style)` triple and cached, so repeated resolutions
/// are a hash lookup. Face bytes load lazily on first use.
///
/// There is no process-wide instance: every owner constructs its own
/// `FontCache` and threads it explicitly (the app worker holds one per
/// window; the headless binary builds one and reuses it across renders).
/// Owners that need the cache on several threads, or behind a
/// `&dyn FontMetrics`, share it through [`SharedFontCache`] (cheap
/// `Arc`-based clone). `FontKey`s are cache-local, so a display list must
/// always be rendered against the same cache that resolved its keys.
///
/// Shared font-data lookup hook, installed on the GPU backend via
/// `WgpuBackend::set_font_data` (the CPU backend's `Rc` hook in
/// `crate::render` is built the same way). Cloning the `Arc` shares the
/// closure, and the [`FontData`] it returns shares the font bytes through
/// its own `Arc`, so reinstalling the same hook is a no-op by pointer
/// identity and keeps the backends' thread-local font-data caches warm.
pub(crate) type FontHook = Arc<dyn Fn(FontKey) -> Option<FontData> + Send + Sync>;

/// Cached measurement setup for one `(font, size)` pair.
///
/// The swash proxy is a plain integer (`Copy`): it records where the
/// metrics tables live in the font, borrowing nothing. Each
/// [`FontMetrics`] line-height call materializes it against the
/// already-resolved [`FontData`] (one struct copy — no table walks).
/// Previously every [`FontCache::advance`] call re-ran the SFNT table
/// walks (`cmap` subtable search, `head`/`maxp`/`os2`/`hhea` reads) that
/// build these proxies; layout calls `advance` per word/fragment, so the
/// walks dominated measurement. Advance now goes through paint's
/// [`shape_text`], whose thread-local cache plays the same role;
/// `line_height` keeps the proxy cache below. Materializing a cached
/// proxy yields bit-identical results to building it fresh.
#[derive(Clone, Copy)]
struct MeasureSetup {
    /// Owned metrics-table locations; materialize per call, then
    /// `.scale(size)`.
    metrics: MetricsProxy,
}

/// Bound on [`FontCache::measure_cache`] entries: `(font, size)` pairs
/// are few in practice (one default font, a handful of sizes); the bound
/// keeps a pathological size sequence (e.g. animated font-size) from
/// growing the map without bound. Eviction only costs recomputation —
/// entries are pure functions of their key.
const MEASURE_CACHE_CAP: usize = 64;

pub struct FontCache {
    faces: Vec<Face>,
    fallback: Face,
    /// Resolved queries: normalized family name -> `(weight, style)` ->
    /// face key. Two levels so the hot lookup
    /// ([`FontCache::resolve`]) probes the outer map with the caller's
    /// `&str` directly (`String: Borrow<str>`): a repeat query for an
    /// already-normalized family name (`"serif"`, `"sans-serif"`, ...)
    /// allocates nothing. The owned family string is materialized only
    /// on the miss path, when a new outer entry is inserted.
    ///
    /// `RefCell` because [`FontMetrics`] takes `&self` (see
    /// [`FontCache::resolve`]): resolution is logically a cache fill,
    /// never a mutation of the resolved faces.
    by_query: RefCell<ByQuery>,
    by_key: RefCell<FnvHashMap<FontKey, FontData>>,
    next_key: Cell<u32>,
    default_key: Option<FontKey>,
    /// [`FontFamilyId`] -> normalized family name, populated by
    /// [`FontMetrics::intern_family`]. `RefCell` because
    /// [`FontMetrics`] is `&self`; entries are keyed inserts only (a
    /// family name never changes meaning), so nothing cached here can
    /// go stale.
    family_names: RefCell<FnvHashMap<FontFamilyId, String>>,
    /// [`FontFamilyId`] -> resolved face for (400, Normal), populated on
    /// first measurement through [`FontCache::resolve_family_id`].
    /// `RefCell` for the same reason; cleared by
    /// [`FontCache::install_font`] / [`FontCache::install_font_owned`]
    /// (a new face may match better).
    family_faces: RefCell<FnvHashMap<FontFamilyId, FontKey>>,
    /// Per-`(font_key, size_bits)` measurement setup (see
    /// [`MeasureSetup`]): the owned swash metrics proxy behind
    /// [`line_height`](FontMetrics::line_height), so repeat calls skip
    /// the SFNT table walks. (`advance` shapes through paint's
    /// [`shape_text`], whose thread-local cache covers it.)
    /// `RefCell` because [`FontMetrics`] takes `&self`; entries are
    /// keyed inserts only (faces are never removed, bytes immutable),
    /// so nothing cached here can go stale.
    measure_cache: RefCell<FnvHashMap<(FontKey, u32), MeasureSetup>>,
    /// The last [`FontCache::resolver`] hook built, shared until a new
    /// face is resolved. Rebuilding the resolver clones the whole
    /// resolved-face map, so backends that reinstall it every frame
    /// (notably [`WgpuBackend::render_offscreen`]) share one `Arc`
    /// instead of re-cloning per frame; hook identity (`Arc::ptr_eq`)
    /// is what keeps the backends' thread-local font-data caches warm.
    /// Dropped by [`FontCache::insert_data`] whenever a new face lands
    /// in `by_key`, so a later `resolver()` sees the full key space.
    cached_resolver: RefCell<Option<FontHook>>,
}

// Public API: the render and backend workers resolve fonts through
// `FontCache`; the shaping worker builds on it next.
impl FontCache {
    /// Discover system fonts and resolve the default (serif) face.
    ///
    /// The default is resolved eagerly so [`FontMetrics`] (which only gets
    /// `&self`) can measure without mutation. It is the face for CSS's
    /// initial `font-family: serif` (`bowser_css`'s initial value), which
    /// is what paint resolves for every unstyled text run — measuring
    /// with it keeps laid-out widths aligned with the painted glyphs.
    /// The embedded fallback face guarantees a default even on fontless
    /// systems.
    pub fn new() -> Self {
        let mut cache = Self {
            faces: discover_system_faces(),
            fallback: fallback_face(),
            by_query: RefCell::new(FnvHashMap::default()),
            by_key: RefCell::new(FnvHashMap::default()),
            next_key: Cell::new(0),
            default_key: None,
            family_names: RefCell::new(FnvHashMap::default()),
            family_faces: RefCell::new(FnvHashMap::default()),
            measure_cache: RefCell::new(FnvHashMap::default()),
            cached_resolver: RefCell::new(None),
        };
        cache.default_key = cache.resolve("serif", 400, FontStyle::Normal);
        cache
    }

    /// Number of resolved faces currently cached.
    // Sibling workers (app) inspect cache pressure through
    // `SharedFontCache::font_count`.
    #[allow(dead_code)]
    pub fn font_count(&self) -> usize {
        self.by_key.borrow().len()
    }

    /// Resolve `(family, weight, style)` to a [`FontKey`].
    ///
    /// CSS generic families (`serif`, `sans-serif`, `monospace`, `cursive`,
    /// `fantasy`) map to ordered well-known family-name lists (a heuristic
    /// standing in for fontconfig's generic-alias configuration); anything
    /// else is a named family. Cache hits return the same key as the first
    /// resolution. When no system face matches, the embedded fallback face
    /// is returned, so this effectively never returns `None`.
    ///
    /// Takes `&self`: resolution only fills the query caches (interior
    /// mutability), which is what lets [`FontMetrics`] measure through a
    /// shared cache.
    pub fn resolve(&self, family: &str, weight: u16, style: FontStyle) -> Option<FontKey> {
        let weight = weight.clamp(1, 1000);
        // Normalize once. When the caller's string is already in
        // normalized form (the common case: `"serif"`, `"sans-serif"`,
        // single-word family names) this borrows it, so the lookup
        // below allocates nothing.
        let family = normalize_family_cow(family);
        let family_str: &str = family.as_ref();
        let hit = self
            .by_query
            .borrow()
            .get(family_str)
            .and_then(|by_style| by_style.get(&(weight, style)))
            .copied();
        if let Some(key) = hit {
            return Some(key);
        }
        // Miss: materialize the owned family string once and cache the
        // outcome. The embedded fallback means this is always `Some`,
        // so a repeated query for an unknown family never re-walks the
        // face list either.
        let family_owned = family.into_owned();
        let matched = self.resolve_uncached(&family_owned, weight, style);
        if let Some(font_key) = matched {
            self.by_query
                .borrow_mut()
                .entry(family_owned)
                .or_default()
                .insert((weight, style), font_key);
        }
        matched
    }

    /// The default measurement font's key (always `Some`: the embedded
    /// fallback guarantees it). This is the face layout measures with
    /// (see [`SwashFontMetrics`]).
    // Intentionally-public API: sibling workers (app) need the default
    // key directly.
    #[allow(dead_code)]
    pub fn default_key(&self) -> Option<FontKey> {
        self.default_key
    }

    /// Access the decoded bytes for `key`. The glyph/shaping worker uses
    /// this to get [`FontData::font_ref`]. Cloned (the bytes are
    /// `Arc`-shared, so this is a refcount bump, not a copy).
    pub fn font_data(&self, key: FontKey) -> Option<FontData> {
        self.by_key.borrow().get(&key).cloned()
    }

    /// A `Send + Sync + 'static` font-data hook over the faces resolved so
    /// far, for backends whose hook must be shareable across threads (the
    /// GPU backend; the CPU backend's `Rc` hook is built the same way in
    /// `crate::render`).
    ///
    /// The built hook is cached and stable across calls while no new face
    /// is resolved: every call used to clone the whole resolved-face map
    /// (including every font byte vector), and each fresh `Arc` flushed
    /// the backends' thread-local font-data caches (their entries are
    /// validated by hook identity). The bytes themselves are `Arc`-shared
    /// ([`FontData`]), so even the (rare) rebuild only bumps refcounts.
    /// Resolving a new face drops the cached hook, so the next call
    /// rebuilds over the full key space.
    pub fn resolver(&self) -> FontHook {
        if let Some(hook) = self.cached_resolver.borrow().as_ref() {
            return Arc::clone(hook);
        }
        let map = Arc::new(self.snapshot());
        let hook: FontHook = Arc::new(move |key| map.get(&key).cloned());
        *self.cached_resolver.borrow_mut() = Some(Arc::clone(&hook));
        hook
    }

    /// Copy of the resolved faces, for building backend hooks.
    pub(crate) fn snapshot(&self) -> FnvHashMap<FontKey, FontData> {
        self.by_key.borrow().clone()
    }

    /// Install an in-memory font face: indexes its family/weight/style
    /// from the name table like a system face, so [`resolve`](Self::resolve)
    /// (and family-aware [`FontMetrics`] measurement) can match it.
    /// Returns `false` when the bytes do not parse as a font. The bytes
    /// must live for `'static` (e.g. [`include_bytes!`]) because the
    /// cache retains them.
    ///
    /// Called at setup time, before the cache is shared: it mutates the
    /// face list. Cached resolutions for the installed families are
    /// dropped so the new face wins immediately.
    pub fn install_font(&mut self, bytes: &'static [u8]) -> bool {
        let mut installed = false;
        let mut families: Vec<String> = Vec::new();
        for (index, font) in FontRef::fonts(bytes).enumerate() {
            let Ok(font) = font else { continue };
            let index = index as u32;
            let names = family_names(&font);
            if names.is_empty() {
                continue;
            }
            let (weight, width, style) = face_attributes(&font);
            families.extend(names.iter().cloned());
            self.faces.push(Face {
                source: FaceSource::Embedded { bytes, index },
                families: names,
                weight,
                width,
                style,
            });
            installed = true;
        }
        if installed {
            self.invalidate_families(&families);
        }
        installed
    }

    /// Install a fetched font face from owned bytes: the same indexing
    /// as [`install_font`](Self::install_font), but the bytes need not
    /// live for `'static` — fetched fonts arrive as an owned `Vec<u8>`
    /// (there is no `'static` slice to hand the cache), so the cache
    /// retains them in a [`FaceSource::Owned`] face instead.
    ///
    /// Returns `false` when the bytes are neither an SFNT font nor a
    /// decodable WOFF1 font: woff2 (`wOF2`) has no decoder yet and is
    /// rejected by the format check below, as is anything that does
    /// not parse via `FontRef`. WOFF1 bytes are decoded to SFNT by
    /// [`decode_webfont`] first, so dedupe and storage see the decoded
    /// bytes (a WOFF1 file and its source TTF dedupe to one face).
    /// Reinstalling identical bytes is a no-op returning `true`
    /// (content dedupe, so a load that installs twice — loader plus
    /// window install — does not duplicate faces).
    ///
    /// Called at setup/load time, before the cache is shared: it
    /// mutates the face list. Cached resolutions for the installed
    /// families are dropped so the new face wins immediately.
    pub fn install_font_owned(&mut self, bytes: Arc<[u8]>) -> bool {
        self.install_font_owned_inner(bytes, None)
    }

    /// Install a fetched font face under the `@font-face` rule's own
    /// descriptors, not the file's internal metadata: a rule that
    /// renames a file (`font-family: Brand`, `src: Ahem.ttf`) must
    /// make `Brand` resolve to the webfont, and its weight/style
    /// descriptors must win over the file's OS/2 values for matching.
    /// The file's internal family names are kept as extra aliases, so
    /// a page that also names the internal family keeps resolving.
    /// Dedupe is per (bytes, family, weight, style): the same file
    /// installed for two families is two faces, not one dropped rule.
    /// Returns `false` when the bytes are neither an SFNT font nor a
    /// decodable WOFF1 font (same gate as
    /// [`install_font_owned`](Self::install_font_owned)).
    pub fn install_font_owned_as(
        &mut self,
        bytes: Arc<[u8]>,
        family: &str,
        weight: u16,
        style: FontStyle,
    ) -> bool {
        let family = normalize_family(family);
        if family.is_empty() {
            return false;
        }
        self.install_font_owned_inner(bytes, Some((family, weight.clamp(1, 1000), style)))
    }

    fn install_font_owned_inner(
        &mut self,
        bytes: Arc<[u8]>,
        override_: Option<(String, u16, FontStyle)>,
    ) -> bool {
        // WOFF1 arrives here from every installer (loader, navigation,
        // JS `FontFace`): decoding up front means all three get it for
        // free, and dedupe below compares decoded SFNT bytes. `None`
        // for woff2/garbage/malformed input.
        let Some(bytes) = decode_webfont(&bytes) else {
            return false;
        };
        if !is_sfnt(&bytes) {
            return false;
        }
        if let Some((family, weight, style)) = &override_ {
            // Same bytes, same override identity: already installed.
            if self.faces.iter().any(|face| {
                matches!(&face.source, FaceSource::Owned { bytes: have, .. } if have.as_slice() == bytes.as_ref())
                    && face.families.iter().any(|name| name == family)
                    && face.weight == *weight
                    && face.style == *style
            }) {
                return true;
            }
        } else if self.faces.iter().any(|face| {
            matches!(&face.source, FaceSource::Owned { bytes: have, .. } if have.as_slice() == bytes.as_ref())
        }) {
            return true;
        }
        let owned: Arc<Vec<u8>> = Arc::new(bytes.to_vec());
        let mut installed = false;
        let mut families: Vec<String> = Vec::new();
        for (index, font) in FontRef::fonts(owned.as_slice()).enumerate() {
            let Ok(font) = font else { continue };
            let index = index as u32;
            let names = family_names(&font);
            if names.is_empty() {
                continue;
            }
            let (font_weight, width, font_style) = face_attributes(&font);
            let (weight, style) = override_
                .as_ref()
                .map(|(_, weight, style)| (*weight, *style))
                .unwrap_or((font_weight, font_style));
            let mut face_families = names;
            if let Some((family, _, _)) = &override_
                && !face_families.contains(family)
            {
                face_families.insert(0, family.clone());
            }
            families.extend(face_families.iter().cloned());
            self.faces.push(Face {
                source: FaceSource::Owned {
                    bytes: Arc::clone(&owned),
                    index,
                },
                families: face_families,
                weight,
                width,
                style,
            });
            installed = true;
        }
        if installed {
            self.invalidate_families(&families);
        }
        installed
    }

    /// Drop cached resolutions for `families` (and the family-id face
    /// cache): a newly installed face may be a better match than
    /// whatever the query caches currently hold for these families.
    /// Generic-family entries are evicted too: a cached `serif`
    /// resolution may have arrived through a candidate name (`noto
    /// serif`) that a new face just claimed, so the generic key must
    /// not keep pointing at the stale face.
    fn invalidate_families(&mut self, families: &[String]) {
        let mut by_query = self.by_query.borrow_mut();
        for name in families {
            let name = normalize_family(name);
            by_query.remove(&name);
            for generic in ["serif", "sans-serif", "monospace", "cursive", "fantasy"] {
                if let Some(parsed) = generic_family(generic)
                    && generic_candidates(parsed).contains(&name.as_str())
                {
                    by_query.remove(generic);
                }
            }
        }
        drop(by_query);
        self.family_faces.borrow_mut().clear();
    }

    /// Install the WPT test fonts vendored under `tests/fixtures/`
    /// (currently Ahem) so `@font-face` families from `/fonts/*.css`
    /// resolve during WPT runs without network font fetching. The WPT
    /// driver calls this once on its shared cache before a run;
    /// `document.fonts.ready` is an already-resolved shim, so no async
    /// wait is needed.
    pub fn install_test_fonts(&mut self) {
        self.install_font(AHEM_FONT_BYTES);
    }

    /// Resolve a [`FontFamilyId`] (from [`FontMetrics::intern_family`])
    /// to a face key at weight 400 / normal style, caching by id. Ids
    /// that were never interned (or whose family has no face) fall back
    /// to the default face, so this effectively never returns `None`.
    fn resolve_family_id(&self, family: FontFamilyId) -> Option<FontKey> {
        if let Some(&key) = self.family_faces.borrow().get(&family) {
            return Some(key);
        }
        let name = self.family_names.borrow().get(&family).cloned();
        let key = match name {
            Some(name) => self.resolve(&name, 400, FontStyle::Normal),
            None => self.default_key,
        }
        .or(self.default_key);
        if let Some(key) = key {
            self.family_faces.borrow_mut().insert(family, key);
        }
        key
    }

    /// `(ascent, descent)` magnitudes in px at `font_size` for the face
    /// `key` resolves to. `None` when the bytes do not parse or the
    /// metrics are degenerate (callers then keep the 0.8/0.2 probe
    /// split).
    fn face_ascent_descent(&self, key: FontKey, font_size: f32) -> Option<(f32, f32)> {
        let data = self.font_data(key)?;
        let font = data.font_ref()?;
        let setup = self.measure_setup(key, &font, font_size);
        let coords: &[swash::NormalizedCoord] = &[];
        let m = setup
            .metrics
            .materialize_metrics(&font, coords)
            .scale(font_size);
        // swash 0.2 reports both as positive distances from the
        // baseline, which is exactly the magnitude pair the
        // [`FontMetrics::ascent_descent`] contract deals in.
        (m.ascent > 0.0 && m.descent > 0.0).then_some((m.ascent, m.descent))
    }

    /// x-height of the face `key` at `font_size`, from swash metrics.
    /// `None` when the face has no positive x-height (some symbol /
    /// fallback faces report 0).
    fn face_x_height(&self, key: FontKey, font_size: f32) -> Option<f32> {
        let data = self.font_data(key)?;
        let font = data.font_ref()?;
        let setup = self.measure_setup(key, &font, font_size);
        let coords: &[swash::NormalizedCoord] = &[];
        let m = setup
            .metrics
            .materialize_metrics(&font, coords)
            .scale(font_size);
        (m.x_height > 0.0).then_some(m.x_height)
    }

    /// Run one query and cache the winning face: named/generic family
    /// lookup first, embedded fallback when nothing matches.
    fn resolve_uncached(&self, family: &str, weight: u16, style: FontStyle) -> Option<FontKey> {
        if let Some(generic) = generic_family(family) {
            for name in generic_candidates(generic) {
                if let Some(face) = self.best_face(name, weight, style) {
                    return self.load_face(face);
                }
            }
        } else if let Some(face) = self.best_face(family, weight, style) {
            return self.load_face(face);
        }
        // Last resort: the embedded fallback face. Loading embedded bytes
        // cannot fail, so this is always `Some`.
        let data = face_data(&self.fallback.source)?;
        Some(self.insert_data(data))
    }

    /// Best system face for `family` (already normalized): CSS weight
    /// ordering, then style closeness, then width closeness to normal
    /// (queries carry no stretch, so a condensed "Noto Sans" must not
    /// shadow the regular one). `None` when no system face carries that
    /// family name.
    fn best_face(&self, family: &str, weight: u16, style: FontStyle) -> Option<usize> {
        let candidates: Vec<usize> = self
            .faces
            .iter()
            .enumerate()
            .filter(|(_, face)| face.families.iter().any(|name| name == family))
            .map(|(index, _)| index)
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let mut weights: Vec<u16> = candidates.iter().map(|&i| self.faces[i].weight).collect();
        weights.sort_unstable();
        weights.dedup();
        for w in weight_preference_order(weight, &weights) {
            let mut best: Option<usize> = None;
            // (style penalty, width distance from normal): exact style
            // and normal width win; first in discovery order wins ties.
            let mut best_score = (u8::MAX, u16::MAX);
            for &i in &candidates {
                let face = &self.faces[i];
                if face.weight != w {
                    continue;
                }
                let score = (style_penalty(style, face.style), face.width.abs_diff(5));
                if score < best_score {
                    best = Some(i);
                    best_score = score;
                    if score == (0, 0) {
                        break;
                    }
                }
            }
            if best.is_some() {
                return best;
            }
        }
        None
    }

    /// Load a system face's bytes and mint a [`FontKey`] for it. `None`
    /// when the file is unreadable (it changed since discovery).
    fn load_face(&self, index: usize) -> Option<FontKey> {
        let data = face_data(&self.faces[index].source)?;
        Some(self.insert_data(data))
    }

    /// Mint a [`FontKey`] for decoded face bytes.
    fn insert_data(&self, data: FontData) -> FontKey {
        let key = FontKey(self.next_key.get());
        self.next_key.set(key.0 + 1);
        self.by_key.borrow_mut().insert(key, data);
        // A new face widens the key space: drop the cached resolver so
        // the next `resolver()` rebuilds over every resolved face.
        self.cached_resolver.borrow_mut().take();
        key
    }

    /// Cached [`MeasureSetup`] for `(key, font_size)`, building the
    /// swash metrics proxy (the SFNT table walks) on first use.
    ///
    /// The proxy borrows nothing, so it is safe to retain across
    /// calls; each caller materializes it against its own
    /// [`swash::FontRef`] borrow.
    fn measure_setup(&self, key: FontKey, font: &swash::FontRef, font_size: f32) -> MeasureSetup {
        let cache_key = (key, font_size.to_bits());
        if let Some(&setup) = self.measure_cache.borrow().get(&cache_key) {
            return setup;
        }
        let setup = MeasureSetup {
            metrics: MetricsProxy::from_font(font),
        };
        let mut cache = self.measure_cache.borrow_mut();
        if cache.len() >= MEASURE_CACHE_CAP {
            cache.clear();
        }
        cache.insert(cache_key, setup);
        setup
    }
}

impl Default for FontCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Shareable, thread-safe handle to one [`FontCache`].
///
/// [`FontCache::new`] walks the system font directories once; sharing one
/// instance across renders (instead of constructing one per render) is
/// what makes that walk a one-time cost. `SharedFontCache` is the handle
/// for owners that need the cache on several threads or behind a
/// `&dyn FontMetrics`: cloning it is an `Arc` clone, and every method
/// takes `&self` (a short internal lock), so no per-call directory walk
/// ever happens through a shared handle.
///
/// It implements [`bowser_layout::FontMetrics`], so layout can measure
/// with real glyph advances through the same cache paint resolves and
/// rasterizes with: laid-out line breaks and positions then match the
/// painted glyphs.
///
/// Locking: [`SharedFontCache::lock`] hands out the inner `&mut
/// FontCache` for phases that need it (display-list building resolves
/// faces mutably). Do not hold that guard while calling [`FontMetrics`]
/// methods on the same handle; they lock internally and would deadlock.
#[derive(Clone)]
pub struct SharedFontCache {
    inner: Arc<Mutex<FontCache>>,
}

impl SharedFontCache {
    /// Discover system fonts once and share the resulting cache.
    pub fn new() -> Self {
        Self::from_cache(FontCache::new())
    }

    /// Share an already-built cache (e.g. one a window owns).
    pub fn from_cache(cache: FontCache) -> Self {
        Self {
            inner: Arc::new(Mutex::new(cache)),
        }
    }

    /// Share the cache behind an existing `Arc<Mutex<FontCache>>` —
    /// the window's live cache, not a copy: installs through either
    /// handle (the window's lock, or a `FontFace.load()` through this
    /// handle) are visible through the other. This is the handle
    /// embedders pass to [`bowser_js::share_font_cache`]; `from_cache`
    /// would mint a disconnected cache instead.
    pub fn from_arc(inner: Arc<Mutex<FontCache>>) -> Self {
        Self { inner }
    }

    /// The inner cache, for phases that need `&mut` access (font
    /// resolution during display-list building). See the type docs for
    /// the locking rule.
    pub fn lock(&self) -> std::sync::MutexGuard<'_, FontCache> {
        self.inner.lock().expect("font cache lock poisoned")
    }

    /// Resolve `(family, weight, style)` to a [`FontKey`]; see
    /// [`FontCache::resolve`].
    pub fn resolve(&self, family: &str, weight: u16, style: FontStyle) -> Option<FontKey> {
        self.lock().resolve(family, weight, style)
    }

    /// The default measurement font's key; see [`FontCache::default_key`].
    pub fn default_key(&self) -> Option<FontKey> {
        self.lock().default_key()
    }

    /// Access the decoded bytes for `key`; see [`FontCache::font_data`].
    pub fn font_data(&self, key: FontKey) -> Option<FontData> {
        self.lock().font_data(key)
    }

    /// Number of resolved faces currently cached; see
    /// [`FontCache::font_count`].
    pub fn font_count(&self) -> usize {
        self.lock().font_count()
    }

    /// A `Send + Sync + 'static` font-data hook over the faces resolved
    /// so far; see [`FontCache::resolver`].
    pub fn resolver(&self) -> FontHook {
        self.lock().resolver()
    }
}

impl Default for SharedFontCache {
    fn default() -> Self {
        Self::new()
    }
}

impl From<FontCache> for SharedFontCache {
    fn from(cache: FontCache) -> Self {
        Self::from_cache(cache)
    }
}

impl FontMetrics for SharedFontCache {
    /// Register `family` and return its id (short internal lock).
    fn intern_family(&self, family: &str) -> FontFamilyId {
        self.lock().intern_family(family)
    }

    /// Real glyph advances from the shared cache (short internal lock),
    /// in the `family` face.
    fn advance(&self, text: &str, font_size: f32, family: FontFamilyId) -> f32 {
        self.lock().advance(text, font_size, family)
    }

    /// Typographic line height from the shared cache (short internal
    /// lock), in the `family` face.
    fn line_height(&self, font_size: f32, family: FontFamilyId) -> f32 {
        self.lock().line_height(font_size, family)
    }

    /// Real ascent/descent from the shared cache (short internal lock),
    /// in the `family` face.
    fn ascent_descent(&self, font_size: f32, family: FontFamilyId) -> (f32, f32) {
        self.lock().ascent_descent(font_size, family)
    }

    /// x-height from the shared cache (short internal lock), in the
    /// `family` face.
    fn x_height(&self, font_size: f32, family: FontFamilyId) -> f32 {
        self.lock().x_height(font_size, family)
    }
}

/// Load a face's bytes from its source. `None` for unreadable files;
/// the embedded fallback always loads.
fn face_data(source: &FaceSource) -> Option<FontData> {
    match source {
        FaceSource::File { path, index } => std::fs::read(path).ok().map(|bytes| FontData {
            // `Arc`: the bytes are immutable once loaded and are shared
            // (not copied) by every resolver snapshot, hook invocation,
            // and thread-local cache entry that names this face.
            bytes: Arc::new(bytes),
            face_index: *index as usize,
        }),
        FaceSource::Embedded { bytes, index } => Some(FontData {
            bytes: Arc::new(bytes.to_vec()),
            face_index: *index as usize,
        }),
        FaceSource::Owned { bytes, index } => Some(FontData {
            bytes: Arc::clone(bytes),
            face_index: *index as usize,
        }),
    }
}

/// SFNT format check: TrueType (`0x00010000`, `true`, `typ1`),
/// OpenType (`OTTO`), and collections (`ttcf`) pass; woff (`wOFF`),
/// woff2 (`wOF2`), and anything else fail. This is the format gate
/// behind [`FontCache::install_font_owned`]: WOFF1 never reaches it
/// as `wOFF` because [`decode_webfont`] runs first, and woff2 has no
/// decoder yet, so it is rejected here before `FontRef` parsing.
/// Public so the app loader reuses the one gate instead of carrying
/// a duplicated magic list (the loader's fetch step applies the same
/// check before trying a rule's next `src` entry).
pub fn is_sfnt(bytes: &[u8]) -> bool {
    let Some(magic) = bytes.get(..4) else {
        return false;
    };
    magic == [0u8, 1, 0, 0]
        || magic == b"true"
        || magic == b"typ1"
        || magic == b"OTTO"
        || magic == b"ttcf"
}

/// Maximum bytes a decoded webfont may occupy: the same 32 MiB
/// per-font cap the loader (`FONT_MAX_BYTES`) and the JS bindings
/// (`FONT_DATA_URL_MAX_BYTES`) apply to fetched bytes. The decoder
/// enforces it on the *rebuilt SFNT* size (and on the summed table
/// sizes) before allocating the output, so a compressed bomb cannot
/// expand past it: a WOFF1 file whose `totalSfntSize` or summed
/// uncompressed table sizes exceed this cap is rejected outright,
/// and each table's inflation is additionally capped at its declared
/// `origLength` (see [`decode_woff1`]).
///
/// Budget accounting note (read before touching the loader): the
/// loader's 192 MiB aggregate budget (`WEBFONT_BUDGET_BYTES` in
/// `bowser_app::loader`) counts *decoded* bytes — unique decoded
/// SFNT bytes, deduped by shared `Arc` — because the cache retains
/// the decoded font, not the fetched WOFF. The per-font cap above
/// bounds any single font and that budget bounds their aggregate
/// on the loader path.
/// Remaining gap, deliberately not closed here: the JS
/// `FontFace.load()` install path installs straight into the shared
/// cache with only the per-font 32 MiB cap and no aggregate
/// accounting, so it can retain past the loader budget. A
/// cache-level byte budget covering every install path is a
/// wave-77 item, not this decoder's job.
pub const MAX_DECODED_FONT_BYTES: usize = 32 * 1024 * 1024;

/// Normalize fetched font bytes to SFNT, decoding by magic bytes
/// (no CSS `format()` hint is consulted: the hints are parsed but
/// dropped upstream, and the magic is authoritative).
///
/// - SFNT input passes through unchanged.
/// - WOFF1 (`wOFF`) input is decoded to a rebuilt SFNT by
///   [`decode_woff1`].
/// - Anything else — woff2 (`wOF2`, needs a Brotli decoder plus the
///   WOFF2 transform, explicitly out of scope here), garbage,
///   truncated or malformed WOFF1 — returns `None`.
///
/// This is the hook [`FontCache::install_font_owned`] calls before
/// the [`is_sfnt`] gate, so every installer (loader, navigation, JS)
/// gets WOFF1 for free and dedupe/storage see decoded bytes.
pub fn decode_webfont(bytes: &[u8]) -> Option<Arc<[u8]>> {
    if is_sfnt(bytes) {
        return Some(Arc::from(bytes));
    }
    if bytes.get(..4) == Some(b"wOFF".as_slice()) {
        return decode_woff1(bytes).map(Arc::from);
    }
    None
}

/// One WOFF1 table-directory entry, in file order.
struct WoffTable {
    tag: [u8; 4],
    offset: u32,
    comp_length: u32,
    orig_length: u32,
    checksum: u32,
}

fn be_u16(bytes: &[u8], at: usize) -> Option<u16> {
    bytes
        .get(at..at + 2)
        .map(|w| u16::from_be_bytes([w[0], w[1]]))
}

fn be_u32(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .map(|w| u32::from_be_bytes([w[0], w[1], w[2], w[3]]))
}

/// Decode a WOFF1 file to a rebuilt SFNT font (WOFF 1.0 spec).
///
/// The header (44 bytes) and 20-byte-per-table directory are parsed,
/// each table is inflated (zlib-deflated, or copied verbatim when
/// `compLength == origLength`, which the spec defines as stored, not
/// deflated), and a fresh SFNT directory is rebuilt: table records
/// sorted by tag, offsets recomputed from the header size with each
/// table padded to a 4-byte boundary, and the WOFF `origChecksum`
/// carried into each record. Extended metadata and private data are
/// ignored — they carry no glyph data — but their ranges, when
/// present, must lie inside the file and must not overlap the
/// directory, any table range, or each other.
///
/// Validation (any failure returns `None`, never a partial font):
/// the WOFF length field must not exceed the input (truncation) and
/// must cover the whole directory and every table range (table data
/// may not hide in trailing bytes past it); the reserved header
/// field must be zero; the flavor must be a single-font SFNT flavor
/// (`ttcf` collections are rejected — a WOFF1 directory cannot
/// describe one); the table count is capped at 4095, the largest
/// count whose rebuilt SFNT `searchRange` still fits a u16 (4096
/// tables would need a `searchRange` of 65536, which overflows); every table's
/// compressed range must lie inside the file past the directory (its
/// compressed length may exceed its declared uncompressed length —
/// naive encoders deflate incompressible tables anyway; the output
/// cap below still bounds inflation), tags must be unique,
/// compressed ranges must not overlap, no table may be empty, and
/// the recomputed SFNT size (header + directory +
/// 4-byte-padded tables) must equal `totalSfntSize` exactly and fit
/// under [`MAX_DECODED_FONT_BYTES`]. Each decoded table — stored
/// or deflated — is verified against its `origChecksum` (the SFNT
/// checksum: big-endian u32 sum over the 4-byte-padded table, with
/// the `head` table's `checkSumAdjustment` zeroed per the SFNT
/// spec) before it is emitted, so a stored table with no zlib
/// integrity check of its own cannot smuggle corrupt bytes through.
/// Inflation is bounded twice: the summed `origLength`s are capped
/// before decoding, and each table's decoder is limited to
/// `origLength + 1` bytes with the output required to be exactly
/// `origLength` and the zlib stream required to consume exactly
/// `compLength` bytes, so neither a lying header nor a crafted
/// stream can expand past the cap.
///
/// Deliberately unchecked: `majorVersion`/`minorVersion` (bytes
/// 20-23) mirror the font's own version and the WOFF spec fixes no
/// value for them, so there is nothing sound to enforce; and the
/// metadata block's `metaOrigLength` (its uncompressed length is
/// immaterial once the block is ignored, and a compressed block may
/// legitimately be larger than its source).
fn sfnt_table_checksum(tag: &[u8; 4], data: &[u8]) -> u32 {
    let mut sum: u32 = 0;
    for chunk_start in (0..data.len()).step_by(4) {
        let mut word = [0u8; 4];
        for (j, slot) in word.iter_mut().enumerate() {
            let at = chunk_start + j;
            // SFNT rule: the `head` table's checksum is computed
            // with `checkSumAdjustment` (bytes 8..12) zeroed.
            if tag == b"head" && (8..12).contains(&at) {
                *slot = 0;
            } else {
                *slot = data.get(at).copied().unwrap_or(0);
            }
        }
        sum = sum.wrapping_add(u32::from_be_bytes(word));
    }
    sum
}
fn decode_woff1(bytes: &[u8]) -> Option<Vec<u8>> {
    use std::io::Read as _;
    if bytes.len() < 44 || bytes.get(..4) != Some(b"wOFF".as_slice()) {
        return None;
    }
    let flavor = be_u32(bytes, 4)?;
    // Single-font flavors only: `ttcf` would need a collection
    // directory the WOFF1 table list cannot describe.
    if !(flavor.to_be_bytes() == [0u8, 1, 0, 0]
        || flavor.to_be_bytes() == *b"true"
        || flavor.to_be_bytes() == *b"typ1"
        || flavor.to_be_bytes() == *b"OTTO")
    {
        return None;
    }
    let woff_length = be_u32(bytes, 8)? as usize;
    if woff_length > bytes.len() {
        return None;
    }
    // Reserved field (bytes 14-15) must be zero per the WOFF spec.
    if be_u16(bytes, 14)? != 0 {
        return None;
    }
    let num_tables = be_u16(bytes, 12)? as usize;
    if num_tables == 0 {
        return None;
    }
    // The SFNT directory's `searchRange`/`rangeShift` are u16: the
    // rebuilt directory's `searchRange` is the largest power of two
    // no greater than `num_tables * 16`, so 4096 tables would need
    // a `searchRange` of 65536, which does not fit a u16 (it wraps
    // to 0 in release, stalling the loop below in an infinite loop,
    // and panics on overflow in debug). 4095 is therefore the
    // largest representable count: its `searchRange` is 32768 and
    // its `rangeShift` (4095 * 16 - 32768 = 32752) fits too. (Note
    // `rangeShift` is not the binding constraint: for 4096 tables
    // it would be 0, which fits fine — it is `searchRange` itself
    // that overflows.) No valid SFNT can store a 65536
    // `searchRange` anyway.
    if num_tables > 4095 {
        return None;
    }
    // Extended metadata / private data ranges, validated against
    // the directory and table ranges below (see the overlap check).
    let meta_offset = be_u32(bytes, 24)? as usize;
    let meta_length = be_u32(bytes, 28)? as usize;
    let priv_offset = be_u32(bytes, 36)? as usize;
    let priv_length = be_u32(bytes, 40)? as usize;
    let total_sfnt_size = be_u32(bytes, 16)? as usize;
    if total_sfnt_size > MAX_DECODED_FONT_BYTES {
        return None;
    }
    let dir_end = 44usize.checked_add(num_tables.checked_mul(20)?)?;
    if dir_end > bytes.len() || dir_end > woff_length {
        return None;
    }
    let mut tables: Vec<WoffTable> = Vec::with_capacity(num_tables);
    for i in 0..num_tables {
        let at = 44 + i * 20;
        let tag: [u8; 4] = bytes.get(at..at + 4)?.try_into().ok()?;
        tables.push(WoffTable {
            tag,
            offset: be_u32(bytes, at + 4)?,
            comp_length: be_u32(bytes, at + 8)?,
            orig_length: be_u32(bytes, at + 12)?,
            checksum: be_u32(bytes, at + 16)?,
        });
    }
    // Tags unique, ranges in-bounds past the directory and inside
    // the declared WOFF length, ranges non-overlapping, no empty
    // tables. No `comp_length <= orig_length` check: a deflated
    // incompressible table can be larger than its source (the
    // fixture's `gasp`/`head` tables are), and the per-table output
    // cap below bounds inflation regardless.
    let mut seen_tags: Vec<[u8; 4]> = Vec::with_capacity(num_tables);
    for table in &tables {
        if seen_tags.contains(&table.tag) {
            return None;
        }
        seen_tags.push(table.tag);
        if table.orig_length == 0
            || (table.offset as usize) < dir_end
            || (table.offset as usize).checked_add(table.comp_length as usize)? > woff_length
        {
            return None;
        }
    }
    let mut ranges: Vec<(usize, usize)> = tables
        .iter()
        .map(|t| {
            (
                t.offset as usize,
                t.offset as usize + t.comp_length as usize,
            )
        })
        .collect();
    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            return None;
        }
    }
    // Metadata/private blocks carry no glyph data and are dropped,
    // but a range pointing into the directory, a table, or the
    // other block (or past the declared file length) is malformed
    // and must not be silently accepted. Empty ranges (length 0)
    // are unconstrained: a zero offset means "absent" per spec.
    let overlaps = |a_start: usize, a_end: usize, b_start: usize, b_end: usize| {
        a_start < b_end && b_start < a_end
    };
    for (offset, length) in [(meta_offset, meta_length), (priv_offset, priv_length)] {
        if length == 0 {
            continue;
        }
        let end = offset.checked_add(length)?;
        if offset < dir_end || end > woff_length {
            return None;
        }
        if ranges.iter().any(|&(s, e)| overlaps(offset, end, s, e)) {
            return None;
        }
    }
    if meta_length > 0 && priv_length > 0 {
        let meta_end = meta_offset.checked_add(meta_length)?;
        let priv_end = priv_offset.checked_add(priv_length)?;
        if overlaps(meta_offset, meta_end, priv_offset, priv_end) {
            return None;
        }
    }
    // Recomputed SFNT size must match the header exactly.
    let padded_tables: usize = tables.iter().try_fold(0usize, |acc, t| {
        acc.checked_add((t.orig_length as usize).checked_add(3)? & !3)
    })?;
    let sfnt_size = 12usize
        .checked_add(num_tables.checked_mul(16)?)?
        .checked_add(padded_tables)?;
    if sfnt_size != total_sfnt_size || sfnt_size > MAX_DECODED_FONT_BYTES {
        return None;
    }
    // SFNT directory, sorted by tag (the SFNT spec's record order).
    let mut sorted: Vec<&WoffTable> = tables.iter().collect();
    sorted.sort_by_key(|t| t.tag);
    let mut search_range: u16 = 16;
    while (search_range as usize) * 2 <= num_tables * 16 {
        search_range *= 2;
    }
    let entry_selector = (search_range / 16).ilog2() as u16;
    let range_shift = (num_tables * 16) as u16 - search_range;
    let mut out: Vec<u8> = Vec::with_capacity(sfnt_size);
    out.extend_from_slice(&flavor.to_be_bytes());
    out.extend_from_slice(&(num_tables as u16).to_be_bytes());
    out.extend_from_slice(&search_range.to_be_bytes());
    out.extend_from_slice(&entry_selector.to_be_bytes());
    out.extend_from_slice(&range_shift.to_be_bytes());
    let mut cursor = 12 + num_tables * 16;
    for table in &sorted {
        out.extend_from_slice(&table.tag);
        out.extend_from_slice(&table.checksum.to_be_bytes());
        out.extend_from_slice(&(cursor as u32).to_be_bytes());
        out.extend_from_slice(&table.orig_length.to_be_bytes());
        cursor += (table.orig_length as usize + 3) & !3;
    }
    for table in &sorted {
        let data =
            bytes.get(table.offset as usize..table.offset as usize + table.comp_length as usize)?;
        let decoded: std::borrow::Cow<'_, [u8]> = if table.comp_length == table.orig_length {
            // Stored, not deflated (WOFF 1.0 spec).
            std::borrow::Cow::Borrowed(data)
        } else {
            let mut decoder = flate2::read::ZlibDecoder::new(data);
            let mut buf: Vec<u8> = Vec::with_capacity(table.orig_length as usize);
            // Borrowed `take`: the decoder itself stays alive so its
            // `total_in` can be checked below.
            (&mut decoder)
                .take(table.orig_length as u64 + 1)
                .read_to_end(&mut buf)
                .ok()?;
            if buf.len() != table.orig_length as usize
                || decoder.total_in() != table.comp_length as u64
            {
                return None;
            }
            std::borrow::Cow::Owned(buf)
        };
        // Integrity check for both paths: the deflated path already
        // gets zlib's Adler-32 plus exact-length checks, but the
        // stored path has no checksum of its own, so the SFNT
        // checksum is the only thing standing between a flipped
        // stored byte and a corrupt installed font.
        if sfnt_table_checksum(&table.tag, &decoded) != table.checksum {
            return None;
        }
        out.extend_from_slice(&decoded);
        while !out.len().is_multiple_of(4) {
            out.push(0);
        }
    }
    if out.len() != sfnt_size {
        return None;
    }
    Some(out)
}

/// Decoded font face owned by the `FontCache`.
///
/// The byte buffer is the stable backing store; callers that need swash
/// access borrow a [`swash::FontRef`] via [`FontData::font_ref`] for as
/// long as they hold the `&FontData` borrow. This is the accessor the
/// glyph/shaping worker builds on.
///
/// The bytes are `Arc`-shared: cloning a `FontData` (resolver snapshots,
/// backend hooks, thread-local caches) bumps a refcount instead of
/// copying the whole font file.
#[derive(Clone, Debug)]
pub struct FontData {
    bytes: Arc<Vec<u8>>,
    face_index: usize,
}

impl FontData {
    /// Raw font file bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The shared font bytes.
    ///
    /// Crate-internal: lets the shaping cache (see `crate::text`) key
    /// entries by font identity — the `Arc` allocation itself — without
    /// copying any bytes.
    pub(crate) fn bytes_arc(&self) -> &Arc<Vec<u8>> {
        &self.bytes
    }

    /// Index of this face inside a font collection (0 for single fonts).
    pub fn face_index(&self) -> usize {
        self.face_index
    }

    /// Borrow a swash font reference for this face. `None` if the bytes do
    /// not parse (should not happen for faces that came out of discovery).
    pub fn font_ref(&self) -> Option<swash::FontRef<'_>> {
        swash::FontRef::from_index(&self.bytes, self.face_index)
    }
}

/// Lowercase/trim/collapse-whitespace the family name so equivalent
/// requests hit the cache.
fn normalize_family(family: &str) -> String {
    family
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

/// [`normalize_family`] without the allocation when the input is
/// already normalized: all-ASCII, no leading/trailing whitespace, no
/// ASCII uppercase, and interior whitespace only as single `0x20` spaces
/// (a tab or a double space still needs collapsing). For such inputs the
/// borrowed string is exactly what [`normalize_family`] would produce
/// (whitespace collapsing and ASCII-lowering are both no-ops), so
/// [`FontCache::resolve`] can probe its query cache with the caller's
/// `&str` directly. Anything else (mixed case, stray whitespace,
/// non-ASCII) takes the owned path.
fn normalize_family_cow(family: &str) -> Cow<'_, str> {
    // `str::split_whitespace` (used by `normalize_family`) splits on
    // Unicode White_Space, which for ASCII also matches `\x0b`
    // (vertical tab) — a byte `u8::is_ascii_whitespace` misses. It is
    // the only ASCII byte where the two whitespace notions differ, and
    // the fast path must agree with `split_whitespace` exactly, so the
    // predicate counts it as whitespace too.
    fn is_ws(b: u8) -> bool {
        b.is_ascii_whitespace() || b == b'\x0b'
    }
    let bytes = family.as_bytes();
    let already_normalized = bytes.is_ascii()
        && !bytes.first().is_some_and(|b| is_ws(*b))
        && !bytes.last().is_some_and(|b| is_ws(*b))
        && !bytes.iter().any(|b| b.is_ascii_uppercase())
        && !bytes.iter().any(|b| is_ws(*b) && *b != b' ')
        && !bytes.windows(2).any(|w| w[0] == b' ' && w[1] == b' ');
    if already_normalized {
        Cow::Borrowed(family)
    } else {
        Cow::Owned(normalize_family(family))
    }
}

/// CSS generic family names; the candidate lists below stand in for
/// fontconfig's generic-alias configuration (a heuristic: well-known
/// family names in preference order).
#[derive(Clone, Copy)]
enum GenericFamily {
    Serif,
    SansSerif,
    Monospace,
    Cursive,
    Fantasy,
}

fn generic_family(name: &str) -> Option<GenericFamily> {
    match name {
        "serif" => Some(GenericFamily::Serif),
        "sans-serif" => Some(GenericFamily::SansSerif),
        "monospace" => Some(GenericFamily::Monospace),
        "cursive" => Some(GenericFamily::Cursive),
        "fantasy" => Some(GenericFamily::Fantasy),
        _ => None,
    }
}

/// Well-known family names per generic, in preference order. All entries
/// are already normalized (lowercase).
fn generic_candidates(generic: GenericFamily) -> &'static [&'static str] {
    match generic {
        GenericFamily::Serif => &[
            "noto serif",
            "dejavu serif",
            "liberation serif",
            "times new roman",
            "times",
            "georgia",
            "garamond",
            "droid serif",
            "freeserif",
            "pt serif",
        ],
        GenericFamily::SansSerif => &[
            "noto sans",
            "dejavu sans",
            "liberation sans",
            "arial",
            "helvetica",
            "verdana",
            "tahoma",
            "droid sans",
            "freesans",
            "ubuntu",
            "pt sans",
        ],
        GenericFamily::Monospace => &[
            "noto sans mono",
            "dejavu sans mono",
            "liberation mono",
            "courier new",
            "courier",
            "droid sans mono",
            "freemono",
            "pt mono",
        ],
        GenericFamily::Cursive => &[
            "comic sans ms",
            "chalkboard",
            "urw chancery l",
            "snell roundhand",
        ],
        GenericFamily::Fantasy => &["impact", "luminari", "papyrus"],
    }
}

/// CSS Fonts 4 weight-matching order over the available weights
/// (ascending, deduped): exact first, then per the 400/500/below/above
/// rules. Returns the available weights in preference order.
fn weight_preference_order(desired: u16, available: &[u16]) -> Vec<u16> {
    let mut order = Vec::with_capacity(available.len());
    let mut push = |w: u16| {
        if !order.contains(&w) {
            order.push(w);
        }
    };
    if available.contains(&desired) {
        push(desired);
    }
    let below: Vec<u16> = available.iter().copied().filter(|&w| w < desired).collect();
    let above: Vec<u16> = available.iter().copied().filter(|&w| w > desired).collect();
    match desired {
        400 => {
            // 500, then lighter descending, then heavier ascending.
            if available.contains(&500) {
                push(500);
            }
            for &w in below.iter().rev() {
                push(w);
            }
            for &w in &above {
                push(w);
            }
        }
        500 => {
            // 400, then lighter descending, then heavier ascending.
            if available.contains(&400) {
                push(400);
            }
            for &w in below.iter().rev() {
                push(w);
            }
            for &w in &above {
                push(w);
            }
        }
        _ if desired < 400 => {
            for &w in below.iter().rev() {
                push(w);
            }
            for &w in &above {
                push(w);
            }
        }
        _ => {
            for &w in &above {
                push(w);
            }
            for &w in below.iter().rev() {
                push(w);
            }
        }
    }
    order
}

/// Style mismatch penalty: exact 0, italic/oblique cross 1, slanted vs
/// upright 2. (No synthesis: an italic request on an upright-only family
/// still resolves the upright face.)
fn style_penalty(want: FontStyle, have: FontStyle) -> u8 {
    if want == have {
        0
    } else if matches!(
        (want, have),
        (FontStyle::Italic, FontStyle::Oblique) | (FontStyle::Oblique, FontStyle::Italic)
    ) {
        1
    } else {
        2
    }
}

/// Platform font directories (Linux-first per crate policy; macOS and
/// Windows covered for completeness).
fn font_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    #[cfg(target_os = "linux")]
    {
        dirs.push(PathBuf::from("/usr/share/fonts"));
        dirs.push(PathBuf::from("/usr/local/share/fonts"));
        if let Ok(home) = std::env::var("HOME") {
            dirs.push(PathBuf::from(format!("{home}/.fonts")));
            dirs.push(PathBuf::from(format!("{home}/.local/share/fonts")));
        }
    }
    #[cfg(target_os = "macos")]
    {
        dirs.push(PathBuf::from("/System/Library/Fonts"));
        dirs.push(PathBuf::from("/Library/Fonts"));
        if let Ok(home) = std::env::var("HOME") {
            dirs.push(PathBuf::from(format!("{home}/Library/Fonts")));
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(root) = std::env::var("SystemRoot") {
            dirs.push(PathBuf::from(format!("{root}\\Fonts")));
        }
    }
    dirs
}

/// Maximum directory recursion depth for the font walk.
const MAX_WALK_DEPTH: u8 = 8;

/// Walk the platform font directories and index every font file's
/// family/weight/style. Symlinked directories are followed; the
/// canonicalized-dir set guards against cycles. Faces are sorted by path
/// so tie-breaking in matching is deterministic (directory iteration
/// order is unspecified).
fn discover_system_faces() -> Vec<Face> {
    let mut faces = Vec::new();
    let mut seen_dirs: FnvHashSet<PathBuf> = FnvHashSet::default();
    for dir in font_dirs() {
        visit_dir(&dir, 0, &mut seen_dirs, &mut faces);
    }
    faces.sort_by(|a, b| {
        let path_a = match &a.source {
            FaceSource::File { path, .. } => path,
            FaceSource::Embedded { .. } | FaceSource::Owned { .. } => Path::new(""),
        };
        let path_b = match &b.source {
            FaceSource::File { path, .. } => path,
            FaceSource::Embedded { .. } | FaceSource::Owned { .. } => Path::new(""),
        };
        path_a.cmp(path_b)
    });
    faces
}

fn visit_dir(dir: &Path, depth: u8, seen_dirs: &mut FnvHashSet<PathBuf>, faces: &mut Vec<Face>) {
    if depth > MAX_WALK_DEPTH {
        return;
    }
    // Canonicalize: dedupes repeated dirs and resolves symlinks so the
    // `seen_dirs` set below also breaks symlink cycles.
    let Ok(canonical) = dir.canonicalize() else {
        return;
    };
    if !seen_dirs.insert(canonical) {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        // `metadata` (not `symlink_metadata`): follow symlinks so
        // symlinked font files and directories are indexed.
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        if meta.is_dir() {
            visit_dir(&path, depth + 1, seen_dirs, faces);
        } else if meta.is_file() && is_font_file(&path) {
            index_font_file(&path, faces);
        }
    }
}

/// Font file extensions we index (case-insensitive).
fn is_font_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("ttf") | Some("otf") | Some("ttc") | Some("otc")
    )
}

/// Index every face in one font file (collections included).
fn index_font_file(path: &Path, faces: &mut Vec<Face>) {
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };
    for (index, font) in FontRef::fonts(&bytes).enumerate() {
        let Ok(font) = font else { continue };
        let families = family_names(&font);
        if families.is_empty() {
            continue;
        }
        let (weight, width, style) = face_attributes(&font);
        faces.push(Face {
            source: FaceSource::File {
                path: path.to_path_buf(),
                index: index as u32,
            },
            families,
            weight,
            width,
            style,
        });
    }
}

/// The embedded fallback face, indexed like a system face.
fn fallback_face() -> Face {
    let font = FontRef::new(FALLBACK_FONT_BYTES).expect("embedded fallback font must parse");
    let mut families = family_names(&font);
    if families.is_empty() {
        families.push("bowser-fallback".to_string());
    }
    let (weight, width, style) = face_attributes(&font);
    Face {
        source: FaceSource::Embedded {
            bytes: FALLBACK_FONT_BYTES,
            index: 0,
        },
        families,
        weight,
        width,
        style,
    }
}

/// Normalized family names for one face: typographic family (name ID 16)
/// plus legacy family (name ID 1), English preferred.
fn family_names(font: &FontRef) -> Vec<String> {
    let Ok(name) = font.name() else {
        return Vec::new();
    };
    let storage = name.string_data();
    let mut out = Vec::new();
    for want in [NameId::TYPOGRAPHIC_FAMILY_NAME, NameId::FAMILY_NAME] {
        // Best language per name ID: en-US (0x0409) first, then
        // language 0, then anything that decodes.
        let mut best: Option<(u8, String)> = None;
        for record in name.name_record() {
            if record.name_id() != want {
                continue;
            }
            let Ok(string) = record.string(storage) else {
                continue;
            };
            let text = normalize_family(&string.chars().collect::<String>());
            if text.is_empty() {
                continue;
            }
            let rank = match record.language_id() {
                0x0409 => 0,
                0 => 1,
                _ => 2,
            };
            let replace = match &best {
                None => true,
                Some((best_rank, _)) => rank < *best_rank,
            };
            if replace {
                best = Some((rank, text));
                if rank == 0 {
                    break;
                }
            }
        }
        if let Some((_, text)) = best
            && !out.contains(&text)
        {
            out.push(text);
        }
    }
    out
}

/// `(weight, width, style)` for one face: OS/2 first, head `macStyle`
/// fallback for weight/style, normal width when OS/2 is missing.
fn face_attributes(font: &FontRef) -> (u16, u16, FontStyle) {
    use read_fonts::tables::{head::MacStyle, os2::SelectionFlags};
    if let Ok(os2) = font.os2() {
        let weight = os2.us_weight_class().clamp(1, 1000);
        let width = os2.us_width_class().clamp(1, 9);
        let selection = os2.fs_selection();
        let style = if selection.contains(SelectionFlags::ITALIC) {
            FontStyle::Italic
        } else if selection.contains(SelectionFlags::OBLIQUE) {
            FontStyle::Oblique
        } else {
            FontStyle::Normal
        };
        return (weight, width, style);
    }
    if let Ok(head) = font.head() {
        let mac = head.mac_style();
        let style = if mac.contains(MacStyle::ITALIC) {
            FontStyle::Italic
        } else {
            FontStyle::Normal
        };
        let weight = if mac.contains(MacStyle::BOLD) {
            700
        } else {
            400
        };
        return (weight, 5, style);
    }
    (400, 5, FontStyle::Normal)
}

/// Advance fallback when no font resolved: matches `FixedFontMetrics`.
#[allow(dead_code)]
const FALLBACK_ADVANCE_PER_CHAR: f32 = 0.6;
/// Line-height fallback when no font resolved: matches `FixedFontMetrics`.
#[allow(dead_code)]
const FALLBACK_LINE_HEIGHT: f32 = 1.2;

impl FontCache {
    /// Shaped advance of `text` at `font_size` in the face `key`
    /// resolves to, via paint's [`shape_text`] (kerning, ligature
    /// substitution) — the same widths the backends paint.
    ///
    /// Results are cached per thread by the shaper: repeat measurements
    /// of the same `(font, size, text)` hit the cache instead of
    /// re-running swash.
    fn shaped_advance(&self, key: FontKey, text: &str, font_size: f32) -> f32 {
        let Some(data) = self.font_data(key) else {
            return text.chars().count() as f32 * FALLBACK_ADVANCE_PER_CHAR * font_size;
        };
        shape_text(&data, text, font_size).width
    }

    /// Typographic line height (ascent + descent + leading) at
    /// `font_size` for the face `key` resolves to.
    ///
    /// The font metrics come from the cached [`MeasureSetup`] (see
    /// [`FontCache::measure_setup`]): one proxy materialization per call,
    /// no table walks.
    fn typo_line_height(&self, key: FontKey, font_size: f32) -> f32 {
        let Some(data) = self.font_data(key) else {
            return FALLBACK_LINE_HEIGHT * font_size;
        };
        let Some(font) = data.font_ref() else {
            return FALLBACK_LINE_HEIGHT * font_size;
        };
        let setup = self.measure_setup(key, &font, font_size);
        let coords: &[swash::NormalizedCoord] = &[];
        let m = setup
            .metrics
            .materialize_metrics(&font, coords)
            .scale(font_size);
        // swash 0.2 reports `descent` as a positive distance from the
        // baseline to the bottom of the alignment box (it negates the
        // SFNT descender), so the typographic line height adds it.
        m.ascent + m.descent + m.leading
    }
}

impl FontMetrics for FontCache {
    /// Register `family` for later id-keyed measurement: the normalized
    /// name is retained so [`resolve_family_id`](FontCache::resolve_family_id)
    /// can turn the id back into a face.
    fn intern_family(&self, family: &str) -> FontFamilyId {
        let id = FontFamilyId::of(family);
        self.family_names
            .borrow_mut()
            .entry(id)
            .or_insert_with(|| normalize_family(family));
        id
    }

    /// Horizontal advance of `text` at `font_size`, shaped with paint's
    /// own shaper in the `family` face: kerning and ligature substitution
    /// included, so this is the width the backends will paint.
    fn advance(&self, text: &str, font_size: f32, family: FontFamilyId) -> f32 {
        let Some(key) = self.resolve_family_id(family) else {
            return text.chars().count() as f32 * FALLBACK_ADVANCE_PER_CHAR * font_size;
        };
        self.shaped_advance(key, text, font_size)
    }

    /// Typographic line height (ascent + descent + leading) of the
    /// `family` face at `font_size`.
    fn line_height(&self, font_size: f32, family: FontFamilyId) -> f32 {
        let Some(key) = self.resolve_family_id(family) else {
            return FALLBACK_LINE_HEIGHT * font_size;
        };
        self.typo_line_height(key, font_size)
    }

    /// Real ascent/descent magnitudes of the `family` face at
    /// `font_size`; falls back to the 0.8/0.2 probe split when the face
    /// has no usable metrics.
    fn ascent_descent(&self, font_size: f32, family: FontFamilyId) -> (f32, f32) {
        let extents = self
            .resolve_family_id(family)
            .and_then(|key| self.face_ascent_descent(key, font_size));
        match extents {
            Some((a, d)) => (a, d),
            None => {
                let lh = self.line_height(font_size, family);
                (0.8 * lh, 0.2 * lh)
            }
        }
    }

    /// x-height of the `family` face from swash metrics, falling back
    /// to 0.5em (css-inline-3 suggested fallback) when the face has no
    /// usable value.
    fn x_height(&self, font_size: f32, family: FontFamilyId) -> f32 {
        let xh = self
            .resolve_family_id(family)
            .and_then(|key| self.face_x_height(key, font_size));
        xh.unwrap_or(0.5 * font_size)
    }
}

/// Production [`bowser_layout::FontMetrics`] over a borrowed [`FontCache`].
///
/// This is what `bowser_app`'s `refresh` passes to layout: the window's
/// own font cache, so layout measures text with the same shaped advances
/// paint will rasterize. `advance` shapes with paint's [`shape_text`] —
/// kerning and ligature substitution included — in the text run's
/// [`FontFamilyId`] face; `line_height` is that face's typographic
/// ascent + descent + leading, and `ascent_descent` its real
/// ascent/descent magnitudes.
///
/// The [`bowser_layout::FontMetrics`] contract carries no weight/style
/// (runs always measure at 400/normal) — a known residual mismatch. The
/// per-key helpers below are the building blocks for closing it once
/// layout can thread a run's resolved [`FontKey`] through.
#[derive(Clone, Copy)]
pub struct SwashFontMetrics<'a> {
    fonts: &'a FontCache,
}

impl<'a> SwashFontMetrics<'a> {
    /// Measure through `fonts`: the same cache paint resolves and
    /// rasterizes with.
    pub fn new(fonts: &'a FontCache) -> Self {
        Self { fonts }
    }

    /// The default measurement face's key.
    pub fn default_key(&self) -> Option<FontKey> {
        self.fonts.default_key()
    }

    /// Shaped advance of `text` at `font_size` in the face `key`
    /// resolves to, via paint's [`shape_text`].
    pub fn advance_for_key(&self, key: FontKey, text: &str, font_size: f32) -> f32 {
        self.fonts.shaped_advance(key, text, font_size)
    }

    /// Typographic line height (ascent + descent + leading) at
    /// `font_size` for the face `key` resolves to.
    pub fn line_height_for_key(&self, key: FontKey, font_size: f32) -> f32 {
        self.fonts.typo_line_height(key, font_size)
    }

    /// `(ascent, descent, leading)` in px at `font_size` for the face
    /// `key` resolves to. `None` when the face's bytes do not parse.
    pub fn extents_for_key(&self, key: FontKey, font_size: f32) -> Option<(f32, f32, f32)> {
        let data = self.fonts.font_data(key)?;
        let font = data.font_ref()?;
        let setup = self.fonts.measure_setup(key, &font, font_size);
        let coords: &[swash::NormalizedCoord] = &[];
        let m = setup
            .metrics
            .materialize_metrics(&font, coords)
            .scale(font_size);
        Some((m.ascent, m.descent, m.leading))
    }
}

impl FontMetrics for SwashFontMetrics<'_> {
    /// Register `family` on the borrowed cache; see
    /// [`FontMetrics::intern_family`].
    fn intern_family(&self, family: &str) -> FontFamilyId {
        self.fonts.intern_family(family)
    }

    /// Shaped advance in the `family` face — the widths paint uses for
    /// runs in that family.
    fn advance(&self, text: &str, font_size: f32, family: FontFamilyId) -> f32 {
        self.fonts.advance(text, font_size, family)
    }

    /// Typographic line height of the `family` face.
    fn line_height(&self, font_size: f32, family: FontFamilyId) -> f32 {
        self.fonts.line_height(font_size, family)
    }

    /// Real ascent/descent of the `family` face.
    fn ascent_descent(&self, font_size: f32, family: FontFamilyId) -> (f32, f32) {
        self.fonts.ascent_descent(font_size, family)
    }

    /// x-height of the `family` face.
    fn x_height(&self, font_size: f32, family: FontFamilyId) -> f32 {
        self.fonts.x_height(font_size, family)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_font_is_embedded_and_parses() {
        // The fixture the module docs promise: OFL Cousine subset.
        assert!(
            !FALLBACK_FONT_BYTES.is_empty(),
            "fallback font must be embedded"
        );
        let font =
            FontRef::new(FALLBACK_FONT_BYTES).expect("fallback font bytes must parse as a font");
        let families = family_names(&font);
        assert!(
            families.iter().any(|n| n == "cousine"),
            "fallback family should be Cousine, got {families:?}"
        );
        let (weight, width, style) = face_attributes(&font);
        assert_eq!(weight, 400);
        assert_eq!(width, 5, "fallback must be normal width");
        assert_eq!(style, FontStyle::Normal);
        // The subset must cover ASCII: every printable char maps.
        let swash_font =
            swash::FontRef::from_index(FALLBACK_FONT_BYTES, 0).expect("fallback must parse");
        let charmap = swash_font.charmap();
        for c in '!'..='~' {
            assert_ne!(charmap.map(c), 0, "fallback must cover {c:?}");
        }
    }

    #[test]
    fn fallback_face_resolves_by_name() {
        // Even with no system fonts, "cousine" (the fallback's own
        // family) resolves to the embedded face.
        let mut cache = FontCache::new();
        cache.faces.clear();
        let key = cache
            .resolve("cousine", 400, FontStyle::Normal)
            .expect("fallback must resolve");
        let data = cache.font_data(key).expect("fallback key has data");
        assert_eq!(data.bytes(), FALLBACK_FONT_BYTES);
    }

    #[test]
    fn install_test_fonts_registers_ahem() {
        let mut cache = FontCache::new();
        cache.faces.clear();
        cache.install_test_fonts();
        let key = cache
            .resolve("Ahem", 400, FontStyle::Normal)
            .expect("ahem must resolve");
        let data = cache.font_data(key).expect("ahem key has data");
        assert_eq!(data.bytes(), AHEM_FONT_BYTES);
        // CSS family matching is case-insensitive.
        assert_eq!(cache.resolve("ahem", 400, FontStyle::Normal), Some(key));
    }

    #[test]
    fn install_font_rejects_garbage() {
        let mut cache = FontCache::new();
        assert!(!cache.install_font(b"definitely not a font"));
    }

    #[test]
    fn install_font_owned_registers_fetched_bytes() {
        // Fetched fonts arrive as an owned Vec<u8>, not a 'static
        // slice: the owned install path must register them so a
        // webfont family measures with its own advances.
        let mut cache = FontCache::new();
        cache.faces.clear();
        let owned: std::sync::Arc<[u8]> = std::sync::Arc::from(AHEM_FONT_BYTES.to_vec());
        assert!(cache.install_font_owned(owned));
        let ahem = cache.intern_family("Ahem");
        assert!((cache.advance("X", 50.0, ahem) - 50.0).abs() < 0.01);
    }

    #[test]
    fn install_font_owned_as_resolves_renamed_family() {
        // w75f1: @font-face family Brand, src Ahem.ttf — the rule's
        // family must resolve to the webfont even though the file's
        // internal name is Ahem, not Brand.
        let mut cache = FontCache::new();
        cache.faces.clear();
        let owned: std::sync::Arc<[u8]> = std::sync::Arc::from(AHEM_FONT_BYTES.to_vec());
        assert!(cache.install_font_owned_as(owned, "Brand", 400, FontStyle::Normal));
        let brand = cache.intern_family("Brand");
        assert!(
            (cache.advance("X", 50.0, brand) - 50.0).abs() < 0.01,
            "Brand must measure as Ahem (1em), got {}",
            cache.advance("X", 50.0, brand)
        );
    }

    #[test]
    fn invalidate_families_evicts_generic_queries() {
        // w75f1: a cached `serif` resolution must not survive installing
        // a face under a name `serif` resolves through ("noto serif"
        // is the first generic candidate).
        let mut cache = FontCache::new();
        cache.faces.clear();
        let before = cache
            .resolve("serif", 400, FontStyle::Normal)
            .expect("serif must resolve to the fallback");
        let owned: std::sync::Arc<[u8]> = std::sync::Arc::from(AHEM_FONT_BYTES.to_vec());
        assert!(cache.install_font_owned_as(owned, "Noto Serif", 400, FontStyle::Normal));
        let after = cache
            .resolve("serif", 400, FontStyle::Normal)
            .expect("serif must still resolve");
        assert_ne!(before, after, "cached serif resolution must be evicted");
        let data = cache.font_data(after).expect("serif key has data");
        assert_eq!(
            data.bytes(),
            AHEM_FONT_BYTES,
            "serif must now resolve to the webfont"
        );
    }

    #[test]
    fn shared_font_cache_from_arc_shares_live_cache() {
        // w75f1: `from_arc` must wrap the caller's Mutex, not copy it:
        // an install through the Arc is visible through the shared
        // handle, and vice versa.
        let arc = std::sync::Arc::new(std::sync::Mutex::new(FontCache::new()));
        let shared = SharedFontCache::from_arc(std::sync::Arc::clone(&arc));
        {
            let mut guard = arc.lock().expect("lock");
            guard.faces.clear();
            guard.install_test_fonts();
        }
        let key = shared
            .resolve("ahem", 400, FontStyle::Normal)
            .expect("ahem must resolve through the shared handle");
        let data = shared.font_data(key).expect("ahem key has data");
        assert_eq!(data.bytes(), AHEM_FONT_BYTES);
    }

    #[test]
    fn install_font_owned_accepts_woff1_and_rejects_garbage() {
        let mut cache = FontCache::new();
        // A real WOFF1 file (Ahem.ttf repacked, see the fixture
        // note below) decodes to SFNT and installs.
        let woff: std::sync::Arc<[u8]> = std::sync::Arc::from(WOFF_AHEM_BYTES.to_vec());
        assert!(cache.install_font_owned(woff));
        // A truncated WOFF1 header, a woff2 header, and garbage are
        // still rejected: woff2 has no decoder in this slice.
        let truncated: std::sync::Arc<[u8]> =
            std::sync::Arc::from(b"wOFF\x00\x01\x00\x00not a font".to_vec());
        assert!(!cache.install_font_owned(truncated));
        let woff2: std::sync::Arc<[u8]> =
            std::sync::Arc::from(b"wOF2\x00\x01\x00\x00not a font".to_vec());
        assert!(!cache.install_font_owned(woff2));
        let garbage: std::sync::Arc<[u8]> = std::sync::Arc::from(b"definitely not a font".to_vec());
        assert!(!cache.install_font_owned(garbage));
    }

    #[test]
    fn decode_webfont_round_trips_woff1_to_source_ttf() {
        // Fixture: Ahem.ttf repacked as WOFF1 with zlib-compressed
        // tables (generated by tools/make_woff.py). The
        // generator deflates every table, so `gasp`/`head` arrive
        // with `compLength > origLength` — the decoder must inflate
        // them anyway, not reject them. The rebuilt SFNT must be
        // byte-identical to the source TTF:
        // Ahem's directory is already tag-sorted with compact
        // 4-byte-aligned offsets, so the rebuild reproduces it.
        let decoded = decode_webfont(WOFF_AHEM_BYTES).expect("woff1 must decode");
        assert!(is_sfnt(&decoded));
        assert_eq!(
            decoded.as_ref(),
            AHEM_FONT_BYTES,
            "decoded WOFF1 must rebuild the source SFNT exactly"
        );
        // SFNT input passes through unchanged; woff2 is rejected.
        assert_eq!(
            decode_webfont(AHEM_FONT_BYTES)
                .expect("sfnt must pass through")
                .as_ref(),
            AHEM_FONT_BYTES
        );
        assert!(decode_webfont(b"wOF2\x00\x01\x00\x00not a font").is_none());
    }

    #[test]
    fn decode_webfont_handles_stored_tables() {
        // Second fixture: same Ahem repack, but every table stored
        // verbatim (`compLength == origLength`) instead of deflated.
        // The stored path must rebuild the identical SFNT.
        let stored = include_bytes!("../tests/fixtures/Ahem-stored.woff");
        let decoded = decode_webfont(stored).expect("stored woff1 must decode");
        assert_eq!(decoded.as_ref(), AHEM_FONT_BYTES);
    }

    #[test]
    fn decode_webfont_rejects_malformed_woff1() {
        // Corrupt the compressed `glyf` table bytes: inflation must
        // fail (or the stream-length check must trip), never yield a
        // partial font.
        let mut corrupt = WOFF_AHEM_BYTES.to_vec();
        let glyf_at = corrupt
            .windows(4)
            .position(|w| w == b"glyf")
            .expect("glyf record must exist");
        let offset = u32::from_be_bytes(
            corrupt[glyf_at + 4..glyf_at + 8]
                .try_into()
                .expect("offset field"),
        ) as usize;
        corrupt[offset] ^= 0xff;
        corrupt[offset + 1] ^= 0xff;
        assert!(decode_webfont(&corrupt).is_none());
        // Lie about the total SFNT size: the recomputed-size check
        // must reject it even though every table still decodes.
        let mut lying = WOFF_AHEM_BYTES.to_vec();
        lying[16..20].copy_from_slice(&999_999u32.to_be_bytes());
        assert!(decode_webfont(&lying).is_none());
        // Claim a decoded size over the 32 MiB cap.
        let mut huge = WOFF_AHEM_BYTES.to_vec();
        huge[16..20].copy_from_slice(&(MAX_DECODED_FONT_BYTES as u32 + 1).to_be_bytes());
        assert!(decode_webfont(&huge).is_none());
    }

    fn set_u16(buf: &mut [u8], at: usize, v: u16) {
        buf[at..at + 2].copy_from_slice(&v.to_be_bytes());
    }

    fn set_u32(buf: &mut [u8], at: usize, v: u32) {
        buf[at..at + 4].copy_from_slice(&v.to_be_bytes());
    }

    /// (record offset, table offset, comp length, orig length) for a
    /// directory entry found by tag in a WOFF fixture.
    fn woff_record(buf: &[u8], tag: &[u8; 4]) -> (usize, usize, usize, usize) {
        let num = u16::from_be_bytes(buf[12..14].try_into().unwrap()) as usize;
        for i in 0..num {
            let at = 44 + i * 20;
            if &buf[at..at + 4] == tag {
                let offset = u32::from_be_bytes(buf[at + 4..at + 8].try_into().unwrap()) as usize;
                let comp = u32::from_be_bytes(buf[at + 8..at + 12].try_into().unwrap()) as usize;
                let orig = u32::from_be_bytes(buf[at + 12..at + 16].try_into().unwrap()) as usize;
                return (at, offset, comp, orig);
            }
        }
        panic!("tag {tag:?} not in directory");
    }

    /// Minimal one-table WOFF1: `comp` is the table's stored bytes
    /// (stored when `comp.len() == orig_len`, otherwise a zlib
    /// stream), `checksum` its declared `origChecksum`, and
    /// `meta`/`priv_` optional (offset, length) blocks placed after
    /// the table data with zero fill.
    fn build_single_table_woff(
        tag: &[u8; 4],
        comp: &[u8],
        orig_len: u32,
        checksum: u32,
        meta: Option<(usize, usize)>,
        priv_: Option<(usize, usize)>,
    ) -> Vec<u8> {
        let total_sfnt = 12 + 16 + ((orig_len as usize + 3) & !3);
        let mut out = vec![0u8; 44 + 20 + comp.len()];
        out[..4].copy_from_slice(b"wOFF");
        set_u32(&mut out, 4, 0x0001_0000);
        set_u16(&mut out, 12, 1);
        set_u32(&mut out, 16, total_sfnt as u32);
        set_u16(&mut out, 20, 1);
        if let Some((offset, length)) = meta {
            set_u32(&mut out, 24, offset as u32);
            set_u32(&mut out, 28, length as u32);
            set_u32(&mut out, 32, length as u32);
        }
        if let Some((offset, length)) = priv_ {
            set_u32(&mut out, 36, offset as u32);
            set_u32(&mut out, 40, length as u32);
        }
        out[44..48].copy_from_slice(tag);
        set_u32(&mut out, 48, 64);
        set_u32(&mut out, 52, comp.len() as u32);
        set_u32(&mut out, 56, orig_len);
        set_u32(&mut out, 60, checksum);
        out[64..64 + comp.len()].copy_from_slice(comp);
        // Extend for metadata/private blocks (zero fill).
        let mut end = 64 + comp.len();
        for (offset, length) in [meta, priv_].into_iter().flatten() {
            end = end.max(offset + length);
        }
        out.resize(end, 0);
        let woff_len = out.len() as u32;
        set_u32(&mut out, 8, woff_len);
        out
    }

    fn zlib(payload: &[u8]) -> Vec<u8> {
        use std::io::Write as _;
        let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(9));
        encoder.write_all(payload).unwrap();
        encoder.finish().unwrap()
    }

    /// WOFF1 with `num` stored (uncompressed) tables, each 4 bytes:
    /// unique numeric tags, correct SFNT checksums, non-overlapping
    /// ranges past the directory, and a `totalSfntSize` matching the
    /// recomputed SFNT size exactly. Nothing in this input is
    /// malformed except, for `num = 4096`, the table count itself —
    /// so it sails past the empty-table, overlap, and size guards
    /// and stands or falls on the count guard alone.
    fn build_stored_many_table_woff(num: usize) -> Vec<u8> {
        let dir_end = 44 + num * 20;
        let total_sfnt = 12 + num * 16 + num * 4;
        let woff_len = dir_end + num * 4;
        let mut out = vec![0u8; woff_len];
        out[..4].copy_from_slice(b"wOFF");
        set_u32(&mut out, 4, 0x0001_0000);
        set_u32(&mut out, 8, woff_len as u32);
        set_u16(&mut out, 12, num as u16);
        set_u32(&mut out, 16, total_sfnt as u32);
        set_u16(&mut out, 20, 1);
        for i in 0..num {
            let at = 44 + i * 20;
            let tag = (i as u32).to_be_bytes();
            let payload = (i as u32).to_be_bytes();
            out[at..at + 4].copy_from_slice(&tag);
            set_u32(&mut out, at + 4, (dir_end + i * 4) as u32);
            set_u32(&mut out, at + 8, 4);
            set_u32(&mut out, at + 12, 4);
            set_u32(&mut out, at + 16, sfnt_table_checksum(&tag, &payload));
            out[dir_end + i * 4..dir_end + i * 4 + 4].copy_from_slice(&payload);
        }
        out
    }

    #[test]
    fn decode_webfont_rejects_huge_table_count_before_search_range() {
        // 4096 tables, all non-empty, uniquely tagged, correctly
        // checksummed, with a matching `totalSfntSize`: every guard
        // before the `searchRange` loop passes, so only the count
        // guard rejects it. (The old version of this test built
        // all-empty tables, which the empty-table guard rejects
        // first — it passed even with the count guard deleted.)
        // With the guard removed this input would reach the loop,
        // where `searchRange` must become 65536 and cannot fit a
        // u16 (release: wraps to 0, infinite loop; debug: panic).
        let woff = build_stored_many_table_woff(4096);
        assert!(decode_webfont(&woff).is_none());
        // 4095 tables with the same shape must decode: the count
        // guard must not reject a representable count, and the
        // rebuilt SFNT must carry the full directory.
        let woff = build_stored_many_table_woff(4095);
        let decoded = decode_webfont(&woff).expect("4095 tables must decode");
        assert!(is_sfnt(&decoded));
        assert_eq!(decoded.len(), 12 + 4095 * 16 + 4095 * 4);
        assert_eq!(u16::from_be_bytes(decoded[4..6].try_into().unwrap()), 4095);
        assert_eq!(
            u16::from_be_bytes(decoded[6..8].try_into().unwrap()),
            32768,
            "searchRange for 4095 tables"
        );
        // First record (lowest tag, 0) points at the first data block.
        let data_at = 12 + 4095 * 16;
        assert_eq!(&decoded[12..16], &[0, 0, 0, 0]);
        assert_eq!(
            u32::from_be_bytes(decoded[20..24].try_into().unwrap()) as usize,
            data_at
        );
        assert_eq!(&decoded[data_at..data_at + 4], &[0, 0, 0, 0]);
    }

    #[test]
    fn decode_webfont_rejects_stored_table_checksum_mismatch() {
        // Flip one byte in a stored table: no zlib stream stands
        // guard, so only the SFNT checksum check can reject it.
        let stored = include_bytes!("../tests/fixtures/Ahem-stored.woff");
        let (_, offset, _, _) = woff_record(stored, b"glyf");
        let mut corrupt = stored.to_vec();
        corrupt[offset] ^= 0xff;
        assert!(decode_webfont(&corrupt).is_none());
        // Same for a deflated table whose stream is valid but whose
        // declared checksum lies: the checksum guard, not zlib, must
        // reject it.
        let payload = b"hello font table, hello font table";
        let comp = zlib(payload);
        let bad_checksum = sfnt_table_checksum(b"test", payload).wrapping_add(1);
        let woff = build_single_table_woff(
            b"test",
            &comp,
            payload.len() as u32,
            bad_checksum,
            None,
            None,
        );
        assert!(decode_webfont(&woff).is_none());
        // Control: the correct checksum decodes.
        let good = sfnt_table_checksum(b"test", payload);
        let woff = build_single_table_woff(b"test", &comp, payload.len() as u32, good, None, None);
        assert!(decode_webfont(&woff).is_some());
    }

    #[test]
    fn sfnt_table_checksum_zeroes_head_adjustment() {
        let mut head = [0u8; 16];
        head[8..12].copy_from_slice(&0x0102_0304u32.to_be_bytes());
        let mut zeroed = head;
        zeroed[8..12].copy_from_slice(&[0; 4]);
        assert_eq!(
            sfnt_table_checksum(b"head", &head),
            sfnt_table_checksum(b"head", &zeroed)
        );
        assert_ne!(
            sfnt_table_checksum(b"head", &head),
            sfnt_table_checksum(b"post", &head),
            "only the head table gets the adjustment treatment"
        );
    }

    #[test]
    fn decode_webfont_rejects_each_malformed_directory_guard() {
        // Duplicate tag: second record copies the first record's tag.
        let mut dup = WOFF_AHEM_BYTES.to_vec();
        let first_tag = dup[44..48].to_vec();
        dup[64..68].copy_from_slice(&first_tag);
        assert!(decode_webfont(&dup).is_none(), "duplicate tag");
        // Overlapping compressed ranges: second record starts where
        // the first starts.
        let mut overlap = WOFF_AHEM_BYTES.to_vec();
        let (_, first_offset, _, _) = woff_record(WOFF_AHEM_BYTES, b"OS/2");
        set_u32(&mut overlap, 64 + 4, first_offset as u32);
        assert!(decode_webfont(&overlap).is_none(), "overlapping ranges");
        // Offset inside the directory.
        let mut inside = WOFF_AHEM_BYTES.to_vec();
        set_u32(&mut inside, 44 + 4, 44);
        assert!(decode_webfont(&inside).is_none(), "offset < dir_end");
        // Range past the input.
        let mut past_input = WOFF_AHEM_BYTES.to_vec();
        set_u32(&mut past_input, 44 + 8, WOFF_AHEM_BYTES.len() as u32 + 1000);
        assert!(
            decode_webfont(&past_input).is_none(),
            "offset + compLength > input"
        );
        // Zero tables.
        let mut zero = WOFF_AHEM_BYTES.to_vec();
        set_u16(&mut zero, 12, 0);
        assert!(decode_webfont(&zero).is_none(), "num_tables = 0");
        // Collection flavor.
        let mut ttcf = WOFF_AHEM_BYTES.to_vec();
        ttcf[4..8].copy_from_slice(b"ttcf");
        assert!(decode_webfont(&ttcf).is_none(), "ttcf flavor");
        // Declared length past the input.
        let mut long = WOFF_AHEM_BYTES.to_vec();
        set_u32(&mut long, 8, WOFF_AHEM_BYTES.len() as u32 + 100);
        assert!(decode_webfont(&long).is_none(), "woff_length > input");
        // Directory past the declared length.
        let mut dir_past = WOFF_AHEM_BYTES.to_vec();
        set_u32(&mut dir_past, 8, 64);
        assert!(decode_webfont(&dir_past).is_none(), "dir_end > woff_length");
        // A table range past the declared length (but inside the
        // input, i.e. hiding in trailing bytes).
        let (_, first_offset, first_comp, _) = woff_record(WOFF_AHEM_BYTES, b"OS/2");
        let mut past_declared = WOFF_AHEM_BYTES.to_vec();
        set_u32(
            &mut past_declared,
            8,
            (first_offset + first_comp - 1) as u32,
        );
        assert!(
            decode_webfont(&past_declared).is_none(),
            "offset + compLength > woff_length"
        );
        // Nonzero reserved field.
        let mut reserved = WOFF_AHEM_BYTES.to_vec();
        set_u16(&mut reserved, 14, 1);
        assert!(decode_webfont(&reserved).is_none(), "reserved != 0");
        // Metadata range past the file.
        let mut meta_past = WOFF_AHEM_BYTES.to_vec();
        set_u32(&mut meta_past, 24, WOFF_AHEM_BYTES.len() as u32);
        set_u32(&mut meta_past, 28, 10);
        assert!(
            decode_webfont(&meta_past).is_none(),
            "meta past woff_length"
        );
        // Metadata range overlapping a table range.
        let mut meta_overlap = WOFF_AHEM_BYTES.to_vec();
        set_u32(&mut meta_overlap, 24, first_offset as u32);
        set_u32(&mut meta_overlap, 28, 10);
        assert!(
            decode_webfont(&meta_overlap).is_none(),
            "meta overlaps table"
        );
        // Private range past the file.
        let mut priv_past = WOFF_AHEM_BYTES.to_vec();
        set_u32(&mut priv_past, 36, WOFF_AHEM_BYTES.len() as u32);
        set_u32(&mut priv_past, 40, 10);
        assert!(
            decode_webfont(&priv_past).is_none(),
            "priv past woff_length"
        );
    }

    #[test]
    fn decode_webfont_rejects_meta_priv_overlap_and_accepts_valid_blocks() {
        // One stored table at 64..68, then metadata/private blocks.
        let payload = b"abcd";
        let checksum = sfnt_table_checksum(b"test", payload);
        // Overlapping metadata/private ranges: rejected.
        let woff = build_single_table_woff(
            b"test",
            payload,
            payload.len() as u32,
            checksum,
            Some((68, 10)),
            Some((73, 10)),
        );
        assert!(decode_webfont(&woff).is_none(), "meta/priv overlap");
        // Disjoint, in-bounds blocks: accepted, glyph data intact.
        let woff = build_single_table_woff(
            b"test",
            payload,
            payload.len() as u32,
            checksum,
            Some((68, 10)),
            Some((78, 10)),
        );
        let decoded = decode_webfont(&woff).expect("valid meta/priv must decode");
        assert!(is_sfnt(&decoded));
        assert_eq!(&decoded[28..32], payload, "table bytes must survive");
    }

    #[test]
    fn decode_webfont_rejects_crafted_stream_length_guards() {
        // Stream inflating to origLength + 1: the `take(origLength
        // + 1)` guard must reject it, not emit a truncated font.
        let payload = vec![b'A'; 65];
        let comp = zlib(&payload);
        let checksum = sfnt_table_checksum(b"test", &payload);
        let woff = build_single_table_woff(b"test", &comp, 64, checksum, None, None);
        assert!(decode_webfont(&woff).is_none(), "inflate to origLength + 1");
        // Huge expansion against a small declared length: the
        // stream expands to 1 MiB while `origLength` is 64, so the
        // pre-decode size caps cannot fire (the recomputed SFNT
        // size is 12 + 16 + 64, far under the 32 MiB cap) and the
        // rejection must come from the decode loop itself: `take`
        // truncates the expansion at `origLength + 1` = 65 bytes,
        // so the exact-length check (65 != 64) rejects. Without
        // `take`, the decoder would buffer the full 1 MiB before
        // rejecting — the cap is what bounds the allocation, not
        // just the outcome. (The old 65-byte leg above cannot pin
        // this: a 65-byte expansion never exceeds the `take` limit,
        // so `take` is inert there.)
        let payload = vec![0u8; 1 << 20];
        let comp = zlib(&payload);
        let checksum = sfnt_table_checksum(b"test", &payload);
        let woff = build_single_table_woff(b"test", &comp, 64, checksum, None, None);
        assert!(
            decode_webfont(&woff).is_none(),
            "1 MiB expansion against origLength 64 must be rejected by the take cap"
        );
        // Complete zlib stream plus trailing garbage inside the
        // declared range: `total_in` must equal compLength exactly.
        let payload = vec![b'B'; 64];
        let mut comp = zlib(&payload);
        comp.push(0xff);
        let checksum = sfnt_table_checksum(b"test", &payload);
        let woff = build_single_table_woff(b"test", &comp, 64, checksum, None, None);
        assert!(
            decode_webfont(&woff).is_none(),
            "trailing garbage after zlib stream"
        );
        // Empty table: origLength 0 is not a table.
        let woff = build_single_table_woff(b"test", &[], 0, 0, None, None);
        assert!(decode_webfont(&woff).is_none(), "origLength = 0");
    }

    #[test]
    fn install_woff1_measures_identically_to_source_ttf() {
        // Same face, same metrics, whether installed as TTF or WOFF1:
        // Ahem advances are exactly 1em per char at any size.
        let mut cache = FontCache::new();
        cache.faces.clear();
        cache.install_test_fonts();
        let ttf_family = cache.intern_family("Ahem");
        let ttf_advance = cache.advance("Xp", 50.0, ttf_family);

        let mut cache = FontCache::new();
        cache.faces.clear();
        let woff: std::sync::Arc<[u8]> = std::sync::Arc::from(WOFF_AHEM_BYTES.to_vec());
        assert!(cache.install_font_owned(woff));
        let woff_family = cache.intern_family("Ahem");
        assert!((cache.advance("Xp", 50.0, woff_family) - ttf_advance).abs() < 0.01);
        assert!((cache.advance("Xp", 50.0, woff_family) - 100.0).abs() < 0.01);
        let (a, d) = cache.ascent_descent(50.0, woff_family);
        assert!((a - 40.0).abs() < 0.01, "ascent {a}");
        assert!((d - 10.0).abs() < 0.01, "descent {d}");

        // Dedupe works on decoded bytes: installing the WOFF1 and
        // then the equivalent TTF is a no-op second install.
        let ttf: std::sync::Arc<[u8]> = std::sync::Arc::from(AHEM_FONT_BYTES.to_vec());
        assert!(cache.install_font_owned(ttf));
        assert_eq!(
            cache
                .faces
                .iter()
                .filter(|f| f.families.iter().any(|n| n == "ahem"))
                .count(),
            1,
            "woff1 + ttf of the same face must dedupe to one face"
        );
    }

    #[test]
    fn family_aware_measurement_uses_installed_face() {
        // Ahem: every advance is exactly 1em; typo ascent 0.8em,
        // descent 0.2em, no leading.
        let mut cache = FontCache::new();
        cache.faces.clear();
        cache.install_test_fonts();
        let ahem = cache.intern_family("Ahem");
        assert!((cache.advance("X", 50.0, ahem) - 50.0).abs() < 0.01);
        assert!((cache.advance("Xp", 50.0, ahem) - 100.0).abs() < 0.01);
        assert!((cache.line_height(50.0, ahem) - 50.0).abs() < 0.01);
        let (a, d) = cache.ascent_descent(50.0, ahem);
        assert!((a - 40.0).abs() < 0.01, "ascent {a}");
        assert!((d - 10.0).abs() < 0.01, "descent {d}");
        // Interning is stable and ASCII case-insensitive.
        assert_eq!(cache.intern_family("ahem"), ahem);
        assert_eq!(cache.intern_family("AHEM"), ahem);
    }

    #[test]
    fn unknown_family_id_falls_back_to_default_face() {
        let cache = FontCache::new();
        // Never interned: measures with the default face, exactly like
        // the old family-less contract did.
        let w = cache.advance("X", 20.0, FontFamilyId::of("nope-not-a-family"));
        let def = cache.advance("X", 20.0, FontFamilyId::DEFAULT);
        assert!((w - def).abs() < 0.001);
    }

    #[test]
    fn layout_measures_ahem_runs_with_ahem() {
        // End-to-end: an Ahem-styled run laid out through
        // `SwashFontMetrics` must measure with Ahem advances (1em per
        // char), not the default serif face's.
        let mut cache = FontCache::new();
        cache.install_test_fonts();
        let metrics = SwashFontMetrics::new(&cache);
        let (doc, _) = bowser_dom::Document::parse_html(
            "<html><body><div class=\"t\">Xp</div></body></html>".as_bytes(),
        );
        let ss = bowser_css::parse_stylesheet(".t { font-family: Ahem; font-size: 50px; }");
        let tree = bowser_layout::layout(
            &doc,
            &[&ss],
            bowser_layout::Size { w: 800.0, h: 600.0 },
            &metrics,
        );
        let mut widths = Vec::new();
        let mut stack = vec![&tree];
        while let Some(b) = stack.pop() {
            stack.extend(&b.children);
            if b.box_type == bowser_layout::BoxType::Text {
                widths.extend(b.fragments.iter().map(|f| f.rect.w));
            }
        }
        assert_eq!(widths.len(), 1, "one text fragment, got {widths:?}");
        assert!(
            (widths[0] - 100.0).abs() < 0.01,
            "Ahem 'Xp' at 50px must be 100px wide, got {}",
            widths[0]
        );
    }

    #[test]
    fn resolve_never_returns_none() {
        // The embedded fallback is the last resort: unknown families and
        // generics always resolve, on fontless systems too.
        let mut cache = FontCache::new();
        cache.faces.clear(); // simulate a fontless system
        for family in [
            "definitely-not-a-real-font-family-xyz",
            "serif",
            "monospace",
        ] {
            let key = cache
                .resolve(family, 700, FontStyle::Italic)
                .expect("resolve must not return None");
            let data = cache.font_data(key).expect("resolved key must have data");
            assert!(
                data.font_ref().is_some(),
                "resolved bytes must parse as a font"
            );
        }
        assert!(
            cache.default_key().is_some(),
            "default key is guaranteed by the fallback"
        );
    }

    #[test]
    fn system_font_loads() {
        let cache = FontCache::new();
        let key = cache.default_key().expect("default key always resolves");
        let data = cache.font_data(key).expect("default key resolves");
        assert!(!data.bytes().is_empty());
        assert!(data.font_ref().is_some(), "bytes must parse as a font");
    }

    #[test]
    fn measure_hello_width_positive() {
        let cache = FontCache::new();
        let w = cache.advance("hello", 16.0, FontFamilyId::DEFAULT);
        assert!(w > 0.0, "advance of \"hello\" must be positive, got {w}");
        assert!(
            cache.line_height(16.0, FontFamilyId::DEFAULT) > 0.0,
            "line height must be positive"
        );
    }

    #[test]
    fn advance_scales_with_size() {
        let cache = FontCache::new();
        let a = cache.advance("hello", 16.0, FontFamilyId::DEFAULT);
        let b = cache.advance("hello", 32.0, FontFamilyId::DEFAULT);
        assert!(
            (b - 2.0 * a).abs() < 1.0,
            "advance should scale ~linearly with size: {a} vs {b}"
        );
    }

    #[test]
    fn advance_is_shaped_and_cached() {
        // `advance` goes through paint's shaper now (kerning, ligatures):
        // repeat calls return bit-identical values via the shaper's
        // thread-local cache, and the per-(font, size) proxy cache is
        // only fed by `line_height`.
        let cache = FontCache::new();
        let a = cache.advance("hello world", 16.0, FontFamilyId::DEFAULT);
        assert!(a > 0.0, "advance must be positive, got {a}");
        let b = cache.advance("hello world", 16.0, FontFamilyId::DEFAULT);
        assert_eq!(a, b, "repeat advance must be bit-identical");
        assert_eq!(
            cache.measure_cache.borrow().len(),
            0,
            "advance must not feed the proxy cache"
        );
        let lh = cache.line_height(16.0, FontFamilyId::DEFAULT);
        assert!(lh > 0.0, "line height must be positive, got {lh}");
        assert_eq!(
            cache.measure_cache.borrow().len(),
            1,
            "line_height builds one (font, size) entry"
        );
        let lh2 = cache.line_height(16.0, FontFamilyId::DEFAULT);
        assert_eq!(lh, lh2, "cached line height must be bit-identical");
    }

    #[test]
    fn swash_metrics_advance_matches_paint_shaping() {
        // The core agreement invariant: what layout measures for a run
        // is exactly what paint's shaper produces for the same string.
        let cache = FontCache::new();
        let metrics = SwashFontMetrics::new(&cache);
        let key = cache.default_key().expect("default key always resolves");
        let data = cache.font_data(key).expect("default key resolves");
        for text in [
            "hello",
            "AV To",
            "fi fl",
            "the quick brown fox",
            "  spaced  ",
            "0123456789",
        ] {
            let want = shape_text(&data, text, 16.0).width;
            assert_eq!(
                metrics.advance(text, 16.0, FontFamilyId::DEFAULT),
                want,
                "advance({text:?}) must equal paint's shaped width"
            );
            assert_eq!(
                metrics.advance_for_key(key, text, 16.0),
                want,
                "advance_for_key({text:?}) must equal paint's shaped width"
            );
        }
    }

    #[test]
    fn swash_metrics_line_height_is_typographic() {
        let cache = FontCache::new();
        let metrics = SwashFontMetrics::new(&cache);
        let key = cache.default_key().expect("default key always resolves");
        let (ascent, descent, leading) = metrics
            .extents_for_key(key, 16.0)
            .expect("default face must parse");
        assert!(ascent > 0.0, "ascent must be positive, got {ascent}");
        assert!(
            leading >= 0.0,
            "leading must be non-negative, got {leading}"
        );
        let lh = metrics.line_height(16.0, FontFamilyId::DEFAULT);
        assert!(
            (lh - (ascent + descent + leading)).abs() < 1e-4,
            "line_height must be ascent + descent + leading, got {lh}"
        );
        assert_eq!(lh, metrics.line_height_for_key(key, 16.0));
        // Linear in size (scaled metrics, not a fixed multiple).
        let lh2 = metrics.line_height(32.0, FontFamilyId::DEFAULT);
        assert!(
            (lh2 - 2.0 * lh).abs() < 1e-3,
            "line height must scale linearly: {lh} vs {lh2}"
        );
    }

    #[test]
    fn layout_fragment_widths_match_painted_advances() {
        // End-to-end wrapping agreement: lay out a wrapping paragraph
        // with `SwashFontMetrics`, then check every text fragment's
        // laid-out width against paint's shaped width for the same
        // slice. Line breaks are decided with the widths paint uses.
        use bowser_dom::Document;
        use bowser_layout::{Size, layout};

        let cache = FontCache::new();
        let metrics = SwashFontMetrics::new(&cache);
        let key = cache.default_key().expect("default key always resolves");
        let data = cache.font_data(key).expect("default key resolves");

        let words = "AV To fi fl the quick brown fox jumps over the lazy dog ".repeat(6);
        let html = format!("<html><body><p style=\"font-size: 20px\">{words}</p></body></html>");
        let (doc, _) = Document::parse_html(html.as_bytes());
        let root = layout(&doc, &[], Size { w: 220.0, h: 800.0 }, &metrics);

        let mut stack = vec![&root];
        let mut frag_count = 0usize;
        let mut rows = std::collections::HashSet::new();
        while let Some(b) = stack.pop() {
            stack.extend(&b.children);
            let Some(text) = b.text.as_deref() else {
                continue;
            };
            for frag in &b.fragments {
                let slice = text.get(frag.text_range.clone()).unwrap_or("");
                if slice.is_empty() {
                    continue;
                }
                frag_count += 1;
                rows.insert(frag.rect.y.to_bits());
                let want = shape_text(&data, slice, 20.0).width;
                assert!(
                    (frag.rect.w - want).abs() < 0.01,
                    "fragment {slice:?}: laid-out width {} != painted width {want}",
                    frag.rect.w
                );
            }
        }
        assert!(
            frag_count > 4,
            "paragraph must wrap into several fragments, got {frag_count}"
        );
        assert!(rows.len() > 1, "wrapping must produce multiple rows");
    }

    #[test]
    fn resolve_cache_hit_returns_same_key() {
        let cache = FontCache::new();
        let a = cache.resolve("serif", 400, FontStyle::Normal);
        let Some(a) = a else {
            return; // unreachable with the fallback, kept defensive
        };
        let b = cache.resolve("Serif", 400, FontStyle::Normal);
        assert_eq!(Some(a), b, "normalized family must hit the same key");
        let c = cache.resolve("serif", 700, FontStyle::Normal);
        let Some(c) = c else {
            return; // unreachable with the fallback, kept defensive
        };
        // Different weights may or may not map to the same face file;
        // the cache must at least return valid data for both.
        assert!(cache.font_data(a).is_some());
        assert!(cache.font_data(c).is_some());
    }

    #[test]
    fn resolve_unknown_family_falls_back() {
        let cache = FontCache::new();
        // Unknown families degrade through sans-serif to the embedded
        // fallback: always `Some` now.
        let key = cache.resolve(
            "definitely-not-a-real-font-family-xyz",
            400,
            FontStyle::Normal,
        );
        assert!(key.is_some(), "unknown family must fall back, not fail");
    }

    #[test]
    fn font_data_unknown_key_is_none() {
        let cache = FontCache::new();
        assert!(cache.font_data(FontKey(u32::MAX)).is_none());
    }

    #[test]
    fn distinct_triples_mint_distinct_keys() {
        // Keys are per (family, weight, style) query even when the
        // underlying face coincides (e.g. everything hits the fallback).
        // "monospace" (not the eager default) forces the fallback path
        // here: `faces.clear()` runs after `new` already resolved serif.
        let mut cache = FontCache::new();
        cache.faces.clear(); // force everything onto the fallback face
        let a = cache.resolve("monospace", 400, FontStyle::Normal).unwrap();
        let b = cache.resolve("monospace", 700, FontStyle::Normal).unwrap();
        let c = cache.resolve("monospace", 400, FontStyle::Italic).unwrap();
        assert!(a != b && b != c && a != c);
        assert_eq!(cache.font_data(a).unwrap().bytes(), FALLBACK_FONT_BYTES);
        assert_eq!(cache.font_data(b).unwrap().bytes(), FALLBACK_FONT_BYTES);
    }

    #[test]
    fn weight_preference_order_cases() {
        // Exact match always wins.
        assert_eq!(
            weight_preference_order(400, &[300, 400, 700]),
            vec![400, 300, 700]
        );
        // 400 prefers 500, then lighter descending, then heavier.
        assert_eq!(
            weight_preference_order(400, &[300, 500, 700]),
            vec![500, 300, 700]
        );
        // 500 prefers 400, then lighter descending, then heavier.
        assert_eq!(
            weight_preference_order(500, &[300, 400, 700]),
            vec![400, 300, 700]
        );
        // Below 400: lighter descending first.
        assert_eq!(
            weight_preference_order(300, &[200, 400, 700]),
            vec![200, 400, 700]
        );
        // Above 500: heavier ascending first.
        assert_eq!(
            weight_preference_order(800, &[400, 700, 900]),
            vec![900, 700, 400]
        );
    }

    #[test]
    fn style_penalty_cases() {
        assert_eq!(style_penalty(FontStyle::Normal, FontStyle::Normal), 0);
        assert_eq!(style_penalty(FontStyle::Italic, FontStyle::Oblique), 1);
        assert_eq!(style_penalty(FontStyle::Oblique, FontStyle::Italic), 1);
        assert_eq!(style_penalty(FontStyle::Italic, FontStyle::Normal), 2);
        assert_eq!(style_penalty(FontStyle::Normal, FontStyle::Italic), 2);
    }

    #[test]
    fn normalize_family_cases() {
        assert_eq!(normalize_family("  DejaVu  Sans "), "dejavu sans");
        assert_eq!(normalize_family("SERIF"), "serif");
    }

    #[test]
    fn normalize_family_cow_borrows_when_already_normalized() {
        // The common query shapes (`css_font_family` hands `resolve`
        // values like these per text run) must not allocate: the
        // borrowed form is the proof — no `String` is constructed.
        for already in [
            "serif",
            "sans-serif",
            "monospace",
            "cursive",
            "fantasy",
            "arial",
            "dejavu sans",
            "a b",
            "",
            "a",
        ] {
            assert!(
                matches!(normalize_family_cow(already), Cow::Borrowed(_)),
                "{already:?} should borrow, not allocate"
            );
        }
    }

    #[test]
    fn normalize_family_cow_owns_only_when_needed() {
        // Mixed case, stray/double whitespace, and non-ASCII take the
        // owned path, with the same value `normalize_family` produces.
        for (input, expected) in [
            ("Serif", "serif"),
            ("  DejaVu  Sans ", "dejavu sans"),
            ("Noto Sans", "noto sans"),
            ("a\tb", "a b"),
            ("a\n\nb", "a b"),
            ("É", "É"), // non-ASCII: conservative owned path
            ("ﬁ", "ﬁ"),
        ] {
            match normalize_family_cow(input) {
                Cow::Owned(owned) => assert_eq!(owned, expected, "input {input:?}"),
                Cow::Borrowed(_) => panic!("{input:?} must take the owned path"),
            }
        }
    }

    #[test]
    fn normalize_family_cow_matches_normalize_family() {
        // The fast path must be value-identical to the allocating one,
        // whichever branch an input takes.
        for input in [
            "",
            " ",
            "  ",
            "serif",
            "Serif",
            "SERIF",
            " s ",
            "a  b",
            "a\tb",
            "a\u{a0}b", // non-breaking space: real whitespace to split
            "Noto Sans CJK",
            "NOTO  SANS",
            "ß",
            "İ", // non-ASCII uppercase: to_ascii_lowercase leaves it
            "école",
            "a b ",
            " a",
            "a\x0bb", // vertical tab: split_whitespace splits here,
            "\x0ba",  // leading vertical tab: stray leading whitespace
            "a\x0b",  // trailing vertical tab: stray trailing whitespace
            "\x0b",   // lone vertical tab: all whitespace
            "a\x0b\x0bb",
        ] {
            assert_eq!(
                normalize_family_cow(input).into_owned(),
                normalize_family(input),
                "input {input:?}"
            );
        }
    }

    #[test]
    fn resolve_repeated_query_hits_one_cache_entry() {
        // The per-text-run pattern: the same family resolved over and
        // over. Every repeat must hit the cached entry — one outer
        // family entry, no re-walk of the face list, same key back.
        let mut cache = FontCache::new();
        let base = cache.by_query.borrow().len(); // `new` resolves the default face
        cache.faces.clear(); // force everything onto the fallback face
        // "monospace": `new` eagerly resolved serif, so this is the
        // first query for a fresh family — one new family entry.
        let first = cache.resolve("monospace", 400, FontStyle::Normal).unwrap();
        for _ in 0..50 {
            assert_eq!(
                cache.resolve("monospace", 400, FontStyle::Normal),
                Some(first)
            );
        }
        assert_eq!(
            cache.by_query.borrow().len(),
            base + 1,
            "one new family entry"
        );
        assert_eq!(
            cache.by_query.borrow()["monospace"].len(),
            1,
            "one (weight, style) entry"
        );
    }

    #[test]
    fn resolve_case_and_whitespace_variants_share_one_entry() {
        // Normalization happens once per call and equivalent spellings
        // collapse onto the same cache entry.
        let mut cache = FontCache::new();
        let base = cache.by_query.borrow().len(); // `new` resolves the default face
        cache.faces.clear();
        // "monospace" (not the eager serif default): a fresh family, so
        // the variants collapse onto one new entry.
        let keys = [
            cache.resolve("monospace", 400, FontStyle::Normal).unwrap(),
            cache.resolve("Monospace", 400, FontStyle::Normal).unwrap(),
            cache
                .resolve(" MONOSPACE ", 400, FontStyle::Normal)
                .unwrap(),
            cache.resolve("mOnOsPaCe", 400, FontStyle::Normal).unwrap(),
        ];
        assert!(
            keys.windows(2).all(|w| w[0] == w[1]),
            "variants must resolve to the same key: {keys:?}"
        );
        assert_eq!(
            cache.by_query.borrow().len(),
            base + 1,
            "variants share one entry"
        );
    }

    #[test]
    fn resolve_distinct_triples_share_one_family_entry() {
        // Different (weight, style) triples on one family live under the
        // single family entry instead of minting one map entry per
        // triple at the top level.
        let mut cache = FontCache::new();
        let base = cache.by_query.borrow().len(); // `new` resolves the default face
        cache.faces.clear();
        let triples = [
            (400, FontStyle::Normal),
            (700, FontStyle::Normal),
            (400, FontStyle::Italic),
            (700, FontStyle::Oblique),
        ];
        // "monospace" (not the eager serif default): one fresh family
        // entry holding one entry per triple.
        let keys: Vec<FontKey> = triples
            .iter()
            .map(|&(w, s)| cache.resolve("monospace", w, s).unwrap())
            .collect();
        assert_eq!(
            cache.by_query.borrow().len(),
            base + 1,
            "one new family entry"
        );
        assert_eq!(
            cache.by_query.borrow()["monospace"].len(),
            triples.len(),
            "one entry per triple"
        );
        // Repeat each triple: all hits, nothing new inserted.
        for (&(w, s), &k) in triples.iter().zip(keys.iter()) {
            assert_eq!(cache.resolve("monospace", w, s), Some(k));
        }
        assert_eq!(cache.by_query.borrow()["monospace"].len(), triples.len());
    }

    #[test]
    fn resolve_unknown_family_caches_the_fallback() {
        // The fallback face is the answer for unknown families, and it
        // is cached like any other hit: repeats never re-walk the faces.
        let mut cache = FontCache::new();
        let base = cache.by_query.borrow().len(); // `new` resolves the default face
        cache.faces.clear();
        let first = cache
            .resolve(
                "definitely-not-a-real-font-family-xyz",
                400,
                FontStyle::Normal,
            )
            .unwrap();
        assert_eq!(cache.font_data(first).unwrap().bytes(), FALLBACK_FONT_BYTES);
        for _ in 0..10 {
            assert_eq!(
                cache.resolve(
                    "definitely-not-a-real-font-family-xyz",
                    400,
                    FontStyle::Normal
                ),
                Some(first)
            );
        }
        assert_eq!(cache.by_query.borrow().len(), base + 1);
    }

    #[test]
    fn resolver_hook_serves_resolved_faces() {
        let cache = FontCache::new();
        let key = cache.resolve("sans-serif", 400, FontStyle::Normal).unwrap();
        let hook = cache.resolver();
        assert!(hook(key).is_some());
        assert!(hook(FontKey(u32::MAX)).is_none());
        // `Send + Sync + 'static`: movable across threads.
        let moved = std::thread::spawn(move || hook(key).is_some())
            .join()
            .expect("hook must work on another thread");
        assert!(moved);
    }

    #[test]
    fn snapshot_covers_resolved_faces() {
        let cache = FontCache::new();
        let key = cache.resolve("sans-serif", 400, FontStyle::Normal).unwrap();
        let snap = cache.snapshot();
        assert!(snap.contains_key(&key));
    }

    #[test]
    fn resolver_is_stable_until_a_new_face_resolves() {
        // The built hook is cached: repeated `resolver()` calls share one
        // `Arc` (hook identity is what keeps backend font-data caches
        // warm), so per-frame rebuilds neither re-clone the face map nor
        // flush the thread-local caches.
        let cache = FontCache::new();
        let first = cache.resolver();
        let second = cache.resolver();
        assert!(
            Arc::ptr_eq(&first, &second),
            "resolver must be stable across calls"
        );
        // Resolving a new face widens the key space: the cached hook is
        // dropped and the next call builds a fresh one covering it.
        let key = cache
            .resolve("monospace", 400, FontStyle::Normal)
            .expect("monospace resolves");
        let third = cache.resolver();
        assert!(
            !Arc::ptr_eq(&first, &third),
            "new face must invalidate the cached resolver"
        );
        assert!(third(key).is_some(), "rebuilt hook covers the new face");
        // And the rebuilt hook is stable again.
        assert!(Arc::ptr_eq(&third, &cache.resolver()));
    }

    #[test]
    fn font_data_clone_shares_bytes() {
        // `FontData` clones are `Arc` bumps, not font-file memcpys: two
        // clones of one face name the same backing allocation.
        let cache = FontCache::new();
        let key = cache.default_key().expect("default face resolves");
        let a = cache.font_data(key).expect("face present").clone();
        let b = cache.font_data(key).expect("face present").clone();
        assert!(
            Arc::ptr_eq(&a.bytes, &b.bytes),
            "cloned FontData must share the byte buffer"
        );
    }

    #[test]
    fn shared_cache_measures_with_real_advances() {
        // The shareable handle implements `FontMetrics` with the same
        // real advances as the cache itself (not the 0.6em placeholder).
        let shared = SharedFontCache::new();
        let direct = FontCache::new();
        let fonts: &dyn FontMetrics = &shared;
        let text = "hello";
        assert_eq!(
            fonts.advance(text, 16.0, FontFamilyId::DEFAULT),
            direct.advance(text, 16.0, FontFamilyId::DEFAULT)
        );
        assert_eq!(
            fonts.line_height(16.0, FontFamilyId::DEFAULT),
            direct.line_height(16.0, FontFamilyId::DEFAULT)
        );
        assert!(fonts.advance(text, 16.0, FontFamilyId::DEFAULT) > 0.0);
    }

    #[test]
    fn shared_cache_clone_shares_state() {
        // Cloning is an `Arc` clone: a face resolved through one handle
        // is visible through the other, with no second directory walk.
        let a = SharedFontCache::new();
        let b = a.clone();
        let key = a.resolve("serif", 700, FontStyle::Italic).unwrap();
        assert!(b.font_data(key).is_some());
        assert_eq!(a.font_count(), b.font_count());
        assert_eq!(a.default_key(), b.default_key());
        // `Send + Sync`: movable across threads like the resolver hook.
        let moved = std::thread::spawn(move || b.advance("hi", 16.0, FontFamilyId::DEFAULT) > 0.0)
            .join()
            .expect("shared cache must work on another thread");
        assert!(moved);
    }

    #[test]
    fn shared_cache_lock_gives_mut_access() {
        // Display-list building needs `&mut FontCache` for resolution;
        // `lock` provides it without giving up sharing.
        let shared = SharedFontCache::new();
        let key = shared.lock().resolve("monospace", 400, FontStyle::Normal);
        assert!(key.is_some());
        assert!(shared.font_data(key.unwrap()).is_some());
    }
}

/// Wave 65 perf tests: the query/face/measurement caches moved from
/// `RandomState` to the local FNV-1a hasher. Behavior pins: resolution,
/// fallback, and measurement results are identical — only the hash
/// function changed.
#[cfg(test)]
mod w65_p_fonts_tests {
    use super::*;

    #[test]
    fn fnv_hasher_is_deterministic() {
        let hash = |s: &str| {
            use std::hash::{BuildHasher, Hasher};
            let mut h = FnvBuildHasher.build_hasher();
            h.write(s.as_bytes());
            h.finish()
        };
        assert_eq!(hash("serif"), hash("serif"));
        assert_ne!(hash("serif"), hash("sans-serif"));
        // Integer keys hash too (FontKey / (FontKey, u32) map keys).
        let key_hash = |k: u32| {
            use std::hash::{BuildHasher, Hasher};
            let mut h = FnvBuildHasher.build_hasher();
            h.write_u32(k);
            h.finish()
        };
        assert_eq!(key_hash(7), key_hash(7));
    }

    #[test]
    fn resolve_is_stable_across_fnv_query_cache() {
        // Repeat resolutions hit `by_query` (now FNV-1a-hashed) and return
        // the same key as the first resolution.
        let cache = FontCache::new();
        let a = cache.resolve("serif", 400, FontStyle::Normal);
        let b = cache.resolve("serif", 400, FontStyle::Normal);
        assert_eq!(a, b);
        assert!(a.is_some());
        // A distinct (weight, style) triple caches separately.
        let c = cache.resolve("serif", 700, FontStyle::Italic);
        assert!(c.is_some());
        assert_eq!(cache.resolve("serif", 700, FontStyle::Italic), c);
    }

    #[test]
    fn unknown_family_falls_back_and_caches() {
        // The miss path (embedded fallback) is cached too: a repeat
        // query for an unknown family never re-walks the face list.
        let mut cache = FontCache::new();
        cache.faces.clear(); // simulate a fontless system
        let a = cache
            .resolve("not-a-real-family-xyz", 400, FontStyle::Normal)
            .expect("resolve never returns None");
        let b = cache
            .resolve("not-a-real-family-xyz", 400, FontStyle::Normal)
            .expect("resolve never returns None");
        assert_eq!(a, b);
        let data = cache.font_data(a).expect("resolved key has data");
        assert_eq!(data.bytes(), FALLBACK_FONT_BYTES);
    }

    #[test]
    fn snapshot_and_font_data_agree() {
        // `snapshot()` (now an FNV-1a map) covers every resolved face.
        let cache = FontCache::new();
        let key = cache
            .resolve("monospace", 400, FontStyle::Normal)
            .expect("monospace resolves");
        let snap = cache.snapshot();
        assert!(snap.contains_key(&key));
        assert_eq!(
            snap.get(&key).map(|d| d.face_index()),
            cache.font_data(key).map(|d| d.face_index())
        );
    }

    #[test]
    fn measure_cache_still_dedupes() {
        // `measure_cache` (now FNV-1a-hashed) still dedupes repeat
        // measurements bit-identically.
        let cache = FontCache::new();
        let a = cache.line_height(16.0, FontFamilyId::DEFAULT);
        let b = cache.line_height(16.0, FontFamilyId::DEFAULT);
        assert_eq!(a, b);
        assert!(a > 0.0);
        assert_eq!(cache.measure_cache.borrow().len(), 1);
    }
}
