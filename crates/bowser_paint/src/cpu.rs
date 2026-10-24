//! CPU fallback paint backend.
//!
//! Renders the same [`DisplayList`] as the wgpu backend using tiny-skia,
//! for use when no GPU adapter is available.
//!
//! # Sibling status (2026-09-25)
//!
//! Landed siblings this backend codes against:
//! - `crate::display_list` (real, landed): `DisplayList`, `DisplayItem`,
//!   `Backend`, `Quad { rect, color }`, `Color` (packed RGBA8 u32;
//!   0..1 channel accessors), `BorderStyle`, `DecoKind`, `MarkerKind`, `FontKey`,
//!   `FontData`, `ImageId`, `DecodedImage`, `DisplayListBuilder`,
//!   `build_display_list`. `Rect`/`Size` are `bowser_layout`'s.
//! - `crate::borders`: `tessellate_border(rect, widths, styles, colors)
//!   -> Vec<Quad>`, `tessellate_border_into` /
//!   `tessellate_border_rounded_into` (fill the pooled
//!   `CpuBackend::border_scratch` instead of allocating),
//!   `marker_geometry(kind, x, y, size, color) ->
//!   Vec<Quad>`, `decoration_geometry(kind, x0, x1, y, thickness, color)
//!   -> Vec<Quad>`.
//! - `crate::images`: `decode_image`; `ImageId`/`DecodedImage` are
//!   canonically defined in `crate::display_list` (deduped 2026-09-25 by
//!   the display-list worker) and only re-exported by `crate::images` —
//!   this backend uses `display_list`'s, since `DisplayList::image`
//!   returns those.
//!
//! Still missing (LOUD): line breaking belongs to the layout worker.
//! Text shaping itself is real: [`Painter::paint_text_run`] shapes
//! through the text worker's [`crate::text::shape_text`] (kerning,
//! ligatures, complex scripts) and rasterizes coverage through the text
//! worker's real [`crate::text::rasterize_glyph`], cached in its CPU
//! [`crate::text::GlyphAtlas`] (keyed like the GPU [`crate::text::Atlas`]).
//! The atlas lives on the [`CpuBackend`] and persists across frames,
//! bounded by a miss-count eviction policy (see [`Painter`]).
//!
//! # Interface notes
//!
//! - `render.rs` calls `crate::cpu::CpuBackend::render_to_rgba(&display_list,
//!   w, h)` — the static 3-argument associated function. The `&mut self`
//!   5-argument form is named [`CpuBackend::render_to_rgba_full`] instead:
//!   Rust forbids an associated function and a method sharing one name
//!   (E0592), so both cannot be `render_to_rgba`.
//! - The task sketch for `present` took `&softbuffer::Surface`; that type
//!   cannot be named without a `raw-window-handle` dependency (Cargo.toml
//!   is frozen), so [`CpuBackend::present`] takes the already-borrowed
//!   `&mut [u32]` buffer. See its docs.
//!
//! # Documented approximations
//!
//! - Clipping: tiny-skia 0.11 has no clip-rect stack, so clips are
//!   implemented as a full-surface [`tiny_skia::Mask`] rasterized from
//!   the intersected clip rect whenever the stack changes. The buffer
//!   is persistent (reused across ops and frames) and updated
//!   incrementally — only the pixels whose clip membership changed are
//!   rewritten — and a clip covering the whole surface is handed to
//!   tiny-skia as no mask at all, so its per-pixel mask read is
//!   skipped. Exact for rect clips.
//! - Opacity: `PushOpacity` renders an isolated group. Groups with
//!   opacity 1.0 are no-ops, groups with opacity 0.0 paint nothing, and
//!   groups holding a single paint item (no nested opacity groups) keep
//!   the old fast path of folding the opacity into per-item alpha —
//!   exactly equivalent to isolated compositing for one item. Anything
//!   else (overlapping items, nested groups) renders into an offscreen
//!   pixmap at its own alphas and is composited with the group opacity,
//!   matching CSS group semantics: overlapping translucent items inside
//!   the group composite against each other first, then the whole group
//!   fades as one.
//! - Transforms: `PushTransform` renders an isolated group
//!   untransformed into an offscreen pixmap, then warps it through the
//!   2D matrix onto the target (bilinear, edge-clamped). The identity
//!   matrix paints inline; a non-invertible matrix paints nothing (per
//!   spec). Clips pushed outside the group are drained before painting
//!   and applied per-pixel to the warped result, so ancestor clips see
//!   transformed pixels.
//! - Shadows: `BoxShadow` rasterizes the spread-adjusted rounded rect
//!   (inflated for outset, deflated for inset) to a coverage buffer,
//!   blurs it with a multi-pass box blur (≈ gaussian σ = blur/2), and
//!   composites it in the shadow color; inset shadows knock out the
//!   sharp border box first (`S − B`). `TextShadow` reuses the text
//!   path at the shadow offset, blurring through a temp pixmap when
//!   the blur radius is positive.
//! - `Decoration` line placement from the ink box (the display list
//!   carries no baseline in the item): underline at the ink box's bottom
//!   edge, overline at its top edge, line-through through its vertical
//!   middle. Wavy variants share their straight counterpart's baseline
//!   and are tessellated as sine waves by `decoration_geometry`.
//! - `ListMarker` glyph markers are centered in their rect with diameter
//!   `min(rect.w, rect.h)`; `Decimal`/`DecimalLeadingZero` render the
//!   1-based index as text through the text path (the item's font), at the
//!   GPU's size and baseline.
//! - `RepeatImage` tiling is expanded here in the rasterizer: the item
//!   carries the area, tile size, repeat mode, and offset, and this
//!   backend tiles the image across the (clamped) area with the same
//!   grid the old per-tile expansion used — no per-tile display items.
//!   `BackgroundImage` items are still stretched to their `rect` like
//!   replaced-element images.

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::collections::hash_map::Entry as HashMapEntry;
use std::rc::Rc;
use std::sync::Arc;

use bowser_layout::{Rect, Size};
use tiny_skia::Rect as TsRect;
use tiny_skia::{IntSize, Mask, Paint, Pixmap, PixmapPaint, Transform};

use crate::borders::{
    decoration_geometry, marker_geometry, marker_text, tessellate_border_into,
    tessellate_border_rounded_into,
};
use crate::clip::{CORNERS, MAX_POLY_VERTS, clip_stack_coverage_byte};
use crate::display_list::{
    Backend, BackgroundRepeat, BlendMode, Color, DecoKind, DisplayItem, DisplayList,
    EDGE_GLOW_COLOR, Edge4, FontData, FontKey, ImageId, MAX_REPEAT_TILES, MarkerKind, Quad,
    background_space_axis, background_tile_grid, cap_repeat_tile, clamp_repeat_area,
    edge_glow_alpha,
};
use crate::filter::{
    FilterOp, FilterPass, apply_color_matrix_rgba8, apply_drop_shadow, filter_spill,
    plan_filter_passes,
};
use crate::gradient::{Gradient, bake_gradient_rgba8};
use crate::text::{GlyphAtlas, RasterGlyph, rounded_size, shape_text};

/// Straight (non-premultiplied) RGBA8 frame at CSS-px scale, produced by
/// [`Backend::paint`].
// Backing store for `last_frame` (intentionally-public app API, see
// below); covered by the `backend_trait_paint_stores_frame` test.
#[allow(dead_code)]
struct Frame {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
}

/// One cached decoded image: premultiplied pixmap plus the content
/// identity it was decoded from. The pixmap is an [`Arc`] so the
/// content-hash-keyed shared cache ([`ImageCache::shared`]) and the
/// per-id front entries can name one decode without copying pixels.
///
/// `hash` alone is never trusted as content identity: FNV-1a 64 is not
/// collision-resistant, so a front-cache hit additionally requires the
/// stored `width`/`height`/`byte_len` to match the image's current
/// source (see [`matches`](Self::matches)). A hash hit with mismatched
/// shape or length is treated as a miss and decoded fresh.
struct CpuImageEntry {
    pixmap: Arc<Pixmap>,
    hash: u64,
    width: u32,
    height: u32,
    byte_len: usize,
}

impl CpuImageEntry {
    /// Hit verification: every component of the stored content identity
    /// must match the image's current source before the cached pixmap
    /// is served.
    fn matches(&self, hash: u64, width: u32, height: u32, byte_len: usize) -> bool {
        self.hash == hash
            && self.width == width
            && self.height == height
            && self.byte_len == byte_len
    }
}

/// One cross-id shared decode: the pixmap plus the shape it was
/// decoded from. The shared map is keyed by content hash alone, so a
/// hit must still verify `width`/`height` before the pixmap is served:
/// an FNV-1a 64 collision naming a different-shaped image is treated
/// as a miss and decoded fresh (replacing the stale entry).
struct SharedDecode {
    pixmap: Arc<Pixmap>,
    width: u32,
    height: u32,
}

/// All image decode caching state for the CPU backend, owned by the
/// backend and borrowed by the per-frame [`Painter`].
///
/// Three layers, cheapest first:
/// 1. `by_id`: per-[`ImageId`] front cache; a hit is validated by the
///    entry's content hash and replaced on mismatch (ids are only
///    unique within one [`DisplayList`]).
/// 2. The content hash itself is O(1): [`DisplayList::image_content_hash`]
///    caches [`DecodedImage::content_hash`] at registration, so no
///    pixel-buffer rehash per frame (tiled backgrounds emit one item per
///    tile).
/// 3. `shared`: content-hash-keyed decoded pixmaps, shared across
///    [`ImageId`]s — a rebuilt display list minting fresh ids for the
///    same bytes decodes once, not once per id. Bounded by FIFO
///    eviction (see [`MAX_SHARED_DECODED_IMAGES`] and
///    [`MAX_DECODED_IMAGE_BYTES`]). The key is the hash alone, so each
///    entry also stores the shape it was decoded from and a hit is
///    only served when the dimensions match (a hash collision with a
///    different shape decodes fresh instead of serving wrong pixels).
///
/// The byte budget covers *both* `shared` and the `by_id` front
/// entries: every image referenced by the current frame holds an
/// independent `Arc<Pixmap>` in `by_id`, which survives `shared`
/// eviction by design, so bounding `shared` alone would not bound
/// retained decoded memory. `by_id` bytes are counted conservatively
/// (independently of `shared_bytes` even when both name the same
/// `Arc`), so `shared_bytes + by_id_bytes` is an upper bound on the
/// decoded memory actually retained.
///
/// Thread-safety: the backend is `!Send` (the font hook is an [`Rc`]),
/// so this state is only ever touched through `&mut` borrows on the
/// painting thread — no locking needed.
#[derive(Default)]
struct ImageCache {
    by_id: HashMap<ImageId, CpuImageEntry>,
    shared: HashMap<u64, SharedDecode>,
    /// Insertion order of `shared`, for FIFO eviction.
    shared_order: VecDeque<u64>,
    /// Bytes currently pinned by `shared` (w * h * 4 per entry).
    shared_bytes: usize,
    /// Bytes currently pinned by `by_id` front entries (w * h * 4 per
    /// entry; see the budget discussion above).
    by_id_bytes: usize,
    /// Insertion order of `by_id`, for FIFO eviction. Each live id
    /// appears exactly once: replaces refresh recency by moving to the
    /// back (see [`refresh_by_id_order`](Self::refresh_by_id_order)),
    /// and frame-start eviction prunes it alongside `by_id`.
    by_id_order: VecDeque<ImageId>,
}

impl ImageCache {
    /// Record an insert into `by_id_order`: fresh ids push to the back,
    /// and a replaced id moves to the back (recency refresh) so the
    /// queue stays in true insertion order. This matters for
    /// [`evict_over_budget`](Self::evict_over_budget): the protected
    /// id is always the just-inserted one, so it must sit at the back
    /// and never be popped before older, evictable entries — otherwise
    /// eviction re-queues it and stops with the combined total still
    /// over budget by more than one image of slack.
    fn refresh_by_id_order(&mut self, id: ImageId, fresh: bool) {
        if !fresh && let Some(pos) = self.by_id_order.iter().position(|&x| x == id) {
            self.by_id_order.remove(pos);
        }
        self.by_id_order.push_back(id);
    }

    /// Saturating decrement of the `by_id_bytes` pin counter. The
    /// invariant `by_id_bytes == sum(entry.byte_len)` is maintained by
    /// hand at every insert/remove site, so a plain `-=` would panic in
    /// debug (or wrap to `usize::MAX` in release) if a site ever
    /// desynced. Two desync shapes behave differently: a transient
    /// one-off oversubtraction only skews the budget bound by a bounded
    /// amount in either direction (the counter feeds a budget check, not
    /// an allocation), while a systematic desync — a future asymmetric
    /// site firing every frame — pins the counter near zero in release,
    /// so the accounted total stays under budget while real decoded
    /// memory grows silently toward OOM. Debug catches either shape
    /// loudly via the assertion below; release saturates instead of
    /// wrapping, and the saturation path logs (see the `eprintln!`
    /// below) so a systematic desync is observable rather than silent.
    fn sub_by_id_bytes(&mut self, n: usize) {
        debug_assert!(self.by_id_bytes >= n, "by_id_bytes underflow: {n}");
        if n > self.by_id_bytes {
            eprintln!(
                "[bowser_paint] image cache by_id_bytes desynced (sub {n} > {total}); saturating instead of wrapping",
                total = self.by_id_bytes
            );
        }
        self.by_id_bytes = self.by_id_bytes.saturating_sub(n);
    }

    /// Saturating decrement of the `shared_bytes` pin counter. The
    /// invariant `shared_bytes == sum(decoded_bytes(entry.pixmap))` is
    /// maintained by hand at every insert/remove site, but unlike
    /// `by_id_bytes` the two sides use different expressions: inserts
    /// add the source buffer length (`decoded.rgba.len()`), while
    /// removals recompute `w * h * 4` from the stored pixmap. The two
    /// agree today because the insert path rejects sources whose length
    /// mismatches `w * h * 4`, and tiny-skia pixmaps always hold exactly
    /// `w * h * 4` bytes — but any future insert path bypassing that
    /// guard would desync this counter while `by_id_bytes` stays exact.
    /// Same two-shape story as
    /// [`sub_by_id_bytes`](Self::sub_by_id_bytes): a transient desync
    /// skews the budget bound by a bounded amount either way; a
    /// systematic one pins the counter near zero in release and grows
    /// decoded memory silently toward OOM. Debug asserts loudly;
    /// release saturates and logs.
    fn sub_shared_bytes(&mut self, n: usize) {
        debug_assert!(self.shared_bytes >= n, "shared_bytes underflow: {n}");
        if n > self.shared_bytes {
            eprintln!(
                "[bowser_paint] image cache shared_bytes desynced (sub {n} > {total}); saturating instead of wrapping",
                total = self.shared_bytes
            );
        }
        self.shared_bytes = self.shared_bytes.saturating_sub(n);
    }

    /// Enforce the decode-memory bounds after a front-entry insert.
    ///
    /// `protected` is the id just inserted: it is never evicted (the
    /// caller is about to paint from it), so afterwards
    /// `shared_bytes + by_id_bytes <= MAX_DECODED_IMAGE_BYTES` plus at
    /// most one image. Cross-frame `shared` entries are evicted before
    /// per-id entries (this frame's working set), both in FIFO order.
    /// Replaces refresh recency (see
    /// [`refresh_by_id_order`](Self::refresh_by_id_order)), so the
    /// protected id is always the queue's newest and the re-queue below
    /// only triggers when it is the sole survivor.
    fn evict_over_budget(&mut self, protected: ImageId) {
        while self.shared.len() > MAX_SHARED_DECODED_IMAGES {
            let Some(old) = self.shared_order.pop_front() else {
                break;
            };
            if let Some(evicted) = self.shared.remove(&old) {
                self.sub_shared_bytes(decoded_bytes(&evicted.pixmap));
            }
        }
        while self.shared_bytes + self.by_id_bytes > MAX_DECODED_IMAGE_BYTES {
            if let Some(old) = self.shared_order.pop_front() {
                if let Some(evicted) = self.shared.remove(&old) {
                    self.sub_shared_bytes(decoded_bytes(&evicted.pixmap));
                }
                continue;
            }
            let Some(id) = self.by_id_order.pop_front() else {
                break;
            };
            if id == protected {
                // Sole survivor: re-queue at the back and stop;
                // everything older is already gone.
                self.by_id_order.push_back(id);
                break;
            }
            if let Some(entry) = self.by_id.remove(&id) {
                self.sub_by_id_bytes(entry.byte_len);
            }
        }
        // Debug-only: the hand-maintained counter must exactly match the
        // map contents after eviction.
        debug_assert_eq!(
            self.shared_bytes,
            self.shared
                .values()
                .map(|e| decoded_bytes(&e.pixmap))
                .sum::<usize>(),
            "shared_bytes desynced from shared contents after evict_over_budget"
        );
    }
}

/// Decoded RGBA8 bytes held by a pixmap.
fn decoded_bytes(pm: &Pixmap) -> usize {
    pm.width() as usize * pm.height() as usize * 4
}

/// Lifetime counters for the image decode cache and the opacity-group
/// pixmap pool. Cheap integer bumps on cache paths, off the pixel hot
/// path; read by tests to measure hit rates and pool reuse.
#[derive(Default, Debug)]
struct CacheStats {
    /// Real premultiply decodes performed.
    image_decodes: u64,
    /// Per-id front-cache hits (same id, same content).
    image_id_hits: u64,
    /// Content-hash hits across different ids (cross-id dedup).
    image_shared_hits: u64,
    /// Opacity-group pixmap checkouts.
    pool_checkouts: u64,
    /// Checkouts satisfied from the pool, not freshly allocated.
    pool_reuses: u64,
    /// Gradient rasters baked (cache misses).
    gradient_bakes: u64,
    /// Gradient paints served from the bake cache (cache hits).
    gradient_bake_hits: u64,
}

/// Max entries in the shared content-hash decode cache. The per-id
/// front cache is bounded by the frame-start eviction of unreferenced
/// ids, but the shared cache is cross-frame persistent, so it needs
/// its own bound: FIFO eviction drops the oldest content first.
const MAX_SHARED_DECODED_IMAGES: usize = 512;

/// Byte budget for decoded-image memory retained by the image cache:
/// the content-hash-keyed `shared` map *plus* the per-id `by_id` front
/// entries. Entry-count alone does not bound memory (one entry can be
/// a multi-megapixel pixmap), and bounding `shared` alone does not
/// either: every image referenced by the current frame also holds an
/// independent `Arc<Pixmap>` in `by_id`, which survives `shared`
/// eviction by design. `by_id` bytes are counted conservatively
/// (independently of `shared_bytes` even when both name the same
/// `Arc`), so `shared_bytes + by_id_bytes` is an upper bound on the
/// decoded memory actually retained; eviction runs while the combined
/// total (or [`MAX_SHARED_DECODED_IMAGES`]) is exceeded.
///
/// Slack: the entry inserted by the in-progress lookup is never
/// evicted, so after any lookup the accounted total is at most this
/// budget plus one image.
const MAX_DECODED_IMAGE_BYTES: usize = 64 * 1024 * 1024;

/// Thread-local font-bytes cache for [`Painter::paint_text_run`].
///
/// The hook clones the whole font byte buffer on every call, so bytes are
/// resolved once per [`FontKey`] and shared across text runs through an
/// `Rc` (one refcount bump per run instead of a multi-hundred-KB memcpy).
/// Thread-local: each backend thread resolves independently.
///
/// `FontKey`s are minted sequentially per [`FontCache`](crate::fonts::FontCache)
/// instance, so the same key can name different faces in different caches.
/// Entries are therefore validated against the installing hook's identity:
/// when the hook changes, every entry is flushed before resolving. The
/// installing hook is retained so its allocation cannot be recycled for a
/// different closure (which would defeat the `Rc::ptr_eq` identity check).
/// Entries are bounded by distinct `FontKey`s (one key per resolved face),
/// so the cache cannot grow without bound.
///
/// Reentrancy: the hook is caller-supplied and may re-enter text painting
/// on this thread, so callers use the two-phase [`get`](Self::get)/
/// [`insert`](Self::insert) (hook runs outside the `RefCell` borrow);
/// never invoke the hook while the borrow is held.
struct FontDataCache {
    hook: Option<Rc<dyn Fn(FontKey) -> Option<FontData>>>,
    entries: HashMap<FontKey, Rc<FontData>>,
}

impl FontDataCache {
    fn new() -> Self {
        Self {
            hook: None,
            entries: HashMap::new(),
        }
    }

    /// Install `hook`, flushing all entries when it changed. Factored out
    /// so [`get`](Self::get) and [`insert`](Self::insert) validate the
    /// hook identity identically.
    fn install(&mut self, hook: &Rc<dyn Fn(FontKey) -> Option<FontData>>) {
        if !self.hook.as_ref().is_some_and(|h| Rc::ptr_eq(h, hook)) {
            self.entries.clear();
            self.hook = Some(Rc::clone(hook));
        }
    }

    /// Cached entry for `key` without invoking the hook. Never calls out,
    /// so it is safe to hold the cache borrow across it.
    fn get(
        &mut self,
        hook: &Rc<dyn Fn(FontKey) -> Option<FontData>>,
        key: FontKey,
    ) -> Option<Rc<FontData>> {
        self.install(hook);
        self.entries.get(&key).map(Rc::clone)
    }

    /// Store hook-resolved `data` for `key`, re-validating the hook
    /// identity first: the hook ran outside the cache borrow and may have
    /// re-entered and installed a different hook, so a blind insert could
    /// file another hook's bytes under this one.
    fn insert(
        &mut self,
        hook: &Rc<dyn Fn(FontKey) -> Option<FontData>>,
        key: FontKey,
        data: FontData,
    ) -> Rc<FontData> {
        self.install(hook);
        let shared = Rc::new(data);
        self.entries.insert(key, Rc::clone(&shared));
        shared
    }
}

/// Regression-test scale for the CPU image cache. Display lists are
/// rebuilt every frame with per-list [`ImageId`]s; the frame-start
/// `retain` in [`render_to_rgba_full`](Self::render_to_rgba_full) evicts
/// ids the incoming list does not reference, so the cache holds at most
/// the frame's distinct referenced images and needs no capacity clear
/// (mirrors the GPU backend's `MAX_CACHED_IMAGES` convention).
#[cfg(test)]
const MAX_CACHED_CPU_IMAGES: usize = 512;

/// CPU rasterizer backend. Renders a [`DisplayList`] with tiny-skia.
pub struct CpuBackend {
    get_font_data: Rc<dyn Fn(FontKey) -> Option<FontData>>,
    frame: Option<Frame>,
    /// CPU glyph coverage cache, persistent across frames (see
    /// [`Painter`]). Bounded by the eviction policy in
    /// [`render_to_rgba_full`](Self::render_to_rgba_full).
    glyphs: GlyphAtlas,
    /// Atlas misses since the last eviction; bounds `glyphs` growth.
    glyph_misses: usize,
    /// Persistent full-surface clip-mask buffer, reused across clip
    /// ops and frames (grow-only; see
    /// [`Painter::update_clip_mask`]).
    clip_mask: Option<Mask>,
    /// The opaque region `clip_mask`'s bytes currently encode (`None`
    /// = all zeros, i.e. the empty clip). [`Mask::new`] zero-fills, so
    /// this starts at `None`, and [`Painter::update_clip_mask`] keeps
    /// it exact across clip ops and frames — clip updates then rewrite
    /// only the bytes whose clip membership changed instead of the
    /// whole surface.
    mask_region: Option<MaskRect>,
    /// The clip stack `clip_mask`'s bytes currently encode, entry-wise
    /// (`[]` = the buffer is all zeros). Fractional clip edges change
    /// boundary bytes without changing the bounding `mask_region`, so
    /// the update's early-return compares this, not the region; for
    /// all-rect stacks the comparison reduces to the intersected bound
    /// (see [`Painter::update_clip_mask`]).
    mask_stack: Vec<ClipEntry>,
    /// Image decode caches (per-id front and content-hash-keyed shared
    /// decodes), persistent across frames; see [`ImageCache`].
    image_cache: ImageCache,
    /// Lifetime cache/pool counters, read by tests (see
    /// [`CacheStats`]).
    cache_stats: CacheStats,
    /// Free-list of bounds-sized pixmaps for isolated opacity groups,
    /// checked out by [`Painter::paint_opacity_group`] and returned
    /// after the group's composite — one allocation per concurrent
    /// group size, not one per group per frame (mirrors the GPU
    /// backend's `group_target_pool`, capped at
    /// [`MAX_POOLED_GROUP_PIXMAPS`]). Borrowed into the [`Painter`];
    /// entries are cleared to transparent on checkout and stale sizes
    /// are dropped at frame start.
    group_pixmap_pool: Vec<Pixmap>,
    /// Reusable byte scratch for filter passes (blur temp, drop-shadow
    /// tint buffer), grown to the largest group pixmap seen — one
    /// allocation per size class, not one per filtered element per
    /// frame. Borrowed into the [`Painter`].
    filter_scratch: Vec<u8>,
    /// Free-list of scratch byte buffers for shadow temps (box-shadow
    /// coverage mask and blur temp, sharp knockout mask, text-shadow
    /// RGBA blur scratch), checked out by exact length by the shadow
    /// painters and returned when the shadow is done — one allocation
    /// per (shadow, size class), not one per shadow per frame.
    /// Borrowed into the [`Painter`]; capped at
    /// [`MAX_POOLED_SHADOW_SCRATCH`] buffers of at most
    /// [`MAX_POOLED_SHADOW_SCRATCH_BYTES`] bytes each, so a one-frame
    /// spike of many shadows does not pin megabytes across all later
    /// frames.
    shadow_scratch_pool: Vec<Vec<u8>>,
    /// Reusable scratch for border tessellation: [`Painter`] clears
    /// and fills this with [`tessellate_border_into`] instead of
    /// allocating a fresh `Vec<Quad>` per bordered box per paint.
    /// Borrowed into the [`Painter`]; grows to the largest single
    /// border's quad count and is bounded by the tessellator's own
    /// `MAX_BORDER_PATTERN_QUADS` cap, so no high-water trim is
    /// needed.
    border_scratch: Vec<Quad>,
    /// Baked gradient rasters by [`gradient_bake_key`], persistent
    /// across frames; see [`CpuGradientBake`]. Borrowed into the
    /// [`Painter`]; entries this frame does not stamp are evicted at
    /// the frame boundary in
    /// [`render_to_rgba_full`](Self::render_to_rgba_full).
    gradient_bake_cache: HashMap<u64, CpuGradientBake>,
    /// Frame generation, bumped once per
    /// [`render_to_rgba_full`](Self::render_to_rgba_full): cache
    /// entries stamped with it survive the frame-boundary eviction.
    gradient_frame: u64,
    /// Reusable scratch for the referenced-image-id set used by the
    /// frame-start image-cache eviction in
    /// [`render_to_rgba_full`](Self::render_to_rgba_full): cleared at
    /// frame start instead of allocating a fresh `HashSet` per frame
    /// (H5). Must not be borrowed across any call that can reenter
    /// `render_to_rgba_full` (the font hook and image-decoding paths
    /// can run caller code); capacity is trimmed when disproportionate
    /// at the frame boundary so a one-frame spike does not pin peak
    /// memory for the backend's lifetime.
    evict_referenced: HashSet<ImageId>,
    /// Reusable scratch for rebuilding the cache FIFO queue
    /// (`image_cache.by_id_order`) after the frame-start eviction:
    /// live ids drain into this buffer and swap back, so no
    /// `VecDeque` is allocated per frame (H5). Must not be borrowed
    /// across any call that can reenter
    /// [`render_to_rgba_full`](Self::render_to_rgba_full); capacity is
    /// trimmed when disproportionate at the frame boundary, same as
    /// [`evict_referenced`](Self::evict_referenced).
    evict_order_buf: VecDeque<ImageId>,
}

/// Max pooled group pixmaps kept for isolated opacity groups. A
/// one-frame spike of many groups must not pin dozens of buffers
/// across all later frames (mirrors the GPU backend's
/// `MAX_POOLED_GROUP_TARGETS = 8`).
const MAX_POOLED_GROUP_PIXMAPS: usize = 8;

/// Max pooled shadow-scratch buffers kept (see
/// [`CpuBackend::shadow_scratch_pool`]). One box shadow holds three
/// temps at once (coverage, blur temp, sharp mask), so the cap leaves
/// room for two same-size shadows in flight plus the text-shadow RGBA
/// size classes.
const MAX_POOLED_SHADOW_SCRATCH: usize = 8;

/// Max bytes of one pooled shadow-scratch buffer. Shadow temps can
/// reach [`MAX_SHADOW_TEMP_DIM`]<sup>2</sup> (coverage) or 4× that
/// (RGBA) — 16–64 MiB — which must not be pinned across frames; only
/// buffers at or below this cap are pooled, larger ones allocate and
/// drop per use exactly as before pooling.
const MAX_POOLED_SHADOW_SCRATCH_BYTES: usize = 4 * 1024 * 1024;

/// Max atlas misses before the whole glyph cache is evicted. Steady
/// pages stop missing after the first frame, so this only fires on
/// pages churning through thousands of distinct glyphs.
const GLYPH_ATLAS_CAP: usize = 4096;

/// Max bake dimension (device px, per axis) for one gradient item. A
/// hostile page-px rect must not allocate gigabytes; gradient data is
/// smooth color, so downscaling a huge rect to the cap and stretching
/// it back through the pixmap transform is visually lossless.
const MAX_GRADIENT_BAKE_DIM: u32 = 4096;

/// One cached CPU gradient bake: straight-RGBA8 bytes at the capped
/// device size, stamped with the frame that last used it.
struct CpuGradientBake {
    bytes: Vec<u8>,
    width: u32,
    height: u32,
    /// [`CpuBackend::gradient_frame`] at last use; entries not stamped
    /// by the current frame are evicted at the frame boundary, so the
    /// cache holds at most the frame's distinct gradients (mirrors the
    /// WGPU backend's `gradient_cache` / `last_frame` eviction).
    last_frame: u64,
}

/// Bake-cache key: the gradient's content hash mixed with the bake
/// dimensions (one gradient at two sizes is two bakes). Mirrors the
/// WGPU backend's gradient-texture key.
fn gradient_bake_key(gradient: &Gradient, w: u32, h: u32) -> u64 {
    let mut key = gradient.content_hash();
    for b in w.to_le_bytes().into_iter().chain(h.to_le_bytes()) {
        key ^= u64::from(b);
        key = key.wrapping_mul(0x0100_0000_0000_01b3);
    }
    key
}

impl CpuBackend {
    /// `get_font_data` resolves a [`FontKey`] to raw font bytes for the
    /// text worker's shaper/rasterizer (pending; see
    /// [`Painter::paint_text_run`]).
    pub fn new(get_font_data: Rc<dyn Fn(FontKey) -> Option<FontData>>) -> Self {
        Self {
            get_font_data,
            frame: None,
            glyphs: GlyphAtlas::new(),
            glyph_misses: 0,
            clip_mask: None,
            mask_region: None,
            mask_stack: Vec::new(),
            image_cache: ImageCache::default(),
            cache_stats: CacheStats::default(),
            group_pixmap_pool: Vec::new(),
            filter_scratch: Vec::new(),
            shadow_scratch_pool: Vec::new(),
            border_scratch: Vec::new(),
            gradient_bake_cache: HashMap::new(),
            gradient_frame: 0,
            evict_referenced: HashSet::new(),
            evict_order_buf: VecDeque::new(),
        }
    }

    /// Render `list` to straight RGBA8 row-major pixels.
    ///
    /// `width_px`/`height_px` are the device-pixel target size, `scale` is
    /// the device-pixel ratio applied to CSS-px display-list coordinates,
    /// and `scroll` is the scroll offset in CSS px.
    pub fn render_to_rgba_full(
        &mut self,
        list: &DisplayList,
        width_px: u32,
        height_px: u32,
        scale: f32,
        scroll: (f32, f32),
    ) -> Vec<u8> {
        if width_px == 0 || height_px == 0 {
            return Vec::new();
        }
        // Bump the gradient-bake generation: entries this frame's
        // paints stamp get to survive the frame-boundary eviction
        // below (mirrors the WGPU backend's `gradient_frame`).
        self.gradient_frame = self.gradient_frame.wrapping_add(1);
        // Frame-start image-cache eviction (mirrors the GPU backend):
        // `ImageId`s are only unique within one `DisplayList`, so entries
        // for ids this frame does not reference are stale cross-frame
        // accumulations. Retaining only referenced ids bounds the cache at
        // the frame's distinct image count without the old
        // clear-at-capacity, which wiped hot entries for still-visible
        // images and forced a full re-decode once the 513th distinct image
        // decoded.
        {
            // Hoisted per-frame scratch (H5): both buffers are cleared
            // and reused across frames instead of allocated fresh here.
            let referenced = &mut self.evict_referenced;
            referenced.clear();
            for item in &list.items {
                match item {
                    DisplayItem::Image { image, .. }
                    | DisplayItem::BackgroundImage { image, .. }
                    | DisplayItem::RepeatImage { image, .. } => {
                        referenced.insert(*image);
                    }
                    _ => {}
                }
            }
            let cache = &mut self.image_cache;
            let mut evicted_bytes = 0usize;
            cache.by_id.retain(|id, entry| {
                if referenced.contains(id) {
                    true
                } else {
                    evicted_bytes += entry.byte_len;
                    false
                }
            });
            cache.sub_by_id_bytes(evicted_bytes);
            // Debug-only: the hand-maintained counter must exactly match
            // the map contents after the frame-start retain.
            debug_assert_eq!(
                cache.by_id_bytes,
                cache.by_id.values().map(|e| e.byte_len).sum::<usize>(),
                "by_id_bytes desynced from by_id contents after frame-start eviction"
            );
            // Keep the FIFO eviction queue exact: ids evicted above must
            // not linger, or the queue would grow without bound across
            // frames (one push per insert, no pops while under budget).
            // Live ids drain into the reused scratch buffer and swap
            // back, preserving order exactly like the old
            // take/filter/collect.
            let order_buf = &mut self.evict_order_buf;
            order_buf.clear();
            let live = &cache.by_id;
            let by_id_order = &mut cache.by_id_order;
            for id in by_id_order.drain(..) {
                if live.contains_key(&id) {
                    order_buf.push_back(id);
                }
            }
            std::mem::swap(order_buf, by_id_order);
        }
        // High-water-mark hygiene for the hoisted per-frame scratch
        // (P3): `clear()` retains capacity, so a one-frame spike of
        // many distinct images would otherwise pin that peak capacity
        // for the backend's lifetime. Shrink when capacity is
        // disproportionate to the steady-state contents — pure
        // capacity arithmetic, so the common path allocates nothing.
        // (The deque still holds the pre-eviction queue contents here;
        // the next frame clears it first, and shrinking against that
        // stale length still releases the spike's excess.)
        if self.evict_referenced.capacity() > 4 * self.evict_referenced.len().max(64) {
            self.evict_referenced.shrink_to_fit();
        }
        if self.evict_order_buf.capacity() > 4 * self.evict_order_buf.len().max(64) {
            self.evict_order_buf.shrink_to_fit();
        }
        // Per-frame state reset (mirrors the GPU backend): the image
        // Pooled group pixmaps are bounds-sized (see
        // `Painter::paint_opacity_group`), so the pool holds many sizes
        // and no size is "stale" after a resize; the cap below is what
        // bounds it (see `render_into`'s `group_target_pool` handling).
        // The content-hash-keyed shared decode cache is deliberately
        // *not* cleared: it is bounded by its own FIFO eviction, and
        // clearing it would re-decode identical images every frame.
        while self.group_pixmap_pool.len() > MAX_POOLED_GROUP_PIXMAPS {
            self.group_pixmap_pool.pop();
        }
        let mut px = match Pixmap::new(width_px, height_px) {
            Some(px) => px,
            None => return Vec::new(),
        };
        {
            let mut painter = Painter::new(
                &mut px,
                &self.get_font_data,
                &mut self.glyphs,
                &mut self.glyph_misses,
                &mut self.clip_mask,
                &mut self.mask_region,
                &mut self.mask_stack,
                &mut self.image_cache,
                &mut self.cache_stats,
                &mut self.group_pixmap_pool,
                &mut self.filter_scratch,
                &mut self.shadow_scratch_pool,
                &mut self.border_scratch,
                &mut self.gradient_bake_cache,
                self.gradient_frame,
                scale,
                scroll,
            );
            painter.paint_list(list);
        }
        // Gradient-bake eviction (mirrors the WGPU backend): the cache
        // holds at most the frame's distinct gradients. Entries this
        // frame did not stamp are stale — gradient content is keyed by
        // value, so a stale entry can never be needed again.
        {
            let frame = self.gradient_frame;
            self.gradient_bake_cache
                .retain(|_, entry| entry.last_frame == frame);
        }
        // Eviction policy: the atlas persists across frames, so bound it.
        // Misses only accumulate for newly seen glyphs; when they pass
        // the cap the whole cache is dropped (same "evict all"
        // convention as the GPU atlas) and hot glyphs re-rasterize on
        // demand.
        if self.glyph_misses > GLYPH_ATLAS_CAP {
            self.glyphs = GlyphAtlas::new();
            self.glyph_misses = 0;
        }
        unpremultiply(px.take())
    }

    /// Static 3-argument bridge for `crate::render`
    /// (`render_layout_to_rgba`): renders at CSS-px scale with no scroll.
    ///
    /// Font data is unavailable here, so text runs paint nothing on this
    /// path; use [`CpuBackend::new`] plus [`CpuBackend::render_to_rgba_full`]
    /// when font data exists.
    pub fn render_to_rgba(list: &DisplayList, width_px: u32, height_px: u32) -> Vec<u8> {
        Self::new(Rc::new(|_| None)).render_to_rgba_full(list, width_px, height_px, 1.0, (0.0, 0.0))
    }

    /// Present straight-RGBA8 pixels into a softbuffer buffer.
    ///
    /// NOTE: the sketched signature `surface: &softbuffer::Surface` cannot
    /// be written: `softbuffer::Surface<D, W>` is generic over the display
    /// and window handle types, `buffer_mut` requires `&mut`, and naming
    /// the `HasDisplayHandle`/`HasWindowHandle` bounds would need a
    /// `raw-window-handle` dependency, which this crate must not add
    /// (Cargo.toml is frozen). Callers do:
    ///
    /// ```ignore
    /// let mut buffer = surface.buffer_mut().expect("surface buffer");
    /// backend.present(&mut buffer, &rgba, width_px, height_px);
    /// buffer.present().expect("present");
    /// ```
    ///
    /// Pixels are packed as softbuffer requires: `0x00RRGGBB` row-major.
    /// The alpha channel is dropped (surfaces are opaque).
    // Intentionally-public app API: the app worker presents into its
    // softbuffer surface through this (covered by the `present_packs_xrgb`
    // test; no in-tree caller wires it up yet).
    #[allow(dead_code)]
    pub fn present(&self, buffer: &mut [u32], rgba: &[u8], width_px: u32, height_px: u32) {
        // Saturating: an overflowing product must not wrap into a small
        // `want` that silently under-presents the frame.
        let want = (width_px as usize).saturating_mul(height_px as usize);
        let n = want.min(buffer.len()).min(rgba.len() / 4);
        let (chunks, _remainder) = rgba.as_chunks::<4>();
        for (dst, src) in buffer.iter_mut().zip(chunks).take(n) {
            *dst = (u32::from(src[0]) << 16) | (u32::from(src[1]) << 8) | u32::from(src[2]);
        }
    }

    /// The frame produced by the last [`Backend::paint`] call, as
    /// `(straight RGBA8 pixels, width, height)`.
    // Intentionally-public app API: exposes the last painted frame to the
    // app worker (covered by the `backend_trait_paint_stores_frame` test;
    // no in-tree caller wires it up yet).
    #[allow(dead_code)]
    pub fn last_frame(&self) -> Option<(&[u8], u32, u32)> {
        self.frame
            .as_ref()
            .map(|f| (f.rgba.as_slice(), f.width, f.height))
    }
}

impl Backend for CpuBackend {
    /// Render `list` to an internal pixmap at CSS-px scale (scale = 1.0).
    /// `viewport` is the visible page area in CSS px, per the trait docs.
    fn paint(&mut self, list: &DisplayList, viewport: Size, scroll: (f32, f32)) {
        let w = viewport.w.max(1.0).round() as u32;
        let h = viewport.h.max(1.0).round() as u32;
        let rgba = self.render_to_rgba_full(list, w, h, 1.0, scroll);
        self.frame = Some(Frame {
            rgba,
            width: w,
            height: h,
        });
    }
}

/// Convert a 0..1 f32 channel to u8.
fn u8c(c: f32) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Build a solid-color paint with the opacity folded into alpha.
/// Returns `None` when the result is fully transparent.
fn paint_for(color: Color, opacity: f32) -> Option<Paint<'static>> {
    // `r8()/g8()/b8()` are exactly what `u8c(color.r())` computed before
    // (the pack/unpack roundtrip is exact), minus the float detour.
    let a = u8c(color.a() * opacity);
    if a == 0 {
        return None;
    }
    let mut paint = Paint::default();
    paint.set_color_rgba8(color.r8(), color.g8(), color.b8(), a);
    Some(paint)
}

/// Cap on isolated opacity groups per frame, shared by both backends:
/// each isolated group pins a full-target offscreen surface until it
/// composites, and nested groups recurse, so the cap bounds transient
/// memory (GPU) and stack depth (both). Past the cap the group opacity
/// folds into the running product — exact only for non-overlapping
/// primitives, a documented degradation for pathological pages versus
/// unbounded memory or stack overflow.
pub(crate) const MAX_OPACITY_GROUPS: u32 = 32;

/// Scan from `idx` (a `PushOpacity`) to its matching `PopOpacity`,
/// honoring nesting. Returns `(end, paint_items, has_nested_group,
/// single_item)`: `end` is the index of the matching `PopOpacity`, or
/// `items.len()` when the list is unbalanced; `paint_items` counts
/// paint-producing items at the group's top level (clips don't count,
/// and items inside nested groups count toward the nested group, not
/// the outer one); `has_nested_group` reports a nested `PushOpacity`
/// anywhere inside; `single_item` is the top-level paint item's index
/// when `paint_items == 1`, else `None`.
///
/// Shared with the wgpu backend, which lowers the same groups to
/// offscreen textures.
pub(crate) fn scan_opacity_group(
    items: &[DisplayItem],
    idx: usize,
) -> (usize, usize, bool, Option<usize>) {
    let mut depth = 0usize;
    let mut paint_items = 0usize;
    let mut single_item = None;
    let mut nested = false;
    let mut end = idx + 1;
    while end < items.len() {
        match &items[end] {
            DisplayItem::PushOpacity { .. } => {
                depth += 1;
                nested = true;
            }
            DisplayItem::PopOpacity => {
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
            DisplayItem::PushClip { .. }
            | DisplayItem::PushRoundedClip { .. }
            | DisplayItem::PopClip => {}
            _ => {
                if depth == 0 {
                    paint_items += 1;
                    single_item = if paint_items == 1 { Some(end) } else { None };
                }
            }
        }
        end += 1;
    }
    (end, paint_items, nested, single_item)
}

/// Scan from `idx` (a `PushBlend`) to its matching `PopBlend`,
/// honoring nesting of both group kinds. Returns the index of the
/// matching `PopBlend`, or `items.len()` when the list is unbalanced.
///
/// Shared with the wgpu backend, which lowers the same groups to
/// offscreen textures.
pub(crate) fn scan_blend_group(items: &[DisplayItem], idx: usize) -> usize {
    let mut depth = 0usize;
    let mut end = idx + 1;
    while end < items.len() {
        match &items[end] {
            DisplayItem::PushOpacity { .. } | DisplayItem::PushBlend { .. } => {
                depth += 1;
            }
            DisplayItem::PopOpacity | DisplayItem::PopBlend => {
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
            _ => {}
        }
        end += 1;
    }
    end
}

/// Scan from `idx` (a `PushFilter` or `PushBackdropFilter`) to its
/// matching pop, honoring nesting of all group kinds. Returns the index
/// of the matching pop, or `items.len()` when the list is unbalanced.
///
/// Shared with the wgpu backend, which lowers the same groups to
/// offscreen textures (filter) or a backdrop capture (backdrop-filter).
pub(crate) fn scan_filter_group(items: &[DisplayItem], idx: usize) -> usize {
    let mut depth = 0usize;
    let mut end = idx + 1;
    while end < items.len() {
        match &items[end] {
            DisplayItem::PushOpacity { .. }
            | DisplayItem::PushBlend { .. }
            | DisplayItem::PushTransform { .. }
            | DisplayItem::PushFilter { .. }
            | DisplayItem::PushBackdropFilter { .. } => {
                depth += 1;
            }
            DisplayItem::PopOpacity
            | DisplayItem::PopBlend
            | DisplayItem::PopTransform
            | DisplayItem::PopFilter
            | DisplayItem::PopBackdropFilter => {
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
            _ => {}
        }
        end += 1;
    }
    end
}

/// Match every `PushOpacity` with its `PopOpacity` in a single O(n)
/// pass, returning the [`scan_opacity_group`] tuple per `PushOpacity`
/// index. Backends call this once per frame instead of scanning at each
/// group: nested groups would otherwise re-walk the same items, for
/// O(items x nesting depth) total, and a pathological list could spend
/// unbounded time scanning before the group cap ever applies. Results
/// are identical to calling [`scan_opacity_group`] at each index:
/// `end` is the matching `PopOpacity`, or `items.len()` when
/// unbalanced; unmatched `PopOpacity`s are ignored.
pub(crate) fn precompute_group_scans(
    items: &[DisplayItem],
) -> HashMap<usize, (usize, usize, bool, Option<usize>)> {
    struct Frame {
        push_idx: usize,
        paint_items: usize,
        nested: bool,
        single_item: Option<usize>,
    }
    let mut out = HashMap::new();
    let mut stack: Vec<Frame> = Vec::new();
    // Insert a finished frame's scan tuple.
    let mut record = |frame: Frame, end: usize| {
        out.insert(
            frame.push_idx,
            (end, frame.paint_items, frame.nested, frame.single_item),
        );
    };
    for (j, item) in items.iter().enumerate() {
        match item {
            DisplayItem::PushOpacity { .. } => stack.push(Frame {
                push_idx: j,
                paint_items: 0,
                nested: false,
                single_item: None,
            }),
            DisplayItem::PopOpacity => {
                if let Some(frame) = stack.pop() {
                    // The popped frame was a `PushOpacity` inside its
                    // parent, so the parent is nested.
                    if let Some(parent) = stack.last_mut() {
                        parent.nested = true;
                    }
                    record(frame, j);
                }
                // Unmatched `PopOpacity`: ignored, like the scan (which
                // only ever runs forward from a `PushOpacity`).
            }
            DisplayItem::PushClip { .. }
            | DisplayItem::PushRoundedClip { .. }
            | DisplayItem::PopClip => {}
            _ => {
                // Top-level paint item of the innermost open group only;
                // items inside nested groups count toward those groups.
                if let Some(frame) = stack.last_mut() {
                    frame.paint_items += 1;
                    frame.single_item = if frame.paint_items == 1 {
                        Some(j)
                    } else {
                        None
                    };
                }
            }
        }
    }
    // Unclosed groups run to the end, like the scan. Propagate nesting
    // outward first: each still-open frame sits inside its parent.
    for i in (1..stack.len()).rev() {
        stack[i - 1].nested = true;
    }
    let end = items.len();
    for frame in stack {
        record(frame, end);
    }
    out
}

/// Scan from `idx` (a `PushTransform`) to its matching `PopTransform`,
/// honoring nesting of all group kinds. Returns the index of the
/// matching `PopTransform`, or `items.len()` when the list is
/// unbalanced.
///
/// Shared with the wgpu backend, which lowers the same groups to
/// offscreen textures.
pub(crate) fn scan_transform_group(items: &[DisplayItem], idx: usize) -> usize {
    let mut depth = 0usize;
    let mut end = idx + 1;
    while end < items.len() {
        match &items[end] {
            DisplayItem::PushOpacity { .. }
            | DisplayItem::PushBlend { .. }
            | DisplayItem::PushTransform { .. } => {
                depth += 1;
            }
            DisplayItem::PopOpacity | DisplayItem::PopBlend | DisplayItem::PopTransform => {
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
            _ => {}
        }
        end += 1;
    }
    end
}

/// Match every `PushTransform` with its `PopTransform` in a single O(n)
/// pass, returning the matching `PopTransform` index per `PushTransform`
/// index (`items.len()` when unbalanced; unmatched pops ignored).
/// Backends call this once per frame instead of scanning at each group:
/// nested groups would otherwise re-walk the same items. Results are
/// identical to [`scan_transform_group`] on well-formed lists (pops
/// always close the innermost open group of any kind there, so plain
/// LIFO matching agrees with the depth-counting scan).
pub(crate) fn precompute_transform_ends(items: &[DisplayItem]) -> HashMap<usize, usize> {
    let mut out = HashMap::new();
    let mut stack: Vec<usize> = Vec::new();
    for (j, item) in items.iter().enumerate() {
        match item {
            DisplayItem::PushTransform { .. } => stack.push(j),
            DisplayItem::PopTransform => {
                if let Some(push_idx) = stack.pop() {
                    out.insert(push_idx, j);
                }
                // Unmatched `PopTransform`: ignored, like the scan (which
                // only ever runs forward from a `PushTransform`).
            }
            _ => {}
        }
    }
    // Unclosed groups run to the end, like the scan.
    let end = items.len();
    for push_idx in stack {
        out.insert(push_idx, end);
    }
    out
}

/// Multiply two 2D matrices in `(a, b, c, d, e, f)` component order
/// (see [`DisplayItem::PushTransform`]): the result applies `rhs`
/// first, then `lhs` — the same order as
/// `bowser_css::TransformMatrix::mul`, so composing a transform stack
/// left-to-right applies the innermost group first.
fn mul6(lhs: [f32; 6], rhs: [f32; 6]) -> [f32; 6] {
    [
        lhs[0] * rhs[0] + lhs[2] * rhs[1],
        lhs[1] * rhs[0] + lhs[3] * rhs[1],
        lhs[0] * rhs[2] + lhs[2] * rhs[3],
        lhs[1] * rhs[2] + lhs[3] * rhs[3],
        lhs[0] * rhs[4] + lhs[2] * rhs[5] + lhs[4],
        lhs[1] * rhs[4] + lhs[3] * rhs[5] + lhs[5],
    ]
}

/// Conservative page-px bounding box of `rect` under the 2D matrix
/// `(a, b, c, d, e, f)`: the bbox of the four transformed corners.
/// Used to size outer isolated groups around transformed content;
/// shared with the wgpu backend (see `lowered_bounds`).
pub(crate) fn transform_rect_bounds(m: [f32; 6], rect: Rect) -> Rect {
    let [a, b, c, d, e, f] = m;
    let p = |x: f32, y: f32| (a * x + c * y + e, b * x + d * y + f);
    let (x0, y0) = p(rect.x, rect.y);
    let (x1, y1) = p(rect.x + rect.w, rect.y);
    let (x2, y2) = p(rect.x, rect.y + rect.h);
    let (x3, y3) = p(rect.x + rect.w, rect.y + rect.h);
    let lx = x0.min(x1).min(x2).min(x3);
    let ly = y0.min(y1).min(y2).min(y3);
    Rect {
        x: lx,
        y: ly,
        w: x0.max(x1).max(x2).max(x3) - lx,
        h: y0.max(y1).max(y2).max(y3) - ly,
    }
}

/// Whether the `(a, b, c, d, e, f)` matrix is finite and invertible:
/// what [`Painter::paint_transform_group`] needs for the warp. The
/// determinant test is exact-zero: tiny-but-valid transforms (e.g.
/// `scale(1e-7)`, determinant 1e-14) are mathematically invertible and
/// must not be swallowed, while a genuinely singular matrix (zero
/// determinant) renders nothing.
fn invertible_matrix(m: [f32; 6]) -> bool {
    m.iter().all(|v| v.is_finite()) && m[0] * m[3] - m[1] * m[2] != 0.0
}

/// Bilinear sample of premultiplied RGBA bytes at `(u, v)`, edge
/// clamped, returned as f32 0..=255. Texel `(i, j)` is hit exactly at
/// integer coordinates, so integer translations resample exactly.
fn bilinear_sample(data: &[u8], w: i32, h: i32, u: f32, v: f32) -> [f32; 4] {
    let x0 = u.floor() as i32;
    let y0 = v.floor() as i32;
    let fx = (u - x0 as f32).clamp(0.0, 1.0);
    let fy = (v - y0 as f32).clamp(0.0, 1.0);
    let x0c = x0.clamp(0, w - 1);
    let y0c = y0.clamp(0, h - 1);
    let x1c = (x0 + 1).clamp(0, w - 1);
    let y1c = (y0 + 1).clamp(0, h - 1);
    let texel = |x: i32, y: i32| {
        let o = ((y * w + x) * 4) as usize;
        [
            data[o] as f32,
            data[o + 1] as f32,
            data[o + 2] as f32,
            data[o + 3] as f32,
        ]
    };
    let a = texel(x0c, y0c);
    let b = texel(x1c, y0c);
    let c = texel(x0c, y1c);
    let d = texel(x1c, y1c);
    let ix = 1.0 - fx;
    let iy = 1.0 - fy;
    [
        (a[0] * ix + b[0] * fx) * iy + (c[0] * ix + d[0] * fx) * fy,
        (a[1] * ix + b[1] * fx) * iy + (c[1] * ix + d[1] * fx) * fy,
        (a[2] * ix + b[2] * fx) * iy + (c[2] * ix + d[2] * fx) * fy,
        (a[3] * ix + b[3] * fx) * iy + (c[3] * ix + d[3] * fx) * fy,
    ]
}

/// Point-in-rounded-rect test for one subpixel sample: `(x0, y0, x1, y1)`
/// is the bounding box (device px), `radii` the circular corner radii
/// `[TL, TR, BR, BL]`. A local copy of the sampling test in
/// `crate::clip` (`clip.rs` is owned by a sibling worker); the shadow
/// rasterizer below needs it directly.
fn shadow_point_in_rounded_rect(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    radii: [f32; 4],
    sx: f32,
    sy: f32,
) -> bool {
    if sx < x0 || sx >= x1 || sy < y0 || sy >= y1 {
        return false;
    }
    let [rtl, rtr, rbr, rbl] = radii;
    // Each corner box is tested only when its radius is positive; a
    // zero/negative radius is a square corner (the sample is inside by
    // the rect test above). Circle tests compare squared distances —
    // no sqrt, no division.
    if rtl > 0.0 && sx < x0 + rtl && sy < y0 + rtl {
        let dx = sx - (x0 + rtl);
        let dy = sy - (y0 + rtl);
        return dx * dx + dy * dy <= rtl * rtl;
    }
    if rtr > 0.0 && sx > x1 - rtr && sy < y0 + rtr {
        let dx = sx - (x1 - rtr);
        let dy = sy - (y0 + rtr);
        return dx * dx + dy * dy <= rtr * rtr;
    }
    if rbr > 0.0 && sx > x1 - rbr && sy > y1 - rbr {
        let dx = sx - (x1 - rbr);
        let dy = sy - (y1 - rbr);
        return dx * dx + dy * dy <= rbr * rbr;
    }
    if rbl > 0.0 && sx < x0 + rbl && sy > y1 - rbl {
        let dx = sx - (x0 + rbl);
        let dy = sy - (y1 - rbl);
        return dx * dx + dy * dy <= rbl * rbl;
    }
    true
}

/// Fill `buf` (`w`×`h` bytes, row-major) with the coverage of the filled
/// rounded rect `(x0, y0, x1, y1)` + `radii` (device px, in the buffer's
/// coordinate space), 4×4-supersampled and quantized to 0..=255. Used
/// for box-shadow shape rasterization.
#[allow(clippy::too_many_arguments)]
fn rasterize_rounded_rect(
    buf: &mut [u8],
    w: usize,
    h: usize,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    radii: [f32; 4],
) {
    debug_assert_eq!(buf.len(), w * h);
    for (i, px) in buf.iter_mut().enumerate() {
        let x = (i % w) as f32;
        let y = (i / w) as f32;
        let mut inside = 0u32;
        for sy in 0..4 {
            for sx in 0..4 {
                let sxx = x + (sx as f32 + 0.5) / 4.0;
                let syy = y + (sy as f32 + 0.5) / 4.0;
                if shadow_point_in_rounded_rect(x0, y0, x1, y1, radii, sxx, syy) {
                    inside += 1;
                }
            }
        }
        *px = (inside * 255 / 16) as u8;
    }
}

/// One horizontal box-blur pass of half-width `r` over a u8 coverage
/// buffer, clamped edges (`SpreadMode::Pad` convention).
fn box_blur_pass_h(src: &[u8], dst: &mut [u8], w: usize, h: usize, r: usize) {
    debug_assert_eq!(src.len(), w * h);
    debug_assert_eq!(dst.len(), w * h);
    let width = 2 * r + 1;
    for y in 0..h {
        let row = y * w;
        // Seed the window for x = 0: indices [-r, r] clamped to the
        // row — r copies of the left edge, src[0..=min(r, w-1)], and
        // the clamped right tail when the row is shorter than r.
        let mut sum = src[row] as u32 * r as u32;
        for x in 0..(r + 1).min(w) {
            sum += src[row + x] as u32;
        }
        if w <= r {
            sum += src[row + w - 1] as u32 * (r + 1 - w) as u32;
        }
        for x in 0..w {
            dst[row + x] = (sum / width as u32) as u8;
            let add = src[row + (x + r + 1).min(w - 1)] as u32;
            let sub = src[row + x.saturating_sub(r)] as u32;
            sum = sum + add - sub;
        }
    }
}

/// One vertical box-blur pass of half-width `r`, clamped edges.
fn box_blur_pass_v(src: &[u8], dst: &mut [u8], w: usize, h: usize, r: usize) {
    debug_assert_eq!(src.len(), w * h);
    debug_assert_eq!(dst.len(), w * h);
    let width = 2 * r + 1;
    for x in 0..w {
        // Seed the window for y = 0: r copies of the top edge,
        // src[0..=min(r, h-1)] down the column, and the clamped bottom
        // tail when the column is shorter than r.
        let mut sum = src[x] as u32 * r as u32;
        for y in 0..(r + 1).min(h) {
            sum += src[y * w + x] as u32;
        }
        if h <= r {
            sum += src[(h - 1) * w + x] as u32 * (r + 1 - h) as u32;
        }
        for y in 0..h {
            dst[y * w + x] = (sum / width as u32) as u8;
            let add = src[(y + r + 1).min(h - 1) * w + x] as u32;
            let sub = src[y.saturating_sub(r) * w + x] as u32;
            sum = sum + add - sub;
        }
    }
}

/// Integer box-blur half-width whose 3 (H, V) pass pairs approximate
/// gaussian `sigma` (in px): three box passes of width `2r+1` have
/// total variance `3·r·(r+1)`, so `r² + r − σ² = 0`. Delegates to the
/// shared [`crate::filter::blur_box_radius`] so both backends blur the
/// same radius.
fn box_blur_radius(sigma: f32) -> usize {
    crate::filter::blur_box_radius(sigma) as usize
}

/// Absolute cap on a box-blur half-width: the sliding-window sum holds
/// exactly `2r+1` taps of at most 255 in a u32, so `255·(2r+1)` must
/// stay within `u32::MAX`. `255·(2·8421504+1) == u32::MAX` exactly.
/// Radii past this are visually indistinguishable from it anyway.
const MAX_BLUR_HALF_WIDTH: usize = 8_421_504;

/// Clamp a box-blur half-width to what the image can actually resolve:
/// a radius beyond `max(w, h)` is visually identical to `max(w, h)`
/// (every window already covers the whole image, edges clamped), and
/// bounding `r` also bounds the O(r) seeding work. Together with
/// [`MAX_BLUR_HALF_WIDTH`] this keeps every `255·r` product and window
/// sum inside u32, so `filter: blur(1e8px)` can neither panic (debug)
/// nor wrap (release). (F8)
fn clamp_blur_radius(r: usize, w: usize, h: usize) -> usize {
    r.min(w.max(h)).min(MAX_BLUR_HALF_WIDTH)
}

/// Multi-pass box blur approximating gaussian `sigma` (in px) on a u8
/// coverage buffer, in place through `tmp` (same size as `buf`).
/// `sigma <= 0` is a no-op.
fn box_blur_sigma(buf: &mut [u8], tmp: &mut [u8], w: usize, h: usize, sigma: f32) {
    let r = clamp_blur_radius(box_blur_radius(sigma), w, h);
    if r == 0 {
        return;
    }
    for _ in 0..3 {
        box_blur_pass_h(buf, tmp, w, h, r);
        box_blur_pass_v(tmp, buf, w, h, r);
    }
}

/// Multi-pass box blur approximating gaussian `sigma` (in px) on
/// premultiplied RGBA bytes, in place through `tmp`. Each channel
/// blurs independently (standard for premultiplied data).
pub(crate) fn box_blur_sigma_rgba(buf: &mut [u8], tmp: &mut [u8], w: usize, h: usize, sigma: f32) {
    let r = clamp_blur_radius(box_blur_radius(sigma), w, h);
    if r == 0 {
        return;
    }
    debug_assert_eq!(buf.len(), w * h * 4);
    debug_assert_eq!(tmp.len(), w * h * 4);
    let width = 2 * r + 1;
    for _ in 0..3 {
        // Horizontal.
        for y in 0..h {
            let row = y * w * 4;
            let mut sum = [0u32; 4];
            for c in 0..4 {
                sum[c] = buf[row + c] as u32 * r as u32;
            }
            for x in 0..(r + 1).min(w) {
                let o = row + x * 4;
                for c in 0..4 {
                    sum[c] += buf[o + c] as u32;
                }
            }
            if w <= r {
                let o = row + (w - 1) * 4;
                for c in 0..4 {
                    sum[c] += buf[o + c] as u32 * (r + 1 - w) as u32;
                }
            }
            for x in 0..w {
                let o = row + x * 4;
                for c in 0..4 {
                    tmp[o + c] = (sum[c] / width as u32) as u8;
                    let add = buf[row + (x + r + 1).min(w - 1) * 4 + c] as u32;
                    let sub = buf[row + x.saturating_sub(r) * 4 + c] as u32;
                    sum[c] = sum[c] + add - sub;
                }
            }
        }
        // Vertical.
        for x in 0..w {
            let mut sum = [0u32; 4];
            for c in 0..4 {
                sum[c] = tmp[x * 4 + c] as u32 * r as u32;
            }
            for y in 0..(r + 1).min(h) {
                let o = y * w * 4 + x * 4;
                for c in 0..4 {
                    sum[c] += tmp[o + c] as u32;
                }
            }
            if h <= r {
                let o = (h - 1) * w * 4 + x * 4;
                for c in 0..4 {
                    sum[c] += tmp[o + c] as u32 * (r + 1 - h) as u32;
                }
            }
            for y in 0..h {
                let o = y * w * 4 + x * 4;
                for c in 0..4 {
                    buf[o + c] = (sum[c] / width as u32) as u8;
                    let add = tmp[(y + r + 1).min(h - 1) * w * 4 + x * 4 + c] as u32;
                    let sub = tmp[y.saturating_sub(r) * w * 4 + x * 4 + c] as u32;
                    sum[c] = sum[c] + add - sub;
                }
            }
        }
    }
}

/// Cap on one box-shadow temp buffer's dimension (px): past it the
/// blur radius is clamped to fit (see `paint_box_shadow`). Bounds the
/// transient allocation for absurd blurs; a 4096px blur is already far
/// past any visible distinction.
const MAX_SHADOW_TEMP_DIM: usize = 4096;

/// Union of two page-px rects.
fn union_rect(a: Rect, b: Rect) -> Rect {
    let x0 = a.x.min(b.x);
    let y0 = a.y.min(b.y);
    Rect {
        x: x0,
        y: y0,
        w: (a.x + a.w).max(b.x + b.w) - x0,
        h: (a.y + a.h).max(b.y + b.h) - y0,
    }
}

/// Inflate a page-px rect by `d` on every side.
fn inflate_rect(r: Rect, d: f32) -> Rect {
    Rect {
        x: r.x - d,
        y: r.y - d,
        w: r.w + 2.0 * d,
        h: r.h + 2.0 * d,
    }
}

/// Intersect two page-px rects; the result may be empty (non-positive
/// size) when they do not overlap.
fn intersect_rect(a: Rect, b: Rect) -> Rect {
    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    Rect {
        x,
        y,
        w: (a.x + a.w).min(b.x + b.w) - x,
        h: (a.y + a.h).min(b.y + b.h) - y,
    }
}

/// Conservative paint bounds of one display item in page px: every pixel
/// the item can paint lies inside the returned rect. `None` for
/// non-paint items (clips, opacity markers). Used to size isolated
/// opacity groups' offscreen pixmaps (see
/// [`Painter::paint_opacity_group`]).
///
/// Conservatism notes: text is inflated by its full size on every side —
/// glyph ink can exceed the fragment rect through negative bearings and
/// italic overhang, and bearings are fractions of an em, so one em
/// covers them with margin. Decorations inflate by their thickness; the
/// line sits on the ink box edge. Decimal list markers render as text
/// (same inflation as a text run); geometric markers stay inside their
/// rect.
fn item_paint_bounds(item: &DisplayItem) -> Option<Rect> {
    match item {
        DisplayItem::FillRect { rect, .. } => Some(*rect),
        DisplayItem::Border { rect, .. } => Some(*rect),
        DisplayItem::TextRun { rect, size_px, .. } => Some(inflate_rect(*rect, size_px.max(0.0))),
        DisplayItem::Image { rect, .. } | DisplayItem::BackgroundImage { rect, .. } => Some(*rect),
        DisplayItem::Gradient { rect, .. } => Some(*rect),
        // The repeat area is backend-clamped before expansion, so bound
        // group offscreen textures by the clamped area, not the raw one.
        DisplayItem::RepeatImage { area, .. } => Some(clamp_repeat_area(*area)),
        DisplayItem::Decoration {
            rect, thickness, ..
        } => Some(inflate_rect(*rect, thickness.max(0.0))),
        DisplayItem::ListMarker {
            rect, kind, index, ..
        } => {
            if Painter::marker_numeral(*kind, *index).is_some() {
                Some(inflate_rect(*rect, rect.w.min(rect.h).max(1.0)))
            } else {
                Some(*rect)
            }
        }
        DisplayItem::BoxShadow {
            rect,
            offset,
            blur,
            spread,
            inset,
            ..
        } => {
            if *inset {
                // Inset shadows never leave the border box (they are
                // clipped to its interior by construction).
                Some(*rect)
            } else {
                // Outset: the spread-inflated shape, blurred and
                // offset. Negative spreads only shrink the shape, so
                // clamping at zero stays conservative.
                let m = spread.max(0.0) + blur.max(0.0);
                Some(Rect {
                    x: rect.x - m + offset.0,
                    y: rect.y - m + offset.1,
                    w: rect.w + 2.0 * m,
                    h: rect.h + 2.0 * m,
                })
            }
        }
        DisplayItem::EdgeGlow { rect, .. } => Some(*rect),
        DisplayItem::TextShadow {
            rect,
            size_px,
            offset,
            blur,
            ..
        } => {
            let r = inflate_rect(*rect, size_px.max(0.0));
            let m = blur.max(0.0);
            Some(Rect {
                x: r.x - m + offset.0,
                y: r.y - m + offset.1,
                w: r.w + 2.0 * m,
                h: r.h + 2.0 * m,
            })
        }
        _ => None,
    }
}

/// Conservative paint bounds of an opacity/blend/transform group's item
/// range (`items[start..end]`, `end` exclusive) in page px: the union of
/// every paint item's [`item_paint_bounds`], nested groups included
/// (their items are in the same range). `None` when nothing in the
/// range paints. Clips are deliberately ignored — the union can only
/// over-cover — and the caller intersects the result with the live clip
/// and the target surface.
///
/// Transform-aware: a [`DisplayItem::PushTransform`] pushes its matrix
/// onto a stack (composed with the enclosing transforms, innermost
/// last), and every paint item's bounds are transformed by the
/// composed matrix. A transform group's own range starts *after* its
/// push, so its pixmap sizes to the untransformed content; an outer
/// group's range includes the push, so it sizes to the transformed
/// extents the inner composite will cover. Non-finite matrices fall
/// back to untransformed bounds (the group paints nothing anyway).
fn group_paint_bounds(items: &[DisplayItem], start: usize, end: usize) -> Option<Rect> {
    let mut acc: Option<Rect> = None;
    let mut stack: Vec<[f32; 6]> = Vec::new();
    for item in items.iter().take(end.min(items.len())).skip(start) {
        match item {
            DisplayItem::PushTransform { matrix } => {
                let composed = match stack.last() {
                    Some(top) => mul6(*top, *matrix),
                    None => *matrix,
                };
                stack.push(composed);
            }
            DisplayItem::PopTransform => {
                stack.pop();
            }
            _ => {
                if let Some(r) = item_paint_bounds(item) {
                    let r = match stack.last() {
                        Some(m) if m.iter().all(|v| v.is_finite()) => transform_rect_bounds(*m, r),
                        _ => r,
                    };
                    acc = Some(match acc {
                        Some(a) => union_rect(a, r),
                        None => r,
                    });
                }
            }
        }
    }
    acc
}

/// Whether folding a group opacity into this item's alpha equals
/// isolated-group compositing.
///
/// Folding is only valid when the item lowers to non-overlapping
/// primitives. A [`DisplayItem::TextRun`] lowers to one quad per glyph
/// and kerned/italic glyphs can overlap, so folding would double-apply
/// the opacity where they intersect (source-over is not linear in the
/// folded alpha); decimal list markers render as text runs and have the
/// same problem. Every other item kind lowers to a single primitive or
/// a provably disjoint tessellation (border sides/corner bands,
/// decoration bars, geometric marker discs).
pub(crate) fn opacity_foldable(item: &DisplayItem) -> bool {
    match item {
        DisplayItem::TextRun { .. } => false,
        // A blend, transform, or filter group is never foldable: folding
        // the outer opacity into its items' alphas would change the
        // blend math (the mode blends the *isolated* group against the
        // backdrop; item alphas feed the blend formula, so the opacity
        // must apply after the blend, not before), a transform group
        // resamples its content — folding would fade the source before
        // resampling, which is not isolated compositing — and a filter
        // group post-processes the isolated rendering, which folding
        // cannot reproduce.
        DisplayItem::PushBlend { .. }
        | DisplayItem::PopBlend
        | DisplayItem::PushTransform { .. }
        | DisplayItem::PopTransform
        | DisplayItem::PushFilter { .. }
        | DisplayItem::PopFilter
        | DisplayItem::PushBackdropFilter { .. }
        | DisplayItem::PopBackdropFilter => false,
        // A text shadow paints the same overlapping glyphs as a text
        // run (blurred or not), so it shares the text run's
        // non-foldability.
        DisplayItem::TextShadow { .. } => false,
        DisplayItem::ListMarker { kind, .. } => !matches!(
            kind,
            MarkerKind::Decimal(_) | MarkerKind::DecimalLeadingZero(_)
        ),
        _ => true,
    }
}

/// Effective clip state derived from the clip stack: the bounding
/// intersection of the stack's rects. Rounded entries contribute only
/// their bounding rect here (a superset of the true clip, so bound
/// culling stays sound); per-pixel coverage evaluates the full stack
/// via [`clip_stack_coverage_byte`](crate::clip::clip_stack_coverage_byte).
#[derive(Clone, Copy)]
enum EffectiveClip {
    /// No clip items on the stack.
    None,
    /// Clipped to an empty region: draw nothing.
    Empty,
    /// Intersected clip bound, device px.
    Rect(TsRect),
}

/// One entry of the painter's clip stack: a device-px bounding rect
/// plus circular corner radii `[TL, TR, BR, BL]` in device px.
/// All-zero radii are a plain rect clip
/// ([`DisplayItem::PushClip`]); `None` rects clip everything out.
///
/// Nested clips compose by geometric intersection (see `crate::clip`):
/// [`clip_stack_coverage_byte`](crate::clip::clip_stack_coverage_byte)
/// evaluates the whole stack per pixel.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct ClipEntry {
    /// Device-px bounding rect; `None` = degenerate/empty clip.
    pub(crate) rect: Option<TsRect>,
    /// Device-px corner radii; all-zero = plain rect.
    pub(crate) radii: [f32; CORNERS],
    /// `clip-path` polygon vertices in device px (relative to the
    /// current paint target's origin, like `rect`); `None` = not a
    /// polygon entry. `rect` still carries the polygon's bounding box
    /// so bound-culling keeps working. Fixed-size (48 verts) so the
    /// entry stays `Copy`; see [`ClipEntry::poly_points`].
    pub(crate) poly: Option<([[f32; 2]; MAX_POLY_VERTS], u8)>,
}

impl ClipEntry {
    /// Plain rect entry.
    fn rect(rect: Option<TsRect>) -> Self {
        Self {
            rect,
            radii: [0.0; CORNERS],
            poly: None,
        }
    }

    /// Whether this entry is an exact rect (no rounded corners, no
    /// polygon).
    fn is_rect(&self) -> bool {
        self.radii == [0.0; CORNERS] && self.poly.is_none()
    }

    /// The polygon vertices as a slice (`None` = not a polygon entry).
    pub(crate) fn poly_points(&self) -> Option<&[[f32; 2]]> {
        self.poly
            .as_ref()
            .map(|(pts, n)| &pts[..(*n as usize).min(MAX_POLY_VERTS)])
    }
}

/// Intersected bounding rect of the stack's entries (`None` when the
/// stack is empty, any entry is `None`, or the intersection is empty).
/// Radii are ignored: the bound is a superset of the true (possibly
/// rounded) clip.
fn stack_bound(stack: &[ClipEntry]) -> Option<TsRect> {
    let mut acc: Option<TsRect> = None;
    for e in stack {
        let r = e.rect?;
        acc = Some(match acc {
            Some(a) => a.intersect(&r)?,
            None => r,
        });
    }
    acc
}

/// Clamped device-px bounds of the clip mask's opaque (255) region:
/// pixels with `x0 <= x < x1` and `y0 <= y < y1` read 255, everything
/// else reads 0. Invariant: `x0 <= x1`, `y0 <= y1`; an empty rect is
/// never constructed — it normalizes to `None`, the all-zero mask.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct MaskRect {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

/// Intersection of two mask regions (`None` when empty).
fn intersect_mask_rect(a: MaskRect, b: MaskRect) -> Option<MaskRect> {
    let r = MaskRect {
        x0: a.x0.max(b.x0),
        y0: a.y0.max(b.y0),
        x1: a.x1.min(b.x1),
        y1: a.y1.min(b.y1),
    };
    (r.x0 < r.x1 && r.y0 < r.y1).then_some(r)
}

/// The opaque region of the clip bound on a `w` x `h` surface, using
/// the same floor/ceil/clamp convention the old full rebuild used, so
/// the incremental buffer is byte-identical to it. `None` is the
/// all-zero mask (empty clip, or a bound clipped fully off-surface).
fn clip_mask_region(bound: Option<TsRect>, w: usize, h: usize) -> Option<MaskRect> {
    let r = bound?;
    let x0 = (r.left().floor().max(0.0) as usize).min(w);
    let x1 = (r.right().ceil().max(0.0) as usize).min(w);
    let y0 = (r.top().floor().max(0.0) as usize).min(h);
    let y1 = (r.bottom().ceil().max(0.0) as usize).min(h);
    if x0 >= x1 || y0 >= y1 {
        return None;
    }
    Some(MaskRect { x0, y0, x1, y1 })
}

/// Paint the set difference `paint − keep` with `value`: each row of
/// `paint` either fills fully (outside `keep`'s y-range) or fills just
/// the side columns outside `keep`'s x-range. `None` is the empty set.
/// Runs in O(changed pixels), never O(surface).
fn paint_region_diff(
    data: &mut [u8],
    w: usize,
    paint: Option<MaskRect>,
    keep: Option<MaskRect>,
    value: u8,
) {
    let Some(p) = paint else { return };
    let (kx0, ky0, kx1, ky1) = match keep {
        Some(k) => (k.x0, k.y0, k.x1, k.y1),
        None => (0, 0, 0, 0),
    };
    for y in p.y0..p.y1 {
        let row = y * w;
        if y < ky0 || y >= ky1 {
            data[row + p.x0..row + p.x1].fill(value);
        } else {
            let lx = p.x1.min(kx0);
            if lx > p.x0 {
                data[row + p.x0..row + lx].fill(value);
            }
            let rx = p.x0.max(kx1);
            if rx < p.x1 {
                data[row + rx..row + p.x1].fill(value);
            }
        }
    }
}

/// Fractional coverage of device pixel `(x, y)` — the unit square
/// `[x, x+1) x [y, y+1)` — by the clip rect `(l, t, r, b)` in device px,
/// quantized to a mask byte.
///
/// This is the single definition of fractional clip semantics, shared
/// by the clip-mask fill ([`paint_coverage_diff`]), glyph stamping
/// ([`Painter::stamp_glyph`]), and the GPU backend's clip shader: all
/// three compute the identical value bit-for-bit (same IEEE-754 ops in
/// the same order), so clip boundaries agree pixel-for-pixel across
/// backends. Quantization is round-half-away-from-zero via
/// `floor(v + 0.5)`; the WGSL shader spells the same expression.
///
/// [`clip_stack_coverage_byte`](crate::clip::clip_stack_coverage_byte)
/// extends this to rounded corners and nested stacks; with all-zero
/// radii it reproduces this function bit-for-bit on the entries'
/// intersection (tested in `crate::clip`).
pub(crate) fn clip_coverage_byte(l: f32, t: f32, r: f32, b: f32, x: usize, y: usize) -> u8 {
    let px = x as f32;
    let py = y as f32;
    let cx = (r.min(px + 1.0) - l.max(px)).clamp(0.0, 1.0);
    let cy = (b.min(py + 1.0) - t.max(py)).clamp(0.0, 1.0);
    (cx * cy * 255.0 + 0.5).floor() as u8
}

/// Pixels of `r` the clip covers fully, hence bytes that read 255 under
/// both the old and the new clip: the incremental mask update
/// ([`Painter::update_clip_mask`]) keeps these instead of repainting
/// them. Conservative (sufficient, not necessary): a pixel is kept only
/// when it sits at least one pixel inside every edge of the bounding
/// region, which implies full coverage — see the argument in
/// [`Painter::update_clip_mask`].
fn mask_interior(r: MaskRect) -> Option<MaskRect> {
    let inner = MaskRect {
        x0: r.x0 + 1,
        y0: r.y0 + 1,
        // `x1`/`y1` are exclusive: pixels `p <= x1 - 2` are `p < x1 - 1`.
        x1: r.x1.saturating_sub(1),
        y1: r.y1.saturating_sub(1),
    };
    (inner.x0 < inner.x1 && inner.y0 < inner.y1).then_some(inner)
}

/// Paint `paint − keep` with the fractional coverage bytes of `clip`:
/// the coverage counterpart to [`paint_region_diff`]. `None` paints
/// nothing. Runs in O(painted pixels); pixels in the clip's
/// guaranteed-fully-covered interior (see [`mask_interior`]) fill 255
/// without per-pixel math.
fn paint_coverage_diff(
    data: &mut [u8],
    w: usize,
    paint: Option<MaskRect>,
    keep: Option<MaskRect>,
    clip: &TsRect,
) {
    let Some(p) = paint else { return };
    let (l, t, r, b) = (clip.x(), clip.y(), clip.right(), clip.bottom());
    let (kx0, ky0, kx1, ky1) = match keep {
        Some(k) => (k.x0, k.y0, k.x1, k.y1),
        None => (0, 0, 0, 0),
    };
    let inner = mask_interior(p);
    for y in p.y0..p.y1 {
        let row = y * w;
        let in_inner_y = inner.is_some_and(|n| y >= n.y0 && y < n.y1);
        // The row's painted x-ranges: the whole row, minus `keep`'s
        // x-range on rows `keep` covers (mirrors `paint_region_diff`).
        let mut segs = [(p.x0, p.x1), (0, 0)];
        let mut nsegs = 1;
        if y >= ky0 && y < ky1 {
            nsegs = 0;
            let (a0, a1) = (p.x0, p.x1.min(kx0));
            let (b0, b1) = (p.x0.max(kx1), p.x1);
            if a0 < a1 {
                segs[nsegs] = (a0, a1);
                nsegs += 1;
            }
            if b0 < b1 {
                segs[nsegs] = (b0, b1);
                nsegs += 1;
            }
        }
        for &(xa, xb) in &segs[..nsegs] {
            if in_inner_y && let Some(n) = inner {
                let (fx0, fx1) = (xa.max(n.x0), xb.min(n.x1));
                if fx0 < fx1 {
                    data[row + fx0..row + fx1].fill(255);
                }
                for x in xa..fx0 {
                    data[row + x] = clip_coverage_byte(l, t, r, b, x, y);
                }
                for x in fx1..xb {
                    data[row + x] = clip_coverage_byte(l, t, r, b, x, y);
                }
                continue;
            }
            for x in xa..xb {
                data[row + x] = clip_coverage_byte(l, t, r, b, x, y);
            }
        }
    }
}

/// Paint `paint` with the fractional coverage bytes of a rounded clip
/// stack: each pixel gets the coverage of the stack's geometric
/// intersection (see `crate::clip`).
///
/// Unlike [`paint_coverage_diff`] there is no `keep`/interior fast
/// path: a pixel a full px inside the bounding region can still be
/// partial under a rounded corner, so every painted pixel is
/// evaluated. `paint` is `None` (empty clip) paints nothing.
fn paint_rounded_coverage_diff(
    data: &mut [u8],
    w: usize,
    paint: Option<MaskRect>,
    stack: &[ClipEntry],
) {
    let Some(p) = paint else { return };
    for y in p.y0..p.y1 {
        let row = y * w;
        for x in p.x0..p.x1 {
            // The stack's geometric intersection, evaluated per pixel
            // (see `crate::clip`).
            data[row + x] = clip_stack_coverage_byte(stack, x, y);
        }
    }
}

/// Per-render painter state.
///
/// The glyph atlas, clip-mask buffer, and image cache are borrowed from
/// the [`CpuBackend`] and persist across frames; the backend evicts the
/// atlas when misses pass the cap.
struct Painter<'a> {
    px: &'a mut Pixmap,
    get_font_data: &'a Rc<dyn Fn(FontKey) -> Option<FontData>>,
    scale: f32,
    scroll: (f32, f32),
    /// Device-px origin of the current paint target relative to the
    /// surface: `(0, 0)` on the main surface, the group's snapped bounds
    /// origin (accumulated through nesting) inside an isolated opacity
    /// group. [`dev_x`](Self::dev_x)/[`dev_y`](Self::dev_y) subtract it,
    /// so items keep painting in page px while the bounded group pixmap
    /// (see [`Painter::paint_opacity_group`]) only holds the group's
    /// region. Clip rects pushed while inside a group are therefore
    /// group-local, exactly like the pixels.
    dev_origin: (f32, f32),
    /// Device-px clip entries; `None` rects clip everything out.
    /// Rounded entries carry their corner radii — per-pixel coverage
    /// is the min over the stack (see `crate::clip`).
    clip_stack: Vec<ClipEntry>,
    /// Persistent full-surface clip-mask buffer (owned by the
    /// backend). `None` until the first clip op; afterwards reused in
    /// place across ops and frames — only reallocated when the surface
    /// size changes. Consulted only when the clip stack is non-empty
    /// (see [`Painter::active_clip_mask`]).
    clip_mask: &'a mut Option<Mask>,
    /// The opaque region `clip_mask` currently encodes, tracked exactly
    /// by [`Painter::update_clip_mask`] so clip ops rewrite only the
    /// bytes whose membership changed. Borrowed from the backend:
    /// it survives across frames like the buffer itself.
    mask_region: &'a mut Option<MaskRect>,
    /// The clip stack the mask's bytes encode (`[]` = all zeros); the
    /// update's early-return compares this, not the region (fractional
    /// edges change bytes without changing the region).
    mask_stack: &'a mut Vec<ClipEntry>,
    opacity_stack: Vec<f32>,
    /// Nesting depth of isolated opacity groups currently being
    /// painted; capped at [`MAX_OPACITY_GROUPS`] (see
    /// [`Painter::paint_opacity_group`]).
    group_depth: u32,
    /// `PushOpacity`-index -> [`scan_opacity_group`] tuple, built once per
    /// frame by [`precompute_group_scans`]. Nested groups would otherwise
    /// re-walk the same items (O(items x depth)); the single pass keeps
    /// group discovery O(items).
    group_scans: HashMap<usize, (usize, usize, bool, Option<usize>)>,
    /// `PushTransform`-index -> matching `PopTransform` index, built
    /// once per frame by [`precompute_transform_ends`] (same O(items)
    /// rationale as `group_scans`).
    transform_ends: HashMap<usize, usize>,
    /// Image decode caches, persistent across frames (owned by the
    /// backend; see [`ImageCache`]).
    image_cache: &'a mut ImageCache,
    /// Lifetime cache/pool counters (owned by the backend; see
    /// [`CacheStats`]).
    cache_stats: &'a mut CacheStats,
    /// Free-list of bounds-sized pixmaps for isolated opacity groups
    /// (owned by the backend, persists across frames); checked out and
    /// returned by [`Painter::paint_opacity_group`].
    group_pixmap_pool: &'a mut Vec<Pixmap>,
    /// Reusable byte scratch for filter passes (owned by the backend,
    /// persists across frames); grown to the largest group pixmap seen.
    filter_scratch: &'a mut Vec<u8>,
    /// Free-list of scratch byte buffers for shadow temps (owned by the
    /// backend, persists across frames); checked out and returned by
    /// the shadow painters (see [`CpuBackend::shadow_scratch_pool`]).
    shadow_scratch_pool: &'a mut Vec<Vec<u8>>,
    /// Reusable quad scratch for border tessellation (owned by the
    /// backend, persists across frames); cleared and filled per
    /// bordered box by [`tessellate_border_into`], so the border path
    /// allocates nothing after the first border's capacity is grown
    /// (see [`CpuBackend::border_scratch`]).
    border_scratch: &'a mut Vec<Quad>,
    /// CPU glyph coverage cache, keyed like the GPU [`crate::text::Atlas`].
    /// Borrowed from the backend: it survives across frames.
    glyphs: &'a mut GlyphAtlas,
    /// Backend's miss counter; incremented on atlas misses for eviction.
    glyph_misses: &'a mut usize,
    /// Baked gradient rasters (owned by the backend, persists across
    /// frames); stamped with `gradient_frame` on use, evicted at the
    /// frame boundary (see [`CpuGradientBake`]).
    gradient_bake_cache: &'a mut HashMap<u64, CpuGradientBake>,
    /// Current frame generation (owned by the backend).
    gradient_frame: u64,
}

/// Format a list-marker `index` as decimal into `buf`, returning the
/// written digits. `pad_two` left-pads a single digit with `'0'`
/// (`decimal-leading-zero`). `buf` needs 20 bytes (`u32::MAX` is 10
/// digits, plus the pad). Stack-allocated: the per-frame marker path
/// never touches the heap. Same formula as the GPU backend.
fn format_marker_numeral(buf: &mut [u8; 20], index: u32, pad_two: bool) -> &str {
    let mut i = buf.len();
    let mut n = index;
    loop {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    if pad_two && i == buf.len() - 1 {
        i -= 1;
        buf[i] = b'0';
    }
    // Only ASCII digits were written.
    str::from_utf8(&buf[i..]).expect("decimal digits are valid UTF-8")
}

impl<'a> Painter<'a> {
    // Constructor threads all of the painter's borrowed state; grouping
    // the borrows into a struct would just move the arity elsewhere.
    #[allow(clippy::too_many_arguments)]
    fn new(
        px: &'a mut Pixmap,
        get_font_data: &'a Rc<dyn Fn(FontKey) -> Option<FontData>>,
        glyphs: &'a mut GlyphAtlas,
        glyph_misses: &'a mut usize,
        clip_mask: &'a mut Option<Mask>,
        mask_region: &'a mut Option<MaskRect>,
        mask_stack: &'a mut Vec<ClipEntry>,
        image_cache: &'a mut ImageCache,
        cache_stats: &'a mut CacheStats,
        group_pixmap_pool: &'a mut Vec<Pixmap>,
        filter_scratch: &'a mut Vec<u8>,
        shadow_scratch_pool: &'a mut Vec<Vec<u8>>,
        border_scratch: &'a mut Vec<Quad>,
        gradient_bake_cache: &'a mut HashMap<u64, CpuGradientBake>,
        gradient_frame: u64,
        scale: f32,
        scroll: (f32, f32),
    ) -> Self {
        Self {
            px,
            get_font_data,
            scale,
            scroll,
            dev_origin: (0.0, 0.0),
            clip_stack: Vec::new(),
            clip_mask,
            mask_region,
            mask_stack,
            opacity_stack: Vec::new(),
            group_depth: 0,
            group_scans: HashMap::new(),
            transform_ends: HashMap::new(),
            image_cache,
            cache_stats,
            group_pixmap_pool,
            filter_scratch,
            shadow_scratch_pool,
            border_scratch,
            glyphs,
            glyph_misses,
            gradient_bake_cache,
            gradient_frame,
        }
    }

    /// Return the cached glyph, rasterizing and caching on a miss.
    /// Misses are counted for the backend's eviction policy. `None`
    /// when the glyph has no coverage or the font is unusable.
    fn cached_glyph(
        &mut self,
        font_key: FontKey,
        font_data: &FontData,
        glyph_id: u16,
        size_px: f32,
    ) -> Option<&RasterGlyph> {
        let already = self.glyphs.get(font_key, glyph_id, size_px).is_some();
        let glyph = self
            .glyphs
            .get_or_rasterize(font_key, font_data, glyph_id, size_px);
        if !already && glyph.is_some() {
            // A genuinely new rasterization: count it toward the eviction
            // budget. Failed rasterizations (spaces, broken glyphs) insert
            // nothing and must not churn the counter.
            *self.glyph_misses += 1;
        }
        glyph
    }

    /// Device-space X of a CSS-px coordinate, relative to the current
    /// paint target's origin (see `dev_origin`).
    fn dev_x(&self, x: f32) -> f32 {
        (x - self.scroll.0) * self.scale - self.dev_origin.0
    }

    /// Device-space Y of a CSS-px coordinate, relative to the current
    /// paint target's origin (see `dev_origin`).
    fn dev_y(&self, y: f32) -> f32 {
        (y - self.scroll.1) * self.scale - self.dev_origin.1
    }

    /// Map a CSS-px rect to a device-px tiny-skia rect.
    fn to_device_rect(&self, rect: Rect) -> Option<TsRect> {
        if rect.w <= 0.0 || rect.h <= 0.0 {
            return None;
        }
        TsRect::from_xywh(
            self.dev_x(rect.x),
            self.dev_y(rect.y),
            rect.w * self.scale,
            rect.h * self.scale,
        )
    }

    /// Map a CSS-px rounded rect to a device-px clip entry. Radii
    /// scale by the same factor as the rect's edges (uniform scale, so
    /// the builder's adjacent-sum clamp still holds); they are
    /// translation-invariant ([`shift_clips`] only moves rects).
    fn to_device_rounded_rect(&self, rect: Rect, radii: [f32; CORNERS]) -> ClipEntry {
        let mut entry = ClipEntry::rect(self.to_device_rect(rect));
        for (d, r) in entry.radii.iter_mut().zip(radii) {
            *d = (r * self.scale).max(0.0);
        }
        entry
    }

    /// Map page-px `clip-path` polygon points to a device-px clip entry.
    /// Non-finite points are dropped; fewer than three surviving
    /// vertices (or a degenerate bounding box) empty the clip (a
    /// zero-area polygon clips everything out, per spec). Vertices past
    /// [`MAX_POLY_VERTS`] are dropped: the display-list builder caps
    /// shapes there, so real input never hits this.
    fn to_device_polygon(&self, points: &[[f32; 2]]) -> ClipEntry {
        let mut verts = [[0.0f32; 2]; MAX_POLY_VERTS];
        let mut n = 0usize;
        for p in points.iter() {
            if n >= MAX_POLY_VERTS {
                break;
            }
            let (x, y) = (self.dev_x(p[0]), self.dev_y(p[1]));
            if x.is_finite() && y.is_finite() {
                verts[n] = [x, y];
                n += 1;
            }
        }
        if n < 3 {
            return ClipEntry {
                rect: None,
                radii: [0.0; CORNERS],
                poly: None,
            };
        }
        let mut x0 = f32::INFINITY;
        let mut y0 = f32::INFINITY;
        let mut x1 = f32::NEG_INFINITY;
        let mut y1 = f32::NEG_INFINITY;
        for p in verts[..n].iter() {
            x0 = x0.min(p[0]);
            y0 = y0.min(p[1]);
            x1 = x1.max(p[0]);
            y1 = y1.max(p[1]);
        }
        ClipEntry {
            // `from_xywh` is `None` for a zero-area bbox (all points
            // collinear): the empty clip, per spec.
            rect: TsRect::from_xywh(x0, y0, x1 - x0, y1 - y0),
            radii: [0.0; CORNERS],
            poly: Some((verts, n as u8)),
        }
    }

    fn effective_clip(&self) -> EffectiveClip {
        if self.clip_stack.is_empty() {
            return EffectiveClip::None;
        }
        match stack_bound(&self.clip_stack) {
            Some(r) => EffectiveClip::Rect(r),
            None => EffectiveClip::Empty,
        }
    }

    /// Bring the full-surface clip mask up to date after a
    /// PushClip/PopClip, reusing the backend's persistent buffer in
    /// place and rewriting only the bytes whose clip membership
    /// changed.
    ///
    /// The buffer is allocated (or grown) only when the surface size
    /// changes; same-size updates touch O(changed pixels), not
    /// O(surface) — a clip op that leaves the effective clip unchanged
    /// (e.g. pushing a rect that contains the current clip) writes
    /// nothing at all. The buffer's current content is tracked exactly
    /// in `mask_region` (a fresh buffer is zero-filled by
    /// [`Mask::new`], i.e. the empty region), so the diff is always
    /// sound, across ops and across frames. When the stack is empty the
    /// parked buffer is kept for the next clip op but not consulted
    /// (see [`Painter::active_clip_mask`]).
    fn update_clip_mask(&mut self) {
        if self.clip_stack.is_empty() {
            return;
        }
        let (w, h) = (self.px.width(), self.px.height());
        let reuse = self
            .clip_mask
            .as_ref()
            .is_some_and(|m| m.width() == w && m.height() == h);
        if !reuse {
            *self.clip_mask = Mask::new(w, h);
            // A fresh buffer is zero-filled: it encodes the empty
            // region and the empty clip, whatever the tracked state
            // said before the resize.
            *self.mask_region = None;
            self.mask_stack.clear();
        } // Compute the clip before borrowing the mask buffer: the borrow
        // of `clip_mask` below is disjoint from the `&self` read here,
        // but the borrow checker sees the method call as a whole-`self`
        // borrow.
        let all_rect = self.clip_stack.iter().all(ClipEntry::is_rect);
        let old_all_rect = self.mask_stack.iter().all(ClipEntry::is_rect);
        if all_rect && old_all_rect {
            // Fast path: every entry is a plain rect, so the bytes are
            // a function of the intersected bound alone. A clip op that
            // leaves the bound unchanged (e.g. pushing a rect that
            // contains the current clip) writes nothing at all.
            let new_bound = stack_bound(&self.clip_stack);
            if stack_bound(self.mask_stack) == new_bound {
                return;
            }
            self.update_clip_mask_rect(w as usize, h as usize, new_bound);
            return;
        }
        // Rounded path: the bytes are a function of the whole stack's
        // geometric intersection (see `crate::clip`), so the
        // early-return compares the stacks entry-wise.
        if *self.mask_stack == self.clip_stack {
            return;
        }
        let new_bound = stack_bound(&self.clip_stack);
        let new_region = clip_mask_region(new_bound, w as usize, h as usize);
        let old_region = *self.mask_region;
        let Some(mask) = self.clip_mask.as_mut() else {
            return;
        };
        let data = mask.data_mut();
        // Pixels leaving the bounding region go to 0 (exact set
        // difference). Pixels under the new clip get the coverage of the
        // stack's geometric intersection (see `crate::clip`); no
        // `keep`/interior shortcut — a pixel a full px inside the bound
        // can still be partial under a rounded corner.
        paint_region_diff(data, w as usize, old_region, new_region, 0);
        paint_rounded_coverage_diff(data, w as usize, new_region, &self.clip_stack);
        *self.mask_region = new_region;
        *self.mask_stack = self.clip_stack.clone();
    }

    /// All-rect half of [`update_clip_mask`](Self::update_clip_mask):
    /// the incremental mask update for a stack of plain rect clips,
    /// byte-identical to the old single-rect code.
    fn update_clip_mask_rect(&mut self, w: usize, h: usize, new_bound: Option<TsRect>) {
        let new_region = clip_mask_region(new_bound, w, h);
        let old_region = *self.mask_region;
        let Some(mask) = self.clip_mask.as_mut() else {
            return;
        };
        let data = mask.data_mut();
        // Pixels leaving the bounding region go to 0 (exact set
        // difference). Pixels under the new clip get their exact
        // fractional coverage bytes, except the guaranteed-fully-
        // covered interior both clips share, which keeps its 255s:
        // only changed bytes are touched and the result is
        // byte-identical to a full rebuild.
        //
        // The shared interior is sound: for the old clip, a pixel in
        // `interior(old_region)` sits at least one pixel inside every
        // edge of old_region's bound, and old_region's bound is
        // floor/ceil-outward of the old rect — so the pixel lies fully
        // inside the old rect (byte 255 there). Same for the new clip;
        // their intersection's bytes are 255 under both.
        paint_region_diff(data, w, old_region, new_region, 0);
        let keep = match (
            old_region.and_then(mask_interior),
            new_region.and_then(mask_interior),
        ) {
            (Some(a), Some(b)) => intersect_mask_rect(a, b),
            _ => None,
        };
        if let Some(r) = new_bound {
            paint_coverage_diff(data, w, new_region, keep, &r);
        }
        *self.mask_region = new_region;
        *self.mask_stack = self.clip_stack.clone();
    }

    /// The clip mask to hand tiny-skia: `None` when the clip stack is
    /// empty (the parked buffer is kept for reuse, not consulted), and
    /// also `None` when the tracked opaque region is the whole surface
    /// with an all-integral all-rect clip: every byte is then 255, so an
    /// all-255 mask is equivalent to no mask, and tiny-skia takes its
    /// mask-free fast path instead of reading a mask byte per pixel.
    /// A fractional edge always leaves partial boundary bytes, so the
    /// mask is required even for a full-surface region; any rounded
    /// entry likewise forces the mask (rounded corners always leave
    /// partial bytes). Takes the fields (not `&self`) so callers can
    /// hold `&mut self.px` at the same time: the borrows are disjoint.
    fn active_clip_mask<'m>(
        clip_stack: &[ClipEntry],
        clip_mask: &'m Option<Mask>,
        mask_region: &Option<MaskRect>,
        mask_stack: &[ClipEntry],
    ) -> Option<&'m Mask> {
        if clip_stack.is_empty() {
            return None;
        }
        let mask = clip_mask.as_ref()?;
        // `update_clip_mask` keeps `mask_region` exactly in sync with
        // the buffer's bytes, and `mask_stack` in sync with the stack
        // those bytes were painted from, so the test below is exact.
        let full_surface = match mask_region {
            Some(r) => {
                r.x0 == 0
                    && r.y0 == 0
                    && r.x1 == mask.width() as usize
                    && r.y1 == mask.height() as usize
            }
            None => false,
        };
        let integral = mask_stack.iter().all(ClipEntry::is_rect)
            && match stack_bound(mask_stack) {
                Some(r) => {
                    r.x().fract() == 0.0
                        && r.y().fract() == 0.0
                        && r.right().fract() == 0.0
                        && r.bottom().fract() == 0.0
                }
                // Empty clip: the buffer is all zero, never all-255.
                None => false,
            };
        if full_surface && integral {
            None
        } else {
            Some(mask)
        }
    }

    /// Product of the opacity stack, clamped to 0..=1.
    fn opacity(&self) -> f32 {
        self.opacity_stack.iter().product::<f32>().clamp(0.0, 1.0)
    }

    /// Map a CSS-px rect to device px and intersect with the clip's
    /// outward-snapped bound. Returns `None` when the rect is
    /// degenerate or clipped out.
    ///
    /// The geometry must NOT pre-clip to the exact clip edge: the
    /// fractional clip mask ([`Painter::active_clip_mask`]) enforces
    /// the edge per pixel, and pre-clipping would clip boundary pixels
    /// twice. The snapped bound only culls fully-outside pixels; it
    /// uses the same floor/ceil convention as [`clip_mask_region`] so
    /// no pixel the mask can cover is ever culled.
    fn dev_rect(&self, rect: Rect) -> Option<TsRect> {
        let r = self.to_device_rect(rect)?;
        match self.effective_clip() {
            EffectiveClip::None => Some(r),
            EffectiveClip::Empty => None,
            EffectiveClip::Rect(c) => {
                let (w, h) = (self.px.width() as f32, self.px.height() as f32);
                let bound = TsRect::from_xywh(
                    c.x().floor().max(0.0),
                    c.y().floor().max(0.0),
                    (c.right().ceil().min(w) - c.x().floor().max(0.0)).max(0.0),
                    (c.bottom().ceil().min(h) - c.y().floor().max(0.0)).max(0.0),
                )?;
                r.intersect(&bound)
            }
        }
    }

    fn paint_list(&mut self, list: &DisplayList) {
        // One O(items) pass matches every opacity group up front; group
        // painting below is then pure lookup, never a re-scan.
        self.group_scans = precompute_group_scans(&list.items);
        // Same for transform groups (see `precompute_transform_ends`).
        self.transform_ends = precompute_transform_ends(&list.items);
        let mut i = 0;
        while i < list.items.len() {
            i = self.paint_item(list, i);
        }
        // Unbalanced input (missing PopClip/PopOpacity) must not leak
        // folded state into later items: the stacks must drain exactly.
        debug_assert!(
            self.clip_stack.is_empty(),
            "unbalanced PushClip/PopClip: clip stack not drained"
        );
        debug_assert!(
            self.opacity_stack.is_empty(),
            "unbalanced PushOpacity/PopOpacity: opacity stack not drained"
        );
    }

    /// Paint the item at `idx`; returns the index of the next item to
    /// paint. Opacity groups consume their whole `PushOpacity`..
    /// `PopOpacity` range and return the index past it.
    fn paint_item(&mut self, list: &DisplayList, idx: usize) -> usize {
        // Opacity groups consume an item range (see
        // `paint_opacity_group`); everything else paints one item.
        if let DisplayItem::PushOpacity { opacity } = &list.items[idx] {
            return self.paint_opacity_group(list, idx, *opacity);
        }
        // Blend groups likewise consume a range (see
        // `paint_blend_group`).
        if let DisplayItem::PushBlend { mode } = &list.items[idx] {
            return self.paint_blend_group(list, idx, *mode);
        }
        // Transform groups likewise consume a range (see
        // `paint_transform_group`).
        if let DisplayItem::PushTransform { matrix } = &list.items[idx] {
            return self.paint_transform_group(list, idx, *matrix);
        }
        // Filter groups likewise consume a range (see
        // `paint_filter_group`).
        if let DisplayItem::PushFilter { ops } = &list.items[idx] {
            return self.paint_filter_group(list, idx, ops);
        }
        // Backdrop-filter groups likewise consume a range (see
        // `paint_backdrop_filter_group`).
        if let DisplayItem::PushBackdropFilter { ops, rect } = &list.items[idx] {
            return self.paint_backdrop_filter_group(list, idx, ops, *rect);
        }
        let item = &list.items[idx];
        match item {
            DisplayItem::FillRect { rect, color } => {
                self.paint_quad(&Quad {
                    rect: *rect,
                    color: *color,
                });
            }
            DisplayItem::Border {
                rect,
                widths,
                styles,
                colors,
                radii,
            } => {
                // No fallback: when tessellation yields nothing (every
                // side None/Hidden or zero-width), paint nothing,
                // matching the GPU backend.
                //
                // The tessellator fills the backend's pooled scratch
                // buffer instead of allocating a fresh Vec per bordered
                // box: after the first border the buffer's capacity
                // covers the common case and this path allocates
                // nothing. `mem::take` moves the buffer out for the
                // duration so `paint_quad` can borrow `self` while the
                // quads borrow the buffer; the (possibly grown) buffer
                // goes back into the pool afterwards.
                let mut quads = std::mem::take(self.border_scratch);
                if radii.iter().any(|r| *r > 0.0) {
                    tessellate_border_rounded_into(
                        &mut quads, *rect, *widths, *styles, *colors, *radii,
                    )
                } else {
                    tessellate_border_into(&mut quads, *rect, *widths, *styles, *colors)
                };
                for quad in &quads {
                    self.paint_quad(quad);
                }
                *self.border_scratch = quads;
            }
            DisplayItem::TextRun {
                text,
                text_range,
                rect,
                baseline,
                font_key,
                size_px,
                color,
            } => {
                self.paint_text_run(
                    text.get(text_range.clone()).unwrap_or(""),
                    *rect,
                    *baseline,
                    *font_key,
                    *size_px,
                    *color,
                );
            }
            DisplayItem::Image { rect, image } => {
                self.draw_image(list, *image, *rect);
            }
            DisplayItem::BackgroundImage { rect, image } => {
                // A single background tile: stretched to its `rect` like
                // a replaced-element image.
                self.draw_image(list, *image, *rect);
            }
            DisplayItem::RepeatImage {
                area,
                image,
                tile,
                repeat,
                offset,
            } => {
                // Tiling is expanded natively in the rasterizer (see
                // `paint_repeat_image`): no per-tile display items.
                self.paint_repeat_image(list, *area, *image, *tile, *repeat, *offset);
            }
            DisplayItem::Gradient { rect, gradient } => {
                self.paint_gradient(*rect, gradient);
            }
            DisplayItem::Decoration {
                rect,
                kind,
                color,
                thickness,
            } => {
                // Position the line from the ink box (see module docs);
                // wavy kinds share their straight counterpart's baseline
                // and are tessellated by `decoration_geometry`.
                let (y, x0, x1) = match kind {
                    DecoKind::Underline | DecoKind::WavyUnderline => {
                        (rect.y + rect.h - thickness, rect.x, rect.x + rect.w)
                    }
                    DecoKind::Overline | DecoKind::WavyOverline => {
                        (rect.y, rect.x, rect.x + rect.w)
                    }
                    DecoKind::LineThrough | DecoKind::WavyLineThrough => {
                        (rect.y + (rect.h - thickness) / 2.0, rect.x, rect.x + rect.w)
                    }
                };
                for quad in &decoration_geometry(*kind, x0, x1, y, *thickness, *color) {
                    self.paint_quad(quad);
                }
            }
            DisplayItem::ListMarker {
                rect,
                kind,
                color,
                index,
                font_key,
            } => {
                // Decimal numerals are text, not geometry: the numeral
                // comes from the item's 1-based index, formatted into
                // a stack buffer so the per-frame marker path never
                // touches the heap (same formula as the GPU backend).
                // Rendered through the text path with the item's font
                // at the GPU's size and baseline.
                let mut num_buf = [0u8; 20];
                let owned_marker: Option<String>;
                let numeral: Option<&str> = match kind {
                    MarkerKind::Decimal(_) => {
                        Some(format_marker_numeral(&mut num_buf, *index, false))
                    }
                    MarkerKind::DecimalLeadingZero(_) => {
                        Some(format_marker_numeral(&mut num_buf, *index, true))
                    }
                    _ => {
                        owned_marker = marker_text(*kind);
                        owned_marker.as_deref()
                    }
                };
                if let Some(numeral) = numeral {
                    let size_px = rect.w.min(rect.h).max(1.0);
                    let baseline = rect.y + 0.8 * (rect.h - size_px) + size_px;
                    self.paint_text_run(numeral, *rect, baseline, *font_key, size_px, *color);
                } else {
                    // Centered in `rect`, diameter = min(w, h).
                    let x = rect.x + rect.w / 2.0;
                    let y = rect.y + rect.h / 2.0;
                    let size = rect.w.min(rect.h);
                    for quad in &marker_geometry(*kind, x, y, size, *color) {
                        self.paint_quad(quad);
                    }
                }
            }
            DisplayItem::PushClip { rect } => {
                self.clip_stack
                    .push(ClipEntry::rect(self.to_device_rect(*rect)));
                self.update_clip_mask();
            }
            DisplayItem::PushRoundedClip { rect, radii } => {
                self.clip_stack
                    .push(self.to_device_rounded_rect(*rect, *radii));
                self.update_clip_mask();
            }
            DisplayItem::PushClipPath { points } => {
                self.clip_stack.push(self.to_device_polygon(points));
                self.update_clip_mask();
            }
            DisplayItem::PopClip => {
                self.clip_stack.pop();
                self.update_clip_mask();
            }
            DisplayItem::PushOpacity { .. } => {
                // Handled above: opacity groups consume a range.
                debug_assert!(false, "PushOpacity must go through paint_opacity_group");
            }
            DisplayItem::PopOpacity => {
                self.opacity_stack.pop();
            }
            DisplayItem::PushBlend { .. } => {
                // Handled above: blend groups consume a range.
                debug_assert!(false, "PushBlend must go through paint_blend_group");
            }
            DisplayItem::PushFilter { .. } => {
                // Handled above: filter groups consume a range.
                debug_assert!(false, "PushFilter must go through paint_filter_group");
            }
            // No filter stack: filter groups consume their whole range,
            // so the pop is always a no-op (like PopBlend).
            DisplayItem::PopFilter => {}
            DisplayItem::PushBackdropFilter { .. } => {
                // Handled above: backdrop-filter groups consume a range.
                debug_assert!(
                    false,
                    "PushBackdropFilter must go through paint_backdrop_filter_group"
                );
            }
            // No backdrop-filter stack: the group consumes its whole
            // range, so the pop is always a no-op.
            DisplayItem::PopBackdropFilter => {}
            DisplayItem::PushTransform { .. } => {
                // Handled above: transform groups consume a range.
                debug_assert!(false, "PushTransform must go through paint_transform_group");
            }
            // No transform stack: transform groups consume their whole
            // range (like blend groups), so the pop is always a no-op.
            DisplayItem::PopTransform => {}
            DisplayItem::BoxShadow {
                rect,
                radii,
                color,
                offset,
                blur,
                spread,
                inset,
            } => {
                self.paint_box_shadow(*rect, *radii, *color, *offset, *blur, *spread, *inset);
            }
            DisplayItem::TextShadow {
                text,
                text_range,
                rect,
                baseline,
                font_key,
                size_px,
                color,
                offset,
                blur,
            } => {
                self.paint_text_shadow(
                    text.get(text_range.clone()).unwrap_or(""),
                    *rect,
                    *baseline,
                    *font_key,
                    *size_px,
                    *color,
                    *offset,
                    *blur,
                );
            }
            // No blend stack: blend groups consume their whole range
            // (like opacity groups), and `Normal` groups paint inline,
            // so the pop is always a no-op.
            DisplayItem::PopBlend => {}
            DisplayItem::EdgeGlow {
                edge,
                rect,
                strength,
            } => {
                self.paint_edge_glow(*edge, *rect, *strength);
            }
        }
        idx + 1
    }

    /// Look up the precomputed [`scan_opacity_group`] tuple for the
    /// `PushOpacity` at `idx`. Every `PushOpacity` in the list has an
    /// entry; the on-the-fly scan is a defensive fallback only.
    fn group_scan(&self, list: &DisplayList, idx: usize) -> (usize, usize, bool, Option<usize>) {
        self.group_scans
            .get(&idx)
            .copied()
            .unwrap_or_else(|| scan_opacity_group(&list.items, idx))
    }

    /// Check out a bounds-sized pixmap for an isolated opacity group
    /// from the backend's pool ([`MAX_POOLED_GROUP_PIXMAPS`]); exact-size
    /// matches only — pooled pixmaps come in the sizes past groups used,
    /// so an exact size match is the (width, height) bucket. A miss allocates a
    /// fresh pixmap, exactly as before pooling. Checked-out entries are
    /// zeroed (transparent black): pooled pixmaps may still hold a
    /// previous group's rendering, and compositing would otherwise
    /// blend stale pixels into the new group. Nested groups check out
    /// independently — each checkout removes its pixmap from the pool,
    /// so parent and child can never alias.
    fn checkout_group_pixmap(&mut self, w: u32, h: u32) -> Option<Pixmap> {
        self.cache_stats.pool_checkouts += 1;
        if let Some(idx) = self
            .group_pixmap_pool
            .iter()
            .position(|p| p.width() == w && p.height() == h)
        {
            let mut px = self.group_pixmap_pool.swap_remove(idx);
            px.data_mut().fill(0);
            self.cache_stats.pool_reuses += 1;
            return Some(px);
        }
        Pixmap::new(w, h)
    }

    /// Return a group pixmap to the backend's pool after its composite.
    /// The pool is capped at [`MAX_POOLED_GROUP_PIXMAPS`]: past the cap
    /// the pixmap is dropped instead of pinning another buffer across all
    /// later frames.
    fn return_group_pixmap(&mut self, px: Pixmap) {
        if self.group_pixmap_pool.len() < MAX_POOLED_GROUP_PIXMAPS {
            self.group_pixmap_pool.push(px);
        }
    }

    /// Check out a scratch byte buffer of exactly `len` from the
    /// backend's shadow scratch pool ([`MAX_POOLED_SHADOW_SCRATCH`]);
    /// exact-length matches only, mirroring
    /// [`checkout_group_pixmap`](Self::checkout_group_pixmap). A miss
    /// allocates a fresh zeroed buffer, exactly as before pooling.
    /// Nested shadow paints check out independently — each checkout
    /// removes its buffer from the pool, so two live temps can never
    /// alias. The buffer's contents are *not* zeroed: every caller
    /// overwrites its temp before reading it (use
    /// [`checkout_shadow_scratch_zeroed`](Self::checkout_shadow_scratch_zeroed)
    /// when that is not true).
    fn checkout_shadow_scratch(&mut self, len: usize) -> Vec<u8> {
        self.cache_stats.pool_checkouts += 1;
        if let Some(idx) = self.shadow_scratch_pool.iter().position(|b| b.len() == len) {
            self.cache_stats.pool_reuses += 1;
            return self.shadow_scratch_pool.swap_remove(idx);
        }
        vec![0u8; len]
    }

    /// Like [`checkout_shadow_scratch`](Self::checkout_shadow_scratch),
    /// but the buffer is zeroed first — for temps that are read without
    /// being fully written, like the box shadow's coverage mask on the
    /// empty-shape path (a zeroed coverage is the only correct value
    /// there).
    fn checkout_shadow_scratch_zeroed(&mut self, len: usize) -> Vec<u8> {
        let mut buf = self.checkout_shadow_scratch(len);
        buf.fill(0);
        buf
    }

    /// Return a shadow scratch buffer to the backend's pool after the
    /// shadow is done. The pool is capped at
    /// [`MAX_POOLED_SHADOW_SCRATCH`] buffers of at most
    /// [`MAX_POOLED_SHADOW_SCRATCH_BYTES`] bytes each: past either cap
    /// the buffer is dropped instead of pinning memory across all later
    /// frames.
    fn return_shadow_scratch(&mut self, buf: Vec<u8>) {
        if buf.len() <= MAX_POOLED_SHADOW_SCRATCH_BYTES
            && self.shadow_scratch_pool.len() < MAX_POOLED_SHADOW_SCRATCH
        {
            self.shadow_scratch_pool.push(buf);
        }
    }

    /// Check out a group pixmap sized to the paint bounds of
    /// `items[start..end]` (see [`group_paint_bounds`]), snapped outward
    /// to device px and intersected with the live clip and the current
    /// target surface. Returns the pixmap plus its device-px origin in
    /// the current target's coordinates. `None` when the group paints
    /// nothing visible (empty bounds or fully clipped out) — the caller
    /// then skips the group entirely, since compositing would be a no-op.
    fn checkout_bounded_group_pixmap(
        &mut self,
        items: &[DisplayItem],
        start: usize,
        end: usize,
    ) -> Option<(Pixmap, f32, f32)> {
        self.checkout_bounded_group_pixmap_inflated(items, start, end, 0.0)
    }

    /// Like [`Painter::checkout_bounded_group_pixmap`], but the content
    /// bounds inflate by `inflate_page` page px first: filter groups
    /// (blur, drop-shadow) paint outside their content bounds, and the
    /// offscreen pixmap must cover the spill or it gets culled.
    fn checkout_bounded_group_pixmap_inflated(
        &mut self,
        items: &[DisplayItem],
        start: usize,
        end: usize,
        inflate_page: f32,
    ) -> Option<(Pixmap, f32, f32)> {
        let mut page = group_paint_bounds(items, start, end)?;
        if inflate_page > 0.0 && inflate_page.is_finite() {
            page = Rect {
                x: page.x - inflate_page,
                y: page.y - inflate_page,
                w: page.w + 2.0 * inflate_page,
                h: page.h + 2.0 * inflate_page,
            };
        }
        // Snap outward in device px (current origin): the pixmap must
        // cover every painted pixel.
        let mut x0 = self.dev_x(page.x).floor();
        let mut y0 = self.dev_y(page.y).floor();
        let mut x1 = self.dev_x(page.x + page.w).ceil();
        let mut y1 = self.dev_y(page.y + page.h).ceil();
        // The group cannot affect anything outside the live clip or the
        // target surface.
        let (pw, ph) = (self.px.width() as f32, self.px.height() as f32);
        x0 = x0.max(0.0);
        y0 = y0.max(0.0);
        x1 = x1.min(pw);
        y1 = y1.min(ph);
        match self.effective_clip() {
            EffectiveClip::None => {}
            EffectiveClip::Empty => return None,
            EffectiveClip::Rect(c) => {
                // Outward snap keeps the origin integral: the group
                // pixmap's pixel grid must align with the target's, or
                // fractional clip edges would quantize to different
                // bytes inside the group than in the parent (clip
                // drift). The clip still applies inside the group via
                // its own mask; the bounds only cull.
                x0 = x0.max(c.x().floor());
                y0 = y0.max(c.y().floor());
                x1 = x1.min((c.x() + c.width()).ceil());
                y1 = y1.min((c.y() + c.height()).ceil());
            }
        }
        // `x0`/`y0` are floors and `x1`/`y1` ceils, so the differences
        // are integral-valued; the `max` guards the clipped-empty case.
        let (gw, gh) = ((x1 - x0).max(0.0) as u32, (y1 - y0).max(0.0) as u32);
        if gw == 0 || gh == 0 {
            return None;
        }
        // `checkout_group_pixmap` only fails for zero-size targets,
        // excluded above; the checkout is zeroed, exactly like
        // `Pixmap::new`.
        let px = self.checkout_group_pixmap(gw, gh)?;
        Some((px, x0, y0))
    }

    /// Translate every clip-stack rect by `(dx, dy)` device px, for
    /// entering/exiting a bounded opacity group whose origin moved (see
    /// `dev_origin`). Shift-then-shift-back is float-exact for finite
    /// rects, so exiting restores the parent's rects bit-for-bit.
    /// Polygon vertices shift the same way (translation-invariant
    /// coverage, like the radii).
    fn shift_clips(&mut self, dx: f32, dy: f32) {
        for e in self.clip_stack.iter_mut() {
            if let Some((pts, n)) = e.poly.as_mut() {
                for p in pts[..(*n as usize).min(MAX_POLY_VERTS)].iter_mut() {
                    p[0] += dx;
                    p[1] += dy;
                }
            }
            let Some(r) = e.rect else { continue };
            // Defensive fallback: rects here are finite (they came
            // from `to_device_rect`), so the shift cannot produce a
            // non-finite rect on real input; keep the old rect rather
            // than inventing an empty clip. Radii are scale-invariant
            // under translation.
            if let Some(shifted) = TsRect::from_xywh(r.x() + dx, r.y() + dy, r.width(), r.height())
            {
                e.rect = Some(shifted);
            }
        }
    }

    /// Force a full rebuild of the clip mask for the current pixmap:
    /// zero the buffer and forget the tracked region, then
    /// [`update_clip_mask`](Self::update_clip_mask). Needed when
    /// entering/exiting a bounded group — the tracked region is in the
    /// other pixmap's coordinates, and a same-size coincidence must not
    /// reuse stale bytes through the incremental diff.
    fn rebuild_clip_mask(&mut self) {
        *self.mask_region = None;
        // The buffer is zeroed: it now encodes the empty clip, whatever
        // the tracked stack said before.
        self.mask_stack.clear();
        if let Some(m) = self.clip_mask.as_mut() {
            m.data_mut().fill(0);
        }
        self.update_clip_mask();
    }

    /// Paint an opacity group starting at `idx` (a `PushOpacity`).
    /// Returns the index past the matching `PopOpacity`.
    ///
    /// True isolated-group compositing (CSS semantics): the group's
    /// items render into an offscreen pixmap at their own alphas — the
    /// group's opacity is *not* folded into them — then the pixmap is
    /// composited over the target with the group opacity. Fast paths:
    /// opacity 1.0 is a no-op (items paint with the outer opacity as if
    /// the group were absent), opacity 0.0 paints nothing, and a group
    /// holding a single *foldable* paint item with no nested groups
    /// folds the opacity into that item's alpha, which is exactly
    /// isolated compositing for one source-over. Foldable means the
    /// item's primitives cannot overlap (see [`opacity_foldable`]): a
    /// lone [`DisplayItem::TextRun`]'s glyphs can, so it always takes
    /// the isolated path.
    ///
    /// Correctness note: the group's own opacity must not be folded into
    /// an isolated subgroup's items (source-over is not linear in the
    /// folded alpha when items overlap), so only groups whose top-level
    /// content is a single paint item — and no nested opacity groups —
    /// take the fold path. Every enclosing group of an isolated group is
    /// therefore itself isolated (or a no-op), so the fold stack is empty
    /// inside isolated groups and items render at their own alphas.
    fn paint_opacity_group(&mut self, list: &DisplayList, idx: usize, opacity: f32) -> usize {
        // Non-finite opacity normalizes to 0.0: `clamp` passes NaN
        // through, and a NaN alpha must not reach the blend paths (the
        // GPU backend's blend equation is driver-undefined on NaN).
        let o = if opacity.is_finite() {
            opacity.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if o >= 1.0 {
            // Identity group: no compositing needed. Push for balance
            // (the matching PopOpacity pops); the product is unchanged.
            self.opacity_stack.push(o);
            return idx + 1;
        }
        let (end, paint_items, nested, single_item) = self.group_scan(list, idx);
        // `scan_opacity_group` returns `items.len()` when no PopOpacity
        // matches; the group paths below would then silently run to the
        // end of the list, so catch the malformed list explicitly.
        debug_assert!(
            end < list.items.len(),
            "PushOpacity without matching PopOpacity"
        );
        if o <= 0.0 {
            // Fully transparent: nothing in the group can show.
            return end + 1;
        }
        // Fold fast path: only when the single item's primitives cannot
        // overlap (see `opacity_foldable`) — a lone TextRun's glyphs can,
        // and folding would double-apply the opacity where they do.
        let foldable = single_item.is_none_or(|i| opacity_foldable(&list.items[i]));
        if paint_items <= 1 && !nested && foldable {
            // Fast path: folding into the single item's alpha is exactly
            // isolated compositing (one source-over either way).
            self.opacity_stack.push(o);
            return idx + 1;
        }
        // Isolated group: render the range into an offscreen pixmap at
        // the items' own alphas (the fold stack holds only outer,
        // already-applied opacity — empty by the note above), then
        // composite with the group opacity. Past MAX_OPACITY_GROUPS
        // nested isolated groups the opacity folds into the running
        // product instead of allocating another pixmap and
        // recursing deeper: exact only for non-overlapping primitives, a
        // documented degradation for pathological pages.
        if self.group_depth >= MAX_OPACITY_GROUPS {
            #[cfg(debug_assertions)]
            eprintln!("[bowser_paint] opacity group depth cap hit; folding group opacity");
            self.opacity_stack.push(o);
            return idx + 1;
        }
        self.group_depth += 1;
        // Bounded group target: size the offscreen pixmap to the group's
        // actual paint bounds (see `group_paint_bounds`), snapped outward
        // to device px and intersected with the live clip and the target
        // surface — a group covering a small region of a large surface
        // allocates a small pixmap, not a full-surface one. `None` when
        // the group paints nothing visible (empty bounds or fully clipped
        // out): compositing would be a no-op, so skip it entirely.
        // Decrement first: `group_depth` counts live isolated renders,
        // and this one never started.
        let Some((mut group_px, gx0, gy0)) =
            self.checkout_bounded_group_pixmap(&list.items, idx + 1, end)
        else {
            self.group_depth -= 1;
            return end + 1;
        };
        // Swap the pixmap referents (not the `&mut`s): the painter keeps
        // painting through `self.px`, now the blank bounded group pixmap,
        // with shared glyph/image caches. `dev_origin` shifts to the
        // bounds origin (nested groups accumulate) so items keep their
        // page-px coordinates while pixels and clip rects land in the
        // bounded pixmap; the clip mask rebuilds for the new size.
        std::mem::swap(&mut *self.px, &mut group_px);
        let old_origin = self.dev_origin;
        self.dev_origin = (old_origin.0 + gx0, old_origin.1 + gy0);
        self.shift_clips(-gx0, -gy0);
        self.rebuild_clip_mask();
        // A missing PopOpacity/PopClip inside the group must not corrupt
        // the outer stacks' pairing: the inner range has to leave both
        // depths exactly as found (the end-of-paint_list emptiness check
        // alone would still pass if an inner pop stole an outer entry).
        let (opacity_depth, clip_depth) = (self.opacity_stack.len(), self.clip_stack.len());
        let mut i = idx + 1;
        while i < end {
            i = self.paint_item(list, i);
        }
        debug_assert_eq!(
            self.opacity_stack.len(),
            opacity_depth,
            "unbalanced PushOpacity/PopOpacity inside isolated opacity group"
        );
        debug_assert_eq!(
            self.clip_stack.len(),
            clip_depth,
            "unbalanced PushClip/PopClip inside isolated opacity group"
        );
        std::mem::swap(&mut *self.px, &mut group_px);
        // Restore the parent's origin, clips, and clip mask before the
        // composite: the group pixmap's (0, 0) is the parent target's
        // `(gx0, gy0)`. The shift-then-shift-back is float-exact, so the
        // parent's clip rects come back bit-for-bit.
        self.dev_origin = old_origin;
        self.shift_clips(gx0, gy0);
        self.rebuild_clip_mask();
        // `group_px` now holds the isolated rendering; composite it over
        // the target at its bounds origin with the group opacity. No
        // clip mask here: the group's content was already clipped while
        // painting (the group painter ran with the clip active), so
        // outside-clip pixels are transparent — and re-applying the
        // mask would square the fractional boundary bytes instead of
        // being idempotent.
        let paint = PixmapPaint {
            opacity: o,
            ..PixmapPaint::default()
        };
        self.px.draw_pixmap(
            gx0 as i32,
            gy0 as i32,
            group_px.as_ref(),
            &paint,
            Transform::identity(),
            None,
        );
        // The frame is still painting: unlike the GPU backend (whose
        // targets stay checked out until the frame submits because a
        // pending composite may still sample them), the CPU composite
        // above is already done, so the pixmap returns to the pool
        // immediately and a sibling group later in this frame reuses it.
        self.return_group_pixmap(group_px);
        self.group_depth -= 1;
        end + 1
    }

    /// Paint a blend group starting at `idx` (a `PushBlend`).
    /// Returns the index past the matching `PopBlend`.
    ///
    /// True isolated-group compositing (CSS semantics): the group's
    /// items render into an offscreen pixmap at their own alphas, then
    /// the pixmap is blended over the target with `mode` against the
    /// backdrop (whatever the target holds at composite time).
    /// `Normal` is the no-op fast path: no isolation, items paint
    /// inline with plain source-over, exactly as if the group items
    /// were absent.
    ///
    /// The isolation dance mirrors [`paint_opacity_group`] (bounded
    /// pixmap checkout, origin/clip shifting, stack balance checks);
    /// only the composite step differs. Past [`MAX_OPACITY_GROUPS`]
    /// nested isolated groups the range paints inline with plain
    /// source-over instead of allocating another pixmap and recursing
    /// deeper: a documented degradation for pathological pages.
    fn paint_blend_group(&mut self, list: &DisplayList, idx: usize, mode: BlendMode) -> usize {
        if mode.is_normal() {
            return idx + 1;
        }
        let end = scan_blend_group(&list.items, idx);
        debug_assert!(
            end < list.items.len(),
            "PushBlend without matching PopBlend"
        );
        if self.group_depth >= MAX_OPACITY_GROUPS {
            #[cfg(debug_assertions)]
            eprintln!(
                "[bowser_paint] blend group depth cap hit; compositing inline as source-over"
            );
            let mut i = idx + 1;
            while i < end {
                i = self.paint_item(list, i);
            }
            return end + 1;
        }
        self.group_depth += 1;
        // Bounded group target: same sizing as the opacity path (see
        // `paint_opacity_group`); `None` when the group paints nothing
        // visible, so the blend would be a no-op.
        let Some((mut group_px, gx0, gy0)) =
            self.checkout_bounded_group_pixmap(&list.items, idx + 1, end)
        else {
            self.group_depth -= 1;
            return end + 1;
        };
        // Swap the pixmap referents (not the `&mut`s): the painter keeps
        // painting through `self.px`, now the blank bounded group pixmap.
        std::mem::swap(&mut *self.px, &mut group_px);
        let old_origin = self.dev_origin;
        self.dev_origin = (old_origin.0 + gx0, old_origin.1 + gy0);
        self.shift_clips(-gx0, -gy0);
        self.rebuild_clip_mask();
        // A missing PopBlend/PopClip inside the group must not corrupt
        // the outer stacks' pairing.
        let (opacity_depth, clip_depth) = (self.opacity_stack.len(), self.clip_stack.len());
        let mut i = idx + 1;
        while i < end {
            i = self.paint_item(list, i);
        }
        debug_assert_eq!(
            self.opacity_stack.len(),
            opacity_depth,
            "unbalanced PushOpacity/PopOpacity inside isolated blend group"
        );
        debug_assert_eq!(
            self.clip_stack.len(),
            clip_depth,
            "unbalanced PushClip/PopClip inside isolated blend group"
        );
        std::mem::swap(&mut *self.px, &mut group_px);
        // Restore the parent's origin, clips, and clip mask before the
        // composite: the group pixmap's (0, 0) is the parent target's
        // `(gx0, gy0)`.
        self.dev_origin = old_origin;
        self.shift_clips(gx0, gy0);
        self.rebuild_clip_mask();
        // `group_px` now holds the isolated rendering; blend-composite
        // it over the target at its bounds origin. No clip mask here:
        // the group's content was already clipped while painting (the
        // group painter ran with the clip active), and transparent
        // group pixels leave the backdrop untouched for every mode —
        // so re-applying the mask is unnecessary as well as wrong.
        self.composite_blend_group(&group_px, gx0, gy0, mode);
        // The CPU composite above is already done, so the pixmap
        // returns to the pool immediately (same as the opacity path).
        self.return_group_pixmap(group_px);
        self.group_depth -= 1;
        end + 1
    }

    /// Blend-composite an isolated group pixmap over the target at
    /// device-px `(gx0, gy0)` with `mode`, per-pixel, against the
    /// backdrop (the target's current content). Premultiplied u8
    /// in and out; the math runs in f32 (see [`blend_pixel`]).
    fn composite_blend_group(&mut self, group: &Pixmap, gx0: f32, gy0: f32, mode: BlendMode) {
        let (gw, gh) = (group.width() as i32, group.height() as i32);
        let tw = self.px.width() as i32;
        let th = self.px.height() as i32;
        // Same truncation as the opacity path's `draw_pixmap` origin.
        let (x0, y0) = (gx0 as i32, gy0 as i32);
        let src = group.data();
        let dst = self.px.data_mut();
        for gy in 0..gh {
            let ty = y0 + gy;
            if ty < 0 || ty >= th {
                continue;
            }
            for gx in 0..gw {
                let tx = x0 + gx;
                if tx < 0 || tx >= tw {
                    continue;
                }
                let so = ((gy * gw + gx) * 4) as usize;
                let sa = src[so + 3];
                // Transparent source pixel: the blend formula reduces to
                // the backdrop for every mode (Co = αb·Cb, αo = αb), so
                // skip it.
                if sa == 0 {
                    continue;
                }
                let doff = ((ty * tw + tx) * 4) as usize;
                let [r, g, b, a] = blend_pixel(
                    [src[so], src[so + 1], src[so + 2], sa],
                    [dst[doff], dst[doff + 1], dst[doff + 2], dst[doff + 3]],
                    mode,
                );
                dst[doff] = r;
                dst[doff + 1] = g;
                dst[doff + 2] = b;
                dst[doff + 3] = a;
            }
        }
    }

    /// Paint a `PushFilter` group: render the range isolated (exactly
    /// like [`Painter::paint_blend_group`]), run the filter chain's
    /// passes over the isolated pixmap (see
    /// [`Painter::apply_filter_passes`]), then source-over composite it
    /// at its bounds origin. Returns the index after the matching
    /// `PopFilter`.
    fn paint_filter_group(&mut self, list: &DisplayList, idx: usize, ops: &[FilterOp]) -> usize {
        let end = scan_filter_group(&list.items, idx);
        debug_assert!(
            end < list.items.len(),
            "PushFilter without matching PopFilter"
        );
        // The builder never emits empty filter groups; stay total on
        // hand-built lists by painting the range inline, unfiltered.
        if ops.is_empty() {
            let mut i = idx + 1;
            while i < end {
                i = self.paint_item(list, i);
            }
            return end + 1;
        }
        if self.group_depth >= MAX_OPACITY_GROUPS {
            #[cfg(debug_assertions)]
            eprintln!("[bowser_paint] filter group depth cap hit; compositing inline unfiltered");
            let mut i = idx + 1;
            while i < end {
                i = self.paint_item(list, i);
            }
            return end + 1;
        }
        self.group_depth += 1;
        // The filter can spill outside the content bounds (blur,
        // drop-shadow): inflate the group pixmap by the chain's
        // conservative page-px spill so filtered pixels are not culled.
        // `None` when the group paints nothing visible, exactly like the
        // blend path.
        let Some((mut group_px, gx0, gy0)) = self.checkout_bounded_group_pixmap_inflated(
            &list.items,
            idx + 1,
            end,
            filter_spill(ops),
        ) else {
            self.group_depth -= 1;
            return end + 1;
        };
        // Isolation dance, identical to `paint_blend_group`.
        std::mem::swap(&mut *self.px, &mut group_px);
        let old_origin = self.dev_origin;
        self.dev_origin = (old_origin.0 + gx0, old_origin.1 + gy0);
        self.shift_clips(-gx0, -gy0);
        self.rebuild_clip_mask();
        let (opacity_depth, clip_depth) = (self.opacity_stack.len(), self.clip_stack.len());
        let mut i = idx + 1;
        while i < end {
            i = self.paint_item(list, i);
        }
        debug_assert_eq!(
            self.opacity_stack.len(),
            opacity_depth,
            "unbalanced PushOpacity/PopOpacity inside isolated filter group"
        );
        debug_assert_eq!(
            self.clip_stack.len(),
            clip_depth,
            "unbalanced PushClip/PopClip inside isolated filter group"
        );
        std::mem::swap(&mut *self.px, &mut group_px);
        self.dev_origin = old_origin;
        self.shift_clips(gx0, gy0);
        self.rebuild_clip_mask();
        // `group_px` now holds the isolated rendering; filter it in
        // place, then source-over composite it at its bounds origin (no
        // clip mask: the content was already clipped while painting).
        self.apply_filter_passes(&mut group_px, ops);
        let paint = PixmapPaint::default();
        self.px.draw_pixmap(
            gx0 as i32,
            gy0 as i32,
            group_px.as_ref(),
            &paint,
            tiny_skia::Transform::identity(),
            None,
        );
        self.return_group_pixmap(group_px);
        self.group_depth -= 1;
        end + 1
    }

    /// Run a filter chain's passes over an isolated group pixmap
    /// (premultiplied RGBA8, device px) in place. `plan_filter_passes`
    /// folds the chain first, so this is one pass per merged step.
    /// Lengths (blur radii, shadow offsets) scale from CSS px to device
    /// px here — the passes after this point are unit-free.
    fn apply_filter_passes(&mut self, px: &mut Pixmap, ops: &[FilterOp]) {
        let passes = plan_filter_passes(ops);
        if passes.is_empty() {
            return;
        }
        let (w, h) = (px.width() as usize, px.height() as usize);
        let Some(len) = w.checked_mul(h).and_then(|n| n.checked_mul(4)) else {
            return;
        };
        if len == 0 {
            return;
        }
        // Two same-size scratch buffers (blur needs one temp,
        // drop-shadow needs two): one reused backend allocation, grown
        // to the largest group seen, split in half.
        let need = len.saturating_mul(2);
        if self.filter_scratch.len() < need {
            self.filter_scratch.resize(need, 0);
        }
        let (scratch, tmp) = self.filter_scratch[..need].split_at_mut(len);
        let buf = px.data_mut();
        for pass in &passes {
            match pass {
                FilterPass::Color(m) => apply_color_matrix_rgba8(buf, m),
                // σ = radius / 2 (the `box-shadow` blur convention),
                // scaled to device px.
                FilterPass::Blur(r) => {
                    box_blur_sigma_rgba(buf, tmp, w, h, r * self.scale / 2.0);
                }
                FilterPass::DropShadow {
                    offset,
                    blur,
                    color,
                } => {
                    apply_drop_shadow(
                        buf,
                        (w, h),
                        scratch,
                        tmp,
                        (offset.0 * self.scale, offset.1 * self.scale),
                        blur * self.scale,
                        *color,
                    );
                }
            }
        }
    }

    /// Paint a `PushBackdropFilter` group: capture the already-painted
    /// backdrop under the element's border box, run the filter chain's
    /// passes over the capture, write it back, then paint the group's
    /// content over it (source-over, inline — the element's own
    /// rendering is *not* isolated). Returns the index after the
    /// matching `PopBackdropFilter`.
    ///
    /// The filter region is exactly the border box (Filter Effects 2:
    /// the Backdrop Root Image is the backdrop clipped to the border
    /// box); blurs clamp at its edges like every other blur here.
    fn paint_backdrop_filter_group(
        &mut self,
        list: &DisplayList,
        idx: usize,
        ops: &[FilterOp],
        rect: Rect,
    ) -> usize {
        let end = scan_filter_group(&list.items, idx);
        debug_assert!(
            end < list.items.len(),
            "PushBackdropFilter without matching PopBackdropFilter"
        );
        // The builder never emits empty backdrop-filter groups; stay
        // total on hand-built lists by painting the range inline.
        if !ops.is_empty() {
            // Device-px border box, outward-snapped, clamped to the
            // target surface.
            let (tw, th) = (self.px.width() as f32, self.px.height() as f32);
            let x0 = self.dev_x(rect.x).floor().max(0.0);
            let y0 = self.dev_y(rect.y).floor().max(0.0);
            let x1 = self.dev_x(rect.x + rect.w).ceil().min(tw);
            let y1 = self.dev_y(rect.y + rect.h).ceil().min(th);
            if x1 > x0 && y1 > y0 {
                let (w, h) = ((x1 - x0) as u32, (y1 - y0) as u32);
                if let Some(mut cap) = self.checkout_group_pixmap(w, h) {
                    // Capture the backdrop region.
                    {
                        let src = self.px.data();
                        let dst = cap.data_mut();
                        let (tw, x0i) = (tw as usize, x0 as usize);
                        for y in 0..h as usize {
                            let soff = ((y0 as usize + y) * tw + x0i) * 4;
                            let doff = y * w as usize * 4;
                            dst[doff..doff + w as usize * 4]
                                .copy_from_slice(&src[soff..soff + w as usize * 4]);
                        }
                    }
                    // Filter the capture in place, then write it back
                    // masked by the current clip: the filtered backdrop
                    // is part of the element's rendering, so the
                    // element's clip-path (and ancestor clips) clip it
                    // exactly like content. Per pixel: out = src*q +
                    // dst*(1-src_a*q) with q the clip coverage byte —
                    // the same idiom the paint paths use, and the same
                    // formula the WGPU `composite_clip` pipeline
                    // evaluates.
                    self.apply_filter_passes(&mut cap, ops);
                    {
                        let src = cap.data();
                        // The mask is rebuilt on every clip push/pop
                        // while the stack is non-empty, so it is current
                        // here; no clip → q = 255 everywhere.
                        let mask = self.clip_mask.as_ref().map(|m| m.data());
                        let no_clip = self.clip_stack.is_empty();
                        let dst = self.px.data_mut();
                        let (tw, x0i, y0i) = (tw as usize, x0 as usize, y0 as usize);
                        for y in 0..h as usize {
                            for x in 0..w as usize {
                                let q = if no_clip {
                                    255
                                } else {
                                    mask.map(|d| d[(y0i + y) * tw + x0i + x]).unwrap_or(255)
                                };
                                if q == 0 {
                                    continue;
                                }
                                let soff = (y * w as usize + x) * 4;
                                let doff = ((y0i + y) * tw + x0i + x) * 4;
                                // Fold q into the premultiplied source,
                                // then source-over.
                                let sa = (u32::from(src[soff + 3]) * u32::from(q) + 127) / 255;
                                if sa == 0 {
                                    continue;
                                }
                                let sr = (u32::from(src[soff]) * u32::from(q) + 127) / 255;
                                let sg = (u32::from(src[soff + 1]) * u32::from(q) + 127) / 255;
                                let sb = (u32::from(src[soff + 2]) * u32::from(q) + 127) / 255;
                                let inv = 255 - sa;
                                let da = u32::from(dst[doff + 3]);
                                dst[doff] =
                                    (sr + (u32::from(dst[doff]) * inv + 127) / 255).min(255) as u8;
                                dst[doff + 1] = (sg + (u32::from(dst[doff + 1]) * inv + 127) / 255)
                                    .min(255) as u8;
                                dst[doff + 2] = (sb + (u32::from(dst[doff + 2]) * inv + 127) / 255)
                                    .min(255) as u8;
                                dst[doff + 3] = (sa + (da * inv + 127) / 255).min(255) as u8;
                            }
                        }
                    }
                    self.return_group_pixmap(cap);
                }
            }
        }
        // The element's own content paints over the filtered backdrop.
        // A missing pop inside must not corrupt the outer stacks'
        // pairing (same guard as the isolated group paths).
        let (opacity_depth, clip_depth) = (self.opacity_stack.len(), self.clip_stack.len());
        let mut i = idx + 1;
        while i < end {
            i = self.paint_item(list, i);
        }
        debug_assert_eq!(
            self.opacity_stack.len(),
            opacity_depth,
            "unbalanced PushOpacity/PopOpacity inside backdrop-filter group"
        );
        debug_assert_eq!(
            self.clip_stack.len(),
            clip_depth,
            "unbalanced PushClip/PopClip inside backdrop-filter group"
        );
        end + 1
    }

    /// Look up the precomputed [`precompute_transform_ends`] entry for
    /// the `PushTransform` at `idx`; the on-the-fly scan is a defensive
    /// fallback only.
    fn transform_end(&self, list: &DisplayList, idx: usize) -> usize {
        self.transform_ends
            .get(&idx)
            .copied()
            .unwrap_or_else(|| scan_transform_group(&list.items, idx))
    }

    /// Paint a transform group starting at `idx` (a `PushTransform`).
    /// Returns the index past the matching `PopTransform`.
    ///
    /// True isolated-group compositing (CSS semantics): the group's
    /// items render into an offscreen pixmap untransformed — at their
    /// own alphas, with the group's inner clips active — then the
    /// pixmap is warped through the matrix onto the target (bilinear).
    /// Clips pushed outside the group are drained before painting and
    /// apply per-pixel to the warped result via the live clip mask, so
    /// an ancestor `overflow: hidden` clips the transformed pixels,
    /// not the untransformed ones (matching the GPU backend, which
    /// lowers the group with empty clips and composites with the outer
    /// clip state).
    ///
    /// Fast paths: the identity matrix paints inline (no group); a
    /// non-finite or non-invertible matrix paints nothing (per CSS
    /// Transforms 1 the element renders nothing). Past
    /// [`MAX_OPACITY_GROUPS`] nested isolated groups the range paints
    /// inline untransformed: a documented degradation for pathological
    /// pages, mirroring the blend path.
    fn paint_transform_group(&mut self, list: &DisplayList, idx: usize, matrix: [f32; 6]) -> usize {
        let end = self.transform_end(list, idx);
        if matrix == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] {
            // Identity: no compositing needed. The matching
            // PopTransform is a no-op below.
            return idx + 1;
        }
        debug_assert!(
            end < list.items.len(),
            "PushTransform without matching PopTransform"
        );
        if !invertible_matrix(matrix) {
            // Non-finite or non-invertible: per CSS Transforms 1 the
            // element renders nothing.
            return end + 1;
        }
        let [a, b, c, d, e, f] = matrix;
        let det = a * d - b * c;
        // Degrade like the blend path past the cap: paint inline,
        // untransformed.
        if self.group_depth >= MAX_OPACITY_GROUPS {
            #[cfg(debug_assertions)]
            eprintln!("[bowser_paint] transform group depth cap hit; painting untransformed");
            let mut i = idx + 1;
            while i < end {
                i = self.paint_item(list, i);
            }
            return end + 1;
        }
        let s = self.scale;
        if !s.is_finite() || s <= 0.0 {
            let mut i = idx + 1;
            while i < end {
                i = self.paint_item(list, i);
            }
            return end + 1;
        }
        self.group_depth += 1;
        // Absolute device-px matrix `M_abs`: `dest_dev = M_abs(src_dev)`
        // with both sides in the current target's device coordinates
        // (`dev_x`/`dev_y` space). Conjugating the page-space matrix
        // with the page<->device mapping (uniform scale `s`, scroll,
        // current group origin `(ox, oy)`):
        //   dev(M(p)) = s*(M(p) - sc) - o,  p = (src_dev + o)/s + sc.
        let (scx, scy) = self.scroll;
        let (ox, oy) = self.dev_origin;
        let e_abs = a * ox + c * oy + s * (a * scx + c * scy + e - scx) - ox;
        let f_abs = b * ox + d * oy + s * (b * scx + d * scy + f - scy) - oy;
        // Untransformed paint bounds in absolute device px (unsnapped).
        // Transform-aware (see `group_paint_bounds`): nested groups
        // contribute their transformed extents.
        let page = match group_paint_bounds(&list.items, idx + 1, end) {
            Some(r) => r,
            None => {
                self.group_depth -= 1;
                return end + 1;
            }
        };
        let ux0 = self.dev_x(page.x);
        let uy0 = self.dev_y(page.y);
        let ux1 = self.dev_x(page.x + page.w);
        let uy1 = self.dev_y(page.y + page.h);
        if !(ux1 > ux0 && uy1 > uy0) {
            self.group_depth -= 1;
            return end + 1;
        }
        // Only source texels that can land on the target are needed:
        // the inverse matrix maps the target corners back to source
        // space. (Ancestor clips apply post-transform at composite
        // time, so they must not bound the source here: untransformed
        // content outside an ancestor clip can transform inside it.)
        let (tw, th) = (self.px.width() as f32, self.px.height() as f32);
        let inv_det = 1.0 / det;
        let src_of = |dx: f32, dy: f32| {
            let x = dx - e_abs;
            let y = dy - f_abs;
            ((d * x - c * y) * inv_det, (a * y - b * x) * inv_det)
        };
        let mut qx0 = f32::INFINITY;
        let mut qy0 = f32::INFINITY;
        let mut qx1 = f32::NEG_INFINITY;
        let mut qy1 = f32::NEG_INFINITY;
        for (cx, cy) in [(0.0, 0.0), (tw, 0.0), (0.0, th), (tw, th)] {
            let (qx, qy) = src_of(cx, cy);
            qx0 = qx0.min(qx);
            qy0 = qy0.min(qy);
            qx1 = qx1.max(qx);
            qy1 = qy1.max(qy);
        }
        // Snap the pixmap origin outward (same pixel-grid alignment as
        // the opacity path); the pixmap holds the untransformed
        // content, so the warp below maps through `M_abs` directly.
        let gx0 = qx0.max(ux0).floor();
        let gy0 = qy0.max(uy0).floor();
        let gx1 = qx1.min(ux1).ceil();
        let gy1 = qy1.min(uy1).ceil();
        let (gw, gh) = ((gx1 - gx0).max(0.0) as u32, (gy1 - gy0).max(0.0) as u32);
        if gw == 0 || gh == 0 {
            self.group_depth -= 1;
            return end + 1;
        }
        let Some(mut group_px) = self.checkout_group_pixmap(gw, gh) else {
            self.group_depth -= 1;
            return end + 1;
        };
        // Drain ancestor clips: they apply post-transform at composite
        // time (see `warp_composite`). The mask rebuilds empty for the
        // group paint, then from the restored stack before compositing.
        let saved_clips = std::mem::take(&mut self.clip_stack);
        std::mem::swap(&mut *self.px, &mut group_px);
        let old_origin = self.dev_origin;
        self.dev_origin = (old_origin.0 + gx0, old_origin.1 + gy0);
        self.rebuild_clip_mask();
        // A missing PopOpacity/PopClip inside the group must not corrupt
        // the outer stacks' pairing (same discipline as the opacity
        // path).
        let (opacity_depth, clip_depth) = (self.opacity_stack.len(), self.clip_stack.len());
        let mut i = idx + 1;
        while i < end {
            i = self.paint_item(list, i);
        }
        debug_assert_eq!(
            self.opacity_stack.len(),
            opacity_depth,
            "unbalanced PushOpacity/PopOpacity inside transform group"
        );
        debug_assert_eq!(
            self.clip_stack.len(),
            clip_depth,
            "unbalanced PushClip/PopClip inside transform group"
        );
        std::mem::swap(&mut *self.px, &mut group_px);
        self.dev_origin = old_origin;
        self.clip_stack = saved_clips;
        self.rebuild_clip_mask();
        // Warp composite: dest range = transformed pixmap bbox,
        // intersected with the target.
        let mut dx0 = f32::INFINITY;
        let mut dy0 = f32::INFINITY;
        let mut dx1 = f32::NEG_INFINITY;
        let mut dy1 = f32::NEG_INFINITY;
        for (lx, ly) in [
            (0.0, 0.0),
            (gw as f32, 0.0),
            (0.0, gh as f32),
            (gw as f32, gh as f32),
        ] {
            let wx = a * (lx + gx0) + c * (ly + gy0) + e_abs;
            let wy = b * (lx + gx0) + d * (ly + gy0) + f_abs;
            dx0 = dx0.min(wx);
            dy0 = dy0.min(wy);
            dx1 = dx1.max(wx);
            dy1 = dy1.max(wy);
        }
        let (ix0, iy0) = (dx0.floor().max(0.0) as i32, dy0.floor().max(0.0) as i32);
        let (ix1, iy1) = (dx1.ceil().min(tw) as i32, dy1.ceil().min(th) as i32);
        if ix1 > ix0 && iy1 > iy0 {
            // Inverse matrix into pixmap-local coords: dest_dev ->
            // src_abs via M_abs⁻¹, then shift by the pixmap origin.
            // u = (d*(dx-e_abs) - c*(dy-f_abs))/det - gx0, likewise v.
            let ku = -d * e_abs + c * f_abs - gx0 * det;
            let kv = -a * f_abs + b * e_abs - gy0 * det;
            let inv = [d / det, -b / det, -c / det, a / det, ku / det, kv / det];
            self.warp_composite(&group_px, inv, ix0, iy0, ix1, iy1);
        }
        self.return_group_pixmap(group_px);
        self.group_depth -= 1;
        end + 1
    }

    /// Composite `src` (premultiplied RGBA) over the current target
    /// through the inverse device-px matrix `inv`
    /// (`u = inv[0]*dx + inv[2]*dy + inv[4]`,
    /// `v = inv[1]*dx + inv[3]*dy + inv[5]`), bilinearly sampled with
    /// edge clamping, over the dest pixel range `[x0, x1) × [y0, y1)`
    /// (already intersected with the target), honoring the live clip
    /// mask per pixel. Texel `(i, j)` is hit exactly at integer
    /// `(u, v)`, so integer translations resample exactly and
    /// unscaled translated groups stay pixel-sharp.
    fn warp_composite(&mut self, src: &Pixmap, inv: [f32; 6], x0: i32, y0: i32, x1: i32, y1: i32) {
        let (sw, sh) = (src.width() as i32, src.height() as i32);
        if sw <= 0 || sh <= 0 {
            return;
        }
        let sdata = src.data();
        let tw = self.px.width() as i32;
        // Disjoint from `px`: the borrow checker sees field projections.
        let mask = Self::active_clip_mask(
            &self.clip_stack,
            self.clip_mask,
            self.mask_region,
            self.mask_stack,
        );
        let (mw, mdata) = match mask {
            Some(m) => (m.width() as i32, m.data()),
            None => (0, &[][..]),
        };
        let dst = self.px.data_mut();
        for dy in y0..y1 {
            for dx in x0..x1 {
                let q = if mask.is_some() {
                    mdata[(dy * mw + dx) as usize] as f32 / 255.0
                } else {
                    1.0
                };
                if q <= 0.0 {
                    continue;
                }
                let fx = dx as f32;
                let fy = dy as f32;
                let u = inv[0] * fx + inv[2] * fy + inv[4];
                let v = inv[1] * fx + inv[3] * fy + inv[5];
                let [sr, sg, sb, sa] = bilinear_sample(sdata, sw, sh, u, v);
                let sa = sa * q;
                if sa <= 0.0 {
                    continue;
                }
                let (sr, sg, sb) = (sr * q, sg * q, sb * q);
                let o = ((dy * tw + dx) * 4) as usize;
                let isa = 1.0 - sa / 255.0;
                dst[o] = (sr + dst[o] as f32 * isa).round().clamp(0.0, 255.0) as u8;
                dst[o + 1] = (sg + dst[o + 1] as f32 * isa).round().clamp(0.0, 255.0) as u8;
                dst[o + 2] = (sb + dst[o + 2] as f32 * isa).round().clamp(0.0, 255.0) as u8;
                dst[o + 3] = (sa + dst[o + 3] as f32 * isa).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    /// Fill a [`Quad`] (axis-aligned rect + color), honoring clip + opacity.
    fn paint_quad(&mut self, quad: &Quad) {
        let Some(r) = self.dev_rect(quad.rect) else {
            return;
        };
        let Some(paint) = paint_for(quad.color, self.opacity()) else {
            return;
        };
        let mask = Self::active_clip_mask(
            &self.clip_stack,
            self.clip_mask,
            self.mask_region,
            self.mask_stack,
        );
        self.px.fill_rect(r, &paint, Transform::identity(), mask);
    }

    /// Paint an overscroll edge glow: per-scanline interpolation of the
    /// shared [`edge_glow_alpha`] profile across the band — one
    /// `paint_quad` per page-px row, so the gradient stays smooth at
    /// any device scale. Rows inherit the active clip (the emitter
    /// wraps the item in a scrollport `PushClip`/`PopClip`) and the
    /// opacity stack through `paint_quad`, so the glow paints above
    /// scrolled content but never outside the scrollport.
    fn paint_edge_glow(&mut self, edge: Edge4, band: Rect, strength: f32) {
        let strength = strength.clamp(0.0, 1.0);
        if strength <= 0.0 || band.w <= 0.0 || band.h <= 0.0 {
            return;
        }
        let horizontal = matches!(edge, Edge4::Left | Edge4::Right);
        let depth = if horizontal { band.w } else { band.h };
        if depth <= 0.0 {
            return;
        }
        // Clamp: a hand-built item with a huge rect must not hang the
        // CPU backend (the emitter caps depth at 96).
        let steps = depth.ceil().clamp(1.0, 4096.0) as usize;
        let step = depth / steps as f32;
        for i in 0..steps {
            // Row slab at distance `[d0, d1)` from the glowing edge;
            // `t` sampled at the slab center.
            let d0 = i as f32 * step;
            let d1 = d0 + step;
            let t = (d0 + d1) / (2.0 * depth);
            let alpha = edge_glow_alpha(strength, t);
            if alpha <= 0.0 {
                continue;
            }
            let (x, y, w, h) = match edge {
                Edge4::Top => (band.x, band.y + d0, band.w, step),
                Edge4::Bottom => (band.x, band.y + band.h - d1, band.w, step),
                Edge4::Left => (band.x + d0, band.y, step, band.h),
                Edge4::Right => (band.x + band.w - d1, band.y, step, band.h),
            };
            self.paint_quad(&Quad {
                rect: Rect { x, y, w, h },
                color: EDGE_GLOW_COLOR.with_opacity(alpha),
            });
        }
    }

    /// Paint a verbatim text run, shaped with the shared
    /// [`crate::text::shape_text`] (kerning, ligatures, complex
    /// scripts) — the same run the GPU backend lowers, so both
    /// backends' pixels agree. Coverage rasterizes through the text
    /// worker's real [`crate::text::rasterize_glyph`], cached in the CPU
    /// [`GlyphAtlas`].
    ///
    /// Runs paint nothing when no font data resolves for `font_key`
    /// (e.g. the static [`CpuBackend::render_to_rgba`] path).
    fn paint_text_run(
        &mut self,
        text: &str,
        rect: Rect,
        baseline: f32,
        font_key: FontKey,
        size_px: f32,
        color: Color,
    ) {
        let opacity = self.opacity();
        if opacity <= 0.0 || text.is_empty() {
            return;
        }
        let (r, g, b, a) = (color.r8(), color.g8(), color.b8(), u8c(color.a() * opacity));
        if a == 0 {
            return;
        }
        // M5 font-data sharing: resolve once per FontKey and share the
        // bytes across text runs (see [`FontDataCache`]): the installing
        // hook's identity guards against one cache's keys resolving to
        // another cache's bytes.
        //
        // Two-phase lookup for reentrancy: the hook is caller-supplied and
        // runs OUTSIDE the cache borrow. A hook that re-enters text
        // painting on this thread (e.g. measuring fallback text while
        // resolving a font) would otherwise re-borrow the `RefCell` and
        // panic. `insert` re-validates the hook identity in case the
        // re-entrant call installed a different hook.
        thread_local! {
            static FONT_DATA_CACHE: std::cell::RefCell<FontDataCache> =
                std::cell::RefCell::new(FontDataCache::new());
        }
        let hook = self.get_font_data;
        let cached = FONT_DATA_CACHE.with(|cache| cache.borrow_mut().get(hook, font_key));
        let Some(font_data) = cached.or_else(|| {
            let data = hook(font_key)?;
            Some(FONT_DATA_CACHE.with(|cache| cache.borrow_mut().insert(hook, font_key, data)))
        }) else {
            return;
        };
        // Rasterize at device px so glyphs stay crisp under `scale`.
        // Shared rounding convention with the GPU backend
        // ([`rounded_size`]): both backends shape at whole px, so glyph
        // advances agree.
        let raster_size = f32::from(rounded_size(size_px * self.scale));
        let run = shape_text(&font_data, text, raster_size);
        let mut pen_x = self.dev_x(rect.x);
        let pen_y = self.dev_y(baseline);
        for shaped in &run.glyphs {
            // Populate the atlas on miss (counted for eviction); glyphs
            // with no coverage (e.g. space) report a miss and paint
            // nothing, but the pen still advances below.
            let cached = self
                .cached_glyph(font_key, &font_data, shaped.glyph_id, raster_size)
                .is_some();
            // Disjoint field borrows: `glyphs` immutably for the cache
            // lookup, `px` mutably for stamping (avoids cloning the
            // coverage bitmap per glyph per frame).
            let px = &mut self.px;
            if cached {
                let glyph = self
                    .glyphs
                    .get(font_key, shaped.glyph_id, raster_size)
                    .expect("populated by cached_glyph above");
                // Real bearings: the ink stamps at the shaped offset
                // plus the glyph's own (left, top).
                let dx = (pen_x + shaped.x_offset + glyph.left as f32).round() as i32;
                let dy = (pen_y + shaped.y_offset - glyph.top as f32).round() as i32;
                Self::stamp_glyph(px, &self.clip_stack, glyph, dx, dy, [r, g, b, a]);
            }
            pen_x += shaped.advance;
        }
    }

    /// Composite an 8-bit coverage buffer over the current target: each
    /// texel paints `color` (straight RGB) with alpha
    /// `color.a() * opacity * cov/255`, source-over into the
    /// premultiplied target, honoring the live clip mask per pixel.
    /// `(ox, oy)` is the buffer's device-px origin in the target.
    fn composite_coverage(
        &mut self,
        cov: &[u8],
        w: usize,
        _h: usize,
        ox: i32,
        oy: i32,
        color: Color,
    ) {
        let base_a = color.a() * self.opacity();
        if base_a <= 0.0 {
            return;
        }
        let (r8, g8, b8) = (color.r8() as f32, color.g8() as f32, color.b8() as f32);
        // Disjoint from `px`: the borrow checker sees field projections.
        let mask = Self::active_clip_mask(
            &self.clip_stack,
            self.clip_mask,
            self.mask_region,
            self.mask_stack,
        );
        let (mw, mdata) = match mask {
            Some(m) => (m.width() as i32, m.data()),
            None => (0, &[][..]),
        };
        let tw = self.px.width() as i32;
        let th = self.px.height() as i32;
        let dst = self.px.data_mut();
        for (i, &c) in cov.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let dx = ox + (i % w) as i32;
            let dy = oy + (i / w) as i32;
            if dx < 0 || dy < 0 || dx >= tw || dy >= th {
                continue;
            }
            let q = if mask.is_some() {
                mdata[(dy * mw + dx) as usize] as f32 / 255.0
            } else {
                1.0
            };
            if q <= 0.0 {
                continue;
            }
            let sa = base_a * (c as f32 / 255.0) * q;
            if sa <= 0.0 {
                continue;
            }
            let o = ((dy * tw + dx) * 4) as usize;
            let isa = 1.0 - sa;
            dst[o] = (r8 * sa + dst[o] as f32 * isa).round().clamp(0.0, 255.0) as u8;
            dst[o + 1] = (g8 * sa + dst[o + 1] as f32 * isa)
                .round()
                .clamp(0.0, 255.0) as u8;
            dst[o + 2] = (b8 * sa + dst[o + 2] as f32 * isa)
                .round()
                .clamp(0.0, 255.0) as u8;
            dst[o + 3] = (255.0 * sa + dst[o + 3] as f32 * isa)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }

    /// Paint one [`DisplayItem::BoxShadow`].
    ///
    /// Outset: the spread-inflated rounded rect, shifted by `offset`,
    /// is rasterized to a coverage buffer, blurred with a multi-pass
    /// box blur approximating gaussian σ = `blur`/2, and composited in
    /// `color` under the live clip. The sharp border box is knocked out
    /// of the blurred coverage (`cov = cov·(1 − sharp)`): outer shadows
    /// are drawn outside the border edge only (CSS Backgrounds 3 §7.2),
    /// so a transparent background yields a ring, not a filled blob.
    ///
    /// Inset: the spread-*deflated* shape (spread expands the shadow
    /// inward for inset) is blurred the same way; the visible shadow is
    /// the blurred shape knocked out by the sharp border box
    /// (`S − B`, clamped to ≥ 0), clipped to the border box interior —
    /// the darkening hugs the inner edges and fades toward the center.
    /// A deflated-empty shape blurs to nothing, so the whole interior
    /// goes dark (matching huge inset spreads). A shadow whose blurred
    /// shape cannot reach the border box interior is skipped.
    ///
    /// Absurd blurs are clamped so the temp buffer stays under
    /// [`MAX_SHADOW_TEMP_DIM`] per side (documented degradation; past
    /// ~1300px the blob is indistinguishable anyway).
    #[allow(clippy::too_many_arguments)]
    fn paint_box_shadow(
        &mut self,
        rect: Rect,
        radii: [f32; 4],
        color: Color,
        offset: (f32, f32),
        blur: f32,
        spread: f32,
        inset: bool,
    ) {
        if self.opacity() <= 0.0 || color.a() <= 0.0 {
            return;
        }
        let s = self.scale;
        if !s.is_finite() || s <= 0.0 {
            return;
        }
        let blur = blur.max(0.0);
        // Device-px border box.
        let bx0 = self.dev_x(rect.x);
        let by0 = self.dev_y(rect.y);
        let bx1 = self.dev_x(rect.x + rect.w);
        let by1 = self.dev_y(rect.y + rect.h);
        if !(bx1 > bx0 && by1 > by0) {
            return;
        }
        // Shadow shape in page px: inflated by spread (outset) or
        // deflated (inset); radii follow the inflation, clamped ≥ 0.
        let inflate = if inset { -spread } else { spread };
        let shape = Rect {
            x: rect.x - inflate,
            y: rect.y - inflate,
            w: rect.w + 2.0 * inflate,
            h: rect.h + 2.0 * inflate,
        };
        let radii_s = radii.map(|r| (r + inflate).max(0.0));
        let empty_shape = shape.w <= 0.0 || shape.h <= 0.0;
        // Device-px shape bbox at its offset position.
        let (ox, oy) = (offset.0 * s, offset.1 * s);
        let sx0 = self.dev_x(shape.x) + ox;
        let sy0 = self.dev_y(shape.y) + oy;
        let sx1 = self.dev_x(shape.x + shape.w) + ox;
        let sy1 = self.dev_y(shape.y + shape.h) + oy;
        // Blur radius in device px (σ = blur/2); the temp margin covers
        // the 3-pass support (3r) plus a pixel of raster slack.
        let mut sigma = blur * s / 2.0;
        let mut r = box_blur_radius(sigma);
        let mut margin = (3 * r + 1) as f32;
        // Temp bounds: the border box (sharp knockout) ∪ shape ± margin.
        // The outset knockout needs the box rasterized on the same
        // domain as the blurred shape.
        let (mut tx0, mut ty0, mut tx1, mut ty1) = (
            bx0.min(sx0 - margin).floor(),
            by0.min(sy0 - margin).floor(),
            bx1.max(sx1 + margin).ceil(),
            by1.max(sy1 + margin).ceil(),
        );
        let mut tw = (tx1 - tx0).max(0.0) as usize;
        let mut th = (ty1 - ty0).max(0.0) as usize;
        if tw == 0 || th == 0 {
            return;
        }
        if tw > MAX_SHADOW_TEMP_DIM || th > MAX_SHADOW_TEMP_DIM {
            // Absurd blur: shrink the radius so the temp fits. `r <=
            // sigma` always, so capping sigma at the radius that fits
            // keeps the margin (3r+1) in bounds; past ~1300px the
            // difference is invisible.
            let shape_w = (sx1 - sx0).max(bx1 - bx0).max(1.0);
            let max_r = ((MAX_SHADOW_TEMP_DIM as f32 - shape_w - 2.0) / 6.0).max(0.0);
            sigma = sigma.min(max_r);
            r = box_blur_radius(sigma);
            margin = (3 * r + 1) as f32;
            tx0 = bx0.min(sx0 - margin).floor();
            ty0 = by0.min(sy0 - margin).floor();
            tx1 = bx1.max(sx1 + margin).ceil();
            ty1 = by1.max(sy1 + margin).ceil();
            tw = (tx1 - tx0).max(0.0).min(MAX_SHADOW_TEMP_DIM as f32) as usize;
            th = (ty1 - ty0).max(0.0).min(MAX_SHADOW_TEMP_DIM as f32) as usize;
            if tw == 0 || th == 0 {
                return;
            }
        }
        // Temps from the backend's shadow scratch pool (returned after
        // the composite below): `cov` is checked out zeroed because the
        // empty-shape path reads it without rasterizing; `tmp` and
        // `sharp` are fully overwritten before any read.
        let mut cov = self.checkout_shadow_scratch_zeroed(tw * th);
        let mut tmp = self.checkout_shadow_scratch(tw * th);
        let radii_dev = radii_s.map(|v| v * s);
        if !empty_shape {
            rasterize_rounded_rect(
                &mut cov,
                tw,
                th,
                sx0 - tx0,
                sy0 - ty0,
                sx1 - tx0,
                sy1 - ty0,
                radii_dev,
            );
            box_blur_sigma(&mut cov, &mut tmp, tw, th, sigma);
        }
        // Knock out the sharp border box. Outset: `cov = cov·(1 − sharp)`
        // — outer shadows are drawn outside the border edge only (CSS
        // Backgrounds 3 §7.2). Inset: `cov = clamp(sharp − cov)` — a
        // detached inset shadow (blurred shape cannot reach the box)
        // leaves cov all zero and paints nothing.
        let mut sharp = self.checkout_shadow_scratch(tw * th);
        let radii_box = radii.map(|v| (v * s).max(0.0));
        rasterize_rounded_rect(
            &mut sharp,
            tw,
            th,
            bx0 - tx0,
            by0 - ty0,
            bx1 - tx0,
            by1 - ty0,
            radii_box,
        );
        if inset {
            for (c, &k) in cov.iter_mut().zip(sharp.iter()) {
                *c = k.saturating_sub(*c);
            }
        } else {
            for (c, &k) in cov.iter_mut().zip(sharp.iter()) {
                *c = ((*c as u16 * (255 - k as u16)) / 255) as u8;
            }
        }
        self.composite_coverage(&cov, tw, th, tx0 as i32, ty0 as i32, color);
        // Return the temps so the next shadow reuses them instead of
        // reallocating (pixel-identical: every checkout is fully
        // overwritten before any read, `cov` excepted — checked out
        // zeroed above).
        self.return_shadow_scratch(cov);
        self.return_shadow_scratch(tmp);
        self.return_shadow_scratch(sharp);
    }

    /// Paint one [`DisplayItem::TextShadow`]: the run reshaped in
    /// `color` at `offset`, blurred when `blur > 0`, under the live
    /// clip. A zero blur reuses the text path directly; a positive blur
    /// renders the run into a temp pixmap (unclipped — the clip applies
    /// once, at composite, so blurred ink past the clip edge is cut
    /// correctly), blurs it, and composites with the clip mask.
    #[allow(clippy::too_many_arguments)]
    fn paint_text_shadow(
        &mut self,
        text: &str,
        rect: Rect,
        baseline: f32,
        font_key: FontKey,
        size_px: f32,
        color: Color,
        offset: (f32, f32),
        blur: f32,
    ) {
        let shadow_rect = Rect {
            x: rect.x + offset.0,
            y: rect.y + offset.1,
            w: rect.w,
            h: rect.h,
        };
        let shadow_baseline = baseline + offset.1;
        if blur <= 0.0 {
            self.paint_text_run(text, shadow_rect, shadow_baseline, font_key, size_px, color);
            return;
        }
        if self.opacity() <= 0.0 || color.a() <= 0.0 || text.is_empty() {
            return;
        }
        let s = self.scale;
        if !s.is_finite() || s <= 0.0 {
            return;
        }
        // Temp bounds: text bounds (like `item_paint_bounds`) at the
        // offset position, plus blur margin.
        let m = size_px.max(0.0) + blur.max(0.0);
        let bx0 = self.dev_x(shadow_rect.x - m).floor();
        let by0 = self.dev_y(shadow_rect.y - m).floor();
        let bx1 = self.dev_x(shadow_rect.x + shadow_rect.w + m).ceil();
        let by1 = self.dev_y(shadow_rect.y + shadow_rect.h + m).ceil();
        let (tw, th) = (
            (bx1 - bx0).max(0.0).min(MAX_SHADOW_TEMP_DIM as f32) as u32,
            (by1 - by0).max(0.0).min(MAX_SHADOW_TEMP_DIM as f32) as u32,
        );
        if tw == 0 || th == 0 {
            return;
        }
        let Some(mut tmp_px) = self.checkout_group_pixmap(tw, th) else {
            return;
        };
        // Unclipped paint into the temp (the clip applies once, at
        // composite); the opacity stack stays live so the shadow fades
        // with its group.
        let saved_clips = std::mem::take(&mut self.clip_stack);
        std::mem::swap(&mut *self.px, &mut tmp_px);
        let old_origin = self.dev_origin;
        self.dev_origin = (old_origin.0 + bx0, old_origin.1 + by0);
        self.rebuild_clip_mask();
        self.paint_text_run(text, shadow_rect, shadow_baseline, font_key, size_px, color);
        std::mem::swap(&mut *self.px, &mut tmp_px);
        self.dev_origin = old_origin;
        self.clip_stack = saved_clips;
        self.rebuild_clip_mask();
        // Blur the premultiplied layer, then composite with the clip mask.
        // The RGBA blur scratch comes from the backend's shadow scratch
        // pool (`box_blur_sigma_rgba` overwrites it fully before any
        // read, so no zeroing is needed).
        {
            let (w, h) = (tw as usize, th as usize);
            let mut tmp = self.checkout_shadow_scratch(w * h * 4);
            box_blur_sigma_rgba(tmp_px.data_mut(), &mut tmp, w, h, blur * s / 2.0);
            self.return_shadow_scratch(tmp);
        }
        let twi = tw as i32;
        let thi = th as i32;
        let twidth = self.px.width() as i32;
        let theight = self.px.height() as i32;
        let mask = Self::active_clip_mask(
            &self.clip_stack,
            self.clip_mask,
            self.mask_region,
            self.mask_stack,
        );
        let (mw, mdata) = match mask {
            Some(m) => (m.width() as i32, m.data()),
            None => (0, &[][..]),
        };
        let src = tmp_px.data();
        let dst = self.px.data_mut();
        let (bx0i, by0i) = (bx0 as i32, by0 as i32);
        for y in 0..thi {
            let ty = by0i + y;
            if ty < 0 || ty >= theight {
                continue;
            }
            for x in 0..twi {
                let tx = bx0i + x;
                if tx < 0 || tx >= twidth {
                    continue;
                }
                let so = ((y * twi + x) * 4) as usize;
                let sa = src[so + 3] as f32 / 255.0;
                if sa <= 0.0 {
                    continue;
                }
                let q = if mask.is_some() {
                    mdata[(ty * mw + tx) as usize] as f32 / 255.0
                } else {
                    1.0
                };
                if q <= 0.0 {
                    continue;
                }
                // Premultiplied source: scale by the clip coverage.
                let o = ((ty * twidth + tx) * 4) as usize;
                let isa = 1.0 - sa * q;
                dst[o] = (src[so] as f32 * q + dst[o] as f32 * isa)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                dst[o + 1] = (src[so + 1] as f32 * q + dst[o + 1] as f32 * isa)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                dst[o + 2] = (src[so + 2] as f32 * q + dst[o + 2] as f32 * isa)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                dst[o + 3] = (src[so + 3] as f32 * q + dst[o + 3] as f32 * isa)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
        self.return_group_pixmap(tmp_px);
    }

    /// Numeral text for a list marker: for `Decimal`/`DecimalLeadingZero`
    /// the 1-based item `index`, formatted per `kind` into a stack
    /// buffer (no `format!`); glyph markers have no numeral. Same
    /// formula as the GPU backend.
    fn marker_numeral(kind: MarkerKind, index: u32) -> Option<String> {
        let mut buf = [0u8; 20];
        match kind {
            MarkerKind::Decimal(_) => {
                Some(format_marker_numeral(&mut buf, index, false).to_owned())
            }
            MarkerKind::DecimalLeadingZero(_) => {
                Some(format_marker_numeral(&mut buf, index, true).to_owned())
            }
            _ => marker_text(kind),
        }
    }

    /// Stamp one rasterized glyph's coverage onto the pixmap in straight
    /// RGBA `(r, g, b, a)`, source-over, honoring the clip.
    ///
    /// Per-pixel blending (instead of a mask + `draw_pixmap`) because the
    /// clip here is a rect intersection, applied as a bounds check.
    /// Takes `px`/`clip` instead of `&mut self` so the caller can hold
    /// the glyph-cache borrow across the call.
    fn stamp_glyph(
        px: &mut Pixmap,
        clip_stack: &[ClipEntry],
        glyph: &RasterGlyph,
        dx: i32,
        dy: i32,
        rgba: [u8; 4],
    ) {
        let [r, g, b, a] = rgba;
        let (w, h) = (px.width() as i32, px.height() as i32);
        // The integer bound is an outward-snap cull only: the exact
        // edge comes from the per-pixel coverage below, using the same
        // [`clip_stack_coverage_byte`](crate::clip::clip_stack_coverage_byte)
        // formula the clip mask fill and the GPU clip shader use.
        if clip_stack.iter().any(|e| e.rect.is_none()) {
            return;
        }
        let (cx0, cy0, cx1, cy1) = match stack_bound(clip_stack) {
            None if clip_stack.is_empty() => (0, 0, w, h),
            None => return,
            Some(cr) => (
                cr.left().floor().max(0.0) as i32,
                cr.top().floor().max(0.0) as i32,
                cr.right().ceil().min(w as f32) as i32,
                cr.bottom().ceil().min(h as f32) as i32,
            ),
        };
        // Straight -> premultiplied source color.
        let (pr, pg, pb) = (
            (u32::from(r) * u32::from(a) + 127) / 255,
            (u32::from(g) * u32::from(a) + 127) / 255,
            (u32::from(b) * u32::from(a) + 127) / 255,
        );
        let pixels = px.pixels_mut();
        let stride = w as usize;
        for gy in 0..glyph.height {
            let y = dy + gy as i32;
            if y < cy0 || y >= cy1 {
                continue;
            }
            for gx in 0..glyph.width {
                let x = dx + gx as i32;
                if x < cx0 || x >= cx1 {
                    continue;
                }
                let mut cov = u32::from(glyph.coverage[(gy * glyph.width + gx) as usize]);
                if cov == 0 {
                    continue;
                }
                // Fractional clip: fold the pixel's coverage under the
                // clip stack's geometric intersection into the glyph
                // coverage — the same byte the clip mask fill computes
                // for this pixel (see `crate::clip`).
                if !clip_stack.is_empty() {
                    let m = u32::from(clip_stack_coverage_byte(clip_stack, x as usize, y as usize));
                    cov = (cov * m + 127) / 255;
                    if cov == 0 {
                        continue;
                    }
                }
                // Source-over in premultiplied space.
                let sa = (u32::from(a) * cov + 127) / 255;
                let sr = (pr * cov + 127) / 255;
                let sg = (pg * cov + 127) / 255;
                let sb = (pb * cov + 127) / 255;
                let dst = &mut pixels[(y as usize) * stride + x as usize];
                let (dr, dg, db, da) = (
                    u32::from(dst.red()),
                    u32::from(dst.green()),
                    u32::from(dst.blue()),
                    u32::from(dst.alpha()),
                );
                let inv = 255 - sa;
                let out_a = sa + (da * inv + 127) / 255;
                let out_r = sr + (dr * inv + 127) / 255;
                let out_g = sg + (dg * inv + 127) / 255;
                let out_b = sb + (db * inv + 127) / 255;
                // Unreachable-None: out_rgb <= sa + da*inv/255 = out_a
                // because sr <= sa and dr <= da component-wise.
                if let Some(blended) = tiny_skia::PremultipliedColorU8::from_rgba(
                    out_r as u8,
                    out_g as u8,
                    out_b as u8,
                    out_a as u8,
                ) {
                    *dst = blended;
                }
            }
        }
    }

    /// Get (decoding + caching) the premultiplied pixmap for an image id.
    ///
    /// Three layers, cheapest first (see [`ImageCache`]):
    /// 1. The per-id front entry, validated by [`DecodedImage::content_hash`]
    ///    *plus* the stored dimensions and byte length (the hash alone is
    ///    not trusted as content identity): on mismatch the stale entry
    ///    is replaced, so an [`ImageId`] reused with different bytes (ids
    ///    are only unique within one [`DisplayList`]) can never serve a
    ///    stale pixmap.
    /// 2. The content-hash-keyed shared cache: a different id (or a
    ///    stale front entry) with identical bytes reuses the existing
    ///    decode — identical images decode once, not once per id.
    /// 3. Only a miss in both decodes, inserts into the shared cache
    ///    (FIFO-evicted past its bounds), and refreshes the front entry.
    ///
    /// The hash itself is O(1): [`DisplayList::image_content_hash`] caches
    /// it at registration, so no pixel-buffer rehash per frame.
    fn image_pixmap(&mut self, list: &DisplayList, image: ImageId) -> Option<&Pixmap> {
        let decoded = list.image(image)?;
        // Checked: the `width * height * 4` product overflows even 64-bit
        // usize above ~4 gigapixels; a wrapped product would accept a
        // mismatched buffer (release) or panic (debug) instead of
        // skipping the image.
        let expected = (decoded.width as usize)
            .checked_mul(decoded.height as usize)
            .and_then(|pixels| pixels.checked_mul(4));
        if decoded.width == 0 || decoded.height == 0 || expected != Some(decoded.rgba.len()) {
            return None;
        }
        // Registration-cached content hash (see
        // [`DisplayList::image_content_hash`]): O(1) per reference, no
        // per-frame pixel-buffer rehash. The hash (not the buffer
        // address) is the content identity, so display lists rebuilt at
        // reused addresses can never serve a stale pixmap. The on-demand
        // fallback only covers a hypothetical map divergence (`add_image`
        // populates both maps together, so for a registered id it never
        // triggers); it keeps this total.
        let hash = list
            .image_content_hash(image)
            .unwrap_or_else(|| decoded.content_hash());
        // 1. Per-id front cache: same id, same content. The hash alone
        // is not trusted as content identity (see `CpuImageEntry`): the
        // stored dimensions and byte length must also match the current
        // source, else this is a miss. The probe borrow ends before the
        // miss path below takes mutable borrows, so this re-gets the
        // entry for the return instead of holding the probe across them.
        if self
            .image_cache
            .by_id
            .get(&image)
            .is_some_and(|e| e.matches(hash, decoded.width, decoded.height, decoded.rgba.len()))
        {
            self.cache_stats.image_id_hits += 1;
            let entry = self.image_cache.by_id.get(&image)?;
            return Some(&entry.pixmap);
        }
        // 2. Content-hash-keyed shared cache: identical bytes under a
        // different (or stale) id share one decode. The `Arc` means the
        // front entry and the shared map name the same pixels without
        // copying them. The hash alone is not trusted: the stored
        // dimensions must match the current source (an FNV-1a collision
        // with a different shape is a miss, decoded fresh), mirroring
        // the per-id front cache's verification.
        let pixmap = match self.image_cache.shared.get(&hash) {
            Some(shared) if shared.width == decoded.width && shared.height == decoded.height => {
                self.cache_stats.image_shared_hits += 1;
                Arc::clone(&shared.pixmap)
            }
            _ => {
                // 3. Real decode: straight RGBA -> premultiplied pixmap.
                let size = IntSize::from_wh(decoded.width, decoded.height)?;
                let pixmap = Arc::new(Pixmap::from_vec(premultiply_rgba(&decoded.rgba), size)?);
                self.cache_stats.image_decodes += 1;
                let cache = &mut self.image_cache;
                // A shape-mismatched entry under this hash (a genuine
                // collision) is replaced outright: its bytes stop
                // counting and its queue slot is recycled, keeping
                // `shared_order` exact (one entry per live hash).
                if let Some(old) = cache.shared.insert(
                    hash,
                    SharedDecode {
                        pixmap: Arc::clone(&pixmap),
                        width: decoded.width,
                        height: decoded.height,
                    },
                ) {
                    cache.sub_shared_bytes(decoded_bytes(&old.pixmap));
                    cache.shared_order.retain(|&h| h != hash);
                }
                cache.shared_bytes += decoded.rgba.len();
                cache.shared_order.push_back(hash);
                // Debug-only: the hand-maintained counter must exactly
                // match the map contents after insert/replace.
                debug_assert_eq!(
                    cache.shared_bytes,
                    cache
                        .shared
                        .values()
                        .map(|e| decoded_bytes(&e.pixmap))
                        .sum::<usize>(),
                    "shared_bytes desynced from shared contents after insert"
                );
                pixmap
            }
        };
        // Refresh the per-id front entry (stale ids are overwritten),
        // accounting its bytes toward the combined decode-memory budget:
        // front entries hold their own `Arc` and survive `shared`
        // eviction, so they are budgeted too (see
        // [`MAX_DECODED_IMAGE_BYTES`]).
        let cache = &mut self.image_cache;
        let fresh = !cache.by_id.contains_key(&image);
        let byte_len = decoded.rgba.len();
        if let Some(old) = cache.by_id.insert(
            image,
            CpuImageEntry {
                pixmap: Arc::clone(&pixmap),
                hash,
                width: decoded.width,
                height: decoded.height,
                byte_len,
            },
        ) {
            cache.sub_by_id_bytes(old.byte_len);
        }
        cache.by_id_bytes += byte_len;
        // Replaces refresh recency (move to back), so eviction order
        // stays FIFO and the protected id is never popped early.
        cache.refresh_by_id_order(image, fresh);
        // One eviction point for both maps (see `evict_over_budget`):
        // the just-inserted entry is protected, so the re-get below
        // cannot miss.
        cache.evict_over_budget(image);
        let entry = self.image_cache.by_id.get(&image)?;
        Some(&entry.pixmap)
    }

    /// Draw an image stretched into a CSS-px rect.
    fn draw_image(&mut self, list: &DisplayList, image: ImageId, rect: Rect) {
        let opacity = self.opacity();
        if opacity <= 0.0 {
            return;
        }
        let Some(r) = self.to_device_rect(rect) else {
            return;
        };
        // Early-out only when fully clipped: the image transform is
        // built from the *unclipped* device rect below, and the clip mask
        // cuts it down, so a partial clip must not shrink `r` (that
        // would squish the whole image into the visible intersection
        // instead of showing the matching sub-region). A non-degenerate
        // rect (checked above) failing dev_rect is fully clipped out.
        if self.dev_rect(rect).is_none() {
            return;
        };
        let (iw, ih) = match self.image_pixmap(list, image) {
            Some(pm) => (pm.width(), pm.height()),
            None => return,
        };
        if iw == 0 || ih == 0 {
            return;
        }
        // Re-borrow the cached pixmap (`image_cache` and `px` are disjoint
        // fields). draw_pixmap applies `transform` to the destination
        // rect; (x, y) = (0, 0) here and the transform maps image px ->
        // device px.
        let Some(entry) = self.image_cache.by_id.get(&image) else {
            return;
        };
        let pm: &Pixmap = &entry.pixmap;
        let sx = r.width() / iw as f32;
        let sy = r.height() / ih as f32;
        let paint = PixmapPaint {
            opacity,
            quality: tiny_skia::FilterQuality::Bilinear,
            ..PixmapPaint::default()
        };
        let transform = Transform::from_row(sx, 0.0, 0.0, sy, r.x(), r.y());
        self.px.draw_pixmap(
            0,
            0,
            pm.as_ref(),
            &paint,
            transform,
            Self::active_clip_mask(
                &self.clip_stack,
                self.clip_mask,
                self.mask_region,
                self.mask_stack,
            ),
        );
    }

    /// Page-px rect currently visible on the paint target, or `None`
    /// when the scale is degenerate. Used to bound [`Self::paint_repeat_image`]'s
    /// tile loop to tiles that can actually paint.
    fn visible_page_rect(&self) -> Option<Rect> {
        if self.scale <= 0.0 || !self.scale.is_finite() {
            return None;
        }
        // Inverse of `dev_x`/`dev_y`: device (0,0)..(w,h) maps to this
        // page rect (`dev_origin` shifts the target inside an isolated
        // opacity group; see the field docs).
        Some(Rect {
            x: self.scroll.0 + self.dev_origin.0 / self.scale,
            y: self.scroll.1 + self.dev_origin.1 / self.scale,
            w: self.px.width() as f32 / self.scale,
            h: self.px.height() as f32 / self.scale,
        })
    }

    /// Paint a [`DisplayItem::RepeatImage`]: tile the image across `area`
    /// with the tile grid anchored at the positioned origin (`area`'s
    /// top-left plus `offset`).
    ///
    /// The area is clamped to [`MAX_REPEAT_AREA_DIM`] per axis first
    /// (the hostile-input bound, shared with the GPU backend via
    /// [`clamp_repeat_area`]), then intersected with the visible page
    /// rect so tiles that cannot be seen are never drawn — the grid
    /// stays anchored at the original origin, so the phase matches a
    /// full-area expansion exactly. Each tile goes through
    /// [`Self::draw_image`], the same path the old per-tile
    /// `BackgroundImage` items used, so the output is pixel-identical
    /// to the old expansion. The area clip is pushed for the tile draws
    /// (like the old expansion's `PushClip`/`PopClip` pair), trimming
    /// partial edge tiles.
    ///
    /// The tile is enlarged via [`cap_repeat_tile`] (the builder's
    /// [`MAX_REPEAT_TILES`] cap, shared here) before either the space
    /// or the repeat branch runs: a hand-built item can carry a hostile
    /// tile (e.g. `1e-30` px over an 8192 px area) that the builder
    /// would have capped, and without the cap the loops below iterate
    /// ~`i64::MAX` times.
    fn paint_repeat_image(
        &mut self,
        list: &DisplayList,
        area: Rect,
        image: ImageId,
        tile: (f32, f32),
        repeat: BackgroundRepeat,
        offset: (f32, f32),
    ) {
        if self.opacity() <= 0.0 {
            return;
        }
        let (tw, th) = tile;
        if tw <= 0.0
            || th <= 0.0
            || !tw.is_finite()
            || !th.is_finite()
            || !offset.0.is_finite()
            || !offset.1.is_finite()
        {
            return;
        }
        let area = clamp_repeat_area(area);
        if area.w <= 0.0 || area.h <= 0.0 {
            return;
        }
        // Cap the tile count before either branch below (see the fn
        // docs): without this a hand-built hostile tile hangs the
        // space/repeat loops.
        let (tw, th) = cap_repeat_tile((tw, th), area);
        // Bound the tile loop to what can actually paint. Tiles outside
        // the visible rect are fully clipped by `draw_image` anyway;
        // skipping them up front keeps a huge area over a small viewport
        // from expanding millions of no-op tiles.
        let cover = match self.visible_page_rect() {
            Some(v) => intersect_rect(area, v),
            None => area,
        };
        if cover.w <= 0.0 || cover.h <= 0.0 {
            return;
        }
        self.clip_stack
            .push(ClipEntry::rect(self.to_device_rect(area)));
        self.update_clip_mask();
        if repeat == BackgroundRepeat::Space {
            let (nx, sx0, px) = background_space_axis(area.x, area.w, tw, offset.0);
            let (ny, sy0, py) = background_space_axis(area.y, area.h, th, offset.1);
            // Only tiles overlapping the visible cover; tiles are whole
            // and unclipped in `space` mode.
            let i0 = ((cover.x - sx0) / px).floor().max(0.0) as i64;
            let i1 = (((cover.x + cover.w - sx0) / px).floor() + 1.0).min(nx as f32) as i64;
            let j0 = ((cover.y - sy0) / py).floor().max(0.0) as i64;
            let j1 = (((cover.y + cover.h - sy0) / py).floor() + 1.0).min(ny as f32) as i64;
            for j in j0.max(0)..j1.max(0) {
                for i in i0.max(0)..i1.max(0) {
                    self.draw_image(
                        list,
                        image,
                        Rect {
                            x: sx0 + i as f32 * px,
                            y: sy0 + j as f32 * py,
                            w: tw,
                            h: th,
                        },
                    );
                }
            }
        } else {
            let (repeat_x, repeat_y) = match repeat {
                BackgroundRepeat::Repeat | BackgroundRepeat::Round => (true, true),
                BackgroundRepeat::RepeatX => (true, false),
                BackgroundRepeat::RepeatY => (false, true),
                BackgroundRepeat::NoRepeat => (false, false),
                // Handled by the branch above.
                BackgroundRepeat::Space => (false, false),
            };
            let (x0, y0) = (area.x + offset.0, area.y + offset.1);
            let (i_start, j_start, nx, ny) =
                background_tile_grid(cover, (x0, y0), (tw, th), repeat_x, repeat_y);
            // The tile cap above already bounds the grid, but clamp
            // defensively anyway: `j_start + ny` must not overflow i64
            // (a debug-mode panic) on hugely negative origins, and the
            // loops below must stay bounded even if a future path
            // bypasses the cap.
            let nx = nx.clamp(0, MAX_REPEAT_TILES as i64);
            let ny = ny.clamp(0, MAX_REPEAT_TILES as i64);
            for j in j_start..j_start.saturating_add(ny) {
                for i in i_start..i_start.saturating_add(nx) {
                    self.draw_image(
                        list,
                        image,
                        Rect {
                            x: x0 + i as f32 * tw,
                            y: y0 + j as f32 * th,
                            w: tw,
                            h: th,
                        },
                    );
                }
            }
        }
        self.clip_stack.pop();
        self.update_clip_mask();
    }

    /// Paint a CSS gradient over `rect` (page px): bake the resolved
    /// gradient at device resolution with the shared
    /// [`bake_gradient_rgba8`] rasterizer (the same code the GPU
    /// backend bakes textures with), then composite it like an image:
    /// group opacity through `PixmapPaint`, clips through the active
    /// clip mask.
    fn paint_gradient(&mut self, rect: Rect, gradient: &Gradient) {
        let opacity = self.opacity();
        if opacity <= 0.0 {
            return;
        }
        let Some(r) = self.to_device_rect(rect) else {
            return;
        };
        // Same fully-clipped early-out as `draw_image`: the bake
        // covers the unclipped device rect and the clip mask cuts it
        // down, so a partial clip must not shrink the bake (that would
        // squish the whole gradient into the visible intersection).
        if self.dev_rect(rect).is_none() {
            return;
        }
        // Cap the bake (see `MAX_GRADIENT_BAKE_DIM`): the pixmap
        // transform stretches the capped bake back over the device
        // rect, which is lossless for smooth gradient data.
        let bw = (r.width() as u32).clamp(1, MAX_GRADIENT_BAKE_DIM);
        let bh = (r.height() as u32).clamp(1, MAX_GRADIENT_BAKE_DIM);
        // Bake cache, keyed by (gradient content, bake size): a
        // full-screen gradient otherwise re-bakes every frame. Hits
        // stamp the entry with this frame so the frame-boundary
        // eviction keeps it.
        let frame = self.gradient_frame;
        let key = gradient_bake_key(gradient, bw, bh);
        let entry = match self.gradient_bake_cache.entry(key) {
            HashMapEntry::Occupied(occupied) => {
                self.cache_stats.gradient_bake_hits += 1;
                occupied.into_mut()
            }
            HashMapEntry::Vacant(vacant) => {
                self.cache_stats.gradient_bakes += 1;
                vacant.insert(CpuGradientBake {
                    bytes: bake_gradient_rgba8(gradient, &rect, bw, bh),
                    width: bw,
                    height: bh,
                    last_frame: frame,
                })
            }
        };
        entry.last_frame = frame;
        let Some(size) = IntSize::from_wh(entry.width, entry.height) else {
            return;
        };
        let Some(pm) = Pixmap::from_vec(premultiply_rgba(&entry.bytes), size) else {
            return;
        };
        let paint = PixmapPaint {
            opacity,
            quality: tiny_skia::FilterQuality::Bilinear,
            ..PixmapPaint::default()
        };
        let transform = Transform::from_row(
            r.width() / bw as f32,
            0.0,
            0.0,
            r.height() / bh as f32,
            r.x(),
            r.y(),
        );
        self.px.draw_pixmap(
            0,
            0,
            pm.as_ref(),
            &paint,
            transform,
            Self::active_clip_mask(
                &self.clip_stack,
                self.clip_mask,
                self.mask_region,
                self.mask_stack,
            ),
        );
    }
}

/// Convert straight RGBA8 to premultiplied RGBA8 (what tiny-skia stores).
fn premultiply_rgba(data: &[u8]) -> Vec<u8> {
    let (chunks, _remainder) = data.as_chunks::<4>();
    let mut out = Vec::with_capacity(chunks.len() * 4);
    for px in chunks {
        let a = u32::from(px[3]);
        out.push(((u32::from(px[0]) * a + 127) / 255) as u8);
        out.push(((u32::from(px[1]) * a + 127) / 255) as u8);
        out.push(((u32::from(px[2]) * a + 127) / 255) as u8);
        out.push(px[3]);
    }
    out
}

/// One channel of a blend mode's blend function B(Cb, Cs), applied to
/// unpremultiplied colors (CSS Compositing and Blending L1). The GPU
/// backend's `fs_composite_blend` implements the same function; keep
/// the two in sync.
fn blend_channel(mode: BlendMode, cb: f32, cs: f32) -> f32 {
    match mode {
        BlendMode::Normal => cs,
        BlendMode::Multiply => cb * cs,
        BlendMode::Screen => cb + cs - cb * cs,
        BlendMode::Overlay => {
            if cb <= 0.5 {
                2.0 * cb * cs
            } else {
                1.0 - 2.0 * (1.0 - cb) * (1.0 - cs)
            }
        }
        BlendMode::Darken => cb.min(cs),
        BlendMode::Lighten => cb.max(cs),
    }
}

/// Blend one premultiplied-RGBA8 source pixel over one
/// premultiplied-RGBA8 backdrop pixel with `mode`, returning the
/// premultiplied-RGBA8 result. Implements CSS Compositing and Blending
/// L1 in f32:
///
/// ```text
/// Cs' = (1 − αb)·Cs + αb·B(Cb, Cs)
/// Co  = αs·Cs' + αb·(1 − αs)·Cb
/// αo  = αs + αb·(1 − αs)
/// ```
///
/// with unpremultiplied source (Cs, αs) and backdrop (Cb, αb).
/// `src[3]` must be nonzero (the caller skips transparent source
/// pixels, for which the formula reduces to the backdrop anyway).
/// The GPU backend's `fs_composite_blend` runs the same formula with
/// the same unpremultiply (channel/alpha in f32), so the backends
/// agree within rounding.
fn blend_pixel(src: [u8; 4], dst: [u8; 4], mode: BlendMode) -> [u8; 4] {
    let sa = f32::from(src[3]) / 255.0;
    let da = f32::from(dst[3]) / 255.0;
    // Unpremultiply: channel/alpha in f32, channels normalized to
    // 0..1 first. `sa > 0` by contract.
    let inv_sa = 1.0 / sa;
    let cs = [
        f32::from(src[0]) / 255.0 * inv_sa,
        f32::from(src[1]) / 255.0 * inv_sa,
        f32::from(src[2]) / 255.0 * inv_sa,
    ];
    let cb = if dst[3] > 0 {
        let inv_da = 1.0 / da;
        [
            f32::from(dst[0]) / 255.0 * inv_da,
            f32::from(dst[1]) / 255.0 * inv_da,
            f32::from(dst[2]) / 255.0 * inv_da,
        ]
    } else {
        [0.0; 3]
    };
    let ao = sa + da * (1.0 - sa);
    let mut out = [0u8; 4];
    for c in 0..3 {
        let b = blend_channel(mode, cb[c], cs[c]);
        let csp = (1.0 - da) * cs[c] + da * b;
        // `co` is already the premultiplied output
        // (αs·Cs' + αb·(1−αs)·Cb): no extra `* ao`.
        let co = (sa * csp + da * (1.0 - sa) * cb[c]).clamp(0.0, 1.0);
        // Back to premultiplied u8. `co <= 1` keeps the premultiplied
        // channels within alpha after rounding.
        out[c] = (co * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    out[3] = (ao * 255.0).round().clamp(0.0, 255.0) as u8;
    out
}

/// Convert premultiplied RGBA8 (tiny-skia storage) back to straight RGBA8.
fn unpremultiply(data: Vec<u8>) -> Vec<u8> {
    let mut data = data;
    let (chunks, _remainder) = data.as_chunks_mut::<4>();
    for px in chunks {
        let a = u32::from(px[3]);
        if a == 0 {
            px[0] = 0;
            px[1] = 0;
            px[2] = 0;
        } else if a < 255 {
            px[0] = ((u32::from(px[0]) * 255 + a / 2) / a).min(255) as u8;
            px[1] = ((u32::from(px[1]) * 255 + a / 2) / a).min(255) as u8;
            px[2] = ((u32::from(px[2]) * 255 + a / 2) / a).min(255) as u8;
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display_list::{
        BorderStyle, Color, DecodedImage, DisplayItem, DisplayList, MarkerKind,
    };
    use std::sync::Arc;

    fn rgba_at(rgba: &[u8], stride: u32, x: u32, y: u32) -> [u8; 4] {
        let o = ((y * stride + x) * 4) as usize;
        [rgba[o], rgba[o + 1], rgba[o + 2], rgba[o + 3]]
    }

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect { x, y, w, h }
    }

    fn red() -> Color {
        Color::rgba8(255, 0, 0, 255)
    }

    fn red_rect_list() -> DisplayList {
        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            rect: rect(10.0, 10.0, 80.0, 80.0),
            color: red(),
        });
        list
    }

    #[test]
    fn fill_rect_pixel_colors() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = red_rect_list();
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert_eq!(rgba.len(), 100 * 100 * 4);
        // Inside the rect: solid red, opaque.
        assert_eq!(rgba_at(&rgba, 100, 50, 50), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 10, 10), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 89, 89), [255, 0, 0, 255]);
        // Outside: transparent black.
        assert_eq!(rgba_at(&rgba, 100, 5, 5), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 100, 90, 90), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 100, 99, 99), [0, 0, 0, 0]);
    }

    fn transform_list(matrix: [f32; 6], inner: Vec<DisplayItem>) -> DisplayList {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushTransform { matrix });
        for it in inner {
            list.push(it);
        }
        list.push(DisplayItem::PopTransform);
        list
    }

    fn fill_rect_item(x: f32, y: f32, w: f32, h: f32) -> DisplayItem {
        DisplayItem::FillRect {
            rect: rect(x, y, w, h),
            color: red(),
        }
    }

    fn render_200x200(list: &DisplayList) -> Vec<u8> {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        backend.render_to_rgba_full(list, 200, 200, 1.0, (0.0, 0.0))
    }

    /// `translate(20px, 0)`: the rect paints 20px right, pixel-sharp
    /// (integer translations resample exactly).
    #[test]
    fn transform_translate_moves_fill() {
        let list = transform_list(
            [1.0, 0.0, 0.0, 1.0, 20.0, 0.0],
            vec![fill_rect_item(10.0, 10.0, 80.0, 80.0)],
        );
        let rgba = render_200x200(&list);
        assert_eq!(rgba_at(&rgba, 200, 10, 50), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 200, 30, 50), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 200, 109, 89), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 200, 110, 89), [0, 0, 0, 0]);
    }

    /// `scale(2)` about the page origin: (10,10,20,20) -> (20,20,40,40).
    #[test]
    fn transform_scale_about_origin() {
        let list = transform_list(
            [2.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            vec![fill_rect_item(10.0, 10.0, 20.0, 20.0)],
        );
        let rgba = render_200x200(&list);
        assert_eq!(rgba_at(&rgba, 200, 10, 10), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 200, 30, 30), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 200, 59, 59), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 200, 60, 60), [0, 0, 0, 0]);
    }

    /// `rotate(90deg)` about (40, 40): rect (10,30,60,20) maps to the
    /// bbox x in [30,50], y in [10,70].
    #[test]
    fn transform_rotate_90_about_center() {
        // T(40,40) · R(90°) · T(−40,−40) = [0,1,−1,0,80,0].
        let list = transform_list(
            [0.0, 1.0, -1.0, 0.0, 80.0, 0.0],
            vec![fill_rect_item(10.0, 30.0, 60.0, 20.0)],
        );
        let rgba = render_200x200(&list);
        // Center maps to itself.
        assert_eq!(rgba_at(&rgba, 200, 40, 40), [255, 0, 0, 255]);
        // (60,40): inside the original rect, outside the rotated bbox.
        assert_eq!(rgba_at(&rgba, 200, 60, 40), [0, 0, 0, 0]);
        // (40,60): outside the original rect, inside the rotated bbox.
        assert_eq!(rgba_at(&rgba, 200, 40, 60), [255, 0, 0, 255]);
    }

    /// The identity matrix paints inline, exactly like no transform.
    #[test]
    fn transform_identity_paints_inline() {
        let list = transform_list(
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![fill_rect_item(10.0, 10.0, 80.0, 80.0)],
        );
        let rgba = render_200x200(&list);
        assert_eq!(rgba_at(&rgba, 200, 50, 50), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 200, 5, 5), [0, 0, 0, 0]);
    }

    /// A non-invertible matrix renders nothing (per spec).
    #[test]
    fn transform_noninvertible_paints_nothing() {
        let list = transform_list(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![fill_rect_item(10.0, 10.0, 80.0, 80.0)],
        );
        let rgba = render_200x200(&list);
        assert!(
            rgba.iter().all(|&b| b == 0),
            "non-invertible transform must paint nothing"
        );
    }

    /// The invertibility predicate uses an exact-zero determinant:
    /// tiny-but-valid transforms (e.g. `scale(1e-7)`) count as
    /// invertible, while genuinely singular matrices do not.
    #[test]
    fn invertible_matrix_tiny_but_valid_counts() {
        assert!(invertible_matrix([1e-7, 0.0, 0.0, 1e-7, 0.0, 0.0]));
        assert!(invertible_matrix([1.0, 0.0, 0.0, 1.0, 5.0, 5.0]));
        assert!(!invertible_matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        assert!(!invertible_matrix([1.0, 2.0, 2.0, 4.0, 0.0, 0.0]));
        assert!(!invertible_matrix([1.0, 0.0, 0.0, 1.0, f32::NAN, 0.0]));
        assert!(!invertible_matrix([f32::INFINITY, 0.0, 0.0, 1.0, 0.0, 0.0]));
    }

    /// Nested transforms compose: outer translate(10,0), inner
    /// translate(0,20) moves (10,10,20,20) to (20,30).
    #[test]
    fn transform_nested_compose() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushTransform {
            matrix: [1.0, 0.0, 0.0, 1.0, 10.0, 0.0],
        });
        list.push(DisplayItem::PushTransform {
            matrix: [1.0, 0.0, 0.0, 1.0, 0.0, 20.0],
        });
        list.push(fill_rect_item(10.0, 10.0, 20.0, 20.0));
        list.push(DisplayItem::PopTransform);
        list.push(DisplayItem::PopTransform);
        let rgba = render_200x200(&list);
        assert_eq!(rgba_at(&rgba, 200, 15, 15), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 200, 25, 35), [255, 0, 0, 255]);
    }

    /// Ancestor clips apply to the *transformed* pixels: content that is
    /// outside the clip untransformed but inside it transformed still
    /// paints, and vice versa.
    #[test]
    fn transform_ancestor_clip_applies_post_transform() {
        // Rect at (60,10) is outside the (0,0,50,100) clip; translating
        // it to (10,10) brings it inside: must paint.
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(0.0, 0.0, 50.0, 100.0),
        });
        list.push(DisplayItem::PushTransform {
            matrix: [1.0, 0.0, 0.0, 1.0, -50.0, 0.0],
        });
        list.push(fill_rect_item(60.0, 10.0, 20.0, 20.0));
        list.push(DisplayItem::PopTransform);
        list.push(DisplayItem::PopClip);
        let rgba = render_200x200(&list);
        assert_eq!(
            rgba_at(&rgba, 200, 20, 20),
            [255, 0, 0, 255],
            "translated-into-clip content must paint"
        );
        // Rect at (10,10) is inside the clip; translating it to (70,10)
        // moves it outside: must not paint.
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(0.0, 0.0, 50.0, 100.0),
        });
        list.push(DisplayItem::PushTransform {
            matrix: [1.0, 0.0, 0.0, 1.0, 60.0, 0.0],
        });
        list.push(fill_rect_item(10.0, 10.0, 20.0, 20.0));
        list.push(DisplayItem::PopTransform);
        list.push(DisplayItem::PopClip);
        let rgba = render_200x200(&list);
        assert_eq!(
            rgba_at(&rgba, 200, 20, 20),
            [0, 0, 0, 0],
            "translated-out-of-clip content must not paint"
        );
        assert!(
            rgba.iter().all(|&b| b == 0),
            "nothing may paint outside the clip"
        );
    }

    fn shadow_list(inset: bool, blur: f32, offset: (f32, f32)) -> DisplayList {
        let mut list = DisplayList::new();
        let shadow = DisplayItem::BoxShadow {
            rect: rect(50.0, 50.0, 40.0, 40.0),
            radii: [0.0; 4],
            color: Color::rgba8(0, 0, 0, 255),
            offset,
            blur,
            spread: 0.0,
            inset,
        };
        // The element's background paints over the outset shadow (CSS
        // painting order); the shadow's own knockout already hollows
        // the middle, so the fill only stands in for a real background.
        let fill = DisplayItem::FillRect {
            rect: rect(50.0, 50.0, 40.0, 40.0),
            color: Color::rgba8(255, 255, 255, 255),
        };
        // CSS painting order: outset shadows under the background,
        // inset shadows above it.
        if inset {
            list.push(fill);
            list.push(shadow);
        } else {
            list.push(shadow);
            list.push(fill);
        }
        list
    }

    /// Outset shadow with no blur: sharp black shape behind the box,
    /// offset so it peeks out; covered inside by the background fill.
    #[test]
    fn box_shadow_outset_paints_sharp() {
        let rgba = render_200x200(&shadow_list(false, 0.0, (6.0, 0.0)));
        // Right of the box: shadow (shape at (56,50)-(96,90)).
        assert_eq!(rgba_at(&rgba, 200, 94, 70), [0, 0, 0, 255]);
        // Inside the box: background covers the shadow.
        assert_eq!(rgba_at(&rgba, 200, 70, 70), [255, 255, 255, 255]);
        // Far away: nothing.
        assert_eq!(rgba_at(&rgba, 200, 5, 5), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 200, 40, 70), [0, 0, 0, 0]);
    }

    /// Outset shadow with blur: soft edge fading with distance.
    #[test]
    fn box_shadow_blur_spreads_soft_edge() {
        let rgba = render_200x200(&shadow_list(false, 8.0, (0.0, 0.0)));
        // Just outside the box: strong but partial.
        let [r, g, b, a] = rgba_at(&rgba, 200, 47, 70);
        assert_eq!((r, g, b), (0, 0, 0));
        assert!(a > 0 && a < 255, "soft edge alpha, got {a}");
        // Further out: fades (monotone non-increasing with distance).
        let a_far = rgba_at(&rgba, 200, 40, 70)[3];
        assert!(a_far < a, "blur must fade with distance ({a_far} < {a})");
        // Past the blur: nothing.
        assert_eq!(rgba_at(&rgba, 200, 20, 70), [0, 0, 0, 0]);
    }

    /// Inset shadow: darkens the inner edge, fading toward the center.
    /// (The background under it is opaque, so alpha stays 255 and the
    /// RGB darkens.)
    #[test]
    fn box_shadow_inset_darkens_inner_edge() {
        let rgba = render_200x200(&shadow_list(true, 8.0, (0.0, 0.0)));
        // Near the inner edge: partially darkened.
        let [r, g, b, a] = rgba_at(&rgba, 200, 52, 70);
        assert_eq!(a, 255, "opaque background stays opaque");
        assert_eq!(r, g, "shadow is black over white");
        assert_eq!(g, b, "shadow is black over white");
        assert!(r > 0 && r < 255, "inner edge darkened, got {r}");
        // Center: the knockout removes the shadow.
        let c = rgba_at(&rgba, 200, 70, 70);
        assert_eq!(c, [255, 255, 255, 255], "center untouched, got {c:?}");
    }

    /// Outset shadow with spread and a transparent background: the
    /// sharp border box is knocked out of the coverage (CSS Backgrounds
    /// 3 §7.2 — outer shadows are drawn outside the border edge only),
    /// so the shadow is a ring, not a filled blob. (An opaque
    /// background used to mask the missing knockout by covering the
    /// middle.)
    #[test]
    fn box_shadow_outset_transparent_background_paints_ring() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::BoxShadow {
            rect: rect(50.0, 50.0, 40.0, 40.0),
            radii: [0.0; 4],
            color: Color::rgba8(255, 0, 0, 255),
            offset: (0.0, 0.0),
            blur: 0.0,
            spread: 10.0,
            inset: false,
        });
        let rgba = render_200x200(&list);
        // Ring band (shape (40,40)-(100,100) minus the box): opaque red.
        assert_eq!(rgba_at(&rgba, 200, 95, 70), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 200, 70, 45), [255, 0, 0, 255]);
        // Inside the border box: knocked out, with no background fill
        // to cover the middle.
        assert_eq!(rgba_at(&rgba, 200, 70, 70), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 200, 55, 55), [0, 0, 0, 0]);
        // Outside the spread shape: nothing.
        assert_eq!(rgba_at(&rgba, 200, 110, 70), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 200, 30, 70), [0, 0, 0, 0]);
    }

    /// Blurred outset shadow with a transparent background: the
    /// knockout applies to the blurred coverage, so the middle stays
    /// hollow while the ring band fades softly.
    #[test]
    fn box_shadow_outset_blurred_transparent_background_stays_hollow() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::BoxShadow {
            rect: rect(50.0, 50.0, 40.0, 40.0),
            radii: [0.0; 4],
            color: Color::rgba8(255, 0, 0, 255),
            offset: (0.0, 0.0),
            blur: 8.0,
            spread: 10.0,
            inset: false,
        });
        let rgba = render_200x200(&list);
        // Deep inside the box: sharp=255 knocks the blurred coverage out.
        assert_eq!(rgba_at(&rgba, 200, 70, 70), [0, 0, 0, 0]);
        // In the ring band: partial coverage from the blur.
        let a = rgba_at(&rgba, 200, 95, 70)[3];
        assert!(a > 0 && a < 255, "blurred ring band, got {a}");
    }

    /// A uniform buffer survives the blur unchanged (clamped edges
    /// replicate); a hot pixel spreads but stays local.
    #[test]
    fn box_blur_sigma_properties() {
        let (w, h) = (21usize, 21usize);
        let mut buf = vec![100u8; w * h];
        let mut tmp = vec![0u8; w * h];
        box_blur_sigma(&mut buf, &mut tmp, w, h, 2.0);
        assert!(
            buf.iter().all(|&b| b == 100),
            "uniform input is a fixed point of the blur"
        );
        let mut buf = vec![0u8; w * h];
        buf[10 * w + 10] = 255;
        box_blur_sigma(&mut buf, &mut tmp, w, h, 2.0);
        assert!(buf[10 * w + 10] < 255, "hot pixel spreads");
        assert!(buf[10 * w + 11] > 0, "neighbor gains coverage");
        assert_eq!(buf[10 * w + 18], 0, "far pixel untouched");
        assert_eq!(buf[0], 0, "corner untouched");
    }

    /// F8: `filter: blur(1e8px)` (sigma 5e7) must not overflow the u32
    /// window sums — a debug panic pre-fix, wraparound garbage in
    /// release. The radius clamps to the image extent; the result must
    /// exactly match a naive u64 reference blur (no wrap corruption).
    #[test]
    fn box_blur_huge_radius_no_overflow() {
        /// Naive clamped box blur, one H+V pair, u64 sums.
        fn reference(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
            let width = 2 * r + 1;
            let mut tmp = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    let mut sum = 0u64;
                    for k in 0..width {
                        let sx = (x as isize + k as isize - r as isize).clamp(0, w as isize - 1)
                            as usize;
                        sum += src[y * w + sx] as u64;
                    }
                    tmp[y * w + x] = (sum / width as u64) as u8;
                }
            }
            let mut out = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    let mut sum = 0u64;
                    for k in 0..width {
                        let sy = (y as isize + k as isize - r as isize).clamp(0, h as isize - 1)
                            as usize;
                        sum += tmp[sy * w + x] as u64;
                    }
                    out[y * w + x] = (sum / width as u64) as u8;
                }
            }
            out
        }
        let (w, h) = (4usize, 3usize);
        // Non-uniform pattern so wraparound corruption would show.
        let mut buf: Vec<u8> = (0..w * h).map(|i| (i * 37 % 251) as u8).collect();
        let mut tmp = vec![0u8; w * h];
        box_blur_sigma(&mut buf, &mut tmp, w, h, 5.0e7);
        // Clamped radius is max(w, h); three H+V pairs like the real thing.
        let r = w.max(h);
        let mut expected: Vec<u8> = (0..w * h).map(|i| (i * 37 % 251) as u8).collect();
        for _ in 0..3 {
            expected = reference(&expected, w, h, r);
        }
        assert_eq!(
            buf, expected,
            "huge-radius blur must match the u64 reference exactly"
        );
    }

    /// F8, RGBA variant: same overflow via `box_blur_sigma_rgba`, same
    /// clamping, per-channel exact match against a u64 reference.
    #[test]
    fn box_blur_sigma_rgba_huge_radius_no_overflow() {
        /// Naive clamped box blur, one H+V pair, u64 sums, per channel.
        fn reference(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
            let width = 2 * r + 1;
            let mut tmp = vec![0u8; w * h * 4];
            for y in 0..h {
                for x in 0..w {
                    for c in 0..4 {
                        let mut sum = 0u64;
                        for k in 0..width {
                            let sx = (x as isize + k as isize - r as isize).clamp(0, w as isize - 1)
                                as usize;
                            sum += src[(y * w + sx) * 4 + c] as u64;
                        }
                        tmp[(y * w + x) * 4 + c] = (sum / width as u64) as u8;
                    }
                }
            }
            let mut out = vec![0u8; w * h * 4];
            for y in 0..h {
                for x in 0..w {
                    for c in 0..4 {
                        let mut sum = 0u64;
                        for k in 0..width {
                            let sy = (y as isize + k as isize - r as isize).clamp(0, h as isize - 1)
                                as usize;
                            sum += tmp[(sy * w + x) * 4 + c] as u64;
                        }
                        out[(y * w + x) * 4 + c] = (sum / width as u64) as u8;
                    }
                }
            }
            out
        }
        let (w, h) = (4usize, 3usize);
        let mut buf: Vec<u8> = (0..w * h * 4)
            .map(|i| ((i * 53 + i / 4) % 251) as u8)
            .collect();
        let mut tmp = vec![0u8; w * h * 4];
        box_blur_sigma_rgba(&mut buf, &mut tmp, w, h, 5.0e7);
        let r = w.max(h);
        let mut expected: Vec<u8> = (0..w * h * 4)
            .map(|i| ((i * 53 + i / 4) % 251) as u8)
            .collect();
        for _ in 0..3 {
            expected = reference(&expected, w, h, r);
        }
        assert_eq!(
            buf, expected,
            "huge-radius RGBA blur must match the u64 reference exactly"
        );
    }

    /// Text shadows take the fontless path without panicking; with no
    /// font data nothing paints.
    #[test]
    fn text_shadow_no_font_paints_nothing() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::TextShadow {
            text: Arc::from("hi"),
            text_range: 0..2,
            rect: rect(10.0, 10.0, 40.0, 20.0),
            baseline: 24.0,
            font_key: FontKey(0),
            size_px: 16.0,
            color: Color::rgba8(0, 0, 0, 255),
            offset: (2.0, 2.0),
            blur: 4.0,
        });
        let rgba = render_200x200(&list);
        assert!(
            rgba.iter().all(|&b| b == 0),
            "fontless text shadow paints nothing"
        );
    }

    #[test]
    fn device_scale_doubles_rect() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = red_rect_list();
        let rgba = backend.render_to_rgba_full(&list, 200, 200, 2.0, (0.0, 0.0));
        assert_eq!(rgba.len(), 200 * 200 * 4);
        // CSS rect [10,10,80,80] at 2x -> device [20,20,160,160].
        assert_eq!(rgba_at(&rgba, 200, 100, 100), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 200, 19, 19), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 200, 180, 180), [0, 0, 0, 0]);
    }

    #[test]
    fn scroll_offsets_content() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = red_rect_list();
        // Scroll by (10, 10): rect moves to device [0,0,80,80].
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (10.0, 10.0));
        assert_eq!(rgba_at(&rgba, 100, 0, 0), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 79, 79), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 80, 80), [0, 0, 0, 0]);
    }

    #[test]
    fn push_clip_restricts_paint() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let green = Color::rgba8(0, 255, 0, 255);
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(0.0, 0.0, 50.0, 50.0),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: green,
        });
        list.push(DisplayItem::PopClip);
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert_eq!(rgba_at(&rgba, 100, 25, 25), [0, 255, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 75, 75), [0, 0, 0, 0]);
    }

    /// Nested clips intersect pixel-exactly: with outer (10,10,60,60)
    /// and inner (40,40,60,60) only [40,70)x[40,70) may paint.
    #[test]
    fn nested_clips_paint_intersection() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(10.0, 10.0, 60.0, 60.0),
        });
        list.push(DisplayItem::PushClip {
            rect: rect(40.0, 40.0, 60.0, 60.0),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: red(),
        });
        list.push(DisplayItem::PopClip);
        list.push(DisplayItem::PopClip);
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // Inside the intersection.
        assert_eq!(rgba_at(&rgba, 100, 40, 40), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 69, 69), [255, 0, 0, 255]);
        // Inside the outer clip but outside the inner one.
        assert_eq!(rgba_at(&rgba, 100, 20, 20), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 100, 39, 50), [0, 0, 0, 0]);
        // Inside the inner clip but outside the outer one.
        assert_eq!(rgba_at(&rgba, 100, 80, 80), [0, 0, 0, 0]);
        // Outside both.
        assert_eq!(rgba_at(&rgba, 100, 5, 5), [0, 0, 0, 0]);
    }

    /// Nested clips with a disjoint inner rect paint nothing at all.
    #[test]
    fn nested_clips_empty_intersection_paints_nothing() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(0.0, 0.0, 30.0, 30.0),
        });
        list.push(DisplayItem::PushClip {
            rect: rect(50.0, 50.0, 30.0, 30.0),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: red(),
        });
        list.push(DisplayItem::PopClip);
        list.push(DisplayItem::PopClip);
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert!(rgba.iter().all(|&b| b == 0));
    }

    /// Popping the inner clip restores the outer clip exactly: paint
    /// after the pop covers the outer rect, not the intersection.
    #[test]
    fn clip_pop_restores_outer_clip() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(10.0, 10.0, 50.0, 50.0),
        });
        list.push(DisplayItem::PushClip {
            rect: rect(30.0, 30.0, 50.0, 50.0),
        });
        list.push(DisplayItem::PopClip);
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: red(),
        });
        list.push(DisplayItem::PopClip);
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // Outer rect painted, including the part the inner clip had cut.
        assert_eq!(rgba_at(&rgba, 100, 15, 15), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 59, 59), [255, 0, 0, 255]);
        // Outside the outer rect stays transparent.
        assert_eq!(rgba_at(&rgba, 100, 70, 70), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 100, 9, 50), [0, 0, 0, 0]);
    }

    /// A clip covering the whole surface must render byte-identical to
    /// no clip at all. This exercises the full-surface fast path in
    /// `active_clip_mask` (all-255 mask handed over as `None`).
    #[test]
    fn full_surface_clip_matches_unclipped() {
        fn render(items: &[DisplayItem]) -> Vec<u8> {
            let mut backend = CpuBackend::new(Rc::new(|_| None));
            let mut list = DisplayList::new();
            for item in items {
                list.push(item.clone());
            }
            backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0))
        }
        let fill = DisplayItem::FillRect {
            rect: rect(10.0, 10.0, 80.0, 80.0),
            color: red(),
        };
        let clipped = render(&[
            DisplayItem::PushClip {
                rect: rect(0.0, 0.0, 100.0, 100.0),
            },
            fill.clone(),
            DisplayItem::PopClip,
        ]);
        let unclipped = render(std::slice::from_ref(&fill));
        assert_eq!(clipped, unclipped);
        // Sanity: the fill really painted.
        assert_eq!(rgba_at(&clipped, 100, 50, 50), [255, 0, 0, 255]);
    }

    /// The persistent clip mask is reused across frames: repainting a
    /// same-size clip does not reallocate the mask buffer.
    #[test]
    fn clip_mask_buffer_is_reused_across_frames() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(0.0, 0.0, 50.0, 50.0),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: red(),
        });
        list.push(DisplayItem::PopClip);
        backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        let ptr1 = backend
            .clip_mask
            .as_ref()
            .expect("clip mask parked on backend")
            .data()
            .as_ptr();
        // Same clip again: the buffer must be reused, not reallocated.
        backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        let ptr2 = backend
            .clip_mask
            .as_ref()
            .expect("clip mask still parked")
            .data()
            .as_ptr();
        assert_eq!(ptr1, ptr2, "clip mask buffer reused across frames");
    }

    /// The incremental mask diff must be byte-identical to a
    /// from-scratch full repaint after every op: random-walk the
    /// tracked region through arbitrary (including empty and
    /// surface-clamped) rects, the way PushClip/PopClip sequences do.
    #[test]
    fn mask_region_diff_matches_full_repaint() {
        fn naive(data: &mut [u8], w: usize, region: Option<MaskRect>) {
            data.fill(0);
            if let Some(r) = region {
                for y in r.y0..r.y1 {
                    data[y * w + r.x0..y * w + r.x1].fill(255);
                }
            }
        }
        fn rand_region(s: &mut u64, w: usize, h: usize) -> Option<MaskRect> {
            // xorshift64*.
            let mut n = || {
                *s ^= *s >> 12;
                *s ^= *s << 25;
                *s ^= *s >> 27;
                *s = s.wrapping_mul(0x2545F4914F6CDD1D);
                (*s >> 32) as usize
            };
            // 1 in 8: the empty (all-zero) region.
            if n() % 8 == 0 {
                return None;
            }
            let (mut x0, mut x1) = (n() % (w + 40), n() % (w + 40));
            let (mut y0, mut y1) = (n() % (h + 40), n() % (h + 40));
            if x0 > x1 {
                std::mem::swap(&mut x0, &mut x1);
            }
            if y0 > y1 {
                std::mem::swap(&mut y0, &mut y1);
            }
            // Clamp like `clip_mask_region`; normalize empties to None.
            let (x0, x1, y0, y1) = (x0.min(w), x1.min(w), y0.min(h), y1.min(h));
            if x0 >= x1 || y0 >= y1 {
                return None;
            }
            Some(MaskRect { x0, y0, x1, y1 })
        }
        const W: usize = 97;
        const H: usize = 61;
        let mut inc = vec![0u8; W * H];
        let mut full = vec![0u8; W * H];
        // A fresh buffer is zero-filled: the tracked region starts empty.
        let mut tracked: Option<MaskRect> = None;
        let mut s = 0x9E3779B97F4A7C15u64;
        for _ in 0..500 {
            let next = rand_region(&mut s, W, H);
            paint_region_diff(&mut inc, W, tracked, next, 0);
            paint_region_diff(&mut inc, W, next, tracked, 255);
            tracked = next;
            naive(&mut full, W, next);
            assert_eq!(inc, full, "incremental mask diverged at {tracked:?}");
        }
    }

    #[test]
    fn push_opacity_fades_paint() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: red(),
        });
        list.push(DisplayItem::PopOpacity);
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        let px = rgba_at(&rgba, 100, 50, 50);
        // 50% opacity red over transparent black -> (255, 0, 0, 128)
        // (allow +-1 for rounding).
        assert_eq!(px[0], 255);
        assert_eq!(px[1], 0);
        assert_eq!(px[2], 0);
        assert!((px[3] as i16 - 128).abs() <= 1, "alpha = {}", px[3]);
    }

    /// Overlapping translucent items inside an opacity group must
    /// composite against each other first (isolated group), then fade as
    /// one: red(0.5) under blue(0.5), group opacity 0.5. Folding the group
    /// opacity into each item (the old approximation) would give
    /// (109, 0, 146, 112) here instead of (85, 0, 170, 96).
    #[test]
    fn opacity_group_overlapping_items_composite_isolated() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(10.0, 10.0, 80.0, 80.0),
            color: Color::from_f32(1.0, 0.0, 0.0, 0.5),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(50.0, 50.0, 80.0, 80.0),
            color: Color::from_f32(0.0, 0.0, 1.0, 0.5),
        });
        list.push(DisplayItem::PopOpacity);
        let rgba = backend.render_to_rgba_full(&list, 130, 130, 1.0, (0.0, 0.0));
        // Overlap region: isolated composite of the two translucent rects
        // (premul ~(64,0,128,192)), then x0.5 -> straight (85, 0, 170, 96).
        let px = rgba_at(&rgba, 130, 60, 60);
        for (got, want) in px.iter().zip([85u8, 0, 170, 96]) {
            assert!(
                (*got as i16 - want as i16).abs() <= 2,
                "overlap pixel {px:?}, want ~(85, 0, 170, 96)"
            );
        }
        // Red-only region: no overlap, so fold and isolated agree:
        // (255, 0, 0, 64).
        let px = rgba_at(&rgba, 130, 20, 20);
        for (got, want) in px.iter().zip([255u8, 0, 0, 64]) {
            assert!(
                (*got as i16 - want as i16).abs() <= 2,
                "red-only pixel {px:?}, want ~(255, 0, 0, 64)"
            );
        }
        // Outside both rects: untouched.
        assert_eq!(rgba_at(&rgba, 130, 5, 5), [0, 0, 0, 0]);
    }

    /// The overscroll edge glow paints the shared alpha gradient —
    /// brightest at the edge, fading to nothing at the band's inner
    /// end — and the emitter's scrollport clip cuts the band: pixels
    /// outside the scrollport stay untouched.
    #[test]
    fn edge_glow_paints_alpha_gradient_clipped_to_scrollport() {
        use crate::display_list::{Edge4, edge_glow_band};

        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        // White backdrop: the glow's alpha reads as darkening.
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 200.0, 200.0),
            color: Color::rgba8(255, 255, 255, 255),
        });
        let scrollport = rect(20.0, 20.0, 160.0, 160.0);
        list.push(DisplayItem::PushClip { rect: scrollport });
        // Band deliberately wider than the scrollport: the clip must
        // cut it back.
        let band = edge_glow_band(scrollport, Edge4::Top).expect("band");
        let wide_band = Rect {
            x: 0.0,
            w: 200.0,
            ..band
        };
        list.push(DisplayItem::EdgeGlow {
            edge: Edge4::Top,
            rect: wide_band,
            strength: 1.0,
        });
        list.push(DisplayItem::PopClip);
        let rgba = backend.render_to_rgba_full(&list, 200, 200, 1.0, (0.0, 0.0));

        // Glow amount = how far the red channel moved from white.
        let glow_at = |x: u32, y: u32| 255i16 - rgba_at(&rgba, 200, x, y)[0] as i16;
        // Alpha gradient: strongest at the edge, fading inward.
        let g_edge = glow_at(100, 21);
        let g_mid = glow_at(100, 70);
        let g_end = glow_at(100, 115);
        assert!(g_edge > 30, "bright at the edge, got {g_edge}");
        assert!(
            g_mid > 0 && g_mid < g_edge,
            "fading inward: edge {g_edge}, mid {g_mid}"
        );
        assert!(g_end <= 4, "gone at the band end: {g_end}");
        // Below the band, inside the scrollport: untouched white.
        assert_eq!(glow_at(100, 150), 0);
        // Outside the scrollport clip: the wide band must not bleed.
        assert_eq!(glow_at(10, 25), 0, "clipped left");
        assert_eq!(glow_at(190, 25), 0, "clipped right");
        assert_eq!(glow_at(100, 10), 0, "clipped above");
    }

    /// Isolated opacity groups allocate a bounds-sized pixmap, not a
    /// full-surface one: a small two-item group on a large surface must
    /// pool a small pixmap, with pixels identical to the full-surface
    /// path.
    #[test]
    fn opacity_group_allocates_bounded_pixmap() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(100.0, 100.0, 100.0, 50.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(120.0, 110.0, 100.0, 50.0),
            color: Color::rgba8(0, 0, 255, 255),
        });
        list.push(DisplayItem::PopOpacity);
        let rgba = backend.render_to_rgba_full(&list, 800, 600, 1.0, (0.0, 0.0));
        // The pooled group pixmap must hug the group bounds
        // (100,100)-(220,160) snapped to device px, not the 800x600
        // full surface.
        assert!(
            !backend.group_pixmap_pool.is_empty(),
            "group pixmap should return to the pool after compositing"
        );
        for px in &backend.group_pixmap_pool {
            assert!(
                px.width() < 800 && px.height() < 600,
                "group pixmap must not be full-surface: {}x{}",
                px.width(),
                px.height()
            );
            assert!(
                px.width() <= 130 && px.height() <= 70,
                "group pixmap should hug the group bounds, got {}x{}",
                px.width(),
                px.height()
            );
        }
        // Pixel correctness: the overlap region is blue-over-red at
        // group opacity 0.5 over transparent black.
        let p = rgba_at(&rgba, 800, 150, 125);
        assert!(p[3] > 0, "group region must paint, got {p:?}");
        assert_eq!(rgba_at(&rgba, 800, 10, 10), [0, 0, 0, 0]);
    }

    /// Nested isolated groups accumulate the device origin through
    /// bounded pixmaps: the inner group's pixels must land at the right
    /// surface position.
    #[test]
    fn opacity_group_nested_bounded_origin_accumulates() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(50.0, 50.0, 100.0, 100.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(100.0, 100.0, 60.0, 60.0),
            color: Color::rgba8(0, 255, 0, 255),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(120.0, 120.0, 60.0, 60.0),
            color: Color::rgba8(0, 0, 255, 255),
        });
        list.push(DisplayItem::PopOpacity);
        list.push(DisplayItem::PopOpacity);
        let rgba = backend.render_to_rgba_full(&list, 400, 400, 1.0, (0.0, 0.0));
        // (130,130) is inside both inner rects: inner pixmap is opaque
        // blue there; inner composite at 0.5 over the outer's opaque red
        // gives (128, 0, 128, 255) in the outer pixmap; the outer
        // composite at 0.5 over transparent black gives (128, 0, 128,
        // 128).
        let p = rgba_at(&rgba, 400, 130, 130);
        for (got, want) in p.iter().zip([128u8, 0, 128, 128]) {
            assert!(
                (*got as i16 - want as i16).abs() <= 2,
                "nested group pixel {p:?}, want ~(128, 0, 128, 128)"
            );
        }
        // (60,60) is outer-red only: (255, 0, 0) at 0.5 over
        // transparent.
        let p = rgba_at(&rgba, 400, 60, 60);
        for (got, want) in p.iter().zip([255u8, 0, 0, 128]) {
            assert!(
                (*got as i16 - want as i16).abs() <= 2,
                "outer-only pixel {p:?}, want ~(255, 0, 0, 128)"
            );
        }
        // Outside everything: untouched.
        assert_eq!(rgba_at(&rgba, 400, 10, 10), [0, 0, 0, 0]);
    }

    /// Nested opacity groups: the inner group's own opacity must not be
    /// folded into its items' alphas (source-over is not linear in the
    /// folded alpha for overlapping items). Opaque red under opaque blue,
    /// opacities 0.5/0.5: overlap -> (0, 0, 255, 64).
    #[test]
    fn opacity_group_nested_groups_composite_correctly() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(10.0, 10.0, 80.0, 80.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(50.0, 50.0, 80.0, 80.0),
            color: Color::rgba8(0, 0, 255, 255),
        });
        list.push(DisplayItem::PopOpacity);
        list.push(DisplayItem::PopOpacity);
        let rgba = backend.render_to_rgba_full(&list, 130, 130, 1.0, (0.0, 0.0));
        let px = rgba_at(&rgba, 130, 60, 60);
        for (got, want) in px.iter().zip([0u8, 0, 255, 64]) {
            assert!(
                (*got as i16 - want as i16).abs() <= 2,
                "overlap pixel {px:?}, want ~(0, 0, 255, 64)"
            );
        }
        let px = rgba_at(&rgba, 130, 20, 20);
        for (got, want) in px.iter().zip([255u8, 0, 0, 64]) {
            assert!(
                (*got as i16 - want as i16).abs() <= 2,
                "red-only pixel {px:?}, want ~(255, 0, 0, 64)"
            );
        }
    }

    /// An opacity group under a clip: group items are clipped when
    /// rendering into the offscreen pixmap, and the composite honors the
    /// clip too.
    #[test]
    fn opacity_group_inside_clip() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(0.0, 0.0, 65.0, 65.0),
        });
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(10.0, 10.0, 80.0, 80.0),
            color: Color::from_f32(1.0, 0.0, 0.0, 0.5),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(50.0, 50.0, 80.0, 80.0),
            color: Color::from_f32(0.0, 0.0, 1.0, 0.5),
        });
        list.push(DisplayItem::PopOpacity);
        list.push(DisplayItem::PopClip);
        let rgba = backend.render_to_rgba_full(&list, 130, 130, 1.0, (0.0, 0.0));
        // Inside the clip: same isolated result as the unclipped group.
        let px = rgba_at(&rgba, 130, 60, 60);
        for (got, want) in px.iter().zip([85u8, 0, 170, 96]) {
            assert!(
                (*got as i16 - want as i16).abs() <= 2,
                "clipped overlap pixel {px:?}, want ~(85, 0, 170, 96)"
            );
        }
        // Outside the clip: nothing, even though the rects extend there.
        assert_eq!(rgba_at(&rgba, 130, 100, 100), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 130, 70, 70), [0, 0, 0, 0]);
    }

    /// Opacity 0.0: the whole group paints nothing (fast path, no
    /// offscreen needed).
    #[test]
    fn opacity_group_zero_paints_nothing() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushOpacity { opacity: 0.0 });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: red(),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(20.0, 20.0, 60.0, 60.0),
            color: Color::rgba8(0, 255, 0, 255),
        });
        list.push(DisplayItem::PopOpacity);
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert!(rgba.iter().all(|&b| b == 0));
    }

    /// NaN opacity normalizes to 0.0 at group entry: `clamp` passes NaN
    /// through, and a NaN alpha must not reach the blend paths. The
    /// group paints nothing and rendering must not panic.
    #[test]
    fn opacity_group_nan_paints_nothing() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushOpacity { opacity: f32::NAN });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: red(),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(20.0, 20.0, 60.0, 60.0),
            color: Color::rgba8(0, 255, 0, 255),
        });
        list.push(DisplayItem::PopOpacity);
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert!(rgba.iter().all(|&b| b == 0));
    }

    /// Opacity 1.0 groups are no-ops: overlapping translucent items paint
    /// exactly as if the group were absent (bit-identical output).
    #[test]
    fn opacity_group_full_opacity_is_noop() {
        let paint = |grouped: bool| {
            let mut backend = CpuBackend::new(Rc::new(|_| None));
            let mut list = DisplayList::new();
            if grouped {
                list.push(DisplayItem::PushOpacity { opacity: 1.0 });
            }
            list.push(DisplayItem::FillRect {
                rect: rect(10.0, 10.0, 80.0, 80.0),
                color: Color::from_f32(1.0, 0.0, 0.0, 0.5),
            });
            list.push(DisplayItem::FillRect {
                rect: rect(50.0, 50.0, 80.0, 80.0),
                color: Color::from_f32(0.0, 0.0, 1.0, 0.5),
            });
            if grouped {
                list.push(DisplayItem::PopOpacity);
            }
            backend.render_to_rgba_full(&list, 130, 130, 1.0, (0.0, 0.0))
        };
        assert_eq!(paint(true), paint(false));
    }

    #[test]
    fn solid_border_paints_edges() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::Border {
            rect: rect(10.0, 10.0, 80.0, 80.0),
            widths: [5.0, 5.0, 5.0, 5.0],
            styles: [BorderStyle::Solid; 4],
            colors: [red(); 4],
            radii: [0.0; 4],
        });
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // Top edge painted, interior untouched.
        assert_eq!(rgba_at(&rgba, 100, 50, 12), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 50, 50), [0, 0, 0, 0]);
    }

    #[test]
    fn rounded_border_paints_corner_band() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::Border {
            rect: rect(10.0, 10.0, 80.0, 80.0),
            widths: [5.0, 5.0, 5.0, 5.0],
            styles: [BorderStyle::Solid; 4],
            colors: [red(); 4],
            radii: [10.0; 4],
        });
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // Straight top bar painted.
        assert_eq!(rgba_at(&rgba, 100, 50, 12), [255, 0, 0, 255]);
        // Inside the corner band (dist ~7 from the arc center, band
        // 5..10): painted.
        assert_eq!(rgba_at(&rgba, 100, 15, 15), [255, 0, 0, 255]);
        // Outside the outer radius (dist ~12.7 > 10): cut out.
        assert_eq!(rgba_at(&rgba, 100, 11, 11), [0, 0, 0, 0]);
        // Interior untouched.
        assert_eq!(rgba_at(&rgba, 100, 50, 50), [0, 0, 0, 0]);
    }

    #[test]
    fn border_with_no_visible_sides_paints_nothing() {
        // No asymmetric fallback: empty tessellation paints nothing,
        // matching the GPU backend.
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::Border {
            rect: rect(10.0, 10.0, 80.0, 80.0),
            widths: [5.0, 5.0, 5.0, 5.0],
            styles: [BorderStyle::None; 4],
            colors: [red(); 4],
            radii: [0.0; 4],
        });
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert!(rgba.iter().all(|&b| b == 0));
    }

    #[test]
    fn image_draws_rgba_pixels() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        // 2x2 straight RGBA: red, green / blue, white.
        list.add_image(
            ImageId(7),
            crate::display_list::DecodedImage::from_rgba(
                2,
                2,
                vec![
                    255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
                ],
            ),
        );
        list.push(DisplayItem::Image {
            rect: rect(10.0, 10.0, 2.0, 2.0),
            image: ImageId(7),
        });
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert_eq!(rgba_at(&rgba, 100, 10, 10), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 11, 10), [0, 255, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 10, 11), [0, 0, 255, 255]);
        assert_eq!(rgba_at(&rgba, 100, 11, 11), [255, 255, 255, 255]);
        assert_eq!(rgba_at(&rgba, 100, 12, 12), [0, 0, 0, 0]);
    }

    /// The persistent image cache replaces (never serves stale) an entry
    /// whose id is reused with different bytes.
    #[test]
    fn image_cache_replaces_entry_on_content_change() {
        use crate::display_list::DecodedImage;
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let red = DecodedImage::from_rgba(2, 2, [255u8, 0, 0, 255].repeat(4));
        let mut list = DisplayList::new();
        list.add_image(ImageId(3), red.clone());
        list.push(DisplayItem::Image {
            rect: rect(0.0, 0.0, 2.0, 2.0),
            image: ImageId(3),
        });
        let rgba = backend.render_to_rgba_full(&list, 2, 2, 1.0, (0.0, 0.0));
        assert_eq!(rgba_at(&rgba, 2, 0, 0), [255, 0, 0, 255]);
        let hash1 = backend
            .image_cache
            .by_id
            .get(&ImageId(3))
            .expect("cached")
            .hash;

        // Same id, blue bytes: the cache must re-decode, not serve red.
        let blue = DecodedImage::from_rgba(2, 2, [0u8, 0, 255, 255].repeat(4));
        let mut list2 = DisplayList::new();
        list2.add_image(ImageId(3), blue.clone());
        list2.push(DisplayItem::Image {
            rect: rect(0.0, 0.0, 2.0, 2.0),
            image: ImageId(3),
        });
        let rgba2 = backend.render_to_rgba_full(&list2, 2, 2, 1.0, (0.0, 0.0));
        assert_eq!(rgba_at(&rgba2, 2, 0, 0), [0, 0, 255, 255]);
        let entry = backend
            .image_cache
            .by_id
            .get(&ImageId(3))
            .expect("still cached");
        assert_eq!(entry.hash, blue.content_hash());
        assert_ne!(entry.hash, hash1);
        assert_eq!(backend.image_cache.by_id.len(), 1, "one entry per id");
    }

    /// Frame-boundary eviction is precise, not a blind clear: a frame
    /// referencing `MAX_CACHED_CPU_IMAGES + 1` distinct images keeps all
    /// of them (no cap reset), and a following frame that references only
    /// one id evicts the stale entries while the still-referenced entry
    /// stays hot (identical bytes: hash unchanged, no re-decode churn).
    #[test]
    fn image_cache_evicts_only_unreferenced_entries_at_frame_start() {
        use crate::display_list::DecodedImage;
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        for i in 0..=MAX_CACHED_CPU_IMAGES as u32 {
            list.add_image(
                ImageId(i),
                DecodedImage::from_rgba(1, 1, vec![i as u8, 0, 0, 255]),
            );
            list.push(DisplayItem::Image {
                rect: rect(0.0, 0.0, 1.0, 1.0),
                image: ImageId(i),
            });
        }
        backend.render_to_rgba_full(&list, 1, 1, 1.0, (0.0, 0.0));
        assert_eq!(
            backend.image_cache.by_id.len(),
            MAX_CACHED_CPU_IMAGES + 1,
            "no cap reset: every referenced image stays cached"
        );

        let hash_before = backend.image_cache.by_id[&ImageId(0)].hash;
        let mut second = DisplayList::new();
        second.add_image(
            ImageId(0),
            DecodedImage::from_rgba(1, 1, vec![0, 0, 0, 255]),
        );
        second.push(DisplayItem::Image {
            rect: rect(0.0, 0.0, 1.0, 1.0),
            image: ImageId(0),
        });
        backend.render_to_rgba_full(&second, 1, 1, 1.0, (0.0, 0.0));
        assert_eq!(backend.image_cache.by_id.len(), 1, "stale entries evicted");
        assert_eq!(
            backend.image_cache.by_id[&ImageId(0)].hash,
            hash_before,
            "still-referenced entry stayed hot"
        );
    }

    /// The frame-start eviction rebuilds `by_id_order` exactly through
    /// the hoisted scratch buffers: insert A, B, C, then a frame
    /// referencing only B, and the queue is exactly `[B]` — no stale
    /// ids linger, and each live id appears exactly once.
    #[test]
    fn frame_eviction_rebuilds_fifo_queue_exactly_once() {
        use crate::display_list::DecodedImage;
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let pixel = |r: u8| DecodedImage::from_rgba(1, 1, vec![r, 0, 0, 255]);
        let mut list = DisplayList::new();
        for (id, r) in [(0u32, 10u8), (1, 20), (2, 30)] {
            list.add_image(ImageId(id), pixel(r));
            list.push(DisplayItem::Image {
                rect: rect(0.0, 0.0, 1.0, 1.0),
                image: ImageId(id),
            });
        }
        backend.render_to_rgba_full(&list, 1, 1, 1.0, (0.0, 0.0));
        let order: Vec<u32> = backend
            .image_cache
            .by_id_order
            .iter()
            .map(|id| id.0)
            .collect();
        assert_eq!(order, [0, 1, 2], "all three ids queued in order");

        // Second frame references only B: A and C are evicted, and the
        // rebuilt queue must be exactly [B].
        let mut second = DisplayList::new();
        second.add_image(ImageId(1), pixel(20));
        second.push(DisplayItem::Image {
            rect: rect(0.0, 0.0, 1.0, 1.0),
            image: ImageId(1),
        });
        backend.render_to_rgba_full(&second, 1, 1, 1.0, (0.0, 0.0));
        let order: Vec<u32> = backend
            .image_cache
            .by_id_order
            .iter()
            .map(|id| id.0)
            .collect();
        assert_eq!(order, [1], "queue rebuilt to exactly [B]");
        assert!(
            !backend.image_cache.by_id.contains_key(&ImageId(0))
                && !backend.image_cache.by_id.contains_key(&ImageId(2)),
            "evicted ids gone from by_id"
        );
        // Exactly-once invariant: queue mirrors the live set, each id
        // exactly once.
        let cache = &backend.image_cache;
        assert_eq!(
            cache.by_id_order.len(),
            cache.by_id.len(),
            "queue length matches live entries"
        );
        let mut seen = HashSet::new();
        for id in &cache.by_id_order {
            assert!(seen.insert(id.0), "id {:?} appears more than once", id);
            assert!(cache.by_id.contains_key(id), "queued id is live");
        }
    }

    /// Two different [`ImageId`]s with identical bytes decode exactly
    /// once: the second id hits the content-hash-keyed shared cache
    /// instead of premultiplying again.
    #[test]
    fn identical_images_share_one_decode() {
        use crate::display_list::DecodedImage;
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let bytes = [9u8, 128, 200, 255].repeat(16); // 4x4, opaque.
        let mut list = DisplayList::new();
        list.add_image(ImageId(1), DecodedImage::from_rgba(4, 4, bytes.clone()));
        list.add_image(ImageId(2), DecodedImage::from_rgba(4, 4, bytes));
        list.push(DisplayItem::Image {
            rect: rect(0.0, 0.0, 4.0, 4.0),
            image: ImageId(1),
        });
        list.push(DisplayItem::Image {
            rect: rect(4.0, 0.0, 4.0, 4.0),
            image: ImageId(2),
        });
        let rgba = backend.render_to_rgba_full(&list, 8, 4, 1.0, (0.0, 0.0));
        let stats = &backend.cache_stats;
        assert_eq!(stats.image_decodes, 1, "identical bytes decode once");
        assert_eq!(stats.image_shared_hits, 1, "second id shares the decode");
        assert_eq!(stats.image_id_hits, 0);
        // Both ids render the same pixels (alpha 255: no premultiply shift).
        assert_eq!(rgba_at(&rgba, 8, 0, 0), [9, 128, 200, 255]);
        assert_eq!(rgba_at(&rgba, 8, 4, 0), [9, 128, 200, 255]);
    }

    /// Measurement: a repeated-image scene — one image tiled 50 times
    /// per frame, with fresh ids per frame the way a rebuilt display
    /// list mints them — decodes once and serves every other reference
    /// from cache. Hit rate must round to ~1 (149/150 lookups).
    #[test]
    fn image_cache_hit_rate_on_repeated_image_scene() {
        use crate::display_list::DecodedImage;
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let bytes = [40u8, 90, 160, 255].repeat(64); // 8x8, opaque.
        let mut next_id = 0u32;
        for _ in 0..3 {
            let mut list = DisplayList::new();
            for i in 0..50 {
                let id = ImageId(next_id);
                next_id += 1;
                list.add_image(id, DecodedImage::from_rgba(8, 8, bytes.clone()));
                list.push(DisplayItem::Image {
                    rect: rect((i % 10) as f32 * 8.0, (i / 10) as f32 * 8.0, 8.0, 8.0),
                    image: id,
                });
            }
            backend.render_to_rgba_full(&list, 80, 40, 1.0, (0.0, 0.0));
        }
        let s = &backend.cache_stats;
        let lookups = s.image_decodes + s.image_id_hits + s.image_shared_hits;
        let hits = s.image_id_hits + s.image_shared_hits;
        let hit_rate = hits as f64 / lookups as f64;
        eprintln!(
            "[measure] repeated-image scene: decodes={} id_hits={} shared_hits={} lookups={} hit_rate={:.4}",
            s.image_decodes, s.image_id_hits, s.image_shared_hits, lookups, hit_rate
        );
        assert_eq!(lookups, 150);
        assert_eq!(s.image_decodes, 1, "one decode for 150 references");
        assert_eq!(s.image_shared_hits, 149, "fresh ids still share content");
        assert!(hit_rate > 0.99, "hit_rate={hit_rate}");
    }

    /// The shared decode cache is bounded: past
    /// [`MAX_SHARED_DECODED_IMAGES`] distinct contents the oldest are
    /// evicted FIFO, and the newest stay.
    #[test]
    fn shared_decode_cache_evicts_fifo_past_cap() {
        use crate::display_list::DecodedImage;
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let first_hash = DecodedImage::from_rgba(1, 1, vec![0, 0, 0, 255]).content_hash();
        let last_hash = DecodedImage::from_rgba(1, 1, vec![2, 0, 0, 255]).content_hash();
        let mut list = DisplayList::new();
        for i in 0..=MAX_SHARED_DECODED_IMAGES as u32 {
            // Two varying bytes: all 513 contents distinct.
            let hi = (i / 256) as u8;
            let lo = (i % 256) as u8;
            list.add_image(
                ImageId(i),
                DecodedImage::from_rgba(1, 1, vec![hi, lo, 0, 255]),
            );
            list.push(DisplayItem::Image {
                rect: rect(0.0, 0.0, 1.0, 1.0),
                image: ImageId(i),
            });
        }
        backend.render_to_rgba_full(&list, 1, 1, 1.0, (0.0, 0.0));
        let cache = &backend.image_cache;
        assert_eq!(
            cache.shared.len(),
            MAX_SHARED_DECODED_IMAGES,
            "shared cache capped at MAX_SHARED_DECODED_IMAGES"
        );
        assert_eq!(cache.shared_order.len(), cache.shared.len());
        assert!(
            cache.shared_bytes + cache.by_id_bytes <= MAX_DECODED_IMAGE_BYTES,
            "combined byte budget respected"
        );
        assert!(
            !cache.shared.contains_key(&first_hash),
            "oldest content evicted FIFO"
        );
        assert!(
            cache.shared.contains_key(&last_hash),
            "newest content retained"
        );
    }

    /// `CpuImageEntry::matches` verifies every component of the stored
    /// content identity: a hash match alone (simulating an FNV-1a
    /// collision) with a different shape or byte length is not a hit.
    #[test]
    fn cpu_image_entry_matches_verifies_shape_and_length() {
        use crate::display_list::DecodedImage;
        let entry = CpuImageEntry {
            pixmap: Arc::new(Pixmap::new(2, 2).expect("pixmap")),
            hash: 0x1234_5678,
            width: 2,
            height: 2,
            byte_len: 16,
        };
        assert!(entry.matches(0x1234_5678, 2, 2, 16));
        assert!(
            !entry.matches(0x1234_5678, 4, 1, 16),
            "same hash and byte count, different shape: not a hit"
        );
        assert!(
            !entry.matches(0x1234_5678, 2, 2, 15),
            "byte length mismatch: not a hit"
        );
        assert!(
            !entry.matches(0x1234_5679, 2, 2, 16),
            "hash mismatch: not a hit"
        );
        // Sanity: the byte length of a real image is w*h*4.
        let img = DecodedImage::from_rgba(2, 2, vec![7u8; 16]);
        assert_eq!(img.rgba.len(), 2 * 2 * 4);
    }

    /// Test helper: insert a front-cache entry the way
    /// [`Painter::image_pixmap`] does, keeping the byte accounting and
    /// FIFO queue exact (replaces refresh recency, like production).
    fn plant_front_entry(cache: &mut ImageCache, id: ImageId, entry: CpuImageEntry) {
        let fresh = !cache.by_id.contains_key(&id);
        let byte_len = entry.byte_len;
        if let Some(old) = cache.by_id.insert(id, entry) {
            cache.sub_by_id_bytes(old.byte_len);
        }
        cache.by_id_bytes += byte_len;
        cache.refresh_by_id_order(id, fresh);
    }

    /// Test helper: plant a shared-cache entry the way
    /// [`Painter::image_pixmap`] does.
    fn plant_shared_entry(
        cache: &mut ImageCache,
        hash: u64,
        width: u32,
        height: u32,
        pixmap: Arc<Pixmap>,
    ) {
        if let Some(old) = cache.shared.insert(
            hash,
            SharedDecode {
                pixmap: Arc::clone(&pixmap),
                width,
                height,
            },
        ) {
            cache.sub_shared_bytes(decoded_bytes(&old.pixmap));
            cache.shared_order.retain(|&h| h != hash);
        }
        cache.shared_bytes += decoded_bytes(&pixmap);
        cache.shared_order.push_back(hash);
    }

    /// A content-hash hit is verified against the stored dimensions and
    /// byte length before the cached pixmap is served. Plant a front
    /// entry whose hash equals the image's real hash (a forced FNV-1a
    /// collision) but whose stored shape disagrees with the actual
    /// source: the lookup must decode fresh instead of serving the
    /// planted pixels.
    #[test]
    fn image_cache_hit_verifies_dimensions_before_serving() {
        use crate::display_list::DecodedImage;

        let mut px = Pixmap::new(8, 8).expect("pixmap");
        let hook: Rc<dyn Fn(FontKey) -> Option<FontData>> = Rc::new(|_| None);
        let mut glyphs = GlyphAtlas::new();
        let mut glyph_misses = 0usize;
        let mut clip_mask: Option<Mask> = None;
        let mut mask_region: Option<MaskRect> = None;
        let mut mask_stack: Vec<ClipEntry> = Vec::new();
        let mut image_cache = ImageCache::default();
        let mut cache_stats = CacheStats::default();
        let mut group_pool: Vec<Pixmap> = Vec::new();
        let mut gradient_bake_cache: HashMap<u64, CpuGradientBake> = HashMap::new();
        let mut filter_scratch = Vec::new();
        let mut shadow_scratch_pool: Vec<Vec<u8>> = Vec::new();
        let mut border_scratch: Vec<Quad> = Vec::new();
        let mut painter = Painter::new(
            &mut px,
            &hook,
            &mut glyphs,
            &mut glyph_misses,
            &mut clip_mask,
            &mut mask_region,
            &mut mask_stack,
            &mut image_cache,
            &mut cache_stats,
            &mut group_pool,
            &mut filter_scratch,
            &mut shadow_scratch_pool,
            &mut border_scratch,
            &mut gradient_bake_cache,
            1,
            1.0,
            (0.0, 0.0),
        );

        // 2x2 opaque red image under id 9.
        let mut list = DisplayList::new();
        list.add_image(
            ImageId(9),
            DecodedImage::from_rgba(2, 2, [255u8, 0, 0, 255].repeat(4)),
        );
        let real_hash = list.image(ImageId(9)).expect("registered").content_hash();

        let planted = |w: u32, h: u32| CpuImageEntry {
            pixmap: Arc::new(
                Pixmap::from_vec(
                    [0u8, 255, 0, 255].repeat(4),
                    IntSize::from_wh(2, 2).expect("size"),
                )
                .expect("pixmap"),
            ),
            hash: real_hash,
            width: w,
            height: h,
            byte_len: 16,
        };
        // Hash matches, stored shape (4x1) disagrees with the 2x2 source.
        plant_front_entry(painter.image_cache, ImageId(9), planted(4, 1));
        let pm = painter
            .image_pixmap(&list, ImageId(9))
            .expect("decodes on dimension mismatch");
        assert_eq!(
            &pm.data()[..4],
            &[255, 0, 0, 255],
            "dimension mismatch must decode fresh, not serve planted pixels"
        );
        assert_eq!(painter.cache_stats.image_decodes, 1, "treated as miss");
        assert_eq!(painter.cache_stats.image_id_hits, 0);

        // Positive control: the same planted entry with a matching shape
        // IS served from the front cache.
        plant_front_entry(painter.image_cache, ImageId(9), planted(2, 2));
        let pm = painter.image_pixmap(&list, ImageId(9)).expect("hit");
        assert_eq!(
            &pm.data()[..4],
            &[0, 255, 0, 255],
            "matching shape serves the cached pixmap"
        );
        assert_eq!(painter.cache_stats.image_id_hits, 1);
    }

    /// The shared content-hash cache verifies stored dimensions before
    /// serving, like the per-id front cache. Plant a shared entry whose
    /// key equals the image's real hash (a forced FNV-1a collision) but
    /// whose stored shape disagrees with the actual source: the lookup
    /// must decode fresh instead of serving the planted pixels, and the
    /// stale entry is replaced by the fresh decode.
    #[test]
    fn image_cache_shared_hit_verifies_dimensions_before_serving() {
        use crate::display_list::DecodedImage;

        let mut px = Pixmap::new(8, 8).expect("pixmap");
        let hook: Rc<dyn Fn(FontKey) -> Option<FontData>> = Rc::new(|_| None);
        let mut glyphs = GlyphAtlas::new();
        let mut glyph_misses = 0usize;
        let mut clip_mask: Option<Mask> = None;
        let mut mask_region: Option<MaskRect> = None;
        let mut mask_stack: Vec<ClipEntry> = Vec::new();
        let mut image_cache = ImageCache::default();
        let mut cache_stats = CacheStats::default();
        let mut group_pool: Vec<Pixmap> = Vec::new();
        let mut gradient_bake_cache: HashMap<u64, CpuGradientBake> = HashMap::new();
        let mut filter_scratch = Vec::new();
        let mut shadow_scratch_pool: Vec<Vec<u8>> = Vec::new();
        let mut border_scratch: Vec<Quad> = Vec::new();
        let mut painter = Painter::new(
            &mut px,
            &hook,
            &mut glyphs,
            &mut glyph_misses,
            &mut clip_mask,
            &mut mask_region,
            &mut mask_stack,
            &mut image_cache,
            &mut cache_stats,
            &mut group_pool,
            &mut filter_scratch,
            &mut shadow_scratch_pool,
            &mut border_scratch,
            &mut gradient_bake_cache,
            1,
            1.0,
            (0.0, 0.0),
        );

        // 2x2 opaque red image under id 9.
        let mut list = DisplayList::new();
        list.add_image(
            ImageId(9),
            DecodedImage::from_rgba(2, 2, [255u8, 0, 0, 255].repeat(4)),
        );
        let real_hash = list.image(ImageId(9)).expect("registered").content_hash();

        // Shared entry keyed by the real hash, but decoded from 4x1
        // green pixels: same hash, different shape.
        let green_4x1 = Arc::new(
            Pixmap::from_vec(
                [0u8, 255, 0, 255].repeat(4),
                IntSize::from_wh(4, 1).expect("size"),
            )
            .expect("pixmap"),
        );
        plant_shared_entry(painter.image_cache, real_hash, 4, 1, green_4x1);
        let pm = painter
            .image_pixmap(&list, ImageId(9))
            .expect("decodes on dimension mismatch");
        assert_eq!(
            &pm.data()[..4],
            &[255, 0, 0, 255],
            "shared dimension mismatch must decode fresh, not serve planted pixels"
        );
        assert_eq!(painter.cache_stats.image_decodes, 1, "treated as miss");
        assert_eq!(painter.cache_stats.image_shared_hits, 0);
        // The stale colliding entry was replaced by the fresh decode:
        // the shared slot now names the real 2x2 shape.
        let shared = painter
            .image_cache
            .shared
            .get(&real_hash)
            .expect("shared entry");
        assert_eq!((shared.width, shared.height), (2, 2));
        assert_eq!(
            painter.image_cache.shared_order.len(),
            1,
            "no duplicate queue slot"
        );

        // Positive control: the same bytes under a fresh id hit the
        // shared cache (the front cache misses on the new id).
        list.add_image(
            ImageId(10),
            DecodedImage::from_rgba(2, 2, [255u8, 0, 0, 255].repeat(4)),
        );
        let pm = painter.image_pixmap(&list, ImageId(10)).expect("hit");
        assert_eq!(
            &pm.data()[..4],
            &[255, 0, 0, 255],
            "matching shape serves the shared decode"
        );
        assert_eq!(painter.cache_stats.image_decodes, 1, "no second decode");
        assert_eq!(painter.cache_stats.image_shared_hits, 1);
    }

    /// The gradient bake cache serves repeat paints of the same
    /// (gradient, size) without re-baking; a different bake size is a
    /// different entry.
    #[test]
    fn gradient_bake_cache_serves_repeat_paints() {
        use crate::gradient::parse_gradient;
        use bowser_layout::Rect;

        let mut px = Pixmap::new(16, 16).expect("pixmap");
        let hook: Rc<dyn Fn(FontKey) -> Option<FontData>> = Rc::new(|_| None);
        let mut glyphs = GlyphAtlas::new();
        let mut glyph_misses = 0usize;
        let mut clip_mask: Option<Mask> = None;
        let mut mask_region: Option<MaskRect> = None;
        let mut mask_stack: Vec<ClipEntry> = Vec::new();
        let mut image_cache = ImageCache::default();
        let mut cache_stats = CacheStats::default();
        let mut group_pool: Vec<Pixmap> = Vec::new();
        let mut gradient_bake_cache: HashMap<u64, CpuGradientBake> = HashMap::new();
        let mut filter_scratch = Vec::new();
        let mut shadow_scratch_pool: Vec<Vec<u8>> = Vec::new();
        let mut border_scratch: Vec<Quad> = Vec::new();
        let mut painter = Painter::new(
            &mut px,
            &hook,
            &mut glyphs,
            &mut glyph_misses,
            &mut clip_mask,
            &mut mask_region,
            &mut mask_stack,
            &mut image_cache,
            &mut cache_stats,
            &mut group_pool,
            &mut filter_scratch,
            &mut shadow_scratch_pool,
            &mut border_scratch,
            &mut gradient_bake_cache,
            1,
            1.0,
            (0.0, 0.0),
        );
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 16.0,
            h: 16.0,
        };
        let parsed = parse_gradient("linear-gradient(red, blue)").expect("parse");
        let gradient = parsed
            .resolve(&rect, Color::rgba8(0, 0, 0, 255))
            .expect("resolve");
        painter.paint_gradient(rect, &gradient);
        painter.paint_gradient(rect, &gradient);
        assert_eq!(painter.cache_stats.gradient_bakes, 1, "one bake");
        assert_eq!(painter.cache_stats.gradient_bake_hits, 1, "one hit");
        assert_eq!(painter.gradient_bake_cache.len(), 1);
        // A different bake size is a different entry.
        painter.paint_gradient(
            Rect {
                x: 0.0,
                y: 0.0,
                w: 8.0,
                h: 8.0,
            },
            &gradient,
        );
        assert_eq!(painter.cache_stats.gradient_bakes, 2);
        assert_eq!(painter.gradient_bake_cache.len(), 2);
        // The cached bytes match a fresh bake.
        let key = gradient_bake_key(&gradient, 16, 16);
        let entry = painter.gradient_bake_cache.get(&key).expect("cached");
        assert_eq!(entry.bytes, bake_gradient_rgba8(&gradient, &rect, 16, 16));
    }

    /// Frame-boundary eviction (mirrors the WGPU backend): the cache
    /// holds at most the frame's distinct gradients; an entry the
    /// current frame does not stamp is evicted.
    #[test]
    fn gradient_bake_cache_evicts_stale_entries_per_frame() {
        use crate::display_list::DisplayListBuilder;
        use crate::gradient::parse_gradient;
        use bowser_layout::Rect;

        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list_with = |css: &str| {
            let mut builder = DisplayListBuilder::new();
            let rect = Rect {
                x: 0.0,
                y: 0.0,
                w: 16.0,
                h: 16.0,
            };
            let parsed = parse_gradient(css).expect("parse");
            builder.push_gradient(rect, &parsed, Color::rgba8(0, 0, 0, 255));
            builder.build()
        };
        let red_blue = list_with("linear-gradient(red, blue)");
        backend.render_to_rgba_full(&red_blue, 16, 16, 1.0, (0.0, 0.0));
        assert_eq!(backend.gradient_bake_cache.len(), 1);
        assert_eq!(backend.cache_stats.gradient_bakes, 1);
        // Same gradient next frame: served from the cache, still one
        // entry.
        backend.render_to_rgba_full(&red_blue, 16, 16, 1.0, (0.0, 0.0));
        assert_eq!(backend.gradient_bake_cache.len(), 1);
        assert_eq!(backend.cache_stats.gradient_bakes, 1);
        assert_eq!(backend.cache_stats.gradient_bake_hits, 1);
        // A frame without that gradient evicts the stale entry.
        let green_yellow = list_with("linear-gradient(green, yellow)");
        backend.render_to_rgba_full(&green_yellow, 16, 16, 1.0, (0.0, 0.0));
        assert_eq!(backend.gradient_bake_cache.len(), 1);
        assert_eq!(backend.cache_stats.gradient_bakes, 2);
    }

    /// The byte bound covers retained decoded memory, not just the
    /// shared map: a frame referencing five distinct 16 MiB images
    /// (80 MiB total) must leave `shared_bytes + by_id_bytes` within
    /// the budget plus one image of slack, even though every image
    /// holds an independent `Arc` in the per-id front cache. Before the
    /// fix, the front entries alone retained all 80 MiB past `shared`
    /// eviction.
    #[test]
    fn image_cache_combined_byte_budget_bounds_front_entries() {
        use crate::display_list::DecodedImage;
        const W: u32 = 2048;
        const H: u32 = 2048;
        const IMAGE_BYTES: usize = W as usize * H as usize * 4; // 16 MiB.
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        for i in 0..5u32 {
            // Distinct contents (varying first byte) so nothing dedups.
            let mut bytes = vec![0u8; IMAGE_BYTES];
            bytes[0] = i as u8;
            bytes[3] = 255;
            list.add_image(ImageId(i), DecodedImage::from_rgba(W, H, bytes));
            list.push(DisplayItem::Image {
                rect: rect(0.0, 0.0, 8.0, 8.0),
                image: ImageId(i),
            });
        }
        backend.render_to_rgba_full(&list, 64, 64, 1.0, (0.0, 0.0));
        let cache = &backend.image_cache;
        // Accounting is exact: every live entry's bytes are counted once.
        let by_id_accounted: usize = cache.by_id.values().map(|e| e.byte_len).sum();
        assert_eq!(
            cache.by_id_bytes, by_id_accounted,
            "by_id_bytes tracks live entries exactly"
        );
        let shared_accounted: usize = cache
            .shared
            .values()
            .map(|e| decoded_bytes(&e.pixmap))
            .sum();
        assert_eq!(
            cache.shared_bytes, shared_accounted,
            "shared_bytes tracks live entries exactly"
        );
        assert!(
            cache.shared_bytes + cache.by_id_bytes <= MAX_DECODED_IMAGE_BYTES + IMAGE_BYTES,
            "combined decoded memory bounded: shared={} by_id={}",
            cache.shared_bytes,
            cache.by_id_bytes,
        );
        // The per-id FIFO queue stayed exact (no unbounded growth).
        assert_eq!(
            cache.by_id_order.len(),
            cache.by_id.len(),
            "eviction queue mirrors live entries"
        );
    }

    /// `evict_over_budget` never evicts the just-inserted (`protected`)
    /// entry, even when it alone exceeds the budget: the caller is about
    /// to paint from it. Older entries go first, FIFO.
    #[test]
    fn evict_over_budget_never_evicts_protected_entry() {
        let mut cache = ImageCache::default();
        for (id, hash) in [(ImageId(1), 11u64), (ImageId(2), 22u64)] {
            let pm = Arc::new(Pixmap::new(2, 2).expect("pixmap"));
            plant_shared_entry(&mut cache, hash, 2, 2, Arc::clone(&pm));
            cache.by_id.insert(
                id,
                CpuImageEntry {
                    pixmap: pm,
                    hash,
                    width: 2,
                    height: 2,
                    byte_len: 16,
                },
            );
            cache.by_id_order.push_back(id);
            cache.by_id_bytes += 16;
        }
        // Simulate a huge just-inserted entry pushing the accounted
        // total over budget.
        cache.by_id_bytes += MAX_DECODED_IMAGE_BYTES;
        cache.evict_over_budget(ImageId(2));
        assert!(
            cache.by_id.contains_key(&ImageId(2)),
            "protected entry must survive"
        );
        assert!(
            !cache.by_id.contains_key(&ImageId(1)),
            "older entries evicted first"
        );
        assert!(
            cache.shared_bytes + cache.by_id_bytes <= MAX_DECODED_IMAGE_BYTES + 16,
            "budget restored to budget plus one image"
        );
    }

    /// Replacing an old id refreshes its FIFO recency: the replaced id
    /// moves to the back of `by_id_order`, so `evict_over_budget` pops
    /// older evictable entries first instead of hitting the protected
    /// (just-inserted) id at its stale front position, re-queueing it,
    /// and stopping with the combined total still over budget by more
    /// than one image of slack. Before the fix, id 1 sat at the front
    /// after its replace, so eviction stopped immediately and the total
    /// stayed at 88 MiB (over the 64 + 16 MiB allowed).
    #[test]
    fn evict_over_budget_replaced_id_refreshes_fifo_order() {
        const MIB: usize = 1024 * 1024;
        let entry = |w: u32, h: u32| CpuImageEntry {
            pixmap: Arc::new(Pixmap::new(w, h).expect("pixmap")),
            hash: 0,
            width: w,
            height: h,
            byte_len: w as usize * h as usize * 4,
        };
        let mut cache = ImageCache::default();
        // Id 1 first (4 MiB), then ids 2-4 (24 MiB each): 76 MiB total.
        plant_front_entry(&mut cache, ImageId(1), entry(1024, 1024));
        for id in 2..=4u32 {
            plant_front_entry(&mut cache, ImageId(id), entry(2048, 3072));
        }
        // Replace id 1 with a larger image (16 MiB), as `image_pixmap`
        // does on content change; id 1 is the protected just-inserted
        // entry. Total is now 88 MiB, over the 64 MiB budget.
        plant_front_entry(&mut cache, ImageId(1), entry(2048, 2048));
        cache.evict_over_budget(ImageId(1));
        assert!(
            cache.by_id.contains_key(&ImageId(1)),
            "protected replaced entry must survive"
        );
        assert!(
            !cache.by_id.contains_key(&ImageId(2)),
            "oldest evictable entry goes first"
        );
        assert!(
            cache.shared_bytes + cache.by_id_bytes <= MAX_DECODED_IMAGE_BYTES + 16 * MIB,
            "combined total within budget plus one-image slack, got shared={} by_id={}",
            cache.shared_bytes,
            cache.by_id_bytes,
        );
        // The per-id FIFO queue stayed exact (no unbounded growth) and
        // reflects the refreshed order.
        assert_eq!(
            cache.by_id_order.len(),
            cache.by_id.len(),
            "eviction queue mirrors live entries"
        );
        let order: Vec<u32> = cache.by_id_order.iter().map(|id| id.0).collect();
        assert_eq!(
            order,
            vec![3, 4, 1],
            "replaced id moved to the back, got {order:?}"
        );
    }

    /// `sub_by_id_bytes` keeps the hand-maintained counter exactly in
    /// sync with the map contents across plants, replaces, and
    /// removals: after every mutation the counter must equal the sum of
    /// the live entries' `byte_len`.
    #[test]
    fn by_id_bytes_tracks_entries_exactly() {
        fn sum_bytes(cache: &ImageCache) -> usize {
            cache.by_id.values().map(|e| e.byte_len).sum()
        }
        let entry = |w: u32, h: u32| CpuImageEntry {
            pixmap: Arc::new(Pixmap::new(w, h).expect("pixmap")),
            hash: 0,
            width: w,
            height: h,
            byte_len: w as usize * h as usize * 4,
        };
        let mut cache = ImageCache::default();
        assert_eq!(cache.by_id_bytes, 0);

        // Plant three entries; the counter must match exactly.
        plant_front_entry(&mut cache, ImageId(1), entry(2, 2)); // 16 B
        plant_front_entry(&mut cache, ImageId(2), entry(4, 4)); // 64 B
        plant_front_entry(&mut cache, ImageId(3), entry(8, 8)); // 256 B
        assert_eq!(cache.by_id_bytes, 16 + 64 + 256);
        assert_eq!(cache.by_id_bytes, sum_bytes(&cache));

        // Replace id 2 with a bigger image via the helper: old bytes
        // come off through `sub_by_id_bytes`, new bytes go on.
        plant_front_entry(&mut cache, ImageId(2), entry(16, 16)); // 1024 B
        assert_eq!(cache.by_id_bytes, 16 + 1024 + 256);
        assert_eq!(cache.by_id_bytes, sum_bytes(&cache));

        // Remove id 1 the way `evict_over_budget` does: remove the entry
        // first, then subtract its exact length.
        let removed = cache.by_id.remove(&ImageId(1)).expect("planted");
        cache.sub_by_id_bytes(removed.byte_len);
        assert_eq!(cache.by_id_bytes, 1024 + 256);
        assert_eq!(cache.by_id_bytes, sum_bytes(&cache));

        // Drain everything through the helper: the counter lands back at
        // zero with no drift.
        let lens: Vec<usize> = cache.by_id.values().map(|e| e.byte_len).collect();
        for len in lens {
            cache.sub_by_id_bytes(len);
        }
        cache.by_id.clear();
        assert_eq!(cache.by_id_bytes, 0);
        assert_eq!(cache.by_id_bytes, sum_bytes(&cache));
    }

    /// A desynced counter (subtracting more than is pinned) trips the
    /// debug assertion instead of silently wrapping. Debug-only: in
    /// release the subtraction saturates to zero.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "by_id_bytes underflow")]
    fn by_id_bytes_underflow_trips_debug_assertion() {
        let mut cache = ImageCache {
            by_id_bytes: 10,
            ..Default::default()
        };
        cache.sub_by_id_bytes(11);
    }

    /// Oversubtraction saturates to zero instead of wrapping to
    /// `usize::MAX` in release. Release-only: mirrors
    /// `by_id_bytes_underflow_trips_debug_assertion`, which covers the
    /// debug half of the contract.
    #[test]
    #[cfg(not(debug_assertions))]
    fn by_id_bytes_oversubtraction_saturates_in_release() {
        let mut cache = ImageCache {
            by_id_bytes: 10,
            ..Default::default()
        };
        cache.sub_by_id_bytes(11);
        assert_eq!(cache.by_id_bytes, 0, "release saturates, does not wrap");
    }

    /// Same release contract for the shared counter: oversubtraction
    /// saturates to zero. Release-only: mirrors the debug-only
    /// `shared_bytes` coverage gap the same way.
    #[test]
    #[cfg(not(debug_assertions))]
    fn shared_bytes_oversubtraction_saturates_in_release() {
        let mut cache = ImageCache {
            shared_bytes: 10,
            ..Default::default()
        };
        cache.sub_shared_bytes(11);
        assert_eq!(cache.shared_bytes, 0, "release saturates, does not wrap");
    }

    /// The group pixmap pool hands the same-size buffer back across
    /// frames: one isolated opacity group per frame allocates once,
    /// then reuses.
    #[test]
    fn group_pixmap_pool_reuses_buffer_across_frames() {
        fn group_list(color: Color, x: f32) -> DisplayList {
            let mut list = DisplayList::new();
            list.push(DisplayItem::PushOpacity { opacity: 0.5 });
            // Two rects: forces the isolated group path (the single-item
            // fold fast path does not check out a pixmap).
            list.push(DisplayItem::FillRect {
                rect: rect(x, 10.0, 10.0, 10.0),
                color,
            });
            list.push(DisplayItem::FillRect {
                rect: rect(x + 20.0, 30.0, 10.0, 10.0),
                color,
            });
            list.push(DisplayItem::PopOpacity);
            list
        }
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        for _ in 0..3 {
            let list = group_list(red(), 10.0);
            backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        }
        let s = &backend.cache_stats;
        eprintln!(
            "[measure] group pixmap pool: checkouts={} reuses={}",
            s.pool_checkouts, s.pool_reuses
        );
        assert_eq!(s.pool_checkouts, 3);
        assert_eq!(s.pool_reuses, 2, "frames 2 and 3 reuse frame 1's buffer");
        assert_eq!(
            backend.group_pixmap_pool.len(),
            1,
            "one pooled buffer, not one per frame"
        );
    }

    /// A pooled group pixmap is cleared on checkout: stale pixels from
    /// a previous group's rendering must not leak into the next group.
    #[test]
    fn group_pixmap_pool_checkout_is_cleared() {
        fn group_list(color: Color, x: f32) -> DisplayList {
            let mut list = DisplayList::new();
            list.push(DisplayItem::PushOpacity { opacity: 0.5 });
            list.push(DisplayItem::FillRect {
                rect: rect(x, 10.0, 10.0, 10.0),
                color,
            });
            list.push(DisplayItem::FillRect {
                rect: rect(x + 20.0, 30.0, 10.0, 10.0),
                color,
            });
            list.push(DisplayItem::PopOpacity);
            list
        }
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        // Frame 1: group paints red at x=10.
        let rgba1 =
            backend.render_to_rgba_full(&group_list(red(), 10.0), 100, 100, 1.0, (0.0, 0.0));
        let [r, _, _, _] = rgba_at(&rgba1, 100, 15, 15);
        assert!(r > 100, "frame 1 group painted red");
        // Frame 2: group paints blue at x=60, reusing frame 1's pooled
        // pixmap. Stale red at (15,15) would composite if not cleared.
        let blue = Color::rgba8(0, 0, 255, 255);
        let rgba2 = backend.render_to_rgba_full(&group_list(blue, 60.0), 100, 100, 1.0, (0.0, 0.0));
        assert_eq!(
            rgba_at(&rgba2, 100, 15, 15),
            [0, 0, 0, 0],
            "no stale red leaked from the pooled pixmap"
        );
        let [r2, g2, b2, a2] = rgba_at(&rgba2, 100, 65, 15);
        assert_eq!((r2, g2), (0, 0));
        assert!(b2 > 100 && a2 > 100, "frame 2 group painted blue");
    }

    /// Shadow temps are checked out from the backend's scratch pool
    /// and returned after the shadow: the same shadow on a later frame
    /// reuses the buffers instead of allocating fresh ones.
    #[test]
    fn shadow_scratch_pool_reuses_temps_across_frames() {
        let list = shadow_list(false, 8.0, (0.0, 0.0));
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        for _ in 0..3 {
            backend.render_to_rgba_full(&list, 200, 200, 1.0, (0.0, 0.0));
        }
        let s = &backend.cache_stats;
        eprintln!(
            "[measure] shadow scratch pool: checkouts={} reuses={}",
            s.pool_checkouts, s.pool_reuses
        );
        // One box shadow = 3 temp checkouts (coverage, blur temp, sharp
        // mask) per frame; frames 2-3 reuse frame 1's buffers.
        assert_eq!(s.pool_checkouts, 9);
        assert_eq!(s.pool_reuses, 6, "frames 2 and 3 reuse frame 1's temps");
        assert_eq!(
            backend.shadow_scratch_pool.len(),
            3,
            "three temps pooled, not one per frame"
        );
    }

    /// Recycled shadow temps cannot leak stale pixels: poisoning every
    /// pooled buffer between renders must not change the output. The
    /// fixture covers all three pooled temps — the blurred outset
    /// shadow's coverage/blur/sharp buffers, an empty-shape inset
    /// shadow (its coverage is read without being rasterized, so it
    /// must come back zeroed), and a blurred text shadow's RGBA
    /// scratch.
    #[test]
    fn box_shadow_pooled_scratch_poison_is_pixel_identical() {
        let mut list = shadow_list(false, 8.0, (0.0, 0.0));
        // Empty-shape inset shadow: the spread deflates the shape below
        // zero, so the coverage temp is never rasterized — a recycled
        // buffer must still read as zeros.
        list.push(DisplayItem::BoxShadow {
            rect: rect(120.0, 120.0, 40.0, 40.0),
            radii: [0.0; 4],
            color: Color::rgba8(0, 0, 0, 255),
            offset: (0.0, 0.0),
            blur: 8.0,
            spread: -30.0,
            inset: true,
        });
        // Blurred text shadow: exercises the RGBA blur scratch path
        // (no font data, so no ink paints — the temp is still checked
        // out and returned).
        list.push(DisplayItem::TextShadow {
            text: Arc::from("hello"),
            text_range: 0..5,
            rect: rect(10.0, 150.0, 60.0, 20.0),
            baseline: 165.0,
            font_key: FontKey(0),
            size_px: 16.0,
            color: Color::rgba8(0, 0, 0, 255),
            offset: (2.0, 2.0),
            blur: 4.0,
        });
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let rgba1 = backend.render_to_rgba_full(&list, 200, 200, 1.0, (0.0, 0.0));
        // Poison every pooled buffer with bytes a fresh buffer could
        // never hold: any stale read shows up as changed pixels.
        assert!(
            !backend.shadow_scratch_pool.is_empty(),
            "shadow temps were returned to the pool"
        );
        for buf in &mut backend.shadow_scratch_pool {
            buf.fill(0xAB);
        }
        let rgba2 = backend.render_to_rgba_full(&list, 200, 200, 1.0, (0.0, 0.0));
        assert_eq!(
            rgba1, rgba2,
            "poisoned scratch must not change shadow pixels"
        );
    }

    #[test]
    fn decoration_underline_paints_bottom_of_ink_box() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::Decoration {
            rect: rect(10.0, 10.0, 80.0, 20.0),
            kind: DecoKind::Underline,
            color: red(),
            thickness: 2.0,
        });
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // Underline at ink-box bottom: y in [28, 30).
        assert_eq!(rgba_at(&rgba, 100, 50, 28), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 50, 29), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 50, 20), [0, 0, 0, 0]);
    }

    #[test]
    fn decoration_wavy_underline_paints_wave() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::Decoration {
            rect: rect(10.0, 10.0, 80.0, 20.0),
            kind: DecoKind::WavyUnderline,
            color: red(),
            thickness: 2.0,
        });
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // A straight underline would paint only rows 28..30. The wave is
        // centered on row 29 with amplitude 2 and period 8 starting at
        // phase zero at x=10, so it paints rows the straight line never
        // touches: row 31 near the peak at x=52, row 27 near the trough
        // at x=56, and the center row at phase zero (x=50).
        assert_eq!(rgba_at(&rgba, 100, 51, 31), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 55, 27), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 49, 29), [255, 0, 0, 255]);
        // Far above and below the wave's band nothing paints.
        assert_eq!(rgba_at(&rgba, 100, 50, 20), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 100, 50, 40), [0, 0, 0, 0]);
    }

    #[test]
    fn list_marker_disc_paints_center() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::ListMarker {
            rect: rect(10.0, 10.0, 8.0, 8.0),
            kind: MarkerKind::Disc,
            color: red(),
            index: 1,
            font_key: FontKey(0),
        });
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // Disc approximated as an 8x8 square centered in the rect.
        assert_eq!(rgba_at(&rgba, 100, 14, 14), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 50, 50), [0, 0, 0, 0]);
    }

    #[test]
    fn static_render_to_rgba_red_8x8() {
        // Task-mandated call-site signature: associated 3-arg function.
        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 8.0, 8.0),
            color: red(),
        });
        let rgba = CpuBackend::render_to_rgba(&list, 8, 8);
        assert_eq!(rgba.len(), 8 * 8 * 4);
        assert!(
            rgba.as_chunks::<4>()
                .0
                .iter()
                .all(|px| *px == [255, 0, 0, 255]),
            "every pixel must be solid red"
        );
    }

    #[test]
    fn text_run_without_font_data_paints_nothing() {
        // The static render path has no font data, so verbatim text runs
        // cannot be shaped: no panic, pixels untouched.
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::TextRun {
            text: Arc::from("hello"),
            text_range: 0..5,
            rect: rect(10.0, 10.0, 80.0, 20.0),
            baseline: 26.0,
            font_key: FontKey(0),
            size_px: 16.0,
            color: red(),
        });
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert!(rgba.iter().all(|&b| b == 0));
    }

    /// Regression: the thread-local font-data cache must not alias
    /// `FontKey` across `FontCache` instances. Keys are minted
    /// sequentially per cache, so cache A (sans-serif resolved first)
    /// and cache B (monospace resolved first) hand out the same key for
    /// different faces. Rendering with B on the same thread after A must
    /// re-consult B's hook instead of serving A's stale bytes.
    #[test]
    fn font_data_cache_does_not_alias_keys_across_caches() {
        use crate::fonts::{FontCache, FontStyle};
        use std::cell::Cell;

        let cache_a = FontCache::new();
        let key_a = cache_a
            .resolve("sans-serif", 400, FontStyle::Normal)
            .expect("sans-serif resolves");
        let cache_b = FontCache::new();
        let key_b = cache_b
            .resolve("monospace", 400, FontStyle::Normal)
            .expect("monospace resolves");
        // Swapped resolution orders collide on one FontKey: the premise
        // of the bug. Deterministic: each cache mints FontKey(0) for its
        // eager default face, then FontKey(1) for the first resolve.
        assert_eq!(
            key_a, key_b,
            "swapped resolution orders must collide on one FontKey"
        );

        let data_a = cache_a.font_data(key_a).expect("key_a resolves").clone();
        let data_b = cache_b.font_data(key_b).expect("key_b resolves").clone();

        // Counting hooks: the mechanism-level assertion is that B's hook
        // is actually consulted after A's primed the thread-local.
        let calls_a = Rc::new(Cell::new(0usize));
        let calls_b = Rc::new(Cell::new(0usize));
        let map_a = cache_a.snapshot();
        let hook_a: Rc<dyn Fn(FontKey) -> Option<FontData>> = {
            let calls = Rc::clone(&calls_a);
            Rc::new(move |key: FontKey| {
                calls.set(calls.get() + 1);
                map_a.get(&key).cloned()
            })
        };
        let map_b = cache_b.snapshot();
        let hook_b: Rc<dyn Fn(FontKey) -> Option<FontData>> = {
            let calls = Rc::clone(&calls_b);
            Rc::new(move |key: FontKey| {
                calls.set(calls.get() + 1);
                map_b.get(&key).cloned()
            })
        };

        let text_list = |key: FontKey| {
            let mut list = DisplayList::new();
            list.push(DisplayItem::TextRun {
                text: Arc::from("Hello"),
                text_range: 0..5,
                rect: rect(10.0, 10.0, 180.0, 40.0),
                baseline: 40.0,
                font_key: key,
                size_px: 32.0,
                color: red(),
            });
            list
        };

        // Prime the thread-local with cache A's bytes for the key.
        let mut backend_a = CpuBackend::new(hook_a);
        let px_a = backend_a.render_to_rgba_full(&text_list(key_a), 200, 60, 1.0, (0.0, 0.0));
        assert!(calls_a.get() > 0, "hook A must be consulted on first use");

        // Same thread, different cache, same FontKey: B's hook must be
        // re-consulted, not served A's stale bytes.
        let mut backend_b = CpuBackend::new(hook_b);
        let px_b = backend_b.render_to_rgba_full(&text_list(key_b), 200, 60, 1.0, (0.0, 0.0));
        assert!(
            calls_b.get() > 0,
            "hook B must be re-consulted after the hook changes"
        );

        // When the two faces actually differ, the pixels must differ too:
        // the aliased cache rendered monospace text with serif bytes.
        if data_a.bytes() != data_b.bytes() {
            assert_ne!(
                px_a, px_b,
                "same FontKey from another cache rendered with stale bytes"
            );
        }
    }

    /// Backend with real font data, or `None` when the system has no
    /// font (fontless CI): tests skip gracefully.
    fn font_backend() -> Option<CpuBackend> {
        let cache = crate::fonts::FontCache::new();
        let font_key = cache.default_key()?;
        let data = {
            let fd = cache.font_data(font_key).expect("default key resolves");
            fd.clone()
        };
        Some(CpuBackend::new(Rc::new(move |_| Some(data.clone()))))
    }

    #[test]
    fn marker_numeral_comes_from_index() {
        // Agreement with the GPU backend: the numeral is the 1-based
        // index, formatted per kind; the kind's payload is ignored.
        assert_eq!(
            Painter::marker_numeral(MarkerKind::Decimal(7), 3),
            Some("3".to_string())
        );
        assert_eq!(
            Painter::marker_numeral(MarkerKind::DecimalLeadingZero(9), 3),
            Some("03".to_string())
        );
        assert_eq!(Painter::marker_numeral(MarkerKind::Disc, 3), None);
    }

    #[test]
    fn decimal_marker_paints_numeral_text() {
        // Decimal markers render as text through the text path (like the
        // GPU backend), not as geometry.
        let Some(mut backend) = font_backend() else {
            // No system font here: skip gracefully.
            return;
        };
        let mut list = DisplayList::new();
        list.push(DisplayItem::ListMarker {
            rect: rect(10.0, 10.0, 8.0, 8.0),
            kind: MarkerKind::Decimal(2),
            color: red(),
            index: 2,
            font_key: FontKey(0),
        });
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        let painted: Vec<[u8; 4]> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[3] > 0)
            .map(|px| [px[0], px[1], px[2], px[3]])
            .collect();
        assert!(!painted.is_empty(), "numeral must paint pixels");
        for px in &painted {
            assert_eq!([px[0], px[1], px[2]], [255, 0, 0], "numeral tinted red");
        }
    }

    #[test]
    fn decimal_leading_zero_marker_paints_numeral_text() {
        let Some(mut backend) = font_backend() else {
            // No system font here: skip gracefully.
            return;
        };
        let mut list = DisplayList::new();
        list.push(DisplayItem::ListMarker {
            rect: rect(10.0, 10.0, 8.0, 8.0),
            kind: MarkerKind::DecimalLeadingZero(3),
            color: red(),
            index: 3,
            font_key: FontKey(0),
        });
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert!(
            rgba.as_chunks::<4>().0.iter().any(|px| px[3] > 0),
            "\"03\" must paint pixels"
        );
    }

    #[test]
    fn decimal_marker_without_font_data_paints_nothing() {
        // No fallback face, no text: like TextRun, paints nothing.
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::ListMarker {
            rect: rect(10.0, 10.0, 8.0, 8.0),
            kind: MarkerKind::Decimal(2),
            color: red(),
            index: 2,
            font_key: FontKey(0),
        });
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert!(rgba.iter().all(|&b| b == 0));
    }

    #[test]
    fn glyph_atlas_persists_across_frames() {
        let Some(mut backend) = font_backend() else {
            // No system font here: skip gracefully.
            return;
        };
        let mut list = DisplayList::new();
        list.push(DisplayItem::TextRun {
            text: Arc::from("Ag"),
            text_range: 0..2,
            rect: rect(10.0, 10.0, 80.0, 20.0),
            baseline: 26.0,
            font_key: crate::display_list::FontKey(0),
            size_px: 16.0,
            color: red(),
        });
        backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        let misses = backend.glyph_misses;
        assert!(misses > 0, "first frame must miss the empty atlas");
        // Second frame with the same glyphs: no new misses, so the atlas
        // survived instead of being rebuilt per frame.
        backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert_eq!(backend.glyph_misses, misses);
    }

    #[test]
    fn glyph_atlas_evicts_past_cap() {
        let Some(mut backend) = font_backend() else {
            // No system font here: skip gracefully.
            return;
        };
        let mut list = DisplayList::new();
        list.push(DisplayItem::TextRun {
            text: Arc::from("z"),
            text_range: 0..1,
            rect: rect(10.0, 10.0, 80.0, 20.0),
            baseline: 26.0,
            font_key: crate::display_list::FontKey(0),
            size_px: 16.0,
            color: red(),
        });
        backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert_eq!(backend.glyph_misses, 1, "one miss for the fresh glyph");
        // Simulate churn past the cap; the next frame evicts.
        backend.glyph_misses = GLYPH_ATLAS_CAP + 1;
        backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert_eq!(backend.glyph_misses, 0, "eviction resets the counter");
        // The atlas itself was dropped: the same glyph misses again.
        backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert_eq!(backend.glyph_misses, 1, "evicted glyph re-rasterizes");
    }

    /// `glyph_misses` counts only successful rasterizations: glyphs that
    /// fail to rasterize (spaces, broken outlines) insert nothing and must
    /// not churn the eviction counter frame after frame.
    #[test]
    fn glyph_miss_counts_only_successful_rasterizations() {
        let Some(mut backend) = font_backend() else {
            // No system font here: skip gracefully.
            return;
        };
        let mut list = DisplayList::new();
        list.push(DisplayItem::TextRun {
            text: Arc::from("A A"),
            text_range: 0..3,
            rect: rect(10.0, 10.0, 80.0, 20.0),
            baseline: 26.0,
            font_key: crate::display_list::FontKey(0),
            size_px: 16.0,
            color: red(),
        });
        backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        let after_first = backend.glyph_misses;
        assert!(after_first >= 1, "the rasterizable glyph counts");
        backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert_eq!(
            backend.glyph_misses, after_first,
            "unrasterizable glyphs must not miss again"
        );
    }

    /// A hook that re-enters the font-data cache mid-resolve (as a
    /// text-measuring hook would) must not panic on the `RefCell` borrow:
    /// the two-phase lookup runs the hook outside the borrow. Under the
    /// old single-phase `resolve` this test panics with "already
    /// borrowed".
    #[test]
    fn font_data_cache_hook_reentrancy() {
        type Hook = Rc<dyn Fn(FontKey) -> Option<FontData>>;
        thread_local! {
            static CACHE: std::cell::RefCell<FontDataCache> =
                std::cell::RefCell::new(FontDataCache::new());
        }
        // Mirrors the `paint_text_run` lookup: `get` under the borrow,
        // hook outside it, `insert` under a fresh borrow.
        fn two_phase(hook: &Hook, key: FontKey) -> Option<Rc<FontData>> {
            CACHE
                .with(|cache| cache.borrow_mut().get(hook, key))
                .or_else(|| {
                    let data = hook(key)?;
                    Some(CACHE.with(|cache| cache.borrow_mut().insert(hook, key, data)))
                })
        }

        let cache = crate::fonts::FontCache::new();
        let Some(font_key) = cache.default_key() else {
            // No system font in this environment: skip gracefully.
            return;
        };
        let data: FontData = cache
            .font_data(font_key)
            .expect("default key resolves")
            .clone();

        // `slot` hands the hook a reference to itself so it can re-enter
        // the cache the way nested text painting would.
        let slot: Rc<std::cell::RefCell<Option<Hook>>> = Rc::new(std::cell::RefCell::new(None));
        let hook: Hook = {
            let slot = Rc::clone(&slot);
            Rc::new(move |key| {
                // Nested resolve of a different key while the outer
                // resolve's hook is still running. Guarded to FontKey(7)
                // so the nesting terminates.
                if key != FontKey(7) {
                    let reenter = slot.borrow().as_ref().map(Rc::clone);
                    if let Some(self_hook) = reenter {
                        let _ = two_phase(&self_hook, FontKey(7));
                    }
                }
                Some(data.clone())
            })
        };
        *slot.borrow_mut() = Some(Rc::clone(&hook));

        let resolved = two_phase(&hook, FontKey(0));
        assert!(resolved.is_some(), "outer resolve must succeed");
        // The nested insert for FontKey(7) landed under the same hook.
        let nested = CACHE.with(|cache| cache.borrow_mut().get(&hook, FontKey(7)));
        assert!(nested.is_some(), "nested resolve must have inserted");
    }
    #[test]
    fn interim_shaping_paints_text_with_font_data() {
        // Interim simple shaping (see `paint_text_run` docs): with real
        // font data, a text run stamps non-zero coverage pixels.
        let cache = crate::fonts::FontCache::new();
        let Some(font_key) = cache.default_key() else {
            // No system font in this environment: skip gracefully.
            return;
        };
        let data = {
            let fd = cache.font_data(font_key).expect("default key resolves");
            fd.clone()
        };
        let mut backend = CpuBackend::new(Rc::new(move |_| Some(data.clone())));
        let mut list = DisplayList::new();
        list.push(DisplayItem::TextRun {
            text: Arc::from("Ag"),
            text_range: 0..2,
            rect: rect(10.0, 10.0, 80.0, 20.0),
            baseline: 26.0,
            font_key,
            size_px: 16.0,
            color: red(),
        });
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert_eq!(rgba.len(), 100 * 100 * 4);
        let painted = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[3] > 0)
            .count();
        assert!(painted > 0, "glyph run must paint some pixels");
        // Painted pixels are red (unpremultiplied straight RGBA).
        for px in rgba.as_chunks::<4>().0.iter().filter(|px| px[3] > 0) {
            assert_eq!([px[0], px[1], px[2]], [255, 0, 0], "glyph tinted red");
        }
    }

    #[test]
    fn backend_trait_paint_stores_frame() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = red_rect_list();
        Backend::paint(&mut backend, &list, Size { w: 100.0, h: 100.0 }, (0.0, 0.0));
        let (rgba, w, h) = backend.last_frame().expect("frame stored");
        assert_eq!((w, h), (100, 100));
        assert_eq!(rgba_at(rgba, 100, 50, 50), [255, 0, 0, 255]);
        assert_eq!(rgba_at(rgba, 100, 5, 5), [0, 0, 0, 0]);
    }

    #[test]
    fn present_packs_xrgb() {
        let backend = CpuBackend::new(Rc::new(|_| None));
        // Straight RGBA in, 0x00RRGGBB out (alpha dropped).
        let rgba = vec![255u8, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0];
        let mut buffer = vec![0u32; 3];
        backend.present(&mut buffer, &rgba, 3, 1);
        assert_eq!(buffer, vec![0x00FF0000, 0x0000FF00, 0x000000FF]);
    }

    #[test]
    fn zero_size_renders_empty() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = red_rect_list();
        assert!(
            backend
                .render_to_rgba_full(&list, 0, 100, 1.0, (0.0, 0.0))
                .is_empty()
        );
        assert!(
            backend
                .render_to_rgba_full(&list, 100, 0, 1.0, (0.0, 0.0))
                .is_empty()
        );
    }

    /// Regression: `draw_image` must build its transform from the
    /// *unclipped* device rect and let the clip mask do the clipping.
    /// A partial clip used to shrink the destination rect, squishing the
    /// whole image into the visible intersection.
    fn clipped_image_list() -> DisplayList {
        let mut list = DisplayList::new();
        // 2x2 straight RGBA: red, green / blue, white.
        list.add_image(
            ImageId(7),
            crate::display_list::DecodedImage::from_rgba(
                2,
                2,
                vec![
                    255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
                ],
            ),
        );
        list
    }

    #[test]
    fn image_partial_clip_shows_sub_region_not_squished() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = clipped_image_list();
        // Clip away the right half of the 2x2 image at (10, 10): the
        // visible column must still be red over blue, not a bilinear
        // blend of red+green / blue+white.
        list.push(DisplayItem::PushClip {
            rect: rect(10.0, 10.0, 1.0, 2.0),
        });
        list.push(DisplayItem::Image {
            rect: rect(10.0, 10.0, 2.0, 2.0),
            image: ImageId(7),
        });
        list.push(DisplayItem::PopClip);
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert_eq!(rgba_at(&rgba, 100, 10, 10), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 10, 11), [0, 0, 255, 255]);
        // Clipped-out column stays transparent.
        assert_eq!(rgba_at(&rgba, 100, 11, 10), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&rgba, 100, 11, 11), [0, 0, 0, 0]);
    }

    #[test]
    fn image_fully_clipped_paints_nothing() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = clipped_image_list();
        list.push(DisplayItem::PushClip {
            rect: rect(50.0, 50.0, 10.0, 10.0),
        });
        list.push(DisplayItem::Image {
            rect: rect(10.0, 10.0, 2.0, 2.0),
            image: ImageId(7),
        });
        list.push(DisplayItem::PopClip);
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert!(rgba.iter().all(|&b| b == 0));
    }

    #[test]
    fn image_degenerate_rect_paints_nothing() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = clipped_image_list();
        list.push(DisplayItem::Image {
            rect: rect(10.0, 10.0, 0.0, 2.0),
            image: ImageId(7),
        });
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert!(rgba.iter().all(|&b| b == 0));
    }

    /// Manual benchmark harness (std-only, no new dependencies).
    ///
    /// Runs only when `BOWSER_BENCH=1` is set so `cargo test` stays fast:
    /// `BOWSER_BENCH=1 cargo test -p bowser_paint bench_cpu -- --nocapture`
    #[test]
    fn bench_cpu() {
        if std::env::var("BOWSER_BENCH").is_err() {
            return;
        }
        // Deterministic LCG so before/after runs paint the same scene.
        fn rng(state: &mut u64) -> f32 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 33) as f32) / (u32::MAX as f32)
        }

        fn bench(name: &str, iters: usize, mut f: impl FnMut()) {
            for _ in 0..2 {
                f(); // warmup
            }
            let mut times: Vec<u128> = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = std::time::Instant::now();
                f();
                times.push(t.elapsed().as_micros());
            }
            times.sort_unstable();
            let median = times[iters / 2] as f64 / 1000.0;
            eprintln!("bench {name}: {median:.2} ms/iter (median of {iters})");
        }

        const W: u32 = 1280;
        const H: u32 = 800;

        // A: many solid fill rects (fill_rect + unpremultiply bound).
        let mut list = DisplayList::new();
        let mut s = 0x12345678u64;
        for _ in 0..3000 {
            let (x, y) = (rng(&mut s) * 1200.0, rng(&mut s) * 740.0);
            let (w, h) = (4.0 + rng(&mut s) * 120.0, 4.0 + rng(&mut s) * 90.0);
            list.push(DisplayItem::FillRect {
                rect: rect(x, y, w, h),
                color: Color::from_f32(rng(&mut s), rng(&mut s), rng(&mut s), 1.0),
            });
        }
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        bench("fill_rects_3k", 7, || {
            backend.render_to_rgba_full(&list, W, H, 1.0, (0.0, 0.0));
        });

        // B: clip push/pop churn with rects inside (mask rebuild bound).
        let mut list = DisplayList::new();
        let mut s = 0xabcdef01u64;
        for _ in 0..300 {
            let (x, y) = (rng(&mut s) * 1000.0, rng(&mut s) * 600.0);
            list.push(DisplayItem::PushClip {
                rect: rect(x, y, 200.0, 150.0),
            });
            for _ in 0..10 {
                list.push(DisplayItem::FillRect {
                    rect: rect(x - 50.0, y - 50.0, 300.0, 250.0),
                    color: red(),
                });
            }
            list.push(DisplayItem::PopClip);
        }
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        bench("clip_churn", 7, || {
            backend.render_to_rgba_full(&list, W, H, 1.0, (0.0, 0.0));
        });

        // C: translucent overdraw (blending + unpremultiply with partial alpha).
        let mut list = DisplayList::new();
        let mut s = 0x55555555u64;
        for _ in 0..1200 {
            list.push(DisplayItem::FillRect {
                rect: rect(rng(&mut s) * 1100.0, rng(&mut s) * 650.0, 180.0, 140.0),
                color: Color::from_f32(rng(&mut s), rng(&mut s), rng(&mut s), 0.35),
            });
        }
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        bench("alpha_overdraw_1200", 7, || {
            backend.render_to_rgba_full(&list, W, H, 1.0, (0.0, 0.0));
        });

        // D: text runs (glyph stamping bound); skipped when no system font.
        if let Some(key) = crate::fonts::FontCache::new().default_key() {
            let cache = crate::fonts::FontCache::new();
            let data = cache.font_data(key).expect("default key resolves").clone();
            let mut backend = CpuBackend::new(Rc::new(move |_| Some(data.clone())));
            let mut list = DisplayList::new();
            let words = "the quick brown fox jumps over the lazy dog 0123456789 ";
            for row in 0..150 {
                list.push(DisplayItem::TextRun {
                    text: Arc::from(words.repeat(3)),
                    text_range: 0..words.len() * 3,
                    rect: rect(10.0, 10.0 + row as f32 * 5.0, 1000.0, 20.0),
                    baseline: 24.0 + row as f32 * 5.0,
                    font_key: key,
                    size_px: 16.0,
                    color: red(),
                });
            }
            bench("text_runs_150", 7, || {
                backend.render_to_rgba_full(&list, W, H, 1.0, (0.0, 0.0));
            });
        } else {
            eprintln!("bench text_runs_150: skipped (no system font)");
        }

        // E: same as A at 2x scale (pixel-count scaling of fill+unpremultiply).
        let mut list = DisplayList::new();
        let mut s = 0x12345678u64;
        for _ in 0..3000 {
            let (x, y) = (rng(&mut s) * 1200.0, rng(&mut s) * 740.0);
            let (w, h) = (4.0 + rng(&mut s) * 120.0, 4.0 + rng(&mut s) * 90.0);
            list.push(DisplayItem::FillRect {
                rect: rect(x, y, w, h),
                color: Color::from_f32(rng(&mut s), rng(&mut s), rng(&mut s), 1.0),
            });
        }
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        bench("fill_rects_3k_2x", 7, || {
            backend.render_to_rgba_full(&list, W * 2, H * 2, 2.0, (0.0, 0.0));
        });

        // F: clip-mask update cost — the incremental region diff
        // (`paint_region_diff`) versus the old full-rebuild algorithm
        // (reconstructed from git history as the baseline; byte-identical
        // output, O(surface) work per op), over the same deterministic
        // 2000-op PushClip/PopClip-style region sequence on a 1280x800
        // mask. Both variants end with identical bytes; only the write
        // volume differs.
        {
            fn full_rebuild(data: &mut [u8], w: usize, region: Option<MaskRect>) {
                data.fill(0);
                if let Some(r) = region {
                    for y in r.y0..r.y1 {
                        data[y * w + r.x0..y * w + r.x1].fill(255);
                    }
                }
            }
            const BW: usize = 1280;
            const BH: usize = 800;
            let mut s = 0xC11FFEE5u64;
            let mut next_region = || {
                // xorshift64*.
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                s = s.wrapping_mul(0x2545F4914F6CDD1D);
                let n = (s >> 32) as usize;
                // 1 in 8: the empty (all-zero) region.
                if n.is_multiple_of(8) {
                    return None;
                }
                let mut coords = [0usize; 4];
                for c in &mut coords {
                    s ^= s >> 12;
                    s ^= s << 25;
                    s ^= s >> 27;
                    s = s.wrapping_mul(0x2545F4914F6CDD1D);
                    *c = (s >> 32) as usize;
                }
                let (x0, x1, y0, y1) = (
                    (coords[0] % (BW + 40)).min(BW),
                    (coords[1] % (BW + 40)).min(BW),
                    (coords[2] % (BH + 40)).min(BH),
                    (coords[3] % (BH + 40)).min(BH),
                );
                let (x0, x1) = (x0.min(x1), x0.max(x1));
                let (y0, y1) = (y0.min(y1), y0.max(y1));
                if x0 >= x1 || y0 >= y1 {
                    return None;
                }
                Some(MaskRect { x0, y0, x1, y1 })
            };
            // Fix the transition sequence once so both variants perform
            // identical logical work.
            let mut seq: Vec<(Option<MaskRect>, Option<MaskRect>)> = Vec::with_capacity(2000);
            let mut cur: Option<MaskRect> = None;
            for _ in 0..2000 {
                let nxt = next_region();
                seq.push((cur, nxt));
                cur = nxt;
            }
            let mut data = vec![0u8; BW * BH];
            bench("mask_update_incremental", 5, || {
                for &(old, new) in &seq {
                    paint_region_diff(&mut data, BW, old, new, 0);
                    paint_region_diff(&mut data, BW, new, old, 255);
                }
                std::hint::black_box(&data);
            });
            bench("mask_update_full_rebuild", 5, || {
                for &(_, new) in &seq {
                    full_rebuild(&mut data, BW, new);
                }
                std::hint::black_box(&data);
            });
        }
    }

    /// `precompute_group_scans` must agree with `scan_opacity_group` at
    /// every `PushOpacity`, including nesting, clips, unbalanced input,
    /// and stray pops.
    #[test]
    fn group_precompute_matches_scan() {
        let mut list = DisplayList::new();
        // 0: group with two rects.
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 1.0, 1.0),
            color: red(),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(1.0, 1.0, 1.0, 1.0),
            color: red(),
        });
        list.push(DisplayItem::PopOpacity);
        // 4: nested groups with a clip inside.
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::PushClip {
            rect: rect(0.0, 0.0, 4.0, 4.0),
        });
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 1.0, 1.0),
            color: red(),
        });
        list.push(DisplayItem::PopOpacity);
        list.push(DisplayItem::PopClip);
        list.push(DisplayItem::PopOpacity);
        // 11: stray pop (ignored), then an unclosed group.
        list.push(DisplayItem::PopOpacity);
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 1.0, 1.0),
            color: red(),
        });
        assert_eq!(list.items.len(), 14);

        let scans = precompute_group_scans(&list.items);
        let pushes: Vec<usize> = list
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| matches!(item, DisplayItem::PushOpacity { .. }))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(pushes, vec![0, 4, 6, 12]);
        for idx in pushes {
            assert_eq!(
                scans.get(&idx).copied(),
                Some(scan_opacity_group(&list.items, idx)),
                "precompute/scan mismatch at PushOpacity {idx}"
            );
        }
        // Spot-check the interesting tuples: (end, paint_items, nested,
        // single_item). Clips don't count; the stray pop is ignored; the
        // unclosed group runs to the end.
        assert_eq!(scans[&0], (3, 2, false, None));
        assert_eq!(scans[&4], (10, 0, true, None));
        assert_eq!(scans[&6], (8, 1, false, Some(7)));
        assert_eq!(scans[&12], (14, 1, false, Some(13)));
    }

    /// Pathological nesting (1000 isolated groups) must terminate
    /// instead of overflowing the stack: past `MAX_OPACITY_GROUPS` the
    /// groups fold instead of recursing into another offscreen pixmap.
    #[test]
    fn deeply_nested_opacity_groups_terminate() {
        let mut list = DisplayList::new();
        for _ in 0..1000 {
            list.push(DisplayItem::PushOpacity { opacity: 0.5 });
            // Two rects per level: every level is isolated (never the
            // single-item fold fast path), so without the cap this
            // recurses 1000 deep in `paint_opacity_group`.
            list.push(DisplayItem::FillRect {
                rect: rect(0.0, 0.0, 1.0, 1.0),
                color: red(),
            });
            list.push(DisplayItem::FillRect {
                rect: rect(2.0, 2.0, 1.0, 1.0),
                color: red(),
            });
        }
        for _ in 0..1000 {
            list.push(DisplayItem::PopOpacity);
        }
        let rgba = CpuBackend::render_to_rgba(&list, 4, 4);
        assert_eq!(rgba.len(), 4 * 4 * 4);
    }

    // -- RepeatImage --------------------------------------------------------

    /// 4x4 test tile with distinct pixels per quadrant: TL red, TR
    /// green, BL blue, BR white.
    fn checker_tile() -> DecodedImage {
        let mut px = vec![0u8; 4 * 4 * 4];
        for y in 0..4 {
            for x in 0..4 {
                let c = match (x < 2, y < 2) {
                    (true, true) => [255, 0, 0, 255],
                    (false, true) => [0, 255, 0, 255],
                    (true, false) => [0, 0, 255, 255],
                    (false, false) => [255, 255, 255, 255],
                };
                let o = (y * 4 + x) * 4;
                px[o..o + 4].copy_from_slice(&c);
            }
        }
        DecodedImage::from_rgba(4, 4, px)
    }

    /// The old per-tile expansion of a tiled background (what
    /// `push_background_image_tiled` used to emit): `PushClip` + one
    /// `BackgroundImage` per tile + `PopClip`.
    fn old_tiled_list(
        area_: Rect,
        id: ImageId,
        decoded: &DecodedImage,
        tile: (f32, f32),
        repeat: BackgroundRepeat,
        offset: (f32, f32),
    ) -> DisplayList {
        let mut list = DisplayList::new();
        list.add_image(id, decoded.clone());
        list.push(DisplayItem::PushClip { rect: area_ });
        if repeat == BackgroundRepeat::Space {
            let (nx, x0, px_) = background_space_axis(area_.x, area_.w, tile.0, offset.0);
            let (ny, y0, py) = background_space_axis(area_.y, area_.h, tile.1, offset.1);
            for j in 0..ny {
                for i in 0..nx {
                    list.push(DisplayItem::BackgroundImage {
                        rect: Rect {
                            x: x0 + i as f32 * px_,
                            y: y0 + j as f32 * py,
                            w: tile.0,
                            h: tile.1,
                        },
                        image: id,
                    });
                }
            }
        } else {
            let (rx, ry) = match repeat {
                // `round` rescales the tile before tiling like `repeat`;
                // the rescaled tile arrives precomputed.
                BackgroundRepeat::Repeat | BackgroundRepeat::Round => (true, true),
                BackgroundRepeat::RepeatX => (true, false),
                BackgroundRepeat::RepeatY => (false, true),
                _ => (false, false),
            };
            let (x0, y0) = (area_.x + offset.0, area_.y + offset.1);
            let (i0, j0, nx, ny) = background_tile_grid(area_, (x0, y0), tile, rx, ry);
            for j in j0..j0 + ny {
                for i in i0..i0 + nx {
                    list.push(DisplayItem::BackgroundImage {
                        rect: Rect {
                            x: x0 + i as f32 * tile.0,
                            y: y0 + j as f32 * tile.1,
                            w: tile.0,
                            h: tile.1,
                        },
                        image: id,
                    });
                }
            }
        }
        list.push(DisplayItem::PopClip);
        list
    }

    /// The new `RepeatImage` primitive renders pixel-identically to the
    /// old per-tile expansion on a small case (fractional offsets
    /// exercise the grid phase on both paths).
    #[test]
    fn repeat_image_matches_old_tile_list_pixels() {
        use crate::display_list::{DisplayListBuilder, round_tile_len};
        let decoded = checker_tile();
        let area_ = rect(0.0, 0.0, 100.0, 60.0);
        // (repeat mode, tile): `round` rescales the tile before tiling,
        // like the old builder did.
        let cases = [
            (BackgroundRepeat::Repeat, (30.0, 30.0)),
            (BackgroundRepeat::RepeatX, (30.0, 30.0)),
            (BackgroundRepeat::RepeatY, (30.0, 30.0)),
            (BackgroundRepeat::NoRepeat, (30.0, 30.0)),
            (BackgroundRepeat::Space, (30.0, 30.0)),
            (
                BackgroundRepeat::Round,
                (round_tile_len(area_.w, 30.0), round_tile_len(area_.h, 30.0)),
            ),
        ];
        for (repeat, tile) in cases {
            let mut b = DisplayListBuilder::new();
            let id = b.add_image(decoded.clone());
            b.push_background_image_tiled(area_, id, (30.0, 30.0), repeat, (10.0, 7.0));
            let new_list = b.build();
            assert_eq!(
                new_list.items.len(),
                1,
                "one RepeatImage item for {repeat:?}"
            );
            let old_list = old_tiled_list(area_, id, &decoded, tile, repeat, (10.0, 7.0));
            let new_px = CpuBackend::render_to_rgba(&new_list, 100, 60);
            let old_px = CpuBackend::render_to_rgba(&old_list, 100, 60);
            assert_eq!(new_px, old_px, "pixel mismatch for {repeat:?}");
        }
    }

    /// f32 replica of the WGSL `repeat_axis` wrap branch (see
    /// `SHADER_IMAGE` in `wgpu_be.rs`): wraps `t` (tile units) to the
    /// sampled texture coordinate, snapping the interpolation-residue
    /// band on both sides of the next tile boundary to 1.0 — the CPU
    /// backend's left-tile convention (a boundary fragment samples the
    /// left tile's last texel, which clamp-to-edge reads at full
    /// weight).
    fn gpu_wrap_axis(t: f32, band: f32) -> f32 {
        let u = t - t.floor(); // WGSL fract(): x - floor(x)
        if u < band || u > 1.0 - band { 1.0 } else { u }
    }

    /// CPU/GPU pixel agreement at repeat tile boundaries, without a
    /// GPU: the CPU backend's boundary convention is pinned (a pixel
    /// center exactly on a tile boundary is painted by the tile on its
    /// left, sampling its right edge — the last texel column, white
    /// here), and the GPU fragment math — the f32 affine interpolation
    /// of the tile-unit vertex UVs plus the `repeat_axis` wrap branch
    /// with [`repeat_snap_band`](crate::display_list::repeat_snap_band)
    /// — is replicated in f32 and must canonicalize every boundary
    /// pixel to that same last texel.
    ///
    /// The offset battery is the point: `1000.5` (and `100000.5`) keep
    /// the tile phase identical to `10.5` (`1000.5 mod 30 == 10.5`), so
    /// the CPU renders pixel-identically, but push the tile-unit UV
    /// magnitude far past the span. The pre-fix span-based snap band
    /// (`4*eps*span`) is narrower than the interpolation residue there,
    /// so boundary fragments escaped the snap and the GPU sampled the
    /// next tile's first texel (red) instead of the last (white); the
    /// magnitude-based band closes that hole. The test asserts the old
    /// formula would have failed, proving it guards the fix.
    #[test]
    fn repeat_tile_boundary_agrees_with_gpu_wrap_math() {
        use crate::display_list::repeat_snap_band;
        let area = rect(0.0, 0.0, 100.0, 60.0);
        let tile = (30.0f32, 30.0f32);
        // (x offset, y offset); each is congruent mod 30 to (10.5, 7.5).
        let offsets = [
            (10.5f32, 7.5f32),
            (1000.5, 7.5),
            (100000.5, 7.5),
            (10.5, 997.5),
            (100000.5, 997.5),
        ];
        // CPU: identical tile phase => pixel-identical renders, and the
        // boundary columns/rows sample the last texel (white).
        let mut reference: Option<Vec<u8>> = None;
        for (ox, oy) in offsets {
            let mut list = DisplayList::new();
            let id = ImageId(1);
            list.add_image(id, checker_tile());
            list.push(DisplayItem::RepeatImage {
                area,
                image: id,
                tile,
                repeat: BackgroundRepeat::Repeat,
                offset: (ox, oy),
            });
            let px = CpuBackend::render_to_rgba(&list, 100, 60);
            match &reference {
                None => reference = Some(px),
                Some(r) => {
                    assert_eq!(&px, r, "CPU render changed with offset ({ox}, {oy})")
                }
            }
        }
        let px = reference.unwrap();
        // Tile edges at ox + k*30 = {10.5, 40.5, 70.5} (mod-30 phase);
        // boundary pixels sample the left tile's right edge: image
        // column 3. Row 0 maps to image row 3 (see checker_tile).
        for col in [10u32, 40, 70] {
            assert_eq!(
                rgba_at(&px, 100, col, 0),
                [255, 255, 255, 255],
                "boundary column {col} must sample the left tile's last texel"
            );
        }
        // y edges at oy + j*30 = {7.5, 37.5}: the tile above's bottom
        // edge (image row 3, white for x in 2..4).
        for row in [7u32, 37] {
            assert_eq!(
                rgba_at(&px, 100, 0, row),
                [255, 255, 255, 255],
                "boundary row {row} must sample the tile above's last texel row"
            );
        }

        // GPU fragment math: for each offset, the wrap branch must map
        // every boundary pixel center to the last texel (1.0 exactly
        // when snapped, else within the residue band below 1.0 — still
        // the last texel for this 4-texel tile since band <= 0.25).
        let mut old_band_would_fail = false;
        for (ox, oy) in offsets {
            for (o, a0, len, tdim, bmax) in [
                (ox, area.x, area.w, tile.0, 100.0f32),
                (oy, area.y, area.h, tile.1, 60.0f32),
            ] {
                // Exactly the `uv_span` computation in
                // `lower_repeat_image`.
                let x0 = a0 + o;
                let u0 = (a0 - x0) / tdim;
                let u1 = (a0 + len - x0) / tdim;
                let band = repeat_snap_band(u0, u1);
                // Pre-fix span-based band, kept to prove this test
                // catches the regression (see below).
                let old_band = (4.0 * f32::EPSILON * (u1 - u0).abs().max(1.0)).min(0.25);
                let mut k = ((0.0 - x0) / tdim).ceil() as i32 - 1;
                loop {
                    k += 1;
                    let bpos = x0 + k as f32 * tdim;
                    if bpos >= bmax {
                        break;
                    }
                    // Boundary pixel centers are exact half-integers by
                    // construction (offsets are N + 0.5, tile is 30).
                    assert_eq!(
                        (bpos - 0.5).fract(),
                        0.0,
                        "boundary {bpos} not on a pixel center"
                    );
                    // f32 affine interpolation of the vertex UVs, in the
                    // two evaluation orders a rasterizer may use.
                    let s = (bpos - a0) / len;
                    for t in [u0 + s * (u1 - u0), u0 * (1.0 - s) + u1 * s] {
                        let w = gpu_wrap_axis(t, band);
                        assert!(
                            w == 1.0 || (1.0 - w) <= band,
                            "offset ({ox}, {oy}): boundary {bpos} wraps to {w}, not the last texel"
                        );
                        // Regression pin: the old span-based band lets
                        // this boundary escape into the next tile.
                        let u = t - t.floor();
                        let w_old = if u < old_band { 1.0 } else { u };
                        if w_old != 1.0 && (1.0 - w_old) > old_band {
                            old_band_would_fail = true;
                        }
                    }
                }
            }
        }
        assert!(
            old_band_would_fail,
            "expected the pre-fix span-based band to mis-snap at least one boundary pixel"
        );
    }

    /// `repeat-x` paints one row of tiles at the vertical position
    /// offset and nothing elsewhere.
    #[test]
    fn repeat_x_paints_single_row() {
        let mut list = DisplayList::new();
        let id = ImageId(1);
        list.add_image(id, checker_tile());
        list.push(DisplayItem::RepeatImage {
            area: rect(0.0, 0.0, 12.0, 12.0),
            image: id,
            tile: (4.0, 4.0),
            repeat: BackgroundRepeat::RepeatX,
            offset: (0.0, 4.0),
        });
        let px = CpuBackend::render_to_rgba(&list, 12, 12);
        // Three tiles across the row y=4..8.
        assert_eq!(rgba_at(&px, 12, 0, 4), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&px, 12, 4, 4), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&px, 12, 8, 4), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&px, 12, 2, 6), [255, 255, 255, 255]);
        // Outside the row: untouched.
        assert_eq!(rgba_at(&px, 12, 6, 0), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 12, 6, 3), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 12, 6, 8), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 12, 6, 11), [0, 0, 0, 0]);
    }

    /// `repeat-y` paints one column of tiles at the horizontal position
    /// offset and nothing elsewhere.
    #[test]
    fn repeat_y_paints_single_column() {
        let mut list = DisplayList::new();
        let id = ImageId(1);
        list.add_image(id, checker_tile());
        list.push(DisplayItem::RepeatImage {
            area: rect(0.0, 0.0, 12.0, 12.0),
            image: id,
            tile: (4.0, 4.0),
            repeat: BackgroundRepeat::RepeatY,
            offset: (4.0, 0.0),
        });
        let px = CpuBackend::render_to_rgba(&list, 12, 12);
        // Three tiles down the column x=4..8.
        assert_eq!(rgba_at(&px, 12, 4, 0), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&px, 12, 4, 4), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&px, 12, 4, 8), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&px, 12, 6, 6), [255, 255, 255, 255]);
        // Outside the column: untouched.
        assert_eq!(rgba_at(&px, 12, 0, 6), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 12, 3, 6), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 12, 8, 6), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 12, 11, 6), [0, 0, 0, 0]);
    }

    /// `no-repeat` paints exactly one tile at the position offset,
    /// clipped to the area like the old clip pair did.
    #[test]
    fn no_repeat_paints_single_positioned_tile() {
        let mut list = DisplayList::new();
        let id = ImageId(1);
        list.add_image(id, checker_tile());
        list.push(DisplayItem::RepeatImage {
            area: rect(0.0, 0.0, 12.0, 12.0),
            image: id,
            tile: (4.0, 4.0),
            repeat: BackgroundRepeat::NoRepeat,
            offset: (5.0, 3.0),
        });
        let px = CpuBackend::render_to_rgba(&list, 12, 12);
        // Tile covers x=5..9, y=3..7.
        assert_eq!(rgba_at(&px, 12, 5, 3), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&px, 12, 7, 5), [255, 255, 255, 255]);
        assert_eq!(rgba_at(&px, 12, 8, 6), [255, 255, 255, 255]);
        assert_eq!(rgba_at(&px, 12, 4, 3), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 12, 9, 3), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 12, 5, 2), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 12, 5, 7), [0, 0, 0, 0]);
    }

    /// A hostile repeat (1 px tile over a 10M-px area) stays bounded:
    /// one display item, the area clamped, and only the visible tiles
    /// expanded — the render finishes fast and the viewport still shows
    /// the full-density tiling.
    #[test]
    fn repeat_image_huge_area_stays_bounded() {
        let mut list = DisplayList::new();
        let id = ImageId(1);
        list.add_image(id, DecodedImage::from_rgba(1, 1, vec![200, 10, 10, 255]));
        list.push(DisplayItem::RepeatImage {
            area: rect(0.0, 0.0, 10_000_000.0, 10_000_000.0),
            image: id,
            tile: (1.0, 1.0),
            repeat: BackgroundRepeat::Repeat,
            offset: (0.0, 0.0),
        });
        assert_eq!(list.items.len(), 1, "huge repeat is one item");
        let start = std::time::Instant::now();
        let px = CpuBackend::render_to_rgba(&list, 64, 64);
        assert!(
            start.elapsed().as_secs() < 10,
            "hostile repeat must stay bounded"
        );
        // The visible 64x64 viewport is fully tiled at 1px density.
        assert_eq!(rgba_at(&px, 64, 0, 0), [200, 10, 10, 255]);
        assert_eq!(rgba_at(&px, 64, 63, 63), [200, 10, 10, 255]);
        assert_eq!(rgba_at(&px, 64, 31, 17), [200, 10, 10, 255]);
    }

    /// A hand-built `RepeatImage` with a hostile subnormal tile
    /// (`f32::from_bits(1)`, the smallest positive subnormal) over an
    /// 8192 px area must terminate: `paint_repeat_image` enlarges the
    /// tile through the shared [`MAX_REPEAT_TILES`] cap
    /// (see [`cap_repeat_tile`]) before the space/repeat loops run, so
    /// the grid stays bounded. The render runs on a thread with a
    /// timeout so a regression of the cap fails the test instead of
    /// hanging the suite.
    #[test]
    fn repeat_image_subnormal_tile_terminates() {
        use std::sync::mpsc;
        use std::time::Duration;

        for repeat in [BackgroundRepeat::Repeat, BackgroundRepeat::Space] {
            let mut list = DisplayList::new();
            let id = ImageId(1);
            list.add_image(id, DecodedImage::from_rgba(1, 1, vec![200, 10, 10, 255]));
            // Hand-built, bypassing the builder's tile cap.
            list.push(DisplayItem::RepeatImage {
                area: rect(0.0, 0.0, 8192.0, 8192.0),
                image: id,
                tile: (f32::from_bits(1), f32::from_bits(1)),
                repeat,
                offset: (0.0, 0.0),
            });
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                let px = CpuBackend::render_to_rgba(&list, 64, 64);
                let _ = tx.send(px);
            });
            let px = rx
                .recv_timeout(Duration::from_secs(30))
                .expect("subnormal repeat tile render must terminate (tile cap regressed?)");
            assert_eq!(px.len(), 64 * 64 * 4);
            // The enlarged tile still covers the area: every viewport
            // pixel is painted.
            let (chunks, _) = px.as_chunks::<4>();
            assert!(
                chunks.iter().all(|p| *p == [200, 10, 10, 255]),
                "enlarged tile covers the viewport for {repeat:?}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Fractional clip exactness
    // ------------------------------------------------------------------

    /// Direct unit test of the shared coverage formula: quantized
    /// pixel-area coverage of the clip rect. This is the contract the
    /// GPU clip shader mirrors, so clip boundaries agree pixel-for-pixel
    /// across backends.
    #[test]
    fn clip_coverage_byte_matches_pixel_center_rule() {
        // Clip x=10.5, w=100.25 (right=110.75), full height.
        let cov = |x: usize| clip_coverage_byte(10.5, 0.0, 110.75, 100.0, x, 50);
        assert_eq!(cov(9), 0, "pixel left of the edge is excluded");
        assert_eq!(cov(10), 128, "half-covered pixel");
        assert_eq!(cov(11), 255, "fully covered pixel");
        assert_eq!(cov(110), 191, "0.75-covered pixel");
        assert_eq!(cov(111), 0, "pixel right of the edge is excluded");
        // Vertical: clip y=20.1, h=30.5 (bottom=50.6).
        let cov_y = |y: usize| clip_coverage_byte(0.0, 20.1, 100.0, 50.6, 50, y);
        assert_eq!(cov_y(19), 0);
        // 0.9 in f32 is 21.0 - 20.1f32 = 0.8999996, hence 229 not 230:
        // the formula's f32 ops are the contract, not real-number math.
        assert_eq!(cov_y(20), 229, "0.9-covered pixel");
        assert_eq!(cov_y(50), 153, "0.6-covered pixel");
        assert_eq!(cov_y(51), 0);
    }

    /// A fractional clip on a solid fill: boundary pixels carry the
    /// exact coverage byte, the pixel outside the edge is untouched,
    /// and interior pixels are fully painted. Guards the old
    /// floor/ceil-outward mask, which painted the half-outside pixel at
    /// full coverage.
    #[test]
    fn fractional_clip_solid_fill_boundary_is_exact() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        // x=10.5, width=100.25.
        list.push(DisplayItem::PushClip {
            rect: rect(10.5, 0.0, 100.25, 100.0),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 200.0, 100.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::PopClip);
        let px = backend.render_to_rgba_full(&list, 200, 100, 1.0, (0.0, 0.0));
        // Left edge at 10.5: pixel 9 untouched, pixel 10 at half.
        assert_eq!(rgba_at(&px, 200, 9, 50), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 200, 10, 50), [255, 0, 0, 128]);
        assert_eq!(rgba_at(&px, 200, 11, 50), [255, 0, 0, 255]);
        // Right edge at 110.75: pixel 110 at 0.75, pixel 111 untouched.
        assert_eq!(rgba_at(&px, 200, 110, 50), [255, 0, 0, 191]);
        assert_eq!(rgba_at(&px, 200, 111, 50), [0, 0, 0, 0]);
    }

    /// Nested fractional clips intersect exactly: the boundary bytes
    /// come from the intersection, with no leak at either clip's edge.
    #[test]
    fn fractional_nested_clip_intersection_is_exact() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(10.5, 10.5, 100.25, 80.5),
        });
        list.push(DisplayItem::PushClip {
            rect: rect(20.25, 20.75, 50.5, 40.25),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 200.0, 100.0),
            color: Color::rgba8(0, 255, 0, 255),
        });
        list.push(DisplayItem::PopClip);
        list.push(DisplayItem::PopClip);
        let px = backend.render_to_rgba_full(&list, 200, 100, 1.0, (0.0, 0.0));
        // Intersection is (20.25, 20.75)-(70.75, 61.0).
        assert_eq!(rgba_at(&px, 200, 19, 30), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 200, 20, 30), [0, 255, 0, 191]);
        assert_eq!(rgba_at(&px, 200, 70, 30), [0, 255, 0, 191]);
        assert_eq!(rgba_at(&px, 200, 71, 30), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 200, 30, 20), [0, 255, 0, 64]);
        assert_eq!(rgba_at(&px, 200, 30, 60), [0, 255, 0, 255]);
        assert_eq!(rgba_at(&px, 200, 30, 61), [0, 0, 0, 0]);
        // The outer clip still binds: (15, 30) is inside the outer clip
        // but outside the inner one.
        assert_eq!(rgba_at(&px, 200, 15, 30), [0, 0, 0, 0]);
    }

    /// No pixel row or column outside a fractional clip is touched:
    /// sweeps the rows/columns just outside every edge.
    #[test]
    fn fractional_clip_touches_no_outside_row_or_column() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(10.9, 20.1, 50.5, 30.5),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 200.0, 100.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::PopClip);
        let px = backend.render_to_rgba_full(&list, 200, 100, 1.0, (0.0, 0.0));
        // Clip is (10.9, 20.1)-(61.4, 50.6): covered columns 10..62,
        // covered rows 20..51.
        for x in 0..200 {
            assert_eq!(
                rgba_at(&px, 200, x, 19),
                [0, 0, 0, 0],
                "row above clip, x={x}"
            );
            assert_eq!(
                rgba_at(&px, 200, x, 51),
                [0, 0, 0, 0],
                "row below clip, x={x}"
            );
        }
        for y in 0..100 {
            assert_eq!(
                rgba_at(&px, 200, 9, y),
                [0, 0, 0, 0],
                "col left of clip, y={y}"
            );
            assert_eq!(
                rgba_at(&px, 200, 62, y),
                [0, 0, 0, 0],
                "col right of clip, y={y}"
            );
        }
        // Corner pixel carries the exact corner coverage.
        let corner = clip_coverage_byte(10.9, 20.1, 61.4, 50.6, 10, 20);
        assert_eq!(rgba_at(&px, 200, 10, 20)[3], corner);
        assert!(corner > 0 && corner < 255);
    }

    /// An isolated opacity group inside a fractional clip does not leak,
    /// and its boundary is clipped exactly once: the group inherits the
    /// clip while painting, and the composite must not re-apply it
    /// (which would square the boundary coverage: 128 -> ~32).
    #[test]
    fn fractional_clip_opacity_group_no_leak_no_double_clip() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(10.5, 0.0, 100.25, 100.0),
        });
        list.push(DisplayItem::PushOpacity { opacity: 0.5 });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 200.0, 100.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::PopOpacity);
        list.push(DisplayItem::PopClip);
        let px = backend.render_to_rgba_full(&list, 200, 100, 1.0, (0.0, 0.0));
        // No leak outside the clip through the offscreen composite.
        assert_eq!(rgba_at(&px, 200, 9, 50), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 200, 111, 50), [0, 0, 0, 0]);
        // Boundary pixel: clip coverage 128, group opacity 0.5.
        assert_eq!(rgba_at(&px, 200, 10, 50), [255, 0, 0, 64]);
        // Interior: group opacity only.
        assert_eq!(rgba_at(&px, 200, 50, 50), [255, 0, 0, 128]);
    }

    /// Moving a clip's fractional edge without changing its bounding
    /// region still updates the boundary bytes: the mask update tracks
    /// the rect itself, not just the region (the old region-only
    /// early-return left stale bytes here).
    #[test]
    fn fractional_clip_edge_move_updates_bytes() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let render = |backend: &mut CpuBackend, x: f32| {
            let mut list = DisplayList::new();
            list.push(DisplayItem::PushClip {
                rect: rect(x, 0.0, 100.0, 100.0),
            });
            list.push(DisplayItem::FillRect {
                rect: rect(0.0, 0.0, 200.0, 100.0),
                color: Color::rgba8(255, 0, 0, 255),
            });
            list.push(DisplayItem::PopClip);
            backend.render_to_rgba_full(&list, 200, 100, 1.0, (0.0, 0.0))
        };
        // Same bounding region (floor(10.5) == floor(10.7) == 10),
        // different edge: 0.5 -> 128, 0.3 -> 77.
        let a = render(&mut backend, 10.5);
        let b = render(&mut backend, 10.7);
        assert_eq!(rgba_at(&a, 200, 10, 50), [255, 0, 0, 128]);
        assert_eq!(rgba_at(&b, 200, 10, 50), [255, 0, 0, 77]);
        // And back again: the bytes restore exactly.
        let c = render(&mut backend, 10.5);
        assert_eq!(a, c);
    }

    /// A fractional clip whose outward-snapped bound covers the whole
    /// surface must still apply the mask: the boundary bytes are
    /// partial, not 255.
    #[test]
    fn fractional_clip_full_surface_bound_keeps_mask() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(0.5, 0.5, 199.0, 99.0),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 200.0, 100.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::PopClip);
        let px = backend.render_to_rgba_full(&list, 200, 100, 1.0, (0.0, 0.0));
        // Bound is (0,0,200,100) = full surface, but every edge pixel
        // is half-covered.
        assert_eq!(rgba_at(&px, 200, 0, 0), [255, 0, 0, 64]);
        assert_eq!(rgba_at(&px, 200, 199, 99), [255, 0, 0, 64]);
        assert_eq!(rgba_at(&px, 200, 100, 50), [255, 0, 0, 255]);
    }

    /// A fractional scroll offset moves the clip to fractional device px;
    /// the boundary bytes follow the clip, not the pixel grid.
    #[test]
    fn fractional_clip_with_fractional_scroll() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        // Integer clip in page px; the fractional scroll lands it at
        // fractional device px.
        list.push(DisplayItem::PushClip {
            rect: rect(10.0, 0.0, 100.0, 100.0),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 200.0, 100.0),
            color: Color::rgba8(255, 0, 0, 255),
        });
        list.push(DisplayItem::PopClip);
        let px = backend.render_to_rgba_full(&list, 200, 100, 1.0, (0.5, 0.0));
        // Device clip is (9.5, 0)-(109.5, 100).
        assert_eq!(rgba_at(&px, 200, 8, 50), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 200, 9, 50), [255, 0, 0, 128]);
        assert_eq!(rgba_at(&px, 200, 109, 50), [255, 0, 0, 128]);
        assert_eq!(rgba_at(&px, 200, 110, 50), [0, 0, 0, 0]);
    }

    /// A rounded clip cuts the corners: the corner pixel is empty, the
    /// center and edge midpoints are solid, arc pixels are partial.
    #[test]
    fn rounded_clip_cuts_corners() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushRoundedClip {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            radii: [20.0, 20.0, 20.0, 20.0],
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: red(),
        });
        list.push(DisplayItem::PopClip);
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let px = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // Corner pixel (0,0): fully outside the r=20 disc.
        assert_eq!(rgba_at(&px, 100, 0, 0), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 100, 99, 0), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 100, 0, 99), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 100, 99, 99), [0, 0, 0, 0]);
        // Center and edge midpoints: fully covered.
        assert_eq!(rgba_at(&px, 100, 50, 50), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&px, 100, 50, 0), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&px, 100, 0, 50), [255, 0, 0, 255]);
        // Arc pixel (5,5): the disc cuts through it — partial alpha.
        let arc = rgba_at(&px, 100, 5, 5);
        assert_eq!(&arc[..3], &[255, 0, 0]);
        assert!(
            arc[3] > 0 && arc[3] < 255,
            "arc pixel must be partial, got {arc:?}"
        );
    }

    /// A circular clip (`border-radius: 50%` on a square): the center
    /// is solid, the corners are empty.
    #[test]
    fn rounded_clip_circle() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushRoundedClip {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            radii: [50.0, 50.0, 50.0, 50.0],
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: red(),
        });
        list.push(DisplayItem::PopClip);
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let px = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert_eq!(rgba_at(&px, 100, 50, 50), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&px, 100, 0, 0), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 100, 99, 99), [0, 0, 0, 0]);
        // Edge midpoints sit on the disc: fully covered.
        assert_eq!(rgba_at(&px, 100, 50, 0), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&px, 100, 99, 50), [255, 0, 0, 255]);
    }

    /// Nested rounded + rect clips intersect: the inner rounded clip
    /// still cuts corners inside an outer rect clip, and the outer
    /// rect still bounds the rounded one.
    #[test]
    fn rounded_clip_nested_with_rect() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            rect: rect(0.0, 0.0, 60.0, 100.0),
        });
        list.push(DisplayItem::PushRoundedClip {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            radii: [20.0, 20.0, 20.0, 20.0],
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: red(),
        });
        list.push(DisplayItem::PopClip);
        list.push(DisplayItem::PopClip);
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let px = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // Inside both: solid.
        assert_eq!(rgba_at(&px, 100, 50, 50), [255, 0, 0, 255]);
        // Rounded corner cut applies inside the outer rect.
        assert_eq!(rgba_at(&px, 100, 0, 0), [0, 0, 0, 0]);
        // Outside the outer rect: empty even where the rounded clip
        // would cover.
        assert_eq!(rgba_at(&px, 100, 80, 50), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 100, 99, 99), [0, 0, 0, 0]);
    }

    /// The reverse nesting: rect clip inside a rounded clip.
    #[test]
    fn rect_clip_nested_in_rounded() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushRoundedClip {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            radii: [20.0, 20.0, 20.0, 20.0],
        });
        list.push(DisplayItem::PushClip {
            rect: rect(40.0, 40.0, 20.0, 20.0),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: red(),
        });
        list.push(DisplayItem::PopClip);
        list.push(DisplayItem::PopClip);
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let px = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // Inner rect is far from the rounded corners: solid.
        assert_eq!(rgba_at(&px, 100, 50, 50), [255, 0, 0, 255]);
        // Outside the inner rect: empty.
        assert_eq!(rgba_at(&px, 100, 10, 10), [0, 0, 0, 0]);
        assert_eq!(rgba_at(&px, 100, 70, 70), [0, 0, 0, 0]);
    }

    /// Popping a rounded clip restores unclipped painting (the mask
    /// rebuild after the pop must clear the rounded bytes).
    #[test]
    fn rounded_clip_pop_restores() {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushRoundedClip {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            radii: [20.0, 20.0, 20.0, 20.0],
        });
        list.push(DisplayItem::PopClip);
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: red(),
        });
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let px = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert_eq!(rgba_at(&px, 100, 0, 0), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&px, 100, 99, 99), [255, 0, 0, 255]);
    }

    /// All-zero radii behave byte-identically to `PushClip`.
    #[test]
    fn rounded_clip_zero_radii_matches_push_clip() {
        let mut plain = DisplayList::new();
        plain.push(DisplayItem::PushClip {
            rect: rect(10.5, 20.25, 80.0, 60.5),
        });
        plain.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: red(),
        });
        plain.push(DisplayItem::PopClip);
        let mut rounded = DisplayList::new();
        rounded.push(DisplayItem::PushRoundedClip {
            rect: rect(10.5, 20.25, 80.0, 60.5),
            radii: [0.0; 4],
        });
        rounded.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: red(),
        });
        rounded.push(DisplayItem::PopClip);
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let a = backend.render_to_rgba_full(&plain, 100, 100, 1.0, (0.0, 0.0));
        let b = backend.render_to_rgba_full(&rounded, 100, 100, 1.0, (0.0, 0.0));
        assert_eq!(a, b, "zero-radii rounded clip must match PushClip exactly");
    }

    /// `mix-blend-mode` math against hand-computed values (CSS
    /// Compositing and Blending L1). Backdrop [64, 128, 192, 255],
    /// source [192, 64, 128, 255], both opaque; the channels avoid
    /// .5 rounding boundaries in f32.
    #[test]
    fn blend_pixel_known_backdrop_values() {
        let dst = [64u8, 128, 192, 255];
        let src = [192u8, 64, 128, 255];
        let cases = [
            (BlendMode::Normal, [192, 64, 128, 255]),
            (BlendMode::Multiply, [48, 32, 96, 255]),
            (BlendMode::Screen, [208, 160, 224, 255]),
            (BlendMode::Overlay, [96, 65, 192, 255]),
            (BlendMode::Darken, [64, 64, 128, 255]),
            (BlendMode::Lighten, [192, 128, 192, 255]),
        ];
        for (mode, expected) in cases {
            assert_eq!(blend_pixel(src, dst, mode), expected, "{mode:?}");
        }
        // Translucent source over the opaque backdrop (multiply):
        // unpremultiplied Cs = [0.75, 0.25, 0.5], αs = 128/255.
        assert_eq!(
            blend_pixel([96, 32, 64, 128], dst, BlendMode::Multiply),
            [56, 80, 144, 255]
        );
        // Transparent backdrop: the blend reduces to the source for
        // every mode (Cs' = Cs, Co = αs·Cs, αo = αs).
        for (mode, _) in cases {
            assert_eq!(
                blend_pixel(src, [0, 0, 0, 0], mode),
                [192, 64, 128, 255],
                "{mode:?} over transparent"
            );
        }
    }

    /// Full-pipeline blend groups: an opaque backdrop rect, one
    /// isolated group per mode painting the same source rect, checked
    /// at the group's center pixel.
    #[test]
    fn blend_group_renders_known_pixels() {
        fn render(mode: BlendMode) -> Vec<u8> {
            let mut list = DisplayList::new();
            list.push(DisplayItem::FillRect {
                rect: rect(0.0, 0.0, 100.0, 100.0),
                color: Color::rgba8(64, 128, 192, 255),
            });
            list.push(DisplayItem::PushBlend { mode });
            list.push(DisplayItem::FillRect {
                rect: rect(0.0, 0.0, 100.0, 100.0),
                color: Color::rgba8(192, 64, 128, 255),
            });
            list.push(DisplayItem::PopBlend);
            let mut backend = CpuBackend::new(Rc::new(|_| None));
            backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0))
        }
        let cases = [
            (BlendMode::Normal, [192, 64, 128, 255]),
            (BlendMode::Multiply, [48, 32, 96, 255]),
            (BlendMode::Screen, [208, 160, 224, 255]),
            (BlendMode::Overlay, [96, 65, 192, 255]),
            (BlendMode::Darken, [64, 64, 128, 255]),
            (BlendMode::Lighten, [192, 128, 192, 255]),
        ];
        for (mode, expected) in cases {
            let rgba = render(mode);
            assert_eq!(rgba_at(&rgba, 100, 50, 50), expected, "{mode:?}");
        }
    }

    /// `Normal` is the no-op fast path: a `PushBlend { Normal }` group
    /// renders byte-identical to the same items with no group at all
    /// (plain source-over, no isolated group).
    #[test]
    fn normal_blend_group_matches_ungrouped() {
        fn render(grouped: bool) -> Vec<u8> {
            let mut list = DisplayList::new();
            list.push(DisplayItem::FillRect {
                rect: rect(0.0, 0.0, 100.0, 100.0),
                color: Color::rgba8(64, 128, 192, 255),
            });
            if grouped {
                list.push(DisplayItem::PushBlend {
                    mode: BlendMode::Normal,
                });
            }
            list.push(DisplayItem::FillRect {
                rect: rect(10.0, 10.0, 80.0, 80.0),
                color: Color::rgba8(192, 64, 128, 200),
            });
            if grouped {
                list.push(DisplayItem::PopBlend);
            }
            let mut backend = CpuBackend::new(Rc::new(|_| None));
            backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0))
        }
        assert_eq!(render(true), render(false));
    }

    fn gradient_list(css: &str, r: Rect) -> DisplayList {
        use crate::display_list::DisplayListBuilder;
        use crate::gradient::parse_gradient;
        let mut b = DisplayListBuilder::new();
        let parsed = parse_gradient(css).expect("gradient css parses");
        b.push_gradient(r, &parsed, Color::rgba8(0, 0, 0, 255));
        let list = b.build();
        assert_eq!(list.items.len(), 1, "gradient item was pushed");
        list
    }

    /// Linear gradient: left edge ~red, right edge ~blue, middle is an
    /// Oklab blend (neither endpoint).
    #[test]
    fn gradient_linear_golden() {
        let list = gradient_list(
            "linear-gradient(to right, red, blue)",
            rect(10.0, 10.0, 80.0, 80.0),
        );
        let rgba = render_200x200(&list);
        // Pixel centers sit slightly inside the gradient anchors, so
        // assert dominant (not exact) endpoint colors.
        let left = rgba_at(&rgba, 200, 10, 50);
        assert!(
            left[0] >= 250 && left[2] <= 15 && left[0] > left[1] && left[0] > left[2],
            "left ~red: {left:?}"
        );
        let right = rgba_at(&rgba, 200, 89, 50);
        assert!(
            right[2] >= 250 && right[0] <= 15 && right[2] > right[0] && right[2] > right[1],
            "right ~blue: {right:?}"
        );
        let mid = rgba_at(&rgba, 200, 50, 50);
        assert!(
            mid[0] > 50 && mid[2] > 50,
            "midpoint blends red and blue: {mid:?}"
        );
        assert_ne!(mid, [255, 0, 0, 255]);
        assert_ne!(mid, [0, 0, 255, 255]);
        // Outside the rect: transparent.
        assert_eq!(rgba_at(&rgba, 200, 5, 50), [0, 0, 0, 0]);
    }

    /// Radial gradient with an explicit circle radius: exact red at the
    /// center, exact blue at the corners (past the radius, t clamps).
    #[test]
    fn gradient_radial_golden() {
        let list = gradient_list(
            "radial-gradient(circle 40px, red, blue)",
            rect(10.0, 10.0, 80.0, 80.0),
        );
        let rgba = render_200x200(&list);
        // Center pixel is dominantly red (its center sits 0.7px off the
        // gradient center); corners are past the 40px radius, so t
        // clamps to exactly blue.
        let center = rgba_at(&rgba, 200, 50, 50);
        assert!(
            center[0] > 240 && center[0] > center[2] * 5,
            "center ~red: {center:?}"
        );
        assert_eq!(rgba_at(&rgba, 200, 10, 10), [0, 0, 255, 255]);
        assert_eq!(rgba_at(&rgba, 200, 89, 89), [0, 0, 255, 255]);
        assert_eq!(rgba_at(&rgba, 200, 5, 5), [0, 0, 0, 0]);
    }

    /// Conic gradient: 0% at the top, clockwise; right side is the 25%
    /// blend.
    #[test]
    fn gradient_conic_golden() {
        let list = gradient_list("conic-gradient(red, blue)", rect(10.0, 10.0, 80.0, 80.0));
        let rgba = render_200x200(&list);
        let top = rgba_at(&rgba, 200, 50, 10);
        assert!(top[0] > 250 && top[2] < 5, "top ~red: {top:?}");
        let quarter = rgba_at(&rgba, 200, 89, 50);
        assert!(
            quarter[0] > 50 && quarter[2] > 50,
            "right side blends: {quarter:?}"
        );
    }

    /// Repeating gradient with hard stops: exact 20px red/blue bands.
    #[test]
    fn gradient_repeating_golden() {
        let list = gradient_list(
            "repeating-linear-gradient(to right, red 0px 10px, blue 10px 20px)",
            rect(10.0, 10.0, 80.0, 80.0),
        );
        let rgba = render_200x200(&list);
        assert_eq!(rgba_at(&rgba, 200, 10, 50), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 200, 25, 50), [0, 0, 255, 255]);
        assert_eq!(rgba_at(&rgba, 200, 35, 50), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 200, 45, 50), [0, 0, 255, 255]);
    }

    /// Invalid CSS and degenerate rects push nothing.
    #[test]
    fn gradient_push_skips_invalid() {
        use crate::display_list::DisplayListBuilder;
        use crate::gradient::parse_gradient;
        assert!(parse_gradient("linear-gradient(red blue)").is_err());
        // Zero-area rect resolves to nothing.
        let mut b = DisplayListBuilder::new();
        let parsed = parse_gradient("linear-gradient(red, blue)").unwrap();
        b.push_gradient(rect(10.0, 10.0, 0.0, 80.0), &parsed, red());
        assert!(b.build().items.is_empty());
    }

    // ------------------------------------------------------------------
    // clip-path
    // ------------------------------------------------------------------

    use crate::filter::FilterOp;

    fn filter_list(ops: Vec<FilterOp>, content: DisplayItem) -> DisplayList {
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushFilter { ops: ops.into() });
        list.push(content);
        list.push(DisplayItem::PopFilter);
        list
    }

    fn red_rect() -> DisplayItem {
        DisplayItem::FillRect {
            rect: rect(10.0, 10.0, 80.0, 80.0),
            color: red(),
        }
    }

    /// `clip-path: polygon()` clips content to the polygon: inside the
    /// triangle paints, outside stays transparent.
    #[test]
    fn clip_path_polygon_clips_content() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClipPath {
            points: vec![[10.0, 10.0], [90.0, 10.0], [50.0, 90.0]].into(),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 100.0, 100.0),
            color: red(),
        });
        list.push(DisplayItem::PopClip);
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // Inside the triangle.
        assert_eq!(rgba_at(&rgba, 100, 50, 20), [255, 0, 0, 255]);
        // Outside the triangle (bottom-left corner of the rect).
        assert_eq!(rgba_at(&rgba, 100, 10, 80), [0, 0, 0, 0]);
        // Outside the triangle (bottom-right corner).
        assert_eq!(rgba_at(&rgba, 100, 90, 80), [0, 0, 0, 0]);
    }

    /// A degenerate `clip-path` polygon (fewer than 3 vertices) clips
    /// everything out, per spec (zero-area clips paint nothing).
    #[test]
    fn clip_path_degenerate_polygon_clips_everything() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClipPath {
            points: vec![[10.0, 10.0], [90.0, 10.0]].into(),
        });
        list.push(red_rect());
        list.push(DisplayItem::PopClip);
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert_eq!(rgba_at(&rgba, 100, 50, 50), [0, 0, 0, 0]);
    }

    // ------------------------------------------------------------------
    // filter: color ops
    // ------------------------------------------------------------------

    /// `grayscale(1)` on red: 0.2126 * 255 ≈ 54.
    #[test]
    fn filter_grayscale_red_becomes_gray() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = filter_list(vec![FilterOp::Grayscale(1.0)], red_rect());
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        let px = rgba_at(&rgba, 100, 50, 50);
        for (got, want) in px.iter().zip([54u8, 54, 54, 255]) {
            assert!((*got as i16 - want as i16).abs() <= 1, "px={px:?}");
        }
        // Outside the rect: untouched.
        assert_eq!(rgba_at(&rgba, 100, 5, 5), [0, 0, 0, 0]);
    }

    /// `invert(1)` on red gives cyan.
    #[test]
    fn filter_invert_red_becomes_cyan() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = filter_list(vec![FilterOp::Invert(1.0)], red_rect());
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        let px = rgba_at(&rgba, 100, 50, 50);
        for (got, want) in px.iter().zip([0u8, 255, 255, 255]) {
            assert!((*got as i16 - want as i16).abs() <= 1, "px={px:?}");
        }
    }

    /// `brightness(0.5)` halves the red channel.
    #[test]
    fn filter_brightness_halves_red() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = filter_list(vec![FilterOp::Brightness(0.5)], red_rect());
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        let px = rgba_at(&rgba, 100, 50, 50);
        assert!((px[0] as i16 - 128).abs() <= 1, "px={px:?}");
        assert_eq!(px[1], 0);
        assert_eq!(px[2], 0);
        assert_eq!(px[3], 255);
    }

    /// `contrast(0)` flattens everything to the 0.5 pivot gray.
    #[test]
    fn filter_contrast_zero_flattens_to_pivot() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = filter_list(vec![FilterOp::Contrast(0.0)], red_rect());
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        let px = rgba_at(&rgba, 100, 50, 50);
        for (got, want) in px.iter().zip([128u8, 128, 128, 255]) {
            assert!((*got as i16 - want as i16).abs() <= 1, "px={px:?}");
        }
    }

    /// `opacity(0.5)` fades the isolated group: straight (255, 0, 0, 128).
    #[test]
    fn filter_opacity_fades_group() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = filter_list(vec![FilterOp::Opacity(0.5)], red_rect());
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        let px = rgba_at(&rgba, 100, 50, 50);
        assert_eq!(px[0], 255);
        assert_eq!(px[1], 0);
        assert_eq!(px[2], 0);
        assert!((px[3] as i16 - 128).abs() <= 1, "alpha = {}", px[3]);
    }

    /// `saturate(0)` desaturates red to the same gray as `grayscale(1)`.
    #[test]
    fn filter_saturate_zero_grays_red() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = filter_list(vec![FilterOp::Saturate(0.0)], red_rect());
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        let px = rgba_at(&rgba, 100, 50, 50);
        for (got, want) in px.iter().zip([54u8, 54, 54, 255]) {
            assert!((*got as i16 - want as i16).abs() <= 1, "px={px:?}");
        }
    }

    /// `sepia(1)` on red: the spec matrix gives (100, 89, 69).
    #[test]
    fn filter_sepia_red() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = filter_list(vec![FilterOp::Sepia(1.0)], red_rect());
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        let px = rgba_at(&rgba, 100, 50, 50);
        for (got, want) in px.iter().zip([100u8, 89, 69, 255]) {
            assert!((*got as i16 - want as i16).abs() <= 2, "px={px:?}");
        }
    }

    /// `hue-rotate(120deg)` on red: the spec's linear matrix maps red
    /// to (0, 113, 0) — green dominates, red and blue vanish.
    #[test]
    fn filter_hue_rotate_red_toward_green() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = filter_list(vec![FilterOp::HueRotate(120.0)], red_rect());
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        let px = rgba_at(&rgba, 100, 50, 50);
        assert!((px[1] as i16 - 113).abs() <= 2, "px={px:?}");
        assert_eq!(px[0], 0);
        assert_eq!(px[2], 0);
        assert_eq!(px[3], 255);
    }

    /// A chained `grayscale(1) invert(1)` applies left to right:
    /// red -> gray 54 -> 201.
    #[test]
    fn filter_chain_applies_left_to_right() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = filter_list(
            vec![FilterOp::Grayscale(1.0), FilterOp::Invert(1.0)],
            red_rect(),
        );
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        let px = rgba_at(&rgba, 100, 50, 50);
        for (got, want) in px.iter().zip([201u8, 201, 201, 255]) {
            assert!((*got as i16 - want as i16).abs() <= 1, "px={px:?}");
        }
    }

    // ------------------------------------------------------------------
    // filter: blur and drop-shadow
    // ------------------------------------------------------------------

    /// `blur(4px)` spreads the rect's edges: just outside the edge is
    /// dim but non-zero, far away is untouched, the center stays solid.
    #[test]
    fn filter_blur_spreads_edges() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = filter_list(
            vec![FilterOp::Blur(4.0)],
            DisplayItem::FillRect {
                rect: rect(45.0, 45.0, 10.0, 10.0),
                color: Color::rgba8(255, 255, 255, 255),
            },
        );
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // Center stays (near-)solid.
        assert!(rgba_at(&rgba, 100, 50, 50)[3] > 200);
        // 2px outside the edge: blurred spill, dim but non-zero.
        let edge = rgba_at(&rgba, 100, 43, 50)[3];
        assert!(edge > 0 && edge < 255, "edge alpha = {edge}");
        // 12px outside: beyond the blur's finite support, untouched.
        assert_eq!(rgba_at(&rgba, 100, 33, 50)[3], 0);
    }

    /// The group pixmap inflates by the filter spill: `blur(8px)` on a
    /// rect at the surface corner still spreads past the content
    /// bounds instead of being culled.
    #[test]
    fn filter_blur_spill_inflates_group_bounds() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = filter_list(
            vec![FilterOp::Blur(8.0)],
            DisplayItem::FillRect {
                rect: rect(0.0, 0.0, 10.0, 10.0),
                color: red(),
            },
        );
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // 5px right of the rect: inside the ~12px spill, must be red-ish.
        let px = rgba_at(&rgba, 100, 15, 5);
        assert!(px[0] > 0 && px[3] > 0, "spill pixel = {px:?}");
    }

    /// `drop-shadow(4px 0 black)` paints the shadow under the source,
    /// offset by (4, 0).
    #[test]
    fn filter_drop_shadow_offsets_under_source() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = filter_list(
            vec![FilterOp::DropShadow {
                offset: (4.0, 0.0),
                blur: 0.0,
                color: Color::rgba8(0, 0, 0, 255),
            }],
            DisplayItem::FillRect {
                rect: rect(10.0, 10.0, 20.0, 20.0),
                color: red(),
            },
        );
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // Inside the shadow but outside the rect: black.
        assert_eq!(rgba_at(&rgba, 100, 32, 20), [0, 0, 0, 255]);
        // Inside the rect: the red source over the shadow.
        assert_eq!(rgba_at(&rgba, 100, 20, 20), [255, 0, 0, 255]);
        // Far from both: untouched.
        assert_eq!(rgba_at(&rgba, 100, 50, 50), [0, 0, 0, 0]);
    }

    /// An empty filter chain paints its content inline, unfiltered
    /// (the builder never emits these; the backend stays total).
    #[test]
    fn filter_empty_chain_paints_inline() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = filter_list(vec![], red_rect());
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        assert_eq!(rgba_at(&rgba, 100, 50, 50), [255, 0, 0, 255]);
    }

    // ------------------------------------------------------------------
    // backdrop-filter
    // ------------------------------------------------------------------

    fn backdrop_list(ops: Vec<FilterOp>, region: Rect, content: Vec<DisplayItem>) -> DisplayList {
        let mut list = DisplayList::new();
        // Backdrop: red left half, blue right half.
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 50.0, 100.0),
            color: red(),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(50.0, 0.0, 50.0, 100.0),
            color: Color::rgba8(0, 0, 255, 255),
        });
        list.push(DisplayItem::PushBackdropFilter {
            ops: ops.into(),
            rect: region,
        });
        for item in content {
            list.push(item);
        }
        list.push(DisplayItem::PopBackdropFilter);
        list
    }

    /// `backdrop-filter: grayscale(1)` grays the backdrop inside the
    /// region only; outside stays colorful.
    #[test]
    fn backdrop_filter_grayscale_applies_in_region() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = backdrop_list(
            vec![FilterOp::Grayscale(1.0)],
            rect(25.0, 0.0, 50.0, 100.0),
            vec![],
        );
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // Inside the region, over red: gray 54.
        let px = rgba_at(&rgba, 100, 30, 50);
        for (got, want) in px.iter().zip([54u8, 54, 54, 255]) {
            assert!((*got as i16 - want as i16).abs() <= 1, "px={px:?}");
        }
        // Inside the region, over blue: gray 18.
        let px = rgba_at(&rgba, 100, 70, 50);
        for (got, want) in px.iter().zip([18u8, 18, 18, 255]) {
            assert!((*got as i16 - want as i16).abs() <= 1, "px={px:?}");
        }
        // Outside the region: untouched red and blue.
        assert_eq!(rgba_at(&rgba, 100, 10, 50), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&rgba, 100, 90, 50), [0, 0, 255, 255]);
    }

    /// The element's own content paints over the filtered backdrop.
    #[test]
    fn backdrop_filter_content_paints_over() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let list = backdrop_list(
            vec![FilterOp::Grayscale(1.0)],
            rect(25.0, 0.0, 50.0, 100.0),
            vec![DisplayItem::FillRect {
                rect: rect(40.0, 40.0, 20.0, 20.0),
                color: Color::rgba8(0, 255, 0, 255),
            }],
        );
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // Content green paints over the grayed backdrop.
        assert_eq!(rgba_at(&rgba, 100, 50, 50), [0, 255, 0, 255]);
        // Backdrop beside the content is still filtered.
        let px = rgba_at(&rgba, 100, 30, 50);
        assert!((px[0] as i16 - 54).abs() <= 1, "px={px:?}");
    }

    /// `backdrop-filter: blur(4px)` softens the backdrop's sharp
    /// black/white edge inside the region; outside stays sharp.
    #[test]
    fn backdrop_filter_blur_softens_backdrop_edge() {
        let mut backend = CpuBackend::new(Rc::new(|_| None));
        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            rect: rect(0.0, 0.0, 50.0, 100.0),
            color: Color::rgba8(0, 0, 0, 255),
        });
        list.push(DisplayItem::FillRect {
            rect: rect(50.0, 0.0, 50.0, 100.0),
            color: Color::rgba8(255, 255, 255, 255),
        });
        list.push(DisplayItem::PushBackdropFilter {
            ops: vec![FilterOp::Blur(4.0)].into(),
            rect: rect(25.0, 0.0, 50.0, 100.0),
        });
        list.push(DisplayItem::PopBackdropFilter);
        let rgba = backend.render_to_rgba_full(&list, 100, 100, 1.0, (0.0, 0.0));
        // On the edge inside the region: blurred to mid-gray.
        let px = rgba_at(&rgba, 100, 50, 50);
        assert!(
            px[0] > 80 && px[0] < 175,
            "edge should be mid-gray, px={px:?}"
        );
        // Far left inside the region: still black.
        assert_eq!(rgba_at(&rgba, 100, 30, 50), [0, 0, 0, 255]);
        // Far right inside the region: still white.
        assert_eq!(rgba_at(&rgba, 100, 70, 50), [255, 255, 255, 255]);
    }
}
