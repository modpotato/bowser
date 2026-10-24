//! Glyph rasterization (swash) and atlas caches.
//!
//! [`rasterize_glyph`] rasterizes one glyph to an 8-bit coverage mask via
//! swash, resolving the face through [`FontData::font_ref`]. [`GlyphAtlas`]
//! is the CPU-side cache over it (used by the CPU backend); [`Atlas`] is
//! the GPU texture cache consumed by `wgpu_be`. Both caches rasterize
//! *real* glyph coverage through [`rasterize_glyph`] over caller-supplied
//! [`FontData`]; the procedural stand-in is gone (removed 2026-09-26).
//!
//! [`Atlas`] public surface:
//!
//! - `Atlas::new(device) -> Self`
//! - `get_or_rasterize(&mut self, queue: &wgpu::Queue, font_key: FontKey,
//!   font_data: &FontData, glyph_id: u16, size_px: f32)
//!   -> Option<GlyphCell>` — the glyph's atlas cell plus its ink's real
//!   placement (bearings, ink size); `None` when the glyph has no coverage
//!   (e.g. space) or the font is unusable. (Changed 2026-09-26: gained
//!   `font_data`, `glyph_id` narrowed from `u32` (codepoint-as-id) to
//!   `u16` (real glyph id), and the return is now `Option` — the old form
//!   always produced procedural coverage and never failed. Changed
//!   2026-09-27: returns [`GlyphCell`] — the cell rect plus the ink's
//!   real bearings — so backends stamp the ink sub-rect instead of the
//!   whole square cell. Changed 2026-09-27: the atlas is generational —
//!   overflow retires the full page and allocates in a fresh one instead
//!   of evicting live cells, so a quad's cell rect is never overwritten
//!   between lowering and GPU submit.)
//! - `texture_view(&self) -> &wgpu::TextureView` — the current
//!   generation's view.
//! - `end_frame(&mut self)` — drop retired generations; call after
//!   `queue.submit` each frame.
//!
//! Real shaping (added 2026-09-27): [`shape_text`] /
//! [`shape_text_with_direction`] shape a run with swash — kerning,
//! ligature substitution, cluster-aware [`ShapedGlyph`] runs whose ids
//! feed [`GlyphAtlas`]/[`Atlas`] directly — falling back to the interim
//! per-char path when the font cannot be shaped. Both paint backends
//! consume [`shape_text`] (positioning by `x_offset`/`advance`, stamping
//! each glyph's real ink rect); the per-char path survives only as the
//! fallback inside [`shape_text_with_direction`].

use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;
use std::sync::{Arc, Weak};

use crate::display_list::FontKey;
use crate::fonts::{FnvHashMap, FontData};

/// One rasterized glyph: 8-bit coverage plus placement.
///
/// `coverage` is row-major, `width * height` bytes, top row first.
/// (`left`, `top`) is the bitmap's top-left corner relative to the pen
/// position, in raster px: the bitmap stamps at
/// `(pen_x + left, pen_y - top)`.
#[derive(Clone, Debug)]
pub struct RasterGlyph {
    pub width: u32,
    pub height: u32,
    pub left: i32,
    pub top: i32,
    pub coverage: Vec<u8>,
}

/// Rasterize one glyph to an 8-bit coverage mask with swash.
///
/// The face comes from [`FontData::font_ref`]; `glyph_id` is the font's
/// glyph id (not a char — map through the font's charmap first);
/// `size_px` is the rasterized pixel size. Returns `None` when the font
/// does not parse, the size is degenerate, or the glyph has no coverage
/// (e.g. space).
pub fn rasterize_glyph(font_data: &FontData, glyph_id: u16, size_px: f32) -> Option<RasterGlyph> {
    use swash::scale::{Render, Source, StrikeWith};
    use swash::zeno::Format;

    if !size_px.is_finite() || size_px <= 0.0 {
        return None;
    }
    let font = font_data.font_ref()?;
    let mut scale_ctx = swash::scale::ScaleContext::new();
    let mut scaler = scale_ctx.builder(font).size(size_px).hint(true).build();
    let image = Render::new(&[
        Source::ColorOutline(0),
        Source::ColorBitmap(StrikeWith::BestFit),
        Source::Outline,
    ])
    .format(Format::Alpha)
    .render(&mut scaler, glyph_id)?;
    let p = image.placement;
    if p.width == 0 || p.height == 0 || image.data.len() != p.width as usize * p.height as usize {
        return None;
    }
    Some(RasterGlyph {
        width: p.width,
        height: p.height,
        left: p.left,
        top: p.top,
        coverage: image.data,
    })
}

/// Cache key for [`GlyphAtlas`]: font + glyph id + rasterized pixel size.
///
/// Both glyph caches ([`GlyphAtlas`] and the GPU [`Atlas`]) key by the
/// same [`rounded_size`]: font + glyph id + rounded px, so a glyph
/// requested at the same size lands on the same cache entry in both
/// backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CpuGlyphKey {
    font: FontKey,
    glyph_id: u16,
    size_px: u16,
}

/// Rasterized glyph size ceiling in px, shared by both glyph caches.
///
/// [`GlyphAtlas`] and [`Atlas`] key by [`rounded_size`] clamped to this
/// ceiling, so a glyph requested at the same size lands on the same
/// cache entry in both backends. 1024 px covers any on-screen glyph
/// (the CPU backend additionally clamps its device-px raster size to
/// this range before shaping).
pub(crate) const MAX_RASTER_PX: u16 = 1024;

/// Round `size_px` to whole px, clamped to 1..=[`MAX_RASTER_PX`].
///
/// Shared rounding convention for both glyph caches: `round()`, not
/// `ceil()`, so CPU and GPU atlas keys agree.
pub(crate) fn rounded_size(size_px: f32) -> u16 {
    size_px.round().clamp(1.0, f32::from(MAX_RASTER_PX)) as u16
}

/// CPU-side glyph cache over [`rasterize_glyph`].
///
/// Keyed `(FontKey, glyph id, raster px)` exactly like the GPU [`Atlas`],
/// so CPU and GPU caches stay comparable. Misses rasterize through
/// [`rasterize_glyph`]; glyphs with no coverage (space) are not cached
/// and report a miss. The cache lives per render (the CPU backend builds
/// a fresh one per frame), so no eviction policy is needed.
///
/// FNV-1a-hashed (see [`FnvBuildHasher`](crate::fonts::FnvBuildHasher)):
/// keys are `(FontKey, glyph id, raster px)` structs probed per glyph.
/// Hash-only change; values are untouched.
#[derive(Debug, Default)]
pub struct GlyphAtlas {
    cache: FnvHashMap<CpuGlyphKey, RasterGlyph>,
}

impl GlyphAtlas {
    /// Empty atlas.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the cached glyph, rasterizing and caching on a miss.
    /// `None` when the glyph has no coverage or the font is unusable.
    pub fn get_or_rasterize(
        &mut self,
        font_key: FontKey,
        font_data: &FontData,
        glyph_id: u16,
        size_px: f32,
    ) -> Option<&RasterGlyph> {
        use std::collections::hash_map::Entry;
        let size = rounded_size(size_px);
        let key = CpuGlyphKey {
            font: font_key,
            glyph_id,
            size_px: size,
        };
        match self.cache.entry(key) {
            Entry::Occupied(_) => {}
            Entry::Vacant(slot) => {
                let raster = rasterize_glyph(font_data, glyph_id, f32::from(size))?;
                slot.insert(raster);
            }
        }
        self.cache.get(&key)
    }

    /// Look up a previously cached glyph without rasterizing.
    pub fn get(&self, font_key: FontKey, glyph_id: u16, size_px: f32) -> Option<&RasterGlyph> {
        let key = CpuGlyphKey {
            font: font_key,
            glyph_id,
            size_px: rounded_size(size_px),
        };
        self.cache.get(&key)
    }
}

/// Atlas texture size in px (fixed; the real worker may grow this).
pub const ATLAS_SIZE: u32 = 1024;
/// Padding between packed glyph cells, in px.
const CELL_PAD: u16 = 1;

/// Cache key: font + glyph id + rasterized pixel size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GlyphKey {
    font: FontKey,
    glyph_id: u32,
    size_px: u16,
}

/// One uploaded glyph cell: the atlas cell plus the ink's real placement.
///
/// The cell is a `cell`-px square with the ink zero-padded at its
/// top-left (see [`glyph_cell_bitmap`]); `ink_w`/`ink_h` are the ink's
/// real size and (`left`, `top`) its bearings, so backends stamp the ink
/// sub-rect at `(pen_x + x_offset + left, baseline + y_offset - top)`
/// instead of drawing the whole square cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlyphCell {
    /// Atlas cell rect `(x, y, w, h)` in atlas px; `w == h == cell`.
    pub cell: (u16, u16, u16, u16),
    /// Atlas generation that owns this cell (see [`Atlas`]): the cell
    /// rect is only valid against that generation's texture, which the
    /// atlas keeps alive until [`Atlas::end_frame`].
    pub generation: u64,
    /// Ink width/height in px: the uploaded bitmap's real content,
    /// clamped to the cell.
    pub ink_w: u16,
    pub ink_h: u16,
    /// Bearings from [`rasterize_glyph`]: the ink stamps at
    /// `(pen_x + left, baseline - top)`.
    pub left: i32,
    pub top: i32,
}

/// One atlas generation: a single [`ATLAS_SIZE`]x[`ATLAS_SIZE`] texture
/// plus its shelf-packing state and glyph cache.
///
/// A page is never mutated after retirement (see [`Atlas::rollover`]):
/// its texture stays alive so quads lowered before the overflow keep
/// sampling valid cells.
struct AtlasPage {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    generation: u64,
    /// FNV-1a-hashed (see [`FnvBuildHasher`](crate::fonts::FnvBuildHasher)):
    /// probed per glyph on the hot atlas path. Hash-only change.
    cache: FnvHashMap<GlyphKey, GlyphCell>,
    shelf_y: u16,
    shelf_h: u16,
    shelf_x: u16,
}

impl AtlasPage {
    fn new(device: &wgpu::Device, generation: u64) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bowser-glyph-atlas"),
            size: wgpu::Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                // COPY_SRC for test readback / debug dumps of atlas contents.
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            generation,
            cache: FnvHashMap::default(),
            shelf_y: 0,
            shelf_h: 0,
            shelf_x: 0,
        }
    }

    /// The page has no shelf allocations yet.
    ///
    /// Shelf state, not cache occupancy: after a rollover the fresh page
    /// inherits the retired page's glyph cache (see [`Atlas::rollover`])
    /// while its shelves start empty.
    fn is_empty(&self) -> bool {
        self.shelf_x == 0 && self.shelf_y == 0
    }
}

/// Shelf-packaged grayscale glyph atlas (R8Unorm, coverage in `r`).
///
/// Generational: the atlas never evicts a live cell. When a glyph does
/// not fit the current page, the whole page is retired — pushed to a
/// per-frame retired list whose textures stay alive — and the glyph is
/// allocated in a fresh page (a new [`GlyphCell::generation`]). A
/// quad's cell rect is therefore never overwritten between lowering and
/// GPU submit, which the old `cache.clear()` + cell-reuse policy
/// violated (mid-frame eviction uploaded the new glyph over
/// still-referenced bitmaps).
///
/// The retired list is dropped by [`Atlas::end_frame`], which the
/// caller invokes after `queue.submit` each frame: by then no new draw
/// can reference the retired quads. Cost per overflow is one 1 MiB
/// texture until `end_frame`; the fresh page inherits the retired page's
/// glyph cache (see [`Atlas::rollover`]), so glyphs cached before the
/// overflow stay hits instead of re-rasterizing on next use.
pub struct Atlas {
    device: wgpu::Device,
    current: AtlasPage,
    retired: Vec<AtlasPage>,
    generation: u64,
}

impl Atlas {
    /// Create an empty atlas texture on `device`.
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            device: device.clone(),
            current: AtlasPage::new(device, 0),
            retired: Vec::new(),
            generation: 0,
        }
    }

    /// View of the current generation's atlas texture for shader binding.
    pub fn texture_view(&self) -> &wgpu::TextureView {
        &self.current.view
    }

    /// View of one generation's atlas texture for shader binding.
    ///
    /// The current generation's view, or a retired page's view when the
    /// generation rolled over mid-frame. `None` for unknown generations
    /// or generations dropped by [`Atlas::end_frame`].
    pub fn view_for_generation(&self, generation: u64) -> Option<&wgpu::TextureView> {
        if generation == self.current.generation {
            return Some(&self.current.view);
        }
        self.retired
            .iter()
            .find(|page| page.generation == generation)
            .map(|page| &page.view)
    }

    /// Drop retired atlas generations.
    ///
    /// Call after `queue.submit` each frame. Retired pages stay alive
    /// until this call so quads lowered before an overflow keep sampling
    /// valid cells; once the frame is submitted no new draw can reference
    /// them. Skipping this call leaks one 1 MiB texture per overflow.
    // Intentionally-exposed frame-lifecycle API: the wgpu backend calls
    // this after `queue.submit`; nothing in-tree calls it yet.
    #[allow(dead_code)]
    pub fn end_frame(&mut self) {
        self.retired.clear();
    }

    /// The atlas texture itself (tests: readback verification).
    #[cfg(test)]
    pub(crate) fn texture(&self) -> &wgpu::Texture {
        &self.current.texture
    }

    /// Texture backing one generation (tests: readback verification).
    ///
    /// `None` for unknown generations or generations dropped by
    /// [`Atlas::end_frame`].
    #[cfg(test)]
    pub(crate) fn texture_for_generation(&self, generation: u64) -> Option<&wgpu::Texture> {
        if generation == self.current.generation {
            return Some(&self.current.texture);
        }
        self.retired
            .iter()
            .find(|page| page.generation == generation)
            .map(|page| &page.texture)
    }

    /// Number of retired generations held (tests/debugging).
    #[cfg(test)]
    pub(crate) fn retired_len(&self) -> usize {
        self.retired.len()
    }

    /// Number of glyphs currently cached (tests/debugging).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.current.cache.len()
    }

    /// Return the glyph's uploaded cell plus its ink's real placement
    /// (see [`GlyphCell`]), rasterizing and uploading it on a cache miss.
    ///
    /// Coverage is the real swash rasterization of `glyph_id` in
    /// `font_data` (see [`glyph_cell_bitmap`]), not a procedural pattern.
    /// Returns `None` — without caching anything — when the glyph has no
    /// coverage (e.g. space) or the font is unusable, mirroring
    /// [`GlyphAtlas::get_or_rasterize`]. Callers skip such glyphs (but
    /// still advance the pen).
    pub fn get_or_rasterize(
        &mut self,
        queue: &wgpu::Queue,
        font_key: FontKey,
        font_data: &FontData,
        glyph_id: u16,
        size_px: f32,
    ) -> Option<GlyphCell> {
        let key_size = rounded_size(size_px);
        let key = GlyphKey {
            font: font_key,
            glyph_id: u32::from(glyph_id),
            size_px: key_size,
        };
        if let Some(&cached) = self.current.cache.get(&key) {
            // A migrated cell's generation is only live while its page
            // is: the current generation, or a retired one kept alive
            // until `end_frame`. After `end_frame` drops the retired
            // page, the cached generation is dead — the cell rect refers
            // to a texture that no longer exists, and
            // `view_for_generation` would return `None`, silently
            // dropping the glyph's draw batch. Treat the hit as a miss:
            // drop the stale entry and re-rasterize into the current
            // page. Without this the dead entry would be returned
            // forever, since the cache is never otherwise evicted.
            let live = cached.generation == self.current.generation
                || self
                    .retired
                    .iter()
                    .any(|page| page.generation == cached.generation);
            if live {
                return Some(cached);
            }
            self.current.cache.remove(&key);
        }
        // Rasterize once: the uploaded bitmap and the placement come
        // from the same `RasterGlyph`.
        let glyph = rasterize_glyph(font_data, glyph_id, f32::from(key_size))?;
        // Size the cell to the ink's real bounds, not just the nominal
        // size: overhanging ink (italic overhang, large bearings, tall
        // diacritics) would otherwise be truncated to the `rounded_size`
        // square — the CPU backend keeps the full `RasterGlyph`, so the
        // GPU atlas must too. Clamped to `ATLAS_SIZE` so the upload
        // always fits the page texture (`alloc_cell` gives an
        // over-`ATLAS_SIZE` cell a fresh page to itself, but the upload
        // extent itself must fit).
        let ink_max = glyph.width.max(glyph.height).min(ATLAS_SIZE) as u16;
        let cell = key_size.max(ink_max);
        let bitmap = glyph_cell_bitmap(&glyph, cell);

        let (x, y) = self.alloc_cell(cell);
        let rect = (x, y, cell, cell);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.current.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: u32::from(x),
                    y: u32::from(y),
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &bitmap,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(u32::from(cell)),
                rows_per_image: Some(u32::from(cell)),
            },
            wgpu::Extent3d {
                width: u32::from(cell),
                height: u32::from(cell),
                depth_or_array_layers: 1,
            },
        );
        let placed = GlyphCell {
            cell: rect,
            // The owning page's generation: `alloc_cell` may have rolled
            // the page over just above, so read it off `current` rather
            // than the atlas counter.
            generation: self.current.generation,
            ink_w: glyph.width.min(u32::from(cell)) as u16,
            ink_h: glyph.height.min(u32::from(cell)) as u16,
            left: glyph.left,
            top: glyph.top,
        };
        self.current.cache.insert(key, placed);
        Some(placed)
    }

    /// Retire the current page and start a fresh generation.
    ///
    /// The retired page (texture, view, packing state) is pushed to the
    /// retired list: its texture stays alive, so quads lowered before
    /// this call keep sampling valid cells. Reclaimed by
    /// [`Atlas::end_frame`] after submit.
    ///
    /// The retired page's glyph cache migrates into the fresh page, so
    /// lookups for glyphs cached before the overflow stay hits: the
    /// migrated cells keep their original [`GlyphCell::generation`], and
    /// their rects stay valid because a retired page is never mutated
    /// and its texture stays alive in `retired` until `end_frame`
    /// ([`Atlas::view_for_generation`] binds the right texture per
    /// cell). The fresh page's shelf packing starts empty — migrated
    /// rects address the old texture, so they can never collide with new
    /// allocations. Moving the map (rather than cloning) also preserves
    /// its capacity, so the fresh page skips rehash growth while
    /// re-populating.
    fn rollover(&mut self) {
        self.generation += 1;
        let fresh = AtlasPage::new(&self.device, self.generation);
        let mut old = std::mem::replace(&mut self.current, fresh);
        let migrated = std::mem::take(&mut old.cache);
        debug_assert!(self.current.cache.is_empty());
        self.current.cache = migrated;
        self.retired.push(old);
    }

    /// Find shelf space for a `cell`x`cell` glyph in the current page.
    ///
    /// Never reuses a live cell: on overflow the current page is retired
    /// ([`Atlas::rollover`]) and allocation continues in a fresh page, so
    /// cell rects handed out earlier stay valid until
    /// [`Atlas::end_frame`].
    fn alloc_cell(&mut self, cell: u16) -> (u16, u16) {
        let need = cell + CELL_PAD;
        if need > ATLAS_SIZE as u16 {
            // A cell this large can never satisfy the shelf tests below,
            // not even in an empty page (a 1024px cell needs 1025px of
            // shelf): give it a fresh page to itself at the origin. The
            // upload itself (`cell` <= ATLAS_SIZE px) fits. Without this
            // branch the shelf tests would roll over on every call —
            // unboundedly.
            if !self.current.is_empty() {
                self.rollover();
            }
            self.current.shelf_x = need;
            self.current.shelf_h = need;
            return (0, 0);
        }
        if self.current.shelf_x + need > ATLAS_SIZE as u16 {
            // Next shelf.
            self.current.shelf_y += self.current.shelf_h + CELL_PAD;
            self.current.shelf_x = 0;
            self.current.shelf_h = 0;
        }
        if self.current.shelf_y + need > ATLAS_SIZE as u16 {
            // Page full: retire it and allocate in the fresh page, whose
            // shelves are reset — `need <= ATLAS_SIZE` always fits at the
            // origin, so this rolls over at most once per call.
            self.rollover();
        }
        let x = self.current.shelf_x;
        let y = self.current.shelf_y;
        self.current.shelf_x += need;
        self.current.shelf_h = self.current.shelf_h.max(need);
        (x, y)
    }
}

/// Zero-pad a rasterized glyph's coverage to a full `cell`x`cell`
/// bitmap (row-major, 8-bit coverage), ready for a square atlas-cell
/// upload.
///
/// The ink sits at the cell's top-left; every padding texel stays zero
/// (the atlas texture is zero-initialized on creation), so the bitmap
/// uploads 1:1 into the allocated cell. Placement (bearings, baseline)
/// comes from the [`RasterGlyph`] itself; see [`GlyphCell`].
fn glyph_cell_bitmap(glyph: &RasterGlyph, cell: u16) -> Vec<u8> {
    let n = usize::from(cell);
    // Clamp defensively: a rasterized bitmap should never exceed the cell
    // it was rasterized for, but never overrun the upload on a hostile
    // font.
    let w = (glyph.width as usize).min(n);
    let h = (glyph.height as usize).min(n);
    let mut out = vec![0u8; n * n];
    for (dst_row, src_row) in out
        .chunks_exact_mut(n)
        .zip(glyph.coverage.chunks_exact(glyph.width as usize))
        .take(h)
    {
        dst_row[..w].copy_from_slice(&src_row[..w]);
    }
    out
}

// ---------------------------------------------------------------------------
// Real text shaping (swash).
//
// Both paint backends consume [`shape_text`]: kerning, ligature
// substitution, and cluster-aware glyph runs whose ids feed
// [`GlyphAtlas`]/[`Atlas`] directly. The per-char path survives as
// [`shape_per_char`], the fallback when the font cannot be shaped.
// ---------------------------------------------------------------------------

/// Writing direction for shaping.
///
/// Writing direction for [`shape_text_with_direction`].
///
/// `RightToLeft` is wired up: [`shape_text`] picks it per script run
/// automatically (see [`is_rtl_script`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextDirection {
    /// Left to right (default).
    LeftToRight,
    /// Right to left.
    RightToLeft,
}

/// One positioned glyph out of the shaper, ready to feed [`GlyphAtlas`].
///
/// `glyph_id` is the font's real glyph id (feeds the atlas caches
/// directly). (`x_offset`, `y_offset`) is the pen-relative glyph origin
/// in px; `advance` is the pen advance in px (already kerned). `cluster`
/// is the byte range of the source text this glyph belongs to: several
/// glyphs may share one cluster (complex scripts, marks), and one glyph
/// may span several chars ([`ShapedGlyph::is_ligature`]).
#[derive(Clone, Debug)]
pub struct ShapedGlyph {
    pub glyph_id: u16,
    pub x_offset: f32,
    pub y_offset: f32,
    pub advance: f32,
    // Cluster mapping and ligature flag: shaping API for future
    // selection/hit-testing. No paint path reads them yet.
    #[allow(dead_code)]
    pub cluster: Range<usize>,
    #[allow(dead_code)]
    pub is_ligature: bool,
}

/// A shaped text run: cluster-aware positioned glyphs plus the total pen
/// advance in px.
#[derive(Clone, Debug, Default)]
pub struct ShapedRun {
    pub glyphs: Vec<ShapedGlyph>,
    /// Total advance of the run in px (sum of the glyph advances).
    pub width: f32,
}

/// Shape `text` with real typography: kerning, ligature substitution,
/// and cluster-aware glyph runs.
///
/// The text is first split into maximal script runs
/// ([`script_segments`]); each run is shaped with its own script tag and
/// writing direction — runs in right-to-left scripts (see
/// [`is_rtl_script`]) shape with [`TextDirection::RightToLeft`] — and the
/// runs are concatenated on a single pen in visual order, so mixed-script
/// and bidirectional text lays out correctly. When the font cannot be
/// shaped, falls back to the interim per-char path
/// ([`shape_per_char`]); a degenerate size or empty text yields an empty
/// run.
///
/// Results are cached per thread (see [`SHAPE_CACHE`]): shaping is a
/// pure function of its inputs, so repeat frames share the returned
/// [`Rc`] instead of re-running swash and reallocating the glyph `Vec`.
/// The returned run must be treated as immutable — mutating it would be
/// visible to every later caller that hits the cache.
pub fn shape_text(font_data: &FontData, text: &str, size_px: f32) -> Rc<ShapedRun> {
    if !size_px.is_finite() || size_px <= 0.0 || text.is_empty() {
        return Rc::new(ShapedRun::default());
    }
    let bytes = font_data.bytes_arc();
    // Borrowed probe: the (font, face, size) prefix is `Copy`, and the
    // text is looked up through `Arc<str>: Borrow<str>`, so a cache hit
    // hashes the borrowed text without allocating. The pre-wave-62
    // `String`-keyed map probed the same borrowed way, so hits never
    // allocated there either; `Arc<str>` wins on the miss path: the
    // `evict_one` victim scan clones a candidate key per entry (a
    // refcount bump now, a full string copy before — see
    // [`ShapeCache::evict_one`]), and insert is marginally cheaper.
    let prefix = (
        Arc::as_ptr(bytes) as usize,
        font_data.face_index(),
        size_px.to_bits(),
    );
    // `Weak` identifying the same allocation: verifies the `usize` key
    // on a hit (see the [`ShapeCache`] docs).
    let identity = Arc::downgrade(bytes);
    // Hit path: one thread-local borrow, no shaping, no allocation.
    enum Probe {
        Hit(Rc<ShapedRun>),
        Stale,
        Miss,
    }
    let probe = SHAPE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.tick = cache.tick.wrapping_add(1);
        let tick = cache.tick;
        let Some(inner) = cache.map.get_mut(&prefix) else {
            return Probe::Miss;
        };
        match inner.get_mut(text) {
            Some(slot) if slot.font.ptr_eq(&identity) => {
                slot.last_used = tick;
                Probe::Hit(Rc::clone(&slot.run))
            }
            // Address reuse: a freed font's entry collided with a live
            // font at the same address. Flag the stale entry for removal
            // below and fall through to the miss path.
            Some(_) => Probe::Stale,
            None => Probe::Miss,
        }
    });
    match probe {
        Probe::Hit(run) => return run,
        Probe::Miss => {}
        Probe::Stale => SHAPE_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(inner) = cache.map.get_mut(&prefix) {
                inner.remove(text);
                if inner.is_empty() {
                    cache.map.remove(&prefix);
                }
            }
        }),
    }
    // Miss: shape outside the cache borrow, then insert. The text is
    // cloned exactly once, here, on the miss path only.
    let run = Rc::new(shape_uncached(font_data, text, size_px));
    SHAPE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() >= SHAPE_CACHE_CAP {
            cache.evict_one();
        }
        let tick = cache.tick;
        cache.map.entry(prefix).or_default().insert(
            Arc::from(text),
            ShapeCacheSlot {
                run: Rc::clone(&run),
                font: identity,
                last_used: tick,
            },
        );
    });
    run
}

/// One [`SHAPE_CACHE`] entry: the shared shaped run, the [`Weak`]
/// verifying the key's font identity (see [`ShapeCache`]), and its
/// last-use tick for LRU eviction.
struct ShapeCacheSlot {
    run: Rc<ShapedRun>,
    font: Weak<Vec<u8>>,
    last_used: u64,
}

/// Bound on [`SHAPE_CACHE`] entries: enough for a page's distinct text
/// runs (runs are per text fragment, not per word) while keeping the
/// worst-case retained memory to a few hundred KB.
const SHAPE_CACHE_CAP: usize = 512;

thread_local! {
    /// Per-thread shaped-run cache.
    ///
    /// `shape_text` re-runs for every `DisplayItem::TextRun` on every frame
    /// (lowering runs per frame), and swash shaping — shaper setup, script
    /// segmentation, per-cluster callbacks — plus the fresh `ShapedRun` and
    /// glyph `Vec` allocation dominated text cost on scroll/animation
    /// repaints. Shaping is a pure function of `(font, size, text)`, so the
    /// cache is trivially sound; both paint backends share it with zero
    /// call-site changes. Thread-local (not a global lock) because
    /// [`Rc`] is `!Send` and the render threads never need to share shaped
    /// runs; shaping itself invokes no caller code, so the borrow cannot
    /// re-enter.
    static SHAPE_CACHE: RefCell<ShapeCache> = RefCell::new(ShapeCache::new());
}

/// Bounded LRU map behind [`SHAPE_CACHE`].
///
/// Two-level so the hot lookup borrows: the outer key `(font address,
/// face index, size bits)` is `Copy`, and the inner map is keyed by the
/// text as `Arc<str>` but probed with the caller's borrowed `&str`
/// through `Arc<str>: Borrow<str>` — a hit allocates nothing. (A single
/// flat `HashMap` over an owned 4-field key cannot do this: `Borrow`
/// cannot fabricate a borrowed multi-field view out of owned fields.)
/// The `Arc` key also makes [`evict_one`][ShapeCache::evict_one]'s
/// victim scan cheap: cloning a candidate key is a refcount bump, not
/// a string copy.
///
/// The font is identified by the address of its `Arc`-shared bytes (a
/// [`FontKey`](crate::display_list::FontKey) is not visible at this
/// layer — the backends resolve keys to [`FontData`] before calling).
/// An address alone is not a sound identity: if the font is freed, a
/// later font's bytes could land at the same address. Each slot
/// therefore also stores a [`Weak`] to the original allocation, and a
/// hit is only accepted when
/// [`Weak::ptr_eq`](std::sync::Weak::ptr_eq) confirms the slot's font
/// is the looked-up one; a mismatch drops the stale entry and reshapes.
/// The [`Weak`] keeps only the (tiny) allocation control block alive,
/// never the font bytes. `size_px` keys by exact [`f32::to_bits`] —
/// callers already pass whole `rounded_size` px, so the key space stays
/// discrete.
///
/// Both levels are FNV-1a-hashed (see
/// [`FnvBuildHasher`](crate::fonts::FnvBuildHasher)): the outer key is a
/// `(usize, usize, u32)` tuple and the inner key a text string, probed on
/// every [`shape_text`] call. Hash-only change; slots are untouched.
struct ShapeCache {
    map: FnvHashMap<(usize, usize, u32), FnvHashMap<Arc<str>, ShapeCacheSlot>>,
    tick: u64,
}

impl ShapeCache {
    fn new() -> Self {
        Self {
            map: FnvHashMap::default(),
            tick: 0,
        }
    }

    /// Total entries across the inner maps.
    fn len(&self) -> usize {
        self.map.values().map(FnvHashMap::len).sum()
    }

    /// Make room for one insertion: first reclaim entries whose font
    /// died (they can never hit again), then evict the
    /// least-recently-used entry. Empty inner maps are pruned so the
    /// outer map cannot grow without bound. The victim scan clones one
    /// key per entry, but keys are `Arc<str>`, so each clone is a
    /// refcount bump — O(entries) pointer bumps, no string copies.
    fn evict_one(&mut self) {
        for inner in self.map.values_mut() {
            inner.retain(|_, slot| slot.font.upgrade().is_some());
        }
        self.map.retain(|_, inner| !inner.is_empty());
        if self.len() < SHAPE_CACHE_CAP {
            return;
        }
        let victim = self
            .map
            .iter()
            .flat_map(|(prefix, inner)| {
                inner
                    .iter()
                    .map(|(text, slot)| (*prefix, text.clone(), slot.last_used))
            })
            .min_by_key(|(_, _, last_used)| *last_used);
        if let Some((prefix, text, _)) = victim
            && let Some(inner) = self.map.get_mut(&prefix)
        {
            inner.remove(&text);
            if inner.is_empty() {
                self.map.remove(&prefix);
            }
        }
    }
}

/// The uncached shaping core: split into script segments, shape each
/// with its direction, concatenate on one pen in visual order.
///
/// Separated from [`shape_text`] so the cache hit path never runs it.
fn shape_uncached(font_data: &FontData, text: &str, size_px: f32) -> ShapedRun {
    let mut run = ShapedRun::default();
    for (script, range) in script_segments(text) {
        let direction = if is_rtl_script(script) {
            TextDirection::RightToLeft
        } else {
            TextDirection::LeftToRight
        };
        // `shape_text_with_direction` re-derives the segment's script via
        // `dominant_script`: segments are script-homogeneous by
        // construction, so this is the same tag (weak-only segments are
        // Latin on both sides).
        let seg = shape_text_with_direction(font_data, &text[range.clone()], size_px, direction);
        push_segment(&mut run, seg, range.start, direction);
    }
    run
}

/// [`shape_text`] with an explicit writing direction and no script
/// segmentation: one shaper invocation tagged with the text's
/// [`dominant_script`].
///
/// This is the per-segment shaping entry: [`shape_text`] segments by
/// script and calls this once per segment with the segment's direction.
pub fn shape_text_with_direction(
    font_data: &FontData,
    text: &str,
    size_px: f32,
    direction: TextDirection,
) -> ShapedRun {
    if !size_px.is_finite() || size_px <= 0.0 || text.is_empty() {
        return ShapedRun::default();
    }
    let Some(font) = font_data.font_ref() else {
        return shape_per_char(font_data, text, size_px);
    };
    use swash::shape::{Direction as SwashDirection, ShapeContext};

    let direction = match direction {
        TextDirection::LeftToRight => SwashDirection::LeftToRight,
        TextDirection::RightToLeft => SwashDirection::RightToLeft,
    };
    let mut ctx = ShapeContext::new();
    let mut shaper = ctx
        .builder(font)
        .script(dominant_script(text))
        .direction(direction)
        .size(size_px)
        // Kern and liga are on in swash by default; spell them out so the
        // contract is visible and disabling one is a one-line edit.
        .features([("kern", 1), ("liga", 1)])
        .build();
    shaper.add_str(text);
    let mut run = ShapedRun::default();
    shaper.shape_with(|cluster| {
        let ligature = cluster.is_ligature();
        let range: Range<usize> = cluster.source.into();
        for glyph in cluster.glyphs {
            run.glyphs.push(ShapedGlyph {
                glyph_id: glyph.id,
                x_offset: glyph.x,
                y_offset: glyph.y,
                advance: glyph.advance,
                cluster: range.clone(),
                is_ligature: ligature,
            });
        }
    });
    run.width = run.glyphs.iter().map(|g| g.advance).sum();
    run
}

/// Interim per-char shaping: charmap + advances, one glyph per char, no
/// kerning or ligatures.
///
/// Survives only as the fallback for [`shape_text`] when the font cannot
/// be shaped.
fn shape_per_char(font_data: &FontData, text: &str, size_px: f32) -> ShapedRun {
    let mut run = ShapedRun::default();
    let Some(font) = font_data.font_ref() else {
        return run;
    };
    let charmap = font.charmap();
    let coords: &[swash::NormalizedCoord] = &[];
    let metrics = font.glyph_metrics(coords).scale(size_px);
    for (i, ch) in text.char_indices() {
        let glyph_id = charmap.map(ch);
        run.glyphs.push(ShapedGlyph {
            glyph_id,
            x_offset: 0.0,
            y_offset: 0.0,
            advance: metrics.advance_width(glyph_id),
            cluster: i..i + ch.len_utf8(),
            is_ligature: false,
        });
    }
    run.width = run.glyphs.iter().map(|g| g.advance).sum();
    run
}

/// First non-Common/Inherited/Unknown script in the text, else Latin.
///
/// One script per run is the interim model (it matches the single
/// shaper call per run); mixed-script runs get one shaper each, which a
/// later line-breaking worker will own.
fn dominant_script(text: &str) -> swash::text::Script {
    use swash::text::{Codepoint, Script};
    text.chars()
        .map(Codepoint::script)
        .find(|s| !matches!(s, Script::Common | Script::Inherited | Script::Unknown))
        .unwrap_or(Script::Latin)
}

/// `true` for scripts written right-to-left (Unicode bidi classes R and
/// AL), including the historic ones.
fn is_rtl_script(script: swash::text::Script) -> bool {
    use swash::text::Script::*;
    matches!(
        script,
        Adlam
            | Arabic
            | Avestan
            | Chorasmian
            | Cypriot
            | Elymaic
            | HanifiRohingya
            | Hatran
            | Hebrew
            | ImperialAramaic
            | InscriptionalPahlavi
            | InscriptionalParthian
            | Kharoshthi
            | Mandaic
            | Manichaean
            | Nabataean
            | Nko
            | OldNorthArabian
            | OldSouthArabian
            | Palmyrene
            | Phoenician
            | PsalterPahlavi
            | Samaritan
            | Sogdian
            | OldSogdian
            | Syriac
            | Thaana
            | Yezidi
    )
}

/// Reusable scratch buffers for [`script_segments`]: the three O(n)
/// scratch vecs (per-char items, preceding-strong scripts, effective
/// scripts) live per-thread and are reused across uncached shaping calls
/// instead of being reallocated per call. Thread-local (not a global
/// lock) for the same reason as [`SHAPE_CACHE`]; segmentation is a pure
/// function of the text, so the borrow cannot re-enter — the
/// `try_borrow_mut` fallback below is belt-and-braces.
#[derive(Default)]
struct ScriptSegScratch {
    items: Vec<(usize, usize, swash::text::Script)>,
    prev_strong: Vec<Option<swash::text::Script>>,
    eff: Vec<swash::text::Script>,
}

thread_local! {
    static SCRIPT_SEG_SCRATCH: RefCell<ScriptSegScratch> = RefCell::new(ScriptSegScratch::default());
}

/// Split `text` into `(Script, byte_range)` segments: maximal runs of one
/// script.
///
/// `Common`/`Inherited`/`Unknown` chars (digits, punctuation, spaces)
/// join the surrounding script — each takes the preceding strong script,
/// or the following one when the text starts weak; all-weak text is a
/// single `Latin` segment. Ranges are byte ranges into `text`.
fn script_segments(text: &str) -> Vec<(swash::text::Script, Range<usize>)> {
    use swash::text::{Codepoint, Script};
    fn strong(s: Script) -> bool {
        !matches!(s, Script::Common | Script::Inherited | Script::Unknown)
    }
    SCRIPT_SEG_SCRATCH.with(|cell| {
        let mut borrowed = cell.try_borrow_mut().ok();
        let mut fallback;
        let scratch: &mut ScriptSegScratch = match borrowed.as_mut() {
            Some(s) => s,
            // Re-entered mid-segmentation (cannot happen today:
            // segmentation calls no user code): fresh buffers, same result.
            None => {
                fallback = ScriptSegScratch::default();
                &mut fallback
            }
        };
        let ScriptSegScratch {
            items,
            prev_strong,
            eff,
        } = scratch;
        items.clear();
        items.extend(
            text.char_indices()
                .map(|(i, ch)| (i, i + ch.len_utf8(), Codepoint::script(ch))),
        );
        let n = items.len();
        // Effective script per char: weak chars inherit the nearest strong
        // script, preferring the preceding one.
        prev_strong.clear();
        prev_strong.resize(n, None);
        let mut last: Option<Script> = None;
        for (k, &(_, _, s)) in items.iter().enumerate() {
            if strong(s) {
                last = Some(s);
            }
            prev_strong[k] = last;
        }
        let mut next: Option<Script> = None;
        eff.clear();
        eff.resize(n, Script::Latin);
        for k in (0..n).rev() {
            let s = items[k].2;
            if strong(s) {
                next = Some(s);
                eff[k] = s;
            } else {
                eff[k] = prev_strong[k].or(next).unwrap_or(Script::Latin);
            }
        }
        // Merge consecutive chars with equal effective scripts into ranges.
        let mut segs: Vec<(Script, Range<usize>)> = Vec::new();
        for (k, &(start, end, _)) in items.iter().enumerate() {
            if let Some((s, range)) = segs.last_mut()
                && *s == eff[k]
            {
                range.end = end;
                continue;
            }
            segs.push((eff[k], start..end));
        }
        segs
    })
}

/// Append a shaped script segment to `run`.
///
/// `seg_start` re-bases the segment's cluster byte ranges onto the
/// original text (segments are shaped independently, so their ranges are
/// segment-relative).
///
/// RTL segments arrive from swash in logical cluster order — swash, unlike
/// HarfBuzz, does not reverse RTL runs; visual reordering is the caller's
/// job during bidi layout. The clusters are reversed here, never the glyphs
/// within a cluster (intra-cluster order carries mark and cursive
/// placement), and the visual sequence is laid out left-to-right on the pen
/// like any LTR run: glyph `x_offset`s stay cluster-relative, so the paint
/// backends' pen model (`pen_x + x_offset`, `pen_x += advance`) needs no
/// direction awareness.
fn push_segment(run: &mut ShapedRun, seg: ShapedRun, seg_start: usize, direction: TextDirection) {
    if direction == TextDirection::RightToLeft {
        // Reverse by cluster with no per-cluster `Vec`: walk the glyphs
        // back-to-front, recording each cluster as an index range, and
        // drain the ranges from the back (draining a suffix keeps all
        // earlier indices valid). Intra-cluster glyph order is preserved.
        let mut glyphs = seg.glyphs;
        let mut end = glyphs.len();
        while end > 0 {
            let mut start = end - 1;
            while start > 0 && glyphs[start - 1].cluster == glyphs[end - 1].cluster {
                start -= 1;
            }
            for mut glyph in glyphs.drain(start..end) {
                glyph.cluster.start += seg_start;
                glyph.cluster.end += seg_start;
                run.glyphs.push(glyph);
            }
            end = start;
        }
    } else {
        for mut glyph in seg.glyphs {
            glyph.cluster.start += seg_start;
            glyph.cluster.end += seg_start;
            run.glyphs.push(glyph);
        }
    }
    run.width += seg.width;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fonts::FontCache;

    /// The default system font's decoded data for rasterizer tests.
    ///
    /// `None` on fontless systems (CI): such tests return early and pass.
    fn test_font_data() -> Option<(FontCache, FontKey, FontData)> {
        let cache = FontCache::new();
        let key = cache.default_key()?;
        let data = cache.font_data(key)?.clone();
        Some((cache, key, data))
    }

    fn glyph_id_for(data: &FontData, ch: char) -> u16 {
        data.font_ref()
            .expect("test font must parse")
            .charmap()
            .map(ch)
    }

    #[test]
    fn rounded_size_convention() {
        // Both atlases key by round(), not ceil(): 15.4 -> 15, 15.6 -> 16.
        // The ceiling is the shared MAX_RASTER_PX (1024).
        assert_eq!(rounded_size(15.4), 15);
        assert_eq!(rounded_size(15.6), 16);
        assert_eq!(rounded_size(0.2), 1);
        assert_eq!(rounded_size(5000.0), MAX_RASTER_PX);
    }

    #[test]
    fn gpu_cell_bitmap_uses_real_coverage() {
        let Some((_cache, _key, data)) = test_font_data() else {
            return; // fontless CI: nothing to rasterize
        };
        let gid = glyph_id_for(&data, 'A');
        let glyph = rasterize_glyph(&data, gid, 16.0).expect("real 'A' coverage");
        let bmp = glyph_cell_bitmap(&glyph, 16);
        assert_eq!(bmp.len(), 256, "cell bitmap is cell x cell");
        assert!(bmp.iter().any(|&v| v > 0), "needs coverage");
        assert!(bmp.contains(&0), "needs clear pixels");
        // The cell bitmap must be exactly the swash coverage zero-padded
        // to the cell: this is what proves the atlas uploads real glyph
        // data rather than a procedural pattern.
        let w = glyph.width as usize;
        assert!(w <= 16 && glyph.height as usize <= 16);
        for (row, brow) in bmp
            .as_chunks::<16>()
            .0
            .iter()
            .zip(glyph.coverage.chunks_exact(w))
        {
            assert_eq!(&row[..w], brow, "ink matches the real rasterization");
            assert!(row[w..].iter().all(|&v| v == 0), "padding texels stay zero");
        }
    }

    #[test]
    fn gpu_cell_reports_real_placement() {
        let Some((_cache, _key, data)) = test_font_data() else {
            return; // fontless CI: nothing to rasterize
        };
        let Some((device, queue)) = crate::wgpu_be::test_headless_device() else {
            return; // no Vulkan adapter in this environment
        };
        let gid = glyph_id_for(&data, 'A');
        let mut atlas = Atlas::new(&device);
        let placed = atlas
            .get_or_rasterize(&queue, _key, &data, gid, 16.0)
            .expect("real 'A' has coverage");
        // The cell is a 16px square; the ink is the real rasterization's
        // box, stamped at its bearings.
        assert_eq!(placed.cell.2, 16);
        assert_eq!(placed.cell.3, 16);
        let glyph = rasterize_glyph(&data, gid, 16.0).expect("'A' rasterizes");
        assert_eq!(placed.ink_w, glyph.width.min(16) as u16);
        assert_eq!(placed.ink_h, glyph.height.min(16) as u16);
        assert_eq!(placed.left, glyph.left);
        assert_eq!(placed.top, glyph.top);
        // Cache hit returns the identical placement.
        let again = atlas
            .get_or_rasterize(&queue, _key, &data, gid, 16.0)
            .expect("hit");
        assert_eq!(placed, again);
    }

    /// Read one rect back out of a GPU texture (R8Unorm, 1 byte/px).
    /// Rows are padded to the 256-byte copy alignment.
    fn read_texture_cell(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        rect: (u16, u16, u16, u16),
    ) -> Option<Vec<u8>> {
        let (x, y, w, h) = rect;
        let padded_bpr = u32::from(w).div_ceil(256) * 256;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bowser-atlas-readback"),
            size: u64::from(padded_bpr) * u64::from(h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: u32::from(x),
                    y: u32::from(y),
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(u32::from(h)),
                },
            },
            wgpu::Extent3d {
                width: u32::from(w),
                height: u32::from(h),
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok()?;
        rx.recv().ok()?.ok()?;
        let out = {
            let mapped = slice.get_mapped_range().ok()?;
            let mut pixels = vec![0u8; usize::from(w) * usize::from(h)];
            for row in 0..usize::from(h) {
                let src =
                    &mapped[row * padded_bpr as usize..row * padded_bpr as usize + usize::from(w)];
                let dst = &mut pixels[row * usize::from(w)..(row + 1) * usize::from(w)];
                dst.copy_from_slice(src);
            }
            pixels
        };
        buf.unmap();
        Some(out)
    }

    /// The GPU atlas uploads real glyph coverage: rasterize 'A' into the
    /// atlas, read the cell back, and compare against the swash output.
    ///
    /// Needs a GPU adapter (lavapipe in CI); returns early where there is
    /// none, like the fontless-CI skips above.
    #[test]
    fn gpu_atlas_uploads_real_coverage() {
        let Some((_cache, key, data)) = test_font_data() else {
            return; // fontless CI: nothing to rasterize
        };
        let Some((device, queue)) = crate::wgpu_be::test_headless_device() else {
            return; // no Vulkan adapter in this environment
        };
        let gid = glyph_id_for(&data, 'A');
        let mut atlas = Atlas::new(&device);
        let placed = atlas
            .get_or_rasterize(&queue, key, &data, gid, 16.0)
            .expect("real 'A' has coverage");
        // Cache hit: the same cell, no re-upload.
        let placed2 = atlas
            .get_or_rasterize(&queue, key, &data, gid, 16.0)
            .expect("hit");
        assert_eq!(placed, placed2, "hit must return the cached cell");
        assert_eq!(atlas.len(), 1);

        let expected = {
            let glyph = rasterize_glyph(&data, gid, 16.0).expect("real coverage");
            glyph_cell_bitmap(&glyph, 16)
        };
        let got = read_texture_cell(&device, &queue, atlas.texture(), placed.cell)
            .expect("atlas readback must succeed");
        assert_eq!(
            got, expected,
            "atlas cell must hold the real rasterized coverage"
        );

        // Glyphs with no coverage (space) are not cached and report a
        // miss, mirroring `GlyphAtlas`.
        let space = glyph_id_for(&data, ' ');
        assert!(
            atlas
                .get_or_rasterize(&queue, key, &data, space, 16.0)
                .is_none()
        );
        assert_eq!(atlas.len(), 1, "empty glyphs must not be cached");
    }

    /// Mid-frame atlas overflow must never overwrite live cells.
    ///
    /// Repro: rasterize small "Ag" at 24px, then force an overflow with
    /// a 1024px "W" in the same frame (no submit / `end_frame` between).
    /// The old `cache.clear()` policy reused cell (0,0) for the huge
    /// glyph and uploaded over the small glyphs' bitmaps while quads
    /// lowered earlier in the frame still referenced those rects.
    ///
    /// The small cells' pixels must be byte-identical before and after
    /// the overflow, and identical to rendering "Ag" alone with no
    /// overflow at all.
    #[test]
    fn atlas_overflow_never_overwrites_live_cells() {
        let Some((_cache, key, data)) = test_font_data() else {
            return; // fontless CI: nothing to rasterize
        };
        let Some((device, queue)) = crate::wgpu_be::test_headless_device() else {
            return; // no Vulkan adapter in this environment
        };
        let (gid_a, gid_g, gid_w, gid_m) = (
            glyph_id_for(&data, 'A'),
            glyph_id_for(&data, 'g'),
            glyph_id_for(&data, 'W'),
            glyph_id_for(&data, 'M'),
        );

        let mut atlas = Atlas::new(&device);
        let cell_a = atlas
            .get_or_rasterize(&queue, key, &data, gid_a, 24.0)
            .expect("'A' at 24px must rasterize");
        let cell_g = atlas
            .get_or_rasterize(&queue, key, &data, gid_g, 24.0)
            .expect("'g' at 24px must rasterize");
        assert_eq!(
            cell_a.generation, cell_g.generation,
            "both small glyphs share the first generation"
        );
        let small_tex = atlas
            .texture_for_generation(cell_a.generation)
            .expect("small glyphs' generation must exist");
        let before_a = read_texture_cell(&device, &queue, small_tex, cell_a.cell)
            .expect("atlas readback must succeed");
        let before_g = read_texture_cell(&device, &queue, small_tex, cell_g.cell)
            .expect("atlas readback must succeed");
        assert!(
            before_a.iter().any(|&v| v > 0),
            "sanity: the 'A' cell holds real coverage"
        );

        // Force the overflow: a 1024px cell needs 1025px of shelf, more
        // than the whole atlas — the old code evicted everything here.
        let cell_w = atlas
            .get_or_rasterize(&queue, key, &data, gid_w, 1024.0)
            .expect("'W' at 1024px must rasterize");
        assert_ne!(
            cell_w.generation, cell_a.generation,
            "the oversized glyph must roll to a fresh generation, not reuse live cells"
        );

        // A second oversized glyph: rollover must terminate (the old
        // shelf tests could never place a 1025px need) and must not
        // disturb either earlier generation.
        let cell_m = atlas
            .get_or_rasterize(&queue, key, &data, gid_m, 1024.0)
            .expect("'M' at 1024px must rasterize");
        assert_ne!(
            cell_m.generation, cell_w.generation,
            "each oversized glyph gets its own fresh page"
        );
        assert_eq!(atlas.retired_len(), 2, "two rollovers retired two pages");

        // The small cells' pixels are unchanged by both overflows: the
        // retired generation's texture stayed alive, unmodified.
        let small_tex_after = atlas
            .texture_for_generation(cell_a.generation)
            .expect("retired generation must stay alive until end_frame");
        let after_a = read_texture_cell(&device, &queue, small_tex_after, cell_a.cell)
            .expect("atlas readback must succeed");
        let after_g = read_texture_cell(&device, &queue, small_tex_after, cell_g.cell)
            .expect("atlas readback must succeed");
        assert_eq!(
            before_a, after_a,
            "overflow must not overwrite the small 'A' cell"
        );
        assert_eq!(
            before_g, after_g,
            "overflow must not overwrite the small 'g' cell"
        );

        // ...and identical to rendering "Ag" alone with no overflow.
        let mut alone = Atlas::new(&device);
        let alone_a = alone
            .get_or_rasterize(&queue, key, &data, gid_a, 24.0)
            .expect("'A' at 24px must rasterize");
        let alone_g = alone
            .get_or_rasterize(&queue, key, &data, gid_g, 24.0)
            .expect("'g' at 24px must rasterize");
        assert_eq!(cell_a.cell, alone_a.cell, "same order => same rects");
        assert_eq!(cell_g.cell, alone_g.cell, "same order => same rects");
        let alone_tex = alone
            .texture_for_generation(alone_a.generation)
            .expect("generation 0 must exist");
        let pixels_alone_a = read_texture_cell(&device, &queue, alone_tex, alone_a.cell)
            .expect("atlas readback must succeed");
        let pixels_alone_g = read_texture_cell(&device, &queue, alone_tex, alone_g.cell)
            .expect("atlas readback must succeed");
        assert_eq!(
            after_a, pixels_alone_a,
            "small 'A' pixels must match the no-overflow render"
        );
        assert_eq!(
            after_g, pixels_alone_g,
            "small 'g' pixels must match the no-overflow render"
        );

        // `end_frame` reclaims the retired pages.
        atlas.end_frame();
        assert_eq!(atlas.retired_len(), 0);
        assert!(
            atlas.texture_for_generation(cell_a.generation).is_none(),
            "reclaimed generations must be gone"
        );
        // The current generation still works after reclaim.
        assert!(atlas.texture_for_generation(cell_m.generation).is_some());
    }

    #[test]
    fn rasterize_a_has_nonempty_coverage() {
        let Some((_cache, _key, data)) = test_font_data() else {
            return; // fontless CI: nothing to rasterize
        };
        let gid = glyph_id_for(&data, 'A');
        let glyph = rasterize_glyph(&data, gid, 16.0).expect("'A' must rasterize");
        assert!(glyph.width > 0 && glyph.height > 0);
        assert_eq!(
            glyph.coverage.len(),
            glyph.width as usize * glyph.height as usize
        );
        assert!(
            glyph.coverage.iter().any(|&v| v > 0),
            "'A' must have non-zero coverage"
        );
    }

    #[test]
    fn rasterize_space_has_no_coverage() {
        let Some((_cache, _key, data)) = test_font_data() else {
            return; // fontless CI: nothing to rasterize
        };
        let gid = glyph_id_for(&data, ' ');
        assert!(
            rasterize_glyph(&data, gid, 16.0).is_none(),
            "space must report no coverage"
        );
    }

    #[test]
    fn rasterize_rejects_degenerate_size() {
        let Some((_cache, _key, data)) = test_font_data() else {
            return; // fontless CI: nothing to rasterize
        };
        let gid = glyph_id_for(&data, 'A');
        assert!(rasterize_glyph(&data, gid, 0.0).is_none());
        assert!(rasterize_glyph(&data, gid, -4.0).is_none());
        assert!(rasterize_glyph(&data, gid, f32::NAN).is_none());
    }

    #[test]
    fn golden_fixture_text_rasterizes() {
        // tests/fixtures/text_line.html renders "hi": every char in the
        // golden fixtures' text must produce real coverage.
        let Some((_cache, _key, data)) = test_font_data() else {
            return; // fontless CI: nothing to rasterize
        };
        for ch in "hi".chars() {
            let gid = glyph_id_for(&data, ch);
            let glyph = rasterize_glyph(&data, gid, 16.0)
                .unwrap_or_else(|| panic!("{ch:?} must rasterize"));
            assert!(
                glyph.coverage.iter().any(|&v| v > 0),
                "{ch:?} must have non-zero coverage"
            );
        }
    }

    #[test]
    fn advance_sanity() {
        // Advance widths from the same face the rasterizer uses: positive
        // and ~linear in size.
        let Some((_cache, _key, data)) = test_font_data() else {
            return; // fontless CI: nothing to measure
        };
        let font = data.font_ref().expect("test font must parse");
        let coords: &[swash::NormalizedCoord] = &[];
        let gid = glyph_id_for(&data, 'A');
        let a16 = font.glyph_metrics(coords).scale(16.0).advance_width(gid);
        let a32 = font.glyph_metrics(coords).scale(32.0).advance_width(gid);
        assert!(a16 > 0.0, "advance of 'A' must be positive, got {a16}");
        assert!(
            (a32 - 2.0 * a16).abs() < 1.0,
            "advance should scale ~linearly: {a16} vs {a32}"
        );
    }

    #[test]
    fn cpu_atlas_caches_and_dedupes() {
        let Some((_cache, key, data)) = test_font_data() else {
            return; // fontless CI: nothing to rasterize
        };
        let gid_a = glyph_id_for(&data, 'A');
        let gid_b = glyph_id_for(&data, 'B');
        let mut atlas = GlyphAtlas::new();
        assert!(atlas.get(key, gid_a, 16.0).is_none());

        // Raw pointers: the returned `&RasterGlyph` borrows the atlas, so
        // it cannot be held across the second `&mut` call.
        let first_ptr = {
            let first = atlas
                .get_or_rasterize(key, &data, gid_a, 16.0)
                .expect("'A' must rasterize");
            assert_eq!(
                first.coverage.len(),
                first.width as usize * first.height as usize
            );
            first as *const RasterGlyph
        };
        assert!(atlas.get(key, gid_a, 16.0).is_some());

        // Cache hit: same entry, no re-rasterization.
        let second_ptr = {
            let second = atlas
                .get_or_rasterize(key, &data, gid_a, 16.0)
                .expect("hit");
            second as *const RasterGlyph
        };
        assert_eq!(first_ptr, second_ptr, "hit must return the cached glyph");

        // `get` without rasterizing finds it; unknown glyphs miss.
        assert!(atlas.get(key, gid_a, 16.0).is_some());
        assert!(atlas.get(key, gid_b, 16.0).is_none());

        // A different glyph populates a separate entry.
        let other_ptr = {
            let other = atlas
                .get_or_rasterize(key, &data, gid_b, 16.0)
                .expect("'B' must rasterize");
            other as *const RasterGlyph
        };
        assert_ne!(first_ptr, other_ptr, "distinct glyphs, distinct entries");
        assert!(atlas.get(key, gid_b, 16.0).is_some());
    }

    #[test]
    fn cpu_atlas_space_reports_miss() {
        let Some((_cache, key, data)) = test_font_data() else {
            return; // fontless CI: nothing to rasterize
        };
        let gid = glyph_id_for(&data, ' ');
        let mut atlas = GlyphAtlas::new();
        assert!(atlas.get_or_rasterize(key, &data, gid, 16.0).is_none());
        assert!(
            atlas.get(key, gid, 16.0).is_none(),
            "empty glyphs must not be cached"
        );
    }

    /// The shaper must apply kerning: a kerned pair's shaped width differs
    /// from the sum of the two chars shaped in isolation.
    ///
    /// The probe list covers the classic Latin kern pairs; the first pair
    /// with a measurable adjustment is used, so the test adapts to
    /// whichever font the CI image ships. It passes vacuously only when
    /// the font kerns none of the probes.
    #[test]
    fn shaping_applies_kerning() {
        let Some((_cache, _key, data)) = test_font_data() else {
            return; // fontless CI: nothing to shape
        };
        let pair = ["AV", "To", "Wa", "Yo", "Ta", "LT", "PA", "ro"]
            .into_iter()
            .find(|pair| {
                let shaped = shape_text(&data, pair, 32.0).width;
                let naive: f32 = pair
                    .chars()
                    .map(|ch| shape_text(&data, &ch.to_string(), 32.0).width)
                    .sum();
                (shaped - naive).abs() > 0.05
            });
        let Some(pair) = pair else {
            return; // font kerns none of the probe pairs
        };
        let shaped = shape_text(&data, pair, 32.0);
        let naive: f32 = pair
            .chars()
            .map(|ch| shape_text(&data, &ch.to_string(), 32.0).width)
            .sum();
        assert!(
            (shaped.width - naive).abs() > 0.05,
            "{pair:?}: shaped width {} must differ from the unkerned sum {naive}",
            shaped.width,
        );
        // Two chars, no ligature: kerning must not merge or drop glyphs.
        assert_eq!(shaped.glyphs.len(), 2, "kerning must keep both glyphs");
        assert!(
            shaped.glyphs.iter().all(|g| !g.is_ligature),
            "no ligature involved in {pair:?}"
        );
    }

    /// The shaper must substitute the "fi" ligature: one glyph whose
    /// cluster spans both source chars, with a glyph id distinct from
    /// the plain f and i.
    ///
    /// Passes vacuously when the font has no fi ligature.
    #[test]
    fn shaping_forms_fi_ligature() {
        let Some((_cache, key, data)) = test_font_data() else {
            return; // fontless CI: nothing to shape
        };
        let run = shape_text(&data, "fi", 32.0);
        if run.glyphs.len() != 1 {
            return; // font has no fi ligature
        }
        let glyph = &run.glyphs[0];
        assert!(
            glyph.is_ligature,
            "the single glyph for \"fi\" must be marked a ligature"
        );
        assert_eq!(
            glyph.cluster,
            0..2,
            "ligature cluster must span both source chars"
        );
        assert_ne!(glyph.glyph_id, glyph_id_for(&data, 'f'));
        assert_ne!(glyph.glyph_id, glyph_id_for(&data, 'i'));
        assert!(run.width > 0.0);

        // The shaped ids feed the existing atlas: the ligature glyph
        // rasterizes to real coverage through `GlyphAtlas`.
        let mut atlas = GlyphAtlas::new();
        assert!(
            atlas
                .get_or_rasterize(key, &data, glyph.glyph_id, 32.0)
                .is_some(),
            "shaped ligature glyph must rasterize through the atlas"
        );
    }

    /// Cluster integrity must hold regardless of direction: every glyph's
    /// cluster is a non-empty, char-boundary-aligned byte range, and the
    /// distinct clusters partition the whole input with no gaps or
    /// overlaps. Missing script coverage degrades to .notdef glyphs but
    /// must never break the partition.
    #[test]
    fn shaping_cluster_integrity_rtl_and_ltr() {
        let Some((_cache, _key, data)) = test_font_data() else {
            return; // fontless CI: nothing to shape
        };
        let cases = [
            ("Hello, world!", TextDirection::LeftToRight),
            ("fi AV", TextDirection::LeftToRight),
            ("שלום עולם", TextDirection::RightToLeft),
            ("aבc", TextDirection::RightToLeft),
            ("مرحبا", TextDirection::RightToLeft),
        ];
        for (text, direction) in cases {
            let run = shape_text_with_direction(&data, text, 32.0, direction);
            assert!(
                !run.glyphs.is_empty(),
                "{text:?} ({direction:?}) must produce glyphs"
            );
            for glyph in &run.glyphs {
                assert!(
                    !glyph.cluster.is_empty(),
                    "cluster must be non-empty for {text:?} ({direction:?})"
                );
                assert!(
                    glyph.cluster.end <= text.len(),
                    "cluster {:?} out of bounds for {text:?}",
                    glyph.cluster,
                );
                assert!(
                    text.get(glyph.cluster.clone()).is_some(),
                    "cluster {:?} must sit on char boundaries for {text:?}",
                    glyph.cluster,
                );
            }
            let mut ranges: Vec<Range<usize>> =
                run.glyphs.iter().map(|g| g.cluster.clone()).collect();
            ranges.sort_by_key(|r| r.start);
            ranges.dedup();
            let mut pos = 0;
            for range in &ranges {
                assert_eq!(
                    range.start, pos,
                    "gap or overlap in cluster coverage for {text:?} ({direction:?})"
                );
                pos = range.end;
            }
            assert_eq!(
                pos,
                text.len(),
                "clusters must cover the whole text for {text:?} ({direction:?})"
            );
        }
    }

    /// The per-char fallback path: one glyph per char, no kerning, no
    /// ligatures, zero offsets, char-aligned clusters, width exactly the
    /// advance sum.
    #[test]
    fn fallback_per_char_path() {
        let Some((_cache, _key, data)) = test_font_data() else {
            return; // fontless CI: nothing to shape
        };
        let run = shape_per_char(&data, "AV", 32.0);
        assert_eq!(run.glyphs.len(), 2, "one glyph per char");
        assert_eq!(run.glyphs[0].cluster, 0..1);
        assert_eq!(run.glyphs[1].cluster, 1..2);
        for glyph in &run.glyphs {
            assert_eq!(glyph.x_offset, 0.0);
            assert_eq!(glyph.y_offset, 0.0);
            assert!(!glyph.is_ligature, "fallback never forms ligatures");
            assert_eq!(
                glyph.glyph_id,
                glyph_id_for(&data, "AV".chars().nth(glyph.cluster.start).unwrap())
            );
        }
        // No kerning on the fallback path: width is exactly the sum of
        // the raw advances.
        let font = data.font_ref().expect("test font must parse");
        let coords: &[swash::NormalizedCoord] = &[];
        let expected: f32 = "AV"
            .chars()
            .map(|ch| {
                font.glyph_metrics(coords)
                    .scale(32.0)
                    .advance_width(font.charmap().map(ch))
            })
            .sum();
        assert!(
            (run.width - expected).abs() < 1e-3,
            "fallback width {} must equal the raw advance sum {expected}",
            run.width,
        );
        assert!(run.width > 0.0);
    }

    /// Degenerate sizes and empty text yield empty runs, never panics or
    /// NaN widths.
    #[test]
    fn shape_text_degenerate_and_empty() {
        let Some((_cache, _key, data)) = test_font_data() else {
            return; // fontless CI: nothing to shape
        };
        for size in [0.0, -4.0, f32::NAN, f32::INFINITY] {
            let run = shape_text(&data, "hi", size);
            assert!(run.glyphs.is_empty(), "size {size} must yield no glyphs");
            assert_eq!(run.width, 0.0, "size {size} must yield zero width");
        }
        let empty = shape_text(&data, "", 32.0);
        assert!(empty.glyphs.is_empty());
        assert_eq!(empty.width, 0.0);
    }

    /// The shape cache hit path must not allocate: the lookup probes the
    /// cache with the caller's borrowed `&str` (via `String: Borrow<str>`)
    /// and clones the text only on a miss. Reuses the crate's shared
    /// per-thread counting allocator from `images::test_alloc`.
    #[test]
    fn shape_text_cache_hit_allocates_nothing() {
        use crate::images::test_alloc::ALLOCATED_BYTES;

        // A vacuous pass here would hide a real regression: the
        // crate embeds a fallback face, so `FontCache` always
        // resolves a default — fail loudly if that ever stops being
        // true instead of silently skipping the test.
        let Some((_cache, _key, data)) = test_font_data() else {
            panic!("shape_text_cache_hit_allocates_nothing requires a font");
        };
        let text = "The quick brown fox jumps over the lazy dog";
        // Miss: shapes and populates the cache.
        let first = shape_text(&data, text, 32.0);
        assert!(!first.glyphs.is_empty(), "test font shapes ASCII");
        // Hit: must return the identical cached run ...
        let before = ALLOCATED_BYTES.get();
        let second = shape_text(&data, text, 32.0);
        let after = ALLOCATED_BYTES.get();
        assert!(
            Rc::ptr_eq(&first, &second),
            "second shape_text must be a cache hit returning the same run"
        );
        // ... without touching the heap.
        assert_eq!(
            after,
            before,
            "cache hit must not allocate ({} bytes)",
            after.saturating_sub(before)
        );
    }

    /// Script segmentation: maximal single-script runs, weak chars
    /// joining the surrounding script.
    #[test]
    fn script_segments_splits_by_script() {
        use swash::text::Script;
        // "a" Latin, U+0645 ARABIC LETTER MEEM, "b" Latin.
        let segs = script_segments("a\u{645}b");
        assert_eq!(
            segs.iter()
                .map(|(s, r)| (*s, r.clone()))
                .collect::<Vec<_>>(),
            vec![
                (Script::Latin, 0..1),
                (Script::Arabic, 1..3),
                (Script::Latin, 3..4),
            ]
        );
        // Leading weak chars join the following strong script.
        let segs = script_segments("1a");
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].0, Script::Latin);
        assert_eq!(segs[0].1, 0..2);
        // Trailing weak chars join the preceding strong script.
        let segs = script_segments("a ");
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].1, 0..2);
        // All-weak text is one Latin segment.
        let segs = script_segments("123");
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].0, Script::Latin);
        assert_eq!(segs[0].1, 0..3);
        // Empty text: no segments.
        assert!(script_segments("").is_empty());
    }

    /// `script_segments` reuses its thread-local scratch across calls: a
    /// long input followed by a short one must not leak stale state.
    #[test]
    fn script_segments_scratch_reuse() {
        use swash::text::Script;
        // Alternating Latin/Arabic single-char runs: exercises the full
        // scratch capacity, then a short call must see cleared buffers.
        let first = script_segments("a\u{645}b\u{645}c\u{645}d");
        assert_eq!(first.len(), 7);
        let segs = script_segments("ab");
        assert_eq!(
            segs.iter()
                .map(|(s, r)| (*s, r.clone()))
                .collect::<Vec<_>>(),
            vec![(Script::Latin, 0..2)]
        );
        let segs = script_segments("\u{645}");
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].0, Script::Arabic);
        assert_eq!(segs[0].1, 0..2);
        // Weak-only still collapses to one Latin segment after reuse.
        let segs = script_segments("1 2");
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].0, Script::Latin);
        assert_eq!(segs[0].1, 0..3);
    }

    /// `push_segment` RTL reverses clusters without per-cluster
    /// allocation: cluster order flips, intra-cluster glyph order is
    /// preserved, ranges are re-based onto the original text.
    #[test]
    fn push_segment_rtl_reverses_clusters() {
        fn g(id: u16, cluster: Range<usize>) -> ShapedGlyph {
            ShapedGlyph {
                glyph_id: id,
                x_offset: 0.0,
                y_offset: 0.0,
                advance: 10.0,
                cluster,
                is_ligature: false,
            }
        }
        // Logical order: [base+mark][single][base+mark].
        let seg = ShapedRun {
            glyphs: vec![g(1, 0..2), g(2, 0..2), g(3, 2..3), g(4, 3..5), g(5, 3..5)],
            width: 50.0,
        };
        let mut run = ShapedRun::default();
        push_segment(&mut run, seg, 100, TextDirection::RightToLeft);
        let ids: Vec<u16> = run.glyphs.iter().map(|g| g.glyph_id).collect();
        assert_eq!(ids, vec![4, 5, 3, 1, 2]);
        let clusters: Vec<Range<usize>> = run.glyphs.iter().map(|g| g.cluster.clone()).collect();
        assert_eq!(
            clusters,
            vec![103..105, 103..105, 102..103, 100..102, 100..102]
        );
        assert_eq!(run.width, 50.0);
        // LTR path: order preserved, ranges re-based.
        let seg = ShapedRun {
            glyphs: vec![g(1, 0..2), g(2, 2..3)],
            width: 20.0,
        };
        let mut run = ShapedRun::default();
        push_segment(&mut run, seg, 7, TextDirection::LeftToRight);
        let ids: Vec<u16> = run.glyphs.iter().map(|g| g.glyph_id).collect();
        assert_eq!(ids, vec![1, 2]);
        let clusters: Vec<Range<usize>> = run.glyphs.iter().map(|g| g.cluster.clone()).collect();
        assert_eq!(clusters, vec![7..9, 9..10]);
        assert_eq!(run.width, 20.0);
        // Empty RTL segment: no panic, width still accumulated.
        let mut run = ShapedRun::default();
        push_segment(
            &mut run,
            ShapedRun {
                glyphs: Vec::new(),
                width: 3.0,
            },
            0,
            TextDirection::RightToLeft,
        );
        assert!(run.glyphs.is_empty());
        assert_eq!(run.width, 3.0);
    }

    /// RTL scripts are detected for direction selection.
    #[test]
    fn rtl_scripts_detected() {
        use swash::text::Script;
        for script in [
            Script::Arabic,
            Script::Hebrew,
            Script::Syriac,
            Script::Thaana,
        ] {
            assert!(is_rtl_script(script), "{script:?} must be RTL");
        }
        for script in [Script::Latin, Script::Greek, Script::Han, Script::Common] {
            assert!(!is_rtl_script(script), "{script:?} must be LTR");
        }
    }

    /// `shape_text` lays RTL runs out in visual order: the glyph sequence
    /// is the logical order reversed by cluster, with the same total
    /// width, so the backends' pen model paints right-to-left.
    #[test]
    fn shape_text_rtl_visual_order() {
        use crate::fonts::FontStyle;
        let cache = FontCache::new();
        let Some(key) = cache.resolve("Noto Sans Arabic", 400, FontStyle::Normal) else {
            return; // fontless CI: nothing to shape
        };
        let data = cache.font_data(key).expect("arabic font data").clone();
        // "marhaba": meem, reh, hah, beh, alef.
        let text = "\u{645}\u{631}\u{62d}\u{628}\u{627}";
        let run = shape_text(&data, text, 32.0);
        assert!(!run.glyphs.is_empty());
        // Visual order: clusters strictly decreasing in source order.
        let mut prev_start = usize::MAX;
        for glyph in &run.glyphs {
            assert!(
                glyph.cluster.start < prev_start,
                "RTL glyphs must run right-to-left in source order, got {:?} after start {prev_start}",
                glyph.cluster,
            );
            prev_start = glyph.cluster.start;
        }
        // Same total advance as the unreordered directional shaping.
        let logical = shape_text_with_direction(&data, text, 32.0, TextDirection::RightToLeft);
        assert_eq!(run.glyphs.len(), logical.glyphs.len());
        assert!((run.width - logical.width).abs() < 0.001);
        // Glyph multiset identical, order reversed by cluster.
        let mut logical_clusters: Vec<(Range<usize>, Vec<u16>)> = Vec::new();
        for glyph in &logical.glyphs {
            match logical_clusters.last_mut() {
                Some(last) if last.0 == glyph.cluster => last.1.push(glyph.glyph_id),
                _ => logical_clusters.push((glyph.cluster.clone(), vec![glyph.glyph_id])),
            }
        }
        let mut visual_clusters: Vec<(Range<usize>, Vec<u16>)> = Vec::new();
        for glyph in &run.glyphs {
            match visual_clusters.last_mut() {
                Some(last) if last.0 == glyph.cluster => last.1.push(glyph.glyph_id),
                _ => visual_clusters.push((glyph.cluster.clone(), vec![glyph.glyph_id])),
            }
        }
        let logical_ids: Vec<Vec<u16>> = logical_clusters
            .into_iter()
            .rev()
            .map(|(_, ids)| ids)
            .collect();
        let visual_ids: Vec<Vec<u16>> = visual_clusters.into_iter().map(|(_, ids)| ids).collect();
        assert_eq!(visual_ids, logical_ids);
    }

    /// Mixed-script text shapes each run with its own script and direction
    /// and concatenates the runs on one pen with segment-relative cluster
    /// ranges re-based onto the original text.
    #[test]
    fn shape_text_mixed_script_segments() {
        use crate::fonts::FontStyle;
        let cache = FontCache::new();
        let Some(key) = cache.resolve("Noto Sans Arabic", 400, FontStyle::Normal) else {
            return; // fontless CI: nothing to shape
        };
        let data = cache.font_data(key).expect("arabic font data").clone();
        // Latin "a", Arabic meem, Latin "b": the Arabic run is shaped
        // with the Arabic script tag (not the dominant Latin one).
        let text = "a\u{645}b";
        let run = shape_text(&data, text, 32.0);
        assert!(!run.glyphs.is_empty());
        // Cluster ranges tile the original text with no gaps or overlaps.
        let mut ranges: Vec<Range<usize>> = run.glyphs.iter().map(|g| g.cluster.clone()).collect();
        ranges.sort_by_key(|r| r.start);
        ranges.dedup();
        let mut pos = 0;
        for range in &ranges {
            assert_eq!(range.start, pos, "gap or overlap in {ranges:?}");
            pos = range.end;
        }
        assert_eq!(pos, text.len());
        // The Arabic glyph is really shaped (nonzero id from the Arabic
        // face), not .notdef from a Latin-script shaper call.
        let arabic_glyph = run
            .glyphs
            .iter()
            .find(|g| g.cluster == (1..3))
            .expect("arabic cluster must be present");
        assert_ne!(arabic_glyph.glyph_id, 0);
        assert!(run.width > 0.0);
    }

    /// `shape_text` caches shaped runs per thread: a repeat call with
    /// the same inputs shares the cached `Rc` (no re-shaping, no new
    /// glyph `Vec`), while different text or size misses.
    #[test]
    fn shape_text_caches_runs() {
        let Some((_cache, _key, data)) = test_font_data() else {
            return; // fontless CI: nothing to shape
        };
        let first = shape_text(&data, "Hello, world!", 32.0);
        assert!(!first.glyphs.is_empty());
        let second = shape_text(&data, "Hello, world!", 32.0);
        assert!(
            Rc::ptr_eq(&first, &second),
            "repeat shaping must hit the cache and share the run"
        );
        // Different text misses.
        let other_text = shape_text(&data, "Hello, world?", 32.0);
        assert!(!Rc::ptr_eq(&first, &other_text));
        // Different size misses.
        let other_size = shape_text(&data, "Hello, world!", 33.0);
        assert!(!Rc::ptr_eq(&first, &other_size));
        // ...but shapes to equal content.
        assert_eq!(first.width, second.width);
        assert_eq!(first.glyphs.len(), second.glyphs.len());
        // Degenerate inputs bypass the cache: a fresh empty run each
        // time, never an entry.
        let empty_a = shape_text(&data, "", 32.0);
        let empty_b = shape_text(&data, "", 32.0);
        assert!(!Rc::ptr_eq(&empty_a, &empty_b));
        assert!(empty_a.glyphs.is_empty() && empty_b.glyphs.is_empty());
    }

    /// Atlas overflow migrates the glyph cache into the fresh page:
    /// glyphs cached before the overflow stay hits (the identical cell,
    /// still owned by the retired generation) instead of re-rasterizing
    /// and re-uploading on next use.
    #[test]
    fn atlas_overflow_migrates_cache() {
        let Some((_cache, key, data)) = test_font_data() else {
            return; // fontless CI: nothing to rasterize
        };
        let Some((device, queue)) = crate::wgpu_be::test_headless_device() else {
            return; // no Vulkan adapter in this environment
        };
        let gid_a = glyph_id_for(&data, 'A');
        let gid_w = glyph_id_for(&data, 'W');
        let mut atlas = Atlas::new(&device);
        let cell_a = atlas
            .get_or_rasterize(&queue, key, &data, gid_a, 24.0)
            .expect("'A' at 24px must rasterize");
        let gen0 = cell_a.generation;
        // Force the overflow: a 1024px cell needs 1025px of shelf.
        let cell_w = atlas
            .get_or_rasterize(&queue, key, &data, gid_w, 1024.0)
            .expect("'W' at 1024px must rasterize");
        assert_ne!(
            cell_w.generation, gen0,
            "the oversized glyph must roll to a fresh generation"
        );
        // Re-requesting 'A' hits the migrated entry: the identical cell
        // (same rect, same retired generation) — without the migration
        // this would re-rasterize into the new generation.
        let cell_a2 = atlas
            .get_or_rasterize(&queue, key, &data, gid_a, 24.0)
            .expect("'A' must still resolve after the overflow");
        assert_eq!(
            cell_a2, cell_a,
            "post-overflow lookup must hit the migrated cache entry, not re-rasterize"
        );
        assert_eq!(cell_a2.generation, gen0);
        // The migrated cell's generation still binds the right texture.
        assert!(
            atlas.view_for_generation(gen0).is_some(),
            "the migrated cell's generation must stay alive until end_frame"
        );
        atlas.end_frame();
    }

    /// Rollover, then `end_frame`, then re-query: a migrated cache entry
    /// whose generation died with the retired page must be treated as a
    /// miss and re-rasterized into the *current* generation. Returning
    /// the stale entry would bind a dropped texture (its
    /// `view_for_generation` is `None`) and leave the glyph permanently
    /// invisible — the cache is never otherwise evicted.
    #[test]
    fn atlas_overflow_end_frame_then_requery_rerasterizes() {
        let Some((_cache, key, data)) = test_font_data() else {
            return; // fontless CI: nothing to rasterize
        };
        let Some((device, queue)) = crate::wgpu_be::test_headless_device() else {
            return; // no Vulkan adapter in this environment
        };
        let gid_a = glyph_id_for(&data, 'A');
        let gid_w = glyph_id_for(&data, 'W');
        let mut atlas = Atlas::new(&device);
        let cell_a = atlas
            .get_or_rasterize(&queue, key, &data, gid_a, 24.0)
            .expect("'A' at 24px must rasterize");
        let gen0 = cell_a.generation;
        // Force the overflow: a 1024px cell needs 1025px of shelf.
        atlas
            .get_or_rasterize(&queue, key, &data, gid_w, 1024.0)
            .expect("'W' at 1024px must rasterize");
        atlas.end_frame();
        // The migrated 'A' entry is now dead (its generation dropped with
        // the retired page); the lookup must drop it and re-rasterize
        // into the current generation.
        let cell_a2 = atlas
            .get_or_rasterize(&queue, key, &data, gid_a, 24.0)
            .expect("'A' must resolve after end_frame");
        assert_ne!(
            cell_a2.generation, gen0,
            "post-end_frame lookup must re-rasterize into the current generation, not return the stale entry"
        );
        assert!(
            atlas.view_for_generation(cell_a2.generation).is_some(),
            "the re-rasterized cell's generation must bind a live texture"
        );
        assert!(
            atlas.texture_for_generation(cell_a2.generation).is_some(),
            "the re-rasterized cell's texture must be alive"
        );
        // And it is now a real cache hit again: same cell, same current
        // generation.
        let cell_a3 = atlas
            .get_or_rasterize(&queue, key, &data, gid_a, 24.0)
            .expect("'A' must stay cached after re-rasterization");
        assert_eq!(cell_a3, cell_a2, "second lookup must hit the cache");
    }

    /// Fill a [`ShapeCache`] directly (bypassing shaping) with live
    /// fonts and distinct `last_used` ticks.
    fn full_cache(slots: usize) -> (ShapeCache, Vec<Arc<Vec<u8>>>) {
        let mut cache = ShapeCache::new();
        let prefix = (1usize, 0usize, 16u32);
        let live: Vec<Arc<Vec<u8>>> = (0..slots).map(|i| Arc::new(vec![i as u8])).collect();
        for (i, font) in live.iter().enumerate() {
            cache.map.entry(prefix).or_default().insert(
                Arc::from(format!("run-{i}")),
                ShapeCacheSlot {
                    run: Rc::new(ShapedRun::default()),
                    font: Arc::downgrade(font),
                    last_used: i as u64,
                },
            );
        }
        (cache, live)
    }

    #[test]
    fn evict_one_drops_least_recently_used() {
        // Full cache, every font alive: the smallest `last_used` goes,
        // everything else stays. Keys are `Arc<str>`, so the victim
        // scan's per-entry clone is a refcount bump, not a string copy.
        let (mut cache, _live) = full_cache(SHAPE_CACHE_CAP);
        assert_eq!(cache.len(), SHAPE_CACHE_CAP);
        cache.evict_one();
        assert_eq!(cache.len(), SHAPE_CACHE_CAP - 1);
        let inner = &cache.map[&(1usize, 0usize, 16u32)];
        assert!(!inner.contains_key("run-0"), "LRU victim must be evicted");
        assert!(inner.contains_key("run-1"), "newer entries must survive");
        assert!(
            inner.contains_key(format!("run-{}", SHAPE_CACHE_CAP - 1).as_str()),
            "newest entry must survive"
        );
    }

    #[test]
    fn evict_one_reclaims_dead_fonts_first() {
        // Entries whose font died can never hit again: they are reclaimed
        // before the LRU victim is chosen (and no LRU eviction happens
        // while the cache is under capacity).
        let mut cache = ShapeCache::new();
        let prefix = (1usize, 0usize, 16u32);
        {
            let dead = Arc::new(vec![0u8]);
            cache.map.entry(prefix).or_default().insert(
                Arc::from("dead"),
                ShapeCacheSlot {
                    run: Rc::new(ShapedRun::default()),
                    font: Arc::downgrade(&dead),
                    last_used: 0,
                },
            );
        } // `dead` drops here; its slot can never hit again.
        let live = Arc::new(vec![1u8]);
        cache.map.entry(prefix).or_default().insert(
            Arc::from("live"),
            ShapeCacheSlot {
                run: Rc::new(ShapedRun::default()),
                font: Arc::downgrade(&live),
                last_used: 1,
            },
        );
        cache.evict_one();
        let inner = &cache.map[&prefix];
        assert!(
            !inner.contains_key("dead"),
            "dead-font entry must be reclaimed"
        );
        assert!(inner.contains_key("live"), "live entry must survive");
    }
}

/// Wave 66 perf tests: the glyph atlas, GPU atlas page, and shape caches
/// moved from `RandomState` (SipHash) to the local FNV-1a hasher.
/// Behavior pins: map semantics are unchanged — only the hash function
/// changed, so lookups return identical values for every key shape used.
#[cfg(test)]
mod w66_p_paint_hash_tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::Instant;

    /// FNV-1a maps agree with SipHash maps on identical insert sequences
    /// for all three cache key shapes (glyph tuple keys, shape-cache
    /// `(usize, usize, u32)` keys, and text strings).
    #[test]
    fn fnv_maps_agree_with_siphash() {
        // CpuGlyphKey/GlyphKey-shaped: (FontKey, glyph id, size px).
        let tuple_keys: Vec<(u32, u32, u16)> = (0..2000)
            .map(|i: u32| (i % 17, i.wrapping_mul(7919) % 5003, (i % 64) as u16))
            .collect();
        let sip: HashMap<_, _> = tuple_keys
            .iter()
            .map(|k| (*k, k.0.wrapping_add(k.1)))
            .collect();
        let fnv: FnvHashMap<_, _> = tuple_keys
            .iter()
            .map(|k| (*k, k.0.wrapping_add(k.1)))
            .collect();
        for k in &tuple_keys {
            assert_eq!(sip.get(k), fnv.get(k), "tuple key {k:?} must agree");
        }
        // Shape-cache outer key: (font address, face index, size bits).
        let shape_keys: Vec<(usize, usize, u32)> = (0..512)
            .map(|i: usize| (i.wrapping_mul(0x9E3779B9), i % 3, (16 << 20) + i as u32))
            .collect();
        let sip: HashMap<_, _> = shape_keys.iter().map(|k| (*k, k.2)).collect();
        let fnv: FnvHashMap<_, _> = shape_keys.iter().map(|k| (*k, k.2)).collect();
        for k in &shape_keys {
            assert_eq!(sip.get(k), fnv.get(k), "shape key {k:?} must agree");
        }
        // Inner text keys (Arc<str> probes through Borrow<str>).
        let texts: Vec<Arc<str>> = (0..512)
            .map(|i: usize| Arc::from(format!("glyph run {i} with shaping")))
            .collect();
        let sip: HashMap<_, _> = texts.iter().map(|t| (t.clone(), t.len())).collect();
        let fnv: FnvHashMap<_, _> = texts.iter().map(|t| (t.clone(), t.len())).collect();
        for t in &texts {
            let probe: &str = t;
            assert_eq!(sip.get(probe), fnv.get(probe), "text key must agree");
        }
    }

    /// Micro-benchmark: FNV-1a vs SipHash `get` on glyph-key-sized
    /// structs and on shaped-text strings. Prints ns/get; asserts nothing
    /// (shared CI boxes are too noisy for timing asserts) — the numbers
    /// are the regression harness for this swap.
    #[test]
    fn fnv_vs_siphash_get_timing() {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        struct GlyphProbe {
            font: u32,
            glyph_id: u16,
            size_px: u16,
        }
        let keys: Vec<GlyphProbe> = (0..4000)
            .map(|i: u32| GlyphProbe {
                font: i % 13,
                glyph_id: (i.wrapping_mul(7919) % 6000) as u16,
                size_px: 16,
            })
            .collect();
        let sip: HashMap<GlyphProbe, u32> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| (*k, i as u32))
            .collect();
        let fnv: FnvHashMap<GlyphProbe, u32> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| (*k, i as u32))
            .collect();
        let time_get = |get: &dyn Fn(&GlyphProbe) -> u32| {
            let start = Instant::now();
            let mut acc = 0u64;
            let n = 50 * keys.len();
            for i in 0..n {
                acc += u64::from(get(&keys[i % keys.len()]));
            }
            let elapsed = start.elapsed();
            (elapsed.as_nanos() as f64 / n as f64, acc)
        };
        let (sip_ns, sip_acc) = time_get(&|k| *sip.get(k).unwrap_or(&0));
        let (fnv_ns, fnv_acc) = time_get(&|k| *fnv.get(k).unwrap_or(&0));
        assert_eq!(sip_acc, fnv_acc, "both maps return the same values");
        println!(
            "w66 glyph-tuple keys: SipHash {sip_ns:.1}ns/get vs FNV-1a {fnv_ns:.1}ns/get ({:.2}x)",
            sip_ns / fnv_ns.max(0.01)
        );
    }
}
