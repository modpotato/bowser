//! Resource loading pipeline: document fetch, stylesheets, images.
//!
//! `load_document_with_user_agent` turns a URL into a [`LoadedDocument`]: the parsed DOM,
//! the author stylesheets in document order, intrinsic image sizes for
//! `bowser_layout::layout_with_intrinsics`, and decoded image pixels in
//! the loader image cache (see [`cached_image`]) for paint.
//!
//! # Blocking v1
//!
//! Every network round-trip here blocks the calling thread. Call
//! `load_document_with_user_agent` on a worker thread, never the UI thread. The intended
//! pattern for the navigation worker:
//!
//! ```rust,ignore
//! // The profile owns one jar, shared across every load on every thread:
//! let jar: Arc<Mutex<CookieJar>> = profile.cookie_jar();
//! let (tx, rx) = std::sync::mpsc::channel();
//! let url = url.clone();
//! std::thread::spawn(move || {
//!     // `load_document_with_user_agent` blocks until the document, its stylesheets, and
//!     // its image headers have all arrived (or failed).
//!     let _ = tx.send(loader::load_document_with_user_agent(
//!         &url,
//!         &jar,
//!         bowser_net::fetch::USER_AGENT,
//!     ));
//! });
//! // ... later, on the navigation task:
//! let loaded = rx.recv().unwrap()?;
//! ```
//!
//! The jar is a `Mutex<CookieJar>`, not a `RefCell`: loads happen on worker
//! threads, and cookies must be shared across them. The lock is taken once
//! per request — the document fetch, then each subresource fetch — and is
//! never held across parsing or between requests, so loads on other
//! threads are never serialized behind a whole page load.
//!
//! Every request, the document and each subresource alike, carries the
//! same `User-Agent` (see [`load_document_with_user_agent`]). Subresource
//! fetches run in the document's `SameSite` top context (not as top-level
//! navigations), so `Lax` cookies are not over-sent on cross-site redirect
//! chains.
//!
//! # Pipeline
//!
//! 1. GET the document URL with `fetch_with_trace` (redirects followed,
//!    cookies sent and stored via the shared jar, one lock per request).
//!    A non-2xx final status is `LoadError::HttpStatus`.
//! 2. Content-Type gate: only `text/html` proceeds. A missing Content-Type
//!    is sniffed (body starts with `<` after whitespace); anything else is
//!    `LoadError::UnsupportedContentType`.
//! 3. `Document::parse_html`, then one document-order walk collects `<style>`
//!    text and `<link rel=stylesheet href>` URLs. `<base href>` (the first
//!    one, in document order) sets the base URL for resolving relative
//!    references. Linked stylesheets are fetched sequentially in document
//!    order, capped at [`MAX_STYLESHEETS`]; a failed CSS fetch skips that
//!    sheet without failing the page.
//! 4. `<img src>` URLs are resolved (capped at [`MAX_IMAGES`]) and each is
//!    fetched with `Range` requests: the first asks for bytes
//!    `0..`[`IMAGE_FETCH_BYTES`]. A server that honors `Range` answers
//!    206 and the loader follows up with further range requests until the
//!    whole body has arrived (or bails when `Content-Range` advertises
//!    more than [`IMAGE_MAX_BYTES`]); a server that ignores `Range`
//!    answers 200 with the full body (already capped at
//!    [`IMAGE_MAX_BYTES`] by `bowser_net`'s body cap). The complete bytes
//!    go through `bowser_paint::images::decode_image`: success records
//!    intrinsic sizes by `NodeId` for
//!    `bowser_layout::layout_with_intrinsics` and inserts the decoded
//!    pixels into the loader image cache (see [`cached_image`]) for
//!    paint, subject to [`IMAGE_CACHE_BUDGET_BYTES`]. Any failure — fetch
//!    error, non-2xx status, an over-cap body, or decode failure — records
//!    nothing and the element lays out as a 0x0 box. `srcset` is ignored
//!    in v1. CSS `background-image`s are prefetched the same way (see
//!    step 4's `collect_background_image_urls`): paint resolves them
//!    through the same cache, so without prefetching they would always
//!    miss. They share the [`MAX_IMAGES`] cap with `<img>` but record no
//!    intrinsics.
//! 5. Scripts: one document-order walk collects every `<script>` element
//!    ([`crate::script::collect_script_refs`]); external classic sources
//!    are prefetched sequentially, capped at [`crate::script::MAX_SCRIPTS`]
//!    ([`crate::script_executor::prefetch_script_sources`]), in the
//!    document's SameSite top context with the loader's `User-Agent` —
//!    the same fetch discipline as stylesheets and images. Modules, data
//!    blocks, bad URLs, and failed or over-cap fetches are recorded in
//!    the prefetch map; a script never fails the page.
//!    [`LoadedDocument::script_sources`] carries the map; the install
//!    re-parses the body with the page's [`crate::script::PageVm`]
//!    attached mid-parse, so parser-inserted scripts execute during tree
//!    construction and serve external sources from the map.
//!
//! # The image cache
//!
//! Decoded pixels live in a loader-level (process-wide) cache, not in
//! [`LoadedDocument`]: `LoadedDocument` has construction sites outside
//! this module that a new field would break, and a per-navigation map
//! would re-decode assets shared across pages. The cache is keyed by
//! serialized *request* URL (the resolved `src`, before redirects), so
//! paint can match the raw `src` attribute resolved against the base URL
//! without replaying redirect chains. It is bounded by
//! [`IMAGE_CACHE_BUDGET_BYTES`] with FIFO eviction; images decoded past
//! the budget still record intrinsics, only their pixels are dropped
//! (paint shows the placeholder).
//!
//! # Future work (documented, not done)
//!
//! - Fetch linked stylesheets in parallel: one thread per sheet sharing
//!   `Arc<Mutex<CookieJar>>`, joining results back in document order.
//! - `srcset`/`sizes` selection for `<img>`.
//! - Image cache revalidation: the cache never revalidates in v1 (no
//!   ETag/Last-Modified tracking) and is shared across profiles; a future
//!   worker should add cache-control awareness and per-profile isolation.

use bowser_css::{
    CssValue, Styler, Stylesheet, Viewport, default_ua_stylesheet, parse_stylesheet, props,
};
use bowser_dom::{Document, NodeId, NodeKind};
use bowser_layout::IntrinsicDims;
use bowser_net::cookies::CookieJar;
use bowser_net::fetch::{
    FetchError, FetchTrace, NavigationContext, Request, Response, fetch_with_jar_in_context,
    fetch_with_trace,
};
use bowser_paint::fonts::{FontCache, FontStyle, decode_webfont};
use bowser_paint::images::{DecodedImage, decode_image};
use bowser_paint::svg::svg_intrinsics;
use bowser_url::{Origin, Url, base64_decode, cors_check, percent_decode};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, LazyLock, Mutex};

use crate::script::{collect_script_refs, fetch_script_source};
use crate::script_executor::prefetch_script_sources;

/// Maximum `<link rel=stylesheet>` sheets fetched per document. Extra
/// links are skipped; inline `<style>` text is always free (no fetch).
const MAX_STYLESHEETS: usize = 128;
/// Maximum images fetched per document (`<img>` plus CSS
/// `background-image`s). Extra images lay out as 0x0 boxes / placeholders.
const MAX_IMAGES: usize = 256;
/// First image byte window fetched per image (see [`load_image`]).
/// Servers that honor `Range` get follow-up range requests past this
/// window until the whole body arrives; servers that ignore `Range`
/// transfer the full body (up to `bowser_net`'s body cap) on the first
/// request.
const IMAGE_FETCH_BYTES: usize = 64 * 1024;
/// Hard cap on the total bytes assembled for one image across all its
/// range follow-ups. Matches `bowser_net`'s body cap, so a 200 response
/// (server ignored `Range`) is already bounded; a 206 whose `Content-Range`
/// advertises more than this is abandoned before the follow-ups start.
const IMAGE_MAX_BYTES: usize = 32 * 1024 * 1024;
/// Maximum `@font-face` fonts fetched per document. Extra rules are
/// skipped; a rule that fails every `src` entry records nothing.
const MAX_WEBFONTS: usize = 64;
/// Hard cap on one fetched font's bytes. Matches `bowser_net`'s body
/// cap, so an HTTP body is already bounded; `data:` fonts and any
/// future fetch path are checked explicitly against the same cap.
const FONT_MAX_BYTES: usize = 32 * 1024 * 1024;
/// Aggregate budget for one document's webfonts: fetching stops once
/// the unique DECODED bytes (reused bytes count once, since the cache
/// shares one `Arc` per file) pass this total, so a page with dozens
/// of max-size faces cannot retain gigabytes. Decoded, not fetched,
/// is the counted quantity, deliberately: the cache retains the
/// decoded SFNT, and many small highly-compressible WOFF1 files could
/// otherwise decode to far more than the budget while staying under
/// it as fetched bytes. The per-font 32 MiB decoded cap (enforced by
/// `decode_webfont` in `bowser_paint`) bounds single fonts; this
/// budget bounds their aggregate. 192 MiB sits inside the 128-256 MiB
/// band: far above any realistic page (a handful of ~100 KiB faces),
/// far below the 64-face x 32 MiB worst case.
const WEBFONT_BUDGET_BYTES: usize = 192 * 1024 * 1024;
/// Maximum `@import` nesting depth for the font-fetch walk (see
/// [`collect_imported_sheets`]): imported sheets at depth 1..=4 are
/// fetched; deeper imports are ignored, and the seen-URL set breaks
/// cycles at any depth.
const MAX_IMPORT_DEPTH: usize = 4;
/// Cap on imported sheets fetched for one document's fonts, matching
/// [`MAX_STYLESHEETS`] in spirit: an adversarial import fan-out cannot
/// grow the font walk without bound.
const MAX_IMPORTED_SHEETS: usize = 128;
/// [`cached_image`]). Without it, 256 max-size images could cache
/// gigabytes; with it, memory stays bounded per the crate's cleanliness
/// rules. Images decoded past the budget still record intrinsics — only
/// their pixels are dropped, so paint shows the placeholder.
const IMAGE_CACHE_BUDGET_BYTES: usize = 256 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Loader image cache
// ---------------------------------------------------------------------------

/// Process-wide decoded-image cache, filled by [`load_document_with_user_agent`] and read
/// by paint (see [`cached_image`]).
///
/// Keyed by serialized *request* URL (the resolved `src`, before
/// redirects): that is the URL paint can reconstruct from the raw `src`
/// attribute plus the base URL, without replaying redirect chains.
/// Bounded by [`IMAGE_CACHE_BUDGET_BYTES`] with FIFO eviction, so the
/// cache cannot grow without limit. v1 limitations, documented for the
/// next worker: entries are never revalidated (no ETag/Last-Modified
/// tracking) and the cache is shared across profiles.
struct ImageCache {
    entries: HashMap<String, DecodedImage>,
    /// Insertion order for FIFO eviction.
    order: VecDeque<String>,
    /// Sum of `rgba.len()` over `entries`.
    bytes: usize,
}

impl ImageCache {
    fn new() -> Self {
        ImageCache {
            entries: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
        }
    }

    fn insert(&mut self, key: String, image: DecodedImage) {
        self.insert_budgeted(key, image, IMAGE_CACHE_BUDGET_BYTES);
    }

    fn insert_budgeted(&mut self, key: String, image: DecodedImage, budget: usize) {
        let px = image.rgba.len();
        // A single image larger than the whole budget is never cached;
        // its intrinsics were still recorded by the caller.
        if px > budget {
            return;
        }
        if let Some(old) = self.entries.insert(key.clone(), image) {
            self.bytes -= old.rgba.len();
            self.order.retain(|k| k != &key);
        }
        self.order.push_back(key);
        self.bytes += px;
        while self.bytes > budget {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(evicted) = self.entries.remove(&oldest) {
                self.bytes -= evicted.rgba.len();
            }
        }
    }

    fn get(&self, key: &str) -> Option<DecodedImage> {
        self.entries.get(key).cloned()
    }
}

static IMAGE_CACHE: LazyLock<Mutex<ImageCache>> = LazyLock::new(|| Mutex::new(ImageCache::new()));

/// Look up decoded pixels cached by [`load_document_with_user_agent`].
///
/// `serialized_url` is the image's request URL in serialized form — i.e.
/// `bowser_url::parse(src, Some(&base)).serialize()`, where `base` is the
/// document base URL. Returns a clone of the cached [`DecodedImage`]; the
/// cache owns its copy, so callers can hand the pixels to paint freely.
///
/// This is the read half paint's image resolver uses; the write half is
/// the image step of [`load_document_with_user_agent`].
pub fn cached_image(serialized_url: &str) -> Option<DecodedImage> {
    IMAGE_CACHE
        .lock()
        .expect("loader: image cache mutex poisoned")
        .get(serialized_url)
}

/// Look up the cached decode for one `<img>` element's raw `src`, honoring
/// `<base href>` exactly like [`load_document_with_user_agent`]'s image step does.
///
/// `document_url` is the document's own URL (before `<base href>` is
/// applied); `src` is resolved against [`effective_base_url`] and the
/// resulting serialized request URL is used as the cache key — the same
/// key the loader wrote. Returns `None` when `src` does not resolve or no
/// decode was cached (fetch/decode failure, eviction, or the image was
/// never loaded).
///
/// This is the lookup paint's image resolver should use: it keeps `<base
/// href>` handling in one place instead of re-resolving `src` at the
/// call site.
pub fn cached_image_for_src(
    document: &Document,
    document_url: &Url,
    src: &str,
) -> Option<DecodedImage> {
    let base = effective_base_url(document, document_url);
    let url = bowser_url::parse(src, Some(&base)).ok()?;
    cached_image(&url.serialize())
}

/// Test-only: seed the decoded-image cache for `serialized_url`.
#[cfg(test)]
pub(crate) fn prime_image_cache(serialized_url: &str, image: DecodedImage) {
    IMAGE_CACHE
        .lock()
        .expect("loader: image cache mutex poisoned")
        .insert(serialized_url.to_string(), image);
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Everything a navigation needs after the load phase: the final document
/// plus the subresources layout and paint consume.
#[derive(Debug)]
pub struct LoadedDocument {
    /// Final URL after following redirects.
    pub url: Url,
    /// Every URL actually fetched, in order (initial URL first).
    //
    // (Read by the browser worker once it lands: the allow keeps
    // `cargo clippy -- -D warnings` green until then.)
    #[allow(dead_code)]
    pub trace: FetchTrace,
    /// The parsed DOM.
    pub document: Document,
    /// Author stylesheets in document order: `<style>` text and fetched
    /// `<link rel=stylesheet>` sheets interleaved as they appear.
    pub stylesheets: Vec<Stylesheet>,
    /// Intrinsic image dimensions by serialized `<img>` request URL
    /// (the resolved `src`, before redirects), for
    /// `bowser_layout::layout_with_intrinsics`. Raster images record
    /// their decoded size; SVGs record `width`/`height`/`viewBox`
    /// (unknown components are `None`, so layout falls back to the
    /// 300x150 default and/or the `viewBox` ratio). Images that failed
    /// to fetch or decode are absent (0x0 box). Keyed by URL rather
    /// than node because the install re-parses the document with the
    /// page's `PageVm` (parser-inserted scripts run mid-parse), which
    /// makes node IDs from the loader's discovery parse stale; the
    /// install re-keys by URL when it walks the scripted document.
    pub image_sizes: HashMap<String, IntrinsicDims>,
    /// Prefetched external classic script sources by serialized URL,
    /// mirroring [`crate::script::resolve_scripts`]' gating: modules and
    /// data blocks are never fetched, and failed or over-cap fetches are
    /// recorded as errors. The mid-parse executor serves `src` scripts
    /// from this map; the post-parse path resolves the declined scripts
    /// against the same map. Fetched in step 5 of
    /// [`load_document_with_user_agent`].
    pub script_sources: HashMap<String, Result<String, String>>,
    /// The raw response body the document was parsed from. The install
    /// re-parses it with the page's `PageVm` attached mid-parse, so
    /// parser-inserted scripts execute during tree construction on the
    /// installing thread (the VM is `!Send`).
    pub body: Vec<u8>,
    /// Fonts fetched for the document's `@font-face` rules, in rule
    /// order (see `fetch_webfonts`). `load_document_with_fonts`
    /// already installed these into the given cache during the load;
    /// callers that used the plain loader get them installed into the
    /// window's cache by [`crate::navigation::install_loaded_document`]
    /// before first layout — installing twice is harmless, since the
    /// owned install dedupes by (bytes, family, weight, style).
    /// A rule whose every `src` entry failed is absent, and the page
    /// falls back to its next `font-family` entry as usual.
    pub webfonts: Vec<Webfont>,
}

/// One font fetched for an `@font-face` rule: the rule's descriptors
/// plus the fetched bytes. The bytes are `Arc`-shared with the font
/// cache once installed, so handing them to the installer is a
/// refcount bump, not a copy.
//
// Weight/style are retained here (not yet consumed: this slice's
// install indexes the face's own OS/2 weight/style, as `install_font`
// always has) so the face-matching slice can use the rule's
// descriptors without re-fetching; the allow keeps that staging
// explicit instead of warning.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Webfont {
    /// The rule's `font-family` descriptor, as authored.
    pub family: String,
    /// The rule's `font-weight` descriptor (single weight, or a range's
    /// start; see `FontFaceRule::font_weight`).
    pub weight: u16,
    /// The full range of a range `font-weight` descriptor, if authored.
    pub weight_range: Option<(u16, u16)>,
    /// The rule's `font-style` descriptor as a lowercase keyword.
    pub style: String,
    /// The decoded SFNT font bytes. WOFF1 sources are decoded by
    /// `fetch_webfonts` before storage, so the cache always holds
    /// SFNT; woff2 and garbage are rejected at fetch time (the next
    /// `src` entry is tried instead).
    pub bytes: Arc<[u8]>,
}

/// Every way a document load can fail.
#[derive(Debug)]
pub enum LoadError {
    /// The document fetch itself failed (DNS, TLS, too many redirects...).
    Fetch(FetchError),
    /// The document fetch completed with a non-2xx status (3xx chains are
    /// already followed by fetch, so this is 4xx/5xx in practice).
    HttpStatus(u16),
    /// The document is not HTML: carries the offending media type, or a
    /// note when the Content-Type was missing and the body did not sniff
    /// as HTML.
    UnsupportedContentType(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fetch(e) => write!(f, "document fetch failed: {e}"),
            Self::HttpStatus(s) => write!(f, "document fetch returned HTTP {s}"),
            Self::UnsupportedContentType(t) => {
                write!(f, "unsupported document content type: {t}")
            }
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Fetch(e) => Some(e),
            _ => None,
        }
    }
}

impl From<FetchError> for LoadError {
    fn from(e: FetchError) -> Self {
        Self::Fetch(e)
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Load a document and its subresources, blocking.
///
/// `jar` is the profile's shared cookie jar; it is locked once per
/// request and never held across parsing or between requests, so other
/// threads' loads are not serialized behind this one. Run this on a
/// worker thread (see the module docs).
///
/// Every request — the document and each subresource — carries
/// `user_agent` as its `User-Agent` header.
///
/// `@font-face` fonts are fetched (see `fetch_webfonts`) and returned
/// in `LoadedDocument::webfonts`, but NOT installed here: the install
/// path ([`crate::navigation::install_loaded_document`]) installs them
/// into the window's font cache before first layout.
pub fn load_document_with_user_agent(
    url: &Url,
    jar: &Mutex<CookieJar>,
    user_agent: &str,
) -> Result<LoadedDocument, LoadError> {
    load_document_inner(url, jar, user_agent, None)
}

/// Like `load_document_with_user_agent`, but installs the fetched
/// `@font-face` fonts into `fonts` synchronously, before returning — so
/// the first layout after the load measures with the webfonts. The
/// fonts are also returned in `LoadedDocument::webfonts`; callers
/// using this entry point need not install them again (a second
/// install is harmless: the owned install dedupes by (bytes, family,
/// weight, style)).
//
// (Not yet called outside tests: the allow keeps
// `cargo clippy -- -D warnings` green until the navigation worker
// wires this entry point into `run_loader`.)
#[allow(dead_code)]
pub fn load_document_with_fonts(
    url: &Url,
    jar: &Mutex<CookieJar>,
    user_agent: &str,
    fonts: &Mutex<FontCache>,
) -> Result<LoadedDocument, LoadError> {
    load_document_inner(url, jar, user_agent, Some(fonts))
}

fn load_document_inner(
    url: &Url,
    jar: &Mutex<CookieJar>,
    user_agent: &str,
    fonts: Option<&Mutex<FontCache>>,
) -> Result<LoadedDocument, LoadError> {
    // 1. Document fetch. Redirects are followed inside fetch; the trace
    //    records every hop. The jar is locked only for this request.
    let request = get_request(url, user_agent, Vec::new());
    let (response, trace) = {
        let mut guard = jar.lock().expect("loader: cookie jar mutex poisoned");
        fetch_with_trace(&request, Some(&mut guard))?
    };
    if !(200..300).contains(&response.status) {
        return Err(LoadError::HttpStatus(response.status));
    }
    check_html_content_type(&response)?;

    // 2. Parse.
    let (document, _quirks) = Document::parse_html(&response.body);
    let final_url = response.url.clone();
    let base = effective_base_url(&document, &final_url);

    // Subresource fetches share the document's SameSite top context (not
    // top-level navigation), so Lax cookies are not over-sent on
    // cross-site redirect chains.
    let sub = SubresourceFetch {
        jar,
        user_agent,
        document_origin: final_url.origin(),
        nav: NavigationContext {
            top_host: final_url.host().map(str::to_owned),
            top_secure: Some(final_url.scheme() == "https"),
            top_level_safe_navigation: false,
        },
    };

    // 3. Stylesheets, document order: <style> text and <link> sheets
    //    interleaved as they appear. Linked sheets are fetched
    //    sequentially, capped at MAX_STYLESHEETS; a failed sheet is
    //    skipped without failing the page.
    let mut stylesheets = Vec::new();
    let mut sheet_bases: Vec<Url> = Vec::new();
    // Raw text and (for linked sheets) URL per sheet, parallel to
    // `stylesheets`: the @import walk below needs the text the CSS
    // parser discarded, and the cycle set needs the sheet's own URL.
    let mut sheet_texts: Vec<String> = Vec::new();
    let mut sheet_urls: Vec<Option<Url>> = Vec::new();
    let mut sheets_fetched = 0usize;
    for source in collect_style_sources(&document, &base) {
        match source {
            StyleSource::Inline(css) => {
                // An inline <style> sheet resolves its @font-face src
                // URLs against the document base, exactly like its
                // other relative references.
                sheet_bases.push(base.clone());
                stylesheets.push(parse_stylesheet(&css));
                sheet_texts.push(css);
                sheet_urls.push(None);
            }
            StyleSource::Linked(sheet_url) => {
                if sheets_fetched >= MAX_STYLESHEETS {
                    continue;
                }
                sheets_fetched += 1;
                if let Some((sheet, text)) = fetch_stylesheet(&sheet_url, &sub) {
                    // A linked sheet resolves its @font-face src URLs
                    // against ITS OWN url — never the document base
                    // (the background-image step's documented v1
                    // limitation, not repeated here: the sheet URL is
                    // threaded through this loop alongside the sheet).
                    sheet_bases.push((*sheet_url).clone());
                    stylesheets.push(sheet);
                    sheet_texts.push(text);
                    sheet_urls.push(Some((*sheet_url).clone()));
                }
            }
        }
    }

    // 3b. @font-face fonts, synchronously, before first layout: walk
    //    every sheet's font_faces() in document order, resolve each
    //    src against that sheet's URL (see the loop above), fetch via
    //    the same bowser_net path stylesheets use, try src entries in
    //    order, and — for the with-fonts entry point — install into
    //    the window's FontCache here, on the load thread, so no async
    //    arrival or reflow machinery is involved in this slice. A
    //    failed font never fails the page. Sheets reached through
    //    @import are walked for fonts only (their style rules stay
    //    out of `tab.author`: the CSS parser drops @import, so the
    //    cascade never saw them either — this slice only stops
    //    dropping their fonts).
    let mut font_sheets = stylesheets.clone();
    let mut font_bases = sheet_bases.clone();
    {
        let mut seen: HashSet<String> = HashSet::new();
        let mut imported: Vec<(Stylesheet, Url)> = Vec::new();
        for ((text, base), own_url) in sheet_texts
            .iter()
            .zip(sheet_bases.iter())
            .zip(sheet_urls.iter())
        {
            if let Some(own) = own_url {
                seen.insert(own.serialize());
            }
            collect_imported_sheets(text, base, &sub, 0, &mut seen, &mut imported);
        }
        for (sheet, sheet_base) in imported {
            font_bases.push(sheet_base);
            font_sheets.push(sheet);
        }
    }
    let webfonts = fetch_webfonts(&font_sheets, &font_bases, &sub);
    if let Some(fonts) = fonts {
        let mut guard = fonts.lock().expect("loader: font cache mutex poisoned");
        for webfont in &webfonts {
            // The rule's descriptors, not the file's internals: a
            // renamed rule must resolve under its declared family.
            guard.install_font_owned_as(
                webfont.bytes.clone(),
                &webfont.family,
                webfont.weight,
                FontStyle::from_descriptor(&webfont.style),
            );
        }
    }

    // 4. Images: resolve, fetch, decode, cache. One range-fetch per
    //    image; any failure records nothing (0x0 box + placeholder).
    //    Repeated URLs share one fetch/decode via `fetched`, and a second
    //    load of an image already in the shared cache skips the fetch.
    //    `<img src>`s come first in document order so their intrinsics
    //    land in `image_sizes`; CSS `background-image`s follow —
    //    paint resolves those through the same cache (keyed by serialized
    //    request URL), so without this step they would always miss.
    let mut image_sizes: HashMap<String, IntrinsicDims> = HashMap::new();
    let mut fetched: HashMap<String, Option<LoadedImage>> = HashMap::new();
    let mut image_urls: Vec<(Option<NodeId>, Url)> = collect_images(&document, &base)
        .into_iter()
        .map(|(node, url)| (Some(node), url))
        .collect();
    image_urls.extend(
        collect_background_image_urls(&document, &stylesheets, &base)
            .into_iter()
            .map(|url| (None, url)),
    );
    for (node, img_url) in image_urls.into_iter().take(MAX_IMAGES) {
        let key = img_url.serialize();
        let loaded = fetch_and_cache_image(&key, &img_url, &sub, &mut fetched);
        let Some(loaded) = loaded else { continue };
        // Intrinsics are recorded for `<img>`s only, keyed by serialized
        // request URL; background images do not size their element. The
        // install re-parses the document with the page's `PageVm`
        // (parser-inserted scripts run mid-parse), so node IDs from this
        // discovery parse would be stale there — the install re-keys by
        // URL when it walks the scripted document.
        if node.is_some() {
            image_sizes.insert(key, loaded.dims);
        }
    }

    // 5. Scripts: prefetch external classic sources the page may run.
    //    The install re-parses the document with the page's own `PageVm`
    //    attached mid-parse (the VM is `!Send`, so it must be created on
    //    the installing thread), and the mid-parse executor serves
    //    external sources from this map — no network happens there.
    //    Modules, data blocks, bad URLs, and failed or over-cap fetches
    //    are recorded as errors in the map; the post-parse path turns
    //    them into recorded skips. A script never fails the page.
    let script_sources = prefetch_script_sources(&collect_script_refs(&document), &base, |url| {
        fetch_script_source(url, sub.jar, sub.user_agent, &sub.nav)
    });

    Ok(LoadedDocument {
        url: final_url,
        trace,
        document,
        stylesheets,
        image_sizes,
        script_sources,
        body: response.body,
        webfonts,
    })
}

/// Shared settings for one load's subresource fetches.
struct SubresourceFetch<'a> {
    /// Profile jar; locked once per request, never across requests.
    jar: &'a Mutex<CookieJar>,
    user_agent: &'a str,
    /// The document's origin (from the final document URL, after
    /// redirects): the CORS reference for [`fetch_font_bytes`].
    /// Cross-origin font fetches deliberately bypass `jar` (see
    /// there), so this is a full origin, not just the `nav` top host.
    document_origin: Origin,
    /// Document-host SameSite context (not a top-level navigation).
    nav: NavigationContext,
}

/// A GET request carrying the loader's `User-Agent` plus `extra` headers.
fn get_request(url: &Url, user_agent: &str, extra: Vec<(String, String)>) -> Request {
    let mut headers = Vec::with_capacity(extra.len() + 1);
    headers.push(("User-Agent".to_string(), user_agent.to_string()));
    headers.extend(extra);
    Request {
        method: "GET".to_string(),
        url: url.clone(),
        headers,
        body: None,
    }
}

/// Absolute URLs of external scripts (`<script src>`) in document order.
///
/// This is the cheap enumeration for callers that only need the URLs; the
/// loader's own script step (step 5 of [`load_document_with_user_agent`])
/// uses [`crate::script_executor::prefetch_script_sources`] instead, so it
/// also classifies inline scripts, modules, and data blocks and fetches
/// the sources.
//
// (Not yet called: the allow keeps `cargo clippy -- -D warnings` green
// until the bindings worker wires it up.)
#[allow(dead_code)]
pub fn script_srcs(document: &Document, base_url: &Url) -> Vec<Url> {
    let mut out = Vec::new();
    for node in document.descendants(document.root()) {
        let NodeKind::Element(data) = document.kind(node) else {
            continue;
        };
        if data.tag_name == "script"
            && let Some(src) = document.get_attribute(node, "src")
            && let Ok(url) = bowser_url::parse(src, Some(base_url))
        {
            out.push(url);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Document fetch helpers
// ---------------------------------------------------------------------------

/// Reject non-HTML documents. A missing Content-Type falls back to
/// sniffing: the body must start with `<` (after ASCII whitespace).
fn check_html_content_type(response: &Response) -> Result<(), LoadError> {
    match response.header("content-type") {
        Some(ct) => {
            let media_type = ct.split(';').next().unwrap_or("").trim();
            if media_type.eq_ignore_ascii_case("text/html") {
                Ok(())
            } else {
                Err(LoadError::UnsupportedContentType(media_type.to_string()))
            }
        }
        None if sniff_html(&response.body) => Ok(()),
        None => Err(LoadError::UnsupportedContentType(
            "missing content-type; body does not sniff as HTML".to_string(),
        )),
    }
}

fn sniff_html(body: &[u8]) -> bool {
    body.iter()
        .copied()
        .find(|b| !b.is_ascii_whitespace())
        .is_some_and(|b| b == b'<')
}

/// The base URL for resolving relative subresource references: the first
/// `<base href>` in document order, else the document's final URL.
///
/// Also the rule for resolving clicked link hrefs: `window::link_base_url`
/// delegates here so the rule lives in exactly one place.
pub(crate) fn effective_base_url(document: &Document, document_url: &Url) -> Url {
    for node in document.descendants(document.root()) {
        let NodeKind::Element(data) = document.kind(node) else {
            continue;
        };
        if data.tag_name == "base"
            && let Some(href) = document.get_attribute(node, "href")
            && !href.trim().is_empty()
        {
            // Only the first <base href> counts; if it fails to parse, the
            // document URL stays the base.
            return bowser_url::parse(href, Some(document_url))
                .unwrap_or_else(|_| document_url.clone());
        }
    }
    document_url.clone()
}

// ---------------------------------------------------------------------------
// Stylesheets
// ---------------------------------------------------------------------------

/// A stylesheet source in document order: inline text or a URL to fetch.
///
/// `Linked` boxes its `Url`: `Url` is ~10x the size of `String`, and this
/// enum is built once per document (never a hot path), so the box keeps
/// the enum small.
enum StyleSource {
    Inline(String),
    Linked(Box<Url>),
}

fn collect_style_sources(document: &Document, base: &Url) -> Vec<StyleSource> {
    let mut out = Vec::new();
    for node in document.descendants(document.root()) {
        let NodeKind::Element(data) = document.kind(node) else {
            continue;
        };
        match data.tag_name.as_str() {
            "style" => out.push(StyleSource::Inline(document.text_content(node))),
            "link" => {
                if is_stylesheet_link(document, node)
                    && let Some(href) = document.get_attribute(node, "href")
                    && let Ok(url) = bowser_url::parse(href, Some(base))
                {
                    // Unparseable hrefs are skipped, not fatal.
                    out.push(StyleSource::Linked(Box::new(url)));
                }
            }
            _ => {}
        }
    }
    out
}

/// `<link>` whose `rel` list contains the `stylesheet` token
/// (ASCII case-insensitive, per the HTML spec).
fn is_stylesheet_link(document: &Document, node: NodeId) -> bool {
    document.get_attribute(node, "rel").is_some_and(|rel| {
        rel.split(|c: char| c.is_ascii_whitespace())
            .any(|token| token.eq_ignore_ascii_case("stylesheet"))
    })
}

/// Fetch and parse one linked stylesheet, keeping its raw text: the
/// CSS parser drops `@import`, so the font walk needs the text to
/// find imported sheets. `None` on any failure: the sheet is skipped
/// and the page load continues. The jar is locked only for the fetch
/// itself.
fn fetch_stylesheet(url: &Url, sub: &SubresourceFetch<'_>) -> Option<(Stylesheet, String)> {
    let request = get_request(url, sub.user_agent, Vec::new());
    let response = {
        let mut guard = sub.jar.lock().expect("loader: cookie jar mutex poisoned");
        fetch_with_jar_in_context(&request, Some(&mut guard), &sub.nav).ok()?
    };
    if !(200..300).contains(&response.status) {
        return None;
    }
    // Parsed leniently: parse_stylesheet never fails, dropping only the
    // offending rules, so even a wrong content type degrades gracefully.
    let text = response.text();
    Some((parse_stylesheet(&text), text))
}

/// `@import` target URLs in `css`, in authored order (deduped by the
/// caller). A byte-level scan, not the CSS parser: the parser drops
/// `@import` entirely, and this slice must not touch CSS parse code.
/// Handles `@import "x"`, `@import 'x'`, and `@import url(x)` /
/// `url("x")` (media queries after the URL are ignored); `/* */`
/// comments are skipped so a commented-out import is not fetched.
/// Byte slices only (never `str` indexing), so non-ASCII text cannot
/// panic the scan at a character boundary.
fn import_urls(css: &str) -> Vec<String> {
    fn skip_ws_comments(bytes: &[u8], mut i: usize) -> usize {
        loop {
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
                match bytes[i + 2..].windows(2).position(|w| w == b"*/") {
                    Some(end) => {
                        i += 2 + end + 2;
                    }
                    None => return bytes.len(),
                }
            } else {
                return i;
            }
        }
    }
    fn quoted(bytes: &[u8], i: usize) -> Option<(String, usize)> {
        let quote = *bytes.get(i)?;
        if quote != b'"' && quote != b'\'' {
            return None;
        }
        let mut j = i + 1;
        while j < bytes.len() && bytes[j] != quote {
            j += 1;
        }
        if j >= bytes.len() {
            return None;
        }
        Some((
            std::str::from_utf8(&bytes[i + 1..j])
                .unwrap_or("")
                .to_string(),
            j + 1,
        ))
    }
    let bytes = css.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            i = skip_ws_comments(bytes, i);
            continue;
        }
        if i + 7 <= bytes.len()
            && bytes[i..i + 7].eq_ignore_ascii_case(b"@import")
            && !bytes
                .get(i + 7)
                .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'-' || *b == b'_')
        {
            let mut j = skip_ws_comments(bytes, i + 7);
            if let Some((url, end)) = quoted(bytes, j) {
                out.push(url);
                i = end;
                continue;
            }
            if j + 3 <= bytes.len() && bytes[j..j + 3].eq_ignore_ascii_case(b"url") {
                j = skip_ws_comments(bytes, j + 3);
                if bytes.get(j) == Some(&b'(') {
                    j = skip_ws_comments(bytes, j + 1);
                    if let Some((url, end)) = quoted(bytes, j) {
                        out.push(url);
                        i = end;
                        continue;
                    }
                    let start = j;
                    while j < bytes.len() && bytes[j] != b')' && !bytes[j].is_ascii_whitespace() {
                        j += 1;
                    }
                    if j > start {
                        out.push(
                            std::str::from_utf8(&bytes[start..j])
                                .unwrap_or("")
                                .to_string(),
                        );
                    }
                    i = j;
                    continue;
                }
            }
            i += 7;
            continue;
        }
        i += 1;
    }
    out
}

/// Recursively fetch the sheets `text` imports, in authored order,
/// appending `(sheet, sheet_url)` pairs to `out`. `depth` is the
/// nesting level of `text`'s own sheet (top sheets call with 0);
/// sheets deeper than [`MAX_IMPORT_DEPTH`] are never fetched, and
/// `seen` (serialized URLs, seeded with the top sheets' own URLs)
/// breaks import cycles and dedupes fan-in. A failed import is
/// skipped, never fatal — like a failed linked sheet.
fn collect_imported_sheets(
    text: &str,
    base: &Url,
    sub: &SubresourceFetch<'_>,
    depth: usize,
    seen: &mut HashSet<String>,
    out: &mut Vec<(Stylesheet, Url)>,
) {
    if depth >= MAX_IMPORT_DEPTH || out.len() >= MAX_IMPORTED_SHEETS {
        return;
    }
    for raw in import_urls(text) {
        if out.len() >= MAX_IMPORTED_SHEETS {
            return;
        }
        let Ok(url) = bowser_url::parse(&raw, Some(base)) else {
            continue;
        };
        if !seen.insert(url.serialize()) {
            continue;
        }
        let Some((sheet, sheet_text)) = fetch_stylesheet(&url, sub) else {
            continue;
        };
        out.push((sheet, url.clone()));
        collect_imported_sheets(&sheet_text, &url, sub, depth + 1, seen, out);
    }
}

/// Fetch every `@font-face` font in `stylesheets`, in document order.
///
/// `bases` runs parallel to `stylesheets`: the URL each sheet's `src`
/// entries resolve against — the document base for inline `<style>`
/// sheets, the sheet's own URL for linked sheets (threaded through the
/// stylesheet loop in `load_document_inner`, never the document base
/// for linked sheets). A shorter `bases` simply truncates the walk.
///
/// Per rule, `src` entries are tried in authored order (`local(...)`
/// entries are skipped by `url_srcs`: they name already-installed
/// faces); the first entry that fetches AND decodes to SFNT wins, and
/// later entries are not fetched. SFNT passes through
/// [`decode_webfont`]; WOFF1 is decoded to SFNT there; woff2 and
/// garbage return `None` and fall through to the next entry, exactly
/// like a failed fetch. A rule whose entries all fail records
/// nothing. Counts are capped at [`MAX_WEBFONTS`], single fetched
/// fonts at [`FONT_MAX_BYTES`], single decoded fonts at paint's
/// 32 MiB decoded cap, and the load's unique DECODED bytes at
/// [`WEBFONT_BUDGET_BYTES`] (fetching stops past it).
///
/// Dedupe is per (URL, family, weight, style), not per URL: two
/// rules reusing one file for different families (or weights) each
/// land, and the second reuses the first rule's decoded bytes — no
/// refetch, no dropped rule. A failed fetch is never cached, so a
/// later rule may still retry the URL. Nothing here fails the page:
/// any failure yields fewer webfonts.
fn fetch_webfonts(
    stylesheets: &[Stylesheet],
    bases: &[Url],
    sub: &SubresourceFetch<'_>,
) -> Vec<Webfont> {
    let mut out: Vec<Webfont> = Vec::new();
    // Successfully decoded bytes by serialized URL: the reuse cache.
    // Decoded bytes are what the cache retains, so they are also what
    // the budget counts (see [`WEBFONT_BUDGET_BYTES`]).
    let mut fetched: HashMap<String, Arc<[u8]>> = HashMap::new();
    // (URL, family, weight, style) triples already emitted.
    let mut emitted: HashSet<(String, String, u16, String)> = HashSet::new();
    let mut unique_bytes: usize = 0;
    'outer: for (sheet, base) in stylesheets.iter().zip(bases.iter()) {
        for face in sheet.font_faces() {
            if out.len() >= MAX_WEBFONTS || unique_bytes >= WEBFONT_BUDGET_BYTES {
                break 'outer;
            }
            let family = face.family_name().to_owned();
            let weight = face.font_weight();
            let style = face.font_style_name().to_owned();
            for raw in face.url_srcs() {
                let Ok(url) = bowser_url::parse(raw, Some(base)) else {
                    continue;
                };
                let url_key = url.serialize();
                let key = (
                    url_key.clone(),
                    family.to_lowercase(),
                    weight,
                    style.clone(),
                );
                if emitted.contains(&key) {
                    continue;
                }
                if let Some(bytes) = fetched.get(&url_key) {
                    out.push(Webfont {
                        family: family.clone(),
                        weight,
                        weight_range: face.font_weight_range(),
                        style: style.clone(),
                        bytes: Arc::clone(bytes),
                    });
                    emitted.insert(key);
                    break;
                }
                let Some(bytes) = fetch_font_bytes(&url, sub) else {
                    continue;
                };
                // Decode before the budget check: the budget counts
                // decoded bytes, and a WOFF1 file's decoded size is
                // what the cache will retain. woff2/garbage decode to
                // `None` and fall through to the next src entry.
                let Some(bytes) = decode_webfont(&bytes) else {
                    continue;
                };
                if unique_bytes + bytes.len() > WEBFONT_BUDGET_BYTES {
                    // Over budget: drop the bytes and stop fetching
                    // further rules for this document.
                    break 'outer;
                }
                unique_bytes += bytes.len();
                fetched.insert(url_key, Arc::clone(&bytes));
                out.push(Webfont {
                    family: family.clone(),
                    weight,
                    weight_range: face.font_weight_range(),
                    style: style.clone(),
                    bytes,
                });
                emitted.insert(key);
                break;
            }
        }
    }
    out
}

/// One font's bytes: `data:` URLs decode inline, anything else goes
/// through the same `bowser_net` fetch path stylesheets use.
/// `None` on fetch failure, non-2xx status, a failed CORS check, or
/// an over-cap body. Check order: status, then CORS, then the size
/// cap; decoding runs later, in [`fetch_webfonts`].
///
/// CORS (no preflight: a GET carrying only `Origin` is a simple
/// request, and no other custom headers are sent):
/// - Same-origin (request URL origin vs the document origin): the
///   shared jar is used as before. `Origin` is sent unconditionally
///   (harmless same-origin, and it matches the JS side): without it,
///   a same-origin request redirected cross-origin would be
///   ACAO-checked without `Origin` ever having been sent, so a
///   server that emits ACAO only when it sees `Origin` would fail a
///   font browsers load.
/// - Cross-origin: the shared jar is NOT used — no cookies are sent
///   and none are stored — and an `Origin: <document origin>`
///   header is sent via `get_request`'s `extra` headers. The final
///   response (after redirects, so `response.url`'s origin, not the
///   request URL's) must then pass [`cors_check`]: same-origin
///   responses pass unconditionally, otherwise exactly one ACAO
///   value must be `*` or exactly the document origin. Because the
///   jar is bypassed, no credentials were sent on a cross-origin
///   request, so `*` is valid there; when the request itself was
///   same-origin (so the jar was in play) but a redirect landed
///   cross-origin, `*` is not accepted. A failed check fails that
///   `src` entry, so `fetch_webfonts` tries the rule's next entry.
fn fetch_font_bytes(url: &Url, sub: &SubresourceFetch<'_>) -> Option<Vec<u8>> {
    if url.scheme().eq_ignore_ascii_case("data") {
        let (bytes, _) = decode_data_url(url)?;
        return (bytes.len() <= FONT_MAX_BYTES).then_some(bytes);
    }
    let cross_origin = !sub.document_origin.is_same_origin(&url.origin());
    // `Origin` goes out unconditionally (see the doc above): the jar
    // choice below is unchanged — cross-origin still bypasses it.
    let request = get_request(
        url,
        sub.user_agent,
        vec![("Origin".to_string(), sub.document_origin.serialize())],
    );
    let response = if cross_origin {
        fetch_with_jar_in_context(&request, None, &sub.nav).ok()?
    } else {
        let mut guard = sub.jar.lock().expect("loader: cookie jar mutex poisoned");
        fetch_with_jar_in_context(&request, Some(&mut guard), &sub.nav).ok()?
    };
    if !(200..300).contains(&response.status) {
        return None;
    }
    // Every ACAO value, not just the first: fetch fails a response
    // carrying two (see `cors_check`).
    let acao: Vec<&str> = response
        .headers
        .iter()
        .filter(|(n, _)| n.eq_ignore_ascii_case("access-control-allow-origin"))
        .map(|(_, v)| v.as_str())
        .collect();
    if !cors_check(
        &sub.document_origin,
        &response.url.origin(),
        &acao,
        // Credentials were in play exactly when the jar was used.
        !cross_origin,
    ) {
        return None;
    }
    if response.body.len() > FONT_MAX_BYTES {
        return None;
    }
    Some(response.body)
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

pub(crate) fn collect_images(document: &Document, base: &Url) -> Vec<(NodeId, Url)> {
    let mut out = Vec::new();
    for node in document.descendants(document.root()) {
        let NodeKind::Element(data) = document.kind(node) else {
            continue;
        };
        // v1: `srcset` is ignored; only `src` is fetched.
        if data.tag_name == "img"
            && let Some(src) = document.get_attribute(node, "src")
            && let Ok(url) = bowser_url::parse(src, Some(base))
        {
            out.push((node, url));
        }
    }
    out
}

/// `background-image` URLs from every rendered element's computed style,
/// resolved against the document base.
///
/// Paint resolves `background-image` through the same image cache as
/// `<img>` (the raw [`CssValue::Url`] string goes through
/// [`cached_image_for_src`]), so without this step every CSS background
/// image is a guaranteed cache miss and paints as a placeholder.
/// Elements with computed `display: none` are skipped: layout never
/// builds boxes for them, so paint would never resolve their URLs. The
/// `background` shorthand is handled: the css cascade expands it to
/// `background-image` (and friends) before computed-style read time.
///
/// Resolution is against the document base URL (`base`, which already
/// includes `<base href>`), never the originating stylesheet's URL: the
/// css crate keeps `CssValue::Url` in raw (possibly relative) form and
/// `parse_stylesheet` records no base URL for linked sheets, so a
/// stylesheet-relative URL cannot be honored in v1. Absolute URLs are
/// unaffected. The `@media`/viewport-unit viewport is the fixed 800x600
/// default; media queries keyed on the real window size may misfire here.
fn collect_background_image_urls(
    document: &Document,
    stylesheets: &[Stylesheet],
    base: &Url,
) -> Vec<Url> {
    let author: Vec<&Stylesheet> = stylesheets.iter().collect();
    let ua = default_ua_stylesheet();
    let styler = Styler::new(document, &author, &ua, Viewport { w: 800.0, h: 600.0 });
    let mut out = Vec::new();
    // `descendants` is document order, parents before children, so each
    // `style_for` reuses its ancestors' memoized cascade.
    for node in document.descendants(document.root()) {
        let NodeKind::Element(_) = document.kind(node) else {
            continue;
        };
        let style = styler.style_for(node);
        if style.get_str_by_index(props::DISPLAY) == Some("none") {
            continue;
        }
        if let Some(CssValue::Url(raw)) = style.get_by_index(props::BACKGROUND_IMAGE)
            && let Ok(url) = bowser_url::parse(raw, Some(base))
        {
            out.push(url);
        }
    }
    out
}

/// One loaded image: raster pixels for paint (when the format has a
/// rasterizer) plus intrinsic dimensions for layout (raster and SVG).
/// SVG has intrinsics but no pixels — there is no SVG rasterizer in
/// v1, so paint keeps showing the placeholder for SVG `<img>`s.
#[derive(Clone, Debug)]
struct LoadedImage {
    pixels: Option<DecodedImage>,
    dims: IntrinsicDims,
}

/// Fetch, decode, and cache one image, returning its load.
///
/// The shared [`IMAGE_CACHE`] is consulted first, so a second load of an
/// image already cached (e.g. a reload) skips the network; `fetched`
/// dedupes repeated URLs within this load. On a fetch/decode failure
/// nothing is cached and `None` is returned. `key` is the serialized
/// request URL (`Url::serialize`) — the same key paint looks up through
/// [`cached_image`]/[`cached_image_for_src`], so `<img>` keys stay
/// `parse(src, effective_base_url).serialize()`.
fn fetch_and_cache_image(
    key: &str,
    url: &Url,
    sub: &SubresourceFetch<'_>,
    fetched: &mut HashMap<String, Option<LoadedImage>>,
) -> Option<LoadedImage> {
    let loaded = fetched.entry(key.to_string()).or_insert_with(|| {
        if let Some(pixels) = cached_image(key) {
            let dims = IntrinsicDims::from_size(pixels.width as f32, pixels.height as f32);
            return Some(LoadedImage {
                pixels: Some(pixels),
                dims,
            });
        }
        load_image(url, sub)
    });
    let loaded = loaded.as_ref()?;
    // Cache the pixels for paint; intrinsics are recorded by the caller.
    // SVG has no pixels: nothing to cache.
    if let Some(pixels) = &loaded.pixels {
        IMAGE_CACHE
            .lock()
            .expect("loader: image cache mutex poisoned")
            .insert(key.to_string(), pixels.clone());
    }
    Some(loaded.clone())
}

/// Load one image's bytes and decode them: raster formats to pixels
/// plus dimensions, SVG to dimensions only.
///
/// `data:` URLs decode inline (no network); anything else goes through
/// the `Range`-fetch pipeline. Returns `None` when the bytes cannot be
/// fetched or are not a recognized image.
/// Decode an image's intrinsic dimensions from its bytes, without
/// allocating pixels: SVG (declared or sniffed) yields its markup
/// intrinsics, a decodable raster yields its pixel size.
fn decode_image_dims(bytes: &[u8], mediatype: &str) -> Option<IntrinsicDims> {
    if mediatype.contains("svg") {
        let svg = svg_intrinsics(bytes)?;
        return Some(IntrinsicDims {
            w: svg.width,
            h: svg.height,
            ratio: svg.ratio,
        });
    }
    match decode_image(bytes) {
        Ok(decoded) => Some(IntrinsicDims::from_size(
            decoded.width as f32,
            decoded.height as f32,
        )),
        Err(_) => {
            // Not a decodable raster image; an SVG served under a
            // non-SVG media type (or none) still yields intrinsics.
            let svg = svg_intrinsics(bytes)?;
            Some(IntrinsicDims {
                w: svg.width,
                h: svg.height,
                ratio: svg.ratio,
            })
        }
    }
}

/// Synchronously decode the intrinsics of a `data:` image URL — the
/// headless pump's late-image drain calls this when no background fetch
/// thread is involved. `None` for non-`data:` schemes or undecodable
/// payloads (the caller treats that as a failed load).
pub fn decode_data_image_url(url: &Url) -> Option<IntrinsicDims> {
    if !url.scheme().eq_ignore_ascii_case("data") {
        return None;
    }
    let (bytes, mediatype) = decode_data_url(url)?;
    decode_image_dims(&bytes, &mediatype)
}

fn load_image(url: &Url, sub: &SubresourceFetch<'_>) -> Option<LoadedImage> {
    let (bytes, mediatype) = image_bytes(url, sub)?;
    let dims = decode_image_dims(&bytes, &mediatype)?;
    let pixels = if mediatype.contains("svg") {
        // Declared SVG: intrinsics from the markup, no rasterizer.
        None
    } else {
        decode_image(&bytes).ok() // SVG sniffed under a non-SVG media type.
    };
    Some(LoadedImage { pixels, dims })
}

/// One image's complete bytes plus its media type: `data:` URLs decode
/// inline, anything else is assembled via `Range` requests (see
/// [`fetch_image_bytes`]). `None` when the image cannot or should not
/// be fetched in full.
fn image_bytes(url: &Url, sub: &SubresourceFetch<'_>) -> Option<(Vec<u8>, String)> {
    if let Some((bytes, mediatype)) = decode_data_url(url) {
        return Some((bytes, mediatype));
    }
    fetch_image_bytes(url, sub).map(|bytes| (bytes, String::new()))
}

/// Decode a `data:` URL to its bytes and media type (lowercased),
/// per RFC 2397: `data:[<mediatype>][;base64],<data>`.
///
/// `;base64` (which must be the last `;`-parameter) selects base64;
/// otherwise the data is percent-decoded. The media type defaults to
/// `text/plain;charset=US-ASCII` when absent. Returns `None` when the
/// URL is not a `data:` URL, has no `,`, or its payload does not
/// decode. Decoded bodies are capped at [`IMAGE_MAX_BYTES`] like
/// network fetches.
fn decode_data_url(url: &Url) -> Option<(Vec<u8>, String)> {
    if !url.scheme().eq_ignore_ascii_case("data") {
        return None;
    }
    let opaque = url.path();
    let comma = opaque.find(',')?;
    let (meta, data) = opaque.split_at(comma);
    let data = &data[1..];
    let (mediatype, is_base64) = match meta.strip_suffix(";base64") {
        Some(mt) => (mt, true),
        None => (meta, false),
    };
    let mediatype = if mediatype.is_empty() {
        "text/plain;charset=US-ASCII".to_string()
    } else {
        mediatype.to_ascii_lowercase()
    };
    let bytes = if is_base64 {
        base64_decode(data).ok()?
    } else {
        percent_decode(data)
    };
    if bytes.len() > IMAGE_MAX_BYTES {
        return None;
    }
    Some((bytes, mediatype))
}

/// Fetch, decode, and cache one image for a document *after* its load
/// phase completed: the completion hook for a late `<img>` — a node
/// inserted by script or `innerHTML` once the document has already laid
/// out, which the load-time image step never saw, or an `<img>` whose
/// `src` changed after load.
///
/// Returns the decoded intrinsic size on success, `None` on any failure
/// (fetch error, non-2xx status, over-cap body, decode failure): a
/// failed load leaves the element at whatever size layout assumed, so
/// callers must invalidate only when the outcome differs from the
/// recorded intrinsics (see
/// [`Tab::note_image_loaded`](crate::tab::Tab::note_image_loaded)).
///
/// The decoded pixels land in the shared [`IMAGE_CACHE`] under the
/// serialized request URL, exactly like the load-time image step, so
/// paint resolves the late image through the same cache; the cache is
/// consulted first, so an already-decoded image skips the network. `url`
/// is the resolved request URL — the same
/// `bowser_url::parse(src, effective_base_url(document, final_url)).serialize()`
/// key the loader wrote at load time — and `document_url` is the
/// document's final URL (post-redirect): it sets the SameSite top
/// context of the fetch, like the load-time image step.
///
/// # Driver contract
///
/// The driver is whatever pumps page scripts after install (no such
/// worker exists yet). Its trigger is
/// [`Tab::pending_late_images`](crate::tab::Tab::pending_late_images):
/// it scans the installed document's `<img>` elements in document order
/// and starts a fetch for each one that has no entry in the tab's
/// image-load map, or whose `src` resolved against the document base
/// differs from the request URL recorded there. In particular an
/// already-loaded `<img>` whose `src` changes IS refetched — a
/// "no intrinsics entry" trigger would miss exactly this. (A future
/// optimization may restrict the scan to the dirtied subtrees of the
/// script pump's [`DirtyReport`](bowser_dom::DirtyReport): attribute
/// writes are visible there, so the trigger stays exact.)
///
/// Each started fetch bumps the node's image generation and records the
/// request URL. The driver passes the returned `(node, url, generation)`
/// triple to this function on a worker thread, then feeds `(node,
/// generation, result)` back into
/// [`Tab::note_image_loaded`](crate::tab::Tab::note_image_loaded),
/// which applies the completion only when the generation is still
/// current. Out-of-order completions (src A→B→A, B's slow fetch landing
/// last) are stale and dropped instead of installing the wrong size; a
/// repeat delivery of the same completion is dropped as a duplicate.
/// Every *new* completion repaints — even at an unchanged size, the
/// pixels are new — while layout re-runs only when the size changed. A
/// failed completion (`None`) leaves the recorded size untouched.
///
/// # Threading
///
/// [`Tab`](crate::tab::Tab) is `!Send` (it holds `Rc`-shared document
/// and scroll state): [`Tab::pending_late_images`] and
/// [`Tab::note_image_loaded`](crate::tab::Tab::note_image_loaded) run
/// on the thread that owns the tab — the window worker's event thread,
/// between script pumps — never on a fetch worker. Only this function
/// is worker-safe: it takes no `Tab` and touches only its arguments
/// plus the process-wide [`IMAGE_CACHE`].
///
/// Blocking like every other load here: run on a worker thread (tests
/// prime the cache with [`prime_image_cache`] instead).
//
// (Not yet called outside tests: the allow keeps
// `cargo clippy -- -D warnings` green until the script-pump worker wires
// the driver up.)
#[allow(dead_code)]
pub fn fetch_late_image(
    url: &Url,
    document_url: &Url,
    jar: &Mutex<CookieJar>,
    user_agent: &str,
) -> Option<IntrinsicDims> {
    let sub = SubresourceFetch {
        jar,
        user_agent,
        document_origin: document_url.origin(),
        nav: NavigationContext {
            top_host: document_url.host().map(str::to_owned),
            top_secure: Some(document_url.scheme() == "https"),
            top_level_safe_navigation: false,
        },
    };
    let key = url.serialize();
    // The per-load URL dedup map the load-time step threads through has
    // no meaning for a one-shot late fetch: pass a throwaway.
    let mut fetched: HashMap<String, Option<LoadedImage>> = HashMap::new();
    fetch_and_cache_image(&key, url, &sub, &mut fetched).map(|loaded| loaded.dims)
}

/// Fetch one image and decode it: `Range` requests starting at the first
/// [`IMAGE_FETCH_BYTES`] bytes, then `bowser_paint::images::decode_image`
/// over the complete body.
///
/// The first request asks for `bytes=0-<window>`. A 206 answer means the
/// server honors `Range`: follow-up requests fetch the rest in
/// [`IMAGE_FETCH_BYTES`] windows until the `Content-Range` total is
/// reached (or, without a usable `Content-Range`, until a short read or
/// 416). A 200 answer means the server ignored `Range`: the body is
/// already complete. Anything else is a failure. Bodies are bounded by
/// [`IMAGE_MAX_BYTES`]: a larger advertised total aborts before the
/// follow-ups, and assembly stops if the cap is reached.
///
/// Assemble one image's complete bytes via `Range` requests (see
/// [`image_bytes`]). `None` when the image cannot or should not be
/// fetched in full.
fn fetch_image_bytes(url: &Url, sub: &SubresourceFetch<'_>) -> Option<Vec<u8>> {
    let first = fetch_image_range(url, sub, 0, IMAGE_FETCH_BYTES as u64 - 1)?;
    match first.status {
        // The server ignored Range: the body is already the whole
        // resource (bowser_net capped it at IMAGE_MAX_BYTES).
        200 => Some(first.body),
        206 => {
            let mut body = first.body;
            if body.is_empty() {
                return None;
            }
            let total = first.content_range_total;
            if total.is_some_and(|t| t > IMAGE_MAX_BYTES as u64) {
                return None;
            }
            // A short first 206 without a Content-Range total is the final
            // chunk: there is nothing more to request.
            if total.is_none() && body.len() < IMAGE_FETCH_BYTES {
                return Some(body);
            }
            let mut next = body.len() as u64;
            loop {
                // The known total is checked before the cap: an image
                // exactly at IMAGE_MAX_BYTES is legal, and its final chunk
                // must not be rejected as over-cap.
                if total.is_some_and(|t| next >= t) {
                    break;
                }
                if next >= IMAGE_MAX_BYTES as u64 {
                    return None;
                }
                let end = match total {
                    Some(t) => (t - 1).min(next + IMAGE_FETCH_BYTES as u64 - 1),
                    None => next + IMAGE_FETCH_BYTES as u64 - 1,
                };
                let chunk = fetch_image_range(url, sub, next, end)?;
                match chunk.status {
                    206 => {
                        if chunk.body.is_empty() {
                            // No progress; bail instead of spinning on a
                            // server that answers 206 with empty bodies.
                            break;
                        }
                        // Cap assembly before extending: a 206 body larger
                        // than the requested range must not grow the buffer
                        // past IMAGE_MAX_BYTES, not even transiently.
                        if body.len() + chunk.body.len() > IMAGE_MAX_BYTES {
                            return None;
                        }
                        next += chunk.body.len() as u64;
                        body.extend_from_slice(&chunk.body);
                        // Without a Content-Range total, a short read
                        // marks the end of the resource.
                        if total.is_none() && chunk.body.len() < IMAGE_FETCH_BYTES {
                            break;
                        }
                    }
                    // The server stopped honoring Range mid-stream: a 200
                    // body is the whole resource from byte 0.
                    200 => return Some(chunk.body),
                    // Past the end of the resource: keep what we have.
                    416 => break,
                    _ => return None,
                }
            }
            Some(body)
        }
        _ => None,
    }
}

/// One image range request: `Range: bytes={start}-{end}`.
///
/// Returns the status, the body, and the `Content-Range` total when the
/// server sent a parseable one. `None` on fetch failure. The jar is
/// locked only for this request.
struct ImageRangeResponse {
    status: u16,
    body: Vec<u8>,
    /// Total resource length from `Content-Range: bytes a-b/total`, when
    /// the server sent one we could parse (`*` totals stay `None`).
    content_range_total: Option<u64>,
}

fn fetch_image_range(
    url: &Url,
    sub: &SubresourceFetch<'_>,
    start: u64,
    end: u64,
) -> Option<ImageRangeResponse> {
    let request = get_request(
        url,
        sub.user_agent,
        vec![("Range".to_string(), format!("bytes={start}-{end}"))],
    );
    let response = {
        let mut guard = sub.jar.lock().expect("loader: cookie jar mutex poisoned");
        fetch_with_jar_in_context(&request, Some(&mut guard), &sub.nav).ok()?
    };
    let content_range_total = response
        .header("content-range")
        .and_then(parse_content_range_total);
    Some(ImageRangeResponse {
        status: response.status,
        body: response.body,
        content_range_total,
    })
}

/// Parse the total out of `Content-Range: bytes <first>-<last>/<total>`.
/// Returns `None` for anything else, including `bytes <first>-<last>/*`.
fn parse_content_range_total(value: &str) -> Option<u64> {
    let value = value.strip_prefix("bytes ")?;
    let (_, total) = value.split_once('/')?;
    total.trim().parse().ok()
}

// ---------------------------------------------------------------------------
// Tests: local TcpListener server serving HTML + CSS + PNG
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::thread;

    const INDEX_HTML: &str = "<!DOCTYPE html>\n\
        <html><head>\n\
        <style>p { color: red; }</style>\n\
        <link rel=\"stylesheet\" href=\"/a.css\">\n\
        <style>div { margin: 0; }</style>\n\
        <link rel=\"stylesheet\" href=\"/missing.css\">\n\
        <link rel=\"StyleSheet\" href=\"/b.css\">\n\
        </head><body>\n\
        <img src=\"/img.png\" alt=\"x\">\n\
        <img src=\"/img.gif\" alt=\"z\">\n\
        <img src=\"/broken.png\" alt=\"y\">\n\
        <script src=\"/app.js\"></script>\n\
        </body></html>";

    const BASE_HTML: &str = "<!DOCTYPE html>\n\
        <html><head>\n\
        <base href=\"/assets/\">\n\
        <link rel=\"stylesheet\" href=\"theme.css\">\n\
        </head><body></body></html>";

    /// Build a minimal but fully decodable RGBA PNG: signature, IHDR,
    /// IDAT (zlib stream with one stored deflate block), IEND, all with
    /// correct CRCs. The loader decodes images now instead of sniffing
    /// headers, so fixtures must survive `decode_image`.
    fn png_bytes(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
        assert_eq!(pixels.len(), width as usize * height as usize * 4);
        let mut raw = Vec::with_capacity(height as usize * (1 + width as usize * 4));
        for row in pixels.chunks_exact(width as usize * 4) {
            raw.push(0); // filter type: None
            raw.extend_from_slice(row);
        }
        // zlib: 2-byte header, stored (uncompressed) deflate blocks of at
        // most 65535 bytes each, adler32.
        let mut zlib = vec![0x78, 0x01];
        let mut blocks = raw.chunks(65535).peekable();
        while let Some(block) = blocks.next() {
            let last = blocks.peek().is_none();
            zlib.push(if last { 0x01 } else { 0x00 }); // BFINAL, BTYPE=00 (stored)
            let len = block.len() as u16;
            zlib.extend_from_slice(&len.to_le_bytes());
            zlib.extend_from_slice(&(!len).to_le_bytes());
            zlib.extend_from_slice(block);
        }
        zlib.extend_from_slice(&adler32(&raw).to_be_bytes());

        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA

        let mut v = Vec::new();
        v.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);
        v.extend_from_slice(&png_chunk(b"IHDR", &ihdr));
        v.extend_from_slice(&png_chunk(b"IDAT", &zlib));
        v.extend_from_slice(&png_chunk(b"IEND", &[]));
        v
    }

    fn png_chunk(ty: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut c = Vec::with_capacity(12 + data.len());
        c.extend_from_slice(&(data.len() as u32).to_be_bytes());
        c.extend_from_slice(ty);
        c.extend_from_slice(data);
        let mut crc_input = Vec::with_capacity(4 + data.len());
        crc_input.extend_from_slice(ty);
        crc_input.extend_from_slice(data);
        c.extend_from_slice(&crc32(&crc_input).to_be_bytes());
        c
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }

    fn adler32(data: &[u8]) -> u32 {
        let mut a = 1u32;
        let mut b = 0u32;
        for &byte in data {
            a = (a + u32::from(byte)) % 65521;
            b = (b + a) % 65521;
        }
        (b << 16) | a
    }

    /// (status, reason, extra headers, body) for each test route.
    fn route(path: &str) -> (u16, &'static str, Vec<(&'static str, String)>, Vec<u8>) {
        let css: fn(&str) -> Vec<(&str, String)> =
            |_| vec![("Content-Type", "text/css".to_string())];
        match path {
            "/index.html" => (
                200,
                "OK",
                vec![("Content-Type", "text/html; charset=utf-8".to_string())],
                INDEX_HTML.as_bytes().to_vec(),
            ),
            "/a.css" => (200, "OK", css(""), b"a { color: blue; }".to_vec()),
            "/b.css" => (200, "OK", css(""), b"b { color: green; }".to_vec()),
            "/bad.css" => (200, "OK", css(""), b"p {{{ not css".to_vec()),
            "/missing.css" => (404, "Not Found", vec![], b"nope".to_vec()),
            "/img.png" => (
                200,
                "OK",
                vec![("Content-Type", "image/png".to_string())],
                // Solid red 4x2.
                png_bytes(4, 2, &[255u8, 0, 0, 255].repeat(8)),
            ),
            // Canonical GIF fixture (4x2, distinct colors), shared with
            // the bowser_paint GIF tests.
            "/img.gif" => (
                200,
                "OK",
                vec![("Content-Type", "image/gif".to_string())],
                include_bytes!("../../bowser_paint/tests/fixtures/gif_4x2.gif").to_vec(),
            ),
            "/broken.png" => (404, "Not Found", vec![], b"nope".to_vec()),
            "/app.js" => (
                200,
                "OK",
                vec![("Content-Type", "text/javascript".to_string())],
                b"var fromServer = 42;".to_vec(),
            ),
            "/bad.js" => (404, "Not Found", vec![], b"nope".to_vec()),
            "/file.txt" => (
                200,
                "OK",
                vec![("Content-Type", "text/plain".to_string())],
                b"hello".to_vec(),
            ),
            "/missing" => (
                404,
                "Not Found",
                vec![("Content-Type", "text/html".to_string())],
                b"<p>not found</p>".to_vec(),
            ),
            "/redir" => (
                302,
                "Found",
                vec![("Location", "/index.html".to_string())],
                Vec::new(),
            ),
            "/sub/page.html" => (
                200,
                "OK",
                vec![("Content-Type", "text/html".to_string())],
                BASE_HTML.as_bytes().to_vec(),
            ),
            "/assets/theme.css" => (200, "OK", css(""), b".t { color: red; }".to_vec()),
            // Page under /sub/ whose <base href> points at /assets/; the
            // relative img src must resolve (and cache) against the base.
            "/sub/gallery.html" => (
                200,
                "OK",
                vec![("Content-Type", "text/html; charset=utf-8".to_string())],
                b"<!DOCTYPE html><html><head><base href=\"/assets/\">\
                  </head><body><img src=\"pic.png\"></body></html>"
                    .to_vec(),
            ),
            "/assets/pic.png" => (
                200,
                "OK",
                vec![("Content-Type", "image/png".to_string())],
                // Solid green 2x2.
                png_bytes(2, 2, &[0u8, 255, 0, 255].repeat(4)),
            ),
            // Page whose div paints a CSS background image via the
            // `background` shorthand; the relative url() must resolve
            // against the document URL.
            "/bg/page.html" => (
                200,
                "OK",
                vec![("Content-Type", "text/html; charset=utf-8".to_string())],
                b"<!DOCTYPE html><html><head><style>\
                  div { background: url(\"bg.png\"); }</style>\
                  </head><body><div>hello</div></body></html>"
                    .to_vec(),
            ),
            "/bg/bg.png" => (
                200,
                "OK",
                vec![("Content-Type", "image/png".to_string())],
                // Solid blue 3x1.
                png_bytes(3, 1, &[0u8, 0, 255, 255].repeat(3)),
            ),
            _ => (404, "Not Found", vec![], b"nope".to_vec()),
        }
    }

    /// Serve one connection; returns false when the server should stop.
    /// `respond` maps the request path (and the `Range` header, if the
    /// client sent one) to (status, reason, headers, body).
    type Route = (u16, &'static str, Vec<(&'static str, String)>, Vec<u8>);

    fn handle_one(stream: &mut TcpStream, respond: &dyn Fn(&str, Option<&str>) -> Route) -> bool {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 1024];
        loop {
            match stream.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    buf.extend_from_slice(&chunk[..n]);
                    if buf.windows(4).any(|w| w == b"\r\n\r\n") || buf.len() > 8192 {
                        break;
                    }
                }
                Err(_) => return true,
            }
        }
        let request = String::from_utf8_lossy(&buf);
        let target = request
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("/");
        // The fetch client sends absolute-form request targets
        // ("GET http://host:port/path HTTP/1.1"); strip to origin-form.
        let path = target
            .strip_prefix("http://")
            .or_else(|| target.strip_prefix("https://"))
            .and_then(|rest| rest.find('/').map(|i| &rest[i..]))
            .unwrap_or(target);
        let path = path.split('?').next().unwrap_or(path);
        if path == "/__shutdown" {
            return false;
        }
        let range = request.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case("range")
                .then(|| value.trim().to_owned())
        });
        let (status, reason, headers, body) = respond(path, range.as_deref());
        let mut head = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n",
            body.len()
        );
        for (name, value) in headers {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        head.push_str("\r\n");
        let _ = stream.write_all(head.as_bytes());
        let _ = stream.write_all(&body);
        true
    }

    struct TestServer {
        base: String,
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl TestServer {
        fn start() -> Self {
            Self::start_with(|path, _| route(path))
        }

        fn start_with<F>(respond: F) -> Self
        where
            F: Fn(&str, Option<&str>) -> Route + Send + 'static,
        {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
            let addr = listener.local_addr().expect("test server addr");
            let stop = Arc::new(AtomicBool::new(false));
            let stop_flag = Arc::clone(&stop);
            let handle = thread::spawn(move || {
                for stream in listener.incoming() {
                    if stop_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    match stream {
                        Ok(mut s) => {
                            if !handle_one(&mut s, &respond) {
                                break;
                            }
                        }
                        Err(_) => continue,
                    }
                }
            });
            TestServer {
                base: format!("http://{addr}"),
                stop,
                handle: Some(handle),
            }
        }

        fn url(&self, path: &str) -> Url {
            bowser_url::parse(&format!("{}{path}", self.base), None).expect("test url")
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            // Wake the blocking accept() so the thread can observe `stop`.
            let _ = TcpStream::connect(self.base.trim_start_matches("http://"));
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn img_node_with_src(document: &Document, src: &str) -> Option<NodeId> {
        for node in document.descendants(document.root()) {
            let NodeKind::Element(data) = document.kind(node) else {
                continue;
            };
            if data.tag_name == "img" && document.get_attribute(node, "src") == Some(src) {
                return Some(node);
            }
        }
        None
    }

    #[test]
    fn stylesheets_load_in_document_order_and_failed_sheets_are_skipped() {
        let server = TestServer::start();
        let jar = Mutex::new(CookieJar::new());
        let loaded = load_document_with_user_agent(
            &server.url("/index.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
        )
        .expect("load");

        // <style>, a.css, <style>, b.css — /missing.css (404) is skipped.
        // The css crate expands `margin` into its longhands; [0] is margin-top.
        assert_eq!(loaded.stylesheets.len(), 4);
        let props: Vec<&str> = loaded
            .stylesheets
            .iter()
            .map(|s| s.rules()[0].declarations[0].property.as_str())
            .collect();
        assert_eq!(props, ["color", "color", "margin-top", "color"]);
    }

    /// Test server for `@font-face` loads: serves the vendored Ahem
    /// fixture (from `bowser_paint`'s fixtures, never inline) at the
    /// paths below, a woff-shaped decoy, and 404s elsewhere.
    fn font_server() -> TestServer {
        let ahem: Vec<u8> = include_bytes!("../../bowser_paint/tests/fixtures/Ahem.ttf").to_vec();
        let woff: Vec<u8> = include_bytes!("../../bowser_paint/tests/fixtures/Ahem.woff").to_vec();
        TestServer::start_with(move |path, _| {
            let css = vec![("Content-Type", "text/css".to_string())];
            let html = vec![("Content-Type", "text/html; charset=utf-8".to_string())];
            let font = vec![("Content-Type", "font/ttf".to_string())];
            match path {
                "/woff.html" => (
                    200,
                    "OK",
                    html,
                    b"<!DOCTYPE html><html><head><style>@font-face { font-family: 'Ahem'; \
                      src: url('/Ahem.woff'); } \
                      p { font-family: Ahem; }</style></head><body><p>X</p></body></html>"
                        .to_vec(),
                ),
                "/Ahem.woff" => (
                    200,
                    "OK",
                    vec![("Content-Type", "font/woff".to_string())],
                    woff.clone(),
                ),
                "/font.html" => (
                    200,
                    "OK",
                    html,
                    b"<!DOCTYPE html><html><head><style>@font-face { font-family: 'Ahem'; \
                      src: url('/fake.woff'), url('/missing.ttf'), url('/Ahem.ttf'); } \
                      p { font-family: Ahem; }</style></head><body><p>X</p></body></html>"
                        .to_vec(),
                ),
                "/brand.html" => (
                    200,
                    "OK",
                    html,
                    b"<!DOCTYPE html><html><head><style>@font-face { font-family: 'Brand'; \
                      src: url('/Ahem.ttf'); }</style></head><body><p>X</p></body></html>"
                        .to_vec(),
                ),
                "/Ahem.ttf" => (200, "OK", font.clone(), ahem.clone()),
                "/fake.woff" => (
                    200,
                    "OK",
                    vec![("Content-Type", "font/woff".to_string())],
                    b"wOFF\x00\x01\x00\x00not a font".to_vec(),
                ),
                "/linked.html" => (
                    200,
                    "OK",
                    html,
                    b"<!DOCTYPE html><html><head> \
                      <link rel=\"stylesheet\" href=\"/css/fonts.css\"> \
                      </head><body><p>X</p></body></html>"
                        .to_vec(),
                ),
                "/css/fonts.css" => (
                    200,
                    "OK",
                    css,
                    b"@font-face { font-family: 'Ahem'; src: url('Ahem.ttf'); }".to_vec(),
                ),
                // Only the stylesheet-relative path serves the font:
                // resolving against the document base (/Ahem.ttf)
                // would 404 here, proving the sheet-URL threading.
                "/css/Ahem.ttf" => (200, "OK", font, ahem.clone()),
                _ => (404, "Not Found", vec![], b"nope".to_vec()),
            }
        })
    }

    #[test]
    fn webfont_loads_from_http_and_changes_measurement() {
        use bowser_layout::FontMetrics;
        let server = font_server();
        let jar = Mutex::new(CookieJar::new());
        // No Ahem on the system: measurement starts on the embedded
        // fallback (Cousine, ~0.6em per char), never on Ahem (1em).
        let cache = FontCache::new();
        let before = {
            let ahem = cache.intern_family("Ahem");
            cache.advance("X", 50.0, ahem)
        };
        assert!(
            before < 40.0,
            "fallback advance must not be Ahem's 50, got {before}"
        );
        let fonts = Mutex::new(cache);
        let loaded = load_document_with_fonts(
            &server.url("/font.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
            &fonts,
        )
        .expect("load");
        // The woff decoy and the 404 were skipped in src order; the
        // third entry (Ahem) won.
        assert_eq!(loaded.webfonts.len(), 1);
        assert_eq!(loaded.webfonts[0].family, "Ahem");
        let after = {
            let guard = fonts.lock().expect("font cache lock poisoned");
            let ahem = guard.intern_family("Ahem");
            guard.advance("X", 50.0, ahem)
        };
        assert!(
            (after - 50.0).abs() < 0.01,
            "webfont advance must be Ahem's 50, got {after}"
        );
    }

    #[test]
    fn webfont_woff1_loads_via_loader_decode() {
        // w76: the loader's old SFNT gate dropped WOFF1 before the
        // paint worker's decode hook could see it. The loader now
        // decodes via `decode_webfont`, so a WOFF1-only rule lands
        // and measures as Ahem — and the stored bytes are SFNT, not
        // the wOFF file, proving the decode ran in the loader.
        use bowser_layout::FontMetrics;
        let server = font_server();
        let jar = Mutex::new(CookieJar::new());
        let fonts = Mutex::new(FontCache::new());
        let loaded = load_document_with_fonts(
            &server.url("/woff.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
            &fonts,
        )
        .expect("load");
        assert_eq!(loaded.webfonts.len(), 1);
        assert_eq!(loaded.webfonts[0].family, "Ahem");
        assert_eq!(
            &loaded.webfonts[0].bytes[..4],
            [0u8, 1, 0, 0],
            "stored bytes must be decoded SFNT, not the wOFF file"
        );
        let after = {
            let guard = fonts.lock().expect("font cache lock poisoned");
            let ahem = guard.intern_family("Ahem");
            guard.advance("X", 50.0, ahem)
        };
        assert!(
            (after - 50.0).abs() < 0.01,
            "woff webfont advance must be Ahem's 50, got {after}"
        );
    }

    /// Cross-origin font server for CORS tests: `TestServer` routes
    /// see only path + Range, so this minimal server records full
    /// request heads instead. It serves a cookie-priming page at
    /// `/prime.html` (its `Set-Cookie` must NOT leak into the later
    /// cross-origin font request — cookies are host-scoped, and the
    /// font host here shares the document host `127.0.0.1`, differing
    /// only by port, so the old shared-jar path would have leaked
    /// it) and the Ahem fixture at `/Ahem.ttf` with the given ACAO
    /// header (`None` = absent). A different port from the document
    /// server means a different origin.
    struct CorsFontServer {
        base: String,
        heads: Arc<Mutex<Vec<String>>>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl CorsFontServer {
        fn start(acao: Option<String>) -> Self {
            let ahem: Vec<u8> =
                include_bytes!("../../bowser_paint/tests/fixtures/Ahem.ttf").to_vec();
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind cors font server");
            let addr = listener.local_addr().expect("cors font server addr");
            let heads = Arc::new(Mutex::new(Vec::new()));
            let heads_srv = Arc::clone(&heads);
            let handle = thread::spawn(move || {
                for stream in listener.incoming() {
                    let Ok(mut s) = stream else { continue };
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 1024];
                    loop {
                        match s.read(&mut chunk) {
                            Ok(0) => break,
                            Ok(n) => {
                                buf.extend_from_slice(&chunk[..n]);
                                if buf.windows(4).any(|w| w == b"\r\n\r\n") || buf.len() > 8192 {
                                    break;
                                }
                            }
                            Err(_) => return,
                        }
                    }
                    let text = String::from_utf8_lossy(&buf).into_owned();
                    let target = text
                        .lines()
                        .next()
                        .and_then(|line| line.split_whitespace().nth(1))
                        .unwrap_or("/");
                    let path = target
                        .strip_prefix("http://")
                        .or_else(|| target.strip_prefix("https://"))
                        .and_then(|rest| rest.find('/').map(|i| &rest[i..]))
                        .unwrap_or(target);
                    let path = path.split('?').next().unwrap_or(path).to_owned();
                    if path == "/__shutdown" {
                        break;
                    }
                    heads_srv
                        .lock()
                        .expect("cors heads lock poisoned")
                        .push(text);
                    let (status, reason, head, body): (u16, &str, String, Vec<u8>) =
                        match path.as_str() {
                            "/prime.html" => (
                                200,
                                "OK",
                                "Content-Type: text/html\r\nSet-Cookie: probe=1; Path=/\r\n"
                                    .to_string(),
                                b"<!DOCTYPE html><html><body>prime</body></html>".to_vec(),
                            ),
                            "/Ahem.ttf" => {
                                let acao_head = acao.as_deref().map_or(String::new(), |v| {
                                    format!("Access-Control-Allow-Origin: {v}\r\n")
                                });
                                (
                                    200,
                                    "OK",
                                    format!("Content-Type: font/ttf\r\n{acao_head}"),
                                    ahem.clone(),
                                )
                            }
                            _ => (404, "Not Found", String::new(), b"nope".to_vec()),
                        };
                    let _ = s.write_all(
                        format!(
                            "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n{head}\r\n",
                            body.len()
                        )
                        .as_bytes(),
                    );
                    let _ = s.write_all(&body);
                }
            });
            Self {
                base: format!("http://{addr}"),
                heads,
                handle: Some(handle),
            }
        }

        fn url(&self, path: &str) -> Url {
            bowser_url::parse(&format!("{}{path}", self.base), None).expect("cors font url")
        }

        /// Request heads for the font fetch only (the priming page
        /// head is excluded).
        fn font_heads(&self) -> Vec<String> {
            self.heads
                .lock()
                .expect("cors heads lock poisoned")
                .iter()
                .filter(|h| {
                    h.lines()
                        .next()
                        .is_some_and(|line| line.contains("Ahem.ttf"))
                })
                .cloned()
                .collect()
        }

        fn shutdown(mut self) {
            if let Ok(mut s) = TcpStream::connect(self.base.trim_start_matches("http://")) {
                let _ = s.write_all(b"GET /__shutdown HTTP/1.1\r\n\r\n");
            }
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    /// Load a document from one test server whose single `@font-face`
    /// rule points at a cross-origin font server, with the ACAO value
    /// chosen by `make_acao` from the document origin. Returns the
    /// webfont count, the font request heads, and the document origin.
    fn cross_origin_font_load(
        make_acao: impl Fn(&str) -> Option<String>,
    ) -> (usize, Vec<String>, String) {
        let html = vec![("Content-Type", "text/html; charset=utf-8".to_string())];
        // The document server binds first, so its origin is known
        // before the font server's ACAO is fixed; the font URL is
        // slotted into the stylesheet at request time.
        let slot = Arc::new(Mutex::new(String::new()));
        let slot_srv = Arc::clone(&slot);
        let doc_server = TestServer::start_with(move |path, _| match path {
            "/cors.html" => {
                let font_url = slot_srv.lock().expect("slot lock poisoned").clone();
                (
                    200,
                    "OK",
                    html.clone(),
                    format!(
                        "<!DOCTYPE html><html><head><style>@font-face {{ font-family: 'Ahem'; \
                          src: url('{font_url}'); }} \
                          p {{ font-family: Ahem; }}</style></head><body><p>X</p></body></html>"
                    )
                    .into_bytes(),
                )
            }
            _ => (404, "Not Found", vec![], b"nope".to_vec()),
        });
        let doc_url = doc_server.url("/cors.html");
        let doc_origin = doc_url.origin_serialized();
        let font_server = CorsFontServer::start(make_acao(&doc_origin));
        *slot.lock().expect("slot lock poisoned") = font_server.url("/Ahem.ttf").serialize();
        let jar = Mutex::new(CookieJar::new());
        // Prime the jar with a cookie for the font host (same host,
        // different port): the cross-origin fetch must not send it.
        load_document_with_user_agent(
            &font_server.url("/prime.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
        )
        .expect("prime load");
        let fonts = Mutex::new(FontCache::new());
        let loaded =
            load_document_with_fonts(&doc_url, &jar, bowser_net::fetch::USER_AGENT, &fonts)
                .expect("load");
        let count = loaded.webfonts.len();
        let heads = font_server.font_heads();
        font_server.shutdown();
        (count, heads, doc_origin)
    }

    #[test]
    fn webfont_cross_origin_without_acao_is_rejected() {
        for make_acao in [
            (|_: &str| None) as fn(&str) -> Option<String>,
            (|_: &str| Some("https://evil.example".to_string())) as fn(&str) -> Option<String>,
        ] {
            let (count, heads, doc_origin) = cross_origin_font_load(make_acao);
            assert_eq!(
                count, 0,
                "cross-origin font without a valid ACAO must not load"
            );
            assert_eq!(heads.len(), 1, "exactly one font request expected");
            let head = &heads[0];
            assert!(
                head.contains(&format!("Origin: {doc_origin}")),
                "cross-origin request must carry Origin: {doc_origin}, got:\n{head}"
            );
            assert!(
                !head
                    .lines()
                    .any(|line| line.to_lowercase().starts_with("cookie:")),
                "cross-origin request must not send jar cookies, got:\n{head}"
            );
        }
    }

    #[test]
    fn webfont_cross_origin_with_acao_star_or_exact_origin_loads() {
        for make_acao in [
            (|_: &str| Some("*".to_string())) as fn(&str) -> Option<String>,
            (|origin: &str| Some(origin.to_string())) as fn(&str) -> Option<String>,
        ] {
            let (count, heads, doc_origin) = cross_origin_font_load(make_acao);
            assert_eq!(count, 1, "cross-origin font with valid ACAO must load");
            assert_eq!(heads.len(), 1, "exactly one font request expected");
            assert!(
                heads[0].contains(&format!("Origin: {doc_origin}")),
                "cross-origin request must carry Origin: {doc_origin}"
            );
        }
    }

    #[test]
    fn webfont_src_resolves_against_stylesheet_url() {
        use bowser_layout::FontMetrics;
        let server = font_server();
        let jar = Mutex::new(CookieJar::new());
        let cache = FontCache::new();
        let fonts = Mutex::new(cache);
        let loaded = load_document_with_fonts(
            &server.url("/linked.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
            &fonts,
        )
        .expect("load");
        // The sheet lives at /css/fonts.css, so its relative
        // `url('Ahem.ttf')` must fetch /css/Ahem.ttf — not /Ahem.ttf
        // (which this server 404s, so a document-base resolution
        // would record no webfont at all).
        assert_eq!(loaded.webfonts.len(), 1);
        let after = {
            let guard = fonts.lock().expect("font cache lock poisoned");
            let ahem = guard.intern_family("Ahem");
            guard.advance("X", 50.0, ahem)
        };
        assert!(
            (after - 50.0).abs() < 0.01,
            "stylesheet-relative webfont must measure as Ahem, got {after}"
        );
    }

    #[test]
    fn webfont_renamed_family_resolves_to_webfont() {
        // w75f1: the loader test that masked the discard — declared
        // name == internal name — is `webfont_loads_from_http...`
        // above. Here the rule renames the file: Brand must measure
        // as Ahem, not as the fallback.
        use bowser_layout::FontMetrics;
        let server = font_server();
        let jar = Mutex::new(CookieJar::new());
        let fonts = Mutex::new(FontCache::new());
        let loaded = load_document_with_fonts(
            &server.url("/brand.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
            &fonts,
        )
        .expect("load");
        assert_eq!(loaded.webfonts.len(), 1);
        assert_eq!(loaded.webfonts[0].family, "Brand");
        let after = {
            let guard = fonts.lock().expect("font cache lock poisoned");
            let brand = guard.intern_family("Brand");
            guard.advance("X", 50.0, brand)
        };
        assert!(
            (after - 50.0).abs() < 0.01,
            "Brand advance must be Ahem's 50, got {after}"
        );
    }

    #[test]
    fn webfont_same_file_two_families_reuses_bytes() {
        // w75f1: URL-alone dedupe dropped the second rule. Both
        // rules must land, and the file must be fetched only once.
        use bowser_layout::FontMetrics;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let ahem: Vec<u8> = include_bytes!("../../bowser_paint/tests/fixtures/Ahem.ttf").to_vec();
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_srv = Arc::clone(&hits);
        let server = TestServer::start_with(move |path, _| {
            let html = vec![("Content-Type", "text/html; charset=utf-8".to_string())];
            let font = vec![("Content-Type", "font/ttf".to_string())];
            match path {
                "/two.html" => (
                    200,
                    "OK",
                    html,
                    b"<!DOCTYPE html><html><head><style> \
                      @font-face { font-family: 'Alpha'; src: url('/Ahem.ttf'); } \
                      @font-face { font-family: 'Beta'; src: url('/Ahem.ttf'); } \
                      </style></head><body><p>X</p></body></html>"
                        .to_vec(),
                ),
                "/Ahem.ttf" => {
                    hits_srv.fetch_add(1, Ordering::Relaxed);
                    (200, "OK", font, ahem.clone())
                }
                _ => (404, "Not Found", vec![], b"nope".to_vec()),
            }
        });
        let jar = Mutex::new(CookieJar::new());
        let fonts = Mutex::new(FontCache::new());
        let loaded = load_document_with_fonts(
            &server.url("/two.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
            &fonts,
        )
        .expect("load");
        assert_eq!(loaded.webfonts.len(), 2, "both rules must land");
        assert_eq!(loaded.webfonts[0].family, "Alpha");
        assert_eq!(loaded.webfonts[1].family, "Beta");
        assert_eq!(
            hits.load(Ordering::Relaxed),
            1,
            "second rule must reuse the fetched bytes, not refetch"
        );
        let (alpha, beta) = {
            let guard = fonts.lock().expect("font cache lock poisoned");
            let a = guard.intern_family("Alpha");
            let b = guard.intern_family("Beta");
            (guard.advance("X", 50.0, a), guard.advance("X", 50.0, b))
        };
        assert!(
            (alpha - 50.0).abs() < 0.01 && (beta - 50.0).abs() < 0.01,
            "Alpha/Beta advances must be Ahem's 50, got {alpha}/{beta}"
        );
    }

    #[test]
    fn webfont_in_imported_sheet_is_fetched() {
        // w75f1: @font-face rules in @imported sheets must be
        // fetched (recursed, with cycle/depth caps), not dropped
        // with the skipped @import.
        use bowser_layout::FontMetrics;
        let ahem: Vec<u8> = include_bytes!("../../bowser_paint/tests/fixtures/Ahem.ttf").to_vec();
        let server = TestServer::start_with(move |path, _| {
            let html = vec![("Content-Type", "text/html; charset=utf-8".to_string())];
            let css = vec![("Content-Type", "text/css".to_string())];
            let font = vec![("Content-Type", "font/ttf".to_string())];
            match path {
                "/import.html" => (
                    200,
                    "OK",
                    html,
                    b"<!DOCTYPE html><html><head> \
                      <link rel=\"stylesheet\" href=\"/css/main.css\"> \
                      </head><body><p>X</p></body></html>"
                        .to_vec(),
                ),
                "/css/main.css" => (
                    200,
                    "OK",
                    css.clone(),
                    b"@import url('fonts.css'); @import url('cycle.css');".to_vec(),
                ),
                "/css/fonts.css" => (
                    200,
                    "OK",
                    css.clone(),
                    b"@font-face { font-family: 'Imported'; src: url('Ahem.ttf'); }".to_vec(),
                ),
                // Self-importing sheet: the cycle cap must terminate
                // the walk, and the sheet's own face still lands.
                "/css/cycle.css" => (
                    200,
                    "OK",
                    css,
                    b"@import url('cycle.css'); \
                      @font-face { font-family: 'Cycled'; src: url('Ahem.ttf'); }"
                        .to_vec(),
                ),
                "/css/Ahem.ttf" => (200, "OK", font, ahem.clone()),
                _ => (404, "Not Found", vec![], b"nope".to_vec()),
            }
        });
        let jar = Mutex::new(CookieJar::new());
        let fonts = Mutex::new(FontCache::new());
        let loaded = load_document_with_fonts(
            &server.url("/import.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
            &fonts,
        )
        .expect("load");
        let families: Vec<&str> = loaded.webfonts.iter().map(|w| w.family.as_str()).collect();
        assert!(
            families.contains(&"Imported") && families.contains(&"Cycled"),
            "imported-sheet faces must land, got {families:?}"
        );
        let after = {
            let guard = fonts.lock().expect("font cache lock poisoned");
            let imported = guard.intern_family("Imported");
            guard.advance("X", 50.0, imported)
        };
        assert!(
            (after - 50.0).abs() < 0.01,
            "Imported advance must be Ahem's 50, got {after}"
        );
    }

    #[test]
    fn webfont_aggregate_budget_stops_fetching() {
        // w75f1: unique fetched bytes past WEBFONT_BUDGET_BYTES
        // (192 MiB) stop the walk. Seven 28 MiB faces: the first six
        // (168 MiB) land, the seventh would cross the budget and is
        // dropped, so fetching stops there. Bodies are SFNT-magic
        // only (the fetch gate checks the magic, not full parse), so
        // the install step rejects them — the count here pins the
        // fetch/budget behavior, not the parse.
        let server = TestServer::start_with(move |path, _| {
            let html = vec![("Content-Type", "text/html; charset=utf-8".to_string())];
            let font = vec![("Content-Type", "font/ttf".to_string())];
            if path == "/budget.html" {
                let mut style = String::from("<!DOCTYPE html><html><head><style>");
                for i in 1..=7 {
                    style.push_str(&format!(
                        "@font-face {{ font-family: 'Big{i}'; src: url('/big{i}.ttf'); }} "
                    ));
                }
                style.push_str("</style></head><body><p>X</p></body></html>");
                return (200, "OK", html, style.into_bytes());
            }
            if let Some(rest) = path.strip_prefix("/big")
                && let Some(num) = rest.strip_suffix(".ttf")
                && num.parse::<u32>().is_ok()
            {
                let mut body = vec![0u8; 28 * 1024 * 1024];
                body[0] = 0;
                body[1] = 1;
                return (200, "OK", font, body);
            }
            (404, "Not Found", vec![], b"nope".to_vec())
        });
        let jar = Mutex::new(CookieJar::new());
        let loaded = load_document_with_user_agent(
            &server.url("/budget.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
        )
        .expect("load");
        assert_eq!(
            loaded.webfonts.len(),
            6,
            "six 28 MiB faces fit the 192 MiB budget, the seventh must not"
        );
    }

    #[test]
    fn images_decode_for_real_and_pixels_are_cached() {
        let server = TestServer::start();
        let jar = Mutex::new(CookieJar::new());
        let loaded = load_document_with_user_agent(
            &server.url("/index.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
        )
        .expect("load");

        // Both images decode for real: intrinsics come from the decode,
        // keyed by serialized request URL (the install re-keys by URL
        // when it walks the scripted document), and the pixels land in
        // the loader image cache for paint.
        let base = server.url("/index.html");
        assert_eq!(loaded.image_sizes.len(), 2);
        for src in ["/img.png", "/img.gif"] {
            let key = bowser_url::parse(src, Some(&base))
                .expect("test url")
                .serialize();
            assert_eq!(loaded.image_sizes[&key], IntrinsicDims::from_size(4.0, 2.0));
        }

        // The cache is keyed by serialized request URL, exactly as the
        // loader computed it (src resolved against the document URL).
        for src in ["/img.png", "/img.gif"] {
            let key = bowser_url::parse(src, Some(&base))
                .expect("test url")
                .serialize();
            let img = cached_image(&key).expect("pixels cached for paint");
            assert_eq!((img.width, img.height), (4, 2));
            assert_eq!(img.rgba.len(), 4 * 2 * 4);
        }
        // The PNG fixture is solid red; spot-check the cached pixels.
        let key = bowser_url::parse("/img.png", Some(&base))
            .expect("test url")
            .serialize();
        let png = cached_image(&key).expect("png pixels cached");
        assert!(
            png.rgba
                .as_chunks::<4>()
                .0
                .iter()
                .all(|px| px == &[255, 0, 0, 255]),
            "cached PNG pixels must be solid red"
        );

        // /broken.png (404): no entry, no failure.
        assert!(img_node_with_src(&loaded.document, "/broken.png").is_some());
    }

    #[test]
    fn non_html_documents_are_rejected() {
        let server = TestServer::start();
        let jar = Mutex::new(CookieJar::new());

        let err = load_document_with_user_agent(
            &server.url("/file.txt"),
            &jar,
            bowser_net::fetch::USER_AGENT,
        )
        .expect_err("text/plain");
        assert!(matches!(err, LoadError::UnsupportedContentType(_)));

        let err = load_document_with_user_agent(
            &server.url("/missing"),
            &jar,
            bowser_net::fetch::USER_AGENT,
        )
        .expect_err("404 page");
        assert!(matches!(err, LoadError::HttpStatus(404)));
    }

    #[test]
    fn redirects_are_followed_and_traced() {
        let server = TestServer::start();
        let jar = Mutex::new(CookieJar::new());
        let loaded = load_document_with_user_agent(
            &server.url("/redir"),
            &jar,
            bowser_net::fetch::USER_AGENT,
        )
        .expect("load");

        assert!(loaded.url.serialize().ends_with("/index.html"));
        assert_eq!(loaded.trace.hops.len(), 2);
        assert!(loaded.trace.hops[0].serialize().ends_with("/redir"));
    }

    #[test]
    fn base_href_resolves_relative_subresources() {
        let server = TestServer::start();
        let jar = Mutex::new(CookieJar::new());
        let loaded = load_document_with_user_agent(
            &server.url("/sub/page.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
        )
        .expect("load");

        // theme.css resolved against <base href="/assets/">.
        assert_eq!(loaded.stylesheets.len(), 1);
        assert_eq!(loaded.stylesheets[0].rules().len(), 1);
    }

    #[test]
    fn cached_image_for_src_honors_base_href() {
        let server = TestServer::start();
        let jar = Mutex::new(CookieJar::new());
        let doc_url = server.url("/sub/gallery.html");
        let loaded = load_document_with_user_agent(&doc_url, &jar, bowser_net::fetch::USER_AGENT)
            .expect("load");

        // The relative src resolves against <base href="/assets/">, so the
        // helper finds the decode the loader cached under that key.
        let img = cached_image_for_src(&loaded.document, &doc_url, "pic.png")
            .expect("cached via base href");
        assert_eq!((img.width, img.height), (2, 2));
        assert!(
            img.rgba
                .as_chunks::<4>()
                .0
                .iter()
                .all(|px| px == &[0, 255, 0, 255]),
            "cached pic.png pixels must be solid green"
        );

        // Same document, absolute src that was never loaded: no entry.
        // (Each test server binds its own port, so keys cannot collide
        // with other tests' cached images.)
        assert!(cached_image_for_src(&loaded.document, &doc_url, "/img.png").is_none());
        // Unresolvable src: no entry, no panic.
        assert!(cached_image_for_src(&loaded.document, &doc_url, "http://[::1").is_none());
    }

    #[test]
    fn background_images_are_fetched_and_cached() {
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_srv = Arc::clone(&hits);
        let server = TestServer::start_with(move |path, _| {
            if path == "/bg/bg.png" {
                hits_srv.fetch_add(1, Ordering::Relaxed);
            }
            route(path)
        });
        let jar = Mutex::new(CookieJar::new());
        let doc_url = server.url("/bg/page.html");
        let loaded = load_document_with_user_agent(&doc_url, &jar, bowser_net::fetch::USER_AGENT)
            .expect("load");

        // The loader prefetches the div's `background-image` (set via the
        // `background` shorthand, so this also covers shorthand
        // expansion). The lookup below passes the raw CssValue::Url
        // string exactly the way paint's image resolver does, so a hit
        // here means paint will find the pixels instead of a placeholder.
        let img = cached_image_for_src(&loaded.document, &doc_url, "bg.png")
            .expect("background image cached for paint");
        assert_eq!((img.width, img.height), (3, 1));
        assert!(
            img.rgba
                .as_chunks::<4>()
                .0
                .iter()
                .all(|px| px == &[0, 0, 255, 255]),
            "cached bg.png pixels must be solid blue"
        );
        // Background images do not feed image_sizes (only `<img>`).
        assert!(loaded.image_sizes.is_empty());
        assert_eq!(hits.load(Ordering::Relaxed), 1, "image fetched once");

        // A second load of the same document is served from the shared
        // image cache: no refetch.
        let loaded2 = load_document_with_user_agent(&doc_url, &jar, bowser_net::fetch::USER_AGENT)
            .expect("reload");
        assert!(cached_image_for_src(&loaded2.document, &doc_url, "bg.png").is_some());
        assert_eq!(hits.load(Ordering::Relaxed), 1, "no refetch on second load");
    }

    /// Parse `Range: bytes=<start>-<end>` against `total` for the test
    /// server. `None` when absent or unsatisfiable.
    fn parse_test_range(range: &str, total: usize) -> Option<(usize, usize)> {
        let range = range.strip_prefix("bytes=")?;
        let (start, end) = range.split_once('-')?;
        let start: usize = start.parse().ok()?;
        let end: usize = if end.is_empty() {
            total.saturating_sub(1)
        } else {
            end.parse().ok()?
        };
        if start >= total || end < start {
            return None;
        }
        Some((start, end.min(total - 1)))
    }

    /// Serve `body` honoring `Range` with 206 + `Content-Range`; without a
    /// usable `Range` header, serve 200 with the full body. `hits` counts
    /// requests so tests can assert follow-ups happened (or didn't).
    fn range_response(body: &[u8], range: Option<&str>, hits: &AtomicUsize) -> Route {
        hits.fetch_add(1, Ordering::Relaxed);
        let total = body.len();
        let mut headers = vec![
            ("Content-Type", "image/png".to_string()),
            ("Accept-Ranges", "bytes".to_string()),
        ];
        match range.and_then(|r| parse_test_range(r, total)) {
            Some((start, end)) => {
                headers.push(("Content-Range", format!("bytes {start}-{end}/{total}")));
                (206, "Partial Content", headers, body[start..=end].to_vec())
            }
            None => (200, "OK", headers, body.to_vec()),
        }
    }

    #[test]
    fn images_larger_than_range_window_decode_via_followup_ranges() {
        // 200x200 RGBA: ~160 KiB of stored PNG, well past the 64 KiB first
        // window. One distinct color per row, so a truncation would fail
        // the far-row pixel check below.
        let mut pixels = Vec::with_capacity(200 * 200 * 4);
        for y in 0..200u32 {
            for _ in 0..200 {
                pixels.extend_from_slice(&[(y % 256) as u8, 0, 128, 255]);
            }
        }
        let png = Arc::new(png_bytes(200, 200, &pixels));
        assert!(
            png.len() > IMAGE_FETCH_BYTES,
            "fixture must exceed the first window"
        );
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_srv = Arc::clone(&hits);
        let server = TestServer::start_with(move |path, range| match path {
            "/big.html" => (
                200,
                "OK",
                vec![("Content-Type", "text/html".to_string())],
                b"<!DOCTYPE html><html><body><img src=\"/big.png\"></body></html>".to_vec(),
            ),
            "/big.png" => range_response(&png, range, &hits_srv),
            _ => route(path),
        });
        let jar = Mutex::new(CookieJar::new());
        let loaded = load_document_with_user_agent(
            &server.url("/big.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
        )
        .expect("load");

        assert!(
            hits.load(Ordering::Relaxed) > 1,
            "the loader must issue follow-up range requests past the first window"
        );
        let base = server.url("/big.html");
        let key = bowser_url::parse("/big.png", Some(&base))
            .expect("test url")
            .serialize();
        assert_eq!(
            loaded.image_sizes[&key],
            IntrinsicDims::from_size(200.0, 200.0)
        );
        let img = cached_image(&key).expect("pixels cached for paint");
        assert_eq!((img.width, img.height), (200, 200));
        // A pixel from row 150 lives past the first 64 KiB: it only
        // decodes when the follow-up ranges arrived.
        let px = &img.rgba[150 * 200 * 4..150 * 200 * 4 + 4];
        assert_eq!(px, &[150, 0, 128, 255]);
    }

    #[test]
    fn images_decode_when_server_ignores_range() {
        // ~160 KiB PNG, like above, but the server answers 200 with the
        // full body despite the Range header: the loader must decode the
        // whole response instead of slicing the first window.
        let png = Arc::new(png_bytes(200, 200, &[9u8, 9, 9, 255].repeat(200 * 200)));
        assert!(
            png.len() > IMAGE_FETCH_BYTES,
            "fixture must exceed the first window"
        );
        let server = TestServer::start_with(move |path, _| match path {
            "/big.html" => (
                200,
                "OK",
                vec![("Content-Type", "text/html".to_string())],
                b"<!DOCTYPE html><html><body><img src=\"/big.png\"></body></html>".to_vec(),
            ),
            "/big.png" => (
                200,
                "OK",
                vec![("Content-Type", "image/png".to_string())],
                png.to_vec(),
            ),
            _ => route(path),
        });
        let jar = Mutex::new(CookieJar::new());
        let loaded = load_document_with_user_agent(
            &server.url("/big.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
        )
        .expect("load");

        let base = server.url("/big.html");
        let key = bowser_url::parse("/big.png", Some(&base))
            .expect("test url")
            .serialize();
        assert_eq!(
            loaded.image_sizes[&key],
            IntrinsicDims::from_size(200.0, 200.0)
        );
    }

    #[test]
    fn images_with_over_cap_content_range_total_are_skipped() {
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_srv = Arc::clone(&hits);
        let server = TestServer::start_with(move |path, range| match path {
            "/page.html" => (
                200,
                "OK",
                vec![("Content-Type", "text/html".to_string())],
                b"<!DOCTYPE html><html><body><img src=\"/huge.png\"></body></html>".to_vec(),
            ),
            "/huge.png" => {
                hits_srv.fetch_add(1, Ordering::Relaxed);
                assert!(range.is_some(), "first image request must use Range");
                // Advertises a 1 GiB total while serving only junk: the
                // loader must bail on the advertised total without issuing
                // any follow-up requests.
                (
                    206,
                    "Partial Content",
                    vec![
                        ("Content-Type", "image/png".to_string()),
                        (
                            "Content-Range",
                            format!("bytes 0-{}/1073741824", IMAGE_FETCH_BYTES - 1),
                        ),
                    ],
                    vec![0u8; 1024],
                )
            }
            _ => route(path),
        });
        let jar = Mutex::new(CookieJar::new());
        let loaded = load_document_with_user_agent(
            &server.url("/page.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
        )
        .expect("load");

        assert!(loaded.image_sizes.is_empty(), "over-cap image skipped");
        assert_eq!(
            hits.load(Ordering::Relaxed),
            1,
            "no follow-up requests for an over-cap advertised total"
        );
    }

    /// `fetch_late_image` maps every failure mode to `None`: fetch error,
    /// non-2xx status, over-cap body, and decode failure. (The
    /// success/cache-hit path is covered by the window worker's
    /// late-image test.)
    #[test]
    fn fetch_late_image_failure_modes_map_to_none() {
        let server = TestServer::start_with(|path, _| match path {
            "/junk.png" => (
                200,
                "OK",
                vec![("Content-Type", "image/png".to_string())],
                b"this is not image data".to_vec(),
            ),
            "/huge.png" => (
                206,
                "Partial Content",
                vec![
                    ("Content-Type", "image/png".to_string()),
                    (
                        "Content-Range",
                        format!("bytes 0-1023/{}", IMAGE_MAX_BYTES as u64 + 1),
                    ),
                ],
                vec![0u8; 1024],
            ),
            _ => route(path),
        });
        let jar = Mutex::new(CookieJar::new());
        let doc_url = server.url("/page.html");
        // Non-2xx (the default route arm serves 404), decode failure
        // (200 with garbage), over-cap (206 advertising past the cap).
        for path in ["/missing.png", "/junk.png", "/huge.png"] {
            let url = server.url(path);
            assert!(
                fetch_late_image(&url, &doc_url, &jar, "bowser-test").is_none(),
                "{path} must map to None"
            );
        }

        // Fetch error: nothing listening on this port (bound to learn a
        // free port, then dropped, so connects are refused).
        let port = {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind probe");
            listener.local_addr().expect("probe addr").port()
        };
        let url =
            bowser_url::parse(&format!("http://127.0.0.1:{port}/img.png"), None).expect("test url");
        assert!(
            fetch_late_image(&url, &doc_url, &jar, "bowser-test").is_none(),
            "connection refused must map to None"
        );
    }

    /// Drive `fetch_image_bytes` directly against a test server, bypassing
    /// HTML parsing and image decoding.
    fn fetch_image_bytes_for_test(server: &TestServer, path: &str) -> Option<Vec<u8>> {
        let jar = Mutex::new(CookieJar::new());
        let url = server.url(path);
        let sub = SubresourceFetch {
            jar: &jar,
            user_agent: "bowser-test",
            document_origin: url.origin(),
            nav: NavigationContext {
                top_host: url.host().map(str::to_owned),
                top_secure: Some(url.scheme() == "https"),
                top_level_safe_navigation: false,
            },
        };
        fetch_image_bytes(&url, &sub)
    }

    #[test]
    fn image_exactly_at_cap_succeeds() {
        // A 206 sequence whose Content-Range total is exactly
        // IMAGE_MAX_BYTES and which delivers exactly that many bytes is
        // legal: the known total must be checked before the cap, so the
        // final chunk is not rejected as over-cap.
        let total = IMAGE_MAX_BYTES;
        let server = TestServer::start_with(move |path, range| match path {
            "/exact.png" => {
                assert!(range.is_some(), "first image request must use Range");
                // Serve the whole resource in one 206 regardless of the
                // requested window: the trigger is `next == total == cap`
                // after the final chunk.
                (
                    206,
                    "Partial Content",
                    vec![
                        ("Content-Type", "image/png".to_string()),
                        ("Content-Range", format!("bytes 0-{}/{}", total - 1, total)),
                    ],
                    vec![0xABu8; total],
                )
            }
            _ => route(path),
        });
        let bytes = fetch_image_bytes_for_test(&server, "/exact.png").expect("at-cap image loads");
        assert_eq!(bytes.len(), IMAGE_MAX_BYTES);
    }

    #[test]
    fn oversized_followup_chunk_is_rejected_before_extending() {
        // ~cap bytes assembled, then a 206 body far larger than the small
        // requested range: assembly must return None without ever growing
        // the buffer past the cap (the check runs before `extend`).
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_srv = Arc::clone(&hits);
        let server = TestServer::start_with(move |path, range| match path {
            "/bloated.png" => {
                let n = hits_srv.fetch_add(1, Ordering::Relaxed);
                if n == 0 {
                    assert!(range.is_some(), "first image request must use Range");
                    (
                        206,
                        "Partial Content",
                        vec![
                            ("Content-Type", "image/png".to_string()),
                            (
                                "Content-Range",
                                format!("bytes 0-{}/{}", IMAGE_MAX_BYTES - 2, IMAGE_MAX_BYTES),
                            ),
                        ],
                        vec![0xCDu8; IMAGE_MAX_BYTES - 1],
                    )
                } else {
                    // A small range was requested; the server misbehaves and
                    // answers with a body that would push assembly over the
                    // cap.
                    (
                        206,
                        "Partial Content",
                        vec![("Content-Type", "image/png".to_string())],
                        vec![0xEFu8; IMAGE_MAX_BYTES],
                    )
                }
            }
            _ => route(path),
        });
        assert!(
            fetch_image_bytes_for_test(&server, "/bloated.png").is_none(),
            "over-cap follow-up chunk must fail"
        );
        assert_eq!(
            hits.load(Ordering::Relaxed),
            2,
            "the oversized chunk was fetched, then rejected before extending"
        );
    }

    #[test]
    fn short_first_chunk_without_content_range_terminates() {
        // A short first 206 with no Content-Range is the final chunk: the
        // loader must keep it and issue no follow-up requests.
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_srv = Arc::clone(&hits);
        let server = TestServer::start_with(move |path, _| match path {
            "/short.png" => {
                hits_srv.fetch_add(1, Ordering::Relaxed);
                (
                    206,
                    "Partial Content",
                    vec![("Content-Type", "image/png".to_string())],
                    vec![0x11u8; 1000],
                )
            }
            _ => route(path),
        });
        let bytes = fetch_image_bytes_for_test(&server, "/short.png")
            .expect("short first chunk is the whole body");
        assert_eq!(bytes.len(), 1000);
        assert_eq!(
            hits.load(Ordering::Relaxed),
            1,
            "no follow-up requests after a short first chunk"
        );
    }

    #[test]
    fn script_srcs_lists_external_scripts() {
        let server = TestServer::start();
        let jar = Mutex::new(CookieJar::new());
        let loaded = load_document_with_user_agent(
            &server.url("/index.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
        )
        .expect("load");

        let srcs = script_srcs(&loaded.document, &loaded.url);
        assert_eq!(srcs.len(), 1);
        assert!(srcs[0].serialize().ends_with("/app.js"));
    }

    #[test]
    fn script_step_prefetches_external_sources() {
        let server = TestServer::start();
        let jar = Mutex::new(CookieJar::new());
        // INDEX_HTML carries one external classic script.
        let loaded = load_document_with_user_agent(
            &server.url("/index.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
        )
        .expect("load");

        assert_eq!(loaded.script_sources.len(), 1);
        let (key, source) = loaded.script_sources.iter().next().unwrap();
        assert!(key.ends_with("/app.js"), "key: {key}");
        assert_eq!(source.as_deref(), Ok("var fromServer = 42;"));
    }

    #[test]
    fn script_step_records_failed_fetches_and_skips_non_classic() {
        let server = TestServer::start_with(|path, _| match path {
            "/page.html" => (
                200,
                "OK",
                vec![("Content-Type", "text/html".to_string())],
                b"<!DOCTYPE html><html><head>\
                  <script>var inlineOk = 1;</script>\
                  <script src=\"/bad.js\"></script>\
                  <script type=\"module\">import x from './m.js';</script>\
                  <script type=\"application/json\">{\"a\":1}</script>\
                  </head><body></body></html>"
                    .to_vec(),
            ),
            _ => route(path),
        });
        let jar = Mutex::new(CookieJar::new());
        let loaded = load_document_with_user_agent(
            &server.url("/page.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
        )
        .expect("load");

        // Only the 404'd classic external script lands in the prefetch
        // map (as an error); inline scripts, modules, and data blocks
        // are never fetched. Classification into skips happens in
        // `resolve_scripts` at install time.
        assert_eq!(loaded.script_sources.len(), 1);
        let (key, source) = loaded.script_sources.iter().next().unwrap();
        assert!(key.ends_with("/bad.js"), "key: {key}");
        assert!(source.is_err());
    }

    #[test]
    fn script_src_resolves_against_base_href() {
        let server = TestServer::start_with(|path, _| match path {
            "/page.html" => (
                200,
                "OK",
                vec![("Content-Type", "text/html".to_string())],
                b"<!DOCTYPE html><html><head><base href=\"/assets/\">\
                  </head><body><script src=\"lib.js\"></script></body></html>"
                    .to_vec(),
            ),
            "/assets/lib.js" => (
                200,
                "OK",
                vec![("Content-Type", "text/javascript".to_string())],
                b"var fromBase = true;".to_vec(),
            ),
            _ => route(path),
        });
        let jar = Mutex::new(CookieJar::new());
        let loaded = load_document_with_user_agent(
            &server.url("/page.html"),
            &jar,
            bowser_net::fetch::USER_AGENT,
        )
        .expect("load");

        assert_eq!(loaded.script_sources.len(), 1);
        let (key, source) = loaded.script_sources.iter().next().unwrap();
        assert!(key.ends_with("/assets/lib.js"), "unexpected key: {key}");
        assert_eq!(source.as_deref(), Ok("var fromBase = true;"));
    }

    #[test]
    fn script_sources_are_capped_per_document() {
        use crate::script::MAX_SCRIPTS;

        let mut html = String::from("<!DOCTYPE html><html><head>");
        for i in 0..MAX_SCRIPTS + 5 {
            html.push_str(&format!(r#"<script src="/s{i}.js"></script>"#));
        }
        html.push_str("</head><body></body></html>");
        let html = Arc::new(html);
        let server = TestServer::start_with(move |path, _| {
            if path == "/" {
                (
                    200,
                    "OK",
                    vec![("Content-Type", "text/html".to_string())],
                    html.as_bytes().to_vec(),
                )
            } else if path.starts_with("/s") && path.ends_with(".js") {
                (
                    200,
                    "OK",
                    vec![("Content-Type", "text/javascript".to_string())],
                    b"var s = 1;".to_vec(),
                )
            } else {
                route(path)
            }
        });
        let jar = Mutex::new(CookieJar::new());
        let loaded =
            load_document_with_user_agent(&server.url("/"), &jar, bowser_net::fetch::USER_AGENT)
                .expect("load");

        // Extra scripts are never fetched past the cap: the prefetch
        // map records the over-cap URLs as errors instead.
        assert_eq!(loaded.script_sources.len(), MAX_SCRIPTS + 5);
        let ok = loaded.script_sources.values().filter(|r| r.is_ok()).count();
        let err = loaded
            .script_sources
            .values()
            .filter(|r| r.is_err())
            .count();
        assert_eq!(ok, MAX_SCRIPTS, "first MAX_SCRIPTS scripts fetched");
        assert_eq!(err, 5, "over-cap scripts recorded as errors");
    }

    #[test]
    fn image_cache_evicts_fifo_under_budget() {
        let mut cache = ImageCache::new();
        let img = |fill: u8| DecodedImage {
            width: 1,
            height: 1,
            rgba: vec![fill, fill, fill, 255],
        };
        // Budget fits two 4-byte images; the third evicts the first.
        cache.insert_budgeted("a".into(), img(1), 8);
        cache.insert_budgeted("b".into(), img(2), 8);
        assert!(cache.get("a").is_some());
        cache.insert_budgeted("c".into(), img(3), 8);
        assert!(cache.get("a").is_none(), "oldest entry evicted");
        assert!(cache.get("b").is_some());
        assert!(cache.get("c").is_some());
        // Re-inserting moves the key to the back of the FIFO.
        cache.insert_budgeted("b".into(), img(4), 8);
        cache.insert_budgeted("d".into(), img(5), 8);
        assert!(cache.get("c").is_none(), "c now oldest, evicted");
        assert_eq!(cache.get("b").expect("b kept").rgba[0], 4);
        // A single image larger than the whole budget is never cached.
        cache.insert_budgeted("huge".into(), img(6), 3);
        assert!(cache.get("huge").is_none());
    }

    #[test]
    fn subresource_counts_are_capped() {
        let links = MAX_STYLESHEETS + 22;
        let imgs = MAX_IMAGES + 44;
        let mut html = String::from("<!DOCTYPE html><html><head>");
        for _ in 0..links {
            html.push_str("<link rel=\"stylesheet\" href=\"/a.css\">");
        }
        html.push_str("</head><body>");
        for i in 0..imgs {
            html.push_str(&format!("<img src=\"/img{i}.png\">"));
        }
        html.push_str("</body></html>");
        let html = Arc::new(html);

        let server = TestServer::start_with(move |path, _| {
            if path == "/" {
                (
                    200,
                    "OK",
                    vec![("Content-Type", "text/html".to_string())],
                    html.as_bytes().to_vec(),
                )
            } else if path.starts_with("/img") && path.ends_with(".png") {
                // Distinct URLs so the cap is observable: intrinsics are
                // keyed by serialized request URL.
                (
                    200,
                    "OK",
                    vec![("Content-Type", "image/png".to_string())],
                    png_bytes(4, 2, &[255u8, 0, 0, 255].repeat(8)),
                )
            } else {
                route(path)
            }
        });
        let jar = Mutex::new(CookieJar::new());
        let loaded =
            load_document_with_user_agent(&server.url("/"), &jar, bowser_net::fetch::USER_AGENT)
                .expect("load");

        // Extra links/images are skipped, never fetched.
        assert_eq!(loaded.stylesheets.len(), MAX_STYLESHEETS);
        assert_eq!(loaded.image_sizes.len(), MAX_IMAGES);
    }

    #[test]
    fn configured_user_agent_sent_on_all_requests_and_images_use_range() {
        #[derive(Default)]
        struct Seen {
            user_agents: Vec<String>,
            ranges: Vec<Option<String>>,
        }
        let seen = Arc::new(Mutex::new(Seen::default()));
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let seen_srv = Arc::clone(&seen);
        // Document, one stylesheet, one image: exactly three requests.
        let server = thread::spawn(move || {
            for _ in 0..3 {
                let (mut s, _) = listener.accept().expect("accept");
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    match s.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            buf.extend_from_slice(&chunk[..n]);
                            if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                                break;
                            }
                        }
                    }
                }
                let req = String::from_utf8_lossy(&buf).into_owned();
                let header = |name: &str| {
                    req.lines()
                        .find(|l| l.to_ascii_lowercase().starts_with(&format!("{name}:")))
                        .map(|l| l[name.len() + 1..].trim().to_owned())
                };
                let target = req
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/");
                let path = target
                    .rsplit('/')
                    .next()
                    .map(|last| format!("/{last}"))
                    .unwrap_or_default();
                let mut seen = seen_srv.lock().unwrap();
                seen.user_agents
                    .push(header("user-agent").unwrap_or_default());
                seen.ranges.push(header("range"));
                drop(seen);
                // The "image" is padded past the old 64 KiB fetch window;
                // the server ignores the Range request (200, full body),
                // so the loader decodes the whole response.
                let (content_type, body): (&str, Vec<u8>) = match path.as_str() {
                    "/s.css" => ("text/css", b"x { color: red; }".to_vec()),
                    "/i.gif" => {
                        // Real GIF (4x2 fixture), padded with zeros past
                        // the fetch window. The decoder stops at the GIF
                        // trailer and ignores the padding, so slicing to
                        // the window still decodes.
                        let mut gif =
                            include_bytes!("../../bowser_paint/tests/fixtures/gif_4x2.gif")
                                .to_vec();
                        gif.extend(std::iter::repeat_n(0u8, 100 * 1024));
                        ("image/gif", gif)
                    }
                    _ => (
                        "text/html",
                        b"<link rel=\"stylesheet\" href=\"/s.css\"><img src=\"/i.gif\">".to_vec(),
                    ),
                };
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = s.write_all(head.as_bytes());
                let _ = s.write_all(&body);
            }
        });

        let url = bowser_url::parse(&format!("http://{addr}/page"), None).expect("url");
        let jar = Mutex::new(CookieJar::new());
        let loaded = load_document_with_user_agent(&url, &jar, "TestBrowser/2.0").expect("load");
        server.join().expect("server thread");

        assert_eq!(loaded.stylesheets.len(), 1);
        assert_eq!(loaded.image_sizes.len(), 1);
        let seen = seen.lock().unwrap();
        // Document, stylesheet, image: same configured UA on all three.
        assert_eq!(seen.user_agents, vec!["TestBrowser/2.0"; 3]);
        // Only the image fetch asks for a leading byte range.
        assert!(seen.ranges[0].is_none());
        assert!(seen.ranges[1].is_none());
        assert_eq!(
            seen.ranges[2].as_deref(),
            Some("bytes=0-65535"),
            "image fetch must range-request the sniff window"
        );
    }
}
