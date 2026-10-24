//! Browser-level state: tabs, profile data, and shared configuration.
//!
//! [`Browser`] is the top-level owner of browsing state for one browser
//! instance: a [`Profile`] (data shared across every tab, notably the
//! cookie jar) plus the tab arena (see [`TabId`]). [`Tab`] itself is
//! defined in the sibling [`crate::tab`] module.

use std::sync::{Arc, Mutex};

use bowser_layout::Size;
use bowser_net::cookies::CookieJar;
use bowser_paint::fonts::FontCache;

use crate::navigation::{self, NavError};
use crate::tab::Tab;

/// Opaque handle to a tab.
///
/// A `TabId` is a slot index plus a generation. Closing a tab frees its
/// slot and bumps the generation, so a `TabId` captured before the close
/// can never resolve to a *different* tab later opened in the same slot:
/// stale ids simply fail to resolve (lookups return `None`, mutating
/// calls are no-ops).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TabId {
    /// Arena slot index.
    pub idx: u32,
    /// Generation of the slot when the id was issued.
    pub generation: u32,
}

/// One arena slot: a generation plus the occupying tab, if any.
struct Slot {
    generation: u32,
    tab: Option<Tab>,
}

/// Generational arena backing tab storage.
///
/// Slots are reused on close; the generation bump on free is what keeps
/// stale [`TabId`]s from aliasing new tabs.
///
/// (`Tab` has no `Debug` impl — it owns a `LayoutBox` — so neither does
/// the arena.)
#[derive(Default)]
pub struct TabArena {
    slots: Vec<Slot>,
}

impl TabArena {
    /// Insert a tab, returning its id. Reuses a freed slot when one is
    /// available.
    pub fn insert(&mut self, tab: Tab) -> TabId {
        if let Some((idx, slot)) = self
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, slot)| slot.tab.is_none())
        {
            let id = TabId {
                idx: idx as u32,
                generation: slot.generation,
            };
            slot.tab = Some(tab);
            return id;
        }
        let id = TabId {
            idx: self.slots.len() as u32,
            generation: 0,
        };
        self.slots.push(Slot {
            generation: 0,
            tab: Some(tab),
        });
        id
    }

    /// Remove the tab `id`, freeing its slot and bumping the generation.
    /// Unknown or stale ids yield `None`.
    pub fn remove(&mut self, id: TabId) -> Option<Tab> {
        let slot = self.slots.get_mut(id.idx as usize)?;
        if slot.generation != id.generation || slot.tab.is_none() {
            return None;
        }
        slot.generation = slot.generation.wrapping_add(1);
        slot.tab.take()
    }

    /// Look up a tab by id. Returns `None` for unknown or stale ids.
    pub fn get(&self, id: TabId) -> Option<&Tab> {
        let slot = self.slots.get(id.idx as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.tab.as_ref()
    }

    /// Mutably look up a tab by id. Returns `None` for unknown or stale
    /// ids.
    pub fn get_mut(&mut self, id: TabId) -> Option<&mut Tab> {
        let slot = self.slots.get_mut(id.idx as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.tab.as_mut()
    }

    /// Ids of all live tabs, in slot order.
    pub fn live_ids(&self) -> Vec<TabId> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.tab.is_some())
            .map(|(idx, slot)| TabId {
                idx: idx as u32,
                generation: slot.generation,
            })
            .collect()
    }
}

/// Configuration for a [`Browser`] instance.
#[derive(Debug, Clone)]
pub struct BrowserConfig {
    /// Viewport in CSS pixels, `(width, height)`.
    pub viewport: (u32, u32),
    /// Value sent as the `User-Agent` request header.
    pub user_agent: String,
    /// Headless mode: render offscreen, no window.
    pub headless: bool,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            viewport: (1280, 800),
            user_agent: format!("bowser/{}", env!("CARGO_PKG_VERSION")),
            headless: false,
        }
    }
}

/// Profile-wide state, shared across every tab of a [`Browser`].
///
/// (Later: history, permissions, storage. The cookie jar is first because
/// the network stack needs it.)
pub struct Profile {
    /// The profile's cookie jar, shared across all tabs and all loader
    /// threads.
    ///
    /// `Mutex`, not `RefCell`: loads run on worker threads, and
    /// [`crate::loader::load_document_with_user_agent`] requires `&Mutex<CookieJar>` so
    /// the jar can cross into those threads. The lock is held for the
    /// whole load in v1 (see the loader docs); keep critical sections
    /// short everywhere else.
    pub cookie_jar: Mutex<CookieJar>,
}

impl Profile {
    /// A fresh profile with an empty cookie jar.
    pub fn new() -> Self {
        Self {
            cookie_jar: Mutex::new(CookieJar::new()),
        }
    }
}

impl Default for Profile {
    fn default() -> Self {
        Self::new()
    }
}

/// Top-level browser state: one [`Profile`] plus the tab arena.
pub struct Browser {
    /// Profile-wide data shared across every tab.
    pub profile: Profile,
    /// Live tabs; see [`TabId`] for the stale-id contract.
    pub tabs: TabArena,
    config: BrowserConfig,
    active: Option<TabId>,
    /// Font cache for layout during navigation installs (the
    /// initial-load forced-sync-layout slot needs real metrics).
    fonts: Arc<Mutex<FontCache>>,
}

impl Browser {
    /// Create a browser with the given configuration, a fresh profile,
    /// and no tabs.
    pub fn new(config: BrowserConfig) -> Self {
        Self {
            profile: Profile::new(),
            tabs: TabArena::default(),
            config,
            active: None,
            fonts: Arc::new(Mutex::new(FontCache::new())),
        }
    }

    /// The configuration this browser was created with.
    pub fn config(&self) -> &BrowserConfig {
        &self.config
    }

    /// The profile's shared cookie jar.
    ///
    /// `&Mutex<_>` is what [`crate::loader::load_document_with_user_agent`] and
    /// [`crate::navigation`] require; this is the reference to hand them.
    /// [`Browser::navigate`] and friends already plumb it in.
    pub fn cookie_jar(&self) -> &Mutex<CookieJar> {
        &self.profile.cookie_jar
    }

    /// The browser's font cache handle, for the `@font-face` install
    /// step: fetched webfonts are installed into this cache before
    /// first layout (see `crate::loader::load_document_with_fonts`).
    pub fn font_cache(&self) -> &Mutex<FontCache> {
        &self.fonts
    }

    /// Open a new tab and make it the active tab.
    ///
    /// The URL is parsed and recorded as the tab's initial history entry;
    /// nothing is fetched. Call [`Browser::navigate`] to load it. A URL
    /// that fails to parse leaves the tab with empty history.
    pub fn new_tab(&mut self, url: &str) -> TabId {
        let (w, h) = self.config.viewport;
        let mut tab = Tab::new(Size {
            w: w as f32,
            h: h as f32,
        });
        if let Ok(parsed) = bowser_url::parse(url, None) {
            tab.history.push(parsed);
        }
        let id = self.tabs.insert(tab);
        self.active = Some(id);
        id
    }

    /// Close the tab `id`. Unknown or stale ids are a no-op.
    pub fn close_tab(&mut self, id: TabId) {
        if self.tabs.remove(id).is_some() && self.active == Some(id) {
            self.active = None;
        }
    }

    /// The currently active tab, if any.
    pub fn active_tab(&self) -> Option<TabId> {
        self.active
    }

    /// Make `id` the active tab. Unknown or stale ids are ignored.
    pub fn set_active_tab(&mut self, id: TabId) {
        if self.tabs.get(id).is_some() {
            self.active = Some(id);
        }
    }

    /// Look up a tab by id. Returns `None` for unknown or stale ids.
    pub fn tab(&self, id: TabId) -> Option<&Tab> {
        self.tabs.get(id)
    }

    /// Mutably look up a tab by id. Returns `None` for unknown or stale
    /// ids.
    pub fn tab_mut(&mut self, id: TabId) -> Option<&mut Tab> {
        self.tabs.get_mut(id)
    }

    /// Navigate the tab `id` to `url`, using the profile's shared cookie
    /// jar for the load. See [`crate::navigation::navigate`] for the
    /// semantics (fragment fast path, history, invalidation).
    ///
    /// Unknown or stale ids are a no-op returning `Ok`, matching the
    /// [`TabId`] contract.
    pub fn navigate(&mut self, id: TabId, url: &str) -> Result<(), NavError> {
        let Some(tab) = self.tabs.get_mut(id) else {
            return Ok(());
        };
        navigation::navigate(
            tab,
            &self.profile.cookie_jar,
            url,
            &self.config.user_agent,
            &self.fonts,
        )
    }

    /// Go back one session-history entry in the tab `id`, using the
    /// profile's shared cookie jar. Unknown or stale ids are a no-op
    /// returning `Ok`.
    pub fn go_back(&mut self, id: TabId) -> Result<(), NavError> {
        let Some(tab) = self.tabs.get_mut(id) else {
            return Ok(());
        };
        navigation::go_back(
            tab,
            &self.profile.cookie_jar,
            &self.config.user_agent,
            &self.fonts,
        )
    }

    /// Go forward one session-history entry in the tab `id`, using the
    /// profile's shared cookie jar. Unknown or stale ids are a no-op
    /// returning `Ok`.
    pub fn go_forward(&mut self, id: TabId) -> Result<(), NavError> {
        let Some(tab) = self.tabs.get_mut(id) else {
            return Ok(());
        };
        navigation::go_forward(
            tab,
            &self.profile.cookie_jar,
            &self.config.user_agent,
            &self.fonts,
        )
    }

    /// Reload the current entry of the tab `id`, using the profile's
    /// shared cookie jar. Unknown or stale ids are a no-op returning
    /// `Ok`.
    pub fn reload(&mut self, id: TabId) -> Result<(), NavError> {
        let Some(tab) = self.tabs.get_mut(id) else {
            return Ok(());
        };
        navigation::reload(
            tab,
            &self.profile.cookie_jar,
            &self.config.user_agent,
            &self.fonts,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Duration;

    fn browser() -> Browser {
        Browser::new(BrowserConfig::default())
    }

    #[test]
    fn open_tab_becomes_active_and_records_url() {
        let mut b = browser();
        assert_eq!(b.active_tab(), None);
        let id = b.new_tab("about:blank");
        assert_eq!(b.active_tab(), Some(id));
        assert_eq!(b.tabs.live_ids(), vec![id]);
        let tab = b.tab(id).unwrap();
        assert_eq!(
            tab.current_url().expect("history").serialize(),
            "about:blank"
        );
        // Viewport comes from the browser config.
        assert_eq!(
            tab.viewport,
            Size {
                w: 1280.0,
                h: 800.0
            }
        );
    }

    #[test]
    fn unparseable_url_leaves_empty_history() {
        let mut b = browser();
        let id = b.new_tab("://bad");
        assert!(b.tab(id).unwrap().current_url().is_none());
    }

    #[test]
    fn close_tab_frees_slot_and_clears_active() {
        let mut b = browser();
        let a = b.new_tab("https://a.example/");
        let c = b.new_tab("https://c.example/");
        assert_eq!(b.active_tab(), Some(c));
        b.close_tab(a);
        assert_eq!(b.tabs.live_ids(), vec![c]);
        assert!(b.tab(a).is_none());
        // Closing the active tab clears the active tab.
        b.close_tab(c);
        assert_eq!(b.active_tab(), None);
        assert!(b.tabs.live_ids().is_empty());
    }

    #[test]
    fn close_unknown_or_stale_id_is_noop() {
        let mut b = browser();
        let a = b.new_tab("https://a.example/");
        // Bogus index.
        b.close_tab(TabId {
            idx: 99,
            generation: 0,
        });
        assert_eq!(b.tabs.live_ids(), vec![a]);
        // Stale generation after close + reopen.
        b.close_tab(a);
        let d = b.new_tab("https://d.example/");
        b.close_tab(a);
        assert_eq!(b.tabs.live_ids(), vec![d]);
        assert!(b.tab(d).is_some());
    }

    #[test]
    fn generation_prevents_aliasing_on_slot_reuse() {
        let mut b = browser();
        let a = b.new_tab("https://a.example/");
        b.close_tab(a);
        // Slot 0 is reused, but the generation must differ.
        let e = b.new_tab("https://e.example/");
        assert_eq!(a.idx, e.idx);
        assert_ne!(a.generation, e.generation);
        // The stale id resolves to nothing and mutates nothing.
        assert!(b.tab(a).is_none());
        assert!(b.tab_mut(a).is_none());
        assert_eq!(
            b.tab(e).unwrap().current_url().unwrap().serialize(),
            "https://e.example/"
        );
    }

    #[test]
    fn set_active_tab_ignores_stale_ids() {
        let mut b = browser();
        let a = b.new_tab("https://a.example/");
        let c = b.new_tab("https://c.example/");
        b.set_active_tab(a);
        assert_eq!(b.active_tab(), Some(a));
        b.close_tab(a);
        let e = b.new_tab("https://e.example/");
        b.set_active_tab(a); // stale: ignored, active stays `e`
        assert_eq!(b.active_tab(), Some(e));
        b.set_active_tab(c); // live: honored
        assert_eq!(b.active_tab(), Some(c));
    }

    #[test]
    fn navigate_on_stale_id_is_noop_ok() {
        let mut b = browser();
        let a = b.new_tab("https://a.example/");
        b.close_tab(a);
        // Must not panic; returns Ok per the TabId contract.
        assert!(b.navigate(a, "https://x.example/").is_ok());
        assert!(b.go_back(a).is_ok());
        assert!(b.go_forward(a).is_ok());
        assert!(b.reload(a).is_ok());
    }

    #[test]
    fn cookie_jar_is_shared_across_tabs() {
        use bowser_net::cookies::RequestContext;
        let mut b = browser();
        let _t1 = b.new_tab("https://a.example/");
        let _t2 = b.new_tab("https://b.example/");
        // Both tabs (and the profile itself) see the same jar.
        assert!(b.cookie_jar().lock().unwrap().is_empty());
        {
            let ctx = RequestContext {
                host: "a.example",
                path: "/",
                secure: false,
                http_api: true,
                top_host: Some("a.example"),
                top_secure: Some(false),
                top_level_safe_navigation: true,
            };
            b.profile
                .cookie_jar
                .lock()
                .unwrap()
                .set_cookie("session=abc; Path=/", &ctx, 0);
        }
        // Visible through the accessor regardless of which tab is active.
        assert_eq!(b.cookie_jar().lock().unwrap().len(), 1);
    }

    // ------------------------------------------------------------------
    // Profile-level cookie jar plumbed into navigations: a local server
    // sets a cookie on /set and echoes the Cookie header on /echo. A
    // second tab must send the cookie the first tab's navigation stored.
    // ------------------------------------------------------------------

    /// Serve one connection: /set replies Set-Cookie, /echo replies the
    /// received Cookie header as the body. Returns false to stop.
    fn handle_one(stream: &mut std::net::TcpStream) -> bool {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            match stream.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    buf.extend_from_slice(&chunk[..n]);
                    if buf.windows(4).any(|w| w == b"\r\n\r\n") || buf.len() > 65_536 {
                        break;
                    }
                }
                Err(_) => return true,
            }
        }
        let head = String::from_utf8_lossy(&buf);
        // The net stack sends absolute-form request targets
        // (`GET http://host:port/path ...`), so strip `scheme://authority`
        // (and any query) before matching routes.
        let target = head
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("/");
        let after_authority = match target.split_once("://") {
            Some((_, rest)) => &rest[rest.find('/').unwrap_or(rest.len())..],
            None => target,
        };
        let path = after_authority.split('?').next().unwrap_or(after_authority);
        if path == "/__shutdown" {
            return false;
        }
        let (extra_headers, body) = match path {
            "/set" => (
                "Set-Cookie: session=abc; Path=/\r\n".to_string(),
                "<!DOCTYPE html><html><head><title>set</title></head>\
                 <body></body></html>"
                    .as_bytes()
                    .to_vec(),
            ),
            "/echo" => {
                let cookie = head
                    .lines()
                    .find_map(|line| line.strip_prefix("Cookie:"))
                    .map(str::trim)
                    .unwrap_or("(none)");
                (
                    String::new(),
                    format!(
                        "<!DOCTYPE html><html><head><title>{cookie}</title></head>\
                         <body></body></html>"
                    )
                    .into_bytes(),
                )
            }
            _ => (
                String::new(),
                "<!DOCTYPE html><html><head><title>404</title></head>\
                 <body></body></html>"
                    .as_bytes()
                    .to_vec(),
            ),
        };
        let response = format!(
            "HTTP/1.1 200 OK\r\n\
             Content-Type: text/html; charset=utf-8\r\n\
             Content-Length: {}\r\n\
             {extra_headers}Connection: close\r\n\r\n",
            body.len()
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.write_all(&body);
        true
    }

    struct CookieServer {
        base: String,
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl CookieServer {
        fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
            let addr = listener.local_addr().expect("test server addr");
            let stop = Arc::new(AtomicBool::new(false));
            let stop_flag = Arc::clone(&stop);
            let handle = thread::spawn(move || {
                let _ = listener.set_nonblocking(false);
                for stream in listener.incoming() {
                    if stop_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    match stream {
                        Ok(mut s) => {
                            let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
                            if !handle_one(&mut s) {
                                break;
                            }
                        }
                        Err(_) => continue,
                    }
                }
            });
            CookieServer {
                base: format!("http://{addr}"),
                stop,
                handle: Some(handle),
            }
        }
    }

    impl Drop for CookieServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            let _ = std::net::TcpStream::connect(self.base.trim_start_matches("http://"));
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    #[test]
    fn profile_cookie_jar_flows_through_navigations() {
        let server = CookieServer::start();
        let mut b = browser();

        // Tab 1 navigates to /set: the loader stores the Set-Cookie in
        // the profile jar via &Mutex<CookieJar>. (New tabs start at
        // about:blank so the navigation is a full load, not the
        // same-document fast path.)
        let t1 = b.new_tab("about:blank");
        b.navigate(t1, &format!("{}/set", server.base))
            .expect("navigate /set");
        assert_eq!(b.tab(t1).unwrap().title, "set");
        assert_eq!(b.cookie_jar().lock().unwrap().len(), 1);

        // Tab 2 navigates to /echo on the same profile: the stored
        // cookie must be sent, proving the jar is shared across tabs
        // and across loader threads.
        let t2 = b.new_tab("about:blank");
        b.navigate(t2, &format!("{}/echo", server.base))
            .expect("navigate /echo");
        assert_eq!(b.tab(t2).unwrap().title, "session=abc");

        // Reloading through the Browser API also uses the profile jar.
        b.reload(t1).expect("reload");
        assert_eq!(b.cookie_jar().lock().unwrap().len(), 1);
    }
}
