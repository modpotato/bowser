//! Browser binary and windowed UI.
//!
//! Implementation lands via subsystem workers. See /AGENTS.md.

/// Test-only WPT harness, gated behind the `wpt-driver` feature (off by
/// default) so it never ships in the `bowser` binary.
#[cfg(feature = "wpt-driver")]
pub mod wpt_driver;

mod loader;

pub use loader::cached_image_for_src;

mod browser;

pub use browser::{Browser, BrowserConfig, Profile, TabArena, TabId};

mod navigation;

pub use navigation::{NavError, go_back, go_forward, navigate, reload};

mod scroll;

pub use scroll::{
    SCROLLBAR_MIN_THUMB, SCROLLBAR_THICKNESS, ScrollContainer, ScrollOutcome, ScrollTree,
    ScrollbarAxis, ScrollbarAxisGeom, ScrollbarDrag, ScrollbarGeom, ScrollbarHit,
};

mod scroll_touch;

mod script;

pub use script::{
    MAX_SCRIPTS, PageEventDispatch, PageScript, PageVm, ScriptError, ScriptErrorKind, ScriptRef,
    ScriptReport, SkipReason, collect_script_refs, resolve_scripts, script_node,
};

mod script_executor;

pub use script_executor::{
    HandledScript, ParseOutcome, ParserExecutor, filter_handled_scripts, prefetch_script_sources,
};

mod tab;

pub use tab::{InstalledDocument, Tab};

/// testdriver.js Action API (phase 1: bless/click/send_keys). Unconditional:
/// the headless pump drains automation requests, and the WPT fixture
/// server (feature `wpt-driver`) appends the vendor shim.
mod testdriver;

mod window;

pub use window::{
    HEADLESS_PUMP_BUDGET, ImageIdMap, JsDispatch, REFTEST_SETTLE_BUDGET, WELCOME_HTML, WindowError,
    font_key_for_style, load_tab_headless, run_windowed, run_windowed_with_js, settle_reftest_wait,
};

// __BOWSER_MODULES__
