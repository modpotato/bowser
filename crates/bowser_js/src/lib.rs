//! Browser-specific JavaScript runtime.
//!
//! The public embedder surface is intentionally minimal: [`Vm`] evaluates
//! scripts, [`VmError`] reports failures, [`Value`] is the JS value type,
//! [`NativeError`] is the error type returned by native (Rust) builtins,
//! and [`modules`] evaluates ES modules. The whole compiler pipeline
//! (lexer, parser, AST, opcodes, compiler), the GC, the event loop, the
//! builtins, and the bindings are crate-private. See /AGENTS.md
//! ("Public APIs are minimal and documented").

mod gc;
mod value;
pub use value::{NativeError, Property, Str, Value};
/// WTF-8 string storage: [`Wtf8`] is the borrowed view (the analogue of
/// `str`), [`Wtf8Buf`] the owned buffer (the analogue of `String`).
/// Public: [`Str::as_wtf8`](crate::Str::as_wtf8) exposes `&Wtf8` in the
/// embedder surface, so the types must be nameable from outside.
pub mod wtf8;
pub use wtf8::{Wtf8, Wtf8Buf};

/// Unicode ID_Start/ID_Continue tables used by the lexer.
mod id_tables;
mod lexer;

mod event_loop;

mod builtin_collections;
mod builtin_date;

mod builtin_reflect_proxy;
/// Test-harness surface: installs the test262 `$262` helper object
/// (`$262.detachArrayBuffer`) on a realm's global. Called by the
/// test262 runner only; never installed for real pages.
#[doc(hidden)]
pub use builtin_reflect_proxy::install_test262_helpers;

/// Wave-33 typed-array foundation: native `Uint8Array` construction API
/// for `TextEncoder` and friends. The JS-visible constructors stay owned
/// by [`builtin_reflect_proxy`]; this module owns the native layer.
mod builtin_typedarray;

mod builtin_json;

// __BOWSER_MODULES__
mod vm;
/// [`OwnDescriptor`] is the Phase-B test-harness reset surface: it lets the
/// test262 runner snapshot and compare own-property descriptors natively
/// instead of evaluating generated-JS `Object.getOwnPropertyDescriptor`
/// calls. [`Property`] and [`Str`] are re-exported for the same consumer
/// (native repair writes and key scans).
pub use vm::{OwnDescriptor, ScriptHandle, Vm, VmError};
mod ast;
mod builtin_error_function;
mod builtin_object_array;

mod parser;

mod opcode;

mod compiler;

mod bindings;
/// `TextEncoder`/`TextDecoder` per the WHATWG Encoding Standard, installed
/// on the page global next to the URL bindings.
mod bindings_encoding;
mod bindings_storage;
mod bindings_timers;
mod bindings_url;
/// Viewport scroll state shared with the embedder: [`ScrollState`] is the
/// tab's scroll offset (and its content-bound maximum), [`share_scroll_state`]
/// attaches it to a page's web APIs so `window.scrollTo` and friends
/// operate on the tab's offset, and [`take_scroll_into_view_request`]
/// drains pending `scrollIntoView()` targets for the embedder to resolve
/// against the layout tree. [`ElementScrollStates`] is the per-element
/// counterpart for nested scroll containers, attached with
/// [`share_element_scroll_states`] so `Element.scrollTop`/`scrollLeft`
/// read and write the tab's per-container offsets. [`share_layout_tree`]
/// publishes the latest layout tree so the CSSOM View geometry getters
/// (`offsetWidth`, `offsetParent`, ...) measure against real layout.
pub use bindings::{
    ActionItem, ActionItemKind, ActionOrigin, ActionSource, ActionSourceKind, ContextId,
    ElementScrollStates, LayoutEnsurer, ParseShimGuard, PendingSmoothScroll, PointerMouseKind,
    ScrollBehavior, ScrollIntoViewAlign, ScrollIntoViewRequest, ScrollState, TestDriverAction,
    TestDriverActionSequence, TestDriverRequest, clear_layout_tree, drain_page_microtasks,
    inject_font_load_completion, install_parse_document_shim, layout_ensurer, pump_animation_frame,
    pump_page_event_loop, pump_page_event_loop_with_budget, set_layout_ensurer,
    set_parse_shim_document_url, share_completed_image_loads, share_element_scroll_states,
    share_font_cache, share_layout_tree, share_scroll_state, take_pending_element_smooth_scroll,
    take_pending_font_load_invalidation, take_pending_smooth_scroll, take_scroll_into_view_request,
    take_testdriver_requests, web_apis_context_registered, web_apis_now_ms,
};
/// Out-of-VM animation-frame driver: the headless pump's entry point for
/// firing `requestAnimationFrame` callbacks from outside the JS engine.
pub use bindings_timers::pump_animation_frame_in_vm;
mod builtin_regexp;
mod builtin_string_number;
mod module_loader;
mod normalize;
/// ES module loading: [`ModuleLoader`] and [`ModuleError`].
///
/// The loader's internals (module map, linking, source rewriting) stay
/// crate-private; this namespace exposes exactly what an embedder needs:
/// construct a [`ModuleLoader`], optionally pre-register sources with
/// [`ModuleLoader::add_source`] or disable the network fallback with
/// [`ModuleLoader::set_allow_net`], then fetch, link, and evaluate with
/// [`ModuleLoader::load`] or [`ModuleLoader::load_source`]. All failures
/// surface as [`ModuleError`].
pub mod modules {
    pub use super::module_loader::{ModuleError, ModuleLoader};
}
mod regex_engine;
