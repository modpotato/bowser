//! AST to bytecode compiler for the Bowser JavaScript VM.
//!
//! [`compile`] turns a [`Program`](crate::ast::Program) into a
//! [`FuncTemplate`] (the top level is an implicit function); see
//! [`compile_module`] when the import/export side table is needed too.
//!
//! # Compiler/VM contract (beyond [`crate::opcode`])
//!
//! * **Prologue.** Every function starts with compiler-emitted prologue code
//!   that initializes its declared bindings: `var`s to `undefined`, `let`/`const`
//!   to `undefined` with their TDZ flag cleared, function declarations to
//!   their closures, and `#param#i` temps to `undefined` (overwritten by the
//!   argument-binding dance that follows). The VM must additionally
//!   initialize three hidden bindings on entry: `#this` (the call's
//!   `this` value), `#newtarget` (`new.target`, `undefined` for plain
//!   calls), and `#argc` (the number of arguments actually passed).
//! * **Hidden names.** Every compiler-generated binding or intrinsic name
//!   contains a `#`, which can never appear in a user-written identifier, so
//!   collisions are impossible. TDZ state for a lexical binding `x` is a
//!   dedicated boolean binding named `#tdz#x` in the same scope (`false` /
//!   falsy = uninitialized, `true` = initialized); reads emit an explicit
//!   check that throws `ReferenceError` when the flag is falsy.
//! * **Upvalues.** [`UpvalueDesc::Local`] carries a compiler-assigned binding
//!   id (the index into the function's binding table, unique across all of
//!   the function's block scopes). The VM resolves it by walking the current
//!   frame's environment chain for that id and boxes the binding *by cell*.
//! * **Intrinsics.** The VM provides these hidden globals (all take
//!   `this = undefined`):
//!   `#enum_keys(obj)` (own enumerable string keys, for `for-in`),
//!   `#define_accessor(obj, key, getter, setter)`,
//!   `#set_proto(obj, proto)`, `#obj_spread(target, src)`,
//!   `#obj_rest(obj, excluded_keys)`, `#arr_rest(array, start)`,
//!   `#arr_append(arr, value)` (returns the array),
//!   `#construct(fn, args_array)`,
//!   `#define_class(ctor, heritage, proto_desc, static_desc)` (returns
//!   `[ctor, brand]` and marks the constructor),
//!   `#super_construct(super_ctor, new_target, args_array)` (initializes the
//!   frame's `this` and returns the instance),
//!   `#priv_get(brand, obj, key)` / `#priv_set(brand, obj, key, value)`
//!   (private state lives in a VM-side table keyed by `(brand, instance)` —
//!   never as object properties), `#bigint(raw)`, `#import_meta()`.
//!
//!   (Spread calls, regex literals, and dynamic `import()` compile to the
//!   dedicated [`Op::SpreadCall`], [`Op::NewRegex`], [`Op::DynamicImport`],
//!   and [`Op::RestArgs`] opcodes instead of hidden intrinsics, so they
//!   work while a `vm_call` closure is on the stack.)
//! * **v1 limits** (documented divergences): `with` is rejected
//!   ([`CompileError::Unsupported`]); per-iteration `let`/`const` bindings in
//!   `for` loops are shared across iterations (closures see the last value);
//!   `for-in` enumerates string keys only; spreading a non-array/string
//!   throws; `yield*` is emitted as [`Op::YieldStar`] (the VM drives the
//!   inner iterator's protocol); `await` is emitted as [`Op::Await`] (no
//!   async desugar — the VM implements the state machine or traps);
//!   derived constructors without a `super()` call leave `this`
//!   uninitialized; private getters/setters are stored as plain private
//!   fields; `arguments` is provided but not sloppy-mapped (parameter
//!   assignments do not update `arguments` entries and vice versa).

use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::*;
use crate::gc::WellKnown;
use crate::lexer::Span;
use crate::opcode::{Op, decode, encode, tag};
use crate::wtf8::{Wtf8, Wtf8Buf};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A constant-pool entry.
// `FuncTemplate` dominates the size: every template carries its code,
// consts, and binding tables inline, and the `[[ScriptOrModule]]` URL
// keeps module identity on the template itself (see
// [`FuncTemplate::script_or_module`]). Boxing the variant would churn
// every `Constant::FuncTemplate` site for no hot-path gain — const
// pools are built once per compile — so the large variant is
// intentional.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum Constant {
    /// String constant (WTF-8: may hold lone surrogates from `\u` escapes).
    Str(Wtf8Buf),
    Float(f64),
    Int(i32),
    FuncTemplate(FuncTemplate),
}

/// Tag prefix marking a string constant as a compiler-emitted throw
/// message. The VM's `interpret_thrown` converts a thrown string with this
/// prefix into a real error object (`TAG + kind + NUL + message`). Shared
/// here so the compiler can pre-tag unconditional throw sites directly;
/// the VM-side tagger only recognizes the TDZ/const guard idiom.
pub(crate) const COMPILER_THROW_TAG: &str = "\u{0}bowser:compiler-throw\u{0}";

/// A compiled function: code plus its own constant pool.
#[derive(Clone, Debug)]
pub struct FuncTemplate {
    pub name: String,
    pub arity: u16,
    pub code: Vec<u8>,
    pub consts: Vec<Constant>,
    pub upvalues: Vec<UpvalueDesc>,
    /// Source name of each upvalue, parallel to [`FuncTemplate::upvalues`]:
    /// the binding the desc captures, as named in the declaring scope.
    /// `#tdz#` flag upvalues keep their flag name. The VM uses this to
    /// resolve direct-`eval` names against a caller frame's captured
    /// ancestor scopes (see `capture_eval_scope`).
    pub upvalue_names: Vec<String>,
    /// Binding-table slot names in order: `binding_names[id]` is the source
    /// name bound at environment slot `id`. The VM uses this to resolve
    /// `UpvalueDesc::Local(id)` captures. Empty for hand-built templates.
    pub binding_names: Vec<String>,
    /// Source names of this template's `var`-scoped declarations (top-level
    /// `var` bindings plus function declarations when sloppy, in declaration
    /// order, deduplicated). The VM's `eval` implementation uses this to
    /// share sloppy direct-eval declarations with the caller's environment
    /// and to install indirect-eval declarations as globals.
    pub var_names: Vec<String>,
    /// Compiler [`BindingKind`] of each binding, parallel to
    /// [`FuncTemplate::binding_names`]. The VM uses this to reject sloppy
    /// direct-`eval` `var` declarations that collide with a caller's
    /// lexical (`let`/`const`) binding (an early `SyntaxError`).
    pub(crate) binding_kinds: Vec<BindingKind>,
    pub is_async: bool,
    pub is_generator: bool,
    pub strict: bool,
    /// Method shorthand / class method / getter / setter: has a `[[HomeObject]]`
    /// and is not a constructor (`new obj.method()` throws `TypeError`).
    pub is_method: bool,
    /// A derived class constructor (`class D extends B { constructor() ... }`,
    /// explicit or implicit). The VM starts its `#this` binding uninitialized
    /// (TDZ until `super()` runs) and skips the fresh-instance pre-creation.
    pub is_derived_ctor: bool,
    /// Sloppy mapped-`arguments` support: for a sloppy non-arrow function
    /// with a simple parameter list (every parameter a plain identifier),
    /// the environment slot of each parameter, in order (`None` = that
    /// index is unmapped: an earlier duplicate of a later parameter name).
    /// The VM aliases `arguments[i]` to these slots (spec: the mapped
    /// arguments object, `CreateMappedArgumentsObject`). `None` for strict
    /// functions, arrows, and non-simple parameter lists (unmapped
    /// arguments object).
    pub mapped_arg_slots: Option<Vec<Option<u16>>>,
    /// The function's `[[ScriptOrModule]]` for dynamic-`import()`
    /// referrer resolution (ES2024 §8.3.1 `GetActiveScriptOrModule`):
    /// `Some(url)` when this template was compiled as part of the module
    /// with that URL, `None` for classic scripts (and for indirect
    /// `eval`, which runs as a classic script). The compiler always
    /// leaves it `None`; the module loader stamps the whole compilation
    /// unit via [`FuncTemplate::set_script_or_module`], and direct
    /// `eval` inherits its caller's value.
    pub script_or_module: Option<String>,
}

impl FuncTemplate {
    /// Stamp `url` as the `[[ScriptOrModule]]` of this template and every
    /// nested template (nested functions live in
    /// [`Constant::FuncTemplate`] consts). `None` leaves the classic-
    /// script `None` in place.
    pub fn set_script_or_module(&mut self, url: Option<&str>) {
        self.script_or_module = url.map(str::to_owned);
        for c in &mut self.consts {
            if let Constant::FuncTemplate(t) = c {
                t.set_script_or_module(url);
            }
        }
    }
}

/// How a closure captures one outer binding.
#[derive(Clone, Copy, Debug)]
pub enum UpvalueDesc {
    /// Capture binding id `0` of the directly enclosing function.
    Local(u16),
    /// Capture upvalue `0` of the directly enclosing function (transitive).
    Upvalue(u16),
}

/// How the current function loads a TDZ flag binding.
#[derive(Clone, Copy, Debug)]
enum TdzAccess {
    /// Flag is a local binding; the value is its name constant.
    Local(u16),
    /// Flag is captured; the value is the upvalue index.
    Upvalue(u16),
}

/// Compilation failure.
#[derive(Debug)]
pub enum CompileError {
    TooManyConstants,
    TooManyLocals,
    JumpTooFar,
    InvalidLhs,
    /// Expression/statement/pattern nesting too deep to compile without
    /// risking a host-stack overflow. Mirrors the parser's depth guard:
    /// the parser builds some shapes (left-deep binary chains, member/call
    /// suffixes) iteratively, so they can reach the compiler arbitrarily
    /// deep even though the parser accepted them.
    DepthExceeded,
    /// The construct exists in the AST but v1 codegen does not support it
    /// (e.g. `with`). The `&'static str` names the construct.
    Unsupported(&'static str),
    /// A regular expression literal failed to compile (spec early error:
    /// invalid pattern or flags). The `String` carries the regex engine's
    /// diagnostic.
    InvalidRegex(String),
    /// An internal compiler invariant was violated (e.g. a declare/codegen
    /// scope mismatch). The `&'static str` names the invariant. This is a
    /// defense-in-depth fallback: code should never reach here, but a scope
    /// bookkeeping bug must surface as a recoverable error, not a panic.
    Internal(&'static str),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::TooManyConstants => write!(f, "too many constants"),
            CompileError::TooManyLocals => write!(f, "too many locals"),
            CompileError::JumpTooFar => write!(f, "jump out of range"),
            CompileError::InvalidLhs => write!(f, "invalid left-hand side"),
            CompileError::DepthExceeded => write!(f, "maximum compile depth exceeded"),
            CompileError::Unsupported(what) => write!(f, "unsupported: {what}"),
            CompileError::InvalidRegex(what) => write!(f, "invalid regular expression: {what}"),
            CompileError::Internal(what) => write!(f, "internal compiler error: {what}"),
        }
    }
}

impl std::error::Error for CompileError {}

// ---------------------------------------------------------------------------
// Module records
// ---------------------------------------------------------------------------

/// The import/export side table for a module. `import`/`export` statements
/// compile to [`Op::Nop`] (plus normal code for exported declarations); the
/// module linker consumes this record when the module is instantiated.
#[derive(Clone, Debug, Default)]
pub struct ModuleRecord {
    pub imports: Vec<ImportDecl>,
    pub exports: Vec<ExportEntry>,
}

/// One export entry.
#[derive(Clone, Debug)]
pub enum ExportEntry {
    /// `export { local as exported }` / `export { local as exported } from
    /// "source"`. `local` is `None` for re-exports from another module.
    Named {
        local: Option<String>,
        exported: String,
        source: Option<String>,
    },
    /// `export * from "source"` / `export * as ns from "source"`.
    All {
        source: String,
        exported: Option<String>,
    },
    /// `export default <expr>` — the value lives in the `#default_export`
    /// binding after the module top-level function runs.
    Default,
    /// `export <decl>` — names bound by the declaration statement.
    Decl(Vec<String>),
}

// ---------------------------------------------------------------------------
// Compiler state
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BindingKind {
    Var,
    Let,
    Const,
    Function,
    Hidden,
    /// Named function expression's own name (ECMA-262 `NamedEvaluation`):
    /// an immutable binding in the function's own scope, initialized by
    /// the VM with the callee on frame entry. Later declarations of the
    /// same source name (params, `var`s, function declarations) shadow it
    /// with a fresh binding instead of merging.
    SelfName,
    /// Initialized by the VM on function entry (`#this`, `#newtarget`, `#argc`).
    VmProvided,
}

impl BindingKind {
    pub(crate) fn is_lexical(self) -> bool {
        matches!(self, BindingKind::Let | BindingKind::Const)
    }
}

#[derive(Clone)]
struct Binding {
    name_const: u16,
    kind: BindingKind,
    /// For [`BindingKind::Function`]: the template's const index.
    func_const: Option<u16>,
}

struct Scope {
    /// `Rc` so codegen can replay a declare-walk scope without cloning the
    /// map: the declare walk owns the only strong reference while it
    /// mutates it (via `Rc::make_mut`), and codegen only ever reads the
    /// shared maps.
    names: Rc<HashMap<String, u16>>,
    /// True for function scopes; `var` declarations land here.
    function: bool,
    /// Index into [`Compiler::fns`] of the function that owns this scope.
    fn_idx: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LoopKind {
    Loop,
    /// A `for-of` loop: carries iterator-close obligations. `break` out of
    /// one (and `return`/`throw` through one) must run `IteratorClose` on
    /// the in-flight iterator (spec 14.7.5); `continue` must not.
    ForOf,
    Switch,
    Block,
}

/// One active `with (obj)` statement being compiled: the hidden temp's
/// name constant plus the scope-stack depth at entry. Name
/// loads/stores inside the body probe the with-objects (innermost first)
/// via `in` before falling back to the ordinary compile-time-resolved
/// access — but only for withs entered *inside* the name's declaring
/// scope, so an inner lexical binding still shadows the with-object per
/// spec. The stack is cleared when a nested function body starts
/// compiling (a closure keeps ordinary lexical resolution; it does not
/// observe the with-object — a documented v1 deviation), except for the
/// compiler-generated `#for_body` extraction arrow, which captures each
/// with-object as an upvalue and keeps probing.
struct WithEntry {
    /// Source name of the hidden temp holding the with-object (e.g.
    /// `#t3`). Needed to re-capture the temp as an upvalue inside the
    /// synthetic `#for_body` arrow, where the enclosing function's name
    /// constant cannot be reused verbatim.
    name: String,
    /// Name constant of the hidden temp holding the with-object (for
    /// `Op::LoadName`; the peephole rewrites it to `Op::LoadLocal`).
    name_const: u16,
    /// `self.scopes.len()` when the `with` was entered.
    scope_depth: usize,
}

struct LoopFrame {
    kind: LoopKind,
    labels: Vec<String>,
    break_patches: Vec<usize>,
    continue_patches: Vec<usize>,
    protected_depth: usize,
    /// Index into [`FnState::protected`] of this loop's
    /// [`Protected::ForOf`] region (`Some` only when
    /// `kind == LoopKind::ForOf`). Needed so `continue` targeting the loop
    /// itself can skip exactly its own region: the loop is not exited, so
    /// its iterator must stay open.
    for_of_region: Option<usize>,
}

/// A `try`/`finally` region the current code position is inside.
#[derive(Clone)]
enum Protected {
    Try,
    /// A `for-of` loop's iterator-close region (spec 14.7.5): an abrupt
    /// exit unwinding through it (`break`, `return`, or a compiler-visible
    /// unwind) emits [`Op::IterClose`]/[`Op::IterCloseRet`] to run
    /// `IteratorClose` on the loop's in-flight iterator. Purely a
    /// compile-time marker — no AST, no scopes — so unlike
    /// [`Protected::Finally`] it needs no declare-walk slot info.
    /// (`throw` unwinding is handled at runtime by the [`Op::IterGuard`]
    /// guard on the frame's handler stack, which the loop also emits.)
    ForOf,
    /// A `finally` region whose statements are inlined at each abrupt exit.
    /// `inlining` is true while this region's body is being emitted, so a
    /// reentrant [`Compiler::unwind_protected`] skips just this region
    /// instead of re-emitting it (which would recurse forever).
    Finally {
        /// Pointer to the finalizer's statement slice in the AST, plus its
        /// length. Stored as a raw pointer rather than a cloned `Vec` so
        /// that a `try` nested inside the finalizer compiles against the
        /// original AST nodes, whose pointers the declare walk recorded in
        /// [`FnState::finally_bases`] (a clone would hand nested `try`s
        /// unrecorded pointers).
        ///
        /// SAFETY: the AST outlives the entire compilation and is never
        /// mutated while the compiler runs, so the pointer stays valid for
        /// every inline. Reconstructed with
        /// [`std::slice::from_raw_parts`] in
        /// [`Compiler::inline_finally`].
        stmts_ptr: *const Stmt,
        stmts_len: usize,
        inlining: bool,
        /// [`FnState::saved`] slot of the try-statement (outer) scope, which
        /// owns the finalizer's lexicals.
        try_base: usize,
        /// [`FnState::saved`] slot where the finalizer's own block scopes
        /// begin (recorded by the declare walk in
        /// [`FnState::finally_bases`]).
        finally_base: usize,
        /// `self.scopes.len()` just after the try-statement (outer) scope
        /// was pushed: [`Compiler::inline_finally`] truncates the scope
        /// stack to this depth so the finalizer resolves against the outer
        /// scope rather than the try block's live scopes.
        outer_depth: usize,
    },
}

#[derive(Clone)]
struct ClassCtx {
    super_name: String,
    /// This class definition's compile-time private brand: every private
    /// member access lexically inside the class carries it. The compiler
    /// starts at a VM-globally-unique base per compilation unit, so brands
    /// never collide across separately compiled scripts, evals, or
    /// modules; each evaluation of the class definition shares its site's
    /// brand. Instances (plus the constructor, for static private members)
    /// record it in the VM-side brand table.
    brand: u16,
    /// True while compiling static members: `super.x` resolves against
    /// the base constructor instead of the base prototype.
    is_static: bool,
}

#[derive(Clone)]
enum FieldKey {
    Name(Wtf8Buf),
    /// A computed key already evaluated exactly once at class definition
    /// time and parked in the named hidden binding (see
    /// [`Compiler::precompute_field_key`]); install sites load the stored
    /// key instead of re-evaluating the expression.
    Precomputed(String),
    Private(String),
}

/// One static initializer in textual order: a static field or a static
/// block. Both compile to strict wrapper closures (see
/// `compile_static_field_wrapper` / `compile_static_block_wrapper`).
enum StaticInit<'a> {
    Field(FieldKey, &'a Option<Expr>),
    Block(&'a Vec<Stmt>),
}

struct CtorCtx {
    fields: Vec<(FieldKey, Option<Expr>)>,
    /// The class's private brand, installed on the fresh instance before
    /// field initialization (see [`Op::AddPrivateBrand`]).
    brand: u16,
    /// Whether the class declares any private member.
    has_private: bool,
}

struct FnState {
    name: String,
    arity: u16,
    is_arrow: bool,
    is_async: bool,
    is_generator: bool,
    strict: bool,
    is_method: bool,
    /// Set from [`CtorInfo::is_derived`] by [`Compiler::compile_function_body`]
    /// (or directly by [`Compiler::compile_implicit_ctor`)).
    is_derived_ctor: bool,
    /// Parameter binding ids for sloppy mapped `arguments`, recorded by
    /// [`Compiler::compile_function_body`] (see
    /// [`FuncTemplate::mapped_arg_slots`]). `None` unless the function is
    /// sloppy, non-arrow, and has a simple parameter list.
    mapped_arg_slots: Option<Vec<Option<u16>>>,
    code: Vec<u8>,
    consts: Vec<Constant>,
    str_map: HashMap<Wtf8Buf, u16>,
    float_map: HashMap<u64, u16>,
    int_map: HashMap<i32, u16>,
    bindings: Vec<Binding>,
    /// Source names of this function's `var`-scoped declarations
    /// ([`BindingKind::Var`] and [`BindingKind::Function`]), in declaration
    /// order, deduplicated. Copied to [`FuncTemplate::var_names`] for the
    /// VM's `eval` implementation.
    var_names: Vec<String>,
    upvalues: Vec<UpvalueDesc>,
    upvalue_names: HashMap<String, u16>,
    loops: Vec<LoopFrame>,
    protected: Vec<Protected>,
    class_stack: Vec<ClassCtx>,
    ctor_ctx: Option<CtorCtx>,
    scope_base: usize,
    /// `with` statements active in the enclosing function when this
    /// function body started compiling (see [`WithEntry`]): stashed here
    /// by [`Compiler::push_function`] and restored by
    /// [`Compiler::pop_function`], so nested bodies keep lexical
    /// resolution while the enclosing body keeps probing.
    outer_with: Vec<WithEntry>,
    /// Block scopes saved by the declare walk, in pre-order of their
    /// pushes. Codegen replays them in the same order via
    /// [`Compiler::push_block_scope`], so each block's lexical bindings
    /// resolve to the ids declared up front. Shared by `Rc`: replaying a
    /// scope is O(1).
    saved: Vec<Rc<HashMap<String, u16>>>,
    /// Next index into [`FnState::saved`] for codegen to consume.
    scope_cursor: usize,
    /// Maps each `finally` block's statement-slice pointer to the
    /// [`FnState::saved`] slot index where the declare walk began recording
    /// that finalizer's block scopes. The declare walk fills this before
    /// codegen runs; [`Compiler::inline_finally`] uses it to replay a
    /// finalizer's scopes when an abrupt exit (`return`/`throw`/`break`/
    /// `continue`) inlines the finalizer mid-try-block, at which point the
    /// live [`FnState::scope_cursor`] has not yet reached the finalizer's
    /// slots. Keyed by pointer because the AST carries no statement ids;
    /// the AST outlives compilation, so the keys stay valid. Re-inlining
    /// the same finalizer (normal path, abrupt paths) looks up the same
    /// key, making the lookup idempotent.
    finally_bases: HashMap<*const Stmt, usize>,
    /// True while compiling a per-iteration extracted loop body: `break`,
    /// `continue`, and `return` targeting outside the body are compiled to
    /// tagged array returns (`[tag, ...]`) instead of jumps, so the
    /// per-iteration wrapper can dispatch them.
    extracted: bool,
    /// Loop-frame index of the extracted loop in this function's `loops`
    /// stack (meaningful only when `extracted` is set).
    extracted_loop: usize,
    /// Head `let`/`const` names of the extracted loop: the arrow's
    /// parameters, whose values are snapshotted into tagged returns.
    extracted_params: Vec<String>,
    /// Labels targeted by `break`/`continue` inside an extracted body that
    /// did not resolve locally; the wrapper validates and dispatches them.
    unresolved_labels: Vec<String>,
    /// Function declarations hoisted out of a per-iteration extracted loop
    /// body: `(name, template index)` pairs. The template is compiled
    /// during this arrow's declare walk (see `declare_stmt_inner`) and
    /// interned in this arrow's constants;
    /// [`Compiler::compile_extracted_body`] instantiates a fresh closure
    /// per iteration so each captures its own iteration environment.
    extracted_fn_inits: Vec<(String, u16)>,
    /// When true, `declare_stmt` skips `var` declarations and function
    /// declarations (they were declared in the enclosing function by
    /// `declare_body_vars`); used for per-iteration extracted loop bodies.
    /// Function declarations still get their templates compiled during the
    /// arrow's declare walk (see `declare_stmt_inner`); their names are
    /// recorded in [`FnState::extracted_fn_inits`] for per-iteration
    /// instantiation by [`Compiler::compile_extracted_body`].
    no_vars: bool,
    /// Free-list of reusable hidden-temp indices: [`Compiler::hidden_temp`]
    /// pops an index here before allocating a fresh `#t{n}` binding, and
    /// [`Compiler::release_temp`] returns indices whose temps are dead.
    /// Reuse keeps the binding table (and the prologue) small.
    temp_free: Vec<u32>,
    /// TDZ flag access cache, keyed by `(decl_fn << 32) | binding_id`, so a
    /// lexical binding's flag is resolved (and upvalue-captured) once per
    /// function instead of once per access.
    tdz_flag_cache: HashMap<u64, TdzAccess>,
    /// Shared TDZ-violation throw stubs: `(binding_key, const_check)`
    /// maps to the code offset of a `LoadConst(msg); Throw` sequence
    /// recorded at its first use site. Later checks for the same binding
    /// jump back to it instead of re-emitting the sequence, which shrinks
    /// bytecode and compile time.
    tdz_stubs: HashMap<(u64, bool), usize>,
    /// True for the implicit strict wrapper functions compiled around
    /// static field initializers and static blocks: no `arguments` object
    /// is created, and `return` inside is a compile error.
    is_class_init: bool,
}

enum Resolved {
    Local(u16, BindingKind),
    Upvalue(u16, BindingKind),
    Global(u16),
}

pub struct Compiler {
    fns: Vec<FnState>,
    scopes: Vec<Scope>,
    module: ModuleRecord,
    temp_counter: u32,
    class_counter: u32,
    /// Sequence for per-class private brands (see [`Compiler::next_brand`]).
    /// Starts at the VM-supplied `brand_base` so brands are unique across
    /// every compilation unit evaluated by the VM (the VM keeps a global
    /// brand sequence and hands each compilation a fresh base); the `u16`
    /// operand encoding bounds the usable range.
    brand_counter: u32,
    /// The brand this compilation started at (for [`Compiler::brands_used`]).
    brand_base: u32,
    /// Sequence for unique internal binding names (block-scoped lexicals).
    bind_seq: u32,
    /// Stack of [`FnState::saved`] slots currently open in the declare walk.
    open_declare: Vec<usize>,
    pending_labels: Vec<String>,
    /// Current recursion depth of the compile walkers (host-stack guard).
    depth: u32,
    /// Native stack address captured at [`Compiler::new`]; [`Compiler::guarded`]
    /// measures consumption against it (see [`MAX_COMPILE_NATIVE_STACK`]).
    stack_base: usize,
    /// Active `with (obj)` statements being compiled, innermost last (see
    /// [`WithEntry`]). Cleared on nested-function entry in
    /// [`Compiler::push_function`] and restored in [`Compiler::pop_function`].
    with_stack: Vec<WithEntry>,
}

/// Maximum compiler recursion depth (host-stack guard). The parser already
/// rejects deeper nesting for the constructs it parses recursively, but
/// left-deep binary chains and member/call suffixes are built iteratively
/// by the parser and can be arbitrarily deep, so the compiler enforces its
/// own bound and reports [`CompileError::DepthExceeded`] instead of
/// overflowing the host stack (an uncatchable process abort).
const MAX_COMPILE_DEPTH: u32 = 512;

/// Native stack bytes the compiler may consume from the base captured in
/// [`Compiler::new`] before [`Compiler::guarded`] reports
/// [`CompileError::DepthExceeded`]. The depth counter alone cannot bound
/// native use: one walker frame is several KiB in unoptimized builds, so
/// 512 guarded levels overflow a 2 MiB thread stack long before the
/// counter trips (observed: a 200-deep `.bind(null)…` chain aborts the
/// process on a 2 MiB thread, where the counter would allow ~256 links).
/// The probe keeps small-stack environments failing clean instead of
/// aborting; the depth counter remains as a backstop.
///
/// The 512 KiB budget assumes [`Compiler::new`] runs on a fresh stack: the
/// guard measures *distance* from the base address, so constructing the
/// compiler deep inside an existing stack (e.g. a VM eval entry point
/// called from a deeply nested host callback) eats into the same budget
/// and can overflow before the probe trips. Keep construction on the same
/// thread as compilation (the probe address is thread-local stack memory)
/// and as close to the stack top as the call path allows; do not move a
/// live `Compiler` across threads or resume it after deep intervening
/// frames without a fresh instance.
const MAX_COMPILE_NATIVE_STACK: usize = 512 * 1024;

/// The hidden-temp binding name for index `n`.
fn temp_name(n: u32) -> String {
    format!("#t{n}")
}

/// True when `e` is (or contains as its chain spine) an optional link:
/// `a?.b`, `a?.b.c`, `a?.()`. Used to route chain expressions to
/// [`Compiler::compile_optional_chain`].
///
/// Iterative over the chain spine: the parser builds left-deep member/call
/// chains without counting them toward its depth guard, so a 3000-deep
/// `f.bind(null).bind(null)…` chain is a legal AST. The recursive walk that
/// used to live here overflowed the host stack on such chains (an
/// uncatchable process abort); the loop keeps the scan O(1) in stack.
fn is_opt_chain(e: &Expr) -> bool {
    let mut cur = e;
    loop {
        match cur {
            Expr::Call {
                callee, optional, ..
            } => {
                if *optional {
                    return true;
                }
                cur = callee;
            }
            Expr::Member { obj, optional, .. } => {
                if *optional {
                    return true;
                }
                cur = obj;
            }
            _ => return false,
        }
    }
}

/// One flattened link of a member/call/`new` chain, produced by
/// [`Compiler::compile_chain`]. `links[0]` is the outermost link; the base
/// expression the links apply to is tracked separately (see [`ChainBase`]).
enum ChainLink<'a> {
    /// `callee(...args)`: the callee value is on the stack top.
    Call { args: &'a [Expr], direct_eval: bool },
    /// `obj.prop(...args)` / `obj[prop](...args)`: the receiver is on the
    /// stack top and must be preserved as `this_arg`.
    MethodCall {
        prop: &'a Expr,
        computed: bool,
        args: &'a [Expr],
    },
    /// `super(...args)` in a derived constructor.
    SuperCall { args: &'a [Expr] },
    /// `super.m(...args)` / `super[m](...args)`.
    SuperMethodCall {
        prop: &'a Expr,
        computed: bool,
        args: &'a [Expr],
    },
    /// `import(x)`: dynamic import of the single argument.
    DynamicImport { args: &'a [Expr] },
    /// `.prop` / `[prop]` load: the object is on the stack top.
    Member { prop: &'a Expr, computed: bool },
    /// `new` application: the callee value is on the stack top.
    New { args: &'a [Expr] },
}

/// What a flattened [`ChainLink`] spine bottoms out in.
enum ChainBase<'a> {
    /// An ordinary expression, compiled before the links apply.
    Expr(&'a Expr),
    /// `import(x)`: the innermost link carries the argument; no base value.
    Import,
    /// `super(...)` / `super.m(...)`: the innermost link loads its own
    /// receiver; no base value.
    Super,
}

impl Compiler {
    /// The `stack_base` probe anchors the native-stack guard in
    /// [`Compiler::guarded`]; see [`MAX_COMPILE_NATIVE_STACK`] for the
    /// same-thread, fresh-stack requirement this imposes.
    fn new(brand_base: u32) -> Compiler {
        // Address-taken so the probe local really lives on the stack; the
        // base anchors the native-stack probe in `guarded`.
        let probe = 0u8;
        let stack_base = &probe as *const u8 as usize;
        Compiler {
            fns: Vec::new(),
            scopes: Vec::new(),
            module: ModuleRecord::default(),
            temp_counter: 0,
            class_counter: 0,
            brand_counter: brand_base,
            brand_base,
            bind_seq: 0,
            open_declare: Vec::new(),
            pending_labels: Vec::new(),
            depth: 0,
            stack_base,
            with_stack: Vec::new(),
        }
    }

    /// A fresh private brand for a class definition site. Brands are
    /// unique per compilation *unit* only when the compiler starts at a
    /// VM-globally-unique base (see [`Compiler::new`]); each evaluation of
    /// the class definition shares its site's brand.
    fn next_brand(&mut self) -> Result<u16, CompileError> {
        let b = self.brand_counter;
        if b > u16::MAX as u32 {
            return Err(CompileError::Unsupported("too many private brands"));
        }
        self.brand_counter += 1;
        Ok(b as u16)
    }

    /// How many private brands this compilation consumed, so the VM can
    /// advance its global brand sequence past this unit's range.
    fn brands_used(&self) -> u32 {
        self.brand_counter - self.brand_base
    }

    /// Run `f` with the recursion guard held, returning
    /// [`CompileError::DepthExceeded`] instead of overflowing the host
    /// stack. Every recursive compile walker funnels through here.
    fn guarded<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, CompileError>,
    ) -> Result<T, CompileError> {
        if self.depth >= MAX_COMPILE_DEPTH || self.native_stack_low() {
            return Err(CompileError::DepthExceeded);
        }
        self.depth += 1;
        let r = f(self);
        self.depth -= 1;
        r
    }

    /// Whether the compiler's native-stack budget is spent, measured by
    /// comparing the current stack pointer against the base captured in
    /// [`Compiler::new`]. The probe local is address-taken so it really
    /// lives on the stack. Checked in [`Compiler::guarded`] alongside the
    /// depth counter: the counter alone cannot bound native use because
    /// one walker frame is several KiB in unoptimized builds.
    fn native_stack_low(&self) -> bool {
        let probe = 0u8;
        let sp = &probe as *const u8 as usize;
        self.stack_base.abs_diff(sp) >= MAX_COMPILE_NATIVE_STACK
    }

    // -- code emission ----------------------------------------------------

    fn code(&mut self) -> &mut Vec<u8> {
        &mut self.fns.last_mut().expect("no active function").code
    }

    fn emit(&mut self, op: Op) {
        let code = self.code();
        encode(op, code);
    }

    /// Emit a jump with a dummy offset; returns the operand position for [`Compiler::patch`].
    fn emit_jump(&mut self, op: Op) -> usize {
        let pos = self.code().len();
        self.emit(op);
        pos + 1
    }

    fn patch(&mut self, operand_pos: usize) -> Result<(), CompileError> {
        let target = self.code().len();
        let off = target as isize - (operand_pos as isize + 2);
        if !(i16::MIN as isize..=i16::MAX as isize).contains(&off) {
            return Err(CompileError::JumpTooFar);
        }
        let bytes = (off as i16).to_le_bytes();
        let code = self.code();
        code[operand_pos] = bytes[0];
        code[operand_pos + 1] = bytes[1];
        Ok(())
    }

    /// Patch an [`Op::EnterTry`] placeholder with the absolute handler pc.
    fn patch_handler(&mut self, operand_pos: usize) -> Result<(), CompileError> {
        let target = self.code().len();
        if target > u16::MAX as usize {
            return Err(CompileError::JumpTooFar);
        }
        let bytes = (target as u16).to_le_bytes();
        let code = self.code();
        code[operand_pos] = bytes[0];
        code[operand_pos + 1] = bytes[1];
        Ok(())
    }

    // -- constants --------------------------------------------------------

    fn intern(&mut self, s: &str) -> Result<u16, CompileError> {
        self.intern_wtf8(Wtf8::from_str(s))
    }

    /// Intern a (possibly ill-formed) string constant.
    fn intern_wtf8(&mut self, s: &Wtf8) -> Result<u16, CompileError> {
        let f = self.fns.last_mut().expect("no active function");
        if let Some(&idx) = f.str_map.get(s) {
            return Ok(idx);
        }
        if f.consts.len() > u16::MAX as usize {
            return Err(CompileError::TooManyConstants);
        }
        let idx = f.consts.len() as u16;
        f.consts.push(Constant::Str(s.to_owned()));
        f.str_map.insert(s.to_owned(), idx);
        Ok(idx)
    }

    fn intern_int(&mut self, v: i32) -> Result<u16, CompileError> {
        let f = self.fns.last_mut().expect("no active function");
        if let Some(&idx) = f.int_map.get(&v) {
            return Ok(idx);
        }
        if f.consts.len() > u16::MAX as usize {
            return Err(CompileError::TooManyConstants);
        }
        let idx = f.consts.len() as u16;
        f.consts.push(Constant::Int(v));
        f.int_map.insert(v, idx);
        Ok(idx)
    }

    fn intern_float(&mut self, v: f64) -> Result<u16, CompileError> {
        let f = self.fns.last_mut().expect("no active function");
        if let Some(&idx) = f.float_map.get(&v.to_bits()) {
            return Ok(idx);
        }
        if f.consts.len() > u16::MAX as usize {
            return Err(CompileError::TooManyConstants);
        }
        let idx = f.consts.len() as u16;
        f.consts.push(Constant::Float(v));
        f.float_map.insert(v.to_bits(), idx);
        Ok(idx)
    }

    fn intern_template(&mut self, t: FuncTemplate) -> Result<u16, CompileError> {
        let f = self.fns.last_mut().expect("no active function");
        if f.consts.len() > u16::MAX as usize {
            return Err(CompileError::TooManyConstants);
        }
        let idx = f.consts.len() as u16;
        f.consts.push(Constant::FuncTemplate(t));
        Ok(idx)
    }

    // -- scopes and bindings ----------------------------------------------

    fn push_scope(&mut self, function: bool) {
        let fn_idx = self.fns.len() - 1;
        self.scopes.push(Scope {
            names: Rc::new(HashMap::new()),
            function,
            fn_idx,
        });
    }

    /// Pop scopes until (and not including) `base`.
    fn pop_scopes_to(&mut self, base: usize) {
        self.scopes.truncate(base);
    }

    /// Pop one scope (the innermost).
    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn function_scope_idx(&self) -> usize {
        self.scopes
            .iter()
            .rposition(|s| s.function)
            .expect("no function scope")
    }

    /// Declare `name` in the scope at `scope_idx` (idempotent). Returns the binding id.
    ///
    /// Name resolution is by source name through the compile-time scope
    /// stack, but the VM resolves slots by interned name string, so a
    /// lexical binding in a block scope gets a unique internal name
    /// (`x#7`): sibling blocks declaring the same name must not collide in
    /// the VM's name-to-slot map. The scope map keeps the plain source name
    /// (and `#tdz#x` for the flag) as its key, so lookups are unchanged.
    ///
    /// A named class expression's inner name ([`BindingKind::SelfName`])
    /// is likewise uniquified when declared outside a function scope: it
    /// shares the enclosing function's binding table with that function's
    /// own bindings, and a same-named outer `var` (plain internal name)
    /// must keep resolving to its own slot.
    fn declare_in(
        &mut self,
        scope_idx: usize,
        name: &str,
        kind: BindingKind,
    ) -> Result<u16, CompileError> {
        // Lexical bindings outside the function scope always need a unique
        // name; a same-scope lexical/`var` collision also shadows with a
        // fresh (unique) binding so the lexical keeps its own TDZ flag.
        let mut unique = (kind.is_lexical() || kind == BindingKind::SelfName)
            && !self.scopes[scope_idx].function;
        if let Some(&id) = self.scopes[scope_idx].names.get(name) {
            let existing_kind =
                self.fns.last().expect("no active function").bindings[id as usize].kind;
            // A named function expression's self-name binding never merges
            // with a later declaration: params, `var`s, and function
            // declarations shadow it with a fresh binding (the spec's
            // func-env sits outside the function's var env, so the inner
            // declaration wins name resolution while the self-name slot
            // keeps holding the callee).
            if existing_kind == BindingKind::SelfName {
                unique = true;
            } else if kind == BindingKind::Function {
                // `var f; function f(){}` — the function declaration wins.
                self.fns.last_mut().expect("no active function").bindings[id as usize].kind =
                    BindingKind::Function;
                return Ok(id);
            } else if kind.is_lexical() == existing_kind.is_lexical() {
                return Ok(id);
            } else {
                unique = true;
            }
        }
        let internal = if unique {
            let n = self.bind_seq;
            self.bind_seq += 1;
            format!("{name}#{n}")
        } else {
            name.to_string()
        };
        let name_const = self.intern(&internal)?;
        let id = {
            let f = self.fns.last_mut().expect("no active function");
            if f.bindings.len() >= u16::MAX as usize {
                return Err(CompileError::TooManyLocals);
            }
            let id = f.bindings.len() as u16;
            f.bindings.push(Binding {
                name_const,
                kind,
                func_const: None,
            });
            id
        };
        Rc::make_mut(&mut self.scopes[scope_idx].names).insert(name.to_string(), id);
        // `eval` support: record `var`-scoped source names on the owning
        // function, deduplicated (`var x; var x;` and `var f; function f(){}`
        // share one slot). The VM uses these for sloppy direct-eval sharing
        // and indirect-eval global installation.
        if matches!(kind, BindingKind::Var | BindingKind::Function) {
            let f = self.fns.last_mut().expect("no active function");
            if !f.var_names.iter().any(|n| n == name) {
                f.var_names.push(name.to_string());
            }
        }
        // Lexical bindings get a hidden TDZ flag in the same scope. The
        // flag's interned name is derived from the (possibly unique)
        // internal binding name so sibling blocks' flags cannot collide;
        // the scope-map key stays the plain `#tdz#name`.
        if kind.is_lexical() {
            let flag = format!("#tdz#{name}");
            let flag_internal = format!("#tdz#{internal}");
            let flag_const = self.intern(&flag_internal)?;
            let fid = {
                let f = self.fns.last_mut().expect("no active function");
                if f.bindings.len() >= u16::MAX as usize {
                    return Err(CompileError::TooManyLocals);
                }
                let fid = f.bindings.len() as u16;
                f.bindings.push(Binding {
                    name_const: flag_const,
                    kind: BindingKind::Hidden,
                    func_const: None,
                });
                fid
            };
            Rc::make_mut(&mut self.scopes[scope_idx].names).insert(flag, fid);
        }
        Ok(id)
    }

    fn declare_current(&mut self, name: &str, kind: BindingKind) -> Result<u16, CompileError> {
        let idx = self.scopes.len() - 1;
        self.declare_in(idx, name, kind)
    }

    fn declare_function(&mut self, name: &str) -> Result<u16, CompileError> {
        let idx = self.function_scope_idx();
        self.declare_in(idx, name, BindingKind::Function)
    }

    /// A hidden temporary binding in the current function scope. Indices
    /// are recycled through the function's free-list
    /// ([`FnState::temp_free`]): a temp whose last use has been passed to
    /// [`Compiler::release_temp`] is handed out again instead of declaring
    /// a fresh binding, which keeps the binding table and the prologue
    /// small. Callers must release a temp only after its last use; every
    /// release site below was audited for that.
    fn hidden_temp(&mut self) -> String {
        let n = match self
            .fns
            .last_mut()
            .expect("no active function")
            .temp_free
            .pop()
        {
            Some(n) => n,
            None => {
                let n = self.temp_counter;
                self.temp_counter += 1;
                let name = temp_name(n);
                let idx = self.function_scope_idx();
                self.declare_in(idx, &name, BindingKind::Hidden)
                    .expect("temp declaration cannot fail");
                n
            }
        };
        temp_name(n)
    }

    /// Return a hidden temp index to the current function's free-list. The
    /// temp must be dead: no later-emitted code may read or write it.
    fn release_temp(&mut self, name: &str) {
        if let Some(suffix) = name.strip_prefix("#t")
            && let Ok(n) = suffix.parse::<u32>()
        {
            self.fns
                .last_mut()
                .expect("no active function")
                .temp_free
                .push(n);
        }
    }

    /// Innermost-first name lookup across all scopes. Returns the declaring
    /// function's index and the binding id.
    fn lookup(&self, name: &str) -> Option<(usize, u16)> {
        for scope in self.scopes.iter().rev() {
            if let Some(&id) = scope.names.get(name) {
                return Some((scope.fn_idx, id));
            }
        }
        None
    }

    /// Resolve `name` to an access path, capturing upvalues as needed.
    fn resolve(&mut self, name: &str) -> Result<Resolved, CompileError> {
        self.resolve_full(name).map(|(r, _)| r)
    }

    /// Like [`Compiler::resolve`], but also returns the declaring
    /// function's index and the binding id for declared names (`None` for
    /// globals).
    fn resolve_full(
        &mut self,
        name: &str,
    ) -> Result<(Resolved, Option<(usize, u16)>), CompileError> {
        let cur = self.fns.len() - 1;
        match self.lookup(name) {
            Some((decl_fn, id)) => {
                let (kind, nc) = {
                    let b = &self.fns[decl_fn].bindings[id as usize];
                    (b.kind, b.name_const)
                };
                if decl_fn == cur {
                    Ok((Resolved::Local(nc, kind), Some((decl_fn, id))))
                } else {
                    let idx = self.resolve_upvalue(name, decl_fn, id)?;
                    Ok((Resolved::Upvalue(idx, kind), Some((decl_fn, id))))
                }
            }
            None => Ok((Resolved::Global(self.intern(name)?), None)),
        }
    }

    /// Ensure functions `decl_fn+1..=cur` capture `name`; returns the upvalue
    /// index in the current function.
    fn resolve_upvalue(
        &mut self,
        name: &str,
        decl_fn: usize,
        id: u16,
    ) -> Result<u16, CompileError> {
        let cur = self.fns.len() - 1;
        if let Some(&idx) = self.fns[cur].upvalue_names.get(name) {
            return Ok(idx);
        }
        let mut prev = 0u16;
        for k in decl_fn + 1..=cur {
            if let Some(&idx) = self.fns[k].upvalue_names.get(name) {
                prev = idx;
                continue;
            }
            let desc = if k == decl_fn + 1 {
                UpvalueDesc::Local(id)
            } else {
                UpvalueDesc::Upvalue(prev)
            };
            if self.fns[k].upvalues.len() > u16::MAX as usize {
                return Err(CompileError::TooManyLocals);
            }
            let idx = self.fns[k].upvalues.len() as u16;
            self.fns[k].upvalues.push(desc);
            self.fns[k].upvalue_names.insert(name.to_string(), idx);
            prev = idx;
        }
        Ok(prev)
    }

    /// Force-capture every in-scope binding as an upvalue of the current
    /// function, for direct `eval` support. A bare `eval(...)` call reads
    /// and writes bindings from its *lexical* scope chain, but the names
    /// are only known at runtime, so the compiler cannot capture them
    /// lazily like ordinary references. Instead, when compiling a direct
    /// eval call, every binding visible at that point — innermost scope
    /// first, so shadowing resolves correctly — is threaded through
    /// [`resolve_upvalue`](Self::resolve_upvalue) into this function (and
    /// through every intermediate function, as usual). The VM then
    /// resolves eval names against the caller frame's captured cells (see
    /// `resolve_eval_caller`).
    ///
    /// The current function's own bindings are skipped: the eval frame
    /// resolves those through the caller frame's environment directly.
    /// Hidden (`#`-prefixed) bindings are skipped, except `#tdz#` flags,
    /// which are captured alongside the lexical binding they guard — but
    /// only when that binding itself was captured from the same scope, so
    /// a flag can never be paired with a shadowed value from another
    /// scope.
    fn capture_eval_scope(&mut self) -> Result<(), CompileError> {
        let cur = self.fns.len() - 1;
        // Snapshot the scope stack: `resolve_upvalue` mutates `self.fns`.
        let scopes: Vec<(usize, Vec<(String, u16)>)> = self
            .scopes
            .iter()
            .rev()
            .map(|s| {
                (
                    s.fn_idx,
                    s.names.iter().map(|(k, v)| (k.clone(), *v)).collect(),
                )
            })
            .collect();
        for (fn_idx, names) in &scopes {
            if *fn_idx == cur {
                continue;
            }
            // Value bindings first, innermost scope first: `resolve_upvalue`
            // keeps the first registration per name, so the innermost
            // binding wins over shadowed outer ones.
            let mut won: Vec<&str> = Vec::new();
            for (name, id) in names {
                if name.starts_with('#') {
                    continue;
                }
                let fresh = !self.fns[cur].upvalue_names.contains_key(name.as_str());
                self.resolve_upvalue(name, *fn_idx, *id)?;
                if fresh {
                    won.push(name.as_str());
                }
            }
            // TDZ flags, paired with the binding they guard.
            for (name, id) in names {
                if !name.starts_with("#tdz#") {
                    continue;
                }
                let src = &name["#tdz#".len()..];
                if won.contains(&src) {
                    self.resolve_upvalue(name, *fn_idx, *id)?;
                }
            }
        }
        Ok(())
    }

    // -- variable access ----------------------------------------------------

    /// Resolve how the current function loads the TDZ flag of the lexical
    /// binding `(decl_fn, id)`, capturing it as an upvalue when needed. The
    /// result is cached per function: a lexical binding's flag is resolved
    /// once per function instead of once per access.
    fn tdz_flag(
        &mut self,
        source_name: &str,
        decl_fn: usize,
        id: u16,
    ) -> Result<TdzAccess, CompileError> {
        let key = ((decl_fn as u64) << 32) | id as u64;
        if let Some(&access) = self
            .fns
            .last()
            .expect("no active function")
            .tdz_flag_cache
            .get(&key)
        {
            return Ok(access);
        }
        let flag = format!("#tdz#{source_name}");
        let access = match self.resolve(&flag)? {
            Resolved::Local(fc, _) => TdzAccess::Local(fc),
            Resolved::Upvalue(fi, _) => TdzAccess::Upvalue(fi),
            Resolved::Global(_) => {
                // A TDZ flag is always declared at the same scope as the
                // lexical binding it guards. If the caller's VarKind disagrees
                // with the resolved binding, this is a scope bookkeeping bug;
                // surface it as a recoverable compile error, never a panic.
                return Err(CompileError::Internal(
                    "tdz flag resolved to global: declare/codegen scope mismatch",
                ));
            }
        };
        self.fns
            .last_mut()
            .expect("no active function")
            .tdz_flag_cache
            .insert(key, access);
        Ok(access)
    }

    fn emit_flag_load(&mut self, access: TdzAccess) {
        match access {
            TdzAccess::Local(fc) => self.emit(Op::LoadName(fc)),
            TdzAccess::Upvalue(fi) => self.emit(Op::LoadUpvalue(fi)),
        }
    }

    /// Find the shared throw stub for `(key, const_check)`.
    fn find_tdz_stub(&self, key: u64, const_check: bool) -> Option<usize> {
        self.fns
            .last()
            .expect("no active function")
            .tdz_stubs
            .get(&(key, const_check))
            .copied()
    }

    /// Emit a TDZ read check for a lexical binding; leaves the stack
    /// unchanged. The throw sequence is shared: the first check per
    /// binding uses the classic inline form (and records its throw
    /// position); later checks jump back to it instead of re-emitting.
    fn emit_tdz_check(
        &mut self,
        source_name: &str,
        decl_fn: usize,
        id: u16,
    ) -> Result<(), CompileError> {
        let key = ((decl_fn as u64) << 32) | id as u64;
        let access = self.tdz_flag(source_name, decl_fn, id)?;
        self.emit_flag_load(access);
        if let Some(target) = self.find_tdz_stub(key, false) {
            let pos = self.emit_jump(Op::JumpIfFalse(0));
            return self.patch_to(pos, target);
        }
        let ok = self.emit_jump(Op::JumpIfTrue(0));
        let target = self.code_len();
        let msg = self.intern(&format!(
            "ReferenceError: cannot access '{source_name}' before initialization"
        ))?;
        self.emit(Op::LoadConst(msg));
        self.emit(Op::Throw);
        self.patch(ok)?;
        self.fns
            .last_mut()
            .expect("no active function")
            .tdz_stubs
            .insert((key, false), target);
        Ok(())
    }

    /// Compile a variable read; leaves `[value]`. Inside a `with` body the
    /// read probes the with-objects first (see [`WithEntry`]).
    fn compile_load(&mut self, name: &str) -> Result<(), CompileError> {
        let probes = self.with_probes(name);
        if probes.is_empty() {
            return self.compile_load_inner(name);
        }
        self.compile_with_load(name, &probes)
    }

    fn compile_load_inner(&mut self, name: &str) -> Result<(), CompileError> {
        match self.resolve_full(name)? {
            (Resolved::Local(nc, kind), identity) => {
                if kind.is_lexical() {
                    let (decl_fn, id) = identity.expect("lexical binding is declared");
                    self.emit_tdz_check(name, decl_fn, id)?;
                }
                self.emit(Op::LoadName(nc));
            }
            (Resolved::Upvalue(idx, kind), identity) => {
                if kind.is_lexical() {
                    let (decl_fn, id) = identity.expect("lexical binding is declared");
                    self.emit_tdz_check(name, decl_fn, id)?;
                }
                self.emit(Op::LoadUpvalue(idx));
            }
            (Resolved::Global(nc), _) => self.emit(Op::LoadName(nc)),
        }
        Ok(())
    }

    /// Compile `typeof <ident>`: mirrors `compile_load` above, but a name
    /// that resolves to the global scope uses `Op::TypeofName`, which yields
    /// `"undefined"` for undeclared names instead of throwing
    /// `ReferenceError`. Lexical locals/upvalues keep their TDZ guard and
    /// use the throwing load (spec: `typeof` of a TDZ binding throws).
    /// Inside a `with` body the with-objects are probed first.
    fn compile_typeof_name(&mut self, name: &str) -> Result<(), CompileError> {
        let probes = self.with_probes(name);
        if probes.is_empty() {
            return self.compile_typeof_name_inner(name);
        }
        self.compile_with_typeof(name, &probes)
    }

    fn compile_typeof_name_inner(&mut self, name: &str) -> Result<(), CompileError> {
        match self.resolve_full(name)? {
            (Resolved::Local(nc, kind), identity) => {
                if kind.is_lexical() {
                    let (decl_fn, id) = identity.expect("lexical binding is declared");
                    self.emit_tdz_check(name, decl_fn, id)?;
                }
                self.emit(Op::LoadName(nc));
                self.emit(Op::Typeof);
            }
            (Resolved::Upvalue(idx, kind), identity) => {
                if kind.is_lexical() {
                    let (decl_fn, id) = identity.expect("lexical binding is declared");
                    self.emit_tdz_check(name, decl_fn, id)?;
                }
                self.emit(Op::LoadUpvalue(idx));
                self.emit(Op::Typeof);
            }
            (Resolved::Global(nc), _) => self.emit(Op::TypeofName(nc)),
        }
        Ok(())
    }

    /// Compile a plain assignment store; `[v] -> []`. Inside a `with` body
    /// the store probes the with-objects first (see [`WithEntry`]).
    fn compile_store(&mut self, name: &str) -> Result<(), CompileError> {
        let probes = self.with_probes(name);
        if probes.is_empty() {
            return self.compile_store_inner(name);
        }
        self.compile_with_store(name, &probes)
    }

    fn compile_store_inner(&mut self, name: &str) -> Result<(), CompileError> {
        match self.resolve(name)? {
            // A named function expression's own name is immutable: the
            // store is a silent no-op in sloppy mode, a `TypeError` in
            // strict mode (ECMA-262 `NamedEvaluation`).
            Resolved::Local(_, BindingKind::SelfName)
            | Resolved::Upvalue(_, BindingKind::SelfName) => self.compile_store_self_name()?,
            Resolved::Local(nc, _) => self.emit(Op::StoreName(nc)),
            Resolved::Upvalue(idx, _) => self.emit(Op::StoreUpvalue(idx)),
            Resolved::Global(nc) => self.emit(Op::StoreName(nc)),
        }
        Ok(())
    }

    /// Compile a store to a named function expression's self-name binding;
    /// `[v] -> []`. The value is discarded in both modes; strict mode
    /// additionally throws `TypeError`. The message is pre-tagged with
    /// [`COMPILER_THROW_TAG`] so the VM builds a real `TypeError` object
    /// (the `tag_compiler_throws` idiom only matches the TDZ guard shape,
    /// which an unconditional throw cannot use).
    fn compile_store_self_name(&mut self) -> Result<(), CompileError> {
        let strict = self.fns.last().expect("no active function").strict;
        self.emit(Op::Pop);
        if strict {
            let msg = self.intern(&format!(
                "{COMPILER_THROW_TAG}TypeError\u{0}assignment to constant variable"
            ))?;
            self.emit(Op::LoadConst(msg));
            self.emit(Op::Throw);
        }
        Ok(())
    }

    // -- `with` statement -------------------------------------------------
    //
    // Sloppy-mode `with` (the parser rejects strict-mode `with`) is
    // compiled without any VM support: a hidden temp holds
    // `ToObject(obj)`, and every identifier load/store/`typeof` in the
    // body probes the with-objects (innermost first) with `in`
    // (chain-walking `HasProperty`) before falling back to the ordinary
    // compile-time-resolved access. Only withs entered strictly inside
    // the name's declaring scope are probed, so an inner lexical binding
    // shadows the with-object per spec. No runtime scope is pushed, so
    // abrupt completions need no extra unwinding.
    //
    // v1 deviations: primitives throw a `TypeError` instead of boxing
    // (wrapper objects are a v1 gap; the conversion never consults the
    // lexical `Object` binding, so a shadowing `let Object = ...` cannot
    // hijack it); closures defined inside a `with` body keep ordinary
    // lexical resolution and do not observe the with-object — except the
    // compiler-generated `#for_body` extraction arrow, which captures
    // each with-object as an upvalue and keeps probing; direct `eval`
    // inside `with` likewise misses the object environment.

    /// Name constants (innermost first) of the active `with` objects that
    /// `name` must probe. Hidden (`#`-prefixed) temps never probe.
    fn with_probes(&self, name: &str) -> Vec<u16> {
        if self.with_stack.is_empty() || name.starts_with('#') {
            return Vec::new();
        }
        // Innermost scope declaring `name`, if any.
        let mut decl_depth = None;
        for (i, scope) in self.scopes.iter().enumerate().rev() {
            if scope.names.contains_key(name) {
                decl_depth = Some(i);
                break;
            }
        }
        self.with_stack
            .iter()
            .rev()
            .filter(|w| decl_depth.is_none_or(|d| w.scope_depth > d))
            .map(|w| w.name_const)
            .collect()
    }

    /// `with`-aware variable read (`[..] -> [.., value]`).
    fn compile_with_load(&mut self, name: &str, probes: &[u16]) -> Result<(), CompileError> {
        let key = self.intern(name)?;
        let mut ends = Vec::with_capacity(probes.len());
        for wnc in probes {
            self.emit(Op::LoadConst(key)); // [k]
            self.emit(Op::LoadName(*wnc)); // [k, w]
            self.emit(Op::In); // [hit]
            let miss = self.emit_jump(Op::JumpIfFalse(0));
            self.emit(Op::LoadName(*wnc)); // [w]
            self.emit(Op::LoadConst(key)); // [w, k]
            self.emit(Op::GetElem); // [v]
            ends.push(self.emit_jump(Op::Jump(0)));
            self.patch(miss)?;
        }
        self.compile_load_inner(name)?;
        for e in ends {
            self.patch(e)?;
        }
        Ok(())
    }

    /// `with`-aware `typeof` (`[..] -> [.., string]`).
    fn compile_with_typeof(&mut self, name: &str, probes: &[u16]) -> Result<(), CompileError> {
        let key = self.intern(name)?;
        let mut ends = Vec::with_capacity(probes.len());
        for wnc in probes {
            self.emit(Op::LoadConst(key));
            self.emit(Op::LoadName(*wnc));
            self.emit(Op::In);
            let miss = self.emit_jump(Op::JumpIfFalse(0));
            self.emit(Op::LoadName(*wnc));
            self.emit(Op::LoadConst(key));
            self.emit(Op::GetElem);
            self.emit(Op::Typeof);
            ends.push(self.emit_jump(Op::Jump(0)));
            self.patch(miss)?;
        }
        self.compile_typeof_name_inner(name)?;
        for e in ends {
            self.patch(e)?;
        }
        Ok(())
    }

    /// `with`-aware assignment store (`[v] -> []`).
    fn compile_with_store(&mut self, name: &str, probes: &[u16]) -> Result<(), CompileError> {
        let vt = self.hidden_temp();
        let vnc = self.binding_name_const(&vt)?;
        self.emit(Op::StoreName(vnc));
        let key = self.intern(name)?;
        let mut ends = Vec::with_capacity(probes.len());
        for wnc in probes {
            self.emit(Op::LoadConst(key)); // [k]
            self.emit(Op::LoadName(*wnc)); // [k, w]
            self.emit(Op::In); // [hit]
            let miss = self.emit_jump(Op::JumpIfFalse(0));
            self.emit(Op::LoadName(*wnc)); // [w]
            self.emit(Op::LoadConst(key)); // [w, k]
            self.emit(Op::LoadName(vnc)); // [w, k, v]
            self.emit(Op::SetProp); // []
            ends.push(self.emit_jump(Op::Jump(0)));
            self.patch(miss)?;
        }
        self.emit(Op::LoadName(vnc)); // [v]
        self.compile_store_inner(name)?; // [] — pops the value
        for e in ends {
            self.patch(e)?;
        }
        self.release_temp(&vt);
        Ok(())
    }

    /// Compile `with (obj) body`.
    fn compile_with(&mut self, obj: &Expr, body: &Stmt) -> Result<(), CompileError> {
        // The object expression evaluates in the enclosing environment, so
        // outer `with` probes still apply to it.
        self.compile_expr(obj)?; // [v]
        // `ToObject(v)`: null/undefined throw a `TypeError`; primitives
        // are boxed through the global `Object` constructor.
        self.emit(Op::Dup); // [v, v]
        self.emit(Op::PushNull); // [v, v, null]
        self.emit(Op::StrictEq); // [v, is_null]
        let not_null = self.emit_jump(Op::JumpIfFalse(0));
        let throw_msg = self.intern(&format!(
            "{COMPILER_THROW_TAG}TypeError\u{0}cannot convert null or undefined to object"
        ))?;
        self.emit(Op::LoadConst(throw_msg));
        self.emit(Op::Throw);
        self.patch(not_null)?;
        self.emit(Op::Dup); // [v, v]
        self.emit(Op::PushUndefined); // [v, v, undefined]
        self.emit(Op::StrictEq); // [v, is_undef]
        let not_undef = self.emit_jump(Op::JumpIfFalse(0));
        self.emit(Op::LoadConst(throw_msg));
        self.emit(Op::Throw);
        self.patch(not_undef)?;
        // `ToObject(v)`: null/undefined throw a `TypeError` (above).
        // Objects and functions pass through; primitives throw a
        // `TypeError` — real primitive boxing (wrapper objects) is a v1
        // gap, and throwing here directly (rather than calling the
        // *lexical* `Object` binding, which a user `let Object = ...`
        // could shadow) keeps the conversion unhijackable. A hidden
        // `#to_object` intrinsic would be the cleaner fix, but intrinsic
        // registration lives outside this module; this is equivalent
        // modulo the error message.
        self.emit(Op::Dup); // [v, v]
        self.emit(Op::Typeof); // [v, t]
        let obj_s = self.intern("object")?;
        self.emit(Op::LoadConst(obj_s));
        self.emit(Op::StrictEq); // [v, is_obj]
        let is_obj = self.emit_jump(Op::JumpIfTrue(0));
        self.emit(Op::Dup); // [v, v]
        self.emit(Op::Typeof); // [v, t]
        let fun_s = self.intern("function")?;
        self.emit(Op::LoadConst(fun_s));
        self.emit(Op::StrictEq); // [v, is_fun]
        let is_fun = self.emit_jump(Op::JumpIfTrue(0));
        let prim_msg = self.intern(&format!(
            "{COMPILER_THROW_TAG}TypeError\u{0}cannot convert primitive to object"
        ))?;
        self.emit(Op::LoadConst(prim_msg));
        self.emit(Op::Throw);
        self.patch(is_obj)?;
        self.patch(is_fun)?; // [v]
        let w = self.hidden_temp();
        let wnc = self.binding_name_const(&w)?;
        self.emit(Op::StoreName(wnc)); // []
        let depth = self.scopes.len();
        self.with_stack.push(WithEntry {
            name: w.clone(),
            name_const: wnc,
            scope_depth: depth,
        });
        let r = self.compile_stmt(body);
        self.with_stack.pop();
        self.release_temp(&w);
        r
    }

    /// Compile a declaration store (`var`/`let`/`const` init); `[v] -> []`.
    /// For lexicals the TDZ flag is set (no const-reassignment check: this is
    /// the initializing write). The flag access is cached per binding, so
    /// repeated stores do not re-resolve it.
    fn compile_store_decl(&mut self, name: &str, kind: VarKind) -> Result<(), CompileError> {
        let lexical = kind != VarKind::Var;
        match self.resolve_full(name)? {
            (Resolved::Local(nc, _), identity) => {
                if lexical {
                    self.emit(Op::Dup);
                }
                self.emit(Op::StoreName(nc));
                if lexical {
                    let (decl_fn, id) = identity.expect("lexical binding is declared");
                    let access = self.tdz_flag(name, decl_fn, id)?;
                    self.emit(Op::PushTrue);
                    match access {
                        TdzAccess::Local(fc) => self.emit(Op::StoreName(fc)),
                        TdzAccess::Upvalue(fi) => self.emit(Op::StoreUpvalue(fi)),
                    }
                    self.emit(Op::Pop);
                }
            }
            (Resolved::Upvalue(idx, _), identity) => {
                if lexical {
                    self.emit(Op::Dup);
                }
                self.emit(Op::StoreUpvalue(idx));
                if lexical {
                    let (decl_fn, id) = identity.expect("lexical binding is declared");
                    let access = self.tdz_flag(name, decl_fn, id)?;
                    self.emit(Op::PushTrue);
                    match access {
                        TdzAccess::Local(fc) => self.emit(Op::StoreName(fc)),
                        TdzAccess::Upvalue(fi) => self.emit(Op::StoreUpvalue(fi)),
                    }
                    self.emit(Op::Pop);
                }
            }
            (Resolved::Global(nc), _) => {
                // Undeclared assignment target: sloppy global (strict throws
                // at runtime via StoreName).
                self.emit(Op::StoreName(nc));
            }
        }
        Ok(())
    }

    /// Compile the "assignment to constant" runtime check; `[v] -> [v]`.
    /// Throws `TypeError` if the target is an initialized `const` binding.
    /// The throw sequence is shared per binding: the first check uses the
    /// classic inline form (and records its throw position); later checks
    /// jump back to it.
    fn compile_const_check(&mut self, name: &str) -> Result<(), CompileError> {
        let target_fn = match self.lookup(name) {
            Some((decl_fn, id))
                if self.fns[decl_fn].bindings[id as usize].kind == BindingKind::Const =>
            {
                Some((decl_fn, id))
            }
            _ => None,
        };
        let (decl_fn, id) = match target_fn {
            Some(x) => x,
            None => return Ok(()),
        };
        let key = ((decl_fn as u64) << 32) | id as u64;
        // [v] -> [v, v]: the duplicate stays for the caller's store.
        self.emit(Op::Dup);
        let access = self.tdz_flag(name, decl_fn, id)?;
        self.emit_flag_load(access);
        if let Some(target) = self.find_tdz_stub(key, true) {
            // Jump *to* the shared throw stub: the throw fires when the
            // initialized flag is true, so this is JumpIfTrue (the first
            // site's JumpIfFalse skips *past* the throw instead).
            let pos = self.emit_jump(Op::JumpIfTrue(0));
            return self.patch_to(pos, target);
        }
        let first = self.emit_jump(Op::JumpIfFalse(0));
        let target = self.code_len();
        let msg = self.intern("TypeError: assignment to constant variable")?;
        self.emit(Op::LoadConst(msg));
        self.emit(Op::Throw);
        self.patch(first)?;
        self.fns
            .last_mut()
            .expect("no active function")
            .tdz_stubs
            .insert((key, true), target);
        Ok(())
    }

    /// Call a hidden intrinsic: stack `[a0..a_{n-1}] -> [result]`.
    /// The argument temps are released back to the free-list afterwards;
    /// they are dead once the call completes.
    fn call_intrinsic(&mut self, name: &str, argc: u8) -> Result<(), CompileError> {
        let mut tmps = Vec::with_capacity(argc as usize);
        for _ in 0..argc {
            tmps.push(self.hidden_temp());
        }
        for t in tmps.iter().rev() {
            let nc = self.binding_name_const(t)?;
            self.emit(Op::StoreName(nc));
        }
        let idx = self.intern(&format!("#{name}"))?;
        self.emit(Op::LoadName(idx));
        self.emit(Op::PushUndefined);
        for t in &tmps {
            let nc = self.binding_name_const(t)?;
            self.emit(Op::LoadName(nc));
        }
        self.emit(Op::Call(argc));
        for t in tmps {
            self.release_temp(&t);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// The output of compiling one unit: the top-level template plus how
/// many private brands it consumed, so the VM can advance its global
/// brand sequence past this unit's range and the next compilation starts
/// at a fresh, non-colliding base.
#[derive(Debug)]
pub struct CompiledUnit {
    /// The compiled top-level template.
    pub template: FuncTemplate,
    /// Private brands consumed (see [`Compiler::next_brand`]).
    pub brands_used: u32,
}

/// Compile a parsed script [`Program`] into its top-level [`FuncTemplate`].
///
/// The top level is an implicit function named `<main>` whose strictness
/// comes from the program (`"use strict"` directive or module goal).
/// `brand_base` is the VM-globally-unique base for this unit's private
/// brands; the returned [`CompiledUnit::brands_used`] lets the caller
/// advance the sequence.
pub fn compile(program: &Program, brand_base: u32) -> Result<CompiledUnit, CompileError> {
    let (unit, _) = compile_module(program, brand_base)?;
    Ok(unit)
}

/// Compile a parsed module [`Program`], returning the compiled unit plus
/// the import/export side table for the module linker.
pub fn compile_module(
    program: &Program,
    brand_base: u32,
) -> Result<(CompiledUnit, ModuleRecord), CompileError> {
    let mut c = Compiler::new(brand_base);
    c.push_function(
        "<main>".to_string(),
        0,
        false,
        false,
        false,
        program.strict,
        false,
    );
    c.hoist_names(&program.body)?;
    c.declare_stmts(&program.body)?;
    c.emit_prologue()?;
    // The script completion value is the last statement's value when it is
    // an expression statement, `undefined` otherwise.
    let body = &program.body;
    match body.last() {
        Some(Stmt::Expr(e)) => {
            for s in &body[..body.len() - 1] {
                c.compile_stmt(s)?;
            }
            c.compile_expr(e)?;
        }
        _ => {
            c.compile_stmts(body)?;
            c.emit(Op::PushUndefined);
        }
    }
    c.emit(Op::Return);
    let module = std::mem::take(&mut c.module);
    let brands_used = c.brands_used();
    let unit = CompiledUnit {
        template: c.pop_function(),
        brands_used,
    };
    Ok((unit, module))
}

// ---------------------------------------------------------------------------
// Function setup
// ---------------------------------------------------------------------------

impl Compiler {
    #[allow(clippy::too_many_arguments)]
    fn push_function(
        &mut self,
        name: String,
        arity: u16,
        is_arrow: bool,
        is_async: bool,
        is_generator: bool,
        strict: bool,
        is_method: bool,
    ) {
        // Arrows and nested functions inherit the enclosing class context so
        // `super.prop` keeps resolving lexically.
        let class_stack = self
            .fns
            .last()
            .map(|f| f.class_stack.clone())
            .unwrap_or_default();
        let scope_base = self.scopes.len();
        // A nested function body keeps ordinary lexical name resolution:
        // its loads/stores must not probe an enclosing `with` object's
        // hidden temp (which lives in the outer frame and would resolve
        // to the global object — or a ReferenceError — from here).
        // Stash the stack on the incoming frame; `pop_function` restores
        // it. (v1 deviation: closures do not observe the with-object.)
        let outer_with = std::mem::take(&mut self.with_stack);
        self.fns.push(FnState {
            name,
            arity,
            is_arrow,
            is_async,
            is_generator,
            strict,
            is_method,
            is_derived_ctor: false,
            mapped_arg_slots: None,
            code: Vec::new(),
            consts: Vec::new(),
            str_map: HashMap::new(),
            float_map: HashMap::new(),
            int_map: HashMap::new(),
            bindings: Vec::new(),
            var_names: Vec::new(),
            upvalues: Vec::new(),
            upvalue_names: HashMap::new(),
            loops: Vec::new(),
            protected: Vec::new(),
            class_stack,
            ctor_ctx: None,
            scope_base,
            outer_with,
            saved: Vec::new(),
            scope_cursor: 0,
            finally_bases: HashMap::new(),
            extracted: false,
            extracted_loop: 0,
            extracted_params: Vec::new(),
            unresolved_labels: Vec::new(),
            extracted_fn_inits: Vec::new(),
            no_vars: false,
            temp_free: Vec::new(),
            tdz_flag_cache: HashMap::new(),
            tdz_stubs: HashMap::new(),
            is_class_init: false,
        });
        self.push_scope(true);
        // `#this` / `#newtarget` are initialized by the VM on entry.
        // Arrows declare neither: they capture the enclosing `this` /
        // `new.target` lexically via upvalues (see `compile_this` /
        // `compile_new_target`), and the VM ignores call-time values
        // for them.
        if !is_arrow {
            self.declare_current("#this", BindingKind::VmProvided)
                .expect("fresh function");
            self.declare_current("#newtarget", BindingKind::VmProvided)
                .expect("fresh function");
        }
        self.declare_current("#argc", BindingKind::VmProvided)
            .expect("fresh function");
        for i in 0..arity {
            self.declare_current(&format!("#param#{i}"), BindingKind::Hidden)
                .expect("fresh function");
        }
        self.declare_current("#args", BindingKind::Hidden)
            .expect("fresh function");
    }

    fn pop_function(&mut self) -> FuncTemplate {
        let mut f = self.fns.pop().expect("no active function");
        self.pop_scopes_to(f.scope_base);
        // Restore the enclosing function's `with` stack stashed by
        // `push_function` (nested bodies compile with an empty stack).
        self.with_stack = std::mem::take(&mut f.outer_with);
        // Peephole: rewrite compile-time-resolved local accesses to direct
        // slot ops. `LoadName`/`StoreName` carry the binding's name
        // constant, which the VM would resolve through `name_to_id` (a
        // hash lookup plus a `const_str` call) on every execution. Each
        // function-scoped binding has a unique internal name — shadowing
        // renames to `{name}#{n}` in `declare_in` — so the name constant
        // identifies exactly one binding id, and the rewrite is a pure
        // in-place tag swap: code size and jump targets are untouched.
        // `TypeofName` is deliberately left alone: it is only emitted for
        // global-scope names, where the typeof protocol (not the value
        // load) is what matters.
        {
            let mut slot_of: Vec<Option<u16>> = vec![None; f.consts.len()];
            for (id, b) in f.bindings.iter().enumerate() {
                slot_of[b.name_const as usize] = Some(id as u16);
            }
            let mut pc = 0;
            while let Some((op, next)) = decode(&f.code, pc) {
                let rewrite = match op {
                    Op::LoadName(nc) => slot_of
                        .get(nc as usize)
                        .copied()
                        .flatten()
                        .map(Op::LoadLocal),
                    Op::StoreName(nc) => slot_of
                        .get(nc as usize)
                        .copied()
                        .flatten()
                        .map(Op::StoreLocal),
                    Op::StrAppendName(nc) => slot_of
                        .get(nc as usize)
                        .copied()
                        .flatten()
                        .map(Op::StrAppendLocal),
                    Op::StrAppendBaseName(nc) => slot_of
                        .get(nc as usize)
                        .copied()
                        .flatten()
                        .map(Op::StrAppendBaseLocal),
                    _ => None,
                };
                if let Some(new_op) = rewrite {
                    // `LoadLocal`/`StoreLocal`/`StrAppendLocal`/
                    // `StrAppendBaseLocal` encode as tag + u16 LE, the
                    // same shape as `LoadName`/`StoreName`: patch in
                    // place.
                    f.code[pc] = tag(new_op);
                    let bytes = match new_op {
                        Op::LoadLocal(s)
                        | Op::StoreLocal(s)
                        | Op::StrAppendLocal(s)
                        | Op::StrAppendBaseLocal(s) => s.to_le_bytes(),
                        _ => unreachable!(),
                    };
                    f.code[pc + 1] = bytes[0];
                    f.code[pc + 2] = bytes[1];
                }
                pc = next;
            }
        }
        // Binding slot names for the VM's upvalue resolution. Every
        // `Binding::name_const` is a `Constant::Str` by construction.
        let binding_names: Vec<String> = f
            .bindings
            .iter()
            .map(|b| match &f.consts[b.name_const as usize] {
                // Binding names are identifiers: always well-formed.
                Constant::Str(s) => s.to_utf8_lossy().into_owned(),
                _ => String::new(),
            })
            .collect();
        // Upvalue source names, parallel to `upvalues`: invert the
        // name->index map built by `resolve_upvalue`. Each pushed desc
        // registers exactly one name, so the inversion is total.
        let mut upvalue_names = vec![String::new(); f.upvalues.len()];
        for (name, idx) in &f.upvalue_names {
            if let Some(slot) = upvalue_names.get_mut(*idx as usize) {
                *slot = name.clone();
            }
        }
        let binding_kinds: Vec<BindingKind> = f.bindings.iter().map(|b| b.kind).collect();
        FuncTemplate {
            name: f.name,
            arity: f.arity,
            code: f.code,
            consts: f.consts,
            upvalues: f.upvalues,
            upvalue_names,
            binding_names,
            binding_kinds,
            var_names: f.var_names,
            is_async: f.is_async,
            is_generator: f.is_generator,
            strict: f.strict,
            is_method: f.is_method,
            is_derived_ctor: f.is_derived_ctor,
            mapped_arg_slots: f.mapped_arg_slots,
            script_or_module: None,
        }
    }

    /// Emit the prologue: initialize every binding declared in this function.
    /// `var`s and hidden temps become `undefined`; lexical bindings become
    /// `undefined` with a falsy TDZ flag; function declarations become their
    /// closures. `#this`/`#newtarget`/`#argc` are VM-provided and skipped.
    fn emit_prologue(&mut self) -> Result<(), CompileError> {
        let n = self.fns.last().expect("no active function").bindings.len();
        for i in 0..n {
            // Copy the fields out so `emit` can take `&mut self`.
            let (kind, name_const, func_const) = {
                let b = &self.fns.last().expect("no active function").bindings[i];
                (b.kind, b.name_const, b.func_const)
            };
            match kind {
                // Initialized by the VM on frame entry, not by the
                // prologue (which would clobber the values).
                BindingKind::VmProvided | BindingKind::SelfName => {}
                BindingKind::Function => match func_const {
                    Some(ti) => {
                        self.emit(Op::NewFunction(ti));
                        self.emit(Op::StoreName(name_const));
                    }
                    None => {
                        // Declared but never given a template; treat as var.
                        self.emit(Op::PushUndefined);
                        self.emit(Op::StoreName(name_const));
                    }
                },
                _ => {
                    self.emit(Op::PushUndefined);
                    self.emit(Op::StoreName(name_const));
                }
            }
        }
        Ok(())
    }

    /// Bind actual arguments to `#param#i` temps, then bind parameter patterns.
    fn bind_params(&mut self, params: &[Param]) -> Result<(), CompileError> {
        if params.is_empty() {
            return Ok(());
        }
        // `#args = Op::RestArgs(0)`.
        self.emit(Op::PushInt(0));
        self.emit(Op::RestArgs);
        let args_nc = self.binding_name_const("#args")?;
        self.emit(Op::StoreName(args_nc));
        for (i, p) in params.iter().enumerate() {
            let idx = self.intern_int(i as i32)?;
            match &p.pat {
                Pat::Rest(inner, _) => {
                    self.emit(Op::LoadName(args_nc));
                    self.emit(Op::LoadConst(idx));
                    self.call_intrinsic("arr_rest", 2)?;
                    self.compile_pat_bind(inner, VarKind::Var)?;
                }
                _ => {
                    self.emit(Op::LoadName(args_nc));
                    self.emit(Op::LoadConst(idx));
                    self.emit(Op::GetElem);
                    self.compile_pat_bind(&p.pat, VarKind::Var)?;
                }
            }
        }
        Ok(())
    }

    fn binding_name_const(&mut self, name: &str) -> Result<u16, CompileError> {
        let cur = self.fns.len() - 1;
        match self.lookup(name) {
            Some((decl_fn, id)) if decl_fn == cur => {
                Ok(self.fns[cur].bindings[id as usize].name_const)
            }
            _ => Err(CompileError::InvalidLhs),
        }
    }

    fn code_len(&self) -> usize {
        self.fns.last().expect("no active function").code.len()
    }

    fn protected_len(&self) -> usize {
        self.fns.last().expect("no active function").protected.len()
    }

    /// Patch a jump operand position to an absolute target.
    fn patch_to(&mut self, operand_pos: usize, target: usize) -> Result<(), CompileError> {
        let off = target as isize - (operand_pos as isize + 2);
        if !(i16::MIN as isize..=i16::MAX as isize).contains(&off) {
            return Err(CompileError::JumpTooFar);
        }
        let bytes = (off as i16).to_le_bytes();
        let code = self.code();
        code[operand_pos] = bytes[0];
        code[operand_pos + 1] = bytes[1];
        Ok(())
    }

    fn emit_jump_to(&mut self, target: usize) -> Result<(), CompileError> {
        let pos = self.emit_jump(Op::Jump(0));
        self.patch_to(pos, target)
    }

    // -- declaration walk -------------------------------------------------

    /// Declare every binding in `stmts` (recursively) before the prologue is
    /// emitted. This walk pushes and pops real block scopes: each popped
    /// scope's name map is saved in pre-order into the current function's
    /// `saved` list so codegen can replay it with
    /// [`Compiler::push_block_scope`]. Function declarations are compiled
    /// to templates immediately — their bodies need the enclosing block
    /// scopes for name resolution — so the prologue can reference them.
    /// `var`s (and imports) land in the function scope; `let`/`const` and
    /// class names land in the innermost block scope.
    ///
    /// Before the source-order walk, [`Compiler::predeclare_lexicals`]
    /// declares this level's lexical and import bindings, so a function
    /// compiled earlier in source order still resolves a later sibling
    /// `let`/`const` as an upvalue (with its TDZ flag) instead of a global.
    fn declare_stmts(&mut self, stmts: &[Stmt]) -> Result<(), CompileError> {
        self.predeclare_lexicals(stmts)?;
        for s in stmts {
            self.declare_stmt(s)?;
        }
        Ok(())
    }

    /// Pre-declare this statement level's lexical (`let`/`const`/`using`/
    /// class) and import bindings before the source-order declare walk.
    ///
    /// The declare walk compiles function declarations to templates
    /// eagerly, in source order, so a function defined textually *before*
    /// a later sibling `let`/`const` would otherwise resolve that name as
    /// a global and throw `ReferenceError: X is not defined` at runtime —
    /// even when the function is called after the initializer ran. Per
    /// spec the binding exists (in TDZ) for the whole block, so every
    /// level-local lexical is declared up front: the walk's re-declaration
    /// is idempotent ([`Compiler::declare_in`] returns the existing id for
    /// same-kind collisions), and TDZ is still enforced at runtime by the
    /// `#tdz#` flag, which the prologue initializes to falsy and the
    /// declaration's initializer sets.
    ///
    /// Only the current level is scanned: nested blocks, branches, and
    /// loops get their own pre-declaration when the walk reaches them, so
    /// a lexical scoped to a nested block never leaks into an enclosing
    /// one. `var`s and function declarations are already pre-declared by
    /// [`Compiler::hoist_names`].
    fn predeclare_lexicals(&mut self, stmts: &[Stmt]) -> Result<(), CompileError> {
        for s in stmts {
            self.predeclare_stmt(s)?;
        }
        Ok(())
    }

    /// Pre-declare one statement's level-local lexical/import bindings; see
    /// [`Compiler::predeclare_lexicals`]. Mirrors the corresponding arms of
    /// [`Compiler::declare_stmt_inner`].
    fn predeclare_stmt(&mut self, s: &Stmt) -> Result<(), CompileError> {
        match s {
            Stmt::VarDecl { kind, decls, .. } if *kind != VarKind::Var => {
                for d in decls {
                    self.declare_pat_current(&d.id, *kind)?;
                }
            }
            Stmt::ClassDecl { name: Some(n), .. } => {
                self.declare_current(n, BindingKind::Let)?;
            }
            Stmt::Import(d) => {
                if let Some((n, _)) = &d.default {
                    self.declare_function_scope(n, BindingKind::Var)?;
                }
                if let Some((n, _)) = &d.namespace {
                    self.declare_function_scope(n, BindingKind::Var)?;
                }
                for s in &d.named {
                    self.declare_function_scope(&s.local, BindingKind::Var)?;
                }
            }
            Stmt::Export(ExportDecl::Decl { inner, .. }) => self.predeclare_stmt(inner)?,
            _ => {}
        }
        Ok(())
    }

    /// Pre-declare every hoisted binding in `stmts` — `var` declarators and
    /// function declarations — in the current function scope, recursing
    /// through blocks, branches, loops, `try`, `switch`, and labels, but
    /// never into a nested function, class, or static-block body (those get
    /// their own hoist pass when their template compiles).
    ///
    /// This must run before [`Compiler::declare_stmts`]: the declare walk
    /// compiles each function declaration to its template eagerly, in
    /// source order, so a nested function that references a *later*
    /// sibling (`function f() { return g(); } function g() {}`) would
    /// otherwise resolve `g` as a global and throw `ReferenceError` at
    /// runtime. Pre-declaring makes every hoisted name visible to every
    /// template compiled during the walk, matching spec hoisting
    /// (`FunctionDeclarationInstantiation`). `declare_in` is idempotent
    /// for same-kind collisions (and a function declaration still wins
    /// over an earlier `var`), so the declare walk re-declaring these
    /// names changes nothing.
    ///
    /// The walk pushes no block scopes and reserves no
    /// [`FnState::saved`](crate::compiler::FnState::saved) slots — only
    /// function-scope names are declared — so the declare walk's pre-order
    /// slot layout is unaffected. `let`/`const`/class names are
    /// intentionally skipped here: they are block-scoped, so this
    /// scope-blind walk cannot place them correctly; instead
    /// [`Compiler::predeclare_lexicals`] declares them at each level at
    /// the start of the declare walk, before any function template at
    /// that level is compiled.
    fn hoist_names(&mut self, stmts: &[Stmt]) -> Result<(), CompileError> {
        for s in stmts {
            self.hoist_stmt_names(s)?;
        }
        Ok(())
    }

    fn hoist_stmt_names(&mut self, s: &Stmt) -> Result<(), CompileError> {
        match s {
            Stmt::FunctionDecl { name, .. } => {
                self.declare_function(name)?;
            }
            Stmt::VarDecl {
                kind: VarKind::Var,
                decls,
                ..
            } => {
                for d in decls {
                    self.declare_pat_top(&d.id, VarKind::Var)?;
                }
            }
            Stmt::VarDecl { .. } => {}
            Stmt::Block(ss) => self.hoist_names(ss)?,
            Stmt::If { then_b, else_b, .. } => {
                self.hoist_stmt_names(then_b)?;
                if let Some(e) = else_b {
                    self.hoist_stmt_names(e)?;
                }
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                self.hoist_stmt_names(body)?;
            }
            Stmt::For { init, body, .. } => {
                if let Some(ForInit::Var(VarKind::Var, ds)) = init {
                    for d in ds {
                        self.declare_pat_top(&d.id, VarKind::Var)?;
                    }
                }
                self.hoist_stmt_names(body)?;
            }
            Stmt::ForInOf {
                kind, lhs, body, ..
            } => {
                if *kind == VarKind::Var
                    && let ForLhs::Pat(p) = lhs
                {
                    self.declare_pat_top(p, VarKind::Var)?;
                }
                self.hoist_stmt_names(body)?;
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    self.hoist_names(&c.body)?;
                }
            }
            Stmt::Try {
                block,
                handler,
                finalizer,
            } => {
                self.hoist_names(block)?;
                if let Some(c) = handler {
                    self.hoist_names(&c.body)?;
                }
                if let Some(f) = finalizer {
                    self.hoist_names(f)?;
                }
            }
            Stmt::Labeled { body, .. } => self.hoist_stmt_names(body)?,
            Stmt::With { body, .. } => self.hoist_stmt_names(body)?,
            Stmt::Export(ExportDecl::Decl { inner, .. }) => self.hoist_stmt_names(inner)?,
            // Everything else declares nothing hoistable at this level:
            // nested functions/classes/static blocks hoist within their own
            // bodies, `let`/`const` stay block-scoped, and imports are
            // declared by the declare walk.
            _ => {}
        }
        Ok(())
    }

    /// Push a block scope in the declare walk, reserving a pre-order slot
    /// in [`FnState::saved`] for its name map.
    fn push_declare_scope(&mut self) {
        let f = self.fns.last_mut().expect("no active function");
        let slot = f.saved.len();
        f.saved.push(Rc::new(HashMap::new()));
        self.open_declare.push(slot);
        self.push_scope(false);
    }

    /// Pop a declare-walk block scope, storing its name map in the reserved slot.
    fn pop_declare_scope(&mut self) {
        let slot = self.open_declare.pop().expect("declare scope underflow");
        let idx = self.scopes.len() - 1;
        debug_assert!(!self.scopes[idx].function, "popping a function scope");
        let names = std::mem::take(&mut self.scopes[idx].names);
        self.scopes.pop();
        self.fns.last_mut().expect("no active function").saved[slot] = names;
    }

    /// Replay the next declare-walk block scope during codegen: share its
    /// saved name map (in the same pre-order the declare walk pushed) via
    /// `Rc` and push it as the innermost block scope. Sharing (rather than
    /// cloning) is sound because codegen never mutates a replayed map, and
    /// it lets a scope be replayed multiple times (e.g., a `finally` body
    /// compiled on both the normal and abrupt paths); the caller resets
    /// the cursor to reuse slots.
    fn push_block_scope(&mut self) {
        let cursor = self.fns.last().expect("no active function").scope_cursor;
        let names = Rc::clone(
            self.fns
                .last()
                .expect("no active function")
                .saved
                .get(cursor)
                .expect("declare walk / codegen block-scope mismatch"),
        );
        self.fns
            .last_mut()
            .expect("no active function")
            .scope_cursor = cursor + 1;
        self.push_scope(false);
        self.scopes.last_mut().expect("just pushed").names = names;
    }

    /// Push a saved declare-walk scope by slot index, sharing its name map
    /// via `Rc` so the same scope can be replayed multiple times (e.g., a
    /// `finally` body compiled on both the normal and abrupt paths). Does
    /// not advance the cursor; the caller is responsible for advancing it
    /// past all slots the declare walk reserved.
    fn push_saved_scope(&mut self, slot: usize) {
        let names = Rc::clone(
            self.fns
                .last()
                .expect("no active function")
                .saved
                .get(slot)
                .expect("declare walk / codegen block-scope mismatch"),
        );
        self.push_scope(false);
        self.scopes.last_mut().expect("just pushed").names = names;
    }

    fn declare_stmt(&mut self, s: &Stmt) -> Result<(), CompileError> {
        self.guarded(|this| this.declare_stmt_inner(s))
    }

    fn declare_stmt_inner(&mut self, s: &Stmt) -> Result<(), CompileError> {
        match s {
            Stmt::FunctionDecl {
                name,
                params,
                body,
                is_async,
                is_generator,
                strict,
                ..
            } => {
                if !self.fns.last().expect("no active function").no_vars {
                    // Declare the name in the enclosing scope *before*
                    // compiling the template so the body's own references to
                    // its name resolve as an upvalue capture of the outer
                    // binding instead of falling back to `Resolved::Global`.
                    // `declare_in` is idempotent for same-kind collisions
                    // (and a Function kind wins over an earlier `var`), so
                    // pre-declaring keeps the var/function merge rules.
                    let id = self.declare_function(name)?;
                    let t = self.compile_function_template(
                        name.clone(),
                        params,
                        body,
                        false,
                        *is_async,
                        *is_generator,
                        *strict,
                        false,
                        None,
                        None,
                    )?;
                    let idx = self.intern_template(t)?;
                    self.fns.last_mut().expect("no active function").bindings[id as usize]
                        .func_const = Some(idx);
                } else {
                    // Per-iteration extracted body: the name was hoisted to
                    // the enclosing function by `declare_body_vars` (so it
                    // keeps resolving to the shared binding), but the
                    // template is compiled here, where this level's
                    // lexicals are already pre-declared by `declare_stmts`.
                    // Compiling it in the enclosing scope instead would
                    // resolve those lexicals as globals (`ReferenceError`
                    // at runtime); here they become upvalues capturing the
                    // per-iteration environment. The template is
                    // instantiated once per iteration by
                    // `compile_extracted_body`.
                    let t = self.compile_function_template(
                        name.clone(),
                        params,
                        body,
                        false,
                        *is_async,
                        *is_generator,
                        *strict,
                        false,
                        None,
                        None,
                    )?;
                    let idx = self.intern_template(t)?;
                    self.fns
                        .last_mut()
                        .expect("no active function")
                        .extracted_fn_inits
                        .push((name.clone(), idx));
                }
            }
            Stmt::VarDecl { kind, decls, .. } => {
                let no_vars = self.fns.last().expect("no active function").no_vars;
                for d in decls {
                    if *kind == VarKind::Var {
                        // `var`s belong to the enclosing function (see
                        // `declare_body_vars`); the extracted body must not
                        // redeclare them.
                        if !no_vars {
                            self.declare_pat_top(&d.id, *kind)?;
                        }
                    } else {
                        self.declare_pat_current(&d.id, *kind)?;
                    }
                }
            }
            // Class names are lexical: they live in the innermost scope.
            // Static blocks compile to their own wrapper closures (see
            // `compile_static_block_wrapper`), which run their own declare
            // walk, so the enclosing declare walk skips them.
            Stmt::ClassDecl { name: Some(n), .. } => {
                self.declare_current(n, BindingKind::Let)?;
            }
            Stmt::ClassDecl { name: None, .. } => {}
            Stmt::Block(ss) => {
                self.push_declare_scope();
                let r = self.declare_stmts(ss);
                self.pop_declare_scope();
                r?;
            }
            Stmt::If { then_b, else_b, .. } => {
                self.declare_stmt(then_b)?;
                if let Some(e) = else_b {
                    self.declare_stmt(e)?;
                }
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => self.declare_stmt(body)?,
            Stmt::For {
                init,
                cond,
                update,
                body,
            } => {
                // The loop head gets its own scope so `for (let i ...)`
                // neither leaks nor collides with the enclosing scope.
                self.push_declare_scope();
                if let Some(ForInit::Var(k, ds)) = init {
                    for d in ds {
                        if *k == VarKind::Var {
                            self.declare_pat_top(&d.id, *k)?;
                        } else {
                            self.declare_pat_current(&d.id, *k)?;
                        }
                    }
                }
                // With per-iteration extraction the body is declare-walked
                // inside the per-iteration arrow; here only its `var`s and
                // function declarations are hoisted to the enclosing
                // function (see `declare_body_vars`).
                let f = self.fns.last().expect("no active function");
                let extract = for_needs_extraction(
                    f.is_async || f.is_generator,
                    init.as_ref(),
                    cond.as_ref(),
                    update.as_ref(),
                    body,
                );
                let r = if extract {
                    self.declare_body_vars(body)
                } else {
                    self.declare_stmt(body)
                };
                self.pop_declare_scope();
                r?;
            }
            Stmt::ForInOf {
                kind, lhs, body, ..
            } => {
                self.push_declare_scope();
                match lhs {
                    ForLhs::Pat(p) => {
                        if *kind == VarKind::Var {
                            self.declare_pat_top(p, *kind)?;
                        } else {
                            self.declare_pat_current(p, *kind)?;
                        }
                    }
                    ForLhs::Ident(n, _) => {
                        if *kind == VarKind::Var {
                            // Bare `for (var x in/of ...)`: `var` hoists to
                            // the function scope like any other `var`
                            // declaration. Without this, strict-mode
                            // stores throw `ReferenceError: assignment to
                            // undeclared variable`.
                            self.declare_pat_top(
                                &Pat::Ident(n.clone(), Span { start: 0, end: 0 }),
                                VarKind::Var,
                            )?;
                        } else {
                            self.declare_current(n, binding_kind_of(*kind))?;
                        }
                    }
                    _ => {}
                }
                let f = self.fns.last().expect("no active function");
                let extract = for_in_of_needs_extraction(f.is_async || f.is_generator, *kind, body);
                let r = if extract {
                    self.declare_body_vars(body)
                } else {
                    self.declare_stmt(body)
                };
                self.pop_declare_scope();
                r?;
            }
            Stmt::Switch { cases, .. } => {
                // One scope for all case clauses (they share a lexical
                // environment); nested blocks push their own.
                self.push_declare_scope();
                let mut r = Ok(());
                for c in cases {
                    r = self.declare_stmts(&c.body);
                    if r.is_err() {
                        break;
                    }
                }
                self.pop_declare_scope();
                r?;
            }
            Stmt::Try {
                block,
                handler,
                finalizer,
            } => {
                // The try-statement scope owns the finalizer's lexicals, so
                // every inline copy of the finalizer (normal path, abrupt
                // paths via `unwind_protected`) resolves through the live
                // scope stack.
                self.push_declare_scope();
                self.push_declare_scope();
                let mut r = self.declare_stmts(block);
                self.pop_declare_scope();
                if r.is_ok() {
                    if let Some(f) = finalizer {
                        // Record where the finalizer's scopes begin: an
                        // abrupt exit inside the try block inlines the
                        // finalizer before the try block has finished
                        // compiling, so `inline_finally` cannot rely on the
                        // live cursor to find them.
                        let key = f.as_ptr();
                        let base = self.fns.last().expect("no active function").saved.len();
                        self.fns
                            .last_mut()
                            .expect("no active function")
                            .finally_bases
                            .insert(key, base);
                        r = self.declare_stmts(f);
                    }
                    if r.is_ok()
                        && let Some(c) = handler
                    {
                        // The catch parameter gets its own scope so it
                        // shadows (rather than collides with) an outer
                        // binding of the same name; the body nests inside.
                        self.push_declare_scope();
                        if let Some(p) = &c.param {
                            r = self.declare_pat_current(&p.pat, VarKind::Let);
                        }
                        if r.is_ok() {
                            self.push_declare_scope();
                            r = self.declare_stmts(&c.body);
                            self.pop_declare_scope();
                        }
                        self.pop_declare_scope();
                    }
                }
                self.pop_declare_scope();
                r?;
            }
            Stmt::Labeled { body, .. } => self.declare_stmt(body)?,
            Stmt::With { body, .. } => self.declare_stmt(body)?,
            Stmt::Export(e) => match e {
                ExportDecl::Decl { inner, .. } => self.declare_stmt(inner)?,
                ExportDecl::Default { .. } => {
                    self.declare_function_scope("#default_export", BindingKind::Var)?;
                }
                _ => {}
            },
            Stmt::Import(d) => {
                if let Some((n, _)) = &d.default {
                    self.declare_function_scope(n, BindingKind::Var)?;
                }
                if let Some((n, _)) = &d.namespace {
                    self.declare_function_scope(n, BindingKind::Var)?;
                }
                for s in &d.named {
                    self.declare_function_scope(&s.local, BindingKind::Var)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn declare_function_scope(
        &mut self,
        name: &str,
        kind: BindingKind,
    ) -> Result<u16, CompileError> {
        let idx = self.function_scope_idx();
        self.declare_in(idx, name, kind)
    }

    /// Declare a per-iteration extracted loop body's `var`s and function
    /// declarations in the enclosing function scope. Function-declaration
    /// *names* are hoisted here (Annex B var-like semantics), but their
    /// templates are compiled later, during the extracted arrow's own
    /// declare walk (see `declare_stmt_inner`): compiling them here would
    /// resolve the body's per-iteration lexicals as globals, since the
    /// arrow's predeclare has not run yet. `let`/`const`/class bindings
    /// are per-iteration and belong to the arrow (declared by its own walk
    /// with `no_vars` set), so they are skipped here. No block scopes are
    /// pushed: the outer codegen only replays the loop-head scope for an
    /// extracted loop.
    fn declare_body_vars(&mut self, s: &Stmt) -> Result<(), CompileError> {
        match s {
            Stmt::VarDecl {
                kind: VarKind::Var,
                decls,
                ..
            } => {
                for d in decls {
                    self.declare_pat_top(&d.id, VarKind::Var)?;
                }
            }
            Stmt::VarDecl { .. } => {}
            Stmt::FunctionDecl { name, .. } => {
                // Hoist the name only; the template is compiled during the
                // extracted arrow's declare walk, where the body's lexicals
                // are pre-declared so the template captures the
                // per-iteration environment as upvalues. `declare_in` is
                // idempotent for same-kind collisions.
                self.declare_function(name)?;
            }
            Stmt::Block(ss) => {
                for x in ss {
                    self.declare_body_vars(x)?;
                }
            }
            Stmt::If { then_b, else_b, .. } => {
                self.declare_body_vars(then_b)?;
                if let Some(e) = else_b {
                    self.declare_body_vars(e)?;
                }
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                self.declare_body_vars(body)?;
            }
            Stmt::For { init, body, .. } => {
                if let Some(ForInit::Var(VarKind::Var, ds)) = init {
                    for d in ds {
                        self.declare_pat_top(&d.id, VarKind::Var)?;
                    }
                }
                self.declare_body_vars(body)?;
            }
            Stmt::ForInOf {
                kind, lhs, body, ..
            } => {
                if *kind == VarKind::Var
                    && let ForLhs::Pat(p) = lhs
                {
                    self.declare_pat_top(p, VarKind::Var)?;
                }
                self.declare_body_vars(body)?;
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    for x in &c.body {
                        self.declare_body_vars(x)?;
                    }
                }
            }
            Stmt::Try {
                block,
                handler,
                finalizer,
            } => {
                for x in block {
                    self.declare_body_vars(x)?;
                }
                if let Some(c) = handler {
                    for x in &c.body {
                        self.declare_body_vars(x)?;
                    }
                }
                if let Some(f) = finalizer {
                    for x in f {
                        self.declare_body_vars(x)?;
                    }
                }
            }
            Stmt::Labeled { body, .. } => self.declare_body_vars(body)?,
            Stmt::With { body, .. } => self.declare_body_vars(body)?,
            Stmt::Export(ExportDecl::Decl { inner, .. }) => self.declare_body_vars(inner)?,
            _ => {}
        }
        Ok(())
    }

    /// Declare a pattern's identifiers in the function scope (for `var`).
    fn declare_pat_top(&mut self, pat: &Pat, kind: VarKind) -> Result<(), CompileError> {
        let bk = binding_kind_of(kind);
        self.declare_pat_in(pat, bk, true)
    }

    /// Declare a pattern's identifiers in the innermost block scope (for `let`/`const`).
    fn declare_pat_current(&mut self, pat: &Pat, kind: VarKind) -> Result<(), CompileError> {
        let bk = binding_kind_of(kind);
        self.declare_pat_in(pat, bk, false)
    }

    fn declare_pat_in(
        &mut self,
        pat: &Pat,
        kind: BindingKind,
        function_scope: bool,
    ) -> Result<(), CompileError> {
        let idx = if function_scope {
            self.function_scope_idx()
        } else {
            self.scopes.len() - 1
        };
        match pat {
            Pat::Ident(name, _) => {
                self.declare_in(idx, name, kind)?;
            }
            Pat::Array(elems, _) => {
                for el in elems {
                    if let Some(ArrayPatElem::Pat(p)) = el {
                        self.declare_pat_inner(p, kind, idx)?;
                    }
                }
            }
            Pat::Object(props, _) => {
                for pr in props {
                    match pr {
                        ObjectPatProp::KeyValue { value, .. } => {
                            self.declare_pat_inner(value, kind, idx)?;
                        }
                        ObjectPatProp::Shorthand { name, .. } => {
                            self.declare_in(idx, name, kind)?;
                        }
                        ObjectPatProp::Rest(p, _) => self.declare_pat_inner(p, kind, idx)?,
                    }
                }
            }
            Pat::Rest(p, _) => self.declare_pat_inner(p, kind, idx)?,
            Pat::Assign(p, _, _) => self.declare_pat_inner(p, kind, idx)?,
            // Member targets declare nothing (they only arise from
            // assignment patterns, never binding declarations).
            Pat::Member(..) => {}
        }
        Ok(())
    }

    fn declare_pat_inner(
        &mut self,
        pat: &Pat,
        kind: BindingKind,
        idx: usize,
    ) -> Result<(), CompileError> {
        self.guarded(|s| s.declare_pat_inner_guarded(pat, kind, idx))
    }

    fn declare_pat_inner_guarded(
        &mut self,
        pat: &Pat,
        kind: BindingKind,
        idx: usize,
    ) -> Result<(), CompileError> {
        match pat {
            Pat::Ident(name, _) => {
                self.declare_in(idx, name, kind)?;
            }
            Pat::Array(elems, _) => {
                for el in elems {
                    if let Some(ArrayPatElem::Pat(p)) = el {
                        self.declare_pat_inner(p, kind, idx)?;
                    }
                }
            }
            Pat::Object(props, _) => {
                for pr in props {
                    match pr {
                        ObjectPatProp::KeyValue { value, .. } => {
                            self.declare_pat_inner(value, kind, idx)?;
                        }
                        ObjectPatProp::Shorthand { name, .. } => {
                            self.declare_in(idx, name, kind)?;
                        }
                        ObjectPatProp::Rest(p, _) => self.declare_pat_inner(p, kind, idx)?,
                    }
                }
            }
            Pat::Rest(p, _) => self.declare_pat_inner(p, kind, idx)?,
            Pat::Assign(p, _, _) => self.declare_pat_inner(p, kind, idx)?,
            // Member targets declare nothing (they only arise from
            // assignment patterns, never binding declarations).
            Pat::Member(..) => {}
        }
        Ok(())
    }

    /// Compile a function (or constructor) to a template. `ctor` carries
    /// class-constructor context: field initializers run at the end of the
    /// body, in textual order.
    ///
    /// `strict` is the function's final strictness, computed by the parser
    /// from the raw source (inherited strictness plus a `"use strict"`
    /// directive in the body's prologue; escaped spellings do not count).
    #[allow(clippy::too_many_arguments)]
    fn compile_function_template(
        &mut self,
        name: String,
        params: &[Param],
        body: &[Stmt],
        is_arrow: bool,
        is_async: bool,
        is_generator: bool,
        strict: bool,
        is_method: bool,
        ctor: Option<CtorInfo>,
        // Named function expression's own name (`Some` only for
        // `Expr::Function` with a name): declared as an immutable
        // [`BindingKind::SelfName`] binding in the function's own scope
        // before params and hoisted declarations, so the body's references
        // resolve to it and later declarations shadow it.
        self_name: Option<&str>,
    ) -> Result<FuncTemplate, CompileError> {
        let arity = params.len().min(u16::MAX as usize) as u16;
        self.push_function(
            name,
            arity,
            is_arrow,
            is_async,
            is_generator,
            strict,
            is_method,
        );
        if let Some(sname) = self_name {
            let scope = self.function_scope_idx();
            self.declare_in(scope, sname, BindingKind::SelfName)?;
        }
        // If compilation fails, pop the pushed function so the caller's
        // error unwinding (e.g., declare-walk scope pops) sees consistent
        // state instead of panicking on a stale function entry.
        match self.compile_function_body(params, body, ctor) {
            Ok(()) => Ok(self.pop_function()),
            Err(e) => {
                let f = self.fns.pop().expect("no active function");
                self.pop_scopes_to(f.scope_base);
                Err(e)
            }
        }
    }

    /// Body of [`Compiler::compile_function_template`] after the function
    /// state is pushed: declare params, declare hoisted names, emit the
    /// prologue, bind params, and compile the statements.
    fn compile_function_body(
        &mut self,
        params: &[Param],
        body: &[Stmt],
        ctor: Option<CtorInfo>,
    ) -> Result<(), CompileError> {
        // Class context (for `super`) is inherited from the enclosing
        // function state by push_function.
        if let Some(ci) = &ctor {
            let f = self.fns.last_mut().expect("no active function");
            f.ctor_ctx = Some(CtorCtx {
                fields: ci.fields.clone(),
                brand: ci.brand,
                has_private: ci.has_private,
            });
            // Derived constructors start with uninitialized `this`; the
            // VM reads this flag (see `FuncTemplate::is_derived_ctor`).
            f.is_derived_ctor = ci.is_derived;
        }
        // Parameters are function-scoped `var`-like bindings. Declaring them
        // before the body lets body-level `var`s and function declarations
        // merge with the same binding instead of resolving to globals.
        //
        // Sloppy mapped `arguments`: a sloppy non-arrow function whose
        // parameter list is simple (every parameter a plain identifier)
        // gets a *mapped* arguments object — `arguments[i]` aliases
        // parameter `i` (spec: `CreateMappedArgumentsObject`). Record each
        // parameter's binding id now, before body declarations can shadow
        // the names; the VM force-captures these slots when it builds the
        // arguments object. Duplicate parameter names merge onto one
        // binding in `declare_in`; only the last duplicate keeps its
        // mapping (filtered right-to-left below), as the spec requires.
        let want_mapped = {
            let f = self.fns.last().expect("no active function");
            !f.strict && !f.is_arrow && params.iter().all(|p| matches!(p.pat, Pat::Ident(..)))
        };
        let mut mapped_slots: Option<Vec<Option<u16>>> = want_mapped.then(Vec::new);
        for p in params {
            self.declare_pat_top(&p.pat, VarKind::Var)?;
            if let (Some(slots), Pat::Ident(name, _)) = (mapped_slots.as_mut(), &p.pat) {
                let scope = self.function_scope_idx();
                let slot = self.scopes[scope].names.get(name.as_str()).copied();
                slots.push(slot);
            }
        }
        if let Some(slots) = mapped_slots {
            // Duplicate parameter names merge onto one binding in
            // `declare_in`; the spec maps only the *last* duplicate, so
            // null out earlier occurrences right-to-left.
            let mut slots = slots;
            let mut seen = Vec::new();
            for slot in slots.iter_mut().rev() {
                if let Some(id) = *slot {
                    if seen.contains(&id) {
                        *slot = None;
                    } else {
                        seen.push(id);
                    }
                }
            }
            self.fns
                .last_mut()
                .expect("no active function")
                .mapped_arg_slots = Some(slots);
        }
        // Hoist every `var` and function declaration in the body into the
        // function scope before the declare walk: the walk compiles
        // function templates eagerly in source order, and a template that
        // references a later sibling must resolve it as an upvalue, not a
        // global (see `hoist_names`).
        self.hoist_names(body)?;
        self.declare_stmts(body)?;
        // `arguments`: every non-arrow function gets an arguments object
        // unless the name is already bound in the function scope (param,
        // `var`/`function` declaration). Arrows inherit the outer
        // `arguments`.
        let f = self.fns.last().expect("no active function");
        let make_arguments =
            !f.is_arrow && !self.scopes[f.scope_base].names.contains_key("arguments");
        if make_arguments {
            let scope = self.fns.last().expect("no active function").scope_base;
            self.declare_in(scope, "arguments", BindingKind::Var)?;
        }
        self.emit_prologue()?;
        self.bind_params(params)?;
        if make_arguments {
            // [args]; store into the declared binding.
            self.emit(Op::NewArguments);
            self.compile_store("arguments")?;
        }
        self.compile_stmts(body)?;
        if self
            .fns
            .last()
            .expect("no active function")
            .ctor_ctx
            .is_some()
        {
            let ctx = self
                .fns
                .last_mut()
                .expect("no active function")
                .ctor_ctx
                .take()
                .expect("checked");
            if ctx.has_private {
                // Brand the fresh instance before any field initializer
                // runs, so private field installs and brand checks inside
                // inits see it.
                self.compile_this()?;
                self.emit(Op::AddPrivateBrand(ctx.brand));
                self.emit(Op::Pop);
            }
            self.emit_field_inits(&ctx.fields, ctx.brand)?;
        }
        // Constructors return `this` via the VM's New protocol; every other
        // function falls off the end with `undefined`.
        self.emit(Op::PushUndefined);
        self.emit(Op::Return);
        Ok(())
    }
}

/// Constructor context passed to [`Compiler::compile_function_template`].
#[derive(Clone)]
struct CtorInfo {
    fields: Vec<(FieldKey, Option<Expr>)>,
    super_name: Option<String>,
    /// True when the class has a superclass: the constructor is derived,
    /// so the VM must start `#this` uninitialized (TDZ until `super()`).
    is_derived: bool,
    /// The class's private brand, installed on each fresh instance before
    /// field initialization.
    brand: u16,
    /// Whether the class declares any private member: when false, no
    /// brand is installed (no brand check can name this class).
    has_private: bool,
}

fn binding_kind_of(kind: VarKind) -> BindingKind {
    match kind {
        VarKind::Var => BindingKind::Var,
        VarKind::Let => BindingKind::Let,
        // `using` bindings are immutable like `const`. (The parser
        // desugars `using` before codegen, so these arms are defensive.)
        VarKind::Const | VarKind::Using | VarKind::AwaitUsing => BindingKind::Const,
    }
}

// -- per-iteration loop analysis --------------------------------------------

/// Maximum recursion depth for the closure-presence scan below. The scan
/// runs over parser-produced ASTs whose guarded constructs are already
/// depth-capped at [`MAX_DEPTH`] — but left-deep binary/member/call chains
/// are built iteratively by the parser and can be arbitrarily deep, so the
/// scan enforces its own bound instead of overflowing the host stack. On
/// overflow it conservatively reports "present": that only triggers
/// per-iteration body extraction, never a miscompile.
const MAX_SCAN_DEPTH: u32 = 256;

/// True if the expression subtree contains a function, arrow, or class
/// expression. Any of those may capture a per-iteration `let`/`const`, so
/// this is a conservative trigger for per-iteration body extraction.
fn has_closure_expr(e: &Expr) -> bool {
    scan_closure_expr(e, 0)
}

fn scan_closure_expr(e: &Expr, depth: u32) -> bool {
    if depth > MAX_SCAN_DEPTH {
        return true;
    }
    let d = depth + 1;
    match e {
        Expr::Function { .. } | Expr::Arrow { .. } | Expr::Class { .. } => true,
        Expr::Ident(..)
        | Expr::PrivateIdent(..)
        | Expr::Lit(..)
        | Expr::This(..)
        | Expr::Super(..)
        | Expr::NewTarget(..)
        | Expr::ImportMeta(..)
        | Expr::Regex { .. } => false,
        Expr::Array(elems, _) => elems.iter().flatten().any(|x| scan_closure_expr(x, d)),
        Expr::Object(props, _) => props.iter().any(|p| match p {
            PropDef::KeyValue { key, value, .. } => {
                scan_closure_key(key, d) || scan_closure_expr(value, d)
            }
            PropDef::Shorthand { .. } => false,
            PropDef::Spread(x, _) => scan_closure_expr(x, d),
            // Methods are closures.
            PropDef::Method { .. } | PropDef::Getter { .. } | PropDef::Setter { .. } => true,
        }),
        Expr::Call { callee, args, .. } => {
            scan_closure_expr(callee, d) || args.iter().any(|x| scan_closure_expr(x, d))
        }
        Expr::New { callee, args, .. } => {
            scan_closure_expr(callee, d) || args.iter().any(|x| scan_closure_expr(x, d))
        }
        Expr::Member { obj, prop, .. } => scan_closure_expr(obj, d) || scan_closure_expr(prop, d),
        Expr::Unary { arg, .. } | Expr::Spread(arg, _) | Expr::Await(arg, _) => {
            scan_closure_expr(arg, d)
        }
        Expr::Update { arg, .. } => scan_closure_expr(arg, d),
        Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
            scan_closure_expr(left, d) || scan_closure_expr(right, d)
        }
        Expr::Assign { left, right, .. } => {
            scan_closure_expr(left, d) || scan_closure_expr(right, d)
        }
        Expr::Conditional {
            cond,
            then_e,
            else_e,
            ..
        } => {
            scan_closure_expr(cond, d)
                || scan_closure_expr(then_e, d)
                || scan_closure_expr(else_e, d)
        }
        Expr::Sequence(es, _) => es.iter().any(|x| scan_closure_expr(x, d)),
        Expr::Template { parts, .. } => parts.iter().any(|p| match p {
            TemplatePart::Str { .. } => false,
            TemplatePart::Expr(x) => scan_closure_expr(x, d),
        }),
        Expr::TaggedTemplate { tag, parts, .. } => {
            scan_closure_expr(tag, d)
                || parts.iter().any(|p| match p {
                    TemplatePart::Str { .. } => false,
                    TemplatePart::Expr(x) => scan_closure_expr(x, d),
                })
        }
        Expr::Yield { arg, .. } => arg.as_ref().is_some_and(|x| scan_closure_expr(x, d)),
    }
}

fn scan_closure_key(k: &PropKey, depth: u32) -> bool {
    matches!(k, PropKey::Computed(e) if scan_closure_expr(e, depth))
}

/// True if the pattern's default values contain a closure.
fn scan_closure_pat(p: &Pat, depth: u32) -> bool {
    if depth > MAX_SCAN_DEPTH {
        return true;
    }
    let d = depth + 1;
    match p {
        Pat::Ident(..) => false,
        Pat::Array(elems, _) => elems.iter().flatten().any(|el| match el {
            ArrayPatElem::Pat(q) => scan_closure_pat(q, d),
        }),
        Pat::Object(props, _) => props.iter().any(|pr| match pr {
            ObjectPatProp::KeyValue { key, value, .. } => {
                scan_closure_key(key, d) || scan_closure_pat(value, d)
            }
            ObjectPatProp::Shorthand { .. } => false,
            ObjectPatProp::Rest(q, _) => scan_closure_pat(q, d),
        }),
        Pat::Rest(q, _) => scan_closure_pat(q, d),
        Pat::Assign(q, x, _) => scan_closure_pat(q, d) || scan_closure_expr(x, d),
        // A member target's base/key are full expressions and may contain
        // closures (e.g. `[(function(){})().x] = v`).
        Pat::Member(e) => scan_closure_expr(e, d),
    }
}

/// True if the statement subtree contains a closure.
fn has_closure_stmt(s: &Stmt) -> bool {
    scan_closure_stmt(s, 0)
}

/// True if the expression subtree contains a *named* class expression
/// (not descending into nested function/arrow bodies: those are separate
/// compilation units whose class evaluations already get fresh name
/// bindings per call). Each evaluation of a named class expression binds
/// its name in a fresh inner scope, so a loop that evaluates one needs
/// per-iteration body extraction for the self-name to stay fresh.
fn has_named_class_expr(e: &Expr) -> bool {
    scan_named_class_expr(e, 0)
}

/// Statement-level counterpart of [`has_named_class_expr`].
fn has_named_class_stmt(s: &Stmt) -> bool {
    scan_named_class_stmt(s, 0)
}

fn scan_named_class_expr(e: &Expr, depth: u32) -> bool {
    if depth > MAX_SCAN_DEPTH {
        return true;
    }
    let d = depth + 1;
    match e {
        Expr::Class { name, .. } => name.is_some(),
        // Nested function/arrow bodies are not scanned: a named class
        // inside one gets a fresh binding per call already.
        Expr::Function { .. } | Expr::Arrow { .. } => false,
        Expr::Ident(..)
        | Expr::PrivateIdent(..)
        | Expr::Lit(..)
        | Expr::This(..)
        | Expr::Super(..)
        | Expr::NewTarget(..)
        | Expr::ImportMeta(..)
        | Expr::Regex { .. } => false,
        Expr::Array(elems, _) => elems.iter().flatten().any(|x| scan_named_class_expr(x, d)),
        Expr::Object(props, _) => props.iter().any(|p| match p {
            PropDef::KeyValue { key, value, .. } => {
                scan_named_class_key(key, d) || scan_named_class_expr(value, d)
            }
            PropDef::Shorthand { .. } => false,
            PropDef::Spread(x, _) => scan_named_class_expr(x, d),
            // Methods/getters/setters are function bodies (not scanned).
            PropDef::Method { .. } | PropDef::Getter { .. } | PropDef::Setter { .. } => false,
        }),
        Expr::Call { callee, args, .. } => {
            scan_named_class_expr(callee, d) || args.iter().any(|x| scan_named_class_expr(x, d))
        }
        Expr::New { callee, args, .. } => {
            scan_named_class_expr(callee, d) || args.iter().any(|x| scan_named_class_expr(x, d))
        }
        Expr::Member { obj, prop, .. } => {
            scan_named_class_expr(obj, d) || scan_named_class_expr(prop, d)
        }
        Expr::Unary { arg, .. } | Expr::Spread(arg, _) | Expr::Await(arg, _) => {
            scan_named_class_expr(arg, d)
        }
        Expr::Update { arg, .. } => scan_named_class_expr(arg, d),
        Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
            scan_named_class_expr(left, d) || scan_named_class_expr(right, d)
        }
        Expr::Assign { left, right, .. } => {
            scan_named_class_expr(left, d) || scan_named_class_expr(right, d)
        }
        Expr::Conditional {
            cond,
            then_e,
            else_e,
            ..
        } => {
            scan_named_class_expr(cond, d)
                || scan_named_class_expr(then_e, d)
                || scan_named_class_expr(else_e, d)
        }
        Expr::Sequence(es, _) => es.iter().any(|x| scan_named_class_expr(x, d)),
        Expr::Template { parts, .. } => parts.iter().any(|p| match p {
            TemplatePart::Str { .. } => false,
            TemplatePart::Expr(x) => scan_named_class_expr(x, d),
        }),
        Expr::TaggedTemplate { tag, parts, .. } => {
            scan_named_class_expr(tag, d)
                || parts.iter().any(|p| match p {
                    TemplatePart::Str { .. } => false,
                    TemplatePart::Expr(x) => scan_named_class_expr(x, d),
                })
        }
        Expr::Yield { arg, .. } => arg.as_ref().is_some_and(|x| scan_named_class_expr(x, d)),
    }
}

fn scan_named_class_key(k: &PropKey, depth: u32) -> bool {
    matches!(k, PropKey::Computed(e) if scan_named_class_expr(e, depth))
}

fn scan_named_class_stmt(s: &Stmt, depth: u32) -> bool {
    if depth > MAX_SCAN_DEPTH {
        return true;
    }
    let d = depth + 1;
    match s {
        Stmt::Expr(e) | Stmt::Throw(e) => scan_named_class_expr(e, d),
        Stmt::Block(ss) => ss.iter().any(|x| scan_named_class_stmt(x, d)),
        Stmt::VarDecl { decls, .. } => decls
            .iter()
            .any(|x| x.init.as_ref().is_some_and(|e| scan_named_class_expr(e, d))),
        // A class *declaration* binds its name via the outer `let`-like
        // binding (per-iteration block environments already keep it fresh);
        // only class *expressions* need the inner self-name scope.
        Stmt::FunctionDecl { .. } | Stmt::ClassDecl { .. } => false,
        Stmt::Return(e) => e.as_ref().is_some_and(|x| scan_named_class_expr(x, d)),
        Stmt::If {
            cond,
            then_b,
            else_b,
        } => {
            scan_named_class_expr(cond, d)
                || scan_named_class_stmt(then_b, d)
                || else_b.as_ref().is_some_and(|x| scan_named_class_stmt(x, d))
        }
        Stmt::While { cond, body } => {
            scan_named_class_expr(cond, d) || scan_named_class_stmt(body, d)
        }
        Stmt::DoWhile { body, cond } => {
            scan_named_class_stmt(body, d) || scan_named_class_expr(cond, d)
        }
        Stmt::For {
            init,
            cond,
            update,
            body,
        } => {
            init.as_ref().is_some_and(|i| match i {
                ForInit::Var(_, ds) => ds
                    .iter()
                    .any(|x| x.init.as_ref().is_some_and(|e| scan_named_class_expr(e, d))),
                ForInit::Expr(e) => scan_named_class_expr(e, d),
            }) || cond.as_ref().is_some_and(|e| scan_named_class_expr(e, d))
                || update.as_ref().is_some_and(|e| scan_named_class_expr(e, d))
                || scan_named_class_stmt(body, d)
        }
        Stmt::ForInOf { rhs, body, .. } => {
            scan_named_class_expr(rhs, d) || scan_named_class_stmt(body, d)
        }
        Stmt::Break(..) | Stmt::Continue(..) | Stmt::Debugger | Stmt::Empty => false,
        Stmt::Switch { disc, cases } => {
            scan_named_class_expr(disc, d)
                || cases.iter().any(|c| {
                    c.test.as_ref().is_some_and(|e| scan_named_class_expr(e, d))
                        || c.body.iter().any(|x| scan_named_class_stmt(x, d))
                })
        }
        Stmt::Try {
            block,
            handler,
            finalizer,
        } => {
            block.iter().any(|x| scan_named_class_stmt(x, d))
                || handler
                    .as_ref()
                    .is_some_and(|c| c.body.iter().any(|x| scan_named_class_stmt(x, d)))
                || finalizer
                    .as_ref()
                    .is_some_and(|f| f.iter().any(|x| scan_named_class_stmt(x, d)))
        }
        Stmt::Labeled { body, .. } => scan_named_class_stmt(body, d),
        Stmt::With { obj, body, .. } => {
            scan_named_class_expr(obj, d) || scan_named_class_stmt(body, d)
        }
        Stmt::Import(..) => false,
        Stmt::Export(e) => match e {
            ExportDecl::Decl { inner, .. } => scan_named_class_stmt(inner, d),
            ExportDecl::Default { expr, .. } => scan_named_class_expr(expr, d),
            _ => false,
        },
    }
}

fn scan_closure_stmt(s: &Stmt, depth: u32) -> bool {
    if depth > MAX_SCAN_DEPTH {
        return true;
    }
    let d = depth + 1;
    match s {
        Stmt::Expr(e) | Stmt::Throw(e) => scan_closure_expr(e, d),
        Stmt::Block(ss) => ss.iter().any(|x| scan_closure_stmt(x, d)),
        Stmt::VarDecl { decls, .. } => decls.iter().any(|x| {
            scan_closure_pat(&x.id, d) || x.init.as_ref().is_some_and(|e| scan_closure_expr(e, d))
        }),
        // Function/class declarations are closures.
        Stmt::FunctionDecl { .. } | Stmt::ClassDecl { .. } => true,
        Stmt::Return(e) => e.as_ref().is_some_and(|x| scan_closure_expr(x, d)),
        Stmt::If {
            cond,
            then_b,
            else_b,
        } => {
            scan_closure_expr(cond, d)
                || scan_closure_stmt(then_b, d)
                || else_b.as_ref().is_some_and(|x| scan_closure_stmt(x, d))
        }
        Stmt::While { cond, body } => scan_closure_expr(cond, d) || scan_closure_stmt(body, d),
        Stmt::DoWhile { body, cond } => scan_closure_stmt(body, d) || scan_closure_expr(cond, d),
        Stmt::For {
            init,
            cond,
            update,
            body,
        } => {
            init.as_ref().is_some_and(|i| match i {
                ForInit::Var(_, ds) => ds.iter().any(|x| {
                    scan_closure_pat(&x.id, d)
                        || x.init.as_ref().is_some_and(|e| scan_closure_expr(e, d))
                }),
                ForInit::Expr(e) => scan_closure_expr(e, d),
            }) || cond.as_ref().is_some_and(|e| scan_closure_expr(e, d))
                || update.as_ref().is_some_and(|e| scan_closure_expr(e, d))
                || scan_closure_stmt(body, d)
        }
        Stmt::ForInOf { lhs, rhs, body, .. } => {
            (match lhs {
                ForLhs::Ident(..) => false,
                ForLhs::Pat(p) => scan_closure_pat(p, d),
                ForLhs::Member(e) => scan_closure_expr(e, d),
            }) || scan_closure_expr(rhs, d)
                || scan_closure_stmt(body, d)
        }
        Stmt::Break(..) | Stmt::Continue(..) | Stmt::Debugger | Stmt::Empty => false,
        Stmt::Switch { disc, cases } => {
            scan_closure_expr(disc, d)
                || cases.iter().any(|c| {
                    c.test.as_ref().is_some_and(|e| scan_closure_expr(e, d))
                        || c.body.iter().any(|x| scan_closure_stmt(x, d))
                })
        }
        Stmt::Try {
            block,
            handler,
            finalizer,
        } => {
            block.iter().any(|x| scan_closure_stmt(x, d))
                || handler.as_ref().is_some_and(|c| {
                    c.param
                        .as_ref()
                        .is_some_and(|p| scan_closure_pat(&p.pat, d))
                        || c.body.iter().any(|x| scan_closure_stmt(x, d))
                })
                || finalizer
                    .as_ref()
                    .is_some_and(|f| f.iter().any(|x| scan_closure_stmt(x, d)))
        }
        Stmt::Labeled { body, .. } => scan_closure_stmt(body, d),
        Stmt::With { obj, body, .. } => scan_closure_expr(obj, d) || scan_closure_stmt(body, d),
        Stmt::Import(..) => false,
        Stmt::Export(e) => match e {
            ExportDecl::Decl { inner, .. } => scan_closure_stmt(inner, d),
            ExportDecl::Default { expr, .. } => scan_closure_expr(expr, d),
            _ => false,
        },
    }
}

/// True if the statement declares a `let`/`const` or class binding outside
/// any nested function. Such bindings are fresh for every loop iteration.
fn stmt_has_lexical(s: &Stmt) -> bool {
    match s {
        Stmt::VarDecl { kind, .. } => *kind != VarKind::Var,
        // A class declaration binds its name lexically; the class value is
        // created fresh per evaluation.
        Stmt::ClassDecl { .. } => true,
        Stmt::Block(ss) => ss.iter().any(stmt_has_lexical),
        // Do not descend into nested functions: their bindings are per-call.
        Stmt::FunctionDecl { .. } => false,
        Stmt::If { then_b, else_b, .. } => {
            stmt_has_lexical(then_b) || else_b.as_ref().is_some_and(|x| stmt_has_lexical(x))
        }
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => stmt_has_lexical(body),
        Stmt::For { init, body, .. } => {
            init.as_ref()
                .is_some_and(|i| matches!(i, ForInit::Var(k, _) if *k != VarKind::Var))
                || stmt_has_lexical(body)
        }
        Stmt::ForInOf { kind, body, .. } => *kind != VarKind::Var || stmt_has_lexical(body),
        Stmt::Switch { cases, .. } => cases.iter().any(|c| c.body.iter().any(stmt_has_lexical)),
        Stmt::Try {
            block,
            handler,
            finalizer,
        } => {
            block.iter().any(stmt_has_lexical)
                || handler
                    .as_ref()
                    .is_some_and(|c| c.body.iter().any(stmt_has_lexical))
                || finalizer
                    .as_ref()
                    .is_some_and(|f| f.iter().any(stmt_has_lexical))
        }
        Stmt::Labeled { body, .. } => stmt_has_lexical(body),
        Stmt::With { body, .. } => stmt_has_lexical(body),
        Stmt::Export(e) => matches!(e, ExportDecl::Decl { inner, .. } if stmt_has_lexical(inner)),
        _ => false,
    }
}

/// Collect the identifiers bound by a pattern, in order.
fn pat_bound_names(p: &Pat, out: &mut Vec<String>) {
    match p {
        Pat::Ident(n, _) => out.push(n.clone()),
        Pat::Array(elems, _) => {
            for el in elems.iter().flatten() {
                let ArrayPatElem::Pat(q) = el;
                pat_bound_names(q, out);
            }
        }
        Pat::Object(props, _) => {
            for pr in props {
                match pr {
                    ObjectPatProp::KeyValue { value, .. } => pat_bound_names(value, out),
                    ObjectPatProp::Shorthand { name, .. } => out.push(name.clone()),
                    ObjectPatProp::Rest(q, _) => pat_bound_names(q, out),
                }
            }
        }
        Pat::Rest(q, _) => pat_bound_names(q, out),
        Pat::Assign(q, _, _) => pat_bound_names(q, out),
        // Member targets bind nothing.
        Pat::Member(..) => {}
    }
}

/// True when a classic `for` loop needs per-iteration body extraction: a
/// `let`/`const` binding that is fresh per iteration (head or body lexical)
/// may be captured by a closure in the body/cond/update. Conservative: any
/// closure triggers it, since extraction preserves semantics via writeback
/// of the head bindings.
fn for_needs_extraction(
    is_async_or_gen: bool,
    init: Option<&ForInit>,
    cond: Option<&Expr>,
    update: Option<&Expr>,
    body: &Stmt,
) -> bool {
    // Extraction moves the body into a fresh arrow per iteration; that
    // cannot carry `yield`/`await`, so async functions and generators keep
    // the shared-slot behavior.
    if is_async_or_gen {
        return false;
    }
    // A named class expression binds its name in a fresh inner scope at
    // each evaluation; extraction gives every iteration its own binding,
    // so a named class in the head or body forces extraction on its own.
    if cond.is_some_and(has_named_class_expr)
        || update.is_some_and(has_named_class_expr)
        || has_named_class_stmt(body)
    {
        return true;
    }
    let head_lex = matches!(init, Some(ForInit::Var(k, _)) if *k != VarKind::Var);
    if !head_lex && !stmt_has_lexical(body) {
        return false;
    }
    cond.is_some_and(has_closure_expr)
        || update.is_some_and(has_closure_expr)
        || has_closure_stmt(body)
}

/// True when a `for-in`/`for-of` loop needs per-iteration body extraction.
fn for_in_of_needs_extraction(is_async_or_gen: bool, kind: VarKind, body: &Stmt) -> bool {
    if is_async_or_gen {
        return false;
    }
    // A named class expression in the body binds its name afresh at each
    // evaluation; extraction gives every iteration its own binding.
    if has_named_class_stmt(body) {
        return true;
    }
    // `for (let k ...)` / `for (const k ...)` bind per iteration; a bare
    // `for (k ...)` assigns to an existing binding.
    if kind == VarKind::Var && !stmt_has_lexical(body) {
        return false;
    }
    has_closure_stmt(body)
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

impl Compiler {
    fn compile_stmts(&mut self, stmts: &[Stmt]) -> Result<(), CompileError> {
        for s in stmts {
            self.compile_stmt(s)?;
        }
        Ok(())
    }

    fn compile_stmt(&mut self, s: &Stmt) -> Result<(), CompileError> {
        self.guarded(|this| this.compile_stmt_inner(s))
    }

    fn compile_stmt_inner(&mut self, s: &Stmt) -> Result<(), CompileError> {
        match s {
            Stmt::Expr(e) => {
                // Fused discarded-value string append (`s += rhs`,
                // `s = s + rhs`): amortized O(1) per append via the
                // frame's chunk accumulator instead of O(n) copies.
                if self.compile_str_append_stmt(e)? {
                    return Ok(());
                }
                self.compile_expr(e)?;
                self.emit(Op::Pop);
                Ok(())
            }
            Stmt::Block(ss) => self.compile_block(ss),
            Stmt::VarDecl { kind, decls, .. } => self.compile_var_decl(*kind, decls),
            // Function declarations are compiled by hoist() and initialized
            // in the prologue.
            Stmt::FunctionDecl { .. } => Ok(()),
            Stmt::ClassDecl {
                name,
                superclass,
                body,
                ..
            } => {
                let n = name.clone().unwrap_or_else(|| "#anon".to_string());
                self.compile_class_value(Some(n.clone()), superclass.as_ref(), body, false)?;
                self.compile_store_decl(&n, VarKind::Let)?;
                Ok(())
            }
            Stmt::Return(e) => {
                // `return` is illegal directly inside a static block (the
                // wrapper is synthetic; a `return` there would silently
                // return from the wrapper instead of being a SyntaxError).
                if self.fns.last().expect("no active function").is_class_init {
                    return Err(CompileError::Unsupported("return in static block"));
                }
                match e {
                    Some(x) => self.compile_expr(x)?,
                    None => self.emit(Op::PushUndefined),
                }
                self.unwind_protected(0, false, None)?;
                if self.fns.last().expect("no active function").extracted {
                    // In an extracted body, `return v` becomes `[3, v]` so
                    // the wrapper can return from the enclosing function.
                    self.emit(Op::PushInt(3));
                    self.emit(Op::Swap);
                    self.emit(Op::NewArray(2));
                }
                self.emit(Op::Return);
                Ok(())
            }
            Stmt::If {
                cond,
                then_b,
                else_b,
            } => {
                self.compile_expr(cond)?;
                let l_else = self.emit_jump(Op::JumpIfFalse(0));
                self.compile_stmt(then_b)?;
                let l_end = self.emit_jump(Op::Jump(0));
                self.patch(l_else)?;
                if let Some(e) = else_b {
                    self.compile_stmt(e)?;
                }
                self.patch(l_end)?;
                Ok(())
            }
            Stmt::While { cond, body } => self.compile_while(cond, body),
            Stmt::DoWhile { body, cond } => {
                let labels = std::mem::take(&mut self.pending_labels);
                let depth = self.protected_len();
                let frame = self.push_loop(LoopKind::Loop, labels, depth);
                let l_body = self.code_len();
                self.compile_stmt(body)?;
                let l_cont = self.code_len();
                self.patch_continues_to(frame, l_cont)?;
                self.compile_expr(cond)?;
                self.emit_jump_to_cond(l_body, true)?;
                let l_end = self.code_len();
                self.pop_loop_to(frame, l_end)?;
                Ok(())
            }
            Stmt::For {
                init,
                cond,
                update,
                body,
            } => {
                let extract = {
                    let f = self.fns.last().expect("no active function");
                    for_needs_extraction(
                        f.is_async || f.is_generator,
                        init.as_ref(),
                        cond.as_ref(),
                        update.as_ref(),
                        body,
                    )
                };
                if extract {
                    return self.compile_for_extracted(
                        init.as_ref(),
                        cond.as_ref(),
                        update.as_ref(),
                        body,
                    );
                }
                // Replay the head scope declared by the declare walk.
                self.push_block_scope();
                if let Some(i) = init {
                    match i {
                        ForInit::Var(k, ds) => self.compile_var_decl(*k, ds)?,
                        ForInit::Expr(e) => {
                            self.compile_expr(e)?;
                            self.emit(Op::Pop);
                        }
                    }
                }
                let labels = std::mem::take(&mut self.pending_labels);
                let depth = self.protected_len();
                let frame = self.push_loop(LoopKind::Loop, labels, depth);
                let l_cond = self.code_len();
                if let Some(c) = cond {
                    self.compile_expr(c)?;
                    let brk = self.emit_jump(Op::JumpIfFalse(0));
                    self.fns.last_mut().expect("no active function").loops[frame]
                        .break_patches
                        .push(brk);
                }
                self.compile_stmt(body)?;
                let l_cont = self.code_len();
                self.patch_continues_to(frame, l_cont)?;
                if let Some(u) = update {
                    self.compile_expr(u)?;
                    self.emit(Op::Pop);
                }
                self.emit_jump_to(l_cond)?;
                let l_end = self.code_len();
                self.pop_loop_to(frame, l_end)?;
                self.pop_scope();
                Ok(())
            }
            Stmt::ForInOf {
                kind,
                lhs,
                rhs,
                body,
                is_of,
            } => self.compile_for_in_of(*kind, lhs, rhs, body, *is_of),
            Stmt::Break(label) => self.compile_break(label),
            Stmt::Continue(label) => self.compile_continue(label),
            Stmt::Switch { disc, cases } => self.compile_switch(disc, cases),
            Stmt::Try {
                block,
                handler,
                finalizer,
            } => self.compile_try(block, handler.as_ref(), finalizer.as_deref()),
            Stmt::Throw(e) => {
                self.compile_expr(e)?;
                // No unwinding code: the VM's throw dispatch runs `finally`
                // bodies and `catch` handlers via the registered handlers.
                // (Emitting `LeaveTry` here would deregister the catcher;
                // inlining `finally` here would run it twice.)
                self.emit(Op::Throw);
                Ok(())
            }
            Stmt::Debugger => {
                self.emit(Op::Nop);
                Ok(())
            }
            Stmt::Empty => Ok(()),
            Stmt::Labeled { label, body, .. } => {
                let n = self.pending_labels.len();
                self.pending_labels.push(label.clone());
                let r = self.compile_stmt(body);
                self.pending_labels.truncate(n);
                r
            }
            Stmt::With { obj, body } => self.compile_with(obj, body),
            Stmt::Import(d) => {
                self.module.imports.push(d.clone());
                self.emit(Op::Nop);
                Ok(())
            }
            Stmt::Export(e) => self.compile_export(e),
        }
    }

    /// Compile a block: push its declare-walk scope so lexical bindings
    /// resolve to the ids declared up front, compile the statements, then
    /// pop. Scopes pop on the error path too.
    fn compile_block(&mut self, stmts: &[Stmt]) -> Result<(), CompileError> {
        let labels = std::mem::take(&mut self.pending_labels);
        self.push_block_scope();
        let frame = if labels.is_empty() {
            None
        } else {
            Some(self.push_loop(LoopKind::Block, labels, self.protected_len()))
        };
        for s in stmts {
            if let Err(e) = self.compile_stmt(s) {
                if let Some(f) = frame {
                    self.fns
                        .last_mut()
                        .expect("no active function")
                        .loops
                        .truncate(f);
                }
                self.pop_scope();
                return Err(e);
            }
        }
        if let Some(f) = frame {
            let end = self.code_len();
            self.pop_loop_to(f, end)?;
        }
        self.pop_scope();
        Ok(())
    }

    fn compile_var_decl(&mut self, kind: VarKind, decls: &[VarDecl]) -> Result<(), CompileError> {
        // Bindings were declared by declare_stmts(); only initializers run here.
        for d in decls {
            match (&d.init, kind) {
                (Some(e), _) => {
                    self.compile_expr(e)?;
                    self.compile_pat_bind(&d.id, kind)?;
                }
                // `var x;` is already `undefined` from the prologue.
                (None, VarKind::Var) => {}
                // Bare `let x;` initializes to `undefined` and clears the
                // TDZ flag, exactly like an explicit `= undefined`
                // initializer. (Non-`Ident` patterns without an initializer
                // stay in TDZ; `const x;` never reaches here: the parser
                // rejects it.)
                (None, VarKind::Let) => {
                    if let Pat::Ident(name, _) = &d.id {
                        self.emit(Op::PushUndefined);
                        self.compile_store_decl(name, VarKind::Let)?;
                    }
                }
                // `const`/`using` without an initializer never reach here:
                // the parser rejects them.
                (None, VarKind::Const | VarKind::Using | VarKind::AwaitUsing) => {}
            }
        }
        Ok(())
    }

    // -- loops ------------------------------------------------------------

    fn push_loop(&mut self, kind: LoopKind, labels: Vec<String>, protected_depth: usize) -> usize {
        let f = self.fns.last_mut().expect("no active function");
        f.loops.push(LoopFrame {
            kind,
            labels,
            break_patches: Vec::new(),
            continue_patches: Vec::new(),
            protected_depth,
            for_of_region: None,
        });
        f.loops.len() - 1
    }

    fn patch_continues_to(&mut self, frame: usize, target: usize) -> Result<(), CompileError> {
        let patches = std::mem::take(
            &mut self.fns.last_mut().expect("no active function").loops[frame].continue_patches,
        );
        for p in patches {
            self.patch_to(p, target)?;
        }
        Ok(())
    }

    fn pop_loop_to(&mut self, frame: usize, end: usize) -> Result<(), CompileError> {
        let fr = self
            .fns
            .last_mut()
            .expect("no active function")
            .loops
            .pop()
            .expect("loop frame");
        debug_assert_eq!(
            self.fns.last().expect("no active function").loops.len(),
            frame
        );
        for p in fr.break_patches {
            self.patch_to(p, end)?;
        }
        Ok(())
    }

    /// `JumpIfTrue target` for do-while (pops the condition).
    fn emit_jump_to_cond(&mut self, target: usize, jump_if_true: bool) -> Result<(), CompileError> {
        let pos = self.emit_jump(if jump_if_true {
            Op::JumpIfTrue(0)
        } else {
            Op::JumpIfFalse(0)
        });
        self.patch_to(pos, target)
    }

    fn compile_while(&mut self, cond: &Expr, body: &Stmt) -> Result<(), CompileError> {
        let labels = std::mem::take(&mut self.pending_labels);
        let depth = self.protected_len();
        let frame = self.push_loop(LoopKind::Loop, labels, depth);
        let l_cond = self.code_len();
        self.compile_expr(cond)?;
        let brk = self.emit_jump(Op::JumpIfFalse(0));
        self.fns.last_mut().expect("no active function").loops[frame]
            .break_patches
            .push(brk);
        self.compile_stmt(body)?;
        let l_cont = self.code_len();
        self.patch_continues_to(frame, l_cont)?;
        self.emit_jump_to(l_cond)?;
        let l_end = self.code_len();
        self.pop_loop_to(frame, l_end)?;
        Ok(())
    }

    fn compile_for_in_of(
        &mut self,
        kind: VarKind,
        lhs: &ForLhs,
        rhs: &Expr,
        body: &Stmt,
        is_of: bool,
    ) -> Result<(), CompileError> {
        let extract = {
            let f = self.fns.last().expect("no active function");
            for_in_of_needs_extraction(f.is_async || f.is_generator, kind, body)
        };
        if extract {
            return self.compile_for_in_of_extracted(kind, lhs, rhs, body, is_of);
        }
        // Replay the head scope declared by the declare walk.
        self.push_block_scope();
        // Stack discipline: [iterable, index] across iterations.
        self.compile_expr(rhs)?;
        if !is_of {
            self.call_intrinsic("enum_keys", 1)?;
        }
        self.emit(Op::PushInt(0));
        let labels = std::mem::take(&mut self.pending_labels);
        let depth = self.protected_len();
        // A `for-of` loop owns iterator-close obligations (spec 14.7.5):
        // the runtime guard lets a `throw` unwinding through the loop run
        // `IteratorClose` on the in-flight iterator, and the compile-time
        // `Protected::ForOf` region makes `break`/`return` emit the close
        // inline via `unwind_protected`. `for-in` iterates keys with no
        // iterator, so it gets neither.
        let for_of_region = if is_of {
            self.emit(Op::IterGuard);
            let region = self.fns.last().expect("no active function").protected.len();
            self.fns
                .last_mut()
                .expect("no active function")
                .protected
                .push(Protected::ForOf);
            Some(region)
        } else {
            None
        };
        let frame = self.push_loop(
            if is_of {
                LoopKind::ForOf
            } else {
                LoopKind::Loop
            },
            labels,
            depth,
        );
        if let Some(region) = for_of_region {
            self.fns.last_mut().expect("no active function").loops[frame].for_of_region =
                Some(region);
        }
        let l_head = self.code_len();
        self.emit(Op::IterNext); // [it, i+1, value, done]
        let done_jump = self.emit_jump(Op::JumpIfTrue(0));
        // [it, i+1, value]
        match lhs {
            ForLhs::Ident(name, _) if kind == VarKind::Var => {
                // Bare `for (x of ...)`: plain assignment.
                self.compile_store(name)?;
            }
            ForLhs::Ident(name, _) => {
                self.compile_store_decl(name, kind)?;
            }
            ForLhs::Pat(p) => {
                // `var`/bare patterns assign per iteration (spec
                // `PutValue`, with-aware); `let`/`const` declare fresh
                // per-iteration bindings.
                if kind == VarKind::Var {
                    self.compile_pat_bind_assign(p)?;
                } else {
                    self.compile_pat_bind(p, kind)?;
                }
            }
            ForLhs::Member(e) => {
                let t = self.compile_target_begin(e)?;
                self.compile_target_store_pop(&t)?;
            }
        }
        self.compile_stmt(body)?;
        // [it, i+1]
        let l_cont = self.code_len();
        self.patch_continues_to(frame, l_cont)?;
        self.emit_jump_to(l_head)?;
        let l_done = self.code_len();
        self.patch_to(done_jump, l_done)?;
        if is_of {
            // Normal exhaustion: disarm the runtime guard without closing
            // (spec: `IteratorClose` runs only on abrupt completion).
            self.emit(Op::IterUnguard);
        }
        self.emit(Op::Pop);
        self.emit(Op::Pop);
        self.emit(Op::Pop);
        let l_end = self.code_len();
        self.pop_loop_to(frame, l_end)?;
        if for_of_region.is_some() {
            let popped = self
                .fns
                .last_mut()
                .expect("no active function")
                .protected
                .pop();
            debug_assert!(matches!(popped, Some(Protected::ForOf)));
        }
        self.pop_scope();
        Ok(())
    }

    // -- per-iteration extraction ------------------------------------------

    /// Emit `return [tag, ...extracted_params]` in an extracted body.
    fn emit_tagged_return(&mut self, tag: i8) -> Result<(), CompileError> {
        let params = self
            .fns
            .last()
            .expect("no active function")
            .extracted_params
            .clone();
        if params.len() > 254 {
            return Err(CompileError::Unsupported(
                "too many extracted loop variables",
            ));
        }
        self.emit(Op::PushInt(tag));
        for n in &params {
            self.compile_load(n)?;
        }
        self.emit(Op::NewArray(1 + params.len() as u8));
        self.emit(Op::Return);
        Ok(())
    }

    /// Emit `return [tag, label]` in an extracted body.
    fn emit_tagged_label(&mut self, tag: i8, label: &str) -> Result<(), CompileError> {
        self.emit(Op::PushInt(tag));
        let li = self.intern(label)?;
        self.emit(Op::LoadConst(li));
        self.emit(Op::NewArray(2));
        self.emit(Op::Return);
        Ok(())
    }

    /// Compile a per-iteration loop body to an arrow template. The arrow is
    /// called once per iteration with the head `let`/`const` values as
    /// arguments, so every iteration gets fresh parameter slots and closures
    /// created in the body capture per-iteration values. Abrupt completions
    /// targeting outside the body (`break`, `continue`, `return`, labeled
    /// jumps) become tagged array returns that the wrapper dispatches.
    /// Returns the template const index plus labels that did not resolve
    /// inside the body (the wrapper validates them against enclosing
    /// labels).
    fn compile_extracted_body(
        &mut self,
        params: &[Param],
        body: &[Stmt],
    ) -> Result<(u16, Vec<String>), CompileError> {
        let strict = self.fns.last().expect("no active function").strict;
        if params.len() > u16::MAX as usize {
            return Err(CompileError::Unsupported(
                "too many extracted loop variables",
            ));
        }
        let arity = params.len() as u16;
        // Snapshot the enclosing `with` stack: `push_function` clears it
        // for the nested body, but this arrow is compiler-generated (not
        // a user closure) and must keep observing the with-objects — e.g.
        // `with (o) { for (let z of [1]) { let q = x; ... } }` reads
        // `o.x` per iteration.
        let preserved_with: Vec<(String, usize)> = self
            .with_stack
            .iter()
            .map(|w| (w.name.clone(), w.scope_depth))
            .collect();
        self.push_function(
            "#for_body".to_string(),
            arity,
            true,
            false,
            false,
            strict,
            false,
        );
        // Parameters are function-scoped bindings that shadow the head
        // bindings of the same name in the enclosing function.
        for p in params {
            self.declare_pat_top(&p.pat, VarKind::Var)?;
        }
        {
            let f = self.fns.last_mut().expect("no active function");
            f.extracted = true;
            f.no_vars = true;
            let mut names = Vec::new();
            for p in params {
                pat_bound_names(&p.pat, &mut names);
            }
            f.extracted_params = names;
        }
        self.declare_stmts(body)?;
        self.fns.last_mut().expect("no active function").no_vars = false;
        self.emit_prologue()?;
        self.bind_params(params)?;
        // Preserve the enclosing `with` bindings across the synthetic
        // arrow: each with-object is captured as an upvalue and copied
        // into a hidden arrow-local, so probes inside the extracted body
        // keep working. The original `scope_depth` is kept — the
        // enclosing scopes are still on the scope stack below the
        // arrow's, so shadowing resolves exactly as in the enclosing
        // function. (User closures defined in the body still get an empty
        // stack from `push_function`: the documented v1 deviation.)
        for (wname, wdepth) in &preserved_with {
            let local = self.hidden_temp();
            let lnc = self.binding_name_const(&local)?;
            match self.resolve(wname)? {
                Resolved::Local(nc, _) => self.emit(Op::LoadName(nc)),
                Resolved::Upvalue(idx, _) => self.emit(Op::LoadUpvalue(idx)),
                Resolved::Global(_) => return Err(CompileError::InvalidLhs),
            }
            self.emit(Op::StoreName(lnc));
            self.with_stack.push(WithEntry {
                name: local,
                name_const: lnc,
                scope_depth: *wdepth,
            });
        }
        // Labels on the `for` itself label the extracted loop; move them
        // into the arrow's loop frame so `break`/`continue` targeting them
        // produce tag 1/2 rather than tag 4/5.
        let labels = std::mem::take(&mut self.pending_labels);
        let frame = self.push_loop(LoopKind::Loop, labels, 0);
        self.fns
            .last_mut()
            .expect("no active function")
            .extracted_loop = frame;
        // Per-iteration instantiation of the body-level function declarations
        // hoisted by `declare_body_vars` (Annex B block-level semantics):
        // a fresh closure per iteration, capturing this iteration's
        // environment — the template was compiled during this arrow's
        // declare walk, so its upvalues target this arrow's scope. The
        // store targets the same binding the body resolves (usually the
        // hoisted enclosing binding, via upvalue) and bypasses `with`
        // probes, like the prologue's function initialization does.
        let fn_inits = std::mem::take(
            &mut self
                .fns
                .last_mut()
                .expect("no active function")
                .extracted_fn_inits,
        );
        for (name, ti) in fn_inits {
            self.emit(Op::NewFunction(ti));
            self.compile_store_inner(&name)?;
        }
        let r = self.compile_stmts(body);
        // Pop the extracted frame even on error (the whole compile is
        // aborted anyway, but keep the state consistent).
        self.fns.last_mut().expect("no active function").loops.pop();
        r?;
        // Fall off the end: normal completion `[0, ...params]`.
        self.emit_tagged_return(0)?;
        // `emit_tagged_return` already returned; this is dead code kept for
        // structural symmetry with other function epilogues.
        self.emit(Op::PushUndefined);
        self.emit(Op::Return);
        let unresolved = std::mem::take(
            &mut self
                .fns
                .last_mut()
                .expect("no active function")
                .unresolved_labels,
        );
        let t = self.pop_function();
        let idx = self.intern_template(t)?;
        Ok((idx, unresolved))
    }

    /// Emit a wrapper-side `break` out of the extracted loop `frame`.
    fn emit_wrapper_break(&mut self, frame: usize) -> Result<(), CompileError> {
        let depth = self.fns.last().expect("no active function").loops[frame].protected_depth;
        self.unwind_protected(depth, true, None)?;
        let pos = self.emit_jump(Op::Jump(0));
        self.fns.last_mut().expect("no active function").loops[frame]
            .break_patches
            .push(pos);
        Ok(())
    }

    /// Emit a wrapper-side `continue` of the extracted loop `frame`.
    fn emit_wrapper_continue(&mut self, frame: usize) -> Result<(), CompileError> {
        let fr = &self.fns.last().expect("no active function").loops[frame];
        let depth = fr.protected_depth;
        let skip = fr.for_of_region;
        self.unwind_protected(depth, true, skip)?;
        let pos = self.emit_jump(Op::Jump(0));
        self.fns.last_mut().expect("no active function").loops[frame]
            .continue_patches
            .push(pos);
        Ok(())
    }

    /// Emit dispatch for tag 4/5: the label string is on the stack; compare
    /// against each enclosing label and break/continue to the match.
    fn emit_label_dispatch(
        &mut self,
        comp_nc: u16,
        labels: &[String],
        is_continue: bool,
    ) -> Result<(), CompileError> {
        self.emit(Op::LoadName(comp_nc));
        self.emit(Op::PushInt(1));
        self.emit(Op::GetElem);
        // [label_str]
        for l in labels {
            self.emit(Op::Dup);
            let li = self.intern(l)?;
            self.emit(Op::LoadConst(li));
            self.emit(Op::StrictEq);
            let jn = self.emit_jump(Op::JumpIfFalse(0));
            self.emit(Op::Pop);
            let opt = Some(l.clone());
            if is_continue {
                self.compile_continue(&opt)?;
            } else {
                self.compile_break(&opt)?;
            }
            self.patch(jn)?;
        }
        self.emit(Op::Pop);
        Ok(())
    }

    /// Emit the tag dispatch shared by both extracted loop wrappers.
    /// `comp_nc` holds `$comp`; `head_lets` are the wrapper's head lexical
    /// names to write back on tags 0/2 (empty for the for-in/of bare-target
    /// case); `update` is the classic update expression (None for
    /// for-in/of). `frame` is the wrapper loop frame.
    #[allow(clippy::too_many_arguments)]
    fn emit_extracted_dispatch(
        &mut self,
        comp_nc: u16,
        frame: usize,
        head_lets: &[String],
        enclosing_labels: &[String],
        update: Option<&Expr>,
    ) -> Result<(), CompileError> {
        // $t = $comp[0]
        self.emit(Op::LoadName(comp_nc));
        self.emit(Op::PushInt(0));
        self.emit(Op::GetElem);
        // tag 3: return $comp[1]
        self.emit(Op::Dup);
        self.emit(Op::PushInt(3));
        self.emit(Op::StrictEq);
        let j3 = self.emit_jump(Op::JumpIfFalse(0));
        self.emit(Op::Pop);
        self.emit(Op::LoadName(comp_nc));
        self.emit(Op::PushInt(1));
        self.emit(Op::GetElem);
        self.unwind_protected(0, false, None)?;
        self.emit(Op::Return);
        self.patch(j3)?;
        // tag 1: break
        self.emit(Op::Dup);
        self.emit(Op::PushInt(1));
        self.emit(Op::StrictEq);
        let j1 = self.emit_jump(Op::JumpIfFalse(0));
        self.emit(Op::Pop);
        self.emit_wrapper_break(frame)?;
        self.patch(j1)?;
        // tag 4: break <label>
        self.emit(Op::Dup);
        self.emit(Op::PushInt(4));
        self.emit(Op::StrictEq);
        let j4 = self.emit_jump(Op::JumpIfFalse(0));
        self.emit(Op::Pop);
        self.emit_label_dispatch(comp_nc, enclosing_labels, false)?;
        self.emit_wrapper_break(frame)?;
        self.patch(j4)?;
        // tag 5: continue <label>
        self.emit(Op::Dup);
        self.emit(Op::PushInt(5));
        self.emit(Op::StrictEq);
        let j5 = self.emit_jump(Op::JumpIfFalse(0));
        self.emit(Op::Pop);
        self.emit_label_dispatch(comp_nc, enclosing_labels, true)?;
        self.emit_wrapper_continue(frame)?;
        self.patch(j5)?;
        // tags 0 and 2: write the per-iteration values back to the head
        // bindings, then run the update expression if any.
        self.emit(Op::Pop);
        for (i, n) in head_lets.iter().enumerate() {
            self.emit(Op::LoadName(comp_nc));
            let ci = self.intern_int(i as i32 + 1)?;
            self.emit(Op::LoadConst(ci));
            self.emit(Op::GetElem);
            self.compile_store(n)?;
        }
        if let Some(u) = update {
            self.compile_expr(u)?;
            self.emit(Op::Pop);
        }
        Ok(())
    }

    /// Compile `for (init; cond; update) body` with per-iteration body
    /// extraction. See [`Self::compile_extracted_body`].
    fn compile_for_extracted(
        &mut self,
        init: Option<&ForInit>,
        cond: Option<&Expr>,
        update: Option<&Expr>,
        body: &Stmt,
    ) -> Result<(), CompileError> {
        // Head lexical names become the arrow's parameters.
        let mut head_lets = Vec::new();
        if let Some(ForInit::Var(k, ds)) = init
            && *k != VarKind::Var
        {
            for d in ds {
                pat_bound_names(&d.id, &mut head_lets);
            }
        }
        if head_lets.len() > 254 {
            return Err(CompileError::Unsupported(
                "too many extracted loop variables",
            ));
        }
        // Labels enclosing the loop, for the tag-4/5 dispatch. Collected
        // before compiling the arrow (which pushes its own function).
        let mut enclosing_labels: Vec<String> = Vec::new();
        for fr in &self.fns.last().expect("no active function").loops {
            for l in &fr.labels {
                if !enclosing_labels.contains(l) {
                    enclosing_labels.push(l.clone());
                }
            }
        }
        let params: Vec<Param> = head_lets
            .iter()
            .map(|n| Param {
                pat: Pat::Ident(n.clone(), Span { start: 0, end: 0 }),
                span: Span { start: 0, end: 0 },
            })
            .collect();
        let (arrow_idx, unresolved) =
            self.compile_extracted_body(&params, std::slice::from_ref(body))?;
        for l in &unresolved {
            if !enclosing_labels.contains(l) {
                return Err(CompileError::InvalidLhs);
            }
        }
        let body_fn = self.hidden_temp();
        let comp = self.hidden_temp();
        // Replay the head scope: init, cond, update and the writeback run
        // in the wrapper. (On compile error the whole compile is aborted,
        // so the scope is only popped on the success path.)
        self.push_block_scope();
        if let Some(i) = init {
            match i {
                ForInit::Var(k, ds) => self.compile_var_decl(*k, ds)?,
                ForInit::Expr(e) => {
                    self.compile_expr(e)?;
                    self.emit(Op::Pop);
                }
            }
        }
        self.emit(Op::NewFunction(arrow_idx));
        let body_fn_nc = self.binding_name_const(&body_fn)?;
        self.emit(Op::StoreName(body_fn_nc));
        let comp_nc = self.binding_name_const(&comp)?;
        if head_lets.len() > 255 {
            return Err(CompileError::Unsupported(
                "too many extracted loop variables",
            ));
        }
        let frame = self.push_loop(LoopKind::Loop, Vec::new(), self.protected_len());
        let l_cond = self.code_len();
        if let Some(c) = cond {
            self.compile_expr(c)?;
            let brk = self.emit_jump(Op::JumpIfFalse(0));
            self.fns.last_mut().expect("no active function").loops[frame]
                .break_patches
                .push(brk);
        }
        // $comp = $body_fn(...head_lets)
        self.emit(Op::LoadName(body_fn_nc));
        self.emit(Op::PushUndefined);
        for n in &head_lets {
            self.compile_load(n)?;
        }
        self.emit(Op::Call(head_lets.len() as u8));
        self.emit(Op::StoreName(comp_nc));
        self.emit_extracted_dispatch(comp_nc, frame, &head_lets, &enclosing_labels, update)?;
        let l_cont = self.code_len();
        self.patch_continues_to(frame, l_cont)?;
        self.emit_jump_to(l_cond)?;
        let l_end = self.code_len();
        self.pop_loop_to(frame, l_end)?;
        // body_fn and comp are dead after the dispatch loop.
        self.release_temp(&body_fn);
        self.release_temp(&comp);
        self.pop_scope();
        Ok(())
    }

    /// Compile `for (kind lhs of/in rhs) body` with per-iteration body
    /// extraction. See [`Self::compile_extracted_body`].
    fn compile_for_in_of_extracted(
        &mut self,
        kind: VarKind,
        lhs: &ForLhs,
        rhs: &Expr,
        body: &Stmt,
        is_of: bool,
    ) -> Result<(), CompileError> {
        // A lexical loop variable becomes the arrow's single parameter (a
        // fresh binding per iteration); a `var`/bare/member target is
        // stored in the wrapper and the arrow takes no parameters.
        let declares_lexical = kind != VarKind::Var && !matches!(lhs, ForLhs::Member(_));
        let params: Vec<Param> = if declares_lexical {
            let pat = match lhs {
                ForLhs::Pat(p) => p.clone(),
                ForLhs::Ident(n, _) => Pat::Ident(n.clone(), Span { start: 0, end: 0 }),
                // Unreachable: `declares_lexical` is false for member
                // targets (the parser only produces `Member` with
                // `VarKind::Var`, for bare `for (a.b in/of ...)`), so this
                // arm never runs. Fail closed rather than miscompile.
                ForLhs::Member(_) => {
                    return Err(CompileError::Unsupported("for-in/of lhs pattern"));
                }
            };
            vec![Param {
                pat,
                span: Span { start: 0, end: 0 },
            }]
        } else {
            Vec::new()
        };
        let mut enclosing_labels: Vec<String> = Vec::new();
        for fr in &self.fns.last().expect("no active function").loops {
            for l in &fr.labels {
                if !enclosing_labels.contains(l) {
                    enclosing_labels.push(l.clone());
                }
            }
        }
        let (arrow_idx, unresolved) =
            self.compile_extracted_body(&params, std::slice::from_ref(body))?;
        for l in &unresolved {
            if !enclosing_labels.contains(l) {
                return Err(CompileError::InvalidLhs);
            }
        }
        let body_fn = self.hidden_temp();
        let comp = self.hidden_temp();
        let val = self.hidden_temp();
        // Replay the head scope declared by the declare walk (it holds the
        // loop variable declaration; the arrow's parameter shadows it).
        self.push_block_scope();
        self.compile_expr(rhs)?;
        if !is_of {
            self.call_intrinsic("enum_keys", 1)?;
        }
        self.emit(Op::PushInt(0));
        self.emit(Op::NewFunction(arrow_idx));
        let body_fn_nc = self.binding_name_const(&body_fn)?;
        self.emit(Op::StoreName(body_fn_nc));
        let comp_nc = self.binding_name_const(&comp)?;
        let val_nc = self.binding_name_const(&val)?;
        // Same iterator-close obligations as the inline path (spec
        // 14.7.5): the wrapper owns the loop, so the guard, the
        // `Protected::ForOf` region, and the disarm all live here. `break`
        // / `return` inside the extracted body arrive as tagged completions
        // and are dispatched through `unwind_protected` below.
        let for_of_region = if is_of {
            self.emit(Op::IterGuard);
            let region = self.fns.last().expect("no active function").protected.len();
            self.fns
                .last_mut()
                .expect("no active function")
                .protected
                .push(Protected::ForOf);
            Some(region)
        } else {
            None
        };
        let frame = self.push_loop(
            if is_of {
                LoopKind::ForOf
            } else {
                LoopKind::Loop
            },
            Vec::new(),
            self.protected_len(),
        );
        if let Some(region) = for_of_region {
            self.fns.last_mut().expect("no active function").loops[frame].for_of_region =
                Some(region);
        }
        let l_head = self.code_len();
        self.emit(Op::IterNext); // [it, i+1, value, done]
        let done_jump = self.emit_jump(Op::JumpIfTrue(0));
        // [it, i+1, value]
        self.emit(Op::StoreName(val_nc)); // [it, i+1]
        if !declares_lexical {
            // Store the value to the wrapper-side target.
            self.emit(Op::LoadName(val_nc));
            match lhs {
                ForLhs::Ident(n, _) => {
                    if kind == VarKind::Var {
                        self.compile_store(n)?;
                    } else {
                        self.compile_store_decl(n, kind)?;
                    }
                }
                ForLhs::Member(e) => {
                    let t = self.compile_target_begin(e)?;
                    self.compile_target_store_pop(&t)?;
                }
                // `var`/bare destructuring (`for (var {a} of ...)`,
                // `for ([b] of ...)`): per-iteration assignment shape
                // (spec `PutValue`, with-aware), like the non-extracted
                // path; the value is already on the stack from
                // `LoadName(val_nc)` above.
                ForLhs::Pat(p) => {
                    self.compile_pat_bind_assign(p)?;
                }
            }
        }
        // $comp = $body_fn(value?)
        self.emit(Op::LoadName(body_fn_nc));
        self.emit(Op::PushUndefined);
        if declares_lexical {
            self.emit(Op::LoadName(val_nc));
        }
        self.emit(Op::Call(if declares_lexical { 1 } else { 0 }));
        self.emit(Op::StoreName(comp_nc));
        self.emit_extracted_dispatch(comp_nc, frame, &[], &enclosing_labels, None)?;
        let l_cont = self.code_len();
        self.patch_continues_to(frame, l_cont)?;
        self.emit_jump_to(l_head)?;
        let l_done = self.code_len();
        self.patch_to(done_jump, l_done)?;
        if is_of {
            self.emit(Op::IterUnguard);
        }
        self.emit(Op::Pop);
        self.emit(Op::Pop);
        self.emit(Op::Pop);
        let l_end = self.code_len();
        self.pop_loop_to(frame, l_end)?;
        if for_of_region.is_some() {
            let popped = self
                .fns
                .last_mut()
                .expect("no active function")
                .protected
                .pop();
            debug_assert!(matches!(popped, Some(Protected::ForOf)));
        }
        // body_fn, comp and val are dead after the dispatch loop.
        self.release_temp(&body_fn);
        self.release_temp(&comp);
        self.release_temp(&val);
        self.pop_scope();
        Ok(())
    }

    fn compile_switch(&mut self, disc: &Expr, cases: &[SwitchCase]) -> Result<(), CompileError> {
        self.compile_expr(disc)?;
        let t = self.hidden_temp();
        let t_nc = self.binding_name_const(&t)?;
        self.emit(Op::StoreName(t_nc));
        // Replay the case-clause scope declared by the declare walk.
        self.push_block_scope();
        let labels = std::mem::take(&mut self.pending_labels);
        let depth = self.protected_len();
        let frame = self.push_loop(LoopKind::Switch, labels, depth);
        // Dispatch: jump to the first matching case body (or default).
        let mut jumps: Vec<(usize, usize)> = Vec::new();
        let mut default_idx = None;
        for (i, c) in cases.iter().enumerate() {
            match &c.test {
                Some(test) => {
                    self.emit(Op::LoadName(t_nc));
                    self.compile_expr(test)?;
                    self.emit(Op::StrictEq);
                    jumps.push((self.emit_jump(Op::JumpIfTrue(0)), i));
                }
                None => default_idx = Some(i),
            }
        }
        let no_match = self.emit_jump(Op::Jump(0));
        // The discriminant temp is dead once dispatch is emitted; case
        // bodies may reuse its slot.
        self.release_temp(&t);
        // Bodies fall through in order.
        let mut default_patch = None;
        for (i, c) in cases.iter().enumerate() {
            if Some(i) == default_idx {
                default_patch = Some(self.code_len());
            }
            // Patch this case's dispatch jumps directly; no per-case
            // allocation.
            let body_pc = self.code_len();
            for (pos, ci) in jumps.iter() {
                if *ci == i {
                    self.patch_to(*pos, body_pc)?;
                }
            }
            self.compile_stmts(&c.body)?;
        }
        match default_patch {
            Some(dp) => self.patch_to(no_match, dp)?,
            None => self.patch_to(no_match, self.code_len())?,
        }
        let end = self.code_len();
        self.pop_loop_to(frame, end)?;
        self.pop_scope();
        Ok(())
    }

    fn find_loop(&self, label: &Option<String>, for_continue: bool) -> Option<usize> {
        let f = self.fns.last().expect("no active function");
        for (i, fr) in f.loops.iter().enumerate().rev() {
            let hit = match label {
                None => {
                    if for_continue {
                        matches!(fr.kind, LoopKind::Loop | LoopKind::ForOf)
                    } else {
                        matches!(fr.kind, LoopKind::Loop | LoopKind::ForOf | LoopKind::Switch)
                    }
                }
                Some(l) => fr.labels.iter().any(|x| x == l),
            };
            if hit {
                if for_continue && !matches!(fr.kind, LoopKind::Loop | LoopKind::ForOf) {
                    return None; // `continue` of a labeled block/switch.
                }
                return Some(i);
            }
        }
        None
    }

    fn compile_break(&mut self, label: &Option<String>) -> Result<(), CompileError> {
        let (extracted, extracted_loop) = {
            let f = self.fns.last().expect("no active function");
            (f.extracted, f.extracted_loop)
        };
        let i = match self.find_loop(label, false) {
            Some(i) => i,
            None => {
                // In an extracted body, an unresolved label targets an
                // enclosing construct; unwind and let the wrapper dispatch.
                if extracted && let Some(l) = label {
                    self.unwind_protected(0, true, None)?;
                    self.fns
                        .last_mut()
                        .expect("no active function")
                        .unresolved_labels
                        .push(l.clone());
                    return self.emit_tagged_label(4, l);
                }
                return Err(CompileError::InvalidLhs);
            }
        };
        let depth = self.fns.last().expect("no active function").loops[i].protected_depth;
        self.unwind_protected(depth, true, None)?;
        if extracted && i == extracted_loop {
            return self.emit_tagged_return(1);
        }
        let pos = self.emit_jump(Op::Jump(0));
        self.fns.last_mut().expect("no active function").loops[i]
            .break_patches
            .push(pos);
        Ok(())
    }

    fn compile_continue(&mut self, label: &Option<String>) -> Result<(), CompileError> {
        let (extracted, extracted_loop) = {
            let f = self.fns.last().expect("no active function");
            (f.extracted, f.extracted_loop)
        };
        let i = match self.find_loop(label, true) {
            Some(i) => i,
            None => {
                // In an extracted body, an unresolved label targets an
                // enclosing construct; unwind and let the wrapper dispatch.
                if extracted && let Some(l) = label {
                    self.unwind_protected(0, true, None)?;
                    self.fns
                        .last_mut()
                        .expect("no active function")
                        .unresolved_labels
                        .push(l.clone());
                    return self.emit_tagged_label(5, l);
                }
                return Err(CompileError::InvalidLhs);
            }
        };
        // `continue` does not exit the target loop, so its own `for-of`
        // region is skipped: the iterator stays open. Inner `for-of`
        // loops (being exited) still close.
        let fr = &self.fns.last().expect("no active function").loops[i];
        let depth = fr.protected_depth;
        let skip = fr.for_of_region;
        self.unwind_protected(depth, true, skip)?;
        if extracted && i == extracted_loop {
            return self.emit_tagged_return(2);
        }
        let pos = self.emit_jump(Op::Jump(0));
        self.fns.last_mut().expect("no active function").loops[i]
            .continue_patches
            .push(pos);
        Ok(())
    }

    // -- try --------------------------------------------------------------

    /// Emit unwinding code for the protected regions above `depth`:
    /// `finally` bodies are always inlined, and [`Op::LeaveTry`] is emitted
    /// for `try` regions only when `leave_try` is set. `return` and `throw`
    /// must not emit `LeaveTry`: a return discards the whole frame (and its
    /// handlers) while a throw relies on the registered handler to catch it.
    /// `break`/`continue` stay in the function, so they do deregister.
    /// The regions stay on the compile-time stack: the enclosing `try`
    /// still owns its entries and pops them itself, so an abrupt statement
    /// inside a `try` block must not disturb them.
    /// Unwind `protected` regions above `depth`, emitting cleanup code for
    /// an abrupt exit. `skip_for_of` names a [`Protected::ForOf`] region
    /// to leave alone: `continue` targeting a `for-of` loop itself does
    /// not exit the loop, so its iterator stays open.
    fn unwind_protected(
        &mut self,
        depth: usize,
        leave_try: bool,
        skip_for_of: Option<usize>,
    ) -> Result<(), CompileError> {
        let cur = self.fns.len() - 1;
        let mut i = self.fns[cur].protected.len();
        while i > depth {
            i -= 1;
            if matches!(self.fns[cur].protected[i], Protected::Try) {
                if leave_try {
                    self.emit(Op::LeaveTry);
                }
                continue;
            }
            if matches!(self.fns[cur].protected[i], Protected::ForOf) {
                if skip_for_of == Some(i) {
                    continue;
                }
                // `IteratorClose` on abrupt exit through a `for-of` loop
                // (spec 14.7.5). Stack shape at every call site: `break`
                // and `continue` (`leave_try == true`) run at a statement
                // boundary with `[.., iterable, state]` on top;
                // `return` (`leave_try == false`) runs with the return
                // value pushed, `[.., iterable, state, retval]`. The op
                // closes the iterator (`return?.()`), then pops the pair
                // (and the loop's `Op::IterGuard` guard), so the next
                // region's pair is on top for its own close.
                self.emit(if leave_try {
                    Op::IterClose
                } else {
                    Op::IterCloseRet
                });
                continue;
            }
            // `Finally`: inline the body with scope/cursor management (see
            // `inline_finally`), saving and restoring the scope cursor:
            // the inline runs mid-try-block, before the cursor has reached
            // the finalizer's slots. A region already being inlined is
            // skipped: an abrupt exit from inside a `finally` body must not
            // re-emit that same body (it would recurse forever). The guard
            // is per-region, so a `try`/`finally` nested inside the
            // `finally` body still gets its own finalizer inlined for
            // abrupt exits.
            let cursor_before = self.fns[cur].scope_cursor;
            let r = self.inline_finally(i);
            // Re-fetch: a failed inline may leave functions on the stack.
            let cur2 = self.fns.len() - 1;
            self.fns[cur2].scope_cursor = cursor_before;
            r?;
        }
        Ok(())
    }

    /// Inline a `finally` region's body, replaying the try-statement's
    /// scopes from their declare-walk slots. The declare walk records the
    /// finalizer's scopes after the try block's scopes, so when this runs
    /// for an abrupt exit *inside* the try block (via
    /// [`Compiler::unwind_protected`]) the live
    /// [`FnState::scope_cursor`] has not yet reached the finalizer's slots:
    /// the cursor is repositioned to the recorded
    /// [`FnState::finally_bases`] slot before compiling. The scope stack is
    /// normalized to `[..., outer]` as well: the finalizer's lexicals live
    /// in the try-statement (outer) scope, so compiling against the try
    /// block's live scopes would resolve its names to the wrong bindings.
    /// The cursor is left advanced past the finalizer's scopes (it is NOT
    /// restored): callers inlining mid-try-block (`unwind_protected`) must
    /// save/restore it themselves, while `compile_try`'s normal path wants
    /// the advance so the catch scopes (declared after the finalizer's)
    /// replay correctly.
    fn inline_finally(&mut self, region_idx: usize) -> Result<(), CompileError> {
        // `cur` is captured because a failed nested compile may leave
        // functions on the stack; the whole compile is aborted then.
        let cur = self.fns.len() - 1;
        let (stmts_ptr, stmts_len, try_base, finally_base, outer_depth) =
            match &mut self.fns[cur].protected[region_idx] {
                Protected::Finally {
                    stmts_ptr,
                    stmts_len,
                    inlining,
                    try_base,
                    finally_base,
                    outer_depth,
                } => {
                    if *inlining {
                        return Ok(());
                    }
                    *inlining = true;
                    (
                        *stmts_ptr,
                        *stmts_len,
                        *try_base,
                        *finally_base,
                        *outer_depth,
                    )
                }
                Protected::Try => unreachable!("inline_finally on a Try region"),
                Protected::ForOf => unreachable!("inline_finally on a ForOf region"),
            };
        // Normalize the scope stack to `[..., outer]`: when inlining for an
        // abrupt exit the try block's scopes are still live above the
        // outer scope, so drop them (saving them for restore); when
        // inlining on `compile_try`'s normal/abrupt paths the outer scope
        // was already popped, so replay it by slot instead.
        let tail = if self.scopes.len() >= outer_depth {
            Some(self.scopes.split_off(outer_depth))
        } else {
            self.push_saved_scope(try_base);
            None
        };
        self.fns[cur].scope_cursor = finally_base;
        // SAFETY: `stmts_ptr`/`stmts_len` come from the finalizer's `&[Stmt]`
        // in the AST, which outlives compilation and is never mutated while
        // the compiler runs (see the field docs).
        let stmts = unsafe { std::slice::from_raw_parts(stmts_ptr, stmts_len) };
        let r = self.compile_stmts(stmts);
        match tail {
            Some(tail) => {
                self.scopes.truncate(outer_depth);
                self.scopes.extend(tail);
            }
            None => self.pop_scope(),
        }
        match &mut self.fns[cur].protected[region_idx] {
            Protected::Finally { inlining, .. } => *inlining = false,
            Protected::Try => unreachable!("inline_finally on a Try region"),
            Protected::ForOf => unreachable!("inline_finally on a ForOf region"),
        }
        r
    }

    fn compile_try(
        &mut self,
        block: &[Stmt],
        handler: Option<&CatchClause>,
        finalizer: Option<&[Stmt]>,
    ) -> Result<(), CompileError> {
        // The declare walk pushes scopes in pre-order: outer (owns the
        // finalizer's lexicals), inner (the try block's lexicals), then any
        // nested scopes from the block and finalizer, then (if a handler is
        // present) the catch-param scope and catch-body scope. Codegen
        // replays them in the same order via the cursor. The finalizer is
        // compiled several times (normal path, abrupt paths); each inline
        // goes through `inline_finally`, which replays the outer scope by
        // explicit slot and resets the cursor so every copy reuses the same
        // nested slots.
        let base = self.fns.last().expect("no active function").scope_cursor;
        // The declare walk recorded where this finalizer's scopes begin
        // (see `FnState::finally_bases`); abrupt exits inside the try block
        // need it before the try block has finished compiling.
        let finally_base = finalizer.map(|fin| {
            *self
                .fns
                .last()
                .expect("no active function")
                .finally_bases
                .get(&fin.as_ptr())
                .expect("declare walk must have recorded the finalizer's base")
        });
        let hpos = self.emit_jump(Op::EnterTry(0));
        self.push_block_scope(); // outer
        // `inline_finally` truncates the scope stack to this depth when
        // inlining for an abrupt exit, so the finalizer resolves against
        // the outer scope rather than the try block's live scopes.
        let outer_depth = self.scopes.len();
        self.push_block_scope(); // inner
        if let Some(fin) = finalizer {
            self.fns
                .last_mut()
                .expect("no active function")
                .protected
                .push(Protected::Finally {
                    stmts_ptr: fin.as_ptr(),
                    stmts_len: fin.len(),
                    inlining: false,
                    try_base: base,
                    finally_base: finally_base.expect("finalizer is_some"),
                    outer_depth,
                });
        }
        self.fns
            .last_mut()
            .expect("no active function")
            .protected
            .push(Protected::Try);
        let r = self.compile_stmts(block);
        self.pop_scope(); // inner
        self.pop_scope(); // outer
        r?;
        let popped = self
            .fns
            .last_mut()
            .expect("no active function")
            .protected
            .pop();
        debug_assert!(matches!(popped, Some(Protected::Try)));
        self.emit(Op::LeaveTry);
        // Normal path: run the finalizer inline.
        if finalizer.is_some() {
            let cur = self.fns.len() - 1;
            let idx = self.fns[cur].protected.len() - 1;
            self.inline_finally(idx)?;
        }
        let end = self.emit_jump(Op::Jump(0));
        // Handler: [thrown].
        self.patch_handler(hpos)?;
        match handler {
            Some(c) => {
                self.push_block_scope(); // catch param
                match &c.param {
                    Some(p) => self.compile_pat_bind(&p.pat, VarKind::Let)?,
                    None => self.emit(Op::Pop),
                }
                self.push_block_scope(); // catch body
                let r = self.compile_stmts(&c.body);
                self.pop_scope(); // body
                self.pop_scope(); // param
                r?;
            }
            None => {
                // try/finally: run the finalizer, then rethrow.
                let t = self.hidden_temp();
                let t_nc = self.binding_name_const(&t)?;
                self.emit(Op::StoreName(t_nc));
                if finalizer.is_some() {
                    let cur = self.fns.len() - 1;
                    let idx = self.fns[cur].protected.len() - 1;
                    self.inline_finally(idx)?;
                }
                self.emit(Op::LoadName(t_nc));
                self.emit(Op::Throw);
                // The thrown-value temp is dead after the rethrow.
                self.release_temp(&t);
            }
        }
        // Abrupt path with a catch: run the finalizer inline, then continue.
        // Save/restore the scope cursor around the inline: the catch-param
        // and catch-body scopes were already replayed (+2) above, and
        // `inline_finally` unconditionally resets `scope_cursor` to
        // `finally_base` and leaves it at `finally_base + k`, which would
        // discard that advance and corrupt every later `push_block_scope`.
        if handler.is_some() && finalizer.is_some() {
            let cur = self.fns.len() - 1;
            let idx = self.fns[cur].protected.len() - 1;
            let cursor_before = self.fns[cur].scope_cursor;
            self.inline_finally(idx)?;
            let cur2 = self.fns.len() - 1;
            self.fns[cur2].scope_cursor = cursor_before;
        }
        self.patch(end)?;
        // Pop the `Finally` region pushed above (only when a finalizer
        // exists; a bare try/catch pushes just the `Try` region).
        if finalizer.is_some() {
            let popped = self
                .fns
                .last_mut()
                .expect("no active function")
                .protected
                .pop();
            debug_assert!(matches!(popped, Some(Protected::Finally { .. })));
        }
        Ok(())
    }

    // -- modules ----------------------------------------------------------

    fn compile_export(&mut self, e: &ExportDecl) -> Result<(), CompileError> {
        match e {
            ExportDecl::All {
                source, exported, ..
            } => {
                self.module.exports.push(ExportEntry::All {
                    source: source.clone(),
                    exported: exported.clone(),
                });
                self.emit(Op::Nop);
            }
            ExportDecl::Named {
                specifiers, source, ..
            } => {
                for s in specifiers {
                    let local = module_name_string(&s.local);
                    let exported = module_name_string(&s.exported);
                    self.module.exports.push(ExportEntry::Named {
                        local: if source.is_some() { None } else { Some(local) },
                        exported,
                        source: source.clone(),
                    });
                }
                self.emit(Op::Nop);
            }
            ExportDecl::Default { expr, .. } => {
                self.compile_expr(expr)?;
                let nc = self.binding_name_const("#default_export")?;
                self.emit(Op::StoreName(nc));
                self.module.exports.push(ExportEntry::Default);
            }
            ExportDecl::Decl { inner, .. } => {
                let names = bound_names(inner);
                self.compile_stmt(inner)?;
                self.module.exports.push(ExportEntry::Decl(names));
            }
        }
        Ok(())
    }
}

fn module_name_string(n: &ModuleName) -> String {
    match n {
        ModuleName::Ident(s, _) => s.clone(),
        // Module specifiers feed URL resolution (`&str`): lossy.
        ModuleName::Str(s, _) => s.to_utf8_lossy().into_owned(),
    }
}

/// Names bound by a declaration statement (for `export <decl>` records).
fn bound_names(s: &Stmt) -> Vec<String> {
    let mut out = Vec::new();
    fn pat_names(p: &Pat, out: &mut Vec<String>) {
        match p {
            Pat::Ident(n, _) => out.push(n.clone()),
            Pat::Array(elems, _) => {
                for el in elems {
                    if let Some(ArrayPatElem::Pat(p)) = el {
                        pat_names(p, out);
                    }
                }
            }
            Pat::Object(props, _) => {
                for pr in props {
                    match pr {
                        ObjectPatProp::KeyValue { value, .. } => pat_names(value, out),
                        ObjectPatProp::Shorthand { name, .. } => out.push(name.clone()),
                        ObjectPatProp::Rest(p, _) => pat_names(p, out),
                    }
                }
            }
            Pat::Rest(p, _) | Pat::Assign(p, _, _) => pat_names(p, out),
            // Member targets bind nothing.
            Pat::Member(..) => {}
        }
    }
    match s {
        Stmt::VarDecl { decls, .. } => {
            for d in decls {
                pat_names(&d.id, &mut out);
            }
        }
        Stmt::FunctionDecl { name, .. } => out.push(name.clone()),
        Stmt::ClassDecl { name: Some(n), .. } => out.push(n.clone()),
        Stmt::ClassDecl { name: None, .. } => {}
        _ => {}
    }
    out
}

// ---------------------------------------------------------------------------
// Expressions (`[..] -> [.., value]`)
// ---------------------------------------------------------------------------

/// An assignment target with its object/key pre-evaluated into temps.
enum Target {
    Ident(String),
    Member {
        t_obj: String,
        /// `Some(const)` for statically-known keys, else the key temp name.
        key_const: Option<u16>,
        t_key: String,
        /// Private brand for `"#name"` keys; `None` for ordinary keys.
        private_brand: Option<u16>,
    },
    /// `super.prop` target: temps holding receiver (`this`), base (the
    /// home object's prototype, or the base constructor for static
    /// members), and the key (a const or a temp).
    Super {
        t_recv: String,
        t_base: String,
        key_const: Option<u16>,
        t_key: String,
    },
}

impl Compiler {
    fn compile_expr(&mut self, e: &Expr) -> Result<(), CompileError> {
        self.guarded(|s| s.compile_expr_inner(e))
    }

    fn compile_expr_inner(&mut self, e: &Expr) -> Result<(), CompileError> {
        match e {
            Expr::Ident(name, _) => self.compile_load(name),
            Expr::PrivateIdent(_, _) => Err(CompileError::Unsupported("stray private identifier")),
            Expr::Lit(l, _) => self.compile_lit(l),
            Expr::Array(elems, _) => {
                self.emit(Op::NewArray(0));
                for el in elems {
                    match el {
                        None => {
                            self.emit(Op::PushUndefined);
                            self.call_intrinsic("arr_append", 2)?;
                        }
                        Some(Expr::Spread(inner, _)) => {
                            self.compile_expr(inner)?;
                            self.emit(Op::ArrAppendSpread);
                        }
                        Some(x) => {
                            self.compile_expr(x)?;
                            self.call_intrinsic("arr_append", 2)?;
                        }
                    }
                }
                Ok(())
            }
            Expr::Object(props, _) => self.compile_object(props),
            Expr::Function {
                name,
                params,
                body,
                is_async,
                is_generator,
                strict,
                ..
            } => {
                let n = name.clone().unwrap_or_else(|| "#anon".to_string());
                let t = self.compile_function_template(
                    n,
                    params,
                    body,
                    false,
                    *is_async,
                    *is_generator,
                    *strict,
                    false,
                    None,
                    // Named function expression: bind the function's own
                    // name in its scope (ECMA-262 `NamedEvaluation`).
                    name.as_deref(),
                )?;
                let i = self.intern_template(t)?;
                self.emit(Op::NewFunction(i));
                Ok(())
            }
            Expr::Class {
                name,
                superclass,
                body,
                ..
            } => self.compile_class_value(name.clone(), superclass.as_deref(), body, true),
            Expr::Arrow {
                params,
                body,
                is_async,
                strict,
                ..
            } => {
                let stmts: Vec<Stmt> = match body {
                    ArrowBody::Expr(x) => vec![Stmt::Return(Some(x.as_ref().clone()))],
                    ArrowBody::Block(ss) => ss.clone(),
                };
                let t = self.compile_function_template(
                    "#arrow".to_string(),
                    params,
                    &stmts,
                    true,
                    *is_async,
                    false,
                    *strict,
                    false,
                    None,
                    None,
                )?;
                let i = self.intern_template(t)?;
                self.emit(Op::NewFunction(i));
                Ok(())
            }
            Expr::Call { .. } | Expr::New { .. } | Expr::Member { .. } => self.compile_chain(e),
            Expr::Unary { op, arg, .. } => self.compile_unary(*op, arg),
            Expr::Update {
                op, arg, prefix, ..
            } => self.compile_update(*op, arg, *prefix),
            Expr::Binary {
                op, left, right, ..
            } => {
                // `#x in obj` is a brand check, not a property lookup.
                if *op == BinaryOp::In
                    && let Expr::PrivateIdent(n, _) = left.as_ref()
                {
                    let brand = self
                        .current_private_brand()
                        .ok_or(CompileError::Unsupported("private name outside class"))?;
                    let name = self.intern(&format!("#{n}"))?;
                    self.compile_expr(right)?; // [obj]
                    self.emit(Op::HasPrivate(brand, name)); // [bool]
                    return Ok(());
                }
                self.compile_expr(left)?;
                self.compile_expr(right)?;
                self.emit(match op {
                    BinaryOp::Eq => Op::Eq,
                    BinaryOp::Ne => Op::Ne,
                    BinaryOp::StrictEq => Op::StrictEq,
                    BinaryOp::StrictNe => Op::StrictNe,
                    BinaryOp::Lt => Op::Lt,
                    BinaryOp::LtEq => Op::Le,
                    BinaryOp::Gt => Op::Gt,
                    BinaryOp::GtEq => Op::Ge,
                    BinaryOp::LShift => Op::Shl,
                    BinaryOp::RShift => Op::Shr,
                    BinaryOp::URShift => Op::ShrU,
                    BinaryOp::Add => Op::Add,
                    BinaryOp::Sub => Op::Sub,
                    BinaryOp::Mul => Op::Mul,
                    BinaryOp::Div => Op::Div,
                    BinaryOp::Mod => Op::Mod,
                    BinaryOp::Pow => Op::Pow,
                    BinaryOp::BitAnd => Op::BitAnd,
                    BinaryOp::BitOr => Op::BitOr,
                    BinaryOp::BitXor => Op::BitXor,
                    // `[key, obj]`
                    BinaryOp::In => Op::In,
                    BinaryOp::Instanceof => Op::Instanceof,
                });
                Ok(())
            }
            Expr::Logical {
                op, left, right, ..
            } => self.compile_logical(*op, left, right),
            Expr::Assign {
                op, left, right, ..
            } => self.compile_assign(*op, left, right),
            Expr::Conditional {
                cond,
                then_e,
                else_e,
                ..
            } => {
                self.compile_expr(cond)?;
                let l_else = self.emit_jump(Op::JumpIfFalse(0));
                self.compile_expr(then_e)?;
                let l_end = self.emit_jump(Op::Jump(0));
                self.patch(l_else)?;
                self.compile_expr(else_e)?;
                self.patch(l_end)?;
                Ok(())
            }
            Expr::Sequence(exprs, _) => {
                if exprs.is_empty() {
                    self.emit(Op::PushUndefined);
                    return Ok(());
                }
                let mut it = exprs.iter().peekable();
                while let Some(x) = it.next() {
                    self.compile_expr(x)?;
                    if it.peek().is_some() {
                        self.emit(Op::Pop);
                    }
                }
                Ok(())
            }
            Expr::Template { parts, .. } => {
                let mut first = true;
                for p in parts {
                    match p {
                        TemplatePart::Str { cooked, raw, .. } => {
                            let i = match cooked {
                                Some(c) => self.intern_wtf8(c.as_wtf8())?,
                                None => self.intern(raw)?,
                            };
                            self.emit(Op::LoadConst(i));
                        }
                        TemplatePart::Expr(x) => self.compile_expr(x)?,
                    }
                    if !first {
                        // The VM implements JS `+` (string concat / numeric add).
                        self.emit(Op::Add);
                    }
                    first = false;
                }
                if first {
                    let i = self.intern("")?;
                    self.emit(Op::LoadConst(i));
                }
                Ok(())
            }
            Expr::TaggedTemplate { tag, parts, .. } => {
                self.compile_expr(tag)?;
                self.emit(Op::PushUndefined);
                // Cooked strings array.
                self.emit(Op::NewArray(0));
                let mut argc: u8 = 1;
                for p in parts {
                    if let TemplatePart::Str { cooked, raw, .. } = p {
                        let i = match cooked {
                            Some(c) => self.intern_wtf8(c.as_wtf8())?,
                            None => self.intern(raw)?,
                        };
                        self.emit(Op::LoadConst(i));
                        self.call_intrinsic("arr_append", 2)?;
                    }
                }
                for p in parts {
                    if let TemplatePart::Expr(x) = p {
                        self.compile_expr(x)?;
                        argc = argc
                            .checked_add(1)
                            .ok_or(CompileError::Unsupported("too many tag arguments"))?;
                    }
                }
                self.emit(Op::Call(argc));
                Ok(())
            }
            Expr::Yield { arg, delegate, .. } => {
                if *delegate {
                    // `yield*` requires an operand.
                    match arg {
                        Some(x) => self.compile_expr(x)?,
                        None => return Err(CompileError::InvalidLhs),
                    }
                    self.emit(Op::YieldStar);
                } else {
                    match arg {
                        Some(x) => self.compile_expr(x)?,
                        None => self.emit(Op::PushUndefined),
                    }
                    self.emit(Op::Yield);
                }
                Ok(())
            }
            Expr::Await(x, _) => {
                self.compile_expr(x)?;
                self.emit(Op::Await);
                Ok(())
            }
            Expr::Spread(_, _) => Err(CompileError::Unsupported("stray spread element")),
            Expr::This(_) => self.compile_this(),
            Expr::Super(_) => Err(CompileError::Unsupported("stray super")),
            Expr::NewTarget(_) => self.compile_new_target(),
            Expr::ImportMeta(_) => self.call_intrinsic("import_meta", 0),
            Expr::Regex { pattern, flags, .. } => {
                // Spec early error: validate the literal at compile time so
                // parse-phase negative tests see the SyntaxError before
                // evaluation. Uses the same compiler the runtime's
                // `Op::NewRegex` uses, so a pattern accepted here is
                // accepted there.
                crate::regex_engine::CompiledRegex::compile(Wtf8::from_str(pattern), flags)
                    .map_err(|e| CompileError::InvalidRegex(format!("{e}")))?;
                let p = self.intern(pattern)?;
                let f = self.intern(flags)?;
                self.emit(Op::LoadConst(p));
                self.emit(Op::LoadConst(f));
                self.emit(Op::NewRegex);
                Ok(())
            }
        }
    }

    fn compile_lit(&mut self, l: &Literal) -> Result<(), CompileError> {
        match l {
            Literal::Number(v) => self.compile_number(*v),
            Literal::BigInt(raw) => {
                let i = self.intern(raw)?;
                self.emit(Op::LoadConst(i));
                self.call_intrinsic("bigint", 1)
            }
            Literal::Str(s) => {
                let i = self.intern_wtf8(s.as_wtf8())?;
                self.emit(Op::LoadConst(i));
                Ok(())
            }
            Literal::Bool(b) => {
                self.emit(if *b { Op::PushTrue } else { Op::PushFalse });
                Ok(())
            }
            Literal::Null => {
                self.emit(Op::PushNull);
                Ok(())
            }
        }
    }

    fn compile_number(&mut self, v: f64) -> Result<(), CompileError> {
        if v.fract() == 0.0 && (-128.0..=127.0).contains(&v) {
            self.emit(Op::PushInt(v as i8));
        } else if v.fract() == 0.0 && (i32::MIN as f64..=i32::MAX as f64).contains(&v) {
            let i = self.intern_int(v as i32)?;
            self.emit(Op::LoadConst(i));
        } else {
            let i = self.intern_float(v)?;
            self.emit(Op::LoadConst(i));
        }
        Ok(())
    }

    /// The private brand of the lexically enclosing class: the innermost
    /// [`ClassCtx::brand`].
    fn current_private_brand(&self) -> Option<u16> {
        self.fns
            .last()
            .expect("no active function")
            .class_stack
            .last()
            .map(|c| c.brand)
    }

    /// The class context governing a `super` reference: the innermost
    /// [`ClassCtx`]. Static blocks compile in their own strict function,
    /// which never lives lexically inside the class context, so they
    /// deliberately do *not* see the class: `super` in a static block is
    /// a compile error (as the spec requires).
    fn current_super_ctx(&self) -> Option<&ClassCtx> {
        self.fns
            .last()
            .expect("no active function")
            .class_stack
            .last()
    }

    /// Compile `this`. Arrows load the captured `#this` cell.
    fn compile_this(&mut self) -> Result<(), CompileError> {
        if self.fns.last().expect("no active function").is_arrow {
            // Arrows capture `this` lexically.
            match self.resolve("#this")? {
                Resolved::Local(nc, _) => self.emit(Op::LoadName(nc)),
                Resolved::Upvalue(i, _) => self.emit(Op::LoadUpvalue(i)),
                Resolved::Global(_) => unreachable!("#this is always declared"),
            }
        } else {
            self.emit(Op::This);
        }
        Ok(())
    }

    fn compile_new_target(&mut self) -> Result<(), CompileError> {
        if self.fns.last().expect("no active function").is_arrow {
            match self.resolve("#newtarget")? {
                Resolved::Local(nc, _) => self.emit(Op::LoadName(nc)),
                Resolved::Upvalue(i, _) => self.emit(Op::LoadUpvalue(i)),
                Resolved::Global(_) => unreachable!("#newtarget is always declared"),
            }
        } else {
            self.emit(Op::NewTarget);
        }
        Ok(())
    }

    fn current_super_name(&mut self) -> Result<String, CompileError> {
        self.fns
            .last()
            .expect("no active function")
            .class_stack
            .last()
            .map(|c| c.super_name.clone())
            .filter(|s| !s.is_empty())
            .ok_or(CompileError::Unsupported("super outside a derived class"))
    }

    // -- calls ------------------------------------------------------------

    /// Compile a member/call/`new` chain iteratively.
    ///
    /// The old recursive compilation walked the callee spine with one
    /// `compile_expr` frame per link, so a 3000-deep `f.bind().bind()…`
    /// chain blew past [`MAX_COMPILE_DEPTH`] (and, worse, the native-stack
    /// guard) with an uncatchable `DepthExceeded`. This instead flattens
    /// the spine into an explicit heap-allocated link vector in one pass,
    /// compiles the innermost base once, then applies the links
    /// innermost-first: O(1) compiler stack regardless of chain depth.
    ///
    /// Any optional link bails out to [`Compiler::compile_optional_chain`]
    /// with the original expression, exactly as before.
    fn compile_chain(&mut self, e: &Expr) -> Result<(), CompileError> {
        // Single optionality scan. `is_opt_chain` walks the whole unvisited
        // spine, so calling it once per link made deep chains O(n^2) (e.g. a
        // 3000-deep `.bind()` nesting). The union of the old per-link checks
        // (`*optional` here plus `is_opt_chain` of the inner spine) is
        // exactly "some link on the spine is optional", so one upfront check
        // routes to `compile_optional_chain` identically. (An optional link
        // inside a `new` callee is not on this spine: `is_opt_chain` does not
        // descend into `New`, so that case is still handled in the loop.)
        if is_opt_chain(e) {
            return self.compile_optional_chain(e);
        }
        // Flatten the spine. `links[0]` is the outermost link.
        let mut links: Vec<ChainLink<'_>> = Vec::new();
        // Assigned on every `break` below; the loop only exits via `break`
        // or `return`.
        let base: ChainBase<'_>;
        let mut cur = e;
        loop {
            match cur {
                Expr::Call { callee, args, .. } => {
                    match callee.as_ref() {
                        // `import(x)` is dynamic import. One or two
                        // arguments: the optional second is the options
                        // object (`import(x, { with: ... })`), evaluated
                        // for side effects and ignored by the op (v1).
                        Expr::Ident(n, _) if n == "import" => {
                            if args.is_empty() || args.len() > 2 {
                                return Err(CompileError::InvalidLhs);
                            }
                            links.push(ChainLink::DynamicImport { args });
                            base = ChainBase::Import;
                            break;
                        }
                        // A bare `eval(...)` identifier call is a *direct*
                        // eval: the VM needs the syntactic form to pick
                        // direct (caller-context) semantics. Aliases
                        // (`var e = eval; e(x)`), comma expressions
                        // (`(0, eval)(x)`), and member calls
                        // (`globalThis.eval(x)`) compile to ordinary calls
                        // and run with indirect (global) semantics.
                        Expr::Ident(n, _) if n == "eval" => {
                            links.push(ChainLink::Call {
                                args,
                                direct_eval: true,
                            });
                            cur = callee.as_ref();
                        }
                        Expr::Super(_) => {
                            links.push(ChainLink::SuperCall { args });
                            base = ChainBase::Super;
                            break;
                        }
                        Expr::Member {
                            obj,
                            prop,
                            computed,
                            ..
                        } => {
                            // `super.m(...)`: the function lives on the
                            // heritage prototype (or on the base
                            // constructor when static); `this` is the
                            // current instance. Implemented via the same
                            // SuperGet path as a plain `super.m` read.
                            if matches!(obj.as_ref(), Expr::Super(_)) {
                                links.push(ChainLink::SuperMethodCall {
                                    prop: prop.as_ref(),
                                    computed: *computed,
                                    args,
                                });
                                base = ChainBase::Super;
                                break;
                            }
                            links.push(ChainLink::MethodCall {
                                prop: prop.as_ref(),
                                computed: *computed,
                                args,
                            });
                            cur = obj.as_ref();
                        }
                        _ => {
                            links.push(ChainLink::Call {
                                args,
                                direct_eval: false,
                            });
                            cur = callee.as_ref();
                        }
                    }
                }
                Expr::Member {
                    obj,
                    prop,
                    computed,
                    ..
                } => {
                    links.push(ChainLink::Member {
                        prop: prop.as_ref(),
                        computed: *computed,
                    });
                    cur = obj.as_ref();
                }
                Expr::New { callee, args, .. } => {
                    if is_opt_chain(callee) {
                        // The optional chain is in the `new` callee (e.g.
                        // `new (a?.b)()`); compile the callee (which
                        // handles the optional short-circuit), then apply
                        // the `new`. `compile_optional_chain` cannot
                        // handle an outermost `New`.
                        self.compile_expr(callee)?;
                        self.compile_new_args(args)?;
                        return Ok(());
                    }
                    links.push(ChainLink::New { args });
                    cur = callee.as_ref();
                }
                _ => {
                    base = ChainBase::Expr(cur);
                    break;
                }
            }
        }
        // Apply the links innermost-first (`links` was pushed
        // outermost-first, so iterate in reverse).
        //
        // `with`-aware call callee: `with (o) { f() }` invokes `f` with
        // `this === o` when the with-object provides `f` (the callee
        // reference's base is the object environment record). When the
        // innermost link is a plain `Call` of an identifier that probes a
        // with-object, the base emits `[func, this]` and the link skips
        // its `PushUndefined`.
        let with_call_this: Option<Vec<u16>> = match (&base, links.last()) {
            (
                ChainBase::Expr(Expr::Ident(name, _)),
                Some(ChainLink::Call {
                    direct_eval: false, ..
                }),
            ) => {
                let probes = self.with_probes(name);
                if probes.is_empty() {
                    None
                } else {
                    Some(probes)
                }
            }
            _ => None,
        };
        let mut with_this_pending = with_call_this.is_some();
        let mut rev = links.iter().rev();
        match base {
            ChainBase::Import => {
                // The innermost link is the dynamic import itself; it
                // compiles its own argument, so no base value is emitted.
                match rev.next() {
                    Some(ChainLink::DynamicImport { args }) => {
                        self.compile_expr(&args[0])?;
                        if args.len() == 2 {
                            // Options object: evaluate (side effects are
                            // real) and leave it for the op to pop and
                            // ignore.
                            self.compile_expr(&args[1])?;
                        } else {
                            self.emit(Op::PushUndefined);
                        }
                        self.emit(Op::DynamicImport);
                    }
                    _ => return Err(CompileError::InvalidLhs),
                }
            }
            ChainBase::Super => {
                // The innermost link is the `super(...)` call or
                // `super.m(...)` method call; it loads its own receiver.
                match rev.next() {
                    Some(ChainLink::SuperCall { args }) => self.compile_super_call(args)?,
                    Some(ChainLink::SuperMethodCall {
                        prop,
                        computed,
                        args,
                    }) => {
                        self.compile_super_prop_load(prop, *computed)?;
                        // [fn]
                        self.compile_this()?;
                        // [fn, this]
                        self.compile_call_args(args, false)?;
                    }
                    _ => return Err(CompileError::InvalidLhs),
                }
            }
            ChainBase::Expr(b) => {
                if matches!(b, Expr::Super(_)) {
                    // `super.x…`: the innermost link consumes the super
                    // base through the SuperGet path; no base value is
                    // compiled. Anything else on a bare `super` keeps the
                    // old "stray super" error.
                    match rev.next() {
                        Some(ChainLink::Member { prop, computed }) => {
                            self.compile_super_prop_load(prop, *computed)?;
                        }
                        _ => self.compile_expr(b)?,
                    }
                } else if let (Expr::Ident(name, _), Some(probes)) = (b, &with_call_this) {
                    // `[func, this]`: probe each with-object (innermost
                    // first); on a hit `this` is the with-object, else
                    // the ordinary resolved access with `undefined` this.
                    let key = self.intern(name)?;
                    let mut ends = Vec::with_capacity(probes.len());
                    for wnc in probes {
                        self.emit(Op::LoadConst(key)); // [k]
                        self.emit(Op::LoadName(*wnc)); // [k, w]
                        self.emit(Op::In); // [hit]
                        let miss = self.emit_jump(Op::JumpIfFalse(0));
                        self.emit(Op::LoadName(*wnc)); // [w]
                        self.emit(Op::LoadConst(key)); // [w, k]
                        self.emit(Op::GetElem); // [func]
                        self.emit(Op::LoadName(*wnc)); // [func, this=w]
                        ends.push(self.emit_jump(Op::Jump(0)));
                        self.patch(miss)?;
                    }
                    self.compile_load_inner(name)?; // [func]
                    self.emit(Op::PushUndefined); // [func, undefined]
                    for e in ends {
                        self.patch(e)?;
                    }
                } else {
                    self.compile_expr(b)?;
                }
            }
        }
        for link in rev {
            match link {
                ChainLink::Call { args, direct_eval } => {
                    // [callee] -> [callee, this_arg, ...args]: `Op::Call`
                    // and `Op::EvalCall` both consume
                    // `[callee, this_arg, args…]`. A `with`-aware callee
                    // already pushed `[func, this]`; only the innermost
                    // `Call` may consume it.
                    if with_this_pending {
                        with_this_pending = false;
                    } else {
                        self.emit(Op::PushUndefined);
                    }
                    self.compile_call_args(args, *direct_eval)?;
                }
                ChainLink::MethodCall {
                    prop,
                    computed,
                    args,
                } => {
                    // [obj] -> [fn, obj]: keep the receiver for the call.
                    self.emit(Op::Dup);
                    self.compile_prop_load(prop, *computed)?;
                    // [obj, fn]
                    self.emit(Op::Swap); // [fn, obj]
                    self.compile_call_args(args, false)?;
                }
                ChainLink::Member { prop, computed } => {
                    self.compile_prop_load(prop, *computed)?;
                }
                ChainLink::New { args } => {
                    self.compile_new_args(args)?;
                }
                // Import/super links only ever sit innermost (handled above).
                ChainLink::DynamicImport { .. }
                | ChainLink::SuperCall { .. }
                | ChainLink::SuperMethodCall { .. } => {
                    return Err(CompileError::InvalidLhs);
                }
            }
        }
        Ok(())
    }

    /// Stack: `[callee, this_arg] -> [result]`. Spread arguments go through
    /// [`Op::SpreadCall`]; `new` with spread through `#construct`.
    fn compile_call_args(&mut self, args: &[Expr], direct_eval: bool) -> Result<(), CompileError> {
        if direct_eval {
            // A direct `eval(...)` reads and writes its lexical scope chain
            // through names known only at runtime: force-capture every
            // in-scope binding as an upvalue now, so the VM can resolve
            // eval names against the caller frame's captured cells.
            self.capture_eval_scope()?;
        }
        if args.iter().any(|a| matches!(a, Expr::Spread(..))) {
            self.emit(Op::NewArray(0));
            for a in args {
                match a {
                    Expr::Spread(inner, _) => {
                        self.compile_expr(inner)?;
                        self.emit(Op::ArrAppendSpread);
                    }
                    _ => {
                        self.compile_expr(a)?;
                        self.call_intrinsic("arr_append", 2)?;
                    }
                }
            }
            // [callee, this, args_array]
            if direct_eval {
                self.emit(Op::EvalCallSpread);
                Ok(())
            } else {
                self.emit(Op::SpreadCall);
                Ok(())
            }
        } else {
            if args.len() > 255 {
                return Err(CompileError::Unsupported("too many call arguments"));
            }
            for a in args {
                self.compile_expr(a)?;
            }
            self.emit(if direct_eval {
                Op::EvalCall(args.len() as u8)
            } else {
                Op::Call(args.len() as u8)
            });
            Ok(())
        }
    }

    /// Compile an optional chain (`a?.b.c()`, `a?.[k]?.()`).
    ///
    /// The chain is flattened into links. Only a link explicitly marked
    /// optional (`?.`, `?.()`) emits a nullish check: when its base value
    /// is nullish the whole chain short-circuits to `undefined` via one
    /// shared tail. A non-optional link after an optional one performs
    /// ordinary property access, which throws a `TypeError` on a
    /// nullish base (`a?.b.c` throws when `a.b` is nullish; only
    /// `a?.b?.c` short-circuits). Each check precedes key/argument
    /// evaluation, so short-circuiting skips all later side effects.
    /// Member calls keep their receiver via a hidden temp: `[fn]` is
    /// nullish-checked while the receiver waits in the temp, then
    /// `[fn, this]` is rebuilt for the call.
    fn compile_optional_chain(&mut self, e: &Expr) -> Result<(), CompileError> {
        enum Link<'a> {
            Member {
                prop: &'a Expr,
                computed: bool,
                optional: bool,
            },
            Call {
                args: &'a [Expr],
                optional: bool,
            },
        }
        impl Link<'_> {
            fn is_optional(&self) -> bool {
                match self {
                    Link::Member { optional, .. } | Link::Call { optional, .. } => *optional,
                }
            }
        }
        let mut links: Vec<Link<'_>> = Vec::new();
        let mut base = e;
        loop {
            match base {
                Expr::Member {
                    obj,
                    prop,
                    computed,
                    optional,
                    ..
                } => {
                    if matches!(obj.as_ref(), Expr::Super(_)) {
                        return Err(CompileError::Unsupported("optional chaining on super"));
                    }
                    links.push(Link::Member {
                        prop,
                        computed: *computed,
                        optional: *optional,
                    });
                    base = obj;
                }
                Expr::Call {
                    callee,
                    args,
                    optional,
                    ..
                } => {
                    if matches!(callee.as_ref(), Expr::Super(_)) {
                        return Err(CompileError::Unsupported("optional chaining on super"));
                    }
                    if let Expr::Ident(n, _) = callee.as_ref()
                        && n == "import"
                    {
                        return Err(CompileError::Unsupported("optional dynamic import"));
                    }
                    links.push(Link::Call {
                        args,
                        optional: *optional,
                    });
                    base = callee;
                }
                _ => break,
            }
        }
        links.reverse();
        debug_assert!(
            links.iter().any(Link::is_optional),
            "chain compiler called without an optional link"
        );
        if matches!(base, Expr::Super(_)) {
            return Err(CompileError::Unsupported("optional chaining on super"));
        }

        self.compile_expr(base)?;
        let mut jumps: Vec<usize> = Vec::new();
        let mut i = 0;
        while i < links.len() {
            match &links[i] {
                Link::Member { prop, computed, .. } => {
                    // Fuse `obj.prop(args)`: the receiver must survive the
                    // callee's nullish check. The check before the property
                    // load belongs to this member link; the check on the
                    // loaded function belongs to the call link.
                    if let Some(Link::Call { args, .. }) = links.get(i + 1) {
                        if links[i].is_optional() {
                            jumps.push(self.emit_jump(Op::JumpIfNullish(0)));
                        }
                        self.emit(Op::Dup);
                        self.compile_prop_load(prop, *computed)?;
                        // [obj, fn]
                        self.emit(Op::Swap); // [fn, obj]
                        let t = self.hidden_temp();
                        let nc = self.binding_name_const(&t)?;
                        self.emit(Op::StoreName(nc)); // [fn]
                        if links[i + 1].is_optional() {
                            jumps.push(self.emit_jump(Op::JumpIfNullish(0)));
                        }
                        self.emit(Op::LoadName(nc)); // [fn, obj]
                        self.release_temp(&t);
                        let args: &[Expr] = args;
                        self.compile_call_args(args, false)?;
                        i += 2;
                        continue;
                    }
                    if links[i].is_optional() {
                        jumps.push(self.emit_jump(Op::JumpIfNullish(0)));
                    }
                    self.compile_prop_load(prop, *computed)?;
                }
                Link::Call { args, .. } => {
                    if links[i].is_optional() {
                        jumps.push(self.emit_jump(Op::JumpIfNullish(0)));
                    }
                    self.emit(Op::PushUndefined);
                    let args: &[Expr] = args;
                    self.compile_call_args(args, false)?;
                }
            }
            i += 1;
        }
        // Success path jumps over the shared nullish tail; without this
        // the successful result would fall through into `PushUndefined`.
        let done = self.emit_jump(Op::Jump(0));
        for j in jumps {
            self.patch(j)?;
        }
        self.emit(Op::PushUndefined);
        self.patch(done)?;
        Ok(())
    }

    /// Stack: `[callee] -> [instance]`: compile `new` arguments and emit
    /// construction. Spread arguments go through `#construct`.
    fn compile_new_args(&mut self, args: &[Expr]) -> Result<(), CompileError> {
        if args.iter().any(|a| matches!(a, Expr::Spread(..))) {
            self.emit(Op::NewArray(0));
            for a in args {
                match a {
                    Expr::Spread(inner, _) => {
                        self.compile_expr(inner)?;
                        self.emit(Op::ArrAppendSpread);
                    }
                    _ => {
                        self.compile_expr(a)?;
                        self.call_intrinsic("arr_append", 2)?;
                    }
                }
            }
            // [callee, args_array]
            self.call_intrinsic("construct", 2)
        } else {
            if args.len() > 255 {
                return Err(CompileError::Unsupported("too many call arguments"));
            }
            for a in args {
                self.compile_expr(a)?;
            }
            self.emit(Op::New(args.len() as u8));
            Ok(())
        }
    }

    /// `super(...args)` inside a derived constructor: initializes `#this`
    /// via `#super_construct` and leaves the instance on the stack.
    fn compile_super_call(&mut self, args: &[Expr]) -> Result<(), CompileError> {
        let sn = self.current_super_name()?;
        self.emit(Op::NewArray(0));
        for a in args {
            match a {
                Expr::Spread(inner, _) => {
                    self.compile_expr(inner)?;
                    self.emit(Op::ArrAppendSpread);
                }
                _ => {
                    self.compile_expr(a)?;
                    self.call_intrinsic("arr_append", 2)?;
                }
            }
        }
        // [args]; want [super_ctor, new_target, args].
        self.compile_load(&sn)?;
        self.emit(Op::Swap);
        self.compile_new_target()?;
        self.emit(Op::Swap);
        self.call_intrinsic("super_construct", 3)?;
        // Store the instance as this frame's `#this`.
        let nc = self.binding_name_const("#this")?;
        self.emit(Op::Dup);
        self.emit(Op::StoreName(nc));
        Ok(())
    }

    // -- member access ------------------------------------------------------

    /// `[obj] -> [value]`.
    fn compile_prop_load(&mut self, prop: &Expr, computed: bool) -> Result<(), CompileError> {
        if !computed {
            match prop {
                Expr::Ident(n, _) => {
                    let i = self.intern(n)?;
                    self.emit(Op::GetPropConst(i));
                    return Ok(());
                }
                Expr::PrivateIdent(n, _) => {
                    let brand = self
                        .current_private_brand()
                        .ok_or(CompileError::Unsupported(
                            "private member access outside a class",
                        ))?;
                    let i = self.intern(&format!("#{n}"))?;
                    self.emit(Op::GetPrivate(brand, i));
                    return Ok(());
                }
                _ => {}
            }
        }
        self.compile_expr(prop)?;
        self.emit(Op::GetElem);
        Ok(())
    }

    /// `[receiver, base, key] -> [value]`: `super.prop` read. `receiver`
    /// is the current `this`; `base` is the home object's prototype —
    /// the superclass's `.prototype` for instance members, or the
    /// superclass constructor itself for static members.
    fn compile_super_prop_load(&mut self, prop: &Expr, computed: bool) -> Result<(), CompileError> {
        self.compile_this()?;
        let sn = self.current_super_name()?;
        self.compile_load(&sn)?;
        let is_static = self
            .current_super_ctx()
            .map(|c| c.is_static)
            .unwrap_or(false);
        if !is_static {
            let pi = self.intern("prototype")?;
            self.emit(Op::GetPropConst(pi));
        }
        match (computed, prop) {
            (false, Expr::Ident(n, _)) => {
                let i = self.intern(n)?;
                self.emit(Op::LoadConst(i));
            }
            _ => {
                self.compile_expr(prop)?;
            }
        }
        self.emit(Op::SuperGet);
        Ok(())
    }

    /// `[obj, obj, value] -> [obj]` for object literals / method definitions.
    fn emit_keyed_store(&mut self, key: &PropKey) -> Result<(), CompileError> {
        match key {
            PropKey::Ident(s, _) => {
                let i = self.intern(s)?;
                self.emit(Op::SetPropConst(i));
            }
            PropKey::Str(s) => {
                let i = self.intern_wtf8(s)?;
                self.emit(Op::SetPropConst(i));
            }
            _ => {
                self.compile_key(key)?;
                self.emit(Op::Swap);
                self.emit(Op::SetElem);
            }
        }
        Ok(())
    }

    /// Push a property key value: `[..] -> [.., key]`.
    fn compile_key(&mut self, key: &PropKey) -> Result<(), CompileError> {
        match key {
            PropKey::Ident(s, _) => {
                let i = self.intern(s)?;
                self.emit(Op::LoadConst(i));
            }
            PropKey::Str(s) => {
                let i = self.intern_wtf8(s)?;
                self.emit(Op::LoadConst(i));
            }
            PropKey::Num(f, _) => self.compile_number(*f)?,
            PropKey::Computed(e) => self.compile_expr(e)?,
        }
        Ok(())
    }

    /// Push a class member key value at class-definition time:
    /// `[..] -> [.., key]`. Computed member keys are strict mode code
    /// (spec 10.2.1); evaluate them in an immediately-invoked strict
    /// wrapper (see [`Self::compile_strict_expr_wrapper`]) so sloppy
    /// outer code still gets strict runtime semantics
    /// (`class C { [y = 2]() {} }` must throw ReferenceError instead of
    /// creating a global).
    fn compile_class_key(&mut self, key: &PropKey) -> Result<(), CompileError> {
        match key {
            PropKey::Computed(e) => self.compile_strict_expr_wrapper(e),
            _ => self.compile_key(key),
        }
    }

    fn compile_object(&mut self, props: &[PropDef]) -> Result<(), CompileError> {
        self.emit(Op::NewObject);
        // Accessors are grouped by key so getter+setter share one
        // `#define_accessor` call.
        let mut accessors: Vec<Accessor> = Vec::new();
        for prop in props {
            match prop {
                PropDef::KeyValue { key, value, .. } => {
                    self.emit(Op::Dup);
                    self.compile_expr(value)?;
                    self.emit_keyed_store(key)?;
                }
                PropDef::Shorthand { name, .. } => {
                    self.emit(Op::Dup);
                    self.compile_load(name)?;
                    let i = self.intern(name)?;
                    self.emit(Op::SetPropConst(i));
                }
                PropDef::Spread(e, _) => {
                    self.compile_expr(e)?;
                    self.call_intrinsic("obj_spread", 2)?;
                }
                PropDef::Method {
                    key,
                    params,
                    body,
                    is_async,
                    is_generator,
                    strict,
                    ..
                } => {
                    let t = self.compile_function_template(
                        prop_key_name(key),
                        params,
                        body,
                        false,
                        *is_async,
                        *is_generator,
                        *strict,
                        true,
                        None,
                        None,
                    )?;
                    let ti = self.intern_template(t)?;
                    self.emit(Op::Dup);
                    self.emit(Op::NewFunction(ti));
                    self.emit_keyed_store_fn(key)?;
                }
                PropDef::Getter {
                    key, body, strict, ..
                } => {
                    let t = self.compile_function_template(
                        prop_key_name(key),
                        &[],
                        body,
                        false,
                        false,
                        false,
                        *strict,
                        true,
                        None,
                        None,
                    )?;
                    let ti = self.intern_template(t)?;
                    accessor_push(&mut accessors, key.clone(), Some(ti), None);
                }
                PropDef::Setter {
                    key,
                    param,
                    body,
                    strict,
                    ..
                } => {
                    let t = self.compile_function_template(
                        prop_key_name(key),
                        std::slice::from_ref(param),
                        body,
                        false,
                        false,
                        false,
                        *strict,
                        true,
                        None,
                        None,
                    )?;
                    let ti = self.intern_template(t)?;
                    accessor_push(&mut accessors, key.clone(), None, Some(ti));
                }
            }
        }
        for a in &accessors {
            // [obj]; Dup; key; getter|undef; setter|undef; #define_accessor; Pop.
            self.emit(Op::Dup);
            self.compile_key(&a.key)?;
            match a.getter {
                Some(ti) => self.emit(Op::NewFunction(ti)),
                None => self.emit(Op::PushUndefined),
            }
            match a.setter {
                Some(ti) => self.emit(Op::NewFunction(ti)),
                None => self.emit(Op::PushUndefined),
            }
            self.call_intrinsic("define_accessor", 4)?;
            self.emit(Op::Pop);
        }
        Ok(())
    }

    /// `[obj, obj, fn] -> [obj]`: store a freshly built method function.
    fn emit_keyed_store_fn(&mut self, key: &PropKey) -> Result<(), CompileError> {
        match key {
            PropKey::Ident(s, _) => {
                let i = self.intern(s)?;
                self.emit(Op::SetPropConst(i));
            }
            PropKey::Str(s) => {
                let i = self.intern_wtf8(s)?;
                self.emit(Op::SetPropConst(i));
            }
            _ => {
                self.compile_key(key)?;
                // [o, o, f, k] -> Swap -> [o, o, k, f] -> SetProp -> [o]
                self.emit(Op::Swap);
                self.emit(Op::SetProp);
            }
        }
        Ok(())
    }

    // -- unary / update -------------------------------------------------------

    fn compile_unary(&mut self, op: UnaryOp, arg: &Expr) -> Result<(), CompileError> {
        match op {
            UnaryOp::Minus => {
                self.compile_expr(arg)?;
                self.emit(Op::Neg);
            }
            UnaryOp::Plus => {
                self.compile_expr(arg)?;
                self.emit(Op::Pos);
            }
            UnaryOp::Bang => {
                self.compile_expr(arg)?;
                self.emit(Op::Not);
            }
            UnaryOp::Tilde => {
                self.compile_expr(arg)?;
                self.emit(Op::BitNot);
            }
            UnaryOp::Typeof => match arg {
                // `typeof <ident>` on an undeclared name yields "undefined"
                // (spec 13.5.3); any other operand keeps Throw-on-missing.
                Expr::Ident(name, _) => self.compile_typeof_name(name)?,
                _ => {
                    self.compile_expr(arg)?;
                    self.emit(Op::Typeof);
                }
            },
            UnaryOp::Void => {
                self.compile_expr(arg)?;
                self.emit(Op::Void);
            }
            UnaryOp::Delete => match arg {
                Expr::Member {
                    obj,
                    prop,
                    computed,
                    optional,
                    ..
                } => {
                    // The base may itself be an optional chain
                    // (`delete a?.b.c`): compile it link-by-link so that
                    // only an actual `?.` short-circuit yields `true`. A
                    // non-optional link evaluates normally, so a nullish
                    // base throws through the ordinary property-access /
                    // delete path.
                    let mut jumps: Vec<usize> = Vec::new();
                    self.compile_delete_base(obj, &mut jumps)?;
                    if *optional {
                        jumps.push(self.emit_jump(Op::JumpIfNullish(0)));
                    }
                    if *computed {
                        self.compile_expr(prop)?;
                    } else {
                        match prop.as_ref() {
                            Expr::Ident(n, _) => {
                                let i = self.intern(n)?;
                                self.emit(Op::LoadConst(i));
                            }
                            Expr::PrivateIdent(_, _) => {
                                return Err(CompileError::Unsupported("delete of private field"));
                            }
                            _ => self.compile_expr(prop)?,
                        }
                    }
                    self.emit(Op::Delete);
                    if !jumps.is_empty() {
                        let done = self.emit_jump(Op::Jump(0));
                        for j in jumps {
                            self.patch(j)?;
                        }
                        // `JumpIfNullish` popped the nullish base.
                        self.emit(Op::PushTrue);
                        self.patch(done)?;
                    }
                }
                Expr::Ident(name, _) => {
                    // `delete x`: probe enclosing `with` objects like the
                    // store path — a hit deletes the with-object's
                    // property (spec: the object environment record's
                    // `DeleteBinding`). A miss keeps the pre-existing
                    // no-op-`true` behavior. (Strict-mode `delete x` is a
                    // parse-time `SyntaxError`, so no strict check here.)
                    let probes = self.with_probes(name);
                    if probes.is_empty() {
                        self.emit(Op::PushTrue);
                    } else {
                        let key = self.intern(name)?;
                        let mut ends = Vec::with_capacity(probes.len());
                        for wnc in probes {
                            self.emit(Op::LoadConst(key)); // [k]
                            self.emit(Op::LoadName(wnc)); // [k, w]
                            self.emit(Op::In); // [hit]
                            let miss = self.emit_jump(Op::JumpIfFalse(0));
                            self.emit(Op::LoadName(wnc)); // [w]
                            self.emit(Op::LoadConst(key)); // [w, k]
                            self.emit(Op::Delete); // [deleted]
                            ends.push(self.emit_jump(Op::Jump(0)));
                            self.patch(miss)?;
                        }
                        self.emit(Op::PushTrue);
                        for e in ends {
                            self.patch(e)?;
                        }
                    }
                }
                _ => {
                    // `delete <non-ref>`: no observable effect.
                    self.compile_expr(arg)?;
                    self.emit(Op::Pop);
                    self.emit(Op::PushTrue);
                }
            },
        }
        Ok(())
    }

    /// Compile the base object of a `delete` member target.
    ///
    /// Member/call links on the spine are compiled inside-out. A link
    /// explicitly marked `?.` emits a nullish check (recorded in `jumps`;
    /// the caller patches them to a tail that yields `true`), so an
    /// actual short-circuit deletes to `true`. Non-optional links compile
    /// exactly like ordinary evaluation, so a nullish base throws through
    /// the normal property-access path. Anything off the spine (computed
    /// keys, arguments, the ultimate base) goes through `compile_expr`
    /// and is self-contained.
    fn compile_delete_base(
        &mut self,
        e: &Expr,
        jumps: &mut Vec<usize>,
    ) -> Result<(), CompileError> {
        match e {
            Expr::Member {
                obj,
                prop,
                computed,
                optional,
                ..
            } => {
                if matches!(obj.as_ref(), Expr::Super(_)) {
                    return Err(CompileError::Unsupported("optional chaining on super"));
                }
                self.compile_delete_base(obj, jumps)?;
                if *optional {
                    jumps.push(self.emit_jump(Op::JumpIfNullish(0)));
                }
                self.compile_prop_load(prop, *computed)?;
            }
            Expr::Call {
                callee,
                args,
                optional,
                ..
            } => {
                if let Expr::Ident(n, _) = callee.as_ref()
                    && n == "import"
                {
                    return Err(CompileError::Unsupported("optional dynamic import"));
                }
                match callee.as_ref() {
                    Expr::Member {
                        obj,
                        prop,
                        computed,
                        optional: mopt,
                        ..
                    } => {
                        if matches!(obj.as_ref(), Expr::Super(_)) {
                            return Err(CompileError::Unsupported("optional chaining on super"));
                        }
                        // Fuse `obj.prop(args)` so the receiver survives
                        // the callee's nullish check, mirroring
                        // `compile_optional_chain`.
                        self.compile_delete_base(obj, jumps)?;
                        if *mopt {
                            jumps.push(self.emit_jump(Op::JumpIfNullish(0)));
                        }
                        self.emit(Op::Dup);
                        self.compile_prop_load(prop, *computed)?;
                        // [obj, fn]
                        self.emit(Op::Swap); // [fn, obj]
                        let t = self.hidden_temp();
                        let nc = self.binding_name_const(&t)?;
                        self.emit(Op::StoreName(nc)); // [fn]
                        if *optional {
                            jumps.push(self.emit_jump(Op::JumpIfNullish(0)));
                        }
                        self.emit(Op::LoadName(nc)); // [fn, obj]
                        self.release_temp(&t);
                        self.compile_call_args(args, false)?;
                    }
                    _ => {
                        self.compile_delete_base(callee, jumps)?;
                        if *optional {
                            jumps.push(self.emit_jump(Op::JumpIfNullish(0)));
                        }
                        self.emit(Op::PushUndefined);
                        self.compile_call_args(args, false)?;
                    }
                }
            }
            _ => self.compile_expr(e)?,
        }
        Ok(())
    }

    fn compile_update(
        &mut self,
        op: UpdateOp,
        arg: &Expr,
        prefix: bool,
    ) -> Result<(), CompileError> {
        let t = self.compile_target_begin(arg)?;
        self.compile_target_load(&t)?;
        if prefix {
            self.emit(Op::PushInt(1));
            self.emit(if op == UpdateOp::Inc {
                Op::Add
            } else {
                Op::Sub
            });
            self.compile_target_store(&t)?; // [new]
        } else {
            self.emit(Op::Dup);
            self.emit(Op::PushInt(1));
            self.emit(if op == UpdateOp::Inc {
                Op::Add
            } else {
                Op::Sub
            });
            self.compile_target_store_pop(&t)?; // [old]
        }
        Ok(())
    }

    // -- logical / assignment ---------------------------------------------------

    fn compile_logical(
        &mut self,
        op: LogicalOp,
        left: &Expr,
        right: &Expr,
    ) -> Result<(), CompileError> {
        match op {
            LogicalOp::And => {
                self.compile_expr(left)?;
                self.emit(Op::Dup);
                let end = self.emit_jump(Op::JumpIfFalse(0));
                self.emit(Op::Pop);
                self.compile_expr(right)?;
                self.patch(end)?;
            }
            LogicalOp::Or => {
                self.compile_expr(left)?;
                self.emit(Op::Dup);
                let end = self.emit_jump(Op::JumpIfTrue(0));
                self.emit(Op::Pop);
                self.compile_expr(right)?;
                self.patch(end)?;
            }
            LogicalOp::Nullish => {
                self.compile_expr(left)?; // [v]
                self.emit(Op::Dup); // [v, v]
                self.emit(Op::PushUndefined);
                self.emit(Op::StrictEq); // [v, v===undefined]
                let is_null = self.emit_jump(Op::JumpIfTrue(0));
                self.emit(Op::Dup); // [v, v]
                self.emit(Op::PushNull);
                self.emit(Op::StrictEq); // [v, v===null]
                let is_null2 = self.emit_jump(Op::JumpIfTrue(0));
                let end = self.emit_jump(Op::Jump(0));
                self.patch(is_null)?;
                self.patch(is_null2)?;
                // [v]
                self.emit(Op::Pop);
                self.compile_expr(right)?;
                self.patch(end)?;
            }
        }
        Ok(())
    }

    fn compile_assign(
        &mut self,
        op: AssignOp,
        left: &Expr,
        right: &Expr,
    ) -> Result<(), CompileError> {
        if matches!(op, AssignOp::Assign) && matches!(left, Expr::Array(..) | Expr::Object(..)) {
            // Destructuring assignment (`[a] = rhs`, `({a} = rhs)`):
            // patterns are only valid simple-assignment targets, so
            // compound operators keep falling through to
            // `compile_target_begin` (which rejects them with
            // `InvalidLhs`).
            return self.compile_destructuring_assign(left, right);
        }
        let t = self.compile_target_begin(left)?;
        match op {
            AssignOp::Assign => {
                self.compile_expr(right)?;
            }
            AssignOp::Add
            | AssignOp::Sub
            | AssignOp::Mul
            | AssignOp::Div
            | AssignOp::Mod
            | AssignOp::Pow
            | AssignOp::LShift
            | AssignOp::RShift
            | AssignOp::URShift
            | AssignOp::BitAnd
            | AssignOp::BitOr
            | AssignOp::BitXor => {
                self.compile_target_load(&t)?;
                self.compile_expr(right)?;
                self.emit(match op {
                    AssignOp::Add => Op::Add,
                    AssignOp::Sub => Op::Sub,
                    AssignOp::Mul => Op::Mul,
                    AssignOp::Div => Op::Div,
                    AssignOp::Mod => Op::Mod,
                    AssignOp::Pow => Op::Pow,
                    AssignOp::LShift => Op::Shl,
                    AssignOp::RShift => Op::Shr,
                    AssignOp::URShift => Op::ShrU,
                    AssignOp::BitAnd => Op::BitAnd,
                    AssignOp::BitOr => Op::BitOr,
                    _ => Op::BitXor,
                });
            }
            AssignOp::And => {
                self.compile_target_load(&t)?;
                self.emit(Op::Dup);
                let end = self.emit_jump(Op::JumpIfFalse(0));
                self.emit(Op::Pop);
                self.compile_expr(right)?;
                self.patch(end)?;
            }
            AssignOp::Or => {
                self.compile_target_load(&t)?;
                self.emit(Op::Dup);
                let end = self.emit_jump(Op::JumpIfTrue(0));
                self.emit(Op::Pop);
                self.compile_expr(right)?;
                self.patch(end)?;
            }
            AssignOp::Nullish => {
                self.compile_target_load(&t)?;
                self.emit(Op::Dup);
                self.emit(Op::PushUndefined);
                self.emit(Op::StrictEq);
                let is_null = self.emit_jump(Op::JumpIfTrue(0));
                self.emit(Op::Dup);
                self.emit(Op::PushNull);
                self.emit(Op::StrictEq);
                let is_null2 = self.emit_jump(Op::JumpIfTrue(0));
                let end = self.emit_jump(Op::Jump(0));
                self.patch(is_null)?;
                self.patch(is_null2)?;
                self.emit(Op::Pop);
                self.compile_expr(right)?;
                self.patch(end)?;
            }
        }
        // [value] -> [value]
        self.compile_target_store(&t)
    }

    /// Conservative syntactic check: can evaluating `rhs` rebind the
    /// variable `name`? Used by [`Compiler::compile_str_append_stmt`] to
    /// pick the fused-append shape.
    ///
    /// Direct assignments and updates to `name` anywhere in the rhs
    /// (e.g. `s += (s = "Z", "c")`) count, as do nested function, class,
    /// and arrow literals — their bodies could run if the rhs invokes
    /// them. Ordinary calls do *not* count: rebinding through a callee's
    /// body (e.g. `s += (f(), "c")` where `f` assigns `s`) would need
    /// interprocedural effect analysis, so it keeps the historic fused
    /// behavior. That is a known, pre-existing deviation, not a
    /// regression.
    fn rhs_may_rebind(rhs: &Expr, name: &str) -> bool {
        match rhs {
            Expr::Ident(..)
            | Expr::PrivateIdent(..)
            | Expr::Lit(..)
            | Expr::This(..)
            | Expr::Super(..)
            | Expr::NewTarget(..)
            | Expr::ImportMeta(..)
            | Expr::Regex { .. } => false,
            // A nested scope's body runs only if the rhs invokes it —
            // which it may. Conservatively assume it rebinds.
            Expr::Function { .. } | Expr::Class { .. } | Expr::Arrow { .. } => true,
            Expr::Assign { left, right, .. } => {
                matches!(left.as_ref(), Expr::Ident(n, _) if n == name)
                    || Self::rhs_may_rebind(left, name)
                    || Self::rhs_may_rebind(right, name)
            }
            Expr::Update { arg, .. } => {
                matches!(arg.as_ref(), Expr::Ident(n, _) if n == name)
                    || Self::rhs_may_rebind(arg, name)
            }
            Expr::Array(items, ..) => items
                .iter()
                .flatten()
                .any(|e| Self::rhs_may_rebind(e, name)),
            Expr::Object(props, ..) => props.iter().any(|p| Self::prop_may_rebind(p, name)),
            Expr::Call { callee, args, .. } => {
                Self::rhs_may_rebind(callee, name)
                    || args.iter().any(|a| Self::rhs_may_rebind(a, name))
            }
            Expr::New { callee, args, .. } => {
                Self::rhs_may_rebind(callee, name)
                    || args.iter().any(|a| Self::rhs_may_rebind(a, name))
            }
            Expr::Member { obj, prop, .. } => {
                Self::rhs_may_rebind(obj, name) || Self::rhs_may_rebind(prop, name)
            }
            Expr::Unary { arg, .. } => Self::rhs_may_rebind(arg, name),
            Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
                Self::rhs_may_rebind(left, name) || Self::rhs_may_rebind(right, name)
            }
            Expr::Conditional {
                cond,
                then_e,
                else_e,
                ..
            } => {
                Self::rhs_may_rebind(cond, name)
                    || Self::rhs_may_rebind(then_e, name)
                    || Self::rhs_may_rebind(else_e, name)
            }
            Expr::Sequence(exprs, ..) => exprs.iter().any(|e| Self::rhs_may_rebind(e, name)),
            Expr::Template { parts, .. } => parts.iter().any(|p| match p {
                TemplatePart::Expr(e) => Self::rhs_may_rebind(e, name),
                TemplatePart::Str { .. } => false,
            }),
            Expr::TaggedTemplate { tag, parts, .. } => {
                Self::rhs_may_rebind(tag, name)
                    || parts.iter().any(|p| match p {
                        TemplatePart::Expr(e) => Self::rhs_may_rebind(e, name),
                        TemplatePart::Str { .. } => false,
                    })
            }
            Expr::Yield { arg, .. } => arg.as_ref().is_some_and(|e| Self::rhs_may_rebind(e, name)),
            Expr::Await(e, ..) | Expr::Spread(e, ..) => Self::rhs_may_rebind(e, name),
        }
    }

    /// [`Compiler::rhs_may_rebind`] over an object-literal property: the
    /// value, spread, and computed keys evaluate during rhs evaluation;
    /// method/getter/setter bodies are nested scopes (conservative).
    fn prop_may_rebind(prop: &PropDef, name: &str) -> bool {
        match prop {
            PropDef::KeyValue { key, value, .. } => {
                Self::prop_key_may_rebind(key, name) || Self::rhs_may_rebind(value, name)
            }
            PropDef::Shorthand { .. } => false,
            PropDef::Spread(e, ..) => Self::rhs_may_rebind(e, name),
            PropDef::Method { .. } | PropDef::Getter { .. } | PropDef::Setter { .. } => true,
        }
    }

    /// [`Compiler::rhs_may_rebind`] over a property key: only a computed
    /// key evaluates.
    fn prop_key_may_rebind(key: &PropKey, name: &str) -> bool {
        match key {
            PropKey::Computed(e) => Self::rhs_may_rebind(e, name),
            PropKey::Ident(..) | PropKey::Str(..) | PropKey::Num(..) => false,
        }
    }

    /// Try to compile `e` — a discarded-value `name += rhs` or
    /// `name = name + rhs` statement — as a fused string append.
    ///
    /// Returns `Ok(true)` when the fused form was emitted (value
    /// discarded); `Ok(false)` means the caller falls back to the
    /// generic compilation. Upvalue targets stay on the slow path (their
    /// storage is a shared cell, not a frame slot).
    ///
    /// Two fused shapes: when the rhs cannot rebind the target, the op
    /// reads the base from the environment slot itself (`[rhs] -> []`) so
    /// an in-progress accumulation is not materialized on every
    /// iteration. When the rhs may assign the target (e.g.
    /// `s += (s = "Z", "c")`), the compiler captures the pre-rhs value
    /// with a load first (`[base, rhs] -> []`,
    /// [`Op::StrAppendBaseName`]), so the append still sees the value the
    /// statement started with, as the spec requires. Conversion-time side
    /// effects (`valueOf` writing `name`) observe the old base in both
    /// shapes.
    fn compile_str_append_stmt(&mut self, e: &Expr) -> Result<bool, CompileError> {
        // Inside a `with` body the fused op's raw `LoadName` would bypass
        // the with-object probes; fall back to the generic (probed)
        // assignment path instead.
        if !self.with_stack.is_empty() {
            return Ok(false);
        }
        // Extract `(name, rhs)` from the two fused shapes.
        let (name, rhs) = match e {
            Expr::Assign {
                op: AssignOp::Add,
                left,
                right,
                ..
            } => match left.as_ref() {
                Expr::Ident(name, _) => (name.clone(), right.as_ref()),
                _ => return Ok(false),
            },
            Expr::Assign {
                op: AssignOp::Assign,
                left,
                right,
                ..
            } => {
                let name = match left.as_ref() {
                    Expr::Ident(name, _) => name,
                    _ => return Ok(false),
                };
                match right.as_ref() {
                    Expr::Binary {
                        op: BinaryOp::Add,
                        left: inner,
                        right: rhs,
                        ..
                    } => match inner.as_ref() {
                        Expr::Ident(inner, _) if inner == name => (name.clone(), rhs.as_ref()),
                        _ => return Ok(false),
                    },
                    _ => return Ok(false),
                }
            }
            _ => return Ok(false),
        };
        // Resolve like `compile_load`: TDZ guard for lexicals, then the
        // fused op over the name constant (the `pop_function` peephole
        // rewrites locals to `StrAppendLocal` / `StrAppendBaseLocal`).
        let nc = match self.resolve_full(&name)? {
            (Resolved::Local(nc, kind), identity) => {
                // The self-name binding is immutable: skip the fused
                // append and fall back to the generic `+=` path, whose
                // store is a silent no-op (sloppy) or throws (strict).
                if kind == BindingKind::SelfName {
                    return Ok(false);
                }
                if kind.is_lexical() {
                    let (decl_fn, id) = identity.expect("lexical binding is declared");
                    self.emit_tdz_check(&name, decl_fn, id)?;
                }
                nc
            }
            (Resolved::Global(nc), _) => nc,
            (Resolved::Upvalue(..), _) => return Ok(false),
        };
        if Self::rhs_may_rebind(rhs, &name) {
            // The rhs may rebind the target: capture the pre-rhs value
            // with a load first, so the append sees what the statement
            // started with (spec: `lval` is read before the rhs runs).
            // The peephole rewrites the load and the append to the local
            // slot forms together.
            self.emit(Op::LoadName(nc));
            self.compile_expr(rhs)?;
            self.compile_const_check(&name)?;
            self.emit(Op::StrAppendBaseName(nc));
        } else {
            self.compile_expr(rhs)?;
            self.compile_const_check(&name)?;
            self.emit(Op::StrAppendName(nc));
        }
        Ok(true)
    }

    // -- assignment targets -----------------------------------------------------

    /// Evaluate `obj`/`key` for a member target into hidden temps.
    /// Returns the target descriptor; the value stack is untouched.
    fn compile_target_begin(&mut self, left: &Expr) -> Result<Target, CompileError> {
        match left {
            Expr::Ident(name, _) => Ok(Target::Ident(name.clone())),
            Expr::Member {
                obj,
                prop,
                computed,
                optional,
                ..
            } => {
                if *optional {
                    return Err(CompileError::Unsupported("optional chaining"));
                }
                if matches!(obj.as_ref(), Expr::Super(_)) {
                    // `super.prop = v`: receiver is the current `this`,
                    // base is the home object's prototype (or the base
                    // constructor for static members).
                    self.compile_this()?;
                    let t_recv = self.hidden_temp();
                    let rnc = self.binding_name_const(&t_recv)?;
                    self.emit(Op::StoreName(rnc));
                    let sn = self.current_super_name()?;
                    self.compile_load(&sn)?;
                    let is_static = self
                        .current_super_ctx()
                        .map(|c| c.is_static)
                        .unwrap_or(false);
                    if !is_static {
                        let pi = self.intern("prototype")?;
                        self.emit(Op::GetPropConst(pi));
                    }
                    let t_base = self.hidden_temp();
                    let bnc = self.binding_name_const(&t_base)?;
                    self.emit(Op::StoreName(bnc));
                    let (key_const, t_key) = if !computed {
                        match prop.as_ref() {
                            Expr::Ident(n, _) => (Some(self.intern(n)?), String::new()),
                            _ => {
                                let t_key = self.hidden_temp();
                                self.compile_expr(prop)?;
                                let knc = self.binding_name_const(&t_key)?;
                                self.emit(Op::StoreName(knc));
                                (None, t_key)
                            }
                        }
                    } else {
                        let t_key = self.hidden_temp();
                        self.compile_expr(prop)?;
                        let knc = self.binding_name_const(&t_key)?;
                        self.emit(Op::StoreName(knc));
                        (None, t_key)
                    };
                    return Ok(Target::Super {
                        t_recv,
                        t_base,
                        key_const,
                        t_key,
                    });
                }
                self.compile_expr(obj)?;
                let t_obj = self.hidden_temp();
                let nc = self.binding_name_const(&t_obj)?;
                self.emit(Op::StoreName(nc));
                let (key_const, t_key, private_brand) = if !computed {
                    match prop.as_ref() {
                        Expr::Ident(n, _) => (Some(self.intern(n)?), String::new(), None),
                        Expr::PrivateIdent(n, _) => {
                            let brand =
                                self.current_private_brand()
                                    .ok_or(CompileError::Unsupported(
                                        "private member access outside a class",
                                    ))?;
                            (
                                Some(self.intern(&format!("#{n}"))?),
                                String::new(),
                                Some(brand),
                            )
                        }
                        _ => {
                            let t_key = self.hidden_temp();
                            self.compile_expr(prop)?;
                            let knc = self.binding_name_const(&t_key)?;
                            self.emit(Op::StoreName(knc));
                            (None, t_key, None)
                        }
                    }
                } else {
                    let t_key = self.hidden_temp();
                    self.compile_expr(prop)?;
                    let knc = self.binding_name_const(&t_key)?;
                    self.emit(Op::StoreName(knc));
                    (None, t_key, None)
                };
                Ok(Target::Member {
                    t_obj,
                    key_const,
                    t_key,
                    private_brand,
                })
            }
            _ => Err(CompileError::InvalidLhs),
        }
    }

    /// `[..] -> [.., value]`.
    fn compile_target_load(&mut self, t: &Target) -> Result<(), CompileError> {
        match t {
            Target::Ident(name) => self.compile_load(name),
            Target::Member {
                t_obj,
                key_const,
                t_key,
                private_brand,
            } => {
                let nc = self.binding_name_const(t_obj)?;
                self.emit(Op::LoadName(nc));
                match key_const {
                    Some(kc) => match private_brand {
                        Some(brand) => self.emit(Op::GetPrivate(*brand, *kc)),
                        None => self.emit(Op::GetPropConst(*kc)),
                    },
                    None => {
                        let knc = self.binding_name_const(t_key)?;
                        self.emit(Op::LoadName(knc));
                        self.emit(Op::GetElem);
                    }
                }
                Ok(())
            }
            Target::Super {
                t_recv,
                t_base,
                key_const,
                t_key,
            } => {
                let rnc = self.binding_name_const(t_recv)?;
                self.emit(Op::LoadName(rnc));
                let bnc = self.binding_name_const(t_base)?;
                self.emit(Op::LoadName(bnc));
                match key_const {
                    Some(kc) => self.emit(Op::LoadConst(*kc)),
                    None => {
                        let knc = self.binding_name_const(t_key)?;
                        self.emit(Op::LoadName(knc));
                    }
                }
                self.emit(Op::SuperGet);
                Ok(())
            }
        }
    }

    /// `[value] -> [value]`: duplicate, store, keep the copy.
    fn compile_target_store(&mut self, t: &Target) -> Result<(), CompileError> {
        match t {
            Target::Ident(name) => {
                self.emit(Op::Dup);
                self.compile_const_check(name)?;
                self.compile_store(name)?;
                Ok(())
            }
            Target::Member {
                t_obj,
                key_const,
                t_key,
                private_brand,
            } => {
                let t_v = self.hidden_temp();
                let vnc = self.binding_name_const(&t_v)?;
                self.emit(Op::StoreName(vnc));
                let onc = self.binding_name_const(t_obj)?;
                self.emit(Op::LoadName(onc));
                match key_const {
                    Some(kc) => {
                        self.emit(Op::LoadName(vnc));
                        match private_brand {
                            Some(brand) => self.emit(Op::SetPrivate(*brand, *kc)),
                            None => self.emit(Op::SetPropConst(*kc)),
                        }
                    }
                    None => {
                        let knc = self.binding_name_const(t_key)?;
                        self.emit(Op::LoadName(knc));
                        self.emit(Op::LoadName(vnc));
                        self.emit(Op::SetElem);
                    }
                }
                self.emit(Op::LoadName(vnc));
                // t_v, t_obj and t_key are dead once the store completes.
                self.release_temp(&t_v);
                self.release_temp(t_obj);
                self.release_temp(t_key);
                Ok(())
            }
            Target::Super {
                t_recv,
                t_base,
                key_const,
                t_key,
            } => {
                // [value] -> [value]: SuperSet takes [receiver, base, key,
                // value], so stash the value first.
                let t_v = self.hidden_temp();
                let vnc = self.binding_name_const(&t_v)?;
                self.emit(Op::StoreName(vnc));
                let rnc = self.binding_name_const(t_recv)?;
                self.emit(Op::LoadName(rnc));
                let bnc = self.binding_name_const(t_base)?;
                self.emit(Op::LoadName(bnc));
                match key_const {
                    Some(kc) => self.emit(Op::LoadConst(*kc)),
                    None => {
                        let knc = self.binding_name_const(t_key)?;
                        self.emit(Op::LoadName(knc));
                    }
                }
                self.emit(Op::LoadName(vnc));
                self.emit(Op::SuperSet);
                self.emit(Op::LoadName(vnc));
                // t_v, t_recv, t_base and t_key are dead once the store
                // completes.
                self.release_temp(&t_v);
                self.release_temp(t_recv);
                self.release_temp(t_base);
                self.release_temp(t_key);
                Ok(())
            }
        }
    }

    /// `[value] -> []`.
    fn compile_target_store_pop(&mut self, t: &Target) -> Result<(), CompileError> {
        match t {
            Target::Ident(name) => {
                self.compile_const_check(name)?;
                self.compile_store(name)
            }
            Target::Member {
                t_obj,
                key_const,
                t_key,
                private_brand,
            } => {
                let t_v = self.hidden_temp();
                let vnc = self.binding_name_const(&t_v)?;
                self.emit(Op::StoreName(vnc));
                let onc = self.binding_name_const(t_obj)?;
                self.emit(Op::LoadName(onc));
                match key_const {
                    Some(kc) => {
                        self.emit(Op::LoadName(vnc));
                        match private_brand {
                            Some(brand) => self.emit(Op::SetPrivate(*brand, *kc)),
                            None => self.emit(Op::SetPropConst(*kc)),
                        }
                    }
                    None => {
                        let knc = self.binding_name_const(t_key)?;
                        self.emit(Op::LoadName(knc));
                        self.emit(Op::LoadName(vnc));
                        self.emit(Op::SetElem);
                    }
                }
                // t_v, t_obj and t_key are dead once the store completes.
                self.release_temp(&t_v);
                self.release_temp(t_obj);
                self.release_temp(t_key);
                Ok(())
            }
            Target::Super {
                t_recv,
                t_base,
                key_const,
                t_key,
            } => {
                // [value] -> []: stash the value, then SuperSet on
                // [receiver, base, key, value].
                let t_v = self.hidden_temp();
                let vnc = self.binding_name_const(&t_v)?;
                self.emit(Op::StoreName(vnc));
                let rnc = self.binding_name_const(t_recv)?;
                self.emit(Op::LoadName(rnc));
                let bnc = self.binding_name_const(t_base)?;
                self.emit(Op::LoadName(bnc));
                match key_const {
                    Some(kc) => self.emit(Op::LoadConst(*kc)),
                    None => {
                        let knc = self.binding_name_const(t_key)?;
                        self.emit(Op::LoadName(knc));
                    }
                }
                self.emit(Op::LoadName(vnc));
                self.emit(Op::SuperSet);
                // t_v, t_recv, t_base and t_key are dead once the store
                // completes.
                self.release_temp(&t_v);
                self.release_temp(t_recv);
                self.release_temp(t_base);
                self.release_temp(t_key);
                Ok(())
            }
        }
    }
}

fn prop_key_name(key: &PropKey) -> String {
    match key {
        PropKey::Ident(s, _) => s.clone(),
        // Diagnostic only: lossy.
        PropKey::Str(s) => s.to_utf8_lossy().into_owned(),
        PropKey::Num(f, _) => format!("{f}"),
        PropKey::Computed(_) => "<computed>".to_string(),
    }
}

/// One getter/setter pair grouped by key for `#define_accessor`.
struct Accessor {
    key: PropKey,
    getter: Option<u16>,
    setter: Option<u16>,
}

/// One *private* getter/setter pair, grouped by private name. Unlike
/// [`Accessor`], private pairs install into the VM-side slot table via
/// [`Op::DefinePrivateAccessor`] instead of the public property map.
struct PrivateAccessor {
    name: String,
    getter: Option<u16>,
    setter: Option<u16>,
}

fn private_accessor_push(
    acc: &mut Vec<PrivateAccessor>,
    name: &str,
    getter: Option<u16>,
    setter: Option<u16>,
) {
    if let Some(a) = acc.iter_mut().find(|a| a.name == name) {
        if getter.is_some() {
            a.getter = getter;
        }
        if setter.is_some() {
            a.setter = setter;
        }
        return;
    }
    acc.push(PrivateAccessor {
        name: name.to_string(),
        getter,
        setter,
    });
}

fn accessor_push(acc: &mut Vec<Accessor>, key: PropKey, getter: Option<u16>, setter: Option<u16>) {
    let same = |a: &PropKey| match (&a, &key) {
        (PropKey::Computed(..), _) | (_, PropKey::Computed(..)) => false,
        _ => prop_key_name(a) == prop_key_name(&key),
    };
    if let Some(a) = acc.iter_mut().find(|a| same(&a.key)) {
        if getter.is_some() {
            a.getter = getter;
        }
        if setter.is_some() {
            a.setter = setter;
        }
        return;
    }
    acc.push(Accessor {
        key,
        getter,
        setter,
    });
}

// ---------------------------------------------------------------------------
// Classes
// ---------------------------------------------------------------------------

impl Compiler {
    /// Compile an expression as an immediately-invoked strict arrow;
    /// `[..] -> [.., value]`.
    ///
    /// Class heritage expressions and computed member keys are strict mode
    /// code (spec 10.2.1) but evaluate inside the enclosing function, so
    /// compiling them inline would apply the enclosing strictness at
    /// runtime: the VM keys `StoreName`-on-undeclared, strict property
    /// writes, and friends off the executing frame's strict flag. The
    /// wrapper is strict, restoring strict runtime semantics even in
    /// sloppy outer code. It is an arrow so `this`, `super`, `new.target`
    /// and `arguments` keep their enclosing meaning, exactly as if the
    /// expression had been evaluated inline (spec: heritage evaluation
    /// only swaps the LexicalEnvironment).
    fn compile_strict_expr_wrapper(&mut self, e: &Expr) -> Result<(), CompileError> {
        self.push_function(
            "#strict_expr".to_string(),
            0,
            true,  // is_arrow
            false, // is_async
            false, // is_generator
            true,  // strict
            false, // is_method
        );
        // Mirror `compile_static_block_wrapper`'s error unwinding. The
        // enclosing class context is inherited (never pushed here), so
        // there is no class_stack entry to pop.
        let tpl = match (|| {
            self.emit_prologue()?;
            self.compile_expr(e)?;
            self.emit(Op::Return);
            Ok(())
        })() {
            Ok(()) => self.pop_function(),
            Err(e) => {
                let f = self.fns.pop().expect("no active function");
                self.pop_scopes_to(f.scope_base);
                return Err(e);
            }
        };
        let ci = self.intern_template(tpl)?;
        self.emit(Op::NewFunction(ci));
        // Arrows ignore the call-time receiver (`this` arrives via the
        // `#this` upvalue), so pass `undefined`.
        self.emit(Op::PushUndefined);
        self.emit(Op::Call(0));
        Ok(())
    }

    /// Compile a class to a value; leaves `[class]` on the stack.
    ///
    /// `is_expr` marks a class *expression*: a named one binds its name in
    /// an inner scope (ECMA-262 `NamedEvaluation`), mirroring the named
    /// function expression machinery, so methods, field initializers,
    /// static blocks, and computed keys resolve the class name. Class
    /// declarations pass `false`: their name resolves via the outer `let`
    /// binding instead.
    ///
    /// v1 object model: methods live on `ctor.prototype`, statics on the
    /// constructor, instance fields initialize at the end of the constructor
    /// body in textual order, and private members live under `#`-prefixed
    /// property keys (which can never collide with user identifiers).
    fn compile_class_value(
        &mut self,
        name: Option<String>,
        superclass: Option<&Expr>,
        members: &[ClassMember],
        is_expr: bool,
    ) -> Result<(), CompileError> {
        let cname = name.clone().unwrap_or_else(|| "#anon".to_string());
        // Heritage goes in a hidden temp so constructors capture it. The
        // heritage is strict mode code (spec 10.2.1): evaluate it in an
        // immediately-invoked strict wrapper so runtime strictness
        // applies even when the enclosing function is sloppy
        // (`class C extends (x = 1, Object) {}` must throw ReferenceError
        // instead of creating a global).
        let super_name: Option<String> = match superclass {
            Some(sc) => {
                self.compile_strict_expr_wrapper(sc)?;
                let t = format!("#super#{}", self.class_counter);
                self.class_counter += 1;
                let fi = self.function_scope_idx();
                self.declare_in(fi, &t, BindingKind::Hidden)?;
                let nc = self.binding_name_const(&t)?;
                self.emit(Op::StoreName(nc));
                Some(t)
            }
            None => None,
        };

        // A named class *expression* binds its name in a fresh inner scope
        // (pushed after the heritage, which was evaluated in the outer
        // environment and must not see the name). Member templates compiled
        // below capture it as an upvalue, and stores to it get the
        // immutable self-name semantics via `compile_store`. The binding is
        // initialized after the method/accessor installs (see below) —
        // computed property keys evaluate while it is still uninitialized
        // (spec TDZ, read as `undefined` here) — and before the static
        // initializers run, which observe the name. The scope stays pushed
        // through install emission so computed keys resolve the name, and
        // is popped at the end of this function.
        let scope_base = self.scopes.len();
        // Run the fallible body in a closure so the class-name scope is
        // popped on error paths too, not just on success.
        let result: Result<(), CompileError> = (|| {
            let self_name_nc: Option<u16> = match (&name, is_expr) {
                (Some(n), true) => {
                    self.push_scope(false);
                    let scope = self.scopes.len() - 1;
                    let id = self.declare_in(scope, n, BindingKind::SelfName)?;
                    // Pre-intern the name const for the initializing store.
                    // (`declare_in` uniquified the internal name, so this const
                    // identifies exactly this binding.)
                    Some(
                        self.fns.last().expect("no active function").bindings[id as usize]
                            .name_const,
                    )
                }
                _ => None,
            };

            // Partition members.
            let mut ctor: Option<&ClassMember> = None;
            let mut proto_methods: Vec<&ClassMember> = Vec::new();
            let mut proto_accessors: Vec<Accessor> = Vec::new();
            let mut proto_private_accessors: Vec<PrivateAccessor> = Vec::new();
            let mut static_methods: Vec<&ClassMember> = Vec::new();
            let mut static_accessors: Vec<Accessor> = Vec::new();
            let mut static_private_accessors: Vec<PrivateAccessor> = Vec::new();
            // Getter/setter members are collected here and compiled after
            // the class context (private brand) is pushed below, so private
            // member access in their bodies resolves. Compiling them inline
            // during partitioning left `class_stack` empty and rejected
            // valid `this.#x` accesses.
            let mut proto_getset: Vec<(&ClassMember, bool)> = Vec::new();
            let mut static_getset: Vec<(&ClassMember, bool)> = Vec::new();
            let mut fields: Vec<(FieldKey, Option<Expr>)> = Vec::new();
            // Static initializers (fields and blocks) in textual order.
            let mut static_inits: Vec<StaticInit<'_>> = Vec::new();
            // Whether any private member exists: controls brand installation
            // (no private members means no brand checks can name this class).
            let has_private = members.iter().any(|m| match m {
                ClassMember::Method { key, .. } => matches!(key, ClassKey::Private(..)),
                ClassMember::Field { key, .. } => matches!(key, ClassKey::Private(..)),
                ClassMember::StaticBlock { .. } => false,
            });
            for m in members {
                match m {
                    ClassMember::Method {
                        kind: MethodKind::Constructor,
                        is_static: false,
                        ..
                    } => {
                        if ctor.is_some() {
                            return Err(CompileError::InvalidLhs);
                        }
                        ctor = Some(m);
                    }
                    ClassMember::Method {
                        kind: MethodKind::Constructor,
                        ..
                    } => return Err(CompileError::Unsupported("static constructor")),
                    ClassMember::Method {
                        is_static: false,
                        kind: kind @ (MethodKind::Getter | MethodKind::Setter),
                        ..
                    } => {
                        proto_getset.push((m, matches!(kind, MethodKind::Getter)));
                    }
                    ClassMember::Method {
                        is_static: false, ..
                    } => proto_methods.push(m),
                    ClassMember::Method {
                        is_static: true,
                        kind: kind @ (MethodKind::Getter | MethodKind::Setter),
                        ..
                    } => {
                        static_getset.push((m, matches!(kind, MethodKind::Getter)));
                    }
                    ClassMember::Method {
                        is_static: true, ..
                    } => static_methods.push(m),
                    ClassMember::Field {
                        key,
                        init,
                        is_static: false,
                        ..
                    } => {
                        let fk = self.precompute_field_key(key)?;
                        fields.push((fk, init.clone()))
                    }
                    ClassMember::Field {
                        key,
                        init,
                        is_static: true,
                        ..
                    } => {
                        static_inits.push(StaticInit::Field(self.precompute_field_key(key)?, init))
                    }
                    ClassMember::StaticBlock { body } => static_inits.push(StaticInit::Block(body)),
                }
            }

            // Compile templates with the class context visible (inherited by
            // every nested function state). Each class definition site gets a
            // fresh private brand.
            let brand = self.next_brand()?;
            self.fns
                .last_mut()
                .expect("no active function")
                .class_stack
                .push(ClassCtx {
                    super_name: super_name.clone().unwrap_or_default(),
                    brand,
                    is_static: false,
                });
            let ci = CtorInfo {
                fields,
                super_name: super_name.clone(),
                is_derived: super_name.is_some(),
                brand,
                has_private,
            };
            let ctor_tpl = match ctor {
                Some(ClassMember::Method { params, body, .. }) => self.compile_function_template(
                    cname.clone(),
                    params,
                    body,
                    false,
                    false,
                    false,
                    true,
                    false,
                    Some(ci),
                    None,
                )?,
                _ => self.compile_implicit_ctor(&cname, ci)?,
            };
            let mut proto_tpls: Vec<(&ClassKey, u16)> = Vec::new();
            for m in &proto_methods {
                let ti = self.compile_method_template(m)?;
                if let ClassMember::Method { key, .. } = m {
                    proto_tpls.push((key, ti));
                }
            }
            // Accessor templates are compiled here — after the class
            // context push — so private member access in getter/setter
            // bodies resolves the class's private brand.
            for (m, is_getter) in &proto_getset {
                let ti = self.compile_method_template(m)?;
                let ClassMember::Method { key, .. } = m else {
                    continue;
                };
                let (getter, setter) = if *is_getter {
                    (Some(ti), None)
                } else {
                    (None, Some(ti))
                };
                match key {
                    ClassKey::Private(s, _) => {
                        private_accessor_push(&mut proto_private_accessors, s, getter, setter)
                    }
                    _ => {
                        accessor_push(&mut proto_accessors, class_key_to_prop(key), getter, setter)
                    }
                }
            }
            let mut static_tpls: Vec<(&ClassKey, u16)> = Vec::new();
            // Static members resolve `super` against the base constructor
            // rather than its `.prototype`.
            self.fns
                .last_mut()
                .expect("no active function")
                .class_stack
                .last_mut()
                .expect("class context")
                .is_static = true;
            for m in &static_methods {
                let ti = self.compile_method_template(m)?;
                if let ClassMember::Method { key, .. } = m {
                    static_tpls.push((key, ti));
                }
            }
            // Static accessors compile under the static class context, like
            // static methods above.
            for (m, is_getter) in &static_getset {
                let ti = self.compile_method_template(m)?;
                let ClassMember::Method { key, .. } = m else {
                    continue;
                };
                let (getter, setter) = if *is_getter {
                    (Some(ti), None)
                } else {
                    (None, Some(ti))
                };
                match key {
                    ClassKey::Private(s, _) => {
                        private_accessor_push(&mut static_private_accessors, s, getter, setter)
                    }
                    _ => accessor_push(
                        &mut static_accessors,
                        class_key_to_prop(key),
                        getter,
                        setter,
                    ),
                }
            }
            self.fns
                .last_mut()
                .expect("no active function")
                .class_stack
                .pop();

            let ci = self.intern_template(ctor_tpl)?;
            self.emit(Op::NewFunction(ci)); // [class]
            if has_private {
                // The constructor carries its own brand so static private
                // members pass brand checks.
                self.emit(Op::AddPrivateBrand(brand));
            }
            if let Some(sn) = &super_name {
                // Link `ctor.prototype` and `ctor` itself to the heritage.
                // A null heritage is legal (ECMA-262
                // ClassDefinitionEvaluation): `D.prototype`'s prototype
                // becomes null, and `D` keeps the default constructor
                // prototype (%Function.prototype%, installed by
                // `NewFunction`) instead of throwing on `null.prototype`.
                let pi = self.intern("prototype")?;
                // [class] -> [class]: `D.prototype`'s proto is
                // `(super === null) ? null : super.prototype`.
                self.emit(Op::Dup); // [class, class]
                self.emit(Op::GetPropConst(pi)); // [class, Dproto]
                self.compile_load(sn)?; // [class, Dproto, super]
                self.emit(Op::Dup); // [class, Dproto, super, super]
                self.emit(Op::PushNull);
                self.emit(Op::StrictEq); // [class, Dproto, super, isNull]
                let not_null = self.emit_jump(Op::JumpIfFalse(0));
                self.emit(Op::Pop); // [class, Dproto]
                self.emit(Op::PushNull); // [class, Dproto, null]
                let proto_done = self.emit_jump(Op::Jump(0));
                self.patch(not_null)?; // [class, Dproto, super]
                self.emit(Op::GetPropConst(pi)); // [class, Dproto, superProto]
                self.patch(proto_done)?;
                self.call_intrinsic("set_proto", 2)?; // [class, Dproto]
                self.emit(Op::Pop); // [class]
                // [class] -> [class]: `D`'s own proto becomes `super`,
                // unless `super` is null (spec: constructorParent stays
                // %Function.prototype% for a null heritage).
                self.emit(Op::Dup); // [class, class]
                self.compile_load(sn)?; // [class, class, super]
                self.emit(Op::Dup); // [class, class, super, super]
                self.emit(Op::PushNull);
                self.emit(Op::StrictEq); // [class, class, super, isNull]
                let is_null = self.emit_jump(Op::JumpIfTrue(0));
                self.call_intrinsic("set_proto", 2)?; // [class, class]
                self.emit(Op::Pop); // [class]
                let ctor_done = self.emit_jump(Op::Jump(0));
                self.patch(is_null)?; // [class, class, super]
                self.emit(Op::Pop);
                self.emit(Op::Pop); // [class]
                self.patch(ctor_done)?;
            }
            // Prototype methods.
            let pi = self.intern("prototype")?;
            for (key, ti) in &proto_tpls {
                self.emit(Op::Dup);
                self.emit(Op::GetPropConst(pi));
                self.emit(Op::NewFunction(*ti));
                self.emit_method_store(key, brand)?;
            }
            self.emit_class_accessors(&proto_accessors, true)?;
            self.emit_private_class_accessors(&proto_private_accessors, true, brand)?;
            // Static methods.
            for (key, ti) in &static_tpls {
                self.emit(Op::Dup);
                self.emit(Op::NewFunction(*ti));
                self.emit_method_store(key, brand)?;
            }
            self.emit_class_accessors(&static_accessors, false)?;
            self.emit_private_class_accessors(&static_private_accessors, false, brand)?;
            if let Some(nc) = self_name_nc {
                // Initialize the class-name binding now that the class value
                // exists and all computed method/accessor keys have been
                // evaluated: `[class] -> [class]`. The keys are evaluated
                // while the binding is still uninitialized (spec TDZ, read
                // as `undefined` here), matching the field-key precomputes
                // emitted above. A raw slot write (the peephole turns it into
                // `Op::StoreLocal`), not the immutable self-name store path —
                // this is the initializing write, and it must run before the
                // static initializers below, which observe the name.
                self.emit(Op::Dup);
                self.emit(Op::StoreName(nc));
            }
            // Static initializers run in a wrapper closure so each gets the
            // correct receiver (`this` = the class), a private-brand context,
            // and — for static blocks — strict mode (see
            // `compile_static_init_wrapper`). `static_inits` is already in
            // textual order (fields and blocks interleaved).
            for init in &static_inits {
                let sn = super_name.clone().unwrap_or_default();
                let tpl = match init {
                    StaticInit::Field(key, init) => {
                        self.compile_static_field_wrapper(key, init, brand, &sn)?
                    }
                    StaticInit::Block(body) => {
                        self.compile_static_block_wrapper(body, brand, &sn)?
                    }
                };
                let ci = self.intern_template(tpl)?;
                self.emit(Op::Dup); // [class, class]
                self.emit(Op::NewFunction(ci)); // [class, init]
                self.emit(Op::Swap); // [init, class] = [callee, this_arg]
                self.emit(Op::Call(0)); // [result]
                self.emit(Op::Pop); // [class]
            }
            // Pop the class-name scope (pushed above for named class
            // expressions; a no-op when nothing was pushed). The binding itself
            // stays in the function's binding table for the VM. Runs on both
            // success and error paths (see the closure above).
            Ok(())
        })();
        self.pop_scopes_to(scope_base);
        result
    }

    /// Compile a static-field initializer as a strict wrapper closure.
    /// The wrapper is called with `this` = the class and carries the
    /// class's private brand, so `this` and `#x` resolve correctly inside
    /// the initializer. Returns the template.
    fn compile_static_field_wrapper(
        &mut self,
        key: &FieldKey,
        init: &Option<Expr>,
        brand: u16,
        super_name: &str,
    ) -> Result<FuncTemplate, CompileError> {
        self.push_function(
            "<static-field>".to_string(),
            0,
            false,
            false,
            false,
            true,
            false,
        );
        // Private-brand context for `#x` in the initializer.
        self.fns
            .last_mut()
            .expect("no active function")
            .class_stack
            .push(ClassCtx {
                super_name: super_name.to_string(),
                brand,
                is_static: true,
            });
        let r = self.compile_static_field_body(key, init, brand);
        self.fns
            .last_mut()
            .expect("no active function")
            .class_stack
            .pop();
        r?;
        let tpl = self.pop_function();
        Ok(tpl)
    }

    /// Body of a static-field wrapper: `this.<key> = init`.
    fn compile_static_field_body(
        &mut self,
        key: &FieldKey,
        init: &Option<Expr>,
        brand: u16,
    ) -> Result<(), CompileError> {
        self.emit_prologue()?;
        self.compile_this()?; // [this]
        match init {
            Some(e) => self.compile_expr(e)?,
            None => self.emit(Op::PushUndefined),
        }
        // [this, v]
        match key {
            FieldKey::Name(s) => {
                let i = self.intern_wtf8(s)?;
                self.emit(Op::SetPropConst(i));
            }
            FieldKey::Private(s) => {
                // Like instance fields: the slot table, not a property.
                // The constructor was branded at class evaluation, so no
                // brand check is needed here.
                let i = self.intern(&format!("#{s}"))?;
                self.emit(Op::InitPrivate(brand, i));
            }
            FieldKey::Precomputed(t) => {
                // Computed keys are strict-mode code (spec 10.2.1):
                // evaluated once at class definition time in the
                // definition scope through the immediately-invoked strict
                // wrapper (see `precompute_field_key`), so sloppy outer
                // code still gets strict runtime semantics — matching
                // computed method/accessor keys and heritage. The
                // static-field wrapper just reuses the stored key.
                self.compile_load(t)?; // [this, v, k]
                self.emit(Op::Swap); // [this, k, v]
                self.emit(Op::SetElem); // []
            }
        }
        self.emit(Op::PushUndefined);
        self.emit(Op::Return);
        Ok(())
    }

    /// Compile a static block as a strict wrapper closure. The wrapper is
    /// called with `this` = the class and carries the class's private
    /// brand. Static blocks are always strict, and they get no `arguments`
    /// object (`arguments` resolves to the outer scope).
    fn compile_static_block_wrapper(
        &mut self,
        body: &[Stmt],
        brand: u16,
        super_name: &str,
    ) -> Result<FuncTemplate, CompileError> {
        self.push_function(
            "<static-block>".to_string(),
            0,
            false,
            false,
            false,
            true,
            false,
        );
        {
            let f = self.fns.last_mut().expect("no active function");
            f.is_class_init = true;
            f.class_stack.push(ClassCtx {
                super_name: super_name.to_string(),
                brand,
                is_static: true,
            });
        }
        // The wrapper runs its own declare walk; the enclosing walk skips
        // static blocks.
        let r = (|| {
            self.hoist_names(body)?;
            self.declare_stmts(body)?;
            self.emit_prologue()?;
            self.compile_stmts(body)?;
            self.emit(Op::PushUndefined);
            self.emit(Op::Return);
            Ok(())
        })();
        self.fns
            .last_mut()
            .expect("no active function")
            .class_stack
            .pop();
        // Mirror `compile_function_template`'s error unwinding.
        match r {
            Ok(()) => {
                let tpl = self.pop_function();
                Ok(tpl)
            }
            Err(e) => {
                let f = self.fns.pop().expect("no active function");
                self.pop_scopes_to(f.scope_base);
                Err(e)
            }
        }
    }

    fn compile_method_template(&mut self, m: &ClassMember) -> Result<u16, CompileError> {
        match m {
            ClassMember::Method {
                key,
                params,
                body,
                is_async,
                is_generator,
                kind,
                ..
            } => {
                let ps: &[Param] = match kind {
                    MethodKind::Setter => params.get(..1).unwrap_or(&[]),
                    MethodKind::Getter => &[],
                    _ => params,
                };
                let t = self.compile_function_template(
                    class_method_name(key),
                    ps,
                    body,
                    false,
                    *is_async,
                    *is_generator,
                    true,
                    true,
                    None,
                    None,
                )?;
                self.intern_template(t)
            }
            _ => Err(CompileError::InvalidLhs),
        }
    }

    fn compile_implicit_ctor(
        &mut self,
        name: &str,
        ci: CtorInfo,
    ) -> Result<FuncTemplate, CompileError> {
        let CtorInfo {
            fields,
            super_name,
            is_derived,
            brand,
            has_private,
        } = ci;
        // The class context is inherited from the enclosing function state.
        self.push_function(name.to_string(), 0, false, false, false, true, false);
        let f = self.fns.last_mut().expect("no active function");
        f.ctor_ctx = Some(CtorCtx {
            fields,
            brand,
            has_private,
        });
        // An implicit derived constructor (`class D extends B {}` with no
        // explicit constructor) calls `super(...args)`: same TDZ rule.
        f.is_derived_ctor = is_derived;
        // Like every other non-arrow function, the implicit constructor
        // gets an `arguments` object (spec: it behaves as
        // `constructor(...args)`). Declared before the prologue so the
        // slot is initialized with the other bindings; the effective
        // parameter list is non-simple (a rest parameter), so the object
        // is unmapped.
        let scope = self.fns.last().expect("no active function").scope_base;
        self.declare_in(scope, "arguments", BindingKind::Var)?;
        self.emit_prologue()?;
        self.emit(Op::NewArguments);
        self.compile_store("arguments")?;
        if super_name.is_some() {
            // `super(...args)`.
            let sn = super_name
                .filter(|s| !s.is_empty())
                .ok_or(CompileError::InvalidLhs)?;
            self.emit(Op::PushInt(0));
            self.emit(Op::RestArgs);
            self.compile_load(&sn)?;
            self.emit(Op::Swap);
            self.compile_new_target()?;
            self.emit(Op::Swap);
            self.call_intrinsic("super_construct", 3)?;
            let nc = self.binding_name_const("#this")?;
            self.emit(Op::StoreName(nc));
        }
        let ctx = self
            .fns
            .last_mut()
            .expect("no active function")
            .ctor_ctx
            .take()
            .expect("just set");
        if ctx.has_private {
            self.compile_this()?;
            self.emit(Op::AddPrivateBrand(ctx.brand));
            self.emit(Op::Pop);
        }
        self.emit_field_inits(&ctx.fields, ctx.brand)?;
        self.emit(Op::PushUndefined);
        self.emit(Op::Return);
        Ok(self.pop_function())
    }

    /// Emit instance field initializers in textual order: `this.<key> = init`.
    fn emit_field_inits(
        &mut self,
        fields: &[(FieldKey, Option<Expr>)],
        brand: u16,
    ) -> Result<(), CompileError> {
        for (key, init) in fields {
            self.compile_this()?; // [this]
            match init {
                Some(e) => self.compile_expr(e)?,
                None => self.emit(Op::PushUndefined),
            }
            // [this, v]
            match key {
                FieldKey::Name(s) => {
                    let i = self.intern_wtf8(s)?;
                    self.emit(Op::SetPropConst(i));
                }
                FieldKey::Private(s) => {
                    // Private fields live in the VM-side slot table, not
                    // as string-keyed properties: no brand check is needed
                    // here (the instance was just branded), and user code
                    // can never observe the slot through `this["#s"]`.
                    let i = self.intern(&format!("#{s}"))?;
                    self.emit(Op::InitPrivate(brand, i));
                }
                FieldKey::Precomputed(t) => {
                    let t_v = self.hidden_temp();
                    let vnc = self.binding_name_const(&t_v)?;
                    self.emit(Op::StoreName(vnc)); // [this]
                    self.compile_this()?; // [this, this]
                    // The key was evaluated exactly once at class
                    // definition time (spec: ClassDefinitionEvaluation)
                    // and parked in a hidden binding the constructor
                    // captures; reuse it here without re-evaluating.
                    // The wrapper that computed it is an arrow invoked in
                    // the definition scope, so `this` was the outer
                    // `this`, not the instance.
                    self.compile_load(t)?; // [this, this, k]
                    self.emit(Op::LoadName(vnc)); // [this, this, k, v]
                    self.emit(Op::SetElem); // [this]
                    self.emit(Op::Pop); // []
                    // t_v is dead once the field is stored.
                    self.release_temp(&t_v);
                }
            }
        }
        Ok(())
    }

    /// After `[holder, holder, fn]`: store the method; leaves `[holder]`.
    fn emit_method_store(&mut self, key: &ClassKey, brand: u16) -> Result<(), CompileError> {
        match key {
            ClassKey::Public(PropKey::Ident(s, _)) => {
                let i = self.intern(s)?;
                self.emit(Op::SetPropConst(i));
            }
            ClassKey::Public(PropKey::Str(s)) => {
                let i = self.intern_wtf8(s)?;
                self.emit(Op::SetPropConst(i));
            }
            ClassKey::Public(pk) => {
                self.compile_class_key(pk)?;
                // [h, h, f, k] -> Swap -> [h, h, k, f] -> SetProp -> [h]
                self.emit(Op::Swap);
                self.emit(Op::SetProp);
            }
            ClassKey::Private(s, _) => {
                // Private methods live in the VM-side slot table, not as
                // string-keyed properties on the prototype.
                let i = self.intern(&format!("#{s}"))?;
                self.emit(Op::InitPrivate(brand, i));
            }
        }
        Ok(())
    }

    fn emit_class_accessors(
        &mut self,
        accs: &[Accessor],
        on_proto: bool,
    ) -> Result<(), CompileError> {
        if accs.is_empty() {
            return Ok(());
        }
        let pi = self.intern("prototype")?;
        for a in accs {
            // [class]; Dup; [GetPropConst prototype]; key; getter|undef;
            // setter|undef; #define_accessor(4); Pop.
            self.emit(Op::Dup);
            if on_proto {
                self.emit(Op::GetPropConst(pi));
            }
            self.compile_class_key(&a.key)?;
            match a.getter {
                Some(ti) => self.emit(Op::NewFunction(ti)),
                None => self.emit(Op::PushUndefined),
            }
            match a.setter {
                Some(ti) => self.emit(Op::NewFunction(ti)),
                None => self.emit(Op::PushUndefined),
            }
            self.call_intrinsic("define_accessor", 4)?;
            self.emit(Op::Pop);
        }
        Ok(())
    }

    /// Install private getters/setters into the VM-side slot table.
    /// After `[class]`: for each pair, `Dup`, the holder, the getter (or
    /// `undefined`), the setter (or `undefined`), then
    /// `DefinePrivateAccessor(brand, i)`; leaves `[class]`.
    fn emit_private_class_accessors(
        &mut self,
        accs: &[PrivateAccessor],
        on_proto: bool,
        brand: u16,
    ) -> Result<(), CompileError> {
        if accs.is_empty() {
            return Ok(());
        }
        let pi = self.intern("prototype")?;
        for a in accs {
            // [class]; Dup; [GetPropConst prototype]; getter|undef;
            // setter|undef; DefinePrivateAccessor(brand, i) -> [class, holder];
            // Pop -> [class].
            self.emit(Op::Dup);
            if on_proto {
                self.emit(Op::GetPropConst(pi));
            }
            match a.getter {
                Some(ti) => self.emit(Op::NewFunction(ti)),
                None => self.emit(Op::PushUndefined),
            }
            match a.setter {
                Some(ti) => self.emit(Op::NewFunction(ti)),
                None => self.emit(Op::PushUndefined),
            }
            let i = self.intern(&format!("#{}", a.name))?;
            self.emit(Op::DefinePrivateAccessor(brand, i));
            // `DefinePrivateAccessor` pushes the holder back; drop it so
            // each iteration is `[class] -> [class]` (the public-accessor
            // emitter pops its intrinsic result the same way).
            self.emit(Op::Pop);
        }
        Ok(())
    }
}

fn class_method_name(key: &ClassKey) -> String {
    match key {
        ClassKey::Public(pk) => prop_key_name(pk),
        ClassKey::Private(s, _) => format!("#{s}"),
    }
}

fn class_key_to_prop(key: &ClassKey) -> PropKey {
    match key {
        ClassKey::Public(pk) => pk.clone(),
        ClassKey::Private(s, _) => PropKey::Str(Wtf8Buf::from(&format!("#{s}"))),
    }
}

impl Compiler {
    /// Resolve a class field key at class definition time. Computed keys
    /// are evaluated exactly once, in the definition scope, through the
    /// immediately-invoked strict arrow wrapper (spec 10.2.1), and parked
    /// in a hidden binding; the constructor and static-field wrappers
    /// capture that binding and reuse the stored key for every field
    /// install, so the key expression never re-evaluates (spec:
    /// ClassDefinitionEvaluation / DefineField). The strict wrapper keeps
    /// the R5 strictness requirement (an implicit-global assignment in a
    /// sloppy program still throws ReferenceError), and because the
    /// wrapper is an arrow invoked in the definition scope, `this` inside
    /// a key is the outer `this` — not the instance, not the class.
    fn precompute_field_key(&mut self, key: &ClassKey) -> Result<FieldKey, CompileError> {
        match key {
            ClassKey::Public(PropKey::Ident(s, _)) => Ok(FieldKey::Name(Wtf8Buf::from(s.as_str()))),
            ClassKey::Public(PropKey::Str(s)) => Ok(FieldKey::Name(s.clone())),
            ClassKey::Public(PropKey::Num(f, span)) => {
                self.precompute_key_expr(&Expr::Lit(Literal::Number(*f), *span))
            }
            ClassKey::Public(PropKey::Computed(e)) => self.precompute_key_expr(e),
            ClassKey::Private(s, _) => Ok(FieldKey::Private(s.clone())),
        }
    }

    /// Evaluate one computed field-key expression at class definition time
    /// and park the result in a fresh hidden binding; returns the key as
    /// [`FieldKey::Precomputed`].
    fn precompute_key_expr(&mut self, e: &Expr) -> Result<FieldKey, CompileError> {
        let t = format!("#fieldkey#{}", self.class_counter);
        self.class_counter += 1;
        let fi = self.function_scope_idx();
        self.declare_in(fi, &t, BindingKind::Hidden)?;
        // The arrow inherits the definition scope's `this` (the outer
        // `this`, not the instance or the class).
        self.compile_strict_expr_wrapper(e)?; // [k]
        let nc = self.binding_name_const(&t)?;
        self.emit(Op::StoreName(nc)); // []
        Ok(FieldKey::Precomputed(t))
    }
}

// ---------------------------------------------------------------------------
// Patterns (`[value] -> []`)
// ---------------------------------------------------------------------------

/// Split a member-ish destructuring target into its member expression and
/// optional default: `Pat::Member(e)` -> `(e, None)`,
/// `Pat::Assign(Pat::Member(e), dflt)` -> `(e, Some(dflt))`. `None` for
/// any other shape. Used to hoist `compile_target_begin` ahead of the
/// value read per ECMA-262 13.15.5.
fn member_target_parts(p: &Pat) -> Option<(&Expr, Option<&Expr>)> {
    match p {
        Pat::Member(e) => Some((e.as_ref(), None)),
        Pat::Assign(inner, dflt, _) => match inner.as_ref() {
            Pat::Member(e) => Some((e.as_ref(), Some(dflt.as_ref()))),
            _ => None,
        },
        _ => None,
    }
}

/// True for a member-expression destructuring target, with or without a
/// default (`[a.b]`, `[a.b = d]`), including under an array rest
/// (`[...a.b]`). ECMA-262 13.15.5.3
/// (IteratorDestructuringAssignmentEvaluation) evaluates such a target's
/// reference *before* the iterator step (`next()` call), unlike other
/// element kinds.
fn is_member_target_pat(p: &Pat) -> bool {
    match p {
        Pat::Rest(inner, _) => member_target_parts(inner).is_some(),
        _ => member_target_parts(p).is_some(),
    }
}

impl Compiler {
    /// Compile an object-pattern key with eager `ToPropertyKey`: `[..] ->
    /// [.., keystr]`. ECMA-262 13.15.5.2 evaluates the `PropertyName`
    /// (including `ToPropertyKey`) before the destructuring target's
    /// reference, so when the target is hoisted ahead of `GetElem` the
    /// key conversion must already have run. Constant keys need no
    /// conversion; for a computed key the engine's own conversion runs
    /// exactly once through a scratch object (`SetElem` converts,
    /// `enum_keys` reads the converted key back), so a user
    /// `toString`/`valueOf` fires a single time, before the hoisted
    /// target evaluation, and the later `GetElem` sees an already-string
    /// key.
    fn compile_key_converted(&mut self, key: &PropKey) -> Result<(), CompileError> {
        if !matches!(key, PropKey::Computed(_)) {
            return self.compile_key(key);
        }
        self.compile_key(key)?; // [.., key]
        let t_scratch = self.hidden_temp();
        let snc = self.binding_name_const(&t_scratch)?;
        self.emit(Op::Dup); // [.., key, key]
        self.emit(Op::NewObject); // [.., key, key, scratch]
        self.emit(Op::StoreName(snc)); // [.., key, key]
        let snc = self.binding_name_const(&t_scratch)?;
        self.emit(Op::LoadName(snc)); // [.., key, key, scratch]
        self.emit(Op::Swap); // [.., key, scratch, key]
        self.emit(Op::PushInt(0)); // [.., key, scratch, key, 0]
        // The engine's key conversion (user `toString`/`valueOf`) runs
        // here, exactly once.
        self.emit(Op::SetElem); // [.., key]
        let snc = self.binding_name_const(&t_scratch)?;
        self.emit(Op::LoadName(snc)); // [.., key, scratch]
        self.call_intrinsic("enum_keys", 1)?; // [.., key, keys]
        self.release_temp(&t_scratch);
        self.emit(Op::PushInt(0)); // [.., key, keys, 0]
        self.emit(Op::GetElem); // [.., key, keystr]
        self.emit(Op::Swap); // [.., keystr, key]
        self.emit(Op::Pop); // [.., keystr]
        Ok(())
    }

    /// Bind a destructuring pattern; consumes the value on the stack.
    /// Bindings were declared by [`Compiler::hoist`].
    fn compile_pat_bind(&mut self, pat: &Pat, kind: VarKind) -> Result<(), CompileError> {
        self.compile_pat_bind_guarded(pat, kind, false)
    }

    /// Compile a for-in/of per-iteration *assignment* into a `var`/bare
    /// pattern (`[v] -> []`). Unlike [`compile_pat_bind`] (declaration
    /// mode), `Pat::Ident` leaves store through [`compile_store`] — the
    /// with-aware assignment — so an enclosing `with` object's properties
    /// are updated per spec `PutValue`, matching the `ForLhs::Ident`
    /// `kind == Var` arm. Only the two for-in/of compile paths use this
    /// (for `kind == VarKind::Var`, which covers both `for (var {a} of
    /// ...)` and bare `for ({a} of ...)`); `let`/`const` keep declaration
    /// mode (fresh per-iteration bindings correctly skip probing), and so
    /// do parameter binding and ordinary `var` declarations.
    fn compile_pat_bind_assign(&mut self, pat: &Pat) -> Result<(), CompileError> {
        self.compile_pat_bind_guarded(pat, VarKind::Var, true)
    }

    /// Depth-guarded entry for pattern binding: every nesting level passes
    /// through [`guarded`](Self::guarded) so deep patterns fail cleanly
    /// with `DepthExceeded` instead of overflowing the native stack.
    fn compile_pat_bind_guarded(
        &mut self,
        pat: &Pat,
        kind: VarKind,
        assign: bool,
    ) -> Result<(), CompileError> {
        self.guarded(|s| s.compile_pat_bind_inner(pat, kind, assign))
    }

    /// Store one pattern identifier leaf: assignment mode routes through
    /// the with-aware [`compile_store`] (with a const check, like plain
    /// identifier assignment), declaration mode initializes the declared
    /// binding via [`compile_store_decl`].
    fn compile_pat_leaf_store(
        &mut self,
        name: &str,
        kind: VarKind,
        assign: bool,
    ) -> Result<(), CompileError> {
        if assign {
            self.compile_const_check(name)?;
            self.compile_store(name)
        } else {
            self.compile_store_decl(name, kind)
        }
    }

    fn compile_pat_bind_inner(
        &mut self,
        pat: &Pat,
        kind: VarKind,
        assign: bool,
    ) -> Result<(), CompileError> {
        match pat {
            Pat::Ident(name, _) => self.compile_pat_leaf_store(name, kind, assign),
            Pat::Assign(inner, default, _) => {
                self.emit(Op::Dup);
                self.emit(Op::PushUndefined);
                self.emit(Op::StrictEq);
                let l = self.emit_jump(Op::JumpIfFalse(0));
                self.emit(Op::Pop);
                self.compile_expr(default)?;
                self.patch(l)?;
                self.compile_pat_bind_guarded(inner, kind, assign)
            }
            Pat::Array(elems, _) => {
                // ECMA-262 13.15.5: array patterns destructure through the
                // *iterator protocol* (`GetIterator` + `IteratorStep` +
                // `IteratorValue`), not indexed reads — generators, strings
                // and custom `[Symbol.iterator]` RHS values all work.
                // `Op::IterNext` stages the first-step `@@iterator()` call
                // and every `next()` call; its indexed fast path keeps
                // plain arrays and strings cheap.
                //
                // Stack discipline: `[v] -> [v, st]` across elements (the
                // value doubles as the iterable), `[]` on exit.
                //
                // ECMA-262 13.15.5.3 IteratorDestructuringAssignmentEvaluation:
                // for a non-literal target (a member reference), `lref =
                // Evaluation(DestructuringAssignmentTarget)` runs *before*
                // `IteratorValue` (the `next()` call) but *after*
                // `GetIterator`. The first `IterNext` fuses `@@iterator()`
                // and `next()`, so when the first element is a member-ish
                // target the iterator is acquired eagerly here; the element
                // loop's `IterNext` calls then only perform `next()`.
                // `[base()[key()]] = iter` orders
                // `@@iterator,base,key,next`, matching V8.
                if elems.first().is_some_and(
                    |el| matches!(el, Some(ArrayPatElem::Pat(p)) if is_member_target_pat(p)),
                ) {
                    // [v] -> [v, st]: acquire the iterator eagerly so the
                    // element loop's `IterNext` calls only perform `next()`.
                    // `v[Symbol.iterator]` is read under the real well-known
                    // symbol key via `PushWellKnownSymbol` + `GetElem`,
                    // exactly as the VM's own iterator lookup does (no
                    // dependency on the `Symbol` binding). When absent
                    // (plain arrays/strings in this engine) the `Int(0)`
                    // state is kept, so `IterNext` uses its indexed fast
                    // path — no user code runs either way there, so the
                    // order is unobservable. The extra `Dup` keeps a copy
                    // of `v`: the `Call` below consumes the receiver copy.
                    self.emit(Op::Dup); // [v, v]
                    self.emit(Op::Dup); // [v, v, v]
                    self.emit(Op::PushWellKnownSymbol(WellKnown::Iterator as u8)); // [v, v, v, sym]
                    self.emit(Op::GetElem); // [v, v, method]
                    self.emit(Op::Dup); // [v, v, method, method]
                    self.emit(Op::PushUndefined); // [v, v, method, method, undefined]
                    self.emit(Op::StrictEq); // [v, v, method, is_undef]
                    let l_fast = self.emit_jump(Op::JumpIfTrue(0));
                    self.emit(Op::Swap); // [v, method, v]
                    self.emit(Op::Call(0)); // [v, iter]
                    // Mirror the normal path's GetIterator validation
                    // (`iter_next` in vm.rs rejects a non-object
                    // `Symbol.iterator()` result with TypeError): `k in obj`
                    // throws TypeError when `obj` is not an object
                    // (including `null`), and runs no user code on the
                    // way, so the eager result is validated in place.
                    self.emit(Op::Dup); // [v, iter, iter]
                    self.emit(Op::PushWellKnownSymbol(WellKnown::Iterator as u8)); // [v, iter, iter, sym]
                    self.emit(Op::Swap); // [v, iter, sym, iter]
                    self.emit(Op::In); // [v, iter, ok]
                    self.emit(Op::Pop); // [v, iter]
                    let l_done = self.emit_jump(Op::Jump(0));
                    self.patch(l_fast)?; // [v, v, method]
                    self.emit(Op::Pop); // [v, v]
                    self.emit(Op::Pop); // [v]
                    self.emit(Op::PushInt(0)); // [v, 0]
                    self.patch(l_done)?; // [v, st]
                } else {
                    self.emit(Op::PushInt(0)); // [v, st]
                }
                // Abrupt completion during element binding must run
                // `IteratorClose` on this pattern's iterator (spec
                // 13.15.5.3): arm the runtime guard now; the normal
                // epilogue below disarms it. `state` is either the
                // protocol iterator object (closed via `return?.()`) or
                // the index fast-path `Int` (no close needed).
                self.emit(Op::IterGuard);
                for el in elems {
                    match el {
                        // Elisions still consume an iterator step.
                        None => {
                            self.emit(Op::IterNext); // [v, st', e, done]
                            self.emit(Op::Pop);
                            self.emit(Op::Pop); // [v, st']
                        }
                        Some(ArrayPatElem::Pat(Pat::Rest(inner, _))) => {
                            // `...rest`: collect the remaining iterator
                            // values into a fresh array. When the iterator
                            // is already done the rest is `[]`.
                            //
                            // ECMA-262 13.15.5.3: for a member-expression
                            // rest target, `lref = Evaluation(
                            // DestructuringAssignmentTarget)` runs *before*
                            // the collection loop's `next()` calls. Hoist
                            // the reference evaluation for member targets
                            // (the `compile_target_begin` spill keeps the
                            // `[v, st]` stack shape); other targets keep
                            // the old order, where the loop runs first.
                            let (t, dflt) = match member_target_parts(inner) {
                                Some((e, dflt)) => (Some(self.compile_target_begin(e)?), dflt),
                                None => (None, None),
                            };
                            self.emit(Op::NewArray(0)); // [v, st, arr]
                            let t_arr = self.hidden_temp();
                            let anc = self.binding_name_const(&t_arr)?;
                            self.emit(Op::StoreName(anc)); // [v, st]
                            let l_head = self.code_len();
                            self.emit(Op::IterNext); // [v, st', e, done]
                            let done_jump = self.emit_jump(Op::JumpIfTrue(0));
                            // [v, st', e]
                            let anc = self.binding_name_const(&t_arr)?;
                            self.emit(Op::LoadName(anc)); // [v, st', e, arr]
                            self.emit(Op::Swap); // [v, st', arr, e]
                            self.call_intrinsic("arr_append", 2)?; // [v, st', arr]
                            let anc = self.binding_name_const(&t_arr)?;
                            self.emit(Op::StoreName(anc)); // [v, st']
                            self.emit_jump_to(l_head)?;
                            self.patch(done_jump)?; // [v, st', e=undef]
                            self.emit(Op::Pop); // [v, st']
                            self.emit(Op::Pop); // [v]
                            let anc = self.binding_name_const(&t_arr)?;
                            self.emit(Op::LoadName(anc)); // [v, arr]
                            self.release_temp(&t_arr);
                            match t {
                                Some(t) => {
                                    // A collected array is never
                                    // `undefined`, so the default (only
                                    // reachable via a hand-built AST, since
                                    // the grammar forbids rest
                                    // initializers) never fires; keep the
                                    // shape uniform with the element arm.
                                    if let Some(d) = dflt {
                                        self.emit(Op::Dup);
                                        self.emit(Op::PushUndefined);
                                        self.emit(Op::StrictEq);
                                        let l = self.emit_jump(Op::JumpIfFalse(0));
                                        self.emit(Op::Pop);
                                        self.compile_expr(d)?;
                                        self.patch(l)?;
                                    }
                                    self.compile_target_store_pop(&t)?;
                                }
                                None => {
                                    self.compile_pat_bind_guarded(inner, kind, assign)?;
                                }
                            }
                            // [v]; `...rest` is last per the grammar, so
                            // restore the `[v, st]` shape for the uniform
                            // epilogue below.
                            self.emit(Op::PushInt(0)); // [v, st]
                        }
                        Some(ArrayPatElem::Pat(p)) => {
                            // ECMA-262 13.15.5.3
                            // IteratorDestructuringAssignmentEvaluation:
                            // for a non-literal target (a member
                            // reference), `lref =
                            // Evaluation(DestructuringAssignmentTarget)`
                            // runs *before* `IteratorValue` (the
                            // `next()` call). Hoist the reference
                            // evaluation ahead of `IterNext` for member
                            // targets (and `Member = default`, keeping
                            // the default check after `next()`). When the
                            // member target is the first element the
                            // iterator was already acquired above, so this
                            // `IterNext` only performs `next()`.
                            if is_member_target_pat(p) {
                                // `compile_target_begin` spills base/key
                                // to hidden temps, so the
                                // `[v, st] -> [v, st', e] -> [v, st']`
                                // stack discipline is preserved. (`p`
                                // cannot be `Pat::Rest` here: the rest arm
                                // above matches first.)
                                let (e, dflt) =
                                    member_target_parts(p).expect("is_member_target_pat");
                                let t = self.compile_target_begin(e)?;
                                self.emit(Op::IterNext); // [v, st', e, done]
                                // `done` is dropped: an exhausted
                                // iterator yields `undefined`, which
                                // the element binds.
                                self.emit(Op::Pop); // [v, st', e]
                                if let Some(dflt) = dflt {
                                    self.emit(Op::Dup);
                                    self.emit(Op::PushUndefined);
                                    self.emit(Op::StrictEq);
                                    let l = self.emit_jump(Op::JumpIfFalse(0));
                                    self.emit(Op::Pop);
                                    self.compile_expr(dflt)?;
                                    self.patch(l)?;
                                }
                                self.compile_target_store_pop(&t)?;
                                // [v, st']
                            } else {
                                self.emit(Op::IterNext); // [v, st', e, done]
                                // `done` is dropped: an exhausted iterator
                                // yields `undefined`, which the element binds.
                                self.emit(Op::Pop); // [v, st', e]
                                self.compile_pat_bind_guarded(p, kind, assign)?;
                                // [v, st']
                            }
                        }
                    }
                }
                self.emit(Op::IterUnguard);
                self.emit(Op::Pop); // [v]
                self.emit(Op::Pop); // []
                Ok(())
            }
            Pat::Object(props, _) => {
                for pr in props {
                    match pr {
                        ObjectPatProp::KeyValue { key, value, .. } => {
                            // ECMA-262 13.15.5.2
                            // KeyedDestructuringAssignmentEvaluation: for a
                            // non-pattern target, `lref = Evaluation(
                            // DestructuringAssignmentTarget)` runs *before*
                            // `GetV(obj, key)`. Hoist the reference
                            // evaluation ahead of `GetElem` for member
                            // targets (mirroring the array arm's `IterNext`
                            // hoist; the default check stays after the
                            // read). Nested patterns keep the GetV-first
                            // order: only the *target* reference moves.
                            if let Some((e, dflt)) = member_target_parts(value) {
                                self.emit(Op::Dup); // [v, v]
                                // Eager `ToPropertyKey`: the spec's
                                // `PropertyName` evaluation (including key
                                // conversion) runs before `lref`, so the
                                // key's `toString` must fire before the
                                // hoisted target below.
                                self.compile_key_converted(key)?; // [v, v, keystr]
                                // Stack-neutral: base/key spill to hidden
                                // temps.
                                let t = self.compile_target_begin(e)?;
                                self.emit(Op::GetElem); // [v, val]
                                if let Some(dflt) = dflt {
                                    self.emit(Op::Dup);
                                    self.emit(Op::PushUndefined);
                                    self.emit(Op::StrictEq);
                                    let l = self.emit_jump(Op::JumpIfFalse(0));
                                    self.emit(Op::Pop);
                                    self.compile_expr(dflt)?;
                                    self.patch(l)?;
                                }
                                self.compile_target_store_pop(&t)?; // [v]
                            } else {
                                self.emit(Op::Dup);
                                self.compile_key(key)?;
                                self.emit(Op::GetElem);
                                self.compile_pat_bind_guarded(value, kind, assign)?;
                            }
                        }
                        ObjectPatProp::Shorthand { name, .. } => {
                            self.emit(Op::Dup);
                            let i = self.intern(name)?;
                            self.emit(Op::GetPropConst(i));
                            self.compile_pat_leaf_store(name, kind, assign)?;
                        }
                        ObjectPatProp::Rest(inner, _) => {
                            // [v] -> [v, rest]: `...rest` collects the own
                            // enumerable keys minus the destructured ones
                            // (spec `CopyDataProperties`). The `#obj_rest`
                            // intrinsic was never implemented in the VM, so
                            // lower it here: `#obj_spread` copies the
                            // own-enumerable data properties (sharing the
                            // engine's existing spread semantics), then each
                            // statically known destructured key is deleted.
                            // Computed keys cannot be known statically and
                            // are not excluded (pre-existing gap, shared
                            // with the old lowering).
                            //
                            // ECMA-262 13.15.5.2
                            // (RestDestructuringAssignmentEvaluation):
                            // for a member-expression rest target the
                            // reference evaluates before the property
                            // copy. Unobservable in this engine
                            // (`obj_spread` copies own data properties
                            // without running user getters), but hoist it
                            // anyway for spec shape.
                            let t = match member_target_parts(inner) {
                                Some((e, _)) => Some(self.compile_target_begin(e)?),
                                None => None,
                            };
                            self.emit(Op::Dup); // [v, v]
                            self.emit(Op::NewObject); // [v, v, {}]
                            self.emit(Op::Swap); // [v, {}, v]
                            self.call_intrinsic("obj_spread", 2)?; // [v, rest]
                            for pr2 in props.iter() {
                                match pr2 {
                                    ObjectPatProp::KeyValue {
                                        key: PropKey::Ident(s, _),
                                        ..
                                    } => {
                                        let ki = self.intern(s)?;
                                        self.emit(Op::Dup);
                                        self.emit(Op::LoadConst(ki));
                                        self.emit(Op::Delete);
                                        self.emit(Op::Pop);
                                    }
                                    ObjectPatProp::KeyValue {
                                        key: PropKey::Str(s),
                                        ..
                                    } => {
                                        let ki = self.intern_wtf8(s)?;
                                        self.emit(Op::Dup);
                                        self.emit(Op::LoadConst(ki));
                                        self.emit(Op::Delete);
                                        self.emit(Op::Pop);
                                    }
                                    ObjectPatProp::KeyValue {
                                        key: PropKey::Num(f, _),
                                        ..
                                    } => {
                                        // Numeric key: `Delete` canonicalizes
                                        // through the same key conversion as
                                        // the keyed accessors.
                                        self.emit(Op::Dup);
                                        self.compile_number(*f)?;
                                        self.emit(Op::Delete);
                                        self.emit(Op::Pop);
                                    }
                                    ObjectPatProp::Shorthand { name, .. } => {
                                        let ki = self.intern(name)?;
                                        self.emit(Op::Dup);
                                        self.emit(Op::LoadConst(ki));
                                        self.emit(Op::Delete);
                                        self.emit(Op::Pop);
                                    }
                                    _ => {}
                                }
                            }
                            match t {
                                // The collected object is never
                                // `undefined`, so no default check (the
                                // grammar forbids rest initializers).
                                Some(t) => self.compile_target_store_pop(&t)?,
                                None => self.compile_pat_bind_guarded(inner, kind, assign)?,
                            }
                        }
                    }
                }
                self.emit(Op::Pop);
                Ok(())
            }
            Pat::Rest(inner, _) => self.compile_pat_bind_guarded(inner, kind, assign),
            Pat::Member(e) => {
                // Member-expression destructuring target (`[a.b] = v`,
                // `({p: o.q} = v)`): resolve the reference at bind time —
                // after the RHS value was produced — and store through it.
                // This is the
                // IteratorDestructuringAssignmentEvaluation /
                // KeyedDestructuringAssignmentEvaluation order (ECMA-262
                // 13.15.5): the RHS evaluates before each element's target
                // reference. (Bare `a.b = v` keeps reference-before-RHS via
                // `compile_assign`, matching the spec's rule for
                // non-pattern LHS.)
                let t = self.compile_target_begin(e)?;
                self.compile_target_store_pop(&t)?;
                Ok(())
            }
        }
    }

    /// Destructuring assignment (`[a, b = 1, ...r] = rhs`,
    /// `({a, b: c, ...r} = rhs)`), ECMA-262 13.15.5
    /// (DestructuringAssignmentEvaluation). The parser leaves the pattern
    /// as an `Expr::Array`/`Expr::Object`; convert it back to a [`Pat`]
    /// and reuse the declaration-side binder in assignment mode
    /// (`PutValue`-style with-aware stores). `[..] -> [value]`: the RHS
    /// value stays on the stack as the assignment expression's value.
    fn compile_destructuring_assign(
        &mut self,
        left: &Expr,
        right: &Expr,
    ) -> Result<(), CompileError> {
        self.compile_expr(right)?;
        self.emit(Op::Dup);
        let pat = self.expr_to_pat(left)?;
        self.compile_pat_bind_guarded(&pat, VarKind::Var, true)?;
        Ok(())
    }

    /// Convert a destructuring-assignment LHS (`Expr::Array` /
    /// `Expr::Object`, as validated by the parser's `check_assign_target`)
    /// back into a [`Pat`]. Mirrors the parser's `expr_to_pat`; shapes
    /// the parser rejected surface here as [`CompileError::InvalidLhs`]
    /// (unreachable for successfully parsed programs).
    fn expr_to_pat(&mut self, e: &Expr) -> Result<Pat, CompileError> {
        match e {
            Expr::Ident(name, span) => Ok(Pat::Ident(name.clone(), *span)),
            Expr::Array(elems, span) => {
                let mut out = Vec::with_capacity(elems.len());
                for el in elems {
                    match el {
                        None => out.push(None),
                        Some(Expr::Spread(arg, s)) => {
                            let p = self.expr_to_pat(arg)?;
                            if matches!(p, Pat::Assign(..)) {
                                return Err(CompileError::InvalidLhs);
                            }
                            out.push(Some(ArrayPatElem::Pat(Pat::Rest(Box::new(p), *s))));
                        }
                        Some(x) => out.push(Some(ArrayPatElem::Pat(self.expr_to_pat(x)?))),
                    }
                }
                Ok(Pat::Array(out, *span))
            }
            Expr::Object(props, span) => {
                let mut out = Vec::with_capacity(props.len());
                for p in props {
                    match p {
                        PropDef::KeyValue {
                            key,
                            value,
                            span: ps,
                        } => {
                            let value = match value {
                                Expr::Assign {
                                    op: AssignOp::Assign,
                                    left,
                                    right,
                                    span: aspan,
                                } => {
                                    let inner = self.expr_to_pat(left)?;
                                    Pat::Assign(Box::new(inner), right.clone(), *aspan)
                                }
                                v => self.expr_to_pat(v)?,
                            };
                            out.push(ObjectPatProp::KeyValue {
                                key: key.clone(),
                                value,
                                span: *ps,
                            });
                        }
                        PropDef::Shorthand { name, span: ps } => {
                            out.push(ObjectPatProp::Shorthand {
                                name: name.clone(),
                                span: *ps,
                            })
                        }
                        PropDef::Spread(arg, ps) => {
                            let p = self.expr_to_pat(arg)?;
                            if matches!(p, Pat::Assign(..)) {
                                return Err(CompileError::InvalidLhs);
                            }
                            out.push(ObjectPatProp::Rest(Box::new(p), *ps));
                        }
                        PropDef::Method { .. }
                        | PropDef::Getter { .. }
                        | PropDef::Setter { .. } => return Err(CompileError::InvalidLhs),
                    }
                }
                Ok(Pat::Object(out, *span))
            }
            Expr::Assign {
                op: AssignOp::Assign,
                left,
                right,
                span,
            } => {
                let inner = self.expr_to_pat(left)?;
                Ok(Pat::Assign(Box::new(inner), right.clone(), *span))
            }
            Expr::Member { optional, .. } => {
                // ECMA-262 13.15.5: `MemberExpression` is a valid
                // destructuring assignment target (`[a.b] = x`,
                // `({p: o.q} = y)`).
                if *optional {
                    return Err(CompileError::InvalidLhs);
                }
                Ok(Pat::Member(Box::new(e.clone())))
            }
            _ => Err(CompileError::InvalidLhs),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opcode::disassemble;
    use crate::parser::parse_script;

    fn find_template<'a>(t: &'a FuncTemplate, name: &str) -> Option<&'a FuncTemplate> {
        for c in &t.consts {
            if let Constant::FuncTemplate(f) = c {
                if f.name == name {
                    return Some(f);
                }
                if let Some(found) = find_template(f, name) {
                    return Some(found);
                }
            }
        }
        None
    }

    #[test]
    fn closure_captures_upvalue() {
        let prog = parse_script(
            "function outer() { let x = 1; function inner() { return x; } return inner; }",
        )
        .unwrap();
        let t = compile(&prog, 0).unwrap().template;
        let inner = find_template(&t, "inner").expect("inner template");
        assert!(
            inner
                .upvalues
                .iter()
                .any(|u| matches!(u, UpvalueDesc::Local(_))),
            "inner should capture x from outer"
        );
        let d = disassemble(&inner.code, &inner.consts);
        assert!(d.contains("LoadUpvalue"), "expected upvalue load:\n{d}");
    }

    #[test]
    fn tdz_check_emitted() {
        let prog = parse_script("function f() { return x; let x = 1; }").unwrap();
        let t = compile(&prog, 0).unwrap().template;
        let f = find_template(&t, "f").expect("f template");
        let d = disassemble(&f.code, &f.consts);
        assert!(
            d.contains("ReferenceError: cannot access 'x' before initialization"),
            "TDZ check missing:\n{d}"
        );
        assert!(d.contains("Throw"), "TDZ throw missing:\n{d}");
    }

    #[test]
    fn try_finally_emits_handlers() {
        let prog = parse_script("try { f(); } finally { g(); }").unwrap();
        let t = compile(&prog, 0).unwrap().template;
        let d = disassemble(&t.code, &t.consts);
        assert!(d.contains("EnterTry"), "missing EnterTry:\n{d}");
        assert!(d.contains("LeaveTry"), "missing LeaveTry:\n{d}");
        // f() once, g() on the normal path and the handler path.
        assert_eq!(
            d.matches("Call 0").count(),
            3,
            "unexpected call count:\n{d}"
        );
    }

    #[test]
    fn class_fields_init_in_order() {
        let prog = parse_script("class C { a = 1; b = 2; c = 3; }").unwrap();
        let t = compile(&prog, 0).unwrap().template;
        let ctor = find_template(&t, "C").expect("ctor template");
        let d = disassemble(&ctor.code, &ctor.consts);
        let pa = d.find("; \"a\"").expect("field a");
        let pb = d.find("; \"b\"").expect("field b");
        let pc = d.find("; \"c\"").expect("field c");
        assert!(pa < pb && pb < pc, "fields out of order:\n{d}");
    }

    #[test]
    fn smoke_compile_script() {
        let prog = parse_script("let x = 1 + 2 * 3; x;").unwrap();
        let t = compile(&prog, 0).unwrap().template;
        let d = disassemble(&t.code, &t.consts);
        assert!(d.contains("Return"), "missing return:\n{d}");
        assert!(d.contains("Mul"), "missing multiply:\n{d}");
    }

    #[test]
    fn new_compiles_callee_then_args() {
        let prog = parse_script("new F(a, b)").unwrap();
        let t = compile(&prog, 0).unwrap().template;
        let d = disassemble(&t.code, &t.consts);
        // `new` with two args emits a single New op.
        assert!(d.contains("New 2"), "missing New 2:\n{d}");
    }

    #[test]
    fn new_paren_call_not_flattened() {
        // `new (F(a))` must construct the call result with zero args.
        let prog = parse_script("new (F(a))").unwrap();
        let t = compile(&prog, 0).unwrap().template;
        let d = disassemble(&t.code, &t.consts);
        assert!(d.contains("New 0"), "missing New 0:\n{d}");
        assert!(d.contains("Call"), "missing inner call:\n{d}");
    }

    #[test]
    fn catch_param_tdz() {
        // The catch param is TDZ-guarded like a let.
        let prog = parse_script("try { f(); } catch (e) { e; }").unwrap();
        let t = compile(&prog, 0).unwrap().template;
        let d = disassemble(&t.code, &t.consts);
        assert!(d.contains("EnterTry"), "missing EnterTry:\n{d}");
    }

    #[test]
    fn try_with_nested_block_scopes() {
        // A try whose body and handler contain blocks must resolve all
        // scope slots without panicking.
        let prog = parse_script("try { { let a = 1; f(a); } } catch (e) { { let b = 2; g(b); } }")
            .unwrap();
        let t = compile(&prog, 0).unwrap().template;
        let d = disassemble(&t.code, &t.consts);
        assert!(d.contains("EnterTry"), "missing EnterTry:\n{d}");
        assert!(d.contains("LeaveTry"), "missing LeaveTry:\n{d}");
    }

    #[test]
    fn for_let_closure_capture() {
        // `for (let i ...)` with a closure must extract the body.
        let prog =
            parse_script("var fs = []; for (let i = 0; i < 3; i++) { fs.push(() => i); }").unwrap();
        let t = compile(&prog, 0).unwrap().template;
        // The extracted body becomes a hidden arrow template.
        let found = t.consts.iter().any(|c| match c {
            Constant::FuncTemplate(f) => f.name.starts_with("#for_body"),
            _ => false,
        });
        assert!(found, "missing extracted for-body template");
    }

    #[test]
    fn catch_shadows_outer() {
        // A catch param shadows an outer binding of the same name.
        let prog = parse_script("let e = 1; try { f(); } catch (e) { e; }").unwrap();
        let t = compile(&prog, 0).unwrap().template;
        let d = disassemble(&t.code, &t.consts);
        assert!(d.contains("EnterTry"), "missing EnterTry:\n{d}");
    }

    /// End-to-end driver for the extracted-loop regression tests below.
    fn eval_to_int(src: &str) -> i32 {
        let v = crate::vm::Vm::new()
            .eval(src)
            .unwrap_or_else(|e| panic!("eval failed for {src:?}: {e:?}"));
        match v {
            crate::value::Value::Int(i) => i,
            other => panic!("expected int for {src:?}, got {other:?}"),
        }
    }

    #[test]
    fn extracted_loop_fn_decl_captures_per_iteration_lexical() {
        // Regression (M2): a function declaration in a per-iteration
        // extracted loop body must close over the body's per-iteration
        // lexicals. `declare_body_vars` used to compile the template in
        // the enclosing scope before the extracted arrow's predeclare
        // ran, so `y` resolved as a global and the call threw
        // `ReferenceError: y is not defined`.
        let r = eval_to_int(
            "var r; for (let i = 0; i < 1; i++) { function F() { return y; } let y = 5; r = F(); } r;",
        );
        assert_eq!(r, 5);
    }

    #[test]
    fn extracted_loop_fn_decl_per_iteration_binding() {
        // Each iteration instantiates a fresh closure capturing its own
        // iteration environment (Annex B block-level semantics).
        let r = eval_to_int(
            "var fs = []; for (let i = 0; i < 3; i++) { function F() { return i; } fs.push(F); }\
             fs[0]() + fs[1]() + fs[2]();",
        );
        assert_eq!(r, 3);
        // The `for`-`of` extraction path shares the mechanism.
        let r = eval_to_int(
            "var r; for (let x of [1]) { function F() { return y; } let y = 9; r = F(); } r;",
        );
        assert_eq!(r, 9);
    }

    #[test]
    fn extracted_loop_fn_decl_tdz_still_enforced() {
        // The per-iteration closure captures the lexical's slot, not a
        // snapshot: calling before the `let` initializer runs still throws.
        let r = eval_to_int(
            "var r = 0; try { for (let i = 0; i < 1; i++) { function F() { return y; } F(); let y = 5; } }\
             catch (e) { r = 1; } r;",
        );
        assert_eq!(r, 1);
    }

    #[test]
    fn const_check_second_site_jumps_to_throw_when_initialized() {
        // The shared const throw stub fires when the initialized flag is
        // true, so a second (or later) check site must JumpIfTrue *to* the
        // stub; JumpIfFalse there would invert the check.
        let prog = parse_script("function f() { const x = 1; x = 2; x = 3; }").unwrap();
        let t = compile(&prog, 0).unwrap().template;
        let f = find_template(&t, "f").expect("f template");
        let d = disassemble(&f.code, &f.consts);
        let throw_pc: usize = d
            .lines()
            .find(|l| l.contains("TypeError: assignment to constant variable"))
            .and_then(|l| l.split(':').next())
            .and_then(|pc| pc.trim().parse().ok())
            .expect("shared const throw stub");
        let target = format!("-> {throw_pc:04}");
        let mut saw_true = false;
        for line in d.lines().filter(|l| l.contains(&target)) {
            if line.contains("JumpIfTrue") {
                saw_true = true;
            } else {
                panic!("jump to const throw stub is not JumpIfTrue:\n{line}\n{d}");
            }
        }
        assert!(saw_true, "no JumpIfTrue to the const throw stub:\n{d}");
    }

    #[test]
    fn finally_inline_on_abrupt_exit_replays_scopes() {
        // Regression: `return` inside a try block inlined the finalizer via
        // `unwind_protected` without scope/cursor management, panicking
        // with "declare walk / codegen block-scope mismatch" when the
        // finalizer contained a block (found via testharness.js).
        let prog =
            parse_script("function f() { try { if (a) {} return rv; } finally { if (b) {} } }")
                .unwrap();
        let t = compile(&prog, 0).unwrap().template;
        let f = find_template(&t, "f").expect("f template");
        // Must compile without panicking; the function has no observable
        // behavior to assert beyond successful compilation.
        assert!(!f.code.is_empty());
    }

    #[test]
    fn finally_resolves_against_outer_scope_on_abrupt_exit() {
        // The abrupt-path finalizer inline must resolve names against the
        // try-statement (outer) scope, not the try block's live scopes.
        let prog = parse_script(
            r#"function f() {
                let x = "outer";
                try { { let x = "inner"; return 1; } }
                finally { return x; }
            }"#,
        )
        .unwrap();
        let t = compile(&prog, 0).unwrap().template;
        let f = find_template(&t, "f").expect("f template");
        assert!(!f.code.is_empty());
    }

    #[test]
    fn nested_finally_in_finally_body_runs_on_abrupt_exit() {
        // `return` inside a try nested in a finally body must still inline
        // the *inner* finalizer for the abrupt path; only the region
        // currently being inlined (the outer finally) is skipped.
        let prog = parse_script(
            "function f() { try { a(); } finally { try { return 1; } finally { b(); } } }",
        )
        .unwrap();
        let t = compile(&prog, 0).unwrap().template;
        let f = find_template(&t, "f").expect("f template");
        let d = disassemble(&f.code, &f.consts);
        // The outer finally body is emitted on its normal and handler
        // paths; each copy inlines the inner finalizer three times: the
        // abrupt (`return`) path, the normal path, and the handler path.
        // If the inner finalizer were skipped for the abrupt path there
        // would be only four copies.
        let copies = d.matches("; \"b\"").count();
        assert_eq!(copies, 6, "inner finalizer copies:\n{d}");
        // The abrupt-path copies run the finalizer and then return
        // (normal path jumps, handler path rethrows).
        let lines: Vec<&str> = d.lines().collect();
        let abrupt = lines
            .windows(3)
            .filter(|w| {
                w[0].contains("Call 0") && w[1].ends_with("Pop") && w[2].ends_with("Return")
            })
            .count();
        assert_eq!(abrupt, 2, "abrupt-path finalizer+return sequences:\n{d}");
    }

    /// DoS regression: shapes the parser builds iteratively (left-deep
    /// binary/member/call chains, flat sequences) reach the compiler
    /// arbitrarily deep and must fail cleanly with `DepthExceeded`, never
    /// abort. Each trigger runs in a subprocess (see
    /// `dos_nesting_regressions`) so a regression kills only the child.
    /// Each trigger runs in a subprocess on a thread with an explicit stack
    /// size (8 MiB by default, overridable with `BOWSER_DOS_STACK_KIB`): a
    /// stack overflow aborts the process, so the probe must not run on the
    /// harness thread itself. The 2 MiB probes pin the native-stack guard
    /// in `Compiler::guarded`: small stacks must fail clean with
    /// `DepthExceeded`, never abort.
    #[test]
    fn dos_probe_helper() {
        let trigger = match std::env::var("BOWSER_DOS_TRIGGER") {
            Ok(t) => t,
            // Not launched as a probe: nothing to do.
            Err(_) => return,
        };
        let stack = std::env::var("BOWSER_DOS_STACK_KIB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|kib| kib * 1024)
            .unwrap_or(8 * 1024 * 1024);
        let h = std::thread::Builder::new()
            .name(format!("dos-probe-{trigger}"))
            .stack_size(stack)
            .spawn(move || run_dos_trigger(&trigger))
            .expect("spawn dos probe thread");
        // A stack overflow aborts the process before join returns; a
        // panic here means the trigger produced the wrong result.
        h.join().expect("dos probe thread panicked");
    }

    fn run_dos_trigger(trigger: &str) {
        match trigger {
            "add2000" | "member2000" | "call2000" | "bind2000" => {
                let src = match trigger {
                    "add2000" => format!("{}1;", "1+".repeat(2000)),
                    "member2000" => format!("a{};", ".b".repeat(2000)),
                    "bind2000" => format!("f{};", ".bind(null)".repeat(2000)),
                    _ => format!("f{};", "()".repeat(2000)),
                };
                let prog = parse_script(&src).expect("trigger must parse");
                match trigger {
                    // Binary `+` spines are still compiled recursively;
                    // the depth guard turns them into a clean error.
                    "add2000" => match compile(&prog, 0) {
                        Err(CompileError::DepthExceeded) => {}
                        r => panic!("expected DepthExceeded for {trigger}, got {r:?}"),
                    },
                    // Member/call/bind spines compile iteratively (no
                    // depth limit); the requirement is simply no abort.
                    _ => {
                        compile(&prog, 0).unwrap_or_else(|e| {
                            panic!("{trigger} must compile cleanly, got {e:?}")
                        });
                    }
                }
            }
            // Flat sequence: parses and compiles fine.
            "seq20000" => {
                let src = format!("{}1;", "1,".repeat(20_000));
                let prog = parse_script(&src).expect("trigger must parse");
                compile(&prog, 0).expect("flat sequence must compile");
            }
            // Deep expression in a `for` condition: the loop analysis
            // (`for_needs_extraction`) scans it for closures before
            // compiling. The scan is depth-bounded; compiling the deep
            // chain then fails cleanly.
            "closure_scan" => {
                let src = format!("for (let i = 0; {}1; i++) {{}}", "1+".repeat(20_000));
                let prog = parse_script(&src).expect("trigger must parse");
                match compile(&prog, 0) {
                    Err(CompileError::DepthExceeded) => {}
                    r => panic!("expected DepthExceeded for {trigger}, got {r:?}"),
                }
            }
            // Bypass the parser entirely: drive `compile_pat_bind`
            // itself on a directly-constructed deep pattern. The guard
            // fires before any leaf is bound, so no declarations are
            // needed; a function frame is pushed so `emit` works.
            "pat_bind_direct" => {
                let span = Span { start: 0, end: 0 };
                let mut pat = Pat::Ident("x".to_string(), span);
                for _ in 0..600 {
                    pat = Pat::Array(vec![Some(ArrayPatElem::Pat(pat))], span);
                }
                let mut cx = Compiler::new(0);
                cx.push_function("test".to_string(), 0, false, false, false, true, false);
                match cx.compile_pat_bind(&pat, VarKind::Let) {
                    Err(CompileError::DepthExceeded) => {}
                    r => panic!("expected DepthExceeded for {trigger}, got {r:?}"),
                }
            }
            _ => panic!("unknown BOWSER_DOS_TRIGGER={trigger}"),
        }
    }

    #[test]
    fn dos_nesting_regressions() {
        for (trigger, stack_kib) in [
            ("add2000", 8192),
            ("member2000", 8192),
            ("call2000", 8192),
            ("bind2000", 8192),
            ("seq20000", 8192),
            ("closure_scan", 8192),
            ("pat_bind_direct", 8192),
            // Small-stack probes: this is the environment where a pure
            // depth counter overflows before tripping (see
            // `MAX_COMPILE_NATIVE_STACK`); the native-stack guard must
            // turn these into a clean `DepthExceeded`, never an abort.
            ("add2000", 2048),
            ("member2000", 2048),
            ("call2000", 2048),
            ("bind2000", 2048),
            // Tiny-stack probes: the iterative chain compiler must
            // survive even on 256 KiB stacks (no abort, clean compile).
            ("member2000", 256),
            ("call2000", 256),
            ("bind2000", 256),
        ] {
            let exe = std::env::current_exe().expect("test executable path");
            let out = std::process::Command::new(&exe)
                .arg("--exact")
                .arg("compiler::tests::dos_probe_helper")
                .arg("--nocapture")
                .env("BOWSER_DOS_TRIGGER", trigger)
                .env("BOWSER_DOS_STACK_KIB", stack_kib.to_string())
                .output()
                .expect("failed to spawn probe subprocess");
            assert!(
                out.status.success(),
                "trigger `{trigger}` (stack {stack_kib} KiB) did not exit cleanly (abort or assertion failure):\n--- stdout ---\n{}\n--- stderr ---\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
        }
    }

    // -- derived classes ----------------------------------------------------
    // Regression tests: a derived class's instances must link to the
    // derived `prototype` (ECMA-262 OrdinaryCreateFromConstructor), even
    // when the heritage is a *native* builtin (`Object`, `Array`,
    // `Function`) whose constructor used to ignore `newTarget` and stamp
    // %Object.prototype% / %Array.prototype% / %Function.prototype% on
    // every instance — making installed prototype methods unreachable
    // (`new D().m()` threw "callee is not a function"). A null heritage
    // is also legal: it used to throw at class-definition time.

    /// Evaluate JS in a full VM and stringify the completion value.
    fn eval_str(source: &str) -> String {
        let mut vm = crate::vm::Vm::new();
        let v = vm
            .eval(source)
            .unwrap_or_else(|e| panic!("eval failed: {e}"));
        crate::builtin_object_array::to_string_js(vm.heap_mut(), v)
            .map(|w| w.to_string())
            .unwrap_or_else(|e| panic!("to_string failed: {e:?}"))
    }

    #[test]
    fn derived_extends_object_installs_prototype_methods() {
        // The reported case: `D.prototype.m` was installed but instances
        // never saw it.
        assert_eq!(
            eval_str("class D extends Object { m() { return 2; } } new D().m()"),
            "2"
        );
        assert_eq!(
            eval_str("class D extends Object { m() {} } new D() instanceof D"),
            "true"
        );
        assert_eq!(
            eval_str(
                "class D extends Object { m() {} } \
                 Object.getPrototypeOf(new D()) === D.prototype"
            ),
            "true"
        );
    }

    #[test]
    fn derived_extends_object_explicit_constructor() {
        assert_eq!(
            eval_str(
                "class D extends Object { \
                   constructor(a) { super(); this.a = a; } \
                   get() { return this.a; } \
                 } new D(5).get()"
            ),
            "5"
        );
    }

    #[test]
    fn derived_extends_object_member_kinds() {
        assert_eq!(
            eval_str("class D extends Object { get g() { return 10; } } new D().g"),
            "10"
        );
        assert_eq!(
            eval_str(
                "class D extends Object { \
                   set s(v) { this._s = v * 2; } \
                   get s() { return this._s; } \
                 } var d = new D(); d.s = 21; d.s"
            ),
            "42"
        );
        assert_eq!(
            eval_str(
                "class D extends Object { #p = 1; get() { return this.#p; } } \
                 new D().get()"
            ),
            "1"
        );
        assert_eq!(
            eval_str(
                "class D extends Object { static s() { return 3; } m() { return 4; } } \
                 D.s() + new D().m()"
            ),
            "7"
        );
        assert_eq!(
            eval_str("class D extends Object { x = 9; m() { return this.x; } } new D().m()"),
            "9"
        );
        assert_eq!(
            eval_str(
                "class D extends Object { static { this.z = 4; } m() { return 1; } } \
                 D.z + new D().m()"
            ),
            "5"
        );
    }

    #[test]
    fn derived_extends_object_class_expressions() {
        assert_eq!(
            eval_str("var C = class N extends Object { m() { return 7; } }; new C().m()"),
            "7"
        );
        assert_eq!(
            eval_str("var C = class extends Object { m() { return 8; } }; new C().m()"),
            "8"
        );
        assert_eq!(
            eval_str(
                "var C = class extends (function() { return Object; })() { \
                   m() { return 13; } \
                 }; new C().m()"
            ),
            "13"
        );
    }

    #[test]
    fn derived_extends_array() {
        assert_eq!(
            eval_str("class D extends Array { m() { return 2; } } new D().m()"),
            "2"
        );
        assert_eq!(
            eval_str(
                "class D extends Array { m() {} } \
                 Object.getPrototypeOf(new D()) === D.prototype"
            ),
            "true"
        );
        // Array exotic behavior survives the newTarget fix.
        assert_eq!(
            eval_str("class D extends Array {} var d = new D(); d.push(1, 2); d.length"),
            "2"
        );
        assert_eq!(
            eval_str("Object.getPrototypeOf(new Array()) === Array.prototype"),
            "true"
        );
    }

    #[test]
    fn derived_extends_function() {
        assert_eq!(
            eval_str(
                "class D extends Function { m() { return 2; } } \
                 Object.getPrototypeOf(new D()) === D.prototype"
            ),
            "true"
        );
        assert_eq!(
            eval_str("Object.getPrototypeOf(new Function()) === Function.prototype"),
            "true"
        );
    }

    #[test]
    fn derived_extends_null() {
        // Class definition must not throw on a null heritage.
        assert_eq!(eval_str("class D extends null {} typeof D"), "function");
        assert_eq!(
            eval_str("class D extends null {} Object.getPrototypeOf(D.prototype) === null"),
            "true"
        );
        // Spec: constructorParent stays %Function.prototype%.
        assert_eq!(
            eval_str("class D extends null {} Object.getPrototypeOf(D) === Function.prototype"),
            "true"
        );
        // Instantiation throws TypeError (super constructor is null).
        assert_eq!(
            eval_str(
                "class D extends null {} var t = 'no'; \
                 try { new D(); } catch (e) { t = e instanceof TypeError ? 'yes' : 'other'; } t"
            ),
            "yes"
        );
    }

    #[test]
    fn derived_two_level_through_object() {
        assert_eq!(
            eval_str(
                "class B extends Object { bm() { return 1; } } \
                 class D extends B { dm() { return 2; } } \
                 var d = new D(); d.bm() + d.dm()"
            ),
            "3"
        );
    }

    #[test]
    fn plain_object_construction_unchanged() {
        assert_eq!(
            eval_str("Object.getPrototypeOf(new Object()) === Object.prototype"),
            "true"
        );
        assert_eq!(
            eval_str("Object.getPrototypeOf(Object()) === Object.prototype"),
            "true"
        );
    }

    // -- strict-mode `for (var x in/of ...)` ---------------------------------
    // Regression tests: strict-mode `for (var <ident> of/in ...)` used to
    // throw `ReferenceError: assignment to undeclared variable` because the
    // declare walk never declared the `var` binding at function scope;
    // only sloppy mode was covered, which silently tolerated the missing
    // binding.

    #[test]
    fn strict_for_var_of_declares_binding() {
        assert_eq!(
            eval_str("\"use strict\"; var s = 0; for (var y of [1, 2]) { s += y; } s;"),
            "3"
        );
    }

    #[test]
    fn strict_for_var_in_declares_binding() {
        assert_eq!(
            eval_str(
                "\"use strict\"; var ks = []; for (var k in {a: 1}) { ks.push(k); } ks.join(',');"
            ),
            "a"
        );
    }

    #[test]
    fn strict_for_var_of_hoists_to_function_scope() {
        // `var` in the for-of head is function-scoped, not iteration-scoped.
        assert_eq!(
            eval_str("\"use strict\"; function f() { for (var y of [1, 2]) {} return y; } f();"),
            "2"
        );
        assert_eq!(
            eval_str("\"use strict\"; function f() { for (var k in {b: 2}) {} return k; } f();"),
            "b"
        );
    }

    #[test]
    fn strict_for_var_of_wpt_helper_shape() {
        // The exact shape that blocked animations/border-spacing-interpolation.html.
        assert_eq!(
            eval_str(
                "\"use strict\"; var interpolationMethods = ['m1', 'm2']; \
                 var seen = []; \
                 for (var interpolationMethod of interpolationMethods) { seen.push(interpolationMethod); } \
                 seen.join(',');"
            ),
            "m1,m2"
        );
    }

    #[test]
    fn sloppy_for_var_of_still_works() {
        assert_eq!(
            eval_str("var s = 0; for (var y of [1, 2]) { s += y; } s;"),
            "3"
        );
        assert_eq!(
            eval_str("var ks = []; for (var k in {a: 1}) { ks.push(k); } ks.join(',');"),
            "a"
        );
    }

    #[test]
    fn for_let_of_unaffected() {
        // `let` still declares a per-iteration lexical binding.
        assert_eq!(
            eval_str("\"use strict\"; var s = 0; for (let y of [1, 2]) { s += y; } s;"),
            "3"
        );
        // ... and the `let` binding is not visible after the loop.
        assert_eq!(
            eval_str(
                "\"use strict\"; for (let y of [1]) {} \
                 var t = 'no'; try { y; } catch (e) { t = 'threw'; } t;"
            ),
            "threw"
        );
    }
}
