//! Spanned JavaScript AST (ES2024 surface).
//!
//! Produced by [`crate::parser`]. Shapes follow the task contract and stay
//! close to ESTree conventions so the compiler worker can pattern-match
//! directly. Every node that corresponds to source text carries a
//! [`Span`](crate::lexer::Span); spans are byte offsets into the original
//! source.
//!
//! # Deviations from the draft skeleton
//!
//! Four additive fixes were required for a complete grammar surface:
//!
//! * [`Stmt::Labeled`] — labelled statements (`a: for(;;) break a;`).
//! * [`Stmt::With`] — sloppy-mode `with` (rejected in strict mode).
//! * [`Expr::Call`] gains `optional: bool` — `a?.()` is otherwise
//!   unrepresentable.
//! * [`Expr::Yield`] gains `delegate: bool` — `yield*` is otherwise
//!   unrepresentable.

use crate::lexer::Span;
use crate::wtf8::Wtf8Buf;

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------

/// A parsed module (`parse`) or script (`parse_script`).
#[derive(Clone, Debug)]
pub struct Program {
    pub body: Vec<Stmt>,
    /// Script-level strictness: true for modules (always strict) and for
    /// scripts whose directive prologue contains `"use strict"`. The parser
    /// computes this from the raw source form, so escaped spellings like
    /// `"\u{75}se strict"` do not count.
    pub strict: bool,
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

/// A statement.
#[derive(Clone, Debug)]
pub enum Stmt {
    /// `expr;`
    Expr(Expr),
    /// `{ stmts }`
    Block(Vec<Stmt>),
    /// `var/let/const a = 1, [b] = c;`
    VarDecl { kind: VarKind, decls: Vec<VarDecl> },
    /// `function f() {}` (also `async function`, `function*`).
    FunctionDecl {
        name: String,
        params: Vec<Param>,
        body: Vec<Stmt>,
        is_async: bool,
        is_generator: bool,
        /// Final strictness: inherited strictness plus a `"use strict"`
        /// directive in the body (raw source form, per the parser).
        strict: bool,
    },
    /// `class C extends D {}` (statement position; name always present
    /// except `export default class {}`).
    ClassDecl {
        name: Option<String>,
        superclass: Option<Expr>,
        body: Vec<ClassMember>,
    },
    /// `return expr?;`
    Return(Option<Expr>),
    /// `if (c) t else e?`
    If {
        cond: Expr,
        then_b: Box<Stmt>,
        else_b: Option<Box<Stmt>>,
    },
    /// `while (c) body`
    While { cond: Expr, body: Box<Stmt> },
    /// `do body while (c);`
    DoWhile { body: Box<Stmt>, cond: Expr },
    /// `for (init; cond; update) body`
    For {
        init: Option<ForInit>,
        cond: Option<Expr>,
        update: Option<Expr>,
        body: Box<Stmt>,
    },
    /// `for ([kind] lhs in|of rhs) body`. `is_of` selects `of` vs `in`.
    /// `kind` is the declaration kind for `for (let x of ...)`; for a bare
    /// `for (x of ...)` the parser still records `VarKind::Var` — check
    /// `lhs` to distinguish.
    ForInOf {
        kind: VarKind,
        lhs: ForLhs,
        rhs: Expr,
        body: Box<Stmt>,
        is_of: bool,
    },
    /// `break label?;`
    Break(Option<String>),
    /// `continue label?;`
    Continue(Option<String>),
    /// `switch (d) { case t: ...; default: ... }`
    Switch { disc: Expr, cases: Vec<SwitchCase> },
    /// `try b catch (p)? b? finally b?`
    Try {
        block: Vec<Stmt>,
        handler: Option<CatchClause>,
        finalizer: Option<Vec<Stmt>>,
    },
    /// `throw expr;`
    Throw(Expr),
    /// `debugger;`
    Debugger,
    /// `;`
    Empty,
    /// `label: stmt`
    Labeled { label: String, body: Box<Stmt> },
    /// `with (obj) stmt` — sloppy mode only; rejected in strict mode.
    With { obj: Expr, body: Box<Stmt> },
    /// `import ...`
    Import(ImportDecl),
    /// `export ...`
    Export(ExportDecl),
}

/// `var` / `let` / `const`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarKind {
    Var,
    Let,
    Const,
    /// `using x = ...` (explicit resource management). The parser
    /// desugars these to `try`/`finally` + hidden intrinsics before the
    /// compiler runs, so codegen only sees `Let`.
    Using,
    /// `await using x = ...`.
    AwaitUsing,
}

/// One declarator: `id = init?`.
#[derive(Clone, Debug)]
pub struct VarDecl {
    pub id: Pat,
    pub init: Option<Expr>,
    pub span: Span,
}

/// `case test?: stmts` (`test == None` is `default:`).
#[derive(Clone, Debug)]
pub struct SwitchCase {
    pub test: Option<Expr>,
    pub body: Vec<Stmt>,
}

/// `catch (param)? { body }`.
#[derive(Clone, Debug)]
pub struct CatchClause {
    pub param: Option<Param>,
    pub body: Vec<Stmt>,
}

/// `for` initializer: a declaration or an expression.
#[derive(Clone, Debug)]
pub enum ForInit {
    Var(VarKind, Vec<VarDecl>),
    Expr(Expr),
}

/// Left-hand side of `for-in` / `for-of`.
#[derive(Clone, Debug)]
pub enum ForLhs {
    /// `for (x in y)` — bare identifier.
    Ident(String, Span),
    /// `for ([a] of b)`, `for ({c} in d)`.
    Pat(Pat),
    /// `for (a.b in c)` / `for (a[b] of d)`.
    Member(Box<Expr>),
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

/// An expression.
#[derive(Clone, Debug)]
pub enum Expr {
    /// `foo`
    Ident(String, Span),
    /// `#foo` — private name reference (member access or `in`).
    PrivateIdent(String, Span),
    /// `1`, `"s"`, `true`, `null`, `10n`
    Lit(Literal, Span),
    /// `[a, , ...b]` — `None` elements are elisions.
    Array(Vec<Option<Expr>>, Span),
    /// `{a: 1, ...b}`
    Object(Vec<PropDef>, Span),
    /// `function name?() {}` (also async / generator).
    Function {
        name: Option<String>,
        params: Vec<Param>,
        body: Vec<Stmt>,
        is_async: bool,
        is_generator: bool,
        /// Final strictness: inherited strictness plus a `"use strict"`
        /// directive in the body (raw source form, per the parser).
        strict: bool,
        span: Span,
    },
    /// `class name? extends super? {}`
    Class {
        name: Option<String>,
        superclass: Option<Box<Expr>>,
        body: Vec<ClassMember>,
        span: Span,
    },
    /// `(a, b = 1) => body`
    Arrow {
        params: Vec<Param>,
        body: ArrowBody,
        is_async: bool,
        /// Final strictness: inherited strictness plus a `"use strict"`
        /// directive in a block body (raw source form, per the parser).
        /// Arrows never get their own `this` binding either way.
        strict: bool,
        span: Span,
    },
    /// `f(a, ...b)` — `optional` marks `f?.(...)`.
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
        optional: bool,
        span: Span,
    },
    /// `new F(a)`
    New {
        callee: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
    /// `a.b`, `a[b]`, `a?.b` — `optional` marks `?.`.
    Member {
        obj: Box<Expr>,
        prop: Box<Expr>,
        computed: bool,
        optional: bool,
        span: Span,
    },
    /// `-x`, `typeof x`, `delete x.y`
    Unary {
        op: UnaryOp,
        arg: Box<Expr>,
        span: Span,
    },
    /// `++x`, `x--`
    Update {
        op: UpdateOp,
        arg: Box<Expr>,
        prefix: bool,
        span: Span,
    },
    /// `a + b`, `a ** b`, `x in y`, `x instanceof Y`
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
        span: Span,
    },
    /// `a && b`, `a || b`, `a ?? b`
    Logical {
        op: LogicalOp,
        left: Box<Expr>,
        right: Box<Expr>,
        span: Span,
    },
    /// `a = b`, `a += b`, `a ??= b`
    Assign {
        op: AssignOp,
        left: Box<Expr>,
        right: Box<Expr>,
        span: Span,
    },
    /// `c ? t : e`
    Conditional {
        cond: Box<Expr>,
        then_e: Box<Expr>,
        else_e: Box<Expr>,
        span: Span,
    },
    /// `a, b, c`
    Sequence(Vec<Expr>, Span),
    /// `` `a${b}c` ``
    Template {
        parts: Vec<TemplatePart>,
        span: Span,
    },
    /// `` tag`a${b}c` ``
    TaggedTemplate {
        tag: Box<Expr>,
        parts: Vec<TemplatePart>,
        span: Span,
    },
    /// `yield arg?`, `yield* arg` (`delegate` marks `*`).
    Yield {
        arg: Option<Box<Expr>>,
        delegate: bool,
        span: Span,
    },
    /// `await x`
    Await(Box<Expr>, Span),
    /// `...x` (call args, array literals)
    Spread(Box<Expr>, Span),
    /// `this`
    This(Span),
    /// `super` (property access is `Member`; bare `super` only valid in
    /// constructors via `super(...)` calls)
    Super(Span),
    /// `new.target`
    NewTarget(Span),
    /// `import.meta`
    ImportMeta(Span),
    /// `/pat/flags` — the parser represents dynamic `import(x)` as
    /// `Call { callee: Ident("import"), .. }`.
    Regex {
        pattern: String,
        flags: String,
        span: Span,
    },
}

/// A template chunk: cooked string or interpolated expression.
#[derive(Clone, Debug)]
pub enum TemplatePart {
    /// A literal chunk. `cooked` is `None` when the chunk contains an
    /// invalid escape sequence (only legal in tagged templates); `raw` is
    /// the source text without delimiters. `cooked` is WTF-8: escapes like
    /// `` `\uD800` `` produce lone surrogates.
    Str {
        cooked: Option<Wtf8Buf>,
        raw: String,
    },
    Expr(Expr),
}

/// Arrow body: `=> expr` or `=> { stmts }`.
#[derive(Clone, Debug)]
pub enum ArrowBody {
    Expr(Box<Expr>),
    Block(Vec<Stmt>),
}

/// Literals.
#[derive(Clone, Debug)]
pub enum Literal {
    /// value (raw source text dropped: nothing consumed it).
    Number(f64),
    /// raw source text including the trailing `n`.
    BigInt(String),
    /// cooked string value (WTF-8: may hold lone surrogates from `\u`
    /// escapes).
    Str(Wtf8Buf),
    Bool(bool),
    Null,
}

// ---------------------------------------------------------------------------
// Patterns (destructuring)
// ---------------------------------------------------------------------------

/// A binding pattern.
#[derive(Clone, Debug)]
pub enum Pat {
    /// `a`
    Ident(String, Span),
    /// `[a, , b = 1, ...rest]` — `None` is an elision.
    Array(Vec<Option<ArrayPatElem>>, Span),
    /// `{a, b: c, ...rest}`
    Object(Vec<ObjectPatProp>, Span),
    /// `...rest`
    Rest(Box<Pat>, Span),
    /// `a = 1` (pattern with default)
    Assign(Box<Pat>, Box<Expr>, Span),
    /// `a.b` / `a[b]` — a member-expression destructuring *assignment*
    /// target. ECMA-262 13.15.5 allows `MemberExpression` as an
    /// assignment-pattern target; binding declarations never produce
    /// this (the parser's `parse_pat` has no member production). The
    /// wrapped `Expr` carries its own span.
    Member(Box<Expr>),
}

/// Array pattern element (holes are represented as `None` by the parser).
#[derive(Clone, Debug)]
pub enum ArrayPatElem {
    Pat(Pat),
}

/// Object pattern property.
#[derive(Clone, Debug)]
pub enum ObjectPatProp {
    /// `{a: b}`, `{[k]: v}`, `{"s": v}`
    KeyValue {
        key: PropKey,
        value: Pat,
        span: Span,
    },
    /// `{a}`, `{a = 1}` (default via `Pat::Assign`)
    Shorthand { name: String, span: Span },
    /// `{...rest}`
    Rest(Box<Pat>, Span),
}

/// A function parameter: `pat` or `...pat`; defaults via `Pat::Assign`.
#[derive(Clone, Debug)]
pub struct Param {
    pub pat: Pat,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// Object / class members
// ---------------------------------------------------------------------------

/// A property key: `a`, `"s"`, `1`, `[expr]`.
#[derive(Clone, Debug)]
pub enum PropKey {
    Ident(String, Span),
    /// cooked value (WTF-8: `{["\uD800"]: 1}` is legal).
    Str(Wtf8Buf),
    /// value.
    Num(f64, Span),
    Computed(Expr),
}

/// An object literal property.
#[derive(Clone, Debug)]
pub enum PropDef {
    /// `{a: 1}`, `{[k]: v}`, `{__proto__: x}`
    KeyValue {
        key: PropKey,
        value: Expr,
        span: Span,
    },
    /// `{a}`
    Shorthand { name: String, span: Span },
    /// `{...x}`
    Spread(Expr, Span),
    /// `{m() {}}`, `{*g() {}}`, `{async m() {}}`
    Method {
        key: PropKey,
        params: Vec<Param>,
        body: Vec<Stmt>,
        is_async: bool,
        is_generator: bool,
        /// Final strictness: inherited strictness plus a `"use strict"`
        /// directive in the body (raw source form, per the parser).
        strict: bool,
        span: Span,
    },
    /// `{get x() {}}`
    Getter {
        key: PropKey,
        body: Vec<Stmt>,
        /// Final strictness: inherited strictness plus a `"use strict"`
        /// directive in the body (raw source form, per the parser).
        strict: bool,
        span: Span,
    },
    /// `{set x(v) {}}`
    Setter {
        key: PropKey,
        param: Param,
        body: Vec<Stmt>,
        /// Final strictness: inherited strictness plus a `"use strict"`
        /// directive in the body (raw source form, per the parser).
        strict: bool,
        span: Span,
    },
}

/// A class element.
#[derive(Clone, Debug)]
pub enum ClassMember {
    /// methods incl. constructor/getter/setter, private methods, statics.
    Method {
        key: ClassKey,
        params: Vec<Param>,
        body: Vec<Stmt>,
        is_async: bool,
        is_generator: bool,
        is_static: bool,
        kind: MethodKind,
    },
    /// `x = 1;`, `#x;`, `static y;`
    Field {
        key: ClassKey,
        init: Option<Expr>,
        is_static: bool,
    },
    /// `static { ... }`
    StaticBlock { body: Vec<Stmt> },
}

/// `constructor` / method / getter / setter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MethodKind {
    Method,
    Getter,
    Setter,
    Constructor,
}

/// A class member key: public (incl. computed) or `#private`.
#[derive(Clone, Debug)]
pub enum ClassKey {
    Public(PropKey),
    Private(String, Span),
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

/// `-x`, `+x`, `!x`, `~x`, `typeof x`, `void x`, `delete x`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Minus,
    Plus,
    Bang,
    Tilde,
    Typeof,
    Void,
    Delete,
}

/// `++`, `--`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateOp {
    Inc,
    Dec,
}

/// `+ - * / % **`, comparisons, shifts, bitwise, `in`, `instanceof`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Eq,
    Ne,
    StrictEq,
    StrictNe,
    Lt,
    LtEq,
    Gt,
    GtEq,
    LShift,
    RShift,
    URShift,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    BitAnd,
    BitOr,
    BitXor,
    In,
    Instanceof,
}

/// `&&`, `||`, `??`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicalOp {
    And,
    Or,
    Nullish,
}

/// `=`, `+=`, … `&&=`, `||=`, `??=`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignOp {
    Assign,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    LShift,
    RShift,
    URShift,
    BitAnd,
    BitOr,
    BitXor,
    And,
    Or,
    Nullish,
}

// ---------------------------------------------------------------------------
// Modules
// ---------------------------------------------------------------------------

/// `import ... from "source"`.
#[derive(Clone, Debug)]
pub struct ImportDecl {
    pub default: Option<(String, Span)>,
    pub namespace: Option<(String, Span)>,
    pub named: Vec<ImportSpec>,
    /// cooked module specifier.
    pub source: String,
    pub span: Span,
}

/// `import { imported as local }`.
#[derive(Clone, Debug)]
pub struct ImportSpec {
    pub imported: ModuleName,
    pub local: String,
}

/// `export ...`.
#[derive(Clone, Debug)]
pub enum ExportDecl {
    /// `export * from "m"`, `export * as ns from "m"`.
    All {
        source: String,
        exported: Option<String>,
        span: Span,
    },
    /// `export {a, b as c}` (+ optional `from "m"`).
    Named {
        specifiers: Vec<ExportSpec>,
        source: Option<String>,
        span: Span,
    },
    /// `export default <expr>;` — function/class expressions keep their
    /// (optional) name; hoisting is the compiler's concern.
    Default { expr: Expr, span: Span },
    /// `export var/let/const/function/class ...`.
    Decl { inner: Box<Stmt>, span: Span },
}

/// `export { local as exported }`.
#[derive(Clone, Debug)]
pub struct ExportSpec {
    pub local: ModuleName,
    pub exported: ModuleName,
}

/// An imported/exported name: identifier or arbitrary string
/// (`import {"a-b" as x}`).
#[derive(Clone, Debug)]
pub enum ModuleName {
    Ident(String, Span),
    Str(Wtf8Buf, Span),
}

// ---------------------------------------------------------------------------
// Span helpers
// ---------------------------------------------------------------------------

impl Expr {
    /// Span of this expression.
    pub fn span(&self) -> Span {
        match self {
            Expr::Ident(_, s)
            | Expr::PrivateIdent(_, s)
            | Expr::Lit(_, s)
            | Expr::Array(_, s)
            | Expr::Object(_, s)
            | Expr::Sequence(_, s)
            | Expr::Template { span: s, .. }
            | Expr::This(s)
            | Expr::Super(s)
            | Expr::NewTarget(s)
            | Expr::ImportMeta(s)
            | Expr::Await(_, s)
            | Expr::Spread(_, s) => *s,
            Expr::Function { span: s, .. }
            | Expr::Class { span: s, .. }
            | Expr::Arrow { span: s, .. }
            | Expr::Call { span: s, .. }
            | Expr::New { span: s, .. }
            | Expr::Member { span: s, .. }
            | Expr::Unary { span: s, .. }
            | Expr::Update { span: s, .. }
            | Expr::Binary { span: s, .. }
            | Expr::Logical { span: s, .. }
            | Expr::Assign { span: s, .. }
            | Expr::Conditional { span: s, .. }
            | Expr::TaggedTemplate { span: s, .. }
            | Expr::Regex { span: s, .. } => *s,
            Expr::Yield { span: s, .. } => *s,
        }
    }
}

// ---------------------------------------------------------------------------
// Iterative drop
// ---------------------------------------------------------------------------

/// Dummy expression used as a placeholder while dismantling a spine node:
/// `this` carries no allocation and drops trivially.
fn dummy_expr() -> Expr {
    Expr::This(Span { start: 0, end: 0 })
}

/// If `e` is a left-deep spine node (see [`Drop for Expr`]), take its spine
/// child out — leaving a dummy behind so the rest of the node drops without
/// recursing down the spine — and return it. Returns `None` for every other
/// node, including a node whose spine slot already holds the dummy (which
/// guarantees the drop loop terminates instead of looping on the
/// placeholder).
fn take_spine_child(e: &mut Expr) -> Option<Expr> {
    let child: &mut Expr = match e {
        Expr::Call { callee, .. } => callee.as_mut(),
        Expr::Member { obj, .. } => obj.as_mut(),
        Expr::New { callee, .. } => callee.as_mut(),
        Expr::Unary { arg, .. } => arg.as_mut(),
        Expr::Update { arg, .. } => arg.as_mut(),
        Expr::Binary { left, .. } => left.as_mut(),
        Expr::Logical { left, .. } => left.as_mut(),
        _ => return None,
    };
    // The dummy doubles as the termination marker: a spine slot holding
    // `This` is never descended into, so the re-entered `drop` on a
    // dismantled node breaks out of its loop immediately.
    if matches!(child, Expr::This(_)) {
        return None;
    }
    Some(std::mem::replace(child, dummy_expr()))
}

impl Drop for Expr {
    /// Free left-deep expression spines at O(1) native stack.
    ///
    /// The parser builds `Call`/`Member`/`New`/`Unary`/`Update`/`Binary`/
    /// `Logical` chains in a loop, so source like `f.bind().bind()…` × 3000
    /// is a legal AST whose derived recursive `Drop` would overflow the
    /// host stack (an uncatchable abort) when the value is freed. This
    /// instead peels the spine iteratively: each iteration extracts the
    /// spine child (leaving a dummy in its slot), drops the node's other
    /// children — shallow, since the spine itself is handled by this loop —
    /// and continues with the child.
    fn drop(&mut self) {
        let mut next = take_spine_child(self);
        while let Some(mut child) = next {
            next = take_spine_child(&mut child);
            // `child` drops at the end of this iteration with its spine
            // slot already replaced by the dummy: the re-entered `drop`
            // takes no child and returns at once, so native-stack depth
            // stays constant no matter how long the spine is. The node's
            // other children (call arguments, …) drop with it, recursively
            // but shallow — only the spine itself is 3000-deep.
        }
    }
}
