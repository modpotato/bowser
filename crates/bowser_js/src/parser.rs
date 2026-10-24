//! Recursive-descent + Pratt parser for ES2024 JavaScript.
//!
//! The parser drives [`Lexer`](crate::lexer::Lexer) incrementally so it can
//! intervene for regular-expression literals: when a `/` token appears where
//! the grammar expects the start of an expression, the parser re-lexes it via
//! [`lex_regex`](crate::lexer::lex_regex) and resumes the lexer past the
//! regex. ASI uses the line-terminator spans collected by the lexer.
//!
//! The parser only builds the AST ([`Program`]); it never evaluates.
//! Grammar early errors (strict-mode violations, invalid assignment targets,
//! duplicate `__proto__`, `??` mixing, bad `**` operands, Annex B limits)
//! are reported as [`SyntaxError::EarlyError`].

use crate::ast::*;
use crate::lexer::{LexError, Lexer, Span, Token, TokenKind, lex_regex};
use crate::wtf8::{Wtf8, Wtf8Buf};
use std::collections::{HashMap, HashSet};

/// `(name, params, body, is_generator, strict)` returned by `parse_function_common`:
/// `strict` is the function's final strictness (inherited strictness plus a
/// `"use strict"` directive in its body, raw source form).
type FunctionCommon = (Option<String>, Vec<Param>, Vec<Stmt>, bool, bool);
/// `(name, superclass, members)` returned by `parse_class_common`.
type ClassCommon = (Option<String>, Option<Box<Expr>>, Vec<ClassMember>);

/// Kind of a private name declaration, for duplicate checking.
/// A getter and setter may share a name; all other duplicates are errors.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PrivateKind {
    Getter,
    Setter,
    Other,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A syntax error with a source span.
#[derive(Clone, Debug)]
pub enum SyntaxError {
    /// The parser needed one thing and found another.
    UnexpectedToken {
        expected: String,
        found: String,
        span: Span,
    },
    /// Input ended where more was required.
    UnexpectedEof { expected: String, span: Span },
    /// A grammar early error: strict-mode violation, invalid assignment
    /// target, duplicate `__proto__`, bad `**` operand, etc.
    EarlyError { message: String, span: Span },
    /// The lexer rejected the input.
    Lex { message: String, span: Span },
}

impl std::fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SyntaxError::UnexpectedToken {
                expected,
                found,
                span,
            } => write!(
                f,
                "SyntaxError: expected {expected}, found {found} at offset {}",
                span.start
            ),
            SyntaxError::UnexpectedEof { expected, span } => write!(
                f,
                "SyntaxError: unexpected end of input, expected {expected} at offset {}",
                span.start
            ),
            SyntaxError::EarlyError { message, span } => {
                write!(f, "SyntaxError: {message} at offset {}", span.start)
            }
            SyntaxError::Lex { message, span } => {
                write!(f, "SyntaxError: {message} at offset {}", span.start)
            }
        }
    }
}

impl std::error::Error for SyntaxError {}

impl From<LexError> for SyntaxError {
    fn from(e: LexError) -> Self {
        let span = match e {
            LexError::UnexpectedChar(_, off)
            | LexError::UnterminatedString(off)
            | LexError::UnterminatedTemplate(off)
            | LexError::UnterminatedComment(off)
            | LexError::UnterminatedRegex(off)
            | LexError::InvalidNumber(off) => Span {
                start: off,
                end: off,
            },
        };
        SyntaxError::Lex {
            message: e.to_string(),
            span,
        }
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Parse `src` as a Module (strict mode, import/export allowed).
pub fn parse(src: &str) -> Result<Program, SyntaxError> {
    let mut p = Parser::new(src, true);
    p.parse_program()
}

/// Parse `src` as a Script (sloppy mode unless a `"use strict"` directive).
/// `Vm::eval` uses this: page scripts are sloppy by default.
pub fn parse_script(src: &str) -> Result<Program, SyntaxError> {
    let mut p = Parser::new(src, false);
    p.parse_program()
}

// ---------------------------------------------------------------------------
// Parser state
// ---------------------------------------------------------------------------

/// Maximum expression/statement nesting (stack-overflow guard).
///
/// Calibrated so the guard fires before the host stack is exhausted on a
/// standard 8 MiB stack, including unoptimized (debug) builds where each
/// nesting level holds several kilobytes of parser frames: e.g. one level
/// of parentheses costs ~4 depth units and ~70 KiB, so the guard trips at
/// ~64 parens while a raw stack overflow would need ~115.
///
/// Object/array literals (and destructuring patterns) nest a full
/// expression/pattern parse per level through ~15 native frames, costing
/// ~300 KiB per level in debug builds — far more than the 1 depth unit a
/// plain [`Parser::guarded`] call charges. They use
/// [`Parser::guarded_cost`] with [`HEAVY_LITERAL_COST`] so the guard trips
/// (~10 levels) well before native exhaustion (~25 levels).
const MAX_DEPTH: u32 = 256;

/// Depth units charged per object/array literal (or pattern) nesting
/// level. See [`MAX_DEPTH`].
const HEAVY_LITERAL_COST: u32 = 24;

struct Label {
    name: String,
    is_iteration: bool,
}

struct Parser<'a> {
    src: &'a str,
    lexer: Lexer<'a>,
    /// Lookahead buffer; `toks[pos]` is the current token.
    toks: Vec<Token>,
    pos: usize,
    strict: bool,
    is_module: bool,
    top_level: bool,
    // function context
    in_async: bool,
    in_generator: bool,
    func_depth: u32,
    // class context
    class_depth: u32,
    in_static_block: bool,
    in_field_init: bool,
    in_method: bool,
    in_derived_ctor: bool,
    /// Whether the innermost class being parsed has an `extends` clause.
    /// Used to validate `super()` calls in derived constructors.
    class_is_derived: bool,
    /// Private names declared by each enclosing class (innermost last).
    private_declared: Vec<HashMap<String, PrivateKind>>,
    /// Private-name uses recorded for each enclosing class (innermost last).
    private_used: Vec<Vec<(String, Span)>>,
    /// `new.target` is valid iff this is nonzero. Incremented on entering
    /// non-arrow function/method/static-block/field-init bodies; arrows
    /// inherit the enclosing value.
    nt_depth: u32,
    // expression context
    allow_in: bool,
    ban_optional: bool,
    /// Inside a `for (...)` head init, before `in`/`of` is confirmed.
    in_for_head: bool,
    /// Inside a parenthesized cover that may become arrow parameters.
    in_paren_cover: bool,
    /// `true` while parsing a potential destructuring target: `{a = 1}`
    /// shorthand initializers are tentatively allowed.
    cover_ok: bool,
    /// `??` may not appear here (inside `||`/`&&`); see `parse_coalesce`.
    ban_mix_nullish: bool,
    /// `||`/`&&` may not join here (after an unparenthesized `??`).
    ban_mix_logical: bool,
    /// `> 0` while parsing formal parameters: `yield`/`await` may not be
    /// used as operators there.
    param_depth: u32,
    /// A shorthand-default cover initializer (`{a = 1}`, `[a = 1]`) was
    /// parsed and not yet validated as a destructuring target.
    pending_cover: bool,
    /// `> 0` while parsing a `${...}` template substitution: a
    /// `TemplateTail`/`TemplateHead` here ends the substitution, it is not
    /// a tagged template.
    template_subst_depth: u32,
    /// Undo log for [`Self::take`]: `(buffer index, original token)` for
    /// tokens taken by value while a speculative checkpoint is armed.
    /// [`Self::rewind_to`] restores them so backtracking re-reads intact
    /// tokens.
    take_undo: Vec<(usize, Token)>,
    /// Active speculative checkpoints, as `(token pos, undo-log length)`.
    checkpoints: Vec<(usize, usize)>,
    // statement context
    labels: Vec<Label>,
    iter_depth: u32,
    breakable_depth: u32,
    depth: u32,
}

/// Snapshot of the parser context, for entering nested function bodies.
struct CtxSave {
    strict: bool,
    in_async: bool,
    in_generator: bool,
    func_depth: u32,
    class_depth: u32,
    in_static_block: bool,
    in_field_init: bool,
    in_method: bool,
    in_derived_ctor: bool,
    class_is_derived: bool,
    nt_depth: u32,
    allow_in: bool,
    ban_optional: bool,
    in_for_head: bool,
    in_paren_cover: bool,
    cover_ok: bool,
    ban_mix_nullish: bool,
    ban_mix_logical: bool,
    param_depth: u32,
    pending_cover: bool,
    template_subst_depth: u32,
    labels_len: usize,
    iter_depth: u32,
    breakable_depth: u32,
    top_level: bool,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str, is_module: bool) -> Self {
        Parser {
            src,
            lexer: Lexer::new(src),
            toks: Vec::new(),
            pos: 0,
            strict: is_module,
            is_module,
            top_level: true,
            in_async: false,
            in_generator: false,
            func_depth: 0,
            class_depth: 0,
            in_static_block: false,
            in_field_init: false,
            in_method: false,
            in_derived_ctor: false,
            class_is_derived: false,
            private_declared: Vec::new(),
            private_used: Vec::new(),
            nt_depth: 0,
            allow_in: true,
            ban_optional: false,
            in_for_head: false,
            in_paren_cover: false,
            cover_ok: false,
            ban_mix_nullish: false,
            ban_mix_logical: false,
            param_depth: 0,
            pending_cover: false,
            template_subst_depth: 0,
            labels: Vec::new(),
            iter_depth: 0,
            breakable_depth: 0,
            depth: 0,
            take_undo: Vec::new(),
            checkpoints: Vec::new(),
        }
    }

    // -- token buffer --------------------------------------------------------

    /// Lex ahead until `toks[pos+n]` exists (Eof repeats, so this terminates).
    fn ensure(&mut self, n: usize) -> Result<(), SyntaxError> {
        while self.toks.len() <= self.pos + n {
            let t = self.lexer.next_token().map_err(SyntaxError::from)?;
            let is_eof = matches!(t.kind, TokenKind::Eof);
            self.toks.push(t);
            if is_eof {
                break;
            }
        }
        Ok(())
    }

    fn peek_at(&mut self, n: usize) -> Result<&TokenKind, SyntaxError> {
        self.ensure(n)?;
        let idx = (self.pos + n).min(self.toks.len() - 1);
        Ok(&self.toks[idx].kind)
    }

    fn peek_span_at(&mut self, n: usize) -> Result<Span, SyntaxError> {
        self.ensure(n)?;
        let idx = (self.pos + n).min(self.toks.len() - 1);
        Ok(self.toks[idx].span)
    }

    fn peek_span(&mut self) -> Result<Span, SyntaxError> {
        self.peek_span_at(0)
    }

    /// Span of the most recently consumed token.
    fn prev_span(&mut self) -> Result<Span, SyntaxError> {
        if self.pos == 0 {
            self.peek_span()
        } else {
            Ok(self.toks[self.pos - 1].span)
        }
    }

    /// Consume the current token.
    fn bump(&mut self) -> Result<Span, SyntaxError> {
        let span = self.peek_span()?;
        if !matches!(self.toks[self.pos].kind, TokenKind::Eof) {
            self.pos += 1;
        }
        Ok(span)
    }

    /// Consume the current token by value, leaving an [`TokenKind::Eof`]
    /// placeholder behind. The span is kept, so [`Self::prev_span`] still
    /// works; positions below `pos` are never re-read as tokens. This moves
    /// `String` payloads out instead of cloning them.
    ///
    /// When a speculative checkpoint is armed (see [`Self::checkpoint`]),
    /// the original token is also logged so [`Self::rewind_to`] can restore
    /// it if the speculation fails.
    fn take(&mut self) -> Result<Token, SyntaxError> {
        self.ensure(0)?;
        let idx = self.pos;
        let mut tok = Token {
            kind: TokenKind::Eof,
            span: self.toks[idx].span,
        };
        std::mem::swap(&mut tok, &mut self.toks[idx]);
        // `tok` now holds the original token; the buffer has a placeholder.
        if !matches!(tok.kind, TokenKind::Eof) {
            if !self.checkpoints.is_empty() {
                self.take_undo.push((idx, tok.clone()));
            }
            self.pos += 1;
        }
        Ok(tok)
    }

    /// Arm a speculative checkpoint. Returns an id to pass to
    /// [`Self::rewind_to`] (on failure) or [`Self::release`] (on success).
    fn checkpoint(&mut self) -> usize {
        let id = self.checkpoints.len();
        self.checkpoints.push((self.pos, self.take_undo.len()));
        id
    }

    /// Restore the token position saved by [`Self::checkpoint`], putting
    /// back every token taken since the checkpoint so the re-parse sees
    /// intact tokens.
    fn rewind_to(&mut self, id: usize) {
        let (pos, undo_len) = self.checkpoints.pop().expect("unbalanced checkpoint");
        debug_assert_eq!(id, self.checkpoints.len());
        for (idx, tok) in self.take_undo.drain(undo_len..) {
            self.toks[idx] = tok;
        }
        self.pos = pos;
    }

    /// Drop a [`Self::checkpoint`] after successful speculation: the taken
    /// tokens stay consumed and their undo-log entries are discarded, but
    /// only when no outer checkpoint is still armed. An outer checkpoint
    /// may still rewind over this region (e.g. the async-arrow speculation
    /// wraps the paren-arrow one), so its entries must be retained for a
    /// potential outer [`Self::rewind_to`].
    fn release(&mut self, id: usize) {
        let (_, undo_len) = self.checkpoints.pop().expect("unbalanced checkpoint");
        debug_assert_eq!(id, self.checkpoints.len());
        if self.checkpoints.is_empty() {
            self.take_undo.truncate(undo_len);
        }
    }

    /// Take a string-literal token by value. The caller must have checked
    /// that the current token is [`TokenKind::Str`].
    fn take_str(&mut self) -> Result<(Wtf8Buf, Span), SyntaxError> {
        let tok = self.take()?;
        let span = tok.span;
        match tok.kind {
            TokenKind::Str(s) => Ok((s, span)),
            _ => Err(self.unexpected("a string literal")),
        }
    }

    /// Take a private-identifier token by value. The caller must have
    /// checked that the current token is [`TokenKind::PrivateIdent`].
    fn take_private_ident(&mut self) -> Result<(String, Span), SyntaxError> {
        let tok = self.take()?;
        let span = tok.span;
        match tok.kind {
            TokenKind::PrivateIdent(n) => Ok((n, span)),
            _ => Err(self.unexpected("a private identifier")),
        }
    }

    /// If the current token is `/` where an expression may start, re-lex it
    /// as a regular expression and resume the lexer past it.
    ///
    /// Must be called before any lookahead past the `/`: the buffer holds no
    /// tokens after it at this point (all multi-token peeks in this parser
    /// happen with a non-`/` current token).
    fn maybe_regex(&mut self) -> Result<(), SyntaxError> {
        if !matches!(self.peek_at(0)?, TokenKind::Slash) {
            return Ok(());
        }
        debug_assert_eq!(self.toks.len(), self.pos + 1);
        let start = self.peek_span()?.start;
        let (kind, end) = lex_regex(self.src, start).map_err(SyntaxError::from)?;
        self.lexer.set_pos(end);
        self.toks.truncate(self.pos);
        self.toks.push(Token {
            kind,
            span: Span { start, end },
        });
        Ok(())
    }

    // -- error helpers ------------------------------------------------------

    fn unexpected(&mut self, expected: &str) -> SyntaxError {
        let idx = self.pos.min(self.toks.len().saturating_sub(1));
        // Ensure at least the current token is lexed for a decent span.
        let _ = self.ensure(0);
        let idx = idx.min(self.toks.len().saturating_sub(1));
        let tok = &self.toks[idx];
        if matches!(tok.kind, TokenKind::Eof) {
            SyntaxError::UnexpectedEof {
                expected: expected.into(),
                span: tok.span,
            }
        } else {
            SyntaxError::UnexpectedToken {
                expected: expected.into(),
                found: kind_name(&tok.kind),
                span: tok.span,
            }
        }
    }

    fn early(message: &str, span: Span) -> SyntaxError {
        SyntaxError::EarlyError {
            message: message.into(),
            span,
        }
    }

    fn expect(&mut self, kind: &TokenKind, what: &str) -> Result<Span, SyntaxError> {
        if std::mem::discriminant(self.peek_at(0)?) == std::mem::discriminant(kind) {
            self.bump()
        } else {
            Err(self.unexpected(what))
        }
    }

    /// Run `f` with the recursion guard held.
    fn guarded<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, SyntaxError>,
    ) -> Result<T, SyntaxError> {
        self.guarded_cost(1, f)
    }

    /// Depth guard with an explicit cost. Most nesting charges 1 unit via
    /// [`Parser::guarded`]; shapes whose native frames are far heavier
    /// than one unit assumes (object/array literals and patterns) charge
    /// [`HEAVY_LITERAL_COST`] so the guard trips before native exhaustion.
    fn guarded_cost<T>(
        &mut self,
        cost: u32,
        f: impl FnOnce(&mut Self) -> Result<T, SyntaxError>,
    ) -> Result<T, SyntaxError> {
        if self.depth + cost > MAX_DEPTH {
            let span = self.peek_span()?;
            return Err(Self::early("maximum parse depth exceeded", span));
        }
        self.depth += cost;
        let r = f(self);
        self.depth -= cost;
        r
    }

    /// Snapshot the function/class/expression context so it can be replaced
    /// while parsing a nested function body and restored afterwards.
    fn save_ctx(&self) -> CtxSave {
        CtxSave {
            strict: self.strict,
            in_async: self.in_async,
            in_generator: self.in_generator,
            func_depth: self.func_depth,
            class_depth: self.class_depth,
            in_static_block: self.in_static_block,
            in_field_init: self.in_field_init,
            in_method: self.in_method,
            in_derived_ctor: self.in_derived_ctor,
            class_is_derived: self.class_is_derived,
            nt_depth: self.nt_depth,
            allow_in: self.allow_in,
            ban_optional: self.ban_optional,
            in_for_head: self.in_for_head,
            in_paren_cover: self.in_paren_cover,
            cover_ok: self.cover_ok,
            ban_mix_nullish: self.ban_mix_nullish,
            ban_mix_logical: self.ban_mix_logical,
            param_depth: self.param_depth,
            pending_cover: self.pending_cover,
            template_subst_depth: self.template_subst_depth,
            labels_len: self.labels.len(),
            iter_depth: self.iter_depth,
            breakable_depth: self.breakable_depth,
            top_level: self.top_level,
        }
    }

    fn restore_ctx(&mut self, s: CtxSave) {
        self.strict = s.strict;
        self.in_async = s.in_async;
        self.in_generator = s.in_generator;
        self.func_depth = s.func_depth;
        self.class_depth = s.class_depth;
        self.in_static_block = s.in_static_block;
        self.in_field_init = s.in_field_init;
        self.in_method = s.in_method;
        self.in_derived_ctor = s.in_derived_ctor;
        self.class_is_derived = s.class_is_derived;
        self.nt_depth = s.nt_depth;
        self.allow_in = s.allow_in;
        self.ban_optional = s.ban_optional;
        self.in_for_head = s.in_for_head;
        self.in_paren_cover = s.in_paren_cover;
        self.cover_ok = s.cover_ok;
        self.ban_mix_nullish = s.ban_mix_nullish;
        self.ban_mix_logical = s.ban_mix_logical;
        self.param_depth = s.param_depth;
        self.pending_cover = s.pending_cover;
        self.template_subst_depth = s.template_subst_depth;
        self.labels.truncate(s.labels_len);
        self.iter_depth = s.iter_depth;
        self.breakable_depth = s.breakable_depth;
        self.top_level = s.top_level;
    }

    /// `true` when `await` may be parsed as an operator here.
    fn await_allowed(&self) -> bool {
        self.param_depth == 0
            && !self.in_field_init
            && !self.in_static_block
            && (self.in_async || (self.is_module && self.func_depth == 0))
    }

    /// `true` when `yield` may be parsed as an operator here.
    fn yield_is_operator(&self) -> bool {
        self.in_generator && self.param_depth == 0 && !self.in_field_init && !self.in_static_block
    }

    /// `true` when `super` property access is allowed here.
    fn super_allowed(&self) -> bool {
        self.in_method || self.in_field_init || self.in_static_block
    }

    // -- ASI ------------------------------------------------------------------

    /// `true` if a line terminator sits between token `pos+n-1` and `pos+n`.
    fn nl_before(&mut self, n: usize) -> Result<bool, SyntaxError> {
        self.ensure(n)?;
        let idx = self.pos + n;
        if idx == 0 || idx >= self.toks.len() {
            return Ok(false);
        }
        let prev_end = self.toks[idx - 1].span.end;
        let cur_start = self.toks[idx].span.start;
        // Binary search over the ascending trivia spans (see
        // `Lexer::has_line_term_between`); the old linear scan was O(n^2)
        // on newline-heavy inputs.
        Ok(self.lexer.has_line_term_between(prev_end, cur_start))
    }

    /// Consume a semicolon: explicit `;`, or ASI before `}` / EOF / a line
    /// terminator.
    fn expect_semi(&mut self) -> Result<(), SyntaxError> {
        if matches!(self.peek_at(0)?, TokenKind::Semicolon) {
            self.bump()?;
            return Ok(());
        }
        if matches!(self.peek_at(0)?, TokenKind::RBrace | TokenKind::Eof) {
            return Ok(());
        }
        if self.nl_before(0)? {
            return Ok(());
        }
        Err(self.unexpected("';'"))
    }

    // -- programs ---------------------------------------------------------------

    fn parse_program(&mut self) -> Result<Program, SyntaxError> {
        let mut body = Vec::new();
        let mut in_prologue = !self.is_module;
        while !matches!(self.peek_at(0)?, TokenKind::Eof) {
            let stmt = self.parse_stmt()?;
            // Script directive prologue: `"use strict";` flips strict mode.
            // The raw form must be unescaped (see peek_use_strict).
            if in_prologue {
                if let Stmt::Expr(Expr::Lit(Literal::Str(s), span)) = &stmt
                    && s == "use strict"
                {
                    let raw = self.raw_of_span(*span);
                    if raw == "\"use strict\"" || raw == "'use strict'" {
                        self.strict = true;
                        body.push(stmt);
                        continue;
                    }
                }
                in_prologue = false;
            }
            body.push(stmt);
        }
        // Desugar `using`/`await using` declarations to `try`/`finally` +
        // hidden disposal intrinsics before the compiler runs.
        desugar_using(&mut body);
        Ok(Program {
            body,
            strict: self.strict,
        })
    }

    /// `true` if the input at the current position is a `"use strict";`
    /// directive (used for function bodies, before parsing them). The raw
    /// source must be exactly `"use strict"` or `'use strict'`: an escaped
    /// form (`"\u0075se strict"`) is not a directive.
    fn peek_use_strict(&mut self) -> Result<bool, SyntaxError> {
        match self.peek_at(0)? {
            TokenKind::Str(s) if s == "use strict" => {
                let span = self.peek_span()?;
                let raw = self.raw_of_span(span);
                let plain = raw == "\"use strict\"" || raw == "'use strict'";
                Ok(plain
                    && (matches!(
                        self.peek_at(1)?,
                        TokenKind::Semicolon | TokenKind::RBrace | TokenKind::Eof
                    ) || self.nl_before(1)?))
            }
            _ => Ok(false),
        }
    }

    /// Source text covered by `span`.
    fn raw_of_span(&self, span: Span) -> String {
        let start = span.start as usize;
        let end = span.end as usize;
        if start <= end
            && end <= self.src.len()
            && self.src.is_char_boundary(start)
            && self.src.is_char_boundary(end)
        {
            self.src[start..end].to_string()
        } else {
            String::new()
        }
    }
}

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Text of a contextual-keyword token (`let`, `async`, `await`, ...), which
/// lex as keyword kinds unconditionally; the parser decides when they act as
/// identifiers.
fn contextual_name(kind: &TokenKind) -> Option<&'static str> {
    match kind {
        TokenKind::Await => Some("await"),
        TokenKind::Yield => Some("yield"),
        TokenKind::Let => Some("let"),
        TokenKind::Static => Some("static"),
        TokenKind::Of => Some("of"),
        TokenKind::Get => Some("get"),
        TokenKind::Set => Some("set"),
        TokenKind::Async => Some("async"),
        TokenKind::From => Some("from"),
        TokenKind::As => Some("as"),
        TokenKind::Target => Some("target"),
        _ => None,
    }
}

/// `true` if the token can serve as an identifier name.
fn is_identifierish(kind: &TokenKind) -> bool {
    matches!(kind, TokenKind::Ident(_)) || contextual_name(kind).is_some()
}

impl<'a> Parser<'a> {
    /// Parse an identifier name: `Ident` or any contextual keyword.
    /// Does not validate reserved-ness; callers apply the context checks.
    /// The token is taken by value so `String` payloads move, not clone.
    fn parse_ident_name(&mut self) -> Result<(String, Span), SyntaxError> {
        let ok = match self.peek_at(0)? {
            TokenKind::Ident(_) => true,
            k => contextual_name(k).is_some(),
        };
        if !ok {
            return Err(self.unexpected("an identifier"));
        }
        let tok = self.take()?;
        let span = tok.span;
        let name = match tok.kind {
            TokenKind::Ident(s) => s,
            k => contextual_name(&k).unwrap_or("").to_string(),
        };
        Ok((name, span))
    }

    /// Parse a property name after `.` / `?.`: an identifier, a private
    /// name, or any reserved word (property names are IdentifierNames).
    /// Private names keep their leading `#`.
    fn parse_prop_name(&mut self) -> Result<(String, Span), SyntaxError> {
        let ok = match self.peek_at(0)? {
            TokenKind::Ident(_) | TokenKind::PrivateIdent(_) => true,
            k => contextual_name(k).or_else(|| keyword_text(k)).is_some(),
        };
        if !ok {
            return Err(self.unexpected("a property name"));
        }
        let tok = self.take()?;
        let span = tok.span;
        let name = match tok.kind {
            TokenKind::Ident(s) => s,
            TokenKind::PrivateIdent(s) => format!("#{s}"),
            k => contextual_name(&k)
                .or_else(|| keyword_text(&k))
                .unwrap_or("")
                .to_string(),
        };
        Ok((name, span))
    }

    /// If the current token is identifier-ish and followed by `:`, parse it
    /// as a label name. `await`/`yield` acting as operators are excluded by
    /// the caller.
    fn peek_label_name(&mut self) -> Result<Option<(String, Span)>, SyntaxError> {
        if !is_identifierish(self.peek_at(0)?) {
            return Ok(None);
        }
        if !matches!(self.peek_at(1)?, TokenKind::Colon) {
            return Ok(None);
        }
        // Peek only; parse_labeled will bump.
        let span = self.peek_span()?;
        let name = match self.peek_at(0)? {
            TokenKind::Ident(s) => s.clone(),
            k => contextual_name(k).unwrap_or("").to_string(),
        };
        Ok(Some((name, span)))
    }
}

/// Source text of a reserved-word token (for property names like `a.if`).
fn keyword_text(kind: &TokenKind) -> Option<&'static str> {
    match kind {
        TokenKind::Break => Some("break"),
        TokenKind::Case => Some("case"),
        TokenKind::Catch => Some("catch"),
        TokenKind::Class => Some("class"),
        TokenKind::Const => Some("const"),
        TokenKind::Continue => Some("continue"),
        TokenKind::Debugger => Some("debugger"),
        TokenKind::Default => Some("default"),
        TokenKind::Delete => Some("delete"),
        TokenKind::Do => Some("do"),
        TokenKind::Else => Some("else"),
        TokenKind::Enum => Some("enum"),
        TokenKind::Export => Some("export"),
        TokenKind::Extends => Some("extends"),
        TokenKind::False => Some("false"),
        TokenKind::Finally => Some("finally"),
        TokenKind::For => Some("for"),
        TokenKind::Function => Some("function"),
        TokenKind::If => Some("if"),
        TokenKind::Import => Some("import"),
        TokenKind::In => Some("in"),
        TokenKind::Instanceof => Some("instanceof"),
        TokenKind::New => Some("new"),
        TokenKind::Null => Some("null"),
        TokenKind::Return => Some("return"),
        TokenKind::Super => Some("super"),
        TokenKind::Switch => Some("switch"),
        TokenKind::This => Some("this"),
        TokenKind::Throw => Some("throw"),
        TokenKind::True => Some("true"),
        TokenKind::Try => Some("try"),
        TokenKind::Typeof => Some("typeof"),
        TokenKind::Var => Some("var"),
        TokenKind::Void => Some("void"),
        TokenKind::While => Some("while"),
        TokenKind::With => Some("with"),
        // Contextual keywords.
        TokenKind::Await => Some("await"),
        TokenKind::Yield => Some("yield"),
        TokenKind::Let => Some("let"),
        TokenKind::Static => Some("static"),
        TokenKind::Of => Some("of"),
        TokenKind::Get => Some("get"),
        TokenKind::Set => Some("set"),
        TokenKind::Async => Some("async"),
        TokenKind::From => Some("from"),
        TokenKind::As => Some("as"),
        TokenKind::Target => Some("target"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

impl<'a> Parser<'a> {
    fn parse_stmt(&mut self) -> Result<Stmt, SyntaxError> {
        self.guarded(|s| s.parse_stmt_inner(false))
    }

    /// `single`: single-statement context (`if (x) <here>`); declarations
    /// are restricted there (Annex B in sloppy mode).
    fn parse_stmt_inner(&mut self, single: bool) -> Result<Stmt, SyntaxError> {
        // `await` / `yield` as operators start expression statements.
        if matches!(self.peek_at(0)?, TokenKind::Await) && self.await_allowed() {
            // `await using x = ...` is an `await using` declaration, not
            // `await` applied to an expression (`using` is at position 1).
            if self.using_decl_at(1)? {
                // Lexical declarations cannot be a single-statement body.
                if single {
                    return Err(Self::early(
                        "await using declaration in single-statement position is not allowed",
                        self.peek_span()?,
                    ));
                }
                return self.parse_using_decl(true);
            }
            return self.parse_expr_stmt();
        }
        // `await using` outside an async context: `await` is a plain
        // identifier there, so give the targeted error instead of a
        // confusing expression-parse failure.
        if matches!(self.peek_at(0)?, TokenKind::Await) && self.using_decl_at(1)? {
            return Err(Self::early(
                "await using is only allowed in async functions and modules",
                self.peek_span()?,
            ));
        }
        if matches!(self.peek_at(0)?, TokenKind::Yield) && self.yield_is_operator() {
            return self.parse_expr_stmt();
        }
        match self.peek_at(0)? {
            TokenKind::Semicolon => {
                self.bump()?;
                Ok(Stmt::Empty)
            }
            TokenKind::LBrace => Ok(Stmt::Block(self.parse_block()?)),
            TokenKind::Var => self.parse_var_stmt(VarKind::Var),
            TokenKind::Const => {
                // Lexical declarations cannot be a single-statement body.
                if single {
                    return Err(Self::early(
                        "lexical declaration in single-statement position is not allowed",
                        self.peek_span()?,
                    ));
                }
                self.parse_var_stmt(VarKind::Const)
            }
            TokenKind::Let => self.parse_let_or_expr(single),
            TokenKind::Ident(s) if s.as_str() == "using" => {
                // `using: foo` is a labeled statement, not a declaration.
                if let Some((name, span)) = self.peek_label_name()? {
                    return self.parse_labeled(name, span, single);
                }
                // Lexical declarations cannot be a single-statement body.
                if single && self.using_decl_at(0)? {
                    return Err(Self::early(
                        "using declaration in single-statement position is not allowed",
                        self.peek_span()?,
                    ));
                }
                self.parse_using_or_expr()
            }
            TokenKind::Function => self.parse_function_decl(false, single),
            TokenKind::Async => self.parse_async_stmt(single),
            TokenKind::Class => self.parse_class_decl(single),
            TokenKind::If => self.parse_if(single),
            TokenKind::While => self.parse_while(),
            TokenKind::Do => self.parse_do(),
            TokenKind::For => self.parse_for(),
            TokenKind::Switch => self.parse_switch(),
            TokenKind::Try => self.parse_try(),
            TokenKind::Return => self.parse_return(),
            TokenKind::Throw => self.parse_throw(),
            TokenKind::Break => self.parse_break(),
            TokenKind::Continue => self.parse_continue(),
            TokenKind::Debugger => {
                self.bump()?;
                self.expect_semi()?;
                Ok(Stmt::Debugger)
            }
            TokenKind::With => self.parse_with(single),
            TokenKind::Import => {
                // `import(` (dynamic import) and `import.meta` are expressions,
                // not declarations. They can appear in scripts too.
                if matches!(self.peek_at(1)?, TokenKind::LParen | TokenKind::Dot) {
                    self.parse_expr_stmt()
                } else {
                    if !self.is_module {
                        let span = self.peek_span()?;
                        return Err(Self::early(
                            "import declarations are only allowed in modules",
                            span,
                        ));
                    }
                    self.parse_import()
                }
            }
            TokenKind::Export => {
                if !self.is_module {
                    return Err(Self::early(
                        "export declarations are only allowed in modules",
                        self.peek_span()?,
                    ));
                }
                self.parse_export()
            }
            _ => {
                // Labeled statement or expression statement.
                if let Some((name, span)) = self.peek_label_name()? {
                    return self.parse_labeled(name, span, single);
                }
                self.parse_expr_stmt()
            }
        }
    }

    fn parse_expr_stmt(&mut self) -> Result<Stmt, SyntaxError> {
        // `{` and `function` cannot start an expression statement; the
        // dispatcher already routes them, but `async function` looks like an
        // expression start — it was routed to parse_async_stmt above.
        let e = self.parse_expr()?;
        self.expect_semi()?;
        Ok(Stmt::Expr(e))
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, SyntaxError> {
        self.expect(&TokenKind::LBrace, "'{'")?;
        let save_top = self.top_level;
        self.top_level = false;
        let mut stmts = Vec::new();
        while !matches!(self.peek_at(0)?, TokenKind::RBrace | TokenKind::Eof) {
            stmts.push(self.parse_stmt()?);
        }
        self.expect(&TokenKind::RBrace, "'}'")?;
        self.top_level = save_top;
        Ok(stmts)
    }

    // -- var / let / const --------------------------------------------------------

    fn parse_var_stmt(&mut self, kind: VarKind) -> Result<Stmt, SyntaxError> {
        self.bump()?; // var | let | const
        let decls = self.parse_binding_list(kind, true)?;
        self.expect_semi()?;
        Ok(Stmt::VarDecl { kind, decls })
    }

    /// `let` starts a declaration when followed by `[`, `{`, or an
    /// identifier-ish token other than `in` (e.g. `let of = 1` declares
    /// `of`); otherwise `let` is an identifier beginning an expression.
    /// A `let` declaration in single-statement position (`if (x) <here>`)
    /// is an early error; `let` as an identifier is unaffected.
    fn parse_let_or_expr(&mut self, single: bool) -> Result<Stmt, SyntaxError> {
        let is_decl = match self.peek_at(1)? {
            TokenKind::LBracket | TokenKind::LBrace => true,
            TokenKind::In => false,
            k => is_identifierish(k),
        };
        if is_decl {
            if single {
                return Err(Self::early(
                    "lexical declaration in single-statement position is not allowed",
                    self.peek_span()?,
                ));
            }
            self.parse_var_stmt(VarKind::Let)
        } else {
            self.parse_expr_stmt()
        }
    }

    /// `true` when the token at `pos+n` is `using` starting a declaration:
    /// followed by `[`, `{`, or an identifier-ish token other than `in`/`of`
    /// (so `using in x` and `using of y` stay expressions).
    fn using_decl_at(&mut self, n: usize) -> Result<bool, SyntaxError> {
        if !matches!(self.peek_at(n)?, TokenKind::Ident(s) if s.as_str() == "using") {
            return Ok(false);
        }
        Ok(match self.peek_at(n + 1)? {
            TokenKind::LBracket | TokenKind::LBrace => true,
            TokenKind::In | TokenKind::Of => false,
            k => is_identifierish(k),
        })
    }

    /// `using` as a statement start: a declaration when followed by a
    /// binding pattern, otherwise an identifier-led expression statement
    /// (`using = 5;`, `using.foo();`).
    fn parse_using_or_expr(&mut self) -> Result<Stmt, SyntaxError> {
        if self.using_decl_at(0)? {
            self.parse_using_decl(false)
        } else {
            self.parse_expr_stmt()
        }
    }

    /// Parse `using <bindings>;`, or `await using <bindings>;` when
    /// `await_using` (the leading `await` is consumed here).
    fn parse_using_decl(&mut self, await_using: bool) -> Result<Stmt, SyntaxError> {
        if await_using {
            self.bump()?; // await
        }
        self.bump()?; // using
        let kind = if await_using {
            VarKind::AwaitUsing
        } else {
            VarKind::Using
        };
        let decls = self.parse_binding_list(kind, true)?;
        self.expect_semi()?;
        Ok(Stmt::VarDecl { kind, decls })
    }

    /// Parse `a = 1, [b] = c`. When `require_init` is set, `const` and
    /// `using` bindings must have initializers (not for `for-in`/`of` heads).
    fn parse_binding_list(
        &mut self,
        kind: VarKind,
        require_init: bool,
    ) -> Result<Vec<VarDecl>, SyntaxError> {
        let mut decls = Vec::new();
        loop {
            let start = self.peek_span()?;
            // Use parse_pat_primary (not parse_pat) so the top-level `=`
            // is treated as the initializer, not a pattern default.
            let id = self.parse_pat_primary()?;
            self.check_pat_binding(&id)?;
            let init = if matches!(self.peek_at(0)?, TokenKind::Eq) {
                self.bump()?;
                Some(self.parse_assignment()?)
            } else if require_init
                && matches!(kind, VarKind::Const | VarKind::Using | VarKind::AwaitUsing)
            {
                let what = if kind == VarKind::Const {
                    "const"
                } else {
                    "using"
                };
                return Err(Self::early(
                    &format!("{what} declarations must have an initializer"),
                    start,
                ));
            } else {
                None
            };
            let span = join_span(start, self.prev_span()?);
            decls.push(VarDecl { id, init, span });
            if matches!(self.peek_at(0)?, TokenKind::Comma) {
                self.bump()?;
            } else {
                break;
            }
        }
        Ok(decls)
    }

    // -- async statements ---------------------------------------------------------

    /// `async function f() {}` (declaration) or something starting with the
    /// identifier `async`.
    fn parse_async_stmt(&mut self, single: bool) -> Result<Stmt, SyntaxError> {
        if matches!(self.peek_at(1)?, TokenKind::Function) && !self.nl_before(1)? {
            return self.parse_function_decl(true, single);
        }
        self.parse_expr_stmt()
    }

    fn parse_function_decl(&mut self, is_async: bool, single: bool) -> Result<Stmt, SyntaxError> {
        let start = self.peek_span()?;
        if is_async {
            self.bump()?; // async
        }
        self.bump()?; // function
        let (name, params, body, is_generator, strict) =
            self.parse_function_common(is_async, true)?;
        let name = name.expect("declaration requires a name");
        if single {
            // Annex B.3.2: function declarations in single-statement position
            // are allowed in sloppy mode, rejected in strict mode.
            if self.strict {
                return Err(Self::early(
                    "function declaration in single-statement position is not allowed in strict mode",
                    start,
                ));
            }
        }
        Ok(Stmt::FunctionDecl {
            name,
            params,
            body,
            is_async,
            is_generator,
            strict,
        })
    }

    // -- if / loops -----------------------------------------------------------------

    fn parse_if(&mut self, _single: bool) -> Result<Stmt, SyntaxError> {
        self.bump()?; // if
        self.expect(&TokenKind::LParen, "'('")?;
        let cond = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "')'")?;
        // Branches are single-statement positions.
        let then_b = Box::new(self.guarded(|s| s.parse_stmt_inner(true))?);
        let else_b = if matches!(self.peek_at(0)?, TokenKind::Else) {
            self.bump()?;
            Some(Box::new(self.guarded(|s| s.parse_stmt_inner(true))?))
        } else {
            None
        };
        Ok(Stmt::If {
            cond,
            then_b,
            else_b,
        })
    }

    fn parse_while(&mut self) -> Result<Stmt, SyntaxError> {
        self.bump()?; // while
        self.expect(&TokenKind::LParen, "'('")?;
        let cond = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "')'")?;
        self.iter_depth += 1;
        self.breakable_depth += 1;
        let body = self.parse_loop_body()?;
        self.iter_depth -= 1;
        self.breakable_depth -= 1;
        Ok(Stmt::While {
            cond,
            body: Box::new(body),
        })
    }

    fn parse_do(&mut self) -> Result<Stmt, SyntaxError> {
        self.bump()?; // do
        self.iter_depth += 1;
        self.breakable_depth += 1;
        let body = self.parse_loop_body()?;
        self.iter_depth -= 1;
        self.breakable_depth -= 1;
        self.expect(&TokenKind::While, "'while'")?;
        self.expect(&TokenKind::LParen, "'('")?;
        let cond = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "')'")?;
        // `do-while` semicolon is optional via ASI as usual.
        self.expect_semi()?;
        Ok(Stmt::DoWhile {
            body: Box::new(body),
            cond,
        })
    }

    /// Parse a loop body with iteration-label tracking. A loop body is a
    /// single-statement position (`while (x) <here>`), so lexical
    /// declarations and (in strict mode) function/class declarations are
    /// rejected there, exactly like `if` branches.
    fn parse_loop_body(&mut self) -> Result<Stmt, SyntaxError> {
        self.guarded(|s| s.parse_stmt_inner(true))
    }

    fn parse_for(&mut self) -> Result<Stmt, SyntaxError> {
        let for_span = self.bump()?; // for
        let is_await = if matches!(self.peek_at(0)?, TokenKind::Await) && !self.nl_before(0)? {
            // `for await (` — only valid inside async functions.
            if !self.in_async {
                return Err(Self::early(
                    "for await is only allowed inside async functions",
                    for_span,
                ));
            }
            self.bump()?;
            true
        } else {
            false
        };
        self.expect(&TokenKind::LParen, "'('")?;

        // Declaration head?
        let mut init: Option<ForInit> = None;
        let mut decl_kind: Option<VarKind> = None;
        // `using` as a for-head declaration (computed before the match so
        // the `?` does not sit in a match guard).
        let using_is_decl = matches!(self.peek_at(0)?, TokenKind::Ident(s) if s.as_str() == "using")
            && self.using_decl_at(0)?;
        match self.peek_at(0)? {
            TokenKind::Var => {
                self.bump()?;
                decl_kind = Some(VarKind::Var);
            }
            TokenKind::Const => {
                self.bump()?;
                decl_kind = Some(VarKind::Const);
            }
            TokenKind::Let => {
                // `for (let ...`: declaration unless `let in` / `let of`.
                let is_decl = match self.peek_at(1)? {
                    TokenKind::LBracket | TokenKind::LBrace => true,
                    TokenKind::In | TokenKind::Of => false,
                    k => is_identifierish(k),
                };
                if is_decl {
                    self.bump()?;
                    decl_kind = Some(VarKind::Let);
                }
            }
            TokenKind::Ident(s) if s.as_str() == "using" && using_is_decl => {
                // `for (using x of ...)`. (`await using` is not allowed in
                // for heads.)
                self.bump()?;
                decl_kind = Some(VarKind::Using);
            }
            _ => {}
        }
        if let Some(kind) = decl_kind {
            // Disable `in` as a binary operator; `for (var x = 1 in y)` —
            // the `in` is the for-in keyword, not part of the initializer.
            let save_in = self.allow_in;
            self.allow_in = false;
            let decls = self.parse_binding_list(kind, false)?;
            self.allow_in = save_in;
            init = Some(ForInit::Var(kind, decls));
        } else {
            // Expression head (may be empty).
            if !matches!(self.peek_at(0)?, TokenKind::Semicolon) {
                let save = self.allow_in;
                self.allow_in = false;
                self.in_for_head = true;
                let e = self.parse_expr();
                self.in_for_head = false;
                self.allow_in = save;
                init = Some(ForInit::Expr(e?));
            }
        }

        // `for-in` / `for-of`?
        let is_in = matches!(self.peek_at(0)?, TokenKind::In);
        let is_of = matches!(self.peek_at(0)?, TokenKind::Of);
        if is_in || is_of {
            if is_await && !is_of {
                return Err(Self::early(
                    "for await...of requires 'of'",
                    self.peek_span()?,
                ));
            }
            let lhs_span = self.peek_span()?;
            // Declared heads record their kind; bare LHS forms use Var.
            let mut kind = VarKind::Var;
            let lhs = match init {
                Some(ForInit::Var(k, mut decls)) => {
                    if decls.len() != 1 {
                        return Err(Self::early(
                            "for-in/of loop head must declare exactly one variable",
                            lhs_span,
                        ));
                    }
                    kind = k;
                    let d = decls.pop().expect("one decl");
                    if d.init.is_some() {
                        // Annex B.3.5: `for (var x = init in y)` in sloppy mode.
                        let ok = !self.strict && k == VarKind::Var && is_in;
                        if !ok {
                            return Err(Self::early(
                                "for-in/of loop variable declaration may not have an initializer",
                                d.span,
                            ));
                        }
                    }
                    match d.id {
                        Pat::Ident(name, span) => ForLhs::Ident(name, span),
                        pat => ForLhs::Pat(pat),
                    }
                }
                Some(ForInit::Expr(e)) => {
                    let span = e.span();
                    match self.expr_to_for_lhs(&e) {
                        Ok(lhs) => lhs,
                        Err(_) => {
                            return Err(Self::early("invalid left-hand side in for-loop", span));
                        }
                    }
                }
                None => return Err(self.unexpected("a for-loop head")),
            };
            // Strict-mode `eval`/`arguments` as loop target.
            self.check_for_lhs_strict(&lhs, lhs_span)?;
            // The LHS validated as a for-in/of target: any pending cover
            // initializer is consumed. Clear BEFORE parsing the RHS so the
            // RHS's own tail check doesn't trip on the stale flag
            // (`for ([a = 1] of b)` would otherwise die on the `b`).
            self.pending_cover = false;
            self.bump()?; // in | of
            let rhs = self.parse_expr()?;
            self.expect(&TokenKind::RParen, "')'")?;
            self.iter_depth += 1;
            self.breakable_depth += 1;
            let body = self.parse_loop_body()?;
            self.iter_depth -= 1;
            self.breakable_depth -= 1;
            return Ok(Stmt::ForInOf {
                kind,
                lhs,
                rhs,
                body: Box::new(body),
                is_of,
            });
        }

        // Classic `for (init; cond; update)`.
        // `using` in a C-style for head still requires an initializer
        // (for-in/of heads get their value from iteration instead).
        if let Some(ForInit::Var(k, decls)) = &init
            && matches!(k, VarKind::Using | VarKind::AwaitUsing)
            && let Some(d) = decls.iter().find(|d| d.init.is_none())
        {
            return Err(Self::early(
                "using declarations must have an initializer",
                d.span,
            ));
        }
        self.expect(&TokenKind::Semicolon, "';'")?;
        let cond = if matches!(self.peek_at(0)?, TokenKind::Semicolon) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        self.expect(&TokenKind::Semicolon, "';'")?;
        let update = if matches!(self.peek_at(0)?, TokenKind::RParen) {
            None
        } else {
            let save = self.allow_in;
            self.allow_in = true;
            let e = self.parse_expr();
            self.allow_in = save;
            Some(e?)
        };
        self.expect(&TokenKind::RParen, "')'")?;
        self.iter_depth += 1;
        self.breakable_depth += 1;
        let body = self.parse_loop_body()?;
        self.iter_depth -= 1;
        self.breakable_depth -= 1;
        let _ = is_await; // `for await (` without `of` was rejected above.
        Ok(Stmt::For {
            init,
            cond,
            update,
            body: Box::new(body),
        })
    }

    /// Strict-mode early error for `eval`/`arguments` as a for-in/of target.
    fn check_for_lhs_strict(&mut self, lhs: &ForLhs, span: Span) -> Result<(), SyntaxError> {
        if !self.strict {
            return Ok(());
        }
        let bad = match lhs {
            ForLhs::Ident(name, _) => name == "eval" || name == "arguments",
            ForLhs::Pat(pat) => pat_is_eval_or_arguments(pat),
            ForLhs::Member(_) => false,
        };
        if bad {
            return Err(Self::early(
                "eval/arguments may not be a for-in/of target in strict mode",
                span,
            ));
        }
        Ok(())
    }
}

impl<'a> Parser<'a> {
    // -- switch -------------------------------------------------------------------

    fn parse_switch(&mut self) -> Result<Stmt, SyntaxError> {
        self.bump()?; // switch
        self.expect(&TokenKind::LParen, "'('")?;
        let disc = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "')'")?;
        self.expect(&TokenKind::LBrace, "'{'")?;
        let save_top = self.top_level;
        self.top_level = false;
        self.breakable_depth += 1;
        let mut cases = Vec::new();
        let mut seen_default = false;
        while !matches!(self.peek_at(0)?, TokenKind::RBrace | TokenKind::Eof) {
            let start = self.peek_span()?;
            let test = if matches!(self.peek_at(0)?, TokenKind::Case) {
                self.bump()?;
                Some(self.parse_expr()?)
            } else if matches!(self.peek_at(0)?, TokenKind::Default) {
                if seen_default {
                    return Err(Self::early(
                        "more than one default clause in switch statement",
                        start,
                    ));
                }
                seen_default = true;
                self.bump()?;
                None
            } else {
                return Err(self.unexpected("'case' or 'default'"));
            };
            self.expect(&TokenKind::Colon, "':'")?;
            let mut body = Vec::new();
            while !matches!(
                self.peek_at(0)?,
                TokenKind::Case | TokenKind::Default | TokenKind::RBrace | TokenKind::Eof
            ) {
                body.push(self.parse_stmt()?);
            }
            cases.push(SwitchCase { test, body });
        }
        self.expect(&TokenKind::RBrace, "'}'")?;
        self.breakable_depth -= 1;
        self.top_level = save_top;
        Ok(Stmt::Switch { disc, cases })
    }

    // -- try ------------------------------------------------------------------------

    fn parse_try(&mut self) -> Result<Stmt, SyntaxError> {
        self.bump()?; // try
        let block = self.parse_block()?;
        let mut handler = None;
        let mut finalizer = None;
        if matches!(self.peek_at(0)?, TokenKind::Catch) {
            let _ = self.bump()?;
            let param = if matches!(self.peek_at(0)?, TokenKind::LParen) {
                self.bump()?;
                let p = if matches!(self.peek_at(0)?, TokenKind::RParen) {
                    None
                } else {
                    let start = self.peek_span()?;
                    let pat = self.parse_pat()?;
                    // `catch (let)` etc: `let` is disallowed as a catch binding.
                    self.check_pat_binding(&pat)?;
                    let span = join_span(start, self.prev_span()?);
                    Some(Param { pat, span })
                };
                self.expect(&TokenKind::RParen, "')'")?;
                p
            } else {
                None
            };
            let body = self.parse_block()?;
            handler = Some(CatchClause { param, body });
        }
        if matches!(self.peek_at(0)?, TokenKind::Finally) {
            self.bump()?;
            finalizer = Some(self.parse_block()?);
        }
        if handler.is_none() && finalizer.is_none() {
            return Err(Self::early(
                "try statement requires catch or finally",
                self.peek_span()?,
            ));
        }
        Ok(Stmt::Try {
            block,
            handler,
            finalizer,
        })
    }

    // -- jumps ------------------------------------------------------------------------

    fn parse_return(&mut self) -> Result<Stmt, SyntaxError> {
        // `return` outside any function is always an error (Script and Module).
        // Static blocks are not functions even when nested inside one.
        if self.func_depth == 0 || self.in_static_block {
            return Err(Self::early("return outside function", self.peek_span()?));
        }
        self.bump()?; // return
        let arg = if self.nl_before(0)?
            || matches!(
                self.peek_at(0)?,
                TokenKind::Semicolon | TokenKind::RBrace | TokenKind::Eof
            ) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        self.expect_semi()?;
        Ok(Stmt::Return(arg))
    }

    fn parse_throw(&mut self) -> Result<Stmt, SyntaxError> {
        self.bump()?; // throw
        // No line terminator allowed between `throw` and its argument; no ASI.
        if self.nl_before(0)?
            || matches!(
                self.peek_at(0)?,
                TokenKind::Semicolon | TokenKind::RBrace | TokenKind::Eof
            )
        {
            return Err(self.unexpected("an expression"));
        }
        let arg = self.parse_expr()?;
        self.expect_semi()?;
        Ok(Stmt::Throw(arg))
    }

    fn parse_break(&mut self) -> Result<Stmt, SyntaxError> {
        self.bump()?; // break
        let label = self.parse_jump_label()?;
        self.expect_semi()?;
        match &label {
            None => {
                if self.breakable_depth == 0 {
                    return Err(Self::early(
                        "break outside of a loop or switch",
                        self.prev_span()?,
                    ));
                }
            }
            Some((name, span)) => {
                if !self.labels.iter().any(|l| &l.name == name) {
                    return Err(Self::early(&format!("undefined label '{name}'"), *span));
                }
            }
        }
        Ok(Stmt::Break(label.map(|(n, _)| n)))
    }

    fn parse_continue(&mut self) -> Result<Stmt, SyntaxError> {
        self.bump()?; // continue
        let label = self.parse_jump_label()?;
        self.expect_semi()?;
        match &label {
            None => {
                if self.iter_depth == 0 {
                    return Err(Self::early("continue outside of a loop", self.prev_span()?));
                }
            }
            Some((name, span)) => match self.labels.iter().find(|l| &l.name == name) {
                Some(l) if l.is_iteration => {}
                Some(_) => {
                    return Err(Self::early(
                        &format!("continue label '{name}' does not name an iteration statement"),
                        *span,
                    ));
                }
                None => {
                    return Err(Self::early(&format!("undefined label '{name}'"), *span));
                }
            },
        }
        Ok(Stmt::Continue(label.map(|(n, _)| n)))
    }

    /// Parse the optional label after `break` / `continue` (no line break).
    fn parse_jump_label(&mut self) -> Result<Option<(String, Span)>, SyntaxError> {
        if self.nl_before(0)? {
            return Ok(None);
        }
        match self.peek_at(0)? {
            TokenKind::Ident(_) => {
                let (name, span) = self.parse_ident_name()?;
                Ok(Some((name, span)))
            }
            k => match contextual_name(k) {
                // `await`/`yield` as operators can't be labels; they would
                // have been parsed as expression starts. As identifiers they
                // can still label.
                Some(n)
                    if !(n == "await" && self.await_allowed())
                        && !(n == "yield" && self.yield_is_operator()) =>
                {
                    let span = self.bump()?;
                    Ok(Some((n.to_string(), span)))
                }
                _ => Ok(None),
            },
        }
    }

    // -- labels -------------------------------------------------------------------------

    fn parse_labeled(
        &mut self,
        name: String,
        span: Span,
        single: bool,
    ) -> Result<Stmt, SyntaxError> {
        // Duplicate labels in the active set are an early error.
        if self.labels.iter().any(|l| l.name == name) {
            return Err(Self::early(&format!("duplicate label '{name}'"), span));
        }
        self.bump()?; // label name
        self.expect(&TokenKind::Colon, "':'")?;
        // Annex B.3.4: labelled function declarations in sloppy mode.
        let is_iteration = matches!(
            self.peek_at(0)?,
            TokenKind::For | TokenKind::While | TokenKind::Do
        );
        self.labels.push(Label {
            name: name.clone(),
            is_iteration,
        });
        let body = self.guarded(|s| s.parse_stmt_inner(single))?;
        self.labels.pop();
        // Strict-mode labelled function declaration is an error; the function
        // parser already rejected it via `single`.
        Ok(Stmt::Labeled {
            label: name,
            body: Box::new(body),
        })
    }

    // -- with -------------------------------------------------------------------------------

    fn parse_with(&mut self, _single: bool) -> Result<Stmt, SyntaxError> {
        let start = self.bump()?; // with
        if self.strict {
            return Err(Self::early(
                "with statement is not allowed in strict mode",
                start,
            ));
        }
        self.expect(&TokenKind::LParen, "'('")?;
        let obj = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "')'")?;
        let body = Box::new(self.guarded(|s| s.parse_stmt_inner(false))?);
        Ok(Stmt::With { obj, body })
    }

    // -- class declarations -------------------------------------------------------------------

    fn parse_class_decl(&mut self, single: bool) -> Result<Stmt, SyntaxError> {
        let start = self.peek_span()?;
        if single && self.strict {
            return Err(Self::early(
                "class declaration in single-statement position is not allowed in strict mode",
                start,
            ));
        }
        let (name, superclass, body) = self.parse_class_common(start, true)?;
        let name = name.expect("class declaration requires a name");
        Ok(Stmt::ClassDecl {
            name: Some(name),
            superclass: superclass.map(|b| *b),
            body,
        })
    }
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

impl<'a> Parser<'a> {
    /// Parse an Expression.
    /// Parse an Expression (including the comma operator).
    fn parse_expr(&mut self) -> Result<Expr, SyntaxError> {
        self.guarded(|s| {
            let first = s.parse_assignment()?;
            if !matches!(s.peek_at(0)?, TokenKind::Comma) {
                return Ok(first);
            }
            let mut exprs = vec![first];
            while matches!(s.peek_at(0)?, TokenKind::Comma) {
                s.bump()?;
                exprs.push(s.parse_assignment()?);
            }
            let span = join_span(exprs[0].span(), s.prev_span()?);
            Ok(Expr::Sequence(exprs, span))
        })
    }

    fn parse_assignment(&mut self) -> Result<Expr, SyntaxError> {
        // Depth-guarded entry: every nested assignment (`x = ...` RHS,
        // `?:` branches, call arguments, array/object literals, arrow
        // bodies) funnels through here, so one guard covers all of the
        // unbounded recursion instead of each call site.
        self.guarded(|s| s.parse_assignment_inner())
    }

    fn parse_assignment_inner(&mut self) -> Result<Expr, SyntaxError> {
        // `yield` is an AssignmentExpression (not a primary): handle it
        // before the conditional so `typeof yield` stays an error.
        if matches!(self.peek_at(0)?, TokenKind::Yield) && self.yield_is_operator() {
            return self.parse_yield();
        }
        // The LHS may turn out to be a destructuring target, so cover
        // initializers (`{a = 1}`) are tentatively allowed while parsing it.
        let cover = self.cover_ok;
        self.cover_ok = true;
        let lhs = self.parse_conditional();
        self.cover_ok = cover;
        let mut lhs = lhs?;
        // Single-identifier arrow: `x => ...` (no line break before `=>`).
        if matches!(self.peek_at(0)?, TokenKind::Arrow) && !self.nl_before(0)? {
            if !matches!(lhs, Expr::Ident(_, _)) {
                return Err(Self::early(
                    "unexpected '=>' after expression",
                    self.peek_span()?,
                ));
            }
            // `Expr` has a manual `Drop` (iterative spine freeing), so its
            // fields cannot move out of a by-value pattern: take them
            // through `&mut` instead (no clones).
            let (name, span) = match &mut lhs {
                Expr::Ident(name, span) => (std::mem::take(name), *span),
                _ => unreachable!("checked above"),
            };
            self.bump()?; // =>
            self.check_binding_name(&name, span)?;
            let params = vec![Param {
                pat: Pat::Ident(name, span),
                span,
            }];
            return self.finish_arrow_body(span, params, false);
        }
        if let Some(op) = self.peek_assign_op()? {
            self.bump()?;
            self.check_assign_target(&lhs)?;
            // The target validated: any pending cover initializer is consumed
            // by this `=`. Clear BEFORE parsing the RHS so the RHS's own
            // tail check (`check_no_cover_init`) doesn't trip on the stale
            // flag — otherwise `[a = 1] = []` dies on the `[]`.
            self.pending_cover = false;
            let rhs = self.parse_assignment()?;
            let span = join_span(lhs.span(), self.prev_span()?);
            return Ok(Expr::Assign {
                op,
                left: Box::new(lhs),
                right: Box::new(rhs),
                span,
            });
        }
        // `for (LHS in/of ...)`: validation deferred to parse_for.
        let defer_for =
            self.in_for_head && (matches!(self.peek_at(0)?, TokenKind::In | TokenKind::Of));
        // `(...)`: validation deferred to the paren handler.
        if !(defer_for || self.in_paren_cover) {
            self.check_no_cover_init(&lhs)?;
        }
        Ok(lhs)
    }

    /// Reject a tentatively-parsed cover initializer (`{a = 1}`) that did
    /// not become a destructuring target.
    fn check_no_cover_init(&mut self, e: &Expr) -> Result<(), SyntaxError> {
        if self.pending_cover {
            return Err(Self::early(
                "invalid initializer in object/array pattern",
                e.span(),
            ));
        }
        Ok(())
    }

    fn peek_assign_op(&mut self) -> Result<Option<AssignOp>, SyntaxError> {
        Ok(match self.peek_at(0)? {
            TokenKind::Eq => Some(AssignOp::Assign),
            TokenKind::PlusEq => Some(AssignOp::Add),
            TokenKind::MinusEq => Some(AssignOp::Sub),
            TokenKind::StarEq => Some(AssignOp::Mul),
            TokenKind::SlashEq => Some(AssignOp::Div),
            TokenKind::PercentEq => Some(AssignOp::Mod),
            TokenKind::StarStarEq => Some(AssignOp::Pow),
            TokenKind::LtLtEq => Some(AssignOp::LShift),
            TokenKind::GtGtEq => Some(AssignOp::RShift),
            TokenKind::GtGtGtEq => Some(AssignOp::URShift),
            TokenKind::AmpEq => Some(AssignOp::BitAnd),
            TokenKind::PipeEq => Some(AssignOp::BitOr),
            TokenKind::CaretEq => Some(AssignOp::BitXor),
            TokenKind::AmpAmpEq => Some(AssignOp::And),
            TokenKind::PipePipeEq => Some(AssignOp::Or),
            TokenKind::QuestionQuestionEq => Some(AssignOp::Nullish),
            _ => None,
        })
    }

    fn parse_conditional(&mut self) -> Result<Expr, SyntaxError> {
        let cond = self.parse_logical_or()?;
        if matches!(self.peek_at(0)?, TokenKind::Question) {
            self.bump()?;
            let save = self.allow_in;
            self.allow_in = true;
            let then_e = self.parse_assignment();
            self.allow_in = save;
            let then_e = then_e?;
            self.expect(&TokenKind::Colon, "':'")?;
            let else_e = self.parse_assignment()?;
            let span = join_span(cond.span(), self.prev_span()?);
            return Ok(Expr::Conditional {
                cond: Box::new(cond),
                then_e: Box::new(then_e),
                else_e: Box::new(else_e),
                span,
            });
        }
        Ok(cond)
    }

    // -- `||`, `&&`, `??` with mixing bans ------------------------------------------------------

    fn parse_logical_or(&mut self) -> Result<Expr, SyntaxError> {
        // Owns `ban_mix_logical`: set by parse_coalesce when it consumes an
        // unparenthesized `??`.
        let save = self.ban_mix_logical;
        self.ban_mix_logical = false;
        let mut lhs = self.parse_logical_and()?;
        while matches!(self.peek_at(0)?, TokenKind::PipePipe) {
            if self.ban_mix_logical {
                let span = self.peek_span()?;
                self.ban_mix_logical = save;
                return Err(Self::early(
                    "'||' cannot be mixed with '??' without parentheses",
                    span,
                ));
            }
            self.bump()?;
            let save2 = self.ban_mix_nullish;
            self.ban_mix_nullish = true;
            let rhs = self.parse_logical_and();
            self.ban_mix_nullish = save2;
            let rhs = rhs?;
            let span = join_span(lhs.span(), self.prev_span()?);
            lhs = Expr::Logical {
                op: LogicalOp::Or,
                left: Box::new(lhs),
                right: Box::new(rhs),
                span,
            };
            self.ban_mix_logical = false;
        }
        self.ban_mix_logical = save;
        Ok(lhs)
    }

    fn parse_logical_and(&mut self) -> Result<Expr, SyntaxError> {
        let mut lhs = self.parse_coalesce()?;
        while matches!(self.peek_at(0)?, TokenKind::AmpAmp) {
            if self.ban_mix_logical {
                return Err(Self::early(
                    "'&&' cannot be mixed with '??' without parentheses",
                    self.peek_span()?,
                ));
            }
            self.bump()?;
            let save = self.ban_mix_nullish;
            self.ban_mix_nullish = true;
            let rhs = self.parse_coalesce();
            self.ban_mix_nullish = save;
            let rhs = rhs?;
            let span = join_span(lhs.span(), self.prev_span()?);
            lhs = Expr::Logical {
                op: LogicalOp::And,
                left: Box::new(lhs),
                right: Box::new(rhs),
                span,
            };
            self.ban_mix_logical = false;
        }
        Ok(lhs)
    }

    fn parse_coalesce(&mut self) -> Result<Expr, SyntaxError> {
        let mut lhs = self.parse_bitwise_or()?;
        let mut saw = false;
        while matches!(self.peek_at(0)?, TokenKind::QuestionQuestion) {
            if self.ban_mix_nullish {
                return Err(Self::early(
                    "'??' cannot be mixed with '&&'/'||' without parentheses",
                    self.peek_span()?,
                ));
            }
            saw = true;
            self.bump()?;
            let save = self.ban_mix_logical;
            self.ban_mix_logical = true;
            let rhs = self.parse_bitwise_or();
            self.ban_mix_logical = save;
            let rhs = rhs?;
            let span = join_span(lhs.span(), self.prev_span()?);
            lhs = Expr::Logical {
                op: LogicalOp::Nullish,
                left: Box::new(lhs),
                right: Box::new(rhs),
                span,
            };
        }
        if saw {
            // Signal the enclosing `||`/`&&` level (owned by parse_logical_or).
            self.ban_mix_logical = true;
        }
        Ok(lhs)
    }

    // -- binary levels ----------------------------------------------------------------------------

    fn parse_bitwise_or(&mut self) -> Result<Expr, SyntaxError> {
        let mut lhs = self.parse_bitwise_xor()?;
        while matches!(self.peek_at(0)?, TokenKind::Pipe) {
            self.bump()?;
            let rhs = self.parse_bitwise_xor()?;
            let span = join_span(lhs.span(), self.prev_span()?);
            lhs = Expr::Binary {
                op: BinaryOp::BitOr,
                left: Box::new(lhs),
                right: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_bitwise_xor(&mut self) -> Result<Expr, SyntaxError> {
        let mut lhs = self.parse_bitwise_and()?;
        while matches!(self.peek_at(0)?, TokenKind::Caret) {
            self.bump()?;
            let rhs = self.parse_bitwise_and()?;
            let span = join_span(lhs.span(), self.prev_span()?);
            lhs = Expr::Binary {
                op: BinaryOp::BitXor,
                left: Box::new(lhs),
                right: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_bitwise_and(&mut self) -> Result<Expr, SyntaxError> {
        let mut lhs = self.parse_equality()?;
        while matches!(self.peek_at(0)?, TokenKind::Amp) {
            self.bump()?;
            let rhs = self.parse_equality()?;
            let span = join_span(lhs.span(), self.prev_span()?);
            lhs = Expr::Binary {
                op: BinaryOp::BitAnd,
                left: Box::new(lhs),
                right: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_equality(&mut self) -> Result<Expr, SyntaxError> {
        let mut lhs = self.parse_relational()?;
        loop {
            let op = match self.peek_at(0)? {
                TokenKind::EqEq => BinaryOp::Eq,
                TokenKind::NotEq => BinaryOp::Ne,
                TokenKind::EqEqEq => BinaryOp::StrictEq,
                TokenKind::NotEqEq => BinaryOp::StrictNe,
                _ => break,
            };
            self.bump()?;
            let rhs = self.parse_relational()?;
            let span = join_span(lhs.span(), self.prev_span()?);
            lhs = Expr::Binary {
                op,
                left: Box::new(lhs),
                right: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_relational(&mut self) -> Result<Expr, SyntaxError> {
        let mut lhs = self.parse_shift()?;
        // A bare `#x` is only valid as the LHS of `in`.
        if let Expr::PrivateIdent(pname, pspan) = &lhs {
            let is_in = matches!(self.peek_at(0)?, TokenKind::In) && self.allow_in;
            if is_in {
                self.bump()?; // in
                let rhs = self.parse_shift()?;
                self.use_private(pname, *pspan)?;
                let span = join_span(*pspan, self.prev_span()?);
                return Ok(Expr::Binary {
                    op: BinaryOp::In,
                    left: Box::new(Expr::PrivateIdent(pname.clone(), *pspan)),
                    right: Box::new(rhs),
                    span,
                });
            } else {
                return Err(Self::early(
                    "private name must be used in a member access or 'in' expression",
                    *pspan,
                ));
            }
        }
        loop {
            let allow_in = self.allow_in;
            let op = match self.peek_at(0)? {
                TokenKind::Lt => BinaryOp::Lt,
                TokenKind::Gt => BinaryOp::Gt,
                TokenKind::LtEq => BinaryOp::LtEq,
                TokenKind::GtEq => BinaryOp::GtEq,
                TokenKind::Instanceof => BinaryOp::Instanceof,
                TokenKind::In if allow_in => BinaryOp::In,
                _ => break,
            };
            self.bump()?;
            let rhs = self.parse_shift()?;
            let span = join_span(lhs.span(), self.prev_span()?);
            lhs = Expr::Binary {
                op,
                left: Box::new(lhs),
                right: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_shift(&mut self) -> Result<Expr, SyntaxError> {
        let mut lhs = self.parse_additive()?;
        loop {
            let op = match self.peek_at(0)? {
                TokenKind::LtLt => BinaryOp::LShift,
                TokenKind::GtGt => BinaryOp::RShift,
                TokenKind::GtGtGt => BinaryOp::URShift,
                _ => break,
            };
            self.bump()?;
            let rhs = self.parse_additive()?;
            let span = join_span(lhs.span(), self.prev_span()?);
            lhs = Expr::Binary {
                op,
                left: Box::new(lhs),
                right: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_additive(&mut self) -> Result<Expr, SyntaxError> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            let op = match self.peek_at(0)? {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.bump()?;
            let rhs = self.parse_multiplicative()?;
            let span = join_span(lhs.span(), self.prev_span()?);
            lhs = Expr::Binary {
                op,
                left: Box::new(lhs),
                right: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, SyntaxError> {
        let mut lhs = self.parse_exponentiation()?;
        loop {
            let op = match self.peek_at(0)? {
                TokenKind::Star => BinaryOp::Mul,
                // `/` here is always division: regexes are re-lexed in
                // parse_primary before the operator loop can see them.
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::Percent => BinaryOp::Mod,
                _ => break,
            };
            self.bump()?;
            let rhs = self.parse_exponentiation()?;
            let span = join_span(lhs.span(), self.prev_span()?);
            lhs = Expr::Binary {
                op,
                left: Box::new(lhs),
                right: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    /// `**` is right-associative and rejects a unary/`await` left operand
    /// unless parenthesized.
    fn parse_exponentiation(&mut self) -> Result<Expr, SyntaxError> {
        // Depth-guarded entry: `**` is right-recursive.
        self.guarded(|s| s.parse_exponentiation_inner())
    }

    fn parse_exponentiation_inner(&mut self) -> Result<Expr, SyntaxError> {
        let (lhs, paren) = self.parse_unary_p()?;
        if matches!(self.peek_at(0)?, TokenKind::StarStar) {
            if !paren
                && matches!(
                    lhs,
                    Expr::Unary { .. } | Expr::Update { .. } | Expr::Await(..)
                )
            {
                return Err(Self::early(
                    "unary operator used immediately before exponentiation expression",
                    lhs.span(),
                ));
            }
            self.bump()?;
            let rhs = self.parse_exponentiation()?;
            let span = join_span(lhs.span(), self.prev_span()?);
            return Ok(Expr::Binary {
                op: BinaryOp::Pow,
                left: Box::new(lhs),
                right: Box::new(rhs),
                span,
            });
        }
        Ok(lhs)
    }

    /// Parse a UnaryExpression. Returns `(expr, parenthesized)`: the flag
    /// records whether the top-level node came from `( ... )`, which the
    /// `**` check needs.
    fn parse_unary_p(&mut self) -> Result<(Expr, bool), SyntaxError> {
        // Depth-guarded entry: prefix operators recurse here.
        self.guarded(|s| s.parse_unary_p_inner())
    }

    fn parse_unary_p_inner(&mut self) -> Result<(Expr, bool), SyntaxError> {
        match self.peek_at(0)? {
            TokenKind::PlusPlus | TokenKind::MinusMinus => {
                let start = self.peek_span()?;
                let op = if matches!(self.peek_at(0)?, TokenKind::PlusPlus) {
                    UpdateOp::Inc
                } else {
                    UpdateOp::Dec
                };
                self.bump()?;
                let (arg, _) = self.parse_unary_p()?;
                self.check_update_target(&arg)?;
                let span = join_span(start, self.prev_span()?);
                Ok((
                    Expr::Update {
                        op,
                        arg: Box::new(arg),
                        prefix: true,
                        span,
                    },
                    false,
                ))
            }
            TokenKind::Typeof
            | TokenKind::Void
            | TokenKind::Delete
            | TokenKind::Plus
            | TokenKind::Minus
            | TokenKind::Bang
            | TokenKind::Tilde => {
                let start = self.peek_span()?;
                let op = match self.peek_at(0)? {
                    TokenKind::Typeof => UnaryOp::Typeof,
                    TokenKind::Void => UnaryOp::Void,
                    TokenKind::Delete => UnaryOp::Delete,
                    TokenKind::Plus => UnaryOp::Plus,
                    TokenKind::Minus => UnaryOp::Minus,
                    TokenKind::Bang => UnaryOp::Bang,
                    _ => UnaryOp::Tilde,
                };
                self.bump()?;
                let (arg, _) = self.parse_unary_p()?;
                if op == UnaryOp::Delete {
                    self.check_delete(&arg)?;
                }
                let span = join_span(start, self.prev_span()?);
                Ok((
                    Expr::Unary {
                        op,
                        arg: Box::new(arg),
                        span,
                    },
                    false,
                ))
            }
            _ => {
                // `await` as an operator (its operand is a UnaryExpression).
                if matches!(self.peek_at(0)?, TokenKind::Await) && self.await_allowed() {
                    let start = self.bump()?;
                    let (arg, _) = self.parse_unary_p()?;
                    let span = join_span(start, self.prev_span()?);
                    // `await a ** b` is rejected by parse_exponentiation.
                    return Ok((Expr::Await(Box::new(arg), span), false));
                }
                self.parse_lhs()
            }
        }
    }

    /// `delete` operand rules: strict-mode bare identifier is an error;
    /// `#x` is an error everywhere.
    fn check_delete(&mut self, arg: &Expr) -> Result<(), SyntaxError> {
        match arg {
            Expr::Ident(name, span) if self.strict => {
                let _ = name;
                Err(Self::early(
                    "delete of an unqualified identifier is not allowed in strict mode",
                    *span,
                ))
            }
            Expr::PrivateIdent(_, span) => Err(Self::early(
                "delete of a private field is not allowed",
                *span,
            )),
            Expr::Member { prop, .. } if matches!(**prop, Expr::PrivateIdent(..)) => Err(
                Self::early("delete of a private field is not allowed", prop.span()),
            ),
            _ => Ok(()),
        }
    }
}

impl<'a> Parser<'a> {
    /// Parse a LeftHandSideExpression: primaries, `new`, member/call
    /// suffixes, and postfix updates. Regexes are re-lexed here when the
    /// slash is in operator position.
    fn parse_lhs(&mut self) -> Result<(Expr, bool), SyntaxError> {
        // `new` prefix.
        let mut news = 0;
        while matches!(self.peek_at(0)?, TokenKind::New) {
            self.bump()?;
            news += 1;
        }
        let paren = matches!(self.peek_at(0)?, TokenKind::LParen);
        let mut e = if matches!(self.peek_at(0)?, TokenKind::MetaPropertyNewTarget) {
            let span = self.bump()?;
            self.finish_new_target(span)?
        } else {
            if news > 0 && matches!(self.peek_at(0)?, TokenKind::Dot) {
                // `new . target` — not the `new.target` meta property.
                return Err(self.unexpected("an expression"));
            }
            self.parse_primary()?
        };
        e = self.finish_member_suffix(e)?;
        // An optional chain may not be the direct operand of `new`
        // (`new (a?.b)` with explicit parens is fine).
        if news > 0 && !paren && is_optional_chain(&e) {
            return Err(Self::early(
                "optional chain may not be the operand of new",
                e.span(),
            ));
        }
        // Postfix `++` / `--`: no line break between operand and operator.
        while !self.nl_before(0)?
            && matches!(
                self.peek_at(0)?,
                TokenKind::PlusPlus | TokenKind::MinusMinus
            )
        {
            let op = if matches!(self.peek_at(0)?, TokenKind::PlusPlus) {
                UpdateOp::Inc
            } else {
                UpdateOp::Dec
            };
            let op_span = self.bump()?;
            self.check_update_target(&e)?;
            let span = join_span(e.span(), op_span);
            e = Expr::Update {
                op,
                arg: Box::new(e),
                prefix: false,
                span,
            };
        }
        // Give the argument list to the innermost `new`: `new F(a)(b)` is
        // `(new F(a))(b)`, and `new (F(a))` is `new` with no arguments
        // applied to the parenthesized call. Outer `new`s never take an
        // argument list directly (`new new F(a)` is `new (new F(a))`).
        if news > 0 {
            news -= 1;
            if matches!(self.peek_at(0)?, TokenKind::LParen) {
                // `super(...)` is only valid in a derived constructor.
                if matches!(e, Expr::Super(_)) && !self.in_derived_ctor {
                    return Err(Self::early(
                        "super() is only valid in a derived class constructor",
                        e.span(),
                    ));
                }
                let args = self.parse_args()?;
                let span = join_span(e.span(), self.prev_span()?);
                e = Expr::New {
                    callee: Box::new(e),
                    args,
                    span,
                };
            } else {
                let span = join_span(self.prev_span()?, e.span());
                // `new f` without parens: args = [].
                e = Expr::New {
                    callee: Box::new(e),
                    args: Vec::new(),
                    span,
                };
            }
        }
        // Remaining call suffixes apply outside the `new`.
        e = self.finish_suffix(e)?;
        while news > 0 {
            news -= 1;
            let span = join_span(self.prev_span()?, e.span());
            e = Expr::New {
                callee: Box::new(e),
                args: Vec::new(),
                span,
            };
        }
        Ok((e, paren))
    }

    /// Member/call suffixes: `.x`, `[x]`, `?.`, calls, tagged templates.
    /// The optional-chain rules (no assignment target, no tagged-template
    /// tag, no direct `new` operand) are enforced at the use sites.
    fn finish_suffix(&mut self, e: Expr) -> Result<Expr, SyntaxError> {
        self.finish_suffix_inner(e, true)
    }

    /// Member-access suffixes only (`.x`, `[x]`, `?.` chains): no calls and
    /// no tagged templates. Used between a `new` prefix and its argument
    /// list, where a call `(args)` belongs to the innermost `new`.
    fn finish_member_suffix(&mut self, e: Expr) -> Result<Expr, SyntaxError> {
        self.finish_suffix_inner(e, false)
    }

    fn finish_suffix_inner(&mut self, mut e: Expr, calls: bool) -> Result<Expr, SyntaxError> {
        loop {
            // Tagged template: the lexer emits TemplateHead/TemplateTail right
            // after the tag with no trivia in between. Not inside a `${...}`
            // substitution (there the token ends the substitution).
            // (Calls only: a template right after a `new` operand is parsed
            // after the `new` takes its argument list.)
            if calls
                && self.template_subst_depth == 0
                && matches!(
                    self.peek_at(0)?,
                    TokenKind::TemplateHead(_) | TokenKind::TemplateTail(_)
                )
            {
                if is_optional_chain(&e) {
                    return Err(Self::early(
                        "optional chain may not be a tagged-template tag",
                        e.span(),
                    ));
                }
                e = self.finish_tagged(e)?;
                continue;
            }
            match self.peek_at(0)? {
                TokenKind::Dot => {
                    self.bump()?;
                    if matches!(self.peek_at(0)?, TokenKind::PrivateIdent(_)) {
                        let (pname, pspan) = self.take_private_ident()?;
                        self.use_private(&pname, pspan)?;
                        let span = join_span(e.span(), pspan);
                        e = Expr::Member {
                            obj: Box::new(e),
                            prop: Box::new(Expr::PrivateIdent(pname, pspan)),
                            computed: false,
                            optional: false,
                            span,
                        };
                    } else {
                        let (name, span) = self.parse_prop_name()?;
                        let span = join_span(e.span(), span);
                        e = Expr::Member {
                            obj: Box::new(e),
                            prop: Box::new(Expr::Ident(name, span)),
                            computed: false,
                            optional: false,
                            span,
                        };
                    }
                }
                TokenKind::LBracket => {
                    self.bump()?;
                    let prop = self.parse_expr()?;
                    let rb = self.expect(&TokenKind::RBracket, "']'")?;
                    let span = join_span(e.span(), rb);
                    e = Expr::Member {
                        obj: Box::new(e),
                        prop: Box::new(prop),
                        computed: true,
                        optional: false,
                        span,
                    };
                }
                TokenKind::OptChain => {
                    // Distinguish `?.` from `? .` (conditional): the lexer
                    // produces OptChain only for the joined token.
                    self.bump()?;
                    match self.peek_at(0)? {
                        TokenKind::LParen => {
                            // `f?.(args)` — optional call.
                            let args = self.parse_args()?;
                            let span = join_span(e.span(), self.prev_span()?);
                            e = Expr::Call {
                                callee: Box::new(e),
                                args,
                                optional: true,
                                span,
                            };
                        }
                        TokenKind::LBracket => {
                            self.bump()?;
                            let prop = self.parse_expr()?;
                            let rb = self.expect(&TokenKind::RBracket, "']'")?;
                            let span = join_span(e.span(), rb);
                            e = Expr::Member {
                                obj: Box::new(e),
                                prop: Box::new(prop),
                                computed: true,
                                optional: true,
                                span,
                            };
                        }
                        _ => {
                            if matches!(self.peek_at(0)?, TokenKind::PrivateIdent(_)) {
                                let (pname, pspan) = self.take_private_ident()?;
                                self.use_private(&pname, pspan)?;
                                let span = join_span(e.span(), pspan);
                                e = Expr::Member {
                                    obj: Box::new(e),
                                    prop: Box::new(Expr::PrivateIdent(pname, pspan)),
                                    computed: false,
                                    optional: true,
                                    span,
                                };
                            } else {
                                let (name, span) = self.parse_prop_name()?;
                                let span = join_span(e.span(), span);
                                e = Expr::Member {
                                    obj: Box::new(e),
                                    prop: Box::new(Expr::Ident(name, span)),
                                    computed: false,
                                    optional: true,
                                    span,
                                };
                            }
                        }
                    }
                }
                TokenKind::LParen if calls => {
                    // `super(...)` is only valid in a derived constructor.
                    if matches!(e, Expr::Super(_)) && !self.in_derived_ctor {
                        return Err(Self::early(
                            "super() is only valid in a derived class constructor",
                            e.span(),
                        ));
                    }
                    let args = self.parse_args()?;
                    let span = join_span(e.span(), self.prev_span()?);
                    e = Expr::Call {
                        callee: Box::new(e),
                        args,
                        optional: false,
                        span,
                    };
                }
                _ => break,
            }
        }
        Ok(e)
    }

    fn parse_args(&mut self) -> Result<Vec<Expr>, SyntaxError> {
        self.expect(&TokenKind::LParen, "'('")?;
        let mut args = Vec::new();
        while !matches!(self.peek_at(0)?, TokenKind::RParen | TokenKind::Eof) {
            if matches!(self.peek_at(0)?, TokenKind::Ellipsis) {
                let start = self.bump()?;
                let arg = self.parse_assignment()?;
                let span = join_span(start, self.prev_span()?);
                args.push(Expr::Spread(Box::new(arg), span));
            } else {
                args.push(self.parse_assignment()?);
            }
            if matches!(self.peek_at(0)?, TokenKind::Comma) {
                self.bump()?;
                // Trailing comma allowed.
                if matches!(self.peek_at(0)?, TokenKind::RParen) {
                    break;
                }
            } else {
                break;
            }
        }
        self.expect(&TokenKind::RParen, "')'")?;
        Ok(args)
    }

    /// Parse a PrimaryExpression.
    fn parse_primary(&mut self) -> Result<Expr, SyntaxError> {
        // Payload-carrying tokens are taken by value so their `String`s
        // move instead of cloning (twice, in the old code).
        match self.peek_at(0)? {
            TokenKind::BigInt(_) | TokenKind::Str(_) | TokenKind::PrivateIdent(_) => {
                let tok = self.take()?;
                let span = tok.span;
                return match tok.kind {
                    TokenKind::BigInt(raw) => Ok(Expr::Lit(Literal::BigInt(raw), span)),
                    TokenKind::Str(s) => {
                        // Strict-mode octal escape check happens via the raw text.
                        self.check_strict_string_escapes(span)?;
                        Ok(Expr::Lit(Literal::Str(s), span))
                    }
                    TokenKind::PrivateIdent(name) => {
                        // `#x` — only valid in `in` expressions and member access.
                        // Produce the node; validity is checked by the caller.
                        Ok(Expr::PrivateIdent(name, span))
                    }
                    _ => unreachable!("kind checked above"),
                };
            }
            TokenKind::Slash => {
                // Regex literal in expression position: re-lex the `/` and
                // split the raw `/pattern/flags` into its parts.
                self.maybe_regex()?;
                let tok = self.take()?;
                let span = tok.span;
                let raw = match tok.kind {
                    TokenKind::RegexLiteral(raw) => raw,
                    _ => return Err(self.unexpected("a regular expression")),
                };
                let close = raw.rfind('/').unwrap_or(raw.len());
                let pattern = raw[1..close].to_string();
                let flags = raw[close + 1..].to_string();
                return Ok(Expr::Regex {
                    pattern,
                    flags,
                    span,
                });
            }
            _ => {}
        }
        // Clone the kind to avoid holding a borrow across `&mut self` calls.
        // (Cheap here: the `String`-payload kinds are handled above.)
        let kind = self.peek_at(0)?.clone();
        match kind {
            TokenKind::Numeric(v) => {
                self.bump()?;
                Ok(Expr::Lit(Literal::Number(v), self.prev_span()?))
            }
            TokenKind::NumericOctal(v) => {
                self.check_strict_octal()?;
                self.bump()?;
                Ok(Expr::Lit(Literal::Number(v), self.prev_span()?))
            }
            TokenKind::True => {
                let span = self.bump()?;
                Ok(Expr::Lit(Literal::Bool(true), span))
            }
            TokenKind::False => {
                let span = self.bump()?;
                Ok(Expr::Lit(Literal::Bool(false), span))
            }
            TokenKind::Null => {
                let span = self.bump()?;
                Ok(Expr::Lit(Literal::Null, span))
            }
            TokenKind::This => {
                let span = self.bump()?;
                Ok(Expr::This(span))
            }
            TokenKind::Super => {
                let span = self.bump()?;
                if !self.super_allowed() {
                    return Err(Self::early(
                        "super is only valid inside class methods",
                        span,
                    ));
                }
                Ok(Expr::Super(span))
            }
            TokenKind::Ident(_) => {
                let (name, span) = self.parse_ident_name()?;
                self.check_ident_binding(&name, span)?;
                Ok(Expr::Ident(name, span))
            }
            TokenKind::Import => {
                let start = self.bump()?;
                // Dynamic `import(x)`.
                if matches!(self.peek_at(0)?, TokenKind::LParen) {
                    let args = self.parse_args()?;
                    let span = join_span(start, self.prev_span()?);
                    return Ok(Expr::Call {
                        callee: Box::new(Expr::Ident("import".to_string(), span)),
                        args,
                        optional: false,
                        span,
                    });
                }
                // `import.meta`.
                if matches!(self.peek_at(0)?, TokenKind::Dot) {
                    self.bump()?;
                    let (name, name_span) = self.parse_prop_name()?;
                    if name != "meta" {
                        return Err(Self::early("expected 'meta' after 'import.'", name_span));
                    }
                    if !self.is_module {
                        return Err(Self::early("import.meta is only allowed in modules", start));
                    }
                    let span = join_span(start, name_span);
                    return Ok(Expr::ImportMeta(span));
                }
                Err(self.unexpected("an expression"))
            }
            TokenKind::MetaPropertyNewTarget => {
                let span = self.bump()?;
                self.finish_new_target(span)
            }
            TokenKind::TemplateHead(_) | TokenKind::TemplateTail(_) => self.parse_template(false),
            TokenKind::LBracket => Ok(self.parse_array()?),
            TokenKind::LBrace => Ok(self.parse_object()?),
            TokenKind::LParen => self.parse_paren_or_arrow(),
            TokenKind::Function => {
                let start = self.bump()?;
                let (name, params, body, is_generator, strict) =
                    self.parse_function_common(false, false)?;
                let span = join_span(start, self.prev_span()?);
                Ok(Expr::Function {
                    name,
                    params,
                    body,
                    is_async: false,
                    is_generator,
                    strict,
                    span,
                })
            }
            TokenKind::Class => {
                let start = self.peek_span()?;
                let (name, superclass, body) = self.parse_class_common(start, false)?;
                let span = join_span(start, self.prev_span()?);
                Ok(Expr::Class {
                    name,
                    superclass,
                    body,
                    span,
                })
            }
            TokenKind::Async => self.parse_async_primary(),
            // `yield` as an operator is handled in parse_assignment (it is
            // an AssignmentExpression, not a primary). Here it can only be
            // an identifier.
            TokenKind::Yield => {
                let span = self.bump()?;
                self.check_ident_binding("yield", span)?;
                Ok(Expr::Ident("yield".to_string(), span))
            }
            TokenKind::Await if self.await_allowed() => {
                // `await` in primary position (e.g. arrow body start).
                let (e, _) = self.parse_unary_p()?;
                Ok(e)
            }
            TokenKind::Await => {
                // `await` as an identifier (sloppy scripts).
                let span = self.bump()?;
                self.check_ident_binding("await", span)?;
                Ok(Expr::Ident("await".to_string(), span))
            }
            // Contextual keywords (`target`, `of`, `from`, `as`, `let`,
            // `static`, `get`, `set`) are ordinary identifiers in primary
            // position; the dedicated arms above keep their special
            // meanings (`async` arrows, `await`/`yield` expressions).
            kind if contextual_name(&kind).is_some() => {
                let name = contextual_name(&kind)
                    .expect("guard ensures Some")
                    .to_string();
                let span = self.bump()?;
                self.check_ident_binding(&name, span)?;
                Ok(Expr::Ident(name, span))
            }
            _ => Err(self.unexpected("an expression")),
        }
    }

    /// `async ...` in primary position: async function expression, async
    /// arrow, or the identifier `async`.
    fn parse_async_primary(&mut self) -> Result<Expr, SyntaxError> {
        let start = self.peek_span()?;
        // `async function` — no line break allowed.
        if matches!(self.peek_at(1)?, TokenKind::Function) && !self.nl_before(1)? {
            self.bump()?; // async
            self.bump()?; // function
            let (name, params, body, is_generator, strict) =
                self.parse_function_common(true, false)?;
            let span = join_span(start, self.prev_span()?);
            return Ok(Expr::Function {
                name,
                params,
                body,
                is_async: true,
                is_generator,
                strict,
                span,
            });
        }
        // `async (...)` — maybe an async arrow.
        if matches!(self.peek_at(1)?, TokenKind::LParen) {
            let cp = self.checkpoint();
            self.bump()?; // async
            match self.try_parse_arrow_from_paren(start, true) {
                Ok(Some(e)) => {
                    self.release(cp);
                    return Ok(e);
                }
                Ok(None) => self.rewind_to(cp),
                Err(e) => {
                    if matches!(e, SyntaxError::Lex { .. }) {
                        self.release(cp);
                        return Err(e);
                    }
                    self.rewind_to(cp);
                }
            }
        }
        // `async x =>` — maybe an async arrow with a single identifier param.
        if is_identifierish(self.peek_at(1)?) && !self.nl_before(1)? {
            let cp = self.checkpoint();
            self.bump()?; // async
            let (pname, pspan) = self.parse_ident_name()?;
            if !self.nl_before(0)? && matches!(self.peek_at(0)?, TokenKind::Arrow) {
                self.bump()?; // =>
                if pname == "await" {
                    self.release(cp);
                    return Err(Self::early("await may not be bound here", pspan));
                }
                self.release(cp);
                self.check_binding_name(&pname, pspan)?;
                let params = vec![Param {
                    pat: Pat::Ident(pname, pspan),
                    span: pspan,
                }];
                return self.finish_arrow_body(start, params, true);
            }
            self.rewind_to(cp);
        }
        // Otherwise `async` is an identifier.
        let (name, span) = self.parse_ident_name()?;
        self.check_ident_binding(&name, span)?;
        Ok(Expr::Ident(name, span))
    }

    fn parse_yield(&mut self) -> Result<Expr, SyntaxError> {
        let start = self.bump()?; // yield
        let delegate = matches!(self.peek_at(0)?, TokenKind::Star);
        if delegate {
            self.bump()?;
        }
        let arg = if self.nl_before(0)?
            || matches!(
                self.peek_at(0)?,
                TokenKind::Semicolon
                    | TokenKind::RBrace
                    | TokenKind::RParen
                    | TokenKind::RBracket
                    | TokenKind::Comma
                    | TokenKind::Eof
            ) {
            None
        } else {
            Some(Box::new(self.parse_assignment()?))
        };
        let span = join_span(start, self.prev_span()?);
        Ok(Expr::Yield {
            arg,
            delegate,
            span,
        })
    }

    /// `new.target` — valid when `nt_depth > 0`: inside a non-arrow
    /// function (arrows inherit), class field initializer, or static block.
    fn finish_new_target(&mut self, span: Span) -> Result<Expr, SyntaxError> {
        if self.nt_depth == 0 {
            return Err(Self::early(
                "new.target is only valid inside functions or classes",
                span,
            ));
        }
        Ok(Expr::NewTarget(span))
    }
}

/// `true` if the expression is (or ends in) an optional chain.
///
/// Iterative over the chain spine: the parser builds left-deep member/call
/// chains in a loop without counting them toward its depth guard, so a
/// 3000-deep `f.bind(null).bind(null)…` chain is a legal AST. The recursive
/// walk that used to live here overflowed the host stack on such chains (an
/// uncatchable process abort); the loop keeps the scan O(1) in stack.
fn is_optional_chain(e: &Expr) -> bool {
    let mut cur = e;
    loop {
        match cur {
            Expr::Call {
                callee, optional, ..
            } => {
                if *optional {
                    return true;
                }
                cur = callee.as_ref();
            }
            Expr::Member { obj, optional, .. } => {
                if *optional {
                    return true;
                }
                cur = obj.as_ref();
            }
            _ => return false,
        }
    }
}

// ---------------------------------------------------------------------------
// Arrays, objects, templates
// ---------------------------------------------------------------------------

impl<'a> Parser<'a> {
    fn parse_array(&mut self) -> Result<Expr, SyntaxError> {
        self.guarded_cost(HEAVY_LITERAL_COST, |s| s.parse_array_inner())
    }

    fn parse_array_inner(&mut self) -> Result<Expr, SyntaxError> {
        let start = self.expect(&TokenKind::LBracket, "'['")?;
        let mut elems = Vec::new();
        while !matches!(self.peek_at(0)?, TokenKind::RBracket | TokenKind::Eof) {
            if matches!(self.peek_at(0)?, TokenKind::Comma) {
                self.bump()?;
                elems.push(None);
                continue;
            }
            if matches!(self.peek_at(0)?, TokenKind::Ellipsis) {
                let s = self.bump()?;
                let arg = self.parse_assignment()?;
                let span = join_span(s, self.prev_span()?);
                elems.push(Some(Expr::Spread(Box::new(arg), span)));
            } else if let Some(e) = self.try_parse_array_cover()? {
                elems.push(Some(e));
            } else {
                elems.push(Some(self.parse_assignment()?));
            }
            if matches!(self.peek_at(0)?, TokenKind::Comma) {
                self.bump()?;
            } else {
                break;
            }
        }
        let end = self.expect(&TokenKind::RBracket, "']'")?;
        Ok(Expr::Array(elems, join_span(start, end)))
    }

    /// Try to parse an array element as a `[pat = init]` cover initializer.
    /// Returns `Ok(None)` (with the position restored) when the element is
    /// not a top-level `pattern =`, so the caller can parse it as a normal
    /// expression. Lex errors always propagate.
    fn try_parse_array_cover(&mut self) -> Result<Option<Expr>, SyntaxError> {
        let cp = self.checkpoint();
        let start = self.peek_span()?;
        let pat = match self.parse_pat_primary() {
            Ok(p) => p,
            Err(e) => {
                if matches!(e, SyntaxError::Lex { .. }) {
                    self.release(cp);
                    return Err(e);
                }
                self.rewind_to(cp);
                return Ok(None);
            }
        };
        if !matches!(self.peek_at(0)?, TokenKind::Eq) {
            self.rewind_to(cp);
            return Ok(None);
        }
        self.release(cp);
        if !self.cover_ok && !self.in_paren_cover && !self.in_for_head {
            return Err(Self::early("unexpected '=' in array literal", start));
        }
        self.pending_cover = true;
        self.bump()?; // =
        // The initializer parses as a fresh assignment: the outer
        // `pending_cover` marks *this* element's initializer as unvalidated
        // and must not leak into the initializer's own tail check
        // (`check_no_cover_init`), or `[a = 1] = rhs` would die on the
        // inner `1`. Save/restore around the nested parse.
        let outer_cover = std::mem::replace(&mut self.pending_cover, false);
        let init = self.parse_assignment();
        self.pending_cover = outer_cover;
        let init = init?;
        let span = join_span(start, self.prev_span()?);
        Ok(Some(Expr::Assign {
            op: AssignOp::Assign,
            left: Box::new(pat_to_expr(&pat)),
            right: Box::new(init),
            span,
        }))
    }

    fn parse_object(&mut self) -> Result<Expr, SyntaxError> {
        self.guarded_cost(HEAVY_LITERAL_COST, |s| s.parse_object_inner())
    }

    fn parse_object_inner(&mut self) -> Result<Expr, SyntaxError> {
        let start = self.expect(&TokenKind::LBrace, "'{'")?;
        let mut props = Vec::new();
        let mut seen_proto: Option<Span> = None;
        while !matches!(self.peek_at(0)?, TokenKind::RBrace | TokenKind::Eof) {
            let prop = self.parse_prop_def(&mut seen_proto)?;
            props.push(prop);
            if matches!(self.peek_at(0)?, TokenKind::Comma) {
                self.bump()?;
            } else {
                break;
            }
        }
        let end = self.expect(&TokenKind::RBrace, "'}'")?;
        Ok(Expr::Object(props, join_span(start, end)))
    }

    /// Parse one object-literal property. `seen_proto` tracks a duplicate
    /// data `__proto__` property (an early error).
    fn parse_prop_def(&mut self, seen_proto: &mut Option<Span>) -> Result<PropDef, SyntaxError> {
        let start = self.peek_span()?;
        // `...x`
        if matches!(self.peek_at(0)?, TokenKind::Ellipsis) {
            self.bump()?;
            let arg = self.parse_assignment()?;
            let span = join_span(start, self.prev_span()?);
            return Ok(PropDef::Spread(arg, span));
        }
        // `async` / `*` / `get` / `set` modifiers.
        let mut is_async = false;
        let mut is_generator = false;
        if matches!(self.peek_at(0)?, TokenKind::Async)
            && !self.nl_before(1)?
            && !matches!(self.peek_at(1)?, TokenKind::Comma | TokenKind::RBrace)
        {
            // `async` followed by a key starts an async method — unless the
            // key itself can't start a method (e.g. `async: 1` shorthand?
            // no: `async: 1` is a KeyValue with key `async`). Distinguish:
            // `async` + `(` is a shorthand call? No — `async()` is a method
            // named async... Actually `{async() {}}` is a method named
            // "async". `{async m() {}}` is an async method.
            // Rule: if token after `async` can start a property key and the
            // token after that is `(` or the key is followed by `(`,
            // it's an async method. `{async, ...}` / `{async: 1}` /
            // `{async = 1}` are shorthand/KeyValue/cover-init.
            let after = self.peek_at(1)?;
            let looks_like_method_key = match after {
                TokenKind::LParen => false, // `async()` — method named async
                _ => self.peek_is_method_key(1)?,
            };
            if looks_like_method_key {
                self.bump()?; // async
                is_async = true;
            }
        }
        if matches!(self.peek_at(0)?, TokenKind::Star) {
            self.bump()?;
            is_generator = true;
        }
        let key = self.parse_prop_key()?;
        // `get x()` / `set x(v)` — only when not async/generator and the key
        // is the identifier `get`/`set` followed by a key.
        if !is_async
            && !is_generator
            && matches!(key, PropKey::Ident(ref n, _) if n == "get" || n == "set")
            && self.peek_starts_key()?
        {
            let name = match key {
                PropKey::Ident(n, _) => n,
                _ => unreachable!(),
            };
            let key2 = self.parse_prop_key()?;
            let span = join_span(start, self.peek_span()?);
            if name == "get" {
                let save_m = self.in_method;
                self.in_method = true;
                let (params, body, strict) = self.parse_function_tail(false, false)?;
                self.in_method = save_m;
                if !params.is_empty() {
                    return Err(Self::early("getter must have no parameters", span));
                }
                let span = join_span(span, self.prev_span()?);
                return Ok(PropDef::Getter {
                    key: key2,
                    body,
                    strict,
                    span,
                });
            } else {
                let save_m = self.in_method;
                self.in_method = true;
                // Parse `(param)` in the setter's own context.
                let save = self.save_ctx();
                self.func_depth += 1;
                self.nt_depth += 1;
                self.param_depth += 1;
                let params = self.parse_params();
                self.param_depth -= 1;
                let params = params?;
                if params.len() != 1 {
                    self.restore_ctx(save);
                    self.in_method = save_m;
                    return Err(Self::early("setter must have exactly one parameter", span));
                }
                let param = params.into_iter().next().expect("one param");
                let body = self.parse_function_body(std::slice::from_ref(&param))?;
                let strict = self.strict;
                self.validate_params(std::slice::from_ref(&param), strict)?;
                self.restore_ctx(save);
                self.in_method = save_m;
                let span = join_span(span, self.prev_span()?);
                return Ok(PropDef::Setter {
                    key: key2,
                    param,
                    body,
                    strict,
                    span,
                });
            }
        }
        // Method: `key(...)`.
        if matches!(self.peek_at(0)?, TokenKind::LParen) {
            let save_m = self.in_method;
            self.in_method = true;
            let (params, body, strict) = self.parse_function_tail(is_async, is_generator)?;
            self.in_method = save_m;
            let span = join_span(start, self.prev_span()?);
            return Ok(PropDef::Method {
                key,
                params,
                body,
                is_async,
                is_generator,
                strict,
                span,
            });
        }
        // `key: value`
        if matches!(self.peek_at(0)?, TokenKind::Colon) {
            self.bump()?;
            let value = self.parse_assignment()?;
            let span = join_span(start, self.prev_span()?);
            // Duplicate data `__proto__` check.
            let is_proto = match &key {
                PropKey::Str(s) => s.as_wtf8() == Wtf8::from_str("__proto__"),
                PropKey::Ident(s, _) => s == "__proto__",
                _ => false,
            };
            if is_proto {
                if let Some(prev) = *seen_proto {
                    let _ = prev;
                    return Err(Self::early(
                        "duplicate __proto__ property in object literal",
                        span,
                    ));
                }
                *seen_proto = Some(span);
            }
            return Ok(PropDef::KeyValue { key, value, span });
        }
        // Shorthand: `{a}` / `{a = 1}`.
        match key {
            PropKey::Ident(name, key_span) => {
                let span = join_span(start, self.prev_span()?);
                if matches!(self.peek_at(0)?, TokenKind::Eq) {
                    // Shorthand with default: cover grammar. In a pure
                    // expression this is an error; as a destructuring
                    // target it becomes `a = init`.
                    if !self.cover_ok && !self.in_paren_cover && !self.in_for_head {
                        return Err(Self::early(
                            "shorthand property may not have an initializer here",
                            span,
                        ));
                    }
                    self.pending_cover = true;
                    self.bump()?;
                    // Fresh assignment context for the initializer: see the
                    // array-literal cover path above.
                    let outer_cover = std::mem::replace(&mut self.pending_cover, false);
                    let init = self.parse_assignment();
                    self.pending_cover = outer_cover;
                    let init = init?;
                    // Represent as `{a: a = init}`; the pattern converter
                    // treats it identically to a shorthand default.
                    let init_span = join_span(key_span, self.prev_span()?);
                    let value = Expr::Assign {
                        op: AssignOp::Assign,
                        left: Box::new(Expr::Ident(name.clone(), key_span)),
                        right: Box::new(init),
                        span: init_span,
                    };
                    return Ok(PropDef::KeyValue {
                        key: PropKey::Ident(name, key_span),
                        value,
                        span,
                    });
                }
                Ok(PropDef::Shorthand { name, span })
            }
            _ => Err(Self::early(
                "shorthand property must be an identifier",
                start,
            )),
        }
    }

    /// Parse a property key: identifier, string, number, or `[expr]`.
    fn parse_prop_key(&mut self) -> Result<PropKey, SyntaxError> {
        let span = self.peek_span()?;
        match self.peek_at(0)? {
            TokenKind::LBracket => {
                self.bump()?;
                let e = self.parse_assignment()?;
                self.expect(&TokenKind::RBracket, "']'")?;
                Ok(PropKey::Computed(e))
            }
            TokenKind::Numeric(v) => {
                let v = *v;
                self.bump()?;
                Ok(PropKey::Num(v, span))
            }
            TokenKind::NumericOctal(v) => {
                let v = *v;
                self.check_strict_octal()?;
                self.bump()?;
                Ok(PropKey::Num(v, span))
            }
            TokenKind::Str(_) => {
                let (s, span) = self.take_str()?;
                self.check_strict_string_escapes(span)?;
                Ok(PropKey::Str(s))
            }
            _ => {
                let (name, span) = self.parse_ident_name()?;
                // `let`/`yield`/`await` as keys are fine via parse_ident_name.
                Ok(PropKey::Ident(name, span))
            }
        }
    }

    /// `true` if the token at `offset` can start a property key.
    fn peek_is_method_key(&mut self, offset: usize) -> Result<bool, SyntaxError> {
        Ok(match self.peek_at(offset)? {
            TokenKind::LBracket
            | TokenKind::Str(_)
            | TokenKind::Numeric(_)
            | TokenKind::NumericOctal(_)
            | TokenKind::BigInt(_) => true,
            k => is_identifierish(k) || keyword_text(k).is_some(),
        })
    }

    /// `true` if the current token can start a property key (used after
    /// `get`/`set` to decide accessor vs. method named `get`).
    fn peek_starts_key(&mut self) -> Result<bool, SyntaxError> {
        self.peek_is_method_key(0)
    }

    // -- templates ------------------------------------------------------------------

    /// Parse a template literal starting at a TemplateHead/TemplateTail
    /// token. `tagged` selects error behavior for invalid escapes:
    /// untagged templates reject them, tagged templates keep raw.
    fn parse_template(&mut self, tagged: bool) -> Result<Expr, SyntaxError> {
        let start = self.peek_span()?;
        let mut parts = Vec::new();
        loop {
            match self.peek_at(0)? {
                TokenKind::TemplateHead(_)
                | TokenKind::TemplateMiddle(_)
                | TokenKind::TemplateTail(_) => {}
                _ => return Err(self.unexpected("a template chunk")),
            }
            let tok = self.take()?;
            let span = tok.span;
            let (cooked, is_tail) = match tok.kind {
                TokenKind::TemplateHead(c) => (c, false),
                TokenKind::TemplateMiddle(c) => (c, false),
                TokenKind::TemplateTail(c) => (c, true),
                _ => unreachable!("kind checked above"),
            };
            let raw = self.template_raw_of(span);
            let cooked_opt = if template_has_bad_escape(&raw) {
                if !tagged {
                    return Err(Self::early(
                        "invalid escape sequence in template literal",
                        span,
                    ));
                }
                None
            } else {
                Some(cooked)
            };
            parts.push(TemplatePart::Str {
                cooked: cooked_opt,
                raw,
            });
            if is_tail {
                break;
            }
            // `${ expr }`
            self.template_subst_depth += 1;
            let e = self.parse_expr()?;
            self.template_subst_depth -= 1;
            parts.push(TemplatePart::Expr(e));
            // The next token must be TemplateMiddle/TemplateTail.
            match self.peek_at(0)? {
                TokenKind::TemplateMiddle(_) | TokenKind::TemplateTail(_) => {}
                _ => return Err(self.unexpected("'}'")),
            }
        }
        let span = join_span(start, self.prev_span()?);
        Ok(Expr::Template { parts, span })
    }

    /// Parse a tagged template after the tag expression.
    fn finish_tagged(&mut self, tag: Expr) -> Result<Expr, SyntaxError> {
        let start = tag.span();
        // Re-parse the head as a tagged template (bad escapes allowed).
        let mut tpl = self.parse_template(true)?;
        // `Expr` has a manual `Drop` (iterative spine freeing), so its
        // fields cannot move out of a by-value pattern: take them through
        // `&mut` instead (no clones).
        let (parts, span) = match &mut tpl {
            Expr::Template { parts, span } => (std::mem::take(parts), join_span(start, *span)),
            _ => unreachable!("parse_template(true) always yields a Template"),
        };
        Ok(Expr::TaggedTemplate {
            tag: Box::new(tag),
            parts,
            span,
        })
    }

    /// Raw text of a template chunk: source between the opening delimiter
    /// (backtick or `}`) and the closing delimiter (`${` or backtick).
    /// The lexer's chunk span runs from the opening delimiter through the
    /// closing delimiter, so strip them by inspecting the source bytes
    /// (all delimiters are ASCII).
    fn template_raw_of(&self, span: Span) -> String {
        let bytes = self.src.as_bytes();
        let start = span.start as usize;
        let end = span.end as usize;
        if start >= end || end > bytes.len() {
            return String::new();
        }
        let inner_start = start + 1; // backtick or `}`
        let inner_end = if end >= start + 2 && bytes[end - 2] == b'$' && bytes[end - 1] == b'{' {
            end - 2 // `${`
        } else {
            end - 1 // closing backtick
        };
        if inner_start > inner_end {
            return String::new();
        }
        self.src[inner_start..inner_end].to_string()
    }
}

/// `true` if a raw template chunk contains an invalid escape sequence.
/// The lexer keeps the source intact, so scan for `\` followed by a char
/// that can't start a valid escape in the cooked payload... Conservative:
/// compare is impossible without the cooked mapping, so detect the known
/// bad patterns: `\u` not followed by valid hex/`{...}`, `\x` not
/// followed by 2 hex digits. `\8`, `\9`, `\1`-style octals are legal in
/// sloppy untagged templates, so only truly-invalid escapes count.
fn template_has_bad_escape(raw: &str) -> bool {
    let b = raw.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\' {
            i += 1;
            if i >= b.len() {
                return true;
            }
            match b[i] {
                b'x' => {
                    if i + 2 >= b.len()
                        || !b[i + 1].is_ascii_hexdigit()
                        || !b[i + 2].is_ascii_hexdigit()
                    {
                        return true;
                    }
                    i += 3;
                }
                b'u' => {
                    if i + 1 < b.len() && b[i + 1] == b'{' {
                        let mut j = i + 2;
                        while j < b.len() && b[j] != b'}' {
                            if !b[j].is_ascii_hexdigit() {
                                return true;
                            }
                            j += 1;
                        }
                        if j >= b.len() || j == i + 2 {
                            return true;
                        }
                        // Check the code point is in range (<= 0x10FFFF).
                        let hex_str = &raw[i + 2..j];
                        if let Ok(cp) = u32::from_str_radix(hex_str, 16) {
                            if cp > 0x10FFFF {
                                return true;
                            }
                        } else {
                            return true;
                        }
                        i = j + 1;
                    } else {
                        if i + 4 >= b.len()
                            || !b[i + 1..i + 5].iter().all(|c| c.is_ascii_hexdigit())
                        {
                            return true;
                        }
                        i += 5;
                    }
                }
                _ => {
                    i += 1;
                }
            }
        } else {
            i += 1;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Join two spans into their minimal covering span.
fn join_span(a: Span, b: Span) -> Span {
    Span {
        start: a.start.min(b.start),
        end: a.end.max(b.end),
    }
}

/// Human-readable token name for error messages.
fn kind_name(kind: &TokenKind) -> String {
    match kind {
        TokenKind::Eof => "end of input".to_string(),
        TokenKind::Numeric(_) | TokenKind::NumericOctal(_) => "a number".to_string(),
        TokenKind::BigInt(_) => "a bigint".to_string(),
        TokenKind::Str(_) => "a string".to_string(),
        TokenKind::TemplateHead(_) | TokenKind::TemplateMiddle(_) | TokenKind::TemplateTail(_) => {
            "a template".to_string()
        }
        TokenKind::RegexLiteral(_) => "a regular expression".to_string(),
        TokenKind::Ident(s) => format!("identifier '{s}'"),
        TokenKind::PrivateIdent(s) => format!("private name '#{s}'"),
        _ => {
            if let Some(n) = contextual_name(kind).or_else(|| keyword_text(kind)) {
                format!("'{n}'")
            } else {
                // Punctuation: map the common ones.
                let p = match kind {
                    TokenKind::LBrace => "{",
                    TokenKind::RBrace => "}",
                    TokenKind::LParen => "(",
                    TokenKind::RParen => ")",
                    TokenKind::LBracket => "[",
                    TokenKind::RBracket => "]",
                    TokenKind::Dot => ".",
                    TokenKind::Semicolon => ";",
                    TokenKind::Comma => ",",
                    TokenKind::Colon => ":",
                    TokenKind::Question => "?",
                    TokenKind::Arrow => "=>",
                    TokenKind::Ellipsis => "...",
                    TokenKind::OptChain => "?.",
                    _ => "a token",
                };
                format!("'{p}'")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Identifier validation
// ---------------------------------------------------------------------------

impl<'a> Parser<'a> {
    /// Validate an identifier used as a value reference.
    fn check_ident_binding(&mut self, name: &str, span: Span) -> Result<(), SyntaxError> {
        if name == "let" && self.strict {
            return Err(Self::early("let is reserved in strict mode", span));
        }
        if name == "yield" && (self.strict || self.in_generator) {
            return Err(Self::early(
                "yield may not be used as an identifier here",
                span,
            ));
        }
        if name == "await" && (self.in_async || self.is_module) {
            return Err(Self::early(
                "await may not be used as an identifier here",
                span,
            ));
        }
        Ok(())
    }

    /// Validate an identifier used as a binding name (declaration, param,
    /// catch, arrow param).
    fn check_binding_name(&mut self, name: &str, span: Span) -> Result<(), SyntaxError> {
        if self.strict {
            if name == "eval" || name == "arguments" {
                return Err(Self::early(
                    "eval/arguments may not be bound in strict mode",
                    span,
                ));
            }
            if name == "let" {
                return Err(Self::early("let is reserved in strict mode", span));
            }
            if name == "yield" {
                return Err(Self::early("yield is reserved in strict mode", span));
            }
        }
        if name == "await" && (self.in_async || self.is_module) {
            return Err(Self::early("await may not be bound here", span));
        }
        if name == "yield" && self.in_generator {
            return Err(Self::early(
                "yield may not be bound inside a generator",
                span,
            ));
        }
        Ok(())
    }

    /// Validate every name bound by a pattern (var/let/const, catch, params).
    fn check_pat_binding(&mut self, pat: &Pat) -> Result<(), SyntaxError> {
        for (name, span) in pat_bound_names(pat) {
            self.check_binding_name(&name, span)?;
        }
        Ok(())
    }

    /// Strict-mode check for octal escapes in a string literal's raw text.
    /// `\0` is allowed only when not followed by an octal digit; `\1`-`\7`,
    /// `\8`, `\9` (and `\0` followed by a digit) are errors in strict mode.
    fn check_strict_string_escapes(&mut self, span: Span) -> Result<(), SyntaxError> {
        if !self.strict {
            return Ok(());
        }
        let raw = self.raw_of_span(span);
        let b = raw.as_bytes();
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'\\' && i + 1 < b.len() {
                let c = b[i + 1];
                if c == b'0' {
                    // `\0` followed by 0-9 is an error in strict mode.
                    if i + 2 < b.len() && b[i + 2].is_ascii_digit() {
                        return Err(Self::early(
                            "octal escape is not allowed in strict mode",
                            span,
                        ));
                    }
                    i += 2;
                } else if matches!(c, b'1'..=b'9') {
                    return Err(Self::early(
                        "octal escape is not allowed in strict mode",
                        span,
                    ));
                } else {
                    i += 2;
                }
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    /// Strict-mode check for a legacy octal / octal-like numeric token
    /// (`NumericOctal`). The caller has peeked such a token; reject it in
    /// strict mode (spec Annex B.1.1).
    fn check_strict_octal(&mut self) -> Result<(), SyntaxError> {
        if self.strict {
            let span = self.peek_span()?;
            return Err(Self::early(
                "legacy octal literals are not allowed in strict mode",
                span,
            ));
        }
        Ok(())
    }
}

/// `true` iff every parameter is a plain identifier: no defaults,
/// destructuring, or rest. (Spec: `IsSimpleParameterList`.)
fn is_simple_params(params: &[Param]) -> bool {
    params.iter().all(|p| matches!(p.pat, Pat::Ident(..)))
}

/// All identifier names bound by a pattern, with spans.
fn pat_bound_names(pat: &Pat) -> Vec<(String, Span)> {
    let mut out = Vec::new();
    fn walk(pat: &Pat, out: &mut Vec<(String, Span)>) {
        match pat {
            Pat::Ident(n, s) => out.push((n.clone(), *s)),
            Pat::Array(elems, _) => {
                for e in elems.iter().flatten() {
                    let ArrayPatElem::Pat(p) = e;
                    walk(p, out);
                }
            }
            Pat::Object(props, _) => {
                for p in props {
                    match p {
                        ObjectPatProp::KeyValue { value, .. } => walk(value, out),
                        ObjectPatProp::Shorthand { name, span } => out.push((name.clone(), *span)),
                        ObjectPatProp::Rest(p, _) => walk(p, out),
                    }
                }
            }
            Pat::Rest(p, _) => walk(p, out),
            Pat::Assign(p, _, _) => walk(p, out),
            // Member targets bind nothing.
            Pat::Member(..) => {}
        }
    }
    walk(pat, &mut out);
    out
}

/// `true` if a pattern binds `eval` or `arguments`.
fn pat_is_eval_or_arguments(pat: &Pat) -> bool {
    pat_bound_names(pat)
        .iter()
        .any(|(n, _)| n == "eval" || n == "arguments")
}

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

impl<'a> Parser<'a> {
    /// Parse a BindingPattern (with defaults).
    ///
    /// Depth-guarded: patterns nest (`[[[[...]]]]`, `{a: {b: ...}}`), and
    /// only this entry point recurses, so one guard covers the whole
    /// pattern grammar instead of each call site.
    fn parse_pat(&mut self) -> Result<Pat, SyntaxError> {
        self.guarded(|s| {
            let start = s.peek_span()?;
            let mut pat = s.parse_pat_primary()?;
            if matches!(s.peek_at(0)?, TokenKind::Eq) {
                s.bump()?;
                let init = s.parse_assignment()?;
                let span = join_span(start, s.prev_span()?);
                pat = Pat::Assign(Box::new(pat), Box::new(init), span);
            }
            Ok(pat)
        })
    }

    /// Parse a BindingPattern without consuming a top-level `= default`
    /// (used for array-cover detection).
    fn parse_pat_primary(&mut self) -> Result<Pat, SyntaxError> {
        match self.peek_at(0)? {
            TokenKind::LBracket => self.parse_array_pat(),
            TokenKind::LBrace => self.parse_object_pat(),
            _ => {
                let (name, span) = self.parse_ident_name()?;
                Ok(Pat::Ident(name, span))
            }
        }
    }

    fn parse_array_pat(&mut self) -> Result<Pat, SyntaxError> {
        self.guarded_cost(HEAVY_LITERAL_COST, |s| s.parse_array_pat_inner())
    }

    fn parse_array_pat_inner(&mut self) -> Result<Pat, SyntaxError> {
        let start = self.expect(&TokenKind::LBracket, "'['")?;
        let mut elems: Vec<Option<ArrayPatElem>> = Vec::new();
        while !matches!(self.peek_at(0)?, TokenKind::RBracket | TokenKind::Eof) {
            if matches!(self.peek_at(0)?, TokenKind::Comma) {
                self.bump()?;
                elems.push(None);
                continue;
            }
            let estart = self.peek_span()?;
            if matches!(self.peek_at(0)?, TokenKind::Ellipsis) {
                self.bump()?;
                let inner = self.parse_pat()?;
                if matches!(inner, Pat::Assign(..)) {
                    return Err(Self::early(
                        "rest element may not have an initializer",
                        estart,
                    ));
                }
                let span = join_span(estart, self.prev_span()?);
                elems.push(Some(ArrayPatElem::Pat(Pat::Rest(Box::new(inner), span))));
            } else {
                let p = self.parse_pat()?;
                elems.push(Some(ArrayPatElem::Pat(p)));
            }
            if matches!(self.peek_at(0)?, TokenKind::Comma) {
                self.bump()?;
            } else {
                break;
            }
        }
        let end = self.expect(&TokenKind::RBracket, "']'")?;
        Ok(Pat::Array(elems, join_span(start, end)))
    }

    fn parse_object_pat(&mut self) -> Result<Pat, SyntaxError> {
        self.guarded_cost(HEAVY_LITERAL_COST, |s| s.parse_object_pat_inner())
    }

    fn parse_object_pat_inner(&mut self) -> Result<Pat, SyntaxError> {
        let start = self.expect(&TokenKind::LBrace, "'{'")?;
        let mut props = Vec::new();
        while !matches!(self.peek_at(0)?, TokenKind::RBrace | TokenKind::Eof) {
            let pstart = self.peek_span()?;
            if matches!(self.peek_at(0)?, TokenKind::Ellipsis) {
                self.bump()?;
                let inner = self.parse_pat()?;
                if matches!(inner, Pat::Assign(..)) {
                    return Err(Self::early(
                        "rest property may not have an initializer",
                        pstart,
                    ));
                }
                let span = join_span(pstart, self.prev_span()?);
                props.push(ObjectPatProp::Rest(Box::new(inner), span));
            } else {
                let key = self.parse_prop_key()?;
                if matches!(self.peek_at(0)?, TokenKind::Colon) {
                    self.bump()?;
                    let value = self.parse_pat()?;
                    let span = join_span(pstart, self.prev_span()?);
                    props.push(ObjectPatProp::KeyValue { key, value, span });
                } else {
                    // Shorthand: `{a}` / `{a = 1}`.
                    match key {
                        PropKey::Ident(name, kspan) => {
                            if matches!(self.peek_at(0)?, TokenKind::Eq) {
                                self.bump()?;
                                let init = self.parse_assignment()?;
                                let span = join_span(pstart, self.prev_span()?);
                                let aspan = join_span(kspan, self.prev_span()?);
                                props.push(ObjectPatProp::KeyValue {
                                    key: PropKey::Ident(name.clone(), kspan),
                                    value: Pat::Assign(
                                        Box::new(Pat::Ident(name, kspan)),
                                        Box::new(init),
                                        aspan,
                                    ),
                                    span,
                                });
                            } else {
                                let span = join_span(pstart, self.prev_span()?);
                                props.push(ObjectPatProp::Shorthand { name, span });
                            }
                        }
                        _ => {
                            return Err(Self::early(
                                "shorthand property must be an identifier",
                                pstart,
                            ));
                        }
                    }
                }
            }
            if matches!(self.peek_at(0)?, TokenKind::Comma) {
                self.bump()?;
            } else {
                break;
            }
        }
        let end = self.expect(&TokenKind::RBrace, "'}'")?;
        Ok(Pat::Object(props, join_span(start, end)))
    }
}

/// Convert a pattern back to an expression (inverse of [`expr_to_pat`]
/// for the pattern subset).
fn pat_to_expr(pat: &Pat) -> Expr {
    match pat {
        Pat::Ident(n, s) => Expr::Ident(n.clone(), *s),
        Pat::Array(elems, s) => Expr::Array(
            elems
                .iter()
                .map(|e| {
                    e.as_ref().map(|a| {
                        let ArrayPatElem::Pat(p) = a;
                        pat_to_expr(p)
                    })
                })
                .collect(),
            *s,
        ),
        Pat::Object(props, s) => Expr::Object(
            props
                .iter()
                .map(|p| match p {
                    ObjectPatProp::KeyValue { key, value, span } => PropDef::KeyValue {
                        key: key.clone(),
                        value: pat_to_expr(value),
                        span: *span,
                    },
                    ObjectPatProp::Shorthand { name, span } => PropDef::Shorthand {
                        name: name.clone(),
                        span: *span,
                    },
                    ObjectPatProp::Rest(p, span) => PropDef::Spread(pat_to_expr(p), *span),
                })
                .collect(),
            *s,
        ),
        Pat::Rest(p, s) => Expr::Spread(Box::new(pat_to_expr(p)), *s),
        Pat::Assign(p, init, s) => Expr::Assign {
            op: AssignOp::Assign,
            left: Box::new(pat_to_expr(p)),
            right: init.clone(),
            span: *s,
        },
        // A member target already is an expression.
        Pat::Member(e) => e.as_ref().clone(),
    }
}

// ---------------------------------------------------------------------------
// Destructuring-target validation
// ---------------------------------------------------------------------------

impl<'a> Parser<'a> {
    /// Convert an expression to a binding pattern (destructuring assignment
    /// targets, for-in/of heads).
    fn expr_to_pat(&mut self, e: &Expr) -> Result<Pat, SyntaxError> {
        match e {
            Expr::Ident(name, span) => {
                if *name == "let" && self.strict {
                    return Err(Self::early("let is reserved in strict mode", *span));
                }
                Ok(Pat::Ident(name.clone(), *span))
            }
            Expr::Array(elems, span) => {
                let mut out = Vec::new();
                for el in elems {
                    match el {
                        None => out.push(None),
                        Some(Expr::Spread(arg, s)) => {
                            let p = self.expr_to_pat(arg)?;
                            if matches!(p, Pat::Assign(..)) {
                                return Err(Self::early(
                                    "rest element may not have an initializer",
                                    *s,
                                ));
                            }
                            out.push(Some(ArrayPatElem::Pat(Pat::Rest(Box::new(p), *s))));
                        }
                        Some(x) => out.push(Some(ArrayPatElem::Pat(self.expr_to_pat(x)?))),
                    }
                }
                Ok(Pat::Array(out, *span))
            }
            Expr::Object(props, span) => {
                let mut out = Vec::new();
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
                                return Err(Self::early(
                                    "rest property may not have an initializer",
                                    *ps,
                                ));
                            }
                            out.push(ObjectPatProp::Rest(Box::new(p), *ps));
                        }
                        PropDef::Method { span: ps, .. }
                        | PropDef::Getter { span: ps, .. }
                        | PropDef::Setter { span: ps, .. } => {
                            return Err(Self::early(
                                "method definition is not a valid destructuring target",
                                *ps,
                            ));
                        }
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
            Expr::Member { .. } => {
                // ECMA-262 13.15.5: `MemberExpression` is a valid
                // destructuring assignment target (`[a.b] = x`,
                // `({p: o.q} = y)`). Optional chains are rejected like
                // everywhere else (`check_assign_target`).
                if is_optional_chain(e) {
                    return Err(Self::early(
                        "optional chain may not be an assignment target",
                        e.span(),
                    ));
                }
                Ok(Pat::Member(Box::new(e.clone())))
            }
            _ => Err(Self::early("invalid destructuring target", e.span())),
        }
    }

    /// Validate an assignment target: identifier, member, or destructuring
    /// pattern. Optional chains are rejected.
    fn check_assign_target(&mut self, lhs: &Expr) -> Result<(), SyntaxError> {
        if is_optional_chain(lhs) {
            return Err(Self::early(
                "optional chain may not be an assignment target",
                lhs.span(),
            ));
        }
        match lhs {
            Expr::Ident(name, span) => {
                if self.strict && (name == "eval" || name == "arguments") {
                    return Err(Self::early(
                        "invalid assignment to eval/arguments in strict mode",
                        *span,
                    ));
                }
                if name == "yield" && self.yield_is_operator() {
                    return Err(Self::early(
                        "yield may not be an assignment target inside a generator",
                        *span,
                    ));
                }
                if name == "await" && (self.in_async || self.is_module) {
                    return Err(Self::early(
                        "await may not be an assignment target here",
                        *span,
                    ));
                }
                Ok(())
            }
            Expr::Member { .. } => Ok(()),
            Expr::Array(..) | Expr::Object(..) => {
                let pat = self.expr_to_pat(lhs)?;
                if self.strict {
                    for (name, span) in pat_bound_names(&pat) {
                        if name == "eval" || name == "arguments" {
                            return Err(Self::early(
                                "invalid assignment to eval/arguments in strict mode",
                                span,
                            ));
                        }
                    }
                }
                Ok(())
            }
            _ => Err(Self::early("invalid assignment target", lhs.span())),
        }
    }

    /// Validate a `++`/`--` operand.
    fn check_update_target(&mut self, arg: &Expr) -> Result<(), SyntaxError> {
        if is_optional_chain(arg) {
            return Err(Self::early(
                "optional chain may not be an update target",
                arg.span(),
            ));
        }
        match arg {
            Expr::Ident(name, span) => {
                if self.strict && (name == "eval" || name == "arguments") {
                    return Err(Self::early(
                        "invalid update of eval/arguments in strict mode",
                        *span,
                    ));
                }
                Ok(())
            }
            Expr::Member { .. } => Ok(()),
            _ => Err(Self::early("invalid update target", arg.span())),
        }
    }

    /// Convert a for-in/of left-hand side expression to [`ForLhs`].
    fn expr_to_for_lhs(&mut self, e: &Expr) -> Result<ForLhs, SyntaxError> {
        if is_optional_chain(e) {
            return Err(Self::early(
                "optional chain may not be a for-in/of target",
                e.span(),
            ));
        }
        match e {
            Expr::Ident(name, span) => Ok(ForLhs::Ident(name.clone(), *span)),
            Expr::Member { .. } => Ok(ForLhs::Member(Box::new(e.clone()))),
            Expr::Array(..) | Expr::Object(..) => Ok(ForLhs::Pat(self.expr_to_pat(e)?)),
            _ => Err(Self::early("invalid left-hand side in for-loop", e.span())),
        }
    }
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

impl<'a> Parser<'a> {
    /// Parse one formal parameter (without the surrounding parens).
    fn parse_param(&mut self) -> Result<Param, SyntaxError> {
        let start = self.peek_span()?;
        let pat = if matches!(self.peek_at(0)?, TokenKind::Ellipsis) {
            self.bump()?;
            let inner = self.parse_pat()?;
            let span = join_span(start, self.prev_span()?);
            if matches!(inner, Pat::Assign(..)) {
                return Err(Self::early(
                    "rest parameter may not have an initializer",
                    span,
                ));
            }
            Pat::Rest(Box::new(inner), span)
        } else {
            self.parse_pat()?
        };
        let span = join_span(start, self.prev_span()?);
        Ok(Param { pat, span })
    }

    fn parse_params(&mut self) -> Result<Vec<Param>, SyntaxError> {
        self.expect(&TokenKind::LParen, "'('")?;
        let mut params = Vec::new();
        while !matches!(self.peek_at(0)?, TokenKind::RParen | TokenKind::Eof) {
            let param = self.parse_param()?;
            if matches!(param.pat, Pat::Rest(..)) && !matches!(self.peek_at(0)?, TokenKind::RParen)
            {
                return Err(Self::early("rest parameter must be last", param.span));
            }
            params.push(param);
            if matches!(self.peek_at(0)?, TokenKind::Comma) {
                self.bump()?;
            } else {
                break;
            }
        }
        self.expect(&TokenKind::RParen, "')'")?;
        Ok(params)
    }

    /// Validate parameter names: per-name binding rules plus strict-mode
    /// duplicate detection. `strict` is the function's final strictness
    /// (including a `"use strict"` directive in its body).
    fn validate_params(&mut self, params: &[Param], strict: bool) -> Result<(), SyntaxError> {
        let mut seen = HashSet::new();
        for p in params {
            self.check_pat_binding(&p.pat)?;
            if strict {
                for (name, span) in pat_bound_names(&p.pat) {
                    if !seen.insert(name.clone()) {
                        return Err(Self::early(
                            &format!("duplicate parameter name '{name}'"),
                            span,
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Parse `*? name? (params) { body }` for functions and methods.
    /// Returns `(name, params, body, is_generator, strict)`.
    fn parse_function_common(
        &mut self,
        is_async: bool,
        require_name: bool,
    ) -> Result<FunctionCommon, SyntaxError> {
        let is_generator = self.parse_function_star()?;
        // Name (optional for expressions).
        let mut name: Option<String> = None;
        if require_name || is_identifierish(self.peek_at(0)?) {
            let (n, nspan) = self.parse_ident_name()?;
            self.check_function_name(&n, nspan, is_async, is_generator)?;
            name = Some(n);
        }
        if require_name && name.is_none() {
            return Err(self.unexpected("a function name"));
        }
        let (params, body, strict) = self.parse_function_tail(is_async, is_generator)?;
        Ok((name, params, body, is_generator, strict))
    }

    /// Parse a generator `*` (for functions and methods).
    fn parse_function_star(&mut self) -> Result<bool, SyntaxError> {
        if matches!(self.peek_at(0)?, TokenKind::Star) {
            self.bump()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Validate a function name against the function's own kind.
    fn check_function_name(
        &mut self,
        n: &str,
        nspan: Span,
        is_async: bool,
        is_generator: bool,
    ) -> Result<(), SyntaxError> {
        if self.strict && (n == "eval" || n == "arguments") {
            return Err(Self::early(
                "eval/arguments may not be a function name in strict mode",
                nspan,
            ));
        }
        if n == "await" && (is_async || self.is_module) {
            return Err(Self::early("await may not be a function name here", nspan));
        }
        if n == "yield" && (is_generator || self.strict) {
            return Err(Self::early("yield may not be a function name here", nspan));
        }
        Ok(())
    }

    /// Parse `(params) { body }` in the function's context. Handles the
    /// directive prologue (which may enable strict mode) and validates the
    /// parameters against the final strictness.
    fn parse_function_tail(
        &mut self,
        is_async: bool,
        is_generator: bool,
    ) -> Result<(Vec<Param>, Vec<Stmt>, bool), SyntaxError> {
        // Enter the function's context for params and body.
        let save = self.save_ctx();
        self.func_depth += 1;
        self.in_async = is_async;
        self.in_generator = is_generator;
        self.nt_depth += 1;
        // `yield`/`await` are not operators in the parameter list.
        self.param_depth += 1;
        let params = self.parse_params();
        self.param_depth -= 1;
        let params = params?;
        let body = self.parse_function_body(&params)?;
        let strict = self.strict;
        self.validate_params(&params, strict)?;
        self.restore_ctx(save);
        Ok((params, body, strict))
    }

    /// Parse a function body block. The caller has entered the function's
    /// context; this handles the directive prologue (which may enable
    /// strict mode) and restores nothing. `params` is the function's
    /// parameter list: a body that opens with a `"use strict"` directive is
    /// a SyntaxError when the parameter list is non-simple (spec 15.1.1).
    fn parse_function_body(&mut self, params: &[Param]) -> Result<Vec<Stmt>, SyntaxError> {
        self.expect(&TokenKind::LBrace, "'{'")?;
        if !is_simple_params(params) && self.peek_use_strict()? {
            return Err(Self::early(
                "a function with a non-simple parameter list may not contain a \"use strict\" directive",
                self.peek_span()?,
            ));
        }
        if !self.strict && self.peek_use_strict()? {
            self.strict = true;
        }
        let save_top = self.top_level;
        self.top_level = false;
        let mut body = Vec::new();
        while !matches!(self.peek_at(0)?, TokenKind::RBrace | TokenKind::Eof) {
            body.push(self.parse_stmt()?);
        }
        self.expect(&TokenKind::RBrace, "'}'")?;
        self.top_level = save_top;
        // Desugar `using` here: the top-level pass cannot reach function
        // bodies without an expression walk.
        desugar_stmts(&mut body);
        Ok(body)
    }
}

// ---------------------------------------------------------------------------
// Parentheses and arrow functions
// ---------------------------------------------------------------------------

impl<'a> Parser<'a> {
    /// Parse `(...)`: an arrow function (via backtracking) or a
    /// parenthesized expression.
    fn parse_paren_or_arrow(&mut self) -> Result<Expr, SyntaxError> {
        let start = self.peek_span()?;
        // `((` can never begin an arrow's parameter list: skip the
        // backtracking speculation (and its stack frame) entirely. In
        // unoptimized builds the frame is allocated on entry regardless
        // of the early return inside, so the call itself must be skipped
        // to keep deeply nested parentheses cheap.
        if !matches!(self.peek_at(1)?, TokenKind::LParen)
            && let Some(e) = self.try_parse_arrow_from_paren(start, false)?
        {
            return Ok(e);
        }
        // Parenthesized expression (a cover for destructuring assignment).
        self.bump()?; // (
        let save_cover = self.in_paren_cover;
        self.in_paren_cover = true;
        let save_in = self.allow_in;
        self.allow_in = true;
        let e = self.parse_expr();
        self.allow_in = save_in;
        self.in_paren_cover = save_cover;
        let e = e?;
        self.expect(&TokenKind::RParen, "')'")?;
        Ok(e)
    }

    /// Try to parse `(params) => body` starting at `(`. Returns `Ok(None)`
    /// (position restored) if this is not an arrow. Lex errors propagate.
    fn try_parse_arrow_from_paren(
        &mut self,
        start: Span,
        is_async: bool,
    ) -> Result<Option<Expr>, SyntaxError> {
        // `((` can never begin an arrow's parameter list: bail out before
        // spending any stack frames on the backtracking parse. This keeps
        // deeply nested parentheses cheap enough for the depth guard to
        // fire before the host stack is exhausted.
        if matches!(self.peek_at(1)?, TokenKind::LParen) {
            return Ok(None);
        }
        let cp = self.checkpoint();
        let save_cover = self.cover_ok;
        let save_param_depth = self.param_depth;
        let res: Result<Vec<Param>, SyntaxError> = (|| {
            self.bump()?; // (
            self.cover_ok = true;
            self.param_depth += 1;
            let mut params = Vec::new();
            while !matches!(self.peek_at(0)?, TokenKind::RParen | TokenKind::Eof) {
                let param = self.parse_param()?;
                if matches!(param.pat, Pat::Rest(..))
                    && !matches!(self.peek_at(0)?, TokenKind::RParen)
                {
                    return Err(Self::early("rest parameter must be last", param.span));
                }
                params.push(param);
                if matches!(self.peek_at(0)?, TokenKind::Comma) {
                    self.bump()?;
                    if matches!(self.peek_at(0)?, TokenKind::RParen) {
                        break; // trailing comma
                    }
                } else {
                    break;
                }
            }
            self.expect(&TokenKind::RParen, "')'")?;
            // No line break allowed before `=>`.
            if self.nl_before(0)? || !matches!(self.peek_at(0)?, TokenKind::Arrow) {
                return Err(self.unexpected("'=>'"));
            }
            self.bump()?; // =>
            Ok(params)
        })();
        self.cover_ok = save_cover;
        self.param_depth = save_param_depth;
        match res {
            Ok(params) => {
                self.release(cp);
                // Validate in the arrow's async context (`await` is reserved
                // for async arrows).
                let save_async = self.in_async;
                if is_async {
                    self.in_async = true;
                }
                let r = self.validate_params(&params, self.strict);
                self.in_async = save_async;
                r?;
                let e = self.finish_arrow_body(start, params, is_async)?;
                Ok(Some(e))
            }
            Err(e) if matches!(e, SyntaxError::Lex { .. }) => {
                self.release(cp);
                Err(e)
            }
            Err(_) => {
                self.rewind_to(cp);
                Ok(None)
            }
        }
    }

    /// Parse the body of an arrow function. Parameters were parsed in the
    /// outer context; the body runs in the arrow's context (arrows are never
    /// generators, never have their own `new.target`, and inherit `super`).
    fn finish_arrow_body(
        &mut self,
        start: Span,
        params: Vec<Param>,
        is_async: bool,
    ) -> Result<Expr, SyntaxError> {
        let save = self.save_ctx();
        self.func_depth += 1;
        self.in_async = is_async;
        self.in_generator = false;
        // `nt_depth`, `in_method`, `in_field_init`, `in_static_block`,
        // `in_derived_ctor` are inherited by arrows: do not reset.
        let (body, strict) = if matches!(self.peek_at(0)?, TokenKind::LBrace) {
            self.bump()?; // {
            // Spec 15.2.1: a non-simple parameter list with a "use strict"
            // directive in the body is a SyntaxError.
            if !is_simple_params(&params) && self.peek_use_strict()? {
                return Err(Self::early(
                    "a function with a non-simple parameter list may not contain a \"use strict\" directive",
                    self.peek_span()?,
                ));
            }
            if !self.strict && self.peek_use_strict()? {
                self.strict = true;
            }
            // A "use strict" directive in an arrow body validates params.
            let strict = self.strict;
            let save_top = self.top_level;
            self.top_level = false;
            let mut stmts = Vec::new();
            while !matches!(self.peek_at(0)?, TokenKind::RBrace | TokenKind::Eof) {
                stmts.push(self.parse_stmt()?);
            }
            self.expect(&TokenKind::RBrace, "'}'")?;
            self.top_level = save_top;
            self.validate_params(&params, strict)?;
            // Desugar `using`: unreachable from the top-level pass.
            desugar_stmts(&mut stmts);
            (ArrowBody::Block(stmts), strict)
        } else {
            // Concise body: the params are already validated. No directive
            // prologue is possible, so strictness is just the inherited one.
            let strict = self.strict;
            self.validate_params(&params, strict)?;
            (ArrowBody::Expr(Box::new(self.parse_assignment()?)), strict)
        };
        self.restore_ctx(save);
        let span = join_span(start, self.prev_span()?);
        Ok(Expr::Arrow {
            params,
            body,
            is_async,
            strict,
            span,
        })
    }
}

// ---------------------------------------------------------------------------
// Classes
// ---------------------------------------------------------------------------

impl<'a> Parser<'a> {
    /// Parse `class` (declaration or expression).
    /// Returns `(name, superclass, members)`.
    fn parse_class_common(
        &mut self,
        start: Span,
        require_name: bool,
    ) -> Result<ClassCommon, SyntaxError> {
        // Class bodies are always strict (spec 15.7.1): force strict mode
        // for the whole class parse, restoring the outer strictness after.
        let save_strict = self.strict;
        self.strict = true;
        let r = self.parse_class_common_strict(start, require_name);
        self.strict = save_strict;
        r
    }

    /// [`Self::parse_class_common`] with strict mode forced on entry.
    fn parse_class_common_strict(
        &mut self,
        start: Span,
        require_name: bool,
    ) -> Result<ClassCommon, SyntaxError> {
        self.bump()?; // class
        let mut name: Option<String> = None;
        if require_name || is_identifierish(self.peek_at(0)?) {
            let (n, nspan) = self.parse_ident_name()?;
            if self.strict && (n == "eval" || n == "arguments") {
                return Err(Self::early(
                    "eval/arguments may not be a class name in strict mode",
                    nspan,
                ));
            }
            if n == "await" && self.is_module {
                return Err(Self::early("await may not be a class name here", nspan));
            }
            name = Some(n);
        }
        if require_name && name.is_none() {
            return Err(self.unexpected("a class name"));
        }
        // `extends` clause: a left-hand-side expression. Depth-guarded:
        // `class A extends (class B extends ...)` nests `parse_lhs` ->
        // `parse_primary` -> `parse_class_common` without any other guard
        // on the path. The class body is parsed by [`Self::parse_class_body`]
        // *after* the heritage returns, so this frame stays small while the
        // heritage recurses (keeps the depth guard effective on debug
        // builds, where a fat frame per level would exhaust the host stack
        // before the guard fires).
        let superclass = if keyword_text(self.peek_at(0)?) == Some("extends") {
            self.bump()?;
            Some(Box::new(self.guarded(|s| s.parse_lhs())?.0))
        } else {
            None
        };
        let members = self.parse_class_body(superclass.is_some())?;
        let _ = start;
        Ok((name, superclass, members))
    }

    /// Parse a class `{ ... }` body after the header (name + heritage).
    /// Split out of [`Self::parse_class_common`] so the heritage recursion
    /// above does not hold this function's (large) frame on the stack.
    fn parse_class_body(&mut self, is_derived: bool) -> Result<Vec<ClassMember>, SyntaxError> {
        self.expect(&TokenKind::LBrace, "'{'")?;
        // Push a fresh private-name scope for this class.
        self.private_declared.push(HashMap::new());
        self.private_used.push(Vec::new());
        // Track whether this class is derived (for `super()` validation).
        let save_derived = self.class_is_derived;
        self.class_is_derived = is_derived;
        let mut has_constructor = false;
        let mut members = Vec::new();
        while !matches!(self.peek_at(0)?, TokenKind::RBrace | TokenKind::Eof) {
            if matches!(self.peek_at(0)?, TokenKind::Semicolon) {
                self.bump()?;
                continue;
            }
            members.push(self.parse_class_element(&mut has_constructor)?);
        }
        self.expect(&TokenKind::RBrace, "'}'")?;
        self.class_is_derived = save_derived;
        // Validate private-name uses against declarations.
        let declared = self.private_declared.pop().expect("class private scope");
        let used = self.private_used.pop().expect("class private uses");
        for (pname, pspan) in used {
            if !declared.contains_key(&pname) {
                return Err(Self::early(
                    &format!("private name '#{pname}' is not declared in this class"),
                    pspan,
                ));
            }
        }
        Ok(members)
    }

    /// Declare a private name in the innermost class scope.
    /// A getter and setter may share a name; all other duplicates are errors.
    fn declare_private(
        &mut self,
        name: &str,
        span: Span,
        kind: PrivateKind,
    ) -> Result<(), SyntaxError> {
        match self.private_declared.last_mut() {
            Some(decl) => {
                if let Some(&existing) = decl.get(name) {
                    // Allow getter+setter pair.
                    let is_pair = matches!(
                        (existing, kind),
                        (PrivateKind::Getter, PrivateKind::Setter)
                            | (PrivateKind::Setter, PrivateKind::Getter)
                    );
                    if !is_pair {
                        return Err(Self::early(
                            &format!("duplicate private name '#{name}'"),
                            span,
                        ));
                    }
                    // Record the pair as Other to prevent a third declaration.
                    decl.insert(name.to_string(), PrivateKind::Other);
                } else {
                    decl.insert(name.to_string(), kind);
                }
                Ok(())
            }
            None => Err(Self::early(
                &format!("private name '#{name}' declared outside of class"),
                span,
            )),
        }
    }

    /// Record a use of a private name in the innermost class scope.
    fn use_private(&mut self, name: &str, span: Span) -> Result<(), SyntaxError> {
        match self.private_used.last_mut() {
            Some(used) => {
                used.push((name.to_string(), span));
                Ok(())
            }
            None => Err(Self::early(
                &format!("private name '#{name}' used outside of class"),
                span,
            )),
        }
    }

    /// Parse a computed class-element key in the OUTER private scope
    /// (computed keys are evaluated outside the class).
    fn parse_class_computed_key(&mut self) -> Result<PropKey, SyntaxError> {
        self.bump()?; // [
        // Temporarily pop this class's private scope.
        let decl = self.private_declared.pop();
        let used = self.private_used.pop();
        let e = self.parse_assignment();
        if let Some(d) = decl {
            self.private_declared.push(d);
        }
        if let Some(u) = used {
            self.private_used.push(u);
        }
        let e = e?;
        self.expect(&TokenKind::RBracket, "']'")?;
        Ok(PropKey::Computed(e))
    }

    /// Parse a class element key: `#private` or a public property key.
    fn parse_class_key(&mut self) -> Result<ClassKey, SyntaxError> {
        if matches!(self.peek_at(0)?, TokenKind::PrivateIdent(_)) {
            let (name, span) = self.take_private_ident()?;
            // `#constructor` is not a valid private method name; checked by
            // the caller for methods. Declare here for fields.
            return Ok(ClassKey::Private(name, span));
        }
        if matches!(self.peek_at(0)?, TokenKind::LBracket) {
            return Ok(ClassKey::Public(self.parse_class_computed_key()?));
        }
        Ok(ClassKey::Public(self.parse_prop_key()?))
    }

    /// Parse one class element (not `;`).
    fn parse_class_element(
        &mut self,
        has_constructor: &mut bool,
    ) -> Result<ClassMember, SyntaxError> {
        let start = self.peek_span()?;
        // `static` modifier — may cross a line break. It is a NAME (not a
        // modifier) when followed by `=`, `;`, `(`, or `}`.
        let mut is_static = false;
        if keyword_text(self.peek_at(0)?) == Some("static") {
            match self.peek_at(1)? {
                TokenKind::LBrace => {
                    self.bump()?; // static
                    return self.parse_static_block(start);
                }
                TokenKind::Eq
                | TokenKind::Semicolon
                | TokenKind::LParen
                | TokenKind::RBrace
                | TokenKind::Eof => {}
                _ => {
                    if self.peek_starts_class_key(1)? {
                        self.bump()?; // static
                        is_static = true;
                    }
                }
            }
        }
        // `async` modifier — may NOT cross a line break, and `async static`
        // is invalid (must be `static async`).
        let mut is_async = false;
        if !is_static && keyword_text(self.peek_at(0)?) == Some("async") && !self.nl_before(1)? {
            if keyword_text(self.peek_at(1)?) == Some("static") && self.peek_starts_class_key(2)? {
                return Err(Self::early(
                    "invalid order: `static` must come before `async`",
                    start,
                ));
            }
            match self.async_method_shape()? {
                AsyncShape::Method => {
                    self.bump()?; // async
                    is_async = true;
                }
                AsyncShape::AsyncAccessor => {
                    return Err(Self::early("async accessors are not allowed", start));
                }
                AsyncShape::Name => {}
            }
        }
        // Generator `*`.
        let is_generator = self.parse_function_star()?;
        // Key.
        let key = self.parse_class_key()?;
        // Declare private names (fields and methods).
        if let ClassKey::Private(pname, pspan) = &key {
            // `#constructor` is never valid.
            if pname == "constructor" {
                return Err(Self::early(
                    "private method '#constructor' is not allowed",
                    *pspan,
                ));
            }
            self.declare_private(pname, *pspan, PrivateKind::Other)?;
        }
        // `get`/`set` accessors (not async/generator).
        if !is_async
            && !is_generator
            && matches!(key, ClassKey::Public(PropKey::Ident(ref n, _)) if n == "get" || n == "set")
            && self.peek_starts_class_key(0)?
        {
            let acc = match key {
                ClassKey::Public(PropKey::Ident(n, _)) => n,
                _ => unreachable!(),
            };
            // Async accessors are invalid; `async` was ruled out above.
            let key2 = self.parse_class_key()?;
            if let ClassKey::Private(pname, pspan) = &key2 {
                if pname == "constructor" {
                    return Err(Self::early(
                        "private accessor '#constructor' is not allowed",
                        *pspan,
                    ));
                }
                let pk = if acc == "get" {
                    PrivateKind::Getter
                } else {
                    PrivateKind::Setter
                };
                self.declare_private(pname, *pspan, pk)?;
            }
            let kind = if acc == "get" {
                MethodKind::Getter
            } else {
                MethodKind::Setter
            };
            let (params, body) = self.parse_class_method_tail(is_static, false, false, kind)?;
            if acc == "get" && !params.is_empty() {
                return Err(Self::early("getter must have no parameters", start));
            }
            if acc == "set" && params.len() != 1 {
                return Err(Self::early("setter must have exactly one parameter", start));
            }
            return Ok(ClassMember::Method {
                key: key2,
                params,
                body,
                is_async: false,
                is_generator: false,
                is_static,
                kind,
            });
        }
        // Method vs field.
        if matches!(self.peek_at(0)?, TokenKind::LParen) {
            let kind =
                self.class_method_kind(&key, is_static, is_async, is_generator, has_constructor)?;
            let (params, body) =
                self.parse_class_method_tail(is_static, is_async, is_generator, kind)?;
            return Ok(ClassMember::Method {
                key,
                params,
                body,
                is_async,
                is_generator,
                is_static,
                kind,
            });
        }
        // Field: `key` or `key = init`, terminated by `;`/ASI.
        if is_async || is_generator {
            return Err(Self::early("a field cannot be async or a generator", start));
        }
        // Instance fields may not be named `constructor` (also static).
        let ctor_key_span = match &key {
            ClassKey::Public(PropKey::Ident(n, s)) if n == "constructor" => Some(*s),
            ClassKey::Public(PropKey::Str(n)) if n == "constructor" => Some(start),
            _ => None,
        };
        if let Some(kspan) = ctor_key_span {
            return Err(Self::early(
                "field named 'constructor' is not allowed",
                kspan,
            ));
        }
        let init = if matches!(self.peek_at(0)?, TokenKind::Eq) {
            self.bump()?;
            // Field initializers run in a non-async, non-generator scope
            // with `super` property access and `new.target` available.
            let save = self.save_ctx();
            self.in_async = false;
            self.in_generator = false;
            self.in_method = true;
            self.in_derived_ctor = false;
            self.in_field_init = true;
            self.nt_depth += 1;
            let e = self.parse_assignment();
            self.restore_ctx(save);
            Some(e?)
        } else {
            None
        };
        self.expect_class_semi()?;
        Ok(ClassMember::Field {
            key,
            init,
            is_static,
        })
    }

    /// Determine the [`MethodKind`] and validate constructor rules.
    fn class_method_kind(
        &mut self,
        key: &ClassKey,
        is_static: bool,
        is_async: bool,
        is_generator: bool,
        has_constructor: &mut bool,
    ) -> Result<MethodKind, SyntaxError> {
        let is_ctor_name = matches!(
            key,
            ClassKey::Public(PropKey::Ident(n, _)) if n == "constructor"
        );
        if is_ctor_name && !is_static {
            if is_async || is_generator {
                return Err(Self::early(
                    "constructor may not be async or a generator",
                    self.peek_span()?,
                ));
            }
            if *has_constructor {
                return Err(Self::early("duplicate constructor", self.peek_span()?));
            }
            *has_constructor = true;
            Ok(MethodKind::Constructor)
        } else {
            Ok(MethodKind::Method)
        }
    }

    /// Parse `(params) { body }` for a class method in the method's context.
    /// `is_static` methods still get `super` property access.
    fn parse_class_method_tail(
        &mut self,
        _is_static: bool,
        is_async: bool,
        is_generator: bool,
        kind: MethodKind,
    ) -> Result<(Vec<Param>, Vec<Stmt>), SyntaxError> {
        let save = self.save_ctx();
        self.func_depth += 1;
        self.in_async = is_async;
        self.in_generator = is_generator;
        self.in_method = true;
        // `super()` calls are only valid in derived constructors.
        self.in_derived_ctor = kind == MethodKind::Constructor && self.class_is_derived;
        self.nt_depth += 1;
        self.param_depth += 1;
        let params = self.parse_params();
        self.param_depth -= 1;
        let params = params?;
        let body = self.parse_function_body(&params)?;
        let strict = self.strict;
        self.validate_params(&params, strict)?;
        self.restore_ctx(save);
        Ok((params, body))
    }

    /// Parse `static { ... }`.
    fn parse_static_block(&mut self, _start: Span) -> Result<ClassMember, SyntaxError> {
        self.expect(&TokenKind::LBrace, "'{'")?;
        // Static blocks run in a non-async scope; `super` property access
        // and `new.target` are available; `return` is invalid.
        let save = self.save_ctx();
        self.in_async = false;
        self.in_generator = false;
        self.in_method = true;
        self.in_derived_ctor = false;
        self.in_static_block = true;
        self.nt_depth += 1;
        let save_top = self.top_level;
        self.top_level = false;
        let mut body = Vec::new();
        while !matches!(self.peek_at(0)?, TokenKind::RBrace | TokenKind::Eof) {
            body.push(self.parse_stmt()?);
        }
        self.expect(&TokenKind::RBrace, "'}'")?;
        self.top_level = save_top;
        self.restore_ctx(save);
        // Desugar `using`: unreachable from the top-level pass.
        desugar_stmts(&mut body);
        Ok(ClassMember::StaticBlock { body })
    }

    /// Expect `;` or ASI after a class field.
    fn expect_class_semi(&mut self) -> Result<(), SyntaxError> {
        if matches!(self.peek_at(0)?, TokenKind::Semicolon) {
            self.bump()?;
            return Ok(());
        }
        if matches!(self.peek_at(0)?, TokenKind::RBrace | TokenKind::Eof) || self.nl_before(0)? {
            return Ok(());
        }
        Err(self.unexpected("';'"))
    }

    /// Does the token at `off` start a class-element key?
    /// (`*` and `{` are handled separately by the caller.)
    fn peek_starts_class_key(&mut self, off: usize) -> Result<bool, SyntaxError> {
        Ok(match self.peek_at(off)? {
            TokenKind::LBracket => true,
            TokenKind::PrivateIdent(_) => true,
            TokenKind::Str(_) | TokenKind::Numeric(_) | TokenKind::NumericOctal(_) => true,
            t if is_identifierish(t) => true,
            _ => false,
        })
    }

    /// Classify `async` (at current pos, no line break before peek(1)) in a
    /// class body: is it an async-method modifier, an (invalid) async
    /// accessor, or just the element name?
    fn async_method_shape(&mut self) -> Result<AsyncShape, SyntaxError> {
        let mut off = 1;
        if matches!(self.peek_at(off)?, TokenKind::Star) {
            off += 1;
        }
        // Check for `async get/set <key>` (async accessor → error).
        if matches!(keyword_text(self.peek_at(off)?), Some("get") | Some("set"))
            && self.peek_starts_class_key(off + 1)?
        {
            // `async get()` — method named `get`, not an accessor.
            if !matches!(self.peek_at(off + 1)?, TokenKind::LParen) {
                return Ok(AsyncShape::AsyncAccessor);
            }
        }
        // Find the token after the key.
        match self.peek_at(off)? {
            TokenKind::LBracket => {
                // Scan past balanced brackets.
                let mut depth = 0u32;
                loop {
                    match self.peek_at(off)? {
                        TokenKind::LBracket => depth += 1,
                        TokenKind::RBracket => {
                            depth -= 1;
                            if depth == 0 {
                                off += 1;
                                break;
                            }
                        }
                        TokenKind::Eof => return Ok(AsyncShape::Name),
                        _ => {}
                    }
                    off += 1;
                    if off > 100000 {
                        return Ok(AsyncShape::Name);
                    }
                }
            }
            t if is_identifierish(t)
                || matches!(
                    t,
                    TokenKind::Str(_)
                        | TokenKind::Numeric(_)
                        | TokenKind::NumericOctal(_)
                        | TokenKind::PrivateIdent(_)
                ) =>
            {
                off += 1;
            }
            _ => return Ok(AsyncShape::Name),
        }
        if matches!(self.peek_at(off)?, TokenKind::LParen) {
            Ok(AsyncShape::Method)
        } else {
            Ok(AsyncShape::Name)
        }
    }
}

/// What `async` means at the start of a class element.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AsyncShape {
    /// `async` is a method modifier.
    Method,
    /// `async get/set` — invalid.
    AsyncAccessor,
    /// `async` is the element name.
    Name,
}

// ---------------------------------------------------------------------------
// Modules: import / export declarations
// ---------------------------------------------------------------------------

impl<'a> Parser<'a> {
    /// Parse an `import` declaration. The `import` token is current.
    fn parse_import(&mut self) -> Result<Stmt, SyntaxError> {
        let start = self.bump()?; // import
        if !self.is_module {
            return Err(Self::early(
                "import declarations are only allowed in modules",
                start,
            ));
        }
        // `import "source";` — side-effect import.
        if matches!(self.peek_at(0)?, TokenKind::Str(_)) {
            let source = self.parse_module_source()?;
            self.expect_semi()?;
            let span = join_span(start, self.prev_span()?);
            return Ok(Stmt::Import(ImportDecl {
                default: None,
                namespace: None,
                named: Vec::new(),
                source,
                span,
            }));
        }
        let mut default: Option<(String, Span)> = None;
        let mut namespace: Option<(String, Span)> = None;
        let mut named: Vec<ImportSpec> = Vec::new();
        // Default import: `import def`, `import def, ...`
        if is_identifierish(self.peek_at(0)?) {
            let (n, nspan) = self.parse_ident_name()?;
            self.check_binding_name(&n, nspan)?;
            default = Some((n, nspan));
            if matches!(self.peek_at(0)?, TokenKind::Comma) {
                self.bump()?;
            }
        }
        // Namespace or named imports.
        if matches!(self.peek_at(0)?, TokenKind::Star) {
            self.bump()?; // *
            // `as ns`
            match keyword_text(self.peek_at(0)?) {
                Some("as") => {
                    self.bump()?;
                }
                _ => return Err(self.unexpected("'as'")),
            }
            let (n, nspan) = self.parse_ident_name()?;
            self.check_binding_name(&n, nspan)?;
            namespace = Some((n, nspan));
        } else if matches!(self.peek_at(0)?, TokenKind::LBrace) {
            self.bump()?; // {
            while !matches!(self.peek_at(0)?, TokenKind::RBrace | TokenKind::Eof) {
                let imported = self.parse_module_name()?;
                let local = if keyword_text(self.peek_at(0)?) == Some("as") {
                    self.bump()?; // as
                    let (n, _) = self.parse_ident_name()?;
                    n
                } else {
                    // `import {a}` — local is `a` (must be an identifier).
                    match &imported {
                        ModuleName::Ident(n, _) => n.clone(),
                        ModuleName::Str(_, sspan) => {
                            return Err(Self::early("string import name requires 'as'", *sspan));
                        }
                    }
                };
                let lspan = self.prev_span()?;
                self.check_binding_name(&local, lspan)?;
                named.push(ImportSpec { imported, local });
                if matches!(self.peek_at(0)?, TokenKind::Comma) {
                    self.bump()?;
                } else {
                    break;
                }
            }
            self.expect(&TokenKind::RBrace, "'}'")?;
        } else if default.is_none() {
            return Err(self.unexpected("a module specifier"));
        }
        // `from "source"`
        match keyword_text(self.peek_at(0)?) {
            Some("from") => {
                self.bump()?;
            }
            _ => return Err(self.unexpected("'from'")),
        }
        let source = self.parse_module_source()?;
        self.expect_semi()?;
        let span = join_span(start, self.prev_span()?);
        Ok(Stmt::Import(ImportDecl {
            default,
            namespace,
            named,
            source,
            span,
        }))
    }

    /// Parse a module specifier string (cooked). Specifiers feed URL
    /// resolution (`&str`): a lone surrogate degrades lossily, and the
    /// resolution then fails as it would for any unresolvable specifier.
    fn parse_module_source(&mut self) -> Result<String, SyntaxError> {
        match self.peek_at(0)? {
            TokenKind::Str(_) => Ok(self.take_str()?.0.to_utf8_lossy().into_owned()),
            _ => Err(self.unexpected("a module specifier string")),
        }
    }

    /// Parse an imported/exported name: identifier or string.
    fn parse_module_name(&mut self) -> Result<ModuleName, SyntaxError> {
        let start = self.peek_span()?;
        match self.peek_at(0)? {
            TokenKind::Str(_) => {
                let (s, span) = self.take_str()?;
                let span = join_span(start, span);
                Ok(ModuleName::Str(s, span))
            }
            _ => {
                let (n, nspan) = self.parse_ident_name()?;
                Ok(ModuleName::Ident(n, nspan))
            }
        }
    }

    /// Parse an `export` declaration. The `export` token is current.
    fn parse_export(&mut self) -> Result<Stmt, SyntaxError> {
        let start = self.bump()?; // export
        if !self.is_module {
            return Err(Self::early(
                "export declarations are only allowed in modules",
                start,
            ));
        }
        // `export * from "m"`, `export * as ns from "m"`.
        if matches!(self.peek_at(0)?, TokenKind::Star) {
            self.bump()?; // *
            let mut exported: Option<String> = None;
            if keyword_text(self.peek_at(0)?) == Some("as") {
                self.bump()?; // as
                // `as` name can be identifier or string.
                exported = Some(match self.parse_module_name()? {
                    ModuleName::Ident(n, _) => n,
                    // Module specifiers feed URL resolution (`&str`): lossy.
                    ModuleName::Str(s, _) => s.to_utf8_lossy().into_owned(),
                });
            }
            match keyword_text(self.peek_at(0)?) {
                Some("from") => {
                    self.bump()?;
                }
                _ => return Err(self.unexpected("'from'")),
            }
            let source = self.parse_module_source()?;
            self.expect_semi()?;
            let span = join_span(start, self.prev_span()?);
            return Ok(Stmt::Export(ExportDecl::All {
                source,
                exported,
                span,
            }));
        }
        // `export default ...`
        if keyword_text(self.peek_at(0)?) == Some("default") {
            self.bump()?; // default
            // `export default function` / `export default class` /
            // `export default async function`
            let expr = if matches!(self.peek_at(0)?, TokenKind::Function) {
                let fstart = self.bump()?; // function
                let (name, params, body, is_generator, strict) =
                    self.parse_function_common(false, false)?;
                let span = join_span(fstart, self.prev_span()?);
                Expr::Function {
                    name,
                    params,
                    body,
                    is_async: false,
                    is_generator,
                    strict,
                    span,
                }
            } else if matches!(self.peek_at(0)?, TokenKind::Class) {
                let cstart = self.peek_span()?;
                let (name, superclass, body) = self.parse_class_common(cstart, false)?;
                let span = join_span(cstart, self.prev_span()?);
                Expr::Class {
                    name,
                    superclass,
                    body,
                    span,
                }
            } else if keyword_text(self.peek_at(0)?) == Some("async")
                && matches!(self.peek_at(1)?, TokenKind::Function)
                && !self.nl_before(1)?
            {
                let fstart = self.bump()?; // async
                self.bump()?; // function
                let (name, params, body, is_generator, strict) =
                    self.parse_function_common(true, false)?;
                let span = join_span(fstart, self.prev_span()?);
                Expr::Function {
                    name,
                    params,
                    body,
                    is_async: true,
                    is_generator,
                    strict,
                    span,
                }
            } else {
                let e = self.parse_assignment()?;
                self.expect_semi()?;
                let span = join_span(start, self.prev_span()?);
                return Ok(Stmt::Export(ExportDecl::Default { expr: e, span }));
            };
            let span = join_span(start, self.prev_span()?);
            return Ok(Stmt::Export(ExportDecl::Default { expr, span }));
        }
        // `export { ... }` (with optional `from "m"`).
        if matches!(self.peek_at(0)?, TokenKind::LBrace) {
            self.bump()?; // {
            let mut specifiers = Vec::new();
            while !matches!(self.peek_at(0)?, TokenKind::RBrace | TokenKind::Eof) {
                let local = self.parse_module_name()?;
                let exported = if keyword_text(self.peek_at(0)?) == Some("as") {
                    self.bump()?; // as
                    self.parse_module_name()?
                } else {
                    match &local {
                        ModuleName::Ident(n, s) => ModuleName::Ident(n.clone(), *s),
                        ModuleName::Str(_, sspan) => {
                            return Err(Self::early("string export name requires 'as'", *sspan));
                        }
                    }
                };
                specifiers.push(ExportSpec { local, exported });
                if matches!(self.peek_at(0)?, TokenKind::Comma) {
                    self.bump()?;
                } else {
                    break;
                }
            }
            self.expect(&TokenKind::RBrace, "'}'")?;
            let source = if keyword_text(self.peek_at(0)?) == Some("from") {
                self.bump()?;
                Some(self.parse_module_source()?)
            } else {
                None
            };
            self.expect_semi()?;
            let span = join_span(start, self.prev_span()?);
            return Ok(Stmt::Export(ExportDecl::Named {
                specifiers,
                source,
                span,
            }));
        }
        // `export var/let/const/function/class ...`
        let decl = self.parse_stmt()?;
        let span = join_span(start, self.prev_span()?);
        // Must be a declaration.
        match &decl {
            Stmt::VarDecl { .. } | Stmt::FunctionDecl { .. } | Stmt::ClassDecl { .. } => {}
            _ => {
                return Err(Self::early(
                    "export must be followed by a declaration",
                    span,
                ));
            }
        }
        Ok(Stmt::Export(ExportDecl::Decl {
            inner: Box::new(decl),
            span,
        }))
    }
}

// ---------------------------------------------------------------------------
// `using` / `await using` desugar
// ---------------------------------------------------------------------------

/// Desugar explicit resource management after parsing (top level) or during
/// parsing (function bodies, arrow bodies, static blocks — statement lists
/// the top-level pass cannot reach without an expression walk).
///
/// A statement list containing `using` declarations becomes:
///
/// ```js
/// let #using_threw = false;
/// let #using_thrown;
/// try {
///   // for each `using <pat> = e` (in order):
///   let #using_tN = e;
///   let #using_rN = #add_disposable(#using_tN, is_async);
///   const <pat> = #using_tN;
///   // ...rest of the list...
/// } catch (#using_e) {
///   #using_threw = true;
///   #using_thrown = #using_e;
/// } finally {
///   // dispose #using_rN in reverse order, chaining errors with
///   // #suppressed so a disposal error never masks the original throw
/// }
/// ```
///
/// All synthetic names start with `#`, which user code cannot spell, so they
/// never collide with (or capture) user bindings. `for (using ...)` heads
/// are rewritten first: C-style to a synthetic block, for-in/of to a
/// `let #using_tmp` loop with a per-iteration `using` in the body.
fn desugar_using(body: &mut Vec<Stmt>) {
    desugar_stmts(body);
}

/// Span for synthetic nodes (never surfaces: synthetic code cannot fail to
/// compile, and runtime errors blame the disposal call site).
const USING_SPAN: Span = Span { start: 0, end: 0 };

fn uid(name: &str) -> Expr {
    Expr::Ident(name.to_string(), USING_SPAN)
}

fn udecl(kind: VarKind, name: &str, init: Option<Expr>) -> Stmt {
    Stmt::VarDecl {
        kind,
        decls: vec![VarDecl {
            id: Pat::Ident(name.to_string(), USING_SPAN),
            init,
            span: USING_SPAN,
        }],
    }
}

fn uassign(left: Expr, right: Expr) -> Expr {
    Expr::Assign {
        op: AssignOp::Assign,
        left: Box::new(left),
        right: Box::new(right),
        span: USING_SPAN,
    }
}

/// `true` for a direct `using`/`await using` declaration statement.
fn is_using_decl(s: &Stmt) -> bool {
    matches!(
        s,
        Stmt::VarDecl {
            kind: VarKind::Using | VarKind::AwaitUsing,
            ..
        }
    )
}

fn desugar_stmts(stmts: &mut Vec<Stmt>) {
    for stmt in stmts.iter_mut() {
        rewrite_for_using(stmt);
    }
    if let Some(first) = stmts.iter().position(is_using_decl) {
        wrap_using_block(stmts, first);
    }
    for stmt in stmts.iter_mut() {
        desugar_nested(stmt);
    }
}

/// Wrap `stmts[first..]` (everything from the first `using` on) in the
/// try/catch/finally disposal scaffold.
fn wrap_using_block(stmts: &mut Vec<Stmt>, first: usize) {
    let rest: Vec<Stmt> = stmts.drain(first..).collect();
    let mut try_body = Vec::with_capacity(rest.len());
    let mut records: Vec<(String, bool)> = Vec::new();
    let mut counter = 0usize;
    for s in rest {
        match s {
            Stmt::VarDecl { kind, decls }
                if matches!(kind, VarKind::Using | VarKind::AwaitUsing) =>
            {
                let is_async = kind == VarKind::AwaitUsing;
                try_body.extend(expand_using_decl(
                    is_async,
                    decls,
                    &mut records,
                    &mut counter,
                ));
            }
            s => try_body.push(s),
        }
    }
    stmts.extend(build_dispose_wrap(try_body, records));
}

/// Expand one `using` declaration list into temp/record/binding statements.
/// `records` collects `(record_name, is_async)` in evaluation order.
/// Temps and records are *assigned* here; they are declared in the
/// prologue by [`build_dispose_wrap`] so the finalizer can see them.
fn expand_using_decl(
    is_async: bool,
    decls: Vec<VarDecl>,
    records: &mut Vec<(String, bool)>,
    counter: &mut usize,
) -> Vec<Stmt> {
    let mut out = Vec::with_capacity(decls.len() * 3);
    for d in decls {
        let n = *counter;
        *counter += 1;
        let temp = format!("#using_t{n}");
        let rec = format!("#using_r{n}");
        records.push((rec.clone(), is_async));
        // The parser requires an initializer.
        let init = d.init.expect("using declaration without initializer");
        // #using_tN = e;
        out.push(Stmt::Expr(uassign(uid(&temp), init)));
        // #using_rN = #add_disposable(#using_tN, is_async);
        let add = Expr::Call {
            callee: Box::new(uid("#add_disposable")),
            args: vec![uid(&temp), Expr::Lit(Literal::Bool(is_async), USING_SPAN)],
            optional: false,
            span: USING_SPAN,
        };
        out.push(Stmt::Expr(uassign(uid(&rec), add)));
        // const <pat> = #using_tN; (`using` bindings are immutable)
        out.push(Stmt::VarDecl {
            kind: VarKind::Const,
            decls: vec![VarDecl {
                id: d.id,
                init: Some(uid(&temp)),
                span: d.span,
            }],
        });
    }
    out
}

/// Build the prologue + try/catch/finally around `try_body`, disposing
/// `records` (in reverse order) in the finalizer.
fn build_dispose_wrap(try_body: Vec<Stmt>, records: Vec<(String, bool)>) -> Vec<Stmt> {
    let mut out = Vec::new();
    out.push(udecl(
        VarKind::Let,
        "#using_threw",
        Some(Expr::Lit(Literal::Bool(false), USING_SPAN)),
    ));
    out.push(udecl(VarKind::Let, "#using_thrown", None));
    // Declare the temps/records here (outside the try) so the finalizer
    // can read them.
    for i in 0..records.len() {
        out.push(udecl(VarKind::Let, &format!("#using_t{i}"), None));
        out.push(udecl(VarKind::Let, &format!("#using_r{i}"), None));
    }

    let handler = CatchClause {
        param: Some(Param {
            pat: Pat::Ident("#using_e".to_string(), USING_SPAN),
            span: USING_SPAN,
        }),
        body: vec![
            Stmt::Expr(uassign(
                uid("#using_threw"),
                Expr::Lit(Literal::Bool(true), USING_SPAN),
            )),
            Stmt::Expr(uassign(uid("#using_thrown"), uid("#using_e"))),
        ],
    };

    let mut finalizer: Vec<Stmt> = vec![
        udecl(VarKind::Let, "#using_err", None),
        udecl(VarKind::Let, "#using_is_throw", Some(uid("#using_threw"))),
        Stmt::If {
            cond: uid("#using_threw"),
            then_b: Box::new(Stmt::Expr(uassign(uid("#using_err"), uid("#using_thrown")))),
            else_b: None,
        },
    ];
    for (rec, is_async) in records.into_iter().rev() {
        let dispose = Expr::Call {
            callee: Box::new(uid("#call_dispose")),
            args: vec![uid(&rec)],
            optional: false,
            span: USING_SPAN,
        };
        let dispose = if is_async {
            Expr::Await(Box::new(dispose), USING_SPAN)
        } else {
            dispose
        };
        let suppressed = Expr::Call {
            callee: Box::new(uid("#suppressed")),
            args: vec![uid("#using_e2"), uid("#using_err")],
            optional: false,
            span: USING_SPAN,
        };
        let chained = Expr::Conditional {
            cond: Box::new(uid("#using_is_throw")),
            then_e: Box::new(suppressed),
            else_e: Box::new(uid("#using_e2")),
            span: USING_SPAN,
        };
        let catcher = CatchClause {
            param: Some(Param {
                pat: Pat::Ident("#using_e2".to_string(), USING_SPAN),
                span: USING_SPAN,
            }),
            body: vec![
                Stmt::Expr(uassign(uid("#using_err"), chained)),
                Stmt::Expr(uassign(
                    uid("#using_is_throw"),
                    Expr::Lit(Literal::Bool(true), USING_SPAN),
                )),
            ],
        };
        finalizer.push(Stmt::Try {
            block: vec![Stmt::Expr(dispose)],
            handler: Some(catcher),
            finalizer: None,
        });
    }
    finalizer.push(Stmt::If {
        cond: uid("#using_is_throw"),
        then_b: Box::new(Stmt::Throw(uid("#using_err"))),
        else_b: None,
    });

    out.push(Stmt::Try {
        block: try_body,
        handler: Some(handler),
        finalizer: Some(finalizer),
    });
    out
}

/// Rewrite `for (using ...)` heads:
/// - `for (using x = e; c; u) body` → `{ using x = e; for (; c; u) body }`
/// - `for (using x of/in y) body` →
///   `for (let #using_tmp of/in y) { using x = #using_tmp; body }`
///
/// The synthetic lists are desugared recursively.
fn rewrite_for_using(stmt: &mut Stmt) {
    let is_using_for = matches!(
        stmt,
        Stmt::For {
            init: Some(ForInit::Var(VarKind::Using | VarKind::AwaitUsing, _)),
            ..
        } | Stmt::ForInOf {
            kind: VarKind::Using | VarKind::AwaitUsing,
            ..
        }
    );
    if !is_using_for {
        return;
    }
    let old = std::mem::replace(stmt, Stmt::Empty);
    match old {
        Stmt::For {
            init: Some(ForInit::Var(kind, decls)),
            cond,
            update,
            body,
        } => {
            debug_assert!(matches!(kind, VarKind::Using | VarKind::AwaitUsing));
            let decl = Stmt::VarDecl { kind, decls };
            let for_stmt = Stmt::For {
                init: None,
                cond,
                update,
                body,
            };
            let mut block = vec![decl, for_stmt];
            desugar_stmts(&mut block);
            *stmt = Stmt::Block(block);
        }
        Stmt::ForInOf {
            kind,
            lhs,
            rhs,
            body,
            is_of,
        } => {
            debug_assert!(matches!(kind, VarKind::Using | VarKind::AwaitUsing));
            let pat = match lhs {
                ForLhs::Ident(name, span) => Pat::Ident(name, span),
                ForLhs::Pat(p) => p,
                // Unreachable: the parser only produces Ident/Pat with a
                // declaration kind. Restore and bail rather than panic.
                ForLhs::Member(e) => {
                    *stmt = Stmt::ForInOf {
                        kind,
                        lhs: ForLhs::Member(e),
                        rhs,
                        body,
                        is_of,
                    };
                    return;
                }
            };
            let using_decl = Stmt::VarDecl {
                kind,
                decls: vec![VarDecl {
                    id: pat,
                    init: Some(uid("#using_tmp")),
                    span: USING_SPAN,
                }],
            };
            let mut new_body = vec![using_decl, *body];
            desugar_stmts(&mut new_body);
            *stmt = Stmt::ForInOf {
                kind: VarKind::Let,
                lhs: ForLhs::Ident("#using_tmp".to_string(), USING_SPAN),
                rhs,
                body: Box::new(Stmt::Block(new_body)),
                is_of,
            };
        }
        old => *stmt = old,
    }
}

/// Recurse into a statement's child statement lists. Function and class
/// bodies are already desugared at parse time; expressions are never
/// entered (a `using` cannot appear in one).
fn desugar_nested(stmt: &mut Stmt) {
    match stmt {
        Stmt::Block(ss) => desugar_stmts(ss),
        Stmt::If { then_b, else_b, .. } => {
            desugar_single(then_b);
            if let Some(e) = else_b {
                desugar_single(e);
            }
        }
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => desugar_single(body),
        Stmt::For { body, .. } => desugar_single(body),
        Stmt::ForInOf { body, .. } => desugar_single(body),
        Stmt::Try {
            block,
            handler,
            finalizer,
        } => {
            desugar_stmts(block);
            if let Some(h) = handler {
                desugar_stmts(&mut h.body);
            }
            if let Some(f) = finalizer {
                desugar_stmts(f);
            }
        }
        Stmt::Switch { .. } => desugar_switch(stmt),
        Stmt::Labeled { body, .. } | Stmt::With { body, .. } => desugar_single(body),
        _ => {}
    }
}

/// Desugar a single-statement position (`if (x) <here>`). A block result is
/// spliced back as one statement.
fn desugar_single(s: &mut Box<Stmt>) {
    let mut v = vec![std::mem::replace(s.as_mut(), Stmt::Empty)];
    desugar_stmts(&mut v);
    **s = if v.len() == 1 {
        v.pop().expect("one statement")
    } else {
        Stmt::Block(v)
    };
}

/// Desugar `switch`: `using` in a case body belongs to the switch block's
/// disposal scope (fallthrough can reach later cases), so the records are
/// hoisted and the whole switch is wrapped.
fn desugar_switch(stmt: &mut Stmt) {
    let has_using = matches!(stmt,
        Stmt::Switch { cases, .. }
        if cases.iter().any(|c| c.body.iter().any(is_using_decl))
    );
    if !has_using {
        if let Stmt::Switch { cases, .. } = stmt {
            for case in cases.iter_mut() {
                desugar_stmts(&mut case.body);
            }
        }
        return;
    }
    let mut records: Vec<(String, bool)> = Vec::new();
    let mut counter = 0usize;
    if let Stmt::Switch { cases, .. } = stmt {
        for case in cases.iter_mut() {
            for s in case.body.iter_mut() {
                rewrite_for_using(s);
                desugar_nested(s);
            }
            let old_body = std::mem::take(&mut case.body);
            let mut new_body = Vec::with_capacity(old_body.len());
            for s in old_body {
                match s {
                    Stmt::VarDecl { kind, decls }
                        if matches!(kind, VarKind::Using | VarKind::AwaitUsing) =>
                    {
                        let is_async = kind == VarKind::AwaitUsing;
                        new_body.extend(expand_using_decl(
                            is_async,
                            decls,
                            &mut records,
                            &mut counter,
                        ));
                    }
                    s => new_body.push(s),
                }
            }
            case.body = new_body;
        }
    }
    if !records.is_empty() {
        let old = std::mem::replace(stmt, Stmt::Empty);
        *stmt = Stmt::Block(build_dispose_wrap(vec![old], records));
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::ast::{Expr, Stmt};

    fn parse_expr(src: &str) -> Expr {
        let prog = parse_script(src).expect("parse failed");
        match prog.body.into_iter().next().expect("empty") {
            Stmt::Expr(expr) => expr,
            s => panic!("not an expr stmt: {s:?}"),
        }
    }

    #[test]
    fn new_takes_arg_list() {
        // `new F(a)` gives the args to the new, not a call.
        match &parse_expr("new F(a)") {
            Expr::New { callee, args, .. } => {
                assert!(matches!(callee.as_ref(), Expr::Ident(n, _) if n == "F"));
                assert_eq!(args.len(), 1);
            }
            e => panic!("wrong shape: {e:?}"),
        }
    }

    #[test]
    fn new_call_suffix_applies_outside() {
        // `new F(a)(b)` is `(new F(a))(b)`.
        match &parse_expr("new F(a)(b)") {
            Expr::Call { callee, args, .. } => {
                assert_eq!(args.len(), 1);
                match callee.as_ref() {
                    Expr::New { args: new_args, .. } => assert_eq!(new_args.len(), 1),
                    e => panic!("inner not a new: {e:?}"),
                }
            }
            e => panic!("wrong shape: {e:?}"),
        }
    }

    #[test]
    fn new_paren_call_gets_no_args() {
        // `new (F(a))` is a new with no args on the call result.
        match &parse_expr("new (F(a))") {
            Expr::New { args, .. } => assert!(args.is_empty()),
            e => panic!("wrong shape: {e:?}"),
        }
    }

    #[test]
    fn nested_checkpoint_outer_rewind_restores_inner_tokens() {
        // `async (a, b) => }`: the inner paren-arrow speculation succeeds
        // (and is released), then the arrow body fails to parse, so the
        // outer async-arrow speculation rewinds. The re-parse must see
        // intact tokens: `async(a, b)` parses as a call and the error is
        // about the unexpected `=>`, never a spurious end-of-input from
        // unrestored placeholder tokens.
        match parse_script("async (a, b) => }") {
            Err(SyntaxError::EarlyError { message, .. }) => {
                assert!(message.contains("=>"), "wrong error: {message}");
            }
            r => panic!("wrong result: {r:?}"),
        }
    }

    #[test]
    fn paren_expr_after_failed_arrow_speculation() {
        // The failed `(a + b) =>` speculation must restore the taken
        // identifier tokens so the re-parse sees intact names.
        match &parse_expr("(alpha + beta) * gamma") {
            Expr::Binary {
                op, left, right, ..
            } => {
                assert_eq!(*op, BinaryOp::Mul);
                assert!(matches!(left.as_ref(), Expr::Binary { .. }));
                assert!(matches!(right.as_ref(), Expr::Ident(n, _) if n == "gamma"));
            }
            e => panic!("wrong shape: {e:?}"),
        }
    }

    #[test]
    fn arrow_params_keep_names() {
        match &parse_expr("(alpha, beta) => alpha + beta") {
            Expr::Arrow { params, .. } => {
                assert_eq!(params.len(), 2);
                assert!(matches!(&params[0].pat, Pat::Ident(n, _) if n == "alpha"));
                assert!(matches!(&params[1].pat, Pat::Ident(n, _) if n == "beta"));
            }
            e => panic!("wrong shape: {e:?}"),
        }
    }

    #[test]
    fn async_arrow_and_ident_fallback() {
        // `async (x) => x`
        assert!(matches!(parse_expr("async (x) => x"), Expr::Arrow { .. }));
        // `async x => x`
        assert!(matches!(parse_expr("async x => x"), Expr::Arrow { .. }));
        // `async (x)` — call of the identifier `async`, not an arrow.
        match &parse_expr("async (x)") {
            Expr::Call { callee, args, .. } => {
                assert!(matches!(callee.as_ref(), Expr::Ident(n, _) if n == "async"));
                assert_eq!(args.len(), 1);
            }
            e => panic!("wrong shape: {e:?}"),
        }
        // `async` alone is an identifier.
        assert!(matches!(
            &parse_expr("async"),
            Expr::Ident(n, _) if n == "async"
        ));
    }

    #[test]
    fn array_cover_fallback_keeps_names() {
        // `[alpha + beta]`: the `[pat = init]` cover attempt fails at `+`
        // and must rewind with identifier tokens intact.
        match &parse_expr("[alpha + beta]") {
            Expr::Array(elems, _) => {
                assert_eq!(elems.len(), 1);
                assert!(matches!(&elems[0], Some(Expr::Binary { .. })));
            }
            e => panic!("wrong shape: {e:?}"),
        }
    }

    #[test]
    fn nested_new_outer_gets_no_args() {
        // `new new F(a)` is `new (new F(a))`.
        match &parse_expr("new new F(a)") {
            Expr::New { callee, args, .. } => {
                assert!(args.is_empty());
                assert!(matches!(callee.as_ref(), Expr::New { .. }));
            }
            e => panic!("wrong shape: {e:?}"),
        }
    }

    #[test]
    fn exponentiation_rejects_unary() {
        // `-a ** b` is a syntax error (unparenthesized unary).
        assert!(parse_script("(-a) ** b").is_ok());
        assert!(parse_script("-a ** b").is_err());
    }

    #[test]
    fn exponentiation_right_assoc() {
        // `a ** b ** c` is `a ** (b ** c)`.
        match &parse_expr("a ** b ** c") {
            Expr::Binary {
                op: BinaryOp::Pow,
                right,
                ..
            } => {
                assert!(matches!(
                    right.as_ref(),
                    Expr::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            e => panic!("wrong shape: {e:?}"),
        }
    }

    /// DoS regression: deeply nested inputs must fail with a clean
    /// "maximum parse depth exceeded" error, never abort the process.
    /// Each trigger runs in a subprocess (see `dos_nesting_regressions`)
    /// so a regression kills only the child: the parent sees a failed
    /// exit status instead of the whole test harness dying. The trigger
    /// itself runs on an explicit 8 MiB thread because the depth guard
    /// is calibrated for a standard 8 MiB stack while the test harness
    /// uses smaller stacks.
    #[test]
    fn dos_probe_helper() {
        let trigger = match std::env::var("BOWSER_DOS_TRIGGER") {
            Ok(t) => t,
            // Not launched as a probe: nothing to do.
            Err(_) => return,
        };
        let h = std::thread::Builder::new()
            .name(format!("dos-probe-{trigger}"))
            .stack_size(8 * 1024 * 1024)
            .spawn(move || run_dos_trigger(&trigger))
            .expect("spawn dos probe thread");
        // A stack overflow aborts the process before join returns; a
        // panic here means the trigger produced the wrong result.
        h.join().expect("dos probe thread panicked");
    }

    fn run_dos_trigger(trigger: &str) {
        let src = match trigger {
            "paren300" => format!("({}1{};", "(".repeat(300), ")".repeat(300)),
            "class300" => {
                let mut s = String::from("Object");
                for i in 0..300 {
                    s = format!("class C{i} extends {s} {{}}");
                }
                format!("class Top extends {s} {{}};")
            }
            "pat50000" => format!("let {}x{} = 0;", "[".repeat(50_000), "]".repeat(50_000)),
            // Deeply nested literals: their native frames are far heavier
            // than one depth unit, so they use the heavy-literal cost.
            "obj3000" => {
                let mut s = String::from("var o = {a:");
                for _ in 0..3000 {
                    s.push_str("{a:");
                }
                s.push_str("42");
                for _ in 0..=3000 {
                    s.push('}');
                }
                s.push(';');
                s
            }
            "arr3000" => {
                let mut s = String::from("var a = [");
                for _ in 0..3000 {
                    s.push('[');
                }
                s.push_str("42");
                for _ in 0..=3000 {
                    s.push(']');
                }
                s.push(';');
                s
            }
            _ => panic!("unknown BOWSER_DOS_TRIGGER={trigger}"),
        };
        let err = parse_script(&src).expect_err("expected a parse-depth error");
        assert!(
            err.to_string().contains("maximum parse depth exceeded"),
            "wrong error for {trigger}: {err}"
        );
    }

    #[test]
    fn dos_nesting_regressions() {
        for trigger in ["paren300", "class300", "pat50000", "obj3000", "arr3000"] {
            let exe = std::env::current_exe().expect("test executable path");
            let out = std::process::Command::new(&exe)
                .arg("--exact")
                .arg("parser::tests::dos_probe_helper")
                .arg("--nocapture")
                .env("BOWSER_DOS_TRIGGER", trigger)
                .output()
                .expect("failed to spawn probe subprocess");
            assert!(
                out.status.success(),
                "trigger `{trigger}` did not exit cleanly (abort or assertion failure):\n--- stdout ---\n{}\n--- stderr ---\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
        }
    }

    #[test]
    fn class_body_is_always_strict() {
        // Class bodies are strict even in a sloppy script: `with`, legacy
        // octal, and unqualified `delete` are all early errors.
        for src in [
            "class C { m() { with (x) {} } }",
            "class C { m() { return 0755; } }",
            "class C { m() { delete x; } }",
            "class C { static { with (x) {} } }",
            "class C { constructor() { with (x) {} } }",
            "(class { m() { return 0755; } });",
            "var D = class { m() { delete y; } };",
        ] {
            assert!(parse_script(src).is_err(), "expected SyntaxError: {src}");
        }
        // The same constructs stay legal in sloppy non-class code.
        for src in ["with (x) {}", "var y = 0755;", "delete z;"] {
            assert!(parse_script(src).is_ok(), "expected Ok: {src}");
        }
    }

    #[test]
    fn strict_mode_rejects_legacy_octal() {
        for src in [
            "\"use strict\"; 0755;",
            "\"use strict\"; var x = 078;",
            "\"use strict\"; 08 + 1;",
            "function f() { \"use strict\"; return 0755; }",
            "\"use strict\"; var o = { 0755: 1 };",
            "\"use strict\"; var a = [08];",
        ] {
            assert!(parse_script(src).is_err(), "expected SyntaxError: {src}");
        }
        // Sloppy legacy octal still parses with its legacy value.
        match &parse_expr("0755") {
            Expr::Lit(crate::ast::Literal::Number(v), _) => {
                assert_eq!(*v, 493.0);
            }
            e => panic!("wrong shape: {e:?}"),
        }
        assert!(parse_script("var y = 078;").is_ok());
        // Modern `0o` octal is fine in strict mode.
        assert!(parse_script("\"use strict\"; var y = 0o755;").is_ok());
    }

    #[test]
    fn non_simple_params_reject_use_strict_directive() {
        for src in [
            "function f(a = 1) { \"use strict\"; }",
            "function f({a}) { \"use strict\"; }",
            "function f([a]) { \"use strict\"; }",
            "function f(...a) { \"use strict\"; }",
            "(function (a = 1) { \"use strict\"; });",
            "(a = 1) => { \"use strict\"; }",
            "async (a = 1) => { \"use strict\"; };",
            "class C { m(a = 1) { \"use strict\"; } }",
            "({ m(a = 1) { \"use strict\"; } });",
            "({ set x({a}) { \"use strict\"; } });",
            "\"use strict\"; function f(a = 1) { \"use strict\"; }",
        ] {
            assert!(parse_script(src).is_err(), "expected SyntaxError: {src}");
        }
        // Simple params with a directive, and non-simple params without
        // one, are both fine.
        for src in [
            "function f(a) { \"use strict\"; }",
            "function f(a = 1) {}",
            "function f({a}, ...rest) { return a; }",
            "(a = 1) => a;",
            "(a = 1) => { return a; }",
            "class C { m(a = 1) {} }",
        ] {
            assert!(parse_script(src).is_ok(), "expected Ok: {src}");
        }
    }

    #[test]
    fn using_decl_rejected_in_single_statement_position() {
        // Per ECMA-262, a lexical declaration (`using`/`await using` included)
        // cannot be the body of an `if`/`else`/`while`/`for` without braces.
        // The `using` desugar wraps statement *lists* in blocks; it must never
        // see a single-statement `using` because parsing rejects it first.
        for src in [
            "if (x) using a = d;",
            "if (x) using a = d; else {}",
            "if (x) {} else using a = d;",
            "if (x) if (y) using a = d;",
            "while (x) using a = d;",
            "do using a = d; while (x);",
            "for (;;) using a = d;",
            "for (let i = 0;;) using a = d;",
            "for (x of y) using a = d;",
            "for (x in y) using a = d;",
            "for (using x of y) using z = w;",
            "if (x) lbl: using a = d;",
            "while (x) lbl: using a = d;",
            "async function f() { if (x) await using a = d; }",
            "async function f() { while (x) await using a = d; }",
        ] {
            assert!(parse_script(src).is_err(), "expected SyntaxError: {src}");
        }
        // Braced bodies, `using` as an identifier, and `using` for-heads
        // stay legal.
        for src in [
            "if (x) { using a = d; }",
            "while (x) { using a = d; }",
            "for (;;) { using a = d; }",
            "if (x) using = 5;",
            "if (x) using.foo();",
            "for (using x of y) { let z = 1; }",
            "for (using x = d;;) {}",
            "lbl: using a = d;",
        ] {
            assert!(parse_script(src).is_ok(), "expected Ok: {src}");
        }
    }

    #[test]
    fn lexical_decl_rejected_in_single_statement_position() {
        // `let`/`const` are lexical declarations too: rejected as unbraced
        // `if`/`else`/`while`/`for` bodies, exactly like `using`. `var` is
        // not lexical and stays legal.
        for src in [
            "if (x) let a = 1;",
            "if (x) const a = 1;",
            "if (x) {} else let a = 1;",
            "while (x) let a = 1;",
            "do const a = 1; while (x);",
            "for (;;) let a = 1;",
            "for (x of y) const a = 1;",
            "if (x) let [a] = b;",
            "if (x) lbl: const a = 1;",
        ] {
            assert!(parse_script(src).is_err(), "expected SyntaxError: {src}");
        }
        for src in [
            "if (x) var a = 1;",
            "while (x) var a = 1;",
            "for (;;) var a = 1;",
            "if (x) { let a = 1; }",
            "if (x) let = 5;",
            "if (x) let.foo();",
        ] {
            assert!(parse_script(src).is_ok(), "expected Ok: {src}");
        }
        // Loop bodies are single-statement positions for declarations in
        // general: a function declaration there is an early error in strict
        // mode (Annex B.3.2 keeps it legal in sloppy mode).
        assert!(parse_script("\"use strict\"; while (x) function f() {}").is_err());
        assert!(parse_script("\"use strict\"; for (;;) function f() {}").is_err());
        assert!(parse_script("while (x) function f() {}").is_ok());
        assert!(parse_script("for (;;) function f() {}").is_ok());
    }

    #[test]
    fn class_body_strict_early_errors() {
        // Class bodies are strict-mode code even in a sloppy script
        // (spec 15.7.1): strict-only early errors fire for heritage,
        // methods, fields, and static blocks alike.
        for src in [
            // Duplicate parameter names (strict-only).
            "class C { m(a, a) {} }",
            "class C { m(a, a = 1) {} }",
            "class C { static m(a, a) {} }",
            "var D = class { constructor(a, a) {} };",
            // Heritage is strict code too.
            "class C extends (delete z, Object) {}",
            "class C extends (0755, Object) {}",
            // Fields: initializers and computed keys are strict.
            "class C { x = 0755; }",
            "class C { [0755] = 1; }",
            "class C { x = delete y; }",
            // eval/arguments may not be bound or assigned.
            "class C { static { let eval = 1; } }",
            "class C { m() { let arguments = 1; } }",
            "class C { m() { eval = 1; } }",
            "class eval {}",
            "class C { m() { with (x) {} } }",
            "class C { get x() { with (y) {} } }",
            "class A { m() { class B { n() { with (x) {} } } } }",
        ] {
            assert!(parse_script(src).is_err(), "expected SyntaxError: {src}");
        }
        // Strictness does not leak out of the class: sloppy code after a
        // class keeps sloppy semantics.
        for src in [
            "class C {} with (x) {}",
            "class C { m() {} } delete z;",
            "class C {} var y = 0755;",
            "var o = { m(a, a) {} };",
            "function f(a, a) {}",
        ] {
            assert!(parse_script(src).is_ok(), "expected Ok: {src}");
        }
    }
}
