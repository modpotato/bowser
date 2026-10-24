//! Selectors: parsing and specificity.
//!
//! [`Selector`] is a single parsed complex selector (one comma group of a
//! selector list). Fields stay private; the cascade/matching step lives in
//! this crate and can reach the internals directly.

use std::error::Error;
use std::fmt;

use bowser_dom::Namespace;

use crate::token::{Token, tokenize};

/// A parsed complex selector: compound selectors joined by combinators.
#[derive(Clone, Debug, PartialEq)]
pub struct Selector {
    pub(crate) compounds: Vec<CompoundSelector>,
    /// `combinators[i]` sits between `compounds[i]` and `compounds[i+1]`.
    pub(crate) combinators: Vec<Combinator>,
}

/// One compound selector: an optional type/universal selector plus any
/// number of id, class, attribute, and pseudo-class selectors, and at most
/// one pseudo-element.
#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct CompoundSelector {
    pub(crate) tag: Option<TagSelector>,
    pub(crate) ids: Vec<String>,
    pub(crate) classes: Vec<String>,
    pub(crate) attrs: Vec<AttrSelector>,
    pub(crate) pseudos: Vec<PseudoClass>,
    pub(crate) pseudo_element: Option<String>,
}

/// A type selector (`div`) or universal selector (`*`), with an optional
/// namespace prefix (`svg|div`, `*|*`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TagSelector {
    /// `None` when no prefix was written (`div`, `*`): matches any
    /// namespace. `Some(..)` records an explicit qualifier.
    pub(crate) namespace: Option<NamespaceConstraint>,
    /// `None` for the universal selector (`*`, `*|*`).
    pub(crate) name: Option<String>,
}

/// A namespace qualifier on a type or attribute selector.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum NamespaceConstraint {
    /// `*|div`, `[*|title]`: any namespace, including none.
    Any,
    /// `|div`, `[|title]`: elements with no namespace. Every element in
    /// this DOM carries a namespace, so this never matches.
    NoNamespace,
    /// `svg|div`, `[svg|viewBox]`: the written prefix, resolved by
    /// [`resolve_namespace_prefix`]. Unknown prefixes never match.
    Named(String),
}

/// Resolve a namespace prefix against the built-in prefix table. There is
/// no `@namespace` support, so only the well-known prefixes exist:
/// `html`, `svg`, `math`/`mml` (ASCII case-insensitive).
pub(crate) fn resolve_namespace_prefix(prefix: &str) -> Option<Namespace> {
    if prefix.eq_ignore_ascii_case("html") {
        Some(Namespace::Html)
    } else if prefix.eq_ignore_ascii_case("svg") {
        Some(Namespace::Svg)
    } else if prefix.eq_ignore_ascii_case("math") || prefix.eq_ignore_ascii_case("mml") {
        Some(Namespace::MathMl)
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Combinator {
    Descendant,
    Child,
    NextSibling,
    SubsequentSibling,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AttrSelector {
    /// Optional `ns|` prefix; unknown prefixes never match.
    pub(crate) namespace: Option<NamespaceConstraint>,
    pub(crate) name: String,
    pub(crate) op: AttrOp,
    pub(crate) value: String,
    /// `[attr=value i]`: ASCII case-insensitive value comparison.
    pub(crate) insensitive: bool,
    /// Whether `value` contains ASCII whitespace, precomputed at parse
    /// time. A whitespace-containing value can never match `~=`; hoisting
    /// the scan out of the matcher keeps it off the hot path.
    pub(crate) value_has_ws: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum AttrOp {
    Exists,
    Equals,
    Includes,
    DashMatch,
    Prefix,
    Suffix,
    Substring,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PseudoClass {
    Root,
    FirstChild,
    LastChild,
    /// `:nth-child(an+b)`: the (a, b) coefficients.
    NthChild(i32, i32),
    /// `:nth-last-child(an+b)`: the (a, b) coefficients.
    NthLastChild(i32, i32),
    /// `:nth-of-type(an+b)`: the (a, b) coefficients.
    NthOfType(i32, i32),
    /// `:nth-last-of-type(an+b)`: the (a, b) coefficients.
    NthLastOfType(i32, i32),
    FirstOfType,
    LastOfType,
    OnlyChild,
    OnlyOfType,
    Empty,
    /// Dynamic pseudo-classes: parsed and flagged, matching decides.
    Link,
    Visited,
    Hover,
    Active,
    Focus,
    /// `:checked`: checked checkbox/radio inputs and selected options.
    Checked,
    /// `:disabled` / `:enabled`: disableable form controls.
    Disabled,
    Enabled,
    /// `:not()` with a forgiving complex-selector list inside (Selectors
    /// 4). Specificity is the most specific argument's, like `:is()`.
    Not(Vec<Selector>),
    /// `:is()` with a forgiving selector list inside. Specificity is the
    /// most specific argument's.
    Is(Vec<Selector>),
    /// `:where()` with a forgiving selector list inside. Contributes zero
    /// specificity.
    Where(Vec<Selector>),
    /// `:has()` with a forgiving relative-selector list inside.
    /// Specificity is the most specific argument's.
    Has(Vec<RelativeSelector>),
}

/// A relative selector inside `:has()`: an optional leading combinator
/// (defaulting to the descendant combinator) plus a complex selector.
/// Matching anchors the complex selector's leftmost compound to the
/// `:has()` anchor element (the Selectors 4 "absolutize" rule).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RelativeSelector {
    pub(crate) combinator: Combinator,
    pub(crate) selector: Selector,
}

/// Selector parse failure; the whole rule is dropped.
#[derive(Debug)]
pub struct SelectorParseError {
    message: String,
}

impl fmt::Display for SelectorParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid selector: {}", self.message)
    }
}

impl Error for SelectorParseError {}

fn sel_err(message: impl Into<String>) -> SelectorParseError {
    SelectorParseError {
        message: message.into(),
    }
}

/// Maximum functional-pseudo-class nesting depth (`:not()`, `:is()`,
/// `:where()`, `:has()`). Each nesting level recurses the parser (and
/// later the matcher); stylesheets are attacker-controlled, so depth is
/// capped instead of letting hostile input like `:not(a:not(a:…))` × 3000
/// overflow the stack and abort the process. Anything deeper invalidates
/// the whole selector (the rule is dropped); inside the forgiving
/// `:is()`/`:where()`/`:has()` lists an over-deep group is just dropped.
/// The cap also bounds matcher recursion, which only recurses through
/// these pseudo-classes.
const MAX_FUNC_NESTING: u32 = 64;

/// Maximum compound count of the relative selector itself inside `:has()`
/// (its own compounds only; nested functional-pseudo arguments have their
/// own budget, [`MAX_HAS_NESTED_COMPOUNDS`]).
///
/// The relative-selector search (`dfs_relative` in `matching.rs`) costs
/// O(k x subtree^2) per anchor for k compounds, with an uncapped
/// failed-state memo holding up to k x subtree entries — and capping
/// the memo is not an option: evicting recorded failures re-expands
/// them exponentially, turning the polynomial search back into an
/// exponential one (a memo cap was tried and reverted for exactly this
/// reason). So the bound lives here, at parse time: k is capped instead
/// of the memo, the matcher keeps its uncapped (and therefore correct)
/// memo, and the residual worst case is O(16 x subtree^2) per anchor
/// with the memo bounded by 16 x subtree entries. Real-world `:has()`
/// selectors use a handful of compounds (`:has(> img)`,
/// `:has(.menu .item)`); 16 leaves wide headroom while making hostile
/// shapes infeasible.
///
/// Over-cap groups are dropped exactly like any other invalid group of
/// the forgiving `:has()` list; a list with no surviving group
/// invalidates the selector (the existing rule). `:has()` nesting needs
/// no depth cap on top: it cannot nest inside itself at all (rejected
/// outright via `SelCursor::in_has` in `parse_pseudo`).
const MAX_HAS_RELATIVE_COMPOUNDS: usize = 16;

/// Maximum total compounds nested inside the `:is()`/`:not()`/`:where()`
/// arguments of one `:has()` relative selector, counted recursively
/// (see [`nested_compound_count`]).
///
/// Why a separate, larger budget, and why this shape: nested arguments do
/// not grow the relative search structure — the `dfs_relative`
/// O(k x subtree^2) shape and its k x subtree memo are driven by the
/// relative selector's own compounds, capped by
/// [`MAX_HAS_RELATIVE_COMPOUNDS`]. Instead the nested arguments multiply
/// the cost of each candidate compound check: `matches_compound`
/// evaluates every nested alternative with `matches_cascade`, a full
/// complex-selector match of the argument against the same node
/// (short-circuiting on the first success), costing O(c x D) per
/// alternative of c compounds over ancestor depth D <= subtree. Per anchor
/// that is O(k x subtree^2) candidate checks each carrying O(n x subtree)
/// nested work in the absolute worst case — O(k x n x subtree^3),
/// polynomial with a fixed constant: 16 x 256 = 4096 here. A per-argument
/// cap alone would not bound this multiplier (an `:is()` with 10k
/// one-compound alternatives has 1 compound per argument), so the budget
/// is a single total over every nested compound, counted recursively
/// through nested functional pseudos; nesting depth itself is separately
/// capped by [`MAX_FUNC_NESTING`].
///
/// 256 is 16x the relative-compound cap: one nesting level buys an order
/// of magnitude more headroom than any realistic selector needs (nested
/// alternatives are typically one-compound class or type matches), while
/// keeping the worst-case constant small and the parse-time counting walk
/// linear in the already-parsed group.
const MAX_HAS_NESTED_COMPOUNDS: usize = 256;

/// Own compound count of a `:has()` relative selector: the compounds of
/// the relative selector itself, not counting anything nested in
/// `:is()`/`:not()`/`:where()` arguments (those have the separate
/// [`MAX_HAS_NESTED_COMPOUNDS`] budget).
fn relative_selector_compound_count(sel: &Selector) -> usize {
    sel.compounds.len()
}

/// Total compounds nested inside the `:is()`/`:not()`/`:where()` arguments
/// of a `:has()` relative selector, counted recursively: an `:is()` inside
/// an `:is()` argument still counts. A 10k-compound `:is()` buried in a
/// single `:has()` compound would otherwise dodge
/// [`MAX_HAS_RELATIVE_COMPOUNDS`] while costing just as much per
/// candidate. (`:has()` cannot nest, so there is no `Has` arm to count.)
/// The walk is linear in the already-parsed selector, so even a 10k-compound
/// argument is counted (and rejected) at parse time, fast.
fn nested_compound_count(sel: &Selector) -> usize {
    fn count(sel: &Selector) -> usize {
        let mut n = 0;
        for c in &sel.compounds {
            for p in &c.pseudos {
                match p {
                    PseudoClass::Not(args) | PseudoClass::Is(args) | PseudoClass::Where(args) => {
                        n += args
                            .iter()
                            .map(|arg| arg.compounds.len() + count(arg))
                            .sum::<usize>();
                    }
                    _ => {}
                }
            }
        }
        n
    }
    count(sel)
}

/// Maximum compound count of any single complex selector.
///
/// Every complex selector funnels through [`parse_complex`]: top-level
/// rule selectors, `:is()`/`:not()`/`:where()` arguments, and `:has()`
/// relative selectors (which keep their own tighter
/// [`MAX_HAS_RELATIVE_COMPOUNDS`] cap on top).
///
/// The bound exists for `match_complex`'s per-call failed-state memo
/// (`matching.rs`): it holds up to compounds x nodes entries at ~19.4
/// B each, and capping the memo is not an option — the wave-27 lesson
/// stands, evicting recorded failures re-expands them exponentially.
/// With compounds uncapped, a 10^5-compound selector over 10^3 nodes
/// pins ~2 GB transient per anchor from the stylesheet alone. So the
/// bound lives here, at parse time: the memo stays uncapped (and
/// therefore correct), and the residual worst case is 64 x nodes
/// entries per match call with the search itself still polynomial.
///
/// Grading mirrors the wave-28 `:has()` caps: at top level an over-cap
/// complex selector is a parse error and the rule is dropped, exactly
/// like any other invalid selector; inside the forgiving
/// `:is()`/`:not()`/`:where()`/`:has()` lists an over-cap group is
/// just dropped, and only a list with no surviving group invalidates
/// the selector.
///
/// 64 is 4x the `:has()` relative cap: real-world selectors use a
/// handful of compounds, so this leaves wide headroom while making
/// hostile shapes infeasible.
const MAX_COMPLEX_COMPOUNDS: usize = 64;

/// Pseudo-elements this engine recognizes. Anything else (e.g.
/// `::bogus`) invalidates the whole selector. Matching never matches a
/// pseudo-element against a real element; this list only governs which
/// selectors parse.
const SUPPORTED_PSEUDO_ELEMENTS: &[&str] = &[
    "before",
    "after",
    "first-line",
    "first-letter",
    "selection",
    "marker",
    "placeholder",
    "backdrop",
];

/// Parse a comma-separated selector list.
pub fn parse_selector_list(input: &str) -> Result<Vec<Selector>, SelectorParseError> {
    // Raw-text pre-check: rejects decimal An+B coefficients (`2.0`) with
    // a precise error before tokenizing. The token path now carries the
    // `<integer>`-lexeme flag itself (see [`Token::Number`]), so this is
    // defense in depth; both entry points agree.
    check_nth_fn_args(input)?;
    let tokens = tokenize(input);
    parse_selector_list_from_tokens(&tokens)
}

pub(crate) fn parse_selector_list_from_tokens(
    tokens: &[Token],
) -> Result<Vec<Selector>, SelectorParseError> {
    let mut out = Vec::with_capacity(1);
    for group in split_comma_groups(tokens) {
        let mut cur = SelCursor::new(group);
        cur.skip_ws();
        if cur.peek().is_none() {
            return Err(sel_err("empty selector in list"));
        }
        out.push(parse_complex(&mut cur)?);
        cur.skip_ws();
        if cur.peek().is_some() {
            return Err(sel_err("trailing tokens after selector"));
        }
    }
    Ok(out)
}

/// Split a token slice at top-level commas. `depth` tracks `()`/`[]` so
/// commas inside `:not()`, `:is()`, attribute values, etc. don't split.
fn split_comma_groups(tokens: &[Token]) -> Vec<&[Token]> {
    let mut groups: Vec<&[Token]> = Vec::new();
    let mut start = 0;
    let mut depth: u32 = 0;
    for (i, t) in tokens.iter().enumerate() {
        match t {
            Token::LParen | Token::Function(_) | Token::LBracket => depth += 1,
            Token::RParen | Token::RBracket => depth = depth.saturating_sub(1),
            Token::Comma if depth == 0 => {
                groups.push(&tokens[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    groups.push(&tokens[start..]);
    groups
}

/// Specificity per Selectors L4: (ids, classes/attrs/pseudo-classes,
/// type selectors/pseudo-elements). `:not()`, `:is()`, and `:has()`
/// contribute their most specific argument's specificity; `:where()`
/// contributes zero.
pub fn specificity(selector: &Selector) -> (u32, u32, u32) {
    let mut spec = (0u32, 0u32, 0u32);
    for compound in &selector.compounds {
        add_compound_specificity(compound, &mut spec);
    }
    spec
}

/// Whether the selector uses any combinator (descendant, child, or
/// sibling): such a selector can only match when the subject element's
/// ancestor/sibling context is present, so a partial document rooted
/// below that context cannot apply it.
pub fn has_combinator(selector: &Selector) -> bool {
    !selector.combinators.is_empty()
}

fn add_compound_specificity(c: &CompoundSelector, spec: &mut (u32, u32, u32)) {
    spec.0 += c.ids.len() as u32;
    spec.1 += (c.classes.len() + c.attrs.len()) as u32;
    for p in &c.pseudos {
        match p {
            // `:not()` and `:is()` contribute their most specific
            // argument; `:has()` its most specific relative selector;
            // `:where()` contributes nothing.
            PseudoClass::Not(args) | PseudoClass::Is(args) => {
                let (a, b, cc) = max_specificity(args.iter());
                spec.0 += a;
                spec.1 += b;
                spec.2 += cc;
            }
            PseudoClass::Has(args) => {
                let (a, b, cc) = max_specificity(args.iter().map(|r| &r.selector));
                spec.0 += a;
                spec.1 += b;
                spec.2 += cc;
            }
            PseudoClass::Where(_) => {}
            _ => spec.1 += 1,
        }
    }
    if c.tag.as_ref().is_some_and(|t| t.name.is_some()) {
        spec.2 += 1;
    }
    if c.pseudo_element.is_some() {
        spec.2 += 1;
    }
}

/// The greatest specificity in an argument list, compared as a whole
/// tuple (lexicographic `(a, b, c)` ordering is exactly specificity
/// ordering). Empty lists contribute zero; the parser never produces
/// them, but the matcher must not panic on one either.
fn max_specificity<'a, I>(selectors: I) -> (u32, u32, u32)
where
    I: Iterator<Item = &'a Selector>,
{
    selectors.map(specificity).max().unwrap_or((0, 0, 0))
}

struct SelCursor<'a> {
    toks: &'a [Token],
    pos: usize,
    /// Functional-pseudo-class nesting depth of this cursor (`:not()`,
    /// `:is()`, `:where()`, `:has()`); see [`MAX_FUNC_NESTING`].
    nesting_depth: u32,
    /// Whether this cursor is inside a `:has()` argument. `:has()` may
    /// not nest inside itself.
    in_has: bool,
}

impl<'a> SelCursor<'a> {
    fn new(toks: &'a [Token]) -> Self {
        SelCursor {
            toks,
            pos: 0,
            nesting_depth: 0,
            in_has: false,
        }
    }

    /// `Token::Eof` reads as end-of-input.
    fn peek(&self) -> Option<&'a Token> {
        match self.toks.get(self.pos) {
            None | Some(Token::Eof) => None,
            t => t,
        }
    }

    fn peek_at(&self, off: usize) -> Option<&'a Token> {
        match self.toks.get(self.pos + off) {
            None | Some(Token::Eof) => None,
            t => t,
        }
    }

    fn bump(&mut self) -> Option<&'a Token> {
        let t = self.peek()?;
        self.pos += 1;
        Some(t)
    }

    fn skip_ws(&mut self) -> bool {
        let mut any = false;
        while matches!(self.peek(), Some(Token::Whitespace)) {
            self.pos += 1;
            any = true;
        }
        any
    }
}

fn parse_complex(cur: &mut SelCursor) -> Result<Selector, SelectorParseError> {
    let mut compounds = vec![parse_compound(cur)?];
    let mut combinators = Vec::new();
    loop {
        let had_ws = cur.skip_ws();
        let comb = match cur.peek() {
            None | Some(Token::Comma) => break,
            Some(Token::Delim('>')) => {
                cur.bump();
                Combinator::Child
            }
            Some(Token::Delim('+')) => {
                cur.bump();
                Combinator::NextSibling
            }
            Some(Token::Delim('~')) => {
                cur.bump();
                Combinator::SubsequentSibling
            }
            _ if had_ws => Combinator::Descendant,
            _ => return Err(sel_err("expected combinator between compounds")),
        };
        combinators.push(comb);
        cur.skip_ws();
        compounds.push(parse_compound(cur)?);
    }
    // A pseudo-element may only end the complex selector (`div::before p`
    // is invalid); `parse_compound` already forces it last in its own
    // compound.
    if compounds[..compounds.len() - 1]
        .iter()
        .any(|c| c.pseudo_element.is_some())
    {
        return Err(sel_err("pseudo-element must end the complex selector"));
    }
    // Compound-count cap (see [`MAX_COMPLEX_COMPOUNDS`]): bounds
    // `match_complex`'s per-call failed-state memo at parse time. Every
    // complex selector passes through here, so this covers top-level
    // rules (parse error: the rule is dropped) as well as
    // `:is()`/`:not()`/`:where()` arguments and `:has()` relative
    // selectors (over-cap group dropped by the forgiving list).
    if compounds.len() > MAX_COMPLEX_COMPOUNDS {
        return Err(sel_err(format!(
            "complex selector has more than {MAX_COMPLEX_COMPOUNDS} compounds"
        )));
    }
    Ok(Selector {
        compounds,
        combinators,
    })
}

fn bump_ident(cur: &mut SelCursor) -> Result<String, SelectorParseError> {
    match cur.bump() {
        Some(Token::Ident(n)) => Ok(n.clone()),
        _ => Err(sel_err("expected identifier")),
    }
}

fn parse_compound(cur: &mut SelCursor) -> Result<CompoundSelector, SelectorParseError> {
    let mut c = CompoundSelector::default();
    let mut kind_seen = false; // type selector or universal seen
    let mut rest_seen = false; // any id/class/attr/pseudo seen
    loop {
        match cur.peek() {
            None
            | Some(Token::Whitespace)
            | Some(Token::Comma)
            | Some(Token::Delim('>' | '+' | '~'))
            | Some(Token::RParen) => break,
            // A pseudo-element must be the last component of its compound
            // selector (`div::before:hover` is invalid).
            _ if c.pseudo_element.is_some() => {
                return Err(sel_err("pseudo-element must be last in its compound"));
            }
            Some(Token::Ident(_)) => {
                if kind_seen || rest_seen {
                    return Err(sel_err("type selector must come first in a compound"));
                }
                let first = bump_ident(cur)?;
                // Optional `prefix|` qualifier (`svg|div`, `svg|*`).
                let (namespace, name) = if cur.peek() == Some(&Token::Delim('|'))
                    && matches!(
                        cur.peek_at(1),
                        Some(Token::Ident(_)) | Some(Token::Delim('*'))
                    ) {
                    cur.bump(); // '|'
                    let namespace = Some(NamespaceConstraint::Named(first));
                    match cur.bump() {
                        Some(Token::Ident(n2)) => (namespace, Some(n2.clone())),
                        Some(Token::Delim('*')) => (namespace, None),
                        _ => return Err(sel_err("expected name or '*' after '|'")),
                    }
                } else {
                    (None, Some(first))
                };
                c.tag = Some(TagSelector { namespace, name });
                kind_seen = true;
            }
            Some(Token::Delim('*')) => {
                if kind_seen || rest_seen {
                    return Err(sel_err("universal selector must come first in a compound"));
                }
                cur.bump();
                // Optional `*|` qualifier (`*|div`, `*|*`).
                let (namespace, name) = if cur.peek() == Some(&Token::Delim('|')) {
                    cur.bump(); // '|'
                    let namespace = Some(NamespaceConstraint::Any);
                    match cur.bump() {
                        Some(Token::Ident(n)) => (namespace, Some(n.clone())),
                        Some(Token::Delim('*')) => (namespace, None),
                        _ => return Err(sel_err("expected name or '*' after '*|'")),
                    }
                } else {
                    (None, None)
                };
                c.tag = Some(TagSelector { namespace, name });
                kind_seen = true;
            }
            Some(Token::Delim('|')) => {
                if kind_seen || rest_seen {
                    return Err(sel_err("unexpected '|' in selector"));
                }
                cur.bump();
                // `|div`: no namespace (never matches here; every element
                // has a namespace).
                let (namespace, name) = match cur.bump() {
                    Some(Token::Ident(n)) => {
                        (Some(NamespaceConstraint::NoNamespace), Some(n.clone()))
                    }
                    Some(Token::Delim('*')) => (Some(NamespaceConstraint::NoNamespace), None),
                    _ => return Err(sel_err("expected name or '*' after '|'")),
                };
                c.tag = Some(TagSelector { namespace, name });
                kind_seen = true;
            }
            Some(Token::Hash(_, _)) => {
                let (id, is_id) = match cur.bump() {
                    Some(Token::Hash(n, flag)) => (n.clone(), *flag),
                    _ => return Err(sel_err("expected id selector")),
                };
                if !is_id {
                    return Err(sel_err("invalid id selector"));
                }
                c.ids.push(id);
                rest_seen = true;
            }
            Some(Token::Delim('.')) => {
                cur.bump();
                match cur.bump() {
                    Some(Token::Ident(cls)) => c.classes.push(cls.clone()),
                    _ => return Err(sel_err("expected class name after '.'")),
                }
                rest_seen = true;
            }
            Some(Token::LBracket) => {
                c.attrs.push(parse_attr(cur)?);
                rest_seen = true;
            }
            Some(Token::Colon) => {
                parse_pseudo(cur, &mut c)?;
                rest_seen = true;
            }
            _ => return Err(sel_err("unexpected token in selector")),
        }
    }
    if !kind_seen && !rest_seen {
        return Err(sel_err("empty compound selector"));
    }
    Ok(c)
}

fn parse_attr(cur: &mut SelCursor) -> Result<AttrSelector, SelectorParseError> {
    cur.bump(); // '['
    cur.skip_ws();
    // Optional namespace prefix: `[ns|attr]`, `[*|attr]`, `[|attr]`.
    // The `|` here must be followed by an ident, otherwise it's the `|=`
    // operator handled below.
    let mut namespace: Option<NamespaceConstraint> = None;
    let name = match (cur.peek(), cur.peek_at(1), cur.peek_at(2)) {
        (Some(Token::Delim('*')), Some(Token::Delim('|')), Some(Token::Ident(_))) => {
            cur.bump();
            cur.bump(); // '*' '|'
            namespace = Some(NamespaceConstraint::Any);
            bump_ident(cur)?
        }
        (Some(Token::Delim('|')), Some(Token::Ident(_)), _) => {
            cur.bump(); // '|'
            namespace = Some(NamespaceConstraint::NoNamespace);
            bump_ident(cur)?
        }
        (Some(Token::Ident(_)), Some(Token::Delim('|')), Some(Token::Ident(_))) => {
            let prefix = bump_ident(cur)?;
            cur.bump(); // '|'
            namespace = Some(NamespaceConstraint::Named(prefix));
            bump_ident(cur)?
        }
        _ => bump_ident(cur)?,
    };
    cur.skip_ws();
    let op = match cur.bump() {
        Some(Token::RBracket) => {
            return Ok(AttrSelector {
                namespace,
                name,
                op: AttrOp::Exists,
                value: String::new(),
                insensitive: false,
                value_has_ws: false,
            });
        }
        Some(Token::Delim('=')) => AttrOp::Equals,
        Some(Token::Delim(d)) => {
            let op = match d {
                '~' => AttrOp::Includes,
                '|' => AttrOp::DashMatch,
                '^' => AttrOp::Prefix,
                '$' => AttrOp::Suffix,
                '*' => AttrOp::Substring,
                _ => return Err(sel_err("invalid attribute operator")),
            };
            match cur.bump() {
                Some(Token::Delim('=')) => op,
                _ => return Err(sel_err("expected '=' after attribute operator")),
            }
        }
        _ => return Err(sel_err("expected attribute operator or ']'")),
    };
    cur.skip_ws();
    let value = match cur.bump() {
        Some(Token::Ident(v)) => v.clone(),
        Some(Token::String(v)) => v.clone(),
        _ => return Err(sel_err("expected attribute value")),
    };
    cur.skip_ws();
    // Optional case-sensitivity flag: `i` = ASCII case-insensitive, `s` =
    // explicit case-sensitive (the default).
    let mut insensitive = false;
    if let Some(Token::Ident(flag)) = cur.peek()
        && (flag.eq_ignore_ascii_case("i") || flag.eq_ignore_ascii_case("s"))
    {
        insensitive = flag.eq_ignore_ascii_case("i");
        cur.bump();
        cur.skip_ws();
    }
    match cur.bump() {
        Some(Token::RBracket) => {
            let value_has_ws = value.bytes().any(|b| b.is_ascii_whitespace());
            Ok(AttrSelector {
                namespace,
                name,
                op,
                value,
                insensitive,
                value_has_ws,
            })
        }
        _ => Err(sel_err("expected ']'")),
    }
}

fn parse_pseudo(cur: &mut SelCursor, c: &mut CompoundSelector) -> Result<(), SelectorParseError> {
    cur.bump(); // first ':'
    if cur.peek() == Some(&Token::Colon) {
        cur.bump();
        let name = match cur.bump() {
            Some(Token::Ident(n)) => n.to_ascii_lowercase(),
            _ => return Err(sel_err("expected pseudo-element name")),
        };
        set_pseudo_element(cur, c, name)?;
        return Ok(());
    }
    match cur.bump() {
        Some(Token::Ident(name)) => {
            let n = name.to_ascii_lowercase();
            match n.as_str() {
                "root" => c.pseudos.push(PseudoClass::Root),
                "first-child" => c.pseudos.push(PseudoClass::FirstChild),
                "last-child" => c.pseudos.push(PseudoClass::LastChild),
                "first-of-type" => c.pseudos.push(PseudoClass::FirstOfType),
                "last-of-type" => c.pseudos.push(PseudoClass::LastOfType),
                "only-child" => c.pseudos.push(PseudoClass::OnlyChild),
                "only-of-type" => c.pseudos.push(PseudoClass::OnlyOfType),
                "empty" => c.pseudos.push(PseudoClass::Empty),
                "link" => c.pseudos.push(PseudoClass::Link),
                "visited" => c.pseudos.push(PseudoClass::Visited),
                "hover" => c.pseudos.push(PseudoClass::Hover),
                "active" => c.pseudos.push(PseudoClass::Active),
                "focus" => c.pseudos.push(PseudoClass::Focus),
                "checked" => c.pseudos.push(PseudoClass::Checked),
                "disabled" => c.pseudos.push(PseudoClass::Disabled),
                "enabled" => c.pseudos.push(PseudoClass::Enabled),
                // Legacy single-colon pseudo-elements.
                "before" | "after" | "first-line" | "first-letter" | "selection" | "marker"
                | "placeholder" | "backdrop" => {
                    set_pseudo_element(cur, c, n)?;
                }
                _ => return Err(sel_err(format!("unknown pseudo-class ':{n}'"))),
            }
            Ok(())
        }
        Some(Token::Function(fname)) => {
            let f = fname.to_ascii_lowercase();
            let inner = collect_until_rparen(cur)?;
            match f.as_str() {
                "not" => {
                    // Selectors 4: `:not()` takes a forgiving selector
                    // list, like `:is()`. Invalid comma groups are
                    // dropped; only no-valid-argument-at-all fails.
                    let args = parse_forgiving_selector_list(cur, &inner, "not")?;
                    c.pseudos.push(PseudoClass::Not(args));
                }
                "nth-child" => {
                    let (a, b) = parse_nth_from_tokens(&inner)
                        .ok_or_else(|| sel_err("invalid :nth-child() argument"))?;
                    c.pseudos.push(PseudoClass::NthChild(a, b));
                }
                "nth-last-child" | "nth-of-type" | "nth-last-of-type" => {
                    let (a, b) = parse_nth_from_tokens(&inner)
                        .ok_or_else(|| sel_err(format!("invalid :{f}() argument")))?;
                    let pc = match f.as_str() {
                        "nth-last-child" => PseudoClass::NthLastChild(a, b),
                        "nth-of-type" => PseudoClass::NthOfType(a, b),
                        _ => PseudoClass::NthLastOfType(a, b),
                    };
                    c.pseudos.push(pc);
                }
                "is" => {
                    let args = parse_forgiving_selector_list(cur, &inner, "is")?;
                    c.pseudos.push(PseudoClass::Is(args));
                }
                "where" => {
                    let args = parse_forgiving_selector_list(cur, &inner, "where")?;
                    c.pseudos.push(PseudoClass::Where(args));
                }
                "has" => {
                    if cur.in_has {
                        return Err(sel_err(":has() cannot be nested inside :has()"));
                    }
                    let args = parse_forgiving_relative_list(cur, &inner)?;
                    c.pseudos.push(PseudoClass::Has(args));
                }
                _ => return Err(sel_err(format!("unknown functional pseudo-class ':{f}()'"))),
            }
            Ok(())
        }
        _ => Err(sel_err("expected pseudo-class name")),
    }
}

/// Parse a forgiving selector list for `:is()`/`:where()`: comma groups
/// that fail to parse are dropped, and only an argument with no valid
/// group at all invalidates the selector. Pseudo-elements are rejected
/// in the arguments (via the nesting depth check in
/// [`set_pseudo_element`]).
fn parse_forgiving_selector_list(
    cur: &SelCursor,
    tokens: &[Token],
    name: &str,
) -> Result<Vec<Selector>, SelectorParseError> {
    let mut out = Vec::new();
    for group in split_comma_groups(tokens) {
        // Too deep to recurse: drop the group instead of risking the
        // stack. The forgiving list only fails when nothing parses.
        if cur.nesting_depth >= MAX_FUNC_NESTING {
            continue;
        }
        let mut gc = SelCursor::new(group);
        gc.nesting_depth = cur.nesting_depth + 1;
        gc.in_has = cur.in_has;
        gc.skip_ws();
        if let Ok(sel) = parse_complex(&mut gc) {
            gc.skip_ws();
            if gc.peek().is_none() {
                out.push(sel);
            }
        }
    }
    if out.is_empty() {
        return Err(sel_err(format!(":{name}() has no valid selector argument")));
    }
    Ok(out)
}

/// Parse a forgiving relative-selector list for `:has()`: each comma
/// group is an optional leading combinator (default: descendant) plus a
/// complex selector. Invalid groups are dropped; no valid group
/// invalidates the selector. Groups over [`MAX_HAS_RELATIVE_COMPOUNDS`]
/// relative compounds, or over [`MAX_HAS_NESTED_COMPOUNDS`] nested
/// `:is()`/`:not()`/`:where()` compounds, are dropped as invalid groups:
/// either would make the `:has()` search's polynomial worst case
/// infeasible (see the constants' docs for the work bound).
fn parse_forgiving_relative_list(
    cur: &SelCursor,
    tokens: &[Token],
) -> Result<Vec<RelativeSelector>, SelectorParseError> {
    let mut out = Vec::new();
    for group in split_comma_groups(tokens) {
        if cur.nesting_depth >= MAX_FUNC_NESTING {
            continue;
        }
        if let Ok(rel) = parse_relative_selector(cur, group)
            && relative_selector_compound_count(&rel.selector) <= MAX_HAS_RELATIVE_COMPOUNDS
            && nested_compound_count(&rel.selector) <= MAX_HAS_NESTED_COMPOUNDS
        {
            out.push(rel);
        }
    }
    if out.is_empty() {
        return Err(sel_err(":has() has no valid relative selector argument"));
    }
    Ok(out)
}

/// Parse one relative selector: an optional leading combinator, then a
/// complex selector. The returned cursor runs one nesting level deeper
/// with `:has()` nesting flagged.
fn parse_relative_selector(
    cur: &SelCursor,
    tokens: &[Token],
) -> Result<RelativeSelector, SelectorParseError> {
    let mut rc = SelCursor::new(tokens);
    rc.nesting_depth = cur.nesting_depth + 1;
    rc.in_has = true;
    rc.skip_ws();
    let combinator = match rc.peek() {
        Some(Token::Delim('>')) => {
            rc.bump();
            Combinator::Child
        }
        Some(Token::Delim('+')) => {
            rc.bump();
            Combinator::NextSibling
        }
        Some(Token::Delim('~')) => {
            rc.bump();
            Combinator::SubsequentSibling
        }
        _ => Combinator::Descendant,
    };
    rc.skip_ws();
    let selector = parse_complex(&mut rc)?;
    rc.skip_ws();
    if rc.peek().is_some() {
        return Err(sel_err("trailing tokens in :has() argument"));
    }
    Ok(RelativeSelector {
        combinator,
        selector,
    })
}

/// Record a pseudo-element on the compound, validating it: unknown names
/// are rejected, and pseudo-elements are not allowed inside functional
/// pseudo-classes (`:not()`, `:is()`, `:where()`, `:has()`).
/// (Position — last in the compound — is enforced by `parse_compound`.)
fn set_pseudo_element(
    cur: &SelCursor,
    c: &mut CompoundSelector,
    name: String,
) -> Result<(), SelectorParseError> {
    if !SUPPORTED_PSEUDO_ELEMENTS.contains(&name.as_str()) {
        return Err(sel_err(format!("unknown pseudo-element '::{name}'")));
    }
    if cur.nesting_depth > 0 {
        return Err(sel_err(
            "pseudo-elements are not allowed inside functional pseudo-classes",
        ));
    }
    if c.pseudo_element.is_some() {
        return Err(sel_err("multiple pseudo-elements in one compound"));
    }
    c.pseudo_element = Some(name);
    Ok(())
}

/// Consume tokens until the matching `)` (already inside the function paren).
fn collect_until_rparen(cur: &mut SelCursor) -> Result<Vec<Token>, SelectorParseError> {
    let mut out = Vec::new();
    let mut depth: u32 = 1;
    loop {
        match cur.bump() {
            None => return Err(sel_err("unclosed function in selector")),
            Some(Token::LParen) | Some(Token::Function(_)) => {
                depth += 1;
                out.push(cur.toks[cur.pos - 1].clone());
            }
            Some(Token::RParen) => {
                depth -= 1;
                if depth == 0 {
                    return Ok(out);
                }
                out.push(Token::RParen);
            }
            Some(t) => out.push(t.clone()),
        }
    }
}

/// Parse `:nth-child()` arguments from tokens.
///
/// This is a direct token-slice matcher for the `an+b` microsyntax — it
/// does not rebuild text, because the tokenizer erases information text
/// rebuilding needs (e.g. `2n+1` becomes `[Dimension(2,"n"),
/// Number(1.0)]`, losing the `+`). It implements the same language as
/// [`parse_nth_str`]; the `&str` entry point runs both, so they must
/// agree.
///
/// Note: numeric tokens carry their `<integer>`-lexeme flag (see
/// [`Token::Number`]), so this path rejects decimal An+B coefficients
/// (`:nth-child(2.0)`, `:nth-child(2.0n)`, `:nth-child(2n+1.5)`) exactly
/// like the raw-text [`check_nth_fn_args`] pre-check the `&str` entry
/// point runs.
fn parse_nth_from_tokens(tokens: &[Token]) -> Option<(i32, i32)> {
    let toks: Vec<&Token> = tokens
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace))
        .collect();
    let int_val = |f: f32| num_int_wide(f);
    match toks.as_slice() {
        [Token::Ident(s)] => {
            let low = s.to_ascii_lowercase();
            if low == "odd" {
                return Some((2, 1));
            }
            if low == "even" {
                return Some((2, 0));
            }
            let (a, b) = ident_anb(s)?;
            Some((i32::try_from(a).ok()?, i32::try_from(b.unwrap_or(0)).ok()?))
        }
        [
            Token::Number {
                value: f,
                integer: true,
            },
        ] => Some((0, i32::try_from(int_val(*f)?).ok()?)),
        [
            Token::Dimension {
                value: f,
                integer: true,
                unit: u,
            },
        ] => {
            let (a, b) = dim_anb(*f, u)?;
            match b {
                DimB::Bare => Some((i32::try_from(a).ok()?, 0)),
                DimB::Val(v) => Some((i32::try_from(a).ok()?, i32::try_from(v).ok()?)),
                DimB::Neg | DimB::Pos => None,
            }
        }
        // `2n+1`, `2n 1`, `2n- 1`, `2n+ 1` (the `+`/`-` may be glued to
        // the number token or the dimension unit).
        [
            Token::Dimension {
                value: f,
                integer: true,
                unit: u,
            },
            Token::Number {
                value: g,
                integer: true,
            },
        ] => {
            let (a, b) = dim_anb(*f, u)?;
            let g = int_val(*g)?;
            let bv = match b {
                DimB::Bare => g,
                DimB::Neg => -g,
                DimB::Pos => g,
                DimB::Val(_) => return None,
            };
            Some((i32::try_from(a).ok()?, i32::try_from(bv).ok()?))
        }
        // `2n + 1`, `2n - 1`, `2n- + 1`.
        [
            Token::Dimension {
                value: f,
                integer: true,
                unit: u,
            },
            Token::Delim(d),
            Token::Number {
                value: g,
                integer: true,
            },
        ] => {
            let (a, b) = dim_anb(*f, u)?;
            let g = int_val(*g)?;
            if g < 0 {
                return None; // `<n-dimension> <delim> <signless-integer>`
            }
            let bv = match (b, d) {
                (DimB::Val(_), _) => return None,
                (DimB::Bare, '+') | (DimB::Neg, '-') | (DimB::Pos, '+') => g,
                (DimB::Bare, '-') | (DimB::Neg, '+') | (DimB::Pos, '-') => -g,
                _ => return None,
            };
            Some((i32::try_from(a).ok()?, i32::try_from(bv).ok()?))
        }
        // `n + 1`, `-n - 3`.
        [
            Token::Ident(s),
            Token::Delim(d),
            Token::Number {
                value: g,
                integer: true,
            },
        ] => {
            let (a, b) = ident_anb(s)?;
            if b.is_some() {
                return None;
            }
            let g = int_val(*g)?;
            if g < 0 {
                return None;
            }
            let bv = match d {
                '+' => g,
                '-' => -g,
                _ => return None,
            };
            Some((i32::try_from(a).ok()?, i32::try_from(bv).ok()?))
        }
        // `n 1`, `-n -3`.
        [
            Token::Ident(s),
            Token::Number {
                value: g,
                integer: true,
            },
        ] => {
            let (a, b) = ident_anb(s)?;
            if b.is_some() {
                return None;
            }
            let g = int_val(*g)?;
            Some((i32::try_from(a).ok()?, i32::try_from(g).ok()?))
        }
        // `+n`, `+n + 1`, `+n 2`.
        [Token::Delim('+'), Token::Ident(s)] if s.eq_ignore_ascii_case("n") => Some((1, 0)),
        [
            Token::Delim('+'),
            Token::Ident(s),
            Token::Delim(d),
            Token::Number {
                value: g,
                integer: true,
            },
        ] if s.eq_ignore_ascii_case("n") => {
            let g = int_val(*g)?;
            if g < 0 {
                return None;
            }
            let bv = match d {
                '+' => g,
                '-' => -g,
                _ => return None,
            };
            Some((1, i32::try_from(bv).ok()?))
        }
        [
            Token::Delim('+'),
            Token::Ident(s),
            Token::Number {
                value: g,
                integer: true,
            },
        ] if s.eq_ignore_ascii_case("n") => {
            let g = int_val(*g)?;
            Some((1, i32::try_from(g).ok()?))
        }
        _ => None,
    }
}

/// An `f32` number token as an exact `i64` when it is integral and fits.
/// Allows up to 2^31 so a negated `-2147483648` still works; callers
/// narrow to `i32` with `try_from`.
fn num_int_wide(f: f32) -> Option<i64> {
    if f.fract() != 0.0 {
        return None;
    }
    if !(-2147483648.0..=2147483648.0).contains(&f) {
        return None;
    }
    Some(f as i64)
}

/// Parse an ident of the form `[+-]?<digits>n([+-]<digits>)?`
/// (ASCII case-insensitive `n`): `n`, `-n`, `2N`, `n-3`, `-n+3`.
/// Returns `(a, b)` with `b = None` when absent. `odd`/`even` are handled
/// by the caller.
fn ident_anb(s: &str) -> Option<(i64, Option<i64>)> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut neg = false;
    if b.get(i) == Some(&b'+') || b.get(i) == Some(&b'-') {
        neg = b[i] == b'-';
        i += 1;
    }
    let ds = i;
    while b.get(i).is_some_and(|c| c.is_ascii_digit()) {
        i += 1;
    }
    let digits = &s[ds..i];
    if !b.get(i).is_some_and(|c| *c == b'n' || *c == b'N') {
        return None;
    }
    let a: i64 = if digits.is_empty() {
        if neg { -1 } else { 1 }
    } else {
        let v: i64 = digits.parse().ok()?;
        if neg { -v } else { v }
    };
    i += 1;
    let bo = if i < b.len() {
        let bneg = match b[i] {
            b'+' => false,
            b'-' => true,
            _ => return None,
        };
        i += 1;
        let ds2 = i;
        while b.get(i).is_some_and(|c| c.is_ascii_digit()) {
            i += 1;
        }
        if ds2 == i {
            return None;
        }
        let v: i64 = s[ds2..i].parse().ok()?;
        Some(if bneg { -v } else { v })
    } else {
        None
    };
    if i != b.len() {
        return None;
    }
    Some((a, bo))
}

/// What follows the `n` in a dimension unit.
#[derive(Clone, Copy, PartialEq)]
enum DimB {
    /// `2n`: no b part yet.
    Bare,
    /// `2n-`: b's sign is `-`, digits follow as a separate token.
    Neg,
    /// `2n+`: b's sign is `+`, digits follow as a separate token.
    Pos,
    /// `2n-3`: complete b value.
    Val(i64),
}

/// Parse a dimension's `(a, b-part)` from its value and unit: `2n`,
/// `2n-`, `2n+`, `2n-3` (unit ASCII case-insensitive).
fn dim_anb(f: f32, unit: &str) -> Option<(i64, DimB)> {
    let a = num_int_wide(f)?;
    let ul = unit.to_ascii_lowercase();
    let rest = ul.strip_prefix('n')?;
    let b = match rest {
        "" => DimB::Bare,
        "-" => DimB::Neg,
        "+" => DimB::Pos,
        _ => {
            let (sign, digits) = rest
                .strip_prefix('-')
                .map(|d| (-1i64, d))
                .or_else(|| rest.strip_prefix('+').map(|d| (1i64, d)))?;
            if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
                return None;
            }
            let v: i64 = digits.parse().ok()?;
            DimB::Val(sign * v)
        }
    };
    Some((a, b))
}

/// Strict parser for the `an+b` microsyntax (CSS Syntax §8.1).
///
/// Accepts `odd`, `even`, a bare `<integer>`, and `<a>n` with an optional
/// b part: `[+-]<digits>` (whitespace allowed around it) or bare
/// `<digits>` after separating whitespace (`2n 1`). Everything else is
/// rejected: `n2` and `2n2` (a digit directly after `n`), `2.0` (a
/// `<number>`, not an `<integer>`), `n+m`, `bogus`, …
///
/// Values are computed in `i64` and must fit `i32`; out-of-range values
/// invalidate the selector rather than wrapping.
fn parse_nth_str(input: &str) -> Option<(i32, i32)> {
    let t = input.trim();
    let low = t.to_ascii_lowercase();
    if low == "odd" {
        return Some((2, 1));
    }
    if low == "even" {
        return Some((2, 0));
    }
    let b = t.as_bytes();
    let mut i = 0;
    let mut neg = false;
    if b.get(i) == Some(&b'+') || b.get(i) == Some(&b'-') {
        neg = b[i] == b'-';
        i += 1;
    }
    let ds = i;
    while b.get(i).is_some_and(|c| c.is_ascii_digit()) {
        i += 1;
    }
    let digits = &t[ds..i];
    if b.get(i).is_some_and(|c| *c == b'n' || *c == b'N') {
        // `<a>n` with optional b part: `[+-]<digits>` (whitespace allowed
        // around it) or bare `<digits>` after separating whitespace
        // (`2n 1`). A digit directly after `n` (`n2`, `2n2`) is invalid.
        let a: i64 = if digits.is_empty() {
            if neg { -1 } else { 1 }
        } else {
            let v: i64 = digits.parse().ok()?;
            if neg { -v } else { v }
        };
        i += 1;
        let mut bv: i64 = 0;
        if i < b.len() {
            let saw_ws = matches!(b[i], b' ' | b'\t' | b'\n' | b'\r' | 0x0C);
            i = skip_css_ws(b, i);
            let bneg = match b.get(i) {
                Some(b'+') => {
                    i += 1;
                    false
                }
                Some(b'-') => {
                    i += 1;
                    true
                }
                // Unsigned b needs separating whitespace.
                _ if saw_ws => false,
                _ => return None,
            };
            i = skip_css_ws(b, i);
            let ds2 = i;
            while b.get(i).is_some_and(|c| c.is_ascii_digit()) {
                i += 1;
            }
            if ds2 == i {
                return None;
            }
            let v: i64 = t[ds2..i].parse().ok()?;
            bv = if bneg { -v } else { v };
            i = skip_css_ws(b, i);
        }
        if i != b.len() {
            return None;
        }
        Some((i32::try_from(a).ok()?, i32::try_from(bv).ok()?))
    } else {
        // Bare `<integer>`: ASCII digits only — no `.`, no exponent.
        if digits.is_empty() || i != b.len() {
            return None;
        }
        let v: i64 = digits.parse().ok()?;
        Some((0, i32::try_from(if neg { -v } else { v }).ok()?))
    }
}

fn skip_css_ws(b: &[u8], mut i: usize) -> usize {
    while b
        .get(i)
        .is_some_and(|c| matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0C))
    {
        i += 1;
    }
    i
}

/// Raw-text pre-validation of `:nth-child()` / `:nth-last-child()` /
/// `:nth-of-type()` / `:nth-last-of-type()` arguments.
///
/// The An+B microsyntax only allows `<integer>` tokens in most positions,
/// so `:nth-child(2.0)` must invalidate the whole selector. Numeric
/// tokens now carry their `<integer>`-lexeme flag (see [`Token::Number`]),
/// so the token path rejects decimals on its own; this scan works on the
/// raw source as a precise pre-check: it finds `:nth-*(...)` occurrences
/// outside strings and comments (resolving CSS escapes in the function
/// name) and strictly validates each argument with [`parse_nth_str`].
/// Shape errors are still caught from the token stream as well.
///
/// Inside the forgiving functional pseudo-classes (`:is()`, `:where()`,
/// `:has()`, `:not()`) an invalid `:nth-*()` does *not* invalidate the
/// selector: the group is dropped forgivingly by the token path instead.
/// This keeps the `&str` entry point consistent with
/// `parse_selector_list_from_tokens` (used by the stylesheet parser).
///
/// Only runs on the `&str` entry point; the token path enforces the same
/// rule for stylesheets.
fn check_nth_fn_args(input: &str) -> Result<(), SelectorParseError> {
    let b = input.as_bytes();
    let mut i = 0;
    // Paren depths at which a forgiving functional pseudo (`:is(`,
    // `:where(`, `:has(`, `:not(`) was opened; innermost last.
    let mut forgiving: Vec<u32> = Vec::new();
    let mut depth: u32 = 0;
    while i < b.len() {
        match b[i] {
            b'\'' | b'"' => i = skip_css_string(b, i),
            b'/' if b.get(i + 1) == Some(&b'*') => i = skip_css_comment(b, i),
            b'\\' => i = skip_css_escape(b, i),
            b'(' => {
                depth += 1;
                i += 1;
            }
            b')' => {
                depth = depth.saturating_sub(1);
                while forgiving.last() == Some(&depth) {
                    forgiving.pop();
                }
                i += 1;
            }
            b':' => {
                if let Some((name, arg, next)) = match_nth_fn_at(b, i) {
                    // match_nth_fn_at skips a balanced function, so depth
                    // is unchanged.
                    if parse_nth_str(&arg).is_none() && forgiving.is_empty() {
                        return Err(sel_err(format!("invalid :{name}() argument")));
                    }
                    i = next;
                } else if let Some(next) = match_forgiving_fn_at(b, i) {
                    forgiving.push(depth);
                    depth += 1;
                    i = next;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }
    Ok(())
}

/// If `b[i]` is the `:` of a `:is(`, `:where(`, `:has(`, or `:not(`
/// occurrence, return the index just past the `(`.
fn match_forgiving_fn_at(b: &[u8], i: usize) -> Option<usize> {
    let mut j = skip_ws_comments(b, i + 1);
    // Function name with CSS escapes resolved, ASCII-lowercased.
    let mut name = String::new();
    loop {
        match b.get(j) {
            Some(b'\\') => {
                let (ch, nj) = decode_css_escape(b, j)?;
                name.push(ch.to_ascii_lowercase());
                j = nj;
            }
            Some(c) if c.is_ascii_alphanumeric() || *c == b'-' || *c == b'_' || *c >= 0x80 => {
                name.push((*c as char).to_ascii_lowercase());
                j += 1;
            }
            _ => break,
        }
    }
    if !matches!(name.as_str(), "is" | "where" | "has" | "not") {
        return None;
    }
    j = skip_ws_comments(b, j);
    if b.get(j) != Some(&b'(') {
        return None;
    }
    Some(j + 1)
}

/// If `b[i]` is the `:` of a `:nth-child(`, `:nth-last-child(`,
/// `:nth-of-type(`, or `:nth-last-of-type(` occurrence, return the
/// function name, the raw argument text, and the index just past the
/// matching `)`.
fn match_nth_fn_at(b: &[u8], i: usize) -> Option<(String, String, usize)> {
    let mut j = skip_ws_comments(b, i + 1);
    // Function name with CSS escapes resolved, ASCII-lowercased.
    let mut name = String::new();
    loop {
        match b.get(j) {
            Some(b'\\') => {
                let (ch, nj) = decode_css_escape(b, j)?;
                name.push(ch.to_ascii_lowercase());
                j = nj;
            }
            Some(c) if c.is_ascii_alphanumeric() || *c == b'-' || *c == b'_' || *c >= 0x80 => {
                name.push((*c as char).to_ascii_lowercase());
                j += 1;
            }
            _ => break,
        }
    }
    if !matches!(
        name.as_str(),
        "nth-child" | "nth-last-child" | "nth-of-type" | "nth-last-of-type"
    ) {
        return None;
    }
    j = skip_ws_comments(b, j);
    if b.get(j) != Some(&b'(') {
        return None;
    }
    let arg_start = j + 1;
    let mut depth = 1u32;
    let mut k = arg_start;
    while k < b.len() {
        match b[k] {
            b'\'' | b'"' => k = skip_css_string(b, k),
            b'/' if b.get(k + 1) == Some(&b'*') => k = skip_css_comment(b, k),
            b'\\' => k = skip_css_escape(b, k),
            b'(' => {
                depth += 1;
                k += 1;
            }
            b')' => {
                depth -= 1;
                if depth == 0 {
                    let arg = String::from_utf8_lossy(&b[arg_start..k]).into_owned();
                    return Some((name, arg, k + 1));
                }
                k += 1;
            }
            _ => k += 1,
        }
    }
    None // unbalanced; the token parser reports it
}

fn skip_ws_comments(b: &[u8], mut j: usize) -> usize {
    loop {
        j = skip_css_ws(b, j);
        if b.get(j) == Some(&b'/') && b.get(j + 1) == Some(&b'*') {
            j = skip_css_comment(b, j);
        } else {
            return j;
        }
    }
}

fn skip_css_string(b: &[u8], mut i: usize) -> usize {
    let q = b[i];
    i += 1;
    while i < b.len() {
        match b[i] {
            b'\\' => i += 2,
            c if c == q => return i + 1,
            _ => i += 1,
        }
    }
    i
}

fn skip_css_comment(b: &[u8], mut i: usize) -> usize {
    i += 2; // `/*`
    while i < b.len() {
        if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
            return i + 2;
        }
        i += 1;
    }
    i
}

/// Skip a CSS escape starting at `b[i]` (which is `\`).
fn skip_css_escape(b: &[u8], i: usize) -> usize {
    let (_, next) = decode_css_escape(b, i).unwrap_or(('\0', i + 1));
    next.min(b.len())
}

/// Decode a CSS escape at `b[j]` (`j` points at `\`): hex escapes
/// (`\68 `) and single-char escapes (`\:`). Returns the decoded char and
/// the index past the escape, or `None` on a truncated escape.
fn decode_css_escape(b: &[u8], j: usize) -> Option<(char, usize)> {
    let mut k = j + 1;
    if k < b.len() && b[k].is_ascii_hexdigit() {
        let mut val: u32 = 0;
        for _ in 0..6 {
            if k < b.len() && b[k].is_ascii_hexdigit() {
                val = val * 16 + (b[k] as char).to_digit(16).unwrap_or(0);
                k += 1;
            } else {
                break;
            }
        }
        if k < b.len() && matches!(b[k], b' ' | b'\t' | b'\n' | b'\r' | 0x0C) {
            k += 1;
        }
        if val == 0 || val > char::MAX as u32 {
            val = 0xFFFD;
        }
        Some((char::from_u32(val).unwrap_or('\u{FFFD}'), k))
    } else if k < b.len() {
        // Single-char escape; non-ASCII bytes cannot spell "nth-child",
        // so a lossy mapping is fine for name matching.
        Some((b[k] as char, k + 1))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sel(input: &str) -> Selector {
        let list = parse_selector_list(input).expect("selector should parse");
        assert_eq!(list.len(), 1);
        list.into_iter().next().unwrap()
    }

    fn spec(input: &str) -> (u32, u32, u32) {
        specificity(&sel(input))
    }

    #[test]
    fn basic_selectors_parse() {
        sel("div");
        sel("*");
        sel("#main");
        sel(".container");
        sel("div#main.container");
        sel("*.x");
        sel("div p");
        sel("div > p");
        sel("div + p");
        sel("div ~ p");
        sel("ul li a");
        sel("a:hover");
        sel("::before");
        sel(":before"); // legacy single-colon pseudo-element
        sel(":root");
        sel("p:first-child");
        sel("p:last-child");
        sel("li:nth-child(2n+1)");
        sel("div:empty");
        sel("a:link");
        sel(":not(.x)");
        sel("[href]");
        sel("[type=text]");
        sel("[class~=a]");
        sel("[lang|=en]");
        sel("[href^=https]");
        sel("[href$=\".pdf\"]"); // unquoted `.pdf` would be invalid: idents can't start with '.'
        sel("[title*=hello]");
        sel("[data-x=\"a b\"]");
    }

    #[test]
    fn comma_groups() {
        let list = parse_selector_list("div, .a, #b").unwrap();
        assert_eq!(list.len(), 3);
        let list = parse_selector_list("div > p, a + b.c").unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn selector_errors() {
        assert!(parse_selector_list("").is_err());
        assert!(parse_selector_list("   ").is_err());
        assert!(parse_selector_list("div,").is_err());
        assert!(parse_selector_list(",div").is_err());
        assert!(parse_selector_list("div..x").is_err());
        assert!(parse_selector_list("div#").is_err());
        assert!(parse_selector_list("div >").is_err());
        assert!(parse_selector_list("[a").is_err());
        assert!(parse_selector_list("[a=").is_err());
        assert!(parse_selector_list("[a=b").is_err());
        assert!(parse_selector_list("div:bogus").is_err());
        assert!(parse_selector_list("div:bogus()").is_err());
        assert!(parse_selector_list("divdiv span").is_ok()); // descendant, fine
        assert!(parse_selector_list("div>span").is_ok());
        assert!(parse_selector_list("#1").is_err()); // invalid id
        assert!(parse_selector_list(":nth-child(bogus)").is_err());
        assert!(parse_selector_list(":not()").is_err());
        assert!(parse_selector_list("::before::after").is_err());
    }

    #[test]
    fn error_is_display_and_error() {
        let e = parse_selector_list("div:bogus").unwrap_err();
        let s = format!("{e}");
        assert!(s.contains("bogus"));
        let _: &dyn Error = &e;
        let d = format!("{e:?}");
        assert!(d.contains("SelectorParseError"));
    }

    #[test]
    fn nth_child_forms() {
        assert!(matches!(
            sel("li:nth-child(odd)").compounds[0].pseudos[0],
            PseudoClass::NthChild(2, 1)
        ));
        assert!(matches!(
            sel("li:nth-child(even)").compounds[0].pseudos[0],
            PseudoClass::NthChild(2, 0)
        ));
        assert!(matches!(
            sel("li:nth-child(3)").compounds[0].pseudos[0],
            PseudoClass::NthChild(0, 3)
        ));
        assert!(matches!(
            sel("li:nth-child(n)").compounds[0].pseudos[0],
            PseudoClass::NthChild(1, 0)
        ));
        assert!(matches!(
            sel("li:nth-child(-n+3)").compounds[0].pseudos[0],
            PseudoClass::NthChild(-1, 3)
        ));
        assert!(matches!(
            sel("li:nth-child(2n + 1)").compounds[0].pseudos[0],
            PseudoClass::NthChild(2, 1)
        ));
        assert!(matches!(
            sel("li:nth-child(+3n-2)").compounds[0].pseudos[0],
            PseudoClass::NthChild(3, -2)
        ));
    }

    #[test]
    fn parse_nth_unit() {
        assert_eq!(parse_nth_str("odd"), Some((2, 1)));
        assert_eq!(parse_nth_str("even"), Some((2, 0)));
        assert_eq!(parse_nth_str("2n+1"), Some((2, 1)));
        assert_eq!(parse_nth_str("-n+3"), Some((-1, 3)));
        assert_eq!(parse_nth_str("n"), Some((1, 0)));
        assert_eq!(parse_nth_str("5"), Some((0, 5)));
        assert_eq!(parse_nth_str("-5"), Some((0, -5)));
        assert_eq!(parse_nth_str("2N"), Some((2, 0)));
        assert_eq!(parse_nth_str("bogus"), None);
        assert_eq!(parse_nth_str("n+m"), None);
        // Strict microsyntax: these are all invalid.
        assert_eq!(parse_nth_str("n2"), None); // single ident, not an+b
        assert_eq!(parse_nth_str("2n2"), None);
        assert_eq!(parse_nth_str("2.0"), None); // <number>, not <integer>
        assert_eq!(parse_nth_str("+2.5n"), None);
        assert_eq!(parse_nth_str("1e3"), None);
        assert_eq!(parse_nth_str(""), None);
        assert_eq!(parse_nth_str("n+"), None);
        assert_eq!(parse_nth_str("2n++1"), None);
        assert_eq!(parse_nth_str("--n"), None);
        // Valid forms the old lax parser also accepted, plus more.
        assert_eq!(parse_nth_str("+n"), Some((1, 0)));
        assert_eq!(parse_nth_str("-n"), Some((-1, 0)));
        assert_eq!(parse_nth_str("-n-3"), Some((-1, -3)));
        assert_eq!(parse_nth_str("n-3"), Some((1, -3)));
        assert_eq!(parse_nth_str("3n-2"), Some((3, -2)));
        assert_eq!(parse_nth_str("2n + 1"), Some((2, 1)));
        assert_eq!(parse_nth_str("  -2n - 1  "), Some((-2, -1)));
        assert_eq!(parse_nth_str("+3n-2"), Some((3, -2)));
        assert_eq!(parse_nth_str("0"), Some((0, 0)));
        assert_eq!(parse_nth_str("2n 1"), Some((2, 1)));
        assert_eq!(parse_nth_str("2n - 1"), Some((2, -1)));
        assert_eq!(parse_nth_str("n-1"), Some((1, -1)));
        assert_eq!(parse_nth_str("n - 1"), Some((1, -1)));
        // i32::MIN is exactly representable; larger magnitudes are invalid.
        assert_eq!(parse_nth_str("n-2147483648"), Some((1, i32::MIN)));
        assert_eq!(parse_nth_str("n-2147483649"), None);
        assert_eq!(parse_nth_str("2147483648"), None);
    }

    #[test]
    fn nth_child_token_shapes() {
        // `parse_nth_from_tokens` shares the strict grammar with the raw-text
        // parser; spot-check tokenization-shaped inputs.
        use crate::token::tokenize;
        fn from(arg: &str) -> Option<(i32, i32)> {
            let toks: Vec<Token> = tokenize(arg)
                .into_iter()
                .filter(|t| !matches!(t, Token::Whitespace | Token::Eof))
                .collect();
            parse_nth_from_tokens(&toks)
        }
        assert_eq!(from("2n+1"), Some((2, 1)));
        assert_eq!(from("2n + 1"), Some((2, 1)));
        assert_eq!(from("2n - 1"), Some((2, -1)));
        assert_eq!(from("2n 1"), Some((2, 1)));
        assert_eq!(from("-n-3"), Some((-1, -3)));
        assert_eq!(from("-n + 3"), Some((-1, 3)));
        assert_eq!(from("+n"), Some((1, 0)));
        assert_eq!(from("n2"), None);
        assert_eq!(from("2n2"), None);
        // Numeric tokens carry the `<integer>`-lexeme flag, so the token
        // path rejects decimal coefficients exactly like the raw-text
        // pre-scan.
        assert_eq!(from("2.0"), None);
        assert_eq!(from("2.0n"), None);
        assert_eq!(from("2n+1.5"), None);
        assert_eq!(from("2n + 1.0"), None);
        assert_eq!(from("2"), Some((0, 2)));
        assert_eq!(from("odd"), Some((2, 1)));
        assert_eq!(from("3n - 2"), Some((3, -2)));
    }

    #[test]
    fn nth_child_invalid_args_rejected() {
        for bad in [
            "li:nth-child(n2)",
            "li:nth-child(2n2)",
            "li:nth-child(2.0)",
            "li:nth-child(bogus)",
            "li:nth-child(n+m)",
            "li:nth-child()",
            "li:nth-child(2n+)",
        ] {
            assert!(parse_selector_list(bad).is_err(), "{bad} should be invalid");
        }
        for good in [
            "li:nth-child(2n+1)",
            "li:nth-child(odd)",
            "li:nth-child(3)",
            "li:nth-child(-n+3)",
            "li:nth-child(2n + 1)",
            "li:nth-child(n-2147483648)",
        ] {
            assert!(parse_selector_list(good).is_ok(), "{good} should parse");
        }
    }

    #[test]
    fn nth_child_raw_scan_ignores_strings() {
        // `:nth-child(` inside an attribute string is not a real occurrence.
        assert!(parse_selector_list("[title=\":nth-child(2.0)\"]").is_ok());
        // …but a real one with a bad argument still fails, even nested.
        assert!(parse_selector_list(":not(:nth-child(2.0))").is_err());
        // CSS escapes in the function name are resolved by the scan.
        assert!(parse_selector_list(":nt\\68 -child(2.0)").is_err());
        assert!(parse_selector_list(":nt\\68 -child(2n)").is_ok());
    }

    #[test]
    fn specificity_basics() {
        assert_eq!(spec("*"), (0, 0, 0));
        assert_eq!(spec("div"), (0, 0, 1));
        assert_eq!(spec(".a"), (0, 1, 0));
        assert_eq!(spec("#a"), (1, 0, 0));
        assert_eq!(spec("div#a.b.c"), (1, 2, 1));
        assert_eq!(spec("ul li a"), (0, 0, 3));
        assert_eq!(spec("[type=text]"), (0, 1, 0));
        assert_eq!(spec("a:hover"), (0, 1, 1));
        assert_eq!(spec("::before"), (0, 0, 1));
        assert_eq!(spec("div::before"), (0, 0, 2));
        assert_eq!(spec(":root"), (0, 1, 0));
        assert_eq!(spec("li:nth-child(2n+1)"), (0, 1, 1));
    }

    #[test]
    fn specificity_not() {
        // :not() itself adds nothing; its most specific argument does.
        assert_eq!(spec(":not(#x)"), (1, 0, 0));
        assert_eq!(spec("div:not(.a)"), (0, 1, 1));
        assert_eq!(spec(":not(div.a)"), (0, 1, 1));
        assert_eq!(spec(":not(.a, #x)"), (1, 0, 0));
        assert_eq!(spec(":not(.a, .b)"), (0, 1, 0));
    }

    #[test]
    fn not_takes_forgiving_list() {
        // Selectors 4: `:not()` accepts a comma-separated forgiving
        // selector list; invalid groups are dropped.
        sel(":not(.a, .b)");
        sel("div:not(.a, #b)");
        sel(":not(.a, bogus##)"); // invalid group dropped, `.a` remains
        assert!(parse_selector_list(":not()").is_err());
        assert!(parse_selector_list(":not(bogus##)").is_err()); // no valid group
    }

    #[test]
    fn not_nesting_depth_capped() {
        // 3000-deep `:not()` nesting must fail the selector, not abort the
        // process with a stack overflow.
        let deep = ":not(a".repeat(3000) + &")".repeat(3000);
        assert!(parse_selector_list(&deep).is_err());
        // Just under the cap still parses.
        let ok = ":not(a".repeat(63) + &")".repeat(63);
        assert!(parse_selector_list(&ok).is_ok());
        // …and just over it does not.
        let over = ":not(a".repeat(65) + &")".repeat(65);
        assert!(parse_selector_list(&over).is_err());
    }

    #[test]
    fn pseudo_element_validation() {
        // Unknown pseudo-elements invalidate the whole selector.
        assert!(parse_selector_list("div::bogus").is_err());
        assert!(parse_selector_list("div::BOGUS").is_err());
        // A pseudo-element must be last in its compound.
        assert!(parse_selector_list("div::before:hover").is_err());
        assert!(parse_selector_list("div:before:hover").is_err());
        assert!(parse_selector_list("div::before::after").is_err());
        // Pseudo-elements are not allowed inside :not().
        assert!(parse_selector_list(":not(::before)").is_err());
        assert!(parse_selector_list(":not(div::before)").is_err());
        // Supported pseudo-elements still parse, including legacy forms.
        for good in [
            "div::before",
            "div::after",
            "p::first-line",
            "p::first-letter",
            "::selection",
            "::marker",
            "::placeholder",
            "::backdrop",
            "div:before",
            "div:hover::before",
            "a:link::after",
        ] {
            assert!(parse_selector_list(good).is_ok(), "{good} should parse");
        }
    }

    #[test]
    fn namespace_prefixes_parse() {
        let tag = |input: &str| sel(input).compounds[0].tag.clone().expect("type selector");
        assert_eq!(
            tag("svg|div"),
            TagSelector {
                namespace: Some(NamespaceConstraint::Named("svg".into())),
                name: Some("div".into()),
            }
        );
        assert_eq!(
            tag("*|div"),
            TagSelector {
                namespace: Some(NamespaceConstraint::Any),
                name: Some("div".into()),
            }
        );
        assert_eq!(
            tag("|div"),
            TagSelector {
                namespace: Some(NamespaceConstraint::NoNamespace),
                name: Some("div".into()),
            }
        );
        assert_eq!(
            tag("svg|*"),
            TagSelector {
                namespace: Some(NamespaceConstraint::Named("svg".into())),
                name: None,
            }
        );
        assert_eq!(
            tag("div"),
            TagSelector {
                namespace: None,
                name: Some("div".into()),
            }
        );
        assert_eq!(
            tag("*"),
            TagSelector {
                namespace: None,
                name: None,
            }
        );
        // Attribute namespace prefixes.
        let attr = |input: &str| sel(input).compounds[0].attrs[0].clone();
        assert_eq!(
            attr("[svg|viewBox]").namespace,
            Some(NamespaceConstraint::Named("svg".into()))
        );
        assert_eq!(attr("[svg|viewBox]").name, "viewBox");
        assert_eq!(attr("[*|title]").namespace, Some(NamespaceConstraint::Any));
        assert_eq!(
            attr("[|title]").namespace,
            Some(NamespaceConstraint::NoNamespace)
        );
        assert_eq!(attr("[title]").namespace, None);
        // `|=` is an operator, not a namespace.
        assert!(matches!(attr("[lang|=en]").op, AttrOp::DashMatch));
        assert_eq!(attr("[lang|=en]").namespace, None);
    }

    #[test]
    fn specificity_complex() {
        assert_eq!(spec("div#a.b[c]:hover::before"), (1, 3, 2));
        assert_eq!(spec("#a #b #c"), (3, 0, 0));
    }

    #[test]
    fn is_where_parse() {
        let is = |input: &str| {
            let s = sel(input);
            assert_eq!(s.compounds[0].pseudos.len(), 1);
            s.compounds[0].pseudos[0].clone()
        };
        assert!(matches!(is("div:is(.a)"), PseudoClass::Is(_)));
        assert!(matches!(is("div:where(.a)"), PseudoClass::Where(_)));
        // Selector lists, including combinators, inside.
        match is(":is(div.a, #x > p)") {
            PseudoClass::Is(args) => assert_eq!(args.len(), 2),
            _ => panic!("expected :is"),
        }
        match is(":where(div, .a, #x)") {
            PseudoClass::Where(args) => assert_eq!(args.len(), 3),
            _ => panic!("expected :where"),
        }
        // Nesting functional pseudo-classes.
        sel(":is(:not(.a), div:where(p))");
        sel(":not(:is(.a))");
        // Invalid forms.
        assert!(parse_selector_list(":is()").is_err());
        assert!(parse_selector_list(":where()").is_err());
        assert!(parse_selector_list(":is(div").is_err());
        // Pseudo-elements are not allowed inside.
        assert!(parse_selector_list(":is(::before)").is_err());
        assert!(parse_selector_list(":where(div::after)").is_err());
        assert!(parse_selector_list(":not(::before)").is_err());
    }

    #[test]
    fn nth_invalid_inside_forgiving_lists() {
        // An invalid :nth-*() inside a forgiving list drops that group;
        // the selector survives only if some other group is valid. The
        // token path now sees the `<integer>`-lexeme flag, so invalid
        // groups are dropped (not kept leniently as `2`).
        assert!(parse_selector_list(":is(:nth-child(2.0), div)").is_ok());
        assert!(parse_selector_list(":where(div, :nth-of-type(1.5))").is_ok());
        assert!(parse_selector_list(":has(:nth-child(2.0), div)").is_ok());
        assert!(parse_selector_list(":is(:not(:nth-child(2.0)), div)").is_ok());
        // No valid group left: the forgiving pseudo is invalid, so the
        // whole selector fails.
        assert!(parse_selector_list("div:has(:nth-last-child(2.0))").is_err());
        assert!(parse_selector_list(":is(:nth-child(2.0))").is_err());
        // Outside forgiving lists it still invalidates the selector.
        assert!(parse_selector_list(":nth-child(2.0)").is_err());
        assert!(parse_selector_list("div:nth-of-type(1.5)").is_err());
        assert!(parse_selector_list(":not(:nth-child(2.0))").is_err());
        // Escapes and case in the forgiving function name are honored.
        assert!(parse_selector_list(":IS(:nth-child(2.0), div)").is_ok());
    }

    #[test]
    fn is_where_forgiving() {
        // Forgiving selector lists: invalid groups are dropped, valid
        // ones still work.
        let is = |input: &str| {
            let s = sel(input);
            assert_eq!(s.compounds[0].pseudos.len(), 1);
            s.compounds[0].pseudos[0].clone()
        };
        match is(":is(div, ::bogus)") {
            PseudoClass::Is(args) => assert_eq!(args.len(), 1),
            _ => panic!("expected :is"),
        }
        match is(":where(::bogus, .a)") {
            PseudoClass::Where(args) => assert_eq!(args.len(), 1),
            _ => panic!("expected :where"),
        }
        // All-invalid still invalidates the whole selector.
        assert!(parse_selector_list(":is(::bogus)").is_err());
        assert!(parse_selector_list(":where(::bogus, ::also-bogus)").is_err());
    }

    #[test]
    fn is_where_specificity() {
        // `:is()` takes its most specific argument; `:where()` takes zero.
        assert_eq!(spec(":is(div)"), (0, 0, 1));
        assert_eq!(spec(":is(.a, #b)"), (1, 0, 0));
        assert_eq!(spec(":is(#b, .a)"), (1, 0, 0));
        assert_eq!(spec(":is(div.a, div#b.c)"), (1, 1, 1));
        assert_eq!(spec(":where(div)"), (0, 0, 0));
        assert_eq!(spec(":where(#b, .a)"), (0, 0, 0));
        assert_eq!(spec("div:where(#b)"), (0, 0, 1));
        assert_eq!(spec(":is(:where(#b))"), (0, 0, 0));
        assert_eq!(spec(":where(:is(#b))"), (0, 0, 0));
        assert_eq!(spec(":not(:is(#b))"), (1, 0, 0));
        assert_eq!(spec(":is(div:not(#b))"), (1, 0, 1));
        // Universal selector inside :is adds nothing.
        assert_eq!(spec(":is(*)"), (0, 0, 0));
        assert_eq!(spec(":is(*, #b)"), (1, 0, 0));
    }

    #[test]
    fn has_parse() {
        let has = |input: &str| {
            let s = sel(input);
            assert_eq!(s.compounds[0].pseudos.len(), 1);
            match &s.compounds[0].pseudos[0] {
                PseudoClass::Has(args) => args.clone(),
                _ => panic!("expected :has"),
            }
        };
        // Default (descendant) anchor.
        let args = has("div:has(p)");
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].combinator, Combinator::Descendant);
        // Explicit leading combinators.
        let args = has("div:has(> p)");
        assert_eq!(args[0].combinator, Combinator::Child);
        let args = has("div:has(+ p)");
        assert_eq!(args[0].combinator, Combinator::NextSibling);
        let args = has("div:has(~ p)");
        assert_eq!(args[0].combinator, Combinator::SubsequentSibling);
        // Multi-compound relative selectors.
        let args = has("div:has(> p.a + span)");
        assert_eq!(args[0].selector.compounds.len(), 2);
        // Relative selector lists, with forgiving parsing.
        let args = has("div:has(> img, ::bogus, video)");
        assert_eq!(args.len(), 2);
        assert_eq!(args[0].combinator, Combinator::Child);
        assert_eq!(args[1].combinator, Combinator::Descendant);
        // Nesting other functional pseudo-classes is fine.
        sel("div:has(:is(p, span))");
        sel("div:has(p:not(.x))");
        // Invalid forms.
        assert!(parse_selector_list(":has()").is_err());
        assert!(parse_selector_list("div:has(> )").is_err());
        assert!(parse_selector_list("div:has(::bogus)").is_err());
        assert!(parse_selector_list("div:has(::before)").is_err());
        assert!(parse_selector_list("div:has(p::after)").is_err());
        // `:has()` cannot nest inside `:has()`, even through `:is()`.
        assert!(parse_selector_list("div:has(span:has(p))").is_err());
        assert!(parse_selector_list("div:has(:is(span:has(p)))").is_err());
        // ...but it can nest inside `:not()`.
        sel("div:not(:has(p))");
    }

    #[test]
    fn has_complexity_caps() {
        let has = |input: &str| {
            let s = sel(input);
            assert_eq!(s.compounds[0].pseudos.len(), 1);
            match &s.compounds[0].pseudos[0] {
                PseudoClass::Has(args) => args.clone(),
                _ => panic!("expected :has"),
            }
        };
        let chain = |n: usize| vec!["div"; n].join(" ");
        // At the cap: parses and the group is kept.
        let args = has(&format!("div:has({})", chain(16)));
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].selector.compounds.len(), 16);
        // Over the cap: the group is dropped like any other invalid
        // group of the forgiving list...
        let args = has(&format!("div:has({}, p)", chain(17)));
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].selector.compounds.len(), 1);
        // ...and with no valid group left the selector is invalid.
        assert!(parse_selector_list(&format!("div:has({})", chain(17))).is_err());
        assert!(parse_selector_list(&format!("div:has({})", chain(100))).is_err());
        assert!(parse_selector_list(&format!("div:has({})", chain(10_000))).is_err());
        // Two-tier caps: the 16 cap now covers only the relative
        // selector's own compounds; nested :is()/:not()/:where()
        // compounds have their own 256 budget. The old behavior dropped
        // `div:has(div:is(.a,...,.p))` (1 relative compound + 16
        // alternatives counted 1:1 against 16); now it parses.
        sel(&format!("div:has(div:is({}))", chain(16)));
        sel(&format!("div:has(div:not({}))", chain(16)));
        sel(&format!("div:has(div:where({}))", chain(16)));
        // 16 own compounds + a large-but-under-nested-cap :is() parses.
        // The nested budget is a total across arguments: four
        // 60-compound alternatives (240 <= 256), each itself a complex
        // selector under the 64 per-complex cap
        // ([`MAX_COMPLEX_COMPOUNDS`]).
        let big_is = (0..4).map(|_| chain(60)).collect::<Vec<_>>().join(", ");
        let args = has(&format!("div:has({} p:is({}))", chain(15), big_is));
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].selector.compounds.len(), 16);
        // Nested-in-nested compounds count toward the nested budget
        // (51 + 51 + 1 = 103 <= 256 here). Each nested argument is
        // itself a complex selector, so each stays under the 64
        // per-complex cap ([`MAX_COMPLEX_COMPOUNDS`]) while the total
        // still exercises the nested budget.
        sel(&format!(
            "div:has(p:is(span:is({0}), em:is({0}), b))",
            chain(50)
        ));
        let args = has("div:has(div:is(.a, .b) span)");
        assert_eq!(args.len(), 1);
        // Over the nested cap: the group is dropped like any other
        // invalid group of the forgiving list...
        let args = has(&format!("div:has(div:is({}), p)", chain(300)));
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].selector.compounds.len(), 1);
        // ...and with no valid group left the selector is invalid,
        // including a 10k-compound nested argument (rejected at parse,
        // fast: the counting walk is linear in the parsed group).
        assert!(parse_selector_list(&format!("div:has(div:is({}))", chain(300))).is_err());
        assert!(parse_selector_list(&format!("div:has(div:is({}))", chain(10_000))).is_err());
        assert!(parse_selector_list(&format!("div:has(p:is(span:is({})))", chain(300))).is_err());
        // Deeply nested :has() is rejected outright (stronger than any
        // depth cap): it cannot nest inside itself at all.
        assert!(parse_selector_list("div:has(:has(p))").is_err());
        assert!(parse_selector_list("div:has(:has(:has(p)))").is_err());
        assert!(parse_selector_list("div:has(:is(span:has(:has(p))))").is_err());
    }

    #[test]
    fn complex_compound_count_capped() {
        let chain = |n: usize| vec!["div"; n].join(" ");
        // At the cap: parses, with mixed combinators too.
        let s = sel(&chain(64));
        assert_eq!(s.compounds.len(), 64);
        assert_eq!(s.combinators.len(), 63);
        sel(&("div > ".repeat(63) + "div"));
        sel(&("div + ".repeat(31) + &"div ~ ".repeat(32) + "div"));
        // Over the cap: parse error, so the rule is dropped like any
        // other invalid selector.
        assert!(parse_selector_list(&chain(65)).is_err());
        // Hostile input is rejected at parse, fast: a 100k-compound
        // selector must never reach the matcher (a 10^5 x 10^3
        // compounds-x-nodes memo would pin ~2 GB transient per anchor).
        let hostile = chain(100_000);
        let start = std::time::Instant::now();
        assert!(parse_selector_list(&hostile).is_err());
        assert!(
            start.elapsed() < std::time::Duration::from_secs(5),
            "hostile complex selector took too long to reject at parse"
        );
        // The cap grades, not just rejects: in the forgiving lists an
        // over-cap group is dropped like any other invalid group.
        let is_args = |input: &str| {
            let s = sel(input);
            match &s.compounds[0].pseudos[0] {
                PseudoClass::Is(args) => args.len(),
                _ => panic!("expected :is"),
            }
        };
        assert_eq!(is_args(&format!("div:is({}, p)", chain(65))), 1);
        assert!(parse_selector_list(&format!("div:is({})", chain(65))).is_err());
        sel(&format!("div:not({}, p)", chain(65)));
        assert!(parse_selector_list(&format!("div:not({})", chain(65))).is_err());
        // A 64-compound argument is still fine everywhere.
        sel(&format!("div:is({})", chain(64)));
        sel(&format!("div:not({})", chain(64)));
        // `:has()` relative selectors keep their tighter 16-cap; a
        // 100-compound relative group is dropped by the complex cap on
        // the way in, exactly like the old 16-cap drop.
        let args = {
            let s = sel(&format!("div:has({}, p)", chain(100)));
            match &s.compounds[0].pseudos[0] {
                PseudoClass::Has(args) => args.clone(),
                _ => panic!("expected :has"),
            }
        };
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].selector.compounds.len(), 1);
        assert!(parse_selector_list(&format!("div:has({})", chain(100))).is_err());
    }

    #[test]
    fn has_specificity() {
        // `:has()` takes its most specific relative-selector argument;
        // the anchor combinator adds nothing.
        assert_eq!(spec(":has(p)"), (0, 0, 1));
        assert_eq!(spec(":has(> #b)"), (1, 0, 0));
        assert_eq!(spec(":has(.a, #b)"), (1, 0, 0));
        assert_eq!(spec("div:has(> p.a)"), (0, 1, 2));
        assert_eq!(spec(":has(:where(#b))"), (0, 0, 0));
        assert_eq!(spec(":has(:is(#b))"), (1, 0, 0));
    }

    #[test]
    fn nth_family_parse() {
        let nth = |input: &str| {
            let s = sel(input);
            assert_eq!(s.compounds[0].pseudos.len(), 1);
            s.compounds[0].pseudos[0].clone()
        };
        assert!(matches!(
            nth("p:nth-last-child(2n+1)"),
            PseudoClass::NthLastChild(2, 1)
        ));
        assert!(matches!(
            nth("p:nth-of-type(odd)"),
            PseudoClass::NthOfType(2, 1)
        ));
        assert!(matches!(
            nth("p:nth-last-of-type(-n+3)"),
            PseudoClass::NthLastOfType(-1, 3)
        ));
        assert!(matches!(nth("p:first-of-type"), PseudoClass::FirstOfType));
        assert!(matches!(nth("p:last-of-type"), PseudoClass::LastOfType));
        assert!(matches!(nth("p:only-child"), PseudoClass::OnlyChild));
        assert!(matches!(nth("p:only-of-type"), PseudoClass::OnlyOfType));
        // The an+b microsyntax is shared with :nth-child.
        for bad in [
            "p:nth-last-child(2.0)",
            "p:nth-of-type(bogus)",
            "p:nth-last-of-type(n+m)",
            "p:nth-of-type()",
        ] {
            assert!(parse_selector_list(bad).is_err(), "{bad} should be invalid");
        }
        for good in [
            "p:nth-last-child(2n)",
            "p:nth-of-type(even)",
            "p:nth-last-of-type(3)",
            "p:nth-of-type(-n+2)",
        ] {
            assert!(parse_selector_list(good).is_ok(), "{good} should parse");
        }
        // Case-insensitive function names.
        assert!(matches!(
            nth("p:NTH-OF-TYPE(2n)"),
            PseudoClass::NthOfType(2, 0)
        ));
    }

    #[test]
    fn nth_family_specificity() {
        // All structural pseudo-classes count (0, 1, 0).
        for s in [
            "p:nth-last-child(2n)",
            "p:nth-of-type(2n)",
            "p:nth-last-of-type(2n)",
            "p:first-of-type",
            "p:last-of-type",
            "p:only-child",
            "p:only-of-type",
            "input:checked",
            "input:disabled",
            "input:enabled",
        ] {
            assert_eq!(spec(s), (0, 1, 1), "{s}");
        }
    }

    #[test]
    fn form_pseudo_classes_parse() {
        sel("input:checked");
        sel("option:checked");
        sel("input:disabled");
        sel("input:enabled");
        sel("input:checked:disabled");
        // Unknown neighbors still fail.
        assert!(parse_selector_list("input:check").is_err());
    }

    #[test]
    fn pseudo_element_must_end_complex_selector() {
        // A pseudo-element anywhere but the end is invalid.
        assert!(parse_selector_list("div::before p").is_err());
        assert!(parse_selector_list("div::before > p").is_err());
        assert!(parse_selector_list("div::before + p").is_err());
        // Trailing positions are still fine.
        sel("div p::before");
        sel("div > p::after");
    }

    #[test]
    fn nesting_depth_capped_for_new_pseudos() {
        // Deep `:is()`/`:has()` nesting fails instead of overflowing the
        // stack; the forgiving lists just drop the over-deep group.
        let deep_is = ":is(".repeat(3000) + &")".repeat(3000);
        assert!(parse_selector_list(&deep_is).is_err());
        let deep_has = ":has(".repeat(3000) + &")".repeat(3000);
        assert!(parse_selector_list(&deep_has).is_err());
        // Just under the cap still parses.
        let ok = ":is(".repeat(63) + "div" + &")".repeat(63);
        assert!(parse_selector_list(&ok).is_ok());
    }
}
