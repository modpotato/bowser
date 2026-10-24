//! Fixed containing-block establishers beyond `transform`.
//!
//! Per spec, a box also becomes the containing block for fixed (and
//! absolutely positioned) descendants when it has a non-`none` `filter`
//! (Filter Effects 1), a non-`none` `backdrop-filter` (Filter Effects
//! 2), a non-`none` `perspective` (CSS Transforms 2), a `will-change`
//! naming a stacking-context trigger (CSS Will Change 1 §3), or
//! `contain: paint | layout | strict | content` (CSS Contain 1/2) — the
//! same effect `transform: <non-none>` already has here.
//! This module owns the property-specific value types and parsing for
//! those five properties:
//!
//! - `filter` needs a dedicated type: its functions (`blur()`,
//!   `drop-shadow()`, …) are not transform functions, so the generic
//!   value parser rejects them. They are validated by name and retained
//!   opaque, one entry per function (like [`TransformFunction::Other`]);
//!   full filter-function semantics are future work.
//! - `will-change`, `perspective`, and `contain` parse to the generic
//!   [`CssValue`] shapes (`Keyword` / `List` / `Length`) the value parser
//!   already produces; the cascade's per-property grammar check
//!   validates them, exactly like `transform`.

use crate::cascade::{ComputedStyle, props};
use crate::token::Token;
use crate::value::{CssValue, parse_value, tokens_to_css};

/// One filter function in a `filter: <filter-value-list>` (Filter
/// Effects 1 §8). Stored opaque: only the function name is validated at
/// parse time; argument arity and used-value computation are future work.
/// `name` is the lowercased function name; `raw_args` is the argument
/// body re-serialized from tokens (whitespace-trimmed) so
/// `getComputedStyle` round-trips it. For `url(<unquoted>)` (which
/// tokenizes as [`Token::Url`]) `raw_args` is the URL itself.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterFunction {
    /// Lowercased function name (`blur`, `drop-shadow`, `url`, …).
    pub name: String,
    /// Argument body re-serialized from tokens, whitespace-trimmed.
    pub raw_args: String,
}

/// `filter: none | <filter-value-list>` (Filter Effects 1 §8).
#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    /// `none`: the initial value. Establishes no containing block.
    None,
    /// One entry per function, in source order. Any non-empty list
    /// establishes a containing block for fixed (and absolute)
    /// descendants, like a non-`none` transform.
    List(Vec<FilterFunction>),
}

/// Filter functions (Filter Effects 1 §8) the `filter` parser accepts.
/// `url()` is handled separately: unquoted it tokenizes as
/// [`Token::Url`], quoted as `Function("url")` with a string body.
const FILTER_FUNCTIONS: &[&str] = &[
    "blur",
    "brightness",
    "contrast",
    "drop-shadow",
    "grayscale",
    "hue-rotate",
    "invert",
    "opacity",
    "saturate",
    "sepia",
];

/// Whether the token slice holds a top-level `var()` function reference
/// (one whose `(` is not nested inside another function or parens).
fn has_top_level_var(tokens: &[Token]) -> bool {
    let mut depth: u32 = 0;
    for t in tokens {
        match t {
            Token::Function(name) if depth == 0 && name.eq_ignore_ascii_case("var") => return true,
            Token::Function(_) | Token::LParen | Token::LBracket => depth += 1,
            Token::RParen | Token::RBracket => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    false
}

/// Split tokens into whitespace-separated groups at the top level
/// (whitespace inside functions/parens never separates). Trailing `Eof`
/// tokens are dropped first; the tokenizer emits a single trailing one.
fn split_ws_groups(tokens: &[Token]) -> Vec<&[Token]> {
    let mut tokens = tokens;
    while matches!(
        tokens.last(),
        Some(Token::Eof) | Some(Token::Whitespace) | Some(Token::Semicolon)
    ) {
        tokens = &tokens[..tokens.len() - 1];
    }
    let mut groups = Vec::new();
    let mut start = 0;
    let mut depth: u32 = 0;
    let mut i = 0;
    while i <= tokens.len() {
        if i < tokens.len() {
            match &tokens[i] {
                Token::Function(_) | Token::LParen | Token::LBracket => depth += 1,
                Token::RParen | Token::RBracket => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
        let is_sep = i == tokens.len() || (depth == 0 && matches!(tokens[i], Token::Whitespace));
        if is_sep {
            if start < i {
                groups.push(&tokens[start..i]);
            }
            start = i + 1;
        }
        i += 1;
    }
    groups
}

/// Trim leading/trailing whitespace tokens from a slice.
fn trim_ws(tokens: &[Token]) -> &[Token] {
    let mut t = tokens;
    while matches!(t.first(), Some(Token::Whitespace)) {
        t = &t[1..];
    }
    while matches!(t.last(), Some(Token::Whitespace)) {
        t = &t[..t.len() - 1];
    }
    t
}

/// Parse a `filter` declaration value from its token slice.
///
/// `none` parses to [`Filter::None`]; a lone CSS-wide keyword (or any
/// other lone ident, which the cascade's grammar check then rejects)
/// parses to a bare [`CssValue::Keyword`] for the cascade's wide-keyword
/// path; anything else must be a whitespace-separated list of filter
/// functions, each validated by name and retained opaque as
/// [`FilterFunction`]. A top-level `var()` reference cannot be validated
/// until substitution, so it defers to the generic value parser and
/// keeps its `Var` node for the cascade to substitute (a substituted
/// `none` computes normally; a substituted function list cannot re-parse
/// and is invalid at computed-value time).
///
/// Anything else (unknown functions, empty bodies, stray tokens) is
/// invalid and the declaration drops.
pub(crate) fn parse_filter_value(tokens: &[Token]) -> Option<CssValue> {
    if has_top_level_var(tokens) {
        return parse_value(tokens).ok();
    }
    let non_ws: Vec<&Token> = tokens
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace | Token::Eof))
        .collect();
    match non_ws.as_slice() {
        [Token::Ident(kw)] if kw.eq_ignore_ascii_case("none") => {
            return Some(CssValue::Filter(Filter::None));
        }
        // A lone ident: a CSS-wide keyword passes through for the
        // cascade's wide-keyword path; anything else (e.g. `filter:
        // blur` without parens) dies in the cascade's grammar check.
        [Token::Ident(kw)] => {
            return Some(CssValue::Keyword(kw.to_ascii_lowercase().into()));
        }
        _ => {}
    }
    let mut fns = Vec::new();
    for group in split_ws_groups(tokens) {
        match group {
            // `url(<unquoted>)` tokenizes as a single `Url` token.
            [Token::Url(u)] => fns.push(FilterFunction {
                name: "url".to_string(),
                raw_args: u.clone(),
            }),
            [Token::Function(name), rest @ ..] => {
                let body = rest.strip_suffix(&[Token::RParen])?;
                let lower = name.to_ascii_lowercase();
                if lower == "url" {
                    // `url("…")` tokenizes as a function with a string
                    // body.
                    match trim_ws(body) {
                        [Token::String(s)] => fns.push(FilterFunction {
                            name: "url".to_string(),
                            raw_args: s.clone(),
                        }),
                        _ => return None,
                    }
                } else {
                    if !FILTER_FUNCTIONS.contains(&lower.as_str()) {
                        return None;
                    }
                    let body = trim_ws(body);
                    if body.is_empty() {
                        return None;
                    }
                    fns.push(FilterFunction {
                        name: lower,
                        raw_args: tokens_to_css(body),
                    });
                }
            }
            _ => return None,
        }
    }
    if fns.is_empty() {
        return None;
    }
    Some(CssValue::Filter(Filter::List(fns)))
}

/// Whether the computed style establishes a containing block for
/// `position: fixed` (and absolutely positioned) descendants via one of
/// this module's properties:
///
/// - `will-change` naming `transform`, `filter`, or `perspective` (CSS
///   Will Change 1 §3, narrowed by CSS Position 3 §2.1, which lists only
///   those three as will-change containing-block triggers),
/// - a non-`none` `filter` (Filter Effects 1 §8),
/// - a non-`none` `backdrop-filter` (Filter Effects 2 §2),
/// - a non-`none` `perspective` (CSS Transforms 2),
/// - `contain: paint | layout | strict | content` (CSS Contain 1/2).
///
/// Note `opacity < 1`, `isolation: isolate`, and non-`normal`
/// `mix-blend-mode` create stacking contexts (CSS Will Change 1 §3) but
/// never containing blocks, so naming them in `will-change` does not
/// count — and neither does `will-change: contain`, for which no spec
/// lists a containing-block trigger (CSS Position 3 §2.1 names only
/// `transform`/`perspective`/`filter`).
///
/// (The sibling case, a non-`none` `transform`, stays with the
/// layout-side `has_non_none_transform`.) Layout calls this as part of
/// its full fixed-CB predicate, which also covers `transform`.
pub fn establishes_fixed_cb(style: &ComputedStyle) -> bool {
    fixed_cb_props(|idx| style.get_by_index(idx))
}

/// Core of [`establishes_fixed_cb`] over a const property-index getter,
/// so unit tests can drive the decision logic without building a
/// [`ComputedStyle`] (its slots are private to the cascade).
fn fixed_cb_props<'a>(get: impl Fn(u8) -> Option<&'a CssValue>) -> bool {
    // CSS Will Change 1 §3: a hint naming a property whose non-initial
    // value would create a stacking context forces one. CSS Position 3
    // §2.1 then narrows which of those hints establish a *containing
    // block*: only `transform`, `perspective`, and `filter`.
    // `opacity`, `mix-blend-mode`, and `isolation` create stacking
    // contexts but never containing blocks, and `contain` has no spec
    // basis as a will-change containing-block trigger at all.
    // `auto` (the initial) and non-trigger hints (`top`,
    // `scroll-position`, …) do not.
    let will_change = match get(props::WILL_CHANGE) {
        Some(CssValue::Keyword(k)) => will_change_triggers(k),
        Some(CssValue::List(items)) => items
            .iter()
            .any(|i| matches!(i, CssValue::Keyword(k) if will_change_triggers(k))),
        _ => false,
    };
    if will_change {
        return true;
    }
    // Filter Effects 1 §8: any non-`none` filter. (`none` also arrives
    // as a bare keyword when a `var()` substituted to it.)
    if matches!(get(props::FILTER), Some(CssValue::Filter(Filter::List(_)))) {
        return true;
    }
    // Filter Effects 2 §2: any non-`none` backdrop-filter establishes a
    // containing block exactly like `filter`.
    if matches!(
        get(props::BACKDROP_FILTER),
        Some(CssValue::Filter(Filter::List(_)))
    ) {
        return true;
    }
    // CSS Transforms 2: any non-`none` perspective. Only `none` computes
    // to a keyword; anything else that passed validation is a length.
    let perspective = match get(props::PERSPECTIVE) {
        Some(CssValue::Keyword(k)) => k != "none",
        Some(_) => true,
        None => false,
    };
    if perspective {
        return true;
    }
    // CSS Contain 1/2: `paint`, `layout`, `strict`, or `content`.
    // (`strict` is `size layout paint`, `content` is `layout paint`;
    // lone `size`/`style` create no stacking context.)
    match get(props::CONTAIN) {
        Some(CssValue::Keyword(k)) => {
            matches!(k.as_str(), "strict" | "content" | "paint" | "layout")
        }
        Some(CssValue::List(items)) => items.iter().any(|i| {
            matches!(i, CssValue::Keyword(k) if matches!(k.as_str(), "strict" | "content" | "paint" | "layout"))
        }),
        _ => false,
    }
}

/// `will-change` hints whose non-initial value establishes a
/// containing block for fixed/absolutely positioned descendants: only
/// `transform`, `filter`, and `perspective` (CSS Position 3 §2.1).
/// `opacity`, `mix-blend-mode`, and `isolation` create stacking contexts
/// (CSS Will Change 1 §3) but never containing blocks; `contain` has no
/// spec basis as a will-change containing-block trigger.
fn will_change_triggers(hint: &str) -> bool {
    matches!(hint, "transform" | "filter" | "perspective")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_declarations;

    fn filter_of(css: &str) -> Option<CssValue> {
        let decls = parse_declarations(css);
        if decls.len() != 1 {
            return None;
        }
        assert_eq!(decls[0].property, "filter");
        Some(decls[0].value.clone())
    }

    fn names(v: &CssValue) -> Vec<&str> {
        match v {
            CssValue::Filter(Filter::List(fns)) => fns.iter().map(|f| f.name.as_str()).collect(),
            _ => panic!("expected Filter::List, got {v:?}"),
        }
    }

    #[test]
    fn filter_none() {
        assert_eq!(
            filter_of("filter: none"),
            Some(CssValue::Filter(Filter::None))
        );
        // Case-insensitive.
        assert_eq!(
            filter_of("filter: NONE"),
            Some(CssValue::Filter(Filter::None))
        );
    }

    #[test]
    fn filter_function_list() {
        let v = filter_of("filter: blur(5px) grayscale(50%)").expect("parses");
        assert_eq!(names(&v), vec!["blur", "grayscale"]);
        match &v {
            CssValue::Filter(Filter::List(fns)) => {
                assert_eq!(fns[0].raw_args, "5px");
                assert_eq!(fns[1].raw_args, "50%");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn filter_drop_shadow_keeps_space_separated_args() {
        let v = filter_of("filter: drop-shadow(2px 2px 2px black)").expect("parses");
        match &v {
            CssValue::Filter(Filter::List(fns)) => {
                assert_eq!(fns.len(), 1);
                assert_eq!(fns[0].name, "drop-shadow");
                assert_eq!(fns[0].raw_args, "2px 2px 2px black");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn filter_url_forms() {
        let v = filter_of("filter: url(noise.png)").expect("parses");
        match &v {
            CssValue::Filter(Filter::List(fns)) => {
                assert_eq!(fns[0].name, "url");
                assert_eq!(fns[0].raw_args, "noise.png");
            }
            _ => unreachable!(),
        }
        let v = filter_of("filter: url(\"noise.png\")").expect("parses");
        match &v {
            CssValue::Filter(Filter::List(fns)) => {
                assert_eq!(fns[0].name, "url");
                assert_eq!(fns[0].raw_args, "noise.png");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn filter_invalid_values_drop() {
        // Unknown function.
        assert_eq!(filter_of("filter: frobnicate(1)"), None);
        // Empty body.
        assert_eq!(filter_of("filter: blur()"), None);
        // Bare ident (not a function): parses to a bare keyword so the
        // cascade's wide-keyword path sees it; the cascade's `filter`
        // grammar check (`none | <filter-value-list>`) drops anything
        // that isn't `none` (see cascade.rs validation).
        assert_eq!(
            filter_of("filter: blur"),
            Some(CssValue::Keyword("blur".into()))
        );
        // Comma-separated (filter takes a space-separated list).
        assert_eq!(filter_of("filter: blur(5px), grayscale(50%)"), None);
        // Stray token.
        assert_eq!(filter_of("filter: blur(5px) 10px"), None);
    }

    #[test]
    fn filter_wide_keyword_passes_through() {
        assert_eq!(
            filter_of("filter: inherit"),
            Some(CssValue::Keyword("inherit".into()))
        );
    }

    #[test]
    fn filter_var_defers() {
        let v = filter_of("filter: var(--f)").expect("parses");
        assert!(matches!(v, CssValue::Var(_, _)), "got {v:?}");
    }

    fn value_of(css: &str, prop: &str) -> Option<CssValue> {
        let decls = parse_declarations(css);
        if decls.len() != 1 {
            return None;
        }
        assert_eq!(decls[0].property, prop);
        Some(decls[0].value.clone())
    }

    #[test]
    fn will_change_and_perspective_parse_shapes() {
        use crate::value::LengthUnit;

        assert_eq!(
            value_of("will-change: transform", "will-change"),
            Some(CssValue::Keyword("transform".into()))
        );
        assert!(matches!(
            value_of("will-change: transform, opacity", "will-change"),
            Some(CssValue::List(_))
        ));
        assert_eq!(
            value_of("perspective: 500px", "perspective"),
            Some(CssValue::Length(500.0, LengthUnit::Px))
        );
        assert_eq!(
            value_of("perspective: none", "perspective"),
            Some(CssValue::Keyword("none".into()))
        );
    }

    #[test]
    fn fixed_cb_predicate_from_parsed_values() {
        use crate::cascade::props;
        use std::collections::HashMap;

        // (declaration, property index, establishes a fixed CB?).
        let cases = [
            ("will-change: transform", props::WILL_CHANGE, true),
            ("will-change: transform, opacity", props::WILL_CHANGE, true),
            ("will-change: Transform", props::WILL_CHANGE, true),
            ("will-change: filter", props::WILL_CHANGE, true),
            ("will-change: perspective", props::WILL_CHANGE, true),
            // Stacking-context-only hints establish no containing block
            // (CSS Position 3 §2.1), and neither does `contain`.
            ("will-change: opacity", props::WILL_CHANGE, false),
            ("will-change: mix-blend-mode", props::WILL_CHANGE, false),
            ("will-change: isolation", props::WILL_CHANGE, false),
            ("will-change: contain", props::WILL_CHANGE, false),
            ("will-change: auto", props::WILL_CHANGE, false),
            ("will-change: top", props::WILL_CHANGE, false),
            ("filter: blur(1px)", props::FILTER, true),
            ("filter: blur(1px) grayscale(50%)", props::FILTER, true),
            ("filter: none", props::FILTER, false),
            ("perspective: 500px", props::PERSPECTIVE, true),
            ("perspective: none", props::PERSPECTIVE, false),
            ("contain: paint", props::CONTAIN, true),
            ("contain: layout paint", props::CONTAIN, true),
            ("contain: strict", props::CONTAIN, true),
            ("contain: content", props::CONTAIN, true),
            ("contain: size", props::CONTAIN, false),
            ("contain: none", props::CONTAIN, false),
        ];
        for (css, idx, expect) in cases {
            let prop = css.split(':').next().unwrap().trim();
            let v = value_of(css, prop).unwrap_or_else(|| panic!("{css} parses"));
            let map: HashMap<u8, CssValue> = [(idx, v)].into_iter().collect();
            // Same decision logic `establishes_fixed_cb` applies to a
            // real `ComputedStyle` (whose slots are private to the
            // cascade, hence the getter indirection).
            assert_eq!(fixed_cb_props(|i| map.get(&i)), expect, "{css}");
        }
        // Absent properties establish nothing.
        let empty: HashMap<u8, CssValue> = HashMap::new();
        assert!(!fixed_cb_props(|i| empty.get(&i)));
    }
}
