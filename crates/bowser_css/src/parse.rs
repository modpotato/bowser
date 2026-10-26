//! Stylesheet parser: qualified rules, at-rules, declarations, and
//! shorthand expansion.
//!
//! [`parse_stylesheet`] never fails: per CSS Syntax L3, a parse error drops
//! just the offending rule or declaration and parsing continues.

use crate::cascade::{is_css_wide_keyword, is_valid_for_property, split_place_self};
use crate::propatom::PropertyName;
use crate::selector::{Selector, parse_selector_list_from_tokens};
use crate::token::{Token, tokenize};
use crate::value::{
    ClipPath, Color, CssValue, LengthUnit, Shadow, css_source_contains_var, is_math_fn_name,
    parse_basic_shape, parse_math_value, parse_value, parse_value_preserving_case,
    system_font_values,
};

/// A parsed stylesheet: rules in source order.
#[derive(Clone, Debug, Default)]
pub struct Stylesheet {
    rules: Vec<Rule>,
    font_faces: Vec<FontFaceRule>,
}

impl Stylesheet {
    /// All rules in source order (including rules inside `@media` blocks).
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    /// Parsed `@font-face` rules in source order. These are font
    /// descriptors, not style rules, so they are deliberately kept out
    /// of [`rules`](Self::rules). The app loader fetches their `src`
    /// URLs (see `bowser_app::loader`); weight/style descriptors are
    /// retained on each rule for face matching.
    /// (The Ahem test font that WPT text-measurement tests depend on is
    /// installed directly via `FontCache::install_test_fonts()`, not
    /// through these rules.)
    pub fn font_faces(&self) -> &[FontFaceRule] {
        &self.font_faces
    }
}

/// One `url(...)` / `local(...)` entry of an `@font-face` `src`
/// descriptor, in authored order.
#[derive(Clone, Debug, PartialEq)]
pub enum FontFaceSrc {
    /// `url(...)` with the optional `format(...)` hint. The URL is kept
    /// exactly as authored; relative URLs resolve against the
    /// stylesheet's base URL when the font is fetched.
    Url { url: String, format: Option<String> },
    /// `local(...)`: a locally installed face name.
    Local(String),
}

/// A parsed `@font-face` rule: the `font-family` descriptor plus the
/// ordered `src` list. `font-weight` and `font-style` descriptors are
/// retained (other descriptors, e.g. `unicode-range`, are parsed past
/// and ignored). The loader's `@font-face` fetch step uses the `src`
/// list; weight/style are retained for face matching by a later slice.
#[derive(Clone, Debug, PartialEq)]
pub struct FontFaceRule {
    /// The `font-family` descriptor, unquoted, as authored.
    pub family: String,
    /// The `src` descriptors in authored order.
    pub src: Vec<FontFaceSrc>,
    /// The `font-weight` descriptor: a single weight (default 400).
    /// For a range descriptor (`100 900`) this is the range's start
    /// and [`weight_range`](Self::weight_range) holds the full range.
    pub weight: u16,
    /// The full range of a range `font-weight` descriptor, if authored.
    pub weight_range: Option<(u16, u16)>,
    /// The `font-style` descriptor (default normal).
    pub style: FontFaceStyle,
    /// Source order among this stylesheet's `@font-face` rules.
    pub order: usize,
}

/// A `@font-face` `font-style` descriptor value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FontFaceStyle {
    #[default]
    Normal,
    Italic,
    Oblique,
}

impl FontFaceRule {
    /// The `font-family` descriptor, as authored.
    pub fn family_name(&self) -> &str {
        &self.family
    }

    /// The `font-weight` descriptor (single weight, or a range's start).
    pub fn font_weight(&self) -> u16 {
        self.weight
    }

    /// The full range of a range `font-weight` descriptor, if authored.
    pub fn font_weight_range(&self) -> Option<(u16, u16)> {
        self.weight_range
    }

    /// The `font-style` descriptor as a lowercase keyword.
    pub fn font_style_name(&self) -> &'static str {
        match self.style {
            FontFaceStyle::Normal => "normal",
            FontFaceStyle::Italic => "italic",
            FontFaceStyle::Oblique => "oblique",
        }
    }

    /// The `url(...)` entries of `src` in authored order, skipping
    /// `local(...)` entries (those resolve against already-installed
    /// faces, so there is nothing to fetch).
    pub fn url_srcs(&self) -> impl Iterator<Item = &str> {
        self.src.iter().filter_map(|entry| match entry {
            FontFaceSrc::Url { url, .. } => Some(url.as_str()),
            FontFaceSrc::Local(_) => None,
        })
    }
}

/// A single style rule.
#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
    /// Selector list (one [`Selector`] per comma group).
    pub selectors: Vec<Selector>,
    /// Declarations with shorthands already expanded to longhands.
    pub declarations: Vec<Declaration>,
    /// Source order across the whole stylesheet (media-nested included).
    pub order: usize,
    /// The `@media` query gating this rule, if any.
    pub media: Option<MediaQuery>,
}

/// A single `property: value` declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct Declaration {
    /// Lowercased property name (custom properties keep their case),
    /// interned: well-known names are a 1-byte atom, the rest ride a
    /// heap sidecar.
    pub property: PropertyName,
    pub value: CssValue,
    pub important: bool,
}

/// A parsed `@media` query: comma-separated parts (OR), each an optional
/// media type plus `(min-width)`/`(max-width)`/`(min-height)`/
/// `(max-height)` conditions (AND).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MediaQuery {
    parts: Vec<MediaPart>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct MediaPart {
    negate: bool,
    media_type: Option<String>,
    conditions: Vec<MediaCondition>,
    /// True when the part could not be understood; it never matches.
    unsupported: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct MediaCondition {
    is_min: bool,
    /// Whether the condition tests the viewport height (`min-height` /
    /// `max-height`) rather than the width.
    is_height: bool,
    value_px: f32,
}

impl MediaQuery {
    /// Whether this query matches a viewport of the given size.
    /// Only `screen`/`all` media types match; `min-width`/`max-width`
    /// are compared against `viewport_w` in px, `min-height` /
    /// `max-height` against `viewport_h`.
    pub fn matches(&self, viewport_w: f32, viewport_h: f32) -> bool {
        self.parts.iter().any(|p| p.matches(viewport_w, viewport_h))
    }

    /// AND two queries (used for nested `@media`): cross product of parts,
    /// truncated to [`MAX_MEDIA_PARTS`] so adversarial nesting cannot
    /// blow up memory exponentially.
    fn and(&self, other: &MediaQuery) -> MediaQuery {
        let mut parts = Vec::new();
        'outer: for a in &self.parts {
            for b in &other.parts {
                if parts.len() >= MAX_MEDIA_PARTS {
                    break 'outer;
                }
                parts.push(a.and_part(b));
            }
        }
        MediaQuery { parts }
    }
}

impl MediaPart {
    fn matches(&self, w: f32, h: f32) -> bool {
        if self.unsupported {
            return false;
        }
        let type_ok = match &self.media_type {
            None => true,
            Some(t) => t == "all" || t == "screen",
        };
        let mut ok = type_ok;
        for c in &self.conditions {
            let v = if c.is_height { h } else { w };
            ok &= if c.is_min {
                v >= c.value_px
            } else {
                v <= c.value_px
            };
        }
        if self.negate { !ok } else { ok }
    }

    fn and_part(&self, other: &MediaPart) -> MediaPart {
        if self.negate || other.negate || self.unsupported || other.unsupported {
            return MediaPart {
                unsupported: true,
                ..MediaPart::default()
            };
        }
        let media_type = match (&self.media_type, &other.media_type) {
            (None, None) => None,
            (Some(t), None) | (None, Some(t)) => Some(t.clone()),
            (Some(a), Some(b)) if a == b => Some(a.clone()),
            (Some(a), Some(b))
                if (a == "all" && b == "screen") || (a == "screen" && b == "all") =>
            {
                Some("screen".to_string())
            }
            _ => {
                return MediaPart {
                    unsupported: true,
                    ..MediaPart::default()
                };
            }
        };
        let mut conditions = self.conditions.clone();
        conditions.extend(other.conditions.iter().cloned());
        MediaPart {
            negate: false,
            media_type,
            conditions,
            unsupported: false,
        }
    }
}

/// Parse a whole stylesheet. Never fails; broken rules/declarations are dropped.
pub fn parse_stylesheet(input: &str) -> Stylesheet {
    let tokens = tokenize(input);
    let mut p = Parser {
        toks: &tokens,
        pos: 0,
        order: 0,
        media_depth: 0,
        font_faces: Vec::new(),
    };
    let mut rules = Vec::new();
    p.parse_rule_list(&mut rules, None, false);
    Stylesheet {
        rules,
        font_faces: p.font_faces,
    }
}

/// Parse a declaration block body, e.g. an element's `style` attribute.
/// Invalid declarations are dropped.
pub fn parse_declarations(input: &str) -> Vec<Declaration> {
    let tokens = tokenize(input);
    let mut p = Parser {
        toks: &tokens,
        pos: 0,
        order: 0,
        media_depth: 0,
        font_faces: Vec::new(),
    };
    p.parse_declaration_block()
}

struct Parser<'a> {
    toks: &'a [Token],
    pos: usize,
    order: usize,
    /// Current `@media` nesting depth. `parse_rule_list` recurses through
    /// `parse_at_rule` once per nested `@media`; the depth is capped at
    /// [`MAX_MEDIA_DEPTH`] so `@media{` x 100k fails closed instead of
    /// overflowing the stack.
    media_depth: u32,
    /// `@font-face` rules collected while parsing (kept separate from
    /// style rules; moved into the [`Stylesheet`] at the end).
    font_faces: Vec<FontFaceRule>,
}

/// Maximum tokens retained from one `@font-face` block. The block holds
/// a handful of descriptors; the cap keeps an adversarial rule from
/// forcing unbounded memory (consistent with [`MAX_MEDIA_DEPTH`] /
/// [`MAX_MEDIA_PARTS`] above).
const MAX_FONT_FACE_TOKENS: usize = 512;
/// Maximum `src` entries retained from one `@font-face` rule.
const MAX_FONT_FACE_SRCS: usize = 32;

/// Maximum `@media` nesting depth parsed; deeper blocks are skipped.
const MAX_MEDIA_DEPTH: u32 = 64;

/// Maximum parts in a combined `@media` query. Nesting ANDs queries via a
/// cross product of parts, which grows exponentially; this cap bounds the
/// memory a pathological stylesheet can force.
const MAX_MEDIA_PARTS: usize = 256;

impl Parser<'_> {
    fn peek(&self) -> Option<&Token> {
        match self.toks.get(self.pos) {
            None | Some(Token::Eof) => None,
            t => t,
        }
    }

    fn bump(&mut self) -> Option<Token> {
        let t = match self.toks.get(self.pos) {
            None | Some(Token::Eof) => None,
            t => t.cloned(),
        };
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(Token::Whitespace)) {
            self.pos += 1;
        }
    }

    fn parse_rule_list(&mut self, rules: &mut Vec<Rule>, media: Option<&MediaQuery>, nested: bool) {
        loop {
            self.skip_ws();
            match self.peek() {
                None => return,
                Some(Token::RBrace) => {
                    self.bump();
                    if nested {
                        return;
                    }
                    // Stray `}` at top level: ignored.
                }
                Some(Token::Semicolon) => {
                    self.bump();
                }
                Some(Token::AtKeyword(_)) => self.parse_at_rule(rules, media),
                Some(Token::Cdo) | Some(Token::Cdc) => {
                    self.bump();
                }
                _ => {
                    if let Some((selectors, declarations)) = self.parse_qualified_rule() {
                        let order = self.order;
                        self.order += 1;
                        rules.push(Rule {
                            selectors,
                            declarations,
                            order,
                            media: media.cloned(),
                        });
                    }
                }
            }
        }
    }

    fn parse_at_rule(&mut self, rules: &mut Vec<Rule>, media: Option<&MediaQuery>) {
        let name = match self.bump() {
            Some(Token::AtKeyword(n)) => n.to_ascii_lowercase(),
            _ => return,
        };
        if name == "media" {
            let prelude = self.consume_prelude();
            let query = parse_media_query(&prelude);
            let combined = match media {
                Some(outer) => outer.and(&query),
                None => query,
            };
            if matches!(self.peek(), Some(Token::LBrace)) {
                self.bump();
                if self.media_depth < MAX_MEDIA_DEPTH {
                    self.media_depth += 1;
                    self.parse_rule_list(rules, Some(&combined), true);
                    self.media_depth -= 1;
                } else {
                    // Past the nesting cap: skip the block without
                    // parsing so adversarial `@media{` x 100k cannot
                    // overflow the stack.
                    self.skip_balanced_block();
                }
            }
            // `@media ... ;` with no block: nothing to do.
        } else if name == "font-face" {
            // @font-face is retained (family + src) for a future font
            // pipeline; @keyframes, @import, and anything else are
            // parse-and-skipped.
            self.parse_font_face_rule();
        } else {
            // @keyframes, @import, and anything else: parse-and-skip
            // the prelude and block.
            self.skip_at_prelude_and_block();
        }
    }

    /// Parse `@font-face { font-family: …; src: …; }`, retaining the
    /// family name and `src` list on [`Stylesheet::font_faces`]. The
    /// rule's prelude must be empty; other descriptors are parsed past
    /// and ignored. A malformed rule is dropped without disturbing the
    /// surrounding rules (per CSS Syntax L3, a parse error drops just
    /// the offending rule).
    fn parse_font_face_rule(&mut self) {
        // Prelude: must be empty (whitespace only) up to `{`.
        let mut prelude_ok = true;
        loop {
            match self.bump() {
                None => return,
                Some(Token::Semicolon) | Some(Token::RBrace) => return,
                Some(Token::LBrace) => break,
                Some(Token::Whitespace) => {}
                Some(_) => prelude_ok = false,
            }
        }
        // Collect the block's tokens (bounded; see
        // [`MAX_FONT_FACE_TOKENS`]).
        let mut toks = Vec::new();
        let mut depth: u32 = 1;
        while depth > 0 {
            match self.bump() {
                None => break,
                Some(Token::LBrace) => depth += 1,
                Some(Token::RBrace) => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                Some(t) => {
                    if toks.len() < MAX_FONT_FACE_TOKENS {
                        toks.push(t);
                    }
                }
            }
        }
        if !prelude_ok {
            return;
        }
        let mut family = String::new();
        let mut src = Vec::new();
        let mut weight = 400u16;
        let mut weight_range = None;
        let mut style = FontFaceStyle::Normal;
        for decl in split_declarations(&toks) {
            let Some((name, value)) = split_declaration_name(decl) else {
                continue;
            };
            // Descriptor names are ASCII case-insensitive.
            if name.eq_ignore_ascii_case("font-family") {
                // A single family name: a string (`'Ahem'`) or a bare
                // identifier (`Ahem`).
                if let Some(n) = value
                    .iter()
                    .find(|t| !matches!(t, Token::Whitespace))
                    .and_then(|t| match t {
                        Token::String(s) | Token::Ident(s) => Some(s.as_str()),
                        _ => None,
                    })
                {
                    family = n.to_owned();
                }
            } else if name.eq_ignore_ascii_case("src") {
                src.extend(parse_font_face_src(value));
            } else if name.eq_ignore_ascii_case("font-weight") {
                (weight, weight_range) = parse_font_face_weight(value);
            } else if name.eq_ignore_ascii_case("font-style") {
                style = parse_font_face_style(value);
            }
            if src.len() >= MAX_FONT_FACE_SRCS {
                src.truncate(MAX_FONT_FACE_SRCS);
                break;
            }
        }
        if !family.is_empty() && !src.is_empty() {
            let order = self.font_faces.len();
            self.font_faces.push(FontFaceRule {
                family,
                src,
                weight,
                weight_range,
                style,
                order,
            });
        }
    }

    /// Consume tokens up to (not including) `{`, `;`, `}` or EOF at depth 0.
    fn consume_prelude(&mut self) -> Vec<Token> {
        let mut out = Vec::new();
        let mut depth: u32 = 0;
        loop {
            match self.peek() {
                None => break,
                Some(Token::LBrace) | Some(Token::Semicolon) | Some(Token::RBrace)
                    if depth == 0 =>
                {
                    break;
                }
                Some(Token::LParen) | Some(Token::Function(_)) | Some(Token::LBracket) => {
                    depth += 1;
                    out.push(self.bump().expect("peeked token"));
                }
                Some(Token::RParen) | Some(Token::RBracket) => {
                    depth = depth.saturating_sub(1);
                    out.push(self.bump().expect("peeked token"));
                }
                Some(_) => out.push(self.bump().expect("peeked token")),
            }
        }
        out
    }

    /// Skip an at-rule's prelude and optional `{ ... }` block.
    fn skip_at_prelude_and_block(&mut self) {
        // Prelude: up to `;` (no block) or the opening `{`.
        loop {
            match self.bump() {
                None => return,
                Some(Token::Semicolon) => return,
                Some(Token::LBrace) => break,
                Some(Token::RBrace) => return,
                Some(_) => {}
            }
        }
        // Balanced block.
        let mut depth: u32 = 1;
        loop {
            match self.bump() {
                None => return,
                Some(Token::LBrace) => depth += 1,
                Some(Token::RBrace) => {
                    depth -= 1;
                    if depth == 0 {
                        return;
                    }
                }
                Some(_) => {}
            }
        }
    }

    /// Skip an at-rule nested inside a declaration block.
    fn skip_at_rule_in_block(&mut self) {
        self.skip_at_prelude_and_block();
    }

    /// Skip a balanced `{ ... }` block whose opening brace was already
    /// consumed. Used when `@media` nesting exceeds [`MAX_MEDIA_DEPTH`]:
    /// the block is dropped without parsing (per CSS Syntax, a dropped
    /// at-rule block drops its nested rules too).
    fn skip_balanced_block(&mut self) {
        let mut depth: u32 = 1;
        while depth > 0 {
            match self.bump() {
                None => return,
                Some(Token::LBrace) => depth += 1,
                Some(Token::RBrace) => depth -= 1,
                _ => {}
            }
        }
    }

    fn parse_qualified_rule(&mut self) -> Option<(Vec<Selector>, Vec<Declaration>)> {
        let prelude = self.consume_prelude();
        if !matches!(self.peek(), Some(Token::LBrace)) {
            // Ran into `;`, `}` or EOF before any block: not a rule.
            return None;
        }
        self.bump(); // '{'
        let selectors = parse_selector_list_from_tokens(&prelude).ok()?;
        let declarations = self.parse_declaration_block();
        Some((selectors, declarations))
    }

    fn parse_declaration_block(&mut self) -> Vec<Declaration> {
        let mut out = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                None => return out,
                Some(Token::RBrace) => {
                    self.bump();
                    return out;
                }
                Some(Token::Semicolon) => {
                    self.bump();
                }
                Some(Token::AtKeyword(_)) => self.skip_at_rule_in_block(),
                Some(Token::Ident(_)) => {
                    if let Some(decls) = self.parse_declaration() {
                        out.extend(decls);
                    }
                }
                _ => self.skip_to_decl_end(),
            }
        }
    }

    /// Skip to the end of the current declaration (`;` consumed, `}` left).
    fn skip_to_decl_end(&mut self) {
        loop {
            match self.peek() {
                None => return,
                Some(Token::Semicolon) => {
                    self.bump();
                    return;
                }
                Some(Token::RBrace) => return,
                Some(_) => {
                    self.bump();
                }
            }
        }
    }

    fn parse_declaration(&mut self) -> Option<Vec<Declaration>> {
        let prop = match self.bump() {
            Some(Token::Ident(p)) => {
                // Custom property names are case-sensitive; everything else
                // is ASCII case-insensitive. Interning lowercases in place
                // and keeps well-known names allocation-free.
                if p.starts_with("--") {
                    PropertyName::from(p)
                } else {
                    PropertyName::intern_lowered_owned(p)
                }
            }
            _ => return None,
        };
        self.skip_ws();
        if !matches!(self.peek(), Some(Token::Colon)) {
            self.skip_to_decl_end();
            return None;
        }
        self.bump(); // ':'

        // Find the value token range, tracking ()/[] depth so `;` and `}`
        // inside functions don't end the declaration. Every value parser
        // below takes `&[Token]`, so the range borrows the token slice
        // directly: no per-token clone (the old `Vec<Token>` collection
        // cloned each token's `String` payload once per declaration).
        let (start, mut end) = {
            let start = self.pos;
            let mut end = start;
            let mut depth: u32 = 0;
            loop {
                match self.peek() {
                    None => break,
                    Some(Token::Semicolon) if depth == 0 => {
                        self.bump();
                        break;
                    }
                    Some(Token::RBrace) if depth == 0 => break,
                    Some(Token::LParen) | Some(Token::Function(_)) | Some(Token::LBracket) => {
                        depth += 1;
                    }
                    Some(Token::RParen) | Some(Token::RBracket) => {
                        depth = depth.saturating_sub(1);
                    }
                    Some(_) => {}
                }
                self.bump();
                end = self.pos;
            }
            (start, end)
        };

        while end > start && matches!(self.toks[end - 1], Token::Whitespace) {
            end -= 1;
        }
        let mut important = false;
        if let Some(stripped_end) = strip_important_end(&self.toks[start..end]) {
            important = true;
            end = start + stripped_end;
            while end > start && matches!(self.toks[end - 1], Token::Whitespace) {
                end -= 1;
            }
        }
        if start == end {
            return None;
        }
        let toks: &[Token] = &self.toks[start..end];

        // `grid-template-columns`/`grid-template-rows` take a
        // `<track-list>` (CSS Grid §7.1), which has no commas: a
        // top-level comma makes the whole declaration invalid. The
        // generic value parser flattens single-item comma groups (so
        // `100px, 200px` would parse exactly like `100px 200px`), and
        // post-parse the comma-ness is unrecoverable — so the comma is
        // detected here, at the token level, before parsing. Commas
        // nested inside functions (`minmax(…)`, `repeat(…)`, `var(…)`)
        // are not top-level and are unaffected; a comma smuggled in via
        // `var()` substitution is rejected by the cascade's
        // grammar-check site instead (see `value_contains_nested_list`).
        if (prop == "grid-template-columns" || prop == "grid-template-rows")
            && has_top_level_comma(toks)
        {
            return None;
        }

        // `font` needs the `/` in `12px/1.5`, which the generic value
        // parser cannot represent; split on `/` at the token level and
        // store the `/`-separated form as a nested-list marker the value
        // model can represent (see `parse_slash_aware_value`). A font
        // shorthand containing var() cannot be expanded until the cascade
        // substitutes the references, so it is kept as-is (see
        // `expand_shorthand` for the contract); the cascade-side expansion
        // (`expand_font_value` in cascade.rs) reads the marker back.
        if prop == "font" {
            if toks
                .iter()
                .any(|t| matches!(t, Token::Function(name) if name.eq_ignore_ascii_case("var")))
            {
                let value = parse_deferred_font_value(toks)?;
                // `prop` is already the interned name: move it, don't
                // re-intern.
                return Some(vec![decl(prop, value, important)]);
            }
            return parse_font_shorthand(toks, important);
        }

        // `border-radius` needs the `/` between horizontal and vertical
        // radii, which the generic value parser cannot represent; parse it
        // from tokens directly (like `font` with its `/` between size and
        // line-height). A border-radius containing var() is kept as-is
        // under the shorthand name (see `expand_shorthand` for the
        // contract).
        if prop == "border-radius" {
            return parse_border_radius(toks, important);
        }

        // `grid-column` / `grid-row` place items with `start / end` line
        // syntax; the `/` cannot survive the generic value parser, so
        // split on it like `font` and `border-radius`. The `/`-separated
        // form is stored as `List([List(before), List(after)])`.
        if prop == "grid-column" || prop == "grid-row" {
            let value = parse_slash_aware_value(toks)?;
            return Some(expand_shorthand(&prop, value, important));
        }

        // `aspect-ratio` takes `auto || <ratio>` (css-sizing-4 §3.1);
        // the `/` in `16 / 9` cannot survive the generic value parser,
        // so parse it from tokens directly (like `font` and
        // `border-radius`). A value containing `var()` is kept in the
        // `/`-separated marker shape and folded by the cascade after
        // substitution (see `fold_aspect_ratio`).
        if prop == "aspect-ratio" {
            let value = parse_aspect_ratio(toks)?;
            return Some(expand_shorthand(&prop, value, important));
        }

        // `filter` takes filter functions (`blur()`, `drop-shadow()`,
        // …) that the generic value parser rejects (they are not
        // transform functions), so parse it from tokens directly (like
        // `font` and `border-radius`; see `cbprops::parse_filter_value`).
        // `backdrop-filter` takes the same `<filter-value-list>` grammar
        // (Filter Effects 2 §2), so it shares the parser. A top-level
        // `var()` reference keeps its `Var` node for the
        // cascade to substitute.
        if prop == "filter" || prop == "backdrop-filter" {
            let value = crate::cbprops::parse_filter_value(toks)?;
            return Some(expand_shorthand(&prop, value, important));
        }

        // `clip-path` takes a `<basic-shape>` function (`inset()`,
        // `circle()`, `ellipse()`, `polygon()`), `none`, or `url()`
        // (css-clip-path-1 §4), parsed from tokens directly (like
        // `filter` above; see `parse_clip_path_value`). A `var()`
        // anywhere in the value defers to the generic value parser and
        // keeps its `Var` node: the cascade substitutes it and the
        // generic parser's `<basic-shape>` arms (see `parse_function`
        // in value.rs) reassemble the `ClipPath`.
        if prop == "clip-path" {
            let value = parse_clip_path_value(toks)?;
            return Some(expand_shorthand(&prop, value, important));
        }

        // `box-shadow` / `text-shadow` take comma-separated `<shadow>`
        // layers (CSS Backgrounds 3 §7 / CSS Text Decoration 3 §4.3).
        // The generic value parser flattens single-item comma groups, so
        // the comma structure is split here at the token level first
        // (see `parse_shadow_value`); each group's whitespace-separated
        // components then parse through the generic value parser,
        // reusing its length/color parsers. A `var()` anywhere in the
        // value defers to the generic parser and keeps its `Var` node
        // for the cascade to substitute (the cascade's
        // `fold_shadow_value` assembles the substituted components).
        if prop == "box-shadow" || prop == "text-shadow" {
            let value = parse_shadow_value(toks, prop == "box-shadow")?;
            return Some(expand_shorthand(&prop, value, important));
        }

        // Custom properties are token-preserving: store the raw token
        // stream verbatim and parse it only when a `var()` reference
        // substitutes it (see `Styler::resolve` in cascade.rs). Parsing
        // eagerly here mangles the stream (a top-level `/`, as in
        // `--f: italic 12px/1.5 serif`, cannot survive the generic value
        // parser) and wrongly drops valid declarations such as
        // `--x: 1/2/3`. Leading whitespace is trimmed, matching the
        // trailing trim above; `expand_shorthand` keeps custom
        // properties as-is.
        let value = if prop.starts_with("--") {
            let start = toks
                .iter()
                .take_while(|t| matches!(t, Token::Whitespace))
                .count();
            CssValue::Tokens(toks[start..].to_vec())
        } else {
            parse_value(toks).ok()?
        };
        Some(expand_shorthand(&prop, value, important))
    }
}

/// Split an `@font-face` block's token slice into declarations on
/// top-level `;` (nesting depth tracked so `;` inside `url(...)` /
/// `format(...)` does not split).
fn split_declarations(toks: &[Token]) -> Vec<&[Token]> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut depth: u32 = 0;
    for (i, t) in toks.iter().enumerate() {
        match t {
            Token::Function(_) | Token::LParen | Token::LBracket | Token::LBrace => depth += 1,
            Token::RParen | Token::RBracket | Token::RBrace => depth = depth.saturating_sub(1),
            Token::Semicolon if depth == 0 => {
                out.push(&toks[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < toks.len() {
        out.push(&toks[start..]);
    }
    out
}

/// Split one declaration slice into its (lowercased) property name and
/// value tokens at the first top-level `:`. `None` when there is no
/// colon or the name is not a single identifier.
fn split_declaration_name(decl: &[Token]) -> Option<(&str, &[Token])> {
    let mut depth: u32 = 0;
    for (i, t) in decl.iter().enumerate() {
        match t {
            Token::Function(_) | Token::LParen | Token::LBracket | Token::LBrace => depth += 1,
            Token::RParen | Token::RBracket | Token::RBrace => depth = depth.saturating_sub(1),
            Token::Colon if depth == 0 => {
                let name = decl[..i]
                    .iter()
                    .find(|t| !matches!(t, Token::Whitespace))
                    .and_then(|t| match t {
                        Token::Ident(n) => Some(n.as_str()),
                        _ => None,
                    })?;
                // The name must be exactly one identifier (no stray
                // tokens around it).
                let non_ws = decl[..i]
                    .iter()
                    .filter(|t| !matches!(t, Token::Whitespace))
                    .count();
                if non_ws != 1 {
                    return None;
                }
                return Some((name, &decl[i + 1..]));
            }
            _ => {}
        }
    }
    None
}

/// Parse an `@font-face` `src` descriptor value into its entries:
/// comma-separated `url(...)` (with optional `format(...)` hint) and
/// `local(...)` items. Both the single-token `url(…)` form and the
/// functional `url("…")` form are accepted; malformed entries are
/// skipped without disturbing the rest.
fn parse_font_face_src(toks: &[Token]) -> Vec<FontFaceSrc> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut depth: u32 = 0;
    // Comma-split the descriptor list, tracking function/paren depth.
    let mut comma_split = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        match t {
            Token::Function(_) | Token::LParen | Token::LBracket => depth += 1,
            Token::RParen | Token::RBracket => depth = depth.saturating_sub(1),
            Token::Comma if depth == 0 => {
                comma_split.push(&toks[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    comma_split.push(&toks[start..]);
    for part in comma_split {
        if out.len() >= MAX_FONT_FACE_SRCS {
            break;
        }
        // Significant tokens: whitespace never matters inside one entry.
        let sig: Vec<&Token> = part
            .iter()
            .filter(|t| !matches!(t, Token::Whitespace))
            .collect();
        let mut it = sig.iter().peekable();
        match it.next() {
            Some(Token::Url(u)) => {
                let format = parse_format_hint(&mut it);
                out.push(FontFaceSrc::Url {
                    url: (*u).clone(),
                    format,
                });
            }
            Some(Token::Function(f)) if f.eq_ignore_ascii_case("url") => {
                let url = match it.next() {
                    Some(Token::String(s)) | Some(Token::Ident(s)) => (*s).clone(),
                    _ => continue,
                };
                // Consume through the closing paren; anything after is
                // an optional `format(...)` hint.
                for t in it.by_ref() {
                    if matches!(t, Token::RParen) {
                        break;
                    }
                }
                let format = parse_format_hint(&mut it);
                out.push(FontFaceSrc::Url { url, format });
            }
            Some(Token::Function(f)) if f.eq_ignore_ascii_case("local") => {
                let name = match it.next() {
                    Some(Token::String(s)) | Some(Token::Ident(s)) => (*s).clone(),
                    _ => continue,
                };
                out.push(FontFaceSrc::Local(name));
            }
            _ => {}
        }
    }
    out
}

/// Parse an optional `format("…")` hint following a `src` URL entry.
/// `it` is positioned just after the URL (and its closing paren).
fn parse_format_hint<'a, 'b>(
    it: &mut std::iter::Peekable<std::slice::Iter<'a, &'b Token>>,
) -> Option<String> {
    match it.next() {
        Some(Token::Function(f)) if f.eq_ignore_ascii_case("format") => {
            let hint = match it.next() {
                Some(Token::String(s)) | Some(Token::Ident(s)) => Some((*s).clone()),
                _ => None,
            };
            for t in it.by_ref() {
                if matches!(t, Token::RParen) {
                    break;
                }
            }
            hint
        }
        _ => None,
    }
}

/// Parse an `@font-face` `font-weight` descriptor value (CSS Fonts 4
/// §4.3): `normal` (400), `bold` (700), a single `<number>`, or a
/// `<number> <number>` range. Anything else (including `auto` and a
/// reversed range) is ignored, leaving the default 400 — an invalid
/// descriptor never drops the rule.
fn parse_font_face_weight(toks: &[Token]) -> (u16, Option<(u16, u16)>) {
    let sig: Vec<&Token> = toks
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace))
        .collect();
    match sig.as_slice() {
        [Token::Ident(kw)] if kw.eq_ignore_ascii_case("normal") => (400, None),
        [Token::Ident(kw)] if kw.eq_ignore_ascii_case("bold") => (700, None),
        [Token::Number { value, .. }] => (weight_number(*value), None),
        [
            Token::Number { value: lo, .. },
            Token::Number { value: hi, .. },
        ] => {
            let (lo, hi) = (weight_number(*lo), weight_number(*hi));
            if lo <= hi {
                (lo, Some((lo, hi)))
            } else {
                (400, None)
            }
        }
        _ => (400, None),
    }
}

/// Clamp a `font-weight` number token to the CSS 1..=1000 range.
fn weight_number(value: f32) -> u16 {
    value.round().clamp(1.0, 1000.0) as u16
}

/// Parse an `@font-face` `font-style` descriptor value (CSS Fonts 4
/// §4.2): `normal`, `italic`, or `oblique` (an optional trailing angle
/// after `oblique` is ignored). Anything else leaves the default.
fn parse_font_face_style(toks: &[Token]) -> FontFaceStyle {
    let first = toks.iter().find(|t| !matches!(t, Token::Whitespace));
    match first {
        Some(Token::Ident(kw)) if kw.eq_ignore_ascii_case("italic") => FontFaceStyle::Italic,
        Some(Token::Ident(kw)) if kw.eq_ignore_ascii_case("oblique") => FontFaceStyle::Oblique,
        _ => FontFaceStyle::Normal,
    }
}

/// Whether any token is a `var()` function reference, at any nesting
/// depth (including inside `calc()`). A shadow value containing one
/// cannot be validated until the cascade substitutes it.
fn tokens_contain_var(tokens: &[Token]) -> bool {
    tokens
        .iter()
        .any(|t| matches!(t, Token::Function(name) if name.eq_ignore_ascii_case("var")))
}

/// Parse a `box-shadow` / `text-shadow` declaration value from its token
/// slice (CSS Backgrounds 3 §7 / CSS Text Decoration 3 §4.3):
/// `none | <shadow>#`, one comma-separated `<shadow>` per layer.
///
/// The comma groups are split at the token level first: the generic
/// value parser flattens single-item comma groups, so post-parse the
/// comma structure is unrecoverable (the same reason `filter` parses
/// from tokens). Each group's whitespace-separated components parse
/// through the generic value parser — reusing its length/color parsers
/// — and assemble into a [`Shadow`] via [`Shadow::from_components`].
/// `allow_inset` selects the `box-shadow` grammar (`inset?` and a
/// spread radius allowed) versus `text-shadow` (neither).
///
/// `none` (case-insensitive) parses to an empty [`CssValue::Shadows`]
/// list. A lone ident that is not `none` passes through as a bare
/// [`CssValue::Keyword`] for the cascade's CSS-wide-keyword path (the
/// cascade's grammar check drops anything that is not a wide keyword).
/// A `var()` anywhere in the value defers to the generic value parser
/// and keeps its `Var` node: the cascade substitutes it and
/// `fold_shadow_value` assembles the substituted components.
///
/// Anything else — wrong component counts, a second color, a second
/// `inset`, `inset`/spread on `text-shadow`, a negative blur radius,
/// empty comma groups, stray tokens — is invalid and the declaration
/// drops.
fn parse_shadow_value(tokens: &[Token], allow_inset: bool) -> Option<CssValue> {
    if tokens_contain_var(tokens) {
        return parse_value(tokens).ok();
    }
    let non_ws: Vec<&Token> = tokens
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace | Token::Eof | Token::Semicolon))
        .collect();
    match non_ws.as_slice() {
        [Token::Ident(kw)] if kw.eq_ignore_ascii_case("none") => {
            return Some(CssValue::Shadows(Vec::new()));
        }
        // A lone ident: a CSS-wide keyword passes through for the
        // cascade's wide-keyword path; anything else (e.g. a bare
        // `inset` or color) dies in the cascade's grammar check.
        [Token::Ident(kw)] => {
            return Some(CssValue::Keyword(kw.to_ascii_lowercase().into()));
        }
        _ => {}
    }
    // One `Vec<CssValue>` of whitespace-separated components per
    // top-level comma group. `split_top_comma` returns `None` both when
    // there is no top-level comma and when a group fails to parse, so
    // the comma is detected first: a present-but-unparseable comma
    // group must drop the declaration, not fall back to the flattened
    // comma-less parse.
    let groups: Vec<Vec<CssValue>> = if has_top_level_comma(tokens) {
        split_top_comma(tokens, false)?
    } else {
        vec![as_list(&parse_value(tokens).ok()?)]
    };
    let mut shadows = Vec::with_capacity(groups.len());
    for group in &groups {
        shadows.push(Shadow::from_components(group, allow_inset)?);
    }
    if shadows.is_empty() {
        return None;
    }
    Some(CssValue::Shadows(shadows))
}

/// Whether `name` is a `<geometry-box>` keyword (css-clip-path-1 §4):
/// `clip-path` accepts `[<basic-shape> || <geometry-box>]`, but the
/// reference box is always the border box here, so the box is accepted
/// and ignored.
fn is_geometry_box(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "border-box"
            | "padding-box"
            | "content-box"
            | "margin-box"
            | "fill-box"
            | "stroke-box"
            | "view-box"
    )
}

/// Parse a `clip-path` declaration value from its token slice
/// (css-clip-path-1 §4): `none | <basic-shape> | url()`.
///
/// `none` (case-insensitive) parses to [`ClipPath::None`]. A lone ident
/// that is not `none` passes through as a bare [`CssValue::Keyword`]
/// for the cascade's CSS-wide-keyword path. `url(<unquoted>)`
/// tokenizes as a single [`Token::Url`]; `url("…")` as a function with
/// a string body; both become [`ClipPath::Url`]. A trailing
/// `<geometry-box>` keyword is accepted and ignored (the reference box
/// is always the border box). Anything else goes through
/// [`parse_basic_shape`](crate::value::parse_basic_shape).
///
/// A `var()` anywhere in the value defers to the generic value parser
/// and keeps its `Var` node: the cascade substitutes it and the
/// generic parser's `<basic-shape>` arms reassemble the `ClipPath`
/// (see `parse_function` in value.rs).
fn parse_clip_path_value(tokens: &[Token]) -> Option<CssValue> {
    if tokens_contain_var(tokens) {
        return parse_value(tokens).ok();
    }
    // Trim leading/trailing whitespace (and sentinels), but keep
    // interior whitespace intact: `parse_basic_shape` splits its shape
    // body on `Token::Whitespace` (see `shape_ws_groups`,
    // `parse_shape_position`, and the polygon point-pair splitter in
    // value.rs), so stripping whitespace everywhere silently drops
    // every multi-token shape (`inset(10px 20px)`, `circle(50% at
    // 20% 30%)`, all of `polygon(...)`, ...).
    fn is_edge(t: &Token) -> bool {
        matches!(t, Token::Whitespace | Token::Eof | Token::Semicolon)
    }
    let mut lo = 0;
    while lo < tokens.len() && is_edge(&tokens[lo]) {
        lo += 1;
    }
    let mut hi = tokens.len();
    while hi > lo && is_edge(&tokens[hi - 1]) {
        hi -= 1;
    }
    let toks: &[Token] = &tokens[lo..hi];
    match toks {
        [Token::Ident(kw)] if kw.eq_ignore_ascii_case("none") => {
            return Some(CssValue::ClipPath(ClipPath::None));
        }
        // A lone ident: a CSS-wide keyword passes through for the
        // cascade's wide-keyword path; anything else dies in the
        // cascade's grammar check.
        [Token::Ident(kw)] => {
            return Some(CssValue::Keyword(kw.to_ascii_lowercase().into()));
        }
        [Token::Url(u)] => return Some(CssValue::ClipPath(ClipPath::Url(u.clone()))),
        _ => {}
    }
    // An optional trailing `<geometry-box>`: accepted, ignored.
    // Trailing whitespace is already trimmed, so the last token of a
    // `<basic-shape> <geometry-box>` pair is the ident itself.
    let mut toks: &[Token] = toks;
    if let [rest @ .., Token::Ident(gb)] = toks
        && is_geometry_box(gb)
    {
        toks = rest;
    }
    match toks {
        [Token::Function(name), rest @ ..] => {
            let body = rest.strip_suffix(&[Token::RParen])?;
            let lower = name.to_ascii_lowercase();
            if lower == "url" {
                // `url("…")` tokenizes as a function with a string body.
                let body: Vec<&Token> = body
                    .iter()
                    .filter(|t| !matches!(t, Token::Whitespace | Token::Eof))
                    .collect();
                match body.as_slice() {
                    [Token::String(s)] => Some(CssValue::ClipPath(ClipPath::Url(s.clone()))),
                    _ => None,
                }
            } else {
                parse_basic_shape(&lower, body).ok().map(CssValue::ClipPath)
            }
        }
        _ => None,
    }
}

/// If `toks` ends with `!important` (whitespace allowed around it), return
/// the length of the token prefix without that suffix.
fn strip_important_end(toks: &[Token]) -> Option<usize> {
    let mut i = toks.len();
    while i > 0 && matches!(toks[i - 1], Token::Whitespace) {
        i -= 1;
    }
    if i == 0 {
        return None;
    }
    if !matches!(&toks[i-1], Token::Ident(s) if s.eq_ignore_ascii_case("important")) {
        return None;
    }
    i -= 1;
    while i > 0 && matches!(toks[i - 1], Token::Whitespace) {
        i -= 1;
    }
    if i == 0 || !matches!(toks[i - 1], Token::Delim('!')) {
        return None;
    }
    i -= 1;
    Some(i)
}

fn parse_media_query(tokens: &[Token]) -> MediaQuery {
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
    let mut parts: Vec<MediaPart> = groups.iter().map(|g| parse_media_part(g)).collect();
    if parts.is_empty() {
        parts.push(MediaPart::default());
    }
    MediaQuery { parts }
}

fn parse_media_part(tokens: &[Token]) -> MediaPart {
    let t: Vec<&Token> = tokens
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace | Token::Eof))
        .collect();
    let mut part = MediaPart::default();
    let mut i = 0;
    // Legacy `only` keyword: `@media only screen` behaves like
    // `@media screen`. It must directly precede a media type; anything
    // else (`only` alone, `only not ...`, `only (...)`) is unknown.
    if matches!(t.get(i), Some(Token::Ident(s)) if s.eq_ignore_ascii_case("only")) {
        i += 1;
        if !matches!(t.get(i), Some(Token::Ident(s)) if !s.eq_ignore_ascii_case("not")) {
            part.unsupported = true;
            return part;
        }
    }
    if matches!(t.get(i), Some(Token::Ident(s)) if s.eq_ignore_ascii_case("not")) {
        part.negate = true;
        i += 1;
    }
    // Optional media type: an ident followed by `and` or end of query.
    if let Some(Token::Ident(mt)) = t.get(i) {
        let followed_by_and =
            matches!(t.get(i + 1), Some(Token::Ident(a)) if a.eq_ignore_ascii_case("and"));
        if followed_by_and || i + 1 >= t.len() {
            part.media_type = Some(mt.to_ascii_lowercase());
            i += 1;
        }
    }
    // Conditions are `and`-separated: a missing `and` between conditions, a
    // leading `and`, or a trailing `and` makes the whole query unknown.
    let mut saw_condition = false;
    loop {
        let is_and = matches!(t.get(i), Some(Token::Ident(a)) if a.eq_ignore_ascii_case("and"));
        if is_and {
            // `and` is only valid between a media type / condition and the
            // next condition.
            if part.media_type.is_none() && !saw_condition {
                part.unsupported = true;
                break;
            }
            i += 1;
        } else if part.media_type.is_some() || saw_condition {
            match t.get(i) {
                None => break,
                _ => {
                    part.unsupported = true; // missing `and`
                    break;
                }
            }
        }
        match t.get(i) {
            None => {
                if is_and {
                    part.unsupported = true; // trailing `and`
                }
                break;
            }
            Some(Token::LParen) => {
                // Find the matching RParen.
                let mut depth: u32 = 0;
                let mut j = i;
                while j < t.len() {
                    match t[j] {
                        Token::LParen => depth += 1,
                        Token::RParen => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    j += 1;
                }
                if depth != 0 {
                    part.unsupported = true;
                    break;
                }
                parse_media_condition(&t[i + 1..j], &mut part);
                i = j + 1;
                saw_condition = true;
            }
            _ => {
                part.unsupported = true;
                break;
            }
        }
    }
    part
}

fn parse_media_condition(tokens: &[&Token], part: &mut MediaPart) {
    // `(min-width: 100px)` — exactly feature, colon, value. Trailing
    // garbage (`(min-width: 100px foo)`) or a unitless length
    // (`(min-width: 100)`) makes the query unknown.
    if tokens.len() != 3 {
        part.unsupported = true;
        return;
    }
    let feature = match tokens[0] {
        Token::Ident(f) => f.to_ascii_lowercase(),
        _ => {
            part.unsupported = true;
            return;
        }
    };
    if !matches!(tokens[1], Token::Colon) {
        part.unsupported = true;
        return;
    }
    let value_px = match tokens[2] {
        Token::Dimension { value: n, unit, .. } => match unit.to_ascii_lowercase().as_str() {
            "px" => *n,
            "em" | "rem" => n * 16.0,
            _ => {
                part.unsupported = true;
                return;
            }
        },
        // Lengths in media features require a unit.
        Token::Number { .. } => {
            part.unsupported = true;
            return;
        }
        _ => {
            part.unsupported = true;
            return;
        }
    };
    let is_min = match feature.as_str() {
        "min-width" | "min-height" => true,
        "max-width" | "max-height" => false,
        _ => {
            // Unknown media features evaluate to false per spec.
            part.unsupported = true;
            return;
        }
    };
    let is_height = feature.ends_with("-height");
    part.conditions.push(MediaCondition {
        is_min,
        is_height,
        value_px,
    });
}

// ---------------------------------------------------------------------------
// Shorthand expansion
// ---------------------------------------------------------------------------

fn decl(property: PropertyName, value: CssValue, important: bool) -> Declaration {
    // The name arrives already interned: `expand_shorthand` threads the
    // parse-time `PropertyName` through, and the shorthand builders intern
    // each literal at their call site, so no name is interned twice.
    Declaration {
        property,
        value,
        important,
    }
}

/// Split a value into its whitespace/comma-separated items.
fn as_list(value: &CssValue) -> Vec<CssValue> {
    match value {
        CssValue::List(items) => items.clone(),
        _ => vec![value.clone()],
    }
}

/// Expand `margin`, `padding`, `border-width`, `border-style`,
/// `border-color`, `scroll-margin`, `scroll-padding` with 1-4 values into
/// per-side longhands; `border` and `border-<side>` from
/// `<width> <style> <color>` components; `background`, `list-style`,
/// `font`, `flex` into their longhands; `overscroll-behavior` into its
/// x/y longhands.
/// Anything else is not a shorthand and is returned as-is.
///
/// Every shorthand expansion emits ALL of its longhands: components absent
/// from the shorthand are reset to their initial values, so
/// `background: red` after `background-image: url(a.png)` clears the image.
///
/// A shorthand whose value contains any `var()` reference is NOT expanded
/// here: substitution happens in the cascade, where custom property values
/// are known. The declaration is stored as-is under the shorthand property
/// name, and the cascade expands it after substitution.
///
/// Deferred-shorthand contract (for the cascade-side expansion):
///
/// * Input: `Declaration { property: "<shorthand>", value, important }`
///   where `value` is the parsed component list — a [`CssValue::List`] for
///   multi-component shorthands, a single [`CssValue`] otherwise —
///   possibly containing [`CssValue::Var`] nodes (with their fallbacks)
///   at any depth. E.g. `border: var(--w) solid red` arrives as
///   `Declaration { property: "border", value: List([Var("--w", None),
///   Keyword("solid"), Color(red)]), important: false }`.
/// * The cascade substitutes every `Var` against the element's custom
///   properties (using the var's fallback when the property is unset).
///   A `var()` with no fallback and no set property makes the declaration
///   invalid at computed-value time, so it is dropped (treated as unset).
/// * After successful substitution, the cascade runs the same expansion
///   the parser would have (`expand_border_sides`, `expand_column_rule`,
///   `expand_border_radius`, `expand_background`, `expand_list_style`,
///   `expand_box`, or the `font`
///   token parser),
///   emitting ALL longhands with omitted ones reset to initial values,
///   preserving `important`.
fn expand_shorthand(property: &PropertyName, value: CssValue, important: bool) -> Vec<Declaration> {
    if value_contains_var(&value) {
        // Already interned at parse time: clone, don't re-intern (custom
        // property names would otherwise pay a second sidecar
        // allocation).
        return vec![decl(property.clone(), value, important)];
    }
    match property.as_str() {
        "margin" | "padding" | "border-width" | "border-style" | "border-color"
        | "scroll-margin" | "scroll-padding" => expand_box(property, value, important),
        // CSS Position 3: the `inset` shorthand sets the four physical
        // offsets (`top`/`right`/`bottom`/`left`) with the 1-4-value box
        // rule. Unlike `margin` it does not prefix the side name, so it
        // gets its own expander rather than sharing `expand_box`.
        "inset" => expand_inset(value, important),
        // CSS Logical Properties 1: two-value shorthands for the
        // flow-relative longhands (`margin-inline: auto` centering and
        // friends).
        "margin-inline"
        | "margin-block"
        | "padding-inline"
        | "padding-block"
        | "inset-inline"
        | "inset-block"
        | "scroll-margin-inline"
        | "scroll-margin-block"
        | "scroll-padding-inline"
        | "scroll-padding-block"
        | "border-inline-width"
        | "border-inline-style"
        | "border-inline-color"
        | "border-block-width"
        | "border-block-style"
        | "border-block-color" => expand_logical_sides(property, value, important),
        "border" => expand_border_sides(&["border"], value, important),
        "border-top" | "border-right" | "border-bottom" | "border-left" => {
            expand_border_sides(&[property.as_str()], value, important)
        }
        "gap" | "grid-gap" => expand_gap(value, important),
        // CSS Box Alignment 3 §6.1: `place-self` sets `align-self` and
        // `justify-self` (one value sets both).
        "place-self" => expand_place_self(value, important),
        "overflow" => expand_overflow(value, important),
        "overscroll-behavior" => expand_overscroll_behavior(value, important),
        "columns" => expand_columns(value, important),
        "column-rule" => expand_column_rule(value, important),
        "background" => expand_background(value, important),
        "list-style" => expand_list_style(value, important),
        "flex" => expand_flex(value, important),
        _ => vec![decl(property.clone(), value, important)],
    }
}

/// Whether a parsed value contains a `var()` reference anywhere (including
/// nested inside lists and inside `calc()`/`min()`/`max()`/`clamp()`
/// math trees; a `Var` node itself always counts). A verbatim
/// [`CssValue::Tokens`] stream is scanned for `var()` function tokens.
fn value_contains_var(value: &CssValue) -> bool {
    match value {
        CssValue::Var(_, _) => true,
        CssValue::Math(e) => e.contains_var(),
        CssValue::List(items) => items.iter().any(value_contains_var),
        // A shadow layer's length fields can hold `calc(var(--x))`;
        // the color is always a plain color or `currentcolor`.
        CssValue::Shadows(shadows) => shadows.iter().any(|s| {
            [&s.offset_x, &s.offset_y, &s.blur, &s.spread, &s.color]
                .into_iter()
                .any(value_contains_var)
        }),
        CssValue::Tokens(toks) => toks
            .iter()
            .any(|t| matches!(t, Token::Function(name) if name.eq_ignore_ascii_case("var"))),
        // A `Gradient` retains its source verbatim: a `var(` in it means
        // the cascade must defer substitution. The parser no longer
        // emits these (see `parse_function`'s gradient arm), so this is
        // defense-in-depth for values built elsewhere.
        CssValue::Gradient { source, .. } => css_source_contains_var(source),
        _ => false,
    }
}

/// Expand a logical shorthand (`margin-inline`, `padding-block`,
/// `inset-inline`, `border-inline-width`, …; CSS Logical Properties 1)
/// into its two flow-relative longhands (`<shorthand>-start` and
/// `<shorthand>-end`). One or two values, like `gap`; anything else is
/// invalid and the declaration drops. A `var()`-containing value never
/// reaches here: `expand_shorthand` keeps it deferred and the cascade
/// expands it after substitution (see `expand_logical_sides_value` in
/// cascade.rs).
fn expand_logical_sides(property: &str, value: CssValue, important: bool) -> Vec<Declaration> {
    let items = as_list(&value);
    let [start, end] = match items.len() {
        1 => [items[0].clone(), items[0].clone()],
        2 => [items[0].clone(), items[1].clone()],
        _ => return Vec::new(), // invalid: drop the declaration
    };
    // A component outside the longhand grammar invalidates the whole
    // shorthand; CSS-wide keywords cannot be combined with other values.
    // A lone CSS-wide keyword still expands; the cascade intercepts it
    // per longhand.
    let lone_wide = items.len() == 1 && is_css_wide_keyword(&items[0]);
    if !lone_wide {
        if items.iter().any(is_css_wide_keyword) {
            return Vec::new(); // invalid: drop the declaration
        }
        let probe = if let Some(rest) = property.strip_prefix("border-") {
            let Some((axis, kind)) = rest.rsplit_once('-') else {
                // Defense-in-depth: every current dispatch arm names a
                // `<axis>-<kind>` shape, so this is unreachable today; a
                // future caller passing a no-dash name drops the
                // declaration instead of panicking.
                return Vec::new();
            };
            format!("border-{axis}-start-{kind}")
        } else {
            format!("{property}-start")
        };
        if !items.iter().all(|i| is_valid_for_property(&probe, i)) {
            return Vec::new(); // invalid: drop the declaration
        }
    }
    let longhand = |edge: &str| -> Option<String> {
        if let Some(rest) = property.strip_prefix("border-") {
            // `border-inline-width` -> `border-inline-start-width`.
            let (axis, kind) = rest.rsplit_once('-')?;
            Some(format!("border-{axis}-{edge}-{kind}"))
        } else {
            // `margin-inline` -> `margin-inline-start`.
            Some(format!("{property}-{edge}"))
        }
    };
    // Defense-in-depth: same as the probe split above; drop the
    // declaration instead of panicking on a no-dash name.
    let (start_name, end_name) = match (longhand("start"), longhand("end")) {
        (Some(start), Some(end)) => (start, end),
        _ => return Vec::new(),
    };
    vec![
        decl(PropertyName::from(start_name), start, important),
        decl(PropertyName::from(end_name), end, important),
    ]
}

/// Expand `gap` / `grid-gap` (`<row> || <column>`) into `row-gap` and
/// `column-gap`. The cascade fills every longhand with its initial value,
/// so the shorthand must expand at parse time: otherwise the layout
/// engine cannot tell a specified longhand from an initial one.
fn expand_gap(value: CssValue, important: bool) -> Vec<Declaration> {
    let items = as_list(&value);
    if items.len() > 2 {
        return Vec::new(); // invalid: drop the declaration
    }
    // A CSS-wide keyword cannot be combined with another value (the
    // lone-keyword case below already copies it to both longhands).
    if items.len() > 1 && items.iter().any(is_css_wide_keyword) {
        return Vec::new(); // invalid: drop the declaration
    }
    let row = items
        .first()
        .cloned()
        .unwrap_or(CssValue::Keyword("normal".into()));
    let col = items.get(1).cloned().unwrap_or_else(|| row.clone());
    vec![
        decl(PropertyName::intern("row-gap"), row, important),
        decl(PropertyName::intern("column-gap"), col, important),
    ]
}

/// Expand `place-self` (`<align-self> <justify-self>?`) into
/// `align-self` and `justify-self`; a lone value sets both. A
/// component outside the longhand grammar invalidates the whole
/// shorthand (it must not partially apply). A lone CSS-wide keyword
/// still expands; the cascade intercepts it per longhand. A `var()`
/// -containing value never reaches here: `expand_shorthand` keeps it
/// deferred and the cascade expands it after substitution (see
/// `expand_place_self_value` in cascade.rs).
fn expand_place_self(value: CssValue, important: bool) -> Vec<Declaration> {
    let items = as_list(&value);
    let lone_wide = items.len() == 1 && is_css_wide_keyword(&items[0]);
    if lone_wide {
        let kw = items.into_iter().next().unwrap();
        return vec![
            decl(PropertyName::intern("align-self"), kw.clone(), important),
            decl(PropertyName::intern("justify-self"), kw, important),
        ];
    }
    // The `place-self` validation arm encodes the 1-2-value grammar
    // against the two longhands (a leading `safe`/`unsafe` binds to the
    // following keyword as one longhand), so one call checks both
    // components.
    match split_place_self(&value) {
        Some((align, justify)) => vec![
            decl(PropertyName::intern("align-self"), align, important),
            decl(PropertyName::intern("justify-self"), justify, important),
        ],
        None => Vec::new(), // invalid: drop the declaration
    }
}

/// Expand `columns` (`<column-width> || <column-count>`) into
/// `column-width` and `column-count`. The cascade fills every longhand
/// with its initial value, so the shorthand must expand at parse time:
/// otherwise the layout engine cannot tell a specified longhand from an
/// initial one.
fn expand_columns(value: CssValue, important: bool) -> Vec<Declaration> {
    let items = as_list(&value);
    if items.is_empty() || items.len() > 2 {
        return Vec::new(); // invalid: drop the declaration
    }
    // w78fix H4: a lone CSS-wide keyword expands per longhand, so
    // `columns: inherit` is not a silent no-op.
    if items.len() == 1 && is_css_wide_keyword(&items[0]) {
        let kw = items[0].clone();
        return vec![
            decl(PropertyName::intern("column-width"), kw.clone(), important),
            decl(PropertyName::intern("column-count"), kw, important),
        ];
    }
    let auto = || CssValue::Keyword("auto".into());
    let mut width: Option<CssValue> = None;
    let mut count: Option<CssValue> = None;
    for item in items {
        let is_auto = matches!(&item, CssValue::Keyword(k) if k.eq_ignore_ascii_case("auto"));
        let is_count = matches!(&item, CssValue::Number(n) if n.fract() == 0.0 && *n >= 1.0);
        let is_width = matches!(
            &item,
            CssValue::Length(_, _) | CssValue::Math(_) | CssValue::MathDeferred { .. }
        ) || matches!(&item, CssValue::Number(n) if *n == 0.0)
            || is_auto;
        if is_auto {
            // `auto` fills whichever side is still empty (width first,
            // so `columns: auto 3` is width:auto count:3).
            if width.is_none() {
                width = Some(auto());
            } else if count.is_none() {
                count = Some(auto());
            } else {
                return Vec::new();
            }
        } else if is_count && count.is_none() {
            count = Some(item);
        } else if is_width && width.is_none() {
            if matches!(&item, CssValue::Length(n, _) if *n < 0.0) {
                return Vec::new();
            }
            width = Some(item);
        } else {
            return Vec::new(); // two widths, two counts, or junk
        }
    }
    vec![
        decl(
            PropertyName::intern("column-width"),
            width.unwrap_or_else(auto),
            important,
        ),
        decl(
            PropertyName::intern("column-count"),
            count.unwrap_or_else(auto),
            important,
        ),
    ]
}

/// Expand `overflow` (`<overflow-x> <overflow-y>?`) into `overflow-x` and
/// `overflow-y`. The cascade fills every longhand with its initial value,
/// so the shorthand must expand at parse time: otherwise the layout
/// engine cannot tell a specified longhand from an initial one, and
/// declaration order against the longhands (`overflow: hidden;
/// overflow-x: visible`) would be unobservable.
fn expand_overflow(value: CssValue, important: bool) -> Vec<Declaration> {
    let items = as_list(&value);
    if items.len() > 2 {
        return Vec::new(); // invalid: drop the declaration
    }
    // A component outside the longhand grammar invalidates the whole
    // shorthand: it must not partially apply. A lone CSS-wide keyword
    // still expands; the cascade intercepts it per longhand.
    let lone_wide = items.len() == 1 && is_css_wide_keyword(&items[0]);
    if !lone_wide && !items.iter().all(|i| is_valid_for_property("overflow-x", i)) {
        return Vec::new(); // invalid: drop the declaration
    }
    let x = items
        .first()
        .cloned()
        .unwrap_or(CssValue::Keyword("visible".into()));
    let y = items.get(1).cloned().unwrap_or_else(|| x.clone());
    vec![
        decl(PropertyName::intern("overflow-x"), x, important),
        decl(PropertyName::intern("overflow-y"), y, important),
    ]
}

/// Expand `overscroll-behavior` (`<overscroll-behavior-x>
/// <overscroll-behavior-y>?`) into `overscroll-behavior-x` and
/// `overscroll-behavior-y`: one value sets both axes, two values set x
/// then y. Like `overflow`, it expands at parse time so the cascade can
/// tell a specified longhand from an initial one and declaration order
/// against the longhands stays observable.
fn expand_overscroll_behavior(value: CssValue, important: bool) -> Vec<Declaration> {
    let items = as_list(&value);
    if items.len() > 2 {
        return Vec::new(); // invalid: drop the declaration
    }
    // A component outside the longhand grammar invalidates the whole
    // shorthand: it must not partially apply. A lone CSS-wide keyword
    // still expands; the cascade intercepts it per longhand.
    let lone_wide = items.len() == 1 && is_css_wide_keyword(&items[0]);
    if !lone_wide
        && !items
            .iter()
            .all(|i| is_valid_for_property("overscroll-behavior-x", i))
    {
        return Vec::new(); // invalid: drop the declaration
    }
    let x = items
        .first()
        .cloned()
        .unwrap_or(CssValue::Keyword("auto".into()));
    let y = items.get(1).cloned().unwrap_or_else(|| x.clone());
    vec![
        decl(PropertyName::intern("overscroll-behavior-x"), x, important),
        decl(PropertyName::intern("overscroll-behavior-y"), y, important),
    ]
}

fn expand_box(property: &str, value: CssValue, important: bool) -> Vec<Declaration> {
    let items = as_list(&value);
    let [top, right, bottom, left] = match items.len() {
        1 => [0, 0, 0, 0].map(|_| items[0].clone()),
        2 => [
            items[0].clone(),
            items[1].clone(),
            items[0].clone(),
            items[1].clone(),
        ],
        3 => [
            items[0].clone(),
            items[1].clone(),
            items[2].clone(),
            items[1].clone(),
        ],
        4 => [
            items[0].clone(),
            items[1].clone(),
            items[2].clone(),
            items[3].clone(),
        ],
        _ => return Vec::new(), // invalid: drop the declaration
    };
    // `border-width` -> `border-top-width`; `margin` -> `margin-top`.
    let longhand = |side: &str| {
        if let Some(rest) = property.strip_prefix("border-") {
            format!("border-{side}-{rest}")
        } else {
            format!("{property}-{side}")
        }
    };
    // A component outside the longhand grammar invalidates the whole
    // shorthand: it must not partially apply. A lone CSS-wide keyword
    // still expands; the cascade intercepts it per longhand.
    let probe = longhand("top");
    let lone_wide = items.len() == 1 && is_css_wide_keyword(&items[0]);
    if !lone_wide && !items.iter().all(|i| is_valid_for_property(&probe, i)) {
        return Vec::new(); // invalid: drop the declaration
    }
    ["top", "right", "bottom", "left"]
        .into_iter()
        .zip([top, right, bottom, left])
        .map(|(side, v)| decl(PropertyName::from(longhand(side)), v, important))
        .collect()
}

/// Expand `inset` (`<length-percentage> | auto`, 1-4 values) into the
/// four physical longhands `top`/`right`/`bottom`/`left`, following the
/// 1-4-value box rule. A component outside the longhand grammar
/// invalidates the whole shorthand; a lone CSS-wide keyword still
/// expands and the cascade intercepts it per longhand.
fn expand_inset(value: CssValue, important: bool) -> Vec<Declaration> {
    let items = as_list(&value);
    let [top, right, bottom, left] = match items.len() {
        1 => [0, 0, 0, 0].map(|_| items[0].clone()),
        2 => [
            items[0].clone(),
            items[1].clone(),
            items[0].clone(),
            items[1].clone(),
        ],
        3 => [
            items[0].clone(),
            items[1].clone(),
            items[2].clone(),
            items[1].clone(),
        ],
        4 => [
            items[0].clone(),
            items[1].clone(),
            items[2].clone(),
            items[3].clone(),
        ],
        _ => return Vec::new(), // invalid: drop the declaration
    };
    let lone_wide = items.len() == 1 && is_css_wide_keyword(&items[0]);
    if !lone_wide && !items.iter().all(|i| is_valid_for_property("top", i)) {
        return Vec::new(); // invalid: drop the declaration
    }
    ["top", "right", "bottom", "left"]
        .into_iter()
        .zip([top, right, bottom, left])
        .map(|(side, v)| decl(PropertyName::from(side), v, important))
        .collect()
}

fn is_border_style(kw: &str) -> bool {
    matches!(
        kw,
        "none"
            | "hidden"
            | "dotted"
            | "dashed"
            | "solid"
            | "double"
            | "groove"
            | "ridge"
            | "inset"
            | "outset"
    )
}

fn is_border_width(kw: &str) -> bool {
    matches!(kw, "thin" | "medium" | "thick")
}

/// `border: <width> <style> <color>` and `border-<side>: ...` in any order,
/// 1-3 components. `border` covers all four sides and expands to per-side
/// longhands (`border-top-width`, ...): the cascade only resolves per-side
/// longhands, so emitting the intermediate `border-width`/`border-style`/
/// `border-color` would silently drop the values. Every longhand is
/// emitted; components absent from the shorthand reset to their initial
/// values (`medium`, `none`, `currentcolor`). Unclassifiable items
/// (unknown keywords) drop the whole declaration; `var()` references never
/// reach here (see `expand_shorthand`).
fn expand_border_sides(prefixes: &[&str], value: CssValue, important: bool) -> Vec<Declaration> {
    let mut width = None;
    let mut style = None;
    let mut color = None;
    // w78fix H4: a lone CSS-wide keyword expands per longhand (as
    // `expand_box` does), so `border: inherit` sets every border
    // longhand to `inherit` instead of silently dropping the whole
    // declaration. A wide keyword combined with other components is
    // still invalid and drops, via the loop below.
    if let [item] = as_list(&value).as_slice()
        && is_css_wide_keyword(item)
    {
        let kw = item.clone();
        let mut out = Vec::new();
        let push_side = |prefix: &str, out: &mut Vec<Declaration>| {
            out.push(decl(
                PropertyName::from(format!("{prefix}-width")),
                kw.clone(),
                important,
            ));
            out.push(decl(
                PropertyName::from(format!("{prefix}-style")),
                kw.clone(),
                important,
            ));
            out.push(decl(
                PropertyName::from(format!("{prefix}-color")),
                kw.clone(),
                important,
            ));
        };
        for prefix in prefixes {
            if *prefix == "border" {
                for side in ["top", "right", "bottom", "left"] {
                    push_side(&format!("border-{side}"), &mut out);
                }
            } else {
                push_side(prefix, &mut out);
            }
        }
        return out;
    }
    for item in as_list(&value) {
        match item {
            // Unitless zero is a valid `<length>` (CSS-wide rule), and a
            // `calc()` width evaluates in the cascade after expansion.
            CssValue::Length(_, _) | CssValue::Math(_) => {
                if width.is_some() {
                    return Vec::new();
                }
                width = Some(item);
            }
            CssValue::Number(0.0) => {
                if width.is_some() {
                    return Vec::new();
                }
                width = Some(item);
            }
            CssValue::Color(_) => {
                if color.is_some() {
                    return Vec::new();
                }
                color = Some(item);
            }
            CssValue::Keyword(ref kw) => {
                let k = kw.as_str();
                if k == "currentcolor" {
                    if color.is_some() {
                        return Vec::new();
                    }
                    color = Some(item);
                } else if is_border_width(k) {
                    if width.is_some() {
                        return Vec::new();
                    }
                    width = Some(item);
                } else if is_border_style(k) {
                    if style.is_some() {
                        return Vec::new();
                    }
                    style = Some(item);
                } else {
                    return Vec::new();
                }
            }
            _ => return Vec::new(),
        }
    }
    // Omitted components reset to initial values.
    let width = width.unwrap_or(CssValue::Keyword("medium".into()));
    let style = style.unwrap_or(CssValue::Keyword("none".into()));
    let color = color.unwrap_or(CssValue::Keyword("currentcolor".into()));
    let mut out = Vec::new();
    let push_side = |prefix: &str, out: &mut Vec<Declaration>| {
        out.push(decl(
            PropertyName::from(format!("{prefix}-width")),
            width.clone(),
            important,
        ));
        out.push(decl(
            PropertyName::from(format!("{prefix}-style")),
            style.clone(),
            important,
        ));
        out.push(decl(
            PropertyName::from(format!("{prefix}-color")),
            color.clone(),
            important,
        ));
    };
    for prefix in prefixes {
        if *prefix == "border" {
            // All four sides: one longhand per side per component.
            for side in ["top", "right", "bottom", "left"] {
                push_side(&format!("border-{side}"), &mut out);
            }
        } else {
            push_side(prefix, &mut out);
        }
    }
    out
}

/// Expand `column-rule` (`<column-rule-width> || <column-rule-style> ||
/// <column-rule-color>`, in any order, 1-3 components) into
/// `column-rule-width`, `column-rule-style`, and `column-rule-color`.
/// Component classification matches `expand_border_sides`; unlike
/// `border` there are no sides, so the three longhands are emitted
/// directly and components absent from the shorthand reset to their
/// initial values (`medium`, `none`, `currentcolor`). Unclassifiable
/// items (unknown keywords) drop the whole declaration; `var()`
/// references never reach here (see `expand_shorthand`).
fn expand_column_rule(value: CssValue, important: bool) -> Vec<Declaration> {
    let mut width = None;
    let mut style = None;
    let mut color = None;
    for item in as_list(&value) {
        match item {
            // Unitless zero is a valid `<length>` (CSS-wide rule), and a
            // `calc()` width evaluates in the cascade after expansion.
            // Kept in sync with cascade.rs's `expand_column_rule_value`
            // (which also accepts `MathDeferred`, produced only in the
            // cascade, never by the parser).
            CssValue::Length(_, _) | CssValue::Math(_) | CssValue::MathDeferred { .. } => {
                if width.is_some() {
                    return Vec::new();
                }
                width = Some(item);
            }
            CssValue::Number(0.0) => {
                if width.is_some() {
                    return Vec::new();
                }
                width = Some(item);
            }
            CssValue::Color(_) => {
                if color.is_some() {
                    return Vec::new();
                }
                color = Some(item);
            }
            CssValue::Keyword(ref kw) => {
                let k = kw.as_str();
                if k == "currentcolor" {
                    if color.is_some() {
                        return Vec::new();
                    }
                    color = Some(item);
                } else if is_border_width(k) {
                    if width.is_some() {
                        return Vec::new();
                    }
                    width = Some(item);
                } else if is_border_style(k) {
                    if style.is_some() {
                        return Vec::new();
                    }
                    style = Some(item);
                } else {
                    return Vec::new();
                }
            }
            _ => return Vec::new(),
        }
    }
    // Omitted components reset to initial values.
    let width = width.unwrap_or(CssValue::Keyword("medium".into()));
    let style = style.unwrap_or(CssValue::Keyword("none".into()));
    let color = color.unwrap_or(CssValue::Keyword("currentcolor".into()));
    vec![
        decl(PropertyName::intern("column-rule-width"), width, important),
        decl(PropertyName::intern("column-rule-style"), style, important),
        decl(PropertyName::intern("column-rule-color"), color, important),
    ]
}

/// Map a `font`-shorthand dimension unit to [`LengthUnit`]. The shorthand
/// accepts any length unit for size/line-height (matching the deferred
/// cascade expansion, which takes any parsed `Length`).
fn font_length_unit(unit: &str) -> Option<LengthUnit> {
    Some(match unit.to_ascii_lowercase().as_str() {
        "px" => LengthUnit::Px,
        "cm" => LengthUnit::Cm,
        "mm" => LengthUnit::Mm,
        "in" => LengthUnit::In,
        "pt" => LengthUnit::Pt,
        "pc" => LengthUnit::Pc,
        "q" => LengthUnit::Q,
        "em" => LengthUnit::Em,
        "rem" => LengthUnit::Rem,
        "vw" => LengthUnit::Vw,
        "vh" => LengthUnit::Vh,
        "vmin" => LengthUnit::Vmin,
        "vmax" => LengthUnit::Vmax,
        _ => return None,
    })
}

/// Parse a math function (`calc()`/`min()`/`max()`/`clamp()` and the
/// other CSS Values 4 math functions, `if()`, `toggle()`) starting at
/// `t[i]` (a `Function` token) in a token-level shorthand parser.
/// Returns the parsed `Math` value and the index just past the closing
/// paren.
fn math_fn_at(t: &[&Token], i: usize) -> Option<(CssValue, usize)> {
    let name = match t.get(i) {
        Some(Token::Function(n)) => n.clone(),
        _ => return None,
    };
    if !is_math_fn_name(&name.to_ascii_lowercase()) {
        return None;
    }
    let mut depth: u32 = 0;
    let mut j = i + 1;
    while j < t.len() {
        match t[j] {
            Token::Function(_) | Token::LParen | Token::LBracket => depth += 1,
            Token::RParen if depth == 0 => {
                let v = parse_math_value(&name, &t[i + 1..j], 0).ok()?;
                return Some((v, j + 1));
            }
            Token::RParen | Token::RBracket => depth = depth.saturating_sub(1),
            _ => {}
        }
        j += 1;
    }
    None
}

/// `font: [style] [variant] [weight] size [/ line-height] family`.
/// Parsed from raw tokens because the `/` separator has no value form.
/// All six longhands are always emitted; omitted ones reset to `normal`.
///
/// Size and line-height accept any length unit plus math functions
/// (a `Math` value, evaluated later in the cascade), matching the
/// deferred cascade expansion in cascade.rs: the eager and deferred
/// paths must agree.
fn parse_font_shorthand(toks: &[Token], important: bool) -> Option<Vec<Declaration>> {
    let t: Vec<&Token> = toks
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace | Token::Eof))
        .collect();
    // System font keywords (`font: caption` etc.): per CSS Fonts 3 §3.2
    // the keyword must be the *sole* value and computes to the UA's font
    // for that purpose (`system_font_values` in value.rs). Anything else
    // alongside it (`font: caption 12px`) is invalid.
    if t.len() == 1
        && let Token::Ident(kw) = t[0]
        && let Some(vals) = system_font_values(&kw.to_ascii_lowercase())
    {
        let [style, variant, weight, size, line_height, family] = vals;
        return Some(vec![
            decl(PropertyName::intern("font-style"), style, important),
            decl(PropertyName::intern("font-variant"), variant, important),
            decl(PropertyName::intern("font-weight"), weight, important),
            decl(PropertyName::intern("font-size"), size, important),
            decl(PropertyName::intern("line-height"), line_height, important),
            decl(PropertyName::intern("font-family"), family, important),
        ]);
    }
    // w78fix H4: a lone CSS-wide keyword expands per longhand (as
    // `expand_box` does), so `font: inherit` sets every font
    // longhand to `inherit` instead of silently dropping.
    if t.len() == 1
        && let Token::Ident(kw) = t[0]
        && matches!(
            kw.to_ascii_lowercase().as_str(),
            "inherit" | "initial" | "unset" | "revert" | "revert-layer"
        )
    {
        let wide = CssValue::Keyword(kw.to_ascii_lowercase().into());
        return Some(vec![
            decl(PropertyName::intern("font-style"), wide.clone(), important),
            decl(
                PropertyName::intern("font-variant"),
                wide.clone(),
                important,
            ),
            decl(PropertyName::intern("font-weight"), wide.clone(), important),
            decl(PropertyName::intern("font-size"), wide.clone(), important),
            decl(PropertyName::intern("line-height"), wide.clone(), important),
            decl(PropertyName::intern("font-family"), wide, important),
        ]);
    }
    let mut i = 0;
    let mut style = None;
    let mut variant = None;
    let mut weight = None;
    while i < t.len() {
        match t[i] {
            Token::Ident(kw) => {
                let k = kw.to_ascii_lowercase();
                match k.as_str() {
                    "italic" | "oblique" => {
                        if style.is_some() {
                            return None;
                        }
                        style = Some(CssValue::Keyword(k.into()));
                    }
                    "small-caps" => {
                        if variant.is_some() {
                            return None;
                        }
                        variant = Some(CssValue::Keyword(k.into()));
                    }
                    "bold" | "bolder" | "lighter" => {
                        if weight.is_some() {
                            return None;
                        }
                        weight = Some(CssValue::Keyword(k.into()));
                    }
                    "normal" => {}
                    _ => break,
                }
                i += 1;
            }
            Token::Number { value: n, .. }
                if n.fract() == 0.0
                    && matches!(
                        *n as i32,
                        100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900
                    ) =>
            {
                if weight.is_some() {
                    return None;
                }
                weight = Some(CssValue::Number(*n));
                i += 1;
            }
            _ => break,
        }
    }
    let size;
    match t.get(i) {
        Some(Token::Dimension { value: n, unit, .. }) => {
            let u = font_length_unit(unit)?;
            size = CssValue::Length(*n, u);
            i += 1;
        }
        Some(Token::Percentage(p)) => {
            size = CssValue::Percentage(*p);
            i += 1;
        }
        Some(Token::Function(_)) => match math_fn_at(&t, i) {
            Some((v, next)) => {
                size = v;
                i = next;
            }
            None => return None,
        },
        _ => return None,
    }
    let mut line_height = None;
    if matches!(t.get(i), Some(Token::Delim('/'))) {
        i += 1;
        line_height = Some(match t.get(i) {
            Some(Token::Number { value: n, .. }) => {
                i += 1;
                CssValue::Number(*n)
            }
            Some(Token::Dimension { value: n, unit, .. }) => {
                let u = font_length_unit(unit)?;
                i += 1;
                CssValue::Length(*n, u)
            }
            Some(Token::Percentage(p)) => {
                i += 1;
                CssValue::Percentage(*p)
            }
            Some(Token::Ident(kw)) => {
                i += 1;
                CssValue::Keyword(kw.to_ascii_lowercase().into())
            }
            Some(Token::Function(_)) => match math_fn_at(&t, i) {
                Some((v, next)) => {
                    i = next;
                    v
                }
                None => return None,
            },
            _ => return None,
        });
    }
    if i >= t.len() {
        return None; // font requires a family
    }
    // Family: comma-separated; each part is a quoted string or ident runs.
    // A comma must separate families ( `"a" "b"` is invalid).
    let mut families = Vec::new();
    let mut part: Vec<String> = Vec::new();
    let mut last_was_family = false;
    let mut idx = i;
    while idx < t.len() {
        match t[idx] {
            Token::Comma => {
                if !part.is_empty() {
                    families.push(CssValue::String(part.join(" ")));
                    part.clear();
                    // A comma resets adjacency: a quoted family may follow
                    // it (`Arial, "Times New Roman"`), exactly like an
                    // ident run can. Only a string directly after a string
                    // (`"a" "b"`) is invalid.
                    last_was_family = false;
                } else if !last_was_family {
                    return None; // leading or doubled comma
                } else {
                    last_was_family = false;
                }
            }
            Token::String(s) => {
                if !part.is_empty() || last_was_family {
                    return None;
                }
                families.push(CssValue::String(s.clone()));
                last_was_family = true;
            }
            Token::Ident(name) => {
                part.push(name.clone());
                last_was_family = false;
            }
            _ => return None,
        }
        idx += 1;
    }
    if !part.is_empty() {
        families.push(CssValue::String(part.join(" ")));
    } else if !last_was_family {
        return None;
    }
    if matches!(t[t.len() - 1], Token::Comma) {
        return None; // trailing comma
    }
    let family = if families.len() == 1 {
        families.pop().expect("one family")
    } else {
        CssValue::List(families)
    };

    // Every longhand is emitted; omitted ones reset to their initial
    // values (`normal`).
    let normal = || CssValue::Keyword("normal".into());
    Some(vec![
        decl(
            PropertyName::intern("font-style"),
            style.unwrap_or_else(&normal),
            important,
        ),
        decl(
            PropertyName::intern("font-variant"),
            variant.unwrap_or_else(&normal),
            important,
        ),
        decl(
            PropertyName::intern("font-weight"),
            weight.unwrap_or_else(&normal),
            important,
        ),
        decl(PropertyName::intern("font-size"), size, important),
        decl(
            PropertyName::intern("line-height"),
            line_height.unwrap_or_else(&normal),
            important,
        ),
        decl(PropertyName::intern("font-family"), family, important),
    ])
}

/// `background: <color> <image> <repeat> <attachment> <position>` in any order.
/// All five longhands are always emitted; omitted components reset to
/// initial values (`transparent`, `none`, `repeat`, `scroll`, `0% 0%`).
fn expand_background(value: CssValue, important: bool) -> Vec<Declaration> {
    let mut color = None;
    let mut image = None;
    let mut repeat = None;
    let mut attachment = None;
    let mut position: Vec<CssValue> = Vec::new();
    for item in as_list(&value) {
        match item {
            CssValue::Color(_) => {
                if color.is_some() {
                    return Vec::new();
                }
                color = Some(item);
            }
            CssValue::Url(_) | CssValue::Gradient { .. } => {
                if image.is_some() {
                    return Vec::new();
                }
                image = Some(item);
            }
            CssValue::Length(_, _) | CssValue::Percentage(_) => position.push(item),
            CssValue::Keyword(kw) => {
                let k = kw.as_str();
                if k == "currentcolor" {
                    if color.is_some() {
                        return Vec::new();
                    }
                    color = Some(CssValue::Keyword(kw));
                } else if matches!(
                    k,
                    "repeat" | "no-repeat" | "repeat-x" | "repeat-y" | "space" | "round"
                ) {
                    if repeat.is_some() {
                        return Vec::new();
                    }
                    repeat = Some(CssValue::Keyword(kw));
                } else if matches!(k, "scroll" | "fixed" | "local") {
                    if attachment.is_some() {
                        return Vec::new();
                    }
                    attachment = Some(CssValue::Keyword(kw));
                } else if matches!(k, "left" | "right" | "top" | "bottom" | "center") {
                    position.push(CssValue::Keyword(kw));
                } else if k == "none" {
                    if image.is_some() {
                        return Vec::new();
                    }
                    image = Some(CssValue::Keyword(kw));
                } else {
                    return Vec::new();
                }
            }
            _ => return Vec::new(),
        }
    }
    if position.len() > 2 {
        return Vec::new();
    }
    // Every longhand is emitted; omitted ones reset to initial values.
    let color = color.unwrap_or(CssValue::Color(Color {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    })); // transparent
    let image = image.unwrap_or(CssValue::Keyword("none".into()));
    let repeat = repeat.unwrap_or(CssValue::Keyword("repeat".into()));
    let attachment = attachment.unwrap_or(CssValue::Keyword("scroll".into()));
    let position = if position.is_empty() {
        CssValue::List(vec![CssValue::Percentage(0.0), CssValue::Percentage(0.0)])
    } else if position.len() == 1 {
        position.pop().expect("one position")
    } else {
        CssValue::List(position)
    };
    vec![
        decl(PropertyName::intern("background-color"), color, important),
        decl(PropertyName::intern("background-image"), image, important),
        decl(PropertyName::intern("background-repeat"), repeat, important),
        decl(
            PropertyName::intern("background-attachment"),
            attachment,
            important,
        ),
        decl(
            PropertyName::intern("background-position"),
            position,
            important,
        ),
    ]
}

/// `list-style: <type> <position> <image>` in any order. All three
/// longhands are always emitted; omitted components reset to initial
/// values (`disc`, `outside`, `none`).
fn expand_list_style(value: CssValue, important: bool) -> Vec<Declaration> {
    fn is_type(kw: &str) -> bool {
        matches!(
            kw,
            "disc"
                | "circle"
                | "square"
                | "decimal"
                | "decimal-leading-zero"
                | "lower-roman"
                | "upper-roman"
                | "lower-greek"
                | "lower-latin"
                | "upper-latin"
                | "armenian"
                | "georgian"
                | "lower-alpha"
                | "upper-alpha"
                | "none"
        )
    }
    let mut style_type = None;
    let mut position = None;
    let mut image = None;
    for item in as_list(&value) {
        match item {
            // Gradients are valid `<image>`s for markers too
            // (CSS Images 3 §3.1).
            CssValue::Url(_) | CssValue::Gradient { .. } => {
                if image.is_some() {
                    return Vec::new();
                }
                image = Some(item);
            }
            CssValue::Keyword(kw) => {
                let k = kw.as_str();
                if is_type(k) {
                    if style_type.is_some() {
                        return Vec::new();
                    }
                    style_type = Some(CssValue::Keyword(kw));
                } else if matches!(k, "inside" | "outside") {
                    if position.is_some() {
                        return Vec::new();
                    }
                    position = Some(CssValue::Keyword(kw));
                } else if k == "none" {
                    if image.is_some() {
                        return Vec::new();
                    }
                    image = Some(CssValue::Keyword(kw));
                } else {
                    return Vec::new();
                }
            }
            _ => return Vec::new(),
        }
    }
    // Every longhand is emitted; omitted ones reset to initial values.
    vec![
        decl(
            PropertyName::intern("list-style-type"),
            style_type.unwrap_or(CssValue::Keyword("disc".into())),
            important,
        ),
        decl(
            PropertyName::intern("list-style-position"),
            position.unwrap_or(CssValue::Keyword("outside".into())),
            important,
        ),
        decl(
            PropertyName::intern("list-style-image"),
            image.unwrap_or(CssValue::Keyword("none".into())),
            important,
        ),
    ]
}

/// `flex: none | [ <flex-grow> <flex-shrink>? || <flex-basis> ]`
/// (CSS flexbox-1 §7.1.1). Emits all three longhands; omitted components
/// reset per the shorthand's own rules, not to initial values:
/// `flex: none` -> `0 0 auto`; a single number -> `<n> 1 0%`; a single
/// basis -> `1 1 <basis>`; otherwise an absent grow/shrink defaults to 1
/// and an absent basis to `0%`. Numbers (in any position) are flex
/// factors — a unitless zero is a factor, never a basis, per the "single
/// number" disambiguation; lengths/percentages and the `auto`/`content`
/// keywords are the basis. `calc()` components are rejected: their type
/// (number vs length) is not known at parse time, so they cannot be
/// placed unambiguously (`flex-grow: calc(..)` / `flex-basis: calc(..)`
/// still work as longhands). Unclassifiable items drop the whole
/// declaration; `var()` references never reach here (see
/// `expand_shorthand`).
fn expand_flex(value: CssValue, important: bool) -> Vec<Declaration> {
    let items = as_list(&value);
    // w78fix H4: a lone CSS-wide keyword expands per longhand, so
    // `flex: inherit` is not a silent no-op.
    if items.len() == 1 && is_css_wide_keyword(&items[0]) {
        let kw = items[0].clone();
        return vec![
            decl(PropertyName::intern("flex-grow"), kw.clone(), important),
            decl(PropertyName::intern("flex-shrink"), kw.clone(), important),
            decl(PropertyName::intern("flex-basis"), kw, important),
        ];
    }
    // `flex: none` -> `0 0 auto`.
    if items.len() == 1
        && let CssValue::Keyword(k) = &items[0]
        && k.eq_ignore_ascii_case("none")
    {
        return vec![
            decl(
                PropertyName::intern("flex-grow"),
                CssValue::Number(0.0),
                important,
            ),
            decl(
                PropertyName::intern("flex-shrink"),
                CssValue::Number(0.0),
                important,
            ),
            decl(
                PropertyName::intern("flex-basis"),
                CssValue::Keyword("auto".into()),
                important,
            ),
        ];
    }
    /// A `<number>`: a flex factor (`flex-grow` first, then `flex-shrink`).
    fn is_factor(v: &CssValue) -> bool {
        matches!(v, CssValue::Number(_))
    }
    /// A `<flex-basis>`: a width, or the `auto`/`content`/`fit-content`/
    /// `min-content`/`max-content`/`stretch` keywords — the same set
    /// the `flex-basis` longhand accepts (see `is_valid_value`).
    fn is_basis(v: &CssValue) -> bool {
        matches!(v, CssValue::Length(_, _) | CssValue::Percentage(_))
            || matches!(v, CssValue::Number(n) if *n == 0.0)
            || matches!(v, CssValue::Keyword(k) if {
                let k = k.to_ascii_lowercase();
                k == "auto"
                    || k == "content"
                    || k == "fit-content"
                    || k == "min-content"
                    || k == "max-content"
                    || k == "stretch"
            })
    }
    let mut grow: Option<CssValue> = None;
    let mut shrink: Option<CssValue> = None;
    let mut basis: Option<CssValue> = None;
    for item in &items {
        if is_factor(item) {
            // A unitless zero checks here first: per spec a lone number
            // is always a factor, so `flex: 0` is `0 1 0%`, never a basis.
            if grow.is_none() {
                grow = Some(item.clone());
            } else if shrink.is_none() {
                shrink = Some(item.clone());
            } else {
                return Vec::new(); // three factors: invalid
            }
        } else if is_basis(item) {
            if basis.is_none() {
                basis = Some(item.clone());
            } else {
                return Vec::new(); // two bases: invalid
            }
        } else {
            return Vec::new(); // unclassifiable component: invalid
        }
    }
    if items.is_empty() || (grow.is_none() && basis.is_none()) {
        return Vec::new();
    }
    vec![
        decl(
            PropertyName::intern("flex-grow"),
            grow.unwrap_or(CssValue::Number(1.0)),
            important,
        ),
        decl(
            PropertyName::intern("flex-shrink"),
            shrink.unwrap_or(CssValue::Number(1.0)),
            important,
        ),
        decl(
            PropertyName::intern("flex-basis"),
            basis.unwrap_or(CssValue::Percentage(0.0)),
            important,
        ),
    ]
}

/// One component of a border-radius longhand: a `<length-percentage>`,
/// with unitless zero accepted as a length.
fn is_radius_component(value: &CssValue) -> bool {
    // `Math` is admitted: it evaluates in the cascade after expansion
    // (matching the deferred expansion in cascade.rs).
    matches!(
        value,
        CssValue::Length(_, _) | CssValue::Percentage(_) | CssValue::Math(_)
    ) || matches!(value, CssValue::Number(n) if *n == 0.0)
}

/// Parse value tokens that may contain a top-level `/`, which the generic
/// value parser cannot represent (`Token::Delim('/')` has no `parse_item`
/// arm, so any `/` fails the whole declaration). Split once on a top-level
/// `/` (one nested inside `()`/`[]`/functions is left alone) and store the
/// `/`-separated form as `List([List(before), List(after)])`; without `/`
/// this is just `parse_value`. A comma group never nests this way outside
/// `font-family` lists, so the nesting marks `/` for the cascade-side
/// expansion (see `parse_border_radius` and the deferred `font` path).
/// Two top-level slashes, or an empty side, drop the declaration.
///
/// `pub(crate)` because the cascade reuses it to parse a custom
/// property's stored token stream at `var()`-substitution time (see
/// `Styler::resolve`): substitution sees the same slash handling as the
/// shorthand parsers.
/// Font-path variant of [`split_top_slash`]: top-level commas on either
/// side of the `/` stay visible as nested groups — empty groups kept as
/// empty lists — so the deferred `font` expansion can reject leading and
/// doubled commas exactly like the eager token parser
/// (`parse_font_shorthand`). The plain [`split_top_slash`] flattens
/// commas away (`12px/1.5, serif` parses like `12px/1.5 serif`), which
/// the deferred path must not accept. Only used for `font`
/// substitution (see `preserve_case` in
/// [`crate::Styler::resolve_inner`]); every other shorthand keeps the
/// historical shape.
pub(crate) fn split_top_slash_font(toks: &[Token]) -> Option<(Vec<CssValue>, Vec<CssValue>)> {
    let mut depth = 0u32;
    let mut slash = None;
    for (i, t) in toks.iter().enumerate() {
        match t {
            Token::LParen | Token::Function(_) | Token::LBracket => depth += 1,
            Token::RParen | Token::RBracket => depth = depth.saturating_sub(1),
            Token::Delim('/') if depth == 0 => {
                if slash.is_some() {
                    return None; // two slashes: invalid
                }
                slash = Some(i);
            }
            _ => {}
        }
    }
    let i = slash?;
    let before = comma_groups_keep_empty(&toks[..i])?;
    let after = comma_groups_keep_empty(&toks[i + 1..])?;
    // A top-level comma on the `/`-side must follow a family word:
    // `12px/1.5, serif` is a leading comma in the family list
    // (invalid, like the eager parser), while `12px/1.5 Arial, serif`
    // is fine. Commas arriving later through a nested `var()`
    // substitution were not top-level in these tokens and are
    // unaffected.
    if after.iter().any(|v| matches!(v, CssValue::List(_))) && as_list(&after[0]).len() < 2 {
        return None;
    }
    Some((before, after))
}

/// Whether `toks` hold a comma outside any `()`/`[]`/function nesting —
/// i.e. a top-level comma separating value groups. Used by
/// [`Parser::parse_declaration`] for `grid-template-columns`/
/// `grid-template-rows`, whose `<track-list>` grammar has no commas:
/// the generic value parser flattens single-item comma groups, so the
/// comma must be detected at the token level before parsing.
fn has_top_level_comma(toks: &[Token]) -> bool {
    let mut depth: u32 = 0;
    for t in toks {
        match t {
            Token::LParen | Token::Function(_) | Token::LBracket => depth += 1,
            Token::RParen | Token::RBracket => depth = depth.saturating_sub(1),
            Token::Comma if depth == 0 => return true,
            _ => {}
        }
    }
    false
}

/// Split top-level commas into one [`CssValue`] per group, preserving
/// identifier case. Empty groups (a leading, trailing, or doubled comma)
/// stay visible as empty lists instead of failing the parse, so the
/// caller can reject them with the same rule the eager parser applies.
/// Every group nests as a list — unlike [`parse_value`], single-item
/// groups do not flatten — so the caller can tell a comma-separated
/// side from a plain one. With no comma at all the result is the flat
/// item list, exactly like [`split_top_slash`].
fn comma_groups_keep_empty(toks: &[Token]) -> Option<Vec<CssValue>> {
    let mut depth = 0u32;
    let mut cuts = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        match t {
            Token::LParen | Token::Function(_) | Token::LBracket => depth += 1,
            Token::RParen | Token::RBracket => depth = depth.saturating_sub(1),
            Token::Comma if depth == 0 => cuts.push(i),
            _ => {}
        }
    }
    if cuts.is_empty() {
        return Some(as_list(&parse_value_preserving_case(toks).ok()?));
    }
    let mut out = Vec::with_capacity(cuts.len() + 1);
    let mut start = 0;
    for c in cuts.iter().chain(std::iter::once(&toks.len())) {
        let group = &toks[start..*c];
        if group
            .iter()
            .all(|t| matches!(t, Token::Whitespace | Token::Eof))
        {
            // Leading/doubled/trailing comma: kept visible for the
            // caller to reject.
            out.push(CssValue::List(Vec::new()));
        } else {
            // No top-level comma inside (cut above); `as_list` flattens
            // the group's items exactly like `split_top_slash` does.
            out.push(CssValue::List(as_list(
                &parse_value_preserving_case(group).ok()?,
            )));
        }
        start = *c + 1;
    }
    Some(out)
}

/// Split `toks` on the single top-level `/` (one nested inside `()`/`[]`
/// /functions is left alone), returning the before/after item lists.
/// Returns `None` when there is no top-level `/`, when there are two,
/// or when either side fails to parse. [`parse_slash_aware_value`] is
/// this in `CssValue` form; the cascade uses the split form to carry
/// the slash marker through `var()` substitution (see
/// [`crate::Styler::resolve`]).
pub(crate) fn split_top_slash(
    toks: &[Token],
    preserve_case: bool,
) -> Option<(Vec<CssValue>, Vec<CssValue>)> {
    let mut depth = 0u32;
    let mut slash = None;
    for (i, t) in toks.iter().enumerate() {
        match t {
            Token::LParen | Token::Function(_) | Token::LBracket => depth += 1,
            Token::RParen | Token::RBracket => depth = depth.saturating_sub(1),
            Token::Delim('/') if depth == 0 => {
                if slash.is_some() {
                    return None; // two slashes: invalid
                }
                slash = Some(i);
            }
            _ => {}
        }
    }
    let i = slash?;
    let parse = if preserve_case {
        parse_value_preserving_case
    } else {
        parse_value
    };
    let before = parse(&toks[..i]).ok()?;
    let after = parse(&toks[i + 1..]).ok()?;
    Some((as_list(&before), as_list(&after)))
}

pub(crate) fn parse_slash_aware_value(toks: &[Token]) -> Option<CssValue> {
    match split_top_slash(toks, false) {
        Some((before, after)) => Some(CssValue::List(vec![
            CssValue::List(before),
            CssValue::List(after),
        ])),
        // No top-level `/` (or an unparseable split, which can only come
        // from a `/` the generic parser rejects): parse as a plain value.
        None => parse_value(toks).ok(),
    }
}

/// Parse a `var()`-containing `font` shorthand into its deferred form.
/// Like [`parse_slash_aware_value`], but a top-level comma also splits
/// into nested groups: the generic value parser would flatten it,
/// losing the family boundaries the cascade-side expansion needs
/// (`font: var(--f), serif` must keep `serif` as its own family rather
/// than merging it into the substituted words). The `/`-separated form
/// keeps its existing two-group marker shape; the cascade-side
/// expansion tries the slash reading first and falls back to comma
/// groups (see `expand_font_value` in cascade.rs). The `/` split keeps
/// commas visible (see [`split_top_slash_font`]) so a leading or doubled
/// comma is rejected exactly like the eager parser rejects it.
fn parse_deferred_font_value(toks: &[Token]) -> Option<CssValue> {
    // Case-preserving: the deferred `font` expansion must see family
    // idents with their authored case, exactly as the eager token
    // parser (`parse_font_shorthand`) does.
    match split_top_slash_font(toks) {
        Some((before, after)) => Some(CssValue::List(vec![
            CssValue::List(before),
            CssValue::List(after),
        ])),
        // No top-level `/`: a top-level comma keeps its groups distinct
        // through substitution; otherwise parse as a plain value.
        None => match split_top_comma(toks, true) {
            Some(groups) => Some(CssValue::List(
                groups.into_iter().map(CssValue::List).collect(),
            )),
            None => parse_value_preserving_case(toks).ok(),
        },
    }
}

/// Split `toks` on top-level commas (one nested inside `()`/`[]`
/// /functions is left alone), returning each group's item list.
/// Returns `None` when there is no top-level comma or when any group
/// fails to parse. `pub(crate)` because the cascade reuses it to keep a
/// custom property's comma-separated value grouped through `var()`
/// substitution (see [`crate::Styler::resolve`]): each group splices as
/// one nested list so the deferred `font` expansion can tell `Arial,
/// sans-serif` (two families) from `Times New Roman` (one).
///
/// With `preserve_case`, identifier keywords keep their authored case
/// (see [`parse_value_preserving_case`]); the deferred `font` path uses
/// this so family names agree with the eager token parser.
pub(crate) fn split_top_comma(toks: &[Token], preserve_case: bool) -> Option<Vec<Vec<CssValue>>> {
    let mut depth = 0u32;
    let mut cuts = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        match t {
            Token::LParen | Token::Function(_) | Token::LBracket => depth += 1,
            Token::RParen | Token::RBracket => depth = depth.saturating_sub(1),
            Token::Comma if depth == 0 => cuts.push(i),
            _ => {}
        }
    }
    if cuts.is_empty() {
        return None;
    }
    let parse = if preserve_case {
        parse_value_preserving_case
    } else {
        parse_value
    };
    let mut groups = Vec::with_capacity(cuts.len() + 1);
    let mut start = 0;
    for c in cuts {
        groups.push(as_list(&parse(&toks[start..c]).ok()?));
        start = c + 1;
    }
    groups.push(as_list(&parse(&toks[start..]).ok()?));
    Some(groups)
}

/// Parse a `border-radius` declaration from tokens. The `/` separating
/// horizontal and vertical radii cannot survive the generic value parser
/// (like `font`'s `/` between size and line-height), so tokens are split
/// once on a top-level `/` first. A value containing any `var()` is NOT
/// expanded here: it is stored as-is under the shorthand name and the
/// cascade expands it after substitution (see `expand_shorthand` for the
/// contract). The `/`-separated form is stored as
/// `List([List(h..), List(v..)])`; a comma group never nests this way, so
/// nesting unambiguously marks `/` for the cascade-side expansion.
fn parse_border_radius(toks: &[Token], important: bool) -> Option<Vec<Declaration>> {
    let value = parse_slash_aware_value(toks)?;
    if value_contains_var(&value) {
        return Some(vec![decl(
            PropertyName::intern("border-radius"),
            value,
            important,
        )]);
    }
    Some(expand_border_radius(value, important))
}

/// Expand `border-radius: [<length-percentage>]{1,4} [ /
/// <length-percentage>{1,4} ]?` into per-corner `border-<corner>-radius`
/// longhands (`top-left`, `top-right`, `bottom-right`, `bottom-left`).
/// Without `/` each corner gets a single value; with `/` a `[horizontal
/// vertical]` pair list. Every longhand is emitted; invalid (bad
/// components, wrong count) drops the declaration.
fn expand_border_radius(value: CssValue, important: bool) -> Vec<Declaration> {
    let (h, v) = match value {
        CssValue::List(mut groups)
            if groups.len() == 2 && groups.iter().all(|g| matches!(g, CssValue::List(_))) =>
        {
            let v = groups.pop().expect("two groups");
            let h = groups.pop().expect("two groups");
            (as_list(&h), Some(as_list(&v)))
        }
        other => (as_list(&other), None),
    };
    if !(1..=4).contains(&h.len()) || !h.iter().all(is_radius_component) {
        return Vec::new();
    }
    let v = match v {
        Some(vv) if (1..=4).contains(&vv.len()) && vv.iter().all(is_radius_component) => Some(vv),
        Some(_) => return Vec::new(),
        None => None,
    };
    // 1-4 values -> [top-left, top-right, bottom-right, bottom-left]:
    // [a], [a b], [a b c], [a b c d] -> [a a a a], [a b a b], [a b c b], [a b c d].
    let quad = |items: &[CssValue], i: usize| match items.len() {
        1 => items[0].clone(),
        2 => items[i % 2].clone(),
        3 => items[if i == 2 { 2 } else { i % 2 }].clone(),
        _ => items[i].clone(),
    };
    ["top-left", "top-right", "bottom-right", "bottom-left"]
        .into_iter()
        .enumerate()
        .map(|(i, corner)| {
            let hv = match &v {
                Some(vv) => CssValue::List(vec![quad(&h, i), quad(vv, i)]),
                None => quad(&h, i),
            };
            decl(
                PropertyName::from(format!("border-{corner}-radius")),
                hv,
                important,
            )
        })
        .collect()
}

/// Fold an `aspect-ratio` value to its computed form
/// ([`CssValue::AspectRatio`]): `auto` stays the `auto` keyword (no
/// preferred ratio); a bare `<number>` becomes `n / 1`; a `w / h` pair
/// becomes the quotient. Both the eager parse ([`parse_aspect_ratio`])
/// and `var()` substitution produce the `/`-separated marker shape
/// `List([List(before), List(after)])` (which `normalize_comma_groups`
/// in cascade.rs preserves), so one fold serves both paths; `auto
/// <ratio>` sets the `auto`-preferred flag. Anything else is left alone
/// so the grammar check drops it (invalid at computed-value time).
pub(crate) fn fold_aspect_ratio(value: CssValue) -> CssValue {
    // Already folded (eager path), or `auto`: nothing to do.
    if matches!(value, CssValue::AspectRatio { .. })
        || matches!(&value, CssValue::Keyword(k) if k == "auto")
    {
        return value;
    }
    /// One side of the `/`: a single `<number>`, optionally paired with
    /// the `auto` keyword (`auto <ratio>` / `<ratio> auto`).
    fn side(items: &[CssValue]) -> Option<(f32, bool)> {
        match items {
            [CssValue::Number(n)] => Some((*n, false)),
            [CssValue::Keyword(k), CssValue::Number(n)] if k == "auto" => Some((*n, true)),
            [CssValue::Number(n), CssValue::Keyword(k)] if k == "auto" => Some((*n, true)),
            _ => None,
        }
    }
    let pair = match &value {
        CssValue::Number(n) => Some((*n, 1.0, false)),
        CssValue::List(groups)
            if groups.len() == 2 && groups.iter().all(|g| matches!(g, CssValue::List(_))) =>
        {
            let before = as_list(&groups[0]);
            let after = as_list(&groups[1]);
            let Some((w, w_auto)) = side(&before) else {
                return value;
            };
            let Some((h, h_auto)) = side(&after) else {
                return value;
            };
            // css-sizing-4 `auto || <ratio>`: `auto` may appear at most
            // once (`auto 16/9 auto` is invalid).
            if w_auto && h_auto {
                return value;
            }
            Some((w, h, w_auto || h_auto))
        }
        _ => None,
    };
    let Some((w, h, auto_preferred)) = pair else {
        return value;
    };
    // css-sizing-4 `<ratio> = <number [0,∞]> [ / <number (0,∞)> ]?`, as a
    // positive finite quotient; anything else is invalid at
    // computed-value time (left alone so the grammar check drops it).
    if w < 0.0 || h <= 0.0 || !w.is_finite() || !h.is_finite() {
        return value;
    }
    let ratio = w / h;
    if !ratio.is_finite() || ratio <= 0.0 {
        return value;
    }
    CssValue::AspectRatio {
        ratio,
        auto_preferred,
    }
}

/// Parse an `aspect-ratio` declaration (`auto || <ratio>`, css-sizing-4
/// §3.1) from its token slice. The `/` in `16 / 9` cannot survive the
/// generic value parser, so split on a top-level `/` first (like `font`
/// and `border-radius`). Concrete values validate and fold eagerly to
/// [`CssValue::AspectRatio`]; a value containing `var()` keeps the
/// `/`-separated marker shape for the cascade to fold after
/// substitution (see [`fold_aspect_ratio`]). CSS-wide keywords pass
/// through for the cascade to apply after substitution, and a
/// `calc()`-shaped `Math` value is kept so the cascade can evaluate it
/// before folding. `None` means invalid: the
/// declaration is dropped.
fn parse_aspect_ratio(toks: &[Token]) -> Option<CssValue> {
    let value = parse_slash_aware_value(toks)?;
    // CSS-wide keywords pass through untouched: the cascade applies
    // them after `var()` substitution. Dropping them here would be
    // observable since `aspect-ratio` is not inherited (a child with
    // `inherit` under a `16/9` parent would wrongly compute `auto`).
    if let CssValue::Keyword(k) = &value
        && matches!(
            k.as_str(),
            "inherit" | "initial" | "unset" | "revert" | "revert-layer"
        )
    {
        return Some(value);
    }
    if toks
        .iter()
        .any(|t| matches!(t, Token::Function(name) if name.eq_ignore_ascii_case("var")))
    {
        // Deferred: the cascade folds after substitution; only the
        // static shape was checked above (empty sides or two `/`s
        // already failed the split).
        return Some(value);
    }
    match fold_aspect_ratio(value) {
        v @ CssValue::AspectRatio { .. } => Some(v),
        CssValue::Keyword(k) if k == "auto" => Some(CssValue::Keyword(k)),
        // `calc()` is a valid `<number>` (css-sizing-4 §3.1): keep the
        // `Math` value so the cascade can evaluate it before folding
        // (see `compute_prop`).
        v @ CssValue::Math(_) => Some(v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{BasicShape, ShapeRadius};

    fn sheet(input: &str) -> Stylesheet {
        parse_stylesheet(input)
    }

    fn decls(input: &str) -> Vec<Declaration> {
        sheet(input).rules()[0].declarations.clone()
    }

    fn get<'a>(decls: &'a [Declaration], prop: &str) -> &'a CssValue {
        &decls.iter().find(|d| d.property == prop).unwrap().value
    }

    #[test]
    fn basic_rule() {
        let s = sheet("div { color: red; margin: 0; }");
        assert_eq!(s.rules().len(), 1);
        let r = &s.rules()[0];
        assert_eq!(r.order, 0);
        assert_eq!(r.media, None);
        assert_eq!(r.declarations.len(), 1 + 4); // color + margin x4
        assert_eq!(r.declarations[0].property, "color");
        assert!(!r.declarations[0].important);
    }

    #[test]
    fn multiple_rules_source_order() {
        let s = sheet("a { color: red; } b { color: blue; }");
        assert_eq!(s.rules().len(), 2);
        assert_eq!(s.rules()[0].order, 0);
        assert_eq!(s.rules()[1].order, 1);
    }

    #[test]
    fn multiple_selectors() {
        let s = sheet("div, .a, #b { color: red; }");
        assert_eq!(s.rules()[0].selectors.len(), 3);
    }

    #[test]
    fn empty_stylesheet() {
        assert_eq!(sheet("").rules().len(), 0);
        assert_eq!(sheet("   /* nothing */ ").rules().len(), 0);
        assert_eq!(sheet(";").rules().len(), 0);
    }

    #[test]
    fn property_names_lowercased() {
        let d = decls("p { COLOR: red; }");
        assert_eq!(d[0].property, "color");
    }

    #[test]
    fn property_names_intern_to_atoms() {
        use crate::propatom::PropertyAtom;
        // Well-known properties parse to pure atoms (no heap sidecar).
        let d = decls("p { COLOR: red; margin-top: 1px; --MyVar: 2; }");
        assert_eq!(d[0].property.atom(), PropertyAtom::COLOR);
        assert_eq!(d[0].property.as_str(), "color");
        assert_eq!(d[1].property.atom(), PropertyAtom::MARGIN_TOP);
        // Custom properties keep their case on the heap sidecar.
        assert_eq!(d[2].property.atom(), PropertyAtom::UNKNOWN);
        assert_eq!(d[2].property.as_str(), "--MyVar");
        assert!(d[2].property.is_custom_property());
        // A deferred shorthand keeps its shorthand atom.
        let d = decls("p { border: var(--w) solid red; }");
        assert_eq!(d[0].property.atom(), PropertyAtom::BORDER);
        assert_eq!(d[0].property.as_str(), "border");
    }

    #[test]
    fn declaration_error_recovery() {
        // Missing colon: declaration dropped, rest of block survives.
        let d = decls("p { color red; margin: 1px; }");
        assert_eq!(d.len(), 4);
        assert_eq!(d[0].property, "margin-top");
        // Bad value: dropped.
        let d = decls("p { color: 10xyz; width: 5px; }");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].property, "width");
        // Empty value: dropped.
        let d = decls("p { color:; width: 5px; }");
        assert_eq!(d.len(), 1);
        // Garbage declaration skipped to next semicolon.
        let d = decls("p { ??? ; width: 5px; }");
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn unclosed_block_recovers() {
        let s = sheet("div { color: red; ");
        assert_eq!(s.rules().len(), 1);
        assert_eq!(s.rules()[0].declarations[0].property, "color");
    }

    #[test]
    fn bad_rule_dropped_rest_survives() {
        let s = sheet("div { color: red; } .broken color: blue; } p { color: green; }");
        assert_eq!(s.rules().len(), 2);
    }

    #[test]
    fn bad_selector_drops_rule() {
        let s = sheet("div:bogus { color: red; } p { color: green; }");
        assert_eq!(s.rules().len(), 1);
        assert_eq!(s.rules()[0].declarations[0].property, "color");
    }

    #[test]
    fn important_detection() {
        let d = decls("p { color: red !important; }");
        assert!(d[0].important);
        let d = decls("p { color: red ! important; }");
        assert!(d[0].important);
        let d = decls("p { color: red !IMPORTANT; }");
        assert!(d[0].important);
        let d = decls("p { color: red; }");
        assert!(!d[0].important);
        // `!important` must be at the end: trailing garbage invalidates
        // the whole declaration, so it is dropped.
        let d = decls("p { color: red !important garbage; }");
        assert!(d.is_empty());
        // important propagates through shorthand expansion
        let d = decls("p { margin: 1px !important; }");
        assert_eq!(d.len(), 4);
        assert!(d.iter().all(|x| x.important));
    }

    #[test]
    fn margin_padding_expansion() {
        let d = decls("p { margin: 1px; }");
        assert_eq!(d.len(), 4);
        for side in ["top", "right", "bottom", "left"] {
            assert_eq!(
                get(&d, &format!("margin-{side}")),
                &CssValue::Length(1.0, LengthUnit::Px)
            );
        }
        let d = decls("p { padding: 1px 2px; }");
        assert_eq!(
            get(&d, "padding-top"),
            &CssValue::Length(1.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "padding-right"),
            &CssValue::Length(2.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "padding-bottom"),
            &CssValue::Length(1.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "padding-left"),
            &CssValue::Length(2.0, LengthUnit::Px)
        );
        let d = decls("p { margin: 1px 2px 3px; }");
        assert_eq!(
            get(&d, "margin-top"),
            &CssValue::Length(1.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "margin-right"),
            &CssValue::Length(2.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "margin-bottom"),
            &CssValue::Length(3.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "margin-left"),
            &CssValue::Length(2.0, LengthUnit::Px)
        );
        let d = decls("p { margin: 1px 2px 3px 4px; }");
        assert_eq!(
            get(&d, "margin-left"),
            &CssValue::Length(4.0, LengthUnit::Px)
        );
        // Too many values: dropped.
        let d = decls("p { margin: 1px 2px 3px 4px 5px; }");
        assert!(d.is_empty());
    }

    #[test]
    fn scroll_margin_padding_expansion() {
        // `scroll-margin` / `scroll-padding` expand through the same
        // box-side machinery as `margin` / `padding`.
        let d = decls("p { scroll-margin: 1px; }");
        assert_eq!(d.len(), 4);
        for side in ["top", "right", "bottom", "left"] {
            assert_eq!(
                get(&d, &format!("scroll-margin-{side}")),
                &CssValue::Length(1.0, LengthUnit::Px)
            );
        }
        let d = decls("p { scroll-padding: 1px 2px 3px; }");
        assert_eq!(d.len(), 4);
        assert_eq!(
            get(&d, "scroll-padding-top"),
            &CssValue::Length(1.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "scroll-padding-right"),
            &CssValue::Length(2.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "scroll-padding-bottom"),
            &CssValue::Length(3.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "scroll-padding-left"),
            &CssValue::Length(2.0, LengthUnit::Px)
        );
        // `auto` is not a valid scroll-margin (CSS Scroll Snap 1:
        // `<length>`, no `auto`), so the declaration is dropped at
        // parse time.
        let d = decls("p { scroll-margin: auto; }");
        assert!(d.is_empty());
        // Too many values: dropped.
        let d = decls("p { scroll-padding: 1px 2px 3px 4px 5px; }");
        assert!(d.is_empty());
        // A `var()` reference defers expansion to the cascade: the
        // declaration stays under the shorthand name.
        let d = decls("p { scroll-margin: var(--m); }");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].property, "scroll-margin");
        let d = decls("p { overscroll-behavior: var(--o); }");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].property, "overscroll-behavior");
    }

    #[test]
    fn inset_shorthand_expansion() {
        // `inset` expands to the four physical longhands (`top` /
        // `right` / `bottom` / `left`), not `inset-top` etc.
        let d = decls("p { inset: 8px; }");
        assert_eq!(d.len(), 4);
        for side in ["top", "right", "bottom", "left"] {
            assert_eq!(get(&d, side), &CssValue::Length(8.0, LengthUnit::Px));
        }
        let d = decls("p { inset: 1px 2px 3px; }");
        assert_eq!(get(&d, "top"), &CssValue::Length(1.0, LengthUnit::Px));
        assert_eq!(get(&d, "right"), &CssValue::Length(2.0, LengthUnit::Px));
        assert_eq!(get(&d, "bottom"), &CssValue::Length(3.0, LengthUnit::Px));
        assert_eq!(get(&d, "left"), &CssValue::Length(2.0, LengthUnit::Px));
        // `auto` and percentages are valid inset components.
        let d = decls("p { inset: auto 10%; }");
        assert_eq!(get(&d, "top"), &CssValue::Keyword("auto".into()));
        assert_eq!(get(&d, "right"), &CssValue::Percentage(10.0));
        // An invalid component drops the whole declaration.
        let d = decls("p { inset: 1px solid; }");
        assert!(d.is_empty());
        let d = decls("p { inset: 1px 2px 3px 4px 5px; }");
        assert!(d.is_empty());
    }

    #[test]
    fn logical_longhands_parse_as_longhands() {
        // Flow-relative longhands are not shorthands: each parses to a
        // single longhand declaration, validated later by the cascade.
        let d = decls(
            "p { margin-inline-start: 10px; padding-block-end: 5%; inset-inline-end: 2em; \
             inline-size: 100px; block-size: auto; writing-mode: vertical-rl; \
             border-inline-start-width: 3px; border-block-end-style: solid; }",
        );
        assert_eq!(
            get(&d, "margin-inline-start"),
            &CssValue::Length(10.0, LengthUnit::Px)
        );
        assert_eq!(get(&d, "padding-block-end"), &CssValue::Percentage(5.0));
        assert_eq!(
            get(&d, "inset-inline-end"),
            &CssValue::Length(2.0, LengthUnit::Em)
        );
        assert_eq!(
            get(&d, "inline-size"),
            &CssValue::Length(100.0, LengthUnit::Px)
        );
        assert_eq!(get(&d, "block-size"), &CssValue::Keyword("auto".into()));
        assert_eq!(
            get(&d, "writing-mode"),
            &CssValue::Keyword("vertical-rl".into())
        );
        assert_eq!(
            get(&d, "border-inline-start-width"),
            &CssValue::Length(3.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "border-block-end-style"),
            &CssValue::Keyword("solid".into())
        );
    }

    #[test]
    fn expand_logical_sides_no_dash_border_name_drops() {
        // Defense-in-depth: `expand_logical_sides` takes a plain `&str`,
        // so a future dispatch arm or caller passing a `border-`-prefixed
        // name without a second dash (e.g. `border-inline`) must drop the
        // declaration instead of panicking.
        let value = CssValue::Length(1.0, LengthUnit::Px);
        let decls = expand_logical_sides("border-inline", value, false);
        assert!(decls.is_empty());
    }

    #[test]
    fn overscroll_behavior_expansion() {
        // One value sets both axes.
        let d = decls("p { overscroll-behavior: contain; }");
        assert_eq!(d.len(), 2);
        assert_eq!(
            get(&d, "overscroll-behavior-x"),
            &CssValue::Keyword("contain".into())
        );
        assert_eq!(
            get(&d, "overscroll-behavior-y"),
            &CssValue::Keyword("contain".into())
        );
        // Two values: x then y.
        let d = decls("p { overscroll-behavior: auto none; }");
        assert_eq!(
            get(&d, "overscroll-behavior-x"),
            &CssValue::Keyword("auto".into())
        );
        assert_eq!(
            get(&d, "overscroll-behavior-y"),
            &CssValue::Keyword("none".into())
        );
        // Too many values: dropped.
        let d = decls("p { overscroll-behavior: auto contain none; }");
        assert!(d.is_empty());
    }

    #[test]
    fn place_self_expansion() {
        // CSS Box Alignment 3 §6.1: one value sets both longhands.
        let d = decls("p { place-self: center; }");
        assert_eq!(d.len(), 2);
        assert_eq!(get(&d, "align-self"), &CssValue::Keyword("center".into()));
        assert_eq!(get(&d, "justify-self"), &CssValue::Keyword("center".into()));
        // Two values: align-self then justify-self.
        let d = decls("p { place-self: start end; }");
        assert_eq!(d.len(), 2);
        assert_eq!(get(&d, "align-self"), &CssValue::Keyword("start".into()));
        assert_eq!(get(&d, "justify-self"), &CssValue::Keyword("end".into()));
        // `auto` and `baseline` are in the self-alignment grammar.
        let d = decls("p { place-self: auto baseline; }");
        assert_eq!(get(&d, "align-self"), &CssValue::Keyword("auto".into()));
        assert_eq!(
            get(&d, "justify-self"),
            &CssValue::Keyword("baseline".into())
        );
        // A component outside the longhand grammar drops the whole
        // declaration (no partial application).
        let d = decls("p { place-self: center 10px; }");
        assert!(d.is_empty());
        // Too many values: dropped.
        let d = decls("p { place-self: center start end; }");
        assert!(d.is_empty());
        // A lone CSS-wide keyword still expands; the cascade
        // intercepts it per longhand.
        let d = decls("p { place-self: inherit; }");
        assert_eq!(d.len(), 2);
        assert_eq!(get(&d, "align-self"), &CssValue::Keyword("inherit".into()));
        assert_eq!(
            get(&d, "justify-self"),
            &CssValue::Keyword("inherit".into())
        );
    }

    #[test]
    fn place_self_safe_unsafe_grammar() {
        // CSS Box Alignment 3 §6: `place-self: unsafe center` is one
        // <align-self> production — the overflow keyword binds to the
        // following keyword, not to a second longhand.
        let d = decls("p { place-self: unsafe center; }");
        assert_eq!(d.len(), 2);
        let pair = CssValue::List(vec![
            CssValue::Keyword("unsafe".into()),
            CssValue::Keyword("center".into()),
        ]);
        assert_eq!(get(&d, "align-self"), &pair);
        assert_eq!(get(&d, "justify-self"), &pair);
        // Direct longhand form validates too.
        let d = decls("p { align-self: safe end; }");
        assert_eq!(d.len(), 1);
        assert_eq!(
            get(&d, "align-self"),
            &CssValue::List(vec![
                CssValue::Keyword("safe".into()),
                CssValue::Keyword("end".into()),
            ])
        );
        // Two overflow-prefixed longhands: 4 tokens split 2+2.
        let d = decls("p { place-self: safe start unsafe end; }");
        assert_eq!(d.len(), 2);
        assert_eq!(
            get(&d, "align-self"),
            &CssValue::List(vec![
                CssValue::Keyword("safe".into()),
                CssValue::Keyword("start".into()),
            ])
        );
        assert_eq!(
            get(&d, "justify-self"),
            &CssValue::List(vec![
                CssValue::Keyword("unsafe".into()),
                CssValue::Keyword("end".into()),
            ])
        );
        // Mixed: plain align-self, overflow-prefixed justify-self.
        let d = decls("p { place-self: center unsafe start; }");
        assert_eq!(d.len(), 2);
        assert_eq!(get(&d, "align-self"), &CssValue::Keyword("center".into()));
        assert_eq!(
            get(&d, "justify-self"),
            &CssValue::List(vec![
                CssValue::Keyword("unsafe".into()),
                CssValue::Keyword("start".into()),
            ])
        );
        // A dangling overflow keyword is invalid: whole declaration drops.
        let d = decls("p { place-self: unsafe; }");
        assert!(d.is_empty());
    }

    #[test]
    fn self_align_safe_unsafe_validity() {
        use crate::cascade::is_valid_for_property;
        let kw = |s: &str| CssValue::Keyword(s.into());
        let pair = |a: &str, b: &str| CssValue::List(vec![kw(a), kw(b)]);
        // Overflow-prefixed pairs validate on both longhands.
        assert!(is_valid_for_property("align-self", &pair("safe", "end")));
        assert!(is_valid_for_property(
            "align-self",
            &pair("unsafe", "center")
        ));
        assert!(is_valid_for_property(
            "justify-self",
            &pair("safe", "start")
        ));
        assert!(is_valid_for_property(
            "justify-self",
            &pair("unsafe", "self-end")
        ));
        // Lone keywords still validate.
        assert!(is_valid_for_property("align-self", &kw("center")));
        assert!(is_valid_for_property("justify-self", &kw("baseline")));
        // A dangling overflow keyword is invalid ...
        assert!(!is_valid_for_property("align-self", &kw("safe")));
        assert!(!is_valid_for_property("justify-self", &kw("unsafe")));
        // ... as is `baseline` after an overflow keyword (grammar) ...
        assert!(!is_valid_for_property(
            "align-self",
            &pair("safe", "baseline")
        ));
        assert!(!is_valid_for_property(
            "align-self",
            &pair("safe", "stretch")
        ));
        assert!(!is_valid_for_property("align-self", &pair("safe", "auto")));
        // ... a non-position second keyword ...
        assert!(!is_valid_for_property(
            "align-self",
            &pair("unsafe", "left")
        ));
        // ... or a three-item list.
        assert!(!is_valid_for_property(
            "align-self",
            &CssValue::List(vec![kw("safe"), kw("unsafe"), kw("center")])
        ));
        // `place-self` validation follows the longhands.
        assert!(is_valid_for_property(
            "place-self",
            &pair("unsafe", "center")
        ));
        assert!(!is_valid_for_property("place-self", &kw("unsafe")));
        assert!(!is_valid_for_property(
            "place-self",
            &pair("safe", "baseline")
        ));
    }

    #[test]
    fn border_expansion() {
        // `border` expands to per-side longhands for all three components.
        let d = decls("p { border: 1px solid red; }");
        assert_eq!(d.len(), 12);
        for side in ["top", "right", "bottom", "left"] {
            assert_eq!(
                get(&d, &format!("border-{side}-width")),
                &CssValue::Length(1.0, LengthUnit::Px)
            );
            assert_eq!(
                get(&d, &format!("border-{side}-style")),
                &CssValue::Keyword("solid".into())
            );
            assert_eq!(
                get(&d, &format!("border-{side}-color")),
                &CssValue::Color(Color {
                    r: 255,
                    g: 0,
                    b: 0,
                    a: 255
                })
            );
        }
        // Any order, partial: omitted components reset to initial values
        // (width medium, color currentcolor), so all 12 longhands appear.
        let d = decls("p { border: dashed; }");
        assert_eq!(d.len(), 12);
        for side in ["top", "right", "bottom", "left"] {
            assert_eq!(
                get(&d, &format!("border-{side}-width")),
                &CssValue::Keyword("medium".into())
            );
            assert_eq!(
                get(&d, &format!("border-{side}-style")),
                &CssValue::Keyword("dashed".into())
            );
            assert_eq!(
                get(&d, &format!("border-{side}-color")),
                &CssValue::Keyword("currentcolor".into())
            );
        }
        let d = decls("p { border: #fff thick double; }");
        assert_eq!(d.len(), 12);
        assert_eq!(
            get(&d, "border-top-width"),
            &CssValue::Keyword("thick".into())
        );
        assert_eq!(
            get(&d, "border-left-color"),
            &CssValue::Color(Color {
                r: 255,
                g: 255,
                b: 255,
                a: 255
            })
        );
        // Single width component: style/color reset to initial values.
        let d = decls("p { border: 2px; }");
        assert_eq!(d.len(), 12);
        assert_eq!(
            get(&d, "border-bottom-width"),
            &CssValue::Length(2.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "border-bottom-style"),
            &CssValue::Keyword("none".into())
        );
        assert_eq!(
            get(&d, "border-bottom-color"),
            &CssValue::Keyword("currentcolor".into())
        );
        // border-width expands to sides.
        let d = decls("p { border-width: 1px 2px; }");
        assert_eq!(
            get(&d, "border-top-width"),
            &CssValue::Length(1.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "border-right-width"),
            &CssValue::Length(2.0, LengthUnit::Px)
        );
        // border-top side shorthand.
        let d = decls("p { border-top: 2px dotted blue; }");
        assert_eq!(d.len(), 3);
        assert_eq!(
            get(&d, "border-top-width"),
            &CssValue::Length(2.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "border-top-style"),
            &CssValue::Keyword("dotted".into())
        );
        // Invalid border dropped.
        let d = decls("p { border: 1px frobnicate red; }");
        assert!(d.is_empty());
        // Duplicate component dropped.
        let d = decls("p { border: 1px 2px solid; }");
        assert!(d.is_empty());
        // important propagates through the side expansion.
        let d = decls("p { border: 1px solid red !important; }");
        assert_eq!(d.len(), 12);
        assert!(d.iter().all(|x| x.important));
    }

    #[test]
    fn border_radius_expansion() {
        // One value: all four corners.
        let d = decls("div { border-radius: 6px; }");
        assert_eq!(d.len(), 4);
        for corner in ["top-left", "top-right", "bottom-right", "bottom-left"] {
            assert_eq!(
                get(&d, &format!("border-{corner}-radius")),
                &CssValue::Length(6.0, LengthUnit::Px)
            );
        }
        // Two values: [top-left/bottom-right, top-right/bottom-left].
        let d = decls("div { border-radius: 1px 2px; }");
        assert_eq!(
            get(&d, "border-top-left-radius"),
            &CssValue::Length(1.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "border-top-right-radius"),
            &CssValue::Length(2.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "border-bottom-right-radius"),
            &CssValue::Length(1.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "border-bottom-left-radius"),
            &CssValue::Length(2.0, LengthUnit::Px)
        );
        // Three and four values.
        let d = decls("div { border-radius: 1px 2px 3px; }");
        assert_eq!(
            get(&d, "border-bottom-right-radius"),
            &CssValue::Length(3.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "border-bottom-left-radius"),
            &CssValue::Length(2.0, LengthUnit::Px)
        );
        let d = decls("div { border-radius: 1px 2px 3px 4px; }");
        assert_eq!(
            get(&d, "border-bottom-right-radius"),
            &CssValue::Length(3.0, LengthUnit::Px)
        );
        assert_eq!(
            get(&d, "border-bottom-left-radius"),
            &CssValue::Length(4.0, LengthUnit::Px)
        );
        // Percentages and unitless zero are valid components.
        let d = decls("div { border-radius: 50% 0; }");
        assert_eq!(
            get(&d, "border-top-left-radius"),
            &CssValue::Percentage(50.0)
        );
        assert_eq!(get(&d, "border-top-right-radius"), &CssValue::Number(0.0));
        // `/`-separated horizontal/vertical radii: each corner gets a
        // [horizontal vertical] pair.
        let d = decls("div { border-radius: 10px / 5px; }");
        assert_eq!(d.len(), 4);
        assert_eq!(
            get(&d, "border-top-left-radius"),
            &CssValue::List(vec![
                CssValue::Length(10.0, LengthUnit::Px),
                CssValue::Length(5.0, LengthUnit::Px),
            ])
        );
        let d = decls("div { border-radius: 1px 2px / 3px 4px; }");
        assert_eq!(
            get(&d, "border-top-left-radius"),
            &CssValue::List(vec![
                CssValue::Length(1.0, LengthUnit::Px),
                CssValue::Length(3.0, LengthUnit::Px),
            ])
        );
        assert_eq!(
            get(&d, "border-bottom-right-radius"),
            &CssValue::List(vec![
                CssValue::Length(1.0, LengthUnit::Px),
                CssValue::Length(3.0, LengthUnit::Px),
            ])
        );
        // B10: the `/`-separated vertical group accepts 1-4 values, like
        // the horizontal group. Three vertical values map [a b c b].
        let d = decls("div { border-radius: 1px 2px / 3px 4px 5px; }");
        assert_eq!(d.len(), 4);
        assert_eq!(
            get(&d, "border-top-left-radius"),
            &CssValue::List(vec![
                CssValue::Length(1.0, LengthUnit::Px),
                CssValue::Length(3.0, LengthUnit::Px),
            ])
        );
        assert_eq!(
            get(&d, "border-bottom-right-radius"),
            &CssValue::List(vec![
                CssValue::Length(1.0, LengthUnit::Px),
                CssValue::Length(5.0, LengthUnit::Px),
            ])
        );
        assert_eq!(
            get(&d, "border-bottom-left-radius"),
            &CssValue::List(vec![
                CssValue::Length(2.0, LengthUnit::Px),
                CssValue::Length(4.0, LengthUnit::Px),
            ])
        );
        // Full 4/4 slash shorthand.
        let d = decls("div { border-radius: 1px 2px 3px 4px / 5px 6px 7px 8px; }");
        assert_eq!(d.len(), 4);
        assert_eq!(
            get(&d, "border-top-left-radius"),
            &CssValue::List(vec![
                CssValue::Length(1.0, LengthUnit::Px),
                CssValue::Length(5.0, LengthUnit::Px),
            ])
        );
        assert_eq!(
            get(&d, "border-top-right-radius"),
            &CssValue::List(vec![
                CssValue::Length(2.0, LengthUnit::Px),
                CssValue::Length(6.0, LengthUnit::Px),
            ])
        );
        assert_eq!(
            get(&d, "border-bottom-right-radius"),
            &CssValue::List(vec![
                CssValue::Length(3.0, LengthUnit::Px),
                CssValue::Length(7.0, LengthUnit::Px),
            ])
        );
        assert_eq!(
            get(&d, "border-bottom-left-radius"),
            &CssValue::List(vec![
                CssValue::Length(4.0, LengthUnit::Px),
                CssValue::Length(8.0, LengthUnit::Px),
            ])
        );
        // Invalid: non-length component, too many values, doubled slash.
        assert!(decls("div { border-radius: solid; }").is_empty());
        assert!(decls("div { border-radius: 1px 2px 3px 4px 5px; }").is_empty());
        assert!(decls("div { border-radius: 1px / 2px / 3px; }").is_empty());
        assert!(decls("div { border-radius: 1px 2px / 3px 4px 5px 6px 7px; }").is_empty());
        // important propagates to the corner longhands.
        let d = decls("div { border-radius: 6px !important; }");
        assert_eq!(d.len(), 4);
        assert!(d.iter().all(|x| x.important));
        // var() defers: stored as-is under the shorthand name.
        let d = decls("div { border-radius: var(--r); }");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].property, "border-radius");
    }

    /// Assert one `flex` shorthand expands to the expected longhands.
    fn assert_flex(input: &str, grow: CssValue, shrink: CssValue, basis: CssValue) {
        let d = decls(&format!("div {{ flex: {input}; }}"));
        assert_eq!(d.len(), 3, "flex: {input}");
        assert_eq!(get(&d, "flex-grow"), &grow, "flex: {input}");
        assert_eq!(get(&d, "flex-shrink"), &shrink, "flex: {input}");
        assert_eq!(get(&d, "flex-basis"), &basis, "flex: {input}");
    }

    #[test]
    fn flex_expansion() {
        use CssValue::{Keyword, Length, Number, Percentage};
        let px = |v: f32| Length(v, LengthUnit::Px);
        // `flex: none` -> `0 0 auto` (case-insensitive).
        assert_flex("none", Number(0.0), Number(0.0), Keyword("auto".into()));
        assert_flex("NONE", Number(0.0), Number(0.0), Keyword("auto".into()));
        // A single number -> `<n> 1 0%`.
        assert_flex("1", Number(1.0), Number(1.0), Percentage(0.0));
        assert_flex("2.5", Number(2.5), Number(1.0), Percentage(0.0));
        // A unitless zero is a factor, never a basis: `flex: 0` -> `0 1 0%`.
        assert_flex("0", Number(0.0), Number(1.0), Percentage(0.0));
        // A single basis -> `1 1 <basis>`.
        assert_flex("100px", Number(1.0), Number(1.0), px(100.0));
        assert_flex("50%", Number(1.0), Number(1.0), Percentage(50.0));
        assert_flex("auto", Number(1.0), Number(1.0), Keyword("auto".into()));
        assert_flex(
            "content",
            Number(1.0),
            Number(1.0),
            Keyword("content".into()),
        );
        // Two values: factor + factor, factor + basis, basis + factor.
        assert_flex("2 1", Number(2.0), Number(1.0), Percentage(0.0));
        assert_flex("2 30px", Number(2.0), Number(1.0), px(30.0));
        assert_flex("30px 2", Number(2.0), Number(1.0), px(30.0));
        // Three values: grow shrink basis in either order around the basis.
        assert_flex("2 1 30px", Number(2.0), Number(1.0), px(30.0));
        assert_flex("30px 2 1", Number(2.0), Number(1.0), px(30.0));
        // Intrinsic-keyword bases: the basis slot takes the full
        // `flex-basis` grammar (CSS flexbox-1 §7.1), not just
        // `auto`/`content`.
        assert_flex(
            "1 1 max-content",
            Number(1.0),
            Number(1.0),
            Keyword("max-content".into()),
        );
        assert_flex(
            "1 1 min-content",
            Number(1.0),
            Number(1.0),
            Keyword("min-content".into()),
        );
        assert_flex(
            "0 0 fit-content",
            Number(0.0),
            Number(0.0),
            Keyword("fit-content".into()),
        );
        assert_flex(
            "2 1 max-content",
            Number(2.0),
            Number(1.0),
            Keyword("max-content".into()),
        );
        assert_flex(
            "max-content",
            Number(1.0),
            Number(1.0),
            Keyword("max-content".into()),
        );
        assert_flex(
            "1 1 stretch",
            Number(1.0),
            Number(1.0),
            Keyword("stretch".into()),
        );
        assert_flex(
            "2 max-content",
            Number(2.0),
            Number(1.0),
            Keyword("max-content".into()),
        );
        assert_flex(
            "max-content 2 1",
            Number(2.0),
            Number(1.0),
            Keyword("max-content".into()),
        );
        // Two intrinsic bases are still invalid: nothing is set.
        assert!(decls("div { flex: max-content min-content; }").is_empty());
        // Regressions: three-value auto/content/percentage bases.
        assert_flex("2 1 auto", Number(2.0), Number(1.0), Keyword("auto".into()));
        assert_flex(
            "2 1 content",
            Number(2.0),
            Number(1.0),
            Keyword("content".into()),
        );
        assert_flex("1 1 50%", Number(1.0), Number(1.0), Percentage(50.0));
        // Invalid: unknown keyword, two bases, three factors, calc().
        assert!(decls("div { flex: solid; }").is_empty());
        assert!(decls("div { flex: 10px 20px; }").is_empty());
        assert!(decls("div { flex: 1 2 3; }").is_empty());
        assert!(decls("div { flex: 1 2 3 4px; }").is_empty());
        assert!(decls("div { flex: calc(1); }").is_empty());
        assert!(decls("div { flex: none 1; }").is_empty());
        // important propagates to the longhands.
        let d = decls("div { flex: 1 !important; }");
        assert_eq!(d.len(), 3);
        assert!(d.iter().all(|decl| decl.important));
    }

    #[test]
    fn font_expansion() {
        let d = decls("p { font: italic bold 12px/1.5 \"Helvetica Neue\", sans-serif; }");
        assert_eq!(get(&d, "font-style"), &CssValue::Keyword("italic".into()));
        assert_eq!(get(&d, "font-weight"), &CssValue::Keyword("bold".into()));
        assert_eq!(
            get(&d, "font-size"),
            &CssValue::Length(12.0, LengthUnit::Px)
        );
        assert_eq!(get(&d, "line-height"), &CssValue::Number(1.5));
        assert_eq!(
            get(&d, "font-family"),
            &CssValue::List(vec![
                CssValue::String("Helvetica Neue".into()),
                CssValue::String("sans-serif".into()),
            ])
        );
        // Minimal font: omitted longhands reset to `normal`.
        let d = decls("p { font: 16px serif; }");
        assert_eq!(d.len(), 6);
        assert_eq!(
            get(&d, "font-size"),
            &CssValue::Length(16.0, LengthUnit::Px)
        );
        assert_eq!(get(&d, "font-family"), &CssValue::String("serif".into()));
        assert_eq!(get(&d, "font-style"), &CssValue::Keyword("normal".into()));
        assert_eq!(get(&d, "font-variant"), &CssValue::Keyword("normal".into()));
        assert_eq!(get(&d, "font-weight"), &CssValue::Keyword("normal".into()));
        assert_eq!(get(&d, "line-height"), &CssValue::Keyword("normal".into()));
        // Multi-word unquoted family.
        let d = decls("p { font: 16px Times New Roman; }");
        assert_eq!(
            get(&d, "font-family"),
            &CssValue::String("Times New Roman".into())
        );
        // Numeric weight + line-height length.
        let d = decls("p { font: 700 12px/18px sans-serif; }");
        assert_eq!(get(&d, "font-weight"), &CssValue::Number(700.0));
        assert_eq!(
            get(&d, "line-height"),
            &CssValue::Length(18.0, LengthUnit::Px)
        );
        // Missing size or family: dropped.
        assert!(decls("p { font: bold; }").is_empty());
        assert!(decls("p { font: 12px; }").is_empty());
    }

    #[test]
    fn font_shorthand_modern_math_fns() {
        use crate::value::{MathExpr, RoundStrategy};
        // The token-level math gate (`math_fn_at`) accepts the CSS Values
        // 4 math functions, not just calc()/min()/max()/clamp().
        let d = decls("p { font: italic abs(12px)/round(1.6) serif; }");
        assert_eq!(
            get(&d, "font-size"),
            &CssValue::Math(MathExpr::Abs(Box::new(MathExpr::Length(
                12.0,
                LengthUnit::Px
            ))))
        );
        assert_eq!(
            get(&d, "line-height"),
            &CssValue::Math(MathExpr::Round {
                strategy: RoundStrategy::Nearest,
                a: Box::new(MathExpr::Number(1.6)),
                b: None,
            })
        );
        // An unknown function in size position still drops the shorthand.
        assert!(decls("p { font: frobnicate(12px) serif; }").is_empty());
    }

    #[test]
    fn shorthand_wide_keyword_expansions() {
        // w78fix H4: a lone CSS-wide keyword expands per longhand
        // for `border`/`border-<side>`/`font`/`flex`/`columns` (and
        // `gap`, which already copied it), instead of silently
        // dropping the declaration. A wide keyword combined with
        // other components is still invalid and drops.
        let inherit = CssValue::Keyword("inherit".into());
        let d = decls("p { border: inherit; }");
        assert_eq!(d.len(), 12);
        assert!(d.iter().all(|decl| decl.value == inherit));
        let d = decls("p { border-top: inherit; }");
        assert_eq!(d.len(), 3);
        assert!(d.iter().all(|decl| decl.value == inherit));
        let d = decls("p { font: inherit; }");
        assert_eq!(d.len(), 6);
        assert!(d.iter().all(|decl| decl.value == inherit));
        let d = decls("p { flex: inherit; }");
        assert_eq!(d.len(), 3);
        assert!(d.iter().all(|decl| decl.value == inherit));
        let d = decls("p { columns: inherit; }");
        assert_eq!(d.len(), 2);
        assert!(d.iter().all(|decl| decl.value == inherit));
        let d = decls("p { gap: inherit; }");
        assert_eq!(d.len(), 2);
        assert!(d.iter().all(|decl| decl.value == inherit));
        assert!(decls("p { border: inherit solid; }").is_empty());
        assert!(decls("p { flex: inherit 1; }").is_empty());
        assert!(decls("p { font: inherit 12px serif; }").is_empty());
        assert!(decls("p { gap: inherit 10px; }").is_empty());
        assert!(decls("p { columns: inherit 3; }").is_empty());
    }

    #[test]
    fn font_system_keywords() {
        // All six system font keywords expand to the UA metrics from
        // `system_font_values` (CSS Fonts 3 §3.2).
        for kw in [
            "caption",
            "icon",
            "menu",
            "message-box",
            "small-caption",
            "status-bar",
        ] {
            let d = decls(&format!("p {{ font: {kw}; }}"));
            assert_eq!(d.len(), 6, "font: {kw}");
            assert_eq!(get(&d, "font-style"), &CssValue::Keyword("normal".into()));
            assert_eq!(get(&d, "font-variant"), &CssValue::Keyword("normal".into()));
            assert_eq!(get(&d, "font-weight"), &CssValue::Keyword("normal".into()));
            let expected_size = if kw == "small-caption" { 12.0 } else { 13.0 };
            assert_eq!(
                get(&d, "font-size"),
                &CssValue::Length(expected_size, LengthUnit::Px),
                "font: {kw}"
            );
            assert_eq!(get(&d, "line-height"), &CssValue::Keyword("normal".into()));
            assert_eq!(
                get(&d, "font-family"),
                &CssValue::String("sans-serif".into())
            );
        }
        // Case-insensitive.
        let d = decls("p { font: CAPTION; }");
        assert_eq!(
            get(&d, "font-size"),
            &CssValue::Length(13.0, LengthUnit::Px)
        );
        // The keyword must be the sole value: any combination is invalid.
        assert!(decls("p { font: caption 12px; }").is_empty());
        assert!(decls("p { font: bold caption; }").is_empty());
        assert!(decls("p { font: caption serif; }").is_empty());
        assert!(decls("p { font: caption caption; }").is_empty());
        // `!important` propagates to all six longhands.
        let d = decls("p { font: menu !important; }");
        assert_eq!(d.len(), 6);
        assert!(d.iter().all(|x| x.important));
    }

    #[test]
    fn background_expansion() {
        let d = decls("p { background: red url(a.png) no-repeat center; }");
        assert_eq!(
            get(&d, "background-color"),
            &CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
        assert_eq!(get(&d, "background-image"), &CssValue::Url("a.png".into()));
        assert_eq!(
            get(&d, "background-repeat"),
            &CssValue::Keyword("no-repeat".into())
        );
        assert_eq!(
            get(&d, "background-position"),
            &CssValue::Keyword("center".into())
        );
        // Omitted attachment resets to its initial value.
        assert_eq!(
            get(&d, "background-attachment"),
            &CssValue::Keyword("scroll".into())
        );
        assert_eq!(d.len(), 5);
        let d = decls("p { background: #fff; }");
        assert_eq!(d.len(), 5);
        assert_eq!(
            get(&d, "background-color"),
            &CssValue::Color(Color {
                r: 255,
                g: 255,
                b: 255,
                a: 255
            })
        );
        // Omitted longhands reset to initial values.
        assert_eq!(
            get(&d, "background-image"),
            &CssValue::Keyword("none".into())
        );
        assert_eq!(
            get(&d, "background-position"),
            &CssValue::List(vec![CssValue::Percentage(0.0), CssValue::Percentage(0.0),])
        );
        let d = decls("p { background: 10px 20px; }");
        assert_eq!(d.len(), 5);
        assert_eq!(
            get(&d, "background-position"),
            &CssValue::List(vec![
                CssValue::Length(10.0, LengthUnit::Px),
                CssValue::Length(20.0, LengthUnit::Px),
            ])
        );
    }

    #[test]
    fn list_style_expansion() {
        let d = decls("ul { list-style: square inside; }");
        assert_eq!(d.len(), 3);
        assert_eq!(
            get(&d, "list-style-type"),
            &CssValue::Keyword("square".into())
        );
        assert_eq!(
            get(&d, "list-style-position"),
            &CssValue::Keyword("inside".into())
        );
        // Omitted image resets to its initial value.
        assert_eq!(
            get(&d, "list-style-image"),
            &CssValue::Keyword("none".into())
        );
        let d = decls("ul { list-style: none; }");
        assert_eq!(d.len(), 3);
        assert_eq!(
            get(&d, "list-style-type"),
            &CssValue::Keyword("none".into())
        );
        assert_eq!(
            get(&d, "list-style-position"),
            &CssValue::Keyword("outside".into())
        );
        assert_eq!(
            get(&d, "list-style-image"),
            &CssValue::Keyword("none".into())
        );
    }

    #[test]
    fn media_queries() {
        let s = sheet("@media (min-width: 100px) { div { color: red; } }");
        assert_eq!(s.rules().len(), 1);
        let mq = s.rules()[0].media.as_ref().unwrap();
        assert!(mq.matches(200.0, 800.0));
        assert!(!mq.matches(50.0, 800.0));

        let s = sheet(
            "@media screen and (min-width: 100px) and (max-width: 200px) { div { color: red; } }",
        );
        let mq = s.rules()[0].media.as_ref().unwrap();
        assert!(mq.matches(150.0, 800.0));
        assert!(!mq.matches(50.0, 800.0));
        assert!(!mq.matches(250.0, 800.0));

        // print never matches
        let s = sheet("@media print { div { color: red; } }");
        assert!(!s.rules()[0].media.as_ref().unwrap().matches(1000.0, 800.0));

        // comma = OR
        let s = sheet("@media (max-width: 100px), (min-width: 500px) { div { color: red; } }");
        let mq = s.rules()[0].media.as_ref().unwrap();
        assert!(mq.matches(50.0, 800.0));
        assert!(mq.matches(600.0, 800.0));
        assert!(!mq.matches(300.0, 800.0));

        // unknown feature never matches
        let s = sheet("@media (color) { div { color: red; } }");
        assert!(!s.rules()[0].media.as_ref().unwrap().matches(1000.0, 800.0));

        // source order continues through media blocks
        let s = sheet(
            "a { color: red; } @media (min-width: 1px) { b { color: blue; } } c { color: green; }",
        );
        assert_eq!(s.rules().len(), 3);
        assert_eq!(s.rules()[0].order, 0);
        assert_eq!(s.rules()[1].order, 1);
        assert_eq!(s.rules()[2].order, 2);
        assert!(s.rules()[1].media.is_some());
        assert!(s.rules()[2].media.is_none());

        // nested @media ANDs the conditions
        let s = sheet(
            "@media (min-width: 100px) { @media (max-width: 200px) { div { color: red; } } }",
        );
        let mq = s.rules()[0].media.as_ref().unwrap();
        assert!(mq.matches(150.0, 800.0));
        assert!(!mq.matches(50.0, 800.0));
        assert!(!mq.matches(250.0, 800.0));

        // em values in media features
        let s = sheet("@media (min-width: 10em) { div { color: red; } }");
        let mq = s.rules()[0].media.as_ref().unwrap();
        assert!(mq.matches(160.0, 800.0));
        assert!(!mq.matches(159.0, 800.0));

        // height features compare against the viewport height
        let s = sheet("@media (min-height: 700px) { div { color: red; } }");
        let mq = s.rules()[0].media.as_ref().unwrap();
        assert!(mq.matches(800.0, 700.0));
        assert!(mq.matches(800.0, 900.0));
        assert!(!mq.matches(800.0, 699.0));
        // width does not satisfy a height condition
        assert!(!mq.matches(900.0, 100.0));

        let s = sheet("@media (max-height: 700px) { div { color: red; } }");
        let mq = s.rules()[0].media.as_ref().unwrap();
        assert!(mq.matches(800.0, 700.0));
        assert!(!mq.matches(800.0, 701.0));

        // width and height conditions AND together
        let s = sheet("@media (min-width: 100px) and (max-height: 700px) { div { color: red; } }");
        let mq = s.rules()[0].media.as_ref().unwrap();
        assert!(mq.matches(150.0, 600.0));
        assert!(!mq.matches(50.0, 600.0));
        assert!(!mq.matches(150.0, 800.0));
    }

    #[test]
    fn at_rules_skipped() {
        let s = sheet("@font-face { font-family: x; src: url(a.woff); } div { color: red; }");
        assert_eq!(s.rules().len(), 1);
        let s = sheet("@import url(a.css); div { color: red; }");
        assert_eq!(s.rules().len(), 1);
        let s = sheet(
            "@keyframes spin { from { opacity: 0; } to { opacity: 1; } } div { color: red; }",
        );
        assert_eq!(s.rules().len(), 1);
        // Unknown at-rule with block skipped.
        let s = sheet("@unknown { div { color: red; } } p { color: blue; }");
        assert_eq!(s.rules().len(), 1);
        // @media without block is fine.
        let s = sheet("@media (min-width: 1px); div { color: red; }");
        assert_eq!(s.rules().len(), 1);
    }

    #[test]
    fn font_face_retained() {
        // The WPT ahem.css shape: quoted family, quoted url().
        let s = sheet(
            "@font-face { font-family: 'Ahem'; src: url('/fonts/Ahem.ttf'); } \
             div { color: red; }",
        );
        // Style rules are unaffected: @font-face is not a style rule.
        assert_eq!(s.rules().len(), 1);
        let faces = s.font_faces();
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].family, "Ahem");
        assert_eq!(faces[0].order, 0);
        assert_eq!(
            faces[0].src,
            vec![FontFaceSrc::Url {
                url: "/fonts/Ahem.ttf".to_owned(),
                format: None,
            }]
        );
    }

    #[test]
    fn font_face_src_forms() {
        let s = sheet(
            "@font-face { font-family: Test; \
             src: url(a.woff2) format('woff2'), local('Test Local'), \
                  url(\"b.ttf\") format(truetype); }",
        );
        let faces = s.font_faces();
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].family, "Test");
        assert_eq!(
            faces[0].src,
            vec![
                FontFaceSrc::Url {
                    url: "a.woff2".to_owned(),
                    format: Some("woff2".to_owned()),
                },
                FontFaceSrc::Local("Test Local".to_owned()),
                FontFaceSrc::Url {
                    url: "b.ttf".to_owned(),
                    format: Some("truetype".to_owned()),
                },
            ]
        );
    }

    #[test]
    fn font_face_weight_style_descriptors() {
        let s = sheet(
            "@font-face { font-family: Test; font-weight: 700; font-style: italic; \
             src: url(a.ttf); }",
        );
        let faces = s.font_faces();
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].weight, 700);
        assert_eq!(faces[0].weight_range, None);
        assert_eq!(faces[0].style, FontFaceStyle::Italic);
        assert_eq!(faces[0].font_style_name(), "italic");

        // Keywords, ranges, and oblique-with-angle.
        let s = sheet(
            "@font-face { font-family: A; font-weight: bold; src: url(a.ttf); } \
             @font-face { font-family: B; font-weight: 100 900; font-style: oblique 14deg; \
             src: url(b.ttf); }",
        );
        let faces = s.font_faces();
        assert_eq!(faces[0].weight, 700);
        assert_eq!(faces[1].weight, 100);
        assert_eq!(faces[1].weight_range, Some((100, 900)));
        assert_eq!(faces[1].style, FontFaceStyle::Oblique);

        // Defaults when the descriptors are absent, and invalid values
        // are ignored without dropping the rule.
        let s = sheet(
            "@font-face { font-family: C; src: url(c.ttf); } \
             @font-face { font-family: D; font-weight: heavy; font-style: slanted; \
             src: url(d.ttf); }",
        );
        let faces = s.font_faces();
        assert_eq!(faces.len(), 2);
        assert_eq!(faces[0].weight, 400);
        assert_eq!(faces[0].style, FontFaceStyle::Normal);
        assert_eq!(faces[1].weight, 400);
        assert_eq!(faces[1].style, FontFaceStyle::Normal);
        // url_srcs skips local() entries.
        let s =
            sheet("@font-face { font-family: E; src: local('E'), url(e.ttf) format('truetype'); }");
        let faces = s.font_faces();
        assert_eq!(faces[0].url_srcs().collect::<Vec<_>>(), vec!["e.ttf"]);
    }

    #[test]
    fn font_face_malformed_dropped() {
        // Missing family: dropped, surrounding rules survive.
        let s = sheet("@font-face { src: url(a.woff); } div { color: red; }");
        assert!(s.font_faces().is_empty());
        assert_eq!(s.rules().len(), 1);
        // Missing src: dropped.
        let s = sheet("@font-face { font-family: x; } div { color: red; }");
        assert!(s.font_faces().is_empty());
        assert_eq!(s.rules().len(), 1);
        // Non-empty prelude: dropped per CSS Fonts 4.
        let s = sheet("@font-face screen { font-family: x; src: url(a.woff); }");
        assert!(s.font_faces().is_empty());
        // Multiple rules keep source order.
        let s = sheet(
            "@font-face { font-family: a; src: url(a.woff); } \
             @font-face { font-family: b; src: local(b); }",
        );
        let faces = s.font_faces();
        assert_eq!(faces.len(), 2);
        assert_eq!(faces[0].family, "a");
        assert_eq!(faces[0].order, 0);
        assert_eq!(faces[1].family, "b");
        assert_eq!(faces[1].order, 1);
    }

    #[test]
    fn cdo_cdc_tolerated() {
        let s = sheet("<!-- div { color: red; } -->");
        assert_eq!(s.rules().len(), 1);
    }

    #[test]
    fn media_not_all_empty() {
        let s = sheet("@media not screen { div { color: red; } }");
        assert!(!s.rules()[0].media.as_ref().unwrap().matches(1000.0, 800.0));
        let s = sheet("@media all and (min-width: 10px) { div { color: red; } }");
        assert!(s.rules()[0].media.as_ref().unwrap().matches(100.0, 800.0));
        // An empty @media query matches everything.
        let s = sheet("@media { div { color: red; } }");
        assert!(s.rules()[0].media.as_ref().unwrap().matches(1.0, 1.0));
        // max-width boundary is inclusive.
        let s = sheet("@media (max-width: 200px) { div { color: red; } }");
        let mq = s.rules()[0].media.as_ref().unwrap();
        assert!(mq.matches(200.0, 600.0));
        assert!(!mq.matches(200.1, 600.0));
    }

    #[test]
    fn kitchen_sink() {
        let css = r#"
            /* reset */
            *, *::before { margin: 0; padding: 0; }
            body { font: 16px/1.4 system-ui, sans-serif; color: #333; }
            #app > .card:hover { border: 1px solid #ccc; background: white url(bg.png) no-repeat; }
            @media screen and (min-width: 768px) {
                #app > .card { padding: 16px 24px !important; }
            }
            @font-face { font-family: x; src: url(x.woff2); }
        "#;
        let s = sheet(css);
        assert_eq!(s.rules().len(), 4);
        assert_eq!(s.rules()[0].selectors.len(), 2);
        assert_eq!(s.rules()[0].declarations.len(), 8); // margin x4 + padding x4
        assert_eq!(s.rules()[3].order, 3);
        assert!(s.rules()[3].media.is_some());
        let mq = s.rules()[3].media.as_ref().unwrap();
        assert!(mq.matches(1024.0, 768.0));
        assert!(!mq.matches(320.0, 568.0));
        assert!(s.rules()[3].declarations.iter().all(|d| d.important));
    }

    #[test]
    fn parse_declarations_fn() {
        let d = parse_declarations("color: red; margin: 1px 2px;");
        assert_eq!(d.len(), 1 + 4);
        assert_eq!(parse_declarations("color red;").len(), 0);
        assert_eq!(parse_declarations("").len(), 0);
    }

    #[test]
    fn var_in_declaration() {
        let d = decls("p { color: var(--main, blue); }");
        assert!(matches!(d[0].value, CssValue::Var(_, _)));
        let d = decls(":root { --main: red; }");
        assert_eq!(d[0].property, "--main");
    }

    #[test]
    fn custom_property_case_preserved() {
        // var() names keep their case; property names lowercase normally.
        let d = decls("p { color: var(--MyColor); }");
        assert_eq!(d[0].value, CssValue::Var("--MyColor".into(), None));
    }

    #[test]
    fn custom_property_stores_raw_tokens() {
        // Custom properties are token-preserving: the raw stream is kept
        // verbatim, even when it could never parse as a normal value
        // (two top-level slashes used to drop the whole declaration).
        let d = decls("p { --x: 1/2/3; }");
        assert!(
            matches!(d[0].value, CssValue::Tokens(_)),
            "expected verbatim tokens, got {:?}",
            d[0].value
        );
        // ... including a single slash, which must not be rewritten here.
        let d = decls("p { --ratio: 16/9; }");
        assert!(matches!(d[0].value, CssValue::Tokens(_)));
        // `!important` is still stripped and recorded.
        let d = decls("p { --x: 1px !important; }");
        assert!(d[0].important);
        assert!(matches!(d[0].value, CssValue::Tokens(_)));
    }

    /// Last declaration for a property (later source order wins in the
    /// cascade, so this is the value that applies).
    fn last<'a>(decls: &'a [Declaration], prop: &str) -> &'a CssValue {
        &decls.iter().rfind(|d| d.property == prop).unwrap().value
    }

    #[test]
    fn media_only_keyword() {
        // `@media only screen` matches like `screen`.
        let s = sheet("@media only screen { div { color: red; } }");
        assert!(s.rules()[0].media.as_ref().unwrap().matches(1000.0, 800.0));
        let s = sheet("@media ONLY screen { div { color: red; } }");
        assert!(s.rules()[0].media.as_ref().unwrap().matches(1000.0, 800.0));
        // `only` with conditions.
        let s = sheet("@media only screen and (min-width: 100px) { div { color: red; } }");
        let mq = s.rules()[0].media.as_ref().unwrap();
        assert!(mq.matches(200.0, 800.0));
        assert!(!mq.matches(50.0, 800.0));
        // `only` must precede a media type: anything else is unknown.
        let s = sheet("@media only { div { color: red; } }");
        assert!(!s.rules()[0].media.as_ref().unwrap().matches(1000.0, 800.0));
        let s = sheet("@media only (min-width: 100px) { div { color: red; } }");
        assert!(!s.rules()[0].media.as_ref().unwrap().matches(1000.0, 800.0));
        let s = sheet("@media not only screen { div { color: red; } }");
        assert!(!s.rules()[0].media.as_ref().unwrap().matches(1000.0, 800.0));
        // `only print` still never matches.
        let s = sheet("@media only print { div { color: red; } }");
        assert!(!s.rules()[0].media.as_ref().unwrap().matches(1000.0, 800.0));
    }

    #[test]
    fn shorthand_resets_omitted_longhands() {
        // background resets a previously set image to none.
        let d = decls("p { background-image: url(a.png); background: red; }");
        assert_eq!(
            last(&d, "background-image"),
            &CssValue::Keyword("none".into())
        );
        assert_eq!(
            last(&d, "background-color"),
            &CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
        // font resets a previously set weight to normal.
        let d = decls("p { font-weight: 700; font: 16px serif; }");
        assert_eq!(last(&d, "font-weight"), &CssValue::Keyword("normal".into()));
        assert_eq!(last(&d, "font-style"), &CssValue::Keyword("normal".into()));
        // border resets a previously set width to medium.
        let d = decls("p { border-top-width: 9px; border: dashed; }");
        assert_eq!(
            last(&d, "border-top-width"),
            &CssValue::Keyword("medium".into())
        );
        assert_eq!(
            last(&d, "border-top-style"),
            &CssValue::Keyword("dashed".into())
        );
        // list-style resets a previously set image to none.
        let d = decls("ul { list-style-image: url(a.png); list-style: none; }");
        assert_eq!(
            last(&d, "list-style-image"),
            &CssValue::Keyword("none".into())
        );
        assert_eq!(
            last(&d, "list-style-type"),
            &CssValue::Keyword("none".into())
        );
    }

    #[test]
    fn var_in_shorthand_defers_expansion() {
        // A shorthand containing var() is stored as-is under the shorthand
        // property name; the cascade substitutes and expands it later.
        let d = decls("p { border: var(--w) solid red; }");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].property, "border");
        assert_eq!(
            d[0].value,
            CssValue::List(vec![
                CssValue::Var("--w".into(), None),
                CssValue::Keyword("solid".into()),
                CssValue::Color(Color {
                    r: 255,
                    g: 0,
                    b: 0,
                    a: 255
                }),
            ])
        );
        assert!(!d[0].important);

        let d = decls("p { background: var(--bg); }");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].property, "background");
        assert!(matches!(d[0].value, CssValue::Var(_, _)));

        let d = decls("ul { list-style: var(--ls); }");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].property, "list-style");
        assert!(matches!(d[0].value, CssValue::Var(_, _)));

        // margin/padding defer the same way.
        let d = decls("p { margin: var(--m); }");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].property, "margin");

        // font with var() is stored as-is too.
        let d = decls("p { font: var(--f); }");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].property, "font");
        assert_eq!(d[0].value, CssValue::Var("--f".into(), None));

        // important is preserved on the deferred declaration.
        let d = decls("p { border: var(--w) solid red !important; }");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].property, "border");
        assert!(d[0].important);
    }

    #[test]
    fn media_query_strict_structure() {
        // Missing `and` between conditions: unknown, never matches.
        let s = sheet("@media (min-width: 100px) (max-width: 200px) { div { color: red; } }");
        assert!(!s.rules()[0].media.as_ref().unwrap().matches(150.0, 800.0));
        // Leading `and`: unknown.
        let s = sheet("@media and (min-width: 100px) { div { color: red; } }");
        assert!(!s.rules()[0].media.as_ref().unwrap().matches(1000.0, 800.0));
        // Trailing `and`: unknown.
        let s = sheet("@media (min-width: 100px) and { div { color: red; } }");
        assert!(!s.rules()[0].media.as_ref().unwrap().matches(1000.0, 800.0));
        // Trailing garbage inside a condition: unknown.
        let s = sheet("@media (min-width: 100px foo) { div { color: red; } }");
        assert!(!s.rules()[0].media.as_ref().unwrap().matches(1000.0, 800.0));
        // Unitless lengths are invalid in media features: unknown.
        let s = sheet("@media (min-width: 100) { div { color: red; } }");
        assert!(!s.rules()[0].media.as_ref().unwrap().matches(1000.0, 800.0));
        // Sanity: the valid forms still match.
        let s = sheet(
            "@media screen and (min-width: 100px) and (max-width: 200px) { div { color: red; } }",
        );
        let mq = s.rules()[0].media.as_ref().unwrap();
        assert!(mq.matches(150.0, 800.0));
        assert!(!mq.matches(50.0, 800.0));
    }

    #[test]
    fn unterminated_url_drops_declaration() {
        // `url(` at EOF is a parse error: the declaration is dropped.
        assert!(decls("p { background: url(").is_empty());
        // An earlier valid declaration in the same block still survives.
        let d = decls("p { color: red; background: url( }");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].property, "color");
    }

    #[test]
    fn background_image_gradient_kept() {
        // `background-image` accepts gradient functions (CSS Images 3
        // §3.1); the declaration survives parse time.
        let d = decls("p { background-image: linear-gradient(red, blue); }");
        assert_eq!(
            get(&d, "background-image"),
            &CssValue::Gradient {
                name: "linear-gradient".into(),
                source: "linear-gradient(red, blue)".into(),
            }
        );
        // All six forms, including `repeating-` variants.
        for name in [
            "linear-gradient",
            "repeating-linear-gradient",
            "radial-gradient",
            "repeating-radial-gradient",
            "conic-gradient",
            "repeating-conic-gradient",
        ] {
            let d = decls(&format!("p {{ background-image: {name}(red); }}"));
            assert!(
                matches!(get(&d, "background-image"), CssValue::Gradient { name: n, .. } if n == name),
                "{name} should be kept"
            );
        }
        // An empty gradient body drops the declaration.
        assert!(decls("p { background-image: linear-gradient(); }").is_empty());
    }

    #[test]
    fn background_image_gradient_var_defers() {
        // A `var()` in the gradient body defers: the parser keeps the
        // whole function as a verbatim token stream (never a `Gradient`
        // with the `var()` frozen into `source`) so the cascade can
        // substitute at computed-value time.
        let d = decls("p { background-image: linear-gradient(var(--a), blue); }");
        let v = get(&d, "background-image");
        assert!(
            matches!(v, CssValue::Tokens(t) if t.iter().any(|t| matches!(t, Token::Function(n) if n.eq_ignore_ascii_case("var")))),
            "expected deferred Tokens, got {v:?}"
        );
        assert!(value_contains_var(v));
        // The `background` shorthand defers as well: it stays under the
        // shorthand name for the cascade to expand after substitution.
        let d = decls("p { background: linear-gradient(var(--a), blue); }");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].property, "background");
        assert!(value_contains_var(&d[0].value));
        // Defense-in-depth: a `Gradient` whose verbatim source still
        // mentions `var(` counts as var-containing.
        assert!(value_contains_var(&CssValue::Gradient {
            name: "linear-gradient".into(),
            source: "linear-gradient(var(--a), blue)".into(),
        }));
        assert!(!value_contains_var(&CssValue::Gradient {
            name: "linear-gradient".into(),
            source: "linear-gradient(red, blue)".into(),
        }));
    }

    #[test]
    fn background_shorthand_gradient_expansion() {
        // The `background` shorthand takes a gradient as its image
        // component and expands to the longhand.
        let d = decls("p { background: linear-gradient(to right, red, blue) no-repeat; }");
        assert_eq!(
            get(&d, "background-image"),
            &CssValue::Gradient {
                name: "linear-gradient".into(),
                source: "linear-gradient(to right, red, blue)".into(),
            }
        );
        assert_eq!(
            get(&d, "background-repeat"),
            &CssValue::Keyword("no-repeat".into())
        );
        // `background: red` still clears a gradient image.
        let d = decls("p { background-image: linear-gradient(red); background: red; }");
        assert_eq!(
            last(&d, "background-image"),
            &CssValue::Keyword("none".into())
        );
        // Two images (gradient + url) are invalid: the whole
        // declaration is dropped, mirroring `url url`.
        assert!(decls("p { background: linear-gradient(red) url(a.png); }").is_empty());
    }

    // ---- B2: @media nesting depth is capped ----

    #[test]
    fn hundred_k_nested_media_fails_closed() {
        // 100k nested `@media{` must not overflow the stack: nesting
        // past MAX_MEDIA_DEPTH (64) fails closed via iterative
        // skip_balanced_block (B2).
        let mut css = String::new();
        for _ in 0..100_000 {
            css.push_str("@media {");
        }
        css.push_str("div { color: red; }");
        for _ in 0..100_000 {
            css.push('}');
        }
        let start = std::time::Instant::now();
        let sheet = parse_stylesheet(&css);
        let elapsed = start.elapsed();
        // The over-deep rules are dropped; parsing terminates quickly.
        assert!(
            elapsed.as_secs() < 5,
            "100k nested @media must parse in <5s, took {elapsed:?}"
        );
        let _ = sheet;
    }

    #[test]
    fn media_nesting_under_cap_parses() {
        // Sanity: modest nesting still works.
        let sheet = parse_stylesheet("@media { @media { div { color: red; } } }");
        assert!(!sheet.rules().is_empty());
    }

    #[test]
    fn nested_media_cross_product_capped() {
        // Each `@media` contributes 2 comma parts; 10 nested levels
        // would cross-product to 2^10 = 1024 parts. The cap bounds it
        // at MAX_MEDIA_PARTS (256).
        let mut css = String::new();
        for i in 0..10 {
            css.push_str(&format!(
                "@media (min-width: {}px), (max-width: {}px) {{ ",
                i + 1,
                1000 + i
            ));
        }
        css.push_str("div { color: red; }");
        for _ in 0..10 {
            css.push_str(" }");
        }
        let sheet = parse_stylesheet(&css);
        assert_eq!(sheet.rules().len(), 1);
        let parts = sheet.rules()[0]
            .media
            .as_ref()
            .expect("rule must be media-gated")
            .parts
            .len();
        assert_eq!(
            parts, MAX_MEDIA_PARTS,
            "cross product must truncate at {MAX_MEDIA_PARTS}, got {parts}"
        );
        // The surviving parts still gate correctly: a 500px viewport
        // satisfies every min-width and max-width above.
        assert!(
            sheet.rules()[0]
                .media
                .as_ref()
                .unwrap()
                .matches(500.0, 500.0)
        );
    }

    #[test]
    fn decimal_nth_rejected_at_stylesheet_level() {
        // The token-only stylesheet parser has no source text to rescan,
        // so the `<integer>`-lexeme flag on numeric tokens (see
        // [`Token::Number`]) must reject decimal An+B coefficients here
        // just like the `&str` entry point: the whole rule is dropped.
        for bad in [
            "p:nth-child(2.0) { color: red; }",
            "p:nth-child(2n+1.5) { color: red; }",
            "p:nth-child(2.0n) { color: red; }",
            "p:nth-last-child(0.5) { color: red; }",
        ] {
            let sheet = parse_stylesheet(bad);
            assert!(sheet.rules().is_empty(), "{bad} should drop the rule");
        }
        // Integer forms still parse.
        for ok in [
            "p:nth-child(2) { color: red; }",
            "p:nth-child(2n+1) { color: red; }",
            "p:nth-child(odd) { color: red; }",
        ] {
            let sheet = parse_stylesheet(ok);
            assert_eq!(sheet.rules().len(), 1, "{ok} should parse");
        }
    }

    fn shadow_of(css: &str, prop: &str) -> Option<CssValue> {
        let decls = parse_declarations(css);
        if decls.len() != 1 {
            return None;
        }
        assert_eq!(decls[0].property, prop);
        Some(decls[0].value.clone())
    }

    fn layers(v: &CssValue) -> &Vec<Shadow> {
        match v {
            CssValue::Shadows(shadows) => shadows,
            _ => panic!("expected Shadows, got {v:?}"),
        }
    }

    fn len_px(v: &CssValue) -> f32 {
        match v {
            CssValue::Length(n, LengthUnit::Px) => *n,
            CssValue::Number(n) => *n,
            _ => panic!("expected px length, got {v:?}"),
        }
    }

    #[test]
    fn box_shadow_none() {
        assert_eq!(
            shadow_of("box-shadow: none", "box-shadow"),
            Some(CssValue::Shadows(Vec::new()))
        );
        // Case-insensitive.
        assert_eq!(
            shadow_of("box-shadow: NONE", "box-shadow"),
            Some(CssValue::Shadows(Vec::new()))
        );
        assert_eq!(
            shadow_of("text-shadow: none", "text-shadow"),
            Some(CssValue::Shadows(Vec::new()))
        );
    }

    #[test]
    fn box_shadow_basic_forms() {
        // Two lengths: offsets only.
        let v = shadow_of("box-shadow: 2px 3px", "box-shadow").expect("parses");
        let s = &layers(&v)[0];
        assert_eq!(layers(&v).len(), 1);
        assert_eq!(len_px(&s.offset_x), 2.0);
        assert_eq!(len_px(&s.offset_y), 3.0);
        assert_eq!(len_px(&s.blur), 0.0);
        assert_eq!(len_px(&s.spread), 0.0);
        assert_eq!(s.color, CssValue::Keyword("currentcolor".into()));
        assert!(!s.inset);
        // Full form: inset, four lengths, color.
        let v = shadow_of("box-shadow: inset 2px 3px 4px 5px red", "box-shadow").expect("parses");
        let s = &layers(&v)[0];
        assert!(s.inset);
        assert_eq!(len_px(&s.offset_x), 2.0);
        assert_eq!(len_px(&s.offset_y), 3.0);
        assert_eq!(len_px(&s.blur), 4.0);
        assert_eq!(len_px(&s.spread), 5.0);
        assert!(matches!(s.color, CssValue::Color(_)));
        // Unitless zero is a valid `<length>`.
        let v = shadow_of("box-shadow: 0 0", "box-shadow").expect("parses");
        assert_eq!(layers(&v).len(), 1);
        // Negative offsets and spread are fine.
        let v = shadow_of("box-shadow: -2px -3px 0 -1px red", "box-shadow").expect("parses");
        let s = &layers(&v)[0];
        assert_eq!(len_px(&s.offset_x), -2.0);
        assert_eq!(len_px(&s.spread), -1.0);
    }

    #[test]
    fn box_shadow_component_order_is_free() {
        // `inset? && <length>{2,4} && <color>?`: color and `inset` may
        // appear anywhere relative to the lengths.
        for css in [
            "box-shadow: red 2px 3px",
            "box-shadow: 2px red 3px",
            "box-shadow: 2px 3px red",
            "box-shadow: inset red 2px 3px",
            "box-shadow: 2px 3px red inset",
            "box-shadow: 2px 3px 4px inset red",
        ] {
            let v = shadow_of(css, "box-shadow").expect("{css} parses");
            let s = &layers(&v)[0];
            assert_eq!(len_px(&s.offset_x), 2.0, "{css}");
            assert_eq!(len_px(&s.offset_y), 3.0, "{css}");
            assert!(matches!(s.color, CssValue::Color(_)), "{css}");
            assert_eq!(s.inset, css.contains("inset"), "{css}");
        }
    }

    #[test]
    fn box_shadow_multiple_layers() {
        let v = shadow_of(
            "box-shadow: 1px 2px red, inset 3px 4px 5px blue",
            "box-shadow",
        )
        .expect("parses");
        let ls = layers(&v);
        assert_eq!(ls.len(), 2);
        assert!(!ls[0].inset);
        assert_eq!(len_px(&ls[0].offset_x), 1.0);
        assert!(ls[1].inset);
        assert_eq!(len_px(&ls[1].blur), 5.0);
        // Three layers, mixed lengths.
        let v = shadow_of(
            "box-shadow: 1px 1px, 2px 2px 2px, 3px 3px 3px 3px",
            "box-shadow",
        )
        .expect("parses");
        assert_eq!(layers(&v).len(), 3);
    }

    #[test]
    fn box_shadow_invalid_declarations_drop() {
        for bad in [
            // Too few / too many lengths.
            "box-shadow: 2px",
            "box-shadow: 2px 3px 4px 5px 6px red",
            // Negative blur radius (spec: "Negative values are not allowed").
            "box-shadow: 2px 3px -4px red",
            "box-shadow: 2px 3px -0.5px",
            // Two colors / two insets.
            "box-shadow: 2px 3px red blue",
            "box-shadow: inset inset 2px 3px",
            // Percentages are not `<length>`.
            "box-shadow: 10% 2px",
            "box-shadow: 2px 3px 10%",
            // Unknown component.
            "box-shadow: 2px 3px solid",
            "box-shadow: outset 2px 3px",
            // Bad comma structure.
            "box-shadow: 2px 3px,",
            "box-shadow: , 2px 3px",
            "box-shadow: 2px 3px,, 4px 5px",
            "box-shadow: 2px 3px, red",
            // `none` must stand alone.
            "box-shadow: none, 2px 3px",
            "box-shadow: 2px 3px none",
        ] {
            assert_eq!(shadow_of(bad, "box-shadow"), None, "{bad} should drop");
        }
        // A lone color or `inset` parses to a bare keyword (for the
        // cascade's wide-keyword path); the cascade's grammar check
        // drops it since it is neither a wide keyword nor a shadow.
        assert_eq!(
            shadow_of("box-shadow: red", "box-shadow"),
            Some(CssValue::Keyword("red".into()))
        );
        // Empty value drops too.
        assert_eq!(parse_declarations("box-shadow:").len(), 0);
    }

    #[test]
    fn text_shadow_grammar() {
        // Two or three lengths plus optional color; no `inset`, no spread.
        let v = shadow_of("text-shadow: 1px 2px", "text-shadow").expect("parses");
        let s = &layers(&v)[0];
        assert!(!s.inset);
        assert_eq!(len_px(&s.spread), 0.0);
        let v = shadow_of("text-shadow: 1px 2px 3px rgba(0, 0, 0, 0.5)", "text-shadow")
            .expect("parses");
        assert_eq!(len_px(&layers(&v)[0].blur), 3.0);
        let v = shadow_of("text-shadow: 1px 1px red, 2px 2px blue", "text-shadow").expect("parses");
        assert_eq!(layers(&v).len(), 2);
        for bad in [
            "text-shadow: inset 1px 2px",
            "text-shadow: 1px 2px inset",
            "text-shadow: 1px 2px 3px 4px red",
            "text-shadow: 1px",
            "text-shadow: 1px 2px -3px red",
        ] {
            assert_eq!(shadow_of(bad, "text-shadow"), None, "{bad} should drop");
        }
    }

    #[test]
    fn shadow_wide_keyword_and_var() {
        // CSS-wide keywords pass through for the cascade.
        assert_eq!(
            shadow_of("box-shadow: inherit", "box-shadow"),
            Some(CssValue::Keyword("inherit".into()))
        );
        // A bare `inset` is not a wide keyword: it parses to a keyword
        // the cascade's grammar check rejects.
        assert_eq!(
            shadow_of("box-shadow: inset", "box-shadow"),
            Some(CssValue::Keyword("inset".into()))
        );
        // `var()` defers: the `Var` node survives for the cascade.
        let v = shadow_of("box-shadow: var(--s)", "box-shadow").expect("parses");
        assert!(matches!(v, CssValue::Var(_, _)), "got {v:?}");
        let v = shadow_of("box-shadow: 2px var(--y) red", "box-shadow").expect("parses");
        assert!(matches!(v, CssValue::List(_)), "got {v:?}");
        // `calc()` lengths parse to math.
        let v = shadow_of("box-shadow: calc(1px + 2px) 0 red", "box-shadow").expect("parses");
        assert!(
            matches!(layers(&v)[0].offset_x, CssValue::Math(_)),
            "got {:?}",
            layers(&v)[0].offset_x
        );
        // Percentage `calc()` is rejected (no percentage base).
        assert_eq!(
            shadow_of("box-shadow: calc(10% + 1px) 0 red", "box-shadow"),
            None
        );
    }

    #[test]
    fn backdrop_filter_blur_parses() {
        let v = shadow_of("backdrop-filter: blur(5px)", "backdrop-filter");
        assert!(v.is_some());
        // `filter` uses the same grammar.
        let v2 = shadow_of("filter: blur(5px)", "filter");
        assert!(v2.is_some());
    }

    #[test]
    fn filter_grayscale_parses() {
        let v = shadow_of("filter: grayscale(1)", "filter");
        assert!(v.is_some());
    }

    fn clip_path_of(css: &str) -> ClipPath {
        let decls = parse_declarations(css);
        assert_eq!(
            decls.len(),
            1,
            "expected exactly one declaration from {css:?}"
        );
        assert_eq!(decls[0].property, "clip-path");
        match &decls[0].value {
            CssValue::ClipPath(cp) => cp.clone(),
            other => panic!("expected ClipPath, got {other:?} from {css:?}"),
        }
    }

    fn px(v: f32) -> CssValue {
        CssValue::Length(v, LengthUnit::Px)
    }

    fn pct(v: f32) -> CssValue {
        CssValue::Percentage(v)
    }

    #[test]
    fn clip_path_inset_two_values() {
        assert_eq!(
            clip_path_of("clip-path: inset(10px 20px)"),
            ClipPath::Shape(Box::new(BasicShape::Inset {
                offsets: [px(10.0), px(20.0), px(10.0), px(20.0)],
                radii: None,
            }))
        );
    }

    #[test]
    fn clip_path_inset_four_values() {
        assert_eq!(
            clip_path_of("clip-path: inset(10px 20px 30px 40px)"),
            ClipPath::Shape(Box::new(BasicShape::Inset {
                offsets: [px(10.0), px(20.0), px(30.0), px(40.0)],
                radii: None,
            }))
        );
    }

    #[test]
    fn clip_path_inset_round() {
        assert_eq!(
            clip_path_of("clip-path: inset(10px round 5px)"),
            ClipPath::Shape(Box::new(BasicShape::Inset {
                offsets: [px(10.0), px(10.0), px(10.0), px(10.0)],
                radii: Some([px(5.0), px(5.0), px(5.0), px(5.0)]),
            }))
        );
    }

    #[test]
    fn clip_path_circle_at() {
        assert_eq!(
            clip_path_of("clip-path: circle(50% at 20% 30%)"),
            ClipPath::Shape(Box::new(BasicShape::Circle {
                radius: ShapeRadius::Length(pct(50.0)),
                cx: pct(20.0),
                cy: pct(30.0),
            }))
        );
    }

    #[test]
    fn clip_path_ellipse_at() {
        assert_eq!(
            clip_path_of("clip-path: ellipse(40% 30% at 50% 50%)"),
            ClipPath::Shape(Box::new(BasicShape::Ellipse {
                rx: ShapeRadius::Length(pct(40.0)),
                ry: ShapeRadius::Length(pct(30.0)),
                cx: pct(50.0),
                cy: pct(50.0),
            }))
        );
    }

    #[test]
    fn clip_path_polygon() {
        assert_eq!(
            clip_path_of("clip-path: polygon(0 0, 100% 0, 50% 100%)"),
            ClipPath::Shape(Box::new(BasicShape::Polygon {
                points: vec![
                    (CssValue::Number(0.0), CssValue::Number(0.0)),
                    (pct(100.0), CssValue::Number(0.0)),
                    (pct(50.0), pct(100.0)),
                ],
            }))
        );
    }

    #[test]
    fn clip_path_shapes_parse() {
        // Clip-path shapes are accepted by the declaration parser
        // (detailed shape assertions live in the per-shape tests above).
        for css in [
            "clip-path: inset(10px 20px)",
            "clip-path: circle(50%)",
            "clip-path: ellipse(50% 25%)",
            "clip-path: polygon(0 0, 100% 0)",
            "clip-path: none",
        ] {
            // Must not panic; well-formed shapes produce a declaration.
            let _ = parse_declarations(css);
        }
    }
}
