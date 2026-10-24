//! From-scratch regular expression engine (std-only, no dependencies).
//!
//! Pipeline: pattern text -> AST -> bytecode -> backtracking VM.
//!
//! # Matching model
//!
//! Input is matched as UTF-16 code units (`Vec<u16>`), which is also the unit
//! of the capture indices returned by [`CompiledRegex::exec`] and of its
//! `start` parameter. This mirrors JavaScript string indexing.
//!
//! - Without the `u`/`v` flag the pattern is parsed as a sequence of UTF-16
//!   code units: an astral pattern character behaves as its two surrogate
//!   units (e.g. `/😀{2}/` means `lead trail{2}`).
//! - With `u`/`v`, astral pattern characters are parsed as single code-point
//!   atoms and match a surrogate pair atomically; a lone surrogate in the
//!   input still matches as one unit (`.` and `[^]` match it).
//!
//! # Flags
//!
//! `d` accepted (captures already carry indices; the flag is a no-op here).
//! `g` accepted (global `lastIndex` iteration is the caller's job; `exec`
//! searches forward from `start`). `i` case-insensitive (ASCII folding plus
//! first-char `char::to_lowercase` for the rest; multi-char folds such as
//! `İ` compare on their first folded char only). `m` makes `^`/`$` match at
//! line boundaries. `s` lets `.` match line terminators. `u` enables Unicode
//! mode (stricter escapes, astral atoms, `\p{...}`). `v` behaves like `u`
//! except nested character classes (`[a[b]]`) and `&&` intersections are
//! rejected with `ParseError` (unsupported). `y` (sticky) makes `exec` try
//! only the exact `start` position. Combining `u` and `v` is a `ParseError`.
//! Unknown or duplicated flags are a `ParseError`.
//!
//! # `\p{...}` curated subset (requires `u` or `v`)
//!
//! General categories: `L`/`Letter`, `Lu`/`Uppercase_Letter`,
//! `Ll`/`Lowercase_Letter`, `Lt` (mapped to uppercase), `Lm`/`Lo` (mapped to
//! alphabetic), `N`/`Number`, `Nd`/`Decimal_Number`, `Nl`, `No` (all mapped to
//! `char::is_numeric`, which also covers `Nl`/`No`), `Z`/`Zs`/`Zl`/`Zp`
//! (mapped to `char::is_whitespace`), `Cc` (0x00-0x1F, 0x7F-0x9F),
//! `Co` (0xE000-0xF8FF), `Cn` (never matches a `char`; matches lone
//! surrogates only via negation, e.g. `\P{Cn}`). Also with `gc=` /
//! `General_Category=` prefixes. Properties: `ASCII`, `Alphabetic`
//! (`char::is_alphabetic`), `White_Space` (`char::is_whitespace`),
//! `Uppercase`/`Lowercase` (`char::is_uppercase`/`is_lowercase`), `Any`,
//! `Assigned` (any `char`; surrogates are never `char`s, so `\P{Assigned}`
//! matches lone surrogates). Scripts via `Script=`/`sc=`: `Latin`, `Greek`,
//! `Cyrillic`, `Armenian`, `Hebrew`, `Arabic`, `Han`, `Hiragana`, `Katakana`,
//! `Hangul` — matched by block ranges (documented approximations, e.g. Latin
//! covers ASCII letters, 0xC0-0xFF letters, 0x100-0x24F and 0x1E00-0x1EFF).
//! Everything else (`P*`, `S*`, `M*`, `Cf`, `Cs`, ...) is a `ParseError`.
//!
//! # Unsupported
//!
//! Lookbehind (`(?<=...)` / `(?<!...)`) is a
//! `ParseError("lookbehind unsupported")`. Possessive quantifiers are a
//! `ParseError`. Backreferences to groups that never participate match the
//! empty string (per spec).
//!
//! # Backtracking budget
//!
//! The VM counts executed instructions; exceeding [`MAX_STEPS`] (1,000,000)
//! per `exec` call aborts the match. `exec` then returns `None` and records
//! [`RegexError::BacktrackLimit`], retrievable via
//! [`CompiledRegex::take_error`]. The budget is shared across all start
//! positions tried by one `exec` call.

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;

use crate::wtf8::Wtf8;

/// Upper bound on VM instructions executed by a single [`CompiledRegex::exec`]
/// call. Catastrophic patterns (e.g. `(a+)+$` on a long non-matching input)
/// hit this instead of hanging.
pub const MAX_STEPS: u64 = 1_000_000;

/// Error type for regex compilation and matching.
#[derive(Debug)]
pub enum RegexError {
    /// The pattern (or flags) could not be parsed.
    ParseError(String),
    /// The backtracking budget ([`MAX_STEPS`]) was exhausted during `exec`.
    /// Reported out-of-band via [`CompiledRegex::take_error`]; `exec` itself
    /// returns `None` in that case.
    BacktrackLimit,
}

impl fmt::Display for RegexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegexError::ParseError(msg) => write!(f, "invalid regular expression: {msg}"),
            RegexError::BacktrackLimit => write!(f, "regex backtracking limit exceeded"),
        }
    }
}

impl std::error::Error for RegexError {}

/// A captured group span. Indices are UTF-16 code-unit offsets into the
/// input (JS string-index semantics).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capture {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct Flags {
    ignore_case: bool,
    /// Accepted and validated; global `lastIndex` iteration is the caller's job.
    #[allow(dead_code)]
    global: bool,
    multiline: bool,
    dot_all: bool,
    unicode: bool,
    unicode_sets: bool,
    sticky: bool,
    /// Accepted and validated; captures already carry indices.
    #[allow(dead_code)]
    has_indices: bool,
}

impl Flags {
    fn unicode_mode(self) -> bool {
        self.unicode || self.unicode_sets
    }
}

fn parse_flags(flags: &str) -> Result<Flags, RegexError> {
    let mut f = Flags::default();
    for c in flags.chars() {
        let slot = match c {
            'd' => &mut f.has_indices,
            'g' => &mut f.global,
            'i' => &mut f.ignore_case,
            'm' => &mut f.multiline,
            's' => &mut f.dot_all,
            'u' => &mut f.unicode,
            'v' => &mut f.unicode_sets,
            'y' => &mut f.sticky,
            _ => return Err(RegexError::ParseError(format!("invalid flag '{c}'"))),
        };
        if *slot {
            return Err(RegexError::ParseError(format!("duplicate flag '{c}'")));
        }
        *slot = true;
    }
    if f.unicode && f.unicode_sets {
        return Err(RegexError::ParseError(
            "the 'u' and 'v' flags cannot be combined".to_string(),
        ));
    }
    Ok(f)
}

/// Curated `\p{...}` property set. See module docs for exact semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Prop {
    Ascii,
    Any,
    Assigned,
    Alphabetic,
    WhiteSpace,
    Upper,
    Lower,
    Numeric,
    Cc,
    Co,
    Cn,
    Latin,
    Greek,
    Cyrillic,
    Armenian,
    Hebrew,
    Arabic,
    Han,
    Hiragana,
    Katakana,
    Hangul,
}

fn in_ranges(v: u32, ranges: &[(u32, u32)]) -> bool {
    ranges.iter().any(|&(lo, hi)| v >= lo && v <= hi)
}

fn prop_test(p: Prop, c: char) -> bool {
    let v = c as u32;
    match p {
        Prop::Ascii => v <= 0x7F,
        Prop::Any => true,
        // `char` can never be a surrogate; every char counts as assigned
        // (noncharacters included, documented approximation).
        Prop::Assigned => true,
        Prop::Alphabetic => c.is_alphabetic(),
        Prop::WhiteSpace => c.is_whitespace(),
        Prop::Upper => c.is_uppercase(),
        Prop::Lower => c.is_lowercase(),
        // Approximation: `is_numeric` covers Nd, Nl and No.
        Prop::Numeric => c.is_numeric(),
        Prop::Cc => in_ranges(v, &[(0x00, 0x1F), (0x7F, 0x9F)]),
        Prop::Co => (0xE000..=0xF8FF).contains(&v),
        // No `char` is unassigned.
        Prop::Cn => false,
        Prop::Latin => {
            c.is_ascii_alphabetic()
                || in_ranges(
                    v,
                    &[
                        (0xC0, 0xD6),
                        (0xD8, 0xF6),
                        (0xF8, 0xFF),
                        (0x100, 0x24F),
                        (0x1E00, 0x1EFF),
                    ],
                )
        }
        Prop::Greek => in_ranges(v, &[(0x370, 0x3FF), (0x1F00, 0x1FFF)]),
        Prop::Cyrillic => in_ranges(
            v,
            &[
                (0x400, 0x4FF),
                (0x500, 0x52F),
                (0x2DE0, 0x2DFF),
                (0xA640, 0xA69F),
            ],
        ),
        Prop::Armenian => (0x530..=0x58F).contains(&v),
        Prop::Hebrew => (0x590..=0x5FF).contains(&v),
        Prop::Arabic => in_ranges(v, &[(0x600, 0x6FF), (0x750, 0x77F), (0x8A0, 0x8FF)]),
        Prop::Han => in_ranges(
            v,
            &[
                (0x3400, 0x4DBF),
                (0x4E00, 0x9FFF),
                (0x20000, 0x2A6DF),
                (0x2A600, 0x2CEAF),
            ],
        ),
        Prop::Hiragana => (0x3040..=0x309F).contains(&v),
        Prop::Katakana => in_ranges(v, &[(0x30A0, 0x30FF), (0x31F0, 0x31FF)]),
        Prop::Hangul => in_ranges(
            v,
            &[
                (0x1100, 0x11FF),
                (0x3130, 0x318F),
                (0xA960, 0xA97F),
                (0xAC00, 0xD7AF),
                (0xD7B0, 0xD7FF),
            ],
        ),
    }
}

/// Map a `\p{...}` name (after stripping any `Script=`/`sc=`/`gc=`
/// prefixes) to a curated [`Prop`].
fn lookup_prop(name: &str) -> Result<Prop, RegexError> {
    let unsupported = |n: &str| {
        RegexError::ParseError(format!("unsupported Unicode property '{n}' in \\p{{...}}"))
    };
    for prefix in ["Script=", "sc="] {
        if let Some(script) = name.strip_prefix(prefix) {
            return match script {
                "Latin" => Ok(Prop::Latin),
                "Greek" => Ok(Prop::Greek),
                "Cyrillic" => Ok(Prop::Cyrillic),
                "Armenian" => Ok(Prop::Armenian),
                "Hebrew" => Ok(Prop::Hebrew),
                "Arabic" => Ok(Prop::Arabic),
                "Han" => Ok(Prop::Han),
                "Hiragana" => Ok(Prop::Hiragana),
                "Katakana" => Ok(Prop::Katakana),
                "Hangul" => Ok(Prop::Hangul),
                _ => Err(unsupported(name)),
            };
        }
    }
    for prefix in ["General_Category=", "gc="] {
        if let Some(cat) = name.strip_prefix(prefix) {
            return lookup_prop(cat);
        }
    }
    match name {
        "ASCII" => Ok(Prop::Ascii),
        "Any" => Ok(Prop::Any),
        "Assigned" => Ok(Prop::Assigned),
        "Alphabetic" => Ok(Prop::Alphabetic),
        "White_Space" => Ok(Prop::WhiteSpace),
        "Uppercase" => Ok(Prop::Upper),
        "Lowercase" => Ok(Prop::Lower),
        "L" | "Letter" => Ok(Prop::Alphabetic),
        "Lu" | "Uppercase_Letter" => Ok(Prop::Upper),
        "Ll" | "Lowercase_Letter" => Ok(Prop::Lower),
        // Titlecase has no std query; uppercase is the closest documented
        // approximation.
        "Lt" | "Titlecase_Letter" => Ok(Prop::Upper),
        "Lm" | "Modifier_Letter" | "Lo" | "Other_Letter" => Ok(Prop::Alphabetic),
        "N" | "Number" | "Nd" | "Decimal_Number" | "Nl" | "Letter_Number" | "No"
        | "Other_Number" => Ok(Prop::Numeric),
        "Z"
        | "Separator"
        | "Zs"
        | "Space_Separator"
        | "Zl"
        | "Line_Separator"
        | "Zp"
        | "Paragraph_Separator" => Ok(Prop::WhiteSpace),
        "Cc" | "Control" => Ok(Prop::Cc),
        "Co" | "Private_Use" => Ok(Prop::Co),
        "Cn" | "Unassigned" => Ok(Prop::Cn),
        _ => Err(unsupported(name)),
    }
}

fn is_line_terminator(v: u32) -> bool {
    matches!(v, 0x0A | 0x0D | 0x2028 | 0x2029)
}

fn is_word_char(v: u32) -> bool {
    matches!(v, 0x30..=0x39 | 0x41..=0x5A | 0x5F | 0x61..=0x7A)
}

/// Case fold used for the `i` flag: ASCII fast path, otherwise the first
/// char of `char::to_lowercase` (documented approximation for multi-char
/// folds such as U+0130).
fn fold_case(v: u32) -> u32 {
    if (0x41..=0x5A).contains(&v) {
        return v + 32;
    }
    char::from_u32(v)
        .and_then(|c| c.to_lowercase().next())
        .map(|c| c as u32)
        .unwrap_or(v)
}

fn is_hex_digit(v: u32) -> bool {
    matches!(v, 0x30..=0x39 | 0x41..=0x46 | 0x61..=0x66)
}

fn hex_val(v: u32) -> u32 {
    match v {
        0x30..=0x39 => v - 0x30,
        0x41..=0x46 => v - 0x41 + 10,
        _ => v - 0x61 + 10,
    }
}

fn is_ascii_alnum(v: u32) -> bool {
    matches!(v, 0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A)
}

fn is_ascii_alpha(v: u32) -> bool {
    matches!(v, 0x41..=0x5A | 0x61..=0x7A)
}

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

/// Character class: code-point ranges plus curated `\p{...}` predicates.
#[derive(Debug, Clone)]
struct Class {
    ranges: Vec<(u32, u32)>,
    negated: bool,
    props: Vec<(Prop, bool)>, // (property, negated)
}

#[derive(Debug, Clone)]
enum Ast {
    Empty,
    Lit(u32),
    Dot,
    Class(Class),
    Start,
    End,
    WordBoundary(bool), // true = \b, false = \B
    Seq(Vec<Ast>),
    Alt(Vec<Ast>),
    Group {
        idx: usize,
        body: Box<Ast>,
    },
    Quant {
        body: Box<Ast>,
        min: u32,
        max: u32,
        greedy: bool,
    },
    BackRef(usize),
    NamedBackRef(String),
    DecimalEscape(String), // \[1-9][0-9]* resolved post-parse (backref vs octal)
    LookAhead {
        neg: bool,
        body: Box<Ast>,
    },
}

fn class_ranges(ranges: &[(u32, u32)]) -> Class {
    Class {
        ranges: ranges.to_vec(),
        negated: false,
        props: vec![],
    }
}

/// Complement of sorted ranges within 0..=0x10FFFF.
fn complement_ranges(ranges: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut next = 0u32;
    for &(lo, hi) in ranges {
        if lo > next {
            out.push((next, lo - 1));
        }
        next = hi.saturating_add(1);
        if next == 0 {
            break; // hi was u32::MAX
        }
    }
    if next <= 0x10FFFF {
        out.push((next, 0x10FFFF));
    }
    out
}

fn digit_class() -> Class {
    class_ranges(&[(0x30, 0x39)])
}
fn not_digit_class() -> Class {
    class_ranges(&complement_ranges(&[(0x30, 0x39)]))
}
fn space_class() -> Class {
    class_ranges(&[
        (0x09, 0x0D),
        (0x20, 0x20),
        (0xA0, 0xA0),
        (0x1680, 0x1680),
        (0x2000, 0x200A),
        (0x2028, 0x2029),
        (0x202F, 0x202F),
        (0x205F, 0x205F),
        (0x3000, 0x3000),
        (0xFEFF, 0xFEFF),
    ])
}
fn not_space_class() -> Class {
    class_ranges(&complement_ranges(&[
        (0x09, 0x0D),
        (0x20, 0x20),
        (0xA0, 0xA0),
        (0x1680, 0x1680),
        (0x2000, 0x200A),
        (0x2028, 0x2029),
        (0x202F, 0x202F),
        (0x205F, 0x205F),
        (0x3000, 0x3000),
        (0xFEFF, 0xFEFF),
    ]))
}
fn word_class() -> Class {
    class_ranges(&[(0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A)])
}
fn not_word_class() -> Class {
    class_ranges(&complement_ranges(&[
        (0x30, 0x39),
        (0x41, 0x5A),
        (0x5F, 0x5F),
        (0x61, 0x7A),
    ]))
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Atom produced by a character-class escape.
#[derive(Debug)]
enum ClassAtom {
    Unit(u32),
    Units(Vec<u32>),
    Ranges(Vec<(u32, u32)>),
    Prop(Prop, bool),
}

struct Parser {
    units: Vec<u32>,
    pos: usize,
    flags: Flags,
    group_count: usize,
    group_names: Vec<Option<String>>, // indexed by group number (0 unused)
    named: HashMap<String, usize>,    // first group index per name
}

impl Parser {
    fn peek(&self) -> Option<u32> {
        self.units.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<u32> {
        let v = self.units.get(self.pos).copied()?;
        self.pos += 1;
        Some(v)
    }

    fn expect(&mut self, v: u32, what: &str) -> Result<(), RegexError> {
        if self.next() == Some(v) {
            Ok(())
        } else {
            Err(RegexError::ParseError(format!("expected {what}")))
        }
    }

    fn unicode_mode(&self) -> bool {
        self.flags.unicode_mode()
    }

    fn parse(&mut self) -> Result<Ast, RegexError> {
        let ast = self.parse_disjunction()?;
        if self.pos != self.units.len() {
            return Err(RegexError::ParseError("unmatched ')'".to_string()));
        }
        Ok(ast)
    }

    fn parse_disjunction(&mut self) -> Result<Ast, RegexError> {
        let mut branches = vec![self.parse_alternative()?];
        while self.peek() == Some(0x7C) {
            // '|'
            self.next();
            branches.push(self.parse_alternative()?);
        }
        if branches.len() == 1 {
            Ok(branches.pop().unwrap())
        } else {
            Ok(Ast::Alt(branches))
        }
    }

    fn parse_alternative(&mut self) -> Result<Ast, RegexError> {
        let mut terms = Vec::new();
        loop {
            match self.peek() {
                None | Some(0x7C) | Some(0x29) => break, // end, '|', ')'
                _ => terms.push(self.parse_term()?),
            }
        }
        match terms.len() {
            0 => Ok(Ast::Empty),
            1 => Ok(terms.pop().unwrap()),
            _ => Ok(Ast::Seq(terms)),
        }
    }

    fn parse_term(&mut self) -> Result<Ast, RegexError> {
        let atom = self.parse_atom()?;
        if let Some((min, max, greedy)) = self.try_parse_quantifier()? {
            Ok(Ast::Quant {
                body: Box::new(atom),
                min,
                max,
                greedy,
            })
        } else {
            Ok(atom)
        }
    }

    /// If the upcoming text is a valid quantifier suffix, consume it.
    /// Invalid `{...}` is left unconsumed so it parses as literals (both
    /// modes, matching JS); `{n,m}` with n > m is a SyntaxError per spec.
    fn try_parse_quantifier(&mut self) -> Result<Option<(u32, u32, bool)>, RegexError> {
        let (min, max) = match self.peek() {
            Some(0x2A) => {
                self.next();
                (0, u32::MAX)
            }
            Some(0x2B) => {
                self.next();
                (1, u32::MAX)
            }
            Some(0x3F) => {
                self.next();
                (0, 1)
            }
            Some(0x7B) => {
                // '{': attempt braced quantifier, else leave for literals.
                let save = self.pos;
                self.next();
                let mut n: u64 = 0;
                let mut digits = 0;
                while let Some(v) = self.peek() {
                    if matches!(v, 0x30..=0x39) && digits < 12 {
                        n = n * 10 + (v - 0x30) as u64;
                        digits += 1;
                        self.next();
                    } else {
                        break;
                    }
                }
                // Consume any further digits without changing the value
                // (saturate).
                while matches!(self.peek(), Some(0x30..=0x39)) {
                    self.next();
                    n = u64::MAX;
                }
                if digits == 0 {
                    self.pos = save;
                    return Ok(None);
                }
                let mut m = n;
                if self.peek() == Some(0x2C) {
                    // ','
                    self.next();
                    if matches!(self.peek(), Some(0x30..=0x39)) {
                        let mut m2: u64 = 0;
                        while let Some(v) = self.peek() {
                            if matches!(v, 0x30..=0x39) {
                                m2 = m2.saturating_mul(10).saturating_add((v - 0x30) as u64);
                                self.next();
                            } else {
                                break;
                            }
                        }
                        m = m2;
                    } else {
                        m = u64::MAX; // open {n,}
                    }
                }
                if self.peek() != Some(0x7D) {
                    // Not a valid quantifier: literal '{...'.
                    self.pos = save;
                    return Ok(None);
                }
                if m < n {
                    return Err(RegexError::ParseError(
                        "numbers out of order in {} quantifier".to_string(),
                    ));
                }
                self.next(); // '}'
                (n.min(u32::MAX as u64) as u32, m.min(u32::MAX as u64) as u32)
            }
            _ => return Ok(None),
        };
        let greedy = if self.peek() == Some(0x3F) {
            self.next();
            false
        } else {
            true
        };
        Ok(Some((min, max, greedy)))
    }

    fn parse_atom(&mut self) -> Result<Ast, RegexError> {
        match self.next() {
            Some(0x5E) => Ok(Ast::Start), // ^
            Some(0x24) => Ok(Ast::End),   // $
            Some(0x2E) => Ok(Ast::Dot),   // .
            Some(0x2A) | Some(0x2B) | Some(0x3F) => {
                Err(RegexError::ParseError("lone quantifier".to_string()))
            }
            Some(0x28) => self.parse_group(), // (
            Some(0x5B) => self.parse_class(), // [
            Some(0x29) => Err(RegexError::ParseError("unmatched ')'".to_string())),
            Some(0x5C) => self.parse_escape(), // backslash
            Some(v) => Ok(Ast::Lit(v)),        // everything else, incl. { } ] |
            None => Err(RegexError::ParseError(
                "unexpected end of pattern".to_string(),
            )),
        }
    }

    fn parse_group(&mut self) -> Result<Ast, RegexError> {
        // '(' already consumed.
        if self.peek() == Some(0x3F) {
            self.next(); // '?'
            match self.next() {
                Some(0x3A) => {
                    // (?:...)
                    let body = self.parse_disjunction()?;
                    self.expect(0x29, "')'")?;
                    Ok(body)
                }
                Some(0x3D) => {
                    // (?=...)
                    let body = self.parse_disjunction()?;
                    self.expect(0x29, "')'")?;
                    Ok(Ast::LookAhead {
                        neg: false,
                        body: Box::new(body),
                    })
                }
                Some(0x21) => {
                    // (?!...)
                    let body = self.parse_disjunction()?;
                    self.expect(0x29, "')'")?;
                    Ok(Ast::LookAhead {
                        neg: true,
                        body: Box::new(body),
                    })
                }
                Some(0x3C) => {
                    // (?<name>...) or (?<=...) / (?<!...)
                    match self.peek() {
                        Some(0x3D) | Some(0x21) => {
                            return Err(RegexError::ParseError(
                                "lookbehind unsupported".to_string(),
                            ));
                        }
                        _ => {}
                    }
                    let name = self.parse_group_name()?;
                    self.expect(0x3E, "'>'")?; // '>'
                    let idx = self.group_count + 1;
                    self.group_count += 1;
                    self.group_names.push(Some(name.clone()));
                    self.named.entry(name).or_insert(idx);
                    let body = self.parse_disjunction()?;
                    self.expect(0x29, "')'")?;
                    Ok(Ast::Group {
                        idx,
                        body: Box::new(body),
                    })
                }
                _ => Err(RegexError::ParseError("invalid group".to_string())),
            }
        } else {
            let idx = self.group_count + 1;
            self.group_count += 1;
            self.group_names.push(None);
            let body = self.parse_disjunction()?;
            self.expect(0x29, "')'")?;
            Ok(Ast::Group {
                idx,
                body: Box::new(body),
            })
        }
    }

    fn parse_group_name(&mut self) -> Result<String, RegexError> {
        let mut name = String::new();
        while let Some(v) = self.peek() {
            if v == 0x3E {
                break; // '>'
            }
            let c = char::from_u32(v).ok_or_else(|| {
                RegexError::ParseError("invalid character in group name".to_string())
            })?;
            if name.is_empty() {
                if !(c == '_' || c == '$' || c.is_alphabetic()) {
                    return Err(RegexError::ParseError("invalid group name".to_string()));
                }
            } else if !(c == '_' || c == '$' || c.is_alphanumeric()) {
                return Err(RegexError::ParseError("invalid group name".to_string()));
            }
            name.push(c);
            self.next();
        }
        if name.is_empty() {
            return Err(RegexError::ParseError("empty group name".to_string()));
        }
        Ok(name)
    }

    // -- escapes ---------------------------------------------------------

    /// Parse `\xHH`. In unicode mode both digits are required; otherwise a
    /// bare `\x` degrades to a literal 'x' (documented Annex-B leniency).
    fn parse_hex_escape(&mut self) -> Result<Ast, RegexError> {
        let a = self.peek();
        let b = self.units.get(self.pos + 1).copied();
        if a.is_some_and(is_hex_digit) && b.is_some_and(is_hex_digit) {
            let hi = self.next().unwrap();
            let lo = self.next().unwrap();
            Ok(Ast::Lit(hex_val(hi) * 16 + hex_val(lo)))
        } else if self.unicode_mode() {
            Err(RegexError::ParseError(
                "expected two hex digits after \\x".to_string(),
            ))
        } else {
            Ok(Ast::Lit(0x78)) // 'x'
        }
    }

    /// Parse `\uHHHH` or `\u{H+}`.
    fn parse_unicode_escape(&mut self) -> Result<Ast, RegexError> {
        if self.peek() == Some(0x7B) {
            // \u{...}: only in unicode mode; otherwise literal 'u'.
            if !self.unicode_mode() {
                return Ok(Ast::Lit(0x75)); // 'u'
            }
            self.next(); // '{'
            let mut v: u32 = 0;
            let mut digits = 0;
            loop {
                match self.next() {
                    Some(d) if is_hex_digit(d) => {
                        v = v * 16 + hex_val(d);
                        digits += 1;
                        if v > 0x10FFFF {
                            return Err(RegexError::ParseError(
                                "unicode escape out of range".to_string(),
                            ));
                        }
                    }
                    Some(0x7D) => break, // '}'
                    _ => {
                        return Err(RegexError::ParseError(
                            "invalid \\u{...} escape".to_string(),
                        ));
                    }
                }
            }
            if digits == 0 {
                return Err(RegexError::ParseError(
                    "invalid \\u{...} escape".to_string(),
                ));
            }
            Ok(Ast::Lit(v))
        } else {
            // Exactly four hex digits required in unicode mode; Annex-B
            // leniency otherwise (literal 'u', rewind what we consumed).
            let start = self.pos;
            let mut v: u32 = 0;
            let mut ok = true;
            for _ in 0..4 {
                match self.next() {
                    Some(d) if is_hex_digit(d) => v = v * 16 + hex_val(d),
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                Ok(Ast::Lit(v))
            } else if self.unicode_mode() {
                Err(RegexError::ParseError(
                    "expected four hex digits after \\u".to_string(),
                ))
            } else {
                self.pos = start;
                Ok(Ast::Lit(0x75)) // 'u'
            }
        }
    }

    /// Parse `\cX`.
    fn parse_control_escape(&mut self) -> Result<Ast, RegexError> {
        match self.peek() {
            Some(v) if is_ascii_alpha(v) => {
                self.next();
                let upper = if v >= 0x61 { v - 32 } else { v };
                Ok(Ast::Lit((upper - 0x40) & 0x1F))
            }
            _ if self.unicode_mode() => Err(RegexError::ParseError(
                "expected a letter after \\c".to_string(),
            )),
            _ => Ok(Ast::Lit(0x63)), // 'c'
        }
    }

    /// Parse an octal escape; the first digit was already validated to be
    /// in range. `allow_long` mirrors `\[1-9]` handling (up to 3 digits).
    fn parse_octal(&mut self, first: u32) -> u32 {
        let mut v = first - 0x30;
        for _ in 0..2 {
            match self.peek() {
                Some(d) if matches!(d, 0x30..=0x37) => {
                    v = v * 8 + (d - 0x30);
                    self.next();
                }
                _ => break,
            }
        }
        v
    }

    fn parse_escape(&mut self) -> Result<Ast, RegexError> {
        let e = self
            .next()
            .ok_or_else(|| RegexError::ParseError("trailing \\".to_string()))?;
        match e {
            0x62 => Ok(Ast::WordBoundary(true)),   // \b
            0x42 => Ok(Ast::WordBoundary(false)),  // \B
            0x64 => Ok(Ast::Class(digit_class())), // \d
            0x44 => Ok(Ast::Class(not_digit_class())),
            0x73 => Ok(Ast::Class(space_class())), // \s
            0x53 => Ok(Ast::Class(not_space_class())),
            0x77 => Ok(Ast::Class(word_class())), // \w
            0x57 => Ok(Ast::Class(not_word_class())),
            0x66 => Ok(Ast::Lit(0x0C)), // \f
            0x6E => Ok(Ast::Lit(0x0A)), // \n
            0x72 => Ok(Ast::Lit(0x0D)), // \r
            0x74 => Ok(Ast::Lit(0x09)), // \t
            0x76 => Ok(Ast::Lit(0x0B)), // \v
            0x30 => {
                // \0: NUL unless followed by a digit.
                match self.peek() {
                    Some(0x30..=0x39) => {
                        if self.unicode_mode() {
                            Err(RegexError::ParseError(
                                "octal escapes are not allowed in unicode mode".to_string(),
                            ))
                        } else {
                            Ok(Ast::Lit(self.parse_octal(0x30)))
                        }
                    }
                    _ => Ok(Ast::Lit(0)),
                }
            }
            0x31..=0x39 => {
                // Backreference or (non-unicode) octal; resolved post-parse.
                let mut s = String::new();
                s.push(char::from_u32(e).unwrap());
                while let Some(d) = self.peek() {
                    if matches!(d, 0x30..=0x39) {
                        s.push(char::from_u32(d).unwrap());
                        self.next();
                    } else {
                        break;
                    }
                }
                Ok(Ast::DecimalEscape(s))
            }
            0x78 => self.parse_hex_escape(),     // \x
            0x75 => self.parse_unicode_escape(), // \u
            0x63 => self.parse_control_escape(), // \c
            0x6B => {
                // \k<name>
                if self.peek() == Some(0x3C) {
                    self.next();
                    let name = self.parse_group_name()?;
                    self.expect(0x3E, "'>'")?;
                    Ok(Ast::NamedBackRef(name))
                } else if self.unicode_mode() {
                    Err(RegexError::ParseError("invalid escape \\k".to_string()))
                } else {
                    Ok(Ast::Lit(0x6B)) // 'k'
                }
            }
            0x70 | 0x50 => {
                // \p / \P
                if self.unicode_mode() {
                    let prop = self.parse_prop()?;
                    Ok(Ast::Class(Class {
                        ranges: vec![],
                        negated: false,
                        props: vec![(prop, e == 0x50)],
                    }))
                } else {
                    Ok(Ast::Lit(e)) // identity
                }
            }
            _ => {
                if self.unicode_mode() && is_ascii_alnum(e) {
                    Err(RegexError::ParseError(format!(
                        "invalid escape \\{}",
                        char::from_u32(e).unwrap_or('?')
                    )))
                } else {
                    Ok(Ast::Lit(e)) // identity escape
                }
            }
        }
    }

    fn parse_prop(&mut self) -> Result<Prop, RegexError> {
        self.expect(0x7B, "'{' after \\p")?; // '{'
        let mut name = String::new();
        loop {
            match self.next() {
                Some(0x7D) => break, // '}'
                Some(v) => {
                    let c = char::from_u32(v).ok_or_else(|| {
                        RegexError::ParseError("invalid character in \\p{...}".to_string())
                    })?;
                    name.push(c);
                }
                None => {
                    return Err(RegexError::ParseError("unterminated \\p{...}".to_string()));
                }
            }
        }
        lookup_prop(&name)
    }

    // -- character classes ------------------------------------------------

    fn parse_class(&mut self) -> Result<Ast, RegexError> {
        // '[' already consumed.
        let mut class = Class {
            ranges: vec![],
            negated: false,
            props: vec![],
        };
        if self.peek() == Some(0x5E) {
            self.next();
            class.negated = true;
        }
        // A ']' in first position is a literal.
        let mut first = true;
        // Literal that may turn out to be the start of a range.
        let mut pending: Option<u32> = None;
        let flush = |class: &mut Class, pending: &mut Option<u32>| {
            if let Some(p) = pending.take() {
                class.ranges.push((p, p));
            }
        };
        loop {
            let v = match self.next() {
                Some(v) => v,
                None => {
                    return Err(RegexError::ParseError(
                        "unterminated character class".to_string(),
                    ));
                }
            };
            if v == 0x5D && !first {
                // ']'
                flush(&mut class, &mut pending);
                return Ok(Ast::Class(class));
            }
            first = false;
            if v == 0x5C {
                // Escape inside class.
                match self.parse_class_escape()? {
                    ClassAtom::Unit(u) => {
                        flush(&mut class, &mut pending);
                        pending = Some(u);
                    }
                    ClassAtom::Units(us) => {
                        flush(&mut class, &mut pending);
                        for u in us {
                            class.ranges.push((u, u));
                        }
                    }
                    ClassAtom::Ranges(rs) => {
                        flush(&mut class, &mut pending);
                        class.ranges.extend(rs);
                    }
                    ClassAtom::Prop(p, neg) => {
                        flush(&mut class, &mut pending);
                        class.props.push((p, neg));
                    }
                }
                continue;
            }
            if self.flags.unicode_sets && v == 0x5B {
                return Err(RegexError::ParseError(
                    "nested character classes are not supported in v mode".to_string(),
                ));
            }
            if v == 0x2D {
                // '-': range operator or literal.
                if let Some(p) = pending.take() {
                    match self.peek() {
                        Some(w) if w != 0x5D => {
                            let end = if w == 0x5C {
                                self.next();
                                match self.parse_class_escape()? {
                                    ClassAtom::Unit(u) => u,
                                    _ => {
                                        return Err(RegexError::ParseError(
                                            "class escape cannot be a range endpoint".to_string(),
                                        ));
                                    }
                                }
                            } else {
                                self.next().unwrap()
                            };
                            if end < p {
                                return Err(RegexError::ParseError(
                                    "range out of order in character class".to_string(),
                                ));
                            }
                            class.ranges.push((p, end));
                        }
                        _ => {
                            class.ranges.push((p, p));
                            class.ranges.push((0x2D, 0x2D));
                        }
                    }
                } else {
                    class.ranges.push((0x2D, 0x2D));
                }
                continue;
            }
            flush(&mut class, &mut pending);
            pending = Some(v);
        }
    }

    fn parse_class_escape(&mut self) -> Result<ClassAtom, RegexError> {
        let e = self
            .next()
            .ok_or_else(|| RegexError::ParseError("trailing \\ in character class".to_string()))?;
        match e {
            0x62 => Ok(ClassAtom::Unit(0x08)), // \b = backspace in class
            0x64 => Ok(ClassAtom::Ranges(digit_class().ranges)),
            0x44 => Ok(ClassAtom::Ranges(not_digit_class().ranges)),
            0x73 => Ok(ClassAtom::Ranges(space_class().ranges)),
            0x53 => Ok(ClassAtom::Ranges(not_space_class().ranges)),
            0x77 => Ok(ClassAtom::Ranges(word_class().ranges)),
            0x57 => Ok(ClassAtom::Ranges(not_word_class().ranges)),
            0x66 => Ok(ClassAtom::Unit(0x0C)),
            0x6E => Ok(ClassAtom::Unit(0x0A)),
            0x72 => Ok(ClassAtom::Unit(0x0D)),
            0x74 => Ok(ClassAtom::Unit(0x09)),
            0x76 => Ok(ClassAtom::Unit(0x0B)),
            0x30..=0x37 => {
                // Octal in class (both modes allow \0; \[1-7] octal too).
                if e == 0x30 && !matches!(self.peek(), Some(0x30..=0x39)) {
                    Ok(ClassAtom::Unit(0))
                } else if self.unicode_mode() && e != 0x30 {
                    Err(RegexError::ParseError(
                        "octal escapes are not allowed in unicode mode".to_string(),
                    ))
                } else {
                    Ok(ClassAtom::Unit(self.parse_octal(e)))
                }
            }
            0x38 | 0x39 => {
                // \8 \9: identity in non-unicode; error in unicode mode.
                // May be followed by more digits (all literal).
                let mut us = vec![e];
                while let Some(d) = self.peek() {
                    if matches!(d, 0x30..=0x39) {
                        us.push(d);
                        self.next();
                    } else {
                        break;
                    }
                }
                if self.unicode_mode() {
                    Err(RegexError::ParseError(
                        "invalid escape in character class".to_string(),
                    ))
                } else {
                    Ok(ClassAtom::Units(us))
                }
            }
            0x78 => match self.parse_hex_escape()? {
                Ast::Lit(v) => Ok(ClassAtom::Unit(v)),
                _ => unreachable!(),
            },
            0x75 => match self.parse_unicode_escape()? {
                Ast::Lit(v) => Ok(ClassAtom::Unit(v)),
                _ => unreachable!(),
            },
            0x63 => match self.parse_control_escape()? {
                Ast::Lit(v) => Ok(ClassAtom::Unit(v)),
                _ => unreachable!(),
            },
            0x70 | 0x50 => {
                if self.unicode_mode() {
                    let prop = self.parse_prop()?;
                    Ok(ClassAtom::Prop(prop, e == 0x50))
                } else {
                    Ok(ClassAtom::Unit(e))
                }
            }
            _ => {
                if self.unicode_mode() && is_ascii_alnum(e) {
                    Err(RegexError::ParseError(
                        "invalid escape in character class".to_string(),
                    ))
                } else {
                    Ok(ClassAtom::Unit(e))
                }
            }
        }
    }
}

/// Resolve [`Ast::DecimalEscape`] (backreference vs. Annex-B octal) and
/// [`Ast::NamedBackRef`] now that the total group count is known.
fn resolve_escapes(
    ast: &mut Ast,
    groups: usize,
    named: &HashMap<String, usize>,
    unicode: bool,
) -> Result<(), RegexError> {
    match ast {
        Ast::DecimalEscape(s) => {
            let n: usize = s.parse().unwrap_or(usize::MAX);
            if n >= 1 && n <= groups {
                *ast = Ast::BackRef(n);
            } else if unicode {
                return Err(RegexError::ParseError(format!(
                    "invalid backreference \\{s}"
                )));
            } else {
                // Annex B: longest octal prefix ([0-7], up to 3 digits),
                // remaining digits are literals; leading 8/9 are literals.
                let bytes = s.as_bytes();
                let mut oct_len = 0;
                while oct_len < bytes.len() && oct_len < 3 && bytes[oct_len] < b'8' {
                    oct_len += 1;
                }
                let mut seq = Vec::new();
                if oct_len > 0 {
                    let v = u32::from_str_radix(&s[..oct_len], 8).unwrap_or(0);
                    seq.push(Ast::Lit(v));
                }
                for c in s[oct_len..].chars() {
                    seq.push(Ast::Lit(c as u32));
                }
                *ast = match seq.len() {
                    1 => seq.pop().unwrap(),
                    _ => Ast::Seq(seq),
                };
            }
        }
        Ast::NamedBackRef(name) => {
            if let Some(&idx) = named.get(name) {
                *ast = Ast::BackRef(idx);
            } else {
                return Err(RegexError::ParseError(format!(
                    "unknown group name '{name}'"
                )));
            }
        }
        Ast::Seq(v) | Ast::Alt(v) => {
            for a in v {
                resolve_escapes(a, groups, named, unicode)?;
            }
        }
        Ast::Group { body, .. } | Ast::Quant { body, .. } | Ast::LookAhead { body, .. } => {
            resolve_escapes(body, groups, named, unicode)?;
        }
        _ => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Bytecode
// ---------------------------------------------------------------------------

/// VM instruction. Targets are absolute code indices.
#[derive(Debug, Clone)]
enum Inst {
    /// Match one input value (code unit, or astral code point in u/v mode).
    Lit(u32),
    /// Case-insensitive literal (`i` flag).
    LitI(u32),
    /// Character class (index into the class table).
    Class(u32),
    /// `.` without `s`: any value except line terminators.
    Dot,
    /// `.` with `s`: any value.
    DotAll,
    /// `^` without `m`.
    Start,
    /// `$` without `m`.
    End,
    /// `^` with `m`.
    LineStart,
    /// `$` with `m`.
    LineEnd,
    /// `\b` (true) / `\B` (false). `\w` is ASCII-only, per spec.
    WordBound(bool),
    /// Record group boundary: slot `2*n` = start, `2*n+1` = end.
    Save(u32),
    /// Choice point: try `a`, on failure backtrack to `b`.
    Split(u32, u32),
    Jump(u32),
    /// Counted quantifier head. `max == u32::MAX` means unbounded.
    Count {
        id: u32,
        min: u32,
        max: u32,
        greedy: bool,
        body: u32,
        exit: u32,
    },
    /// After one body iteration: enforce the empty-iteration rule.
    CountCheck {
        id: u32,
        min: u32,
        head: u32,
        exit: u32,
    },
    /// Numbered backreference.
    BackRef(u32),
    /// Lookahead: run `body` at the current position without consuming.
    Look {
        neg: bool,
        body: u32,
        cont: u32,
    },
    /// End of a lookahead body: report success to the enclosing `Look`.
    /// Only ever executed inside a lookahead sub-VM.
    LookEnd,
    /// Whole-match success.
    Match,
}

/// A compiled character class: ranges over input values plus `\p{...}`
/// predicates. Under the `i` flag the compiler also inserts case-folded
/// copies of every range; the VM additionally tests `fold_case(value)`.
#[derive(Debug, Clone)]
struct CompiledClass {
    ranges: Vec<(u32, u32)>,
    negated: bool,
    props: Vec<(Prop, bool)>,
}

struct Compiler {
    code: Vec<Inst>,
    classes: Vec<CompiledClass>,
    num_counts: u32,
    ignore_case: bool,
    multiline: bool,
    dot_all: bool,
}

impl Compiler {
    fn new(flags: Flags) -> Self {
        Compiler {
            code: Vec::new(),
            classes: Vec::new(),
            num_counts: 0,
            ignore_case: flags.ignore_case,
            multiline: flags.multiline,
            dot_all: flags.dot_all,
        }
    }

    fn emit(&mut self, inst: Inst) -> usize {
        let pc = self.code.len();
        self.code.push(inst);
        pc
    }

    fn add_class(&mut self, class: &Class) -> u32 {
        let mut ranges = class.ranges.clone();
        if self.ignore_case {
            // Insert case-folded copies so `[A-Z]` matches `a` and vice
            // versa. The VM also folds the input value (covers ς/Σ style
            // cases where the folded value lands in an unfolded range).
            let extra: Vec<(u32, u32)> = ranges
                .iter()
                .map(|&(lo, hi)| (fold_case(lo), fold_case(hi)))
                .filter(|&(flo, fhi)| flo <= fhi)
                .collect();
            ranges.extend(extra);
        }
        let idx = self.classes.len() as u32;
        self.classes.push(CompiledClass {
            ranges,
            negated: class.negated,
            props: class.props.clone(),
        });
        idx
    }

    fn compile(&mut self, ast: &Ast) -> Result<(), RegexError> {
        // Group 0 spans the whole match.
        self.emit(Inst::Save(0));
        self.compile_ast(ast)?;
        self.emit(Inst::Save(1));
        self.emit(Inst::Match);
        Ok(())
    }

    fn compile_ast(&mut self, ast: &Ast) -> Result<(), RegexError> {
        match ast {
            Ast::Empty => {}
            Ast::Lit(v) => {
                if self.ignore_case {
                    self.emit(Inst::LitI(*v));
                } else {
                    self.emit(Inst::Lit(*v));
                }
            }
            Ast::Dot => {
                self.emit(if self.dot_all {
                    Inst::DotAll
                } else {
                    Inst::Dot
                });
            }
            Ast::Class(c) => {
                let idx = self.add_class(c);
                self.emit(Inst::Class(idx));
            }
            Ast::Start => {
                self.emit(if self.multiline {
                    Inst::LineStart
                } else {
                    Inst::Start
                });
            }
            Ast::End => {
                self.emit(if self.multiline {
                    Inst::LineEnd
                } else {
                    Inst::End
                });
            }
            Ast::WordBoundary(b) => {
                self.emit(Inst::WordBound(*b));
            }
            Ast::Seq(terms) => {
                for t in terms {
                    self.compile_ast(t)?;
                }
            }
            Ast::Alt(branches) => {
                let n = branches.len();
                let mut prev_split: Option<usize> = None;
                let mut end_jumps: Vec<usize> = Vec::new();
                for (i, b) in branches.iter().enumerate() {
                    let is_last = i + 1 == n;
                    if let Some(s) = prev_split {
                        let target = self.code.len() as u32;
                        if let Inst::Split(_, ref mut cont) = self.code[s] {
                            *cont = target;
                        }
                    }
                    prev_split = if is_last {
                        None
                    } else {
                        Some(self.emit(Inst::Split(0, 0)))
                    };
                    if let Some(s) = prev_split {
                        let body = (s + 1) as u32;
                        if let Inst::Split(ref mut first, _) = self.code[s] {
                            *first = body;
                        }
                    }
                    self.compile_ast(b)?;
                    if !is_last {
                        end_jumps.push(self.emit(Inst::Jump(0)));
                    }
                }
                let end = self.code.len() as u32;
                for j in end_jumps {
                    if let Inst::Jump(ref mut t) = self.code[j] {
                        *t = end;
                    }
                }
            }
            Ast::Group { idx, body } => {
                let i = *idx as u32;
                self.emit(Inst::Save(2 * i));
                self.compile_ast(body)?;
                self.emit(Inst::Save(2 * i + 1));
            }
            Ast::Quant {
                body,
                min,
                max,
                greedy,
            } => {
                let head = self.code.len() as u32;
                let id = self.num_counts;
                self.num_counts += 1;
                self.emit(Inst::Count {
                    id,
                    min: *min,
                    max: *max,
                    greedy: *greedy,
                    body: 0,
                    exit: 0,
                });
                let body_pc = self.code.len() as u32;
                self.compile_ast(body)?;
                let check_pc = self.code.len() as u32;
                self.emit(Inst::CountCheck {
                    id,
                    min: *min,
                    head,
                    exit: 0,
                });
                let exit_pc = self.code.len() as u32;
                self.code[head as usize] = Inst::Count {
                    id,
                    min: *min,
                    max: *max,
                    greedy: *greedy,
                    body: body_pc,
                    exit: exit_pc,
                };
                self.code[check_pc as usize] = Inst::CountCheck {
                    id,
                    min: *min,
                    head,
                    exit: exit_pc,
                };
            }
            Ast::BackRef(n) => {
                self.emit(Inst::BackRef(*n as u32));
            }
            Ast::LookAhead { neg, body } => {
                let look = self.emit(Inst::Look {
                    neg: *neg,
                    body: 0,
                    cont: 0,
                });
                let body_pc = self.code.len() as u32;
                self.compile_ast(body)?;
                // The body ends with LookEnd (success signal for the sub-VM),
                // never by falling through to the outer continuation.
                self.emit(Inst::LookEnd);
                let cont = self.code.len() as u32;
                self.code[look] = Inst::Look {
                    neg: *neg,
                    body: body_pc,
                    cont,
                };
            }
            Ast::NamedBackRef(name) => {
                return Err(RegexError::ParseError(format!(
                    "unresolved named backreference '{name}'"
                )));
            }
            Ast::DecimalEscape(s) => {
                return Err(RegexError::ParseError(format!(
                    "unresolved decimal escape \\{s}"
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Backtracking VM (explicit choice-point stack)
// ---------------------------------------------------------------------------

/// One active counted-loop iteration: the input position where it started
/// and, for greedy loops, the index of the loop-exit choice pushed at the
/// head (whose `saves` snapshot is the iteration-start capture state).
#[derive(Clone, Copy)]
struct LoopEntry {
    pos: usize,
    choice: Option<usize>,
}

struct Choice {
    pc: usize,
    pos: usize,
    saves: Vec<Option<usize>>,
    counts: Vec<u32>,
    prog: Vec<LoopEntry>,
}

struct Vm<'a> {
    prog: &'a [Inst],
    classes: &'a [CompiledClass],
    input: &'a [u16],
    unicode: bool,
    ignore_case: bool,
    steps: u64,
    limit_hit: bool,
    pc: usize,
    pos: usize,
    saves: Vec<Option<usize>>,
    counts: Vec<u32>,
    prog_stack: Vec<LoopEntry>,
    bt: Vec<Choice>,
}

impl<'a> Vm<'a> {
    /// Decode the input value at the current position: in unicode mode a
    /// well-formed surrogate pair decodes to one astral code point,
    /// otherwise every code unit is its own value.
    fn decode(&self) -> Option<(u32, usize)> {
        let u = *self.input.get(self.pos)?;
        if self.unicode
            && (0xD800..0xDC00).contains(&u)
            && let Some(&u2) = self.input.get(self.pos + 1)
            && (0xDC00..0xE000).contains(&u2)
        {
            let cp = 0x10000 + (((u as u32) - 0xD800) << 10) + ((u2 as u32) - 0xDC00);
            return Some((cp, self.pos + 2));
        }
        Some((u as u32, self.pos + 1))
    }

    fn class_match(&self, idx: usize, v: u32) -> bool {
        let cc = &self.classes[idx];
        let mut hit = cc.ranges.iter().any(|&(lo, hi)| v >= lo && v <= hi);
        if !hit && self.ignore_case {
            let fv = fold_case(v);
            hit = cc.ranges.iter().any(|&(lo, hi)| fv >= lo && fv <= hi);
        }
        match char::from_u32(v) {
            Some(c) => {
                for &(p, neg) in &cc.props {
                    if prop_test(p, c) != neg {
                        hit = true;
                        break;
                    }
                }
            }
            // Lone surrogate: plain props never hold; negated ones do
            // (e.g. `\P{Assigned}` matches a lone surrogate).
            None => {
                for &(_, neg) in &cc.props {
                    if neg {
                        hit = true;
                        break;
                    }
                }
            }
        }
        if cc.negated { !hit } else { hit }
    }

    fn push_choice(&mut self, pc: usize) {
        self.bt.push(Choice {
            pc,
            pos: self.pos,
            saves: self.saves.clone(),
            counts: self.counts.clone(),
            prog: self.prog_stack.clone(),
        });
    }

    fn backtrack(&mut self) -> bool {
        match self.bt.pop() {
            Some(ch) => {
                self.pc = ch.pc;
                self.pos = ch.pos;
                self.saves = ch.saves;
                self.counts = ch.counts;
                self.prog_stack = ch.prog;
                true
            }
            None => false,
        }
    }

    fn run(&mut self, start_pc: usize, start_pos: usize) -> Option<usize> {
        self.pc = start_pc;
        self.pos = start_pos;
        loop {
            if self.steps >= MAX_STEPS {
                self.limit_hit = true;
                return None;
            }
            self.steps += 1;
            match self.prog[self.pc].clone() {
                Inst::Lit(v) => match self.decode() {
                    Some((w, np)) if w == v => {
                        self.pos = np;
                        self.pc += 1;
                    }
                    _ => {
                        if !self.backtrack() {
                            return None;
                        }
                    }
                },
                Inst::LitI(v) => match self.decode() {
                    Some((w, np)) if w == v || fold_case(w) == fold_case(v) => {
                        self.pos = np;
                        self.pc += 1;
                    }
                    _ => {
                        if !self.backtrack() {
                            return None;
                        }
                    }
                },
                Inst::Class(i) => match self.decode() {
                    Some((w, np)) if self.class_match(i as usize, w) => {
                        self.pos = np;
                        self.pc += 1;
                    }
                    _ => {
                        if !self.backtrack() {
                            return None;
                        }
                    }
                },
                Inst::Dot => match self.decode() {
                    Some((w, np)) if !is_line_terminator(w) => {
                        self.pos = np;
                        self.pc += 1;
                    }
                    _ => {
                        if !self.backtrack() {
                            return None;
                        }
                    }
                },
                Inst::DotAll => match self.decode() {
                    Some((_, np)) => {
                        self.pos = np;
                        self.pc += 1;
                    }
                    _ => {
                        if !self.backtrack() {
                            return None;
                        }
                    }
                },
                Inst::Start => {
                    if self.pos != 0 {
                        if !self.backtrack() {
                            return None;
                        }
                    } else {
                        self.pc += 1;
                    }
                }
                Inst::End => {
                    if self.pos != self.input.len() {
                        if !self.backtrack() {
                            return None;
                        }
                    } else {
                        self.pc += 1;
                    }
                }
                Inst::LineStart => {
                    let ok = self.pos == 0 || is_line_terminator(self.input[self.pos - 1] as u32);
                    if ok {
                        self.pc += 1;
                    } else if !self.backtrack() {
                        return None;
                    }
                }
                Inst::LineEnd => {
                    let ok = self.pos == self.input.len()
                        || self.decode().is_some_and(|(w, _)| is_line_terminator(w));
                    if ok {
                        self.pc += 1;
                    } else if !self.backtrack() {
                        return None;
                    }
                }
                Inst::WordBound(yes) => {
                    let before = self
                        .pos
                        .checked_sub(1)
                        .map(|p| is_word_char(self.input[p] as u32))
                        .unwrap_or(false);
                    let after = self.decode().map(|(w, _)| is_word_char(w)).unwrap_or(false);
                    if (before != after) == yes {
                        self.pc += 1;
                    } else if !self.backtrack() {
                        return None;
                    }
                }
                Inst::Save(slot) => {
                    self.saves[slot as usize] = Some(self.pos);
                    self.pc += 1;
                }
                Inst::Split(a, b) => {
                    self.push_choice(b as usize);
                    self.pc = a as usize;
                }
                Inst::Jump(t) => {
                    self.pc = t as usize;
                }
                Inst::Count {
                    id,
                    min,
                    max,
                    greedy,
                    body,
                    exit,
                } => {
                    let c = self.counts[id as usize];
                    if greedy {
                        if c < max {
                            let choice = if c >= min {
                                self.push_choice(exit as usize);
                                Some(self.bt.len() - 1)
                            } else {
                                None
                            };
                            self.prog_stack.push(LoopEntry {
                                pos: self.pos,
                                choice,
                            });
                            self.counts[id as usize] = c + 1;
                            self.pc = body as usize;
                        } else if c >= min {
                            self.pc = exit as usize;
                        } else if !self.backtrack() {
                            return None;
                        }
                    } else if c >= min {
                        // Lazy: prefer exit; keep a choice to iterate.
                        if c < max {
                            self.prog_stack.push(LoopEntry {
                                pos: self.pos,
                                choice: None,
                            });
                            self.counts[id as usize] = c + 1;
                            self.push_choice(body as usize);
                            self.prog_stack.pop();
                            self.counts[id as usize] = c;
                        }
                        self.pc = exit as usize;
                    } else if c < max {
                        self.prog_stack.push(LoopEntry {
                            pos: self.pos,
                            choice: None,
                        });
                        self.counts[id as usize] = c + 1;
                        self.pc = body as usize;
                    } else if !self.backtrack() {
                        return None;
                    }
                }
                Inst::CountCheck {
                    id,
                    min,
                    head,
                    exit,
                } => {
                    let c = self.counts[id as usize];
                    let entry = self.prog_stack.pop().expect("CountCheck without entry");
                    // An iteration that consumed nothing terminates the loop
                    // once the minimum is satisfied (spec RepeatMatcher). When
                    // the minimum was already satisfied before this iteration
                    // (c > min), the spec returns the pre-iteration state, so
                    // the empty iteration's captures are discarded.
                    if self.pos == entry.pos && c >= min {
                        if c > min
                            && let Some(ci) = entry.choice
                        {
                            self.saves = self.bt[ci].saves.clone();
                        }
                        self.pc = exit as usize;
                    } else {
                        self.pc = head as usize;
                    }
                }
                Inst::BackRef(n) => {
                    let n = n as usize;
                    match (self.saves[2 * n], self.saves[2 * n + 1]) {
                        (Some(s), Some(e)) => {
                            let len = e - s;
                            let mut ok = self.pos + len <= self.input.len();
                            if ok {
                                for i in 0..len {
                                    let a = self.input[s + i];
                                    let b = self.input[self.pos + i];
                                    if a != b
                                        && !(self.ignore_case
                                            && fold_case(a as u32) == fold_case(b as u32))
                                    {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                            if ok {
                                self.pos += len;
                                self.pc += 1;
                            } else if !self.backtrack() {
                                return None;
                            }
                        }
                        // A group that never participated matches empty.
                        _ => {
                            self.pc += 1;
                        }
                    }
                }
                Inst::Look { neg, body, cont } => {
                    let mut sub = Vm {
                        prog: self.prog,
                        classes: self.classes,
                        input: self.input,
                        unicode: self.unicode,
                        ignore_case: self.ignore_case,
                        steps: self.steps,
                        limit_hit: false,
                        pc: 0,
                        pos: 0,
                        saves: self.saves.clone(),
                        counts: vec![0; self.counts.len()],
                        prog_stack: Vec::new(),
                        bt: Vec::new(),
                    };
                    let matched = sub.run(body as usize, self.pos).is_some();
                    self.steps = sub.steps;
                    if sub.limit_hit {
                        self.limit_hit = true;
                        return None;
                    }
                    if matched != neg {
                        if !neg {
                            // Captures made inside a positive lookahead are kept.
                            self.saves = sub.saves;
                        }
                        self.pc = cont as usize;
                    } else if !self.backtrack() {
                        return None;
                    }
                }
                Inst::Match => return Some(self.pos),
                // Unreachable in the main VM (bodies only run in lookahead
                // sub-VMs); treat defensively as success.
                Inst::LookEnd => return Some(self.pos),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A compiled regular expression. Opaque; build with [`CompiledRegex::compile`].
pub struct CompiledRegex {
    prog: Vec<Inst>,
    classes: Vec<CompiledClass>,
    num_groups: usize,
    named_names: Vec<String>,
    /// 1-based group number per entry of `named_names`.
    /// (Consumed by the RegExp builtins layer for the `groups` object.)
    #[allow(dead_code)]
    named_groups: Vec<usize>,
    num_counts: usize,
    flags: Flags,
    error: Cell<Option<RegexError>>,
}

impl CompiledRegex {
    /// Compile `pattern` with JS `flags` (any combination of `dgimsuvy`).
    ///
    /// `pattern` is WTF-8: JS strings may hold lone surrogates, and they
    /// are preserved here, never replaced with U+FFFD. Pattern values are
    /// code points in unicode mode (astral characters are one atom) and
    /// UTF-16 code units otherwise; in both modes a lone surrogate is a
    /// single unit value (`0xD800..=0xDFFF`).
    ///
    /// Returns [`RegexError::ParseError`] for invalid patterns, invalid
    /// flags, unsupported constructs (lookbehind, nested `v`-mode classes,
    /// possessive quantifiers, unknown `\p{...}` properties).
    pub fn compile(pattern: &Wtf8, flags: &str) -> Result<Self, RegexError> {
        let flags = parse_flags(flags)?;
        // Pattern values: code points in unicode mode (astral chars are one
        // atom), UTF-16 code units otherwise. Lone surrogates surface as
        // their unit values in both modes.
        let units: Vec<u32> = if flags.unicode_mode() {
            pattern.chars_wtf8().collect()
        } else {
            pattern.code_units().map(u32::from).collect()
        };
        let mut parser = Parser {
            units,
            pos: 0,
            flags,
            group_count: 0,
            group_names: Vec::new(),
            named: HashMap::new(),
        };
        let mut ast = parser.parse()?;
        let groups = parser.group_count;
        resolve_escapes(&mut ast, groups, &parser.named, flags.unicode_mode())?;
        let mut compiler = Compiler::new(flags);
        compiler.compile(&ast)?;
        let mut named_names = Vec::new();
        let mut named_groups = Vec::new();
        for (idx, name) in parser.group_names.iter().enumerate() {
            if let Some(name) = name {
                // `group_names` is 0-based; group numbers are 1-based.
                named_groups.push(idx + 1);
                named_names.push(name.clone());
            }
        }
        Ok(CompiledRegex {
            prog: compiler.code,
            classes: compiler.classes,
            num_groups: groups,
            named_names,
            named_groups,
            num_counts: compiler.num_counts as usize,
            flags,
            error: Cell::new(None),
        })
    }

    /// Search `input` for the leftmost match at or after UTF-16 code-unit
    /// offset `start` (`start` is clamped to the input length). With the `y`
    /// (sticky) flag only `start` itself is tried.
    ///
    /// Returns the capture list (index 0 is the whole match; `None` marks a
    /// non-participating group), or `None` when nothing matches. If the
    /// backtracking budget is exhausted this also returns `None` and records
    /// [`RegexError::BacktrackLimit`]; see [`CompiledRegex::take_error`].
    /// Single-shot execution against a `&str`; encodes the input to UTF-16
    /// first. Kept as the convenience entry point (tests/benches); hot
    /// loops over one input should use [`exec_units`](Self::exec_units).
    #[allow(dead_code)]
    pub fn exec(&self, input: &str, start: usize) -> Option<Vec<Option<Capture>>> {
        let text: Vec<u16> = input.encode_utf16().collect();
        self.exec_units(&text, start)
    }

    /// Execute against pre-encoded UTF-16 input. Prefer this over
    /// [`exec`](Self::exec) when matching repeatedly against the same input
    /// (global replace/split/match loops): it skips the O(n) UTF-16
    /// re-encode that [`exec`](Self::exec) pays on every call.
    pub fn exec_units(&self, units: &[u16], start: usize) -> Option<Vec<Option<Capture>>> {
        self.error.set(None);
        let len = units.len();
        let start = start.min(len);
        let mut vm = Vm {
            prog: &self.prog,
            classes: &self.classes,
            input: units,
            unicode: self.flags.unicode_mode(),
            ignore_case: self.flags.ignore_case,
            steps: 0,
            limit_hit: false,
            pc: 0,
            pos: 0,
            saves: vec![None; 2 * (self.num_groups + 1)],
            counts: vec![0; self.num_counts],
            prog_stack: Vec::new(),
            bt: Vec::new(),
        };
        let sticky = self.flags.sticky;
        let mut pos = start;
        loop {
            // Fresh per-position state; the step budget and limit flag are
            // shared across all positions tried by this call.
            vm.saves.iter_mut().for_each(|s| *s = None);
            vm.counts.iter_mut().for_each(|c| *c = 0);
            vm.bt.clear();
            vm.prog_stack.clear();
            if let Some(_end) = vm.run(0, pos) {
                let mut out = Vec::with_capacity(self.num_groups + 1);
                for g in 0..=self.num_groups {
                    out.push(match (vm.saves[2 * g], vm.saves[2 * g + 1]) {
                        (Some(s), Some(e)) => Some(Capture { start: s, end: e }),
                        _ => None,
                    });
                }
                return Some(out);
            }
            if vm.limit_hit {
                self.error.set(Some(RegexError::BacktrackLimit));
                return None;
            }
            if sticky || pos >= len {
                return None;
            }
            pos += 1;
        }
    }

    /// Names of the named capture groups, in group-number order.
    pub fn capture_names(&self) -> &[String] {
        &self.named_names
    }

    /// 1-based group number per entry of [`CompiledRegex::capture_names`].
    /// (Consumed by the RegExp builtins layer for the `groups` object.)
    #[allow(dead_code)]
    pub fn capture_name_groups(&self) -> &[usize] {
        &self.named_groups
    }

    /// Take the error recorded by the last [`CompiledRegex::exec`] call, if
    /// any. Only [`RegexError::BacktrackLimit`] is reported this way.
    pub fn take_error(&self) -> Option<RegexError> {
        self.error.take()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wtf8::Wtf8Buf;

    fn t(pattern: &str, flags: &str, input: &str, start: usize) -> Option<Vec<Option<Capture>>> {
        let re = CompiledRegex::compile(Wtf8::from_str(pattern), flags).expect("compile failed");
        re.exec(input, start)
    }

    fn whole(caps: &[Option<Capture>]) -> (usize, usize) {
        let c = caps[0].expect("expected a match");
        (c.start, c.end)
    }

    #[test]
    fn classes_basic() {
        // [a-z]+ leftmost match.
        let caps = t(r"[a-z]+", "", "abc123", 0).unwrap();
        assert_eq!(whole(&caps), (0, 3));
        // \d+ starts at the digits.
        let caps = t(r"\d+", "", "abc123", 0).unwrap();
        assert_eq!(whole(&caps), (3, 6));
        // Negated class.
        let caps = t(r"[^0-9]+", "", "abc123", 0).unwrap();
        assert_eq!(whole(&caps), (0, 3));
        // \w includes underscore and digits.
        let caps = t(r"\w+", "", "__x9!!", 0).unwrap();
        assert_eq!(whole(&caps), (0, 4));
        // \s matches space.
        let caps = t(r"a\sb", "", "a b", 0).unwrap();
        assert_eq!(whole(&caps), (0, 3));
        // Range endpoints and escapes.
        assert!(t(r"[\d]", "", "5", 0).is_some());
        assert!(t(r"[\d]", "", "x", 0).is_none());
        assert!(t(r"[\D]", "", "x", 0).is_some());
        // Case-insensitive class: [A-Z] matches 'a', [a-z] matches 'A'.
        let caps = t(r"[A-Z]+", "i", "abc", 0).unwrap();
        assert_eq!(whole(&caps), (0, 3));
        let caps = t(r"[a-z]+", "i", "ABC", 0).unwrap();
        assert_eq!(whole(&caps), (0, 3));
    }

    #[test]
    fn backrefs() {
        let caps = t(r"(a)\1", "", "aa", 0).unwrap();
        assert_eq!(whole(&caps), (0, 2));
        assert_eq!(caps[1], Some(Capture { start: 0, end: 1 }));
        assert!(t(r"(a)\1", "", "ab", 0).is_none());
        let caps = t(r"(a|b)\1", "", "bb", 0).unwrap();
        assert_eq!(whole(&caps), (0, 2));
        // Case-insensitive backref.
        assert!(t(r"(a)\1", "i", "aA", 0).is_some());
        // Group that never participates matches empty in a backref.
        assert!(t(r"(a)?b\1", "", "b", 0).is_some());
        // Named backref.
        let caps = t(r"(?<w>\w+) \k<w>", "", "abc abc", 0).unwrap();
        assert_eq!(whole(&caps), (0, 7));
    }

    #[test]
    fn lookahead() {
        let caps = t(r"foo(?=bar)", "", "foobar", 0).unwrap();
        assert_eq!(whole(&caps), (0, 3)); // lookahead consumes nothing
        assert!(t(r"foo(?=bar)", "", "foobaz", 0).is_none());
        let caps = t(r"foo(?!bar)", "", "foobaz", 0).unwrap();
        assert_eq!(whole(&caps), (0, 3));
        assert!(t(r"foo(?!bar)", "", "foobar", 0).is_none());
        // Captures inside a positive lookahead persist.
        let caps = t(r"(?=(a))a\1", "", "aa", 0).unwrap();
        assert_eq!(whole(&caps), (0, 2));
        assert_eq!(caps[1], Some(Capture { start: 0, end: 1 }));
        // Negative lookahead does not leak captures.
        let caps = t(r"(?!(a))b", "", "b", 0).unwrap();
        assert_eq!(caps[1], None);
    }

    #[test]
    fn named_groups() {
        let re =
            CompiledRegex::compile(Wtf8::from_str(r"(?<first>\w+)-(?<second>\d+)"), "").unwrap();
        assert_eq!(
            re.capture_names(),
            &["first".to_string(), "second".to_string()]
        );
        let caps = re.exec("ab-12", 0).unwrap();
        assert_eq!(whole(&caps), (0, 5));
        assert_eq!(caps[1], Some(Capture { start: 0, end: 2 }));
        assert_eq!(caps[2], Some(Capture { start: 3, end: 5 }));
        // Unnamed groups contribute no names.
        let re = CompiledRegex::compile(Wtf8::from_str(r"(a)(?<b>b)"), "").unwrap();
        assert_eq!(re.capture_names(), &["b".to_string()]);
    }

    #[test]
    fn sticky() {
        let caps = t(r"b+", "y", "abbb", 1).unwrap();
        assert_eq!(whole(&caps), (1, 4));
        assert!(t(r"b+", "y", "abbb", 0).is_none());
        assert!(t(r"b+", "y", "abbb", 2).is_some());
        // Non-sticky searches forward from start.
        let caps = t(r"b+", "", "abbb", 0).unwrap();
        assert_eq!(whole(&caps), (1, 4));
    }

    #[test]
    fn astral_u_flag() {
        // UTF-16 code-unit indices: the emoji occupies units 1..3.
        let caps = t("😀", "u", "a😀b", 0).unwrap();
        assert_eq!(whole(&caps), (1, 3));
        // Without u the two surrogates match as two units: same span.
        let caps = t("😀", "", "a😀b", 0).unwrap();
        assert_eq!(whole(&caps), (1, 3));
        // `.` consumes the whole pair in unicode mode.
        let caps = t(".", "u", "😀", 0).unwrap();
        assert_eq!(whole(&caps), (0, 2));
        // (&str inputs are always valid UTF-8, so lone surrogates cannot
        // occur; the engine still handles them defensively as single units.)
        // Quantifier applies to the whole pair in u mode.
        let caps = t("😀{2}", "u", "😀😀", 0).unwrap();
        assert_eq!(whole(&caps), (0, 4));
        assert!(t("😀{2}", "u", "😀", 0).is_none());
    }

    #[test]
    fn lone_surrogate_pattern_exact() {
        // A lone-surrogate pattern unit (what `String.fromCharCode(0xD800)`
        // produces as pattern text) compiles to the single unit 0xD800,
        // not to U+FFFD: it matches a lone-surrogate subject unit and does
        // not match U+FFFD.
        let pattern = Wtf8Buf::from_units(&[0xD800]);
        let re = CompiledRegex::compile(pattern.as_wtf8(), "").unwrap();
        let caps = re.exec_units(&[0xD800], 0).unwrap();
        assert_eq!(whole(&caps), (0, 1));
        assert!(re.exec_units(&[0xFFFD], 0).is_none());
        // In unicode mode the surrogate is still one unit value, matching
        // a lone-surrogate subject unit; astral atoms keep working.
        let re_u = CompiledRegex::compile(pattern.as_wtf8(), "u").unwrap();
        let caps = re_u.exec_units(&[0xD800], 0).unwrap();
        assert_eq!(whole(&caps), (0, 1));
        let re_a = CompiledRegex::compile(Wtf8::from_str("😀"), "u").unwrap();
        let caps = re_a.exec_units(&[0xD83D, 0xDE00], 0).unwrap();
        assert_eq!(whole(&caps), (0, 2));
        // A literal U+FFFD pattern is distinct: it matches U+FFFD, not the
        // lone surrogate.
        let re_f = CompiledRegex::compile(Wtf8::from_str("\u{FFFD}"), "").unwrap();
        assert!(re_f.exec_units(&[0xFFFD], 0).is_some());
        assert!(re_f.exec_units(&[0xD800], 0).is_none());
    }

    #[test]
    fn catastrophic_step_cap() {
        let re = CompiledRegex::compile(Wtf8::from_str(r"(a+)+b"), "").unwrap();
        let input = "a".repeat(30);
        assert!(re.exec(&input, 0).is_none());
        assert!(matches!(re.take_error(), Some(RegexError::BacktrackLimit)));
        // A benign long match on the same shape stays under budget.
        let re = CompiledRegex::compile(Wtf8::from_str(r"(a+)+b"), "").unwrap();
        let input = format!("{}b", "a".repeat(500));
        let caps = re.exec(&input, 0).unwrap();
        assert_eq!(whole(&caps), (0, 501));
        assert!(re.take_error().is_none());
    }

    #[test]
    fn flags_ims() {
        let caps = t("abc", "i", "ABC", 0).unwrap();
        assert_eq!(whole(&caps), (0, 3));
        let caps = t("^b", "m", "a\nb", 0).unwrap();
        assert_eq!(whole(&caps), (2, 3));
        assert!(t("^b", "", "a\nb", 0).is_none());
        let caps = t("a$", "m", "a\nb", 0).unwrap();
        assert_eq!(whole(&caps), (0, 1));
        let caps = t("a.b", "s", "a\nb", 0).unwrap();
        assert_eq!(whole(&caps), (0, 3));
        assert!(t("a.b", "", "a\nb", 0).is_none());
        // d/g flags are accepted.
        assert!(CompiledRegex::compile(Wtf8::from_str("a"), "dg").is_ok());
    }

    #[test]
    fn word_boundaries_anchors_dot() {
        let caps = t(r"\bb\b", "", "a b c", 0).unwrap();
        assert_eq!(whole(&caps), (2, 3));
        assert!(t(r"\bb\b", "", "abc", 0).is_none());
        assert!(t(r"\Bz\B", "", "azzb", 0).is_some());
        assert!(t(r"\Bz\B", "", "az b", 0).is_none());
        assert!(t("^a", "", "ba", 0).is_none());
        let caps = t("a$", "", "ba", 0).unwrap();
        assert_eq!(whole(&caps), (1, 2));
        // Dot does not cross line terminators without s.
        assert!(t(".", "", "\u{2028}", 0).is_none());
        assert!(t(".", "s", "\u{2028}", 0).is_some());
    }

    #[test]
    fn quantifiers_lazy_and_counts() {
        let caps = t("a+?", "", "aaa", 0).unwrap();
        assert_eq!(whole(&caps), (0, 1));
        let caps = t("a+", "", "aaa", 0).unwrap();
        assert_eq!(whole(&caps), (0, 3));
        let caps = t("a{2,3}", "", "aaaa", 0).unwrap();
        assert_eq!(whole(&caps), (0, 3));
        let caps = t("a{2,3}?", "", "aaaa", 0).unwrap();
        assert_eq!(whole(&caps), (0, 2));
        let caps = t("a{2}", "", "aaa", 0).unwrap();
        assert_eq!(whole(&caps), (0, 2));
        // {n,m} with n > m is a syntax error (spec early error, not Annex B).
        assert!(matches!(
            CompiledRegex::compile(Wtf8::from_str("a{2,1}"), ""),
            Err(RegexError::ParseError(_))
        ));
        // Empty-matching loops terminate.
        let caps = t("(a*)*", "", "b", 0).unwrap();
        assert_eq!(whole(&caps), (0, 0));
        let caps = t("(?:)", "", "", 0).unwrap();
        assert_eq!(whole(&caps), (0, 0));
    }

    #[test]
    fn groups_alternation_empty() {
        let caps = t("(a)(b)?", "", "a", 0).unwrap();
        assert_eq!(caps[1], Some(Capture { start: 0, end: 1 }));
        assert_eq!(caps[2], None);
        let caps = t("a|ab", "", "ab", 0).unwrap();
        assert_eq!(whole(&caps), (0, 1)); // leftmost alternative wins
        let caps = t("(?:ab)|(?:cd)", "", "cd", 0).unwrap();
        assert_eq!(whole(&caps), (0, 2));
        // Empty pattern matches empty at start.
        let caps = t("", "", "abc", 1).unwrap();
        assert_eq!(whole(&caps), (1, 1));
    }

    #[test]
    fn unicode_properties() {
        let caps = t(r"\p{Letter}", "u", "é", 0).unwrap();
        assert_eq!(whole(&caps), (0, 1));
        assert!(t(r"\p{Letter}", "u", "5", 0).is_none());
        assert!(t(r"\P{Letter}", "u", "5", 0).is_some());
        let caps = t(r"\p{Nd}+", "u", "123", 0).unwrap();
        assert_eq!(whole(&caps), (0, 3));
        assert!(t(r"\p{Script=Greek}", "u", "α", 0).is_some());
        assert!(t(r"\p{Script=Greek}", "u", "a", 0).is_none());
        assert!(t(r"\p{ASCII}", "u", "z", 0).is_some());
        // \p requires u/v; otherwise it is an identity escape.
        let caps = t(r"\p{L}", "", "p{L}", 0).unwrap();
        assert_eq!(whole(&caps), (0, 4));
    }

    #[test]
    fn escapes_and_octal() {
        // \101 is octal 'A' in non-unicode mode.
        let caps = t(r"\101", "", "A", 0).unwrap();
        assert_eq!(whole(&caps), (0, 1));
        // \1 with a group is a backref; without one it is octal.
        assert!(t(r"(a)\1", "", "aa", 0).is_some());
        let caps = t(r"\1", "", "\u{1}", 0).unwrap();
        assert_eq!(whole(&caps), (0, 1));
        // Control escape, hex, unicode escapes.
        assert!(t(r"\cA", "", "\u{1}", 0).is_some());
        assert!(t(r"\x41", "", "A", 0).is_some());
        assert!(t(r"\u0041", "", "A", 0).is_some());
        assert!(t(r"\u{41}", "u", "A", 0).is_some());
        // Identity escapes in non-unicode mode.
        let caps = t(r"\+", "", "+", 0).unwrap();
        assert_eq!(whole(&caps), (0, 1));
    }

    #[test]
    fn parse_errors() {
        assert!(matches!(
            CompiledRegex::compile(Wtf8::from_str("(?<=a)"), ""),
            Err(RegexError::ParseError(_))
        ));
        assert!(matches!(
            CompiledRegex::compile(Wtf8::from_str("(?<!a)"), ""),
            Err(RegexError::ParseError(_))
        ));
        assert!(matches!(
            CompiledRegex::compile(Wtf8::from_str("a"), "z"),
            Err(RegexError::ParseError(_))
        ));
        assert!(matches!(
            CompiledRegex::compile(Wtf8::from_str("a"), "ii"),
            Err(RegexError::ParseError(_))
        ));
        assert!(matches!(
            CompiledRegex::compile(Wtf8::from_str("a"), "uv"),
            Err(RegexError::ParseError(_))
        ));
        assert!(matches!(
            CompiledRegex::compile(Wtf8::from_str("(a"), ""),
            Err(RegexError::ParseError(_))
        ));
        assert!(matches!(
            CompiledRegex::compile(Wtf8::from_str("a)"), ""),
            Err(RegexError::ParseError(_))
        ));
        assert!(matches!(
            CompiledRegex::compile(Wtf8::from_str(r"\p{P}"), "u"),
            Err(RegexError::ParseError(_))
        ));
        assert!(matches!(
            CompiledRegex::compile(Wtf8::from_str(r"[a[b]]"), "v"),
            Err(RegexError::ParseError(_))
        ));
        assert!(matches!(
            CompiledRegex::compile(Wtf8::from_str(r"\a"), "u"),
            Err(RegexError::ParseError(_))
        ));
        // Duplicate names allowed (first wins for \k).
        assert!(CompiledRegex::compile(Wtf8::from_str(r"(?<a>x)(?<a>y)"), "u").is_ok());
    }

    #[test]
    fn start_clamped_and_backref_completeness() {
        // start past the end clamps to the end; empty pattern still matches.
        let caps = t("", "", "ab", 99).unwrap();
        assert_eq!(whole(&caps), (2, 2));
        assert!(t("b", "", "ab", 99).is_none());
        // Backtracking explores all body alternatives of a loop:
        // (a|ab)* must be able to take "ab" then match 'c'.
        let caps = t(r"(a|ab)*c", "", "abc", 0).unwrap();
        assert_eq!(whole(&caps), (0, 3));
        let caps = t(r"(a|ab)*c", "", "aabc", 0).unwrap();
        assert_eq!(whole(&caps), (0, 4));
    }
}
