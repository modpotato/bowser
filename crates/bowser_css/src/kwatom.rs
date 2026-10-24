//! Interned CSS value keywords.
//!
//! [`CssValue::Keyword`] is the most common computed-value kind, and it used
//! to own a heap [`String`] per occurrence: every parsed declaration value,
//! every cascade default, and every computed-style delta entry paid one
//! allocation for its keyword text. This module interns the well-known
//! keywords into a single immutable, lock-free table: a well-known keyword
//! is a `Copy` [`KeywordAtom`] (its index in [`KEYWORDS`]), so hot-path
//! compares are one `u16` comparison with no allocation, no hashing, and no
//! synchronization. It follows the same two-tier pattern as the property
//! atoms in [`propatom`](crate::propatom): keywords outside the table
//! (arbitrary idents, space-joined compound values like `"row dense"`,
//! `var()`-substituted text) ride a heap sidecar on
//! [`Keyword::UNKNOWN`], so behavior is identical — interning is purely a
//! representation change.
//!
//! Case handling is the caller's job, matching CSS semantics: the value
//! parser ASCII-lowercases idents before interning (see
//! [`Keyword::intern_lowered`]), and every table entry is lowercase.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

/// Well-known CSS value keywords, sorted for [`KeywordAtom::lookup`]'s
/// binary search. Curated from the keywords the codebase compares against
/// plus the common CSS-wide and per-property keywords; anything missing
/// here simply interns onto the heap sidecar, so the table never needs to
/// be exhaustive to stay correct.
const KEYWORDS: &[&str] = &[
    "absolute",
    "alias",
    "all",
    "all-petite-caps",
    "all-scroll",
    "all-small-caps",
    "alternate",
    "alternate-reverse",
    "always",
    "anywhere",
    "aqua",
    "armenian",
    "auto",
    "auto-fill",
    "auto-fit",
    "avoid",
    "avoid-column",
    "avoid-page",
    "avoid-region",
    "backwards",
    "balance",
    "baseline",
    "bidi-override",
    "black",
    "blink",
    "block",
    "blue",
    "blur",
    "bolder",
    "both",
    "both-edges",
    "bottom",
    "break-all",
    "break-spaces",
    "break-word",
    "brightness",
    "brown",
    "capitalize",
    "caption",
    "cell",
    "center",
    "circle",
    "clip",
    "close-quote",
    "col-resize",
    "collapse",
    "color",
    "color-burn",
    "color-dodge",
    "column",
    "column-reverse",
    "compact",
    "condensed",
    "contain",
    "contents",
    "context-menu",
    "contrast",
    "copy",
    "counter",
    "cover",
    "crosshair",
    "currentcolor",
    "cursive",
    "cyan",
    "darken",
    "dashed",
    "decimal",
    "decimal-leading-zero",
    "default",
    "dense",
    "difference",
    "disc",
    "disclosure-closed",
    "disclosure-open",
    "dotted",
    "double",
    "drop-shadow",
    "e-resize",
    "ease",
    "ease-in",
    "ease-in-out",
    "ease-out",
    "ellipsis",
    "embed",
    "emoji",
    "end",
    "evenodd",
    "ew-resize",
    "exclusion",
    "expanded",
    "extra-condensed",
    "extra-expanded",
    "fangsong",
    "fantasy",
    "fill",
    "first",
    "fit-content",
    "fixed",
    "flat",
    "flex",
    "flex-end",
    "flex-start",
    "flow-root",
    "forwards",
    "fuchsia",
    "full-size-kana",
    "full-width",
    "georgian",
    "grab",
    "grabbing",
    "gray",
    "grayscale",
    "green",
    "grey",
    "grid",
    "groove",
    "hard-light",
    "help",
    "hidden",
    "hide",
    "horizontal-tb",
    "hue",
    "hue-rotate",
    "icon",
    "infinite",
    "inherit",
    "initial",
    "inline",
    "inline-block",
    "inline-end",
    "inline-flex",
    "inline-grid",
    "inline-size",
    "inline-start",
    "inline-table",
    "inset",
    "inside",
    "invert",
    "isolate",
    "isolate-override",
    "italic",
    "jump-both",
    "jump-end",
    "jump-none",
    "jump-start",
    "justify",
    "justify-all",
    "keep-all",
    "large",
    "larger",
    "last",
    "layout",
    "left",
    "lighten",
    "lighter",
    "lime",
    "line-through",
    "linear",
    "list-item",
    "local",
    "loose",
    "lower-alpha",
    "lower-greek",
    "lower-latin",
    "lower-roman",
    "lowercase",
    "ltr",
    "luminosity",
    "magenta",
    "mandatory",
    "manipulation",
    "manual",
    "marker",
    "maroon",
    "match-parent",
    "math",
    "max-content",
    "medium",
    "menu",
    "message-box",
    "middle",
    "min-content",
    "mixed",
    "monospace",
    "move",
    "multiply",
    "n-resize",
    "navy",
    "ne-resize",
    "nesw-resize",
    "no-close-quote",
    "no-drop",
    "no-open-quote",
    "no-repeat",
    "none",
    "nonzero",
    "normal",
    "not-allowed",
    "nowrap",
    "ns-resize",
    "nw-resize",
    "nwse-resize",
    "oblique",
    "olive",
    "opacity",
    "open-quote",
    "orange",
    "outset",
    "outside",
    "overlay",
    "overline",
    "page",
    "paint",
    "pan-down",
    "pan-left",
    "pan-right",
    "pan-up",
    "pan-x",
    "pan-y",
    "paused",
    "petite-caps",
    "pinch-zoom",
    "pink",
    "plaintext",
    "pointer",
    "pre",
    "pre-line",
    "pre-wrap",
    "preserve-3d",
    "pretty",
    "progress",
    "proximity",
    "purple",
    "recto",
    "red",
    "region",
    "relative",
    "repeat",
    "repeat-x",
    "repeat-y",
    "reverse",
    "revert",
    "revert-layer",
    "ridge",
    "right",
    "round",
    "row",
    "row-resize",
    "row-reverse",
    "rtl",
    "ruby",
    "ruby-base",
    "ruby-text",
    "run-in",
    "running",
    "s-resize",
    "safe",
    "sans-serif",
    "saturate",
    "saturation",
    "scale-down",
    "screen",
    "scroll",
    "scroll-position",
    "se-resize",
    "self-end",
    "self-start",
    "semi-condensed",
    "semi-expanded",
    "separate",
    "sepia",
    "serif",
    "show",
    "sideways",
    "sideways-lr",
    "sideways-rl",
    "silver",
    "size",
    "small",
    "small-caps",
    "small-caption",
    "smaller",
    "smooth",
    "soft-light",
    "solid",
    "space",
    "space-around",
    "space-between",
    "space-evenly",
    "span",
    "square",
    "stable",
    "start",
    "static",
    "status-bar",
    "step-end",
    "step-start",
    "sticky",
    "stretch",
    "strict",
    "style",
    "sub",
    "subgrid",
    "super",
    "sw-resize",
    "system-ui",
    "table",
    "table-caption",
    "table-cell",
    "table-column",
    "table-column-group",
    "table-footer-group",
    "table-header-group",
    "table-row",
    "table-row-group",
    "teal",
    "text",
    "text-bottom",
    "text-top",
    "thick",
    "thin",
    "titling-caps",
    "top",
    "transparent",
    "ui-monospace",
    "ui-rounded",
    "ui-sans-serif",
    "ui-serif",
    "ultra-condensed",
    "ultra-expanded",
    "underline",
    "unicase",
    "unsafe",
    "unset",
    "upper-alpha",
    "upper-greek",
    "upper-latin",
    "upper-roman",
    "uppercase",
    "upright",
    "verso",
    "vertical-lr",
    "vertical-rl",
    "vertical-text",
    "visible",
    "w-resize",
    "wait",
    "white",
    "will-change",
    "wrap",
    "wrap-reverse",
    "x",
    "x-large",
    "x-small",
    "xx-large",
    "xx-small",
    "xxx-large",
    "y",
    "yellow",
    "zoom-in",
    "zoom-out",
];

/// A CSS value keyword: a well-known [`KeywordAtom`] plus, for keywords
/// outside the atom table, the original string on a heap sidecar.
///
/// `Clone` is cheap for well-known keywords (one `u16` copied); unknown
/// keywords clone their sidecar, exactly like the `String` this type
/// replaces. Equality is by string value: two keywords are equal iff
/// their [`as_str`](Keyword::as_str) strings are equal, regardless of
/// which tier holds them.
#[derive(Clone)]
pub struct Keyword {
    atom: KeywordAtom,
    /// The original string iff `atom == KeywordAtom::UNKNOWN`.
    spill: Option<Box<str>>,
}

impl Keyword {
    /// Intern a borrowed keyword: well-known keywords become pure atoms
    /// (no allocation); anything else is copied onto the heap sidecar.
    pub fn intern(keyword: &str) -> Self {
        match KeywordAtom::lookup(keyword) {
            Some(atom) => Keyword { atom, spill: None },
            None => Keyword {
                atom: KeywordAtom::UNKNOWN,
                spill: Some(keyword.into()),
            },
        }
    }

    /// Intern, ASCII-lowercasing first when the keyword needs it. Skips
    /// the intermediate `String` when the keyword is already lowercase.
    pub fn intern_lowered(keyword: &str) -> Self {
        if keyword.as_bytes().iter().all(|b| !b.is_ascii_uppercase()) {
            return Self::intern(keyword);
        }
        Self::from(keyword.to_ascii_lowercase())
    }

    /// Intern an owned keyword, ASCII-lowercasing in place when it needs
    /// it. This is the parse fast path: an already-lowercase keyword
    /// costs no allocation at all (well-known keywords drop the `String`
    /// and keep just the atom; unknown keywords move the `String` into
    /// the sidecar), and an uppercase keyword is lowered in place rather
    /// than copied.
    pub fn intern_lowered_owned(mut keyword: String) -> Self {
        if keyword.as_bytes().iter().any(|b| b.is_ascii_uppercase()) {
            keyword.make_ascii_lowercase();
        }
        Self::from(keyword)
    }

    /// The atom for this keyword, or [`KeywordAtom::UNKNOWN`]. Comparing
    /// atoms is exactly string equality for well-known keywords:
    /// interning is deterministic and the table holds unique strings.
    pub fn atom(&self) -> KeywordAtom {
        self.atom
    }

    /// The original keyword string.
    pub fn as_str(&self) -> &str {
        match &self.spill {
            Some(s) => s,
            None => self.atom.as_str(),
        }
    }

    /// Whether this keyword spilled to the heap (outside the atom table).
    pub fn is_spilled(&self) -> bool {
        self.spill.is_some()
    }
}

impl PartialEq for Keyword {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for Keyword {}

impl Hash for Keyword {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl From<&str> for Keyword {
    fn from(s: &str) -> Self {
        Self::intern(s)
    }
}

impl From<&String> for Keyword {
    fn from(s: &String) -> Self {
        Self::intern(s.as_str())
    }
}

impl From<String> for Keyword {
    /// Zero-copy for unknown keywords (the `String` becomes the sidecar);
    /// well-known keywords drop the `String` and keep just the atom.
    fn from(s: String) -> Self {
        match KeywordAtom::lookup(&s) {
            Some(atom) => Keyword { atom, spill: None },
            None => Keyword {
                atom: KeywordAtom::UNKNOWN,
                spill: Some(s.into_boxed_str()),
            },
        }
    }
}

impl Deref for Keyword {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for Keyword {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

// `fmt::Debug` renders exactly like the `String` this type replaces
// (`"auto"`, quoted): the derived `Debug` on [`CssValue`] therefore keeps
// rendering `Keyword("auto")`, byte-identical to before (tests pin this
// rendering — see `font_parity.rs`).
impl fmt::Debug for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl fmt::Display for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<str> for Keyword {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for Keyword {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// An interned well-known CSS value keyword: its index in [`KEYWORDS`].
/// `Copy`, so comparing keywords is one `u16` comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KeywordAtom(u16);

impl KeywordAtom {
    /// The atom for keywords outside the table (carried on the heap
    /// sidecar instead).
    pub const UNKNOWN: KeywordAtom = KeywordAtom(u16::MAX);

    /// Look up a keyword's atom. The table is sorted, so this is a
    /// binary search: no allocation, no hashing, no synchronization.
    pub fn lookup(keyword: &str) -> Option<KeywordAtom> {
        KEYWORDS
            .binary_search(&keyword)
            .ok()
            .map(|i| KeywordAtom(i as u16))
    }

    /// The keyword string for this atom. Panics for
    /// [`KeywordAtom::UNKNOWN`], which has no table entry.
    pub fn as_str(self) -> &'static str {
        KEYWORDS[self.0 as usize]
    }

    /// The table index.
    pub fn index(self) -> u16 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_sorted_unique_and_lookup_consistent() {
        assert!(!KEYWORDS.is_empty());
        assert!(KEYWORDS.len() < u16::MAX as usize);
        let mut prev = "";
        for name in KEYWORDS {
            assert!(*name > prev, "table not sorted at {name}");
            prev = name;
            let atom = KeywordAtom::lookup(name).unwrap_or_else(|| panic!("no lookup for {name}"));
            assert_eq!(atom.as_str(), *name);
        }
        assert_eq!(KeywordAtom::lookup("not-a-keyword"), None);
        assert_eq!(KeywordAtom::lookup("AUTO"), None);
        assert_eq!(KeywordAtom::lookup(""), None);
    }

    #[test]
    fn keyword_interning_identity() {
        // Two interns of the same well-known keyword share one atom and
        // allocate nothing.
        let a = Keyword::intern("auto");
        let b = Keyword::intern("auto");
        assert_eq!(a, b);
        assert_eq!(a.atom(), b.atom());
        assert!(!a.is_spilled());
        assert_eq!(a.as_str(), "auto");

        // Unknown keywords keep their exact string on the sidecar and
        // still compare by value.
        let u1 = Keyword::intern("not-a-real-keyword");
        let u2 = Keyword::intern("not-a-real-keyword");
        assert_eq!(u1.atom(), KeywordAtom::UNKNOWN);
        assert!(u1.is_spilled());
        assert_eq!(u1, u2);
        assert_ne!(u1, a);

        // Owned interning drops the String for tabled keywords and
        // reuses it for spilled ones.
        let o = Keyword::from("none".to_string());
        assert_eq!(o.atom(), KeywordAtom::lookup("none").unwrap());
        assert!(!o.is_spilled());
        let o2 = Keyword::intern_lowered_owned("AUTO".to_string());
        assert_eq!(o2.as_str(), "auto");
        assert_eq!(o2.atom(), KeywordAtom::lookup("auto").unwrap());
    }

    #[test]
    fn keyword_debug_shape() {
        // Pinned: renders exactly like the old `String` payload, so
        // `CssValue`'s derived `Debug` keeps rendering `Keyword("medium")`
        // (see `font_parity.rs`).
        assert_eq!(format!("{:?}", Keyword::intern("medium")), "\"medium\"");
        assert_eq!(
            format!("{:?}", crate::CssValue::Keyword(Keyword::intern("medium"))),
            "Keyword(\"medium\")"
        );
    }
}
