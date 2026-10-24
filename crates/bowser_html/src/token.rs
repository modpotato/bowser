//! Tokens produced by the WHATWG HTML tokenizer.

use crate::atom::{AttrName, TagName};

/// Tag token under construction / emitted by the tokenizer.
///
/// Tag and attribute names are interned [`TagName`]/[`AttrName`] atoms:
/// well-known names are a `Copy` atom comparison with zero heap use, so
/// the tokenizer's per-tag scratch buffer is only retained for unknown
/// names (custom elements, `data-*` attributes, ...).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Tag {
    /// Lowercased tag name.
    pub name: TagName,
    /// (name, value) pairs; names lowercased, duplicates already dropped.
    pub attrs: Vec<(AttrName, String)>,
    /// Whether the tag had a self-closing `/`.
    pub self_closing: bool,
}

/// A token emitted by the tokenizer (WHATWG 13.2.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Token {
    Doctype(DoctypeToken),
    StartTag(Tag),
    EndTag(Tag),
    Comment(String),
    ProcessingInstruction {
        target: String,
        data: String,
    },
    /// One maximal run of consecutive character data (WHATWG 13.2.5
    /// character-token batching): the tokenizer merges adjacent
    /// characters — plain runs, character-reference expansions, and
    /// `<`-led fallbacks — into a single token instead of emitting one
    /// token per character.
    Text(String),
    Eof,
}

/// DOCTYPE token data.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct DoctypeToken {
    pub name: Option<String>,
    pub public_id: Option<String>,
    pub system_id: Option<String>,
    pub force_quirks: bool,
}
