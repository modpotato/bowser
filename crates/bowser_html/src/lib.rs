//! WHATWG HTML tokenizer and tree construction, from scratch.
//!
//! [`parse`] runs the full pipeline: input bytes are decoded per WHATWG
//! "decode" (BOM sniff, `<meta charset>` prescan, UTF-8 fallback),
//! tokenized per WHATWG 13.2.5, and built into a tree per WHATWG 13.2.6
//! (insertion modes, adoption agency algorithm, foster parenting,
//! formatting elements, templates).
//!
//! The tree types here are intentionally minimal. [`parse_into`] drives a
//! [`TreeSink`] directly with the finished tree (single pass, no
//! intermediate owned tree); [`parse`] collects into the owned [`Node`]
//! tree via [`NodeSink`] for the test corpus and simple embedders, and
//! downstream crates such as `bowser_dom` sink straight into their own
//! arenas. [`parse_fragment_into`] is the fragment-parsing analogue of
//! [`parse_into`], driving a caller-supplied [`TreeSink`] with no
//! intermediate tree.

mod entities;
mod sink;
mod token;
mod tokenizer;
mod tree;

pub mod atom;

pub use sink::{NodeSink, TreeSink};
pub use tree::{
    Doctype, Element, Namespace, Node, ParseError, ParseErrorCode, QuirksMode, ScriptChild,
    ScriptElement, ScriptExecutor, ScriptHost, ShadowRoot, ShadowRootMode, SourceLoc,
    detect_encoding, parse_fragment, parse_fragment_into, parse_fragment_into_with_ancestors,
    parse_fragment_with_ancestors, parse_fragment_with_errors, parse_fragment_with_scripting,
    parse_into, parse_into_with_scripting_and_executor, parse_with_errors, parse_with_scripting,
    parse_with_scripting_and_executor, parse_with_scripting_and_executor_and_quirks,
    parse_with_scripting_and_quirks,
};

/// Parse HTML document bytes into a [`Node::Document`] tree.
///
/// Thin wrapper over [`parse_into`] with [`NodeSink`].
///
/// The input bytes are decoded per WHATWG "decode" (BOM sniff, `<meta
/// charset>` prescan, UTF-8 fallback); undecodable bytes become U+FFFD.
pub fn parse(html: &[u8]) -> Node {
    tree::parse(html)
}

/// Parse HTML document bytes like [`parse`], additionally returning the
/// document's [`QuirksMode`] as computed from its doctype
/// (WHATWG 13.2.6.4.1).
///
/// Thin wrapper over [`parse_into`] with [`NodeSink`].
pub fn parse_with_quirks(html: &[u8]) -> (Node, QuirksMode) {
    tree::parse_with_quirks(html)
}

/// Tokenizer driver for the html5lib tokenizer conformance harness
/// (`crates/bowser_html/tests/tokenizer.rs`).
///
/// `#[doc(hidden)]`: test support, not public API. Integration tests
/// cannot reach the `pub(crate)` tokenizer, so this module exposes the
/// smallest possible driver: it feeds one input string through the real
/// [`Tokenizer`](crate::tokenizer::Tokenizer) from a fixture-chosen
/// initial state and returns plain-data tokens plus tokenizer errors.
/// Nothing here alters tokenization; it only observes it.
#[doc(hidden)]
pub mod tokenizer_test_support {
    use crate::token::Token;
    use crate::tokenizer::{State, Tokenizer};
    use crate::tree::{ParseError, SourceLoc};

    /// A tokenizer token in plain-data form for conformance comparison.
    #[derive(Debug, Clone, PartialEq)]
    pub enum TestToken {
        Doctype {
            name: String,
            public: Option<String>,
            system: Option<String>,
            force_quirks: bool,
        },
        StartTag {
            name: String,
            attrs: Vec<(String, String)>,
            self_closing: bool,
        },
        EndTag {
            name: String,
        },
        Comment(String),
        Character(String),
        /// Bowser's tokenizer produces processing-instruction tokens per
        /// the current WHATWG spec 13.2.5.72-76. The html5lib fixtures
        /// predate those states and still expect a bogus comment plus an
        /// `unexpected-question-mark-instead-of-tag-name` error, which no
        /// longer exists in the spec.
        ProcessingInstruction {
            target: String,
            data: String,
        },
    }

    /// A tokenizer parse error with its 1-based source location.
    #[derive(Debug, Clone, PartialEq)]
    pub struct TestError {
        pub code: crate::tree::ParseErrorCode,
        pub line: u32,
        pub col: u32,
    }

    /// Tokenize `input` from the given initial state, returning the token
    /// stream (excluding the terminal EOF) and the tokenizer's parse
    /// errors in emission order.
    ///
    /// `initial_state` is one of `"Data"`, `"RCData"`, `"RawText"`,
    /// `"ScriptData"`, `"PlainText"`, `"CDataSection"` (anything else
    /// means `"Data"`). `last_start_tag` seeds the tokenizer's "last
    /// start tag" name for the script-data escaped states (empty means
    /// no start tag has been emitted).
    pub fn tokenize(
        input: &str,
        initial_state: &str,
        last_start_tag: &str,
    ) -> (Vec<TestToken>, Vec<TestError>) {
        let mut t = Tokenizer::new(input);
        t.set_state(match initial_state {
            "RCData" => State::Rcdata,
            "RawText" => State::Rawtext,
            "ScriptData" => State::ScriptData,
            "PlainText" => State::Plaintext,
            "CDataSection" => State::CdataSection,
            _ => State::Data,
        });
        t.set_last_start_tag(last_start_tag);
        let mut tokens = Vec::new();
        loop {
            match t.next_token() {
                Token::Eof => break,
                Token::Doctype(d) => tokens.push(TestToken::Doctype {
                    name: d.name.unwrap_or_default(),
                    public: d.public_id,
                    system: d.system_id,
                    force_quirks: d.force_quirks,
                }),
                Token::StartTag(tag) => tokens.push(TestToken::StartTag {
                    name: tag.name.as_str().to_string(),
                    attrs: tag
                        .attrs
                        .into_iter()
                        .map(|(name, value)| (name.as_str().to_string(), value))
                        .collect(),
                    self_closing: tag.self_closing,
                }),
                Token::EndTag(tag) => tokens.push(TestToken::EndTag {
                    name: tag.name.as_str().to_string(),
                }),
                Token::Comment(data) => tokens.push(TestToken::Comment(data)),
                Token::ProcessingInstruction { target, data } => {
                    tokens.push(TestToken::ProcessingInstruction { target, data })
                }
                Token::Text(text) => tokens.push(TestToken::Character(text)),
            }
        }
        let errors = t
            .take_errors()
            .into_iter()
            .map(|ParseError { code, loc }: ParseError| {
                let SourceLoc { line, col } = loc.unwrap_or(SourceLoc { line: 0, col: 0 });
                TestError { code, line, col }
            })
            .collect();
        (tokens, errors)
    }
}
