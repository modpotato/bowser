//! html5lib tokenizer conformance harness.
//!
//! Fixtures: `tests/fixtures/html5lib-tokenizer/*.test`, vendored
//! 2026-10-04 from html5lib/html5lib-tests at commit `9329e64` (the same
//! commit the tree-construction corpus in `tests/corpus.rs` was vendored
//! from; 15 files, 6810 cases).
//!
//! Format (see `fixtures/html5lib-tokenizer/README.md`): JSON, one
//! `{"tests": [...]}` object per file (`xmlViolation.test` uses
//! `"xmlViolationTests"`). Each case carries `description`, `input`,
//! `output` (the complete expected token list), and optionally
//! `initialStates` (default `["Data state"]`; the case runs once per
//! state), `lastStartTag`, `doubleEscaped` (an extra `\uHHHH`
//! unescaping round applied to input and output), and `errors`
//! (`code`/`line`/`col` per error).
//!
//! The harness drives the real tokenizer through
//! [`bowser_html::tokenizer_test_support::tokenize`] and compares:
//!
//! - the token stream, exactly. Both sides coalesce character runs:
//!   the fixture format merges adjacent `Character` tokens, and our
//!   tokenizer batches maximal text runs into one [`TestToken`]
//!   (`Token::Text`);
//! - the parse-error stream: the same *count* of errors in the same
//!   order, with each expected code covered by our code's family
//!   ([`error_family`]). Error *locations* are deliberately not
//!   compared: WHATWG 13.2.2 does not define them, and the two
//!   implementations point differently by design (html5lib reports at
//!   its read position / the EOF position; we report at the last
//!   consumed character, see `tokenizer_error`). Gating on locations
//!   would enshrine one arbitrary convention.
//!
//! The gate mirrors `tests/corpus.rs`: every case run must pass or be
//! listed in [`KNOWN_DEVIATIONS`] with a reason; a listed run that
//! starts passing fails too, so the list cannot go stale.
//!
//! [`TestToken`]: bowser_html::tokenizer_test_support::TestToken
//! [`ParseErrorCode`]: bowser_html::ParseErrorCode

use bowser_html::ParseErrorCode;
use bowser_html::tokenizer_test_support::{TestError, TestToken, tokenize};

// ---------------------------------------------------------------------------
// Known deviations: (fixture file, case index, initial state, reason).
//
// Each entry is a precisely understood, documented difference — never a
// swept-under-the-rug failure. `state` is the driver state name
// ("Data", "RCData", "RawText", "ScriptData", "PlainText",
// "CDataSection"), or "" to cover all of the case's states.
// ---------------------------------------------------------------------------

/// Shared reason for the processing-instruction cases below: the
/// fixtures predate the WHATWG processing-instruction tokenizer
/// states. Current spec 13.2.5.6 routes `?` in the tag open state to
/// the processing instruction open state (13.2.5.72-76), and
/// `unexpected-question-mark-instead-of-tag-name` no longer exists in
/// the spec. Bowser implements the current states; the fixtures still
/// expect a bogus comment plus the removed error.
const PI_STALE: &str = "stale fixture: spec 13.2.5.72-76 defines PI states; \
    unexpected-question-mark-instead-of-tag-name was removed";

// ---------------------------------------------------------------------------
// Shared reasons for the error-code emission gaps below (wave-32 triage).
//
// Every case in these groups produces exactly the spec's token stream;
// only parse-error *code emissions* are missing. Each constant names one
// gap family; the entries reference them per case. Verified 2026-10-04 by
// probing all 842 failing runs: zero token differences, zero spurious
// emissions (every emitted error's code family appears in the fixture's
// expected list), and the tokenizer always emits strictly fewer errors
// than the fixture expects.
// ---------------------------------------------------------------------------

/// `control-character-in-input-stream` / `noncharacter-in-input-stream`:
/// WHATWG input-stream preprocessing errors with no `ParseErrorCode`
/// variant (preprocessing is not implemented).
const ERR_INPUT_STREAM: &str = "error-code emission not yet implemented: \
    input-stream preprocessing errors (control-character-in-input-stream, \
    noncharacter-in-input-stream) have no ParseErrorCode variant; token stream matches spec";

/// `eof-in-cdata`: no `ParseErrorCode` variant.
const ERR_EOF_CDATA: &str = "error-code emission not yet implemented: \
    eof-in-cdata has no ParseErrorCode variant; token stream matches spec";

/// `eof-in-script-html-comment-like-text`: no `ParseErrorCode` variant.
const ERR_EOF_SCRIPT_LIKE: &str = "error-code emission not yet implemented: \
    eof-in-script-html-comment-like-text has no ParseErrorCode variant; token stream matches spec";

/// `duplicate-attribute`: no `ParseErrorCode` variant; the duplicate is
/// silently dropped per 13.2.5.34.
const ERR_DUP_ATTR: &str = "error-code emission not yet implemented: \
    duplicate-attribute has no ParseErrorCode variant (duplicate silently \
    dropped per 13.2.5.34); token stream matches spec";

/// `end-tag-with-attributes`: no `ParseErrorCode` variant.
const ERR_END_TAG_ATTRS: &str = "error-code emission not yet implemented: \
    end-tag-with-attributes has no ParseErrorCode variant; token stream matches spec";

/// `nested-comment`: no `ParseErrorCode` variant.
const ERR_NESTED_COMMENT: &str = "error-code emission not yet implemented: \
    nested-comment has no ParseErrorCode variant; token stream matches spec";

/// `incorrectly-opened-comment` when EOF follows `<!` in the markup
/// declaration open state. `ParseErrorCode::IncorrectlyOpenedComment`
/// exists and is emitted on the non-EOF paths; only the EOF path skips it.
const ERR_COMMENT_OPEN_EOF: &str = "error-code emission not yet implemented: \
    incorrectly-opened-comment not emitted when EOF follows `<!` \
    (ParseErrorCode::IncorrectlyOpenedComment exists); token stream matches spec";

/// `incorrectly-closed-comment`: no `ParseErrorCode` variant.
const ERR_COMMENT_CLOSE: &str = "error-code emission not yet implemented: \
    incorrectly-closed-comment has no ParseErrorCode variant; token stream matches spec";

/// `end-tag-with-trailing-solidus`: no `ParseErrorCode` variant.
const ERR_END_TAG_SOLIDUS: &str = "error-code emission not yet implemented: \
    end-tag-with-trailing-solidus has no ParseErrorCode variant; token stream matches spec";

/// DOCTYPE error gaps: non-quirks DOCTYPE errors are never emitted
/// individually (at most one `InvalidDoctype` per DOCTYPE where the spec
/// yields 2-4 errors), and `unexpected-null-character` is not emitted in
/// the DOCTYPE name/identifier states (the sibling wave-32 DOCTYPE rework
/// owns those states).
const ERR_DOCTYPE: &str = "error-code emission not yet implemented: \
    non-quirks DOCTYPE errors collapse to at most one InvalidDoctype where \
    the spec yields 2-4, and unexpected-null-character is missing in \
    DOCTYPE-name/identifier paths; token stream matches spec";

/// Cases missing both an input-stream preprocessing error and DOCTYPE errors.
const ERR_DOCTYPE_AND_INPUT_STREAM: &str = "error-code emission not yet implemented: \
    input-stream preprocessing error (no ParseErrorCode variant) plus DOCTYPE error \
    gaps (see ERR_DOCTYPE); token stream matches spec";

/// `duplicate-attribute` plus `end-tag-with-attributes` in one case.
const ERR_DUP_ATTR_AND_END_TAG_ATTRS: &str = "error-code emission not yet implemented: \
    duplicate-attribute and end-tag-with-attributes have no ParseErrorCode \
    variants; token stream matches spec";

/// `control-character-in-input-stream` plus `eof-in-cdata` in one case.
const ERR_INPUT_STREAM_AND_EOF_CDATA: &str = "error-code emission not yet implemented: \
    control-character-in-input-stream and eof-in-cdata have no ParseErrorCode \
    variants; token stream matches spec";

const KNOWN_DEVIATIONS: &[(&str, usize, &str, &str)] = &[
    ("domjs.test", 0, "", PI_STALE),
    ("domjs.test", 1, "", PI_STALE),
    ("domjs.test", 2, "", PI_STALE),
    ("test2.test", 31, "", PI_STALE),
    ("test2.test", 32, "", PI_STALE),
    ("test3.test", 1158, "", PI_STALE),
    ("test3.test", 1159, "", PI_STALE),
    ("test3.test", 1160, "", PI_STALE),
    ("test3.test", 1161, "", PI_STALE),
    ("test3.test", 1162, "", PI_STALE),
    ("test3.test", 1163, "", PI_STALE),
    ("test3.test", 1164, "", PI_STALE),
    ("test3.test", 1165, "", PI_STALE),
    ("test3.test", 1166, "", PI_STALE),
    ("test3.test", 1167, "", PI_STALE),
    ("test3.test", 1168, "", PI_STALE),
    ("test3.test", 1169, "", PI_STALE),
    ("test3.test", 1170, "", PI_STALE),
    ("test3.test", 1171, "", PI_STALE),
    ("test3.test", 1172, "", PI_STALE),
    ("test3.test", 1173, "", PI_STALE),
    ("test3.test", 1174, "", PI_STALE),
    ("test3.test", 1175, "", PI_STALE),
    ("test3.test", 1176, "", PI_STALE),
    ("test3.test", 1177, "", PI_STALE),
    ("test3.test", 1178, "", PI_STALE),
    ("test3.test", 1179, "", PI_STALE),
    ("test3.test", 1180, "", PI_STALE),
    ("test3.test", 1181, "", PI_STALE),
    ("test3.test", 1182, "", PI_STALE),
    ("test3.test", 1183, "", PI_STALE),
    ("test3.test", 1184, "", PI_STALE),
    ("test3.test", 1185, "", PI_STALE),
    ("test3.test", 1186, "", PI_STALE),
    ("test3.test", 1187, "", PI_STALE),
    ("test3.test", 1188, "", PI_STALE),
    ("test3.test", 1189, "", PI_STALE),
    ("test3.test", 1190, "", PI_STALE),
    // xmlViolation: the fixture README states the expected output
    // assumes the spec's "Coercing an HTML DOM into an infoset" tweaks
    // (non-XML chars to U+FFFD, U+000C to space, "--" to "- -" in
    // comments). Bowser is an HTML engine producing a DOM, not an
    // infoset; applying the coercion would corrupt real text, so the
    // HTML-correct (uncoerced) tokens are kept.
    (
        "xmlViolation.test",
        0,
        "",
        "infoset coercion: U+FFFF must round-trip in HTML, not become U+FFFD",
    ),
    (
        "xmlViolation.test",
        1,
        "",
        "infoset coercion: U+000C is emitted as-is in HTML text, not coerced to space",
    ),
    (
        "xmlViolation.test",
        2,
        "",
        "infoset coercion: HTML comments keep \"--\", not \"- -\"",
    ),
    // domjs "leading U+FEFF must pass through": the fixture predates the
    // WHATWG "Preprocessing the input stream" requirement to strip one
    // leading U+FEFF (its own README demands preprocessing). Bowser
    // follows the spec.
    (
        "domjs.test",
        24,
        "",
        "stale fixture: WHATWG preprocessing strips one leading U+FEFF",
    ),
    // unicodeCharsProblematic: unpaired surrogates cannot exist in a Rust
    // `&str` (the driver's input type); WHATWG decode replaces them with
    // U+FFFD before tokenization, so `surrogate-in-input-stream` is
    // un-emittable here. The harness maps them to U+FFFD when decoding
    // the fixture JSON, matching decode behavior.
    (
        "unicodeCharsProblematic.test",
        0,
        "",
        "unpaired surrogates unrepresentable in &str; decode yields U+FFFD",
    ),
    (
        "unicodeCharsProblematic.test",
        1,
        "",
        "unpaired surrogates unrepresentable in &str; decode yields U+FFFD",
    ),
    (
        "unicodeCharsProblematic.test",
        2,
        "",
        "unpaired surrogates unrepresentable in &str; decode yields U+FFFD",
    ),
    (
        "unicodeCharsProblematic.test",
        3,
        "",
        "unpaired surrogates unrepresentable in &str; decode yields U+FFFD",
    ),
    // ERR_INPUT_STREAM: 146 cases.
    ("test3.test", 6, "", ERR_INPUT_STREAM),
    ("test3.test", 72, "", ERR_INPUT_STREAM),
    ("test3.test", 79, "", ERR_INPUT_STREAM),
    ("test3.test", 92, "", ERR_INPUT_STREAM),
    ("test3.test", 98, "", ERR_INPUT_STREAM),
    ("test3.test", 110, "", ERR_INPUT_STREAM),
    ("test3.test", 170, "", ERR_INPUT_STREAM),
    ("test3.test", 182, "", ERR_INPUT_STREAM),
    ("test3.test", 291, "", ERR_INPUT_STREAM),
    ("test3.test", 294, "", ERR_INPUT_STREAM),
    ("test3.test", 297, "", ERR_INPUT_STREAM),
    ("test3.test", 321, "", ERR_INPUT_STREAM),
    ("test3.test", 324, "", ERR_INPUT_STREAM),
    ("test3.test", 327, "", ERR_INPUT_STREAM),
    ("test3.test", 330, "", ERR_INPUT_STREAM),
    ("test3.test", 333, "", ERR_INPUT_STREAM),
    ("test3.test", 336, "", ERR_INPUT_STREAM),
    ("test3.test", 356, "", ERR_INPUT_STREAM),
    ("test3.test", 359, "", ERR_INPUT_STREAM),
    ("test3.test", 362, "", ERR_INPUT_STREAM),
    ("test3.test", 501, "", ERR_INPUT_STREAM),
    ("test3.test", 504, "", ERR_INPUT_STREAM),
    ("test3.test", 507, "", ERR_INPUT_STREAM),
    ("test3.test", 643, "", ERR_INPUT_STREAM),
    ("test3.test", 1122, "", ERR_INPUT_STREAM),
    ("test3.test", 1200, "", ERR_INPUT_STREAM),
    ("test3.test", 1203, "", ERR_INPUT_STREAM),
    ("test3.test", 1206, "", ERR_INPUT_STREAM),
    ("test3.test", 1209, "", ERR_INPUT_STREAM),
    ("test3.test", 1212, "", ERR_INPUT_STREAM),
    ("test3.test", 1215, "", ERR_INPUT_STREAM),
    ("test3.test", 1242, "", ERR_INPUT_STREAM),
    ("test3.test", 1245, "", ERR_INPUT_STREAM),
    ("test3.test", 1248, "", ERR_INPUT_STREAM),
    ("test3.test", 1251, "", ERR_INPUT_STREAM),
    ("test3.test", 1254, "", ERR_INPUT_STREAM),
    ("test3.test", 1257, "", ERR_INPUT_STREAM),
    ("test3.test", 1303, "", ERR_INPUT_STREAM),
    ("test3.test", 1306, "", ERR_INPUT_STREAM),
    ("test3.test", 1309, "", ERR_INPUT_STREAM),
    ("test3.test", 1316, "", ERR_INPUT_STREAM),
    ("test3.test", 1353, "", ERR_INPUT_STREAM),
    ("test3.test", 1362, "", ERR_INPUT_STREAM),
    ("test3.test", 1365, "", ERR_INPUT_STREAM),
    ("test3.test", 1368, "", ERR_INPUT_STREAM),
    ("test3.test", 1436, "", ERR_INPUT_STREAM),
    ("test3.test", 1439, "", ERR_INPUT_STREAM),
    ("test3.test", 1442, "", ERR_INPUT_STREAM),
    ("test3.test", 1507, "", ERR_INPUT_STREAM),
    ("test4.test", 59, "", ERR_INPUT_STREAM),
    ("test4.test", 60, "", ERR_INPUT_STREAM),
    ("test4.test", 61, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 0, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 1, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 2, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 3, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 4, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 5, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 6, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 7, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 8, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 9, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 10, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 11, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 12, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 13, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 14, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 15, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 16, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 17, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 18, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 19, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 20, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 21, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 22, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 23, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 24, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 25, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 26, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 27, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 28, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 29, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 30, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 31, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 32, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 33, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 34, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 35, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 36, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 37, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 38, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 39, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 40, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 41, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 42, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 43, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 44, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 45, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 46, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 47, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 48, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 49, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 50, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 51, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 52, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 53, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 54, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 55, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 56, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 57, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 58, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 59, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 60, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 61, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 62, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 63, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 64, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 65, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 66, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 67, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 68, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 69, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 70, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 71, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 72, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 73, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 74, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 75, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 76, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 77, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 78, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 79, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 80, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 81, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 82, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 83, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 84, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 85, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 86, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 87, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 88, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 89, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 90, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 91, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 92, "", ERR_INPUT_STREAM),
    ("unicodeChars.test", 93, "", ERR_INPUT_STREAM),
    // ERR_EOF_CDATA: 51 cases.
    ("domjs.test", 39, "", ERR_EOF_CDATA),
    ("domjs.test", 40, "", ERR_EOF_CDATA),
    ("domjs.test", 41, "", ERR_EOF_CDATA),
    ("test3.test", 1, "", ERR_EOF_CDATA),
    ("test3.test", 3, "", ERR_EOF_CDATA),
    ("test3.test", 5, "", ERR_EOF_CDATA),
    ("test3.test", 9, "", ERR_EOF_CDATA),
    ("test3.test", 11, "", ERR_EOF_CDATA),
    ("test3.test", 13, "", ERR_EOF_CDATA),
    ("test3.test", 15, "", ERR_EOF_CDATA),
    ("test3.test", 17, "", ERR_EOF_CDATA),
    ("test3.test", 19, "", ERR_EOF_CDATA),
    ("test3.test", 21, "", ERR_EOF_CDATA),
    ("test3.test", 23, "", ERR_EOF_CDATA),
    ("test3.test", 25, "", ERR_EOF_CDATA),
    ("test3.test", 27, "", ERR_EOF_CDATA),
    ("test3.test", 29, "", ERR_EOF_CDATA),
    ("test3.test", 31, "", ERR_EOF_CDATA),
    ("test3.test", 33, "", ERR_EOF_CDATA),
    ("test3.test", 35, "", ERR_EOF_CDATA),
    ("test3.test", 37, "", ERR_EOF_CDATA),
    ("test3.test", 39, "", ERR_EOF_CDATA),
    ("test3.test", 41, "", ERR_EOF_CDATA),
    ("test3.test", 43, "", ERR_EOF_CDATA),
    ("test3.test", 45, "", ERR_EOF_CDATA),
    ("test3.test", 47, "", ERR_EOF_CDATA),
    ("test3.test", 49, "", ERR_EOF_CDATA),
    ("test3.test", 51, "", ERR_EOF_CDATA),
    ("test3.test", 53, "", ERR_EOF_CDATA),
    ("test3.test", 55, "", ERR_EOF_CDATA),
    ("test3.test", 57, "", ERR_EOF_CDATA),
    ("test3.test", 59, "", ERR_EOF_CDATA),
    ("test3.test", 61, "", ERR_EOF_CDATA),
    ("test3.test", 63, "", ERR_EOF_CDATA),
    ("test3.test", 65, "", ERR_EOF_CDATA),
    ("test3.test", 67, "", ERR_EOF_CDATA),
    ("test3.test", 1561, "", ERR_EOF_CDATA),
    ("test3.test", 1563, "", ERR_EOF_CDATA),
    ("test3.test", 1565, "", ERR_EOF_CDATA),
    ("test3.test", 1567, "", ERR_EOF_CDATA),
    ("test3.test", 1569, "", ERR_EOF_CDATA),
    ("test3.test", 1571, "", ERR_EOF_CDATA),
    ("test3.test", 1573, "", ERR_EOF_CDATA),
    ("test3.test", 1575, "", ERR_EOF_CDATA),
    ("test3.test", 1577, "", ERR_EOF_CDATA),
    ("test3.test", 1579, "", ERR_EOF_CDATA),
    ("test3.test", 1581, "", ERR_EOF_CDATA),
    ("test3.test", 1583, "", ERR_EOF_CDATA),
    ("test3.test", 1585, "", ERR_EOF_CDATA),
    ("test3.test", 1587, "", ERR_EOF_CDATA),
    ("test3.test", 1589, "", ERR_EOF_CDATA),
    // ERR_EOF_SCRIPT_LIKE: 6 cases.
    ("domjs.test", 7, "", ERR_EOF_SCRIPT_LIKE),
    ("domjs.test", 8, "", ERR_EOF_SCRIPT_LIKE),
    ("domjs.test", 9, "", ERR_EOF_SCRIPT_LIKE),
    ("domjs.test", 10, "", ERR_EOF_SCRIPT_LIKE),
    ("domjs.test", 11, "", ERR_EOF_SCRIPT_LIKE),
    ("domjs.test", 12, "", ERR_EOF_SCRIPT_LIKE),
    // ERR_DUP_ATTR: 6 cases.
    ("test1.test", 16, "", ERR_DUP_ATTR),
    ("test3.test", 1276, "", ERR_DUP_ATTR),
    ("test3.test", 1282, "", ERR_DUP_ATTR),
    ("test3.test", 1385, "", ERR_DUP_ATTR),
    ("test3.test", 1390, "", ERR_DUP_ATTR),
    ("test4.test", 47, "", ERR_DUP_ATTR),
    // ERR_END_TAG_ATTRS: 2 cases.
    ("test1.test", 13, "", ERR_END_TAG_ATTRS),
    ("test4.test", 48, "", ERR_END_TAG_ATTRS),
    // ERR_NESTED_COMMENT: 2 cases.
    ("test1.test", 31, "", ERR_NESTED_COMMENT),
    ("test1.test", 32, "", ERR_NESTED_COMMENT),
    // ERR_COMMENT_OPEN_EOF: 1 cases.
    ("test3.test", 75, "", ERR_COMMENT_OPEN_EOF),
    // ERR_COMMENT_CLOSE: 1 cases.
    ("test3.test", 191, "", ERR_COMMENT_CLOSE),
    // ERR_END_TAG_SOLIDUS: 1 cases.
    ("test4.test", 52, "", ERR_END_TAG_SOLIDUS),
    // ERR_DOCTYPE: 584 cases.
    ("test2.test", 1, "", ERR_DOCTYPE),
    ("test2.test", 2, "", ERR_DOCTYPE),
    ("test3.test", 281, "", ERR_DOCTYPE),
    ("test3.test", 290, "", ERR_DOCTYPE),
    ("test3.test", 320, "", ERR_DOCTYPE),
    ("test3.test", 329, "", ERR_DOCTYPE),
    ("test3.test", 355, "", ERR_DOCTYPE),
    ("test3.test", 365, "", ERR_DOCTYPE),
    ("test3.test", 366, "", ERR_DOCTYPE),
    ("test3.test", 367, "", ERR_DOCTYPE),
    ("test3.test", 368, "", ERR_DOCTYPE),
    ("test3.test", 370, "", ERR_DOCTYPE),
    ("test3.test", 371, "", ERR_DOCTYPE),
    ("test3.test", 372, "", ERR_DOCTYPE),
    ("test3.test", 373, "", ERR_DOCTYPE),
    ("test3.test", 374, "", ERR_DOCTYPE),
    ("test3.test", 375, "", ERR_DOCTYPE),
    ("test3.test", 376, "", ERR_DOCTYPE),
    ("test3.test", 377, "", ERR_DOCTYPE),
    ("test3.test", 378, "", ERR_DOCTYPE),
    ("test3.test", 379, "", ERR_DOCTYPE),
    ("test3.test", 380, "", ERR_DOCTYPE),
    ("test3.test", 381, "", ERR_DOCTYPE),
    ("test3.test", 382, "", ERR_DOCTYPE),
    ("test3.test", 383, "", ERR_DOCTYPE),
    ("test3.test", 384, "", ERR_DOCTYPE),
    ("test3.test", 385, "", ERR_DOCTYPE),
    ("test3.test", 386, "", ERR_DOCTYPE),
    ("test3.test", 387, "", ERR_DOCTYPE),
    ("test3.test", 388, "", ERR_DOCTYPE),
    ("test3.test", 389, "", ERR_DOCTYPE),
    ("test3.test", 390, "", ERR_DOCTYPE),
    ("test3.test", 391, "", ERR_DOCTYPE),
    ("test3.test", 392, "", ERR_DOCTYPE),
    ("test3.test", 393, "", ERR_DOCTYPE),
    ("test3.test", 394, "", ERR_DOCTYPE),
    ("test3.test", 395, "", ERR_DOCTYPE),
    ("test3.test", 396, "", ERR_DOCTYPE),
    ("test3.test", 397, "", ERR_DOCTYPE),
    ("test3.test", 398, "", ERR_DOCTYPE),
    ("test3.test", 399, "", ERR_DOCTYPE),
    ("test3.test", 402, "", ERR_DOCTYPE),
    ("test3.test", 403, "", ERR_DOCTYPE),
    ("test3.test", 404, "", ERR_DOCTYPE),
    ("test3.test", 405, "", ERR_DOCTYPE),
    ("test3.test", 407, "", ERR_DOCTYPE),
    ("test3.test", 408, "", ERR_DOCTYPE),
    ("test3.test", 409, "", ERR_DOCTYPE),
    ("test3.test", 410, "", ERR_DOCTYPE),
    ("test3.test", 411, "", ERR_DOCTYPE),
    ("test3.test", 412, "", ERR_DOCTYPE),
    ("test3.test", 413, "", ERR_DOCTYPE),
    ("test3.test", 415, "", ERR_DOCTYPE),
    ("test3.test", 416, "", ERR_DOCTYPE),
    ("test3.test", 418, "", ERR_DOCTYPE),
    ("test3.test", 419, "", ERR_DOCTYPE),
    ("test3.test", 421, "", ERR_DOCTYPE),
    ("test3.test", 422, "", ERR_DOCTYPE),
    ("test3.test", 423, "", ERR_DOCTYPE),
    ("test3.test", 424, "", ERR_DOCTYPE),
    ("test3.test", 425, "", ERR_DOCTYPE),
    ("test3.test", 426, "", ERR_DOCTYPE),
    ("test3.test", 427, "", ERR_DOCTYPE),
    ("test3.test", 428, "", ERR_DOCTYPE),
    ("test3.test", 429, "", ERR_DOCTYPE),
    ("test3.test", 430, "", ERR_DOCTYPE),
    ("test3.test", 431, "", ERR_DOCTYPE),
    ("test3.test", 432, "", ERR_DOCTYPE),
    ("test3.test", 433, "", ERR_DOCTYPE),
    ("test3.test", 434, "", ERR_DOCTYPE),
    ("test3.test", 435, "", ERR_DOCTYPE),
    ("test3.test", 436, "", ERR_DOCTYPE),
    ("test3.test", 437, "", ERR_DOCTYPE),
    ("test3.test", 438, "", ERR_DOCTYPE),
    ("test3.test", 439, "", ERR_DOCTYPE),
    ("test3.test", 440, "", ERR_DOCTYPE),
    ("test3.test", 441, "", ERR_DOCTYPE),
    ("test3.test", 442, "", ERR_DOCTYPE),
    ("test3.test", 443, "", ERR_DOCTYPE),
    ("test3.test", 444, "", ERR_DOCTYPE),
    ("test3.test", 445, "", ERR_DOCTYPE),
    ("test3.test", 446, "", ERR_DOCTYPE),
    ("test3.test", 447, "", ERR_DOCTYPE),
    ("test3.test", 448, "", ERR_DOCTYPE),
    ("test3.test", 449, "", ERR_DOCTYPE),
    ("test3.test", 450, "", ERR_DOCTYPE),
    ("test3.test", 451, "", ERR_DOCTYPE),
    ("test3.test", 452, "", ERR_DOCTYPE),
    ("test3.test", 453, "", ERR_DOCTYPE),
    ("test3.test", 454, "", ERR_DOCTYPE),
    ("test3.test", 455, "", ERR_DOCTYPE),
    ("test3.test", 456, "", ERR_DOCTYPE),
    ("test3.test", 457, "", ERR_DOCTYPE),
    ("test3.test", 458, "", ERR_DOCTYPE),
    ("test3.test", 459, "", ERR_DOCTYPE),
    ("test3.test", 460, "", ERR_DOCTYPE),
    ("test3.test", 461, "", ERR_DOCTYPE),
    ("test3.test", 462, "", ERR_DOCTYPE),
    ("test3.test", 463, "", ERR_DOCTYPE),
    ("test3.test", 464, "", ERR_DOCTYPE),
    ("test3.test", 465, "", ERR_DOCTYPE),
    ("test3.test", 466, "", ERR_DOCTYPE),
    ("test3.test", 467, "", ERR_DOCTYPE),
    ("test3.test", 468, "", ERR_DOCTYPE),
    ("test3.test", 469, "", ERR_DOCTYPE),
    ("test3.test", 470, "", ERR_DOCTYPE),
    ("test3.test", 471, "", ERR_DOCTYPE),
    ("test3.test", 472, "", ERR_DOCTYPE),
    ("test3.test", 473, "", ERR_DOCTYPE),
    ("test3.test", 474, "", ERR_DOCTYPE),
    ("test3.test", 498, "", ERR_DOCTYPE),
    ("test3.test", 499, "", ERR_DOCTYPE),
    ("test3.test", 500, "", ERR_DOCTYPE),
    ("test3.test", 510, "", ERR_DOCTYPE),
    ("test3.test", 511, "", ERR_DOCTYPE),
    ("test3.test", 512, "", ERR_DOCTYPE),
    ("test3.test", 513, "", ERR_DOCTYPE),
    ("test3.test", 515, "", ERR_DOCTYPE),
    ("test3.test", 516, "", ERR_DOCTYPE),
    ("test3.test", 517, "", ERR_DOCTYPE),
    ("test3.test", 518, "", ERR_DOCTYPE),
    ("test3.test", 519, "", ERR_DOCTYPE),
    ("test3.test", 520, "", ERR_DOCTYPE),
    ("test3.test", 521, "", ERR_DOCTYPE),
    ("test3.test", 522, "", ERR_DOCTYPE),
    ("test3.test", 523, "", ERR_DOCTYPE),
    ("test3.test", 524, "", ERR_DOCTYPE),
    ("test3.test", 525, "", ERR_DOCTYPE),
    ("test3.test", 526, "", ERR_DOCTYPE),
    ("test3.test", 527, "", ERR_DOCTYPE),
    ("test3.test", 528, "", ERR_DOCTYPE),
    ("test3.test", 529, "", ERR_DOCTYPE),
    ("test3.test", 530, "", ERR_DOCTYPE),
    ("test3.test", 531, "", ERR_DOCTYPE),
    ("test3.test", 532, "", ERR_DOCTYPE),
    ("test3.test", 533, "", ERR_DOCTYPE),
    ("test3.test", 534, "", ERR_DOCTYPE),
    ("test3.test", 535, "", ERR_DOCTYPE),
    ("test3.test", 536, "", ERR_DOCTYPE),
    ("test3.test", 537, "", ERR_DOCTYPE),
    ("test3.test", 538, "", ERR_DOCTYPE),
    ("test3.test", 539, "", ERR_DOCTYPE),
    ("test3.test", 540, "", ERR_DOCTYPE),
    ("test3.test", 541, "", ERR_DOCTYPE),
    ("test3.test", 542, "", ERR_DOCTYPE),
    ("test3.test", 545, "", ERR_DOCTYPE),
    ("test3.test", 546, "", ERR_DOCTYPE),
    ("test3.test", 547, "", ERR_DOCTYPE),
    ("test3.test", 548, "", ERR_DOCTYPE),
    ("test3.test", 550, "", ERR_DOCTYPE),
    ("test3.test", 551, "", ERR_DOCTYPE),
    ("test3.test", 552, "", ERR_DOCTYPE),
    ("test3.test", 553, "", ERR_DOCTYPE),
    ("test3.test", 554, "", ERR_DOCTYPE),
    ("test3.test", 555, "", ERR_DOCTYPE),
    ("test3.test", 556, "", ERR_DOCTYPE),
    ("test3.test", 558, "", ERR_DOCTYPE),
    ("test3.test", 559, "", ERR_DOCTYPE),
    ("test3.test", 561, "", ERR_DOCTYPE),
    ("test3.test", 562, "", ERR_DOCTYPE),
    ("test3.test", 564, "", ERR_DOCTYPE),
    ("test3.test", 565, "", ERR_DOCTYPE),
    ("test3.test", 566, "", ERR_DOCTYPE),
    ("test3.test", 567, "", ERR_DOCTYPE),
    ("test3.test", 568, "", ERR_DOCTYPE),
    ("test3.test", 569, "", ERR_DOCTYPE),
    ("test3.test", 570, "", ERR_DOCTYPE),
    ("test3.test", 571, "", ERR_DOCTYPE),
    ("test3.test", 572, "", ERR_DOCTYPE),
    ("test3.test", 573, "", ERR_DOCTYPE),
    ("test3.test", 574, "", ERR_DOCTYPE),
    ("test3.test", 575, "", ERR_DOCTYPE),
    ("test3.test", 576, "", ERR_DOCTYPE),
    ("test3.test", 577, "", ERR_DOCTYPE),
    ("test3.test", 578, "", ERR_DOCTYPE),
    ("test3.test", 579, "", ERR_DOCTYPE),
    ("test3.test", 580, "", ERR_DOCTYPE),
    ("test3.test", 581, "", ERR_DOCTYPE),
    ("test3.test", 582, "", ERR_DOCTYPE),
    ("test3.test", 583, "", ERR_DOCTYPE),
    ("test3.test", 584, "", ERR_DOCTYPE),
    ("test3.test", 585, "", ERR_DOCTYPE),
    ("test3.test", 586, "", ERR_DOCTYPE),
    ("test3.test", 587, "", ERR_DOCTYPE),
    ("test3.test", 588, "", ERR_DOCTYPE),
    ("test3.test", 589, "", ERR_DOCTYPE),
    ("test3.test", 590, "", ERR_DOCTYPE),
    ("test3.test", 591, "", ERR_DOCTYPE),
    ("test3.test", 592, "", ERR_DOCTYPE),
    ("test3.test", 593, "", ERR_DOCTYPE),
    ("test3.test", 594, "", ERR_DOCTYPE),
    ("test3.test", 595, "", ERR_DOCTYPE),
    ("test3.test", 596, "", ERR_DOCTYPE),
    ("test3.test", 597, "", ERR_DOCTYPE),
    ("test3.test", 598, "", ERR_DOCTYPE),
    ("test3.test", 599, "", ERR_DOCTYPE),
    ("test3.test", 600, "", ERR_DOCTYPE),
    ("test3.test", 601, "", ERR_DOCTYPE),
    ("test3.test", 602, "", ERR_DOCTYPE),
    ("test3.test", 603, "", ERR_DOCTYPE),
    ("test3.test", 604, "", ERR_DOCTYPE),
    ("test3.test", 605, "", ERR_DOCTYPE),
    ("test3.test", 606, "", ERR_DOCTYPE),
    ("test3.test", 607, "", ERR_DOCTYPE),
    ("test3.test", 608, "", ERR_DOCTYPE),
    ("test3.test", 609, "", ERR_DOCTYPE),
    ("test3.test", 610, "", ERR_DOCTYPE),
    ("test3.test", 611, "", ERR_DOCTYPE),
    ("test3.test", 612, "", ERR_DOCTYPE),
    ("test3.test", 613, "", ERR_DOCTYPE),
    ("test3.test", 640, "", ERR_DOCTYPE),
    ("test3.test", 707, "", ERR_DOCTYPE),
    ("test3.test", 708, "", ERR_DOCTYPE),
    ("test3.test", 709, "", ERR_DOCTYPE),
    ("test3.test", 710, "", ERR_DOCTYPE),
    ("test3.test", 711, "", ERR_DOCTYPE),
    ("test3.test", 712, "", ERR_DOCTYPE),
    ("test3.test", 713, "", ERR_DOCTYPE),
    ("test3.test", 714, "", ERR_DOCTYPE),
    ("test3.test", 715, "", ERR_DOCTYPE),
    ("test3.test", 716, "", ERR_DOCTYPE),
    ("test3.test", 717, "", ERR_DOCTYPE),
    ("test3.test", 719, "", ERR_DOCTYPE),
    ("test3.test", 720, "", ERR_DOCTYPE),
    ("test3.test", 721, "", ERR_DOCTYPE),
    ("test3.test", 722, "", ERR_DOCTYPE),
    ("test3.test", 723, "", ERR_DOCTYPE),
    ("test3.test", 724, "", ERR_DOCTYPE),
    ("test3.test", 725, "", ERR_DOCTYPE),
    ("test3.test", 726, "", ERR_DOCTYPE),
    ("test3.test", 727, "", ERR_DOCTYPE),
    ("test3.test", 729, "", ERR_DOCTYPE),
    ("test3.test", 730, "", ERR_DOCTYPE),
    ("test3.test", 732, "", ERR_DOCTYPE),
    ("test3.test", 733, "", ERR_DOCTYPE),
    ("test3.test", 735, "", ERR_DOCTYPE),
    ("test3.test", 736, "", ERR_DOCTYPE),
    ("test3.test", 738, "", ERR_DOCTYPE),
    ("test3.test", 739, "", ERR_DOCTYPE),
    ("test3.test", 741, "", ERR_DOCTYPE),
    ("test3.test", 742, "", ERR_DOCTYPE),
    ("test3.test", 744, "", ERR_DOCTYPE),
    ("test3.test", 745, "", ERR_DOCTYPE),
    ("test3.test", 746, "", ERR_DOCTYPE),
    ("test3.test", 747, "", ERR_DOCTYPE),
    ("test3.test", 748, "", ERR_DOCTYPE),
    ("test3.test", 749, "", ERR_DOCTYPE),
    ("test3.test", 750, "", ERR_DOCTYPE),
    ("test3.test", 751, "", ERR_DOCTYPE),
    ("test3.test", 752, "", ERR_DOCTYPE),
    ("test3.test", 753, "", ERR_DOCTYPE),
    ("test3.test", 754, "", ERR_DOCTYPE),
    ("test3.test", 755, "", ERR_DOCTYPE),
    ("test3.test", 756, "", ERR_DOCTYPE),
    ("test3.test", 757, "", ERR_DOCTYPE),
    ("test3.test", 758, "", ERR_DOCTYPE),
    ("test3.test", 759, "", ERR_DOCTYPE),
    ("test3.test", 760, "", ERR_DOCTYPE),
    ("test3.test", 761, "", ERR_DOCTYPE),
    ("test3.test", 762, "", ERR_DOCTYPE),
    ("test3.test", 764, "", ERR_DOCTYPE),
    ("test3.test", 765, "", ERR_DOCTYPE),
    ("test3.test", 767, "", ERR_DOCTYPE),
    ("test3.test", 768, "", ERR_DOCTYPE),
    ("test3.test", 770, "", ERR_DOCTYPE),
    ("test3.test", 771, "", ERR_DOCTYPE),
    ("test3.test", 772, "", ERR_DOCTYPE),
    ("test3.test", 773, "", ERR_DOCTYPE),
    ("test3.test", 774, "", ERR_DOCTYPE),
    ("test3.test", 775, "", ERR_DOCTYPE),
    ("test3.test", 777, "", ERR_DOCTYPE),
    ("test3.test", 778, "", ERR_DOCTYPE),
    ("test3.test", 779, "", ERR_DOCTYPE),
    ("test3.test", 780, "", ERR_DOCTYPE),
    ("test3.test", 781, "", ERR_DOCTYPE),
    ("test3.test", 782, "", ERR_DOCTYPE),
    ("test3.test", 783, "", ERR_DOCTYPE),
    ("test3.test", 784, "", ERR_DOCTYPE),
    ("test3.test", 785, "", ERR_DOCTYPE),
    ("test3.test", 786, "", ERR_DOCTYPE),
    ("test3.test", 787, "", ERR_DOCTYPE),
    ("test3.test", 788, "", ERR_DOCTYPE),
    ("test3.test", 789, "", ERR_DOCTYPE),
    ("test3.test", 790, "", ERR_DOCTYPE),
    ("test3.test", 791, "", ERR_DOCTYPE),
    ("test3.test", 792, "", ERR_DOCTYPE),
    ("test3.test", 793, "", ERR_DOCTYPE),
    ("test3.test", 794, "", ERR_DOCTYPE),
    ("test3.test", 795, "", ERR_DOCTYPE),
    ("test3.test", 796, "", ERR_DOCTYPE),
    ("test3.test", 797, "", ERR_DOCTYPE),
    ("test3.test", 798, "", ERR_DOCTYPE),
    ("test3.test", 799, "", ERR_DOCTYPE),
    ("test3.test", 800, "", ERR_DOCTYPE),
    ("test3.test", 801, "", ERR_DOCTYPE),
    ("test3.test", 802, "", ERR_DOCTYPE),
    ("test3.test", 803, "", ERR_DOCTYPE),
    ("test3.test", 804, "", ERR_DOCTYPE),
    ("test3.test", 805, "", ERR_DOCTYPE),
    ("test3.test", 806, "", ERR_DOCTYPE),
    ("test3.test", 807, "", ERR_DOCTYPE),
    ("test3.test", 808, "", ERR_DOCTYPE),
    ("test3.test", 809, "", ERR_DOCTYPE),
    ("test3.test", 810, "", ERR_DOCTYPE),
    ("test3.test", 812, "", ERR_DOCTYPE),
    ("test3.test", 813, "", ERR_DOCTYPE),
    ("test3.test", 814, "", ERR_DOCTYPE),
    ("test3.test", 815, "", ERR_DOCTYPE),
    ("test3.test", 816, "", ERR_DOCTYPE),
    ("test3.test", 817, "", ERR_DOCTYPE),
    ("test3.test", 818, "", ERR_DOCTYPE),
    ("test3.test", 820, "", ERR_DOCTYPE),
    ("test3.test", 821, "", ERR_DOCTYPE),
    ("test3.test", 823, "", ERR_DOCTYPE),
    ("test3.test", 824, "", ERR_DOCTYPE),
    ("test3.test", 826, "", ERR_DOCTYPE),
    ("test3.test", 827, "", ERR_DOCTYPE),
    ("test3.test", 828, "", ERR_DOCTYPE),
    ("test3.test", 829, "", ERR_DOCTYPE),
    ("test3.test", 830, "", ERR_DOCTYPE),
    ("test3.test", 831, "", ERR_DOCTYPE),
    ("test3.test", 832, "", ERR_DOCTYPE),
    ("test3.test", 833, "", ERR_DOCTYPE),
    ("test3.test", 834, "", ERR_DOCTYPE),
    ("test3.test", 835, "", ERR_DOCTYPE),
    ("test3.test", 836, "", ERR_DOCTYPE),
    ("test3.test", 837, "", ERR_DOCTYPE),
    ("test3.test", 838, "", ERR_DOCTYPE),
    ("test3.test", 839, "", ERR_DOCTYPE),
    ("test3.test", 840, "", ERR_DOCTYPE),
    ("test3.test", 841, "", ERR_DOCTYPE),
    ("test3.test", 842, "", ERR_DOCTYPE),
    ("test3.test", 843, "", ERR_DOCTYPE),
    ("test3.test", 844, "", ERR_DOCTYPE),
    ("test3.test", 845, "", ERR_DOCTYPE),
    ("test3.test", 846, "", ERR_DOCTYPE),
    ("test3.test", 847, "", ERR_DOCTYPE),
    ("test3.test", 848, "", ERR_DOCTYPE),
    ("test3.test", 849, "", ERR_DOCTYPE),
    ("test3.test", 850, "", ERR_DOCTYPE),
    ("test3.test", 851, "", ERR_DOCTYPE),
    ("test3.test", 852, "", ERR_DOCTYPE),
    ("test3.test", 853, "", ERR_DOCTYPE),
    ("test3.test", 854, "", ERR_DOCTYPE),
    ("test3.test", 855, "", ERR_DOCTYPE),
    ("test3.test", 856, "", ERR_DOCTYPE),
    ("test3.test", 857, "", ERR_DOCTYPE),
    ("test3.test", 858, "", ERR_DOCTYPE),
    ("test3.test", 859, "", ERR_DOCTYPE),
    ("test3.test", 860, "", ERR_DOCTYPE),
    ("test3.test", 861, "", ERR_DOCTYPE),
    ("test3.test", 862, "", ERR_DOCTYPE),
    ("test3.test", 863, "", ERR_DOCTYPE),
    ("test3.test", 864, "", ERR_DOCTYPE),
    ("test3.test", 865, "", ERR_DOCTYPE),
    ("test3.test", 866, "", ERR_DOCTYPE),
    ("test3.test", 867, "", ERR_DOCTYPE),
    ("test3.test", 868, "", ERR_DOCTYPE),
    ("test3.test", 869, "", ERR_DOCTYPE),
    ("test3.test", 870, "", ERR_DOCTYPE),
    ("test3.test", 871, "", ERR_DOCTYPE),
    ("test3.test", 872, "", ERR_DOCTYPE),
    ("test3.test", 873, "", ERR_DOCTYPE),
    ("test3.test", 874, "", ERR_DOCTYPE),
    ("test3.test", 875, "", ERR_DOCTYPE),
    ("test3.test", 876, "", ERR_DOCTYPE),
    ("test3.test", 877, "", ERR_DOCTYPE),
    ("test3.test", 878, "", ERR_DOCTYPE),
    ("test3.test", 879, "", ERR_DOCTYPE),
    ("test3.test", 880, "", ERR_DOCTYPE),
    ("test3.test", 881, "", ERR_DOCTYPE),
    ("test3.test", 882, "", ERR_DOCTYPE),
    ("test3.test", 883, "", ERR_DOCTYPE),
    ("test3.test", 884, "", ERR_DOCTYPE),
    ("test3.test", 885, "", ERR_DOCTYPE),
    ("test3.test", 886, "", ERR_DOCTYPE),
    ("test3.test", 887, "", ERR_DOCTYPE),
    ("test3.test", 888, "", ERR_DOCTYPE),
    ("test3.test", 889, "", ERR_DOCTYPE),
    ("test3.test", 890, "", ERR_DOCTYPE),
    ("test3.test", 891, "", ERR_DOCTYPE),
    ("test3.test", 892, "", ERR_DOCTYPE),
    ("test3.test", 893, "", ERR_DOCTYPE),
    ("test3.test", 894, "", ERR_DOCTYPE),
    ("test3.test", 895, "", ERR_DOCTYPE),
    ("test3.test", 896, "", ERR_DOCTYPE),
    ("test3.test", 897, "", ERR_DOCTYPE),
    ("test3.test", 898, "", ERR_DOCTYPE),
    ("test3.test", 899, "", ERR_DOCTYPE),
    ("test3.test", 901, "", ERR_DOCTYPE),
    ("test3.test", 902, "", ERR_DOCTYPE),
    ("test3.test", 904, "", ERR_DOCTYPE),
    ("test3.test", 905, "", ERR_DOCTYPE),
    ("test3.test", 907, "", ERR_DOCTYPE),
    ("test3.test", 908, "", ERR_DOCTYPE),
    ("test3.test", 909, "", ERR_DOCTYPE),
    ("test3.test", 910, "", ERR_DOCTYPE),
    ("test3.test", 911, "", ERR_DOCTYPE),
    ("test3.test", 912, "", ERR_DOCTYPE),
    ("test3.test", 914, "", ERR_DOCTYPE),
    ("test3.test", 915, "", ERR_DOCTYPE),
    ("test3.test", 916, "", ERR_DOCTYPE),
    ("test3.test", 917, "", ERR_DOCTYPE),
    ("test3.test", 918, "", ERR_DOCTYPE),
    ("test3.test", 919, "", ERR_DOCTYPE),
    ("test3.test", 920, "", ERR_DOCTYPE),
    ("test3.test", 921, "", ERR_DOCTYPE),
    ("test3.test", 922, "", ERR_DOCTYPE),
    ("test3.test", 923, "", ERR_DOCTYPE),
    ("test3.test", 924, "", ERR_DOCTYPE),
    ("test3.test", 925, "", ERR_DOCTYPE),
    ("test3.test", 926, "", ERR_DOCTYPE),
    ("test3.test", 927, "", ERR_DOCTYPE),
    ("test3.test", 928, "", ERR_DOCTYPE),
    ("test3.test", 929, "", ERR_DOCTYPE),
    ("test3.test", 930, "", ERR_DOCTYPE),
    ("test3.test", 931, "", ERR_DOCTYPE),
    ("test3.test", 932, "", ERR_DOCTYPE),
    ("test3.test", 933, "", ERR_DOCTYPE),
    ("test3.test", 934, "", ERR_DOCTYPE),
    ("test3.test", 935, "", ERR_DOCTYPE),
    ("test3.test", 936, "", ERR_DOCTYPE),
    ("test3.test", 937, "", ERR_DOCTYPE),
    ("test3.test", 938, "", ERR_DOCTYPE),
    ("test3.test", 939, "", ERR_DOCTYPE),
    ("test3.test", 940, "", ERR_DOCTYPE),
    ("test3.test", 941, "", ERR_DOCTYPE),
    ("test3.test", 942, "", ERR_DOCTYPE),
    ("test3.test", 943, "", ERR_DOCTYPE),
    ("test3.test", 944, "", ERR_DOCTYPE),
    ("test3.test", 945, "", ERR_DOCTYPE),
    ("test3.test", 946, "", ERR_DOCTYPE),
    ("test3.test", 947, "", ERR_DOCTYPE),
    ("test3.test", 949, "", ERR_DOCTYPE),
    ("test3.test", 950, "", ERR_DOCTYPE),
    ("test3.test", 951, "", ERR_DOCTYPE),
    ("test3.test", 952, "", ERR_DOCTYPE),
    ("test3.test", 953, "", ERR_DOCTYPE),
    ("test3.test", 954, "", ERR_DOCTYPE),
    ("test3.test", 955, "", ERR_DOCTYPE),
    ("test3.test", 957, "", ERR_DOCTYPE),
    ("test3.test", 958, "", ERR_DOCTYPE),
    ("test3.test", 960, "", ERR_DOCTYPE),
    ("test3.test", 961, "", ERR_DOCTYPE),
    ("test3.test", 963, "", ERR_DOCTYPE),
    ("test3.test", 964, "", ERR_DOCTYPE),
    ("test3.test", 965, "", ERR_DOCTYPE),
    ("test3.test", 966, "", ERR_DOCTYPE),
    ("test3.test", 967, "", ERR_DOCTYPE),
    ("test3.test", 968, "", ERR_DOCTYPE),
    ("test3.test", 969, "", ERR_DOCTYPE),
    ("test3.test", 970, "", ERR_DOCTYPE),
    ("test3.test", 971, "", ERR_DOCTYPE),
    ("test3.test", 972, "", ERR_DOCTYPE),
    ("test3.test", 973, "", ERR_DOCTYPE),
    ("test3.test", 974, "", ERR_DOCTYPE),
    ("test3.test", 975, "", ERR_DOCTYPE),
    ("test3.test", 976, "", ERR_DOCTYPE),
    ("test3.test", 977, "", ERR_DOCTYPE),
    ("test3.test", 978, "", ERR_DOCTYPE),
    ("test3.test", 979, "", ERR_DOCTYPE),
    ("test3.test", 980, "", ERR_DOCTYPE),
    ("test3.test", 981, "", ERR_DOCTYPE),
    ("test3.test", 982, "", ERR_DOCTYPE),
    ("test3.test", 983, "", ERR_DOCTYPE),
    ("test3.test", 984, "", ERR_DOCTYPE),
    ("test3.test", 985, "", ERR_DOCTYPE),
    ("test3.test", 986, "", ERR_DOCTYPE),
    ("test3.test", 987, "", ERR_DOCTYPE),
    ("test3.test", 988, "", ERR_DOCTYPE),
    ("test3.test", 989, "", ERR_DOCTYPE),
    ("test3.test", 990, "", ERR_DOCTYPE),
    ("test3.test", 991, "", ERR_DOCTYPE),
    ("test3.test", 992, "", ERR_DOCTYPE),
    ("test3.test", 993, "", ERR_DOCTYPE),
    ("test3.test", 994, "", ERR_DOCTYPE),
    ("test3.test", 995, "", ERR_DOCTYPE),
    ("test3.test", 996, "", ERR_DOCTYPE),
    ("test3.test", 997, "", ERR_DOCTYPE),
    ("test3.test", 998, "", ERR_DOCTYPE),
    ("test3.test", 999, "", ERR_DOCTYPE),
    ("test3.test", 1000, "", ERR_DOCTYPE),
    ("test3.test", 1001, "", ERR_DOCTYPE),
    ("test3.test", 1002, "", ERR_DOCTYPE),
    ("test3.test", 1003, "", ERR_DOCTYPE),
    ("test3.test", 1004, "", ERR_DOCTYPE),
    ("test3.test", 1005, "", ERR_DOCTYPE),
    ("test3.test", 1006, "", ERR_DOCTYPE),
    ("test3.test", 1007, "", ERR_DOCTYPE),
    ("test3.test", 1008, "", ERR_DOCTYPE),
    ("test3.test", 1009, "", ERR_DOCTYPE),
    ("test3.test", 1010, "", ERR_DOCTYPE),
    ("test3.test", 1011, "", ERR_DOCTYPE),
    ("test3.test", 1012, "", ERR_DOCTYPE),
    ("test3.test", 1013, "", ERR_DOCTYPE),
    ("test3.test", 1014, "", ERR_DOCTYPE),
    ("test3.test", 1015, "", ERR_DOCTYPE),
    ("test3.test", 1016, "", ERR_DOCTYPE),
    ("test3.test", 1017, "", ERR_DOCTYPE),
    ("test3.test", 1018, "", ERR_DOCTYPE),
    ("test3.test", 1019, "", ERR_DOCTYPE),
    ("test3.test", 1020, "", ERR_DOCTYPE),
    ("test3.test", 1021, "", ERR_DOCTYPE),
    ("test3.test", 1022, "", ERR_DOCTYPE),
    ("test3.test", 1023, "", ERR_DOCTYPE),
    ("test3.test", 1024, "", ERR_DOCTYPE),
    ("test3.test", 1025, "", ERR_DOCTYPE),
    ("test3.test", 1026, "", ERR_DOCTYPE),
    ("test3.test", 1027, "", ERR_DOCTYPE),
    ("test3.test", 1028, "", ERR_DOCTYPE),
    ("test3.test", 1029, "", ERR_DOCTYPE),
    ("test3.test", 1030, "", ERR_DOCTYPE),
    ("test3.test", 1031, "", ERR_DOCTYPE),
    ("test3.test", 1032, "", ERR_DOCTYPE),
    ("test3.test", 1033, "", ERR_DOCTYPE),
    ("test3.test", 1034, "", ERR_DOCTYPE),
    ("test3.test", 1035, "", ERR_DOCTYPE),
    ("test3.test", 1036, "", ERR_DOCTYPE),
    ("test3.test", 1037, "", ERR_DOCTYPE),
    ("test3.test", 1038, "", ERR_DOCTYPE),
    ("test3.test", 1039, "", ERR_DOCTYPE),
    ("test3.test", 1041, "", ERR_DOCTYPE),
    ("test3.test", 1042, "", ERR_DOCTYPE),
    ("test3.test", 1043, "", ERR_DOCTYPE),
    ("test3.test", 1044, "", ERR_DOCTYPE),
    ("test3.test", 1045, "", ERR_DOCTYPE),
    ("test3.test", 1046, "", ERR_DOCTYPE),
    ("test3.test", 1047, "", ERR_DOCTYPE),
    ("test3.test", 1048, "", ERR_DOCTYPE),
    ("test3.test", 1049, "", ERR_DOCTYPE),
    ("test3.test", 1050, "", ERR_DOCTYPE),
    ("test3.test", 1051, "", ERR_DOCTYPE),
    ("test3.test", 1052, "", ERR_DOCTYPE),
    ("test3.test", 1053, "", ERR_DOCTYPE),
    ("test3.test", 1054, "", ERR_DOCTYPE),
    ("test3.test", 1055, "", ERR_DOCTYPE),
    ("test3.test", 1056, "", ERR_DOCTYPE),
    ("test3.test", 1057, "", ERR_DOCTYPE),
    ("test3.test", 1058, "", ERR_DOCTYPE),
    ("test3.test", 1059, "", ERR_DOCTYPE),
    ("test3.test", 1060, "", ERR_DOCTYPE),
    ("test3.test", 1061, "", ERR_DOCTYPE),
    ("test3.test", 1062, "", ERR_DOCTYPE),
    ("test3.test", 1063, "", ERR_DOCTYPE),
    ("test3.test", 1064, "", ERR_DOCTYPE),
    ("test3.test", 1065, "", ERR_DOCTYPE),
    ("test3.test", 1066, "", ERR_DOCTYPE),
    ("test3.test", 1067, "", ERR_DOCTYPE),
    ("test3.test", 1068, "", ERR_DOCTYPE),
    ("test3.test", 1069, "", ERR_DOCTYPE),
    ("test3.test", 1070, "", ERR_DOCTYPE),
    ("test3.test", 1071, "", ERR_DOCTYPE),
    ("test3.test", 1072, "", ERR_DOCTYPE),
    ("test3.test", 1073, "", ERR_DOCTYPE),
    ("test3.test", 1074, "", ERR_DOCTYPE),
    ("test3.test", 1075, "", ERR_DOCTYPE),
    ("test3.test", 1076, "", ERR_DOCTYPE),
    ("test3.test", 1077, "", ERR_DOCTYPE),
    ("test3.test", 1078, "", ERR_DOCTYPE),
    ("test3.test", 1079, "", ERR_DOCTYPE),
    ("test3.test", 1080, "", ERR_DOCTYPE),
    ("test3.test", 1081, "", ERR_DOCTYPE),
    ("test3.test", 1082, "", ERR_DOCTYPE),
    ("test3.test", 1083, "", ERR_DOCTYPE),
    ("test3.test", 1084, "", ERR_DOCTYPE),
    ("test3.test", 1085, "", ERR_DOCTYPE),
    ("test3.test", 1086, "", ERR_DOCTYPE),
    ("test3.test", 1087, "", ERR_DOCTYPE),
    ("test3.test", 1088, "", ERR_DOCTYPE),
    ("test3.test", 1089, "", ERR_DOCTYPE),
    ("test3.test", 1090, "", ERR_DOCTYPE),
    ("test3.test", 1091, "", ERR_DOCTYPE),
    ("test3.test", 1092, "", ERR_DOCTYPE),
    ("test3.test", 1093, "", ERR_DOCTYPE),
    ("test3.test", 1094, "", ERR_DOCTYPE),
    ("test3.test", 1095, "", ERR_DOCTYPE),
    ("test3.test", 1096, "", ERR_DOCTYPE),
    ("test3.test", 1097, "", ERR_DOCTYPE),
    ("test3.test", 1098, "", ERR_DOCTYPE),
    ("test3.test", 1099, "", ERR_DOCTYPE),
    ("test3.test", 1100, "", ERR_DOCTYPE),
    ("test3.test", 1101, "", ERR_DOCTYPE),
    ("test3.test", 1102, "", ERR_DOCTYPE),
    ("test3.test", 1103, "", ERR_DOCTYPE),
    // ERR_DOCTYPE_AND_INPUT_STREAM: 36 cases.
    ("test3.test", 282, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 285, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 288, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 369, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 406, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 414, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 417, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 420, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 514, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 549, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 557, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 560, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 563, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 728, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 731, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 734, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 737, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 740, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 743, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 763, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 766, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 769, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 776, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 811, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 819, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 822, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 825, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 900, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 903, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 906, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 913, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 948, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 956, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 959, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 962, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    ("test3.test", 1040, "", ERR_DOCTYPE_AND_INPUT_STREAM),
    // ERR_DUP_ATTR_AND_END_TAG_ATTRS: 1 cases.
    ("test4.test", 49, "", ERR_DUP_ATTR_AND_END_TAG_ATTRS),
    // ERR_INPUT_STREAM_AND_EOF_CDATA: 1 cases.
    ("test3.test", 7, "", ERR_INPUT_STREAM_AND_EOF_CDATA),
];

// ---------------------------------------------------------------------------
// Minimal JSON parser (std-only; no serde per workspace rules).
//
// Supports exactly what the fixtures contain: objects, arrays, strings
// (with `\" \\ \/ \b \f \n \r \t \uHHHH` escapes and surrogate pairs),
// integers, `true`/`false`/`null`. A lone `\uD800`-`\uDFFF` escape with
// no partner becomes U+FFFD, matching WHATWG decode of unpaired
// surrogates.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Json {
    Null,
    Bool(bool),
    Num(i64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

struct JsonParser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> JsonParser<'a> {
    fn parse(b: &'a [u8]) -> Result<Json, String> {
        let mut p = JsonParser { b, i: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        if p.i != p.b.len() {
            return Err(format!("trailing bytes at offset {}", p.i));
        }
        Ok(v)
    }

    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn expect(&mut self, c: u8) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected {c:?} at offset {}", self.i))
        }
    }

    fn lit(&mut self, s: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(v)
        } else {
            Err(format!("bad literal at offset {}", self.i))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'n') => self.lit("null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            other => Err(format!("unexpected {other:?} at offset {}", self.i)),
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut obj = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(obj));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return Err(format!("expected string key at offset {}", self.i));
            }
            let k = self.string()?;
            self.ws();
            self.expect(b':')?;
            self.ws();
            let v = self.value()?;
            obj.push((k, v));
            self.ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(obj));
                }
                other => return Err(format!("unexpected {other:?} in object at {}", self.i)),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut arr = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(arr));
        }
        loop {
            self.ws();
            arr.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(arr));
                }
                other => return Err(format!("unexpected {other:?} in array at {}", self.i)),
            }
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
        // Fixtures only use integers; reject anything else loudly rather
        // than mis-parsing it.
        if self.peek().is_some_and(|c| matches!(c, b'.' | b'e' | b'E')) {
            return Err(format!("non-integer number at offset {start}"));
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|e| e.to_string())?;
        s.parse::<i64>()
            .map(Json::Num)
            .map_err(|_| format!("bad number at offset {start}"))
    }

    /// Parse one `\uHHHH` escape at the current position (just past the
    /// `\u`); returns the code unit and advances past the four digits.
    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() {
            return Err("truncated \\u escape".to_string());
        }
        let mut v: u32 = 0;
        for k in 0..4 {
            let d = self.b[self.i + k];
            let d = (d as char)
                .to_digit(16)
                .ok_or("bad hex digit in \\u escape")?;
            v = v * 16 + d;
        }
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let c = self.peek().ok_or("unterminated string")?;
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.i += 1;
                    match self.peek().ok_or("truncated escape")? {
                        b'"' => {
                            out.push('"');
                            self.i += 1;
                        }
                        b'\\' => {
                            out.push('\\');
                            self.i += 1;
                        }
                        b'/' => {
                            out.push('/');
                            self.i += 1;
                        }
                        b'b' => {
                            out.push('\u{8}');
                            self.i += 1;
                        }
                        b'f' => {
                            out.push('\u{C}');
                            self.i += 1;
                        }
                        b'n' => {
                            out.push('\n');
                            self.i += 1;
                        }
                        b'r' => {
                            out.push('\r');
                            self.i += 1;
                        }
                        b't' => {
                            out.push('\t');
                            self.i += 1;
                        }
                        b'u' => {
                            self.i += 1;
                            let hi = self.hex4()?;
                            // Surrogate pair: `\uD83D\uDE00`.
                            if (0xD800..0xDC00).contains(&hi)
                                && self.b.get(self.i..self.i + 2) == Some(b"\\u".as_slice())
                            {
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xDC00..0xE000).contains(&lo) {
                                    let cp = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                                    out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                                    continue;
                                }
                                // Not a low surrogate: the high surrogate
                                // is lone.
                                out.push('\u{FFFD}');
                                out.push(char::from_u32(lo).unwrap_or('\u{FFFD}'));
                                continue;
                            }
                            out.push(char::from_u32(hi).unwrap_or('\u{FFFD}'));
                        }
                        other => return Err(format!("bad escape \\{other}")),
                    }
                }
                _ => {
                    // Maximal run of ordinary bytes (no quote/backslash):
                    // validate and copy in one go instead of char by
                    // char (char-by-char revalidates the whole tail,
                    // which is O(n^2) on the 1.1 MB fixtures).
                    let mut j = self.i;
                    while j < self.b.len() && !matches!(self.b[j], b'"' | b'\\') {
                        j += 1;
                    }
                    let s = std::str::from_utf8(&self.b[self.i..j]).map_err(|e| e.to_string())?;
                    out.push_str(s);
                    self.i = j;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fixture model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum ExpToken {
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
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // line/col are parsed for documentation; comparison is code-family only
struct ExpError {
    code: String,
    line: u32,
    col: u32,
}

struct Case {
    index: usize,
    description: String,
    input: String,
    output: Vec<ExpToken>,
    errors: Vec<ExpError>,
    initial_states: Vec<&'static str>,
    last_start_tag: String,
}

fn obj_get<'j>(obj: &'j [(String, Json)], key: &str) -> Option<&'j Json> {
    obj.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn as_str(j: &Json) -> Result<&str, String> {
    match j {
        Json::Str(s) => Ok(s),
        _ => Err("expected string".to_string()),
    }
}

/// The `doubleEscaped` unescaping (fixture README): every `\uHHHH`
/// sequence becomes the corresponding character, in both input and
/// output strings. Surrogate pairs combine; lone surrogates become
/// U+FFFD (WHATWG decode behavior).
fn unescape_double_escaped(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\'
            && i + 5 < b.len()
            && b[i + 1] == b'u'
            && b[i + 2..i + 6].iter().all(|c| c.is_ascii_hexdigit())
        {
            let hi =
                u32::from_str_radix(std::str::from_utf8(&b[i + 2..i + 6]).unwrap(), 16).unwrap();
            i += 6;
            // Surrogate pair.
            if (0xD800..0xDC00).contains(&hi)
                && i + 5 < b.len()
                && b[i] == b'\\'
                && b[i + 1] == b'u'
                && b[i + 2..i + 6].iter().all(|c| c.is_ascii_hexdigit())
            {
                let lo = u32::from_str_radix(std::str::from_utf8(&b[i + 2..i + 6]).unwrap(), 16)
                    .unwrap();
                if (0xDC00..0xE000).contains(&lo) {
                    i += 6;
                    let cp = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                    out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                    continue;
                }
            }
            out.push(char::from_u32(hi).unwrap_or('\u{FFFD}'));
        } else {
            // Copy a maximal run with no backslash in one go (see the
            // JSON parser: char-by-char is O(n^2)).
            let mut j = i;
            while j < b.len() && b[j] != b'\\' {
                j += 1;
            }
            let s = std::str::from_utf8(&b[i..j]).unwrap();
            out.push_str(s);
            i = j;
        }
    }
    out
}

fn parse_token(arr: &[Json], de: bool) -> Result<ExpToken, String> {
    let un = |s: &str| {
        if de {
            unescape_double_escaped(s)
        } else {
            s.to_string()
        }
    };
    let kind = as_str(arr.first().ok_or("empty token")?)?;
    match kind {
        "DOCTYPE" => {
            if arr.len() != 5 {
                return Err(format!("DOCTYPE token has {} items", arr.len()));
            }
            // A null name means "missing" (e.g. `<!DOCTYPE>`); the
            // driver maps a missing name to "" as well.
            let name = match &arr[1] {
                Json::Null => String::new(),
                Json::Str(s) => un(s),
                _ => return Err("DOCTYPE name not string/null".to_string()),
            };
            let public = match &arr[2] {
                Json::Null => None,
                Json::Str(s) => Some(un(s)),
                _ => return Err("DOCTYPE public_id not string/null".to_string()),
            };
            let system = match &arr[3] {
                Json::Null => None,
                Json::Str(s) => Some(un(s)),
                _ => return Err("DOCTYPE system_id not string/null".to_string()),
            };
            // `correctness`: true means force-quirks is off.
            let correctness = match &arr[4] {
                Json::Bool(b) => *b,
                _ => return Err("DOCTYPE correctness not bool".to_string()),
            };
            Ok(ExpToken::Doctype {
                name,
                public,
                system,
                force_quirks: !correctness,
            })
        }
        "StartTag" => {
            if arr.len() != 3 && arr.len() != 4 {
                return Err(format!("StartTag token has {} items", arr.len()));
            }
            let name = un(as_str(&arr[1])?);
            let attrs = match &arr[2] {
                Json::Obj(pairs) => pairs
                    .iter()
                    .map(|(k, v)| Ok((un(k), un(as_str(v)?))))
                    .collect::<Result<Vec<_>, String>>()?,
                _ => return Err("StartTag attrs not object".to_string()),
            };
            let self_closing = match arr.get(3) {
                None => false,
                Some(Json::Bool(b)) => *b,
                _ => return Err("StartTag self-closing flag not bool".to_string()),
            };
            Ok(ExpToken::StartTag {
                name,
                attrs,
                self_closing,
            })
        }
        "EndTag" => {
            if arr.len() != 2 {
                return Err(format!("EndTag token has {} items", arr.len()));
            }
            Ok(ExpToken::EndTag {
                name: un(as_str(&arr[1])?),
            })
        }
        "Comment" => {
            if arr.len() != 2 {
                return Err(format!("Comment token has {} items", arr.len()));
            }
            Ok(ExpToken::Comment(un(as_str(&arr[1])?)))
        }
        "Character" => {
            if arr.len() != 2 {
                return Err(format!("Character token has {} items", arr.len()));
            }
            Ok(ExpToken::Character(un(as_str(&arr[1])?)))
        }
        other => Err(format!("unknown token kind {other:?}")),
    }
}

fn map_state(name: &str) -> Result<&'static str, String> {
    match name {
        "Data state" => Ok("Data"),
        "RCDATA state" => Ok("RCData"),
        "RAWTEXT state" => Ok("RawText"),
        "Script data state" => Ok("ScriptData"),
        "PLAINTEXT state" => Ok("PlainText"),
        "CDATA section state" => Ok("CDataSection"),
        other => Err(format!("unknown initial state {other:?}")),
    }
}

fn parse_case(index: usize, obj: &[(String, Json)]) -> Result<Case, String> {
    let str_field = |key: &str| -> Result<String, String> {
        obj_get(obj, key)
            .map(as_str)
            .transpose()?
            .map(|s| s.to_string())
            .ok_or_else(|| format!("missing {key}"))
    };
    let description = str_field("description")?;
    let de = matches!(obj_get(obj, "doubleEscaped"), Some(Json::Bool(true)));
    let input_raw = str_field("input")?;
    let input = if de {
        unescape_double_escaped(&input_raw)
    } else {
        input_raw
    };
    let output = match obj_get(obj, "output") {
        Some(Json::Arr(tokens)) => tokens
            .iter()
            .map(|t| match t {
                Json::Arr(arr) => parse_token(arr, de),
                _ => Err("token not array".to_string()),
            })
            .collect::<Result<Vec<_>, String>>()?,
        _ => return Err("missing output array".to_string()),
    };
    let errors = match obj_get(obj, "errors") {
        None => Vec::new(),
        Some(Json::Arr(errs)) => errs
            .iter()
            .map(|e| match e {
                Json::Obj(pairs) => {
                    let code = obj_get(pairs, "code")
                        .map(as_str)
                        .transpose()?
                        .ok_or("error without code")?
                        .to_string();
                    let num = |key: &str| -> Result<u32, String> {
                        match obj_get(pairs, key) {
                            Some(Json::Num(n)) => {
                                u32::try_from(*n).map_err(|_| format!("bad {key}"))
                            }
                            _ => Err(format!("error without {key}")),
                        }
                    };
                    Ok(ExpError {
                        code,
                        line: num("line")?,
                        col: num("col")?,
                    })
                }
                _ => Err("error not object".to_string()),
            })
            .collect::<Result<Vec<_>, String>>()?,
        _ => return Err("errors not array".to_string()),
    };
    let initial_states = match obj_get(obj, "initialStates") {
        None => vec!["Data"],
        Some(Json::Arr(states)) => states
            .iter()
            .map(|s| map_state(as_str(s).map_err(|e| e.to_string())?))
            .collect::<Result<Vec<_>, String>>()?,
        _ => return Err("initialStates not array".to_string()),
    };
    let last_start_tag = match obj_get(obj, "lastStartTag") {
        None => String::new(),
        Some(Json::Str(s)) => {
            if de {
                unescape_double_escaped(s)
            } else {
                s.clone()
            }
        }
        _ => return Err("lastStartTag not string".to_string()),
    };
    Ok(Case {
        index,
        description,
        input,
        output,
        errors,
        initial_states,
        last_start_tag,
    })
}

fn load_cases(path: &std::path::Path, _file: &str) -> Vec<Case> {
    let content =
        std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let root = JsonParser::parse(&content)
        .unwrap_or_else(|e| panic!("bad JSON in {}: {e}", path.display()));
    let tests = match &root {
        Json::Obj(pairs) => {
            // `xmlViolation.test` uses "xmlViolationTests".
            obj_get(pairs, "tests")
                .or_else(|| obj_get(pairs, "xmlViolationTests"))
                .unwrap_or_else(|| panic!("no tests array in {}", path.display()))
        }
        _ => panic!("root not object in {}", path.display()),
    };
    let arr = match tests {
        Json::Arr(a) => a,
        _ => panic!("tests not array in {}", path.display()),
    };
    arr.iter()
        .enumerate()
        .map(|(i, t)| match t {
            Json::Obj(pairs) => parse_case(i, pairs)
                .unwrap_or_else(|e| panic!("bad case {i} in {}: {e}", path.display())),
            _ => panic!("case {i} not object in {}", path.display()),
        })
        .collect()
}

fn fixture_files() -> Vec<(std::path::PathBuf, String)> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("html5lib-tokenizer");
    let mut files: Vec<_> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", root.display()))
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "test"))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            (p, name)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Map our [`ParseErrorCode`] to the WHATWG 13.2.2 code strings it
/// covers. Our codes are deliberately coarser in places (see the
/// variant docs in `tree.rs`); the harness accepts any family member
/// rather than pretending we track codes we collapsed away.
fn error_family(code: ParseErrorCode) -> &'static [&'static str] {
    match code {
        ParseErrorCode::UnexpectedNullCharacter => &["unexpected-null-character"],
        // Every quirks-forcing path through the DOCTYPE states collapses
        // into one code (see `emit_doctype`).
        ParseErrorCode::InvalidDoctype => &[
            "missing-doctype-name",
            "abrupt-doctype-public-identifier",
            "abrupt-doctype-system-identifier",
            "eof-in-doctype",
            "missing-doctype-name",
            "missing-doctype-public-identifier",
            "missing-doctype-system-identifier",
            "missing-quote-before-doctype-public-identifier",
            "missing-quote-before-doctype-system-identifier",
            "missing-whitespace-before-doctype-name",
            "missing-whitespace-after-doctype-public-keyword",
            "missing-whitespace-after-doctype-system-keyword",
            "missing-whitespace-between-doctype-public-and-system-identifiers",
            "invalid-character-sequence-after-doctype-name",
            "unexpected-character-after-doctype-system-identifier",
        ],
        ParseErrorCode::EofBeforeTagName => &["eof-before-tag-name"],
        ParseErrorCode::MissingEndTagName => &["missing-end-tag-name"],
        ParseErrorCode::InvalidFirstCharacterOfTagName => &["invalid-first-character-of-tag-name"],
        ParseErrorCode::EofInTag => &["eof-in-tag"],
        ParseErrorCode::UnexpectedEqualsSignBeforeAttributeName => {
            &["unexpected-equals-sign-before-attribute-name"]
        }
        ParseErrorCode::UnexpectedCharacterInAttributeName => {
            &["unexpected-character-in-attribute-name"]
        }
        ParseErrorCode::MissingWhitespaceBetweenAttributes => {
            &["missing-whitespace-between-attributes"]
        }
        ParseErrorCode::MissingAttributeValue => &["missing-attribute-value"],
        ParseErrorCode::UnexpectedCharacterInUnquotedAttributeValue => {
            &["unexpected-character-in-unquoted-attribute-value"]
        }
        ParseErrorCode::UnexpectedSolidusInTag => &["unexpected-solidus-in-tag"],
        ParseErrorCode::IncorrectlyOpenedComment => &["incorrectly-opened-comment"],
        ParseErrorCode::AbruptClosingOfEmptyComment => &["abrupt-closing-of-empty-comment"],
        ParseErrorCode::CdataInHtmlContent => &["cdata-in-html-content"],
        ParseErrorCode::EofInComment => &["eof-in-comment"],
        ParseErrorCode::NullCharacterReference => &["null-character-reference"],
        ParseErrorCode::CharacterReferenceOutsideUnicodeRange => {
            &["character-reference-outside-unicode-range"]
        }
        ParseErrorCode::SurrogateCharacterReference => &["surrogate-character-reference"],
        ParseErrorCode::NoncharacterCharacterReference => &["noncharacter-character-reference"],
        ParseErrorCode::ControlCharacterReference => &["control-character-reference"],
        ParseErrorCode::MissingSemicolonAfterCharacterReference => {
            &["missing-semicolon-after-character-reference"]
        }
        ParseErrorCode::AbsenceOfDigitsInNumericCharacterReference => {
            &["absence-of-digits-in-numeric-character-reference"]
        }
        ParseErrorCode::UnknownNamedCharacterReference => &["unknown-named-character-reference"],
        // Tree-construction codes never arise from the tokenizer driver.
        _ => &[],
    }
}

fn sorted_attrs(attrs: &[(String, String)]) -> Vec<(&str, &str)> {
    let mut v: Vec<(&str, &str)> = attrs
        .iter()
        .map(|(n, val)| (n.as_str(), val.as_str()))
        .collect();
    v.sort();
    v
}

fn compare_tokens(actual: &[TestToken], expected: &[ExpToken]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "token count {} != {}:\n  actual:   {actual:?}\n  expected: {expected:?}",
            actual.len(),
            expected.len()
        ));
    }
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let ok = match (a, e) {
            (
                TestToken::Doctype {
                    name: an,
                    public: ap,
                    system: asys,
                    force_quirks: aq,
                },
                ExpToken::Doctype {
                    name: en,
                    public: ep,
                    system: esys,
                    force_quirks: eq,
                },
            ) => an == en && ap == ep && asys == esys && aq == eq,
            (
                TestToken::StartTag {
                    name: an,
                    attrs: aa,
                    self_closing: asc,
                },
                ExpToken::StartTag {
                    name: en,
                    attrs: ea,
                    self_closing: esc,
                },
            ) => an == en && asc == esc && sorted_attrs(aa) == sorted_attrs(ea),
            (TestToken::EndTag { name: an }, ExpToken::EndTag { name: en }) => an == en,
            (TestToken::Comment(ad), ExpToken::Comment(ed)) => ad == ed,
            (TestToken::Character(at), ExpToken::Character(et)) => at == et,
            // No expected form exists: any PI token is a deviation (the
            // spec emits a bogus comment here).
            (TestToken::ProcessingInstruction { .. }, _) => false,
            _ => false,
        };
        if !ok {
            return Err(format!(
                "token {i} differs:\n  actual:   {a:?}\n  expected: {e:?}"
            ));
        }
    }
    Ok(())
}

fn compare_errors(actual: &[TestError], expected: &[ExpError]) -> Result<(), String> {
    // Locations are intentionally not compared (see the module docs):
    // only the error count, order, and code families are gated.
    if actual.len() != expected.len() {
        return Err(format!(
            "error count {} != {}:\n  actual:   {actual:?}\n  expected: {expected:?}",
            actual.len(),
            expected.len()
        ));
    }
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        if !error_family(a.code).contains(&e.code.as_str()) {
            return Err(format!(
                "error {i}: code {:?} does not cover expected {:?}",
                a.code, e.code
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

fn is_known(file: &str, index: usize, state: &str) -> Option<&'static str> {
    KNOWN_DEVIATIONS
        .iter()
        .find(|(f, i, s, _)| *f == file && *i == index && (*s == state || s.is_empty()))
        .map(|(_, _, _, reason)| *reason)
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let mut end = n;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[test]
fn html5lib_tokenizer() {
    let mut pass = 0usize;
    let mut known = 0usize;
    let mut runs = 0usize;
    let mut unexpected: Vec<String> = Vec::new();
    let mut stale: Vec<String> = Vec::new();

    for (path, file) in fixture_files() {
        for case in load_cases(&path, &file) {
            for state in &case.initial_states {
                runs += 1;
                let id = format!("{file}#{}[{state}]", case.index);
                let (tokens, errors) = tokenize(&case.input, state, &case.last_start_tag);
                let result = compare_tokens(&tokens, &case.output)
                    .and_then(|()| compare_errors(&errors, &case.errors));
                match (result, is_known(&file, case.index, state)) {
                    (Ok(_), None) => pass += 1,
                    (Ok(_), Some(reason)) => stale.push(format!(
                        "{id} {:?}: listed as known deviation ({reason}) but now passes",
                        case.description
                    )),
                    (Err(_), Some(_)) => known += 1,
                    (Err(diff), None) => unexpected.push(format!(
                        "{id} {:?}: {diff}; input: {:?}",
                        case.description,
                        truncate(&case.input, 90),
                        diff = diff.lines().next().unwrap_or(&diff),
                    )),
                }
            }
        }
    }

    println!("html5lib tokenizer: {pass} passed, {known} known deviations ({runs} case runs)");
    for u in &unexpected {
        println!("UNEXPECTED: {u}");
    }
    let mut msg = String::new();
    if !stale.is_empty() {
        msg.push_str(&format!(
            "\n{} STALE known deviations (now passing):\n  ",
            stale.len()
        ));
        msg.push_str(&stale.join("\n  "));
    }
    if !unexpected.is_empty() {
        msg.push_str(&format!(
            "\n{} UNEXPECTED deviations:\n  ",
            unexpected.len()
        ));
        let shown: Vec<&str> = unexpected.iter().take(40).map(|s| s.as_str()).collect();
        msg.push_str(&shown.join("\n  "));
        if unexpected.len() > 40 {
            msg.push_str(&format!(
                "\n  … and {} more (see stdout)",
                unexpected.len() - 40
            ));
        }
    }
    assert!(msg.is_empty(), "tokenizer failures:{msg}");
}
