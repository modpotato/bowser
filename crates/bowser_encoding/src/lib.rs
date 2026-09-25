//! WHATWG Encoding Standard: decode paths, label table, BOM and prescan hooks.
//!
//! This crate implements the decode side of
//! [Encoding](https://encoding.spec.whatwg.org/): the full encoding label
//! table ("get an encoding"), BOM sniffing, every single-byte legacy decoder,
//! UTF-8, UTF-16LE/BE, Big5, GBK/gb18030, replacement and x-user-defined, plus
//! the HTML Standard's prescan algorithm hooks used to sniff a document's
//! encoding from `<meta>` bytes.
//!
//! EUC-JP, ISO-2022-JP, Shift_JIS and EUC-KR resolve labels correctly but have
//! no decoder yet ([`DecodeError::Unsupported`]); everything in scope for real
//! web content is implemented.
//!
//! Encoding (bytes from text) is intentionally out of scope: the web platform
//! only needs UTF-8 encoding plus label/decoder coverage for legacy content.

mod data;
mod decode;
mod sniff;

pub use data::INDEX_VERSIONS;
pub use sniff::{prescan_encoding, sniff_bom, xml_encoding};

/// A character encoding from the WHATWG Encoding Standard.
///
/// The canonical [`name`](Encoding::name) matches the spec's lowercase
/// serialization (e.g. `"utf-8"`, `"windows-1252"`, `"shift_jis"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Encoding {
    /// UTF-8.
    Utf8,
    /// IBM866.
    Ibm866,
    /// ISO-8859-2.
    Iso8859_2,
    /// ISO-8859-3.
    Iso8859_3,
    /// ISO-8859-4.
    Iso8859_4,
    /// ISO-8859-5.
    Iso8859_5,
    /// ISO-8859-6.
    Iso8859_6,
    /// ISO-8859-7.
    Iso8859_7,
    /// ISO-8859-8.
    Iso8859_8,
    /// ISO-8859-8-I (distinct name; decodes like ISO-8859-8).
    Iso8859_8I,
    /// ISO-8859-10.
    Iso8859_10,
    /// ISO-8859-13.
    Iso8859_13,
    /// ISO-8859-14.
    Iso8859_14,
    /// ISO-8859-15.
    Iso8859_15,
    /// ISO-8859-16.
    Iso8859_16,
    /// KOI8-R.
    Koi8R,
    /// KOI8-U.
    Koi8U,
    /// macintosh.
    Macintosh,
    /// windows-874.
    Windows874,
    /// windows-1250.
    Windows1250,
    /// windows-1251.
    Windows1251,
    /// windows-1252.
    Windows1252,
    /// windows-1253.
    Windows1253,
    /// windows-1254.
    Windows1254,
    /// windows-1255.
    Windows1255,
    /// windows-1256.
    Windows1256,
    /// windows-1257.
    Windows1257,
    /// windows-1258.
    Windows1258,
    /// x-mac-cyrillic.
    XMacCyrillic,
    /// GBK (decodes with gb18030's decoder per the standard).
    Gbk,
    /// gb18030.
    Gb18030,
    /// Big5.
    Big5,
    /// EUC-JP (label only; no decoder yet).
    EucJp,
    /// ISO-2022-JP (label only; no decoder yet).
    Iso2022Jp,
    /// Shift_JIS (label only; no decoder yet).
    ShiftJis,
    /// EUC-KR (label only; no decoder yet).
    EucKr,
    /// replacement.
    Replacement,
    /// UTF-16BE.
    Utf16Be,
    /// UTF-16LE.
    Utf16Le,
    /// x-user-defined.
    XUserDefined,
}

impl Encoding {
    /// The spec's canonical lowercase name, e.g. `"windows-1252"`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Encoding::Utf8 => "utf-8",
            Encoding::Ibm866 => "ibm866",
            Encoding::Iso8859_2 => "iso-8859-2",
            Encoding::Iso8859_3 => "iso-8859-3",
            Encoding::Iso8859_4 => "iso-8859-4",
            Encoding::Iso8859_5 => "iso-8859-5",
            Encoding::Iso8859_6 => "iso-8859-6",
            Encoding::Iso8859_7 => "iso-8859-7",
            Encoding::Iso8859_8 => "iso-8859-8",
            Encoding::Iso8859_8I => "iso-8859-8-i",
            Encoding::Iso8859_10 => "iso-8859-10",
            Encoding::Iso8859_13 => "iso-8859-13",
            Encoding::Iso8859_14 => "iso-8859-14",
            Encoding::Iso8859_15 => "iso-8859-15",
            Encoding::Iso8859_16 => "iso-8859-16",
            Encoding::Koi8R => "koi8-r",
            Encoding::Koi8U => "koi8-u",
            Encoding::Macintosh => "macintosh",
            Encoding::Windows874 => "windows-874",
            Encoding::Windows1250 => "windows-1250",
            Encoding::Windows1251 => "windows-1251",
            Encoding::Windows1252 => "windows-1252",
            Encoding::Windows1253 => "windows-1253",
            Encoding::Windows1254 => "windows-1254",
            Encoding::Windows1255 => "windows-1255",
            Encoding::Windows1256 => "windows-1256",
            Encoding::Windows1257 => "windows-1257",
            Encoding::Windows1258 => "windows-1258",
            Encoding::XMacCyrillic => "x-mac-cyrillic",
            Encoding::Gbk => "gbk",
            Encoding::Gb18030 => "gb18030",
            Encoding::Big5 => "big5",
            Encoding::EucJp => "euc-jp",
            Encoding::Iso2022Jp => "iso-2022-jp",
            Encoding::ShiftJis => "shift_jis",
            Encoding::EucKr => "euc-kr",
            Encoding::Replacement => "replacement",
            Encoding::Utf16Be => "utf-16be",
            Encoding::Utf16Le => "utf-16le",
            Encoding::XUserDefined => "x-user-defined",
        }
    }

    /// All encodings in spec order.
    pub const ALL: [Encoding; 40] = [
        Encoding::Utf8,
        Encoding::Ibm866,
        Encoding::Iso8859_2,
        Encoding::Iso8859_3,
        Encoding::Iso8859_4,
        Encoding::Iso8859_5,
        Encoding::Iso8859_6,
        Encoding::Iso8859_7,
        Encoding::Iso8859_8,
        Encoding::Iso8859_8I,
        Encoding::Iso8859_10,
        Encoding::Iso8859_13,
        Encoding::Iso8859_14,
        Encoding::Iso8859_15,
        Encoding::Iso8859_16,
        Encoding::Koi8R,
        Encoding::Koi8U,
        Encoding::Macintosh,
        Encoding::Windows874,
        Encoding::Windows1250,
        Encoding::Windows1251,
        Encoding::Windows1252,
        Encoding::Windows1253,
        Encoding::Windows1254,
        Encoding::Windows1255,
        Encoding::Windows1256,
        Encoding::Windows1257,
        Encoding::Windows1258,
        Encoding::XMacCyrillic,
        Encoding::Gbk,
        Encoding::Gb18030,
        Encoding::Big5,
        Encoding::EucJp,
        Encoding::Iso2022Jp,
        Encoding::ShiftJis,
        Encoding::EucKr,
        Encoding::Replacement,
        Encoding::Utf16Be,
        Encoding::Utf16Le,
        Encoding::XUserDefined,
    ];

    /// The single-byte decode table, if this is a legacy single-byte encoding.
    pub(crate) const fn single_byte_table(self) -> Option<&'static [u16; 128]> {
        match self {
            Encoding::Ibm866 => Some(&data::SINGLE_IBM866),
            Encoding::Iso8859_2 => Some(&data::SINGLE_ISO_8859_2),
            Encoding::Iso8859_3 => Some(&data::SINGLE_ISO_8859_3),
            Encoding::Iso8859_4 => Some(&data::SINGLE_ISO_8859_4),
            Encoding::Iso8859_5 => Some(&data::SINGLE_ISO_8859_5),
            Encoding::Iso8859_6 => Some(&data::SINGLE_ISO_8859_6),
            Encoding::Iso8859_7 => Some(&data::SINGLE_ISO_8859_7),
            Encoding::Iso8859_8 | Encoding::Iso8859_8I => Some(&data::SINGLE_ISO_8859_8),
            Encoding::Iso8859_10 => Some(&data::SINGLE_ISO_8859_10),
            Encoding::Iso8859_13 => Some(&data::SINGLE_ISO_8859_13),
            Encoding::Iso8859_14 => Some(&data::SINGLE_ISO_8859_14),
            Encoding::Iso8859_15 => Some(&data::SINGLE_ISO_8859_15),
            Encoding::Iso8859_16 => Some(&data::SINGLE_ISO_8859_16),
            Encoding::Koi8R => Some(&data::SINGLE_KOI8_R),
            Encoding::Koi8U => Some(&data::SINGLE_KOI8_U),
            Encoding::Macintosh => Some(&data::SINGLE_MACINTOSH),
            Encoding::Windows874 => Some(&data::SINGLE_WINDOWS_874),
            Encoding::Windows1250 => Some(&data::SINGLE_WINDOWS_1250),
            Encoding::Windows1251 => Some(&data::SINGLE_WINDOWS_1251),
            Encoding::Windows1252 => Some(&data::SINGLE_WINDOWS_1252),
            Encoding::Windows1253 => Some(&data::SINGLE_WINDOWS_1253),
            Encoding::Windows1254 => Some(&data::SINGLE_WINDOWS_1254),
            Encoding::Windows1255 => Some(&data::SINGLE_WINDOWS_1255),
            Encoding::Windows1256 => Some(&data::SINGLE_WINDOWS_1256),
            Encoding::Windows1257 => Some(&data::SINGLE_WINDOWS_1257),
            Encoding::Windows1258 => Some(&data::SINGLE_WINDOWS_1258),
            Encoding::XMacCyrillic => Some(&data::SINGLE_X_MAC_CYRILLIC),
            _ => None,
        }
    }
}

/// WHATWG "get an encoding": strip ASCII whitespace, ASCII-lowercase the
/// label, and look it up in the spec's label table.
#[must_use]
pub fn encoding_from_label(label: &str) -> Option<Encoding> {
    fn is_ws(c: char) -> bool {
        matches!(c, '\u{9}' | '\u{A}' | '\u{C}' | '\u{D}' | ' ')
    }
    let trimmed = label.trim_matches(is_ws);
    if trimmed.is_empty() {
        return None;
    }
    // Fast path: labels are lowercase ASCII already.
    if let Some(enc) = data::label_to_encoding(trimmed) {
        return Some(enc);
    }
    let lowered = trimmed.to_ascii_lowercase();
    data::label_to_encoding(&lowered)
}

/// The result of [`decode`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeResult {
    /// The encoding actually used (BOM sniffing can override the fallback).
    pub encoding: Encoding,
    /// The decoded text; undecodable bytes became U+FFFD.
    pub text: String,
    /// Whether any decode error (U+FFFD substitution) occurred.
    pub had_errors: bool,
}

/// Why [`decode`] / [`decode_with`] could not produce text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// The encoding is known but this crate has no decoder for it yet
    /// (EUC-JP, ISO-2022-JP, Shift_JIS, EUC-KR).
    Unsupported(Encoding),
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DecodeError::Unsupported(enc) => {
                write!(f, "no decoder implemented for {}", enc.name())
            }
        }
    }
}

impl core::error::Error for DecodeError {}

/// WHATWG "decode": BOM-sniff `bytes` (a BOM overrides `fallback` and is
/// stripped), then decode with the resulting encoding, substituting U+FFFD
/// for undecodable input.
pub fn decode(bytes: &[u8], fallback: Encoding) -> Result<DecodeResult, DecodeError> {
    let (encoding, input) = sniff::strip_bom(bytes);
    let encoding = encoding.unwrap_or(fallback);
    let (text, had_errors) = decode_with(input, encoding)?;
    Ok(DecodeResult {
        encoding,
        text,
        had_errors,
    })
}

/// Decode with exactly `encoding`: no BOM sniffing, no substitution.
/// Undecodable bytes become U+FFFD and set the returned error flag.
pub fn decode_with(bytes: &[u8], encoding: Encoding) -> Result<(String, bool), DecodeError> {
    let mut text = String::new();
    let had_errors = decode::decode_into(encoding, bytes, &mut text)?;
    Ok((text, had_errors))
}

/// WHATWG "get an output encoding": replacement and UTF-16BE/LE become UTF-8
/// (used by URL parsing and form submission).
#[must_use]
pub const fn output_encoding(encoding: Encoding) -> Encoding {
    match encoding {
        Encoding::Replacement | Encoding::Utf16Be | Encoding::Utf16Le => Encoding::Utf8,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_versions_cover_all_tables() {
        // One entry per distinct index file used at codegen time.
        assert_eq!(data::INDEX_VERSIONS.len(), 30);
        for (file, ident, date) in data::INDEX_VERSIONS {
            assert!(file.starts_with("index-"), "{file}");
            assert_eq!(ident.len(), 64, "{file}");
            assert!(!date.is_empty(), "{file}");
        }
    }

    #[test]
    fn names_are_canonical_and_unique() {
        let mut names: Vec<&str> = Encoding::ALL.iter().map(|e| e.name()).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n);
        assert_eq!(Encoding::Utf8.name(), "utf-8");
        assert_eq!(Encoding::Windows1252.name(), "windows-1252");
        assert_eq!(Encoding::ShiftJis.name(), "shift_jis");
        assert_eq!(Encoding::Iso8859_8I.name(), "iso-8859-8-i");
    }

    #[test]
    fn every_name_resolves_to_itself() {
        for enc in Encoding::ALL {
            assert_eq!(encoding_from_label(enc.name()), Some(enc), "{}", enc.name());
        }
    }

    #[test]
    fn output_encoding_substitutions() {
        assert_eq!(output_encoding(Encoding::Replacement), Encoding::Utf8);
        assert_eq!(output_encoding(Encoding::Utf16Be), Encoding::Utf8);
        assert_eq!(output_encoding(Encoding::Utf16Le), Encoding::Utf8);
        assert_eq!(output_encoding(Encoding::Gbk), Encoding::Gbk);
        assert_eq!(
            output_encoding(Encoding::Windows1252),
            Encoding::Windows1252
        );
        assert_eq!(
            output_encoding(Encoding::XUserDefined),
            Encoding::XUserDefined
        );
    }
}
