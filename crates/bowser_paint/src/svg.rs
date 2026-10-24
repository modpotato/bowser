//! SVG intrinsic dimensions: `width`/`height`/`viewBox` from the root
//! `<svg>` element, per the SVG sizing rules used for `<img>`.
//!
//! This is metadata only — no rasterization. Absolute lengths
//! (unitless or `px`) are intrinsic; percentages and other units are
//! not (they resolve against the layout context). The `viewBox`
//! supplies the intrinsic aspect ratio when `width`/`height` are
//! absent.

/// Intrinsic dimensions parsed from an SVG document's root element.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SvgIntrinsics {
    /// Intrinsic width in px, from the `width` attribute (absolute
    /// length only).
    pub width: Option<f32>,
    /// Intrinsic height in px, from the `height` attribute (absolute
    /// length only).
    pub height: Option<f32>,
    /// Intrinsic aspect ratio (`viewBox` width / height).
    pub ratio: Option<f32>,
}

/// Parse the root `<svg>` element's `width`/`height`/`viewBox`
/// attributes from `bytes` (UTF-8 or ASCII SVG source).
///
/// Returns `None` when no `<svg>` root element is found. Never panics
/// on malformed input; unparseable attributes are simply absent from
/// the result.
pub fn svg_intrinsics(bytes: &[u8]) -> Option<SvgIntrinsics> {
    let text = std::str::from_utf8(bytes).ok()?;
    let tag = find_svg_root(text)?;
    let mut out = SvgIntrinsics {
        width: None,
        height: None,
        ratio: None,
    };
    for (name, value) in attributes(tag) {
        if name.eq_ignore_ascii_case("width") {
            out.width = parse_absolute_length(value);
        } else if name.eq_ignore_ascii_case("height") {
            out.height = parse_absolute_length(value);
        } else if name.eq_ignore_ascii_case("viewBox") {
            out.ratio = parse_viewbox_ratio(value);
        }
    }
    Some(out)
}

/// Locate the `<svg ...>` root tag: skip an XML declaration, doctype,
/// and comments; the first `<svg` open tag wins.
fn find_svg_root(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'<' {
            i += 1;
            continue;
        }
        let rest = &text[i..];
        if rest.starts_with("<!--") {
            // Skip comments.
            let end = rest.find("-->")?;
            i += end + 3;
            continue;
        }
        if rest.starts_with("<?") {
            // Skip the XML declaration / processing instructions.
            let end = rest.find("?>")?;
            i += end + 2;
            continue;
        }
        if rest.starts_with("<!") {
            // Skip doctype and other declarations.
            let end = rest.find('>')?;
            i += end + 1;
            continue;
        }
        if rest.starts_with("</") {
            return None; // A close tag before any `<svg>`: not an SVG doc.
        }
        // An open tag: check the name.
        let after_lt = &rest[1..];
        let name_end = after_lt
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == b'-' as char || c == b':' as char))
            .unwrap_or(after_lt.len());
        if after_lt[..name_end].eq_ignore_ascii_case("svg") {
            let end = rest.find('>')?;
            let tag_end = end + 1;
            return Some(&rest[..tag_end]);
        }
        // Some other element before `<svg>`: keep scanning (an SVG
        // document's root must be `<svg>`, but scanning on is harmless
        // — only the first `<svg>` tag's attributes are read).
        i += 1;
    }
    None
}

/// Iterate `(name, value)` attribute pairs of a tag's source text.
/// Handles `name="value"`, `name='value'`, and bare `name` (yielded
/// with an empty value). Malformed pairs are skipped.
fn attributes(tag: &str) -> Vec<(&str, &str)> {
    let bytes = tag.as_bytes();
    let mut out = Vec::new();
    // Skip `<name`.
    let mut i = 1;
    while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'>' && bytes[i] != b'/'
    {
        i += 1;
    }
    while i < bytes.len() {
        // Skip whitespace and the tag close.
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] == b'>' || bytes[i] == b'/' {
            break;
        }
        let name_start = i;
        while i < bytes.len()
            && !bytes[i].is_ascii_whitespace()
            && bytes[i] != b'='
            && bytes[i] != b'>'
            && bytes[i] != b'/'
        {
            i += 1;
        }
        let name = &tag[name_start..i];
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'=' {
            out.push((name, ""));
            continue;
        }
        i += 1; // '='
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            out.push((name, ""));
            break;
        }
        let value;
        if bytes[i] == b'"' || bytes[i] == b'\'' {
            let quote = bytes[i];
            i += 1;
            let value_start = i;
            while i < bytes.len() && bytes[i] != quote {
                i += 1;
            }
            value = &tag[value_start..i];
            i += 1; // closing quote (or end)
        } else {
            let value_start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'>' {
                i += 1;
            }
            value = &tag[value_start..i];
        }
        out.push((name, value));
    }
    out
}

/// Parse an SVG number: `[+-]? (\d+ (\.\d*)? | \.\d+) ([eE][+-]?\d+)?`.
fn parse_svg_number(text: &str) -> Option<f32> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let bytes = text.as_bytes();
    let mut i = 0;
    if bytes[i] == b'+' || bytes[i] == b'-' {
        i += 1;
    }
    let mut digits = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
        digits += 1;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return None;
    }
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        i += 1;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            return None;
        }
    }
    if i != bytes.len() {
        return None;
    }
    text.parse::<f32>().ok()
}

/// Parse an absolute SVG length: a number with no unit or `px`.
/// Percentages and other units are not intrinsic (`None`).
fn parse_absolute_length(value: &str) -> Option<f32> {
    let value = value.trim();
    let num = value.strip_suffix("px").unwrap_or(value);
    // A bare `%` or any other trailing unit disqualifies.
    if num.len() != value.len() && !value.ends_with("px") {
        return None;
    }
    if value.ends_with('%') {
        return None;
    }
    let n = parse_svg_number(num)?;
    if !n.is_finite() || n < 0.0 {
        return None;
    }
    Some(n)
}

/// Parse a `viewBox` (`min-x min-y width height`, whitespace/comma
/// separated) into its aspect ratio (`width / height`).
fn parse_viewbox_ratio(value: &str) -> Option<f32> {
    let parts: Vec<&str> = value
        .split(|c: char| c.is_ascii_whitespace() || c == ',')
        .filter(|p| !p.is_empty())
        .collect();
    if parts.len() != 4 {
        return None;
    }
    let w = parse_svg_number(parts[2])?;
    let h = parse_svg_number(parts[3])?;
    if w <= 0.0 || h <= 0.0 || !w.is_finite() || !h.is_finite() {
        return None;
    }
    Some(w / h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_height_attrs() {
        let svg = b"<svg xmlns='http://www.w3.org/2000/svg' width='100px' height='50'></svg>";
        let s = svg_intrinsics(svg).unwrap();
        assert_eq!(s.width, Some(100.0));
        assert_eq!(s.height, Some(50.0));
        assert_eq!(s.ratio, None);
    }

    #[test]
    fn viewbox_only() {
        let svg = b"<svg viewBox='0 0 100 10' xmlns='http://www.w3.org/2000/svg'></svg>";
        let s = svg_intrinsics(svg).unwrap();
        assert_eq!(s.width, None);
        assert_eq!(s.height, None);
        assert_eq!(s.ratio, Some(10.0));
    }

    #[test]
    fn no_dims_no_viewbox() {
        let svg = b"<svg xmlns='http://www.w3.org/2000/svg'><rect width='100%'/></svg>";
        let s = svg_intrinsics(svg).unwrap();
        assert_eq!(s.width, None);
        assert_eq!(s.height, None);
        assert_eq!(s.ratio, None);
    }

    #[test]
    fn percent_not_intrinsic() {
        let svg = b"<svg width='100%' height='100%' viewBox='0 0 4 2'></svg>";
        let s = svg_intrinsics(svg).unwrap();
        assert_eq!(s.width, None);
        assert_eq!(s.height, None);
        assert_eq!(s.ratio, Some(2.0));
    }

    #[test]
    fn skips_prolog_and_comments() {
        let svg = b"<?xml version='1.0'?><!-- hi --><!DOCTYPE svg><svg width='7'></svg>";
        let s = svg_intrinsics(svg).unwrap();
        assert_eq!(s.width, Some(7.0));
    }

    #[test]
    fn malformed_is_none_or_partial() {
        assert!(svg_intrinsics(b"<div></div>").is_none());
        assert!(svg_intrinsics(b"not xml at all").is_none());
        // Unparseable viewBox: no ratio, but the tag still parses.
        let s = svg_intrinsics(b"<svg viewBox='0 0 0 0'></svg>").unwrap();
        assert_eq!(s.ratio, None);
    }
}
