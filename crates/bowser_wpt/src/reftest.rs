//! Reftest primitives: reference extraction and screenshot comparison.
//!
//! A reftest passes when the rendering of the test matches every
//! `<link rel=match>` reference and differs from every
//! `<link rel=mismatch>` reference. The full pipeline (serving the test and
//! its references, rendering both, fuzzy metadata) is described in
//! `docs/reftest-design.md`; this module holds the pieces that do not need a
//! browser: pulling reference links out of a test file and comparing two
//! screenshots.

use std::fmt;
use std::path::Path;

use crate::driver::Screenshot;

/// A `<link rel=match|mismatch href=...>` found in a reftest file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    pub rel: RefRel,
    /// The raw `href` attribute value, resolved relative to the test later.
    pub href: String,
}

/// Whether the reference must match or must not match the test rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefRel {
    Match,
    Mismatch,
}

/// Extract `<link rel=match|mismatch href=...>` references from a reftest file.
///
/// This is a deliberately small scanner, not an HTML parser: it looks for
/// `<link` tags and reads their `rel` and `href` attributes. Attribute
/// values may be quoted or unquoted.
pub fn extract_references(path: &Path) -> Result<Vec<Reference>, ReftestError> {
    let text = std::fs::read_to_string(path).map_err(ReftestError::Io)?;
    Ok(extract_references_from_str(&text))
}

fn extract_references_from_str(text: &str) -> Vec<Reference> {
    let mut out = Vec::new();
    let lower = text.to_ascii_lowercase();
    let mut rest = lower.as_str();
    let orig = text;
    // Walk the lowercased copy but slice attributes from the original by
    // tracking byte offsets.
    let mut offset = 0;
    while let Some(i) = rest.find("<link") {
        // The tag name must end here: `<linkfoo` is not a link tag.
        let is_tag = rest[i + "<link".len()..]
            .chars()
            .next()
            .is_some_and(|c| c.is_whitespace() || c == '/' || c == '>');
        if !is_tag {
            offset += i + "<link".len();
            rest = &lower[offset..];
            continue;
        }
        let tag_start = offset + i;
        let tag_end = match lower[tag_start..].find('>') {
            Some(j) => tag_start + j,
            None => break,
        };
        let tag_orig = &orig[tag_start..tag_end];
        let tag_lower = &lower[tag_start..tag_end];
        if let (Some(rel), Some(href)) = (
            attr(tag_lower, tag_orig, "rel"),
            attr(tag_lower, tag_orig, "href"),
        ) {
            let rel = rel.to_ascii_lowercase();
            if rel == "match" {
                out.push(Reference {
                    rel: RefRel::Match,
                    href,
                });
            } else if rel == "mismatch" {
                out.push(Reference {
                    rel: RefRel::Mismatch,
                    href,
                });
            }
        }
        offset = tag_end + 1;
        rest = &lower[offset..];
    }
    out
}

/// Read attribute `name` from a tag. `tag_lower` and `tag_orig` must be the
/// same byte range; the value is sliced from the original to preserve case.
fn attr(tag_lower: &str, tag_orig: &str, name: &str) -> Option<String> {
    let mut rest = tag_lower;
    let mut orig_rest = tag_orig;
    loop {
        let i = rest.find(name)?;
        // Make sure this is the attribute name, not a suffix of another name.
        let before = rest[..i].chars().next_back();
        if before.is_some_and(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            rest = &rest[i + name.len()..];
            orig_rest = &orig_rest[i + name.len()..];
            continue;
        }
        let after = rest[i + name.len()..].trim_start();
        let after_orig = orig_rest[i + name.len()..].trim_start();
        // A same-prefix lookalike (`relx="..."`) leaves junk instead of `=`;
        // skip past it and keep searching rather than aborting the scan.
        let (Some(value_start), Some(value_start_orig)) =
            (after.strip_prefix('='), after_orig.strip_prefix('='))
        else {
            rest = &rest[i + name.len()..];
            orig_rest = &orig_rest[i + name.len()..];
            continue;
        };
        let value = value_start.trim_start();
        let value_orig = value_start_orig.trim_start();
        let raw: &str = if let Some(stripped) = value.strip_prefix('"') {
            let end = stripped.find('"')?;
            &value_orig[1..1 + end]
        } else if let Some(stripped) = value.strip_prefix('\'') {
            let end = stripped.find('\'')?;
            &value_orig[1..1 + end]
        } else {
            // Unquoted values run to whitespace or `>`; `/` is legal inside.
            let end = value
                .find(|c: char| c.is_whitespace() || c == '>')
                .unwrap_or(value.len());
            &value_orig[..end]
        };
        return Some(raw.to_owned());
    }
}

/// How strict a screenshot comparison is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReftestTolerance {
    /// Per-channel absolute difference still counted as "same".
    pub per_channel: u8,
    /// Fraction of pixels allowed to differ (0.0 = pixel-perfect).
    pub max_diff_fraction: f64,
}

impl ReftestTolerance {
    /// Pixel-perfect comparison.
    pub const EXACT: ReftestTolerance = ReftestTolerance {
        per_channel: 0,
        max_diff_fraction: 0.0,
    };
}

impl Default for ReftestTolerance {
    fn default() -> Self {
        ReftestTolerance::EXACT
    }
}

/// The result of comparing two screenshots.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    /// Whether the screenshots count as equal under the tolerance.
    pub equal: bool,
    pub diff_pixels: u64,
    pub diff_fraction: f64,
}

/// Compare two screenshots under `tolerance`.
///
/// Both screenshots must have identical dimensions; a size mismatch is an
/// error, not a pass or fail, because it usually means the driver or the
/// test setup is broken rather than the rendering.
pub fn compare(
    a: &Screenshot,
    b: &Screenshot,
    tolerance: ReftestTolerance,
) -> Result<Comparison, ReftestError> {
    if a.width != b.width || a.height != b.height {
        return Err(ReftestError::SizeMismatch {
            a: (a.width, a.height),
            b: (b.width, b.height),
        });
    }
    let total = a.width as u64 * a.height as u64;
    let mut diff_pixels: u64 = 0;
    for (pa, pb) in a
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.rgba.as_chunks::<4>().0)
    {
        let differs = pa
            .iter()
            .zip(pb.iter())
            .any(|(x, y)| x.abs_diff(*y) > tolerance.per_channel);
        if differs {
            diff_pixels += 1;
        }
    }
    let diff_fraction = if total == 0 {
        0.0
    } else {
        diff_pixels as f64 / total as f64
    };
    Ok(Comparison {
        equal: diff_fraction <= tolerance.max_diff_fraction,
        diff_pixels,
        diff_fraction,
    })
}

/// Reftest plumbing can fail on I/O or on incomparable screenshots.
#[derive(Debug)]
pub enum ReftestError {
    Io(std::io::Error),
    SizeMismatch { a: (u32, u32), b: (u32, u32) },
}

impl fmt::Display for ReftestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReftestError::Io(e) => write!(f, "reftest I/O error: {e}"),
            ReftestError::SizeMismatch { a, b } => {
                write!(
                    f,
                    "screenshot size mismatch: {}x{} vs {}x{}",
                    a.0, a.1, b.0, b.1
                )
            }
        }
    }
}

impl std::error::Error for ReftestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ReftestError::Io(e) => Some(e),
            ReftestError::SizeMismatch { .. } => None,
        }
    }
}

impl From<std::io::Error> for ReftestError {
    fn from(e: std::io::Error) -> Self {
        ReftestError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_match_and_mismatch() {
        let refs = extract_references_from_str(
            r#"<!doctype html>
            <link rel=match href="ref.html">
            <link rel="mismatch" href='other/ref2.html'>
            <link rel=stylesheet href="x.css">
            <link rel=MATCH href="upper.html">"#,
        );
        assert_eq!(
            refs,
            [
                Reference {
                    rel: RefRel::Match,
                    href: "ref.html".into()
                },
                Reference {
                    rel: RefRel::Mismatch,
                    href: "other/ref2.html".into()
                },
                Reference {
                    rel: RefRel::Match,
                    href: "upper.html".into()
                },
            ]
        );
    }

    #[test]
    fn extracts_nothing_without_ref_links() {
        assert!(extract_references_from_str("<link rel=stylesheet href=x>").is_empty());
        assert!(extract_references_from_str("no tags here").is_empty());
    }

    #[test]
    fn unquoted_href_keeps_slashes() {
        let refs = extract_references_from_str("<link rel=match href=other/ref2.html>");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].href, "other/ref2.html");
    }

    #[test]
    fn same_prefix_lookalike_does_not_abort_scan() {
        let refs = extract_references_from_str(r#"<link relx="y" rel="match" href="z.html">"#);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].rel, RefRel::Match);
        assert_eq!(refs[0].href, "z.html");
    }

    #[test]
    fn linkfoo_is_not_a_link_tag() {
        assert!(extract_references_from_str(r#"<linkfoo rel=match href="x.html">"#).is_empty());
        let refs = extract_references_from_str(
            r#"<linkfoo rel=match href="x.html"><link rel=match href="y.html">"#,
        );
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].href, "y.html");
    }

    #[test]
    fn exact_compare() {
        let a = Screenshot::solid(4, 4, 10, 20, 30, 255);
        let b = Screenshot::solid(4, 4, 10, 20, 30, 255);
        let c = Screenshot::solid(4, 4, 11, 20, 30, 255);
        let r = compare(&a, &b, ReftestTolerance::EXACT).unwrap();
        assert!(r.equal);
        assert_eq!(r.diff_pixels, 0);
        assert_eq!(r.diff_fraction, 0.0);
        let r = compare(&a, &c, ReftestTolerance::EXACT).unwrap();
        assert!(!r.equal);
        assert_eq!(r.diff_pixels, 16);
        assert_eq!(r.diff_fraction, 1.0);
    }

    #[test]
    fn tolerance_allows_small_differences() {
        let a = Screenshot::solid(10, 10, 100, 100, 100, 255);
        let mut b = a.clone();
        // One pixel off by 2 in one channel.
        b.rgba[0] = 102;
        let tol = ReftestTolerance {
            per_channel: 1,
            max_diff_fraction: 0.0,
        };
        assert!(!compare(&a, &b, tol).unwrap().equal);
        let tol = ReftestTolerance {
            per_channel: 2,
            max_diff_fraction: 0.0,
        };
        assert!(compare(&a, &b, tol).unwrap().equal);
        // Fraction tolerance: 1 of 100 pixels may differ.
        let tol = ReftestTolerance {
            per_channel: 0,
            max_diff_fraction: 0.01,
        };
        let r = compare(&a, &b, tol).unwrap();
        assert!(r.equal);
        assert_eq!(r.diff_pixels, 1);
    }

    #[test]
    fn size_mismatch_is_error() {
        let a = Screenshot::solid(4, 4, 0, 0, 0, 255);
        let b = Screenshot::solid(8, 4, 0, 0, 0, 255);
        let err = compare(&a, &b, ReftestTolerance::EXACT).unwrap_err();
        assert!(matches!(err, ReftestError::SizeMismatch { .. }));
        assert!(err.to_string().contains("4x4 vs 8x4"));
    }
}
