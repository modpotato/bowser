//! `parse_html` output snapshots: regression gate for the single-pass
//! refactor.
//!
//! Each record is `Document::parse_html(input)` rendered as
//! `{quirks:?}` + `serialize()` for every runnable html5lib corpus input
//! plus a set of hand-written edge inputs. The snapshots were generated
//! from the pre-refactor two-pass implementation (`parse_with_quirks` +
//! `append_converted`); this test fails if the single-pass sink path ever
//! produces a different tree.
//!
//! To regenerate (only when an intentional tree-construction change
//! lands): `BOWSER_GEN_SNAPSHOTS=1 cargo test -p bowser_dom
//! --test dom_snapshots`, then inspect the diff.
//!
//! Intentional updates (not pre-refactor output):
//! - `webkit02.dat#48` (`<font><select><option>a</option></font></select>`):
//!   updated 2026-09-28. Wave-H added `select` to the default in-scope
//!   boundary (`SCOPE_DEFAULT`, WHATWG 13.2.4.2): the standard removed the
//!   "in select" insertion modes, and the select-content rules now live in
//!   "in body" using default-scope checks, so the adoption agency for
//!   `</font>` finds `font` out of scope behind the open `select` and
//!   ignores the token (parse error, return). The pre-refactor snapshot
//!   recorded the old buggy tree (adoption agency split it into an empty
//!   `<font></font>` plus `<select><font><option>a</option></font></select>`);
//!   the new serialization matches the html5lib expected tree for this entry
//!   (`body > font > select > option > "a"`).

use bowser_dom::Document;

const SNAP_DIR: &str = "tests/fixtures/dom_snapshots";

/// Hand-written edge inputs covering paths the corpus barely touches:
/// deep nesting (iterative walk), template fragments, the
/// selectedcontent fixup, foreign content, PIs/comments/doctype.
fn edge_inputs() -> Vec<(&'static str, Vec<u8>)> {
    let mut deep = String::from("<!DOCTYPE html><div>");
    for _ in 0..20_000 {
        deep.push_str("<div>");
    }
    deep.push('x');
    for _ in 0..20_000 {
        deep.push_str("</div>");
    }
    deep.push_str("</div>");

    vec![
        ("deep-nesting", deep.into_bytes()),
        (
            "template-nesting",
            b"<template><div>a<template><p>b</p></template></div></template>".to_vec(),
        ),
        (
            "selectedcontent",
            b"<select><option>one</option><option selected>two</option><button><selectedcontent></selectedcontent></button></select>".to_vec(),
        ),
        (
            "foreign",
            b"<svg><foreignObject><div>x</div></foreignObject></svg><math><mi>y</mi><annotation-xml encoding=\"text/html\"><div>z</div></annotation-xml></math>".to_vec(),
        ),
        (
            "pi-comment-doctype",
            b"<?pi target data?><!--c--><!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01//EN\"><p>t".to_vec(),
        ),
        (
            "foster-adoption",
            b"<table><b><tr><td>cell</td></tr></b></table><a href=\"1\"><div><a href=\"2\">x</a></div></a>".to_vec(),
        ),
        (
            "formatting",
            b"<p><b>bold<i>both</b>italic</i>after</p>".to_vec(),
        ),
    ]
}

/// Extract `(input, fragment, script_on)` per case from a `.dat` file.
/// Ports the section scanning of `bowser_html`'s corpus runner; only the
/// `#data` payload is needed here.
fn dat_inputs(content: &str) -> Vec<(String, bool, bool)> {
    let mut cases = Vec::new();
    let mut lines = content.lines().peekable();
    while lines.peek().is_some() {
        while lines.peek().is_some_and(|l| l.is_empty()) {
            lines.next();
        }
        if lines.peek().is_none() {
            break;
        }
        assert_eq!(lines.next().unwrap(), "#data");
        let mut data = Vec::new();
        for line in lines.by_ref() {
            if line == "#errors" {
                break;
            }
            data.push(line);
        }
        let mut fragment = false;
        let mut script_on = false;
        let mut done = false;
        // After `#document-fragment`, the next line is the context element:
        // skip it.
        let mut skip_next = false;
        for line in lines.by_ref() {
            if skip_next {
                skip_next = false;
                continue;
            }
            match line {
                "#document-fragment" => {
                    fragment = true;
                    skip_next = true;
                }
                "#script-on" => script_on = true,
                "#document" => {
                    // Skip to the blank line ending the case. A blank line
                    // inside an open `"` text node is content, not a
                    // separator (same rule as the corpus runner).
                    let mut open_quote = false;
                    for dline in lines.by_ref() {
                        if dline.is_empty() && !open_quote {
                            break;
                        }
                        open_quote = if let Some(rest) = dline.strip_prefix("| ") {
                            let content = rest.trim_start_matches(' ');
                            content.starts_with('"')
                                && !(content.len() > 1 && content.ends_with('"'))
                        } else {
                            open_quote && !dline.ends_with('"')
                        };
                    }
                    done = true;
                    break;
                }
                _ => {}
            }
        }
        assert!(done, "case missing #document");
        cases.push((data.join("\n"), fragment, script_on));
    }
    cases
}

fn fixture_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bowser_html")
        .join("tests")
        .join("fixtures")
        .join("html5lib")
}

fn dat_files() -> Vec<std::path::PathBuf> {
    let root = fixture_root();
    let mut files = Vec::new();
    let mut dirs = vec![root];
    while let Some(dir) = dirs.pop() {
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .into_iter()
            .map(|e| e.path())
            .collect();
        entries.sort();
        for p in entries {
            if p.is_dir() {
                dirs.push(p);
            } else if p.extension().is_some_and(|x| x == "dat") {
                files.push(p);
            }
        }
    }
    files.sort();
    files
}

fn snap_path_for(dat: &std::path::Path) -> std::path::PathBuf {
    let rel = dat.strip_prefix(fixture_root()).unwrap();
    let mut name = rel
        .to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, "__");
    assert!(name.ends_with(".dat"));
    name.truncate(name.len() - 4);
    name.push_str(".snap");
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(SNAP_DIR)
        .join(name)
}

/// Render one case: quirks mode + full serialization.
fn render(input: &[u8]) -> (String, String) {
    let (doc, quirks) = Document::parse_html(input);
    (format!("{quirks:?}"), doc.serialize())
}

/// Write records as `CASE bytes={n} quirks={q} id={id}\n` + `{n}` bytes +
/// `\n`. Length-prefixing keeps the format robust against serialized text
/// containing newlines or `#`-prefixed lines.
fn write_records(path: &std::path::Path, records: &[(String, String, String)]) {
    let mut out = Vec::new();
    for (id, quirks, serialized) in records {
        let header = format!("CASE bytes={} quirks={quirks} id={id}\n", serialized.len());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(serialized.as_bytes());
        out.push(b'\n');
    }
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, out).unwrap();
}

fn read_records(path: &std::path::Path) -> Vec<(String, String, String)> {
    let data = std::fs::read(path).unwrap();
    let mut records = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let eol = data[pos..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|i| pos + i)
            .unwrap();
        let header = std::str::from_utf8(&data[pos..eol]).unwrap();
        let rest = header.strip_prefix("CASE ").unwrap();
        let (meta, id) = rest.split_once(" id=").unwrap();
        let bytes: usize = meta
            .split(' ')
            .find_map(|p| p.strip_prefix("bytes="))
            .unwrap()
            .parse()
            .unwrap();
        let quirks = meta
            .split(' ')
            .find_map(|p| p.strip_prefix("quirks="))
            .unwrap()
            .to_string();
        pos = eol + 1;
        let serialized = std::str::from_utf8(&data[pos..pos + bytes])
            .unwrap()
            .to_string();
        pos += bytes;
        assert_eq!(data[pos], b'\n');
        pos += 1;
        records.push((id.to_string(), quirks, serialized));
    }
    records
}

fn generate() {
    for dat in dat_files() {
        let content = std::fs::read_to_string(&dat).unwrap();
        let rel = dat
            .strip_prefix(fixture_root())
            .unwrap()
            .to_string_lossy()
            .to_string();
        let mut records = Vec::new();
        for (idx, (input, fragment, script_on)) in dat_inputs(&content).iter().enumerate() {
            if *fragment || *script_on {
                continue;
            }
            let id = format!("{rel}#{idx}");
            let (quirks, serialized) = render(input.as_bytes());
            records.push((id, quirks, serialized));
        }
        let path = snap_path_for(&dat);
        write_records(&path, &records);
        println!("wrote {} ({} cases)", path.display(), records.len());
    }
    // Edge inputs live in one shared file.
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(SNAP_DIR)
        .join("edge.snap");
    let records: Vec<_> = edge_inputs()
        .iter()
        .map(|(name, input)| {
            let (quirks, serialized) = render(input);
            (format!("edge/{name}"), quirks, serialized)
        })
        .collect();
    write_records(&path, &records);
    println!("wrote {} ({} cases)", path.display(), records.len());
}

fn corpus_case_inputs() -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    for dat in dat_files() {
        let content = std::fs::read_to_string(&dat).unwrap();
        let rel = dat
            .strip_prefix(fixture_root())
            .unwrap()
            .to_string_lossy()
            .to_string();
        for (idx, (input, fragment, script_on)) in dat_inputs(&content).iter().enumerate() {
            if *fragment || *script_on {
                continue;
            }
            out.push((format!("{rel}#{idx}"), input.as_bytes().to_vec()));
        }
    }
    for (name, input) in edge_inputs() {
        out.push((format!("edge/{name}"), input));
    }
    out
}

/// The selected option's own subtree contains `button > selectedcontent`.
/// The old interleaved fixup hung on this input (every fill created a new
/// target); collect-then-fill terminates. Per the WHATWG `selectedcontent`
/// post-connection steps, a `selectedcontent` with an `option` ancestor is
/// disabled, so no fill happens: the expected output is the input tree
/// unchanged (termination, not filling, is the property under test).
#[test]
fn selectedcontent_cycle_terminates() {
    let input = b"<select><option selected><button><selectedcontent></selectedcontent></button></option></select>";
    let (doc, _) = Document::parse_html(input);
    assert_eq!(
        doc.serialize(),
        "<html><head></head><body><select><option selected=\"\"><button><selectedcontent></selectedcontent></button></option></select></body></html>"
    );
}

#[test]
fn parse_html_matches_pre_refactor_snapshots() {
    if std::env::var("BOWSER_GEN_SNAPSHOTS").is_ok() {
        generate();
        return;
    }
    // Map snapshot id -> (quirks, serialized), across all .snap files.
    let snap_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(SNAP_DIR);
    let mut expected = std::collections::HashMap::new();
    let mut files: Vec<_> = std::fs::read_dir(&snap_dir)
        .unwrap_or_else(|e| {
            panic!(
                "snapshot dir missing ({}); run with BOWSER_GEN_SNAPSHOTS=1 to generate: {e}",
                snap_dir.display()
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .map(|e| e.path())
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no snapshot files");
    for f in &files {
        for (id, quirks, serialized) in read_records(f) {
            assert!(
                expected.insert(id.clone(), (quirks, serialized)).is_none(),
                "duplicate snapshot id {id}"
            );
        }
    }

    let mut checked = 0usize;
    let mut mismatches = Vec::new();
    for (id, input) in corpus_case_inputs() {
        let (exp_quirks, exp_serialized) = expected
            .get(&id)
            .unwrap_or_else(|| panic!("no snapshot for case {id}"));
        let (quirks, serialized) = render(&input);
        if &quirks != exp_quirks || &serialized != exp_serialized {
            mismatches.push(id.clone());
            if mismatches.len() < 5 {
                eprintln!(
                    "MISMATCH {id}: quirks {quirks} vs {exp_quirks}; serialized len {} vs {}",
                    serialized.len(),
                    exp_serialized.len()
                );
            }
        }
        checked += 1;
    }
    assert_eq!(
        checked,
        expected.len(),
        "snapshot/input count drift: {checked} inputs vs {} snapshots",
        expected.len()
    );
    assert!(
        mismatches.is_empty(),
        "{} snapshot mismatches (first: {:?})",
        mismatches.len(),
        &mismatches[..mismatches.len().min(5)]
    );
    println!("{checked} snapshot cases match");
}
