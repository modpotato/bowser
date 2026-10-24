#!/usr/bin/env python3
"""Generate crates/bowser_encoding/src/data.rs from official WHATWG encoding data.

Input: directory with encodings.json, index-*.txt, index-gb18030-ranges.txt
       (downloaded 2026-09-25 from https://encoding.spec.whatwg.org/index-*.txt
       and https://raw.githubusercontent.com/whatwg/encoding/main/encodings.json)

Output: data.rs with:
  - LABELS: exhaustive (label, Encoding) table, sorted for binary search
  - SINGLE_BYTE_TABLES: 27 x 128-entry u16 tables (0xFFFF = null)
  - GB18030_INDEX: 23940-entry u16 table (0xFFFF = null)
  - BIG5_INDEX: 19782-entry u32 table (0xFFFFFFFF = null)
  - GB18030_RANGES: sorted (pointer, code point) pairs for 4-byte sequences

Usage:
    python3 gen_tables.py /tmp/bowser-w-enc-data /tmp/bowser-w-enc/crates/bowser_encoding/src/data.rs

The WHATWG index files are public domain-ish reference data (see the Encoding
Standard's copyright notice, BSD-3-Clause for the standard's text). The numeric
tables below are facts derived from those files.
"""

import json
import re
import sys

DATA_DIR, OUT = sys.argv[1], sys.argv[2]

VARIANT = {
    "UTF-8": "Utf8", "IBM866": "Ibm866", "ISO-8859-2": "Iso88592",
    "ISO-8859-3": "Iso88593", "ISO-8859-4": "Iso88594", "ISO-8859-5": "Iso88595",
    "ISO-8859-6": "Iso88596", "ISO-8859-7": "Iso88597", "ISO-8859-8": "Iso88598",
    "ISO-8859-8-I": "Iso88598I", "ISO-8859-10": "Iso885910",
    "ISO-8859-13": "Iso885913", "ISO-8859-14": "Iso885914",
    "ISO-8859-15": "Iso885915", "ISO-8859-16": "Iso885916",
    "KOI8-R": "Koi8R", "KOI8-U": "Koi8U", "macintosh": "Macintosh",
    "windows-874": "Windows874", "windows-1250": "Windows1250",
    "windows-1251": "Windows1251", "windows-1252": "Windows1252",
    "windows-1253": "Windows1253", "windows-1254": "Windows1254",
    "windows-1255": "Windows1255", "windows-1256": "Windows1256",
    "windows-1257": "Windows1257", "windows-1258": "Windows1258",
    "x-mac-cyrillic": "XMacCyrillic", "GBK": "Gbk", "gb18030": "Gb18030",
    "Big5": "Big5", "EUC-JP": "EucJp", "ISO-2022-JP": "Iso2022Jp",
    "Shift_JIS": "ShiftJis", "EUC-KR": "EucKr", "replacement": "Replacement",
    "UTF-16BE": "Utf16Be", "UTF-16LE": "Utf16Le",
    "x-user-defined": "XUserDefined",
}

# encoding canonical name -> single-byte index file (ISO-8859-8-I shares ISO-8859-8)
SINGLE_BYTE = [
    ("IBM866", "index-ibm866.txt"), ("ISO-8859-2", "index-iso-8859-2.txt"),
    ("ISO-8859-3", "index-iso-8859-3.txt"), ("ISO-8859-4", "index-iso-8859-4.txt"),
    ("ISO-8859-5", "index-iso-8859-5.txt"), ("ISO-8859-6", "index-iso-8859-6.txt"),
    ("ISO-8859-7", "index-iso-8859-7.txt"), ("ISO-8859-8", "index-iso-8859-8.txt"),
    ("ISO-8859-10", "index-iso-8859-10.txt"),
    ("ISO-8859-13", "index-iso-8859-13.txt"),
    ("ISO-8859-14", "index-iso-8859-14.txt"),
    ("ISO-8859-15", "index-iso-8859-15.txt"),
    ("ISO-8859-16", "index-iso-8859-16.txt"), ("KOI8-R", "index-koi8-r.txt"),
    ("KOI8-U", "index-koi8-u.txt"), ("macintosh", "index-macintosh.txt"),
    ("windows-874", "index-windows-874.txt"),
    ("windows-1250", "index-windows-1250.txt"),
    ("windows-1251", "index-windows-1251.txt"),
    ("windows-1252", "index-windows-1252.txt"),
    ("windows-1253", "index-windows-1253.txt"),
    ("windows-1254", "index-windows-1254.txt"),
    ("windows-1255", "index-windows-1255.txt"),
    ("windows-1256", "index-windows-1256.txt"),
    ("windows-1257", "index-windows-1257.txt"),
    ("windows-1258", "index-windows-1258.txt"),
    ("x-mac-cyrillic", "index-x-mac-cyrillic.txt"),
]

LINE = re.compile(r"^\s*(\d+)\s+0x([0-9A-Fa-f]+)")


def parse_index(path):
    entries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE.match(line)
            if m:
                i, cp = int(m.group(1)), int(m.group(2), 16)
                assert i not in entries, f"duplicate index {i} in {path}"
                entries[i] = cp
    return entries


def fmt_array(vals, per_line, hex_fmt=None):
    out = []
    for i in range(0, len(vals), per_line):
        chunk = vals[i:i + per_line]
        if hex_fmt:
            out.append("    " + ", ".join(hex_fmt % v for v in chunk) + ",")
        else:
            out.append("    " + ", ".join(str(v) for v in chunk) + ",")
    return "\n".join(out)


with open(f"{DATA_DIR}/encodings.json", encoding="utf-8") as f:
    groups = json.load(f)
encodings = [e for g in groups for e in g["encodings"]]
assert len(encodings) == 40, len(encodings)
assert all(e["name"] in VARIANT for e in encodings)

labels = []
for e in encodings:
    for label in e["labels"]:
        labels.append((label, VARIANT[e["name"]]))
labels.sort(key=lambda p: p[0])
assert len({l for l, _ in labels}) == len(labels), "duplicate labels"

single_tables = []
for name, fname in SINGLE_BYTE:
    entries = parse_index(f"{DATA_DIR}/{fname}")
    assert all(0 <= i < 128 for i in entries), fname
    table = [entries.get(i, 0xFFFF) for i in range(128)]
    assert all(v <= 0xFFFF for v in table)
    single_tables.append((name, table))

gb = parse_index(f"{DATA_DIR}/index-gb18030.txt")
assert len(gb) == 23940 and min(gb) == 0 and max(gb) == 23939
assert all(v <= 0xFFFF for v in gb.values())
gb_table = [gb[i] for i in range(23940)]

big5 = parse_index(f"{DATA_DIR}/index-big5.txt")
assert max(big5) < 19782, max(big5)
assert all(v <= 0x10FFFF for v in big5.values())
# pointers 1133/1135/1164/1166 are special-cased two-code-point mappings, must be null
assert all(p not in big5 for p in (1133, 1135, 1164, 1166))
big5_table = [big5.get(i, 0xFFFFFFFF) for i in range(19782)]

ranges = []
with open(f"{DATA_DIR}/index-gb18030-ranges.txt", encoding="utf-8") as f:
    for line in f:
        m = LINE.match(line)
        if m:
            ranges.append((int(m.group(1)), int(m.group(2), 16)))
assert ranges == sorted(ranges), "ranges not sorted"
assert all(cp <= 0x10FFFF for _, cp in ranges)

with open(OUT, "w", encoding="utf-8") as o:
    o.write("""//! Generated character-encoding tables. DO NOT EDIT BY HAND.
//!
//! Regenerate with:
//!     python3 <workspace>/bowser-gen/gen_tables.py <index-data-dir> crates/bowser_encoding/src/data.rs
//! using the WHATWG Encoding Standard index files
//! (https://encoding.spec.whatwg.org/, index files dated 2024-09-18).
//! The tables are numeric facts derived from those files; the standard's text
//! is BSD-3-Clause.
//!
//! Sentinel values: `0xFFFF` (u16) / `0xFFFF_FFFF` (u32) mark null entries,
//! which decoders map to U+FFFD in replacement mode.

use crate::Encoding;

""")
    # labels
    o.write("/// Exhaustive WHATWG label table, sorted by label for binary search.\n")
    o.write("pub(crate) static LABELS: &[(&str, Encoding)] = &[\n")
    for label, var in labels:
        o.write(f'    ("{label}", Encoding::{var}),\n')
    o.write("];\n\n")

    # single-byte tables
    o.write("/// WHATWG single-byte indexes; slot `i` covers byte `0x80 + i`.\n")
    o.write("pub(crate) static SINGLE_BYTE_TABLES: [[u16; 128]; 27] = [\n")
    for _, table in single_tables:
        o.write("    [\n" + fmt_array(table, 16, "0x%04X") + "\n    ],\n")
    o.write("];\n\n")
    o.write("/// Maps a single-byte encoding to its index table.\n")
    o.write("/// ISO-8859-8-I shares the ISO-8859-8 index per the standard.\n")
    o.write("pub(crate) fn single_byte_table(encoding: Encoding) -> Option<&'static [u16; 128]> {\n")
    o.write("    let i = match encoding {\n")
    for n, (name, _) in enumerate(SINGLE_BYTE):
        o.write(f"        Encoding::{VARIANT[name]} => {n},\n")
    o.write("        Encoding::Iso88598I => 7,\n")
    o.write("        _ => return None,\n    };\n")
    o.write("    Some(&SINGLE_BYTE_TABLES[i])\n}\n\n")

    # gb18030 index
    o.write("/// gb18030 two-byte index: pointer -> code point.\n")
    o.write("pub(crate) static GB18030_INDEX: [u16; 23940] = [\n")
    o.write(fmt_array(gb_table, 12, "0x%04X") + "\n];\n\n")

    # big5 index
    o.write("/// Big5 index: pointer -> code point (`0xFFFF_FFFF` = null).\n")
    o.write("/// Pointers 1133/1135/1164/1166 are absent: they map to two code points\n")
    o.write("/// each and are special-cased in the decoder.\n")
    o.write("pub(crate) static BIG5_INDEX: [u32; 19782] = [\n")
    o.write(fmt_array(big5_table, 8, "0x%08X") + "\n];\n\n")

    # ranges
    o.write("/// gb18030 four-byte ranges: (range start pointer, start code point),\n")
    o.write("/// sorted by pointer. A pointer maps to the last range at or below it.\n")
    o.write(f"pub(crate) static GB18030_RANGES: [(u32, u32); {len(ranges)}] = [\n")
    for p, cp in ranges:
        o.write(f"    ({p}, 0x{cp:06X}),\n")
    o.write("];\n")

print(f"labels: {len(labels)}")
print(f"single-byte tables: {len(single_tables)}")
print(f"gb18030 index: {len(gb_table)}, big5 index: {len(big5_table)}, ranges: {len(ranges)}")
