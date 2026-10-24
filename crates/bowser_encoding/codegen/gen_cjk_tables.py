#!/usr/bin/env python3
"""Append the WHATWG CJK index tables (JIS X 0208, JIS X 0212, EUC-KR) to
crates/bowser_encoding/src/data.rs.

The main generator (gen_tables.py) predates these tables and its input data
directory no longer exists, so this focused script handles just the three
tables the Shift_JIS / EUC-JP / EUC-KR decoders need. It is idempotent: if
the tables are already present in data.rs it changes nothing.

Input: directory with index-jis0208.txt, index-jis0212.txt, index-euc-kr.txt
       (from https://encoding.spec.whatwg.org/index-*.txt; the 2024-09-18
       vintage matches the rest of data.rs).

Usage:
    python3 gen_cjk_tables.py <index-data-dir> <path-to-data.rs>

Table sizes follow the existing convention of "max computable pointer + 1":
  - JIS0208: Shift_JIS pointer = (lead - leadOffset) * 188 + byte - offset,
    max (0xFC - 0xC1) * 188 + (0xFC - 0x41) = 11279 -> 11280 entries.
  - JIS0212: EUC-JP pointer = (lead - 0xA1) * 94 + byte - 0xA1,
    max 93 * 94 + 93 = 8835 -> 8836 entries.
  - EUCKR: EUC-KR pointer = (lead - 0x81) * 190 + byte - 0x41,
    max (0xFE - 0x81) * 190 + (0xFE - 0x41) = 23939 -> 23940 entries.

Sentinel 0xFFFF marks null entries (decoder maps them to U+FFFD).
"""

import re
import sys

DATA_DIR, DATA_RS = sys.argv[1], sys.argv[2]

TABLES = [
    # (index file, Rust name, size, doc comment)
    ("index-jis0208.txt", "JIS0208", 11280,
     "JIS X 0208 index: pointer -> code point (`0xFFFF` = null).\n"
     "/// Used by the Shift_JIS, EUC-JP and ISO-2022-JP decoders.\n"
     "/// Pointers 8836..=10715 are Windows EUDC: the Shift_JIS decoder maps\n"
     "/// them to U+E000..=U+E757 algorithmically per the standard, before\n"
     "/// consulting this table (the index file carries no entries there)."),
    ("index-jis0212.txt", "JIS0212", 8836,
     "JIS X 0212 index: pointer -> code point (`0xFFFF` = null).\n"
     "/// Only used by the EUC-JP decoder (0x8F-introduced pairs)."),
    ("index-euc-kr.txt", "EUCKR", 23940,
     "EUC-KR index (KS X 1001 + Unified Hangul Code): pointer -> code point\n"
     "/// (`0xFFFF` = null). Used by the EUC-KR decoder."),
]

LINE = re.compile(r"^\s*(\d+)\s+0x([0-9A-Fa-f]+)")
IDENT = re.compile(r"^# Identifier: ([0-9a-f]{64})")
DATE = re.compile(r"^# Date: (\S+)")


def parse_index(path):
    entries, ident, date = {}, None, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE.match(line)
            if m:
                i, cp = int(m.group(1)), int(m.group(2), 16)
                assert i not in entries, f"duplicate index {i} in {path}"
                assert cp <= 0xFFFF, f"code point > 0xFFFF in {path}"
                entries[i] = cp
            mi = IDENT.match(line)
            if mi:
                ident = mi.group(1)
            md = DATE.match(line)
            if md:
                date = md.group(1)
    assert ident and date, f"missing provenance in {path}"
    return entries, ident, date


def fmt_array(vals, per_line=12):
    out = []
    for i in range(0, len(vals), per_line):
        chunk = vals[i:i + per_line]
        out.append("    " + ", ".join("0x%04X" % v for v in chunk) + ",")
    return "\n".join(out)


blocks = []
for fname, name, size, doc in TABLES:
    entries, ident, date = parse_index(f"{DATA_DIR}/{fname}")
    assert max(entries) < size, f"{fname}: pointer {max(entries)} >= {size}"
    table = [entries.get(i, 0xFFFF) for i in range(size)]
    blocks.append(
        f"/// {doc}\n"
        f"/// Source: {fname}, Identifier: {ident}, Date: {date}.\n"
        f"pub(crate) static {name}: [u16; {size}] = [\n"
        f"{fmt_array(table)}\n"
        f"];\n"
    )

with open(DATA_RS, encoding="utf-8") as f:
    src = f.read()

if "pub(crate) static JIS0208" in src:
    print("tables already present; no change")
    sys.exit(0)

addition = (
    "\n"
    "// ---- CJK index tables (JIS0208/JIS0212/EUCKR) ----\n"
    "// Appended by codegen/gen_cjk_tables.py from the WHATWG index files;\n"
    "// do not edit by hand. Provenance per table is recorded above it.\n"
    "\n" + "\n".join(blocks)
)
if not src.endswith("\n"):
    src += "\n"
with open(DATA_RS, "w", encoding="utf-8") as f:
    f.write(src + addition)

print(f"appended {len(blocks)} tables: " + ", ".join(n for _, n, _, _ in TABLES))
