#!/usr/bin/env python3
"""Build the compact Public Suffix List trie embedded by bowser_net.

Reads the canonical list from https://publicsuffix.org/list/public_suffix_list.dat
and writes a compact binary trie to ../src/psl_data.bin, which `psl.rs` loads
with `include_bytes!`.

Binary layout (all integers little-endian):
    0..8    magic "BPSLv001"
    8       u8: length of the list version string, then the version bytes
    ...     u8: length of the commit hash string, then the commit bytes
    ...     u32: number of parsed rules (informational)
    ...     trie root node

    node := flags:u8, child_count:u16, children...
    child := label_len:u8, label bytes, subtree_len:u32, subtree node...

    flags bit 0: an exact rule ends at this node      ("com", "co.uk")
    flags bit 1: a wildcard rule ends at this node    ("*.ck" stored at "ck")
    flags bit 2: an exception rule ends at this node  ("!www.ck" stored at "ck"."www")

Labels are stored reversed (TLD first), IDNA-encoded to ASCII and lowercased,
so the trie is pure ASCII. Regenerate with:

    python3 crates/bowser_net/psl/gen_psl.py /path/to/public_suffix_list.dat
"""

import struct
import sys

MAGIC = b"BPSLv001"

EXACT = 1
WILDCARD = 2
EXCEPTION = 4


class Node:
    __slots__ = ("children", "flags")

    def __init__(self):
        self.children = {}
        self.flags = 0


def parse_rules(path):
    """Return (rules, version, commit). rules are (flag_bit, [labels])."""
    rules = []
    version = ""
    commit = ""
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("//"):
                if line.startswith("// VERSION:"):
                    version = line.split(":", 1)[1].strip()
                elif line.startswith("// COMMIT:"):
                    commit = line.split(":", 1)[1].strip()
                continue
            if not line:
                continue
            flag = EXACT
            if line.startswith("!"):
                flag, line = EXCEPTION, line[1:]
            elif line.startswith("*."):
                flag, line = WILDCARD, line[2:]
            labels = []
            for part in line.split("."):
                # IDNA (nameprep) to ASCII; the stdlib codec implements IDNA 2003,
                # which is what the PSL's Unicode entries expect.
                lab = part.encode("idna").decode("ascii").lower()
                if not lab or len(lab) > 63 or any(
                    not (c.isascii() and (c.isalnum() or c == "-")) for c in lab
                ):
                    raise ValueError(f"bad label {part!r} in rule {raw.strip()!r}")
                labels.append(lab)
            if sum(map(len, labels)) + len(labels) - 1 > 253:
                raise ValueError(f"rule too long: {raw.strip()!r}")
            rules.append((flag, labels))
    return rules, version, commit


def build_trie(rules):
    root = Node()
    for flag, labels in rules:
        node = root
        for lab in reversed(labels):
            node = node.children.setdefault(lab, Node())
        node.flags |= flag
    return root


def serialize_node(node, out):
    out.append(node.flags & 0xFF)
    items = sorted(node.children.items())
    out += struct.pack("<H", len(items))
    for lab, child in items:
        labb = lab.encode("ascii")
        assert len(labb) <= 255
        child_buf = bytearray()
        serialize_node(child, child_buf)
        out.append(len(labb))
        out += labb
        out += struct.pack("<I", len(child_buf))
        out += child_buf


def count_nodes(node):
    return 1 + sum(count_nodes(c) for c in node.children.values())


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/public_suffix_list.dat", file=sys.stderr)
        sys.exit(2)
    rules, version, commit = parse_rules(sys.argv[1])
    root = build_trie(rules)

    out = bytearray()
    out += MAGIC
    vb, cb = version.encode("utf-8"), commit.encode("utf-8")
    assert len(vb) < 256 and len(cb) < 256
    out.append(len(vb))
    out += vb
    out.append(len(cb))
    out += cb
    out += struct.pack("<I", len(rules))
    serialize_node(root, out)

    dest = __import__("os").path.join(
        __import__("os").path.dirname(__file__), "..", "src", "psl_data.bin"
    )
    with open(dest, "wb") as f:
        f.write(out)
    print(f"rules={len(rules)} nodes={count_nodes(root)} bytes={len(out)} -> {dest}")
    print(f"version={version} commit={commit}")


if __name__ == "__main__":
    main()
