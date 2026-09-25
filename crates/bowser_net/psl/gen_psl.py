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

Round-trip the generated output without touching the committed bin:

    python3 crates/bowser_net/psl/gen_psl.py --check /path/to/public_suffix_list.dat
"""

import os
import struct
import sys
import tempfile

MAGIC = b"BPSLv001"

EXACT = 1
WILDCARD = 2
EXCEPTION = 4


class Node:
    __slots__ = ("children", "flags")

    def __init__(self):
        self.children = {}
        self.flags = 0


def idna_encode(label):
    """ASCII-encode one label. The PSL's Unicode entries are defined under
    IDNA 2008 / UTS-46 while the stdlib codec implements the older IDNA 2003;
    the current list encodes byte-identically under both (verified against
    all 493 non-ASCII labels with a UTS-46 implementation), so the stdlib
    codec suffices today — but prefer a UTS-46 implementation if the two
    ever diverge for a future list."""
    return label.encode("idna").decode("ascii").lower()


def parse_rules(path):
    """Return (rules, idn_rules, version, commit).

    rules are (flag_bit, [labels]); idn_rules is the subset whose raw text
    contained non-ASCII labels (for the --check IDN sample). The PSL format
    reads each line only up to the first whitespace, so tokenize before any
    comment/wildcard/exception handling.
    """
    rules = []
    idn_rules = []
    version = ""
    commit = ""
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("//"):
                if line.startswith("// VERSION:"):
                    version = line.split(":", 1)[1].strip()
                elif line.startswith("// COMMIT:"):
                    commit = line.split(":", 1)[1].strip()
                continue
            token = line.split()[0]
            flag = EXACT
            if token.startswith("!"):
                flag, token = EXCEPTION, token[1:]
            elif token.startswith("*."):
                flag, token = WILDCARD, token[2:]
            labels = []
            for part in token.split("."):
                lab = idna_encode(part)
                if not lab or len(lab) > 63 or any(
                    not (c.isascii() and (c.isalnum() or c == "-")) for c in lab
                ):
                    raise ValueError(f"bad label {part!r} in rule {raw.strip()!r}")
                labels.append(lab)
            if sum(map(len, labels)) + len(labels) - 1 > 253:
                raise ValueError(f"rule too long: {raw.strip()!r}")
            rules.append((flag, labels))
            if any(ord(c) > 127 for c in token):
                idn_rules.append((flag, labels))
    return rules, idn_rules, version, commit


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


def encode_bin(rules, version, commit, root):
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
    return bytes(out)


# Spot-check rules that must round-trip through the generated trie, as
# (rule text, expected flag bit). Hardcoded independently of the parser so a
# parser/serializer skew that drops rules consistently still fails loudly.
# Unicode entries go through the same IDNA path as the parser.
KNOWN_RULES = [
    ("com", EXACT),
    ("co.uk", EXACT),
    ("*.ck", WILDCARD),
    ("!www.ck", EXCEPTION),
    ("*.kawasaki.jp", WILDCARD),
    ("!city.kawasaki.jp", EXCEPTION),
    ("*.yokohama.jp", WILDCARD),
    ("xn--55qx5d.cn", EXACT),  # xn--55qx5d.cn, ASCII form
    ("公司.cn", EXACT),  # same rule, Unicode form (exercises the IDNA path)
    ("appspot.com", EXACT),  # private section
]

# The canonical list currently carries 8 exception rules; pin the count so a
# silently dropped exception fails loudly. Update when the list changes.
KNOWN_EXCEPTION_COUNT = 8


def read_bin(path):
    """Return (version, commit, rule_count, trie_bytes) from a generated bin."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != MAGIC:
        raise ValueError(f"{path}: bad magic {data[:8]!r}")
    pos = 8
    vl = data[pos]
    version = data[pos + 1 : pos + 1 + vl].decode("utf-8")
    pos += 1 + vl
    cl = data[pos]
    commit = data[pos + 1 : pos + 1 + cl].decode("utf-8")
    pos += 1 + cl
    (rule_count,) = struct.unpack_from("<I", data, pos)
    pos += 4
    return version, commit, rule_count, data[pos:]


def trie_lookup(trie, labels):
    """Walk reversed labels from the trie root; return the node's flags."""
    off = 0
    for lab in reversed(labels):
        labb = lab.encode("ascii")
        (count,) = struct.unpack_from("<H", trie, off + 1)
        pos = off + 3
        for _ in range(count):
            llen = trie[pos]
            pos += 1
            clab = trie[pos : pos + llen]
            pos += llen
            (slen,) = struct.unpack_from("<I", trie, pos)
            pos += 4
            if clab == labb:
                off = pos
                break
            pos += slen
        else:
            raise KeyError(f"rule labels {labels!r}: {lab!r} not in trie")
    return trie[off]


def check_rule(trie, text, flag):
    """Assert the rule `text` resolves in the trie with the expected flag."""
    body = text
    if body.startswith("!"):
        body = body[1:]
    elif body.startswith("*."):
        body = body[2:]
    labels = [idna_encode(part) for part in body.split(".")]
    got = trie_lookup(trie, labels)
    name = {EXACT: "exact", WILDCARD: "wildcard", EXCEPTION: "exception"}[flag]
    if not got & flag:
        raise AssertionError(f"rule {text!r}: expected {name} flag, node has {got:#05b}")


def check_bin(path, rules, idn_rules, version, commit):
    """Re-read the generated bin and assert sampled rules round-trip."""
    bversion, bcommit, rule_count, trie = read_bin(path)
    assert bversion == version, f"version skew: {bversion!r} != {version!r}"
    assert bcommit == commit, f"commit skew: {bcommit!r} != {commit!r}"
    assert rule_count == len(rules), f"rule count skew: {rule_count} != {len(rules)}"

    exceptions = [r for r in rules if r[0] == EXCEPTION]
    assert len(exceptions) == KNOWN_EXCEPTION_COUNT, (
        f"expected {KNOWN_EXCEPTION_COUNT} exception rules, found {len(exceptions)}; "
        "update KNOWN_EXCEPTION_COUNT when the list changes"
    )
    checked = 0
    idn_set = {".".join(labels) for _, labels in idn_rules}
    # Every exception rule, every wildcard, every IDN rule, and a
    # deterministic sample of exact rules must round-trip with their flag.
    for i, (flag, labels) in enumerate(rules):
        if (
            flag == EXCEPTION
            or flag == WILDCARD
            or ".".join(labels) in idn_set
            or (flag == EXACT and i % 97 == 0)
        ):
            got = trie_lookup(trie, labels)
            assert got & flag, f"rule {'.'.join(labels)!r} lost its flag {flag:#05b}"
            checked += 1
    # Plus parser-independent known rules.
    for text, flag in KNOWN_RULES:
        check_rule(trie, text, flag)
        checked += 1
    return checked


def main():
    args = sys.argv[1:]
    check = "--check" in args
    args = [a for a in args if a != "--check"]
    out_path = None
    if "--out" in args:
        i = args.index("--out")
        if i + 1 >= len(args):
            print("usage: --out needs a path", file=sys.stderr)
            sys.exit(2)
        out_path = args[i + 1]
        args = args[:i] + args[i + 2 :]
    if len(args) != 1:
        print(
            f"usage: {sys.argv[0]} [--check] [--out PATH] /path/to/public_suffix_list.dat",
            file=sys.stderr,
        )
        sys.exit(2)
    rules, idn_rules, version, commit = parse_rules(args[0])
    root = build_trie(rules)
    blob = encode_bin(rules, version, commit, root)

    if check:
        # Never touch the committed bin in check mode; verify a temp copy
        # (or --out) and report the round-trip.
        tmp = out_path is None
        if tmp:
            fd, out_path = tempfile.mkstemp(suffix=".bin", prefix="psl-check-")
            os.close(fd)
        with open(out_path, "wb") as f:
            f.write(blob)
        try:
            checked = check_bin(out_path, rules, idn_rules, version, commit)
        finally:
            if tmp:
                os.unlink(out_path)
        print(f"check OK: {checked} sampled rules round-tripped ({len(rules)} total)")
        print(f"version={version} commit={commit}")
        return

    dest = out_path or os.path.join(
        os.path.dirname(__file__), "..", "src", "psl_data.bin"
    )
    with open(dest, "wb") as f:
        f.write(blob)
    print(f"rules={len(rules)} nodes={count_nodes(root)} bytes={len(blob)} -> {dest}")
    print(f"version={version} commit={commit}")


if __name__ == "__main__":
    main()
