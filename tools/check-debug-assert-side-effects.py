#!/usr/bin/env python3
"""Wave gate: no side effects inside `debug_assert!` expressions.

`debug_assert!` / `debug_assert_eq!` / `debug_assert_ne!` compile their
arguments away in release builds. Housing a mutating call (e.g.
`map.insert`, `stack.pop`) inside one silently changes release behavior:
wave 42 found two live instances this way (an empty `ScrollTree::index`
-> 141 release-only scroll/window test failures in `bowser_app`, and an
unpopped getter placeholder -> stack corruption in `bowser_js` release).

Usage: python3 tools/check-debug-assert-side-effects.py
Exits nonzero listing offending spans.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PAT = re.compile(r"debug_assert(?:_eq|_ne)?!\s*\(", re.M)
# Method calls that (almost) always mutate or consume their receiver.
MUT = re.compile(
    r"\.(insert|remove|push|entry|clear|retain|extend|append|take|replace|"
    r"swap|set|or_insert|and_modify|drain|truncate|resize|sort|reverse|"
    r"shift|unshift|pop|get_or_insert|get_or_insert_with)\s*[\(<:]"
)


CHAR_LIT = re.compile(r"'([^'\\\n]|\\.)'")


def spans(src):
    """Yield (start, body) for each debug_assert!(...) span.

    The paren balancer skips char literals, string literals (raw or not),
    and comments: '(' / ')' inside them do not affect grouping, but a
    naive counter miscounts them (e.g. the '(' in '('), runs the span
    past the macro's real end, and reports a false positive.
    """
    n = len(src)
    for m in PAT.finditer(src):
        i, depth = m.end(), 1
        while i < n and depth:
            c = src[i]
            # Line comment.
            if c == "/" and i + 1 < n and src[i + 1] == "/":
                j = src.find("\n", i)
                i = n if j == -1 else j
                continue
            # Block comment (nesting is rare enough to ignore one level).
            if c == "/" and i + 1 < n and src[i + 1] == "*":
                j = src.find("*/", i + 2)
                i = n if j == -1 else j + 2
                continue
            # Raw string r"..." / r#"..."# / r##"..."##.
            if c == "r" and i + 1 < n and src[i + 1] in '#"':
                j = i + 1
                hashes = 0
                while j < n and src[j] == "#":
                    hashes += 1
                    j += 1
                if j < n and src[j] == '"':
                    close = '"' + "#" * hashes
                    k = src.find(close, j + 1)
                    i = n if k == -1 else k + len(close)
                    continue
            # Char literal, e.g. '(' (but not a lifetime like 'a).
            if c == "'":
                cm = CHAR_LIT.match(src, i)
                i = cm.end() if cm else i + 1
                continue
            # Ordinary string literal.
            if c == '"':
                i += 1
                while i < n and src[i] != '"':
                    i += 2 if src[i] == "\\" else 1
                i += 1
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        yield m.start(), src[m.start() : i]


def main():
    bad = []
    for path in sorted(ROOT.joinpath("crates").rglob("*.rs")):
        if "target" in path.parts:
            continue
        src = path.read_text(errors="ignore")
        for start, body in spans(src):
            if MUT.search(body):
                line = src[:start].count("\n") + 1
                one_line = " ".join(body.split())[:140]
                bad.append(f"{path.relative_to(ROOT)}:{line}: {one_line}")
    if bad:
        print("side effects inside debug_assert!:")
        for b in bad:
            print(" ", b)
        return 1
    print("ok: no side effects inside debug_assert! spans")
    return 0


if __name__ == "__main__":
    sys.exit(main())
