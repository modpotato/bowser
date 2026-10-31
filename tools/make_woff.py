#!/usr/bin/env python3
"""Build WOFF1 fixtures from Ahem.ttf (test-fixture generation only).

Reads the SFNT table directory, zlib-compresses each table (or stores
it raw for the -stored variant), and writes a WOFF1 file per the WOFF
1.0 spec: 44-byte header, 20-byte table records, table data, no
metadata / private data. SFNT tables are emitted in the source file's
order. For a source whose directory is already tag-sorted (Ahem's
is), decode rebuilds the original SFNT byte-for-byte. That
byte-for-byte claim holds only for such sorted inputs: the decoder
sorts records by tag, so an unsorted source would round-trip with
the same table bytes but a re-sorted directory, not byte-for-byte.

An optional fourth argument writes a WOFF2 file via fontTools
(`TTFont` with `flavor = "woff2"`, default transforms: transformed
`glyf`/`loca`, untransformed `hmtx` for Ahem). Regenerating is
equivalent but not byte-identical: fontTools stamps `head.modified`
on save (and recomputes `checkSumAdjustment` from it), so the
`head` bytes differ between runs; every other table is stable.
WOFF2 round-tripping is semantic, not byte-for-byte: the decoder
re-encodes glyph coordinates (no short forms) and pads short-`loca`
glyphs to even lengths, so its rebuilt `glyf`/`loca` differ in bytes
from the source while describing identical glyphs.

An optional fifth argument writes a second WOFF2 file with the
`hmtx` transform enabled as well (fontTools `compress` with
`transform_tables={"glyf", "loca", "hmtx"}`); it fails loudly if
fontTools decides the `hmtx` transform does not apply (e.g. a font
whose side bearings do not match its xMins), so a stale fixture
can never silently become an untransformed-`hmtx` file.

`make_woff.py synthetic <outdir>` instead writes the synthetic
WOFF2 fixtures used by the `decode_woff2` adversarial tests in
`crates/bowser_paint/src/fonts.rs`. These are hand-built minimal
fonts (a few glyphs, hand-encoded transformed `glyf` streams),
not repacks of Ahem: each file isolates one decoder path
(transformed `hmtx`, head `indexToLocFormat` patching, composite
glyphs with x/y-scale + 2x2 matrix + instructions, empty-glyph
bboxes, truncated substreams, and the various rejection cases).
Regenerating them must produce the same test outcomes; the files
are small enough that any drift is visible in the test suite.
"""
import os
import struct
import sys
import zlib


def build(ttf: bytes, stored: bool) -> bytes:
    (flavor, num_tables, _sr, _es, _rs) = struct.unpack(">IHHHH", ttf[:12])
    tables = []
    for i in range(num_tables):
        tag, checksum, offset, length = struct.unpack(
            ">4sIII", ttf[12 + i * 16: 28 + i * 16]
        )
        data = ttf[offset:offset + length]
        assert len(data) == length
        comp = data if stored else zlib.compress(data, 9)
        tables.append((tag, checksum, comp, length))
    header_size = 44 + num_tables * 20
    total_sfnt = 12 + num_tables * 16 + sum((length + 3) & ~3 for _, _, _, length in tables)
    out = bytearray()
    out += struct.pack(
        ">4sIIHHIHHIIIII",
        b"wOFF", flavor, 0, num_tables, 0, total_sfnt,
        1, 0, 0, 0, 0, 0, 0,
    )
    assert len(out) == 44, len(out)
    offset = header_size
    for tag, checksum, comp, length in tables:
        out += struct.pack(">4sIIII", tag, offset, len(comp), length, checksum)
        offset += len(comp)
    assert len(out) == header_size
    for _tag, _checksum, comp, _length in tables:
        out += comp
    # Patch the total WOFF length.
    struct.pack_into(">I", out, 8, len(out))
    return bytes(out)


# ---------------------------------------------------------------------------
# Synthetic WOFF2 fixtures (see the module docstring).
# ---------------------------------------------------------------------------

_WOFF2_KNOWN = [
    b"cmap", b"head", b"hhea", b"hmtx", b"maxp", b"name", b"OS/2", b"post",
    b"cvt ", b"fpgm", b"glyf", b"loca", b"prep", b"CFF ", b"VORG", b"EBDT",
    b"EBLC", b"gasp", b"hdmx", b"kern", b"LTSH", b"PCLT", b"VDMX", b"vhea",
    b"vmtx", b"BASE", b"GDEF", b"GPOS", b"GSUB", b"EBSC", b"JSTF", b"MATH",
    b"CBDT", b"CBLC", b"COLR", b"CPAL", b"SVG ", b"sbix", b"acnt", b"avar",
    b"bdat", b"bloc", b"bsln", b"cvar", b"fdsc", b"feat", b"fmtx", b"fvar",
    b"gvar", b"hsty", b"just", b"lcar", b"mort", b"morx", b"opbd", b"prop",
    b"trak", b"Zapf", b"Silf", b"Glat", b"Gloc", b"Feat", b"Sill",
]
_WOFF2_KNOWN_IDX = {t: i for i, t in enumerate(_WOFF2_KNOWN)}


def _base128(v: int) -> bytes:
    if v == 0:
        return b"\x00"
    groups = []
    while v:
        groups.append(v & 0x7F)
        v >>= 7
    out = bytearray()
    for g in reversed(groups[1:]):
        out.append(g | 0x80)
    out.append(groups[0])
    return bytes(out)


def _u255(v: int) -> bytes:
    if v < 253:
        return bytes([v])
    if v <= 65535:
        return b"\xfd" + struct.pack(">H", v)
    raise ValueError(v)


def _head(index_format: int) -> bytes:
    h = bytearray(54)
    struct.pack_into(">H", h, 50, index_format)
    return bytes(h)


def _maxp(num: int) -> bytes:
    # Full 32-byte maxp v1.0 table.
    return struct.pack(">IH13H", 0x00010000, num, *([0] * 13))


def _hhea(num_h: int) -> bytes:
    h = bytearray(36)
    struct.pack_into(">H", h, 34, num_h)
    return bytes(h)


def _triplet(dx: int, dy: int, on_curve: bool) -> bytes:
    # Full-precision triplet (flag 124..127): magnitudes are the
    # absolute deltas, sign bits are flag bits 0 (x) and 1 (y),
    # bit 7 marks an off-curve point.
    flag = 124 | (1 if dx >= 0 else 0) | (2 if dy >= 0 else 0)
    if not on_curve:
        flag |= 0x80
    return bytes([flag]) + struct.pack(">h", abs(dx)) + struct.pack(">h", abs(dy))


def _build_glyf(glyphs, index_format: int) -> bytes:
    """Transformed `glyf` stream. Each glyph is a dict:
      empty: {'kind': 'empty', 'bbox': None | 8 bytes} (explicit bbox)
      simple: {'kind': 'simple', 'points': [(x, y, on_curve)],
               'contours': [end_pts], 'bbox': None | (xMin, yMin, xMax, yMax),
               'instructions': bytes}
      composite: {'kind': 'composite',
                  'components': [(flags, glyph_index, args, scale)],
                  'bbox': (xMin, yMin, xMax, yMax),
                  'instructions': bytes | None}
    """
    n_contour = bytearray()
    n_points = bytearray()
    flag = bytearray()
    glyph_stream = bytearray()
    composite = bytearray()
    bbox_data = bytearray()
    instr = bytearray()
    bitmap = bytearray(((len(glyphs) + 31) // 32) * 4)
    for gid, g in enumerate(glyphs):
        if g["kind"] == "empty":
            n_contour += struct.pack(">h", 0)
            if g.get("bbox") is not None:
                bitmap[gid >> 3] |= 0x80 >> (gid & 7)
                bbox_data += g["bbox"]
        elif g["kind"] == "simple":
            pts = g["points"]
            n_contour += struct.pack(">h", len(g["contours"]))
            prev_end = -1
            for end in g["contours"]:
                n_points += _u255(end - prev_end)
                prev_end = end
            px = py = 0
            for (x, y, on) in pts:
                trip = _triplet(x - px, y - py, on)
                flag.append(trip[0])
                glyph_stream += trip[1:]
                px, py = x, y
            ins = g.get("instructions", b"")
            glyph_stream += _u255(len(ins))
            instr += ins
            if g.get("bbox") is not None:
                bitmap[gid >> 3] |= 0x80 >> (gid & 7)
                bbox_data += struct.pack(">hhhh", *g["bbox"])
        elif g["kind"] == "composite":
            n_contour += struct.pack(">h", -1)
            for (cflags, gidx, args, scale) in g["components"]:
                composite += struct.pack(">HH", cflags, gidx)
                composite += args
                composite += scale
            bitmap[gid >> 3] |= 0x80 >> (gid & 7)
            bbox_data += struct.pack(">hhhh", *g["bbox"])
            if g.get("instructions") is not None:
                glyph_stream += _u255(len(g["instructions"]))
                instr += g["instructions"]
        else:
            raise ValueError(g["kind"])
    bbox = bytes(bitmap) + bytes(bbox_data)
    streams = [bytes(n_contour), bytes(n_points), bytes(flag), bytes(glyph_stream),
               bytes(composite), bbox, bytes(instr)]
    out = bytearray()
    out += struct.pack(">HHHH", 0, 0, len(glyphs), index_format)
    for s in streams:
        out += struct.pack(">I", len(s))
    for s in streams:
        out += s
    return bytes(out)


def _build_woff2(tables, total_sfnt=1000, flavor=0x00010000) -> bytes:
    """tables: list of (tag, orig_len, stream_bytes, version) where
    version is the WOFF2 transform version (glyf/loca: 0 transformed,
    3 null; hmtx: 1 transformed, 0 null; anything else: 0)."""
    import brotli

    directory = bytearray()
    stream = bytearray()
    for tag, orig, sdata, version in tables:
        if tag in _WOFF2_KNOWN_IDX:
            directory.append((version << 6) | _WOFF2_KNOWN_IDX[tag])
        else:
            directory.append((version << 6) | 63)
            directory += tag
        directory += _base128(orig)
        transformed = (
            (tag in (b"glyf", b"loca") and version == 0)
            or (tag == b"hmtx" and version == 1)
        )
        if transformed:
            directory += _base128(len(sdata))
        stream += sdata
    comp = brotli.compress(bytes(stream))
    total_len = 48 + len(directory) + len(comp)
    # WOFF2 header: sig, flavor, length, numTables, reserved,
    # totalSfntSize, totalCompressedSize, major, minor, metaOffset,
    # metaLength, metaOrigLength, privOffset, privLength.
    hdr = struct.pack(">4sIIHHIIHHIIIII", b"wOF2", flavor, total_len,
                      len(tables), 0, total_sfnt, len(comp), 1, 0,
                      0, 0, 0, 0, 0)
    assert len(hdr) == 48, len(hdr)
    return bytes(hdr) + bytes(directory) + comp


def _raw(tag, data):
    return (tag, len(data), data, 0)


def _glyf_tables(glyphs, index_format, loca_orig, extra):
    return [(b"glyf", 100, _build_glyf(glyphs, index_format), 0),
            (b"loca", loca_orig, b"", 0)] + extra


_SYNTH_SIMPLE_2PT = {"kind": "simple", "points": [(0, 0, False), (10, 5, True)],
                     "contours": [1], "instructions": b""}
_SYNTH_COMP = [
    # MORE|XY|WORDS: 4-byte args, then 4-byte x/y scale.
    (0x0061, 0, struct.pack(">hh", 3, 4), struct.pack(">hh", 0x0100, 0x0200)),
    # 2x2 | INSTR, last, WORDS clear: 2-byte args, then 8-byte matrix.
    (0x0180, 0, bytes([5, 6]), struct.pack(">hhhh", 0x0100, 0, 0, 0x0100)),
]
_SYNTH_BASE_GLYPHS = [
    _SYNTH_SIMPLE_2PT,
    {"kind": "composite", "components": _SYNTH_COMP, "bbox": (1, 2, 30, 40),
     "instructions": b"\x01\x02"},
    # Trailing empty glyph: no explicit bbox. WOFF2 REC §5.1
    # rejects any explicit bbox on an empty glyph, so the
    # bit must stay clear here or every base-derived
    # fixture would reject.
    {"kind": "empty"},
]
# Untransformed 3-glyph font for the raw-`glyf`/`loca` hmtx cases:
# glyph0 empty, glyph1 xMin 10, glyph2 xMin 30; short loca 0,0,5,10
# (offsets halved: 0, 0, 10, 20).
_SYNTH_RAW_GLYF = (b"\x00\x01\x00\x0a\x00\x00\x00\x14\x00\x0a"
                   b"\x00\x01\x00\x1e\x00\x00\x00\x28\x00\x14")
_SYNTH_RAW_LOCA = struct.pack(">HHHH", 0, 0, 5, 10)


def _synth_base_tables(glyphs, index_format, loca_orig):
    return _glyf_tables(glyphs, index_format, loca_orig, [
        _raw(b"head", _head(index_format)),
        _raw(b"maxp", _maxp(len(glyphs))),
    ])


def synthetic(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    def write(name, data):
        with open(os.path.join(outdir, name), "wb") as f:
            f.write(data)

    # Positive: transformed glyf, long loca, head fmt mismatch (0 -> 1),
    # off-curve first point on the simple glyph.
    glyphs = [
        {"kind": "empty"},
        {"kind": "simple", "points": [(0, 0, False), (10, 5, True)],
         "contours": [1], "instructions": b""},
    ]
    tabs = _glyf_tables(glyphs, 1, 12, [
        _raw(b"head", _head(0)),
        _raw(b"maxp", _maxp(2)),
        _raw(b"hmtx", struct.pack(">HHHH", 500, 0, 600, 0)),
    ])
    write("woff2-synth-head-patch.woff2", _build_woff2(tabs))

    # Positive: composite end-to-end (XY scale + 2x2 matrix +
    # instructions) plus an empty glyph with no explicit bbox.
    write("woff2-synth-composite.woff2",
          _build_woff2(_synth_base_tables(_SYNTH_BASE_GLYPHS, 1, 16)))

    # Negative: empty glyph with a zero explicit bbox, then a
    # simple glyph with an explicit bbox. WOFF2 REC §5.1
    # rejects any explicit bbox on an empty glyph, zero
    # included, so this is a reject case alongside the
    # nonzero fixture below.
    glyphs = [
        {"kind": "empty", "bbox": b"\x00" * 8},
        {"kind": "simple", "points": [(0, 0, True), (10, 5, True)],
         "contours": [1], "bbox": (3, 4, 10, 5), "instructions": b""},
    ]
    write("woff2-synth-empty-bbox-zero.woff2",
          _build_woff2(_synth_base_tables(glyphs, 1, 12)))

    # Negative: same, but the empty glyph's explicit bbox is nonzero.
    glyphs = [
        {"kind": "empty", "bbox": struct.pack(">hhhh", 1, 2, 3, 4)},
        {"kind": "simple", "points": [(0, 0, True), (10, 5, True)],
         "contours": [1], "bbox": (3, 4, 10, 5), "instructions": b""},
    ]
    write("woff2-synth-empty-bbox-nonzero.woff2",
          _build_woff2(_synth_base_tables(glyphs, 1, 12)))

    # Positive: same glyph order with the empty glyph's bbox bit
    # cleared, so the simple glyph's explicit bbox is the one
    # consumed. (3, 4, 10, 5) differs from the points' own bounds
    # (0, 0, 10, 5), so it pins the simple-explicit-bbox branch
    # against the recompute-from-points path, and the empty
    # glyph first pins that it consumes no bbox bytes.
    glyphs = [
        {"kind": "empty"},
        {"kind": "simple", "points": [(0, 0, True), (10, 5, True)],
         "contours": [1], "bbox": (3, 4, 10, 5), "instructions": b""},
    ]
    write("woff2-synth-simple-explicit-bbox.woff2",
          _build_woff2(_synth_base_tables(glyphs, 1, 12)))

    # Negative: transformed hmtx without hhea.
    tabs = [
        _raw(b"head", _head(0)),
        _raw(b"maxp", _maxp(3)),
        (b"glyf", len(_SYNTH_RAW_GLYF), _SYNTH_RAW_GLYF, 3),
        (b"loca", len(_SYNTH_RAW_LOCA), _SYNTH_RAW_LOCA, 3),
        (b"hmtx", 10, bytes([0x03]) + struct.pack(">HH", 500, 600), 1),
    ]
    write("woff2-synth-hmtx-no-hhea.woff2", _build_woff2(tabs))

    # Negative: transformed loca without any glyf.
    tabs = [
        _raw(b"head", _head(1)),
        _raw(b"maxp", _maxp(2)),
        (b"loca", 12, b"", 0),
    ]
    write("woff2-synth-loca-no-glyf.woff2", _build_woff2(tabs))

    # Negative: transformed glyf with an untransformed loca.
    glyphs = [{"kind": "empty"}, _SYNTH_SIMPLE_2PT]
    tabs = [
        (b"glyf", 100, _build_glyf(glyphs, 1), 0),
        (b"loca", 12, struct.pack(">III", 0, 0, 20), 3),
        _raw(b"head", _head(1)),
        _raw(b"maxp", _maxp(2)),
    ]
    write("woff2-synth-glyf-untrans-loca.woff2", _build_woff2(tabs))

    # Negative: transformed glyf with no loca at all.
    tabs = [
        (b"glyf", 100, _build_glyf(glyphs, 1), 0),
        _raw(b"head", _head(1)),
        _raw(b"maxp", _maxp(2)),
    ]
    write("woff2-synth-glyf-no-loca.woff2", _build_woff2(tabs))

    # Negative: maxp count mismatch (glyf has 2 glyphs, maxp says 3).
    tabs = _synth_base_tables(glyphs, 1, 12)
    tabs = [(t if t[0] != b"maxp" else _raw(b"maxp", _maxp(3))) for t in tabs]
    write("woff2-synth-maxp-mismatch.woff2", _build_woff2(tabs))

    # Negative: zero glyphs in a transformed glyf.
    tabs = [
        (b"glyf", 100, _build_glyf([], 1), 0),
        (b"loca", 4, b"", 0),
        _raw(b"head", _head(1)),
        _raw(b"maxp", _maxp(0)),
    ]
    write("woff2-synth-zero-glyph.woff2", _build_woff2(tabs))

    # Negative: transformed hmtx, hhea numberOfHMetrics = 0.
    tabs = [
        _raw(b"head", _head(0)),
        _raw(b"maxp", _maxp(3)),
        _raw(b"hhea", _hhea(0)),
        (b"glyf", len(_SYNTH_RAW_GLYF), _SYNTH_RAW_GLYF, 3),
        (b"loca", len(_SYNTH_RAW_LOCA), _SYNTH_RAW_LOCA, 3),
        (b"hmtx", 10, bytes([0x03]) + struct.pack(">HH", 500, 600), 1),
    ]
    write("woff2-synth-zero-metrics.woff2", _build_woff2(tabs))

    # Positive: transformed hmtx, metrics == glyph count (boundary).
    # xMins are [0, 10, 30], so the rebuilt hmtx is (500, 0),
    # (600, 10), (700, 30) with no tail.
    tabs = [
        _raw(b"head", _head(0)),
        _raw(b"maxp", _maxp(3)),
        _raw(b"hhea", _hhea(3)),
        (b"glyf", len(_SYNTH_RAW_GLYF), _SYNTH_RAW_GLYF, 3),
        (b"loca", len(_SYNTH_RAW_LOCA), _SYNTH_RAW_LOCA, 3),
        (b"hmtx", 12, bytes([0x03]) + struct.pack(">HHH", 500, 600, 700), 1),
    ]
    write("woff2-synth-hmtx-metrics-eq-count.woff2", _build_woff2(tabs))

    # Negative: transformed hmtx, metrics > glyph count. origLength
    # 12 is exactly what the clamped (buggy) rebuild produces, so
    # the old code accepts and the fixed code rejects.
    tabs = [
        _raw(b"head", _head(0)),
        _raw(b"maxp", _maxp(3)),
        _raw(b"hhea", _hhea(4)),
        (b"glyf", len(_SYNTH_RAW_GLYF), _SYNTH_RAW_GLYF, 3),
        (b"loca", len(_SYNTH_RAW_LOCA), _SYNTH_RAW_LOCA, 3),
        (b"hmtx", 12, bytes([0x03]) + struct.pack(">HHH", 500, 600, 700), 1),
    ]
    write("woff2-synth-hmtx-metrics-over-count.woff2", _build_woff2(tabs))

    # Positive: arbitrary tag (index 63).
    tabs = _synth_base_tables(_SYNTH_BASE_GLYPHS, 1, 16) + [
        (b"Xyz1", 4, b"ABCD", 0),
    ]
    write("woff2-synth-arbitrary.woff2", _build_woff2(tabs))

    # Positive (after the zero-length-table fix): zero-length table.
    tabs = _synth_base_tables(_SYNTH_BASE_GLYPHS, 1, 16) + [
        (b"post", 0, b"", 0),
    ]
    write("woff2-synth-zero-table.woff2", _build_woff2(tabs))

    # Negative: transformed hmtx + untransformed glyf/loca + no maxp
    # + loca origLength 1 (loca.len() < one loca entry).
    tabs = [
        _raw(b"head", _head(0)),
        _raw(b"hhea", _hhea(1)),
        (b"glyf", len(_SYNTH_RAW_GLYF), _SYNTH_RAW_GLYF, 3),
        (b"loca", 1, b"\x00", 3),
        (b"hmtx", 4, bytes([0x03]) + struct.pack(">H", 500), 1),
    ]
    write("woff2-synth-hmtx-short-loca.woff2", _build_woff2(tabs))

    # Positive base for the meta/private and trailing-byte Rust tests
    # (which mutate its header) and for the truncation variants.
    write("woff2-synth-base.woff2",
          _build_woff2(_synth_base_tables(_SYNTH_BASE_GLYPHS, 1, 16)))

    def glyf_with_trunc(idx):
        full = _build_glyf(_SYNTH_BASE_GLYPHS, 1)
        sizes = [struct.unpack(">I", full[8 + i * 4:12 + i * 4])[0]
                 for i in range(7)]
        pos = 36
        parts = []
        for i, sz in enumerate(sizes):
            part = full[pos:pos + sz]
            pos += sz
            if i == idx:
                part = part[:-1]
            parts.append(part)
        out = bytearray(full[:36])
        for i, p in enumerate(parts):
            struct.pack_into(">I", out, 8 + i * 4, len(p))
        for p in parts:
            out += p
        return bytes(out)

    for i, name in enumerate(["ncontour", "npoints", "flag", "glyph",
                              "composite", "bbox", "instruction"]):
        tabs = [
            (b"glyf", 100, glyf_with_trunc(i), 0),
            (b"loca", 16, b"", 0),
            _raw(b"head", _head(1)),
            _raw(b"maxp", _maxp(3)),
        ]
        write(f"woff2-synth-trunc-{name}.woff2", _build_woff2(tabs))


def _assert_hmtx_transformed(path: str) -> None:
    """Fail if the WOFF2 at `path` does not carry a transformed `hmtx`."""
    data = open(path, "rb").read()
    known = _WOFF2_KNOWN

    def b128(at):
        v = 0
        while True:
            b = data[at]
            at += 1
            v = (v << 7) | (b & 0x7F)
            if not b & 0x80:
                return v, at

    num = struct.unpack(">H", data[12:14])[0]
    at = 48
    for _ in range(num):
        flags = data[at]
        at += 1
        idx = flags & 0x3F
        ver = flags >> 6
        tag = data[at:at + 4] if idx == 63 else known[idx]
        if idx == 63:
            at += 4
        _orig, at = b128(at)
        transformed = (tag in (b"glyf", b"loca") and ver != 3) or \
            (tag == b"hmtx" and ver != 0)
        if transformed:
            _tlen, at = b128(at)
        if tag == b"hmtx":
            if ver != 1:
                raise SystemExit(f"{path}: hmtx is not transformed (version {ver})")
            return
    raise SystemExit(f"{path}: no hmtx table found")


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "synthetic":
        synthetic(sys.argv[2])
        return
    src = sys.argv[1]
    ttf = open(src, "rb").read()
    open(sys.argv[2], "wb").write(build(ttf, stored=False))
    open(sys.argv[3], "wb").write(build(ttf, stored=True))
    if len(sys.argv) > 4:
        # WOFF2 via fontTools (needs the fonttools + brotli packages;
        # fixture generation only, never a build dependency).
        from fontTools.ttLib import TTFont

        font = TTFont(src)
        font.flavor = "woff2"
        font.save(sys.argv[4])
    if len(sys.argv) > 5:
        # WOFF2 with the hmtx transform enabled as well.
        from fontTools.ttLib.woff2 import compress

        compress(src, sys.argv[5], transform_tables={"glyf", "loca", "hmtx"})
        _assert_hmtx_transformed(sys.argv[5])


if __name__ == "__main__":
    main()
