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
"""
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


def main() -> None:
    src = sys.argv[1]
    ttf = open(src, "rb").read()
    open(sys.argv[2], "wb").write(build(ttf, stored=False))
    open(sys.argv[3], "wb").write(build(ttf, stored=True))


if __name__ == "__main__":
    main()
