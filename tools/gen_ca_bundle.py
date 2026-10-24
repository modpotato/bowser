#!/usr/bin/env python3
"""Build tests/fixtures/ca_bundle.bin from the Mozilla CA bundle.

The bundle is the curl project's PEM conversion of Mozilla's CA
Certificate store (https://curl.se/ca/cacert.pem, "Certificate data
from Mozilla"). It is vendored as *data*: this script downloads the PEM,
DER-decodes each certificate, and writes a flat binary file of
[u32 big-endian DER length][DER bytes] records, which
crates/bowser_net/src/crypto/x509.rs loads with include_bytes!().

Usage:
    python3 tools/gen_ca_bundle.py [output]

    output defaults to crates/bowser_net/tests/fixtures/ca_bundle.bin.
    Pass --check to only report what would change.

Mozilla's CA data is offered under the Mozilla Public License 2.0.
Attribution (source URL + retrieval date) is recorded in x509.rs's
module docs; update the date there when regenerating.
"""

import base64
import os
import re
import struct
import sys
import urllib.request

SOURCE_URL = "https://curl.se/ca/cacert.pem"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(
    HERE, "..", "crates", "bowser_net", "tests", "fixtures", "ca_bundle.bin"
)


def main():
    check = "--check" in sys.argv
    out = next((a for a in sys.argv[1:] if not a.startswith("--")), DEFAULT_OUT)

    print("downloading", SOURCE_URL, file=sys.stderr)
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "bowser/gen_ca_bundle.py"})
    pem = urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "replace")

    # Header comment carries the Mozilla data date; surface it.
    m = re.search(r"Certificate data from Mozilla as of: ([^\n]+)", pem)
    print("mozilla data date:", m.group(1).strip() if m else "unknown", file=sys.stderr)

    blocks = re.findall(
        r"-----BEGIN CERTIFICATE-----\s*([A-Za-z0-9+/=\s]+)-----END CERTIFICATE-----", pem
    )
    ders = []
    for b in blocks:
        der = base64.b64decode(re.sub(r"\s", "", b))
        # Sanity: must start with a SEQUENCE tag.
        assert der[0] == 0x30, "non-DER certificate in bundle"
        ders.append(der)
    print(f"{len(ders)} certificates", file=sys.stderr)

    blob = b"".join(struct.pack(">I", len(d)) + d for d in ders)
    if check:
        with open(out, "rb") as f:
            cur = f.read()
        print("identical" if cur == blob else "DIFFERS", file=sys.stderr)
        sys.exit(0 if cur == blob else 1)
    with open(out, "wb") as f:
        f.write(blob)
    print(f"wrote {out} ({len(blob)} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
