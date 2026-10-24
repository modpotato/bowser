#!/usr/bin/env python3
"""Generate bowser_net TLS/X.509 test fixtures.

All fixtures are DER (or hex text) generated once with the openssl CLI and
committed. Nothing here runs at test time; tests only read the fixtures.

Regenerate:  python3 gen-fixtures.py   (from this directory)

Requires: openssl CLI, python3 (stdlib only).

Layout (all dates fixed, serials fixed; keys are random per generation):

  Test PKI (all RSA-2048 unless noted):
    test-ca.der            self-signed root CA,        2020-01-01 .. 2040-01-01
    test-inter.der         intermediate CA (pathlen:0), signed by test-ca
    leaf-rsa.der           server cert, SAN DNS:bowser.test, signed by test-inter
    leaf-rsa-priv.txt      hex(n) / hex(d) of leaf-rsa's key (lets tests RSA-sign
                           CertificateVerify messages without a signer)
    leaf-rsa-expired.der   SAN bowser.test, 2019-01-01 .. 2020-01-01 (expired)
    leaf-rsa-wronghost.der SAN DNS:other.test (hostname mismatch)
    leaf-rsa-wildcard.der  SAN DNS:*.bowser.test
    leaf-rsa-ip.der        SAN IP:127.0.0.1, IP:::1
    leaf-rsa-cnonly.der    no SAN, CN=bowser.test (CN fallback)
    leaf-selfsigned.der    self-signed server cert (untrusted)
    inter-noca.der         signed by test-ca but WITHOUT CA basicConstraints
    leaf-under-noca.der    server cert signed by inter-noca (must be rejected:
                           issuer is not a CA)
    evil-ca.der            unrelated self-signed root (untrusted)
    evil-leaf.der          server cert signed by evil-ca (untrusted chain)
  ECDSA PKI (P-256):
    ec-ca.der              self-signed ECDSA root CA
    ec-leaf.der            server cert, SAN DNS:bowser.test, signed by ec-ca

  Signature KATs (openssl as oracle):
    rsa_kat.txt            n/e/msg/sig for sha256, sha384, sha512
    ecdsa_kat.txt          qx/qy/msg/r/s for p256/sha256, p384/sha384
"""

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OPENSSL = "openssl"


def run(*args, **kw):
    r = subprocess.run([OPENSSL, *args], capture_output=True, **kw)
    if r.returncode != 0:
        print("FAILED:", " ".join(args), file=sys.stderr)
        print(r.stderr.decode(), file=sys.stderr)
        sys.exit(1)
    return r


def w(path, data):
    with open(os.path.join(HERE, path), "wb") as f:
        f.write(data)


# --- minimal DER reader (enough for INTEGER / SEQUENCE / BIT STRING) ---

class Der:
    def __init__(self, b):
        self.b = b
        self.pos = 0

    def _tl(self):
        tag = self.b[self.pos]
        self.pos += 1
        ln = self.b[self.pos]
        self.pos += 1
        if ln & 0x80:
            n = ln & 0x7F
            ln = int.from_bytes(self.b[self.pos:self.pos + n], "big")
            self.pos += n
        return tag, ln

    def read(self):
        tag, ln = self._tl()
        v = self.b[self.pos:self.pos + ln]
        self.pos += ln
        return tag, v

    def seq(self):
        tag, v = self.read()
        assert tag == 0x30, f"expected SEQUENCE, got {tag:#x}"
        return Der(v)

    def integer(self):
        tag, v = self.read()
        assert tag == 0x02, f"expected INTEGER, got {tag:#x}"
        return v.lstrip(b"\x00")


def pem_to_der(pem_bytes):
    m = re.search(rb"-----BEGIN [^-]+-----\r?\n([A-Za-z0-9+/=\r\n]+)-----END", pem_bytes)
    assert m, "no PEM block found"
    import base64
    return base64.b64decode(re.sub(rb"\s", b"", m.group(1)))


def extfile(path, text):
    with open(os.path.join(HERE, path), "w") as f:
        f.write(text)


# --- DER validity rewriter ---
# openssl 3.0's `x509 -req` cannot set explicit notBefore/notAfter, so certs
# are generated with -days and then rewritten here to fixed dates. The cert
# is parsed into a TLV tree, the Validity SEQUENCE is replaced, and the
# tree is re-encoded (lossless for DER).

def parse_node(b, pos):
    tag = b[pos]; pos += 1
    ln = b[pos]; pos += 1
    if ln & 0x80:
        n = ln & 0x7F
        ln = int.from_bytes(b[pos:pos + n], "big"); pos += n
    hdr = pos
    val = b[pos:pos + ln]; pos += ln
    constructed = (tag & 0x20) != 0
    if constructed:
        children = []
        p = 0
        while p < len(val):
            child, p2 = parse_node(val, p)
            children.append(child); p = p2
        return (tag, children), pos
    return (tag, val), pos


def encode_len(n):
    if n < 128:
        return bytes([n])
    lb = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(lb)]) + lb


def encode_node(node):
    tag, val = node
    if isinstance(val, list):
        body = b"".join(encode_node(c) for c in val)
    else:
        body = val
    return bytes([tag]) + encode_len(len(body)) + body


def is_time(node):
    return node[0] in (0x17, 0x18)  # UTCTime, GeneralizedTime


def time_node(s):
    # s: "YYMMDDHHMMSSZ" (UTCTime) or "YYYYMMDDHHMMSSZ" (GeneralizedTime)
    tag = 0x17 if len(s) == 13 else 0x18
    assert len(s) in (13, 15) and s.endswith("Z")
    return (tag, s.encode())


def set_validity(der, nb, na):
    (tag, children), _ = parse_node(der, 0)
    assert tag == 0x30
    tbs_tag, tbs_children = children[0]
    assert tbs_tag == 0x30
    new_tbs = []
    replaced = False
    for child in tbs_children:
        ctag, cval = child
        if (ctag == 0x30 and isinstance(cval, list) and len(cval) == 2
                and all(is_time(c) for c in cval)):
            child = (0x30, [time_node(nb), time_node(na)])
            replaced = True
        new_tbs.append(child)
    assert replaced, "validity not found"
    return encode_node((tag, [(tbs_tag, new_tbs)] + children[1:]))


VALID_NB, VALID_NA = "200101000000Z", "400101000000Z"   # 2020-01-01 .. 2040-01-01
EXP_NB, EXP_NA = "190101000000Z", "200101000000Z"       # 2019-01-01 .. 2020-01-01


def decode_oid(b):
    nums = [b[0] // 40, b[0] % 40]
    i = 1
    while i < len(b):
        v = 0
        while True:
            v = (v << 7) | (b[i] & 0x7F)
            i += 1
            if not (b[i - 1] & 0x80):
                break
        nums.append(v)
    return ".".join(str(n) for n in nums)


SIG_OID_TO_DGST = {
    "1.2.840.113549.1.1.11": "sha256",  # sha256WithRSAEncryption
    "1.2.840.10045.4.3.2": "sha256",    # ecdsa-with-SHA256
}


def resign(der_path, signer_key):
    """Re-sign a cert's TBSCertificate (after validity rewriting)."""
    p = os.path.join(HERE, der_path)
    with open(p, "rb") as f:
        der = f.read()
    (tag, children), _ = parse_node(der, 0)
    assert tag == 0x30
    tbs_der = encode_node(children[0])
    alg_seq = children[1]
    assert alg_seq[0] == 0x30
    oid = decode_oid(alg_seq[1][0][1])
    dgst = SIG_OID_TO_DGST[oid]
    sig = run("dgst", f"-{dgst}", "-sign", signer_key, cwd=HERE, input=tbs_der).stdout
    new_cert = encode_node((0x30, [children[0], children[1], (0x03, b"\x00" + sig)]))
    with open(p, "wb") as f:
        f.write(new_cert)


def final(der_path, signer_key, nb=VALID_NB, na=VALID_NA):
    p = os.path.join(HERE, der_path)
    with open(p, "rb") as f:
        der = f.read()
    with open(p, "wb") as f:
        f.write(set_validity(der, nb, na))
    resign(p, signer_key)


# --- PKI ---

CA_EXT = """\
basicConstraints=critical,CA:TRUE
keyUsage=critical,keyCertSign,cRLSign
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid:always,issuer
"""
INTER_EXT = """\
basicConstraints=critical,CA:TRUE,pathlen:0
keyUsage=critical,keyCertSign,cRLSign
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid:always,issuer
"""


def leaf_ext(san_extra=""):
    return f"""\
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=@alt
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid:always,issuer
[alt]
{san_extra}
"""


def selfsigned(key, subj, out_der, serial, ext):
    """Self-signed cert via `x509 -req -signkey`; validity fixed by final()."""
    gen_rsa_key(key)
    gen_csr(key, subj, "_self.csr")
    run("x509", "-req", "-in", "_self.csr", "-signkey", key, "-days", "3650",
        "-set_serial", serial, "-extfile", ext,
        "-out", out_der, "-outform", "DER", cwd=HERE)
    os.remove(os.path.join(HERE, "_self.csr"))
    final(out_der, key)


def sign_csr(csr, ca_der, ca_key, out_der, serial, ext, nb=VALID_NB, na=VALID_NA):
    run("x509", "-req", "-in", csr, "-CA", ca_der, "-CAform", "DER",
        "-CAkey", ca_key, "-CAcreateserial", "-days", "3650",
        "-set_serial", serial, "-extfile", ext,
        "-out", out_der, "-outform", "DER", cwd=HERE)
    final(out_der, ca_key, nb, na)


def gen_rsa_key(name):
    run("genrsa", "-out", name, "2048", cwd=HERE)


def gen_csr(key, subj, csr):
    run("req", "-new", "-key", key, "-subj", subj, "-out", csr, cwd=HERE)


os.makedirs(HERE, exist_ok=True)

# Root CA (self-signed).
extfile("ca.ext", CA_EXT)
selfsigned("test-ca.key", "/CN=Bowser Test Root CA", "test-ca.der", "0x1000", "ca.ext")

# Intermediate CA.
gen_rsa_key("test-inter.key")
gen_csr("test-inter.key", "/CN=Bowser Test Intermediate CA", "test-inter.csr")
extfile("inter.ext", INTER_EXT)
sign_csr("test-inter.csr", "test-ca.der", "test-ca.key", "test-inter.der",
         "0x1001", "inter.ext")

# Valid leaf (RSA), SAN bowser.test. Key kept for CertificateVerify signing.
extfile("leaf.ext", leaf_ext("DNS.1=bowser.test"))
gen_rsa_key("leaf-rsa.key")
gen_csr("leaf-rsa.key", "/CN=bowser.test", "leaf-rsa.csr")
sign_csr("leaf-rsa.csr", "test-inter.der", "test-inter.key", "leaf-rsa.der",
         "0x2001", "leaf.ext")

# Dump leaf private key material (n, d as hex) for in-test RSA signing.
# Handles PKCS#8 ("PRIVATE KEY": SEQ{ver, algId, OCTET STRING(PKCS#1)})
# and raw PKCS#1 ("RSA PRIVATE KEY").
raw_key = open(os.path.join(HERE, "leaf-rsa.key"), "rb").read()
seq = Der(pem_to_der(raw_key)).seq()
seq.integer()  # version
second_tag, second_val = seq.read()
if second_tag == 0x02:  # PKCS#1: version, n, e, d, ...
    n, e, d = second_val.lstrip(b"\x00"), seq.integer(), seq.integer()
else:  # PKCS#8: version, algId, OCTET STRING(PKCS#1)
    assert second_tag == 0x30
    tag, octet = seq.read()
    assert tag == 0x04
    inner = Der(octet).seq()
    inner.integer()  # version
    n = inner.integer()
    inner.integer()  # e
    d = inner.integer()
with open(os.path.join(HERE, "leaf-rsa-priv.txt"), "w") as f:
    f.write("n: " + n.hex() + "\n")
    f.write("d: " + d.hex() + "\n")

# Expired leaf.
extfile("leaf-exp.ext", leaf_ext("DNS.1=bowser.test"))
gen_rsa_key("tmp.key")
gen_csr("tmp.key", "/CN=bowser.test", "tmp.csr")
sign_csr("tmp.csr", "test-inter.der", "test-inter.key", "leaf-rsa-expired.der",
         "0x2002", "leaf-exp.ext", nb=EXP_NB, na=EXP_NA)

# Wrong-host leaf.
extfile("leaf-wh.ext", leaf_ext("DNS.1=other.test"))
gen_rsa_key("tmp.key")
gen_csr("tmp.key", "/CN=other.test", "tmp.csr")
sign_csr("tmp.csr", "test-inter.der", "test-inter.key", "leaf-rsa-wronghost.der",
         "0x2003", "leaf-wh.ext")

# Wildcard leaf.
extfile("leaf-wc.ext", leaf_ext("DNS.1=*.bowser.test"))
gen_rsa_key("tmp.key")
gen_csr("tmp.key", "/CN=*.bowser.test", "tmp.csr")
sign_csr("tmp.csr", "test-inter.der", "test-inter.key", "leaf-rsa-wildcard.der",
         "0x2004", "leaf-wc.ext")

# IP SAN leaf.
extfile("leaf-ip.ext", leaf_ext("IP.1=127.0.0.1\nIP.2=::1"))
gen_rsa_key("tmp.key")
gen_csr("tmp.key", "/CN=127.0.0.1", "tmp.csr")
sign_csr("tmp.csr", "test-inter.der", "test-inter.key", "leaf-rsa-ip.der",
         "0x2005", "leaf-ip.ext")

# CN-only leaf (no SAN).
extfile("leaf-cn.ext", """\
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid:always,issuer
""")
gen_rsa_key("tmp.key")
gen_csr("tmp.key", "/CN=bowser.test", "tmp.csr")
sign_csr("tmp.csr", "test-inter.der", "test-inter.key", "leaf-rsa-cnonly.der",
         "0x2006", "leaf-cn.ext")

# Self-signed leaf (untrusted).
extfile("self.ext", leaf_ext("DNS.1=bowser.test"))
selfsigned("tmp.key", "/CN=selfsigned.test", "leaf-selfsigned.der",
           "0x3001", "self.ext")

# Intermediate WITHOUT CA basicConstraints + leaf under it.
extfile("noca.ext", """\
keyUsage=critical,digitalSignature
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid:always,issuer
""")
gen_rsa_key("inter-noca.key")
gen_csr("inter-noca.key", "/CN=Not A CA", "tmp.csr")
sign_csr("tmp.csr", "test-ca.der", "test-ca.key", "inter-noca.der",
         "0x1002", "noca.ext")
extfile("leaf-noca.ext", leaf_ext("DNS.1=bowser.test"))
gen_rsa_key("tmp.key")
gen_csr("tmp.key", "/CN=bowser.test", "tmp.csr")
sign_csr("tmp.csr", "inter-noca.der", "inter-noca.key", "leaf-under-noca.der",
         "0x2007", "leaf-noca.ext")

# Evil PKI (untrusted root).
extfile("evilca.ext", CA_EXT)
selfsigned("evil-ca.key", "/CN=Evil Root CA", "evil-ca.der",
           "0x4000", "evilca.ext")
extfile("evil.ext", leaf_ext("DNS.1=bowser.test"))
gen_rsa_key("tmp.key")
gen_csr("tmp.key", "/CN=bowser.test", "tmp.csr")
sign_csr("tmp.csr", "evil-ca.der", "evil-ca.key", "evil-leaf.der",
         "0x4001", "evil.ext")

# ECDSA PKI (P-256).
run("ecparam", "-genkey", "-name", "prime256v1", "-noout", "-out", "ec-ca.key", cwd=HERE)
extfile("ecca.ext", CA_EXT)
run("req", "-new", "-key", "ec-ca.key", "-subj", "/CN=Bowser Test EC Root",
    "-out", "_ecc.csr", cwd=HERE)
run("x509", "-req", "-in", "_ecc.csr", "-signkey", "ec-ca.key",
    "-days", "3650", "-set_serial", "0x5000",
    "-extfile", "ecca.ext", "-out", "ec-ca.der", "-outform", "DER", cwd=HERE)
os.remove(os.path.join(HERE, "_ecc.csr"))
final("ec-ca.der", "ec-ca.key")
run("ecparam", "-genkey", "-name", "prime256v1", "-noout", "-out", "tmp-ec.key", cwd=HERE)
run("req", "-new", "-key", "tmp-ec.key", "-subj", "/CN=bowser.test",
    "-out", "tmp-ec.csr", cwd=HERE)
extfile("ecleaf.ext", leaf_ext("DNS.1=bowser.test"))
run("x509", "-req", "-in", "tmp-ec.csr", "-CA", "ec-ca.der", "-CAform", "DER",
    "-CAkey", "ec-ca.key", "-CAcreateserial",
    "-days", "3650", "-set_serial", "0x5001",
    "-extfile", "ecleaf.ext", "-out", "ec-leaf.der", "-outform", "DER", cwd=HERE)
final("ec-leaf.der", "ec-ca.key")

# --- RSA KATs (openssl signs; our verifier is the oracle's student) ---
with open(os.path.join(HERE, "rsa_kat.txt"), "w") as kat:
    kat.write("# openssl-generated RSA PKCS#1 v1.5 KATs. First vector's signature\n")
    kat.write("# has a leading zero byte (needed by short_signature_left_padded).\n")
    gen_rsa_key("kat-rsa.key")
    pub_der = run("rsa", "-in", "kat-rsa.key", "-pubout", "-outform", "DER", cwd=HERE).stdout
    spki = Der(pub_der).seq()
    spki.seq()  # algorithm identifier
    tag, bitstr = spki.read()
    assert tag == 0x03 and bitstr[0] == 0
    inner = Der(bitstr[1:]).seq()
    kat_n, kat_e = inner.integer(), inner.integer()
    for dgst, hname in [("sha256", "sha256"), ("sha384", "sha384"), ("sha512", "sha512")]:
        counter = 0
        while True:
            msg = f"bowser rsa kat {hname} #{counter}".encode()
            sig = run("dgst", f"-{dgst}", "-sign", "kat-rsa.key", cwd=HERE, input=msg).stdout
            counter += 1
            if hname == "sha256" and sig[0] != 0:
                continue  # first vector needs a leading zero byte
            break
        kat.write(f"hash: {hname}\n")
        kat.write(f"n: {kat_n.hex()}\n")
        kat.write(f"e: {kat_e.hex()}\n")
        kat.write(f"msg: {msg.hex()}\n")
        kat.write(f"sig: {sig.hex()}\n\n")

# --- ECDSA KATs ---
with open(os.path.join(HERE, "ecdsa_kat.txt"), "w") as kat:
    kat.write("# openssl-generated ECDSA KATs (DER signatures parsed to r/s).\n")
    for curve, cname, dgst, hname in [
        ("prime256v1", "p256", "sha256", "sha256"),
        ("secp384r1", "p384", "sha384", "sha384"),
    ]:
        run("ecparam", "-genkey", "-name", curve, "-noout", "-out", "kat-ec.key", cwd=HERE)
        spki = Der(pem_to_der(run("ec", "-in", "kat-ec.key", "-pubout", "-outform", "PEM",
                                      cwd=HERE).stdout)).seq()
        spki.seq()  # algorithm
        tag, bitstr = spki.read()
        assert tag == 0x03 and bitstr[0] == 0 and bitstr[1] == 0x04
        flen = (len(bitstr) - 2) // 2
        qx, qy = bitstr[2:2 + flen], bitstr[2 + flen:]
        msg = f"bowser ecdsa kat {cname}".encode()
        sig_der = run("dgst", f"-{dgst}", "-sign", "kat-ec.key", cwd=HERE, input=msg).stdout
        rs = Der(sig_der).seq()
        r, s = rs.integer(), rs.integer()
        kat.write(f"hash: {hname}\n")
        kat.write(f"curve: {cname}\n")
        kat.write(f"qx: {qx.hex()}\n")
        kat.write(f"qy: {qy.hex()}\n")
        kat.write(f"msg: {msg.hex()}\n")
        kat.write(f"r: {r.hex()}\n")
        kat.write(f"s: {s.hex()}\n\n")

# --- cleanup: keep only fixtures ---
for f in os.listdir(HERE):
    if f.endswith((".key", ".csr", ".ext", ".srl")) or f in ("tmp.key",):
        os.remove(os.path.join(HERE, f))

print("fixtures written to", HERE)
for f in sorted(os.listdir(HERE)):
    print(" ", f)
