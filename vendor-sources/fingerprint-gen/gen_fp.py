#!/usr/bin/env python3
"""Transcribe wreq-util fingerprint profile DATA into compact Rust tables.

Reads:
  ~/workspace/bowser/vendor-sources/wreq-util/src/emulate/profile/{chrome,firefox,safari,opera,okhttp}.rs
  .../profile/{chrome,firefox,safari,opera}/tls.rs   (string consts)
  ~/workspace/bowser/vendor-sources/wreq-util/src/emulate.rs (public profile names)
Writes:
  data.rs  (generated static tables; included by fingerprint/mod.rs)

Only DATA is transcribed (cipher/curve/sigalg lists, flags, HTTP/2 settings,
header order, UA strings). No wreq code is copied.
"""
import re, os, sys

HOME = os.path.expanduser("~")
SRC = os.path.join(HOME, "workspace/bowser/vendor-sources/wreq-util/src/emulate")
P = os.path.join(SRC, "profile")

# ---------------------------------------------------------------- parsing utils

def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()

def parse_string(s, i):
    """Parse a Rust string literal starting at s[i] ('"' or 'r#"'). Returns (value, end)."""
    if s.startswith('r#"', i):
        j = s.index('"#', i + 3)
        return s[i + 3:j], j + 2
    assert s[i] == '"', s[i:i + 20]
    out = []
    i += 1
    while True:
        c = s[i]
        if c == '\\':
            out.append(s[i + 1]); i += 2
        elif c == '"':
            return "".join(out), i + 1
        else:
            out.append(c); i += 1

def strip_comments(s):
    """Remove // line comments, respecting string literals."""
    out = []
    i, n = 0, len(s)
    while i < n:
        if s[i] == '"' or s.startswith('r#"', i):
            v, j = parse_string(s, i)
            out.append(s[i:j]); i = j; continue
        if s.startswith('//', i):
            j = s.find('\n', i)
            i = len(s) if j == -1 else j
            continue
        out.append(s[i]); i += 1
    return "".join(out)

def split_tokens(tokens):
    """Split a token list on top-level commas; returns list of token lists."""
    parts, depth, cur = [], 0, []
    for t in tokens:
        if t[0] == 'ch':
            if t[1] in '([{': depth += 1
            elif t[1] in ')]}': depth -= 1
            if t[1] == ',' and depth == 0:
                parts.append(cur); cur = []
                continue
        cur.append(t)
    parts.append(cur)
    return parts

def split_top(s):
    """Split on top-level commas (depth 0), respecting strings and nesting.
    Returns list of (text_with_markers, [string values], tokens)."""
    tokens = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == '"' or s.startswith('r#"', i):
            v, i = parse_string(s, i)
            tokens.append(('str', v))
            continue
        tokens.append(('ch', c))
        i += 1
    out = []
    for part in split_tokens(tokens):
        txt = "".join(t[1] if t[0] == 'ch' else '\x00' + t[1] + '\x00' for t in part).strip()
        strs = [t[1] for t in part if t[0] == 'str']
        out.append((txt, strs, part))
    return out

def find_invocations(src, macro):
    """Yield balanced bodies of `macro!( ... )` occurrences."""
    res = []
    for m in re.finditer(re.escape(macro) + r'!\s*\(', src):
        i = m.end() - 1
        depth = 0
        j = i
        instr = False
        while j < len(src):
            if src.startswith('r#"', j):
                _, j = parse_string(src, j); continue
            c = src[j]
            if c == '"':
                _, j = parse_string(src, j); continue
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        res.append(src[i + 1:j])
    return res

def norm(txt):
    return re.sub(r'\s+', '', txt)

# ------------------------------------------------- string-list consts from tls.rs

def extract_join_consts(raw):
    """Map NAME -> joined string for `const NAME: &str = join!(":", ...);`."""
    src = strip_comments(raw)
    consts = {}
    for m in re.finditer(r'(?:pub(?:\([^)]*\))?\s+)?const\s+(\w+)\s*:\s*&str\s*=\s*join!\s*\(', src):
        i = m.end() - 1
        depth = 0; j = i
        while True:
            if src.startswith('r#"', j):
                _, j = parse_string(src, j); continue
            c = src[j]
            if c == '"':
                _, j = parse_string(src, j); continue
            if c == '(': depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0: break
            j += 1
        body = src[i + 1:j]
        args = split_top(body)
        sep = args[0][1][0]
        vals = [a[1][0] for a in args[1:]]
        consts[m.group(1)] = sep.join(vals)
    return consts

def extract_keyshares(raw):
    src = strip_comments(raw)
    ks = {}
    for m in re.finditer(r'pub\s+const\s+(KEY_SHARES_\d+)\s*:[^=]*=\s*&\[([^\]]*)\]', src):
        items = [x.strip().split("::")[-1] for x in m.group(2).split(",") if x.strip()]
        ks[m.group(1)] = items
    return ks

CHROME_STR = extract_join_consts(read(os.path.join(P, "chrome", "tls.rs")))
FF_STR = extract_join_consts(read(os.path.join(P, "firefox", "tls.rs")))
SAFARI_STR = extract_join_consts(read(os.path.join(P, "safari", "tls.rs")))
OPERA_STR = extract_join_consts(read(os.path.join(P, "opera", "tls.rs")))
OKHTTP_STR = extract_join_consts(read(os.path.join(P, "okhttp.rs")))
FF_KS = extract_keyshares(read(os.path.join(P, "firefox", "tls.rs")))
KS_NAME = {"X25519": "X25519", "P256": "P-256", "X25519_MLKEM768": "X25519MLKEM768"}

# ------------------------------------------------- public profile names

def public_names():
    src = read(os.path.join(SRC, "..", "emulate.rs"))
    names = {}
    for m in re.finditer(r'(\w+)\s*=>\s*\("([^"]+)",\s*(\w+)::emulation\)', src):
        names[m.group(3)] = m.group(2)
    return names

NAMES = public_names()

# ------------------------------------------------- parsed mod_generator! invocations

def parse_platforms_toks(toks):
    """Parse `[ (OS, sec_ch_ua, ua), ... ]` / `[ (OS, ua), ... ]` from tokens."""
    # strip outer [ ]
    chars = [(i, x[1]) for i, x in enumerate(toks) if x[0] == 'ch']
    open_i = next(i for i, c in chars if c == '[')
    close_i = max(i for i, c in chars if c == ']')
    inner_toks = toks[open_i + 1:close_i]
    entries = []
    for part in split_tokens(inner_toks):
        pchars = [(i, x[1]) for i, x in enumerate(part) if x[0] == 'ch']
        po = next(i for i, c in pchars if c == '(')
        pc = max(i for i, c in pchars if c == ')')
        args = split_tokens(part[po + 1:pc])
        osname = norm("".join(x[1] for x in args[0] if x[0] == 'ch'))
        strs = [x[1] for a in args[1:] for x in a if x[0] == 'str']
        if len(strs) == 2:
            entries.append((osname, strs[0], strs[1]))
        elif len(strs) == 1:
            entries.append((osname, "", strs[0]))
        else:
            raise ValueError(f"bad platform tuple ({len(strs)} strings)")
    return entries

def single_ua(strs):
    """A lone UA string literal (already extracted): -> [("INFER", "", ua)]."""
    assert len(strs) == 1, strs
    return [("INFER", "", strs[0])]


# ------------------------------------------------- flag tables (transcribed from tls.rs/http2.rs semantics)

def chrome_tls(n, curves, sigalgs):
    d = dict(family="chrome", ciphers=CHROME_STR["CIPHER_LIST"],
             curves=CHROME_STR[curves or "CURVES_1"],
             sigalgs=CHROME_STR[sigalgs or "SIGALGS_LIST"],
             alpn=("h2", "http/1.1"), grease=True, permute=False, ech=False,
             psk=False, psk_skip=False, psk_dhe_ke=True, session_ticket=True,
             ocsp=True, sct=True, alps="h2", alps_new=False, cert_comp="brotli",
             sigalg_grease=False, trust_anchors=False, min_tls="Tls12",
             max_tls="Tls13", key_shares=(), delegated="", rsl=0, ext_order=())
    if n == 2: d["ech"] = True
    elif n == 3: d["permute"] = True
    elif n == 4: d.update(permute=True, ech=True)
    elif n == 5: d.update(permute=True, ech=True, psk=True)
    elif n == 6: d.update(permute=True, ech=True, psk=True)
    elif n == 7: d.update(permute=True, ech=True, psk=True, alps_new=True)
    elif n == 8: d.update(permute=True, ech=True, psk=True, alps_new=True)
    elif n == 9: d.update(permute=True, ech=True, psk=True, alps_new=True,
                          sigalgs=CHROME_STR["NEW_SIGALGS_LIST"],
                          sigalg_grease=True, trust_anchors=True)
    elif n != 1: raise ValueError(n)
    return d

FF_EXT_ORDER = (0, 23, 65281, 10, 11, 35, 16, 5, 34, 18, 51, 43, 13, 45, 28, 27, 65037)
FF_DC = FF_STR["DELEGATED_CREDENTIALS"]

def firefox_tls(n, cipher, curves, keyshares):
    d = dict(family="firefox", ciphers=FF_STR[cipher], curves=FF_STR[curves],
             sigalgs=FF_STR["SIGALGS_LIST"], alpn=("h2", "http/1.1"),
             grease=False, permute=False, ech=False, psk=False, psk_skip=False,
             psk_dhe_ke=True, session_ticket=True, ocsp=True, sct=False,
             alps="", alps_new=False, cert_comp="", sigalg_grease=False,
             trust_anchors=False, min_tls="Tls12", max_tls="Tls13",
             key_shares=(), delegated=FF_DC, rsl=0x4001, ext_order=FF_EXT_ORDER)
    ks = tuple(KS_NAME[k] for k in FF_KS[keyshares]) if keyshares else ()
    if n == 1: d.update(ech=True, psk=True, psk_skip=True, key_shares=ks,
                        cert_comp="zlib,brotli,zstd")
    elif n == 2: pass
    elif n == 3: d.update(session_ticket=False, ech=True, psk_dhe_ke=False, key_shares=ks)
    elif n == 4: d.update(ech=True, sct=True, psk=True, psk_skip=True,
                          key_shares=ks, cert_comp="zlib,brotli,zstd")
    elif n == 5: d.update(ech=True, psk=True, psk_skip=True, key_shares=ks,
                          cert_comp="zlib,brotli,zstd")
    elif n == 6: d.update(ech=True, sct=True, session_ticket=False,
                          psk_dhe_ke=False, key_shares=ks,
                          cert_comp="zlib,brotli,zstd")
    else: raise ValueError(n)
    return d

def safari_tls(n, cipher, sigalgs, curves):
    d = dict(family="safari", ciphers=SAFARI_STR[cipher],
             curves=SAFARI_STR[curves or "CURVES_1"],
             sigalgs=SAFARI_STR[sigalgs or "SIGALGS_LIST_1"], alpn=(),
             grease=True, permute=False, ech=False, psk=False, psk_skip=False,
             psk_dhe_ke=True, session_ticket=False, ocsp=True, sct=True,
             alps="", alps_new=False, cert_comp="zlib", sigalg_grease=False,
             trust_anchors=False, min_tls="Tls10", max_tls="Tls13",
             key_shares=(), delegated="", rsl=0, ext_order=())
    if n == 3:
        d.update(min_tls="Tls12", preserve_tls13=True)
    elif n not in (1, 2):
        raise ValueError(n)
    return d

def opera_tls(curves):
    return dict(family="opera", ciphers=OPERA_STR["CIPHER_LIST"],
                curves=OPERA_STR[curves or "CURVES"], sigalgs=OPERA_STR["SIGALGS_LIST"],
                alpn=("h2", "http/1.1"), grease=True, permute=True, ech=True,
                psk=True, psk_skip=False, psk_dhe_ke=True, session_ticket=True,
                ocsp=True, sct=True, alps="h2", alps_new=False,
                cert_comp="brotli", sigalg_grease=False, trust_anchors=False,
                min_tls="Tls12", max_tls="Tls13", key_shares=(),
                delegated="", rsl=0, ext_order=())

def okhttp_tls(expr):
    _, txt, strs = expr
    if txt.startswith("join!"):
        # strs[0] is the separator, strs[1:] are cipher names
        cipher = strs[0].join(strs[1:])
    else:
        cipher = OKHTTP_STR[txt]
    return dict(family="okhttp", ciphers=cipher, curves=OKHTTP_STR["CURVES"],
                sigalgs=OKHTTP_STR["SIGALGS_LIST"], alpn=(), grease=False,
                permute=False, ech=False, psk=False, psk_skip=False,
                psk_dhe_ke=True, session_ticket=True, ocsp=True, sct=False,
                alps="", alps_new=False, cert_comp="", sigalg_grease=False,
                trust_anchors=False, min_tls="Tls12", max_tls="Tls13",
                key_shares=(), delegated="", rsl=0, ext_order=())

ORDER8 = ("header_table_size", "enable_push", "max_concurrent_streams",
          "initial_window_size", "max_frame_size", "max_header_list_size",
          "enable_connect_protocol", "no_rfc7540_priorities")
SAFARI_ORDER1 = ("header_table_size", "enable_push", "initial_window_size",
                 "max_concurrent_streams", "max_frame_size",
                 "max_header_list_size", "enable_connect_protocol",
                 "no_rfc7540_priorities")

def chrome_h2(n):
    d = dict(win=6291456, conn_win=15728640, max_header_list=262144,
             header_table=65536, max_concurrent=None, max_frame=None,
             enable_push=None, enable_connect=False, no_rfc7540=False,
             stream_id=None, dep=(0, 255, True), priorities=(),
             settings_order=ORDER8,
             pseudo=("method", "authority", "scheme", "path"))
    if n == 1: d["max_concurrent"] = 1000
    elif n == 2: d.update(max_concurrent=1000, enable_push=False)
    elif n == 3: d["enable_push"] = False
    else: raise ValueError(n)
    return d

FF_PRIOS = ((3, 0, 200, False), (5, 0, 100, False), (7, 0, 0, False),
            (9, 7, 0, False), (11, 3, 0, False), (13, 0, 240, False))

def firefox_h2(n):
    d = dict(win=131072, conn_win=12517377 + 65535, max_header_list=None,
             header_table=None, max_concurrent=None, max_frame=16384,
             enable_push=None, enable_connect=False, no_rfc7540=False,
             stream_id=None, dep=None, priorities=(),
             settings_order=ORDER8,
             pseudo=("method", "path", "authority", "scheme"))
    if n == 1: d.update(stream_id=3, header_table=65536, enable_push=False,
                        dep=(0, 41, False))
    elif n == 2: d.update(stream_id=15, header_table=65536, dep=(13, 41, False),
                          priorities=FF_PRIOS)
    elif n == 3: d.update(stream_id=3, header_table=65536, enable_push=False,
                          max_concurrent=0, dep=(0, 41, False))
    elif n == 4: d.update(stream_id=3, header_table=4096, enable_push=False,
                          win=32768, dep=(0, 41, False))
    else: raise ValueError(n)
    return d

def safari_h2(n):
    d = dict(win=None, conn_win=None, max_header_list=None, header_table=None,
             max_concurrent=100, max_frame=None, enable_push=None,
             enable_connect=False, no_rfc7540=False, stream_id=None, dep=None,
             priorities=(), settings_order=None, pseudo=None)
    p1 = ("method", "scheme", "path", "authority")
    p2 = ("method", "scheme", "authority", "path")
    if n == 1: d.update(win=2097152, conn_win=10551295, pseudo=p1, settings_order=SAFARI_ORDER1)
    elif n == 2: d.update(win=2097152, conn_win=10551295, enable_push=False, pseudo=p1, settings_order=SAFARI_ORDER1)
    elif n == 3: d.update(win=2097152, conn_win=10485760, enable_push=False, enable_connect=True, no_rfc7540=True, pseudo=p2, settings_order=ORDER8)
    elif n == 4: d.update(win=4194304, conn_win=10551295, pseudo=p1, settings_order=SAFARI_ORDER1)
    elif n == 5: d.update(win=4194304, conn_win=10551295, enable_push=False, pseudo=p1, settings_order=SAFARI_ORDER1)
    elif n == 6: d.update(win=2097152, conn_win=10485760, enable_push=False, no_rfc7540=True, pseudo=p2, settings_order=ORDER8)
    else: raise ValueError(n)
    return d

def opera_h2():
    d = chrome_h2(3)
    return d

def okhttp_h2():
    return dict(win=16777216, conn_win=16777216, max_header_list=None,
                header_table=None, max_concurrent=None, max_frame=None,
                enable_push=None, enable_connect=False, no_rfc7540=False,
                stream_id=None, dep=None, priorities=(), settings_order=ORDER8,
                pseudo=("method", "path", "authority", "scheme"))

HEADER_SET = {
    "header_initializer": "Chrome",
    "header_initializer_with_zstd": "ChromeZstd",
    "header_initializer_with_zstd_priority": "ChromeZstdPriority",
    "header_initializer_chrome152": "Chrome152",
    "header_initializer_for_15": "Safari15",
    "header_initializer_for_16_17": "Safari1617",
    "header_initializer_for_18": "Safari18",
    "okhttp": "OkHttp",
}
# firefox header_initializer -> Firefox, with_zstd -> FirefoxZstd (disambiguate by family)

# firefox variant expr carries keyshares as 4th positional; fix parse:
def parse_tls_expr_ff(t):
    m = re.fullmatch(r'tls_options!\((\d+),([A-Za-z0-9_]+),([A-Za-z0-9_]+)(?:,([A-Za-z0-9_]+))?\)', t)
    if m: return ("variant", int(m.group(1)), m.group(2), m.group(3), m.group(4))
    raise ValueError(t)

def resolve_tls2(family, expr):
    kind = expr[0]
    if kind == "variant":
        if family == "chrome":
            _, n, a, b, _ = expr; return chrome_tls(n, a, b)
        if family == "firefox":
            _, n, a, b, ks = expr; return firefox_tls(n, a, b, ks)
        if family == "safari":
            _, n, a, b, c = expr; return safari_tls(n, a, b, c)
        if family == "opera":
            return opera_tls(expr[2])
    if kind == "okhttp": return okhttp_tls(expr)
    raise ValueError(expr)

def resolve_h2(family, expr):
    kind = expr[0]
    if kind == "variant":
        n = expr[1]
        if family == "chrome": return chrome_h2(n)
        if family == "firefox": return firefox_h2(n)
        if family == "safari": return safari_h2(n)
        if family == "opera": return opera_h2()
    if kind == "okhttp": return okhttp_h2()
    raise ValueError(expr)

# re-parse with firefox/safari-aware tls expr
def parse_h2_expr(t):
    m = re.fullmatch(r'http2_options!\((\d*)\)', t)
    if m:
        return ("variant", int(m.group(1)) if m.group(1) else None)
    raise ValueError("bad h2 expr: " + t)

def parse_tls_expr_safari(t):
    m = re.fullmatch(r'tls_options!\((\d+),([A-Za-z0-9_]+)(?:,([A-Za-z0-9_]+))?(?:,([A-Za-z0-9_]+))?\)', t)
    if m:
        return ("variant", int(m.group(1)), m.group(2), m.group(3), m.group(4))
    raise ValueError("bad safari tls expr: " + t)

def parse_tls_expr2(t, family):
    if family == "firefox":
        return parse_tls_expr_ff(t)
    if family == "safari":
        return parse_tls_expr_safari(t)
    m = re.fullmatch(r'tls_options!\((\d+)(?:,([A-Za-z0-9_]+))?(?:,([A-Za-z0-9_]+))?\)', t)
    if m:
        return ("variant", int(m.group(1)), m.group(2), m.group(3), None)
    m = re.fullmatch(r'tls_options!\(([A-Za-z0-9_]+)\)', t)
    if m:
        return ("variant", None, m.group(1), None, None)
    raise ValueError("bad tls expr: " + t)

# ------------------------------------------------- resolve all profiles
def parse_family2(fname, family):
    src = strip_comments(read(os.path.join(P, fname)))
    mods = {}
    for body in find_invocations(src, "mod_generator"):
        args = split_top(body)
        name = norm(args[0][0])
        if family == "okhttp":
            mods[name] = {"tls_raw": ("okhttp", norm(args[1][0]), args[1][1]),
                          "h2_raw": ("okhttp",),
                          "header": "okhttp", "platforms": single_ua(args[2][1])}
            continue
        if len(args) == 5:
            tls_raw = parse_tls_expr2(norm(args[1][0]), family)
            h2_raw = parse_h2_expr(norm(args[2][0]))
            header = norm(args[3][0])
            entries = (parse_platforms_toks(args[4][2]) if args[4][0].lstrip().startswith('[')
                       else single_ua(args[4][1]))
        elif len(args) == 4:
            ref = norm(args[1][0])
            assert ref.endswith("::build_emulation"), ref
            tls_raw = ("reuse", ref.split("::")[0])
            h2_raw = ("reuse", ref.split("::")[0])
            header = norm(args[2][0])
            entries = (parse_platforms_toks(args[3][2]) if args[3][0].lstrip().startswith('[')
                       else single_ua(args[3][1]))
        else:
            raise ValueError(f"{fname}: unexpected arg count in {name}")
        mods[name] = {"tls_raw": tls_raw, "h2_raw": h2_raw,
                      "header": header, "platforms": entries}
    return mods

FAMILIES2 = {}
for fname, fam in [("chrome.rs", "chrome"), ("firefox.rs", "firefox"),
                   ("safari.rs", "safari"), ("opera.rs", "opera"),
                   ("okhttp.rs", "okhttp")]:
    FAMILIES2[fam] = parse_family2(fname, fam)

def resolve_all():
    resolved = {}
    def get(family, name):
        key = (family, name)
        if key in resolved: return resolved[key]
        m = FAMILIES2[family][name]
        if m["tls_raw"][0] == "reuse":
            ref = get(family, m["tls_raw"][1])
            tls, h2 = ref["tls"], ref["h2"]
        else:
            tls = resolve_tls2(family, m["tls_raw"])
            h2 = resolve_h2(family, m["h2_raw"])
        header = m["header"]
        if family == "firefox":
            hs = "Firefox" if header == "header_initializer" else "FirefoxZstd"
        else:
            hs = HEADER_SET[header]
        r = {"tls": tls, "h2": h2, "header_set": hs, "platforms": m["platforms"],
             "family": family, "module": name}
        resolved[key] = r
        return r
    for family, mods in FAMILIES2.items():
        for name in mods:
            pub = NAMES.get(name)
            if not pub:
                raise ValueError("no public name for module " + name)
            r = get(family, name)
            r["name"] = pub
    return [resolved[(f, n)] for f, ms in FAMILIES2.items() for n in ms]

PROFILES = resolve_all()
print(f"resolved {len(PROFILES)} profiles", file=sys.stderr)
