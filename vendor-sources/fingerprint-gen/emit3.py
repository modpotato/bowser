# ---- emission

PLAT = {"MacOS": "MacOS", "MacOs": "MacOS", "Linux": "Linux",
        "Android": "Android", "Windows": "Windows", "IOS": "Ios"}
CANON = ("Windows", "MacOS", "Linux", "Android", "Ios")

def infer_platform(ua, family):
    if family == "okhttp":
        return "Android"
    if "iPhone" in ua or "iPad" in ua:
        return "Ios"
    return "MacOS"

def rs(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

def opt_u32(v):
    return "None" if v is None else f"Some({v})"

def opt_bool(v):
    return "None" if v is None else ("Some(true)" if v else "Some(false)")

def bo(b):
    return "true" if b else "false"

# L-pool: TLS list strings (kept literal; referenced by name from TlsConfig)
lpool = {}
def L(s):
    if s not in lpool:
        lpool[s] = f"L{len(lpool)}"
    return lpool[s]

# S-pool: UA + sec-ch-ua strings -> prefix-compressed (plen, suffix) entries
s_strings = set()
for p in PROFILES:
    d = p["tls"]
    for k in ("ciphers", "curves", "sigalgs", "delegated"):
        if d[k]:
            L(d[k])
    secs = {sec for _, sec, _ in p["platforms"] if sec}
    assert len(secs) <= 1
    p["_sec_str"] = secs.pop() if secs else None
    slots = {}
    for o, sec, ua in p["platforms"]:
        plat = PLAT[o] if o != "INFER" else infer_platform(ua, p["family"])
        slots[plat] = ua
        s_strings.add(ua)
    if p["_sec_str"]:
        s_strings.add(p["_sec_str"])
    p["_ua_strs"] = [slots.get(pl) for pl in CANON]

assert all(s.isascii() for s in s_strings)
ordered = sorted(s_strings)
s_index = {}
pool = []
prev = ""
for s in ordered:
    i = 0
    n = min(len(prev), len(s))
    while i < n and prev[i] == s[i]:
        i += 1
    s_index[s] = len(pool)
    pool.append((i, s[i:]))
    prev = s

for p in PROFILES:
    p["_sec"] = s_index[p["_sec_str"]] if p["_sec_str"] else None
    p["_ua"] = [s_index[u] if u else None for u in p["_ua_strs"]]

out = []
w = out.append
w("//! Generated fingerprint tables. DO NOT EDIT BY HAND.")
w("//!")
w("//! Transcribed from the profile DATA in 0x676e67/wreq-util")
w("//! (`src/emulate/profile/`) at commit e3922a2, under Apache-2.0;")
w("//! see ATTRIBUTION.md.")
w("//!")
w("//! Regenerate with (from this directory):")
w("//!   cat gen_fp.py emit3.py | python3 - data.rs.new")
w("//!   cmp data.rs.new $REPO/crates/bowser_net/src/fingerprint/data.rs")
w("//! (gen_fp.py parses the pinned wreq-util sources, emit3.py emits this file.)")
w("//! The generator must reproduce this file byte-identically; if `cmp`")
w("//! differs, the pin moved or the generator changed -- review before committing.")
w("//!")
w("//! String storage: `UA_POOL` is prefix-compressed. Entry `i` holds the")
w("//! byte length of the prefix it shares with entry `i-1`'s full string,")
w("//! plus the remaining suffix; entry 0 (and any entry with length 0)")
w("//! stands alone. See `ua_pool_get` for the decoder.")
w("")
w("use super::{HeaderSet, Http2Config, Priority, Profile, StreamDependency, TlsConfig, TlsVersion};")
w("")
for s, ident in sorted(lpool.items(), key=lambda kv: kv[1]):
    w(f"const {ident}: &str = {rs(s)};")
w("")
w("#[rustfmt::skip]")
w("pub(super) static UA_POOL: &[(u16, &str)] = &[")
for plen, suffix in pool:
    w(f"({plen},{rs(suffix)}),")
w("];")
w("")

tls_seen, h2_seen = {}, {}

def tls_key(d):
    return (d["ciphers"], d["curves"], d["sigalgs"], d["alpn"], d["grease"],
            d["permute"], d["ech"], d["psk"], d["psk_skip"], d["psk_dhe_ke"],
            d["session_ticket"], d["ocsp"], d["sct"], d["alps"], d["alps_new"],
            d["cert_comp"], d["sigalg_grease"], d["trust_anchors"],
            d["min_tls"], d["max_tls"], d["key_shares"], d["delegated"],
            d["rsl"], d["ext_order"])

def h2_key(d):
    return (d["win"], d["conn_win"], d["max_header_list"], d["header_table"],
            d["max_concurrent"], d["max_frame"], d["enable_push"],
            d["enable_connect"], d["no_rfc7540"], d["stream_id"], d["dep"],
            d["priorities"], d["settings_order"], d["pseudo"])

for p in PROFILES:
    k = tls_key(p["tls"])
    if k not in tls_seen:
        tls_seen[k] = f"TLS_{p['tls']['family'].upper()}_{len(tls_seen)}"
    p["_tls"] = tls_seen[k]
    k = h2_key(p["h2"])
    if k not in h2_seen:
        h2_seen[k] = f"HTTP2_{p['family'].upper()}_{len(h2_seen)}"
    p["_h2"] = h2_seen[k]

print(f"{len(tls_seen)} TLS configs, {len(h2_seen)} HTTP/2 configs, "
      f"{len(lpool)} list strings, {len(pool)} compressed UA strings",
      file=sys.stderr)

def lref(s):
    return L(s) if s else '""'

def opt_idx(v):
    return f"Some({v})" if v is not None else "None"

for k, ident in sorted(tls_seen.items(), key=lambda kv: kv[1]):
    p = next(p for p in PROFILES if tls_key(p["tls"]) == k)
    d = p["tls"]
    w("#[rustfmt::skip]")
    w(f"static {ident}: TlsConfig = TlsConfig{{")
    w(f"ciphers:{lref(d['ciphers'])},curves:{lref(d['curves'])},sigalgs:{lref(d['sigalgs'])},")
    alpn = ",".join(rs(a) for a in d["alpn"])
    fmap = [("grease", "grease"), ("permute", "permute_extensions"),
            ("ech", "ech_grease"), ("psk", "pre_shared_key"),
            ("psk_skip", "psk_skip_session_tickets"),
            ("psk_dhe_ke", "psk_dhe_ke"),
            ("session_ticket", "session_ticket"),
            ("ocsp", "ocsp_stapling"), ("sct", "signed_cert_timestamps"),
            ("alps_new", "alps_new_codepoint"),
            ("sigalg_grease", "sigalg_grease"),
            ("trust_anchors", "trust_anchors")]
    flags = " ".join(f"{pub}:{bo(d[key])}," for key, pub in fmap)
    w(f"alpn:&[{alpn}],{flags}")
    w(f"alps:{rs(d['alps'])},cert_compressors:{rs(d['cert_comp'])},")
    w(f"min_tls:TlsVersion::{d['min_tls']},max_tls:TlsVersion::{d['max_tls']},")
    ks = ",".join(rs(x) for x in d["key_shares"])
    w(f"key_shares:&[{ks}],delegated_credentials:{lref(d['delegated'])},")
    w(f"record_size_limit:{d['rsl']},")
    eo = ",".join(str(x) for x in d["ext_order"])
    w(f"extension_order:&[{eo}],}};")

for k, ident in sorted(h2_seen.items(), key=lambda kv: kv[1]):
    p = next(p for p in PROFILES if h2_key(p["h2"]) == k)
    d = p["h2"]
    w("#[rustfmt::skip]")
    w(f"static {ident}: Http2Config = Http2Config{{")
    w(f"initial_window_size:{opt_u32(d['win'])},")
    w(f"initial_connection_window_size:{opt_u32(d['conn_win'])},")
    w(f"max_header_list_size:{opt_u32(d['max_header_list'])},")
    w(f"header_table_size:{opt_u32(d['header_table'])},")
    w(f"max_concurrent_streams:{opt_u32(d['max_concurrent'])},")
    w(f"max_frame_size:{opt_u32(d['max_frame'])},")
    w(f"enable_push:{opt_bool(d['enable_push'])},")
    w(f"enable_connect_protocol:{bo(d['enable_connect'])},")
    w(f"no_rfc7540_priorities:{bo(d['no_rfc7540'])},")
    w(f"initial_stream_id:{opt_u32(d['stream_id'])},")
    if d["dep"] is None:
        w("stream_dependency:None,")
    else:
        par, wt, ex = d["dep"]
        w(f"stream_dependency:Some(StreamDependency{{parent:{par},weight:{wt},exclusive:{bo(ex)}}}),")
    if d["priorities"]:
        pr = ",".join(
            f"Priority{{stream:{s},parent:{pa},weight:{wt},exclusive:{bo(ex)}}}"
            for s, pa, wt, ex in d["priorities"])
        w(f"priorities:&[{pr}],")
    else:
        w("priorities:&[],")
    so = ",".join(rs(x) for x in d["settings_order"])
    w(f"settings_order:&[{so}],")
    po = ",".join(rs(x) for x in d["pseudo"])
    w(f"pseudo_order:&[{po}],}};")

w("")
w("#[rustfmt::skip]")
w("pub(super) static PROFILES: &[Profile] = &[")
for p in sorted(PROFILES, key=lambda p: p["name"]):
    ua = ",".join(opt_idx(v) for v in p["_ua"])
    w(f'Profile{{name:{rs(p["name"])},tls:&{p["_tls"]},http2:&{p["_h2"]},'
      f'header_set:HeaderSet::{p["header_set"]},sec_ch_ua:{opt_idx(p["_sec"])},'
      f'user_agents:[{ua}]}},')
w("];")
w("")

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("wrote", sys.argv[1], file=sys.stderr)
