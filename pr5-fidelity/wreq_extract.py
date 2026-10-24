#!/usr/bin/env python3
"""Throwaway baseline extractor: canonical wreq-util emulate profile data.
Source: ~/workspace/bowser/vendor-sources/wreq-util @ e3922a2 (v0.2.0).
Output: ~/workspace/bowser/pr5-fidelity/wreq_baseline.json
"""
import json, re, sys, os

ROOT = os.path.expanduser("~/workspace/bowser/vendor-sources/wreq-util/src/emulate")

# ---------- low-level parsing helpers ----------

def strip_line_comments(src):
    """Remove // comments outside string literals (handles r#\"..\"# raw strings)."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c == '/' and i + 1 < n and src[i+1] == '/':
            while i < n and src[i] != '\n':
                i += 1
        elif c == 'r' and i + 1 < n and src[i+1] == '#':
            # raw string r#"..."# (assume single #)
            j = src.index('"', i + 2)
            k = src.index('"#', j + 1)
            out.append(src[i:k+2]); i = k + 2
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == '\\' else 1
            out.append(src[i:j+1]); i = j + 1
        else:
            out.append(c); i += 1
    return ''.join(out)

def split_top(s):
    """Split on top-level commas (depth 0), aware of ()[]{} and strings incl. r#\"\"#."""
    parts, depth, cur, i, n = [], 0, [], 0, len(s)
    while i < n:
        c = s[i]
        if c in '([{': depth += 1; cur.append(c); i += 1
        elif c in ')]}': depth -= 1; cur.append(c); i += 1
        elif c == ',' and depth == 0:
            parts.append(''.join(cur)); cur = []; i += 1
        elif c == 'r' and i + 2 < n and s[i+1] == '#' and s[i+2] == '"':
            j = s.index('"#', i + 3); cur.append(s[i:j+2]); i = j + 2
        elif c == '"':
            j = i + 1
            while s[j] != '"': j += 2 if s[j] == '\\' else 1
            cur.append(s[i:j+1]); i = j + 1
        else:
            cur.append(c); i += 1
    parts.append(''.join(cur))
    return [p.strip() for p in parts]

def extract_macro_invocations(src, name):
    """Yield inner text of each `name!( ... )` with balanced parens."""
    res, i = [], 0
    pat = name + '!('
    while True:
        j = src.find(pat, i)
        if j < 0: break
        k, depth = j + len(pat), 1
        while depth:
            c = src[k]
            if c == 'r' and k + 2 < len(src) and src[k+1] == '#' and src[k+2] == '"':
                k = src.index('"#', k + 3) + 2; continue
            if c == '"':
                k += 1
                while src[k] != '"': k += 2 if src[k] == '\\' else 1
                k += 1; continue
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            k += 1
        res.append(src[j+len(pat):k-1]); i = k
    return res

def parse_str_lit(tok):
    tok = tok.strip()
    if tok.startswith('r#"') and tok.endswith('"#'):
        return tok[3:-2]
    if tok.startswith('"') and tok.endswith('"'):
        return bytes(tok[1:-1], 'utf-8').decode('unicode_escape')
    return None

def read(p):
    with open(p) as f: return f.read()

# ---------- consts ----------

def parse_consts(src):
    """name -> resolved value for join!(&str) and KeyShare list consts."""
    consts = {}
    for m in re.finditer(r'(?:pub(?:\(super\))?\s+)?const\s+(\w+)\s*:\s*&(static\s+)?str\s*=\s*join!\((.*?)\);', src, re.S):
        name, body = m.group(1), m.group(3)
        items = [parse_str_lit(p) for p in split_top(body)]
        sep = items[0]
        consts[name] = sep.join(x for x in items[1:] if x is not None)
    for m in re.finditer(r'pub\s+const\s+(\w+)\s*:\s*&\[[^\]]*\]\s*=\s*&\[(.*?)\];', src, re.S):
        consts[m.group(1)] = [x.strip() for x in split_top(m.group(2))]
    # single-level consts referencing others (e.g. CURVES_3 defined via join! in one line handled above)
    return consts

# ---------- tls_options! / http2_options! macro arms ----------

def parse_macro_rules(src, macroname):
    """Return list of (arm_key, params, body) for `macro_rules! <name>`."""
    m = re.search(r'macro_rules!\s*' + macroname + r'\s*\{', src)
    if not m: return []
    i, depth = m.end(), 1
    while depth:
        c = src[i]
        if c == '{': depth += 1
        elif c == '}': depth -= 1
        i += 1
    body = src[m.end():i-1]
    arms = []
    # arms look like: (1) => { ... };  (6, $curves:expr) => { ... };  (@base $builder:expr) => { ... };
    for am in re.finditer(r'\(\s*(@\w+|\d+)?(.*?)\)\s*=>\s*\{', body):
        key = am.group(1) or '()'
        params = am.group(2).strip()
        j, d = am.end(), 1
        while d:
            if body[j] == '{': d += 1
            elif body[j] == '}': d -= 1
            j += 1
        arms.append((key, params, body[am.end():j-1]))
    return arms

def builder_calls(body):
    """Ordered list of (method, argtext) for `.method(args)` chains in body text."""
    calls = []
    for m in re.finditer(r'\.(\w+)\s*\(', body):
        name = m.group(1)
        j, depth = m.end(), 1
        s = body
        while depth:
            c = s[j]
            if c == 'r' and j + 2 < len(s) and s[j+1] == '#' and s[j+2] == '"':
                j = s.index('"#', j + 3) + 2; continue
            if c == '"':
                j += 1
                while s[j] != '"': j += 2 if s[j] == '\\' else 1
                j += 1; continue
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            j += 1
        calls.append((name, s[m.end():j-1].strip()))
    return calls

def resolve_value(arg, consts, placeholders):
    """Resolve an arg expression to a python value (best effort, canonical)."""
    a = arg.strip()
    if a in placeholders:
        return resolve_value(placeholders[a], consts, {})
    if a in consts:
        v = consts[a]
        return list(v) if isinstance(v, list) else v
    lit = parse_str_lit(a)
    if lit is not None: return lit
    if a == 'true': return True
    if a == 'false': return False
    if re.fullmatch(r'\d+', a): return int(a)
    if re.fullmatch(r'[\d\s+\-*/()]+', a):
        try: return eval(a, {'__builtins__': {}}, {})
        except Exception: pass
    if a.startswith('CHROME_TRUST_ANCHORS'): return '<chromium_roots|empty-by-feature>'
    if a == 'CERTIFICATE_COMPRESSORS': return ['brotli']
    return a  # e.g. AlpsProtocol::HTTP2, TlsVersion::TLS_1_2, val.xxx

# ---------- family TLS struct defaults + From body ----------

def parse_tls_struct(src):
    m = re.search(r'pub struct (\w+TlsConfig)\s*\{(.*?)\n\}', src, re.S)
    if not m: return None, {}
    name, body = m.group(1), m.group(2)
    defaults = {}
    for dm in re.finditer(r'#\[builder\((.*?)\)\]\s*\n\s*(\w+)\s*:', body):
        attrs, field = dm.group(1), dm.group(2)
        dm2 = re.search(r'default\s*=\s*([^\s,\)]+)', attrs)
        if dm2:
            defaults[field] = dm2.group(1)
        else:
            defaults[field] = None  # #[builder(default)] or strip_option
    return name, defaults

def parse_from_body(src, structname):
    m = re.search(r'impl From<' + structname + r'> for TlsOptions\s*\{(.*)', src, re.S)
    if not m: return []
    return builder_calls(m.group(1))

# ---------- header templates (parsed mechanically from header.rs + macros.rs) ----------

HEADER_NAME_MAP = {
    'ACCEPT': 'accept', 'ACCEPT_LANGUAGE': 'accept-language',
    'ACCEPT_ENCODING': 'accept-encoding', 'USER_AGENT': 'user-agent',
}

def resolve_hname(tok):
    tok = tok.strip()
    lit = parse_str_lit(tok)
    if lit is not None:
        return lit
    m = re.match(r'HeaderName::from_static\((.*)\)$', tok, re.S)
    if m: return parse_str_lit(m.group(1))
    return HEADER_NAME_MAP.get(tok, tok)

def resolve_hval(tok):
    tok = tok.strip()
    m = re.match(r'HeaderValue::from_static\((.*)\)$', tok, re.S)
    if m:
        inner = m.group(1).strip()
        lit = parse_str_lit(inner)
        return lit if lit is not None else {'var': inner}
    return tok

def parse_header_macros(macros_src):
    """Extract expansions of the header_* helper macros from macros.rs (macro_rules! bodies)."""
    exp = {}
    arms = parse_macro_rules(macros_src, 'header_chrome_sec_fetch')
    exp['header_chrome_sec_fetch'] = []
    for key, params, btext in arms:
        for cm in re.finditer(r'\$headers\.insert\("((?:[^"\\]|\\.)*)",\s*HeaderValue::from_static\("((?:[^"\\]|\\.)*)"\)\);', btext):
            exp['header_chrome_sec_fetch'].append((cm.group(1), cm.group(2)))
    arms = parse_macro_rules(macros_src, 'header_firefox_sec_fetch')
    exp['header_firefox_sec_fetch'] = []
    for key, params, btext in arms:
        for cm in re.finditer(r'\$headers\.insert\("((?:[^"\\]|\\.)*)",\s*HeaderValue::from_static\("((?:[^"\\]|\\.)*)"\)\);', btext):
            exp['header_firefox_sec_fetch'].append((cm.group(1), cm.group(2)))
    arms = parse_macro_rules(macros_src, 'header_chrome_accept')
    for key, params, btext in arms:
        m = re.search(r'HeaderValue::from_static\(\s*("(?:[^"\\]|\\.)*")\s*\)', btext, re.S)
        exp['header_chrome_accept'] = parse_str_lit(m.group(1))
    arms = parse_macro_rules(macros_src, 'header_firefox_accept')
    exp['header_firefox_accept'] = {}
    for key, params, btext in arms:
        vals = {}
        for cm in re.finditer(r'\$headers\.insert\(\s*(ACCEPT|ACCEPT_LANGUAGE|ACCEPT_ENCODING)\s*,\s*HeaderValue::from_static\(\s*("(?:[^"\\]|\\.)*")', btext):
            vals[HEADER_NAME_MAP[cm.group(1)]] = parse_str_lit(cm.group(2))
        exp['header_firefox_accept']['zstd' if 'zstd' in params else 'base'] = vals
    arms = parse_macro_rules(macros_src, 'header_chrome_accept_encoding')
    exp['header_chrome_accept_encoding'] = {}
    for key, params, btext in arms:
        vals = []
        for cm in re.finditer(r'\$headers\.insert\(\s*(ACCEPT_LANGUAGE|ACCEPT_ENCODING)\s*,\s*HeaderValue::from_static\(\s*("(?:[^"\\]|\\.)*")', btext):
            vals.append((HEADER_NAME_MAP[cm.group(1)], parse_str_lit(cm.group(2))))
        label = 'zstd' if params.startswith('zstd,') else ('zstd_only' if 'zstd_only' in params else 'base')
        exp['header_chrome_accept_encoding'][label] = vals
    return exp

def split_statements(s):
    """Split on top-level semicolons, aware of ()[]{} and strings incl. r#\"\"#."""
    parts, depth, cur, i, n = [], 0, [], 0, len(s)
    while i < n:
        c = s[i]
        if c in '([{': depth += 1; cur.append(c); i += 1
        elif c in ')]}': depth -= 1; cur.append(c); i += 1
        elif c == ';' and depth == 0:
            parts.append(''.join(cur)); cur = []; i += 1
        elif c == 'r' and i + 2 < n and s[i+1] == '#' and s[i+2] == '"':
            j = s.index('"#', i + 3); cur.append(s[i:j+2]); i = j + 2
        elif c == '"':
            j = i + 1
            while s[j] != '"': j += 2 if s[j] == '\\' else 1
            cur.append(s[i:j+1]); i = j + 1
        else:
            cur.append(c); i += 1
    parts.append(''.join(cur))
    return [p.strip() for p in parts if p.strip()]

def parse_header_initializers(header_src, macro_exp):
    """name -> ordered list of (header_name, value-or-placeholder)."""
    inits = {}
    for m in re.finditer(r'pub fn (\w+)\((.*?)\)(?:\s*->\s*HeaderMap)?\s*\{(.*?)\n\}\n', header_src, re.S):
        fname, params, body = m.group(1), m.group(2), m.group(3)
        param_names = [p.strip().split(':')[0].strip() for p in split_top(params)]
        has_sec, has_ua = 'sec_ch_ua' in param_names, 'ua' in param_names
        headers = []
        # walk statements in order
        stmts = split_statements(body)
        for st in stmts:
            st = st.strip()
            # strip leading attribute lines like #[cfg(...)] / #[inline]
            st = re.sub(r'^(?:#\[[^\]]*\]\s*)+', '', st).strip()
            if not st or st.startswith('let mut headers'):
                continue
            im = re.match(r'headers\.insert\((.*)\)$', st, re.S)
            if im:
                parts = [x for x in split_top(im.group(1)) if x]
                a, b = parts[0], parts[1]
                headers.append((resolve_hname(a), resolve_hval(b)))
                continue
            mm = re.match(r'header_chrome_sec_ch_ua!\((.*)\)$', st, re.S)
            if mm:
                a = split_top(mm.group(1))
                headers.append(('sec-ch-ua', {'var': a[1].strip()}))
                headers.append(('sec-ch-ua-mobile', {'mobile_flag': True}))
                headers.append(('sec-ch-ua-platform', {'platform_str': True}))
                continue
            mm = re.match(r'header_chrome_ua!\(headers,\s*(\w+)\)$', st)
            if mm: headers.append(('user-agent', {'var': mm.group(1)})); continue
            mm = re.match(r'header_firefox_ua!\(headers,\s*(\w+)\)$', st)
            if mm: headers.append(('user-agent', {'var': mm.group(1)})); continue
            if re.match(r'header_chrome_accept!\(headers\)$', st):
                headers.append(('accept', macro_exp['header_chrome_accept'])); continue
            if re.match(r'header_chrome_sec_fetch!\(headers\)$', st):
                headers.extend(macro_exp['header_chrome_sec_fetch']); continue
            if re.match(r'header_firefox_sec_fetch!\(headers\)$', st):
                headers.extend(macro_exp['header_firefox_sec_fetch']); continue
            mm = re.match(r'header_chrome_accept_encoding!\((.*)\)$', st, re.S)
            if mm:
                a = split_top(mm.group(1))
                label = a[0] if len(a) == 2 else 'base'
                headers.extend(macro_exp['header_chrome_accept_encoding'][label]); continue
            mm = re.match(r'header_firefox_accept!\((.*)\)$', st, re.S)
            if mm:
                a = split_top(mm.group(1))
                label = a[0] if len(a) == 2 else 'base'
                d = macro_exp['header_firefox_accept'][label]
                headers.extend([(k, d[k]) for k in ('accept', 'accept-language', 'accept-encoding') if k in d])
                continue
        inits[fname] = headers
    return inits

PLATFORM_STR = {'MacOS': '"macOS"', 'Linux': '"Linux"', 'Windows': '"Windows"',
                'Android': '"Android"', 'IOS': '"iOS"'}
IS_MOBILE = {'Android', 'IOS'}

def materialize_headers(template, sec_ch_ua, ua, platform):
    out = []
    for name, val in template:
        if isinstance(val, dict):
            if 'var' in val:
                v = {'sec_ch_ua': sec_ch_ua, 'ua': ua}.get(val['var'], val['var'])
            elif 'mobile_flag' in val:
                v = '?1' if platform in IS_MOBILE else '?0'
            elif 'platform_str' in val:
                v = PLATFORM_STR[platform]
            else:
                v = val
        else:
            v = val
        out.append([name, v])
    return out

# ---------- profile modules ----------

def parse_profile_file(path):
    src = strip_line_comments(read(path))
    mods = {}
    for inner in extract_macro_invocations(src, 'mod_generator'):
        args = split_top(inner)
        mod_name = args[0]
        if len(args) == 5:
            tls_e, h2_e, hinit, tuples_s = args[1], args[2], args[3], args[4]
        elif len(args) == 4:
            tls_e = h2_e = None
            build_e, hinit, tuples_s = args[1], args[2], args[3]
        else:
            raise ValueError(f"unexpected arity {len(args)} in {path}:{mod_name}")
        tuples = []
        t = tuples_s.strip()
        if not t.startswith('['):
            # fixed-UA form: single UA expr for all platforms
            ua = parse_str_lit(t)
            assert ua is not None, t[:80]
            tuples.append((None, None, ua))
        else:
            assert t.endswith(']'), t[:60]
            for tup in split_top(t[1:-1]):
                tup = tup.strip()
                assert tup.startswith('(') and tup.endswith(')')
                f = split_top(tup[1:-1])
                if len(f) == 3:
                    plat, sec, ua = f[0], parse_str_lit(f[1]), parse_str_lit(f[2])
                    assert sec is not None and ua is not None, tup[:80]
                    tuples.append((plat, sec, ua))
                elif len(f) == 2:
                    plat, ua = f[0], parse_str_lit(f[1])
                    assert ua is not None, tup[:80]
                    tuples.append((plat, None, ua))
                else:
                    raise ValueError(f"bad tuple {tup[:80]}")
        mods[mod_name] = {
            'tls_expr': tls_e, 'http2_expr': h2_e,
            'build_emulation': None if len(args) == 5 else build_e,
            'header_init': hinit, 'platforms': tuples,
        }
    return mods

def parse_okhttp(path):
    src = strip_line_comments(read(path))
    consts = parse_consts(src)
    mods = {}
    for inner in extract_macro_invocations(src, 'mod_generator'):
        args = split_top(inner)
        name = args[0]
        cm = re.match(r'join!\((.*)\)$', args[1], re.S)
        if cm:
            items = [parse_str_lit(p) for p in split_top(cm.group(1))]
            cipher = items[0].join(x for x in items[1:])
        else:
            cipher = consts[args[1].strip()]
        mods[name] = {'cipher_list': cipher, 'ua': parse_str_lit(args[2])}
    return mods

# ---------- TLS resolution ----------

def expand_base_calls(body, arms, macroname):
    """Expand a nested `macroname!(@base ...)` invocation inside a variant arm body.
    Returns (base_calls_with_substitution, body_without_nested_invocation)."""
    base = [a for a in arms if a[0] == '@base']
    if not base:
        return [], body
    _, base_params, base_body = base[0]
    base_ph = re.findall(r'\$(\w+)', base_params)
    # find `macroname!(@base` occurrences directly (may be nested inside another invocation)
    for inv in extract_macro_invocations(body, macroname):
        iargs = split_top(inv)
        if iargs and iargs[0].strip() == '@base':
            inner_full = macroname + '!(' + inv + ')'
        else:
            # outer invocation containing a nested @base: search inside
            j = inv.find(macroname + '!(@base')
            if j < 0:
                continue
            k = j + len(macroname + '!(')
            depth = 1
            while depth:
                c = inv[k]
                if c == '(': depth += 1
                elif c == ')': depth -= 1
                k += 1
            inner_full = inv[j:k]
            iargs = split_top(inv[j + len(macroname + '!('):k - 1])
            # @base args are space-separated from the tag: "@base <builder>, ..."
            if not iargs or not iargs[0].strip().startswith('@base'):
                continue
            iargs = [iargs[0].strip()[len('@base'):].strip()] + iargs[1:]
        subst = {}
        # iargs[0] is now the builder arg (the '@base' tag was stripped above,
        # or this was a direct comma-separated @base invocation)
        first = iargs[0].strip()
        if first == '@base':
            iargs = iargs[1:]
        for ph, aval in zip(base_ph, iargs):
            subst['$' + ph] = aval.strip()
        calls = []
        for name, arg in builder_calls(base_body):
            a = arg.strip()
            calls.append((name, subst.get(a, a)))
        body = body.replace(inner_full, '/*base*/', 1)
        return calls, body
    return [], body

def parse_family_tls(famdir):
    src = read(os.path.join(famdir, 'tls.rs'))
    consts = parse_consts(src)
    # numeric consts
    for m in re.finditer(r'(?:pub\s+)?const\s+(\w+)\s*:\s*u\d+\s*=\s*(\d+)\s*;', src):
        consts[m.group(1)] = int(m.group(2))
    structname, defaults = parse_tls_struct(src)
    from_calls = parse_from_body(src, structname)
    arms = parse_macro_rules(src, 'tls_options')
    variants = {}
    for key, params, body in arms:
        if key.startswith('@'):
            continue
        base_calls, rest = expand_base_calls(body, arms, 'tls_options')
        calls = base_calls + builder_calls(rest)
        ph = re.findall(r'\$(\w+)', params)
        variants[key] = {'placeholders': ph, 'calls': calls}
    return {'consts': consts, 'struct': structname, 'defaults': defaults,
            'from_calls': from_calls, 'variants': variants}

def resolve_tls(fam_tls, tls_expr, mod_chain):
    """Return effective field dict for a profile's TLS config."""
    consts, defaults = fam_tls['consts'], fam_tls['defaults']
    m = re.match(r'tls_options!\((.*)\)$', tls_expr.strip(), re.S)
    assert m, tls_expr
    args = split_top(m.group(1))
    varnum = args[0]
    if varnum not in fam_tls['variants'] and '()' in fam_tls['variants']:
        varnum, args = '()', ['()'] + args  # e.g. opera tls_options!(CURVES)
    var = fam_tls['variants'][varnum]
    placeholders = {}
    for ph, aval in zip(var['placeholders'], args[1:]):
        placeholders['$' + ph] = aval.strip()
    fields = dict(defaults)
    for name, arg in var['calls']:
        if name in ('build', 'extend', 'new'):
            continue
        fields[name] = resolve_value(arg, consts, placeholders)
    # resolve default const refs and literals
    for k, v in list(fields.items()):
        if isinstance(v, str):
            fields[k] = resolve_value(v, consts, {})
    return {'variant': varnum, 'variant_args': [a.strip() for a in args[1:]],
            'fields': fields}

# ---------- HTTP/2 resolution ----------

def parse_pseudo_list(body):
    return re.findall(r'PseudoId::(\w+)', body)

def parse_settings_list(body):
    return re.findall(r'SettingId::(\w+)', body)

def parse_stream_dep(argtext):
    m = re.match(r'StreamDependency::new\((.*)\)$', argtext.strip(), re.S)
    a = split_top(m.group(1))
    sid = a[0].strip()
    sm = re.match(r'StreamId::(zero|from)\((\d*)\)', sid)
    sid = 0 if sm.group(1) == 'zero' else int(sm.group(2))
    return {'stream_id': sid, 'weight': int(a[1]), 'exclusive': a[2].strip() == 'true'}

def parse_family_http2(famdir):
    src = read(os.path.join(famdir, 'http2.rs'))
    arms = parse_macro_rules(src, 'http2_options')
    base_calls, variants = [], {}
    for key, params, body in arms:
        calls = builder_calls(body)
        if key == '@base':
            base_calls = calls
        else:
            variants[key] = calls
    # helper macros
    pseudo, settings, streamdep = {}, {}, {}
    for name in ['pseudo_order', 'headers_pseudo_order']:
        for k, p, b in parse_macro_rules(src, name):
            pseudo[k] = parse_pseudo_list(b)
    for k, p, b in parse_macro_rules(src, 'settings_order'):
        settings[k] = parse_settings_list(b)
    for k, p, b in parse_macro_rules(src, 'headers_stream_dependency'):
        streamdep[k] = b
    # priorities blocks (firefox http2_options!(2))
    return {'base': base_calls, 'variants': variants, 'pseudo': pseudo,
            'settings': settings, 'streamdep_raw': streamdep}

def resolve_http2(fam_h2, h2_expr):
    m = re.match(r'http2_options!\((.*)\)$', h2_expr.strip(), re.S)
    assert m, h2_expr
    args = split_top(m.group(1))
    varnum = args[0] if args != [''] else '()'
    calls = list(fam_h2['base']) + fam_h2['variants'].get(varnum, [])
    if varnum == '()' and '()' not in fam_h2['variants']:
        calls = fam_h2['base']  # opera: single () arm stored as base? handled below
    settings_vals, pseudo, sorder, sdep, priorities = {}, None, None, None, None
    for name, arg in calls:
        a = arg.strip()
        if name in ('headers_pseudo_order',):
            mm = re.match(r'(?:pseudo_order|headers_pseudo_order)!\((.*)\)$', a)
            key = mm.group(1).strip() or '()'
            pseudo = fam_h2['pseudo'].get(key, fam_h2['pseudo'].get('()'))
        elif name == 'settings_order':
            mm = re.match(r'settings_order!\((.*)\)$', a)
            key = mm.group(1).strip() or '()'
            sorder = fam_h2['settings'].get(key, fam_h2['settings'].get('()'))
        elif name == 'headers_stream_dependency':
            mm = re.match(r'headers_stream_dependency!\((.*)\)$', a)
            key = mm.group(1).strip() or '()'
            sdep = parse_stream_dep(fam_h2['streamdep_raw'][key])
        elif name == 'priorities':
            priorities = 'PRIORITIES_BLOCK:' + re.sub(r'\s+', ' ', a)[:400]
        elif name == 'build' or name == 'extend':
            continue
        else:
            settings_vals[name] = resolve_value(a, {}, {})
    # opera's http2_options!() has no numbered arm: base==full chain parsed as '()'
    return {'variant': varnum, 'settings': settings_vals, 'pseudo_order': pseudo,
            'settings_order': sorder, 'stream_dependency': sdep,
            'priorities': priorities}

# ---------- main ----------

def main():
    famdirs = {f: os.path.join(ROOT, 'profile', f) for f in
               ['chrome', 'firefox', 'safari', 'opera']}
    # 1. profile enum
    emu = read(os.path.join(ROOT, '..', 'emulate.rs'))
    profiles = []
    for m in re.finditer(r'(\w+)\s*=>\s*\("([^"]+)",\s*(\w+)::emulation\)', emu):
        profiles.append({'variant': m.group(1), 'name': m.group(2), 'module': m.group(3)})
    mod2prof = {p['module']: p for p in profiles}

    # 2. per-family data
    macro_exp = parse_header_macros(read(os.path.join(ROOT, 'macros.rs')))
    fam_mods, fam_tls, fam_h2, fam_headers = {}, {}, {}, {}
    for fam, d in famdirs.items():
        fam_mods[fam] = parse_profile_file(os.path.join(d, '..', fam + '.rs'))
        fam_tls[fam] = parse_family_tls(d)
        fam_h2[fam] = parse_family_http2(d)
        fam_headers[fam] = parse_header_initializers(
            read(os.path.join(d, 'header.rs')), macro_exp)
    okhttp_mods = parse_okhttp(os.path.join(ROOT, 'profile', 'okhttp.rs'))

    # fix opera http2: single () arm parsed as base already (macro_rules arm key '()'?)
    # parse_macro_rules keys '()' for opera since arm is `() => {...}` -> key '()' matches \d+? no.
    # Handle: if variants empty and base non-empty, wrap.
    if not fam_h2['opera']['variants'] and fam_h2['opera']['base']:
        pass  # base already holds the full chain; resolve_http2 handles varnum '()'

    def resolve_chain(fam, modname):
        """Follow build_emulation chains to base tls/http2 exprs.

        Only tls_expr/http2_expr are inherited from the chain base; the
        profile's own header_init and platform tuples always win (a chained
        profile like chrome_129 keeps its own UA/sec-ch-ua list and header
        initializer while reusing the base's TLS/HTTP2 emulation)."""
        own = fam_mods[fam][modname]
        seen = []
        cur = modname
        while True:
            info = fam_mods[fam][cur]
            seen.append(cur)
            if info['build_emulation']:
                nxt = info['build_emulation'].strip()
                assert nxt.endswith('::build_emulation'), nxt
                cur = nxt.split('::')[0]
                assert cur not in seen
            else:
                return {'tls_expr': info['tls_expr'],
                        'http2_expr': info['http2_expr'],
                        'header_init': own['header_init'],
                        'platforms': own['platforms']}

    out_profiles, tls_sigs, h2_sigs, ua_pool = [], {}, {}, set()
    for p in profiles:
        if p['module'] in okhttp_mods:
            continue
        fam = next(f for f in famdirs if p['module'] in fam_mods[f])
        base = resolve_chain(fam, p['module'])
        tls = resolve_tls(fam_tls[fam], base['tls_expr'], None)
        h2 = resolve_http2(fam_h2[fam], base['http2_expr'])
        # attach family + from-body hash to tls signature
        from_hash = str(len(fam_tls[fam]['from_calls']))
        tls_sig = json.dumps({'family': fam, 'from_calls': fam_tls[fam]['from_calls'],
                              'fields': tls['fields']}, sort_keys=True, default=str)
        h2_sig = json.dumps({'family': fam, 'settings': h2['settings'],
                             'pseudo': h2['pseudo_order'], 'sorder': h2['settings_order'],
                             'sdep': h2['stream_dependency'], 'pri': h2['priorities']},
                            sort_keys=True, default=str)
        tls_sigs.setdefault(tls_sig, []).append(p['name'])
        h2_sigs.setdefault(h2_sig, []).append(p['name'])
        tpl = fam_headers[fam][base['header_init']]
        plats = []
        for plat, sec, ua in base['platforms']:
            ua_pool.add(ua)
            plats.append({'platform': plat, 'sec_ch_ua': sec, 'ua': ua,
                          'headers': materialize_headers(tpl, sec, ua, plat)})
        out_profiles.append({
            'name': p['name'], 'variant': p['variant'], 'family': fam,
            'module': p['module'], 'chain': base['tls_expr'],
            'tls': tls, 'http2': h2, 'header_template': base['header_init'],
            'platforms': plats,
        })

    # okhttp
    for modname, info in okhttp_mods.items():
        p = mod2prof[modname]
        pname = p['name']
        ua_pool.add(info['ua'])
        out_profiles.append({
            'name': pname, 'variant': None, 'family': 'okhttp', 'module': modname,
            'tls': {'variant': 'inline', 'variant_args': [],
                    'fields': {'cipher_list': info['cipher_list']}},
            'http2': {'variant': 'inline', 'settings': {
                'initial_window_size': 16777216,
                'initial_connection_window_size': 16777216},
                'pseudo_order': ['Method', 'Path', 'Authority', 'Scheme'],
                'settings_order': ['HeaderTableSize', 'EnablePush', 'MaxConcurrentStreams',
                                   'InitialWindowSize', 'MaxFrameSize', 'MaxHeaderListSize',
                                   'EnableConnectProtocol', 'NoRfc7540Priorities'],
                'stream_dependency': None, 'priorities': None},
            'header_template': 'inline',
            'platforms': [{'platform': None, 'sec_ch_ua': None, 'ua': info['ua'],
                           'headers': [['accept', '*/*'],
                                       ['accept-language', 'en-US,en;q=0.9'],
                                       ['user-agent', info['ua']],
                                       ['accept-encoding', 'gzip']]}],
        })
        tls_sig = json.dumps({'family': 'okhttp', 'fields': {'cipher_list': info['cipher_list']}},
                             sort_keys=True)
        tls_sigs.setdefault(tls_sig, []).append(pname)
        h2_sig = json.dumps({'family': 'okhttp', 'settings': {'initial_window_size': 16777216,
                             'initial_connection_window_size': 16777216},
                             'pseudo': ['Method', 'Path', 'Authority', 'Scheme']}, sort_keys=True)
        h2_sigs.setdefault(h2_sig, []).append(pname)

    baseline = {
        'source': {'repo': '0x676e67/wreq-util', 'commit': 'e3922a2', 'version': '0.2.0'},
        'profile_count': len(profiles),
        'distinct_tls_configs': len(tls_sigs),
        'distinct_http2_configs': len(h2_sigs),
        'tls_config_groups': {str(i): v for i, v in enumerate(tls_sigs.values())},
        'http2_config_groups': {str(i): v for i, v in enumerate(h2_sigs.values())},
        'ua_pool_size': len(ua_pool),
        'ua_pool': sorted(ua_pool),
        'profiles': sorted(out_profiles, key=lambda x: x['name']),
    }
    with open(os.path.expanduser('~/workspace/bowser/pr5-fidelity/wreq_baseline.json'), 'w') as f:
        json.dump(baseline, f, indent=1, default=str)
    print(f"profiles={len(profiles)} tls_configs={len(tls_sigs)} http2_configs={len(h2_sigs)} ua_pool={len(ua_pool)}")
    # family max versions
    famnames = {}
    for p in profiles:
        if p['module'] in okhttp_mods:
            famnames.setdefault('okhttp', []).append(p['name'])
            continue
        fam = next(f for f in famdirs if p['module'] in fam_mods[f])
        famnames.setdefault(fam, []).append(p['name'])
    for fam, names in famnames.items():
        print(fam, len(names), 'max=', sorted(names)[-3:])

if __name__ == '__main__':
    main()
