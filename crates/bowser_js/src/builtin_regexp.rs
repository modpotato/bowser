//! `RegExp` builtins: the constructor, `RegExp.prototype` methods, and the
//! well-known-symbol methods (`Symbol.replace` / `Symbol.split` /
//! `Symbol.match` / `Symbol.matchAll`).
//!
//! The matching itself lives behind [`crate::regex_engine`]; this module
//! only implements the builtins layer (argument coercion, `lastIndex`
//! bookkeeping, result-array construction, `GetSubstitution`).
//!
//! # Key decisions
//!
//! * **State as ordinary properties.** A regexp's `source`, `flags`, and
//!   `lastIndex` are plain data properties on an
//!   [`ObjectKind::RegExp`](crate::value::ObjectKind) object. `lastIndex`
//!   is writable; the rest are non-writable, non-enumerable,
//!   non-configurable.
//! * **Bounded compile cache.** Compiled patterns are cached per
//!   `(source, flags)` in a thread-local, capacity-bounded table (128
//!   entries, LRU eviction), so repeated `exec`/`test` calls — and the
//!   `replace`/`split`/`match` loops — reuse one compiled program instead
//!   of recompiling per call. The key is `(Rc<str>, u8)`: the source text
//!   plus a flag bitmask, so lookup allocates nothing; a one-entry fast
//!   path serves a repeated pattern with a single `memcmp`. The table holds
//!   no GC handles (owned strings plus the compiled program), so it needs
//!   no tracing and cannot dangle; the capacity bound keeps the original
//!   "cannot leak" property. Compile failures are never cached.
//! * **Offsets are UTF-16 code units** everywhere the engine is involved,
//!   matching JS string semantics. All slicing goes through the `units`
//!   helpers below, so a match boundary inside a surrogate pair degrades
//!   to U+FFFD instead of panicking.
//!
//! # Deliberate v1 deviations
//!
//! * The well-known-symbol methods (`Symbol.replace`, `Symbol.split`,
//!   `Symbol.match`, `Symbol.matchAll`, `Symbol.search`) are installed on
//!   `RegExp.prototype` under real symbol keys, which the abstract
//!   operations consult (a computed `o[Symbol.replace]` key resolves to
//!   the same property).
//! * `Symbol.matchAll` returns an `Array` of match arrays, not an iterator
//!   (no iterator protocol in v1).
//! * `NativeFn` cannot see `new.target`, so `RegExp()` and
//!   `new RegExp()` behave identically — except the spec 22.2.3.1
//!   `RegExp(regexp)` short-circuit: the call form with a regexp pattern
//!   and no flags returns the pattern itself when its `"constructor"` is
//!   the installed `RegExp` constructor (keyed off the `REGEXP_CTOR`
//!   install-time record, since the native cannot observe its callee).
//! * Replacer / callback functions go through the shared
//!   [`crate::builtin_object_array::call_function`] choke point: native
//!   functions run directly, bytecode functions re-enter the VM.
//! * `source` escaping covers `/` only (plus empty → `"(?:)"`); lone
//!   surrogates cannot occur because engine strings are Rust `String`s.
//! * `ToString` on a non-regexp object without a custom `toString`
//!   stays `"[object Object]"` (the local `to_string_rust`); spec
//!   `RegExp.prototype.toString` (`"/" + source + "/" + flags`, requiring
//!   an Object receiver per step 2) is installed on `%RegExp.prototype%`.
//! * Backtracking-budget exhaustion ([`RegexError::BacktrackLimit`]) throws
//!   `RangeError` instead of hanging or silently mis-matching.

use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use crate::builtin_string_number::{define_native_key, push_code_point, with_temp_root};
use crate::gc::{Gc, Heap, PropKey, WellKnown};
use crate::regex_engine::{Capture, CompiledRegex, RegexError};
use crate::value::{NativeError, NativeFn, Object, ObjectKind, Property, Value};
use crate::wtf8::{Wtf8, Wtf8Buf};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// `args[i]` or `undefined`.
fn arg(args: &[Value], i: usize) -> Value {
    args.get(i).copied().unwrap_or(Value::Undefined)
}

/// Cheapest numeric value for `n`: [`Value::Int`] when integral and fitting
/// `i32` (preserving `-0` as `Float`), else [`Value::Float`].
fn js_num(n: f64) -> Value {
    if n == 0.0 {
        return if n.is_sign_negative() {
            Value::Float(n)
        } else {
            Value::Int(0)
        };
    }
    if n.fract() == 0.0 && n > -2147483648.0 && n < 2147483648.0 {
        Value::Int(n as i32)
    } else {
        Value::Float(n)
    }
}

fn type_error(msg: &str) -> NativeError {
    NativeError::TypeError(msg.to_string())
}

fn syntax_error(msg: String) -> NativeError {
    NativeError::SyntaxError(msg)
}

/// Map an engine error to a JS error: syntax problems are `SyntaxError`;
/// backtracking-budget exhaustion is `RangeError` (a resource limit, in
/// the same family as stack overflow).
fn regex_native_error(e: RegexError) -> NativeError {
    match e {
        RegexError::ParseError(msg) => syntax_error(format!("invalid regular expression: {msg}")),
        RegexError::BacktrackLimit => {
            NativeError::RangeError("regex backtracking limit exceeded".to_string())
        }
    }
}

/// The `StrWhiteSpace` code points (spec 12.2).
fn is_js_whitespace(c: char) -> bool {
    matches!(
        c,
        '\u{0009}'
            | '\u{000A}'
            | '\u{000B}'
            | '\u{000C}'
            | '\u{000D}'
            | '\u{0020}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'
            ..='\u{200A}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202F}'
                | '\u{205F}'
                | '\u{3000}'
                | '\u{FEFF}'
    )
}

fn prefixed_int(digits: &str, radix: u32, negative: bool) -> f64 {
    if digits.is_empty() {
        return f64::NAN;
    }
    let mut v = 0.0f64;
    for c in digits.chars() {
        match c.to_digit(radix) {
            Some(d) => v = v * f64::from(radix) + f64::from(d),
            None => return f64::NAN,
        }
    }
    if negative { -v } else { v }
}

/// Strict decimal literal covering the whole input.
fn is_strict_decimal(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = 0;
    let n = b.len();
    let int_start = i;
    while i < n && b[i].is_ascii_digit() {
        i += 1;
    }
    let has_int = i > int_start;
    let mut has_frac = false;
    if i < n && b[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < n && b[i].is_ascii_digit() {
            i += 1;
        }
        has_frac = i > frac_start;
    }
    if !has_int && !has_frac {
        return false;
    }
    if i < n && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        if i < n && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < n && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            return false;
        }
    }
    i == n
}

/// `ToNumber` applied to a string.
fn parse_js_number(s: &str) -> f64 {
    let t = s.trim_matches(is_js_whitespace);
    if t.is_empty() {
        return 0.0;
    }
    let bytes = t.as_bytes();
    let (negative, rest) = match bytes[0] {
        b'-' => (true, &t[1..]),
        b'+' => (false, &t[1..]),
        _ => (false, t),
    };
    if rest == "Infinity" {
        return if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    if let Some(hex) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
        return prefixed_int(hex, 16, negative);
    }
    if !is_strict_decimal(rest) {
        return f64::NAN;
    }
    let mut v: f64 = rest.parse().unwrap_or(f64::NAN);
    if negative {
        v = -v;
    }
    v
}

/// Shortest-round-trip decimal digits of `|x|` plus the exponent `n` with
/// `value = 0.d1..dk * 10^n`. Relies on Rust's `Display` for `f64`, which
/// is shortest-round-trip and never uses exponent notation.
fn decimal_digits(x: f64) -> (String, i32) {
    debug_assert!(x.is_finite() && x != 0.0);
    let plain = format!("{}", x.abs());
    let mut digits = String::new();
    let mut n: i32;
    if let Some(dot) = plain.find('.') {
        let intpart = &plain[..dot];
        let stripped = intpart.trim_start_matches('0');
        n = stripped.len() as i32;
        digits.push_str(stripped);
        digits.push_str(&plain[dot + 1..]);
    } else {
        n = plain.len() as i32;
        digits.push_str(plain.trim_start_matches('0'));
    }
    let leading = digits.len() - digits.trim_start_matches('0').len();
    n -= leading as i32;
    digits = digits.trim_start_matches('0').to_string();
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    if digits.is_empty() {
        digits.push('0');
    }
    (digits, n)
}

/// Spec `Number::toString(x)` for radix 10.
fn js_number_to_string(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x == 0.0 {
        return "0".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        };
    }
    let (digits, n) = decimal_digits(x);
    let k = digits.len() as i32;
    let body = if k <= n && n <= 21 {
        format!("{digits}{}", "0".repeat((n - k) as usize))
    } else if 0 < n && n <= 21 {
        format!("{}.{}", &digits[..n as usize], &digits[n as usize..])
    } else if -6 < n && n <= 0 {
        format!("0.{}{digits}", "0".repeat((-n) as usize))
    } else {
        let exp = n - 1;
        let mantissa = if k == 1 {
            digits
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        format!("{mantissa}e{exp:+}")
    };
    if x < 0.0 { format!("-{body}") } else { body }
}

/// Render sign-magnitude base-2^32 little-endian digits as decimal.
fn bigint_to_string(negative: bool, digits: &[u32]) -> String {
    let mut m: Vec<u32> = digits.to_vec();
    while m.last() == Some(&0) {
        m.pop();
    }
    if m.is_empty() {
        return "0".to_string();
    }
    let mut parts: Vec<u32> = Vec::new();
    while !m.is_empty() {
        let mut rem: u64 = 0;
        for w in m.iter_mut().rev() {
            let cur = (rem << 32) | u64::from(*w);
            *w = (cur / 1_000_000_000) as u32;
            rem = cur % 1_000_000_000;
        }
        while m.last() == Some(&0) {
            m.pop();
        }
        parts.push(rem as u32);
    }
    let mut s = String::new();
    if negative {
        s.push('-');
    }
    let mut it = parts.iter().rev();
    s.push_str(&it.next().unwrap_or(&0).to_string());
    for p in it {
        s.push_str(&format!("{p:09}"));
    }
    s
}

/// JS `ToString` as an owned Rust string. v1: objects are
/// `"[object Object]"` (no user-defined `toString` dispatch yet).
fn to_string_rust(v: Value) -> Result<Wtf8Buf, NativeError> {
    match v {
        Value::Undefined => Ok(Wtf8Buf::from("undefined")),
        Value::Null => Ok(Wtf8Buf::from("null")),
        Value::Bool(b) => Ok(Wtf8Buf::from(if b { "true" } else { "false" })),
        Value::Int(i) => Ok(Wtf8Buf::from(&i.to_string())),
        Value::Float(f) => Ok(Wtf8Buf::from(&js_number_to_string(f))),
        Value::Str(g) => Ok(g.as_wtf8().to_owned()),
        Value::BigInt(g) => Ok(Wtf8Buf::from(&bigint_to_string(g.negative, &g.digits))),
        Value::Symbol(_) => Err(type_error("Cannot convert a Symbol value to a string")),
        // v1 gap: no ToPrimitive dispatch.
        Value::Object(_) => Ok(Wtf8Buf::from("[object Object]")),
    }
}

/// JS `ToNumber`. v1: objects are `NaN` (no `ToPrimitive` dispatch).
fn to_number(v: Value) -> Result<f64, NativeError> {
    match v {
        Value::Undefined => Ok(f64::NAN),
        Value::Null => Ok(0.0),
        Value::Bool(b) => Ok(if b { 1.0 } else { 0.0 }),
        Value::Int(i) => Ok(f64::from(i)),
        Value::Float(f) => Ok(f),
        Value::Str(g) => Ok(g
            .as_wtf8()
            .as_utf8()
            .map(parse_js_number)
            .unwrap_or(f64::NAN)),
        Value::BigInt(_) => Err(type_error("Cannot convert a BigInt value to a number")),
        Value::Symbol(_) => Err(type_error("Cannot convert a Symbol value to a number")),
        Value::Object(_) => Ok(f64::NAN),
    }
}

/// JS `ToLength`: clamped to `[0, 2^53 - 1]`.
fn to_length(v: Value) -> Result<u64, NativeError> {
    const MAX: f64 = 9007199254740991.0;
    let n = to_number(v)?;
    if n.is_nan() || n <= 0.0 {
        Ok(0)
    } else if n >= MAX {
        Ok(MAX as u64)
    } else {
        Ok(n as u64)
    }
}

/// JS `ToUint32` (modulo 2^32, infinities/NaN -> 0).
fn to_uint32(v: Value) -> Result<u32, NativeError> {
    let n = to_number(v)?;
    if !n.is_finite() {
        return Ok(0);
    }
    let m = n % 4294967296.0;
    let m = if m < 0.0 { m + 4294967296.0 } else { m };
    Ok(m as u32)
}

// ---------------------------------------------------------------------------
// UTF-16 string model
// ---------------------------------------------------------------------------
///
/// JS strings are indexed in UTF-16 code units. Every boundary that touches
/// the engine works in unit offsets; materializing a JS string goes through
/// [`units_to_string`], which maps lone surrogates to U+FFFD instead of
/// panicking (a Rust `String` cannot hold them).
fn is_lead_surrogate(u: u16) -> bool {
    (0xD800..0xDC00).contains(&u)
}

fn is_trail_surrogate(u: u16) -> bool {
    (0xDC00..0xE000).contains(&u)
}

/// Build a JS string from code units; lone surrogates are preserved as
/// WTF-8 (the old code lossy-converted them to U+FFFD).
fn units_to_string(units: &[u16]) -> Wtf8Buf {
    Wtf8Buf::from_units(units)
}

/// Spec `AdvanceStringIndex(S, index, unicode)`. Deliberately *not* capped
/// at the length: callers rely on `index + 1 > len` to terminate
/// empty-match loops (see [`exec_at`]).
fn advance_string_index(units: &[u16], index: usize, unicode: bool) -> usize {
    if !unicode {
        return index + 1;
    }
    if index + 1 < units.len()
        && is_lead_surrogate(units[index])
        && is_trail_surrogate(units[index + 1])
    {
        index + 2
    } else {
        index + 1
    }
}

// ---------------------------------------------------------------------------
// Flags
// ---------------------------------------------------------------------------

/// Parsed regexp flags.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Flags {
    d: bool,
    g: bool,
    i: bool,
    m: bool,
    s: bool,
    u: bool,
    v: bool,
    y: bool,
}

/// Validate a flags string: only `dgimsuvy`, no duplicates, and `u`/`v`
/// never together. Errors are plain strings so callers can wrap them as
/// `SyntaxError` (builtins) or [`RegexError::ParseError`] (heapless
/// helpers) as fits their boundary.
/// Parse a flags string. Operates on code units: any non-ASCII unit
/// (including lone surrogates) is an invalid flag, so ill-formed input
/// yields a `SyntaxError` at the call sites instead of being lossy.
fn parse_flags(flags: &Wtf8) -> Result<Flags, String> {
    let mut f = Flags::default();
    for u in flags.code_units() {
        let slot = match u {
            0x64 => &mut f.d, // d
            0x67 => &mut f.g, // g
            0x69 => &mut f.i, // i
            0x6d => &mut f.m, // m
            0x73 => &mut f.s, // s
            0x75 => &mut f.u, // u
            0x76 => &mut f.v, // v
            0x79 => &mut f.y, // y
            _ => return Err(format!("invalid flag '{u}'")),
        };
        if *slot {
            return Err(format!("duplicate flag '{u}'"));
        }
        *slot = true;
    }
    if f.u && f.v {
        return Err("the 'u' and 'v' flags cannot be combined".to_string());
    }
    Ok(f)
}

/// Canonical flag order: `"dgimsuvy"`.
fn flags_string(f: &Flags) -> String {
    let mut s = String::new();
    for (on, c) in [
        (f.d, 'd'),
        (f.g, 'g'),
        (f.i, 'i'),
        (f.m, 'm'),
        (f.s, 's'),
        (f.u, 'u'),
        (f.v, 'v'),
        (f.y, 'y'),
    ] {
        if on {
            s.push(c);
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Bounded compile cache
// ---------------------------------------------------------------------------
//
// `CompiledRegex::compile` is the expensive step; matching itself is cheap.
// Compiling on every `exec`/`test` paid that cost per call, so compiled
// patterns are cached per `(source, flags)` instead.
//
// Why a value-keyed table rather than per-object storage: the compiled
// program is a pure function of `(source, flags)` — matching behavior does
// not depend on which regexp object runs it (`lastIndex`, stickiness, and
// unicode handling all live in the builtins layer above) — so one entry
// serves every regexp object sharing a pattern, including the heapless
// one-shot helpers below. The table holds no GC handles at all: keys are
// the source's WTF-8 bytes (lone surrogates preserved exactly) plus a flag
// bitmask, values are `Rc<CompiledRegex>`
// (plain `Vec`s/`String`s plus an error `Cell`). The GC never sees it, so
// no `Trace` impl is needed and there are no object handles that could
// outlive their objects. The table is bounded (`COMPILE_CACHE_CAP`
// entries, LRU eviction) so a program generating fresh patterns cannot grow
// it without limit; that bound is what keeps the original "never cache and
// you cannot leak" property. Compile failures are never cached.
//
// `CompiledRegex` is `!Sync` (interior error `Cell`), and the engine is
// single-threaded, so the table is `thread_local!` rather than a shared
// static: no lock on the match path, no `Sync` requirement, and the error
// cell is never aliased across threads. `exec` resets the error cell on
// entry, so sequential reuse cannot leak error state between calls.

/// Maximum compiled patterns retained per thread.
const COMPILE_CACHE_CAP: usize = 128;

/// `Flags` as a bitmask (`d g i m s u v y` → bits 0..=7). Canonical like
/// [`flags_string`]: equal flag sets map to equal masks, so the cache key
/// needs no string allocation on the match path.
fn flags_bits(f: &Flags) -> u8 {
    let mut b = 0u8;
    if f.d {
        b |= 0x01;
    }
    if f.g {
        b |= 0x02;
    }
    if f.i {
        b |= 0x04;
    }
    if f.m {
        b |= 0x08;
    }
    if f.s {
        b |= 0x10;
    }
    if f.u {
        b |= 0x20;
    }
    if f.v {
        b |= 0x40;
    }
    if f.y {
        b |= 0x80;
    }
    b
}

/// Bitmask for a flags string (the [`compile_cached`] entry point).
/// Unknown characters map to no bits; the original string is still passed
/// to the compiler on a miss, so invalid flags keep erroring there.
fn flags_str_bits(flags: &str) -> u8 {
    let mut b = 0u8;
    for c in flags.bytes() {
        b |= match c {
            b'd' => 0x01,
            b'g' => 0x02,
            b'i' => 0x04,
            b'm' => 0x08,
            b's' => 0x10,
            b'u' => 0x20,
            b'v' => 0x40,
            b'y' => 0x80,
            _ => 0,
        };
    }
    b
}

/// Canonical flag string for a bitmask, for the compile-on-miss path.
fn flags_string_from_bits(bits: u8) -> String {
    let mut s = String::with_capacity(8);
    for (mask, c) in [
        (0x01, 'd'),
        (0x02, 'g'),
        (0x04, 'i'),
        (0x08, 'm'),
        (0x10, 's'),
        (0x20, 'u'),
        (0x40, 'v'),
        (0x80, 'y'),
    ] {
        if bits & mask != 0 {
            s.push(c);
        }
    }
    s
}

struct CacheEntry {
    re: Rc<CompiledRegex>,
    /// Last-use tick; eviction removes the smallest.
    tick: u64,
}

struct CompileCache {
    /// `(source WTF-8 bytes, flag bits)`. Byte keys keep lone-surrogate
    /// patterns distinct from their U+FFFD lookalikes.
    map: HashMap<(Vec<u8>, u8), CacheEntry>,
    tick: u64,
    /// One-entry fast path: repeated `exec`/`test`/`replace` on one pattern
    /// hit here with a single `memcmp` — no hashing, no allocation.
    last: Option<(Vec<u8>, u8, Rc<CompiledRegex>)>,
}

impl CompileCache {
    fn get_or_compile(
        &mut self,
        source: &Wtf8,
        bits: u8,
        flags_for_compile: Option<&str>,
    ) -> Result<Rc<CompiledRegex>, RegexError> {
        if let Some((ref s, b, ref re)) = self.last
            && b == bits
            && s == source.as_bytes()
        {
            // Pure fast path: no hashing, no allocation, no map touch.
            // Skipping the recency refresh is safe — the `Rc` here is
            // authoritative for this key even if the map entry is later
            // evicted; a later miss simply recompiles and re-inserts.
            return Ok(Rc::clone(re));
        }
        let key = (source.as_bytes().to_vec(), bits);
        if let Some(e) = self.map.get_mut(&key) {
            self.tick += 1;
            e.tick = self.tick;
            let re = Rc::clone(&e.re);
            self.last = Some((key.0.clone(), bits, Rc::clone(&re)));
            return Ok(re);
        }
        // Failures are not cached: `?` returns before insertion.
        let flags = flags_for_compile
            .map(str::to_owned)
            .unwrap_or_else(|| flags_string_from_bits(bits));
        let compiled = Rc::new(CompiledRegex::compile(source, &flags)?);
        if self.map.len() >= COMPILE_CACHE_CAP
            && let Some(oldest) = self
                .map
                .iter()
                .min_by_key(|(_, e)| e.tick)
                .map(|(k, _)| k.clone())
        {
            self.map.remove(&oldest);
        }
        self.tick += 1;
        self.map.insert(
            key.clone(),
            CacheEntry {
                re: Rc::clone(&compiled),
                tick: self.tick,
            },
        );
        self.last = Some((key.0, bits, Rc::clone(&compiled)));
        Ok(compiled)
    }
}

thread_local! {
    static COMPILE_CACHE: RefCell<CompileCache> = RefCell::new(CompileCache {
        map: HashMap::new(),
        tick: 0,
        last: None,
    });

    /// `%RegExp.prototype%` per heap, recorded by [`install_inner`].
    /// `regexp_construct` needs it when `NewTarget` is undefined (the call
    /// form — every regexp literal arrives via `op_new_regex` invoking the
    /// constructor as a plain function): spec 22.2.4.1 substitutes the
    /// active function object, i.e. the `RegExp` constructor, whose
    /// `"prototype"` is exactly this object. The native cannot observe its
    /// own callee (see `active_new_target`), so the install-time record is
    /// the stand-in. Keyed by heap id: `Gc` handles are heap-local and must
    /// never cross heaps.
    static REGEXP_PROTO: RefCell<Option<(u64, Gc<Object>)>> = const { RefCell::new(None) };

    /// The `RegExp` constructor per heap, recorded by [`install_inner`]:
    /// the call form's "active function object" for the spec 22.2.3.1
    /// `RegExp(regexp)` short-circuit (`SameValue(newTarget,
    /// Get(pattern, "constructor"))`). Same heap-id keying as
    /// [`REGEXP_PROTO`].
    static REGEXP_CTOR: RefCell<Option<(u64, Gc<Object>)>> = const { RefCell::new(None) };
}

/// The compiled pattern for `(source, flags)`: compiles once, then reuses
/// the cached program. `flags` should be in canonical [`flags_string`]
/// order so equal flag sets share one entry with the [`Flags`] bitmask
/// path below.
fn compile_cached(source: &Wtf8, flags: &str) -> Result<Rc<CompiledRegex>, RegexError> {
    COMPILE_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .get_or_compile(source, flags_str_bits(flags), Some(flags))
    })
}

/// Hot-path entry for already-validated [`Flags`]: the bitmask key avoids
/// the flags-string allocation that [`compile_cached`] pays per call.
fn compile_cached_flags(source: &Wtf8, flags: &Flags) -> Result<Rc<CompiledRegex>, RegexError> {
    COMPILE_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .get_or_compile(source, flags_bits(flags), None)
    })
}

/// Entry count of the compile cache. Test-only introspection.
#[cfg(test)]
fn compile_cache_len() -> usize {
    COMPILE_CACHE.with(|cache| cache.borrow().map.len())
}

// ---------------------------------------------------------------------------
// RegExp objects
// ---------------------------------------------------------------------------

/// The receiver as a regexp object, or a `TypeError` naming the method.
fn this_regexp(this: Value, method: &str) -> Result<Gc<Object>, NativeError> {
    match this {
        Value::Object(o) if o.kind == ObjectKind::RegExp => Ok(o),
        _ => Err(type_error(&format!(
            "RegExp.prototype.{method} called on incompatible receiver"
        ))),
    }
}

/// Some value if it is a regexp object.
fn as_regexp(v: Value) -> Option<Gc<Object>> {
    match v {
        Value::Object(o) if o.kind == ObjectKind::RegExp => Some(o),
        _ => None,
    }
}

/// Read an own string property, defaulting to `""` when absent or not a
/// string (defensive: `install` always sets these).
fn own_str_prop(heap: &Heap, obj: Gc<Object>, name: &Wtf8) -> Wtf8Buf {
    match heap.get_property(obj, name).map(|p| p.value) {
        Some(Value::Str(g)) => g.as_wtf8().to_owned(),
        _ => Wtf8Buf::new(),
    }
}

/// `[[Set]]` on a data property, preserving its attributes. Non-writable
/// properties are silently ignored (sloppy-mode semantics).
fn set_data_prop(
    heap: &mut Heap,
    obj: Gc<Object>,
    name: &str,
    value: Value,
) -> Result<(), NativeError> {
    match heap.get_property(obj, Wtf8::from_str(name)) {
        Some(p) => {
            if p.writable {
                heap.define_property_str(obj, name, Property { value, ..p })?;
            }
            Ok(())
        }
        None => {
            heap.define_property_str(obj, name, Property::data(value))?;
            Ok(())
        }
    }
}

/// The `lastIndex` value (defaults to `0`).
fn get_last_index_val(heap: &Heap, rx: Gc<Object>) -> Value {
    heap.get_property(rx, Wtf8::from_str("lastIndex"))
        .map(|p| p.value)
        .unwrap_or(Value::Int(0))
}

/// Write `lastIndex`, throwing `TypeError` when the property is
/// non-writable. This is spec `Set(rx, "lastIndex", n, true)` (throw on
/// failure), used by `RegExpBuiltinExec` and by the `@@replace` /
/// `@@match` / `@@matchAll` / `@@search` initializations. A refused write
/// must surface: silently ignoring it leaves `lastIndex` stuck, so a
/// global scan re-matches at the same position forever (and the
/// accumulated duplicate matches would slice backwards in `@@replace`).
fn set_last_index(heap: &mut Heap, rx: Gc<Object>, n: f64) -> Result<(), NativeError> {
    match heap.get_property(rx, Wtf8::from_str("lastIndex")) {
        Some(p) if !p.writable => Err(type_error(
            "Cannot assign to read only property 'lastIndex' of object",
        )),
        _ => set_data_prop(heap, rx, "lastIndex", js_num(n)),
    }
}

/// Number `SameValue` (spec 6.1.6.1): unlike `==`, `-0` and `+0` are
/// distinct, while `NaN` is equal to itself.
fn same_value_f64(a: f64, b: f64) -> bool {
    (a == b && a.is_sign_negative() == b.is_sign_negative()) || (a.is_nan() && b.is_nan())
}

/// `SameValue` for the numeric `Value` shapes a `lastIndex` can hold
/// (`Int`/`Float`); anything else falls back to `==`.
fn last_index_same_value(a: Value, b: Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => same_value_f64(x, y),
        (Value::Int(x), Value::Float(y)) => same_value_f64(x as f64, y),
        (Value::Float(x), Value::Int(y)) => same_value_f64(x, y as f64),
        _ => a == b,
    }
}

/// `lastIndex` as a code-unit offset. The engine layer always writes
/// non-negative `Int`s, so the `ToLength` float round-trip is skipped on
/// that fast path (this runs once per match in `replace`/`match` loops).
fn last_index_start(heap: &Heap, rx: Gc<Object>) -> Result<usize, NativeError> {
    match get_last_index_val(heap, rx) {
        Value::Int(i) if i >= 0 => Ok(i as usize),
        v => Ok(to_length(v)? as usize),
    }
}

/// The `source` text for a fresh pattern: `/` is escaped so the source can
/// be re-embedded in a literal; empty becomes `"(?:)"` per spec.
fn escape_source(pattern: &Wtf8) -> Wtf8Buf {
    if pattern.is_empty() {
        return Wtf8Buf::from("(?:)");
    }
    let mut out = Wtf8Buf::with_capacity(pattern.len());
    for c in pattern.chars_wtf8() {
        if c == '/' as u32 {
            out.push_char('\\');
        }
        push_code_point(&mut out, c);
    }
    out
}

/// Allocate a regexp object. `source` must already be in final (escaped)
/// form; the pattern is compiled eagerly so invalid patterns throw
/// `SyntaxError` at construction, per spec.
///
/// The `source` *property* preserves lone surrogates exactly, and so does
/// the compiled pattern: the engine takes `&Wtf8` and parses the pattern
/// as UTF-16 code units outside unicode mode, so a lone-surrogate pattern
/// matches a lone-surrogate subject instead of compiling lossily as
/// U+FFFD (the old deviation; fixed by compiling from WTF-8).
fn make_regexp(heap: &mut Heap, source: &Wtf8, flags: &Flags) -> Result<Gc<Object>, NativeError> {
    // Eager compile so invalid patterns throw `SyntaxError` at
    // construction, per spec; the program is cached for the match path.
    compile_cached(source, &flags_string(flags)).map_err(regex_native_error)?;
    let rx = heap.alloc_object(Object::new(ObjectKind::RegExp))?;
    let constant = |value: Value| Property {
        value,
        writable: false,
        enumerable: false,
        configurable: false,
    };
    let source_v = Value::Str(heap.alloc_str_wtf8(source)?);
    heap.define_property_str(rx, "source", constant(source_v))?;
    let flags_v = Value::Str(heap.alloc_str(&flags_string(flags))?);
    heap.define_property_str(rx, "flags", constant(flags_v))?;
    for (name, on) in [
        ("global", flags.g),
        ("ignoreCase", flags.i),
        ("multiline", flags.m),
        ("dotAll", flags.s),
        ("unicode", flags.u),
        ("sticky", flags.y),
        ("hasIndices", flags.d),
    ] {
        heap.define_property_str(rx, name, constant(Value::Bool(on)))?;
    }
    heap.define_property_str(
        rx,
        "lastIndex",
        Property {
            value: Value::Int(0),
            writable: true,
            enumerable: false,
            configurable: false,
        },
    )?;
    Ok(rx)
}

/// `RegExp(rx)` with no flags: a copy sharing source and flags.
/// `lastIndex` starts at `0` (spec `RegExpInitialize`); the
/// same-constructor identity short-circuit is a v1 gap (always copies).
fn copy_regexp(heap: &mut Heap, rx: Gc<Object>) -> Result<Gc<Object>, NativeError> {
    let source = own_str_prop(heap, rx, Wtf8::from_str("source"));
    let flags =
        parse_flags(&own_str_prop(heap, rx, Wtf8::from_str("flags"))).map_err(syntax_error)?;
    let new_rx = make_regexp(heap, &source, &flags)?;
    heap.set_prototype(new_rx, heap.get(rx).prototype);
    Ok(new_rx)
}

// ---------------------------------------------------------------------------
// GetSubstitution (spec 22.1.3.17.1)
// ---------------------------------------------------------------------------

/// Capture `idx` (0 = whole match) rendered as a JS string, memoized in
/// `cache` (`None` slot = not computed yet). Non-participating groups —
/// and out-of-range indices — are `None`. Templates usually touch few
/// groups, so materializing on demand beats converting them all up front.
fn capture_text(
    units: &[u16],
    mt: &Match,
    idx: usize,
    cache: &mut [Option<Option<Wtf8Buf>>],
) -> Option<Wtf8Buf> {
    let slot = cache.get_mut(idx)?;
    if slot.is_none() {
        let len = units.len();
        let t = mt
            .captures
            .get(idx)
            .copied()
            .flatten()
            .map(|cp| units_to_string(&units[cp.start.min(len)..cp.end.min(len)]));
        *slot = Some(t);
    }
    slot.clone().expect("slot just initialized")
}

/// Expand a replacement template: `$$`, `$&`, `` $` ``, `$'`, `$1`..`$99`,
/// `$<name>`. A `$` followed by anything else is literal.
///
/// Scans the template as bytes (`$` and all pattern sigils are ASCII) and
/// copies literal chunks verbatim instead of pushing char-by-char; group
/// text is converted on first use and the named-group index is built only
/// when the template actually contains `$<name>`.
fn apply_substitution(
    template: &Wtf8,
    input_units: &[u16],
    m_start: usize,
    m_end: usize,
    mt: &Match,
) -> Wtf8Buf {
    let bytes = template.as_bytes();
    // Fast path: no `$` means the template is literal.
    if !bytes.contains(&b'$') {
        return template.to_owned();
    }
    // `mt.captures[0]` is the whole match; `m` counts the groups.
    let m = mt.captures.len().saturating_sub(1);
    let mut cache: Vec<Option<Option<Wtf8Buf>>> = vec![None; m + 1];
    let mut names: Option<HashMap<String, usize>> = None;
    let mut out = Wtf8Buf::with_capacity(template.len());
    let mut i = 0usize;
    let mut chunk_start = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'$' {
            i += 1;
            continue;
        }
        // Flush the literal chunk before this `$`: `$` and all sigils are
        // ASCII, so chunk boundaries are always valid WTF-8 boundaries.
        out.push_wtf8_bytes(&bytes[chunk_start..i]);
        i += 1; // consume `$`
        match bytes.get(i).copied() {
            // Trailing `$` is literal.
            None => {
                out.push_char('$');
                chunk_start = i;
            }
            // `$$` -> `$`.
            Some(b'$') => {
                out.push_char('$');
                i += 1;
                chunk_start = i;
            }
            // `$&` -> the whole match.
            Some(b'&') => {
                if let Some(t) = capture_text(input_units, mt, 0, &mut cache) {
                    out.push_wtf8(t.as_wtf8());
                }
                i += 1;
                chunk_start = i;
            }
            // `` $` `` -> input before the match.
            Some(b'`') => {
                out.push_wtf8(units_to_string(&input_units[..m_start]).as_wtf8());
                i += 1;
                chunk_start = i;
            }
            // `$'` -> input after the match.
            Some(b'\'') => {
                out.push_wtf8(units_to_string(&input_units[m_end..]).as_wtf8());
                i += 1;
                chunk_start = i;
            }
            // `$<name>` -> named group (the index builds on first use).
            Some(b'<') => {
                match bytes[i + 1..].iter().position(|&b| b == b'>') {
                    Some(rel) => {
                        let name_bytes = &bytes[i + 1..i + 1 + rel];
                        let map = names.get_or_insert_with(|| name_map(mt));
                        // Unknown names expand to `""`; group names come
                        // from the (`&str`) pattern, so a non-UTF-8 name
                        // slice can never match.
                        let idx = std::str::from_utf8(name_bytes)
                            .ok()
                            .and_then(|n| map.get(n).copied())
                            .unwrap_or(usize::MAX);
                        if let Some(t) = capture_text(input_units, mt, idx, &mut cache) {
                            out.push_wtf8(t.as_wtf8());
                        }
                        i += 1 + rel + 1;
                        chunk_start = i;
                    }
                    // No closing `>`: `$<` and the rest are literal.
                    None => {
                        out.push_wtf8_bytes(b"$<");
                        chunk_start = i + 1;
                        break;
                    }
                }
            }
            // `$1`..`$99`.
            Some(d) if d.is_ascii_digit() => {
                let d1 = (d - b'0') as usize;
                i += 1;
                // Two-digit reference when valid and in range; otherwise a
                // single digit when in range; otherwise literal.
                let mut num = d1;
                let mut two_digit = false;
                if d1 != 0
                    && let Some(d2) = bytes.get(i)
                    && d2.is_ascii_digit()
                {
                    let nn = d1 * 10 + (d2 - b'0') as usize;
                    if nn >= 1 && nn <= m {
                        num = nn;
                        two_digit = true;
                        i += 1;
                    }
                }
                if !two_digit && (d1 < 1 || d1 > m) {
                    out.push_char('$');
                    out.push_char((b'0' + d1 as u8) as char);
                } else if let Some(t) = capture_text(input_units, mt, num, &mut cache) {
                    out.push_wtf8(t.as_wtf8());
                }
                chunk_start = i;
            }
            // `$` + anything else: the `$` is literal, the next char
            // rescans (it may start the next chunk).
            _ => {
                out.push_char('$');
                chunk_start = i;
            }
        }
    }
    out.push_wtf8_bytes(&bytes[chunk_start..]);
    out
}

// ---------------------------------------------------------------------------
// Match execution
// ---------------------------------------------------------------------------

/// One engine match: raw captures plus the compiled program (kept so the
/// named-group table can be materialized lazily — most matches never need
/// it). `captures[0]` is the whole match.
struct Match {
    captures: Vec<Option<Capture>>,
    re: Rc<CompiledRegex>,
    /// `(1-based group, name)` in group-number order, built once on first
    /// use by [`Match::named_groups`].
    named: RefCell<Option<Vec<(usize, String)>>>,
}

impl Match {
    /// Named groups as `(1-based group, name)` in group-number order
    /// (parallel to the engine's `capture_names()` /
    /// `capture_name_groups()`). Computed once; patterns without named
    /// groups — and callers like `test`/`split` that never ask — skip the
    /// per-match `String` clones entirely.
    fn named_groups(&self) -> Ref<'_, Vec<(usize, String)>> {
        if self.named.borrow().is_none() {
            let pairs = self
                .re
                .capture_names()
                .iter()
                .zip(self.re.capture_name_groups().iter())
                .map(|(name, &group)| (group, name.clone()))
                .collect();
            *self.named.borrow_mut() = Some(pairs);
        }
        Ref::map(self.named.borrow(), |opt| {
            opt.as_ref().expect("named groups initialized above")
        })
    }
}

/// Run one match with the cached compiled pattern for `(source, flags)`.
/// `start` is a UTF-16 code-unit offset. Stickiness is implemented by the
/// engine itself (with `y` only `start` is tried).
fn run_exec(
    source: &Wtf8,
    flags: &Flags,
    units: &[u16],
    start: usize,
) -> Result<Option<Match>, RegexError> {
    let re = compile_cached_flags(source, flags)?;
    let found = re.exec_units(units, start);
    if let Some(err) = re.take_error() {
        return Err(err);
    }
    Ok(found.map(|captures| Match {
        captures,
        re,
        named: RefCell::new(None),
    }))
}

/// One match attempt at `start`, returning the match with its whole-match
/// span. `start` past the end is an automatic miss (this is what makes
/// `lastIndex` overflow fail the match and reset, per spec).
fn exec_at(
    source: &Wtf8,
    flags: &Flags,
    units: &[u16],
    start: usize,
) -> Result<Option<(Match, usize, usize)>, RegexError> {
    if start > units.len() {
        return Ok(None);
    }
    let m = run_exec(source, flags, units, start)?;
    Ok(m.and_then(|mt| {
        let c = mt.captures.first().copied().flatten()?;
        Some((mt, c.start, c.end))
    }))
}

/// Spec `RegExpExec`: honors `global`/`sticky` `lastIndex` (read via
/// `ToLength`), updates it to the match end on success and resets it to
/// `0` on failure. Non-global, non-sticky searches start at `0` and leave
/// `lastIndex` untouched.
fn spec_exec(
    heap: &mut Heap,
    rx: Gc<Object>,
    units: &[u16],
    source: &Wtf8,
    flags: &Flags,
) -> Result<Option<(Match, usize, usize)>, NativeError> {
    let start = if flags.g || flags.y {
        last_index_start(heap, rx)?
    } else {
        0
    };
    // The pattern flows to the engine as `&Wtf8`: lone surrogates are
    // preserved exactly, and matching runs on the exact UTF-16 code units.
    let found = exec_at(source, flags, units, start).map_err(regex_native_error)?;
    match found {
        Some((mt, a, b)) => {
            if flags.g || flags.y {
                // Empty match: advance one code point so iteration makes
                // progress (spec `AdvanceStringIndex`).
                let next = if b == a {
                    advance_string_index(units, a, flags.u)
                } else {
                    b
                };
                set_last_index(heap, rx, next as f64)?;
            }
            Ok(Some((mt, a, b)))
        }
        None => {
            if flags.g || flags.y {
                set_last_index(heap, rx, 0.0)?;
            }
            Ok(None)
        }
    }
}

/// Materialize every capture as a JS string (`None` = non-participating).
fn group_texts(units: &[u16], mt: &Match) -> Vec<Option<Wtf8Buf>> {
    mt.captures
        .iter()
        .map(|c| {
            c.map(|cp| units_to_string(&units[cp.start.min(units.len())..cp.end.min(units.len())]))
        })
        .collect()
}

/// Map each group name to its 1-based index (first declaration wins).
fn name_map(mt: &Match) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for (group, name) in mt.named_groups().iter() {
        map.entry(name.clone()).or_insert(*group);
    }
    map
}

// ---------------------------------------------------------------------------
// Result arrays
// ---------------------------------------------------------------------------

/// Build an `Array` of `elems` (indexed properties plus `length`).
///
/// Elements go through dense storage and are materialized once, so batch
/// construction is O(n) instead of O(n²).
fn new_array_elems(heap: &mut Heap, elems: Vec<Value>) -> Result<Gc<Object>, NativeError> {
    let mut base = None::<usize>;
    for &e in &elems {
        base.get_or_insert(heap.protect(e));
    }
    let arr = heap.new_dense_array()?;
    if let Some(b) = base {
        heap.release_temp(b);
    }
    for e in elems {
        heap.dense_push(arr, e)?;
    }
    heap.materialize_dense(arr)?;
    Ok(arr)
}

/// The `exec` result array: `[match, g1, ...]` with `index`, `input`,
/// `groups`, and (for `d`) `indices` properties.
fn build_result_array(
    heap: &mut Heap,
    input: &Wtf8,
    units: &[u16],
    mt: &Match,
    with_indices: bool,
) -> Result<Value, NativeError> {
    let texts = group_texts(units, mt);
    let mut elems = Vec::with_capacity(texts.len());
    for t in &texts {
        elems.push(match t {
            Some(s) => Value::Str(heap.alloc_str_wtf8(s)?),
            None => Value::Undefined,
        });
    }
    let arr = new_array_elems(heap, elems)?;
    let writable_enum = |value: Value| Property {
        value,
        writable: true,
        enumerable: true,
        configurable: true,
    };
    let (wstart, _) = mt
        .captures
        .first()
        .copied()
        .flatten()
        .map(|c| (c.start, c.end))
        .unwrap_or((0, 0));
    heap.define_property_str(arr, "index", writable_enum(js_num(wstart as f64)))?;
    let input_v = Value::Str(heap.alloc_str_wtf8(input)?);
    heap.define_property_str(arr, "input", writable_enum(input_v))?;
    let groups_v = groups_object(heap, units, mt)?;
    heap.define_property_str(arr, "groups", writable_enum(groups_v))?;
    if with_indices {
        let mut idx_elems = Vec::with_capacity(mt.captures.len());
        for c in &mt.captures {
            idx_elems.push(match c {
                Some(cp) => {
                    let pair = new_array_elems(
                        heap,
                        vec![js_num(cp.start as f64), js_num(cp.end as f64)],
                    )?;
                    Value::Object(pair)
                }
                None => Value::Undefined,
            });
        }
        let ia = new_array_elems(heap, idx_elems)?;
        let groups_v = groups_indices_object(heap, mt)?;
        heap.define_property_str(ia, "groups", writable_enum(groups_v))?;
        heap.define_property_str(arr, "indices", writable_enum(Value::Object(ia)))?;
    }
    Ok(Value::Object(arr))
}

/// The `groups` object: name → matched text or `undefined`. `undefined`
/// when the pattern has no named groups.
fn groups_object(heap: &mut Heap, units: &[u16], mt: &Match) -> Result<Value, NativeError> {
    let named = mt.named_groups();
    if named.is_empty() {
        return Ok(Value::Undefined);
    }
    let g = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    for (group, name) in named.iter() {
        if heap.get_property(g, Wtf8::from_str(name)).is_some() {
            continue;
        }
        let v = mt
            .captures
            .get(*group)
            .copied()
            .flatten()
            .map(|cp| units_to_string(&units[cp.start..cp.end]));
        let v = match v {
            Some(t) => {
                let s = heap.alloc_str_wtf8(&t)?;
                Value::Str(s)
            }
            None => Value::Undefined,
        };
        heap.define_property_str(g, name, Property::data(v))?;
    }
    Ok(Value::Object(g))
}

/// The `indices.groups` object for the `d` flag.
fn groups_indices_object(heap: &mut Heap, mt: &Match) -> Result<Value, NativeError> {
    let named = mt.named_groups();
    if named.is_empty() {
        return Ok(Value::Undefined);
    }
    let g = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    for (group, name) in named.iter() {
        if heap.get_property(g, Wtf8::from_str(name)).is_some() {
            continue;
        }
        let v = mt
            .captures
            .get(*group)
            .copied()
            .flatten()
            .map(|cp| -> Result<Value, NativeError> {
                let pair =
                    new_array_elems(heap, vec![js_num(cp.start as f64), js_num(cp.end as f64)])?;
                Ok(Value::Object(pair))
            })
            .transpose()?
            .unwrap_or(Value::Undefined);
        heap.define_property_str(g, name, Property::data(v))?;
    }
    Ok(Value::Object(g))
}

// ---------------------------------------------------------------------------
// Constructor and prototype methods
// ---------------------------------------------------------------------------

pub(crate) fn regexp_construct(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    // v1: call and construct forms are identical except for prototype
    // wiring below (`NativeFn` cannot take `new.target` as a parameter; it
    // observes construction via `active_new_target` instead).
    let pattern = arg(args, 0);
    let flags_v = arg(args, 1);
    let rx = if let Some(rx) = as_regexp(pattern) {
        if flags_v.is_undefined() {
            // Spec 22.2.3.1 step 1.c: the call form (`NewTarget` undefined)
            // with a regexp pattern and no flags returns the pattern
            // itself when `SameValue(newTarget, Get(pattern,
            // "constructor"))`. `newTarget` is the active function object —
            // the installed `RegExp` constructor (see `REGEXP_CTOR`).
            if crate::vm::active_new_target().is_none()
                && let Some((id, ctor)) = REGEXP_CTOR.with(|c| *c.borrow())
                && id == heap.id()
                && crate::builtin_reflect_proxy::get_prop_chain(
                    heap,
                    rx,
                    Wtf8::from_str("constructor"),
                )
                .is_some_and(|c| c == Value::Object(ctor))
            {
                return Ok(pattern);
            }
            copy_regexp(heap, rx)?
        } else {
            let source = own_str_prop(heap, rx, Wtf8::from_str("source"));
            // A lone-surrogate flag is an invalid flag (spec: SyntaxError).
            let flags = parse_flags(&to_string_rust(flags_v)?).map_err(syntax_error)?;
            make_regexp(heap, source.as_wtf8(), &flags)?
        }
    } else {
        let raw = if pattern.is_undefined() {
            Wtf8Buf::new()
        } else {
            to_string_rust(pattern)?
        };
        let flags = if flags_v.is_undefined() {
            Flags::default()
        } else {
            // A lone-surrogate flag is an invalid flag (spec: SyntaxError).
            parse_flags(&to_string_rust(flags_v)?).map_err(syntax_error)?
        };
        make_regexp(heap, escape_source(raw.as_wtf8()).as_wtf8(), &flags)?
    };
    // Spec 22.2.4.1: when `NewTarget` is undefined (the call form — every
    // regexp literal arrives this way via `op_new_regex` invoking the
    // constructor as a plain function), it defaults to the active function
    // object, i.e. the `RegExp` constructor, and `proto` is that
    // constructor's `"prototype"`. `OrdinaryCreateFromConstructor` falls
    // back to `%RegExp.prototype%` when the property is not an object,
    // and the prototype is always set: literals used to keep the null
    // default, which broke every inherited method (`/a/.exec('a')` threw
    // "callee is not a function", `Object.getPrototypeOf(/a/)` was null).
    let from_new_target: Option<Value> = match crate::vm::active_new_target() {
        Some(Value::Object(ctor)) => heap
            .get_property(ctor, Wtf8::from_str("prototype"))
            .map(|p| p.value)
            .filter(|v| matches!(v, Value::Object(_))),
        _ => None,
    };
    let proto = from_new_target.unwrap_or_else(|| {
        REGEXP_PROTO.with(|c| match *c.borrow() {
            Some((id, proto)) if id == heap.id() => Value::Object(proto),
            _ => Value::Null,
        })
    });
    heap.set_prototype(rx, proto);
    Ok(Value::Object(rx))
}

fn regexp_exec_fn(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let rx = this_regexp(this, "exec")?;
    let input = to_string_rust(arg(args, 0))?;
    let units: Vec<u16> = input.to_units();
    let source = own_str_prop(heap, rx, Wtf8::from_str("source"));
    let flags =
        parse_flags(&own_str_prop(heap, rx, Wtf8::from_str("flags"))).map_err(syntax_error)?;
    match spec_exec(heap, rx, &units, &source, &flags)? {
        Some((mt, _, _)) => build_result_array(heap, &input, &units, &mt, flags.d),
        None => Ok(Value::Null),
    }
}

fn regexp_test_fn(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let rx = this_regexp(this, "test")?;
    let input = to_string_rust(arg(args, 0))?;
    let units: Vec<u16> = input.to_units();
    let source = own_str_prop(heap, rx, Wtf8::from_str("source"));
    let flags =
        parse_flags(&own_str_prop(heap, rx, Wtf8::from_str("flags"))).map_err(syntax_error)?;
    let found = spec_exec(heap, rx, &units, &source, &flags)?.is_some();
    Ok(Value::Bool(found))
}

/// Spec 22.2.5.15 `RegExp.prototype.toString`: `"/" + source + "/" +
/// flags`. Step 2 requires an Object receiver — primitives throw a
/// `TypeError` (null/undefined are already rejected as
/// non-Object-coercible). `source`/`flags` go through `ToString`, so an
/// absent or `undefined` property stringifies to `"undefined"` (no
/// `(?:)` default: that default belongs to the `RegExp` constructor's
/// `source` own-property, not to this method).
fn regexp_to_string_fn(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    // Step 1 (`RequireObjectCoercible`) and step 2 (Object receiver).
    let Value::Object(o) = this else {
        return Err(type_error(
            "RegExp.prototype.toString called on incompatible receiver",
        ));
    };
    // `Get(R, key)`: absent/`undefined` yield `None`; anything else goes
    // through `ToString` (`"undefined"` for `None`). (Data properties
    // only — mirrors the file's other `source`/`flags` reads.)
    let mut get = |key: &str| -> Result<Wtf8Buf, NativeError> {
        match crate::builtin_reflect_proxy::get_prop_chain(heap, o, Wtf8::from_str(key)) {
            None | Some(Value::Undefined) => Ok(Wtf8Buf::from("undefined")),
            Some(v) => crate::builtin_object_array::to_string_js(heap, v),
        }
    };
    let source = get("source")?;
    let flags = get("flags")?;
    let mut out = Wtf8Buf::from("/");
    out.push_wtf8(source.as_wtf8());
    out.push_wtf8(Wtf8::from_str("/"));
    out.push_wtf8(flags.as_wtf8());
    Ok(Value::Str(heap.alloc_str_wtf8(&out)?))
}

// ---------------------------------------------------------------------------
// Replace
// ---------------------------------------------------------------------------

/// How the replacement string is produced.
enum Replacer {
    Template(Wtf8Buf),
    /// A callable value; invoked through the shared
    /// [`crate::builtin_object_array::call_function`] choke point so
    /// bytecode replacers re-enter the VM.
    Func(Value),
}

/// Evaluate one replacement: template substitution or a replacer call
/// `(match, p1..pn, offset, string, groups)`.
fn eval_replacement(
    heap: &mut Heap,
    replacer: &Replacer,
    s: &Wtf8,
    units: &[u16],
    mt: &Match,
    a: usize,
    b: usize,
) -> Result<Wtf8Buf, NativeError> {
    match replacer {
        Replacer::Template(t) => Ok(apply_substitution(t, units, a, b, mt)),
        Replacer::Func(f) => {
            let mut fargs = Vec::with_capacity(mt.captures.len() + 3);
            for c in &mt.captures {
                fargs.push(match c {
                    Some(cp) => {
                        Value::Str(heap.alloc_str_wtf8(&units_to_string(&units[cp.start..cp.end]))?)
                    }
                    None => Value::Undefined,
                });
            }
            fargs.push(js_num(a as f64));
            fargs.push(Value::Str(heap.alloc_str_wtf8(s)?));
            fargs.push(groups_object(heap, units, mt)?);
            to_string_rust(crate::builtin_object_array::call_function(
                heap,
                *f,
                Value::Undefined,
                &fargs,
            )?)
        }
    }
}

/// `RegExp.prototype [ @@replace ]`: also reached via the generic
/// `@@replace` lookup in `String.prototype.replace` / `replaceAll`
/// (see `builtin_string_number::string_replace`).
pub(crate) fn regexp_symbol_replace(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let rx = this_regexp(this, "Symbol.replace")?;
    let s = to_string_rust(arg(args, 0))?;
    let replace_v = arg(args, 1);
    let source = own_str_prop(heap, rx, Wtf8::from_str("source"));
    let flags =
        parse_flags(&own_str_prop(heap, rx, Wtf8::from_str("flags"))).map_err(syntax_error)?;
    let units: Vec<u16> = s.to_units();

    let replacer = match replace_v {
        Value::Object(o) => match o.kind {
            // Native and bytecode replacers both go through the shared
            // `call_function` choke point (bytecode re-enters the VM).
            ObjectKind::NativeFunction(_) | ObjectKind::Function(_) => Replacer::Func(replace_v),
            // Non-callable objects go through the template path.
            _ => Replacer::Template(to_string_rust(replace_v)?),
        },
        _ => Replacer::Template(to_string_rust(replace_v)?),
    };

    if !flags.g {
        return match spec_exec(heap, rx, &units, &source, &flags)? {
            None => Ok(Value::Str(heap.alloc_str_wtf8(&s)?)),
            Some((mt, a, b)) => {
                let r = eval_replacement(heap, &replacer, &s, &units, &mt, a, b)?;
                let mut out = units_to_string(&units[..a]);
                out.push_wtf8(&r);
                out.push_wtf8(&units_to_string(&units[b..]));
                Ok(Value::Str(heap.alloc_str_wtf8(&out)?))
            }
        };
    }

    set_last_index(heap, rx, 0.0)?;
    // Spec `RegExp.prototype [ @@replace ]`: collect every match before
    // running the replacer. Interleaving exec with replacer calls lets a
    // replacer that mutates `lastIndex` corrupt the scan (and panicked on
    // the resulting backwards slice); the scan must be immune to it.
    let mut found = Vec::new();
    loop {
        match spec_exec(heap, rx, &units, &source, &flags)? {
            // `spec_exec` already reset `lastIndex` to 0.
            None => break,
            // `spec_exec` already advanced `lastIndex` past an empty match.
            Some(m) => found.push(m),
        }
    }
    let out = accumulate_replacements(heap, &replacer, &s, &units, &found)?;
    Ok(Value::Str(heap.alloc_str_wtf8(&out)?))
}

/// Stitch the collected `@@replace` matches into the output string.
///
/// `found` holds `(match, start, end)` in scan order. Spec 22.2.9.8 step
/// 16.p: a match whose position moved backwards (`start < nextSourcePosition`,
/// from an ill-behaving rx) has its substitution ignored — slicing
/// `units[last..start]` would panic, so the result is skipped.
fn accumulate_replacements(
    heap: &mut Heap,
    replacer: &Replacer,
    s: &Wtf8,
    units: &[u16],
    found: &[(Match, usize, usize)],
) -> Result<Wtf8Buf, NativeError> {
    let mut out = Wtf8Buf::new();
    let mut last = 0usize;
    for (mt, a, b) in found {
        if *a < last {
            continue;
        }
        out.push_wtf8(&units_to_string(&units[last..*a]));
        out.push_wtf8(&eval_replacement(heap, replacer, s, units, mt, *a, *b)?);
        last = *b;
    }
    out.push_wtf8(&units_to_string(&units[last..]));
    Ok(out)
}

// ---------------------------------------------------------------------------
// Split
// ---------------------------------------------------------------------------

/// Spec `RegExp.prototype [ @@split ]` without the heap: pieces of `input`
/// with captures interleaved (`None` = non-participating group). `flags`
/// must *not* include `y`; the splitter adds it internally so the original
/// regexp is never disturbed.
fn split_raw(
    source: &Wtf8,
    flags: &Flags,
    units: &[u16],
    limit: u64,
) -> Result<Vec<Option<Wtf8Buf>>, RegexError> {
    let mut out: Vec<Option<Wtf8Buf>> = Vec::new();
    if limit == 0 {
        return Ok(out);
    }
    let size = units.len();
    let mut sflags = *flags;
    sflags.y = true;
    if size == 0 {
        if exec_at(source, &sflags, units, 0)?.is_some() {
            return Ok(out);
        }
        out.push(Some(Wtf8Buf::new()));
        return Ok(out);
    }
    let mut p = 0usize;
    let mut q = 0usize;
    while q < size {
        match exec_at(source, &sflags, units, q)? {
            None => {
                q = advance_string_index(units, q, flags.u);
            }
            Some((mt, a, b)) => {
                debug_assert!(a == q, "sticky splitter must match at the search position");
                let e = b.min(size);
                if e == p {
                    q = advance_string_index(units, q, flags.u);
                    continue;
                }
                out.push(Some(units_to_string(&units[p..q])));
                if out.len() as u64 >= limit {
                    return Ok(out);
                }
                for c in mt.captures.iter().skip(1) {
                    out.push(c.map(|cp| units_to_string(&units[cp.start..cp.end])));
                    if out.len() as u64 >= limit {
                        return Ok(out);
                    }
                }
                p = e;
                q = p;
                if out.len() as u64 >= limit {
                    return Ok(out);
                }
            }
        }
    }
    out.push(Some(units_to_string(&units[p..])));
    Ok(out)
}

fn regexp_symbol_split(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let rx = this_regexp(this, "Symbol.split")?;
    let s = to_string_rust(arg(args, 0))?;
    let lim = match arg(args, 1) {
        Value::Undefined => u64::from(u32::MAX),
        v => u64::from(to_uint32(v)?),
    };
    let units: Vec<u16> = s.to_units();
    let source = own_str_prop(heap, rx, Wtf8::from_str("source"));
    let flags =
        parse_flags(&own_str_prop(heap, rx, Wtf8::from_str("flags"))).map_err(syntax_error)?;
    // The pattern compiles exactly (lone surrogates preserved); pieces stay exact.
    let pieces = split_raw(source.as_wtf8(), &flags, &units, lim).map_err(regex_native_error)?;
    let mut elems = Vec::with_capacity(pieces.len());
    for p in pieces {
        elems.push(match p {
            Some(t) => Value::Str(heap.alloc_str_wtf8(&t)?),
            None => Value::Undefined,
        });
    }
    Ok(Value::Object(new_array_elems(heap, elems)?))
}

// ---------------------------------------------------------------------------
// Match / matchAll
// ---------------------------------------------------------------------------

pub(crate) fn regexp_symbol_match(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let rx = this_regexp(this, "@@match")?;
    let s = to_string_rust(arg(args, 0))?;
    let units: Vec<u16> = s.to_units();
    let source = own_str_prop(heap, rx, Wtf8::from_str("source"));
    let flags =
        parse_flags(&own_str_prop(heap, rx, Wtf8::from_str("flags"))).map_err(syntax_error)?;
    if !flags.g {
        return match spec_exec(heap, rx, &units, &source, &flags)? {
            Some((mt, _, _)) => build_result_array(heap, &s, &units, &mt, flags.d),
            None => Ok(Value::Null),
        };
    }
    set_last_index(heap, rx, 0.0)?;
    let mut elems = Vec::new();
    loop {
        match spec_exec(heap, rx, &units, &source, &flags)? {
            None => break,
            Some((_, a, b)) => {
                elems.push(Value::Str(
                    heap.alloc_str_wtf8(&units_to_string(&units[a..b]))?,
                ));
                // `spec_exec` already advanced `lastIndex` past an empty match.
            }
        }
    }
    if elems.is_empty() {
        Ok(Value::Null)
    } else {
        Ok(Value::Object(new_array_elems(heap, elems)?))
    }
}

/// v1: returns an `Array` of match arrays. The spec returns a
/// `RegExpStringIterator`; that needs the iterator protocol, which does
/// not exist yet.
pub(crate) fn regexp_symbol_match_all(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let rx = this_regexp(this, "@@matchAll")?;
    let s = to_string_rust(arg(args, 0))?;
    let units: Vec<u16> = s.to_units();
    let source = own_str_prop(heap, rx, Wtf8::from_str("source"));
    let flags =
        parse_flags(&own_str_prop(heap, rx, Wtf8::from_str("flags"))).map_err(syntax_error)?;
    if !flags.g {
        return Err(type_error(
            "RegExp.prototype.Symbol.matchAll requires a global regular expression",
        ));
    }
    set_last_index(heap, rx, 0.0)?;
    let mut elems = Vec::new();
    loop {
        match spec_exec(heap, rx, &units, &source, &flags)? {
            None => break,
            Some((mt, _, _)) => {
                elems.push(build_result_array(heap, &s, &units, &mt, flags.d)?);
                // `spec_exec` already advanced `lastIndex` past an empty match.
            }
        }
    }
    Ok(Value::Object(new_array_elems(heap, elems)?))
}

/// `RegExp.prototype [ @@search ]` (spec 22.2.9.13): the code-unit index of
/// the first match, or `-1` when there is none.
///
/// Installed on the prototype so `String.prototype.search` dispatches
/// through it; the string builtin also calls it directly for the
/// `RegExpCreate` fallback (a freshly constructed regexp always chains to
/// this prototype).
pub(crate) fn regexp_symbol_search(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let rx = this_regexp(this, "@@search")?;
    let s = to_string_rust(arg(args, 0))?;
    let units: Vec<u16> = s.to_units();
    let source = own_str_prop(heap, rx, Wtf8::from_str("source"));
    let flags =
        parse_flags(&own_str_prop(heap, rx, Wtf8::from_str("flags"))).map_err(syntax_error)?;
    // Spec 22.2.9.13 `RegExp.prototype [ @@search ]`: the search always
    // starts at index 0; `lastIndex` is saved and restored around the
    // one-shot exec. The restore comparison must be `SameValue`, not
    // `!=`: `-0` and `+0` compare equal under `==` but are observably
    // distinct (`Object.is(rx.lastIndex, -0)`).
    let previous = get_last_index_val(heap, rx);
    set_last_index(heap, rx, 0.0)?;
    let found = spec_exec(heap, rx, &units, &source, &flags)?;
    let current = get_last_index_val(heap, rx);
    if !last_index_same_value(current, previous) {
        set_data_prop(heap, rx, "lastIndex", previous)?;
    }
    match found {
        Some((_, a, _)) => Ok(js_num(a as f64)),
        None => Ok(js_num(-1.0)),
    }
}

// ---------------------------------------------------------------------------
// Heapless one-shot helpers (integration surface for the string builtins)
// ---------------------------------------------------------------------------
///
/// These reuse the cached compiled pattern for `(pattern, flags)` (see the
/// compile-cache section above) and never touch `lastIndex`. The string
/// builtins worker calls these instead of going through the
/// well-known-symbol dispatch.
///
/// Does `pattern` match `input`?
#[allow(dead_code)]
pub(crate) fn regexp_test(pattern: &Wtf8, flags: &Wtf8, input: &Wtf8) -> Result<bool, RegexError> {
    let flags = parse_flags(flags).map_err(RegexError::ParseError)?;
    let units: Vec<u16> = input.to_units();
    Ok(exec_at(pattern, &flags, &units, 0)?.is_some())
}

/// Replace the first (or every, with `g`) match of `pattern` in `input`
/// using a `$`-pattern template (see [`apply_substitution`]).
#[allow(dead_code)]
pub(crate) fn regexp_replace(
    pattern: &Wtf8,
    flags: &Wtf8,
    input: &Wtf8,
    replacement: &Wtf8,
) -> Result<Wtf8Buf, RegexError> {
    let flags = parse_flags(flags).map_err(RegexError::ParseError)?;
    let units: Vec<u16> = input.to_units();
    let subst = |mt: &Match, a: usize, b: usize| apply_substitution(replacement, &units, a, b, mt);
    if !flags.g {
        return match exec_at(pattern, &flags, &units, 0)? {
            None => Ok(input.to_owned()),
            Some((mt, a, b)) => {
                let mut out = units_to_string(&units[..a]);
                out.push_wtf8(subst(&mt, a, b).as_wtf8());
                out.push_wtf8(units_to_string(&units[b..]).as_wtf8());
                Ok(out)
            }
        };
    }
    let mut out = Wtf8Buf::new();
    let mut last = 0usize;
    let mut q = 0usize;
    while q <= units.len() {
        match exec_at(pattern, &flags, &units, q)? {
            None => break,
            Some((mt, a, b)) => {
                out.push_wtf8(units_to_string(&units[last..a]).as_wtf8());
                out.push_wtf8(subst(&mt, a, b).as_wtf8());
                last = b;
                q = if a == b {
                    advance_string_index(&units, b, flags.u)
                } else {
                    b
                };
            }
        }
    }
    out.push_wtf8(&units_to_string(&units[last..]));
    Ok(out)
}

/// Split `input` on `pattern`, keeping captures (`None` marks a
/// non-participating group). At most `limit` pieces.
#[allow(dead_code)]
pub(crate) fn regexp_split_raw(
    pattern: &Wtf8,
    flags: &Wtf8,
    input: &Wtf8,
    limit: u32,
) -> Result<Vec<Option<Wtf8Buf>>, RegexError> {
    let flags = parse_flags(flags).map_err(RegexError::ParseError)?;
    let units: Vec<u16> = input.to_units();
    split_raw(pattern, &flags, &units, u64::from(limit))
}

/// Split `input` on `pattern`; non-participating captures surface as empty
/// strings here (use [`regexp_split_raw`] when the distinction matters).
#[allow(dead_code)]
pub(crate) fn regexp_split(
    pattern: &Wtf8,
    flags: &Wtf8,
    input: &Wtf8,
    limit: u32,
) -> Result<Vec<Wtf8Buf>, RegexError> {
    Ok(regexp_split_raw(pattern, flags, input, limit)?
        .into_iter()
        .map(|p| p.unwrap_or_default())
        .collect())
}

// ---------------------------------------------------------------------------
// install
// ---------------------------------------------------------------------------

/// Allocate a native function object with `name`/`length` installed.
fn new_native_function(
    heap: &mut Heap,
    name: &str,
    func: NativeFn,
    length: u32,
) -> Result<Gc<Object>, NativeError> {
    let f = heap.alloc_object(Object::new(ObjectKind::NativeFunction(func)))?;
    heap.define_property_str(
        f,
        "length",
        Property {
            value: js_num(f64::from(length)),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    let name_v = Value::Str(with_temp_root(heap, Value::Object(f), |heap| {
        // `f` is not reachable from any root yet; hold it across the string
        // allocation, which may trigger a collection.
        heap.alloc_str(name).map_err(|_| NativeError::OutOfMemory)
    })?);
    heap.define_property_str(
        f,
        "name",
        Property {
            value: name_v,
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(f)
}

/// Define `target[name]` as a builtin method (writable, non-enumerable,
/// configurable, per spec).
fn define_native(
    heap: &mut Heap,
    target: Gc<Object>,
    name: &str,
    func: NativeFn,
    length: u32,
) -> Result<(), NativeError> {
    // Hold `target` across the function-object allocation below, which may
    // trigger a collection while the target is not yet otherwise rooted.
    let f = with_temp_root(heap, Value::Object(target), |heap| {
        new_native_function(heap, name, func, length)
    })?;
    heap.define_property_str(
        target,
        name,
        Property {
            value: Value::Object(f),
            writable: true,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(())
}

/// Install the `RegExp` constructor on `global`.
pub fn install(heap: &mut Heap, global: Gc<Object>) -> Result<(), NativeError> {
    install_inner(heap, global)
}

fn install_inner(heap: &mut Heap, global: Gc<Object>) -> Result<(), NativeError> {
    let proto = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    // `proto` is not reachable from any traced root until `RegExp` is
    // linked below; hold it across the constructor allocation, which may
    // trigger a collection. The temp root is released on both paths.
    with_temp_root(heap, Value::Object(proto), |heap| {
        define_native(heap, proto, "exec", regexp_exec_fn, 1)?;
        define_native(heap, proto, "test", regexp_test_fn, 1)?;
        // Spec 22.2.6.15: `"/" + source + "/" + flags` on any receiver.
        define_native(heap, proto, "toString", regexp_to_string_fn, 0)?;
        // Well-known-symbol methods, installed under real symbol keys: a
        // computed `rx[Symbol.replace]` key resolves to the same property
        // the abstract operations consult.
        for (wk, name, func, len) in [
            (
                WellKnown::Replace,
                "[Symbol.replace]",
                regexp_symbol_replace as NativeFn,
                2,
            ),
            (
                WellKnown::Split,
                "[Symbol.split]",
                regexp_symbol_split as NativeFn,
                2,
            ),
            (
                WellKnown::Match,
                "[Symbol.match]",
                regexp_symbol_match as NativeFn,
                1,
            ),
            (
                WellKnown::MatchAll,
                "[Symbol.matchAll]",
                regexp_symbol_match_all as NativeFn,
                1,
            ),
            (
                WellKnown::Search,
                "[Symbol.search]",
                regexp_symbol_search as NativeFn,
                1,
            ),
        ] {
            let sym = heap.well_known_symbol(wk)?;
            define_native_key(heap, proto, PropKey::Symbol(sym), name, func, len)?;
        }

        let ctor = new_native_function(heap, "RegExp", regexp_construct, 2)?;
        // `RegExp` is a constructor (ECMA-262 `IsConstructor`).
        heap.mark_native_constructor(regexp_construct);
        heap.define_property_str(
            ctor,
            "prototype",
            Property {
                value: Value::Object(proto),
                writable: false,
                enumerable: false,
                configurable: false,
            },
        )?;
        heap.define_property_str(proto, "constructor", Property::data(Value::Object(ctor)))?;
        heap.define_property_str(global, "RegExp", Property::data(Value::Object(ctor)))?;
        // Record `%RegExp.prototype%` for the call form of `regexp_construct`
        // (spec 22.2.4.1's active-function-object default); see `REGEXP_PROTO`.
        REGEXP_PROTO.with(|c| *c.borrow_mut() = Some((heap.id(), proto)));
        // Record the constructor itself for the spec 22.2.3.1
        // `RegExp(regexp)` short-circuit; see `REGEXP_CTOR`.
        REGEXP_CTOR.with(|c| *c.borrow_mut() = Some((heap.id(), ctor)));
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn install_global(heap: &mut Heap) -> Gc<Object> {
        let global = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        install(heap, global).unwrap();
        global
    }

    /// The `RegExp` constructor as (object, native fn).
    fn regexp_ctor(heap: &Heap, global: Gc<Object>) -> (Gc<Object>, NativeFn) {
        match heap
            .get_property(global, Wtf8::from_str("RegExp"))
            .unwrap()
            .value
        {
            Value::Object(o) => {
                let kind = heap.get(o).kind;
                match kind {
                    ObjectKind::NativeFunction(f) => (o, f),
                    _ => panic!("RegExp is not a native function"),
                }
            }
            v => panic!("no RegExp on global: {v:?}"),
        }
    }

    fn construct(heap: &mut Heap, ctor: NativeFn, args: &[Value]) -> Gc<Object> {
        match ctor(heap, Value::Undefined, args).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        }
    }

    fn s(heap: &mut Heap, text: &str) -> Value {
        Value::Str(heap.alloc_str(text).unwrap())
    }

    fn prop(heap: &Heap, o: Gc<Object>, name: &str) -> Value {
        heap.get_property(o, Wtf8::from_str(name))
            .unwrap_or_else(|| panic!("missing property {name}"))
            .value
    }

    fn str_prop(heap: &Heap, o: Gc<Object>, name: &str) -> String {
        match prop(heap, o, name) {
            Value::Str(g) => g.as_wtf8().to_string(),
            v => panic!("property {name} is not a string: {v:?}"),
        }
    }

    fn bool_prop(heap: &Heap, o: Gc<Object>, name: &str) -> bool {
        match prop(heap, o, name) {
            Value::Bool(b) => b,
            v => panic!("property {name} is not a bool: {v:?}"),
        }
    }

    fn num_val(v: Value) -> f64 {
        match v {
            Value::Int(i) => f64::from(i),
            Value::Float(f) => f,
            v => panic!("not a number: {v:?}"),
        }
    }

    fn as_rust_str(v: Value) -> String {
        match v {
            Value::Str(g) => g.as_wtf8().to_string(),
            v => panic!("not a string: {v:?}"),
        }
    }

    /// Build `args` (which may allocate into `heap`) before invoking the
    /// native fn, so call sites never nest `&mut heap` borrows.
    macro_rules! vcall {
        ($heap:expr, $f:expr, $this:expr, [$($a:expr),* $(,)?]) => {{
            let __vcall_args: Vec<Value> = vec![$($a),*];
            $f($heap, $this, &__vcall_args)
        }};
    }

    /// `construct` through [`vcall`].
    macro_rules! rx {
        ($heap:expr, $ctor:expr, [$($a:expr),* $(,)?]) => {{
            let __rx_args: Vec<Value> = vec![$($a),*];
            construct($heap, $ctor, &__rx_args)
        }};
    }

    /// A `RegExp.prototype` method as a native fn.
    fn proto_method(heap: &Heap, ctor_obj: Gc<Object>, name: &str) -> NativeFn {
        let proto = match prop(heap, ctor_obj, "prototype") {
            Value::Object(o) => o,
            v => panic!("prototype is not an object: {v:?}"),
        };
        match prop(heap, proto, name) {
            Value::Object(o) => {
                let kind = heap.get(o).kind;
                match kind {
                    ObjectKind::NativeFunction(f) => f,
                    _ => panic!("{name} is not a native function"),
                }
            }
            v => panic!("no method {name}: {v:?}"),
        }
    }

    /// A `RegExp.prototype` well-known-symbol method as a native fn,
    /// fetched under the real symbol key.
    fn proto_method_key(heap: &Heap, ctor_obj: Gc<Object>, wk: WellKnown) -> NativeFn {
        let proto = match prop(heap, ctor_obj, "prototype") {
            Value::Object(o) => o,
            v => panic!("prototype is not an object: {v:?}"),
        };
        let sym = heap.well_known[wk as usize].expect("well-known symbol installed");
        match heap
            .get_property_key(proto, PropKey::Symbol(sym))
            .map(|p| p.value)
        {
            Some(Value::Object(o)) => {
                let kind = heap.get(o).kind;
                match kind {
                    ObjectKind::NativeFunction(f) => f,
                    _ => panic!("{wk:?} is not a native function"),
                }
            }
            v => panic!("no method {wk:?}: {v:?}"),
        }
    }

    #[test]
    fn substitution_patterns() {
        // Pattern `(a)(b)?` matches "a" at 2..3 of "xb ax" (leftmost);
        // group 1 = "a", group 2 is unset.
        let r = |t: &Wtf8| {
            regexp_replace(
                Wtf8::from_str("(a)(b)?"),
                Wtf8::from_str(""),
                Wtf8::from_str("xb ax"),
                t,
            )
            .unwrap()
            .to_string()
        };
        assert_eq!(r(Wtf8::from_str("$$")), "xb $x");
        assert_eq!(r(Wtf8::from_str("$&")), "xb ax");
        assert_eq!(r(Wtf8::from_str("$`")), "xb xb x");
        assert_eq!(r(Wtf8::from_str("$'")), "xb xx");
        assert_eq!(r(Wtf8::from_str("<$1>")), "xb <a>x");
        assert_eq!(r(Wtf8::from_str("<$2>")), "xb <>x");
        assert_eq!(r(Wtf8::from_str("$0")), "xb $0x");
        assert_eq!(r(Wtf8::from_str("$9")), "xb $9x");
        assert_eq!(r(Wtf8::from_str("$x")), "xb $xx");
        assert_eq!(r(Wtf8::from_str("a$")), "xb a$x");
        assert_eq!(r(Wtf8::from_str("$1$2$&")), "xb aax");
        // No `$` at all: the template is literal.
        assert_eq!(r(Wtf8::from_str("plain")), "xb plainx");
    }

    #[test]
    fn substitution_named_groups() {
        let r = |t: &Wtf8| {
            regexp_replace(
                Wtf8::from_str("(?<first>a)(b)?"),
                Wtf8::from_str(""),
                Wtf8::from_str("xb ax"),
                t,
            )
            .unwrap()
            .to_string()
        };
        assert_eq!(r(Wtf8::from_str("$<first>")), "xb ax");
        assert_eq!(r(Wtf8::from_str("$<missing>")), "xb x");
        assert_eq!(r(Wtf8::from_str("$<oops")), "xb $<oopsx");
        assert_eq!(r(Wtf8::from_str("$<>")), "xb x");
    }

    #[test]
    fn substitution_two_digit_group() {
        // 12 groups; group 12 is "l".
        let r = |t: &Wtf8| {
            regexp_replace(
                Wtf8::from_str("(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)(k)(l)"),
                Wtf8::from_str(""),
                Wtf8::from_str("abcdefghijkl"),
                t,
            )
            .unwrap()
            .to_string()
        };
        assert_eq!(r(Wtf8::from_str("$12")), "l");
        // 13 > m (12 groups): single-digit `$1`, then literal `3`.
        assert_eq!(r(Wtf8::from_str("$13")), "a3");
        assert_eq!(r(Wtf8::from_str("$1!")), "a!");
        assert_eq!(r(Wtf8::from_str("$01")), "$01");
    }

    #[test]
    fn install_wires_constructor_and_prototype() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, _) = regexp_ctor(&heap, global);
        assert_eq!(str_prop(&heap, ctor_obj, "name"), "RegExp");
        assert_eq!(num_val(prop(&heap, ctor_obj, "length")), 2.0);
        for name in ["exec", "test"] {
            proto_method(&heap, ctor_obj, name);
        }
        // The well-known-symbol methods install under real symbol keys —
        // and the `"@@name"` string keys do NOT alias them.
        for wk in [
            WellKnown::Replace,
            WellKnown::Split,
            WellKnown::Match,
            WellKnown::MatchAll,
            WellKnown::Search,
        ] {
            proto_method_key(&heap, ctor_obj, wk);
        }
        let proto = match prop(&heap, ctor_obj, "prototype") {
            Value::Object(o) => o,
            v => panic!("prototype is not an object: {v:?}"),
        };
        for alias in ["@@replace", "@@split", "@@match", "@@matchAll", "@@search"] {
            assert!(
                heap.get_property(proto, Wtf8::from_str(alias)).is_none(),
                "\"{alias}\" string key must not alias the symbol method"
            );
        }
        assert_eq!(prop(&heap, proto, "constructor"), Value::Object(ctor_obj));
    }

    // Note: the install path must survive a collection on every allocation
    // (threshold 1): `%RegExp.prototype%` is allocated before the
    // constructor, so without the temp-root it would be freed mid-install
    // and the constructor's `prototype` link would dangle.
    // (Covered by `install_paths_survive_threshold_1_collections` in
    // `tests/vm.rs`.)

    #[test]
    fn call_form_inherits_regexp_prototype() {
        // Spec 22.2.4.1: the call form (`RegExp(...)` — how `op_new_regex`
        // builds every regexp literal) defaults `NewTarget` to the active
        // function object, so instances still chain to
        // `%RegExp.prototype%`. Before the fix the prototype stayed null
        // and inherited methods were unreachable from literals.
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "a"), s(&mut heap, "")]);
        assert_eq!(heap.get(rx).prototype, prop(&heap, ctor_obj, "prototype"));
        // End to end through a real VM: the literal's inherited methods
        // resolve and `Object.getPrototypeOf` sees `%RegExp.prototype%`.
        let mut vm = crate::Vm::new();
        let exec_ok = vm.eval("/a/.exec('a')[0]").unwrap();
        assert!(matches!(exec_ok, Value::Str(_)));
        let proto_name = vm.eval("Object.getPrototypeOf(/a/) === RegExp.prototype");
        assert!(matches!(proto_name, Ok(Value::Bool(true))));
        let hooked = vm.eval("'aba'.matchAll(/a/g).length");
        assert!(matches!(hooked, Ok(Value::Int(2))));
    }

    #[test]
    fn flags_validate() {
        assert_eq!(
            flags_string(&parse_flags(Wtf8::from_str("gim")).unwrap()),
            "gim"
        );
        assert_eq!(
            flags_string(&parse_flags(Wtf8::from_str("yus")).unwrap()),
            "suy"
        );
        assert_eq!(flags_string(&parse_flags(Wtf8::from_str("")).unwrap()), "");
        assert_eq!(
            flags_string(&parse_flags(Wtf8::from_str("dgimsuy")).unwrap()),
            "dgimsuy"
        );
        assert!(parse_flags(Wtf8::from_str("gg")).is_err());
        assert!(parse_flags(Wtf8::from_str("z")).is_err());
        assert!(parse_flags(Wtf8::from_str("uv")).is_err());
        assert!(parse_flags(Wtf8::from_str("vu")).is_err());
    }

    #[test]
    fn constructor_sets_properties() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (_, ctor) = regexp_ctor(&heap, global);
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "ab"), s(&mut heap, "gi")]);
        assert_eq!(str_prop(&heap, rx, "source"), "ab");
        assert_eq!(str_prop(&heap, rx, "flags"), "gi");
        assert!(bool_prop(&heap, rx, "global"));
        assert!(bool_prop(&heap, rx, "ignoreCase"));
        assert!(!bool_prop(&heap, rx, "multiline"));
        assert!(!bool_prop(&heap, rx, "dotAll"));
        assert!(!bool_prop(&heap, rx, "unicode"));
        assert!(!bool_prop(&heap, rx, "sticky"));
        assert!(!bool_prop(&heap, rx, "hasIndices"));
        assert_eq!(num_val(prop(&heap, rx, "lastIndex")), 0.0);
        // `source` is read-only, `lastIndex` is writable.
        let zz = s(&mut heap, "zz");
        set_data_prop(&mut heap, rx, "source", zz).unwrap();
        assert_eq!(str_prop(&heap, rx, "source"), "ab");
        set_last_index(&mut heap, rx, 4.0).unwrap();
        assert_eq!(num_val(prop(&heap, rx, "lastIndex")), 4.0);
    }

    #[test]
    fn constructor_empty_pattern_becomes_non_capturing_group() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (_, ctor) = regexp_ctor(&heap, global);
        let rx = rx!(&mut heap, ctor, []);
        assert_eq!(str_prop(&heap, rx, "source"), "(?:)");
        let rx = rx!(&mut heap, ctor, [Value::Undefined]);
        assert_eq!(str_prop(&heap, rx, "source"), "(?:)");
    }

    #[test]
    fn constructor_escapes_slash_in_source() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (_, ctor) = regexp_ctor(&heap, global);
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "/")]);
        assert_eq!(str_prop(&heap, rx, "source"), "\\/");
    }

    #[test]
    fn constructor_copies_regexp_argument() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let rx1 = rx!(&mut heap, ctor, [s(&mut heap, "a+b"), s(&mut heap, "m")]);
        // Spec 22.2.3.1: the call form with a regexp pattern, no flags,
        // and `Get(pattern, "constructor")` identical to the active
        // function object (the `RegExp` constructor) returns the pattern
        // itself instead of copying.
        let rx2 = rx!(&mut heap, ctor, [Value::Object(rx1)]);
        assert_eq!(rx1, rx2);
        // Explicit flags still copy: source is kept (already escaped),
        // flags replaced.
        let rx3 = rx!(&mut heap, ctor, [Value::Object(rx1), s(&mut heap, "gy")]);
        assert_ne!(rx1, rx3);
        assert_eq!(str_prop(&heap, rx3, "source"), "a+b");
        assert_eq!(str_prop(&heap, rx3, "flags"), "gy");
        // An overridden `constructor` disables the short-circuit.
        heap.define_property_str(rx1, "constructor", Property::data(Value::Null))
            .unwrap();
        let rx4 = rx!(&mut heap, ctor, [Value::Object(rx1)]);
        assert_ne!(rx1, rx4);
        heap.define_property_str(rx1, "constructor", Property::data(Value::Object(ctor_obj)))
            .unwrap();
        // The construct form (`new`) always copies.
        let prev = crate::vm::set_new_target(Some(Value::Object(ctor_obj)));
        let rx5 = rx!(&mut heap, ctor, [Value::Object(rx1)]);
        crate::vm::set_new_target(prev);
        assert_ne!(rx1, rx5);
        assert_eq!(str_prop(&heap, rx5, "source"), "a+b");
        assert_eq!(str_prop(&heap, rx5, "flags"), "m");
    }

    #[test]
    fn constructor_rejects_bad_flags() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (_, ctor) = regexp_ctor(&heap, global);
        let r = vcall!(
            &mut heap,
            ctor,
            Value::Undefined,
            [s(&mut heap, "a"), s(&mut heap, "q")]
        );
        assert!(matches!(r, Err(NativeError::SyntaxError(_))));
    }

    #[test]
    fn exec_rejects_incompatible_receiver() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, _) = regexp_ctor(&heap, global);
        let exec = proto_method(&heap, ctor_obj, "exec");
        let r = vcall!(&mut heap, exec, Value::Undefined, [s(&mut heap, "a")]);
        assert!(matches!(r, Err(NativeError::TypeError(_))));
    }

    #[test]
    fn exec_miss_returns_null_and_resets_last_index() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let exec = proto_method(&heap, ctor_obj, "exec");
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "z"), s(&mut heap, "g")]);
        set_last_index(&mut heap, rx, 2.0).unwrap();
        let r = vcall!(&mut heap, exec, Value::Object(rx), [s(&mut heap, "abc")]).unwrap();
        assert_eq!(r, Value::Null);
        assert_eq!(num_val(prop(&heap, rx, "lastIndex")), 0.0);
    }

    #[test]
    fn test_miss_is_false() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let test = proto_method(&heap, ctor_obj, "test");
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "z")]);
        let r = vcall!(&mut heap, test, Value::Object(rx), [s(&mut heap, "abc")]).unwrap();
        assert_eq!(r, Value::Bool(false));
    }

    #[test]
    fn replace_without_match_returns_input() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let replace = proto_method_key(&heap, ctor_obj, WellKnown::Replace);
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "z"), s(&mut heap, "g")]);
        let r = vcall!(
            &mut heap,
            replace,
            Value::Object(rx),
            [s(&mut heap, "abc"), s(&mut heap, "X")]
        )
        .unwrap();
        assert_eq!(as_rust_str(r), "abc");
        // Heapless helper forms.
        assert_eq!(
            regexp_replace(
                Wtf8::from_str("z"),
                Wtf8::from_str("g"),
                Wtf8::from_str("abc"),
                Wtf8::from_str("X")
            )
            .unwrap()
            .as_wtf8(),
            Wtf8::from_str("abc")
        );
        assert_eq!(
            regexp_replace(
                Wtf8::from_str("z"),
                Wtf8::from_str(""),
                Wtf8::from_str("abc"),
                Wtf8::from_str("X")
            )
            .unwrap()
            .as_wtf8(),
            Wtf8::from_str("abc")
        );
        assert!(
            !regexp_test(
                Wtf8::from_str("z"),
                Wtf8::from_str(""),
                Wtf8::from_str("abc")
            )
            .unwrap()
        );
    }

    #[test]
    fn replace_global_template_terminates() {
        // Regression test for the `bench_replace_global_template` /
        // `bench_replace_global_named_template` stall: those benches ran
        // thousands of global template replaces, which take minutes in a
        // debug build and stalled `cargo test` past every timeout, looking
        // hung. A single replace must finish well inside the bound below;
        // it runs on a worker thread so a reintroduced non-termination
        // fails this test (via the timeout) instead of hanging the suite.
        fn run_bounded(pattern: &str, flags: &str, input: &str, replacement: &str) -> String {
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            let (pattern, flags, input, replacement) = (
                pattern.to_owned(),
                flags.to_owned(),
                input.to_owned(),
                replacement.to_owned(),
            );
            std::thread::spawn(move || {
                let out = regexp_replace(
                    Wtf8::from_str(&pattern),
                    Wtf8::from_str(&flags),
                    Wtf8::from_str(&input),
                    Wtf8::from_str(&replacement),
                );
                let _ = done_tx.send(out);
            });
            done_rx
                .recv_timeout(std::time::Duration::from_secs(30))
                .expect("global replace did not terminate within 30s")
                .expect("global replace failed")
                .to_string()
        }

        // The exact inputs from the two stalled benches.
        let input = "ab ".repeat(500);
        assert_eq!(run_bounded("ab", "g", &input, "<$&>"), "<ab> ".repeat(500));
        assert_eq!(
            run_bounded("(?<x>a)(?<y>b)", "g", &input, "<$<x>$<y>>"),
            "<ab> ".repeat(500)
        );
    }

    /// Non-writable `lastIndex` must throw `TypeError`, not hang.
    ///
    /// test262: `RegExp/prototype/Symbol.replace/g-init-lastindex-err.js`.
    /// Spec 22.2.9.8 step 10.c performs `Set(rx, "lastIndex", 0, true)`:
    /// a refused write throws. Previously the write was silently ignored,
    /// so the global scan re-matched at position 0 forever (unbounded
    /// `found` growth, OOM).
    #[test]
    fn replace_global_nonwritable_lastindex_throws() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let replace = proto_method_key(&heap, ctor_obj, WellKnown::Replace);
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "."), s(&mut heap, "g")]);
        heap.define_property_str(
            rx,
            "lastIndex",
            Property {
                value: Value::Int(0),
                writable: false,
                enumerable: false,
                configurable: false,
            },
        )
        .unwrap();
        let args = vec![s(&mut heap, "x"), s(&mut heap, "x")];
        match replace(&mut heap, Value::Object(rx), &args) {
            Err(NativeError::TypeError(_)) => {}
            r => panic!("expected TypeError, got {r:?}"),
        }
        // `lastIndex` is untouched by the failed call.
        assert_eq!(num_val(prop(&heap, rx, "lastIndex")), 0.0);
    }

    /// `@@match` shares the global scan loop: same `TypeError`, no hang.
    #[test]
    fn match_global_nonwritable_lastindex_throws() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let mat = proto_method_key(&heap, ctor_obj, WellKnown::Match);
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "a"), s(&mut heap, "g")]);
        heap.define_property_str(
            rx,
            "lastIndex",
            Property {
                value: Value::Int(0),
                writable: false,
                enumerable: false,
                configurable: false,
            },
        )
        .unwrap();
        let args = vec![s(&mut heap, "aab")];
        match mat(&mut heap, Value::Object(rx), &args) {
            Err(NativeError::TypeError(_)) => {}
            r => panic!("expected TypeError, got {r:?}"),
        }
    }

    /// Spec 22.2.9.8 step 16.p: a match whose position moved backwards
    /// (`position < nextSourcePosition`) has its substitution ignored.
    /// Without the guard, `units[last..a]` panics with
    /// "slice index starts at 1 but ends at 0" on the duplicate match
    /// below (the shape a stuck-`lastIndex` scan produced).
    #[test]
    fn accumulate_replacements_ignores_backwards_match() {
        let mut heap = Heap::new();
        let units: Vec<u16> = Wtf8::from_str("aa").to_units();
        let flags = parse_flags(Wtf8::from_str("g")).unwrap();
        // Two identical matches, as a non-advancing scan would collect.
        let m1 = exec_at(Wtf8::from_str("."), &flags, &units, 0)
            .unwrap()
            .unwrap();
        let m2 = exec_at(Wtf8::from_str("."), &flags, &units, 0)
            .unwrap()
            .unwrap();
        let found = vec![(m1.0, m1.1, m1.2), (m2.0, m2.1, m2.2)];
        assert_eq!(found[0].1, 0);
        assert_eq!(found[0].2, 1);
        let replacer = Replacer::Template(Wtf8Buf::from("X"));
        // Must not panic; the backwards second match is skipped.
        let out =
            accumulate_replacements(&mut heap, &replacer, Wtf8::from_str("aa"), &units, &found)
                .unwrap();
        assert_eq!(out.as_wtf8(), Wtf8::from_str("Xa"));
    }

    #[test]
    fn split_without_match() {
        assert_eq!(
            regexp_split(
                Wtf8::from_str("z"),
                Wtf8::from_str("g"),
                Wtf8::from_str("abc"),
                u32::MAX
            )
            .unwrap(),
            vec![Wtf8Buf::from("abc")]
        );
        assert!(
            regexp_split(
                Wtf8::from_str("z"),
                Wtf8::from_str("g"),
                Wtf8::from_str("abc"),
                0
            )
            .unwrap()
            .is_empty()
        );

        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let split = proto_method_key(&heap, ctor_obj, WellKnown::Split);
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "z")]);
        let r = vcall!(
            &mut heap,
            split,
            Value::Object(rx),
            [s(&mut heap, "abc"), Value::Int(0)]
        )
        .unwrap();
        let arr = match r {
            Value::Object(o) => o,
            v => panic!("expected array, got {v:?}"),
        };
        assert_eq!(num_val(prop(&heap, arr, "length")), 0.0);
    }

    #[test]
    fn match_without_match_is_null() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let m = proto_method_key(&heap, ctor_obj, WellKnown::Match);
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "z")]);
        let r = vcall!(&mut heap, m, Value::Object(rx), [s(&mut heap, "abc")]).unwrap();
        assert_eq!(r, Value::Null);
        let rxg = rx!(&mut heap, ctor, [s(&mut heap, "z"), s(&mut heap, "g")]);
        let r = vcall!(&mut heap, m, Value::Object(rxg), [s(&mut heap, "abc")]).unwrap();
        assert_eq!(r, Value::Null);
    }

    #[test]
    fn match_all_global_without_match_is_empty_array() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let ma = proto_method_key(&heap, ctor_obj, WellKnown::MatchAll);
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "z"), s(&mut heap, "g")]);
        let r = vcall!(&mut heap, ma, Value::Object(rx), [s(&mut heap, "abc")]).unwrap();
        let arr = match r {
            Value::Object(o) => o,
            v => panic!("expected array, got {v:?}"),
        };
        assert_eq!(num_val(prop(&heap, arr, "length")), 0.0);
    }

    #[test]
    fn match_all_rejects_non_global() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let ma = proto_method_key(&heap, ctor_obj, WellKnown::MatchAll);
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "z")]);
        let r = vcall!(&mut heap, ma, Value::Object(rx), [s(&mut heap, "abc")]);
        assert!(matches!(r, Err(NativeError::TypeError(_))));
    }

    #[test]
    fn symbol_search_returns_index_or_minus_one() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let search = proto_method_key(&heap, ctor_obj, WellKnown::Search);
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "b")]);
        let r = vcall!(&mut heap, search, Value::Object(rx), [s(&mut heap, "abc")]).unwrap();
        assert_eq!(num_val(r), 1.0);
        let r = vcall!(&mut heap, search, Value::Object(rx), [s(&mut heap, "xyz")]).unwrap();
        assert_eq!(num_val(r), -1.0);
        // Global: honors `lastIndex` like `RegExpExec`.
        let rxg = rx!(&mut heap, ctor, [s(&mut heap, "b"), s(&mut heap, "g")]);
        let r = vcall!(&mut heap, search, Value::Object(rxg), [s(&mut heap, "abc")]).unwrap();
        assert_eq!(num_val(r), 1.0);
        // Spec 22.2.9.13: the search always starts at 0 and `lastIndex`
        // is restored afterwards, even for global regexps.
        set_last_index(&mut heap, rxg, 2.0).unwrap();
        let r = vcall!(&mut heap, search, Value::Object(rxg), [s(&mut heap, "abc")]).unwrap();
        assert_eq!(num_val(r), 1.0);
        assert_eq!(num_val(get_last_index_val(&heap, rxg)), 2.0);
    }

    #[test]
    fn symbol_search_restores_negative_zero_last_index() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let search = proto_method_key(&heap, ctor_obj, WellKnown::Search);
        // Reviewer repro: `rx.lastIndex = -0; rx[Symbol.search]("");`
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "")]);
        // `js_num(-0.0)` keeps the sign, like `rx.lastIndex = -0`.
        set_last_index(&mut heap, rx, -0.0).unwrap();
        let r = vcall!(&mut heap, search, Value::Object(rx), [s(&mut heap, "")]).unwrap();
        assert_eq!(num_val(r), 0.0);
        // `SameValue`: `-0` must survive the restore, not become `+0`
        // (`Object.is(rx.lastIndex, -0)` must hold).
        match prop(&heap, rx, "lastIndex") {
            Value::Float(f) => assert!(
                f == 0.0 && f.is_sign_negative(),
                "expected -0 lastIndex, got {f}"
            ),
            v => panic!("expected Float(-0) lastIndex, got {v:?}"),
        }
    }

    #[test]
    fn same_value_f64_distinguishes_signed_zero() {
        assert!(same_value_f64(-0.0, -0.0));
        assert!(same_value_f64(0.0, 0.0));
        assert!(!same_value_f64(-0.0, 0.0));
        assert!(same_value_f64(f64::NAN, f64::NAN));
        assert!(!same_value_f64(f64::NAN, 0.0));
        assert!(same_value_f64(1.5, 1.5));
        assert!(!same_value_f64(1.5, 2.5));
    }

    #[test]
    fn last_index_same_value_covers_value_shapes() {
        // The case the old `!=` got wrong: `-0` vs `+0` must restore.
        assert!(!last_index_same_value(
            Value::Float(0.0),
            Value::Float(-0.0)
        ));
        assert!(!last_index_same_value(Value::Float(-0.0), Value::Int(0)));
        assert!(last_index_same_value(
            Value::Float(-0.0),
            Value::Float(-0.0)
        ));
        assert!(last_index_same_value(Value::Int(2), Value::Int(2)));
        assert!(!last_index_same_value(Value::Int(2), Value::Int(3)));
        assert!(last_index_same_value(
            Value::Float(f64::NAN),
            Value::Float(f64::NAN)
        ));
    }

    #[test]
    fn advance_string_index_steps() {
        let units: Vec<u16> = Wtf8::from_str("a").to_units();
        assert_eq!(advance_string_index(&units, 0, false), 1);
        // U+1D11E is a surrogate pair in UTF-16.
        let units: Vec<u16> = Wtf8::from_str("𝄞x").to_units();
        assert_eq!(units.len(), 3);
        assert_eq!(advance_string_index(&units, 0, true), 2);
        assert_eq!(advance_string_index(&units, 0, false), 1);
        assert_eq!(advance_string_index(&units, 2, true), 3);
    }

    // --- Behavioral tests: exercise the real regex engine through the
    // builtins layer. ---

    #[test]
    fn exec_advances_last_index_on_match() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let exec = proto_method(&heap, ctor_obj, "exec");
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "a"), s(&mut heap, "g")]);
        let input = s(&mut heap, "banana");
        let r = vcall!(&mut heap, exec, Value::Object(rx), [input]).unwrap();
        let arr = match r {
            Value::Object(o) => o,
            v => panic!("expected match array, got {v:?}"),
        };
        assert_eq!(as_rust_str(prop(&heap, arr, "0")), "a");
        assert_eq!(num_val(prop(&heap, arr, "index")), 1.0);
        assert_eq!(as_rust_str(prop(&heap, arr, "input")), "banana");
        assert_eq!(num_val(prop(&heap, rx, "lastIndex")), 2.0);
        let r2 = vcall!(&mut heap, exec, Value::Object(rx), [input]).unwrap();
        let arr2 = match r2 {
            Value::Object(o) => o,
            v => panic!("expected match array, got {v:?}"),
        };
        assert_eq!(num_val(prop(&heap, arr2, "index")), 3.0);
    }

    #[test]
    fn sticky_anchors_match_at_last_index() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let exec = proto_method(&heap, ctor_obj, "exec");
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "b"), s(&mut heap, "y")]);
        let input = s(&mut heap, "ab");
        let r = vcall!(&mut heap, exec, Value::Object(rx), [input]).unwrap();
        assert_eq!(r, Value::Null);
        assert_eq!(num_val(prop(&heap, rx, "lastIndex")), 0.0);
        set_last_index(&mut heap, rx, 1.0).unwrap();
        let r = vcall!(&mut heap, exec, Value::Object(rx), [input]).unwrap();
        let arr = match r {
            Value::Object(o) => o,
            v => panic!("expected match array, got {v:?}"),
        };
        assert_eq!(num_val(prop(&heap, arr, "index")), 1.0);
        assert_eq!(num_val(prop(&heap, rx, "lastIndex")), 2.0);
    }

    #[test]
    fn replace_applies_capture_patterns() {
        assert_eq!(
            regexp_replace(
                Wtf8::from_str("(a)(b)?"),
                Wtf8::from_str(""),
                Wtf8::from_str("ab"),
                Wtf8::from_str("$2$1")
            )
            .unwrap()
            .as_wtf8(),
            Wtf8::from_str("ba")
        );
        assert_eq!(
            regexp_replace(
                Wtf8::from_str("a"),
                Wtf8::from_str("g"),
                Wtf8::from_str("banana"),
                Wtf8::from_str("[$&]")
            )
            .unwrap(),
            "b[a]n[a]n[a]"
        );
        assert_eq!(
            regexp_replace(
                Wtf8::from_str("(?<v>[aeiou])"),
                Wtf8::from_str("g"),
                Wtf8::from_str("banana"),
                Wtf8::from_str("<$<v>>")
            )
            .unwrap(),
            "b<a>n<a>n<a>"
        );
        assert!(
            regexp_test(
                Wtf8::from_str("b?n"),
                Wtf8::from_str(""),
                Wtf8::from_str("banana")
            )
            .unwrap()
        );
    }

    fn shout_replacer(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
        let matched = match arg(args, 0) {
            Value::Str(g) => g.as_wtf8().to_string().to_uppercase(),
            v => panic!("replacer got non-string match: {v:?}"),
        };
        assert_eq!(args.len(), 4); // match, offset, string, groups
        Ok(Value::Str(heap.alloc_str(&matched)?))
    }

    #[test]
    fn replace_calls_function_replacer() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let replace = proto_method_key(&heap, ctor_obj, WellKnown::Replace);
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "a"), s(&mut heap, "g")]);
        let replacer = heap
            .alloc_object(Object::new(ObjectKind::NativeFunction(shout_replacer)))
            .unwrap();
        let r = vcall!(
            &mut heap,
            replace,
            Value::Object(rx),
            [s(&mut heap, "banana"), Value::Object(replacer)]
        )
        .unwrap();
        assert_eq!(as_rust_str(r), "bAnAnA");
    }

    #[test]
    fn split_keeps_captures() {
        let parts = regexp_split_raw(
            Wtf8::from_str("(,)"),
            Wtf8::from_str(""),
            Wtf8::from_str("a,b"),
            u32::MAX,
        )
        .unwrap();
        assert_eq!(
            parts,
            vec![
                Some(Wtf8Buf::from("a")),
                Some(Wtf8Buf::from(",")),
                Some(Wtf8Buf::from("b"))
            ]
        );
        assert_eq!(
            regexp_split(
                Wtf8::from_str(":"),
                Wtf8::from_str(""),
                Wtf8::from_str("a:b:c"),
                2
            )
            .unwrap(),
            vec![Wtf8Buf::from("a"), Wtf8Buf::from("b")]
        );
    }

    #[test]
    fn exec_exposes_named_groups() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let exec = proto_method(&heap, ctor_obj, "exec");
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "(?<word>[a-z]+)")]);
        let r = vcall!(
            &mut heap,
            exec,
            Value::Object(rx),
            [s(&mut heap, "hi there")]
        )
        .unwrap();
        let arr = match r {
            Value::Object(o) => o,
            v => panic!("expected match array, got {v:?}"),
        };
        assert_eq!(as_rust_str(prop(&heap, arr, "0")), "hi");
        let groups = match prop(&heap, arr, "groups") {
            Value::Object(o) => o,
            v => panic!("expected groups object, got {v:?}"),
        };
        assert_eq!(str_prop(&heap, groups, "word"), "hi");
    }

    // -- VM re-entry regression tests ------------------------------------

    /// Evaluate JS in a full VM and stringify the completion value.
    fn eval_str(source: &str) -> String {
        let mut vm = crate::vm::Vm::new();
        let v = vm
            .eval(source)
            .unwrap_or_else(|e| panic!("eval failed: {e}"));
        crate::builtin_object_array::to_string_js(vm.heap_mut(), v)
            .map(|w| w.to_string())
            .unwrap_or_else(|e| panic!("to_string failed: {e:?}"))
    }

    #[test]
    fn symbol_replace_accepts_bytecode_replacer() {
        // The well-known-symbol methods live under reserved string keys
        // until the object model supports symbol keys. `new` temporaries
        // cannot receive method calls yet (separate VM gap), so bind first.
        let r = eval_str(
            r#"let rx = new RegExp("b"); rx[Symbol.replace]("abc", (m) => m.toUpperCase())"#,
        );
        assert_eq!(r, "aBc");
    }

    #[test]
    fn symbol_replace_replacer_receives_match_args() {
        let r = eval_str(
            r#"let rx = new RegExp("(\\w+)@(\\w+)"); rx[Symbol.replace]("a@b", (m, u, d, off) => d + "+" + u + "@" + off)"#,
        );
        assert_eq!(r, "b+a@0");
    }

    #[test]
    fn symbol_replace_propagates_replacer_throw() {
        let mut vm = crate::vm::Vm::new();
        let err = vm
            .eval(r#"let rx = new RegExp("b"); rx[Symbol.replace]("abc", () => { throw "nope"; })"#)
            .unwrap_err();
        assert!(
            matches!(err, crate::vm::VmError::Thrown(ref s) if s == "nope"),
            "unexpected error: {err}"
        );
    }

    // -- Compile-cache tests ---------------------------------------------

    #[test]
    fn lone_surrogate_pattern_matches_exactly() {
        // The pattern text `String.fromCharCode(0xD800)` produces: WTF-8
        // bytes holding the lone-surrogate CESU-8 sequence. It must match
        // a lone-surrogate subject and must NOT match U+FFFD (the old
        // lossy `to_utf8_lossy` compilation target).
        let pattern = Wtf8Buf::from_units(&[0xD800]);
        let flags = Flags::default();
        let re = compile_cached(pattern.as_wtf8(), "").unwrap();
        let lone: Vec<u16> = vec![0xD800];
        let (mt, a, b) = exec_at(pattern.as_wtf8(), &flags, &lone, 0)
            .unwrap()
            .expect("lone-surrogate pattern must match lone-surrogate subject");
        assert_eq!((a, b), (0, 1));
        assert_eq!(
            group_texts(&lone, &mt)[0].as_deref(),
            Some(pattern.as_wtf8())
        );
        let fffd: Vec<u16> = vec![0xFFFD];
        assert!(
            exec_at(pattern.as_wtf8(), &flags, &fffd, 0)
                .unwrap()
                .is_none(),
            "lone-surrogate pattern must not match U+FFFD"
        );
        // The cache keys on exact WTF-8 bytes: a U+FFFD pattern is a
        // distinct entry with distinct behavior.
        let fffd_pat = Wtf8::from_str("\u{FFFD}");
        let re2 = compile_cached(fffd_pat, "").unwrap();
        assert!(!Rc::ptr_eq(&re, &re2));
        assert!(exec_at(fffd_pat, &flags, &fffd, 0).unwrap().is_some());
        assert!(exec_at(fffd_pat, &flags, &lone, 0).unwrap().is_none());
    }

    #[test]
    fn unicode_mode_unchanged_for_well_formed_patterns() {
        // Regression: valid-UTF-8 patterns behave exactly as before —
        // astral characters are single code-point atoms in `u` mode.
        let re = compile_cached(Wtf8::from_str("😀"), "u").unwrap();
        let units: Vec<u16> = Wtf8::from_str("a😀b").to_units();
        let flags = parse_flags(Wtf8::from_str("u")).unwrap();
        let (mt, a, b) = exec_at(Wtf8::from_str("😀"), &flags, &units, 0)
            .unwrap()
            .expect("astral pattern must match");
        assert_eq!((a, b), (1, 3));
        assert_eq!(
            group_texts(&units, &mt)[0].as_deref(),
            Some(Wtf8::from_str("😀"))
        );
        drop(re);
    }

    #[test]
    fn compile_cache_returns_one_object_per_pattern() {
        // Same (source, flags): identical `Rc` — the pattern compiled once.
        let a = compile_cached(Wtf8::from_str("ab+c"), "gi").unwrap();
        let b = compile_cached(Wtf8::from_str("ab+c"), "gi").unwrap();
        assert!(Rc::ptr_eq(&a, &b));
        // Different flags: a different entry.
        let c = compile_cached(Wtf8::from_str("ab+c"), "g").unwrap();
        assert!(!Rc::ptr_eq(&a, &c));
        // Compile failures are not cached, and still surface as errors.
        assert!(compile_cached(Wtf8::from_str("("), "").is_err());
        assert!(compile_cached(Wtf8::from_str("("), "").is_err());
    }

    #[test]
    fn compile_cache_is_bounded() {
        for i in 0..COMPILE_CACHE_CAP + 32 {
            compile_cached(Wtf8::from_str(&format!("cache_bound_test_{i}")), "").unwrap();
        }
        assert!(compile_cache_len() <= COMPILE_CACHE_CAP);
    }

    #[test]
    fn evicted_pattern_still_matches() {
        let source = "evict_me_[0-9]+";
        assert!(compile_cached(Wtf8::from_str(source), "").is_ok());
        // Flood the cache so the entry above is evicted...
        for i in 0..COMPILE_CACHE_CAP + 8 {
            compile_cached(Wtf8::from_str(&format!("evict_flood_{i}")), "").unwrap();
        }
        // ...then matching recompiles transparently and stays correct.
        let input = "xx evict_me_42 yy";
        let units: Vec<u16> = Wtf8::from_str(input).to_units();
        let flags = Flags::default();
        let (mt, a, b) = exec_at(Wtf8::from_str(source), &flags, &units, 0)
            .unwrap()
            .expect("evicted pattern should still match");
        assert_eq!(
            group_texts(&units, &mt)[0].as_deref(),
            Some(Wtf8::from_str("evict_me_42"))
        );
        assert_eq!((a, b), (3, 14));
    }

    #[test]
    fn repeated_exec_reuses_compiled_pattern_and_tracks_last_index() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let exec = proto_method(&heap, ctor_obj, "exec");
        let test = proto_method(&heap, ctor_obj, "test");
        let rx = rx!(&mut heap, ctor, [s(&mut heap, "a+"), s(&mut heap, "g")]);
        let input = s(&mut heap, "baa baaa ba");
        // Repeated execs stay correct while `lastIndex` advances; every
        // call reuses the one cached program for ("a+", "g").
        let mut indices = Vec::new();
        for _ in 0..10 {
            let r = vcall!(&mut heap, exec, Value::Object(rx), [input]).unwrap();
            match r {
                Value::Null => break,
                Value::Object(arr) => indices.push(num_val(prop(&heap, arr, "index"))),
                v => panic!("unexpected exec result: {v:?}"),
            }
        }
        assert_eq!(indices, vec![1.0, 5.0, 10.0]);
        assert_eq!(num_val(prop(&heap, rx, "lastIndex")), 0.0);
        // And `test` on a same-pattern object is consistent too.
        let rx2 = rx!(&mut heap, ctor, [s(&mut heap, "a+"), s(&mut heap, "g")]);
        assert_eq!(
            vcall!(&mut heap, test, Value::Object(rx2), [input]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            vcall!(&mut heap, test, Value::Object(rx2), [input]).unwrap(),
            Value::Bool(true)
        );
    }

    // -- Regression tests: builtin result arrays inherit Array.prototype --
    //
    // `RegExp.prototype.exec` results (plus `match`/`split`-with-regexp
    // and the `d`-flag index pairs) are `CreateArrayFromList` arrays, so
    // they must inherit `%Array.prototype%` (a null-prototype regression
    // once made `result instanceof Array === false`).

    #[test]
    fn regexp_result_arrays_are_real_arrays() {
        // `exec` results (and the `d`-flag index pairs) are
        // `CreateArrayFromList` arrays: they must inherit
        // `%Array.prototype%` (a null-prototype regression once made
        // `result instanceof Array === false`). Tested at the native
        // level: method calls on regexp objects are a separate VM gap.
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (ctor_obj, ctor) = regexp_ctor(&heap, global);
        let exec = proto_method(&heap, ctor_obj, "exec");
        // The regexp test harness installs only the RegExp intrinsics; the
        // real VM installs `%Array.prototype%` first (builtin_object_array),
        // so simulate that post-init state here.
        let array_proto = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.array_prototype = Some(array_proto);
        for (pattern, flags) in [("b(.)", ""), ("b(.)", "d")] {
            let rx = rx!(
                &mut heap,
                ctor,
                [s(&mut heap, pattern), s(&mut heap, flags)]
            );
            let r = vcall!(&mut heap, exec, Value::Object(rx), [s(&mut heap, "abc")]).unwrap();
            let arr = match r {
                Value::Object(o) => o,
                v => panic!("expected match array, got {v:?}"),
            };
            assert_eq!(
                heap.get(arr).prototype,
                Value::Object(array_proto),
                "exec result of /{pattern}/{flags} must inherit Array.prototype"
            );
            if !flags.is_empty() {
                let indices = match prop(&heap, arr, "indices") {
                    Value::Object(o) => o,
                    v => panic!("expected indices array, got {v:?}"),
                };
                assert_eq!(
                    heap.get(indices).prototype,
                    Value::Object(array_proto),
                    "d-flag indices pair must inherit Array.prototype"
                );
            }
        }
        // The regexp `split` path builds the same way; reachable from the VM.
        assert_eq!(eval_str(r"'a,b'.split(/,/).map(x => x).length"), "2");
    }
}

/// Manual benchmarks (no external deps per AGENTS.md). Run:
/// `CARGO_TARGET_DIR=~/workspace/scratch/bowser-regexp-opt2/target cargo test
/// -p bowser_js --release builtin_regexp::bench -- --nocapture`.
/// They print ns/op; they assert nothing and must stay fast enough to run
/// inside the normal `cargo test` suite. Warmup and timing are
/// wall-clock bounded (see `timeit`), so they stay fast in debug builds
/// as well as release.
#[cfg(test)]
mod bench {
    use super::*;
    use std::time::{Duration, Instant};

    fn timeit(name: &str, iters: u64, mut f: impl FnMut()) {
        // Benches must stay fast inside the normal `cargo test` suite in
        // every profile: a debug build runs this engine an order of
        // magnitude slower than release, and the heavy benches below once
        // stalled the suite past every timeout (looking hung). Bound the
        // warmup and the timed section by wall-clock; `iters` is now only
        // a cap, so release behavior (which hits the cap) is unchanged.
        let warm_t0 = Instant::now();
        let mut w = 0u64;
        while w < iters.min(1000) && warm_t0.elapsed() < Duration::from_secs(2) {
            f();
            w += 1;
        }
        let t0 = Instant::now();
        let mut n = 0u64;
        while n < iters && t0.elapsed() < Duration::from_secs(5) {
            f();
            n += 1;
        }
        let n = n.max(1);
        let ns = t0.elapsed().as_nanos() as f64 / n as f64;
        println!("bench {name}: {ns:.1} ns/op ({n} iters)");
    }

    fn flags(s: &str) -> Flags {
        parse_flags(Wtf8::from_str(s)).unwrap()
    }

    #[test]
    fn bench_compile_cache_hit() {
        timeit("compile_cache_hit", 200_000, || {
            std::hint::black_box(
                compile_cached(Wtf8::from_str("a+b"), &flags_string(&flags(""))).unwrap(),
            );
        });
    }

    #[test]
    fn bench_compile_cache_hit_flags() {
        // Hot-path entry: validated `Flags`, no flags-string allocation.
        let f = flags("");
        timeit("compile_cache_hit_flags", 200_000, || {
            std::hint::black_box(compile_cached_flags(Wtf8::from_str("a+b"), &f).unwrap());
        });
    }

    #[test]
    fn bench_run_exec_plain() {
        let f = flags("");
        let input = "x".repeat(200) + "aaab";
        let units: Vec<u16> = Wtf8::from_str(&input).to_units();
        timeit("run_exec_plain", 50_000, || {
            let m = run_exec(Wtf8::from_str("a+b"), &f, &units, 0).unwrap();
            std::hint::black_box(m);
        });
    }

    #[test]
    fn bench_run_exec_named_groups() {
        let f = flags("");
        let input = "x".repeat(200) + "aaab";
        let units: Vec<u16> = Wtf8::from_str(&input).to_units();
        timeit("run_exec_named", 50_000, || {
            let m = run_exec(Wtf8::from_str("(?<letters>a+)(?<tail>b)"), &f, &units, 0).unwrap();
            std::hint::black_box(m);
        });
    }

    #[test]
    fn bench_test_fn_full_builtin_path() {
        let mut heap = Heap::new();
        let rx = make_regexp(&mut heap, Wtf8::from_str("a+b"), &flags("")).unwrap();
        let input = Value::Str(heap.alloc_str(&("x".repeat(200) + "aaab")).unwrap());
        let args = [input];
        timeit("test_fn_full_path", 50_000, || {
            std::hint::black_box(regexp_test_fn(&mut heap, Value::Object(rx), &args).unwrap());
        });
    }

    #[test]
    fn bench_replace_global_template() {
        let input = "ab ".repeat(500);
        timeit("replace_global_dollar_amp", 2_000, || {
            std::hint::black_box(
                regexp_replace(
                    Wtf8::from_str("ab"),
                    Wtf8::from_str("g"),
                    Wtf8::from_str(&input),
                    Wtf8::from_str("<$&>"),
                )
                .unwrap(),
            );
        });
    }

    #[test]
    fn bench_replace_global_named_template() {
        let input = "ab ".repeat(500);
        timeit("replace_global_named", 2_000, || {
            std::hint::black_box(
                regexp_replace(
                    Wtf8::from_str("(?<x>a)(?<y>b)"),
                    Wtf8::from_str("g"),
                    Wtf8::from_str(&input),
                    Wtf8::from_str("<$<x>$<y>>"),
                )
                .unwrap(),
            );
        });
    }

    #[test]
    fn bench_split() {
        let input = "ab,".repeat(500);
        timeit("split", 2_000, || {
            std::hint::black_box(
                regexp_split(
                    Wtf8::from_str(","),
                    Wtf8::from_str(""),
                    Wtf8::from_str(&input),
                    u32::MAX,
                )
                .unwrap(),
            );
        });
    }

    #[test]
    fn bench_exec_result_array() {
        let mut heap = Heap::new();
        let rx = make_regexp(
            &mut heap,
            Wtf8::from_str("(?<w>\\w+)@(?<d>[\\w.]+)"),
            &flags(""),
        )
        .unwrap();
        let input = Value::Str(heap.alloc_str("contact bob@example.com today").unwrap());
        let args = [input];
        timeit("exec_result_array", 20_000, || {
            std::hint::black_box(regexp_exec_fn(&mut heap, Value::Object(rx), &args).unwrap());
        });
    }
}
