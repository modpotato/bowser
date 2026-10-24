//! `Object` and `Array` builtins.
//!
//! Implements the `Object` constructor plus its static methods and the
//! `Array` constructor, statics, and prototype methods, then installs them
//! on the global object via [`install`].
//!
//! ## Array storage (v1 decision)
//!
//! Arrays are [`ObjectKind::Array`] objects whose indexed elements live as
//! ordinary properties named `"0"`, `"1"`, ... plus a `"length"` data
//! property. There is no packed backing store: every indexed access is a
//! property lookup, so some operations are O(n) that a dense array would do
//! in O(1). A packed-elements representation is future work; the public
//! helper surface (`get_idx`, `set_idx`, `array_length`, ...) is the seam a
//! packed representation would reimplement.
//!
//! Holes are simply absent properties: `get_idx` returns `undefined` for a
//! missing index, `has_idx` distinguishes holes, and methods that must
//! preserve holes (`slice`, `splice`, `toReversed`, ...) copy presence, not
//! just values.
//!
//! ## Length invariant
//!
//! The `"length"` property is a non-configurable, non-enumerable, writable
//! data property. `set_idx` on an `Array`-kind object extends it when the
//! index is at or past the end; `set_len` truncates (deleting indices) or
//! extends it. All prototype methods go through these helpers.
//!
//! ## Gaps vs. the sibling contracts (for the vm worker)
//!
//! - **VM re-entry is wired.** [`call_function`] invokes
//!   [`ObjectKind::NativeFunction`] callees directly and re-enters the VM
//!   for bytecode ([`ObjectKind::Function`]) callees via
//!   [`crate::vm::reenter_call`].
//! - **No `ToPrimitive` dispatch.** `to_number` on objects yields `NaN`;
//!   `to_string_js` on objects tries a callable native `toString`, else
//!   `"[object Object]"`.
//! - **No iterators.** `keys`/`values`/`entries` return fresh `Array`s of
//!   indices / values / `[index, value]` pairs instead of iterator objects.
//! - **No symbols as property keys in v1 paths.** `to_property_key` is
//!   string-only; `Symbol` values throw `TypeError` on string coercion.
//! - **Accessor properties.** Getters/setters work via object literals,
//!   classes, `Object.defineProperty`/`defineProperties`, and the
//!   `#define_accessor` intrinsic; they live in a side table on `Object`
//!   so the data `Property` shape stays untouched.
//! - **`ArraySpeciesCreate`, no `Symbol.isConcatSpreadable`.**
//!   `map`/`filter`/`slice`/`splice`/`concat`/`flat`/`flatMap` build
//!   their results through `ArraySpeciesCreate` (with `%Array%`'s
//!   `get [Symbol.species]` installed, so subclass constructors
//!   propagate); `toSorted`/`toReversed`/`toSpliced` deliberately use
//!   plain `ArrayCreate` (no species) per ECMA-262; `concat` spreads
//!   `isArray` arguments only.
//! - **`new.target` is invisible.** `NativeFn` cannot distinguish `Object()`
//!   from `new Object()`; both take the construct path.
//! - **Integrity without flags.** `Object` carries no sealed/frozen flags,
//!   so `isFrozen`/`isSealed` compute the spec `[[TestIntegrityLevel]]`
//!   predicate from `extensible` plus property attributes.
//! - **Intrinsics on the heap.** `NativeFn` receives no runtime context, so
//!   `install` stashes `%Object.prototype%` / `%Array.prototype%` in two
//!   `pub` fields on [`Heap`](crate::gc::Heap) (added by this worker; the
//!   function worker will add `%Function.prototype%` the same way).
//!
//! ## Shared-helper pattern (for the other builtins workers)
//!
//! The `pub(crate)` helpers in this file (`get_arg`,
//! `get_arg_string`, `num`, the `to_*` conversions, the property helpers,
//! `call_function`) are written to be copied verbatim into sibling builtin
//! modules: they assume only `&mut Heap` context. Reads go through
//! `heap.get` and never cross a `&mut Heap` call (the borrow checker
//! enforces this via `get`'s signature); the arena never collects, so no
//! rooting is needed in v1.

use crate::builtin_reflect_proxy::{
    as_proxy, define_hidden, descriptor_object, get_prop_chain, new_native_constructor,
    ordinary_set, proxy_define_property, proxy_set, reflect_construct_inner, species_constructor,
};
use crate::builtin_string_number::{
    define_native, new_native_function, string_to_number, with_temp_root,
};
use crate::gc::{
    AllocError, ArrayDefineDesc, ElemKey, Gc, Heap, PropKey, WellKnown, dense_is_active,
    u64_to_value,
};
use crate::value::{AccessorProp, NativeError, NativeFn, Object, ObjectKind, Property, Str, Value};
use crate::vm::active_new_target;
use crate::vm::{ChainProp, find_prop_in_chain};
use crate::wtf8::{Wtf8, Wtf8Buf};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Argument `i`, or `undefined` when the caller passed fewer.
pub(crate) fn get_arg(args: &[Value], i: usize) -> Value {
    args.get(i).copied().unwrap_or(Value::Undefined)
}

/// Argument `i` coerced with `ToString`.
//
// (Not yet called by a builtin; upcoming string builtins take string args.
// The allow keeps `cargo clippy -- -D warnings` green until then.)
#[allow(dead_code)]
pub(crate) fn get_arg_string(
    heap: &mut Heap,
    args: &[Value],
    i: usize,
) -> Result<Wtf8Buf, NativeError> {
    to_string_js(heap, get_arg(args, i))
}

/// Build the cheapest numeric value: [`Value::Int`] when `n` is integral and
/// fits `i32` (preserving `-0` as `Float`), else [`Value::Float`].
pub(crate) fn num(n: f64) -> Value {
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

/// True when a value is callable: a native or bytecode function object.
pub(crate) fn is_callable(heap: &Heap, v: Value) -> bool {
    match v {
        Value::Object(o) => matches!(
            heap.get(o).kind,
            ObjectKind::NativeFunction(_) | ObjectKind::Function(_)
        ),
        _ => false,
    }
}

/// Extract the receiver as an object, boxing primitives per `ToObject`.
///
/// `null`/`undefined` throw `TypeError` (spec 7.1.18). Strings box into
/// a fresh ordinary object carrying the String-exotic observable shape
/// the array builtins read: a non-writable `length` in UTF-16 code
/// units and one single-character string per index (lone surrogates
/// surface as U+FFFD, the engine's v1 string policy — a Rust `String`
/// cannot hold them). Numbers, booleans, bigints, and symbols box into
/// an empty ordinary object: the array builtins then see no `length`,
/// exactly as with the spec wrapper objects.
pub(crate) fn this_object(heap: &mut Heap, this: Value) -> Result<Gc<Object>, NativeError> {
    match this {
        Value::Object(o) => Ok(o),
        Value::Undefined | Value::Null => Err(NativeError::TypeError(
            "method called on null or undefined".into(),
        )),
        Value::Str(s) => box_string_primitive(heap, s),
        _ => {
            let proto = heap
                .object_prototype
                .map(Value::Object)
                .unwrap_or(Value::Null);
            Ok(heap.alloc_object(Object {
                kind: ObjectKind::Ordinary,
                prototype: proto,
                ..Default::default()
            })?)
        }
    }
}

/// `ToObject` for a string primitive: a fresh ordinary object with the
/// indexed characters and `length` a String exotic would expose.
fn box_string_primitive(heap: &mut Heap, s: Str) -> Result<Gc<Object>, NativeError> {
    // Hold the source string across the loop: every index allocates and
    // may collect, and the `&str` below borrows its heap allocation.
    let sguard = heap.protect(Value::Str(s));
    let text = s.as_wtf8();
    let proto = heap
        .object_prototype
        .map(Value::Object)
        .unwrap_or(Value::Null);
    let obj = heap.alloc_object(Object {
        kind: ObjectKind::Ordinary,
        prototype: proto,
        ..Default::default()
    })?;
    let oguard = heap.protect(Value::Object(obj));
    let result = (|| -> Result<(), NativeError> {
        // UTF-16 code-unit length, matching `String.prototype.length`.
        // ASCII fast path: code units are bytes.
        let len = if text.is_ascii() {
            text.len()
        } else {
            text.to_units().len()
        };
        heap.define_property_str(
            obj,
            "length",
            Property {
                value: num(len as f64),
                writable: false,
                enumerable: false,
                configurable: false,
            },
        )?;
        if text.is_ascii() {
            for (i, b) in text.as_bytes().iter().enumerate() {
                let ch = (*b as char).to_string();
                let v = Value::Str(heap.alloc_str(&ch)?);
                heap.define_property_str(obj, &i.to_string(), Property::data(v))?;
            }
        } else {
            for (i, unit) in text.to_units().iter().enumerate() {
                // One code unit per index property; lone surrogates are
                // preserved (the old `from_utf16_lossy` produced U+FFFD).
                let v = Value::Str(heap.alloc_wtf8_owned(Wtf8Buf::from_units(&[*unit]))?);
                heap.define_property_str(obj, &i.to_string(), Property::data(v))?;
            }
        }
        Ok(())
    })();
    heap.release_temp(oguard);
    heap.release_temp(sguard);
    result?;
    Ok(obj)
}

// ---------------------------------------------------------------------------
// Number <-> string conversions (spec algorithms)
// ---------------------------------------------------------------------------

/// Sink for the shared [`js_number_to_string`] core below: the heap
/// `String` path and the stack-buffer path write through this so the
/// digit/point/exponent placement logic exists exactly once.
trait NumStrSink {
    fn push(&mut self, c: char);
    fn push_str(&mut self, s: &str);
    fn push_i64(&mut self, v: i64);
}

impl NumStrSink for String {
    #[inline]
    fn push(&mut self, c: char) {
        String::push(self, c);
    }
    #[inline]
    fn push_str(&mut self, s: &str) {
        String::push_str(self, s);
    }
    #[inline]
    fn push_i64(&mut self, v: i64) {
        String::push_str(self, &v.to_string());
    }
}

/// Stack-buffer [`NumStrSink`]. Every byte the number core emits is ASCII
/// (digits, `.`, `e`, `+`, `-`), so `push(char)` is a single byte store.
/// Sized by the caller for the worst case: the core emits at most 25
/// bytes (`-0.00000` followed by 17 digits).
struct StackNumStr<'a> {
    buf: &'a mut [u8],
    len: usize,
}

impl NumStrSink for StackNumStr<'_> {
    #[inline]
    fn push(&mut self, c: char) {
        // Release-mode assert: the `c as u8` truncation below is only
        // sound for ASCII, and the number core's output must stay ASCII
        // for the `from_utf8_unchecked` in `js_number_to_str`. Fail-stop
        // here rather than silently corrupting the buffer if a future
        // caller pushes non-ASCII.
        assert!(c.is_ascii());
        assert!(self.len < self.buf.len());
        self.buf[self.len] = c as u8;
        self.len += 1;
    }
    #[inline]
    fn push_str(&mut self, s: &str) {
        let end = self.len + s.len();
        assert!(end <= self.buf.len());
        self.buf[self.len..end].copy_from_slice(s.as_bytes());
        self.len = end;
    }
    #[inline]
    fn push_i64(&mut self, v: i64) {
        // Only the decimal exponent reaches here (|v| <= 324), but format
        // it generally: digits reversed into a small temp, then emitted in
        // order — no heap allocation.
        let mut n = v;
        if n < 0 {
            self.push('-');
            n = -n;
        }
        let mut tmp = [0u8; 20];
        let mut tlen = 0;
        if n == 0 {
            tmp[0] = b'0';
            tlen = 1;
        } else {
            while n > 0 {
                tmp[tlen] = b'0' + (n % 10) as u8;
                tlen += 1;
                n /= 10;
            }
        }
        for i in (0..tlen).rev() {
            self.push(tmp[i] as char);
        }
    }
}

/// Push `digits[a..b]` (window-relative; see [`js_number_format`]) without
/// materializing the concatenated digit string.
fn push_num_digits(
    int_part: &str,
    frac_part: &str,
    first: usize,
    out: &mut impl NumStrSink,
    a: usize,
    b: usize,
) {
    let il = int_part.len();
    let s = first + a;
    let e = first + b;
    if s < il {
        let me = e.min(il);
        out.push_str(&int_part[s..me]);
        if e > il {
            out.push_str(&frac_part[..e - il]);
        }
    } else {
        out.push_str(&frac_part[s - il..e - il]);
    }
}

/// Shared `Number::toString` core over `full`, the shortest-digits decimal
/// expansion of `|n|` with no exponent (Rust's `Display` for `f64`). Places
/// the point or switches to exponential form. The digit-window logic is
/// verbatim the historical implementation; only the output goes through
/// the sink, and the `int_part ++ frac_part` concatenation is addressed
/// virtually instead of `format!`ed into a transient `String`.
fn js_number_format(full: &str, out: &mut impl NumStrSink) {
    let (int_part, frac_part) = match full.split_once('.') {
        Some((i, f)) => (i, f),
        None => (full, ""),
    };
    let int_len = int_part.len();
    let ib = int_part.as_bytes();
    let fb = frac_part.as_bytes();
    let total = ib.len() + fb.len();
    let byte_at = |i: usize| -> u8 {
        if i < ib.len() {
            ib[i]
        } else {
            fb[i - ib.len()]
        }
    };
    let mut first = 0;
    while first < total && byte_at(first) == b'0' {
        first += 1;
    }
    let mut last = total;
    while last > first && byte_at(last - 1) == b'0' {
        last -= 1;
    }
    let count = (last - first) as i64;
    // k with 10^(k-1) <= m < 10^k.
    let k = int_len as i64 - first as i64;
    if k >= 1 && k <= count {
        // Decimal point inside the digits: "123.456".
        push_num_digits(int_part, frac_part, first, out, 0, k as usize);
        if k < count {
            out.push('.');
            push_num_digits(int_part, frac_part, first, out, k as usize, count as usize);
        }
    } else if (1..=21).contains(&k) {
        // Integer with trailing zeros: "100".
        push_num_digits(int_part, frac_part, first, out, 0, count as usize);
        for _ in 0..(k - count) {
            out.push('0');
        }
    } else if k > -6 {
        // Small fraction: "0.0001".
        out.push_str("0.");
        for _ in 0..-k {
            out.push('0');
        }
        push_num_digits(int_part, frac_part, first, out, 0, count as usize);
    } else {
        // Exponential: "1e+21", "1.5e-7".
        push_num_digits(int_part, frac_part, first, out, 0, 1);
        if count > 1 {
            out.push('.');
            push_num_digits(int_part, frac_part, first, out, 1, count as usize);
        }
        out.push('e');
        let exp = k - 1;
        if exp >= 0 {
            out.push('+');
        }
        out.push_i64(exp);
    }
}

/// `Number::toString`: shortest-roundtrip digits with the spec's
/// decimal/exponential selection. Rust's `Display` for `f64` already yields
/// shortest digits in full decimal expansion, so this only has to place the
/// point or switch to exponential form.
///
/// Kept for call sites that genuinely need an owned `String`; hot paths
/// that immediately intern the result use [`js_number_to_str`] instead.
pub(crate) fn js_number_to_string(n: f64) -> String {
    if n.is_nan() {
        return "NaN".to_owned();
    }
    if n == 0.0 {
        // Covers -0.0: `(-0).toString() === "0"`.
        return "0".to_owned();
    }
    if n.is_infinite() {
        return if n > 0.0 {
            "Infinity".to_owned()
        } else {
            "-Infinity".to_owned()
        };
    }
    let mut out = String::new();
    if n < 0.0 {
        out.push('-');
    }
    // Shortest digits, full decimal expansion, no exponent.
    let full = format!("{}", n.abs());
    js_number_format(&full, &mut out);
    out
}

/// Worst case for `format!("{}", f64)`: 342 bytes (`4.9406564584124654e-324`
/// expands to `0.` + 323 zeros + 17 digits). 1 KiB of stack scratch leaves
/// ample margin; the `debug_assert!` in [`js_number_to_str`] pins the bound.
const F64_DECIMAL_EXPANSION_MAX: usize = 1024;

/// `Number::toString` into a stack buffer: byte-identical to
/// [`js_number_to_string`], zero heap allocation. The intermediate decimal
/// expansion lives in a stack scratch buffer; the result needs at most 25
/// bytes, so a `[u8; 32]` suffices.
///
/// For hot paths that immediately [`Heap::alloc_str`] the result (integer
/// and float `ToString`, template literals, `s += x` appends): short
/// results go inline, so the transient `String` the old path malloc'd was
/// pure waste.
pub(crate) fn js_number_to_str(n: f64, buf: &mut [u8; 32]) -> &str {
    fn lit<'a>(buf: &'a mut [u8; 32], s: &str) -> &'a str {
        buf[..s.len()].copy_from_slice(s.as_bytes());
        // SAFETY: the literals below are ASCII.
        unsafe { core::str::from_utf8_unchecked(&buf[..s.len()]) }
    }
    if n.is_nan() {
        return lit(buf, "NaN");
    }
    if n == 0.0 {
        // Covers -0.0: `(-0).toString() === "0"`.
        return lit(buf, "0");
    }
    if n.is_infinite() {
        return lit(buf, if n > 0.0 { "Infinity" } else { "-Infinity" });
    }
    // Shortest digits, full decimal expansion, no exponent — on the stack.
    let mut tmp = [0u8; F64_DECIMAL_EXPANSION_MAX];
    let full: &str = {
        struct Expansion<'b> {
            buf: &'b mut [u8],
            len: usize,
        }
        impl core::fmt::Write for Expansion<'_> {
            fn write_str(&mut self, s: &str) -> core::fmt::Result {
                let end = self.len + s.len();
                if end > self.buf.len() {
                    return Err(core::fmt::Error);
                }
                self.buf[self.len..end].copy_from_slice(s.as_bytes());
                self.len = end;
                Ok(())
            }
        }
        use core::fmt::Write as _;
        let full_len = {
            let mut w = Expansion {
                buf: &mut tmp,
                len: 0,
            };
            // Identical bytes to `format!("{}", n.abs())`: same `Display`.
            write!(w, "{}", n.abs()).expect("f64 decimal expansion exceeds 1 KiB");
            debug_assert!(
                w.len <= 342,
                "f64 decimal expansion longer than the 342-byte bound"
            );
            w.len
        };
        // SAFETY: `Display` for `f64` emits ASCII digits and at most one '.'.
        unsafe { core::str::from_utf8_unchecked(&tmp[..full_len]) }
    };
    let mut sink = StackNumStr {
        buf: &mut buf[..],
        len: 0,
    };
    if n < 0.0 {
        sink.push('-');
    }
    js_number_format(full, &mut sink);
    let len = sink.len;
    debug_assert!(len <= 32, "number formatting exceeded the 32-byte buffer");
    // SAFETY: every byte the core emits is ASCII (digits, '.', 'e', '+', '-').
    unsafe { core::str::from_utf8_unchecked(&buf[..len]) }
}

/// `i32` → decimal into a stack buffer (`[u8; 11]` covers `i32::MIN`).
/// Zero heap allocation: the `Value::Int` arms of [`to_string_js_str`]
/// used to `i.to_string()` (a malloc) only to immediately `alloc_str` the
/// result — and every integer stringifies to at most 11 bytes, so it
/// always goes inline and the malloc was pure waste.
pub(crate) fn int_to_str(v: i32, buf: &mut [u8; 11]) -> &str {
    // `i64` so `i32::MIN` negates without overflow.
    let mut n = i64::from(v);
    let neg = n < 0;
    if neg {
        n = -n;
    }
    let mut end = buf.len();
    if n == 0 {
        end -= 1;
        buf[end] = b'0';
    } else {
        while n > 0 {
            end -= 1;
            buf[end] = b'0' + (n % 10) as u8;
            n /= 10;
        }
    }
    if neg {
        end -= 1;
        buf[end] = b'-';
    }
    // SAFETY: only ASCII digits and '-' were written above.
    unsafe { core::str::from_utf8_unchecked(&buf[end..]) }
}

/// `ToNumber` applied to a string. Delegates to
/// [`string_to_number`](crate::builtin_string_number::string_to_number) so
/// every string→number conversion in the engine agrees (e.g.
/// `Number('0b101')` and `+'0b101'` both yield `5`).
#[allow(dead_code)]
pub(crate) fn parse_js_number(s: &Wtf8) -> f64 {
    string_to_number(s)
}

/// `ToNumber`. Needs no heap: string contents are read through the handle.
/// v1 gap: objects coerce to `NaN` (no `ToPrimitive` dispatch).
pub(crate) fn to_number(v: Value) -> Result<f64, NativeError> {
    match v {
        Value::Undefined => Ok(f64::NAN),
        Value::Null => Ok(0.0),
        Value::Bool(b) => Ok(if b { 1.0 } else { 0.0 }),
        Value::Int(i) => Ok(f64::from(i)),
        Value::Float(f) => Ok(f),
        Value::Str(s) => Ok(string_to_number(s.as_wtf8())),
        Value::Symbol(_) => Err(NativeError::TypeError(
            "cannot convert symbol to number".into(),
        )),
        Value::BigInt(_) => Err(NativeError::TypeError(
            "cannot convert bigint to number".into(),
        )),
        Value::Object(_) => Ok(f64::NAN),
    }
}

/// Decimal rendering of a [`BigInt`] (sign plus base-2^32 magnitude,
/// least significant digit first): repeated division by 10^9, most
/// significant chunk unpadded, the rest zero-padded to 9 digits.
fn bigint_to_decimal(negative: bool, digits: &[u32]) -> String {
    let mut m: Vec<u32> = digits.to_vec();
    while m.last() == Some(&0) {
        m.pop();
    }
    if m.is_empty() {
        return "0".to_owned();
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

/// Hint selecting the `OrdinaryToPrimitive` method order (and the hint
/// string handed to `@@toPrimitive`).
#[derive(Clone, Copy)]
pub(crate) enum PrimitiveHint {
    /// `"number"`: `valueOf` first.
    Number,
    /// `"string"`: `toString` first. No builtin call path currently
    /// constructs this (`to_string_js` inlines its own single-walk
    /// version in `to_primitive_string`); kept so
    /// `to_primitive_with_hint` still models all three spec hints.
    #[allow(dead_code)]
    String,
    /// `"default"`: handed to `@@toPrimitive` as `"default"`; the ordinary
    /// method order is `valueOf` first (like `Number`).
    Default,
}

/// `ToPrimitive(input, hint)` for builtin call paths: consults the
/// `@@toPrimitive` hook first (hint name `"number"`/`"string"`), then
/// `OrdinaryToPrimitive` in hint order. Returns a primitive value, or a
/// `TypeError` when no method yields one. Mirrors
/// `Interp::to_primitive_hint` (vm.rs), which builtin natives cannot call
/// (no interpreter borrow on these paths).
pub(crate) fn to_primitive_with_hint(
    heap: &mut Heap,
    obj: Gc<Object>,
    hint: PrimitiveHint,
) -> Result<Value, NativeError> {
    let this = Value::Object(obj);
    // `Symbol.toPrimitive` first, consulted under the real symbol key.
    // A stale `"@@toPrimitive"` *string* property no longer counts.
    // Mirrors `Interp::to_primitive_hint` (vm.rs), which builtin natives
    // cannot call (no interpreter borrow on these paths).
    let hook = match heap.well_known[WellKnown::ToPrimitive as usize] {
        Some(sym) => get_prop_key(heap, obj, PropKey::Symbol(sym)),
        None => None,
    };
    if hook.is_some_and(|h| is_callable(heap, h)) {
        let hint_name = match hint {
            PrimitiveHint::Number => "number",
            PrimitiveHint::String => "string",
            PrimitiveHint::Default => "default",
        };
        // `this` lives only in a Rust local here; root it across the
        // allocation, matching the `error_construct` discipline.
        let base = heap.protect(this);
        let hs = heap.alloc_str(hint_name);
        heap.release_temp(base);
        let r = call_function(heap, hook.unwrap(), this, &[Value::Str(hs?)])?;
        if matches!(r, Value::Object(_)) {
            return Err(NativeError::TypeError(
                "@@toPrimitive must return a primitive value".into(),
            ));
        }
        return Ok(r);
    }
    let (first, second) = match hint {
        PrimitiveHint::String => ("toString", "valueOf"),
        // `Number` and `Default`: the ordinary order is `valueOf` first.
        PrimitiveHint::Number | PrimitiveHint::Default => ("valueOf", "toString"),
    };
    for name in [first, second] {
        let m = get_prop(heap, obj, Wtf8::from_str(name));
        if m.is_some_and(|f| is_callable(heap, f)) {
            let r = call_function(heap, m.unwrap(), this, &[])?;
            if !matches!(r, Value::Object(_)) {
                return Ok(r);
            }
        }
    }
    // Every object inherits `Object.prototype.toString`/`valueOf`, so
    // reaching here means a broken prototype chain (e.g. `Object.create(null)`
    // with no hooks installed).
    Err(NativeError::TypeError(
        "cannot convert object to primitive".into(),
    ))
}

/// `ToPrimitive(input, "string")` for [`to_string_js`]. Observably
/// identical to `to_primitive_with_hint(heap, obj, PrimitiveHint::String)`
/// (same method order, same `TypeError`-as-`Err` contract, same v1
/// `"[object Object]"` fast path when neither `@@toPrimitive` nor
/// `toString` is found on the chain), but each method name costs exactly
/// one prototype-chain walk: `@@toPrimitive`, then `toString`, then
/// `valueOf` only when `toString` did not yield a primitive. Routing
/// through `to_primitive_with_hint` after the fast-path probes would walk
/// the chain a second time for every name.
fn to_primitive_string(heap: &mut Heap, obj: Gc<Object>) -> Result<Value, NativeError> {
    let this = Value::Object(obj);
    // `Symbol.toPrimitive` first, consulted under the real symbol key
    // (see `to_primitive_with_hint`).
    let hook = match heap.well_known[WellKnown::ToPrimitive as usize] {
        Some(sym) => get_prop_key(heap, obj, PropKey::Symbol(sym)),
        None => None,
    };
    if hook.is_some_and(|h| is_callable(heap, h)) {
        // `this` lives only in a Rust local here; root it across the
        // allocation, matching the `error_construct` discipline.
        let base = heap.protect(this);
        let hs = heap.alloc_str("string");
        heap.release_temp(base);
        let r = call_function(heap, hook.unwrap(), this, &[Value::Str(hs?)])?;
        if matches!(r, Value::Object(_)) {
            return Err(NativeError::TypeError(
                "@@toPrimitive must return a primitive value".into(),
            ));
        }
        return Ok(r);
    }
    let to_string = get_prop(heap, obj, Wtf8::from_str("toString"));
    if to_string.is_none() {
        // v1: `Object.prototype.toString` is not installed, so with no
        // hook and no `toString` on the chain there is nothing to call.
        // The caller maps this to the historical `"[object Object]"`.
        return Err(NativeError::TypeError(
            "cannot convert object to primitive".into(),
        ));
    }
    if to_string.is_some_and(|f| is_callable(heap, f)) {
        let r = call_function(heap, to_string.unwrap(), this, &[])?;
        if !matches!(r, Value::Object(_)) {
            return Ok(r);
        }
    }
    // Walked only when `toString` was missing, uncallable, or returned an
    // object.
    let value_of = get_prop(heap, obj, Wtf8::from_str("valueOf"));
    if value_of.is_some_and(|f| is_callable(heap, f)) {
        let r = call_function(heap, value_of.unwrap(), this, &[])?;
        if !matches!(r, Value::Object(_)) {
            return Ok(r);
        }
    }
    Err(NativeError::TypeError(
        "cannot convert object to primitive".into(),
    ))
}

/// `ToString`, returning the string as a [`Str`] handle instead of an owned
/// copy: `Value::Str` inputs (and `toString` results that are already
/// strings) come back as a free 16-byte handle copy, and short results
/// (at most [`STR_INLINE_MAX`](crate::value::STR_INLINE_MAX) bytes) are
/// packed inline by [`Heap::alloc_str`] with no heap allocation at all.
/// Number formatting still goes through a stack [`Wtf8Buf`] temporary.
///
/// # Rooting
///
/// A returned [`Str::Heap`] handle is **not rooted**: any subsequent heap
/// allocation may trigger a collection that frees it. Callers must either
/// consume it immediately with no intervening `&mut Heap` allocation
/// (e.g. copy the bytes into a `Wtf8Buf`, as [`to_string_js`] does), or
/// root it with [`Heap::protect`] until the last use (releasing with
/// [`Heap::release_temp`]). `Str::Inline` results need no rooting.
pub(crate) fn to_string_js_str(heap: &mut Heap, v: Value) -> Result<Str, NativeError> {
    match v {
        Value::Undefined => Ok(heap.alloc_str("undefined")?),
        Value::Null => Ok(heap.alloc_str("null")?),
        Value::Bool(b) => Ok(heap.alloc_str(if b { "true" } else { "false" })?),
        Value::Int(i) => {
            // Stack-format: every int stringifies to <= 11 bytes, so it
            // always goes inline and the old `i.to_string()` malloc was
            // pure waste.
            let mut buf = [0u8; 11];
            Ok(heap.alloc_str(int_to_str(i, &mut buf))?)
        }
        Value::Float(f) => {
            // Stack-format: the transient `String` from
            // `js_number_to_string` was malloc'd only to be copied into
            // the (usually inline) `Str`.
            let mut buf = [0u8; 32];
            Ok(heap.alloc_str(js_number_to_str(f, &mut buf))?)
        }
        // Free 16-byte handle copy; rooting stays the caller's business,
        // exactly as for the `Value` it came from.
        Value::Str(s) => Ok(s),
        Value::Symbol(_) => Err(NativeError::TypeError(
            "cannot convert symbol to string".into(),
        )),
        Value::BigInt(b) => {
            let bi = heap.get(b);
            let dec = bigint_to_decimal(bi.negative, &bi.digits);
            Ok(heap.alloc_str(&dec)?)
        }
        Value::Object(o) => {
            // Spec 7.1.2: `ToString` on an object is `ToPrimitive(input,
            // string)`: `@@toPrimitive("string")` first, then `toString`,
            // then `valueOf`. (v1: `Object.prototype.toString` is not
            // installed, so when no `toString`/`@@toPrimitive` is found on
            // the chain the historical `"[object Object]"` is kept instead
            // of falling through to `valueOf`, which the installed
            // `Object.prototype.toString` would shadow. When no method
            // yields a primitive the same fallback replaces the spec's
            // `TypeError`.)
            //
            // `to_primitive_string` fetches each hook with a single
            // prototype-chain walk; routing through `to_primitive_with_hint`
            // after the probes above would walk the chain again per name.
            let prim = match to_primitive_string(heap, o) {
                Ok(p) => p,
                Err(_) => return Ok(heap.alloc_str("[object Object]")?),
            };
            match prim {
                // Unrooted: `call_function` released its temporary roots
                // before returning. The caller must root or consume
                // immediately (see the rooting note above).
                Value::Str(s) => Ok(s),
                Value::Int(i) => {
                    let mut buf = [0u8; 11];
                    Ok(heap.alloc_str(int_to_str(i, &mut buf))?)
                }
                Value::Float(f) => {
                    let mut buf = [0u8; 32];
                    Ok(heap.alloc_str(js_number_to_str(f, &mut buf))?)
                }
                Value::Bool(b) => Ok(heap.alloc_str(if b { "true" } else { "false" })?),
                _ => Ok(heap.alloc_str("[object Object]")?),
            }
        }
    }
}

/// `ToString`. Thin wrapper over [`to_string_js_str`]: one owned copy of
/// the bytes, so cold paths (thrown-error stringification, console,
/// one-shot conversions) don't churn on handle rooting.
pub(crate) fn to_string_js(heap: &mut Heap, v: Value) -> Result<Wtf8Buf, NativeError> {
    // No heap allocation between the conversion and the copy (`to_owned`
    // grows a Rust `Vec`), so the possibly-unrooted handle is safe here.
    Ok(to_string_js_str(heap, v)?.as_wtf8().to_owned())
}

/// `ToPropertyKey` returning the owned [`ElemKey`]. Every symbol —
///
/// well-known or user-created — passes through as [`ElemKey::Symbol`]:
/// there is no alias translation and no `TypeError` for symbols.
///
/// Same rooting contract as [`to_string_js_str`]: an [`ElemKey::Str`]
/// holding a `Str::Heap` is unrooted, so callers must consume it before
/// any further `&mut Heap` allocation or hold it with [`Heap::protect`]
/// until the last use.
pub(crate) fn to_property_key(heap: &mut Heap, v: Value) -> Result<ElemKey, NativeError> {
    if let Value::Symbol(s) = v {
        return Ok(ElemKey::Symbol(s));
    }
    Ok(ElemKey::Str(to_string_js_str(heap, v)?))
}

/// `ToLength`: clamped to `[0, 2^53 - 1]`.
pub(crate) fn to_length(v: Value) -> Result<u64, NativeError> {
    const MAX: f64 = 9007199254740991.0;
    let n = to_number(v)?;
    if n.is_nan() || n <= 0.0 {
        return Ok(0);
    }
    if n.is_infinite() {
        return Ok(MAX as u64);
    }
    Ok(n.trunc().min(MAX) as u64)
}

/// `ToIntegerOrInfinity`: `NaN` -> `+0`, truncates toward zero.
pub(crate) fn to_integer_or_infinity(v: Value) -> Result<f64, NativeError> {
    let n = to_number(v)?;
    if n.is_nan() {
        return Ok(0.0);
    }
    Ok(n.trunc())
}

// ---------------------------------------------------------------------------
// Equality
// ---------------------------------------------------------------------------

/// `Strict Equality Comparison` (`===`). Needs no heap: string contents are
/// read through the handle.
pub(crate) fn strict_equal(a: Value, b: Value) -> bool {
    match (a, b) {
        (Value::Undefined, Value::Undefined) | (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x == y,
        (Value::Int(x), Value::Float(y)) => f64::from(x) == y,
        (Value::Float(x), Value::Int(y)) => x == f64::from(y),
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Object(x), Value::Object(y)) => x == y,
        (Value::Symbol(x), Value::Symbol(y)) => x == y,
        (Value::BigInt(x), Value::BigInt(y)) => x == y,
        _ => false,
    }
}

/// `SameValueZero`: like `===` but `NaN` equals `NaN`.
pub(crate) fn same_value_zero(a: Value, b: Value) -> bool {
    if let (Value::Float(x), Value::Float(y)) = (a, b)
        && x.is_nan()
        && y.is_nan()
    {
        return true;
    }
    strict_equal(a, b)
}

// ---------------------------------------------------------------------------
// Property helpers (spec [[Get]]/[[Set]]/[[DefineOwnProperty]]/[[Delete]])
// ---------------------------------------------------------------------------

/// Is `s` a canonical array index (`"0"` .. `"4294967294"`)?
pub(crate) fn array_index_key(s: &Wtf8) -> Option<u32> {
    if s.is_empty() || s.len() > 10 {
        return None;
    }
    if s.len() > 1 && s.as_bytes().starts_with(b"0") {
        return None;
    }
    if !s.as_bytes().iter().all(|b| b.is_ascii_digit()) {
        return None;
    }
    // ASCII digits: always well-formed, so `as_utf8` is `Some`.
    let n: u64 = s.as_utf8()?.parse().ok()?;
    if n >= 4294967295 {
        return None;
    }
    Some(n as u32)
}

/// Own property value, or `None`.
pub(crate) fn get_own(heap: &Heap, obj: Gc<Object>, key: &Wtf8) -> Option<Value> {
    heap.get_property(obj, key).map(|p| p.value)
}

/// `HasOwnProperty`: no prototype walk.
pub(crate) fn has_own(heap: &Heap, obj: Gc<Object>, key: &Wtf8) -> bool {
    heap.get_property(obj, key).is_some()
}

/// Prototype-chain walk with cycle protection. Visits `obj`, then each
/// successive `[[Prototype]]`, returning the first `Some` that `f` produces.
/// The walk ends (as if the chain had ended) if a prototype cycle is
/// detected, so a cyclic chain can never hang the caller.
pub(crate) fn walk_proto_chain<T>(
    heap: &Heap,
    obj: Gc<Object>,
    mut f: impl FnMut(Gc<Object>) -> Option<T>,
) -> Option<T> {
    let mut current = Value::Object(obj);
    // Every `[[SetPrototypeOf]]` path rejects cycles up front (see
    // `check_prototype_cycle`), but this visited set keeps the walk total
    // even if a cycle ever slips through another path.
    let mut visited: Vec<Gc<Object>> = Vec::new();
    loop {
        match current {
            Value::Object(o) => {
                if visited.contains(&o) {
                    return None;
                }
                visited.push(o);
                if let Some(found) = f(o) {
                    return Some(found);
                }
                current = heap.get(o).prototype;
            }
            _ => return None,
        }
    }
}

/// Reject `proto` as the new `[[Prototype]]` of `obj` when it would create a
/// prototype cycle, i.e. when `obj` is reachable by walking `proto`'s chain.
/// Every `[[SetPrototypeOf]]` path must call this before
/// `heap.set_prototype`: a cyclic chain would hang every prototype walk.
pub(crate) fn check_prototype_cycle(
    heap: &Heap,
    obj: Gc<Object>,
    proto: Value,
) -> Result<(), NativeError> {
    let mut cur = proto;
    while let Value::Object(o) = cur {
        if o == obj {
            return Err(NativeError::TypeError("cyclic prototype chain".into()));
        }
        cur = heap.get(o).prototype;
    }
    Ok(())
}

/// `[[Get]]`: own property, else walk the prototype chain.
pub(crate) fn get_prop(heap: &Heap, obj: Gc<Object>, key: &Wtf8) -> Option<Value> {
    walk_proto_chain(heap, obj, |o| heap.get_property(o, key).map(|p| p.value))
}

/// [`get_prop`] over a [`PropKey`]: string keys take the historical path
/// verbatim; symbol keys scan the symbol side tables at each level.
/// Data properties only, like [`get_prop`] (no accessor invocation).
pub(crate) fn get_prop_key(heap: &Heap, obj: Gc<Object>, key: PropKey<'_>) -> Option<Value> {
    if let PropKey::Str(s) = key {
        return get_prop(heap, obj, s);
    }
    walk_proto_chain(heap, obj, |o| {
        heap.get_property_key(o, key).map(|p| p.value)
    })
}

/// `HasProperty`: own-or-inherited presence (data *and* accessor
/// properties), prototype chain included.
pub(crate) fn has_prop(heap: &Heap, obj: Gc<Object>, key: &Wtf8) -> bool {
    find_prop_in_chain(heap, obj, key).is_some()
}

/// Spec `Get(O, key)`: walk the prototype chain; invoke an accessor
/// getter with `O` as the receiver (re-entering the VM for bytecode
/// getters, via [`call_function`], which also roots `O` across the
/// call); propagate getter throws as [`NativeError`]. Data properties
/// and absent keys behave like [`get_prop`].
///
/// Native builtins must use this — not [`get_prop`] — wherever the spec
/// says `Get`: `get_prop` reads the stored data value and never fires
/// accessors, so getter side effects (and throws) would be silently
/// lost.
pub(crate) fn spec_get(heap: &mut Heap, obj: Gc<Object>, key: &Wtf8) -> Result<Value, NativeError> {
    match find_prop_in_chain(heap, obj, key) {
        Some(ChainProp::Data { prop, .. }) => Ok(prop.value),
        Some(ChainProp::Accessor { prop }) => match prop.get {
            Value::Undefined => Ok(Value::Undefined),
            get => {
                if !is_callable(heap, get) {
                    return Err(NativeError::TypeError(format!(
                        "getter for '{key}' is not callable"
                    )));
                }
                call_function(heap, get, Value::Object(obj), &[])
            }
        },
        None => Ok(Value::Undefined),
    }
}

/// `[[Set]]` on an own-or-new property: updates writable data properties,
/// creates new ones when extensible, silently ignores non-writable ones
/// (sloppy mode). Returns `Err` only for non-extensible targets.
pub(crate) fn set_prop(
    heap: &mut Heap,
    obj: Gc<Object>,
    key: &Wtf8,
    value: Value,
) -> Result<(), AllocError> {
    match heap.get_property(obj, key) {
        Some(existing) => {
            if existing.writable {
                heap.define_property(obj, key, Property { value, ..existing })?;
            }
            Ok(())
        }
        None => {
            if heap.get(obj).extensible {
                heap.define_property(obj, key, Property::data(value))?;
                Ok(())
            } else {
                Err(AllocError)
            }
        }
    }
}

/// [`set_prop`] over a [`PropKey`]: string keys take the historical path
/// verbatim; symbol keys update/define in the symbol side tables.
pub(crate) fn set_prop_key(
    heap: &mut Heap,
    obj: Gc<Object>,
    key: PropKey<'_>,
    value: Value,
) -> Result<(), AllocError> {
    if let PropKey::Str(s) = key {
        return set_prop(heap, obj, s, value);
    }
    match heap.get_property_key(obj, key) {
        Some(existing) => {
            if existing.writable {
                heap.define_property_key(obj, key, Property { value, ..existing })?;
            }
            Ok(())
        }
        None => {
            if heap.get(obj).extensible {
                heap.define_property_key(obj, key, Property::data(value))?;
                Ok(())
            } else {
                Err(AllocError)
            }
        }
    }
}

/// `[[DefineOwnProperty]]` with validation: rejects incompatible
/// redefinitions of non-configurable properties, rejects definitions on
/// non-extensible objects when the property is new.
pub(crate) fn define_prop(
    heap: &mut Heap,
    obj: Gc<Object>,
    key: PropKey<'_>,
    prop: Property,
) -> Result<(), NativeError> {
    if heap.get_property_key(obj, key).is_none()
        && let Some(existing) = heap.get_accessor_key(obj, key)
    {
        // Accessor -> data conversion: allowed only when configurable.
        if !existing.configurable {
            return Err(NativeError::TypeError(format!(
                "cannot convert non-configurable accessor property '{}' to data",
                heap.format_key(key)
            )));
        }
    }
    match heap.get_property_key(obj, key) {
        Some(existing) => {
            if !existing.configurable {
                if prop.configurable {
                    return Err(NativeError::TypeError(format!(
                        "cannot redefine non-configurable property '{}'",
                        heap.format_key(key)
                    )));
                }
                if prop.enumerable != existing.enumerable {
                    return Err(NativeError::TypeError(format!(
                        "cannot change enumerability of non-configurable property '{}'",
                        heap.format_key(key)
                    )));
                }
                if !existing.writable
                    && (prop.writable || !strict_equal(prop.value, existing.value))
                {
                    return Err(NativeError::TypeError(format!(
                        "cannot redefine non-configurable, non-writable property '{}'",
                        heap.format_key(key)
                    )));
                }
            }
            heap.define_property_key(obj, key, prop)?;
            Ok(())
        }
        None => {
            if !heap.get(obj).extensible {
                return Err(NativeError::TypeError(format!(
                    "cannot define property '{}' on non-extensible object",
                    heap.format_key(key)
                )));
            }
            heap.define_property_key(obj, key, prop)?;
            Ok(())
        }
    }
}

/// `[[Delete]]`: `false` for non-configurable own properties.
pub(crate) fn delete_prop(heap: &mut Heap, obj: Gc<Object>, key: &Wtf8) -> bool {
    match heap.get_property(obj, key) {
        Some(p) => {
            if !p.configurable {
                return false;
            }
            heap.delete_property(obj, key)
        }
        None => match heap.get_accessor(obj, key) {
            Some(a) => {
                if !a.configurable {
                    return false;
                }
                heap.delete_property(obj, key)
            }
            None => true,
        },
    }
}

/// Own enumerable keys in spec order: ascending array indices, then other
/// string keys in insertion order. Symbol keys are enumerated separately
/// via [`Heap::own_enumerable_symbol_keys`](crate::gc::Heap::own_enumerable_symbol_keys):
/// `Object.keys`/`values`/`entries` exclude symbols per spec, while
/// `Object.assign` and object spread append the symbol tail after this
/// string list.
///
/// Dense arrays keep their elements in `ObjectCold::dense`, not in the property
/// map, so the index keys are synthesized from the dense store: every
/// present slot (`Some`) contributes its decimal index; holes contribute
/// nothing (they are absent properties, hence non-enumerable). While
/// dense storage is active no index property lives in `properties` (see
/// [`ObjectCold::dense`](crate::value::ObjectCold)), so the
/// synthesized indices merge cleanly into the ascending-`indices` bucket.
///
/// The index bucket is already ascending when no index key comes from
/// the property map (the dense scan yields indices in order), so the
/// sort is skipped in that case — an O(n) fast path for dense-only
/// objects. [`own_enumerable_keys_detailed`] also reports whether the
/// sort ran.
pub(crate) fn own_enumerable_keys(heap: &Heap, obj: Gc<Object>) -> Vec<Str> {
    own_enumerable_keys_detailed(heap, obj).0
}

/// [`own_enumerable_keys`], additionally returning whether the index
/// bucket was sorted: `false` means the dense-only fast path engaged (no
/// property-map index key was merged, so the ascending dense scan needed
/// no sort).
pub(crate) fn own_enumerable_keys_detailed(heap: &Heap, obj: Gc<Object>) -> (Vec<Str>, bool) {
    let o = heap.get(obj);
    let props: Vec<(Str, bool)> = o
        .properties
        .iter()
        .map(|(k, p)| (*k, p.enumerable))
        .chain(o.accessors.iter().map(|(k, a)| (*k, a.enumerable)))
        .collect();
    let mut indices: Vec<(u32, Str)> = Vec::new();
    if dense_is_active(o)
        && let Some(dense) = o.dense()
    {
        indices.extend(
            dense
                .iter()
                .enumerate()
                .filter(|(_, slot)| slot.is_some())
                .map(|(i, _)| (i as u32, dense_index_str(i as u32))),
        );
    }
    let mut others: Vec<Str> = Vec::new();
    let mut merged_map_index = false;
    for (k, enumerable) in props {
        if !enumerable {
            continue;
        }
        match array_index_key(k.as_wtf8()) {
            Some(i) => {
                indices.push((i, k));
                merged_map_index = true;
            }
            None => others.push(k),
        }
    }
    if merged_map_index {
        indices.sort_by_key(|(i, _)| *i);
    }
    let keys = indices.into_iter().map(|(_, k)| k).chain(others).collect();
    (keys, merged_map_index)
}

/// [`Str`] key for dense index `i`, packed inline with no allocation:
/// canonical array indices stringify to at most 10 digits, well under
/// [`STR_INLINE_MAX`](crate::value::STR_INLINE_MAX).
fn dense_index_str(i: u32) -> Str {
    let mut buf = [0u8; 16];
    let mut pos = buf.len();
    let mut n = i;
    loop {
        pos -= 1;
        buf[pos] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    let s = std::str::from_utf8(&buf[pos..]).expect("dense_index_str emits ASCII digits");
    Str::inline(s).expect("array index fits inline")
}

// ---------------------------------------------------------------------------
// Array storage
// ---------------------------------------------------------------------------

/// Current array length (the `"length"` data property; `0` if absent).
pub(crate) fn array_length(heap: &Heap, arr: Gc<Object>) -> u64 {
    match get_own(heap, arr, Wtf8::from_str("length")) {
        Some(Value::Int(i)) if i >= 0 => i as u64,
        Some(Value::Float(f)) if f >= 0.0 => f as u64,
        _ => 0,
    }
}

/// Allocate an empty array with the `%Array.prototype%` from the heap.
pub(crate) fn new_array(heap: &mut Heap) -> Result<Gc<Object>, AllocError> {
    let proto = heap
        .array_prototype
        .map(Value::Object)
        .unwrap_or(Value::Null);
    let arr = heap.alloc_object(Object {
        kind: ObjectKind::Array,
        prototype: proto,
        ..Default::default()
    })?;
    heap.define_property_str(
        arr,
        "length",
        Property {
            value: Value::Int(0),
            writable: true,
            enumerable: false,
            configurable: false,
        },
    )?;
    Ok(arr)
}

/// `new_array` pre-sized to `len` (indices start as holes).
pub(crate) fn new_array_with_length(heap: &mut Heap, len: u64) -> Result<Gc<Object>, NativeError> {
    let arr = new_array(heap)?;
    set_len(heap, arr, len)?;
    Ok(arr)
}

/// `new_array` filled with the given optional elements (holes preserved).
pub(crate) fn new_array_from_opt(
    heap: &mut Heap,
    elems: &[Option<Value>],
) -> Result<Gc<Object>, NativeError> {
    let arr = new_array(heap)?;
    for (i, e) in elems.iter().enumerate() {
        if let Some(v) = e {
            set_idx_raw(heap, arr, i as u64, *v)?;
        }
    }
    set_len(heap, arr, elems.len() as u64)?;
    Ok(arr)
}

/// ECMA-262 `SpeciesConstructor(originalArray, %Array%)`: resolves the
/// `@@species` constructor for `originalArray`, defaulting to the
/// `%Array%` intrinsic. Returns the constructor and whether it is the
/// intrinsic default (the dense fast paths in `flat`/`flatMap` only
/// apply to the default: a hostile species result is not a dense
/// array). `method` names the caller for error messages.
///
/// A throwing `@@species` getter propagates; `undefined`/`null`
/// species falls back to `%Array%`; a non-constructor species throws
/// `TypeError` — all handled by [`species_constructor`].
fn array_species_ctor(
    heap: &mut Heap,
    original: Gc<Object>,
    method: &str,
) -> Result<(Value, bool), NativeError> {
    let default = heap.array_ctor.map(Value::Object);
    let target = species_constructor(heap, original, default, method)?;
    Ok((target, Some(target) == default))
}

/// `Construct(target, « len »)` for an already-resolved species
/// constructor: the second half of `ArraySpeciesCreate`.
fn species_construct(
    heap: &mut Heap,
    target: Value,
    len: u64,
    method: &str,
) -> Result<Gc<Object>, NativeError> {
    // `target` is the `@@species` getter's return value — hostile code
    // may hand back a fresh proxy — and it lives only in this Rust local
    // across `reflect_construct_inner`. The proxy-with-`construct`-trap
    // path allocates (`new_value_array`) before the trap runs, and that
    // allocation can collect while the proxy is unreachable from any
    // root; the subsequent trap call would then use dangling handles.
    // (Bytecode-class targets are saved by `invoke`'s `call_roots`, but
    // the proxy path is not.) Hold `target` as a temporary root for the
    // whole construction.
    with_temp_root(heap, target, |heap| {
        match reflect_construct_inner(heap, target, target, &[u64_to_value(len)])? {
            Value::Object(o) => Ok(o),
            _ => Err(NativeError::TypeError(format!(
                "{method}: @@species constructor did not return an object"
            ))),
        }
    })
}

/// ECMA-262 `ArraySpeciesCreate(originalArray, length)`: resolve the
/// species constructor, then `Construct(C, « length »)`. Returns the
/// constructed object plus the default-constructor flag from
/// [`array_species_ctor`].
fn array_species_create(
    heap: &mut Heap,
    original: Gc<Object>,
    len: u64,
    method: &str,
) -> Result<(Gc<Object>, bool), NativeError> {
    let (target, is_default) = array_species_ctor(heap, original, method)?;
    let a = species_construct(heap, target, len, method)?;
    Ok((a, is_default))
}

/// ECMA-262 `CreateDataPropertyOrThrow` for an index of a species-created
/// result: the array-exotic `[[DefineOwnProperty]]` path (`set_idx`,
/// which throws `RangeError` past 2^32-1 like the old `filter` append
/// path did) when the species constructor returned a real array. A
/// hostile `@@species` may return a non-array object — the exotic
/// helpers (`set_idx`, `set_len`) debug-assert `ObjectKind::Array`, so
/// they cannot be used blindly here — and then the write is a true
/// `[[DefineOwnProperty]]` with validation: `define_prop` throws
/// `TypeError` on non-extensible targets and on incompatible
/// redefinitions of non-configurable properties (a non-writable
/// existing index, a non-configurable accessor), instead of silently
/// skipping the write or misreporting `AllocError` as out-of-memory. A
/// proxy result goes through its own `[[DefineOwnProperty]]` so the
/// `defineProperty` trap fires (a falsy trap return becomes
/// `TypeError`), never landing the write on the proxy object itself.
/// `v` and `a` are temp-rooted across the descriptor allocation and the
/// trap, both of which can collect.
fn species_set_idx(heap: &mut Heap, a: Gc<Object>, i: u64, v: Value) -> Result<(), NativeError> {
    if heap.get(a).kind == ObjectKind::Array {
        return set_idx(heap, a, i, v);
    }
    let key = i.to_string();
    with_temp_root(heap, v, |heap| {
        with_temp_root(heap, Value::Object(a), |heap| {
            let key = Wtf8::from_str(&key);
            if let Some(proxy) = as_proxy(heap, Value::Object(a)) {
                let desc = descriptor_object(heap, Property::data(v))?;
                if !proxy_define_property(heap, proxy, PropKey::Str(key), Value::Object(desc))? {
                    return Err(NativeError::TypeError(format!(
                        "Array species result: proxy 'defineProperty' trap rejected index '{}'",
                        heap.format_key(PropKey::Str(key))
                    )));
                }
                return Ok(());
            }
            define_prop(heap, a, PropKey::Str(key), Property::data(v))
        })
    })
}

/// ECMA-262 `Set(A, "length", len)` with throw semantics: the final
/// length write of `Array.prototype.concat` (ES2024 §23.1.3.2 step 6).
/// Unlike the index writes above this is `[[Set]]`, not
/// `CreateDataPropertyOrThrow`: a `length` setter (own or inherited)
/// fires on hostile results, and a proxy result goes through its `set`
/// trap. Real arrays take the exotic `ArraySetLength` path (`set_len`).
/// A rejected set throws `TypeError`.
fn species_set_length(
    heap: &mut Heap,
    a: Gc<Object>,
    len: u64,
    method: &str,
) -> Result<(), NativeError> {
    if heap.get(a).kind == ObjectKind::Array {
        return set_len(heap, a, len);
    }
    let v = u64_to_value(len);
    with_temp_root(heap, v, |heap| {
        with_temp_root(heap, Value::Object(a), |heap| {
            let key = Wtf8::from_str("length");
            let ok = if let Some(proxy) = as_proxy(heap, Value::Object(a)) {
                proxy_set(heap, proxy, PropKey::Str(key), v, Value::Object(a), 0)?
            } else {
                ordinary_set(heap, a, key, v, Value::Object(a), 0)?
            };
            if !ok {
                return Err(NativeError::TypeError(format!(
                    "{method}: @@species result rejected length assignment"
                )));
            }
            Ok(())
        })
    })
}

/// Read index `i`; holes read as `undefined`.
//
// (Index-access seam used by the unit tests; upcoming array builtins read
// through it. The allow keeps `cargo clippy -- -D warnings` green on the
// non-test lib target.)
#[allow(dead_code)]
pub(crate) fn get_idx(heap: &Heap, arr: Gc<Object>, i: u64) -> Value {
    get_own(heap, arr, Wtf8::from_str(&i.to_string())).unwrap_or(Value::Undefined)
}

/// Does index `i` exist (i.e. is it not a hole)?
//
// (Hole-detection seam used by the unit tests; upcoming array builtins need
// it. The allow keeps `cargo clippy -- -D warnings` green on the non-test
// lib target.)
#[allow(dead_code)]
pub(crate) fn has_idx(heap: &Heap, arr: Gc<Object>, i: u64) -> bool {
    has_own(heap, arr, Wtf8::from_str(&i.to_string()))
}

/// Write index `i` as a plain data property (internal bulk path; the
/// array exotic still extends `length` when `i` is past the end, so
/// callers that need an exact final length call [`set_len`] after).
/// Holes bookkeeping is preserved: only `i` is written.
pub(crate) fn set_idx_raw(
    heap: &mut Heap,
    arr: Gc<Object>,
    i: u64,
    v: Value,
) -> Result<(), AllocError> {
    set_prop(heap, arr, Wtf8::from_str(&i.to_string()), v)
}

/// Write index `i`, extending `length` when needed, and enforcing the
/// `2^32 - 1` maximum (throws `RangeError` past it). Goes through the
/// array exotic `[[DefineOwnProperty]]`: the write extends `length` to
/// `i + 1` and fails when `length` is non-writable.
pub(crate) fn set_idx(
    heap: &mut Heap,
    arr: Gc<Object>,
    i: u64,
    v: Value,
) -> Result<(), NativeError> {
    if i >= 4294967295 {
        return Err(NativeError::RangeError("invalid array length".into()));
    }
    array_define_own_property(
        heap,
        arr,
        Wtf8::from_str(&i.to_string()),
        ArrayDefineDesc {
            value: Some(v),
            writable: Some(true),
            enumerable: Some(true),
            configurable: Some(true),
            ..Default::default()
        },
    )
}

/// Set `length` through the array exotic `[[DefineOwnProperty]]`
/// (`ArraySetLength`): extending only changes `length`; shrinking
/// deletes trailing indices in descending order, restoring `length`
/// past a non-configurable element. Non-Uint32 lengths throw
/// `RangeError`; redefinition failures throw `TypeError`.
pub(crate) fn set_len(heap: &mut Heap, arr: Gc<Object>, len: u64) -> Result<(), NativeError> {
    array_define_own_property(
        heap,
        arr,
        Wtf8::from_str("length"),
        ArrayDefineDesc {
            value: Some(u64_to_value(len)),
            ..Default::default()
        },
    )
}

/// `true` for the keys the array exotic `[[DefineOwnProperty]]`
/// intercepts: `"length"` and canonical array indices.
pub(crate) fn is_array_exotic_key(key: &Wtf8) -> bool {
    key == "length" || array_index_key(key).is_some()
}

/// The array exotic `[[DefineOwnProperty]]` (spec 10.4.2) for the
/// throwing builtin paths (`Object.defineProperty`, `set_idx`,
/// `set_len`): `Ok(())` on success, `TypeError` when the define is
/// rejected, `RangeError` for non-Uint32 `length` values (both
/// propagated from the heap helper).
pub(crate) fn array_define_own_property(
    heap: &mut Heap,
    arr: Gc<Object>,
    key: &Wtf8,
    desc: ArrayDefineDesc,
) -> Result<(), NativeError> {
    if !heap.define_array_own_property(arr, key, &desc)? {
        return Err(NativeError::TypeError(format!(
            "cannot define property '{key}'"
        )));
    }
    Ok(())
}

/// Get `length` coerced to a number (used by generic array methods on any
/// receiver): reads own `"length"` then applies `ToLength`.
pub(crate) fn get_length(heap: &Heap, obj: Gc<Object>) -> Result<u64, NativeError> {
    match get_own(heap, obj, Wtf8::from_str("length")) {
        Some(v) => to_length(v),
        None => Ok(0),
    }
}

// ---------------------------------------------------------------------------
// Callback invocation choke point
// ---------------------------------------------------------------------------

/// Invoke a callable from the array iteration methods.
///
/// Bytecode ([`ObjectKind::Function`]) callees re-enter the VM via
/// [`crate::vm::reenter_call`]; native callees run directly. Bound
/// functions (see `Function.prototype.bind`) are intercepted first: the
/// hidden target / this / bound-args slots are resolved, the bound args
/// are concatenated with the call args, and dispatch restarts at the
/// target — mirroring the VM's own dispatch (`vm.rs` `invoke`), so
/// `boundFn.call(...)`, `boundFn.apply(...)`, and builtins taking callbacks
/// (e.g. `arr.map(boundFn)`) all work. `bind` flattens chains at bind time,
/// so the loop below normally hops once; it still terminates on
/// adversarial nesting because each hop strictly shrinks the remaining
/// chain (a bound target is never itself bound).
pub(crate) fn call_function(
    heap: &mut Heap,
    callee: Value,
    this_arg: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    // Root the callee, receiver, and arguments across the call: invoking
    // a JavaScript function can allocate and trigger a collection, which
    // must not free the very values the call is operating on. Released
    // on both success and error paths via the closure below.
    let base = heap.protect(callee);
    heap.protect(this_arg);
    for &a in args {
        heap.protect(a);
    }
    let result = (|| -> Result<Value, NativeError> {
        let mut callee = callee;
        let mut this_arg = this_arg;
        // Borrowed in the common (unbound) case; owned only when a bound
        // hop concatenates the argument vectors.
        let mut owned: Option<Vec<Value>> = None;
        loop {
            let args_now: &[Value] = owned.as_deref().unwrap_or(args);
            let obj = match callee {
                Value::Object(o) => o,
                _ => {
                    return Err(NativeError::TypeError("callback is not a function".into()));
                }
            };
            // Proxy dispatch: a proxy of a callable is itself callable.
            // (The VM's `invoke_inner_depth` carries the same check for
            // the bytecode call path.)
            if let Some(proxy) = crate::builtin_reflect_proxy::as_proxy(heap, callee) {
                return crate::builtin_reflect_proxy::proxy_apply(heap, proxy, this_arg, args_now);
            }
            if heap.get(obj).invoke_flags & Object::INVOKE_BOUND != 0
                && let Some((target, bthis, bargs)) = crate::vm::bound_parts(heap, obj)
            {
                let mut all = bargs;
                all.extend_from_slice(args_now);
                callee = target;
                this_arg = bthis;
                owned = Some(all);
                continue;
            }
            match heap.get(obj).kind {
                ObjectKind::NativeFunction(f) => {
                    // Builtins observe the raw thisArgument: per ECMA-262,
                    // `Call` invokes `F.[[Call]](V, ...)` with no
                    // this-substitution for natives (`OrdinaryCallBindThis`
                    // applies only to ECMAScript functions, and the VM's
                    // sloppy coercion in `coerce_this` likewise skips
                    // natives). Passing `undefined`/`null` through lets
                    // `ToObject`-based builtins (`this_object`) throw the
                    // spec `TypeError` instead of silently operating on
                    // the global object.
                    return f(heap, this_arg, args_now);
                }
                ObjectKind::Function(_) => {
                    return crate::vm::reenter_call(heap, callee, this_arg, args_now);
                }
                _ => return Err(NativeError::TypeError("callback is not a function".into())),
            }
        }
    })();
    heap.release_temp(base);
    result
}

/// Shared callback argument-count validation for the iteration methods:
/// `call`/`construct` trap with zero callbacks; `map` with one.
pub(crate) fn expect_callback(args: &[Value], name: &str) -> Result<Value, NativeError> {
    match args.first() {
        Some(cb) => Ok(*cb),
        None => Err(NativeError::TypeError(format!(
            "{name} requires a callback"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Object constructor + statics
// ---------------------------------------------------------------------------

fn object_constructor(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    match get_arg(args, 0) {
        Value::Undefined | Value::Null => {
            // ECMA-262 OrdinaryCreateFromConstructor(newTarget,
            // "%Object.prototype%"): when a derived class (`class D extends
            // Object`) runs its implicit constructor, `super()` constructs
            // through this builtin with newTarget = D, so the instance's
            // prototype must be D.prototype — not %Object.prototype%.
            // The VM stashes newTarget in a thread-local for native
            // constructors (see `NEW_TARGET` in vm.rs); `None` covers the
            // plain-call path, where `new Object()` still resolves
            // `Object.prototype` through the newTarget branch below.
            let proto = match active_new_target() {
                Some(Value::Object(nt)) => {
                    match get_prop_chain(heap, nt, Wtf8::from_str("prototype")) {
                        Some(Value::Object(p)) => Value::Object(p),
                        _ => heap
                            .object_prototype
                            .map(Value::Object)
                            .unwrap_or(Value::Null),
                    }
                }
                _ => heap
                    .object_prototype
                    .map(Value::Object)
                    .unwrap_or(Value::Null),
            };
            let o = heap.alloc_object(Object {
                kind: ObjectKind::Ordinary,
                prototype: proto,
                ..Default::default()
            })?;
            Ok(Value::Object(o))
        }
        Value::Object(o) => Ok(Value::Object(o)),
        // v1 gap: no ToObject wrapping of primitives.
        _ => Err(NativeError::TypeError(
            "Object(): primitive argument wrapping is a v1 gap".into(),
        )),
    }
}

fn object_create(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let proto = get_arg(args, 0);
    match proto {
        Value::Object(_) | Value::Null => {}
        _ => {
            return Err(NativeError::TypeError(
                "Object.create: prototype must be an object or null".into(),
            ));
        }
    }
    let o = heap.alloc_object(Object {
        kind: ObjectKind::Ordinary,
        prototype: proto,
        ..Default::default()
    })?;
    if args.len() >= 2 {
        object_define_properties_impl(heap, Value::Object(o), get_arg(args, 1))?;
    }
    Ok(Value::Object(o))
}

fn own_key_strings(heap: &mut Heap, v: Value, name: &str) -> Result<Vec<Wtf8Buf>, NativeError> {
    let obj = match v {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(format!(
                "Object.{name}: argument must be an object"
            )));
        }
    };
    Ok(own_enumerable_keys(heap, obj)
        .iter()
        .map(|k| k.as_wtf8().to_owned())
        .collect())
}

fn object_keys(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let keys = own_key_strings(heap, get_arg(args, 0), "keys")?;
    let arr = new_array(heap)?;
    for (i, k) in keys.iter().enumerate() {
        let s = heap.alloc_str_wtf8(k.as_wtf8())?;
        set_idx_raw(heap, arr, i as u64, Value::Str(s))?;
    }
    set_len(heap, arr, keys.len() as u64)?;
    Ok(Value::Object(arr))
}

fn object_values(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let obj = match get_arg(args, 0) {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Object.values: argument must be an object".into(),
            ));
        }
    };
    let keys = own_enumerable_keys(heap, obj);
    let arr = new_array(heap)?;
    for (i, k) in keys.iter().enumerate() {
        let v = get_own(heap, obj, k.as_wtf8()).unwrap_or(Value::Undefined);
        set_idx_raw(heap, arr, i as u64, v)?;
    }
    set_len(heap, arr, keys.len() as u64)?;
    Ok(Value::Object(arr))
}

fn object_entries(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let obj = match get_arg(args, 0) {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Object.entries: argument must be an object".into(),
            ));
        }
    };
    let keys = own_enumerable_keys(heap, obj);
    let arr = new_array(heap)?;
    for (i, k) in keys.iter().enumerate() {
        let v = get_own(heap, obj, k.as_wtf8()).unwrap_or(Value::Undefined);
        let pair = new_array(heap)?;
        set_idx_raw(heap, pair, 0, Value::from(*k))?;
        set_idx_raw(heap, pair, 1, v)?;
        set_len(heap, pair, 2)?;
        set_idx_raw(heap, arr, i as u64, Value::Object(pair))?;
    }
    set_len(heap, arr, keys.len() as u64)?;
    Ok(Value::Object(arr))
}

/// Own keys in spec order, enumerable or not: ascending array indices,
/// then other string keys in insertion order. Symbol keys are enumerated
/// separately via [`Heap::own_symbol_keys`](crate::gc::Heap::own_symbol_keys):
/// `Object.getOwnPropertyNames` excludes symbols per spec, while
/// `Reflect.ownKeys` appends the symbol tail after this string list.
///
/// Dense arrays synthesize their index keys from the dense store exactly as
/// [`own_enumerable_keys`] does (holes are absent properties, so they
/// contribute nothing here either).
fn own_all_keys(heap: &Heap, obj: Gc<Object>) -> Vec<Str> {
    let o = heap.get(obj);
    let props: Vec<Str> = o
        .properties
        .iter()
        .map(|(k, _)| *k)
        .chain(o.accessors.iter().map(|(k, _)| *k))
        .collect();
    let dense_active = dense_is_active(o) && o.dense().is_some();
    let mut indices: Vec<(u32, Str)> = Vec::new();
    if dense_active && let Some(dense) = o.dense() {
        indices.extend(
            dense
                .iter()
                .enumerate()
                .filter(|(_, slot)| slot.is_some())
                .map(|(i, _)| (i as u32, dense_index_str(i as u32))),
        );
    }
    let mut others: Vec<Str> = Vec::new();
    let mut has_length = false;
    for k in props {
        if k.as_wtf8() == Wtf8::from_str("length") {
            has_length = true;
        }
        match array_index_key(k.as_wtf8()) {
            Some(i) => indices.push((i, k)),
            None => others.push(k),
        }
    }
    indices.sort_by_key(|(i, _)| *i);
    let mut out: Vec<Str> = indices.into_iter().map(|(_, k)| k).collect();
    // `Heap::get_property` synthesizes `"length"` for dense arrays (a
    // dense-active array never carries it in the property map: plain
    // defines resize the store, attributed ones materialize). `[[OwnPropertyKeys]]`
    // must list it right after the indices.
    if dense_active && !has_length {
        out.push(Str::inline("length").expect("\"length\" fits inline"));
    }
    out.extend(others);
    out
}

fn object_get_own_property_names(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let obj = match get_arg(args, 0) {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Object.getOwnPropertyNames: argument must be an object".into(),
            ));
        }
    };
    let keys = own_all_keys(heap, obj);
    let arr = new_array(heap)?;
    for (i, k) in keys.iter().enumerate() {
        set_idx_raw(heap, arr, i as u64, Value::from(*k))?;
    }
    set_len(heap, arr, keys.len() as u64)?;
    Ok(Value::Object(arr))
}

fn object_get_own_property_symbols(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let obj = match get_arg(args, 0) {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Object.getOwnPropertySymbols: argument must be an object".into(),
            ));
        }
    };
    // Both enumerable and non-enumerable symbol keys, in
    // `[[OwnPropertyKeys]]` tail order (data side table, then accessor
    // side table); string keys never appear here.
    let keys = heap.own_symbol_keys(obj);
    // The symbol handles are GC-managed: hold them (and the result
    // array) across the per-element allocations below, each of which
    // may collect.
    let base = heap.temp_roots_len();
    for &s in &keys {
        heap.protect(Value::Symbol(s));
    }
    let arr = new_array(heap)?;
    heap.protect(Value::Object(arr));
    for (i, s) in keys.iter().enumerate() {
        set_idx_raw(heap, arr, i as u64, Value::Symbol(*s))?;
    }
    set_len(heap, arr, keys.len() as u64)?;
    heap.release_temp(base);
    Ok(Value::Object(arr))
}

fn object_assign(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let target = match get_arg(args, 0) {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Object.assign: target must be an object".into(),
            ));
        }
    };
    for src in args.iter().skip(1) {
        let src_obj = match src {
            Value::Object(o) => *o,
            Value::Undefined | Value::Null => continue,
            // v1 gap: no ToObject for primitives.
            _ => {
                return Err(NativeError::TypeError(
                    "Object.assign: source must be an object (v1 gap: no primitive wrapping)"
                        .into(),
                ));
            }
        };
        let keys = own_enumerable_keys(heap, src_obj);
        for k in keys {
            let v = get_own(heap, src_obj, k.as_wtf8()).unwrap_or(Value::Undefined);
            set_prop(heap, target, k.as_wtf8(), v).map_err(|_| {
                NativeError::TypeError("Object.assign: target is not extensible".into())
            })?;
        }
        // Spec `Object.assign` also copies own *enumerable* symbol-keyed
        // properties. Data properties copy by value; symbol-keyed
        // accessors are skipped exactly like string-keyed ones (the
        // string path above reads data only via `get_own`).
        for sym in heap.own_enumerable_symbol_keys(src_obj) {
            let v = heap
                .get_property_key(src_obj, PropKey::Symbol(sym))
                .map(|p| p.value)
                .unwrap_or(Value::Undefined);
            set_prop_key(heap, target, PropKey::Symbol(sym), v).map_err(|_| {
                NativeError::TypeError("Object.assign: target is not extensible".into())
            })?;
        }
    }
    Ok(Value::Object(target))
}

/// A parsed property descriptor: data or accessor.
enum Descriptor {
    Data(Property),
    Accessor(AccessorProp),
}

/// A partially-specified property descriptor as read from the descriptor
/// object: each field is `Some` when present, `None` when absent. Absent
/// fields are filled from the current property (or spec defaults when the
/// property is new) by [`complete_descriptor`], per
/// `ValidateAndApplyPropertyDescriptor` — they must not reset to defaults.
struct PartialDescriptor {
    value: Option<Value>,
    writable: Option<bool>,
    get: Option<Value>,
    set: Option<Value>,
    enumerable: Option<bool>,
    configurable: Option<bool>,
}

impl PartialDescriptor {
    /// `true` when the descriptor carries accessor fields (`get`/`set`).
    fn is_accessor(&self) -> bool {
        self.get.is_some() || self.set.is_some()
    }

    /// `true` when the descriptor carries data fields (`value`/`writable`).
    fn is_data(&self) -> bool {
        self.value.is_some() || self.writable.is_some()
    }
}

/// Shared descriptor reader for `defineProperty`/`defineProperties`:
/// converts a descriptor object into a [`PartialDescriptor`]. A descriptor
/// with both accessor (`get`/`set`) and data (`value`/`writable`) fields is
/// a `TypeError`, as are non-callable getters/setters.
fn read_descriptor(heap: &Heap, desc_obj: Gc<Object>) -> Result<PartialDescriptor, NativeError> {
    if (has_own(heap, desc_obj, Wtf8::from_str("get"))
        || has_own(heap, desc_obj, Wtf8::from_str("set")))
        && (has_own(heap, desc_obj, Wtf8::from_str("value"))
            || has_own(heap, desc_obj, Wtf8::from_str("writable")))
    {
        return Err(NativeError::TypeError(
            "property descriptor cannot be both a data and an accessor descriptor".into(),
        ));
    }
    let flag = |name: &Wtf8| get_own(heap, desc_obj, name).map(|v| v.is_truthy());
    let get = get_own(heap, desc_obj, Wtf8::from_str("get"));
    let set = get_own(heap, desc_obj, Wtf8::from_str("set"));
    for (name, v) in [("get", get), ("set", set)] {
        if !matches!(v, Some(Value::Undefined) | None) && !is_callable(heap, v.unwrap()) {
            return Err(NativeError::TypeError(format!(
                "property descriptor '{name}' must be callable or undefined"
            )));
        }
    }
    Ok(PartialDescriptor {
        value: get_own(heap, desc_obj, Wtf8::from_str("value")),
        writable: flag(Wtf8::from_str("writable")),
        get,
        set,
        enumerable: flag(Wtf8::from_str("enumerable")),
        configurable: flag(Wtf8::from_str("configurable")),
    })
}

/// Complete a partial descriptor against the current own property per
/// `ValidateAndApplyPropertyDescriptor`. Absent fields are preserved from
/// the current property when its kind matches; converting between data and
/// accessor descriptors resets the other kind's fields to spec defaults
/// (`value: undefined`, `writable: false`, `get`/`set: undefined`), while
/// `enumerable`/`configurable` always carry over. A brand-new property
/// fills every absent field with spec defaults.
fn complete_descriptor(
    heap: &Heap,
    obj: Gc<Object>,
    key: PropKey<'_>,
    desc: &PartialDescriptor,
) -> Descriptor {
    let current_data = heap.get_property_key(obj, key);
    let current_accessor = if current_data.is_none() {
        heap.get_accessor_key(obj, key)
    } else {
        None
    };
    match (current_data, current_accessor) {
        (Some(p), _) if !desc.is_accessor() => Descriptor::Data(Property {
            value: desc.value.unwrap_or(p.value),
            writable: desc.writable.unwrap_or(p.writable),
            enumerable: desc.enumerable.unwrap_or(p.enumerable),
            configurable: desc.configurable.unwrap_or(p.configurable),
        }),
        (_, Some(a)) if !desc.is_data() => Descriptor::Accessor(AccessorProp {
            get: desc.get.unwrap_or(a.get),
            set: desc.set.unwrap_or(a.set),
            enumerable: desc.enumerable.unwrap_or(a.enumerable),
            configurable: desc.configurable.unwrap_or(a.configurable),
        }),
        // Data -> accessor: accessor fields reset to defaults.
        (Some(p), _) => Descriptor::Accessor(AccessorProp {
            get: desc.get.unwrap_or(Value::Undefined),
            set: desc.set.unwrap_or(Value::Undefined),
            enumerable: desc.enumerable.unwrap_or(p.enumerable),
            configurable: desc.configurable.unwrap_or(p.configurable),
        }),
        // Accessor -> data: data fields reset to defaults.
        (_, Some(a)) => Descriptor::Data(Property {
            value: desc.value.unwrap_or(Value::Undefined),
            writable: desc.writable.unwrap_or(false),
            enumerable: desc.enumerable.unwrap_or(a.enumerable),
            configurable: desc.configurable.unwrap_or(a.configurable),
        }),
        (None, None) if desc.is_accessor() => Descriptor::Accessor(AccessorProp {
            get: desc.get.unwrap_or(Value::Undefined),
            set: desc.set.unwrap_or(Value::Undefined),
            enumerable: desc.enumerable.unwrap_or(false),
            configurable: desc.configurable.unwrap_or(false),
        }),
        (None, None) => Descriptor::Data(Property {
            value: desc.value.unwrap_or(Value::Undefined),
            writable: desc.writable.unwrap_or(false),
            enumerable: desc.enumerable.unwrap_or(false),
            configurable: desc.configurable.unwrap_or(false),
        }),
    }
}

/// `Object.defineProperty`/`defineProperties` core: read the descriptor
/// object, complete absent fields from the current property per
/// `ValidateAndApplyPropertyDescriptor`, then define with validation.
fn define_from_descriptor(
    heap: &mut Heap,
    obj: Gc<Object>,
    key: PropKey<'_>,
    desc_obj: Gc<Object>,
) -> Result<(), NativeError> {
    let desc = read_descriptor(heap, desc_obj)?;
    // Array exotic `[[DefineOwnProperty]]` (spec 10.4.2): string keys
    // only — symbol keys never address `"length"` or indices. The heap
    // helper completes the partial descriptor against the existing own
    // property, validates it, extends/truncates `length`, and enforces
    // the Uint32 rule. A rejected define throws `TypeError` here, per
    // `Object.defineProperty`; a non-Uint32 `length` throws `RangeError`
    // from inside the helper.
    if let PropKey::Str(name) = key
        && heap.get(obj).kind == ObjectKind::Array
        && is_array_exotic_key(name)
    {
        let array_desc = ArrayDefineDesc {
            value: desc.value,
            writable: desc.writable,
            enumerable: desc.enumerable,
            configurable: desc.configurable,
            get: desc.get,
            set: desc.set,
        };
        return array_define_own_property(heap, obj, name, array_desc);
    }
    match complete_descriptor(heap, obj, key, &desc) {
        Descriptor::Data(p) => define_prop(heap, obj, key, p),
        Descriptor::Accessor(a) => define_accessor_prop(heap, obj, key, a),
    }
}

/// Define an own accessor property with `Object.defineProperty` semantics:
/// non-configurable properties cannot change kind (data <-> accessor) or
/// enumerability, and non-configurable accessors keep their getter/setter.
pub(crate) fn define_accessor_prop(
    heap: &mut Heap,
    obj: Gc<Object>,
    key: PropKey<'_>,
    prop: AccessorProp,
) -> Result<(), NativeError> {
    match heap.get_property_key(obj, key) {
        Some(existing) => {
            // Data -> accessor conversion: allowed only when configurable.
            if !existing.configurable {
                return Err(NativeError::TypeError(format!(
                    "cannot convert non-configurable data property '{}' to an accessor",
                    heap.format_key(key)
                )));
            }
        }
        None => match heap.get_accessor_key(obj, key) {
            Some(existing) => {
                if !existing.configurable {
                    if prop.configurable {
                        return Err(NativeError::TypeError(format!(
                            "cannot redefine non-configurable property '{}'",
                            heap.format_key(key)
                        )));
                    }
                    if prop.enumerable != existing.enumerable {
                        return Err(NativeError::TypeError(format!(
                            "cannot change enumerability of non-configurable property '{}'",
                            heap.format_key(key)
                        )));
                    }
                    if prop.get != existing.get || prop.set != existing.set {
                        return Err(NativeError::TypeError(format!(
                            "cannot change getter/setter of non-configurable property '{}'",
                            heap.format_key(key)
                        )));
                    }
                }
            }
            None => {
                if !heap.get(obj).extensible {
                    return Err(NativeError::TypeError(format!(
                        "cannot define property '{}' on non-extensible object",
                        heap.format_key(key)
                    )));
                }
            }
        },
    }
    heap.define_accessor_key(obj, key, prop)?;
    Ok(())
}

fn object_define_property(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let obj = match get_arg(args, 0) {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Object.defineProperty: target must be an object".into(),
            ));
        }
    };
    let key = to_property_key(heap, get_arg(args, 1))?;
    let desc = match get_arg(args, 2) {
        Value::Object(d) => d,
        _ => {
            return Err(NativeError::TypeError(
                "Object.defineProperty: descriptor must be an object".into(),
            ));
        }
    };
    // `define_from_descriptor` allocates while the key borrow is live:
    // root the handle across the call (released on both paths).
    let key_base = heap.protect(key.as_value());
    let r = define_from_descriptor(heap, obj, key.as_prop_key(), desc);
    heap.release_temp(key_base);
    r?;
    Ok(Value::Object(obj))
}

fn object_define_properties_impl(
    heap: &mut Heap,
    target: Value,
    props: Value,
) -> Result<(), NativeError> {
    let obj = match target {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Object.defineProperties: target must be an object".into(),
            ));
        }
    };
    let props_obj = match props {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Object.defineProperties: properties must be an object".into(),
            ));
        }
    };
    let keys = own_enumerable_keys(heap, props_obj);
    for k in keys {
        let desc = get_own(heap, props_obj, k.as_wtf8()).unwrap_or(Value::Undefined);
        let desc_obj = match desc {
            Value::Object(d) => d,
            _ => {
                return Err(NativeError::TypeError(
                    "Object.defineProperties: descriptor must be an object".into(),
                ));
            }
        };
        define_from_descriptor(heap, obj, PropKey::Str(k.as_wtf8()), desc_obj)?;
    }
    Ok(())
}

fn object_define_properties(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let target = get_arg(args, 0);
    object_define_properties_impl(heap, target, get_arg(args, 1))?;
    Ok(target)
}

fn descriptor_to_object(heap: &mut Heap, p: Property) -> Result<Gc<Object>, NativeError> {
    let o = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    set_prop(heap, o, Wtf8::from_str("value"), p.value)?;
    set_prop(heap, o, Wtf8::from_str("writable"), Value::Bool(p.writable))?;
    set_prop(
        heap,
        o,
        Wtf8::from_str("enumerable"),
        Value::Bool(p.enumerable),
    )?;
    set_prop(
        heap,
        o,
        Wtf8::from_str("configurable"),
        Value::Bool(p.configurable),
    )?;
    Ok(o)
}

fn accessor_descriptor_to_object(
    heap: &mut Heap,
    a: AccessorProp,
) -> Result<Gc<Object>, NativeError> {
    let o = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    set_prop(heap, o, Wtf8::from_str("get"), a.get)?;
    set_prop(heap, o, Wtf8::from_str("set"), a.set)?;
    set_prop(
        heap,
        o,
        Wtf8::from_str("enumerable"),
        Value::Bool(a.enumerable),
    )?;
    set_prop(
        heap,
        o,
        Wtf8::from_str("configurable"),
        Value::Bool(a.configurable),
    )?;
    Ok(o)
}

fn object_get_own_property_descriptor(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let obj = match get_arg(args, 0) {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Object.getOwnPropertyDescriptor: target must be an object".into(),
            ));
        }
    };
    let key = to_property_key(heap, get_arg(args, 1))?;
    let key = key.as_prop_key();
    // `get_property_key`/`get_accessor_key` take `&Heap`: no collection
    // can run while the key is borrowed, so the handle needs no root
    // here. The allocating `descriptor_to_object` calls below run after
    // the last key borrow.
    match heap.get_property_key(obj, key) {
        Some(p) => Ok(Value::Object(descriptor_to_object(heap, p)?)),
        None => match heap.get_accessor_key(obj, key) {
            Some(a) => Ok(Value::Object(accessor_descriptor_to_object(heap, a)?)),
            None => Ok(Value::Undefined),
        },
    }
}

fn object_get_prototype_of(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let obj = match get_arg(args, 0) {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Object.getPrototypeOf: argument must be an object".into(),
            ));
        }
    };
    Ok(heap.get(obj).prototype)
}

fn object_set_prototype_of(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let obj = match get_arg(args, 0) {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Object.setPrototypeOf: target must be an object".into(),
            ));
        }
    };
    match get_arg(args, 1) {
        Value::Object(_) | Value::Null => {}
        _ => {
            return Err(NativeError::TypeError(
                "Object.setPrototypeOf: prototype must be an object or null".into(),
            ));
        }
    }
    let proto = get_arg(args, 1);
    check_prototype_cycle(heap, obj, proto)?;
    heap.set_prototype(obj, proto);
    Ok(Value::Object(obj))
}

fn object_has_own(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let obj = match get_arg(args, 0) {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Object.hasOwn: argument must be an object".into(),
            ));
        }
    };
    let key = to_property_key(heap, get_arg(args, 1))?;
    // `get_property_key` takes `&Heap`: no allocation, so the handle
    // needs no root.
    Ok(Value::Bool(
        heap.get_property_key(obj, key.as_prop_key()).is_some(),
    ))
}

/// `[[SetIntegrityLevel]]`: freeze (`frozen=true`) or seal.
fn set_integrity_level(heap: &mut Heap, obj: Gc<Object>, frozen: bool) -> Result<(), NativeError> {
    // Copy keys first: define_property must not run while iterating the map.
    let keys: Vec<Str> = heap.get(obj).properties.iter().map(|(k, _)| *k).collect();
    for k in keys {
        let Some(p) = heap.get_property(obj, k.as_wtf8()) else {
            continue;
        };
        if p.configurable {
            heap.define_property(
                obj,
                k.as_wtf8(),
                Property {
                    configurable: false,
                    writable: if frozen { false } else { p.writable },
                    ..p
                },
            )?;
        } else if frozen && p.writable {
            heap.define_property(
                obj,
                k.as_wtf8(),
                Property {
                    writable: false,
                    ..p
                },
            )?;
        }
    }
    heap.prevent_extensions(obj);
    Ok(())
}

/// `[[TestIntegrityLevel]]`: all properties non-configurable, and
/// non-writable when testing `frozen`.
fn test_integrity_level(heap: &Heap, obj: Gc<Object>, frozen: bool) -> bool {
    if heap.get(obj).extensible {
        return false;
    }
    heap.get(obj)
        .properties
        .iter()
        .all(|(_, p)| !p.configurable && (!frozen || !p.writable))
}

fn object_freeze(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let obj = match get_arg(args, 0) {
        Value::Object(o) => o,
        _ => return Ok(get_arg(args, 0)),
    };
    set_integrity_level(heap, obj, true)?;
    Ok(Value::Object(obj))
}

fn object_seal(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let obj = match get_arg(args, 0) {
        Value::Object(o) => o,
        _ => return Ok(get_arg(args, 0)),
    };
    set_integrity_level(heap, obj, false)?;
    Ok(Value::Object(obj))
}

fn object_prevent_extensions(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let obj = match get_arg(args, 0) {
        Value::Object(o) => o,
        // Per spec (19.1.2.13), a non-object argument is returned as-is.
        _ => return Ok(get_arg(args, 0)),
    };
    heap.prevent_extensions(obj);
    Ok(Value::Object(obj))
}

fn object_is_frozen(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    match get_arg(args, 0) {
        Value::Object(o) => Ok(Value::Bool(test_integrity_level(heap, o, true))),
        _ => Ok(Value::Bool(true)),
    }
}

fn object_is_sealed(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    match get_arg(args, 0) {
        Value::Object(o) => Ok(Value::Bool(test_integrity_level(heap, o, false))),
        _ => Ok(Value::Bool(true)),
    }
}

fn object_from_entries(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let iterable = get_arg(args, 0);
    let proto = heap
        .object_prototype
        .map(Value::Object)
        .unwrap_or(Value::Null);
    let obj = heap.alloc_object(Object {
        kind: ObjectKind::Ordinary,
        prototype: proto,
        ..Default::default()
    })?;
    let entries_obj = match iterable {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Object.fromEntries: argument must be an iterable of pairs (v1: array of pairs)"
                    .into(),
            ));
        }
    };
    // v1: the iterable must be array-like (an Array or object with length).
    let len = get_length(heap, entries_obj)?;
    for i in 0..len {
        let pair =
            get_own(heap, entries_obj, Wtf8::from_str(&i.to_string())).unwrap_or(Value::Undefined);
        let pair_obj = match pair {
            Value::Object(o) => o,
            _ => {
                return Err(NativeError::TypeError(
                    "Object.fromEntries: each entry must be an object".into(),
                ));
            }
        };
        let k = get_own(heap, pair_obj, Wtf8::from_str("0")).unwrap_or(Value::Undefined);
        let v = get_own(heap, pair_obj, Wtf8::from_str("1")).unwrap_or(Value::Undefined);
        let key = to_property_key(heap, k)?;
        // `set_prop_key` allocates while the key borrow is live: root the
        // handle across the call (released on both paths).
        let key_base = heap.protect(key.as_value());
        let r = set_prop_key(heap, obj, key.as_prop_key(), v);
        heap.release_temp(key_base);
        r?;
    }
    Ok(Value::Object(obj))
}

fn object_group_by(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let items = get_arg(args, 0);
    let callback = match args.get(1) {
        Some(cb) => *cb,
        None => {
            return Err(NativeError::TypeError(
                "Object.groupBy requires a callback".into(),
            ));
        }
    };
    let items_obj = match items {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Object.groupBy: items must be an object".into(),
            ));
        }
    };
    // The result has a null prototype, per spec.
    let groups = heap.alloc_object(Object {
        kind: ObjectKind::Ordinary,
        prototype: Value::Null,
        ..Default::default()
    })?;
    let len = get_length(heap, items_obj)?;
    for i in 0..len {
        let v =
            get_own(heap, items_obj, Wtf8::from_str(&i.to_string())).unwrap_or(Value::Undefined);
        let key_v = call_function(heap, callback, Value::Undefined, &[v, num(i as f64), items])?;
        let key = to_property_key(heap, key_v)?;
        // Root the key across the match: the miss arm allocates
        // (`new_array`, `set_prop_key`) while the key borrow is still live
        // below. Released on both paths via `group_res`.
        let key_base = heap.protect(key.as_value());
        let key = key.as_prop_key();
        let group_res: Result<Gc<Object>, NativeError> =
            match heap.get_property_key(groups, key).map(|p| p.value) {
                Some(Value::Object(g)) => Ok(g),
                _ => {
                    let g = new_array(heap)?;
                    set_prop_key(heap, groups, key, Value::Object(g))?;
                    Ok(g)
                }
            };
        heap.release_temp(key_base);
        let group = group_res?;
        let glen = array_length(heap, group);
        set_idx(heap, group, glen, v)?;
    }
    Ok(Value::Object(groups))
}

// ---------------------------------------------------------------------------
// Array constructor + statics
// ---------------------------------------------------------------------------

fn array_constructor(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let arr = new_array(heap)?;
    // ECMA-262 OrdinaryCreateFromConstructor(newTarget,
    // "%Array.prototype%"): a derived `class D extends Array` calling
    // `super()` must get D.prototype as the instance's prototype so
    // prototype methods resolve (see object_constructor above).
    //
    // `arr` is freshly allocated and not yet reachable from any root:
    // hold it across the `prototype` lookup and `set_prototype`. The
    // lookup is side-effect-free today (`get_prop_chain` performs raw
    // property reads), but any future allocating or user-code-invoking
    // step inserted here would collect the unrooted array out from
    // under the `set_prototype`.
    with_temp_root(heap, Value::Object(arr), |heap| {
        if let Some(Value::Object(nt)) = active_new_target()
            && let Some(Value::Object(p)) = get_prop_chain(heap, nt, Wtf8::from_str("prototype"))
        {
            heap.set_prototype(arr, Value::Object(p));
        }
        Ok::<(), NativeError>(())
    })?;
    match args {
        [] => Ok(Value::Object(arr)),
        [single] => match single {
            Value::Int(i) => {
                if *i < 0 {
                    return Err(NativeError::RangeError("invalid array length".into()));
                }
                set_len(heap, arr, *i as u64)?;
                Ok(Value::Object(arr))
            }
            Value::Float(f) => {
                // Spec: only lengths *above* 2^32-1 throw; 4294967295
                // itself is a valid array length.
                if f.fract() != 0.0 || *f < 0.0 || *f > 4294967295.0 {
                    return Err(NativeError::RangeError("invalid array length".into()));
                }
                set_len(heap, arr, *f as u64)?;
                Ok(Value::Object(arr))
            }
            v => {
                set_idx(heap, arr, 0, *v)?;
                Ok(Value::Object(arr))
            }
        },
        many => {
            for (i, v) in many.iter().enumerate() {
                set_idx_raw(heap, arr, i as u64, *v)?;
            }
            set_len(heap, arr, many.len() as u64)?;
            Ok(Value::Object(arr))
        }
    }
}

fn array_is_array(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    match get_arg(args, 0) {
        Value::Object(o) => Ok(Value::Bool(_heap.get(o).kind == ObjectKind::Array)),
        _ => Ok(Value::Bool(false)),
    }
}

fn array_from(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let src = match get_arg(args, 0) {
        Value::Object(o) => o,
        Value::Undefined | Value::Null => {
            return Err(NativeError::TypeError(
                "Array.from: source must be an object".into(),
            ));
        }
        _ => {
            return Err(NativeError::TypeError(
                "Array.from: source must be an object (v1 gap: no iterable protocol)".into(),
            ));
        }
    };
    // v1: no iterator protocol; only array-likes (objects with length).
    let len = get_length(heap, src)?;
    let map_fn = get_arg(args, 1);
    let has_map = !matches!(map_fn, Value::Undefined);
    if has_map && !is_callable(heap, map_fn) {
        return Err(NativeError::TypeError(
            "Array.from: map function is not callable".into(),
        ));
    }
    let arr = new_array_with_length(heap, len)?;
    for i in 0..len {
        let mut v = get_own(heap, src, Wtf8::from_str(&i.to_string())).unwrap_or(Value::Undefined);
        if has_map {
            v = call_function(heap, map_fn, Value::Undefined, &[v, num(i as f64)])?;
        }
        set_idx_raw(heap, arr, i, v)?;
    }
    Ok(Value::Object(arr))
}

fn array_of(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let arr = new_array(heap)?;
    for (i, v) in args.iter().enumerate() {
        set_idx_raw(heap, arr, i as u64, *v)?;
    }
    set_len(heap, arr, args.len() as u64)?;
    Ok(Value::Object(arr))
}

// ---------------------------------------------------------------------------
// Array prototype methods
// ---------------------------------------------------------------------------

/// Receiver as object plus its `length` (`ToLength` of own `"length"`).
fn array_receiver(
    heap: &mut Heap,
    this: Value,
    name: &str,
) -> Result<(Gc<Object>, u64), NativeError> {
    let obj = this_object(heap, this).map_err(|_| {
        NativeError::TypeError(format!(
            "Array.prototype.{name} called on incompatible receiver"
        ))
    })?;
    let len = get_length(heap, obj)?;
    Ok((obj, len))
}

/// Overwrite the `"length"` data property (standard attributes).
fn set_length_prop(heap: &mut Heap, obj: Gc<Object>, len: u64) -> Result<(), AllocError> {
    heap.define_property_str(
        obj,
        "length",
        Property {
            value: num(len as f64),
            writable: true,
            enumerable: false,
            configurable: false,
        },
    )
}

/// Clamp a possibly-negative index into `[0, len]`.
fn clamp_index(v: Value, len: u64) -> Result<u64, NativeError> {
    let i = to_integer_or_infinity(v)?;
    if i < 0.0 {
        Ok(len.saturating_sub((-i) as u64))
    } else if i.is_infinite() || i >= len as f64 {
        Ok(len)
    } else {
        Ok(i as u64)
    }
}

fn array_push(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, mut len) = array_receiver(heap, this, "push")?;
    let is_array = heap.get(obj).kind == ObjectKind::Array;
    for v in args {
        if is_array {
            set_idx(heap, obj, len, *v)?;
        } else {
            set_prop(heap, obj, Wtf8::from_str(&len.to_string()), *v).map_err(|_| {
                NativeError::TypeError("Array.prototype.push: target not extensible".into())
            })?;
        }
        len += 1;
    }
    if is_array {
        set_len(heap, obj, len)?;
    } else {
        set_length_prop(heap, obj, len)?;
    }
    Ok(num(len as f64))
}

fn array_pop(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let (obj, len) = array_receiver(heap, this, "pop")?;
    if len == 0 {
        return Ok(Value::Undefined);
    }
    let i = len - 1;
    let v = get_own(heap, obj, Wtf8::from_str(&i.to_string())).unwrap_or(Value::Undefined);
    delete_prop(heap, obj, Wtf8::from_str(&i.to_string()));
    if heap.get(obj).kind == ObjectKind::Array {
        set_len(heap, obj, i)?;
    } else {
        set_length_prop(heap, obj, i)?;
    }
    Ok(v)
}

fn array_shift(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let (obj, len) = array_receiver(heap, this, "shift")?;
    if len == 0 {
        return Ok(Value::Undefined);
    }
    let first = get_own(heap, obj, Wtf8::from_str("0")).unwrap_or(Value::Undefined);
    for k in 1..len {
        let v = get_own(heap, obj, Wtf8::from_str(&k.to_string()));
        match v {
            Some(v) => set_prop(heap, obj, Wtf8::from_str(&(k - 1).to_string()), v)?,
            None => {
                delete_prop(heap, obj, Wtf8::from_str(&(k - 1).to_string()));
            }
        }
    }
    delete_prop(heap, obj, Wtf8::from_str(&(len - 1).to_string()));
    if heap.get(obj).kind == ObjectKind::Array {
        set_len(heap, obj, len - 1)?;
    } else {
        set_length_prop(heap, obj, len - 1)?;
    }
    Ok(first)
}

fn array_unshift(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, len) = array_receiver(heap, this, "unshift")?;
    let n = args.len() as u64;
    if n > 0 {
        for k in (0..len).rev() {
            let v = get_own(heap, obj, Wtf8::from_str(&k.to_string()));
            match v {
                Some(v) => set_prop(heap, obj, Wtf8::from_str(&(k + n).to_string()), v)?,
                None => {
                    delete_prop(heap, obj, Wtf8::from_str(&(k + n).to_string()));
                }
            }
        }
        for (j, v) in args.iter().enumerate() {
            set_prop(heap, obj, Wtf8::from_str(&j.to_string()), *v)?;
        }
    }
    let new_len = len + n;
    if heap.get(obj).kind == ObjectKind::Array {
        set_len(heap, obj, new_len)?;
    } else {
        set_length_prop(heap, obj, new_len)?;
    }
    Ok(num(new_len as f64))
}

fn array_slice(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, len) = array_receiver(heap, this, "slice")?;
    let start = clamp_index(get_arg(args, 0), len)?;
    let end = if args.len() >= 2 {
        clamp_index(get_arg(args, 1), len)?
    } else {
        len
    };
    let count = end.saturating_sub(start);
    // ArrayCreate(count) throws RangeError when count exceeds 2^32 - 1;
    // check before allocating.
    if count > 4294967295 {
        return Err(NativeError::RangeError(
            "Array.prototype.slice: result length exceeds 2^32 - 1".into(),
        ));
    }
    let mut n = 0u64;
    let is_view = crate::builtin_reflect_proxy::is_typed_array_view(heap, obj);
    let (a, _) = array_species_create(heap, obj, count, "Array.prototype.slice")?;
    let guard = heap.protect(Value::Object(a));
    let result = (|| -> Result<(), NativeError> {
        for k in start..end {
            // Typed-array exotic (spec 10.4.5): a view's indexed elements
            // live in the backing store, not as own properties, so
            // `get_own` sees missing indices
            // (`[].slice.call(uint8array)` came back holey). Route
            // integer indices through the exotic `[[Get]]` for views;
            // ordinary objects keep the own-property read (which
            // preserves holes).
            let v = if is_view {
                Some(crate::builtin_reflect_proxy::typed_index_get(heap, obj, k)?)
            } else {
                get_own(heap, obj, Wtf8::from_str(&k.to_string()))
            };
            // Holes stay holes: only present elements are written.
            if let Some(v) = v {
                species_set_idx(heap, a, n, v)?;
            }
            n += 1;
        }
        Ok(())
    })();
    heap.release_temp(guard);
    result?;
    // No trailing length write: spec never sets `length` after the loop
    // for `slice`. The default species path constructed the result with
    // `length` already `count`; a hostile Array-kind result extended it
    // through the exotic writes; a hostile non-Array result gets no
    // `length` at all.
    Ok(Value::Object(a))
}

fn array_splice(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, len) = array_receiver(heap, this, "splice")?;
    let actual_start = clamp_index(get_arg(args, 0), len)?;
    let item_count = args.len().saturating_sub(2) as u64;
    let actual_delete = if args.len() < 2 {
        len - actual_start
    } else {
        let dc = to_integer_or_infinity(get_arg(args, 1))?;
        if dc <= 0.0 {
            0
        } else if dc.is_infinite() {
            len - actual_start
        } else {
            (dc as u64).min(len - actual_start)
        }
    };
    // Snapshot the deleted segment into a fresh array. Spec 23.1.3.30
    // copies with `Get` + `CreateDataPropertyOrThrow` and no
    // `HasProperty` guard, so holes are *filled* with present
    // `undefined`, not preserved.
    if actual_delete > 4294967295 {
        return Err(NativeError::RangeError(
            "Array.prototype.splice: result length exceeds 2^32 - 1".into(),
        ));
    }
    // Never buffer the segment in a Rust `Vec` sized from the untrusted
    // `actual_delete`: up to 2^32-1 entries is a multi-gigabyte
    // allocation that aborts the process in the allocator, whether
    // pre-sized or grown amortized. Values go straight into the result
    // array, which is temp-rooted across the loop.
    let (deleted_arr, _) =
        array_species_create(heap, obj, actual_delete, "Array.prototype.splice")?;
    let dguard = heap.protect(Value::Object(deleted_arr));
    let snapshot = (|| -> Result<(), NativeError> {
        if actual_delete <= FRESH_ARRAY_DENSE_LIMIT {
            for k in 0..actual_delete {
                // Holes read as `undefined` and are created present.
                let v = get_own(heap, obj, Wtf8::from_str(&((actual_start + k).to_string())))
                    .unwrap_or(Value::Undefined);
                species_set_idx(heap, deleted_arr, k, v)?;
            }
        } else {
            // Sparse hostile-length path: `length` is set, and only the
            // source's own index properties in the deleted range are
            // copied; everything else stays a hole. Documented
            // deviation: spec fills every index with present
            // `undefined`, and prototype-chain getters do not fire.
            for key in own_all_keys(heap, obj) {
                let k = match array_index_key(key.as_wtf8()) {
                    Some(k) => k as u64,
                    None => continue,
                };
                if k >= actual_start
                    && k < actual_start + actual_delete
                    && let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string()))
                {
                    species_set_idx(heap, deleted_arr, k - actual_start, v)?;
                }
            }
        }
        Ok(())
    })();
    heap.release_temp(dguard);
    snapshot?;
    if item_count < actual_delete {
        let shift = actual_delete - item_count;
        let new_len = len - shift;
        if len - actual_delete > FRESH_ARRAY_DENSE_LIMIT || shift > FRESH_ARRAY_DENSE_LIMIT {
            // Sparse hostile-length path: the source range
            // `[actual_start + actual_delete, len)` and the tail
            // `[new_len, len)` are far too long to walk (up to 2^32 - 1
            // iterations). Move only the source's own index properties
            // down by `shift`, then replicate the dense loop's
            // hole/tail semantics: a destination whose source is a hole
            // is deleted, and the tail is deleted. Values are
            // snapshotted before any write: a destination can coincide
            // with a not-yet-moved source.
            let mut src_keys: Vec<u64> = Vec::new();
            let mut moves: Vec<(u64, Value)> = Vec::new();
            for key in own_all_keys(heap, obj) {
                let k = match array_index_key(key.as_wtf8()) {
                    Some(k) => k as u64,
                    None => continue,
                };
                if k >= actual_start + actual_delete && k < len {
                    if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string())) {
                        moves.push((k - shift, v));
                    }
                    src_keys.push(k);
                }
            }
            src_keys.sort_unstable();
            for (d, v) in moves {
                set_prop(heap, obj, Wtf8::from_str(&d.to_string()), v)?;
            }
            for key in own_all_keys(heap, obj) {
                let k = match array_index_key(key.as_wtf8()) {
                    Some(k) => k as u64,
                    None => continue,
                };
                if (k >= new_len && k < len)
                    || (k >= actual_start + item_count
                        && k < new_len
                        && src_keys.binary_search(&(k + shift)).is_err())
                {
                    delete_prop(heap, obj, Wtf8::from_str(&k.to_string()));
                }
            }
        } else {
            for k in actual_start..(len - actual_delete) {
                let v = get_own(heap, obj, Wtf8::from_str(&(k + actual_delete).to_string()));
                match v {
                    Some(v) => {
                        set_prop(heap, obj, Wtf8::from_str(&(k + item_count).to_string()), v)?
                    }
                    None => {
                        delete_prop(heap, obj, Wtf8::from_str(&(k + item_count).to_string()));
                    }
                }
            }
            for k in (len - shift)..len {
                delete_prop(heap, obj, Wtf8::from_str(&k.to_string()));
            }
        }
    } else if item_count > actual_delete {
        let shift = item_count - actual_delete;
        if len - actual_delete > FRESH_ARRAY_DENSE_LIMIT {
            // Sparse hostile-length path: move only the source's own
            // index properties up by `shift` (descending destination
            // order mirrors the dense loop), then delete destinations
            // whose source was a hole. Values are snapshotted before
            // any write so a move never reads an already-moved source.
            let mut src_keys: Vec<u64> = Vec::new();
            let mut moves: Vec<(u64, Value)> = Vec::new();
            for key in own_all_keys(heap, obj) {
                let k = match array_index_key(key.as_wtf8()) {
                    Some(k) => k as u64,
                    None => continue,
                };
                if k >= actual_start + actual_delete && k < len {
                    if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string())) {
                        moves.push((k + shift, v));
                    }
                    src_keys.push(k);
                }
            }
            src_keys.sort_unstable();
            moves.sort_by_key(|a| std::cmp::Reverse(a.0));
            for (d, v) in moves {
                set_prop(heap, obj, Wtf8::from_str(&d.to_string()), v)?;
            }
            for key in own_all_keys(heap, obj) {
                let k = match array_index_key(key.as_wtf8()) {
                    Some(k) => k as u64,
                    None => continue,
                };
                // `k - shift` cannot underflow: `k >= actual_start +
                // item_count >= shift`.
                if k >= actual_start + item_count
                    && k < len + shift
                    && src_keys.binary_search(&(k - shift)).is_err()
                {
                    delete_prop(heap, obj, Wtf8::from_str(&k.to_string()));
                }
            }
        } else {
            for k in (actual_start..(len - actual_delete)).rev() {
                let v = get_own(heap, obj, Wtf8::from_str(&(k + actual_delete).to_string()));
                match v {
                    Some(v) => {
                        set_prop(heap, obj, Wtf8::from_str(&(k + item_count).to_string()), v)?
                    }
                    None => {
                        delete_prop(heap, obj, Wtf8::from_str(&(k + item_count).to_string()));
                    }
                }
            }
            let _ = shift;
        }
    }
    for (j, v) in args.iter().skip(2).enumerate() {
        set_prop(
            heap,
            obj,
            Wtf8::from_str(&(actual_start + j as u64).to_string()),
            *v,
        )?;
    }
    let new_len = len - actual_delete + item_count;
    if heap.get(obj).kind == ObjectKind::Array {
        set_len(heap, obj, new_len)?;
    } else {
        set_length_prop(heap, obj, new_len)?;
    }
    // No trailing length write on the deleted array: spec never sets
    // `length` after the loop for `splice`. The default species path
    // constructed it with `length` already `actual_delete`; a hostile
    // Array-kind result extended it through the exotic writes; a hostile
    // non-Array result gets no `length` at all.
    Ok(Value::Object(deleted_arr))
}

fn array_concat(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let obj = this_object(heap, this).map_err(|_| {
        NativeError::TypeError("Array.prototype.concat called on incompatible receiver".into())
    })?;
    // `ArraySpeciesCreate(O, 0)` runs before any element is read (spec
    // order); the result is temp-rooted across the loop.
    let (a, _) = array_species_create(heap, obj, 0, "Array.prototype.concat")?;
    let guard = heap.protect(Value::Object(a));
    let result = (|| -> Result<(), NativeError> {
        let mut n = 0u64;
        // v1: no Symbol.isConcatSpreadable; spread exactly the isArray arguments.
        let push_value = |heap: &mut Heap, v: Value, n: &mut u64| -> Result<(), NativeError> {
            match v {
                Value::Object(o) if heap.get(o).kind == ObjectKind::Array => {
                    let l = array_length(heap, o);
                    for i in 0..l {
                        // Holes stay holes: only present elements are written.
                        if let Some(v) = get_own(heap, o, Wtf8::from_str(&i.to_string())) {
                            species_set_idx(heap, a, *n, v)?;
                        }
                        *n += 1;
                        if *n >= 9007199254740991 {
                            return Err(NativeError::RangeError(
                                "maximum array length exceeded".into(),
                            ));
                        }
                    }
                    Ok(())
                }
                v => {
                    species_set_idx(heap, a, *n, v)?;
                    *n += 1;
                    Ok(())
                }
            }
        };
        push_value(heap, Value::Object(obj), &mut n)?;
        for v in args {
            push_value(heap, *v, &mut n)?;
        }
        // Spec step 6 (`Set(A, "length", n)`): `[[Set]]`, not a define —
        // a `length` setter fires on hostile results.
        species_set_length(heap, a, n, "Array.prototype.concat")?;
        Ok(())
    })();
    heap.release_temp(guard);
    result?;
    Ok(Value::Object(a))
}

fn array_join(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    // Typed-array exotic: indexed elements live in the backing store and
    // `length` is a prototype accessor — neither is visible to the
    // own-property reads below (the same reason `array_slice` routes
    // views through `typed_index_get`). Delegate to the typed-array join,
    // which is also what `%TypedArray%.prototype.toString` resolves to.
    if let Value::Object(o) = this
        && crate::builtin_reflect_proxy::is_typed_array_view(heap, o)
    {
        return crate::builtin_reflect_proxy::typed_array_join(heap, this, args);
    }
    let (obj, len) = array_receiver(heap, this, "join")?;
    let sep = match get_arg(args, 0) {
        Value::Undefined => heap.alloc_str(",")?,
        v => to_string_js_str(heap, v)?,
    };
    // Each element's `to_string_js_str` can invoke user code (`toString`)
    // that allocates and triggers a collection; the separator handle must
    // survive the whole loop, so it is rooted until the last use.
    let sep_base = heap.protect(Value::Str(sep));
    // `out` is WTF-8: joined elements may contain lone surrogates.
    let mut out = Wtf8Buf::new();
    for k in 0..len {
        if k > 0 {
            out.push_wtf8(sep.as_wtf8());
        }
        match get_own(heap, obj, Wtf8::from_str(&k.to_string())) {
            Some(Value::Undefined) | Some(Value::Null) | None => {}
            Some(v) => {
                let s = to_string_js_str(heap, v)?;
                // No heap allocation between the conversion and the copy
                // (`push_wtf8` grows a Rust `Vec`), so the unrooted handle
                // is safe to borrow here with no per-element owned copy.
                out.push_wtf8(s.as_wtf8());
            }
        }
    }
    heap.release_temp(sep_base);
    let s = heap.alloc_wtf8_owned(out)?;
    Ok(Value::Str(s))
}

fn array_index_of(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, len) = array_receiver(heap, this, "indexOf")?;
    let search = get_arg(args, 0);
    let mut k = if args.len() >= 2 {
        let n = to_integer_or_infinity(get_arg(args, 1))?;
        if n < 0.0 {
            len.saturating_sub((-n) as u64)
        } else {
            n as u64
        }
    } else {
        0
    };
    while k < len {
        if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string()))
            && strict_equal(v, search)
        {
            return Ok(num(k as f64));
        }
        k += 1;
    }
    Ok(Value::Int(-1))
}

fn array_last_index_of(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, len) = array_receiver(heap, this, "lastIndexOf")?;
    let search = get_arg(args, 0);
    let mut k: i64 = if args.len() >= 2 {
        let n = to_integer_or_infinity(get_arg(args, 1))?;
        if n < 0.0 {
            len as i64 + n as i64
        } else {
            (n as u64).min(len.saturating_sub(1)) as i64
        }
    } else {
        len as i64 - 1
    };
    while k >= 0 {
        if let Some(v) = get_own(heap, obj, Wtf8::from_str(&(k as u64).to_string()))
            && strict_equal(v, search)
        {
            return Ok(num(k as f64));
        }
        k -= 1;
    }
    Ok(Value::Int(-1))
}

fn array_includes(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, len) = array_receiver(heap, this, "includes")?;
    let search = get_arg(args, 0);
    let mut k = if args.len() >= 2 {
        let n = to_integer_or_infinity(get_arg(args, 1))?;
        if n < 0.0 {
            len.saturating_sub((-n) as u64)
        } else {
            n as u64
        }
    } else {
        0
    };
    while k < len {
        // Holes read as undefined, so includes() finds holes.
        let v = get_own(heap, obj, Wtf8::from_str(&k.to_string())).unwrap_or(Value::Undefined);
        if same_value_zero(v, search) {
            return Ok(Value::Bool(true));
        }
        k += 1;
    }
    Ok(Value::Bool(false))
}

fn array_at(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let (obj, len) = array_receiver(heap, this, "at")?;
    let n = to_integer_or_infinity(get_arg(args, 0))?;
    let k: i64 = if n < 0.0 {
        len as i64 + n as i64
    } else {
        n as i64
    };
    if k < 0 || k as u64 >= len {
        return Ok(Value::Undefined);
    }
    Ok(get_own(heap, obj, Wtf8::from_str(&(k as u64).to_string())).unwrap_or(Value::Undefined))
}

fn array_map(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let callback = expect_callback(args, "Array.prototype.map")?;
    let (obj, len) = array_receiver(heap, this, "map")?;
    let this_arg = get_arg(args, 1);
    let arr = Value::Object(obj);
    if len > 4294967295 {
        return Err(NativeError::RangeError(
            "Array.prototype.map: result length exceeds 2^32 - 1".into(),
        ));
    }
    // `ArraySpeciesCreate(O, len)` runs before the loop (spec order);
    // the result is temp-rooted across the callbacks, which run user
    // code and may collect.
    let (a, _) = array_species_create(heap, obj, len, "Array.prototype.map")?;
    let guard = heap.protect(Value::Object(a));
    let result = (|| -> Result<(), NativeError> {
        for k in 0..len {
            // Holes stay holes; the callback never sees them.
            if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string())) {
                let mapped = call_function(heap, callback, this_arg, &[v, num(k as f64), arr])?;
                species_set_idx(heap, a, k, mapped)?;
            }
        }
        Ok(())
    })();
    heap.release_temp(guard);
    result?;
    // No trailing length write: spec never sets `length` after the loop
    // for `map`. The default species path constructed the result with
    // `length` already `len`; a hostile Array-kind result extended it
    // through the exotic writes; a hostile non-Array result gets no
    // `length` at all.
    Ok(Value::Object(a))
}

fn array_filter(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let callback = expect_callback(args, "Array.prototype.filter")?;
    let (obj, len) = array_receiver(heap, this, "filter")?;
    let this_arg = get_arg(args, 1);
    let arr = Value::Object(obj);
    let (a, _) = array_species_create(heap, obj, 0, "Array.prototype.filter")?;
    let guard = heap.protect(Value::Object(a));
    let result = (|| -> Result<(), NativeError> {
        let mut to = 0u64;
        for k in 0..len {
            // Holes are skipped entirely (never passed to the callback).
            if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string())) {
                let keep = call_function(heap, callback, this_arg, &[v, num(k as f64), arr])?;
                if keep.is_truthy() {
                    species_set_idx(heap, a, to, v)?;
                    to += 1;
                }
            }
        }
        Ok(())
    })();
    heap.release_temp(guard);
    result?;
    Ok(Value::Object(a))
}

fn array_for_each(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let callback = expect_callback(args, "Array.prototype.forEach")?;
    let (obj, len) = array_receiver(heap, this, "forEach")?;
    let this_arg = get_arg(args, 1);
    let arr = Value::Object(obj);
    for k in 0..len {
        if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string())) {
            call_function(heap, callback, this_arg, &[v, num(k as f64), arr])?;
        }
    }
    Ok(Value::Undefined)
}

fn array_reduce(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let callback = expect_callback(args, "Array.prototype.reduce")?;
    let (obj, len) = array_receiver(heap, this, "reduce")?;
    let this_arg = Value::Undefined;
    let arr = Value::Object(obj);
    let mut k = 0u64;
    // Without an initial value the first present element seeds the
    // accumulator; an empty (or all-hole) array throws.
    let mut acc = if args.len() < 2 {
        loop {
            if k >= len {
                return Err(NativeError::TypeError(
                    "Array.prototype.reduce of empty array with no initial value".into(),
                ));
            }
            if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string())) {
                k += 1;
                break v;
            }
            k += 1;
        }
    } else {
        get_arg(args, 1)
    };
    while k < len {
        if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string())) {
            acc = call_function(heap, callback, this_arg, &[acc, v, num(k as f64), arr])?;
        }
        k += 1;
    }
    Ok(acc)
}

fn array_reduce_right(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let callback = expect_callback(args, "Array.prototype.reduceRight")?;
    let (obj, len) = array_receiver(heap, this, "reduceRight")?;
    let this_arg = Value::Undefined;
    let arr = Value::Object(obj);
    let mut k = len;
    let mut acc = if args.len() < 2 {
        loop {
            if k == 0 {
                return Err(NativeError::TypeError(
                    "Array.prototype.reduceRight of empty array with no initial value".into(),
                ));
            }
            k -= 1;
            if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string())) {
                break v;
            }
        }
    } else {
        get_arg(args, 1)
    };
    while k > 0 {
        k -= 1;
        if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string())) {
            acc = call_function(heap, callback, this_arg, &[acc, v, num(k as f64), arr])?;
        }
    }
    Ok(acc)
}

fn array_find(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype.find";
    let callback = expect_callback(args, NAME)?;
    // Spec 23.1.3.8 step 3: the callability check is unconditional — it
    // throws even when `len` is 0 or every index is a hole.
    if !is_callable(heap, callback) {
        return Err(NativeError::TypeError(format!(
            "{NAME}: callback is not callable"
        )));
    }
    let (obj, len) = array_receiver(heap, this, "find")?;
    let this_arg = get_arg(args, 1);
    let arr = Value::Object(obj);
    // Spec 23.1.3.8 step 5: no `HasProperty` guard — the predicate runs
    // for every index, holes included (`Get` yields `undefined` there and
    // fires prototype-chain getters, via `spec_get`).
    for k in 0..len {
        let v = spec_get(heap, obj, Wtf8::from_str(&k.to_string()))?;
        if call_function(heap, callback, this_arg, &[v, num(k as f64), arr])?.is_truthy() {
            return Ok(v);
        }
    }
    Ok(Value::Undefined)
}

fn array_find_index(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype.findIndex";
    let callback = expect_callback(args, NAME)?;
    // Spec 23.1.3.9 step 3: the callability check is unconditional — it
    // throws even when `len` is 0 or every index is a hole.
    if !is_callable(heap, callback) {
        return Err(NativeError::TypeError(format!(
            "{NAME}: callback is not callable"
        )));
    }
    let (obj, len) = array_receiver(heap, this, "findIndex")?;
    let this_arg = get_arg(args, 1);
    let arr = Value::Object(obj);
    // Spec 23.1.3.9 step 5: no `HasProperty` guard — the predicate runs
    // for every index, holes included (`Get` yields `undefined` there and
    // fires prototype-chain getters, via `spec_get`).
    for k in 0..len {
        let v = spec_get(heap, obj, Wtf8::from_str(&k.to_string()))?;
        if call_function(heap, callback, this_arg, &[v, num(k as f64), arr])?.is_truthy() {
            return Ok(num(k as f64));
        }
    }
    Ok(num(-1.0))
}

fn array_some(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let callback = expect_callback(args, "Array.prototype.some")?;
    let (obj, len) = array_receiver(heap, this, "some")?;
    let this_arg = get_arg(args, 1);
    let arr = Value::Object(obj);
    for k in 0..len {
        if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string()))
            && call_function(heap, callback, this_arg, &[v, num(k as f64), arr])?.is_truthy()
        {
            return Ok(Value::Bool(true));
        }
    }
    Ok(Value::Bool(false))
}

fn array_every(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let callback = expect_callback(args, "Array.prototype.every")?;
    let (obj, len) = array_receiver(heap, this, "every")?;
    let this_arg = get_arg(args, 1);
    let arr = Value::Object(obj);
    for k in 0..len {
        if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string()))
            && !call_function(heap, callback, this_arg, &[v, num(k as f64), arr])?.is_truthy()
        {
            return Ok(Value::Bool(false));
        }
    }
    Ok(Value::Bool(true))
}

// ---------------------------------------------------------------------------
// sort / toSorted / reverse / toReversed / fill / copyWithin
// ---------------------------------------------------------------------------

/// Shared ordering core for `sort` and `toSorted`: snapshot the present
/// elements (holes excluded) and return them ordered. A provided comparator
/// is called as `cmp(a, b)` with its `ToNumber` result deciding the order
/// (`NaN` keeps the original order); without one, `undefined` sorts after
/// every defined value and the rest order by the UTF-16 code units of
/// `ToString(v)`. Rust's `sort_by` is stable, matching the spec.
///
/// Element values are temp-rooted for the whole ordering pass: comparators
/// and `toString` hooks run user code that may mutate (or detach) the very
/// array being sorted.
fn sort_present(
    heap: &mut Heap,
    obj: Gc<Object>,
    len: u64,
    comparefn: Value,
    name: &str,
) -> Result<Vec<Value>, NativeError> {
    use std::cmp::Ordering;
    let has_cmp = !matches!(comparefn, Value::Undefined);
    if has_cmp && !is_callable(heap, comparefn) {
        return Err(NativeError::TypeError(format!(
            "{name}: comparator is not a function"
        )));
    }
    let mut items: Vec<Value> = Vec::new();
    for k in 0..len {
        if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string())) {
            items.push(v);
        }
    }
    // Root every element; the dummy `Undefined` gives `release_temp` a base
    // even when the array is empty.
    let base = heap.protect(Value::Undefined);
    for &v in &items {
        heap.protect(v);
    }
    let mut err: Option<NativeError> = None;
    if has_cmp {
        items.sort_by(|&a, &b| {
            if err.is_some() {
                return Ordering::Equal;
            }
            let ord = (|| -> Result<Ordering, NativeError> {
                let r = call_function(heap, comparefn, Value::Undefined, &[a, b])?;
                let n = to_number(r)?;
                Ok(if n.is_nan() || n == 0.0 {
                    Ordering::Equal
                } else if n < 0.0 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                })
            })();
            match ord {
                Ok(o) => o,
                Err(e) => {
                    err = Some(e);
                    Ordering::Equal
                }
            }
        });
    } else {
        // Decorate with the UTF-16 code units of `ToString(v)`; `undefined`
        // sorts after every defined value and never reaches the comparator.
        let mut decorated: Vec<(bool, Vec<u16>, Value)> = Vec::with_capacity(items.len());
        for &v in &items {
            if matches!(v, Value::Undefined) {
                decorated.push((true, Vec::new(), v));
                continue;
            }
            match to_string_js(heap, v) {
                Ok(s) => decorated.push((false, s.to_units(), v)),
                Err(e) => {
                    err = Some(e);
                    break;
                }
            }
        }
        if err.is_none() {
            decorated.sort_by(|a, b| match (a.0, b.0) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => a.1.cmp(&b.1),
            });
            items = decorated.into_iter().map(|(_, _, v)| v).collect();
        }
    }
    heap.release_temp(base);
    if let Some(e) = err {
        return Err(e);
    }
    Ok(items)
}

fn array_sort(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype.sort";
    let (obj, len) = array_receiver(heap, this, NAME)?;
    let items = sort_present(heap, obj, len, get_arg(args, 0), NAME)?;
    // Write the ordered values densely at the front; the vacated tail
    // indices become holes.
    for (i, v) in items.iter().enumerate() {
        set_prop(heap, obj, Wtf8::from_str(&(i as u64).to_string()), *v)?;
    }
    for i in items.len() as u64..len {
        delete_prop(heap, obj, Wtf8::from_str(&i.to_string()));
    }
    Ok(this)
}

fn array_to_sorted(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype.toSorted";
    let (obj, len) = array_receiver(heap, this, NAME)?;
    // Spec `ArrayCreate(len)`: lengths above 2^32-1 throw RangeError
    // before anything is read — and without the guard the `resize`
    // below is a multi-gigabyte allocation attempt.
    if len > u32::MAX as u64 {
        return Err(NativeError::RangeError(format!(
            "{NAME}: invalid array length {len}"
        )));
    }
    let items = sort_present(heap, obj, len, get_arg(args, 0), NAME)?;
    // Same length as the source; holes stay holes at the tail. The
    // length property carries the size — no need to materialize `len`
    // hole slots in the transient vec.
    // Per ECMA-262 this method uses plain `ArrayCreate(len)`, not
    // `ArraySpeciesCreate`: a `class B extends Array` input yields a
    // plain `Array` result.
    let mut elems: Vec<Option<Value>> = items.into_iter().map(Some).collect();
    let arr = new_array(heap)?;
    for (i, e) in elems.drain(..).enumerate() {
        if let Some(v) = e {
            set_idx_raw(heap, arr, i as u64, v)?;
        }
    }
    set_len(heap, arr, len)?;
    Ok(Value::Object(arr))
}

fn array_reverse(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    // Spec `Array.prototype.reverse`: the in-place pair-swap loop. Never
    // materialize all `len` elements: `len` can be up to 2^53-1 (e.g. an
    // array-like with `length: 2**53+2`, clamped by ToLength), so a
    // `Vec::with_capacity(len)` is a multi-exabyte allocation attempt.
    // Only existing properties are touched, via HasProperty/Get like the
    // spec (getters fire in index order, low side first per iteration).
    let (obj, len) = array_receiver(heap, this, "reverse")?;
    let middle = len / 2;
    let mut lower: u64 = 0;
    while lower != middle {
        let upper = len - lower - 1;
        let lower_p = lower.to_string();
        let upper_p = upper.to_string();
        let lower_exists = has_prop(heap, obj, Wtf8::from_str(&lower_p));
        let lower_value = if lower_exists {
            Some(spec_get(heap, obj, Wtf8::from_str(&lower_p))?)
        } else {
            None
        };
        let upper_exists = has_prop(heap, obj, Wtf8::from_str(&upper_p));
        let upper_value = if upper_exists {
            Some(spec_get(heap, obj, Wtf8::from_str(&upper_p))?)
        } else {
            None
        };
        match (lower_value, upper_value) {
            (Some(lv), Some(uv)) => {
                set_prop(heap, obj, Wtf8::from_str(&lower_p), uv)?;
                set_prop(heap, obj, Wtf8::from_str(&upper_p), lv)?;
            }
            (None, Some(uv)) => {
                set_prop(heap, obj, Wtf8::from_str(&lower_p), uv)?;
                delete_prop(heap, obj, Wtf8::from_str(&upper_p));
            }
            (Some(lv), None) => {
                delete_prop(heap, obj, Wtf8::from_str(&lower_p));
                set_prop(heap, obj, Wtf8::from_str(&upper_p), lv)?;
            }
            (None, None) => {}
        }
        lower += 1;
    }
    Ok(this)
}

fn array_to_reversed(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype.toReversed";
    let (obj, len) = array_receiver(heap, this, NAME)?;
    // Spec `ArrayCreate(len)`: lengths above 2^32-1 throw RangeError
    // before anything is read — and without the guard the
    // `with_capacity(len)` below is a multi-exabyte allocation attempt.
    if len > u32::MAX as u64 {
        return Err(NativeError::RangeError(format!(
            "{NAME}: invalid array length {len}"
        )));
    }
    // Per ECMA-262 this method uses plain `ArrayCreate(len)`, not
    // `ArraySpeciesCreate`: a `class B extends Array` input yields a
    // plain `Array` result.
    let len_usize = len as usize;
    let mut elems: Vec<Option<Value>> = Vec::with_capacity(len_usize);
    for k in (0..len).rev() {
        elems.push(get_own(heap, obj, Wtf8::from_str(&k.to_string())));
    }
    Ok(Value::Object(new_array_from_opt(heap, &elems)?))
}

fn array_fill(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype.fill";
    let (obj, len) = array_receiver(heap, this, NAME)?;
    let value = get_arg(args, 0);
    let start = if args.len() > 1 {
        clamp_index(args[1], len)?
    } else {
        0
    };
    let end = if args.len() > 2 {
        clamp_index(args[2], len)?
    } else {
        len
    };
    for k in start..end {
        set_prop(heap, obj, Wtf8::from_str(&k.to_string()), value)?;
    }
    Ok(this)
}

fn array_copy_within(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype.copyWithin";
    let (obj, len) = array_receiver(heap, this, NAME)?;
    let target = clamp_index(get_arg(args, 0), len)?;
    let start = if args.len() > 1 {
        clamp_index(args[1], len)?
    } else {
        0
    };
    let end = if args.len() > 2 {
        clamp_index(args[2], len)?
    } else {
        len
    };
    // Snapshot before writing: source and target ranges may overlap.
    let count = end.saturating_sub(start).min(len.saturating_sub(target));
    let mut tmp: Vec<Option<Value>> = Vec::with_capacity(count as usize);
    for k in start..start + count {
        tmp.push(get_own(heap, obj, Wtf8::from_str(&k.to_string())));
    }
    for (j, slot) in tmp.iter().enumerate() {
        let key = (target + j as u64).to_string();
        match slot {
            Some(v) => set_prop(heap, obj, Wtf8::from_str(&key), *v)?,
            None => {
                delete_prop(heap, obj, Wtf8::from_str(&key));
            }
        }
    }
    Ok(this)
}

// ---------------------------------------------------------------------------
// flat / flatMap / findLast / findLastIndex
// ---------------------------------------------------------------------------

/// Recursive worker for `flat`: appends the present elements of the
/// `len`-long object `obj` to `out`, descending into array elements while
/// `depth >= 1`. Holes are dropped, matching the spec.
///
/// `out` must be a dense array ([`Heap::new_dense_array`]); elements are
/// appended with [`Heap::dense_push`] (amortized O(1), no per-element
/// property interning or linear scan), and the caller finishes with
/// [`Heap::materialize_dense`] before exposing the result.
fn flatten_into(
    heap: &mut Heap,
    obj: Gc<Object>,
    len: u64,
    depth: f64,
    out: Gc<Object>,
) -> Result<(), NativeError> {
    // Explicit stack instead of recursion: `flat(Infinity)` on a
    // pathologically nested array would otherwise nest one native frame
    // per level and overflow the thread's stack (an uncatchable process
    // abort, not a catchable error). Iterative, like the GC's `drain`.
    struct Pending {
        obj: Gc<Object>,
        len: u64,
        next: u64,
        depth: f64,
    }
    let mut stack = vec![Pending {
        obj,
        len,
        next: 0,
        depth,
    }];
    while let Some(top) = stack.last_mut() {
        if top.next >= top.len {
            stack.pop();
            continue;
        }
        let k = top.next;
        top.next += 1;
        let (obj, depth) = (top.obj, top.depth);
        if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string())) {
            let sub = match v {
                Value::Object(o) if heap.get(o).kind == ObjectKind::Array => Some(o),
                _ => None,
            };
            if let Some(o) = sub.filter(|_| depth >= 1.0) {
                let inner_len = get_length(heap, o)?;
                stack.push(Pending {
                    obj: o,
                    len: inner_len,
                    next: 0,
                    depth: depth - 1.0,
                });
            } else {
                heap.dense_push(out, v)?;
            }
        }
    }
    Ok(())
}

/// `flatten_into` writing into a species-created result instead of a
/// dense array: indices are created with [`species_set_idx`] (a hostile
/// `@@species` may return a non-array) and counted in `n`, since the
/// result's `length` cannot be trusted to track the exotic.
fn flatten_into_species(
    heap: &mut Heap,
    obj: Gc<Object>,
    len: u64,
    depth: f64,
    out: Gc<Object>,
    n: &mut u64,
) -> Result<(), NativeError> {
    // Explicit stack instead of recursion: `flat(Infinity)` on a
    // pathologically nested array would otherwise nest one native frame
    // per level and overflow the thread's stack (an uncatchable process
    // abort, not a catchable error). Iterative, like the GC's `drain`.
    struct Pending {
        obj: Gc<Object>,
        len: u64,
        next: u64,
        depth: f64,
    }
    let mut stack = vec![Pending {
        obj,
        len,
        next: 0,
        depth,
    }];
    while let Some(top) = stack.last_mut() {
        if top.next >= top.len {
            stack.pop();
            continue;
        }
        let k = top.next;
        top.next += 1;
        let (obj, depth) = (top.obj, top.depth);
        if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string())) {
            let sub = match v {
                Value::Object(o) if heap.get(o).kind == ObjectKind::Array => Some(o),
                _ => None,
            };
            if let Some(o) = sub.filter(|_| depth >= 1.0) {
                let inner_len = get_length(heap, o)?;
                stack.push(Pending {
                    obj: o,
                    len: inner_len,
                    next: 0,
                    depth: depth - 1.0,
                });
            } else {
                species_set_idx(heap, out, *n, v)?;
                *n += 1;
            }
        }
    }
    Ok(())
}

fn array_flat(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype.flat";
    let (obj, len) = array_receiver(heap, this, NAME)?;
    let depth_arg = get_arg(args, 0);
    let depth = if matches!(depth_arg, Value::Undefined) {
        1.0
    } else {
        to_integer_or_infinity(depth_arg)?
    };
    // Species resolution runs before any element is read (spec order).
    // The default species keeps the dense fast path bit-for-bit; a
    // hostile species flattens into the constructed object instead
    // (which is not a dense array, so `dense_push` cannot be used).
    let (target, is_default) = array_species_ctor(heap, obj, NAME)?;
    if is_default {
        let out = heap.new_dense_array()?;
        flatten_into(heap, obj, len, depth, out)?;
        heap.materialize_dense(out)?;
        return Ok(Value::Object(out));
    }
    let a = species_construct(heap, target, 0, NAME)?;
    let guard = heap.protect(Value::Object(a));
    let mut n = 0u64;
    let result = flatten_into_species(heap, obj, len, depth, a, &mut n);
    heap.release_temp(guard);
    result?;
    Ok(Value::Object(a))
}

fn array_flat_map(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype.flatMap";
    let callback = expect_callback(args, NAME)?;
    let (obj, len) = array_receiver(heap, this, NAME)?;
    let this_arg = get_arg(args, 1);
    let arr = Value::Object(obj);
    // Species resolution runs before any element is read (spec order).
    // The default species keeps the dense fast path bit-for-bit; a
    // hostile species collects into the constructed object instead
    // (which is not a dense array, so `dense_push` cannot be used).
    let (target, is_default) = array_species_ctor(heap, obj, NAME)?;
    if is_default {
        let out = heap.new_dense_array()?;
        // Root `out` across the loop: the callback is user code and can
        // allocate (triggering a collection) while the array is otherwise
        // unreachable. Elements already pushed stay alive through the dense
        // store, which the tracer visits.
        let guard = heap.protect(Value::Object(out));
        let result = flat_map_into_dense(heap, obj, len, callback, this_arg, arr, out);
        heap.release_temp(guard);
        result?;
        heap.materialize_dense(out)?;
        return Ok(Value::Object(out));
    }
    let a = species_construct(heap, target, 0, NAME)?;
    // Root `a` across the loop: the callback is user code and can
    // allocate (triggering a collection) while the array is otherwise
    // unreachable. Elements already written stay alive through the
    // array's properties, which the tracer visits.
    let guard = heap.protect(Value::Object(a));
    let result = (|| -> Result<(), NativeError> {
        let mut n = 0u64;
        for k in 0..len {
            // Holes are skipped entirely (never passed to the callback).
            if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string())) {
                let mapped = call_function(heap, callback, this_arg, &[v, num(k as f64), arr])?;
                match mapped {
                    Value::Object(o) if heap.get(o).kind == ObjectKind::Array => {
                        // One level of flattening; holes in the mapped array
                        // are dropped.
                        let inner_len = get_length(heap, o)?;
                        for j in 0..inner_len {
                            if let Some(iv) = get_own(heap, o, Wtf8::from_str(&j.to_string())) {
                                species_set_idx(heap, a, n, iv)?;
                                n += 1;
                            }
                        }
                    }
                    _ => {
                        species_set_idx(heap, a, n, mapped)?;
                        n += 1;
                    }
                }
            }
        }
        Ok(())
    })();
    heap.release_temp(guard);
    result?;
    Ok(Value::Object(a))
}

/// The `flatMap` callback loop writing into a dense builder array
/// (extracted so the default-species fast path keeps its exact shape).
fn flat_map_into_dense(
    heap: &mut Heap,
    obj: Gc<Object>,
    len: u64,
    callback: Value,
    this_arg: Value,
    arr: Value,
    out: Gc<Object>,
) -> Result<(), NativeError> {
    for k in 0..len {
        // Holes are skipped entirely (never passed to the callback).
        if let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string())) {
            let mapped = call_function(heap, callback, this_arg, &[v, num(k as f64), arr])?;
            match mapped {
                Value::Object(o) if heap.get(o).kind == ObjectKind::Array => {
                    // One level of flattening; holes in the mapped array
                    // are dropped.
                    let inner_len = get_length(heap, o)?;
                    for j in 0..inner_len {
                        if let Some(iv) = get_own(heap, o, Wtf8::from_str(&j.to_string())) {
                            heap.dense_push(out, iv)?;
                        }
                    }
                }
                _ => {
                    heap.dense_push(out, mapped)?;
                }
            }
        }
    }
    Ok(())
}

fn array_find_last(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype.findLast";
    let callback = expect_callback(args, NAME)?;
    // Spec 23.1.3.10 step 3: the callability check is unconditional — it
    // throws even when `len` is 0 or every index is a hole.
    if !is_callable(heap, callback) {
        return Err(NativeError::TypeError(format!(
            "{NAME}: callback is not callable"
        )));
    }
    let (obj, len) = array_receiver(heap, this, "findLast")?;
    let this_arg = get_arg(args, 1);
    let arr = Value::Object(obj);
    // Spec 23.1.3.10 step 5: no `HasProperty` guard — the predicate runs
    // for every index from the end, holes included (`Get` yields
    // `undefined` there and fires prototype-chain getters, via
    // `spec_get`).
    for k in (0..len).rev() {
        let v = spec_get(heap, obj, Wtf8::from_str(&k.to_string()))?;
        if call_function(heap, callback, this_arg, &[v, num(k as f64), arr])?.is_truthy() {
            return Ok(v);
        }
    }
    Ok(Value::Undefined)
}

fn array_find_last_index(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype.findLastIndex";
    let callback = expect_callback(args, NAME)?;
    // Spec 23.1.3.11 step 3: the callability check is unconditional — it
    // throws even when `len` is 0 or every index is a hole.
    if !is_callable(heap, callback) {
        return Err(NativeError::TypeError(format!(
            "{NAME}: callback is not callable"
        )));
    }
    let (obj, len) = array_receiver(heap, this, "findLastIndex")?;
    let this_arg = get_arg(args, 1);
    let arr = Value::Object(obj);
    // Spec 23.1.3.11 step 5: no `HasProperty` guard — the predicate runs
    // for every index from the end, holes included (`Get` yields
    // `undefined` there and fires prototype-chain getters, via
    // `spec_get`).
    for k in (0..len).rev() {
        let v = spec_get(heap, obj, Wtf8::from_str(&k.to_string()))?;
        if call_function(heap, callback, this_arg, &[v, num(k as f64), arr])?.is_truthy() {
            return Ok(num(k as f64));
        }
    }
    Ok(num(-1.0))
}

// ---------------------------------------------------------------------------
// keys / values / entries (v1: fresh arrays, no iterator objects)
// ---------------------------------------------------------------------------

fn array_keys(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype.keys";
    let (obj, len) = array_receiver(heap, this, NAME)?;
    // v1 returns a fresh array (spec: lazy iterator); the result array
    // goes through ArrayCreate semantics, so lengths above 2^32-1 throw
    // RangeError — and without the guard the collect below attempts a
    // multi-exabyte allocation.
    if len > u32::MAX as u64 {
        return Err(NativeError::RangeError(format!(
            "{NAME}: invalid array length {len}"
        )));
    }
    if len <= FRESH_ARRAY_DENSE_LIMIT {
        // `keys` visits every index in `0..len`, holes included. The
        // bound above makes the `collect` pre-size safe: at most 2^20
        // entries, never a hostile-length allocation.
        let elems: Vec<Option<Value>> = (0..len).map(|k| Some(num(k as f64))).collect();
        return Ok(Value::Object(new_array_from_opt(heap, &elems)?));
    }
    // Sparse hostile-length path: `length` is set, and the source's own
    // present indices materialize as themselves; everything else stays
    // a hole. Documented deviation: spec creates every index present.
    // The loop is over the source's own properties, never `0..len`.
    let arr = new_array_with_length(heap, len)?;
    let guard = heap.protect(Value::Object(arr));
    let result = (|| -> Result<(), NativeError> {
        for key in own_all_keys(heap, obj) {
            // Re-derive the index before any allocation: the `&str`
            // borrow of the heap string must not be held across
            // `set_idx_raw`, which can trigger a collection.
            let k = match array_index_key(key.as_wtf8()) {
                Some(k) => k as u64,
                None => continue,
            };
            if k < len {
                set_idx_raw(heap, arr, k, num(k as f64))?;
            }
        }
        Ok(())
    })();
    heap.release_temp(guard);
    result?;
    Ok(Value::Object(arr))
}

/// Hidden slots for the `Array.prototype[@@iterator]` iterator object.
const ARRAY_ITER_ARRAY: &str = "\u{0}bowser:array-iter-array";
const ARRAY_ITER_INDEX: &str = "\u{0}bowser:array-iter-index";

/// `Array.prototype [ @@iterator ]` (ECMA-262 22.1.4.1): returns a real
/// iterator object over the array's elements. (This engine's `values`
/// is a v1 array-returning deviation, so `@@iterator` gets its own
/// iterator rather than aliasing `values`.)
///
/// `pub(crate)`: installed from `builtin_reflect_proxy::install`, which
/// runs after `builtin_collections::install` creates the canonical
/// well-known symbols — installing here in `builtin_object_array`
/// would allocate a doomed symbol that gets overwritten.
pub(crate) fn array_iterator(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype[Symbol.iterator]";
    let (obj, _len) = array_receiver(heap, this, NAME)?;
    let iter = heap
        .alloc_object(Object::new(ObjectKind::Ordinary))
        .map_err(|_| NativeError::OutOfMemory)?;
    heap.set_prototype(iter, Value::Null);
    let iter_val = Value::Object(iter);
    let base = heap.protect(iter_val);
    let out = (|| -> Result<Value, NativeError> {
        define_hidden(heap, iter, ARRAY_ITER_ARRAY, Value::Object(obj))?;
        define_hidden(heap, iter, ARRAY_ITER_INDEX, Value::Int(0))?;
        let next_fn = new_native_function(heap, "next", array_iterator_next, 0)?;
        heap.define_property_str(
            iter,
            "next",
            Property {
                value: Value::Object(next_fn),
                writable: true,
                enumerable: false,
                configurable: true,
            },
        )
        .map_err(|_| NativeError::OutOfMemory)?;
        // Iterators are iterable: `for (x of arr[Symbol.iterator]())`.
        let self_fn = new_native_function(heap, "[Symbol.iterator]", array_iterator_self, 0)?;
        let iterator_sym = heap
            .well_known_symbol(WellKnown::Iterator)
            .map_err(|_| NativeError::OutOfMemory)?;
        heap.define_property_key(
            iter,
            PropKey::Symbol(iterator_sym),
            Property {
                value: Value::Object(self_fn),
                writable: true,
                enumerable: false,
                configurable: true,
            },
        )
        .map_err(|_| NativeError::OutOfMemory)?;
        Ok(iter_val)
    })();
    heap.release_temp(base);
    out
}

/// The `next` method of the array iterator: `{value, done}` per iteration.
fn array_iterator_next(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let incompatible =
        || NativeError::TypeError("array iterator.next called on incompatible receiver".into());
    let Value::Object(iter) = this else {
        return Err(incompatible());
    };
    let arr = match heap
        .get_property(iter, Wtf8::from_str(ARRAY_ITER_ARRAY))
        .map(|p| p.value)
    {
        Some(Value::Object(o)) => o,
        _ => return Err(incompatible()),
    };
    let index = match heap
        .get_property(iter, Wtf8::from_str(ARRAY_ITER_INDEX))
        .map(|p| p.value)
    {
        Some(Value::Int(n)) => n.max(0) as u64,
        Some(Value::Float(f)) => f.max(0.0) as u64,
        _ => return Err(incompatible()),
    };
    let len = get_length(heap, arr)?;
    if index >= len {
        return iter_result_object(heap, Value::Undefined, true);
    }
    let value = get_own(heap, arr, Wtf8::from_str(&index.to_string())).unwrap_or(Value::Undefined);
    heap.define_property_str(
        iter,
        ARRAY_ITER_INDEX,
        Property::data(u64_to_value(index + 1)),
    )
    .map_err(|_| NativeError::OutOfMemory)?;
    iter_result_object(heap, value, false)
}

/// `[Symbol.iterator]` on the array iterator itself: returns `this`.
fn array_iterator_self(
    _heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    Ok(this)
}

/// Build a `{value, done}` result object (writable/enumerable/configurable,
/// per spec `CreateIterResultObject`).
fn iter_result_object(heap: &mut Heap, value: Value, done: bool) -> Result<Value, NativeError> {
    let obj = heap
        .alloc_object(Object::new(ObjectKind::Ordinary))
        .map_err(|_| NativeError::OutOfMemory)?;
    heap.set_prototype(obj, Value::Null);
    let g = heap.protect(Value::Object(obj));
    let r = (|| -> Result<Value, NativeError> {
        heap.define_property_str(
            obj,
            "value",
            Property {
                value,
                writable: true,
                enumerable: true,
                configurable: true,
            },
        )
        .map_err(|_| NativeError::OutOfMemory)?;
        heap.define_property_str(
            obj,
            "done",
            Property {
                value: Value::Bool(done),
                writable: true,
                enumerable: true,
                configurable: true,
            },
        )
        .map_err(|_| NativeError::OutOfMemory)?;
        Ok(Value::Object(obj))
    })();
    heap.release_temp(g);
    r
}

fn array_values(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype.values";
    let (obj, len) = array_receiver(heap, this, NAME)?;
    // See `array_keys`: fresh-array v1 with ArrayCreate length semantics.
    if len > u32::MAX as u64 {
        return Err(NativeError::RangeError(format!(
            "{NAME}: invalid array length {len}"
        )));
    }
    if len <= FRESH_ARRAY_DENSE_LIMIT {
        let mut elems: Vec<Option<Value>> = Vec::with_capacity(len as usize);
        for k in 0..len {
            elems.push(get_own(heap, obj, Wtf8::from_str(&k.to_string())));
        }
        return Ok(Value::Object(new_array_from_opt(heap, &elems)?));
    }
    // Sparse hostile-length path: `length` is set, and only the source's
    // own index properties are copied (holes stay holes, exactly as the
    // dense path's `get_own` would leave them). The loop is over the
    // source's own properties, never `0..len`, so no pre-sized or
    // amortized multi-gigabyte allocation is possible. Known deviation:
    // prototype-chain getters do not fire.
    let arr = new_array_with_length(heap, len)?;
    let guard = heap.protect(Value::Object(arr));
    let result = (|| -> Result<(), NativeError> {
        for key in own_all_keys(heap, obj) {
            let k = match array_index_key(key.as_wtf8()) {
                Some(k) => k as u64,
                None => continue,
            };
            if k < len
                && let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string()))
            {
                set_idx_raw(heap, arr, k, v)?;
            }
        }
        Ok(())
    })();
    heap.release_temp(guard);
    result?;
    Ok(Value::Object(arr))
}

fn array_entries(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype.entries";
    let (obj, len) = array_receiver(heap, this, NAME)?;
    // See `array_keys`: fresh-array v1 with ArrayCreate length semantics.
    if len > u32::MAX as u64 {
        return Err(NativeError::RangeError(format!(
            "{NAME}: invalid array length {len}"
        )));
    }
    if len <= FRESH_ARRAY_DENSE_LIMIT {
        let mut elems: Vec<Option<Value>> = Vec::with_capacity(len as usize);
        for k in 0..len {
            // Holes surface as `undefined` values, matching iterator semantics.
            let v = get_own(heap, obj, Wtf8::from_str(&k.to_string())).unwrap_or(Value::Undefined);
            let pair = new_array(heap)?;
            set_idx(heap, pair, 0, num(k as f64))?;
            set_idx(heap, pair, 1, v)?;
            elems.push(Some(Value::Object(pair)));
        }
        return Ok(Value::Object(new_array_from_opt(heap, &elems)?));
    }
    // Sparse hostile-length path: `length` is set, and only the source's
    // own index properties become `[k, v]` pairs (holes stay holes, as
    // the dense path leaves them). The loop is over the source's own
    // properties, never `0..len`. Known deviation: prototype-chain
    // getters do not fire.
    let arr = new_array_with_length(heap, len)?;
    let guard = heap.protect(Value::Object(arr));
    let result = (|| -> Result<(), NativeError> {
        for key in own_all_keys(heap, obj) {
            let k = match array_index_key(key.as_wtf8()) {
                Some(k) => k as u64,
                None => continue,
            };
            if k < len
                && let Some(v) = get_own(heap, obj, Wtf8::from_str(&k.to_string()))
            {
                let pair = new_array(heap)?;
                // `pair` is unreachable until stored below; the `set_idx`
                // calls can allocate and collect, so hold it.
                let pguard = heap.protect(Value::Object(pair));
                let built = (|| -> Result<(), NativeError> {
                    set_idx(heap, pair, 0, num(k as f64))?;
                    set_idx(heap, pair, 1, v)?;
                    Ok(())
                })();
                heap.release_temp(pguard);
                built?;
                set_idx_raw(heap, arr, k, Value::Object(pair))?;
            }
        }
        Ok(())
    })();
    heap.release_temp(guard);
    result?;
    Ok(Value::Object(arr))
}

// ---------------------------------------------------------------------------
// with / toSpliced
// ---------------------------------------------------------------------------

fn array_with(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype.with";
    let (obj, len) = array_receiver(heap, this, NAME)?;
    // Spec `ArrayCreate(len)`: lengths above 2^32-1 throw RangeError
    // before anything is read — and without the guard the
    // `with_capacity(len)` below is a multi-exabyte allocation attempt.
    if len > u32::MAX as u64 {
        return Err(NativeError::RangeError(format!(
            "{NAME}: invalid array length {len}"
        )));
    }
    let value = get_arg(args, 1);
    let rel = to_integer_or_infinity(get_arg(args, 0))?;
    let actual = if rel < 0.0 { len as f64 + rel } else { rel };
    if actual < 0.0 || actual >= len as f64 {
        return Err(NativeError::RangeError(format!(
            "{NAME}: index out of range"
        )));
    }
    let at = actual as u64;
    if len <= FRESH_ARRAY_DENSE_LIMIT {
        let mut elems: Vec<Option<Value>> = Vec::with_capacity(len as usize);
        for k in 0..len {
            elems.push(if k == at {
                Some(value)
            } else {
                get_own(heap, obj, Wtf8::from_str(&k.to_string()))
            });
        }
        return Ok(Value::Object(new_array_from_opt(heap, &elems)?));
    }
    // Sparse hostile-length path: `length` is set; the source's own
    // index properties are copied with the substitution applied at
    // `at`, and `at` itself is always created present (spec). The loop
    // is over the source's own properties, never `0..len`. Known
    // deviation: prototype-chain getters do not fire.
    let arr = new_array_with_length(heap, len)?;
    let guard = heap.protect(Value::Object(arr));
    let result = (|| -> Result<(), NativeError> {
        for key in own_all_keys(heap, obj) {
            let k = match array_index_key(key.as_wtf8()) {
                Some(k) => k as u64,
                None => continue,
            };
            if k < len {
                let v = if k == at {
                    value
                } else {
                    match get_own(heap, obj, Wtf8::from_str(&k.to_string())) {
                        Some(v) => v,
                        None => continue,
                    }
                };
                set_idx_raw(heap, arr, k, v)?;
            }
        }
        // `at` may not have been an own property of the source (a hole
        // there, or beyond the source's present keys): spec still
        // creates it present with the substituted value.
        set_idx_raw(heap, arr, at, value)?;
        Ok(())
    })();
    heap.release_temp(guard);
    result?;
    Ok(Value::Object(arr))
}

// ---------------------------------------------------------------------------
// toSpliced
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// toSpliced
// ---------------------------------------------------------------------------

/// Upper bound on a result length for the spec-exact dense build used by
/// the fresh-array builtins (`keys`, `values`, `entries`, `with`,
/// `toSpliced`, and `splice`'s deleted array).
/// The dense path performs one `Get` plus one property creation per
/// index; above this bound the result is built sparse instead (length
/// set, only the source's own index properties materialized), so an
/// untrusted `length` near 2^32 can never turn into a multi-gigabyte
/// allocation that aborts the process in the allocator — neither via a
/// single `Vec::with_capacity(len)` nor via amortized `push` growth.
/// Known deviation on the sparse path: spec creates every index present
/// (holes filled with `undefined`), and prototype-chain getters do not
/// fire.
const FRESH_ARRAY_DENSE_LIMIT: u64 = 1 << 20;

/// Spec 23.1.3.32 `Array.prototype.toSpliced`: the non-mutating twin of
/// `splice`. Copies `O[0..actualStart)`, then the `items` arguments, then
/// `O[actualStart+actualDelete..len)` into a fresh array; the receiver is
/// never written.
///
/// Every copied index goes through `Get` (spec steps 17/19 — not an
/// own-property read, so prototype-chain getters fire, via `spec_get`)
/// and `CreateDataPropertyOrThrow`. There is no `HasProperty` guard, so
/// holes are *filled* with present `undefined`, not preserved:
/// `[1,,3].toSpliced(0, 0)` has `1 in result === true`.
///
/// Note the one asymmetry with `splice`: when `start` is absent the delete
/// count is 0 (a copy), not `len - actualStart` — only the
/// `start`-present/`deleteCount`-absent form deletes to the end.
fn array_to_spliced(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    const NAME: &str = "Array.prototype.toSpliced";
    let (obj, len) = array_receiver(heap, this, "toSpliced")?;
    let actual_start = clamp_index(get_arg(args, 0), len)?;
    let insert_count = args.len().saturating_sub(2) as u64;
    let actual_delete = if args.is_empty() {
        0
    } else if args.len() < 2 {
        len - actual_start
    } else {
        let dc = to_integer_or_infinity(get_arg(args, 1))?;
        if dc <= 0.0 {
            0
        } else if dc.is_infinite() {
            len - actual_start
        } else {
            (dc as u64).min(len - actual_start)
        }
    };
    let new_len = len - actual_delete + insert_count;
    // Spec `ArrayCreate(newLen)`: lengths above 2^32-1 throw RangeError
    // before anything is read.
    if new_len > u32::MAX as u64 {
        return Err(NativeError::RangeError(format!(
            "{NAME}: invalid array length {new_len}"
        )));
    }
    if new_len <= FRESH_ARRAY_DENSE_LIMIT {
        // Dense, spec-exact path: one `Get` + one fresh data property per
        // index, holes included (a hole reads as `undefined` and is
        // created present). Values are written into the result array —
        // which is temp-rooted across the loop because getters run user
        // code and may collect — immediately after each `Get`, never
        // buffered in an unrooted Rust `Vec`. In particular the `Vec`
        // is never pre-sized from the untrusted `new_len`.
        // Per ECMA-262 this method uses plain `ArrayCreate(newLen)`,
        // not `ArraySpeciesCreate`: a `class B extends Array` input
        // yields a plain `Array` result.
        let arr = new_array(heap)?;
        let guard = heap.protect(Value::Object(arr));
        let result = (|| -> Result<(), NativeError> {
            // The result is a fresh `ArrayCreate` array, so every index
            // write is the exotic `[[DefineOwnProperty]]`
            // (`CreateDataPropertyOrThrow`): `set_idx`, not the plain
            // `set_prop` of `set_idx_raw`.
            for k in 0..actual_start {
                let v = spec_get(heap, obj, Wtf8::from_str(&k.to_string()))?;
                set_idx(heap, arr, k, v)?;
            }
            for (j, v) in args.iter().skip(2).enumerate() {
                set_idx(heap, arr, actual_start + j as u64, *v)?;
            }
            let tail_start = actual_start + actual_delete;
            for k in tail_start..len {
                let v = spec_get(heap, obj, Wtf8::from_str(&k.to_string()))?;
                set_idx(heap, arr, actual_start + insert_count + (k - tail_start), v)?;
            }
            Ok(())
        })();
        heap.release_temp(guard);
        result?;
        set_len(heap, arr, new_len)?;
        return Ok(Value::Object(arr));
    }
    // Huge `newLen`: the loop above is O(newLen) allocations and would
    // abort the process in the allocator. Build sparse instead: `length`
    // is set, the inserted items and the source's own copied index
    // properties (own getters fire via `spec_get`) are materialized, and
    // everything else stays a hole. Known deviation on this path: spec
    // creates every index present, and prototype-chain getters do not
    // fire.
    // Plain `ArrayCreate(newLen)` here too (see the dense path).
    let arr = new_array_with_length(heap, new_len)?;
    let guard = heap.protect(Value::Object(arr));
    let result = (|| -> Result<(), NativeError> {
        for (j, v) in args.iter().skip(2).enumerate() {
            set_idx(heap, arr, actual_start + j as u64, *v)?;
        }
        for key in own_all_keys(heap, obj) {
            let Some(k) = array_index_key(key.as_wtf8()) else {
                continue;
            };
            let k = k as u64;
            let dest = if k < actual_start {
                k
            } else if k >= actual_start + actual_delete {
                k - actual_delete + insert_count
            } else {
                continue; // inside the deleted range
            };
            // Re-derive the key from the index: the `&str` borrow of the
            // heap string must not be held across `spec_get`, which runs
            // user code.
            let v = spec_get(heap, obj, Wtf8::from_str(&k.to_string()))?;
            set_idx(heap, arr, dest, v)?;
        }
        Ok(())
    })();
    heap.release_temp(guard);
    result?;
    Ok(Value::Object(arr))
}

// ---------------------------------------------------------------------------
// install
// ---------------------------------------------------------------------------

/// Make `ctor`/`proto` a constructor/prototype pair:
/// `ctor.prototype = proto` ({writable:false, enumerable:false,
/// configurable:false}) and `proto.constructor = ctor`
/// ({writable:true, enumerable:false, configurable:true}).
fn wire_ctor_proto(
    heap: &mut Heap,
    ctor: Gc<Object>,
    proto: Gc<Object>,
) -> Result<(), NativeError> {
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
    heap.define_property_str(
        proto,
        "constructor",
        Property {
            value: Value::Object(ctor),
            writable: true,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(())
}

/// A global binding: writable, non-enumerable, configurable (spec 18).
fn define_global(
    heap: &mut Heap,
    global: Gc<Object>,
    name: &str,
    value: Value,
) -> Result<(), NativeError> {
    heap.define_property_str(
        global,
        name,
        Property {
            value,
            writable: true,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(())
}

/// A prototype method: writable, non-enumerable, configurable (spec 18).
/// Non-enumerability matters: `for-in`/`Object.keys` must not surface
/// `toString` and friends.
fn define_proto_method(
    heap: &mut Heap,
    proto: Gc<Object>,
    name: &str,
    func: NativeFn,
    length: u32,
) -> Result<(), NativeError> {
    let f = new_native_function(heap, name, func, length)?;
    heap.define_property_str(
        proto,
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

/// `Object.prototype.toString`: `"[object Tag]"`, honoring a string
/// `@@toStringTag` on the chain.
fn object_proto_to_string(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let tag: &str = match this {
        Value::Undefined => "Undefined",
        Value::Null => "Null",
        Value::Object(o) => {
            // `Symbol.toStringTag` override, consulted under the real
            // symbol key (data properties only; accessor tags are a v1
            // gap, matching `to_string_js`).
            let tag_override = match heap.well_known[WellKnown::ToStringTag as usize] {
                Some(sym) => get_prop_key(heap, o, PropKey::Symbol(sym)),
                None => None,
            };
            if let Some(Value::Str(s)) = tag_override {
                // The tag is interpolated verbatim (lone surrogates
                // preserved), not lossy-formatted.
                let mut out = Wtf8Buf::from("[object ");
                out.push_wtf8(s.as_wtf8());
                out.push_char(']');
                return Ok(Value::Str(heap.alloc_wtf8_owned(out)?));
            }
            match heap.get(o).kind {
                ObjectKind::Array => "Array",
                ObjectKind::Function(_) | ObjectKind::NativeFunction(_) => "Function",
                ObjectKind::RegExp => "RegExp",
                ObjectKind::Date { .. } => "Date",
                ObjectKind::TypedArrayView { etype, .. } => etype.name(),
                ObjectKind::ArrayBuffer => "ArrayBuffer",
                ObjectKind::DataView { .. } => "DataView",
                ObjectKind::Ordinary => "Object",
            }
        }
        Value::Str(_) => "String",
        Value::Int(_) | Value::Float(_) => "Number",
        Value::Bool(_) => "Boolean",
        Value::BigInt(_) => "BigInt",
        Value::Symbol(_) => "Symbol",
    };
    Ok(Value::Str(heap.alloc_str(&format!("[object {tag}]"))?))
}

/// `Object.prototype.valueOf`: the receiver itself (v1: no primitive
/// boxing; `valueOf.call(5)` yields `5`, not a `Number` wrapper).
fn object_proto_value_of(
    _heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    Ok(this)
}

/// `Object.prototype.hasOwnProperty(key)`.
fn object_proto_has_own_property(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    // Spec boxes primitives via `ToObject`; v1 strings/numbers have no
    // wrapper exotics, so non-objects simply have no own properties.
    let o = match this {
        Value::Object(o) => o,
        _ => return Ok(Value::Bool(false)),
    };
    let key = to_property_key(heap, get_arg(args, 0))?;
    // `get_property_key` takes `&Heap`: no allocation, so the handle
    // needs no root.
    Ok(Value::Bool(
        heap.get_property_key(o, key.as_prop_key()).is_some(),
    ))
}

/// `Object.prototype.isPrototypeOf(value)`: prototype-chain membership.
fn object_proto_is_prototype_of(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let o = match this {
        Value::Object(o) => o,
        _ => return Ok(Value::Bool(false)),
    };
    let v = match get_arg(args, 0) {
        Value::Object(v) => v,
        _ => return Ok(Value::Bool(false)),
    };
    // Cycle-guarded walk (tortoise-and-hare), like the VM's `[[Get]]`.
    let mut hare = heap.get(v).prototype;
    let mut tortoise = hare;
    let mut advance = false;
    loop {
        match hare {
            Value::Object(h) if h == o => return Ok(Value::Bool(true)),
            Value::Object(h) => hare = heap.get(h).prototype,
            _ => return Ok(Value::Bool(false)),
        }
        if advance {
            match tortoise {
                Value::Object(t) => {
                    tortoise = heap.get(t).prototype;
                    if tortoise == hare {
                        return Ok(Value::Bool(false));
                    }
                }
                _ => return Ok(Value::Bool(false)),
            }
        }
        advance = !advance;
    }
}

/// `Object.prototype.propertyIsEnumerable(key)`.
fn object_proto_property_is_enumerable(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let o = match this {
        Value::Object(o) => o,
        _ => return Ok(Value::Bool(false)),
    };
    let key = to_property_key(heap, get_arg(args, 0))?;
    let key = key.as_prop_key();
    // `get_property_key`/`get_accessor_key` take `&Heap`: no collection
    // can run while the key is borrowed, so the handle needs no root here.
    let enumerable = heap
        .get_property_key(o, key)
        .map(|p| p.enumerable)
        .or_else(|| heap.get_accessor_key(o, key).map(|a| a.enumerable))
        .unwrap_or(false);
    Ok(Value::Bool(enumerable))
}

/// Install the `Object` constructor (+ statics) and the `Array` constructor
/// (+ statics and prototype methods) on `global`, stashing
/// `%Object.prototype%` / `%Array.prototype%` in the [`Heap`] fields the
/// constructors read back (`object_constructor` chains new objects to
/// `%Object.prototype%`; `new_array` chains new arrays to
/// `%Array.prototype%`).
///
/// Call once per heap; not idempotent.
#[allow(dead_code)] // Entry point for the runtime worker; nothing calls it yet.
pub fn install(heap: &mut Heap, global: Gc<Object>) -> Result<(), NativeError> {
    // Root `global` for the whole install: with an embedder-enabled GC
    // threshold, the allocations below could otherwise trigger a
    // collection while the global object (and the intrinsics being linked
    // into it) is still unreachable from any root.
    let base = heap.protect(Value::Object(global));
    let result = install_inner(heap, global);
    heap.release_temp(base);
    result
}

fn install_inner(heap: &mut Heap, global: Gc<Object>) -> Result<(), NativeError> {
    // ---- Object ---------------------------------------------------------
    let object_proto = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    // Rooted as a heap intrinsic before any further allocation below.
    heap.object_prototype = Some(object_proto);
    let object_ctor = new_native_constructor(heap, "Object", object_constructor, 1)?;
    wire_ctor_proto(heap, object_ctor, object_proto)?;
    // Linked into the (rooted) global before the method-install loop, so
    // the constructor is reachable across every allocation that follows.
    define_global(heap, global, "Object", Value::Object(object_ctor))?;
    // `Object.prototype` methods (non-enumerable): without `toString` the
    // VM's `ToPrimitive` cannot stringify plain objects (`"" + {}`).
    for (name, arity, f) in [
        ("toString", 0, object_proto_to_string as NativeFn),
        ("valueOf", 0, object_proto_value_of as NativeFn),
        (
            "hasOwnProperty",
            1,
            object_proto_has_own_property as NativeFn,
        ),
        ("isPrototypeOf", 1, object_proto_is_prototype_of as NativeFn),
        (
            "propertyIsEnumerable",
            1,
            object_proto_property_is_enumerable as NativeFn,
        ),
    ] {
        define_proto_method(heap, object_proto, name, f, arity)?;
    }
    for (name, arity, f) in [
        ("create", 2, object_create as NativeFn),
        ("defineProperty", 3, object_define_property as NativeFn),
        ("defineProperties", 2, object_define_properties as NativeFn),
        ("assign", 2, object_assign as NativeFn),
        ("keys", 1, object_keys as NativeFn),
        ("values", 1, object_values as NativeFn),
        ("entries", 1, object_entries as NativeFn),
        (
            "getOwnPropertyDescriptor",
            2,
            object_get_own_property_descriptor as NativeFn,
        ),
        (
            "getOwnPropertyNames",
            1,
            object_get_own_property_names as NativeFn,
        ),
        (
            "getOwnPropertySymbols",
            1,
            object_get_own_property_symbols as NativeFn,
        ),
        ("getPrototypeOf", 1, object_get_prototype_of as NativeFn),
        ("setPrototypeOf", 2, object_set_prototype_of as NativeFn),
        ("hasOwn", 2, object_has_own as NativeFn),
        ("freeze", 1, object_freeze as NativeFn),
        ("seal", 1, object_seal as NativeFn),
        (
            "preventExtensions",
            1,
            object_prevent_extensions as NativeFn,
        ),
        ("isFrozen", 1, object_is_frozen as NativeFn),
        ("isSealed", 1, object_is_sealed as NativeFn),
        ("fromEntries", 1, object_from_entries as NativeFn),
        ("groupBy", 2, object_group_by as NativeFn),
    ] {
        define_native(heap, object_ctor, name, f, arity)?;
    }

    // ---- Array ----------------------------------------------------------
    let array_proto = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    // Rooted as a heap intrinsic before any further allocation below.
    heap.array_prototype = Some(array_proto);
    heap.set_prototype(array_proto, Value::Object(object_proto));
    for (name, arity, f) in [
        ("push", 1, array_push as NativeFn),
        ("pop", 0, array_pop as NativeFn),
        ("shift", 0, array_shift as NativeFn),
        ("unshift", 1, array_unshift as NativeFn),
        ("slice", 2, array_slice as NativeFn),
        ("splice", 2, array_splice as NativeFn),
        ("concat", 1, array_concat as NativeFn),
        ("join", 1, array_join as NativeFn),
        ("indexOf", 1, array_index_of as NativeFn),
        ("lastIndexOf", 1, array_last_index_of as NativeFn),
        ("includes", 1, array_includes as NativeFn),
        ("at", 1, array_at as NativeFn),
        ("map", 1, array_map as NativeFn),
        ("filter", 1, array_filter as NativeFn),
        ("forEach", 1, array_for_each as NativeFn),
        ("reduce", 1, array_reduce as NativeFn),
        ("reduceRight", 1, array_reduce_right as NativeFn),
        ("find", 1, array_find as NativeFn),
        ("findIndex", 1, array_find_index as NativeFn),
        ("some", 1, array_some as NativeFn),
        ("every", 1, array_every as NativeFn),
        ("sort", 1, array_sort as NativeFn),
        ("toSorted", 1, array_to_sorted as NativeFn),
        ("reverse", 0, array_reverse as NativeFn),
        ("toReversed", 0, array_to_reversed as NativeFn),
        ("fill", 1, array_fill as NativeFn),
        ("copyWithin", 2, array_copy_within as NativeFn),
        ("flat", 0, array_flat as NativeFn),
        ("flatMap", 1, array_flat_map as NativeFn),
        ("findLast", 1, array_find_last as NativeFn),
        ("findLastIndex", 1, array_find_last_index as NativeFn),
        ("entries", 0, array_entries as NativeFn),
        ("keys", 0, array_keys as NativeFn),
        ("values", 0, array_values as NativeFn),
        ("with", 2, array_with as NativeFn),
        ("toSpliced", 2, array_to_spliced as NativeFn),
    ] {
        define_native(heap, array_proto, name, f, arity)?;
    }
    // `Array.prototype.toString` is the same function object as
    // `Array.prototype.join` (ECMA-262 22.1.3.27) — one shared object,
    // not a second wrapper, so identity comparisons hold.
    if let Some(join) = heap
        .get_property(array_proto, Wtf8::from_str("join"))
        .map(|p| p.value)
    {
        heap.define_property_str(
            array_proto,
            "toString",
            Property {
                value: join,
                writable: true,
                enumerable: false,
                configurable: true,
            },
        )?;
    }
    // `%Array.prototype%[Symbol.toStringTag]` = `"Array"` (ECMA-262
    // 22.1.3.31): the real well-known symbol key, never an `"@@…"` string.
    let tag_sym = heap
        .well_known_symbol(WellKnown::ToStringTag)
        .map_err(|_| NativeError::OutOfMemory)?;
    let tag_str = heap
        .alloc_str("Array")
        .map_err(|_| NativeError::OutOfMemory)?;
    heap.define_property_key(
        array_proto,
        PropKey::Symbol(tag_sym),
        Property {
            value: Value::Str(tag_str),
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )
    .map_err(|_| NativeError::OutOfMemory)?;
    let array_ctor = new_native_constructor(heap, "Array", array_constructor, 1)?;
    wire_ctor_proto(heap, array_ctor, array_proto)?;
    // Rooted as a heap intrinsic: the `ArraySpeciesCreate` default
    // constructor (`SpeciesConstructor(O, %Array%)`).
    heap.array_ctor = Some(array_ctor);
    // Linked into the (rooted) global before the static-install loop.
    define_global(heap, global, "Array", Value::Object(array_ctor))?;
    for (name, arity, f) in [
        ("isArray", 1, array_is_array as NativeFn),
        ("from", 1, array_from as NativeFn),
        ("of", 0, array_of as NativeFn),
    ] {
        define_native(heap, array_ctor, name, f, arity)?;
    }
    // `get Array [ @@species ]` (ECMA-262 23.1.2.5): returns `this`,
    // so subclass constructors propagate through `ArraySpeciesCreate`
    // (`map`/`filter`/`slice`/etc. on a `class B extends Array` build
    // `B` instances by default). Mirrors the `%TypedArray%` install.
    let species_sym = heap
        .well_known_symbol(WellKnown::Species)
        .map_err(|_| NativeError::OutOfMemory)?;
    let species_get = new_native_function(heap, "get [Symbol.species]", array_species_get, 0)?;
    heap.define_accessor_key(
        array_ctor,
        PropKey::Symbol(species_sym),
        AccessorProp {
            get: Value::Object(species_get),
            set: Value::Undefined,
            enumerable: false,
            configurable: true,
        },
    )
    .map_err(|_| NativeError::OutOfMemory)?;
    Ok(())
}

/// `get Array [ @@species ]`: returns `this` (see the install above).
fn array_species_get(_heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    Ok(this)
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

    /// A global constructor as (object, native fn).
    fn ctor(heap: &Heap, global: Gc<Object>, name: &str) -> (Gc<Object>, NativeFn) {
        match heap
            .get_property(global, Wtf8::from_str(name))
            .unwrap()
            .value
        {
            Value::Object(o) => match heap.get(o).kind {
                ObjectKind::NativeFunction(f) => (o, f),
                _ => panic!("{name} is not a native function"),
            },
            v => panic!("no {name} on global: {v:?}"),
        }
    }

    fn proto_of(heap: &Heap, ctor_obj: Gc<Object>) -> Gc<Object> {
        match heap
            .get_property(ctor_obj, Wtf8::from_str("prototype"))
            .unwrap()
            .value
        {
            Value::Object(o) => o,
            v => panic!("prototype is not an object: {v:?}"),
        }
    }

    fn method(heap: &Heap, proto: Gc<Object>, name: &str) -> NativeFn {
        match heap
            .get_property(proto, Wtf8::from_str(name))
            .unwrap()
            .value
        {
            Value::Object(o) => match heap.get(o).kind {
                ObjectKind::NativeFunction(f) => f,
                _ => panic!("{name} is not a native function"),
            },
            v => panic!("no method {name}: {v:?}"),
        }
    }

    fn int_array(heap: &mut Heap, elems: &[i32]) -> Gc<Object> {
        let arr = new_array(heap).unwrap();
        for (i, e) in elems.iter().enumerate() {
            set_idx(heap, arr, i as u64, Value::Int(*e)).unwrap();
        }
        arr
    }

    fn to_vec(heap: &Heap, arr: Gc<Object>) -> Vec<Value> {
        (0..array_length(heap, arr))
            .map(|i| get_idx(heap, arr, i))
            .collect()
    }

    fn double(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
        Ok(num(to_number(get_arg(args, 0))? * 2.0))
    }

    fn is_even(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
        Ok(Value::Bool(to_number(get_arg(args, 0))? % 2.0 == 0.0))
    }

    fn native_callback(heap: &mut Heap, name: &str, f: NativeFn) -> Value {
        // Reuse the sibling helper for spec-shaped name/length.
        Value::Object(crate::builtin_string_number::new_native_function(heap, name, f, 1).unwrap())
    }

    #[test]
    fn install_wires_constructors_prototypes_and_heap_fields() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (object_ctor, _) = ctor(&heap, global, "Object");
        let (array_ctor, _) = ctor(&heap, global, "Array");
        let object_proto = proto_of(&heap, object_ctor);
        let array_proto = proto_of(&heap, array_ctor);
        // Heap intrinsics stashed for the constructors to find.
        assert_eq!(heap.object_prototype, Some(object_proto));
        assert_eq!(heap.array_prototype, Some(array_proto));
        // Prototype chain: %Array.prototype% -> %Object.prototype% -> null.
        assert_eq!(heap.get(array_proto).prototype, Value::Object(object_proto));
        assert_eq!(heap.get(object_proto).prototype, Value::Null);
        // constructor links both ways.
        assert_eq!(
            heap.get_property(object_proto, Wtf8::from_str("constructor"))
                .unwrap()
                .value,
            Value::Object(object_ctor)
        );
        assert_eq!(
            heap.get_property(array_proto, Wtf8::from_str("constructor"))
                .unwrap()
                .value,
            Value::Object(array_ctor)
        );
        // New objects/arrays chain to the installed prototypes.
        let arr = new_array(&mut heap).unwrap();
        assert_eq!(heap.get(arr).prototype, Value::Object(array_proto));
    }

    #[test]
    fn object_statics_installed() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (object_ctor, _) = ctor(&heap, global, "Object");
        for name in [
            "create",
            "defineProperty",
            "defineProperties",
            "assign",
            "keys",
            "values",
            "entries",
            "getOwnPropertyDescriptor",
            "getOwnPropertyNames",
            "getPrototypeOf",
            "setPrototypeOf",
            "hasOwn",
            "freeze",
            "seal",
            "preventExtensions",
            "isFrozen",
            "isSealed",
            "fromEntries",
            "groupBy",
        ] {
            assert!(
                heap.get_property(object_ctor, Wtf8::from_str(name))
                    .is_some(),
                "Object.{name} installed"
            );
        }
    }

    #[test]
    fn array_statics_and_methods_installed() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (array_ctor, _) = ctor(&heap, global, "Array");
        for name in ["isArray", "from", "of"] {
            assert!(
                heap.get_property(array_ctor, Wtf8::from_str(name))
                    .is_some(),
                "Array.{name} installed"
            );
        }
        let array_proto = proto_of(&heap, array_ctor);
        for name in [
            "push",
            "pop",
            "shift",
            "unshift",
            "slice",
            "splice",
            "concat",
            "join",
            "indexOf",
            "lastIndexOf",
            "includes",
            "at",
            "map",
            "filter",
        ] {
            assert!(
                heap.get_property(array_proto, Wtf8::from_str(name))
                    .is_some(),
                "Array.prototype.{name} installed"
            );
        }
    }

    #[test]
    fn object_get_own_property_names_sees_non_enumerable() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (object_ctor, _) = ctor(&heap, global, "Object");
        let get_names = method(&heap, object_ctor, "getOwnPropertyNames");
        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        // Enumerable data property.
        heap.define_property_str(obj, "b", Property::data(Value::Int(1)))
            .unwrap();
        // Non-enumerable data property: invisible to Object.keys, visible
        // to getOwnPropertyNames.
        heap.define_property_str(
            obj,
            "a",
            Property {
                value: Value::Int(2),
                writable: true,
                enumerable: false,
                configurable: true,
            },
        )
        .unwrap();
        // Array-index keys sort numerically first, regardless of insertion
        // order.
        heap.define_property_str(obj, "10", Property::data(Value::Int(3)))
            .unwrap();
        heap.define_property_str(obj, "2", Property::data(Value::Int(4)))
            .unwrap();
        // Accessor property lives in the side table; still an own key.
        heap.define_accessor(
            obj,
            Wtf8::from_str("acc"),
            AccessorProp {
                get: Value::Undefined,
                set: Value::Undefined,
                enumerable: false,
                configurable: true,
            },
        )
        .unwrap();
        let names = match get_names(&mut heap, Value::Undefined, &[Value::Object(obj)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected array, got {v:?}"),
        };
        let got: Vec<String> = to_vec(&heap, names)
            .iter()
            .map(|v| match v {
                Value::Str(s) => s.as_wtf8().to_string(),
                v => panic!("expected string key, got {v:?}"),
            })
            .collect();
        assert_eq!(got, vec!["2", "10", "b", "a", "acc"]);
        // Non-object argument throws a TypeError.
        assert!(matches!(
            get_names(&mut heap, Value::Undefined, &[Value::Int(1)]),
            Err(NativeError::TypeError(_))
        ));
        assert!(matches!(
            get_names(&mut heap, Value::Undefined, &[]),
            Err(NativeError::TypeError(_))
        ));
    }

    fn key_strings(heap: &Heap, arr: Gc<Object>) -> Vec<String> {
        to_vec(heap, arr)
            .iter()
            .map(|v| match v {
                Value::Str(s) => s.as_wtf8().to_string(),
                v => panic!("expected string key, got {v:?}"),
            })
            .collect()
    }

    #[test]
    fn object_get_own_property_symbols_lists_all_symbol_keys() {
        // Enumerable and non-enumerable symbol keys, in definition order;
        // string keys never appear.
        assert_eq!(
            eval_str(
                "var s1 = Symbol('a'), s2 = Symbol(); \
                 var o = {x: 1}; \
                 o[s1] = 10; \
                 Object.defineProperty(o, s2, {value: 20, enumerable: false}); \
                 var syms = Object.getOwnPropertySymbols(o); \
                 syms.length + ',' + (syms[0] === s1) + ',' + (syms[1] === s2)"
            ),
            "2,true,true"
        );
        // String-only sibling is unaffected.
        assert_eq!(
            eval_str(
                "var s = Symbol('a'); \
                 var o = {}; o[s] = 1; o.y = 2; \
                 Object.getOwnPropertyNames(o).join(',')"
            ),
            "y"
        );
        // Non-object argument throws.
        assert_eq!(
            eval_str(
                "var r = 'no-throw'; \
                 try { Object.getOwnPropertySymbols(1); } \
                 catch (e) { r = 'threw'; } r"
            ),
            "threw"
        );
    }

    #[test]
    fn object_assign_copies_enumerable_symbol_keys() {
        assert_eq!(
            eval_str(
                "var s = Symbol('s'), hidden = Symbol('h'); \
                 var src = {a: 1}; \
                 src[s] = 42; \
                 Object.defineProperty(src, hidden, {value: 9, enumerable: false}); \
                 var t = Object.assign({}, src); \
                 t.a + ',' + t[s] + ',' + Object.getOwnPropertySymbols(t).length"
            ),
            "1,42,1"
        );
    }

    #[test]
    fn object_spread_copies_enumerable_symbol_keys() {
        assert_eq!(
            eval_str(
                "var s = Symbol('s'); \
                 var src = {b: 2}; src[s] = 7; \
                 var t = {...src}; \
                 t.b + ',' + t[s]"
            ),
            "2,7"
        );
        // Non-enumerable symbol keys are not spread.
        assert_eq!(
            eval_str(
                "var s = Symbol('s'); \
                 var src = {}; \
                 Object.defineProperty(src, s, {value: 7, enumerable: false}); \
                 var t = {...src}; \
                 Object.getOwnPropertySymbols(t).length"
            ),
            "0"
        );
    }

    #[test]
    fn object_keys_sees_dense_array_indices() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (object_ctor, _) = ctor(&heap, global, "Object");
        let keys = method(&heap, object_ctor, "keys");
        let values = method(&heap, object_ctor, "values");
        let entries = method(&heap, object_ctor, "entries");
        let get_names = method(&heap, object_ctor, "getOwnPropertyNames");

        // Dense-active array: indices live in `dense`, not the property
        // map. `Object.keys` must still see them (the dense-array bug).
        let arr = heap.new_dense_array().unwrap();
        for v in [10, 20, 30] {
            heap.dense_push(arr, Value::Int(v)).unwrap();
        }
        // A named property enumerates after the indices, in insertion order.
        heap.define_property_str(arr, "extra", Property::data(Value::Int(99)))
            .unwrap();

        let k = match keys(&mut heap, Value::Undefined, &[Value::Object(arr)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected array, got {v:?}"),
        };
        assert_eq!(key_strings(&heap, k), vec!["0", "1", "2", "extra"]);

        let v = match values(&mut heap, Value::Undefined, &[Value::Object(arr)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected array, got {v:?}"),
        };
        assert_eq!(
            to_vec(&heap, v),
            vec![
                Value::Int(10),
                Value::Int(20),
                Value::Int(30),
                Value::Int(99)
            ]
        );

        let e = match entries(&mut heap, Value::Undefined, &[Value::Object(arr)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected array, got {v:?}"),
        };
        let pairs: Vec<(String, Value)> = to_vec(&heap, e)
            .iter()
            .map(|p| match p {
                Value::Object(pair) => {
                    let kv = to_vec(&heap, *pair);
                    let k = match kv[0] {
                        Value::Str(s) => s.as_wtf8().to_string(),
                        v => panic!("expected string key, got {v:?}"),
                    };
                    (k, kv[1])
                }
                v => panic!("expected pair array, got {v:?}"),
            })
            .collect();
        assert_eq!(
            pairs,
            vec![
                ("0".to_owned(), Value::Int(10)),
                ("1".to_owned(), Value::Int(20)),
                ("2".to_owned(), Value::Int(30)),
                ("extra".to_owned(), Value::Int(99)),
            ]
        );

        // `getOwnPropertyNames` sees the same index keys on dense arrays,
        // plus the synthesized `"length"` (which never lives in the
        // property map while dense storage is active).
        let n = match get_names(&mut heap, Value::Undefined, &[Value::Object(arr)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected array, got {v:?}"),
        };
        assert_eq!(
            key_strings(&heap, n),
            vec!["0", "1", "2", "length", "extra"]
        );
    }

    #[test]
    fn object_keys_skips_dense_holes() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (object_ctor, _) = ctor(&heap, global, "Object");
        let keys = method(&heap, object_ctor, "keys");
        let get_names = method(&heap, object_ctor, "getOwnPropertyNames");

        let arr = heap.new_dense_array().unwrap();
        heap.dense_set(arr, 0, Value::Int(1)).unwrap();
        heap.dense_set(arr, 3, Value::Int(4)).unwrap(); // holes at 1, 2
        assert!(heap.get(arr).dense().is_some(), "array must stay dense");

        let k = match keys(&mut heap, Value::Undefined, &[Value::Object(arr)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected array, got {v:?}"),
        };
        assert_eq!(key_strings(&heap, k), vec!["0", "3"]);

        let n = match get_names(&mut heap, Value::Undefined, &[Value::Object(arr)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected array, got {v:?}"),
        };
        assert_eq!(key_strings(&heap, n), vec!["0", "3", "length"]);
    }

    #[test]
    fn object_keys_dense_only_skips_sort() {
        let mut heap = Heap::new();
        // Large dense-only array: no property-map index key exists, so the
        // ascending dense scan needs no sort (O(n) fast path).
        let arr = heap.new_dense_array().unwrap();
        const N: usize = 10_000;
        for i in 0..N {
            heap.dense_push(arr, Value::Int(i as i32)).unwrap();
        }
        let (keys, sorted) = own_enumerable_keys_detailed(&heap, arr);
        assert!(!sorted, "dense-only keys must take the no-sort fast path");
        assert_eq!(keys.len(), N);
        for (i, k) in keys.iter().enumerate() {
            assert_eq!(
                k.as_wtf8(),
                Wtf8::from_str(&i.to_string()),
                "keys ascending, no holes"
            );
        }
        // A property-map index key forces the merge path: the sort runs
        // and still yields ascending indices.
        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        for (k, v) in [("10", 10), ("2", 2), ("name", 0)] {
            heap.define_property_str(obj, k, Property::data(Value::Int(v)))
                .unwrap();
        }
        let (keys, sorted) = own_enumerable_keys_detailed(&heap, obj);
        assert!(sorted, "merged property-map index keys need the sort");
        let strs: Vec<&Wtf8> = keys.iter().map(|k| k.as_wtf8()).collect();
        assert_eq!(
            strs,
            vec![
                Wtf8::from_str("2"),
                Wtf8::from_str("10"),
                Wtf8::from_str("name")
            ]
        );
    }

    #[test]
    fn property_keys_pack_inline_when_short() {
        let mut heap = Heap::new();
        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let base = heap.protect(Value::Object(obj));
        heap.define_property_str(obj, "x", Property::data(Value::Int(1)))
            .unwrap();
        let long = "a-much-longer-property-name";
        assert!(long.len() > crate::value::STR_INLINE_MAX);
        heap.define_property_str(obj, long, Property::data(Value::Int(2)))
            .unwrap();
        heap.define_accessor(
            obj,
            Wtf8::from_str("acc"),
            AccessorProp {
                get: Value::Undefined,
                set: Value::Undefined,
                enumerable: true,
                configurable: true,
            },
        )
        .unwrap();
        {
            let o = heap.get(obj);
            assert!(o.properties[0].0.is_inline(), "short key packs inline");
            assert!(
                !o.properties[1].0.is_inline(),
                "long key stays heap-allocated"
            );
            assert!(o.accessors[0].0.is_inline(), "short accessor key inline");
        }
        // Lookups round-trip by content across both representations.
        assert_eq!(
            heap.get_property(obj, Wtf8::from_str("x")).unwrap().value,
            Value::Int(1)
        );
        assert_eq!(
            heap.get_property(obj, Wtf8::from_str(long)).unwrap().value,
            Value::Int(2)
        );
        assert!(heap.get_accessor(obj, Wtf8::from_str("acc")).is_some());
        // A collection frees nothing reachable: inline keys need no
        // tracing, the long key is traced through the object.
        heap.collect();
        assert_eq!(
            heap.get_property(obj, Wtf8::from_str("x")).unwrap().value,
            Value::Int(1)
        );
        assert_eq!(
            heap.get_property(obj, Wtf8::from_str(long)).unwrap().value,
            Value::Int(2)
        );
        assert!(heap.get_accessor(obj, Wtf8::from_str("acc")).is_some());
        heap.release_temp(base);
    }

    #[test]
    fn array_constructor_behaviors() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (_, array) = ctor(&heap, global, "Array");
        // new Array(3): length 3, holes.
        let a3 = match array(&mut heap, Value::Undefined, &[Value::Int(3)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        };
        assert_eq!(array_length(&heap, a3), 3);
        assert!(!has_idx(&heap, a3, 0));
        // new Array(1, 2): elements.
        let a12 = match array(&mut heap, Value::Undefined, &[Value::Int(1), Value::Int(2)]).unwrap()
        {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        };
        assert_eq!(to_vec(&heap, a12), vec![Value::Int(1), Value::Int(2)]);
        // Negative length throws RangeError.
        assert!(matches!(
            array(&mut heap, Value::Undefined, &[Value::Int(-1)]),
            Err(NativeError::RangeError(_))
        ));
    }

    #[test]
    fn array_is_array_static() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (array_ctor, _) = ctor(&heap, global, "Array");
        let is_array = method(&heap, array_ctor, "isArray");
        let arr = Value::Object(int_array(&mut heap, &[1]));
        assert_eq!(
            is_array(&mut heap, Value::Undefined, &[arr]).unwrap(),
            Value::Bool(true)
        );
        let plain = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        assert_eq!(
            is_array(&mut heap, Value::Undefined, &[Value::Object(plain)]).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn array_map_doubles_and_preserves_holes() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (array_ctor, _) = ctor(&heap, global, "Array");
        let map = method(&heap, proto_of(&heap, array_ctor), "map");
        let arr = int_array(&mut heap, &[1, 2, 3]);
        let cb = native_callback(&mut heap, "double", double);
        let out = match map(&mut heap, Value::Object(arr), &[cb]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        };
        assert_eq!(
            to_vec(&heap, out),
            vec![Value::Int(2), Value::Int(4), Value::Int(6)]
        );
        // Holes stay holes and the callback never sees them.
        let sparse = new_array_with_length(&mut heap, 3).unwrap();
        set_idx(&mut heap, sparse, 0, Value::Int(5)).unwrap();
        set_idx(&mut heap, sparse, 2, Value::Int(7)).unwrap();
        let out = match map(&mut heap, Value::Object(sparse), &[cb]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        };
        assert_eq!(array_length(&heap, out), 3);
        assert_eq!(get_idx(&heap, out, 0), Value::Int(10));
        assert!(!has_idx(&heap, out, 1));
        assert_eq!(get_idx(&heap, out, 2), Value::Int(14));
        // Missing callback throws.
        assert!(matches!(
            map(&mut heap, Value::Object(arr), &[]),
            Err(NativeError::TypeError(_))
        ));
    }

    #[test]
    fn array_filter_keeps_matches_and_skips_holes() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (array_ctor, _) = ctor(&heap, global, "Array");
        let filter = method(&heap, proto_of(&heap, array_ctor), "filter");
        let arr = int_array(&mut heap, &[1, 2, 3, 4]);
        let cb = native_callback(&mut heap, "isEven", is_even);
        let out = match filter(&mut heap, Value::Object(arr), &[cb]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        };
        assert_eq!(to_vec(&heap, out), vec![Value::Int(2), Value::Int(4)]);
    }

    #[test]
    fn array_slice_copies_range() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (array_ctor, _) = ctor(&heap, global, "Array");
        let slice = method(&heap, proto_of(&heap, array_ctor), "slice");
        let arr = int_array(&mut heap, &[1, 2, 3, 4]);
        let out = match slice(
            &mut heap,
            Value::Object(arr),
            &[Value::Int(1), Value::Int(3)],
        )
        .unwrap()
        {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        };
        assert_eq!(to_vec(&heap, out), vec![Value::Int(2), Value::Int(3)]);
        // Negative start counts from the end.
        let out = match slice(&mut heap, Value::Object(arr), &[Value::Int(-2)]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        };
        assert_eq!(to_vec(&heap, out), vec![Value::Int(3), Value::Int(4)]);
    }

    #[test]
    fn object_constructor_chains_heap_prototype() {
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let (_, object) = ctor(&heap, global, "Object");
        let o = match object(&mut heap, Value::Undefined, &[]).unwrap() {
            Value::Object(o) => o,
            v => panic!("expected object, got {v:?}"),
        };
        assert_eq!(
            heap.get(o).prototype,
            Value::Object(heap.object_prototype.unwrap())
        );
        // Object.keys on a fresh object.
        let (object_ctor, _) = ctor(&heap, global, "Object");
        let keys = method(&heap, object_ctor, "keys");
        heap.define_property_str(o, "a", Property::data(Value::Int(1)))
            .unwrap();
        let ks = match keys(&mut heap, Value::Undefined, &[Value::Object(o)]).unwrap() {
            Value::Object(a) => a,
            v => panic!("expected object, got {v:?}"),
        };
        assert_eq!(array_length(&heap, ks), 1);
        match get_idx(&heap, ks, 0) {
            Value::Str(g) => assert_eq!(g.as_wtf8(), Wtf8::from_str("a")),
            v => panic!("expected string key, got {v:?}"),
        }
    }

    // -- BigInt decimal conversion ---------------------------------------

    #[test]
    fn bigint_to_decimal_digits() {
        // Zero (also with stray zero digits and a negative sign).
        assert_eq!(bigint_to_decimal(false, &[]), "0");
        assert_eq!(bigint_to_decimal(false, &[0, 0]), "0");
        assert_eq!(bigint_to_decimal(true, &[0]), "0");
        // Single digit / single limb.
        assert_eq!(bigint_to_decimal(false, &[10]), "10");
        assert_eq!(bigint_to_decimal(true, &[10]), "-10");
        // Exactly 10^9: crosses one chunk boundary.
        assert_eq!(bigint_to_decimal(false, &[1_000_000_000]), "1000000000");
        // 2^32 = 4294967296: needs the multi-limb divide path.
        assert_eq!(bigint_to_decimal(false, &[0, 1]), "4294967296");
        // 10^20 - 1 = 99999999999999999999 (three chunks, zero-padded).
        assert_eq!(
            bigint_to_decimal(false, &[0x630FFFFF, 0x6BC75E2D, 0x5]),
            "99999999999999999999"
        );
        assert_eq!(
            bigint_to_decimal(true, &[0x630FFFFF, 0x6BC75E2D, 0x5]),
            "-99999999999999999999"
        );
    }

    /// Evaluate JS in a full VM and stringify the completion value.
    fn eval_str(source: &str) -> String {
        let mut vm = crate::vm::Vm::new();
        let v = vm
            .eval(source)
            .unwrap_or_else(|e| panic!("eval failed: {e}"));
        to_string_js(vm.heap_mut(), v)
            .map(|w| w.to_string())
            .unwrap_or_else(|e| panic!("to_string failed: {e:?}"))
    }

    #[test]
    fn to_string_js_converts_bigint() {
        assert_eq!(eval_str("String(10n)"), "10");
        assert_eq!(
            eval_str("String(-123456789012345678901234567890n)"),
            "-123456789012345678901234567890"
        );
        assert_eq!(eval_str("'' + 10n"), "10");
        assert_eq!(eval_str("10n + ''"), "10");
        assert_eq!(eval_str("String(0n)"), "0");
    }

    /// Differential: the zero-alloc stack-buffer formatters must be
    /// byte-identical to the allocating `String` paths, and float output
    /// must stay within the hand-verified 25-byte bound (< 32), pinning
    /// the `[u8; 32]` sizing of [`js_number_to_str`].
    #[test]
    fn stack_number_formats_match_heap_formats() {
        let edge_floats: &[f64] = &[
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            5e-324,                    // smallest positive subnormal
            1.7976931348623157e308,    // f64::MAX
            -0.0000012345678901234567, // decimal/exponential boundary
            1e21,                      // exponential threshold
            -1e21,
            123_456_789.123_456_79,
            1e-7,
            0.1,
            0.3,
            2.5,
            -9999999999999998.0,
        ];
        for &n in edge_floats {
            let mut buf = [0u8; 32];
            let stack = js_number_to_str(n, &mut buf);
            if n.is_nan() {
                // NaN never equals itself; pin the canonical spelling.
                assert_eq!(stack, "NaN", "js_number_to_str(NaN)");
            } else {
                let heap = js_number_to_string(n);
                assert_eq!(stack, heap, "js_number_to_str({n:?})");
            }
            assert!(
                stack.len() <= 25,
                "js_number_to_str({n:?}) emitted {} bytes, over the 25-byte bound",
                stack.len()
            );
        }

        let edge_ints: &[i32] = &[
            i32::MIN,
            i32::MIN + 1,
            -1_000_000,
            -1,
            0,
            1,
            1_000_000,
            i32::MAX - 1,
            i32::MAX,
        ];
        for &v in edge_ints {
            let mut buf = [0u8; 11];
            assert_eq!(int_to_str(v, &mut buf), v.to_string(), "int_to_str({v})");
        }
        // Dense sweep: every small int formats exactly.
        for v in -200_000..=200_000i32 {
            let mut buf = [0u8; 11];
            assert_eq!(int_to_str(v, &mut buf), v.to_string(), "int_to_str({v})");
        }
    }

    /// Native `toString` for [`array_join_sep_rooted_across_forced_collect`]:
    /// forces a collection in the middle of `array_join`'s element loop
    /// (freeing any unrooted string handle the builtin is holding), then
    /// allocates garbage in the same 16-byte size class so a freed block
    /// is observably reused. Returns a long (>15 byte) heap string.
    fn gc_stress_collecting_to_string(
        heap: &mut Heap,
        _this: Value,
        _args: &[Value],
    ) -> Result<Value, NativeError> {
        heap.collect();
        for _ in 0..200 {
            let _ = heap.alloc_str("GARBAGEGARBAGEGA")?;
        }
        Ok(Value::Str(heap.alloc_str("0123456789ABCDEF")?))
    }

    #[test]
    fn array_join_sep_rooted_across_forced_collect() {
        // Rooting test for the separator handle in `array_join`: the
        // separator is converted once via `to_string_js_str` into a
        // fresh, unrooted heap string, then element 0's `toString` forces
        // a collection while the separator borrow is live across loop
        // iterations. Without the temporary root on the separator the
        // collection frees it, the garbage allocs reuse its block, and
        // the join emits the garbage bytes. (A plain `set_gc_threshold(1)`
        // would not work here: the threshold ratchets to twice the live
        // set after the first collection, and the install's live set is
        // far larger than this script allocates.)
        let mut heap = Heap::new();
        let global = install_global(&mut heap);
        let gbase = heap.protect(Value::Object(global));
        let array_ctor = ctor(&heap, global, "Array").0;
        let join = method(&heap, proto_of(&heap, array_ctor), "join");

        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let to_string_fn = native_callback(&mut heap, "toString", gc_stress_collecting_to_string);
        set_prop(&mut heap, obj, Wtf8::from_str("toString"), to_string_fn).unwrap();
        let arr = new_array(&mut heap).unwrap();
        let abase = heap.protect(Value::Object(arr));
        set_idx(&mut heap, arr, 0, Value::Object(obj)).unwrap();
        let second = heap.alloc_str("second").unwrap();
        set_idx(&mut heap, arr, 1, Value::Str(second)).unwrap();
        // Fresh 16-byte heap string, deliberately unrooted: `array_join`
        // itself must root it across the element loop.
        let sep = heap.alloc_str("SEPSEPSEPSEPSEPSE").unwrap();
        let v = join(&mut heap, Value::Object(arr), &[Value::Str(sep)]).unwrap();
        let s = to_string_js(&mut heap, v).map(|w| w.to_string()).unwrap();
        heap.release_temp(abase);
        assert_eq!(s, "0123456789ABCDEFSEPSEPSEPSEPSEPSEsecond");
        heap.release_temp(gbase);
    }

    #[test]
    fn join_and_property_key_str_paths() {
        // Behavior pin for the migrated `Str`-returning paths: long keys
        // and separators (>15 bytes) take the `Str::Heap` arms.
        assert_eq!(
            eval_str(
                "['aaaaaaaaaaaaaaaaaaaaaaaa','bbbbbbbbbbbbbbbbbbbbbbbb'].join('0123456789ABCDEF')"
            ),
            "aaaaaaaaaaaaaaaaaaaaaaaa0123456789ABCDEFbbbbbbbbbbbbbbbbbbbbbbbb",
        );
        assert_eq!(
            eval_str("Object.keys(Object.fromEntries([['kkkkkkkkkkkkkkkkkkkkkkkkkk', 42]]))[0]"),
            "kkkkkkkkkkkkkkkkkkkkkkkkkk",
        );
        assert_eq!(
            eval_str("[{ toString: () => 'ffffffffffffffffffffffff' }].join('0123456789ABCDEF')"),
            "ffffffffffffffffffffffff",
        );
    }
    #[test]
    fn string_object_uses_toprimitive_string_hint() {
        // Spec 7.1.2: `String(obj)` is `ToPrimitive(obj, string)`:
        // `@@toPrimitive("string")` first, then `toString`, then `valueOf`.
        for src in [
            "String({ valueOf: () => 'v', toString: () => 't' }) === 't';",
            "var s = { [Symbol.toPrimitive]: (h) => 'S:' + h, toString: () => 't' }; String(s) === 'S:string';",
            "String({ toString: 5, valueOf: () => 'v' }) === 'v';",
            // No usable hook: the historical "[object Object]" fallback.
            "String({ valueOf: () => 'v' }) === '[object Object]';",
            "String(Object.create(null)) === '[object Object]';",
        ] {
            assert_eq!(eval_str(src), "true", "{src}");
        }
    }

    #[test]
    fn to_string_js_object_arm_hook_paths() {
        // The object arm of `to_string_js` fetches each hook with a single
        // prototype-chain walk; these pin the observable outcomes of every
        // path through the restructured lookup.
        for (src, want) in [
            // @@toPrimitive hook returning a primitive.
            ("String({ [Symbol.toPrimitive]: () => 42 })", "42"),
            (
                "String({ [Symbol.toPrimitive]: (h) => 'S:' + h })",
                "S:string",
            ),
            // @@toPrimitive returning an object: the fallback, not a throw.
            (
                "String({ [Symbol.toPrimitive]: () => ({}) })",
                "[object Object]",
            ),
            // toString returning an object falls through to valueOf.
            ("String({ toString: () => ({}), valueOf: () => 'v' })", "v"),
            // Neither yields a primitive: the fallback.
            (
                "String({ toString: () => ({}), valueOf: () => ({}) })",
                "[object Object]",
            ),
            // Non-callable toString is skipped; valueOf is still tried.
            ("String({ toString: 5, valueOf: () => 'v' })", "v"),
            // A throwing toString maps to the fallback (v1 behavior).
            (
                "String({ toString: () => { throw 1; } })",
                "[object Object]",
            ),
            // Inherited hooks work through the prototype chain.
            (
                "var p = { toString: () => 'inh' }; String(Object.create(p))",
                "inh",
            ),
            (
                "var p = { valueOf: () => 'pv' }; String(Object.create(p))",
                "[object Object]",
            ),
        ] {
            assert_eq!(eval_str(src), want, "{src}");
        }
    }

    // -- Prototype-cycle hang regressions ----------------------------------

    #[test]
    fn check_prototype_cycle_rejects_cycles() {
        let mut heap = Heap::new();
        let a = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let b = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.set_prototype(a, Value::Object(b));
        // b -> a would close a 2-cycle.
        assert!(check_prototype_cycle(&heap, b, Value::Object(a)).is_err());
        // a -> a is a self-cycle.
        assert!(check_prototype_cycle(&heap, a, Value::Object(a)).is_err());
        // Null prototype and acyclic chains are fine.
        assert!(check_prototype_cycle(&heap, a, Value::Null).is_ok());
        let c = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        assert!(check_prototype_cycle(&heap, c, Value::Object(a)).is_ok());
    }

    #[test]
    fn object_set_prototype_of_throws_on_cycle() {
        let mut heap = Heap::new();
        let a = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let err = object_set_prototype_of(
            &mut heap,
            Value::Undefined,
            &[Value::Object(a), Value::Object(a)],
        )
        .unwrap_err();
        assert!(matches!(err, NativeError::TypeError(_)));
    }

    #[test]
    fn get_prop_terminates_on_prototype_cycle() {
        // A 2-cycle built directly on the heap, bypassing the
        // `[[SetPrototypeOf]]` cycle checks: the guarded walks must
        // terminate (returning "not found") instead of looping forever.
        let mut heap = Heap::new();
        let a = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let b = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.set_prototype(a, Value::Object(b));
        heap.set_prototype(b, Value::Object(a));
        assert_eq!(get_prop(&heap, a, Wtf8::from_str("toString")), None);
        assert!(!has_prop(&heap, a, Wtf8::from_str("toString")));
    }

    // -- Regression tests: flat / flatMap ----------------------------------
    //
    // Both used to append through `set_idx` (per-element property
    // interning plus a linear property scan: O(n²) per call). Results are
    // now built with dense storage and materialized once.

    #[test]
    fn flat_depth_and_shape() {
        // Depth semantics (default 1).
        assert_eq!(eval_str("[1, [2, [3]]].flat().length"), "3");
        assert_eq!(eval_str("[1, [2, [3]]].flat(0).length"), "2");
        assert_eq!(eval_str("[1, [2, [3]]].flat(5).join(',')"), "1,2,3");
        // Deep nesting flattens fully with a big depth.
        assert_eq!(eval_str("[1, [2, [3, [4]]]].flat(10).join(',')"), "1,2,3,4");
        // Results are real arrays.
        assert_eq!(eval_str("Array.isArray([1, [2]].flat())"), "true");
        assert_eq!(eval_str("[1, [2]].flat() instanceof Array"), "true");
    }

    #[test]
    fn flat_large_is_linear() {
        // 2000 single-element sub-arrays: previously 2000 linear scans.
        assert_eq!(
            eval_str(
                "var a = []; for (var i = 0; i < 2000; i++) a.push([i]); var f = a.flat(); f.length === 2000 && f[1999] === 1999;"
            ),
            "true"
        );
    }

    #[test]
    fn flat_map_basic_and_callback_shapes() {
        assert_eq!(
            eval_str("[1, 2, 3].flatMap(function(x) { return [x, -x]; }).join(',')"),
            "1,-1,2,-2,3,-3"
        );
        // Non-array returns are appended as-is.
        assert_eq!(
            eval_str("[1, 2].flatMap(function(x) { return x * 10; }).join(',')"),
            "10,20"
        );
        // A bound callback goes through the `call_function` bound-function
        // interception; bound args precede the callback's own args.
        assert_eq!(
            eval_str(
                "[1, 2].flatMap(function(p, x) { return [p + x]; }.bind(null, 100)).join(',')"
            ),
            "101,102"
        );
        assert_eq!(
            eval_str("Array.isArray([1].flatMap(function(x) { return x; }))"),
            "true"
        );
    }

    /// Regression: partial descriptors used to reset absent fields to
    /// defaults (`Object.defineProperty(Array.prototype, "map",
    /// {enumerable: true})` wiped the function value to `undefined`). Per
    /// `ValidateAndApplyPropertyDescriptor`, absent fields are preserved
    /// from the current property.
    #[test]
    fn define_property_partial_descriptor_preserves_absent_fields() {
        assert_eq!(
            eval_str(
                "var o = {x: 42}; \
                 Object.defineProperty(o, 'x', {enumerable: false}); \
                 var d = Object.getOwnPropertyDescriptor(o, 'x'); \
                 d.value + ',' + d.writable + ',' + d.enumerable + ',' + d.configurable"
            ),
            "42,true,false,true"
        );
        // The exact finding: flipping enumerable on Array.prototype.map
        // must not wipe its function value or the other attributes.
        assert_eq!(
            eval_str(
                "Object.defineProperty(Array.prototype, 'map', {enumerable: true}); \
                 var d = Object.getOwnPropertyDescriptor(Array.prototype, 'map'); \
                 typeof d.value + ',' + d.writable + ',' + d.enumerable + ',' + d.configurable"
            ),
            "function,true,true,true"
        );
        // defineProperties shares the same merge path.
        assert_eq!(
            eval_str(
                "var o = {}; \
                 Object.defineProperties(o, {x: {value: 3, configurable: true}}); \
                 Object.defineProperties(o, {x: {enumerable: true}}); \
                 var d = Object.getOwnPropertyDescriptor(o, 'x'); \
                 d.value + ',' + d.writable + ',' + d.enumerable + ',' + d.configurable"
            ),
            "3,false,true,true"
        );
    }

    #[test]
    fn define_property_partial_writable_preserves_value() {
        assert_eq!(
            eval_str(
                "var o = {}; \
                 Object.defineProperty(o, 'x', {value: 7}); \
                 Object.defineProperty(o, 'x', {writable: false}); \
                 var d = Object.getOwnPropertyDescriptor(o, 'x'); \
                 d.value + ',' + d.writable + ',' + d.enumerable + ',' + d.configurable"
            ),
            "7,false,false,false"
        );
    }

    #[test]
    fn define_property_data_to_accessor_transition() {
        // Data -> accessor: get/set reset to defaults, but
        // enumerable/configurable carry over from the data property.
        assert_eq!(
            eval_str(
                "var o = {}; \
                 Object.defineProperty(o, 'x', {value: 5, enumerable: false, configurable: true}); \
                 Object.defineProperty(o, 'x', {get: function() { return 9; }}); \
                 var d = Object.getOwnPropertyDescriptor(o, 'x'); \
                 o.x + ',' + typeof d.get + ',' + (d.value === undefined) + ',' \
                     + d.enumerable + ',' + d.configurable"
            ),
            "9,function,true,false,true"
        );
    }

    #[test]
    fn define_property_accessor_to_data_transition() {
        // Accessor -> data: value/writable reset to spec defaults, but
        // enumerable/configurable carry over from the accessor.
        assert_eq!(
            eval_str(
                "var o = {}; \
                 Object.defineProperty(o, 'x', {get: function() { return 1; }, enumerable: false, configurable: true}); \
                 Object.defineProperty(o, 'x', {value: 7}); \
                 var d = Object.getOwnPropertyDescriptor(o, 'x'); \
                 d.value + ',' + d.writable + ',' + d.enumerable + ',' + d.configurable \
                     + ',' + (d.get === undefined)"
            ),
            "7,false,false,true,true"
        );
    }

    #[test]
    fn define_property_partial_keeps_non_configurable_validation() {
        // Absent fields merge to current values, so validation still sees
        // the real current state: a non-configurable, non-writable property
        // cannot change value.
        assert_eq!(
            eval_str(
                "var o = {}; \
                 Object.defineProperty(o, 'x', {value: 1, writable: false, configurable: false}); \
                 var r = 'unset'; \
                 try { Object.defineProperty(o, 'x', {value: 2}); r = 'no-throw'; } \
                 catch (e) { r = 'threw:' + o.x; } \
                 r;"
            ),
            "threw:1"
        );
        // writable true -> false is still allowed on non-configurable props.
        assert_eq!(
            eval_str(
                "var o = {}; \
                 Object.defineProperty(o, 'x', {value: 1, configurable: false}); \
                 Object.defineProperty(o, 'x', {writable: false}); \
                 var d = Object.getOwnPropertyDescriptor(o, 'x'); \
                 d.value + ',' + d.writable + ',' + d.configurable"
            ),
            "1,false,false"
        );
    }

    // ------------------------------------------------------------------
    // Array exotic [[DefineOwnProperty]] (spec 10.4.2)
    // ------------------------------------------------------------------

    /// Evaluate `source`, expecting it to throw; returns the displayed
    /// error (e.g. `"uncaught: RangeError: invalid array length"`).
    fn eval_throws(source: &str) -> String {
        let mut vm = crate::vm::Vm::new();
        match vm.eval(source) {
            Ok(v) => panic!(
                "expected throw, got {}",
                to_string_js(vm.heap_mut(), v).unwrap()
            ),
            Err(e) => format!("{e}"),
        }
    }

    #[test]
    fn array_define_property_index_extends_length() {
        // Spec 10.4.2.1: an indexed define at/past the end extends
        // `length` to `index + 1`.
        assert_eq!(
            eval_str(
                "var a = [1, 2]; \
                 Object.defineProperty(a, '5', {value: 9}); \
                 a.length + '/' + a[5];"
            ),
            "6/9"
        );
        // Within bounds: length untouched, value written.
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3]; \
                 Object.defineProperty(a, '1', {value: 9}); \
                 a.length + '/' + a[1];"
            ),
            "3/9"
        );
        // A partial descriptor keeps the spec default attributes
        // (all false for `Object.defineProperty`).
        assert_eq!(
            eval_str(
                "var a = []; \
                 Object.defineProperty(a, '0', {value: 7}); \
                 var d = Object.getOwnPropertyDescriptor(a, '0'); \
                 a.length + '/' + d.writable + '/' + d.enumerable + '/' + d.configurable;"
            ),
            "1/false/false/false"
        );
    }

    #[test]
    fn array_index_assignment_extends_length() {
        // `a[5] = 9` on a length-2 array yields length 6.
        assert_eq!(eval_str("var a = [1, 2]; a[5] = 9; a.length;"), "6");
        assert_eq!(
            eval_str(
                "var a = [1, 2]; a[5] = 9; \
                 a[5] + '/' + a.hasOwnProperty('5') + '/' + a.hasOwnProperty('2');"
            ),
            "9/true/false"
        );
        assert_eq!(eval_str("var a = []; a[3] = 'x'; a.length;"), "4");
    }

    #[test]
    fn array_length_shrink_deletes_trailing() {
        // Shrink deletes trailing indexed properties (sparse array).
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3, 4]; \
                 a.length = 2; \
                 a.length + '/' + a.join(',') + '/' + a.hasOwnProperty('2') + '/' + a.hasOwnProperty('1');"
            ),
            "2/1,2/false/true"
        );
        // Same through defineProperty.
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3, 4]; \
                 Object.defineProperty(a, 'length', {value: 2}); \
                 a.length + '/' + a.join(',');"
            ),
            "2/1,2"
        );
        // Dense array shrink truncates coherently.
        assert_eq!(
            eval_str("var a = [1, 2, 3, 4]; a.length = 2; a.join(',') + '/' + a.length;"),
            "1,2/2"
        );
        // Growing only changes length (index 3 stays a hole).
        assert_eq!(
            eval_str(
                "var a = [1]; a.length = 4; \
                 a.length + '/' + a.hasOwnProperty('0') + '/' + a.hasOwnProperty('3') + '/' + a.join(',');"
            ),
            "4/true/false/1,,,"
        );
    }

    #[test]
    fn array_length_shrink_non_configurable_restores() {
        // A non-configurable trailing element stops the shrink: length
        // is restored to failed_index + 1 and the throwing defineProperty
        // path reports TypeError.
        let err = eval_throws(
            "var a = [1, 2, 3]; \
             Object.defineProperty(a, '2', {configurable: false}); \
             Object.defineProperty(a, 'length', {value: 1});",
        );
        assert!(err.contains("TypeError"), "got: {err}");
        // Length was restored past the failed delete; the other elements
        // survive.
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3]; \
                 Object.defineProperty(a, '2', {configurable: false}); \
                 try { Object.defineProperty(a, 'length', {value: 1}); } catch (e) {} \
                 a.length + '/' + a.join(',');"
            ),
            "3/1,2,3"
        );
        // Sloppy assignment: the failed set is silently ignored, length
        // stays restored.
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3]; \
                 Object.defineProperty(a, '2', {configurable: false}); \
                 a.length = 1; \
                 a.length;"
            ),
            "3"
        );
        // Strict assignment throws.
        let err = eval_throws(
            "'use strict'; \
             var a = [1, 2, 3]; \
             Object.defineProperty(a, '2', {configurable: false}); \
             a.length = 1;",
        );
        assert!(err.contains("TypeError"), "got: {err}");
    }

    #[test]
    fn array_non_writable_length_blocks_extension() {
        // defineProperty past the end with non-writable length throws,
        // and the index is not created (spec 10.4.2.1 step 2.f).
        let err = eval_throws(
            "var a = [1, 2]; \
             Object.defineProperty(a, 'length', {writable: false}); \
             Object.defineProperty(a, '5', {value: 9});",
        );
        assert!(err.contains("TypeError"), "got: {err}");
        assert_eq!(
            eval_str(
                "var a = [1, 2]; \
                 Object.defineProperty(a, 'length', {writable: false}); \
                 try { Object.defineProperty(a, '5', {value: 9}); } catch (e) {} \
                 a.length + '/' + a.hasOwnProperty('5');"
            ),
            "2/false"
        );
        // Strict assignment throws; sloppy ignores.
        let err = eval_throws(
            "'use strict'; \
             var a = [1, 2]; \
             Object.defineProperty(a, 'length', {writable: false}); \
             a[5] = 9;",
        );
        assert!(err.contains("TypeError"), "got: {err}");
        assert_eq!(
            eval_str(
                "var a = [1, 2]; \
                 Object.defineProperty(a, 'length', {writable: false}); \
                 a[5] = 9; \
                 a.length + '/' + a.hasOwnProperty('5');"
            ),
            "2/false"
        );
        // Shrinking a non-writable length is rejected too.
        let err = eval_throws(
            "'use strict'; \
             var a = [1, 2, 3]; \
             Object.defineProperty(a, 'length', {writable: false}); \
             a.length = 1;",
        );
        assert!(err.contains("TypeError"), "got: {err}");
    }

    #[test]
    fn array_invalid_lengths_throw_range_error() {
        for src in [
            "var a = []; a.length = -1;",
            "var a = []; a.length = 4294967296;",
            "var a = []; a.length = 1.5;",
            "var a = []; Object.defineProperty(a, 'length', {value: -1});",
            "var a = []; Object.defineProperty(a, 'length', {value: 4294967296});",
            "var a = []; Object.defineProperty(a, 'length', {value: 1.5});",
        ] {
            let err = eval_throws(src);
            assert!(err.contains("RangeError"), "{src} got: {err}");
        }
        // 2^32 - 1 is a valid length.
        assert_eq!(
            eval_str("var a = []; a.length = 4294967295; a.length;"),
            "4294967295"
        );
        // A rejected length leaves the old length alone.
        assert_eq!(
            eval_str(
                "var a = [1, 2]; \
                 try { a.length = -1; } catch (e) {} \
                 a.length + '/' + a.join(',');"
            ),
            "2/1,2"
        );
    }

    #[test]
    fn splice_two_arg_form_honors_delete_count() {
        // `splice(start, deleteCount)` with exactly two arguments must use
        // the delete count; only the one-argument form deletes to the end.
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3, 4]; var d = a.splice(1, 1); \
                 a.join(',') + '|' + d.join(',')"
            ),
            "1,3,4|2"
        );
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3, 4]; var d = a.splice(1); \
                 a.join(',') + '|' + d.join(',')"
            ),
            "1|2,3,4"
        );
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3, 4]; var d = a.splice(1, 2, 'x', 'y'); \
                 a.join(',') + '|' + d.join(',')"
            ),
            "1,x,y,4|2,3"
        );
    }

    #[test]
    fn to_spliced_returns_new_array_without_mutating() {
        // Basic: new array has the splice applied; the receiver is untouched.
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3, 4]; var b = a.toSpliced(1, 2, 'x'); \
                 a.join(',') + '|' + b.join(',')"
            ),
            "1,2,3,4|1,x,4"
        );
        // Negative start counts from the end.
        assert_eq!(eval_str("[1, 2, 3, 4].toSpliced(-2, 1).join(',')"), "1,2,4");
        // `start` present, `deleteCount` omitted: delete to the end.
        assert_eq!(eval_str("[1, 2, 3, 4].toSpliced(1).join(',')"), "1");
        // `start` absent: no deletion — unlike `splice()`, this is a copy.
        assert_eq!(eval_str("[1, 2, 3].toSpliced().join(',')"), "1,2,3");
        // Insertion only.
        assert_eq!(
            eval_str("[1, 2].toSpliced(1, 0, 'a', 'b').join(',')"),
            "1,a,b,2"
        );
        // Holes are *filled* with present `undefined` in both copied
        // segments (spec: no `HasProperty` guard; real holes via
        // `delete`; literal elisions compile to present `undefined`).
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3, 4]; delete a[2]; var b = a.toSpliced(0, 1); b.length + '/' + (1 in b) + '/' + (2 in b)"
            ),
            "3/true/true"
        );
        // Observable spec case: a hole becomes a present `undefined`.
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3]; delete a[1]; var b = a.toSpliced(0, 0); \
                 (1 in b) + '/' + (b[1] === undefined) + '/' + b.join(',')"
            ),
            "true/true/1,,3"
        );
        // Result is a fresh array, not the receiver.
        assert_eq!(
            eval_str("var a = [1]; var b = a.toSpliced(0, 0); (a !== b) && (b instanceof Array)"),
            "true"
        );
        // `length` is settable on the result and reflects the new length.
        assert_eq!(eval_str("[1, 2, 3].toSpliced(0, 3, 'x').length"), "1");
        // Non-mutating on array-likes via the prototype.
        assert_eq!(
            eval_str(
                "var o = { 0: 'a', 1: 'b', length: 2 }; Array.prototype.toSpliced.call(o, 0, 1).join(',')"
            ),
            "b"
        );
        // Prototype-chain index properties are read through `Get`
        // (the receiver's own `length` gates the range; the engine
        // reads `length` own-only, a separate v1 gap).
        assert_eq!(
            eval_str(
                "var o = Object.create({ 0: 'x' }); o.length = 1; \
                 Array.prototype.toSpliced.call(o, 0, 0).join(',')"
            ),
            "x"
        );
        // Index getters fire during the copy.
        assert_eq!(
            eval_str(
                "var o = { length: 2 }; var fired = []; \
                 Object.defineProperty(o, '1', { get: function () { fired.push(1); return 'g'; } }); \
                 var b = Array.prototype.toSpliced.call(o, 0, 0); \
                 b.join(',') + '|' + fired.join(',')"
            ),
            ",g|1"
        );
    }

    #[test]
    fn to_spliced_huge_length_is_sparse_not_fatal() {
        // `length: 2**32 - 1` must not abort the process in the
        // allocator: the result is built sparse (length set, holes
        // elsewhere) instead of materializing ~4.3B properties.
        assert_eq!(
            eval_str("Array.prototype.toSpliced.call({ length: 4294967295 }, 0, 0).length"),
            "4294967295"
        );
        // The source's own copied index properties survive on the sparse
        // path.
        assert_eq!(
            eval_str(
                "var o = { 0: 'a', length: 4294967295 }; \
                 var b = Array.prototype.toSpliced.call(o, 0, 0); \
                 b.length + '/' + b[0] + '/' + (1 in b)"
            ),
            "4294967295/a/false"
        );
        // Past 2**32 - 1 the result length itself is invalid: RangeError.
        assert_eq!(
            eval_str(
                "var r; try { Array.prototype.toSpliced.call({ length: 4294967295 }, 0, 0, 'x'); r = 'no-throw'; } \
                 catch (e) { r = e instanceof RangeError ? 'RangeError' : 'wrong'; } r"
            ),
            "RangeError"
        );
    }

    #[test]
    fn species_subclass_propagates() {
        // `get Array [ @@species ]` returns `this`, so subclass
        // instances build subclass instances through `ArraySpeciesCreate`.
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 var r = new B(1, 2, 3).map(function (x) { return x * 2; }); \
                 (r instanceof B) + '/' + (r instanceof Array) + '/' + r.join(',')"
            ),
            "true/true/2,4,6"
        );
        // Plain arrays still produce plain Arrays.
        assert_eq!(
            eval_str(
                "var r = [1, 2].map(function (x) { return x; }); \
                 (r instanceof Array) + '/' + (r.constructor === Array)"
            ),
            "true/true"
        );
    }

    #[test]
    fn species_propagates_across_methods() {
        // `toSorted`/`toReversed`/`toSpliced` are deliberately absent:
        // per ECMA-262 they use plain `ArrayCreate`, not
        // `ArraySpeciesCreate` (see `copy_methods_use_array_create`).
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 [new B(1, 2, 3).filter(function (x) { return x > 1; }) instanceof B, \
                  new B(1, 2, 3).slice(1) instanceof B, \
                  new B(1, 2, 3).splice(0, 1) instanceof B, \
                  new B(1, 2, 3).concat([4]) instanceof B, \
                  new B(1, 2, 3).flat() instanceof B, \
                  new B(1, 2, 3).flatMap(function (x) { return [x]; }) instanceof B].join(',')"
            ),
            "true,true,true,true,true,true"
        );
    }

    #[test]
    fn copy_methods_use_array_create() {
        // Per ECMA-262 23.1.3.34/36/38 the "change Array by copy"
        // methods (`toSorted`/`toReversed`/`toSpliced`) use
        // `ArrayCreate(len)` — species-ignorant: the result is a plain
        // `Array` even for a subclass receiver.
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 var s = new B(3, 1, 2).toSorted(); \
                 (s instanceof Array) + '/' + (s instanceof B) + '/' + \
                 (s.constructor === Array) + '/' + s.join(',')"
            ),
            "true/false/true/1,2,3"
        );
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 var r = new B(1, 2, 3).toReversed(); \
                 (r instanceof Array) + '/' + (r instanceof B) + '/' + \
                 (r.constructor === Array) + '/' + r.join(',')"
            ),
            "true/false/true/3,2,1"
        );
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 var p = new B(1, 2, 3).toSpliced(1, 1, 9); \
                 (p instanceof Array) + '/' + (p instanceof B) + '/' + \
                 (p.constructor === Array) + '/' + p.join(',')"
            ),
            "true/false/true/1,9,3"
        );
        // Guard against over-reverting: the species methods must still
        // build subclass instances.
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 var m = new B(1, 2, 3).map(function (x) { return x * 2; }); \
                 (m instanceof B) + '/' + (m.constructor === B) + '/' + m.join(',')"
            ),
            "true/true/2,4,6"
        );
    }

    #[test]
    fn species_hostile_throws_typeerror() {
        // A non-constructor @@species throws TypeError instead of being
        // silently "constructed".
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 Object.defineProperty(B, Symbol.species, { value: 42 }); \
                 var r; try { new B(1).map(function (x) { return x; }); r = 'no-throw'; } \
                 catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
        // A throwing @@species getter propagates its own error.
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 Object.defineProperty(B, Symbol.species, \
                   { get: function () { throw new Error('boom'); } }); \
                 var r; try { new B(1).filter(function (x) { return true; }); r = 'no-throw'; } \
                 catch (e) { r = e instanceof Error && e.message === 'boom' ? 'boom' : 'wrong'; } r"
            ),
            "boom"
        );
        // null @@species falls back to %Array%: a plain Array result.
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 Object.defineProperty(B, Symbol.species, { value: null }); \
                 var r = new B(1, 2).map(function (x) { return x * 3; }); \
                 (r instanceof Array) + '/' + (r instanceof B) + '/' + r.join(',')"
            ),
            "true/false/3,6"
        );
    }

    #[test]
    fn species_non_array_result_still_populates() {
        // A @@species constructor returning a non-array object: indices
        // are still populated via `CreateDataPropertyOrThrow`, but spec
        // creates no `length` on a hostile non-Array result — only
        // `concat` performs a final `Set(A, "length", …)`.
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 class C { constructor(n) { this.n = n; } } \
                 Object.defineProperty(B, Symbol.species, { value: C }); \
                 var r = new B(1, 2).map(function (x) { return x * 2; }); \
                 (r instanceof Array) + '/' + (r instanceof C) + '/' + r.n + '/' + \
                 r[0] + '/' + r[1] + '/' + ('length' in r) + '/' + r.length"
            ),
            "false/true/2/2/4/false/undefined"
        );
        // Same for `slice`: no `length` is forced onto a hostile
        // non-Array result.
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 class C { constructor(n) { this.n = n; } } \
                 Object.defineProperty(B, Symbol.species, { value: C }); \
                 var r = new B(1, 2, 3).slice(1); \
                 r[0] + '/' + r[1] + '/' + ('length' in r)"
            ),
            "2/3/false"
        );
    }

    /// Install a `gc()` global that forces a collection, for tests that
    /// must exercise GC hazards the (disabled-by-default) collector
    /// would otherwise never trigger.
    fn install_gc(vm: &mut crate::vm::Vm) {
        fn gc_collect(
            heap: &mut Heap,
            _this: Value,
            _args: &[Value],
        ) -> Result<Value, NativeError> {
            heap.collect();
            Ok(Value::Undefined)
        }
        let f = new_native_function(vm.heap_mut(), "gc", gc_collect, 0).unwrap();
        vm.set_global("gc", Value::Object(f));
    }

    /// Evaluate JS in a full VM with the `gc()` hook installed and
    /// automatic collections enabled, and stringify the completion
    /// value. Unit tests run with GC disabled by default, so hazards
    /// that need a mid-eval collection are invisible without this.
    fn eval_str_gc(source: &str) -> String {
        let mut vm = crate::vm::Vm::new();
        install_gc(&mut vm);
        // Automatic collections enabled: mid-eval allocations can
        // collect, so unrooted handles are swept instead of surviving
        // by luck of the disabled collector.
        vm.heap_mut().set_gc_threshold(1);
        let v = vm
            .eval(source)
            .unwrap_or_else(|e| panic!("eval failed: {e}"));
        to_string_js(vm.heap_mut(), v)
            .map(|w| w.to_string())
            .unwrap_or_else(|e| panic!("to_string failed: {e:?}"))
    }

    #[test]
    fn species_proxy_construct_trap_survives_forced_gc() {
        // The @@species getter's fresh Proxy lived only in a Rust local
        // across `reflect_construct_inner`; the proxy-with-`construct`-
        // trap path allocates (`new_value_array`) before the trap runs,
        // and a collection there freed the unrooted proxy, so the trap
        // call used dangling handles. With collections enabled the
        // allocation inside `proxy_construct` collects deterministically;
        // the temp-rooted target must survive it. (`marker` proves the
        // proxy path ran at all: without the species override the
        // result would be a plain `B` with no marker.)
        assert_eq!(
            eval_str_gc(
                "class B extends Array { \
                   static get [Symbol.species]() { \
                     return new Proxy(class extends Array {}, { \
                       construct: function (t, a) { \
                         var r = new t(...a); r.marker = 'trap-ran'; return r; \
                       } \
                     }); \
                   } \
                 } \
                 var r = new B(1, 2, 3).map(function (x) { return x * 2; }); \
                 (r instanceof Array) + '/' + r.join(',') + '/' + r.marker"
            ),
            "true/2,4,6/trap-ran"
        );
    }

    #[test]
    fn species_create_data_property_or_throw_hostile() {
        // Existing non-configurable, non-writable index on a hostile
        // result: `CreateDataPropertyOrThrow` throws `TypeError`
        // (previously the write was silently skipped).
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 class C { constructor(n) { \
                   Object.defineProperty(this, '0', \
                     { value: 99, writable: false, configurable: false }); } } \
                 Object.defineProperty(B, Symbol.species, { value: C }); \
                 var r; try { new B(1, 2).map(function (x) { return x * 2; }); r = 'no-throw'; } \
                 catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
        // Non-extensible hostile result: `TypeError`, not an
        // out-of-memory misreport.
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 class C { constructor(n) { Object.preventExtensions(this); } } \
                 Object.defineProperty(B, Symbol.species, { value: C }); \
                 var r; try { new B(1, 2).map(function (x) { return x * 2; }); r = 'no-throw'; } \
                 catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong: ' + e; } r"
            ),
            "TypeError"
        );
        // Existing non-configurable accessor: not clobbered, `TypeError`.
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 class C { constructor(n) { \
                   Object.defineProperty(this, '0', \
                     { get: function () { return 42; }, configurable: false }); } } \
                 Object.defineProperty(B, Symbol.species, { value: C }); \
                 var r; try { new B(1, 2).map(function (x) { return x * 2; }); r = 'no-throw'; } \
                 catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
        // Proxy species result: index writes go through the
        // `defineProperty` trap instead of landing on the proxy object.
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 var seen = []; \
                 var p = new Proxy({}, { \
                   defineProperty: function (t, k, d) { seen.push(k + '=' + d.value); return true; } \
                 }); \
                 class C { constructor(n) { return p; } } \
                 Object.defineProperty(B, Symbol.species, { value: C }); \
                 var r = new B(1, 2).map(function (x) { return x * 2; }); \
                 (r === p) + '/' + seen.join(',')"
            ),
            "true/0=2,1=4"
        );
        // A `defineProperty` trap returning falsy becomes `TypeError`.
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 var p = new Proxy({}, { \
                   defineProperty: function (t, k, d) { return false; } \
                 }); \
                 class C { constructor(n) { return p; } } \
                 Object.defineProperty(B, Symbol.species, { value: C }); \
                 var r; try { new B(1, 2).map(function (x) { return x * 2; }); r = 'no-throw'; } \
                 catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
    }

    #[test]
    fn species_length_reflects_actual_writes() {
        // Hostile Array-kind constructor ignoring the length argument:
        // `length` reflects the actual writes (highest index + 1), not a
        // forced `len`. A trailing hole in the source stays a hole, so
        // `length` is 1, not 3.
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 class D extends Array { constructor(n) { super(); } } \
                 Object.defineProperty(B, Symbol.species, { value: D }); \
                 var b = new B(3); b[0] = 5; \
                 var r = b.map(function (x) { return x * 2; }); \
                 (r instanceof D) + '/' + r.length + '/' + r[0] + '/' + (1 in r)"
            ),
            "true/1/10/false"
        );
    }

    #[test]
    fn concat_final_length_write_is_set() {
        // ES2024 §23.1.3.2 step 6 is `Set(A, "length", …)`: an inherited
        // `length` setter fires on a hostile result instead of being
        // shadowed by a defined data property.
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 var fired = []; \
                 class C { constructor(n) {} } \
                 Object.defineProperty(C.prototype, 'length', \
                   { set: function (v) { fired.push(v); }, configurable: true }); \
                 Object.defineProperty(B, Symbol.species, { value: C }); \
                 var r = new B(1, 2).concat([3]); \
                 (r instanceof C) + '/' + fired.join(',')"
            ),
            "true/3"
        );
        // Proxy species result: the final length write goes through the
        // `set` trap (index writes still use `defineProperty`).
        assert_eq!(
            eval_str(
                "class B extends Array {}; \
                 var log = []; \
                 var p = new Proxy({}, { \
                   set: function (t, k, v, r) { log.push(k + '=' + v); return true; } \
                 }); \
                 class C { constructor(n) { return p; } } \
                 Object.defineProperty(B, Symbol.species, { value: C }); \
                 var r = new B(1, 2).concat([3]); \
                 (r === p) + '/' + log.join(',')"
            ),
            "true/length=3"
        );
        // The default path is unchanged: `Set` on a real array is the
        // exotic `ArraySetLength`, a no-op when `length` already matches.
        assert_eq!(
            eval_str("var r = [1, 2].concat([3, 4]); r.length + '/' + r.join(',')"),
            "4/1,2,3,4"
        );
    }

    #[test]
    fn species_sparse_value_correctness() {
        // Real holes via `new Array(n)` + sparse assignment: array
        // literal elisions (`[1,,3]`) are present-`undefined` in this
        // engine, a v1 gap, so they cannot express holes.
        // map: holes stay holes, values map.
        assert_eq!(
            eval_str(
                "var a = new Array(3); a[0] = 1; a[2] = 3; \
                 var r = a.map(function (x) { return x * 2; }); \
                 r.length + '/' + (1 in r) + '/' + r[0] + '/' + r[2]"
            ),
            "3/false/2/6"
        );
        // filter: holes are skipped, kept values compact.
        assert_eq!(
            eval_str(
                "var a = new Array(4); a[0] = 1; a[2] = 3; a[3] = 4; \
                 var r = a.filter(function (x) { return x > 1; }); \
                 r.length + '/' + r.join(',')"
            ),
            "2/3,4"
        );
        // slice: holes preserved by position.
        assert_eq!(
            eval_str(
                "var a = new Array(4); a[0] = 1; a[2] = 3; a[3] = 4; \
                 var r = a.slice(1); \
                 r.length + '/' + (0 in r) + '/' + r[1] + '/' + r[2]"
            ),
            "3/false/3/4"
        );
        // splice: the deleted array fills holes with present `undefined`
        // (spec: `Get` + `CreateDataPropertyOrThrow`, no `HasProperty`).
        assert_eq!(
            eval_str(
                "var a = new Array(4); a[0] = 1; a[2] = 3; a[3] = 4; \
                 var d = a.splice(1, 2); \
                 d.length + '/' + (0 in d) + '/' + (d[0] === undefined) + '/' + d[1] + \
                 '/' + a.join(',')"
            ),
            "2/true/true/3/1,4"
        );
        // concat: holes preserved across segments.
        assert_eq!(
            eval_str(
                "var a = new Array(3); a[0] = 1; a[2] = 2; \
                 var b = new Array(3); b[0] = 3; b[2] = 4; \
                 var r = a.concat(b); \
                 r.length + '/' + (1 in r) + '/' + (4 in r) + '/' + r[0] + '/' + r[5]"
            ),
            "6/false/false/1/4"
        );
        // flat: holes are dropped.
        assert_eq!(
            eval_str(
                "var inner = new Array(3); inner[0] = 2; inner[2] = 4; \
                 var a = [1, inner]; \
                 var r = a.flat(); r.length + '/' + r.join(',')"
            ),
            "3/1,2,4"
        );
        // flatMap: holes skipped, mapped arrays flattened one level.
        assert_eq!(
            eval_str(
                "var a = new Array(3); a[1] = 1; a[2] = 2; \
                 var r = a.flatMap(function (x) { return [x, x * 10]; }); \
                 r.length + '/' + r.join(',')"
            ),
            "4/1,10,2,20"
        );
        // toSorted: holes stay holes at the tail.
        assert_eq!(
            eval_str(
                "var a = new Array(3); a[0] = 3; a[2] = 1; \
                 var r = a.toSorted(); \
                 r.length + '/' + (2 in r) + '/' + r[0] + '/' + r[1]"
            ),
            "3/false/1/3"
        );
        // toReversed: holes preserved by mirrored position.
        assert_eq!(
            eval_str(
                "var a = new Array(3); a[0] = 1; a[2] = 3; \
                 var r = a.toReversed(); \
                 r.length + '/' + (1 in r) + '/' + r[0] + '/' + r[2]"
            ),
            "3/false/3/1"
        );
        // toSpliced (dense path): holes are read as `undefined` and
        // created present.
        assert_eq!(
            eval_str(
                "var a = new Array(3); a[0] = 1; a[2] = 3; \
                 var r = a.toSpliced(0, 0); \
                 r.length + '/' + (1 in r) + '/' + (r[1] === undefined)"
            ),
            "3/true/true"
        );
    }

    #[test]
    fn array_receiver_boxes_primitives() {
        // `ToObject` on the receiver: strings box with indexed
        // characters and `length`.
        assert_eq!(
            eval_str("Array.prototype.toSpliced.call('ab', 0, 1).join(',')"),
            "b"
        );
        assert_eq!(eval_str("Array.prototype.join.call('ab', '-')"), "a-b");
        assert_eq!(eval_str("Array.prototype.keys.call('ab').join(',')"), "0,1");
        assert_eq!(
            eval_str("Array.prototype.map.call('ab', function(c) { return c + '!'; }).join(',')"),
            "a!,b!"
        );
        // Numbers/booleans box to an empty object: no `length`, so the
        // methods see an empty array-like.
        assert_eq!(eval_str("Array.prototype.push.call(5, 1)"), "1");
        assert_eq!(eval_str("Array.prototype.join.call(true, '-')"), "");
        // `null`/`undefined` still throw.
        assert_eq!(
            eval_str(
                "var r; try { Array.prototype.keys.call(null); r = 'no-throw'; } \
                 catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
    }

    #[test]
    fn splice_deleted_array_fills_holes() {
        // Spec 23.1.3.30: the deleted segment is copied with `Get` +
        // `CreateDataPropertyOrThrow` and no `HasProperty` guard, so
        // holes surface as *present* `undefined`.
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3]; delete a[1]; \
                 var d = a.splice(0, 3); \
                 (0 in d) + '/' + (1 in d) + '/' + (2 in d) + '/' + d[1]"
            ),
            "true/true/true/undefined"
        );
    }

    #[test]
    fn keys_values_entries_with_huge_length_are_sparse_not_fatal() {
        // `length: 2**32 - 1` must not abort the process in the
        // allocator (neither `with_capacity(len)` nor amortized `push`
        // growth): the result is built sparse — `length` set, only the
        // source's own index properties materialized.
        assert_eq!(
            eval_str("Array.prototype.keys.call({ length: 4294967295 }).length"),
            "4294967295"
        );
        assert_eq!(
            eval_str(
                "var o = { 0: 'x', length: 4294967295 }; \
                 var k = Array.prototype.keys.call(o); \
                 k.length + '/' + k[0] + '/' + (1 in k)"
            ),
            "4294967295/0/false"
        );
        assert_eq!(
            eval_str(
                "var o = { 1: 'b', length: 4294967295 }; \
                 var v = Array.prototype.values.call(o); \
                 v.length + '/' + v[1] + '/' + (0 in v)"
            ),
            "4294967295/b/false"
        );
        assert_eq!(
            eval_str(
                "var o = { 2: 'c', length: 4294967295 }; \
                 var e = Array.prototype.entries.call(o); \
                 e.length + '/' + e[2].join(',') + '/' + (0 in e)"
            ),
            "4294967295/2,c/false"
        );
        assert_eq!(
            eval_str(
                "var o = { 1: 'b', length: 4294967295 }; \
                 var w = Array.prototype.with.call(o, 5, 'z'); \
                 w.length + '/' + w[1] + '/' + w[5] + '/' + (0 in w)"
            ),
            "4294967295/b/z/false"
        );
        // Past 2**32 - 1 the ArrayCreate length is invalid: RangeError.
        assert_eq!(
            eval_str(
                "var r; try { Array.prototype.values.call({ length: 4294967296 }); r = 'no-throw'; } \
                 catch (e) { r = e instanceof RangeError ? 'RangeError' : 'wrong'; } r"
            ),
            "RangeError"
        );
    }

    #[test]
    fn splice_huge_delete_is_sparse_not_fatal() {
        assert_eq!(
            eval_str(
                "var o = { 0: 'a', length: 4294967295 }; \
                 var d = Array.prototype.splice.call(o, 0, 4294967295); \
                 d.length + '/' + d[0] + '/' + (1 in d) + '/' + o.length"
            ),
            "4294967295/a/false/0"
        );
    }

    #[test]
    fn splice_sparse_shift_down_and_up_match_dense() {
        // Shift-down past the dense limit: only the source's own index
        // properties move; holes delete their destinations and the tail
        // is deleted, exactly like the dense loop.
        assert_eq!(
            eval_str(
                "var o = { 0: 'a', 5: 'e', length: 2097152 }; \
                 var d = Array.prototype.splice.call(o, 0, 3); \
                 [o.length, o[2], (0 in o), (5 in o), d.length, d[0], d[1], (1 in d)].join('/')"
            ),
            "2097149/e/false/false/3/a//true"
        );
        // Shift-up past the dense limit: own properties move up,
        // destinations with hole sources are deleted, inserted items
        // land at `actual_start`.
        assert_eq!(
            eval_str(
                "var o = { 0: 'a', 2097151: 'z', length: 2097152 }; \
                 var d = Array.prototype.splice.call(o, 0, 0, 'x'); \
                 [o.length, o[0], o[1], o[2097152], (2097151 in o), d.length].join('/')"
            ),
            "2097153/x/a/z/false/0"
        );
    }

    #[test]
    fn find_last_index_scans_from_end() {
        assert_eq!(
            eval_str("[1, 2, 3, 2, 1].findLastIndex((x) => x === 2)"),
            "3"
        );
        assert_eq!(eval_str("[1, 2, 3].findLastIndex((x) => x === 9)"), "-1");
        // Sparse arrays: spec 23.1.3.11 has no `HasProperty` guard, so the
        // predicate runs for holes too, with `undefined` as the value
        // (real holes via `delete`; literal elisions compile to present
        // `undefined`).
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3]; delete a[1]; var seen = []; \
                 a.findLastIndex((v, i) => { seen.push(i); return false; }); \
                 seen.join(',')"
            ),
            "2,1,0"
        );
        // The value passed for a hole is `undefined`.
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3]; delete a[1]; \
                 a.findLastIndex((v, i) => i === 1 && v === undefined)"
            ),
            "1"
        );
        // Inherited index properties are visible through `Get`.
        assert_eq!(
            eval_str(
                "var o = Object.create({ 1: 'p' }); o.length = 3; \
                 Array.prototype.findLastIndex.call(o, (v) => v === 'p')"
            ),
            "1"
        );
        // Callback receives (value, index, array).
        assert_eq!(
            eval_str(
                "var got = []; ['a', 'b'].findLastIndex((v, i, a) => { got.push(v + i + a.length); return v === 'b'; }); \
                 got.join(',')"
            ),
            "b12"
        );
        // `thisArg` is honored.
        assert_eq!(
            eval_str("[1, 2, 3].findLastIndex(function (x) { return x === this.n; }, { n: 3 })"),
            "2"
        );
        // Non-callable callback throws — unconditionally, even on an empty
        // or all-hole array (spec step 3 precedes the iteration).
        assert_eq!(
            eval_str(
                "var r; try { [1].findLastIndex(42); r = 'no-throw'; } catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
        assert_eq!(
            eval_str(
                "var r; try { [].findLastIndex(42); r = 'no-throw'; } catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
        assert_eq!(
            eval_str(
                "var a = [1]; delete a[0]; var r; try { a.findLastIndex(null); r = 'no-throw'; } catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
    }

    #[test]
    fn find_visits_holes_with_undefined() {
        // `find`: no `HasProperty` guard — every index is visited.
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3]; delete a[1]; var seen = []; \
                 a.find((v, i) => { seen.push(i); return false; }); \
                 seen.join(',')"
            ),
            "0,1,2"
        );
        // A hole matches an `undefined`-seeking predicate.
        assert_eq!(
            eval_str("var a = [1, 2, 3]; delete a[1]; a.find((v, i) => i === 1) === undefined"),
            "true"
        );
        // Non-callable callback throws on empty and all-hole arrays.
        assert_eq!(
            eval_str(
                "var r; try { [].find({}); r = 'no-throw'; } catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
        assert_eq!(
            eval_str(
                "var a = new Array(1); var r; try { a.find(42); r = 'no-throw'; } catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
    }

    #[test]
    fn find_index_visits_holes_with_undefined() {
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3]; delete a[1]; var seen = []; \
                 a.findIndex((v, i) => { seen.push(i); return false; }); \
                 seen.join(',')"
            ),
            "0,1,2"
        );
        assert_eq!(
            eval_str("var a = [1, 2, 3]; delete a[1]; a.findIndex((v, i) => i === 1)"),
            "1"
        );
        assert_eq!(
            eval_str(
                "var r; try { [].findIndex('x'); r = 'no-throw'; } catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
    }

    #[test]
    fn find_last_visits_holes_with_undefined() {
        // `findLast`: no `HasProperty` guard — every index is visited
        // from the end, holes with `undefined`.
        assert_eq!(
            eval_str(
                "var a = [1, 2, 3]; delete a[1]; var seen = []; \
                 a.findLast((v, i) => { seen.push(i); return false; }); \
                 seen.join(',')"
            ),
            "2,1,0"
        );
        assert_eq!(
            eval_str("var a = [1, 2, 3]; delete a[1]; a.findLast((v, i) => i === 1) === undefined"),
            "true"
        );
        // Inherited index properties are visible through `Get`.
        assert_eq!(
            eval_str(
                "var o = Object.create({ 0: 'p' }); o.length = 2; \
                 Array.prototype.findLast.call(o, (v) => v === 'p')"
            ),
            "p"
        );
        // Non-callable callback throws on empty and all-hole arrays.
        assert_eq!(
            eval_str(
                "var r; try { [].findLast(42); r = 'no-throw'; } catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
        assert_eq!(
            eval_str(
                "var a = new Array(1); var r; try { a.findLast(undefined); r = 'no-throw'; } catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
    }

    #[test]
    fn reverse_huge_length_fires_getters_in_order() {
        // test262: length-exceeding-integer-limit-with-object. The getter
        // at the top index must fire (and its throw propagate) on the
        // first swap iteration — never a multi-exabyte materialization.
        assert_eq!(
            eval_str(
                "function StopReverse() {} \
                 var o = { get '9007199254740990'() { throw new StopReverse(); }, \
                           length: 9007199254740992 }; \
                 var r; \
                 try { Array.prototype.reverse.call(o); r = 'no-throw'; } \
                 catch (e) { r = (e instanceof StopReverse) ? 'StopReverse' : 'wrong'; } \
                 r"
            ),
            "StopReverse"
        );
        // And spec Get (not the raw data slot) is what reverse reads:
        // a getter's value participates in the swap.
        assert_eq!(
            eval_str(
                "var log = []; \
                 var o = { get 0() { log.push('g0'); return 'a'; }, \
                           2: 'c', length: 3 }; \
                 Array.prototype.reverse.call(o); \
                 log.join(',') + '|' + o[0] + ',' + o[2]"
            ),
            "g0|c,a"
        );
    }

    #[test]
    fn object_prevent_extensions_marks_non_extensible() {
        // Spec 19.1.2.13: returns its argument, marked non-extensible.
        assert_eq!(
            eval_str(
                "var o = { a: 1 }; var r = Object.preventExtensions(o); \
                 (r === o) && !Reflect.isExtensible(o) && o.a === 1"
            ),
            "true"
        );
        // Non-object arguments pass through unchanged.
        assert_eq!(eval_str("Object.preventExtensions(1) === 1"), "true");
        assert_eq!(eval_str("Object.preventExtensions('s') === 's'"), "true");
        assert_eq!(
            eval_str("typeof Object.preventExtensions(null) === 'object'"),
            "true"
        );
        // Adding a property afterwards fails (defineProperty throws on a
        // non-extensible object even outside strict mode).
        assert_eq!(
            eval_str(
                "var o = {}; Object.preventExtensions(o); \
                 var r; try { Object.defineProperty(o, 'b', { value: 2 }); r = 'no-throw'; } \
                 catch (e) { r = e instanceof TypeError ? 'TypeError' : 'wrong'; } r"
            ),
            "TypeError"
        );
    }
}
