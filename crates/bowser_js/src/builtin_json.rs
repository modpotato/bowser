//! `JSON.parse` and `JSON.stringify`.
//!
//! - `parse` is a small recursive-descent parser over the heap value model
//!   (objects, arrays, strings, numbers via [`num`]). Depth is capped to
//!   keep hostile nesting from overflowing the Rust stack.
//! - `stringify` serializes values with string escaping, `NaN`/`Infinity`
//!   mapping to `null`, circular-structure detection (`TypeError`), and
//!   `toJSON` hooks (via [`reenter_call`).
//!
//! v1 gaps: `parse` takes no reviver; `stringify` takes no replacer and no
//! `space` argument. `toJSON` lookup is data-properties-only (no accessor
//! invocation), matching the rest of the builtins.

use crate::builtin_object_array::{
    array_length, get_own, is_callable, js_number_to_string, new_array, num, own_enumerable_keys,
    set_idx_raw, to_string_js,
};
use crate::builtin_string_number::{define_native, push_code_point};
use crate::gc::{Gc, Heap};
use crate::value::{NativeError, NativeFn, Object, ObjectKind, Property, Str, Value};
use crate::vm::reenter_call;
use crate::wtf8::{Wtf8, Wtf8Buf};

/// Install the `JSON` namespace object on `global`.
pub(crate) fn install(heap: &mut Heap, global: Gc<Object>) -> Result<(), NativeError> {
    let base = heap.protect(Value::Object(global));
    let json = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    heap.define_property_str(
        global,
        "JSON",
        Property {
            value: Value::Object(json),
            writable: true,
            enumerable: false,
            configurable: true,
        },
    )?;
    heap.release_temp(base);
    let jbase = heap.protect(Value::Object(json));
    let r = (|| {
        define_native(heap, json, "parse", json_parse as NativeFn, 1)?;
        define_native(heap, json, "stringify", json_stringify as NativeFn, 1)?;
        Ok(())
    })();
    heap.release_temp(jbase);
    r
}

fn get_arg(args: &[Value], i: usize) -> Value {
    args.get(i).copied().unwrap_or(Value::Undefined)
}

// ---------------------------------------------------------------------------
// JSON.parse
// ---------------------------------------------------------------------------

/// Maximum nesting depth for `JSON.parse` (hostile `[[[[...` input).
const MAX_PARSE_DEPTH: usize = 1000;

struct JsonParser<'a> {
    text: &'a [u8],
    pos: usize,
    heap: &'a mut Heap,
    depth: usize,
}

impl<'a> JsonParser<'a> {
    fn error(&self, msg: &str) -> NativeError {
        NativeError::SyntaxError(format!("JSON.parse: {msg} at index {}", self.pos))
    }

    fn skip_ws(&mut self) {
        while self.pos < self.text.len()
            && matches!(self.text[self.pos], b' ' | b'\t' | b'\n' | b'\r')
        {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.text.get(self.pos).copied()
    }

    fn expect_byte(&mut self, b: u8) -> Result<(), NativeError> {
        if self.peek() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.error(&format!("expected '{}'", b as char)))
        }
    }

    fn parse_value(&mut self) -> Result<Value, NativeError> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(Value::Str(self.parse_string()?)),
            Some(b't') => self.parse_literal("true", Value::Bool(true)),
            Some(b'f') => self.parse_literal("false", Value::Bool(false)),
            Some(b'n') => self.parse_literal("null", Value::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            _ => Err(self.error("unexpected token")),
        }
    }

    fn parse_literal(&mut self, lit: &str, v: Value) -> Result<Value, NativeError> {
        if self.text[self.pos..].starts_with(lit.as_bytes()) {
            self.pos += lit.len();
            Ok(v)
        } else {
            Err(self.error("unexpected token"))
        }
    }

    fn parse_object(&mut self) -> Result<Value, NativeError> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            return Err(self.error("too deeply nested"));
        }
        self.expect_byte(b'{')?;
        // Root the half-built object across key/value allocations below.
        let obj = self.heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
        let base = self.heap.protect(Value::Object(obj));
        let r = self.parse_object_inner(obj);
        self.heap.release_temp(base);
        self.depth -= 1;
        r?;
        Ok(Value::Object(obj))
    }

    fn parse_object_inner(&mut self, obj: Gc<Object>) -> Result<(), NativeError> {
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(());
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.error("expected string key"));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect_byte(b':')?;
            let val = self.parse_value()?;
            // Copy the key text out: the parsed key is a fresh, unrooted
            // handle and `define_property` (via interning) may collect.
            let key_str = key.as_wtf8().to_owned();
            self.heap
                .define_property(obj, key_str.as_wtf8(), Property::data(val))?;
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(());
                }
                _ => return Err(self.error("expected ',' or '}'")),
            }
        }
    }

    fn parse_array(&mut self) -> Result<Value, NativeError> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            return Err(self.error("too deeply nested"));
        }
        self.expect_byte(b'[')?;
        let arr = new_array(self.heap)?;
        let base = self.heap.protect(Value::Object(arr));
        let r = self.parse_array_inner(arr);
        self.heap.release_temp(base);
        self.depth -= 1;
        r?;
        Ok(Value::Object(arr))
    }

    fn parse_array_inner(&mut self, arr: Gc<Object>) -> Result<(), NativeError> {
        let mut len: u64 = 0;
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(());
        }
        loop {
            let val = self.parse_value()?;
            set_idx_raw(self.heap, arr, len, val)?;
            len += 1;
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(());
                }
                _ => return Err(self.error("expected ',' or ']'")),
            }
        }
    }

    fn parse_string(&mut self) -> Result<Str, NativeError> {
        self.expect_byte(b'"')?;
        let mut out = Wtf8Buf::new();
        loop {
            let c = self
                .peek()
                .ok_or_else(|| self.error("unterminated string"))?;
            self.pos += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self
                        .peek()
                        .ok_or_else(|| self.error("unterminated escape"))?;
                    self.pos += 1;
                    match e {
                        b'"' => out.push_char('"'),
                        b'\\' => out.push_char('\\'),
                        b'/' => out.push_char('/'),
                        b'b' => out.push_char('\u{8}'),
                        b'f' => out.push_char('\u{c}'),
                        b'n' => out.push_char('\n'),
                        b'r' => out.push_char('\r'),
                        b't' => out.push_char('\t'),
                        b'u' => {
                            let hi = self.parse_hex4()?;
                            // Surrogate pair. A high surrogate followed by
                            // `\u` + a low surrogate combines to the
                            // astral scalar; anything else (including a
                            // high surrogate followed by a non-low
                            // `\uXXXX`) is a lone surrogate, preserved
                            // exactly as WTF-8 per spec.
                            if (0xD800..0xDC00).contains(&hi)
                                && self.text.get(self.pos..self.pos + 2) == Some(b"\\u".as_slice())
                                && let Some(lo) = self.peek_hex4()
                                && (0xDC00..0xE000).contains(&lo)
                            {
                                self.pos += 6; // consume `\uXXXX`
                                let cp = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                                out.push_char(
                                    char::from_u32(cp)
                                        .ok_or_else(|| self.error("bad surrogate pair"))?,
                                );
                            } else {
                                out.push_unit(hi as u16);
                            }
                        }
                        _ => return Err(self.error("bad escape")),
                    }
                }
                0x00..=0x1F => return Err(self.error("unescaped control character")),
                _ => {
                    // WTF-8 sequence: find its full length from the lead
                    // byte (surrogate sequences are 3-byte `0xED` leads,
                    // handled like any other).
                    let len = if c < 0x80 {
                        1
                    } else if c < 0xE0 {
                        2
                    } else if c < 0xF0 {
                        3
                    } else {
                        4
                    };
                    let bytes = self
                        .text
                        .get(self.pos - 1..self.pos - 1 + len)
                        .ok_or_else(|| self.error("truncated sequence"))?;
                    // `text` is a `Wtf8Buf`'s bytes sliced at sequence
                    // boundaries, hence valid WTF-8.
                    out.push_wtf8_bytes(bytes);
                }
            }
        }
        Ok(self.heap.alloc_str_wtf8(out.as_wtf8())?)
    }

    /// Peek the 4 hex digits of a `\u` escape at the current position
    /// without consuming; `None` when they are not all hex.
    fn peek_hex4(&self) -> Option<u32> {
        let bytes = self.text.get(self.pos + 2..self.pos + 6)?;
        let s = std::str::from_utf8(bytes).ok()?;
        u32::from_str_radix(s, 16).ok()
    }

    fn parse_hex4(&mut self) -> Result<u32, NativeError> {
        let bytes = self
            .text
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| self.error("truncated \\u escape"))?;
        let s = std::str::from_utf8(bytes).map_err(|_| self.error("bad \\u escape"))?;
        self.pos += 4;
        u32::from_str_radix(s, 16).map_err(|_| self.error("bad \\u escape"))
    }

    fn parse_number(&mut self) -> Result<Value, NativeError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
            }
            Some(c) if c.is_ascii_digit() => {
                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.pos += 1;
                }
            }
            _ => return Err(self.error("bad number")),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(self.error("bad number"));
            }
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(self.error("bad number"));
            }
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        let token = std::str::from_utf8(&self.text[start..self.pos])
            .map_err(|_| self.error("bad number"))?;
        let n: f64 = token.parse().map_err(|_| self.error("bad number"))?;
        Ok(num(n))
    }
}

fn json_parse(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let text = to_string_js(heap, get_arg(args, 0))?;
    let mut p = JsonParser {
        text: text.as_bytes(),
        pos: 0,
        heap,
        depth: 0,
    };
    let v = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.text.len() {
        return Err(p.error("unexpected trailing characters"));
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// JSON.stringify
// ---------------------------------------------------------------------------

/// Stringify one value; `stack` is the active object/array chain for
/// circular detection.
///
/// `STRINGIFY_MAX_DEPTH` bounds native recursion: each nested
/// object/array adds several native frames (`stringify_value` ->
/// `stringify_array`/`stringify_object` -> `stringify_value`, plus a
/// possible `toJSON` re-entry), so unbounded nesting would exhaust the
/// host stack. Past the limit, stringify throws a `RangeError`.
const STRINGIFY_MAX_DEPTH: usize = 1000;

fn stringify_value(
    heap: &mut Heap,
    v: Value,
    stack: &mut Vec<Gc<Object>>,
    out: &mut Wtf8Buf,
) -> Result<(), NativeError> {
    // `toJSON` hook (data properties only, like the other builtins).
    let mut v = v;
    if let Value::Object(o) = v
        && let Some(hook) = get_own(heap, o, Wtf8::from_str("toJSON"))
        && is_callable(heap, hook)
    {
        // The hook runs user code (can GC): root the receiver and the hook
        // across it.
        let base = heap.protect(v);
        heap.protect(hook);
        let r = reenter_call(heap, hook, v, &[]);
        heap.release_temp(base);
        v = r?;
    }
    match v {
        Value::Undefined | Value::Symbol(_) => {
            // Top-level `undefined`/symbol/function serializes as
            // `undefined` (not valid JSON, but what the spec returns).
            return Err(NativeError::TypeError(
                "JSON.stringify: cannot serialize undefined".into(),
            ));
        }
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if b { "true" } else { "false" }),
        Value::Int(i) => out.push_str(&i.to_string()),
        Value::Float(f) => {
            if f.is_nan() || f.is_infinite() {
                out.push_str("null");
            } else {
                out.push_str(&js_number_to_string(f));
            }
        }
        Value::Str(s) => {
            push_quoted(out, s.as_wtf8());
        }
        Value::BigInt(_) => {
            return Err(NativeError::TypeError(
                "JSON.stringify: cannot serialize BigInt".into(),
            ));
        }
        Value::Object(o) => {
            if stack.contains(&o) {
                return Err(NativeError::TypeError(
                    "JSON.stringify: circular structure".into(),
                ));
            }
            if stack.len() >= STRINGIFY_MAX_DEPTH {
                return Err(NativeError::RangeError(
                    "JSON.stringify: maximum nesting depth exceeded".into(),
                ));
            }
            stack.push(o);
            let r = if heap.get(o).kind == ObjectKind::Array {
                stringify_array(heap, o, stack, out)
            } else {
                stringify_object(heap, o, stack, out)
            };
            stack.pop();
            r?;
        }
    }
    Ok(())
}

fn stringify_array(
    heap: &mut Heap,
    arr: Gc<Object>,
    stack: &mut Vec<Gc<Object>>,
    out: &mut Wtf8Buf,
) -> Result<(), NativeError> {
    // Root the array across element serialization (user `toJSON` can GC).
    let base = heap.protect(Value::Object(arr));
    let len = array_length(heap, arr);
    out.push_char('[');
    for i in 0..len {
        if i > 0 {
            out.push_char(',');
        }
        // Holes and `undefined`/function/symbol elements become `null`.
        // Anything else (BigInt, circular, throwing `toJSON`) propagates.
        match get_own(heap, arr, Wtf8::from_str(&i.to_string())) {
            Some(Value::Undefined) | None => out.push_str("null"),
            Some(v) if matches!(v, Value::Symbol(_)) || is_function(heap, v) => {
                out.push_str("null")
            }
            Some(v) => stringify_value(heap, v, stack, out)?,
        }
    }
    out.push_char(']');
    heap.release_temp(base);
    Ok(())
}

fn stringify_object(
    heap: &mut Heap,
    obj: Gc<Object>,
    stack: &mut Vec<Gc<Object>>,
    out: &mut Wtf8Buf,
) -> Result<(), NativeError> {
    let base = heap.protect(Value::Object(obj));
    // Snapshot keys first: user `toJSON` hooks may mutate the object.
    let keys: Vec<Wtf8Buf> = own_enumerable_keys(heap, obj)
        .iter()
        .map(|k| k.as_wtf8().to_owned())
        .collect();
    out.push_char('{');
    let mut first = true;
    for k in keys {
        let v = match get_own(heap, obj, k.as_wtf8()) {
            Some(v) => v,
            None => continue,
        };
        // `undefined`/function/symbol/BigInt values are skipped (spec:
        // values with no JSON representation are omitted from objects).
        if matches!(v, Value::Undefined | Value::Symbol(_) | Value::BigInt(_))
            || is_function(heap, v)
        {
            continue;
        }
        if !first {
            out.push_char(',');
        }
        first = false;
        push_quoted(out, k.as_wtf8());
        out.push_char(':');
        if let Err(e) = stringify_value(heap, v, stack, out) {
            heap.release_temp(base);
            return Err(e);
        }
    }
    out.push_char('}');
    heap.release_temp(base);
    Ok(())
}

/// Approximation: functions are `Function`/`NativeFunction` objects.
fn is_function(heap: &Heap, v: Value) -> bool {
    match v {
        Value::Object(o) => matches!(
            heap.get(o).kind,
            ObjectKind::Function(_) | ObjectKind::NativeFunction(_)
        ),
        _ => false,
    }
}

/// Append a double-quoted, escaped JSON string.
fn push_quoted(out: &mut Wtf8Buf, s: &Wtf8) {
    out.push_char('"');
    for cp in s.chars_wtf8() {
        match cp {
            0x22 => out.push_str("\\\""),
            0x5C => out.push_str("\\\\"),
            0x0A => out.push_str("\\n"),
            0x0D => out.push_str("\\r"),
            0x09 => out.push_str("\\t"),
            0x08 => out.push_str("\\b"),
            0x0C => out.push_str("\\f"),
            c if c < 0x20 => {
                out.push_str("\\u");
                for shift in [12, 8, 4, 0] {
                    out.push_char(
                        "0123456789abcdef".as_bytes()[((c >> shift) & 0xF) as usize] as char,
                    );
                }
            }
            // Lone surrogates pass through raw (spec QuoteJSONString only
            // escapes `"`, `\\`, and control chars).
            c => push_code_point(out, c),
        }
    }
    out.push_char('"');
}

fn json_stringify(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let v = get_arg(args, 0);
    // Root the input across the whole serialization: `toJSON` hooks run
    // user code and can GC at any point.
    let base = heap.protect(v);
    let mut out = Wtf8Buf::new();
    let mut stack = Vec::new();
    let r = match v {
        // Top-level `undefined`/function/symbol → `undefined` (spec).
        Value::Undefined | Value::Symbol(_) => Ok(Value::Undefined),
        v if is_function(heap, v) => Ok(Value::Undefined),
        v => {
            stringify_value(heap, v, &mut stack, &mut out)?;
            Ok(Value::Str(heap.alloc_wtf8_owned(out)?))
        }
    };
    heap.release_temp(base);
    r
}
