//! `TextEncoder` / `TextDecoder` per the WHATWG Encoding Standard.
//!
//! Installs the two globals via [`install_encoding`], mirroring the
//! [`install_url`](crate::bindings_url::install_url) pattern: native-function
//! constructor objects, one ordinary-object prototype per class, and all
//! instance state behind hidden `"\u{0}bowser:…"` properties — no host-side
//! tables or context ids.
//!
//! Encoding and decoding proper are delegated to the [`bowser_encoding`]
//! crate (label table, BOM sniffing, decoders); this module is the JS
//! binding layer: constructors, getters, `Uint8Array` results (built with
//! [`uint8array_from_bytes`](crate::builtin_typedarray::uint8array_from_bytes)),
//! and the streaming input queue.
//!
//! # Conformance notes (v1 simplifications)
//!
//! * **Call form.** Like `URL`, `TextEncoder()` / `TextDecoder()` without
//!   `new` behave like the construct form: `new.target` is invisible without
//!   a VM.
//! * **`ToString` is primitives-only.** `encode`/`encodeInto` inputs and the
//!   decoder label go through the same convention as the URL bindings
//!   (strings, numbers, booleans, null/undefined); objects, symbols, and
//!   bigints throw `TypeError` instead of invoking `toString`.
//! * **`decode` input** accepts `Uint8Array` views (fast path), any other
//!   genuine view type via its `buffer`/`byteOffset`/`byteLength` (covers
//!   `DataView` and the other typed arrays), and `ArrayBuffer`s. Anything
//!   else — including strings, numbers, and duck-typed `{buffer, …}`
//!   objects — throws `TypeError`, per WebIDL `BufferSource`.
//! * **`encodeInto` requires a `Uint8Array`** destination specifically;
//!   other view types throw `TypeError`, matching browsers' observable
//!   behavior.
//! * **Streaming holdback** is exact for UTF-8 and UTF-16. Legacy multibyte
//!   encodings hold back one trailing lead byte (a GB18030 four-byte
//!   sequence split after its second byte may report an error instead of
//!   waiting for more input); ISO-2022-JP holds back a trailing `ESC`.
//! * **The `replacement` label** throws `RangeError` from the `TextDecoder`
//!   constructor, per spec (it is not a usable decode encoding).

use bowser_encoding::{Encoding, decode_with, encoding_from_label};

use crate::bindings::with_temp_root;
use crate::builtin_reflect_proxy::{is_data_view_obj, is_typed_array_view};
use crate::builtin_typedarray::{
    uint8_view_bytes, uint8_view_bytes_mut, uint8array_from_bytes, view_buffer, view_byte_range,
};
use crate::gc::{AllocError, Gc, Heap};
use crate::value::{AccessorProp, NativeError, NativeFn, Object, ObjectKind, Property, Value};
use crate::wtf8::Wtf8;

// ---------------------------------------------------------------------------
// Hidden property keys
// ---------------------------------------------------------------------------

/// Marker on every `TextEncoder` instance (the spec's "encoder" slot).
const ENCODER_KEY: &str = "\u{0}bowser:textencoder";
/// Canonical encoding name (`str`) on every `TextDecoder` instance.
const DECODER_ENCODING_KEY: &str = "\u{0}bowser:decoder:encoding";
/// `fatal` flag (`bool`) on every `TextDecoder` instance.
const DECODER_FATAL_KEY: &str = "\u{0}bowser:decoder:fatal";
/// `ignoreBOM` flag (`bool`) on every `TextDecoder` instance.
const DECODER_IGNORE_BOM_KEY: &str = "\u{0}bowser:decoder:ignore_bom";
/// Streaming input queue: a `Uint8Array` view over the buffered bytes.
/// Absent (or empty) when nothing is buffered.
const DECODER_PENDING_KEY: &str = "\u{0}bowser:decoder:pending";
/// `TextDecoder` "BOM seen" flag (`bool`): set once the first scalar value
/// is serialized. A leading U+FEFF is stripped only while it is unset.
const DECODER_BOM_SEEN_KEY: &str = "\u{0}bowser:decoder:bom_seen";
/// `TextDecoder` "do not flush" flag (`bool`): true when the previous
/// `decode()` call used `stream: true`. A new call resets the decoder
/// (including "BOM seen") only when this is false.
const DECODER_DO_NOT_FLUSH_KEY: &str = "\u{0}bowser:decoder:do_not_flush";
/// The realm's `%Uint8Array.prototype%`, stashed on `TextEncoder.prototype`
/// at install: native methods receive no global handle, so `encode()`
/// resolves the prototype for its result view through the chain from
/// `this`. (Cross-file contract with `crate::builtin_typedarray`.)
const UINT8_PROTO_KEY: &str = "\u{0}bowser:uint8-proto";

// ---------------------------------------------------------------------------
// Installation
// ---------------------------------------------------------------------------

/// Install the encoding globals: `TextEncoder` and `TextDecoder`.
///
/// Called by `install_web_apis_inner` next to
/// [`install_url`](crate::bindings_url::install_url), and by the
/// parse-phase document shim so parser-inserted scripts (WPT's
/// testharness.js and the test scripts themselves) see the globals
/// mid-parse, before the production install runs.
///
/// Idempotent: the shim installs once per parser-inserted script on the
/// same realm, so the first call wins and every later call — including
/// the production install after the parse — is a no-op. One constructor
/// pair per realm, like a browser.
///
/// The probe checks *both* globals, not just `TextEncoder`: installation
/// is two-phase, and probing only the first half would latch a
/// half-finished install — if the `TextDecoder` half failed (e.g. OOM),
/// every later call would no-op and `TextDecoder` would stay missing
/// forever. With both probed, a partial install simply retries from the
/// top (re-running the `TextEncoder` half is harmless: it overwrites).
pub(crate) fn install_encoding(
    heap: &mut Heap,
    global: Gc<Object>,
) -> Result<(), crate::bindings::ApiError> {
    let installed = heap
        .get_property(global, Wtf8::from_str("TextEncoder"))
        .is_some()
        && heap
            .get_property(global, Wtf8::from_str("TextDecoder"))
            .is_some();
    if installed {
        return Ok(());
    }
    // -- TextEncoder ---------------------------------------------------
    // Constructor/prototype are unreachable from any root until linked onto
    // `global`; hold each as a temporary root across its build (the
    // `install_url` hazard: `alloc_str`, per-method `def_native`, and fresh
    // `alloc_object` calls in between can all collect).
    let enc_ctor = heap.alloc_object(Object::new(ObjectKind::NativeFunction(
        text_encoder_constructor,
    )))?;
    // `TextEncoder` constructs in this engine (v1 gap vs. WHATWG, which
    // forbids `new TextEncoder()`): record it so `Reflect.construct`
    // keeps working.
    heap.mark_native_constructor(text_encoder_constructor);
    with_temp_root(heap, Value::Object(enc_ctor), |heap| {
        let enc_proto = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
        with_temp_root(heap, Value::Object(enc_proto), |heap| {
            def_native(heap, enc_proto, "encode", text_encoder_encode)?;
            def_native(heap, enc_proto, "encodeInto", text_encoder_encode_into)?;
            def_getter(heap, enc_proto, "encoding", text_encoder_encoding_get)?;
            def(heap, enc_proto, "constructor", Value::Object(enc_ctor))?;
            // Stash the realm's `%Uint8Array.prototype%` for `encode()`:
            // native methods get no global handle, so the prototype for
            // the result view resolves through the chain from `this`.
            // (Cross-file contract with `crate::builtin_typedarray`.)
            let u8_proto = crate::builtin_reflect_proxy::uint8array_prototype(heap, global);
            heap.define_property_str(
                enc_proto,
                UINT8_PROTO_KEY,
                Property {
                    value: u8_proto,
                    writable: false,
                    enumerable: false,
                    configurable: false,
                },
            )?;
            Ok::<(), crate::bindings::ApiError>(())
        })?;
        def(heap, enc_ctor, "prototype", Value::Object(enc_proto))?;
        def(heap, global, "TextEncoder", Value::Object(enc_ctor))?;
        Ok::<(), crate::bindings::ApiError>(())
    })?;

    // -- TextDecoder ---------------------------------------------------
    let dec_ctor = heap.alloc_object(Object::new(ObjectKind::NativeFunction(
        text_decoder_constructor,
    )))?;
    // `TextDecoder` is a constructor (WHATWG Encoding `IsConstructor`).
    heap.mark_native_constructor(text_decoder_constructor);
    with_temp_root(heap, Value::Object(dec_ctor), |heap| {
        let dec_proto = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
        with_temp_root(heap, Value::Object(dec_proto), |heap| {
            def_native(heap, dec_proto, "decode", text_decoder_decode)?;
            def_getter(heap, dec_proto, "encoding", text_decoder_encoding_get)?;
            def_getter(heap, dec_proto, "fatal", text_decoder_fatal_get)?;
            def_getter(heap, dec_proto, "ignoreBOM", text_decoder_ignore_bom_get)?;
            def(heap, dec_proto, "constructor", Value::Object(dec_ctor))?;
            Ok::<(), crate::bindings::ApiError>(())
        })?;
        def(heap, dec_ctor, "prototype", Value::Object(dec_proto))?;
        def(heap, global, "TextDecoder", Value::Object(dec_ctor))?;
        Ok::<(), crate::bindings::ApiError>(())
    })?;

    Ok(())
}

fn def(heap: &mut Heap, obj: Gc<Object>, name: &str, val: Value) -> Result<(), AllocError> {
    heap.define_property_str(obj, name, Property::data(val))
}

fn def_native(heap: &mut Heap, obj: Gc<Object>, name: &str, f: NativeFn) -> Result<(), AllocError> {
    let v = Value::Object(heap.alloc_object(Object::new(ObjectKind::NativeFunction(f)))?);
    def(heap, obj, name, v)
}

/// A readonly accessor (`get` with no setter) on a prototype.
fn def_getter(heap: &mut Heap, obj: Gc<Object>, name: &str, f: NativeFn) -> Result<(), AllocError> {
    let g = Value::Object(heap.alloc_object(Object::new(ObjectKind::NativeFunction(f)))?);
    heap.define_accessor_str(obj, name, AccessorProp::new(g, Value::Undefined))
}

fn str_val(heap: &mut Heap, s: &str) -> Result<Value, NativeError> {
    heap.alloc_str(s)
        .map(Value::Str)
        .map_err(|_| NativeError::OutOfMemory)
}

fn type_err(msg: impl Into<String>) -> NativeError {
    NativeError::TypeError(msg.into())
}

/// Chain a fresh instance to `new.target.prototype` when the constructor ran
/// through the VM (the `URL` precedent in `bindings_url`); direct native
/// calls keep the null prototype.
fn chain_prototype(heap: &mut Heap, obj: Gc<Object>) -> Result<(), AllocError> {
    if let Some(Value::Object(ctor)) = crate::vm::active_new_target()
        && let Some(p) = heap.get_property(ctor, Wtf8::from_str("prototype"))
        && matches!(p.value, Value::Object(_))
    {
        heap.set_prototype(obj, p.value);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// TextEncoder
// ---------------------------------------------------------------------------

fn text_encoder_constructor(
    heap: &mut Heap,
    _this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let obj = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    with_temp_root(heap, Value::Object(obj), |heap| {
        chain_prototype(heap, obj)?;
        def(heap, obj, ENCODER_KEY, Value::Bool(true))?;
        Ok(Value::Object(obj))
    })
}

/// The spec's "encoder" slot: `TypeError` unless `this` is a `TextEncoder`
/// instance built by the constructor above.
fn require_encoder(heap: &Heap, this: Value) -> Result<Gc<Object>, NativeError> {
    let incompatible = || type_err("TextEncoder method called on incompatible receiver");
    let Value::Object(o) = this else {
        return Err(incompatible());
    };
    match heap
        .get_property(o, Wtf8::from_str(ENCODER_KEY))
        .map(|p| p.value)
    {
        Some(Value::Bool(true)) => Ok(o),
        _ => Err(incompatible()),
    }
}

fn text_encoder_encoding_get(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    require_encoder(heap, this)?;
    // A `TextEncoder` always encodes UTF-8.
    str_val(heap, Encoding::Utf8.name())
}

fn text_encoder_encode(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let encoder = require_encoder(heap, this)?;
    let mut bytes = Vec::new();
    if let Some(v) = args.first()
        && !v.is_undefined()
    {
        let input = to_js_string(*v)?;
        utf8_encode_wtf8(Wtf8::from_str(&input), &mut bytes);
    }
    // Proto-link the result to the realm's `%Uint8Array.prototype%`
    // (stashed on `TextEncoder.prototype` at install; the `Null` fallback
    // preserves the old shape when the stash is absent, e.g. a hand-built
    // prototype chain).
    let proto = crate::builtin_reflect_proxy::get_prop_chain(
        heap,
        encoder,
        Wtf8::from_str(UINT8_PROTO_KEY),
    )
    .unwrap_or(Value::Null);
    uint8array_from_bytes(heap, &bytes, proto)
}

fn text_encoder_encode_into(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    require_encoder(heap, this)?;
    let input = match args.first() {
        None | Some(Value::Undefined) => String::new(),
        Some(v) => to_js_string(*v)?,
    };
    let view = match args.get(1) {
        Some(Value::Object(o)) if uint8_view_bytes(heap, *o).is_some() => *o,
        _ => {
            return Err(type_err(
                "TextEncoder.encodeInto: destination must be a Uint8Array",
            ));
        }
    };
    // The encode loop below performs no allocation, so the mutable borrow of
    // the destination stays valid across it.
    let (read, written) = {
        let dst = uint8_view_bytes_mut(heap, view).expect("view checked above");
        let wtf8 = Wtf8::from_str(&input);
        // Fast path: pure-ASCII input copies byte-for-byte.
        if wtf8.is_ascii() {
            let n = wtf8.len().min(dst.len());
            dst[..n].copy_from_slice(&wtf8.as_bytes()[..n]);
            (n as u32, n)
        } else {
            let mut read: u32 = 0; // UTF-16 code units consumed
            let mut written: usize = 0;
            for cp in wtf8.chars_wtf8() {
                // Lone surrogates surface as 0xD800..=0xDFFF and become
                // U+FFFD; every other yielded value is a valid scalar.
                let c = char::from_u32(cp).unwrap_or('\u{FFFD}');
                let mut buf = [0u8; 4];
                let encoded = c.encode_utf8(&mut buf);
                // Never split a code point across the capacity boundary:
                // stop before a partial write.
                if written + encoded.len() > dst.len() {
                    break;
                }
                dst[written..written + encoded.len()].copy_from_slice(encoded.as_bytes());
                written += encoded.len();
                // Astral code points consume two UTF-16 code units; a lone
                // surrogate (replaced above) consumes one.
                read += if cp > 0xFFFF { 2 } else { 1 };
            }
            (read, written)
        }
    };
    // `view` is not touched past this point, so no rooting is needed across
    // the result allocation; `obj` is held as a temporary root across the
    // property writes (either can collect).
    let obj = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    with_temp_root(heap, Value::Object(obj), |heap| {
        def(heap, obj, "read", Value::Float(read as f64))?;
        def(heap, obj, "written", Value::Float(written as f64))?;
        Ok(Value::Object(obj))
    })
}

/// WHATWG "UTF-8 encode": the UTF-8 bytes of the code points, with lone
/// surrogates replaced by U+FFFD.
fn utf8_encode_wtf8(s: &Wtf8, out: &mut Vec<u8>) {
    // Well-formed input (the common case) is already valid UTF-8.
    if let Some(valid) = s.as_utf8() {
        out.extend_from_slice(valid.as_bytes());
        return;
    }
    for cp in s.chars_wtf8() {
        let c = char::from_u32(cp).unwrap_or('\u{FFFD}');
        let mut buf = [0u8; 4];
        out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
    }
}

// ---------------------------------------------------------------------------
// TextDecoder
// ---------------------------------------------------------------------------

fn text_decoder_constructor(
    heap: &mut Heap,
    _this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    let label = match args.first() {
        None | Some(Value::Undefined) => "utf-8".to_string(),
        Some(v) => to_js_string(*v)?,
    };
    // WHATWG "get an encoding": unknown labels are a RangeError, and so is
    // `replacement` (it can never be used for decoding).
    let enc = encoding_from_label(&label).ok_or_else(|| {
        NativeError::RangeError(format!("TextDecoder: unknown encoding label {label:?}"))
    })?;
    if enc == Encoding::Replacement {
        return Err(NativeError::RangeError(
            "TextDecoder: the replacement encoding cannot be used for decoding".to_string(),
        ));
    }
    let (fatal, ignore_bom) = match args.get(1) {
        None | Some(Value::Undefined) | Some(Value::Null) => (false, false),
        Some(Value::Object(opts)) => (
            heap.get_property(*opts, Wtf8::from_str("fatal"))
                .is_some_and(|p| p.value.is_truthy()),
            heap.get_property(*opts, Wtf8::from_str("ignoreBOM"))
                .is_some_and(|p| p.value.is_truthy()),
        ),
        Some(_) => return Err(type_err("TextDecoder: options must be an object")),
    };
    let obj = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    with_temp_root(heap, Value::Object(obj), |heap| {
        chain_prototype(heap, obj)?;
        // The canonical name re-resolves to `enc` via `encoding_from_label`
        // (bowser_encoding guarantees every name resolves to itself).
        let enc_name = str_val(heap, enc.name())?;
        def(heap, obj, DECODER_ENCODING_KEY, enc_name)?;
        def(heap, obj, DECODER_FATAL_KEY, Value::Bool(fatal))?;
        def(heap, obj, DECODER_IGNORE_BOM_KEY, Value::Bool(ignore_bom))?;
        Ok(Value::Object(obj))
    })
}

/// The decoder's `(encoding, fatal, ignoreBOM)`: `TypeError` unless `this`
/// is a `TextDecoder` instance built by the constructor above.
fn decoder_state(heap: &Heap, this: Value) -> Result<(Encoding, bool, bool), NativeError> {
    let incompatible = || type_err("TextDecoder method called on incompatible receiver");
    let Value::Object(dec) = this else {
        return Err(incompatible());
    };
    let name = match heap
        .get_property(dec, Wtf8::from_str(DECODER_ENCODING_KEY))
        .map(|p| p.value)
    {
        Some(Value::Str(s)) => s.as_wtf8().to_string(),
        _ => return Err(incompatible()),
    };
    let enc = encoding_from_label(&name).ok_or_else(incompatible)?;
    let flag = |key: &str| {
        heap.get_property(dec, Wtf8::from_str(key))
            .is_some_and(|p| p.value.is_truthy())
    };
    Ok((enc, flag(DECODER_FATAL_KEY), flag(DECODER_IGNORE_BOM_KEY)))
}

fn text_decoder_encoding_get(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let (enc, _, _) = decoder_state(heap, this)?;
    str_val(heap, enc.name())
}

fn text_decoder_fatal_get(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let (_, fatal, _) = decoder_state(heap, this)?;
    Ok(Value::Bool(fatal))
}

fn text_decoder_ignore_bom_get(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let (_, _, ignore_bom) = decoder_state(heap, this)?;
    Ok(Value::Bool(ignore_bom))
}

fn text_decoder_decode(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let Value::Object(dec) = this else {
        return Err(type_err(
            "TextDecoder.decode called on incompatible receiver",
        ));
    };
    // The allocations below (pending view, result string) can trigger a
    // collection while `dec` is only held as a raw handle here: root it
    // across the body.
    with_temp_root(heap, Value::Object(dec), |heap| {
        decode_inner(heap, dec, args)
    })
}

fn decode_inner(heap: &mut Heap, dec: Gc<Object>, args: &[Value]) -> Result<Value, NativeError> {
    let (enc, fatal, ignore_bom) = decoder_state(heap, Value::Object(dec))?;
    let new_bytes = match args.first() {
        None | Some(Value::Undefined) => Vec::new(),
        Some(v) => buffer_source_bytes(heap, *v)?,
    };
    let stream = match args.get(1) {
        None | Some(Value::Undefined) => false,
        Some(Value::Object(opts)) => heap
            .get_property(*opts, Wtf8::from_str("stream"))
            .is_some_and(|p| p.value.is_truthy()),
        Some(_) => return Err(type_err("TextDecoder.decode: options must be an object")),
    };

    // The I/O queue: bytes buffered by earlier streaming calls plus the new
    // input. (WHATWG calls this the decoder's I/O queue.)
    let mut queue: Vec<u8> = pending_bytes(heap, dec);
    let had_pending = !queue.is_empty();
    queue.extend_from_slice(&new_bytes);

    // The Encoding Standard's "do not flush" flag records whether the
    // previous `decode()` call streamed. A new call resets the decoder
    // (state, I/O queue, and "BOM seen") only when the previous call did
    // not stream — so a non-streaming call that follows a streaming one
    // keeps "BOM seen" and must not strip a BOM again.
    let prev_streamed = heap
        .get_property(dec, Wtf8::from_str(DECODER_DO_NOT_FLUSH_KEY))
        .is_some_and(|p| matches!(p.value, Value::Bool(true)));
    let bom_seen = prev_streamed
        && heap
            .get_property(dec, Wtf8::from_str(DECODER_BOM_SEEN_KEY))
            .is_some_and(|p| matches!(p.value, Value::Bool(true)));
    // Spec `decode()` step 2 runs before any input is processed, so even a
    // call that later throws (fatal) records its `stream` choice. (The
    // input/options validation above threw before the algorithm started,
    // leaving the decoder state untouched.)
    def(heap, dec, DECODER_DO_NOT_FLUSH_KEY, Value::Bool(stream))?;

    // Per the Encoding Standard "decode" steps, a leading byte order mark is
    // stripped only for the UTF-8/UTF-16 decodings (and never with
    // `ignoreBOM`): every other encoding decodes the BOM bytes as ordinary
    // input. The BOM byte sequences decode to exactly U+FEFF in these
    // encodings, so stripping the decoded U+FEFF is equivalent to removing
    // the bytes up front, and it also covers a BOM split across streaming
    // chunks (the held-back bytes decode together on a later call).
    let input = queue.as_slice();
    let fatal_err = || {
        type_err(format!(
            "TextDecoder: input is not valid {} (fatal decoding)",
            enc.name()
        ))
    };

    // `Some(tail)` re-stores the queue; `None` leaves the stored queue
    // alone (the common no-streaming, nothing-pending case allocates
    // nothing extra).
    let mut store_tail: Option<Vec<u8>> = None;
    let text = if stream {
        // Hold back a possibly-incomplete trailing character for the next
        // chunk instead of decoding (or erroring on) it now.
        let hold = trailing_partial(input, enc);
        let split = input.len() - hold;
        let (text, had_errors) =
            decode_with(&input[..split], enc).map_err(|e| type_err(e.to_string()))?;
        if fatal && had_errors {
            return Err(fatal_err());
        }
        if hold > 0 || had_pending {
            store_tail = Some(input[split..].to_vec());
        }
        text
    } else {
        let (text, had_errors) = decode_with(input, enc).map_err(|e| type_err(e.to_string()))?;
        if fatal && had_errors {
            return Err(fatal_err());
        }
        // A non-streaming decode resets the queue.
        if had_pending {
            store_tail = Some(Vec::new());
        }
        text
    };

    // Per the Encoding Standard's "serialize I/O queue", a leading U+FEFF is
    // stripped only while the decoder's "BOM seen" flag is unset, and the
    // flag is set once the first scalar value is serialized — i.e. when the
    // pre-strip text is non-empty. (A streaming chunk held back entirely
    // serializes nothing, so the flag correctly stays unset.)
    let mut text = text;
    if !ignore_bom
        && !bom_seen
        && matches!(enc, Encoding::Utf8 | Encoding::Utf16Be | Encoding::Utf16Le)
    {
        if !text.is_empty() {
            def(heap, dec, DECODER_BOM_SEEN_KEY, Value::Bool(true))?;
        }
        if let Some(rest) = text.strip_prefix('\u{FEFF}') {
            text = rest.to_owned();
        }
    }

    if let Some(tail) = store_tail {
        // Internal bookkeeping view, never exposed to script: the null
        // prototype is fine.
        let view = uint8array_from_bytes(heap, &tail, Value::Null)?;
        with_temp_root(heap, view, |heap| {
            def(heap, dec, DECODER_PENDING_KEY, view)?;
            Ok::<(), NativeError>(())
        })?;
    }
    str_val(heap, &text)
}

/// Bytes buffered by earlier `stream: true` calls; empty when nothing is
/// buffered (or the key was never stored).
fn pending_bytes(heap: &Heap, dec: Gc<Object>) -> Vec<u8> {
    heap.get_property(dec, Wtf8::from_str(DECODER_PENDING_KEY))
        .and_then(|p| match p.value {
            Value::Object(view) => uint8_view_bytes(heap, view).map(|b| b.to_vec()),
            _ => None,
        })
        .unwrap_or_default()
}

/// WebIDL `BufferSource` as bytes: `Uint8Array` views on the fast path,
/// any other genuine view (typed arrays, `DataView`) via its
/// `buffer`/`byteOffset`/`byteLength`, else a plain `ArrayBuffer` (the only
/// other object kind in this engine carrying a byte store). Everything
/// else is a `TypeError`.
fn buffer_source_bytes(heap: &Heap, v: Value) -> Result<Vec<u8>, NativeError> {
    let bad = || type_err("TextDecoder.decode: input must be an ArrayBuffer or ArrayBufferView");
    let Value::Object(o) = v else {
        return Err(bad());
    };
    if let Some(b) = uint8_view_bytes(heap, o) {
        return Ok(b.to_vec());
    }
    // The `view_buffer` branch answers `Some` for *any* own `"buffer"` data
    // property, so it only runs for genuine views: a duck-typed plain
    // object (`{buffer: someArrayBuffer, …}`) must throw per WebIDL
    // `BufferSource` instead of decoding.
    if is_genuine_view(heap, o)
        && let Some(Value::Object(buf)) = view_buffer(heap, o)
        && let Some(bytes) = heap.object_bytes(buf)
        && let Some((off, len)) = view_byte_range(heap, o)
    {
        // Tag reads, clamped defensively: the view metadata comes from
        // the unforgeable kind tag (`length`/`byteLength`/`byteOffset`
        // are prototype accessors now, invisible to property reads).
        let off = (off as usize).min(bytes.len());
        let end = off.saturating_add(len as usize).min(bytes.len());
        return Ok(bytes[off..end].to_vec());
    }
    if let Some(b) = heap.object_bytes(o) {
        return Ok(b.to_vec());
    }
    Err(bad())
}

// ---------------------------------------------------------------------------
// Genuine-view recognition (the `ArrayBuffer.isView` semantics)
// ---------------------------------------------------------------------------

/// The `ArrayBuffer.isView` semantics the engine's other modules use:
/// genuine typed-array views and `DataView`s, nothing else. Both are
/// recognized by their unforgeable [`ObjectKind`] tags — the NUL-prefixed
/// hidden-slot scheme this module used to duplicate was forgeable from
/// script, so the tag is the only check. A plain object that merely owns
/// a `"buffer"` property fails closed here.
fn is_genuine_view(heap: &Heap, o: Gc<Object>) -> bool {
    is_typed_array_view(heap, o) || is_data_view_obj(heap, o)
}

/// Trailing bytes of `bytes` that may form an incomplete character for
/// `enc`: held back in the streaming queue instead of being decoded now.
fn trailing_partial(bytes: &[u8], enc: Encoding) -> usize {
    match enc {
        Encoding::Utf8 => utf8_trailing_partial(bytes),
        Encoding::Utf16Le | Encoding::Utf16Be => bytes.len() % 2,
        // Legacy multibyte encodings: a trailing lead byte may start a
        // sequence the next chunk completes.
        Encoding::ShiftJis => trailing_lead(bytes, |b| matches!(b, 0x81..=0x9F | 0xE0..=0xFC)),
        Encoding::Gbk | Encoding::Gb18030 | Encoding::EucKr | Encoding::Big5 => {
            trailing_lead(bytes, |b| matches!(b, 0x81..=0xFE))
        }
        Encoding::EucJp => trailing_lead(bytes, |b| matches!(b, 0x8E | 0x8F | 0xA1..=0xFE)),
        // A trailing ESC may open an escape sequence the next chunk
        // finishes.
        Encoding::Iso2022Jp => trailing_lead(bytes, |b| b == 0x1B),
        // Single-byte encodings cannot have partial characters.
        _ => 0,
    }
}

fn trailing_lead(bytes: &[u8], is_lead: impl Fn(u8) -> bool) -> usize {
    usize::from(bytes.last().is_some_and(|&b| is_lead(b)))
}

/// Length of the trailing incomplete UTF-8 sequence, or 0 when the input
/// ends on a character boundary. Stray continuation bytes (no lead byte in
/// the tail) are errors, not partial input, so they are decoded immediately.
/// A lead whose second byte already violates the UTF-8 decoder's narrowed
/// second-byte ranges (`E0`→`A0–BF`, `ED`→`80–9F`, `F0`→`90–BF`,
/// `F4`→`80–8F`) is a certain error too: the sequence can never complete,
/// so only the suffix after it may still be incomplete.
fn utf8_trailing_partial(bytes: &[u8]) -> usize {
    let n = bytes.len();
    if n == 0 {
        return 0;
    }
    // Walk back over continuation bytes (at most 3: the longest tail of a
    // 4-byte sequence), then include the lead byte itself — a bare lead
    // byte is the one-byte incomplete case.
    let mut start = n;
    let mut cont = 0;
    while start > 0 && cont < 3 && (bytes[start - 1] & 0xC0) == 0x80 {
        start -= 1;
        cont += 1;
    }
    if start == 0 {
        // The whole tail is continuation bytes with no lead: stray errors,
        // decode them now.
        return 0;
    }
    start -= 1;
    let lead = bytes[start];
    let expected = match lead {
        0x00..=0x7F => 1,
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        // 0x80..=0xC1 and 0xF5..=0xFF can never lead a sequence: the byte
        // errors immediately, so the hold is whatever the suffix needs.
        _ => return utf8_trailing_partial(&bytes[start + 1..]),
    };
    let have = n - start;
    if have >= 2 {
        let (lo, hi) = match lead {
            0xE0 => (0xA0, 0xBF),
            0xED => (0x80, 0x9F),
            0xF0 => (0x90, 0xBF),
            0xF4 => (0x80, 0x8F),
            _ => (0x80, 0xBF),
        };
        let second = bytes[start + 1];
        if second < lo || second > hi {
            // Already invalid: the lead errors at once and the second byte
            // is reprocessed with fresh decoder state, so the hold is
            // whatever the remaining suffix needs.
            return utf8_trailing_partial(&bytes[start + 1..]);
        }
    }
    if have < expected { have } else { 0 }
}

// ---------------------------------------------------------------------------
// JS value helpers (the `bindings_url` conventions)
// ---------------------------------------------------------------------------

/// JS `ToString` for the primitive cases the encoding bindings need.
/// Objects, symbols, and bigints are rejected (a real `toString` call is the
/// VM's job), matching the `bindings.rs` convention.
fn to_js_string(v: Value) -> Result<String, NativeError> {
    match v {
        Value::Str(s) => Ok(s.as_wtf8().to_string()),
        Value::Int(i) => Ok(i.to_string()),
        Value::Float(f) => Ok(js_num_to_string(f)),
        Value::Bool(b) => Ok(b.to_string()),
        Value::Null => Ok("null".to_string()),
        Value::Undefined => Ok("undefined".to_string()),
        _ => Err(type_err("value cannot be converted to a string")),
    }
}

// v1 number formatting; the exact `Number::toString` (exponents, `-0`) is
// the builtins worker's domain.
fn js_num_to_string(f: f64) -> String {
    if f.is_nan() {
        return "NaN".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        };
    }
    format!("{f}")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Heap, Gc<Object>) {
        let mut heap = Heap::new();
        let global = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        crate::builtin_reflect_proxy::install(&mut heap, global).unwrap();
        install_encoding(&mut heap, global).unwrap();
        (heap, global)
    }

    fn native_fn(heap: &Heap, obj: Gc<Object>, name: &str) -> NativeFn {
        match heap.get_property(obj, Wtf8::from_str(name)).unwrap().value {
            Value::Object(o) => match heap.get(o).kind {
                ObjectKind::NativeFunction(f) => f,
                _ => panic!("{name} is not a native function"),
            },
            v => panic!("{name} missing, got {v:?}"),
        }
    }

    fn global_fn(heap: &Heap, global: Gc<Object>, name: &str) -> NativeFn {
        native_fn(heap, global, name)
    }

    /// The constructor's `prototype` object (methods/getters live there).
    fn proto(heap: &Heap, global: Gc<Object>, ctor: &str) -> Gc<Object> {
        let c = match heap
            .get_property(global, Wtf8::from_str(ctor))
            .unwrap()
            .value
        {
            Value::Object(o) => o,
            v => panic!("{ctor} missing: {v:?}"),
        };
        match heap
            .get_property(c, Wtf8::from_str("prototype"))
            .unwrap()
            .value
        {
            Value::Object(o) => o,
            v => panic!("{ctor}.prototype missing: {v:?}"),
        }
    }

    fn arg(heap: &mut Heap, s: &str) -> Value {
        Value::Str(heap.alloc_str(s).unwrap())
    }

    /// A JS string from raw WTF-8 bytes (for lone-surrogate inputs).
    fn wtf8_arg(heap: &mut Heap, bytes: &[u8]) -> Value {
        let w = Wtf8::from_bytes(bytes).expect("test input must be valid WTF-8");
        Value::Str(heap.alloc_str_wtf8(w).unwrap())
    }

    fn uint8_arg(heap: &mut Heap, bytes: &[u8]) -> Value {
        uint8array_from_bytes(heap, bytes, Value::Null).unwrap()
    }

    /// The backing `ArrayBuffer` of an encoder-result view, via the hidden
    /// slot (these views carry no own `"buffer"` data property).
    fn view_buf(heap: &Heap, v: Value) -> Value {
        match v {
            Value::Object(o) => match view_buffer(heap, o) {
                Some(Value::Object(b)) => Value::Object(b),
                _ => panic!("view has no buffer: {v:?}"),
            },
            v => panic!("not a view: {v:?}"),
        }
    }

    /// `{k: v, …}` options object.
    fn opts(heap: &mut Heap, pairs: &[(&str, bool)]) -> Value {
        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        for (k, v) in pairs {
            heap.define_property_str(obj, k, Property::data(Value::Bool(*v)))
                .unwrap();
        }
        Value::Object(obj)
    }

    fn call_method(
        heap: &mut Heap,
        proto_obj: Gc<Object>,
        name: &str,
        this: Value,
        args: &[Value],
    ) -> Value {
        native_fn(heap, proto_obj, name)(heap, this, args).unwrap()
    }

    /// Invoke a prototype getter with `this` (accessors don't run through
    /// `get_property`; call the getter function directly).
    fn get_getter(heap: &mut Heap, proto_obj: Gc<Object>, name: &str, this: Value) -> Value {
        get_getter_result(heap, proto_obj, name, this).unwrap()
    }

    fn get_getter_result(
        heap: &mut Heap,
        proto_obj: Gc<Object>,
        name: &str,
        this: Value,
    ) -> Result<Value, NativeError> {
        let ap = heap
            .get_accessor(proto_obj, Wtf8::from_str(name))
            .unwrap_or_else(|| panic!("{name} accessor missing"));
        match ap.get {
            Value::Object(o) => match heap.get(o).kind {
                ObjectKind::NativeFunction(f) => f(heap, this, &[]),
                _ => panic!("{name} getter is not a native function"),
            },
            v => panic!("{name} getter missing, got {v:?}"),
        }
    }

    fn prop(heap: &Heap, o: Gc<Object>, name: &str) -> Value {
        heap.get_property(o, Wtf8::from_str(name)).unwrap().value
    }

    fn js_str(v: Value) -> String {
        match v {
            Value::Str(s) => s.as_wtf8().as_utf8().unwrap_or("").to_owned(),
            v => panic!("not a string: {v:?}"),
        }
    }

    fn new_encoder(heap: &mut Heap, global: Gc<Object>) -> Gc<Object> {
        match global_fn(heap, global, "TextEncoder")(heap, Value::Undefined, &[]) {
            Ok(Value::Object(o)) => o,
            v => panic!("TextEncoder() did not return an object: {v:?}"),
        }
    }

    fn new_decoder(heap: &mut Heap, global: Gc<Object>, args: &[Value]) -> Gc<Object> {
        match global_fn(heap, global, "TextDecoder")(heap, Value::Undefined, args) {
            Ok(Value::Object(o)) => o,
            v => panic!("TextDecoder() did not return an object: {v:?}"),
        }
    }

    fn view_bytes_of(heap: &Heap, v: Value) -> Vec<u8> {
        match v {
            Value::Object(o) => uint8_view_bytes(heap, o).expect("uint8 view").to_vec(),
            v => panic!("not a view: {v:?}"),
        }
    }

    // -- installation --------------------------------------------------

    #[test]
    fn encoding_installs_globals() {
        let (heap, global) = setup();
        for name in ["TextEncoder", "TextDecoder"] {
            assert!(
                matches!(
                    heap.get_property(global, Wtf8::from_str(name))
                        .unwrap()
                        .value,
                    Value::Object(_)
                ),
                "{name} not installed"
            );
        }
        let ep = proto(&heap, global, "TextEncoder");
        for name in ["encode", "encodeInto"] {
            native_fn(&heap, ep, name);
        }
        assert!(
            heap.get_accessor(ep, Wtf8::from_str("encoding")).is_some(),
            "TextEncoder.prototype.encoding getter missing"
        );
        let dp = proto(&heap, global, "TextDecoder");
        native_fn(&heap, dp, "decode");
        for name in ["encoding", "fatal", "ignoreBOM"] {
            assert!(
                heap.get_accessor(dp, Wtf8::from_str(name)).is_some(),
                "TextDecoder.prototype.{name} getter missing"
            );
        }
    }

    // -- TextEncoder ---------------------------------------------------

    #[test]
    fn encoder_encoding_is_utf8() {
        let (mut heap, global) = setup();
        let enc = new_encoder(&mut heap, global);
        let p = proto(&heap, global, "TextEncoder");
        let want = arg(&mut heap, "utf-8");
        assert_eq!(
            get_getter(&mut heap, p, "encoding", Value::Object(enc)),
            want
        );
    }

    #[test]
    fn encode_hello_returns_uint8array() {
        let (mut heap, global) = setup();
        let enc = new_encoder(&mut heap, global);
        let p = proto(&heap, global, "TextEncoder");
        let hello = arg(&mut heap, "hello");
        let v = call_method(&mut heap, p, "encode", Value::Object(enc), &[hello]);
        // `ArrayBuffer.isView` agrees it is a view …
        let ab = match prop(&heap, global, "ArrayBuffer") {
            Value::Object(o) => o,
            v => panic!("ArrayBuffer missing: {v:?}"),
        };
        let is_view = native_fn(&heap, ab, "isView");
        assert_eq!(
            is_view(&mut heap, Value::Undefined, &[v]).unwrap(),
            Value::Bool(true)
        );
        // … and the bytes are exactly "hello".
        assert_eq!(view_bytes_of(&heap, v), &[104, 101, 108, 108, 111]);
    }

    #[test]
    fn encode_lone_surrogate_becomes_replacement_char() {
        let (mut heap, global) = setup();
        let enc = new_encoder(&mut heap, global);
        let p = proto(&heap, global, "TextEncoder");
        // CESU-8 U+D800: a lone surrogate in WTF-8.
        let lone = wtf8_arg(&mut heap, &[0xED, 0xA0, 0x80]);
        let v = call_method(&mut heap, p, "encode", Value::Object(enc), &[lone]);
        // U+FFFD in UTF-8.
        assert_eq!(view_bytes_of(&heap, v), &[0xEF, 0xBF, 0xBD]);
    }

    #[test]
    fn encode_defaults_to_empty() {
        let (mut heap, global) = setup();
        let enc = new_encoder(&mut heap, global);
        let p = proto(&heap, global, "TextEncoder");
        let v = call_method(&mut heap, p, "encode", Value::Object(enc), &[]);
        assert_eq!(view_bytes_of(&heap, v), &[]);
    }

    #[test]
    fn encode_into_read_written_counts() {
        let (mut heap, global) = setup();
        let enc = new_encoder(&mut heap, global);
        let p = proto(&heap, global, "TextEncoder");
        let this = Value::Object(enc);

        // Exact fit.
        let dst = uint8_arg(&mut heap, &[0u8; 5]);
        let hello = arg(&mut heap, "hello");
        let r = call_method(&mut heap, p, "encodeInto", this, &[hello, dst]);
        let Value::Object(ro) = r else {
            panic!("encodeInto did not return an object: {r:?}");
        };
        assert_eq!(prop(&heap, ro, "read"), Value::Float(5.0));
        assert_eq!(prop(&heap, ro, "written"), Value::Float(5.0));
        assert_eq!(view_bytes_of(&heap, dst), &[104, 101, 108, 108, 111]);

        // A code point is never split: "a💩c" needs 1+4+1 bytes; with room
        // for 5 only "a💩" is written (read counts UTF-16 code units).
        let dst = uint8_arg(&mut heap, &[0u8; 5]);
        let astral = arg(&mut heap, "a\u{1F4A9}c");
        let r = call_method(&mut heap, p, "encodeInto", this, &[astral, dst]);
        let Value::Object(ro) = r else {
            panic!("encodeInto did not return an object: {r:?}");
        };
        assert_eq!(prop(&heap, ro, "read"), Value::Float(3.0));
        assert_eq!(prop(&heap, ro, "written"), Value::Float(5.0));
        assert_eq!(view_bytes_of(&heap, dst), &[0x61, 0xF0, 0x9F, 0x92, 0xA9]);
    }

    #[test]
    fn encode_into_rejects_non_uint8_destination() {
        let (mut heap, global) = setup();
        let enc = new_encoder(&mut heap, global);
        let p = proto(&heap, global, "TextEncoder");
        let f = native_fn(&heap, p, "encodeInto");
        let this = Value::Object(enc);
        let s = arg(&mut heap, "hi");
        let nope = arg(&mut heap, "nope");
        assert!(matches!(
            f(&mut heap, this, &[s, Value::Undefined]),
            Err(NativeError::TypeError(_))
        ));
        assert!(matches!(
            f(&mut heap, this, &[s, nope]),
            Err(NativeError::TypeError(_))
        ));
    }

    // -- TextDecoder ---------------------------------------------------

    #[test]
    fn decoder_defaults() {
        let (mut heap, global) = setup();
        let dec = new_decoder(&mut heap, global, &[]);
        let p = proto(&heap, global, "TextDecoder");
        let this = Value::Object(dec);
        let utf8 = arg(&mut heap, "utf-8");
        assert_eq!(get_getter(&mut heap, p, "encoding", this), utf8);
        assert_eq!(get_getter(&mut heap, p, "fatal", this), Value::Bool(false));
        assert_eq!(
            get_getter(&mut heap, p, "ignoreBOM", this),
            Value::Bool(false)
        );
    }

    #[test]
    fn decoder_label_and_options() {
        let (mut heap, global) = setup();
        // Labels are ASCII case-insensitive; the getter reports the
        // canonical name.
        let label = arg(&mut heap, "UTF-16LE");
        let options = opts(&mut heap, &[("fatal", true), ("ignoreBOM", true)]);
        let dec = new_decoder(&mut heap, global, &[label, options]);
        let p = proto(&heap, global, "TextDecoder");
        let this = Value::Object(dec);
        let want = arg(&mut heap, "utf-16le");
        assert_eq!(get_getter(&mut heap, p, "encoding", this), want);
        assert_eq!(get_getter(&mut heap, p, "fatal", this), Value::Bool(true));
        assert_eq!(
            get_getter(&mut heap, p, "ignoreBOM", this),
            Value::Bool(true)
        );
    }

    #[test]
    fn unknown_label_throws_range_error() {
        let (mut heap, global) = setup();
        let ctor = global_fn(&heap, global, "TextDecoder");
        for label in ["bogus", "", "replacement"] {
            let l = arg(&mut heap, label);
            assert!(
                matches!(
                    ctor(&mut heap, Value::Undefined, &[l]),
                    Err(NativeError::RangeError(_))
                ),
                "label {label:?} should throw RangeError"
            );
        }
    }

    #[test]
    fn decode_hi() {
        let (mut heap, global) = setup();
        let dec = new_decoder(&mut heap, global, &[]);
        let p = proto(&heap, global, "TextDecoder");
        let input = uint8_arg(&mut heap, &[104, 105]);
        let v = call_method(&mut heap, p, "decode", Value::Object(dec), &[input]);
        assert_eq!(js_str(v), "hi");
    }

    #[test]
    fn decode_bom_stripped_once_per_decoder_lifetime() {
        // Encoding Standard "serialize I/O queue": the BOM is stripped only
        // while the decoder's "BOM seen" flag is unset; the flag is set on
        // the first serialized item, and a non-streaming call does not reset
        // it when the previous call streamed.
        let (mut heap, global) = setup();
        let dec = new_decoder(&mut heap, global, &[]);
        let p = proto(&heap, global, "TextDecoder");
        let this = Value::Object(dec);
        let stream = opts(&mut heap, &[("stream", true)]);
        let a = uint8_arg(&mut heap, &[0x41]);
        assert_eq!(
            js_str(call_method(&mut heap, p, "decode", this, &[a, stream])),
            "A"
        );
        // Previous call streamed, so this non-streaming call keeps "BOM
        // seen": the BOM is preserved, not stripped.
        let bom = uint8_arg(&mut heap, &[0xEF, 0xBB, 0xBF]);
        assert_eq!(
            js_str(call_method(&mut heap, p, "decode", this, &[bom])),
            "\u{FEFF}"
        );
    }

    #[test]
    fn decode_bom_stripped_again_after_non_streaming_reset() {
        // When the previous call did not stream, a new call resets the
        // decoder (including "BOM seen"), so each BOM is stripped.
        let (mut heap, global) = setup();
        let dec = new_decoder(&mut heap, global, &[]);
        let p = proto(&heap, global, "TextDecoder");
        let this = Value::Object(dec);
        let bom_a = uint8_arg(&mut heap, &[0xEF, 0xBB, 0xBF, 0x41]);
        assert_eq!(
            js_str(call_method(&mut heap, p, "decode", this, &[bom_a])),
            "A"
        );
        let bom_b = uint8_arg(&mut heap, &[0xEF, 0xBB, 0xBF, 0x42]);
        assert_eq!(
            js_str(call_method(&mut heap, p, "decode", this, &[bom_b])),
            "B"
        );
    }

    #[test]
    fn decode_bom_split_across_streaming_chunks() {
        // A chunk held back entirely serializes nothing, so "BOM seen"
        // stays unset until the BOM completes on a later call.
        let (mut heap, global) = setup();
        let dec = new_decoder(&mut heap, global, &[]);
        let p = proto(&heap, global, "TextDecoder");
        let this = Value::Object(dec);
        let stream = opts(&mut heap, &[("stream", true)]);
        let part1 = uint8_arg(&mut heap, &[0xEF, 0xBB]);
        assert_eq!(
            js_str(call_method(&mut heap, p, "decode", this, &[part1, stream])),
            ""
        );
        let part2 = uint8_arg(&mut heap, &[0xBF, 0x41]);
        assert_eq!(
            js_str(call_method(&mut heap, p, "decode", this, &[part2])),
            "A"
        );
    }

    #[test]
    fn decode_replaces_malformed_by_default() {
        let (mut heap, global) = setup();
        let dec = new_decoder(&mut heap, global, &[]);
        let p = proto(&heap, global, "TextDecoder");
        let input = uint8_arg(&mut heap, &[0xFF]);
        let v = call_method(&mut heap, p, "decode", Value::Object(dec), &[input]);
        assert_eq!(js_str(v), "\u{FFFD}");
    }

    #[test]
    fn decode_fatal_throws_type_error() {
        let (mut heap, global) = setup();
        let label = arg(&mut heap, "utf-8");
        let options = opts(&mut heap, &[("fatal", true)]);
        let dec = new_decoder(&mut heap, global, &[label, options]);
        let p = proto(&heap, global, "TextDecoder");
        let f = native_fn(&heap, p, "decode");
        let this = Value::Object(dec);
        // Bare 0xFF is malformed UTF-8 …
        let bad = uint8_arg(&mut heap, &[0xFF]);
        assert!(matches!(
            f(&mut heap, this, &[bad]),
            Err(NativeError::TypeError(_))
        ));
        // … and so is a CESU-8-encoded lone surrogate (ED A0 80).
        let surrogate = uint8_arg(&mut heap, &[0xED, 0xA0, 0x80]);
        assert!(matches!(
            f(&mut heap, this, &[surrogate]),
            Err(NativeError::TypeError(_))
        ));
        // Valid input still decodes.
        let good = uint8_arg(&mut heap, &[104, 105]);
        assert_eq!(js_str(f(&mut heap, this, &[good]).unwrap()), "hi");
    }

    #[test]
    fn decode_rejects_non_buffer_source() {
        let (mut heap, global) = setup();
        let dec = new_decoder(&mut heap, global, &[]);
        let p = proto(&heap, global, "TextDecoder");
        let f = native_fn(&heap, p, "decode");
        let this = Value::Object(dec);
        let s = arg(&mut heap, "hi");
        for bad in [s, Value::Int(42), Value::Null] {
            assert!(
                matches!(f(&mut heap, this, &[bad]), Err(NativeError::TypeError(_))),
                "decode({bad:?}) should throw TypeError"
            );
        }
        // `undefined` (and no argument) decode as empty input.
        assert_eq!(js_str(f(&mut heap, this, &[]).unwrap()), "");
        assert_eq!(js_str(f(&mut heap, this, &[Value::Undefined]).unwrap()), "");
    }

    #[test]
    fn decode_accepts_arraybuffer() {
        let (mut heap, global) = setup();
        // Take the `.buffer` of an encoder result: a genuine ArrayBuffer.
        let enc = new_encoder(&mut heap, global);
        let ep = proto(&heap, global, "TextEncoder");
        let hi = arg(&mut heap, "hi");
        let view = call_method(&mut heap, ep, "encode", Value::Object(enc), &[hi]);
        let buf = view_buf(&heap, view);
        let dec = new_decoder(&mut heap, global, &[]);
        let dp = proto(&heap, global, "TextDecoder");
        let v = call_method(&mut heap, dp, "decode", Value::Object(dec), &[buf]);
        assert_eq!(js_str(v), "hi");
    }

    #[test]
    fn decode_rejects_duck_typed_view() {
        let (mut heap, global) = setup();
        let dec = new_decoder(&mut heap, global, &[]);
        let dp = proto(&heap, global, "TextDecoder");
        let f = native_fn(&heap, dp, "decode");
        let this = Value::Object(dec);

        // A genuine ArrayBuffer holding "hi", to hang the forged object's
        // `buffer` property on.
        let ab = {
            let enc = new_encoder(&mut heap, global);
            let ep = proto(&heap, global, "TextEncoder");
            let hi = arg(&mut heap, "hi");
            let view = call_method(&mut heap, ep, "encode", Value::Object(enc), &[hi]);
            view_buf(&heap, view)
        };
        // `{buffer: ab, byteOffset: 0, byteLength: 2}` has the right shape
        // but none of the hidden view slots: it is not an ArrayBufferView,
        // so decoding it must throw instead of yielding "hi".
        let forged = {
            let obj = heap
                .alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
            heap.define_property_str(obj, "buffer", Property::data(ab))
                .unwrap();
            heap.define_property_str(obj, "byteOffset", Property::data(Value::Int(0)))
                .unwrap();
            heap.define_property_str(obj, "byteLength", Property::data(Value::Int(2)))
                .unwrap();
            Value::Object(obj)
        };
        assert!(matches!(
            f(&mut heap, this, &[forged]),
            Err(NativeError::TypeError(_))
        ));
        // The genuine ArrayBuffer itself still decodes.
        assert_eq!(js_str(f(&mut heap, this, &[ab]).unwrap()), "hi");
    }

    #[test]
    fn decode_accepts_genuine_dataview_and_typed_array() {
        let (mut heap, global) = setup();
        // A genuine ArrayBuffer holding "hi".
        let enc = new_encoder(&mut heap, global);
        let ep = proto(&heap, global, "TextEncoder");
        let hi = arg(&mut heap, "hi");
        let view = call_method(&mut heap, ep, "encode", Value::Object(enc), &[hi]);
        let ab = view_buf(&heap, view);

        let dec = new_decoder(&mut heap, global, &[]);
        let dp = proto(&heap, global, "TextDecoder");
        let f = native_fn(&heap, dp, "decode");
        let this = Value::Object(dec);

        // A genuine DataView over the buffer.
        let dv = global_fn(&heap, global, "DataView")(&mut heap, Value::Undefined, &[ab]).unwrap();
        assert_eq!(js_str(f(&mut heap, this, &[dv]).unwrap()), "hi");
        // A genuine non-Uint8 typed array: one Uint16 element spanning the
        // two bytes.
        let u16a =
            global_fn(&heap, global, "Uint16Array")(&mut heap, Value::Undefined, &[ab]).unwrap();
        assert_eq!(js_str(f(&mut heap, this, &[u16a]).unwrap()), "hi");
    }

    #[test]
    fn bom_stripped_by_default_kept_with_ignore_bom() {
        let (mut heap, global) = setup();
        let dp = proto(&heap, global, "TextDecoder");

        let dec = new_decoder(&mut heap, global, &[]);
        let input = uint8_arg(&mut heap, &[0xEF, 0xBB, 0xBF, 104, 105]);
        let v = call_method(&mut heap, dp, "decode", Value::Object(dec), &[input]);
        assert_eq!(js_str(v), "hi");

        let label = arg(&mut heap, "utf-8");
        let options = opts(&mut heap, &[("ignoreBOM", true)]);
        let dec = new_decoder(&mut heap, global, &[label, options]);
        let input = uint8_arg(&mut heap, &[0xEF, 0xBB, 0xBF, 104, 105]);
        let v = call_method(&mut heap, dp, "decode", Value::Object(dec), &[input]);
        assert_eq!(js_str(v), "\u{FEFF}hi");
    }

    #[test]
    fn astral_plane_round_trip() {
        let (mut heap, global) = setup();
        let enc = new_encoder(&mut heap, global);
        let ep = proto(&heap, global, "TextEncoder");
        let pile = arg(&mut heap, "\u{1F4A9}");
        let v = call_method(&mut heap, ep, "encode", Value::Object(enc), &[pile]);
        assert_eq!(view_bytes_of(&heap, v), &[0xF0, 0x9F, 0x92, 0xA9]);
        let dec = new_decoder(&mut heap, global, &[]);
        let dp = proto(&heap, global, "TextDecoder");
        let s = call_method(&mut heap, dp, "decode", Value::Object(dec), &[v]);
        assert_eq!(js_str(s), "\u{1F4A9}");
    }

    #[test]
    fn streaming_buffers_partial_sequences() {
        let (mut heap, global) = setup();
        let dec = new_decoder(&mut heap, global, &[]);
        let dp = proto(&heap, global, "TextDecoder");
        let f = native_fn(&heap, dp, "decode");
        let this = Value::Object(dec);
        let stream = opts(&mut heap, &[("stream", true)]);

        // "€" is E2 82 AC; feed the first two bytes with streaming on: the
        // incomplete tail is buffered, nothing is emitted yet.
        let part = uint8_arg(&mut heap, &[0xE2, 0x82]);
        let v = f(&mut heap, this, &[part, stream]).unwrap();
        assert_eq!(js_str(v), "");
        // Final chunk (streaming off): the buffered bytes join the new input
        // and decode together; the queue resets.
        let rest = uint8_arg(&mut heap, &[0xAC]);
        let v = f(&mut heap, this, &[rest]).unwrap();
        assert_eq!(js_str(v), "€");
        // After the reset, decoding starts fresh.
        let hi = uint8_arg(&mut heap, &[104, 105]);
        let v = f(&mut heap, this, &[hi]).unwrap();
        assert_eq!(js_str(v), "hi");
    }

    #[test]
    fn streaming_split_bom_is_still_sniffed() {
        let (mut heap, global) = setup();
        let dec = new_decoder(&mut heap, global, &[]);
        let dp = proto(&heap, global, "TextDecoder");
        let f = native_fn(&heap, dp, "decode");
        let this = Value::Object(dec);
        let stream = opts(&mut heap, &[("stream", true)]);

        let bom_part = uint8_arg(&mut heap, &[0xEF]);
        let v = f(&mut heap, this, &[bom_part, stream]).unwrap();
        assert_eq!(js_str(v), "");
        let rest = uint8_arg(&mut heap, &[0xBB, 0xBF, 104, 105]);
        let v = f(&mut heap, this, &[rest]).unwrap();
        assert_eq!(js_str(v), "hi");
    }

    #[test]
    fn streaming_fatal_defers_to_chunk_completion() {
        let (mut heap, global) = setup();
        let label = arg(&mut heap, "utf-8");
        let options = opts(&mut heap, &[("fatal", true)]);
        let dec = new_decoder(&mut heap, global, &[label, options]);
        let dp = proto(&heap, global, "TextDecoder");
        let f = native_fn(&heap, dp, "decode");
        let this = Value::Object(dec);
        let stream = opts(&mut heap, &[("stream", true)]);

        // A lone lead byte is buffered, not an immediate error …
        let lead = uint8_arg(&mut heap, &[0xE2]);
        let v = f(&mut heap, this, &[lead, stream]).unwrap();
        assert_eq!(js_str(v), "");
        // … but completing it into an invalid sequence throws on the final
        // (non-streaming) call.
        let bad_rest = uint8_arg(&mut heap, &[0x28, 0xAC]);
        assert!(matches!(
            f(&mut heap, this, &[bad_rest]),
            Err(NativeError::TypeError(_))
        ));
    }

    #[test]
    fn bom_bytes_are_not_sniffed_for_other_encodings() {
        let (mut heap, global) = setup();
        let dp = proto(&heap, global, "TextDecoder");

        // gb18030: 0xFE, 0xFF are invalid input, not a reason to switch to
        // UTF-16BE. A lone invalid lead is a single FFFD (the non-ASCII
        // second byte is consumed by the error, not restored).
        let label = arg(&mut heap, "gb18030");
        let dec = new_decoder(&mut heap, global, &[label]);
        let input = uint8_arg(&mut heap, &[0xFE, 0xFF]);
        let v = call_method(&mut heap, dp, "decode", Value::Object(dec), &[input]);
        assert_eq!(js_str(v), "\u{FFFD}");

        // utf-8: a UTF-16LE BOM is not stripped and does not switch the
        // decoding; both bytes are errors.
        let label = arg(&mut heap, "utf-8");
        let dec = new_decoder(&mut heap, global, &[label]);
        let input = uint8_arg(&mut heap, &[0xFF, 0xFE]);
        let v = call_method(&mut heap, dp, "decode", Value::Object(dec), &[input]);
        assert_eq!(js_str(v), "\u{FFFD}\u{FFFD}");

        // windows-1252: a UTF-8 BOM decodes as ordinary bytes (no strip,
        // no switch).
        let label = arg(&mut heap, "windows-1252");
        let dec = new_decoder(&mut heap, global, &[label]);
        let input = uint8_arg(&mut heap, &[0xEF, 0xBB, 0xBF]);
        let v = call_method(&mut heap, dp, "decode", Value::Object(dec), &[input]);
        assert_eq!(js_str(v), "ï»¿");
    }

    #[test]
    fn streaming_invalid_utf8_prefix_is_not_held_back() {
        let (mut heap, global) = setup();
        let dec = new_decoder(&mut heap, global, &[]);
        let dp = proto(&heap, global, "TextDecoder");
        let f = native_fn(&heap, dp, "decode");
        let this = Value::Object(dec);
        let stream = opts(&mut heap, &[("stream", true)]);

        // 0xF0 may start a 4-byte sequence: buffered, nothing emitted.
        let lead = uint8_arg(&mut heap, &[0xF0]);
        let v = f(&mut heap, this, &[lead, stream]).unwrap();
        assert_eq!(js_str(v), "");
        // 0x8F is outside F0's narrowed second-byte range (0x90-0xBF): the
        // sequence is already invalid, so both bytes error at once instead
        // of being held for a third chunk.
        let bad = uint8_arg(&mut heap, &[0x8F]);
        let v = f(&mut heap, this, &[bad, stream]).unwrap();
        assert_eq!(js_str(v), "\u{FFFD}\u{FFFD}");
        // The queue is empty now: the final flush emits nothing further.
        let v = f(&mut heap, this, &[]).unwrap();
        assert_eq!(js_str(v), "");
    }

    #[test]
    fn utf8_trailing_partial_holds_only_completable_prefixes() {
        // Bare leads are held …
        assert_eq!(utf8_trailing_partial(&[0xF0]), 1);
        assert_eq!(utf8_trailing_partial(&[0xE2, 0x82]), 2);
        assert_eq!(utf8_trailing_partial(&[0xF0, 0x90, 0x80]), 3);
        // … but a second byte outside the lead's narrowed range makes the
        // sequence already invalid: nothing is held.
        assert_eq!(utf8_trailing_partial(&[0xF0, 0x8F]), 0);
        assert_eq!(utf8_trailing_partial(&[0xE0, 0x80]), 0);
        assert_eq!(utf8_trailing_partial(&[0xED, 0xA0]), 0);
        assert_eq!(utf8_trailing_partial(&[0xF4, 0x90]), 0);
        // Valid narrowed second bytes are still held …
        assert_eq!(utf8_trailing_partial(&[0xE0, 0xA0]), 2);
        assert_eq!(utf8_trailing_partial(&[0xF4, 0x8F]), 2);
        // … and bytes that can never lead a sequence error immediately.
        assert_eq!(utf8_trailing_partial(&[0xC0]), 0);
        assert_eq!(utf8_trailing_partial(&[0xF5]), 0);
        // An invalid lead followed by a genuine partial sequence holds
        // just the partial tail.
        assert_eq!(utf8_trailing_partial(&[0xE0, 0x41, 0xE2]), 1);
        // Stray continuations and complete tails hold nothing.
        assert_eq!(utf8_trailing_partial(&[0x80, 0x80]), 0);
        assert_eq!(utf8_trailing_partial(&[0xE2, 0x82, 0xAC]), 0);
        assert_eq!(utf8_trailing_partial(&[]), 0);
    }

    #[test]
    fn methods_reject_foreign_receivers() {
        let (mut heap, global) = setup();
        let ep = proto(&heap, global, "TextEncoder");
        let dp = proto(&heap, global, "TextDecoder");
        assert!(matches!(
            native_fn(&heap, ep, "encode")(&mut heap, Value::Undefined, &[]),
            Err(NativeError::TypeError(_))
        ));
        let input = uint8_arg(&mut heap, &[104]);
        assert!(matches!(
            native_fn(&heap, dp, "decode")(&mut heap, Value::Undefined, &[input]),
            Err(NativeError::TypeError(_))
        ));
        assert!(matches!(
            get_getter_result(&mut heap, dp, "encoding", Value::Undefined),
            Err(NativeError::TypeError(_))
        ));
        // … while the real instance works.
        let dec = new_decoder(&mut heap, global, &[]);
        let want = arg(&mut heap, "utf-8");
        assert_eq!(
            get_getter(&mut heap, dp, "encoding", Value::Object(dec)),
            want
        );
    }

    #[test]
    fn partial_install_retries_text_decoder_half() {
        // A half-finished install must not latch: the idempotency probe
        // checks *both* globals, so a realm with `TextEncoder` but no
        // `TextDecoder` (the old probe saw only the first half and no-op'd
        // forever) gets the missing half installed on retry.
        let mut heap = Heap::new();
        let global = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        crate::builtin_reflect_proxy::install(&mut heap, global).unwrap();
        let gbase = heap.protect(Value::Object(global));
        // Simulate the crashed half: only `TextEncoder` present.
        heap.define_property_str(global, "TextEncoder", Property::data(Value::Undefined))
            .unwrap();
        install_encoding(&mut heap, global).unwrap();
        assert!(
            heap.get_property(global, Wtf8::from_str("TextDecoder"))
                .is_some(),
            "retry must install the missing TextDecoder half"
        );
        // The retried install still ends idempotent: a further call keeps
        // the same constructor pair.
        let te = heap
            .get_property(global, Wtf8::from_str("TextEncoder"))
            .unwrap()
            .value;
        let td = heap
            .get_property(global, Wtf8::from_str("TextDecoder"))
            .unwrap()
            .value;
        install_encoding(&mut heap, global).unwrap();
        assert_eq!(
            heap.get_property(global, Wtf8::from_str("TextEncoder"))
                .unwrap()
                .value,
            te
        );
        assert_eq!(
            heap.get_property(global, Wtf8::from_str("TextDecoder"))
                .unwrap()
                .value,
            td
        );
        heap.release_temp(gbase);
    }
}
