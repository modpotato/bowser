//! Wave-33 typed-array foundation: the native `Uint8Array` API.
//!
//! # Layering (read before moving code)
//!
//! The JS-visible `ArrayBuffer` and `Uint8Array` constructors — and the rest
//! of the typed-array family, `SharedArrayBuffer`, and `DataView` — are
//! implemented by `crate::builtin_reflect_proxy` and installed on every VM
//! global by `install_builtins` (`vm.rs`). That module owns constructor
//! semantics, the kind-tag view identity, and `%Uint8Array.prototype%`.
//! This module is the wave-33 enabler layer on top, owning what the
//! `TextEncoder` follow-up needs:
//!
//! * [`uint8array_from_bytes`]: build a spec-correct `Uint8Array` from
//!   native bytes in one fallible allocation (what
//!   `TextEncoder.prototype.encode` returns). The caller passes the
//!   realm's `%Uint8Array.prototype%` (via
//!   [`uint8array_prototype`](crate::builtin_reflect_proxy::uint8array_prototype))
//!   so native-built views proto-link exactly like constructed ones:
//!   `instanceof Uint8Array` holds and the prototype methods (`slice`,
//!   `set`, `subarray`) plus the `buffer` accessor resolve.
//! * View introspection: [`is_uint8_array`], [`uint8_view_parts`],
//!   [`uint8_view_bytes`] / [`uint8_view_bytes_mut`], [`view_buffer`].
//! * The integer-indexed exotic path at the native level: [`uint8_get`]
//!   (0-255 number, OOB reads `undefined`) and [`uint8_set`] (`ToNumber`
//!   then mod 256, OOB writes ignored), delegating to
//!   `builtin_reflect_proxy`'s `typed_index_get` / `typed_index_set`
//!   (which the VM's indexed load/store path routes through).
//!
//! # GC design
//!
//! The backing store is `ObjectCold::bytes: Option<Box<[u8]>>`, allocated in
//! on the buffer object by the collector (`Heap::alloc_object`). It holds no
//! GC handles, so the tracer skips it and it is freed with the object — no
//! separate rooting, pinning, or finalization. A `Uint8Array` view keeps its
//! buffer alive through the [`ObjectKind::TypedArrayView`] tag's buffer
//! handle, so the collector sees the edge. Nothing in this module adds GC surface:
//! every allocation goes through `Heap::alloc_object`, and multi-step
//! builds hold intermediates in [`with_temp_root`](crate::bindings::with_temp_root).

use crate::bindings::with_temp_root;
use crate::builtin_reflect_proxy::{js_num, typed_array_parts, typed_index_get, typed_index_set};
use crate::gc::{Gc, Heap, PropKey, WellKnown};
use crate::value::{NativeError, Object, ObjectKind, Property, TypedArrayKind, Value};
use crate::wtf8::Wtf8;

// ---------------------------------------------------------------------------
// View identity
// ---------------------------------------------------------------------------
//
// View recognition is a single [`ObjectKind`] tag compare: typed-array
// views carry [`ObjectKind::TypedArrayView`] and buffers
// [`ObjectKind::ArrayBuffer`], set at construction by this module and by
// `crate::builtin_reflect_proxy` (the canonical home of the constructor
// semantics). No hidden properties are involved, so user script cannot
// forge a view or tamper with its metadata — the old NUL-prefixed slot
// scheme (`"\u{0}bowser:ta-*"`, `"\u{0}bowser:arraybuffer"`) was writable
// from script and is gone.

/// Non-writable, non-enumerable, non-configurable numeric data property,
/// the shape `new_typed_array` uses for `length`/`byteLength`/`byteOffset`.
fn const_num_prop(n: f64) -> Property {
    Property {
        value: js_num(n),
        writable: false,
        enumerable: false,
        configurable: false,
    }
}

// ---------------------------------------------------------------------------
// View recognition / introspection
// ---------------------------------------------------------------------------

/// True when `v` is a genuine `Uint8Array` view: the
/// [`ObjectKind::TypedArrayView`] tag with the `Uint8` element type. One
/// tag compare — forgery is impossible, so no slot validation is needed.
/// (Test helper; the production indexed path is now kind-generic.)
#[allow(dead_code)]
pub(crate) fn is_uint8_array(heap: &Heap, v: Value) -> bool {
    matches!(
        v,
        Value::Object(o)
            if matches!(
                heap.get(o).kind,
                ObjectKind::TypedArrayView {
                    etype: TypedArrayKind::Uint8,
                    ..
                }
            )
    )
}

/// Decompose a `Uint8Array` view into `(buffer, byte_offset, len_elems)`.
/// Returns `None` for non-views.
pub(crate) fn uint8_view_parts(heap: &Heap, view: Gc<Object>) -> Option<(Gc<Object>, u64, u64)> {
    match heap.get(view).kind {
        ObjectKind::TypedArrayView {
            buffer,
            byte_offset,
            len_elems,
            etype: TypedArrayKind::Uint8,
        } => Some((buffer, u64::from(byte_offset), u64::from(len_elems))),
        _ => None,
    }
}

/// Borrow the view's bytes: `buffer[byte_offset .. byte_offset + length]`.
/// Bounds are clamped defensively; well-formed views always fit.
pub(crate) fn uint8_view_bytes(heap: &Heap, view: Gc<Object>) -> Option<&[u8]> {
    let (buf, offset, length) = uint8_view_parts(heap, view)?;
    let bytes = heap.object_bytes(buf)?;
    let start = usize::try_from(offset).ok()?;
    let len = usize::try_from(length).ok()?;
    let end = start.checked_add(len)?;
    bytes.get(start..end.min(bytes.len()))
}

/// Mutable borrow of the view's bytes (same clamping as
/// [`uint8_view_bytes`]).
pub(crate) fn uint8_view_bytes_mut(heap: &mut Heap, view: Gc<Object>) -> Option<&mut [u8]> {
    let (buf, offset, length) = uint8_view_parts(heap, view)?;
    let start = usize::try_from(offset).ok()?;
    let len = usize::try_from(length).ok()?;
    let end = start.checked_add(len)?;
    // Read the buffer length first so the mutable borrow below does not
    // alias the immutable one.
    let buf_len = heap.object_bytes(buf)?.len();
    heap.object_bytes_mut(buf)?.get_mut(start..end.min(buf_len))
}

/// The view's backing `ArrayBuffer` as a value: an own `buffer` data
/// property when one is present (user-defined shadowing; `DataView`
/// defines one), else the kind tag's buffer link, validated for every
/// typed-array element type (not just `Uint8Array`). Neither typed-array
/// construction path defines an own `buffer` property — like browsers, it
/// lives as an accessor on `%Uint8Array.prototype%` — so those take the
/// tag path.
pub(crate) fn view_buffer(heap: &Heap, view: Gc<Object>) -> Option<Value> {
    if let Some(p) = heap.get_property(view, Wtf8::from_str("buffer")) {
        return Some(p.value);
    }
    typed_array_parts(heap, view).map(|(buf, _, _, _)| Value::Object(buf))
}

/// `(byte_offset, byte_len)` of a genuine view — a typed array or a
/// `DataView` — read from the unforgeable kind tag, never from
/// properties. (`length`/`byteLength`/`byteOffset` live as accessors on
/// `%TypedArray%.prototype%`, so property reads cannot see them.)
pub(crate) fn view_byte_range(heap: &Heap, view: Gc<Object>) -> Option<(u64, u64)> {
    match heap.get(view).kind {
        ObjectKind::TypedArrayView {
            byte_offset,
            len_elems,
            etype,
            ..
        } => Some((
            u64::from(byte_offset),
            u64::from(len_elems) * etype.size() as u64,
        )),
        ObjectKind::DataView {
            byte_offset,
            byte_len,
            ..
        } => Some((u64::from(byte_offset), u64::from(byte_len))),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Integer-indexed exotic get/set (native-level path)
// ---------------------------------------------------------------------------

/// Integer-indexed `[[Get]]` on a `Uint8Array` view: a 0-255 number;
/// out-of-bounds reads are `undefined`. Type-errors on non-views.
/// (Test helper; the production indexed path is now kind-generic.)
#[allow(dead_code)]
pub(crate) fn uint8_get(heap: &mut Heap, view: Gc<Object>, i: u64) -> Result<Value, NativeError> {
    if uint8_view_parts(heap, view).is_none() {
        return Err(NativeError::TypeError(
            "uint8_get on incompatible receiver".into(),
        ));
    }
    typed_index_get(heap, view, i)
}

/// Integer-indexed `[[Set]]` on a `Uint8Array` view: `ToNumber(v)` then
/// mod 256 (`to_int32(n) as u8`, per spec for Uint8 — no clamping; that is
/// `Uint8ClampedArray`). Out-of-bounds writes are silently ignored.
/// Type-errors on non-views.
/// (Test helper; the production indexed path is now kind-generic.)
#[allow(dead_code)]
pub(crate) fn uint8_set(
    heap: &mut Heap,
    view: Gc<Object>,
    i: u64,
    v: Value,
) -> Result<(), NativeError> {
    if uint8_view_parts(heap, view).is_none() {
        return Err(NativeError::TypeError(
            "uint8_set on incompatible receiver".into(),
        ));
    }
    typed_index_set(heap, view, i, v)
}

// ---------------------------------------------------------------------------
// Native construction (the TextEncoder path)
// ---------------------------------------------------------------------------

/// Allocate an `ArrayBuffer` object holding a copy of `bytes`: one fallible
/// zeroed allocation plus one `memcpy`, never one property per byte.
/// Mirrors `new_array_buffer` in `builtin_reflect_proxy` (same
/// [`ObjectKind::ArrayBuffer`] tag, same `byteLength` shape, same 4 GiB
/// cap) so the result interoperates with the JS constructors there.
fn alloc_buffer(heap: &mut Heap, bytes: &[u8]) -> Result<Gc<Object>, NativeError> {
    // Spec (CreateByteDataBlock): throw RangeError if the Data Block cannot
    // be created. 4 GiB matches common implementation limits and keeps
    // `len as usize` exact on 64-bit.
    if bytes.len() > u32::MAX as usize {
        return Err(NativeError::RangeError(
            "ArrayBuffer length exceeds maximum".into(),
        ));
    }
    // Fallible: `vec![0; n]` would abort the process on OOM instead of
    // reporting `AllocError`.
    let mut store = Vec::new();
    store
        .try_reserve_exact(bytes.len())
        .map_err(|_| NativeError::OutOfMemory)?;
    store.resize(bytes.len(), 0);
    store.copy_from_slice(bytes);
    let mut obj = Object {
        kind: ObjectKind::ArrayBuffer,
        ..Default::default()
    };
    obj.set_bytes(Some(store.into_boxed_slice()));
    let buf = heap.alloc_object(obj)?;
    heap.define_property_str(buf, "byteLength", const_num_prop(bytes.len() as f64))?;
    Ok(buf)
}

/// Allocate a `Uint8Array` view over `buf` at `byte_offset` with
/// `len_elems` elements, proto-linked to `proto` (the realm's
/// `%Uint8Array.prototype%`). Same [`ObjectKind::TypedArrayView`] tag and
/// `length`/`byteLength`/`byteOffset` shape as `new_typed_array` in
/// `builtin_reflect_proxy`; like that path — and like browsers — the view
/// carries no own `buffer` data property (it resolves through the
/// prototype's `buffer` accessor).
fn alloc_uint8_view(
    heap: &mut Heap,
    buf: Gc<Object>,
    byte_offset: u64,
    len_elems: u64,
    proto: Value,
) -> Result<Gc<Object>, NativeError> {
    // `alloc_buffer` caps the store at `u32::MAX` bytes; fail closed
    // rather than truncate.
    let byte_offset = u32::try_from(byte_offset)
        .map_err(|_| NativeError::RangeError("typed array: byte offset out of range".into()))?;
    let len_elems = u32::try_from(len_elems)
        .map_err(|_| NativeError::RangeError("typed array: length out of range".into()))?;
    // Spec-style range check mirroring `typed_array_construct`: the view's
    // span must lie within the buffer's data block. The single caller
    // passes consistent values, but an inconsistent view must fail loudly
    // instead of silently. Checked arithmetic so `byte_offset +
    // len_elems * elem_size` can never wrap small and pass the check.
    let buf_len = heap.object_bytes(buf).map(|b| b.len() as u64).unwrap_or(0);
    let view_bytes = (len_elems as u64)
        .checked_mul(TypedArrayKind::Uint8.size() as u64)
        .ok_or_else(|| NativeError::RangeError("typed array view length overflow".into()))?;
    let end = (byte_offset as u64)
        .checked_add(view_bytes)
        .ok_or_else(|| NativeError::RangeError("typed array view length overflow".into()))?;
    if end > buf_len {
        return Err(NativeError::RangeError(
            "typed array view out of buffer range".into(),
        ));
    }
    let view = heap.alloc_object(Object::new(ObjectKind::TypedArrayView {
        buffer: buf,
        byte_offset,
        len_elems,
        etype: TypedArrayKind::Uint8,
    }))?;
    heap.set_prototype(view, proto);
    // `length` / `byteLength` / `byteOffset` are *not* own properties:
    // they resolve through the `get %TypedArray%.prototype.length`
    // (etc.) accessors, per ECMA-262 22.2.3.16–18 — the same accessors
    // the JS-constructed views use, since `proto` chains to the shared
    // `%TypedArray%.prototype`.
    Ok(view)
}

/// Build a spec-correct `Uint8Array` from native bytes: a fresh
/// `ArrayBuffer` copy plus a full-span `Uint8Array` view proto-linked to
/// `proto` — the realm's `%Uint8Array.prototype%`, resolved via
/// [`uint8array_prototype`](crate::builtin_reflect_proxy::uint8array_prototype)
/// — so `instanceof Uint8Array` and the prototype methods work on the
/// result. This is the constructor the `TextEncoder` worker calls for
/// `encode()` results.
///
/// GC safety: `buf` and `proto` are held as temporary roots across the
/// view allocation (which can trigger a collection) and the property
/// writes.
pub(crate) fn uint8array_from_bytes(
    heap: &mut Heap,
    bytes: &[u8],
    proto: Value,
) -> Result<Value, NativeError> {
    let buf = alloc_buffer(heap, bytes)?;
    with_temp_root(heap, Value::Object(buf), |heap| {
        with_temp_root(heap, proto, |heap| {
            let view = alloc_uint8_view(heap, buf, 0, bytes.len() as u64, proto)?;
            Ok(Value::Object(view))
        })
    })
}

// ---------------------------------------------------------------------------
// Per-type `%<T>Array.prototype%` installation
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Per-type `%<T>Array.prototype%` installation
// ---------------------------------------------------------------------------

/// Build and install `%<T>Array.prototype%` for one typed-array
/// constructor: a fresh ordinary object whose [[Prototype]] is the shared
/// `%TypedArray%.prototype` (which carries the method set and the
/// `length`/`byteLength`/`byteOffset`/`buffer` accessors), carrying the
/// per-type `constructor` link and `@@toStringTag`. Linked as the
/// constructor's non-writable, non-enumerable, non-configurable
/// `"prototype"`.
///
/// Called once per constructor from `builtin_reflect_proxy`'s install
/// loop, which owns constructor semantics and the shared prototype.
pub(crate) fn install_typed_array_prototype(
    heap: &mut Heap,
    ctor: Gc<Object>,
    shared_proto: Gc<Object>,
    name: &str,
) -> Result<(), NativeError> {
    let proto = heap.alloc_object(Object::new(ObjectKind::Ordinary))?;
    // [[Prototype]] → the shared `%TypedArray%.prototype`.
    heap.set_prototype(proto, Value::Object(shared_proto));
    // `proto` is not reachable from any traced root until `prototype` is
    // linked below; hold it (and `ctor`) across the allocating definitions.
    // (Closure return types are annotated: `with_temp_root` is generic
    // over the error type and `?` on `AllocError` results would otherwise
    // leave it ambiguous.)
    with_temp_root(heap, Value::Object(proto), |heap| {
        with_temp_root(
            heap,
            Value::Object(ctor),
            |heap| -> Result<(), NativeError> {
                // `%<T>Array.prototype%.constructor` (writable,
                // non-enumerable, configurable, per ECMA-262 22.2.6).
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
                // `%<T>Array.prototype%[@@toStringTag]` (ECMA-262 22.2.4):
                // the real well-known symbol key, never an `"@@…"` string.
                let tag = heap
                    .well_known_symbol(WellKnown::ToStringTag)
                    .map_err(|_| NativeError::OutOfMemory)?;
                let tag_str = with_temp_root(heap, Value::Object(proto), |heap| {
                    heap.alloc_str(name).map_err(|_| NativeError::OutOfMemory)
                })?;
                heap.define_property_key(
                    proto,
                    PropKey::Symbol(tag),
                    Property {
                        value: Value::Str(tag_str),
                        writable: false,
                        enumerable: false,
                        configurable: true,
                    },
                )
                .map_err(|_| NativeError::OutOfMemory)?;
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
                Ok(())
            },
        )
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin_object_array::to_string_js;
    use crate::vm::Vm;

    /// Read a `%TypedArray%.prototype` accessor (`length` / `byteLength` /
    /// `byteOffset` / `buffer`) on a view: walk the prototype chain for
    /// the accessor and invoke its getter with the view as receiver.
    fn ta_acc(heap: &mut Heap, view: Gc<Object>, name: &str) -> Value {
        let mut current = Value::Object(view);
        loop {
            let o = match current {
                Value::Object(o) => o,
                _ => panic!("ta_acc: no accessor '{name}' on the chain"),
            };
            if let Some(acc) = heap.get_accessor(o, Wtf8::from_str(name)) {
                let Value::Object(f) = acc.get else {
                    panic!("ta_acc: '{name}' has no getter");
                };
                let ObjectKind::NativeFunction(nf) = heap.get(f).kind else {
                    panic!("ta_acc: '{name}' getter is not native");
                };
                return nf(heap, Value::Object(view), &[]).unwrap();
            }
            current = heap.get(o).prototype;
        }
    }

    /// Heap + global with the real builtins installed (the same install
    /// `Vm::new` performs).
    fn setup() -> (Heap, Value) {
        let mut heap = Heap::new();
        let global = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        crate::builtin_reflect_proxy::install(&mut heap, global).unwrap();
        (heap, Value::Object(global))
    }

    fn eval_str(source: &str) -> String {
        let mut vm = Vm::new();
        let v = vm
            .eval(source)
            .unwrap_or_else(|e| panic!("eval failed: {e}"));
        to_string_js(vm.heap_mut(), v)
            .map(|w| w.to_string())
            .unwrap_or_else(|e| panic!("to_string failed: {e:?}"))
    }

    /// The installed `%Uint8Array.prototype%` for `uint8array_from_bytes`.
    fn u8_proto(heap: &Heap, global: Value) -> Value {
        let Value::Object(g) = global else {
            panic!("global not an object")
        };
        crate::builtin_reflect_proxy::uint8array_prototype(heap, g)
    }

    /// Walk a property path from a value (mirrors the sibling module).
    fn mget(heap: &Heap, mut v: Value, path: &[&str]) -> Value {
        for p in path {
            let o = match v {
                Value::Object(o) => o,
                _ => panic!("mget: not an object at '{p}'"),
            };
            v = crate::builtin_reflect_proxy::get_prop_chain(heap, o, Wtf8::from_str(p))
                .unwrap_or_else(|| panic!("mget: missing '{p}'"));
        }
        v
    }

    fn call_path(
        heap: &mut Heap,
        global: Value,
        path: &[&str],
        this: Value,
        args: &[Value],
    ) -> Result<Value, NativeError> {
        let f = mget(heap, global, path);
        match f {
            Value::Object(o) => match heap.get(o).kind {
                ObjectKind::NativeFunction(nf) => nf(heap, this, args),
                _ => panic!("not a native function"),
            },
            _ => panic!("not a function"),
        }
    }

    fn require_object(v: Value, name: &str) -> Gc<Object> {
        match v {
            Value::Object(o) => o,
            _ => panic!("{name} not an object"),
        }
    }

    #[test]
    fn uint8array_length_and_bytelength() {
        assert_eq!(eval_str("new Uint8Array(3).length"), "3");
        assert_eq!(eval_str("new Uint8Array(3).byteLength"), "3");
        assert_eq!(eval_str("new Uint8Array(0).length"), "0");
    }

    #[test]
    fn uint8array_from_arraybuffer_with_offset() {
        let r = eval_str(
            "var b = new ArrayBuffer(8); \
             var v = new Uint8Array(b, 2, 3); \
             v.length + ',' + v.byteLength + ',' + v.byteOffset",
        );
        assert_eq!(r, "3,3,2");
        // Defaulted length spans to the end of the buffer.
        let r = eval_str(
            "var v = new Uint8Array(new ArrayBuffer(8), 6); v.length + ',' + v.byteLength",
        );
        assert_eq!(r, "2,2");
        // Out-of-range offset throws RangeError.
        let r = eval_str(
            "var ok = false; \
             try { new Uint8Array(new ArrayBuffer(8), 9); } \
             catch (e) { ok = e instanceof RangeError; } ok",
        );
        assert_eq!(r, "true");
    }

    #[test]
    fn indexed_set_get_mod_256() {
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        let v = call_path(&mut heap, global, &["Uint8Array"], undef, &[Value::Int(2)]).unwrap();
        let view = require_object(v, "view");
        assert!(is_uint8_array(&heap, v));

        uint8_set(&mut heap, view, 0, Value::Int(300)).unwrap();
        assert_eq!(uint8_get(&mut heap, view, 0).unwrap(), Value::Int(44)); // 300 mod 256
        uint8_set(&mut heap, view, 1, Value::Int(-1)).unwrap();
        assert_eq!(uint8_get(&mut heap, view, 1).unwrap(), Value::Int(255));
        // Floats go through ToNumber then ToInt32 mod 256.
        uint8_set(&mut heap, view, 0, Value::Float(257.9)).unwrap();
        assert_eq!(uint8_get(&mut heap, view, 0).unwrap(), Value::Int(1));
        // Non-views fail closed.
        let plain = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        assert!(matches!(
            uint8_set(&mut heap, plain, 0, Value::Int(1)),
            Err(NativeError::TypeError(_))
        ));
        assert!(matches!(
            uint8_get(&mut heap, plain, 0),
            Err(NativeError::TypeError(_))
        ));
    }

    #[test]
    fn oob_indexed_read_is_undefined() {
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        let v = call_path(&mut heap, global, &["Uint8Array"], undef, &[Value::Int(2)]).unwrap();
        let view = require_object(v, "view");
        assert_eq!(uint8_get(&mut heap, view, 2).unwrap(), Value::Undefined);
        assert_eq!(
            uint8_get(&mut heap, view, u64::MAX).unwrap(),
            Value::Undefined
        );
        // OOB writes are silently ignored.
        uint8_set(&mut heap, view, 5, Value::Int(9)).unwrap();
        assert_eq!(uint8_get(&mut heap, view, 0).unwrap(), Value::Int(0));
    }

    #[test]
    fn arraybuffer_negative_length_throws_rangeerror() {
        let r = eval_str(
            "var ok = false; \
             try { new ArrayBuffer(-1); } \
             catch (e) { ok = e instanceof RangeError; } ok",
        );
        assert_eq!(r, "true");
        // ToIndex per spec: NaN -> +0, fractions truncate; only negatives
        // and > 2^53-1 throw.
        let r = eval_str(
            "var b1 = new ArrayBuffer(NaN); \
             var b2 = new ArrayBuffer(2.5); \
             var ok = b1.byteLength === 0 && b2.byteLength === 2; \
             try { new ArrayBuffer(2 ** 53); ok = false; } \
             catch (e) { ok = ok && e instanceof RangeError; } \
             ok",
        );
        assert_eq!(r, "true");
    }

    #[test]
    fn arraybuffer_slice_copies() {
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        let b = call_path(&mut heap, global, &["ArrayBuffer"], undef, &[Value::Int(4)]).unwrap();
        let buf = require_object(b, "buf");
        // Fill natively (the JS indexed path is not wired yet; see docs).
        heap.object_bytes_mut(buf)
            .unwrap()
            .copy_from_slice(&[10, 20, 30, 40]);

        let slice_fn = mget(&heap, global, &["ArrayBuffer", "prototype", "slice"]);
        let s = match slice_fn {
            Value::Object(o) => match heap.get(o).kind {
                ObjectKind::NativeFunction(nf) => {
                    nf(&mut heap, b, &[Value::Int(1), Value::Int(3)]).unwrap()
                }
                _ => panic!("slice not native"),
            },
            _ => panic!("no slice"),
        };
        let sbuf = require_object(s, "slice");
        assert_eq!(
            heap.object_bytes(sbuf).unwrap(),
            &[20, 30],
            "slice(1,3) copies bytes 1..3"
        );
        // The copy is independent: mutating the original leaves it alone.
        heap.object_bytes_mut(buf).unwrap()[1] = 99;
        assert_eq!(heap.object_bytes(sbuf).unwrap(), &[20, 30]);
        // byteLength of the slice.
        assert_eq!(mget(&heap, s, &["byteLength"]), Value::Int(2));
    }

    #[test]
    fn uint8array_from_bytes_roundtrip() {
        let (mut heap, global) = setup();
        let proto = u8_proto(&heap, global);
        assert!(
            matches!(proto, Value::Object(_)),
            "install must provide %Uint8Array.prototype%"
        );
        let v = uint8array_from_bytes(&mut heap, b"hello", proto).unwrap();
        let view = require_object(v, "view");
        assert!(is_uint8_array(&heap, v));
        // Proto-linked to the installed `%Uint8Array.prototype%`, not null.
        assert_eq!(heap.get(view).prototype, proto);
        // No own `buffer` data property: like browsers, it resolves through
        // the prototype's `buffer` accessor.
        assert!(heap.get_property(view, Wtf8::from_str("buffer")).is_none());
        assert_eq!(uint8_view_bytes(&heap, view), Some(b"hello".as_slice()));
        // length / byteLength / byteOffset resolve through the shared
        // `%TypedArray%.prototype` accessors (no longer own data
        // properties, per ECMA-262 22.2.3.16-18).
        for name in ["length", "byteLength", "byteOffset"] {
            assert!(
                heap.get_property(view, Wtf8::from_str(name)).is_none(),
                "{name} must not be an own data property"
            );
        }
        assert_eq!(ta_acc(&mut heap, view, "length"), Value::Int(5));
        assert_eq!(ta_acc(&mut heap, view, "byteLength"), Value::Int(5));
        assert_eq!(ta_acc(&mut heap, view, "byteOffset"), Value::Int(0));
        let buf_val = view_buffer(&heap, view).expect("buffer");
        let buf = require_object(buf_val, "buffer");
        assert_eq!(heap.object_bytes(buf), Some(b"hello".as_slice()));
        assert_eq!(
            crate::builtin_reflect_proxy::get_prop_chain(&heap, buf, Wtf8::from_str("byteLength")),
            Some(Value::Int(5))
        );
        // Indexed read through the exotic path.
        assert_eq!(
            uint8_get(&mut heap, view, 1).unwrap(),
            Value::Int(b'e' as i32)
        );
        // Mutable view borrow writes through to the buffer.
        uint8_view_bytes_mut(&mut heap, view).unwrap()[0] = b'H';
        assert_eq!(heap.object_bytes(buf).unwrap()[0], b'H');
    }

    #[test]
    fn uint8array_from_array_like_copies_mod_256() {
        let r = eval_str(
            "var v = new Uint8Array([300, -1, 257.9]); \
             v.length + ':' + v.byteLength",
        );
        assert_eq!(r, "3:3");
        // Values copied mod 256, checked natively (JS index reads are not
        // wired yet; see module docs).
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        // Build the array-like natively: {0:300, 1:-1, length:2}.
        let src = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.define_property_str(src, "0", Property::data(Value::Int(300)))
            .unwrap();
        heap.define_property_str(src, "1", Property::data(Value::Int(-1)))
            .unwrap();
        heap.define_property_str(src, "length", Property::data(Value::Int(2)))
            .unwrap();
        let v = call_path(
            &mut heap,
            global,
            &["Uint8Array"],
            undef,
            &[Value::Object(src)],
        )
        .unwrap();
        let view = require_object(v, "view");
        assert_eq!(uint8_get(&mut heap, view, 0).unwrap(), Value::Int(44));
        assert_eq!(uint8_get(&mut heap, view, 1).unwrap(), Value::Int(255));
    }

    #[test]
    fn constructed_views_share_the_buffer() {
        // A view built by `uint8array_from_bytes` is recognized by the
        // sibling module's `ArrayBuffer.isView` and its buffer works as a
        // constructor argument for other views.
        let (mut heap, global) = setup();
        let undef = Value::Undefined;
        let proto = u8_proto(&heap, global);
        let v = uint8array_from_bytes(&mut heap, &[1, 2, 3, 4], proto).unwrap();
        let is_view =
            call_path(&mut heap, global, &["ArrayBuffer", "isView"], undef, &[v]).unwrap();
        assert_eq!(is_view, Value::Bool(true));
        let buf = view_buffer(&heap, require_object(v, "view")).unwrap();
        let v2 = call_path(&mut heap, global, &["Uint8Array"], undef, &[buf]).unwrap();
        let view2 = require_object(v2, "view2");
        assert_eq!(uint8_get(&mut heap, view2, 3).unwrap(), Value::Int(4));
    }

    #[test]
    fn encoder_output_is_uint8array_instance() {
        // The required end-to-end check:
        // `new TextEncoder().encode("hi") instanceof Uint8Array === true`.
        let mut vm = Vm::new();
        let global = match vm.get_global("globalThis") {
            Some(Value::Object(g)) => g,
            _ => panic!("no globalThis"),
        };
        crate::bindings_encoding::install_encoding(vm.heap_mut(), global).unwrap();
        let v = vm
            .eval("new TextEncoder().encode(\"hi\") instanceof Uint8Array")
            .unwrap_or_else(|e| panic!("eval failed: {e}"));
        assert_eq!(v, Value::Bool(true));
        // The prototype methods the finding calls out resolve on encoder
        // output, through the real `%Uint8Array.prototype%`.
        let v = vm
            .eval(
                "var e = new TextEncoder().encode(\"hi\"); \
                 e.slice(0, 1).length + ',' + e.subarray(1).length",
            )
            .unwrap_or_else(|e| panic!("eval failed: {e}"));
        let s = to_string_js(vm.heap_mut(), v)
            .map(|w| w.to_string())
            .unwrap();
        assert_eq!(s, "1,1");
        // `set` writes through (and JS indexed reads hit the exotic path),
        // and `buffer` resolves via the prototype accessor.
        let v = vm
            .eval(
                "var e = new TextEncoder().encode(\"hi\"); \
                 e.set([65], 0); \
                 (e[0] === 65) + ',' + (e.buffer.byteLength === 2)",
            )
            .unwrap_or_else(|e| panic!("eval failed: {e}"));
        let s = to_string_js(vm.heap_mut(), v)
            .map(|w| w.to_string())
            .unwrap();
        assert_eq!(s, "true,true");
    }

    #[test]
    fn install_typed_array_prototype_links_shared_proto_and_constructor() {
        // The per-type `%<T>Array.prototype%` builder: fresh prototype
        // object per constructor, [[Prototype]] → the shared
        // `%TypedArray%.prototype`, `constructor` + `@@toStringTag`
        // installed, `prototype` linked non-writable on the constructor.
        // Methods and the accessor family live on the shared prototype
        // (installed by `builtin_reflect_proxy`), not here.
        let (mut heap, _global) = setup();
        let shared = require_object(
            mget(&heap, _global, &["Uint8Array", "prototype"]),
            "u8 proto",
        );
        let shared_proto = heap.get(shared).prototype;
        let ctor = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        install_typed_array_prototype(
            &mut heap,
            ctor,
            require_object(shared_proto, "shared %TypedArray%.prototype"),
            "TestArray",
        )
        .unwrap();
        let proto = match heap.get_property(ctor, Wtf8::from_str("prototype")) {
            Some(p) => match p.value {
                Value::Object(o) => o,
                v => panic!("prototype not an object: {v:?}"),
            },
            None => panic!("no prototype installed"),
        };
        // Distinct from `%Uint8Array.prototype%`, chained to the shared
        // `%TypedArray%.prototype%`.
        assert_ne!(proto, shared);
        assert_eq!(heap.get(proto).prototype, shared_proto);
        // `constructor` → the constructor (writable, non-enumerable,
        // configurable, per ECMA-262).
        let desc = heap
            .get_property(proto, Wtf8::from_str("constructor"))
            .expect("constructor");
        assert_eq!(desc.value, Value::Object(ctor));
        assert!(desc.writable && !desc.enumerable && desc.configurable);
        // `@@toStringTag` under the real well-known symbol key.
        let tag = heap.well_known_symbol(WellKnown::ToStringTag).unwrap();
        let tag_prop = heap
            .get_property_key(proto, PropKey::Symbol(tag))
            .expect("@@toStringTag");
        assert_eq!(
            tag_prop.value,
            Value::Str(heap.alloc_str("TestArray").unwrap())
        );
        // Methods resolve through the shared prototype.
        let f = crate::builtin_reflect_proxy::get_prop_chain(&heap, proto, Wtf8::from_str("slice"))
            .unwrap_or_else(|| panic!("missing slice"));
        assert!(matches!(
            heap.get(require_object(f, "slice")).kind,
            ObjectKind::NativeFunction(_)
        ));
        // `prototype` is non-writable on the constructor.
        let desc = heap
            .get_property(ctor, Wtf8::from_str("prototype"))
            .expect("prototype");
        assert!(!desc.writable && !desc.enumerable && !desc.configurable);
    }
}
