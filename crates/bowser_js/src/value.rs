//! JavaScript value representation.
//!
//! [`Value`] is the tagged union every part of the engine passes around.
//! Heap-allocated payloads ([`GcStr`], [`Object`], [`Symbol`], [`BigInt`])
//! are referenced through [`Gc`](crate::gc::Gc) handles owned by
//! [`Heap`](crate::gc::Heap). See the "Rooting discipline" section in
//! [`crate::gc`] before holding a [`Value`] across an allocation point.

use crate::gc::{AllocError, Gc, Trace, Tracer};
use crate::wtf8::{Wtf8, Wtf8Buf};
use std::ptr::NonNull;

impl From<AllocError> for NativeError {
    fn from(_: AllocError) -> Self {
        Self::OutOfMemory
    }
}

/// A JavaScript value. `Copy` so it moves through the VM cheaply.
///
/// Tagging note: v1 tags values with a Rust enum discriminant (a `Value` is
/// currently wider than 64 bits). NaN-boxing the `Float(f64)` payload into
/// a single 64-bit word is explicitly deferred to a later optimization
/// pass; the enum shape of the public API stays the same either way.
///
/// Payload handles (`Object`, `Symbol`, `BigInt`, and the [`Str::Heap`]
/// variant) are only valid while their allocation is live; see the rooting
/// discipline in [`crate::gc`]. In particular, the payload-inspecting
/// methods on `Value` assume every handle they touch is live. Inline
/// [`Str`] payloads live in the value itself and need no rooting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Undefined,
    Null,
    Bool(bool),
    /// Small integer fast path. Arbitrary-precision integers are [`BigInt`].
    Int(i32),
    Float(f64),
    Str(Str),
    Object(Gc<Object>),
    Symbol(Gc<Symbol>),
    BigInt(Gc<BigInt>),
}

// Tripwire: `Value` must stay 24 bytes (tag + 16-byte payload). The
// 15-byte [`STR_INLINE_MAX`] keeps [`Str`]'s inline variant at 16 bytes,
// matching the [`Gc`] handle size; if this fires, the inline string
// budget grew (or a new variant did) and every stack slot, call-arg vec,
// dense element, environment slot, and property value just got 16 bytes
// wider.
const _: () = assert!(std::mem::size_of::<Value>() <= 24);

impl Value {
    /// True for `Value::Undefined`.
    pub fn is_undefined(self) -> bool {
        matches!(self, Value::Undefined)
    }

    /// True for `Value::Null`.
    pub fn is_null(self) -> bool {
        matches!(self, Value::Null)
    }

    /// JavaScript truthiness (`ToBoolean`).
    ///
    /// # Rooting
    /// Inspects string/bigint payloads; the handle must be live.
    pub fn is_truthy(self) -> bool {
        match self {
            Value::Undefined | Value::Null => false,
            Value::Bool(b) => b,
            Value::Int(i) => i != 0,
            // +0, -0 and NaN are falsy.
            Value::Float(f) => f != 0.0 && !f.is_nan(),
            Value::Str(s) => !s.is_empty(),
            Value::Object(_) => true,
            Value::Symbol(_) => true,
            Value::BigInt(b) => !crate::gc::bigint_is_zero(b),
        }
    }

    /// The result of the `typeof` operator.
    ///
    /// # Rooting
    /// Inspects the object header for `Value::Object`; the handle must be live.
    pub fn type_of(self) -> &'static str {
        match self {
            Value::Undefined => "undefined",
            // typeof null === "object", per spec.
            Value::Null => "object",
            Value::Bool(_) => "boolean",
            Value::Int(_) | Value::Float(_) => "number",
            Value::Str(_) => "string",
            Value::Symbol(_) => "symbol",
            Value::BigInt(_) => "bigint",
            Value::Object(o) => match crate::gc::object_kind(o) {
                ObjectKind::Function(_) | ObjectKind::NativeFunction(_) => "function",
                _ => "object",
            },
        }
    }
}

impl From<Str> for Value {
    fn from(s: Str) -> Value {
        Value::Str(s)
    }
}

impl From<Gc<Object>> for Value {
    fn from(g: Gc<Object>) -> Value {
        Value::Object(g)
    }
}

impl From<Gc<Symbol>> for Value {
    fn from(g: Gc<Symbol>) -> Value {
        Value::Symbol(g)
    }
}

impl From<Gc<BigInt>> for Value {
    fn from(g: Gc<BigInt>) -> Value {
        Value::BigInt(g)
    }
}

// SAFETY: `t.mark_value` reports exactly the `Gc` handle held by the
// value's variant (or none for non-heap variants, including inline
// [`Str`]s, which live in the value itself), satisfying `Trace`'s
// contract that every owned handle is reported.
unsafe impl Trace for Value {
    fn trace(&self, t: &mut Tracer) {
        t.mark_value(*self);
    }
}

/// Maximum byte length of a string stored inline in [`Str`]: 15 bytes of
/// payload plus the length tag keep the inline variant at 16 bytes, so
/// [`Str`] (and hence [`Value`]) stays small. Every VM stack slot,
/// call-arg vec, dense element, environment slot, and [`Property`] value
/// pays for `Value`'s size, so this budget is load-bearing: see the
/// size tripwire below [`Value`].
pub const STR_INLINE_MAX: usize = 15;

/// A JavaScript string: a sequence of UTF-16 code units, stored as WTF-8
/// (see [`crate::wtf8`]) so lone surrogates survive instead of being
/// replaced with U+FFFD at ingress.
///
/// Short strings (at most [`STR_INLINE_MAX`] bytes) are stored inline in
/// the value itself: no heap allocation, no GC tracing, and `Copy` for
/// free. Longer strings live in a heap-allocated [`GcStr`], exactly as
/// before.
///
/// Equality and hashing are by *content* either way: an inline `"ab"` and
/// a heap `"ab"` compare and hash equal. (The old `Value::Str` compared
/// heap handles by pointer identity; string equality in the VM was already
/// by content, and [`same_value_zero`](crate::builtin_collections) too —
/// this just makes `Value`'s own `PartialEq` agree.)
#[derive(Clone, Copy)]
pub enum Str {
    /// Bytes stored directly in the value. `bytes[..len as usize]` holds
    /// the WTF-8 content; the tail is zeroed. Build with [`Str::inline`]
    /// (for `&str`) or [`Str::inline_wtf8`] (for lone-surrogate content).
    Inline {
        len: u8,
        bytes: [u8; STR_INLINE_MAX],
    },
    /// Longer string on the GC heap.
    Heap(Gc<GcStr>),
}

impl Str {
    /// Pack `s` inline, or `None` when it is too long (use
    /// [`Heap::alloc_str`](crate::gc::Heap::alloc_str), which inlines
    /// automatically, or a heap allocation for longer strings).
    ///
    /// `&str` input is valid UTF-8, hence valid WTF-8: the bytes are
    /// stored as-is.
    pub fn inline(s: &str) -> Option<Str> {
        Self::inline_wtf8(Wtf8::from_str(s))
    }

    /// Pack WTF-8 content inline, or `None` when it is too long. The
    /// lone-surrogate-aware counterpart of [`Str::inline`].
    pub fn inline_wtf8(s: &Wtf8) -> Option<Str> {
        if s.len() > STR_INLINE_MAX {
            return None;
        }
        let mut bytes = [0u8; STR_INLINE_MAX];
        bytes[..s.len()].copy_from_slice(s.as_bytes());
        Some(Str::Inline {
            len: s.len() as u8,
            bytes,
        })
    }

    /// The string's contents as WTF-8.
    ///
    /// For [`Str::Heap`] this derefs the handle: the allocation must be
    /// live (see the rooting discipline in [`crate::gc`]).
    ///
    /// This replaced `as_str() -> &str`: a JS string may hold lone
    /// surrogates, which are not valid UTF-8, so no `&str` view exists.
    /// Use [`Wtf8::as_utf8`] (borrowed when well-formed) or
    /// [`Wtf8::to_utf8_lossy`] at Rust-`&str` boundaries.
    pub fn as_wtf8(&self) -> &Wtf8 {
        match self {
            Str::Inline { len, bytes } => {
                let bytes = &bytes[..(*len as usize).min(STR_INLINE_MAX)];
                // The constructors store exactly `len` bytes of valid
                // WTF-8 here; a hand-built variant with garbage fails
                // loudly instead of causing UB.
                Wtf8::from_bytes(bytes).expect("Str::Inline holds valid WTF-8")
            }
            Str::Heap(g) => g.as_wtf8(),
        }
    }

    /// Byte length of the WTF-8 encoding (not the JS `.length`, which
    /// counts UTF-16 code units — see [`Str::units_len`]).
    pub fn len(&self) -> usize {
        match self {
            Str::Inline { len, .. } => *len as usize,
            Str::Heap(g) => g.len_bytes(),
        }
    }

    /// Length in UTF-16 code units: the JS `.length` semantic.
    pub fn units_len(&self) -> usize {
        self.as_wtf8().units_len()
    }

    /// True when the string has no bytes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True when the string is stored inline (no heap allocation, no GC
    /// tracing needed).
    pub fn is_inline(&self) -> bool {
        matches!(self, Str::Inline { .. })
    }

    /// The heap handle for [`Str::Heap`], or `None` for inline strings.
    /// Used by the collector: inline strings have nothing to mark.
    pub fn heap_handle(self) -> Option<Gc<GcStr>> {
        match self {
            Str::Heap(g) => Some(g),
            Str::Inline { .. } => None,
        }
    }

    /// Wrap a heap handle in a [`Str`]. Explicit on purpose: every heap
    /// string the VM touches must be rooted (see the rooting discipline
    /// in [`crate::gc`]), so the conversion is never implicit.
    pub fn heap(g: Gc<GcStr>) -> Str {
        Str::Heap(g)
    }
}

impl PartialEq for Str {
    fn eq(&self, other: &Self) -> bool {
        // Pointer-identity fast path: two handles to the same heap string
        // (the common case for interned constants and repeated references)
        // are equal without comparing bytes.
        if let (Str::Heap(a), Str::Heap(b)) = (self, other)
            && a == b
        {
            return true;
        }
        self.as_wtf8() == other.as_wtf8()
    }
}

impl Eq for Str {}

impl std::hash::Hash for Str {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_wtf8().hash(state);
    }
}

impl std::fmt::Debug for Str {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Print the contents (quoted), not the representation: far more
        // useful in `Value` debug output and test failures.
        std::fmt::Debug::fmt(self.as_wtf8(), f)
    }
}

/// An immutable JavaScript string stored on the GC heap.
///
/// Immutability is what makes [`Heap::intern`](crate::gc::Heap::intern) sound:
/// two handles to equal contents can be deduplicated to one allocation.
///
/// Strings longer than [`STR_INLINE_MAX`] bytes live here; shorter ones
/// are packed inline in [`Str`] and never allocated.
///
/// The bytes are WTF-8 (see [`crate::wtf8`]), so lone surrogates survive.
/// The header is followed in the same allocation block by the payload
/// bytes (see [`Heap`](crate::gc::Heap)'s string arena): a `GcStr` never
/// owns a separate byte allocation. The fields are crate-visible so the
/// collector can read the size class back on sweep; construct only via
/// [`GcStr::from_buf`], which upholds the invariant that the trailing
/// bytes are valid WTF-8 (inherited from the validated [`Wtf8`] source).
#[repr(C)]
#[derive(Debug)]
pub struct GcStr {
    /// Byte length of the trailing WTF-8 payload.
    pub(crate) len: u32,
    /// String-arena size-class index of the block backing this header
    /// (`u8::MAX` for malloc-fallback blocks above the top class; see the
    /// string arena in [`crate::gc`]).
    pub(crate) class: u8,
}

impl GcStr {
    /// Header byte size: the payload follows the header in the block.
    pub(crate) const HEADER_LEN: usize = std::mem::size_of::<GcStr>();
    /// Header alignment: every string-arena block is aligned to this.
    pub(crate) const HEADER_ALIGN: usize = std::mem::align_of::<GcStr>();

    /// The only constructor: initialize the header at `block` and copy
    /// the validated WTF-8 bytes into the trailing storage. Returns the
    /// header pointer (the block's base), which the heap registers.
    ///
    /// # Safety
    ///
    /// * `block` must be the base of a live string-arena block of class
    ///   `class` with at least `HEADER_LEN + src.len()` usable bytes,
    ///   derived from the chunk's (or fallback malloc's) allocation, so
    ///   the header write and the payload copy stay in-bounds.
    /// * The block must have no other live owner (fresh from the bump
    ///   cursor or reclaimed onto a freelist and not yet reused).
    pub(crate) unsafe fn from_buf(block: NonNull<u8>, class: u8, src: &Wtf8) -> NonNull<GcStr> {
        let len = src.len();
        // 4 GiB strings are absurd, but fail loudly instead of truncating.
        assert!(len <= u32::MAX as usize, "string payload exceeds 4 GiB");
        let header = block.as_ptr() as *mut GcStr;
        // SAFETY: the caller guarantees the block; the header fields are
        // written through raw pointers without forming a reference, and
        // the payload copy stays inside the block's usable bytes.
        unsafe {
            std::ptr::addr_of_mut!((*header).len).write(len as u32);
            std::ptr::addr_of_mut!((*header).class).write(class);
            std::ptr::copy_nonoverlapping(
                src.as_bytes().as_ptr(),
                block.as_ptr().add(Self::HEADER_LEN),
                len,
            );
        }
        // SAFETY: `block` is non-null by construction.
        unsafe { NonNull::new_unchecked(header) }
    }

    /// The contents as WTF-8. The allocation must be live (see
    /// [`Str::as_wtf8`]).
    pub fn as_wtf8(&self) -> &Wtf8 {
        // SAFETY: constructed only via `from_buf`, which copies exactly
        // `len` valid WTF-8 bytes into the trailing storage.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (self as *const GcStr).cast::<u8>().add(Self::HEADER_LEN),
                self.len as usize,
            )
        };
        // SAFETY: see above.
        unsafe { Wtf8::from_bytes_unchecked(bytes) }
    }

    /// Byte length of the WTF-8 encoding.
    pub fn len_bytes(&self) -> usize {
        self.len as usize
    }
}

/// A JavaScript `Symbol`. Unique per allocation; the `id` distinguishes
/// symbols with equal descriptions.
#[derive(Debug)]
pub struct Symbol {
    /// May hold lone surrogates (`Symbol("\uD800").description`).
    pub description: Option<Wtf8Buf>,
    pub id: u64,
}

/// A JavaScript `BigInt`: sign plus base-2^32 magnitude, least significant
/// digit first. Minimal representation here; arithmetic lives in builtins.
#[derive(Debug)]
pub struct BigInt {
    pub negative: bool,
    pub digits: Vec<u32>,
}

/// What an [`Object`] behaves as. The VM fills `Function`'s bytecode;
/// builtins implement host behavior through [`NativeFunction`].
// Function-pointer identity comparison is the intended semantic here (the
// same builtin function item means the same builtin).
#[allow(unpredictable_function_pointer_comparisons)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ObjectKind {
    Ordinary,
    /// A bytecode-compiled function. Shape only; the VM owns the bytecode.
    Function(FunctionData),
    /// A host-implemented callable (builtins).
    NativeFunction(NativeFn),
    /// Marker kind; indexed elements are managed via indexed properties.
    Array,
    /// A `RegExp` instance. `source`, `flags`, and `lastIndex` live as
    /// ordinary data properties; matching always recompiles the pattern
    /// (see `crate::builtin_regexp`).
    RegExp,
    /// A `Date` instance. The time value (ms since the epoch, `NaN` for an
    /// invalid date) is carried in the tag as raw `f64` bits — unforgeable,
    /// unlike the old hidden `"\u{0}bowser:date-value"` data property
    /// (see `crate::builtin_date`).
    Date {
        value_bits: u64,
    },
    /// A typed-array view (`Int8Array` … `Float64Array`). The buffer link,
    /// byte offset, element count, and element type live in the tag, so
    /// user script cannot forge or tamper with view metadata (the old
    /// `"\u{0}bowser:ta-*"` hidden data properties were writable from
    /// script). The tracer follows `buffer` (see `Tracer::drain` in
    /// `crate::gc`).
    TypedArrayView {
        buffer: Gc<Object>,
        byte_offset: u32,
        len_elems: u32,
        etype: TypedArrayKind,
    },
    /// An `ArrayBuffer` (or `SharedArrayBuffer`) instance. Byte storage
    /// lives in [`Object::bytes`]; the tag alone is the brand check
    /// (replacing the old hidden `"\u{0}bowser:arraybuffer"` marker).
    ArrayBuffer,
    /// A `DataView` instance. Buffer link, byte offset, and byte length
    /// live in the tag (replacing the old `"\u{0}bowser:dv-*"` hidden
    /// properties). The tracer follows `buffer`.
    DataView {
        buffer: Gc<Object>,
        byte_offset: u32,
        byte_len: u32,
    },
}

/// Element kinds of the numeric typed arrays (`Int8Array` …
/// `Float64Array`, plus the `BigInt64Array`/`BigUint64Array` bigint
/// flavors). Lives here (rather than in a builtin module) so the
/// [`ObjectKind::TypedArrayView`] tag can name the element type without a
/// forgeable hidden-property lookup. Crate-internal: no external crate
/// names the tag type, so tag forgery from outside is impossible.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TypedArrayKind {
    Int8,
    Uint8,
    Uint8Clamped,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Float32,
    Float64,
    BigInt64,
    BigUint64,
}

impl TypedArrayKind {
    /// Element size in bytes.
    pub(crate) fn size(self) -> usize {
        match self {
            TypedArrayKind::Int8 | TypedArrayKind::Uint8 | TypedArrayKind::Uint8Clamped => 1,
            TypedArrayKind::Int16 | TypedArrayKind::Uint16 => 2,
            TypedArrayKind::Int32 | TypedArrayKind::Uint32 | TypedArrayKind::Float32 => 4,
            TypedArrayKind::Float64 | TypedArrayKind::BigInt64 | TypedArrayKind::BigUint64 => 8,
        }
    }

    /// The JS constructor name, for `Object.prototype.toString` tags.
    pub(crate) fn name(self) -> &'static str {
        match self {
            TypedArrayKind::Int8 => "Int8Array",
            TypedArrayKind::Uint8 => "Uint8Array",
            TypedArrayKind::Uint8Clamped => "Uint8ClampedArray",
            TypedArrayKind::Int16 => "Int16Array",
            TypedArrayKind::Uint16 => "Uint16Array",
            TypedArrayKind::Int32 => "Int32Array",
            TypedArrayKind::Uint32 => "Uint32Array",
            TypedArrayKind::Float32 => "Float32Array",
            TypedArrayKind::Float64 => "Float64Array",
            TypedArrayKind::BigInt64 => "BigInt64Array",
            TypedArrayKind::BigUint64 => "BigUint64Array",
        }
    }

    /// True for the bigint element kinds (`BigInt64Array`,
    /// `BigUint64Array`): elements convert via `ToBigInt` and read back
    /// as `Value::BigInt` instead of going through `ToNumber`/numbers.
    pub(crate) fn is_bigint(self) -> bool {
        matches!(self, TypedArrayKind::BigInt64 | TypedArrayKind::BigUint64)
    }
}

/// Bytecode function metadata stored inline in [`ObjectKind::Function`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FunctionData {
    pub bytecode_id: u32,
    /// Function name, packed inline when short ([`Str::inline`]).
    pub name: Str,
    pub arity: u16,
}

/// Host function called by the VM. Receives `&mut Heap` so builtins can
/// allocate; any handle that must survive an allocation inside the call has
/// to be rooted by the caller (or re-derived afterwards).
pub type NativeFn =
    fn(&mut crate::gc::Heap, this: Value, args: &[Value]) -> Result<Value, NativeError>;

/// Errors a native function can report back to the VM.
#[derive(Debug)]
pub enum NativeError {
    TypeError(String),
    RangeError(String),
    /// Added for `JSON.parse` (spec mandates `SyntaxError`); additive and
    /// non-breaking for existing matches on the other variants.
    SyntaxError(String),
    /// A JavaScript value thrown across the native/bytecode boundary.
    ///
    /// Added for the VM worker: when a native calls back into bytecode
    /// (e.g. `Array.prototype.map` invoking its callback) and the callback
    /// throws, the thrown value must propagate back through the native
    /// frame without losing its identity. The VM converts this back into a
    /// VM-level throw of the original value. Additive and non-breaking for
    /// existing matches on the other variants.
    JsThrown(Value),
    OutOfMemory,
}

impl std::borrow::Borrow<Wtf8> for Str {
    /// Content borrow, so `HashMap<Str, _>` can be probed with WTF-8
    /// (the shape index does this on every property lookup; probe with
    /// [`Wtf8::from_str`] for `&str` keys). Consistent with [`Str`]'s
    /// content-based [`Hash`] and [`Eq`].
    fn borrow(&self) -> &Wtf8 {
        self.as_wtf8()
    }
}

/// A JavaScript object: kind tag, prototype link, hidden class, and a
/// property list.
///
/// String-keyed properties are stored in two parallel lists —
/// `properties` (data) and `accessors` (getter/setter) — but every object
/// also carries a [`ShapeId`]: the interned hidden class describing which
/// keys live in which list at which index. Objects built by the same
/// sequence of definitions share one [`Shape`], so a lookup is one hash
/// probe on the shape's index instead of a linear scan, with no per-object
/// cache to invalidate.
///
/// Symbol-keyed properties live in the parallel `symbol_properties` /
/// `symbol_accessors` side tables and bypass the shape entirely: they
/// never enter [`Shape::index`](Shape), so the O(1) string fast path is
/// untouched and symbol definitions cause no shape transition.
///
/// A key lives in exactly one of `properties` (data) or `accessors`
/// (getter/setter); defining one kind evicts the other. The same
/// one-list invariant holds for symbol keys across `symbol_properties`
/// and `symbol_accessors`.
/// Cold fields of an [`Object`], boxed behind one pointer so ordinary
/// objects carry none of them inline. Allocated lazily by
/// [`Object::cold_mut`] the first time any field is written; a fresh
/// ordinary object has no cold box at all.
#[derive(Debug, Default)]
pub(crate) struct ObjectCold {
    /// Raw byte storage for `ArrayBuffer` objects (see
    /// [`ObjectKind::ArrayBuffer`]). `None` for every other object. Lives
    /// in the cold box so the collector frees it with the object; it
    /// holds no GC handles, so tracing skips it.
    ///
    /// `None` on an `ArrayBuffer`-tagged object is the *detached* state
    /// (`$262.detachArrayBuffer`, and in the future `structuredClone`
    /// neutering): a zero-length buffer still carries `Some([])`, so the
    /// two are unambiguous. Every view accessor consults the buffer's
    /// detached state at read time (see `is_detached_buffer` in
    /// `crate::builtin_reflect_proxy`), so no view registry is needed.
    pub bytes: Option<Box<[u8]>>,
    /// Resizable-`ArrayBuffer` ceiling: `Some(max)` when the buffer was
    /// created with `{ maxByteLength: max }` (making it resizable),
    /// `None` for fixed-size buffers. Meaningful only when `kind` is
    /// [`ObjectKind::ArrayBuffer`]; `None` for every other object.
    /// Unforgeable: not a property.
    pub ab_max_byte_length: Option<u32>,
    /// `SharedArrayBuffer` marker: true when the buffer was created by
    /// the `SharedArrayBuffer` constructor. `SharedArrayBuffer` aliases
    /// the `ArrayBuffer` implementation (same [`ObjectKind::ArrayBuffer`]
    /// tag, same byte storage — no shared-memory semantics yet), so this
    /// flag is the `IsSharedArrayBuffer` discriminator: shared buffers
    /// can never detach, and `transfer` / `slice` / `resize` plus the
    /// `resizable` / `maxByteLength` / `detached` getters throw
    /// `TypeError` on them. Meaningful only when `kind` is
    /// [`ObjectKind::ArrayBuffer`]; false for every other object.
    /// Unforgeable: not a property.
    pub ab_shared: bool,
    /// Length-tracking view: true when a typed-array view or `DataView`
    /// was created with an omitted length over a resizable buffer, so
    /// `length` / `byteLength` recompute from the buffer's *current* byte
    /// length on every read. Meaningful only for
    /// [`ObjectKind::TypedArrayView`] / [`ObjectKind::DataView`]; false
    /// for every other object. Unforgeable: not a property.
    pub view_length_tracking: bool,
    /// Dense element storage for arrays.
    ///
    /// When `Some`, this array's indexed elements live here instead of in
    /// `properties`: `dense[i]` is the element at index `i` (`None` is a
    /// hole) and `dense.len()` is the array's `length`. The `"length"`
    /// property is then synthesized by `Heap::get_property` rather than
    /// stored.
    ///
    /// Dense arrays are a construction-time optimization: builtin code
    /// builds result arrays with `Heap::new_dense_array` /
    /// `Heap::dense_push` and converts back with
    /// `Heap::materialize_dense` before the array becomes user-visible,
    /// so every direct reader of `properties` (enumeration, cloning,
    /// `JSON.stringify`, `for`-`in`) keeps working. Invariant: while
    /// `dense` is `Some`, no index property lives in `properties`.
    ///
    /// The elements are GC handles: the tracer must visit them (see
    /// `Heap` marking in `crate::gc`), exactly as it did when this field
    /// lived inline on `Object`.
    pub dense: Option<Vec<Option<Value>>>,
}

pub struct Object {
    pub(crate) kind: ObjectKind,
    pub prototype: Value,
    /// Hidden class: interned layout of [`Object::properties`] and
    /// [`Object::accessors`]. [`ShapeId::ROOT`] for a fresh object;
    /// every structural mutation (add/remove/data↔accessor conversion)
    /// advances it via the [`Heap`](crate::gc::Heap) shape table, so the
    /// shape always mirrors the two lists.
    pub shape: ShapeId,
    /// Property keys are [`Str`]: short names pack inline with no heap
    /// allocation and no GC tracing; longer names hold interned heap
    /// handles. Comparisons are by content either way.
    pub properties: Vec<(Str, Property)>,
    /// Accessor properties (getter/setter pairs), parallel to `properties`.
    pub accessors: Vec<(Str, AccessorProp)>,
    /// Symbol-keyed data properties, parallel to `properties`.
    ///
    /// Symbol keys bypass the shape/index machinery entirely (see the
    /// struct docs); they are compared by identity ([`Gc`] address
    /// equality — symbols are unique per allocation) and scanned
    /// linearly, which is fine because symbol lists are typically tiny.
    /// The keys are heap handles, so the collector must mark them (see
    /// the tracer): a symbol referenced only as a key stays alive.
    pub symbol_properties: Vec<(Gc<Symbol>, Property)>,
    /// Symbol-keyed accessor properties, parallel to `symbol_properties`.
    pub symbol_accessors: Vec<(Gc<Symbol>, AccessorProp)>,
    pub extensible: bool,
    /// Cold storage, boxed behind a single pointer so ordinary objects
    /// (the overwhelming majority) pay 8 bytes for all of it instead of
    /// ~48 bytes inline. `None` until the object first needs byte
    /// storage (ArrayBuffer / typed array), dense element storage, or
    /// one of the buffer/view flags; see [`ObjectCold`]. GC object pages
    /// are strided at `size_of::<Object>()`, so keeping this boxed
    /// directly raises heap density. Access goes through the
    /// `Object::bytes` / `Object::dense` / … accessors below, which
    /// lazily allocate the box on first write.
    pub(crate) cold: Option<Box<ObjectCold>>,
    /// Invoke-interception bitmask, read once per call in
    /// `Interp::invoke_inner_depth`.
    ///
    /// Ordinary functions have this set to [`Object::INVOKE_PLAIN`]: call
    /// dispatch goes straight to the [`ObjectKind`] match without
    /// scanning hidden slots. The builtins that install interception
    /// slots (bound functions, promise resolver/combinator/`finally`
    /// closures) set the corresponding bit at creation time, so the
    /// dispatch probes only the slots that can exist. User code cannot
    /// set these bits: defining a `\u{0}bowser:`-prefixed property via
    /// `defineProperty` does not touch the bitmask, so forged slots can
    /// never hijack dispatch.
    pub invoke_flags: u8,
}

impl std::fmt::Debug for Object {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Elide the raw bytes: a large ArrayBuffer must not dump megabytes
        // into test output on a failed assert.
        f.debug_struct("Object")
            .field("kind", &self.kind)
            .field("prototype", &self.prototype)
            .field("shape", &self.shape)
            .field("properties", &self.properties)
            .field("accessors", &self.accessors)
            .field("symbol_properties", &self.symbol_properties)
            .field("symbol_accessors", &self.symbol_accessors)
            .field("extensible", &self.extensible)
            .field("bytes", &self.bytes().map(|b| b.len()))
            .field("ab_max_byte_length", &self.ab_max_byte_length())
            .field("ab_shared", &self.ab_shared())
            .field("view_length_tracking", &self.view_length_tracking())
            // Elide element contents for the same reason as `bytes`.
            .field("dense_len", &self.dense().map(|d| d.len()))
            .field("invoke_flags", &self.invoke_flags)
            .finish()
    }
}

impl Object {
    /// A fresh ordinary-style object of the given kind with no properties
    /// and a null prototype.
    pub(crate) fn new(kind: ObjectKind) -> Self {
        Object {
            kind,
            prototype: Value::Null,
            shape: ShapeId::ROOT,
            properties: Vec::new(),
            accessors: Vec::new(),
            symbol_properties: Vec::new(),
            symbol_accessors: Vec::new(),
            extensible: true,
            cold: None,
            invoke_flags: Self::INVOKE_PLAIN,
        }
    }

    // -- Cold-field accessors -------------------------------------------
    //
    // Reads return the field's "empty" value (`None` / `false`) when no
    // cold box exists, which is exactly the value an ordinary object
    // carried inline before the fields were boxed. Writes go through
    // `cold_mut`, which lazily allocates the box.

    /// The cold box, if this object has ever needed cold storage.
    pub(crate) fn cold(&self) -> Option<&ObjectCold> {
        self.cold.as_deref()
    }

    /// The cold box, allocating it on first use.
    pub(crate) fn cold_mut(&mut self) -> &mut ObjectCold {
        self.cold
            .get_or_insert_with(|| Box::new(ObjectCold::default()))
    }

    /// Whether a cold box has been allocated for this object (test and
    /// diagnostic use today).
    #[allow(dead_code)]
    pub(crate) fn has_cold(&self) -> bool {
        self.cold.is_some()
    }

    /// ArrayBuffer byte storage as a slice; `None` when detached or not
    /// a buffer. See [`ObjectCold::bytes`].
    pub(crate) fn bytes(&self) -> Option<&[u8]> {
        self.cold()?.bytes.as_deref()
    }

    /// Mutable view of the byte storage; `None` when detached or not a
    /// buffer (never allocates the storage itself).
    pub(crate) fn bytes_mut(&mut self) -> Option<&mut [u8]> {
        self.cold.as_mut()?.bytes.as_deref_mut()
    }

    /// Replace the byte storage (`None` detaches an ArrayBuffer).
    pub(crate) fn set_bytes(&mut self, bytes: Option<Box<[u8]>>) {
        self.cold_mut().bytes = bytes;
    }

    /// Take the byte storage out, leaving the detached state behind
    /// (the symmetric counterpart of [`Object::take_dense`]).
    #[allow(dead_code)]
    pub(crate) fn take_bytes(&mut self) -> Option<Box<[u8]>> {
        self.cold.as_mut()?.bytes.take()
    }

    /// Resizable-ArrayBuffer ceiling; see [`ObjectCold::ab_max_byte_length`].
    pub(crate) fn ab_max_byte_length(&self) -> Option<u32> {
        self.cold()?.ab_max_byte_length
    }

    pub(crate) fn set_ab_max_byte_length(&mut self, max: Option<u32>) {
        self.cold_mut().ab_max_byte_length = max;
    }

    /// SharedArrayBuffer marker; see [`ObjectCold::ab_shared`].
    pub(crate) fn ab_shared(&self) -> bool {
        self.cold().is_some_and(|c| c.ab_shared)
    }

    pub(crate) fn set_ab_shared(&mut self, shared: bool) {
        self.cold_mut().ab_shared = shared;
    }

    /// Length-tracking view marker; see [`ObjectCold::view_length_tracking`].
    pub(crate) fn view_length_tracking(&self) -> bool {
        self.cold().is_some_and(|c| c.view_length_tracking)
    }

    pub(crate) fn set_view_length_tracking(&mut self, tracking: bool) {
        self.cold_mut().view_length_tracking = tracking;
    }

    /// Dense element storage; see [`ObjectCold::dense`].
    pub(crate) fn dense(&self) -> Option<&Vec<Option<Value>>> {
        self.cold()?.dense.as_ref()
    }

    /// Mutable dense element storage; `None` when the array is not dense
    /// (never allocates the storage itself — use [`Object::set_dense`]
    /// to install it).
    pub(crate) fn dense_mut(&mut self) -> Option<&mut Vec<Option<Value>>> {
        self.cold.as_mut()?.dense.as_mut()
    }

    /// Install or clear dense element storage.
    pub(crate) fn set_dense(&mut self, dense: Option<Vec<Option<Value>>>) {
        self.cold_mut().dense = dense;
    }

    /// Take the dense element storage out, leaving the array non-dense.
    pub(crate) fn take_dense(&mut self) -> Option<Vec<Option<Value>>> {
        self.cold.as_mut()?.dense.take()
    }

    /// No invoke interception: ordinary function dispatch.
    pub const INVOKE_PLAIN: u8 = 0;
    /// Bound function: `\u{0}bowser:bound-target/this/args` slots
    /// (installed by `Function.prototype.bind`).
    pub const INVOKE_BOUND: u8 = 1 << 0;
    /// Promise resolve/reject closure: `\u{0}bowser:promise-resolve` or
    /// `\u{0}bowser:promise-reject` id slot.
    pub const INVOKE_PROMISE_ID: u8 = 1 << 1;
    /// Promise combinator element callback:
    /// `\u{0}bowser:cs-on-fulfilled`/`cs-on-rejected` + `cs-index` slots.
    pub const INVOKE_COMBINATOR: u8 = 1 << 2;
    /// Promise `race` settle callback: `\u{0}bowser:settle-fulfill` or
    /// `\u{0}bowser:settle-reject` id slot.
    pub const INVOKE_SETTLE: u8 = 1 << 3;
    /// `finally` outer closure: `\u{0}bowser:fin-cb` + `fin-throw` slots.
    pub const INVOKE_FINALLY_OUTER: u8 = 1 << 4;
    /// `finally` inner closure: `\u{0}bowser:fin-value` + `fin-throw`
    /// slots.
    pub const INVOKE_FINALLY_INNER: u8 = 1 << 5;
}

impl Default for Object {
    /// A null-prototype ordinary object: [`Object::new(ObjectKind::Ordinary)`].
    ///
    /// Exists so builtin modules can write `Object { kind, prototype,
    /// ..Default::default() }` for kind-tagged allocations.
    fn default() -> Self {
        Object::new(ObjectKind::Ordinary)
    }
}

/// A data property descriptor stored inline in [`Object`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Property {
    pub value: Value,
    pub writable: bool,
    pub enumerable: bool,
    pub configurable: bool,
}

impl Property {
    /// A default data property: writable, enumerable, configurable.
    pub fn data(value: Value) -> Self {
        Property {
            value,
            writable: true,
            enumerable: true,
            configurable: true,
        }
    }
}

/// An accessor property descriptor: getter/setter pair plus attributes.
///
/// `get`/`set` hold the function (or [`Value::Undefined`] when absent).
/// Stored in [`Object::accessors`], parallel to the data [`Property`]
/// list; a key lives in exactly one of the two.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccessorProp {
    pub get: Value,
    pub set: Value,
    pub enumerable: bool,
    pub configurable: bool,
}

impl AccessorProp {
    /// A default accessor property: enumerable, configurable.
    pub fn new(get: Value, set: Value) -> Self {
        AccessorProp {
            get,
            set,
            enumerable: true,
            configurable: true,
        }
    }
}

/// A hidden class: identity of an interned [`Shape`].
///
/// Shapes live in the [`Heap`](crate::gc::Heap) shape table. A `ShapeId`
/// is a plain `u32` index — no GC tracing, no lifetime — so it is only
/// valid until the next collection: [`Heap::collect`] compacts the table
/// and renumbers shapes. Never cache a `ShapeId` across a call that can
/// trigger a collection; re-read it from the object instead.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct ShapeId(pub u32);

impl ShapeId {
    /// The empty shape: every fresh [`Object`] starts here.
    pub const ROOT: ShapeId = ShapeId(0);
}

/// One per-call-site inline-cache entry for named property access
/// (`GetPropConst` / `SetPropConst` / `LoadName`).
///
/// Caches the `(ShapeId -> property-list slot)` resolution for the call
/// site's key: on a hit, the string hash + `HashMap` probe in
/// [`Shape::lookup`] are skipped entirely. `slot` indexes
/// [`Object::properties`] when `list == 0` and [`Object::accessors`]
/// when `list == 1`.
///
/// A site that observes genuinely polymorphic shapes — a shape that
/// is not a [`shape_add`](crate::gc::Heap::shape_add) extension of the
/// cached one — goes *megamorphic* ([`InlineCache::MEGAMORPHIC_LIST`]):
/// the entry stops probing and refilling, and every access takes the
/// slow path, until a shape-table epoch bump resets it to
/// monomorphic. Monotonic growth (adding properties, which extends
/// the cached shape) refills the entry and stays monomorphic.
///
/// The all-zero entry never hits: no shape-`0` object carries
/// properties (the root shape is empty), and the accessor path
/// requires `list == 1`.
///
/// Stored in a [`Cell`](std::cell::Cell) in the VM's per-template side
/// table (parallel to the lazily-interned string constants); plain
/// integers, never GC-traced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InlineCache {
    /// Shape-table epoch this entry was resolved under (see
    /// [`Heap::shape_epoch`](crate::gc::Heap::shape_epoch)).
    /// [`Heap::collect`](crate::gc::Heap::collect) renumbers `ShapeId`s
    /// when it compacts the shape table, so an entry from an older epoch
    /// may name a different layout and must be ignored, never trusted.
    pub epoch: u64,
    /// The [`ShapeId`] (its `.0`) this slot was resolved for.
    pub shape: u32,
    /// Which property list `slot` indexes: `0` = data properties,
    /// `1` = accessor properties, [`InlineCache::MEGAMORPHIC_LIST`] =
    /// megamorphic (never probes, never refills within this epoch).
    pub list: u8,
    /// Index into the property list.
    pub slot: u32,
}

impl InlineCache {
    /// `list` value marking a megamorphic call site: the site saw a
    /// shape that does not extend the cached one (genuinely
    /// polymorphic access), so probing and refilling are skipped and
    /// every access takes the slow path until the next shape-table
    /// epoch bump.
    pub const MEGAMORPHIC_LIST: u8 = 2;

    /// True once this entry has gone megamorphic (see
    /// [`InlineCache::MEGAMORPHIC_LIST`]).
    pub fn is_megamorphic(self) -> bool {
        self.list >= Self::MEGAMORPHIC_LIST
    }
}

/// Which of an [`Object`]'s two property lists holds a key:
/// `0` = [`Object::properties`] (data), `1` = [`Object::accessors`].
pub(crate) type ShapeList = u8;

/// FxHash-style hasher (std-only, no dependency): a rotate-multiply hash
/// over the input bytes, far cheaper than SipHash (`RandomState`) for
/// short keys. Used for the shape transition overflow table, where keys
/// are short property names probed on every object append — throughput
/// matters and HashDoS resistance is a non-goal (keys are the program's
/// own property names, and each table is per-shape, not attacker-fed).
#[derive(Clone, Debug, Default)]
pub(crate) struct FxBuildHasher;

#[derive(Clone, Debug)]
pub(crate) struct FxHasher(u64);

impl std::hash::BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;
    fn build_hasher(&self) -> FxHasher {
        FxHasher(0x51_7c_c1_b7_27_22_0a_95)
    }
}

impl std::hash::Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        const K: u64 = 0x51_7c_c1_b7_27_22_0a_95;
        let (chunks, rem) = bytes.as_chunks::<8>();
        for c in chunks {
            let w = u64::from_le_bytes(*c);
            self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(K);
        }
        if !rem.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rem.len()].copy_from_slice(rem);
            let w = u64::from_le_bytes(buf);
            self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(K);
        }
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

/// A shape's transition table: appending `(key, list)` to the shape
/// yields the child shape.
///
/// The first [`ShapeTransitions::INLINE_CAP`] transitions live in a
/// linear-scanned inline vec — the common case (shapes with a handful
/// of children) probes with no hashing at all; overflow goes to a hash
/// map with the cheap [`FxBuildHasher`], probed before the inline prefix
/// once the table is full. Measured with the crate-internal
/// `shape_transition_bench` (interned heap keys, i.e. the realistic
/// `shape_add` path where `Str::eq`'s pointer-identity fast path
/// applies; all-hit probes, best of 3, debug build):
///
/// | entries | hybrid | `RandomState` `HashMap` (status quo ante) |
/// |---------|--------|---------------------------------------|
/// | 4       | 139 ns | 446 ns                                |
/// | 16      | 561 ns | 446 ns                                |
/// | 64      | 453 ns | 446 ns                                |
///
/// Small tables — the common case — probe ~3x faster than the
/// `RandomState` map; large tables (notably the root shape, one
/// transition per distinct first-key program-wide) are at parity,
/// while a pure linear scan regressed to 2086 ns at 64 entries. The
/// 16-entry middle band slightly favors the plain map, but shapes with
/// 9–16 children are rare (only the root is routinely large), and the
/// per-object build cost (one big-table probe plus several small-table
/// probes) still favors the hybrid. Each `Shape` also shrinks from a
/// 48-byte (usually empty) `HashMap` to a 24-byte `Vec` plus a lazily
/// allocated overflow map.
/// Overflow transition map for shapes with more than [`ShapeTransitions::INLINE_CAP`]
/// children: keyed by `(key, list)`, probed first once it exists.
type ShapeOverflowMap =
    Option<Box<std::collections::HashMap<(Str, ShapeList), ShapeId, FxBuildHasher>>>;

#[derive(Clone, Debug, Default)]
pub(crate) struct ShapeTransitions {
    /// First transitions, linear-scanned (no hashing).
    inline: Vec<(Str, ShapeList, ShapeId)>,
    /// Overflow for shapes with many distinct children.
    overflow: ShapeOverflowMap,
}

impl ShapeTransitions {
    /// Maximum linear-scanned transitions before overflow. Covers the
    /// common case (a shape seldom has more than a few children); the
    /// root shape is the known exception and uses the overflow map.
    const INLINE_CAP: usize = 8;

    /// Find the transition child for appending `(key, list)`.
    pub(crate) fn get(&self, key: Str, list: ShapeList) -> Option<ShapeId> {
        // When the table overflowed, most entries live in the map: probe
        // it first, then the inline prefix. Small tables have no overflow
        // and go straight to the linear scan.
        if let Some(map) = self.overflow.as_ref()
            && let Some(&child) = map.get(&(key, list))
        {
            return Some(child);
        }
        for &(k, l, child) in &self.inline {
            if l == list && k == key {
                return Some(child);
            }
        }
        None
    }

    /// Record a new `(key, list) -> child` transition. The caller
    /// ([`Heap::shape_add`](crate::gc::Heap::shape_add)) probes with
    /// [`get`](Self::get) first, so the key is always fresh here.
    pub(crate) fn insert(&mut self, key: Str, list: ShapeList, child: ShapeId) {
        if self.inline.len() < Self::INLINE_CAP {
            self.inline.push((key, list, child));
        } else {
            self.overflow
                .get_or_insert_with(|| {
                    Box::new(std::collections::HashMap::with_hasher(FxBuildHasher))
                })
                .insert((key, list), child);
        }
    }

    /// Drop entries whose child was evicted by shape-table compaction
    /// (pure cache: re-interned on next use), rewriting survivors.
    /// (Hand-rolled loop: this toolchain's `Vec::retain` only hands the
    /// predicate a shared reference, but eviction rewrites the child id
    /// in place.)
    pub(crate) fn retain(&mut self, mut f: impl FnMut((Str, ShapeList), &mut ShapeId) -> bool) {
        let mut i = 0;
        while i < self.inline.len() {
            let (k, l, child) = &mut self.inline[i];
            if f((*k, *l), child) {
                i += 1;
            } else {
                // Order is irrelevant (lookups scan everything).
                self.inline.swap_remove(i);
            }
        }
        if let Some(m) = self.overflow.as_mut() {
            m.retain(|k, child| f(*k, child));
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.inline.len() + self.overflow.as_ref().map_or(0, |m| m.len())
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.inline.is_empty() && self.overflow.as_ref().is_none_or(|m| m.is_empty())
    }
}

/// An interned object layout: the key sequence an object was built with,
/// as a key → (list, index) map for O(1) lookup. Attributes are
/// deliberately NOT part of the shape: two objects with the same keys in
/// the same lists share a shape even when their attributes differ; the
/// shape only accelerates *lookup*, and attributes are read from the
/// property lists.
///
/// Shapes are created only by appending (the transition table) or by
/// replaying an object's current lists after a removal/conversion (see
/// [`Heap`](crate::gc::Heap) shape methods); both paths intern, so
/// identical layouts share one `Shape`. Shapes are immutable after
/// creation but not immortal: [`Heap::collect`] evicts shapes no live
/// object references (unpinning their key strings), so only layouts still
/// in use pay the strong-rooting cost.
#[derive(Clone, Debug, Default)]
pub(crate) struct Shape {
    /// Key → (list, index), for O(1) lookup. Probed with [`Wtf8`]
    /// via [`Borrow<Wtf8> for Str`](std::borrow::Borrow) (probe `&str`
    /// keys with [`Wtf8::from_str`]).
    pub index: std::collections::HashMap<Str, (ShapeList, u32)>,
    /// Transition table: appending `(key, list)` to this shape yields the
    /// child shape. A hybrid inline-vec + FxHash overflow map (see
    /// [`ShapeTransitions`]), so small shapes probe with no hashing and
    /// the root shape (many distinct children) stays O(1). The child's
    /// new entry index is derived from the entry counts below, so it is
    /// not part of the key.
    pub transitions: ShapeTransitions,
    /// The shape this one was transitioned from ([`ShapeId::ROOT`] for
    /// the root itself). Lets [`Heap::compact_shapes`] keep a live
    /// shape's whole derivation chain: without the ancestors, re-deriving
    /// the same layout would miss the transition cache and intern a
    /// duplicate [`Shape`] for an identical layout.
    pub parent: ShapeId,
    /// Number of data-list entries (next data index for a transition).
    pub data_len: u32,
    /// Number of accessor-list entries (next accessor index).
    pub acc_len: u32,
}

impl Shape {
    /// Maximum keys tracked per property list. Past this the object is in
    /// *dictionary mode*: the shape stops evolving (appends keep the
    /// parent shape) and lookups fall back to the linear scan. This keeps
    /// every transition O(1) — without it, building an n-key shape costs
    /// O(n²) in map clones (one full clone per appended key), which a
    /// 100k-element array materialization turns into an OOM. V8 draws the
    /// same line for the same reason.
    pub const MAX_ENTRIES: u32 = 64;
    /// The empty root shape. Its parent is itself ([`ShapeId::ROOT`]),
    /// which terminates the ancestor walk in shape-table compaction.
    pub fn root() -> Self {
        Shape::default()
    }

    /// Look up `name` in the index: `(list, slot index)` on hit.
    pub fn lookup(&self, name: &Wtf8) -> Option<(ShapeList, u32)> {
        self.index.get(name).copied()
    }

    /// Find the transition child for appending `(key, list)`: an inline
    /// linear scan for small tables, hash lookup once the table overflows
    /// past its inline capacity.
    pub fn transition(&self, key: Str, list: ShapeList) -> Option<ShapeId> {
        self.transitions.get(key, list)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gc::Heap;

    fn native(_heap: &mut Heap, _this: Value, _args: &[Value]) -> Result<Value, NativeError> {
        Ok(Value::Int(42))
    }

    #[test]
    fn truthiness_matches_js() {
        assert!(!Value::Undefined.is_truthy());
        assert!(!Value::Null.is_truthy());
        assert!(!Value::Bool(false).is_truthy());
        assert!(Value::Bool(true).is_truthy());
        assert!(!Value::Int(0).is_truthy());
        assert!(Value::Int(-7).is_truthy());
        assert!(!Value::Float(0.0).is_truthy());
        assert!(!Value::Float(-0.0).is_truthy());
        assert!(!Value::Float(f64::NAN).is_truthy());
        assert!(Value::Float(0.5).is_truthy());

        let mut heap = Heap::new();
        let empty = heap.alloc_str("").unwrap();
        let full = heap.alloc_str("x").unwrap();
        assert!(!Value::Str(empty).is_truthy());
        assert!(Value::Str(full).is_truthy());

        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        assert!(Value::Object(obj).is_truthy());

        let sym = heap.alloc_symbol(None).unwrap();
        assert!(Value::Symbol(sym).is_truthy());

        let zero = heap.alloc_bigint(false, Vec::new()).unwrap();
        let one = heap.alloc_bigint(false, vec![1]).unwrap();
        let neg_zero = heap.alloc_bigint(true, Vec::new()).unwrap();
        assert!(!Value::BigInt(zero).is_truthy());
        assert!(!Value::BigInt(neg_zero).is_truthy());
        assert!(Value::BigInt(one).is_truthy());
    }

    #[test]
    fn type_of_matches_js() {
        assert_eq!(Value::Undefined.type_of(), "undefined");
        assert_eq!(Value::Null.type_of(), "object");
        assert_eq!(Value::Bool(true).type_of(), "boolean");
        assert_eq!(Value::Int(1).type_of(), "number");
        assert_eq!(Value::Float(1.5).type_of(), "number");

        let mut heap = Heap::new();
        let s = heap.alloc_str("s").unwrap();
        assert_eq!(Value::Str(s).type_of(), "string");

        let sym = heap.alloc_symbol(Some(Wtf8::from_str("d"))).unwrap();
        assert_eq!(Value::Symbol(sym).type_of(), "symbol");
        let bi = heap.alloc_bigint(false, vec![1]).unwrap();
        assert_eq!(Value::BigInt(bi).type_of(), "bigint");

        let ordinary = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        assert_eq!(Value::Object(ordinary).type_of(), "object");
        let array = heap.alloc_object(Object::new(ObjectKind::Array)).unwrap();
        assert_eq!(Value::Object(array).type_of(), "object");

        let name = heap.intern_str("f").unwrap();
        let fun = heap
            .alloc_object(Object::new(ObjectKind::Function(FunctionData {
                bytecode_id: 7,
                name,
                arity: 2,
            })))
            .unwrap();
        assert_eq!(Value::Object(fun).type_of(), "function");

        let native_fun = heap
            .alloc_object(Object::new(ObjectKind::NativeFunction(native)))
            .unwrap();
        assert_eq!(Value::Object(native_fun).type_of(), "function");
    }

    #[test]
    fn predicates() {
        assert!(Value::Undefined.is_undefined());
        assert!(!Value::Null.is_undefined());
        assert!(Value::Null.is_null());
        assert!(!Value::Undefined.is_null());
        assert!(!Value::Int(0).is_null());
    }

    #[test]
    fn property_and_symbol_shapes() {
        let p = Property::data(Value::Bool(true));
        assert!(p.writable && p.enumerable && p.configurable);
        assert_eq!(p.value, Value::Bool(true));

        let mut heap = Heap::new();
        let a = heap.alloc_symbol(Some(Wtf8::from_str("desc"))).unwrap();
        let b = heap.alloc_symbol(Some(Wtf8::from_str("desc"))).unwrap();
        // Symbols are unique even with equal descriptions.
        assert_ne!(a, b);
        assert_eq!(
            heap.get(a).description.as_deref(),
            Some(Wtf8::from_str("desc"))
        );
        assert!(heap.get(b).description.is_some());
        let anon = heap.alloc_symbol(None).unwrap();
        assert!(heap.get(anon).description.is_none());
    }

    #[test]
    fn native_errors_and_fn_pointer() {
        let f: NativeFn = native;
        let mut heap = Heap::new();
        let r = f(&mut heap, Value::Undefined, &[]);
        assert_eq!(r.unwrap(), Value::Int(42));

        let te = NativeError::TypeError("bad".to_string());
        let re = NativeError::RangeError("oob".to_string());
        let oe = NativeError::OutOfMemory;
        assert!(format!("{te:?}").contains("bad"));
        assert!(format!("{re:?}").contains("oob"));
        assert!(format!("{oe:?}").contains("OutOfMemory"));
    }

    #[test]
    fn value_from_gc_conversions() {
        let mut heap = Heap::new();
        let s = heap.alloc_str("s").unwrap();
        let o = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let sym = heap.alloc_symbol(None).unwrap();
        let bi = heap.alloc_bigint(false, vec![9]).unwrap();
        assert_eq!(Value::from(s), Value::Str(s));
        assert_eq!(Value::from(o), Value::Object(o));
        assert_eq!(Value::from(sym), Value::Symbol(sym));
        assert_eq!(Value::from(bi), Value::BigInt(bi));
    }

    #[test]
    fn gc_handles_hash_and_eq() {
        use std::collections::HashSet;
        let mut heap = Heap::new();
        // Heap-allocated strings (past the inline limit): handles compare
        // by identity, one entry per allocation.
        let long = "a".repeat(64);
        let a = heap.alloc_str(&long).unwrap();
        let b = heap.alloc_str(&long).unwrap();
        assert_ne!(a.heap_handle(), b.heap_handle());
        let mut set = HashSet::new();
        set.insert(a.heap_handle().unwrap());
        set.insert(b.heap_handle().unwrap());
        assert_eq!(set.len(), 2);
        let i = heap.intern("q").unwrap();
        let j = heap.intern("q").unwrap();
        assert_eq!(i, j);
        assert!(!format!("{i:?}").is_empty());
    }

    // --- Small-string optimization (15-byte inline) ---

    #[test]
    fn str_inline_boundary_is_15_bytes() {
        assert_eq!(STR_INLINE_MAX, 15);
        let max = Str::inline(&"a".repeat(15)).expect("15 bytes inlines");
        assert!(max.is_inline());
        assert_eq!(max.len(), 15);
        assert_eq!(max.as_wtf8(), "a".repeat(15).as_str());
        assert!(Str::inline(&"a".repeat(16)).is_none());
        // Empty strings inline too.
        let empty = Str::inline("").unwrap();
        assert!(empty.is_inline());
        assert!(empty.is_empty());
    }

    #[test]
    fn str_boundary_counts_bytes_not_chars() {
        // "é" is 2 bytes in UTF-8: 7 of them (14 bytes) inline, 8 (16
        // bytes) heap.
        let inline = Str::inline(&"é".repeat(7)).unwrap();
        assert!(inline.is_inline());
        assert_eq!(inline.len(), 14);
        assert_eq!(inline.as_wtf8(), "é".repeat(7).as_str());
        let mut heap = Heap::new();
        let heap_s = heap.alloc_str(&"é".repeat(8)).unwrap();
        assert!(!heap_s.is_inline());
        assert_eq!(heap_s.len(), 16);
        assert_eq!(heap_s.as_wtf8(), "é".repeat(8).as_str());
    }

    #[test]
    fn str_equality_and_hash_agree_across_representations() {
        use std::collections::HashSet;
        use std::hash::{Hash, Hasher};
        fn hash_of(s: &Str) -> u64 {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            s.hash(&mut h);
            h.finish()
        }
        // Same content inline vs heap-backed: `intern_heap` always
        // heap-allocates, even for short strings (`intern` now packs them
        // inline instead).
        let mut heap = Heap::new();
        let inline = Str::inline("abc").unwrap();
        let heap_s = Str::Heap(heap.intern_heap(Wtf8::from_str("abc")).unwrap());
        assert_eq!(inline, heap_s);
        assert_eq!(hash_of(&inline), hash_of(&heap_s));
        // And in a `HashSet` they collapse to one entry.
        let mut set = HashSet::new();
        set.insert(inline);
        set.insert(heap_s);
        assert_eq!(set.len(), 1);
        // Debug shows contents, not representation.
        assert_eq!(format!("{inline:?}"), "\"abc\"");
        assert_eq!(format!("{heap_s:?}"), "\"abc\"");
    }

    #[test]
    fn shape_transition_lookup_hits_and_misses() {
        // Lookups are by (key, list) content: hits return the child, and
        // misses / wrong-list probes return `None`.
        let mut shape = Shape::root();
        let ka = Str::inline("ka").unwrap();
        let kb = Str::inline("kb").unwrap();
        shape.transitions.insert(ka, 0, ShapeId(7));
        shape.transitions.insert(kb, 1, ShapeId(9));
        assert_eq!(shape.transition(ka, 0), Some(ShapeId(7)));
        assert_eq!(shape.transition(kb, 1), Some(ShapeId(9)));
        // The list is part of the key.
        assert_eq!(shape.transition(ka, 1), None);
        assert_eq!(shape.transition(kb, 0), None);
        // Unknown key misses.
        assert_eq!(shape.transition(Str::inline("kc").unwrap(), 0), None);
        // Content equality across representations: `intern_heap` always
        // heap-allocates, so a heap-backed key is found by an inline
        // probe with the same content.
        let mut heap = Heap::new();
        let heap_kd = Str::Heap(heap.intern_heap(Wtf8::from_str("kd")).unwrap());
        assert!(!heap_kd.is_inline());
        shape.transitions.insert(heap_kd, 0, ShapeId(11));
        assert_eq!(
            shape.transition(Str::inline("kd").unwrap(), 0),
            Some(ShapeId(11))
        );
        assert_eq!(shape.transition(heap_kd, 0), Some(ShapeId(11)));
    }

    // ---------------------------------------------------------------------------
    // Shape transition-table microbenchmark (wave-36 finding 5).
    //
    // Compares the append-probe hot path -- `transition(key, list)` on a hit,
    // which `Heap::shape_add` runs once per property append -- across four
    // table designs:
    //
    // * `hashmap_std`: `HashMap<(Str, ShapeList), ShapeId>` with `RandomState`
    //   (the status quo the finding questioned: SipHash per probe, plus a
    //   48-byte `HashMap` on every `Shape` even when empty).
    // * `hashmap_fx`: the same map with the cheap [`FxBuildHasher`].
    // * `vec_linear`: `Vec<(Str, ShapeList, ShapeId)>` linear scan (the design
    //   before the `HashMap`).
    // * `hybrid`: [`ShapeTransitions`] -- inline linear vec for the first
    //   entries, [`FxBuildHasher`] overflow map past that.
    //
    // Workload: `K` distinct short inline keys, `N` lookups in a deterministic
    // pseudo-random order (all hits, like the steady-state probe). Each case
    // gets one warmup pass, then the best ns/op of 3 trials. The test asserts
    // functional equivalence of the four tables (same answers on hits and
    // misses); the timings are printed for the record -- see the measurement
    // note on [`ShapeTransitions`].
    //
    // Run: `cargo test -p bowser_js shape_transition_bench -- --nocapture`
    mod shape_transition_bench {
        use std::collections::HashMap;
        use std::time::Instant;

        use super::{FxBuildHasher, ShapeId, ShapeList, ShapeTransitions, Str};
        use crate::gc::Heap;

        /// Build `K` distinct heap-interned keys: this is the realistic
        /// `shape_add` path (property definition interns the name), so
        /// `Str::eq`'s pointer-identity fast path applies exactly as in
        /// production. The heap is kept alive by the caller.
        fn keys(heap: &mut Heap, k: usize) -> Vec<Str> {
            use crate::wtf8::Wtf8;
            (0..k)
                .map(|i| {
                    let g = heap
                        .intern_heap(Wtf8::from_str(&format!("key{i:03}")))
                        .expect("intern");
                    Str::Heap(g)
                })
                .collect()
        }

        /// Deterministic pseudo-random permutation of `0..n` (xorshift32):
        /// defeats stride prefetching without an RNG dependency.
        fn shuffled(n: usize) -> Vec<usize> {
            let mut order: Vec<usize> = (0..n).collect();
            let mut x: u32 = 0x9e37_79b9;
            for i in (1..n).rev() {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                let j = (x as usize) % (i + 1);
                order.swap(i, j);
            }
            order
        }

        struct Tables {
            std: HashMap<(Str, ShapeList), ShapeId>,
            fx: HashMap<(Str, ShapeList), ShapeId, FxBuildHasher>,
            vec: Vec<(Str, ShapeList, ShapeId)>,
            hybrid: ShapeTransitions,
        }

        fn build(keys: &[Str]) -> Tables {
            let mut std = HashMap::new();
            let mut fx = HashMap::with_hasher(FxBuildHasher);
            let mut vec = Vec::new();
            let mut hybrid = ShapeTransitions::default();
            for (i, &k) in keys.iter().enumerate() {
                let id = ShapeId(i as u32);
                std.insert((k, 0), id);
                fx.insert((k, 0), id);
                vec.push((k, 0, id));
                hybrid.insert(k, 0, id);
            }
            Tables {
                std,
                fx,
                vec,
                hybrid,
            }
        }

        /// Best-of-3 ns/op for the lookups in `order` through `probe`.
        fn bench_probe(
            name: &str,
            k: usize,
            order: &[usize],
            keys: &[Str],
            mut probe: impl FnMut(Str) -> Option<ShapeId>,
        ) {
            // Warmup (also defeats the first-touch page cost).
            let mut acc = 0u64;
            for &i in order {
                acc += probe(keys[i]).map(|s| s.0 as u64).unwrap_or(0);
            }
            std::hint::black_box(acc);
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let mut acc = 0u64;
                let t = Instant::now();
                for &i in order {
                    acc += probe(keys[i]).map(|s| s.0 as u64).unwrap_or(0);
                }
                let ns = t.elapsed().as_nanos() as f64 / order.len() as f64;
                std::hint::black_box(acc);
                if ns < best {
                    best = ns;
                }
            }
            println!("  {name:<12} K={k:<4} {best:8.2} ns/op");
        }

        #[test]
        fn shape_transition_bench() {
            // Functional equivalence first: all four tables answer hits,
            // misses, and wrong-list probes identically.
            let mut heap = Heap::new();
            heap.set_gc_threshold(usize::MAX);
            let ks = keys(&mut heap, 20);
            let t = build(&ks);
            for (i, &k) in ks.iter().enumerate() {
                let want = Some(ShapeId(i as u32));
                assert_eq!(t.std.get(&(k, 0)).copied(), want);
                assert_eq!(t.fx.get(&(k, 0)).copied(), want);
                assert_eq!(
                    t.vec
                        .iter()
                        .find(|(kk, l, _)| *kk == k && *l == 0)
                        .map(|(_, _, id)| *id),
                    want
                );
                assert_eq!(t.hybrid.get(k, 0), want);
                // Wrong list misses everywhere.
                assert_eq!(t.std.get(&(k, 1)).copied(), None);
                assert_eq!(t.fx.get(&(k, 1)).copied(), None);
                assert_eq!(t.hybrid.get(k, 1), None);
            }
            let missing = Str::inline("nope").unwrap();
            assert_eq!(t.std.get(&(missing, 0)).copied(), None);
            assert_eq!(t.hybrid.get(missing, 0), None);
            assert_eq!(t.hybrid.len(), 20);

            println!("shape transition probe (all hits, best of 3):");
            for &k in &[4usize, 16, 64] {
                let ks = keys(&mut heap, k);
                let t = build(&ks);
                // 2M lookups per trial keeps each trial ~0.1s.
                let base_order = shuffled(k);
                let order: Vec<usize> =
                    base_order.iter().cycle().take(2_000_000).copied().collect();
                bench_probe("hashmap_std", k, &order, &ks, |key| {
                    t.std.get(&(key, 0)).copied()
                });
                bench_probe("hashmap_fx", k, &order, &ks, |key| {
                    t.fx.get(&(key, 0)).copied()
                });
                bench_probe("vec_linear", k, &order, &ks, |key| {
                    t.vec
                        .iter()
                        .find(|(kk, l, _)| *kk == key && *l == 0)
                        .map(|(_, _, id)| *id)
                });
                bench_probe("hybrid", k, &order, &ks, |key| t.hybrid.get(key, 0));
            }
        }
    }
}

#[cfg(test)]
mod cold_field_tests {
    use super::{Object, ObjectKind};
    use crate::value::Value;

    /// Regression guard for the cold-field boxing: `Object` was 216
    /// bytes with the ArrayBuffer/typed-array/dense fields inline; the
    /// boxed layout must stay at least 32 bytes under that so GC page
    /// density does not silently regress.
    #[test]
    fn object_size_stays_shrunk() {
        assert!(
            std::mem::size_of::<Object>() <= 216 - 32,
            "size_of::<Object>() = {}, expected <= {}",
            std::mem::size_of::<Object>(),
            216 - 32
        );
    }

    /// An ordinary object carries no cold box; the cold storage appears
    /// only when a cold field is first written (dense storage here, the
    /// typed-array path exercises the same lazy allocation).
    #[test]
    fn ordinary_object_has_no_cold_box_until_cold_use() {
        let mut o = Object::new(ObjectKind::Ordinary);
        assert!(!o.has_cold());
        assert_eq!(o.bytes(), None);
        assert_eq!(o.dense(), None);
        assert_eq!(o.ab_max_byte_length(), None);
        assert!(!o.ab_shared());
        assert!(!o.view_length_tracking());

        o.set_dense(Some(vec![Some(Value::Int(1)), None]));
        assert!(o.has_cold());
        assert_eq!(o.dense().map(|d| d.len()), Some(2));
        assert_eq!(o.dense().and_then(|d| d[0]), Some(Value::Int(1)));

        let taken = o.take_dense();
        assert_eq!(taken.map(|d| d.len()), Some(2));
        assert_eq!(o.dense(), None);
    }

    /// ArrayBuffer-style cold use: byte storage plus the buffer flags
    /// round-trip through the accessors, and detaching (bytes back to
    /// `None`) stays distinguishable from a zero-length buffer.
    #[test]
    fn array_buffer_cold_fields_round_trip() {
        let mut o = Object::new(ObjectKind::ArrayBuffer);
        assert!(!o.has_cold());
        o.set_bytes(Some(Box::new([1, 2, 3])));
        o.set_ab_max_byte_length(Some(8));
        assert!(o.has_cold());
        assert_eq!(o.bytes(), Some(&[1u8, 2, 3][..]));
        assert_eq!(o.ab_max_byte_length(), Some(8));
        o.set_bytes(Some(Box::new([])));
        assert_eq!(o.bytes(), Some(&[][..]));
        o.set_bytes(None);
        assert_eq!(o.bytes(), None);
    }
}
