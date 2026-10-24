//! Precise non-moving mark-sweep garbage collector with explicit rooting.
//!
//! v1 design: the [`Heap`] owns every allocation. [`Gc`] is a `Copy`
//! raw-pointer handle into heap-owned boxes. Individual allocations are
//! never moved (boxes keep stable addresses even when the owning `Vec`
//! reallocates), so a [`Gc`] stays valid as long as its allocation is live.
//! Dead allocations are reclaimed by [`Heap::collect`], a precise
//! mark-sweep: every heap pointer inside a live object is enumerated by
//! [`Tracer`] (there are no interior untracked pointers; see below), so
//! the marked set is exact.
//!
//! # Rooting discipline
//!
//! A [`Gc`] handle is only valid while its allocation is live. An
//! allocation is live when it is reachable from a root during
//! [`Heap::collect`]. Roots are:
//!
//! * every [`HandleScope`] registered with the heap (see [`Handle`]),
//! * the heap's temporary roots (used internally across allocation points),
//! * every embedder [`Trace`] root registered with
//!   [`Heap::add_trace_root`] (e.g. the interpreter's operand stack,
//!   frames, and promise/timer tables),
//! * the intrinsics fields (`object_prototype`, `array_prototype`, …).
//!
//! The interned-string table is a **weak** root set: an interned string
//! that is otherwise unreachable is freed and evicted from the table, so
//! [`Heap::intern`] never resurrects a dead handle (a later `intern` of the
//! same contents allocates fresh). Strings still reachable — as property
//! keys, function names, or ordinary values — are marked through their
//! owners as usual. Callers holding an interned handle across further
//! allocations must root it first; see [`Heap::intern`]'s contract.
//!
//! To keep a value alive across a collection, put it in a [`HandleScope`]
//! with [`HandleScope::add`], which returns a [`Handle`]: a typed, `Copy`
//! smart pointer whose referent the scope keeps alive. A [`Handle`] must
//! not outlive its scope, and neither may outlive the [`Heap`].
//!
//! Automatic collection is **disabled by default** (`gc_threshold` is
//! `usize::MAX`); call [`Heap::set_gc_threshold`] or [`Heap::collect`]
//! explicitly. When enabled, each collection advances the threshold to at
//! least twice the surviving live set, so a heap that just survived a
//! collection near the threshold does not immediately collect again.
//! Builtin (`NativeFn`) implementations allocate through `&mut Heap` and
//! must root any handle that has to survive an allocation they perform.
//!
//! # Heap identity
//!
//! Every [`Heap`] has a process-unique [`id`][Heap::id], and every [`Gc`]
//! records the id of the heap that created it. The engine's [`Value`] is
//! lifetime-free, so handles cannot be borrow-tied to `&Heap` without a
//! cross-crate breaking change; the id is the dynamic equivalent of that
//! heap lifetime. Every heap operation checks handle membership and panics
//! on cross-heap use instead of silently corrupting memory, and a stale
//! handle can never be mistaken for a handle of a younger heap (ids are
//! never reused, even after a heap is dropped).
//!
//! # Safety contract
//!
//! A [`Gc`] must not outlive the [`Heap`] that created it, and must not be
//! used after a collection that did not keep its allocation alive (i.e. it
//! was not reachable from a root). Handles never escape the
//! single-threaded evaluation that owns the heap, so this is upheld by
//! construction today. Cross-heap misuse (a handle of heap A passed to
//! heap B) is rejected loudly: every heap entry point validates the
//! handle's heap id and panics on mismatch.
//!
//! Mutation of heap objects goes through `&mut Heap` (e.g.
//! [`Heap::define_property`]). The shared references produced by [`Gc`]'s
//! `Deref` are short-lived reads; a collection never moves allocations, so
//! holding such a read across an allocation call is sound as long as the
//! referent is rooted (or no collection ran). The one hard rule: no shared
//! `&` derived from a handle may be *used* across a call that mutably
//! borrows the same allocation (`get_object_mut`, used internally by the
//! `define_property` family). In-crate code upholds this by never retaining
//! derefs across those calls.
//!
//! [`HandleScope`] pins itself to its heap: it captures the heap's id at
//! creation, [`HandleScope::add`] rejects handles from any other heap, and
//! dropping the scope unregisters from that same heap. The heap itself must
//! still outlive its scopes and must not be moved while a scope is
//! registered (scopes hold a raw heap pointer; see [`HandleScope::new`]).
//!
//! # Interior pointers (audit)
//!
//! Every pointer from one heap object to another is a [`Gc`] and is traced:
//! [`Object`] fields (`kind`, `prototype`, property keys and values),
//! including [`FunctionData`](crate::value::FunctionData)'s `name`, which
//! [`Tracer::mark_object`] traces explicitly (inline [`Str`](crate::value::Str)
//! keys and names carry no heap pointer and need no tracing).
//! [`NativeFn`] is a plain function pointer, not a heap pointer. [`Symbol`],
//! [`BigInt`], and [`GcStr`] are leaves.

use std::borrow::Borrow;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::value::{
    AccessorProp, BigInt, GcStr, InlineCache, NativeError, NativeFn, Object, ObjectKind, Property,
    Shape, ShapeId, ShapeList, ShapeTransitions, Str, Symbol, Value,
};
use crate::wtf8::{Wtf8, Wtf8Buf};

/// Fallible-allocation failure: the global allocator refused a layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocError;

/// `Box::new` that reports allocation failure instead of aborting.
/// (`Box::try_new` is unstable on this toolchain, so this goes through
/// [`std::alloc`] directly.)
fn try_box<T>(value: T) -> Result<Box<T>, AllocError> {
    // Test-only allocation-failure injection (see `FAIL_ALLOC`): the only
    // deterministic way to exercise fallible-allocation paths in a unit
    // test. Compiled out of non-test builds.
    #[cfg(test)]
    if fail_alloc_injected() {
        return Err(AllocError);
    }
    use std::alloc::{Layout, alloc};
    if Layout::new::<T>().size() == 0 {
        // Zero-sized: no allocation to fail.
        return Ok(Box::new(value));
    }
    let layout = Layout::new::<T>();
    // SAFETY: `layout` describes `T` exactly (size > 0, correct alignment),
    // so a non-null return is a block usable for one `T`.
    let ptr = unsafe { alloc(layout) } as *mut T;
    if ptr.is_null() {
        return Err(AllocError);
    }
    // SAFETY: `ptr` is valid for writes of `T` and was freshly allocated, so
    // no other code can observe the uninitialized memory; `Box::from_raw`
    // takes ownership of a global-allocator block with `T`'s layout, which
    // is exactly what `Box` will deallocate.
    unsafe {
        std::ptr::write(ptr, value);
        Ok(Box::from_raw(ptr))
    }
}

#[cfg(test)]
thread_local! {
    /// When set, [`try_box`] fails every allocation. Test-only
    /// allocation-failure injection: the real global allocator has no
    /// deterministic failure path a unit test could trigger. Thread-local
    /// so tests running on other threads are unaffected.
    static FAIL_ALLOC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Test-only: force [`try_box`] (and thus every fallible heap
/// allocation) to fail, or stop failing. Pair with a guard that resets
/// it: a leaked `true` would fail unrelated tests on this thread.
#[cfg(test)]
fn set_fail_alloc(fail: bool) {
    FAIL_ALLOC.with(|c| c.set(fail));
}

#[cfg(test)]
fn fail_alloc_injected() -> bool {
    FAIL_ALLOC.with(|c| c.get())
}

/// Process-unique id source for [`Heap`]s. Ids are never reused, so a
/// [`Gc`]'s `heap_id` unambiguously names its creating heap even after that
/// heap is dropped: a stale handle can never be mistaken for a handle of a
/// younger heap.
static NEXT_HEAP_ID: AtomicU64 = AtomicU64::new(1);

/// A fresh [`Heap`] id. Never 0 (`heap_id == 0` would mean "no heap"); id
/// exhaustion would take 2^64 heaps. Relaxed ordering suffices: ids only
/// need uniqueness, not synchronization with any other state.
fn alloc_heap_id() -> u64 {
    NEXT_HEAP_ID.fetch_add(1, Ordering::Relaxed).max(1)
}

/// A garbage-collected pointer to a heap allocation.
///
/// `Copy`, pointer-compared, hashable. Derefs to a shared reference; see
/// the module docs for the lifetime contract (must be rooted across any
/// collection that runs while the handle is held).
///
/// # Heap lifetime
///
/// The handle records the [`Heap`] that created it in [`heap_id`][Self::heap_id].
/// The engine's [`Value`] is lifetime-free, so handles cannot be borrow-tied
/// to `&Heap` (that would be a cross-crate breaking change); the id is the
/// dynamic equivalent of the heap lifetime. Every heap operation validates
/// it and panics on cross-heap use instead of silently corrupting memory.
pub struct Gc<T: ?Sized> {
    ptr: NonNull<T>,
    /// Id of the [`Heap`] that created this handle. Checked by every heap
    /// operation; see the "Heap identity" section in the module docs.
    heap_id: u64,
}

impl<T: ?Sized> Gc<T> {
    fn new(ptr: NonNull<T>, heap_id: u64) -> Self {
        Self { ptr, heap_id }
    }

    /// The raw pointer. Used by the collector's sweep bookkeeping.
    pub fn as_ptr(self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Id of the heap that created this handle.
    pub fn heap_id(self) -> u64 {
        self.heap_id
    }
}

impl<T: ?Sized> Clone for Gc<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for Gc<T> {}

impl<T: ?Sized> PartialEq for Gc<T> {
    fn eq(&self, other: &Self) -> bool {
        // Address comparison only; metadata (for `?Sized`) is irrelevant
        // because all heap types are `Sized` in practice.
        std::ptr::addr_eq(self.as_ptr(), other.as_ptr())
    }
}

impl<T: ?Sized> Eq for Gc<T> {}

impl<T: ?Sized> Hash for Gc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
    }
}

impl<T: ?Sized> fmt::Debug for Gc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gc({:p})", self.ptr)
    }
}

impl<T: ?Sized> std::ops::Deref for Gc<T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: `ptr` was produced from an allocation owned by the
        // creating `Heap` (a `Box<T>` or an arena block), and the
        // collector never moves allocations, so the address stays valid
        // while the allocation is live. By the documented contract the
        // caller only derefs a handle whose allocation is live (rooted
        // across any intervening collection, from the same heap —
        // cross-heap handles are rejected at every heap entry point).
        // Callers must not use the resulting shared reference across a
        // mutating borrow of the same allocation (see module docs); the
        // heap's own mutation helpers uphold this.
        unsafe { self.ptr.as_ref() }
    }
}

/// A rooted handle to a heap allocation: a typed, `Copy` smart pointer like
/// [`Gc`], except the referent is guaranteed live because the originating
/// [`HandleScope`] holds a root for it.
///
/// The `'s` lifetime borrows the scope: a `Handle` cannot outlive the
/// [`HandleScope`] that rooted it (the borrow checker rejects dropping or
/// moving the scope while any handle is live), so the scope's registered
/// root list — which is what actually keeps the referent alive across
/// collections — is still in place whenever the handle is usable. Dropping
/// the scope first is a compile error, not a use-after-free.
///
/// Created only by [`HandleScope::add`], which rejects handles from any
/// heap other than the scope's own. The [`Heap`] must still outlive both.
pub struct Handle<'s, T> {
    gc: Gc<T>,
    _scope: PhantomData<&'s HandleScope>,
    _not_thread_safe: PhantomData<*const ()>,
}

impl<T> Handle<'_, T> {
    /// The underlying [`Gc`] pointer.
    //
    // (Test-only: used by the GC unit tests to unwrap handles.)
    #[cfg(test)]
    pub fn get(self) -> Gc<T> {
        self.gc
    }
}

impl<T> Clone for Handle<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<'_, T> {}

impl<T> fmt::Debug for Handle<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Handle({:p})", self.gc.as_ptr())
    }
}

impl<T> std::ops::Deref for Handle<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: the scope this handle borrows is still alive (the borrow
        // checker guarantees it), so its registered root list still keeps
        // the referent live; the pointer itself is non-moving per the `Gc`
        // contract. The handle's heap membership was validated by
        // `HandleScope::add`.
        &self.gc
    }
}

/// An explicit rooting scope. Values added with [`HandleScope::add`] stay
/// alive until the scope is dropped (scopes nest; dropping an inner scope
/// unroots its values at the next collection).
///
/// The scope is pinned to the heap it was created with: it captures the
/// heap's id, [`add`][Self::add] rejects handles from any other heap, and
/// dropping the scope unregisters from that same heap (panicking if the
/// heap pointer no longer refers to it). The scope registers its root list
/// with the [`Heap`] on creation and unregisters on drop.
pub struct HandleScope {
    heap: *mut Heap,
    /// Id of the heap this scope is pinned to, captured in [`new`][Self::new].
    heap_id: u64,
    // `Box::into_raw`'d root list: the address stays stable when the scope
    // itself is moved, and the heap's registered pointer keeps valid
    // provenance for the scope's whole life. (Deriving it from a `&mut`
    // reborrow would be undefined behavior: returning the scope from `new`
    // moves the box, and moving a `Box` retags under Stacked Borrows,
    // popping the derived tag — the same hazard as `Heap::alloc_object`.)
    // Rebuilt and freed in `Drop`.
    roots: NonNull<Vec<Value>>,
}

impl HandleScope {
    /// Create a scope rooted in `heap` and register it for collections.
    ///
    /// # Safety
    ///
    /// * `heap` must point to a valid [`Heap`].
    /// * `heap` must have been created with `std::ptr::addr_of_mut!` on the
    ///   heap's place (not from a `&mut` reborrow like `&mut heap as *mut
    ///   Heap`). A reborrowed pointer carries a child tag that the caller's
    ///   later `&mut` uses pop, so `Drop`'s `(*self.heap)` would be Stacked
    ///   Borrows UB; `addr_of_mut!` carries the place's own tag, which
    ///   outlives every later `&mut` reborrow.
    /// * The heap must outlive the scope (drop scopes before the heap).
    /// * The heap must not be moved while the scope is registered (the
    ///   scope holds a raw heap pointer).
    /// * Single-threaded use only: no `Heap::collect` may run concurrently
    ///   with scope creation/destruction (upheld: the engine is
    ///   single-threaded).
    pub unsafe fn new(heap: *mut Heap) -> Self {
        // SAFETY: `heap` is valid per the caller's contract; reading its id
        // is a plain field read of a live heap.
        let heap_id = unsafe { (*heap).id };
        // `Box::into_raw` does not retag, so the registered pointer carries
        // the allocation's own provenance, which `Heap::collect`'s `&*roots`
        // can use for the scope's whole life.
        // SAFETY: `Box::into_raw` never returns null for an owned box.
        let roots = NonNull::new(Box::into_raw(Box::new(Vec::new()))).unwrap();
        let scope = HandleScope {
            heap,
            heap_id,
            roots,
        };
        // SAFETY: `heap` is valid per the caller's contract, and the scope
        // is fully constructed before registration, so the heap cannot
        // observe a half-built scope.
        unsafe {
            (*heap).push_scope(roots.as_ptr());
        }
        scope
    }

    /// Root `g` in this scope and return a [`Handle`] to it. The referent
    /// stays alive at least until this scope is dropped, and the returned
    /// handle borrows the scope, so using it after the scope is dropped
    /// is a compile error.
    ///
    /// Takes `&self`: several handles may be live at once (each borrows
    /// the scope immutably, which still pins it — the scope cannot be
    /// dropped or moved while any borrow is live). The push itself goes
    /// through the scope's raw root-list pointer, so no `&mut` aliasing
    /// is created; single-threaded, non-reentrant use keeps this sound
    /// (see the `add` safety notes on the raw pointer).
    ///
    /// # Panics
    ///
    /// Panics if `g` was created by a different heap: rooting a foreign
    /// handle would make this heap's collector dereference memory it does
    /// not own.
    pub fn add<'s, T>(&'s self, g: Gc<T>) -> Handle<'s, T>
    where
        Gc<T>: Into<Value>,
    {
        assert!(
            g.heap_id == self.heap_id,
            "cross-heap HandleScope::add: handle belongs to heap #{}, scope pinned to heap #{}",
            g.heap_id,
            self.heap_id,
        );
        // SAFETY: `roots` is a live `Box::into_raw` allocation owned by
        // this scope (rebuilt only in `Drop`), so the pointer is valid;
        // the heap's shared `&*roots` in `Heap::collect` can never overlap
        // this call (`collect` never calls back into scope code, and the
        // engine is single-threaded).
        unsafe { (*self.roots.as_ptr()).push(g.into()) };
        Handle {
            gc: g,
            _scope: PhantomData,
            _not_thread_safe: PhantomData,
        }
    }

    /// Root a [`Value`]'s heap payload, if any: the referent stays alive
    /// until the scope is dropped. Unlike [`add`](Self::add), takes the
    /// value directly and returns no typed [`Handle`] — the caller keeps
    /// whatever handle it already holds.
    ///
    /// Values with no heap payload (ints, bools, inline strings, …) are
    /// ignored: rooting them would accumulate garbage in the roots vector.
    ///
    /// # Panics
    ///
    /// Panics if the value carries a handle created by a different heap:
    /// rooting a foreign handle would make this heap's collector
    /// dereference memory it does not own.
    pub fn add_value(&self, v: Value) {
        let foreign = match v {
            Value::Str(s) => s.heap_handle().map(|g| g.heap_id()),
            Value::Object(g) => Some(g.heap_id()),
            Value::Symbol(g) => Some(g.heap_id()),
            Value::BigInt(g) => Some(g.heap_id()),
            // Immediates and inline strings: nothing to root.
            _ => None,
        };
        let Some(id) = foreign else { return };
        assert!(
            id == self.heap_id,
            "cross-heap HandleScope::add_value: handle belongs to heap #{id}, scope pinned to heap #{}",
            self.heap_id,
        );
        // SAFETY: as in `add` above.
        unsafe { (*self.roots.as_ptr()).push(v) };
    }
}

impl Drop for HandleScope {
    fn drop(&mut self) {
        let roots_ptr: *mut Vec<Value> = self.roots.as_ptr();
        // SAFETY: the heap outlives the scope per `new`'s contract, and
        // `roots_ptr` is the exact pointer registered at construction.
        // `pop_scope` re-validates the heap id before touching the heap.
        unsafe {
            (*self.heap).pop_scope(roots_ptr, self.heap_id);
        }
        // SAFETY: `roots` came from `Box::into_raw` in `new`, and this
        // rebuilds its unique owner exactly once: the heap unregistered
        // the pointer above and the scope is being dropped, so nothing
        // can access the allocation afterwards.
        drop(unsafe { Box::from_raw(roots_ptr) });
    }
}

/// Anything that can enumerate the [`Gc`] handles it owns, for the
/// collector's mark phase.
///
/// # Safety
///
/// `trace` must report every [`Gc`] handle the implementor owns (directly
/// or transitively through other `Trace` values). Missing a handle lets the
/// collector free a live allocation; reporting a non-owned handle keeps
/// garbage alive (safe but leaks).
pub unsafe trait Trace {
    /// Report every owned handle to `t`.
    fn trace(&self, t: &mut Tracer);
}

/// Fixed hash for allocation addresses: splitmix64.
///
/// Deterministic (no per-process seed to store in the heap) and an order of
/// magnitude cheaper per call than the `RandomState` SipHash that
/// `HashSet<usize>` paid on every insert/lookup. The avalanche is strong
/// enough for box addresses, whose low bits are alignment-poor.
fn hash_addr(addr: usize) -> u64 {
    let mut x = (addr as u64).wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// The mark-phase membership set: addresses of allocations proven live.
///
/// Open-addressing hash table over a single flat `Vec<usize>`: linear
/// probing, `usize::MAX` as the empty sentinel (no real allocation ever
/// sits at that address), power-of-two capacity, load factor capped at 1/2.
/// Three wins over `HashSet<usize>` for this workload:
///
/// * one contiguous allocation with no per-entry overhead; probing walks a
///   cache-friendly array, and [`hash_addr`] replaces SipHash;
/// * capacity is pre-sized from the heap's live count before marking, so a
///   collection never rehashes mid-mark;
/// * the table is owned by the [`Heap`] and reused across collections
///   ([`clear`][Self::clear] keeps the buffer), so steady-state
///   collections allocate nothing for the marked set at all.
///
/// Why not the alternatives: a bitset indexed by allocation slot does not
/// fit this allocator — marks arrive as arbitrary stable addresses with no
/// address→slot map (building one would just be another hash table). A
/// sorted-`Vec` with dedup would pay O(M log M) plus duplicate memory on
/// the duplicate-heavy mark path (e.g. one shared prototype marked once
/// per object); this set dedups on insert at amortized O(1).
///
/// Semantics stay exactly the old ones: precise address-equality
/// membership, duplicates reported as already-present.
struct MarkedSet {
    /// Empty, or a power-of-two length with every entry `usize::MAX`.
    slots: Vec<usize>,
    len: usize,
}

/// Empty sentinel. Also the fail-safe value: the impossible
/// `insert(usize::MAX)`/`contains(usize::MAX)` degrades to "already marked",
/// i.e. a leak, never a wrongful free.
const MARKED_EMPTY: usize = usize::MAX;
/// Smallest table capacity (power of two); load ≤ 1/2 ⇒ ≥ 8 entries.
const MARKED_MIN_CAP: usize = 16;

impl MarkedSet {
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            len: 0,
        }
    }

    /// Number of marked addresses. Read at the end of each collection to
    /// size the next collection's mark set against the live set.
    fn len(&self) -> usize {
        self.len
    }

    /// Current slot capacity (test-only): the table never shrinks, so a
    /// high-mortality collection must not inflate this toward the dead
    /// count.
    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.slots.len()
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Ensure room for `n` entries without rehashing (load factor ≤ 1/2).
    /// Never shrinks: the buffer tracks the largest live set seen.
    fn reserve(&mut self, n: usize) {
        let want = n.saturating_mul(2);
        if want <= self.slots.len() {
            return;
        }
        self.rehash(want.max(MARKED_MIN_CAP).next_power_of_two());
    }

    fn grow(&mut self) {
        let new_cap = if self.slots.is_empty() {
            MARKED_MIN_CAP
        } else {
            // `slots.len()` is a power of two bounded by live memory; the
            // doubling cannot overflow in practice.
            self.slots.len() * 2
        };
        self.rehash(new_cap);
    }

    fn rehash(&mut self, new_cap: usize) {
        debug_assert!(new_cap.is_power_of_two());
        debug_assert!(new_cap >= self.len * 2);
        let mut new_slots = vec![MARKED_EMPTY; new_cap];
        let mask = new_cap - 1;
        for &addr in &self.slots {
            if addr == MARKED_EMPTY {
                continue;
            }
            let mut i = (hash_addr(addr) as usize) & mask;
            while new_slots[i] != MARKED_EMPTY {
                i = (i + 1) & mask;
            }
            new_slots[i] = addr;
        }
        self.slots = new_slots;
    }

    /// Insert `addr`; returns true if it was newly marked.
    fn insert(&mut self, addr: usize) -> bool {
        debug_assert_ne!(
            addr, MARKED_EMPTY,
            "marked address collides with the empty sentinel"
        );
        if self.slots.is_empty() || self.len * 2 >= self.slots.len() {
            self.grow();
        }
        let mask = self.slots.len() - 1;
        let mut i = (hash_addr(addr) as usize) & mask;
        loop {
            let s = self.slots[i];
            if s == addr {
                return false;
            }
            if s == MARKED_EMPTY {
                self.slots[i] = addr;
                self.len += 1;
                return true;
            }
            i = (i + 1) & mask;
        }
    }

    /// True if `addr` was marked.
    fn contains(&self, addr: usize) -> bool {
        if self.slots.is_empty() {
            return false;
        }
        let mask = self.slots.len() - 1;
        let mut i = (hash_addr(addr) as usize) & mask;
        loop {
            let s = self.slots[i];
            if s == addr {
                return true;
            }
            if s == MARKED_EMPTY {
                return false;
            }
            i = (i + 1) & mask;
        }
    }

    /// Drop all marks, keeping the buffer for the next collection.
    fn clear(&mut self) {
        self.slots.fill(MARKED_EMPTY);
        self.len = 0;
    }
}

impl Default for MarkedSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Marks the set of allocations reachable from a root set.
/// Cycle-safe: each allocation is expanded at most once.
///
/// The marker is iterative: newly marked objects go on an explicit worklist
/// and are expanded by [`drain`][Self::drain], so arbitrarily deep object
/// graphs (a 100k-long property chain, a deep prototype chain) cannot
/// overflow the call stack the way a recursive marker would.
pub struct Tracer {
    marked: MarkedSet,
    /// Objects marked but not yet expanded. Leaves (strings, symbols,
    /// bigints) have no outgoing edges and are never queued.
    worklist: Vec<Value>,
}

impl Tracer {
    /// Empty tracer.
    pub fn new() -> Self {
        Self::with_marked(MarkedSet::new(), Vec::new())
    }

    /// Tracer over reused buffers (see [`Heap::marked_set`] and
    /// [`Heap::worklist`]). Both buffers must be empty; they are returned
    /// empty by [`take_marked`][Self::take_marked].
    fn with_marked(marked: MarkedSet, worklist: Vec<Value>) -> Self {
        debug_assert!(marked.is_empty());
        debug_assert!(worklist.is_empty());
        Self { marked, worklist }
    }

    /// How many distinct allocations have been marked.
    //
    // (Test-only mark-phase diagnostic asserted by the GC unit tests; will
    // feed collection-tuning heuristics.)
    #[cfg(test)]
    pub fn marked_count(&self) -> usize {
        self.marked.len()
    }

    /// The marked address set and the drained worklist, for the sweep phase
    /// and buffer reuse.
    fn take_marked(self) -> (MarkedSet, Vec<Value>) {
        debug_assert!(
            self.worklist.is_empty(),
            "tracer worklist not drained before sweep"
        );
        (self.marked, self.worklist)
    }

    /// True if the allocation was newly marked.
    fn mark_ptr(&mut self, p: *const ()) -> bool {
        self.marked.insert(p as usize)
    }

    /// Mark the allocation behind `v`; objects are queued for expansion.
    fn push_value(&mut self, v: Value) {
        match v {
            Value::Object(g) => {
                if self.mark_ptr(g.as_ptr() as *const ()) {
                    self.worklist.push(Value::Object(g));
                }
            }
            // Inline strings live in the value itself: nothing to mark.
            Value::Str(s) => {
                if let Some(g) = s.heap_handle() {
                    self.mark_str(g);
                }
            }
            Value::Symbol(g) => self.mark_symbol(g),
            Value::BigInt(g) => self.mark_bigint(g),
            Value::Undefined | Value::Null | Value::Bool(_) | Value::Int(_) | Value::Float(_) => {}
        }
    }

    /// Expand every queued object, queueing its outgoing edges. Iterative:
    /// the loop replaces the recursion a naive marker would use, so deep
    /// graphs cannot overflow the stack. Already-marked objects are never
    /// re-queued, which is what breaks reference cycles.
    fn drain(&mut self) {
        while let Some(v) = self.worklist.pop() {
            // Only objects are ever queued; leaves are filtered defensively.
            let Value::Object(g) = v else { continue };
            // Interior heap pointers inside the kind tag; must be traced.
            // (Views and DataViews used to keep their buffer alive through
            // an object-valued hidden property; the tag-carried handle
            // below replaces that edge.)
            if let ObjectKind::Function(data) = g.kind {
                self.push_value(Value::from(data.name));
            }
            match g.kind {
                ObjectKind::TypedArrayView { buffer, .. } | ObjectKind::DataView { buffer, .. } => {
                    self.push_value(Value::Object(buffer));
                }
                _ => {}
            }
            for (k, p) in g.properties.iter() {
                self.push_value(Value::from(*k));
                self.push_value(p.value);
            }
            for (k, a) in g.accessors.iter() {
                self.push_value(Value::from(*k));
                self.push_value(a.get);
                self.push_value(a.set);
            }
            // Symbol-keyed side tables bypass the shape, but their keys
            // are heap handles: mark each key or a collection frees a
            // symbol referenced only as a key (use-after-free on the next
            // keyed lookup).
            for (k, p) in g.symbol_properties.iter() {
                self.mark_symbol(*k);
                self.push_value(p.value);
            }
            for (k, a) in g.symbol_accessors.iter() {
                self.mark_symbol(*k);
                self.push_value(a.get);
                self.push_value(a.set);
            }
            // Dense element storage holds live values; trace them so a
            // collection cannot free an element still referenced only by
            // the dense array.
            if let Some(dense) = g.dense() {
                for slot in dense.iter().flatten() {
                    self.push_value(*slot);
                }
            }
            self.push_value(g.prototype);
        }
    }

    /// Mark a string handle (leaf).
    pub fn mark_str(&mut self, g: Gc<GcStr>) {
        self.mark_ptr(g.as_ptr() as *const ());
    }

    /// Mark a symbol handle (leaf).
    pub fn mark_symbol(&mut self, g: Gc<Symbol>) {
        self.mark_ptr(g.as_ptr() as *const ());
    }

    /// Mark a bigint handle (leaf).
    pub fn mark_bigint(&mut self, g: Gc<BigInt>) {
        self.mark_ptr(g.as_ptr() as *const ());
    }

    /// Mark an object and transitively its kind payload, keys, values, and
    /// prototype.
    pub fn mark_object(&mut self, g: Gc<Object>) {
        self.push_value(Value::Object(g));
        self.drain();
    }

    /// Mark whatever a value points at.
    pub fn mark_value(&mut self, v: Value) {
        self.push_value(v);
        self.drain();
    }
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new()
    }
}

/// Keep allocations whose address is in `marked`; drop the rest.
/// Non-moving: survivors keep their addresses, so live [`Gc`]s stay valid.
/// Dead allocations are handed to `reclaim`, which takes unique ownership
/// of each one exactly once: either freeing the box (`free_boxed`, the
/// per-type registries) or dropping the referent in place and pushing the
/// slot onto its arena page's freelist (`reclaim_object_slot`, the object
/// pages). One shared marked-set walk keeps the sweep logic — and its
/// safety reasoning — in a single place as more types move to arenas; the
/// [`MarkedSet`] address-based marking itself is untouched.
///
/// `note_live` runs for every survivor. The string registry uses it to
/// accumulate per-block byte sizes (one integer add per live block); the
/// fixed-size registries pass a no-op and multiply the returned survivor
/// count by their block size — one multiply per registry, no per-block
/// work — so the sweep doubles as the heap's byte-accounting pass with
/// no second traversal.
///
/// Returns the survivor count.
fn sweep<T>(
    live: &mut Vec<NonNull<T>>,
    marked: &MarkedSet,
    mut reclaim: impl FnMut(NonNull<T>),
    mut note_live: impl FnMut(NonNull<T>),
) -> usize {
    let mut survivors = 0usize;
    let mut i = 0;
    while i < live.len() {
        // The address alone decides; no reborrow of the allocation.
        let ptr = live[i];
        if marked.contains(ptr.as_ptr() as usize) {
            note_live(ptr);
            survivors += 1;
            i += 1;
        } else {
            live.swap_remove(i);
            // `ptr` was just removed from `live`: the reclaimer takes
            // unique ownership of the dead allocation exactly once.
            reclaim(ptr);
        }
    }
    survivors
}

/// `sweep` reclaimer that frees the allocation: rebuilds the box from the
/// raw pointer and drops it. The behavior for the per-type registries
/// (`strings`, `symbols`, `bigints`), whose pointers came from
/// `Box::into_raw`.
fn free_boxed<T>(ptr: NonNull<T>) {
    // SAFETY: `ptr` was just removed from its registry by `sweep`, so this
    // rebuilds the allocation's unique owner exactly once; dropping the
    // box frees it. The allocation was never moved and never reborrowed
    // through a box since `Box::into_raw` (see the `Heap` struct docs).
    drop(unsafe { Box::from_raw(ptr.as_ptr()) });
}

/// Slots per object arena page. A page is one
/// `Layout::array::<Object>` block of `OBJECT_PAGE_SLOTS *
/// size_of::<Object>()` bytes: large enough to amortize the page
/// allocation over many objects, small enough that a mostly-empty page
/// wastes little before the sweep returns it to the OS.
const OBJECT_PAGE_SLOTS: usize = 128;

/// One bump-allocated page of fixed-size [`Object`] slots: the hot
/// allocation path.
///
/// Slots are handed out by bump pointer; dead slots reclaimed by a sweep
/// are pushed onto `free` for reuse. The collector never moves live slots,
/// so a slot's address is stable from allocation to reclamation and every
/// [`Gc`] keeps its provenance (see the `Heap` struct docs).
struct ObjectPage {
    /// Base of the slot array: the page block's own provenance, from which
    /// every slot pointer is derived by offset.
    base: NonNull<Object>,
    /// Slots handed out so far (bump cursor); always `<= OBJECT_PAGE_SLOTS`.
    used: usize,
    /// Slots currently live (handed out and not yet reclaimed). A page
    /// whose `live` reaches zero in a sweep is deallocated back to the OS.
    live: usize,
    /// Reclaimed slots available for reuse, as a stack.
    free: Vec<*mut Object>,
}

impl ObjectPage {
    /// Allocate a fresh, empty page. Fallible: reports allocation failure
    /// instead of aborting, and honors the `FAIL_ALLOC` test injection
    /// (see [`try_box`]).
    fn new() -> Result<Self, AllocError> {
        #[cfg(test)]
        if fail_alloc_injected() {
            return Err(AllocError);
        }
        use std::alloc::{Layout, alloc};
        let layout = Layout::array::<Object>(OBJECT_PAGE_SLOTS).map_err(|_| AllocError)?;
        // SAFETY: `layout` describes an array of `OBJECT_PAGE_SLOTS`
        // `Object`s (size > 0); a non-null return is a block usable for
        // that many slots.
        let ptr = unsafe { alloc(layout) } as *mut Object;
        if ptr.is_null() {
            return Err(AllocError);
        }
        Ok(ObjectPage {
            // SAFETY: `alloc` never returns null on success.
            base: unsafe { NonNull::new_unchecked(ptr) },
            used: 0,
            live: 0,
            free: Vec::new(),
        })
    }

    /// Hand out one slot: freelist first, else the bump cursor. Returns
    /// `None` when the page is full.
    fn alloc_slot(&mut self) -> Option<NonNull<Object>> {
        if let Some(slot) = self.free.pop() {
            self.live += 1;
            // SAFETY: freelist entries are valid, aligned, unused slot
            // addresses pushed by `push_object_freelist`; never null, and
            // the referent was dropped when the slot was reclaimed.
            return Some(unsafe { NonNull::new_unchecked(slot) });
        }
        if self.used < OBJECT_PAGE_SLOTS {
            // SAFETY: `used < OBJECT_PAGE_SLOTS`, so the slot lies inside
            // the page block; `size_of::<Object>()` is the slot stride
            // (size is always a multiple of alignment).
            let slot = unsafe { self.base.as_ptr().add(self.used) };
            self.used += 1;
            self.live += 1;
            // SAFETY: `base` is non-null and `add` preserves non-null.
            return Some(unsafe { NonNull::new_unchecked(slot) });
        }
        None
    }

    /// Deallocate the page block back to the OS. The caller must have
    /// dropped every live slot's `Object` already (see `Heap`'s `Drop`);
    /// dead slots were dropped when the sweep reclaimed them.
    fn dealloc(self) {
        use std::alloc::{Layout, dealloc};
        // `OBJECT_PAGE_SLOTS` is a nonzero constant: the layout
        // computation that succeeded in `new` cannot fail here.
        let layout = Layout::array::<Object>(OBJECT_PAGE_SLOTS).expect("object page layout");
        // SAFETY: `base` is the block `new` allocated with this layout,
        // and every slot's referent has been dropped.
        unsafe { dealloc(self.base.as_ptr() as *mut u8, layout) };
    }
}

/// Push a dead object slot onto its page's freelist. Pages are kept sorted
/// by base address, so the owning page is found by binary search. A page
/// whose `live` count reaches zero is fully empty: it is deallocated back
/// to the OS (`Vec::remove`, not `swap_remove`, keeps the list sorted).
/// Every handed-out slot was reclaimed by this point, so no freelist entry
/// can dangle.
fn push_object_freelist(pages: &mut Vec<ObjectPage>, ptr: NonNull<Object>) {
    let addr = ptr.as_ptr() as usize;
    let page_bytes = std::mem::size_of::<Object>() * OBJECT_PAGE_SLOTS;
    let idx = pages.partition_point(|p| (p.base.as_ptr() as usize) <= addr);
    let Some(i) = idx.checked_sub(1) else {
        unreachable!("object slot {ptr:p} is not owned by any object page");
    };
    let base = pages[i].base.as_ptr() as usize;
    if addr >= base + page_bytes {
        unreachable!("object slot {ptr:p} is not owned by any object page");
    }
    pages[i].free.push(ptr.as_ptr());
    pages[i].live -= 1;
    if pages[i].live == 0 {
        pages.remove(i).dealloc();
    }
}

/// `sweep` reclaimer for the object registry: drop the dead `Object` in
/// place and hand its slot to its page's freelist. The slot's address never
/// changes, so surviving [`Gc`]s are unaffected.
fn reclaim_object_slot(pages: &mut Vec<ObjectPage>, ptr: NonNull<Object>) {
    // SAFETY: `ptr` was just removed from the object registry by `sweep`,
    // so this takes unique ownership of the dead slot exactly once;
    // dropping in place ends the referent without touching the page block.
    unsafe { std::ptr::drop_in_place(ptr.as_ptr()) };
    push_object_freelist(pages, ptr);
}

/// Size classes for coalesced string blocks: total block bytes (header +
/// trailing WTF-8 payload). Short heap strings dominate, so the classes
/// start small; payloads above the top class fall back to malloc (see
/// `STR_CLASS_OVERSIZE`).
const STR_CLASS_SIZES: [usize; 6] = [64, 128, 256, 512, 1024, 2048];
/// Number of string size classes.
const STR_NUM_CLASSES: usize = STR_CLASS_SIZES.len();

/// Compile-time pin of the table shape the `str_class_for` bit trick
/// depends on: sizes must be consecutive powers of two starting at 64.
/// A future edit inserting a non-doubling class fails the build instead
/// of silently misclassifying in release, where `debug_assert`s vanish
/// (an undersized block would mean an out-of-bounds write).
const fn str_class_sizes_are_doubling() -> bool {
    if STR_CLASS_SIZES[0] != 64 {
        return false;
    }
    let mut i = 0;
    while i + 1 < STR_CLASS_SIZES.len() {
        if STR_CLASS_SIZES[i + 1] != STR_CLASS_SIZES[i] * 2 {
            return false;
        }
        i += 1;
    }
    true
}
const _: () = assert!(str_class_sizes_are_doubling());
/// `GcStr::class` value for blocks above the top size class: individually
/// malloc'd with a layout recomputed from the header's `len`, freed the
/// same way. Never indexes `STR_CLASS_SIZES` or `str_chunks`.
const STR_CLASS_OVERSIZE: u8 = u8::MAX;
/// Bytes per string chunk: one malloc amortized over many same-class
/// blocks (256 of the smallest class, 8 of the largest).
const STR_CHUNK_BYTES: usize = 16 * 1024;

/// One bump-allocated chunk of same-class string blocks: the hot string
/// allocation path.
///
/// Blocks are handed out by bump pointer; dead blocks reclaimed by a
/// sweep are pushed onto `free` for reuse. The collector never moves live
/// blocks, so a block's address is stable from allocation to reclamation
/// and every [`Gc`] keeps its provenance (see the `Heap` struct docs).
/// The freelist is per-chunk, not per-class: a fully-emptied chunk takes
/// its freelist with it when the sweep deallocates it, so no entry can
/// dangle.
struct StrChunk {
    /// Base of the chunk block: the chunk's own provenance, from which
    /// every block pointer is derived by offset.
    base: NonNull<u8>,
    /// Chunk byte size (always `STR_CHUNK_BYTES`).
    size: usize,
    /// Bump cursor (bytes handed out); always `<= size`.
    used: usize,
    /// Blocks currently live (handed out and not yet reclaimed). A chunk
    /// whose `live` reaches zero in a sweep is deallocated back to the OS.
    live: usize,
    /// Size-class index: every block in the chunk is
    /// `STR_CLASS_SIZES[class]` bytes.
    class: u8,
    /// Reclaimed blocks available for reuse, as a stack.
    free: Vec<NonNull<u8>>,
}

impl StrChunk {
    /// Allocate a fresh, empty chunk for `class`. Fallible: reports
    /// allocation failure instead of aborting, and honors the
    /// `FAIL_ALLOC` test injection (see [`try_box`]).
    fn new(class: u8) -> Result<Self, AllocError> {
        debug_assert!((class as usize) < STR_NUM_CLASSES);
        #[cfg(test)]
        if fail_alloc_injected() {
            return Err(AllocError);
        }
        use std::alloc::{Layout, alloc};
        let layout = Layout::from_size_align(STR_CHUNK_BYTES, GcStr::HEADER_ALIGN)
            .map_err(|_| AllocError)?;
        // SAFETY: `layout` is non-zero with the header's alignment; a
        // non-null return is a block usable for it.
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(AllocError);
        }
        Ok(StrChunk {
            // SAFETY: `alloc` never returns null on success.
            base: unsafe { NonNull::new_unchecked(ptr) },
            size: STR_CHUNK_BYTES,
            used: 0,
            live: 0,
            class,
            free: Vec::new(),
        })
    }

    /// Hand out one block: freelist first, else the bump cursor. Returns
    /// `None` when the chunk is full.
    fn alloc_block(&mut self) -> Option<NonNull<u8>> {
        if let Some(block) = self.free.pop() {
            self.live += 1;
            // SAFETY: freelist entries are valid block bases in this
            // chunk, pushed by `push_str_freelist`; never null, and the
            // block has no live owner (it was reclaimed by a sweep).
            return Some(block);
        }
        let block_size = STR_CLASS_SIZES[self.class as usize];
        if self.used + block_size <= self.size {
            // SAFETY: `used + block_size <= size`, so the block lies
            // inside the chunk; every block starts at a multiple of the
            // class size from the aligned base, so it is header-aligned.
            let block = unsafe { self.base.as_ptr().add(self.used) };
            self.used += block_size;
            self.live += 1;
            // SAFETY: `base` is non-null and `add` preserves non-null.
            return Some(unsafe { NonNull::new_unchecked(block) });
        }
        None
    }

    /// Deallocate the chunk block back to the OS. The caller must own
    /// every block: live blocks' payloads are raw bytes (nothing to
    /// drop), and dead blocks hold no freelist entries outside this
    /// chunk (the freelist is per-chunk; see `Heap`'s `Drop` and the
    /// sweep).
    fn dealloc(self) {
        use std::alloc::{Layout, dealloc};
        // `STR_CHUNK_BYTES` is a nonzero constant: the layout computation
        // that succeeded in `new` cannot fail here.
        let layout = Layout::from_size_align(STR_CHUNK_BYTES, GcStr::HEADER_ALIGN)
            .expect("string chunk layout");
        // SAFETY: `base` is the block `new` allocated with this layout.
        unsafe { dealloc(self.base.as_ptr(), layout) };
    }
}

/// Size class for a `len`-byte payload: the first class whose block fits
/// header + payload, or `STR_CLASS_OVERSIZE` above the top class.
///
/// The class sizes are consecutive powers of two starting at 64, so the
/// class is `ceil(log2(need)) - 6`: one `leading_zeros` instead of a
/// 6-iteration scan. The const assert below pins the table shape this bit
/// trick depends on (the `debug_assert`s only check it in debug builds);
/// update this function if `STR_CLASS_SIZES` changes.
fn str_class_for(len: usize) -> u8 {
    let need = GcStr::HEADER_LEN.saturating_add(len);
    if need > STR_CLASS_SIZES[STR_NUM_CLASSES - 1] {
        return STR_CLASS_OVERSIZE;
    }
    debug_assert_eq!(STR_CLASS_SIZES[0], 64);
    debug_assert!(STR_CLASS_SIZES.windows(2).all(|w| w[1] == w[0] * 2));
    // Position of the highest set bit of `need - 1`, i.e. `ceil(log2(need))`
    // for `need >= 1`. `need >= GcStr::HEADER_LEN >= 1`, so the
    // subtraction cannot underflow; the saturating form is belt-and-braces
    // against a future header change.
    let ceil_lg2 = usize::BITS - need.saturating_sub(1).leading_zeros();
    // Class 0 covers `need <= 64 == 2^6`; each class doubles from there.
    ceil_lg2.saturating_sub(6) as u8
}

/// Layout of an oversize string block: header plus `len` payload bytes.
fn str_oversize_layout(len: usize) -> Result<std::alloc::Layout, AllocError> {
    std::alloc::Layout::from_size_align(GcStr::HEADER_LEN.saturating_add(len), GcStr::HEADER_ALIGN)
        .map_err(|_| AllocError)
}

/// Malloc fallback for payloads above the top size class. Fallible:
/// reports allocation failure instead of aborting, and honors the
/// `FAIL_ALLOC` test injection (see [`try_box`]).
fn alloc_str_oversize(len: usize) -> Result<NonNull<u8>, AllocError> {
    #[cfg(test)]
    if fail_alloc_injected() {
        return Err(AllocError);
    }
    let layout = str_oversize_layout(len)?;
    // SAFETY: the layout is non-zero (it holds the header) with the
    // header's alignment; a non-null return is a block usable for it.
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        return Err(AllocError);
    }
    // SAFETY: `alloc` never returns null on success.
    Ok(unsafe { NonNull::new_unchecked(ptr) })
}

/// Free an oversize string block.
///
/// # Safety
///
/// * `ptr` must be the header of an oversize block with `len` payload
///   bytes, uniquely owned (just removed from the string registry, or
///   being dropped with the heap).
/// * `len` must be the payload length the block was allocated with (the
///   header's own record), so the recomputed layout matches.
unsafe fn dealloc_str_oversize(ptr: NonNull<GcStr>, len: usize) {
    // The allocation succeeded with this layout, so recomputing it
    // cannot fail.
    let layout = str_oversize_layout(len).expect("oversize string layout");
    // SAFETY: `ptr` is the block `alloc_str_oversize` returned for this
    // layout, and the caller guarantees unique ownership.
    unsafe { std::alloc::dealloc(ptr.as_ptr() as *mut u8, layout) };
}

/// Push a dead string block onto its chunk's freelist. Chunks are kept
/// sorted by base address within each class, so the owning chunk is found
/// by binary search. A chunk whose `live` count reaches zero is fully
/// empty: it is deallocated back to the OS (`Vec::remove`, not
/// `swap_remove`, keeps the list sorted). The freelist is per-chunk, so
/// an emptied chunk takes its freelist with it and no entry can dangle.
fn push_str_freelist(chunks: &mut Vec<StrChunk>, ptr: NonNull<GcStr>) {
    let addr = ptr.as_ptr() as usize;
    let idx = chunks.partition_point(|c| (c.base.as_ptr() as usize) <= addr);
    let Some(i) = idx.checked_sub(1) else {
        unreachable!("string block {ptr:p} is not owned by any string chunk");
    };
    let base = chunks[i].base.as_ptr() as usize;
    if addr >= base + chunks[i].size {
        unreachable!("string block {ptr:p} is not owned by any string chunk");
    }
    // SAFETY: `ptr` is non-null, and `as` preserves non-null.
    chunks[i]
        .free
        .push(unsafe { NonNull::new_unchecked(ptr.as_ptr() as *mut u8) });
    chunks[i].live -= 1;
    if chunks[i].live == 0 {
        chunks.remove(i).dealloc();
    }
}

/// `sweep` reclaimer for the string registry: hand the dead block to its
/// chunk's freelist (fully-emptied chunks go back to the OS) or free an
/// oversize block. The payload is raw bytes, so there is nothing to drop
/// in place; the block's address never changes, so surviving [`Gc`]s are
/// unaffected.
fn reclaim_str_block(chunks: &mut [Vec<StrChunk>; STR_NUM_CLASSES], ptr: NonNull<GcStr>) {
    // SAFETY: `ptr` was just removed from the string registry by `sweep`,
    // so this takes unique ownership of the dead block exactly once; the
    // block is still allocated (only this reclaimer and `Heap`'s `Drop`
    // free string blocks), so reading the header fields is sound. The
    // fields are read without forming a reference (`addr_of!`).
    let class = unsafe { std::ptr::addr_of!((*ptr.as_ptr()).class).read() };
    if class == STR_CLASS_OVERSIZE {
        let len = unsafe { std::ptr::addr_of!((*ptr.as_ptr()).len).read() as usize };
        // SAFETY: oversize blocks are individually malloc'd with the
        // layout recomputed here; `len` is the header's own record of the
        // payload size, and unique ownership was taken above.
        unsafe { dealloc_str_oversize(ptr, len) };
        return;
    }
    let Some(class_chunks) = chunks.get_mut(class as usize) else {
        unreachable!("string block {ptr:p} has invalid class {class}");
    };
    push_str_freelist(class_chunks, ptr);
}

/// Live-block byte size for the sweep's accounting pass (`sweep`'s
/// `note_live` for the string registry): the class's block size, or
/// header + payload for malloc-fallback oversize blocks.
fn str_block_bytes(ptr: NonNull<GcStr>) -> usize {
    // SAFETY: `ptr` names a live block (called only for sweep survivors),
    // so the block is still allocated and reading the header fields via
    // `addr_of!` is sound; the block stays allocated for the read (only
    // this sweep and `Heap`'s `Drop` free string blocks). `class` is the
    // block's own size-class record, written by `GcStr::from_buf` from
    // `alloc_str_block`'s return, so it is either a valid class index or
    // `STR_CLASS_OVERSIZE` (mirrors `reclaim_str_block`'s read).
    let class = unsafe { std::ptr::addr_of!((*ptr.as_ptr()).class).read() };
    if class == STR_CLASS_OVERSIZE {
        let len = unsafe { std::ptr::addr_of!((*ptr.as_ptr()).len).read() } as usize;
        // The block was malloc'd with `str_oversize_layout(len)` (see
        // `alloc_str_oversize`), which rounds `HEADER_LEN + len` up to
        // `HEADER_ALIGN`; count the rounded size so the accounting matches
        // the actual allocation. (Diagnostic-only, but keep it exact.)
        if let Ok(layout) = str_oversize_layout(len) {
            return layout.size();
        }
        return GcStr::HEADER_LEN.saturating_add(len);
    }
    STR_CLASS_SIZES[class as usize]
}

/// Key type for the interned-string table: the canonical handle, hashed and
/// compared by the string's contents. `GcStr` is immutable, so content
/// hashing is stable for the entry's lifetime. Each distinct text is stored
/// exactly once — in the [`GcStr`] allocation itself — with no duplicate
/// key `String` in the table; entries are just 16-byte handles.
#[derive(Debug, Clone, Copy)]
struct InternedStr(Gc<GcStr>);

impl PartialEq for InternedStr {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_wtf8() == other.0.as_wtf8()
    }
}

impl Eq for InternedStr {}

impl Hash for InternedStr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_wtf8().hash(state);
    }
}

impl Borrow<Wtf8> for InternedStr {
    fn borrow(&self) -> &Wtf8 {
        self.0.as_wtf8()
    }
}

/// Per-arena live-byte accounting, captured by [`Heap::collect`] during the
/// sweep that already walks every registry (one integer add per live block
/// for strings, one multiply per registry for the fixed-size arenas).
/// Snapshot of the most recent collection; zero before the first one.
///
/// Counts live *block* bytes only: object slots, string blocks, symbol
/// boxes, bigint boxes. Chunk/page slack and separately-allocated payloads
/// (a `Symbol`'s description buffer, a `BigInt`'s digit `Vec`, shape tables,
/// private-state side tables) are not included.
#[derive(Debug, Default, Clone, Copy)]
pub struct GcArenaBytes {
    /// Live object slots × `size_of::<Object>()` (object pages).
    pub objects: usize,
    /// Live string blocks: class block size per survivor, header + payload
    /// for oversize blocks (string chunks).
    pub strings: usize,
    /// Live symbols × `size_of::<Symbol>()`.
    pub symbols: usize,
    /// Live bigints × `size_of::<BigInt>()`.
    pub bigints: usize,
}

impl GcArenaBytes {
    /// Total live bytes across all arenas.
    pub fn total(&self) -> usize {
        self.objects + self.strings + self.symbols + self.bigints
    }
}

/// The JS heap: owns all GC allocations.
// The registries hold raw pointers — never `Box`es — and that is the whole
// point: moving a `Box` retags under Stacked Borrows, which would
// invalidate the provenance of every live [`Gc`] handle to that allocation
// (e.g. across an unrelated `Vec` reallocation). Raw pointers carry the
// allocation's own provenance, which stays valid until the allocation is
// reclaimed. `objects` points into bump-allocated [`ObjectPage`]s and
// `strings` into bump-allocated [`StrChunk`]s (or individually-malloc'd
// oversize blocks): a slot's/block's pointer derives from its page/chunk
// block's provenance by offset and the referent is constructed in place
// with raw writes, so the handle's provenance never depends on a box; the
// other registries (`symbols`, `bigints`) hold `Box::into_raw`'d
// allocations, freed by rebuilding the box in `sweep` or `Heap`'s `Drop`.
// The collector is non-moving: survivors keep their addresses, so live
// [`Gc`]s stay valid.
/// A post-sweep hook: runs at the end of [`Heap::collect`], after dead
/// addresses are reclaimed and while the survivor set is still
/// available. Receives a liveness predicate over raw allocation
/// addresses (`Gc::as_ptr() as usize`); embedders use it to evict weak
/// references (the web-API wrapper-identity cache in `bindings.rs`
/// holds weak values and drops entries whose wrappers did not
/// survive).
///
/// Hooks must not allocate on this heap and must not panic: they run
/// with no GC bookkeeping active, at the same phase as the collector's
/// own table retains.
pub(crate) type PostSweepHook = Box<dyn FnMut(&dyn Fn(usize) -> bool)>;

pub struct Heap {
    /// Process-unique identity of this heap. Every [`Gc`] records it, and
    /// every heap operation asserts membership: cross-heap handle misuse
    /// panics loudly instead of silently corrupting memory.
    id: u64,
    /// Live object slots: pointers into [`ObjectPage`] blocks (not
    /// `Box::into_raw` allocations — see `object_pages`). The registry
    /// keeps the sweep's marked-set walk address-based exactly as before;
    /// only the reclaimer changed (freelist-push instead of box-free).
    objects: Vec<NonNull<Object>>,
    /// Arena pages backing `objects`: bump-allocated fixed-size slots with
    /// per-page freelists, kept sorted by base address so the sweep's
    /// freelist push can binary-search the owning page. Fully-emptied
    /// pages are deallocated back to the OS by the sweep.
    object_pages: Vec<ObjectPage>,
    strings: Vec<NonNull<GcStr>>,
    /// Arena chunks backing `strings`: per-class bump chunks of coalesced
    /// header+payload blocks with per-chunk freelists, each class's list
    /// sorted by base address so the sweep's freelist push can
    /// binary-search the owning chunk. Fully-emptied chunks are
    /// deallocated back to the OS by the sweep. Payloads above the top
    /// class live in individually malloc'd blocks, freed from their
    /// headers (see `STR_CLASS_OVERSIZE`).
    str_chunks: [Vec<StrChunk>; STR_NUM_CLASSES],
    /// Per-class hot-chunk cache: `str_hot[c]` is the index into
    /// `str_chunks[c]` of the chunk that most recently handed out a
    /// block, or `u32::MAX` when none is known.
    /// [`Heap::alloc_str_block`] probes it before the reverse chunk scan;
    /// the sweep's chunk removal can leave it stale, which is safe — a
    /// stale index either names a full chunk and falls through to the
    /// scan, or is out of bounds and is ignored.
    str_hot: [u32; STR_NUM_CLASSES],
    symbols: Vec<NonNull<Symbol>>,
    bigints: Vec<NonNull<BigInt>>,
    /// Canonical handles by contents. Entries are weak: an interned string
    /// that is otherwise unreachable is freed by [`collect`][Self::collect]
    /// and evicted from this table. Keyed by handle (hashed by contents,
    /// see [`InternedStr`]) so each text is stored exactly once, in the
    /// [`GcStr`] allocation itself.
    interned: HashSet<InternedStr>,
    next_symbol_id: u64,
    /// Root lists of live [`HandleScope`]s (raw: scopes unregister on drop).
    scopes: Vec<*mut Vec<Value>>,
    /// Embedder [`Trace`] roots (raw: owners unregister before drop).
    /// The interpreter registers itself here so collections keep its
    /// operand stack, frames, and promise/timer tables alive.
    trace_roots: Vec<*mut dyn Trace>,
    /// General post-sweep hooks, run at the end of every [`Heap::collect`]
    /// while the survivor set is still available. The web-context
    /// wrapper-cache hook has its own idempotent slot (see
    /// [`Heap::set_wrapper_cache_hook`]) instead of living here.
    post_sweep_hooks: Vec<PostSweepHook>,
    /// The web-context wrapper-cache eviction hook (see
    /// [`Heap::set_wrapper_cache_hook`]). A single slot, not an entry
    /// in `post_sweep_hooks`: reinstalling a context on the same heap
    /// replaces it instead of accumulating one dead entry per install
    /// (`remove_context` never runs while the heap survives).
    wrapper_cache_hook: Option<PostSweepHook>,
    /// Roots held across a single heap operation (e.g. `define_property`
    /// roots its object across the interning allocation, which may collect).
    temp_roots: Vec<Value>,
    /// Allocation count at which the next allocation triggers a collection.
    /// `usize::MAX` (the default) disables automatic collection. Each
    /// collection advances an enabled threshold to at least twice the
    /// surviving live set.
    gc_threshold: usize,
    /// Live addresses marked by the previous collection (objects, strings,
    /// symbols, bigints, plus live shape key strings). The next
    /// collection pre-sizes its mark set against this instead of the
    /// total allocation count, so a high-mortality heap does not
    /// permanently inflate the reused mark buffer to garbage-inclusive
    /// size. Zero before the first collection.
    last_live: usize,
    /// Per-arena live bytes from the most recent collection's sweep (see
    /// [`GcArenaBytes`]); zero before the first collection.
    arena_bytes: GcArenaBytes,
    /// Intrinsics installed by the builtins workers; traced as roots.
    pub object_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    pub array_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    /// `%Array%`; the `ArraySpeciesCreate` default constructor.
    pub array_ctor: Option<Gc<Object>>,
    /// The 11 typed-array constructors (`Int8Array` … `BigUint64Array`),
    /// in `ETYPES` order; traced as roots. Used as the spec
    /// `TypedArraySpeciesCreate` default constructors.
    pub typed_array_ctors: [Option<Gc<Object>>; 11],
    /// Intrinsics installed by the builtins workers; traced as roots.
    /// `%Error.prototype%`; subtype prototypes chain to it.
    pub error_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    pub eval_error_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    pub range_error_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    pub reference_error_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    pub syntax_error_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    pub type_error_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    pub uri_error_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    pub suppressed_error_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    /// `%Function.prototype%` is itself callable (returns `undefined`).
    pub function_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    pub map_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    pub set_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    pub weakmap_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    pub weakset_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    pub symbol_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    pub boolean_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as roots.
    /// `%ArrayBuffer.prototype%`; backing stores link to it.
    pub arraybuffer_prototype: Option<Gc<Object>>,
    /// Intrinsics installed by the builtins workers; traced as a root.
    /// `%ArrayBuffer%` itself: the spec default constructor for
    /// `ArrayBuffer.prototype.slice`'s `SpeciesConstructor` step.
    pub arraybuffer_ctor: Option<Gc<Object>>,
    /// Native ([`ObjectKind::NativeFunction`]) implementations installed
    /// as constructors (ECMA-262 `IsConstructor`): `Array`, `Map`, the
    /// typed-array constructors, the DOM event constructors, and so on.
    /// Consulted by [`crate::builtin_reflect_proxy::is_constructor`].
    /// Holds no GC handles, so the tracer skips it.
    native_ctors: HashSet<NativeFn>,
    /// The global symbol registry object; traced as a root.
    pub symbol_registry: Option<Gc<Object>>,
    /// Well-known symbols (`Symbol.iterator`, …), installed on the
    /// `Symbol` constructor and rooted here so the abstract operations
    /// can consult them under real symbol keys ([`PropKey::Symbol`]).
    /// Indexed by [`WellKnown`]; `None` before builtins install (a realm
    /// without builtins treats every hook as absent).
    pub well_known: [Option<Gc<Symbol>>; WellKnown::COUNT],
    /// The mark-phase membership set, reused across collections: the
    /// buffer tracks the largest live set seen, so steady-state
    /// collections allocate nothing for marking.
    marked_set: MarkedSet,
    /// The tracer's worklist buffer, reused across collections: capacity
    /// tracks the largest object-graph frontier seen, so steady-state
    /// collections allocate nothing for the mark phase.
    worklist: Vec<Value>,
    /// Hidden classes (shapes): the interned layout table behind
    /// [`Object::shape`]. `shapes[0]` is always the empty root shape;
    /// every other entry is immutable after creation and shared by all
    /// objects built with the same key sequence. Entries strongly root
    /// their key strings (marked in [`Heap::collect`]) — but only while
    /// some live object references the shape: [`Heap::collect`] compacts
    /// the table, evicting shapes no live object uses and unpinning their
    /// key strings, so a long-lived page churning through layouts cannot
    /// grow this table (or the immortal string set) without bound. The
    /// same tradeoff the old per-object inline cache made per cached key,
    /// now paid once per distinct *live* layout instead of once per
    /// cached object.
    ///
    /// Pure accelerator: the property lists are the source of truth and
    /// every lookup re-verifies the slot still holds the indexed key, so
    /// a missed shape update can only cost a fallback to the linear scan,
    /// never a wrong answer. Plain Rust allocation, invisible to the GC;
    /// transitions never trigger a collection.
    shapes: Vec<Shape>,
    /// Private-member state: per-object brand membership plus the
    /// `(brand, name)` slot table that backs the private opcodes
    /// (`GetPrivate`/`SetPrivate`/`HasPrivate`/`AddPrivateBrand`,
    /// `InitPrivate`, `DefinePrivateAccessor`).
    ///
    /// Why a side table instead of properties: private state must never
    /// appear as string-keyed properties — `a["#x"]` must miss, and no
    /// user-visible key may exist to tamper with (the old
    /// `"\u{0}bowser:brands"` hidden property leaked through
    /// `getOwnPropertyNames` and a clobbered key aborted the VM). The
    /// table costs one pointer-hash lookup per private access and
    /// allocates only for objects that actually carry private state.
    ///
    /// `RefCell` because the engine is single-threaded. Keys are strong
    /// handles to live objects (never moving); slot names and values are
    /// traced as strong roots in [`Heap::collect`], and entries of freed
    /// objects are dropped after each sweep.
    private_state: RefCell<HashMap<Gc<Object>, PrivateData>>,
    /// Monotonic mutation epoch for embedder-driven reset fast paths
    /// (e.g. the test262 harness's per-test global scrub). Bumped by
    /// [`Heap::note_reset_write`] on property writes to registered
    /// objects; embedders snapshot it after a reset and skip expensive
    /// scans when it is unchanged. Plain data: never traced, never
    /// reset by collections.
    reset_epoch: u64,
    /// Shape-table epoch for per-call-site inline caches (see
    /// [`InlineCache`](crate::value::InlineCache)). Bumped by
    /// [`Heap::compact_shapes`] whenever a collection renumbers
    /// `ShapeId`s; a cache entry resolved under an older epoch may name
    /// a different layout and must be treated as a miss. Plain data:
    /// never traced.
    shape_epoch: u64,
    /// Objects whose *every* property mutation bumps
    /// [`Heap::reset_epoch`] — the test262 reset's scrub targets and
    /// their global bindings. Empty (and `reset_global` unset) unless an
    /// embedder opted in, so the write path stays a single branch.
    reset_tracked: HashSet<Gc<Object>>,
    /// Tracked objects written since the last
    /// [`Heap::take_dirtied_reset_objects`]. [`Heap::note_reset_write`]
    /// records each tracked-object write here, so the reset fast path
    /// can re-scrub exactly the dirtied targets instead of scanning every
    /// registered target. Plain data: never traced, never reset by
    /// collections.
    reset_dirtied: HashSet<Gc<Object>>,
    /// The global object for the reset fast path, if registered. A
    /// define/delete of a [`Heap::reset_baseline_keys`] entry bumps the
    /// epoch; a define of any other key is recorded in
    /// [`Heap::reset_added_keys`] instead (the key scan deletes exactly
    /// those, so the common per-test additions never bump).
    reset_global: Option<Gc<Object>>,
    /// Baseline own-keys of [`Heap::reset_global`].
    reset_baseline_keys: HashSet<String>,
    /// Non-baseline keys defined on the reset global since the last
    /// [`Heap::take_reset_added_keys`]; drained by the reset fast path.
    reset_added_keys: Vec<String>,
}

impl Default for Heap {
    fn default() -> Self {
        Heap {
            id: alloc_heap_id(),
            objects: Vec::new(),
            object_pages: Vec::new(),
            strings: Vec::new(),
            str_chunks: std::array::from_fn(|_| Vec::new()),
            str_hot: [u32::MAX; STR_NUM_CLASSES],
            symbols: Vec::new(),
            bigints: Vec::new(),
            interned: HashSet::new(),
            next_symbol_id: 0,
            scopes: Vec::new(),
            trace_roots: Vec::new(),
            post_sweep_hooks: Vec::new(),
            wrapper_cache_hook: None,
            temp_roots: Vec::new(),
            // Automatic collection is opt-in; see `set_gc_threshold`.
            gc_threshold: usize::MAX,
            last_live: 0,
            arena_bytes: GcArenaBytes::default(),
            object_prototype: None,
            array_prototype: None,
            array_ctor: None,
            typed_array_ctors: [None; 11],
            error_prototype: None,
            eval_error_prototype: None,
            range_error_prototype: None,
            reference_error_prototype: None,
            syntax_error_prototype: None,
            type_error_prototype: None,
            uri_error_prototype: None,
            suppressed_error_prototype: None,
            function_prototype: None,
            map_prototype: None,
            set_prototype: None,
            weakmap_prototype: None,
            weakset_prototype: None,
            symbol_prototype: None,
            boolean_prototype: None,
            arraybuffer_prototype: None,
            arraybuffer_ctor: None,
            native_ctors: HashSet::new(),
            symbol_registry: None,
            well_known: [None; WellKnown::COUNT],
            marked_set: MarkedSet::new(),
            worklist: Vec::new(),
            shapes: vec![Shape::root()],
            private_state: RefCell::new(HashMap::new()),
            reset_epoch: 0,
            shape_epoch: 0,
            reset_tracked: HashSet::new(),
            reset_dirtied: HashSet::new(),
            reset_global: None,
            reset_baseline_keys: HashSet::new(),
            reset_added_keys: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Dense array element storage
// ---------------------------------------------------------------------------

/// Longest array length dense storage supports (`2^32 - 1` is not a valid
/// array length).
const DENSE_MAX_LEN: u64 = 4294967294;

/// Largest hole run dense storage will allocate in a single extension.
/// Past this, the array materializes back to the ordinary property map
/// (the sparse fallback), so one huge-index write cannot allocate
/// gigabytes of holes.
const DENSE_MAX_GAP: u64 = 1024;

/// True when `o` is an array currently using dense element storage.
pub(crate) fn dense_is_active(o: &Object) -> bool {
    o.kind == ObjectKind::Array && o.dense().is_some()
}

/// True when growing dense storage from its current length to `new_len`
/// stays within the hole budget.
fn dense_gap_ok(o: &Object, new_len: u64) -> bool {
    let cur = o.dense().map_or(0, |d| d.len() as u64);
    new_len <= cur || new_len - cur <= DENSE_MAX_GAP
}

/// Canonical array index for `name` (`"0"` ..= `"4294967294"`), else `None`.
///
/// Same semantics as `builtin_object_array::array_index_key`; kept local so
/// this module does not depend on a file under active concurrent edit.
/// Takes [`Wtf8`](crate::wtf8::Wtf8): indices are ASCII, but keys arrive as
/// WTF-8 from JS strings.
fn dense_index_key(name: &Wtf8) -> Option<u32> {
    if name.is_empty() || name.len() > 10 {
        return None;
    }
    if name.len() > 1 && name.as_bytes()[0] == b'0' {
        return None;
    }
    if !name.as_bytes().iter().all(|b| b.is_ascii_digit()) {
        return None;
    }
    // ASCII digits are well-formed WTF-8, so `as_utf8` always succeeds here.
    let n: u64 = name.as_utf8()?.parse().ok()?;
    if n >= 4294967295 {
        return None;
    }
    Some(n as u32)
}

/// [`Str`] key for dense index `i`, packed inline with no allocation:
/// canonical array indices stringify to at most 10 digits, well under
/// [`STR_INLINE_MAX`](crate::value::STR_INLINE_MAX).
///
/// Copy of `builtin_object_array::dense_index_str`; kept local so this
/// module does not depend on a file under active concurrent edit (same
/// reason as [`dense_index_key`]).
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

/// A `length`-shaped [`Value`] for `len`, using the small-int fast path
/// like the rest of the engine.
fn dense_len_value(len: u64) -> Value {
    if len <= i32::MAX as u64 {
        Value::Int(len as i32)
    } else {
        Value::Float(len as f64)
    }
}

/// Coerce a value written to a dense array's `"length"` (the `ToLength`
/// prefix the slow path's callers apply before reaching `define_property`).
fn dense_coerce_len(v: Value) -> u64 {
    match v {
        Value::Int(i) if i >= 0 => i as u64,
        Value::Float(f) if f >= 0.0 && f.is_finite() => (f as u64).min(9007199254740991),
        _ => 0,
    }
}

/// A private member slot's payload: the spec's PrivateElement record.
///
/// Private fields and methods are `Data`; private getter/setter pairs are
/// `Accessor` (either side may be [`Value::Undefined`] when the class only
/// declared one of the pair).
#[derive(Debug, Clone, Copy)]
pub(crate) enum PrivateValue {
    Data(Value),
    Accessor { getter: Value, setter: Value },
}

/// One `(brand, name)` entry in an object's private-member table.
#[derive(Debug, Clone, Copy)]
struct PrivateSlot {
    brand: u32,
    name: Str,
    value: PrivateValue,
}

/// Per-object private state, kept in a VM-side table
/// ([`Heap::private_state`]) instead of the property map: user code can
/// never name these slots through `a["#x"]`, `Object.keys`, `for`-`in`,
/// or `JSON.stringify`, and no user-visible property key exists to
/// tamper with.
#[derive(Debug, Default)]
struct PrivateData {
    /// Brands installed by `AddPrivateBrand` (idempotent).
    brands: Vec<u32>,
    /// Private member slots. Small per object; linear scan.
    slots: Vec<PrivateSlot>,
}

// ---------------------------------------------------------------------------
// Array exotic `[[DefineOwnProperty]]` helpers (spec 10.4.2)
// ---------------------------------------------------------------------------

/// A (possibly partial) property descriptor for the array exotic, as
/// handed down by `Object.defineProperty` or a VM assignment. `None`
/// fields are absent from the descriptor; unlike [`Property`] they are
/// completed against the existing own property (spec 10.4.2.1 step 2)
/// rather than defaulted.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ArrayDefineDesc {
    pub(crate) value: Option<Value>,
    pub(crate) writable: Option<bool>,
    pub(crate) enumerable: Option<bool>,
    pub(crate) configurable: Option<bool>,
    pub(crate) get: Option<Value>,
    pub(crate) set: Option<Value>,
}

/// A descriptor completed against the existing own property, ready for
/// validation and the ordinary define.
#[derive(Debug, Clone, Copy)]
enum ArrayCompletedDesc {
    Data(Property),
    Accessor {
        get: Value,
        set: Value,
        enumerable: bool,
        configurable: bool,
    },
}

/// Read a `length`-ish value as `u64`: integers directly, floats by
/// truncation (callers only pass canonical lengths or the dense count).
fn as_u64_len(v: Value) -> u64 {
    match v {
        Value::Int(i) => i.max(0) as u64,
        Value::Float(f) => {
            if f.is_nan() || f <= 0.0 {
                0
            } else {
                f.trunc() as u64
            }
        }
        _ => 0,
    }
}

/// A `u64` array length as a [`Value`]: small integers stay `Int`,
/// larger lengths (up to 2^32 - 1) become exact `Float`s.
pub(crate) fn u64_to_value(u: u64) -> Value {
    if u <= i32::MAX as u64 {
        Value::Int(u as i32)
    } else {
        Value::Float(u as f64)
    }
}

/// `SameValue` for descriptor validation: like `===` but `NaN` equals
/// `NaN` and `+0` differs from `-0`.
fn array_same_value(a: Value, b: Value) -> bool {
    match (a, b) {
        (Value::Float(x), Value::Float(y)) => x.to_bits() == y.to_bits(),
        (Value::Int(0), Value::Float(y)) | (Value::Float(y), Value::Int(0)) => {
            y.to_bits() == 0.0f64.to_bits()
        }
        _ => crate::builtin_object_array::strict_equal(a, b),
    }
}

/// Spec `ToUint32`: modulo 2^32, via the float remainder. Non-finite
/// inputs reduce to `0` here; the exotic length path compares back
/// against the original `f64` and raises `RangeError` when they differ,
/// so `NaN`/`Infinity`/fractional/negative lengths are still rejected.
fn to_uint32_bits(n: f64) -> u32 {
    let m = n.rem_euclid(4294967296.0);
    // rem_euclid of a finite f64 by 2^32 is exactly representable below
    // 2^32, so the cast is exact.
    m as u32
}

/// A property key: a string or a symbol.
///
/// String keys flow through the shape/index fast path exactly as before;
/// symbol keys bypass the shape entirely and live in the per-object
/// symbol side tables ([`Object::symbol_properties`] /
/// [`Object::symbol_accessors`]), scanned linearly. Symbol identity is
/// the [`Gc`] address — symbols are unique per allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropKey<'a> {
    /// A string key: the historical property-key kind.
    Str(&'a Wtf8),
    /// A symbol key: bypasses interning, the shape, and the array exotic.
    Symbol(Gc<Symbol>),
}

impl<'a> From<&'a Wtf8> for PropKey<'a> {
    fn from(name: &'a Wtf8) -> Self {
        PropKey::Str(name)
    }
}

impl From<Gc<Symbol>> for PropKey<'_> {
    fn from(sym: Gc<Symbol>) -> Self {
        PropKey::Symbol(sym)
    }
}

/// A well-known symbol (`Symbol.iterator`, …), as an index into
/// [`Heap::well_known`]. The order matches
/// `builtin_collections::WELL_KNOWN_SYMBOLS`, which populates the table
/// at install time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WellKnown {
    AsyncDispose = 0,
    AsyncIterator = 1,
    Dispose = 2,
    HasInstance = 3,
    IsConcatSpreadable = 4,
    Iterator = 5,
    Match = 6,
    MatchAll = 7,
    Replace = 8,
    Search = 9,
    Species = 10,
    Split = 11,
    ToPrimitive = 12,
    ToStringTag = 13,
    Unscopables = 14,
}

impl WellKnown {
    /// Number of well-known symbols (size of [`Heap::well_known`]).
    pub const COUNT: usize = 15;

    /// The `Symbol.xxx` property name, in discriminant order (matches
    /// `builtin_collections::WELL_KNOWN_SYMBOLS`).
    pub fn name(self) -> &'static str {
        match self {
            WellKnown::AsyncDispose => "asyncDispose",
            WellKnown::AsyncIterator => "asyncIterator",
            WellKnown::Dispose => "dispose",
            WellKnown::HasInstance => "hasInstance",
            WellKnown::IsConcatSpreadable => "isConcatSpreadable",
            WellKnown::Iterator => "iterator",
            WellKnown::Match => "match",
            WellKnown::MatchAll => "matchAll",
            WellKnown::Replace => "replace",
            WellKnown::Search => "search",
            WellKnown::Species => "species",
            WellKnown::Split => "split",
            WellKnown::ToPrimitive => "toPrimitive",
            WellKnown::ToStringTag => "toStringTag",
            WellKnown::Unscopables => "unscopables",
        }
    }
}

/// An owned property key: the output of key resolution
/// ([`crate::vm::Interp::elem_key`], `to_property_key`). Every symbol —
///
/// well-known or user-created — passes through as [`ElemKey::Symbol`];
/// there is no alias translation and no `TypeError` for symbols.
///
/// Convert to the borrowed [`PropKey`] with
/// [`as_prop_key`](ElemKey::as_prop_key) at the use site; the borrow is
/// tied to this owned key, so the `Str` handle it carries stays alive
/// for the whole lookup.
#[derive(Clone, Copy, Debug)]
pub enum ElemKey {
    Str(Str),
    Symbol(Gc<Symbol>),
}

impl ElemKey {
    /// The borrowed [`PropKey`] for the `*_key` property APIs.
    pub fn as_prop_key(&self) -> PropKey<'_> {
        match self {
            ElemKey::Str(s) => PropKey::Str(s.as_wtf8()),
            ElemKey::Symbol(s) => PropKey::Symbol(*s),
        }
    }

    /// The key as a [`Value`], for rooting across user code that may GC
    /// (getters/setters, `toString`/`valueOf`, hook calls).
    pub fn as_value(&self) -> Value {
        match *self {
            ElemKey::Str(s) => Value::Str(s),
            ElemKey::Symbol(s) => Value::Symbol(s),
        }
    }
}

impl Heap {
    /// The well-known symbol `w`, installing it on first use.
    ///
    /// Install-time helper: any builtin `install` can name the symbols it
    /// needs without depending on another module's install order. Runtime
    /// consults must use the graceful [`well_known`](Heap::well_known)
    /// table access instead, so a hand-rolled heap without builtins treats
    /// every hook as absent.
    pub fn well_known_symbol(&mut self, w: WellKnown) -> Result<Gc<Symbol>, AllocError> {
        if let Some(sym) = self.well_known[w as usize] {
            return Ok(sym);
        }
        let desc = format!("Symbol.{}", w.name());
        let sym = self.alloc_symbol(Some(Wtf8::from_str(&desc)))?;
        self.well_known[w as usize] = Some(sym);
        Ok(sym)
    }

    /// Render a [`PropKey`] for error messages: string keys as-is (lossy,
    /// like [`Wtf8`]'s `Display`); symbols as `Symbol(description)`, or
    /// `Symbol()` when anonymous.
    pub fn format_key(&self, key: PropKey<'_>) -> String {
        match key {
            PropKey::Str(s) => s.to_utf8_lossy().into_owned(),
            PropKey::Symbol(s) => match &self.get(s).description {
                Some(d) => format!("Symbol({})", d.to_utf8_lossy()),
                None => "Symbol()".to_string(),
            },
        }
    }
}

impl Heap {
    /// Create an empty heap.
    pub fn new() -> Self {
        Self::default()
    }

    /// This heap's process-unique id. Recorded in every [`Gc`] the heap
    /// creates; see the "Heap identity" section in the module docs.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Panic unless `heap_id` is this heap's id. This is what turns
    /// cross-heap handle misuse from silent memory corruption into a loud
    /// failure at the point of misuse.
    #[track_caller]
    fn check_heap(&self, heap_id: u64) {
        assert!(
            heap_id == self.id,
            "cross-heap Gc use: handle belongs to heap #{heap_id}, used with heap #{}",
            self.id,
        );
    }

    /// Panic unless `g` was created by this heap.
    #[track_caller]
    fn check_gc<T: ?Sized>(&self, g: Gc<T>) {
        self.check_heap(g.heap_id);
    }

    /// Panic unless every handle inside `v` was created by this heap.
    #[track_caller]
    fn check_value(&self, v: Value) {
        match v {
            // Inline strings carry no heap handle.
            Value::Str(s) => {
                if let Some(g) = s.heap_handle() {
                    self.check_gc(g);
                }
            }
            Value::Object(g) => self.check_gc(g),
            Value::Symbol(g) => self.check_gc(g),
            Value::BigInt(g) => self.check_gc(g),
            Value::Undefined | Value::Null | Value::Bool(_) | Value::Int(_) | Value::Float(_) => {}
        }
    }

    /// Set the allocation count that triggers an automatic collection on
    /// the next allocation. `usize::MAX` disables automatic collection.
    ///
    /// Each collection advances an enabled threshold to at least twice the
    /// surviving live set, so the threshold tracks heap growth instead of
    /// re-firing on every allocation past the initial value.
    pub fn set_gc_threshold(&mut self, count: usize) {
        self.gc_threshold = count;
    }

    /// Record `f` as a constructible native (ECMA-262 `IsConstructor`).
    /// Called once per builtin constructor at install time; see
    /// [`crate::builtin_reflect_proxy::new_native_constructor`].
    pub(crate) fn mark_native_constructor(&mut self, f: NativeFn) {
        self.native_ctors.insert(f);
    }

    /// True when the native function implementation `f` was installed as
    /// a constructor.
    pub(crate) fn is_native_constructor(&self, f: NativeFn) -> bool {
        self.native_ctors.contains(&f)
    }

    /// Brand-membership test for the private-member side table (spec:
    /// `PrivateBrandCheck`): true when `AddPrivateBrand` recorded `brand`
    /// on `o`.
    pub(crate) fn private_has_brand(&self, o: Gc<Object>, brand: u32) -> bool {
        self.private_state
            .borrow()
            .get(&o)
            .is_some_and(|d| d.brands.contains(&brand))
    }

    /// Record `brand` on `o` (idempotent). Pure side-table update: no
    /// user-visible property is created, so there is nothing to tamper
    /// with and no failure mode.
    pub(crate) fn private_add_brand(&self, o: Gc<Object>, brand: u32) {
        let mut t = self.private_state.borrow_mut();
        let d = t.entry(o).or_default();
        if !d.brands.contains(&brand) {
            d.brands.push(brand);
        }
    }

    /// Define or replace `o`'s own `(brand, name)` slot (the private
    /// install path: field/method/accessor initialization). Never touches
    /// the property map.
    pub(crate) fn private_define(&self, o: Gc<Object>, brand: u32, name: Str, value: PrivateValue) {
        let mut t = self.private_state.borrow_mut();
        let d = t.entry(o).or_default();
        if let Some(s) = d
            .slots
            .iter_mut()
            .find(|s| s.brand == brand && s.name == name)
        {
            s.value = value;
        } else {
            d.slots.push(PrivateSlot { brand, name, value });
        }
    }

    /// Prototype-chain search for a `(brand, name)` slot, mirroring
    /// `find_prop_in_chain` (tortoise-and-hare cycle guard). Returns the
    /// owning object and the slot payload.
    pub(crate) fn private_find(
        &self,
        obj: Gc<Object>,
        brand: u32,
        name: Str,
    ) -> Option<(Gc<Object>, PrivateValue)> {
        let mut slow = obj;
        let mut advance_slow = false;
        let mut cur = obj;
        loop {
            let hit = {
                let t = self.private_state.borrow();
                t.get(&cur).and_then(|d| {
                    d.slots
                        .iter()
                        .find(|s| s.brand == brand && s.name == name)
                        .map(|s| s.value)
                })
            };
            if let Some(v) = hit {
                return Some((cur, v));
            }
            cur = match self.get(cur).prototype {
                Value::Object(p) => p,
                _ => return None,
            };
            // Move the tortoise every other step; if the hare laps it the
            // chain is cyclic: stop, the slot was not found.
            if advance_slow {
                slow = match self.get(slow).prototype {
                    Value::Object(p) => p,
                    // The tortoise reached the end: acyclic.
                    _ => return None,
                };
                if slow == cur {
                    return None;
                }
            }
            advance_slow = !advance_slow;
        }
    }

    /// Run a mark-sweep collection now. Unrooted allocations are freed;
    /// survivors keep their addresses.
    ///
    /// If automatic collection is enabled, the threshold is advanced to at
    /// least twice the surviving live set (see
    /// [`set_gc_threshold`][Self::set_gc_threshold]).
    pub fn collect(&mut self) {
        // Reuse the heap's mark-phase buffers across collections; their
        // capacities track the largest live set / object-graph frontier
        // seen. The tracer is local to this collection and is fully
        // drained before the buffers are handed back, so a re-entrant
        // collection can never observe a half-used worklist.
        let mut tracer = Tracer::with_marked(
            std::mem::take(&mut self.marked_set),
            std::mem::take(&mut self.worklist),
        );
        // Pre-size the marked set against the previous collection's live
        // set: an estimate of reachable addresses, so the mark phase
        // inserts at load ≤ 1/2 without rehashing mid-mark in the steady
        // state. Reserving the total allocation count here instead would
        // count dead objects and permanently inflate the reused buffer
        // on high-mortality heaps.
        tracer.marked.reserve(self.last_live);
        for &roots in &self.scopes {
            // SAFETY: `roots` was registered by a live `HandleScope` (which
            // unregisters on drop before the heap can be dropped), and the
            // engine is single-threaded, so no scope is created or dropped
            // while `&mut self` is held here. The pointer carries the root
            // list's `Box::into_raw` provenance, which stays valid for the
            // scope's whole life (see `HandleScope::new`).
            let values: &[Value] = unsafe { &*roots };
            for &v in values {
                tracer.mark_value(v);
            }
        }
        for &v in &self.temp_roots {
            tracer.mark_value(v);
        }
        for s in self.well_known.iter().flatten() {
            tracer.mark_symbol(*s);
        }
        for &root in &self.trace_roots {
            // SAFETY: `root` was registered by a live `Trace` owner (which
            // unregisters before being dropped or moved), and the engine is
            // single-threaded, so no registration changes while `&mut self`
            // is held here.
            unsafe {
                (*root).trace(&mut tracer);
            }
        }
        if let Some(g) = self.object_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.array_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.array_ctor {
            tracer.mark_object(g);
        }
        for g in self.typed_array_ctors.iter().flatten() {
            tracer.mark_object(*g);
        }
        if let Some(g) = self.error_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.eval_error_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.range_error_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.reference_error_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.syntax_error_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.type_error_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.uri_error_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.suppressed_error_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.function_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.map_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.set_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.weakmap_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.weakset_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.symbol_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.boolean_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.arraybuffer_prototype {
            tracer.mark_object(g);
        }
        if let Some(g) = self.arraybuffer_ctor {
            tracer.mark_object(g);
        }
        if let Some(g) = self.symbol_registry {
            tracer.mark_object(g);
        }
        // The reset fast path keeps a strong handle on its global in
        // `reset_global`; mark it so a collection can never free an object
        // this handle still references, which would leave a dangling
        // handle (today the failure mode is silent epoch bookkeeping
        // that stops firing).
        if let Some(g) = self.reset_global {
            tracer.mark_object(g);
        }
        // Interned strings are weak roots: they are deliberately NOT marked
        // here. A string that is otherwise unreachable is freed by the sweep
        // below and evicted from the table, so `intern` never resurrects a
        // dead handle. Strings still reachable (property keys, function
        // names, ordinary values) are marked through their owners as usual.
        //
        // Private slots hold strong handles (names and values): mark them,
        // or a collection could free a string or value the side table
        // still references, leaving a dangling handle.
        for data in self.private_state.borrow().values() {
            for slot in &data.slots {
                if let Some(g) = slot.name.heap_handle() {
                    tracer.mark_str(g);
                }
                match slot.value {
                    PrivateValue::Data(v) => tracer.mark_value(v),
                    PrivateValue::Accessor { getter, setter } => {
                        tracer.mark_value(getter);
                        tracer.mark_value(setter);
                    }
                }
            }
        }
        // Shape keys are strong roots, but only for shapes that survived
        // compaction (see below): a collection must not free a string a
        // live shape still references, while keys of evicted shapes are
        // freed when otherwise unreferenced. This narrows the old "pinned
        // for the heap's lifetime" tradeoff to "pinned while the layout
        // is in use". Inline keys live in the `Str` value: nothing to
        // mark. Strings are leaves, so inserting into the marked set
        // directly is equivalent to `tracer.mark_str` (no worklist
        // queueing needed).
        let (mut marked, mut worklist) = tracer.take_marked();
        // Compact the shape table now that the live set is known: shapes
        // no live object references are evicted, so the table (and the
        // key strings it roots) stays bounded on heaps that churn through
        // layouts. Must precede the shape-key marking below — evicted
        // shapes must not keep their key strings alive.
        self.compact_shapes(&marked);
        for shape in &self.shapes {
            for k in shape.index.keys() {
                if let Some(g) = k.heap_handle() {
                    marked.insert(g.as_ptr() as *const () as usize);
                }
            }
        }
        // Sweep objects through the arena reclaimer: dead slots are dropped
        // in place and pushed onto their page's freelist (fully-emptied
        // pages go back to the OS); survivors keep their addresses.
        // `pages` is bound first so the closure and the registry borrow
        // disjoint fields.
        let pages = &mut self.object_pages;
        let object_survivors = sweep(
            &mut self.objects,
            &marked,
            |ptr| reclaim_object_slot(pages, ptr),
            |_| {},
        );
        // Drop private-state entries of freed objects: without this the
        // table would pin garbage alive and hand out dangling keys.
        self.private_state
            .borrow_mut()
            .retain(|obj, _| marked.contains(obj.as_ptr() as usize));
        // Drop reset bookkeeping for freed objects: the tracked/dirtied
        // sets hold untraced raw `NonNull` handles (not roots), so without
        // this they would accumulate stale entries across collections.
        self.reset_tracked
            .retain(|obj| marked.contains(obj.as_ptr() as usize));
        self.reset_dirtied
            .retain(|obj| marked.contains(obj.as_ptr() as usize));
        // Sweep strings through the arena reclaimer: dead blocks go onto
        // their chunk's freelist (fully-emptied chunks go back to the OS,
        // oversize blocks are freed); survivors keep their addresses.
        // `chunks` is bound first so the closure and the registry borrow
        // disjoint fields.
        let chunks = &mut self.str_chunks;
        let mut strings_bytes = 0usize;
        sweep(
            &mut self.strings,
            &marked,
            |ptr| reclaim_str_block(chunks, ptr),
            |ptr| strings_bytes += str_block_bytes(ptr),
        );
        let symbol_survivors = sweep(&mut self.symbols, &marked, free_boxed, |_| {});
        let bigint_survivors = sweep(&mut self.bigints, &marked, free_boxed, |_| {});
        self.arena_bytes = GcArenaBytes {
            objects: object_survivors * std::mem::size_of::<Object>(),
            strings: strings_bytes,
            symbols: symbol_survivors * std::mem::size_of::<Symbol>(),
            bigints: bigint_survivors * std::mem::size_of::<BigInt>(),
        };
        // Evict interned entries whose strings did not survive. Without
        // this the table would hand out dangling handles on the next
        // `intern` of the same contents.
        self.interned
            .retain(|k| marked.contains(k.0.as_ptr() as *const () as usize));
        // Run the embedder post-sweep hooks at the same phase as the
        // retains above: every dead address is already reclaimed, and
        // `marked` still describes the survivors. (Borrow: `marked` is
        // a local, so iterating the hooks needs no `mem::take`.)
        for hook in self.post_sweep_hooks.iter_mut() {
            hook(&|addr| marked.contains(addr));
        }
        // The wrapper-cache eviction hook runs at the same phase; it is
        // a single slot (see `set_wrapper_cache_hook`), so at most one
        // context's hook is ever installed.
        if let Some(hook) = self.wrapper_cache_hook.as_mut() {
            hook(&|addr| marked.contains(addr));
        }
        // Advance an enabled threshold to track the live set. Without this,
        // a heap that survives a collection near the threshold would collect
        // on nearly every subsequent allocation.
        if self.gc_threshold != usize::MAX {
            let live = self.alloc_count();
            self.gc_threshold = self.gc_threshold.max(live.saturating_mul(2));
        }
        // Clear and hand the buffers back: next collection reuses them
        // without allocating. Stale marks must not survive — they would
        // wrongfully keep garbage alive; a stale worklist entry would
        // mark garbage as live.
        self.last_live = marked.len();
        marked.clear();
        self.marked_set = marked;
        worklist.clear();
        self.worklist = worklist;
    }

    fn push_scope(&mut self, roots: *mut Vec<Value>) {
        self.scopes.push(roots);
    }

    fn pop_scope(&mut self, roots: *mut Vec<Value>, heap_id: u64) {
        // The scope was pinned to this heap at creation; a mismatch means
        // the scope's heap pointer is stale or wrong. Fail loudly instead of
        // silently leaving a dangling root-list pointer registered.
        assert!(
            heap_id == self.id,
            "HandleScope dropped against the wrong heap (heap #{}, scope pinned to heap #{heap_id})",
            self.id,
        );
        if let Some(i) = self.scopes.iter().position(|&r| r == roots) {
            self.scopes.swap_remove(i);
        }
    }

    /// Register an embedder [`Trace`] root so [`Heap::collect`] keeps the
    /// values it reports alive. Used for value storage the heap cannot see
    /// itself: the interpreter's operand stack, call frames (environments,
    /// upvalue cells, arguments), suspended generator/async state, promise
    /// reactions, and timer records.
    ///
    /// # Safety
    ///
    /// * `root` must point to a valid `T` for as long as it stays
    ///   registered.
    /// * The referent must not move while registered (pin it, e.g. behind
    ///   a `Box`).
    /// * The engine never unregisters: the root lives as long as the heap.
    /// * The traced handles must belong to this heap (cross-heap handles
    ///   are not validated here; the embedder upholds this).
    /// * Single-threaded use only: no registration changes while a
    ///   collection is running (upheld: the engine is single-threaded).
    pub(crate) unsafe fn add_trace_root<T: Trace + 'static>(&mut self, root: *mut T) {
        self.trace_roots.push(root as *mut dyn Trace);
    }

    /// Register a [`PostSweepHook`]: it runs at the end of every
    /// [`Heap::collect`], after dead addresses are reclaimed and while
    /// the survivor set is still available. The hook must not allocate
    /// on this heap and must not panic.
    /// Install (or replace) the web-context wrapper-cache eviction
    /// hook. This is idempotent: reinstalling a context on an
    /// already-used heap swaps the single slot instead of pushing
    /// another entry, so repeated install/remove cycles cannot
    /// accumulate dead hooks (the previous hook's `Weak` roots keep it
    /// harmless but not free — one entry per install, run on every
    /// collection).
    pub(crate) fn set_wrapper_cache_hook(&mut self, hook: PostSweepHook) {
        self.wrapper_cache_hook = Some(hook);
    }

    fn maybe_collect(&mut self) {
        if self.alloc_count() >= self.gc_threshold {
            self.collect();
        }
    }

    /// Hold `v` as a temporary GC root until [`release_temp`][Self::release_temp]
    /// is called with the returned base. Use when a fresh allocation must
    /// survive further allocations before it is linked into a traced root
    /// (operand stack, frame, or object graph).
    ///
    /// # Panics
    ///
    /// Panics if any handle inside `v` was created by a different heap.
    pub fn protect(&mut self, v: Value) -> usize {
        self.check_value(v);
        let base = self.temp_roots.len();
        self.temp_roots.push(v);
        base
    }

    /// Release temporary roots pushed after `base` (see [`protect`][Self::protect]).
    pub fn release_temp(&mut self, base: usize) {
        self.temp_roots.truncate(base);
    }

    /// Current length of the temporary-root stack: the `base` to pass to
    /// [`release_temp`](Self::release_temp) to drop every root pushed
    /// after this point.
    pub fn temp_roots_len(&self) -> usize {
        self.temp_roots.len()
    }

    /// Run [`maybe_collect`][Self::maybe_collect] with `v` held as a
    /// temporary root. A fresh allocation is not yet reachable from any
    /// root, so the collection it triggers would otherwise free it before
    /// the caller receives the handle.
    fn collect_with_root(&mut self, v: Value) {
        let base = self.temp_roots.len();
        self.temp_roots.push(v);
        self.maybe_collect();
        self.temp_roots.truncate(base);
    }

    /// Allocate a managed copy of `s`. Always a fresh value: use
    /// [`Heap::intern`] when deduplication is wanted.
    ///
    /// Short strings (at most [`STR_INLINE_MAX`](crate::value::STR_INLINE_MAX)
    /// bytes) are packed inline in the returned [`Str`](crate::value::Str):
    /// no heap allocation, no collection, no GC tracing. Longer strings
    /// get a fresh heap allocation on every call — never the interned
    /// canonical handle.
    pub fn alloc_str(&mut self, s: &str) -> Result<Str, AllocError> {
        self.alloc_str_wtf8(Wtf8::from_str(s))
    }

    /// WTF-8 entry point for [`Heap::alloc_str`]: lone-surrogate content
    /// goes through here instead of the `&str` overload.
    pub fn alloc_str_wtf8(&mut self, s: &Wtf8) -> Result<Str, AllocError> {
        if let Some(inline) = Str::inline_wtf8(s) {
            return Ok(inline);
        }
        Ok(Str::Heap(self.alloc_str_heap(s)?))
    }

    /// Heap-allocate a managed copy of `s`, unconditionally: never inline,
    /// never interned. For strings that must be [`GcStr`] handles by
    /// construction — [`Heap::intern`]'s table entries and long
    /// [`FunctionData`](crate::value::FunctionData) names.
    fn alloc_str_heap(&mut self, s: &Wtf8) -> Result<Gc<GcStr>, AllocError> {
        // Coalesced block, not `Box::into_raw`: the handle carries the
        // chunk block's own provenance, which stays valid no matter how
        // often the chunk list or the registry `Vec` reallocates (see the
        // `Heap` struct docs).
        let (block, class) = self.alloc_str_block(s.len())?;
        // SAFETY: `alloc_str_block` hands out each block exactly once, and
        // the block has room for the header plus `s.len()` payload bytes.
        let ptr = unsafe { GcStr::from_buf(block, class, s) };
        self.strings.push(ptr);
        let g = Gc::new(ptr, self.id);
        self.collect_with_root(Value::Str(Str::Heap(g)));
        Ok(g)
    }

    /// Hand out one coalesced string block for `len` payload bytes:
    /// freelist first, then a chunk's bump cursor, else a fresh chunk.
    /// Returns the block base and its size class. Chunk growth is
    /// fallible and honors the `FAIL_ALLOC` test injection (see
    /// [`StrChunk::new`]); payloads above the top class take the malloc
    /// fallback.
    fn alloc_str_block(&mut self, len: usize) -> Result<(NonNull<u8>, u8), AllocError> {
        // `GcStr::from_buf` stores the payload length in a `u32`; reject
        // absurd sizes here, before any chunk bump cursor or malloc
        // fallback can hand out a block. Failing later (in `from_buf`'s
        // assert) would strand a phantom live slot in the chunk, since the
        // block is never registered in `self.strings`.
        if len > u32::MAX as usize {
            return Err(AllocError);
        }
        let class = str_class_for(len);
        if class == STR_CLASS_OVERSIZE {
            return Ok((alloc_str_oversize(len)?, STR_CLASS_OVERSIZE));
        }
        let ci = class as usize;
        // Hot chunk first: the chunk that last handed out a block usually
        // still has bump room or the hottest freelist, so the common case
        // skips the reverse scan below entirely. A stale index (the sweep
        // removed chunks, or a fresh chunk shifted the list) is harmless:
        // it either names a full chunk and falls through, or is out of
        // bounds and is ignored.
        let hot = self.str_hot[ci];
        if hot != u32::MAX
            && let Some(block) = self.str_chunks[ci]
                .get_mut(hot as usize)
                .and_then(StrChunk::alloc_block)
        {
            return Ok((block, class));
        }
        // Highest-address chunk first: usually the chunk with bump room or
        // the hottest freelist. Older chunks are only scanned when it is
        // full, so the hot path stays a freelist pop or a bump.
        let mut found: Option<(usize, NonNull<u8>)> = None;
        for (i, chunk) in self.str_chunks[ci].iter_mut().enumerate().rev() {
            if let Some(block) = chunk.alloc_block() {
                found = Some((i, block));
                break;
            }
        }
        if let Some((i, block)) = found {
            self.str_hot[ci] = i as u32;
            return Ok((block, class));
        }
        // Grow: fresh chunk, inserted in base-address order (the sweep's
        // freelist push binary-searches this list).
        let mut chunk = StrChunk::new(class)?;
        // `new` leaves the chunk empty, so the bump cursor must hit.
        let block = chunk.alloc_block().expect("fresh str chunk has room");
        let idx = self.str_chunks[ci]
            .partition_point(|c| (c.base.as_ptr() as usize) < (chunk.base.as_ptr() as usize));
        self.str_chunks[ci].insert(idx, chunk);
        self.str_hot[ci] = idx as u32;
        Ok((block, class))
    }

    /// Allocate a managed string from `s`'s bytes. Same freshness
    /// contract as [`Heap::alloc_str`]: always a fresh value, never
    /// interned. Short results are packed inline with no heap allocation
    /// at all; long results are copied into an arena string block (see
    /// [`Heap::alloc_wtf8_owned`]).
    pub fn alloc_str_owned(&mut self, s: String) -> Result<Str, AllocError> {
        // SAFETY: `String` is valid UTF-8, hence valid WTF-8.
        self.alloc_wtf8_owned(unsafe { Wtf8Buf::from_vec_unchecked(s.into_bytes()) })
    }

    /// WTF-8 counterpart of [`Heap::alloc_str_owned`]: short buffers are
    /// packed inline (the buffer is freed on drop); longer ones are
    /// copied into an arena string block (see [`alloc_str_block`]).
    /// Concatenation funnels its intermediate [`Wtf8Buf`] through here so
    /// lone surrogates survive.
    pub fn alloc_wtf8_owned(&mut self, buf: Wtf8Buf) -> Result<Str, AllocError> {
        if buf.len() <= crate::value::STR_INLINE_MAX {
            // Short result: pack the bytes inline; the buffer is freed on
            // drop. One copy of the bytes, no heap allocation.
            let inline = Str::inline_wtf8(buf.as_wtf8()).expect("length checked above");
            return Ok(inline);
        }
        // Arena-copy path: allocate the block, then copy the buffer's
        // bytes into it (the buffer is freed on drop).
        let (block, class) = self.alloc_str_block(buf.len())?;
        // SAFETY: as in `alloc_str_heap`.
        let ptr = unsafe { GcStr::from_buf(block, class, buf.as_wtf8()) };
        self.strings.push(ptr);
        let g = Gc::new(ptr, self.id);
        self.collect_with_root(Value::Str(Str::Heap(g)));
        Ok(Str::Heap(g))
    }

    /// Concatenate two strings, packing the result inline when it fits.
    ///
    /// Heap-backed results are written straight into the arena block:
    /// one copy of the operands, no intermediate buffer (the old path
    /// copied twice: once into a [`Wtf8Buf`], once into the block).
    ///
    /// `a` and `b` must be live for the call: heap-backed operands are
    /// borrowed for the copy. (Callers that allocate between deriving the
    /// operands and concatenating must root them first, per the usual
    /// discipline.)
    pub fn concat_str(&mut self, a: Str, b: Str) -> Result<Str, AllocError> {
        let (sa, sb) = (a.as_wtf8(), b.as_wtf8());
        let total = sa.len() + sb.len();
        // Maximal pairing can shrink the byte count by at most 2 at the
        // boundary (3+3 -> 4), so only this range can possibly go inline.
        if total <= crate::value::STR_INLINE_MAX + 2 {
            let merged = crate::wtf8::concat_bytes(sa, sb);
            if merged.len() <= crate::value::STR_INLINE_MAX {
                let mut bytes = [0u8; crate::value::STR_INLINE_MAX];
                bytes[..merged.len()].copy_from_slice(&merged);
                return Ok(Str::Inline {
                    len: merged.len() as u8,
                    bytes,
                });
            }
            // SAFETY: `concat_bytes` emits valid WTF-8 by construction.
            return self.alloc_wtf8_owned(unsafe { Wtf8Buf::from_vec_unchecked(merged) });
        }
        // Heap path: maximal pairing can only shrink the boundary by 2,
        // so the final length is known before allocating.
        let fold = crate::wtf8::trailing_lead_unit(sa.as_bytes()).is_some()
            && crate::wtf8::leading_trail_unit(sb.as_bytes()).is_some();
        let merged_len = total - if fold { 2 } else { 0 };
        let (block, class) = self.alloc_str_block(merged_len)?;
        // SAFETY: `block` is a fresh arena block with room for the header
        // plus `merged_len` payload bytes and no other owner;
        // `concat_into` writes exactly `merged_len` valid WTF-8 bytes at
        // the payload offset, so every write stays in-bounds, and the
        // header write lands in the block's own prefix.
        let written =
            unsafe { crate::wtf8::concat_into(sa, sb, block.as_ptr().add(GcStr::HEADER_LEN)) };
        debug_assert_eq!(written, merged_len);
        let header = block.as_ptr() as *mut GcStr;
        unsafe {
            std::ptr::addr_of_mut!((*header).len).write(written as u32);
            std::ptr::addr_of_mut!((*header).class).write(class);
        }
        // SAFETY: `block` is non-null by construction.
        let ptr = unsafe { NonNull::new_unchecked(header) };
        self.strings.push(ptr);
        let g = Gc::new(ptr, self.id);
        self.collect_with_root(Value::Str(Str::Heap(g)));
        Ok(Str::Heap(g))
    }

    /// Return the canonical [`Str`] for `s`, packing short strings inline.
    ///
    /// The SSO-aware interning entry point: strings of at most
    /// [`STR_INLINE_MAX`](crate::value::STR_INLINE_MAX) bytes are returned
    /// as [`Str::Inline`] with **no heap allocation and no intern-table
    /// entry** — inline strings are values compared by content, so there
    /// is nothing to deduplicate and nothing for the collector to track.
    /// Longer strings are weakly interned on the GC heap (see
    /// [`Heap::intern_heap`]) and come back as [`Str::Heap`] holding the
    /// canonical handle.
    ///
    /// Interned heap strings are weak roots: if the returned [`Str`] is
    /// the only reference and a collection frees the string, the table
    /// entry is evicted and a later `intern` of the same contents
    /// allocates fresh. While the string is live elsewhere, repeated
    /// `intern` calls return content-equal [`Str`]s sharing the one
    /// canonical handle.
    ///
    /// # Caller contract
    ///
    /// The table does NOT keep the returned handle alive. A caller that
    /// holds a heap-backed result across any further allocation
    /// (`intern`, `alloc_*`, `define_property`, …) must root it first —
    /// via [`HandleScope::add`], [`Heap::protect`], or by linking it into
    /// an already-rooted object — before that allocation runs. Holding an
    /// interned handle in untraced storage (a local `Vec`, a struct
    /// field, …) across a later allocation is use-after-free: the
    /// collection that allocation triggers frees the string and evicts
    /// its table entry, leaving a dangling handle. (Inline results need
    /// no rooting: they live entirely in the [`Str`] value.)
    pub fn intern(&mut self, s: &str) -> Result<Str, AllocError> {
        self.intern_wtf8(Wtf8::from_str(s))
    }

    /// WTF-8 entry point for [`Heap::intern`]: lone-surrogate content goes
    /// through here instead of the `&str` overload.
    pub fn intern_wtf8(&mut self, s: &Wtf8) -> Result<Str, AllocError> {
        if let Some(inline) = Str::inline_wtf8(s) {
            return Ok(inline);
        }
        Ok(Str::heap(self.intern_heap(s)?))
    }

    /// Alias of [`Heap::intern`]: the SSO-aware interning entry point.
    /// Prefer [`Heap::intern`]; kept for call sites that think in terms of
    /// [`Str`] values.
    pub fn intern_str(&mut self, s: &str) -> Result<Str, AllocError> {
        self.intern_wtf8(Wtf8::from_str(s))
    }

    /// Weakly intern `s` on the GC heap, unconditionally heap-allocating:
    /// the raw-handle half of [`Heap::intern`]. The [`InternedStr`] table
    /// maps contents to the canonical [`GcStr`] handle; entries are
    /// evicted when their string is otherwise unreachable (see
    /// [`Heap::intern`]'s weak-root contract). Crate-visible for call
    /// sites that need a [`GcStr`] handle by construction.
    pub(crate) fn intern_heap(&mut self, s: &Wtf8) -> Result<Gc<GcStr>, AllocError> {
        if let Some(k) = self.interned.get(s) {
            return Ok(k.0);
        }
        let g = self.alloc_str_heap(s)?;
        self.interned.insert(InternedStr(g));
        Ok(g)
    }

    /// Allocate an [`Object`] on the heap: bump-allocate (or freelist-reuse)
    /// a slot from the object pages and construct the referent in place.
    pub fn alloc_object(&mut self, obj: Object) -> Result<Gc<Object>, AllocError> {
        // `ptr::write` into a page slot does not retag: the handle carries
        // the page block's own provenance, which stays valid no matter how
        // often the page list or the registry `Vec` reallocates.
        // (Deriving the pointer from a `&`-reborrow of the slot instead
        // would be undefined behavior under Stacked Borrows — see the
        // `Heap` struct docs.)
        let slot = self.alloc_object_slot()?;
        // SAFETY: `slot` is uniquely owned — fresh from the bump cursor or
        // the freelist, whose referent was dropped when the slot was
        // reclaimed — so constructing the `Object` in place is sound.
        unsafe { std::ptr::write(slot.as_ptr(), obj) };
        self.objects.push(slot);
        // The fresh object is not yet reachable from any root; hold it
        // across the collection below (the tracer then also covers the
        // values it already carries).
        let g = Gc::new(slot, self.id);
        self.collect_with_root(Value::Object(g));
        Ok(g)
    }

    /// Hand out one object slot: freelist first, then a page's bump cursor,
    /// else a fresh page. Page growth is fallible and honors the
    /// `FAIL_ALLOC` test injection (see [`ObjectPage::new`]).
    fn alloc_object_slot(&mut self) -> Result<NonNull<Object>, AllocError> {
        // Highest-address page first: usually the page with bump room or
        // the hottest freelist. Older pages are only scanned when it is
        // full, so the hot path stays a freelist pop or a bump.
        if let Some(slot) = self
            .object_pages
            .iter_mut()
            .rev()
            .find_map(ObjectPage::alloc_slot)
        {
            return Ok(slot);
        }
        // Grow: fresh page, inserted in base-address order (the sweep's
        // freelist push binary-searches this list).
        let mut page = ObjectPage::new()?;
        // `new` leaves the page empty, so the bump cursor must hit.
        let slot = page
            .alloc_slot()
            .expect("fresh object page has a free slot");
        let idx = self
            .object_pages
            .partition_point(|p| (p.base.as_ptr() as usize) < (page.base.as_ptr() as usize));
        self.object_pages.insert(idx, page);
        Ok(slot)
    }

    /// Repair `[[Prototype]]` on native functions allocated before the
    /// Function intrinsics existed (spec 10.2.4: every function object's
    /// prototype is `%Function.prototype%`).
    ///
    /// Builtin installation allocates native functions while
    /// [`Heap::function_prototype`] is still `None`; those are created with
    /// a null prototype and repaired here once installation completes. The
    /// intrinsic function prototype object itself is skipped (its prototype
    /// is `%Object.prototype%`, not null), as is any native function that
    /// already carries a prototype.
    ///
    /// Performs no allocation, so it is safe to call between allocations.
    /// Callers must not hold a shared `&` to any object across this call
    /// (see module docs).
    pub fn fixup_native_function_prototypes(&mut self) {
        let Some(fp) = self.function_prototype else {
            return;
        };
        for &ptr in &self.objects {
            // SAFETY: `ptr` came from an object page slot in
            // `alloc_object`, which never moves or frees a live
            // allocation, and `&mut self` keeps the heap (hence every
            // allocation) live for the duration of this call. The `&mut`
            // is bounded by `&mut self`, so no second `&mut` to the same
            // object can coexist; the caller contract above rules out
            // shared borrows.
            let obj = unsafe { &mut *ptr.as_ptr() };
            if matches!(obj.kind, ObjectKind::NativeFunction(_))
                && matches!(obj.prototype, Value::Null)
            {
                obj.prototype = Value::Object(fp);
            }
        }
    }

    /// Allocate a fresh, unique [`Symbol`].
    pub fn alloc_symbol(&mut self, description: Option<&Wtf8>) -> Result<Gc<Symbol>, AllocError> {
        let id = self.next_symbol_id;
        self.next_symbol_id = self.next_symbol_id.wrapping_add(1);
        // `Box::into_raw`, not a `&`-derived pointer: see `alloc_object`.
        // SAFETY: `Box::into_raw` never returns null for an owned box.
        let ptr: NonNull<Symbol> = NonNull::new(Box::into_raw(try_box(Symbol {
            description: description.map(|s| s.to_owned()),
            id,
        })?))
        .unwrap();
        self.symbols.push(ptr);
        self.collect_with_root(Value::Symbol(Gc::new(ptr, self.id)));
        Ok(Gc::new(ptr, self.id))
    }

    /// Move a [`BigInt`]'s parts into the heap.
    pub fn alloc_bigint(
        &mut self,
        negative: bool,
        digits: Vec<u32>,
    ) -> Result<Gc<BigInt>, AllocError> {
        // `Box::into_raw`, not a `&`-derived pointer: see `alloc_object`.
        // SAFETY: `Box::into_raw` never returns null for an owned box.
        let ptr: NonNull<BigInt> =
            NonNull::new(Box::into_raw(try_box(BigInt { negative, digits })?)).unwrap();
        self.bigints.push(ptr);
        self.collect_with_root(Value::BigInt(Gc::new(ptr, self.id)));
        Ok(Gc::new(ptr, self.id))
    }

    /// Read back any heap allocation through its handle. The handle must be
    /// live: rooted across any collection since its creation, and created
    /// by this heap.
    ///
    /// # Panics
    ///
    /// Panics if `g` was created by a different heap.
    pub fn get<T: ?Sized>(&self, g: Gc<T>) -> &T {
        self.check_gc(g);
        // SAFETY: same contract as `Gc::deref` — the caller guarantees the
        // allocation is live, and the collector never moves allocations.
        // `&self` proves the heap outlives the returned reference.
        // Membership was validated above.
        unsafe { &*g.as_ptr() }
    }

    /// Number of live allocations (diagnostic only).
    pub fn alloc_count(&self) -> usize {
        self.objects.len() + self.strings.len() + self.symbols.len() + self.bigints.len()
    }

    /// Total live bytes across the GC arenas, from the most recent
    /// [`Heap::collect`] sweep (object slots, string blocks, symbol and
    /// bigint boxes; see [`GcArenaBytes`] for what is excluded).
    /// Diagnostic only; zero before the first collection.
    pub fn live_bytes(&self) -> usize {
        self.arena_bytes.total()
    }

    /// Per-arena live bytes from the most recent [`Heap::collect`] sweep.
    /// Diagnostic only; zeroed before the first collection.
    pub fn arena_bytes(&self) -> GcArenaBytes {
        self.arena_bytes
    }

    /// Set the invoke-interception bitmask on an object (see
    /// [`Object::invoke_flags`][crate::value::Object]). Used once, at
    /// creation time, by the builtins that install interception slots;
    /// ordinary property writes never touch the bitmask.
    pub(crate) fn set_invoke_flags(&mut self, obj: Gc<Object>, flags: u8) {
        self.get_object_mut(obj).invoke_flags = flags;
    }

    /// Mark an `ArrayBuffer`-tagged object as a `SharedArrayBuffer` (see
    /// [`ObjectCold::ab_shared`][crate::value::ObjectCold]). Used once, at
    /// creation time, by the `SharedArrayBuffer` constructor; the flag is
    /// never cleared. A plain field write needs no GC barrier (no handles
    /// involved).
    pub(crate) fn mark_shared_array_buffer(&mut self, obj: Gc<Object>) {
        debug_assert!(
            matches!(self.get(obj).kind, crate::value::ObjectKind::ArrayBuffer),
            "mark_shared_array_buffer on non-ArrayBuffer"
        );
        self.get_object_mut(obj).set_ab_shared(true);
    }

    /// Mutable access to an object's storage. The borrow is tied to
    /// `&mut self`; callers must not use a shared `&` derived from `g`
    /// across this call (see module docs).
    ///
    /// # Panics
    ///
    /// Panics if `g` was created by a different heap.
    fn get_object_mut(&mut self, g: Gc<Object>) -> &mut Object {
        self.check_gc(g);
        // SAFETY: `g`'s pointer came from an object page slot in
        // `alloc_object`, so it carries the page block's own provenance,
        // which is never invalidated: the allocation is never moved, and a
        // dead slot's referent is dropped in place before the slot is ever
        // handed out again — which cannot have happened to this slot,
        // since the caller guarantees the handle is live (membership just
        // validated) and the `&mut self` borrow proves the heap, hence the
        // allocation, is live. The returned `&mut` is bounded by the
        // `&mut self` borrow, so no second `&mut` to this object can
        // coexist with it. Callers must still not *use* a shared `&`
        // derived from `g` across this call (see module docs): the reborrow
        // pops any live shared borrows above it in the stack.
        unsafe { &mut *g.ptr.as_ptr() }
    }

    /// Byte storage of an `ArrayBuffer` object, if it carries any.
    ///
    /// # Panics
    ///
    /// Panics if `g` was created by a different heap.
    pub fn object_bytes(&self, g: Gc<Object>) -> Option<&[u8]> {
        self.get(g).bytes()
    }

    /// Mutable byte storage of an `ArrayBuffer` object, if it carries any.
    /// The borrow is tied to `&mut self`; callers must not use a shared
    /// `&` derived from `g` across this call (see module docs).
    ///
    /// Byte writes are heap writes: noted for the reset epoch (free when
    /// nothing is tracked).
    ///
    /// # Panics
    ///
    /// Panics if `g` was created by a different heap.
    pub fn object_bytes_mut(&mut self, g: Gc<Object>) -> Option<&mut [u8]> {
        self.note_reset_write(g, None);
        self.get_object_mut(g).bytes_mut()
    }

    /// Detach an `ArrayBuffer`: drop its backing store. After this the
    /// buffer is in the detached state, observable as
    /// [`Heap::object_bytes`] returning `None` on an
    /// [`ObjectKind::ArrayBuffer`](crate::value::ObjectKind::ArrayBuffer)-tagged
    /// object (a zero-length buffer still carries `Some([])`, so the two
    /// are unambiguous).
    ///
    /// Byte writes are heap writes: noted for the reset epoch (free when
    /// nothing is tracked).
    ///
    /// # Panics
    ///
    /// Panics if `g` was created by a different heap.
    pub fn detach_array_buffer(&mut self, g: Gc<Object>) {
        self.note_reset_write(g, None);
        let o = self.get_object_mut(g);
        debug_assert!(
            matches!(o.kind, crate::value::ObjectKind::ArrayBuffer),
            "detach_array_buffer on non-ArrayBuffer"
        );
        o.set_bytes(None);
    }

    /// Replace an `ArrayBuffer`'s backing store wholesale (the
    /// `ArrayBuffer.prototype.resize` path: the caller allocates the new
    /// store fallibly and copies the preserved prefix first).
    ///
    /// Byte writes are heap writes: noted for the reset epoch (free when
    /// nothing is tracked).
    ///
    /// # Panics
    ///
    /// Panics if `g` was created by a different heap.
    pub fn set_array_buffer_bytes(&mut self, g: Gc<Object>, bytes: Box<[u8]>) {
        self.note_reset_write(g, None);
        let o = self.get_object_mut(g);
        debug_assert!(
            matches!(o.kind, crate::value::ObjectKind::ArrayBuffer),
            "set_array_buffer_bytes on non-ArrayBuffer"
        );
        o.set_bytes(Some(bytes));
    }

    /// Copy `len` bytes from `src[src_off..]` to `dst[dst_off..]`, clamped
    /// to both allocations. A no-op when either object has no byte
    /// storage. Overlap-safe (`ptr::copy` semantics).
    ///
    /// The caller must keep both handles rooted across the call; this
    /// performs no allocation and therefore no collection.
    ///
    /// # Panics
    ///
    /// Panics if either handle was created by a different heap, or if
    /// `src` and `dst` are the same object (all callers copy into a
    /// freshly allocated buffer).
    pub fn copy_object_bytes(
        &mut self,
        src: Gc<Object>,
        dst: Gc<Object>,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) {
        self.check_gc(src);
        self.check_gc(dst);
        assert!(
            !std::ptr::addr_eq(src.as_ptr(), dst.as_ptr()),
            "copy_object_bytes: src and dst must be distinct objects"
        );
        // Extract plain pointers first: `&*src` (shared) and `&mut *dst`
        // cannot be live at once, and the two objects are asserted
        // distinct, so no mutable aliasing is possible.
        // SAFETY: both pointers name live slots in this heap's arena page
        // blocks (`ObjectPage` bump-allocated pages, freed only by the
        // sweep), which never move allocations; the caller keeps
        // both handles live, and no collection runs in between (no
        // allocation happens here), so neither can be freed.
        let (s_ptr, s_len): (*const u8, usize) = unsafe {
            let o = &*src.as_ptr();
            match o.bytes() {
                Some(b) => (b.as_ptr(), b.len()),
                None => (std::ptr::null(), 0),
            }
        };
        let (d_ptr, d_len): (*mut u8, usize) = unsafe {
            let o = &mut *(dst.as_ptr() as *mut Object);
            match o.bytes_mut() {
                Some(b) => (b.as_mut_ptr(), b.len()),
                None => (std::ptr::null_mut(), 0),
            }
        };
        if s_ptr.is_null() || d_ptr.is_null() || len == 0 {
            return;
        }
        let avail_src = s_len.saturating_sub(src_off);
        let avail_dst = d_len.saturating_sub(dst_off);
        let n = len.min(avail_src).min(avail_dst);
        if n == 0 {
            return;
        }
        // SAFETY: `n` is clamped to both allocations; `ptr::copy` is
        // overlap-safe.
        unsafe {
            std::ptr::copy(s_ptr.add(src_off), d_ptr.add(dst_off), n);
        }
    }

    // ------------------------------------------------------------------
    // Hidden classes (shapes)
    // ------------------------------------------------------------------
    //
    // Every object carries a `ShapeId` describing the sequence of keys in
    // its two property lists. Shapes are interned: objects built by the
    // same definition sequence share one `Shape`, so a lookup is one hash
    // probe on the shape's index instead of a linear scan of the lists.
    //
    // Maintenance contract: `obj.shape` always mirrors the tracked prefix
    // of `obj`'s lists (up to `Shape::MAX_ENTRIES` keys per list).
    // - pure append (`define_property`/`define_accessor` adding a new key):
    //   `shape_add` transitions to the child shape (O(1): the cloned map
    //   is bounded by the cap);
    // - anything that shifts indices (removal, data↔accessor conversion):
    //   `shape_rebuild` replays the tracked prefix from the root
    //   (bounded, O(1));
    // - overwrite of an existing key: attributes may change but the
    //   layout does not, so the shape is untouched (shapes do not track
    //   attributes).
    // Past the cap the object is in dictionary mode: the shape stops
    // evolving and lookups fall back to the linear scan — never wrong,
    // only slow (the old "never wrong, only slow" philosophy).
    //
    // All shape operations are plain Rust allocation, invisible to the
    // GC: they never trigger a collection, so no rooting is needed across
    // them. A collection *can* still run between two shape operations
    // (e.g. `intern_str` inside `define_property` collects when the
    // threshold is armed): never cache a `ShapeId` across such a call —
    // [`Heap::collect`] compacts the table ([`Heap::compact_shapes`]) and
    // renumbers shapes, so a saved id may be stale afterwards. Re-read
    // the id from the object instead. The key `Str`s stored in shapes are
    // the same handles already held by the object's lists; while a shape
    // is referenced by a live object, [`Heap::collect`] additionally
    // roots them (shapes outlive any single *use*, not the heap).

    /// Transition `shape` by appending `(key, list)`: return the child
    /// shape, interning it when this is the first time this layout is
    /// seen. The new entry's index is the parent's entry count for that
    /// list, so the transition key is just `(key, list)`. Past
    /// [`Shape::MAX_ENTRIES`] tracked keys per list the shape stops
    /// evolving and the parent is returned (dictionary mode).
    pub(crate) fn shape_add(&mut self, shape: ShapeId, key: Str, list: ShapeList) -> ShapeId {
        let parent = &self.shapes[shape.0 as usize];
        let at_cap = match list {
            0 => parent.data_len >= Shape::MAX_ENTRIES,
            _ => parent.acc_len >= Shape::MAX_ENTRIES,
        };
        if at_cap {
            return shape;
        }
        if let Some(child) = parent.transition(key, list) {
            return child;
        }
        let index = match list {
            0 => parent.data_len,
            _ => parent.acc_len,
        };
        let mut child = Shape {
            index: parent.index.clone(),
            transitions: ShapeTransitions::default(),
            parent: shape,
            data_len: parent.data_len + (list == 0) as u32,
            acc_len: parent.acc_len + (list != 0) as u32,
        };
        child.index.insert(key, (list, index));
        let child_id = ShapeId(self.shapes.len() as u32);
        self.shapes.push(child);
        self.shapes[shape.0 as usize]
            .transitions
            .insert(key, list, child_id);
        child_id
    }

    /// Re-derive `obj`'s shape from its current lists (after a removal or
    /// data↔accessor conversion shifted indices, or after bulk-allocating
    /// an object with pre-filled lists). Replays the lists through the
    /// transition table, so the result is the canonical interned shape
    /// for the layout.
    pub(crate) fn shape_rebuild(&mut self, obj: Gc<Object>) {
        let mut shape = ShapeId::ROOT;
        // Borrow the lists once; `shape_add` only touches `self.shapes`.
        // Only the tracked prefix is replayed: past `MAX_ENTRIES` keys per
        // list the object is in dictionary mode and the shape stops
        // evolving (`shape_add` enforces the cap itself; the `take` just
        // skips the useless calls).
        let cap = Shape::MAX_ENTRIES as usize;
        let o = self.get(obj);
        let data_keys: Vec<Str> = o.properties.iter().map(|(k, _)| *k).take(cap).collect();
        let acc_keys: Vec<Str> = o.accessors.iter().map(|(k, _)| *k).take(cap).collect();
        for key in data_keys {
            shape = self.shape_add(shape, key, 0);
        }
        for key in acc_keys {
            shape = self.shape_add(shape, key, 1);
        }
        self.get_object_mut(obj).shape = shape;
    }

    /// Resolve `name` to `(list, index)` via the object's shape. Returns
    /// `None` when the shape has no entry (key absent, or a missed shape
    /// update — callers fall back to the linear scan, never a wrong
    /// answer).
    fn shape_lookup(&self, obj: Gc<Object>, name: &Wtf8) -> Option<(ShapeList, u32)> {
        let shape_id = self.get(obj).shape;
        self.shapes[shape_id.0 as usize].lookup(name)
    }

    /// Compact the shape table: evict every shape no live object
    /// references, keeping the table dense so it cannot grow without
    /// bound on a long-lived heap that churns through layouts.
    ///
    /// Runs at the end of the mark phase in [`Heap::collect`]: `marked`
    /// holds every live allocation's address. Shapes are plain (non-GC)
    /// data named from live objects by [`ShapeId`], so liveness is
    /// computed by scanning the live objects' `shape` fields — plus the
    /// derivation ancestors of every live shape (see [`Shape::parent`]),
    /// which keeps the transition cache consistent so re-deriving a live
    /// layout reuses its shape instead of interning a duplicate.
    /// Survivors are renumbered densely in stable order
    /// (`ShapeId::ROOT` keeps id 0); every live object's `shape` field
    /// and every surviving transition-table child id are rewritten
    /// through the remap.
    /// Transition entries pointing at evicted shapes are dropped — the
    /// table is a pure cache, so a later [`Heap::shape_add`] simply
    /// re-interns the layout on demand.
    ///
    /// Must run before shape key strings are marked: evicted shapes drop
    /// their strong rooting, so their key strings are freed by the sweep
    /// when nothing else references them.
    ///
    /// # Safety
    ///
    /// Dereferences the allocations in `self.objects` that `marked` says
    /// are live — sound for the same reason as [`sweep`]: survivors keep
    /// their addresses and are never reborrowed through a box while
    /// marked.
    fn compact_shapes(&mut self, marked: &MarkedSet) {
        let n = self.shapes.len();
        if n <= 1 {
            // Only the root: nothing to evict.
            return;
        }
        // Liveness: the root is pinned (fresh objects and `shape_rebuild`
        // derive every layout starting here); every other shape must be
        // named by a live object's `shape` field.
        let mut live = vec![false; n];
        live[0] = true;
        for &ptr in &self.objects {
            if marked.contains(ptr.as_ptr() as usize) {
                // SAFETY: `marked` says this object survived; objects are
                // non-moving, so the pointer is still valid.
                let id = unsafe { (*ptr.as_ptr()).shape }.0 as usize;
                debug_assert!(id < n, "object holds out-of-range ShapeId");
                if id < n {
                    live[id] = true;
                }
            }
        }
        // Ancestors of live shapes stay live: they are the transition
        // path a re-derivation of the same layout walks, so evicting one
        // would make `shape_add` miss the cache and intern a duplicate
        // shape for an identical layout. Depth is bounded by
        // `2 * Shape::MAX_ENTRIES`, so the walk is cheap.
        let mut stack: Vec<usize> = (0..n).filter(|&i| live[i]).collect();
        while let Some(i) = stack.pop() {
            let p = self.shapes[i].parent.0 as usize;
            debug_assert!(p < n, "shape holds out-of-range parent ShapeId");
            if p < n && !live[p] {
                live[p] = true;
                stack.push(p);
            }
        }
        // Dense renumbering in stable order: the root keeps id 0.
        let mut remap = vec![u32::MAX; n];
        let mut kept = Vec::with_capacity(n);
        for (i, shape) in self.shapes.drain(..).enumerate() {
            if live[i] {
                remap[i] = kept.len() as u32;
                kept.push(shape);
            }
        }
        if kept.len() == n {
            // Nothing died: restore the table untouched.
            self.shapes = kept;
            return;
        }
        // Rewrite surviving transition children through the remap; drop
        // entries whose child was evicted (pure cache: re-interned on
        // next use). The parent link is rewritten too: ancestors of live
        // shapes are always live, so the remap entry is valid.
        for shape in &mut kept {
            shape.parent = ShapeId(remap[shape.parent.0 as usize]);
            shape.transitions.retain(|_, child| {
                let new = remap[child.0 as usize];
                if new == u32::MAX {
                    false
                } else {
                    *child = ShapeId(new);
                    true
                }
            });
        }
        // Rewrite every live object's shape field through the remap.
        for &ptr in &self.objects {
            if marked.contains(ptr.as_ptr() as usize) {
                // SAFETY: as above.
                let obj = unsafe { &mut *ptr.as_ptr() };
                let new = remap.get(obj.shape.0 as usize).copied().unwrap_or(u32::MAX);
                debug_assert!(new != u32::MAX, "live object on evicted shape");
                if new != u32::MAX {
                    obj.shape = ShapeId(new);
                }
            }
        }
        // Shape ids were renumbered: any per-call-site inline cache
        // (`InlineCache`) entry resolved under the old numbering may now
        // name a different layout. Bump the epoch so those entries read
        // as stale (miss) instead of wrong; fresh probes refill them.
        // (The early returns above leave the table untouched, so no bump
        // is needed there: ids are still valid.)
        self.shape_epoch = self.shape_epoch.wrapping_add(1);
        self.shapes = kept;
    }

    /// Debug-only consistency check: the shape mirrors the tracked prefix
    /// of the lists exactly — every tracked slot's key resolves through
    /// the shape's index to its own (list, position), and the index holds
    /// no extra keys. (Keys past [`Shape::MAX_ENTRIES`] per list are in
    /// dictionary mode: untracked by the shape, resolved by the fallback
    /// scan.)
    #[cfg(test)]
    fn debug_shape_consistent(&self, obj: Gc<Object>) -> bool {
        let o = self.get(obj);
        let shape = &self.shapes[o.shape.0 as usize];
        let cap = Shape::MAX_ENTRIES as usize;
        let dn = o.properties.len().min(cap);
        let an = o.accessors.len().min(cap);
        if shape.index.len() != dn + an {
            return false;
        }
        for (i, (k, _)) in o.properties.iter().take(dn).enumerate() {
            if shape.lookup(k.as_wtf8()) != Some((0, i as u32)) {
                return false;
            }
        }
        for (i, (k, _)) in o.accessors.iter().take(an).enumerate() {
            if shape.lookup(k.as_wtf8()) != Some((1, i as u32)) {
                return false;
            }
        }
        true
    }

    /// Define or overwrite an own property `name` on `obj`. The key is
    /// interned, so repeated definitions reuse one key allocation.
    ///
    /// This is the string-key entry point; it delegates to
    /// [`Heap::define_property_key`] with [`PropKey::Str`].
    /// Defining a data property evicts an accessor property under the same
    /// key (and vice versa for [`Heap::define_accessor`]): a key lives in
    /// exactly one of the two lists.
    ///
    /// Array exotic `[[DefineOwnProperty]]` (spec 10.4.2) is centralized in
    /// [`Heap::define_array_own_property`]: indexed and `"length"` defines
    /// on arrays extend `length`, truncate trailing elements on shrink,
    /// and enforce the non-writable-`length` and Uint32 rules there.
    ///
    /// # Panics
    ///
    /// Panics if `obj` or any handle inside `prop` was created by a
    /// different heap.
    pub fn define_property(
        &mut self,
        obj: Gc<Object>,
        name: &Wtf8,
        prop: Property,
    ) -> Result<(), AllocError> {
        self.define_property_key(obj, PropKey::Str(name), prop)
    }

    /// [`Heap::define_property`] generalized over the key kind.
    ///
    /// `PropKey::Str` behaves exactly like `define_property` (array
    /// exotic routing, dense fast paths, interning, shape transitions).
    /// `PropKey::Symbol` defines an own symbol-keyed property in the
    /// side tables ([`Object::symbol_properties`] /
    /// [`Object::symbol_accessors`]): no interning (the key is already a
    /// heap handle), no shape transition (symbols bypass the shape), and
    /// no array-exotic routing (the exotic covers `"length"` and
    /// canonical indices, which are strings — a symbol define on an
    /// array is ordinary).
    ///
    /// # Panics
    ///
    /// Panics if `obj`, the symbol key, or any handle inside `prop` was
    /// created by a different heap.
    pub fn define_property_key(
        &mut self,
        obj: Gc<Object>,
        key: PropKey<'_>,
        prop: Property,
    ) -> Result<(), AllocError> {
        match key {
            PropKey::Str(name) => {
                self.note_reset_write(obj, Some(name));
                self.check_value(prop.value);
                if self.get(obj).kind == ObjectKind::Array
                    && (name == "length" || dense_index_key(name).is_some())
                {
                    // Route through the array exotic. Its spec-level failure
                    // (`Ok(false)`) and `RangeError` (non-Uint32 `length`) have no
                    // channel through `Result<(), AllocError>`: the write is
                    // skipped and success reported. Callers that need the failure
                    // signal — `Object.defineProperty`, VM assignment, and
                    // anything else that must throw or return `false` — call
                    // `define_array_own_property` directly instead.
                    let desc = ArrayDefineDesc {
                        value: Some(prop.value),
                        writable: Some(prop.writable),
                        enumerable: Some(prop.enumerable),
                        configurable: Some(prop.configurable),
                        get: None,
                        set: None,
                    };
                    return match self.define_array_own_property(obj, name, &desc) {
                        Ok(_) => Ok(()),
                        Err(NativeError::OutOfMemory) => Err(AllocError),
                        Err(_) => Ok(()),
                    };
                }
                self.define_property_ordinary(obj, name, prop)
            }
            PropKey::Symbol(sym) => {
                // Conservative for the reset epoch: a symbol key is
                // invisible to the reset key scan, so bump the epoch
                // instead of recording the key.
                self.note_reset_write(obj, None);
                self.check_gc(sym);
                self.check_value(prop.value);
                self.define_property_ordinary_key(obj, key, prop)
            }
        }
    }

    /// [`define_property`](Self::define_property) with a `&str` name, for
    /// builtin installation: well-known names (`"length"`, `"prototype"`,
    /// method names) are ASCII and always well-formed, so the conversion
    /// is infallible.
    pub fn define_property_str(
        &mut self,
        obj: Gc<Object>,
        name: &str,
        prop: Property,
    ) -> Result<(), AllocError> {
        self.define_property(obj, Wtf8::from_str(name), prop)
    }

    /// The ordinary (non-exotic) define: dense fast paths plus the
    /// property map. This is what [`Heap::define_property`] used to do
    /// inline; the array exotic ([`Heap::define_array_own_property`])
    /// calls it for the ordinary sub-defines so the dense fast paths stay
    /// coherent (indexed writes stay dense when attribute-less, otherwise
    /// the array materializes — the rules from the dense fix are
    /// unchanged).
    ///
    /// Unlike `define_property` this performs no array-exotic handling and
    /// no validation: callers pass already-validated descriptors. The
    /// value must have passed [`Heap::check_value`].
    fn define_property_ordinary(
        &mut self,
        obj: Gc<Object>,
        name: &Wtf8,
        prop: Property,
    ) -> Result<(), AllocError> {
        self.define_property_ordinary_key(obj, PropKey::Str(name), prop)
    }

    /// [`Heap::define_property_ordinary`] generalized over the key kind.
    ///
    /// The `PropKey::Str` path is the ordinary define documented on
    /// [`Heap::define_property`]: dense fast paths plus the property map,
    /// with the shape discipline described below. The `PropKey::Symbol`
    /// path writes the symbol side tables directly: no dense handling
    /// (dense keys are strings), no interning (the key is already a heap
    /// handle), and no shape work (symbols bypass the shape entirely, so
    /// a symbol define never transitions it). The side-table push cannot
    /// trigger a collection — `Vec` grows through the system allocator,
    /// not the GC heap — so no rooting is needed across it.
    ///
    /// Unlike `define_property` this performs no array-exotic handling and
    /// no validation: callers pass already-validated descriptors. The
    /// value must have passed [`Heap::check_value`].
    fn define_property_ordinary_key(
        &mut self,
        obj: Gc<Object>,
        key: PropKey<'_>,
        prop: Property,
    ) -> Result<(), AllocError> {
        // Extract the string key, or take the symbol side-table path.
        let name = match key {
            PropKey::Str(name) => name,
            PropKey::Symbol(sym) => {
                let o = self.get_object_mut(obj);
                // One-list invariant, mirrored from the string path: a
                // symbol key lives in exactly one of the two side tables,
                // so defining data evicts an accessor under the same
                // symbol.
                if let Some(i) = o.symbol_accessors.iter().position(|(k, _)| *k == sym) {
                    o.symbol_accessors.remove(i);
                }
                match o.symbol_properties.iter().position(|(k, _)| *k == sym) {
                    Some(i) => o.symbol_properties[i].1 = prop,
                    None => o.symbol_properties.push((sym, prop)),
                }
                return Ok(());
            }
        };
        if dense_is_active(self.get(obj)) {
            if name == "length" {
                let new_len = dense_coerce_len(prop.value);
                // The dense store keeps no attributes: a non-writable
                // `length` cannot be represented densely, so materialize
                // and store through the ordinary path, which keeps the
                // full descriptor intact. Lengths past DENSE_MAX_LEN are
                // valid (up to 2^32 - 1) but not densely representable,
                // so they materialize too instead of clamping.
                if !prop.writable
                    || new_len > DENSE_MAX_LEN
                    || !dense_gap_ok(self.get(obj), new_len)
                {
                    // Too sparse, too long, or attributed: fall back to the
                    // property map, then store through the ordinary path below.
                    self.materialize_dense(obj)?;
                } else {
                    let o = self.get_object_mut(obj);
                    let dense = o.dense_mut().expect("dense checked above");
                    dense.resize(new_len as usize, None);
                    return Ok(());
                }
            } else if let Some(i) = dense_index_key(name) {
                let i = u64::from(i);
                // Dense elements are always writable, enumerable, and
                // configurable; a descriptor with any attribute cleared
                // cannot live in the dense store (which keeps values
                // only), so materialize and store through the ordinary
                // path instead of silently dropping the attributes.
                if !prop.writable
                    || !prop.enumerable
                    || !prop.configurable
                    || !dense_gap_ok(self.get(obj), i + 1)
                {
                    // Too sparse or attributed: fall back to the property
                    // map, then store through the ordinary path below.
                    self.materialize_dense(obj)?;
                } else {
                    let evicted = {
                        let o = self.get_object_mut(obj);
                        match o.accessors.iter().position(|(k, _)| k.as_wtf8() == name) {
                            Some(pos) => {
                                o.accessors.remove(pos);
                                true
                            }
                            None => false,
                        }
                    };
                    // Removal shifts accessor indices: rebuild the shape
                    // so it mirrors the lists again.
                    if evicted {
                        self.shape_rebuild(obj);
                    }
                    let o = self.get_object_mut(obj);
                    let dense = o.dense_mut().expect("dense checked above");
                    let i = i as usize;
                    if i < dense.len() {
                        dense[i] = Some(prop.value);
                    } else {
                        dense.resize(i + 1, None);
                        dense[i] = Some(prop.value);
                    }
                    return Ok(());
                }
            }
        }
        // Overwrite fast path: resolve the slot through the object's shape
        // *before* interning. Overwriting an existing property then needs
        // no intern-table operation at all — no hash insert, and no heap
        // allocation even for a short key — because the slot already holds
        // the canonical key and an overwrite leaves the key untouched. The
        // probe re-verifies the slot still holds the shape's key, so a hit
        // is exact; a miss falls through to the path below. (Interning
        // cannot turn a miss into a hit: it never changes the shape, and
        // no allocation — and therefore no collection — occurs between
        // the probe and the slot write, so the shape cannot be renumbered
        // mid-path.)
        if let Some((0, idx)) = self.shape_lookup(obj, name)
            && let Some(slot) = self.get_object_mut(obj).properties.get_mut(idx as usize)
            && slot.0.as_wtf8() == name
        {
            slot.1 = prop;
            return Ok(());
        }
        // `intern_str` may allocate for long keys and therefore collect;
        // root `obj` and the incoming property value across it so a
        // collection cannot free either before they are linked into the
        // object. Short keys pack inline with no allocation at all, so the
        // roots below are then just unused insurance.
        let base = self.temp_roots.len();
        self.temp_roots.push(Value::Object(obj));
        self.temp_roots.push(prop.value);
        let key = self.intern_wtf8(name);
        self.temp_roots.truncate(base);
        let key = key?;
        // The shape always mirrors the lists: a push transitions to the
        // child shape; evicting the accessor shifts that list's indices,
        // so rebuild from the current lists instead. An overwrite changes
        // no layout and leaves the shape alone.
        let shape = self.get(obj).shape;
        let (evicted_accessor, pushed) = {
            let o = self.get_object_mut(obj);
            // Content comparison: short keys are not interned, so slot
            // keys are not pointer-identical across definitions; `Str`'s
            // equality keeps its pointer-identity fast path for the
            // interned (long-key) case.
            let evicted = match o.accessors.iter().position(|(k, _)| *k == key) {
                Some(i) => {
                    o.accessors.remove(i);
                    true
                }
                None => false,
            };
            let pushed = match o.properties.iter().position(|(k, _)| *k == key) {
                Some(i) => {
                    o.properties[i].1 = prop;
                    false
                }
                None => {
                    o.properties.push((key, prop));
                    true
                }
            };
            (evicted, pushed)
        };
        if evicted_accessor {
            self.shape_rebuild(obj);
        } else if pushed {
            let new_shape = self.shape_add(shape, key, 0);
            self.get_object_mut(obj).shape = new_shape;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Array exotic `[[DefineOwnProperty]]` (spec 10.4.2)
    //
    // One checked entry point used by every throwing/boolean path:
    // `Object.defineProperty`, the VM's property assignment, the array
    // builtins' `set_idx`/`set_len`, and `Heap::define_property`'s
    // internal routing. It returns the spec-level result —
    // `Ok(true)`/`Ok(false)` — and throws `NativeError::RangeError` for
    // non-Uint32 `length` values, so each caller can throw, return
    // `false`, or ignore the failure per its own contract.
    // ------------------------------------------------------------------

    /// Array exotic `[[DefineOwnProperty]]` (spec 10.4.2). `name` is
    /// `"length"` or a canonical array index (the caller checks).
    ///
    /// * Indexed define at `index >= length` extends `length` to
    ///   `index + 1`; the extension fails (`Ok(false)`) when the own
    ///   `length` is non-writable.
    /// * `"length"` growth only changes `length`. Shrink deletes
    ///   trailing indexed properties in descending order; a failed
    ///   delete restores `length` to `failed_index + 1`, keeps the
    ///   requested writability per `ArraySetLength`, and returns
    ///   `Ok(false)`.
    /// * A `length` value that is not a Uint32 (`-1`, `2**32`, ...)
    ///   throws `NativeError::RangeError`.
    ///
    /// Dense arrays stay dense when the define is attribute-less; an
    /// attributed define materializes through the ordinary path (the
    /// dense rules from the dense fix are preserved).
    ///
    /// This is itself a reset write entry point: it notes the write via
    /// [`Heap::note_reset_write`] like every other entry point. Callers
    /// arriving through [`Heap::define_property`] were already noted
    /// there, so that path double-notes; the bump is conservative
    /// (a bigger epoch delta can only force a redundant scrub, never
    /// skip one), and the `reset_dirtied` set insert is idempotent.
    pub(crate) fn define_array_own_property(
        &mut self,
        arr: Gc<Object>,
        name: &Wtf8,
        desc: &ArrayDefineDesc,
    ) -> Result<bool, NativeError> {
        self.note_reset_write(arr, Some(name));
        debug_assert_eq!(self.get(arr).kind, ObjectKind::Array);
        if let Some(v) = desc.value {
            self.check_value(v);
        }
        if let Some(v) = desc.get {
            self.check_value(v);
        }
        if let Some(v) = desc.set {
            self.check_value(v);
        }
        if name == "length" {
            return self.array_set_length(arr, desc);
        }
        let index = dense_index_key(name).expect("caller checks the key");
        self.array_define_index(arr, index, desc)
    }

    /// Spec 10.4.2.3 `ArraySetLength`.
    fn array_set_length(
        &mut self,
        arr: Gc<Object>,
        desc: &ArrayDefineDesc,
    ) -> Result<bool, NativeError> {
        // ToUint32/ToNumber on the new length, with the RangeError rule.
        let new_len = if let Some(v) = desc.value {
            let n = crate::builtin_object_array::to_number(v)?;
            let u = to_uint32_bits(n);
            if (u as f64) != n {
                return Err(NativeError::RangeError("invalid array length".into()));
            }
            Some(u as u64)
        } else {
            None
        };
        let old_len_prop = self.array_len_value(arr);
        let (old_len, old_writable) = match &old_len_prop {
            Some(p) => (as_u64_len(p.value), p.writable),
            // Defensive: an array always has `length`, but if it does not
            // the spec reads oldLen as +0 and oldLenDesc as writable.
            None => (0, true),
        };
        // newLenDesc: copy of Desc with [[Value]] canonicalized to newLen.
        let mut new_len_desc = *desc;
        if let Some(len) = new_len {
            new_len_desc.value = Some(u64_to_value(len));
        }
        if new_len.is_none_or(|len| len >= old_len) {
            // Growth, same length, or absent [[Value]] (spec: absent means
            // newLen = oldLen, which always lands here): length-only
            // define, attributes validated. Note the non-writable check
            // below is correctly skipped on this path.
            return self.ordinary_define_validated(arr, Wtf8::from_str("length"), &new_len_desc);
        }
        if !old_writable {
            // Non-writable length rejects any shrink.
            return Ok(false);
        }
        // Shrink: validate the interim descriptor first (spec 8.e), then
        // delete trailing indices in descending order, then publish the
        // new length. Deleting before the dense store is resized keeps
        // elements the loop never reaches intact when a delete fails, so
        // the restore below only reports `failed_index + 1`.
        let new_writable = desc.writable.unwrap_or(true);
        let mut interim = new_len_desc;
        interim.writable = Some(true);
        let completed = self.complete_array_desc(arr, Wtf8::from_str("length"), &interim);
        if !self.validate_desc(arr, Wtf8::from_str("length"), &completed)? {
            return Ok(false);
        }
        if let Some(len) = new_len {
            for index in self.array_own_indices_from(arr, len) {
                if !self.array_delete_index(arr, index) {
                    // Restore length to failed_index + 1 with writability
                    // forced true (ArraySetLength step 8.f.ii: a partial
                    // descriptor, so the other attributes are preserved).
                    // The deletes that succeeded stay deleted, the rest
                    // are untouched.
                    let cur = self.array_len_value(arr);
                    let restore = Property {
                        value: u64_to_value(index as u64 + 1),
                        writable: true,
                        enumerable: cur.as_ref().is_some_and(|p| p.enumerable),
                        configurable: cur.as_ref().is_some_and(|p| p.configurable),
                    };
                    self.define_property_ordinary(arr, Wtf8::from_str("length"), restore)
                        .map_err(NativeError::from)?;
                    return Ok(false);
                }
            }
            // Every trailing index is gone: publish the interim length.
            // Re-validation passes — the deletes never touch "length" —
            // but report `false` rather than asserting if it ever did not.
            if !self.ordinary_define_validated(arr, Wtf8::from_str("length"), &interim)? {
                return Ok(false);
            }
            if !new_writable {
                // Success path: apply the requested non-writable flag
                // (ArraySetLength step 8.g: partial descriptor, other
                // attributes preserved).
                let cur = self.array_len_value(arr);
                let final_desc = Property {
                    value: u64_to_value(len),
                    writable: false,
                    enumerable: cur.as_ref().is_some_and(|p| p.enumerable),
                    configurable: cur.as_ref().is_some_and(|p| p.configurable),
                };
                self.define_property_ordinary(arr, Wtf8::from_str("length"), final_desc)
                    .map_err(NativeError::from)?;
            }
        }
        Ok(true)
    }

    /// Spec 10.4.2.1 indexed define: when the index extends past a
    /// non-writable `length`, fail without defining the index; otherwise
    /// the ordinary define, then extend `length` to `index + 1`.
    fn array_define_index(
        &mut self,
        arr: Gc<Object>,
        index: u32,
        desc: &ArrayDefineDesc,
    ) -> Result<bool, NativeError> {
        let old_len_prop = self.array_len_value(arr);
        let old_len = old_len_prop.as_ref().map_or(0, |p| as_u64_len(p.value));
        if (index as u64) >= old_len && old_len_prop.as_ref().is_some_and(|p| !p.writable) {
            return Ok(false);
        }
        let succeeded =
            self.ordinary_define_validated(arr, Wtf8::from_str(&index.to_string()), desc)?;
        if !succeeded || (index as u64) < old_len {
            return Ok(succeeded);
        }
        // Spec 10.4.2.1 step 2.g: length is set with a partial descriptor
        // ({ [[Value]]: index + 1 }), so the current length attributes are
        // preserved — not reset to the spec defaults. Arrays without a
        // length property (bare `alloc_object` arrays managed manually by
        // bindings) are left alone: the explicit length define that
        // follows establishes it with the caller's attributes.
        let Some(old) = old_len_prop else {
            return Ok(true);
        };
        let len_prop = Property {
            value: u64_to_value(index as u64 + 1),
            writable: old.writable,
            enumerable: old.enumerable,
            configurable: old.configurable,
        };
        self.define_property_ordinary(arr, Wtf8::from_str("length"), len_prop)
            .map_err(NativeError::from)?;
        Ok(true)
    }

    /// Complete a possibly-partial descriptor against the existing own
    /// property (generic descriptors become data descriptors with the
    /// current value), then run `ValidateAndApplyPropertyDescriptor`.
    /// Returns the spec boolean: `false` when validation rejects the
    /// redefinition.
    fn ordinary_define_validated(
        &mut self,
        obj: Gc<Object>,
        name: &Wtf8,
        desc: &ArrayDefineDesc,
    ) -> Result<bool, NativeError> {
        let completed = self.complete_array_desc(obj, name, desc);
        if !self.validate_desc(obj, name, &completed)? {
            return Ok(false);
        }
        match completed {
            ArrayCompletedDesc::Data(p) => {
                self.define_property_ordinary(obj, name, p)
                    .map_err(NativeError::from)?;
            }
            ArrayCompletedDesc::Accessor {
                get,
                set,
                enumerable,
                configurable,
            } => {
                let prop = AccessorProp {
                    get,
                    set,
                    enumerable,
                    configurable,
                };
                self.define_accessor(obj, name, prop)
                    .map_err(NativeError::from)?;
            }
        }
        Ok(true)
    }

    /// Spec 10.4.2.1 step 2 / 10.4.2.2 step 2: complete `desc` against
    /// the current own property. A missing data/accessor field pair
    /// makes a generic descriptor, which completes as a data descriptor
    /// carrying the existing value (or `undefined`).
    fn complete_array_desc(
        &self,
        obj: Gc<Object>,
        name: &Wtf8,
        desc: &ArrayDefineDesc,
    ) -> ArrayCompletedDesc {
        let current_data = self.get_property(obj, name);
        let current_acc = if current_data.is_none() {
            self.get_accessor(obj, name)
        } else {
            None
        };
        let is_acc_desc = desc.get.is_some() || desc.set.is_some();
        let is_data_desc = desc.value.is_some() || desc.writable.is_some();
        // A generic descriptor keeps the current kind; an absent key
        // completes as data.
        if is_acc_desc || (!is_data_desc && current_acc.is_some()) {
            let (get, set, enumerable, configurable) = match current_acc {
                Some(a) => (a.get, a.set, a.enumerable, a.configurable),
                // Data -> accessor: get/set reset to defaults, the
                // attributes carry over from the data property.
                None => match current_data {
                    Some(p) => (
                        Value::Undefined,
                        Value::Undefined,
                        p.enumerable,
                        p.configurable,
                    ),
                    None => (Value::Undefined, Value::Undefined, false, false),
                },
            };
            ArrayCompletedDesc::Accessor {
                get: desc.get.unwrap_or(get),
                set: desc.set.unwrap_or(set),
                enumerable: desc.enumerable.unwrap_or(enumerable),
                configurable: desc.configurable.unwrap_or(configurable),
            }
        } else {
            let (value, writable, enumerable, configurable) = match (current_data, current_acc) {
                (Some(p), _) => (p.value, p.writable, p.enumerable, p.configurable),
                // Accessor -> data: value/writable reset to defaults,
                // the attributes carry over.
                (_, Some(a)) => (Value::Undefined, false, a.enumerable, a.configurable),
                (None, None) => (Value::Undefined, false, false, false),
            };
            ArrayCompletedDesc::Data(Property {
                value: desc.value.unwrap_or(value),
                writable: desc.writable.unwrap_or(writable),
                enumerable: desc.enumerable.unwrap_or(enumerable),
                configurable: desc.configurable.unwrap_or(configurable),
            })
        }
    }

    /// Spec `ValidateAndApplyPropertyDescriptor`: `false` when the
    /// redefinition is rejected (non-configurable/non-writable rules),
    /// `true` when it may proceed. `extensible` is consulted for new
    /// properties. Accessor-vs-data mismatches and attribute changes
    /// follow the spec; the completed descriptor is always complete, so
    /// no defaulting remains.
    fn validate_desc(
        &self,
        obj: Gc<Object>,
        name: &Wtf8,
        desc: &ArrayCompletedDesc,
    ) -> Result<bool, NativeError> {
        let o = self.get(obj);
        let extensible = o.extensible;
        let current_data = self.get_property(obj, name);
        let current_acc = self.get_accessor(obj, name);
        if current_data.is_none() && current_acc.is_none() {
            return Ok(extensible);
        }
        let desc_is_accessor = matches!(desc, ArrayCompletedDesc::Accessor { .. });
        match (current_data, current_acc, desc_is_accessor) {
            (Some(cur), _, false) => {
                if !cur.configurable {
                    let ArrayCompletedDesc::Data(p) = desc else {
                        unreachable!("matched on is_accessor == false")
                    };
                    if p.configurable || p.enumerable != cur.enumerable {
                        return Ok(false);
                    }
                    if !cur.writable && (p.writable || !array_same_value(p.value, cur.value)) {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            (_, Some(cur), true) => {
                if !cur.configurable {
                    let ArrayCompletedDesc::Accessor { get, set, .. } = desc else {
                        unreachable!("matched on is_accessor == true")
                    };
                    if !array_same_value(*get, cur.get) || !array_same_value(*set, cur.set) {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            // Data <-> accessor change.
            (_, _, _) => {
                let configurable = match (current_data, current_acc) {
                    (Some(p), _) => p.configurable,
                    (_, Some(a)) => a.configurable,
                    _ => unreachable!("an own property exists here"),
                };
                Ok(configurable)
            }
        }
    }

    /// The array's own `length` data property, if present. (Spec
    /// `OrdinaryGetOwnProperty(A, "length")`.)
    fn array_len_value(&self, arr: Gc<Object>) -> Option<Property> {
        self.get_property(arr, Wtf8::from_str("length"))
    }

    /// Spec `[[Delete]]` for the `ArraySetLength` loop: returns `false`
    /// without deleting when the own index is non-configurable (data or
    /// accessor). Dense slots are always configurable — an attributed
    /// define materializes the array — so they delete unconditionally.
    fn array_delete_index(&mut self, arr: Gc<Object>, index: u32) -> bool {
        let key = index.to_string();
        if let Some(p) = self.get_property(arr, Wtf8::from_str(&key)) {
            if !p.configurable {
                return false;
            }
        } else if self
            .get_accessor(arr, Wtf8::from_str(&key))
            .is_some_and(|a| !a.configurable)
        {
            return false;
        }
        self.delete_property(arr, Wtf8::from_str(&key))
    }

    /// Own canonical array-index keys at or above `floor`, in descending
    /// numeric order, for the `ArraySetLength` delete loop. Covers dense
    /// entries, materialized/ordinary indexed properties, and indexed
    /// accessors (a shrinking `length` deletes those too).
    fn array_own_indices_from(&self, arr: Gc<Object>, floor: u64) -> Vec<u32> {
        let mut out = Vec::new();
        let o = self.get(arr);
        if let Some(dense) = o.dense() {
            for (i, slot) in dense.iter().enumerate() {
                let idx = i as u64;
                if idx >= floor && slot.is_some() {
                    out.push(i as u32);
                }
            }
        }
        for (k, _) in o.properties.iter() {
            if let Some(idx) = dense_index_key(k.as_wtf8())
                && (idx as u64) >= floor
            {
                out.push(idx);
            }
        }
        for (k, _) in o.accessors.iter() {
            if let Some(idx) = dense_index_key(k.as_wtf8())
                && (idx as u64) >= floor
            {
                out.push(idx);
            }
        }
        out.sort_unstable();
        out.dedup();
        out.reverse();
        out
    }

    /// Define or overwrite an own accessor property `name` on `obj`
    /// (getter/setter pair). Evicts a data property under the same key.
    ///
    /// # Panics
    ///
    /// Panics if `obj` or any handle inside `prop` was created by a
    /// different heap.
    /// [`define_accessor`](Self::define_accessor) with a `&str` name, for
    /// builtin installation (see [`define_property_str`](Self::define_property_str)).
    pub fn define_accessor_str(
        &mut self,
        obj: Gc<Object>,
        name: &str,
        prop: AccessorProp,
    ) -> Result<(), AllocError> {
        self.define_accessor(obj, Wtf8::from_str(name), prop)
    }

    pub fn define_accessor(
        &mut self,
        obj: Gc<Object>,
        name: &Wtf8,
        prop: AccessorProp,
    ) -> Result<(), AllocError> {
        self.define_accessor_key(obj, PropKey::Str(name), prop)
    }

    /// [`Heap::define_accessor`] generalized over the key kind: define or
    /// overwrite an own accessor property (getter/setter pair), evicting
    /// a data property under the same key.
    ///
    /// `PropKey::Str` behaves exactly like `define_accessor`. The
    /// `PropKey::Symbol` path writes the symbol side tables directly: no
    /// interning, no shape work (symbols bypass the shape entirely, so a
    /// symbol define never transitions it), and no dense handling (dense
    /// keys are strings).
    ///
    /// # Panics
    ///
    /// Panics if `obj`, the symbol key, or any handle inside `prop` was
    /// created by a different heap.
    pub fn define_accessor_key(
        &mut self,
        obj: Gc<Object>,
        key: PropKey<'_>,
        prop: AccessorProp,
    ) -> Result<(), AllocError> {
        // Extract the string key, or take the symbol side-table path.
        let name = match key {
            PropKey::Str(name) => name,
            PropKey::Symbol(sym) => {
                // Conservative for the reset epoch, as in
                // `define_property_key`.
                self.note_reset_write(obj, None);
                self.check_gc(sym);
                self.check_value(prop.get);
                self.check_value(prop.set);
                let o = self.get_object_mut(obj);
                // One-list invariant, mirrored from the string path.
                if let Some(i) = o.symbol_properties.iter().position(|(k, _)| *k == sym) {
                    o.symbol_properties.remove(i);
                }
                match o.symbol_accessors.iter().position(|(k, _)| *k == sym) {
                    Some(i) => o.symbol_accessors[i].1 = prop,
                    None => o.symbol_accessors.push((sym, prop)),
                }
                return Ok(());
            }
        };
        self.note_reset_write(obj, Some(name));
        self.check_value(prop.get);
        self.check_value(prop.set);
        // Dense arrays: an accessor on an index evicts the dense data slot,
        // preserving the "a key lives in exactly one of the two lists"
        // invariant.
        if let Some(i) = dense_index_key(name)
            && dense_is_active(self.get(obj))
        {
            let o = self.get_object_mut(obj);
            if let Some(dense) = o.dense_mut()
                && let Some(slot) = dense.get_mut(i as usize)
            {
                *slot = None;
            }
        }
        // Overwrite fast path, mirrored from `define_property`: probe the
        // shape before interning so overwriting an existing accessor needs
        // no intern-table operation at all. The probe re-verifies the slot
        // still holds the shape's key, so a hit is exact.
        if let Some((1, idx)) = self.shape_lookup(obj, name)
            && let Some(slot) = self.get_object_mut(obj).accessors.get_mut(idx as usize)
            && slot.0.as_wtf8() == name
        {
            slot.1 = prop;
            return Ok(());
        }
        let base = self.temp_roots.len();
        self.temp_roots.push(Value::Object(obj));
        self.temp_roots.push(prop.get);
        self.temp_roots.push(prop.set);
        let key = self.intern_wtf8(name);
        self.temp_roots.truncate(base);
        let key = key?;
        // Mirror the shape discipline of `define_property_ordinary`: a
        // push transitions, an evicted data property shifts indices and
        // rebuilds, an overwrite leaves the shape alone.
        let shape = self.get(obj).shape;
        let (evicted_data, pushed) = {
            let o = self.get_object_mut(obj);
            // Content comparison (see `define_property_ordinary`): short
            // keys are not interned.
            let evicted = match o.properties.iter().position(|(k, _)| *k == key) {
                Some(i) => {
                    o.properties.remove(i);
                    true
                }
                None => false,
            };
            let pushed = match o.accessors.iter().position(|(k, _)| *k == key) {
                Some(i) => {
                    o.accessors[i].1 = prop;
                    false
                }
                None => {
                    o.accessors.push((key, prop));
                    true
                }
            };
            (evicted, pushed)
        };
        if evicted_data {
            self.shape_rebuild(obj);
        } else if pushed {
            let new_shape = self.shape_add(shape, key, 1);
            self.get_object_mut(obj).shape = new_shape;
        }
        Ok(())
    }

    /// Look up an own accessor property by name (no prototype walk).
    ///
    /// # Panics
    ///
    /// Panics if `obj` was created by a different heap.
    pub fn get_accessor(&self, obj: Gc<Object>, name: &Wtf8) -> Option<AccessorProp> {
        self.get_accessor_key(obj, PropKey::Str(name))
    }

    /// [`Heap::get_accessor`] generalized over the key kind: look up an
    /// own accessor property (no prototype walk).
    ///
    /// `PropKey::Str` behaves exactly like `get_accessor`. The
    /// `PropKey::Symbol` path scans the symbol side table linearly — no
    /// inline cache, no shape probe.
    ///
    /// # Panics
    ///
    /// Panics if `obj` or the symbol key was created by a different heap.
    pub fn get_accessor_key(&self, obj: Gc<Object>, key: PropKey<'_>) -> Option<AccessorProp> {
        match key {
            PropKey::Str(name) => self.get_accessor_inner(obj, name, None),
            PropKey::Symbol(sym) => {
                self.check_gc(obj);
                self.check_gc(sym);
                self.get(obj)
                    .symbol_accessors
                    .iter()
                    .find(|(k, _)| *k == sym)
                    .map(|(_, a)| *a)
            }
        }
    }

    /// [`Heap::get_accessor`] with a per-call-site inline cache: on a
    /// cache hit the [`Shape::lookup`] string hash is skipped entirely;
    /// on a miss the hash runs and fills the cache. See
    /// [`InlineCache`](crate::value::InlineCache) for the invalidation
    /// contract.
    pub(crate) fn get_accessor_cached(
        &self,
        obj: Gc<Object>,
        name: &Wtf8,
        ic: &Cell<InlineCache>,
    ) -> Option<AccessorProp> {
        self.get_accessor_inner(obj, name, Some(ic))
    }

    fn get_accessor_inner(
        &self,
        obj: Gc<Object>,
        name: &Wtf8,
        ic: Option<&Cell<InlineCache>>,
    ) -> Option<AccessorProp> {
        self.check_gc(obj);
        // Inline-cache probe: a (shape-epoch, shape) hit skips the
        // string hash below; the slot probe re-verifies the key, so a
        // hit is exact.
        if let Some(ic) = ic {
            let e = ic.get();
            let o = self.get(obj);
            if e.list == 1
                && e.epoch == self.shape_epoch()
                && e.shape == o.shape.0
                && let Some((k, p)) = o.accessors.get(e.slot as usize)
                && k.as_wtf8() == name
            {
                return Some(*p);
            }
        }
        // Fast path: resolve the slot through the object's hidden class;
        // the probe re-verifies the slot still holds the indexed key.
        if let Some((1, idx)) = self.shape_lookup(obj, name)
            && let Some((k, p)) = self.get(obj).accessors.get(idx as usize)
            && k.as_wtf8() == name
        {
            // Fill the call-site cache (see `Heap::fill_ic`).
            if let Some(ic) = ic {
                self.fill_ic(ic, obj, 1, idx);
            }
            return Some(*p);
        }
        // Fallback linear scan (absent key, or an unreachable missed
        // shape update).
        self.get(obj)
            .accessors
            .iter()
            .find(|(k, _)| k.as_wtf8() == name)
            .map(|(_, p)| *p)
    }

    /// Look up an own property by name (no prototype walk).
    ///
    /// Dense arrays (see [`Heap::new_dense_array`]) take a fast path:
    /// `"length"` is synthesized from the dense length and integer indices
    /// read straight from the dense store, with no linear scan.
    ///
    /// # Panics
    ///
    /// Panics if `obj` was created by a different heap.
    pub fn get_property(&self, obj: Gc<Object>, name: &Wtf8) -> Option<Property> {
        self.get_property_key(obj, PropKey::Str(name))
    }

    /// [`Heap::get_property`] generalized over the key kind: look up an
    /// own data property (no prototype walk).
    ///
    /// `PropKey::Str` behaves exactly like `get_property`. The
    /// `PropKey::Symbol` path scans the symbol side table linearly: no
    /// dense synthesis, no inline cache, and no shape probe — those are
    /// string-key machinery. Symbol keys never live in the dense store.
    ///
    /// # Panics
    ///
    /// Panics if `obj` or the symbol key was created by a different heap.
    pub fn get_property_key(&self, obj: Gc<Object>, key: PropKey<'_>) -> Option<Property> {
        match key {
            PropKey::Str(name) => self.get_property_inner(obj, name, None),
            PropKey::Symbol(sym) => {
                self.check_gc(obj);
                self.check_gc(sym);
                self.get(obj)
                    .symbol_properties
                    .iter()
                    .find(|(k, _)| *k == sym)
                    .map(|(_, p)| *p)
            }
        }
    }

    /// Own symbol keys in `[[OwnPropertyKeys]]` tail order: the data
    /// side table ([`Object::symbol_properties`]) in list order, then the
    /// accessor side table ([`Object::symbol_accessors`]) in list order.
    ///
    /// Data and accessor properties live in separate lists with no shared
    /// insertion sequence (a data↔accessor conversion moves the key to the
    /// end of the other list), so among symbol keys data properties
    /// precede accessor properties — the same documented approximation
    /// the string-key enumerations use.
    ///
    /// The returned handles are GC-managed: root them (e.g. with
    /// [`Heap::protect`]) before any call that can collect.
    ///
    /// # Panics
    ///
    /// Panics if `obj` was created by a different heap.
    pub fn own_symbol_keys(&self, obj: Gc<Object>) -> Vec<Gc<Symbol>> {
        self.check_gc(obj);
        let o = self.get(obj);
        o.symbol_properties
            .iter()
            .map(|(k, _)| *k)
            .chain(o.symbol_accessors.iter().map(|(k, _)| *k))
            .collect()
    }

    /// [`Heap::own_symbol_keys`], keeping only enumerable properties.
    /// Used by `Object.assign`, object spread, and anywhere else the spec
    /// copies "own enumerable" symbol-keyed properties.
    ///
    /// The returned handles are GC-managed: root them (e.g. with
    /// [`Heap::protect`]) before any call that can collect.
    ///
    /// # Panics
    ///
    /// Panics if `obj` was created by a different heap.
    pub fn own_enumerable_symbol_keys(&self, obj: Gc<Object>) -> Vec<Gc<Symbol>> {
        self.check_gc(obj);
        let o = self.get(obj);
        o.symbol_properties
            .iter()
            .filter(|(_, p)| p.enumerable)
            .map(|(k, _)| *k)
            .chain(
                o.symbol_accessors
                    .iter()
                    .filter(|(_, a)| a.enumerable)
                    .map(|(k, _)| *k),
            )
            .collect()
    }

    /// [`Heap::get_property`] with a per-call-site inline cache: on a
    /// cache hit the [`Shape::lookup`] string hash is skipped entirely;
    /// on a miss the hash runs and fills the cache. See
    /// [`InlineCache`](crate::value::InlineCache) for the invalidation
    /// contract.
    pub(crate) fn get_property_cached(
        &self,
        obj: Gc<Object>,
        name: &Wtf8,
        ic: &Cell<InlineCache>,
    ) -> Option<Property> {
        self.get_property_inner(obj, name, Some(ic))
    }

    fn get_property_inner(
        &self,
        obj: Gc<Object>,
        name: &Wtf8,
        ic: Option<&Cell<InlineCache>>,
    ) -> Option<Property> {
        self.check_gc(obj);
        let o = self.get(obj);
        if dense_is_active(o) {
            if name == "length" {
                let len = o.dense().map_or(0, |d| d.len() as u64);
                return Some(Property {
                    value: dense_len_value(len),
                    writable: true,
                    enumerable: false,
                    configurable: false,
                });
            }
            if let Some(i) = dense_index_key(name) {
                return o
                    .dense()
                    .and_then(|d| d.get(i as usize).copied().flatten())
                    .map(Property::data);
            }
        }
        // Inline-cache probe: a (shape-epoch, shape) hit skips the
        // string hash below; the slot probe re-verifies the key, so a
        // hit is exact.
        if let Some(ic) = ic {
            let e = ic.get();
            if e.list == 0
                && e.epoch == self.shape_epoch()
                && e.shape == o.shape.0
                && let Some((k, p)) = o.properties.get(e.slot as usize)
                && k.as_wtf8() == name
            {
                return Some(*p);
            }
        }
        // Fast path: resolve the slot through the object's hidden class.
        // The probe re-verifies the slot still holds the indexed key, so
        // a hit is exact.
        if let Some((0, idx)) = self.shape_lookup(obj, name)
            && let Some((k, p)) = self.get(obj).properties.get(idx as usize)
            && k.as_wtf8() == name
        {
            // Fill the call-site cache (see `Heap::fill_ic`).
            if let Some(ic) = ic {
                self.fill_ic(ic, obj, 0, idx);
            }
            return Some(*p);
        }
        // Fallback linear scan: the key is absent, or (unreachable in
        // practice) a missed shape update — slow but never wrong.
        self.get(obj)
            .properties
            .iter()
            .find(|(k, _)| k.as_wtf8() == name)
            .map(|(_, p)| *p)
    }

    /// Delete an own property; true if one existed (data or accessor).
    ///
    /// On dense arrays (see [`Heap::new_dense_array`]) deleting an index
    /// punches a hole in the dense store; `"length"` is non-configurable
    /// and reports `false`, matching the stored property's attributes.
    ///
    /// # Panics
    ///
    /// Panics if `obj` was created by a different heap.
    pub fn delete_property(&mut self, obj: Gc<Object>, name: &Wtf8) -> bool {
        self.delete_property_key(obj, PropKey::Str(name))
    }

    /// [`Heap::delete_property`] generalized over the key kind: delete an
    /// own property; true if one existed (data or accessor).
    ///
    /// `PropKey::Str` behaves exactly like `delete_property`. The
    /// `PropKey::Symbol` path removes the key from both symbol side
    /// tables linearly, with no shape rebuild — symbol keys never enter
    /// the shape, so the shape is already correct afterwards.
    ///
    /// # Panics
    ///
    /// Panics if `obj` or the symbol key was created by a different heap.
    pub fn delete_property_key(&mut self, obj: Gc<Object>, key: PropKey<'_>) -> bool {
        // Extract the string key, or take the symbol side-table path.
        let name = match key {
            PropKey::Str(name) => name,
            PropKey::Symbol(sym) => {
                self.check_gc(obj);
                self.check_gc(sym);
                self.note_reset_delete_symbol(obj);
                let removed = {
                    let o = self.get_object_mut(obj);
                    let mut removed = false;
                    if let Some(i) = o.symbol_properties.iter().position(|(k, _)| *k == sym) {
                        o.symbol_properties.remove(i);
                        removed = true;
                    }
                    if let Some(i) = o.symbol_accessors.iter().position(|(k, _)| *k == sym) {
                        o.symbol_accessors.remove(i);
                        removed = true;
                    }
                    removed
                };
                // No shape rebuild: symbol keys never enter the shape.
                return removed;
            }
        };
        self.note_reset_delete(obj, name);
        if dense_is_active(self.get(obj)) {
            if name == "length" {
                return false;
            }
            if let Some(i) = dense_index_key(name) {
                let (removed, evicted_accessor) = {
                    let o = self.get_object_mut(obj);
                    let mut removed = false;
                    if let Some(dense) = o.dense_mut()
                        && let Some(slot) = dense.get_mut(i as usize)
                    {
                        removed = slot.is_some();
                        *slot = None;
                    }
                    let mut evicted = false;
                    if let Some(pos) = o.accessors.iter().position(|(k, _)| k.as_wtf8() == name) {
                        o.accessors.remove(pos);
                        removed = true;
                        evicted = true;
                    }
                    (removed, evicted)
                };
                // An accessor removal shifts that list's indices (punching
                // a dense hole does not touch the property map): rebuild
                // the shape so it mirrors the lists again.
                if evicted_accessor {
                    self.shape_rebuild(obj);
                }
                return removed;
            }
        }
        // Fast path: resolve the slot through the shape; the probe
        // re-verifies the slot still holds the indexed key, so the
        // removal is exact. Any removal shifts indices, so the shape is
        // rebuilt from the lists afterwards.
        if let Some((list, idx)) = self.shape_lookup(obj, name) {
            let removed = {
                let o = self.get_object_mut(obj);
                let at = |k: &Str| k.as_wtf8() == name;
                if list == 0 {
                    if o.properties.get(idx as usize).is_some_and(|(k, _)| at(k)) {
                        o.properties.remove(idx as usize);
                        true
                    } else {
                        false
                    }
                } else if o.accessors.get(idx as usize).is_some_and(|(k, _)| at(k)) {
                    o.accessors.remove(idx as usize);
                    true
                } else {
                    false
                }
            };
            if removed {
                self.shape_rebuild(obj);
                return true;
            }
            // Stale entry (unreachable in practice): fall through to the
            // slow path with the shape about to be rebuilt anyway.
        }
        let removed = {
            let o = self.get_object_mut(obj);
            let mut removed = false;
            if let Some(i) = o.properties.iter().position(|(k, _)| k.as_wtf8() == name) {
                o.properties.remove(i);
                removed = true;
            }
            if let Some(i) = o.accessors.iter().position(|(k, _)| k.as_wtf8() == name) {
                o.accessors.remove(i);
                removed = true;
            }
            removed
        };
        if removed {
            self.shape_rebuild(obj);
        }
        removed
    }

    /// All own string keys in spec `[[OwnPropertyKeys]]` order: ascending
    /// array indices, then other string keys in insertion order. Data and
    /// accessor properties live in separate lists with no shared insertion
    /// sequence, so among non-index keys data properties precede accessor
    /// properties, each in list order (a data↔accessor conversion moves the
    /// key to the end of the other list) — the same order the engine's
    /// `Reflect.ownKeys` / `Object.getOwnPropertyNames` natives produce.
    ///
    /// Dense arrays synthesize their index keys (holes contribute nothing)
    /// and `"length"`, exactly as `Object.getOwnPropertyNames` does; the
    /// `"length"` key is listed right after the indices.
    /// (v1 gap: no symbol keys.)
    ///
    /// # Clobber immunity
    ///
    /// Reads the property lists directly; no JS-visible builtin
    /// (`Reflect.ownKeys`, `Object.getOwnPropertyNames`, ...) is consulted,
    /// so a test that replaced those builtins cannot blind the caller.
    /// Test-harness reset infrastructure: the per-test scrub enumerates
    /// added keys with this instead of the generated-JS union scan.
    ///
    /// # Panics
    ///
    /// Panics if `obj` was created by a different heap.
    pub fn own_property_keys(&self, obj: Gc<Object>) -> Vec<Str> {
        self.check_gc(obj);
        let o = self.get(obj);
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
        let mut has_length = false;
        for k in o
            .properties
            .iter()
            .map(|(k, _)| k)
            .chain(o.accessors.iter().map(|(k, _)| k))
        {
            if k.as_wtf8() == "length" {
                has_length = true;
            }
            match dense_index_key(k.as_wtf8()) {
                Some(i) => indices.push((i, *k)),
                None => others.push(*k),
            }
        }
        indices.sort_by_key(|(i, _)| *i);
        let mut out: Vec<Str> = indices.into_iter().map(|(_, k)| k).collect();
        // `Heap::get_property` synthesizes `"length"` for dense arrays (a
        // dense-active array never carries it in the property map), so
        // `[[OwnPropertyKeys]]` must list it right after the indices.
        if dense_is_active(o) && !has_length {
            out.push(Str::inline("length").expect("\"length\" fits inline"));
        }
        out.extend(others);
        out
    }

    /// Replace the prototype link.
    ///
    /// # Panics
    ///
    /// Panics if `obj` or any handle inside `proto` was created by a
    /// different heap.
    pub fn set_prototype(&mut self, obj: Gc<Object>, proto: Value) {
        self.note_reset_write(obj, None);
        self.check_value(proto);
        self.get_object_mut(obj).prototype = proto;
    }

    /// Replace the object's [`ObjectKind`] tag. Used by `Date` setters to
    /// update the kind-carried time value; the tag may hold GC handles
    /// (view/DataView buffers), hence the write barrier.
    ///
    /// # Panics
    ///
    /// Panics if `obj` was created by a different heap.
    pub(crate) fn set_object_kind(&mut self, obj: Gc<Object>, kind: ObjectKind) {
        self.note_reset_write(obj, None);
        self.get_object_mut(obj).kind = kind;
    }

    /// Mark an object non-extensible (`[[PreventExtensions]]`).
    ///
    /// # Panics
    ///
    /// Panics if `obj` was created by a different heap.
    pub fn prevent_extensions(&mut self, obj: Gc<Object>) {
        self.note_reset_write(obj, None);
        self.get_object_mut(obj).extensible = false;
    }

    /// Whether `obj` is extensible (`Object.isExtensible` state).
    ///
    /// Detection data for the test262 harness's prototype scrub:
    /// [`Heap::prevent_extensions`] already bumps the reset epoch, so a
    /// target made non-extensible shows up in
    /// [`Heap::take_dirtied_reset_objects`]; the harness snapshots this
    /// flag per target, compares it in `check_target` alongside the
    /// attribute/value checks, and repairs with
    /// [`Heap::set_extensible`].
    ///
    /// # Panics
    ///
    /// Panics if `obj` was created by a different heap.
    pub(crate) fn is_extensible(&self, obj: Gc<Object>) -> bool {
        self.get(obj).extensible
    }

    /// Set the extensibility flag directly. Scrub-repair counterpart to
    /// [`Heap::prevent_extensions`]: re-creates an extensible target that
    /// a test made non-extensible. Accounts a reset-epoch write like
    /// every other heap write entry point.
    ///
    /// # Panics
    ///
    /// Panics if `obj` was created by a different heap.
    pub(crate) fn set_extensible(&mut self, obj: Gc<Object>, extensible: bool) {
        // No-op repairs must not bump the epoch or dirty the object.
        if self.get(obj).extensible == extensible {
            return;
        }
        self.note_reset_write(obj, None);
        self.get_object_mut(obj).extensible = extensible;
    }

    // ------------------------------------------------------------------
    // Reset-epoch bookkeeping (embedder opt-in; see `reset_epoch`)
    // ------------------------------------------------------------------

    /// Current value of the mutation epoch. Embedders snapshot this after
    /// a reset and skip their scrub scans while it is unchanged.
    pub(crate) fn reset_epoch(&self) -> u64 {
        self.reset_epoch
    }

    /// Current shape-table epoch. [`Heap::compact_shapes`] bumps it
    /// whenever a collection renumbers `ShapeId`s; per-call-site
    /// inline caches ([`InlineCache`](crate::value::InlineCache))
    /// stamp the epoch they resolved under and ignore older entries.
    pub(crate) fn shape_epoch(&self) -> u64 {
        self.shape_epoch
    }

    /// Fill (or transition) a per-call-site inline-cache entry after a
    /// miss: `slot` is the freshly resolved `(list, slot)` for `obj`'s
    /// key. Within one shape epoch a shape's `(key -> slot)` mapping is
    /// immutable, so a same-shape re-fill just rewrites the entry.
    ///
    /// On a shape change the entry refills and stays monomorphic when
    /// the new shape *extends* the cached one ([`Heap::shape_extends`]):
    /// monotonic growth is the common case (adding properties, the
    /// global object gaining bindings) and must not pessimize the site.
    /// A shape that does not extend the cached one means genuinely
    /// polymorphic access: the entry goes megamorphic, so later
    /// accesses skip probing/refilling and take the slow path. A
    /// megamorphic entry keeps its marker until a shape-table epoch
    /// bump resets it to monomorphic (shape ids may have been
    /// renumbered, so older observations are dropped).
    fn fill_ic(&self, ic: &Cell<InlineCache>, obj: Gc<Object>, list: u8, slot: u32) {
        let epoch = self.shape_epoch();
        let shape = self.get(obj).shape.0;
        let e = ic.get();
        let next = if e == InlineCache::default() || e.epoch != epoch {
            // Fresh entry, or shape ids were renumbered: monomorphic.
            InlineCache {
                epoch,
                shape,
                list,
                slot,
            }
        } else if e.is_megamorphic() {
            // Megamorphic for the rest of this epoch: keep the marker.
            e
        } else if self.shape_extends(e.shape, shape) {
            // Same shape or monotonic growth: refill, stay monomorphic.
            InlineCache {
                epoch,
                shape,
                list,
                slot,
            }
        } else {
            // Genuinely polymorphic: megamorphic fallback.
            InlineCache {
                epoch,
                shape: u32::MAX,
                list: InlineCache::MEGAMORPHIC_LIST,
                slot: 0,
            }
        };
        ic.set(next);
    }

    /// True if `child`'s shape-transition ancestry includes `parent`,
    /// i.e. `child` was reached from `parent` by [`Heap::shape_add`]
    /// transitions (monotonic property growth). Parent ids strictly
    /// decrease along the chain, so the walk terminates at the root.
    fn shape_extends(&self, parent: u32, mut child: u32) -> bool {
        while child != parent {
            if child == 0 {
                return false;
            }
            child = match self.shapes.get(child as usize) {
                Some(s) => s.parent.0,
                None => return false,
            };
        }
        true
    }

    /// Register `obj` as reset-tracked: every property define/delete,
    /// prototype replacement, and `[[PreventExtensions]]` on it bumps
    /// [`Heap::reset_epoch`]. The test262 harness registers the
    /// prototype-scrub targets and their global bindings.
    pub(crate) fn track_reset_object(&mut self, obj: Gc<Object>) {
        self.reset_tracked.insert(obj);
    }

    /// Register the global object and its baseline own-keys for the reset
    /// fast path. A define/delete of a baseline key bumps the epoch; a
    /// define of any other key is recorded for
    /// [`Heap::take_reset_added_keys`] instead.
    pub(crate) fn set_reset_global(&mut self, global: Gc<Object>, baseline_keys: &[String]) {
        self.reset_global = Some(global);
        self.reset_baseline_keys = baseline_keys.iter().cloned().collect();
    }

    /// Drain the non-baseline keys defined on the reset global since the
    /// last drain. The reset fast path deletes exactly these instead of
    /// scanning the global's key set.
    pub(crate) fn take_reset_added_keys(&mut self) -> Vec<String> {
        std::mem::take(&mut self.reset_added_keys)
    }

    /// Drain the tracked objects dirtied by writes since the last drain
    /// (see [`Heap::reset_dirtied`]). The reset fast path re-scrubs
    /// exactly these instead of scanning every registered target. Order
    /// is unspecified; empty when nothing was written.
    pub(crate) fn take_dirtied_reset_objects(&mut self) -> Vec<Gc<Object>> {
        self.reset_dirtied.drain().collect()
    }

    /// Account a property/metadata write for the reset epoch. `key` is `None` for
    /// whole-object mutations (`[[SetPrototypeOf]]`,
    /// `[[PreventExtensions]]`, dense-array element writes, `ArrayBuffer`
    /// byte writes). Called at the top of every property/metadata write
    /// entry point, before validation, so even a rejected write is
    /// conservative (it can only cause a redundant full scrub, never a
    /// skipped one).
    ///
    /// Deletes go through [`Heap::note_reset_delete`] instead: a delete
    /// of a non-baseline key leaves no residue for the key scan, so it
    /// must not be recorded — otherwise the reset fast path would
    /// re-append every key it just deleted and the added list would grow
    /// without bound across tests.
    ///
    /// Deliberate exclusions (un-noted writers):
    /// - [`Heap::set_invoke_flags`] only ever writes freshly-allocated
    ///   objects (its contract: "used once, at creation time"), which
    ///   cannot be reset-tracked or be the reset global.
    /// - [`Heap::fixup_native_function_prototypes`] runs once during
    ///   builtin installation, before the heap is handed to the VM, so no
    ///   reset tracking can exist yet on this heap.
    fn note_reset_write(&mut self, obj: Gc<Object>, key: Option<&Wtf8>) {
        // The reset fast path is opt-in: nothing registered means no
        // bookkeeping at all on the hot write path.
        if self.reset_tracked.is_empty() && self.reset_global.is_none() {
            return;
        }
        if self.reset_tracked.contains(&obj) {
            self.reset_epoch = self.reset_epoch.wrapping_add(1);
            self.reset_dirtied.insert(obj);
            return;
        }
        if self.reset_global == Some(obj) {
            match key {
                // Baseline keys are identifier names (valid UTF-8), so a
                // lone-surrogate key is never a baseline key.
                Some(k)
                    if k.as_utf8()
                        .is_some_and(|s| self.reset_baseline_keys.contains(s)) =>
                {
                    self.reset_epoch = self.reset_epoch.wrapping_add(1);
                }
                Some(k) => {
                    // Non-baseline addition: the key scan would delete it;
                    // record it so the fast path deletes it directly.
                    // Lossy is safe here: the fast path falls back to the
                    // slow key scan when a recorded delete fails, so a
                    // lone-surrogate key just costs one slow reset.
                    self.reset_added_keys.push(k.to_utf8_lossy().into_owned());
                }
                None => self.reset_epoch = self.reset_epoch.wrapping_add(1),
            }
        }
    }

    /// Account a property delete for the reset epoch. Unlike
    /// [`Heap::note_reset_write`], a delete of a non-baseline global key
    /// is *not* recorded: the key is gone, so the key scan would find
    /// nothing to delete. Tracked objects and baseline keys still bump.
    fn note_reset_delete(&mut self, obj: Gc<Object>, key: &Wtf8) {
        if self.reset_tracked.is_empty() && self.reset_global.is_none() {
            return;
        }
        if self.reset_tracked.contains(&obj) {
            self.reset_epoch = self.reset_epoch.wrapping_add(1);
            self.reset_dirtied.insert(obj);
            return;
        }
        // Baseline keys are identifiers (valid UTF-8).
        if self.reset_global == Some(obj)
            && key
                .as_utf8()
                .is_some_and(|s| self.reset_baseline_keys.contains(s))
        {
            self.reset_epoch = self.reset_epoch.wrapping_add(1);
        }
    }

    /// [`Heap::note_reset_delete`] for symbol keys. The reset key scan
    /// cannot see symbol keys (enumeration is phase 2), so on a tracked
    /// or reset-global object bump the epoch conservatively instead of
    /// consulting the baseline set: a bigger delta can only force a
    /// redundant scrub, never skip one.
    fn note_reset_delete_symbol(&mut self, obj: Gc<Object>) {
        if self.reset_tracked.is_empty() && self.reset_global.is_none() {
            return;
        }
        if self.reset_tracked.contains(&obj) {
            self.reset_epoch = self.reset_epoch.wrapping_add(1);
            self.reset_dirtied.insert(obj);
            return;
        }
        if self.reset_global == Some(obj) {
            self.reset_epoch = self.reset_epoch.wrapping_add(1);
        }
    }

    // ------------------------------------------------------------------
    // Dense array element storage
    // ------------------------------------------------------------------
    //
    // Construction-time optimization for array-building builtins
    // (`map`/`filter`/`flat`/`Array.from`/spread, regexp match results,
    // …): elements are appended to a packed `Vec` with no per-element
    // string allocation, interning, or linear property scan, and `length`
    // is the vector length rather than a repeatedly rewritten property.
    // When the result is complete, [`Heap::materialize_dense`] converts
    // the array back to ordinary index properties, so user-visible
    // behavior is identical to the slow path.
    //
    // Invariant: a dense array MUST be materialized before it becomes
    // user-visible (returned to script, stored on another object, …).
    // While dense, indexed elements live only in the dense store;
    // direct readers of `properties` (enumeration, cloning, …) cannot
    // see them.

    /// Whether `arr` is currently using dense element storage (and is an
    /// array). Lets native hot paths (e.g. `#arr_append`) take the
    /// amortized-O(1) [`Heap::dense_push`] instead of the property-map
    /// slow path.
    pub(crate) fn is_dense_array(&self, arr: Gc<Object>) -> bool {
        dense_is_active(self.get(arr))
    }

    /// Allocate a fresh array using dense element storage.
    ///
    /// The array has the `%Array.prototype%` (null when the intrinsics
    /// are not installed yet) and starts empty; there is no stored
    /// `"length"` property — `Heap::get_property` synthesizes it from
    /// the dense length. Fill with [`Heap::dense_push`]/[`Heap::dense_set`]
    /// and finish with [`Heap::materialize_dense`] before exposing the
    /// array to script.
    ///
    /// # Panics
    ///
    /// Panics if called on a handle from a different heap (via
    /// [`Heap::alloc_object`]).
    pub(crate) fn new_dense_array(&mut self) -> Result<Gc<Object>, AllocError> {
        let proto = self
            .array_prototype
            .map(Value::Object)
            .unwrap_or(Value::Null);
        let mut obj = Object {
            kind: ObjectKind::Array,
            prototype: proto,
            ..Default::default()
        };
        obj.set_dense(Some(Vec::new()));
        self.alloc_object(obj)
    }

    /// Current `length` of a dense array; `0` when `arr` is not dense.
    ///
    /// Dense-builder API for the array-builtin hot paths (owned by a
    /// concurrent worker); unused in this crate's own build for now.
    #[allow(dead_code)]
    pub(crate) fn dense_len(&self, arr: Gc<Object>) -> u64 {
        self.get(arr).dense().map_or(0, |d| d.len() as u64)
    }

    /// Read dense index `i`: `None` for holes and out-of-bounds reads
    /// (both read as `undefined`); `None` when `arr` is not dense.
    ///
    /// Dense-builder API for the array-builtin hot paths (owned by a
    /// concurrent worker); unused in this crate's own build for now.
    #[allow(dead_code)]
    pub(crate) fn dense_get(&self, arr: Gc<Object>, i: u64) -> Option<Value> {
        self.get(arr)
            .dense()
            .and_then(|d| d.get(i as usize).copied().flatten())
    }

    /// Presence check for dense index `i`: holes read as absent.
    ///
    /// Dense-builder API for the array-builtin hot paths (owned by a
    /// concurrent worker); unused in this crate's own build for now.
    #[allow(dead_code)]
    pub(crate) fn dense_has(&self, arr: Gc<Object>, i: u64) -> bool {
        self.dense_get(arr, i).is_some()
    }

    /// Append `v` to a dense array: the fast `push`. Amortized O(1); the
    /// dense length (hence `length`) grows by one with no property
    /// machinery involved.
    ///
    /// Errors with `RangeError` past the maximum array length.
    pub(crate) fn dense_push(&mut self, arr: Gc<Object>, v: Value) -> Result<(), NativeError> {
        // A tracked object is never dense-active today, but dense writes
        // are heap writes: note them so a future tracked dense array
        // bumps the epoch instead of silently bypassing the scrub. A
        // dense array can never be the reset global (dense requires
        // `ObjectKind::Array`; the global is `ObjectKind::Ordinary`), so
        // the `reset_global` branch of `note_reset_write` is dead here:
        // skip the call entirely unless something is tracked, avoiding
        // a HashSet probe per element op while a harness holds the
        // global registered.
        if !self.reset_tracked.is_empty() {
            self.note_reset_write(arr, None);
        }
        self.check_value(v);
        let o = self.get_object_mut(arr);
        let Some(dense) = o.dense_mut() else {
            return Err(NativeError::TypeError(
                "dense_push on a non-dense array".into(),
            ));
        };
        if dense.len() as u64 >= DENSE_MAX_LEN {
            return Err(NativeError::RangeError("invalid array length".into()));
        }
        dense.push(Some(v));
        Ok(())
    }

    /// Write dense index `i`, extending with holes when `i` is past the
    /// end (which grows `length` to `i + 1`, per array semantics).
    ///
    /// A gap larger than the hole budget materializes the array to the
    /// property map first (the sparse fallback); the write then lands
    /// there. Errors with `RangeError` for indices past the maximum.
    ///
    /// Dense-builder API for the array-builtin hot paths (owned by a
    /// concurrent worker); unused in this crate's own build for now.
    #[allow(dead_code)]
    pub(crate) fn dense_set(
        &mut self,
        arr: Gc<Object>,
        i: u64,
        v: Value,
    ) -> Result<(), NativeError> {
        // Dense arrays are never the reset global (see `dense_push`); skip
        // the note unless something is reset-tracked.
        if !self.reset_tracked.is_empty() {
            self.note_reset_write(arr, None);
        }
        self.check_value(v);
        if i > DENSE_MAX_LEN {
            return Err(NativeError::RangeError("invalid array length".into()));
        }
        if !dense_is_active(self.get(arr)) {
            return Err(NativeError::TypeError(
                "dense_set on a non-dense array".into(),
            ));
        }
        if !dense_gap_ok(self.get(arr), i + 1) {
            self.materialize_dense(arr)?;
            self.define_property(arr, Wtf8::from_str(&i.to_string()), Property::data(v))
                .map_err(NativeError::from)?;
            // `dense_set` extends `length` past the written index, like
            // `set_idx` does; the ordinary define above does not.
            self.define_property(
                arr,
                Wtf8::from_str("length"),
                Property {
                    value: dense_len_value(i + 1),
                    writable: true,
                    enumerable: false,
                    configurable: false,
                },
            )
            .map_err(NativeError::from)?;
            return Ok(());
        }
        let o = self.get_object_mut(arr);
        let dense = o.dense_mut().expect("dense checked above");
        let i = i as usize;
        if i < dense.len() {
            dense[i] = Some(v);
        } else {
            dense.resize(i + 1, None);
            dense[i] = Some(v);
        }
        Ok(())
    }

    /// Set a dense array's `length`: truncating drops trailing elements,
    /// extending fills holes. A huge extension materializes first (the
    /// sparse fallback). Errors with `RangeError` past the maximum array
    /// length, matching `Array` `length` semantics.
    ///
    /// Dense-builder API for the array-builtin hot paths (owned by a
    /// concurrent worker); unused in this crate's own build for now.
    #[allow(dead_code)]
    pub(crate) fn dense_set_len(&mut self, arr: Gc<Object>, len: u64) -> Result<(), NativeError> {
        // Dense arrays are never the reset global (see `dense_push`); skip
        // the note unless something is reset-tracked.
        if !self.reset_tracked.is_empty() {
            self.note_reset_write(arr, None);
        }
        if len > DENSE_MAX_LEN {
            return Err(NativeError::RangeError("invalid array length".into()));
        }
        if !dense_is_active(self.get(arr)) {
            return Err(NativeError::TypeError(
                "dense_set_len on a non-dense array".into(),
            ));
        }
        if !dense_gap_ok(self.get(arr), len) {
            self.materialize_dense(arr)?;
            return self
                .define_property(
                    arr,
                    Wtf8::from_str("length"),
                    Property {
                        value: dense_len_value(len),
                        writable: true,
                        enumerable: false,
                        configurable: false,
                    },
                )
                .map_err(NativeError::from);
        }
        let o = self.get_object_mut(arr);
        let dense = o.dense_mut().expect("dense checked above");
        dense.resize(len as usize, None);
        Ok(())
    }

    /// Convert dense element storage back to ordinary index properties.
    ///
    /// After this call the array is indistinguishable from one built the
    /// slow way: indexed elements are enumerable data properties,
    /// `"length"` is the usual non-configurable property, and `dense` is
    /// `None`. Named (non-index) properties defined while dense are kept.
    /// This is the spec-correct fallback: call it before a dense array
    /// becomes user-visible.
    ///
    /// # Panics
    ///
    /// Panics if `arr` was created by a different heap.
    pub(crate) fn materialize_dense(&mut self, arr: Gc<Object>) -> Result<(), AllocError> {
        // The inner writer appends index properties directly (not via
        // `define_property`), so it would otherwise bypass the epoch too.
        // Dense arrays are never the reset global (see `dense_push`); skip
        // the note unless something is reset-tracked.
        if !self.reset_tracked.is_empty() {
            self.note_reset_write(arr, None);
        }
        let len = self.get(arr).dense().map_or(0, |d| d.len());
        if self.get(arr).dense().is_none() {
            return Ok(());
        }
        // Root the array across interning (which may collect); the
        // not-yet-materialized values stay reachable through the dense
        // store itself, which the tracer visits. The root is released on
        // every exit path — including allocation failure — so a failed
        // materialization never pins the array alive permanently or grows
        // `temp_roots`.
        let base = self.protect(Value::Object(arr));
        let r = self.materialize_dense_inner(arr, len);
        self.release_temp(base);
        r
    }

    /// [`Heap::materialize_dense`] without the temp root: the caller holds
    /// the array protected, so every `?` below is safe.
    fn materialize_dense_inner(&mut self, arr: Gc<Object>, len: usize) -> Result<(), AllocError> {
        // Transactional: key interning is fallible in principle (a failed
        // `intern` can also collect mid-loop). In practice the dense index
        // keys and "length" all pack inline, so interning cannot fail or
        // collect here; the truncate-on-error below is retained as
        // defense-in-depth. Record the property count up front and
        // truncate back on any failure, so the object keeps `dense: Some`
        // with none of the partially copied index properties: a retry
        // then starts clean instead of appending duplicate keys (whose
        // first, stale copy `get_property`'s linear `find` would keep
        // returning). `dense` is only taken after the last fallible call.
        let props_len = self.get(arr).properties.len();
        // The loop below appends index properties; the shape is rebuilt
        // once at the end. On the error paths the truncate restores the
        // exact prior list, so the shape is re-derived from the lists
        // with `shape_rebuild`: the shape id cannot simply be saved and
        // restored, because `intern_str` may have collected mid-loop and
        // `compact_shapes` renumbers shapes on collection.
        for i in 0..len {
            let v: Option<Value> = self.get(arr).dense().and_then(|d| d[i]);
            if let Some(v) = v {
                let key = match self.intern_str(&i.to_string()) {
                    Ok(key) => key,
                    Err(e) => {
                        self.get_object_mut(arr).properties.truncate(props_len);
                        self.shape_rebuild(arr);
                        return Err(e);
                    }
                };
                // `intern` may have collected; re-derive the object.
                self.get_object_mut(arr)
                    .properties
                    .push((key, Property::data(v)));
            }
        }
        let length_key = match self.intern_str("length") {
            Ok(key) => key,
            Err(e) => {
                self.get_object_mut(arr).properties.truncate(props_len);
                self.shape_rebuild(arr);
                return Err(e);
            }
        };
        {
            let o = self.get_object_mut(arr);
            o.take_dense();
            let prop = Property {
                value: dense_len_value(len as u64),
                writable: true,
                enumerable: false,
                configurable: false,
            };
            match o
                .properties
                .iter_mut()
                .find(|(k, _)| k.as_wtf8() == "length")
            {
                Some(slot) => slot.1 = prop,
                None => o.properties.push((length_key, prop)),
            }
        }
        // Appends (and at most one overwrite of "length", which changes
        // no layout): rebuild the shape from the final lists.
        self.shape_rebuild(arr);
        Ok(())
    }
}

impl Drop for Heap {
    fn drop(&mut self) {
        // A live `HandleScope` holds a raw pointer to this heap and
        // unregisters itself in its own `Drop`; dropping the heap first
        // would leave the scope unregistering against freed memory. Fail
        // loudly instead of corrupting silently.
        assert!(
            self.scopes.is_empty(),
            "Heap dropped while {} HandleScope(s) still registered: drop scopes before the heap",
            self.scopes.len()
        );
        // Reclaim every live allocation. Object slots came from the bump
        // pages: drop each live `Object` in place, then hand each page
        // block back to the OS. (Slots already on a page's freelist were
        // dropped when the sweep reclaimed them; fully-emptied pages were
        // already deallocated by the sweep.) The other registries hold
        // `Box::into_raw`'d allocations, freed by rebuilding the box.
        // (`HandleScope`s must already be dropped per their contract; their
        // root lists are owned by the scopes, not the heap.)
        for &ptr in &self.objects {
            // SAFETY: `ptr` is a live page slot owned by this heap; this
            // `Drop` owns the heap, so it takes unique ownership of every
            // slot exactly once. Dropping in place ends the referent
            // without touching the page block, which `dealloc` below then
            // frees.
            unsafe { std::ptr::drop_in_place(ptr.as_ptr()) };
        }
        for page in std::mem::take(&mut self.object_pages) {
            page.dealloc();
        }
        for &ptr in &self.strings {
            // SAFETY: `ptr` is a live string block owned by this heap; the
            // block is still allocated (only the sweep's reclaimer and
            // this `Drop` free string blocks), so reading the header
            // fields is sound. The fields are read without forming a
            // reference (`addr_of!`).
            let class = unsafe { std::ptr::addr_of!((*ptr.as_ptr()).class).read() };
            if class == STR_CLASS_OVERSIZE {
                let len = unsafe { std::ptr::addr_of!((*ptr.as_ptr()).len).read() as usize };
                // SAFETY: oversize blocks are individually malloc'd with
                // the layout recomputed here; `len` is the header's own
                // record of the payload size, and this `Drop` owns the
                // heap, so it takes unique ownership of every block
                // exactly once.
                unsafe { dealloc_str_oversize(ptr, len) };
            }
            // Chunk-backed blocks die with their chunks below: the payload
            // is raw bytes, so there is nothing to drop in place. (Blocks
            // already on a chunk's freelist were reclaimed by the sweep;
            // fully-emptied chunks were already deallocated by the sweep.)
        }
        for class_chunks in std::mem::take(&mut self.str_chunks) {
            for chunk in class_chunks {
                chunk.dealloc();
            }
        }
        for &ptr in &self.symbols {
            // SAFETY: as above.
            drop(unsafe { Box::from_raw(ptr.as_ptr()) });
        }
        for &ptr in &self.bigints {
            // SAFETY: as above.
            drop(unsafe { Box::from_raw(ptr.as_ptr()) });
        }
    }
}

impl fmt::Debug for Heap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Heap")
            .field("id", &self.id)
            .field("objects", &self.objects.len())
            .field("strings", &self.strings.len())
            .field("symbols", &self.symbols.len())
            .field("bigints", &self.bigints.len())
            .field("scopes", &self.scopes.len())
            .finish()
    }
}

/// True if the bigint is zero (any sign, all-zero digits).
///
/// The handle must be live (see the module docs); there is no heap to
/// validate membership against here.
pub fn bigint_is_zero(b: Gc<BigInt>) -> bool {
    b.digits.iter().all(|&d| d == 0)
}

/// The behavior tag of an object.
///
/// The handle must be live (see the module docs); there is no heap to
/// validate membership against here.
pub fn object_kind(o: Gc<Object>) -> ObjectKind {
    o.kind
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{FunctionData, ObjectKind};

    #[test]
    fn gc_handles_stay_valid_across_allocations() {
        let mut heap = Heap::new();
        // Long enough to heap-allocate: exercises the stable-address path.
        let s0 = heap
            .alloc_str("hello, this string lives on the heap")
            .unwrap();
        // Force several Vec reallocations; arena blocks keep stable
        // addresses. (Automatic collection is off by default, so nothing
        // is freed.)
        for _ in 0..10_000 {
            heap.alloc_str("padding").unwrap();
            heap.alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
        }
        assert_eq!(s0.as_wtf8(), "hello, this string lives on the heap");
        assert!(heap.alloc_count() > 10_000);
    }

    #[test]
    fn intern_dedups_alloc_str_does_not() {
        let mut heap = Heap::new();
        // Long enough to heap-allocate: every call gets a fresh allocation,
        // never the interned canonical handle.
        let long = "a".repeat(64);
        let a = heap.alloc_str(&long).unwrap();
        let b = heap.alloc_str(&long).unwrap();
        let (Some(ga), Some(gb)) = (a.heap_handle(), b.heap_handle()) else {
            panic!("expected heap strings");
        };
        assert_ne!(ga, gb);
        // `intern` returns `Str`: short strings pack inline (content-equal
        // across calls), long ones share the canonical heap handle.
        let i = heap.intern("q").unwrap();
        let j = heap.intern("q").unwrap();
        assert!(i.is_inline() && j.is_inline());
        assert_eq!(i, j);
        // `alloc_str` never returns the interned canonical handle, even
        // for equal contents.
        let k = heap.intern(&long).unwrap();
        assert_ne!(Some(ga), k.heap_handle());
        assert_ne!(Some(gb), k.heap_handle());
        // Short strings pack inline without allocating, compare by
        // content, and never touch the intern table.
        let s1 = heap.alloc_str("q").unwrap();
        let s2 = heap.alloc_str("q").unwrap();
        assert!(s1.is_inline() && s2.is_inline());
        assert_eq!(s1, s2);
        assert_eq!(s1, i);
        assert_eq!(heap.interned.len(), 1, "only the long key is tabled");
        assert!(!heap.interned.contains(Wtf8::from_str("q")));
    }

    #[test]
    fn properties_round_trip() {
        let mut heap = Heap::new();
        let o = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        assert_eq!(heap.get_property(o, Wtf8::from_str("x")), None);
        heap.define_property_str(o, "x", Property::data(Value::Int(7)))
            .unwrap();
        assert_eq!(
            heap.get_property(o, Wtf8::from_str("x")).unwrap().value,
            Value::Int(7)
        );
        heap.define_property(o, Wtf8::from_str("x"), Property::data(Value::Int(8)))
            .unwrap();
        assert_eq!(
            heap.get_property(o, Wtf8::from_str("x")).unwrap().value,
            Value::Int(8)
        );
        assert!(heap.delete_property(o, Wtf8::from_str("x")));
        assert!(!heap.delete_property(o, Wtf8::from_str("x")));
        heap.set_prototype(o, Value::Null);
        assert_eq!(heap.get(o).prototype, Value::Null);
    }

    #[test]
    fn tracer_marks_transitively_and_breaks_cycles() {
        let mut heap = Heap::new();
        let a = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let b = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let s = heap
            .alloc_str("leaf-string-that-lives-on-the-heap")
            .unwrap();
        heap.define_property(a, Wtf8::from_str("b"), Property::data(Value::Object(b)))
            .unwrap();
        heap.define_property(b, Wtf8::from_str("a"), Property::data(Value::Object(a)))
            .unwrap();
        heap.define_property(a, Wtf8::from_str("s"), Property::data(Value::Str(s)))
            .unwrap();
        let mut t = Tracer::new();
        t.mark_value(Value::Object(a));
        // a, b, and the leaf string. The three short keys ("b", "a",
        // "s") pack inline and need no marking.
        assert_eq!(t.marked_count(), 3);
    }

    #[test]
    fn alloc_pressure_triggers_collection() {
        let mut heap = Heap::new();
        heap.set_gc_threshold(64);
        for _ in 0..1000 {
            // Long enough to heap-allocate: short strings inline and
            // would exert no collection pressure at all.
            heap.alloc_str(&"garbage".repeat(8)).unwrap();
        }
        // Without collection we would hold 1000 strings; pressure keeps the
        // heap bounded near the threshold.
        assert!(heap.alloc_count() <= 64);
    }

    #[test]
    fn rooted_objects_survive_collection() {
        let mut heap = Heap::new();
        heap.set_gc_threshold(16);
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
        let root = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let hroot = scope.add(root);
        heap.define_property(root, Wtf8::from_str("x"), Property::data(Value::Int(1)))
            .unwrap();
        // Self-cycle: must not loop the marker, and must survive.
        heap.define_property(
            root,
            Wtf8::from_str("self"),
            Property::data(Value::Object(root)),
        )
        .unwrap();
        // Churn: each iteration allocates past the threshold, collecting.
        for _ in 0..500 {
            heap.alloc_str("trash").unwrap();
            heap.alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
        }
        heap.collect();
        assert_eq!(
            heap.get_property(hroot.get(), Wtf8::from_str("x"))
                .unwrap()
                .value,
            Value::Int(1)
        );
        assert_eq!(
            heap.get_property(hroot.get(), Wtf8::from_str("self"))
                .unwrap()
                .value,
            Value::Object(hroot.get())
        );
        // Survivors: the rooted object. The two short keys ("x", "self")
        // pack inline and allocate nothing.
        assert_eq!(heap.alloc_count(), 1);
        // `Handle` derefs like the underlying object.
        assert_eq!(hroot.properties.len(), 2);
    }

    #[test]
    fn unrooted_objects_die_with_exact_counts() {
        let mut heap = Heap::new();
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
        let keep = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let hkeep = scope.add(keep);
        heap.define_property(keep, Wtf8::from_str("k"), Property::data(Value::Int(1)))
            .unwrap();
        for _ in 0..10 {
            let o = heap
                .alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
            heap.define_property(o, Wtf8::from_str("k"), Property::data(Value::Int(2)))
                .unwrap();
            heap.alloc_str(&"temp".repeat(8)).unwrap(); // unrooted, non-interned
        }
        // 11 objects + 10 temp strings ("k" packs inline: no intern
        // allocation) = 21.
        assert_eq!(heap.alloc_count(), 21);
        heap.collect();
        // Survivors: just `keep` (rooted). Its "k" key is inline, so
        // nothing else is reachable. The 10 unrooted objects and 10 temp
        // strings are freed.
        assert_eq!(heap.alloc_count(), 1);
        assert_eq!(
            heap.get_property(hkeep.get(), Wtf8::from_str("k"))
                .unwrap()
                .value,
            Value::Int(1)
        );
    }

    #[test]
    fn function_name_is_traced() {
        let mut heap = Heap::new();
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
        // `name` is not rooted directly; it must survive via the Function kind.
        // Long enough to stay heap-allocated: an inline name would make the
        // test vacuous.
        let name = heap.intern_str("myFuncWithAVeryLongName").unwrap();
        assert!(!name.is_inline());
        let f = heap
            .alloc_object(Object::new(ObjectKind::Function(FunctionData {
                bytecode_id: 0,
                name,
                arity: 0,
            })))
            .unwrap();
        scope.add(f);
        heap.alloc_str("unrelated-garbage").unwrap();
        heap.collect();
        assert_eq!(name.as_wtf8(), "myFuncWithAVeryLongName");
    }

    #[test]
    fn nested_scopes_unroot_on_drop() {
        let mut heap = Heap::new();
        let outer = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
        let a = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let ha = outer.add(a);
        {
            let inner = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
            let b = heap
                .alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
            inner.add(b);
            heap.collect();
            assert_eq!(heap.alloc_count(), 2);
        } // `inner` drops: `b` becomes unrooted.
        heap.collect();
        assert_eq!(heap.alloc_count(), 1);
        assert!(heap.get(ha.get()).properties.is_empty());
    }

    #[test]
    fn intrinsics_are_roots() {
        let mut heap = Heap::new();
        let proto = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.object_prototype = Some(proto);
        heap.alloc_str("garbage").unwrap();
        heap.collect();
        // The prototype survives via the heap field even with no scope.
        assert_eq!(heap.alloc_count(), 1);
        assert_eq!(heap.get(proto).kind, ObjectKind::Ordinary);
    }

    #[test]
    fn heap_ids_are_unique_per_heap() {
        let h1 = Heap::new();
        let h2 = Heap::new();
        assert_ne!(h1.id(), 0);
        assert_ne!(h1.id(), h2.id());
    }

    #[test]
    fn handles_carry_their_heaps_id() {
        let mut heap = Heap::new();
        // Short strings are inline: no heap handle at all.
        let s = heap.alloc_str("x").unwrap();
        assert!(s.is_inline());
        assert!(s.heap_handle().is_none());
        // Long strings are heap-backed and carry the heap's id.
        let long = heap.alloc_str(&"x".repeat(32)).unwrap();
        assert_eq!(long.heap_handle().unwrap().heap_id(), heap.id());
        let o = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        assert_eq!(o.heap_id(), heap.id());
    }

    #[test]
    fn object_pages_reuse_freed_slots_and_free_empty_pages() {
        let mut heap = Heap::new();
        // Two pages' worth of objects (128 slots/page): the first page
        // also holds the survivor below.
        let mut objs = Vec::new();
        for _ in 0..200 {
            objs.push(
                heap.alloc_object(Object::new(ObjectKind::Ordinary))
                    .unwrap(),
            );
        }
        assert!(
            heap.object_pages.len() >= 2,
            "expected multiple object pages"
        );
        let survivor = objs[0];
        let freed_addrs: HashSet<usize> =
            objs[1..128].iter().map(|g| g.as_ptr() as usize).collect();
        // Root the survivor so its page survives the collection; every
        // other object is garbage.
        let base = heap.protect(Value::Object(survivor));
        drop(objs);
        heap.collect();
        // The all-garbage second page is returned to the OS; the first
        // page survives with one live slot.
        assert_eq!(heap.object_pages.len(), 1, "empty pages must be freed");
        assert_eq!(heap.alloc_count(), 1);
        // The next allocation must reuse a freed slot from the surviving
        // page's freelist, not grow the heap.
        let g = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        assert_eq!(
            heap.object_pages.len(),
            1,
            "freelist reuse must not grow pages"
        );
        assert!(
            freed_addrs.contains(&(g.as_ptr() as usize)),
            "reallocated object must reuse a freed slot"
        );
        heap.release_temp(base);
    }

    #[test]
    fn object_page_growth_honors_fail_alloc() {
        let mut heap = Heap::new();
        set_fail_alloc(true);
        let _guard = FailAllocGuard;
        // No pages yet: the first object allocation must grow a page,
        // which the injection fails.
        let err = match heap.alloc_object(Object::new(ObjectKind::Ordinary)) {
            Ok(_) => panic!("page growth must fail under FAIL_ALLOC"),
            Err(e) => e,
        };
        assert_eq!(err, AllocError);
        assert!(heap.object_pages.is_empty());
        drop(_guard);
        // And the heap still works afterwards.
        heap.alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        assert_eq!(heap.object_pages.len(), 1);
    }

    #[test]
    fn str_arena_coalesces_header_and_payload() {
        let mut heap = Heap::new();
        // 40 bytes: 8 (header) + 40 = 48 fits class 0 (64).
        let long = "s".repeat(40);
        let g = heap
            .alloc_str(&long)
            .unwrap()
            .heap_handle()
            .expect("long string is heap-backed");
        let header = g.as_ptr() as usize;
        // The payload immediately follows the header inside one chunk
        // block: header + payload are contiguous, not a header box plus
        // a separate byte allocation.
        let payload_addr = header + GcStr::HEADER_LEN;
        // SAFETY: the block was just allocated by `alloc_str` and is
        // live; header+40 payload bytes are in-bounds.
        let payload = unsafe { std::slice::from_raw_parts(payload_addr as *const u8, 40) };
        assert_eq!(payload, long.as_bytes());
        assert_eq!(header % GcStr::HEADER_ALIGN, 0);
        let chunk_base = heap.str_chunks[0][0].base.as_ptr() as usize;
        assert!(
            header >= chunk_base && header + STR_CLASS_SIZES[0] <= chunk_base + STR_CHUNK_BYTES,
            "block must lie inside its chunk"
        );
        assert_eq!(g.len_bytes(), 40);
        assert_eq!(g.as_wtf8(), Wtf8::from_str(&long));
        // Header bookkeeping matches the payload size.
        // SAFETY: reading the header of a live arena block allocated by
        // this test.
        let class = unsafe { std::ptr::addr_of!((*g.as_ptr()).class).read() };
        assert_eq!(class, str_class_for(40));
        assert_eq!(class, 0);
        assert_eq!(heap.str_chunks[0].len(), 1);
        // A payload spilling past a class boundary lands in the next one:
        // 8 + 100 = 108 fits class 1 (128), not class 0 (64).
        let big = "b".repeat(100);
        let gb = heap
            .alloc_str(&big)
            .unwrap()
            .heap_handle()
            .expect("long string is heap-backed");
        // SAFETY: reading the header of a live arena block allocated by
        // this test.
        let class_b = unsafe { std::ptr::addr_of!((*gb.as_ptr()).class).read() };
        assert_eq!(class_b, 1);
        assert_eq!(gb.as_wtf8(), Wtf8::from_str(&big));
    }

    #[test]
    fn str_arena_reuses_freed_blocks() {
        let mut heap = Heap::new();
        let a = heap
            .alloc_str(&"a".repeat(40))
            .unwrap()
            .heap_handle()
            .unwrap();
        let addr = a.as_ptr() as usize;
        // `a` is unrooted, so the collection frees its block onto the
        // chunk's freelist.
        heap.collect();
        assert_eq!(heap.alloc_count(), 0);
        let b = heap
            .alloc_str(&"b".repeat(40))
            .unwrap()
            .heap_handle()
            .unwrap();
        assert_eq!(
            b.as_ptr() as usize,
            addr,
            "reallocated string must reuse the freed block"
        );
        assert_eq!(
            heap.str_chunks[0].len(),
            1,
            "freelist reuse must not grow chunks"
        );
        assert_eq!(b.as_wtf8(), Wtf8::from_str(&"b".repeat(40)));
    }

    #[test]
    fn str_arena_returns_empty_chunks_to_os() {
        let mut heap = Heap::new();
        // 600 class-0 blocks of 64 bytes = 38_400 bytes: more than two
        // 16 KiB chunks. All unrooted, so the collection below frees
        // every block.
        for i in 0..600 {
            let s = format!("chunk-fill-{i:04}") + &"x".repeat(40);
            assert!(s.len() > crate::value::STR_INLINE_MAX);
            heap.alloc_str(&s).unwrap();
        }
        assert!(
            heap.str_chunks[0].len() >= 2,
            "expected multiple string chunks"
        );
        heap.collect();
        assert_eq!(heap.alloc_count(), 0);
        assert!(
            heap.str_chunks.iter().all(|v| v.is_empty()),
            "empty chunks must be returned to the OS"
        );
        // And the arena still works afterwards.
        let g = heap.alloc_str(&"after-empty".repeat(4)).unwrap();
        assert_eq!(g.as_wtf8(), Wtf8::from_str(&"after-empty".repeat(4)));
    }

    #[test]
    fn str_chunk_growth_honors_fail_alloc() {
        let mut heap = Heap::new();
        set_fail_alloc(true);
        let _guard = FailAllocGuard;
        // No chunks yet: the first heap string must grow a chunk, which
        // the injection fails.
        let err = match heap.alloc_str(&"f".repeat(64)) {
            Ok(_) => panic!("chunk growth must fail under FAIL_ALLOC"),
            Err(e) => e,
        };
        assert_eq!(err, AllocError);
        assert!(heap.str_chunks.iter().all(|v| v.is_empty()));
        assert_eq!(heap.alloc_count(), 0);
        drop(_guard);
        // And the heap still works afterwards.
        let g = heap.alloc_str(&"f".repeat(64)).unwrap();
        assert_eq!(g.as_wtf8(), Wtf8::from_str(&"f".repeat(64)));
    }

    #[test]
    fn str_arena_oversize_blocks_are_freed() {
        let mut heap = Heap::new();
        // 8 + 3000 > 2048: above the top class, takes the malloc fallback.
        let big = "o".repeat(3000);
        let g = heap
            .alloc_str(&big)
            .unwrap()
            .heap_handle()
            .expect("long string is heap-backed");
        // SAFETY: reading the header of a live arena block allocated by
        // this test.
        let class = unsafe { std::ptr::addr_of!((*g.as_ptr()).class).read() };
        assert_eq!(class, STR_CLASS_OVERSIZE);
        assert_eq!(g.as_wtf8(), Wtf8::from_str(&big));
        assert_eq!(heap.alloc_count(), 1);
        // Rooted: the collection keeps the oversize block alive.
        let base = heap.protect(Value::Str(Str::heap(g)));
        heap.collect();
        assert_eq!(heap.alloc_count(), 1);
        heap.release_temp(base);
        // Unrooted: freed through the oversize path, no chunk involved.
        heap.collect();
        assert_eq!(heap.alloc_count(), 0);
        assert!(heap.str_chunks.iter().all(|v| v.is_empty()));
    }

    #[test]
    fn str_class_for_matches_table_at_boundaries() {
        // Exhaustive around every class edge: header (8) + payload.
        for len in 0..=2100usize {
            let need = GcStr::HEADER_LEN + len;
            let mut expected = STR_CLASS_OVERSIZE;
            for (i, &size) in STR_CLASS_SIZES.iter().enumerate() {
                if need <= size {
                    expected = i as u8;
                    break;
                }
            }
            assert_eq!(str_class_for(len), expected, "len={len}");
        }
        assert_eq!(str_class_for(usize::MAX), STR_CLASS_OVERSIZE);
    }

    #[test]
    fn sweep_accounts_live_bytes_per_arena() {
        let mut heap = Heap::new();
        assert_eq!(heap.live_bytes(), 0, "zero before the first collection");
        assert_eq!(heap.arena_bytes().total(), 0);
        // Live set: 3 objects, 2 heap strings in different size classes
        // (40-byte payload -> class 0, 200-byte payload -> class 2), 1
        // oversize string (header + payload, no class), 1 symbol, 1
        // bigint. One garbage object and one garbage string prove dead
        // blocks are excluded from the accounting.
        let first = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let base = heap.protect(Value::Object(first));
        for _ in 0..2 {
            let g = heap
                .alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
            heap.protect(Value::Object(g));
        }
        let s0 = heap.alloc_str(&"x".repeat(40)).unwrap();
        heap.protect(Value::Str(s0));
        let s2 = heap.alloc_str(&"y".repeat(200)).unwrap();
        heap.protect(Value::Str(s2));
        let big = heap.alloc_str(&"z".repeat(3000)).unwrap();
        heap.protect(Value::Str(big));
        let sym = heap.alloc_symbol(None).unwrap();
        heap.protect(Value::Symbol(sym));
        let bi = heap.alloc_bigint(false, vec![1, 2, 3]).unwrap();
        heap.protect(Value::BigInt(bi));
        // Garbage: unrooted, freed by the collection below.
        heap.alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.alloc_str(&"w".repeat(40)).unwrap();
        heap.collect();
        let b = heap.arena_bytes();
        assert_eq!(b.objects, 3 * std::mem::size_of::<Object>());
        assert_eq!(
            b.strings,
            STR_CLASS_SIZES[0] + STR_CLASS_SIZES[2] + GcStr::HEADER_LEN + 3000
        );
        assert_eq!(b.symbols, std::mem::size_of::<Symbol>());
        assert_eq!(b.bigints, std::mem::size_of::<BigInt>());
        assert_eq!(heap.live_bytes(), b.total());
        assert_eq!(heap.alloc_count(), 8, "garbage must be swept");
        heap.release_temp(base);
        // Unrooted: everything is freed, accounting returns to zero.
        heap.collect();
        assert_eq!(heap.live_bytes(), 0);
        assert_eq!(heap.alloc_count(), 0);
    }

    #[test]
    fn alloc_wtf8_owned_honors_fail_alloc() {
        let mut heap = Heap::new();
        set_fail_alloc(true);
        let _guard = FailAllocGuard;
        // The arena block allocation fails (chunk growth refuses): the
        // buffer is dropped, nothing is registered.
        let err = heap
            .alloc_wtf8_owned(Wtf8Buf::from("f".repeat(64).as_str()))
            .unwrap_err();
        assert_eq!(err, AllocError);
        assert_eq!(heap.alloc_count(), 0);
    }

    #[test]
    fn concat_str_heap_path_folds_boundary_pair() {
        let mut heap = Heap::new();
        // 20-byte operands with a lone lead at the end of `a` and a lone
        // trail at the start of `b`: total 40 > STR_INLINE_MAX + 2 takes
        // the heap path, and maximal pairing folds the 3+3 boundary into
        // one 4-byte astral char (38 bytes out).
        let a_units: Vec<u16> = std::iter::repeat_n(0x78, 17)
            .chain(std::iter::once(0xD800))
            .collect();
        let b_units: Vec<u16> = std::iter::once(0xDC00)
            .chain(std::iter::repeat_n(0x79, 17))
            .collect();
        let a = heap
            .alloc_wtf8_owned(Wtf8Buf::from_units(&a_units))
            .unwrap();
        let b = heap
            .alloc_wtf8_owned(Wtf8Buf::from_units(&b_units))
            .unwrap();
        assert!(!a.is_inline() && !b.is_inline());
        let c = heap.concat_str(a, b).unwrap();
        assert!(!c.is_inline());
        let mut expected_units = a_units;
        expected_units.extend_from_slice(&b_units);
        let expected = Wtf8Buf::from_units(&expected_units);
        assert_eq!(c.as_wtf8().as_bytes(), expected.as_bytes());
        assert_eq!(c.as_wtf8().len(), 38);
    }

    #[test]
    #[should_panic(expected = "cross-heap")]
    fn cross_heap_mutation_is_rejected() {
        let mut a = Heap::new();
        let mut b = Heap::new();
        let foreign = a.alloc_str(&"x".repeat(64)).unwrap();
        let o = b.alloc_object(Object::new(ObjectKind::Ordinary)).unwrap();
        // `foreign` belongs to `a`; storing it into `b`'s object must panic
        // instead of silently corrupting either heap.
        b.define_property(o, Wtf8::from_str("k"), Property::data(Value::Str(foreign)))
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "cross-heap")]
    fn cross_heap_object_is_rejected() {
        let mut a = Heap::new();
        let b = Heap::new();
        let foreign = a.alloc_object(Object::new(ObjectKind::Ordinary)).unwrap();
        // `foreign` belongs to `a`; reading it through `b` must panic.
        let _ = b.get(foreign);
    }

    #[test]
    #[should_panic(expected = "cross-heap")]
    fn scope_rejects_foreign_handles() {
        let mut a = Heap::new();
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(a)) };
        // `foreign` belongs to a different heap; rooting it would make
        // `a`'s collector dereference memory it does not own.
        let foreign = {
            let mut b = Heap::new();
            // Heap-allocated so there is a handle to reject; inline
            // strings carry no heap identity and need no rooting.
            b.alloc_str(&"y".repeat(64)).unwrap().heap_handle().unwrap()
        };
        scope.add_value(Value::Str(Str::heap(foreign)));
    }

    #[test]
    #[should_panic(expected = "HandleScope(s) still registered")]
    fn heap_drop_with_live_scope_panics() {
        let mut heap = Heap::new();
        // The scope registers a raw pointer to the heap; dropping the heap
        // while the scope is live would leave the scope's `Drop`
        // unregistering against freed memory. The heap must refuse.
        let _scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
        // `_scope` is still live here: dropping `heap` must panic, not
        // silently leave a dangling registration.
        drop(heap);
    }

    #[test]
    fn interned_strings_are_weak_roots() {
        let mut heap = Heap::new();
        // Long enough to heap-allocate: only heap-backed interning goes
        // through the weak table (short strings pack inline and need no
        // entry). No handle is kept: the interned string is unreferenced.
        let long = "e".repeat(64);
        let _ = heap.intern(&long).unwrap();
        assert_eq!(heap.alloc_count(), 1);
        heap.collect();
        // The unreferenced interned string was freed and evicted from the
        // table: without weak interning it would leak forever.
        assert_eq!(heap.alloc_count(), 0);
        // Re-interning allocates fresh instead of resurrecting the dead
        // handle.
        let g2 = heap.intern(&long).unwrap();
        assert_eq!(g2.as_wtf8(), Wtf8::from_str(&long));
        assert_eq!(heap.alloc_count(), 1);
    }

    #[test]
    fn live_interned_strings_survive_and_stay_canonical() {
        let mut heap = Heap::new();
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
        // Long enough to heap-allocate: short strings pack inline and
        // never enter the weak table.
        let long = "l".repeat(64);
        let g1 = heap.intern(&long).unwrap();
        scope.add_value(Value::Str(Str::heap(
            g1.heap_handle().expect("long string is heap-backed"),
        )));
        heap.collect();
        // Still reachable, so still canonical: same handle, no re-alloc.
        let g2 = heap.intern(&long).unwrap();
        assert_eq!(g1, g2);
        assert_eq!(heap.alloc_count(), 1);
    }

    #[test]
    fn interned_text_is_stored_once() {
        let mut heap = Heap::new();
        // Long enough to heap-allocate: short strings pack inline and
        // never enter the table.
        let long_a = "a".repeat(64);
        let long_b = "b".repeat(64);
        let a = heap.intern(&long_a).unwrap();
        let b = heap.intern(&long_a).unwrap();
        // Same canonical handle (pointer-compared) for the same contents.
        assert_eq!(a, b);
        assert_eq!(a.heap_handle(), b.heap_handle());
        // One table entry: the text lives only in the `GcStr` allocation,
        // not duplicated as a hash key.
        assert_eq!(heap.interned.len(), 1);
        assert_eq!(a.as_wtf8(), Wtf8::from_str(&long_a));
        // Distinct contents get distinct handles and their own entries.
        let c = heap.intern(&long_b).unwrap();
        assert_ne!(a, c);
        assert_eq!(heap.interned.len(), 2);
        // Canonicality survives a collection: the rooted string stays
        // live and `intern` returns the same handle, no re-allocation.
        // SAFETY: `heap` is a live local, `addr_of_mut!` yields a valid
        // non-dangling pointer, and `scope` is dropped before `heap`.
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
        scope.add_value(Value::Str(Str::heap(
            a.heap_handle().expect("long string is heap-backed"),
        )));
        heap.collect();
        let d = heap.intern(&long_a).unwrap();
        assert_eq!(a, d);
        assert_eq!(heap.alloc_count(), 1);
    }

    #[test]
    fn repeated_collections_keep_live_objects() {
        let mut heap = Heap::new();
        heap.set_gc_threshold(8);
        // SAFETY: `heap` is a live local, `addr_of_mut!` yields a valid
        // non-dangling pointer, and `scope` is dropped before `heap`.
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
        let root = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let hroot = scope.add(root);
        let child = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.define_property(
            hroot.get(),
            Wtf8::from_str("child"),
            Property::data(Value::Object(child)),
        )
        .unwrap();
        let leaf = heap.alloc_str("leaf-string-kept-on-the-heap!").unwrap();
        heap.define_property(
            child,
            Wtf8::from_str("leaf"),
            Property::data(Value::Str(leaf)),
        )
        .unwrap();
        // Allocate, collect, allocate, collect: the tracer's reused
        // worklist must stay coherent across collections (no stale
        // entries resurrecting garbage, no lost marks freeing the live).
        for _ in 0..10 {
            for _ in 0..20 {
                heap.alloc_str("garbage").unwrap();
            }
            heap.collect();
            let child_back = heap
                .get_property(hroot.get(), Wtf8::from_str("child"))
                .unwrap()
                .value;
            assert_eq!(child_back, Value::Object(child));
            let leaf_back = heap
                .get_property(child, Wtf8::from_str("leaf"))
                .unwrap()
                .value;
            assert_eq!(leaf_back, Value::Str(leaf));
        }
        heap.collect();
        // Exactly the live set survives: root, child, and the leaf
        // string. The two short keys ("child", "leaf") pack inline and
        // allocate nothing.
        assert_eq!(heap.alloc_count(), 3);
        assert_eq!(
            heap.get_property(hroot.get(), Wtf8::from_str("child"))
                .unwrap()
                .value,
            Value::Object(child)
        );
    }

    #[test]
    fn threshold_tracks_live_set_after_collect() {
        let mut heap = Heap::new();
        heap.set_gc_threshold(64);
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
        for _ in 0..100 {
            let o = heap
                .alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
            scope.add(o);
        }
        heap.collect();
        // 100 live rooted objects: the threshold advanced to at least twice
        // the live set, so 90 more allocations run without a collection.
        for _ in 0..90 {
            heap.alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
        }
        assert_eq!(heap.alloc_count(), 190);
    }

    #[test]
    fn marker_handles_deep_chains_iteratively() {
        let mut heap = Heap::new();
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
        // A 100k-long "next" chain: a recursive marker would overflow the
        // call stack expanding this.
        let mut prev: Option<Gc<Object>> = None;
        for _ in 0..100_000 {
            let o = heap
                .alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
            if let Some(p) = prev {
                heap.define_property(o, Wtf8::from_str("next"), Property::data(Value::Object(p)))
                    .unwrap();
            }
            prev = Some(o);
        }
        let head = scope.add(prev.unwrap());
        heap.collect();
        // Walk the whole chain: everything survived, in order.
        let mut count = 0;
        let mut cur = head.get();
        loop {
            count += 1;
            match heap.get_property(cur, Wtf8::from_str("next")) {
                Some(p) => match p.value {
                    Value::Object(o) => cur = o,
                    _ => panic!("non-object next link"),
                },
                None => break,
            }
        }
        assert_eq!(count, 100_000);
        // 100k objects. The "next" key packs inline: no intern allocation.
        assert_eq!(heap.alloc_count(), 100_000);
    }

    /// Heavy-allocation stress: builds a large mixed live graph under a
    /// tight threshold (forcing many collections mid-build), then asserts
    /// exact survivor counts (no leaks, no over-freeing) and walks the
    /// whole graph verifying every value (no mis-marking).
    #[test]
    fn gc_stress_heavy_alloc_forced_collections() {
        use std::time::Instant;
        let t0 = Instant::now();

        const N_ORD: usize = 10_000; // ordinary objects in a linked list
        const N_FN: usize = 10; // function objects w/ self-cycles (every 1000th)

        let mut heap = Heap::new();
        heap.set_gc_threshold(1024);
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };

        // One prototype shared by every ordinary object: hammers the
        // duplicate-mark path (the same address is marked N_ORD times).
        let proto = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let _hproto = scope.add(proto);

        let mut roots: Vec<_> = Vec::with_capacity(N_ORD);
        let mut fn_roots: Vec<_> = Vec::with_capacity(N_FN);
        let mut prev: Option<Gc<Object>> = None;
        for i in 0..N_ORD {
            let o = heap
                .alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
            heap.set_prototype(o, Value::Object(proto));
            roots.push(scope.add(o));
            heap.define_property(
                o,
                Wtf8::from_str("idx"),
                Property::data(Value::Int(i as i32)),
            )
            .unwrap();
            // Zero-padded past the inline limit so payloads stay heap
            // strings: this test counts heap allocations exactly, and
            // short payloads would pack inline and not allocate.
            let payload = heap.alloc_str(&format!("payload-{i:020}")).unwrap();
            heap.define_property(
                o,
                Wtf8::from_str("payload"),
                Property::data(Value::Str(payload)),
            )
            .unwrap();
            let sym = heap.alloc_symbol(Some(Wtf8::from_str("tag"))).unwrap();
            heap.define_property(o, Wtf8::from_str("tag"), Property::data(Value::Symbol(sym)))
                .unwrap();
            let bi = heap.alloc_bigint(false, vec![i as u32]).unwrap();
            heap.define_property(o, Wtf8::from_str("big"), Property::data(Value::BigInt(bi)))
                .unwrap();
            if let Some(p) = prev {
                heap.define_property(o, Wtf8::from_str("next"), Property::data(Value::Object(p)))
                    .unwrap();
            }
            prev = Some(o);
            if i % 1000 == 0 {
                // Function-kind object: exercises the kind-payload trace
                // path (name) plus a self-cycle. Short names pack inline in
                // the `Str`; `FunctionData.name` needs no heap allocation.
                let name = heap.intern_str(&format!("cb-{i}")).unwrap();
                let f = heap
                    .alloc_object(Object::new(ObjectKind::Function(FunctionData {
                        bytecode_id: 0,
                        name,
                        arity: 0,
                    })))
                    .unwrap();
                heap.define_property(f, Wtf8::from_str("me"), Property::data(Value::Object(f)))
                    .unwrap();
                fn_roots.push(scope.add(f));
            }
            // Unrooted churn: must all die.
            for _ in 0..2 {
                heap.alloc_object(Object::new(ObjectKind::Ordinary))
                    .unwrap();
                heap.alloc_str("churn").unwrap();
            }
        }
        let t_build = t0.elapsed();

        let t1 = Instant::now();
        heap.collect();
        let t_collect = t1.elapsed();

        // Exact live set: no leaks (churn freed), nothing live mis-freed.
        // objects: N_ORD ordinary + 1 shared proto + N_FN functions
        // strings: N_ORD payloads (function names and all property keys
        //          pack inline and allocate nothing)
        // symbols: N_ORD, bigints: N_ORD
        let expect = (N_ORD + 1 + N_FN) + N_ORD + N_ORD + N_ORD;
        assert_eq!(heap.alloc_count(), expect);
        println!("gc_stress: build={t_build:?} final_collect={t_collect:?} live={expect}");

        // Walk the whole list; every value must be intact (no mis-marking).
        let mut cur = roots[N_ORD - 1].get();
        for expect_i in (0..N_ORD).rev() {
            assert_eq!(
                heap.get_property(cur, Wtf8::from_str("idx")).unwrap().value,
                Value::Int(expect_i as i32)
            );
            match heap
                .get_property(cur, Wtf8::from_str("payload"))
                .unwrap()
                .value
            {
                Value::Str(s) => assert_eq!(
                    s.as_wtf8(),
                    Wtf8::from_str(&format!("payload-{expect_i:020}"))
                ),
                v => panic!("payload mis-marked: {v:?}"),
            }
            match heap.get_property(cur, Wtf8::from_str("tag")).unwrap().value {
                Value::Symbol(sym) => {
                    assert_eq!(
                        heap.get(sym).description.as_deref(),
                        Some(Wtf8::from_str("tag"))
                    )
                }
                v => panic!("tag mis-marked: {v:?}"),
            }
            match heap.get_property(cur, Wtf8::from_str("big")).unwrap().value {
                Value::BigInt(bi) => {
                    let b = heap.get(bi);
                    assert!(!b.negative);
                    assert_eq!(b.digits, vec![expect_i as u32]);
                }
                v => panic!("big mis-marked: {v:?}"),
            }
            // Prototype link survived.
            assert_eq!(heap.get(cur).prototype, Value::Object(proto));
            match heap.get_property(cur, Wtf8::from_str("next")) {
                Some(p) => match p.value {
                    Value::Object(o) => cur = o,
                    v => panic!("next mis-marked: {v:?}"),
                },
                None => assert_eq!(expect_i, 0),
            }
        }
        // Function objects: self-cycle intact, name traced via kind.
        for (k, hf) in fn_roots.iter().enumerate() {
            let f = hf.get();
            assert_eq!(
                heap.get_property(f, Wtf8::from_str("me")).unwrap().value,
                Value::Object(f)
            );
            match heap.get(f).kind {
                ObjectKind::Function(data) => {
                    assert_eq!(
                        data.name.as_wtf8(),
                        Wtf8::from_str(&format!("cb-{}", k * 1000))
                    );
                }
                _ => panic!("function kind lost"),
            }
        }
    }

    /// `MarkedSet` semantics: exact membership, duplicate inserts reported
    /// as already present, `len` counts distinct addresses only.
    #[test]
    fn marked_set_insert_contains_dedup() {
        let mut s = MarkedSet::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);

        assert!(s.insert(0x1000));
        assert!(s.insert(0x2000));
        assert!(!s.is_empty());
        assert_eq!(s.len(), 2);

        // Duplicates report already-present and do not grow the set.
        assert!(!s.insert(0x1000));
        assert!(!s.insert(0x2000));
        assert_eq!(s.len(), 2);

        assert!(s.contains(0x1000));
        assert!(s.contains(0x2000));
        assert!(!s.contains(0x3000));
        assert!(!s.contains(0));

        s.clear();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert!(!s.contains(0x1000));
        // Usable again after clear.
        assert!(s.insert(0x1000));
        assert!(s.contains(0x1000));
    }

    /// `MarkedSet` stays exact across growth: every inserted address is
    /// found afterwards, and lookups of never-inserted addresses fail —
    /// including across rehash boundaries and reserve().
    #[test]
    fn marked_set_growth_and_rehash_exact() {
        // Deterministic pseudo-random addresses (splitmix64 of a counter),
        // aligned like real box allocations.
        fn addr(i: usize) -> usize {
            (hash_addr(i.wrapping_mul(0x100000001B3)) & !0x7) as usize
        }

        let mut s = MarkedSet::new();
        s.reserve(500);
        for i in 0..2000 {
            assert!(s.insert(addr(i)), "insert {i} lost during growth");
        }
        assert_eq!(s.len(), 2000);
        for i in 0..2000 {
            assert!(s.contains(addr(i)), "contains {i} missed after rehash");
            assert!(!s.insert(addr(i)), "dedup {i} failed after rehash");
        }
        // Never-inserted neighbors must not false-positive. Bit 0 is set
        // on every probe but clear on every stored address (`addr` masks
        // the low 3 bits), so no probe can equal a member by construction.
        for i in 0..2000 {
            assert!(!s.contains(addr(i) | 0x1));
        }
        assert_eq!(s.len(), 2000);

        // reserve() smaller than capacity never shrinks or corrupts.
        s.reserve(4);
        for i in 0..2000 {
            assert!(s.contains(addr(i)));
        }
        assert_eq!(s.len(), 2000);
    }

    /// Mark-phase integration: `Tracer` dedups on the duplicate-heavy path
    /// (one address marked N times counts once) and `marked_count` agrees
    /// with the swept survivor count.
    #[test]
    fn tracer_marked_set_dedups_shared_proto() {
        let mut heap = Heap::new();
        heap.set_gc_threshold(usize::MAX);
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };

        let proto = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let _hp = scope.add(proto);
        const N: usize = 500;
        for _ in 0..N {
            let o = heap
                .alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
            heap.set_prototype(o, Value::Object(proto));
            scope.add(o);
        }
        heap.collect();
        // N objects + the one shared prototype; the prototype's N duplicate
        // marks must not leak the set or the sweep.
        assert_eq!(heap.alloc_count(), N + 1);
    }

    // ------------------------------------------------------------------
    // Dense array element storage
    // ------------------------------------------------------------------

    #[test]
    fn dense_index_key_parsing() {
        assert_eq!(dense_index_key(Wtf8::from_str("0")), Some(0));
        assert_eq!(dense_index_key(Wtf8::from_str("7")), Some(7));
        assert_eq!(
            dense_index_key(Wtf8::from_str("4294967294")),
            Some(4294967294)
        );
        // Not canonical indices.
        assert_eq!(dense_index_key(Wtf8::from_str("")), None);
        assert_eq!(dense_index_key(Wtf8::from_str("00")), None);
        assert_eq!(dense_index_key(Wtf8::from_str("01")), None);
        assert_eq!(dense_index_key(Wtf8::from_str("4294967295")), None);
        assert_eq!(dense_index_key(Wtf8::from_str("99999999999")), None);
        assert_eq!(dense_index_key(Wtf8::from_str("length")), None);
        assert_eq!(dense_index_key(Wtf8::from_str("-1")), None);
        assert_eq!(dense_index_key(Wtf8::from_str("1.5")), None);
        assert_eq!(dense_index_key(Wtf8::from_str("1e3")), None);
    }

    #[test]
    fn dense_push_get_len() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        assert_eq!(heap.get(arr).kind, ObjectKind::Array);
        assert_eq!(heap.dense_len(arr), 0);
        assert_eq!(heap.dense_get(arr, 0), None);
        assert!(!heap.dense_has(arr, 0));

        heap.dense_push(arr, Value::Int(10)).unwrap();
        heap.dense_push(arr, Value::Int(20)).unwrap();
        assert_eq!(heap.dense_len(arr), 2);
        assert_eq!(heap.dense_get(arr, 0), Some(Value::Int(10)));
        assert_eq!(heap.dense_get(arr, 1), Some(Value::Int(20)));
        assert_eq!(heap.dense_get(arr, 2), None);
        assert!(heap.dense_has(arr, 1));
        assert!(!heap.dense_has(arr, 2));

        // `length` is synthesized while dense.
        let len_prop = heap.get_property(arr, Wtf8::from_str("length")).unwrap();
        assert_eq!(len_prop.value, Value::Int(2));
        assert!(len_prop.writable && !len_prop.enumerable && !len_prop.configurable);
        // Indices read through the ordinary property path too.
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("1")).unwrap().value,
            Value::Int(20)
        );
        assert_eq!(heap.get_property(arr, Wtf8::from_str("2")), None);
    }

    #[test]
    fn dense_set_holes_and_truncate() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        heap.dense_set(arr, 0, Value::Int(1)).unwrap();
        heap.dense_set(arr, 3, Value::Int(4)).unwrap();
        assert_eq!(heap.dense_len(arr), 4);
        // Gap filled with holes.
        assert_eq!(heap.dense_get(arr, 1), None);
        assert_eq!(heap.dense_get(arr, 2), None);
        assert!(!heap.dense_has(arr, 2));
        assert_eq!(heap.get_property(arr, Wtf8::from_str("2")), None);

        // Overwrite in place.
        heap.dense_set(arr, 1, Value::Int(2)).unwrap();
        assert_eq!(heap.dense_get(arr, 1), Some(Value::Int(2)));
        assert_eq!(heap.dense_len(arr), 4);

        // Truncation drops trailing elements (spec `length` write).
        heap.dense_set_len(arr, 2).unwrap();
        assert_eq!(heap.dense_len(arr), 2);
        assert_eq!(heap.dense_get(arr, 3), None);
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("length"))
                .unwrap()
                .value,
            Value::Int(2)
        );

        // Extension fills holes.
        heap.dense_set_len(arr, 4).unwrap();
        assert_eq!(heap.dense_len(arr), 4);
        assert_eq!(heap.dense_get(arr, 3), None);
        assert!(!heap.dense_has(arr, 3));

        // Past the maximum length is a RangeError.
        assert!(matches!(
            heap.dense_set_len(arr, 4294967295),
            Err(NativeError::RangeError(_))
        ));
        assert!(matches!(
            heap.dense_set(arr, 4294967295, Value::Int(0)),
            Err(NativeError::RangeError(_))
        ));
    }

    #[test]
    fn dense_generic_property_paths() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        // Writes through the generic `define_property` take the dense fast
        // path: no interning, no linear scan.
        heap.define_property(arr, Wtf8::from_str("0"), Property::data(Value::Int(1)))
            .unwrap();
        heap.define_property(arr, Wtf8::from_str("1"), Property::data(Value::Int(2)))
            .unwrap();
        assert_eq!(heap.dense_len(arr), 2);
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("0")).unwrap().value,
            Value::Int(1)
        );

        // Named properties still land in the ordinary property list.
        heap.define_property(arr, Wtf8::from_str("foo"), Property::data(Value::Int(9)))
            .unwrap();
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("foo")).unwrap().value,
            Value::Int(9)
        );
        assert_eq!(heap.dense_len(arr), 2);

        // `length` write through the generic path truncates.
        heap.define_property(
            arr,
            Wtf8::from_str("length"),
            Property {
                value: Value::Int(1),
                writable: true,
                enumerable: false,
                configurable: false,
            },
        )
        .unwrap();
        assert_eq!(heap.dense_len(arr), 1);
        assert_eq!(heap.get_property(arr, Wtf8::from_str("1")), None);

        // Deleting an index punches a hole; `length` is non-configurable.
        heap.define_property(arr, Wtf8::from_str("5"), Property::data(Value::Int(6)))
            .unwrap();
        assert!(heap.delete_property(arr, Wtf8::from_str("5")));
        assert!(!heap.dense_has(arr, 5));
        assert!(!heap.delete_property(arr, Wtf8::from_str("length")));
        assert!(!heap.delete_property(arr, Wtf8::from_str("nope")));
    }

    #[test]
    fn dense_sparse_fallback_on_huge_gap() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        heap.dense_push(arr, Value::Int(1)).unwrap();
        // A gap beyond the hole budget materializes to the property map
        // instead of allocating millions of holes.
        heap.dense_set(arr, 1_000_000, Value::Int(2)).unwrap();
        assert!(heap.get(arr).dense().is_none());
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("0")).unwrap().value,
            Value::Int(1)
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("1000000"))
                .unwrap()
                .value,
            Value::Int(2)
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("length"))
                .unwrap()
                .value,
            Value::Int(1000001)
        );
    }

    #[test]
    fn dense_define_property_huge_gap_falls_back() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        heap.define_property(arr, Wtf8::from_str("0"), Property::data(Value::Int(1)))
            .unwrap();
        assert!(heap.get(arr).dense().is_some());
        heap.define_property(
            arr,
            Wtf8::from_str("5000000"),
            Property::data(Value::Int(2)),
        )
        .unwrap();
        assert!(heap.get(arr).dense().is_none());
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("5000000"))
                .unwrap()
                .value,
            Value::Int(2)
        );
    }

    #[test]
    fn dense_materialize_matches_slow_layout() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        for i in 0..50u64 {
            heap.dense_push(arr, Value::Int(i as i32)).unwrap();
        }
        heap.dense_set(arr, 60, Value::Int(60)).unwrap(); // holes at 50..60
        heap.define_property(arr, Wtf8::from_str("tag"), Property::data(Value::Int(7)))
            .unwrap();
        heap.materialize_dense(arr).unwrap();
        assert!(heap.get(arr).dense().is_none());

        // `length` is an ordinary non-configurable property now.
        let len_prop = heap.get_property(arr, Wtf8::from_str("length")).unwrap();
        assert_eq!(len_prop.value, Value::Int(61));
        assert!(!len_prop.configurable);

        // Elements are ordinary enumerable data properties; holes are absent.
        for i in 0..50u64 {
            let p = heap
                .get_property(arr, Wtf8::from_str(&i.to_string()))
                .unwrap();
            assert_eq!(p.value, Value::Int(i as i32));
            assert!(p.enumerable && p.writable && p.configurable);
        }
        for i in 50..60u64 {
            assert_eq!(heap.get_property(arr, Wtf8::from_str(&i.to_string())), None);
        }
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("60")).unwrap().value,
            Value::Int(60)
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("tag")).unwrap().value,
            Value::Int(7)
        );

        // Materializing twice is a no-op.
        heap.materialize_dense(arr).unwrap();
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("length"))
                .unwrap()
                .value,
            Value::Int(61)
        );
    }

    #[test]
    fn dense_values_survive_collection() {
        let mut heap = Heap::new();
        heap.set_gc_threshold(64);
        let arr = heap.new_dense_array().unwrap();
        let base = heap.protect(Value::Object(arr));
        for i in 0..200 {
            let s = heap.alloc_str(&format!("elem{i}")).unwrap();
            // `dense_push` performs no GC-heap allocation, so the fresh
            // string cannot be freed before it lands in traced storage.
            heap.dense_push(arr, Value::Str(s)).unwrap();
        }
        heap.release_temp(base);
        // The array itself is only reachable from here; a collection must
        // keep every dense element alive through the tracer.
        let base = heap.protect(Value::Object(arr));
        heap.collect();
        heap.release_temp(base);
        let base = heap.protect(Value::Object(arr));
        assert_eq!(heap.dense_len(arr), 200);
        for i in 0..200 {
            match heap.dense_get(arr, i) {
                Some(Value::Str(s)) => assert_eq!(s.as_wtf8(), Wtf8::from_str(&format!("elem{i}"))),
                other => panic!("element {i} lost: {other:?}"),
            }
        }
        heap.release_temp(base);
    }

    /// Perf smoke test: 100k-element construction through dense storage
    /// must be fast (the old property-map path took ~31s for 20k).
    #[test]
    fn dense_100k_push_is_fast() {
        let mut heap = Heap::new();
        heap.set_gc_threshold(usize::MAX);
        let n = 100_000u64;
        let start = std::time::Instant::now();
        let arr = heap.new_dense_array().unwrap();
        for i in 0..n {
            heap.dense_push(arr, Value::Int((i & 0x7fff) as i32))
                .unwrap();
        }
        heap.materialize_dense(arr).unwrap();
        let dt = start.elapsed();
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("length"))
                .unwrap()
                .value,
            Value::Int(n as i32)
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("99999"))
                .unwrap()
                .value,
            Value::Int(99999 & 0x7fff)
        );
        assert!(
            dt.as_secs() < 10,
            "dense 100k push+materialize too slow: {dt:?}"
        );
    }

    /// Perf smoke test for the `flatten_into` pattern: nested input
    /// flattened with a local output cursor (no per-element `length`
    /// property traffic), then materialized once.
    #[test]
    fn dense_flatten_pattern_is_fast() {
        let mut heap = Heap::new();
        heap.set_gc_threshold(usize::MAX);
        // Build a nested input: 1000 inner arrays of 100 elements, kept
        // dense (the flatten loop reads them through the dense store).
        let mut inners = Vec::new();
        for o in 0..1000u64 {
            let inner = heap.new_dense_array().unwrap();
            for i in 0..100u64 {
                heap.dense_push(inner, Value::Int((o * 100 + i) as i32))
                    .unwrap();
            }
            inners.push(inner);
        }
        let start = std::time::Instant::now();
        // flatten_into sketch: local cursor, dense pushes, one length
        // update at the end (materialize).
        let out = heap.new_dense_array().unwrap();
        for inner in &inners {
            let len = heap.dense_len(*inner);
            for i in 0..len {
                let v = heap.dense_get(*inner, i).unwrap();
                heap.dense_push(out, v).unwrap();
            }
        }
        heap.materialize_dense(out).unwrap();
        let dt = start.elapsed();
        assert_eq!(
            heap.get_property(out, Wtf8::from_str("length"))
                .unwrap()
                .value,
            Value::Int(100_000)
        );
        assert_eq!(
            heap.get_property(out, Wtf8::from_str("99999"))
                .unwrap()
                .value,
            Value::Int(99999)
        );
        assert!(dt.as_secs() < 10, "dense flatten pattern too slow: {dt:?}");
    }

    // ------------------------------------------------------------------
    // Hidden classes (shapes)
    // ------------------------------------------------------------------

    #[test]
    fn shape_shared_by_same_definition_sequence() {
        let mut heap = Heap::new();
        let shapes_before = heap.shapes.len();
        let mut objs = Vec::new();
        for _ in 0..3 {
            let o = heap
                .alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
            for (i, name) in ["a", "b", "c"].iter().enumerate() {
                heap.define_property(
                    o,
                    Wtf8::from_str(name),
                    Property::data(Value::Int(i as i32)),
                )
                .unwrap();
            }
            assert!(heap.debug_shape_consistent(o));
            objs.push(o);
        }
        // One new interned shape for the whole sequence, shared by all
        // three objects.
        assert_eq!(heap.shapes.len(), shapes_before + 3);
        let s0 = heap.get(objs[0]).shape;
        assert!(objs.iter().all(|&o| heap.get(o).shape == s0));
        for &o in &objs {
            assert_eq!(
                heap.get_property(o, Wtf8::from_str("b")).unwrap().value,
                Value::Int(1)
            );
        }
    }

    #[test]
    fn shape_diverges_on_different_layout() {
        let mut heap = Heap::new();
        let mk = |heap: &mut Heap, keys: &[&str]| {
            let o = heap
                .alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
            for (i, name) in keys.iter().enumerate() {
                heap.define_property(
                    o,
                    Wtf8::from_str(name),
                    Property::data(Value::Int(i as i32)),
                )
                .unwrap();
            }
            assert!(heap.debug_shape_consistent(o));
            o
        };
        let ab = mk(&mut heap, &["a", "b"]);
        let ba = mk(&mut heap, &["b", "a"]);
        let abc = mk(&mut heap, &["a", "b", "c"]);
        // Different key order / key sets intern to different shapes.
        assert_ne!(heap.get(ab).shape, heap.get(ba).shape);
        assert_ne!(heap.get(ab).shape, heap.get(abc).shape);
        // ...but lookups still resolve correctly on every shape.
        assert_eq!(
            heap.get_property(ba, Wtf8::from_str("a")).unwrap().value,
            Value::Int(1)
        );
        assert_eq!(
            heap.get_property(ba, Wtf8::from_str("b")).unwrap().value,
            Value::Int(0)
        );
        assert_eq!(
            heap.get_property(abc, Wtf8::from_str("c")).unwrap().value,
            Value::Int(2)
        );
    }

    #[test]
    fn shape_lookup_after_add_overwrite_delete() {
        let mut heap = Heap::new();
        let o = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        for (i, name) in ["a", "b", "c", "d"].iter().enumerate() {
            heap.define_property(
                o,
                Wtf8::from_str(name),
                Property::data(Value::Int(i as i32)),
            )
            .unwrap();
        }
        let s4 = heap.get(o).shape;
        // Overwrite: layout unchanged, shape identical.
        for _ in 0..100 {
            heap.define_property(o, Wtf8::from_str("c"), Property::data(Value::Int(30)))
                .unwrap();
            assert_eq!(heap.get(o).shape, s4);
        }
        assert_eq!(
            heap.get_property(o, Wtf8::from_str("c")).unwrap().value,
            Value::Int(30)
        );
        assert!(heap.debug_shape_consistent(o));
        // Delete shifts indices: the shape rebuilds and lookups stay exact.
        assert!(heap.delete_property(o, Wtf8::from_str("a")));
        assert_ne!(heap.get(o).shape, s4);
        assert!(heap.debug_shape_consistent(o));
        assert_eq!(
            heap.get_property(o, Wtf8::from_str("c")).unwrap().value,
            Value::Int(30)
        );
        assert_eq!(
            heap.get_property(o, Wtf8::from_str("b")).unwrap().value,
            Value::Int(1)
        );
        assert_eq!(heap.get_property(o, Wtf8::from_str("a")), None);
        assert!(heap.delete_property(o, Wtf8::from_str("c")));
        assert!(!heap.delete_property(o, Wtf8::from_str("c")));
        assert!(heap.debug_shape_consistent(o));
        // Re-adding appends at the end: insertion order is preserved and
        // the rebuilt shape is the canonical one for the layout.
        heap.define_property(o, Wtf8::from_str("a"), Property::data(Value::Int(10)))
            .unwrap();
        assert!(heap.debug_shape_consistent(o));
        assert_eq!(
            heap.get_property(o, Wtf8::from_str("a")).unwrap().value,
            Value::Int(10)
        );
        let keys: Vec<&Wtf8> = heap
            .get(o)
            .properties
            .iter()
            .map(|(k, _)| k.as_wtf8())
            .collect();
        assert_eq!(
            keys,
            [
                Wtf8::from_str("b"),
                Wtf8::from_str("d"),
                Wtf8::from_str("a")
            ]
        );
    }

    #[test]
    fn inline_cache_fill_growth_stays_monomorphic() {
        let mut heap = Heap::new();
        let o = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.define_property(o, Wtf8::from_str("x"), Property::data(Value::Int(1)))
            .unwrap();
        let ic = Cell::new(InlineCache::default());
        heap.get_property_cached(o, Wtf8::from_str("x"), &ic)
            .unwrap();
        let shape_before = heap.get(o).shape.0;
        assert!(!ic.get().is_megamorphic());
        // Adding a property extends the shape: the next miss refills
        // and stays monomorphic (no megamorphic pessimization for the
        // common growth pattern).
        heap.define_property(o, Wtf8::from_str("y"), Property::data(Value::Int(2)))
            .unwrap();
        let shape_after = heap.get(o).shape.0;
        assert_ne!(shape_before, shape_after);
        let got = heap
            .get_property_cached(o, Wtf8::from_str("x"), &ic)
            .unwrap();
        assert_eq!(got.value, Value::Int(1));
        let e = ic.get();
        assert!(!e.is_megamorphic());
        assert_eq!(e.shape, shape_after);
        // And the refilled entry hits.
        let got = heap
            .get_property_cached(o, Wtf8::from_str("x"), &ic)
            .unwrap();
        assert_eq!(got.value, Value::Int(1));
        assert_eq!(ic.get(), e);
    }

    #[test]
    fn inline_cache_megamorphic_fallback() {
        let mut heap = Heap::new();
        // Two layouts sharing the key "x" at different slots/shapes.
        let a = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.define_property(a, Wtf8::from_str("x"), Property::data(Value::Int(1)))
            .unwrap();
        heap.define_property(a, Wtf8::from_str("y"), Property::data(Value::Int(2)))
            .unwrap();
        let b = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.define_property(b, Wtf8::from_str("z"), Property::data(Value::Int(3)))
            .unwrap();
        heap.define_property(b, Wtf8::from_str("x"), Property::data(Value::Int(4)))
            .unwrap();
        assert_ne!(heap.get(a).shape, heap.get(b).shape);
        let ic = Cell::new(InlineCache::default());
        assert_eq!(
            heap.get_property_cached(a, Wtf8::from_str("x"), &ic)
                .unwrap()
                .value,
            Value::Int(1)
        );
        assert!(!ic.get().is_megamorphic());
        // Second distinct shape at the same call site: megamorphic.
        assert_eq!(
            heap.get_property_cached(b, Wtf8::from_str("x"), &ic)
                .unwrap()
                .value,
            Value::Int(4)
        );
        assert!(ic.get().is_megamorphic());
        // Megamorphic sites take the slow path: correct values for both
        // shapes, and the marker sticks (no refill churn).
        for _ in 0..3 {
            assert_eq!(
                heap.get_property_cached(a, Wtf8::from_str("x"), &ic)
                    .unwrap()
                    .value,
                Value::Int(1)
            );
            assert!(ic.get().is_megamorphic());
            assert_eq!(
                heap.get_property_cached(b, Wtf8::from_str("x"), &ic)
                    .unwrap()
                    .value,
                Value::Int(4)
            );
            assert!(ic.get().is_megamorphic());
        }
    }

    #[test]
    fn shape_data_accessor_conversion_round_trip() {
        let mut heap = Heap::new();
        let o = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.define_property(o, Wtf8::from_str("x"), Property::data(Value::Int(1)))
            .unwrap();
        heap.define_property(o, Wtf8::from_str("y"), Property::data(Value::Int(2)))
            .unwrap();
        assert_eq!(
            heap.get_property(o, Wtf8::from_str("x")).unwrap().value,
            Value::Int(1)
        );
        // Data -> accessor: the data entry is evicted, the shape rebuilds.
        heap.define_accessor(
            o,
            Wtf8::from_str("x"),
            AccessorProp::new(Value::Undefined, Value::Undefined),
        )
        .unwrap();
        assert!(heap.debug_shape_consistent(o));
        assert_eq!(heap.get_property(o, Wtf8::from_str("x")), None);
        assert!(heap.get_accessor(o, Wtf8::from_str("x")).is_some());
        assert_eq!(
            heap.get_property(o, Wtf8::from_str("y")).unwrap().value,
            Value::Int(2)
        );
        // Accessor -> data: the accessor entry is evicted.
        heap.define_property(o, Wtf8::from_str("x"), Property::data(Value::Int(9)))
            .unwrap();
        assert!(heap.debug_shape_consistent(o));
        assert_eq!(
            heap.get_property(o, Wtf8::from_str("x")).unwrap().value,
            Value::Int(9)
        );
        assert_eq!(heap.get_accessor(o, Wtf8::from_str("x")), None);
    }

    #[test]
    fn shape_overwrite_keeps_attributes() {
        let mut heap = Heap::new();
        let o = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let ro = |v| Property {
            value: v,
            writable: false,
            enumerable: false,
            configurable: false,
        };
        heap.define_property(o, Wtf8::from_str("ro"), ro(Value::Int(1)))
            .unwrap();
        // Warm the shape path, then overwrite through it.
        assert_eq!(
            heap.get_property(o, Wtf8::from_str("ro")).unwrap().value,
            Value::Int(1)
        );
        heap.define_property(o, Wtf8::from_str("ro"), ro(Value::Int(2)))
            .unwrap();
        let p = heap.get_property(o, Wtf8::from_str("ro")).unwrap();
        assert_eq!(p.value, Value::Int(2));
        assert!(!p.writable && !p.enumerable && !p.configurable);
    }

    #[test]
    fn shape_survives_collection() {
        let mut heap = Heap::new();
        heap.set_gc_threshold(1); // collect on (nearly) every allocation
        let o = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let base = heap.protect(Value::Object(o));
        heap.define_property(o, Wtf8::from_str("k"), Property::data(Value::Int(42)))
            .unwrap();
        // A long key that cannot inline: the shape's index entry holds a
        // heap string handle, so the shape must root it across collects.
        let long_key = "k".repeat(64);
        heap.define_property(o, Wtf8::from_str(&long_key), Property::data(Value::Int(7)))
            .unwrap();
        assert_eq!(
            heap.get_property(o, Wtf8::from_str("k")).unwrap().value,
            Value::Int(42)
        );
        assert_eq!(
            heap.get_property(o, Wtf8::from_str(&long_key))
                .unwrap()
                .value,
            Value::Int(7)
        );
        // Churn garbage through collections while the shape must keep its
        // keys (and their strings) alive and resolving.
        for i in 0..200 {
            let g = heap
                .alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
            heap.define_property(g, Wtf8::from_str("tmp"), Property::data(Value::Int(i)))
                .unwrap();
            // `g` is not used past this point, so each iteration's object
            // is garbage by the next collection.
        }
        heap.collect();
        assert_eq!(
            heap.get_property(o, Wtf8::from_str("k")).unwrap().value,
            Value::Int(42)
        );
        assert_eq!(
            heap.get_property(o, Wtf8::from_str(&long_key))
                .unwrap()
                .value,
            Value::Int(7)
        );
        heap.define_property(o, Wtf8::from_str("k"), Property::data(Value::Int(43)))
            .unwrap();
        assert_eq!(
            heap.get_property(o, Wtf8::from_str("k")).unwrap().value,
            Value::Int(43)
        );
        assert!(heap.debug_shape_consistent(o));
        heap.release_temp(base);
    }

    #[test]
    fn shape_dictionary_mode_past_cap() {
        let mut heap = Heap::new();
        let o = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let n = Shape::MAX_ENTRIES as usize * 2 + 10;
        for i in 0..n {
            heap.define_property(
                o,
                Wtf8::from_str(&format!("k{i}")),
                Property::data(Value::Int(i as i32)),
            )
            .unwrap();
        }
        assert!(heap.debug_shape_consistent(o));
        // The shape tracks exactly the first MAX_ENTRIES keys; the rest
        // resolve through the fallback scan.
        let shape = &heap.shapes[heap.get(o).shape.0 as usize];
        assert_eq!(shape.index.len(), Shape::MAX_ENTRIES as usize);
        assert_eq!(shape.data_len, Shape::MAX_ENTRIES);
        for i in 0..n {
            assert_eq!(
                heap.get_property(o, Wtf8::from_str(&format!("k{i}")))
                    .unwrap()
                    .value,
                Value::Int(i as i32)
            );
        }
        // Structural changes past the cap keep working: deleting an
        // untracked key leaves the shape alone, deleting a tracked key
        // rebuilds the tracked prefix.
        let s_before = heap.get(o).shape;
        assert!(heap.delete_property(o, Wtf8::from_str(&format!("k{}", n - 1))));
        assert_eq!(heap.get(o).shape, s_before);
        assert!(heap.debug_shape_consistent(o));
        assert!(heap.delete_property(o, Wtf8::from_str("k0")));
        assert!(heap.debug_shape_consistent(o));
        assert_eq!(
            heap.get_property(o, Wtf8::from_str("k1")).unwrap().value,
            Value::Int(1)
        );
        assert_eq!(heap.get_property(o, Wtf8::from_str("k0")), None);
        // Appending past the cap does not grow the shape.
        heap.define_property(o, Wtf8::from_str("k0"), Property::data(Value::Int(-1)))
            .unwrap();
        assert!(heap.debug_shape_consistent(o));
        assert_eq!(
            heap.get_property(o, Wtf8::from_str("k0")).unwrap().value,
            Value::Int(-1)
        );
    }

    #[test]
    fn shape_table_compacts_on_collect() {
        let mut heap = Heap::new();
        let base = heap.shapes.len();
        // Build many objects with pairwise-distinct layouts. None are
        // rooted, so the collection below frees them all.
        const N: usize = 200;
        for i in 0..N {
            let o = heap
                .alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
            heap.define_property(
                o,
                Wtf8::from_str(&format!("layout_key_{i}")),
                Property::data(Value::Int(i as i32)),
            )
            .unwrap();
            assert!(heap.debug_shape_consistent(o));
        }
        // One fresh interned shape per distinct layout, all children of
        // the root.
        assert_eq!(heap.shapes.len(), base + N);
        heap.collect();
        // Every object died: every non-root shape is evicted and the
        // table shrinks back to its baseline. The root's transition
        // entries to the dead shapes drop with them (pure cache).
        assert_eq!(heap.shapes.len(), base);
        assert!(heap.shapes[0].transitions.is_empty());
    }

    #[test]
    fn shape_transition_table_interns_across_many_keys() {
        // The transition table interns children: re-appending the same
        // (key, list) must hit the cached child instead of interning a
        // duplicate, and 200 distinct root transitions (past the inline
        // capacity) must stay correct through the overflow map.
        let mut heap = Heap::new();
        const N: usize = 200;
        let mut child_ids = Vec::with_capacity(N);
        for i in 0..N {
            let key = heap.alloc_str(&format!("tkey_{i}")).unwrap();
            child_ids.push(heap.shape_add(ShapeId::ROOT, key, 0));
        }
        // All N children hang off the root, one transition entry each.
        assert_eq!(heap.shapes[0].transitions.len(), N);
        for (i, &id) in child_ids.iter().enumerate() {
            let key = heap.alloc_str(&format!("tkey_{i}")).unwrap();
            // Same (key, list) re-transition finds the cached child.
            assert_eq!(heap.shape_add(ShapeId::ROOT, key, 0), id);
            assert_eq!(heap.shapes[0].transition(key, 0), Some(id));
            // The list is part of the key: a data key is not an
            // accessor transition.
            assert_eq!(heap.shapes[0].transition(key, 1), None);
        }
        // Unknown keys miss.
        let missing = heap.alloc_str("tkey_missing").unwrap();
        assert_eq!(heap.shapes[0].transition(missing, 0), None);
        // Accessor-list transitions live in the same table under their
        // own (key, list) entries.
        let acc_key = heap.alloc_str("acc_key").unwrap();
        let acc_child = heap.shape_add(ShapeId::ROOT, acc_key, 1);
        assert_eq!(heap.shapes[0].transition(acc_key, 1), Some(acc_child));
        assert_eq!(heap.shapes[0].transition(acc_key, 0), None);
        assert_eq!(heap.shapes[0].transitions.len(), N + 1);
    }

    #[test]
    fn mark_set_reserve_tracks_live_set_not_alloc_count() {
        // High-mortality heap: 2000 allocations, 8 survivors. The mark
        // set must be sized from the live set, not the garbage-inclusive
        // allocation count, or the reused buffer inflates permanently.
        let mut heap = Heap::new();
        let mut first_base = None;
        let mut survivors = 0;
        for i in 0..2000 {
            let o = heap
                .alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
            if i % 250 == 0 {
                let base = heap.protect(Value::Object(o));
                first_base.get_or_insert(base);
                survivors += 1;
            }
        }
        assert_eq!(survivors, 8);
        heap.collect();
        // The live-set estimate counts exactly the survivors (fresh heap:
        // no prototypes or shape keys to mark).
        assert_eq!(heap.last_live, 8);
        // 8 live entries at load ≤ 1/2 fit in 16 slots; the old
        // `reserve(alloc_count())` would have grown this past 4096.
        assert!(
            heap.marked_set.capacity() <= 32,
            "capacity = {}",
            heap.marked_set.capacity()
        );
        // A second collection reuses the buffer without growing it, and
        // the estimate follows the live set down once the roots drop.
        heap.release_temp(first_base.unwrap());
        heap.collect();
        assert_eq!(heap.last_live, 0);
        assert!(
            heap.marked_set.capacity() <= 32,
            "capacity = {}",
            heap.marked_set.capacity()
        );
    }

    #[test]
    fn shape_table_keeps_live_shapes_across_collect() {
        let mut heap = Heap::new();
        // One rooted object with a two-key layout.
        let keeper = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let tbase = heap.protect(Value::Object(keeper));
        heap.define_property(keeper, Wtf8::from_str("ka"), Property::data(Value::Int(1)))
            .unwrap();
        heap.define_property(keeper, Wtf8::from_str("kb"), Property::data(Value::Int(2)))
            .unwrap();
        assert!(heap.debug_shape_consistent(keeper));
        // Churn through many distinct dead layouts.
        const N: usize = 200;
        for i in 0..N {
            let o = heap
                .alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
            heap.define_property(
                o,
                Wtf8::from_str(&format!("churn_key_{i}")),
                Property::data(Value::Int(i as i32)),
            )
            .unwrap();
        }
        let grown = heap.shapes.len();
        assert!(grown > 3);
        heap.collect();
        // The keeper survived, so exactly its layout chain survives:
        // root -> [ka] -> [ka, kb]. All churn shapes are gone.
        assert_eq!(heap.shapes.len(), 3);
        assert!(heap.debug_shape_consistent(keeper));
        assert_eq!(
            heap.get_property(keeper, Wtf8::from_str("ka"))
                .unwrap()
                .value,
            Value::Int(1)
        );
        assert_eq!(
            heap.get_property(keeper, Wtf8::from_str("kb"))
                .unwrap()
                .value,
            Value::Int(2)
        );
        // The survivor's shape still interns: rebuilding the same layout
        // on a fresh object reuses the kept shape instead of growing the
        // table, and a repeat of a churned layout re-interns cleanly.
        let o2 = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.define_property(o2, Wtf8::from_str("ka"), Property::data(Value::Int(3)))
            .unwrap();
        heap.define_property(o2, Wtf8::from_str("kb"), Property::data(Value::Int(4)))
            .unwrap();
        assert_eq!(heap.get(o2).shape, heap.get(keeper).shape);
        assert_eq!(heap.shapes.len(), 3);
        let o3 = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.define_property(
            o3,
            Wtf8::from_str("churn_key_7"),
            Property::data(Value::Int(7)),
        )
        .unwrap();
        assert!(heap.debug_shape_consistent(o3));
        assert_eq!(heap.shapes.len(), 4);
        heap.release_temp(tbase);
    }

    #[test]
    fn inline_cache_hit_skips_hash_and_refills() {
        let mut heap = Heap::new();
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
        let o = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let ho = scope.add(o);
        heap.define_property(o, Wtf8::from_str("x"), Property::data(Value::Int(1)))
            .unwrap();
        heap.define_property(o, Wtf8::from_str("y"), Property::data(Value::Int(2)))
            .unwrap();

        let ic = Cell::new(InlineCache::default());
        // Miss: falls back to the shape hash and fills the cache.
        let got = heap
            .get_property_cached(ho.get(), Wtf8::from_str("x"), &ic)
            .unwrap();
        assert_eq!(got.value, Value::Int(1));
        let filled = ic.get();
        assert_eq!(filled.list, 0);
        assert_eq!(filled.slot, 0);
        // Hit: same object, same shape — the entry is reused, and the
        // value is read from the object (not the cache).
        heap.define_property(o, Wtf8::from_str("x"), Property::data(Value::Int(42)))
            .unwrap();
        let got = heap
            .get_property_cached(ho.get(), Wtf8::from_str("x"), &ic)
            .unwrap();
        assert_eq!(got.value, Value::Int(42));
        assert_eq!(ic.get(), filled, "hit must not rewrite the entry");
        // A sibling object with the same layout shares the entry.
        let o2 = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.define_property(o2, Wtf8::from_str("x"), Property::data(Value::Int(7)))
            .unwrap();
        heap.define_property(o2, Wtf8::from_str("y"), Property::data(Value::Int(8)))
            .unwrap();
        assert_eq!(heap.get(o2).shape, heap.get(ho.get()).shape);
        let got = heap
            .get_property_cached(o2, Wtf8::from_str("x"), &ic)
            .unwrap();
        assert_eq!(got.value, Value::Int(7));
    }

    #[test]
    fn inline_cache_invalidated_by_layout_change() {
        let mut heap = Heap::new();
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
        let o = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let ho = scope.add(o);
        heap.define_property(o, Wtf8::from_str("x"), Property::data(Value::Int(1)))
            .unwrap();

        let ic = Cell::new(InlineCache::default());
        heap.get_property_cached(ho.get(), Wtf8::from_str("x"), &ic)
            .unwrap();
        let shape_before = ic.get().shape;

        // Deleting the key rebuilds the shape: the cached shape id no
        // longer matches, so the stale slot is never trusted.
        heap.delete_property(o, Wtf8::from_str("x"));
        assert!(
            heap.get_property_cached(ho.get(), Wtf8::from_str("x"), &ic)
                .is_none()
        );
        // A new key re-resolves through the hash and refills.
        heap.define_property(o, Wtf8::from_str("y"), Property::data(Value::Int(9)))
            .unwrap();
        let got = heap
            .get_property_cached(ho.get(), Wtf8::from_str("y"), &ic)
            .unwrap();
        assert_eq!(got.value, Value::Int(9));
        assert_ne!(ic.get().shape, shape_before);
    }

    #[test]
    fn inline_cache_invalidated_by_shape_renumbering() {
        let mut heap = Heap::new();
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
        // A dead layout with a lower shape id than the keeper's, so the
        // collection below renumbers the keeper's shape id.
        let doomed = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.define_property(doomed, Wtf8::from_str("zzz"), Property::data(Value::Int(0)))
            .unwrap();
        let keeper = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let hkeeper = scope.add(keeper);
        heap.define_property(keeper, Wtf8::from_str("x"), Property::data(Value::Int(1)))
            .unwrap();
        heap.define_property(keeper, Wtf8::from_str("y"), Property::data(Value::Int(2)))
            .unwrap();

        let ic = Cell::new(InlineCache::default());
        let got = heap
            .get_property_cached(hkeeper.get(), Wtf8::from_str("x"), &ic)
            .unwrap();
        assert_eq!(got.value, Value::Int(1));
        let filled = ic.get();
        assert_eq!(filled.epoch, 0);
        let shape_before = filled.shape;

        // The doomed object's shape dies here: the keeper's shape is
        // renumbered and the epoch bumps.
        heap.collect();
        assert_eq!(heap.shape_epoch(), 1);
        let shape_after = heap.get(hkeeper.get()).shape.0;
        assert_ne!(shape_before, shape_after);

        // The stale entry reads as a miss (never a wrong-shape hit): the
        // lookup re-resolves through the hash and refills under the new
        // epoch.
        let got = heap
            .get_property_cached(hkeeper.get(), Wtf8::from_str("x"), &ic)
            .unwrap();
        assert_eq!(got.value, Value::Int(1));
        let refilled = ic.get();
        assert_eq!(refilled.epoch, 1);
        assert_eq!(refilled.shape, shape_after);
        assert_eq!(refilled.slot, 0);
        let got = heap
            .get_property_cached(hkeeper.get(), Wtf8::from_str("y"), &ic)
            .unwrap();
        assert_eq!(got.value, Value::Int(2));
    }

    #[test]
    fn inline_cache_accessor_and_data_lists_do_not_alias() {
        let mut heap = Heap::new();
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
        let o = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let ho = scope.add(o);
        heap.define_accessor(
            o,
            Wtf8::from_str("a"),
            AccessorProp::new(Value::Undefined, Value::Undefined),
        )
        .unwrap();

        let ic = Cell::new(InlineCache::default());
        let got = heap
            .get_accessor_cached(ho.get(), Wtf8::from_str("a"), &ic)
            .unwrap();
        assert_eq!(got.get, Value::Undefined);
        assert_eq!(ic.get().list, 1);
        // The same call-site slot must not serve the data-property path:
        // "a" is not a data property.
        assert!(
            heap.get_property_cached(ho.get(), Wtf8::from_str("a"), &ic)
                .is_none()
        );
    }

    #[test]
    fn shape_perf_smoke() {
        // 100k shape-resolved property writes + reads on same-shaped
        // objects. Without the shape index each access scans the property
        // list linearly; with it each resolves in O(1). Generous bound:
        // this is a smoke test that the fast path is wired up, not a
        // benchmark.
        let mut heap = Heap::new();
        let mut objs = Vec::new();
        for n in 0..1000 {
            let o = heap
                .alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
            for (i, name) in ["w", "x", "y", "z"].iter().enumerate() {
                heap.define_property(
                    o,
                    Wtf8::from_str(name),
                    Property::data(Value::Int(i as i32)),
                )
                .unwrap();
            }
            heap.define_property(o, Wtf8::from_str("n"), Property::data(Value::Int(n)))
                .unwrap();
            objs.push(o);
        }
        // All 1000 objects share one shape for the w/x/y/z/n layout.
        let s0 = heap.get(objs[0]).shape;
        assert!(objs.iter().all(|&o| heap.get(o).shape == s0));
        let start = std::time::Instant::now();
        let mut sum = 0i64;
        for _ in 0..100 {
            for &o in &objs {
                heap.define_property_str(o, "x", Property::data(Value::Int(7)))
                    .unwrap();
                heap.define_property(o, Wtf8::from_str("z"), Property::data(Value::Int(8)))
                    .unwrap();
                sum += match heap.get_property(o, Wtf8::from_str("x")).unwrap().value {
                    Value::Int(i) => i as i64,
                    _ => unreachable!(),
                };
                sum += match heap.get_property(o, Wtf8::from_str("z")).unwrap().value {
                    Value::Int(i) => i as i64,
                    _ => unreachable!(),
                };
            }
        }
        let dt = start.elapsed();
        // 1000 objects × 100 rounds × (2 writes + 2 reads) = 400k accesses.
        assert_eq!(sum, 1000 * 100 * (7 + 8));
        assert!(dt.as_secs() < 10, "property access too slow: {dt:?}");
    }

    /// RAII reset for [`set_fail_alloc`]: a failing assertion must not
    /// leave injection on for the rest of the test thread.
    struct FailAllocGuard;
    impl Drop for FailAllocGuard {
        fn drop(&mut self) {
            set_fail_alloc(false);
        }
    }

    #[test]
    fn materialize_dense_releases_temp_root_under_alloc_failure() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        heap.dense_push(arr, Value::Int(1)).unwrap();
        heap.dense_push(arr, Value::Int(2)).unwrap();
        let before = heap.temp_roots.len();
        set_fail_alloc(true);
        let _guard = FailAllocGuard;
        // Index keys pack inline, so the injected allocation failure
        // cannot fire inside materialization: it succeeds, and the temp
        // root taken around the inner call is still released.
        heap.materialize_dense(arr).unwrap();
        drop(_guard);
        assert_eq!(
            heap.temp_roots.len(),
            before,
            "temp root leaked on the materialize path"
        );
        // And the array is still usable afterwards.
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("0")).unwrap().value,
            Value::Int(1)
        );
    }

    #[test]
    fn define_property_dense_length_non_writable_materializes() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        heap.dense_push(arr, Value::Int(1)).unwrap();
        heap.dense_push(arr, Value::Int(2)).unwrap();
        heap.define_property(
            arr,
            Wtf8::from_str("length"),
            Property {
                value: Value::Int(2),
                writable: false,
                enumerable: false,
                configurable: false,
            },
        )
        .unwrap();
        assert!(
            heap.get(arr).dense().is_none(),
            "non-writable length must materialize the dense array"
        );
        let p = heap.get_property(arr, Wtf8::from_str("length")).unwrap();
        assert!(!p.writable, "writable:false on length must stick");
        assert_eq!(p.value, Value::Int(2));
    }

    #[test]
    fn define_property_dense_index_keeps_non_default_attributes() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        heap.dense_push(arr, Value::Int(1)).unwrap();
        heap.dense_push(arr, Value::Int(2)).unwrap();
        // Non-default attributes cannot live in the dense store (which
        // keeps values only): the array materializes and the full
        // descriptor is stored through the ordinary path.
        heap.define_property(
            arr,
            Wtf8::from_str("0"),
            Property {
                value: Value::Int(9),
                writable: false,
                enumerable: true,
                configurable: true,
            },
        )
        .unwrap();
        assert!(
            heap.get(arr).dense().is_none(),
            "attributed index define must materialize the dense array"
        );
        let p = heap.get_property(arr, Wtf8::from_str("0")).unwrap();
        assert_eq!(p.value, Value::Int(9));
        assert!(!p.writable, "writable:false must stick");
        assert!(p.enumerable && p.configurable);
        // Untouched elements and length survive materialization.
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("1")).unwrap().value,
            Value::Int(2)
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("length"))
                .unwrap()
                .value,
            Value::Int(2)
        );
    }

    #[test]
    fn define_property_dense_default_attributes_stay_on_fast_path() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        heap.dense_push(arr, Value::Int(1)).unwrap();
        heap.define_property(arr, Wtf8::from_str("0"), Property::data(Value::Int(7)))
            .unwrap();
        assert!(
            heap.get(arr).dense().is_some(),
            "default attributes must keep the dense fast path"
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("0")).unwrap().value,
            Value::Int(7)
        );
        // Writable length keeps the dense path too.
        heap.define_property(
            arr,
            Wtf8::from_str("length"),
            Property {
                value: Value::Int(1),
                writable: true,
                enumerable: false,
                configurable: false,
            },
        )
        .unwrap();
        assert!(heap.get(arr).dense().is_some());
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("length"))
                .unwrap()
                .value,
            Value::Int(1)
        );
    }

    #[test]
    fn materialize_dense_key_interning_survives_alloc_failure() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        heap.dense_push(arr, Value::Int(10)).unwrap();
        heap.dense_push(arr, Value::Int(20)).unwrap();
        heap.dense_push(arr, Value::Int(30)).unwrap();
        set_fail_alloc(true);
        let _guard = FailAllocGuard;
        // Index keys and "length" pack inline, so the injected allocation
        // failure cannot fire mid-loop: materialization succeeds in one
        // pass with no partial index properties and the dense store fully
        // converted.
        heap.materialize_dense(arr).unwrap();
        drop(_guard);
        assert!(heap.get(arr).dense().is_none());
        let mut keys: Vec<Wtf8Buf> = heap
            .get(arr)
            .properties
            .iter()
            .map(|(k, _)| k.as_wtf8().to_owned())
            .collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                Wtf8Buf::from("0"),
                Wtf8Buf::from("1"),
                Wtf8Buf::from("2"),
                Wtf8Buf::from("length"),
            ]
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("0")).unwrap().value,
            Value::Int(10)
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("1")).unwrap().value,
            Value::Int(20)
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("2")).unwrap().value,
            Value::Int(30)
        );
    }

    // --- Small-string optimization (15-byte inline) ---

    #[test]
    fn sso_alloc_boundary_15_inline_16_heap() {
        let mut heap = Heap::new();
        let inline = heap.alloc_str(&"a".repeat(15)).unwrap();
        assert!(inline.is_inline());
        assert!(inline.heap_handle().is_none());
        assert_eq!(inline.as_wtf8(), Wtf8::from_str(&"a".repeat(15)));
        let heap_s = heap.alloc_str(&"a".repeat(16)).unwrap();
        assert!(!heap_s.is_inline());
        assert!(heap_s.heap_handle().is_some());
        assert_eq!(heap_s.as_wtf8(), Wtf8::from_str(&"a".repeat(16)));
    }

    #[test]
    fn sso_short_strings_do_not_allocate() {
        let mut heap = Heap::new();
        let before = heap.alloc_count();
        for i in 0..1_000 {
            let s = heap.alloc_str(&format!("key-{i}")).unwrap();
            assert!(s.is_inline());
        }
        assert_eq!(
            heap.alloc_count(),
            before,
            "short strings must not touch the allocator"
        );
        // Long strings allocate, one per call, and are pointer-distinct.
        let mut handles = Vec::new();
        for i in 0..100 {
            let s = heap
                .alloc_str(&format!("long-key-{i}-{}", "x".repeat(24)))
                .unwrap();
            handles.push(s.heap_handle().expect("long string is heap-backed"));
        }
        assert_eq!(heap.alloc_count(), before + 100);
        for w in handles.windows(2) {
            assert_ne!(w[0], w[1]);
        }
    }

    #[test]
    fn sso_concat_stays_inline_when_short() {
        let mut heap = Heap::new();
        let c = heap
            .concat_str(
                Str::inline("hello, ").unwrap(),
                Str::inline("world").unwrap(),
            )
            .unwrap();
        assert!(c.is_inline());
        assert_eq!(c.as_wtf8(), "hello, world");
        // 13 + 3 = 16 bytes: crosses the boundary, spills to the heap.
        let d = heap
            .concat_str(
                Str::inline(&"x".repeat(13)).unwrap(),
                Str::inline("yz!").unwrap(),
            )
            .unwrap();
        assert!(!d.is_inline());
        assert_eq!(
            d.as_wtf8(),
            Wtf8::from_str(&format!("{}yz!", "x".repeat(13)))
        );
    }

    #[test]
    fn sso_heap_strings_survive_collection_when_rooted() {
        let mut heap = Heap::new();
        let long = heap.alloc_str(&"rooted".repeat(6)).unwrap();
        assert!(!long.is_inline());
        let base = heap.protect(Value::Str(long));
        // Unrooted heap strings for the collector to sweep.
        for _ in 0..10 {
            heap.alloc_str(&"sweep-me".repeat(5)).unwrap();
        }
        let before = heap.alloc_count();
        heap.collect();
        assert_eq!(long.as_wtf8(), Wtf8::from_str(&"rooted".repeat(6)));
        assert_eq!(
            heap.alloc_count(),
            before - 10,
            "only the rooted heap string survives"
        );
        heap.release_temp(base);
        heap.collect();
        assert_eq!(heap.alloc_count(), before - 11);
    }

    #[test]
    fn sso_inline_strings_need_no_marking() {
        let mut heap = Heap::new();
        // Inline strings are values, not allocations: a collection with no
        // roots at all frees nothing and loses nothing.
        let mut vals = Vec::new();
        for i in 0..100 {
            vals.push(heap.alloc_str(&format!("v{i}")).unwrap());
        }
        let before = heap.alloc_count();
        heap.collect();
        assert_eq!(heap.alloc_count(), before);
        for (i, v) in vals.iter().enumerate() {
            assert!(v.is_inline());
            assert_eq!(v.as_wtf8(), Wtf8::from_str(&format!("v{i}")));
        }
    }

    // --- `intern_str`: SSO-aware interning ---

    #[test]
    fn intern_str_short_key_needs_no_heap_allocation() {
        let mut heap = Heap::new();
        let before = heap.alloc_count();
        // 1000 distinct short keys ("k0".."k999", 2-4 bytes): every one
        // packs inline, so interning touches neither the allocator nor
        // the intern table.
        for i in 0..1_000 {
            let s = heap.intern_str(&format!("k{i}")).unwrap();
            assert!(s.is_inline(), "short key should inline");
            assert_eq!(s.as_wtf8(), Wtf8::from_str(&format!("k{i}")));
        }
        assert_eq!(
            heap.alloc_count(),
            before,
            "interning short keys must not heap-allocate"
        );
        assert!(
            heap.interned.is_empty(),
            "inline strings need no intern-table entry"
        );
    }

    #[test]
    fn intern_str_long_keys_stay_canonical() {
        let mut heap = Heap::new();
        let before = heap.alloc_count();
        let long = "x".repeat(64);
        let a = heap.intern_str(&long).unwrap();
        let b = heap.intern_str(&long).unwrap();
        assert!(!a.is_inline());
        assert_eq!(a, b, "equal by content");
        assert_eq!(
            a.heap_handle(),
            b.heap_handle(),
            "long keys still deduplicate to one canonical handle"
        );
        assert_eq!(
            heap.alloc_count(),
            before + 1,
            "one heap allocation for the canonical long string"
        );
    }

    #[test]
    fn intern_returns_str_with_inline_short_keys() {
        let mut heap = Heap::new();
        // Same text twice: content-equal `Str`s, no identity required.
        let a = heap.intern("key").unwrap();
        let b = heap.intern("key").unwrap();
        assert_eq!(a, b);
        // Short keys pack inline: no allocation, no intern-table entry.
        assert!(a.is_inline());
        assert_eq!(heap.interned.len(), 0);
        let before = heap.alloc_count();
        let _ = heap.intern("key").unwrap();
        assert_eq!(heap.alloc_count(), before);
        // Long text still goes through the weak intern table and shares
        // one canonical handle.
        let long = "z".repeat(64);
        let c = heap.intern(&long).unwrap();
        let d = heap.intern(&long).unwrap();
        assert_eq!(c, d);
        assert_eq!(c.heap_handle(), d.heap_handle());
        assert_eq!(heap.interned.len(), 1);
    }

    #[test]
    fn define_property_cached_overwrite_needs_no_intern() {
        let mut heap = Heap::new();
        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.define_property(obj, Wtf8::from_str("ab"), Property::data(Value::Int(1)))
            .unwrap();
        // The first definition built the object's shape; overwriting the
        // shape-resolved slot must not touch `intern` at all (no table
        // operation, no allocation).
        let before = heap.alloc_count();
        for i in 2..100 {
            heap.define_property(obj, Wtf8::from_str("ab"), Property::data(Value::Int(i)))
                .unwrap();
        }
        assert_eq!(
            heap.alloc_count(),
            before,
            "cached overwrites must not allocate or consult the intern table"
        );
        assert_eq!(
            heap.get_property(obj, Wtf8::from_str("ab")).unwrap().value,
            Value::Int(99)
        );
    }

    // ------------------------------------------------------------------
    // Array exotic [[DefineOwnProperty]] (spec 10.4.2)
    // ------------------------------------------------------------------

    fn array_data_desc(value: Value) -> ArrayDefineDesc {
        ArrayDefineDesc {
            value: Some(value),
            writable: Some(true),
            enumerable: Some(true),
            configurable: Some(true),
            ..Default::default()
        }
    }

    fn array_len_desc(value: Value) -> ArrayDefineDesc {
        ArrayDefineDesc {
            value: Some(value),
            ..Default::default()
        }
    }

    #[test]
    fn array_exotic_index_define_extends_length() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        heap.dense_push(arr, Value::Int(1)).unwrap();
        heap.dense_push(arr, Value::Int(2)).unwrap();
        // Index past the end: ordinary define, then length -> index + 1.
        assert!(
            heap.define_array_own_property(
                arr,
                Wtf8::from_str("5"),
                &array_data_desc(Value::Int(9))
            )
            .unwrap()
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("length"))
                .unwrap()
                .value,
            Value::Int(6)
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("5")).unwrap().value,
            Value::Int(9)
        );
        // Attribute-less: stays dense.
        assert!(heap.get(arr).dense().is_some());
        // Within bounds: length untouched.
        assert!(
            heap.define_array_own_property(
                arr,
                Wtf8::from_str("1"),
                &array_data_desc(Value::Int(8))
            )
            .unwrap()
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("length"))
                .unwrap()
                .value,
            Value::Int(6)
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("1")).unwrap().value,
            Value::Int(8)
        );
    }

    #[test]
    fn array_exotic_length_shrink_truncates_dense() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        for i in 1..=4 {
            heap.dense_push(arr, Value::Int(i)).unwrap();
        }
        assert!(
            heap.define_array_own_property(
                arr,
                Wtf8::from_str("length"),
                &array_len_desc(Value::Int(2))
            )
            .unwrap()
        );
        assert_eq!(heap.dense_len(arr), 2);
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("length"))
                .unwrap()
                .value,
            Value::Int(2)
        );
        assert_eq!(heap.get_property(arr, Wtf8::from_str("2")), None);
        assert_eq!(heap.get_property(arr, Wtf8::from_str("3")), None);
        assert_eq!(heap.dense_get(arr, 0), Some(Value::Int(1)));
        assert_eq!(heap.dense_get(arr, 1), Some(Value::Int(2)));
    }

    #[test]
    fn array_exotic_length_grow_keeps_dense() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        heap.dense_push(arr, Value::Int(1)).unwrap();
        assert!(
            heap.define_array_own_property(
                arr,
                Wtf8::from_str("length"),
                &array_len_desc(Value::Int(3))
            )
            .unwrap()
        );
        assert_eq!(heap.dense_len(arr), 3);
        assert!(heap.get(arr).dense().is_some());
        assert_eq!(heap.dense_get(arr, 2), None);
    }

    #[test]
    fn array_exotic_nonwritable_length_rejects_extension() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        heap.dense_push(arr, Value::Int(1)).unwrap();
        heap.dense_push(arr, Value::Int(2)).unwrap();
        // Make length non-writable through the exotic.
        let frozen = ArrayDefineDesc {
            writable: Some(false),
            ..Default::default()
        };
        assert!(
            heap.define_array_own_property(arr, Wtf8::from_str("length"), &frozen)
                .unwrap()
        );
        // The array materialized (dense cannot represent the attribute).
        assert!(
            !heap
                .get_property(arr, Wtf8::from_str("length"))
                .unwrap()
                .writable
        );
        // Extending past the end now fails instead of writing.
        assert!(
            !heap
                .define_array_own_property(
                    arr,
                    Wtf8::from_str("5"),
                    &array_data_desc(Value::Int(9))
                )
                .unwrap()
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("length"))
                .unwrap()
                .value,
            Value::Int(2)
        );
        assert_eq!(heap.get_property(arr, Wtf8::from_str("5")), None);
        // Shrinking a non-writable length fails too.
        assert!(
            !heap
                .define_array_own_property(
                    arr,
                    Wtf8::from_str("length"),
                    &array_len_desc(Value::Int(1))
                )
                .unwrap()
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("length"))
                .unwrap()
                .value,
            Value::Int(2)
        );
    }

    #[test]
    fn array_exotic_invalid_lengths_are_range_errors() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        for bad in [
            Value::Int(-1),
            Value::Float(4294967296.0),
            Value::Float(1.5),
        ] {
            let err =
                heap.define_array_own_property(arr, Wtf8::from_str("length"), &array_len_desc(bad));
            assert!(
                matches!(err, Err(NativeError::RangeError(_))),
                "expected RangeError for {bad:?}, got {err:?}"
            );
        }
        // 2^32 - 1 is a valid length.
        assert!(
            heap.define_array_own_property(
                arr,
                Wtf8::from_str("length"),
                &array_len_desc(Value::Float(4294967295.0))
            )
            .unwrap()
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("length"))
                .unwrap()
                .value,
            Value::Float(4294967295.0)
        );
    }

    #[test]
    fn array_exotic_shrink_restores_on_failed_delete() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        for i in 1..=3 {
            heap.dense_push(arr, Value::Int(i)).unwrap();
        }
        // Non-configurable trailing element: the attributed define
        // materializes the array.
        let pinned = ArrayDefineDesc {
            configurable: Some(false),
            ..Default::default()
        };
        assert!(
            heap.define_array_own_property(arr, Wtf8::from_str("2"), &pinned)
                .unwrap()
        );
        // Shrink past it: the delete fails, length is restored to
        // failed_index + 1, and the exotic reports false.
        assert!(
            !heap
                .define_array_own_property(
                    arr,
                    Wtf8::from_str("length"),
                    &array_len_desc(Value::Int(1))
                )
                .unwrap()
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("length"))
                .unwrap()
                .value,
            Value::Int(3)
        );
        // Untouched elements survive; the failed index keeps its value.
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("0")).unwrap().value,
            Value::Int(1)
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("1")).unwrap().value,
            Value::Int(2)
        );
        assert_eq!(
            heap.get_property(arr, Wtf8::from_str("2")).unwrap().value,
            Value::Int(3)
        );
    }

    #[test]
    fn array_exotic_shrink_applies_requested_writability() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        for i in 1..=3 {
            heap.dense_push(arr, Value::Int(i)).unwrap();
        }
        // Shrink and freeze in one descriptor.
        let desc = ArrayDefineDesc {
            value: Some(Value::Int(2)),
            writable: Some(false),
            ..Default::default()
        };
        assert!(
            heap.define_array_own_property(arr, Wtf8::from_str("length"), &desc)
                .unwrap()
        );
        let len = heap.get_property(arr, Wtf8::from_str("length")).unwrap();
        assert_eq!(len.value, Value::Int(2));
        assert!(!len.writable);
        assert_eq!(heap.get_property(arr, Wtf8::from_str("2")), None);
    }

    /// Regression test for the wave-15 test262 GC investigation: the engine
    /// must survive aggressive automatic GC across the workloads the
    /// test262 harness exercises (eval churn, prototype pollution/repair,
    /// modules, large live graphs). Each phase uses a fresh `Vm` with GC
    /// enabled; all phases assert the heap stays intact (no crash, no
    /// collected-live-object corruption).
    #[test]
    fn gc_harness_workloads_survive() {
        use crate::module_loader::ModuleLoader;
        use crate::vm::Vm;

        // Phase 1: threshold-1 eval churn with diverse workloads.
        {
            let mut vm = Vm::new();
            vm.heap_mut().set_gc_threshold(1);
            let snippets = [
                "let o = {a: 1}; o.b = o; o.a",
                "function f(x) { return x * 2; } f(21)",
                "class C { constructor() { this.p = 42; } } (new C()).p",
                "function* g() { yield 1; yield 2; } [...g()].join(',')",
                "Promise.resolve(5).then(v => v + 1)",
                "new Map([[1, 2]]).get(1)",
                "/ab+c/.test('abbbc')",
                "new Uint8Array([1,2,3]).length",
                "Object.prototype.__t = 1; delete Object.prototype.__t; 'ok'",
                "JSON.stringify({a: [1,2]})",
                "eval('40 + 2')",
                "let [a, ...r] = [1,2,3]; a + r.length",
            ];
            for round in 0..3 {
                // Global lexicals persist across scripts per ECMA-262 9.1.1.4;
                // reset between rounds so re-running `let` snippets is legal.
                vm.reset_global_lexical();
                for src in snippets {
                    let r = vm.eval(src);
                    assert!(r.is_ok(), "round {round} failed on {src:?}: {r:?}");
                    vm.drain_microtasks();
                    let _ = vm.take_uncaught_task_errors();
                }
            }
            // Live state survives.
            let v = vm.eval("globalThis.__live = {x: 99}; __live.x").unwrap();
            assert!(matches!(v, crate::value::Value::Int(99)), "got {v:?}");
            vm.heap_mut().collect();
            let v2 = vm.eval("__live.x").unwrap();
            assert!(matches!(v2, crate::value::Value::Int(99)), "got {v2:?}");
        }

        // Phase 2: prototype pollution and repair (harness scrub cycle).
        {
            let mut vm = Vm::new();
            vm.heap_mut().set_gc_threshold(100);
            for round in 0..10 {
                vm.eval(&format!(
                    "Object.prototype.p{round} = {{v: {round}}}; \
                     globalThis.g{round} = [1,2,3];"
                ))
                .unwrap();
                vm.eval(&format!(
                    "delete Object.prototype.p{round}; delete globalThis.g{round}; 'ok'"
                ))
                .unwrap();
            }
            assert_eq!(
                format!(
                    "{:?}",
                    vm.eval("[1,2,3].map(x => x * 2).join(',')").unwrap()
                ),
                "Str(\"2,4,6\")"
            );
        }

        // Phase 3: module loading with GC pressure.
        {
            let mut vm = Vm::new();
            vm.heap_mut().set_gc_threshold(1);
            let mut loader = ModuleLoader::new();
            loader.set_allow_net(false);
            loader.add_source("https://t.invalid/a.js", "export const x = 41;");
            loader.add_source(
                "https://t.invalid/main.js",
                "import { x } from './a.js'; globalThis.mres = x + 1;",
            );
            loader.load(&mut vm, "https://t.invalid/main.js").unwrap();
            let v = vm.eval("globalThis.mres").unwrap();
            assert!(matches!(v, crate::value::Value::Int(42)), "got {v:?}");
        }

        // Phase 4: moderate threshold with a large interconnected graph.
        {
            let mut vm = Vm::new();
            vm.heap_mut().set_gc_threshold(50_000);
            vm.eval(
                r#"
                globalThis.roots = [];
                for (let i = 0; i < 2000; i++) {
                    let o = { id: i, arr: [1,2,3] };
                    o.self = o;
                    o.fn = function(a) { return a + this.id; };
                    globalThis.roots.push(o);
                }
                "ok";
                "#,
            )
            .unwrap();
            vm.heap_mut().collect();
            let v = vm
                .eval(
                    "globalThis.roots.length === 2000 \
                     && globalThis.roots[1999].fn(1) === 2000 \
                     && globalThis.roots[100].self === globalThis.roots[100]",
                )
                .unwrap();
            assert!(matches!(v, crate::value::Value::Bool(true)), "got {v:?}");
        }
    }

    #[test]
    fn own_property_keys_spec_order_with_dense_synthesis() {
        // Spec [[OwnPropertyKeys]] order: ascending indices, then other
        // string keys in insertion order, data properties before accessor
        // properties.
        let mut heap = Heap::new();
        let o = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.define_property(o, Wtf8::from_str("b"), Property::data(Value::Int(1)))
            .unwrap();
        heap.define_property(o, Wtf8::from_str("10"), Property::data(Value::Int(2)))
            .unwrap();
        heap.define_property(o, Wtf8::from_str("a"), Property::data(Value::Int(3)))
            .unwrap();
        heap.define_property(o, Wtf8::from_str("2"), Property::data(Value::Int(4)))
            .unwrap();
        heap.define_accessor(
            o,
            Wtf8::from_str("acc"),
            AccessorProp::new(Value::Undefined, Value::Undefined),
        )
        .unwrap();
        let keys: Vec<Wtf8Buf> = heap
            .own_property_keys(o)
            .iter()
            .map(|k| k.as_wtf8().to_owned())
            .collect();
        assert_eq!(
            keys,
            vec![
                Wtf8Buf::from("2"),
                Wtf8Buf::from("10"),
                Wtf8Buf::from("b"),
                Wtf8Buf::from("a"),
                Wtf8Buf::from("acc"),
            ]
        );

        // Dense arrays synthesize index keys from the store (holes are
        // absent properties) plus "length" right after the indices.
        let arr = heap.new_dense_array().unwrap();
        heap.dense_push(arr, Value::Int(1)).unwrap();
        heap.dense_push(arr, Value::Int(2)).unwrap();
        heap.dense_push(arr, Value::Int(3)).unwrap();
        assert!(heap.delete_property(arr, Wtf8::from_str("1"))); // punch a hole
        let keys: Vec<Wtf8Buf> = heap
            .own_property_keys(arr)
            .iter()
            .map(|k| k.as_wtf8().to_owned())
            .collect();
        assert_eq!(
            keys,
            vec![
                Wtf8Buf::from("0"),
                Wtf8Buf::from("2"),
                Wtf8Buf::from("length"),
            ]
        );
    }

    #[test]
    fn dirtied_reset_objects_tracks_tracked_writes_only() {
        let mut heap = Heap::new();
        let tracked = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let untracked = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        heap.track_reset_object(tracked);

        // Untracked writes never appear in the drain.
        heap.define_property(
            untracked,
            Wtf8::from_str("x"),
            Property::data(Value::Int(1)),
        )
        .unwrap();
        assert!(heap.take_dirtied_reset_objects().is_empty());

        // A tracked write appears, exactly once even across repeats.
        heap.define_property(tracked, Wtf8::from_str("x"), Property::data(Value::Int(1)))
            .unwrap();
        heap.define_property(tracked, Wtf8::from_str("y"), Property::data(Value::Int(2)))
            .unwrap();
        let dirtied = heap.take_dirtied_reset_objects();
        assert_eq!(dirtied, vec![tracked]);

        // The drain clears: a second drain is empty.
        assert!(heap.take_dirtied_reset_objects().is_empty());

        // Whole-object writes (preventExtensions) dirty too.
        heap.prevent_extensions(tracked);
        assert_eq!(heap.take_dirtied_reset_objects(), vec![tracked]);
        assert!(heap.take_dirtied_reset_objects().is_empty());
    }

    #[test]
    fn extensible_query_and_repair_roundtrip() {
        let mut heap = Heap::new();
        let o = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        assert!(heap.is_extensible(o));
        heap.prevent_extensions(o);
        assert!(!heap.is_extensible(o));
        // Scrub repair: re-create the extensible flag.
        heap.set_extensible(o, true);
        assert!(heap.is_extensible(o));
    }

    #[test]
    fn dense_writes_note_reset_writes() {
        let mut heap = Heap::new();
        let arr = heap.new_dense_array().unwrap();
        heap.track_reset_object(arr);
        let e0 = heap.reset_epoch();
        heap.dense_push(arr, Value::Int(1)).unwrap();
        heap.dense_set(arr, 0, Value::Int(7)).unwrap();
        heap.dense_set_len(arr, 3).unwrap();
        heap.materialize_dense(arr).unwrap();
        // Four entry points, each a tracked write: four bumps, one
        // dirtied entry.
        assert_eq!(heap.reset_epoch(), e0.wrapping_add(4));
        assert_eq!(heap.take_dirtied_reset_objects(), vec![arr]);
    }

    #[test]
    fn oversize_string_len_rejected_before_block_handed_out() {
        // `GcStr::from_buf` asserts `len <= u32::MAX`; the guard in
        // `alloc_str_block` must fail *before* any chunk's bump cursor
        // moves or the malloc fallback runs, so no phantom live slot is
        // ever stranded. Nothing is allocated here: only the boundary is
        // exercised.
        let mut heap = Heap::new();
        // Just past the `u32` boundary: rejected, with no block handed
        // out (the 4 GiB-and-change oversize malloc is never attempted).
        assert!(heap.alloc_str_block(u32::MAX as usize + 1).is_err());
        assert!(heap.alloc_str_block(usize::MAX).is_err());
        // No chunk was created and no block handed out, so nothing leaks:
        // the heap is pristine and still allocates normally afterwards.
        assert!(heap.str_chunks.iter().all(|v| v.is_empty()));
        let g = heap
            .alloc_str("still fine after the rejected giant")
            .unwrap()
            .heap_handle()
            .expect("too long to be inline");
        assert_eq!(
            g.as_wtf8(),
            Wtf8::from_str("still fine after the rejected giant")
        );
    }

    #[test]
    fn symbol_keyed_define_get_overwrite() {
        let mut heap = Heap::new();
        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let sym = heap.alloc_symbol(Some(Wtf8::from_str("s"))).unwrap();
        heap.define_property_key(obj, PropKey::Symbol(sym), Property::data(Value::Int(1)))
            .unwrap();
        assert_eq!(
            heap.get_property_key(obj, PropKey::Symbol(sym))
                .unwrap()
                .value,
            Value::Int(1)
        );
        // Overwrite: same key, no duplicate entry.
        heap.define_property_key(obj, PropKey::Symbol(sym), Property::data(Value::Int(2)))
            .unwrap();
        assert_eq!(
            heap.get_property_key(obj, PropKey::Symbol(sym))
                .unwrap()
                .value,
            Value::Int(2)
        );
        assert_eq!(heap.get(obj).symbol_properties.len(), 1);
        // A different symbol with the same description is a different key.
        let other = heap.alloc_symbol(Some(Wtf8::from_str("s"))).unwrap();
        assert_ne!(sym, other);
        assert!(heap.get_property_key(obj, PropKey::Symbol(other)).is_none());
    }

    #[test]
    fn symbol_keys_do_not_collide_with_string_keys() {
        let mut heap = Heap::new();
        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let sym = heap.alloc_symbol(Some(Wtf8::from_str("k"))).unwrap();
        heap.define_property(obj, Wtf8::from_str("k"), Property::data(Value::Int(10)))
            .unwrap();
        heap.define_property_key(obj, PropKey::Symbol(sym), Property::data(Value::Int(20)))
            .unwrap();
        // Both live side by side, addressed independently.
        assert_eq!(
            heap.get_property(obj, Wtf8::from_str("k")).unwrap().value,
            Value::Int(10)
        );
        assert_eq!(
            heap.get_property_key(obj, PropKey::Symbol(sym))
                .unwrap()
                .value,
            Value::Int(20)
        );
        // Deleting the string key leaves the symbol key untouched.
        assert!(heap.delete_property(obj, Wtf8::from_str("k")));
        assert!(heap.get_property(obj, Wtf8::from_str("k")).is_none());
        assert_eq!(
            heap.get_property_key(obj, PropKey::Symbol(sym))
                .unwrap()
                .value,
            Value::Int(20)
        );
    }

    #[test]
    fn symbol_keys_bypass_shape() {
        let mut heap = Heap::new();
        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let before = heap.get(obj).shape;
        let s1 = heap.alloc_symbol(Some(Wtf8::from_str("a"))).unwrap();
        let s2 = heap.alloc_symbol(Some(Wtf8::from_str("b"))).unwrap();
        heap.define_property_key(obj, PropKey::Symbol(s1), Property::data(Value::Int(1)))
            .unwrap();
        assert_eq!(
            heap.get(obj).shape,
            before,
            "symbol define must not transition the shape"
        );
        heap.define_accessor_key(
            obj,
            PropKey::Symbol(s2),
            AccessorProp::new(Value::Undefined, Value::Undefined),
        )
        .unwrap();
        assert_eq!(
            heap.get(obj).shape,
            before,
            "symbol accessor define must not transition the shape"
        );
        assert!(heap.delete_property_key(obj, PropKey::Symbol(s1)));
        assert_eq!(
            heap.get(obj).shape,
            before,
            "symbol delete must not rebuild the shape"
        );
        // The shape still mirrors exactly the string lists.
        assert!(heap.debug_shape_consistent(obj));
        // Sanity: a string define *does* transition, so the test is not
        // vacuous.
        heap.define_property(obj, Wtf8::from_str("x"), Property::data(Value::Int(1)))
            .unwrap();
        assert_ne!(heap.get(obj).shape, before);
        assert!(heap.debug_shape_consistent(obj));
    }

    #[test]
    fn own_symbol_keys_lists_data_then_accessors_in_order() {
        let mut heap = Heap::new();
        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let s1 = heap.alloc_symbol(Some(Wtf8::from_str("one"))).unwrap();
        let s2 = heap.alloc_symbol(Some(Wtf8::from_str("two"))).unwrap();
        let s3 = heap.alloc_symbol(Some(Wtf8::from_str("three"))).unwrap();
        heap.define_property_key(obj, PropKey::Symbol(s1), Property::data(Value::Int(1)))
            .unwrap();
        heap.define_property_key(obj, PropKey::Symbol(s2), Property::data(Value::Int(2)))
            .unwrap();
        heap.define_accessor_key(
            obj,
            PropKey::Symbol(s3),
            AccessorProp::new(Value::Undefined, Value::Undefined),
        )
        .unwrap();
        // String keys never leak into the symbol enumeration.
        heap.define_property(obj, Wtf8::from_str("x"), Property::data(Value::Int(9)))
            .unwrap();
        assert_eq!(heap.own_symbol_keys(obj), vec![s1, s2, s3]);
    }

    #[test]
    fn own_enumerable_symbol_keys_filters_non_enumerable() {
        let mut heap = Heap::new();
        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let vis = heap.alloc_symbol(Some(Wtf8::from_str("vis"))).unwrap();
        let hid = heap.alloc_symbol(Some(Wtf8::from_str("hid"))).unwrap();
        let acc = heap.alloc_symbol(Some(Wtf8::from_str("acc"))).unwrap();
        heap.define_property_key(obj, PropKey::Symbol(vis), Property::data(Value::Int(1)))
            .unwrap();
        heap.define_property_key(
            obj,
            PropKey::Symbol(hid),
            Property {
                value: Value::Int(2),
                writable: true,
                enumerable: false,
                configurable: true,
            },
        )
        .unwrap();
        heap.define_accessor_key(
            obj,
            PropKey::Symbol(acc),
            AccessorProp {
                get: Value::Undefined,
                set: Value::Undefined,
                enumerable: true,
                configurable: true,
            },
        )
        .unwrap();
        assert_eq!(heap.own_enumerable_symbol_keys(obj), vec![vis, acc]);
        // The unfiltered list still sees all three.
        assert_eq!(heap.own_symbol_keys(obj), vec![vis, hid, acc]);
    }

    #[test]
    fn symbol_keyed_accessor_define_get_delete() {
        let mut heap = Heap::new();
        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let sym = heap.alloc_symbol(None).unwrap();
        heap.define_accessor_key(
            obj,
            PropKey::Symbol(sym),
            AccessorProp::new(Value::Int(3), Value::Int(4)),
        )
        .unwrap();
        let acc = heap.get_accessor_key(obj, PropKey::Symbol(sym)).unwrap();
        assert_eq!((acc.get, acc.set), (Value::Int(3), Value::Int(4)));
        // Defining data under the same symbol evicts the accessor: a
        // symbol key lives in exactly one of the two side tables.
        heap.define_property_key(obj, PropKey::Symbol(sym), Property::data(Value::Int(9)))
            .unwrap();
        assert!(heap.get_accessor_key(obj, PropKey::Symbol(sym)).is_none());
        assert_eq!(
            heap.get_property_key(obj, PropKey::Symbol(sym))
                .unwrap()
                .value,
            Value::Int(9)
        );
        assert_eq!(heap.get(obj).symbol_accessors.len(), 0);
        assert_eq!(heap.get(obj).symbol_properties.len(), 1);
        // Delete reports presence and removes only the targeted entry.
        assert!(heap.delete_property_key(obj, PropKey::Symbol(sym)));
        assert!(!heap.delete_property_key(obj, PropKey::Symbol(sym)));
        assert!(heap.get_property_key(obj, PropKey::Symbol(sym)).is_none());
        assert!(heap.get(obj).symbol_properties.is_empty());
    }

    #[test]
    fn symbol_key_keeps_symbol_alive_across_gc() {
        let mut heap = Heap::new();
        let scope = unsafe { HandleScope::new(std::ptr::addr_of_mut!(heap)) };
        let obj = heap
            .alloc_object(Object::new(ObjectKind::Ordinary))
            .unwrap();
        let hobj = scope.add(obj);
        // The only heap edge to this symbol is the property key below:
        // the local `sym` is a bare `Gc`, not a GC root.
        let sym = heap.alloc_symbol(Some(Wtf8::from_str("keyonly"))).unwrap();
        heap.define_property_key(obj, PropKey::Symbol(sym), Property::data(Value::Int(7)))
            .unwrap();
        // Churn: garbage objects a collection will reclaim.
        for _ in 0..200 {
            heap.alloc_str("trash").unwrap();
            heap.alloc_object(Object::new(ObjectKind::Ordinary))
                .unwrap();
        }
        heap.collect();
        // Survivors: the rooted object plus the symbol, kept alive
        // solely by the key edge. Without key marking the symbol would
        // be freed and the count would be 1.
        assert_eq!(heap.alloc_count(), 2);
        assert_eq!(
            heap.get_property_key(hobj.get(), PropKey::Symbol(sym))
                .unwrap()
                .value,
            Value::Int(7)
        );
        // The symbol allocation itself survived: its description reads
        // back intact (a freed box would be UAF here).
        assert_eq!(sym.description.as_deref(), Some(Wtf8::from_str("keyonly")));
    }
}
