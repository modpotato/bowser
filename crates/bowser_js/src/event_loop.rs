//! HTML-style event loop: macrotask queue, microtask queue, timer heap.
//!
//! # Wiring contract with the future JS VM (`vm.rs`, not on disk yet)
//!
//! This module deliberately does NOT implement a VM. The mechanical wiring
//! points, once `vm.rs` lands, are:
//!
//! 1. The host implements [`TaskRunner`] and passes it to
//!    [`EventLoop::run_until_idle`]. The loop hands it [`Task`] values; the
//!    host executes them against its own function/job tables.
//! 2. `Task::FireTimer { timer_id, callback }`: `callback` is the opaque id
//!    the host passed to [`EventLoop::set_timeout`]/[`EventLoop::set_interval`].
//!    The host resolves it to a JS function and invokes it. `timer_id` is the
//!    loop-assigned handle (what `setTimeout` returns to script).
//! 3. `Task::PromiseJob { job }`: `job` is opaque to the loop. The VM's
//!    promise implementation must call [`EventLoop::queue_microtask`] with one
//!    of these whenever a promise reaction becomes runnable.
//! 4. The VM's `console` builtin must forward to
//!    [`EventLoop::console_log`] with the level and the stringified arguments.
//! 5. `setTimeout`/`setInterval` builtins call
//!    [`EventLoop::set_timeout`]/[`EventLoop::set_interval`] with the host's
//!    own function id as `callback`; `clearTimeout`/`clearInterval` call
//!    [`EventLoop::clear_timeout`].
//! 6. If the host drives the loop itself instead of using `run_until_idle`,
//!    it MUST call [`EventLoop::drain_microtasks`] after every macrotask
//!    (promise jobs run to completion before the next macrotask).
//!
//! Interior mutability (`RefCell`) is used so the host can share one
//! `EventLoop` (e.g. via `Rc`) with the runner without borrow conflicts when
//! a task schedules more work.

use std::cell::{Cell, RefCell};
use std::cmp::{Ordering, Reverse};
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap};
use std::fmt::Display;
use std::hash::{BuildHasherDefault, Hasher};
use std::rc::Rc;
use std::time::Instant;

/// Opaque handle identifying a timer, as returned to script by `setTimeout`.
pub type TimerId = u32;

/// Hasher for `u32`-keyed timer/context tables.
///
/// `HashMap`'s default `RandomState` runs SipHash over every key; timer ids
/// are dense `u32`s handed out sequentially, so a single 64-bit multiply is
/// plenty of mixing for the table and far cheaper on the insert/remove/fire
/// hot paths. Std-only, per AGENTS.md.
#[derive(Default)]
pub(crate) struct TimerIdHasher(u64);

impl Hasher for TimerIdHasher {
    #[inline]
    fn write_u32(&mut self, id: u32) {
        // Sequential ids land in distinct buckets after the multiply.
        self.0 = (id as u64).wrapping_mul(0x9E3779B97F4A7C15);
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Fallback; `u32` keys only ever hit `write_u32`.
        let mut h = self.0;
        for &b in bytes {
            h = h.wrapping_mul(0x100000001B3) ^ u64::from(b);
        }
        self.0 = h;
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// `HashMap` keyed by [`TimerId`] (or any dense `u32`) without SipHash.
pub(crate) type TimerIdMap<V> = HashMap<TimerId, V, BuildHasherDefault<TimerIdHasher>>;

/// A unit of work the event loop hands to the host VM for execution.
#[derive(Debug, Clone)]
pub enum Task {
    /// Execute a script source as a macrotask (e.g. the initial page script).
    Script { source: String },
    /// Invoke the timer callback the host registered under `callback`.
    FireTimer { timer_id: TimerId, callback: u32 },
    /// Run one queued promise job. `job` is opaque to the loop; the host
    /// resolves it against its own job table.
    PromiseJob { job: u32 },
}

/// Minimal host interface the event loop drives.
///
/// `vm.rs` (not on disk yet) provides the concrete implementor: it owns the
/// JS function/job tables and executes each [`Task`] against them.
pub trait TaskRunner {
    fn run_task(&mut self, task: Task);
}

/// Millisecond clock backing timer deadlines.
pub trait Clock {
    fn now_ms(&self) -> u64;
}

/// Wall clock for production use.
#[derive(Debug, Clone)]
pub struct RealClock {
    start: Instant,
}

/// Consume one task from an optional task budget: returns `true` when the
/// task may run (`None` = unbounded) and `false` once the budget is
/// exhausted.
fn consume_task_budget(budget: &mut Option<&mut usize>) -> bool {
    match budget {
        None => true,
        Some(remaining) => {
            if **remaining == 0 {
                false
            } else {
                **remaining -= 1;
                true
            }
        }
    }
}

impl RealClock {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}

impl Default for RealClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for RealClock {
    fn now_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

/// Manually advanced clock for deterministic tests.
///
/// Shared by reference (`Rc`), so a test can advance time while the loop
/// holds the same clock.
#[derive(Debug, Clone, Default)]
#[cfg(test)]
pub struct FakeClock {
    now: Cell<u64>,
}

#[cfg(test)]
impl FakeClock {
    pub fn new() -> Self {
        Self::default()
    }

    /// Move the clock forward by `ms`.
    pub fn advance(&self, ms: u64) {
        self.now.set(self.now.get().saturating_add(ms));
    }
}

#[cfg(test)]
impl Clock for FakeClock {
    fn now_ms(&self) -> u64 {
        self.now.get()
    }
}

/// Console output level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleLevel {
    Log,
    Warn,
    Error,
}

/// One timer registration. Heap entries are matched against this table so
/// `clear_timeout` and interval rescheduling are O(1) amortized; stale heap
/// entries are skipped lazily on pop.
#[derive(Debug, Clone)]
struct Timer {
    fire_at_ms: u64,
    delay_ms: u64,
    repeat: bool,
    callback: u32,
}

/// Min-heap entry ordered by `(fire_at_ms, seq)`; `seq` gives FIFO order for
/// timers that share a deadline. `Copy` so callers can copy out of a peek.
/// Stored in the heap as `Reverse<HeapEntry>` because `std::BinaryHeap` is a
/// max-heap and timer firing needs the earliest deadline first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HeapEntry {
    fire_at_ms: u64,
    seq: u64,
    id: TimerId,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Natural order: earlier deadline (then lower seq) is smaller.
        self.fire_at_ms
            .cmp(&other.fire_at_ms)
            .then_with(|| self.seq.cmp(&other.seq))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The event loop: macrotask queue, microtask queue, timer heap.
///
/// Queue design (all `RefCell`, shared via `Rc` with the host):
/// * Macrotasks and microtasks live in plain `Vec`s drained by buffer swap:
///   the queue is swapped with a reusable scratch buffer, the scratch drains
///   with no borrow held (tasks may reentrantly schedule), and the emptied
///   scratch keeps its capacity for the next round. One borrow per batch,
///   no per-round reallocation after warmup.
/// * Timer inserts are O(1): the record goes in `records` and the heap entry
///   is buffered in `pending`. The buffer merges into the heap with a single
///   O(n) heapify the first time a due timer is needed, instead of one
///   O(log n) push per insert; cleared timers are filtered at merge time so
///   they never bloat the heap.
pub struct EventLoop {
    clock: Rc<dyn Clock>,
    macrotasks: RefCell<Vec<Task>>,
    macro_scratch: RefCell<Vec<Task>>,
    microtasks: RefCell<Vec<Task>>,
    micro_scratch: RefCell<Vec<Task>>,
    timers: RefCell<BinaryHeap<Reverse<HeapEntry>>>,
    pending: RefCell<Vec<HeapEntry>>,
    /// Cached minimum `fire_at_ms` over `pending`. Lets `next_due_timer`
    /// skip the pending->heap merge when nothing can be due yet. Updated on
    /// every insert; invalidated (not recomputed) when a clear may have
    /// removed the minimum, and reset when pending is merged. Stale-early is
    /// safe (it only costs a merge); stale-late is impossible because every
    /// insert refreshes it.
    pending_min: Cell<Option<u64>>,
    records: RefCell<TimerIdMap<Timer>>,
    next_timer_id: Cell<TimerId>,
    seq: Cell<u64>,
}

impl EventLoop {
    /// Create a loop driven by the wall clock.
    pub fn new() -> Self {
        Self::with_clock(Rc::new(RealClock::new()))
    }

    /// Create a loop driven by an explicit clock (e.g. [`FakeClock`] in tests).
    pub fn with_clock(clock: Rc<dyn Clock>) -> Self {
        Self {
            clock,
            macrotasks: RefCell::new(Vec::new()),
            macro_scratch: RefCell::new(Vec::new()),
            microtasks: RefCell::new(Vec::new()),
            micro_scratch: RefCell::new(Vec::new()),
            timers: RefCell::new(BinaryHeap::new()),
            pending: RefCell::new(Vec::new()),
            pending_min: Cell::new(None),
            records: RefCell::new(TimerIdMap::default()),
            next_timer_id: Cell::new(1),
            seq: Cell::new(0),
        }
    }

    /// Enqueue a macrotask (script evaluation, I/O completion, ...).
    pub fn queue_macrotask(&self, task: Task) {
        self.macrotasks.borrow_mut().push(task);
    }

    /// Enqueue a microtask (promise job). Runs before the next macrotask.
    pub fn queue_microtask(&self, job: Task) {
        self.microtasks.borrow_mut().push(job);
    }

    /// Enqueue several microtasks with a single queue borrow. Used for
    /// promise fan-out, where one settle can make thousands of jobs
    /// runnable at once.
    pub fn queue_microtasks(&self, jobs: impl IntoIterator<Item = Task>) {
        self.microtasks.borrow_mut().extend(jobs);
    }

    /// Register a one-shot timer. `callback` is the host's opaque function id;
    /// it is echoed back in `Task::FireTimer`. Returns the timer id.
    pub fn set_timeout(&self, callback: u32, delay_ms: u64) -> TimerId {
        self.insert_timer(callback, delay_ms, false)
    }

    /// Register a repeating timer. Same contract as [`EventLoop::set_timeout`].
    pub fn set_interval(&self, callback: u32, delay_ms: u64) -> TimerId {
        self.insert_timer(callback, delay_ms, true)
    }

    /// Cancel a timer created by `set_timeout`/`set_interval`. No-op for
    /// unknown or already-fired ids.
    pub fn clear_timeout(&self, id: TimerId) {
        if let Some(rec) = self.records.borrow_mut().remove(&id) {
            // The pending buffer may still hold this timer's entry; if it
            // was the cached minimum, drop the cache so a stale-early
            // minimum cannot hide behind it (the entry is filtered at merge
            // time, so staleness only ever costs a merge, never a miss).
            if self.pending_min.get().is_some_and(|m| rec.fire_at_ms <= m) {
                self.pending_min.set(None);
            }
        }
    }

    /// True if any macrotask, microtask, or live timer remains. A timer whose
    /// deadline is in the future still counts as pending work.
    pub fn has_pending_work(&self) -> bool {
        !self.macrotasks.borrow().is_empty()
            || !self.microtasks.borrow().is_empty()
            || !self.records.borrow().is_empty()
    }

    /// Run queued microtasks to completion (including ones enqueued while
    /// draining). Must be called after every macrotask.
    pub fn drain_microtasks(&self, runner: &mut impl TaskRunner) {
        self.drain_microtasks_impl(runner, &mut None);
    }

    /// Budgeted microtask drain: like [`drain_microtasks`](Self::drain_microtasks),
    /// but every job counts against `budget`. Returns `false` when the
    /// budget ran out with jobs still pending: the unrun jobs (plus any
    /// queued while running) are spliced back to the front of the
    /// microtask queue in FIFO order, so no job is lost or reordered.
    fn drain_microtasks_impl(
        &self,
        runner: &mut impl TaskRunner,
        budget: &mut Option<&mut usize>,
    ) -> bool {
        // Fast path: the common case (nothing queued) is a single borrow.
        if self.microtasks.borrow().is_empty() {
            return true;
        }
        // Swap-drain against a reusable scratch buffer, taken once and
        // recycled once: one borrow pair per batch, and both buffers keep
        // their capacity across drains instead of reallocating every round.
        // The batch is a local, so no borrow is held while a task runs: tasks
        // queued *during* the drain land in the fresh queue, which the loop
        // picks up next (FIFO order preserved), and reentrant drains cannot
        // trip the `RefCell`.
        let mut batch = std::mem::take(&mut *self.micro_scratch.borrow_mut());
        loop {
            {
                let mut queue = self.microtasks.borrow_mut();
                std::mem::swap(&mut *queue, &mut batch);
            }
            if batch.is_empty() {
                break;
            }
            let mut jobs = batch.drain(..);
            let mut exhausted = false;
            while let Some(job) = jobs.next() {
                if !consume_task_budget(budget) {
                    self.requeue_microtask_remainder(Some(job), &mut jobs);
                    exhausted = true;
                    break;
                }
                runner.run_task(job);
            }
            drop(jobs);
            if exhausted {
                // Recycle the (now empty) buffer before the early return.
                *self.micro_scratch.borrow_mut() = batch;
                return false;
            }
        }
        // Recycle the buffer; it keeps its capacity for next time.
        *self.micro_scratch.borrow_mut() = batch;
        true
    }

    /// Splice `head` plus the unrun `jobs` back to the front of the
    /// microtask queue, ahead of anything queued while jobs ran,
    /// preserving FIFO order. Used when a task budget runs out mid-drain.
    fn requeue_microtask_remainder(
        &self,
        head: Option<Task>,
        jobs: &mut std::vec::Drain<'_, Task>,
    ) {
        let mut queue = self.microtasks.borrow_mut();
        let mut queued_during = std::mem::take(&mut *queue);
        queue.extend(head);
        queue.extend(jobs.by_ref());
        queue.append(&mut queued_during);
    }

    /// Splice `head` plus the unrun `tasks` back to the front of the
    /// macrotask queue, ahead of anything queued while tasks ran,
    /// preserving FIFO order. Used when a task budget runs out mid-batch.
    fn requeue_macrotask_remainder(
        &self,
        head: Option<Task>,
        tasks: &mut std::vec::Drain<'_, Task>,
    ) {
        let mut queue = self.macrotasks.borrow_mut();
        let mut queued_during = std::mem::take(&mut *queue);
        queue.extend(head);
        queue.extend(tasks.by_ref());
        queue.append(&mut queued_during);
    }

    /// Drive the loop until no macrotask, microtask, or due timer remains.
    /// Timers whose deadlines are in the future are left pending; the loop
    /// never blocks. After each macrotask, microtasks are drained fully.
    pub fn run_until_idle(&self, runner: &mut impl TaskRunner) {
        self.run_impl(runner, None);
    }

    /// Drive the loop like [`run_until_idle`](Self::run_until_idle), but run
    /// at most `max_tasks` tasks: every macrotask, due timer fire, and
    /// microtask counts against the budget.
    ///
    /// Returns `true` when the loop drained fully — no macrotask,
    /// microtask, or due timer remains — and `false` when the budget ran
    /// out first with work still pending. A page that keeps scheduling due
    /// work (a `setInterval` with a 0 ms delay, an always-requeuing
    /// microtask) can never spin this forever: the caller observes
    /// `false` and decides whether to pump again.
    pub fn run_with_task_budget(&self, runner: &mut impl TaskRunner, max_tasks: usize) -> bool {
        let mut remaining = max_tasks;
        self.run_impl(runner, Some(&mut remaining))
    }

    /// Shared implementation for [`run_until_idle`](Self::run_until_idle)
    /// (`None` budget: unbounded) and
    /// [`run_with_task_budget`](Self::run_with_task_budget).
    fn run_impl(&self, runner: &mut impl TaskRunner, mut budget: Option<&mut usize>) -> bool {
        loop {
            // Microtasks first, to completion (swap-drained, as above),
            // counting each job against the budget.
            if !self.drain_microtasks_impl(runner, &mut budget) {
                return false;
            }
            // Fast path: no macrotasks -> the next due timer (if any).
            if self.macrotasks.borrow().is_empty() {
                let Some(task) = self.next_due_timer() else {
                    break;
                };
                if !consume_task_budget(&mut budget) {
                    // `next_due_timer` already popped/rescheduled the
                    // timer: a one-shot's record is gone, so requeue its
                    // fire as a macrotask — the callback still runs
                    // (late) on the next pump instead of being dropped.
                    // A repeat timer rescheduled itself in place above
                    // and stays due, so it needs no requeue.
                    let oneshot = matches!(&task, Task::FireTimer { timer_id, .. }
                        if !self.records.borrow().contains_key(timer_id));
                    if oneshot {
                        self.macrotasks.borrow_mut().push(task);
                    }
                    return false;
                }
                runner.run_task(task);
                continue;
            }
            // Batch-swap the macrotask queue: one borrow per batch instead of
            // one per task. Tasks queued during the batch land in the fresh
            // queue and run before any timer, exactly as with per-task pops,
            // and no borrow is held while a task runs.
            let mut batch = std::mem::take(&mut *self.macro_scratch.borrow_mut());
            {
                let mut queue = self.macrotasks.borrow_mut();
                std::mem::swap(&mut *queue, &mut batch);
            }
            debug_assert!(!batch.is_empty());
            let mut tasks = batch.drain(..);
            let mut exhausted = false;
            while let Some(task) = tasks.next() {
                if !consume_task_budget(&mut budget) {
                    self.requeue_macrotask_remainder(Some(task), &mut tasks);
                    exhausted = true;
                    break;
                }
                runner.run_task(task);
                // Promise jobs run to completion before the next macrotask.
                if !self.drain_microtasks_impl(runner, &mut budget) {
                    self.requeue_macrotask_remainder(None, &mut tasks);
                    exhausted = true;
                    break;
                }
            }
            drop(tasks);
            *self.macro_scratch.borrow_mut() = batch;
            if exhausted {
                return false;
            }
        }
        true
    }

    /// `console.log` / `console.warn` / `console.error` sink.
    /// `log` -> stdout, `warn`/`error` -> stderr, args joined with spaces.
    /// Streams through one locked handle: no intermediate `String` and no
    /// per-argument `to_string()` allocation.
    pub fn console_log(&self, level: ConsoleLevel, args: &[impl Display]) {
        use std::io::Write as _;
        match level {
            ConsoleLevel::Log => {
                let mut out = std::io::stdout().lock();
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        let _ = write!(out, " ");
                    }
                    let _ = write!(out, "{arg}");
                }
                let _ = writeln!(out);
            }
            ConsoleLevel::Warn | ConsoleLevel::Error => {
                let mut out = std::io::stderr().lock();
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        let _ = write!(out, " ");
                    }
                    let _ = write!(out, "{arg}");
                }
                let _ = writeln!(out);
            }
        }
    }

    fn insert_timer(&self, callback: u32, delay_ms: u64, repeat: bool) -> TimerId {
        let id = self.next_timer_id.get();
        self.next_timer_id.set(id.wrapping_add(1).max(1));
        let fire_at_ms = self.clock.now_ms().saturating_add(delay_ms);
        let seq = self.seq.get();
        self.seq.set(seq + 1);
        // O(1): the heap entry is buffered in `pending` and merged lazily by
        // `next_due_timer`, so bulk scheduling never pays O(log n) per timer.
        let mut records = self.records.borrow_mut();
        let mut pending = self.pending.borrow_mut();
        records.insert(
            id,
            Timer {
                fire_at_ms,
                delay_ms,
                repeat,
                callback,
            },
        );
        pending.push(HeapEntry {
            fire_at_ms,
            seq,
            id,
        });
        // Every insert refreshes the cached minimum, so it can never go
        // stale-late (which would hide a due timer from `next_due_timer`).
        if self.pending_min.get().is_none_or(|m| fire_at_ms < m) {
            self.pending_min.set(Some(fire_at_ms));
        }
        id
    }

    /// Cached minimum deadline over the pending insert buffer, recomputed
    /// by scan when the cache is cold (only right after a clear
    /// invalidated it; inserts always keep it warm).
    fn pending_min_cached(&self) -> Option<u64> {
        if let Some(m) = self.pending_min.get() {
            return Some(m);
        }
        let pending = self.pending.borrow();
        if pending.is_empty() {
            return None;
        }
        let m = pending.iter().map(|e| e.fire_at_ms).min();
        self.pending_min.set(m);
        m
    }

    fn next_due_timer(&self) -> Option<Task> {
        // Fast path: compare the heap top against the cached pending
        // minimum before paying for a merge. If the earliest deadline on
        // either side is still in the future, no timer can be due and the
        // merge is pure overhead. This is the common case on a page full
        // of far-future timers (e.g. polling intervals): each pump is a
        // couple of borrows and a clock read instead of an O(n) heapify.
        let heap_min = self.timers.borrow().peek().map(|e| e.0.fire_at_ms);
        let pend_min = self.pending_min_cached();
        if heap_min.is_none() && pend_min.is_none() {
            // Nothing scheduled at all: skip the clock call.
            return None;
        }
        let now = self.clock.now_ms();
        if heap_min
            .into_iter()
            .chain(pend_min)
            .min()
            .is_some_and(|m| m > now)
        {
            return None;
        }
        // Merge lazily-buffered inserts; cleared timers are filtered here
        // so they never bloat the heap. The merge is adaptive: a small
        // batch into a large heap is pushed entry by entry (O(k log n)),
        // while a large batch gets one O(n) heapify.
        let pending = std::mem::take(&mut *self.pending.borrow_mut());
        self.pending_min.set(None);
        if !pending.is_empty() {
            let batch = pending.len();
            let records = self.records.borrow();
            let mut timers = self.timers.borrow_mut();
            let live = pending
                .into_iter()
                .filter(|e| records.contains_key(&e.id))
                .map(Reverse);
            if batch * 8 < timers.len() {
                timers.extend(live);
            } else {
                let mut vec = std::mem::take(&mut *timers).into_vec();
                vec.extend(live);
                *timers = BinaryHeap::from(vec);
            }
        }
        let mut timers = self.timers.borrow_mut();
        if timers.is_empty() {
            // Merge filtered every pending entry (all cleared since).
            return None;
        }
        // Hold both borrows for the whole scan: no per-entry re-borrow,
        // and stale entries are discarded without cloning the heap top.
        let mut records = self.records.borrow_mut();
        loop {
            let (id, fire_at_ms) = {
                let entry = timers.peek()?.0;
                (entry.id, entry.fire_at_ms)
            };
            // One hash-table op per fire via the entry API. `clear_timeout`
            // removes the record (vacant entry), and interval reschedules
            // leave stale heap entries whose deadline no longer matches the
            // record (mismatch).
            let mut slot = match records.entry(id) {
                Entry::Occupied(slot) => slot,
                Entry::Vacant(_) => {
                    timers.pop();
                    continue;
                }
            };
            if slot.get().fire_at_ms != fire_at_ms {
                // Defensive: interval rescheduling below updates the heap
                // entry in place, so entries always mirror their record and
                // only `clear_timeout` can stale one.
                timers.pop();
                continue;
            }
            if fire_at_ms > now {
                return None;
            }
            let (callback, delay_ms, repeat) = {
                let rec = slot.get();
                (rec.callback, rec.delay_ms, rec.repeat)
            };
            if repeat {
                let next_fire = fire_at_ms.saturating_add(delay_ms);
                let seq = self.seq.get();
                self.seq.set(seq + 1);
                // Reschedule in place: the deadline only moves forward, so
                // one sift-down on drop beats pop + push.
                if let Some(mut top) = timers.peek_mut() {
                    debug_assert_eq!(top.0.id, id);
                    top.0.fire_at_ms = next_fire;
                    top.0.seq = seq;
                }
                slot.get_mut().fire_at_ms = next_fire;
            } else {
                timers.pop();
                slot.remove_entry();
            }
            return Some(Task::FireTimer {
                timer_id: id,
                callback,
            });
        }
    }
}

impl Default for EventLoop {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    /// Records fired tasks; on timer callback 0 it schedules a nested timer
    /// (callback 99) 5ms out, and on script "enqueue-micro" it queues a
    /// microtask. Shares the loop via `Rc` since the runner runs while the
    /// loop drives it.
    struct Recorder {
        log: Rc<RefCell<Vec<String>>>,
        el: Rc<EventLoop>,
    }

    impl Recorder {
        fn new(el: &Rc<EventLoop>) -> (Self, Rc<RefCell<Vec<String>>>) {
            let log = Rc::new(RefCell::new(Vec::new()));
            (
                Self {
                    log: Rc::clone(&log),
                    el: Rc::clone(el),
                },
                log,
            )
        }
    }

    impl TaskRunner for Recorder {
        fn run_task(&mut self, task: Task) {
            match task {
                Task::FireTimer {
                    timer_id: _,
                    callback,
                } => {
                    self.log.borrow_mut().push(format!("t{callback}"));
                    if callback == 0 {
                        self.el.set_timeout(99, 5);
                    }
                }
                Task::PromiseJob { job } => {
                    self.log.borrow_mut().push(format!("m{job}"));
                }
                Task::Script { source } => {
                    self.log.borrow_mut().push(format!("s:{source}"));
                    if source == "enqueue-micro" {
                        self.el.queue_microtask(Task::PromiseJob { job: 7 });
                    }
                }
            }
        }
    }

    fn harness() -> (Rc<EventLoop>, Rc<FakeClock>) {
        let clock = Rc::new(FakeClock::new());
        let el = Rc::new(EventLoop::with_clock(Rc::clone(&clock) as Rc<dyn Clock>));
        (el, clock)
    }

    #[test]
    fn timers_fire_in_deadline_order() {
        let (el, clock) = harness();
        el.set_timeout(2, 20);
        el.set_timeout(1, 10);
        el.set_timeout(3, 10); // same deadline as 1: FIFO
        let (mut runner, log) = Recorder::new(&el);

        clock.advance(10);
        el.run_until_idle(&mut runner);
        assert_eq!(*log.borrow(), vec!["t1", "t3"]);

        clock.advance(10);
        el.run_until_idle(&mut runner);
        assert_eq!(*log.borrow(), vec!["t1", "t3", "t2"]);
        assert!(!el.has_pending_work());
    }

    #[test]
    fn nested_set_timeout_runs_after_advance() {
        let (el, clock) = harness();
        el.set_timeout(0, 0);
        let (mut runner, log) = Recorder::new(&el);

        el.run_until_idle(&mut runner);
        // Callback 0 scheduled callback 99 due 5ms out; not due yet.
        assert_eq!(*log.borrow(), vec!["t0"]);
        assert!(el.has_pending_work());

        clock.advance(5);
        el.run_until_idle(&mut runner);
        assert_eq!(*log.borrow(), vec!["t0", "t99"]);
        assert!(!el.has_pending_work());
    }

    #[test]
    fn microtask_queued_by_macrotask_runs_before_next_macrotask() {
        let (el, _clock) = harness();
        el.queue_macrotask(Task::Script {
            source: "enqueue-micro".into(),
        });
        el.queue_macrotask(Task::Script {
            source: "plain".into(),
        });
        let (mut runner, log) = Recorder::new(&el);

        el.run_until_idle(&mut runner);
        assert_eq!(*log.borrow(), vec!["s:enqueue-micro", "m7", "s:plain"]);
    }

    #[test]
    fn microtask_queued_directly_runs_before_macrotask() {
        let (el, _clock) = harness();
        el.queue_macrotask(Task::Script {
            source: "plain".into(),
        });
        el.queue_microtask(Task::PromiseJob { job: 1 });
        let (mut runner, log) = Recorder::new(&el);

        el.run_until_idle(&mut runner);
        assert_eq!(*log.borrow(), vec!["m1", "s:plain"]);
    }

    #[test]
    fn clear_timeout_cancels_pending_timer() {
        let (el, clock) = harness();
        let cancelled = el.set_timeout(1, 10);
        el.set_timeout(2, 10);
        el.clear_timeout(cancelled);
        el.clear_timeout(9999); // unknown id: no-op
        let (mut runner, log) = Recorder::new(&el);

        clock.advance(50);
        el.run_until_idle(&mut runner);
        assert_eq!(*log.borrow(), vec!["t2"]);
        assert!(!el.has_pending_work());
    }

    #[test]
    fn interval_repeats_until_cleared() {
        let (el, clock) = harness();
        let id = el.set_interval(7, 10);
        let (mut runner, log) = Recorder::new(&el);

        clock.advance(35);
        el.run_until_idle(&mut runner);
        assert_eq!(*log.borrow(), vec!["t7", "t7", "t7"]);
        assert!(el.has_pending_work());

        el.clear_timeout(id);
        clock.advance(1000);
        el.run_until_idle(&mut runner);
        assert_eq!(*log.borrow(), vec!["t7", "t7", "t7"]);
        assert!(!el.has_pending_work());
    }

    #[test]
    fn run_until_idle_does_not_wait_for_future_timers() {
        let (el, _clock) = harness();
        el.set_timeout(5, 100);
        let (mut runner, log) = Recorder::new(&el);

        el.run_until_idle(&mut runner);
        assert!(log.borrow().is_empty());
        assert!(el.has_pending_work());
    }

    /// A runner that requeues a microtask on every run: unbounded, the
    /// loop spins forever.
    struct RequeueMicrotask {
        el: Rc<EventLoop>,
        ran: usize,
    }

    impl TaskRunner for RequeueMicrotask {
        fn run_task(&mut self, _task: Task) {
            self.ran += 1;
            self.el.queue_microtask(Task::PromiseJob { job: 42 });
        }
    }

    #[test]
    fn task_budget_terminates_spinning_microtasks() {
        let (el, _clock) = harness();
        el.queue_microtask(Task::PromiseJob { job: 7 });
        let mut runner = RequeueMicrotask {
            el: Rc::clone(&el),
            ran: 0,
        };
        // The requeueing job would spin `run_until_idle` forever; the
        // budget stops it after exactly `max_tasks` tasks.
        assert!(!el.run_with_task_budget(&mut runner, 10));
        assert_eq!(runner.ran, 10);
        assert!(el.has_pending_work());
        // A second budgeted run picks up where the first stopped: no job
        // was lost or reordered by the budget cut.
        let mut runner = RequeueMicrotask {
            el: Rc::clone(&el),
            ran: 0,
        };
        assert!(!el.run_with_task_budget(&mut runner, 5));
        assert_eq!(runner.ran, 5);
    }

    #[test]
    fn task_budget_counts_timers_and_preserves_macrotask_fifo() {
        let (el, clock) = harness();
        // A 0 ms interval reschedules immediately-due: unbounded, the
        // timer branch of the loop spins forever.
        el.set_interval(3, 0);
        for i in 0..3 {
            el.queue_macrotask(Task::Script {
                source: format!("m{i}"),
            });
        }
        clock.advance(1);
        let (mut runner, log) = Recorder::new(&el);
        // 3 macrotasks + timer fires, all counted.
        assert!(!el.run_with_task_budget(&mut runner, 4));
        assert_eq!(*log.borrow(), vec!["s:m0", "s:m1", "s:m2", "t3"]);
        assert!(el.has_pending_work());
        // The interval is still live: another budgeted run fires it again.
        let (mut runner, log) = Recorder::new(&el);
        assert!(!el.run_with_task_budget(&mut runner, 1));
        assert_eq!(*log.borrow(), vec!["t3"]);
    }

    #[test]
    fn task_budget_does_not_drop_due_oneshot_timer() {
        let (el, _clock) = harness();
        el.set_timeout(5, 0); // one-shot, due now
        let (mut runner, log) = Recorder::new(&el);
        // Budget 0: nothing may run, but the due timer must survive the
        // cut instead of being dropped with its popped record.
        assert!(!el.run_with_task_budget(&mut runner, 0));
        assert!(log.borrow().is_empty());
        assert!(el.has_pending_work());
        // Next pump: the timer fires exactly once.
        assert!(el.run_with_task_budget(&mut runner, 10));
        assert_eq!(*log.borrow(), vec!["t5"]);
        assert!(!el.has_pending_work());
    }

    #[test]
    fn task_budget_zero_runs_nothing_but_reports_idle_when_empty() {
        let (el, _clock) = harness();
        let (mut runner, log) = Recorder::new(&el);
        // No work queued: idle, even with a zero budget.
        assert!(el.run_with_task_budget(&mut runner, 0));
        // Work queued: nothing runs, not drained.
        el.queue_microtask(Task::PromiseJob { job: 1 });
        assert!(!el.run_with_task_budget(&mut runner, 0));
        assert!(log.borrow().is_empty());
        assert!(el.has_pending_work());
        // The queued job survived the zero-budget run untouched.
        assert!(el.run_with_task_budget(&mut runner, 1));
        assert_eq!(*log.borrow(), vec!["m1"]);
        assert!(!el.has_pending_work());
    }

    #[test]
    fn console_levels_compile_and_accept_args() {
        let (el, _clock) = harness();
        el.console_log(ConsoleLevel::Log, &["hello", "42"]);
        el.console_log(ConsoleLevel::Warn, &["careful"]);
        el.console_log(ConsoleLevel::Error, &["boom"]);
    }

    #[test]
    fn timer_heap_pops_in_deadline_then_fifo_order() {
        // Deterministic xorshift: total order via unique `seq`, so the pop
        // sequence must exactly match a sort by the same `Ord`. The heap
        // stores `Reverse<HeapEntry>` (max-heap used as a min-heap), so the
        // reference sort is over the wrapped values too.
        let mut state = 0x1234_5678_9ABCu64;
        let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();
        assert!(heap.is_empty());
        assert_eq!(heap.pop(), None);
        assert_eq!(heap.peek(), None);
        let mut entries: Vec<HeapEntry> = Vec::new();
        for i in 0..10_000u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let e = HeapEntry {
                fire_at_ms: state % 500,
                seq: i,
                id: i as u32,
            };
            entries.push(e);
            heap.push(Reverse(e));
        }
        assert_eq!(heap.peek().map(|r| r.0), entries.iter().min().copied());
        entries.sort();
        let mut prev: Option<HeapEntry> = None;
        for expected in entries {
            let got = heap.pop().expect("heap drained early").0;
            assert_eq!(got, expected);
            if let Some(p) = prev {
                assert!(p <= got, "heap popped out of order");
            }
            prev = Some(got);
        }
        assert!(heap.is_empty());
        assert_eq!(heap.pop(), None);
    }

    // ------------------------------------------------------------------
    // Scheduling-throughput benchmarks (run with
    // `cargo test --release -p bowser_js event_loop:: -- --nocapture`).
    // ------------------------------------------------------------------

    struct NopRunner;

    impl TaskRunner for NopRunner {
        fn run_task(&mut self, _task: Task) {}
    }

    /// Warm up once, then report best-of-5 throughput for `ops` operations.
    fn bench_throughput(name: &str, ops: u64, f: impl Fn()) {
        f(); // warmup: page in code, grow allocator caches
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            f();
            best = best.min(t.elapsed().as_secs_f64());
        }
        let per_s = ops as f64 / best;
        println!("[bench] {name}: {per_s:.0} ops/s");
        black_box(per_s);
    }

    #[test]
    fn bench_timer_schedule_and_fire() {
        const N: u64 = 100_000;
        bench_throughput("timers: schedule+fire 100k", N, || {
            let clock = Rc::new(FakeClock::new());
            let el = EventLoop::with_clock(Rc::clone(&clock) as Rc<dyn Clock>);
            for i in 0..N {
                el.set_timeout((i & 0xffff) as u32, i % 1000);
            }
            clock.advance(1000);
            el.run_until_idle(&mut NopRunner);
            assert!(!el.has_pending_work());
        });
    }

    #[test]
    fn bench_timer_schedule_and_clear() {
        const N: u64 = 100_000;
        bench_throughput("timers: schedule+clear 100k", N, || {
            let el = EventLoop::new();
            let mut ids = Vec::with_capacity(N as usize);
            for i in 0..N {
                ids.push(el.set_timeout((i & 0xffff) as u32, i));
            }
            for id in ids.drain(..) {
                el.clear_timeout(id);
            }
            assert!(!el.has_pending_work());
        });
    }

    #[test]
    fn bench_microtask_drain() {
        const N: u64 = 500_000;
        bench_throughput("microtasks: drain 500k", N, || {
            let el = EventLoop::new();
            for i in 0..N {
                el.queue_microtask(Task::PromiseJob {
                    job: (i & 0xffff) as u32,
                });
            }
            el.drain_microtasks(&mut NopRunner);
            assert!(!el.has_pending_work());
        });
    }

    #[test]
    fn bench_macrotask_run_until_idle() {
        const N: u64 = 100_000;
        bench_throughput("macrotasks: run_until_idle 100k", N, || {
            let el = EventLoop::new();
            for _ in 0..N {
                el.queue_macrotask(Task::PromiseJob { job: 1 });
            }
            el.run_until_idle(&mut NopRunner);
        });
    }

    #[test]
    fn bench_macrotask_enqueues_microtask() {
        const N: u64 = 100_000;

        struct Churn<'a> {
            el: &'a EventLoop,
        }

        impl TaskRunner for Churn<'_> {
            fn run_task(&mut self, task: Task) {
                if matches!(task, Task::Script { .. }) {
                    self.el.queue_microtask(Task::PromiseJob { job: 1 });
                }
            }
        }

        bench_throughput("mixed: 100k macrotask->microtask", N, || {
            let el = EventLoop::new();
            for _ in 0..N {
                el.queue_macrotask(Task::Script {
                    source: String::new(),
                });
            }
            el.run_until_idle(&mut Churn { el: &el });
        });
    }

    #[test]
    fn bench_interval_refire() {
        const N: u64 = 20_000;
        bench_throughput("interval: 20k refires", N, || {
            let clock = Rc::new(FakeClock::new());
            let el = EventLoop::with_clock(Rc::clone(&clock) as Rc<dyn Clock>);
            el.set_interval(1, 1);
            for _ in 0..N {
                clock.advance(1);
                el.run_until_idle(&mut NopRunner);
            }
            assert!(el.has_pending_work());
        });
    }

    #[test]
    fn bench_timer_steady_state_churn() {
        // 50k live far-future timers; each round schedules one immediate
        // timer and fires it. Exercises the pending->heap merge under
        // steady state: a trickle of inserts into a large live heap.
        const LIVE: u64 = 50_000;
        const ROUNDS: u64 = 1_000;
        bench_throughput("timers: steady-state churn 1k", ROUNDS, || {
            let clock = Rc::new(FakeClock::new());
            let el = EventLoop::with_clock(Rc::clone(&clock) as Rc<dyn Clock>);
            for i in 0..LIVE {
                el.set_timeout((i & 0xffff) as u32, 1_000_000);
            }
            for _ in 0..ROUNDS {
                el.set_timeout(1, 0);
                el.run_until_idle(&mut NopRunner);
            }
            assert!(el.has_pending_work());
        });
    }

    #[test]
    fn bench_timer_future_pump() {
        // Many far-future timers, pumped with nothing due: the loop must
        // not pay the pending->heap merge until a timer is actually due.
        const N: u64 = 50_000;
        const PUMPS: u64 = 200;
        bench_throughput("timers: pump 50k future x200", PUMPS, || {
            let clock = Rc::new(FakeClock::new());
            let el = EventLoop::with_clock(Rc::clone(&clock) as Rc<dyn Clock>);
            for i in 0..N {
                el.set_timeout((i & 0xffff) as u32, 1_000_000);
            }
            for _ in 0..PUMPS {
                el.run_until_idle(&mut NopRunner);
            }
            assert!(el.has_pending_work());
        });
    }

    // ------------------------------------------------------------------
    // Correctness tests for the timer/microtask optimizations.
    // ------------------------------------------------------------------

    /// Far-future timers must not pay the pending->heap merge while
    /// nothing is due: after idle pumps the heap stays empty and the
    /// entries sit in the pending buffer; they still fire once time
    /// advances past their deadline.
    #[test]
    fn timer_future_pending_not_merged() {
        let clock = Rc::new(FakeClock::new());
        let el = EventLoop::with_clock(Rc::clone(&clock) as Rc<dyn Clock>);
        el.set_timeout(7, 1_000_000);
        el.set_timeout(8, 2_000_000);
        for _ in 0..10 {
            el.run_until_idle(&mut NopRunner);
        }
        assert!(el.timers.borrow().is_empty());
        assert_eq!(el.pending.borrow().len(), 2);
        assert_eq!(el.pending_min.get(), Some(1_000_000));
        clock.advance(1_000_000);
        let task = el.next_due_timer().expect("timer 7 is now due");
        assert!(matches!(task, Task::FireTimer { callback: 7, .. }));
    }

    /// A due timer sitting in the pending buffer must beat a far-future
    /// timer already in the heap: the merge must happen before the heap
    /// top is trusted.
    #[test]
    fn timer_due_pending_beats_future_heap() {
        let clock = Rc::new(FakeClock::new());
        let el = EventLoop::with_clock(Rc::clone(&clock) as Rc<dyn Clock>);
        el.set_timeout(1, 1_000_000);
        el.set_timeout(2, 0);
        // Fires timer 2 and merges timer 1 into the heap.
        let task = el.next_due_timer().expect("timer 2 is due");
        assert!(matches!(task, Task::FireTimer { callback: 2, .. }));
        assert_eq!(el.timers.borrow().len(), 1);
        // A new due timer lands in pending while the heap top is far off.
        el.set_timeout(3, 0);
        let task = el.next_due_timer().expect("timer 3 is due");
        assert!(matches!(task, Task::FireTimer { callback: 3, .. }));
    }

    /// Clearing the timer that supplied the cached pending minimum must
    /// invalidate the cache: the cleared timer must never fire, and a
    /// later timer must still fire on time.
    #[test]
    fn timer_clear_drops_cached_pending_min() {
        let clock = Rc::new(FakeClock::new());
        let el = EventLoop::with_clock(Rc::clone(&clock) as Rc<dyn Clock>);
        let id1 = el.set_timeout(1, 1_000);
        assert_eq!(el.pending_min.get(), Some(1_000));
        el.clear_timeout(id1);
        assert_eq!(el.pending_min.get(), None);
        el.set_timeout(2, 2_000);
        clock.advance(1_500);
        assert!(el.next_due_timer().is_none());
        clock.advance(500);
        let task = el.next_due_timer().expect("timer 2 is due at t=2000");
        assert!(matches!(task, Task::FireTimer { callback: 2, .. }));
        assert!(!el.has_pending_work());
    }

    /// Microtask drain runs FIFO across batches, picks up tasks enqueued
    /// during the drain, and tolerates a reentrant drain without
    /// tripping the `RefCell` (no borrow is held while a task runs).
    #[test]
    fn microtask_drain_fifo_and_reentrant() {
        struct Fifo<'a> {
            el: &'a EventLoop,
            log: Vec<u32>,
            reentered: bool,
        }

        impl TaskRunner for Fifo<'_> {
            fn run_task(&mut self, task: Task) {
                if let Task::PromiseJob { job } = task {
                    self.log.push(job);
                    if job == 1 {
                        self.el.queue_microtask(Task::PromiseJob { job: 10 });
                        self.el.queue_microtask(Task::PromiseJob { job: 11 });
                    }
                    if job == 2 && !self.reentered {
                        self.reentered = true;
                        self.el.drain_microtasks(self);
                    }
                }
            }
        }

        let el = EventLoop::new();
        el.queue_microtask(Task::PromiseJob { job: 1 });
        el.queue_microtask(Task::PromiseJob { job: 2 });
        el.queue_microtask(Task::PromiseJob { job: 3 });
        let mut runner = Fifo {
            el: &el,
            log: Vec::new(),
            reentered: false,
        };
        el.drain_microtasks(&mut runner);
        // Job 2's reentrant drain eagerly ran 10 and 11; job 3 ran after.
        assert_eq!(runner.log, vec![1, 2, 10, 11, 3]);
        assert!(runner.reentered);
        assert!(!el.has_pending_work());
    }
}
