//! Process-wide DNS cache backing [`crate::fetch`] dials.
//!
//! Standard-library Rust only. See /AGENTS.md.
//!
//! Every TCP connect goes through `dial_host`, which funnels DNS lookups
//! here, so direct connects and proxy dials share one cache — matching the
//! process-wide connection [`crate::pool::Pool`]. std exposes no DNS TTLs,
//! so positive entries live for [`POSITIVE_TTL`] and failures for the much
//! shorter [`NEGATIVE_TTL`] (enough to avoid hammering the resolver during
//! an outage, short enough to recover quickly).
//!
//! Concurrent lookups for the same name are single-flighted: the first
//! thread resolves while the rest wait on the in-flight slot and share
//! its answer, so a burst of connects to one host issues exactly one
//! query.
//!
//! DNS resolution is time-bounded: the leader runs the real lookup on a
//! helper thread and waits at most [`DNS_TIMEOUT`], so a blackholed
//! nameserver or a broken `/etc/resolv.conf` can no longer wedge the
//! dial threads. A timed-out lookup fails with
//! [`io::ErrorKind::TimedOut`], is negative-cached for [`NEGATIVE_TTL`],
//! and its followers get the timeout instead of hanging.
//!
//! Cached answers are bare [`IpAddr`]s keyed by host, with no port: the
//! dial applies the request's port at connect time, so one cached answer
//! serves `http`, `https`, and explicit ports alike, and a port resolved
//! for one scheme can never leak into another scheme's dial.
//!
//! Each cached answer lives in one shared [`Arc`]: the cache entry, every
//! cache hit, and every single-flight follower point at the same
//! allocation, so a hit clones the `Arc` (a refcount bump) instead of
//! reallocating the address list.
//!
//! Poisoning resistance: only successful, non-empty resolutions are cached
//! as positive entries. Failures are never cached as answers — a negative
//! entry only replays the same error for its brief lifetime, and an empty
//! resolution is never cached at all.
//!
//! [`DnsCache::invalidate`] drops every entry and bumps the cache
//! generation; it is the hook a future network-change monitor calls so
//! stale answers never survive an interface or VPN change. Lookups in
//! flight across the invalidation resolved against the old network, so
//! their answers are never published and their leader and followers get
//! an invalidation error instead — everyone retries against the new
//! network.

use std::collections::HashMap;
use std::io;
use std::net::IpAddr;
use std::sync::{Arc, Condvar, Mutex, OnceLock, mpsc};
use std::thread;
use std::time::{Duration, Instant};

/// How long a successful resolution stays cached. Chosen as a middle
/// ground: most DNS TTLs are in the minutes range, and a stale entry only
/// costs a failed Happy-Eyeballs attempt (the race falls back to other
/// cached addresses and re-resolves on the next miss).
pub(crate) const POSITIVE_TTL: Duration = Duration::from_secs(60);

/// How long a failed resolution suppresses retries. Short on purpose: DNS
/// outages are usually transient, and a stale failure is worse than one
/// extra lookup.
pub(crate) const NEGATIVE_TTL: Duration = Duration::from_secs(5);

/// Upper bound on one DNS resolution. std's resolver (`ToSocketAddrs`)
/// exposes no timeout knob: a blackholed nameserver or a broken
/// `/etc/resolv.conf` can stall it indefinitely, so the leader runs the
/// lookup on a helper thread and this bounds the wait instead. On expiry
/// the lookup fails with [`io::ErrorKind::TimedOut`], the failure is
/// negative-cached for [`NEGATIVE_TTL`], and waiting followers get the
/// timeout instead of hanging. Five seconds mirrors the glibc per-query
/// default: a working resolver answers far faster, and a broken one must
/// not wedge the dial threads.
pub(crate) const DNS_TIMEOUT: Duration = Duration::from_secs(5);

/// Poll slice while waiting for the DNS helper thread: short enough that
/// an [`DnsCache::invalidate`] retires a hung lookup within a blink,
/// long enough not to spin the polling thread. Mirrors the 50 ms slice
/// of `fetch`'s interruptible dial; kept local so the two stay decoupled.
const DNS_POLL: Duration = Duration::from_millis(50);

/// Upper bound on cached names. DNS answers are small, but without a cap
/// a hostile redirect chain could grow the map forever.
const MAX_ENTRIES: usize = 1024;

/// One cached answer for a host name.
enum DnsEntry {
    /// A successful resolution: the shared address slice and its
    /// deadline. Bare [`IpAddr`]s, no ports — the dial applies the
    /// request's port, so an answer cached for `http` can never leak
    /// port 80 into an `https` dial (or vice versa).
    ///
    /// The addresses live in one shared [`Arc`]: the cache entry, every
    /// cache hit, and every single-flight follower point at the same
    /// allocation. A hit clones the `Arc` (a refcount bump), never the
    /// address list.
    Positive {
        addrs: Arc<[IpAddr]>,
        expires: Instant,
    },
    /// A failed resolution: replay the same error until it expires.
    Negative {
        error: SharedError,
        expires: Instant,
    },
}

impl DnsEntry {
    fn expired(&self, now: Instant) -> bool {
        let expires = match self {
            DnsEntry::Positive { expires, .. } => *expires,
            DnsEntry::Negative { expires, .. } => *expires,
        };
        now >= expires
    }
}

/// Cloneable stand-in for `io::Error`: `io::Error` is not `Clone`, so
/// cached failures and in-flight results store the kind plus message and
/// rebuild the error on replay.
#[derive(Clone)]
struct SharedError {
    kind: io::ErrorKind,
    message: String,
}

impl From<&io::Error> for SharedError {
    fn from(e: &io::Error) -> Self {
        SharedError {
            kind: e.kind(),
            message: e.to_string(),
        }
    }
}

impl SharedError {
    fn into_io(self) -> io::Error {
        io::Error::new(self.kind, self.message)
    }
}

/// Error for lookups whose resolution straddled a
/// [`DnsCache::invalidate`]: the answer was resolved against the old
/// network, so it must neither enter the cache nor reach a dial.
/// Recipients retry and resolve against the new network.
fn invalidated_error() -> io::Error {
    io::Error::other("dns lookup invalidated by network change")
}

/// One in-flight resolution. The leader fills `result` and wakes every
/// follower; followers clone the shared [`Arc`] instead of taking it, so
/// any number of waiters can share one slot — and one allocation.
struct InflightSlot {
    result: Mutex<Option<Result<Arc<[IpAddr]>, SharedError>>>,
    cvar: Condvar,
}

impl InflightSlot {
    fn new() -> Self {
        InflightSlot {
            result: Mutex::new(None),
            cvar: Condvar::new(),
        }
    }

    /// Publish the leader's answer and wake every follower. Lock
    /// poisoning is recovered, not fatal: a panic elsewhere must not
    /// wedge every later lookup (see `unwrap_or_else` posture below).
    fn finish(&self, result: Result<Arc<[IpAddr]>, SharedError>) {
        *self.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
        self.cvar.notify_all();
    }

    /// Block until the leader publishes, then return its answer. The
    /// answer is the shared [`Arc`], so this is a refcount bump, not a
    /// reallocation of the address list.
    fn wait(&self) -> io::Result<Arc<[IpAddr]>> {
        // Poisoned locks are recovered via `into_inner`: a thread that
        // died holding the slot mutex must not wedge every follower.
        let mut guard = self.result.lock().unwrap_or_else(|e| e.into_inner());
        while guard.is_none() {
            guard = self.cvar.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
        match guard.as_ref().expect("slot filled before wake") {
            Ok(addrs) => Ok(Arc::clone(addrs)),
            Err(e) => Err(e.clone().into_io()),
        }
    }
}

struct CacheState {
    entries: HashMap<String, DnsEntry>,
    inflight: HashMap<String, Arc<InflightSlot>>,
    /// Bumped by [`DnsCache::invalidate`]. A leader that finishes with a
    /// stale generation resolved against the old network: its answer is
    /// never cached and its waiters get an invalidation error instead.
    generation: u64,
}

/// What a thread becomes after the fast-path check in
/// [`DnsCache::resolve_at`]: a follower joins the in-flight slot, a
/// leader owns the fresh slot (plus its generation, for staleness
/// checks at publish time).
enum LookupRole {
    Follower(Arc<InflightSlot>),
    Leader(Arc<InflightSlot>, u64),
}

/// Process-wide DNS cache. Thread-safe via a single mutex; the mutex is
/// only held for map lookups and inserts, never across a real DNS
/// resolution, so a slow resolver cannot stall other threads' lookups.
pub(crate) struct DnsCache {
    state: Mutex<CacheState>,
}

impl DnsCache {
    fn new() -> Self {
        DnsCache {
            state: Mutex::new(CacheState {
                entries: HashMap::new(),
                inflight: HashMap::new(),
                generation: 0,
            }),
        }
    }

    /// Resolve `host` through the cache, calling `resolver` on a miss.
    /// `resolver` maps a host to its bare IP addresses; the caller
    /// applies the port at dial time.
    ///
    /// The answer comes back in a shared [`Arc`]: the cache entry, every
    /// cache hit, and every single-flight follower point at one
    /// allocation, so cloning the returned `Arc` is a refcount bump,
    /// never a copy of the address list.
    ///
    /// Concurrent calls for the same uncached name are single-flighted:
    /// exactly one thread runs `resolver`, the rest share its answer.
    ///
    /// The leader's resolution is bounded by [`DNS_TIMEOUT`]: the lookup
    /// runs on a helper thread, and on expiry the leader and every
    /// follower fail with [`io::ErrorKind::TimedOut`] while the failure
    /// is negative-cached for [`NEGATIVE_TTL`]. A hung resolver can never
    /// wedge the dial threads again.
    pub(crate) fn resolve(
        &self,
        host: &str,
        resolver: impl Fn(&str) -> io::Result<Vec<IpAddr>> + Send + 'static,
    ) -> io::Result<Arc<[IpAddr]>> {
        self.resolve_at(host, Instant::now(), resolver)
    }

    /// The cache generation last set by [`invalidate`](Self::invalidate).
    fn generation(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .generation
    }

    /// Drop every cached entry and retire in-flight lookups. The hook for
    /// network-change events: when the network changes (interface up/down,
    /// VPN toggle), cached answers may point at the wrong network, so the
    /// monitor calls this to force fresh resolutions.
    ///
    /// Bumping the generation retires in-flight slots two ways: a
    /// pre-invalidation lookup that completes afterwards finds its
    /// generation stale, so its answer never enters the cache, and its
    /// leader and followers get an invalidation error instead of a
    /// possibly-stale answer — they retry as the new generation and
    /// resolve against the new network. Clearing the slot map stops new
    /// lookups from joining a retired slot.
    ///
    /// No caller yet: the network monitor does not exist in this codebase.
    /// Kept (not deleted) as the designated invalidation hook for it.
    #[allow(dead_code)]
    pub(crate) fn invalidate(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.entries.clear();
        state.inflight.clear();
        state.generation = state.generation.wrapping_add(1);
    }

    /// [`resolve`](Self::resolve) with an injectable clock, so tests can
    /// drive TTL expiry deterministically.
    fn resolve_at(
        &self,
        host: &str,
        now: Instant,
        resolver: impl Fn(&str) -> io::Result<Vec<IpAddr>> + Send + 'static,
    ) -> io::Result<Arc<[IpAddr]>> {
        // IP literals need no DNS: `to_socket_addrs` parses them without
        // touching the resolver, so caching them only wastes entries.
        if is_ip_literal(host) {
            return resolver(host).map(Arc::from);
        }
        let key = cache_key(host);
        // Fast path: cache hit, an existing in-flight lookup to join, or
        // a fresh slot to lead. The lock is released before any blocking.
        let role = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(hit) = lookup(&mut state.entries, &key, now) {
                return hit;
            }
            let generation = state.generation;
            match state.inflight.get(&key).cloned() {
                Some(slot) => LookupRole::Follower(slot),
                None => {
                    let slot = Arc::new(InflightSlot::new());
                    state.inflight.insert(key.clone(), Arc::clone(&slot));
                    LookupRole::Leader(slot, generation)
                }
            }
        };
        // Leader: run the real resolution on a helper thread with a
        // bounded wait, with no locks held, then publish the answer to
        // the cache and the waiting followers. The publish is synchronous
        // in `complete` (its atomic staleness verdict also decides the
        // leader's return); the drop fallback only runs when `complete`
        // never did, so a panicking resolver can never leave followers
        // waiting forever.
        let (slot, generation) = match role {
            LookupRole::Follower(slot) => return slot.wait(),
            LookupRole::Leader(slot, generation) => (slot, generation),
        };
        let publish = LeaderPublish::new(self, key, slot, generation, now);
        let outcome = self.resolve_with_timeout(host, resolver, generation);
        publish.complete(outcome)
    }

    /// Run the leader's `resolver` on a helper thread and wait at most
    /// [`DNS_TIMEOUT`], polling in [`DNS_POLL`] slices so an
    /// [`invalidate`](Self::invalidate) retires a hung lookup promptly
    /// instead of waiting out the whole budget. Mirrors the helper-thread
    /// pattern of `fetch`'s interruptible dial.
    ///
    /// On expiry the lookup fails with [`io::ErrorKind::TimedOut`], which
    /// [`publish`](Self::publish) negative-caches and replays to the
    /// followers — a stalled resolver wedges nobody. The helper thread
    /// keeps running until the resolver returns (std offers no way to
    /// kill it); its late answer is dropped, so at most one stray thread
    /// lingers per abandoned lookup.
    ///
    /// The successful answer is moved into its one shared [`Arc`]
    /// allocation here, so [`publish`](Self::publish) hands the cache
    /// entry, the followers, and the leader refcount clones of it —
    /// the address list is never copied again after this point.
    fn resolve_with_timeout(
        &self,
        host: &str,
        resolver: impl Fn(&str) -> io::Result<Vec<IpAddr>> + Send + 'static,
        generation: u64,
    ) -> io::Result<Arc<[IpAddr]>> {
        let (tx, rx) = mpsc::channel::<io::Result<Vec<IpAddr>>>();
        let host = host.to_owned();
        thread::spawn(move || {
            // The receiver is gone exactly when the leader already gave
            // up (timeout or invalidation); dropping the late answer is
            // correct.
            let _ = tx.send(resolver(&host));
        });
        let start = Instant::now();
        loop {
            // A network change retired this lookup: stop waiting on the
            // old network's resolver. `LeaderPublish::complete` maps this
            // to the invalidation error for the leader, and `publish`
            // wakes the followers the same way.
            if self.generation() != generation {
                return Err(invalidated_error());
            }
            let remaining = DNS_TIMEOUT.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "dns resolution timed out",
                ));
            }
            match rx.recv_timeout(DNS_POLL.min(remaining)) {
                Ok(result) => return result.map(Arc::from),
                // Slice expired without an answer: loop back and re-check
                // the generation; the full `DNS_TIMEOUT` still applies to
                // the one lookup.
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                // The helper died without reporting. The resolver cannot
                // do this in practice; surface it instead of hanging.
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(io::Error::new(
                        io::ErrorKind::ConnectionAborted,
                        "dns helper thread died",
                    ));
                }
            }
        }
    }

    /// Record the leader's answer: insert the cache entry (unless
    /// `cache_entry` is false), remove the in-flight slot, and wake the
    /// followers with the shared answer — unless the lookup's generation
    /// went stale (see [`invalidate`](Self::invalidate)), in which case
    /// nothing is cached and everyone waiting gets an invalidation error
    /// so they retry against the new network.
    ///
    /// `cache_entry` is false only for a panicking leader (see the
    /// [`Drop`] impl on [`LeaderPublish`]): a panic is a one-shot,
    /// possibly transient fault, and negative-caching it would replay
    /// the error to every later lookup for a whole [`NEGATIVE_TTL`].
    /// The error still wakes the followers; it is just not cached, so
    /// the next lookup retries resolution fresh. Genuine DNS failures
    /// (NXDOMAIN, timeouts) keep their negative entries: those come
    /// through `cache_entry = true`.
    ///
    /// Returns whether the generation was stale. The check and the insert
    /// happen under the same lock, so the verdict is atomic: callers must
    /// use it for every decision about this lookup (including the
    /// leader's own return) instead of re-checking the generation
    /// themselves, which would reopen a check/publish TOCTOU.
    fn publish(
        &self,
        key: &str,
        slot: &Arc<InflightSlot>,
        generation: u64,
        now: Instant,
        shared: Result<Arc<[IpAddr]>, SharedError>,
        cache_entry: bool,
    ) -> bool {
        let stale = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let stale = state.generation != generation;
            if !stale && cache_entry {
                match &shared {
                    Ok(addrs) if !addrs.is_empty() => {
                        // Only successful, non-empty resolutions become
                        // positive entries: an empty answer is degenerate and
                        // must not be cached as if it were a real resolution.
                        // The entry shares the leader's allocation: a
                        // refcount bump, not a copy of the address list.
                        insert(
                            &mut state.entries,
                            key.to_owned(),
                            DnsEntry::Positive {
                                addrs: Arc::clone(addrs),
                                expires: now + POSITIVE_TTL,
                            },
                            now,
                        );
                    }
                    Err(error) => {
                        insert(
                            &mut state.entries,
                            key.to_owned(),
                            DnsEntry::Negative {
                                error: error.clone(),
                                expires: now + NEGATIVE_TTL,
                            },
                            now,
                        );
                    }
                    Ok(_) => {}
                }
            }
            // Remove the slot only if it is still ours: `invalidate` may
            // have cleared the map and installed a fresh slot under the
            // same key, which must be left alone. Either way, a lookup
            // arriving between the insert above and the wake below hits
            // the fresh cache entry instead of joining a dead slot.
            if state
                .inflight
                .get(key)
                .is_some_and(|s| Arc::ptr_eq(s, slot))
            {
                state.inflight.remove(key);
            }
            stale
        };
        // A stale generation must not republish: wake our own followers
        // with an invalidation error instead of the old network's answer.
        // The leader's own return is mapped from the returned verdict in
        // `LeaderPublish::complete`.
        let shared = if stale {
            Err(SharedError::from(&invalidated_error()))
        } else {
            shared
        };
        slot.finish(shared);
        stale
    }
}

/// Publishes the leader's answer: inserts the cache entry, wakes
/// followers, and removes the in-flight slot. `complete` publishes
/// synchronously and takes the atomic staleness verdict for the leader's
/// own return; the drop fallback publishes only when `complete` never
/// armed (a panic on the resolution path), so followers never wait
/// forever. A panicking leader's error wakes the followers but is never
/// cached — the next lookup retries resolution fresh.
struct LeaderPublish<'a> {
    cache: &'a DnsCache,
    key: String,
    slot: Arc<InflightSlot>,
    generation: u64,
    now: Instant,
    /// Real time when the wait started. The injected `now` is virtual
    /// test time; the resolution itself takes real time, so on completion
    /// `now` is advanced by the real elapsed wait — otherwise a lookup
    /// that stalls for the full [`DNS_TIMEOUT`] would publish a negative
    /// entry already past its [`NEGATIVE_TTL`].
    wait_start: Instant,
    /// Armed by `complete` BEFORE it calls `publish`, disarming the drop
    /// fallback so a lookup is never published twice — even if `publish`
    /// itself panics mid-way (after the cache insert, before the slot is
    /// finished).
    published: bool,
}

impl<'a> LeaderPublish<'a> {
    fn new(
        cache: &'a DnsCache,
        key: String,
        slot: Arc<InflightSlot>,
        generation: u64,
        now: Instant,
    ) -> Self {
        LeaderPublish {
            cache,
            key,
            slot,
            generation,
            now,
            wait_start: Instant::now(),
            published: false,
        }
    }

    fn complete(mut self, outcome: io::Result<Arc<[IpAddr]>>) -> io::Result<Arc<[IpAddr]>> {
        // Advance the publish timestamp by the real time the resolution
        // took, so cache TTLs are measured from when the answer arrived.
        // For fast resolvers this is a no-op; for a timed-out lookup it
        // keeps the negative entry alive for its full `NEGATIVE_TTL`.
        self.now += self.wait_start.elapsed();
        let shared = match &outcome {
            // The cache entry, the followers, and the leader's return
            // below all share the one allocation: refcount bumps, never
            // copies of the address list.
            Ok(addrs) => Ok(Arc::clone(addrs)),
            Err(e) => Err(SharedError::from(e)),
        };
        // Publish synchronously and take `publish`'s atomic verdict: the
        // generation is checked under the same lock that guards the cache
        // insert, so an `invalidate` landing here can no longer hand the
        // leader a stale answer that the cache (and the followers)
        // refused. A lookup that straddled an `invalidate` resolved
        // against the old network, so it reports the invalidation error
        // and retries as the new generation.
        //
        // The drop guard is armed BEFORE the publish call: if `publish`
        // panics mid-way (after inserting the cache entry but before the
        // slot is finished), unwinding must not publish a second time,
        // which would overwrite the good cache entry with the panic
        // error and re-wake the followers with it. The realistic panic
        // surface here is OOM-class, and lock poisoning is recovered
        // inside `publish` rather than panicking, so this path is not
        // deterministically testable — the reorder is the whole fix.
        self.published = true;
        let stale = self.cache.publish(
            &self.key,
            &self.slot,
            self.generation,
            self.now,
            shared,
            true,
        );
        if stale {
            Err(invalidated_error())
        } else {
            outcome
        }
    }
}

impl Drop for LeaderPublish<'_> {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        // `complete` never ran: the resolution path panicked. Wake the
        // followers with the panic error so nobody hangs, but do NOT
        // cache it (`cache_entry = false`): a panic is a one-shot,
        // possibly transient fault, and negative-caching it would replay
        // the error to every later lookup for a whole `NEGATIVE_TTL` —
        // turning a transient fault into a TTL-long outage. The next
        // lookup retries resolution fresh. Nothing is cached on this
        // path, so the un-advanced `now` below computes no expiry; it is
        // only threaded through for the slot removal, which needs no
        // timestamp.
        //
        // This runs during unwinding, so it must not panic itself: every
        // lock on the publish path recovers poisoning via `into_inner`
        // instead of `expect`, otherwise a poisoned mutex would turn this
        // fallback into a double-panic abort.
        self.cache.publish(
            &self.key,
            &self.slot,
            self.generation,
            self.now,
            Err(SharedError {
                kind: io::ErrorKind::Other,
                message: "dns resolver panicked".to_owned(),
            }),
            false,
        );
    }
}

/// Cache lookup. Expired entries are evicted lazily on read. A hit
/// returns a clone of the cached [`Arc`]: a refcount bump, never a
/// reallocation of the address list.
fn lookup(
    entries: &mut HashMap<String, DnsEntry>,
    key: &str,
    now: Instant,
) -> Option<io::Result<Arc<[IpAddr]>>> {
    if entries.get(key).is_some_and(|e| e.expired(now)) {
        entries.remove(key);
        return None;
    }
    match entries.get(key)? {
        DnsEntry::Positive { addrs, .. } => Some(Ok(Arc::clone(addrs))),
        DnsEntry::Negative { error, .. } => Some(Err(error.clone().into_io())),
    }
}

/// Insert an entry, keeping the map bounded: expired entries go first,
/// then one arbitrary survivor is evicted if still at capacity.
fn insert(entries: &mut HashMap<String, DnsEntry>, key: String, entry: DnsEntry, now: Instant) {
    if entries.len() >= MAX_ENTRIES && !entries.contains_key(&key) {
        entries.retain(|_, e| !e.expired(now));
        if entries.len() >= MAX_ENTRIES
            && let Some(victim) = entries.keys().next().cloned()
        {
            entries.remove(&victim);
        }
    }
    entries.insert(key, entry);
}

/// Normalize a host name for use as a cache key. DNS names are
/// case-insensitive, and a trailing dot is equivalent to no trailing dot.
fn cache_key(host: &str) -> String {
    host.trim_end_matches('.').to_ascii_lowercase()
}

/// True for IP literals (`127.0.0.1`, `::1`, `[::1]`): these resolve
/// without DNS and bypass the cache. Brackets are stripped the same way
/// `fetch` does before parsing.
fn is_ip_literal(host: &str) -> bool {
    let bare = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    bare.parse::<IpAddr>().is_ok()
}

/// The process-wide DNS cache: every `dial_host` funnels through it, so
/// direct connects and proxy dials share resolutions, matching the
/// process-wide [`crate::pool::Pool`].
pub(crate) fn global_dns_cache() -> &'static DnsCache {
    static CACHE: OnceLock<DnsCache> = OnceLock::new();
    CACHE.get_or_init(DnsCache::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    fn ip4() -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34))
    }

    fn ip4_alt() -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(93, 184, 216, 35))
    }

    fn ip6() -> IpAddr {
        IpAddr::V6(Ipv6Addr::new(
            0x2606, 0x2800, 0x220, 1, 0x248, 0x1893, 0x25c8, 0x1946,
        ))
    }

    /// A fake resolver with a call counter and a scripted answer.
    struct FakeResolver {
        calls: AtomicUsize,
        answer: Vec<IpAddr>,
        fail: bool,
    }

    impl FakeResolver {
        fn ok(answer: Vec<IpAddr>) -> Self {
            FakeResolver {
                calls: AtomicUsize::new(0),
                answer,
                fail: false,
            }
        }

        fn failing() -> Self {
            FakeResolver {
                calls: AtomicUsize::new(0),
                answer: Vec::new(),
                fail: true,
            }
        }

        fn resolve(&self, _host: &str) -> io::Result<Vec<IpAddr>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                Err(io::Error::other("dns down"))
            } else {
                Ok(self.answer.clone())
            }
        }

        /// Borrow as an owned `'static` resolver closure, as
        /// [`DnsCache::resolve_at`] requires for its helper thread.
        fn boxed(self: &Arc<Self>) -> impl Fn(&str) -> io::Result<Vec<IpAddr>> + Send + 'static {
            let this = Arc::clone(self);
            move |h| this.resolve(h)
        }
    }

    #[test]
    fn positive_hit_skips_resolver() {
        let cache = DnsCache::new();
        let resolver = Arc::new(FakeResolver::ok(vec![ip4()]));
        let now = Instant::now();
        let first = cache
            .resolve_at("example.com", now, resolver.boxed())
            .unwrap();
        let second = cache
            .resolve_at("example.com", now, resolver.boxed())
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(
            resolver.calls.load(Ordering::SeqCst),
            1,
            "second lookup must hit the cache"
        );
    }

    #[test]
    fn cache_hit_shares_arc_without_reallocating() {
        // The perf fix: a cache hit must clone the `Arc`, not the address
        // list. Pointer equality between the leader's answer and the hit
        // proves they are one allocation; the strong count proves the hit
        // minted no fresh `Arc` and the leader's answer, the cache entry,
        // and the hit all share it.
        let cache = DnsCache::new();
        let resolver = Arc::new(FakeResolver::ok(vec![ip4(), ip6()]));
        let now = Instant::now();
        let first = cache
            .resolve_at("example.com", now, resolver.boxed())
            .unwrap();
        let second = cache
            .resolve_at("example.com", now, resolver.boxed())
            .unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "cache hit must return the cached Arc, not a fresh allocation"
        );
        assert_eq!(
            Arc::strong_count(&second),
            3,
            "leader answer, cache entry, and hit must share one Arc"
        );
        assert_eq!(&second[..], &[ip4(), ip6()]);
        assert_eq!(
            resolver.calls.load(Ordering::SeqCst),
            1,
            "second lookup must hit the cache"
        );
    }

    #[test]
    fn positive_entry_expires_after_ttl() {
        let cache = DnsCache::new();
        let resolver = Arc::new(FakeResolver::ok(vec![ip4()]));
        let t0 = Instant::now();
        cache
            .resolve_at("example.com", t0, resolver.boxed())
            .unwrap();
        // Still fresh just before the deadline.
        cache
            .resolve_at(
                "example.com",
                t0 + POSITIVE_TTL - Duration::from_secs(1),
                resolver.boxed(),
            )
            .unwrap();
        assert_eq!(resolver.calls.load(Ordering::SeqCst), 1);
        // Past the deadline the resolver runs again.
        cache
            .resolve_at(
                "example.com",
                t0 + POSITIVE_TTL + Duration::from_secs(1),
                resolver.boxed(),
            )
            .unwrap();
        assert_eq!(resolver.calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn negative_entry_suppresses_retries_briefly() {
        let cache = DnsCache::new();
        let resolver = Arc::new(FakeResolver::failing());
        let t0 = Instant::now();
        let err = cache
            .resolve_at("example.com", t0, resolver.boxed())
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
        // Within the negative TTL the cached error replays without a new lookup.
        let replayed = cache
            .resolve_at("example.com", t0 + Duration::from_secs(1), resolver.boxed())
            .unwrap_err();
        assert_eq!(replayed.kind(), io::ErrorKind::Other);
        assert_eq!(replayed.to_string(), "dns down");
        assert_eq!(resolver.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn negative_entry_expires_and_retries() {
        let cache = DnsCache::new();
        let resolver = Arc::new(FakeResolver::failing());
        let t0 = Instant::now();
        cache
            .resolve_at("example.com", t0, resolver.boxed())
            .unwrap_err();
        cache
            .resolve_at(
                "example.com",
                t0 + NEGATIVE_TTL + Duration::from_secs(1),
                resolver.boxed(),
            )
            .unwrap_err();
        assert_eq!(
            resolver.calls.load(Ordering::SeqCst),
            2,
            "expired failure must retry the resolver"
        );
    }

    #[test]
    fn failure_does_not_poison_positive_cache() {
        let cache = DnsCache::new();
        let failing = Arc::new(FakeResolver::failing());
        let t0 = Instant::now();
        cache
            .resolve_at("example.com", t0, failing.boxed())
            .unwrap_err();
        // A later success replaces the negative entry with a real answer.
        let working = Arc::new(FakeResolver::ok(vec![ip4_alt()]));
        let addrs = cache
            .resolve_at(
                "example.com",
                t0 + NEGATIVE_TTL + Duration::from_secs(1),
                working.boxed(),
            )
            .unwrap();
        assert_eq!(&addrs[..], &[ip4_alt()]);
        // And that success is now cached positively.
        let cached = cache
            .resolve_at(
                "example.com",
                t0 + NEGATIVE_TTL + Duration::from_secs(2),
                working.boxed(),
            )
            .unwrap();
        assert_eq!(&cached[..], &[ip4_alt()]);
        assert_eq!(working.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn empty_resolution_is_not_cached() {
        let cache = DnsCache::new();
        let resolver = Arc::new(FakeResolver::ok(Vec::new()));
        let now = Instant::now();
        assert!(
            cache
                .resolve_at("example.com", now, resolver.boxed())
                .unwrap()
                .is_empty()
        );
        cache
            .resolve_at("example.com", now, resolver.boxed())
            .unwrap();
        assert_eq!(
            resolver.calls.load(Ordering::SeqCst),
            2,
            "empty answers must not be cached"
        );
    }

    #[test]
    fn ip_literals_bypass_the_cache() {
        let cache = DnsCache::new();
        let resolver = Arc::new(FakeResolver::ok(vec![ip4()]));
        let now = Instant::now();
        cache
            .resolve_at("127.0.0.1", now, resolver.boxed())
            .unwrap();
        cache
            .resolve_at("127.0.0.1", now, resolver.boxed())
            .unwrap();
        cache.resolve_at("[::1]", now, resolver.boxed()).unwrap();
        assert_eq!(
            resolver.calls.load(Ordering::SeqCst),
            3,
            "IP literals must never be cached"
        );
    }

    #[test]
    fn cache_key_is_case_and_dot_insensitive() {
        let cache = DnsCache::new();
        let resolver = Arc::new(FakeResolver::ok(vec![ip4()]));
        let now = Instant::now();
        cache
            .resolve_at("EXAMPLE.com.", now, resolver.boxed())
            .unwrap();
        cache
            .resolve_at("example.com", now, resolver.boxed())
            .unwrap();
        assert_eq!(resolver.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cached_answer_is_port_free_across_schemes() {
        // Regression: the cache used to store `SocketAddr`, so a lookup
        // for `http://example.com/` cached `[ip]:80` and a later
        // `https://example.com/` hit the cache and dialed port 80 with
        // TLS (or vice versa). Answers are bare `IpAddr`s now — the type
        // cannot carry a port — and the dial applies the request's port,
        // so one cached answer serves every scheme.
        let cache = DnsCache::new();
        let resolver = Arc::new(FakeResolver::ok(vec![ip4()]));
        let now = Instant::now();
        // First resolution, as `dial_host` runs it for an http request...
        let addrs = cache
            .resolve_at("example.com", now, resolver.boxed())
            .unwrap();
        assert_eq!(&addrs[..], &[ip4()]);
        // ...then the https lookup for the same host must get the same
        // port-free answer from the cache, not `[ip]:80`.
        let again = cache
            .resolve_at("example.com", now, resolver.boxed())
            .unwrap();
        assert_eq!(&again[..], &[ip4()]);
        assert_eq!(
            resolver.calls.load(Ordering::SeqCst),
            1,
            "the second scheme must hit the cache"
        );
    }

    #[test]
    fn caches_mixed_ipv4_and_ipv6_answers() {
        let cache = DnsCache::new();
        // Happy-Eyeballs needs both families; the cache must keep the
        // whole answer — order and family — intact.
        let answer = vec![ip6(), ip4_alt()];
        let resolver = Arc::new(FakeResolver::ok(answer.clone()));
        let now = Instant::now();
        let first = cache
            .resolve_at("dual.example", now, resolver.boxed())
            .unwrap();
        let second = cache
            .resolve_at("dual.example", now, resolver.boxed())
            .unwrap();
        assert_eq!(&first[..], &answer[..]);
        assert_eq!(&second[..], &answer[..]);
        assert_eq!(resolver.calls.load(Ordering::SeqCst), 1);
        assert!(second[0].is_ipv6());
        assert!(second[1].is_ipv4());
    }

    #[test]
    fn invalidate_drops_cached_entries() {
        let cache = DnsCache::new();
        let resolver = Arc::new(FakeResolver::ok(vec![ip4()]));
        let now = Instant::now();
        cache
            .resolve_at("example.com", now, resolver.boxed())
            .unwrap();
        cache
            .resolve_at("example.com", now, resolver.boxed())
            .unwrap();
        assert_eq!(resolver.calls.load(Ordering::SeqCst), 1);
        // Network change: cached answers may point at the wrong network.
        cache.invalidate();
        cache
            .resolve_at("example.com", now, resolver.boxed())
            .unwrap();
        assert_eq!(
            resolver.calls.load(Ordering::SeqCst),
            2,
            "invalidate must force a fresh lookup"
        );
    }

    #[test]
    fn invalidate_retires_inflight_lookups() {
        use std::sync::atomic::{AtomicBool, Ordering};
        // The leader's resolver blocks until released, so the lookup is
        // still in flight when the network "changes".
        let cache = Arc::new(DnsCache::new());
        let entered = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let leader_cache = Arc::clone(&cache);
        let leader_entered = Arc::clone(&entered);
        let leader_release = Arc::clone(&release);
        let leader = std::thread::spawn(move || {
            leader_cache.resolve("stale.example", move |_| {
                leader_entered.store(true, Ordering::SeqCst);
                while !leader_release.load(Ordering::SeqCst) {
                    std::thread::yield_now();
                }
                Ok(vec![ip4()])
            })
        });
        // Wait until the leader owns the in-flight slot, then park a
        // follower on it. Never resolves: the name is already in flight,
        // so this thread joins the slot instead of running the resolver.
        while !entered.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        let follower_cache = Arc::clone(&cache);
        let follower = std::thread::spawn(move || {
            follower_cache.resolve("stale.example", |_: &str| {
                panic!("follower must join the in-flight slot, not resolve")
            })
        });
        // Let the follower park on the slot before invalidating (it joins
        // within microseconds; the sleep only widens the margin).
        std::thread::sleep(Duration::from_millis(100));
        // Network change mid-lookup.
        cache.invalidate();
        release.store(true, Ordering::SeqCst);
        // The stale answer must reach nobody: the leader...
        let leader_err = leader.join().expect("leader panicked").unwrap_err();
        assert_eq!(
            leader_err.to_string(),
            "dns lookup invalidated by network change"
        );
        // ...and the follower get an invalidation error instead.
        let follower_err = follower.join().expect("follower panicked").unwrap_err();
        assert_eq!(
            follower_err.to_string(),
            "dns lookup invalidated by network change"
        );
        // And the stale answer was not republished: the next lookup
        // resolves fresh.
        let resolver = Arc::new(FakeResolver::ok(vec![ip4_alt()]));
        let now = Instant::now();
        let fresh = cache
            .resolve_at("stale.example", now, resolver.boxed())
            .unwrap();
        assert_eq!(&fresh[..], &[ip4_alt()]);
        assert_eq!(
            resolver.calls.load(Ordering::SeqCst),
            1,
            "the stale answer must not have been cached"
        );
    }

    #[test]
    fn leader_publish_verdict_is_atomic() {
        // `LeaderPublish::complete` takes its staleness verdict from the
        // atomic check-and-insert inside `publish`, so the leader's
        // return, the cache, and the followers can never disagree.
        let cache = DnsCache::new();
        let now = Instant::now();
        let answer = vec![ip4()];

        // Invalidate before completion: the leader, the cache, and the
        // followers all see the stale verdict — stale data reaches
        // nobody and nothing is cached.
        let generation = cache.generation();
        let slot = Arc::new(InflightSlot::new());
        let publish = LeaderPublish::new(
            &cache,
            "atomic.test".to_string(),
            Arc::clone(&slot),
            generation,
            now,
        );
        cache.invalidate();
        let err = publish.complete(Ok(Arc::from(answer.clone()))).unwrap_err();
        assert_eq!(err.to_string(), "dns lookup invalidated by network change");
        assert_eq!(
            slot.wait().unwrap_err().to_string(),
            "dns lookup invalidated by network change"
        );
        let resolver = Arc::new(FakeResolver::ok(vec![ip4_alt()]));
        let fresh = cache
            .resolve_at("atomic.test", now, resolver.boxed())
            .unwrap();
        assert_eq!(&fresh[..], &[ip4_alt()]);
        assert_eq!(
            resolver.calls.load(Ordering::SeqCst),
            1,
            "the stale answer must not have been cached"
        );

        // No invalidate: the answer reaches the leader, the cache, and
        // the followers together.
        let generation = cache.generation();
        let slot = Arc::new(InflightSlot::new());
        let publish = LeaderPublish::new(
            &cache,
            "atomic2.test".to_string(),
            Arc::clone(&slot),
            generation,
            now,
        );
        let got = publish.complete(Ok(Arc::from(answer.clone()))).unwrap();
        assert_eq!(&got[..], &answer[..]);
        assert_eq!(&slot.wait().unwrap()[..], &answer[..]);
        let cached = cache
            .resolve_at("atomic2.test", now, |_: &str| -> io::Result<Vec<IpAddr>> {
                panic!("must hit the published cache entry")
            })
            .unwrap();
        assert_eq!(&cached[..], &answer[..]);
    }

    #[test]
    fn racing_invalidate_never_splits_leader_and_followers() {
        // A background thread invalidates continuously while the main
        // thread publishes lookups. The staleness verdict is taken once,
        // under the publish lock, so for every interleaving the leader's
        // return and the followers' wake must agree: both see the
        // answer, or both see the invalidation error.
        //
        // The answer is deliberately large (1 MiB): the leader clones it
        // before publishing, which stretches the race window to ~0.1 ms
        // so a racing invalidate reliably lands inside it. The old
        // check-then-publish shape handed the leader an `Ok` while the
        // cache refused the entry; this test catches that split.
        let cache = Arc::new(DnsCache::new());
        let stop = Arc::new(AtomicBool::new(false));
        let hammer_cache = Arc::clone(&cache);
        let hammer_stop = Arc::clone(&stop);
        let hammer_count = Arc::new(AtomicUsize::new(0));
        let hammer_tally = Arc::clone(&hammer_count);
        let hammer = std::thread::spawn(move || {
            while !hammer_stop.load(Ordering::Relaxed) {
                hammer_cache.invalidate();
                hammer_tally.fetch_add(1, Ordering::Relaxed);
                // Spin WITHOUT the lock (never yield: a yielding
                // spinner is descheduled for whole timeslices while
                // the publisher memcpys). The gaps let the
                // publisher's generation read see a fresh generation
                // while an invalidate still lands inside the wide
                // clone window that follows.
                for _ in 0..1000 {
                    std::hint::spin_loop();
                }
            }
        });
        // Let the hammer prove it is scheduled before the timed
        // section: on a loaded box a fresh thread can otherwise sit
        // unscheduled for the whole (short) publish loop, and the
        // race would never actually happen.
        std::thread::sleep(Duration::from_millis(5));
        let hammer_before = hammer_count.load(Ordering::Relaxed);
        assert!(
            hammer_before > 0,
            "hammer thread never scheduled; the race was not exercised"
        );
        // Pre-build slots so the generation read and the publish stay
        // as close together as possible.
        let answer = vec![ip4(); 65536];
        let now = Instant::now();
        let slots: Vec<Arc<InflightSlot>> =
            (0..100).map(|_| Arc::new(InflightSlot::new())).collect();
        for (i, slot) in slots.iter().enumerate() {
            let generation = cache.generation();
            let publish = LeaderPublish::new(
                &cache,
                format!("race{i}.test"),
                Arc::clone(slot),
                generation,
                now,
            );
            match publish.complete(Ok(Arc::from(answer.clone()))) {
                Ok(addrs) => {
                    assert_eq!(&addrs[..], &answer[..]);
                    assert_eq!(
                        &slot.wait().expect("slot finished by complete")[..],
                        &answer[..],
                        "leader got Ok but followers did not (iteration {i})"
                    );
                }
                Err(e) => {
                    assert_eq!(e.to_string(), "dns lookup invalidated by network change");
                    assert_eq!(
                        slot.wait().unwrap_err().to_string(),
                        "dns lookup invalidated by network change",
                        "leader got Err but followers did not (iteration {i})"
                    );
                }
            }
        }
        stop.store(true, Ordering::Relaxed);
        hammer.join().expect("hammer thread panicked");
        // Sanity: the hammer actually raced rather than starving, so
        // the per-iteration agreement assertions above mean something.
        // The bar is deliberately low (any activity lands inside the
        // ~0.1 ms clone windows, which cover nearly all of the
        // publisher's time); it only trips if the hammer was starved
        // for the entire loop.
        let hammer_during = hammer_count.load(Ordering::Relaxed) - hammer_before;
        assert!(
            hammer_during > 20,
            "hammer thread was starved ({} invalidates during the loop); \
             the race was not exercised",
            hammer_during
        );
    }

    #[test]
    fn concurrent_lookups_are_safe() {
        let cache = Arc::new(DnsCache::new());
        let mut handles = Vec::new();
        for _ in 0..8 {
            let cache = Arc::clone(&cache);
            handles.push(std::thread::spawn(move || {
                // Real resolver would do DNS; here exercise the lock
                // under contention with a fixed fake answer.
                let answer = vec![ip4()];
                for _ in 0..50 {
                    let one = answer.clone();
                    let got = cache
                        .resolve("threaded.example", move |_| Ok(one.clone()))
                        .unwrap();
                    assert_eq!(&got[..], &answer[..]);
                }
            }));
        }
        for h in handles {
            h.join().expect("lookup thread panicked");
        }
    }

    /// Drives N threads at one uncached name with the resolver stalled
    /// until every thread is inside `resolve`, so all of them overlap on
    /// a single lookup.
    type HammerOutcome = (usize, Vec<Result<Arc<[IpAddr]>, String>>);

    fn hammer_one_name(fail: bool) -> HammerOutcome {
        let cache = Arc::new(DnsCache::new());
        let started = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(AtomicUsize::new(0));
        const N: usize = 8;
        let mut handles = Vec::new();
        for _ in 0..N {
            let cache = Arc::clone(&cache);
            let started = Arc::clone(&started);
            let calls = Arc::clone(&calls);
            handles.push(std::thread::spawn(move || {
                started.fetch_add(1, Ordering::SeqCst);
                let r = cache.resolve("singleflight.example", move |_| {
                    // Wait until every thread is inside `resolve` before
                    // answering, so they all overlap on one lookup.
                    while started.load(Ordering::SeqCst) < N {
                        std::thread::yield_now();
                    }
                    calls.fetch_add(1, Ordering::SeqCst);
                    // Widen the overlap window so followers pile up.
                    std::thread::sleep(Duration::from_millis(20));
                    if fail {
                        Err(io::Error::other("dns down"))
                    } else {
                        Ok(vec![ip4()])
                    }
                });
                r.map_err(|e| e.to_string())
            }));
        }
        let mut results = Vec::new();
        for h in handles {
            results.push(h.join().expect("lookup thread panicked"));
        }
        (calls.load(Ordering::SeqCst), results)
    }

    #[test]
    fn single_flight_coalesces_concurrent_lookups() {
        let (calls, results) = hammer_one_name(false);
        assert_eq!(
            calls, 1,
            "concurrent lookups for one name must issue a single query"
        );
        assert_eq!(results.len(), 8);
        for r in &results {
            assert_eq!(&r.as_ref().unwrap()[..], &[ip4()]);
        }
    }

    #[test]
    fn single_flight_shares_failures() {
        let (calls, results) = hammer_one_name(true);
        assert_eq!(
            calls, 1,
            "concurrent failing lookups must issue a single query"
        );
        for r in &results {
            assert_eq!(r.as_ref().unwrap_err(), "dns down");
        }
    }

    /// A resolver that never answers until released, standing in for a
    /// blackholed nameserver or a broken `/etc/resolv.conf`.
    fn hanging_resolver(
        release: Arc<AtomicBool>,
    ) -> impl Fn(&str) -> io::Result<Vec<IpAddr>> + Send + 'static {
        move |_: &str| {
            while !release.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(10));
            }
            Ok(vec![ip4()])
        }
    }

    #[test]
    fn hung_resolver_times_out_and_negative_caches() {
        let cache = DnsCache::new();
        let release = Arc::new(AtomicBool::new(false));
        let start = Instant::now();
        let err = cache
            .resolve("hang.example", hanging_resolver(Arc::clone(&release)))
            .unwrap_err();
        let elapsed = start.elapsed();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
        assert!(
            elapsed >= DNS_TIMEOUT,
            "must wait out the full budget, returned after {elapsed:?}"
        );
        assert!(
            elapsed < DNS_TIMEOUT + Duration::from_secs(10),
            "must not hang past the budget, took {elapsed:?}"
        );
        // The timeout is negative-cached: a second lookup replays the
        // failure without running the resolver again.
        let replayed = cache
            .resolve("hang.example", |_: &str| {
                panic!("negative-cached timeout must not re-run the resolver")
            })
            .unwrap_err();
        assert_eq!(replayed.kind(), io::ErrorKind::TimedOut);
        // Past the negative TTL the resolver runs again.
        release.store(true, Ordering::SeqCst);
        std::thread::sleep(NEGATIVE_TTL + Duration::from_secs(1));
        let addrs = cache
            .resolve("hang.example", hanging_resolver(Arc::clone(&release)))
            .unwrap();
        assert_eq!(&addrs[..], &[ip4()]);
    }

    #[test]
    fn hung_lookup_wakes_followers_with_timeout() {
        // Leader stalls in the resolver; a follower joins the in-flight
        // slot. Both must get a timeout instead of hanging forever.
        let cache = Arc::new(DnsCache::new());
        let release = Arc::new(AtomicBool::new(false));
        let leader_cache = Arc::clone(&cache);
        let leader_release = Arc::clone(&release);
        let leader = std::thread::spawn(move || {
            leader_cache.resolve("wedged.example", hanging_resolver(leader_release))
        });
        // Give the leader time to own the slot before the follower
        // arrives, so the follower truly parks on the in-flight lookup.
        std::thread::sleep(Duration::from_millis(100));
        let follower_cache = Arc::clone(&cache);
        let follower = std::thread::spawn(move || {
            follower_cache.resolve("wedged.example", |_: &str| {
                panic!("follower must join the in-flight slot, not resolve")
            })
        });
        let start = Instant::now();
        let leader_err = leader.join().expect("leader panicked").unwrap_err();
        let follower_err = follower.join().expect("follower panicked").unwrap_err();
        let elapsed = start.elapsed();
        assert_eq!(leader_err.kind(), io::ErrorKind::TimedOut);
        assert_eq!(follower_err.kind(), io::ErrorKind::TimedOut);
        assert!(
            elapsed < DNS_TIMEOUT + Duration::from_secs(10),
            "follower must not hang past the leader's budget, took {elapsed:?}"
        );
        release.store(true, Ordering::SeqCst);
    }

    #[test]
    fn insert_evicts_expired_entries_first() {
        let cache = DnsCache::new();
        let t0 = Instant::now();
        // Fill the cache with entries that are already expired.
        for i in 0..MAX_ENTRIES {
            let resolver = Arc::new(FakeResolver::ok(vec![ip4()]));
            cache
                .resolve_at(&format!("host{i}.example"), t0, resolver.boxed())
                .unwrap();
        }
        assert_eq!(cache.state.lock().unwrap().entries.len(), MAX_ENTRIES);
        // A fresh insert evicts the expired entries instead of growing.
        let resolver = Arc::new(FakeResolver::ok(vec![ip4_alt()]));
        cache
            .resolve_at(
                "new.example",
                t0 + POSITIVE_TTL + Duration::from_secs(1),
                resolver.boxed(),
            )
            .unwrap();
        let len = cache.state.lock().unwrap().entries.len();
        assert_eq!(len, 1, "expired entries should have been swept, got {len}");
    }

    #[test]
    fn leader_panic_mid_lookup_wakes_followers() {
        // The leader thread dies after electing itself but before
        // `complete`: the `LeaderPublish` drop fallback must publish a
        // panic error so the follower wakes instead of hanging. The
        // follower reports through a channel with a timeout, so a
        // regression fails the test in seconds instead of hanging it
        // forever.
        let cache = Arc::new(DnsCache::new());
        let now = Instant::now();
        let slot = Arc::new(InflightSlot::new());
        let (tx, rx) = mpsc::channel();
        let waiter = Arc::clone(&slot);
        thread::spawn(move || {
            let _ = tx.send(waiter.wait());
        });
        // Let the follower park on the slot before the leader dies.
        thread::sleep(Duration::from_millis(100));
        let generation = cache.generation();
        let leader_cache = Arc::clone(&cache);
        let leader = thread::spawn(move || {
            let _publish = LeaderPublish::new(
                &leader_cache,
                "doomed.example".to_string(),
                slot,
                generation,
                now,
            );
            panic!("leader died mid-lookup");
        });
        assert!(leader.join().is_err(), "leader must have panicked");
        let err = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("follower must wake after the leader dies")
            .unwrap_err();
        assert_eq!(err.to_string(), "dns resolver panicked");
        // The panic error must NOT be negative-cached: a panic is a
        // one-shot, possibly transient fault, and caching it would turn
        // it into a `NEGATIVE_TTL`-long outage. The next lookup retries
        // resolution fresh instead of replaying the cached error. (The
        // `leader.join()` above synchronizes with the leader thread's
        // end, so the in-flight slot is definitely gone by now.)
        let resolver = Arc::new(FakeResolver::ok(vec![ip4()]));
        let retried = cache
            .resolve_at("doomed.example", Instant::now(), resolver.boxed())
            .expect("lookup after a leader panic must retry, not replay the panic error");
        assert_eq!(&retried[..], &[ip4()]);
        assert_eq!(
            resolver.calls.load(Ordering::SeqCst),
            1,
            "the panic error must not have been cached"
        );
    }

    #[test]
    fn leader_panic_with_poisoned_cache_lock_still_wakes_followers() {
        // The drop fallback runs `publish`, which locks the cache. With
        // the old `.expect("dns cache lock poisoned")` posture, a
        // poisoned cache lock turned the fallback into a panic during
        // unwinding — a double-panic abort of the whole process. The
        // `into_inner` recovery must let the followers wake instead.
        let cache = Arc::new(DnsCache::new());
        // Poison the cache lock first: panic while holding it.
        let poisoner = Arc::clone(&cache);
        let died = thread::spawn(move || {
            let _guard = poisoner.state.lock().unwrap();
            panic!("poison the dns cache lock");
        });
        assert!(died.join().is_err(), "poisoner must have panicked");
        assert!(cache.state.is_poisoned());

        let slot = Arc::new(InflightSlot::new());
        let (tx, rx) = mpsc::channel();
        let waiter = Arc::clone(&slot);
        thread::spawn(move || {
            let _ = tx.send(waiter.wait());
        });
        thread::sleep(Duration::from_millis(100));
        // `generation` itself must recover through the poison.
        let generation = cache.generation();
        let leader_cache = Arc::clone(&cache);
        let leader = thread::spawn(move || {
            let _publish = LeaderPublish::new(
                &leader_cache,
                "doomed-poisoned.example".to_string(),
                slot,
                generation,
                Instant::now(),
            );
            panic!("leader died mid-lookup with a poisoned cache lock");
        });
        assert!(leader.join().is_err(), "leader must have panicked");
        let err = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("follower must wake even with a poisoned cache lock")
            .unwrap_err();
        assert_eq!(err.to_string(), "dns resolver panicked");
    }

    #[test]
    fn helper_thread_panic_surfaces_as_error() {
        // The resolver runs on a helper thread; if the resolver itself
        // panics there, the leader's `recv_timeout` sees the disconnect
        // and both leader and followers get an error instead of hanging.
        let cache = Arc::new(DnsCache::new());
        let leader_cache = Arc::clone(&cache);
        let leader = thread::spawn(move || {
            leader_cache.resolve("boom.example", |_: &str| -> io::Result<Vec<IpAddr>> {
                panic!("resolver blew up")
            })
        });
        // Give the leader time to own the slot so the follower parks on
        // the in-flight lookup instead of resolving itself.
        thread::sleep(Duration::from_millis(100));
        let follower_cache = Arc::clone(&cache);
        let follower = thread::spawn(move || {
            follower_cache.resolve("boom.example", |_: &str| -> io::Result<Vec<IpAddr>> {
                panic!("follower must join the in-flight slot, not resolve")
            })
        });
        let leader_err = leader.join().expect("leader panicked").unwrap_err();
        assert_eq!(leader_err.kind(), io::ErrorKind::ConnectionAborted);
        let follower_err = follower.join().expect("follower panicked").unwrap_err();
        assert_eq!(follower_err.kind(), io::ErrorKind::ConnectionAborted);
    }

    #[test]
    fn poisoned_cache_lock_recovers() {
        // A panic while holding the cache lock poisons the mutex; later
        // lookups must recover via `into_inner` instead of panicking.
        let cache = Arc::new(DnsCache::new());
        let poisoner = Arc::clone(&cache);
        let died = thread::spawn(move || {
            let _guard = poisoner.state.lock().unwrap();
            panic!("poison the dns cache lock");
        });
        assert!(died.join().is_err(), "poisoner must have panicked");
        assert!(cache.state.is_poisoned());
        // The cache still resolves, caches, and replays after the poison.
        let resolver = Arc::new(FakeResolver::ok(vec![ip4()]));
        let now = Instant::now();
        let addrs = cache
            .resolve_at("poison.example", now, resolver.boxed())
            .expect("resolve must survive a poisoned cache lock");
        assert_eq!(&addrs[..], &[ip4()]);
        let replayed = cache
            .resolve_at("poison.example", now, resolver.boxed())
            .expect("replay must survive a poisoned cache lock");
        assert_eq!(replayed, addrs);
        assert_eq!(resolver.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn poisoned_slot_lock_recovers() {
        // Same posture for the in-flight slot: a poisoned slot mutex must
        // not wedge `finish` or `wait`.
        let slot = Arc::new(InflightSlot::new());
        let poisoner = Arc::clone(&slot);
        let died = thread::spawn(move || {
            let _guard = poisoner.result.lock().unwrap();
            panic!("poison the dns slot lock");
        });
        assert!(died.join().is_err(), "poisoner must have panicked");
        assert!(slot.result.is_poisoned());
        slot.finish(Ok(Arc::from(vec![ip4()])));
        assert_eq!(
            &slot.wait().expect("wait must survive poison")[..],
            &[ip4()]
        );
    }
}
