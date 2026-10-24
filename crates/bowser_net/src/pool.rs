//! Bounded HTTP/1.1 connection pool with keep-alive reuse and TLS 1.3
//! session ticket caching.
//!
//! Every public fetch funnels through one process-wide [`Pool`]
//! ([`global_pool`]), so both the redirect hops of a single fetch and
//! standalone `fetch` calls made one after another share keep-alive
//! connections. TLS connections additionally contribute their
//! post-handshake session tickets to a per-origin cache, so a later
//! connection to the same origin can resume instead of doing a full
//! handshake.
//!
//! Design notes:
//!
//! * The pool is keyed by normalized `(scheme, host, port)`; an explicit
//!   default port and an omitted one are the same key, and the host is
//!   lowercased (DNS is case-insensitive).
//! * A connection is checked out briefly under the lock and all I/O —
//!   the liveness probe, the request/response exchange — happens outside
//!   it. The pool never blocks a fetch.
//! * Both idle count (per origin and global) and idle age are bounded,
//!   so the pool cannot grow without limit and dead servers do not pin
//!   sockets forever.
//! * Every candidate is liveness-probed before reuse: a server that went
//!   away silently costs one non-blocking probe, not a failed fetch.
//!   Probing a TLS connection also absorbs any session tickets waiting
//!   on the socket into the ticket cache.
//! * H2 sessions idle past `H2_PING_WARM_AGE` fire a PING at take
//!   time instead of the peek probe (which cannot see a peer that
//!   died without FIN, and which would misread buffered inbound
//!   frames as death) — but the take does *not* wait for the ACK. The
//!   PING frame itself resets the server's idle timer, its send
//!   failure is an immediate cheap death signal, and the ACK is
//!   absorbed by the normal frame loop while the fetch's own
//!   request/response exchange — itself a full liveness proof —
//!   proceeds concurrently. The PING write runs under
//!   `min(PING_WRITE_TIMEOUT, fetch_total)` with the dial-time write
//!   timeout restored afterwards, so a stalled peer cannot hold the
//!   session lock past it. No round trip sits on the fetch critical
//!   path, and a peer that died silently is detected at first
//!   response read — where a fetch that burns its whole budget with
//!   zero bytes received for any stream retires the session, so a
//!   blackholed peer is not re-handed-out forever.
//! * Entries idle past `MAX_IDLE_AGE` are reaped opportunistically on
//!   `put`, throttled to one scan per `REAP_INTERVAL` — there is no
//!   background reaper thread by design.
//! * Session tickets are harvested whenever a connection is returned or
//!   discarded, and on every shared-H2 take (a take-time probe absorbs
//!   waiting `NewSessionTicket`s on the fresh path; the warm path
//!   skips the peek, so tickets arriving while warm wait for the first
//!   pump instead); never while a connection is checked out for H1.
//! * H2 sessions are *shared*: concurrent fetches to one origin open
//!   multiplexed streams on a single pooled session instead of racing
//!   to dial their own. Stream owners cooperatively pump the session —
//!   each `next_event` reads at most one frame (socket-bounded by
//!   `PUMP_SLICE`) and forwards other streams' events to their owners
//!   — so no I/O thread per connection is needed and a slow stream
//!   never starves the others past one slice.
//! * Stream concurrency is bounded twice: slot reservations gate
//!   in-flight streams against the peer's
//!   `SETTINGS_MAX_CONCURRENT_STREAMS` (capped locally by
//!   `LOCAL_H2_MAX_STREAMS`), and `Session::open_stream` enforces the
//!   peer's limit as a backstop. A saturated session is simply not
//!   handed out; the fetch falls back to the H1 pool, then a fresh
//!   dial. Shared sessions are capped per origin by
//!   `MAX_SHARED_H2_PER_ORIGIN`.
//! * A shared session is never removed from the pool while streams are
//!   open — eviction and the reaper only drop the pool's `Arc` — so
//!   idle teardown and global-cap eviction never disturb in-flight
//!   streams; the socket closes when the last stream handle drops.

use std::collections::{HashMap, VecDeque};
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    mpsc,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::fetch::check_fetch_deadline;
use crate::http2::{E_CANCEL, E_STREAM_CLOSED, Event, H2Error, H2EventSource, Session};
use crate::psl;
use crate::tls::{CachedTicket, PskOffer, TlsConnection};

/// A transport: plain TCP or the established TLS channel. The TLS half is
/// boxed: `TlsConnection` carries the session state (248+ bytes) and would
/// otherwise bloat every `Conn` value (clippy::large_enum_variant).
pub(crate) enum Conn {
    Plain(TcpStream),
    Tls(Box<TlsConnection>),
}

impl Conn {
    pub(crate) fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::Plain(s) => s.read(buf),
            Self::Tls(s) => s.read(buf),
        }
    }

    pub(crate) fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        match self {
            Self::Plain(s) => s.write_all(buf),
            // `TlsConnection::write` always consumes the whole slice.
            Self::Tls(s) => s.write(buf).map(|_| ()),
        }
    }

    /// Best-effort liveness check. Never blocks: at most one
    /// non-blocking peek is attempted. Any waiting application data (we
    /// never pipeline, so none should exist) or any error/EOF reports
    /// the connection dead.
    pub(crate) fn probe(&mut self) -> bool {
        match self {
            Self::Plain(s) => {
                // Zero-duration read timeouts are rejected by the OS
                // (EINVAL), so toggle non-blocking mode instead: one
                // peek reports WouldBlock on an idle socket, and EOF or
                // an error when the peer went away. O_NONBLOCK and
                // SO_RCVTIMEO are independent flags, so the configured
                // read timeout survives the round trip.
                //
                // `peek` has MSG_PEEK semantics: a waiting byte is
                // reported but never consumed. A plain `read` would
                // discard it and desync the response parser when a
                // pooled peer sent unsolicited bytes.
                if s.set_nonblocking(true).is_err() {
                    return false;
                }
                let mut one = [0u8; 1];
                let alive =
                    matches!(s.peek(&mut one), Err(e) if e.kind() == io::ErrorKind::WouldBlock);
                // A socket left non-blocking would break the caller's
                // blocking reads; if it cannot be restored, drop it.
                s.set_nonblocking(false).is_ok() && alive
            }
            Self::Tls(t) => t.probe(),
        }
    }

    /// Bound blocking reads on the underlying socket. Callers that
    /// shrink the timeout for a bounded operation (the pump slice)
    /// must restore the normal socket timeout afterwards; no caller
    /// should leave a non-default timeout behind.
    pub(crate) fn set_read_timeout(&mut self, timeout: Option<Duration>) -> io::Result<()> {
        match self {
            Self::Plain(s) => s.set_read_timeout(timeout),
            Self::Tls(t) => t.set_read_timeout(timeout),
        }
    }

    /// Bound blocking writes on the underlying socket. Callers that
    /// shrink the timeout for a bounded operation (the pump slice)
    /// must restore the normal socket timeout afterwards; no caller
    /// should leave a non-default timeout behind.
    pub(crate) fn set_write_timeout(&mut self, timeout: Option<Duration>) -> io::Result<()> {
        match self {
            Self::Plain(s) => s.set_write_timeout(timeout),
            Self::Tls(t) => t.set_write_timeout(timeout),
        }
    }

    /// Absolute wall-clock bound for blocking reads on the TLS half.
    /// `Session::probe` sets this around its PING write (the current
    /// keepalive discipline) and clears it on restore; a plain connection
    /// has no record-assembly loop, so its reads are already bounded by
    /// the per-read timeout the probe recomputes.
    pub(crate) fn set_read_deadline(&mut self, deadline: Option<Instant>) {
        if let Self::Tls(t) = self {
            t.set_read_deadline(deadline);
        }
    }

    /// Drain the session tickets a TLS connection received. Plain
    /// connections have none.
    pub(crate) fn take_tls_tickets(&mut self) -> Vec<CachedTicket> {
        match self {
            Self::Plain(_) => Vec::new(),
            Self::Tls(t) => t.take_received_tickets(),
        }
    }

    /// ALPN protocol the TLS handshake negotiated, if any. Plain
    /// connections negotiate nothing.
    pub(crate) fn alpn_protocol(&self) -> Option<&str> {
        match self {
            Self::Plain(_) => None,
            Self::Tls(t) => t.alpn_protocol(),
        }
    }
}

/// Normalized pool key: scheme + host + port. The host goes through the
/// shared [`psl::normalize_host`] canonicalization (one trailing dot
/// stripped, ASCII-lowercased); the port is always explicit by the time
/// it reaches the pool, so an omitted default port and a written-out one
/// are the same key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PoolKey {
    secure: bool,
    host: String,
    port: u16,
}

impl PoolKey {
    pub(crate) fn new(secure: bool, host: &str, port: u16) -> Self {
        Self {
            secure,
            host: psl::normalize_host(host).into_owned(),
            port,
        }
    }
}

/// Maximum idle connections kept per origin.
const MAX_IDLE_PER_ORIGIN: usize = 6;
/// Maximum idle connections kept across all origins.
const MAX_IDLE_TOTAL: usize = 32;
/// Idle connections older than this are evicted instead of reused.
const MAX_IDLE_AGE: Duration = Duration::from_secs(90);
/// H2 sessions idle longer than this get a fire-and-forget PING at
/// take time instead of the cheap peek probe alone: the peek cannot
/// see a peer that died without FIN, and the PING frame resets the
/// server's idle timer, keeping hot origins resumable. The take does
/// not wait for the ACK — see [`SharedH2::probe`] — so the threshold
/// only decides when the keepalive frame goes out, not how long a
/// checkout takes. Well under `MAX_IDLE_AGE` so a stale entry is
/// still young enough to hand out.
const H2_PING_WARM_AGE: Duration = Duration::from_secs(30);
/// Bound on the take-time PING write. The socket write timeout sits at
/// its restored dial-time value (`SOCKET_TIMEOUT` = 60 s) when the
/// warm-path PING goes out, and the session lock is held across
/// `send_ping`: a TCP-alive-but-not-draining peer would stall a warm
/// take up to 60 s there, with every other stream owner queued behind
/// the session lock past their `PUMP_SLICE` quantum. The write
/// therefore runs under `min(PING_WRITE_TIMEOUT, fetch_total)` — a
/// checkout 1 ms before the fetch deadline can burn at most the
/// remaining budget, never the dial-time 60 s — and the dial-time
/// timeout is restored afterwards (the standing keepalive discipline:
/// the tight bound applies to the PING write only, never to later
/// reads or writes on the session). A 17-byte PING that cannot be
/// written in this long is proof the peer is not draining, so the
/// session is retired.
const PING_WRITE_TIMEOUT: Duration = Duration::from_secs(2);
/// Minimum gap between full idle-map reaps. A reap scans every idle
/// entry, so throttling keeps the common `put` path cheap; there is no
/// background thread by design (see `IdleState::reap_stale`).
const REAP_INTERVAL: Duration = Duration::from_secs(30);
/// Session tickets cached per origin; matches the per-connection cap in
/// `tls.rs`, so one connection's tickets can never evict another's
/// entire set on their own.
const MAX_TICKETS_PER_ORIGIN: usize = 8;
/// Maximum shared H2 sessions pooled per origin. One shared session
/// multiplexes up to `LOCAL_H2_MAX_STREAMS` concurrent streams, so a
/// handful of connections covers even large bursts; beyond the cap a
/// fresh H2 session is closed instead of pooled.
const MAX_SHARED_H2_PER_ORIGIN: usize = 4;
/// Local cap on concurrent streams per shared H2 connection. Bounds
/// stream-state growth against a peer that never sends
/// `SETTINGS_MAX_CONCURRENT_STREAMS` (RFC 9113 defaults it to
/// unlimited); the peer's own limit applies on top via
/// [`Session::peer_max_concurrent`].
const LOCAL_H2_MAX_STREAMS: u32 = 100;
/// Pump quantum for cooperative multiplexing: each stream owner reads
/// at most one frame per pump call, bounded by this socket timeout, so
/// another owner's `open_request` or pump never waits longer than this
/// for the session lock.
const PUMP_SLICE: Duration = Duration::from_millis(10);
/// Bound on the pre-send absorb in [`SharedH2::open_request`]: the
/// drain of frames already waiting in the kernel buffer never blocks
/// past this, so a dripping peer cannot stall the open. Well under any
/// plausible fetch budget; a partial frame the absorb leaves behind is
/// simply read by the first pump instead.
const ABSORB_BUDGET: Duration = Duration::from_millis(10);
/// Bound on the best-effort RST_STREAM a dropped [`SharedStream`] sends
/// for an incomplete stream: the peer may have stopped reading, and a
/// drop must never stall on the dial-time write timeout.
const RST_WRITE_TIMEOUT: Duration = Duration::from_secs(1);

struct IdleConn {
    conn: Conn,
    idle_since: Instant,
}

/// One entry evicted from the idle maps, returned to the caller for
/// dropping *after* the idle lock is released: a TLS drop performs a
/// blocking close_notify write, and a shared session's last `Arc` may
/// be the pooled one. Shared entries carry their pool key so harvested
/// tickets land in the right origin's cache.
enum Evicted {
    Conn(Conn),
    Shared(PoolKey, SharedH2),
}

/// A pooled HTTP/2 session shared by concurrent fetches: each fetch
/// opens its own stream on the one connection instead of racing to dial
/// its own. The session lives behind a mutex; stream owners
/// *cooperatively pump* it — whoever holds the lock reads one frame
/// (bounded by `PUMP_SLICE`) and forwards other streams' events to
/// their owners' channels — so no dedicated I/O thread is needed and a
/// slow stream never starves the others past one slice.
///
/// Lock order, everywhere: pool idle lock -> session lock -> `forward`
/// lock. No path takes them in reverse.
#[derive(Clone)]
pub(crate) struct SharedH2 {
    inner: Arc<SharedH2Inner>,
}

pub(crate) struct SharedH2Inner {
    session: Mutex<Session<Conn>>,
    /// stream id -> event channel of the thread owning the stream.
    /// Mutated only under the session lock, so a response frame can
    /// never miss a forwarder registered for its stream.
    forward: Mutex<HashMap<u32, mpsc::Sender<Event>>>,
    /// Streams currently open by client threads; gated against
    /// `max_streams` by [`SharedH2::try_acquire_slot`].
    in_flight: AtomicU32,
    /// min(peer's SETTINGS_MAX_CONCURRENT_STREAMS,
    /// `LOCAL_H2_MAX_STREAMS`); refreshed whenever the session lock is
    /// held after inbound frames may have been absorbed.
    max_streams: AtomicU32,
    /// The transport failed or the peer went away: never handed out
    /// again, reaped on sight.
    dead: AtomicBool,
    /// Last stream open or response event; drives warm-age PING
    /// keepalives and the idle reaper.
    last_activity: Mutex<Instant>,
    /// Count of response events delivered to stream owners since the
    /// session was pooled. Each [`SharedStream`] snapshots it at open;
    /// when a fetch burns its whole total budget, an unmoved epoch
    /// proves no stream — this one or any other owner's — received a
    /// single byte in that window (a blackholed peer), as opposed to a
    /// slow peer that is still progressing. See
    /// [`SharedH2::retire_if_no_progress`].
    progress_epoch: AtomicU64,
    /// The effective socket timeout the session was dialed with
    /// (`Timeouts.rw`). The pump slice shrinks the read/write timeouts
    /// and must restore *this* value — not a constant — so a custom
    /// `rw` survives pumping. std offers no socket-timeout getter, so
    /// the value is snapshotted at `put_shared_h2` time.
    rw_timeout: Duration,
}

impl SharedH2Inner {
    /// Refresh the cached stream cap from the latest SETTINGS. Called
    /// whenever the session lock is held after inbound frames may have
    /// been absorbed.
    fn refresh_max_streams(&self, session: &Session<Conn>) {
        let max = session.peer_max_concurrent().min(LOCAL_H2_MAX_STREAMS);
        if self.max_streams.load(Ordering::Acquire) != max {
            self.max_streams.store(max, Ordering::Release);
        }
    }

    /// Forward an event to its owning stream. A missing owner means the
    /// stream already completed; the event is dropped.
    fn forward(&self, ev: Event) {
        let id = ev.stream_id();
        let fwd = recover_lock(&self.forward);
        if let Some(tx) = fwd.get(&id) {
            // Unbounded channel: never blocks under the session lock.
            let _ = tx.send(ev);
        }
    }

    /// Drain the TLS session tickets accumulated on the transport.
    fn harvest_tickets(&self) -> Vec<CachedTicket> {
        let (mut sess, poisoned) = recover_lock_poisoned(&self.session);
        if poisoned {
            // The tickets were already received and stay valid for
            // fresh connections, so harvest them — but the session
            // itself is retired, never handed out again.
            self.dead.store(true, Ordering::Release);
        }
        sess.conn_mut().take_tls_tickets()
    }

    fn touch(&self) {
        *recover_lock(&self.last_activity) = Instant::now();
    }
}

/// Which side of the "bytes hit the wire" line a
/// [`SharedH2::open_request`] failure falls on. The fetch layer falls
/// through to another pooled session or a fresh connection on
/// [`OpenRequestError::PreSend`] exactly as before; on
/// [`OpenRequestError::PostSend`] the request may already have reached
/// the peer, so only an idempotent method may be retried — a
/// non-idempotent one fails the fetch here instead of being silently
/// re-sent.
#[derive(Debug)]
pub(crate) enum OpenRequestError {
    /// Nothing was sent: GOAWAY at take, stream-limit refusals, or the
    /// pre-send deadline. Safe to send the request elsewhere.
    PreSend(H2Error),
    /// HEADERS or body bytes may have reached the peer. Re-sending
    /// could execute the request twice.
    PostSend(H2Error),
}

/// Whether a [`Session::open_stream`] failure provably happened before
/// any request byte was written. The refusal checks — GOAWAY, stream-id
/// exhaustion, the concurrency backstop — all run before the first
/// HEADERS frame write, so only they can surface pre-write; any
/// `Io`/`Connection`/`Hpack` failure comes from the frame writes
/// themselves and may have reached the peer. (The pre-send deadline is
/// checked explicitly in `open_request` before delegating, so an
/// `Io(TimedOut)` escaping `open_stream` is a mid-write timeout, not
/// the deadline check.)
fn is_pre_send_open_error(e: &H2Error) -> bool {
    matches!(
        e,
        H2Error::GoAway(_) | H2Error::StreamsExhausted | H2Error::ConcurrentStreamsExceeded
    )
}

impl SharedH2 {
    fn new(session: Session<Conn>, rw_timeout: Duration) -> Self {
        let mut session = session;
        session.reap_closed_streams();
        let max = session.peer_max_concurrent().min(LOCAL_H2_MAX_STREAMS);
        Self {
            inner: Arc::new(SharedH2Inner {
                session: Mutex::new(session),
                forward: Mutex::new(HashMap::new()),
                in_flight: AtomicU32::new(0),
                max_streams: AtomicU32::new(max),
                dead: AtomicBool::new(false),
                last_activity: Mutex::new(Instant::now()),
                progress_epoch: AtomicU64::new(0),
                rw_timeout,
            }),
        }
    }

    /// Whether the pool may hand this session out: alive and with a
    /// free stream slot.
    fn has_free_slot(&self) -> bool {
        !self.inner.dead.load(Ordering::Acquire)
            && self.inner.in_flight.load(Ordering::Acquire)
                < self.inner.max_streams.load(Ordering::Acquire)
    }

    /// Reserve a stream slot against the concurrency limit. The pool
    /// counted this reservation when it handed the session out, so
    /// over-subscription only happens on a lost reservation race — and
    /// [`Session::open_stream`] still enforces the peer's limit as a
    /// backstop.
    fn try_acquire_slot(&self) -> bool {
        let mut cur = self.inner.in_flight.load(Ordering::Acquire);
        loop {
            let max = self.inner.max_streams.load(Ordering::Acquire);
            if cur >= max {
                return false;
            }
            match self.inner.in_flight.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => cur = actual,
            }
        }
    }

    fn release_slot(&self) {
        self.inner.in_flight.fetch_sub(1, Ordering::AcqRel);
    }

    fn mark_dead(&self) {
        self.inner.dead.store(true, Ordering::Release);
    }

    /// Flag the session dead when the error shows the transport itself
    /// failed (not just this stream): no later checkout may hand it
    /// out again.
    fn mark_dead_on_transport_error(&self, e: &H2Error) {
        if matches!(
            e,
            H2Error::Io(_) | H2Error::Connection(..) | H2Error::GoAway(_)
        ) {
            self.mark_dead();
        }
    }

    /// Retire the session when a fetch burned its whole total budget
    /// with zero bytes received for any stream. The pump deliberately
    /// does not mark the session dead on a bare deadline expiry — a
    /// slow peer may still be progressing — so without this a
    /// silent-death (blackholed) peer is re-handed-out forever: every
    /// fetch re-pays up to `fetch_total`, and `Drop for SharedStream`
    /// touches `last_activity`, keeping the reaper from ever seeing
    /// the session as stale on a hot origin. The epoch check is what
    /// distinguishes "no progress at all" from "slow peer": the
    /// session is retired only when no stream — this one or any other
    /// owner's — received a single event since this stream opened. A
    /// session other streams are provably progressing on stays pooled;
    /// this fetch simply lost its budget.
    fn retire_if_no_progress(&self, epoch_at_open: u64) {
        if self.inner.progress_epoch.load(Ordering::Acquire) == epoch_at_open {
            self.mark_dead();
        }
    }

    fn is_dead(&self) -> bool {
        self.inner.dead.load(Ordering::Acquire)
    }

    /// Drain the TLS session tickets accumulated on the transport.
    fn harvest_tickets(&self) -> Vec<CachedTicket> {
        self.inner.harvest_tickets()
    }

    /// Take-time liveness check, called with the pool lock *released*.
    /// A session with in-flight streams is alive by definition — and a
    /// peek probe would misread their in-flight bytes as death — so
    /// only idle sessions are probed: the cheap non-blocking peek when
    /// freshly idle, and past `H2_PING_WARM_AGE` a fire-and-forget
    /// PING with no peek.
    ///
    /// The peek cannot see a peer that died without FIN; waiting for
    /// the PING's ACK used to close that gap, at the cost of up to a
    /// full round trip on the fetch critical path — with the session
    /// lock held across it, stalling other stream owners past their
    /// `PUMP_SLICE` quantum. The ACK wait is gone instead: the PING
    /// frame itself resets the server's idle timer, its send failure
    /// is an immediate cheap death signal, and the ACK is absorbed by
    /// the normal frame loop (`Session::on_ping` ignores ACKs, so no
    /// bookkeeping is needed) while the fetch's own request/response
    /// exchange — itself a full liveness proof — proceeds concurrently.
    /// A peer that died silently is therefore detected at first
    /// response read rather than at take: the same discipline as a
    /// stale H1 connection (see `may_retry_stale` — idempotent methods
    /// retry on another connection, non-idempotent fetches fail rather
    /// than risk executing the request twice).
    ///
    /// The warm path deliberately skips the peek: buffered inbound
    /// frames are normal on an idle session — server
    /// SETTINGS/WINDOW_UPDATE, or the previous take's fire-and-forget
    /// PING's stale ACK when no stream was ever opened and pumped —
    /// and the peek would misread them as death (the TLS probe fails
    /// closed on app-data records), quietly defeating H2 pooling past
    /// the warm age. The trade: a peer that sent FIN while the session
    /// sat warm is not detected here either — the PING write to a
    /// FIN'd socket usually succeeds — so the failure surfaces at
    /// `open_request`/first pump under the stale-connection retry
    /// discipline above. For the same reason a buffered GOAWAY is
    /// invisible until something pumps: `open_request` drains already-
    /// waiting frames before its pre-send checks (see
    /// `absorb_waiting_frames`), but a GOAWAY that arrives in the
    /// probe-to-open race still means HEADERS go out to a gone-away
    /// peer — caught at first pump, with request safety intact (an
    /// idempotent method retries, a non-idempotent one fails cleanly
    /// rather than executing twice).
    ///
    /// `fetch_total` bounds the PING write: the socket write timeout
    /// sits at its restored dial-time value here, so the write runs
    /// under `min(PING_WRITE_TIMEOUT, fetch_total)` with the dial-time
    /// timeout restored afterwards (the standing keepalive discipline)
    /// — a checkout 1 ms before the fetch deadline cannot
    /// stall 60 s with the session lock held.
    fn probe(&self, fetch_total: Duration) -> bool {
        let inner = &self.inner;
        let (mut sess, poisoned) = recover_lock_poisoned(&inner.session);
        if poisoned {
            // A thread panicked mid-critical-section while holding the
            // session lock: the session may be logically corrupt in
            // ways no probe can detect (half-fed HPACK decoder,
            // desynced stream map, skewed flow-control windows) while
            // the TCP connection underneath is still alive. Retire it
            // sight-unseen — probing a corrupt session risks worse
            // than dropping a live one.
            drop(sess);
            self.mark_dead();
            return false;
        }
        if sess.goaway().is_some() || inner.dead.load(Ordering::Acquire) {
            return false;
        }
        inner.refresh_max_streams(&sess);
        if inner.in_flight.load(Ordering::Acquire) > 0 {
            return true;
        }
        if recover_lock(&inner.last_activity).elapsed() <= H2_PING_WARM_AGE {
            // Freshly idle: the cheap non-blocking peek. EOF or an
            // error means the peer is gone (or going). For a TLS
            // transport this also absorbs waiting session tickets into
            // the ticket cache.
            return sess.conn_mut().probe();
        }
        // Warm path, past `H2_PING_WARM_AGE`: fire the PING without
        // the peek and without waiting for the ACK (see above). Only
        // a failed send retires the session here — but the send must
        // not run under the dial-time write timeout with the session
        // lock held: a TCP-alive-but-not-draining peer would stall a
        // warm take up to 60 s. Shrink the write timeout around the
        // send and restore the dial-time value afterwards.
        let write_budget = PING_WRITE_TIMEOUT
            .min(fetch_total)
            // A zero write timeout means "block forever" to the OS,
            // not "return immediately": floor at 1 ms.
            .max(Duration::from_millis(1));
        if sess
            .conn_mut()
            .set_write_timeout(Some(write_budget))
            .is_err()
        {
            self.mark_dead();
            return false;
        }
        let ping = sess.send_ping();
        let _ = sess.conn_mut().set_write_timeout(Some(inner.rw_timeout));
        if ping.is_err() {
            self.mark_dead();
            return false;
        }
        true
    }

    /// Open a stream and send the request headers/body, returning the
    /// handle the caller pumps for its response.
    ///
    /// The error distinguishes *pre-send* failures — nothing reached
    /// the peer, so the caller may fall through to another pooled
    /// session or a fresh connection and send the request there — from
    /// *post-send* failures, where HEADERS or body bytes may already be
    /// on the wire and a silent re-send could execute the request
    /// twice. Pre-send: GOAWAY at take, [`H2Error::GoAway`], the
    /// concurrency backstop ([`H2Error::ConcurrentStreamsExceeded`]),
    /// stream-id exhaustion ([`H2Error::StreamsExhausted`]), and the
    /// pre-send deadline check. Post-send: any frame-write failure in
    /// `open_stream` (HEADERS) or `send_data` (body).
    ///
    /// On the warm path the take-time probe no longer round-trips, so
    /// before the pre-send checks the open drains frames already
    /// waiting in the kernel buffer (see [`SharedH2::absorb_waiting_frames`]):
    /// a GOAWAY the server sent while the session sat idle would
    /// otherwise be invisible until the first pump, and HEADERS would
    /// go out to a gone-away peer.
    pub(crate) fn open_request(
        &self,
        headers: &[(&[u8], &[u8])],
        body: &[u8],
        deadline: Instant,
    ) -> Result<SharedStream, OpenRequestError> {
        let (mut sess, poisoned) = recover_lock_poisoned(&self.inner.session);
        if poisoned {
            // The session may be logically corrupt (see `probe`), so
            // it is retired instead of reused. Nothing from *this*
            // request has been written — the lock was never acquired —
            // so the pre-send classification is honest and the caller
            // may retry the request elsewhere.
            self.mark_dead();
            drop(sess);
            return Err(OpenRequestError::PreSend(H2Error::GoAway(0)));
        }
        // Warm take, idle past `H2_PING_WARM_AGE`: the take-time probe
        // skipped its peek, so frames the server sent while the session
        // sat idle may be waiting unread — drain them before the
        // pre-send checks. A fresh take's peek already verified an
        // empty buffer, so the drain is skipped there (one non-blocking
        // peek already paid); a GOAWAY racing in between the peek and
        // this open is still caught at first pump, with request safety
        // intact.
        if recover_lock(&self.inner.last_activity).elapsed() > H2_PING_WARM_AGE
            && let Err(e) = self.absorb_waiting_frames(&mut sess, deadline)
        {
            return Err(e);
        }
        if sess.goaway().is_some() || self.is_dead() {
            let code = sess.goaway().map(|(_, c)| c).unwrap_or(0);
            self.mark_dead();
            return Err(OpenRequestError::PreSend(H2Error::GoAway(code)));
        }
        // Refresh under the lock *before* the reservation decision: a
        // SETTINGS absorbed by another owner's pump must be visible
        // here, or the slot count gates against a stale (usually
        // unlimited) max.
        self.inner.refresh_max_streams(&sess);
        if !self.try_acquire_slot() {
            return Err(OpenRequestError::PreSend(
                H2Error::ConcurrentStreamsExceeded,
            ));
        }
        // The deadline is checked before the first frame write: a fetch
        // already over budget fails here, provably having sent nothing.
        // (`Session::open_stream` re-checks internally; a deadline that
        // lapses in the nanoseconds between the two checks is
        // classified post-send — conservative, and that fetch was
        // doomed regardless.)
        if let Err(e) = check_fetch_deadline(deadline) {
            self.release_slot();
            return Err(OpenRequestError::PreSend(H2Error::Io(e)));
        }
        let stream = match sess.open_stream(headers, body.is_empty(), deadline) {
            Ok(id) => id,
            Err(e) => {
                self.release_slot();
                if is_pre_send_open_error(&e) {
                    return Err(OpenRequestError::PreSend(e));
                }
                // A HEADERS write failed: bytes may have reached the
                // peer, and the transport is suspect.
                self.mark_dead_on_transport_error(&e);
                return Err(OpenRequestError::PostSend(e));
            }
        };
        if !body.is_empty()
            && let Err(e) = sess.send_data(stream, body, deadline)
        {
            // Best-effort reset: the stream never got a complete
            // request, so it must not linger half-open.
            let _ = sess.send_rst(stream, E_STREAM_CLOSED);
            self.release_slot();
            // Transport-level failure: the session cannot serve other
            // streams either.
            self.mark_dead_on_transport_error(&e);
            // Body bytes may have reached the peer: never silently
            // re-send.
            return Err(OpenRequestError::PostSend(e));
        }
        let (tx, rx) = mpsc::channel();
        // Registered under the session lock: a response frame can only
        // be read after this lock is released, so no event for the new
        // stream can miss its forwarder.
        recover_lock(&self.inner.forward).insert(stream, tx);
        self.inner.touch();
        // Snapshot the session progress epoch while the session lock
        // is held: no event for any stream can be observed before this
        // lock is released, so the snapshot predates all of this
        // fetch's progress — the blackhole retire check in
        // `next_event` compares against it.
        let epoch_at_open = self.inner.progress_epoch.load(Ordering::Acquire);
        drop(sess);
        Ok(SharedStream {
            shared: self.clone(),
            stream,
            rx,
            completed: false,
            epoch_at_open,
        })
    }

    /// Drain frames already waiting in the kernel buffer, before
    /// `open_request`'s pre-send checks. The take-time probe no longer
    /// round-trips (lazy PING), so a GOAWAY the server sent while the
    /// session sat idle is invisible until something pumps — without
    /// this drain, HEADERS would go out to a gone-away peer. The
    /// failure would surface at first pump and request safety would
    /// hold (an idempotent method retries on another connection, a
    /// non-idempotent one fails instead of executing twice), but the
    /// write is wasted and a retry is burned. The drain also absorbs
    /// the previous take's stale PING ACK and any server SETTINGS, so
    /// the slot decision below sees fresh peer limits.
    ///
    /// Bounded by `ABSORB_BUDGET` (and the fetch deadline, whichever is
    /// nearer): a dripping peer cannot stall the open past it, and a
    /// partial frame left behind is simply read by the first pump. The
    /// socket timeouts are shrunk for the drain and restored to the
    /// dial-time values on every exit, exactly like the pump slice —
    /// the synchronous frame writes the drain can trigger
    /// (SETTINGS/PING ACKs) must not stall on the dial-time write
    /// timeout against a peer that stopped reading, and the absolute
    /// read deadline keeps a dripping TLS peer from stretching one
    /// record past the budget. Other streams' events are forwarded to
    /// their owners under the session lock, exactly like the pump.
    ///
    /// A transport failure during the drain retires the session and
    /// reports pre-send: nothing from this request was written yet, so
    /// the caller may still send it elsewhere.
    fn absorb_waiting_frames(
        &self,
        sess: &mut Session<Conn>,
        deadline: Instant,
    ) -> Result<(), OpenRequestError> {
        let inner = &self.inner;
        let absorb_end = deadline.min(Instant::now() + ABSORB_BUDGET);
        // A zero timeout means "block forever" to the OS, not "return
        // immediately": floor at 1 ms.
        let bound = absorb_end
            .saturating_duration_since(Instant::now())
            .max(Duration::from_millis(1));
        if let Err(e) = sess.conn_mut().set_read_timeout(Some(bound)) {
            self.mark_dead();
            return Err(OpenRequestError::PreSend(H2Error::Io(e)));
        }
        if let Err(e) = sess.conn_mut().set_write_timeout(Some(bound)) {
            let _ = sess.conn_mut().set_read_timeout(Some(inner.rw_timeout));
            self.mark_dead();
            return Err(OpenRequestError::PreSend(H2Error::Io(e)));
        }
        sess.conn_mut().set_read_deadline(Some(absorb_end));
        let result = loop {
            if sess.goaway().is_some() {
                break Ok(());
            }
            match sess.next_event(absorb_end) {
                Ok(ev) => inner.forward(ev),
                Err(H2Error::Io(e))
                    if matches!(
                        e.kind(),
                        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                    ) =>
                {
                    // Drained: the bound fired with no further frame
                    // waiting.
                    break Ok(());
                }
                Err(e) => {
                    self.mark_dead_on_transport_error(&e);
                    break Err(OpenRequestError::PreSend(e));
                }
            }
        };
        // Restore the dial-time timeouts on every exit: no later pump
        // or exchange may inherit the absorb bound.
        sess.conn_mut().set_read_deadline(None);
        let _ = sess.conn_mut().set_write_timeout(Some(inner.rw_timeout));
        let _ = sess.conn_mut().set_read_timeout(Some(inner.rw_timeout));
        result
    }
}

/// One fetch's handle on a shared H2 session: the opened stream plus
/// the channel its events arrive on. Implements [`H2EventSource`] by
/// cooperatively pumping the session: each `next_event` first drains
/// already-forwarded events, then reads at most one frame under the
/// session lock (bounded by `PUMP_SLICE`), forwarding other streams'
/// events to their owners. Dropping the handle releases the stream
/// slot; a stream dropped before its response completed is reset with
/// RST_STREAM(CANCEL) so the peer stops sending for it and the
/// session's stream budget is freed. The last one out reaps closed
/// stream state.
pub(crate) struct SharedStream {
    shared: SharedH2,
    stream: u32,
    rx: mpsc::Receiver<Event>,
    /// The response ran to a terminal event (END_STREAM headers/data
    /// or a reset): no RST is needed on drop.
    completed: bool,
    /// The session [`SharedH2Inner::progress_epoch`] value when the
    /// stream was opened. Compared at fetch-deadline expiry to tell a
    /// blackholed peer (no stream received anything: retire the
    /// session) from a slow peer (something progressed: keep it).
    epoch_at_open: u64,
}

/// A stream-level event that ends the stream: response headers or data
/// carrying END_STREAM, or a reset from either side.
fn is_terminal_event(ev: &Event) -> bool {
    matches!(
        ev,
        Event::Headers {
            end_stream: true,
            ..
        } | Event::Data {
            end_stream: true,
            ..
        } | Event::Reset { .. }
    )
}

impl SharedStream {
    pub(crate) fn stream_id(&self) -> u32 {
        self.stream
    }

    /// Record a delivered event: a terminal one marks the stream
    /// complete, so dropping the handle later sends no RST.
    fn note_event(&mut self, ev: &Event) {
        if is_terminal_event(ev) {
            self.completed = true;
        }
    }
}

impl H2EventSource for SharedStream {
    fn next_event(&mut self, deadline: Instant) -> Result<Event, H2Error> {
        loop {
            // Events another owner pumped for this stream arrive here.
            if let Ok(ev) = self.rx.try_recv() {
                self.note_event(&ev);
                return Ok(ev);
            }
            if self.shared.is_dead() {
                // Another stream observed a transport-level failure;
                // this stream cannot proceed either.
                return Err(H2Error::GoAway(0));
            }
            if let Err(e) = check_fetch_deadline(deadline) {
                // The whole budget burned before the pump even ran
                // (no event was queued and the session was not
                // already dead): retire a blackholed session — see
                // `retire_if_no_progress`.
                self.shared.retire_if_no_progress(self.epoch_at_open);
                return Err(H2Error::from(e));
            }
            // One frame per pump, socket-bounded by the slice, so a
            // concurrent `open_request` or another owner's pump never
            // waits longer than `PUMP_SLICE` for the session lock.
            let slice_end = deadline.min(Instant::now() + PUMP_SLICE);
            // The whole pump runs under the session lock; the guard
            // drops at the end of each iteration, yielding to other
            // owners between frames.
            let pumped = {
                let inner = &self.shared.inner;
                let (mut sess, poisoned) = recover_lock_poisoned(&inner.session);
                if poisoned {
                    // A panicking pumper may have left the session
                    // logically corrupt; retire it rather than pumping
                    // more frames through a desynced decoder.
                    inner.dead.store(true, Ordering::Release);
                    return Err(H2Error::Connection(0, "session lock poisoned"));
                }
                inner.refresh_max_streams(&sess);
                // Shrink the socket timeouts for the slice: without
                // this, `Session::next_event` blocks up to the
                // dial-time read timeout on an idle socket — and its
                // synchronous frame writes (SETTINGS ACKs, PING ACKs,
                // WINDOW_UPDATEs) could hold the session lock up to
                // the dial-time *write* timeout against a peer that
                // stopped reading. The absolute read deadline keeps a
                // dripping peer from stretching one TLS record past
                // the slice (plain reads are per-read bounded
                // already). The write bound is the time left in the
                // slice: the ACK-sized writes either fit in it or the
                // peer is not draining, and no later owner may inherit
                // the slice bound.
                if let Err(e) = sess.conn_mut().set_read_timeout(Some(PUMP_SLICE)) {
                    inner.dead.store(true, Ordering::Release);
                    return Err(H2Error::Io(e));
                }
                let write_bound = slice_end
                    .saturating_duration_since(Instant::now())
                    // A zero write timeout means "block forever" to the
                    // OS, not "return immediately": floor at 1 ms.
                    .max(Duration::from_millis(1));
                if let Err(e) = sess.conn_mut().set_write_timeout(Some(write_bound)) {
                    let _ = sess.conn_mut().set_read_timeout(Some(inner.rw_timeout));
                    inner.dead.store(true, Ordering::Release);
                    return Err(H2Error::Io(e));
                }
                sess.conn_mut().set_read_deadline(Some(slice_end));
                let event = sess.next_event(slice_end);
                // The pump may have absorbed a SETTINGS: refresh the
                // cached cap before any reservation decision reads it.
                inner.refresh_max_streams(&sess);
                // Restore the dial-time timeouts on every exit: no
                // later pump or exchange may inherit the slice bound.
                sess.conn_mut().set_read_deadline(None);
                let _ = sess.conn_mut().set_write_timeout(Some(inner.rw_timeout));
                let _ = sess.conn_mut().set_read_timeout(Some(inner.rw_timeout));
                match event {
                    Ok(ev) if ev.stream_id() != self.stream => {
                        // Forward under the session lock so same-stream
                        // events keep wire order. Lock order is session
                        // -> forward, never the reverse.
                        inner.forward(ev);
                        None
                    }
                    event => Some(event),
                }
            };
            match pumped {
                None => continue,
                Some(Ok(ev)) => {
                    let inner = &self.shared.inner;
                    inner.touch();
                    // Progress proof for the blackhole retire check:
                    // any event delivered to any stream advances the
                    // session epoch, so a fetch that burns its whole
                    // budget with the epoch unmoved observed zero
                    // bytes for everyone — not a slow peer.
                    inner.progress_epoch.fetch_add(1, Ordering::Release);
                    self.note_event(&ev);
                    return Ok(ev);
                }
                Some(Err(H2Error::Io(e)))
                    if matches!(
                        e.kind(),
                        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                    ) =>
                {
                    // Slice expired with no frame. Linux surfaces a
                    // socket read timeout as `WouldBlock` (EAGAIN),
                    // not `TimedOut`: both mean the slice bound fired.
                    if Instant::now() >= deadline {
                        // Our fetch ran out of time. When no stream —
                        // this one or any other owner's — received a
                        // single event during the whole budget, the
                        // peer is a blackhole: retire the session so
                        // the next fetch dials fresh instead of
                        // re-paying up to `fetch_total` (and `Drop`'s
                        // `touch` keeps the reaper from ever seeing it
                        // as stale on a hot origin). When something
                        // did progress, the session stays pooled for
                        // the other streams; this fetch simply lost
                        // its budget.
                        self.shared.retire_if_no_progress(self.epoch_at_open);
                        return Err(H2Error::Io(e));
                    }
                    // Slice expired with no frame: yield the lock and
                    // pump again.
                }
                Some(Err(e)) => {
                    self.shared.mark_dead();
                    return Err(e);
                }
            }
        }
    }

    fn goaway(&self) -> Option<(u32, u32)> {
        let (sess, poisoned) = recover_lock_poisoned(&self.shared.inner.session);
        if poisoned {
            // A corrupt session may report a stale GOAWAY — or panic
            // reading one — so retire it instead of trusting the read.
            self.shared.mark_dead();
            return None;
        }
        sess.goaway()
    }
}

impl Drop for SharedStream {
    fn drop(&mut self) {
        let inner = &self.shared.inner;
        // Unregister first, without holding the session lock (lock
        // order is session -> forward, never the reverse).
        recover_lock(&inner.forward).remove(&self.stream);
        if !self.completed {
            // Abandoned mid-response: reset the stream so the peer
            // stops sending for it, and transition the session's entry
            // to Closed so it no longer counts against the peer's
            // MAX_CONCURRENT_STREAMS backstop in `Session::open_stream`
            // (which the pool-slot accounting alone does not cover).
            // Best effort: the session may already be dead. The write
            // is bounded — the peer may have stopped reading, and a
            // drop must not stall on the dial-time write timeout —
            // and the dial-time timeout is restored afterwards.
            // `send_rst` marks the entry Closed before releasing the
            // slot below.
            let (mut sess, poisoned) = recover_lock_poisoned(&inner.session);
            if poisoned {
                // A corrupt session must not see more I/O: skip the
                // RST and retire the session; the socket closes when
                // the last `Arc` drops.
                inner.dead.store(true, Ordering::Release);
            } else {
                let _ = sess.conn_mut().set_write_timeout(Some(RST_WRITE_TIMEOUT));
                let _ = sess.send_rst(self.stream, E_CANCEL);
                let _ = sess.conn_mut().set_write_timeout(Some(inner.rw_timeout));
            }
        }
        let was_last = inner.in_flight.fetch_sub(1, Ordering::AcqRel) == 1;
        inner.touch();
        if was_last {
            // Closed stream state is reaped so a long-lived session
            // cannot accumulate one map entry per fetch. A poisoned
            // lock means the session is already retired above; reaping
            // a corrupt stream map would only risk another panic.
            let (mut sess, poisoned) = recover_lock_poisoned(&inner.session);
            if poisoned {
                inner.dead.store(true, Ordering::Release);
            } else {
                sess.reap_closed_streams();
            }
        }
    }
}

/// Idle-connection state (HTTP/1.1 keep-alive entries and pooled shared
/// H2 sessions), under its own lock. Only `take`/`take_shared_h2`/`put`
/// /`put_shared_h2` touch it; ticket operations use the separate ticket
/// lock, so the two are never nested.
struct IdleState {
    idle: HashMap<PoolKey, Vec<IdleConn>>,
    shared: HashMap<PoolKey, Vec<SharedH2>>,
    /// Idle connections across both maps; the global cap covers H1 and
    /// H2 together. Maintained incrementally on every put/take/evict;
    /// `reap_stale` recomputes it from the actual list lengths to heal
    /// any drift a mid-update panic left behind.
    idle_total: usize,
    /// When the idle maps were last fully reaped. Reaping is lazy and
    /// throttled (see `reap_stale`): there is deliberately no
    /// background reaper thread — pool residency is only revisited on
    /// `put`, so a quiet origin's entries are dropped the next time
    /// anything is pooled instead of by a timer.
    last_reap: Instant,
}

/// TLS session-ticket cache, under its own lock. Only `ticket` and
/// `store_tickets` touch it.
struct TicketState {
    tickets: HashMap<PoolKey, VecDeque<CachedTicket>>,
}

impl IdleState {
    /// Decrement the idle census. Every removed entry was counted in
    /// `idle_total`, so this must never underflow: the debug assertion
    /// surfaces accounting drift in test builds, and `saturating_sub`
    /// keeps a release build from wrapping to `usize::MAX` if drift
    /// ever slips through — a wrap would permanently trip the
    /// `idle_total >= MAX_IDLE_TOTAL` eviction loop and silently
    /// disable pooling.
    fn dec_idle_total(&mut self, n: usize) {
        debug_assert!(self.idle_total >= n);
        self.idle_total = self.idle_total.saturating_sub(n);
    }

    /// Evict idle entries older than `MAX_IDLE_AGE` from both maps.
    /// Throttled to one full scan per `REAP_INTERVAL`: a reap is
    /// O(idle entries), and `put` is hot enough that scanning on every
    /// call would be wasteful. Entries this stale are never handed out
    /// (`take` drops them on sight, `take_shared_h2` fails their
    /// probe); the reaper exists so a quiet origin's entries do not pin
    /// sockets until a take happens to pop them. Dropped entries are
    /// simply closed — their session tickets were already harvested at
    /// `put` time, so nothing is lost.
    ///
    /// Returns the evicted entries instead of dropping them: the caller
    /// must retire the returned entries *after* releasing the idle
    /// lock. Dropping a TLS connection performs a blocking
    /// `close_notify` write (bounded by the socket write timeout), and
    /// doing that under the pool mutex — in bulk, for N stale sessions
    /// — would serialize every concurrent checkout behind it.
    fn reap_stale(&mut self) -> Vec<Evicted> {
        let now = Instant::now();
        if now.duration_since(self.last_reap) < REAP_INTERVAL {
            return Vec::new();
        }
        self.last_reap = now;
        let mut evicted = Vec::new();
        for list in self.idle.values_mut() {
            let mut i = 0;
            while i < list.len() {
                if now.duration_since(list[i].idle_since) > MAX_IDLE_AGE {
                    evicted.push(Evicted::Conn(list.remove(i).conn));
                } else {
                    i += 1;
                }
            }
        }
        self.idle.retain(|_, list| !list.is_empty());
        // Shared sessions: evict when dead, or when no stream is open
        // and the session sat idle past `MAX_IDLE_AGE`. Sessions with
        // in-flight streams are never reaped — their owners still hold
        // `Arc`s, so eviction only stops *new* streams from being
        // handed out; leaving active sessions pooled keeps the scan
        // honest about residency.
        let mut empty_keys = Vec::new();
        for (k, list) in self.shared.iter_mut() {
            let mut i = 0;
            while i < list.len() {
                let inner = &list[i].inner;
                let dead = inner.dead.load(Ordering::Acquire);
                let stale = !dead
                    && inner.in_flight.load(Ordering::Acquire) == 0
                    && recover_lock(&inner.last_activity).elapsed() > MAX_IDLE_AGE;
                if dead || stale {
                    evicted.push(Evicted::Shared(k.clone(), list.remove(i)));
                } else {
                    i += 1;
                }
            }
            if list.is_empty() {
                empty_keys.push(k.clone());
            }
        }
        for k in empty_keys {
            self.shared.remove(&k);
        }
        // Self-healing census: `idle_total` is maintained incrementally
        // on every put/take/evict, but a panic between a list mutation
        // and its counter update would skew it permanently —
        // under-eviction (unbounded growth) if the counter reads low,
        // over-eviction if it reads high, and in debug builds the
        // `debug_assert!` in `dec_idle_total` turning one historical
        // drift into recurring panics. The reap just walked both maps,
        // so recompute the census from the actual list lengths instead
        // of trusting the counter: one historical drift heals here
        // instead of compounding, and the throttled reap keeps the
        // O(entries) rescan cheap.
        self.idle_total = self.idle.values().map(Vec::len).sum::<usize>()
            + self.shared.values().map(Vec::len).sum::<usize>();
        evicted
    }

    /// Evict the single oldest idle entry across the H1 and shared-H2
    /// maps. Returns the evicted entry, or `None` when there is nothing
    /// to evict.
    ///
    /// The caller must retire the returned entry *after* releasing the
    /// idle lock: a TLS drop performs a blocking `close_notify` write
    /// (bounded by the socket write timeout), and doing that under the
    /// pool mutex would serialize every concurrent checkout/return
    /// behind it.
    ///
    /// The scan borrows the keys and clones only the winner's: cloning
    /// every pooled origin's key on each eviction was one allocation
    /// per origin under the idle lock, on a path that runs exactly when
    /// the pool is under pressure.
    fn evict_oldest(&mut self) -> Option<Evicted> {
        // Each per-origin H1 list is LIFO (push back, pop back), so its
        // front is that origin's oldest entry. Shared entries are *not*
        // age-ordered — stream activity refreshes them — so the scan
        // finds the oldest by `last_activity`. On timestamp ties the
        // shared entry wins, matching the old H1-then-H2 chained
        // `min_by_key` scan: `Iterator::min_by_key` returns the *last*
        // minimum, so a tie against an H1 front picked the H2 entry.
        // The H1 loop keeps the first H1 minimum with strict `<`; the
        // shared loop uses `<=` so a shared entry at or before the
        // current best displaces it.
        let mut oldest: Option<(&PoolKey, usize, Instant, bool)> = None;
        for (k, v) in &self.idle {
            if let Some(ic) = v.first() {
                let replace = match oldest {
                    None => true,
                    Some((_, _, t, _)) => ic.idle_since < t,
                };
                if replace {
                    oldest = Some((k, 0, ic.idle_since, false));
                }
            }
        }
        for (k, v) in &self.shared {
            for (i, shared) in v.iter().enumerate() {
                let t = *recover_lock(&shared.inner.last_activity);
                let replace = match oldest {
                    None => true,
                    Some((_, _, ot, _)) => t <= ot,
                };
                if replace {
                    oldest = Some((k, i, t, true));
                }
            }
        }
        let (key, idx, _, is_shared) = oldest?;
        // The borrow ends here: only the winner's key is cloned.
        let key = key.clone();
        // Unreachable in practice: the key came from this map a moment
        // ago and the idle lock was held throughout.
        let evicted = if is_shared {
            let (shared, now_empty) = {
                let list = self.shared.get_mut(&key)?;
                (list.remove(idx), list.is_empty())
            };
            self.dec_idle_total(1);
            if now_empty {
                self.shared.remove(&key);
            }
            Evicted::Shared(key, shared)
        } else {
            let (conn, now_empty) = {
                let list = self.idle.get_mut(&key)?;
                (list.remove(0).conn, list.is_empty())
            };
            self.dec_idle_total(1);
            if now_empty {
                self.idle.remove(&key);
            }
            Evicted::Conn(conn)
        };
        Some(evicted)
    }
}

/// Lock a pool or pooled-session mutex, recovering from poisoning
/// instead of panicking. The pool is process-wide (`global_pool()`),
/// so a panic in any fetch thread while holding the idle, ticket,
/// session, forward, or last-activity lock poisoned the mutex and
/// every later `.lock().unwrap()` panicked too — permanently killing
/// the browser's networking on the first mid-critical-section panic.
///
/// Poison means a thread panicked mid-critical-section, so the
/// guarded state may be inconsistent. The recovery posture differs by
/// lock:
///
/// * The idle, ticket, `forward`, and last-activity locks guard state
///   that is re-validated before use: take-time liveness probes reject
///   a corrupted connection, the ticket path evicts expired entries on
///   sight, and the forward map is rebuilt idempotently. Reusing a
///   recovered guard there can at worst hand out one extra dead entry,
///   which is then probed and dropped — not a panic.
/// * The shared-H2 *session* lock is the exception. A panic mid-pump
///   can leave a TCP-live session logically corrupt — a half-fed
///   HPACK decoder, a desynced `streams` map or `next_stream`, skewed
///   flow-control windows — and the take-time probe cannot see any of
///   that: it only checks GOAWAY, the dead flag, a peek probe (fresh
///   path only), and a fire-and-forget PING (warm path). A
///   logically corrupt session is worse than a dead one, so every
///   session-lock site uses [`recover_lock_poisoned`] and retires the
///   session (`mark_dead`) when the lock was poisoned, instead of
///   handing it out.
fn recover_lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Poison-aware variant of [`recover_lock`]: returns the recovered
/// guard plus whether the lock was poisoned. The caller must act on
/// the flag — for the shared-H2 session lock that means retiring the
/// session (see above), since a recovered session may be logically
/// corrupt in ways the take-time probe cannot detect. The other locks
/// use plain [`recover_lock`]: their guarded state is re-validated
/// before use, so reuse is safe there.
fn recover_lock_poisoned<T>(m: &Mutex<T>) -> (std::sync::MutexGuard<'_, T>, bool) {
    match m.lock() {
        Ok(guard) => (guard, false),
        Err(poisoned) => (poisoned.into_inner(), true),
    }
}

/// Bounded pool of keep-alive connections and TLS session tickets.
/// The idle-connection state and the ticket cache live under separate
/// locks: checkout/return only ever takes the idle lock, ticket
/// offer/store only the ticket lock, and the two are never held at
/// once — so splitting cannot deadlock and neither path blocks the
/// other.
pub(crate) struct Pool {
    idle: Mutex<IdleState>,
    tickets: Mutex<TicketState>,
}

impl Pool {
    pub(crate) fn new() -> Self {
        Self {
            idle: Mutex::new(IdleState {
                idle: HashMap::new(),
                shared: HashMap::new(),
                idle_total: 0,
                last_reap: Instant::now(),
            }),
            tickets: Mutex::new(TicketState {
                tickets: HashMap::new(),
            }),
        }
    }

    /// Take an idle connection for `key`, if one is pooled and alive.
    /// Expired entries are dropped; each candidate is liveness-probed
    /// outside the lock, so the pool mutex is never held across I/O.
    pub(crate) fn take(&self, key: &PoolKey) -> Option<Conn> {
        loop {
            // Pop one candidate under the lock; the probe below runs
            // outside it.
            let candidate = {
                let mut st = recover_lock(&self.idle);
                match st.idle.get_mut(key).and_then(|list| list.pop()) {
                    Some(ic) => {
                        st.dec_idle_total(1);
                        if st.idle.get(key).is_some_and(|list| list.is_empty()) {
                            st.idle.remove(key);
                        }
                        Some(ic)
                    }
                    None => {
                        st.idle.remove(key);
                        None
                    }
                }
            };
            let ic = candidate?;
            let mut conn = ic.conn;
            if ic.idle_since.elapsed() > MAX_IDLE_AGE {
                // Too old to reuse, but its tickets may still be live:
                // harvest them for a fresh connection.
                self.store_tickets(key, conn.take_tls_tickets());
                continue;
            }
            if conn.probe() {
                return Some(conn);
            }
            // Dead on arrival: its tickets are still valid for a fresh
            // connection, so harvest them before dropping it, then try
            // the next candidate.
            self.store_tickets(key, conn.take_tls_tickets());
        }
    }

    /// Take a shared H2 session for `key`: concurrent fetches to one
    /// origin multiplex streams on the same connection instead of each
    /// dialing their own. The pool entry is *not* removed while streams
    /// are open — sharing means the session stays available while in
    /// use — so the returned handle is a clone of the pooled `Arc`.
    ///
    /// Returns `None` when no pooled session has a free stream slot
    /// (the caller falls back to the H1 pool, then a fresh dial). Dead
    /// entries are reaped on sight. Liveness is probed outside the lock
    /// via [`SharedH2::probe`]: a non-blocking peek for freshly idle
    /// sessions, a fire-and-forget PING past the warm age — the take
    /// never waits for the ACK, so a checkout costs no round trip.
    /// Tickets are harvested from dropped and probed candidates,
    /// exactly like [`Pool::take`].
    ///
    /// `fetch_total` bounds the take-time PING write (see
    /// [`SharedH2::probe`]): a checkout near the fetch deadline cannot
    /// stall on the dial-time write timeout.
    pub(crate) fn take_shared_h2(&self, key: &PoolKey, fetch_total: Duration) -> Option<SharedH2> {
        loop {
            // Scan for a live session with a free slot under the lock;
            // the probe below runs outside it. Dead entries are
            // collected for retirement *after* the idle guard is
            // released: a dead session's last `Arc` may be the pooled
            // one, and dropping it performs a blocking TLS
            // close_notify write that must not serialize checkouts
            // behind the pool mutex.
            let (candidate, evicted) = {
                let mut st = recover_lock(&self.idle);
                let scanned = (|| {
                    let list = st.shared.get_mut(key)?;
                    let mut pick: Option<SharedH2> = None;
                    let mut dead: Vec<Evicted> = Vec::new();
                    list.retain(|shared| {
                        if shared.is_dead() {
                            dead.push(Evicted::Shared(key.clone(), shared.clone()));
                            false
                        } else {
                            if pick.is_none() && shared.has_free_slot() {
                                pick = Some(shared.clone());
                            }
                            true
                        }
                    });
                    Some((pick, dead, list.is_empty()))
                })();
                let (pick, dead, empty) = scanned?;
                st.dec_idle_total(dead.len());
                if empty {
                    st.shared.remove(key);
                }
                drop(st);
                (pick, dead)
            };
            // Retired after the guard is released: tickets are
            // harvested into the cache first, then the entries drop.
            self.retire_evicted(evicted);
            let shared = candidate?;
            if shared.probe(fetch_total) {
                // A probe absorbs waiting NewSessionTickets into the
                // transport; hand them to the cache like `take` does.
                self.store_tickets(key, shared.harvest_tickets());
                return Some(shared);
            }
            // Dead on arrival: drop the pool entry so it is never
            // handed out again. In-flight streams fail on their own via
            // the dead flag and release their slots when dropped.
            shared.mark_dead();
            self.store_tickets(key, shared.harvest_tickets());
            let mut st = recover_lock(&self.idle);
            let (removed, empty) = (|| {
                let list = st.shared.get_mut(key)?;
                let before = list.len();
                list.retain(|e| !Arc::ptr_eq(&e.inner, &shared.inner));
                Some((before - list.len(), list.is_empty()))
            })()
            .unwrap_or((0, false));
            st.dec_idle_total(removed);
            if empty {
                st.shared.remove(key);
            }
        }
    }

    /// Return a connection after a complete, reusable exchange. There is
    /// no put-time liveness probe: the exchange just succeeded, so
    /// re-probing would only cost syscalls (and a potential TLS record
    /// decrypt); `take` probes before reuse instead. Session tickets
    /// the exchange received are harvested into the ticket cache.
    pub(crate) fn put(&self, key: &PoolKey, mut conn: Conn) {
        self.store_tickets(key, conn.take_tls_tickets());
        let mut st = recover_lock(&self.idle);
        // Opportunistic reap: evict entries no take will ever hand out
        // before enforcing the caps, so the caps count live entries.
        // The reaper returns the evicted transports instead of dropping
        // them — a TLS drop performs a blocking close_notify write —
        // so they are dropped only after the guard is released below.
        let reaped = st.reap_stale();
        // Per-origin bound first: never evict another origin's
        // connection to make room for one this origin cannot keep.
        if st
            .idle
            .get(key)
            .is_some_and(|list| list.len() >= MAX_IDLE_PER_ORIGIN)
        {
            // Release the guard before the evicted entries drop: their
            // `Drop` may block on a close_notify write.
            drop(st);
            drop(reaped);
            return;
        }
        // Enforce the global bound by evicting the single oldest idle
        // entry, wherever it lives. The evicted entries are retired
        // after the guard is released below: their `Drop` may block on
        // a close_notify write.
        let mut evicted = Vec::new();
        while st.idle_total >= MAX_IDLE_TOTAL {
            let Some(e) = st.evict_oldest() else {
                break;
            };
            evicted.push(e);
        }
        let list = st.idle.entry(key.clone()).or_default();
        // LIFO: the most recently used connection is the most likely to
        // still be alive, so it is popped first.
        list.push(IdleConn {
            conn,
            idle_since: Instant::now(),
        });
        st.idle_total += 1;
        drop(st);
        self.retire_evicted(reaped);
        self.retire_evicted(evicted);
    }

    /// Pool a fresh H2 session as a shared connection: the next fetch
    /// to this origin multiplexes a new stream on it instead of
    /// dialing. There is no put-time probe — the exchange just
    /// succeeded — and closed stream state is reaped on wrap. A session
    /// the server is already shutting down, or one beyond the
    /// per-origin shared-session cap, is closed instead of pooled;
    /// session tickets are harvested either way.
    ///
    /// `rw_timeout` is the effective socket timeout the session was
    /// dialed with (`Timeouts.rw`); it is snapshotted here because the
    /// pump slice shrinks the timeouts and must restore this value
    /// afterwards.
    pub(crate) fn put_shared_h2(
        &self,
        key: &PoolKey,
        mut session: Session<Conn>,
        rw_timeout: Duration,
    ) {
        self.store_tickets(key, session.conn_mut().take_tls_tickets());
        if session.goaway().is_some() {
            return;
        }
        let shared = SharedH2::new(session, rw_timeout);
        let mut st = recover_lock(&self.idle);
        // Opportunistic reap, mirroring `put`: the evicted entries are
        // retired after the guard is released (see `put`).
        let reaped = st.reap_stale();
        // Drop dead entries before enforcing the per-origin cap so a
        // dead session never blocks a live one from pooling. The purged
        // entries are collected for retirement *after* the idle guard is
        // released: a purged session's last `Arc` may be the pooled
        // one, and dropping it performs a blocking TLS close_notify
        // write that must not serialize checkouts behind the pool
        // mutex.
        let mut purged = Vec::new();
        let emptied = (|| {
            let list = st.shared.get_mut(key)?;
            let mut i = 0;
            while i < list.len() {
                if list[i].is_dead() {
                    purged.push(Evicted::Shared(key.clone(), list.remove(i)));
                } else {
                    i += 1;
                }
            }
            Some(list.is_empty())
        })()
        .unwrap_or(false);
        st.dec_idle_total(purged.len());
        if emptied {
            st.shared.remove(key);
        }
        // Per-origin bound first: never evict another origin's entry to
        // make room for one this origin cannot keep.
        if st
            .shared
            .get(key)
            .is_some_and(|list| list.len() >= MAX_SHARED_H2_PER_ORIGIN)
        {
            drop(st);
            self.retire_evicted(reaped);
            self.retire_evicted(purged);
            return;
        }
        // Enforce the global bound by evicting the single oldest idle
        // entry, wherever it lives.
        let mut evicted = Vec::new();
        while st.idle_total >= MAX_IDLE_TOTAL {
            let Some(e) = st.evict_oldest() else {
                break;
            };
            evicted.push(e);
        }
        st.shared.entry(key.clone()).or_default().push(shared);
        st.idle_total += 1;
        drop(st);
        self.retire_evicted(reaped);
        self.retire_evicted(purged);
        self.retire_evicted(evicted);
    }

    /// Retire evicted entries after the idle guard is released: a TLS
    /// drop performs a blocking close_notify write, and a shared
    /// session's last `Arc` may be this one. Tickets from evicted
    /// shared sessions go back to the cache first — they are still
    /// valid for fresh connections.
    fn retire_evicted(&self, evicted: Vec<Evicted>) {
        for e in evicted {
            match e {
                Evicted::Conn(conn) => drop(conn),
                Evicted::Shared(key, shared) => {
                    self.store_tickets(&key, shared.harvest_tickets());
                    drop(shared);
                }
            }
        }
    }

    /// Discard a connection that must not be reused (`Connection:
    /// close`, close-delimited body, framing error, ...). Tickets are
    /// still harvested: a later connection can resume from them.
    pub(crate) fn discard(&self, key: &PoolKey, mut conn: Conn) {
        let tickets = conn.take_tls_tickets();
        self.store_tickets(key, tickets);
    }

    /// The best offerable session ticket for `key`, if any. The newest
    /// live ticket is offered, since it is the least likely to have
    /// expired server-side; offering consumes it, so a ticket is never
    /// offered twice. Expired tickets are evicted instead of offered.
    pub(crate) fn ticket(&self, key: &PoolKey) -> Option<PskOffer> {
        let now = now_ms();
        let mut st = recover_lock(&self.tickets);
        let queue = st.tickets.get_mut(key)?;
        while let Some(back) = queue.back() {
            if !back.is_live(now) {
                // Expired: evict and consider the next-newest.
                queue.pop_back();
                continue;
            }
            // Live: pop it, then build the offer exactly once — `offer`
            // clones the ticket identity and PSK, so peeking with it
            // would pay those allocations twice per handshake.
            let fresh = queue.pop_back();
            if queue.is_empty() {
                st.tickets.remove(key);
            }
            // The pool lock is held, so `back()` being `Some` guarantees
            // `pop_back()` is `Some`; still, degrade to "no ticket"
            // instead of panicking if that ever stops holding.
            return fresh.and_then(|t| t.offer(now));
        }
        st.tickets.remove(key);
        None
    }

    fn store_tickets(&self, key: &PoolKey, tickets: Vec<CachedTicket>) {
        if tickets.is_empty() {
            return;
        }
        let mut st = recover_lock(&self.tickets);
        let queue = st.tickets.entry(key.clone()).or_default();
        for t in tickets {
            if queue.len() >= MAX_TICKETS_PER_ORIGIN {
                queue.pop_front();
            }
            queue.push_back(t);
        }
    }

    #[cfg(test)]
    fn idle_count(&self, key: &PoolKey) -> usize {
        recover_lock(&self.idle)
            .idle
            .get(key)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Test-only: seed the ticket cache as if a finished connection's
    /// `NewSessionTicket`s had been harvested on discard.
    #[cfg(test)]
    pub(crate) fn test_seed_tickets(&self, key: &PoolKey, tickets: Vec<CachedTicket>) {
        self.store_tickets(key, tickets);
    }

    #[cfg(test)]
    fn ticket_count(&self, key: &PoolKey) -> usize {
        recover_lock(&self.tickets)
            .tickets
            .get(key)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    #[cfg(test)]
    fn has_ticket_entry(&self, key: &PoolKey) -> bool {
        recover_lock(&self.tickets).tickets.contains_key(key)
    }

    #[cfg(test)]
    fn shared_h2_count(&self, key: &PoolKey) -> usize {
        recover_lock(&self.idle)
            .shared
            .get(key)
            .map(|v| v.len())
            .unwrap_or(0)
    }
}

/// The process-wide pool: every public fetch funnels through it, so both
/// the redirect hops of a single fetch and standalone `fetch` calls made
/// one after another share keep-alive connections and session tickets.
pub(crate) fn global_pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(Pool::new)
}

/// Monotonic ticket clock. `tls.rs` stamps `CachedTicket::received_at_ms`
/// with the wall clock, so the pool's expiry comparison must stay on the
/// unix-millis scale — but re-reading `SystemTime` for every comparison
/// lets a backward wall-clock step (NTP correction, manual change)
/// rewind ticket ages and resurrect expired tickets.
///
/// `now_ms()` therefore returns the maximum of two values:
/// * the epoch origin — one wall snapshot advanced by monotonic
///   `Instant` elapsed time — which never moves backward on its own;
/// * a fresh wall-clock read — so the result never lags the wall
///   stamps `tls.rs` writes (a lagging clock would see a just-created
///   zero-lifetime ticket as live).
///
/// Deliberately *not* floored by the last value ever returned: a floor
/// would pin a forward wall spike. After the wall corrects back down,
/// every ticket would read expired for as long as the spike lasted,
/// silently disabling resumption for hours. Instead a forward step is
/// honored only while it lasts — tickets then only expire *sooner*,
/// the safe direction — and the clock drops back to the monotonic base
/// the moment the wall does. That only un-expires tickets whose true
/// (monotonic) age never passed their lifetime. A backward wall step
/// can only pull the clock down to the monotonic base, never below it,
/// so expiry keeps aging instead of rewinding.
fn ticket_epoch() -> (Instant, u64) {
    static EPOCH: OnceLock<(Instant, u64)> = OnceLock::new();
    *EPOCH.get_or_init(|| {
        (
            Instant::now(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        )
    })
}

fn now_ms() -> u64 {
    let (epoch, wall_ms) = ticket_epoch();
    let elapsed_ms = u64::try_from(epoch.elapsed().as_millis()).unwrap_or(u64::MAX);
    let mono = wall_ms.saturating_add(elapsed_ms);
    let wall = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    clock_now_ms(mono, wall)
}

/// Combine the monotonic unix-millis estimate with a fresh wall read
/// into the ticket clock. The monotonic component is non-decreasing by
/// construction (`Instant` never runs backward), so no atomic floor is
/// needed — and none is kept, because a floor would pin a forward wall
/// spike past its correction. Pure, so tests can drive wall spikes and
/// corrections directly.
fn clock_now_ms(mono_ms: u64, wall_ms: u64) -> u64 {
    mono_ms.max(wall_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::SOCKET_TIMEOUT;
    use crate::tls::CachedTicket;
    use std::sync::Arc;

    fn key() -> PoolKey {
        PoolKey::new(true, "Example.COM", 443)
    }

    /// Generous fetch budget for tests that do not exercise the
    /// deadline: large enough that no take-time bound interferes.
    const FETCH_BUDGET: Duration = Duration::from_secs(30);

    fn socketpair() -> (TcpStream, TcpStream) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        (client, server)
    }

    #[test]
    fn pool_key_normalizes_host_case() {
        assert_eq!(
            PoolKey::new(false, "EXAMPLE.com", 80),
            PoolKey::new(false, "example.com", 80)
        );
        // Same shared canonicalization as cookies/HSTS: a single
        // trailing dot is stripped too.
        assert_eq!(
            PoolKey::new(false, "EXAMPLE.com.", 80),
            PoolKey::new(false, "example.com", 80)
        );
        assert_ne!(
            PoolKey::new(false, "example.com", 80),
            PoolKey::new(false, "example.com", 8080)
        );
        assert_ne!(
            PoolKey::new(false, "example.com", 443),
            PoolKey::new(true, "example.com", 443)
        );
        assert_ne!(
            PoolKey::new(false, "a.com", 80),
            PoolKey::new(false, "b.com", 80)
        );
    }

    #[test]
    fn take_empty_pool_returns_none() {
        let pool = Pool::new();
        assert!(pool.take(&key()).is_none());
    }

    #[test]
    fn put_then_take_roundtrip() {
        let pool = Pool::new();
        let (client, _server) = socketpair();
        pool.put(&key(), Conn::Plain(client));
        assert_eq!(pool.idle_count(&key()), 1);
        assert!(pool.take(&key()).is_some());
        assert_eq!(pool.idle_count(&key()), 0);
        assert!(pool.take(&key()).is_none());
    }

    #[test]
    fn take_drops_dead_connection() {
        let pool = Pool::new();
        let (client, server) = socketpair();
        pool.put(&key(), Conn::Plain(client));
        // The peer goes away: the probe must report the pooled
        // connection dead instead of handing it out.
        drop(server);
        // Give the FIN a moment to arrive; the probe is deterministic
        // either way (a live socket reports alive, a dead one is
        // dropped), but without this the test would be timing-sensitive.
        std::thread::sleep(Duration::from_millis(50));
        assert!(pool.take(&key()).is_none());
        assert_eq!(pool.idle_count(&key()), 0);
    }

    #[test]
    fn probe_reports_idle_socket_alive() {
        let (client, _server) = socketpair();
        let mut conn = Conn::Plain(client);
        assert!(conn.probe());
    }

    #[test]
    fn probe_reports_closed_peer_dead() {
        let (client, server) = socketpair();
        drop(server);
        std::thread::sleep(Duration::from_millis(50));
        let mut conn = Conn::Plain(client);
        assert!(!conn.probe());
    }

    #[test]
    fn probe_does_not_consume_unsolicited_bytes() {
        let (client, mut server) = socketpair();
        // A pooled peer that speaks first: the probe must report the
        // connection unusable (we never pipeline) WITHOUT consuming
        // the byte — the old read-based probe discarded it and desynced
        // the next response.
        server.write_all(b"X").unwrap();
        std::thread::sleep(Duration::from_millis(50));
        let mut conn = Conn::Plain(client);
        assert!(!conn.probe(), "unsolicited data must fail the probe");
        let mut buf = [0u8; 1];
        conn.read(&mut buf)
            .expect("peeked byte must still be there");
        assert_eq!(&buf, b"X");
    }

    #[test]
    fn per_origin_idle_cap() {
        let pool = Pool::new();
        // Hold the server halves so every pooled connection is still
        // alive: `put` no longer probes, and a dead pooled entry would
        // just be evicted by the take-time probe.
        let pairs: Vec<(TcpStream, TcpStream)> =
            (0..MAX_IDLE_PER_ORIGIN + 2).map(|_| socketpair()).collect();
        for (client, _server) in pairs {
            pool.put(&key(), Conn::Plain(client));
        }
        assert_eq!(pool.idle_count(&key()), MAX_IDLE_PER_ORIGIN);
    }

    #[test]
    fn global_idle_cap_evicts_oldest() {
        let pool = Pool::new();
        // Hold the server halves so every pooled connection is still
        // alive: `put` no longer probes, and a dead pooled entry would
        // just be evicted by the take-time probe.
        let mut pairs = Vec::new();
        for i in 0..MAX_IDLE_TOTAL + 3 {
            let k = PoolKey::new(false, &format!("h{i}.example.com"), 80);
            let (client, server) = socketpair();
            pairs.push(server);
            pool.put(&k, Conn::Plain(client));
        }
        let total: usize = (0..MAX_IDLE_TOTAL + 3)
            .map(|i| pool.idle_count(&PoolKey::new(false, &format!("h{i}.example.com"), 80)))
            .sum();
        assert_eq!(total, MAX_IDLE_TOTAL);
    }

    #[test]
    fn discard_harvests_tickets_without_pooling() {
        let pool = Pool::new();
        let (client, _server) = socketpair();
        // Plain connections carry no tickets; the discard path must
        // simply not pool the connection.
        pool.discard(&key(), Conn::Plain(client));
        assert_eq!(pool.idle_count(&key()), 0);
    }

    #[test]
    fn tickets_cached_offered_single_use_and_expire() {
        let pool = Pool::new();
        assert!(pool.ticket(&key()).is_none());
        pool.store_tickets(&key(), vec![CachedTicket::test_ticket(3600)]);
        assert_eq!(pool.ticket_count(&key()), 1);
        let offer = pool.ticket(&key()).expect("ticket is offered");
        assert_eq!(offer.identity, b"tick");
        assert_eq!(offer.psk, vec![0xab; 32]);
        // Single-use: the offered ticket is consumed, never re-offered.
        assert!(pool.ticket(&key()).is_none());
        assert_eq!(pool.ticket_count(&key()), 0);
        // A zero-lifetime ticket is expired on arrival: evicted by the
        // lookup instead of offered.
        pool.store_tickets(&key(), vec![CachedTicket::test_ticket(0)]);
        assert_eq!(pool.ticket_count(&key()), 1);
        assert!(pool.ticket(&key()).is_none());
        assert_eq!(pool.ticket_count(&key()), 0);
    }

    #[test]
    fn ticket_cache_bounded_per_origin() {
        let pool = Pool::new();
        let tickets: Vec<CachedTicket> = (0..MAX_TICKETS_PER_ORIGIN + 4)
            .map(|_| CachedTicket::test_ticket(3600))
            .collect();
        pool.store_tickets(&key(), tickets);
        assert_eq!(pool.ticket_count(&key()), MAX_TICKETS_PER_ORIGIN);
    }

    #[test]
    fn ticket_drained_or_expired_queue_leaves_no_entry() {
        let pool = Pool::new();
        pool.store_tickets(
            &key(),
            vec![
                CachedTicket::test_ticket(3600),
                CachedTicket::test_ticket(3600),
            ],
        );
        assert!(pool.has_ticket_entry(&key()));
        assert!(pool.ticket(&key()).is_some());
        // Each ticket is offered exactly once: no double-offer.
        assert!(pool.ticket(&key()).is_some());
        assert!(pool.ticket(&key()).is_none());
        assert_eq!(pool.ticket_count(&key()), 0);
        // A fully drained queue removes the map entry instead of
        // leaking an empty queue behind.
        assert!(!pool.has_ticket_entry(&key()));
        // The expiry-evict path must clean up the same way.
        pool.store_tickets(&key(), vec![CachedTicket::test_ticket(0)]);
        assert!(pool.has_ticket_entry(&key()));
        assert!(pool.ticket(&key()).is_none());
        assert!(!pool.has_ticket_entry(&key()));
    }

    #[test]
    fn put_after_take_reregisters_slot() {
        let pool = Pool::new();
        let (client, _server) = socketpair();
        pool.put(&key(), Conn::Plain(client));
        assert_eq!(pool.idle_count(&key()), 1);
        let conn = pool.take(&key()).expect("pooled connection");
        assert_eq!(pool.idle_count(&key()), 0);
        // Returning the connection re-registers its slot exactly
        // once: no leaked slot, no double count.
        pool.put(&key(), conn);
        assert_eq!(pool.idle_count(&key()), 1);
        assert!(pool.take(&key()).is_some());
        assert_eq!(pool.idle_count(&key()), 0);
    }

    #[test]
    fn concurrent_take_put_is_safe() {
        let pool = Arc::new(Pool::new());
        let mut handles = Vec::new();
        for _ in 0..8 {
            let pool = pool.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..25 {
                    let (client, _server) = socketpair();
                    let k = key();
                    pool.put(&k, Conn::Plain(client));
                    let _ = pool.take(&k);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn concurrent_checkout_gets_distinct_connections() {
        // H1 has no head-of-line queue: concurrent fetches to one host
        // each check out their own connection (up to the per-origin
        // idle cap) instead of serializing behind a single in-flight
        // exchange. Three pooled connections and three racing checkouts
        // must hand out all three, drain the pool exactly, and never
        // hand the same connection to two threads.
        let pool = Arc::new(Pool::new());
        let k = key();
        let mut servers = Vec::new();
        for _ in 0..3 {
            let (client, server) = socketpair();
            servers.push(server); // held so the probes report alive
            pool.put(&k, Conn::Plain(client));
        }
        let mut handles = Vec::new();
        for _ in 0..3 {
            let pool = pool.clone();
            let k = k.clone();
            handles.push(std::thread::spawn(move || pool.take(&k).is_some()));
        }
        for h in handles {
            assert!(
                h.join().unwrap(),
                "each concurrent checkout must get a connection"
            );
        }
        assert_eq!(pool.idle_count(&k), 0);
        assert!(
            pool.take(&k).is_none(),
            "the pool must be drained exactly: no connection handed out twice"
        );
    }

    fn h2_session() -> (Session<Conn>, TcpStream) {
        let (client, server) = socketpair();
        (Session::new(Conn::Plain(client), 100), server)
    }

    /// Build one H2 frame: 24-bit length, type, flags, 31-bit stream
    /// id, payload.
    fn h2_frame(ty: u8, flags: u8, stream: u32, payload: &[u8]) -> Vec<u8> {
        let mut f = Vec::with_capacity(9 + payload.len());
        let len = payload.len() as u32;
        f.extend_from_slice(&len.to_be_bytes()[1..]);
        f.push(ty);
        f.push(flags);
        f.extend_from_slice(&(stream & 0x7fff_ffff).to_be_bytes());
        f.extend_from_slice(payload);
        f
    }

    /// Minimal fake H2 server for the shared-pool tests: ACKs PINGs,
    /// answers each HEADERS with `:status: 200` (indexed static-table
    /// entry 8, byte `0x88`) carrying END_STREAM, and optionally sends
    /// a SETTINGS frame with MAX_CONCURRENT_STREAMS first. Returns the
    /// answered stream ids in wire order.
    ///
    /// When `hold_open` is `Some`, the socket stays open after the
    /// last answer until the channel fires (a dropped sender unblocks
    /// it the same way) instead of closing immediately. Tests with
    /// concurrent pumps need this: a pump descheduled between its
    /// forward-channel check and its frame read would otherwise read
    /// past the written bytes into the FIN and fail with
    /// UnexpectedEof — a harness race, not a pool bug, since EOF from
    /// a closed peer is a genuine transport failure the pump must
    /// report.
    fn fake_h2_server(
        mut server: TcpStream,
        max_concurrent: Option<u32>,
        expect_streams: usize,
        hold_open: Option<std::sync::mpsc::Receiver<()>>,
    ) -> Vec<u32> {
        if let Some(max) = max_concurrent {
            let mut payload = Vec::new();
            payload.extend_from_slice(&0x3u16.to_be_bytes());
            payload.extend_from_slice(&max.to_be_bytes());
            server.write_all(&h2_frame(0x4, 0, 0, &payload)).unwrap();
        }
        let mut answered = Vec::new();
        while answered.len() < expect_streams {
            let mut hdr = [0u8; 9];
            server.read_exact(&mut hdr).unwrap();
            let len = u32::from_be_bytes([0, hdr[0], hdr[1], hdr[2]]) as usize;
            let ty = hdr[3];
            let flags = hdr[4];
            let stream = u32::from_be_bytes([hdr[5], hdr[6], hdr[7], hdr[8]]) & 0x7fff_ffff;
            let mut payload = vec![0u8; len];
            if len > 0 {
                server.read_exact(&mut payload).unwrap();
            }
            match ty {
                // PING: ACK it unless it already is one.
                0x6 if flags & 0x1 == 0 => {
                    server.write_all(&h2_frame(0x6, 0x1, 0, &payload)).unwrap();
                }
                // HEADERS: answer `:status: 200`, END_STREAM.
                0x1 => {
                    answered.push(stream);
                    server
                        .write_all(&h2_frame(0x1, 0x5, stream, &[0x88]))
                        .unwrap();
                }
                _ => {}
            }
        }
        if let Some(done) = hold_open {
            // Hold the socket open until the test is done reading; a
            // dropped sender unblocks this too, so a panicked test
            // never leaves the peer parked.
            let _ = done.recv();
        }
        answered
    }

    /// Pump a shared stream until its response HEADERS arrive; returns
    /// the stream id they carried.
    fn pump_until_headers(stream: &mut SharedStream, deadline: Instant) -> u32 {
        loop {
            match stream.next_event(deadline).expect("response event") {
                Event::Headers { stream, .. } => return stream,
                _ => continue,
            }
        }
    }

    /// Pool one fresh H2 session for `key` and age it past
    /// `H2_PING_WARM_AGE`, so the next take exercises the warm-age
    /// PING path.
    fn pool_warm_h2(pool: &Pool, session: Session<Conn>, rw_timeout: Duration) {
        pool.put_shared_h2(&key(), session, rw_timeout);
        let st = recover_lock(&pool.idle);
        let entry = st.shared.get(&key()).unwrap().first().unwrap();
        *recover_lock(&entry.inner.last_activity) =
            Instant::now() - H2_PING_WARM_AGE - Duration::from_secs(1);
    }

    #[test]
    fn put_shared_h2_then_take_roundtrip() {
        let pool = Pool::new();
        let (session, _server) = h2_session();
        pool.put_shared_h2(&key(), session, SOCKET_TIMEOUT);
        assert_eq!(pool.shared_h2_count(&key()), 1);
        // Sharing: the entry is *not* removed while checked out.
        let shared = pool
            .take_shared_h2(&key(), FETCH_BUDGET)
            .expect("pooled session");
        assert_eq!(pool.shared_h2_count(&key()), 1);
        // A concurrent take gets the *same* session, not a new dial.
        let shared2 = pool
            .take_shared_h2(&key(), FETCH_BUDGET)
            .expect("second take shares");
        assert!(Arc::ptr_eq(&shared.inner, &shared2.inner));
        assert_eq!(pool.shared_h2_count(&key()), 1);
    }

    #[test]
    fn take_shared_h2_drops_dead_session() {
        let pool = Pool::new();
        let (session, server) = h2_session();
        pool.put_shared_h2(&key(), session, SOCKET_TIMEOUT);
        drop(server);
        std::thread::sleep(Duration::from_millis(50));
        assert!(pool.take_shared_h2(&key(), FETCH_BUDGET).is_none());
        assert_eq!(pool.shared_h2_count(&key()), 0);
    }

    #[test]
    fn take_shared_h2_skips_goaway_session() {
        let pool = Pool::new();
        let (session, mut server) = h2_session();
        pool.put_shared_h2(&key(), session, SOCKET_TIMEOUT);
        // A GOAWAY frame (len 8, type 0x7, stream 0): last_stream_id 0,
        // NO_ERROR.
        let mut frame = vec![0u8, 0, 8, 0x7, 0, 0, 0, 0, 0];
        frame.extend_from_slice(&0u32.to_be_bytes());
        frame.extend_from_slice(&0u32.to_be_bytes());
        server.write_all(&frame).unwrap();
        // No more frames: after the GOAWAY the next read hits EOF.
        server.shutdown(std::net::Shutdown::Write).unwrap();
        std::thread::sleep(Duration::from_millis(50));
        // A session the server is shutting down is never handed out:
        // the take-time probe sees the unsolicited GOAWAY bytes and
        // refuses it.
        assert!(pool.take_shared_h2(&key(), FETCH_BUDGET).is_none());
        assert_eq!(pool.shared_h2_count(&key()), 0);
    }

    #[test]
    fn shared_h2_multiplexes_concurrent_streams() {
        // The core regression test: two concurrent fetches to one
        // origin share one H2 connection, each getting its own
        // stream's response while the other owner's pump forwards
        // interleaved frames.
        let pool = Pool::new();
        let (session, server) = h2_session();
        pool.put_shared_h2(&key(), session, SOCKET_TIMEOUT);
        let shared = pool
            .take_shared_h2(&key(), FETCH_BUDGET)
            .expect("pooled session");
        let shared2 = pool
            .take_shared_h2(&key(), FETCH_BUDGET)
            .expect("second take shares");
        assert!(Arc::ptr_eq(&shared.inner, &shared2.inner));

        // The server holds the socket open until the main thread says
        // the test is done reading. It used to drop the socket the
        // instant it wrote answer #2, and under parallel-test load a
        // client pump descheduled between its forward-channel check
        // (`try_recv`) and its frame read would then read past the 20
        // written bytes into the FIN and fail with UnexpectedEof — a
        // harness race, not a pool bug. (If the test panics first,
        // dropping `done_tx` unblocks the server's recv the same way,
        // so the peer never outlives the test.)
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let server_thread =
            std::thread::spawn(move || fake_h2_server(server, None, 2, Some(done_rx)));

        let deadline = Instant::now() + Duration::from_secs(30);
        let t1 = std::thread::spawn(move || {
            let mut stream = shared
                .open_request(&[], &[], deadline)
                .expect("stream 1 opens");
            let id = stream.stream_id();
            (id, pump_until_headers(&mut stream, deadline))
        });
        let t2 = std::thread::spawn(move || {
            let mut stream = shared2
                .open_request(&[], &[], deadline)
                .expect("stream 2 opens");
            let id = stream.stream_id();
            (id, pump_until_headers(&mut stream, deadline))
        });
        let (id1, got1) = t1.join().unwrap();
        let (id2, got2) = t2.join().unwrap();
        assert_ne!(id1, id2, "concurrent opens get distinct stream ids");
        assert_eq!(got1, id1, "each owner receives its own stream's response");
        assert_eq!(got2, id2);
        // Both responses are proven received: release the peer, then
        // reap it and check both streams ran on the one connection.
        let _ = done_tx.send(());
        let answered = server_thread.join().unwrap();
        assert_eq!(answered.len(), 2);
        assert!(
            answered.contains(&id1) && answered.contains(&id2),
            "both streams ran on the one connection: {answered:?}"
        );
        // Both streams ran on the one connection: the pool still holds
        // exactly one session.
        assert_eq!(pool.shared_h2_count(&key()), 1);
    }

    #[test]
    fn shared_h2_enforces_peer_stream_limit() {
        let pool = Pool::new();
        let (session, server) = h2_session();
        pool.put_shared_h2(&key(), session, SOCKET_TIMEOUT);
        let shared = pool
            .take_shared_h2(&key(), FETCH_BUDGET)
            .expect("pooled session");
        // The server caps concurrency at 1 via SETTINGS and stays up
        // for a second exchange after the slot is released.
        let server_thread = std::thread::spawn(move || fake_h2_server(server, Some(1), 2, None));

        let deadline = Instant::now() + Duration::from_secs(30);
        // The first stream opens and absorbs the SETTINGS on its first
        // pump.
        let mut s1 = shared
            .open_request(&[], &[], deadline)
            .expect("first stream opens");
        let id1 = s1.stream_id();
        assert_eq!(pump_until_headers(&mut s1, deadline), id1);
        // The peer's limit is now known: no second stream may open
        // while the first is still in flight. The refusal is
        // pre-send: no bytes left the client, so the caller may try
        // another session.
        assert!(
            matches!(
                shared.open_request(&[], &[], deadline),
                Err(OpenRequestError::PreSend(
                    H2Error::ConcurrentStreamsExceeded
                ))
            ),
            "peer MAX_CONCURRENT_STREAMS=1 must refuse a second stream"
        );
        assert!(
            pool.take_shared_h2(&key(), FETCH_BUDGET).is_none(),
            "a saturated session is not handed out"
        );
        drop(s1);
        // Slot released: the session is shareable again and serves a
        // second exchange on the same connection.
        let shared2 = pool
            .take_shared_h2(&key(), FETCH_BUDGET)
            .expect("session reusable after slot release");
        let mut s2 = shared2
            .open_request(&[], &[], deadline)
            .expect("second stream opens after release");
        assert_eq!(pump_until_headers(&mut s2, deadline), s2.stream_id());
        let answered = server_thread.join().unwrap();
        assert_eq!(answered.len(), 2);
    }

    #[test]
    fn shared_h2_per_origin_cap() {
        let pool = Pool::new();
        let pairs: Vec<(Session<Conn>, TcpStream)> = (0..MAX_SHARED_H2_PER_ORIGIN + 2)
            .map(|_| h2_session())
            .collect();
        // Hold the server halves so the sessions stay alive.
        let mut servers = Vec::new();
        for (session, server) in pairs {
            servers.push(server);
            pool.put_shared_h2(&key(), session, SOCKET_TIMEOUT);
        }
        assert_eq!(pool.shared_h2_count(&key()), MAX_SHARED_H2_PER_ORIGIN);
    }

    #[test]
    fn shared_h2_and_h1_share_global_idle_cap() {
        let pool = Pool::new();
        let mut held = Vec::new();
        // Fill the global cap with H1 connections on distinct origins.
        for i in 0..MAX_IDLE_TOTAL {
            let k = PoolKey::new(false, &format!("h{i}.example.com"), 80);
            let (client, server) = socketpair();
            held.push(server);
            pool.put(&k, Conn::Plain(client));
        }
        // A shared H2 session must evict the oldest H1 entry to fit.
        let (session, server) = h2_session();
        held.push(server);
        pool.put_shared_h2(&key(), session, SOCKET_TIMEOUT);
        assert_eq!(pool.shared_h2_count(&key()), 1);
        let total: usize = (0..MAX_IDLE_TOTAL)
            .map(|i| pool.idle_count(&PoolKey::new(false, &format!("h{i}.example.com"), 80)))
            .sum::<usize>()
            + pool.shared_h2_count(&key());
        assert_eq!(total, MAX_IDLE_TOTAL);
    }

    #[test]
    fn evict_oldest_prefers_oldest_across_h1_and_shared_h2() {
        // The global-cap eviction must remove the single oldest idle
        // entry wherever it lives: a shared H2 session idle since
        // before every H1 connection must be evicted first, even though
        // the H1 map is scanned first.
        let pool = Pool::new();
        let h2_key = PoolKey::new(true, "h2.example.com", 443);
        let (session, _server) = h2_session();
        {
            let mut st = recover_lock(&pool.idle);
            let shared = SharedH2::new(session, SOCKET_TIMEOUT);
            // Oldest entry, but not stale: under MAX_IDLE_AGE so only
            // the cap eviction (not the reaper) can remove it.
            *recover_lock(&shared.inner.last_activity) = Instant::now() - Duration::from_secs(60);
            st.shared.entry(h2_key.clone()).or_default().push(shared);
            for i in 0..MAX_IDLE_TOTAL - 1 {
                let k = PoolKey::new(false, &format!("h{i}.example.com"), 80);
                let (client, _server) = socketpair();
                st.idle.entry(k).or_default().push(IdleConn {
                    conn: Conn::Plain(client),
                    idle_since: Instant::now(),
                });
            }
            st.idle_total = MAX_IDLE_TOTAL;
        }
        // This put trips the global cap: the oldest entry — the shared
        // H2 session — must go, not a newer H1 connection.
        let (client, _server) = socketpair();
        let new_key = PoolKey::new(false, "new.example.com", 80);
        pool.put(&new_key, Conn::Plain(client));
        assert_eq!(
            pool.shared_h2_count(&h2_key),
            0,
            "the oldest entry across both maps is the shared H2 session: it must be evicted first"
        );
        let h1_total: usize = (0..MAX_IDLE_TOTAL - 1)
            .map(|i| pool.idle_count(&PoolKey::new(false, &format!("h{i}.example.com"), 80)))
            .sum();
        assert_eq!(h1_total, MAX_IDLE_TOTAL - 1);
        assert_eq!(pool.idle_count(&new_key), 1);
    }

    #[test]
    fn evict_oldest_prefers_shared_h2_on_timestamp_ties() {
        // On identical idle timestamps the shared H2 entry must win,
        // matching the old chained-H1-then-H2 `min_by_key` scan
        // (`Iterator::min_by_key` returns the last minimum on ties).
        let pool = Pool::new();
        let h1_key = PoolKey::new(false, "h1.example.com", 80);
        let h2_key = PoolKey::new(true, "h2.example.com", 443);
        let tied = Instant::now();
        let (client, _server) = socketpair();
        let (session, _srv) = h2_session();
        {
            let mut st = recover_lock(&pool.idle);
            st.idle.entry(h1_key.clone()).or_default().push(IdleConn {
                conn: Conn::Plain(client),
                idle_since: tied,
            });
            let shared = SharedH2::new(session, SOCKET_TIMEOUT);
            *recover_lock(&shared.inner.last_activity) = tied;
            st.shared.entry(h2_key.clone()).or_default().push(shared);
            st.idle_total = 2;
        }
        // Retire the evicted entry after releasing the lock, per
        // `evict_oldest`'s contract.
        let evicted = {
            let mut st = recover_lock(&pool.idle);
            st.evict_oldest()
        };
        assert!(evicted.is_some());
        pool.retire_evicted(evicted.into_iter().collect());
        assert_eq!(
            pool.shared_h2_count(&h2_key),
            0,
            "shared H2 entry must win timestamp ties"
        );
        assert_eq!(pool.idle_count(&h1_key), 1, "H1 entry must survive the tie");
    }

    #[test]
    fn reaper_evicts_stale_entries_from_both_maps() {
        let pool = Pool::new();
        let stale = Instant::now() - MAX_IDLE_AGE - Duration::from_secs(1);
        let (c1, _s1) = socketpair();
        let (c2, _s2) = socketpair();
        let (sess1, _sv1) = h2_session();
        let (sess2, _sv2) = h2_session();
        {
            let mut st = recover_lock(&pool.idle);
            let list = st.idle.entry(key()).or_default();
            list.push(IdleConn {
                conn: Conn::Plain(c1),
                idle_since: stale,
            });
            list.push(IdleConn {
                conn: Conn::Plain(c2),
                idle_since: Instant::now(),
            });
            let fresh = SharedH2::new(sess2, SOCKET_TIMEOUT);
            let stale_shared = SharedH2::new(sess1, SOCKET_TIMEOUT);
            *recover_lock(&stale_shared.inner.last_activity) = stale;
            let shlist = st.shared.entry(key()).or_default();
            shlist.push(stale_shared);
            shlist.push(fresh);
            st.idle_total = 4;
            // Force the reap due.
            st.last_reap = Instant::now() - REAP_INTERVAL - Duration::from_secs(1);
            let reaped = st.reap_stale();
            // Two stale entries evicted, returned for the caller to
            // retire after the lock is released.
            assert_eq!(reaped.len(), 2);
            assert_eq!(st.idle.get(&key()).map(|v| v.len()), Some(1));
            assert_eq!(st.shared.get(&key()).map(|v| v.len()), Some(1));
            assert_eq!(st.idle_total, 2);
            drop(reaped);
        }
        assert_eq!(pool.idle_count(&key()), 1);
        assert_eq!(pool.shared_h2_count(&key()), 1);
    }

    #[test]
    fn reaper_spares_shared_session_with_open_streams() {
        // A session with in-flight streams is never reaped, however
        // stale its last activity looks: eviction only stops *new*
        // streams from being handed out, and the owners still hold
        // their `Arc`s.
        let pool = Pool::new();
        let (session, _server) = h2_session();
        pool.put_shared_h2(&key(), session, SOCKET_TIMEOUT);
        let shared = pool
            .take_shared_h2(&key(), FETCH_BUDGET)
            .expect("pooled session");
        let deadline = Instant::now() + Duration::from_secs(30);
        let stream = shared
            .open_request(&[], &[], deadline)
            .expect("stream opens");
        // Age the session past the reap threshold while the stream is
        // open.
        *recover_lock(&shared.inner.last_activity) =
            Instant::now() - MAX_IDLE_AGE - Duration::from_secs(1);
        {
            let mut st = recover_lock(&pool.idle);
            st.last_reap = Instant::now() - REAP_INTERVAL - Duration::from_secs(1);
            let reaped = st.reap_stale();
            assert!(reaped.is_empty(), "active session must not be reaped");
            drop(reaped);
        }
        assert_eq!(pool.shared_h2_count(&key()), 1);
        drop(stream);
        // Now idle and stale: the reaper takes it.
        *recover_lock(&shared.inner.last_activity) =
            Instant::now() - MAX_IDLE_AGE - Duration::from_secs(1);
        {
            let mut st = recover_lock(&pool.idle);
            st.last_reap = Instant::now() - REAP_INTERVAL - Duration::from_secs(1);
            let reaped = st.reap_stale();
            assert_eq!(reaped.len(), 1);
            pool.retire_evicted(reaped);
        }
        assert_eq!(pool.shared_h2_count(&key()), 0);
    }

    #[test]
    fn shared_h2_idle_teardown_closes_socket() {
        // End-to-end idle teardown: once the session sits idle past
        // MAX_IDLE_AGE, the next put's opportunistic reap evicts it and
        // the socket actually closes — the peer sees EOF.
        let pool = Pool::new();
        let (session, mut server) = h2_session();
        pool.put_shared_h2(&key(), session, SOCKET_TIMEOUT);
        assert_eq!(pool.shared_h2_count(&key()), 1);
        {
            let mut st = recover_lock(&pool.idle);
            let entry = st.shared.get(&key()).unwrap().first().unwrap();
            *recover_lock(&entry.inner.last_activity) =
                Instant::now() - MAX_IDLE_AGE - Duration::from_secs(1);
            st.last_reap = Instant::now() - REAP_INTERVAL - Duration::from_secs(1);
        }
        // Any put triggers the opportunistic reap.
        let (client, _server2) = socketpair();
        pool.put(&key(), Conn::Plain(client));
        assert_eq!(pool.shared_h2_count(&key()), 0);
        // The reaped session's socket is closed: the peer sees EOF.
        server.set_nonblocking(true).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let mut one = [0u8; 1];
            match server.read(&mut one) {
                Ok(0) => break,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    assert!(
                        Instant::now() < deadline,
                        "peer never saw EOF after idle teardown"
                    );
                    std::thread::sleep(Duration::from_millis(10));
                }
                r => panic!("unexpected read after teardown: {r:?}"),
            }
        }
    }

    #[test]
    fn reaper_is_throttled() {
        let pool = Pool::new();
        let (client, _server) = socketpair();
        {
            let mut st = recover_lock(&pool.idle);
            st.idle.entry(key()).or_default().push(IdleConn {
                conn: Conn::Plain(client),
                idle_since: Instant::now() - MAX_IDLE_AGE - Duration::from_secs(1),
            });
            st.idle_total = 1;
            // `last_reap` is fresh (`Pool::new` just set it), so the
            // reap must be a no-op even though the entry is stale.
            let reaped = st.reap_stale();
            assert!(reaped.is_empty());
            assert_eq!(st.idle.get(&key()).map(|v| v.len()), Some(1));
            assert_eq!(st.idle_total, 1);
        }
    }

    #[test]
    fn put_reaps_stale_entries() {
        let pool = Pool::new();
        let (stale_client, _stale_server) = socketpair();
        {
            let mut st = recover_lock(&pool.idle);
            st.idle.entry(key()).or_default().push(IdleConn {
                conn: Conn::Plain(stale_client),
                idle_since: Instant::now() - MAX_IDLE_AGE - Duration::from_secs(1),
            });
            st.idle_total = 1;
            st.last_reap = Instant::now() - REAP_INTERVAL - Duration::from_secs(1);
        }
        let (client, _server) = socketpair();
        pool.put(&key(), Conn::Plain(client));
        // The stale entry was reaped by the put; only the fresh one
        // remains.
        assert_eq!(pool.idle_count(&key()), 1);
    }

    #[test]
    fn take_shared_h2_pings_session_idle_past_warm_age() {
        let pool = Pool::new();
        let (session, server) = h2_session();
        pool_warm_h2(&pool, session, SOCKET_TIMEOUT);
        // Fake peer: read the PING, then delay the ACK far past any
        // plausible checkout — the take must not wait for it.
        let peer = std::thread::spawn(move || {
            let mut server = server;
            let mut hdr = [0u8; 9];
            server.read_exact(&mut hdr).unwrap();
            assert_eq!(hdr[3], 0x6, "expected a PING frame");
            let len = u32::from_be_bytes([0, hdr[0], hdr[1], hdr[2]]) as usize;
            assert_eq!(len, 8);
            let mut payload = [0u8; 8];
            server.read_exact(&mut payload).unwrap();
            std::thread::sleep(Duration::from_secs(3));
            // The ACK arrives long after the take; the frame loop
            // absorbs it as a normal event.
            let _ = server.write_all(&h2_frame(0x6, 0x1, 0, &payload));
        });
        let start = Instant::now();
        let taken = pool.take_shared_h2(&key(), FETCH_BUDGET);
        let elapsed = start.elapsed();
        assert!(
            taken.is_some(),
            "a warm session whose PING goes out must be handed out without waiting for the ACK"
        );
        assert!(
            elapsed < Duration::from_secs(1),
            "warm take waited {elapsed:?} for the PING ACK: the round trip is back on the critical path",
        );
        // Hold the checkout so the peer stays connected for its delayed
        // ACK; dropping the pool retires the entry.
        drop(taken);
        drop(pool);
        peer.join().unwrap();
    }

    #[test]
    fn take_shared_h2_hands_out_warm_session_without_ping_ack() {
        let pool = Pool::new();
        let (session, server) = h2_session();
        pool_warm_h2(&pool, session, SOCKET_TIMEOUT);
        // Black-hole peer: it reads the PING (proving the keepalive's
        // write path ran) but never ACKs and never sends FIN — the
        // socket stays open, so the peek probe reports it alive
        // (`WouldBlock`) and the PING send succeeds. The take hands
        // the session out instead of timing out on the missing ACK:
        // silent death is detected at first response read, the same
        // discipline as a stale H1 connection.
        let (hold_tx, hold_rx) = std::sync::mpsc::channel::<()>();
        let peer = std::thread::spawn(move || {
            let mut server = server;
            let mut hdr = [0u8; 9];
            if server.read_exact(&mut hdr).is_ok() {
                assert_eq!(hdr[3], 0x6, "expected a PING frame");
                let len = u32::from_be_bytes([0, hdr[0], hdr[1], hdr[2]]) as usize;
                let mut payload = vec![0u8; len];
                let _ = server.read_exact(&mut payload);
            }
            // Hold the socket open until the test is done measuring.
            let _ = hold_rx.recv();
        });
        let start = Instant::now();
        let taken = pool.take_shared_h2(&key(), FETCH_BUDGET);
        let elapsed = start.elapsed();
        assert!(
            taken.is_some(),
            "a warm session that passes the peek probe and sends its PING must be handed out"
        );
        assert!(
            elapsed < Duration::from_secs(1),
            "black-hole take took {elapsed:?}: the checkout waited for a PING ACK that never comes",
        );
        assert_eq!(pool.shared_h2_count(&key()), 1);
        drop(hold_tx);
        drop(taken);
        drop(pool);
        peer.join().unwrap();
    }

    #[test]
    fn take_shared_h2_skips_ping_for_fresh_session() {
        let pool = Pool::new();
        let (session, mut server) = h2_session();
        pool.put_shared_h2(&key(), session, SOCKET_TIMEOUT);
        // A fresh entry takes the cheap peek probe, not a PING: no
        // bytes may arrive on the peer.
        server.set_nonblocking(true).unwrap();
        let taken = pool.take_shared_h2(&key(), FETCH_BUDGET);
        assert!(taken.is_some());
        // Hold the session so the peer stays connected while we check.
        let _held = taken;
        let mut buf = [0u8; 64];
        match server.read(&mut buf) {
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
            r => panic!("fresh take must not write to the peer, got {r:?}"),
        }
    }

    #[test]
    fn take_shared_h2_leaves_custom_rw_timeout_alone() {
        // A custom `Timeouts.rw` must survive the take path: the warm
        // take shrinks the *write* timeout transiently around the
        // fire-and-forget PING send, then restores it — after
        // `take_shared_h2` the socket timeouts are whatever the dial
        // path installed. The take must neither clobber them nor
        // depend on installing them.
        const CUSTOM_RW: Duration = Duration::from_millis(300);
        let pool = Pool::new();
        let (mut session, server) = h2_session();
        // What the dial path does in production: install the effective
        // `Timeouts.rw` on the socket before pooling.
        session
            .conn_mut()
            .set_read_timeout(Some(CUSTOM_RW))
            .unwrap();
        session
            .conn_mut()
            .set_write_timeout(Some(CUSTOM_RW))
            .unwrap();
        pool_warm_h2(&pool, session, CUSTOM_RW);
        let (hold_tx, hold_rx) = std::sync::mpsc::channel::<()>();
        let peer = std::thread::spawn(move || {
            let mut server = server;
            let mut hdr = [0u8; 9];
            server.read_exact(&mut hdr).unwrap();
            let len = u32::from_be_bytes([0, hdr[0], hdr[1], hdr[2]]) as usize;
            let mut payload = vec![0u8; len];
            server.read_exact(&mut payload).unwrap();
            server.write_all(&h2_frame(0x6, 0x1, 0, &payload)).unwrap();
            // Hold the socket OPEN without reading for the whole
            // measurement below: the client's next read must hit its
            // socket read timeout, not EOF — and the write-timeout
            // check below needs a peer that never drains. The thread
            // exits when the test releases it.
            let _ = hold_rx.recv();
        });
        let shared = pool
            .take_shared_h2(&key(), FETCH_BUDGET)
            .expect("a warm session that sends its PING must be handed out");
        // No more inbound frames: a pump must hit the socket read
        // timeout (300 ms) on its slice. The band is tight on both
        // sides: a take-time shrink left behind and a wrong 60 s
        // `SOCKET_TIMEOUT` constant both land outside it.
        let start = Instant::now();
        let mut sess = recover_lock(&shared.inner.session);
        let _ = sess
            .next_event(Instant::now() + Duration::from_secs(30))
            .expect_err("an idle socket must time out on read");
        let elapsed = start.elapsed();
        drop(sess);
        assert!(
            elapsed >= Duration::from_millis(200) && elapsed < Duration::from_secs(1),
            "read took {elapsed:?}: the custom rw timeout did not survive the take",
        );
        // The take-time PING write ran under a shrunk write timeout;
        // it must have been restored to the custom rw value, not left
        // at the PING budget. Fill the send buffer (the peer is parked
        // in its holding read and never drains) and time a 1-byte
        // write: it must fail in ~300 ms, not ~2 s
        // (PING_WRITE_TIMEOUT) and not 60 s.
        let start = Instant::now();
        let mut sess = recover_lock(&shared.inner.session);
        if let Conn::Plain(s) = sess.conn_mut() {
            fill_send_buffer(s);
        } else {
            panic!("expected a plain connection");
        }
        let werr = sess
            .conn_mut()
            .write_all(b"x")
            .expect_err("a full send buffer must fail the write");
        let welapsed = start.elapsed();
        drop(sess);
        assert!(
            matches!(
                werr.kind(),
                io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
            ),
            "unexpected write error: {werr:?}"
        );
        assert!(
            welapsed >= Duration::from_millis(200) && welapsed < Duration::from_secs(1),
            "write took {welapsed:?}: the custom write timeout was not restored after the PING send",
        );
        // Release the peer, then drop the checkout and the pool.
        drop(shared);
        drop(pool);
        drop(hold_tx);
        peer.join().unwrap();
    }

    #[test]
    fn conn_set_write_timeout_smoke() {
        // The pump slice shrinks the write timeout alongside the read
        // timeout; the API must exist on both transports.
        let (client, _server) = socketpair();
        let mut conn = Conn::Plain(client);
        conn.set_write_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        conn.set_write_timeout(None).unwrap();
    }

    #[test]
    fn reaper_returns_evicted_entries_for_post_lock_drop() {
        let pool = Pool::new();
        let (client, mut server) = socketpair();
        let reaped = {
            let mut st = recover_lock(&pool.idle);
            st.idle.entry(key()).or_default().push(IdleConn {
                conn: Conn::Plain(client),
                idle_since: Instant::now() - MAX_IDLE_AGE - Duration::from_secs(1),
            });
            st.idle_total = 1;
            st.last_reap = Instant::now() - REAP_INTERVAL - Duration::from_secs(1);
            let reaped = st.reap_stale();
            assert_eq!(reaped.len(), 1);
            assert_eq!(st.idle_total, 0);
            assert!(!st.idle.contains_key(&key()));
            // Returned to the caller; the guard is still held here, so
            // nothing may have been dropped yet.
            reaped
        };
        // The lock is released but `reaped` is alive: the socket must
        // still be open — no drop happened under the lock.
        server.set_nonblocking(true).unwrap();
        let mut one = [0u8; 1];
        assert!(
            matches!(server.read(&mut one), Err(e) if e.kind() == io::ErrorKind::WouldBlock),
            "evicted entry was dropped while the idle lock was held"
        );
        drop(reaped);
        // After the explicit drop the peer sees EOF: the close — and
        // with it any blocking close_notify write — happened outside
        // the lock, at the caller's discretion.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match server.read(&mut one) {
                Ok(0) => break,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    assert!(Instant::now() < deadline, "peer never saw EOF after drop");
                    std::thread::sleep(Duration::from_millis(10));
                }
                r => panic!("unexpected read result after drop: {r:?}"),
            }
        }
    }

    #[test]
    fn take_shared_h2_checkout_never_waits_for_ping_ack() {
        // A fetch already near its total deadline must not burn any of
        // it at checkout: the warm-age PING is fire-and-forget, so the
        // take returns without waiting for the ACK no matter how small
        // the remaining budget is.
        let pool = Pool::new();
        let (session, server) = h2_session();
        pool_warm_h2(&pool, session, SOCKET_TIMEOUT);
        // The peer ACKs the PING, but only after 500 ms — past the
        // fetch's remaining 100 ms budget.
        let peer = std::thread::spawn(move || {
            let mut server = server;
            let mut hdr = [0u8; 9];
            server.read_exact(&mut hdr).unwrap();
            let len = u32::from_be_bytes([0, hdr[0], hdr[1], hdr[2]]) as usize;
            let mut payload = vec![0u8; len];
            server.read_exact(&mut payload).unwrap();
            std::thread::sleep(Duration::from_millis(500));
            let _ = server.write_all(&h2_frame(0x6, 0x1, 0, &payload));
        });
        let start = Instant::now();
        let taken = pool.take_shared_h2(&key(), Duration::from_millis(100));
        let elapsed = start.elapsed();
        assert!(
            taken.is_some(),
            "checkout must hand out the warm session instead of waiting out the fetch deadline"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "checkout took {elapsed:?}: it waited for the PING ACK despite the fetch deadline"
        );
        drop(taken);
        drop(pool);
        peer.join().unwrap();
    }

    #[test]
    fn take_shared_h2_respects_fetch_deadline() {
        // Restored invariant: a checkout near the fetch deadline must
        // not burn the dial-time write timeout in the take-time PING
        // send. The PING write runs under
        // `min(PING_WRITE_TIMEOUT, fetch_total)` with the session lock
        // held — a wedged peer (TCP alive, never draining) retires the
        // session after the remaining budget, not after 60 s.
        let pool = Pool::new();
        let (client, server) = socketpair();
        // The dial-time write timeout, as production installs it.
        client.set_write_timeout(Some(SOCKET_TIMEOUT)).unwrap();
        // Wedged peer: accept the TCP connection, then never read —
        // the send buffer stays full, so the PING write blocks.
        fill_send_buffer(&client);
        let (hold_tx, hold_rx) = std::sync::mpsc::channel::<()>();
        let peer = std::thread::spawn(move || {
            let _server = server;
            let _ = hold_rx.recv();
        });
        pool_warm_h2(
            &pool,
            Session::new(Conn::Plain(client), 100),
            SOCKET_TIMEOUT,
        );
        let start = Instant::now();
        let taken = pool.take_shared_h2(&key(), Duration::from_millis(100));
        let elapsed = start.elapsed();
        assert!(
            taken.is_none(),
            "a wedged peer must fail the PING send and retire the session"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "take burned {elapsed:?}: the PING write was not bounded by the fetch deadline"
        );
        assert_eq!(
            pool.shared_h2_count(&key()),
            0,
            "the wedged session must be retired, never handed out again"
        );
        drop(hold_tx);
        peer.join().unwrap();
    }

    #[test]
    fn take_shared_h2_wedged_peer_does_not_stall_sixty_seconds() {
        // The Issue-2 stalled-write path: a peer that accepts TCP but
        // never drains must not stall a warm take for the dial-time
        // 60 s with the session lock held — other stream owners would
        // queue behind it past their `PUMP_SLICE` quantum. The PING
        // write is bounded by `PING_WRITE_TIMEOUT` even when the fetch
        // budget is large.
        let pool = Pool::new();
        let (client, server) = socketpair();
        client.set_write_timeout(Some(SOCKET_TIMEOUT)).unwrap();
        fill_send_buffer(&client);
        let (hold_tx, hold_rx) = std::sync::mpsc::channel::<()>();
        let peer = std::thread::spawn(move || {
            let _server = server;
            let _ = hold_rx.recv();
        });
        pool_warm_h2(
            &pool,
            Session::new(Conn::Plain(client), 100),
            SOCKET_TIMEOUT,
        );
        let start = Instant::now();
        let taken = pool.take_shared_h2(&key(), Duration::from_secs(30));
        let elapsed = start.elapsed();
        assert!(
            taken.is_none(),
            "a wedged peer must fail the PING send and retire the session"
        );
        assert!(
            elapsed >= Duration::from_secs(1),
            "take returned in {elapsed:?}: the PING write did not actually block — the peer was not wedged"
        );
        assert!(
            elapsed < PING_WRITE_TIMEOUT + Duration::from_secs(5),
            "take burned {elapsed:?}: the PING write stalled past its bound toward the dial-time 60 s"
        );
        assert_eq!(pool.shared_h2_count(&key()), 0);
        drop(hold_tx);
        peer.join().unwrap();
    }

    #[test]
    fn fetch_deadline_with_zero_progress_retires_blackholed_session() {
        // A silent-death peer: it reads the HEADERS (the write path
        // works) but never sends a byte back and never FINs. A fetch
        // that burns its whole total budget with zero bytes received
        // for any stream must retire the session — otherwise the next
        // fetch re-pays up to `fetch_total` on the same blackhole,
        // forever (`Drop`'s `touch` keeps the reaper from ever seeing
        // it as stale on a hot origin).
        let pool = Pool::new();
        let (session, server) = h2_session();
        pool.put_shared_h2(&key(), session, SOCKET_TIMEOUT);
        let shared = pool
            .take_shared_h2(&key(), FETCH_BUDGET)
            .expect("pooled session");
        // Blackhole: read the HEADERS, then park — no response, no FIN.
        let (park_tx, park_rx) = std::sync::mpsc::channel::<()>();
        let peer = std::thread::spawn(move || {
            let mut server = server;
            let (ty, _, _, _) = read_frame_from(&mut server);
            assert_eq!(ty, 0x1, "expected HEADERS");
            let _ = park_rx.recv();
        });
        let fetch_deadline = Instant::now() + Duration::from_millis(500);
        let mut stream = shared
            .open_request(&[], &[], fetch_deadline)
            .expect("stream opens on a live-looking session");
        let err = stream
            .next_event(fetch_deadline)
            .expect_err("a blackhole peer never responds");
        assert!(
            matches!(
                err,
                H2Error::Io(ref e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                    )
            ),
            "expected the fetch deadline to expire, got {err:?}"
        );
        drop(stream);
        drop(park_tx);
        peer.join().unwrap();
        assert!(
            shared.is_dead(),
            "zero-progress deadline expiry must retire the blackholed session"
        );
        assert!(
            pool.take_shared_h2(&key(), FETCH_BUDGET).is_none(),
            "the retired session must never be handed out again"
        );
        assert_eq!(pool.shared_h2_count(&key()), 0);
    }

    #[test]
    fn fetch_deadline_with_partial_progress_keeps_session() {
        // The "slow peer" half of the blackhole discipline: the server
        // answers HEADERS (without END_STREAM, so the response never
        // completes), then goes silent. The fetch still burns its whole
        // budget, but the session is provably alive — it delivered
        // bytes — so it stays pooled instead of being retired.
        let pool = Pool::new();
        let (session, server) = h2_session();
        pool.put_shared_h2(&key(), session, SOCKET_TIMEOUT);
        let shared = pool
            .take_shared_h2(&key(), FETCH_BUDGET)
            .expect("pooled session");
        let (park_tx, park_rx) = std::sync::mpsc::channel::<()>();
        let peer = std::thread::spawn(move || {
            let mut server = server;
            let (ty, _, stream, _) = read_frame_from(&mut server);
            assert_eq!(ty, 0x1, "expected HEADERS");
            // Response HEADERS with END_HEADERS but not END_STREAM:
            // the stream stays open, the response never completes.
            server
                .write_all(&h2_frame(0x1, 0x4, stream, &[0x88]))
                .unwrap();
            let _ = park_rx.recv();
        });
        let fetch_deadline = Instant::now() + Duration::from_millis(500);
        let mut stream = shared.open_request(&[], &[], fetch_deadline).expect("open");
        // The HEADERS arrive: progress.
        assert_eq!(
            pump_until_headers(&mut stream, fetch_deadline),
            stream.stream_id()
        );
        // Then silence until the budget burns.
        let err = stream
            .next_event(fetch_deadline)
            .expect_err("the response never completes");
        assert!(
            matches!(err, H2Error::Io(_)),
            "expected the fetch deadline to expire, got {err:?}"
        );
        assert!(
            !shared.is_dead(),
            "a session that delivered bytes must not be retired as a blackhole"
        );
        drop(stream);
        // The session is still pooled (`take` clones; the entry stays)
        // and the peer is still parked — a fresh take must hand it out
        // again. (Joining the peer first would FIN/RST the socket and
        // the probe would rightly refuse it.)
        assert!(
            pool.take_shared_h2(&key(), FETCH_BUDGET).is_some(),
            "a progressing session must stay pooled"
        );
        drop(park_tx);
        peer.join().unwrap();
    }

    #[test]
    fn take_shared_h2_warm_take_survives_buffered_inbound_frames() {
        // Past the warm age the take-time probe skips the peek:
        // buffered inbound frames are normal on an idle session — a
        // server SETTINGS here, or the previous take's fire-and-forget
        // PING's stale ACK when no stream was ever opened and pumped —
        // and must not read as death. The old peek-first discipline
        // dropped the session on these bytes, quietly defeating H2
        // pooling past 30 s idle whenever the server had sent anything.
        let pool = Pool::new();
        let (session, mut server) = h2_session();
        pool_warm_h2(&pool, session, SOCKET_TIMEOUT);
        // The server speaks first: an empty SETTINGS frame.
        server.write_all(&h2_frame(0x4, 0, 0, &[])).unwrap();
        let (hold_tx, hold_rx) = std::sync::mpsc::channel::<()>();
        let peer = std::thread::spawn(move || {
            let mut server = server;
            // Drain the PING the warm take sends; the SETTINGS ACK the
            // open writes is 13 bytes and fits in the kernel buffers.
            let (ty, _, _, _) = read_frame_from(&mut server);
            assert_eq!(ty, 0x6, "expected a PING frame");
            let _ = hold_rx.recv();
        });
        let shared = pool
            .take_shared_h2(&key(), FETCH_BUDGET)
            .expect("a warm session with buffered inbound frames must be handed out, not dropped");
        // The buffered SETTINGS is absorbed by the open's pre-send
        // drain: the stream opens and the session stays pooled.
        let deadline = Instant::now() + Duration::from_secs(30);
        let _stream = shared
            .open_request(&[], &[], deadline)
            .expect("open absorbs the buffered SETTINGS");
        drop(hold_tx);
        peer.join().unwrap();
        assert!(
            !shared.is_dead(),
            "the session absorbed the SETTINGS and must stay live, not retired"
        );
    }

    #[test]
    fn open_request_absorbs_buffered_goaway_on_warm_take() {
        // A GOAWAY the server sent while the session sat warm-idle is
        // invisible to the take-time probe (no round trip anymore) —
        // but the open's pre-send drain must see it before HEADERS go
        // out. The refusal is pre-send (safe to retry elsewhere), not
        // a wasted write to a gone-away peer.
        let pool = Pool::new();
        let (session, mut server) = h2_session();
        pool_warm_h2(&pool, session, SOCKET_TIMEOUT);
        // GOAWAY: last_stream_id 0, NO_ERROR.
        let mut payload = Vec::new();
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        server.write_all(&h2_frame(0x7, 0, 0, &payload)).unwrap();
        // The peer reads the take-time PING, then asserts no HEADERS
        // ever arrive.
        let peer = std::thread::spawn(move || {
            let mut server = server;
            let (ty, _, _, _) = read_frame_from(&mut server);
            assert_eq!(ty, 0x6, "expected the take-time PING");
            server
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut hdr = [0u8; 9];
            match server.read_exact(&mut hdr) {
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                    ) => {}
                other => panic!("peer saw a frame after the PING: {other:?}"),
            }
        });
        let shared = pool
            .take_shared_h2(&key(), FETCH_BUDGET)
            .expect("warm take hands the session out");
        let deadline = Instant::now() + Duration::from_secs(30);
        match shared.open_request(&[], &[], deadline) {
            Err(OpenRequestError::PreSend(H2Error::GoAway(_))) => {}
            Err(e) => panic!("a buffered GOAWAY must refuse pre-send, got PreSend({e:?})"),
            Ok(_) => panic!("a buffered GOAWAY must refuse pre-send, but the stream opened"),
        }
        assert!(shared.is_dead(), "the gone-away session must be retired");
        peer.join().unwrap();
    }

    #[test]
    fn pump_slice_bounds_dripping_peer_for_other_owners() {
        // Integration through the real `SharedStream::next_event`
        // pump: one owner's pump is stuck against a peer that starts
        // a frame and then stalls mid-header (bytes arrive, then
        // nothing — slower than the slice, so every frame read hits
        // the slice's socket timeout). A second owner opening a
        // stream must not wait past ~one slice for the session lock:
        // the pump arms `set_read_timeout(PUMP_SLICE)` and
        // `set_read_deadline(slice_end)` around every frame read and
        // yields the lock between iterations. (The deadline half is
        // what stops a *TLS* peer from stretching one record past the
        // slice — plain reads are per-read bounded; a TLS-flavored
        // variant needs a trusted-cert TLS peer, a seam `tls.rs`
        // owns — see `tls_read_deadline_stops_drip_peer` for the
        // primitive.)
        //
        // The pump's lifetime is main-thread-driven, never
        // wall-clock: A's pump runs one `next_event` with a deadline
        // far past the test, so no deadline can expire — and no
        // blackhole retire can fire — underneath B's open, however
        // long the main thread is descheduled. The pump stops when
        // the main thread holds B's stream and tells the peer to hang
        // up, which fails the pump's in-flight frame read promptly.
        // There is no wall-clock assertion on B's open latency: on a
        // loaded 2-vCPU box thread scheduling alone can add seconds,
        // so the proof is structural — B's open returns while A's
        // pump thread is provably still running mid-drip.
        let (client, server) = socketpair();
        let shared = SharedH2::new(Session::new(Conn::Plain(client), 100), SOCKET_TIMEOUT);
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let peer = std::thread::spawn(move || {
            let mut server = server;
            let (ty, _, _, _) = read_frame_from(&mut server);
            assert_eq!(ty, 0x1, "expected HEADERS");
            // Eight header bytes at 25 ms apart, then silence: the
            // frame never completes, so no protocol error can fire —
            // the pump just keeps hitting the slice bound.
            for _ in 0..8 {
                server.write_all(&[0u8]).unwrap();
                std::thread::sleep(Duration::from_millis(25));
            }
            // Hold the drip — socket open, silence — until the main
            // thread is done. Only then hang up: returning here drops
            // `server`, which fails the pump's in-flight frame read
            // with EOF instead of letting it wait out its deadline,
            // so the test stays fast. (If the test panics first,
            // dropping `stop_tx` unblocks this recv the same way, so
            // the peer never outlives the test.)
            let _ = stop_rx.recv();
        });
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut stream_a = shared
            .open_request(&[], &[], deadline)
            .expect("stream A opens");
        let shared_b = shared.clone();
        let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
        // Owner A pumps against the drip in the background. The
        // deadline is far past the test on purpose: it must never
        // expire on its own, because expiry would retire the session
        // as a blackhole — zero events ever arrive from the drip —
        // before B opens. The pump ends when the peer hangs up after
        // the main thread's stop; the error is still `Io`, and the
        // frame still never completed.
        let pump_a = std::thread::spawn(move || {
            // Signal the moment the pump starts so the main thread
            // opens B while the pump is guaranteed mid-drip. A
            // wall-clock sleep cannot do this job: under parallel-test
            // load the main thread can be descheduled for seconds at
            // a time.
            started_tx.send(()).unwrap();
            let err = stream_a
                .next_event(Instant::now() + Duration::from_secs(30))
                .expect_err("the dripped frame never completes");
            assert!(
                matches!(err, H2Error::Io(_)),
                "expected the peer hangup to surface as an Io error, got {err:?}"
            );
        });
        started_rx
            .recv_timeout(Duration::from_secs(60))
            .expect("pump thread started");
        // Settle into the drip before B asks for the lock:
        // `started_tx` fires before the pump thread's first
        // session-lock acquisition inside `next_event`, so without a
        // pause B's helper can win that race, open with zero
        // contention, and pass vacuously. Waiting ~20 PUMP_SLICE
        // quanta guarantees the pump is mid-drip and cycling the
        // lock before B opens. A fixed sleep is safe here because the
        // pump's lifetime is main-thread-driven with a 30 s deadline:
        // no deadline can expire — and no blackhole retire can fire —
        // underneath the pump no matter how long the main thread is
        // descheduled, so the sleep can never race a wall clock.
        std::thread::sleep(Duration::from_millis(200));
        // Owner B opens on a helper thread so the 5 s bound is on
        // B's open actually returning, not on the main thread's
        // scheduling: even descheduled for seconds, the main thread
        // observes the result whenever it wakes. A regression that
        // lets the pump hold the session lock past the slice fails
        // here in 5 s with a clear message instead of hanging. The
        // 2 s -> 5 s loosening is a deliberate scheduling-headroom
        // tradeoff: `open_request` blocks indefinitely on the session
        // mutex, so a stuck pump is caught by this bound at any
        // duration — but on a loaded 2-vCPU box under parallel-test
        // load, scheduling jitter between pump slices can cost
        // seconds, and the old 2 s bound flaked. The cost is that a
        // regression holding the lock ~4 s now passes where 2 s
        // would have failed; a real past-the-slice regression holds
        // the lock for the whole pump (30 s here), orders of
        // magnitude past the bound either way.
        let (b_tx, b_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let res = shared_b.open_request(&[], &[], Instant::now() + Duration::from_secs(5));
            let _ = b_tx.send(res);
        });
        let stream_b = b_rx
            .recv_timeout(Duration::from_secs(5))
            .expect(
                "B's open did not return within 5 s: the pump held the session lock past the slice",
            )
            .expect("stream B opens");
        // B completed while A's pump was provably still running
        // mid-drip: the old stuck-pump behavior (B serialized behind
        // A's whole pump) cannot satisfy this — the pump would have
        // finished first. The finished check is sampled *before* the
        // stop below (the only way the pump ends on its own is a
        // panic: its 30 s deadline never fires and the peer only
        // hangs up on our stop), and the join runs before the assert:
        // a panicked pump thread reads as finished, and joining first
        // resurfaces the real panic (e.g. `next_event` unexpectedly
        // returning `Ok`) instead of masking it behind the assert's
        // misleading message.
        let pump_done_before_stop = pump_a.is_finished();
        // Main-thread-driven shutdown: hang up the peer, which ends
        // A's pump promptly, then reap both threads.
        let _ = stop_tx.send(());
        pump_a.join().unwrap();
        peer.join().unwrap();
        assert!(
            !pump_done_before_stop,
            "B opened only after A's pump finished: the drip stretched a pump past the slice"
        );
        drop(stream_b);
    }

    #[test]
    fn concurrent_shared_take_and_open_is_safe() {
        // Hammer concurrent takes against one pooled session: slot
        // accounting must stay exact and the pool must keep exactly
        // one entry. Each take actually opens a stream, so the
        // try_acquire_slot CAS — not just the take path — runs under
        // contention.
        let pool = Arc::new(Pool::new());
        let (session, server) = h2_session();
        pool.put_shared_h2(&key(), session, SOCKET_TIMEOUT);
        // Drain-only server: it reads every HEADERS and every
        // RST_STREAM the drops emit, but answers nothing — answered
        // bytes the clients never pump would sit unread in their
        // receive buffer and RST the TCP connection on close.
        let server_thread = std::thread::spawn(move || {
            let mut server = server;
            let mut opened = 0;
            while opened < 8 * 25 {
                let (ty, _, _, _) = read_frame_from(&mut server);
                if ty == 0x1 {
                    opened += 1;
                }
            }
            opened
        });
        let mut handles = Vec::new();
        for _ in 0..8 {
            let pool = pool.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..25 {
                    // Peek probe on a live socketpair: always succeeds.
                    if let Some(shared) = pool.take_shared_h2(&key(), FETCH_BUDGET) {
                        let deadline = Instant::now() + Duration::from_secs(30);
                        let _stream = shared.open_request(&[], &[], deadline).expect(
                            "open under contention: the slot CAS must not lose reservations",
                        );
                        // Dropped here: the slot is released and the
                        // (unanswered) stream is reset.
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let opened = server_thread.join().unwrap();
        assert_eq!(opened, 8 * 25, "every contended open hit the wire");
        assert_eq!(pool.shared_h2_count(&key()), 1);
        // Slot accounting is exact: no phantom reservations.
        let st = recover_lock(&pool.idle);
        let entry = st.shared.get(&key()).unwrap().first().unwrap();
        assert_eq!(entry.inner.in_flight.load(Ordering::Acquire), 0);
    }

    /// Fill the send buffer of a client socket whose peer never reads,
    /// so the next frame write blocks until the write timeout fires.
    fn fill_send_buffer(mut client: &TcpStream) {
        client.set_nonblocking(true).unwrap();
        let chunk = [0xABu8; 65536];
        loop {
            match client.write(&chunk) {
                Ok(_) => continue,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) => panic!("unexpected fill error: {e}"),
            }
        }
        client.set_nonblocking(false).unwrap();
    }

    /// Read one raw frame off the test peer: (type, flags, stream,
    /// payload).
    fn read_frame_from(server: &mut TcpStream) -> (u8, u8, u32, Vec<u8>) {
        let mut hdr = [0u8; 9];
        server.read_exact(&mut hdr).unwrap();
        let len = u32::from_be_bytes([0, hdr[0], hdr[1], hdr[2]]) as usize;
        let mut payload = vec![0u8; len];
        server.read_exact(&mut payload).unwrap();
        (
            hdr[3],
            hdr[4],
            u32::from_be_bytes([hdr[5], hdr[6], hdr[7], hdr[8]]) & 0x7fff_ffff,
            payload,
        )
    }

    /// Like [`read_frame_from`], but bounded: a missing frame fails the
    /// test instead of hanging it.
    fn try_read_frame_from(server: &mut TcpStream) -> Option<(u8, u8, u32, Vec<u8>)> {
        server
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut hdr = [0u8; 9];
        match server.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                ) =>
            {
                return None;
            }
            Err(e) => panic!("unexpected read error: {e}"),
        }
        let len = u32::from_be_bytes([0, hdr[0], hdr[1], hdr[2]]) as usize;
        let mut payload = vec![0u8; len];
        server.read_exact(&mut payload).unwrap();
        Some((
            hdr[3],
            hdr[4],
            u32::from_be_bytes([hdr[5], hdr[6], hdr[7], hdr[8]]) & 0x7fff_ffff,
            payload,
        ))
    }

    #[test]
    fn open_request_classifies_pre_send_and_post_send() {
        // Post-send: the peer stopped reading and the send buffer is
        // full, so the HEADERS/body write times out mid-flight. Bytes
        // may have reached the peer, so the caller must not re-send.
        let (client, _server) = socketpair();
        client
            .set_write_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        fill_send_buffer(&client);
        let shared = SharedH2::new(Session::new(Conn::Plain(client), 100), SOCKET_TIMEOUT);
        let deadline = Instant::now() + Duration::from_secs(30);
        match shared.open_request(&[], b"request-body", deadline) {
            Err(OpenRequestError::PostSend(H2Error::Io(e))) => {
                // Linux surfaces a socket timeout as `WouldBlock`
                // (EAGAIN), not `TimedOut`: both mean the write bound
                // fired mid-send.
                assert!(
                    matches!(
                        e.kind(),
                        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                    ),
                    "unexpected error kind: {:?}",
                    e.kind()
                );
            }
            Err(e) => panic!("mid-send write failure must be PostSend, got {e:?}"),
            Ok(_) => panic!("mid-send write failure must not succeed"),
        }
        // The transport failed: the session is dead, never handed out
        // again.
        assert!(shared.is_dead());

        // Pre-send: a dead session refuses before any byte is written.
        let (session2, _server2) = h2_session();
        let shared2 = SharedH2::new(session2, SOCKET_TIMEOUT);
        shared2.mark_dead();
        match shared2.open_request(&[], &[], deadline) {
            Err(OpenRequestError::PreSend(H2Error::GoAway(_))) => {}
            Err(e) => panic!("dead session must refuse pre-send, got {e:?}"),
            Ok(_) => panic!("dead session must not open"),
        }

        // Pre-send: stream-slot exhaustion refuses before any byte is
        // written.
        let (session3, _server3) = h2_session();
        let shared3 = SharedH2::new(session3, SOCKET_TIMEOUT);
        for _ in 0..LOCAL_H2_MAX_STREAMS {
            assert!(shared3.try_acquire_slot());
        }
        match shared3.open_request(&[], &[], deadline) {
            Err(OpenRequestError::PreSend(H2Error::ConcurrentStreamsExceeded)) => {}
            Err(e) => panic!("saturated session must refuse pre-send, got {e:?}"),
            Ok(_) => panic!("saturated session must not open"),
        }

        // Pre-send: an already-expired deadline fails before any byte
        // is written.
        let (session4, _server4) = h2_session();
        let shared4 = SharedH2::new(session4, SOCKET_TIMEOUT);
        match shared4.open_request(&[], &[], Instant::now() - Duration::from_secs(1)) {
            Err(OpenRequestError::PreSend(_)) => {}
            Err(e) => panic!("expired deadline must fail pre-send, got {e:?}"),
            Ok(_) => panic!("expired deadline must not open"),
        }
    }

    #[test]
    fn take_and_put_retire_dead_shared_sessions_after_unlock() {
        // Dead entries are collected as `Evicted` and retired after
        // the idle guard is released (their drop may block on a TLS
        // close_notify write); the census must stay exact either way.
        let pool = Pool::new();
        let (session, _server) = h2_session();
        pool.put_shared_h2(&key(), session, SOCKET_TIMEOUT);
        {
            let st = recover_lock(&pool.idle);
            st.shared.get(&key()).unwrap().first().unwrap().mark_dead();
        }
        // The take-time dead-scan reaps the entry instead of handing
        // it out.
        assert!(pool.take_shared_h2(&key(), FETCH_BUDGET).is_none());
        assert_eq!(pool.shared_h2_count(&key()), 0);
        assert_eq!(recover_lock(&pool.idle).idle_total, 0);

        // A dead entry purged by a later put: the pool keeps only the
        // live session and the census counts exactly it.
        let (s1, _srv1) = h2_session();
        let (s2, _srv2) = h2_session();
        pool.put_shared_h2(&key(), s1, SOCKET_TIMEOUT);
        {
            let st = recover_lock(&pool.idle);
            st.shared.get(&key()).unwrap().first().unwrap().mark_dead();
        }
        pool.put_shared_h2(&key(), s2, SOCKET_TIMEOUT);
        assert_eq!(pool.shared_h2_count(&key()), 1);
        assert_eq!(recover_lock(&pool.idle).idle_total, 1);
        assert!(
            pool.take_shared_h2(&key(), FETCH_BUDGET).is_some(),
            "the live session must survive the purge"
        );
    }

    #[test]
    fn pump_write_timeout_is_bounded_by_the_slice() {
        // The peer stops reading: the PING ACK the pump writes
        // synchronously would block up to the dial-time write timeout.
        // The pump must bound the write by the slice instead, so the
        // session lock is never held past it.
        let (client, mut server) = socketpair();
        // Simulate the dial-time write timeout the pump must shrink:
        // long enough that this test fails without the fix.
        client
            .set_write_timeout(Some(Duration::from_secs(15)))
            .unwrap();
        let shared = SharedH2::new(
            Session::new(Conn::Plain(client), 100),
            Duration::from_secs(15),
        );
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut stream = shared.open_request(&[], &[], deadline).expect("open");
        // One inbound PING: the next pump answers it with a synchronous
        // ACK write.
        server.write_all(&h2_frame(0x6, 0, 0, b"12345678")).unwrap();
        // The peer never reads again: the send buffer stays full, so
        // the ACK write blocks.
        {
            let mut sess = recover_lock(&shared.inner.session);
            if let Conn::Plain(s) = sess.conn_mut() {
                fill_send_buffer(s);
            } else {
                panic!("expected a plain connection");
            }
        }
        let fetch_deadline = Instant::now() + Duration::from_secs(2);
        let start = Instant::now();
        let err = stream
            .next_event(fetch_deadline)
            .expect_err("a peer that never reads must time the fetch out");
        let elapsed = start.elapsed();
        // Linux surfaces a socket timeout as `WouldBlock` (EAGAIN),
        // not `TimedOut`: both mean a slice bound fired.
        assert!(
            matches!(
                err,
                H2Error::Io(ref e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                    )
            ),
            "expected a slice/deadline timeout, got {err:?}"
        );
        assert!(
            elapsed < Duration::from_secs(10),
            "pump held the session {elapsed:?}: the write timeout was not bounded by the slice"
        );
    }

    #[test]
    fn dropped_incomplete_stream_sends_rst_and_frees_session_budget() {
        let (session, mut server) = h2_session();
        let shared = SharedH2::new(session, SOCKET_TIMEOUT);
        let deadline = Instant::now() + Duration::from_secs(30);
        let stream = shared.open_request(&[], &[], deadline).expect("open");
        let id = stream.stream_id();
        // Dropped mid-response: no event was ever pumped.
        drop(stream);
        // The server saw HEADERS, then the RST_STREAM(CANCEL). The
        // reads are bounded: a regression that stops resetting
        // abandoned streams fails here instead of hanging.
        let head = try_read_frame_from(&mut server)
            .expect("server must see the HEADERS")
            .0;
        assert_eq!(head, 0x1, "first frame must be HEADERS");
        let rst = try_read_frame_from(&mut server).expect("server must see the RST_STREAM");
        assert_eq!(rst.0, 0x3, "an abandoned stream must be reset");
        assert_eq!(rst.2, id, "the reset must name the abandoned stream");
        assert_eq!(
            rst.3,
            0x8u32.to_be_bytes().to_vec(),
            "the reset code must be CANCEL"
        );
        // The session no longer counts the stream against the peer's
        // concurrency backstop, and the slot is released.
        assert!(
            recover_lock(&shared.inner.session).is_stream_closed(id),
            "abandoned stream state must be Closed"
        );
        assert_eq!(shared.inner.in_flight.load(Ordering::Acquire), 0);
    }

    #[test]
    fn dropped_completed_stream_sends_no_rst() {
        let (session, mut server) = h2_session();
        let shared = SharedH2::new(session, SOCKET_TIMEOUT);
        // Answer HEADERS with `:status: 200`, END_STREAM, then expect
        // nothing more: a completed stream must not be reset on drop.
        let server_thread = std::thread::spawn(move || {
            let (ty, _, stream, _) = read_frame_from(&mut server);
            assert_eq!(ty, 0x1, "expected HEADERS");
            server
                .write_all(&h2_frame(0x1, 0x5, stream, &[0x88]))
                .unwrap();
            server
                .set_read_timeout(Some(Duration::from_millis(300)))
                .unwrap();
            let mut hdr = [0u8; 9];
            match server.read_exact(&mut hdr) {
                // No RST arrived: on Linux a socket read timeout
                // surfaces as `WouldBlock` (EAGAIN), not `TimedOut`.
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                    ) => {}
                other => panic!("expected no further frame, got {other:?}"),
            }
        });
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut stream = shared.open_request(&[], &[], deadline).expect("open");
        assert_eq!(
            pump_until_headers(&mut stream, deadline),
            stream.stream_id()
        );
        drop(stream);
        server_thread.join().unwrap();
    }

    /// Poison a fetch thread's lock with a real panic: a thread holds
    /// a lock across a panicking critical section, poisoning the
    /// mutex.
    fn poison_idle_lock(pool: &Pool) {
        let _guard = pool.idle.lock().unwrap();
        panic!("simulated fetch-thread panic while holding the idle lock");
    }

    #[test]
    fn pool_recovers_from_poisoned_idle_lock() {
        // The regression this guards: the pool is process-wide, so a
        // `.lock().unwrap()` on a poisoned idle mutex panicked every
        // later fetch forever, permanently killing networking. The
        // pool must keep serving checkouts and returns through
        // recovery instead.
        let pool = Arc::new(Pool::new());
        let (client, _server) = socketpair();
        pool.put(&key(), Conn::Plain(client));
        let p = pool.clone();
        let h = std::thread::spawn(move || poison_idle_lock(&p));
        assert!(h.join().is_err(), "the poisoner must have panicked");
        // The pooled connection is still checked out (the probe runs
        // outside the lock and the peer is alive).
        let conn = pool.take(&key()).expect("pool must recover from poisoning");
        assert_eq!(pool.idle_count(&key()), 0);
        // Returns work too: no later operation panics on the poisoned
        // mutex.
        pool.put(&key(), conn);
        assert_eq!(pool.idle_count(&key()), 1);
        assert!(pool.take(&key()).is_some());
        assert_eq!(pool.idle_count(&key()), 0);
    }

    #[test]
    fn pool_recovers_from_poisoned_ticket_lock() {
        let pool = Arc::new(Pool::new());
        pool.store_tickets(&key(), vec![CachedTicket::test_ticket(3600)]);
        let p = pool.clone();
        let h = std::thread::spawn(move || {
            let _guard = p.tickets.lock().unwrap();
            panic!("simulated fetch-thread panic while holding the ticket lock");
        });
        assert!(h.join().is_err(), "the poisoner must have panicked");
        // The ticket cache still serves offers through recovery.
        let offer = pool
            .ticket(&key())
            .expect("ticket cache must recover from poisoning");
        assert_eq!(offer.identity, b"tick");
        assert!(pool.ticket(&key()).is_none(), "single-use still holds");
        // Storing into the recovered lock works too.
        pool.store_tickets(&key(), vec![CachedTicket::test_ticket(3600)]);
        assert_eq!(pool.ticket_count(&key()), 1);
    }

    #[test]
    fn shared_h2_poisoned_session_lock_retires_session() {
        // A stream owner's pump thread panics mid-pump while holding
        // the session lock. The take-time probe cannot detect the
        // logical corruption a mid-pump panic may leave behind
        // (half-fed HPACK decoder, desynced stream map), so the
        // session is retired instead of handed out: a poisoned lock
        // must never panic later owners, and must never hand them a
        // corrupt session either.
        let pool = Pool::new();
        let (session, _server) = h2_session();
        pool.put_shared_h2(&key(), session, SOCKET_TIMEOUT);
        let shared = pool
            .take_shared_h2(&key(), FETCH_BUDGET)
            .expect("pooled session");
        let victim = shared.clone();
        let h = std::thread::spawn(move || {
            let _guard = victim.inner.session.lock().unwrap();
            panic!("simulated pump-thread panic while holding the session lock");
        });
        assert!(h.join().is_err(), "the poisoner must have panicked");
        // The take path retires the session sight-unseen instead of
        // handing it out.
        assert!(
            pool.take_shared_h2(&key(), FETCH_BUDGET).is_none(),
            "poisoned session must be discarded, not handed out"
        );
        assert!(shared.is_dead(), "poisoned session must be marked dead");
        assert_eq!(pool.shared_h2_count(&key()), 0);
    }

    #[test]
    fn now_ms_is_monotonic_and_matches_wall_scale() {
        // The ticket clock must never go backwards: a wall-clock step
        // (NTP correction, manual change) must not resurrect expired
        // tickets, so the origin is captured once and advanced with a
        // monotonic `Instant`.
        let a = now_ms();
        std::thread::sleep(Duration::from_millis(10));
        let b = now_ms();
        assert!(
            b - a >= 10,
            "ticket clock must advance with monotonic time: {a} -> {b}"
        );
        // ...while staying on the unix-millis scale `tls.rs` stamps
        // `received_at_ms` with, so cross-crate expiry comparisons stay
        // valid.
        let wall = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        assert!(
            b.abs_diff(wall) < 5_000,
            "ticket clock {b} drifted off the wall scale {wall}"
        );
    }

    #[test]
    fn ticket_expires_by_monotonic_age() {
        // A short-lived ticket is offered while fresh and expires by
        // the monotonic clock even with no wall-clock movement.
        let pool = Pool::new();
        pool.store_tickets(&key(), vec![CachedTicket::test_ticket(1)]);
        assert!(
            pool.ticket(&key()).is_some(),
            "a fresh 1s ticket must be offered"
        );
        pool.store_tickets(&key(), vec![CachedTicket::test_ticket(1)]);
        std::thread::sleep(Duration::from_millis(1100));
        assert!(
            pool.ticket(&key()).is_none(),
            "the 1s ticket must expire by monotonic age"
        );
        assert_eq!(pool.ticket_count(&key()), 0);
    }

    #[test]
    fn forward_wall_spike_does_not_pin_ticket_clock() {
        // `clock_now_ms` is the pure combination step of `now_ms()`:
        // drive a +8h forward wall spike and its correction directly.
        let mono = 1_700_000_000_000u64;
        // While the spike lasts it is honored — tickets only expire
        // sooner, the safe direction.
        assert_eq!(clock_now_ms(mono, mono + 28_800_000), mono + 28_800_000);
        // The moment the wall corrects back, the clock drops to the
        // monotonic base instead of staying pinned at the spike (which
        // would read every ticket expired for hours and silently
        // disable resumption).
        assert_eq!(clock_now_ms(mono, mono), mono);
        // A backward wall step can only pull the clock down to the
        // monotonic base, never below it.
        assert_eq!(clock_now_ms(mono, mono - 3_600_000), mono);
    }

    #[test]
    fn reap_stale_heals_drifted_idle_census() {
        // A mid-`put` panic between the list push and the counter bump
        // would leave `idle_total` permanently skewed; the reap
        // recomputes the census from the actual list lengths, so one
        // historical drift heals instead of compounding.
        let pool = Pool::new();
        let (c1, _s1) = socketpair();
        pool.put(&key(), Conn::Plain(c1));
        let (c2, _s2) = socketpair();
        pool.put(&key(), Conn::Plain(c2));
        {
            let mut st = recover_lock(&pool.idle);
            // Over-counted, as a duplicated bump would leave it.
            st.idle_total = 100;
            st.last_reap = Instant::now() - REAP_INTERVAL - Duration::from_secs(1);
            let reaped = st.reap_stale();
            // Entries are fresh: nothing evicted, census recomputed.
            assert!(reaped.is_empty());
            drop(reaped);
            let actual: usize = st.idle.values().map(Vec::len).sum::<usize>()
                + st.shared.values().map(Vec::len).sum::<usize>();
            assert_eq!(actual, 2);
            assert_eq!(st.idle_total, actual, "census must match the actual lists");
            // Under-counted, as a missed bump would leave it.
            st.idle_total = 0;
            st.last_reap = Instant::now() - REAP_INTERVAL - Duration::from_secs(1);
            let reaped = st.reap_stale();
            assert!(reaped.is_empty());
            drop(reaped);
            assert_eq!(st.idle_total, 2, "census must heal upward too");
        }
        // And the counter stays consistent on the normal path.
        assert!(pool.take(&key()).is_some());
        assert_eq!(recover_lock(&pool.idle).idle_total, 1);
    }
}
