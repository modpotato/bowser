//! Flow control windows (RFC 9113, Section 6.9).
//!
//! Both endpoints track a connection-level window and one window per stream,
//! in each direction. Sending DATA requires room in both the stream and the
//! connection send windows; receiving DATA consumes the receive windows.
//!
//! A sender's window can go negative when the peer lowers
//! SETTINGS_INITIAL_WINDOW_SIZE; sending is blocked until WINDOW_UPDATEs
//! bring it back above zero.

use std::collections::HashMap;

/// Initial flow-control window size for new streams and connections.
pub const DEFAULT_WINDOW_SIZE: u32 = 65_535;
/// Largest legal window size: 2^31 - 1.
pub const MAX_WINDOW_SIZE: u32 = 0x7fff_ffff;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowError {
    /// A WINDOW_UPDATE would push the window past 2^31 - 1.
    Overflow,
    /// A WINDOW_UPDATE carried a zero increment (RFC 9113, Section 6.9.1:
    /// the connection layer maps this to a PROTOCOL_ERROR).
    ZeroIncrement,
    /// Not enough window to send (or the peer sent) `want` bytes.
    ///
    /// On the receive path the connection layer maps this to a
    /// FLOW_CONTROL_ERROR connection error.
    InsufficientWindow { want: u32, have: u32 },
}

/// One direction of one flow-control window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    /// May be negative after SETTINGS_INITIAL_WINDOW_SIZE shrinks.
    size: i64,
}

impl Window {
    pub fn new(size: u32) -> Self {
        debug_assert!(size <= MAX_WINDOW_SIZE);
        Window { size: size as i64 }
    }

    /// Current window size; negative means "in debt".
    pub fn size(&self) -> i64 {
        self.size
    }

    /// Reserve `n` bytes for sending DATA.
    pub fn send(&mut self, n: u32) -> Result<(), FlowError> {
        if self.size < n as i64 {
            return Err(FlowError::InsufficientWindow {
                want: n,
                have: self.size.max(0) as u32,
            });
        }
        self.size -= n as i64;
        Ok(())
    }

    /// Account for `n` bytes of DATA received from the peer.
    pub fn received(&mut self, n: u32) -> Result<(), FlowError> {
        if self.size < n as i64 {
            return Err(FlowError::InsufficientWindow {
                want: n,
                have: self.size.max(0) as u32,
            });
        }
        self.size -= n as i64;
        Ok(())
    }

    /// Apply a WINDOW_UPDATE increment (1..=2^31-1). A zero increment is a
    /// protocol error (RFC 9113, Section 6.9.1).
    pub fn update(&mut self, increment: u32) -> Result<(), FlowError> {
        if increment == 0 {
            return Err(FlowError::ZeroIncrement);
        }
        let next = self.size + increment as i64;
        if next > MAX_WINDOW_SIZE as i64 {
            return Err(FlowError::Overflow);
        }
        self.size = next;
        Ok(())
    }

    /// Restore `n` bytes previously reserved by [`Window::send`] or
    /// [`Window::received`]. Infallible: it returns the window to the exact
    /// state it was in before, so no bounds check applies. Used to roll back
    /// a half-applied multi-window accounting step.
    fn unreserve(&mut self, n: u32) {
        self.size += n as i64;
    }

    /// Adjust for a SETTINGS_INITIAL_WINDOW_SIZE change. May go negative.
    pub fn rebase(&mut self, old: u32, new: u32) {
        self.size += new as i64 - old as i64;
    }

    /// The application consumed `n` received bytes. Grows the window and
    /// returns the increment to advertise in a WINDOW_UPDATE. Consuming
    /// zero bytes is a no-op.
    pub fn consumed(&mut self, n: u32) -> Result<u32, FlowError> {
        if n == 0 {
            return Ok(0);
        }
        if n > MAX_WINDOW_SIZE {
            return Err(FlowError::Overflow);
        }
        self.update(n)?;
        Ok(n)
    }
}

impl Default for Window {
    fn default() -> Self {
        Window::new(DEFAULT_WINDOW_SIZE)
    }
}

/// Connection-level plus per-stream flow-control state.
#[derive(Debug)]
pub struct ConnectionFlow {
    send: Window,
    recv: Window,
    streams: HashMap<u32, (Window, Window)>,
    initial_window: u32,
}

impl Default for ConnectionFlow {
    fn default() -> Self {
        ConnectionFlow::new()
    }
}

impl ConnectionFlow {
    pub fn new() -> Self {
        ConnectionFlow {
            send: Window::default(),
            recv: Window::default(),
            streams: HashMap::new(),
            initial_window: DEFAULT_WINDOW_SIZE,
        }
    }

    /// Register a new stream with fresh windows. Idempotent.
    pub fn open_stream(&mut self, stream_id: u32) {
        self.streams.entry(stream_id).or_insert_with(|| {
            (
                Window::new(self.initial_window),
                Window::new(DEFAULT_WINDOW_SIZE),
            )
        });
    }

    /// Drop a stream's windows (stream closed/reset).
    pub fn close_stream(&mut self, stream_id: u32) {
        self.streams.remove(&stream_id);
    }

    fn stream_windows(&mut self, stream_id: u32) -> Result<(&mut Window, &mut Window), FlowError> {
        // Missing stream: treat as a closed stream with zero windows.
        if !self.streams.contains_key(&stream_id) {
            return Err(FlowError::InsufficientWindow { want: 1, have: 0 });
        }
        let (s, r) = self.streams.get_mut(&stream_id).unwrap();
        Ok((s, r))
    }

    /// Account for sending `n` DATA bytes on `stream_id`.
    ///
    /// Checks the stream send window first, then the connection send window.
    /// Transactional: if the connection window fails, the stream window is
    /// restored.
    pub fn send_data(&mut self, stream_id: u32, n: u32) -> Result<(), FlowError> {
        let (s, _) = self.stream_windows(stream_id)?;
        s.send(n)?;
        // Connection window second; roll back the stream window on failure.
        if let Err(e) = self.send.send(n) {
            let (s, _) = self.stream_windows(stream_id).unwrap();
            s.unreserve(n);
            return Err(e);
        }
        Ok(())
    }

    /// Account for receiving `n` DATA bytes on `stream_id`.
    ///
    /// Transactional: if the connection window fails, the stream window is
    /// restored.
    pub fn recv_data(&mut self, stream_id: u32, n: u32) -> Result<(), FlowError> {
        let (_, r) = self.stream_windows(stream_id)?;
        r.received(n)?;
        if let Err(e) = self.recv.received(n) {
            // Peer violated connection-level flow control; undo the stream
            // accounting so the state is unchanged.
            let (_, r) = self.stream_windows(stream_id).unwrap();
            r.unreserve(n);
            return Err(e);
        }
        Ok(())
    }

    /// Apply a WINDOW_UPDATE. `stream_id == 0` targets the connection window.
    ///
    /// A zero increment is rejected (RFC 9113, Section 6.9.1).
    pub fn window_update(&mut self, stream_id: u32, increment: u32) -> Result<(), FlowError> {
        if increment == 0 {
            return Err(FlowError::ZeroIncrement);
        }
        if stream_id == 0 {
            return self.send.update(increment);
        }
        let (s, _) = self.stream_windows(stream_id)?;
        s.update(increment)
    }

    /// The application consumed `n` received bytes on `stream_id`.
    ///
    /// Returns the increment to advertise back to the peer via WINDOW_UPDATE
    /// frames (one for the stream, one for the connection).
    ///
    /// Transactional: if growing the connection window fails, the stream
    /// window is restored.
    pub fn consumed(&mut self, stream_id: u32, n: u32) -> Result<u32, FlowError> {
        let (_, r) = self.stream_windows(stream_id)?;
        r.consumed(n)?;
        if let Err(e) = self.recv.consumed(n) {
            let (_, r) = self.stream_windows(stream_id).unwrap();
            r.unreserve(n);
            return Err(e);
        }
        Ok(n)
    }

    /// Apply a new SETTINGS_INITIAL_WINDOW_SIZE to all open streams'
    /// send windows (RFC 9113, Section 6.9.2).
    pub fn set_initial_window(&mut self, new: u32) {
        let old = self.initial_window;
        self.initial_window = new;
        for (s, _) in self.streams.values_mut() {
            s.rebase(old, new);
        }
    }

    /// Current send window for a stream (`None` if not tracked).
    pub fn send_window(&self, stream_id: u32) -> Option<i64> {
        self.streams.get(&stream_id).map(|(s, _)| s.size())
    }

    /// Current receive window for a stream (`None` if not tracked).
    pub fn recv_window(&self, stream_id: u32) -> Option<i64> {
        self.streams.get(&stream_id).map(|(_, r)| r.size())
    }

    pub fn connection_send_window(&self) -> i64 {
        self.send.size()
    }

    pub fn connection_recv_window(&self) -> i64 {
        self.recv.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_windows_are_default() {
        let mut fc = ConnectionFlow::new();
        assert_eq!(fc.connection_send_window(), 65_535);
        assert_eq!(fc.connection_recv_window(), 65_535);
        fc.open_stream(1);
        assert_eq!(fc.send_window(1), Some(65_535));
        assert_eq!(fc.recv_window(1), Some(65_535));
    }

    #[test]
    fn send_data_consumes_stream_and_connection_windows() {
        let mut fc = ConnectionFlow::new();
        fc.open_stream(1);
        fc.send_data(1, 1000).unwrap();
        assert_eq!(fc.send_window(1), Some(64_535));
        assert_eq!(fc.connection_send_window(), 64_535);
        // Untouched stream keeps its window.
        fc.open_stream(3);
        assert_eq!(fc.send_window(3), Some(65_535));
    }

    #[test]
    fn send_blocked_when_stream_window_empty() {
        let mut fc = ConnectionFlow::new();
        fc.open_stream(1);
        fc.send_data(1, 65_535).unwrap();
        assert_eq!(
            fc.send_data(1, 1),
            Err(FlowError::InsufficientWindow { want: 1, have: 0 })
        );
        // Connection window untouched by the failed send... it was consumed
        // by the successful one.
        assert_eq!(fc.connection_send_window(), 0);
    }

    #[test]
    fn failed_connection_send_rolls_back_stream_window() {
        let mut fc = ConnectionFlow::new();
        fc.open_stream(1);
        // Drain the connection window via another stream.
        fc.open_stream(3);
        fc.send_data(3, 65_535).unwrap();
        assert_eq!(
            fc.send_data(1, 1),
            Err(FlowError::InsufficientWindow { want: 1, have: 0 })
        );
        // Stream 1's window rolled back.
        assert_eq!(fc.send_window(1), Some(65_535));
    }

    #[test]
    fn default_matches_new() {
        let fc = ConnectionFlow::default();
        assert_eq!(fc.connection_send_window(), 65_535);
        assert_eq!(fc.connection_recv_window(), 65_535);
    }

    #[test]
    fn zero_window_update_rejected() {
        let mut fc = ConnectionFlow::new();
        fc.open_stream(1);
        assert_eq!(fc.window_update(1, 0), Err(FlowError::ZeroIncrement));
        assert_eq!(fc.window_update(0, 0), Err(FlowError::ZeroIncrement));
        assert_eq!(fc.send_window(1), Some(65_535));

        let mut w = Window::new(10);
        assert_eq!(w.update(0), Err(FlowError::ZeroIncrement));
        assert_eq!(w.size(), 10);
        // Consuming zero bytes is a no-op, not an error.
        assert_eq!(w.consumed(0), Ok(0));
        assert_eq!(fc.consumed(1, 0), Ok(0));
    }

    #[test]
    fn failed_connection_recv_rolls_back_stream_window() {
        let mut fc = ConnectionFlow::new();
        fc.open_stream(1);
        fc.open_stream(3);
        // Drain the connection receive window via another stream.
        fc.recv_data(3, 65_535).unwrap();
        assert_eq!(
            fc.recv_data(1, 1),
            Err(FlowError::InsufficientWindow { want: 1, have: 0 })
        );
        // Stream 1's window rolled back.
        assert_eq!(fc.recv_window(1), Some(65_535));
        assert_eq!(fc.connection_recv_window(), 0);
    }

    #[test]
    fn window_update_grows_window() {
        let mut fc = ConnectionFlow::new();
        fc.open_stream(1);
        fc.send_data(1, 65_535).unwrap();
        fc.window_update(1, 100).unwrap();
        assert_eq!(fc.send_window(1), Some(100));
        fc.window_update(0, 50).unwrap();
        assert_eq!(fc.connection_send_window(), 50);
    }

    #[test]
    fn window_update_overflow_rejected() {
        let mut fc = ConnectionFlow::new();
        fc.open_stream(1);
        assert_eq!(
            fc.window_update(1, MAX_WINDOW_SIZE),
            Err(FlowError::Overflow)
        );
        // 65535 + (2^31-1) > 2^31-1.
        assert_eq!(fc.send_window(1), Some(65_535));
    }

    #[test]
    fn shrinking_initial_window_can_go_negative() {
        let mut fc = ConnectionFlow::new();
        fc.open_stream(1);
        fc.send_data(1, 100).unwrap(); // window now 65435
        fc.set_initial_window(16);
        assert_eq!(fc.send_window(1), Some(65435 - 65535 + 16));
        assert_eq!(fc.send_window(1), Some(-84));
        // Sending blocked while negative.
        assert!(fc.send_data(1, 1).is_err());
        // WINDOW_UPDATEs recover it.
        fc.window_update(1, 100).unwrap();
        assert_eq!(fc.send_window(1), Some(16));
        fc.send_data(1, 16).unwrap();
    }

    #[test]
    fn growing_initial_window_applies_to_open_streams() {
        let mut fc = ConnectionFlow::new();
        fc.open_stream(1);
        fc.set_initial_window(131_070);
        assert_eq!(fc.send_window(1), Some(131_070));
        // Streams opened later get the new initial size.
        fc.open_stream(3);
        assert_eq!(fc.send_window(3), Some(131_070));
    }

    #[test]
    fn recv_data_consumes_windows() {
        let mut fc = ConnectionFlow::new();
        fc.open_stream(1);
        fc.recv_data(1, 1000).unwrap();
        assert_eq!(fc.recv_window(1), Some(64_535));
        assert_eq!(fc.connection_recv_window(), 64_535);
    }

    #[test]
    fn recv_beyond_window_rejected() {
        let mut fc = ConnectionFlow::new();
        fc.open_stream(1);
        assert_eq!(
            fc.recv_data(1, 65_536),
            Err(FlowError::InsufficientWindow {
                want: 65_536,
                have: 65_535
            })
        );
    }

    #[test]
    fn consumed_returns_increment_and_grows_windows() {
        let mut fc = ConnectionFlow::new();
        fc.open_stream(1);
        fc.recv_data(1, 1000).unwrap();
        let inc = fc.consumed(1, 1000).unwrap();
        assert_eq!(inc, 1000);
        assert_eq!(fc.recv_window(1), Some(65_535));
        assert_eq!(fc.connection_recv_window(), 65_535);
    }

    #[test]
    fn consumed_huge_increment_rejected() {
        let mut fc = ConnectionFlow::new();
        fc.open_stream(1);
        assert_eq!(fc.consumed(1, u32::MAX), Err(FlowError::Overflow));
    }

    #[test]
    fn close_stream_drops_windows() {
        let mut fc = ConnectionFlow::new();
        fc.open_stream(1);
        fc.close_stream(1);
        assert_eq!(fc.send_window(1), None);
        assert!(fc.send_data(1, 1).is_err());
    }

    #[test]
    fn window_unit_send_receive_update() {
        let mut w = Window::new(10);
        w.send(10).unwrap();
        assert_eq!(w.size(), 0);
        assert!(w.send(1).is_err());
        w.update(5).unwrap();
        assert_eq!(w.size(), 5);
        assert_eq!(w.update(MAX_WINDOW_SIZE), Err(FlowError::Overflow));
    }
}
