//! HTTP/2 framing and HPACK header compression.
//!
//! Implements the wire protocol from RFC 9113 (HTTP/2 framing, flow control,
//! connection preface, SETTINGS) and RFC 7541 (HPACK: integer representation,
//! Huffman coding, static/dynamic tables, encoder and decoder).
//!
//! This is the framing layer only: it encodes and decodes frames and header
//! blocks, but does not own any I/O. A connection driver feeds received bytes
//! into [`frame::Frame::decode`] / [`hpack::Decoder`] and writes produced
//! bytes to the transport.
//!
//! All code is `std`-only.

pub mod flow;
pub mod frame;
pub mod hpack;
pub mod preface;
pub mod settings;

pub use flow::{ConnectionFlow, FlowError, Window};
pub use frame::{ErrorCode, Frame, FrameError, FrameType, Header, Payload, PrioritySpec};
pub use hpack::{Decoder as HpackDecoder, Encoder as HpackEncoder, HeaderField, HpackError};
pub use preface::PrefaceError;
pub use settings::{Setting, Settings, SettingsDelta, SettingsError, SettingsProfile};

/// Any error produced by the [`h2`](self) module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Frame(FrameError),
    Flow(FlowError),
    Settings(SettingsError),
    Hpack(HpackError),
    Preface(PrefaceError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Frame(e) => write!(f, "h2 frame error: {e:?}"),
            Error::Flow(e) => write!(f, "h2 flow-control error: {e:?}"),
            Error::Settings(e) => write!(f, "h2 settings error: {e:?}"),
            Error::Hpack(e) => write!(f, "h2 hpack error: {e:?}"),
            Error::Preface(e) => write!(f, "h2 preface error: {e:?}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<FrameError> for Error {
    fn from(e: FrameError) -> Self {
        Error::Frame(e)
    }
}

impl From<FlowError> for Error {
    fn from(e: FlowError) -> Self {
        Error::Flow(e)
    }
}

impl From<SettingsError> for Error {
    fn from(e: SettingsError) -> Self {
        Error::Settings(e)
    }
}

impl From<HpackError> for Error {
    fn from(e: HpackError) -> Self {
        Error::Hpack(e)
    }
}

impl From<PrefaceError> for Error {
    fn from(e: PrefaceError) -> Self {
        Error::Preface(e)
    }
}
