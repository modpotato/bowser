//! Connection preface (RFC 9113, Section 3.4).
//!
//! A client starts with the 24-octet magic string followed by a SETTINGS
//! frame; a server starts with its SETTINGS frame.

use crate::h2::frame::{Frame, Payload};
use crate::h2::settings::Setting;

/// The client connection preface magic: `PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n`.
pub const CLIENT_MAGIC: &[u8; 24] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefaceError {
    /// Fewer than 24 bytes available.
    Truncated,
    /// Magic bytes do not match.
    BadMagic,
}

/// Write the client preface: magic + SETTINGS frame (stream 0).
pub fn write_client_preface(out: &mut Vec<u8>, settings: &[Setting]) {
    out.extend_from_slice(CLIENT_MAGIC);
    write_settings(out, settings);
}

/// Write the server preface: a SETTINGS frame (stream 0), no magic.
pub fn write_server_preface(out: &mut Vec<u8>, settings: &[Setting]) {
    write_settings(out, settings);
}

fn write_settings(out: &mut Vec<u8>, settings: &[Setting]) {
    Frame {
        stream_id: 0,
        payload: Payload::Settings {
            ack: false,
            params: settings.to_vec(),
        },
    }
    .encode(out);
}

/// Validate the client's 24 magic bytes (server side).
pub fn check_client_magic(buf: &[u8]) -> Result<(), PrefaceError> {
    let magic: &[u8; 24] = buf
        .get(..24)
        .and_then(|s| s.try_into().ok())
        .ok_or(PrefaceError::Truncated)?;
    if magic != CLIENT_MAGIC {
        return Err(PrefaceError::BadMagic);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h2::frame::FrameType;
    use crate::h2::settings::{HEADER_TABLE_SIZE, INITIAL_WINDOW_SIZE, MAX_FRAME_SIZE};

    #[test]
    fn client_preface_layout() {
        let settings = vec![
            Setting {
                id: HEADER_TABLE_SIZE,
                value: 4096,
            },
            Setting {
                id: INITIAL_WINDOW_SIZE,
                value: 65_535,
            },
        ];
        let mut buf = Vec::new();
        write_client_preface(&mut buf, &settings);
        assert_eq!(&buf[..24], CLIENT_MAGIC);
        check_client_magic(&buf).unwrap();

        let (frame, consumed) = Frame::decode(&buf[24..]).unwrap();
        assert_eq!(consumed, buf.len() - 24);
        assert_eq!(frame.ty(), Some(FrameType::Settings));
        assert_eq!(frame.stream_id, 0);
        match frame.payload {
            Payload::Settings { ack, params } => {
                assert!(!ack);
                assert_eq!(params.len(), 2);
                assert_eq!(params[0].id, HEADER_TABLE_SIZE);
            }
            _ => panic!("expected SETTINGS"),
        }
    }

    #[test]
    fn client_preface_empty_settings() {
        let mut buf = Vec::new();
        write_client_preface(&mut buf, &[]);
        // magic + empty SETTINGS header.
        assert_eq!(buf.len(), 24 + 9);
        let (frame, _) = Frame::decode(&buf[24..]).unwrap();
        assert!(matches!(
            frame.payload,
            Payload::Settings { ack: false, params } if params.is_empty()
        ));
    }

    #[test]
    fn server_preface_is_just_settings() {
        let settings = vec![Setting {
            id: MAX_FRAME_SIZE,
            value: 16_384,
        }];
        let mut buf = Vec::new();
        write_server_preface(&mut buf, &settings);
        let (frame, consumed) = Frame::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(frame.ty(), Some(FrameType::Settings));
        match frame.payload {
            Payload::Settings { params, .. } => {
                assert_eq!(
                    params,
                    vec![Setting {
                        id: MAX_FRAME_SIZE,
                        value: 16_384
                    }]
                );
            }
            _ => panic!("expected SETTINGS"),
        }
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bad = *CLIENT_MAGIC;
        bad[0] = b'X';
        assert_eq!(check_client_magic(&bad), Err(PrefaceError::BadMagic));
        assert_eq!(check_client_magic(&bad[..10]), Err(PrefaceError::Truncated));
        assert_eq!(
            check_client_magic(b"GET / HTTP/1.1\r\n\r\nSM\r\n\r\nxx"),
            Err(PrefaceError::BadMagic)
        );
    }
}
