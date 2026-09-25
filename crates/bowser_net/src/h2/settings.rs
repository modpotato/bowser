//! SETTINGS parameters (RFC 9113, Section 6.5).
//!
//! [`Settings`] tracks one side's settings: either the peer's (applied from
//! received SETTINGS frames) or ours (advertised in our SETTINGS frame).
//! [`SettingsProfile`] is the hook for fingerprint profiles (e.g. transcribed
//! wreq-util emulate profiles): it carries a profile's SETTINGS values and
//! applies them to our local settings.

/// SETTINGS_HEADER_TABLE_SIZE: HPACK dynamic table size.
pub const HEADER_TABLE_SIZE: u16 = 0x1;
/// SETTINGS_ENABLE_PUSH.
pub const ENABLE_PUSH: u16 = 0x2;
/// SETTINGS_MAX_CONCURRENT_STREAMS.
pub const MAX_CONCURRENT_STREAMS: u16 = 0x3;
/// SETTINGS_INITIAL_WINDOW_SIZE.
pub const INITIAL_WINDOW_SIZE: u16 = 0x4;
/// SETTINGS_MAX_FRAME_SIZE.
pub const MAX_FRAME_SIZE: u16 = 0x5;
/// SETTINGS_MAX_HEADER_LIST_SIZE.
pub const MAX_HEADER_LIST_SIZE: u16 = 0x6;

/// Protocol defaults (RFC 9113, Section 6.5.2).
pub const DEFAULT_HEADER_TABLE_SIZE: u32 = 4096;
pub const DEFAULT_ENABLE_PUSH: bool = true;
/// "Infinite" is represented as `u32::MAX`.
pub const DEFAULT_MAX_CONCURRENT_STREAMS: u32 = u32::MAX;
pub const DEFAULT_INITIAL_WINDOW_SIZE: u32 = 65_535;
pub const DEFAULT_MAX_FRAME_SIZE: u32 = 16_384;
pub const DEFAULT_MAX_HEADER_LIST_SIZE: u32 = u32::MAX;

/// Bounds from RFC 9113, Section 6.5.2.
pub const MIN_MAX_FRAME_SIZE: u32 = 16_384;
pub const MAX_MAX_FRAME_SIZE: u32 = 16_777_215;
pub const MAX_INITIAL_WINDOW_SIZE: u32 = 0x7fff_ffff;

/// One SETTINGS parameter: 2-byte id + 4-byte value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Setting {
    pub id: u16,
    pub value: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsError {
    /// SETTINGS payload length is not a multiple of 6.
    BadLength,
    /// A parameter value is outside its legal range.
    BadValue { id: u16 },
}

/// Decode a SETTINGS payload into parameters.
pub fn decode_settings(payload: &[u8]) -> Result<Vec<Setting>, SettingsError> {
    if !payload.len().is_multiple_of(6) {
        return Err(SettingsError::BadLength);
    }
    let mut out = Vec::with_capacity(payload.len() / 6);
    let (chunks, _) = payload.as_chunks::<6>();
    for chunk in chunks {
        out.push(Setting {
            id: u16::from_be_bytes([chunk[0], chunk[1]]),
            value: u32::from_be_bytes([chunk[2], chunk[3], chunk[4], chunk[5]]),
        });
    }
    Ok(out)
}

/// Encode parameters into a SETTINGS payload.
pub fn encode_settings(settings: &[Setting], out: &mut Vec<u8>) {
    for s in settings {
        out.extend_from_slice(&s.id.to_be_bytes());
        out.extend_from_slice(&s.value.to_be_bytes());
    }
}

/// One side's HTTP/2 settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    pub header_table_size: u32,
    pub enable_push: bool,
    pub max_concurrent_streams: u32,
    pub initial_window_size: u32,
    pub max_frame_size: u32,
    /// Advisory limit; `u32::MAX` means unset (infinite).
    pub max_header_list_size: u32,
}

impl Settings {
    /// Protocol defaults.
    pub fn new() -> Self {
        Settings {
            header_table_size: DEFAULT_HEADER_TABLE_SIZE,
            enable_push: DEFAULT_ENABLE_PUSH,
            max_concurrent_streams: DEFAULT_MAX_CONCURRENT_STREAMS,
            initial_window_size: DEFAULT_INITIAL_WINDOW_SIZE,
            max_frame_size: DEFAULT_MAX_FRAME_SIZE,
            max_header_list_size: DEFAULT_MAX_HEADER_LIST_SIZE,
        }
    }

    /// Validate and apply received SETTINGS parameters.
    ///
    /// Unknown parameter ids are ignored, per RFC 9113 Section 6.5. The
    /// returned [`SettingsDelta`] tells the connection what changed in a
    /// form it can act on (window rebasing, HPACK table resize).
    pub fn apply(&mut self, params: &[Setting]) -> Result<SettingsDelta, SettingsError> {
        for p in params {
            match p.id {
                HEADER_TABLE_SIZE => {}
                ENABLE_PUSH => {
                    if p.value > 1 {
                        return Err(SettingsError::BadValue { id: p.id });
                    }
                }
                MAX_CONCURRENT_STREAMS => {}
                INITIAL_WINDOW_SIZE => {
                    if p.value > MAX_INITIAL_WINDOW_SIZE {
                        return Err(SettingsError::BadValue { id: p.id });
                    }
                }
                MAX_FRAME_SIZE => {
                    if !(MIN_MAX_FRAME_SIZE..=MAX_MAX_FRAME_SIZE).contains(&p.value) {
                        return Err(SettingsError::BadValue { id: p.id });
                    }
                }
                MAX_HEADER_LIST_SIZE => {}
                _ => {} // extension parameter: ignore
            }
        }

        let old_initial = self.initial_window_size;
        for p in params {
            match p.id {
                HEADER_TABLE_SIZE => self.header_table_size = p.value,
                ENABLE_PUSH => self.enable_push = p.value == 1,
                MAX_CONCURRENT_STREAMS => self.max_concurrent_streams = p.value,
                INITIAL_WINDOW_SIZE => self.initial_window_size = p.value,
                MAX_FRAME_SIZE => self.max_frame_size = p.value,
                MAX_HEADER_LIST_SIZE => self.max_header_list_size = p.value,
                _ => {}
            }
        }
        Ok(SettingsDelta {
            initial_window_delta: self.initial_window_size as i64 - old_initial as i64,
            header_table_size: self.header_table_size,
            max_frame_size: self.max_frame_size,
        })
    }

    /// Encode all parameters (used for our opening SETTINGS frame).
    pub fn encode_all(&self, out: &mut Vec<u8>) {
        encode_settings(
            &[
                Setting {
                    id: HEADER_TABLE_SIZE,
                    value: self.header_table_size,
                },
                Setting {
                    id: ENABLE_PUSH,
                    value: u32::from(self.enable_push),
                },
                Setting {
                    id: MAX_CONCURRENT_STREAMS,
                    value: self.max_concurrent_streams,
                },
                Setting {
                    id: INITIAL_WINDOW_SIZE,
                    value: self.initial_window_size,
                },
                Setting {
                    id: MAX_FRAME_SIZE,
                    value: self.max_frame_size,
                },
                Setting {
                    id: MAX_HEADER_LIST_SIZE,
                    value: self.max_header_list_size,
                },
            ],
            out,
        );
    }
}

impl Default for Settings {
    fn default() -> Self {
        Settings::new()
    }
}

/// What changed when settings were applied.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SettingsDelta {
    /// Change in INITIAL_WINDOW_SIZE; rebase open streams' send windows.
    pub initial_window_delta: i64,
    /// New HEADER_TABLE_SIZE; resize the HPACK dynamic table.
    pub header_table_size: u32,
    /// New MAX_FRAME_SIZE; bound outgoing frame sizes.
    pub max_frame_size: u32,
}

/// A fingerprint profile's HTTP/2 SETTINGS block.
///
/// Profiles (e.g. transcribed from wreq-util's emulate profiles) only carry
/// the parameters they care about; `None` fields are not sent. This type is
/// the seam between the vendored profile data and the wire: profiles fill
/// in a `SettingsProfile`, and the connection applies it to its local
/// settings before writing the opening SETTINGS frame.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SettingsProfile {
    pub header_table_size: Option<u32>,
    pub enable_push: Option<bool>,
    pub max_concurrent_streams: Option<u32>,
    pub initial_window_size: Option<u32>,
    pub max_frame_size: Option<u32>,
    pub max_header_list_size: Option<u32>,
}

impl SettingsProfile {
    /// The profile's parameters in id order, for the SETTINGS payload.
    pub fn to_settings(&self) -> Vec<Setting> {
        let mut v = Vec::new();
        if let Some(x) = self.header_table_size {
            v.push(Setting {
                id: HEADER_TABLE_SIZE,
                value: x,
            });
        }
        if let Some(x) = self.enable_push {
            v.push(Setting {
                id: ENABLE_PUSH,
                value: u32::from(x),
            });
        }
        if let Some(x) = self.max_concurrent_streams {
            v.push(Setting {
                id: MAX_CONCURRENT_STREAMS,
                value: x,
            });
        }
        if let Some(x) = self.initial_window_size {
            v.push(Setting {
                id: INITIAL_WINDOW_SIZE,
                value: x,
            });
        }
        if let Some(x) = self.max_frame_size {
            v.push(Setting {
                id: MAX_FRAME_SIZE,
                value: x,
            });
        }
        if let Some(x) = self.max_header_list_size {
            v.push(Setting {
                id: MAX_HEADER_LIST_SIZE,
                value: x,
            });
        }
        v
    }

    /// Apply the profile's values to our local (to-be-advertised) settings,
    /// validating ranges exactly like received SETTINGS.
    pub fn apply_to(&self, local: &mut Settings) -> Result<SettingsDelta, SettingsError> {
        local.apply(&self.to_settings())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_payload_roundtrip() {
        let params = vec![
            Setting {
                id: HEADER_TABLE_SIZE,
                value: 4096,
            },
            Setting {
                id: INITIAL_WINDOW_SIZE,
                value: 65_535,
            },
            Setting { id: 0x99, value: 7 },
        ];
        let mut buf = Vec::new();
        encode_settings(&params, &mut buf);
        assert_eq!(buf.len(), 18);
        assert_eq!(decode_settings(&buf).unwrap(), params);
    }

    #[test]
    fn decode_bad_length() {
        assert_eq!(
            decode_settings(&[0, 4, 0, 0, 255]),
            Err(SettingsError::BadLength)
        );
        assert!(decode_settings(&[]).unwrap().is_empty());
    }

    #[test]
    fn apply_valid_params() {
        let mut s = Settings::new();
        let delta = s
            .apply(&[
                Setting {
                    id: HEADER_TABLE_SIZE,
                    value: 256,
                },
                Setting {
                    id: INITIAL_WINDOW_SIZE,
                    value: 131_070,
                },
                Setting {
                    id: MAX_FRAME_SIZE,
                    value: 16_777_215,
                },
            ])
            .unwrap();
        assert_eq!(s.header_table_size, 256);
        assert_eq!(s.initial_window_size, 131_070);
        assert_eq!(s.max_frame_size, 16_777_215);
        assert_eq!(delta.initial_window_delta, 131_070 - 65_535);
        assert_eq!(delta.header_table_size, 256);
        assert_eq!(delta.max_frame_size, 16_777_215);
    }

    #[test]
    fn apply_rejects_bad_enable_push() {
        let mut s = Settings::new();
        assert_eq!(
            s.apply(&[Setting {
                id: ENABLE_PUSH,
                value: 2
            }]),
            Err(SettingsError::BadValue { id: ENABLE_PUSH })
        );
        // Nothing applied on failure.
        assert_eq!(s, Settings::new());
    }

    #[test]
    fn apply_rejects_bad_initial_window_size() {
        let mut s = Settings::new();
        assert_eq!(
            s.apply(&[Setting {
                id: INITIAL_WINDOW_SIZE,
                value: 0x8000_0000
            }]),
            Err(SettingsError::BadValue {
                id: INITIAL_WINDOW_SIZE
            })
        );
        // Boundary ok.
        s.apply(&[Setting {
            id: INITIAL_WINDOW_SIZE,
            value: 0x7fff_ffff,
        }])
        .unwrap();
        assert_eq!(s.initial_window_size, 0x7fff_ffff);
    }

    #[test]
    fn apply_rejects_bad_max_frame_size() {
        let mut s = Settings::new();
        for v in [16_383, 16_777_216] {
            assert_eq!(
                s.apply(&[Setting {
                    id: MAX_FRAME_SIZE,
                    value: v
                }]),
                Err(SettingsError::BadValue { id: MAX_FRAME_SIZE }),
                "value {v}"
            );
        }
        for v in [16_384, 16_777_215] {
            s.apply(&[Setting {
                id: MAX_FRAME_SIZE,
                value: v,
            }])
            .unwrap();
        }
    }

    #[test]
    fn apply_ignores_unknown_ids() {
        let mut s = Settings::new();
        let delta = s
            .apply(&[Setting {
                id: 0xfeed,
                value: 123,
            }])
            .unwrap();
        assert_eq!(s, Settings::new());
        assert_eq!(delta.initial_window_delta, 0);
    }

    #[test]
    fn profile_to_settings_id_order() {
        let p = SettingsProfile {
            max_frame_size: Some(16_384),
            header_table_size: Some(4096),
            enable_push: Some(false),
            ..Default::default()
        };
        let ids: Vec<u16> = p.to_settings().iter().map(|s| s.id).collect();
        assert_eq!(ids, vec![HEADER_TABLE_SIZE, ENABLE_PUSH, MAX_FRAME_SIZE]);
    }

    #[test]
    fn profile_apply_to_local() {
        // Mimics a fingerprint profile: small table, no push, custom windows.
        let p = SettingsProfile {
            header_table_size: Some(4096),
            enable_push: Some(false),
            max_concurrent_streams: Some(100),
            initial_window_size: Some(1_048_576),
            max_frame_size: Some(16_384),
            max_header_list_size: None,
        };
        let mut local = Settings::new();
        let delta = p.apply_to(&mut local).unwrap();
        assert!(!local.enable_push);
        assert_eq!(local.max_concurrent_streams, 100);
        assert_eq!(local.initial_window_size, 1_048_576);
        assert_eq!(delta.initial_window_delta, 1_048_576 - 65_535);

        // The opening SETTINGS payload carries exactly the profile's params.
        let mut payload = Vec::new();
        encode_settings(&p.to_settings(), &mut payload);
        assert_eq!(payload.len(), 5 * 6);
        assert_eq!(decode_settings(&payload).unwrap(), p.to_settings());
    }

    #[test]
    fn profile_rejects_bad_values() {
        let p = SettingsProfile {
            max_frame_size: Some(100),
            ..Default::default()
        };
        let mut local = Settings::new();
        assert_eq!(
            p.apply_to(&mut local),
            Err(SettingsError::BadValue { id: MAX_FRAME_SIZE })
        );
    }

    #[test]
    fn encode_all_covers_six_params() {
        let mut buf = Vec::new();
        Settings::new().encode_all(&mut buf);
        assert_eq!(buf.len(), 36);
        let back = decode_settings(&buf).unwrap();
        assert_eq!(back[0].id, HEADER_TABLE_SIZE);
        assert_eq!(back[5].id, MAX_HEADER_LIST_SIZE);
    }
}
