//! `Date`: constructor, statics (`now`, `parse`, `UTC`), and
//! `Date.prototype` methods.
//!
//! # Time value model
//!
//! A `Date` instance is an [`ObjectKind::Date`][crate::value::ObjectKind]
//! object whose time value (milliseconds since the Unix epoch, `NaN` for an
//! invalid date) is carried in the kind tag as raw `f64` bits. The tag is
//! unforgeable (the old hidden `"\u{0}bowser:date-value"` data property was
//! writable from script) and keeps the slot out of enumeration and JSON
//! serialization (see [`crate::bindings_storage`]); the marker kind gives
//! O(1) brand checks and `"[object Date]"` via `Object.prototype.toString`.
//!
//! # Timezone policy (deliberate v1 deviation)
//!
//! `std` has no timezone database and this crate takes no dependencies, so
//! the local-time surface is UTC-based: [`LOCAL_OFFSET_MS`] is `0`,
//! `getTimezoneOffset()` returns `0`, and `toString()` renders `GMT+0000`.
//! Every conversion funnels through [`to_local`]/[`to_utc`], so a real
//! offset lookup can be dropped into [`local_offset_ms`] later without
//! touching the method implementations. DST rules and IANA zones are not
//! implemented.
//!
//! # Deliberate v1 deviations
//!
//! * `Date.parse` implements the spec's simplified ISO 8601 subset only
//!   (`YYYY-MM-DD`, optional `T`/`t`/space time, optional `.sss`, optional
//!   `Z` or numeric offset). Anything else — including V8's
//!   implementation-specific formats like `"Sep 30 2026"` or
//!   `"2026/09/30"` — yields `NaN`.
//! * `toLocaleString` / `toLocaleDateString` / `toLocaleTimeString` use a
//!   fixed en-US format; there is no `Intl`.
//! * `JSON.stringify(date)` does not consult the inherited `toJSON`:
//!   [`crate::builtin_json`]'s `toJSON` hook is own-properties-only (its
//!   documented v1 gap).
//! * `structuredClone(date)` throws `TypeError` (same as `RegExp`).
//! * The date-time math is exact: after [`time_clip`] every time value is an
//!   integral `f64` with `|t| <= 8.64e15 < 2^53`, so all field splits are
//!   exact integer arithmetic in floating point.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::builtin_object_array::{
    PrimitiveHint, call_function, get_prop, is_callable, to_primitive_with_hint,
};
use crate::builtin_string_number::{
    arg, define_native, define_native_key, js_num, js_to_number, js_to_string_rust,
    new_native_function, with_temp_root,
};
use crate::gc::{Gc, Heap, PropKey, WellKnown};
use crate::value::{NativeError, NativeFn, Object, ObjectKind, Property, Value};
use crate::wtf8::{Wtf8, Wtf8Buf};

/// Milliseconds per day.
const MS_PER_DAY: f64 = 86_400_000.0;

/// The spec's `TimeClip` range: `|t| > 8.64e15` is not a time value.
const TIME_CLIP_RANGE: f64 = 8.64e15;

/// v1 local-time policy: the environment is treated as UTC.
/// See the module docs.
const LOCAL_OFFSET_MS: f64 = 0.0;

/// The local UTC offset in milliseconds. v1: always `0` (UTC); the single
/// choke point for a future real lookup.
fn local_offset_ms() -> f64 {
    LOCAL_OFFSET_MS
}

/// UTC time value -> local time value.
fn to_local(t: f64) -> f64 {
    t + local_offset_ms()
}

/// Local time value -> UTC time value.
fn to_utc(t: f64) -> f64 {
    t - local_offset_ms()
}

/// Spec `TimeClip`: `NaN` and out-of-range become `NaN`; otherwise truncate
/// toward zero (preserving `-0`).
fn time_clip(t: f64) -> f64 {
    if t.is_nan() || t.abs() > TIME_CLIP_RANGE {
        f64::NAN
    } else {
        t.trunc()
    }
}

/// Whole days since the epoch: `floor(t / 86400000)`.
fn day(t: f64) -> f64 {
    (t / MS_PER_DAY).floor()
}

/// Milliseconds within the day: `t - Day(t) * 86400000`, in `[0, 86400000)`.
/// Exact: `t` is integral here and `Day(t) * 86400000 <= |t| <= 8.64e15`.
fn time_within_day(t: f64) -> f64 {
    t - day(t) * MS_PER_DAY
}

/// Days since 1970-01-01 for a civil date (`month` 1-12, `day` 1-31).
/// Howard Hinnant's `days_from_civil`; `i128` internally so no year
/// overflows.
fn days_from_civil(year: i64, month: i64, day: i64) -> i128 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400) as i128;
    let yoe = (y - era as i64 * 400) as i128; // [0, 399]
    let mp = (month + 9) % 12; // [0, 11]
    let doy = (153 * mp as i128 + 2) / 5 + day as i128 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// Inverse of [`days_from_civil`]: `(year, month 1-12, day 1-31)`.
fn civil_from_days(z: i128) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    ((if m <= 2 { y + 1 } else { y }) as i64, m, d)
}

fn is_leap_year(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(y) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Spec `MakeTime(hour, min, sec, ms)`.
fn make_time(hour: f64, min: f64, sec: f64, ms: f64) -> f64 {
    hour * 3_600_000.0 + min * 60_000.0 + sec * 1_000.0 + ms
}

/// Spec `MakeDay(year, month, date)` with 0-based `month` (out-of-range
/// months roll into the year, per spec).
fn make_day(year: i64, month: i64, date: i64) -> i128 {
    let ym = year + month.div_euclid(12);
    let mn = month.rem_euclid(12); // [0, 11]
    days_from_civil(ym, mn + 1, 1) + date as i128 - 1
}

/// Spec `MakeDate(day, time)`.
fn make_date(day: i128, time: f64) -> f64 {
    day as f64 * MS_PER_DAY + time
}

/// Broken-down fields of a time value. `month` is 0-based, `weekday` is
/// 0 = Sunday.
struct DateFields {
    year: i64,
    month: u32,
    date: u32,
    weekday: u32,
    hour: u32,
    minute: u32,
    second: u32,
    ms: u32,
}

/// Split a finite time value into fields. Exact: `t` is an integral `f64`
/// with `|t| <= 8.64e15`, so `day(t)` fits `i128` trivially and the
/// within-day remainder is exact.
fn split_fields(t: f64) -> DateFields {
    debug_assert!(t.is_finite());
    let d = day(t);
    let (year, month1, date) = civil_from_days(d as i128);
    // 1970-01-01 was a Thursday (4).
    let weekday = (d.rem_euclid(7.0) + 4.0).rem_euclid(7.0) as u32;
    let tod = time_within_day(t);
    let hour = (tod / 3_600_000.0) as u32;
    let minute = ((tod % 3_600_000.0) / 60_000.0) as u32;
    let second = ((tod % 60_000.0) / 1_000.0) as u32;
    let ms = (tod % 1_000.0) as u32;
    DateFields {
        year,
        month: month1 - 1,
        date,
        weekday,
        hour,
        minute,
        second,
        ms,
    }
}

/// `Date.now()`: system clock as ms since the epoch.
fn now_ms() -> f64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_millis() as f64,
        Err(e) => -(e.duration().as_millis() as f64),
    }
}

// ---------------------------------------------------------------------------
// `Date.parse`: the spec's simplified ISO 8601 subset.
// ---------------------------------------------------------------------------

/// `Date.parse` over the spec's Date Time String Format:
/// `YYYY-MM-DD`, optional `T`/`t`/space time `HH:mm[:ss[.sss]]`, optional
/// `Z` or `±HH:mm` / `±HHmm` / `±HH` offset. `NaN` when the string does not
/// conform (no implementation-specific fallback in v1).
fn parse_date_time(s: &str) -> f64 {
    let s = crate::builtin_string_number::trim_js(s);
    if !s.is_ascii() {
        return f64::NAN;
    }
    let b = s.as_bytes();
    let mut p = IsoParser { b, pos: 0 };
    match p.parse() {
        Some(t) if p.pos == b.len() => time_clip(t),
        _ => f64::NAN,
    }
}

struct IsoParser<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> IsoParser<'a> {
    /// Structural parse of the ISO subset: `None` when the bytes do not
    /// match the grammar, `Some(t)` (possibly `NaN` for semantically
    /// invalid dates like month 13) when they do.
    fn parse(&mut self) -> Option<f64> {
        // Year: `YYYY` or signed `±YYYYYY`.
        let year = if self.peek() == Some(b'+') || self.peek() == Some(b'-') {
            let neg = self.bump() == b'-';
            let y = self.digits(6)?;
            if neg { -(y as i64) } else { y as i64 }
        } else {
            self.digits(4)? as i64
        };
        self.expect(b'-')?;
        let month = self.digits(2)?;
        if !(1..=12).contains(&month) {
            return Some(f64::NAN);
        }
        self.expect(b'-')?;
        let day = self.digits(2)?;
        if day < 1 || day > days_in_month(year, month) {
            return Some(f64::NAN);
        }
        let mut time = 0.0;
        let mut offset_ms = 0.0;
        // Optional time part: `THH:mm[:ss[.sss]]` + optional zone.
        if let Some(c) = self.peek()
            && (c == b'T' || c == b't' || c == b' ')
        {
            self.pos += 1;
            let hour = self.digits(2)?;
            self.expect(b':')?;
            let minute = self.digits(2)?;
            if minute > 59 {
                return Some(f64::NAN);
            }
            let mut second = 0u32;
            let mut ms = 0u32;
            if self.peek() == Some(b':') {
                self.pos += 1;
                second = self.digits(2)?;
                if second > 59 {
                    return Some(f64::NAN);
                }
                if self.peek() == Some(b'.') {
                    self.pos += 1;
                    // 1-3 fractional digits, right-padded to milliseconds.
                    let start = self.pos;
                    while self.pos < self.b.len()
                        && self.b[self.pos].is_ascii_digit()
                        && self.pos - start < 3
                    {
                        self.pos += 1;
                    }
                    let n = self.pos - start;
                    if n == 0 {
                        return Some(f64::NAN);
                    }
                    let mut v = 0u32;
                    for &c in &self.b[start..self.pos] {
                        v = v * 10 + u32::from(c - b'0');
                    }
                    for _ in n..3 {
                        v *= 10;
                    }
                    ms = v;
                }
            }
            // `24:00:00.000` is midnight at the end of the day.
            if hour == 24 {
                if minute != 0 || second != 0 || ms != 0 {
                    return Some(f64::NAN);
                }
            } else if hour > 24 {
                return Some(f64::NAN);
            }
            time = make_time(hour as f64, minute as f64, second as f64, ms as f64);
            // Optional zone: `Z` or `±HH:mm` / `±HHmm` / `±HH`.
            match self.peek() {
                Some(b'Z') | Some(b'z') => {
                    self.pos += 1;
                }
                Some(b'+') | Some(b'-') => {
                    let neg = self.bump() == b'-';
                    let oh = self.digits(2)?;
                    let mut om = 0u32;
                    if self.peek() == Some(b':') {
                        self.pos += 1;
                        om = self.digits(2)?;
                    } else if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                        om = self.digits(2)?;
                    }
                    if oh > 23 || om > 59 {
                        return Some(f64::NAN);
                    }
                    offset_ms = (oh as f64 * 60.0 + om as f64) * 60_000.0;
                    if neg {
                        offset_ms = -offset_ms;
                    }
                }
                _ => {}
            }
        }
        let day_n = make_day(year, month as i64 - 1, day as i64);
        Some(make_date(day_n, time) - offset_ms)
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    fn bump(&mut self) -> u8 {
        let c = self.b[self.pos];
        self.pos += 1;
        c
    }

    fn expect(&mut self, want: u8) -> Option<()> {
        if self.peek() == Some(want) {
            self.pos += 1;
            Some(())
        } else {
            None
        }
    }

    /// Exactly `n` ASCII digits as `u32`; `None` on any mismatch.
    fn digits(&mut self, n: usize) -> Option<u32> {
        if self.pos + n > self.b.len() {
            return None;
        }
        let mut v: u32 = 0;
        for &c in &self.b[self.pos..self.pos + n] {
            if !c.is_ascii_digit() {
                return None;
            }
            v = v * 10 + u32::from(c - b'0');
        }
        self.pos += n;
        Some(v)
    }
}

// ---------------------------------------------------------------------------
// Slot access and receiver brand checks
// ---------------------------------------------------------------------------

/// Read the kind-carried time value. `None` when `obj` is not a `Date`
/// instance (brand check via [`ObjectKind::Date`]).
fn date_slot(heap: &Heap, obj: Gc<Object>) -> Option<f64> {
    match heap.get(obj).kind {
        ObjectKind::Date { value_bits } => Some(f64::from_bits(value_bits)),
        _ => None,
    }
}

/// Store an already-clipped time value in the kind tag.
fn set_time_value(heap: &mut Heap, obj: Gc<Object>, t: f64) {
    heap.set_object_kind(
        obj,
        ObjectKind::Date {
            value_bits: t.to_bits(),
        },
    );
}

/// Spec `thisTimeValue`: the receiver must be a `Date` instance.
fn this_time_value(heap: &Heap, this: Value, method: &str) -> Result<f64, NativeError> {
    match this {
        Value::Object(o) => date_slot(heap, o).ok_or_else(|| {
            NativeError::TypeError(format!(
                "Method Date.prototype.{method} called on incompatible receiver"
            ))
        }),
        _ => Err(NativeError::TypeError(format!(
            "Method Date.prototype.{method} called on incompatible receiver"
        ))),
    }
}

/// The spec's `ToIntegerOrInfinity` for `MakeDay` arguments (`month`,
/// `date`) and the setters' integer components: fractions truncate toward
/// zero (e.g. `new Date(2026, 1.9)` is February, `setMonth(1.9)` sets
/// February). Non-finite components make the whole date `NaN`.
fn int_component(v: Value) -> Result<Option<i64>, NativeError> {
    let n = js_to_number(v)?;
    if !n.is_finite() {
        return Ok(None);
    }
    // `as` saturates on overflow; the resulting huge day count fails
    // `time_clip` below, yielding `NaN` exactly as the spec does.
    Ok(Some(n.trunc() as i64))
}

/// Spec `MakeFullYear`: `NaN`/infinities are invalid; `0..=99` maps to
/// `1900+year`.
fn make_full_year(y: f64) -> Option<i64> {
    if !y.is_finite() {
        return None;
    }
    let yi = y.trunc() as i64;
    Some(if (0..=99).contains(&yi) {
        1900 + yi
    } else {
        yi
    })
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Single-argument construction: a `Date` argument copies its time value;
/// otherwise `ToPrimitive(value, default)` — strings parse, everything
/// else goes through `ToNumber`.
fn single_arg_time(heap: &mut Heap, v: Value) -> Result<f64, NativeError> {
    if let Value::Object(o) = v
        && let Some(t) = date_slot(heap, o)
    {
        return Ok(t);
    }
    let prim = match v {
        Value::Object(o) => to_primitive_with_hint(heap, o, PrimitiveHint::Default)?,
        _ => v,
    };
    match prim {
        Value::Str(s) => Ok(parse_date_time(s.as_wtf8().as_utf8().unwrap_or(""))),
        _ => Ok(js_to_number(prim)?),
    }
}

/// Multi-argument construction (`new Date(y, m, [d, [h, [min, [s, [ms]]]]])`).
/// `local` selects the constructor (components are local time) vs
/// `Date.UTC` (components are already UTC) form.
fn multi_arg_time(args: &[Value], local: bool) -> Result<f64, NativeError> {
    let absent = |i: usize, default: Value| {
        if args.len() > i {
            arg(args, i)
        } else {
            default
        }
    };
    let yr = match make_full_year(js_to_number(arg(args, 0))?) {
        Some(yr) => yr,
        None => return Ok(f64::NAN),
    };
    let m = match int_component(absent(1, Value::Int(0)))? {
        Some(m) => m,
        None => return Ok(f64::NAN),
    };
    let dt = match int_component(absent(2, Value::Int(1)))? {
        Some(dt) => dt,
        None => return Ok(f64::NAN),
    };
    // Time components are real numbers per spec (`MakeTime` folds
    // fractions in; `TimeClip` truncates at the end).
    let h = js_to_number(absent(3, Value::Int(0)))?;
    let min = js_to_number(absent(4, Value::Int(0)))?;
    let s = js_to_number(absent(5, Value::Int(0)))?;
    let ms = js_to_number(absent(6, Value::Int(0)))?;
    let time = make_time(h, min, s, ms);
    if !time.is_finite() {
        return Ok(f64::NAN);
    }
    let date = make_date(make_day(yr, m, dt), time);
    Ok(if local { to_utc(date) } else { date })
}

/// `Date(...)`: the constructor, also callable without `new` (call form
/// returns the date-time string for now, per spec).
fn date_construct(heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let Some(Value::Object(new_target)) = crate::vm::active_new_target() else {
        return Ok(Value::Str(
            heap.alloc_str(&format_date_time_string(now_ms()))?,
        ));
    };
    let t = time_clip(match args.len() {
        0 => now_ms(),
        1 => single_arg_time(heap, arg(args, 0))?,
        _ => multi_arg_time(args, true)?,
    });
    let obj = heap.alloc_object(Object::new(ObjectKind::Date {
        value_bits: t.to_bits(),
    }))?;
    // Instances chain to `new.target`'s `"prototype"` when it is an
    // object (the `RegExp` construction discipline).
    if let Some(p) = heap.get_property(new_target, Wtf8::from_str("prototype"))
        && matches!(p.value, Value::Object(_))
    {
        heap.set_prototype(obj, p.value);
    }
    Ok(Value::Object(obj))
}

// ---------------------------------------------------------------------------
// Static methods
// ---------------------------------------------------------------------------

/// `Date.now()`.
fn date_now(_heap: &mut Heap, _this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(now_ms()))
}

/// `Date.parse(string)`: `ToString` then the ISO subset.
fn date_parse(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let s = js_to_string_rust(arg(args, 0))?;
    // The date parser works on `&str`: an ill-formed string cannot match
    // the ISO subset, so it parses as `NaN`.
    Ok(js_num(parse_date_time(s.as_utf8().unwrap_or(""))))
}

/// `Date.UTC(year, [month, [date, ...]])`: like the constructor's
/// multi-argument form but the components are already UTC (no local
/// conversion). A missing `year` makes the result `NaN`.
fn date_utc(_heap: &mut Heap, _this: Value, args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(time_clip(multi_arg_time(args, false)?)))
}

// ---------------------------------------------------------------------------
// Prototype getters
// ---------------------------------------------------------------------------

/// Shared getter skeleton: brand-check, `NaN` short-circuit, optional
/// local conversion, field split, project.
fn proto_get(
    heap: &Heap,
    this: Value,
    method: &str,
    local: bool,
    pick: impl Fn(&DateFields) -> i64,
) -> Result<Value, NativeError> {
    let t = this_time_value(heap, this, method)?;
    if t.is_nan() {
        return Ok(Value::Float(f64::NAN));
    }
    let fields = split_fields(if local { to_local(t) } else { t });
    Ok(js_num(pick(&fields) as f64))
}

/// `getTime` / `valueOf`: the raw time value.
fn date_get_time(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    Ok(js_num(this_time_value(heap, this, "getTime")?))
}

fn date_get_full_year(heap: &mut Heap, this: Value, _a: &[Value]) -> Result<Value, NativeError> {
    proto_get(heap, this, "getFullYear", true, |f| f.year)
}
fn date_get_utc_full_year(
    heap: &mut Heap,
    this: Value,
    _a: &[Value],
) -> Result<Value, NativeError> {
    proto_get(heap, this, "getUTCFullYear", false, |f| f.year)
}
fn date_get_month(heap: &mut Heap, this: Value, _a: &[Value]) -> Result<Value, NativeError> {
    proto_get(heap, this, "getMonth", true, |f| i64::from(f.month))
}
fn date_get_utc_month(heap: &mut Heap, this: Value, _a: &[Value]) -> Result<Value, NativeError> {
    proto_get(heap, this, "getUTCMonth", false, |f| i64::from(f.month))
}
fn date_get_date(heap: &mut Heap, this: Value, _a: &[Value]) -> Result<Value, NativeError> {
    proto_get(heap, this, "getDate", true, |f| i64::from(f.date))
}
fn date_get_utc_date(heap: &mut Heap, this: Value, _a: &[Value]) -> Result<Value, NativeError> {
    proto_get(heap, this, "getUTCDate", false, |f| i64::from(f.date))
}
fn date_get_day(heap: &mut Heap, this: Value, _a: &[Value]) -> Result<Value, NativeError> {
    proto_get(heap, this, "getDay", true, |f| i64::from(f.weekday))
}
fn date_get_utc_day(heap: &mut Heap, this: Value, _a: &[Value]) -> Result<Value, NativeError> {
    proto_get(heap, this, "getUTCDay", false, |f| i64::from(f.weekday))
}
fn date_get_hours(heap: &mut Heap, this: Value, _a: &[Value]) -> Result<Value, NativeError> {
    proto_get(heap, this, "getHours", true, |f| i64::from(f.hour))
}
fn date_get_utc_hours(heap: &mut Heap, this: Value, _a: &[Value]) -> Result<Value, NativeError> {
    proto_get(heap, this, "getUTCHours", false, |f| i64::from(f.hour))
}
fn date_get_minutes(heap: &mut Heap, this: Value, _a: &[Value]) -> Result<Value, NativeError> {
    proto_get(heap, this, "getMinutes", true, |f| i64::from(f.minute))
}
fn date_get_utc_minutes(heap: &mut Heap, this: Value, _a: &[Value]) -> Result<Value, NativeError> {
    proto_get(heap, this, "getUTCMinutes", false, |f| i64::from(f.minute))
}
fn date_get_seconds(heap: &mut Heap, this: Value, _a: &[Value]) -> Result<Value, NativeError> {
    proto_get(heap, this, "getSeconds", true, |f| i64::from(f.second))
}
fn date_get_utc_seconds(heap: &mut Heap, this: Value, _a: &[Value]) -> Result<Value, NativeError> {
    proto_get(heap, this, "getUTCSeconds", false, |f| i64::from(f.second))
}
fn date_get_milliseconds(heap: &mut Heap, this: Value, _a: &[Value]) -> Result<Value, NativeError> {
    proto_get(heap, this, "getMilliseconds", true, |f| i64::from(f.ms))
}
fn date_get_utc_milliseconds(
    heap: &mut Heap,
    this: Value,
    _a: &[Value],
) -> Result<Value, NativeError> {
    proto_get(heap, this, "getUTCMilliseconds", false, |f| i64::from(f.ms))
}

/// `getTimezoneOffset()`: `(t - LocalTime(t)) / msPerMinute`.
fn date_get_timezone_offset(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let t = this_time_value(heap, this, "getTimezoneOffset")?;
    if t.is_nan() {
        return Ok(Value::Float(f64::NAN));
    }
    Ok(js_num((t - to_local(t)) / 60_000.0))
}

// ---------------------------------------------------------------------------
// Prototype setters
// ---------------------------------------------------------------------------

/// Brand-check a `Date` method receiver, returning the instance.
fn date_receiver(heap: &Heap, this: Value, method: &str) -> Result<Gc<Object>, NativeError> {
    match this {
        Value::Object(o) if matches!(heap.get(o).kind, ObjectKind::Date { .. }) => Ok(o),
        _ => Err(NativeError::TypeError(format!(
            "Method Date.prototype.{method} called on incompatible receiver"
        ))),
    }
}

/// Resolve a setter receiver: the `Date` object and its base time (local
/// when `local`). Per spec, every component setter returns `NaN` without
/// modifying the date when the current time value is `NaN` — `None` here
/// signals that path. (`setTime` is exempt: it replaces the absolute time
/// value, so it does not go through this helper.)
fn setter_base(
    heap: &Heap,
    this: Value,
    method: &str,
    local: bool,
) -> Result<Option<(Gc<Object>, f64)>, NativeError> {
    let o = date_receiver(heap, this, method)?;
    let t = date_slot(heap, o).unwrap_or(f64::NAN);
    if t.is_nan() {
        return Ok(None);
    }
    Ok(Some((o, if local { to_local(t) } else { t })))
}

/// Finish a setter: convert back to UTC when `local`, clip, store, and
/// return the new time value.
fn setter_store(
    heap: &mut Heap,
    o: Gc<Object>,
    new_t: f64,
    local: bool,
) -> Result<Value, NativeError> {
    let u = time_clip(if local { to_utc(new_t) } else { new_t });
    set_time_value(heap, o, u);
    Ok(js_num(u))
}

/// `setTime(timeValue)`: the one setter that works on invalid dates — it
/// replaces the absolute time value rather than combining components.
fn date_set_time(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let o = date_receiver(heap, this, "setTime")?;
    let u = time_clip(js_to_number(arg(args, 0))?);
    set_time_value(heap, o, u);
    Ok(js_num(u))
}

fn set_milliseconds(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
    method: &str,
    local: bool,
) -> Result<Value, NativeError> {
    let Some((o, base)) = setter_base(heap, this, method, local)? else {
        return Ok(Value::Float(f64::NAN));
    };
    let ms = js_to_number(arg(args, 0))?;
    let f = split_fields(base);
    let time = make_time(f.hour as f64, f.minute as f64, f.second as f64, ms);
    setter_store(heap, o, make_date(day(base) as i128, time), local)
}

fn set_seconds(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
    method: &str,
    local: bool,
) -> Result<Value, NativeError> {
    let Some((o, base)) = setter_base(heap, this, method, local)? else {
        return Ok(Value::Float(f64::NAN));
    };
    let f = split_fields(base);
    let s = js_to_number(arg(args, 0))?;
    let ms = if args.len() > 1 {
        js_to_number(arg(args, 1))?
    } else {
        f.ms as f64
    };
    let time = make_time(f.hour as f64, f.minute as f64, s, ms);
    setter_store(heap, o, make_date(day(base) as i128, time), local)
}

fn set_minutes(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
    method: &str,
    local: bool,
) -> Result<Value, NativeError> {
    let Some((o, base)) = setter_base(heap, this, method, local)? else {
        return Ok(Value::Float(f64::NAN));
    };
    let f = split_fields(base);
    let m = js_to_number(arg(args, 0))?;
    let s = if args.len() > 1 {
        js_to_number(arg(args, 1))?
    } else {
        f.second as f64
    };
    let ms = if args.len() > 2 {
        js_to_number(arg(args, 2))?
    } else {
        f.ms as f64
    };
    let time = make_time(f.hour as f64, m, s, ms);
    setter_store(heap, o, make_date(day(base) as i128, time), local)
}

fn set_hours(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
    method: &str,
    local: bool,
) -> Result<Value, NativeError> {
    let Some((o, base)) = setter_base(heap, this, method, local)? else {
        return Ok(Value::Float(f64::NAN));
    };
    let f = split_fields(base);
    let h = js_to_number(arg(args, 0))?;
    let m = if args.len() > 1 {
        js_to_number(arg(args, 1))?
    } else {
        f.minute as f64
    };
    let s = if args.len() > 2 {
        js_to_number(arg(args, 2))?
    } else {
        f.second as f64
    };
    let ms = if args.len() > 3 {
        js_to_number(arg(args, 3))?
    } else {
        f.ms as f64
    };
    let time = make_time(h, m, s, ms);
    setter_store(heap, o, make_date(day(base) as i128, time), local)
}

fn set_date(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
    method: &str,
    local: bool,
) -> Result<Value, NativeError> {
    let Some((o, base)) = setter_base(heap, this, method, local)? else {
        return Ok(Value::Float(f64::NAN));
    };
    let f = split_fields(base);
    let d = match int_component(arg(args, 0))? {
        Some(d) => d,
        None => return setter_store(heap, o, f64::NAN, local),
    };
    let day_n = make_day(f.year, i64::from(f.month), d);
    setter_store(heap, o, make_date(day_n, time_within_day(base)), local)
}

fn set_month(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
    method: &str,
    local: bool,
) -> Result<Value, NativeError> {
    let Some((o, base)) = setter_base(heap, this, method, local)? else {
        return Ok(Value::Float(f64::NAN));
    };
    let f = split_fields(base);
    let m = match int_component(arg(args, 0))? {
        Some(m) => m,
        None => return setter_store(heap, o, f64::NAN, local),
    };
    let d = if args.len() > 1 {
        match int_component(arg(args, 1))? {
            Some(d) => d,
            None => return setter_store(heap, o, f64::NAN, local),
        }
    } else {
        i64::from(f.date)
    };
    let day_n = make_day(f.year, m, d);
    setter_store(heap, o, make_date(day_n, time_within_day(base)), local)
}

fn set_full_year(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
    method: &str,
    local: bool,
) -> Result<Value, NativeError> {
    let Some((o, base)) = setter_base(heap, this, method, local)? else {
        return Ok(Value::Float(f64::NAN));
    };
    let f = split_fields(base);
    let yr = match make_full_year(js_to_number(arg(args, 0))?) {
        Some(yr) => yr,
        None => return setter_store(heap, o, f64::NAN, local),
    };
    let m = if args.len() > 1 {
        match int_component(arg(args, 1))? {
            Some(m) => m,
            None => return setter_store(heap, o, f64::NAN, local),
        }
    } else {
        i64::from(f.month)
    };
    let d = if args.len() > 2 {
        match int_component(arg(args, 2))? {
            Some(d) => d,
            None => return setter_store(heap, o, f64::NAN, local),
        }
    } else {
        i64::from(f.date)
    };
    let day_n = make_day(yr, m, d);
    setter_store(heap, o, make_date(day_n, time_within_day(base)), local)
}

fn date_set_milliseconds(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    set_milliseconds(heap, this, args, "setMilliseconds", true)
}
fn date_set_utc_milliseconds(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    set_milliseconds(heap, this, args, "setUTCMilliseconds", false)
}
fn date_set_seconds(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    set_seconds(heap, this, args, "setSeconds", true)
}
fn date_set_utc_seconds(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    set_seconds(heap, this, args, "setUTCSeconds", false)
}
fn date_set_minutes(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    set_minutes(heap, this, args, "setMinutes", true)
}
fn date_set_utc_minutes(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    set_minutes(heap, this, args, "setUTCMinutes", false)
}
fn date_set_hours(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    set_hours(heap, this, args, "setHours", true)
}
fn date_set_utc_hours(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    set_hours(heap, this, args, "setUTCHours", false)
}
fn date_set_date(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    set_date(heap, this, args, "setDate", true)
}
fn date_set_utc_date(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    set_date(heap, this, args, "setUTCDate", false)
}
fn date_set_month(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    set_month(heap, this, args, "setMonth", true)
}
fn date_set_utc_month(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    set_month(heap, this, args, "setUTCMonth", false)
}
fn date_set_full_year(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    set_full_year(heap, this, args, "setFullYear", true)
}
fn date_set_utc_full_year(
    heap: &mut Heap,
    this: Value,
    args: &[Value],
) -> Result<Value, NativeError> {
    set_full_year(heap, this, args, "setUTCFullYear", false)
}

// ---------------------------------------------------------------------------
// String conversions
// ---------------------------------------------------------------------------

const WEEKDAY_NAMES: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTH_NAMES: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// Year rendering shared by `toString`-family and `toISOString`: plain
/// 4-digit years in `[0, 9999]`, sign + 6 digits outside (V8 style).
fn format_year(y: i64) -> String {
    if (0..=9999).contains(&y) {
        format!("{y:04}")
    } else {
        format!("{y:+07}")
    }
}

/// Spec `Date.prototype.toString` (local fields; v1 renders `GMT+0000`).
fn format_date_time_string(t: f64) -> String {
    let f = split_fields(to_local(t));
    format!(
        "{} {} {:02} {} {:02}:{:02}:{:02} GMT+0000",
        WEEKDAY_NAMES[f.weekday as usize],
        MONTH_NAMES[f.month as usize],
        f.date,
        format_year(f.year),
        f.hour,
        f.minute,
        f.second
    )
}

/// `toDateString`: `"Mon Sep 28 2026"`.
fn format_date_string(t: f64) -> String {
    let f = split_fields(to_local(t));
    format!(
        "{} {} {:02} {}",
        WEEKDAY_NAMES[f.weekday as usize],
        MONTH_NAMES[f.month as usize],
        f.date,
        format_year(f.year)
    )
}

/// `toTimeString`: `"12:52:37 GMT+0000"`.
fn format_time_string(t: f64) -> String {
    let f = split_fields(to_local(t));
    format!("{:02}:{:02}:{:02} GMT+0000", f.hour, f.minute, f.second)
}

/// Spec `toISOString` (UTC fields); throws `RangeError` on invalid dates.
fn format_iso_string(t: f64) -> Result<String, NativeError> {
    if !t.is_finite() {
        return Err(NativeError::RangeError("Invalid time value".to_string()));
    }
    let f = split_fields(t);
    Ok(format!(
        "{}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        format_year(f.year),
        f.month + 1,
        f.date,
        f.hour,
        f.minute,
        f.second,
        f.ms
    ))
}

/// Fixed en-US 12-hour clock for the `toLocaleTimeString` family (v1: no
/// `Intl`; see the module docs).
fn format_locale_time(f: &DateFields) -> String {
    let hour12 = f.hour % 12;
    let hour12 = if hour12 == 0 { 12 } else { hour12 };
    let suffix = if f.hour < 12 { "AM" } else { "PM" };
    format!("{hour12}:{:02}:{:02} {suffix}", f.minute, f.second)
}

fn date_to_string(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let t = this_time_value(heap, this, "toString")?;
    let s = if t.is_nan() {
        "Invalid Date".to_string()
    } else {
        format_date_time_string(t)
    };
    Ok(Value::Str(heap.alloc_str(&s)?))
}

fn date_to_date_string(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let t = this_time_value(heap, this, "toDateString")?;
    let s = if t.is_nan() {
        "Invalid Date".to_string()
    } else {
        format_date_string(t)
    };
    Ok(Value::Str(heap.alloc_str(&s)?))
}

fn date_to_time_string(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let t = this_time_value(heap, this, "toTimeString")?;
    let s = if t.is_nan() {
        "Invalid Date".to_string()
    } else {
        format_time_string(t)
    };
    Ok(Value::Str(heap.alloc_str(&s)?))
}

fn date_to_iso_string(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let t = this_time_value(heap, this, "toISOString")?;
    Ok(Value::Str(heap.alloc_str(&format_iso_string(t)?)?))
}

fn date_to_locale_string(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let t = this_time_value(heap, this, "toLocaleString")?;
    let s = if t.is_nan() {
        "Invalid Date".to_string()
    } else {
        let f = split_fields(to_local(t));
        format!(
            "{}/{}/{}, {}",
            f.month + 1,
            f.date,
            f.year,
            format_locale_time(&f)
        )
    };
    Ok(Value::Str(heap.alloc_str(&s)?))
}

fn date_to_locale_date_string(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let t = this_time_value(heap, this, "toLocaleDateString")?;
    let s = if t.is_nan() {
        "Invalid Date".to_string()
    } else {
        let f = split_fields(to_local(t));
        format!("{}/{}/{}", f.month + 1, f.date, f.year)
    };
    Ok(Value::Str(heap.alloc_str(&s)?))
}

fn date_to_locale_time_string(
    heap: &mut Heap,
    this: Value,
    _args: &[Value],
) -> Result<Value, NativeError> {
    let t = this_time_value(heap, this, "toLocaleTimeString")?;
    let s = if t.is_nan() {
        "Invalid Date".to_string()
    } else {
        format_locale_time(&split_fields(to_local(t)))
    };
    Ok(Value::Str(heap.alloc_str(&s)?))
}

/// `toJSON(key)`: `null` for non-finite time values, else `toISOString`
/// invoked on the receiver (so overrides are honored, per spec).
fn date_to_json(heap: &mut Heap, this: Value, _args: &[Value]) -> Result<Value, NativeError> {
    let o = match this {
        Value::Object(o) if matches!(heap.get(o).kind, ObjectKind::Date { .. }) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Method Date.prototype.toJSON called on incompatible receiver".to_string(),
            ));
        }
    };
    let tv = to_primitive_with_hint(heap, o, PrimitiveHint::Number)?;
    if !js_to_number(tv)?.is_finite() {
        return Ok(Value::Null);
    }
    let m = get_prop(heap, o, Wtf8::from_str("toISOString")).filter(|v| is_callable(heap, *v));
    match m {
        Some(f) => call_function(heap, f, this, &[]),
        None => Err(NativeError::TypeError(
            "Date.prototype.toISOString is not callable".to_string(),
        )),
    }
}

/// `Date.prototype[@@toPrimitive](hint)`: `"default"` behaves like
/// `"string"` (the spec's Date override), otherwise `OrdinaryToPrimitive`
/// in hint order.
fn date_to_primitive(heap: &mut Heap, this: Value, args: &[Value]) -> Result<Value, NativeError> {
    let hint = match arg(args, 0) {
        Value::Str(s) => s.as_wtf8().to_owned(),
        _ => Wtf8Buf::from("default"),
    };
    let order: [&str; 2] = match hint.as_wtf8() {
        h if h == "string" || h == "default" => ["toString", "valueOf"],
        _ => ["valueOf", "toString"],
    };
    let o = match this {
        Value::Object(o) => o,
        _ => {
            return Err(NativeError::TypeError(
                "Method Date.prototype[@@toPrimitive] called on incompatible receiver".to_string(),
            ));
        }
    };
    for name in order {
        if let Some(m) = get_prop(heap, o, Wtf8::from_str(name))
            && is_callable(heap, m)
        {
            let r = call_function(heap, m, this, &[])?;
            if !matches!(r, Value::Object(_)) {
                return Ok(r);
            }
        }
    }
    Err(NativeError::TypeError(
        "cannot convert Date to primitive".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// install
// ---------------------------------------------------------------------------

/// Install `Date` on `global`: the constructor (with `now`/`parse`/`UTC`
/// statics) and `%Date.prototype%` (itself a `Date` instance with a `NaN`
/// time value, per spec), chained to `%Object.prototype%`.
pub(crate) fn install(heap: &mut Heap, global: Gc<Object>) -> Result<(), NativeError> {
    install_inner(heap, global)
}

fn install_inner(heap: &mut Heap, global: Gc<Object>) -> Result<(), NativeError> {
    let proto = heap.alloc_object(Object::new(ObjectKind::Date {
        value_bits: f64::NAN.to_bits(),
    }))?;
    // `proto` is not reachable from any traced root until `Date` is linked
    // below; hold it across the constructor allocation, which may trigger
    // a collection. The temp root is released on both paths.
    with_temp_root(heap, Value::Object(proto), |heap| {
        // `%Object.prototype%` linkage (install order between builtin workers
        // is not fixed; mirrors `builtin_string_number`).
        if let Some(op) = heap.object_prototype {
            heap.set_prototype(proto, Value::Object(op));
        }

        let methods: &[(&str, NativeFn, u32)] = &[
            ("toString", date_to_string, 0),
            ("toDateString", date_to_date_string, 0),
            ("toTimeString", date_to_time_string, 0),
            ("toISOString", date_to_iso_string, 0),
            ("toJSON", date_to_json, 1),
            ("toLocaleString", date_to_locale_string, 0),
            ("toLocaleDateString", date_to_locale_date_string, 0),
            ("toLocaleTimeString", date_to_locale_time_string, 0),
            ("valueOf", date_get_time, 0),
            ("getTime", date_get_time, 0),
            ("getFullYear", date_get_full_year, 0),
            ("getUTCFullYear", date_get_utc_full_year, 0),
            ("getMonth", date_get_month, 0),
            ("getUTCMonth", date_get_utc_month, 0),
            ("getDate", date_get_date, 0),
            ("getUTCDate", date_get_utc_date, 0),
            ("getDay", date_get_day, 0),
            ("getUTCDay", date_get_utc_day, 0),
            ("getHours", date_get_hours, 0),
            ("getUTCHours", date_get_utc_hours, 0),
            ("getMinutes", date_get_minutes, 0),
            ("getUTCMinutes", date_get_utc_minutes, 0),
            ("getSeconds", date_get_seconds, 0),
            ("getUTCSeconds", date_get_utc_seconds, 0),
            ("getMilliseconds", date_get_milliseconds, 0),
            ("getUTCMilliseconds", date_get_utc_milliseconds, 0),
            ("getTimezoneOffset", date_get_timezone_offset, 0),
            ("setTime", date_set_time, 1),
            ("setMilliseconds", date_set_milliseconds, 1),
            ("setUTCMilliseconds", date_set_utc_milliseconds, 1),
            ("setSeconds", date_set_seconds, 2),
            ("setUTCSeconds", date_set_utc_seconds, 2),
            ("setMinutes", date_set_minutes, 3),
            ("setUTCMinutes", date_set_utc_minutes, 3),
            ("setHours", date_set_hours, 4),
            ("setUTCHours", date_set_utc_hours, 4),
            ("setDate", date_set_date, 1),
            ("setUTCDate", date_set_utc_date, 1),
            ("setMonth", date_set_month, 2),
            ("setUTCMonth", date_set_utc_month, 2),
            ("setFullYear", date_set_full_year, 3),
            ("setUTCFullYear", date_set_utc_full_year, 3),
        ];
        for &(name, func, len) in methods {
            define_native(heap, proto, name, func, len)?;
        }
        // `Date.prototype[Symbol.toPrimitive]`: installed under the real
        // symbol key, which `to_primitive_with_hint` consults.
        let to_primitive_sym = heap.well_known_symbol(WellKnown::ToPrimitive)?;
        define_native_key(
            heap,
            proto,
            PropKey::Symbol(to_primitive_sym),
            "[Symbol.toPrimitive]",
            date_to_primitive,
            1,
        )?;

        let ctor = new_native_function(heap, "Date", date_construct, 7)?;
        // `Date` is a constructor (ECMA-262 `IsConstructor`).
        heap.mark_native_constructor(date_construct);
        define_native(heap, ctor, "now", date_now, 0)?;
        define_native(heap, ctor, "parse", date_parse, 1)?;
        define_native(heap, ctor, "UTC", date_utc, 7)?;
        // `ctor.prototype = proto` ({writable:false, enumerable:false,
        // configurable:false}); `proto.constructor = ctor`.
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
        heap.define_property_str(proto, "constructor", Property::data(Value::Object(ctor)))?;
        heap.define_property_str(global, "Date", Property::data(Value::Object(ctor)))?;
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Vm;

    fn eval(src: &str) -> Value {
        Vm::new().eval(src).expect("eval failed")
    }

    /// Evaluate and return the resulting string. The `Vm` must stay alive
    /// while the `GcStr` handle is read: returning the bare `Value` and
    /// dereferencing after the heap is dropped is use-after-free.
    fn str_val(src: &str) -> String {
        let mut vm = Vm::new();
        let v = vm.eval(src).expect("eval failed");
        match v {
            Value::Str(s) => s.as_wtf8().to_string(),
            _ => panic!("expected string"),
        }
    }

    fn num(src: &str) -> f64 {
        match eval(src) {
            Value::Int(i) => f64::from(i),
            Value::Float(f) => f,
            v => panic!("expected number, got {v:?}"),
        }
    }

    fn is_nan_value(src: &str) -> bool {
        matches!(eval(src), Value::Float(f) if f.is_nan())
    }

    /// `true` when `src` throws an instance of `ctor` (`"TypeError"`,
    /// `"RangeError"`, ...). Uses an outer binding because try/catch
    /// completion values do not propagate (engine v1).
    fn throws_instance(src: &str, ctor: &str) -> bool {
        let probe = format!(
            "let __threw = false; try {{ {src}; }} catch (e) {{ __threw = e instanceof {ctor}; }} __threw;"
        );
        eval(&probe).eq(&Value::Bool(true))
    }

    fn throws_type_error(src: &str) -> bool {
        throws_instance(src, "TypeError")
    }

    #[test]
    fn date_math_round_trips() {
        // (civil date, weekday) spot checks through the Hinnant algorithms.
        for (y, m, d, wd) in [
            (1970i64, 1u32, 1u32, 4u32), // Thursday
            (2026, 9, 28, 1),            // Monday
            (2000, 2, 29, 2),            // leap day, Tuesday
            (1900, 1, 1, 1),             // not a leap year
            (1969, 12, 31, 3),
        ] {
            let days = days_from_civil(y, m as i64, d as i64);
            let (ry, rm, rd) = civil_from_days(days);
            assert_eq!((ry, rm, rd), (y, m, d));
            let t = make_date(days, 0.0);
            assert_eq!(split_fields(t).weekday, wd, "{y}-{m}-{d}");
        }
        // Pre-epoch negative time values split exactly.
        let f = split_fields(-1.0);
        assert_eq!((f.year, f.month, f.date), (1969, 11, 31));
        assert_eq!((f.hour, f.minute, f.second, f.ms), (23, 59, 59, 999));
    }

    #[test]
    fn parse_iso_subset() {
        assert_eq!(parse_date_time("1970-01-01T00:00:00.000Z"), 0.0);
        assert_eq!(parse_date_time("1970-01-01"), 0.0);
        assert_eq!(parse_date_time("  1970-01-01T00:00:00Z  "), 0.0);
        assert_eq!(parse_date_time("1969-12-31T24:00:00Z"), 0.0);
        // Numeric offsets shift the instant.
        assert_eq!(parse_date_time("1970-01-01T02:00:00+02:00"), 0.0);
        assert_eq!(
            parse_date_time("1970-01-01T00:00:00.000Z"),
            parse_date_time("1970-01-01T00:00:00.000+00:00")
        );
        // Rejections.
        for bad in [
            "",
            "garbage",
            "Sep 28 2026",
            "2026/09/28",
            "2026-13-01",
            "2026-02-30",
            "2026-09-28T25:00:00Z",
            "2026-09-28T12:60:00Z",
            "2026-09-28T12:00:00Ztrailing",
            "2026-09-28T24:00:01Z",
        ] {
            assert!(parse_date_time(bad).is_nan(), "{bad:?}");
        }
        // "2026-09-28T12:00:00" without a zone parses (local == UTC in v1).
        assert!(!parse_date_time("2026-09-28T12:00:00").is_nan());
    }

    #[test]
    fn constructor_basics() {
        assert_eq!(num("new Date(0).getTime()"), 0.0);
        assert_eq!(num("new Date('1970-01-01T00:00:00.000Z').getTime()"), 0.0);
        assert_eq!(
            num("new Date(2026, 8, 28).getTime()"),
            num("Date.UTC(2026, 8, 28)")
        );
        // Copy construction.
        assert_eq!(num("new Date(new Date(12345)).getTime()"), 12345.0);
        // Two-digit years map to 1900+.
        assert_eq!(num("new Date(99, 0, 1).getFullYear()"), 1999.0);
        assert_eq!(num("new Date(9, 0, 1).getFullYear()"), 1909.0);
        assert_eq!(num("new Date(100, 0, 1).getFullYear()"), 100.0);
        // Month overflow rolls into the year.
        assert_eq!(num("new Date(2026, 12, 1).getMonth()"), 0.0);
        assert_eq!(num("new Date(2026, 12, 1).getFullYear()"), 2027.0);
        // Invalid inputs.
        assert!(is_nan_value("new Date('nope').getTime()"));
        assert!(is_nan_value("new Date(NaN).getTime()"));
        assert!(is_nan_value("new Date(1e21).getTime()"));
        // Fractional month/date components truncate toward zero (spec
        // `ToIntegerOrInfinity`).
        assert_eq!(num("new Date(2026, 1.5).getMonth()"), 1.0);
        assert_eq!(num("new Date(2026, 1.9, 1).getMonth()"), 1.0);
        assert_eq!(num("new Date(2026, 0, 1.9).getDate()"), 1.0);
        // `new Date(2026, -1.5, 1)`: month truncates to -1, rolling back to
        // December 2025.
        assert_eq!(num("new Date(2026, -1.5, 1).getMonth()"), 11.0);
        assert_eq!(num("new Date(2026, -1.5, 1).getFullYear()"), 2025.0);
        // Fractional time components are `ToNumber`, not truncated: they
        // fold into `MakeTime` per spec (1.5 hours = 90 minutes).
        assert_eq!(num("new Date(2026, 0, 1, 1.5).getMinutes()"), 30.0);
        // Non-finite month/date still yield invalid dates.
        assert!(is_nan_value("new Date(2026, Infinity).getTime()"));
        assert!(is_nan_value("new Date(2026, 0, NaN).getTime()"));
        // Call form returns a string.
        assert_eq!(str_val("typeof Date()"), "string");
        // `Date.now()` is near the system clock.
        let before = now_ms();
        let t = num("Date.now()");
        let after = now_ms();
        assert!(
            (before..=after).contains(&t),
            "{t} not in [{before}, {after}]"
        );
    }

    #[test]
    fn getters() {
        let d = "new Date(Date.UTC(2026, 8, 28, 12, 34, 56, 789))";
        for (method, want) in [
            ("getFullYear", 2026.0),
            ("getMonth", 8.0),
            ("getDate", 28.0),
            ("getDay", 1.0),
            ("getHours", 12.0),
            ("getMinutes", 34.0),
            ("getSeconds", 56.0),
            ("getMilliseconds", 789.0),
        ] {
            assert_eq!(num(&format!("{d}.{method}()")), want, "{method}");
            let utc = method.replace("get", "getUTC");
            assert_eq!(num(&format!("{d}.{utc}()")), want, "{utc}");
        }
        // 1970-01-01 was a Thursday.
        assert_eq!(num("new Date(0).getUTCDay()"), 4.0);
        // v1: the environment is UTC.
        assert_eq!(num("new Date(0).getTimezoneOffset()"), 0.0);
        // `valueOf` agrees with `getTime`.
        assert_eq!(num("new Date(42).valueOf()"), 42.0);
        // Brand checks throw (via `Reflect.apply`; `Function.prototype.call`
        // on native functions is a pre-existing engine gap).
        assert!(throws_type_error(
            "Reflect.apply(Date.prototype.getTime, {}, [])"
        ));
        assert!(throws_type_error(
            "Reflect.apply(Date.prototype.setTime, 42, [1])"
        ));
        assert_eq!(
            num("Reflect.apply(Date.prototype.getTime, new Date(0), [])"),
            0.0
        );
    }

    #[test]
    fn setters() {
        assert_eq!(
            num("const d = new Date(0); d.setTime(12345); d.getTime()"),
            12345.0
        );
        assert_eq!(
            num(
                "const d = new Date(Date.UTC(2026, 0, 15, 10, 20, 30, 40)); d.setUTCFullYear(2027); d.getUTCFullYear()"
            ),
            2027.0
        );
        assert_eq!(
            num("const d = new Date(Date.UTC(2026, 0, 15)); d.setUTCMonth(5); d.getUTCMonth()"),
            5.0
        );
        assert_eq!(
            num("const d = new Date(Date.UTC(2026, 0, 15)); d.setUTCDate(20); d.getUTCDate()"),
            20.0
        );
        assert_eq!(
            str_val(
                "const d = new Date(Date.UTC(2026, 0, 15, 10, 20, 30, 40)); d.setUTCHours(1, 2, 3, 4); d.toISOString()"
            ),
            "2026-01-15T01:02:03.004Z"
        );
        // Setters return the new time value.
        assert_eq!(num("new Date(0).setTime(7)"), 7.0);
        // Fractional month/date components truncate toward zero (spec
        // `ToIntegerOrInfinity`); non-finite ones still yield NaN dates.
        assert_eq!(
            num("const d = new Date(Date.UTC(2026, 0, 15)); d.setUTCMonth(5.9); d.getUTCMonth()"),
            5.0
        );
        assert_eq!(
            num("const d = new Date(Date.UTC(2026, 0, 15)); d.setUTCDate(20.7); d.getUTCDate()"),
            20.0
        );
        assert!(is_nan_value(
            "const d = new Date(0); d.setUTCMonth(Infinity); d.getTime()"
        ));
        assert!(is_nan_value(
            "const d = new Date(0); d.setUTCDate(NaN); d.getTime()"
        ));
        // Fractional time components fold via `MakeTime` (per spec):
        // 1.5 hours becomes 90 minutes.
        assert_eq!(
            num("const d = new Date(Date.UTC(2026, 0, 15)); d.setUTCHours(1.5); d.getUTCMinutes()"),
            30.0
        );
    }

    #[test]
    fn setters_on_invalid_dates_stay_nan() {
        // Per spec, every component setter returns NaN and leaves the date
        // invalid when the current time value is NaN.
        for setter in [
            "setFullYear(2026)",
            "setUTCFullYear(2026)",
            "setMonth(5)",
            "setUTCMonth(5)",
            "setDate(15)",
            "setUTCDate(15)",
            "setHours(1)",
            "setUTCHours(1)",
            "setMinutes(1)",
            "setUTCMinutes(1)",
            "setSeconds(1)",
            "setUTCSeconds(1)",
            "setMilliseconds(1)",
            "setUTCMilliseconds(1)",
        ] {
            let retained = format!("const d = new Date(NaN); d.{setter}; d.getTime()");
            assert!(is_nan_value(&retained), "{setter} revived the date");
            let returned = format!("new Date(NaN).{setter}");
            assert!(is_nan_value(&returned), "{setter} did not return NaN");
        }
        // `setTime` is the exception: it replaces the absolute time value,
        // so it revives an invalid date.
        assert_eq!(num("new Date(NaN).setTime(0)"), 0.0);
        assert_eq!(
            num("const d = new Date(NaN); d.setTime(0); d.getTime()"),
            0.0
        );
        // A NaN setter argument on a *valid* date still invalidates it.
        assert!(is_nan_value(
            "const d = new Date(0); d.setUTCFullYear(NaN); d.getTime()"
        ));
    }

    #[test]
    fn single_arg_uses_default_hint() {
        // `ToPrimitive(value)` with the default hint: ordinary objects try
        // `valueOf` first (as with the number hint) ...
        assert_eq!(
            num("new Date({valueOf: () => 5, toString: () => '2024-01-01'}).getTime()"),
            5.0
        );
        // ... and the hint handed to `@@toPrimitive` is `"default"`.
        assert_eq!(
            str_val(
                "let __hint = ''; new Date({[Symbol.toPrimitive](h) { __hint = h; return 42; }}); __hint;"
            ),
            "default"
        );
        assert_eq!(
            num("new Date({[Symbol.toPrimitive](h) { return 42; }}).getTime()"),
            42.0
        );
    }

    #[test]
    fn string_forms() {
        assert_eq!(
            str_val("new Date(0).toISOString()"),
            "1970-01-01T00:00:00.000Z"
        );
        assert_eq!(
            str_val("new Date(Date.UTC(2026, 8, 28, 12, 34, 56, 789)).toISOString()"),
            "2026-09-28T12:34:56.789Z"
        );
        assert_eq!(
            str_val("new Date(0).toString()"),
            "Thu Jan 01 1970 00:00:00 GMT+0000"
        );
        assert_eq!(str_val("new Date(0).toDateString()"), "Thu Jan 01 1970");
        assert_eq!(str_val("new Date(0).toTimeString()"), "00:00:00 GMT+0000");
        assert_eq!(str_val("new Date(0).toLocaleDateString()"), "1/1/1970");
        assert_eq!(str_val("new Date(0).toLocaleTimeString()"), "12:00:00 AM");
        assert_eq!(str_val("new Date('garbage').toString()"), "Invalid Date");
        assert_eq!(str_val("new Date(0).toJSON()"), "1970-01-01T00:00:00.000Z");
        // Non-finite time values serialize as the `null` value (per spec).
        assert!(eval("new Date(NaN).toJSON()").eq(&Value::Null));
        // `toISOString` throws on invalid dates.
        assert!(throws_instance("new Date(NaN).toISOString()", "RangeError"));
        // `[object Date]` branding (via `Reflect.apply`; `.call` on native
        // functions is a pre-existing engine gap).
        assert_eq!(
            str_val("Reflect.apply(Object.prototype.toString, new Date(0), [])"),
            "[object Date]"
        );
    }

    #[test]
    fn primitive_conversion() {
        // Unary plus uses the number hint -> valueOf.
        assert_eq!(num("+new Date(42)"), 42.0);
        // String concatenation uses the default hint -> toString first.
        assert_eq!(
            str_val("'' + new Date(0)"),
            "Thu Jan 01 1970 00:00:00 GMT+0000"
        );
        // Explicit hint routing through @@toPrimitive.
        assert_eq!(
            str_val("new Date(0)[Symbol.toPrimitive]('string')"),
            "Thu Jan 01 1970 00:00:00 GMT+0000"
        );
        assert_eq!(num("new Date(42)[Symbol.toPrimitive]('number')"), 42.0);
    }

    #[test]
    fn statics() {
        assert_eq!(num("Date.UTC(1970, 0, 1)"), 0.0);
        assert_eq!(
            num("Date.parse('2026-09-28T00:00:00.000Z')"),
            num("Date.UTC(2026, 8, 28)")
        );
        assert!(is_nan_value("Date.parse('not a date')"));
        assert_eq!(
            num("Date.UTC(2026, 8, 28)"),
            num("new Date(2026, 8, 28).getTime()")
        );
        assert_eq!(str_val("typeof Date.now"), "function");
        assert_eq!(num("Date.length"), 7.0);
    }

    // Note: the install path must survive a collection on every allocation
    // (threshold 1): `%Date.prototype%` is allocated before the
    // constructor, so without the temp-root it would be freed mid-install
    // and the constructor's `prototype` link would dangle.
    // (Covered by `install_paths_survive_threshold_1_collections` in
    // `tests/vm.rs`.)
}
