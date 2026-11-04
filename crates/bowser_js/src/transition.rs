//! A clock-at-read CSS Transitions engine for discrete properties
//! (w80). No timers, no frame callbacks: progress is computed from a
//! monotonic clock at `getComputedStyle` time, so a harness that sets
//! a negative delay and reads synchronously (WPT's
//! `interpolation-testcommon.js`, which drives progress with
//! `duration: 100s; delay: -50s` plus an easing function) observes
//! the value at exactly the progress it asked for. A timer-only
//! engine cannot satisfy that pattern: the read happens ~0ms after
//! the set, so only the delay math puts progress at 0.5.
//!
//! Scope: discrete properties only, and in this wave only `overlay`
//! is routed through [`apply_transitions`] (see `DISCRETE_PROPS`).
//! Numeric interpolation is future work; the flip rules below are
//! the full discrete behavior.
//!
//! `overlay` rules (CSS Position 4, as pinned by
//! `overlay-interpolation.html`):
//! * An explicit `transition-property: overlay` transitions it, even
//!   with the default `transition-behavior: normal`.
//!   `transition-property: all` with the default `normal` behavior
//!   does NOT transition it (the `all` list excludes it), so with
//!   plain `all` the value flips to the target immediately. With
//!   `all` plus `transition-behavior: allow-discrete` it DOES
//!   transition, at the `all` layer's parameters — the
//!   interpolation file overrides only the plain-`all` method, so
//!   the all+allow-discrete method runs on the default asymmetric
//!   expectations.
//! * The flip is asymmetric: `none -> auto` flips as soon as the
//!   eased progress is above 0 (to-auto flips early, so the element
//!   is in effect for the whole entry), while `auto -> none` holds
//!   `auto` until the eased progress reaches 1 (from-auto flips
//!   late, so the element stays in effect for the whole exit).
//!   Progress here is EASED progress (after the timing function),
//!   as the WPT easing cases require: at eased 1.5 an exit already
//!   reads `none`, and at eased -1 / 0 an entry still reads `none`
//!   (raw progress is 0.5 at every harness read, so raw-progress
//!   keying would fail both). Consequently a step-start exit flips
//!   at the start — its easing jumps to 1 by definition — and an
//!   overshooting bezier flips once eased crosses 1; that is the
//!   timing function operating, not a hold violation (see
//!   `overlay_end_flip_is_keyed_on_eased_progress`).
//! * Any other discrete property follows the generic discrete rule
//!   (flip when the eased progress reaches 0.5) and additionally
//!   requires `transition-behavior: allow-discrete`; without it the
//!   value flips immediately (no transition).
//!
//! State lives in a thread-local registry keyed by a caller-supplied
//! element key: the parse shim passes `parse:{host}:{arena}`, and
//! the production computed-style path (`n_get_computed_style`)
//! passes `prod:{NodeId:?}` — the arena index plus the
//! process-unique document tag. The arena is append-only and tags
//! never repeat, so a key never aliases another element, and the
//! two surfaces never share state for the same element (a shim
//! element upgraded to production re-seeds on its first
//! production read, exactly like a fresh element). Each read
//! records the cascaded value it saw; a change against the previous
//! read, with transition parameters that apply, starts a transition
//! from the previously displayed value. A transition whose target
//! changes mid-flight retargets from the currently displayed value.
//! When no transition applies, the cascaded value is returned as-is
//! and any active transition is cancelled.
//!
//! Limitations, stated not hidden:
//! * No pre-change read, no transition: [`apply_transitions`]
//!   starts a transition only when the new cascaded value differs
//!   from the recorded `last_cascaded` value, so a fresh element
//!   whose first read already shows the target never transitions
//!   (there is no recorded before-value to start from). Seeding
//!   before-values at style-change time would need style-mutation
//!   hooks and is future work; every harness pattern reads first,
//!   so WPT is unaffected.
//! * No element-destruction hook: entries for removed elements
//!   linger until the registry cap (4096) evicts idle entries.
//!   Eviction is behaviorally safe — the next read for an evicted
//!   element simply re-seeds instead of starting a transition, the
//!   same as a fresh element — and entries with a running
//!   transition are never evicted, so the cap bounds idle memory
//!   (one small map per read element) rather than live transitions.
//!   The hook itself is read-only against the document: it mutates
//!   only the calling read's props snapshot, so there is no dirty
//!   or repaint interaction.

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;

/// Discrete properties routed through the engine. Only `overlay`
/// today: it is the only discrete property with computed slots,
/// transition plumbing, and WPT coverage in this wave.
const DISCRETE_PROPS: &[&str] = &["overlay"];

/// A parsed `<timing-function>` for evaluation. Named functions map
/// to their cubic-bezier equivalents (CSS Easing Functions 2);
/// `step-start` / `step-end` map to one-step `steps()`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Timing {
    Linear,
    CubicBezier { x1: f64, y1: f64, x2: f64, y2: f64 },
    Steps { count: u32, position: StepPosition },
}

/// The `<step-position>` of a `steps()` timing function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepPosition {
    Start,
    End,
    JumpNone,
    JumpBoth,
}

/// Parse a serialized `<time>` (`2s`, `100ms`, `0`) to seconds.
/// `None` for anything else.
pub fn parse_time(s: &str) -> Option<f64> {
    let s = s.trim();
    if let Some(ms) = s.strip_suffix("ms") {
        ms.trim()
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .map(|v| v / 1000.0)
    } else if let Some(sec) = s.strip_suffix('s') {
        sec.trim().parse::<f64>().ok().filter(|v| v.is_finite())
    } else if s == "0" {
        Some(0.0)
    } else {
        None
    }
}

/// Parse a serialized `<timing-function>` to an evaluable form.
/// `None` for anything else.
pub fn parse_timing(s: &str) -> Option<Timing> {
    let s = s.trim().to_ascii_lowercase();
    match s.as_str() {
        "linear" => Some(Timing::Linear),
        "ease" => Some(Timing::CubicBezier {
            x1: 0.25,
            y1: 0.1,
            x2: 0.25,
            y2: 1.0,
        }),
        "ease-in" => Some(Timing::CubicBezier {
            x1: 0.42,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
        }),
        "ease-out" => Some(Timing::CubicBezier {
            x1: 0.0,
            y1: 0.0,
            x2: 0.58,
            y2: 1.0,
        }),
        "ease-in-out" => Some(Timing::CubicBezier {
            x1: 0.42,
            y1: 0.0,
            x2: 0.58,
            y2: 1.0,
        }),
        "step-start" => Some(Timing::Steps {
            count: 1,
            position: StepPosition::Start,
        }),
        "step-end" => Some(Timing::Steps {
            count: 1,
            position: StepPosition::End,
        }),
        _ => {
            if let Some(body) = s.strip_prefix("steps(").and_then(|b| b.strip_suffix(')')) {
                let parts: Vec<&str> = body.split(',').map(str::trim).collect();
                if parts.len() > 2 {
                    return None;
                }
                let count: u32 = parts[0].parse().ok()?;
                if count < 1 {
                    return None;
                }
                let position = match parts.get(1).copied().unwrap_or("end") {
                    "start" | "jump-start" => StepPosition::Start,
                    "end" | "jump-end" => StepPosition::End,
                    "jump-none" if count >= 2 => StepPosition::JumpNone,
                    "jump-both" => StepPosition::JumpBoth,
                    _ => return None,
                };
                Some(Timing::Steps { count, position })
            } else if let Some(body) = s
                .strip_prefix("cubic-bezier(")
                .and_then(|b| b.strip_suffix(')'))
            {
                let nums: Vec<f64> = body
                    .split(',')
                    .map(|p| p.trim().parse::<f64>())
                    .collect::<Result<_, _>>()
                    .ok()?;
                if nums.len() != 4
                    || !nums.iter().all(|v| v.is_finite())
                    || !(0.0..=1.0).contains(&nums[0])
                    || !(0.0..=1.0).contains(&nums[2])
                {
                    return None;
                }
                Some(Timing::CubicBezier {
                    x1: nums[0],
                    y1: nums[1],
                    x2: nums[2],
                    y2: nums[3],
                })
            } else {
                None
            }
        }
    }
}

/// Evaluate a cubic-bezier easing function at progress `x` (in
/// `[0, 1]`; values outside clamp to the endpoints). Solves `x(t) =
/// x` for `t` with Newton iterations plus a bisection fallback
/// (the x-curve is monotonic for `x1, x2 in [0, 1]`), then returns
/// `y(t)`.
pub fn cubic_bezier(x1: f64, y1: f64, x2: f64, y2: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bez = |t: f64, a: f64, b: f64| -> f64 {
        let u = 1.0 - t;
        3.0 * u * u * t * a + 3.0 * u * t * t * b + t * t * t
    };
    let dx = |t: f64| -> f64 {
        let u = 1.0 - t;
        3.0 * u * u * x1 + 6.0 * u * t * (x2 - x1) + 3.0 * t * t * (1.0 - x2)
    };
    let mut t = x;
    for _ in 0..8 {
        let err = bez(t, x1, x2) - x;
        if err.abs() < 1e-7 {
            break;
        }
        let d = dx(t);
        if d.abs() < 1e-7 {
            break;
        }
        t -= err / d;
        if !(0.0..=1.0).contains(&t) {
            break;
        }
    }
    if !(0.0..=1.0).contains(&t) || (bez(t, x1, x2) - x).abs() > 1e-4 {
        let (mut lo, mut hi) = (0.0, 1.0);
        t = x;
        for _ in 0..24 {
            t = (lo + hi) / 2.0;
            if bez(t, x1, x2) < x {
                lo = t;
            } else {
                hi = t;
            }
        }
    }
    bez(t, y1, y2)
}

/// Evaluate a `steps()` timing function at progress `x`.
fn steps_eval(count: u32, position: StepPosition, x: f64) -> f64 {
    let n = count as f64;
    match position {
        StepPosition::End => {
            if x >= 1.0 {
                1.0
            } else if x <= 0.0 {
                0.0
            } else {
                (x * n).floor() / n
            }
        }
        StepPosition::Start => {
            if x <= 0.0 {
                0.0
            } else if x >= 1.0 {
                1.0
            } else {
                ((x * n).ceil() / n).min(1.0)
            }
        }
        StepPosition::JumpNone => {
            if count < 2 || x <= 0.0 {
                0.0
            } else if x >= 1.0 {
                1.0
            } else {
                (x * n).floor() / (n - 1.0)
            }
        }
        StepPosition::JumpBoth => {
            if x <= 0.0 {
                0.0
            } else if x >= 1.0 {
                1.0
            } else {
                ((x * n).floor() + 1.0) / (n + 1.0)
            }
        }
    }
}

/// Evaluate a timing function at progress `x`.
pub fn eval_timing(timing: Timing, x: f64) -> f64 {
    match timing {
        Timing::Linear => x,
        Timing::CubicBezier { x1, y1, x2, y2 } => cubic_bezier(x1, y1, x2, y2, x),
        Timing::Steps { count, position } => steps_eval(count, position, x),
    }
}

/// Split a computed transition longhand list on top-level commas
/// (commas inside `()` — `steps()`, `cubic-bezier()` — do not
/// split), trimming each layer.
fn split_layers(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' => {
                depth += 1;
                cur.push(c);
            }
            ')' => {
                depth -= 1;
                cur.push(c);
            }
            ',' if depth <= 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    out.push(cur.trim().to_string());
    out
}

/// The transition parameters that apply to one property, resolved
/// from the computed `transition-*` longhands (CSS list repetition:
/// a shorter list cycles to the property's layer index).
struct ResolvedParams {
    duration_s: f64,
    delay_s: f64,
    timing: Timing,
}

/// Resolve the parameters for `prop`, or `None` when no transition
/// applies to it (property not listed, `all` with `normal`
/// behavior, missing `allow-discrete` for other discrete
/// properties, or a zero duration).
///
/// w81 decision (all+X ordering): the LAST matching item wins,
/// not the explicit listing. CSS Transitions 1 §2.1: "If a property
/// is specified multiple times in the value of
/// transition-property (either on its own, via a shorthand that
/// contains it, or via the all value), then the transition that
/// starts uses the duration, delay, and timing function at the
/// index corresponding to the *last* item in the value of
/// transition-property that calls for animating that property",
/// and "the all value is just like a shorthand that covers all
/// properties". So `all, overlay` resolves at the `overlay`
/// layer, while `overlay, all` (with `allow-discrete`) resolves
/// at the `all` layer. This replaced the w80 rule under which an
/// explicit listing always won regardless of order. One
/// qualification: an item only counts if it "calls for animating"
/// the property, so `all` under the default `normal` behavior
/// does not count for a discrete property (it is excluded there,
/// as before), and `overlay, all` with `normal` behavior still
/// resolves at the explicit layer. A repeated literal listing
/// likewise resolves at its last occurrence. This engine matches
/// literal identifiers only; shorthand expansion is out of scope,
/// so the shorthand arm of the spec rule does not apply here.
fn resolve_params(props: &HashMap<String, String>, prop: &str) -> Option<ResolvedParams> {
    let property_list = split_layers(props.get("transition-property")?);
    let allow_discrete =
        props.get("transition-behavior").map(String::as_str) == Some("allow-discrete");
    // w81 (see the doc comment above): last matching item wins.
    // `all` counts as a match only with `allow-discrete` for a
    // discrete property; with `normal` behavior it does not call
    // for animating it, so it is excluded rather than merely
    // losing to an explicit listing.
    let explicit_idx = property_list.iter().rposition(|p| p == prop);
    let all_idx = if allow_discrete {
        property_list.iter().rposition(|p| p == "all")
    } else {
        None
    };
    let idx = match (explicit_idx, all_idx) {
        (Some(explicit), Some(all)) => explicit.max(all),
        (Some(explicit), None) => explicit,
        (None, Some(all)) => all,
        (None, None) => return None,
    };
    if prop != "overlay" && !allow_discrete {
        // An explicitly listed non-`overlay` discrete property
        // still needs `allow-discrete`; `overlay` is exempt. (Via
        // the `all` arm above, `allow_discrete` is true by
        // construction, so this only fires for explicit listings.)
        return None;
    }
    let at = |name: &str| -> Option<String> {
        let layers = split_layers(props.get(name)?);
        if layers.is_empty() {
            return None;
        }
        Some(layers[idx % layers.len()].clone())
    };
    let duration_s = parse_time(&at("transition-duration")?)?;
    if duration_s <= 0.0 {
        return None;
    }
    let delay_s = at("transition-delay")
        .and_then(|s| parse_time(&s))
        .unwrap_or(0.0);
    let timing = at("transition-timing-function")
        .and_then(|s| parse_timing(&s))
        .unwrap_or(Timing::CubicBezier {
            x1: 0.25,
            y1: 0.1,
            x2: 0.25,
            y2: 1.0,
        });
    Some(ResolvedParams {
        duration_s,
        delay_s,
        timing,
    })
}

/// One running discrete transition.
struct ActiveTransition {
    from: String,
    to: String,
    start: Instant,
    duration_s: f64,
    delay_s: f64,
    timing: Timing,
    property: String,
}

/// Evaluate a running transition at `now`: the displayed value plus
/// whether it has finished (raw progress `>= 1`).
fn eval_transition(active: &ActiveTransition, now: Instant) -> (String, bool) {
    let elapsed = now.duration_since(active.start).as_secs_f64();
    eval_discrete(
        &active.from,
        &active.to,
        &active.property,
        active.duration_s,
        active.delay_s,
        active.timing,
        elapsed,
    )
}

/// Pure discrete evaluation at `elapsed` seconds since the
/// transition started (split out so tests can drive arbitrary
/// progress, including negative elapsed times, without `Instant`
/// arithmetic).
fn eval_discrete(
    from: &str,
    to: &str,
    property: &str,
    duration_s: f64,
    delay_s: f64,
    timing: Timing,
    elapsed: f64,
) -> (String, bool) {
    let progress = (elapsed - delay_s) / duration_s;
    if progress < 0.0 {
        return (from.to_string(), false);
    }
    if progress >= 1.0 {
        return (to.to_string(), true);
    }
    let eased = eval_timing(timing, progress);
    let value = if property == "overlay" {
        match (from, to) {
            ("auto", "none") => {
                if eased >= 1.0 {
                    to.to_string()
                } else {
                    from.to_string()
                }
            }
            ("none", "auto") => {
                if eased > 0.0 {
                    to.to_string()
                } else {
                    from.to_string()
                }
            }
            _ => {
                if eased >= 0.5 {
                    to.to_string()
                } else {
                    from.to_string()
                }
            }
        }
    } else if eased >= 0.5 {
        to.to_string()
    } else {
        from.to_string()
    };
    (value, false)
}

/// Per-element transition state.
struct ElementState {
    /// The cascaded value seen on the previous read, per property.
    last_cascaded: HashMap<String, String>,
    /// The running transition, per property.
    active: HashMap<String, ActiveTransition>,
}

thread_local! {
    static STATES: RefCell<HashMap<String, ElementState>> = RefCell::new(HashMap::new());
}

/// Apply running transitions to a computed-props map, in place.
/// `key` identifies the element across reads (callers pass a stable
/// id: parse host + arena index for the shim, `prod:{NodeId:?}` for
/// the production computed-style path). Only properties in
/// [`DISCRETE_PROPS`] are considered; everything else passes through
/// untouched. Must be called on every `getComputedStyle` read for
/// the element: the first read establishes the before-value, and a
/// later read whose cascaded value changed starts (or retargets) the
/// transition.
pub fn apply_transitions(key: &str, props: &mut HashMap<String, String>) {
    let now = Instant::now();
    STATES.with(|states| {
        let mut states = states.borrow_mut();
        // Bound the registry: keys are never explicitly removed
        // (neither surface has an element-destruction hook; see the
        // module doc for the production lifecycle story), so once
        // it grows past a cap, drop entries with no running
        // transition. Dropping a key is behaviorally safe: the next
        // read for it simply re-establishes the before-value instead
        // of starting a transition, which is also what a fresh
        // element sees. Entries with a running transition are kept,
        // so the cap bounds idle growth, not live transitions.
        if states.len() >= 4096 {
            states.retain(|_, s| !s.active.is_empty());
        }
        let state = states
            .entry(key.to_string())
            .or_insert_with(|| ElementState {
                last_cascaded: HashMap::new(),
                active: HashMap::new(),
            });
        for prop in DISCRETE_PROPS {
            let Some(cascaded) = props.get(*prop).cloned() else {
                continue;
            };
            let mut finished = false;
            if let Some(active) = state.active.get(*prop) {
                if cascaded == active.to {
                    let (value, done) = eval_transition(active, now);
                    finished = done;
                    props.insert(prop.to_string(), value);
                } else {
                    // Retarget: the displayed value right now becomes
                    // the new transition's start value.
                    let (displayed, _) = eval_transition(active, now);
                    match resolve_params(props, prop) {
                        Some(params) if cascaded != displayed => {
                            let new_active = ActiveTransition {
                                from: displayed,
                                to: cascaded.clone(),
                                start: now,
                                duration_s: params.duration_s,
                                delay_s: params.delay_s,
                                timing: params.timing,
                                property: prop.to_string(),
                            };
                            let (value, done) = eval_transition(&new_active, now);
                            finished = done;
                            state.active.insert(prop.to_string(), new_active);
                            props.insert(prop.to_string(), value);
                        }
                        _ => {
                            state.active.remove(*prop);
                            props.insert(prop.to_string(), cascaded.clone());
                        }
                    }
                }
            } else if let Some(prev) = state.last_cascaded.get(*prop).cloned()
                && prev != cascaded
            {
                match resolve_params(props, prop) {
                    Some(params) => {
                        let new_active = ActiveTransition {
                            from: prev,
                            to: cascaded.clone(),
                            start: now,
                            duration_s: params.duration_s,
                            delay_s: params.delay_s,
                            timing: params.timing,
                            property: prop.to_string(),
                        };
                        let (value, done) = eval_transition(&new_active, now);
                        finished = done;
                        state.active.insert(prop.to_string(), new_active);
                        props.insert(prop.to_string(), value);
                    }
                    None => {
                        props.insert(prop.to_string(), cascaded.clone());
                    }
                }
            }
            if finished {
                state.active.remove(*prop);
            }
            state.last_cascaded.insert(prop.to_string(), cascaded);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn easing_evaluates_harness_curves() {
        // The harness's `createEasing(y)`: bezier(0, b, 1, b) must
        // return y at x = 0.5, for b = (8y - 1) / 6.
        for y in [-1.0, 0.0, 0.1, 0.5, 0.9, 1.0, 1.5] {
            let b = (8.0 * y - 1.0) / 6.0;
            let got = cubic_bezier(0.0, b, 1.0, b, 0.5);
            assert!((got - y).abs() < 1e-6, "bezier at 0.5 for y={y}: got {got}");
        }
        assert_eq!(eval_timing(Timing::Linear, 0.5), 0.5);
        assert_eq!(
            eval_timing(
                Timing::Steps {
                    count: 1,
                    position: StepPosition::End,
                },
                0.5
            ),
            0.0
        );
        assert_eq!(
            eval_timing(
                Timing::Steps {
                    count: 1,
                    position: StepPosition::Start,
                },
                0.5
            ),
            1.0
        );
        assert_eq!(parse_timing("step-end"), parse_timing("steps(1, end)"));
        assert_eq!(parse_timing("linear"), Some(Timing::Linear));
        assert_eq!(parse_time("100ms"), Some(0.1));
        assert_eq!(parse_time("60s"), Some(60.0));
        assert_eq!(parse_time("0"), Some(0.0));
        assert_eq!(parse_time("2px"), None);
    }

    #[test]
    fn overlay_flip_rules_are_asymmetric() {
        // Progress p at duration 100s / delay -50s means elapsed =
        // p * 100 - 50 seconds.
        let at = |from: &str, to: &str, progress: f64| {
            eval_discrete(
                from,
                to,
                "overlay",
                100.0,
                -50.0,
                Timing::Linear,
                progress * 100.0 - 50.0,
            )
            .0
        };
        // auto -> none holds auto until progress 1.
        assert_eq!(at("auto", "none", 0.0), "auto");
        assert_eq!(at("auto", "none", 0.1), "auto");
        assert_eq!(at("auto", "none", 0.9), "auto");
        assert_eq!(at("auto", "none", 1.0), "none");
        // none -> auto flips as soon as progress > 0.
        assert_eq!(at("none", "auto", 0.0), "none");
        assert_eq!(at("none", "auto", 0.1), "auto");
        assert_eq!(at("none", "auto", 1.0), "auto");
        // Negative progress (before a positive delay elapses) holds
        // `from`: delay 50s, elapsed 0.
        assert_eq!(
            eval_discrete("none", "auto", "overlay", 100.0, 50.0, Timing::Linear, 0.0).0,
            "none"
        );
    }

    #[test]
    fn transition_params_exclude_all_for_overlay() {
        let mut p = props(&[
            ("transition-property", "all"),
            ("transition-duration", "100s"),
            ("transition-delay", "-50s"),
            ("transition-timing-function", "linear"),
        ]);
        assert!(resolve_params(&p, "overlay").is_none());
        p.insert(
            "transition-property".to_string(),
            "opacity, overlay".to_string(),
        );
        p.insert("transition-duration".to_string(), "1s, 100s".to_string());
        let params = resolve_params(&p, "overlay").expect("explicit overlay");
        assert_eq!(params.duration_s, 100.0);
        // Zero duration never transitions.
        p.insert("transition-duration".to_string(), "0s".to_string());
        assert!(resolve_params(&p, "overlay").is_none());
    }

    #[test]
    fn transition_all_allow_discrete_matches_overlay_at_layer_zero() {
        // w80fix (F-TRANSITION, R1-M3): `transition-property: all`
        // plus `transition-behavior: allow-discrete` transitions
        // `overlay` (layer index 0 for the single-layer shape);
        // `all` with the default `normal` behavior still excludes it.
        let mut p = props(&[
            ("transition-property", "all"),
            ("transition-duration", "100s"),
            ("transition-delay", "-50s"),
            ("transition-timing-function", "linear"),
            ("transition-behavior", "allow-discrete"),
        ]);
        let params = resolve_params(&p, "overlay").expect("all + allow-discrete");
        assert_eq!(params.duration_s, 100.0);
        assert_eq!(params.delay_s, -50.0);
        p.insert("transition-behavior".to_string(), "normal".to_string());
        assert!(resolve_params(&p, "overlay").is_none());
        // A multi-layer list resolves through the `all` layer's own
        // params, not layer 0: `opacity, all` + allow-discrete uses
        // the second layer's duration for `overlay`.
        let p = props(&[
            ("transition-property", "opacity, all"),
            ("transition-duration", "1s, 100s"),
            ("transition-delay", "0s, -50s"),
            ("transition-timing-function", "linear, linear"),
            ("transition-behavior", "allow-discrete"),
        ]);
        let params = resolve_params(&p, "overlay").expect("all layer");
        assert_eq!(params.duration_s, 100.0);
        assert_eq!(params.delay_s, -50.0);
    }

    #[test]
    fn transition_all_x_last_matching_item_wins() {
        // w81 decision, CSS Transitions 1 §2.1 (last item that
        // calls for animating the property wins; `all` is like a
        // shorthand covering all properties). Distinct durations
        // (and delays/timings) per layer prove which layer's
        // parameters `overlay` takes in each order.
        // all-first: the explicit layer is last, so it wins.
        let p = props(&[
            ("transition-property", "all, overlay"),
            ("transition-duration", "1s, 100s"),
            ("transition-delay", "0s, -50s"),
            ("transition-timing-function", "ease, linear"),
            ("transition-behavior", "allow-discrete"),
        ]);
        let params = resolve_params(&p, "overlay").expect("all-first");
        assert_eq!(params.duration_s, 100.0);
        assert_eq!(params.delay_s, -50.0);
        assert_eq!(params.timing, Timing::Linear);
        // all-last: the `all` layer is last, so IT wins. This is
        // the case the w80 explicit-always-wins rule got wrong
        // (it returned the explicit layer's 100s here).
        let p = props(&[
            ("transition-property", "overlay, all"),
            ("transition-duration", "100s, 1s"),
            ("transition-delay", "-50s, 0s"),
            ("transition-timing-function", "linear, ease"),
            ("transition-behavior", "allow-discrete"),
        ]);
        let params = resolve_params(&p, "overlay").expect("all-last");
        assert_eq!(params.duration_s, 1.0);
        assert_eq!(params.delay_s, 0.0);
        assert_eq!(params.timing, parse_timing("ease").unwrap());
        // all-last with the durations swapped: still the `all`
        // layer (100s), proving the rule picks by position, not
        // by min/max duration.
        let p = props(&[
            ("transition-property", "overlay, all"),
            ("transition-duration", "1s, 100s"),
            ("transition-delay", "0s, -50s"),
            ("transition-timing-function", "linear, linear"),
            ("transition-behavior", "allow-discrete"),
        ]);
        let params = resolve_params(&p, "overlay").expect("all-last swapped");
        assert_eq!(params.duration_s, 100.0);
        assert_eq!(params.delay_s, -50.0);
        // Multi-layer: a later non-matching item does not disturb
        // the last MATCHING item. `overlay`'s last match is the
        // middle `all` layer (7s/3s), not the trailing `opacity`.
        let p = props(&[
            ("transition-property", "overlay, all, opacity"),
            ("transition-duration", "100s, 7s, 1s"),
            ("transition-delay", "0s, 3s, 0s"),
            ("transition-timing-function", "linear, linear, linear"),
            ("transition-behavior", "allow-discrete"),
        ]);
        let params = resolve_params(&p, "overlay").expect("multi-layer all middle");
        assert_eq!(params.duration_s, 7.0);
        assert_eq!(params.delay_s, 3.0);
        // Gating preserved: with `normal` behavior `all` does not
        // call for animating a discrete property, so `overlay,
        // all` still resolves at the explicit layer (100s), not
        // at the raw-last `all` layer (1s).
        let p = props(&[
            ("transition-property", "overlay, all"),
            ("transition-duration", "100s, 1s"),
            ("transition-delay", "-50s, 0s"),
            ("transition-timing-function", "linear, linear"),
        ]);
        let params = resolve_params(&p, "overlay").expect("all-last normal");
        assert_eq!(params.duration_s, 100.0);
        assert_eq!(params.delay_s, -50.0);
        // Same last-wins rule for a repeated literal listing.
        let p = props(&[
            ("transition-property", "overlay, overlay"),
            ("transition-duration", "1s, 100s"),
            ("transition-delay", "0s, -50s"),
            ("transition-timing-function", "linear, linear"),
        ]);
        let params = resolve_params(&p, "overlay").expect("repeated explicit");
        assert_eq!(params.duration_s, 100.0);
        assert_eq!(params.delay_s, -50.0);
    }

    #[test]
    fn multi_layer_serialized_longhands_resolve_per_layer() {
        // w80fix (F-TRANSITION, R1-M2): the engine consumes the
        // comma-joined computed serialization, so a two-layer
        // declaration resolves `overlay` (layer 1) to the second
        // layer's duration/delay, not a mangled space-joined token.
        let key = "w80-test-overlay-multilayer";
        let mut p = props(&[
            ("overlay", "auto"),
            ("transition-property", "none"),
            ("transition-duration", "0s"),
            ("transition-delay", "0s"),
            ("transition-timing-function", "ease"),
        ]);
        apply_transitions(key, &mut p);
        assert_eq!(p["overlay"], "auto");
        p.insert("overlay".to_string(), "none".to_string());
        p.insert(
            "transition-property".to_string(),
            "opacity, overlay".to_string(),
        );
        p.insert("transition-duration".to_string(), "1s, 100s".to_string());
        p.insert("transition-delay".to_string(), "0s, -50s".to_string());
        p.insert(
            "transition-timing-function".to_string(),
            "linear, linear".to_string(),
        );
        apply_transitions(key, &mut p);
        // Exit at progress 0.5 (linear) holds `auto`.
        assert_eq!(p["overlay"], "auto");
    }

    #[test]
    fn overlay_end_flip_is_keyed_on_eased_progress() {
        // w80fix (F-TRANSITION, R3-L4): the asymmetric end-flip is
        // deliberately keyed on EASED progress, not raw progress or
        // the finished flag. The WPT easing cases pin this: at
        // eased 1.5 an exit must already read `none`, and at eased
        // -1 / 0 an entry must still read `none` — raw-progress
        // keying (raw is 0.5 at every harness read) would fail
        // both. Consequences, pinned here rather than changed:
        // step-start exits flip as soon as raw progress is above 0
        // (the timing function jumps to 1 at the start, by
        // definition), and an overshooting bezier flips once its
        // eased value crosses 1. Step-end exits hold until raw 1.
        let at = |from: &str, to: &str, timing: Timing, elapsed: f64| {
            eval_discrete(from, to, "overlay", 100.0, 0.0, timing, elapsed).0
        };
        let step_start = Timing::Steps {
            count: 1,
            position: StepPosition::Start,
        };
        let step_end = Timing::Steps {
            count: 1,
            position: StepPosition::End,
        };
        // WPT easing points (bezier mapping 0.5 -> at).
        let bezier_at = |at: f64| {
            let b = (8.0 * at - 1.0) / 6.0;
            Timing::CubicBezier {
                x1: 0.0,
                y1: b,
                x2: 1.0,
                y2: b,
            }
        };
        assert_eq!(at("auto", "none", bezier_at(1.5), 50.0), "none");
        assert_eq!(at("none", "auto", bezier_at(-1.0), 50.0), "none");
        assert_eq!(at("none", "auto", bezier_at(0.0), 50.0), "none");
        // Step-start: entry and exit both jump at the start.
        assert_eq!(at("none", "auto", step_start, 10.0), "auto");
        assert_eq!(at("auto", "none", step_start, 10.0), "none");
        // Step-end: exit holds until the raw end.
        assert_eq!(at("auto", "none", step_end, 10.0), "auto");
        assert_eq!(at("auto", "none", step_end, 99.0), "auto");
        // Overshooting bezier (y > 1 mid-flight) flips the exit
        // once eased crosses 1, before raw progress reaches 1.
        let overshoot = Timing::CubicBezier {
            x1: 0.0,
            y1: 2.0,
            x2: 1.0,
            y2: 2.0,
        };
        assert!(eval_timing(overshoot, 0.5) > 1.0);
        assert_eq!(at("auto", "none", overshoot, 50.0), "none");
    }

    #[test]
    fn apply_transitions_drives_overlay_from_clock() {
        // Mirror the interpolation harness: first read establishes
        // `from`, then transition params + `to` are set, then the
        // read at delay -50s / duration 100s (progress 0.5, linear)
        // must show the asymmetric flip values.
        let key = "w80-test-overlay-clock";
        let mut p = props(&[
            ("overlay", "auto"),
            ("transition-property", "none"),
            ("transition-duration", "0s"),
            ("transition-delay", "0s"),
            ("transition-timing-function", "ease"),
        ]);
        apply_transitions(key, &mut p);
        assert_eq!(p["overlay"], "auto");
        // auto -> none, linear, progress 0.5: holds auto.
        p.insert("overlay".to_string(), "none".to_string());
        p.insert("transition-property".to_string(), "overlay".to_string());
        p.insert("transition-duration".to_string(), "100s".to_string());
        p.insert("transition-delay".to_string(), "-50s".to_string());
        p.insert(
            "transition-timing-function".to_string(),
            "linear".to_string(),
        );
        apply_transitions(key, &mut p);
        assert_eq!(p["overlay"], "auto");
        // With `all` instead, the same change flips immediately.
        let key2 = "w80-test-overlay-all";
        let mut q = props(&[
            ("overlay", "auto"),
            ("transition-property", "none"),
            ("transition-duration", "0s"),
            ("transition-delay", "0s"),
            ("transition-timing-function", "ease"),
        ]);
        apply_transitions(key2, &mut q);
        q.insert("overlay".to_string(), "none".to_string());
        q.insert("transition-property".to_string(), "all".to_string());
        q.insert("transition-duration".to_string(), "100s".to_string());
        q.insert("transition-delay".to_string(), "-50s".to_string());
        q.insert(
            "transition-timing-function".to_string(),
            "linear".to_string(),
        );
        apply_transitions(key2, &mut q);
        assert_eq!(q["overlay"], "none");
    }
}
