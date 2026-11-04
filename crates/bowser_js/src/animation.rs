//! Web Animations (WAAPI) surface skeleton (w81, worker A).
//!
//! What this module is: the `Element.animate()` surface, the
//! animation-state registry, and (w81-B) the discrete-value
//! application engine [`apply_animations`], hooked at the two
//! `getComputedStyle` call sites. Value scope: discrete
//! properties (`overlay` — see [`ANIMATED_DISCRETE_PROPS`]) plus,
//! since the w81 numeric follow-on, single-component numeric
//! interpolation (numbers, lengths, percentages, angles, times,
//! and colors — see [`animated_numeric_value`]) for every other
//! keyframed property present in the computed-props map.
//!
//! What exists today:
//! * `target.animate(keyframes, options)` on both the production
//!   element prototype and the parse-shim element wrappers (see
//!   `bindings.rs`), returning an `Animation` object with working
//!   `pause()` / `play()` / `cancel()` / `finish()`, a `currentTime`
//!   getter/setter, and a `playState` getter.
//! * Options parsing for `duration`, `delay`, `endDelay`, `easing`,
//!   `fill`, `direction`, `iterations`, and `iterationStart`, at
//!   least in the shapes `interpolation-testcommon.js` uses
//!   (`{fill: 'forwards', duration: 100000, easing: ...}` plus
//!   keyframes with `offset` / `composite` / camelCase properties).
//! * Easing reuse, not duplication: [`crate::transition`] owns the
//!   timing-function solver (`parse_timing`, `eval_timing`), and this
//!   module calls it. Do not add a second bezier/steps evaluator here.
//!
//! Clock model (mirrors `transition.rs`: clock-at-read, no timers):
//! `currentTime` is the animation's time in milliseconds: elapsed
//! wall time since its start instant (so a fresh animation reads
//! ~`0`, and the delay phase reads `0..delay`). The EFFECT's local
//! time — what progress is computed from — is `currentTime - delay`,
//! so local `0` is the first keyframe and local `duration` is the
//! last; [`AnimationState::iteration_progress`] and
//! [`AnimationState::applies_at`] subtract the delay themselves, so
//! B must NOT subtract it again. `pause()` freezes `currentTime`;
//! setting `currentTime` while paused stores a hold time, while
//! running re-seeks the start instant so the next read reports the
//! set value. The WPT harness pattern (`animate(...)`, `pause()`,
//! `currentTime = 50000`) therefore reads back exactly `50000`
//! with no frame pump involved, exactly as the transition engine's
//! negative-delay pattern does.
//!
//! Leniency, stated not hidden: invalid enum strings (`fill`,
//! `direction`, `composite`) and unparseable `easing` strings fall
//! back to their defaults instead of throwing, and a
//! property-indexed keyframe object (`{opacity: [0, 1]}`) is NOT
//! supported — a bare object argument is read as a single keyframe.
//! Throwing on an exotic-but-harmless option would turn a whole
//! interpolation file into a test error; tightening this to
//! spec-exact `TypeError`s is future work, not worker B's.
//!
//! # Animation-state key format — the contract worker B must use
//!
//! Two different keys, do not conflate them:
//!
//! 1. Element key (identifies the TARGET across reads). This is
//!    EXACTLY the key `transition::apply_transitions` receives at
//!    the same call site, so transition state and animation state
//!    for one element can never alias a different element:
//!    * Production (`n_get_computed_style` in `bindings.rs`):
//!      `prod:{NodeId:?}` — the `NodeId` Debug form, which carries
//!      the arena index plus the process-unique document tag. The
//!      arena is append-only and tags never repeat.
//!    * Parse shim (`n_parse_get_computed_style` in `bindings.rs`):
//!      `parse:{host}:{arena}` — `host` is the `PARSE_HOSTS` id (the
//!      `PARSE_CTX_KEY` float tag on the shim wrapper) and `arena`
//!      is `ScriptElement::arena_index()` — NOT the wrapper's
//!      `PARSE_EL_KEY` slot number. The shim's `animate` native
//!      resolves its `ScriptElement` first and builds the key from
//!      `arena_index()` for this reason.
//!    The two surfaces never share state for the same element: a
//!    shim element upgraded to production re-seeds under its `prod:`
//!    key on its first production read, exactly like transitions.
//! 2. Animation id (identifies one `animate()` call). A per-thread
//!    `u64` counter starting at 1. The JS `Animation` object carries
//!    it in the hidden property `\0bowser:animation` as a `Float`
//!    (not `Int`: ids are `u64`, and the `PARSE_CTX_KEY` precedent
//!    stores `u64` ids as floats). Registry entries are keyed by
//!    this id; [`animations_for_target`] groups them by element key
//!    in creation order (ascending id) for B.
//!
//! # Hook call sites for worker B (already wired, body is a no-op)
//!
//! B applies animated values OVER cascaded values, AFTER
//! transitions, so an active animation wins over a running
//! transition for the same property (animations are a higher
//! cascade origin than transitions in effect):
//! * Production: `n_get_computed_style`, directly after the
//!   `apply_transitions(&format!("prod:{node:?}"), &mut props)` call,
//!   before `wrap_style`.
//! * Parse shim: `n_parse_get_computed_style`, directly after the
//!   `apply_transitions(&key, &mut values)` call, before the
//!   computed-style object is built.
//! B must NOT hook any other surface (in particular not
//! `n_style_get_property_value` or `computed_props` itself): the
//! WPT driver exercises the parse shim, and both surfaces above are
//! the only ones the transition engine drives, for the same reason.
//! B's implementation reads [`animations_for_target`] for the
//! element key, uses [`AnimationState::iteration_progress`] /
//! [`AnimationState::eased_progress`] / [`AnimationState::applies_at`]
//! per animation, and writes the winning value into the props map.
//! Offsets that are `None` in a stored keyframe are distributed by
//! [`normalize_offsets`] (first `0`, last `1`, evenly spaced in
//! between, per spec).
//!
//! Numeric interpolation (w81 numeric follow-on): every
//! keyframed property already present in the computed-props map
//! interpolates when its segment endpoints are single-component
//! numerics (bare numbers, lengths, percentages, angles, times,
//! `fr`) or colors (named, hex, `transparent`, `rgb()` /
//! `rgba()` in comma or space/`/` syntax); anything else
//! (keywords, `calc()`, transforms, gradients, lists, `hsl()` /
//! `hwb()` / `lab()`, mismatched length/percentage families)
//! falls back to the generic discrete flip at segment `t >= 0.5`.
//! Still NOT done, stated not hidden: `calc()` / math-function
//! interpolation, multi-component values (transform lists,
//! shadows, gradients), `hsl()` and friends, shorthand-keyframe
//! expansion, and `add` / `accumulate` compositing against the
//! cascaded base (all treated as `replace` today).
//!
//! Test-scope note (read before citing a test as production
//! coverage): the numeric tests for `rotate` and
//! `animation-duration` drive `apply_animations` through a
//! synthetic props map that already contains those keys. Neither
//! property is in the production `STYLE_PROPS` list nor in the
//! parse-shim `PARSE_COMPUTED_PROPS` list (both in
//! `bindings.rs`), so in production `apply_animations` skips
//! them — the map never contains them to overwrite — and those
//! tests pin engine behavior only, not a production-visible
//! animation of `rotate` or `animation-duration`. Wiring either
//! property into those lists is a separate bindings change.
//!
//! `!important` precedence (w84, CSS Cascade 5): animations sit
//! below all `!important` origins, so [`apply_animations`] never
//! overwrites a property whose cascade winner is `!important` —
//! the caller passes those names in `blocked`, taken from the
//! cascade before transitions run (transitions legitimately
//! outrank `!important` and are unaffected). The plain-`<div>`
//! case is the UA rule itself: `* { overlay: none !important }`
//! wins the cascade, so a WAAPI-animated plain `<div>` reads
//! `none` at every progress, and
//! `css/css-position/overlay/animation/overlay-interpolation.html`
//! passes 88/88 because its expected targets read the same
//! UA-collapsed `none`. (Its CSS Animations method still passes
//! vacuously: `@keyframes` rules are parse-and-skipped and no CSS
//! animation engine exists at all — see `parse.rs`.) A top-layer
//! element is the mirror case: the UA `none` rule is exempted
//! there, so an author-normal `overlay` winner is unblocked and
//! animates, while a top-layer popover's UA `auto !important`
//! winner stays blocked at `auto`.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::transition::{Timing, eval_timing, parse_timing};

/// `KeyframeEffectOptions.fill` / `KeyframeAnimationOptions.fill`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Fill {
    #[default]
    None,
    Forwards,
    Backwards,
    Both,
    /// Spec `auto`: resolves to `none` for `animate()`-created
    /// effects. Kept distinct so B can see what was authored.
    Auto,
}

impl Fill {
    /// Parse a fill keyword; `None` for anything else (the caller
    /// falls back to the default — see the module leniency note).
    pub fn parse(s: &str) -> Option<Fill> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" => Some(Fill::None),
            "forwards" => Some(Fill::Forwards),
            "backwards" => Some(Fill::Backwards),
            "both" => Some(Fill::Both),
            "auto" => Some(Fill::Auto),
            _ => None,
        }
    }

    /// Whether the fill mode holds the effect before its active
    /// interval (the `backwards` / `both` half).
    pub fn holds_before(self) -> bool {
        matches!(self, Fill::Backwards | Fill::Both)
    }

    /// Whether the fill mode holds the effect after its active
    /// interval (the `forwards` / `both` half).
    pub fn holds_after(self) -> bool {
        matches!(self, Fill::Forwards | Fill::Both)
    }
}

/// `KeyframeEffectOptions.direction`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Direction {
    #[default]
    Normal,
    Reverse,
    Alternate,
    AlternateReverse,
}

impl Direction {
    /// Parse a direction keyword; `None` for anything else.
    pub fn parse(s: &str) -> Option<Direction> {
        match s.trim().to_ascii_lowercase().as_str() {
            "normal" => Some(Direction::Normal),
            "reverse" => Some(Direction::Reverse),
            "alternate" => Some(Direction::Alternate),
            "alternate-reverse" => Some(Direction::AlternateReverse),
            _ => None,
        }
    }
}

/// A keyframe's `composite` operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Composite {
    #[default]
    Replace,
    Add,
    Accumulate,
}

impl Composite {
    /// Parse a composite keyword; `None` for anything else.
    pub fn parse(s: &str) -> Option<Composite> {
        match s.trim().to_ascii_lowercase().as_str() {
            "replace" => Some(Composite::Replace),
            "add" => Some(Composite::Add),
            "accumulate" => Some(Composite::Accumulate),
            _ => None,
        }
    }
}

/// `Animation.playState`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlayState {
    /// No current time (cancelled, or never started): `currentTime`
    /// reads as `null`.
    #[default]
    Idle,
    Running,
    Paused,
    Finished,
}

impl PlayState {
    /// The WebIDL string for this state.
    pub fn as_str(self) -> &'static str {
        match self {
            PlayState::Idle => "idle",
            PlayState::Running => "running",
            PlayState::Paused => "paused",
            PlayState::Finished => "finished",
        }
    }
}

/// One parsed keyframe, as authored. Property names are stored
/// DASHED (CSS longhand form, e.g. `background-color`), converted
/// from the camelCase names `interpolation-testcommon.js` generates
/// by [`camel_to_dashed`]; `offset` stays `None` when the author
/// omitted it (distribution is worker B's job, per the module doc).
#[derive(Clone, Debug, PartialEq)]
pub struct Keyframe {
    /// Keyframe offset in `[0, 1]`, or `None` when omitted.
    pub offset: Option<f64>,
    /// The keyframe's composite operation (default `replace`).
    pub composite: Composite,
    /// A per-keyframe easing, or `None` for the effect easing.
    pub easing: Option<Timing>,
    /// `(dashed property name, serialized value)` pairs, in
    /// authoring order. Reserved keyframe members (`offset`,
    /// `composite`, `easing`) are never stored here.
    pub properties: Vec<(String, String)>,
}

impl Keyframe {
    /// The serialized value for dashed property `prop`, if this
    /// keyframe sets it.
    ///
    /// Worker-B read surface; unused by w81-A itself.
    #[allow(dead_code)]
    pub fn value(&self, prop: &str) -> Option<&str> {
        self.properties
            .iter()
            .find(|(name, _)| name == prop)
            .map(|(_, value)| value.as_str())
    }
}

/// Convert a JS keyframe property name to its dashed CSS form.
/// `cssOffset` / `cssFloat` are the testcommon aliases for the CSS
/// properties `offset` / `float` (a bare `offset` key would collide
/// with the keyframe-offset member); custom properties pass through
/// untouched.
pub fn camel_to_dashed(prop: &str) -> String {
    match prop {
        "cssOffset" => return "offset".to_string(),
        "cssFloat" => return "float".to_string(),
        _ => {}
    }
    if prop.starts_with("--") {
        return prop.to_string();
    }
    let mut out = String::with_capacity(prop.len() + 4);
    for c in prop.chars() {
        if c.is_ascii_uppercase() {
            out.push('-');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// The timing options of one `animate()` call, in milliseconds.
/// `duration_ms` / `delay_ms` / `end_delay_ms` are all local-time
/// milliseconds (JS numbers are already milliseconds; serialized
/// `<time>` strings are converted by the bindings layer with
/// `transition::parse_time`).
#[derive(Clone, Debug, PartialEq)]
pub struct AnimationOptions {
    /// Active-interval length per iteration (default `0`).
    pub duration_ms: f64,
    /// Delay before the active interval starts (default `0`).
    pub delay_ms: f64,
    /// Delay after the active interval ends (default `0`).
    pub end_delay_ms: f64,
    /// The effect easing (default `linear`).
    pub easing: Timing,
    /// The fill mode (default `none`).
    pub fill: Fill,
    /// The playback direction (default `normal`).
    pub direction: Direction,
    /// Iteration count (default `1`; `INFINITY` is allowed).
    pub iterations: f64,
    /// Iteration start offset (default `0`).
    pub iteration_start: f64,
}

impl Default for AnimationOptions {
    fn default() -> Self {
        AnimationOptions {
            duration_ms: 0.0,
            delay_ms: 0.0,
            end_delay_ms: 0.0,
            easing: Timing::Linear,
            fill: Fill::None,
            direction: Direction::Normal,
            iterations: 1.0,
            iteration_start: 0.0,
        }
    }
}

impl AnimationOptions {
    /// The total active-interval length across all iterations, or
    /// `INFINITY` for an unbounded iteration count / duration.
    pub fn active_duration_ms(&self) -> f64 {
        // WAAPI: if either the iteration duration or the
        // iteration count is zero, the repeated duration is
        // zero — the `0 * inf` case resolves to zero, not
        // infinity, so this check must precede the infinity
        // check below (otherwise a zero duration with infinite
        // iterations, or vice versa, would wrongly report an
        // infinite active duration and `finish()` would throw).
        if self.duration_ms == 0.0 || self.iterations == 0.0 {
            0.0
        } else if self.duration_ms.is_infinite() || self.iterations.is_infinite() {
            f64::INFINITY
        } else {
            self.duration_ms * self.iterations
        }
    }

    /// Parse an easing string with the shared transition solver;
    /// unparseable strings fall back to `linear` (module leniency
    /// note) instead of failing the whole `animate()` call.
    pub fn easing_from_str(s: &str) -> Timing {
        parse_timing(s).unwrap_or(Timing::Linear)
    }
}

/// Shift an `Instant` by signed milliseconds, saturating at the
/// `Instant` bounds instead of panicking on overflow.
///
/// The clamp is load-bearing, not cosmetic:
/// `Duration::from_secs_f64` panics on magnitudes it cannot
/// represent (on rustc 1.98 even `Duration::MAX.as_secs_f64()`
/// itself panics on the way back in, because the `f64` rounds
/// just past `Duration::MAX`), so a huge finite JS time such as
/// `anim.currentTime = 1e25` must be clamped *before* conversion.
/// `1e18` seconds is far below both that float-conversion
/// threshold (`~1.844e19` s) and the platform `Instant`'s `i64`
/// timespec range, so the `checked_add` / `checked_sub` below
/// then succeed and the seek saturates at the clamped bound
/// (reading back `~1e21` ms) instead of collapsing to `now`.
/// The residual `unwrap_or(now)` covers only a clock whose
/// range is narrower than that bound.
const MAX_SHIFT_SECS: f64 = 1.0e18;

fn shift_ms(now: Instant, ms: f64) -> Instant {
    if !ms.is_finite() {
        return now;
    }
    let dur = Duration::from_secs_f64((ms / 1000.0).abs().min(MAX_SHIFT_SECS));
    if ms >= 0.0 {
        now.checked_add(dur).unwrap_or(now)
    } else {
        now.checked_sub(dur).unwrap_or(now)
    }
}

/// One live `Animation`: the registry entry behind a JS `Animation`
/// object. `start` is the wall-clock instant at which the animation
/// time is `0`; when paused or finished, `hold_time_ms` carries the
/// frozen animation time and `start` is stale.
///
/// The `id` / `target_key` / `keyframes` fields are worker B's read
/// surface (via [`animations_for_target`]); w81-A never reads them
/// itself, hence the allow below rather than a fake use.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct AnimationState {
    /// Registry id (see the module key-format doc).
    pub id: u64,
    /// The target element key (`prod:…` / `parse:…`, same string
    /// `transition::apply_transitions` receives for this element).
    pub target_key: String,
    /// The keyframes, as authored (offsets may be `None`).
    pub keyframes: Vec<Keyframe>,
    /// The timing options.
    pub options: AnimationOptions,
    /// Wall-clock start instant (running state only).
    pub start: Instant,
    /// Frozen local time while paused / finished; `None` while
    /// running or idle.
    pub hold_time_ms: Option<f64>,
    /// The playback state.
    pub play_state: PlayState,
}

impl AnimationState {
    /// Create a running animation starting now.
    pub fn new(
        id: u64,
        target_key: String,
        keyframes: Vec<Keyframe>,
        options: AnimationOptions,
    ) -> AnimationState {
        AnimationState {
            id,
            target_key,
            keyframes,
            options,
            start: Instant::now(),
            hold_time_ms: None,
            play_state: PlayState::Running,
        }
    }

    /// The animation time at `now` in milliseconds, or `None` when
    /// idle (cancelled): this is what `Animation.currentTime`
    /// reports. Running time is `now - start`; it is `0..delay`
    /// during the delay phase and past `delay + active` after the
    /// active interval ends. The effect local time is this value
    /// minus `delay_ms` (see [`Self::iteration_progress`]).
    pub fn current_time_ms(&self, now: Instant) -> Option<f64> {
        match self.play_state {
            PlayState::Idle => None,
            PlayState::Paused | PlayState::Finished => self.hold_time_ms,
            PlayState::Running => {
                // Signed elapsed time, not a clamp: a negative
                // seek stores a start instant in the future
                // (`set_current_time` shifts by `-value`), so
                // `now < start` is exactly a negative animation
                // time and must read back negative. The old
                // `unwrap_or(0.0)` collapsed every negative seek
                // to `0` on read. The reverse `checked_duration`
                // is infallible here: if `now < start` the start
                // instant is after `now` by construction.
                let elapsed_ms = match now.checked_duration_since(self.start) {
                    Some(d) => d.as_secs_f64() * 1000.0,
                    None => {
                        -self
                            .start
                            .checked_duration_since(now)
                            .unwrap_or(Duration::ZERO)
                            .as_secs_f64()
                            * 1000.0
                    }
                };
                Some(elapsed_ms)
            }
        }
    }

    /// Set the animation time: `None` clears the animation to idle
    /// (the `currentTime = null` path); a value while paused or
    /// finished stores a hold time, while running (or idle) re-seeks
    /// the start instant so the next read reports the value.
    pub fn set_current_time(&mut self, now: Instant, value: Option<f64>) {
        match value {
            None => {
                self.play_state = PlayState::Idle;
                self.hold_time_ms = None;
            }
            Some(v) => match self.play_state {
                PlayState::Paused | PlayState::Finished => {
                    self.hold_time_ms = Some(v);
                }
                PlayState::Running | PlayState::Idle => {
                    self.start = shift_ms(now, -v);
                    self.hold_time_ms = None;
                    self.play_state = PlayState::Running;
                }
            },
        }
    }

    /// Freeze the current animation time. No-op when idle; keeps
    /// the existing hold when already paused or finished.
    pub fn pause(&mut self, now: Instant) {
        match self.play_state {
            PlayState::Running => {
                self.hold_time_ms = self.current_time_ms(now);
                self.play_state = PlayState::Paused;
            }
            PlayState::Idle | PlayState::Paused | PlayState::Finished => {}
        }
    }

    /// Resume (or restart, when idle or finished): a paused
    /// animation re-seeks its start so the hold time is preserved;
    /// an idle or finished animation restarts from animation
    /// time `0`.
    pub fn play(&mut self, now: Instant) {
        match self.play_state {
            PlayState::Paused => {
                let hold = self.hold_time_ms.unwrap_or(0.0);
                self.start = shift_ms(now, -hold);
                self.hold_time_ms = None;
                self.play_state = PlayState::Running;
            }
            PlayState::Idle | PlayState::Finished => {
                self.start = now;
                self.hold_time_ms = None;
                self.play_state = PlayState::Running;
            }
            PlayState::Running => {}
        }
    }

    /// Cancel: back to idle, `currentTime` reads as `null`.
    pub fn cancel(&mut self) {
        self.play_state = PlayState::Idle;
        self.hold_time_ms = None;
    }

    /// The animation's associated effect end (Web Animations 1
    /// §4.5.12: the end time of the associated effect), i.e.
    /// `delay + active duration + endDelay`. The `0 * inf`
    /// case is already resolved to zero inside
    /// [`AnimationOptions::active_duration_ms`], so no
    /// special-casing is needed here: a zero iteration
    /// duration or a zero iteration count yields a finite
    /// effect end (`delay + endDelay`).
    fn effect_end_ms(&self) -> f64 {
        self.options.delay_ms + self.options.active_duration_ms() + self.options.end_delay_ms
    }

    /// Jump to the end of the active interval and finish, per
    /// Web Animations 1 §4.5.13 ("finish an animation"). Returns
    /// `false` — and changes nothing — when the spec says to
    /// throw an `InvalidStateError` instead: the effective
    /// playback rate is zero, or the rate is positive and the
    /// associated effect end is infinity (an infinite iteration
    /// count with a nonzero iteration duration, or an infinite
    /// iteration duration with a nonzero iteration count; a
    /// zero factor makes the effect end finite, see
    /// [`AnimationOptions::active_duration_ms`]). The `whole M2` / `JS M3` conflict is
    /// resolved in favor of throwing: §4.5.13 step 1 is
    /// unconditional, and the no-op reading has no spec text
    /// behind it. Two engine limits, stated not hidden: this
    /// engine has no playback-rate field (the rate is implicitly
    /// `1`, so the rate-0 throw branch is currently vacuous — a
    /// future rate field must check it before the limit step,
    /// exactly as §4.5.13 step 1 orders it), and the `bindings.rs`
    /// `finish` native maps `Some(false)` from [`finish_checked`]
    /// to `InvalidStateError`, while `None` (unknown id) maps to
    /// the stale-animation error; the throw-vs-finish
    /// decision itself is made here. On success the hold time
    /// is the effect end (`delay + active + endDelay`), not
    /// `delay + active` as before: §4.5.13 step 4 sets the
    /// current time to the effect end, which includes endDelay.
    pub fn finish(&mut self, _now: Instant) -> bool {
        if self.effect_end_ms().is_infinite() {
            return false;
        }
        if self.play_state == PlayState::Finished {
            return true;
        }
        self.hold_time_ms = Some(self.effect_end_ms());
        self.play_state = PlayState::Finished;
        true
    }

    /// Apply the playback direction to a simple-iteration
    /// progress: `iter` is the current iteration index (the
    /// final iteration for an end value), `frac` the progress
    /// within it. Alternate parity is computed in `f64` rather
    /// than via an `i64` cast, so a huge iteration index does
    /// not saturate to a bogus parity.
    fn directed_progress(&self, iter: f64, frac: f64) -> f64 {
        let p = match self.options.direction {
            Direction::Normal => frac,
            Direction::Reverse => 1.0 - frac,
            Direction::Alternate => {
                if iter % 2.0 == 1.0 {
                    1.0 - frac
                } else {
                    frac
                }
            }
            Direction::AlternateReverse => {
                if iter % 2.0 == 1.0 {
                    frac
                } else {
                    1.0 - frac
                }
            }
        };
        p.clamp(0.0, 1.0)
    }

    /// The iteration progress at `now` in `[0, 1]` (direction
    /// applied, easing NOT applied — use [`Self::eased_progress`]
    /// for that), or `None` when idle. The effect local time is
    /// `currentTime - delay`.
    ///
    /// Every phase routes through [`Self::directed_progress`];
    /// no phase returns a direction-blind `0` or `1` (the old
    /// code did exactly that for zero/infinite durations and
    /// for the before phase, and the old doc comment described
    /// that skipping as correct — it was not). Before phase
    /// (`local < 0`): the start of iteration 0 under the
    /// direction mapping, so `reverse` and `alternate-reverse`
    /// show the end keyframe during the delay when a backwards
    /// fill applies, while `normal` and `alternate` show the
    /// start keyframe. Zero duration: the final iteration's end
    /// value under the direction mapping, so `reverse` ends at
    /// progress `0` and `normal` at `1`. Infinite duration: the
    /// normal path with `local / inf = 0`, so a finite local
    /// time is progress `0` under `normal` and
    /// direction-adjusted otherwise (e.g. `1` under `reverse`
    /// at the iteration start). Past a finite active end: the
    /// final iteration's end value under the direction mapping,
    /// as before.
    /// A non-finite local time (a `NaN` / infinite hold time
    /// set through the Rust API — the JS layer ignores
    /// non-finite assignments) never yields `NaN` progress:
    /// `NaN` is treated as the active start (`local = 0`), and
    /// an infinite local time with an infinite duration takes
    /// the `local / inf = 0` branch rather than `inf / inf`.
    /// Likewise a non-finite iteration total (finite duration,
    /// infinite iterations, infinite local time) reports the
    /// active-end value rather than `inf - inf = NaN`.
    /// This engine has no playback-rate field, so there is no
    /// rate-0 branch here: the rate is implicitly `1`.
    ///
    /// Worker-B read surface (see the module header); unused by
    /// w81-A itself.
    #[allow(dead_code)]
    pub fn iteration_progress(&self, now: Instant) -> Option<f64> {
        let mut local = self.current_time_ms(now)? - self.options.delay_ms;
        if local.is_nan() {
            local = 0.0;
        }
        if local < 0.0 {
            return Some(self.directed_progress(0.0, 0.0));
        }
        let duration = self.options.duration_ms;
        if duration <= 0.0 {
            let last = self.options.iterations.max(1.0) - 1.0;
            return Some(self.directed_progress(last, 1.0));
        }
        if !duration.is_finite() {
            // Infinite iteration duration: `local / inf` is 0
            // for finite local time; an infinite local time is
            // pinned to the same branch instead of `inf / inf`.
            let ratio = if local.is_finite() {
                local / duration
            } else {
                0.0
            };
            let total = self.options.iteration_start + ratio;
            if total <= 0.0 {
                return Some(self.directed_progress(0.0, 0.0));
            }
            if !total.is_finite() {
                // Non-finite total (only reachable through a
                // non-finite iteration start via the Rust API,
                // since `ratio` is pinned above): `floor` /
                // `frac` would yield NaN, so report the
                // active-end value instead.
                let last = self.options.iterations.max(1.0) - 1.0;
                return Some(self.directed_progress(last, 1.0));
            }
            return Some(self.directed_progress(total.floor(), total - total.floor()));
        }
        let active = self.options.active_duration_ms();
        if active.is_finite() && local >= active {
            let last = self.options.iterations.max(1.0) - 1.0;
            return Some(self.directed_progress(last, 1.0));
        }
        let total = self.options.iteration_start + local / duration;
        if total <= 0.0 {
            return Some(self.directed_progress(0.0, 0.0));
        }
        if !total.is_finite() {
            // Finite duration with infinite iterations and a
            // non-finite local time (e.g. an infinite hold
            // through the Rust API): `local / duration` is
            // infinite, so the active-end branch above was
            // skipped and `total - total.floor()` would be
            // `inf - inf = NaN`. Report the active-end value
            // instead, keeping the "never yields NaN progress"
            // guarantee. (A negative-infinite total is caught
            // by the `total <= 0.0` branch above, not here.)
            let last = self.options.iterations.max(1.0) - 1.0;
            return Some(self.directed_progress(last, 1.0));
        }
        Some(self.directed_progress(total.floor(), total - total.floor()))
    }

    /// [`Self::iteration_progress`] with the effect easing applied,
    /// or `None` when idle.
    ///
    /// Worker-B read surface; unused by w81-A itself.
    #[allow(dead_code)]
    pub fn eased_progress(&self, now: Instant) -> Option<f64> {
        self.iteration_progress(now)
            .map(|p| eval_timing(self.options.easing, p))
    }

    /// Whether this animation contributes a value at `now` under
    /// its fill mode: always inside the active interval; before it
    /// only with a backwards fill, after it only with a forwards
    /// fill. Idle animations never apply. The phase test uses the
    /// effect local time (`currentTime - delay`).
    ///
    /// The active interval is half-open (Web Animations 1
    /// §4.6.9.3 endpoint-exclusive timing, `[begin, end)`; the
    /// after phase is defined in §4.6.6 as local time equal to
    /// the active-after boundary time, forwards direction, with
    /// the endpoint-inclusive flag false). In this module's
    /// terms — `local = currentTime - delay` — that boundary is
    /// `local == active`, so `local >= active` is already the
    /// after phase and takes the forwards-fill branch, not the
    /// always-applies branch: with fill `none`, an animation at
    /// exactly its active end applies nothing. This is
    /// consistent with [`Self::iteration_progress`], which
    /// already treated `local >= active` as the final value.
    /// EndDelay does not change the boundary: the active-after
    /// boundary is `min(delay + active, end time)` in spec
    /// local time, i.e. `active` here, so during endDelay a
    /// forwards fill still holds (via the after-phase branch)
    /// and fill `none` still applies nothing.
    ///
    /// Worker-B read surface; unused by w81-A itself.
    #[allow(dead_code)]
    pub fn applies_at(&self, now: Instant) -> bool {
        let Some(local) = self.current_time_ms(now).map(|t| t - self.options.delay_ms) else {
            return false;
        };
        if local < 0.0 {
            return self.options.fill.holds_before();
        }
        let active = self.options.active_duration_ms();
        if active.is_finite() && local >= active {
            return self.options.fill.holds_after();
        }
        true
    }
}

/// Registry cap, mirroring `transition.rs`: entries for removed
/// elements linger until the cap evicts idle ones. Eviction is
/// behaviorally safe — a dropped entry simply has no animation on
/// the next read — and running / paused / finished entries are
/// never evicted, so the cap bounds idle memory, not live effects.
const REGISTRY_CAP: usize = 4096;

thread_local! {
    static REGISTRY: RefCell<HashMap<u64, AnimationState>> = RefCell::new(HashMap::new());
    static NEXT_ID: Cell<u64> = const { Cell::new(1) };
}

/// Create a running animation for `target_key` and return its
/// registry id. The caller stores the id on the JS `Animation`
/// object (hidden `\0bowser:animation` property).
pub fn create_animation(
    target_key: String,
    keyframes: Vec<Keyframe>,
    options: AnimationOptions,
) -> u64 {
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        if registry.len() >= REGISTRY_CAP {
            registry.retain(|_, s| s.play_state != PlayState::Idle);
        }
        let id = NEXT_ID.with(|next| {
            let id = next.get();
            next.set(id.wrapping_add(1).max(1));
            id
        });
        registry.insert(id, AnimationState::new(id, target_key, keyframes, options));
        id
    })
}

/// Run `f` against the registry entry for `id`; `false` when there
/// is no such animation.
pub fn with_animation_mut<T>(id: u64, f: impl FnOnce(&mut AnimationState) -> T) -> Option<T> {
    REGISTRY.with(|registry| registry.borrow_mut().get_mut(&id).map(f))
}

/// The `currentTime` of animation `id` at this instant:
/// `None` = unknown id, `Some(None)` = idle (`null` in JS),
/// `Some(Some(ms))` = animation time in milliseconds.
pub fn current_time(id: u64) -> Option<Option<f64>> {
    let now = Instant::now();
    with_animation_mut(id, |s| s.current_time_ms(now))
}

/// Set `currentTime` for animation `id`; `false` for an unknown id.
pub fn set_current_time(id: u64, value: Option<f64>) -> bool {
    let now = Instant::now();
    with_animation_mut(id, |s| s.set_current_time(now, value)).is_some()
}

/// Pause animation `id`; `false` for an unknown id.
pub fn pause(id: u64) -> bool {
    let now = Instant::now();
    with_animation_mut(id, |s| s.pause(now)).is_some()
}

/// Play (resume / restart) animation `id`; `false` for an unknown id.
pub fn play(id: u64) -> bool {
    let now = Instant::now();
    with_animation_mut(id, |s| s.play(now)).is_some()
}

/// Cancel animation `id`; `false` for an unknown id.
pub fn cancel(id: u64) -> bool {
    with_animation_mut(id, |s| s.cancel()).is_some()
}

/// Finish animation `id`, distinguishing the spec-throw case
/// from a stale id: `None` = unknown id, `Some(true)` =
/// finished, `Some(false)` = the spec's `InvalidStateError`
/// case (infinite associated effect end; see
/// [`AnimationState::finish`]). The `bindings.rs` native maps
/// `Some(false)` to `InvalidStateError` and `None` to the
/// stale-animation error.
pub fn finish_checked(id: u64) -> Option<bool> {
    let now = Instant::now();
    with_animation_mut(id, |s| s.finish(now))
}

/// The `playState` string of animation `id`, or `None` for an
/// unknown id.
pub fn play_state_str(id: u64) -> Option<&'static str> {
    with_animation_mut(id, |s| s.play_state.as_str())
}

/// All animations targeting `target_key`, cloned in creation order
/// (ascending id). This is worker B's read path: B calls it with
/// exactly the element key it passes to
/// `transition::apply_transitions` for the same element, at the
/// `getComputedStyle` hook sites documented in the module header.
#[allow(dead_code)]
pub fn animations_for_target(target_key: &str) -> Vec<AnimationState> {
    REGISTRY.with(|registry| {
        let mut out: Vec<AnimationState> = registry
            .borrow()
            .values()
            .filter(|s| s.target_key == target_key)
            .cloned()
            .collect();
        out.sort_by_key(|s| s.id);
        out
    })
}

/// Number of live registry entries (tests and diagnostics only).
#[allow(dead_code)]
pub fn animation_count() -> usize {
    REGISTRY.with(|registry| registry.borrow().len())
}

/// Discrete properties `apply_animations` routes through the
/// discrete engine.
///
/// This mirrors `transition.rs`'s `DISCRETE_PROPS`: `overlay`
/// and (w82) `display` are the properties whose WAAPI value is
/// discrete-only with asymmetric flips. Every other keyframed
/// property goes through [`animated_numeric_value`], which
/// interpolates numeric and color values and falls back to the
/// generic discrete flip for anything else; a later engine
/// extends that function, it does not change this list or the
/// discrete logic.
const ANIMATED_DISCRETE_PROPS: &[&str] = &["overlay", "display"];

/// Normalize keyframe offsets per the WAAPI keyframe-processing
/// algorithm (Web Animations 1 §5.3.3, "compute missing keyframe
/// offsets": offsets left `None` by the author are distributed):
/// a missing first offset is `0` — but only when there is more
/// than one keyframe — a missing last offset is `1`, a run of
/// missing offsets between two specified offsets is evenly
/// spaced between them, and the result is forced
/// non-decreasing (an author offset below its predecessor is
/// raised to the predecessor, as the spec's processing does).
/// The single-keyframe case is the exception that matters: the
/// spec's first-offset rule is explicitly guarded by "more than
/// one keyframe", while the last-offset rule is not, so a lone
/// keyframe with no offset reads `1`, not `0`. That is the
/// end-state rule — a lone keyframe is the end state, and the
/// start is inferred from the underlying (cascaded) value by
/// the endpoint padding in [`property_entries`] — not a
/// start-state keyframe held for the whole animation.
/// The returned vector is parallel to `keyframes`.
pub fn normalize_offsets(keyframes: &[Keyframe]) -> Vec<f64> {
    let n = keyframes.len();
    let mut offsets: Vec<Option<f64>> = keyframes.iter().map(|kf| kf.offset).collect();
    if n == 0 {
        return Vec::new();
    }
    if n > 1 && offsets[0].is_none() {
        offsets[0] = Some(0.0);
    }
    if offsets[n - 1].is_none() {
        offsets[n - 1] = Some(1.0);
    }
    let mut i = 0;
    while i < n {
        if offsets[i].is_some() {
            i += 1;
            continue;
        }
        let start = i - 1;
        let mut end = i;
        while end < n && offsets[end].is_none() {
            end += 1;
        }
        let (from, to) = if end < n {
            (offsets[start].unwrap_or(0.0), offsets[end].unwrap())
        } else {
            // No specified offset after the run (only possible
            // when `n == 1`, handled above, or a trailing run
            // after a specified offset — unreachable in practice
            // since the last offset was filled — but stay safe).
            (offsets[start].unwrap_or(0.0), 1.0)
        };
        let span = (end - start) as f64;
        for (j, slot) in offsets[(start + 1)..end].iter_mut().enumerate() {
            let k = start + 1 + j;
            *slot = Some(from + (to - from) * ((k - start) as f64) / span);
        }
        i = end;
    }
    let mut out: Vec<f64> = offsets.into_iter().map(|o| o.unwrap_or(0.0)).collect();
    let mut prev = f64::NEG_INFINITY;
    for o in out.iter_mut() {
        if *o < prev {
            *o = prev;
        }
        prev = *o;
    }
    out
}

/// Pick the discrete value inside one keyframe segment, mirroring
/// the asymmetric flip rules in `transition::eval_discrete` (that
/// helper is private to `transition.rs`, so this reproduces its
/// three-branch rule rather than calling it; the easing solver
/// itself, [`eval_timing`], is genuinely shared, not duplicated).
/// `t` is the segment-local progress (already eased, `0` at the
/// segment's start keyframe, `1` at its end keyframe; it may lie
/// outside `[0, 1]` for bezier easings whose `y` overshoots, and
/// the rules below are written to be correct for those too):
/// * `overlay` `auto -> none` flips only at `t >= 1`;
/// * `overlay` `none -> auto` flips at `t > 0`;
/// * `display` mirrors that asymmetry over `none` vs shown:
///   shown -> `none` flips only at `t >= 1`, `none` -> shown
///   flips at `t > 0`, shown -> shown flips at `t >= 0.5`;
/// * every other discrete pair flips at `t >= 0.5`.
///
/// UNVERIFIED (w82fix JS-m1): no spec or WPT citation for
/// the WAAPI `display` asymmetry was found — the local WPT
/// checkout is sparse (css-position, CSS2, css-tables only;
/// no web-animations display tests) and no in-repo spec note
/// covers it. The branch is deliberately KEPT, not removed:
/// the transition path depends on the same asymmetry and
/// the overlay-transition-dialog reftest passes with it, and
/// removing it would change WAAPI `display` segments to the
/// generic 0.5 flip. If a Chrome datum or a WPT WAAPI display
/// test contradicts the asymmetry, this branch is the place
/// to revisit — not the transition engine.
fn discrete_segment_value<'a>(from: &'a str, to: &'a str, prop: &str, t: f64) -> &'a str {
    if prop == "overlay" {
        match (from, to) {
            ("auto", "none") => {
                if t >= 1.0 {
                    to
                } else {
                    from
                }
            }
            ("none", "auto") => {
                if t > 0.0 {
                    to
                } else {
                    from
                }
            }
            _ => {
                if t >= 0.5 {
                    to
                } else {
                    from
                }
            }
        }
    } else if prop == "display" {
        let from_none = from.eq_ignore_ascii_case("none");
        let to_none = to.eq_ignore_ascii_case("none");
        match (from_none, to_none) {
            (false, true) => {
                if t >= 1.0 {
                    to
                } else {
                    from
                }
            }
            (true, false) => {
                if t > 0.0 {
                    to
                } else {
                    from
                }
            }
            _ => {
                if t >= 0.5 {
                    to
                } else {
                    from
                }
            }
        }
    } else if t >= 0.5 {
        to
    } else {
        from
    }
}

/// The property-specific keyframes for one property, with the
/// spec's implicit endpoints (Web Animations 1 §5.3.4 steps 8-9):
/// when no keyframe for `prop` sits at offset `0` (resp. `1`),
/// a synthetic keyframe at that offset carrying the underlying
/// value — the cascaded value `apply_animations` read from
/// `props` before overwriting it — is prepended (resp.
/// appended), with replace compositing and linear easing (the
/// synthetic entries carry `None` here, which the engines treat
/// as the effect easing, already folded into `p`; only
/// per-keyframe segment easings are stored). `base` is `None`
/// only when the caller has no underlying value (direct engine
/// callers); `apply_animations` always passes one, since it
/// only writes properties already present in `props`.
///
/// Padding is what makes in-range progress non-extrapolating:
/// after padding, the first entry is at `0` and the last at `1`,
/// so progress inside `[0, 1]` always lands inside a real
/// segment, and only eased progress outside `[0, 1]` (an
/// overshooting easing) extrapolates from the nearest segment.
/// Before this, a first offset of `0.5` extrapolated the first
/// segment backwards to progress `0` — e.g. a `0px -> 100px`
/// segment starting at `0.5` read `-100px` at progress `0`
/// (the negative-width symptom) instead of holding the
/// underlying value. The discrete engine shares this helper,
/// so it gets the same endpoints.
fn property_entries<'a>(
    state: &'a AnimationState,
    prop: &str,
    offsets: &[f64],
    base: Option<&'a str>,
) -> Vec<(f64, &'a str, Option<Timing>)> {
    let mut entries: Vec<(f64, &str, Option<Timing>)> = state
        .keyframes
        .iter()
        .zip(offsets.iter())
        .filter_map(|(kf, offset)| kf.value(prop).map(|v| (*offset, v, kf.easing)))
        .collect();
    if let Some(base) = base
        && !entries.is_empty()
    {
        if entries[0].0 > 0.0 {
            entries.insert(0, (0.0, base, None));
        }
        if entries[entries.len() - 1].0 < 1.0 {
            entries.push((1.0, base, None));
        }
    }
    entries
}

/// The animated value of one discrete property for one animation
/// at `now`, or `None` when the animation does not contribute
/// (idle, outside its fill, or the property is not keyframed).
/// `p` is the animation's eased progress (effect easing applied
/// by [`AnimationState::eased_progress`], direction already
/// folded in by [`AnimationState::iteration_progress`]).
/// `base` is the underlying value for endpoint padding (see
/// [`property_entries`]).
fn animated_discrete_value(
    state: &AnimationState,
    p: f64,
    prop: &str,
    offsets: &[f64],
    base: Option<&str>,
) -> Option<String> {
    let entries = property_entries(state, prop, offsets, base);
    if entries.is_empty() {
        return None;
    }
    if entries.len() == 1 {
        return Some(entries[0].1.to_string());
    }
    // Advance past every entry at or before `p`: at an exact
    // keyframe offset the later keyframe is the segment start,
    // and a segment starting there evaluates at `t = 0` to its
    // own value under every flip rule above, so this is exact.
    let mut idx = 0;
    while idx + 1 < entries.len() && p >= entries[idx + 1].0 {
        idx += 1;
    }
    if idx + 1 >= entries.len() {
        return Some(entries[idx].1.to_string());
    }
    let (from_offset, from_value, from_easing) = entries[idx];
    let (to_offset, to_value, _) = entries[idx + 1];
    let mut t = if to_offset > from_offset {
        (p - from_offset) / (to_offset - from_offset)
    } else {
        1.0
    };
    // A per-keyframe easing transforms its own segment's local
    // progress (the effect easing is already in `p`).
    if let Some(easing) = from_easing {
        t = eval_timing(easing, t);
    }
    Some(discrete_segment_value(from_value, to_value, prop, t).to_string())
}

/// Format an interpolated number the way the computed-style
/// serializer does: Rust `Display` for `f64`, with two snaps —
/// `-0` becomes `0`, and values within `1e-9` of an integer
/// print as that integer — so float noise like
/// `0.30000000000000004` (from `0.1 + 0.2`-style arithmetic) and
/// `49.99999999999999` do not leak into the serialized value.
/// Precision past 9 decimal places is rounded away first; no
/// WPT interpolation expectation needs more than that.
fn format_number(v: f64) -> String {
    if !v.is_finite() {
        return "0".to_string();
    }
    let rounded = (v * 1e9).round() / 1e9;
    if rounded == 0.0 {
        return "0".to_string();
    }
    if (rounded - rounded.round()).abs() < 1e-9 {
        // Integer path only inside the `i64` range: `as i64`
        // saturates, so `1e20` used to serialize as
        // `i64::MAX`. Past the range the `f64` Display below
        // prints the full value instead.
        let int = rounded.round();
        if int.abs() < 9223372036854775808.0 {
            return format!("{}", int as i64);
        }
    }
    format!("{rounded}")
}

/// A parsed single-component numeric value: a bare number, a
/// length, a percentage, an angle, a time, or a `<flex>` value.
/// Multi-component values (`calc()`, transforms, lists) are not
/// parsed — they fall back to discrete interpolation below.
#[derive(Clone, Debug, PartialEq)]
struct NumericValue {
    value: f64,
    /// Lowercased unit (`""` for a bare number, `"%"` for a
    /// percentage, `"px"` / `"deg"` / `"s"` / … otherwise).
    unit: String,
}

/// Split a trimmed value into its numeric prefix and unit suffix.
/// The numeric scan accepts an optional sign, digits, a decimal
/// point, and an exponent (`1e2`, `-.5`, `+3.`); anything else
/// fails, as does a non-finite result. The caller validates the
/// unit; this helper only splits.
fn split_numeric(s: &str) -> Option<(f64, &str)> {
    let s = s.trim();
    let bytes = s.as_bytes();
    let mut i = 0;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    let mut digits = 0;
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
        if bytes[i].is_ascii_digit() {
            digits += 1;
        }
        i += 1;
    }
    if digits == 0 {
        return None;
    }
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        let mut j = i + 1;
        if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
            j += 1;
        }
        let mut exp_digits = 0;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            exp_digits += 1;
            j += 1;
        }
        if exp_digits > 0 {
            i = j;
        }
    }
    let value: f64 = s[..i].parse().ok()?;
    if !value.is_finite() {
        return None;
    }
    Some((value, s[i..].trim()))
}

/// Parse a single-component numeric value, or `None` for anything
/// else (keywords, `calc()`, lists, unknown units). Units are
/// matched case-insensitively and normalized to lowercase.
fn parse_numeric(s: &str) -> Option<NumericValue> {
    let (value, unit) = split_numeric(s)?;
    let unit = unit.to_ascii_lowercase();
    let known = matches!(
        unit.as_str(),
        "" | "%"
            | "px"
            | "cm"
            | "mm"
            | "q"
            | "in"
            | "pt"
            | "pc"
            | "em"
            | "rem"
            | "ex"
            | "ch"
            | "vw"
            | "vh"
            | "vmin"
            | "vmax"
            | "svw"
            | "svh"
            | "svmin"
            | "svmax"
            | "lvw"
            | "lvh"
            | "lvmin"
            | "lvmax"
            | "dvw"
            | "dvh"
            | "dvmin"
            | "dvmax"
            | "deg"
            | "grad"
            | "rad"
            | "turn"
            | "s"
            | "ms"
            | "fr"
    );
    if !known {
        return None;
    }
    Some(NumericValue { value, unit })
}

/// The canonical family of a unit, plus the factor that converts
/// a value in that unit to the family's canonical unit. Relative
/// lengths (`em`, `vw`, …) are each their own family — they only
/// interpolate with the same unit, since no font or viewport
/// metrics exist at this layer to convert between them. Absolute
/// lengths, angles, and times convert freely within their family
/// (CSS Values 4 fixed ratios: `96px = 1in`, `360deg = 1turn`,
/// `1000ms = 1s`); numbers, percentages, and `fr` are their own
/// families. Returns `None` for a unit `parse_numeric` would have
/// rejected (defensive; callers only pass parsed units).
fn canonical_family(unit: &str) -> Option<(&'static str, f64)> {
    let (family, factor) = match unit {
        "" => ("number", 1.0),
        "%" => ("percent", 1.0),
        "px" => ("length-abs", 1.0),
        "cm" => ("length-abs", 96.0 / 2.54),
        "mm" => ("length-abs", 96.0 / 25.4),
        "q" => ("length-abs", 96.0 / 101.6),
        "in" => ("length-abs", 96.0),
        "pt" => ("length-abs", 96.0 / 72.0),
        "pc" => ("length-abs", 16.0),
        "deg" => ("angle", 1.0),
        "grad" => ("angle", 360.0 / 400.0),
        "rad" => ("angle", 180.0 / std::f64::consts::PI),
        "turn" => ("angle", 360.0),
        "s" => ("time", 1000.0),
        "ms" => ("time", 1.0),
        "fr" => ("flex", 1.0),
        // Every relative length is its own family (see above).
        "em" => ("length-em", 1.0),
        "rem" => ("length-rem", 1.0),
        "ex" => ("length-ex", 1.0),
        "ch" => ("length-ch", 1.0),
        "vw" => ("length-vw", 1.0),
        "vh" => ("length-vh", 1.0),
        "vmin" => ("length-vmin", 1.0),
        "vmax" => ("length-vmax", 1.0),
        "svw" => ("length-svw", 1.0),
        "svh" => ("length-svh", 1.0),
        "svmin" => ("length-svmin", 1.0),
        "svmax" => ("length-svmax", 1.0),
        "lvw" => ("length-lvw", 1.0),
        "lvh" => ("length-lvh", 1.0),
        "lvmin" => ("length-lvmin", 1.0),
        "lvmax" => ("length-lvmax", 1.0),
        "dvw" => ("length-dvw", 1.0),
        "dvh" => ("length-dvh", 1.0),
        "dvmin" => ("length-dvmin", 1.0),
        "dvmax" => ("length-dvmax", 1.0),
        _ => return None,
    };
    Some((family, factor))
}

/// Properties whose computed value is an integer: the interpolated
/// value rounds to the nearest integer (half away from zero),
/// per the CSS rule for `<integer>` interpolation. Everything
/// else keeps its fractional interpolated value.
const INTEGER_PROPS: &[&str] = &["z-index", "order", "column-count", "font-weight"];

/// Interpolate two parsed numeric values at progress `t`, or
/// `None` when they are not interpolable together (different
/// families, e.g. `px` against `%`). A unitless zero adopts the
/// other side's unit — the CSS unitless-zero rule, so `0` to
/// `100px` interpolates as `0px` to `100px` rather than flipping
/// discretely. The result serializes in the `to` side's unit when
/// the families convert (e.g. `90deg` to `0.5turn` at `0.5` is
/// `0.375turn`), and in the shared unit otherwise.
fn interpolate_numeric(from: NumericValue, to: NumericValue, prop: &str, t: f64) -> Option<String> {
    let (from_family, from_factor) = canonical_family(&from.unit)?;
    let (to_family, to_factor) = canonical_family(&to.unit)?;
    let (from_canon, to_canon) = if from_family == to_family {
        (from.value * from_factor, to.value * to_factor)
    } else if from.unit.is_empty() && from.value == 0.0 {
        (0.0, to.value * to_factor)
    } else if to.unit.is_empty() && to.value == 0.0 {
        (from.value * from_factor, 0.0)
    } else {
        return None;
    };
    let mut value = from_canon + (to_canon - from_canon) * t;
    // Serialize back in the `to` unit (for the unitless-zero
    // cases the `to` unit is the adopted one on the zero side
    // too, because the non-zero side's family wins there).
    let out_unit = if from_family == to_family || from.unit.is_empty() {
        to.unit.as_str()
    } else {
        from.unit.as_str()
    };
    let out_factor = canonical_family(out_unit)?.1;
    value /= out_factor;
    if INTEGER_PROPS.contains(&prop) {
        value = value.round();
    }
    if out_unit == "%" {
        Some(format!("{}%", format_number(value)))
    } else if out_unit.is_empty() {
        Some(format_number(value))
    } else {
        Some(format!("{}{}", format_number(value), out_unit))
    }
}

/// An RGBA color: channels `0..=255`, alpha `0..=1`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rgba {
    r: f64,
    g: f64,
    b: f64,
    a: f64,
}

/// Parse one `rgb()` / `rgba()` component: a number `0..=255` or a
/// percentage (clamped). `None` for anything else.
fn parse_rgb_component(s: &str) -> Option<f64> {
    let s = s.trim();
    if let Some(pct) = s.strip_suffix('%') {
        let v: f64 = pct.trim().parse().ok()?;
        if !v.is_finite() {
            return None;
        }
        Some((v / 100.0 * 255.0).clamp(0.0, 255.0))
    } else {
        let v: f64 = s.parse().ok()?;
        if !v.is_finite() {
            return None;
        }
        Some(v.clamp(0.0, 255.0))
    }
}

/// Parse one `rgb()` / `rgba()` alpha: a number `0..=1` or a
/// percentage (clamped). `None` for anything else.
fn parse_rgb_alpha(s: &str) -> Option<f64> {
    let s = s.trim();
    if let Some(pct) = s.strip_suffix('%') {
        let v: f64 = pct.trim().parse().ok()?;
        if !v.is_finite() {
            return None;
        }
        Some((v / 100.0).clamp(0.0, 1.0))
    } else {
        let v: f64 = s.parse().ok()?;
        if !v.is_finite() {
            return None;
        }
        Some(v.clamp(0.0, 1.0))
    }
}

/// Parse a color value: named colors, `#rgb` / `#rrggbb` (+ alpha
/// forms), and `transparent` via the shared CSS color parser, plus
/// `rgb()` / `rgba()` in both comma and space/`/` syntax.
/// `hsl()` / `hwb()` / `lab()` / `color()` are NOT parsed — they
/// fall back to discrete interpolation (see the module header's
/// remaining-work note). `currentcolor` and system colors are not
/// colors at this layer and also fall back.
fn parse_color(s: &str) -> Option<Rgba> {
    let trimmed = s.trim();
    if let Some(c) = bowser_css::parse_css_color(trimmed) {
        return Some(Rgba {
            r: f64::from(c.r),
            g: f64::from(c.g),
            b: f64::from(c.b),
            a: f64::from(c.a) / 255.0,
        });
    }
    let lower = trimmed.to_ascii_lowercase();
    let inner = lower
        .strip_prefix("rgba(")
        .or_else(|| lower.strip_prefix("rgb("))?
        .strip_suffix(')')?;
    if inner.contains(',') {
        // Legacy comma syntax: `rgb(r, g, b)` or `rgba(r, g, b, a)`.
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() != 3 && parts.len() != 4 {
            return None;
        }
        let a = match parts.get(3) {
            Some(a) => parse_rgb_alpha(a)?,
            None => 1.0,
        };
        return Some(Rgba {
            r: parse_rgb_component(parts[0])?,
            g: parse_rgb_component(parts[1])?,
            b: parse_rgb_component(parts[2])?,
            a,
        });
    }
    // Modern space syntax: `rgb(r g b)` or `rgb(r g b / a)`.
    let (channels, alpha) = match inner.split_once('/') {
        Some((ch, a)) => (ch, Some(a)),
        None => (inner, None),
    };
    let parts: Vec<&str> = channels.split_whitespace().collect();
    if parts.len() != 3 {
        return None;
    }
    let a = match alpha {
        Some(a) => parse_rgb_alpha(a)?,
        None => 1.0,
    };
    Some(Rgba {
        r: parse_rgb_component(parts[0])?,
        g: parse_rgb_component(parts[1])?,
        b: parse_rgb_component(parts[2])?,
        a,
    })
}

/// Serialize a color the way `bindings::css_value_to_string` does:
/// opaque as `rgb(r, g, b)`, otherwise `rgba(r, g, b, a)` with
/// channels rounded to integers and alpha in `0..=1`.
fn serialize_color(c: Rgba) -> String {
    let r = c.r.round().clamp(0.0, 255.0) as i64;
    let g = c.g.round().clamp(0.0, 255.0) as i64;
    let b = c.b.round().clamp(0.0, 255.0) as i64;
    let a = c.a.clamp(0.0, 1.0);
    if (a - 1.0).abs() < 1e-9 {
        format!("rgb({r}, {g}, {b})")
    } else {
        format!("rgba({r}, {g}, {b}, {})", format_number(a))
    }
}

/// Interpolate one keyframe segment's endpoints at segment-local
/// progress `t` (already eased): numeric pairs interpolate
/// numerically, color pairs interpolate in alpha-premultiplied
/// sRGBA (CSS Transitions 1 / CSS Color 4: premultiply each
/// channel by its alpha, interpolate premultiplied channels
/// and alpha linearly, then unpremultiply), and anything else
/// returns `None` so the caller falls back to discrete
/// interpolation. Interpolating channels independently (the
/// old code) darkens semi-transparent midpoints: `transparent`
/// to opaque red at `0.5` must be `rgba(255, 0, 0, 0.5)`, not
/// `rgba(128, 0, 0, 0.5)`. An output alpha of `0` has no
/// defined unpremultiplied color and serializes as transparent
/// black.
fn interpolate_pair(from: &str, to: &str, prop: &str, t: f64) -> Option<String> {
    if let (Some(f), Some(tt)) = (parse_numeric(from), parse_numeric(to)) {
        return interpolate_numeric(f, tt, prop, t);
    }
    if let (Some(f), Some(tt)) = (parse_color(from), parse_color(to)) {
        let a = f.a + (tt.a - f.a) * t;
        if a == 0.0 {
            return Some(serialize_color(Rgba {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            }));
        }
        let premult = |c: f64, alpha: f64, d: f64, dalpha: f64| -> f64 {
            (c * alpha + (d * dalpha - c * alpha) * t) / a
        };
        return Some(serialize_color(Rgba {
            r: premult(f.r, f.a, tt.r, tt.a),
            g: premult(f.g, f.a, tt.g, tt.a),
            b: premult(f.b, f.a, tt.b, tt.a),
            a,
        }));
    }
    None
}

/// The animated value of one non-discrete property for one
/// animation at eased progress `p`, or `None` when the animation
/// does not contribute (the property is not keyframed).
/// Segment selection mirrors [`animated_discrete_value`] except
/// at the ends: where the discrete engine holds the endpoint
/// value, this one extrapolates from the nearest segment, so an
/// overshooting easing (eased progress beyond `[0, 1]`) pushes a
/// numeric value past its endpoint instead of clamping — the
/// same behavior CSS transitions show for overshooting easings.
/// Extrapolation happens ONLY for eased progress outside
/// `[0, 1]`: [`property_entries`] pads the entry list to `0`
/// and `1` with the underlying value, so in-range progress
/// always lands inside a real segment. A per-keyframe easing
/// transforms its own segment's local progress, exactly as in
/// the discrete engine. When the segment endpoints are not
/// interpolable together (mismatched units, keywords, `calc()`,
/// multi-component values), the value falls back to the generic
/// discrete flip at `t >= 0.5` — the spec's
/// discrete-interpolation rule for non-interpolable pairs — via
/// [`discrete_segment_value`].
fn animated_numeric_value(
    state: &AnimationState,
    p: f64,
    prop: &str,
    offsets: &[f64],
    base: Option<&str>,
) -> Option<String> {
    let entries = property_entries(state, prop, offsets, base);
    if entries.is_empty() {
        return None;
    }
    if entries.len() == 1 {
        return Some(entries[0].1.to_string());
    }
    let (idx, t_raw) = if p <= entries[0].0 {
        (0, segment_t(p, entries[0].0, entries[1].0))
    } else if p >= entries[entries.len() - 1].0 {
        let n = entries.len();
        (n - 2, segment_t(p, entries[n - 2].0, entries[n - 1].0))
    } else {
        let mut idx = 0;
        while idx + 1 < entries.len() && p >= entries[idx + 1].0 {
            idx += 1;
        }
        (idx, segment_t(p, entries[idx].0, entries[idx + 1].0))
    };
    let (from_value, to_value, from_easing) = (entries[idx].1, entries[idx + 1].1, entries[idx].2);
    let mut t = t_raw;
    if let Some(easing) = from_easing {
        t = eval_timing(easing, t);
    }
    if let Some(value) = interpolate_pair(from_value, to_value, prop, t) {
        return Some(value);
    }
    Some(discrete_segment_value(from_value, to_value, prop, t).to_string())
}

/// Segment-local progress for global progress `p` between
/// offsets `from` and `to`; a zero-width segment evaluates at
/// its end (`1.0`), matching the discrete engine.
fn segment_t(p: f64, from: f64, to: f64) -> f64 {
    if to > from {
        (p - from) / (to - from)
    } else {
        1.0
    }
}

/// CSS shorthand properties. A keyframed shorthand is SKIPPED
/// here (part 1 of the GCS-consistency fix): writing e.g.
/// `background` into the props map while its longhands kept
/// their cascaded values made computed style internally
/// inconsistent (the shorthand no longer synthesized from the
/// longhands it supposedly covers). Shorthand expansion into
/// longhands is documented as not done, so the honest behavior
/// is to leave the shorthand slot alone; its longhands still
/// animate when keyframed directly. Re-synthesis of a shorthand
/// from animated longhands is a separate, bindings-side item.
/// This list is the unambiguous shorthands only — component
/// longhands (`background-color`, `border-top-width`,
/// `background-position`, …) are NOT listed and still animate.
const ANIMATED_SHORTHAND_PROPS: &[&str] = &[
    "animation",
    "background",
    "border",
    "border-block",
    "border-block-color",
    "border-block-style",
    "border-block-width",
    "border-bottom",
    "border-color",
    "border-inline",
    "border-inline-color",
    "border-inline-style",
    "border-inline-width",
    "border-left",
    "border-radius",
    "border-right",
    "border-style",
    "border-top",
    "border-width",
    "columns",
    "column-rule",
    "flex",
    "flex-flow",
    "font",
    "gap",
    "grid",
    "grid-gap",
    "grid-template",
    "inset",
    "inset-block",
    "inset-inline",
    "list-style",
    "margin",
    "margin-block",
    "margin-inline",
    "outline",
    "overflow",
    "overscroll-behavior",
    "padding",
    "padding-block",
    "padding-inline",
    "place-content",
    "place-items",
    "place-self",
    "scroll-margin",
    "scroll-margin-block",
    "scroll-margin-inline",
    "scroll-padding",
    "scroll-padding-block",
    "scroll-padding-inline",
    "text-decoration",
    "transition",
];

/// w84fix: expand a `!important`-blocked set across the
/// logical-to-physical fold, in both directions. `cascade`'s
/// `important_properties` reports the winning longhand's own name
/// and does not apply the fold, so a `margin-inline-start: 0
/// !important` winner leaves `blocked = {margin-inline-start}`
/// while the computed-props map carries the folded value under
/// `margin-left`: a keyframe on `margin-left` would read that
/// folded value as its base and overwrite it, defeating the
/// `!important` winner through its physical alias (CSS Cascade 5:
/// animations sit below all `!important` origins). The reverse
/// holds too: a physical `!important` winner governs the side the
/// logical longhand folds into, so a keyframe on the logical name
/// is blocked as well. The mapping mirrors
/// `cascade::apply_logical_mapping`'s side table (same
/// writing-mode/direction resolution, read from the props map
/// itself — production maps carry every computed property, so
/// `writing-mode`/`direction` are present; absent keys default to
/// horizontal-tb/ltr). Size pairs (`inline-size`/`block-size` vs
/// `width`/`height`) and flow-relative overflow pairs
/// (`overflow-inline`/`overflow-block` vs `overflow-x`/`overflow-y`)
/// fold the same way and expand too.
fn expand_blocked_across_logical_fold(
    blocked: &HashSet<String>,
    props: &HashMap<String, String>,
) -> HashSet<String> {
    let mut out: HashSet<String> = blocked.clone();
    if blocked.is_empty() {
        return out;
    }
    let wm = props
        .get("writing-mode")
        .map(|s| s.as_str())
        .unwrap_or("horizontal-tb");
    let vertical_rl = wm.eq_ignore_ascii_case("vertical-rl");
    let vertical = vertical_rl || wm.eq_ignore_ascii_case("vertical-lr");
    let rtl = props
        .get("direction")
        .map(|s| s.eq_ignore_ascii_case("rtl"))
        .unwrap_or(false);
    // Physical side index for (inline axis?, start?) — the same
    // table as `cascade::logical_side_idx`: 0=top, 1=right,
    // 2=bottom, 3=left.
    let side_idx = |inline_axis: bool, start: bool| -> usize {
        match (vertical, vertical_rl, rtl, inline_axis, start) {
            (false, _, false, true, true) => 3,
            (false, _, false, true, false) => 1,
            (false, _, true, true, true) => 1,
            (false, _, true, true, false) => 3,
            (false, _, _, false, true) => 0,
            (false, _, _, false, false) => 2,
            (true, _, false, true, true) => 0,
            (true, _, false, true, false) => 2,
            (true, _, true, true, true) => 2,
            (true, _, true, true, false) => 0,
            (true, true, _, false, true) => 1,
            (true, true, _, false, false) => 3,
            (true, false, _, false, true) => 3,
            (true, false, _, false, false) => 1,
        }
    };
    const SIDES: [&str; 4] = ["top", "right", "bottom", "left"];
    // (logical name, physical name) pairs for every side group.
    let mut pairs: Vec<(String, String)> = Vec::new();
    let side_groups: &[(&str, [&str; 4])] = &[
        (
            "margin",
            ["margin-top", "margin-right", "margin-bottom", "margin-left"],
        ),
        (
            "padding",
            [
                "padding-top",
                "padding-right",
                "padding-bottom",
                "padding-left",
            ],
        ),
        (
            "scroll-margin",
            [
                "scroll-margin-top",
                "scroll-margin-right",
                "scroll-margin-bottom",
                "scroll-margin-left",
            ],
        ),
        (
            "scroll-padding",
            [
                "scroll-padding-top",
                "scroll-padding-right",
                "scroll-padding-bottom",
                "scroll-padding-left",
            ],
        ),
        ("inset", ["top", "right", "bottom", "left"]),
    ];
    for (prefix, physical) in side_groups {
        for (axis, inline_axis) in [("inline", true), ("block", false)] {
            for (end, start) in [("start", true), ("end", false)] {
                pairs.push((
                    format!("{prefix}-{axis}-{end}"),
                    physical[side_idx(inline_axis, start)].to_string(),
                ));
            }
        }
    }
    for part in ["width", "style", "color"] {
        for (axis, inline_axis) in [("inline", true), ("block", false)] {
            for (end, start) in [("start", true), ("end", false)] {
                pairs.push((
                    format!("border-{axis}-{end}-{part}"),
                    format!("border-{}-{part}", SIDES[side_idx(inline_axis, start)]),
                ));
            }
        }
    }
    // Size and overflow pairs fold by axis, not by side.
    let (inline_size_phys, block_size_phys) = if vertical {
        ("height", "width")
    } else {
        ("width", "height")
    };
    pairs.push(("inline-size".to_string(), inline_size_phys.to_string()));
    pairs.push(("block-size".to_string(), block_size_phys.to_string()));
    let (inline_overflow_phys, block_overflow_phys) = if vertical {
        ("overflow-y", "overflow-x")
    } else {
        ("overflow-x", "overflow-y")
    };
    pairs.push((
        "overflow-inline".to_string(),
        inline_overflow_phys.to_string(),
    ));
    pairs.push((
        "overflow-block".to_string(),
        block_overflow_phys.to_string(),
    ));
    for (logical, physical) in &pairs {
        if blocked.contains(logical) {
            out.insert(physical.clone());
        }
        if blocked.contains(physical) {
            out.insert(logical.clone());
        }
    }
    out
}

/// Apply animated values over a computed-props map, in place.
///
/// W81-B: fills in the worker-A placeholder. For each animation
/// targeting `key` (creation order, so a later `animate()` call
/// wins over an earlier one for the same property), if it
/// [`AnimationState::applies_at`] this instant, its keyframed
/// values replace the cascaded values in `props`: discrete
/// properties via [`animated_discrete_value`], everything else
/// via [`animated_numeric_value`] (numeric / color interpolation,
/// discrete fallback for non-interpolable pairs). Composite is
/// `replace` only in this change: `add` / `accumulate` keyframes
/// are treated as `replace` — numeric compositing against the
/// cascaded base value is a follow-on, not smuggled in here.
/// Called AFTER `transition::apply_transitions` at both
/// `getComputedStyle` surfaces, so an active animation wins over
/// a running transition for the same property. Only properties
/// already present in `props` are written: the production surface
/// exposes every map entry as a named property on the computed
/// style object, so inventing a slot for an unknown keyframe name
/// would leak it there. `blocked` is the set of property names
/// whose cascade winner is `!important` (CSS Cascade 5: animations
/// sit below all `!important` origins), captured by the caller
/// from the cascade BEFORE `apply_transitions` runs; a blocked
/// keyframed property keeps its cascaded value at every progress.
/// The set is expanded across the logical-to-physical fold on entry
/// (see `expand_blocked_across_logical_fold`), so blocking a logical
/// longhand also blocks its physical alias and vice versa.
/// A plain `<div>` animated `none -> auto` therefore still reads
/// `none` (the UA `* { overlay: none !important }` winner blocks
/// it), while the same div with inline `overlay: auto` and no
/// animation also reads `none` (see the module header's note).
pub fn apply_animations(key: &str, props: &mut HashMap<String, String>, blocked: &HashSet<String>) {
    // w84fix: a blocked logical longhand also blocks its physical
    // alias (and vice versa) — see
    // `expand_blocked_across_logical_fold`.
    let blocked = expand_blocked_across_logical_fold(blocked, props);
    let blocked = &blocked;
    let now = Instant::now();
    for state in animations_for_target(key) {
        if !state.applies_at(now) {
            continue;
        }
        let Some(p) = state.eased_progress(now) else {
            continue;
        };
        if state.keyframes.is_empty() {
            continue;
        }
        let offsets = normalize_offsets(&state.keyframes);
        // Distinct keyframed properties, in first-appearance
        // order across the keyframes.
        let mut keyframed: Vec<&str> = Vec::new();
        for kf in &state.keyframes {
            for (name, _) in &kf.properties {
                if !keyframed.contains(&name.as_str()) {
                    keyframed.push(name.as_str());
                }
            }
        }
        for prop in keyframed {
            // w84: a `!important` cascade winner outranks the
            // animation origin entirely — skip before reading the
            // base value, so no segment (discrete or numeric) can
            // overwrite it at any progress.
            if blocked.contains(prop) {
                continue;
            }
            // Shorthands are skipped, not expanded (see
            // `ANIMATED_SHORTHAND_PROPS`); the underlying value
            // below is read BEFORE this animation's overwrite,
            // so endpoint padding in `property_entries` uses the
            // cascaded value for the first animation and the
            // previous animation's output for a later one (the
            // effect-stack underlying value), never this
            // animation's own output.
            if ANIMATED_SHORTHAND_PROPS.contains(&prop) {
                continue;
            }
            let Some(base) = props.get(prop).cloned() else {
                continue;
            };
            let value = if ANIMATED_DISCRETE_PROPS.contains(&prop) {
                animated_discrete_value(&state, p, prop, &offsets, Some(&base))
            } else {
                animated_numeric_value(&state, p, prop, &offsets, Some(&base))
            };
            if let Some(value) = value {
                props.insert(prop.to_string(), value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(duration_ms: f64) -> AnimationOptions {
        AnimationOptions {
            duration_ms,
            ..AnimationOptions::default()
        }
    }

    fn state_with(duration_ms: f64) -> AnimationState {
        AnimationState::new(
            1,
            "prod:test".to_string(),
            vec![Keyframe {
                offset: Some(0.0),
                composite: Composite::Replace,
                easing: None,
                properties: vec![("overlay".to_string(), "auto".to_string())],
            }],
            opts(duration_ms),
        )
    }

    #[test]
    fn camel_to_dashed_converts_and_keeps_specials() {
        assert_eq!(camel_to_dashed("overlay"), "overlay");
        assert_eq!(camel_to_dashed("backgroundColor"), "background-color");
        assert_eq!(camel_to_dashed("cssOffset"), "offset");
        assert_eq!(camel_to_dashed("cssFloat"), "float");
        assert_eq!(camel_to_dashed("--my-prop"), "--my-prop");
        assert_eq!(camel_to_dashed("borderTopWidth"), "border-top-width");
    }

    #[test]
    fn enum_parsing_falls_back_via_caller_default() {
        assert_eq!(Fill::parse("forwards"), Some(Fill::Forwards));
        assert_eq!(Fill::parse("bogus"), None);
        assert_eq!(
            Direction::parse("alternate-reverse"),
            Some(Direction::AlternateReverse)
        );
        assert_eq!(Direction::parse("bogus"), None);
        assert_eq!(Composite::parse("add"), Some(Composite::Add));
        assert_eq!(Composite::parse("bogus"), None);
        assert_eq!(PlayState::Paused.as_str(), "paused");
        assert!(Fill::Both.holds_before() && Fill::Both.holds_after());
        assert!(!Fill::None.holds_before() && !Fill::None.holds_after());
    }

    #[test]
    fn easing_reuses_transition_solver() {
        // The transition solver's harness curve: bezier(0, b, 1, b)
        // returns y at x = 0.5, where b = (8y - 1) / 6; for y = 0.5,
        // b = 0.5.
        let timing = AnimationOptions::easing_from_str("cubic-bezier(0, 0.5, 1, 0.5)");
        assert!((eval_timing(timing, 0.5) - 0.5).abs() < 1e-6);
        assert_eq!(
            AnimationOptions::easing_from_str("not-a-timing-function"),
            Timing::Linear
        );
        assert_eq!(AnimationOptions::easing_from_str("linear"), Timing::Linear);
    }

    #[test]
    fn pause_freezes_and_set_while_paused_sticks() {
        let mut s = state_with(100_000.0);
        let now = Instant::now();
        s.pause(now);
        let held = s.current_time_ms(now).expect("paused has a time");
        assert!(
            (0.0..1000.0).contains(&held),
            "fresh pause holds ~0, got {held}"
        );
        s.set_current_time(now, Some(50_000.0));
        assert_eq!(s.current_time_ms(now), Some(50_000.0));
        assert_eq!(s.play_state, PlayState::Paused);
    }

    #[test]
    fn set_while_running_seeks_and_play_resumes_from_hold() {
        let mut s = state_with(100_000.0);
        let now = Instant::now();
        s.set_current_time(now, Some(50_000.0));
        let got = s.current_time_ms(now).expect("running has a time");
        assert!((got - 50_000.0).abs() < 100.0, "seek to 50000, got {got}");
        s.pause(now);
        s.set_current_time(now, Some(10_000.0));
        s.play(now);
        let resumed = s.current_time_ms(now).expect("playing has a time");
        assert!(
            (10_000.0..11_000.0).contains(&resumed),
            "resume from 10000, got {resumed}"
        );
    }

    #[test]
    fn cancel_reads_null_and_finish_holds_the_end() {
        let mut s = state_with(100_000.0);
        let now = Instant::now();
        s.cancel();
        assert_eq!(s.current_time_ms(now), None);
        assert_eq!(s.play_state, PlayState::Idle);
        let mut t = state_with(100_000.0);
        t.finish(now);
        assert_eq!(t.current_time_ms(now), Some(100_000.0));
        assert_eq!(t.play_state, PlayState::Finished);
        assert_eq!(t.iteration_progress(now), Some(1.0));
    }

    #[test]
    fn progress_applies_direction_and_easing() {
        let mut s = state_with(100_000.0);
        let now = Instant::now();
        s.set_current_time(now, Some(25_000.0));
        assert_eq!(s.iteration_progress(now), Some(0.25));
        assert_eq!(s.eased_progress(now), Some(0.25));
        s.options.direction = Direction::Reverse;
        assert_eq!(s.iteration_progress(now), Some(0.75));
        // Fill: before the delay nothing applies without a
        // backwards fill; after the end nothing without forwards.
        let mut f = state_with(100_000.0);
        f.options.delay_ms = 10_000.0;
        f.options.fill = Fill::None;
        f.set_current_time(now, Some(-5_000.0));
        assert!(!f.applies_at(now));
        f.options.fill = Fill::Backwards;
        assert!(f.applies_at(now));
        f.options.fill = Fill::Forwards;
        f.set_current_time(now, Some(150_000.0));
        assert!(f.applies_at(now));
        f.options.fill = Fill::None;
        assert!(!f.applies_at(now));
    }

    #[test]
    fn registry_groups_by_target_key_in_creation_order() {
        let a = create_animation("prod:elem-a".to_string(), vec![], opts(1000.0));
        let b = create_animation("prod:elem-b".to_string(), vec![], opts(1000.0));
        let c = create_animation("prod:elem-a".to_string(), vec![], opts(2000.0));
        assert!(a < b && b < c, "ids ascend in creation order");
        let for_a = animations_for_target("prod:elem-a");
        assert_eq!(for_a.len(), 2);
        assert_eq!(for_a[0].id, a);
        assert_eq!(for_a[1].id, c);
        assert!(animations_for_target("prod:nope").is_empty());
        // The testcommon shape through the registry: pause, then
        // set 50000 on a 100000ms forwards animation, reads back.
        let id = create_animation(
            "parse:9:3".to_string(),
            vec![],
            AnimationOptions {
                duration_ms: 100_000.0,
                fill: Fill::Forwards,
                ..AnimationOptions::default()
            },
        );
        assert!(pause(id));
        assert!(set_current_time(id, Some(50_000.0)));
        assert_eq!(current_time(id), Some(Some(50_000.0)));
        assert_eq!(play_state_str(id), Some("paused"));
        assert!(cancel(id));
        assert_eq!(current_time(id), Some(None));
        assert_eq!(current_time(u64::MAX), None);
        // Four registry entries were created in this test thread
        // (three above plus this one); cancelling keeps the entry.
        assert_eq!(animation_count(), 4);
    }

    fn kf(offset: Option<f64>, overlay: &str) -> Keyframe {
        Keyframe {
            offset,
            composite: Composite::Replace,
            easing: None,
            properties: vec![("overlay".to_string(), overlay.to_string())],
        }
    }

    /// Drive one animation to a paused hold time and read back
    /// the overlay `apply_animations` writes over `cascaded`.
    /// Passes an empty blocked set: pure segment coverage, no
    /// `!important` winner in play (see the w84 blocked tests
    /// below for the blocked half).
    fn animated_overlay(
        key: &str,
        keyframes: Vec<Keyframe>,
        options: AnimationOptions,
        hold_ms: f64,
    ) -> String {
        let id = create_animation(key.to_string(), keyframes, options);
        assert!(pause(id));
        assert!(set_current_time(id, Some(hold_ms)));
        let mut props = HashMap::from([("overlay".to_string(), "none".to_string())]);
        apply_animations(key, &mut props, &HashSet::new());
        props["overlay"].clone()
    }

    fn waapi_opts(easing: &str) -> AnimationOptions {
        AnimationOptions {
            duration_ms: 100_000.0,
            fill: Fill::Forwards,
            easing: AnimationOptions::easing_from_str(easing),
            ..AnimationOptions::default()
        }
    }

    #[test]
    fn normalize_offsets_distributes_and_clamps() {
        let kfs = vec![
            kf(None, "none"),
            kf(None, "auto"),
            kf(Some(0.5), "none"),
            kf(None, "auto"),
            kf(None, "none"),
        ];
        assert_eq!(normalize_offsets(&kfs), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        // Single unspecified keyframe reads offset 1 (the
        // end-state rule: the spec's first-offset rule applies
        // only with more than one keyframe, the last-offset
        // rule applies unconditionally).
        assert_eq!(normalize_offsets(&[kf(None, "auto")]), vec![1.0]);
        // A decreasing author offset is raised to its predecessor.
        let kfs = vec![kf(Some(0.8), "none"), kf(Some(0.2), "auto")];
        assert_eq!(normalize_offsets(&kfs), vec![0.8, 0.8]);
        assert!(normalize_offsets(&[]).is_empty());
    }

    #[test]
    fn apply_animations_eased_progress_and_asymmetric_flips() {
        // The testcommon shape: paused at 50000 of 100000 (raw
        // 0.5), one easing per row, mirroring transition.rs's
        // `overlay_flip_rules_are_asymmetric` cases.
        let cases: &[(&str, &str, &str)] = &[
            // (easing, auto->none, none->auto)
            ("linear", "auto", "auto"),
            ("steps(1, start)", "none", "auto"),
            ("steps(1, end)", "auto", "none"),
            ("cubic-bezier(0, 0.5, 1, 0.5)", "auto", "auto"),
        ];
        for (i, (easing, auto_none, none_auto)) in cases.iter().enumerate() {
            let key = format!("prod:flip-{i}");
            let got = animated_overlay(
                &key,
                vec![kf(Some(0.0), "auto"), kf(Some(1.0), "none")],
                waapi_opts(easing),
                50_000.0,
            );
            assert_eq!(got, *auto_none, "auto->none under {easing}");
            let key = format!("prod:flip-{i}-rev");
            let got = animated_overlay(
                &key,
                vec![kf(Some(0.0), "none"), kf(Some(1.0), "auto")],
                waapi_opts(easing),
                50_000.0,
            );
            assert_eq!(got, *none_auto, "none->auto under {easing}");
        }
        // Bezier shapes `createEasing` emits for other `at`
        // values: at 0.1 maps 0.5 -> 0.1 (> 0, < 1), at 0.9 maps
        // 0.5 -> 0.9, at -1 maps 0.5 -> -1, at 1.5 maps 0.5 -> 1.5.
        let key = "prod:flip-bezier-01";
        assert_eq!(
            animated_overlay(
                key,
                vec![kf(Some(0.0), "none"), kf(Some(1.0), "auto")],
                waapi_opts("cubic-bezier(0, -0.03333333333333333, 1, -0.03333333333333333)"),
                50_000.0,
            ),
            "auto",
            "none->auto flips as soon as eased > 0"
        );
        let key = "prod:flip-bezier-neg";
        assert_eq!(
            animated_overlay(
                key,
                vec![kf(Some(0.0), "none"), kf(Some(1.0), "auto")],
                waapi_opts("cubic-bezier(0, -1.5, 1, -1.5)"),
                50_000.0,
            ),
            "none",
            "none->auto does not flip at eased -1"
        );
        let key = "prod:flip-bezier-15";
        assert_eq!(
            animated_overlay(
                key,
                vec![kf(Some(0.0), "auto"), kf(Some(1.0), "none")],
                waapi_opts("cubic-bezier(0, 1.8333333333333333, 1, 1.8333333333333333)"),
                50_000.0,
            ),
            "none",
            "auto->none flips once eased >= 1"
        );
    }

    #[test]
    fn display_segment_value_mirrors_overlay_asymmetry() {
        // w82: WAAPI `display` uses the same `none` vs shown
        // asymmetry as `overlay` (direct unit test of the
        // segment function, no animation registry needed).
        assert_eq!(
            discrete_segment_value("block", "none", "display", 0.5),
            "block"
        );
        assert_eq!(
            discrete_segment_value("block", "none", "display", 1.0),
            "none"
        );
        assert_eq!(
            discrete_segment_value("none", "block", "display", 0.0),
            "none"
        );
        assert_eq!(
            discrete_segment_value("none", "block", "display", 0.5),
            "block"
        );
        assert_eq!(
            discrete_segment_value("block", "flex", "display", 0.4),
            "block"
        );
        assert_eq!(
            discrete_segment_value("block", "flex", "display", 0.5),
            "flex"
        );
    }

    #[test]
    fn apply_animations_fill_forwards_holds_end_and_fill_none_does_not() {
        // Past the end (150000 of 100000): forwards holds the
        // end value, which differs from the cascaded `none` for
        // a none->auto animation; fill none leaves the cascade.
        let key = "prod:fill-forwards";
        assert_eq!(
            animated_overlay(
                key,
                vec![kf(Some(0.0), "none"), kf(Some(1.0), "auto")],
                waapi_opts("linear"),
                150_000.0,
            ),
            "auto"
        );
        let key = "prod:fill-none";
        let mut opts = waapi_opts("linear");
        opts.fill = Fill::None;
        assert_eq!(
            animated_overlay(
                key,
                vec![kf(Some(0.0), "none"), kf(Some(1.0), "auto")],
                opts,
                150_000.0
            ),
            "none"
        );
        // Cancelled (idle) animations never apply, even with a
        // forwards fill.
        let key = "prod:fill-cancelled";
        let id = create_animation(
            key.to_string(),
            vec![kf(Some(0.0), "none"), kf(Some(1.0), "auto")],
            waapi_opts("linear"),
        );
        assert!(cancel(id));
        let mut props = HashMap::from([("overlay".to_string(), "none".to_string())]);
        apply_animations(key, &mut props, &HashSet::new());
        assert_eq!(props["overlay"], "none");
    }

    #[test]
    fn apply_animations_overrides_on_both_surfaces_and_later_wins() {
        // Same discrete logic under a `parse:` key as under a
        // `prod:` key (the function itself is key-agnostic; the
        // hook sites supply the surface's key).
        let key = "parse:7:3";
        assert_eq!(
            animated_overlay(
                key,
                vec![kf(Some(0.0), "none"), kf(Some(1.0), "auto")],
                waapi_opts("linear"),
                50_000.0,
            ),
            "auto"
        );
        // A later animate() call wins over an earlier one, and
        // an animation that does not keyframe overlay leaves the
        // cascaded value alone.
        let key = "prod:later-wins";
        let first = create_animation(
            key.to_string(),
            vec![kf(Some(0.0), "auto"), kf(Some(1.0), "none")],
            waapi_opts("linear"),
        );
        assert!(pause(first));
        assert!(set_current_time(first, Some(50_000.0)));
        let second = create_animation(
            key.to_string(),
            vec![kf(Some(0.0), "none"), kf(Some(1.0), "auto")],
            waapi_opts("linear"),
        );
        assert!(pause(second));
        assert!(set_current_time(second, Some(50_000.0)));
        let mut props = HashMap::from([("overlay".to_string(), "none".to_string())]);
        apply_animations(key, &mut props, &HashSet::new());
        assert_eq!(props["overlay"], "auto");
        // Omitted offsets distribute: no-offset keyframes are
        // 0 and 1, so the same flip applies.
        let key = "prod:omitted-offsets";
        assert_eq!(
            animated_overlay(
                key,
                vec![kf(None, "none"), kf(None, "auto")],
                waapi_opts("linear"),
                50_000.0,
            ),
            "auto"
        );
        // A single keyframe is the end state (offset 1, start
        // padded with the underlying value); at mid-progress
        // the discrete none->auto flip has already moved to it.
        let key = "prod:single-kf";
        assert_eq!(
            animated_overlay(key, vec![kf(None, "auto")], waapi_opts("linear"), 50_000.0),
            "auto"
        );
    }

    /// Blocked variant of `animated_overlay`: `blocked` names the
    /// `!important`-won properties, `cascaded` is the pre-animation
    /// value (the UA-collapsed `none` for a plain `<div>`, `auto`
    /// for a top-layer element).
    fn animated_overlay_blocked(
        key: &str,
        keyframes: Vec<Keyframe>,
        options: AnimationOptions,
        hold_ms: f64,
        blocked: &[&str],
        cascaded: &str,
    ) -> String {
        let id = create_animation(key.to_string(), keyframes, options);
        assert!(pause(id));
        assert!(set_current_time(id, Some(hold_ms)));
        let mut props = HashMap::from([("overlay".to_string(), cascaded.to_string())]);
        let blocked: HashSet<String> = blocked.iter().map(|p| p.to_string()).collect();
        apply_animations(key, &mut props, &blocked);
        props["overlay"].clone()
    }

    /// Blocked variant of `animated_value` (defined below): same
    /// drive, with an explicit blocked set.
    fn animated_value_blocked(
        key: &str,
        keyframes: Vec<Keyframe>,
        options: AnimationOptions,
        hold_ms: f64,
        prop: &str,
        cascaded: &str,
        blocked: &[&str],
    ) -> String {
        animated_value_blocked_with(
            key,
            keyframes,
            options,
            hold_ms,
            prop,
            cascaded,
            blocked,
            &[],
        )
    }

    /// `animated_value_blocked` plus extra props entries (e.g.
    /// `writing-mode` / `direction`, which the fold expansion
    /// reads to resolve logical longhands). Test-only mirror of
    /// the 7-arg helper above, plus the extra-entries slice.
    #[allow(clippy::too_many_arguments)]
    fn animated_value_blocked_with(
        key: &str,
        keyframes: Vec<Keyframe>,
        options: AnimationOptions,
        hold_ms: f64,
        prop: &str,
        cascaded: &str,
        blocked: &[&str],
        extra: &[(&str, &str)],
    ) -> String {
        let id = create_animation(key.to_string(), keyframes, options);
        assert!(pause(id));
        assert!(set_current_time(id, Some(hold_ms)));
        let mut props = HashMap::from([(prop.to_string(), cascaded.to_string())]);
        for (k, v) in extra {
            props.insert(k.to_string(), v.to_string());
        }
        let blocked: HashSet<String> = blocked.iter().map(|p| p.to_string()).collect();
        apply_animations(key, &mut props, &blocked);
        props[prop].clone()
    }

    #[test]
    fn blocked_important_overlay_stays_none_at_all_wpt_progress_points() {
        // w84: mirrors the 13 failing subtests of
        // overlay-interpolation.html (see the css-position.ini
        // block): a plain `<div>` whose `overlay` winner is the UA
        // `none !important` rule must read `none` at every
        // progress, whichever direction the keyframes run.
        // Progress is hold_ms / 100_000 with `waapi_opts`
        // (linear, fill forwards).
        let cases: &[(&str, &str, &str, f64)] = &[
            // (from, to, key suffix, progress)
            ("auto", "none", "an-neg", -1.0),
            ("auto", "none", "an-0", 0.0),
            ("auto", "none", "an-01", 0.1),
            ("auto", "none", "an-09", 0.9),
            ("none", "auto", "na-01", 0.1),
            ("none", "auto", "na-09", 0.9),
            ("none", "auto", "na-1", 1.0),
            ("none", "auto", "na-15", 1.5),
            ("auto", "auto", "aa-neg", -1.0),
            ("auto", "auto", "aa-0", 0.0),
            ("auto", "auto", "aa-05", 0.5),
            ("auto", "auto", "aa-1", 1.0),
            ("auto", "auto", "aa-15", 1.5),
        ];
        assert_eq!(cases.len(), 13);
        for (from, to, suffix, progress) in cases {
            let key = format!("w84:blocked-{suffix}");
            let got = animated_overlay_blocked(
                &key,
                vec![kf(Some(0.0), from), kf(Some(1.0), to)],
                waapi_opts("linear"),
                progress * 100_000.0,
                &["overlay"],
                "none",
            );
            assert_eq!(
                got, "none",
                "blocked overlay at progress {progress} ({from}->{to})"
            );
        }
    }

    #[test]
    fn unblocked_author_normal_overlay_still_animates() {
        // The mirror of the blocked case: no `!important` winner
        // (e.g. a top-layer element whose UA `none` rule is
        // exempted, author-normal `auto` winning) animates freely.
        // auto->none flips only at t >= 1 (asymmetric discrete
        // rule, kept untouched), so drive to the end of the
        // interval; none->auto flips at t > 0, so mid-interval
        // suffices there.
        let key = "w84:unblocked-an";
        assert_eq!(
            animated_overlay_blocked(
                key,
                vec![kf(Some(0.0), "auto"), kf(Some(1.0), "none")],
                waapi_opts("linear"),
                100_000.0,
                &[],
                "auto",
            ),
            "none"
        );
        let key = "w84:unblocked-na";
        assert_eq!(
            animated_overlay_blocked(
                key,
                vec![kf(Some(0.0), "none"), kf(Some(1.0), "auto")],
                waapi_opts("linear"),
                50_000.0,
                &[],
                "auto",
            ),
            "auto"
        );
        // w84fix: controls for the out-of-range progress points
        // the blocked test covers (-1.0, 1.5) and for auto->auto.
        // The blocked test's block is unconditional, so on its own
        // it cannot distinguish "blocked at every progress" from
        // "the engine never applies there"; these unblocked cases
        // prove the engine does evaluate there, and differ from the
        // blocked `none` at every one of them: auto->none still
        // reads its from-value before the interval, none->auto
        // holds its end value past it (fill forwards), and
        // auto->auto reads `auto` at both ends.
        let cases: &[(&str, &str, &str, f64, &str)] = &[
            ("auto", "none", "an-neg", -1.0, "auto"),
            ("none", "auto", "na-15", 1.5, "auto"),
            ("auto", "auto", "aa-neg", -1.0, "auto"),
            ("auto", "auto", "aa-15", 1.5, "auto"),
        ];
        for (from, to, suffix, progress, expected) in cases {
            let key = format!("w84fix:unblocked-{suffix}");
            assert_eq!(
                animated_overlay_blocked(
                    &key,
                    vec![kf(Some(0.0), from), kf(Some(1.0), to)],
                    waapi_opts("linear"),
                    progress * 100_000.0,
                    &[],
                    "auto",
                ),
                *expected,
                "unblocked overlay at progress {progress} ({from}->{to})"
            );
        }
    }

    #[test]
    fn blocked_logical_longhand_blocks_physical_keyframe_and_reverse() {
        // w84fix: the logical-to-physical fold bypass. A
        // `margin-inline-start !important` winner blocks a keyframe
        // on the physical alias it folds into (horizontal-tb, ltr:
        // `margin-left`), and a physical winner blocks a keyframe on
        // the logical name — in both cases the animated value must
        // not overwrite the folded cascaded value (CSS Cascade 5).
        // Each blocked case has an unblocked control proving the
        // engine would otherwise move the value.
        let kfs = |prop: &str| {
            vec![
                kf_prop(Some(0.0), prop, "0px"),
                kf_prop(Some(1.0), prop, "100px"),
            ]
        };
        // Logical blocked, physical keyframed (ltr default).
        let key = "w84fix:blocked-logical-physical";
        assert_eq!(
            animated_value_blocked(
                key,
                kfs("margin-left"),
                waapi_opts("linear"),
                50_000.0,
                "margin-left",
                "0px",
                &["margin-inline-start"],
            ),
            "0px"
        );
        let key = "w84fix:unblocked-logical-physical";
        assert_eq!(
            animated_value_blocked(
                key,
                kfs("margin-left"),
                waapi_opts("linear"),
                50_000.0,
                "margin-left",
                "0px",
                &[],
            ),
            "50px"
        );
        // Reverse: physical blocked, logical keyframed.
        let key = "w84fix:blocked-physical-logical";
        assert_eq!(
            animated_value_blocked(
                key,
                kfs("margin-inline-start"),
                waapi_opts("linear"),
                50_000.0,
                "margin-inline-start",
                "0px",
                &["margin-left"],
            ),
            "0px"
        );
        // Direction-sensitive, not blanket: under rtl the
        // inline-start alias is `margin-right`, so `margin-left`
        // still animates while `margin-right` stays blocked.
        let key = "w84fix:blocked-logical-rtl-right";
        assert_eq!(
            animated_value_blocked_with(
                key,
                kfs("margin-right"),
                waapi_opts("linear"),
                50_000.0,
                "margin-right",
                "0px",
                &["margin-inline-start"],
                &[("direction", "rtl")],
            ),
            "0px"
        );
        let key = "w84fix:unblocked-logical-rtl-left";
        assert_eq!(
            animated_value_blocked_with(
                key,
                kfs("margin-left"),
                waapi_opts("linear"),
                50_000.0,
                "margin-left",
                "0px",
                &["margin-inline-start"],
                &[("direction", "rtl")],
            ),
            "50px"
        );
        // Size fold: `inline-size` blocks `width` in horizontal
        // modes and `height` in vertical ones.
        let key = "w84fix:blocked-inline-size-width";
        assert_eq!(
            animated_value_blocked(
                key,
                kfs("width"),
                waapi_opts("linear"),
                50_000.0,
                "width",
                "0px",
                &["inline-size"],
            ),
            "0px"
        );
        let key = "w84fix:blocked-inline-size-height-vertical";
        assert_eq!(
            animated_value_blocked_with(
                key,
                kfs("height"),
                waapi_opts("linear"),
                50_000.0,
                "height",
                "0px",
                &["inline-size"],
                &[("writing-mode", "vertical-rl")],
            ),
            "0px"
        );
    }

    #[test]
    fn blocked_important_numeric_property_stays_cascaded() {
        // The block is general, not overlay-only: an author
        // `width: 50px !important` winner is not interpolated.
        let key = "w84:blocked-width";
        assert_eq!(
            animated_value_blocked(
                key,
                vec![
                    kf_prop(Some(0.0), "width", "0px"),
                    kf_prop(Some(1.0), "width", "200px"),
                ],
                waapi_opts("linear"),
                50_000.0,
                "width",
                "50px",
                &["width"],
            ),
            "50px"
        );
        // Control: the same animation unblocked interpolates.
        let key = "w84:unblocked-width";
        assert_eq!(
            animated_value_blocked(
                key,
                vec![
                    kf_prop(Some(0.0), "width", "0px"),
                    kf_prop(Some(1.0), "width", "200px"),
                ],
                waapi_opts("linear"),
                50_000.0,
                "width",
                "50px",
                &[],
            ),
            "100px"
        );
    }

    #[test]
    fn top_layer_ua_auto_overlay_stays_auto_under_animation() {
        // Popover (`:popover-open` UA `auto !important`) and modal
        // dialog (`dialog:modal` UA `auto !important`) winners are
        // `!important` too, so an animation towards `none` cannot
        // pull a shown top-layer element out of the top layer.
        for suffix in ["popover", "modal"] {
            let key = format!("w84:top-layer-{suffix}");
            assert_eq!(
                animated_overlay_blocked(
                    &key,
                    vec![kf(Some(0.0), "auto"), kf(Some(1.0), "none")],
                    waapi_opts("linear"),
                    50_000.0,
                    &["overlay"],
                    "auto",
                ),
                "auto",
                "top-layer {suffix} overlay must stay auto"
            );
        }
    }

    fn kf_prop(offset: Option<f64>, prop: &str, value: &str) -> Keyframe {
        Keyframe {
            offset,
            composite: Composite::Replace,
            easing: None,
            properties: vec![(prop.to_string(), value.to_string())],
        }
    }

    /// Drive one animation to a paused hold time and read back
    /// the animated value `apply_animations` writes over
    /// `cascaded` for `prop`.
    fn animated_value(
        key: &str,
        keyframes: Vec<Keyframe>,
        options: AnimationOptions,
        hold_ms: f64,
        prop: &str,
        cascaded: &str,
    ) -> String {
        let id = create_animation(key.to_string(), keyframes, options);
        assert!(pause(id));
        assert!(set_current_time(id, Some(hold_ms)));
        let mut props = HashMap::from([(prop.to_string(), cascaded.to_string())]);
        apply_animations(key, &mut props, &HashSet::new());
        props[prop].clone()
    }

    fn num(s: &str) -> f64 {
        s.trim_end_matches(['p', 'x', '%'])
            .trim_end_matches("deg")
            .parse()
            .expect("test value parses as a number")
    }

    #[test]
    fn numeric_interpolation_at_progress_0_half_1() {
        // Length: 0px -> 100px.
        let key = "prod:num-length";
        let kfs = || {
            vec![
                kf_prop(Some(0.0), "width", "0px"),
                kf_prop(Some(1.0), "width", "100px"),
            ]
        };
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 0.0, "width", "7px"),
            "0px"
        );
        let key = "prod:num-length-half";
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 50_000.0, "width", "7px"),
            "50px"
        );
        let key = "prod:num-length-end";
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 100_000.0, "width", "7px"),
            "100px"
        );
        // Bare number: opacity 0 -> 1.
        let key = "prod:num-opacity";
        let kfs = || {
            vec![
                kf_prop(Some(0.0), "opacity", "0"),
                kf_prop(Some(1.0), "opacity", "1"),
            ]
        };
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 0.0, "opacity", "0.3"),
            "0"
        );
        let key = "prod:num-opacity-half";
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 50_000.0, "opacity", "0.3"),
            "0.5"
        );
        let key = "prod:num-opacity-end";
        assert_eq!(
            animated_value(
                key,
                kfs(),
                waapi_opts("linear"),
                100_000.0,
                "opacity",
                "0.3"
            ),
            "1"
        );
        // Percentage: 0% -> 50%.
        let key = "prod:num-percent";
        let kfs = || {
            vec![
                kf_prop(Some(0.0), "width", "0%"),
                kf_prop(Some(1.0), "width", "50%"),
            ]
        };
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 0.0, "width", "9%"),
            "0%"
        );
        let key = "prod:num-percent-half";
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 50_000.0, "width", "9%"),
            "25%"
        );
        let key = "prod:num-percent-end";
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 100_000.0, "width", "9%"),
            "50%"
        );
    }

    #[test]
    fn numeric_interpolation_easing_changes_midpoint() {
        // Linear at 0.5 is 50px; ease-in at the same raw progress
        // is the eased value (about 31.6px), proving the effect
        // easing — not raw progress — drives interpolation.
        let key = "prod:num-easing";
        let got = animated_value(
            key,
            vec![
                kf_prop(Some(0.0), "width", "0px"),
                kf_prop(Some(1.0), "width", "100px"),
            ],
            waapi_opts("ease-in"),
            50_000.0,
            "width",
            "7px",
        );
        assert_ne!(got, "50px", "ease-in midpoint must differ from linear");
        let expected = eval_timing(AnimationOptions::easing_from_str("ease-in"), 0.5) * 100.0;
        assert!(
            (num(&got) - expected).abs() < 1.0,
            "ease-in midpoint {got} should be near {expected}"
        );
    }

    #[test]
    fn numeric_interpolation_fill_forwards_holds_end() {
        let key = "prod:num-fill-forwards";
        assert_eq!(
            animated_value(
                key,
                vec![
                    kf_prop(Some(0.0), "width", "0px"),
                    kf_prop(Some(1.0), "width", "100px"),
                ],
                waapi_opts("linear"),
                150_000.0,
                "width",
                "7px",
            ),
            "100px"
        );
        let key = "prod:num-fill-none";
        let mut opts = waapi_opts("linear");
        opts.fill = Fill::None;
        assert_eq!(
            animated_value(
                key,
                vec![
                    kf_prop(Some(0.0), "width", "0px"),
                    kf_prop(Some(1.0), "width", "100px"),
                ],
                opts,
                150_000.0,
                "width",
                "7px",
            ),
            "7px"
        );
    }

    #[test]
    fn numeric_interpolation_colors_angles_times() {
        // Colors: named -> named, and rgb() -> hex, both
        // serialize as rgb() at the midpoint (127.5 rounds to 128).
        let key = "prod:num-color-named";
        assert_eq!(
            animated_value(
                key,
                vec![
                    kf_prop(Some(0.0), "background-color", "red"),
                    kf_prop(Some(1.0), "background-color", "blue"),
                ],
                waapi_opts("linear"),
                50_000.0,
                "background-color",
                "rgb(0, 0, 0)",
            ),
            "rgb(128, 0, 128)"
        );
        let key = "prod:num-color-forms";
        assert_eq!(
            animated_value(
                key,
                vec![
                    kf_prop(Some(0.0), "background-color", "rgb(255, 0, 0)"),
                    kf_prop(Some(1.0), "background-color", "#0000ff"),
                ],
                waapi_opts("linear"),
                50_000.0,
                "background-color",
                "rgb(0, 0, 0)",
            ),
            "rgb(128, 0, 128)"
        );
        // Angles: same-unit, and mixed-unit (result in the `to`
        // unit: 90deg -> 0.5turn at 0.5 is 135deg = 0.375turn).
        let key = "prod:num-angle";
        assert_eq!(
            animated_value(
                key,
                vec![
                    kf_prop(Some(0.0), "rotate", "0deg"),
                    kf_prop(Some(1.0), "rotate", "180deg"),
                ],
                waapi_opts("linear"),
                50_000.0,
                "rotate",
                "0deg",
            ),
            "90deg"
        );
        let key = "prod:num-angle-mixed";
        assert_eq!(
            animated_value(
                key,
                vec![
                    kf_prop(Some(0.0), "rotate", "90deg"),
                    kf_prop(Some(1.0), "rotate", "0.5turn"),
                ],
                waapi_opts("linear"),
                50_000.0,
                "rotate",
                "0deg",
            ),
            "0.375turn"
        );
        // Times: 0s -> 2000ms at 0.5 is 1000ms (serialized in
        // the `to` unit).
        let key = "prod:num-time";
        assert_eq!(
            animated_value(
                key,
                vec![
                    kf_prop(Some(0.0), "animation-duration", "0s"),
                    kf_prop(Some(1.0), "animation-duration", "2000ms"),
                ],
                waapi_opts("linear"),
                50_000.0,
                "animation-duration",
                "0s",
            ),
            "1000ms"
        );
        // Unitless zero adopts the other side's unit.
        let key = "prod:num-zero";
        assert_eq!(
            animated_value(
                key,
                vec![
                    kf_prop(Some(0.0), "width", "0"),
                    kf_prop(Some(1.0), "width", "100px"),
                ],
                waapi_opts("linear"),
                50_000.0,
                "width",
                "7px",
            ),
            "50px"
        );
        // Integer properties round: z-index 0 -> 10 at 0.25 is
        // 2.5, rounded half away from zero to 3.
        let key = "prod:num-integer";
        assert_eq!(
            animated_value(
                key,
                vec![
                    kf_prop(Some(0.0), "z-index", "0"),
                    kf_prop(Some(1.0), "z-index", "10"),
                ],
                waapi_opts("linear"),
                25_000.0,
                "z-index",
                "1",
            ),
            "3"
        );
    }

    #[test]
    fn numeric_interpolation_segments_and_fallbacks() {
        // Three keyframes: each segment interpolates on its own.
        let key = "prod:num-segments";
        let kfs = || {
            vec![
                kf_prop(Some(0.0), "width", "0px"),
                kf_prop(Some(0.5), "width", "100px"),
                kf_prop(Some(1.0), "width", "200px"),
            ]
        };
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 25_000.0, "width", "7px"),
            "50px"
        );
        let key = "prod:num-segments-2";
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 75_000.0, "width", "7px"),
            "150px"
        );
        // Mismatched families (px against %) are not interpolable:
        // the generic discrete flip applies (from before 0.5, to
        // at and after 0.5), not a bogus number.
        let key = "prod:num-mismatch-early";
        let kfs = || {
            vec![
                kf_prop(Some(0.0), "width", "0px"),
                kf_prop(Some(1.0), "width", "100%"),
            ]
        };
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 25_000.0, "width", "7px"),
            "0px"
        );
        let key = "prod:num-mismatch-late";
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 75_000.0, "width", "7px"),
            "100%"
        );
        // A per-keyframe easing transforms its own segment: a
        // steps(1, end) start keyframe holds the start value
        // through the segment midpoint, steps(1, start) jumps to
        // the end value immediately.
        let mut start_end = kf_prop(Some(0.0), "width", "0px");
        start_end.easing = Some(AnimationOptions::easing_from_str("steps(1, end)"));
        let key = "prod:num-kf-easing-end";
        assert_eq!(
            animated_value(
                key,
                vec![start_end, kf_prop(Some(1.0), "width", "100px")],
                waapi_opts("linear"),
                50_000.0,
                "width",
                "7px",
            ),
            "0px"
        );
        let mut start_start = kf_prop(Some(0.0), "width", "0px");
        start_start.easing = Some(AnimationOptions::easing_from_str("steps(1, start)"));
        let key = "prod:num-kf-easing-start";
        assert_eq!(
            animated_value(
                key,
                vec![start_start, kf_prop(Some(1.0), "width", "100px")],
                waapi_opts("linear"),
                50_000.0,
                "width",
                "7px",
            ),
            "100px"
        );
        // An overshooting easing extrapolates past the endpoint
        // instead of clamping (see animated_numeric_value).
        let key = "prod:num-overshoot";
        let got = animated_value(
            key,
            vec![
                kf_prop(Some(0.0), "width", "0px"),
                kf_prop(Some(1.0), "width", "100px"),
            ],
            waapi_opts("cubic-bezier(0, 2, 1, 2)"),
            50_000.0,
            "width",
            "7px",
        );
        assert!(
            num(&got) > 100.0,
            "overshooting easing must extrapolate past 100px, got {got}"
        );
    }

    #[test]
    fn shift_ms_huge_finite_time_does_not_panic() {
        // The JS H1 case: `anim.currentTime = 1e25` used to
        // panic inside `Duration::from_secs_f64` before any
        // saturating `checked_add`/`checked_sub` ran.
        let now = Instant::now();
        let fwd = shift_ms(now, 1e25);
        assert!(fwd.checked_duration_since(now).is_some());
        let back = shift_ms(now, -1e25);
        assert!(now.checked_duration_since(back).is_some());
        // Paused (hold-time) path stores the huge value exactly.
        let mut s = state_with(100_000.0);
        s.pause(now);
        s.set_current_time(now, Some(1e25));
        assert_eq!(s.current_time_ms(now), Some(1e25));
        // Running path saturates instead of panicking: the read
        // is finite and positive (the clamped bound, ~1e21 ms),
        // not the literal 1e25 and not a collapse to 0.
        let mut r = state_with(100_000.0);
        r.set_current_time(now, Some(1e25));
        let got = r.current_time_ms(now).expect("running has a time");
        assert!(
            got.is_finite() && got > 0.0,
            "saturated huge seek, got {got}"
        );
    }

    #[test]
    fn negative_current_time_reads_back_negative() {
        let now = Instant::now();
        let mut s = state_with(100_000.0);
        s.set_current_time(now, Some(-5_000.0));
        let got = s.current_time_ms(now).expect("running has a time");
        assert!(
            (got + 5_000.0).abs() < 1.0,
            "negative seek reads back, got {got}"
        );
        let mut p = state_with(100_000.0);
        p.pause(now);
        p.set_current_time(now, Some(-5_000.0));
        assert_eq!(p.current_time_ms(now), Some(-5_000.0));
    }

    #[test]
    fn endpoint_padding_first_offset_last_offset_and_missing_property() {
        // First offset > 0: progress 0 holds the underlying
        // value (7px), not an extrapolation of the first
        // segment backwards (which used to read -100px).
        let key = "prod:pad-first-start";
        let kfs = || {
            vec![
                kf_prop(Some(0.5), "width", "0px"),
                kf_prop(Some(1.0), "width", "100px"),
            ]
        };
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 0.0, "width", "7px"),
            "7px"
        );
        let key = "prod:pad-first-mid";
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 25_000.0, "width", "7px"),
            "3.5px"
        );
        let key = "prod:pad-first-end";
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 75_000.0, "width", "7px"),
            "50px"
        );
        // Last offset < 1: progress 1 returns to the underlying
        // value instead of holding the last keyframe.
        let key = "prod:pad-last-end";
        let kfs = || {
            vec![
                kf_prop(Some(0.0), "width", "0px"),
                kf_prop(Some(0.5), "width", "100px"),
            ]
        };
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 100_000.0, "width", "7px"),
            "7px"
        );
        let key = "prod:pad-last-mid";
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 75_000.0, "width", "7px"),
            "53.5px"
        );
        // Property missing from the first keyframe: its
        // property-specific list starts at the second keyframe,
        // so offset 0 is padded with the underlying value.
        let key = "prod:pad-missing-first";
        let kfs = || {
            vec![
                kf_prop(Some(0.0), "opacity", "0"),
                kf_prop(Some(1.0), "width", "100px"),
            ]
        };
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 0.0, "width", "7px"),
            "7px"
        );
        let key = "prod:pad-missing-first-mid";
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 50_000.0, "width", "7px"),
            "53.5px"
        );
    }

    #[test]
    fn single_keyframe_is_the_end_state() {
        // A lone keyframe with no offset normalizes to 1 and
        // interpolates from the underlying value to itself.
        let key = "prod:single-end-start";
        let kfs = || vec![kf_prop(None, "width", "100px")];
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 0.0, "width", "7px"),
            "7px"
        );
        let key = "prod:single-end-mid";
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 50_000.0, "width", "7px"),
            "53.5px"
        );
        let key = "prod:single-end-end";
        assert_eq!(
            animated_value(key, kfs(), waapi_opts("linear"), 100_000.0, "width", "7px"),
            "100px"
        );
    }

    #[test]
    fn color_interpolation_is_alpha_premultiplied() {
        // Transparent -> opaque red at 0.5: premultiplied
        // midpoint is full red at half alpha, not half red.
        let key = "prod:color-premult";
        assert_eq!(
            animated_value(
                key,
                vec![
                    kf_prop(Some(0.0), "background-color", "transparent"),
                    kf_prop(Some(1.0), "background-color", "red"),
                ],
                waapi_opts("linear"),
                50_000.0,
                "background-color",
                "rgb(0, 0, 0)",
            ),
            "rgba(255, 0, 0, 0.5)"
        );
        // Two fully transparent endpoints: output alpha 0
        // serializes as transparent black, not a NaN channel.
        let key = "prod:color-alpha-zero";
        assert_eq!(
            animated_value(
                key,
                vec![
                    kf_prop(Some(0.0), "background-color", "rgba(255, 0, 0, 0)"),
                    kf_prop(Some(1.0), "background-color", "rgba(0, 0, 255, 0)"),
                ],
                waapi_opts("linear"),
                50_000.0,
                "background-color",
                "rgb(0, 0, 0)",
            ),
            "rgba(0, 0, 0, 0)"
        );
    }

    #[test]
    fn finish_throws_on_infinite_effect_end() {
        // Web Animations 1 §4.5.13 step 1: rate > 0 and
        // associated effect end infinite -> InvalidStateError.
        // At this layer that is `finish() == false` (and
        // `finish_checked() == Some(false)`); the state is
        // unchanged.
        let now = Instant::now();
        let mut inf_iter = state_with(100_000.0);
        inf_iter.options.iterations = f64::INFINITY;
        assert!(!inf_iter.finish(now));
        assert_eq!(inf_iter.play_state, PlayState::Running);
        assert_eq!(inf_iter.hold_time_ms, None);
        let mut inf_dur = state_with(f64::INFINITY);
        assert!(!inf_dur.finish(now));
        assert_eq!(finish_checked(u64::MAX), None);
        let id = create_animation(
            "prod:finish-inf".to_string(),
            vec![kf(Some(0.0), "auto")],
            AnimationOptions {
                duration_ms: 100_000.0,
                iterations: f64::INFINITY,
                ..AnimationOptions::default()
            },
        );
        assert_eq!(finish_checked(id), Some(false));
        // Finite case still finishes, and the hold time is the
        // effect end including endDelay (delay + active +
        // endDelay), per §4.5.13 step 4.
        let mut fin = state_with(100_000.0);
        fin.options.delay_ms = 1_000.0;
        fin.options.end_delay_ms = 5_000.0;
        assert!(fin.finish(now));
        assert_eq!(fin.current_time_ms(now), Some(106_000.0));
        assert_eq!(fin.play_state, PlayState::Finished);
    }

    #[test]
    fn finish_succeeds_for_zero_duration_infinite_iterations() {
        // WAAPI `0 * inf` resolves to zero, not infinity: a
        // zero iteration duration with infinite iterations
        // (and symmetrically infinite duration with zero
        // iterations) has a finite effect end, so `finish()`
        // must succeed, not throw InvalidStateError.
        let now = Instant::now();
        let mut zero_dur = state_with(0.0);
        zero_dur.options.iterations = f64::INFINITY;
        zero_dur.options.delay_ms = 1_000.0;
        zero_dur.options.end_delay_ms = 5_000.0;
        assert_eq!(zero_dur.options.active_duration_ms(), 0.0);
        assert!(zero_dur.finish(now));
        assert_eq!(zero_dur.play_state, PlayState::Finished);
        // Hold time is the finite effect end: delay + active
        // (0) + endDelay.
        assert_eq!(zero_dur.current_time_ms(now), Some(6_000.0));
        let mut zero_iter = state_with(f64::INFINITY);
        zero_iter.options.iterations = 0.0;
        zero_iter.options.delay_ms = 1_000.0;
        zero_iter.options.end_delay_ms = 5_000.0;
        assert_eq!(zero_iter.options.active_duration_ms(), 0.0);
        assert!(zero_iter.finish(now));
        assert_eq!(zero_iter.current_time_ms(now), Some(6_000.0));
        // The same combo is treated as ended, not
        // never-ending, by the phase helpers: at local time 0
        // (past the zero-length active interval) progress is
        // the end value, and with fill none nothing applies.
        let mut ended = state_with(0.0);
        ended.options.iterations = f64::INFINITY;
        ended.options.fill = Fill::None;
        ended.set_current_time(now, Some(0.0));
        assert_eq!(ended.iteration_progress(now), Some(1.0));
        assert!(!ended.applies_at(now));
        ended.options.fill = Fill::Forwards;
        assert!(ended.applies_at(now));
    }

    #[test]
    fn iteration_progress_infinite_iterations_infinite_hold_is_not_nan() {
        // Finite duration + infinite iterations + an infinite
        // hold (Rust API only; JS ignores non-finite
        // assignments): the active-end branch is skipped
        // because the active duration is infinite, so without
        // the non-finite-total guard the progress would be
        // `inf - inf = NaN`. It must instead report the
        // active-end value, finite under every direction.
        let now = Instant::now();
        for (dir, expected) in [
            (Direction::Normal, 1.0),
            (Direction::Reverse, 0.0),
            (Direction::Alternate, 1.0),
            (Direction::AlternateReverse, 0.0),
        ] {
            let mut s = state_with(100_000.0);
            s.options.iterations = f64::INFINITY;
            s.options.direction = dir;
            s.pause(now);
            s.set_current_time(now, Some(f64::INFINITY));
            let p = s.iteration_progress(now).expect("paused has progress");
            assert!(
                p.is_finite(),
                "infinite hold with infinite iterations must not yield NaN progress under {dir:?}, got {p}"
            );
            assert_eq!(p, expected, "active-end progress under {dir:?}");
        }
    }

    #[test]
    fn iteration_progress_routes_all_phases_through_direction() {
        let now = Instant::now();
        // Before phase (negative local time): reverse and
        // alternate-reverse show the end keyframe (progress 1)
        // during the delay; normal and alternate show 0.
        for (dir, expected) in [
            (Direction::Normal, 0.0),
            (Direction::Reverse, 1.0),
            (Direction::Alternate, 0.0),
            (Direction::AlternateReverse, 1.0),
        ] {
            let mut s = state_with(100_000.0);
            s.options.delay_ms = 10_000.0;
            s.options.direction = dir;
            s.set_current_time(now, Some(-5_000.0));
            assert_eq!(
                s.iteration_progress(now),
                Some(expected),
                "before-phase progress under {dir:?}"
            );
        }
        // Zero duration: reverse ends at 0, normal at 1.
        let mut zr = state_with(0.0);
        zr.options.direction = Direction::Reverse;
        zr.set_current_time(now, Some(0.0));
        assert_eq!(zr.iteration_progress(now), Some(0.0));
        let mut zn = state_with(0.0);
        zn.set_current_time(now, Some(0.0));
        assert_eq!(zn.iteration_progress(now), Some(1.0));
        // Infinite duration, finite local time: 0 under normal,
        // direction-adjusted under reverse.
        let mut inf = state_with(f64::INFINITY);
        inf.set_current_time(now, Some(50_000.0));
        assert_eq!(inf.iteration_progress(now), Some(0.0));
        let mut infr = state_with(f64::INFINITY);
        infr.options.direction = Direction::Reverse;
        infr.set_current_time(now, Some(50_000.0));
        assert_eq!(infr.iteration_progress(now), Some(1.0));
        // Non-finite local time never yields NaN progress.
        let mut nan = state_with(100_000.0);
        nan.pause(now);
        nan.set_current_time(now, Some(f64::NAN));
        let p = nan.iteration_progress(now).expect("paused has progress");
        assert!(
            p.is_finite(),
            "NaN hold must not yield NaN progress, got {p}"
        );
        let mut inf_hold = state_with(100_000.0);
        inf_hold.pause(now);
        inf_hold.set_current_time(now, Some(f64::INFINITY));
        let p = inf_hold
            .iteration_progress(now)
            .expect("paused has progress");
        assert!(
            p.is_finite(),
            "infinite hold must not yield NaN progress, got {p}"
        );
    }

    #[test]
    fn applies_at_exact_active_end_is_after_phase() {
        let now = Instant::now();
        // Exactly at the active end with fill none: the
        // half-open interval has ended, so nothing applies.
        let mut none = state_with(100_000.0);
        none.options.fill = Fill::None;
        none.pause(now);
        none.set_current_time(now, Some(100_000.0));
        assert!(!none.applies_at(now));
        // Same instant with forwards fill: the after-phase
        // branch holds the end value.
        let mut fwd = state_with(100_000.0);
        fwd.options.fill = Fill::Forwards;
        fwd.pause(now);
        fwd.set_current_time(now, Some(100_000.0));
        assert!(fwd.applies_at(now));
        // During endDelay (past the active end, before the end
        // time): forwards still holds, none still does not.
        let mut fwd_end = state_with(100_000.0);
        fwd_end.options.fill = Fill::Forwards;
        fwd_end.options.end_delay_ms = 50_000.0;
        fwd_end.pause(now);
        fwd_end.set_current_time(now, Some(125_000.0));
        assert!(fwd_end.applies_at(now));
        let mut none_end = state_with(100_000.0);
        none_end.options.fill = Fill::None;
        none_end.options.end_delay_ms = 50_000.0;
        none_end.pause(now);
        none_end.set_current_time(now, Some(125_000.0));
        assert!(!none_end.applies_at(now));
    }

    #[test]
    fn keyframed_shorthand_is_skipped_but_longhand_animates() {
        let key = "prod:shorthand-skip";
        let id = create_animation(
            key.to_string(),
            vec![
                Keyframe {
                    offset: Some(0.0),
                    composite: Composite::Replace,
                    easing: None,
                    properties: vec![
                        ("background".to_string(), "red".to_string()),
                        ("background-color".to_string(), "rgb(0, 0, 0)".to_string()),
                    ],
                },
                Keyframe {
                    offset: Some(1.0),
                    composite: Composite::Replace,
                    easing: None,
                    properties: vec![
                        ("background".to_string(), "blue".to_string()),
                        (
                            "background-color".to_string(),
                            "rgb(255, 255, 255)".to_string(),
                        ),
                    ],
                },
            ],
            waapi_opts("linear"),
        );
        assert!(pause(id));
        assert!(set_current_time(id, Some(50_000.0)));
        let mut props = HashMap::from([
            ("background".to_string(), "yellow".to_string()),
            ("background-color".to_string(), "rgb(0, 0, 0)".to_string()),
        ]);
        apply_animations(key, &mut props, &HashSet::new());
        assert_eq!(props["background"], "yellow");
        assert_eq!(props["background-color"], "rgb(128, 128, 128)");
    }

    #[test]
    fn format_number_does_not_saturate_huge_integers() {
        assert_eq!(format_number(42.0), "42");
        let huge = format_number(1e20);
        assert_ne!(
            huge, "9223372036854775807",
            "1e20 must not saturate to i64::MAX"
        );
        // Exact round-trip is not expected: `format_number`
        // rounds to 9 decimal places first, which loses low
        // digits at this magnitude. The point is the value
        // stays at its own magnitude, not `i64::MAX`.
        let parsed = huge.parse::<f64>().expect("huge formats as a number");
        assert!(
            (parsed - 1e20).abs() / 1e20 < 1e-9,
            "1e20 must format near itself, got {huge}"
        );
        let neg = format_number(-1e20);
        let parsed_neg = neg.parse::<f64>().expect("negative huge formats");
        assert!(
            (parsed_neg + 1e20).abs() / 1e20 < 1e-9,
            "-1e20 must format near itself, got {neg}"
        );
    }
}
