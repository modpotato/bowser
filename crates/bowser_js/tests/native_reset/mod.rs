//! Native Phase-B reset for the test262 harness (`super::Evaluator`).
//!
//! This replaces the old generated-JS reset evals: every observation and
//! repair goes through the [`Vm`] native reset APIs
//! ([`Vm::own_property_keys`], [`Vm::get_property`],
//! [`Vm::get_own_descriptor`], [`Vm::delete_property`]), which read the
//! heap property lists directly, plus the reset-epoch bookkeeping
//! ([`Vm::mutation_epoch`], [`Vm::track_reset_object`],
//! [`Vm::take_dirtied_reset_objects`]) that narrows each scrub to the
//! targets the previous test actually touched, and the extensibility
//! query/repair pair ([`Vm::is_extensible`], [`Vm::set_extensible`]).
//! No generated JS runs during reset, so a test that clobbered
//! `Reflect`/`Object` builtins cannot blind the scrub, and the snapshot
//! stashes nothing on `globalThis`.
//!
//! Two documented divergences from the old generated-JS checks (pinned by
//! `w17_reset_apis_differential_against_js_scrub_checks` in `vm.rs`):
//!
//! - `get_property` skips getters: accessor values are never snapshotted
//!   or compared (invoking a getter would run arbitrary JS from `&self`).
//!   The old check invoked the getter on every reset.
//! - get/set identity is presence-only (`has_getter`/`has_setter`): a
//!   getter/setter *swap* is invisible. The old check compared identity
//!   and rebuilt the `Vm` on a swap.

use std::cell::Cell;
use std::collections::HashSet;

use bowser_js::{OwnDescriptor, Property, Value, Vm, Wtf8};

use super::{KNOWN_GLOBALS, reset_eq};

/// One baseline property snapshot inside a [`ScrubTarget`].
#[derive(Debug, Clone)]
struct ScrubKey {
    name: String,
    /// Baseline value for data properties (V-check). `None` for accessor
    /// properties (divergence: getters are never invoked).
    value: Option<Value>,
    /// Baseline descriptor snapshot (A-check). Get/set identity is
    /// presence-only (divergence: swaps are invisible).
    desc: OwnDescriptor,
}

/// One scrub target: a builtin prototype (`Array.prototype`), a plain
/// namespace object (`Math`), or the global object itself (`globalThis`,
/// untracked by design — see [`dirtied_targets`]).
#[derive(Debug, Clone)]
struct ScrubTarget {
    /// Name of the global the target hangs off (`"Array"` for
    /// `Array.prototype`, `"Math"` for `Math`); used for the R-check.
    /// Constructor-target `[[Prototype]]` intermediates registered by
    /// [`build_proto_scrub`] get a synthesized name (e.g.
    /// `"Int8Array.prototype.__proto__"`) used for debug output only —
    /// they are never constructor targets, so the R-check never reads
    /// the name as a global.
    global_name: String,
    /// Whether this is a constructor target (`<name>.prototype`, with
    /// binding/prototype identity anchors).
    is_ctor: bool,
    /// True for the `globalThis` target: the global object is
    /// deliberately never epoch-tracked (tracking it would bump the
    /// epoch on every test and kill the fast path), so
    /// [`dirtied_targets`] still checks it on every slow-path reset —
    /// extensibility + baseline attribute triples only.
    is_global: bool,
    /// The target object itself.
    target: Value,
    /// Baseline `globalThis.<global_name>` value (R-check anchor: the
    /// binding itself was replaced).
    binding: Value,
    /// Baseline `[[Prototype]]` of `target` (R-check anchor: reparented).
    /// Only checked for constructor targets, mirroring the old check.
    proto: Value,
    /// Baseline extensibility of `target` (E-check anchor:
    /// `Reflect.preventExtensions(target)`). Extensibility is not a
    /// property, so no key/attribute/value finding observes it.
    extensible: bool,
    /// Baseline own-key snapshots.
    keys: Vec<ScrubKey>,
}

/// Per-test scrub state for builtin-prototype and namespace-object
/// pollution: the pristine shape snapshotted at [`build_proto_scrub`],
/// scrubbed back by [`scrub_prototypes`].
#[derive(Debug, Default, Clone)]
pub struct ProtoScrub {
    targets: Vec<ScrubTarget>,
    /// Set by [`prototypes_clean`] when it finds drift the epoch
    /// bookkeeping missed; the next [`dirtied_targets`] full-scans
    /// instead of trusting the dirtied subset. Interior mutability, as
    /// above.
    force_full_scan: Cell<bool>,
}

impl ProtoScrub {
    /// `(target, binding)` object pairs, for the heap mutation-epoch
    /// registration: any write to either bumps the epoch, so an
    /// unchanged epoch means [`scrub_prototypes`] would find nothing.
    /// The binding is included because the R-check reads the global
    /// binding's `"prototype"` property (`Array.prototype = …` writes
    /// the binding, not the target).
    pub fn tracked_objects(&self) -> Vec<(Value, Value)> {
        self.targets.iter().map(|t| (t.target, t.binding)).collect()
    }
}

/// Snapshot one object's scrub state: its `[[Prototype]]` (the R-check
/// anchor for constructor targets), its extensibility flag (the E-check
/// anchor: `Reflect.preventExtensions(target)` flips this with no
/// property change, so it needs its own snapshot), and the baseline
/// own-key snapshots (data values for the V-check — the chain walk finds
/// the own data property, exactly what the old `target[key]` read
/// observed; accessor properties skip the value, the documented
/// divergence). Non-objects report extensible with no keys; every target
/// here is an object.
fn snapshot_target(vm: &Vm, target: &Value) -> (Value, bool, Vec<ScrubKey>) {
    let proto = match target {
        Value::Object(o) => o.prototype,
        _ => Value::Undefined,
    };
    let extensible = vm.is_extensible(target);
    let mut keys = Vec::new();
    for k in vm.own_property_keys(target) {
        let key_name = k.as_wtf8().to_string();
        let Some(desc) = vm.get_own_descriptor(target, &key_name) else {
            continue; // fail-open: this key gets no checks
        };
        let value = desc
            .value
            .is_some()
            .then(|| vm.get_property(target, &key_name));
        keys.push(ScrubKey {
            name: key_name,
            value,
            desc,
        });
    }
    (proto, extensible, keys)
}

/// Snapshot the pristine shape of builtin prototypes and namespace objects.
///
/// Classification mirrors the old generated-JS classifier: a global whose
/// value is an object with an object-valued `prototype` property is a
/// constructor (scrub `<name>.prototype`); any other object-valued global
/// is a namespace object (scrub it directly); anything else is skipped.
pub fn build_proto_scrub(vm: &Vm) -> ProtoScrub {
    let mut scrub = ProtoScrub::default();
    for name in KNOWN_GLOBALS {
        let Some(binding) = vm.get_global(name) else {
            continue;
        };
        if !matches!(binding, Value::Object(_)) {
            continue;
        }
        // Constructor iff it carries an object-valued `prototype` (the
        // chain walk finds the own property, mirroring the old
        // `typeof v === "function" && v.prototype !== undefined` test).
        let (target, is_ctor) = match vm.get_property(&binding, "prototype") {
            proto @ Value::Object(_) => (proto, true),
            _ => (binding, false),
        };
        let (proto, extensible, keys) = snapshot_target(vm, &target);
        scrub.targets.push(ScrubTarget {
            global_name: name.to_string(),
            is_ctor,
            // Only `globalThis`'s own entry targets the global object;
            // every other target is a builtin prototype or namespace.
            is_global: *name == "globalThis",
            target,
            binding,
            proto,
            extensible,
            keys,
        });
    }
    // Second pass: the constructor targets' own `[[Prototype]]` needs a
    // scrub target too when it is an object no existing target covers —
    // e.g. the shared `%TypedArray%.prototype` behind the typed-array
    // constructors, which no `KNOWN_GLOBALS` entry names (there is no
    // `TypedArray` global). Destructively-specified helpers
    // (`verifyProperty`'s `isConfigurable` does `delete obj[name]`
    // with no restore option) can delete keys on that shared
    // prototype; the check only covers registered targets, so the
    // deletion persisted across tests in the reused `Vm`,
    // order-dependently. The coverage test runs against the complete
    // primary set (a single pass would miss `Object.prototype`, whose
    // `"Object"` entry sorts after e.g. `"Array"`): `Object.prototype`
    // and error-subtype intermediates like `Error.prototype` are
    // already targets, so they are skipped, and the shared
    // `%TypedArray%.prototype` is registered once for all nine
    // typed-array constructors. Registered as a non-constructor target
    // with a synthesized name (the R-check is skipped — the name is not
    // a global); the full added/missing/attribute/value/extensibility
    // checks still run, and epoch tracking covers it through
    // [`ProtoScrub::tracked_objects`].
    let ctors: Vec<(String, Value, Value)> = scrub
        .targets
        .iter()
        .filter(|t| t.is_ctor)
        .map(|t| (t.global_name.clone(), t.binding, t.proto))
        .collect();
    for (name, binding, proto) in ctors {
        if !matches!(proto, Value::Object(_)) || scrub.targets.iter().any(|t| t.target == proto) {
            continue;
        }
        let (sub_proto, sub_extensible, sub_keys) = snapshot_target(vm, &proto);
        scrub.targets.push(ScrubTarget {
            global_name: format!("{name}.prototype.__proto__"),
            is_ctor: false,
            is_global: false,
            target: proto,
            binding,
            proto: sub_proto,
            extensible: sub_extensible,
            keys: sub_keys,
        });
    }
    scrub
}

/// One finding of [`check_target`], mirroring the old generated check's
/// finding kinds.
#[derive(Debug)]
enum Finding {
    /// Added key (not in the baseline).
    Added(String),
    /// Baseline key deleted by the test.
    Missing(String),
    /// Attribute triple changed.
    AttrsChanged(String),
    /// Data value changed.
    ValueChanged(String),
    /// `Reflect.preventExtensions(target)` (or equivalent) flipped the
    /// target's extensibility. Repaired in place with
    /// [`Vm::set_extensible`]: re-setting the flag restores the exact
    /// baseline state (`[[PreventExtensions]]` mutates nothing else),
    /// and the repair is epoch-accounted like any other heap write.
    ExtensibilityChanged,
}

/// Attribute-only comparison of two descriptor snapshots (the A-check):
/// `enumerable`/`writable`/`configurable`, the data-vs-accessor kind
/// (accessors have `writable: None`), and presence-only get/set. The
/// value is compared separately (V-check), mirroring the old check,
// which never compared values in its attribute clause.
fn attrs_eq(a: &OwnDescriptor, b: &OwnDescriptor) -> bool {
    a.enumerable == b.enumerable
        && a.writable == b.writable
        && a.configurable == b.configurable
        && a.has_getter == b.has_getter
        && a.has_setter == b.has_setter
}

/// How thoroughly a scrub target is checked on this reset.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TargetCheck {
    /// Full check: binding/prototype identity, extensibility, added /
    /// missing keys, attribute triples, and data values.
    Full,
    /// The untracked global object only: extensibility plus baseline
    /// attribute triples. Global-slot *values* belong to the harness's
    /// baseline value scan (which rebuilds before the scrub runs) and
    /// added keys belong to [`scrub_global_keys`], so a value-only write
    /// like `Array = 1` reports no finding and the check performs no
    /// heap write.
    GlobalAttrsOnly,
}

impl TargetCheck {
    /// The check mode for a target: the global target always gets the
    /// restricted check (its values and added keys are owned by other
    /// scans), everything else the full check.
    fn for_target(t: &ScrubTarget) -> TargetCheck {
        if t.is_global {
            TargetCheck::GlobalAttrsOnly
        } else {
            TargetCheck::Full
        }
    }
}

/// Check one target natively. `Ok(findings)` lists what must be repaired
/// (empty when pristine); `Err(reason)` is an unrecoverable R-finding:
/// the binding was replaced, the prototype object changed identity, or it
/// was reparented.
fn check_target(vm: &Vm, t: &ScrubTarget) -> Result<Vec<Finding>, &'static str> {
    // R-check (constructor targets only): the binding was replaced, or
    // the prototype object was swapped or reparented.
    if t.is_ctor {
        let ok = vm.get_global(&t.global_name).is_some_and(|b| {
            if b != t.binding {
                return false;
            }
            match vm.get_property(&b, "prototype") {
                Value::Object(o) => Value::Object(o) == t.target && o.prototype == t.proto,
                _ => false,
            }
        });
        if !ok {
            return Err("prototype replaced or reparented");
        }
    }
    let mut findings = Vec::new();
    // E-check: extensibility is not a property, so the key scans below
    // cannot see it. Without this, `Reflect.preventExtensions` on a
    // builtin prototype persisted across tests silently.
    if vm.is_extensible(&t.target) != t.extensible {
        findings.push(Finding::ExtensibilityChanged);
    }
    let baseline: HashSet<&Wtf8> = t
        .keys
        .iter()
        .map(|k| Wtf8::from_str(k.name.as_str()))
        .collect();
    for k in vm.own_property_keys(&t.target) {
        if !baseline.contains(k.as_wtf8()) {
            findings.push(Finding::Added(k.as_wtf8().to_string()));
        }
    }
    for k in &t.keys {
        let cur = vm.get_own_descriptor(&t.target, &k.name);
        let Some(cur) = cur else {
            findings.push(Finding::Missing(k.name.clone()));
            continue;
        };
        if !attrs_eq(&cur, &k.desc) {
            findings.push(Finding::AttrsChanged(k.name.clone()));
        }
        // V-check, data keys only: accessor values are never snapshotted
        // (documented divergence: `get_property` skips getters).
        if let Some(base_val) = k.value
            && !reset_eq(vm.get_property(&t.target, &k.name), base_val)
        {
            findings.push(Finding::ValueChanged(k.name.clone()));
        }
    }
    Ok(findings)
}

/// Repair a target's findings. Mirrors the old repair order: deletes,
/// then attribute repairs (full baseline descriptor, which also restores
/// the value), then value restores. Returns `Err(reason)` when a finding
/// is unrecoverable — the caller rebuilds the `Vm`, exactly as the old
/// re-check did after a failed repair eval.
fn repair_target(vm: &mut Vm, t: &ScrubTarget, findings: &[Finding]) -> Result<(), &'static str> {
    let target_obj = match t.target {
        Value::Object(o) => o,
        _ => return Err("scrub target is not an object"),
    };
    // Extensibility is repaired in place, before the property repairs
    // below: `Vm::set_extensible` restores the baseline flag directly
    // (`[[PreventExtensions]]` mutates nothing else, so the flag flip
    // is the whole drift), accounts a reset-epoch write, and runs no
    // JS. The re-verification in `scrub_target` confirms the restore.
    if findings
        .iter()
        .any(|f| matches!(f, Finding::ExtensibilityChanged))
    {
        vm.set_extensible(&t.target, t.extensible);
    }
    let baseline_key = |name: &str| t.keys.iter().find(|k| k.name == name);
    // Added keys: delete. A non-configurable addition refuses the delete
    // and stays; the re-check below then fails and rebuilds.
    for f in findings {
        if let Finding::Added(name) = f {
            vm.delete_property(&t.target, name);
        }
    }
    // Attribute repairs (before value restores: a value restore needs
    // the baseline writability back first).
    for f in findings {
        let name = match f {
            Finding::AttrsChanged(name) | Finding::Missing(name) => name,
            _ => continue,
        };
        let key = baseline_key(name).ok_or("finding for unknown baseline key")?;
        // Data descriptor fields; accessor baselines are unrecoverable
        // (their get/set pair cannot be rebuilt from the presence-only
        // snapshot), mirroring the old repair, which rebuilt on those.
        let (w, e, c) = match (
            key.desc.writable,
            key.desc.enumerable,
            key.desc.configurable,
        ) {
            (Some(w), Some(e), Some(c)) => (w, e, c),
            _ => return Err("accessor repair unrecoverable"),
        };
        let v = key.value.ok_or("accessor repair unrecoverable")?;
        if matches!(f, Finding::AttrsChanged(_)) {
            // A non-configurable current property cannot be re-defined:
            // the old `Object.defineProperty` repair threw here and the
            // re-check rebuilt.
            let cur_c = vm
                .get_own_descriptor(&t.target, name)
                .and_then(|d| d.configurable);
            if cur_c != Some(true) {
                return Err("non-configurable attribute change");
            }
        }
        vm.heap_mut()
            .define_property(
                target_obj,
                Wtf8::from_str(name),
                Property {
                    value: v,
                    writable: w,
                    enumerable: e,
                    configurable: c,
                },
            )
            .map_err(|_| "out of memory re-defining property")?;
    }
    // Value restores, with plain-assignment semantics: only a writable
    // data property takes the write; anything else rebuilds (the old
    // sloppy assignment no-op'd on non-writable properties and the
    // re-check rebuilt the `Vm`).
    for f in findings {
        let Finding::ValueChanged(name) = f else {
            continue;
        };
        let key = baseline_key(name).ok_or("finding for unknown baseline key")?;
        let base_val = key.value.ok_or("value finding on accessor key")?;
        let cur = vm.get_own_descriptor(&t.target, name);
        if cur.is_none_or(|d| d.writable != Some(true) || d.value.is_none()) {
            return Err("value restore on non-writable property");
        }
        // Attributes are baseline here (no A-finding, or repaired above).
        let (e, c) = match (key.desc.enumerable, key.desc.configurable) {
            (Some(e), Some(c)) => (e, c),
            _ => return Err("value finding on accessor key"),
        };
        vm.heap_mut()
            .define_property(
                target_obj,
                Wtf8::from_str(name),
                Property {
                    value: base_val,
                    writable: true,
                    enumerable: e,
                    configurable: c,
                },
            )
            .map_err(|_| "out of memory restoring value")?;
    }
    Ok(())
}

/// Check-only verification for the untracked global object:
/// extensibility (E-check) plus the baseline attribute triples
/// (A-check). No value checks: the harness baseline value scan owns
/// global-slot values — a deleted or overwritten baseline global
/// rebuilds before the scrub runs — and no added-key checks
/// ([`scrub_global_keys`] owns key additions). Missing baseline keys are
/// likewise the value scan's (a deleted global slot reads as
/// `undefined` there). Always `Ok`: the global target is never a
/// constructor target, so there is no R-check to fail.
fn check_global_attrs(vm: &Vm, t: &ScrubTarget) -> Result<Vec<Finding>, &'static str> {
    debug_assert!(t.is_global);
    let mut findings = Vec::new();
    // E-check: extensibility is not a property, so no key scan observes
    // it — without this, `Reflect.preventExtensions(globalThis)`
    // persisted across tests silently.
    if vm.is_extensible(&t.target) != t.extensible {
        findings.push(Finding::ExtensibilityChanged);
    }
    // A-check over baseline keys only.
    for k in &t.keys {
        let Some(cur) = vm.get_own_descriptor(&t.target, &k.name) else {
            continue; // harness baseline value scan owns deletions
        };
        if !attrs_eq(&cur, &k.desc) {
            findings.push(Finding::AttrsChanged(k.name.clone()));
        }
    }
    Ok(findings)
}

/// Dispatch [`check_target`] / [`check_global_attrs`] by mode.
fn check_target_mode(
    vm: &Vm,
    t: &ScrubTarget,
    mode: TargetCheck,
) -> Result<Vec<Finding>, &'static str> {
    match mode {
        TargetCheck::Full => check_target(vm, t),
        TargetCheck::GlobalAttrsOnly => check_global_attrs(vm, t),
    }
}

/// Check one target and repair its findings; re-verify afterwards.
/// Returns `false` when the target is poisoned beyond cheap repair (the
/// caller rebuilds the `Vm`). `mode` comes from [`dirtied_targets`]:
/// the untracked global target gets the restricted
/// extensibility/attribute check.
fn scrub_target(vm: &mut Vm, t: &ScrubTarget, mode: TargetCheck, debug: bool) -> bool {
    let findings = match check_target_mode(vm, t, mode) {
        Ok(f) => f,
        Err(reason) => {
            if debug {
                eprintln!("reset: scrub target {}: {reason}", t.global_name);
            }
            return false;
        }
    };
    if findings.is_empty() {
        return true;
    }
    if debug {
        eprintln!(
            "reset: scrub target {} findings: {:?}",
            t.global_name, findings
        );
    }
    if repair_target(vm, t, &findings).is_err() {
        return false;
    }
    // Re-verify: anything left is unrecoverable.
    match check_target_mode(vm, t, mode) {
        Ok(f) => {
            if !f.is_empty() && debug {
                eprintln!("reset: scrub target {} incomplete: {:?}", t.global_name, f);
            }
            f.is_empty()
        }
        Err(_) => false,
    }
}

/// The scrub targets to check on this reset: exactly the dirtied subset,
/// or every target when the forced full scan is armed.
///
/// Drains [`Vm::take_dirtied_reset_objects`] and keeps the targets whose
/// target or binding value appears in the dirtied set — the binding is
/// tracked too, because the R-check reads `binding.prototype`
/// (`Array.prototype = …` writes the binding, not the target). The match
/// is object identity: [`Value`]'s `PartialEq` on objects is the
/// heap-address comparison, so a dirtied handle can only match the
/// target it was registered from.
///
/// The global object is deliberately never tracked (tracking it would
/// bump the epoch on every test and kill the fast path), so no dirtied
/// set — empty or not — proves the global target pristine: writes to
/// baseline global keys bump the epoch without dirtying any target, and
/// two of those shapes are invisible to every other scan —
/// `Reflect.preventExtensions(globalThis)` flips no property, and
/// attribute-only redefines of baseline globals (e.g.
/// `Object.defineProperty(globalThis, "Array", {enumerable: false})`)
/// change no value. The harness baseline value scan (which rebuilds
/// before the scrub runs) owns global-slot values and
/// [`scrub_global_keys`] owns added keys, so the single global target is
/// checked on every slow-path reset for exactly its extensibility +
/// baseline attribute triples — cheap (one target), and a value-only
/// write like `Array = 1` reports no finding and performs no heap write.
/// A forced full scan (armed by [`prototypes_clean`] when it finds drift
/// the epoch bookkeeping missed) checks every target instead: a write
/// the epoch missed dirtied nothing either.
fn dirtied_targets<'a>(vm: &mut Vm, scrub: &'a ProtoScrub) -> Vec<(&'a ScrubTarget, TargetCheck)> {
    let dirtied = vm.take_dirtied_reset_objects();
    if scrub.force_full_scan.replace(false) {
        return scrub
            .targets
            .iter()
            .map(|t| (t, TargetCheck::for_target(t)))
            .collect();
    }
    let mut out: Vec<(&ScrubTarget, TargetCheck)> = scrub
        .targets
        .iter()
        .filter(|t| !t.is_global && (dirtied.contains(&t.target) || dirtied.contains(&t.binding)))
        .map(|t| (t, TargetCheck::Full))
        .collect();
    // The untracked global target rides along on every slow-path reset
    // (extensibility + attribute triples only): a test that dirtied a
    // tracked target may also have touched the global, and the dirtied
    // subset alone would miss it.
    out.extend(
        scrub
            .targets
            .iter()
            .filter(|t| t.is_global)
            .map(|t| (t, TargetCheck::GlobalAttrsOnly)),
    );
    out
}

/// Check-only prototype / namespace-object verification: true when every
/// target matches its baseline (no findings, no R-check failures).
/// Repairs nothing, unlike [`scrub_prototypes`]; used by the harness's
/// periodic epoch-fast-path validation. With `debug`, the first drifting
/// target is logged.
///
/// Deliberately checks *every* target instead of the dirtied subset:
/// this is the backstop for the epoch bookkeeping itself, so it cannot
/// trust the epoch (or the dirtied set) to narrow the scan. On drift it
/// arms [`ProtoScrub`]'s forced full scan, so the slow-path
/// [`scrub_prototypes`] that follows repairs the missed write instead of
/// trusting the empty dirtied set.
///
/// Takes `&mut Vm` to match [`scrub_prototypes`]' signature (the harness
/// threads its exclusively-borrowed vm through both); the check itself
/// only reads.
pub fn prototypes_clean(vm: &mut Vm, scrub: &ProtoScrub, debug: bool) -> bool {
    for t in &scrub.targets {
        let clean = matches!(check_target(vm, t), Ok(findings) if findings.is_empty());
        if !clean {
            if debug {
                eprintln!(
                    "reset: validation: prototype target drifted: {}",
                    t.global_name
                );
            }
            scrub.force_full_scan.set(true);
            return false;
        }
    }
    true
}

/// Run the prototype / namespace-object scrub natively: added properties
/// deleted, overwritten data values restored, attribute flips re-defined,
/// extensibility flips re-enabled — enumerable or not. Checks only
/// dirtied targets, plus the untracked global target's extensibility and
/// baseline attribute triples (see [`dirtied_targets`]). Returns `false`
/// when a target is poisoned beyond cheap repair (replaced/reparented
/// prototype, non-configurable interference, accessor attribute
/// changes); the caller rebuilds the `Vm`.
pub fn scrub_prototypes(vm: &mut Vm, scrub: &ProtoScrub, debug: bool) -> bool {
    for (t, mode) in dirtied_targets(vm, scrub) {
        if !scrub_target(vm, t, mode, debug) {
            return false;
        }
    }
    // Every dirtied target re-verified pristine above, and the repairs
    // re-dirtied the targets they touched (epoch-accounted): drop those
    // records instead of re-checking clean targets on the next reset.
    let _ = vm.take_dirtied_reset_objects();
    true
}

/// Delete every `globalThis` own key not present at baseline (enumerable
/// or not — [`Vm::own_property_keys`] sees non-enumerables, including the
/// module registry key) and verify the surviving key set. Returns `false`
/// when a non-configurable addition survives the delete; the caller
/// rebuilds the `Vm`.
pub fn scrub_global_keys(
    vm: &mut Vm,
    global: &Value,
    baseline_keys: &[String],
    debug: bool,
) -> bool {
    // Property keys are WTF-8; the `&str`-keyed reset APIs are a lossy
    // boundary, so lone-surrogate extras cannot be deleted here (they are
    // still reported by the key-set comparison below).
    let baseline: HashSet<&Wtf8> = baseline_keys
        .iter()
        .map(|s| Wtf8::from_str(s.as_str()))
        .collect();
    for k in vm.own_property_keys(global) {
        let name = k.as_wtf8();
        if !baseline.contains(name)
            && let Some(s) = name.as_utf8()
        {
            vm.delete_property(global, s);
        }
    }
    let current = vm.own_property_keys(global);
    let mut keys: Vec<&Wtf8> = current.iter().map(|k| k.as_wtf8()).collect();
    keys.sort_unstable();
    let ok = keys
        == baseline_keys
            .iter()
            .map(|s| Wtf8::from_str(s.as_str()))
            .collect::<Vec<_>>();
    if !ok && debug {
        eprintln!("reset: key set mismatch");
    }
    ok
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Poison scenarios: (setup_js, poison_js, expect_reset_ok). The setup
    /// runs *before* the scrub baseline is snapshotted; `reset_ok == false`
    /// means the poisoning is unrecoverable and the `Vm` must be rebuilt.
    fn scenarios() -> Vec<(&'static str, &'static str, bool)> {
        vec![
            // Added enumerable key on a builtin prototype.
            ("", "Array.prototype.__w18_added = 1;", true),
            // Added non-enumerable key.
            (
                "",
                "Object.defineProperty(Array.prototype, '__w18_hidden', \
                 {value: 7, enumerable: false, configurable: true});",
                true,
            ),
            // Overwritten data value.
            ("", "Array.prototype.join = 999;", true),
            // Flipped attribute (still configurable).
            (
                "",
                "Object.defineProperty(Array.prototype, 'join', {enumerable: false});",
                true,
            ),
            // Flipped attribute, then made non-configurable: unrecoverable.
            (
                "",
                "Object.defineProperty(Array.prototype, 'join', \
                 {enumerable: false, configurable: false});",
                false,
            ),
            // Added non-configurable key: the delete refuses, rebuild.
            (
                "",
                "Object.defineProperty(Array.prototype, '__w18_nc', \
                 {value: 1, configurable: false});",
                false,
            ),
            // Reparented prototype: unrecoverable.
            ("", "Object.setPrototypeOf(Array.prototype, {});", false),
            // preventExtensions on a builtin prototype: repaired in place
            // via `Vm::set_extensible` — and, crucially, no longer
            // silent: the E-check finds it. (This engine exposes it as
            // `Reflect.preventExtensions`; `Object` has no such method.)
            ("", "Reflect.preventExtensions(Array.prototype);", true),
            // preventExtensions on a namespace object: same.
            ("", "Reflect.preventExtensions(Math);", true),
            // preventExtensions combined with a deleted baseline key:
            // the E-check repair runs before the property repairs.
            (
                "",
                "Reflect.preventExtensions(Array.prototype); \
                 delete Array.prototype.join;",
                true,
            ),
            // Deleted baseline key: recreated with baseline value+attrs.
            ("", "delete Array.prototype.join;", true),
            // Data -> data redefine with changed value+attributes: V+A
            // findings, repaired by re-defining the baseline descriptor.
            (
                "",
                "Object.defineProperty(Array.prototype, 'join', \
                 {value: 42, writable: true, enumerable: true, configurable: true});",
                true,
            ),
            // Namespace object pollution.
            ("", "Math.__w18_bar = 2;", true),
            // Accessor attribute flip on a baseline accessor: unrecoverable
            // (presence-only snapshot cannot rebuild the get/set pair).
            // Installed from JS before the baseline; the poison uses a
            // partial descriptor so get/set survive and only the flag
            // flips.
            (
                "Object.defineProperty(Array.prototype, '__w18_acc', \
                 {get: function(){ return 11; }, set: function(v){}, \
                  enumerable: false, configurable: true});",
                "Object.defineProperty(Array.prototype, '__w18_acc', \
                 {enumerable: true});",
                false,
            ),
        ]
    }

    /// Native-observed pristine check: every target's key set, data
    /// values, attribute triples, and extensibility equal the scrub
    /// baseline, and every constructor target's binding/prototype identity
    /// holds.
    fn assert_pristine(vm: &Vm, scrub: &ProtoScrub) {
        for t in &scrub.targets {
            if t.is_ctor {
                let b = vm
                    .get_global(&t.global_name)
                    .expect("ctor binding readable");
                assert_eq!(b, t.binding, "binding identity: {}", t.global_name);
                match vm.get_property(&b, "prototype") {
                    Value::Object(o) => {
                        assert_eq!(Value::Object(o), t.target, "target identity");
                        assert_eq!(o.prototype, t.proto, "prototype identity");
                    }
                    v => panic!("prototype not an object: {v:?}"),
                }
            }
            let mut keys: Vec<String> = vm
                .own_property_keys(&t.target)
                .iter()
                .map(|k| k.as_wtf8().to_string())
                .collect();
            keys.sort();
            let mut base: Vec<String> = t.keys.iter().map(|k| k.name.clone()).collect();
            base.sort();
            assert_eq!(keys, base, "key set: {}", t.global_name);
            assert_eq!(
                vm.is_extensible(&t.target),
                t.extensible,
                "extensibility drift: {}",
                t.global_name
            );
            for k in &t.keys {
                let cur = vm
                    .get_own_descriptor(&t.target, &k.name)
                    .unwrap_or_else(|| panic!("baseline key missing: {}", k.name));
                assert!(
                    attrs_eq(&cur, &k.desc),
                    "descriptor drift on {}::{}",
                    t.global_name,
                    k.name
                );
                if let Some(v) = k.value {
                    assert!(
                        reset_eq(vm.get_property(&t.target, &k.name), v),
                        "value drift on {}::{}",
                        t.global_name,
                        k.name
                    );
                }
            }
        }
    }

    #[test]
    fn native_scrub_repairs_or_rebuilds() {
        for (setup, poison, expect_ok) in scenarios() {
            let mut vm = Vm::new();
            if !setup.is_empty() {
                vm.eval(setup).expect("setup script must run");
            }
            let scrub = build_proto_scrub(&vm);
            track_scrub_targets(&mut vm, &scrub);
            vm.eval(poison).expect("poison script must run");
            let ok = scrub_prototypes(&mut vm, &scrub, false);
            assert_eq!(
                ok, expect_ok,
                "scenario {poison:?}: expected reset_ok={expect_ok}"
            );
            if ok {
                assert_pristine(&vm, &scrub);
            }
        }
    }

    /// Register the scrub's targets/bindings for epoch tracking, mirroring
    /// `Evaluator::new` (which skips the global object: the reset-global
    /// registration covers it with finer granularity). Without registered
    /// targets the engine records no writes, the dirtied set stays empty,
    /// and the epoch never moves. The skipped global target is still
    /// checked — extensibility + baseline attribute triples only — by
    /// [`dirtied_targets`] whenever the dirtied set is empty.
    fn track_scrub_targets(vm: &mut Vm, scrub: &ProtoScrub) {
        let global = vm.get_global("globalThis").expect("globalThis");
        for (target, binding) in scrub.tracked_objects() {
            if target != global {
                vm.track_reset_object(&target);
            }
            if binding != global {
                vm.track_reset_object(&binding);
            }
        }
    }

    #[test]
    fn scrub_repairs_delete_only_pollution() {
        // Deletes bump the mutation epoch and dirty the target
        // (`Heap::note_reset_delete`): the scrub re-checks exactly the
        // dirtied target and repairs the missing key. Note the dirtied
        // set is drained exactly once, by the scrub itself: asserting it
        // non-empty here would consume the evidence the scrub needs, and
        // with no conservative full-scan fallback a delete that failed
        // to dirty would leave `join` missing and fail `assert_pristine`
        // below — so the pristine check pins the dirtying.
        let mut vm = Vm::new();
        let scrub = build_proto_scrub(&vm);
        track_scrub_targets(&mut vm, &scrub);
        let epoch = vm.mutation_epoch();
        vm.eval("delete Array.prototype.join;").unwrap();
        assert_ne!(vm.mutation_epoch(), epoch, "delete must bump the epoch");
        assert!(scrub_prototypes(&mut vm, &scrub, false));
        assert_pristine(&vm, &scrub);
    }

    #[test]
    fn scrub_repairs_mixed_write_and_delete_pollution() {
        // Mixed write + delete across distinct tracked targets in one
        // run: both targets must be dirtied and repaired at reset.
        // Regression: deletes used to bump the epoch without dirtying,
        // so the deleted-key target escaped the dirtied-subset scrub and
        // its pollution survived until a periodic full validation.
        let mut vm = Vm::new();
        let scrub = build_proto_scrub(&vm);
        track_scrub_targets(&mut vm, &scrub);
        vm.eval(
            "Array.prototype.__w29_extra = 1; \
             delete String.prototype.charAt;",
        )
        .unwrap();
        assert!(scrub_prototypes(&mut vm, &scrub, false));
        assert_pristine(&vm, &scrub);
    }

    #[test]
    fn pristine_scrub_writes_nothing() {
        // With no writes since the last drain the dirtied subset is
        // empty: the scrub checks nothing and, in particular, performs
        // no heap write (which would itself bump the epoch).
        let mut vm = Vm::new();
        let scrub = build_proto_scrub(&vm);
        track_scrub_targets(&mut vm, &scrub);
        let epoch = vm.mutation_epoch();
        assert!(scrub_prototypes(&mut vm, &scrub, false));
        assert_eq!(
            vm.mutation_epoch(),
            epoch,
            "pristine scrub must not move the epoch"
        );
    }

    #[test]
    fn scrub_skips_targets_when_only_global_slots_written() {
        // A baseline global slot write (`Array = 1`) bumps the mutation
        // epoch without dirtying any tracked target. The scrub checks
        // zero targets instead of full-scanning: no target could have
        // drifted, and the check performs no heap write. (The harness's
        // baseline value scan owns global-slot drift and rebuilds before
        // the scrub runs, so the scrub need not detect it.)
        let mut vm = Vm::new();
        let scrub = build_proto_scrub(&vm);
        track_scrub_targets(&mut vm, &scrub);
        // Mirror `Evaluator::new`: register the global baseline so slot
        // writes bump the epoch (without dirtying any target).
        let global = vm.get_global("globalThis").expect("globalThis");
        let baseline_keys: Vec<String> = vm
            .own_property_keys(&global)
            .iter()
            .map(|k| k.as_wtf8().to_string())
            .collect();
        vm.set_reset_global_baseline(&global, &baseline_keys);
        let epoch = vm.mutation_epoch();
        vm.eval("Array = 1;").unwrap();
        assert_ne!(
            vm.mutation_epoch(),
            epoch,
            "global slot write must bump the epoch"
        );
        assert!(
            vm.take_dirtied_reset_objects().is_empty(),
            "slot write must dirty no target"
        );
        let epoch = vm.mutation_epoch();
        assert!(scrub_prototypes(&mut vm, &scrub, false));
        assert_eq!(
            vm.mutation_epoch(),
            epoch,
            "scrub with empty dirtied set must not move the epoch"
        );
    }

    #[test]
    fn scrub_repairs_prevent_extensions_on_global() {
        // Regression: `Reflect.preventExtensions(globalThis)` bumps the
        // mutation epoch (the global is the reset global, so the write
        // is epoch-accounted) without dirtying any tracked target — the
        // global object is deliberately untracked. The scrub must still
        // repair the flag through the global target's extensibility
        // check; before that check existed, the pollution persisted
        // across tests invisible to every scan (no property changes for
        // the key/attribute/value scans to see).
        let mut vm = Vm::new();
        let scrub = build_proto_scrub(&vm);
        track_scrub_targets(&mut vm, &scrub);
        // Mirror `Evaluator::new`: register the global baseline so the
        // preventExtensions bumps the epoch without dirtying a target.
        let global = vm.get_global("globalThis").expect("globalThis");
        let baseline_keys: Vec<String> = vm
            .own_property_keys(&global)
            .iter()
            .map(|k| k.as_wtf8().to_string())
            .collect();
        vm.set_reset_global_baseline(&global, &baseline_keys);
        let epoch = vm.mutation_epoch();
        vm.eval("Reflect.preventExtensions(globalThis);").unwrap();
        assert_ne!(
            vm.mutation_epoch(),
            epoch,
            "preventExtensions on the global must bump the epoch"
        );
        assert!(
            vm.take_dirtied_reset_objects().is_empty(),
            "the untracked global must dirty no target"
        );
        assert!(scrub_prototypes(&mut vm, &scrub, false));
        assert!(
            vm.is_extensible(&global),
            "global extensibility must be repaired, not leak into the next test"
        );
        assert_pristine(&vm, &scrub);
    }

    #[test]
    fn scrub_repairs_global_attribute_flip() {
        // Attribute-only redefine of a baseline global: the epoch bumps
        // (reset-global baseline key write) without dirtying any target,
        // and the harness baseline value scan sees no value change —
        // only the global target's attribute-triple check observes it,
        // and the repair re-defines the baseline descriptor.
        let mut vm = Vm::new();
        // Install a configurable baseline global before the snapshot;
        // engine builtins are non-configurable, so their attribute
        // flips are unrecoverable by re-definition (rebuild instead).
        vm.eval(
            "Object.defineProperty(globalThis, '__w30_g', \
             {value: 1, enumerable: true, configurable: true});",
        )
        .unwrap();
        let scrub = build_proto_scrub(&vm);
        track_scrub_targets(&mut vm, &scrub);
        let global = vm.get_global("globalThis").expect("globalThis");
        let baseline_keys: Vec<String> = vm
            .own_property_keys(&global)
            .iter()
            .map(|k| k.as_wtf8().to_string())
            .collect();
        vm.set_reset_global_baseline(&global, &baseline_keys);
        let epoch = vm.mutation_epoch();
        vm.eval("Object.defineProperty(globalThis, '__w30_g', {enumerable: false});")
            .unwrap();
        assert_ne!(
            vm.mutation_epoch(),
            epoch,
            "attribute redefine of a baseline global must bump the epoch"
        );
        assert!(
            vm.take_dirtied_reset_objects().is_empty(),
            "attribute redefine must dirty no target"
        );
        assert!(scrub_prototypes(&mut vm, &scrub, false));
        assert_pristine(&vm, &scrub);
    }

    #[test]
    fn scrub_repairs_mixed_tracked_and_global_pollution() {
        // A test that dirties a tracked target *and* touches the
        // untracked global in one run: the dirtied subset repairs the
        // prototype, and the global target's extensibility check repairs
        // the flag. Without the global check riding along on non-empty
        // dirtied sets, the `preventExtensions` half leaked into the
        // next test.
        let mut vm = Vm::new();
        let scrub = build_proto_scrub(&vm);
        track_scrub_targets(&mut vm, &scrub);
        let global = vm.get_global("globalThis").expect("globalThis");
        let baseline_keys: Vec<String> = vm
            .own_property_keys(&global)
            .iter()
            .map(|k| k.as_wtf8().to_string())
            .collect();
        vm.set_reset_global_baseline(&global, &baseline_keys);
        vm.eval("Array.prototype.__w30_mix = 1; Reflect.preventExtensions(globalThis);")
            .unwrap();
        // NOTE: the dirtied set is drained exactly once, by the scrub
        // itself — asserting it non-empty here would consume the
        // evidence the scrub needs (cf.
        // `scrub_repairs_delete_only_pollution`).
        assert!(scrub_prototypes(&mut vm, &scrub, false));
        assert!(
            vm.is_extensible(&global),
            "global extensibility must be repaired alongside the tracked target"
        );
        assert_pristine(&vm, &scrub);
    }

    #[test]
    fn rescrub_after_repair_is_quiet() {
        // The repair re-dirties its target (epoch-accounted), but the
        // post-repair drain + re-sync means a second scrub with no new
        // writes checks nothing and writes nothing.
        let mut vm = Vm::new();
        let scrub = build_proto_scrub(&vm);
        track_scrub_targets(&mut vm, &scrub);
        vm.eval("Array.prototype.__w29_only = 1;").unwrap();
        assert!(scrub_prototypes(&mut vm, &scrub, false));
        assert_pristine(&vm, &scrub);
        let epoch = vm.mutation_epoch();
        assert!(scrub_prototypes(&mut vm, &scrub, false));
        assert_eq!(
            vm.mutation_epoch(),
            epoch,
            "re-scrub after repair must not move the epoch"
        );
    }

    #[test]
    fn native_global_key_scrub() {
        let mut vm = Vm::new();
        let global = vm.get_global("globalThis").expect("globalThis");
        let mut baseline: Vec<String> = vm
            .own_property_keys(&global)
            .iter()
            .map(|k| k.as_wtf8().to_string())
            .collect();
        baseline.sort();
        // Added enumerable + non-enumerable globals are removed.
        vm.eval("globalThis.__w18_a = 1;").unwrap();
        vm.eval(
            "Object.defineProperty(globalThis, '__w18_b', \
             {value: 2, enumerable: false, configurable: true});",
        )
        .unwrap();
        assert!(scrub_global_keys(&mut vm, &global, &baseline, false));
        let mut keys: Vec<String> = vm
            .own_property_keys(&global)
            .iter()
            .map(|k| k.as_wtf8().to_string())
            .collect();
        keys.sort();
        assert_eq!(keys, baseline);
        // Added non-configurable global survives the delete: rebuild.
        vm.eval(
            "Object.defineProperty(globalThis, '__w18_c', \
             {value: 3, configurable: false});",
        )
        .unwrap();
        assert!(!scrub_global_keys(&mut vm, &global, &baseline, false));
    }

    #[test]
    fn scrub_target_count_sane() {
        // Smoke: the native classifier finds the same shape of targets
        // the old generated classifier did (constructors + namespaces).
        let vm = Vm::new();
        let scrub = build_proto_scrub(&vm);
        let ctors = scrub.targets.iter().filter(|t| t.is_ctor).count();
        let nss = scrub.targets.len() - ctors;
        assert!(ctors >= 10, "expected builtin constructors, got {ctors}");
        assert!(nss >= 3, "expected namespace objects, got {nss}");
        assert!(
            scrub
                .targets
                .iter()
                .any(|t| t.global_name == "Array" && t.is_ctor),
            "Array.prototype must be a scrub target"
        );
    }

    // The shared `%TypedArray%.prototype` — reachable only via
    // `Object.getPrototypeOf(Int8Array.prototype)`, with no `TypedArray`
    // global naming it — must be a scrub target: `verifyProperty`'s
    // `isConfigurable` does `delete obj[name]` with no restore, and
    // without a target the deletion persisted across tests in the reused
    // `Vm` (order-dependent failures).
    #[test]
    fn typedarray_shared_proto_is_scrub_target() {
        let mut vm = Vm::new();
        let scrub = build_proto_scrub(&vm);
        let protos: Vec<_> = scrub
            .targets
            .iter()
            .filter(|t| t.global_name.ends_with(".prototype.__proto__"))
            .collect();
        // Exactly one: the shared prototype is registered once for all
        // nine typed-array constructors; `Object.prototype` and
        // `Error.prototype`-style intermediates are already targets.
        assert_eq!(
            protos.len(),
            1,
            "exactly one intermediate-prototype target expected"
        );
        let t = protos[0];
        assert!(!t.is_ctor, "synthesized target is not a ctor target");
        assert!(!t.is_global, "synthesized target is not the global");
        assert!(
            t.keys.iter().any(|k| k.name == "map"),
            "map key snapshotted, got {} keys",
            t.keys.len()
        );
        // Poison exactly like the destructive helper: delete a method off
        // the shared prototype through the chain.
        track_scrub_targets(&mut vm, &scrub);
        vm.eval("delete Object.getPrototypeOf(Int8Array.prototype).map;")
            .expect("poison must run");
        assert!(
            scrub_prototypes(&mut vm, &scrub, false),
            "data-key delete must be repairable"
        );
        assert_pristine(&vm, &scrub);
    }
}
