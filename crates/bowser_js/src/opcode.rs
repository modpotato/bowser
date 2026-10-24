//! Stack-based bytecode for the Bowser JavaScript VM.
//!
//! Encoding: one `u8` opcode byte followed by inline little-endian operands.
//! See [`operand_width`] for the operand size of each opcode.
//!
//! # VM contract (stack protocols)
//!
//! The operand stack grows upward; `[a, b]` means `a` was pushed first.
//! Unless noted, an opcode name below is followed by its stack effect.
//!
//! * Names: [`Op::LoadName`]`(i)` / [`Op::StoreName`]`(i)` resolve
//!   `consts[i]` (a [`Constant::Str`][crate::compiler::Constant]) through the
//!   runtime scope chain (innermost environment outward, then the global
//!   object). `StoreName` pops the value; if the name is not found it is
//!   created on the global object in sloppy mode and throws `ReferenceError`
//!   in strict mode. The compiler pre-declares every known binding in the
//!   function prologue, so "not found" only happens for genuinely undeclared
//!   names. [`Op::LoadLocal`]`(i)` / [`Op::StoreLocal`]`(i)` are the
//!   compile-time-resolved twins: `i` is the environment slot index, so no
//!   name lookup happens at runtime.
//! * Upvalues: [`Op::LoadUpvalue`]`(i)` / [`Op::StoreUpvalue`]`(i)` read and
//!   write the closure's `i`-th captured cell. [`Op::NewFunction`]`(i)` builds
//!   the closure described by `consts[i]`
//!   ([`FuncTemplate`][crate::compiler::Constant]); for each
//!   [`UpvalueDesc`][crate::compiler::UpvalueDesc] it boxes the referenced
//!   binding *by cell*, so stores made after the capture are visible through
//!   the closure and vice versa.
//! * Calls: [`Op::Call`]`(argc)` expects
//!   `[callee, this_arg, arg_0, ..., arg_{n-1}]` and pushes the result.
//!   Plain calls pass `undefined` as `this_arg` (the VM maps it to the
//!   global object in sloppy mode). [`Op::New`]`(argc)` expects
//!   `[callee, arg_0, ..., arg_{n-1}]` and constructs with
//!   `new.target = callee`.
//! * Property access: [`Op::GetProp`] `[obj, key] -> [value]`,
//!   [`Op::SetProp`] `[obj, key, value] -> []`,
//!   [`Op::GetElem`] / [`Op::SetElem`] are the computed-key twins,
//!   [`Op::GetPropConst`]`(i)` `[obj] -> [value]` and
//!   [`Op::SetPropConst`]`(i)` `[obj, value] -> []` use `consts[i]` as the key.
//! * Jumps: [`Op::Jump`], [`Op::JumpIfFalse`], [`Op::JumpIfTrue`],
//!   [`Op::JumpIfNullish`] take a
//!   signed 16-bit offset *relative to the address of the next instruction*;
//!   `target = pc_after_instruction + offset`. The conditional jumps pop the
//!   tested value.
//! * Exceptions: [`Op::EnterTry`]`(handler)` pushes a handler record holding
//!   the absolute bytecode offset `handler` and the current stack depth.
//!   [`Op::Throw`] unwinds the value stack to that depth, pops the handler,
//!   pushes the thrown value, and resumes at `handler`. [`Op::LeaveTry`]
//!   pops the innermost handler. The compiler emits [`Op::LeaveTry`] on every
//!   abrupt exit (return/break/continue) that leaves a protected region;
//!   only the VM-driven unwind to a handler pops implicitly.
//! * Iteration: [`Op::IterNext`] expects `[iterable, index]` and pushes
//!   `[iterable, index+1, value, done]` (`done` on top; `value` is
//!   `undefined` when `done` is true). v1 supports arrays and strings only;
//!   anything else throws `TypeError`.
//! * [`Op::Yield`] / [`Op::Await`] mark generator/async suspension points.
//!   The compiler emits them verbatim; a VM without async/generator support
//!   must trap (throw) on them.

#[cfg(test)]
use crate::compiler::Constant;
#[cfg(test)]
use crate::wtf8::Wtf8Buf;

/// A single bytecode operation.
///
/// The `u8` discriminant is the on-wire opcode. Discriminants are explicit
/// and stable: bytecode serialized today must decode identically tomorrow.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    // -- constants -------------------------------------------------------
    PushUndefined = 0,
    PushNull = 1,
    PushTrue = 2,
    PushFalse = 3,
    /// Tiny integer literal, inline `i8`.
    PushInt(i8) = 4,
    /// Push `consts[idx]`.
    LoadConst(u16) = 5,

    // -- variables -------------------------------------------------------
    /// Push the value of the interned name `consts[idx]` (scope chain).
    LoadName(u16) = 6,
    /// Pop a value into the interned name `consts[idx]` (scope chain).
    StoreName(u16) = 7,
    /// Push closure upvalue `idx`.
    LoadUpvalue(u16) = 8,
    /// Pop a value into closure upvalue `idx`.
    StoreUpvalue(u16) = 9,
    /// Push the current `this` value.
    This = 10,
    /// Push `new.target` (`undefined` when not constructing).
    NewTarget = 11,

    // -- properties ------------------------------------------------------
    /// `[obj, key] -> [value]`.
    GetProp = 12,
    /// `[obj, key, value] -> []`.
    SetProp = 13,
    /// `[obj] -> [value]`; key is `consts[idx]` (a string).
    GetPropConst(u16) = 14,
    /// `[obj, value] -> []`; key is `consts[idx]` (a string).
    SetPropConst(u16) = 15,
    /// `[obj, key] -> [value]` (computed key).
    GetElem = 16,
    /// `[obj, key, value] -> []` (computed key).
    SetElem = 17,

    // -- arithmetic ------------------------------------------------------
    Add = 18,
    Sub = 19,
    Mul = 20,
    Div = 21,
    Mod = 22,
    Pow = 23,
    BitAnd = 24,
    BitOr = 25,
    BitXor = 26,
    Shl = 27,
    Shr = 28,
    ShrU = 29,

    // -- unary -----------------------------------------------------------
    Neg = 30,
    Pos = 31,
    Not = 32,
    BitNot = 33,
    Typeof = 34,
    /// `[v] -> [undefined]` (evaluates then discards, e.g. `void expr`).
    Void = 35,
    /// `[obj, key] -> [bool]`.
    Delete = 36,
    /// `[key, obj] -> [bool]` (`key in obj`).
    In = 37,
    /// `[obj, ctor] -> [bool]` (`obj instanceof ctor`).
    Instanceof = 38,

    // -- comparisons -----------------------------------------------------
    Eq = 39,
    Ne = 40,
    StrictEq = 41,
    StrictNe = 42,
    Lt = 43,
    Le = 44,
    Gt = 45,
    Ge = 46,

    // -- control flow ----------------------------------------------------
    /// Unconditional relative jump (`i16` from the next instruction).
    Jump(i16) = 47,
    /// Pop a value; jump if it is falsy.
    JumpIfFalse(i16) = 48,
    /// Pop a value; jump if it is truthy.
    JumpIfTrue(i16) = 49,

    // -- calls -----------------------------------------------------------
    /// `[callee, this, args..] -> [result]`; operand is `argc` (`u8`).
    Call(u8) = 50,
    /// `[callee, args..] -> [instance]`; operand is `argc` (`u8`).
    New(u8) = 51,
    /// `[value] -> []`; returns from the current function.
    Return = 52,
    /// `[value] -> []`; unwinds to the innermost handler.
    Throw = 53,

    // -- allocation ------------------------------------------------------
    /// `[] -> [{}]`.
    NewObject = 54,
    /// `[e0..e_{n-1}] -> [array]`; operand is the element count (`u8`).
    NewArray(u8) = 55,
    /// `[] -> [closure]`; `consts[idx]` is a `FuncTemplate`.
    NewFunction(u16) = 56,

    // -- stack manipulation ----------------------------------------------
    /// `[v] -> [v, v]`.
    Dup = 57,
    /// `[v] -> []`.
    Pop = 58,
    /// `[a, b] -> [b, a]`.
    Swap = 59,

    // -- exceptions ------------------------------------------------------
    /// Push a handler record; operand is the absolute catch-handler pc.
    EnterTry(u16) = 60,
    /// Pop the innermost handler record.
    LeaveTry = 61,

    // -- iteration / async -----------------------------------------------
    /// `[iterable, index] -> [iterable, index+1, value, done]`.
    IterNext = 62,
    /// Generator suspension point: `[v] -> [v]`.
    Yield = 63,
    /// Async suspension point: `[v] -> [v]`.
    Await = 64,

    Nop = 65,
    /// `[] -> [typeof name]`; like `LoadName` + `Typeof`, but an undeclared
    /// name yields `"undefined"` instead of throwing `ReferenceError`
    /// (spec 13.5.3). Emitted only for `typeof <ident>` resolving to the
    /// global scope; locals/upvalues keep the TDZ guard + `LoadName`.
    TypeofName(u16) = 66,
    /// Pop a value; jump if it is `null`/`undefined` (else push it back).
    /// The optional-chaining short-circuit: `[v] -> [v]`, or `[]` on jump.
    JumpIfNullish(i16) = 67,
    /// Push environment slot `idx`: a local variable whose slot was
    /// resolved at compile time (see the peephole in
    /// [`crate::compiler::Compiler::pop_function`]). Same value as
    /// `LoadName` would produce for that binding, without the
    /// name-constant lookup.
    LoadLocal(u16) = 68,
    /// Pop a value into environment slot `idx`: the store twin of
    /// [`Op::LoadLocal`].
    StoreLocal(u16) = 69,
    /// `[obj] -> [value]`: brand-checked private member load. `brand` is
    /// the lexically enclosing class's private brand, `idx` names the
    /// `"#name"` key constant. Throws `TypeError` when `obj` does not
    /// carry the class's runtime brand; otherwise reads the VM-side
    /// private slot table (prototype chain included, accessors invoked
    /// with `obj` as receiver). The `u16` brand operand is the
    /// compile-time brand (disassembly aid); the VM resolves the runtime
    /// brand from the executing function.
    GetPrivate(u16, u16) = 70,
    /// `[obj, value] -> []`: brand-checked private member store; the
    /// store twin of [`Op::GetPrivate`]. Writes the VM-side slot table:
    /// an accessor slot's setter runs, otherwise `obj`'s own data slot is
    /// (re)defined (shadowing an inherited one, as before).
    SetPrivate(u16, u16) = 71,
    /// `[obj] -> [bool]`: `#name in obj`. Throws `TypeError` when `obj`
    /// is not an object; returns false when `obj` does not carry the
    /// class's runtime brand; otherwise reports whether a `(brand,
    /// "#name")` slot is present on `obj` or its prototype chain. The
    /// `u16` operand is the compile-time brand (disassembly aid); the VM
    /// resolves the runtime brand from the executing function.
    HasPrivate(u16, u16) = 72,
    /// `[obj] -> [obj]`: add the class's runtime brand to `obj`'s brand set
    /// (idempotent). Emitted once per class evaluation on the constructor
    /// (for static private members) and once per constructor invocation
    /// on the fresh instance. The `u16` operand is the compile-time brand
    /// (disassembly aid); the VM resolves the runtime brand from the
    /// constructor under construction or the executing function.
    AddPrivateBrand(u16) = 73,
    /// `[obj, value] -> []`: define (or replace) `obj`'s own private data
    /// slot for `(brand, "#name")`, without a brand check. The install
    /// path for private fields and methods: field initializers, static
    /// field initializers, and method installation on the prototype or
    /// constructor. Private state lives in a VM-side table keyed by
    /// `(object, brand, name)` — never as string-keyed properties.
    InitPrivate(u16, u16) = 87,
    /// `[holder, getter, setter] -> [holder]`: define (or replace)
    /// `holder`'s own private accessor slot for `(brand, "#name")`,
    /// without a brand check. The install path for private getters/setters
    /// on the prototype or constructor. Either side may be absent
    /// (`undefined` pushed for a missing getter/setter).
    DefinePrivateAccessor(u16, u16) = 88,
    /// `[receiver, base, key] -> [value]`: `super` property load. `base`
    /// is the home object's prototype; getters run with `receiver`.
    SuperGet = 74,
    /// `[receiver, base, key, value] -> []`: `super` property store:
    /// `OrdinarySet(base, key, value, receiver)` — setters run with
    /// `receiver`, and new data properties are created on `receiver`.
    SuperSet = 75,
    /// `[iterable] -> [value]`: `yield*` delegation. Iterates the operand
    /// with the `@@iterator` protocol, suspending with each yielded value
    /// while the inner iterator is unfinished; evaluates to the inner
    /// iterator's return value once it completes.
    YieldStar = 76,
    /// `[] -> [arguments]`: build the call's `arguments` object from the
    /// current frame's actual arguments (indexed elements, `length`,
    /// `callee`).
    NewArguments = 77,
    /// `[rhs] -> []`: `name += rhs` with a discarded value (statement
    /// position). `idx` is the name constant, exactly like [`Op::StoreName`];
    /// the peephole in [`crate::compiler::Compiler::pop_function`] rewrites
    /// it to [`Op::StrAppendLocal`] for compile-time-resolved locals, or to
    /// [`Op::StrAppendBaseName`] when the rhs may rebind the target.
    StrAppendName(u16) = 78,
    /// `[rhs] -> []`: fused string-append to local slot `idx`. The VM keeps
    /// a per-frame chunk accumulator, so `s += x` in a loop costs amortized
    /// O(1) per append; the first read of the slot materializes the full
    /// string with a single allocation. Emitted only when the rhs cannot
    /// rebind the target; otherwise [`Op::StrAppendBaseLocal`] is used.
    StrAppendLocal(u16) = 79,
    /// `[callee, this, args..] -> [result]`: direct `eval` call. Emitted
    /// only for a bare `eval(...)` identifier call (never for aliases,
    /// `(0, eval)(...)`, or member calls). The VM checks the callee: when
    /// it is the intrinsic `eval` function the argument is evaluated as a
    /// script in the *calling* context (direct eval); any other callee is
    /// an ordinary [`Op::Call`]. Operand is `argc` (`u8`).
    EvalCall(u8) = 80,
    /// `[callee, this, args_array] -> [result]`: direct `eval` call with
    /// spread arguments (`eval(...args)`); the direct/indirect split is
    /// the same as [`Op::EvalCall`].
    EvalCallSpread = 81,
    /// `[start] -> [array]`: the current frame's arguments from `start`
    /// onward as a fresh array. Replaces the `#rest_args` intrinsic: it
    /// runs inline, so it works while a `vm_call` closure is on the stack
    /// (e.g. inside generator delegation), where a native re-entry would
    /// fail closed.
    RestArgs = 82,
    /// `[arr, iterable] -> [arr]`: append spread elements to `arr`
    /// (array literals, spread calls, spread `new`, `super(...args)`).
    /// Replaces the `#arr_append_spread` intrinsic.
    ArrAppendSpread = 83,
    /// `[pattern, flags] -> [regexp]`: construct via the global `RegExp`.
    /// Replaces the `#new_regex` intrinsic.
    NewRegex = 84,
    /// `[spec] -> [promise]`: dynamic `import()` (v1: rejected promise).
    /// Replaces the `#dynamic_import` intrinsic.
    DynamicImport = 85,
    /// `[callee, this, args_array] -> [result]`: spread call
    /// (`callee(...args)`). Replaces the `#apply` intrinsic.
    SpreadCall = 86,
    /// `[base, rhs] -> []`: statement-position `name += rhs` where `base`
    /// was captured from the slot *before* the rhs ran. `idx` is the name
    /// constant, exactly like [`Op::StoreName`]; the peephole rewrites it
    /// to [`Op::StrAppendBaseLocal`] for compile-time-resolved locals.
    /// The compiler emits this instead of [`Op::StrAppendName`] when the
    /// rhs may rebind the target (e.g. `s += (s = "Z", "c")`), so the
    /// append still sees the pre-rhs value, as the spec requires.
    StrAppendBaseName(u16) = 89,
    /// `[base, rhs] -> []`: fused string-append to local slot `idx` with
    /// the pre-rhs base value already on the stack (like [`Op::StoreLocal`]
    /// for the operand). The VM snapshots it across rhs evaluation, so a
    /// rhs that rebinds or captures the slot still appends to the value
    /// the statement started with.
    StrAppendBaseLocal(u16) = 90,
    /// `[] -> []`: push an iterator-close guard onto the frame's handler
    /// stack (see [`Op::IterClose`]). Emitted once per `for-of` loop (after
    /// `[iterable, 0]`) and once per array-pattern destructuring (after
    /// `[value, state]`): an abrupt completion unwinding through the guard
    /// runs `IteratorClose` on the in-flight iterator before continuing to
    /// unwind (spec 14.7.5 / 13.15.5).
    IterGuard = 91,
    /// `[] -> []`: pop the innermost handler record, which must be an
    /// iterator-close guard. The normal-completion twin of the
    /// [`Op::IterClose`] family: loop exhaustion (or a finished
    /// destructuring) disarms the guard without closing the iterator.
    IterUnguard = 92,
    /// `[iterable, state] -> []`: `IteratorClose` for an abrupt `for-of`
    /// (or destructuring) exit with no value on top (`break`, or the
    /// compiler-inlined unwind path). When `state` is an iterator object,
    /// calls its `return` method if present (a throw from `return()`
    /// replaces the in-flight abrupt completion, per spec); the index
    /// fast-path state needs no close. Pops the pair and the loop's
    /// [`Op::IterGuard`] guard.
    IterClose = 93,
    /// `[iterable, state, retval] -> [retval]`: the `return`-statement twin
    /// of [`Op::IterClose`]: the return value sits on top of the loop pair,
    /// so the close runs underneath it and the value is preserved.
    IterCloseRet = 94,
    /// `[] -> [symbol]`: push the well-known symbol with discriminant
    /// `b` (a [`crate::gc::WellKnown`] discriminant), or `undefined` when
    /// builtins have not installed it. Lets the compiler name a symbol
    /// key without depending on the `Symbol` binding (which user code
    /// can shadow); pairs with [`Op::GetElem`] for symbol-keyed reads.
    PushWellKnownSymbol(u8) = 95,
}

/// Byte length of the inline operands that follow the opcode byte.
pub fn operand_width(op: Op) -> usize {
    match op {
        Op::PushInt(_)
        | Op::Call(_)
        | Op::New(_)
        | Op::NewArray(_)
        | Op::EvalCall(_)
        | Op::PushWellKnownSymbol(_) => 1,
        Op::LoadConst(_)
        | Op::LoadName(_)
        | Op::StoreName(_)
        | Op::LoadLocal(_)
        | Op::StoreLocal(_)
        | Op::LoadUpvalue(_)
        | Op::StoreUpvalue(_)
        | Op::GetPropConst(_)
        | Op::SetPropConst(_)
        | Op::TypeofName(_)
        | Op::Jump(_)
        | Op::JumpIfFalse(_)
        | Op::JumpIfTrue(_)
        | Op::JumpIfNullish(_)
        | Op::NewFunction(_)
        | Op::EnterTry(_)
        | Op::AddPrivateBrand(_)
        | Op::StrAppendName(_)
        | Op::StrAppendLocal(_)
        | Op::StrAppendBaseName(_)
        | Op::StrAppendBaseLocal(_) => 2,
        Op::GetPrivate(_, _)
        | Op::SetPrivate(_, _)
        | Op::HasPrivate(_, _)
        | Op::InitPrivate(_, _)
        | Op::DefinePrivateAccessor(_, _) => 4,
        _ => 0,
    }
}

/// The stable on-wire opcode byte for `op` (mirrors the discriminants).
pub fn tag(op: Op) -> u8 {
    match op {
        Op::PushUndefined => 0,
        Op::PushNull => 1,
        Op::PushTrue => 2,
        Op::PushFalse => 3,
        Op::PushInt(_) => 4,
        Op::LoadConst(_) => 5,
        Op::LoadName(_) => 6,
        Op::StoreName(_) => 7,
        Op::LoadUpvalue(_) => 8,
        Op::StoreUpvalue(_) => 9,
        Op::This => 10,
        Op::NewTarget => 11,
        Op::GetProp => 12,
        Op::SetProp => 13,
        Op::GetPropConst(_) => 14,
        Op::SetPropConst(_) => 15,
        Op::GetElem => 16,
        Op::SetElem => 17,
        Op::Add => 18,
        Op::Sub => 19,
        Op::Mul => 20,
        Op::Div => 21,
        Op::Mod => 22,
        Op::Pow => 23,
        Op::BitAnd => 24,
        Op::BitOr => 25,
        Op::BitXor => 26,
        Op::Shl => 27,
        Op::Shr => 28,
        Op::ShrU => 29,
        Op::Neg => 30,
        Op::Pos => 31,
        Op::Not => 32,
        Op::BitNot => 33,
        Op::Typeof => 34,
        Op::Void => 35,
        Op::Delete => 36,
        Op::In => 37,
        Op::Instanceof => 38,
        Op::Eq => 39,
        Op::Ne => 40,
        Op::StrictEq => 41,
        Op::StrictNe => 42,
        Op::Lt => 43,
        Op::Le => 44,
        Op::Gt => 45,
        Op::Ge => 46,
        Op::Jump(_) => 47,
        Op::JumpIfFalse(_) => 48,
        Op::JumpIfTrue(_) => 49,
        Op::Call(_) => 50,
        Op::New(_) => 51,
        Op::Return => 52,
        Op::Throw => 53,
        Op::NewObject => 54,
        Op::NewArray(_) => 55,
        Op::NewFunction(_) => 56,
        Op::Dup => 57,
        Op::Pop => 58,
        Op::Swap => 59,
        Op::EnterTry(_) => 60,
        Op::LeaveTry => 61,
        Op::IterNext => 62,
        Op::Yield => 63,
        Op::Await => 64,
        Op::Nop => 65,
        Op::TypeofName(_) => 66,
        Op::JumpIfNullish(_) => 67,
        Op::LoadLocal(_) => 68,
        Op::StoreLocal(_) => 69,
        Op::GetPrivate(_, _) => 70,
        Op::SetPrivate(_, _) => 71,
        Op::HasPrivate(_, _) => 72,
        Op::AddPrivateBrand(_) => 73,
        Op::InitPrivate(_, _) => 87,
        Op::DefinePrivateAccessor(_, _) => 88,
        Op::SuperGet => 74,
        Op::SuperSet => 75,
        Op::YieldStar => 76,
        Op::NewArguments => 77,
        Op::StrAppendName(_) => 78,
        Op::StrAppendLocal(_) => 79,
        Op::EvalCall(_) => 80,
        Op::EvalCallSpread => 81,
        Op::RestArgs => 82,
        Op::ArrAppendSpread => 83,
        Op::NewRegex => 84,
        Op::DynamicImport => 85,
        Op::SpreadCall => 86,
        Op::StrAppendBaseName(_) => 89,
        Op::StrAppendBaseLocal(_) => 90,
        Op::IterGuard => 91,
        Op::IterUnguard => 92,
        Op::IterClose => 93,
        Op::IterCloseRet => 94,
        Op::PushWellKnownSymbol(_) => 95,
    }
}

/// Append the encoding of `op` (opcode byte plus little-endian operands).
pub fn encode(op: Op, out: &mut Vec<u8>) {
    out.push(tag(op));
    match op {
        Op::PushInt(v) => out.push(v as u8),
        Op::Call(n)
        | Op::New(n)
        | Op::NewArray(n)
        | Op::EvalCall(n)
        | Op::PushWellKnownSymbol(n) => out.push(n),
        Op::LoadConst(i)
        | Op::LoadName(i)
        | Op::StoreName(i)
        | Op::LoadLocal(i)
        | Op::StoreLocal(i)
        | Op::StrAppendName(i)
        | Op::StrAppendLocal(i)
        | Op::StrAppendBaseName(i)
        | Op::StrAppendBaseLocal(i)
        | Op::LoadUpvalue(i)
        | Op::StoreUpvalue(i)
        | Op::GetPropConst(i)
        | Op::SetPropConst(i)
        | Op::NewFunction(i)
        | Op::TypeofName(i)
        | Op::EnterTry(i) => out.extend_from_slice(&i.to_le_bytes()),
        Op::Jump(o) | Op::JumpIfFalse(o) | Op::JumpIfTrue(o) | Op::JumpIfNullish(o) => {
            out.extend_from_slice(&o.to_le_bytes());
        }
        Op::AddPrivateBrand(b) => out.extend_from_slice(&b.to_le_bytes()),
        Op::GetPrivate(b, i)
        | Op::SetPrivate(b, i)
        | Op::HasPrivate(b, i)
        | Op::InitPrivate(b, i)
        | Op::DefinePrivateAccessor(b, i) => {
            out.extend_from_slice(&b.to_le_bytes());
            out.extend_from_slice(&i.to_le_bytes());
        }
        _ => {}
    }
}

fn read_u16(code: &[u8], pc: usize) -> Option<u16> {
    let b = code.get(pc..pc + 2)?;
    Some(u16::from_le_bytes([b[0], b[1]]))
}

fn read_i16(code: &[u8], pc: usize) -> Option<i16> {
    let b = code.get(pc..pc + 2)?;
    Some(i16::from_le_bytes([b[0], b[1]]))
}

/// Decode the instruction at `pc`, returning it and the next `pc`.
///
/// Returns `None` for a truncated instruction or an unknown opcode byte.
pub fn decode(code: &[u8], pc: usize) -> Option<(Op, usize)> {
    let byte = *code.get(pc)?;
    let op = match byte {
        0 => Op::PushUndefined,
        1 => Op::PushNull,
        2 => Op::PushTrue,
        3 => Op::PushFalse,
        4 => Op::PushInt(*code.get(pc + 1)? as i8),
        5 => Op::LoadConst(read_u16(code, pc + 1)?),
        6 => Op::LoadName(read_u16(code, pc + 1)?),
        7 => Op::StoreName(read_u16(code, pc + 1)?),
        8 => Op::LoadUpvalue(read_u16(code, pc + 1)?),
        9 => Op::StoreUpvalue(read_u16(code, pc + 1)?),
        10 => Op::This,
        11 => Op::NewTarget,
        12 => Op::GetProp,
        13 => Op::SetProp,
        14 => Op::GetPropConst(read_u16(code, pc + 1)?),
        15 => Op::SetPropConst(read_u16(code, pc + 1)?),
        16 => Op::GetElem,
        17 => Op::SetElem,
        18 => Op::Add,
        19 => Op::Sub,
        20 => Op::Mul,
        21 => Op::Div,
        22 => Op::Mod,
        23 => Op::Pow,
        24 => Op::BitAnd,
        25 => Op::BitOr,
        26 => Op::BitXor,
        27 => Op::Shl,
        28 => Op::Shr,
        29 => Op::ShrU,
        30 => Op::Neg,
        31 => Op::Pos,
        32 => Op::Not,
        33 => Op::BitNot,
        34 => Op::Typeof,
        35 => Op::Void,
        36 => Op::Delete,
        37 => Op::In,
        38 => Op::Instanceof,
        39 => Op::Eq,
        40 => Op::Ne,
        41 => Op::StrictEq,
        42 => Op::StrictNe,
        43 => Op::Lt,
        44 => Op::Le,
        45 => Op::Gt,
        46 => Op::Ge,
        47 => Op::Jump(read_i16(code, pc + 1)?),
        48 => Op::JumpIfFalse(read_i16(code, pc + 1)?),
        49 => Op::JumpIfTrue(read_i16(code, pc + 1)?),
        50 => Op::Call(*code.get(pc + 1)?),
        51 => Op::New(*code.get(pc + 1)?),
        52 => Op::Return,
        53 => Op::Throw,
        54 => Op::NewObject,
        55 => Op::NewArray(*code.get(pc + 1)?),
        56 => Op::NewFunction(read_u16(code, pc + 1)?),
        57 => Op::Dup,
        58 => Op::Pop,
        59 => Op::Swap,
        60 => Op::EnterTry(read_u16(code, pc + 1)?),
        61 => Op::LeaveTry,
        62 => Op::IterNext,
        63 => Op::Yield,
        64 => Op::Await,
        65 => Op::Nop,
        66 => Op::TypeofName(read_u16(code, pc + 1)?),
        67 => Op::JumpIfNullish(read_i16(code, pc + 1)?),
        68 => Op::LoadLocal(read_u16(code, pc + 1)?),
        69 => Op::StoreLocal(read_u16(code, pc + 1)?),
        70 => Op::GetPrivate(read_u16(code, pc + 1)?, read_u16(code, pc + 3)?),
        71 => Op::SetPrivate(read_u16(code, pc + 1)?, read_u16(code, pc + 3)?),
        72 => Op::HasPrivate(read_u16(code, pc + 1)?, read_u16(code, pc + 3)?),
        73 => Op::AddPrivateBrand(read_u16(code, pc + 1)?),
        87 => Op::InitPrivate(read_u16(code, pc + 1)?, read_u16(code, pc + 3)?),
        88 => Op::DefinePrivateAccessor(read_u16(code, pc + 1)?, read_u16(code, pc + 3)?),
        74 => Op::SuperGet,
        75 => Op::SuperSet,
        76 => Op::YieldStar,
        77 => Op::NewArguments,
        78 => Op::StrAppendName(read_u16(code, pc + 1)?),
        79 => Op::StrAppendLocal(read_u16(code, pc + 1)?),
        80 => Op::EvalCall(*code.get(pc + 1)?),
        81 => Op::EvalCallSpread,
        82 => Op::RestArgs,
        83 => Op::ArrAppendSpread,
        84 => Op::NewRegex,
        85 => Op::DynamicImport,
        86 => Op::SpreadCall,
        89 => Op::StrAppendBaseName(read_u16(code, pc + 1)?),
        90 => Op::StrAppendBaseLocal(read_u16(code, pc + 1)?),
        91 => Op::IterGuard,
        92 => Op::IterUnguard,
        93 => Op::IterClose,
        94 => Op::IterCloseRet,
        95 => Op::PushWellKnownSymbol(*code.get(pc + 1)?),
        _ => return None,
    };
    Some((op, pc + 1 + operand_width(op)))
}

#[cfg(test)]
fn fmt_const(c: &Constant) -> String {
    match c {
        Constant::Str(s) => format!("{s:?}"),
        Constant::Float(f) => format!("{f:?}"),
        Constant::Int(i) => format!("{i}"),
        Constant::FuncTemplate(t) => {
            let kind = match (t.is_async, t.is_generator) {
                (true, true) => "async generator",
                (true, false) => "async",
                (false, true) => "generator",
                (false, false) => "function",
            };
            format!("<{kind} {} arity={}>", t.name, t.arity)
        }
    }
}

#[cfg(test)]
fn fmt_op(pc: usize, op: Op, consts: &[Constant]) -> String {
    let name_idx = |i: u16| {
        consts
            .get(i as usize)
            .map(fmt_const)
            .unwrap_or_else(|| "<bad const>".to_string())
    };
    let target = |off: i16| (pc + 1 + operand_width(op)) as isize + off as isize;
    match op {
        Op::PushInt(v) => format!("PushInt {v}"),
        Op::LoadConst(i) => format!("LoadConst {i} ; {}", name_idx(i)),
        Op::LoadName(i) => format!("LoadName {i} ; {}", name_idx(i)),
        Op::TypeofName(i) => format!("TypeofName {i} ; {}", name_idx(i)),
        Op::StoreName(i) => format!("StoreName {i} ; {}", name_idx(i)),
        Op::LoadLocal(i) => format!("LoadLocal {i}"),
        Op::StoreLocal(i) => format!("StoreLocal {i}"),
        Op::StrAppendName(i) => format!("StrAppendName {i} ; {}", name_idx(i)),
        Op::StrAppendLocal(i) => format!("StrAppendLocal {i}"),
        Op::StrAppendBaseName(i) => format!("StrAppendBaseName {i} ; {}", name_idx(i)),
        Op::StrAppendBaseLocal(i) => format!("StrAppendBaseLocal {i}"),
        Op::LoadUpvalue(i) => format!("LoadUpvalue {i}"),
        Op::StoreUpvalue(i) => format!("StoreUpvalue {i}"),
        Op::GetPropConst(i) => format!("GetPropConst {i} ; {}", name_idx(i)),
        Op::SetPropConst(i) => format!("SetPropConst {i} ; {}", name_idx(i)),
        Op::NewFunction(i) => format!("NewFunction {i} ; {}", name_idx(i)),
        Op::Call(n) => format!("Call {n}"),
        Op::EvalCall(n) => format!("EvalCall {n}"),
        Op::EvalCallSpread => "EvalCallSpread".to_string(),
        Op::New(n) => format!("New {n}"),
        Op::NewArray(n) => format!("NewArray {n}"),
        Op::Jump(o) => format!("Jump {o:+} ; -> {:04}", target(o)),
        Op::JumpIfFalse(o) => format!("JumpIfFalse {o:+} ; -> {:04}", target(o)),
        Op::JumpIfTrue(o) => format!("JumpIfTrue {o:+} ; -> {:04}", target(o)),
        Op::JumpIfNullish(o) => format!("JumpIfNullish {o:+} ; -> {:04}", target(o)),
        Op::EnterTry(h) => format!("EnterTry {h} ; handler {:04}", h),
        Op::GetPrivate(b, i) => format!("GetPrivate {b} {} ; {}", i, name_idx(i)),
        Op::SetPrivate(b, i) => format!("SetPrivate {b} {} ; {}", i, name_idx(i)),
        Op::HasPrivate(b, i) => format!("HasPrivate {b} {} ; {}", i, name_idx(i)),
        Op::AddPrivateBrand(b) => format!("AddPrivateBrand {b}"),
        Op::InitPrivate(b, i) => format!("InitPrivate {b} {} ; {}", i, name_idx(i)),
        Op::DefinePrivateAccessor(b, i) => {
            format!("DefinePrivateAccessor {b} {} ; {}", i, name_idx(i))
        }
        _ => format!("{op:?}"),
    }
}

/// Disassemble `code` into human-readable text, one instruction per line.
///
/// Format: `{pc:04}: <op> [operand] [; annotation]`. Constant references are
/// annotated with the constant's value; jumps with their absolute target.
/// The output is deterministic for a given `(code, consts)` pair, which the
/// test suite relies on for round-trip stability checks.
#[cfg(test)]
pub fn disassemble(code: &[u8], consts: &[Constant]) -> String {
    let mut out = String::new();
    let mut pc = 0;
    while pc < code.len() {
        match decode(code, pc) {
            Some((op, next)) => {
                out.push_str(&format!("{pc:04}: {}\n", fmt_op(pc, op, consts)));
                pc = next;
            }
            None => {
                out.push_str(&format!("{pc:04}: <invalid opcode {:02x}>\n", code[pc]));
                pc += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every opcode must survive an encode/decode round trip, and
    /// `operand_width` must agree with the decoded instruction length.
    #[test]
    fn round_trip_all_ops() {
        let ops = [
            Op::PushUndefined,
            Op::PushNull,
            Op::PushTrue,
            Op::PushFalse,
            Op::PushInt(-128),
            Op::PushInt(0),
            Op::PushInt(127),
            Op::LoadConst(0),
            Op::LoadConst(65535),
            Op::LoadName(7),
            Op::TypeofName(7),
            Op::StoreName(7),
            Op::LoadLocal(7),
            Op::StoreLocal(7),
            Op::StrAppendName(7),
            Op::StrAppendLocal(7),
            Op::LoadUpvalue(3),
            Op::StoreUpvalue(3),
            Op::This,
            Op::NewTarget,
            Op::GetProp,
            Op::SetProp,
            Op::GetPropConst(9),
            Op::SetPropConst(9),
            Op::GetElem,
            Op::SetElem,
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::Div,
            Op::Mod,
            Op::Pow,
            Op::BitAnd,
            Op::BitOr,
            Op::BitXor,
            Op::Shl,
            Op::Shr,
            Op::ShrU,
            Op::Neg,
            Op::Pos,
            Op::Not,
            Op::BitNot,
            Op::Typeof,
            Op::Void,
            Op::Delete,
            Op::In,
            Op::Instanceof,
            Op::Eq,
            Op::Ne,
            Op::StrictEq,
            Op::StrictNe,
            Op::Lt,
            Op::Le,
            Op::Gt,
            Op::Ge,
            Op::Jump(-32768),
            Op::Jump(32767),
            Op::JumpIfFalse(-5),
            Op::JumpIfTrue(42),
            Op::Call(0),
            Op::Call(255),
            Op::New(2),
            Op::Return,
            Op::Throw,
            Op::NewObject,
            Op::NewArray(0),
            Op::NewArray(255),
            Op::NewFunction(1234),
            Op::Dup,
            Op::Pop,
            Op::Swap,
            Op::EnterTry(0),
            Op::EnterTry(65535),
            Op::LeaveTry,
            Op::IterNext,
            Op::Yield,
            Op::Await,
            Op::Nop,
            Op::TypeofName(7),
            Op::JumpIfNullish(-3),
            Op::GetPrivate(3, 7),
            Op::SetPrivate(3, 7),
            Op::HasPrivate(3, 7),
            Op::AddPrivateBrand(3),
            Op::InitPrivate(3, 7),
            Op::DefinePrivateAccessor(3, 7),
            Op::SuperGet,
            Op::SuperSet,
            Op::YieldStar,
            Op::NewArguments,
            Op::StrAppendName(7),
            Op::StrAppendLocal(7),
            Op::EvalCall(0),
            Op::EvalCall(255),
            Op::EvalCallSpread,
            Op::RestArgs,
            Op::ArrAppendSpread,
            Op::NewRegex,
            Op::DynamicImport,
            Op::SpreadCall,
            Op::StrAppendBaseName(7),
            Op::StrAppendBaseLocal(7),
            Op::PushWellKnownSymbol(3),
            Op::IterGuard,
            Op::IterUnguard,
            Op::IterClose,
            Op::IterCloseRet,
        ];
        assert_eq!(ops.len(), 107, "every opcode must be covered");
        // Every discriminant 0..=95 must appear at least once.
        let mut seen = [false; 96];
        for op in ops {
            seen[tag(op) as usize] = true;
        }
        assert!(seen.iter().all(|s| *s), "missing opcode discriminants");
        for op in ops {
            let mut buf = Vec::new();
            encode(op, &mut buf);
            assert_eq!(
                buf.len(),
                1 + operand_width(op),
                "width mismatch for {op:?}"
            );
            let (back, next) =
                decode(&buf, 0).unwrap_or_else(|| panic!("decode failed for {op:?}"));
            assert_eq!(back, op);
            assert_eq!(next, buf.len());
        }
    }

    #[test]
    fn decode_rejects_garbage() {
        assert_eq!(decode(&[200], 0), None);
        assert_eq!(decode(&[], 0), None);
        // Truncated operand.
        assert_eq!(decode(&[5, 0], 0), None);
        assert_eq!(decode(&[47, 1], 0), None);
    }

    #[test]
    fn disassemble_is_stable_and_annotated() {
        let consts = vec![
            Constant::Str(Wtf8Buf::from("hello")),
            Constant::Int(42),
            Constant::FuncTemplate(crate::compiler::FuncTemplate {
                name: "f".to_string(),
                arity: 1,
                code: vec![],
                consts: vec![],
                upvalues: vec![],
                upvalue_names: vec![],
                binding_names: vec![],
                var_names: vec![],
                binding_kinds: vec![],
                is_async: false,
                is_generator: false,
                strict: false,
                is_method: false,
                is_derived_ctor: false,
                mapped_arg_slots: None,
                script_or_module: None,
            }),
        ];
        let mut code = Vec::new();
        encode(Op::PushInt(5), &mut code);
        encode(Op::LoadConst(0), &mut code);
        encode(Op::Add, &mut code);
        encode(Op::JumpIfFalse(3), &mut code);
        encode(Op::NewFunction(2), &mut code);
        encode(Op::Return, &mut code);
        let a = disassemble(&code, &consts);
        let b = disassemble(&code, &consts);
        assert_eq!(a, b, "disassembly must be deterministic");
        assert!(
            a.contains("LoadConst 0 ; \"hello\""),
            "const annotation missing:\n{a}"
        );
        assert!(
            a.contains("NewFunction 2 ; <function f arity=1>"),
            "template annotation missing:\n{a}"
        );
        assert!(
            a.contains("-> 0012"),
            "jump target annotation missing:\n{a}"
        );
    }
}
