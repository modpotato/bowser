# Fuzzing bowser's TLS 1.3 / X.509 / HTML parsers and JS engine

Four libfuzzer harnesses (via `cargo-fuzz`) for the hand-rolled,
never-audited TLS 1.3, X.509, and HTML code in `bowser_net` / `bowser_html`,
plus the JavaScript engine in `bowser_js`:

| target | file | what it fuzzes |
|---|---|---|
| `fuzz_tls_handshake` | `fuzz_targets/fuzz_tls_handshake.rs` | TLS record layer, handshake-message framer, ServerHello / extensions / alert parsing, and the encrypted-flight loop. Input bytes are replayed as the *server* flight over a loopback TCP pair into the real `TlsConnection::connect`. |
| `fuzz_x509` | `fuzz_targets/fuzz_x509.rs` | X.509 DER parsing (`parse_cert`) and full RFC 5280 chain validation (`verify_server_chain`) against the vendored Mozilla roots. `x509` is `pub(crate)` upstream, so the target recompiles the real sources in-tree via `#[path]` includes — no crate source is modified or copied. |
| `fuzz_html` | `fuzz_targets/fuzz_html.rs` | HTML tokenizer + tree builder (`bowser_html`): full document parse plus `div`- and `table`-context fragment parses (adoption agency, foster parenting). Expected: a tree, possibly with parse errors. Panic / OOM / hang = bug. |
| `fuzz_js` | `fuzz_targets/fuzz_js.rs` | JavaScript engine (`bowser_js`): input bytes decoded as lossy UTF-8 and parsed + evaluated as a classic script in a fresh `Vm` per iteration, covering the lexer, parser, compiler, bytecode VM, and the GC / `with_active` raw-pointer surface. Runaway scripts are bounded by the engine's per-eval fuel budget; expected: a completion value or an ordinary `VmError`. Panic / OOM / hang = bug. |

`fuzz/` is its own workspace (detached via `[workspace]`), so the main
build is untouched: `cargo check --workspace` at the repo root never sees
these targets, and `libfuzzer-sys` never appears in the main `cargo tree`.

## Prerequisites

```sh
rustup toolchain install nightly          # cargo-fuzz needs nightly
cargo +nightly install cargo-fuzz --locked
```

Use this target dir (never the shared workspace `target/`):

```sh
export CARGO_TARGET_DIR=~/workspace/scratch/bowser-orch5/fix-fuzz/target
```

## Smoke run (what CI / a quick check should do)

```sh
cd ~/workspace/bowser/fuzz
export CARGO_TARGET_DIR=~/workspace/scratch/bowser-orch5/fix-fuzz/target
cargo +nightly fuzz run fuzz_tls_handshake -- -max_total_time=600 -timeout=30 -rss_limit_mb=2048
cargo +nightly fuzz run fuzz_x509          -- -max_total_time=600 -timeout=30 -rss_limit_mb=2048
```

## Full campaign

Longer wall-clock, more workers, and a shared corpus dir so findings and
interesting inputs persist across runs:

```sh
cd ~/workspace/bowser/fuzz
export CARGO_TARGET_DIR=~/workspace/scratch/bowser-orch5/fix-fuzz/target

# build once
cargo +nightly fuzz build -O

# TLS handshake parser: 4 workers, 24 h, corpus kept in fuzz/corpus/tls
cargo +nightly fuzz run fuzz_tls_handshake fuzz/corpus/tls \
  -workers=4 -max_total_time=86400 -timeout=60 -rss_limit_mb=4096 \
  -print_final_stats=1

# X.509 parser: 4 workers, 24 h
cargo +nightly fuzz run fuzz_x509 fuzz/corpus/x509 \
  -workers=4 -max_total_time=86400 -timeout=60 -rss_limit_mb=4096 \
  -print_final_stats=1
```

Seed the corpus with real inputs to get past the shallow-error plateau:

```sh
# a real DER cert, e.g. from the vendored bundle fixtures or:
openssl req -x509 -newkey rsa:2048 -keyout /dev/null -out seed.der \
  -outform DER -days 1 -nodes -subj "/CN=fuzz.invalid"
mkdir -p fuzz/corpus/x509 && cp seed.der fuzz/corpus/x509/
```

(`fuzz/corpus/` is git-ignored; crashes land in `fuzz/artifacts/<target>/`
as `crash-<sha1>` / `timeout-<sha1>` / `oom-<sha1>` plus the libfuzzer log.)

## Triaging a crash

```sh
export CARGO_TARGET_DIR=~/workspace/scratch/bowser-orch5/fix-fuzz/target
# reproduce the exact input (also re-runs under gdb/asan as needed):
cargo +nightly fuzz run fuzz_x509 fuzz/artifacts/fuzz_x509/crash-<sha1>
# minimize the reproducer:
cargo +nightly fuzz tmin fuzz_x509 fuzz/artifacts/fuzz_x509/crash-<sha1>
```

Report the crashing input file, the stack trace, and which target found it;
do not fix parser bugs in the fuzz crate — fixes belong in
`crates/bowser_net/src/`, reviewed separately.

## Notes / known limitations

- `fuzz_tls_handshake` exercises the *client* handshake only (bowser has no
  server side). Reaching the encrypted flight (Certificate /
  CertificateVerify parsing) needs a peer that completes ECDHE, which random
  bytes will not do; that path is covered indirectly via `fuzz_x509`'s
  `verify_server_chain` call, which is what the handshake invokes on the
  received chain.
- The x509 `#[path]`-include trick means the shim silently diverges if the
  upstream files gain new `super::`/`crate::` dependencies; a build failure
  here is loud, a *silent* divergence is not possible (it includes the files,
  it does not copy them).
- libfuzzer flags: `-timeout=N` turns a hang into a `timeout-` artifact
  (a hang finding); `-rss_limit_mb` turns runaway allocation into an `oom-`
  artifact instead of killing the host.
