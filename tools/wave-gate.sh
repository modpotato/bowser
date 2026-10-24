#!/usr/bin/env bash
# Wave gate: the checks every wave must pass before its work lands.
#
# Wave 42 lesson: the "gate green" claim was debug-only. `cargo test`
# passed in debug while 141 scroll/window tests failed in release
# (a `HashMap::insert` housed inside `debug_assert!`, compiled away in
# release). Release was never gated. It is now.
#
# Usage: ./tools/wave-gate.sh [--quick]
#   --quick: skip the release suites (still runs the debug_assert lint,
#            fmt, clippy, and debug tests). For mid-wave checkpoints.
# Full gate (no flag): everything, including release tests.
#
# Build coordination: takes flock on the shared cargo-build lock like
# every other build on this box (3-concurrent-build cap, 2 vCPU).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"

# wgpu backend tests fail loudly without a Vulkan adapter, and this box has
# no GPU. Point the loader at the bundled SwiftShader software ICD when
# nothing else is configured (a box restart wipes /usr-installed drivers,
# which silently turned 64 wgpu_be tests red).
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /opt/meta-chromium/vk_swiftshader_icd.json ]]; then
  export VK_ICD_FILENAMES=/opt/meta-chromium/vk_swiftshader_icd.json
fi

QUICK=0
if [[ "${1:-}" == "--quick" ]]; then QUICK=1; fi

# Per-wave scratch target dir (never the repo's ./target; avoids the
# 512MiB /tmp tmpfs too).
export CARGO_TARGET_DIR="$HOME/workspace/scratch/wave-gate/target"
mkdir -p "$CARGO_TARGET_DIR"

LOCK="$HOME/workspace/build-targets/.cargo-build.lock"
run() { flock -n "$LOCK" -c "$*"; }

echo "=== 1/5 debug_assert side-effect lint ==="
python3 tools/check-debug-assert-side-effects.py

echo "=== 2/5 fmt ==="
run "cargo fmt --check --all"

echo "=== 3/5 clippy ==="
run "cargo clippy --workspace --all-targets -- -D warnings"

echo "=== 4/5 debug tests ==="
run "cargo test --workspace"

if [[ $QUICK -eq 1 ]]; then
  echo "=== 5/5 release tests SKIPPED (--quick) ==="
  exit 0
fi

echo "=== 5/5 release tests (the wave-42 gate) ==="
# bowser_app first: this is where the 141 release-only failures lived.
run "cargo test --release -p bowser_app"
# bowser_js: the VM had the same bug class (stack pop in debug_assert_eq!).
run "cargo test --release -p bowser_js"
# The rest of the workspace on a cadence: full release suite.
run "cargo test --release --workspace"

echo "WAVE GATE GREEN"
