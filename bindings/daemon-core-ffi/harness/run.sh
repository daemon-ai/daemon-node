#!/usr/bin/env bash
# Build the daemon-core-ffi staticlib, compile the C harness against it, and run it.
# The phase-8 FFI gate: proves the C ABI drives StartTurn -> TurnFinished over the session surface.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
cd "$root"

profile="${1:-debug}"
if [ "$profile" = "release" ]; then
  cargo build -p daemon-core-ffi --release
else
  cargo build -p daemon-core-ffi
fi

lib="target/$profile/libdaemon_core_ffi.a"
if [ ! -f "$lib" ]; then
  echo "staticlib not found at $lib" >&2
  exit 1
fi

out="target/$profile/ffi_harness"
echo "compiling C harness -> $out"
cc "$here/harness.c" "$lib" -lpthread -ldl -lm -o "$out"

echo "running $out"
"$out"
