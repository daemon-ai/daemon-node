#!/usr/bin/env bash
# Build the daemon-ffi staticlib, compile the C harness against it, and run it.
# The durable-host FFI gate: proves the C ABI boots a node and drives StartTurn -> TurnFinished
# over the generic `daemon_host_call`.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
cd "$root"

profile="${1:-debug}"
if [ "$profile" = "release" ]; then
  cargo build -p daemon-ffi --release
else
  cargo build -p daemon-ffi
fi

lib="target/$profile/libdaemon_ffi.a"
if [ ! -f "$lib" ]; then
  echo "staticlib not found at $lib" >&2
  exit 1
fi

out="target/$profile/ffi_host_harness"
echo "compiling C harness -> $out"
cc "$here/harness.c" "$lib" -lpthread -ldl -lm -o "$out"

echo "running $out"
"$out"
