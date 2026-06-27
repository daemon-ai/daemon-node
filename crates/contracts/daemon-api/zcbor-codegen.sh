#!/usr/bin/env bash
# Canonical zcbor invocation for the daemon-api client codec.
#
# This is the single source of truth for the entry types + flags. It is pure given `zcbor` on PATH
# (deterministic: a CDDL in, generated C/H out) and deliberately does NOT enter any flake shell, so
# both the pure Nix derivation (superproject `packages.daemon-zcbor-codec`) and `xtask verify-codec`
# can reuse it without duplicating the command.
#
#   zcbor-codegen.sh <cddl> <out-dir> [extra zcbor args...]
#
# Without extra args it writes only the generated codec (the shape `daemon-app` vendors into
# `src/core/daemon/codec/generated/`). Pass `--copy-sources` to also drop the zcbor C runtime
# alongside (used by the verify harness to compile a self-contained binary).
set -euo pipefail

cddl="${1:?usage: zcbor-codegen.sh <cddl> <out-dir> [extra zcbor args...]}"
out="${2:?usage: zcbor-codegen.sh <cddl> <out-dir> [extra zcbor args...]}"
shift 2

base="daemon_api_client"
mkdir -p "$out"
exec zcbor code \
  --cddl "$cddl" \
  --entry-types api-request api-response \
  --decode --encode \
  `# Per-array element cap for the generated C structs (the CDDL arrays stay unbounded for the` \
  `# Rust/cddl-cat conformance side). 64 covers a repo's GGUF/quant file list + a search page +` \
  `# the installed catalog + more log entries per Subscribe page; bump here if a real response` \
  `# is ever truncated. Heap-allocated, so the larger union is fine.` \
  --default-max-qty 64 \
  --output-c "$out/${base}.c" \
  --output-h "$out/${base}.h" \
  --output-h-types "$out/${base}_types.h" \
  "$@"
