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
  `# Per-array element cap for the generated C structs, in lockstep with daemon-api's` \
  `# WIRE_PAGE_MAX (= 64). The paginated response arrays (fs-list-page / fs-search-page /` \
  `# log-page-view / journal-page-view / events-page / session-page) carry an explicit 0*64` \
  `# bound in the CDDL since wire v24 and the node clamps every page to WIRE_PAGE_MAX; this` \
  `# default caps the remaining (small, enumeration-shaped) arrays. Heap-allocated, so the` \
  `# larger union is fine.` \
  --default-max-qty 64 \
  --output-c "$out/${base}.c" \
  --output-h "$out/${base}.h" \
  --output-h-types "$out/${base}_types.h" \
  "$@"
