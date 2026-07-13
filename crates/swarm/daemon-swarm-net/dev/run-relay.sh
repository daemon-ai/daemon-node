#!/usr/bin/env bash
# Self-hosted iroh gossip relay for LAN/loopback dev + testing (swarm P1, spec §7.4; lane B2).
#
# Wraps the devShell's `iroh-relay` (1.0.x) in its localhost DEV mode: plain HTTP, no TLS, no ACME.
# The relay carries gossip for all peers that pin its URL in the run envelope / IrohGossipConfig.
#
# Usage:
#   nix develop --command crates/swarm/daemon-swarm-net/dev/run-relay.sh          # port 3340
#   IROH_RELAY_PORT=4455 nix develop --command .../dev/run-relay.sh               # custom port
#
# Then point clients at the printed relay URL (IrohGossipConfig.relay_urls / envelope [phases]):
#   http://localhost:<port>
#
# See dev/README.md for the ops story (ports, TLS/insecure findings, production notes).
set -euo pipefail

PORT="${IROH_RELAY_PORT:-3340}"

if ! command -v iroh-relay >/dev/null 2>&1; then
  echo "error: iroh-relay not on PATH — run inside 'nix develop' (the devShell ships iroh-relay 1.0)." >&2
  exit 127
fi

echo "iroh-relay dev mode (plain HTTP, no TLS): http://localhost:${PORT}" >&2

# --dev defaults to port 3340; for any other port pass an http_bind_addr via a throwaway config
# (--dev still forces plain HTTP and ignores TLS fields). Metrics disabled to avoid a second bind.
if [[ "${PORT}" == "3340" ]]; then
  exec iroh-relay --dev
fi

cfg="$(mktemp -t iroh-relay-dev.XXXXXX.toml)"
trap 'rm -f "${cfg}"' EXIT
cat >"${cfg}" <<EOF
http_bind_addr = "0.0.0.0:${PORT}"
enable_metrics = false
EOF
exec iroh-relay --dev --config-path "${cfg}"
