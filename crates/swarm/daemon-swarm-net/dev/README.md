# Self-hosted iroh relay — dev ops (lane B2)

The swarm control plane is **iroh gossip**, mandatory for every peer (spec §7.1). Gossip's
sub-4 KB signed messages traverse NAT via **iroh relays**; for the public swarm the relay URLs are
**pinned in the run envelope** (spec §7.4 — Psyche only ever hardcoded hostnames; we do better).
This directory ships a dev runner that stands one up locally for LAN/loopback testing.

## Run it

```
nix develop --command crates/swarm/daemon-swarm-net/dev/run-relay.sh        # http://localhost:3340
IROH_RELAY_PORT=4455 nix develop --command crates/swarm/daemon-swarm-net/dev/run-relay.sh
```

The devShell ships the `iroh-relay` 1.0 binary (flake Wave-0 lane). Point clients at the printed
relay URL by putting it in `IrohGossipConfig.relay_urls` (or the envelope's transport section):

```rust
IrohGossipConfig {
    relay_urls: vec!["http://localhost:3340".to_string()],
    // ... secret_key, roster, topic_input, rebroadcast, bind_addr
}
```

An **empty** `relay_urls` selects `RelayMode::Disabled` (direct-only, e.g. same-host loopback
tests, which need no relay).

## TLS / insecure findings (investigated honestly)

`iroh-relay` 1.0 has a first-class dev mode: **`iroh-relay --dev`** runs "in localhost development
mode over **plain HTTP**", default bind **`[::]:3340`**, and **ignores any config-file TLS fields**
(it sets the internal `dangerous_http_only`). So local dev needs **no certificates, no ACME /
LetsEncrypt** — the relay URL is a plain `http://` URL. The runner uses exactly this; for a custom
port it writes a throwaway TOML with `http_bind_addr` and still passes `--dev` (plain HTTP).

Production/WAN relays (out of scope for P1 — recorded for the P2 WAN gate) use a `[tls]` config
section: `cert_mode = "manual"` (`manual_cert_path` / `manual_key_path`) or `"letsencrypt"`
(`hostname` + `contact`), served over `https://`, optionally with `enable_quic_addr_discovery`
(which *requires* TLS). None of that is needed for the dev/loopback relay.

## Ports

| Port | Purpose | Notes |
|---|---|---|
| `3340` | dev HTTP relay (`--dev` default) | plain HTTP; the relay URL is `http://localhost:3340` |
| `<custom>` | dev HTTP relay | via `IROH_RELAY_PORT`; the runner writes an `http_bind_addr` config |
| `9090` | relay metrics | default on; the custom-port path disables it to avoid a second bind |

## How the test uses it

`tests/iroh_gossip.rs::relay_path_delivers_through_self_hosted_relay` spawns `iroh-relay --dev`
(default port 3340), builds two nodes with `relay_urls` set and a **relay-only** roster (no direct
IPs), and asserts gossip forms + delivers through the relay. It **skips cleanly** when `iroh-relay`
is not on PATH or port 3340 is busy (standalone checkout without the devShell).
