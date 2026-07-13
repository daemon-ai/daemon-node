# Swarm P2 WAN — lane ledger **A3** (worker live attach — the Merge-2 headline)

Lane **A3** of the "Swarm P2 WAN Program", Wave 2. Moves the in-process live recipe (B3's
`live_harness` — a `RoundEngine` over `IrohGossip` + a real payload store) **into the
`daemon-train-worker` subprocess**, wires a continuous worker→node event pump, boots run discovery
at the node, and lands the declared-RunConfig create-request fields (Merge-1 Decision 1, both
halves). Read the program ledger (`docs/specs/swarm-p2-ledger.md`, incl. "Merge 1 — frozen
interfaces" + the Wave-2 launch notes for A3), the P1 ledger (`docs/specs/swarm-p1-ledger.md`), and
the A1 (`swarm-ledger-p2-a1.md`) + B3 (`swarm-ledger-b3.md`) ledgers first — this ledger records
only A3's deltas and consumes the A1/B3-frozen seams verbatim.

- **Repo / branch:** `daemon-node`, `swarm/a3`, base `4e821cd` (Merge 1).
- **Worktree:** `/home/j/experiments/daemon-worktree/p2-a3`.
- **Cloud half:** `/home/j/experiments/daemon-cloud/daemon-api` on `swarm/p2-integration`
  (coordination branch @ `ef9bc8f`); its `master` is never touched, nothing pushed.
- **Files owned (daemon-node):** the worker attach seam
  (`crates/coprocessor/daemon-train/src/bin/daemon-train-worker/{transport.rs,live.rs,main.rs}`
  + `Cargo.toml` — the ledger-sanctioned `daemon-swarm-net` feature edit), the shared credentials +
  telemetry contract (`crates/swarm/daemon-swarm-run/src/protocol.rs`, additive), the event pump
  (`crates/coprocessor/daemon-train-client/src/lib.rs`,
  `crates/swarm/daemon-swarm-node/src/service.rs`), the discovery boot wiring
  (`bins/daemon/src/main.rs` + `crates/swarm/daemon-swarm-run/src/config.rs` additive `[swarm].registry`),
  the run-authoring declared fields (`bins/swarm-local`), and the live worker e2e
  (`tests/daemon-swarm-e2e/*`). `daemon-swarm-net` is touched **ADDITIVE only** (its surfaces are
  frozen). Root `Cargo.toml`/`deny.toml`/`flake.nix` untouched.
- **Files owned (daemon-cloud):** `apps/swarm/coordinator-wasm/src/lib.rs` (InitConfig additive
  fields + wasm rebuild), `apps/swarm/src/{registry.ts,coordinator/shell.ts}`,
  `packages/shared/src/swarm/types.ts` (CreateRunRequest additive), `apps/swarm/test/*`,
  `apps/swarm/scripts/seed_run.mjs`.

---

## 0. The dep-graph change (ledger-sanctioned, off the default gate)

`daemon-train` already carried `daemon-swarm-net = { workspace = true }` (for the `file://`
`ArtifactResolver` in `backend.rs`) at the **default** feature set (no iroh/ws). A3 adds a new
`swarm-net` cargo feature to `daemon-train` that turns on `daemon-swarm-net/ws` +
`daemon-swarm-net/iroh` (and nothing else), gating the live coordinator attach behind it. This is
the program-ledger-sequenced dep-graph change (P2 plan, workstream A3), kept **off the default
gate** so a default `cargo build`/`test`/`deny` never compiles the WS/TLS/iroh/QUIC tree — mirroring
`daemon-swarm-net`'s own `ws`/`iroh` gating. Verified with `cargo tree` (see gates).

## 1. Worker live attach (`daemon-train-worker`, feature `swarm-net`)

The self-driven single round (`transport::join_and_run_round`, B3's in-process representative round)
stays as the **T0 / test / default-gate fallback**. Behind `feature = "swarm-net"` the `JoinRun`
path constructs the real loop, mirroring `daemon_swarm_run::live_harness` but for ONE peer inside
the subprocess:

1. Parse `JoinRun.coordinator` (WS base) + `JoinRun.credentials` (the canonical-CBOR
   `JoinCredentials` contract, §2) → WS auth, optional iroh secret/roster/relay, optional presign
   base, the node signing key, the epoch roster, and the engine knobs.
2. Build `WsControlPlane::connect(WsConfig{ base_url, run_id, auth, reconnect })`.
3. If iroh credentials are present **and** the `iroh` feature is on: build `IrohGossip::connect(...)`
   with the node iroh secret, envelope-pinned relays, admission roster, `topic_input =
   envelope_hash`; compose `DualPlane::pair(ws, iroh)`. **WS-only mode (no iroh creds / feature off)
   runs over the bare `WsControlPlane` — the T0 baseline.**
4. Register the peer's signed `Join` frame via `WsControlPlane::add_resubscribe_frame` so a
   reconnect re-admits (A1 contract). Roster updates arriving as coordinator frames are wired to
   `IrohGossip::update_roster` (a background task decoding admission/roster frames off the plane).
5. Build the payload store: `R2Store` over `HttpPresignClient` when `credentials.presign_base` is
   set; else a per-run `FsPayloadStore` (tests / LAN).
6. Construct `RoundEngine::new(DualPlane, store, WasmBackend, key, corpus, EngineConfig, ev_tx)` and
   `run()` it continuously until `Leave`/stop/`RunOutcome`. Mirror the §7.3 receive-side size-cap
   pre-filter node-side (Merge-1 Decision 2): a `Commitment` above `update_max_bytes` is dropped
   before the engine ingests it (the engine's own path is unchanged; the cap lives in the worker's
   plane adapter, matching the DO shell's pre-filter).

`EngineEvent`s are translated to worker `protocol::Event`s and streamed continuously over the stdio
cut (§3). The engine runs in a spawned task so `main` keeps servicing `Throttle`/`Leave`/`Ping`
while rounds progress.

## 2. `JoinCredentials` contract (frozen at Merge 2) — VERBATIM

The `JoinRun.credentials: Vec<u8>` stays opaque on the frozen worker wire; A3 defines the canonical
CBOR **schema** carried in it (a new additive type in `daemon_swarm_run::protocol`, authored by the
node, parsed by the worker). No change to `Command`/`Event` shapes.

```rust
// daemon_swarm_run::protocol  (additive; canonical-CBOR body of JoinRun.credentials)
// (verbatim contract recorded in §Results after implementation)
```

## 3. Event pump (continuous worker→node stream; additive telemetry)

Before A3, `TrainSupervisor::join` resolved on the first `RunPhase` and the remaining worker events
were discarded. A3 adds `TrainSupervisor::join_streaming(...) -> UnboundedReceiver<Event>` (the
worker's continuous event stream) and a `SwarmService` pump task that feeds each event into the
existing `SwarmService::handle_worker_event` → `NodeEvent::SwarmChanged`, so `swarm.db` reflects
live round progression (phase/metric/round-outcome/warning per round). The self-driven `join` is
retained.

Additive telemetry `Event` variants (the P1-deferred follow-on 2 — the OOM-ladder + micro-batch
telemetry as protocol events instead of B3's stderr-only): recorded verbatim in §Results. Additive
to the frozen §10.2 stream (new variants only; existing round-trips unchanged), mapped through
`SwarmService::translate` onto additive `SwarmEvent`s / `Warning`s.

## 4. bins/daemon discovery boot wiring

`SwarmServiceParts.discovery` was `None` (A1 follow-on). A3 constructs a `RegistryClient`-backed
`EgressRunDiscovery` at boot from an additive `[swarm].registry` config surface (registry base +
`swarm:*` creds: bearer or internal identity), so a node with `[swarm] enabled` + a configured
registry discovers runs, fetches + blake3-verifies the frozen envelope, and runs the real §6.5
`AssessRun` before `JoinRun` (A1's `resolve_join`). Absent registry config → `discovery: None`
(unchanged probe fallback).

## 5. Declared RunConfig (Merge-1 Decision 1, both halves)

- **Cloud:** additive optional fields on `CreateRunRequest` (`warmup_timeout_s`, `round_timeout_s`,
  `cooldown_s`, `global_batch`, `witness_target`) → forwarded verbatim through `createRun` →
  `ShellConfig` → the DO `/init` body → the `coordinator-wasm` `InitConfig`, which **drops the T0
  defaults when a field is present** (else keeps today's coarse constant for back-compat). The
  registry still never parses the envelope (§11.1/§12). Rebuilds `coordinator.wasm` + provenance.
- **Node:** the run-authoring path (`bins/swarm-local` envelope runner) emits the declared fields so
  a node-authored run carries them into the create request.

## 6. Live worker-subprocess e2e (Merge-2 rehearsal)

`tests/daemon-swarm-e2e` gains an env-gated worker-subprocess e2e: a wrangler-dev DO + registry,
2–3 REAL `daemon-train-worker` subprocesses spawned via `TrainSupervisor` with the tiny-llama guest,
WS(+iroh where enabled) control plane, object-proxy R2 store, N≥5 rounds — per-round det digests
byte-identical across workers, the event pump visible in `SwarmService` state, one worker dropped
mid-run recovers per the stall ladder. Env-gated (needs wrangler-dev); the default e2e gate stays
fast + transport-free.

## Seams A3 exports (freeze at Merge 2)

1. worker live-attach config: the `JoinCredentials` contract + the `swarm-net` feature gate.
2. the event-pump protocol additions (additive telemetry `Event` variants + `join_streaming`).
3. declared-RunConfig fields (both halves).

## Gates (A3)

fmt; clippy workspace + feature combos (incl. the new `daemon-train --features swarm-net`); deny;
`cargo test --workspace`; net/node/train/e2e suites; the new live e2e executed green; wasm32 builds;
`build-guests`; typos. Cloud: `pnpm -r typecheck` (don't worsen gateway), apps/swarm tests green.
Known flake (never modify): the `daemon-conformance` detached-delegation trio — pass-in-isolation =
green.

---

## Results (finalized with evidence)

_Populated at lane completion: final HEADs, commit list, the `JoinCredentials`/event contract
verbatim, e2e evidence, dep-graph check, deviations, Merge-2/C2/B2B3 notes._
