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
node, parsed by the worker). No change to `Command`/`Event` shapes. A buffer that does not decode
as `JoinCredentials` (incl. the empty buffer every pre-A3 caller sends) selects the **self-driven
fallback** — full back-compat.

```rust
// daemon_swarm_run::protocol  (additive; canonical-CBOR body of JoinRun.credentials)

pub enum WsAuthSpec {                    // mirror of daemon_swarm_net::WsAuth (never hardcoded)
    #[default] None,
    Bearer(String),                      // gateway `swarm:join` API-key path
    Internal { org_id: String, actor: String },  // direct-to-apps/swarm dev path
}

pub struct IrohRosterPeer {
    pub endpoint_id: [u8; 32],           // iroh EndpointId (32 raw bytes)
    #[serde(default)] pub direct_addrs: Vec<String>,   // "ip:port"; empty = relay-only
    #[serde(default)] pub relay_url: Option<String>,
}

pub struct IrohCredentials {
    pub secret_key: [u8; 32],            // iroh secret — separate from the node identity (§7.2)
    #[serde(default)] pub relay_urls: Vec<String>,     // envelope-pinned
    #[serde(default)] pub roster: Vec<IrohRosterPeer>, // bootstrap; updates via update_roster
}

pub struct EngineParams {
    pub steps_per_round: u32,
    pub micro_batch: u32,                // the assess verdict clamps it at runtime
    pub stall_rounds_max: u32,
    pub checkpoint_every_rounds: u32,
    #[serde(default)] pub update_max_bytes: u64,  // §7.3 receive cap (Decision 2); 0 = uncapped
    pub corpus_seed: u64,                // Corpus::synthetic(seed, shards, tokens/shard, seq_len)
    pub corpus_shards: u32,
    pub corpus_tokens_per_shard: u64,
    pub corpus_seq_len: u32,
    #[serde(default)] pub corpus_vocab_clamp: u32, // token % clamp (B3 shim recipe); 0 = off
}

pub struct JoinCredentials {
    pub node_secret: [u8; 32],           // ed25519 seed — the RoundEngine's signer identity
    #[serde(default)] pub ws_auth: WsAuthSpec,   // drives WS + presign auth
    pub roster: Vec<[u8; 32]>,           // epoch roster (node pubkeys)
    pub envelope_hash: [u8; 32],         // §6.1 anchor: Join binding + iroh topic input
    #[serde(default)] pub iroh: Option<IrohCredentials>,  // None ⇒ WS-only (T0 baseline)
    #[serde(default)] pub presign_base: Option<String>,   // None ⇒ FsPayloadStore fallback
    pub engine: EngineParams,
}
impl JoinCredentials { pub fn to_bytes(&self) -> ...; pub fn from_bytes(&[u8]) -> ...; }
```

## 3. Event pump (continuous worker→node stream; additive telemetry)

Before A3, `TrainSupervisor::join` resolved on the first `RunPhase` and the remaining worker events
were discarded. A3 adds `TrainSupervisor::join_streaming(...) -> UnboundedReceiver<Event>` (the
worker's continuous event stream) and a `SwarmService` pump task that feeds each event into the
existing `SwarmService::handle_worker_event` → `NodeEvent::SwarmChanged`, so `swarm.db` reflects
live round progression (phase/metric/round-outcome/warning per round). The self-driven `join` is
retained.

Additive telemetry `Event` variants (the P1-deferred follow-on 2 — the OOM-ladder + micro-batch
telemetry as protocol events instead of B3's stderr-only), additive to the frozen §10.2 stream (new
variants only; every existing frame round-trips unchanged; no SwarmApi wire change — telemetry
surfaces as `SwarmEvent::Warning` classes `micro_batch` / `oom_ladder`):

```rust
// daemon_swarm_run::protocol::Event — additive variants (A3, Merge 2)
MicroBatch { micro_batch: u32 },                       // the consumed §10.5 autotune verdict
OomLadder  { round: RoundId, from_micro_batch: u32,    // one §10.5 halving rung (real
             to_micro_batch: u32, halvings: u32 },     //  BudgetMemory trap → churn + retry)
```

**The full per-round pump** (live attach): `RunPhase{train}` + `MicroBatch` at join, then per round
`Metric{loss}` per train step, `RunPhase` + `RoundOutcome{round, digest, …}` per `RoundComplete`,
`Warning{straggling|caught_up|left}` on the stall ladder, `CheckpointPublished` on §9 boundaries,
`OomLadder` when the ladder fires — through `TrainSupervisor::join_streaming` →
`SwarmService::handle_worker_event` → `swarm.db` + `NodeEvent::SwarmChanged`.

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

### Commit list

**daemon-node `swarm/a3`** (base `4e821cd` = Merge 1, oldest → newest; the ledger-finalize commit
sits on top):

| Commit | Subject |
|---|---|
| `6a5c7e5` | `mirror(A3): ledger` |
| `5b5c8d0` | `feat(swarm): JoinCredentials contract + telemetry events + worker event pump (green)` |
| `c50dfb4` | `feat(train): worker live attach — RoundEngine over DualPlane(WS,iroh) + R2/Fs store behind swarm-net feature (green)` |
| `5d9a3fb` | `feat(swarm-node): boot-wire EgressRunDiscovery from [swarm.registry] config (green)` |
| `2f14929` | `feat(swarm-local): --emit-create-request with declared RunConfig fields (Decision 1 authoring half) (green)` |
| `2fbacb6` | `test(swarm-e2e): live 4-worker subprocess e2e vs wrangler-dev DO — digests, event pump, drop-rejoin (green)` |
| `0867fc9` | `fix(e2e,deps): clippy needless-range-loop in live e2e + bump yanked spin 0.10.0->0.10.1 (deny gate)` |

**daemon-cloud `daemon-api` `swarm/p2-integration`** (base `ef9bc8f`; master untouched, nothing
pushed): `316db6e` `feat(swarm): declared RunConfig on create-run -> DO init -> wasm (Merge-1
Decision 1, A3)` — `CreateRunRequest` additive optional fields → `registry.ts` shape-validation +
verbatim `/init` forwarding → `ShellConfig`/`shell.ts` (present-only keys) → `coordinator-wasm`
`InitConfig` `#[serde(default)] Option`s replacing the T0 constants; `coordinator.wasm` rebuilt
(wrapper-only — the pure `tick` crate is byte-unchanged at trunk `4e821cd`; provenance updated:
blake3 `e2ce9f1d…`, 568 871 B); `seed_run.mjs` env-configurable (`SWARM_BASE` + declared-field
envs); root `.gitignore` gained `/target/` (the devShell pins `CARGO_TARGET_DIR` there).

### Live e2e evidence (`tests/daemon-swarm-e2e/tests/ws_live_workers.rs`) — EXECUTED GREEN twice

Both variants executed in this session against wrangler-dev (port 8795, cloud branch `316db6e`),
4 REAL `daemon-train-worker` subprocesses (debug build, `swarm-net` feature) spawned via
`TrainSupervisor`, tiny-llama guest (1 layer, vocab 64), **object-proxy R2 store** (every payload
PUT/GET a live presign → `/o` HTTP round-trip), 8 rounds, declared RunConfig (warmup 8 s /
round-timeout 20 s / cooldown 1 s / global_batch 16) driving the DO's real phase timings:

1. **WS-only (T0 baseline)** — run `run-a3-e2e-1783977839`, **122 s wall**: workers 1+2 report all
   8 rounds with **byte-identical per-round digests**; worker 0's progression lands in `swarm.db`
   via the pump (`last_round = 7`, ≥8 `RoundOutcome` events + the `micro_batch` telemetry Warning);
   worker 3 killed after round 1 → coordinator drops it at K=3 record-absences → floor breach parks
   the run in `WaitingForMembers` (epoch 1) → the supervisor's lazy respawn + re-assess + re-`Join`
   (previously-Dropped member rejoin, §6.5) resumes the run → rounds 6–7 complete, DO `/state`
   `finished: true, round: 8`. The rejoined worker contributed rounds 6–7 (its digests intentionally
   outside the byte-identity assertion — fresh-state rejoin, see Deviations).
2. **Dual-plane WS + iroh** (`SWARM_LIVE_RELAY_URL=http://127.0.0.1:3340`, self-hosted
   `iroh-relay --dev`) — run `run-a3-e2e-1783978030`, **121 s wall**: same assertions green with
   every worker running `DualPlane(WsControlPlane, IrohGossip)` (relay-only roster reachability,
   per-worker iroh identities, topic = envelope hash).

**Endpoint configurability (the daemon-swarm-dev note):** the user approved a real Cloudflare dev
deployment — an infra agent is deploying `apps/swarm` as **`daemon-swarm-dev`** (workers.dev) with a
real `swarm-dev` R2 bucket + an iroh-relay on the M1 mini. The e2e targets wrangler-dev (the Merge-2
gate) but every endpoint is env-only: `SWARM_LIVE_WS_URL` (coordinator/registry base),
`SWARM_LIVE_PRESIGN_BASE` (defaults to the WS base), `SWARM_LIVE_RELAY_URL` (iroh relay),
`SWARM_LIVE_ORG`/`SWARM_LIVE_ACTOR` — so Merge 2 can point the identical harness at the real
endpoints with **zero code change**. The node config surface mirrors this (`[swarm.registry]` base +
auth are pure config), and `seed_run.mjs` takes `SWARM_BASE`. NB the workers.dev path fronts no
gateway: it keeps the internal-identity dev headers; the gateway Bearer path is `WsAuthSpec::Bearer`
+ `RegistryAuthConfig::Bearer`, already wired end to end.

### Dep-graph check (the sanctioned change, verified)

`daemon-train` gained the `swarm-net` cargo feature = `daemon-swarm-net/{ws,iroh}` +
`dep:daemon-egress` + `dep:async-trait` (all existing workspace deps; root
`Cargo.toml`/`deny.toml`/`flake.nix` untouched). Verified: **default gate** `cargo tree -p
daemon-train -i iroh` and `-i tokio-tungstenite` → *no matching packages* (the default worker build
compiles no WS/TLS/iroh/QUIC tree); `--features swarm-net -i iroh` → present as expected.
`cargo deny check` fully green (after the `spin` 0.10.0→0.10.1 lock bump — 0.10.0 was yanked
upstream; a lock-only change, no manifest edit).

### Gate results (final HEAD `0867fc9` + this ledger; jobs capped at 16 ≤ nproc/2)

- `cargo fmt --all --check` ✓ · `cargo clippy --workspace --all-targets -- -D warnings` ✓.
- Feature-combo clippy `-D warnings`: `-p daemon-train --features swarm-net` ✓ · `burn-ndarray` ✓ ·
  `-p daemon-swarm-net --features ws` ✓ · `iroh` ✓ · `ws,iroh` ✓ · `-p daemon-swarm-run --features
  iroh` ✓ · `-p daemon-swarm-e2e --features iroh` ✓ · `-p daemon-train-sdk --features sim` ✓ ·
  `-p daemon-api --features arbitrary` ✓. (`--features wgpu` clippy not re-run this lane — no
  A3 file touches the wgpu lane; the Merge-1 result stands.)
- `cargo deny check` ✓ (advisories/bans/licenses/sources; see the `spin` note above).
- `cargo test --workspace` ✓ — **zero failures** (~6 min; the documented `daemon-conformance`
  detached-delegation flake did not fire this run).
- Per-crate suites: `-p daemon-swarm-net --features ws` ✓ (incl. the new `receive_size_cap` test) ·
  `--features iroh` ✓ · `-p daemon-train --features burn-ndarray` ✓ (worker_protocol 4 — the frozen
  self-driven stream unchanged) · `-p daemon-swarm-e2e` default ✓ (11) ·
  `--features iroh --test live_transport` **6/6** ✓.
- The new live e2e **EXECUTED GREEN** in-session, both WS-only and WS+iroh variants (above).
- Both wasm32 builds (`daemon-swarm-{proto,coordinator}`) ✓ · `cargo run -p xtask -- build-guests` ✓
  (manifest matches the committed trunk canonical — no drift) · `typos docs/specs` ✓.
- Cloud (`swarm/p2-integration` @ `316db6e`): `apps/swarm` vitest **38/38** ✓ (incl. the new
  declared-RunConfig forwarding/validation/timing tests) · `pnpm -C packages/shared typecheck` ✓ ·
  `pnpm -C apps/swarm typecheck` ✓ · `pnpm -r typecheck` fails **only** in `apps/gateway` —
  verified **pre-existing at HEAD** via stash (identical errors with A3's changes stashed; gateway
  untouched, not worsened).

### Deviations (recorded honestly)

1. **The API-initiated `swarm_join` authors no `JoinCredentials` yet.** `SwarmService::swarm_join`
   pumps the stream but passes empty credentials (the worker's self-driven fallback), because the
   node identity / roster / engine-params authoring source for an app-initiated join is a
   Merge-2/integration decision (where does a node's swarm signing key live?). The live attach is
   driven via `SwarmService::join_and_pump` (public, used by the e2e + available to the boot site).
2. **Rejoin is fresh-state, not checkpoint-resync.** The drop drill's rejoined worker re-enters with
   a fresh model state (its post-rejoin digests differ — asserted *excluded* from byte-identity).
   Wiring §9 checkpoint resync into the worker's live attach (fetch manifest → `
   resume_from_checkpoint`) is the natural B-lane/Merge-2 follow-on; the engine API already
   supports it.
3. **Worker 0's byte-digest agreement is transitive.** The §10.4 wire events are payload-free, so
   the pump path proves *progression* in `swarm.db` while byte-identity is asserted across the
   direct-stream workers; a worker-0 digest divergence would surface as the coordinator's
   `DigestMismatch`.
4. **Live `Throttle{paused}` = preemption-as-churn.** The live engine owns its backend exclusively,
   so pause stops the run (releasing the instance, §10.5) rather than in-place pausing; the node
   re-joins from durable intent. The self-driven path keeps in-place pause/resume.
5. **`daemon-swarm-net` additive edits** (allowed, surfaces frozen): `DualPlane::
   with_receive_size_cap` (Decision 2's node-side §7.3 pre-filter — drops an oversize `Commitment`
   before delivery, undecodable frames untouched) + `HttpPresignClient::with_internal` (the
   dev-path identity headers, mirroring `RegistryClient::with_internal`). Both additive builders;
   every existing constructor behavior-identical.
6. **`swarm-local --emit-create-request` emits `size: 0` artifacts** for placeholder-hash authoring
   artifacts (the registry accepts size ≥ 0); the e2e's request carries the real module size+hash.
7. **`min_peers` semantics finding (for Merge-2/C2):** joins land roster-direct ONLY in
   `WaitingForMembers`; a join during Warmup/rounds is staged `pending` and only materializes at
   `exit_cooldown` (epoch boundary) — with `epoch_rounds = 0` that is *never mid-run*. A live
   deployment where N workers join a min_peers<N run therefore races the warmup transition (an
   early e2e iteration hit this: 2 of 4 workers silently rode as non-members). Declared-RunConfig
   runs should set `min_peers` = expected initial roster. Flagged as a spec §6.2 sharp edge worth a
   note.

### What Merge-2 / C2 / B2B3 must know

- **Freeze (A3 exports):** the `JoinCredentials`/`WsAuthSpec`/`IrohCredentials`/`IrohRosterPeer`/
  `EngineParams` credentials contract (§2, verbatim above); the additive `MicroBatch`/`OomLadder`
  protocol events + `join_streaming`/`join_and_pump`/`bind_self` pump surface (§3); the declared
  RunConfig fields (both halves, §5); the `swarm-net` worker feature gate; `[swarm.registry]`
  config; the two additive `daemon-swarm-net` builders (deviation 5).
- **Merge-2 headline rehearsed:** the full node↔cloud↔worker loop is green over wrangler-dev with
  real worker subprocesses, live presign/R2 payloads, declared timings, drop + rejoin — the same
  harness re-targets `daemon-swarm-dev` + the M1 relay via env only.
- **B2B3:** the worker live attach consumes `Eligibility.headroom["micro_batch"]` (clamped into
  `EngineParams.micro_batch`) and the OOM ladder emits protocol telemetry now — B3's stderr-only
  deviation 2 is closed. The observe wiring (B2) can subscribe to the same pump.
- **C2:** the worker's `swarm-net` feature is orthogonal to the GPU lanes (`wgpu` etc.); the
  Windows worker build should keep `swarm-net` off until the MinGW cross build is validated with
  the WS/TLS tree.
- **Integration owner:** re-run `cargo deny` at merge (the lock gained webpki roots via the ws
  feature path — green here); the `spin` bump rides `Cargo.lock` only. The cloud coordination
  branch carries the rebuilt `coordinator.wasm` — `dual_shell_parity` (3 tests) green on the cloud
  side with the new wasm.
