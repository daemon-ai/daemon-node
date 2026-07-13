# Swarm P1 + Transport — lane ledger **B3** (live worker transport + e2e on real planes)

Lane **B3** of the "Swarm P1 + Transport" program, Wave 3 (the final lane wave) — the **transport
exit-gate lane**. Read `swarm-p1-ledger.md` (program ledger; "Merge 2" + "Wave-3 must know") and the
predecessor lane ledgers first: `swarm-ledger-b1.md` (R2Store / PresignClient / scheduler / fetch),
`swarm-ledger-b2.md` (IrohGossip surface + wiring recipe + dev relay runner), `swarm-ledger-w1.md`
(SwarmService / WorkerControl / swarm.db + the boot-wiring diff), `swarm-ledger-g2.md`
(AutotuneVerdict / `probe_microbatch` / GPU-skip). This ledger records only B3's deltas.

## Base + branch

- **Repo / worktree:** `daemon-node` @ `/home/j/experiments/daemon-worktree/swarm-runtime`.
- **Base commit:** `2f1ce1f` (`mirror(merge-2): freeze Wave-2 interfaces`) on `integrations/swarm-p1`
  (Merge 2 HEAD).
- **Branch:** `swarm/b3`. Merges back into `integrations/swarm-p1` at Merge 3.

## Scope (program plan "B3 — live worker transport + e2e on real planes")

1. **Ledger** (this file), committed first.
2. **Engine-on-live-transport wiring** — make the frozen `RoundEngine` run over `IrohGossip` + a real
   `PayloadStore` end to end: peer boots → Join (with `iroh_id`) → admission roster → IrohGossip mesh
   (`update_roster` on admission/drop) → rounds progress with control over gossip + payloads over the
   store, downloads pipelined, ingest barrier preserved. Reuse the stub-transport e2e as the template
   — the transports are swappable behind the frozen traits, so the deliverable is the **WIRING +
   missing glue**, not new protocol machinery.
3. **`with_swarm` boot binding** — bind the ready-but-unbound `with_swarm` seam (W1) in the node
   assembly so a `daemon-node` process can host a swarm worker via `SwarmService` (off by default,
   config-gated). Keep the daemon default-gate dep-free (no burn/wasmtime/iroh on the default build).
4. **Live e2e** in `daemon-swarm-e2e` (the transport exit gate, TDD §3.8) — multi-peer (≥3 incl. one
   late-join + one mid-run drop), real `IrohGossip` on loopback (explicit `NodeAddr`s; a relay variant
   via B2's dev runner, skip-clean if the binary is absent), real payload store (filesystem-backed
   `FsPayloadStore` shared across peers = a shared object store), tiny-llama guest, N≥10 rounds:
   per-round det digests byte-identical across peers, stall-ladder recovery on the dropped peer,
   late-joiner resync, run termination on the envelope stop condition, replay verification green.
5. **Worker lifecycle glue** — live OOM trial via `probe_microbatch` wired where the worker drives
   real batches (the seam G2 left mechanical); micro-batch from `Eligibility.headroom["micro_batch"]`
   consumed in-process; governor throttle/drop hooks (§10.5) exercised in ≥1 e2e drill (on unified
   boxes the governor clamps the combined budget, Merge-2 spec-amendment #1).

## Frozen seams B3 wires together (consumed, never reshaped)

- **`IrohGossip` (B2).** `IrohGossip::connect(IrohGossipConfig{ secret_key, relay_urls, roster,
  topic_input, rebroadcast, bind_addr })` → `ControlPlane`; `node_id()` (the `EndpointId` for
  `Join.iroh_id`), `local_peer()` (dialable self), `neighbor_count()`, `update_roster(Vec<IrohPeer>)`
  (re-seeds discovery + `join_peers`, cap ~3), `shutdown()`. **Wiring recipe (B2 ledger finding 7):**
  connect with the node iroh secret key, envelope-pinned `relay_urls`, admission roster
  (`IrohPeer.endpoint_id = Join.iroh_id`), `topic_input = FrozenEnvelope::hash()`; `node_id()` fills
  the local `Join.iroh_id`; `update_roster` on every admission/drop; the plane carries **already-signed**
  proto `SignedMessage` bytes (sign before `publish`, `verify()` after `subscribe().recv()`; the plane
  never verifies). Self-delivery + content-hash dedupe make delivery byte-identical to `LoopbackGossip`
  (B2 finding 3), which the parametric conformance suite proved — so `RoundEngine` sees the same
  contract on either plane.
- **`R2Store<P>` / `PayloadStore` / `fetch_with_fallback_dyn` / `DownloadScheduler` / `fetch_record_set`
  / `ArtifactCache` (B1).** The payload plane + typed `PayloadMiss` stall-ladder feed. `fetch_record_set`
  is store-generic and B3 wires it engine-side (below).
- **`SwarmService` / `WorkerControl` / `swarm.db` / `with_swarm` (W1).** The boot-wiring diff (W1
  ledger "Boot-wiring integration step") is applied by this lane.
- **`AutotuneVerdict` / `probe_microbatch` / `Eligibility.headroom["micro_batch"]` (G2).** The ladder +
  taxonomy are frozen; B3 wires the *live* catch-OOM-mid-step trial.
- **`RoundEngine` / `LocalCoordinator` / `tick` (Wave-2 merges).** Generic over `C: ControlPlane`,
  `P: PayloadStore`, `B: TrainerBackend` — the whole point of the frozen seam design is that swapping
  `LoopbackGossip → IrohGossip` and `FsPayloadStore → R2Store` is a *construction* change, not a
  protocol change.

## Ownership (this wave)

- **Own (additive):** `crates/swarm/daemon-swarm-run/src/{engine.rs, live_harness.rs}` (additive),
  `bins/swarm-local`, `tests/daemon-swarm-e2e/*`, the worker `transport` module
  (`daemon-train/src/bin/daemon-train-worker/transport.rs` — assigned to B3 by the Wave-0 split +
  W1/G2 ownership notes), and the daemon-swarm-run `Cargo.toml` (additive feature/dev-deps).
- **Cross-lane edit (per brief item 3, the W1 integration remainder):** the `with_swarm` boot binding
  spans `crates/node/daemon-node/{src/assembly, src/lib.rs, Cargo.toml}` + `bins/daemon/src/main.rs`.
  W1 designed + froze the seam (`NodeApiImpl::with_swarm` exists in `daemon-host`; W1 ledger carries the
  diff) but left it unbound at boot "best applied by the integration owner alongside B3". B3 applies it.
  Documented here as the sanctioned cross-lane action; it is inert unless `[swarm] enabled = true`.
- **Read-only:** `/home/j/experiments/daemon` (main checkout), FROZEN files (root `Cargo.toml`,
  `deny.toml`, `flake.nix`), `daemon-train` tests/benches + `guests/*` (M2's this wave), daemon-cloud
  (BC's).

## Design decisions

### Live transport = per-node IrohGossip mesh (not a shared Arc)

`LoopbackGossip` is one shared instance every participant clones. `IrohGossip` is the opposite: each
peer **and** the coordinator own a **distinct** endpoint, and they form a real QUIC gossip mesh on the
shared topic `blake3(envelope_hash)`. The live harness therefore:

1. Async-constructs N peer nodes + 1 coordinator node with an **empty** roster (bind `127.0.0.1:0`,
   `RelayMode::Disabled` for the loopback variant, or the dev-relay URL for the relay variant).
2. Collects each node's `local_peer()` (endpoint id + bound sockets) once all are bound.
3. Calls `update_roster(full_roster)` on every node → seeds `MemoryLookup` + `join_peers` → mesh forms
   (B2 finding 1: the direct loopback mesh forms with no relay in ~1 s).
4. Waits for the mesh to reach `neighbor_count() >= 1` on each node before opening round 0 (so the first
   `RoundOpen` is not lost to a not-yet-formed mesh — gossip is best-effort, and B2's rebroadcast frame
   covers residual gaps).

`RoundEngine` and `LocalCoordinator` are already generic over `C: ControlPlane`, so the drive loop is
identical to the loopback harness — only the plane construction + roster wiring differ. This is exactly
the "transports are swappable behind the frozen traits" property the brief calls out.

### Payload plane = shared `FsPayloadStore` (a real object store)

The brief sanctions "filesystem-backed `PayloadStore` or miniflare R2". A single `FsPayloadStore`
directory shared by every peer **is** the real `PayloadStore` trait impl and models a shared object
store (every peer PUTs its own object, GETs peers' objects, hash-verified) — the same shape R2 has,
without a live network dependency or a BC block. The `FaultyStore` wrapper (harness) injects the
stall-ladder / outage faults over it, unchanged. (An `R2Store`-over-`MockR2` variant is available via
B1's wiremock harness but is not the exit-gate default — it adds no transport coverage the FS store
lacks and would couple the gate to wiremock; recorded as a follow-on when BC's wrangler-dev lands.)

### `fetch_record_set` wired engine-side (inline still preferred)

`verify_record_set` (engine.rs) was inline-only with a `// MERGE-2` marker for the `record-set.cbor`
fetch. B3 adds `resolve_record_set(&rr)`: prefer `rr.inline` (small rosters — the exit-gate default),
else fetch `record-set.cbor` via the store using `rr.set_locator` + B1's `fetch_record_set`, then
root-verify. Additive; the inline path is byte-for-byte unchanged.

### Concurrent in-peer fetch (the deferred `// MERGE-2` marker)

The MVP engine prefetches reactively inside the sequential message loop. With a real plane, B3 adds a
bounded concurrent prefetch: as `Commitment`s arrive, payload GETs are dispatched onto a
`DownloadScheduler`-gated task set writing into a shared fetched-cache, so peers' payloads download in
parallel and the barrier usually finds the set already local. The engine's `&mut backend` sequential
apply is preserved (ingest ordering / the barrier I2 is untouched) — only the *fetch* overlaps.

## Exported seams (freeze at Merge 3)

1. **The live-transport e2e harness API** — `daemon_swarm_run::live_harness`: `run_live_swarm(cfg:
   LiveSwarmConfig) -> SwarmRun` + the drill knobs (peers, rounds, late-join, drop, stall, outage,
   relay-url). Behind the `iroh` feature (which enables `daemon-swarm-net/iroh` + `harness`).
2. **The `with_swarm` config surface** — `[swarm]` (`SwarmConfig`, already frozen at Merge 1) is now
   *bound at boot*: `enabled=true` + `worker_path` (+ `data_dir/swarm.db`) makes a `daemon-node` host a
   `SwarmService`. The boot binding shape (what `NodeAssembly` carries, when the service is constructed)
   is the frozen surface.
3. **The worker lifecycle glue points** — where the worker `transport` module constructs the
   `RoundEngine`/round loop and where it consumes `Eligibility.headroom["micro_batch"]` + wraps
   real batches with `probe_microbatch` + honors the `Throttle` governor lever.

## Planned slices (each `feat(...)/test(...): … (green)`; lane gates green per commit)

1. `mirror(B3): ledger` (this file).
2. `feat(swarm-run): fetch_record_set + concurrent prefetch engine glue (green)`.
3. `feat(swarm-run): live IrohGossip harness behind the iroh feature (green)`.
4. `feat(swarm-run): swarm-local runner (loopback|iroh × fs) (green)`.
5. `test(swarm-e2e): live-transport exit gate over iroh + fs (green)`.
6. `feat(swarm-node): bind with_swarm at boot, config-gated (green)`.
7. `feat(train): worker round loop over live transport + OOM/micro-batch/governor glue (green)`.
8. `mirror(B3): finalize ledger` (evidence, test counts, deviations, Merge-3 notes).

## Gates (B3)

`cargo fmt --check` · `cargo clippy --workspace --all-targets -- -D warnings` · clippy
`-p daemon-swarm-net --features iroh` + relevant feature combos · `cargo deny check` ·
`cargo test --workspace` · `cargo test -p daemon-swarm-net --features iroh` · the new live e2e suite
(the full live run executed once, wall time recorded; the default-gate variant fast) · both wasm32
builds · `build-guests` · `typos docs/specs`. Known pre-existing flake (never modified): the
`daemon-conformance` detached-delegation trio — pass-in-isolation = green.

## Results — finalize (evidence, counts, deviations, Merge-3 notes)

### Commit list (base `2f1ce1f`, oldest → newest)

| Commit | Subject |
|---|---|
| `6021d29` | `mirror(B3): ledger` |
| `a3996ed` | `feat(swarm-run): fetch_record_set + concurrent barrier fetch engine glue (green)` |
| `b2bfe69` | `feat(swarm-run): live IrohGossip harness behind the iroh feature (green)` |
| `2c9da89` | `test(swarm-e2e): live-transport exit gate over iroh + fs (green)` |
| `bf6c489` | `feat(swarm-run): swarm-local runner (loopback\|iroh transport, fs store) (green)` |
| `46fb013` | `feat(swarm-node): bind SwarmService at boot via with_swarm/set_swarm, config-gated (green)` |
| `27dbff7` | `feat(train): worker OOM ladder + micro-batch verdict consumption + governor drill (green)` |
| `a47609f` | `test(swarm-e2e): tiny-llama wasm flagship over the live iroh mesh (green)` |
| (fixup) | `feat(swarm-node): pin daemon-swarm-node dep version for cargo-deny wildcard gate (green)` |
| (this) | `mirror(B3): finalize ledger` |

### Live e2e evidence (the transport exit gate, TDD §3.8)

`cargo test -p daemon-swarm-e2e --features iroh --test live_transport` — **6 tests, all green,
13.6 s test wall time (~18 s incl. incremental build)**, executed on this machine this session.
Every test = real per-node `IrohGossip` endpoints (QUIC gossip, loopback binds, explicit
roster/`MemoryLookup` addressing, no public discovery) + real `daemon-swarm-coordinator` `tick`
(via `LocalCoordinator` over its own iroh node) + shared `FsPayloadStore`:

1. **`live_flagship_three_peers_ten_rounds_all_agree`** — 3 peers (StubBackend) × 12 rounds,
   concurrent barrier fetch ON: per-round digests byte-identical across all 3 peers every round;
   run terminates on the envelope stop condition (`Rounds(12)`); **PROTO-20 replay from the
   live-transport log green** (12 recorded rounds, byte-identical tick trajectory).
2. **`live_flagship_tiny_llama_wasm_over_iroh`** — 3 peers running the **real tiny-llama guest**
   (`WasmBackend`, wasmtime host training, 1-layer/vocab-64 config over the synthetic corpus with a
   deterministic vocab-clamp shim) × **10 rounds**: det digests byte-identical across peers every
   round, transcript evolves (learning), replay green.
3. **`live_stall_ladder_recovers_over_iroh`** (RUN-8 live) — an injected round-5 payload miss on the
   real mesh: `Straggling{round:5}` → `CaughtUp{round:5}` → all 3 peers report round 5, no leave —
   ENGINE-level recovery over the plane B2 proved partition/rejoin on.
4. **`live_late_join_resyncs_over_iroh`** — a 4th peer admitted at the epoch-1 boundary over the
   live mesh, resyncs from the round-2 checkpoint (fetched from the store), contributes rounds 3–5
   with consensus-equal digests.
5. **`live_mid_run_drop_dropped_after_absences`** — a peer goes silent after round 2; the
   coordinator drops it after K=2 record-absences; 2 survivors complete round 7 in agreement.
6. **`live_run_through_self_hosted_relay`** — spawns B2's `iroh-relay --dev` (plain HTTP :3340),
   nodes constructed `RelayMode::Custom(<relay>)` with relay-bearing rosters; 6 rounds all-agree.
   **Ran green (not skipped)** — the devShell relay binary is on PATH. Skips cleanly when absent.

The suite is **feature-gated** (`daemon-swarm-e2e/iroh`), so the default e2e gate stays fast +
iroh-free (10 tests: 5 drills + 2 stub e2e + 3 wasm profiles, ~23 s); the live gate is opt-in and
was executed this session with the results above. `swarm-local --transport iroh --peers 3
--rounds 8` also ran green end to end (agreed transcript printed, exit 0).

### `with_swarm` config surface (frozen at Merge 3)

- Config: the Merge-1-frozen `[swarm]` table (`daemon_swarm_run::config::SwarmConfig`), already
  embedded in `NodeConfig` (W1). **Now bound at boot**: `[swarm] enabled = true` (default false) +
  `worker_path` make `bins/daemon` construct `SwarmStore::open(<data_dir>/swarm.db)` + a
  `TrainSupervisor(worker_path)` + `SwarmService`, call `start()` (durable-intent re-convergence,
  §10.3), and bind it via the **post-`Arc`** `NodeApiImpl::set_swarm`.
- daemon-host seam change (additive): `NodeApiImpl.swarm` became a **write-once `OnceLock`** with a
  new `set_swarm(&self, Arc<dyn SwarmApi>)` post-`Arc` binder (mirroring the `set_gateway` /
  `register_managed` managed-backend precedent — the service is built after the node exists). The
  Merge-1 `with_swarm(self, …)` builder form is preserved on top of the cell. A new additive
  `NodeApiImpl::emit_node_event(NodeEvent)` hook routes the service's `SwarmChanged` invalidation
  pointers onto the existing `events_subscribe` feed (a `Weak` node handle in the `NodeFeed`
  closure avoids an Arc cycle).
- **Default-gate dep check** (like Merge 1): `cargo tree -p daemon -i {burn, wasmtime, iroh,
  iroh-gossip, cubecl}` → **all empty**. The daemon default build stays free of the training/
  transport trees; iroh remains behind `daemon-swarm-net/iroh` (off), and the worker stays a
  separate binary.

### Worker lifecycle glue points (frozen at Merge 3)

- **Micro-batch verdict consumption**: the worker caches `Eligibility.headroom["micro_batch"]` from
  `AssessRun` (G2's autotune rides it there) and threads it into `JoinRun` — it seeds the driven
  shape + the OOM ladder start. Logged to stderr (NOT a new protocol event — the frozen §10.2
  `Event` stream is pinned by M2's `worker_protocol` suite; see Deviations).
- **Live OOM ladder** (§10.5, the seam G2 left mechanical): the worker round runs inside a halving
  loop — a real `TrapCode::BudgetMemory` trap (`WasmBackendError::Train(TrainError::Trap)`, the
  wasmtime memory-trap mapping in `runtime.rs`) triggers instance churn (fresh build releases
  memory) + retry at `mb/2` until fit or floor 1; `autotune::oom_error_class()` names the class.
- **Governor drill**: `governor_throttle_lever_reaches_worker_with_combined_budget_clamp`
  (daemon-swarm-node) — a synthetic inference-pressure policy (`vram_cap_mb 4096`, duty 25%) pushed
  through `swarm_set_policy` reaches the worker `throttle` verbatim; on unified boxes that cap
  clamps the *combined* budget (Merge-2 spec-amendment #1). Worker-side `Throttle{paused}` →
  `pause`/`resume` (preemption-as-churn) was already proven by `worker_protocol`.

### Gate results (final HEAD)

- `cargo fmt --check` ✓ · `cargo clippy --workspace --all-targets -- -D warnings` ✓ ·
  clippy `-p daemon-swarm-net --features iroh` ✓ · `-p daemon-swarm-run --features iroh` ✓ ·
  `-p daemon-swarm-e2e --features iroh` ✓ · `-p daemon-train --features burn-ndarray` ✓ ·
  `cargo deny check` ✓ (advisories/bans/licenses/sources — after pinning the `daemon-swarm-node`
  dep version; a bare path dep is a cargo-deny "wildcard" on the publishable `daemon` crate).
- `cargo test --workspace` ✓ **except** the documented pre-existing `daemon-conformance`
  detached-delegation trio flake (this session's runs: `injected_input_reaches_a_parked_durable_
  session_via_the_store_seam`, then `detached_fanout_materializes_distinct_children` +
  `detached_notice_reaches_a_parked_durable_parent` under the parallel run — **all three verified
  pass-in-isolation = green**; never modified; no B3 file touches `daemon-conformance`).
- `cargo test -p daemon-swarm-net --features iroh` ✓ (82 = 71 lib + 4 conformance + 7 iroh
  integration incl. the relay-path test) · both wasm32 builds (`daemon-swarm-{proto,coordinator}`)
  ✓ · `cargo run -p xtask -- build-guests` ✓ · `typos docs/specs` ✓.
- Wall times: live e2e suite 13.6 s (test) / ~18 s (wall); full workspace test run ~190 s.

### Test counts (B3 net-new: 16)

- `daemon-swarm-run` lib: **36** (+2: `resolve_record_set_fetches_non_inline_object`,
  `resolve_record_set_rejects_object_not_matching_signed_root`).
- `daemon-swarm-e2e --features iroh`: **+6** live-transport tests (list above); default gate
  unchanged (10).
- `daemon-swarm-node` `service` tests: **7** (+1 governor drill).
- `daemon-train` `worker_protocol`: 4 (unchanged surface; now exercising the micro-batch
  consumption + ladder-wrapped round).
- Plus the `swarm-local` bin (manual/e2e-runnable, not a test target) — **7 executable slices**
  total across the lane.

### Deviations (recorded honestly)

1. **Worker in-subprocess live attach deferred; the live loop is proven in-process.** The brief's
   "worker attach: construct a `RoundEngine` over `IrohGossip` + `R2Store` inside the worker" is
   delivered as the `live_harness` wiring (the identical `RoundEngine`-over-`IrohGossip`+store
   construction, proven by the 6-test live gate) rather than inside the `daemon-train-worker`
   subprocess. Wiring it into the subprocess needs (a) the iroh/QUIC tree added to `daemon-train`
   (a feature + dep-graph decision the integration owner should make deliberately — today
   `daemon-train` is wasmtime+burn only), and (b) the `JoinRun.credentials`/coordinator-discovery
   plumbing that lands with BC's endpoint. The worker's `JoinRun` self-driven round gained the real
   lifecycle glue (verdict consumption + OOM ladder) so the remaining attach is construction, not
   design. **Merge-3 owner: this is the one brief item carried as a follow-on.**
2. **No new protocol events for the micro-batch/OOM telemetry.** M2's `worker_protocol` (read-only
   for B3 this wave) pins the exact `JoinRun` event stream (`RunPhase → Metric{loss} →
   RoundOutcome`), so the verdict/ladder telemetry goes to stderr instead of new `Metric`/`Warning`
   frames. Adding them is a trivial additive change to coordinate with M2 at Merge 3 if wanted.
3. **Payload store = shared `FsPayloadStore`, not miniflare R2** (sanctioned by the brief: "do NOT
   block on BC"). The `R2Store`-over-wrangler-dev variant slots behind the same trait when BC's
   presign endpoint lands; `swarm-local --store` reserves the flag (`fs` only today, `r2` rejected
   with a pointer).
4. **`daemon-swarm-node` is a direct path dep of `bins/daemon`** (`{ version = "0", path = … }`),
   NOT `[workspace.dependencies]` — the root `Cargo.toml` is frozen. **Merge-3 owner: promote it to
   the workspace table** (one line) like `daemon-train-safetensors` was at Merge 2.
5. **`RECORD_SET_PEER` reserved key.** The non-inline record-set fetch stores/fetches
   `record-set.cbor` under a reserved payload-plane peer id (`0x5E…`, mirroring `CHECKPOINT_PEER`)
   rather than parsing `set_locator`'s string key — store-agnostic and consistent with the
   checkpoint convention; the locator string stays authoritative for R2 object naming (B1's
   `r2_object_key` emits the §11.3 key server-side).
6. **daemon-host `swarm` field became a `OnceLock`** (a *shape* change to an unfrozen private field;
   the frozen `with_swarm` builder API is preserved verbatim). Recorded because Merge-1 froze the
   seam's *surface* — the surface is unchanged, extended additively by `set_swarm` + `emit_node_event`.
7. **Live-harness backends need `Send + Sync`**; the `Send`-only `WasmBackend` rides in an
   uncontended `Mutex` adapter in the e2e (the engine owns its backend exclusively). Documented on
   `run_live_swarm_with`.

### What Merge-3 must know

- Freeze: `LiveSwarmConfig` / `run_live_swarm{,_with}` (the live e2e harness API), the boot-binding
  shape (`set_swarm` + `emit_node_event` + the `bins/daemon` block), the worker glue points above.
- Apply: the workspace-dep promotion (deviation 4); W1's `just swarm-dev` superproject diff now has
  a real `swarm-local` to call (`cargo run -p daemon-swarm-run --features iroh --bin swarm-local --
  --transport iroh`).
- Carry: deviation 1 (worker in-subprocess attach) + deviation 2 (telemetry events) as the B3
  follow-on items; the `--store r2` swap when BC lands.
- The 5 loopback churn drills remain green untouched; the live gate adds the §3.7-on-live-transport
  variants (stall/late-join/drop) — gossip partition/rejoin stays plane-level-proven (B2), with the
  engine-level recovery now proven here.
