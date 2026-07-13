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

## Results / deviations

_(Filled in at finalize — see the "Finalize" section appended by the last `mirror(B3)` commit.)_
