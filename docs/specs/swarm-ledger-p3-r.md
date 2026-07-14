# Swarm P3 Follow-ons — lane ledger **R** (live checkpoint-resync + DO checkpoint pointer)

Lane **R** of the **Swarm P2 Follow-ons Program (P3)**. Converts B4's proven-in-process
`resume_from_checkpoint` / `resync_by_replay` machinery into the **LIVE** worker rejoin path, so a
peer that drops and respawns mid-run rejoins with state **byte-identical to the survivors** — and
upgrades the churn drill to assert that identity (removing B4's documented rejoiner-digest
exclusion). Adds the small cloud half: the coordinator DO tracks the latest published checkpoint
pointer and exposes it in the run-state endpoint + to rejoining peers.

Read first: `swarm-ledger-p3.md`/the program plan; `swarm-p2-ledger.md` (Merge-2 JoinCredentials
contract, the churn drill's fresh-state limitation, Merge-3 carried follow-on #2); `swarm-ledger-p2-b4.md`
(the design note + `run_units.rs::worker_rejoin_via_checkpoint_reaches_consensus_fresh_state_does_not`);
`swarm-ledger-p2-a3.md` (§2 JoinCredentials, §3 event pump); spec §9 (checkpoint/resync).

- **Repo / branch (node):** `daemon-node`, `swarm/r`, base `64e191a` (integrations/swarm-p3 trunk).
- **Worktree:** `/home/j/experiments/daemon-worktree/p3-r`.
- **Cloud half:** `/home/j/experiments/daemon-cloud/daemon-api` on `swarm/p3-integration`
  (coordination branch @ `b13f51d`); `master` never touched, nothing pushed.

## Frozen surfaces this lane respects

`tabi@1` FROZEN; **wire v42** FROZEN — the checkpoint pointer rides the **coordinator DO
WS/registry surface** (a cloud HTTP route + `/state` field), **NOT** the node↔app SwarmApi wire, so
`WireVersion::CURRENT` stays **v42** (no Merge-1 wire decision needed). `JoinCredentials` extended
**additively only** (`#[serde(default)]` field), so a non-decoding / pre-P3 buffer keeps A3's
back-compat self-driven fallback. No consensus `SwarmMessage` variant added (the wasm `tick` /
`coordinator.wasm` is byte-unchanged; the pointer is DO-shell metadata, not consensus state).

## The honest surface (why this shape)

Spec §9: *"the coordinator registers the checkpoint only when both uploaded hashes match"* — the
coordinator is **designed** to receive checkpoint-manifest uploads from the checkpointer peers and
register on both-match (RUN-6 `register_checkpoint`). So the honest surface is:

1. **Checkpointer → coordinator (publish):** on each `EngineEvent::Checkpointed` the worker POSTs
   its `{round, hash, size}` manifest to the DO (`POST /runs/:id/checkpoint`, internal-auth, like
   `/msg`). This is plane-independent (works over the object-proxy plane AND a future SigV4 direct
   plane, where the coordinator would not otherwise *see* the object PUT), and gives the exact
   blake3 the rejoiner needs to verify the checkpoint on load. Every peer checkpoints identical
   bytes (deterministic, §5.6), so uploads cross-check trivially → the pointer is `Registered`
   (cross-checked) as soon as ≥2 peers report, `Degraded` on a single upload (RUN-6 semantics).
2. **Coordinator tracks latest:** the DO keeps the highest-round registered pointer in DO storage
   (separate from the opaque wasm snapshot).
3. **Expose (a) run-state endpoint:** `GET /runs/:id/state` → `data.checkpoint = {round, hash,
   size, cross_checked} | null`. **(b) to rejoining peers:** the rejoiner GETs `/state` (the same
   queryable surface the ceremony harness already polls) and reads `data.checkpoint`.

## The rejoin flow (node, `daemon-train-worker` live attach)

In `join_and_run_live`, BEFORE `engine.run()` (order matters — the engine subscribes at
construction so frames published during resync buffer and are caught up):

1. Connect WS(+iroh), register the resubscribe `Join`, build store/corpus/backend, construct
   `RoundEngine` (subscription starts buffering).
2. GET `{coordinator}/runs/{run}/state` (egress, same auth as WS) → `checkpoint` pointer + current
   `round` + `phase`.
3. **Decide (spec §9 + `plan_resync`):**
   - **No checkpoint yet** (fresh run / nothing published) → fresh-state, current behavior. `run()`.
   - **Checkpoint present**, gap `= target - ckpt.round` within `payload_retention_rounds` →
     `plan_resync` = `ReplayFromCheckpoint`: `resume_from_checkpoint(manifest)` (fetch + blake3-verify
     the checkpoint via the payload store, `checkpoint_load`), then replay retained rounds
     `ckpt.round+1..=target` — fetch each round's committed set (`record-set.cbor`) + payloads from
     the payload store and `ingest` in record order (`resync_by_replay` fold) — reaching the exact
     post-`target` consensus base. Set `last_ingested = target`; the engine then drops any buffered
     record `<= target` (new resync guard) and catches up `target+1..` live.
   - **Gap exceeds retention** (`WaitForEpoch`) → fall back to fresh-state + `Warning`
     (`class="resync"`), rejoin at the next epoch checkpoint (§9 terminal arm).
   - **Fetch failure** (checkpoint/record-set/payload miss) → fresh-state fallback + `Warning`
     (stall-ladder semantics: a miss never hard-fails the rejoin).
4. Progress is surfaced through the A3 event pump as the additive **`Event::ResyncProgress { round,
   from_checkpoint, replayed, total }`** telemetry (→ `SwarmService::handle_worker_event`).

`target = current_round - 1` (the newest finalized round at query time). The checkpoint captures
post-`ckpt.round` state; every peer PUTs identical checkpoint bytes to the same
`CHECKPOINT_PEER` key, so the checkpoint survives even if the elected checkpointer churned.

## Additive changes (freeze at Merge-1)

### proto/run (`daemon-swarm-run`)
- `protocol::Event::ResyncProgress { round, from_checkpoint, replayed, total }` — additive variant
  (telemetry; every existing frame round-trips unchanged).
- `protocol::EngineParams.payload_retention_rounds: u64` (`#[serde(default)]`) — §9 resync-replay
  window; `0` ⇒ unbounded (replay whatever is retained).
- `checkpoint::CheckpointManifest.size: u64` — the checkpoint byte length (the pointer's third
  field); `save_checkpoint` sets it.
- `engine::RoundEngine::resync_from_checkpoint(&manifest, steps)` — additive: `checkpoint_load` +
  fold `ingest` over the replay steps, updating `last_ingested`, emitting `Checkpointed`-adjacent
  progress; composes with the frozen `resume_from_checkpoint`.
- `engine`: `on_round_record` drops a record `<= last_ingested` (resync-composability guard; a
  no-op on the normal monotonic path).

### net (`daemon-swarm-net`, additive — surfaces frozen)
- `R2Store::fetch_record_set_object(round) -> RecordSet` — presigns a `record-set` GET, fetches +
  decodes the committed-set object the DO wrote (`runs/<run>/rounds/<round>/record-set.cbor`), for
  the resync replay-step assembly.

### cloud (`daemon-api`, `swarm/p3-integration`)
- DO route `POST /checkpoint` + `GET /state.checkpoint` (+ registry run-detail passthrough);
  `RunCoordinatorDO` tracks the latest registered pointer in DO storage. `coordinator.wasm`
  byte-unchanged (no `tick` change). Vitest coverage; `pnpm -r typecheck` (gateway pre-existing
  failures tolerated). Dev coordinator redeployed (keep-in-sync rule) + live smoke.

## Drill upgrade (exit criterion)

- `fleet_live_hetero.rs`: `credentials_for` now sets `checkpoint_every_rounds` (so a checkpoint
  fires before the kill) + `payload_retention_rounds`; the ceremony's **rejoiner-digest exclusion
  is removed** — the rejoiner's post-rejoin per-round digests are asserted byte-identical to
  survivors.
- New focused e2e `checkpoint_resync.rs` (wrangler-dev, 3–4 local workers, checkpoint→kill→respawn→
  resync→byte-identity), plus a fresh-state-fallback assertion (no checkpoint published → run still
  finishes).
- Deterministic proof retained (B4's `run_units.rs`); a new `resync` unit pins the replay-step
  assembly + the `plan_resync` edges the live path takes.

## Commit slices (each green per the gates)
1. `mirror(R): ledger` — this file.
2. cloud: `feat(swarm): DO checkpoint pointer surface (register + state) + vitest` (cloud branch).
3. node: `feat(swarm): live checkpoint-resync in the worker rejoin + ResyncProgress telemetry (green)`.
4. node: `test(swarm-e2e): churn drill asserts rejoiner byte-identity + fresh-state fallback (green)`.
5. `mirror(R): ledger — results` (final, after the full gates).

## Results

_(filled at lane close — HEADs, commits, drill evidence, gate matrix.)_
