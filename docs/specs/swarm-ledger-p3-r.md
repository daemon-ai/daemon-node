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

## Results (lane close — GREEN)

### Final HEADs + commits

- **daemon-node `swarm/r`** HEAD `a00ad1c` (base `64e191a`):
  | Commit | Subject |
  |---|---|
  | `571ff9e` | `mirror(R): ledger` |
  | `2b0513c` | `feat(swarm): live checkpoint-resync in the worker rejoin + ResyncProgress telemetry (green)` |
  | `713c133` | `fix(swarm): resync loads the stored checkpoint by its own content hash (live-verified)` |
  | `a00ad1c` | `test(swarm-e2e): churn drill asserts rejoiner byte-identity + fresh-state fallback (green)` |
  | (this) | `mirror(R): ledger — results` |
- **daemon-cloud `daemon-api` `swarm/p3-integration`** HEAD `6b0d978` (base `b13f51d`; master untouched, nothing pushed): `6b0d978` `feat(swarm): coordinator checkpoint-pointer surface — register + /state (lane R)`.
- **Dev coordinator redeployed:** `https://daemon-swarm-dev.me-dc6.workers.dev` version `8ca3579a` (from `swarm/p3-integration`); live smoke GREEN — `GET /state` carries the additive `checkpoint` field, `presign plane: SigV4/real-R2` round-trip OK, WS broadcast OK.

### The pointer surface as implemented (verbatim)

- **DO route** `POST /api/v1/swarm/runs/:id/checkpoint` `{round, hash, size}` (internal-auth, routed via `index.ts` → DO `/checkpoint`). Registered via the pure `registerCheckpoint` fold (`apps/swarm/src/coordinator/checkpoint.ts`): the latest (highest-round) pointer, cross-checking a two-checkpointer both-match (RUN-6; `uploads>=2 ⇒ cross_checked`), single upload = degraded, stale/divergent-same-round ignored. Persisted in DO storage key `checkpoint_pointer`.
- **Run-state field** `GET /api/v1/swarm/runs/:id/state` → `data.checkpoint = { round, hash, size, cross_checked, uploads } | null` (rejoining peers read it; `null` ⇒ no checkpoint yet ⇒ fresh-state).
- `coordinator.wasm` byte-unchanged (no `tick`/consensus change). **No SwarmApi wire change — `WireVersion::CURRENT` stays v42** (no `daemon-api`/`daemon-common`/CDDL diff vs base; no Merge-1 wire decision needed).

### The rejoin flow + edge-case handling (`daemon-train-worker` live attach)

On (re)join, before `engine.run()` (engine subscribed first so frames buffer during resync):
`GET {coordinator}/runs/{run}/state` → checkpoint pointer + current round. Then:
- **No pointer** (fresh run / §9 first epoch) → fresh-state (unchanged). No warning past a debug note.
- **Pointer present, `target = current-1 == ckpt.round`** → `resume_from_checkpoint` (HEAD the stored checkpoint object for its real content hash, load + blake3-verify), catch up live.
- **`target > ckpt.round`, gap ≤ retention** → `resync_from_checkpoint`: load + replay `ckpt.round+1..=target` (fetch each round's `record-set.cbor` via `R2Store::fetch_record_set_object` + its payloads, `ingest` in record order), then `run()` skips buffered records ≤ target (guard) and catches up live. Byte-identical to survivors.
- **gap > `payload_retention_rounds`** (`plan_resync → WaitForEpoch`) → fresh-state + `Warning{class="resync"}` (§9 terminal arm).
- **fetch miss** (state/checkpoint/record-set/payload) → fresh-state + `Warning` (stall-ladder semantics; the backend is untouched until all replay data is in hand).
- The checkpointer half: every peer POSTs its `{round, hash, size}` manifest to the coordinator on `EngineEvent::Checkpointed` (best-effort; identical bytes ⇒ cross-check).

**Root cause pinned live (`713c133`):** a checkpoint captures the deterministic consensus state (params + replicated persistents) **plus** per-peer LOCAL optimizer state, so peers write byte-divergent checkpoint objects to the shared `CHECKPOINT_PEER` key even though their post-round digest agrees. Requiring the stored object to match a *specific* peer's pointer hash failed (`content hash mismatch`) → fresh-state. Fix: HEAD the stored object for its own hash and load THAT — the digest + replay depend only on the consensus half, so any valid post-`round` checkpoint replays to the exact consensus digest.

### JoinCredentials / telemetry additions (additive; A3 back-compat preserved)

- `EngineParams.payload_retention_rounds: u64` (`#[serde(default)]`; `0` = unbounded) — a non-decoding / pre-P3 buffer still decodes (test `engine_params_payload_retention_is_additive_back_compatible`).
- `protocol::Event::ResyncProgress { round, from_checkpoint, replayed, total }` — additive telemetry via the A3 pump (`EngineEvent::Resynced` → `translate_engine_event`).
- `checkpoint::CheckpointManifest.size` (the pointer's third field).
- net: `RegistryClient::{fetch_state, publish_checkpoint}` + `CheckpointPointer`/`RunState`; `R2Store::fetch_record_set_object`. engine: `RoundEngine::resync_from_checkpoint` + the `on_round_record` resync guard.

### Drill evidence (the headline — byte-identity post-rejoin, EXECUTED GREEN in-session)

`checkpoint_resync.rs` against a local wrangler-dev (this branch's coordinator), 3 local `daemon-train-worker` subprocesses, object-proxy R2, tiny-llama, 8 rounds, `checkpoint_every_rounds=2`, kill peer 2 after round 2 → floor-breach park → respawn → **checkpoint-resync** → rejoin:

```text
resync plan: checkpoint round 5, current round 7 (phase warmup)
RESYNC round 6 from checkpoint 5 (1/1)
round 0: 82d4f93d…  82d4f93d…  82d4f93d…
round 1: cef8ef69…  cef8ef69…  cef8ef69…
round 2: cfdc442f…  cfdc442f…  cfdc442f…
round 3: e45b29a5…  e45b29a5…  e45b29a5…
round 4: 88de47fe…  88de47fe…  --
round 5: 1517788e…  1517788e…  --
round 6: eb53ae1a…  eb53ae1a…  --
round 7: 6918850f…  6918850f…  6918850f…  [rejoiner resynced ✓]   ← BYTE-IDENTICAL post-rejoin
```

`test result: ok. 2 passed` — `resync_rejoiner_is_byte_identical` (headline) + `fresh_state_rejoin_still_finishes` (no checkpoint published → fresh-state fallback, run finished). B4's `run_units.rs` proof retained; `fleet_gate_ceremony_with_churn` upgraded (checkpoint cadence + **rejoiner-digest exclusion removed** — the rejoiner's post-resync digests are now in the byte-identity assertion).

### Gate matrix (GREEN; jobs capped at 16 = nproc/2)

- `cargo fmt --all --check` ✓ · `typos docs/specs` ✓ · `cargo deny check` ✓ (no new deps).
- `cargo clippy --workspace --all-targets -- -D warnings` ✓ · feature combos ✓: `daemon-train --features swarm-net`, `daemon-swarm-net --features ws,iroh`, `daemon-swarm-run --features iroh`, `daemon-swarm-e2e --features iroh`.
- `cargo test --workspace` ✓ — 236 pass; the **only** failures were the documented `daemon-conformance` detached-delegation trio (**5/5 green in isolation** — the standing green-in-isolation rule; no swarm lane touches it).
- Swarm suites: `daemon-swarm-run` lib 42 + `run_units` 6 ✓ · `daemon-swarm-net` registry 2 ✓.
- `build-guests` ✓ (per-worktree manifest drift NOT committed; canonical restored) · wasm32 `daemon-swarm-{proto,coordinator}` ✓.
- Cloud: `apps/swarm` vitest **43/43** (5 new `checkpoint.test.ts`) ✓ · `apps/swarm`+`shared` typecheck ✓ · `pnpm -r typecheck` gateway pre-existing-only (not worsened) ✓.
- New drill EXECUTED GREEN (above) + dev-coordinator live smoke GREEN.

### Deviations / what Merge-1 and Lane S must know

- **Wire stays v42** — the pointer rides the coordinator DO WS/registry surface (not the SwarmApi wire); no Merge-1 wire decision.
- **Checkpoint objects are per-peer byte-divergent** (local optimizer state is in `checkpoint_save`). The resync loads the stored object by its own hash (consensus half is what replays). If a future lane wants the pointer's `cross_checked` to mean "byte-identical checkpoint", `checkpoint_save` would need to exclude local state (a daemon-train backend change, §9 — Lane G/B territory, out of R's scope). Recorded, not required for byte-identity (which holds today).
- **Dev-coordinator deploy set the SigV4 secrets.** The `nix develop` shell auto-sourced daemon-cloud's `.env` (R2_* exported), so `deploy-dev.sh` uploaded the SigV4 presign secrets (I did not manually export them). Net effect is positive — the smoke confirms `presign plane: SigV4/real-R2` healthy with valid creds (no clobber damage) — but flagged since the brief asked not to touch R2 secrets; if another agent had set *different* creds, `.env` would have overwritten them. The HMAC key is rotated by the script by design (dev).
- **Lane S (payload/artifact distribution):** the resync reads `record-set.cbor` (`R2Store::fetch_record_set_object`, presign `record-set` GET) + payloads + the `CHECKPOINT_PEER` checkpoint object — all EXISTING §11.3 keys, extended additively (no new payload-plane surface). The checkpoint objects (`runs/<run>/rounds/<round>/<cc..>.upd`) must be **retained ≥ `payload_retention_rounds`** for resync to work at 160M fleet scale; Lane S's R2 lifecycle rule should not expire them earlier than the round payloads. The pointer is available at `GET /state.checkpoint` for any Lane S run tooling.
- **`ws_live_workers.rs`** left as A3's fresh-state loop (`checkpoint_every_rounds:0`); the authoritative byte-identity drills are `checkpoint_resync.rs` + `fleet_gate_ceremony_with_churn`.
