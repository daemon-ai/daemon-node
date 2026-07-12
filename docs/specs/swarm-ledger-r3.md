# Swarm-training MVP — lane R3 ledger (local runner / churn drills / worker-backed run)

Wave-3 coordination record for lane **R3** (`swarm/r3`). Companion to the program ledger
[`swarm-mvp-ledger.md`](swarm-mvp-ledger.md) ("Merge 1" + "Merge 2" frozen sections) and the
Wave-2 runtime record [`swarm-ledger-r2.md`](swarm-ledger-r2.md). Read those first for the frozen
seams (`daemon-swarm-proto` API, `SwarmTransport`, `TrainerBackend`, worker protocol, the pure
`daemon-swarm-coordinator` `tick`, the Merge-2 `RoundEngine` / checkpoint / harness seams) and the
frozen-file rule. This file records what R3 builds on top of those seams, the new seams it exports
(frozen at Merge 3), and every `MERGE-3` marker Merge 3 must resolve.

## Base + branch

- **Branch:** `swarm/r3`, forked from `39c0ebd` (`mirror(merge-2): P0 milestone — freeze Wave-2
  interfaces`) on `integrations/swarm` — Merge 2, all Wave-2 lanes (P2/R2/E2) integrated, the stub
  e2e running on the **real** coordinator `tick` loop.
- **Merge target:** `integrations/swarm` (disjoint file set → conflict-free by construction).
- **Owns (create / edit only within):** `crates/swarm/daemon-swarm-net`,
  `crates/swarm/daemon-swarm-run`, `crates/coprocessor/daemon-train-client`,
  `tests/daemon-swarm-e2e/`, and `bins/` (the swarm runner surface).

## FROZEN — do not touch (single-writer rule)

Root `Cargo.toml`, `deny.toml`, `flake.nix`; every Merge-1 + Merge-2 frozen seam (proto API,
`SwarmTransport`, `TrainerBackend`, worker protocol, `assignment`, the pure coordinator `tick` +
`CoordinatorState`, `RoundEngine` / checkpoint / harness). Extend the frozen seams **additively
only**. Other lanes' directories (coordinator/observe/proto/det-core/train-sdk/train/guests/xtask)
are out of bounds — R3 reads the coordinator + proto crates but never edits them.

## Parallel-lane note (P3 observe, E3 worker)

Lane P3 builds `daemon-swarm-observe` (replay oracle, `DesyncVerdict`) and lane E3 the
`daemon-train` **worker binary** (the child side of the worker protocol) **in parallel**; their new
APIs are **not** available to R3. Every place that would consume them drives a local stand-in,
marked `// MERGE-3: …`, exactly like the Wave-1/2 discipline:

- the desync **trigger** is a local quorum-digest fold (the observe `DesyncVerdict` slots in later);
- worker mode spawns the fake `fake-train-worker` fixture (the E3 `daemon-train` binary slots in via
  `--worker-bin` / `DAEMON_TRAIN_WORKER_BIN`).

## Scope (this wave)

| Slice | Crate | Spec | TDD |
|---|---|---|---|
| `local_coordinator` — promote the Merge-2 `TickCoordinator` shell to a public library module (+ mid-run restart from persisted `CoordinatorState`) | `daemon-swarm-run` | §6.2, §11.2 | PROTO-20 (practical) |
| `swarm-local` runnable — N in-process peers + coordinator shell + TOML envelope load, `--backend stub\|worker` | `bins/swarm-local` | §10.4 (local run), §6.1 | (e2e support) |
| Worker-backed run path — `TrainSupervisor` drives the (fake) worker over the frozen protocol | `daemon-train-client` | §10.2, §10.5 | CLI-2/4, RUN-9/10 |
| Churn/failure drills over the local runner | `tests/daemon-swarm-e2e` | §6.4, §13 | E2E (§3.8), RUN-7/8 |
| RUN-9/10 — preemption-as-churn + assess staging over the worker protocol | `daemon-train-client` | §10.5, §6.5 | RUN-9/10 |
| `[swarm]` figment config surface | `daemon-swarm-run` | §10.6 | (config) |

## Seams R3 exports (freeze at Merge 3)

- **`local_coordinator`** (`daemon_swarm_run::local_coordinator`, `feature = "harness"`) —
  `LocalCoordinator<C>` (the impure shell around the pure `tick`: signs + publishes the
  coordinator's unsigned `RoundOpen`/`RoundRecord`, produces `StorageReceipt` availability
  evidence, and drives finalization deterministically), `LocalCoordinatorConfig`,
  `CoordinatorReplay`, and `snapshot()` / `resume_from(state)` for the mid-run restart drill.
- **`swarm-local` CLI** (`bins/swarm-local`) — the runnable local run mode (flags below).
- **drill harness helpers** (`daemon_swarm_run::harness`, `feature = "harness"`) — the `SwarmConfig`
  scenario knobs (fault / silent-death / store-outage / late-join / restart), `FaultyStore`
  (withhold-key **and** outage-window modes), and `SwarmRun` collectors.
- **`SwarmConfig` figment struct** (`daemon_swarm_run::config::SwarmConfig`) — the typed `[swarm]`
  config section (spec §10.6), serde-deserializable; node figment wiring is post-MVP.

### `swarm-local` CLI surface (frozen at Merge 3)

```
swarm-local --envelope <PATH.toml> [flags]
  --peers <N>            in-process peer count           (default 3)
  --rounds <N>           rounds to drive                 (default from envelope [data].stop)
  --seed <HEX|DEC>       corpus + coordinator seed        (default 0xDAE07E57)
  --state-dir <DIR>      payload-store root               (default: a fresh temp dir)
  --backend stub|worker  peer backend                     (default stub)
  --profile <NAME>       experiment profile passthrough   (recorded in the run header)
  --worker-bin <PATH>    worker binary for --backend worker (default: fake-train-worker)
```

`stub` mode stands up the deterministic in-process `RoundEngine`/`StubBackend` peers + the
`LocalCoordinator` and prints the agreed digest transcript. `worker` mode spawns one supervised
`daemon-train` worker per peer over the frozen worker protocol (`// MERGE-3: point at the
daemon-train worker binary`); the fake worker keeps R3's side testable end to end.

## Design decisions (not obvious from the code)

- **`local_coordinator` is the Merge-2 shell, promoted verbatim, then made public + restartable.**
  The pure `tick` still lives in `daemon-swarm-coordinator`; `LocalCoordinator` is the impure shell
  (clock, signing, receipt production over the shared `FsPayloadStore`). It stays feature-gated
  behind `harness` (the coordinator dep is `harness`-optional; keeping the default participant build
  lean). The e2e still passes on it — the module is a lift-and-rename of `TickCoordinator`, not a
  behavior change.
- **Coordinator restart uses `CoordinatorState` canonical CBOR.** `LocalCoordinator::snapshot()`
  serializes the current pure state; `resume_from(bytes)` rebuilds a fresh shell around the decoded
  state and keeps driving. Because `tick` is pure and the state round-trips byte-identically
  (PROTO-20), a kill+reload mid-run is transparent — the restart drill exercises this in anger.
- **Worker mode does not drive a `TrainerBackend` per round.** In the shipped architecture the
  `daemon-train` worker runs its **own** round loop internally (E3), so R3's worker mode is
  spawn + `AssessRun` + `JoinRun` + event stream + `Throttle`/`Leave` over `TrainSupervisor` — the
  node keeps only durable intent (§10.2). Against the fake worker this proves the supervision +
  protocol path; the real training is E3's worker behind the same protocol.
- **Late-join needs an epoch boundary.** The coordinator stages a mid-epoch `Join` as `pending` and
  applies it only at the next `WaitingForMembers` (`exit_cooldown`). The late-join drill therefore
  runs with `epoch_rounds > 0`; the late peer `checkpoint_load`s the previous epoch's checkpoint
  (new additive `EngineConfig::resume_from`) before its first round so its round base matches
  consensus, then contributes from the epoch boundary.
- **Silent death needs `min_peers` headroom.** A silently-dropped peer lowers `healthy_count`; if it
  falls below `min_peers` the coordinator floor-breaches to `Cooldown`. The death drill sets
  `min_peers < num_peers` so the run continues on the survivors.
- **Desync detection is a local quorum-digest fold (MERGE-3 stand-in).** R3 folds the per-round
  `Digest`s the peers publish, flags the minority peer, and resyncs it via the R2 checkpoint +
  record replay machinery (`resync_by_replay`). The observe `DesyncVerdict` replaces the fold.

## `MERGE-3` marker sites (search `MERGE-3` in the tree)

| Site | What Merge 3 must do |
|---|---|
| `bins/swarm-local/src/main.rs` worker spawn | point the worker backend at the real `daemon-train` worker binary (E3) |
| `daemon-train-client` RUN-9/10 tests | swap the fake worker for the real `daemon-train` worker (real wasm preemption) |
| `tests/daemon-swarm-e2e` desync drill | replace the local quorum-digest fold with `daemon-swarm-observe`'s `DesyncVerdict` |
| `daemon-swarm-run/src/checkpoint.rs` resync quorum digest (carried from R2) | same observe-driven desync trigger |

## Things Merge 3 / later waves must watch for

- **`bins/*` is a workspace-member glob** (root `Cargo.toml` `members = [… "bins/*" …]`), so
  `bins/swarm-local` is picked up with **no** root edit — it landed as its own crate (the preferred
  location), not in `daemon-swarm-run/src/bin/`. It depends on `daemon-swarm-run`'s `harness`
  feature, so a `--workspace` build compiles `daemon-swarm-run` with `harness` (hence the
  coordinator dep) enabled; that is workspace feature-unification only, harmless.
- **The `harness` feature** now gates `local_coordinator` + the drill helpers in addition to the
  peer harness. Additive; no frozen-file change.
- **The `[swarm]` config struct is defined in `daemon-swarm-run`, not the node config crate.** The
  node's main config crate (`bins/daemon/src/config.rs`) is outside lane R's file set, so the typed
  struct + its figment extraction test live here; wiring it into `NodeConfig` is post-MVP node work.
- **Additive-only extension** of the frozen seams remains the rule; the `local_coordinator` /
  runner-CLI / drill-helper / config seams above freeze at Merge 3.

## Delivered (final)

Commits on `swarm/r3` (base `39c0ebd`, oldest → newest):

| Commit | Subject |
|---|---|
| `mirror(R3)` | ledger (this file) |
| `feat(swarm-run)` | promote `local_coordinator` shell + churn-drill harness + `[swarm]` config |
| `feat(swarm-e2e)` | churn/failure drills over the local runner |
| `style(swarm-run)` | clippy `-D warnings` clean |
| `feat(train-client)` | RUN-9/10 preemption-as-churn + assess staging over the worker protocol |
| `feat(swarm-run)` | `swarm-local` local runner CLI — stub + worker backends |

- **Binary landed in `bins/swarm-local`** (the `bins/*` glob picks it up — no root `Cargo.toml` edit),
  not `daemon-swarm-run/src/bin/`.
- **Drills** (`tests/daemon-swarm-e2e/tests/drills.rs`), each asserting the run completes with all
  surviving digests equal: `late_join_mid_run_syncs_and_contributes`,
  `hard_peer_death_dropped_after_absences`, `payload_store_outage_absorbed_by_stall_ladder`,
  `desync_injection_detected_and_resynced`, `coordinator_restart_mid_run_completes`.
- **`MERGE-3` marker sites:** `daemon-swarm-run/src/harness.rs` (`quorum_digests` desync-detector
  stand-in) + `tests/daemon-swarm-e2e/tests/drills.rs` (desync drill) → observe `DesyncVerdict`;
  `bins/swarm-local/src/main.rs`, `daemon-train-client/tests/supervisor.rs`, and
  `daemon-train-client/src/bin/fake-train-worker.rs` → the real E3 `daemon-train` worker binary /
  meta-mode assess. (The R2-carried `checkpoint.rs` resync-quorum `MERGE-2` marker still stands.)
- **Gates (green):** `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `typos docs/specs`, the full swarm-stack `cargo test` (proto/coordinator/net/run/train-client/e2e),
  and `cargo test --workspace --no-run`. Test counts: `daemon-swarm-run` 32 + `record_ordering` 2,
  `daemon-swarm-e2e` 5 drills + 2 P0, `daemon-train-client` 2 unit + 4 integration (incl. RUN-9/10),
  `daemon-swarm-run` config 3.
