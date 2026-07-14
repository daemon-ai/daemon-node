# Swarm P2 — Lane B4 ledger (remaining verification suites + spec/TDD reconciliation)

Lane **B4** of the **Swarm P2 WAN Program**, Wave 3 (the final wave before the P2 WAN gate).
Branch `swarm/b4`, base `fe27b9c` (trunk `integrations/swarm-p2` Merge-2 HEAD), worktree
`/home/j/experiments/daemon-worktree/p2-b4`. This lane closes the **remaining TDD gap register**:
it audits `swarm-training-tdd.md` against what actually landed on the trunk (B1's ~43 suites +
B2B3's observe/replay + RUN-10 + the e2e/live suites), produces the **authoritative coverage map**
below, implements the verified-remaining P2-relevant suites, wires the stronger checkpoint-resync
churn drill, and reconciles the spec + TDD docs against implemented reality.

Program ledger: `swarm-p2-ledger.md`. TDD: `swarm-training-tdd.md`. Spec: `swarm-training-spec.md`.
Sibling lane ledgers: `swarm-ledger-p2-b1.md`, `swarm-ledger-p2-b2b3.md`, `swarm-ledger-p2-a3.md`.

`tabi@1` (66 ops) stays **FROZEN**; wire **v42** stays frozen. This lane is **test-side +
docs-only** plus additive test-support: it adds test suites and doc reconciliations, and touches no
frozen file, no wire type, no `tabi` op. It does **not** modify `p2-c3`'s areas (CI/flake/runbook/
fleet/daemon-telemetry).

## Owned areas (disjoint by construction)

`crates/swarm/daemon-swarm-coordinator/tests/*` (new `pending_join.rs`; extended `commit_rule.rs`),
`crates/swarm/daemon-swarm-run/tests/run_units.rs` (extended: checkpoint-resync proof),
`docs/specs/*` (this ledger; surgical spec + TDD reconciliations). **Not touched:** any frozen file
(root `Cargo.toml`, `deny.toml`, `flake.nix`), any wire/`tabi`/proto type, the worker bin / transport
(A3), `daemon-swarm-net`/`-node`/`-observe`/`-run` **src** (only the run crate's *tests* are touched),
daemon-cloud, and everything C3 owns this wave.

---

## The audit — how P2 test debt actually stands on the trunk

**Headline: the P2 gate register is ~95% closed on the Merge-2 trunk.** B1 landed the golden /
determinism / protocol / checkpoint / supervision suites; B2B3 landed the observe-replay oracle and
RUN-10; A1/A3 landed the transport + live-loop suites; the P1 stack already carried the ABI /
sandbox / kernel suites. This lane's job was to (a) **verify, not trust** the "expected gaps" list,
(b) close the genuine residue, and (c) make the coverage auditable. The genuine residue was small
and consensus-adjacent: the §6.2 pending-join sharp edge, small-n quorum in the *commit rule*, and
the stronger checkpoint-resync churn drill. Everything else on the "expected gaps" list was already
covered (verified below) or is a deliberate P4/P5 deferral.

## Authoritative coverage map (TDD ID → status → suite)

Status legend: **green** = suite exists + passes on the B4 trunk; **green (B4)** = added/extended
this lane; **deferred** = out of the P2 gate scope with rationale. Suites cited by file.

### `daemon-swarm-proto` — consensus core (§3.1)

| TDD ID | Status | Suite / evidence |
|---|---|---|
| DIRECT ports (committee/assignment/shuffle/LCG/merkle) | green | `assignment.rs` units, `assignment_golden.rs`, `merkle.rs` |
| PROTO-1/2/3/14 (pure tick, timeouts, ring, halted) | green | `tests/tick_lifecycle.rs` |
| PROTO-4 witness quorum incl. small-n specials (1→1,2→2,3→2) | green | `assignment.rs::witness_quorum_small_n_specials` |
| PROTO-5/6 commit rule + witness-quorum evidence | green | `tests/commit_rule.rs` |
| **PROTO-5/6 small-n commit-rule quorum edges (n=1,2,3)** | **green (B4)** | `tests/commit_rule.rs::{small_n_attestation_quorum_covers_at_the_special_case_boundary, single_peer_round_finalizes_on_self_evidence}` |
| PROTO-7 heartbeat / K-absences / straggle window | green | `tests/proto_units.rs`, `tests/tick_lifecycle.rs::proto7_*` |
| PROTO-8 weighted assignment + overlap + class ladder | green | `assignment_golden.rs`, `tests/proto8_units.rs` |
| PROTO-9 global-batch ramp / stop / epoch | green | `tests/tick_lifecycle.rs::proto9_*`, `assignment.rs` |
| PROTO-10 checkpointer election (single + n) | green | `tests/proto_units.rs`, `commit_rule.rs::proto10_*` |
| PROTO-11 envelope canonical-CBOR freeze / hash chain | green | `tests/envelope_conformance.rs` |
| PROTO-12/13 capability subset + version gate | green | `tests/capability.rs`, `tests/admission.rs` |
| PROTO-15 verifier no-op at 0% | green | `tests/commit_rule.rs::proto15_*` |
| PROTO-16 no-float / wasm determinism of tick | green | `wasm32` build gate (COORD-3) + `tests/determinism.rs` |
| PROTO-17 epoch-advance disjuncts (hivemind port) | green | `tests/commit_rule.rs::proto17_*` |
| PROTO-18 xxh3-128 sampled round digest | green | `digest.rs`, `tests/host7_digest.rs` |
| PROTO-19 round-message CDDL round-trip + sig reject + size invariance | green | `tests/cddl_conformance.rs`, `tests/cbor_canonical.rs`, `tests/record_set.rs` |
| PROTO-20 replayability oracle | green | observe replay (below) + `drills.rs::coordinator_restart_mid_run_completes` |
| **§6.2 pending-join sharp edge (adjudication (d))** | **green (B4)** | `tests/pending_join.rs` (4 tests) |

### `daemon-swarm-net` — transport (§3.2)

| TDD ID | Status | Suite / evidence |
|---|---|---|
| NET-1 presign PUT/GET round-trip + expiry + signed HEAD receipt | green | `r2_store.rs::{store_presign_roundtrip, store_presign_expired_rejected}`, `presign.rs` |
| NET-2 blake3 artifact verify + tamper | green | `artifact.rs::{verify_artifact_ok, verify_artifact_tamper}` |
| NET-3 scheme resolution (`hf://@rev`, `r2://`) | green | `artifact.rs::{resolve_hf_pinned_ok, r2_to_presign}` |
| NET-4 per-object plane fallback (cost order) | green | `fetch.rs::primary_miss_falls_back_to_second_store` |
| NET-5 `SwarmTransport`/`ControlPlane` conformance (loopback/iroh/ws) | green | `tests/control_plane_conformance.rs` |
| NET-6 signed gossip accept/reject + WS↔gossip dedupe | green | `gossip.rs`, `dual_plane.rs`, `tests/iroh_gossip.rs::ws_gossip_duplicate_message_dedupes` |
| NET-8 retention floor (fetchable / typed miss) | green | `r2_store.rs::{retained_object_fetchable, expired_object_typed_miss}`, `store.rs` |
| NET-7 P2P per-param blob tickets | deferred (P4) | iroh-blobs payload plane is a P4 deliverable (plan "Explicitly deferred") |

### `daemon-swarm-run` — participant runtime (§3.3)

| TDD ID | Status | Suite / evidence |
|---|---|---|
| RUN-1 `update_mb_max` receive-side cap before decode | green | net `receive_size_cap` (`DualPlane::with_receive_size_cap`, Merge-2) |
| RUN-2 commitment→set verify against record root | green | `engine.rs::{resolve_record_set_*, verify_record_set_*}`, `tests/record_ordering.rs` |
| RUN-3 manifest/BatchId mapping + interval slicing | green | `data.rs` units, `tests/tinystories_fixture.rs` |
| RUN-4 artifact LRU cache by `data_cache_gb` | green | `artifact.rs::{artifact_cache_lru_evicts, artifact_cache_from_gb}` |
| RUN-5 round lifecycle + barrier invariant I2 | green | `engine.rs::ingest_barrier_orders_next_round` |
| RUN-6 two-checkpointer both-match + degraded + fp32 roundtrip | green | `tests/run_units.rs`, `checkpoint.rs` |
| RUN-6 `join_prefers_p2p_then_hub` | deferred (P4) | needs the P2P blob plane (NET-7, P4); R2/Hub ordering is the P2 baseline |
| RUN-7 desync → replay-resync + retention decision | green | `tests/run_units.rs`, `checkpoint.rs`, `drills.rs::desync_injection_detected_and_resynced` |
| RUN-8 stall ladder (skip/keep-fetching/catch-up/leave) | green | `engine.rs::{stalled_peer_catches_up_within_budget, stall_budget_exhausted_leaves}`, `drills.rs::payload_store_outage_absorbed_by_stall_ladder` |
| RUN-9 preemption-as-churn | green | `daemon-train-client/tests/supervisor.rs` (CLI-4) |
| RUN-10 staged assess (prescreen + cadence + slow-coordinator) | green | `assess.rs` units, `demo_module_ineligible_on_slow_coordinator` (B2B3) |

### `daemon-train-sdk` (§3.4) / `daemon-train` (§3.5) / `daemon-train-client` (§3.6)

| TDD ID | Status | Suite / evidence |
|---|---|---|
| SDK-1..5 `sparse_loco` + demo/diloco goldens | green | `tests/sparse_loco_golden.rs`, `tests/profiles.rs`, `tests/accumulation.rs` |
| HOST-1/2/5/6 kernels + det-lane | green | `det-core/tests/host_kernels.rs` |
| HOST-3/4 pack / 8-bit opt-state | green (3) / deferred P4 (4) | `absmax`/pack goldens landed; 8-bit opt-state is a P4 SDK option |
| HOST-7 round digest cross-peer | green | `daemon-swarm-proto/tests/host7_digest.rs` |
| HOST-8 meta shape + MetaReport schema | green | `guest_lifecycle.rs::meta_report_layout_and_schema`, `meta.rs::meta_report_cbor_roundtrips` |
| HOST-9 ABI autodiff parity + accumulation invariance | green | `burn_backend_parity.rs`, `burn_wgpu_parity.rs`, `accumulation.rs` |
| HOST-10 sandbox budgets + per-peer ingest scaling | green | `guest_lifecycle.rs::budget_exhaustion_traps_typed`, `wgpu_lifecycle.rs::ingest_budget_scales_with_count` |
| HOST-11 llama numerics | green | `host11_golden.rs`, `tiny_llama.rs` |
| HOST-12 ABI fuzz / trap taxonomy | green | `guest_lifecycle.rs::{phase_violation_traps_typed, budget_exhaustion_traps_typed}`, `abi_surface.rs` |
| HOST-13 canonical CBOR + DAUP framing | green | `daemon-swarm-proto/tests/cbor_canonical.rs` (RFC 8949 vectors, adversarial key order, NaN, nested), `guest_lifecycle.rs::full_round_shape_runs` (`update_bytes` round-trip) |
| HOST-14 T3 re-instantiation replay | green | `guest_lifecycle.rs::reinstantiate_rebuilds_identical_state` |
| HOST-15 mode blindness + manifest purity | green | `guest_lifecycle.rs::manifest_is_pure_no_host_imports` |
| CLI-1..4 worker protocol + supervision + throttle | green | `worker_protocol.rs`, `daemon-train-client/tests/supervisor.rs` |

### Cross-cutting (§3.7–3.9)

| TDD ID | Status | Suite / evidence |
|---|---|---|
| COORD-1 dual-shell parity | green | daemon-cloud `apps/swarm` `coordinator-parity.test.ts` (Merge-2 vitest 38/38) |
| COORD-2 DO ordering / alarms | green | daemon-cloud DO tests + live loop (`ws_live_workers.rs`) |
| COORD-3 wasm32 tick smoke | green | `wasm32-unknown-unknown` build gate (`daemon-swarm-{proto,coordinator}`) |
| WIRE-1/2/3 SwarmApi CDDL + arbitrary + codec | green | `daemon-api` `protocol_conformance` + `contract_wire_version_is_v42` |
| WIRE-4 app eligibility render | deferred (P3) | WIRE-4 app view-model is the P3 program (plan "Explicitly deferred") |
| observe projection ports (§3.9) | green | `daemon-swarm-observe/tests/observe.rs`, `swarm_e2e.rs::observe_record_and_replay_green`, `live_transport.rs::live_observe_record_and_replay_green` |
| e2e churn/failure drills | green | `drills.rs` (6 drills), `live_transport.rs` (7 live), `ws_live_workers.rs` (cloud-DO loop) |

### Verified "expected gaps" that were **already covered** (audit result)

The Wave-3 charter listed candidate gaps "verify, don't trust". Audit outcome:

- **recovery-ladder edge cases (stall→catch-up→park→resync)** — covered: `drills.rs` (5 pre-existing
  + 1 B4), `engine.rs` stall tests, `live_transport.rs::live_stall_ladder_recovers_over_iroh`,
  coordinator floor-breach→`WaitingForMembers` park (`tick.rs`, pinned by
  `tick_lifecycle.rs::proto7_k_absences_drops_and_proto10_rejoin` + `ws_live_workers.rs`).
- **witness/attestation quorum edges (small-n §A.3)** — the pure `witness_quorum` specials were
  covered; the **commit-rule** behaviour at n=1/2/3 was the residue → **added (B4)**.
- **fuel-budget scaling at max vs min peers (S2)** — covered: `wgpu_lifecycle.rs::ingest_budget_scales_with_count`
  (tier-2 GPU) + the CPU meta two-point ingest-per-peer fit (`guest_lifecycle.rs::meta_report_layout_and_schema`).
- **T3 re-instantiation replay** — covered (`reinstantiate_rebuilds_identical_state`, CPU/per-PR).
- **DAUP container fuzz/negative** — canonical-CBOR corners covered (`cbor_canonical.rs`); the DAUP
  payload is opaque bytes the swarm never parses (§7.3) and is defended at the transport boundary by
  blake3 content-addressing (`verify_artifact_tamper`, NET-2). An explicit guest-decode-truncation
  negative is **deferred** (rationale below) — no consensus exposure.
- **canonical-CBOR encoder conformance corners** — covered thoroughly (`cbor_canonical.rs`).
- **MetaReport schema validation** — covered (`meta_report_layout_and_schema` + `meta_report_cbor_roundtrips`).
- **dropout RNG pinning** — **N/A**: the llama preset / tiny-llama model has **no dropout** (grep of
  the whole tree: zero `dropout` occurrences). Recorded so the decision is auditable; if a future
  preset adds dropout, its RNG pinning joins the det-lane contract (§5.6).
- **§6.2 pending-join sharp edge** — genuine gap → **added (B4)**.
- **checkpoint publish/resync round-trip (§9, incl. A3 fresh-vs-checkpoint gap)** — round-trip
  covered (`run_units.rs`, `checkpoint.rs`); the fresh-state-vs-checkpoint-resync rejoin gap →
  **test added (B4, `run_units.rs`)** + live-worker wiring **design note** (below).

---

## New / extended suites this lane (names + what they pin)

1. **`daemon-swarm-coordinator/tests/pending_join.rs` (NEW, 4 tests)** — the §6.2 pending-join sharp
   edge (Merge-2 adjudication (d)):
   - `join_in_waiting_for_members_is_roster_direct` — initial-roster joins land directly (nothing
     staged) while `WaitingForMembers`.
   - `join_after_warmup_transition_is_staged_pending` — a mid-run join is `Admitted` but staged
     `pending`, not a live member; the healthy roster is frozen for the epoch.
   - `pending_join_materializes_at_epoch_boundary` — with `epoch_rounds=2`, the pending join drains
     into the roster only after Cooldown→`WaitingForMembers` (epoch++).
   - `pending_join_never_materializes_mid_run_when_epoch_rounds_zero` — **the sharp edge**: with
     `epoch_rounds=0` the join is stranded pending for the whole run; hence declared-run authors MUST
     set `min_peers` = expected initial roster (the `ws_live_workers` gate harness encodes exactly
     this: `min_peers == NUM_WORKERS`).

2. **`daemon-swarm-coordinator/tests/commit_rule.rs` (EXTENDED, +2 tests)** — small-n commit-rule
   quorum edges (§A.3), in the *commit rule* (not just the pure `witness_quorum` helper):
   - `small_n_attestation_quorum_covers_at_the_special_case_boundary` — for n=1/2/3 witnesses, a
     payload is `has_evidence` exactly at the special-case quorum (1→1, 2→2, 3→2), not one short.
   - `single_peer_round_finalizes_on_self_evidence` — a 1-peer round is `all_evidenced` on the sole
     peer's self-attestation (no deadlock at the smallest churn floor a gate epoch can shrink to).

3. **`daemon-swarm-run/tests/run_units.rs` (EXTENDED, +1 test)** — checkpoint-resync proof (§9,
   Task-4): `worker_rejoin_via_checkpoint_reaches_consensus_fresh_state_does_not` proves, over the
   deterministic `StubBackend` machinery, that a **fresh-state** rejoin (rebuild + ingest current
   round) reaches a digest that does **not** match consensus (the missed history is absent from the
   outer-step base, §5.6), while a **checkpoint-resync** rejoin (`resync_by_replay` — the fold behind
   `RoundEngine::resume_from_checkpoint`, guarded by `plan_resync`) recovers the **exact** consensus
   digest (§9 I1). This is the deterministic, CI-safe proof underlying the live `ws_live_workers`
   fresh-state rejoin.

All new suites are deterministic + CI-safe (in-process `StubBackend`, no live-network deps). The
checkpoint-resync proof was deliberately authored as a **synchronous** `run_units.rs` unit (not a new
`drills.rs` tokio drill): a 6th heavy multi-thread drill measurably tipped the pre-existing
`late_join_mid_run_syncs_and_contributes` timing drill into intermittent 20 s-timeout flakiness under
the drills binary's parallel co-scheduling. Keeping the proof synchronous restores the `drills.rs`
suite to its Merge-2 shape (5 drills, verified stable) while still pinning the property.

## Checkpoint-resync outcome (Task 4)

- **Implemented (deterministic test), within engine/run crates:** the resync fold
  (`checkpoint::resync_by_replay`) and the engine resume entry (`RoundEngine::resume_from_checkpoint`,
  already frozen at Merge-2) are the machinery; the new `run_units.rs` test pins that they recover the
  consensus digest where a fresh rebuild cannot. The `late_join_mid_run_syncs_and_contributes` drill
  already
  exercises `resume_from_checkpoint` through the **live in-process engine** at an epoch boundary — so
  the engine-level rejoin-via-checkpoint path is proven end-to-end in the deterministic harness.
- **Design note — wiring real resync into the LIVE cloud-DO worker rejoin (cross-crate → NOT done here):**
  `ws_live_workers.rs` rejoins fresh-state because the respawned worker calls
  `TrainSupervisor::join_streaming` / `SwarmService::join_and_pump`, which build the `RoundEngine`
  and call `run()` **without** a `resume_from_checkpoint`. To make the gate's churn drill assert
  byte-identity across the rejoin (not merely "the run finishes"), a rejoining worker must, before
  `run()`: (1) learn the latest **registered** checkpoint manifest for the run (the coordinator/DO
  already elects checkpointers and could surface the latest `CheckpointManifest` in `/state` or a
  `RoundRecord` field — additive), (2) fetch + `resume_from_checkpoint(manifest)` (engine API exists),
  (3) `plan_resync` any post-checkpoint retained rounds and replay them (machinery exists). This
  touches the **worker bin** (`daemon-train` transport/live glue, A3's area) and **`SwarmService`**
  (`daemon-swarm-node`, A3/node) — outside B4's test-side/run-crate scope — plus a small additive
  cloud surface (the latest-checkpoint pointer). **Owner: A-lane / Merge-3 gate prep** (pairs with
  Merge-2 adjudication (e), the app-surface join-credential authoring). Contained entirely to
  additive fields + the existing engine resume API; no wire/`tabi` break.

## Spec/TDD reconciliation (Task 3) — surgical, factual, no renumbering

- **`swarm-training-spec.md` §6.2 — applied Merge-2 adjudication (d).** Added an *Operational note
  (declared-run authoring)* bullet: a join is roster-direct only in `WaitingForMembers`; a join after
  the `WaitingForMembers→Warmup` transition is staged `pending` and materializes only at the next
  epoch boundary (never mid-run when `epoch_rounds=0`); authors MUST set `min_peers` = expected
  initial roster. Cites the enforcing `tick` + the new `tests/pending_join.rs`. Additive bullet, no
  renumber/restructure.
- **`swarm-training-tdd.md` status line — factual update.** Changed "Not yet scheduled." to record
  that P0/P1/P2 suites landed through the P2 waves (B1 + B2B3 + B4), with a pointer to **this
  ledger** as the authoritative per-ID coverage map; P4/P5 (§7) remain future debt. No renumber.
- **Drift sweep — no other spec/TDD edit warranted.** The spec is deliberately **version-agnostic**
  (§10.4/§16 say "the next `WireVersion` at merge time" — the concrete v40→v41→v42 history lives in
  the code (`WireVersion::CURRENT`) + `swarm-p2-ledger.md`, correctly; injecting `v42` into the spec
  would introduce future staleness). JoinCredentials / observe surface / lazy-backend host-boundary
  inventory / per-platform `device_limits` / `cuda` feature / live-endpoints posture / RUN-10 are all
  **implementation records** already carried faithfully in the Merge-1/Merge-2 sections of
  `swarm-p2-ledger.md` and the lane ledgers; the design docs describe them at the (still-accurate)
  contract level. Recorded here so the "no change needed" verdict is auditable rather than an omission.
- **Carried spec-amendment proposals remain LEDGER-ONLY (unchanged by B4):** the §10.5 unified-governor
  clamp, §5.1 fp32 note, and the `tabi@1 FROZEN AT:` marker (`swarm-p2-ledger.md` "Carried items").
  These are human sign-off items on the spec's normative body and are **not** part of B4's factual
  reconciliation; left for the human as previously recorded.

## Deviations / notes

- **DAUP guest-decode truncation negative — deliberately deferred (rationale).** The update container
  ("DAUP") is opaque bytes the swarm moves + hashes but never parses (§7.3); corruption/truncation on
  the wire is caught at the transport boundary by blake3 content-addressing (NET-2
  `verify_artifact_tamper`), and the *encoder* corners (the consensus-critical half — hashed/signed
  bytes) are covered by `cbor_canonical.rs`. A guest-internal decode-truncation negative would live in
  `daemon-train` guest code and carries **no cross-peer consensus exposure**; recorded as future ABI
  hardening, not a P2 gate blocker.
- **RUN-6 `join_prefers_p2p_then_hub` + NET-7 — deferred to P4** (iroh-blobs payload plane; plan
  "Explicitly deferred"). The R2/Hub source is the P2 baseline; the P2P-preference ordering needs the
  blob plane that P4 introduces.
- **Known flake untouched:** the `daemon-conformance` detached-delegation trio (no swarm lane touches
  it; green in isolation).
- **NEW pre-existing flake surfaced for Merge-3 — `drills.rs::late_join_mid_run_syncs_and_contributes`
  (NOT a B4 regression; NOT modified).** Empirically, this Merge-2 drill is a **timing flake on this
  box**: it fails intermittently (~1 in 3 here) with a 20 s harness recv-timeout —
  `"the late peer reports a digest for round 3"` — and **passes green in isolation** and on most runs.
  Verified pre-existing: `git diff fe27b9c -- tests/daemon-swarm-e2e/tests/drills.rs` is **empty**
  (B4's transient checkpoint-resync drill was reverted; the proof now lives in synchronous
  `run_units.rs`), and the flake still fires with the file byte-identical to Merge-2. Root cause: the
  late peer must checkpoint-resync + contribute round 3 before the `LocalCoordinator`'s 1500 ms
  quiescence / phase timeouts advance past it; under scheduler pressure that window is occasionally
  missed. **Treat as green-in-isolation** (same disposition as the `daemon-conformance` trio). A
  robust de-flake (bump the harness `quiescence` / recv budget in `daemon-swarm-run/src/harness.rs`)
  is a run-crate/e2e-owner call, deliberately **not** taken here to avoid changing shared harness
  timing for other lanes on the eve of the gate — flagged for Merge-3 to decide.
- **No frozen-file / wire / `tabi` edits.** New test files + doc edits only; `WireVersion` stays v42.

## Gate results (lane close) — GREEN (test-side + docs only)

Jobs capped at 8 (≤ nproc/2); one build at a time.

- `cargo fmt --all --check` ✓ · `typos docs/specs` (edited docs) ✓ · ReadLints (edited files) clean ✓.
- `cargo clippy --all-targets -- -D warnings` on every touched crate — `daemon-swarm-coordinator`,
  `daemon-swarm-run`, `daemon-swarm-e2e` (pulls the swarm/train stack) ✓.
- `cargo deny check` ✓ (advisories/bans/licenses/sources — B4 adds no deps).
- `cargo test -p daemon-swarm-coordinator` ✓ — admission 11, **commit_rule 12** (+2 B4),
  determinism 3, **pending_join 4** (B4), proto_units 6, tick_lifecycle 14.
- `cargo test -p daemon-swarm-run` ✓ — lib 41, record_ordering 2, **run_units 6** (+1 B4),
  tinystories_fixture 5.
- `cargo test -p daemon-swarm-e2e --test drills` — **5/5 in isolation**; `late_join_*` is a
  pre-existing timing flake (green in isolation, drills.rs byte-identical to Merge-2 — see Deviations).
- `cargo build --target wasm32-unknown-unknown --release` (`daemon-swarm-{proto,coordinator}`) ✓ ·
  `cargo run -p xtask -- build-guests` ✓ (per-worktree manifest drift NOT committed — warn-and-rebuild
  per Merge-1 adjudication; canonical `guests.blake3` restored).
- Scope hygiene: `git diff fe27b9c` touches only the 2 coordinator test files, `run_units.rs`,
  the 2 reconciled docs, this ledger, and `pending_join.rs` — **no src / frozen-file / wire / `tabi` /
  guest-manifest change**.

Not re-run (test-side lane touched no src; unchanged vs the Merge-2 green matrix): full
`cargo test --workspace`, the net/train/sdk/observe suites, and the live/GPU lanes — B4 adds only
test files in `daemon-swarm-coordinator`/`daemon-swarm-run` (both run green above) + doc edits.

## What Merge-3 (the gate) must know

1. **The P2 TDD gap register is closed.** The coverage map above is authoritative; the only
   deliberate open items are P4 (NET-7 P2P blobs, RUN-6 `join_prefers_p2p_then_hub`, HOST-4 8-bit
   opt-state) and P3 (WIRE-4 app view-model) — all out of the P2 gate by the program plan.
2. **§6.2 min_peers guidance is now spec + test.** Any live `min_peers` run at the gate MUST set
   `min_peers` = the expected initial roster (the `ws_live_workers` harness already does). A worker
   that races the warmup transition is staged pending and (with `epoch_rounds=0`) never joins mid-run.
3. **Checkpoint-resync in the live worker rejoin is a design note, not wired.** The gate's churn drill
   still rejoins fresh-state (survivors' byte-identity holds; the rejoiner's post-rejoin digests are
   out of the identity assertion). Wiring real resync (surface the latest checkpoint manifest →
   `resume_from_checkpoint` → replay retained rounds) touches the worker bin + `SwarmService` (A-lane)
   + a small additive cloud pointer — see the design note above. Until then, the gate should assert
   "run finishes after churn", not "rejoiner byte-identical".
4. **Pre-existing flake to disposition:** `drills.rs::late_join_mid_run_syncs_and_contributes` is a
   timing flake on this box (green in isolation, ~1-in-3 under load, byte-identical to Merge-2). Treat
   green-in-isolation like the `daemon-conformance` trio, or bump the harness quiescence (run-crate
   owner's call).
