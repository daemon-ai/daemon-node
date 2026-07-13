# Swarm P2 — Lane B1 ledger (protocol/SDK completion)

Lane **B1** of the **Swarm P2 WAN Program**, Wave 1. Branch `swarm/b1`, base `3cd43c1`
(trunk `integrations/swarm-p2`), worktree `/home/j/experiments/daemon-worktree/p2-b1`.
This lane closes the P2-assigned **TDD test debt** — the convergence-gating golden/conformance
suites — on top of the machinery P1 already shipped. It adds tests (and det-core additions only if
additive); it does **not** re-open frozen seams. `tabi@1` stays FROZEN at 66 ops (no new ops needed;
see "tabi coordination" below).

Program ledger: `swarm-p2-ledger.md`. TDD: `swarm-training-tdd.md` (§3). ABI: `swarm-tensor-abi-spec.md`.

## Scope (TDD IDs) and priority

1. Ledger (this file) — `mirror(B1): ledger`.
2. **SDK-1..5** `sparse_loco` flagship goldens + `demo`/`diloco` remaining (TDD §3.4).
3. **HOST-1/2/5/6/7** full suites (TDD §3.5).
4. **PROTO-7/8/9/10** unit suites (TDD §3.1).
5. **RUN-6/7/10** (TDD §3.3).
6. **CLI-2..4** (TDD §3.6).

## Owned areas (disjoint by construction)

`crates/contracts/daemon-train-sdk`, `crates/contracts/det-core`,
`crates/contracts/daemon-swarm-proto`, `crates/coprocessor/daemon-train` (tests),
`crates/coprocessor/daemon-train-client`, `crates/swarm/daemon-swarm-coordinator`,
`crates/swarm/daemon-swarm-run`, `guests/`. NOT touched: `daemon-swarm-net` (A1),
`daemon-swarm-node` (A1), `daemon-swarm-observe` (B2), frozen files (root `Cargo.toml`, `deny.toml`,
`flake.nix`), daemon-cloud (C1).

## Starting-state audit (what P1/prior waves already landed)

The P2 test debt is **largely already implemented** — the wave-0 gate ran `cargo test --workspace`
green. This lane's value-add is filling the *named-ID gaps* with golden literals + recorded oracle
provenance, and honestly recording what is already covered vs. genuinely new. Audit at base `3cd43c1`:

- **det-core** (`crates/contracts/det-core/src/lib.rs`): all det kernels + rich `#[cfg(test)]` unit
  suites already present — `dct2/idct2`, `topk_chunk`, `absmax_pack`/`det_absmax_unpack`,
  `det_sum`, `det_chunk_scatter_add`, `det_axpy`, `streaming_equals_batch_aggregation` (HOST-5 seed).
- **daemon-train-sdk** (`tests/profiles.rs`): SDK-1/2/4 + demo/diloco **property** tests (reproducible,
  differs, beats-dense) — the golden **literal** completion was the gap.
- **daemon-swarm-proto**: `assignment.rs` (weighted split, overlap, `global_batch_at`,
  `elect_checkpointer`, `witness_quorum`), `digest.rs` (PROTO-18) with tests
  (`assignment_golden.rs`, `digest.rs`) — PROTO-8/9/18 mostly covered; the named PROTO-8/10 units
  and class-ladder boundary golden were the gap.
- **daemon-swarm-coordinator** (`tests/tick_lifecycle.rs`): PROTO-1/2/3/7/9/10/14 lifecycle
  scenarios present — the PROTO-7 heartbeat/stale unit and standalone k-absence/ election units were
  the gap.
- **daemon-swarm-run** (`checkpoint.rs`): RUN-6/7 subset (`checkpoint_save_load_roundtrips`,
  `desync_replay_recovers`) present — both-match/degraded + resync-beyond-retention were the gap.
- **daemon-train-client**: CLI-2 (`supervisor_respawn`), CLI-3 (`supervisor_meltdown`), RUN-9/RUN-10
  fixtures present — CLI-4 supervisor-half throttle units were the gap.

## Golden oracle provenance (frozen at Merge 1)

All this lane's goldens use one of two oracle kinds, cited in each test's docstring per the P1 style:

- **From-definition oracle**: the expected value is recomputed inside the test by an *independent*
  expression of the spec math (plain Rust f32 / a second `det-core` call path), then asserted
  bit-for-bit against the profile/kernel under test. Two independent code paths agreeing on the
  bit-exact det lane is the conformance property. No opaque capture needed.
- **Pinned-literal oracle**: small hand-derivable vectors (absmax pack bytes, top-k indices, DCT of a
  constant block) are pinned as literals with the derivation in the test comment. The pinned daemon
  seed for any generated vector is `0xDAE0_7E57` (matches det-core's `SEED` and
  `assignment_golden.rs`'s `GOLDEN_SEED_RAW`).

Frozen seam this lane exports at Merge 1: **the `sparse_loco` golden fixture set + oracle
provenance** (this section), and **det-core additions** (additive only — see below).

## tabi coordination

**None needed.** Every suite is expressible in the frozen 66-op vocabulary. No new `op@version` was
required; `tabi@1` stays frozen. det-core additions (if any) are new *helper functions* consumed only
by tests, not new ABI ops.

## Additive machinery landed (all in owned crates; no tabi ops; freeze at Merge 1)

Filling the named RUN IDs required a little additive runtime machinery (pure decision functions +
helpers, consumed by the new tests and the future live runtime). None touches `tabi@1`:

- `daemon_swarm_proto::assignment::elect_checkpointers(roster, seed, count)` — the n-checkpointer
  committee (§9), extending the existing single `elect_checkpointer` (RUN-6's two-checkpointer set).
- `daemon_swarm_run::checkpoint::{register_checkpoint, CheckpointRegistration}` — the two-checkpointer
  **both-match** registration + single-uploader **degraded** flag (RUN-6, §9).
- `daemon_swarm_run::checkpoint::{plan_resync, ResyncPlan}` — the retention-floor decision:
  replay-from-checkpoint vs. wait-for-epoch (RUN-7, §6.4/§9).
- `daemon_swarm_run::assess` (new module) — the staged pre-screen `prescreen` (capabilities subset +
  round-mode, **pre-fetch**) and post-fetch `verify_manifest` cadence check (RUN-10, §6.5).

## Test suites added (files)

- `daemon-train-sdk/tests/sparse_loco_golden.rs` — SDK-1..5 + demo/diloco goldens (sim).
- `det-core/tests/host_kernels.rs` — HOST-1/2/5/6 full kernel suites.
- `daemon-swarm-proto/tests/host7_digest.rs` — HOST-7 digest (params+replicated, cross-peer).
- `daemon-swarm-proto/tests/proto8_units.rs` — PROTO-8 class ladder + overlap-zero partition.
- `daemon-swarm-coordinator/tests/proto_units.rs` — PROTO-7/9/10 units.
- `daemon-swarm-run/tests/run_units.rs` — RUN-6/7 (+ RUN-10 unit tests in `src/assess.rs`).
- `daemon-train-client/tests/supervisor.rs` (extended) — CLI-4 throttle supervisor-half units.

## Deviations / carried items

- **Guest manifest drift (flagged for Merge 1 — action needed).** At base `3cd43c1` the committed
  `guests/guests.blake3` entry for `test_abi_basic.wasm` (`034d0e09…`) did **not** reproduce in a
  fresh worktree: a clean `build-guests` deterministically yields `f9d80f26…`, while
  `tiny_llama.wasm` reproduced **byte-exactly** (`198ee07f…` == committed). Since the toolchain
  reproduces the other guest exactly, the committed `test_abi_basic` hash is stale at this base (a
  pre-existing wave-0 drift, not caused by B1 — no guest source, SDK lib, or lockfile changed).
  `guests/` is a B1-owned area, so the manifest was **regenerated** here (per the documented
  maintainer byte-shift workflow) to keep the wasm-harness stale-guest guard green; wasm-backed
  suites (`guest_lifecycle` 9/9) then pass. **Merge 1 must adopt the regenerated manifest (or
  reconcile the canonical value) so every lane's worktree agrees.**
- **RUN-6 `join_prefers_p2p_then_hub` — CARRIED (Wave 2/3).** Needs the per-object payload-plane
  fallback (P2P blobs ↔ R2/Hub, NET-4), which lives in the transport lane (`daemon-swarm-net`, A3),
  not in `daemon-swarm-run`. The checkpoint save/load already rides `PayloadStore`; the source
  *ordering* preference is a transport concern to wire when A3's plane fallback lands.
- **RUN-10 `demo_module_ineligible_on_slow_coordinator` — CARRIED (needs a wire/design decision).**
  The frozen `Manifest` carries only `min_round_interval_ms` (a floor); expressing "demo is stale on
  a too-slow coordinator" needs a **max**-interval / staleness-tolerance field — an additive Manifest
  design decision to coordinate at Merge 1, not a lane-local test. `prescreen_rejects_before_fetch`
  and `manifest_envelope_cadence_mismatch_rejected` (the other two RUN-10 IDs) landed.
- **tabi coordination: none.** No new ABI ops; `tabi@1` untouched.

## Gate results (lane close, all green)

- `cargo fmt --all --check` ✓ · `cargo clippy --workspace --all-targets -D warnings` ✓.
- Feature-combo clippy `-D warnings`: `-p daemon-train-sdk --features sim` ✓ · `-p daemon-train
  --features burn-ndarray` ✓ · `.#vulkan -p daemon-train --features wgpu` ✓.
- `cargo test --workspace` ✓ (only the documented `daemon-conformance` detached-delegation flake
  failed under full parallelism; **green in isolation** — `cargo test -p daemon-conformance --lib
  node::detached_delegation` 5/5 — never modified).
- Feature tests: `-p daemon-train-sdk --features sim` ✓ · `-p daemon-train --features burn-ndarray`
  4/4 ✓ · `.#vulkan -p daemon-train --features wgpu --test wgpu_lifecycle` 3/3 ✓ (real RADV GPU).
- `cargo build --target wasm32-unknown-unknown` for `daemon-swarm-proto` + `daemon-swarm-coordinator`
  ✓ · `cargo deny check` ✓ (advisories/bans/licenses/sources ok) · `typos docs/specs` ✓.
- `build-guests` ✓ (manifest regenerated — see drift note); wasm harness guard green
  (`guest_lifecycle` 9/9).

## Per-TDD-ID status

| ID | Status | Notes |
|---|---|---|
| SDK-1 | green | `sparse_loco_round_golden` + cross-run determinism + error-feedback across rounds |
| SDK-2 | green | 2-bit absmax layout legality + compression ratio |
| SDK-3 | green | top-k index codec fits ≤12 bits within 4096 chunk |
| SDK-4 | green | median-norm clip golden (clipped master pinned) |
| SDK-5 | green | outer-step golden (α=1 vs late-α ablation) |
| demo/diloco | green | per-step DCT round golden; Nesterov vs plain golden |
| HOST-1 | green | dct2/idct2 orthonormality + reconstruction across tiles 8..128 |
| HOST-2 | green | topk_chunk ties + all-zero (empty) chunks + k boundaries |
| HOST-5 | green | det_sum record order; streaming≡batch over many payloads; host-stages-record-order |
| HOST-6 | green | outer-step composition (reset+axpy) + barrier-snapshot idempotence |
| HOST-7 | green | digest cross-peer stable; covers params + replicated; one-bit flip |
| PROTO-7 | green | peer_silent_emits_stale, k_absences_drops, straggle_within_window_not_dropped |
| PROTO-8 | green | class_ladder_boundaries + overlap_zero_is_partition (+ existing weighted/overlap) |
| PROTO-9 | green | epoch_ends_at_epoch_rounds (+ existing ramp/stop) |
| PROTO-10 | green | checkpointer_deterministic_from_seed, elects_single_checkpointer |
| RUN-6 | green* | both-match + degraded + fp32-exact roundtrip; `join_prefers_p2p_then_hub` CARRIED |
| RUN-7 | green | digest_mismatch_triggers_replay_resync + resync_beyond_retention_waits_for_epoch |
| RUN-10 | green* | prescreen_rejects_before_fetch + manifest cadence mismatch; `demo…slow_coordinator` CARRIED |
| CLI-2 | green | supervisor_respawn (pre-existing, verified) |
| CLI-3 | green | supervisor_meltdown (pre-existing, verified) |
| CLI-4 | green | throttle_aborts_in_flight_call + throttle_frees_vram_keeps_masters (supervisor half) |
