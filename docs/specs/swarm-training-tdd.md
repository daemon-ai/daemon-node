# daemon swarm training — TDD plan (test port + gap register)

**Status:** test-planning companion to
[`swarm-training-spec.md`](swarm-training-spec.md) (architecture design) and
[`swarm-tensor-abi-spec.md`](swarm-tensor-abi-spec.md) (interface contract; its §11 conformance
surface is owned here as HOST-12..15, §3.5). **P0/P1/P2 suites landed** through the P2 waves
(B1's SDK/HOST/PROTO/RUN/CLI suites, B2B3's observe/replay + RUN-10, B4's §6.2 pending-join +
small-n quorum + checkpoint-resync proof); the authoritative per-ID coverage map (TDD ID →
status → suite) is `swarm-ledger-p2-b4.md`. P4/P5 items (§7) remain future debt.
**Purpose:** enumerate the reference-codebase tests worth **porting** for a test-driven build of
the swarm-training stack, classify each by portability, pin the **golden numeric constants**
parity tests must reproduce, and register the behaviors the spec requires that have **no upstream
test** and must be authored fresh. This document is the "what to test, in what order, against
what oracle" layer under the spec's "what to build".

**Provenance.** Grounded in a study of the reference checkouts under
`~/experiments/decentralised-llm-training/` at the commits the spec's Appendix A pins:

| Checkout | Commit | Role in this plan |
|---|---|---|
| `psyche/` (≡ `nousnet/`, verified identical, 784 files, same HEAD) | `0bdb13d9` | primary Rust test-port source (coordinator, network, data, DisTrO) |
| `OpenDiloco/` | `2d750e5` | `diloco` profile numeric oracle (Python) |
| `hivemind/` | `4bd43b7` | epoch/progress **semantics** only (rest deleted per spec §7.1) |
| `node0/`, `AsyncPP/`, `AsyncMesh/` | `32bd084` / `584658e` / `969c9a7` | Phase-2/3 mechanisms — **test deserts**, paper-anchored future debt |
| papers (`*.md`) | working tree | `sparse_loco` + Phase-2/3 golden oracles (no code checkout) |

**Headline findings that shape the whole plan:**

1. **The consensus-critical core is untested upstream.** Psyche's `coordinator.rs` (1293 LoC:
   `tick`, witness quorum, health-check drops, round ring) has **zero `#[test]` modules**; it is
   exercised only through Docker+Solana / TCP integration shells. Our purified
   `tick(state, events, now) → (state', effects)` (spec §6.2, §11.2) must be **test-first from
   scratch** — this is a feature, not a burden: purity makes it property-testable where Psyche's
   `&mut self` was not.
2. **Psyche's coordination *math* is well-tested and ports cleanly.** 15 committee-selection
   tests, 3 assignment tests, and the shuffle/merkle/LCG primitive suites are near-1:1 DIRECT
   ports and become our determinism baseline — the merkle suite doubles as the §6.4
   set-commitment primitive (Psyche's `broadcast_merkle`, adopted). Blooms are N/A: rejected
   as consensus inputs (spec §18 open q. 12), their Psyche role being a Solana-substrate
   health heuristic.
3. **`nousnet/` == `psyche/`** at the same commit (no material delta) — port from `psyche/`.
4. **Prior-art numerics become GOLDEN vectors, not copy-paste tests.** Everything lives on
   tch-rs `Tensor`; we run Burn/CubeCL behind a wasm tensor-ABI. Compression/optimizer tests are
   reproduced as pinned input→output vectors with tolerances, **never** ported as live tch tests.
5. **The v1 flagship (`sparse_loco`) has no reference checkout** (spec Appendix A.14) — its
   2-bit-pack + chunk-top-k + error-feedback loop must be golden-tested **against the papers**,
   explicitly *not* against Psyche's adjacent 1-bit DCT DisTrO path.
6. **Phase-2/3 repos are test deserts** — `node0`/`AsyncPP`/`AsyncMesh` ship `run.bash` training
   scripts and inline asserts, no pytest. Their mechanisms are future paper-anchored debt, and
   several (Grassmann U refresh, modified AdamW, AsyncMesh λ-cosine) carry **double debt**:
   implement from paper *and* author the only test that will ever exist for them.
7. **The spec's biggest test debt is exactly its biggest deltas from prior art**: the round
   protocol (`RoundRecord` commit rule, attestation, stall ladder — spec §6.4), the presigned R2
   payload store, blake3 content addressing (not sha256), CBOR/CDDL (not postcard),
   `update_mb_max`, capability admission, the wasm sandbox/ABI, determinism agree-path, and the
   GPU governor — none have upstream tests.

---

## 1. Method

### 1.1 Portability taxonomy

Every candidate upstream test is tagged with exactly one class:

| Class | Meaning | Action |
|---|---|---|
| **DIRECT** | Logic ports ~1:1 (rename crate/types; for `coordinator.rs`, purify `&mut self` → `state→state'`). | Port the assertion and its inputs. |
| **ADAPT** | Same invariant, different substrate (sha256→blake3, postcard→CBOR, tch→Burn/opaque bytes, equal-split→weighted). | Keep the intent, rewrite the mechanics. |
| **GOLDEN** | A numeric oracle: pin fixed inputs → expected outputs (bit patterns / tolerances / byte counts). | Extract literals (or generate from paper/reference) into a fixture; assert reproduction. |
| **CONCEPT** | Only an integration/E2E path or an *untested* production path exists upstream. | Re-express the behavior as a focused unit/property test. |
| **N/A** | Not relevant (Solana, tch tensors, hivemind DHT/all-reduce, iroh internals, RL swarm). | Do not port; recorded so the decision is auditable. |

### 1.2 Oracles for parity

- **Rust→Rust logic** (coordination): the Psyche test's own asserted values at commit `0bdb13d9`.
- **Compression/optimizer numerics**: pinned literals from Psyche `distro.rs` tests **for the
  `demo` kernels**; a scripted one-shot PyTorch fixture from OpenDiLoCo for `diloco`; the
  **papers** (DeMo Alg. 1; SparseLoCo/Covenant; Protocol Models; AsyncPP; AsyncMesh) for anything
  without a matching checkout.
- **Determinism**: our own fp32 fixed-order reference; the invariant is *cross-peer bit-identity*,
  which no upstream test checks.
- **Model layers** (P1 llama preset): `llama-burn` (tracel-ai/models) and/or a tch reference,
  per spec §15.3 — Psyche has no RMSNorm/attention unit tests.

> **RNG caveat (hard rule):** Psyche's `set_torch_rng_seed()` is *non-deterministic*
> (`rand::rng().random()`, `psyche/shared/modeling/src/lib.rs:72-77`). Golden fixtures MUST use
> embedded literals or an explicit pinned seed (e.g. `0xDAE07E57`), never that helper.

### 1.3 Mapping to the roadmap gates

Test tiers line up with the spec's P0–P5 gates (§17). Each phase below (§8) lists the test suites
that must be green to claim its gate. House conventions apply throughout: `proptest`, `insta`, and
`arbitrary` are already in use in `daemon-node`, and the `daemon-api` CDDL flow
(`tests/protocol_conformance.rs` + `--features arbitrary` proptest + `xtask verify-codec`) is the
template for every wire-contract suite.

---

## 2. Reference corpus at a glance

| Area (source) | Real upstream tests in scope | Dominant class | Target crate |
|---|---|---|---|
| Committee / assignment / shuffle / merkle / LCG / sha256 (`psyche/shared/{coordinator,core}`; bloom N/A) | ~34 | DIRECT / GOLDEN | `daemon-swarm-proto` |
| Coordinator `tick` / witness quorum / health drop / round ring | **0** (integration only) | CONCEPT (author fresh) | `daemon-swarm-proto` |
| Event sourcing + projection (`psyche/shared/event-sourcing`) | 24 | ADAPT | `daemon-swarm-observe` |
| Network: download scheduler / iroh router / blob retry (`psyche/shared/network`) | ~19 | DIRECT / ADAPT | `daemon-swarm-net` |
| Wire serialization (tch tensor / postcard / serialized-distro) | ~19 | N/A / ADAPT | (replaced by CBOR + opaque bytes) |
| Data provider: HTTP ranged GET / weighted / golden corpora (`psyche/shared/data-provider`) | 16 | DIRECT / GOLDEN / ADAPT | `daemon-swarm-run` |
| Client round lifecycle / commitment verify / checkpoint (`psyche/shared/client`) | **0** | CONCEPT | `daemon-swarm-run` |
| DisTrO/DeMo compression kernels (`psyche/shared/modeling/distro.rs`) | 19 | GOLDEN / ADAPT | `daemon-train` kernels / `daemon-train-sdk` |
| DiLoCo optimizer (`OpenDiloco`) | 5 (2 skipped/E2E) | GOLDEN / CONCEPT | `daemon-train-sdk` |
| SparseLoCo (flagship) | **0** (no checkout) | GOLDEN from paper | `daemon-train-sdk` |
| hivemind | ~200, of which **1** adopted | CONCEPT (epoch semantics) / N/A | `daemon-swarm-proto` |
| node0 / AsyncPP / AsyncMesh (Phase 2/3) | **0** | GOLDEN/CONCEPT (future debt) | Phase-2/3 crates |
| rl-swarm / iroh internals | web/dep-only | N/A | — |

---

## 3. Per-crate TDD plans

### 3.1 `daemon-swarm-proto` (crates/contracts) — the consensus core

Serde-only, wasm32-clean, **no tokio/Burn/wasmtime**. This is where determinism and the purified
`tick` live; it carries the heaviest *authored* test load.

**DIRECT ports (determinism baseline):**

| Suite | Upstream | Grounds |
|---|---|---|
| Committee selection (12 tests) | `psyche/shared/coordinator/src/committee_selection.rs:252-423` | §6.3 |
| Equal-split assignment (3 tests) | `psyche/shared/coordinator/src/data_selection.rs:167-236` | §6.3 (baseline; weighted variant is a gap) |
| Swap-or-not shuffle (4) | `psyche/shared/core/src/swap_or_not.rs:36-85` | §6.3 (GOLDEN vectors) |
| Deterministic shuffle (6) + LCG (7) | `.../deterministic_shuffle.rs:19-80`, `.../lcg.rs:29-90` | §6.3 |
| Merkle (11) | `.../merkle_tree.rs:335-426` | §6.4 — the **set-commitment primitive**: `Attestation`/`RoundRecord` sign merkle roots over the committed/verified sets (Psyche's `broadcast_merkle`, adopted); also checkpoint/audit proofs |
| Bloom (4) | `.../bloom.rs:331-379` | **N/A (recorded)** — blooms are rejected as consensus inputs (spec §18 open q. 12, resolved: set commitments make list growth moot and FPs are inadmissible); Psyche's blooms were Solana-substrate artifacts serving a health heuristic |
| Content-hash golden | `.../sha256.rs:43` → **re-pin as blake3** (artifacts/payloads/checkpoints); round state digest is xxh3-128 over seed-keyed sampled blocks (§5.6) — no upstream analogue, see PROTO-18 | §5.6, §6.3 |

**Authored fresh (no upstream coverage) — the gap register for this crate:**

| # | Behavior | Grounds | Proposed test(s) |
|---|---|---|---|
| PROTO-1 | Purified `tick` contract: no hidden mutation; each event class emits exactly one effect class; identical inputs → identical `(state', effects)` | §6.2, §11.2 | `tick_is_pure`, `tick_effects_deterministic`, per-event effect table |
| PROTO-2 | Phase timeout transitions (warmup→train→witness→train/cooldown) | §6.2 | table-driven `timeout_*` scenarios (port the *intent* of `centralized/testing/integration_tests.rs`) |
| PROTO-3 | `NUM_STORED_ROUNDS=4` ring; `data_index`/`height` threading; cursor advances by `global_batch` per round (sequences/round semantics) | §6.2, §6.1 | `round_ring_wraps_at_4`, `data_index_threads_across_rounds`, `cursor_advances_per_round` |
| PROTO-4 | Witness quorum ⌈⅔·n⌉ incl. adopted small-n specials (1→1, 2→2, 3→2) | §6.3 (`coordinator.rs:710-722`, untested upstream) | `witness_quorum_table` (n=1..32) |
| PROTO-5 | **Round-record commit rule** as a pure function of signed messages (I6): entry iff `Commitment` ∧ (`StorageReceipt` ∨ witness-quorum `Attestation`) — no inline I/O; freeze on all-accounted or deadline; record signs the **set-commitment root**, set ordered by node-pubkey bytes; membership/absence provable against the root | §6.4 | `record_requires_commit_and_evidence`, `commit_rule_consumes_only_signed_messages`, `record_freezes_at_deadline`, `record_root_matches_set`, `set_order_is_pubkey_bytes`, `membership_proof_verifies`, `absent_peer_not_in_record` |
| PROTO-6 | Health-check accusation → `Dropped`; healthy peer rejects accusation | §6.4, §13 | `health_check_marks_dropped`, `healthy_trainer_rejects_accusation` |
| PROTO-7 | 15 s heartbeat + **K record-absences** drop counter (daemon Delta); `Straggle` heartbeats don't count as absence during the stall window | §6.4 | `peer_silent_emits_stale`, `k_absences_drops`, `straggle_within_window_not_dropped` |
| PROTO-8 | Throughput-class-weighted assignment + deliberate 0–10% overlap (daemon Delta vs equal-split); class ladder boundaries (c1..c4) | §6.3, open q.1 | `assignment_weighted_by_class`, `overlap_zero_is_partition`, `overlap_10pct_covers_churn`, `class_ladder_boundaries` |
| PROTO-9 | Global-batch **ramp** schedule (`[data].global_batch start/end/ramp_rounds`) + `[data].stop` termination (tokens/rounds → Cooldown → Finished) + `epoch_rounds` boundary | §6.1, §6.2 | `global_batch_ramps_linearly`, `stop_tokens_finishes_run`, `epoch_ends_at_epoch_rounds` |
| PROTO-10 | Coordinator-elected checkpointer (tie-breaker committee), deterministic from seed | §9 | `checkpointer_deterministic_from_seed`, `elects_single_checkpointer` |
| PROTO-11 | Envelope freeze to **canonical CBOR** (authoring TOML → one signed byte sequence), hash chain envelope→config-bytes→`da_build` input | §6.1, §16, ABI §6.1 | `rejects_unknown_schema_major`, `rejects_missing_artifact`, `freeze_idempotent`, `frozen_bytes_stable_across_toml_formatting`, `config_subslice_is_da_build_input`, `verify_signature_rejects_tamper` |
| PROTO-12 | Capability-set **subset** admission (required ⊆ advertised); envelope capability/cadence lists are pre-screen only — assess re-derives from module bytes | §6.5, §16, ABI §2.1 | `assess_subset_ok`, `assess_missing_op_rejected`, `envelope_capability_mismatch_rejected_at_assess` |
| PROTO-13 | `SwarmProtoVersion` exact-match join rejection | §16 | `join_rejects_mismatched_version` |
| PROTO-14 | Halted states (`Uninitialized`/`Paused`/`Finished`) return error effect; pause/resume only from author/org-admin principals | §6.2, §11.1 | `tick_halted_states_error`, `pause_requires_authorized_principal` |
| PROTO-15 | Verifier committee is a no-op at `verification_percent = 0` (seam, not faked) | §6.4, §12 | `verifier_noop_at_zero_percent` |
| PROTO-16 | **No-float** / WASM determinism of `tick` (open q.7) | §11.2 | `tick_module_no_float` lint-test + integer-only `proptest` |
| PROTO-17 | Epoch/progress semantics adopted from hivemind (3 disjuncts of `ready_to_update_epoch`), DHT removed | §2, A.9 (`hivemind/.../progress_tracker.py:128-134`, `test_optimizer.py:221`) | `epoch_advances_on_batch_target`, `_on_global_lead`, `_on_eta` |
| PROTO-18 | Round state digest: xxh3-128 over seed-keyed sampled blocks — identical block schedule from identical seed; covers params **and** `replicated` persistents; flips on 1-bit change in either | §5.6 | `digest_sampling_schedule_deterministic`, `digest_covers_replicated_persistents`, `digest_changes_on_one_bit` |
| PROTO-19 | Round-protocol message set (`RoundOpen`/`Commitment`/`Attestation`/`StorageReceipt`/`RoundRecord`/`Digest`/`Straggle`) CDDL round-trip + signature-reject fixtures; attestation/record roots constant-size regardless of roster | §6.4, §7.3 | `round_messages_cddl_conformance`, `record_bad_sig_rejected`, `message_size_roster_invariant` |
| PROTO-20 | **Replayability (I1)**: fold (checkpoint, records, payloads) offline → bit-identical post-round state; the resync oracle | §6.4, §9 | `replay_reconstructs_state`, `replay_detects_tampered_payload` |

**Suggested layout** (unit tests beside modules; golden + conformance under `tests/`):

```
crates/contracts/daemon-swarm-proto/
  src/{envelope,capability,version,committee,assignment,witness,tick,types}.rs  (+ #[cfg(test)])
  tests/
    envelope_conformance.rs      # +/- fixtures from spec §6.1 TOML
    golden_psyche_parity.rs      # committee/assignment/shuffle vectors @ 0bdb13d9
    tick_scenarios.rs            # table-driven phases (integration intent as pure sim)
    cddl_conformance.rs + cddl_arbitrary.rs   # daemon-api pattern; first proptest in the stack
```

### 3.2 `daemon-swarm-net` (crates/swarm) — transport

**DIRECT / ADAPT ports:**

| Suite | Upstream | Class | Grounds |
|---|---|---|---|
| Download scheduler (14 tests: capacity, FIFO, retry classes, max-retries) | `psyche/shared/network/src/download/scheduler.rs:411-675` | DIRECT | §7.1, §9 |
| iroh router shutdown + **allowlist** isolation | `.../router.rs:70,105` | DIRECT | §7.1, §7.2 |
| Blob download retry (sender abort / mid-download) | `.../test.rs:233,316` | ADAPT (emulate; blake3) | §7.1, A.10 |
| TCP challenge signature reject | `.../tcp.rs:397` | CONCEPT | §7.2 |

**N/A:** `serializable_tensor.rs` (8, tch), `serialized_distro.rs:188` (tch+postcard), Psyche TCP
coordinator handshake (`tcp.rs:342,370,439`) — daemon has no Psyche-TCP coordinator surface.

**Authored fresh — gap register:**

| # | Behavior | Grounds | Proposed test |
|---|---|---|---|
| NET-1 | R2-store presigned PUT/GET round-trip + URL expiry; coordinator `HEAD` check emitted as a signed `StorageReceipt` — the commit rule never sees the raw response (I6) | §7.1, §6.4, §11.1 | `store_presign_roundtrip`, `store_presign_expired_rejected`, `head_emits_signed_receipt` (mock presign server) |
| NET-2 | **blake3** artifact-map verification (Psyche uses sha256) | §8, §12 | `verify_artifact_ok/tamper`, blake3 golden vector |
| NET-3 | `r2://` / `hf://@rev` / `https://` scheme resolution + revision-pin immutability | §8 | `resolve_hf_pinned_ok`, `unpinned_hf_rejected`, `r2_to_presign` |
| NET-4 | Per-object payload-plane fallback (blobs↔store) from commitment locators | §7.1 | `blob_fetch_fails_falls_back_store`, `locators_tried_in_cost_order` |
| NET-5 | `SwarmTransport` trait conformance across both payload planes (parametric table) | §7.1 | `swarm_transport_conformance` over store + iroh-blobs impls |
| NET-6 | Signed CBOR gossip envelope (ed25519) accept/reject; same message via WS and gossip dedupes | §7.1, §7.2, §7.3 | `signed_gossip_bad_sig_rejected`, `ws_gossip_duplicate_message_dedupes` |
| NET-7 | P2P model sharing per-parameter blob tickets (Psyche `p2p_model_sharing.rs` untested) | §9, A.10 | `model_param_ticket_download_verify` |
| NET-8 | Round-object retention: fetch succeeds within `payload_retention_rounds`, expired object → typed miss (feeds the stall ladder) | §7.4, §6.4 | `retained_object_fetchable`, `expired_object_typed_miss` |

**Support to build:** mock presign (Miniflare/R2) server, in-memory iroh node harness (extract from
Psyche `test.rs` spawn pattern), `SwarmTransport` mock tier.

### 3.3 `daemon-swarm-run` (crates/swarm) — engine-agnostic participant runtime

**DIRECT / GOLDEN ports:**

| Suite | Upstream | Class | Grounds |
|---|---|---|---|
| HTTP ranged-GET provider + `BatchId`→tokens; seeded shuffle determinism | `psyche/shared/data-provider/tests/http.rs:53,116` | DIRECT | §8, §6.3, A.11 |
| Golden corpora decode (fineweb / dolma / hermes3) | `.../tests/local.rs:21,63`, `.../tests/preprocessed.rs:29` | GOLDEN (vendor fixtures) | §8 |
| Weighted multi-corpus mixtures (8 tests) | `.../tests/weighted.rs:56-271`, `src/weighted/http.rs:180` | ADAPT (feature-gated, later) | §8 |

**Authored fresh — gap register:**

| # | Behavior | Grounds | Proposed test |
|---|---|---|---|
| RUN-1 | `update_mb_max` receive-side rejection **before** guest decode | §7.3, §13 | `payload_over_cap_rejected_before_decode` (mock backend records `decode` not called) |
| RUN-2 | Commitment-then-payload: obtain the committed set (inline or `record-set.cbor`), verify it against the record's root, then verify each payload's blake3 against its set entry before staging; mismatch → drop+demerit | §7.3, §6.4, A.10 (`client/state/steps.rs:613`, untested) | `set_verifies_against_record_root`, `tampered_set_object_rejected`, `commitment_verify_ok/mismatch`, `stage_rejects_hash_not_in_set` |
| RUN-3 | Tokenized `manifest.json` validation + `BatchId`→(shard,offset) local mapping; peer slices its interval into `steps_per_round` × micro-batches | §8, §6.3 | `manifest_batchid_maps`, `invalid_manifest_rejected`, `interval_slices_into_h_steps` |
| RUN-4 | Artifact LRU cache bounded by `data_cache_gb` | §8, §10.6 | `artifact_cache_lru_evicts` |
| RUN-5 | Round lifecycle join/warmup/train/cooldown transitions (client is untested upstream), incl. **barrier invariant I2**: first `da_step` of r+1 happens-after `da_ingest_updates(r)` returns; early `RoundOpen(r+1)` doesn't reorder | §10.2, §6.4, ABI §2.3 | `round_lifecycle_transitions`, `ingest_barrier_orders_next_round` (CONCEPT from Psyche E2E) |
| RUN-6 | Checkpoint manager: **2 elected checkpointers, register only on hash match**; `replicated` persistents included fp32-exact; join tries P2P then R2/Hub | §9 (E2E-only upstream: `decentralized/.../integration_tests.rs:571,731`) | `checkpoint_registers_on_both_match`, `single_uploader_degraded_flag`, `checkpoint_roundtrips_replicated_fp32`, `join_prefers_p2p_then_hub` |
| RUN-7 | Desync (digest mismatch) → resync = checkpoint + record/payload **replay** (PROTO-20's oracle, run end-to-end) | §9, §5.6, §6.4 | `digest_mismatch_triggers_replay_resync`, `resync_beyond_retention_waits_for_epoch` |
| RUN-8 | **Stall ladder**: missing committed payload at the barrier → skip training r+1, keep fetching, late-ingest, catch up within `stall_rounds_max`; publishes nothing while stalled; exceed budget → leave for epoch rejoin | §6.4, §13 | `stall_skips_training_keeps_fetching`, `stalled_peer_publishes_nothing`, `catchup_within_budget_rejoins`, `stall_exhausted_leaves` |
| RUN-9 | Preemption-as-churn: `Throttle{paused}` mid-entry-point → abort, VRAM freed, CPU masters retained; resume re-instantiates, rebuilds, rejoins at boundary (crash and pause share the path) | §10.5, §6.4, ABI §2.2 | `preempt_aborts_and_frees_vram`, `resume_reenters_at_boundary`, `crash_and_pause_share_recovery_path` |
| RUN-10 | Assess staging: envelope pre-screen (capabilities, round-mode compatibility) before module fetch; manifest cadence re-derived and verified == envelope | §6.5, ABI §6.2 | `prescreen_rejects_before_fetch`, `manifest_envelope_cadence_mismatch_rejected`, `demo_module_ineligible_on_slow_coordinator` |

**Support:** vendor `psyche/shared/data-provider/tests/resources/` (fineweb/dolma/hermes3 +
tokenizers); port the `TestServer` static-file harness; `MockTrainerBackend` that records ABI calls.

### 3.4 `daemon-train-sdk` (crates/contracts) — guest profiles (numeric golden suite)

All profile math is GOLDEN. Each golden test's docstring MUST record the delta from its nearest
prior-art source (see §5).

**`demo` (§5.3.3) — richest existing oracle:** reproduce Psyche `distro.rs` fixed-literal tests as
vectors: 4×4 DCT/IDCT basis (`distro.rs:807,836`), `test_compress_1d/2d` (`:849,875`),
`test_encode/decode_1d/2d` (`:927-1008`), sign path `test_signed_vals_*`/`test_1bit_*`
(`:1009,1113`), wire roundtrip `serialized_distro.rs:188`. **Delta to encode in tests:** spec `demo`
transmits sparse fp coefficients then signs the *aggregate* and uses α=0.2 partial subtraction
(`demo…md:226`); Psyche transmits 1-bit signs and does full IDCT subtraction with decay 0.999 —
golden the two paths separately.

**`diloco` (§5.3.2):** author a one-shot PyTorch fixture from OpenDiLoCo
`train_diloco_torch.py:342-353` / `hivemind_diloco.py:158-167` for `pseudo_grad = θ_off − θ_main`
and the outer Nesterov step; pin constants (`outer_lr=0.7, momentum=0.9, nesterov`; inner AdamW
`lr=4e-4, wd=0.1, betas=(0.9,0.95)`). Port the **skipped** `WAIT_FOR_ALL`/`NO_WAIT` test
(`test_diloco_hivemind.py:160`) with synthetic timings; add int8/`uniform8bit` pseudo-grad codec
round-trip.

**`sparse_loco` (§5.3.1) — flagship, no checkout (A.14):** author end-to-end from paper +
piecewise DiLoCo/top-k pieces, **not** Psyche's DCT path:

| # | Behavior | Proposed golden/property test |
|---|---|---|
| SDK-1 | Full round: H AdamW → Δ=θ⁽ᵗ⁾−θᵣ → acc=β·e+Δ (β=0.95) → chunk-top-k → 2-bit Q → residual e | `sparse_loco_round_golden` (2 peers, small tensor, fixed θ) |
| SDK-2 | 2-bit absmax pack/unpack (4-level, per-chunk fp16 codebook) | `absmax_pack_2bit_roundtrip` (+ proptest) |
| SDK-3 | Index codec ≤12 bits/value within 4096 chunk | `index_codec_proptest` |
| SDK-4 | **median-norm clip** of contributions before aggregate | `median_norm_clip_golden` |
| SDK-5 | Outer step θ−α·mean(Δ̂), α=1 (and plain-SGD vs Nesterov ablation, open q.2) | `sparse_loco_outer_step_golden` |

### 3.5 `daemon-train` (crates/coprocessor) — host runtime, kernels, ABI, determinism

Feature-gated backend lanes; the default build is a stub. All authored fresh (no upstream tests).

| # | Behavior | Grounds | Proposed test |
|---|---|---|---|
| HOST-1 | `dct2@1`/`idct2@1` orthonormality + reconstruction bound per tile size (8..128) | §15.2, ABI §5.8 | `dct2_orthonormal_per_tile` (extends Psyche 4×4 goldens to 64×64) |
| HOST-2 | `topk_chunk@1` (chunk=4096 / tile²) per-row selection | §15.2, ABI §5.8 | `topk_chunk_golden` (k=64 invariant) |
| HOST-3 | 2/1-bit `absmax_pack@1` GPU-vs-CPU parity + §6.6 layout golden | §15.2, ABI §6.6 | `absmax_pack_golden`, `absmax_layout_bytes_golden` |
| HOST-4 | blockwise 8-bit optimizer-state quant (block=4096, bitsandbytes semantics) | §5.1, §15.2 | `opt_state_quant_8bit_roundtrip` |
| HOST-5 | `det_sum@1` + `det_chunk_scatter_add@1`: bit-identical accumulation given record-order staging; streaming (scatter_add + drop) ≡ batch (`det_sum`) bit-exactly | §5.6, ABI §5.9/§5.11 | `det_sum_record_order`, `streaming_equals_batch_aggregation`, `host_stages_record_order` |
| HOST-6 | Outer step via `det_reset_param_to_base@1` + `det_axpy_param@1`: bit-exact θ⁽ᵗ⁺¹⁾; round-base snapshot taken at the ingest barrier | §5.6, ABI §5.9 | `det_outer_step_golden`, `round_base_snapshots_at_barrier` |
| HOST-7 | Round digest (xxh3-128 sampled, seed-keyed, covers `replicated`) cross-peer stable; full blake3 at checkpoints; flips on 1-bit change | §5.6 | `digest_stable_across_peers`, `digest_changes_on_one_bit` (shares vectors with PROTO-18) |
| HOST-8 | `meta` mode shape-only propagation (param layout, activation/payload/VRAM/**RAM** estimate, fuel + op counts incl. two-point ingest fit) | §5.1, §6.5, ABI §2.4/§6.4 | `meta_mode_shapes`, `meta_mode_payload_size`, `meta_mode_vram_ram_estimates`, `meta_report_schema_valid` |
| HOST-9 | tensor-ABI autodiff parity vs compiled-in Burn (forward + backward); `da_step` loss scaling: grads invariant to host micro-batch slicing (`mb_count` 1 vs 4, same step data) | §5.1, §15.3, ABI §4 | `abi_adamw_step_matches_burn`, `abi_matmul_backward`, `grads_invariant_to_accumulation_split` |
| HOST-10 | wasmtime sandbox budgets: fuel/epoch/memory/op-budget trap → typed `Module` error (worker intact); **ingest budgets scale per-peer** (meta at `min_peers`, execute at `max_peers` must pass) | §5.1, §12, §13, ABI §8 | `budget_exhaustion_traps_typed`, `stale_handle_traps` (generational arena), `ingest_budget_scales_with_count` |
| HOST-11 | Llama preset numerics (RMSNorm/RoPE/SwiGLU/attention) vs llama-burn/tch | §5.1, §15.3 | `rmsnorm_golden`, `rope_golden`, `swiglu_golden`, `attention_golden` |
| HOST-12 | ABI fuzz: arbitrary import streams only ever trap with §3.6 codes — phase-legality matrix, lane rules, handle lifetime (`drop` incl.) as the grammar; every trap code reached | ABI §3.5/§3.6/§11 | `abi_fuzz_traps_only_typed`, `trap_taxonomy_reached` (proptest) |
| HOST-13 | Canonical-CBOR codec conformance (consensus-critical: hashed, signed, fed to `da_build`/`da_manifest`) + DAUP container framing round-trip/truncation/corruption | ABI §6.1/§6.6/§11 | `canonical_cbor_rfc8949_vectors`, `cbor_adversarial_key_order_floats`, `daup_roundtrip`, `daup_truncation_rejected` |
| HOST-14 | T3 re-instantiation replay: drop instance mid-round → re-instantiate → `da_build` re-derives identical stable handles + round-base views | ABI §2.2/§11 | `reinstantiate_rebuilds_identical_state` |
| HOST-15 | Mode blindness + manifest purity: same bytes in meta/trace/execute (no mode-observable import); `da_manifest` pure function of config, cadence block matches envelope copy | ABI §2.4/§4/§6.2 | `module_mode_blind`, `manifest_pure_and_matches_envelope` |

### 3.6 `daemon-train-client` (crates/coprocessor) — node-side supervisor

Mirrors `LocalProvider`↔`daemon-infer`. Upstream analogue is Psyche's length-framed **postcard** over
TCP (`tcp.rs:439`); we use length-framed **CBOR over stdio**.

| # | Behavior | Grounds | Proposed test |
|---|---|---|---|
| CLI-1 | Worker protocol CBOR round-trip (Probe/AssessRun/JoinRun/Throttle/Leave/Shutdown + Events) | §10.2 | `protocol_roundtrip` per message |
| CLI-2 | Supervisor respawn with backoff → re-`JoinRun` after worker exit | §10.2, §13 | `supervisor_respawn` |
| CLI-3 | Crash-loop meltdown after N failures → `Fatal` surfaced | §10.2, §13 | `supervisor_meltdown` |
| CLI-4 | `Throttle{paused}` = epoch-interrupt abort of in-flight guest call + instance drop + VRAM free (CPU masters retained); graceful straggler; resume path shared with crash (RUN-9's supervisor half) | §10.5, ABI §2.2 | `throttle_aborts_in_flight_call`, `throttle_frees_vram_keeps_masters` |

### 3.7 `daemon-swarm-coordinator` (crates/swarm) + daemon-cloud DO

The dual-deployment proof (Psyche Solana + TCP shells over one `tick`, Appendix A.4) becomes:
one `daemon_swarm_proto::tick` driven by (a) a local axum/WS server and (b) a `wasm32` DO shell.

| # | Behavior | Grounds | Proposed test |
|---|---|---|---|
| COORD-1 | Local server and wasm DO produce identical `(state',effects)` for the same event log | §11.2 | `dual_shell_parity` (replay one log through both) |
| COORD-2 | DO single-writer ordering: out-of-order events serialized; alarms implement phase timeouts | §11.2 | `do_serializes_events`, `alarm_drives_timeout` |
| COORD-3 | `tick` compiles clean to `wasm32` (no float, no std surprises) | §11.2, open q.7 | wasm build smoke + `tick_module_no_float` (shared with PROTO-16) |

### 3.8 wire (`daemon-api`) + app mirror

| # | Behavior | Grounds | Proposed test |
|---|---|---|---|
| WIRE-1 | `SwarmApi` request/response CDDL conformance (fixtures + negatives) | §10.4, §7.3 | mirror `daemon-api/tests/protocol_conformance.rs` |
| WIRE-2 | `arbitrary` proptest: every variant validates against CDDL | §10.4 | `--features arbitrary` |
| WIRE-3 | `WireVersion` bump (next at merge) + `xtask verify-codec` zcbor-vs-ciborium | §16 | existing codec gate |
| WIRE-4 | App renders node-computed eligibility (no client re-derivation) | §10.4, §6.5 | GUI+TUI view-model unit test on eligibility payload |

### 3.9 `daemon-swarm-observe` — event-sourced run log

ADAPT the 24 event-sourcing/projection tests (`psyche/shared/event-sourcing/store.rs:315-535`,
`projection.rs:773-1141`): replace the global `EventStore::init` with injectable backends, drop
tokio from the proto crate, and use CBOR/CDDL event shapes. Grounds §14 and
`daemon-cli swarm observe`/`trace` replay.

---

## 4. Consolidated gap register (no upstream test → must author)

Ordered by blast radius. Each is a *net-new* suite (the reference systems never tested it).

1. **The round protocol** (PROTO-5/19/20, RUN-2/5/8, NET-1/8) — commit rule, record ordering,
   replayability, barrier invariant, stall ladder: the consensus contract all three docs now
   hang off spec §6.4; nothing upstream tests it (Psyche's equivalents are E2E-only).
2. **Purified coordinator `tick` + effects** (PROTO-1..3, PROTO-14) — the only other
   consensus-critical logic; upstream has zero unit tests.
3. **Envelope canonical-CBOR freeze / hash chain / sign + capability admission + proto-version
   gate** (PROTO-11..13) — the entire seam-rule enforcement (§4.3) is untested prior art.
4. **Determinism agree-path** (det staging order, streaming aggregation, outer step, digest
   sampling) (HOST-5..7, PROTO-18) — the property (cross-peer bit-identity) that "silently
   destroys DP training" if wrong, untested anywhere upstream.
5. **wasm sandbox/ABI conformance** (HOST-8..15) — no prior art (spec §15.3 flags it); owns the
   ABI spec's §11 surface: fuzz/trap taxonomy, canonical CBOR, DAUP framing, re-instantiation,
   budget scaling, mode blindness, accumulation invariance.
6. **`sparse_loco` full profile** (SDK-1..5) — v1 flagship, no reference checkout (A.14).
7. **blake3 everywhere + artifact scheme resolution + `update_mb_max`** (NET-2/3, RUN-1) —
   integrity model diverges from Psyche's sha256/no-cap design.
8. **R2 payload store + per-object plane fallback + `SwarmTransport` conformance** (NET-1/4/5) —
   the baseline payload plane has no analogue in Psyche.
9. **Worker protocol + supervision + preemption-as-churn** (CLI-1..4, RUN-9) — adapted framing
   pattern only.
10. **Manifest/BatchId mapping, checkpoint both-match registration, replay resync, staged
    assess** (RUN-3/6/7/10).
11. **Dual-shell (local vs DO) parity + DO ordering** (COORD-1..3).
12. **Heartbeat + K-record-absences, weighted assignment + overlap, checkpointer election,
    batch ramp + stop condition** (PROTO-7/8/10, PROTO-9) — daemon Deltas on Psyche's math.
13. **Phase-2/3 debt** (§7 below) — double debt (implement-from-paper + author-only-test).

---

## 5. Golden numeric constants (parity oracle)

Pin these in fixture headers; every golden test cites its source line and its delta.

**Spec-normative (daemon target):**

| Constant | Value | Profile |
|---|---|---|
| Inner steps H | 30 | `sparse_loco` preset |
| Error-feedback decay β | 0.95 | `sparse_loco` |
| Chunking | 64×64 (2-D), 4096 (1-D) | `sparse_loco` / kernels |
| Top-k | 64 per 4096-chunk (1/64) | `sparse_loco` |
| Quant | 2-bit, 4-level, per-chunk absmax fp16 codebook; indices ≤12 bits | `sparse_loco` |
| Outer α / Nesterov | α=1 (opt 0.65 late); mom 0.9 optional | `sparse_loco` |
| Inner AdamW | lr 4e-4, betas [0.9,0.95], wd 0.1, warmup 1500 | shared |
| `diloco` outer | SGD lr 0.7, momentum 0.9, Nesterov | `diloco` |
| `demo` | momentum β 0.999, k 8..16, α 0.2, wd 0.1 | `demo` |
| 8-bit opt-state block | 4096 absmax | SDK option |

**Prior-art reference values (with deltas to record in tests):**

- Psyche DisTrO (`demo` reference): `compression_decay=0.999` (not 0.95), `compression_chunk=64`
  via divisor search (**not** hard 64×64), `compression_topk=8` prod, **1-bit sign wire**, index
  encoding u8/u16/u32 at 256/65536 thresholds (`distro.rs:412-435`), DCT tol `allclose(1e-4,1e-8)`,
  a lookahead `weight += sign(delta)*prev_lr` (`distro.rs:533`) **not in the DeMo paper**.
- DeMo paper: chunk s=64, β 0.999 default / 0.995 best-ablation, α 0.2, k∈{1,2,4,8,16,32}.
- OpenDiLoCo: `outer_lr=0.7`, `local_steps=500` default (spec table uses H=100 illustratively).

**Fixtures to embed as literals** (already present in Psyche tests, extract verbatim): the 4×4
DCT/IDCT bases, `test_compress_2d` idx `[[3,1],[1,2],[3,2],[1,3]]`+values, `test_compress_1d` idx
`[1,2]`, the `arange(8)` encode/decode outputs, and the `serialized_distro.rs:190-197` 4×4 wire
truth.

**Fixtures to vendor** (copy/submodule into `tests/fixtures/resources/`): Psyche's
`shared/data-provider/tests/resources/{fineweb,dolma,hermes3}` corpora + `decoded/` references and
the `llama2/llama3` tokenizers.

---

## 6. Critical deltas from prior art (bake into every relevant test)

| Topic | Prior art | daemon spec | Affected suites |
|---|---|---|---|
| Coordinator tick | `&mut self` (`coordinator.rs:437`) | pure `tick(state,events,now)→(state',effects)` | PROTO-1, COORD-1 |
| Assignment | equal split (`data_selection.rs`) | throughput-weighted + 0–10% overlap | PROTO-8 |
| Committed set | witness blooms (1% FP) imply liveness; `broadcast_merkle` root alongside | signed **`RoundRecord`** = merkle **set commitment** over the exact committed set (+ content-addressed `record-set.cbor`); attestations sign roots too; availability evidence = signed messages only (`StorageReceipt` ∨ quorum attestation, I6); blooms rejected | PROTO-5/19, RUN-2 |
| Apply ordering | round-start apply, one round late (`train.rs:512`) | v1 `barrier`: ingest at round end (I2); Psyche's shape reserved as `pipelined` mode | RUN-5 |
| Fetch failure | resync or drop | **stall ladder** (`stall_rounds_max`, retention floor) before resync/leave | RUN-8, NET-8 |
| Content hash | sha256 (`core/sha256.rs`, `steps.rs:613`) | **blake3** (content); xxh3-128 seed-keyed sampled round digest | NET-2, RUN-2, HOST-7, PROTO-18 |
| Outer optimizer state | transferred peer-to-peer (hivemind `load_state_from_peers`) or peer-local by algorithm (DeMo) | `replicated` persistents: digested + fp32-exact in epoch checkpoints; never peer-served | RUN-6, HOST-7 |
| Control wire | postcard | **CBOR + CDDL** (`daemon-swarm.cddl`) | WIRE-1..3, CLI-1, PROTO-19 |
| Tensor substrate | tch-rs `Tensor` | Burn/CubeCL behind wasm ABI; payloads opaque bytes | all `daemon-train*` |
| Compression | 1-bit sign of DCT coeffs (DisTrO) | `sparse_loco`: 2-bit absmax values+indices | SDK-1..3, HOST-3 |
| EF / momentum | `delta` buf, decay 0.999, full IDCT subtract | separate `eᵣ`, β=0.95; `demo` α=0.2 partial | SDK-1, SDK-4 |
| Reduction | sum-then-decode / hivemind AVG | `det_sum` / streaming `det_chunk_scatter_add`, fp32, record order, CPU | HOST-5 |
| Payload plane | iroh + Solana blobs | presigned **R2 store** baseline + iroh-blobs optimization; gossip control plane mandatory everywhere | NET-1/4/5 |
| Coordinator deploy | Solana program / TCP server | local axum + **wasm DO** over one tick | COORD-1..3 |
| RNG in tests | non-deterministic `set_torch_rng_seed` | pinned literals / fixed seed | all GOLDEN |

---

## 7. Phase-2/3 future test debt (paper-anchored, no upstream tests)

`node0`/`AsyncPP`/`AsyncMesh` ship no pytest — only `run.bash` training scripts and inline
asserts. Treat those scripts as manual E2E baselines, **not** CI gates. All below are authored
against the papers; items flagged **double debt** exist only in the paper, not the checkout.

**Phase 2 (§5.4):** subspace losslessness `compress→decompress` MSE≈0 when Row(W)⊆Col(U)
(`node0/.../layers.py:502-553`); ~100× / ~1.3 MB boundary-payload byte-count parity; **Grassmann U
refresh** every ~500 steps + cross-stage equality (**double debt**, A.12); **modified
row-constant-second-moment AdamW** (**double debt**); AsyncPP **NAdam β₁=0.99** single-step state
parity; weight-stashing ring correctness under bounded staleness + 1F1B `load_old/new/step`
ordering (`AsyncPP/.../optimizer.py:102-160`, `main_with_runtime.py:504-509`); stage
assignment/replication + **pipe composition on churn**.

**Phase 3 (§5.5):** 2-D DP×PP rank mapping (`AsyncMesh/.../setup.py:92-95`); 5%-subset sparse
averaging index/scatter parity; **EMA delay correction** `w_{t+1}=w_avg_{t-τ}+λ·ema(w_t−w_{t-τ})`
(`sparta.py:125-129`); **λ 0.5→0.01 cosine after 1k iters** (**double debt** — checkout cosines
momentum instead, A.13); τ≤50 staleness bound property; "generalizes eager DiLoCo" limit-case
(delay→0, p_sparta→1).

---

## 8. TDD sequencing (mapped to roadmap gates §17)

Write tests before implementation within each block; a phase's gate = its suites green.

- **P0 — skeleton.** PROTO-1..20 (esp. tick purity, round-record commit rule, envelope freeze,
  capability, version, replayability-on-synthetic-payloads), WIRE-1..3, CLI-1,
  `golden_psyche_parity.rs` (committee/assignment/shuffle DIRECT ports), COORD-3 wasm smoke.
  *Gate:* state-machine + round-protocol property tests + envelope/wire conformance green; stub
  worker joins a local run.
- **P1 — single-host training.** HOST-8..15 (meta mode incl. RAM + two-point ingest fit, ABI
  autodiff parity + accumulation invariance, sandbox budgets + per-peer scaling, llama numerics,
  fuzz/trap taxonomy, canonical CBOR + DAUP codecs, re-instantiation replay, mode blindness),
  SDK `demo`/`diloco` skeleton goldens, RUN-3 (manifest/BatchId/H-slicing), RUN data golden
  corpora, CLI-2..4 supervision + preemption-abort.
  *Gate:* 160M pretrains through the module path; loss within tolerance of reference; tokens/s
  within 25%.
- **P2 — DP swarm (R2 store).** SDK-1..5 (`sparse_loco` complete), HOST-1..7 (kernels +
  determinism agree-path incl. streaming≡batch), NET-1..6/8 (store presign + HEAD, blake3,
  schemes, plane fallback, conformance, signed gossip, retention), RUN-1/2/5..10 (payload cap,
  record verify, barrier, checkpoint both-match, replay resync, stall ladder, preemption,
  staged assess), PROTO-7/8/10 Deltas, `daemon-swarm-observe` projection ports.
  *Gate:* heterogeneous ≥4-GPU WAN run with churn matches centralized baseline within ε; round
  overhead <15% incl. the barrier ingest gap.
- **P3 — public swarm.** COORD-1..2 (dual-shell parity + DO ordering), WIRE-4 (eligibility render),
  registry/presign integration, GUI+TUI view-model.
  *Gate:* private→public promotion; self-serve join/leave/rejoin; custom module completes an epoch.
- **P4 — scale + P2P.** NET-7 (P2P model sharing), iroh-blobs integration (emulate Psyche
  `test.rs:233,316`), HOST-4 8-bit opt-state.
  *Gate:* 1.2B ≥16 peers; blob fetch reduces round latency vs the R2 store.
- **P5 — pipeline parallel.** Phase-2 debt (§7): subspace losslessness, Grassmann, modified AdamW,
  AsyncPP stashing, stage assignment/churn.
  *Gate:* 3B+ across peers none can hold alone; boundary compression verified lossless-in-subspace.

### 8.1 Heterogeneous-GPU determinism CI (the hard infrastructure prerequisite)

HOST-5..7 assert cross-peer bit-identity and the P2 gate requires mixed vendors including a
ROCm/Vulkan peer — but bit-identity is a **CPU property by design** (the det lane never runs on
GPU), which is what makes this tractable in layers:

1. **Per-PR (hosted CI, no GPU):** everything consensus-critical — det kernels via the shared
   `det-core` crate (sim ≡ host implementation), digest sampling, round protocol, replay,
   canonical codecs, wasm guest determinism. Bit-exact assertions run on plain runners because
   the det lane is CPU fp32 by contract (spec §5.6); a GPU could only ever affect the native
   lane, which carries no cross-peer contract.
2. **Per-lane scheduled (self-hosted runners, one per backend):** CUDA / ROCm / Vulkan / Metal
   boxes run the native-lane fixture suite (tolerance classes), HOST-9 parity, and the
   cross-lane replay test (ABI spec §11: native losses within tolerance, det digests equal).
   One runner per lane suffices — cross-peer identity never depends on the GPU.
3. **Hardware-in-loop (the P2 research gate, manually triggered):** the ≥4-peer mixed-vendor WAN
   run with forced churn — `just swarm-dev` against real heterogeneous machines (team/homelab
   fleet), asserting ε-convergence + zero digest mismatches over N epochs. This is a gate
   ceremony, not CI: it runs per release candidate, with pinned envelopes and archived
   `RoundRecord` logs so failures replay offline (PROTO-20's oracle).

The layering is the answer to "how does CI get heterogeneous GPUs": it needs them only at
tier 2 (one box per vendor, scheduled) and tier 3 (a manual gate) — never per-PR, because
every bit-exactness claim was deliberately placed on the CPU det lane.

---

## Appendix — upstream test index (port sources, `@` commit)

**Portable (DIRECT/ADAPT/GOLDEN):**
- Committee: `psyche/shared/coordinator/src/committee_selection.rs:252-423`
- Assignment: `psyche/shared/coordinator/src/data_selection.rs:167-236`
- Shuffle/LCG/bloom/merkle/sha256: `psyche/shared/core/src/{swap_or_not.rs:36-85,
  deterministic_shuffle.rs:19-80, lcg.rs:29-90, bloom.rs:331-379, merkle_tree.rs:335-426,
  sha256.rs:43}`
- Event sourcing: `psyche/shared/event-sourcing/src/{store.rs:315-535, projection.rs:773-1141}`
- Network: `psyche/shared/network/src/{download/scheduler.rs:411-675, router.rs:70,105,
  test.rs:233,316}`
- Data: `psyche/shared/data-provider/tests/{http.rs:53,116, local.rs:21,63, preprocessed.rs:29,
  weighted.rs:56-271}`
- DisTrO kernels: `psyche/shared/modeling/src/distro.rs:807-1113`;
  `psyche/shared/network/src/serialized_distro.rs:188`
- DiLoCo: `OpenDiloco/tests/{test_diloco_hivemind.py:54,99,160, test_training/test_train.py:43,116}`;
  refs `OpenDiloco/open_diloco/{train_diloco_torch.py:342-353, hivemind_diloco.py:158-167,285-297,
  utils.py:103-107, train_fsdp.py:79-88,250-253}`
- Epoch semantics (CONCEPT): `hivemind/tests/test_optimizer.py:221`,
  `hivemind/hivemind/optim/progress_tracker.py:128-134`

**Untested-upstream production paths to cover fresh:** `psyche/shared/coordinator/src/coordinator.rs`
(tick/witness/health/quorum), `psyche/shared/client/src/{client.rs:466, state/steps.rs:613,
state/train.rs, state/warmup.rs, state/cooldown.rs:212}`, `psyche/shared/network/src/p2p_model_sharing.rs`,
`node0/.../layers.py:502-553`, `AsyncPP/.../optim/optimizer.py:102-160`,
`AsyncMesh/.../sparta/sparta.py:109-129`.

**N/A (recorded):** Solana suites + chaos tests, `serializable_tensor.rs` (tch), Psyche TCP
coordinator handshake, hivemind DHT/matchmaking/all-reduce/MoE/p2p-daemon/auth suites, iroh
patchbay NAT tests, rl-swarm web/Kinesis/DHT-pub tests.
