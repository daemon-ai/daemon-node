# VHC program state

**This is the only program document an agent reads by default.** It is rewritten — never
appended — at every program boundary, and it is status, not normative text. Normative text lives
in the tracked specs (§2). If this file contradicts a chat log, a memory, or an archived
document, this file wins; if it contradicts a tracked spec, the spec wins.

Last rewritten: 2026-08-05 (pre-C2 hardening Phases 1–5 landed: cadence contract + typed
storage taxonomy, deaf-path root cause + fix, incremental authenticated archive
publication, GREEN replay assembly from the product archive, and sandboxed coordinator
reconstruction in the join transaction; C1 hardware evidence unchanged under
`~/experiments/ceremony-artifacts/c1-20260728/archive/`).

## 1. What we are building, and the next milestone

daemon-vhc is a virtual heterogeneous cluster: every unit of run policy — the training algorithm
and the consensus protocol — is a sandboxed wasm module authored against the SDK, and the host is
capabilities plus cryptographic mechanism (four worlds: `compute@` Burn-shaped, `net@`, `data@`,
`sys@`). Modules train on the native lane and reconcile on the bit-exact det lane.

The next milestone is **C2, the fleet ceremony** (§5): a **replicated-training protocol proof on
heterogeneous consumer hardware** — a TinyLlama-class model trained by three trainer boxes over
real WAN through the product path only, with zero det-digest mismatches, churn drills, checkpoint
restore, and a green offline replay (criteria G-1..G-6 in the ceremony runbook). It is *not* a
7B/40B product certification, *not* a performance study, and *not* a measurement campaign.
Sharded/offloaded execution at larger scales is post-ceremony work.

## 2. Canonical documents

- `docs/specs/vhc-architecture-spec.md` — architecture + resource model (normative)
- `docs/specs/vhc-module-abi-spec.md` — module ABI (normative; drift-gated against the code)
- `docs/specs/vhc-fleet-ceremony-runbook.md` — ceremony definition and runbook (normative)
- `docs/vhc-program-state.md` — this file (status)
- Program history: `~/experiments/decentralised-llm-training/archive/` — **frozen, read-only,
  off-limits to agents**. Never cite it as authority; it is human archaeology only.

## 3. Resource doctrine — the machine is the oracle

The three resource artifacts keep their owners: the guest derives a **logical** resource plan
(no device physics), the backend implementation owns its **execution profile**, the node owns
its measured **capability report**. What changed at the reset is their epistemic role:

1. **Composition output is a conservative estimate, not an authority.** It exists for exactly
   two purposes: sound cheap refusal (if even the estimate exceeds supply, refuse without
   probing) and sizing the enforced runtime budget. Exact high-water prediction is a non-goal.
2. **The fit probe is the authority.** Feasibility of (module, backend implementation revision,
   geometry, grant) is established by running the real module on the real backend at the real
   geometry under the enforced budget, and recording the outcome as a content-addressed,
   memoized **fit verdict**. Fleet feasibility for a frozen roster = every node holds a green
   verdict. There is no roster-wide search or matching protocol while membership is frozen.
3. **A wrong estimate is a contained, typed runtime failure — never an outage and never a human
   question.** Byte, margin, page-size, and allocator questions are not escalatable to the
   owner; the probe answers them. Only product-semantic questions escalate (§6).

## 4. Rules are gates

A rule that is not enforced by a named mechanical gate (test, lint lane, drift check, type) is
advisory. The gate battery is `cargo run -p xtask -- vhc-production-gate` (plus the det/node/t2
lanes and the acceptance suite); the codename scan and the ABI drift check stay mandatory. The
rule-identifier namespace is frozen: no new rule families, and an amendment to the tracked specs
must delete or subsume at least as much normative text as it adds.

## 5. Ceremony ladder

Each rung is a strict subset of the one above and runs continuously below it. Defects must be
discovered at the lowest rung capable of showing them.

- **C0 — every merge (CI).** One box, three real daemon processes (coordinator seat + trainers)
  through the product path: training rounds, digest agreement, churn/hard-kill restore, replay,
  live module switch. This is the acceptance lane (`tests/daemon-vhc-acceptance`), pinned as the
  merge gate's non-negotiable core (`xtask vhc-acceptance`, folded into `vhc-production-gate`).
  Status: **green** — the plan-emitting trainer admits through PC-12 dev-authority provisioning;
  all twelve gates pass.
- **C1 — on demand / nightly.** Two real boxes, real transport (relay), small geometry; adds
  WAN churn drills and remote product-path drive. Status: **green on real hardware.**
  The relay-carried transport posture is a merge gate (`iroh_relay_plane.rs`), and the
  physical rung has now run end to end: run-j (Strix Halo + M4 over the M1-mini relay and
  the Cloudflare registry/R2 content plane) completed 24/24 rounds with zero digest
  mismatch, and run-k passed the hard-kill and graceful leave/rejoin trainer drills with
  digest agreement restored after each (§7).
- **C2 — the milestone.** Three boxes over WAN at ceremony geometry per the runbook
  (G-1..G-6), on a frozen candidate. Preflight is only what G-1..G-6 require — relay, sealed
  binaries, identities, corpus, genesis, seat, and a green fit verdict per box — each item
  memoized by content hash so a re-freeze re-runs only what changed. Per-OS conformance and
  profile-calibration campaigns are **post-ceremony** product work, not preflight.

## 6. Open owner decisions (maximum five; product semantics only)

1. **Milestone labeling** — C2 is claimed as a replicated-training protocol proof on the frozen
   three-box roster (not a product certification). Confirm at freeze.
2. **Availability verification in the ceremony genesis** — on (full coverage) vs off (recorded
   as reduced coverage). Decide at freeze.
3. **Numerics bar** — the tolerance class accepted for trained state across backends. Confirm
   the ratified optimizer-class tolerance at freeze.
4. **Privacy/threat posture** — corpus, journals, and metrics are egress channels; confirm the
   accepted posture for the public ceremony evidence.

None of these blocks C0 or C1.

## 7. Current state (rewrite this section at every boundary)

- Integration trunk: `vhc-integration` at `24293e06`, clean. The One-Lifecycle/Two-Identities
  wave is fully landed and field-proven:
  - **One join transaction** (restart, retry, CLI join, fault recovery share one driver):
    explicit `Starting` state, `Running` only on observed readiness, run-attributed
    pre-session errors, no ghost instances, retry schedule preserved across refusals.
  - **Two identities**: seat lease v2 (new domain tags) separates `leadership_term` from
    `execution_incarnation`; sparse strictly-greater CAS; `SeatTermLedger` fed only by
    verified seat grants (replacing the seat use of `role_floor()`); grant distribution rides
    join bootstrap + WS resubscribe anti-entropy. Roster `RejectedStale` restarts the join
    transaction from verified own-base evidence only; foreign/unverifiable records fail
    closed. Bounded arithmetic both sides (Rust `i64` cast fixed, TS `asU64`/encoder bounds).
    Specs corrected ([ROSTER-1] role in the slot key, [CI-5] per-base supersession, §12.4).
  - **Checkpoint durability seam**: a pointer — live or drain — is announced only when every
    referenced family is durable on the content plane (`upload_referenced_families`;
    `SealedReadError` typed refusals; folds pinned before the upload walk). This closed the
    run-h/run-i poisoned-pointer class.
  - **Tuple-drift adoption**: a drifted `AdmittedTuple` (e.g. macOS `capability_report_digest`
    instability) is a pre-session refusal on join and an adopted fresh assessment on
    reconvergence (REPLACE-mode re-admission) — no infinite retry loop.
  - **Join-time checkpoint freshness**: a restored fence behind the retained record horizon
    (`RETAINED_RECORD_HORIZON_ROUNDS`) refuses typed (`CheckpointStale`) instead of wedging
    into `GapRefused`.
- **Phase A ran on the wire (C1 hardware green).** Two boxes — Strix Halo (coordinator +
  co-located trainer, wgpu/Vulkan SPIR-V) and M4 (trainer, wgpu/Metal MSL) — over the
  M1-mini iroh relay and the `daemon-vhc-dev` Cloudflare registry + R2 content plane:
  - **run-j** (`eff51ffe…`): 24/24 rounds committed, **zero det-digest mismatch** across
    both seats (30 s dual-transcript comparison, 24/24 AGREE); trainer live checkpoints
    every 2 rounds. The durability seam fired in production: R2 briefly served 500s on a
    chunk PUT → typed backoff retries → **pointer withheld**, nothing poisoned.
  - **run-k** (`e09c75da…`): both trainer churn drills PASSED. Hard-kill of the M4 worker
    after the first checkpoint → fresh worker, tuple drift adopted, restore from the live
    round-1 pointer (4 by-ref families streamed from R2, zero payload misses), digest
    agreement restored the next round. Graceful leave/rejoin → drain correctly declined to
    mint mid-round (QuiesceDeadlineExceeded), rejoin restored the role-scoped round-3
    pointer, digest agreement restored — the exact sequence that killed run-i now recovers.
  - run-k epilogue (after the drills): a host disk-full incident killed both host
    instances typed (`journal io error: No space left on device` → `FailedTerminal`);
    recovery re-committed rounds and passed the pre-crash head, but exposed live evidence
    for the named pre-C2 gaps — no coordinator det-state reconstruction (fresh seat,
    operator-cycled trainer realignment), a post-churn deaf-coordinator wedge (zero
    inbound frames while both planes reconnect healthily; second sighting after run-i),
    and the ENOSPC trainer trap misclassified `BadModule`. Stale seat grants from the
    dead generation were correctly refused by the `SeatTermLedger` floors throughout.
    The run was ended by operator leave after the head advanced; full defect ledger in
    the run-k `VERDICT.md`.
  - Evidence: `~/experiments/ceremony-artifacts/c1-20260728/archive/run-{h,i,j,k}/` (each
    with `VERDICT.md`); run-h/run-i are the defect fixtures, run-j/run-k the green runs.
- Doctrine, provisioning, probes, and geometry are unchanged from the 2026-07-28 boundary
  and remain load-bearing: `[RC-15]` fit-verdict authority; PC-12 provisioning
  (`DAEMON_VHC_PROFILE_DIR`, dev authorities doubly-opt-in, integration evidence only);
  ceremony geometry `CEREMONY_SEQ_LEN = 512` (model/seed/expected-root unchanged); **all
  three seats hold GREEN fit verdicts on their native lanes** (bit-identical measured peak
  43,319,296 B; Strix/Vulkan-SPIR-V, 5090/native DX12, M4/native Metal MSL); native-lane
  compile defects D1/D2/D3 fixed and hardware-proven; epoch-watchdog import-liveness
  semantics (ABI §5.6). Note: a worker rebuild changes the backend revision — re-provision
  the profile and re-run the fit probe, or the join refuses `EstimateNotComposable` (typed,
  correct).
- **Pre-C2 hardening (plan `production-ready_ceremony_completion_47713b86`, 8 sequential
  phases) — Phase 1 LANDED (2026-08-05):**
  - **Cadence contract wired**: `remote_ckpt_every` now flows from
    `CeremonyGenesisSpec.remote_ckpt_cadence_rounds` into the trainer's `live` config
    (`daemon-vhc-testkit/src/ceremony.rs`), with authored-vs-consumed regressions
    (`ceremony_authored_round.rs`). **FitProbeKey does NOT move** (test-asserted: the
    harness/assessment form carries no `live` section) — no re-probe needed; envelope
    bytes and hence the run id change unconditionally on the next authoring.
  - **Typed storage taxonomy**: `SinkError` carries a `StorageFault` class
    (`Exhausted` = ENOSPC/quota, `Failed` = permission/corruption/device), classified
    once at the journaling seam from `io::ErrorKind`; new traps `HostStorageExhausted` /
    `HostStorageFailed`; new wire outcome `FailedStorage`; the node persists an M8
    `storage_gated` flag and redispatches a gated run only when the node-state
    filesystem clears `[vhc.storage] reserve_mb` (default 2048; interim gate until the
    disk custodian) — the gated wait consumes NO retry budget and never escalates.
    The run-k ENOSPC → `BadModule` → `FailedTerminal` misattribution is retired, with
    regressions at every layer (sink classification, trap classes, node gate + budget).
  - Docs moved with the change: ABI §7.6 trap table, §12.6 [RS-2] class table,
    §12.10 [RL-4] storage-gated exception, §12.14 [SF-6] cadence wiring note;
    runbook §4.7 cadence check. Housekeeping rules encoded in §9 of this file.
- **Pre-C2 hardening — Phase 2 LANDED (2026-08-05): deaf-path root cause + fix.**
  - **Root cause (DO fan-out audit)**: the registry `RunCoordinatorDO` sequenced WS
    dissemination BEHIND the wasm tick — a throwing/poisoned or uninitialized shell
    silently black-holed all relay while sockets stayed Pong-healthy (hibernation socket
    survival itself was correct). No server liveness signal existed, so clients could not
    distinguish quiet from deaf; Pong-fed idle deadlines can never see this class.
  - **Fix, DO side** (`daemon-cloud` `apps/vhc`): relay-first dissemination (fan-out
    unconditional, tick failures contained + logged, no-shell traffic still relayed) and a
    Binary heartbeat (`DVHC-HB1` + seq + unix-ms) on the alarm cadence (20 s) whenever
    sockets are connected, sharing the alarm with the shell's phase clock. Unit-pinned in
    the new `do-relay` vitest suite (relay with no shell, relay despite a throwing tick,
    heartbeat shape + rescheduling).
  - **Fix, client side** (`daemon-vhc-net::ws_client`): delivery-boundary counters
    (`WsPlaneStats`), heartbeat consumption (never fanned out), and the deafness verdict —
    armed by the FIRST heartbeat of a connection, refreshed by any Binary, Binary silence
    past `ReconnectConfig::binary_silence_deadline` (default 75 s) forces reconnect +
    resubscribe; a pre-heartbeat server never arms it (no blind timer). Regressions in
    `ws_control_plane.rs` (consumed heartbeat; forced deaf reconnect with no cycling).
  - **Instrumentation**: the session counts each boundary (dual-plane forwarded → attach
    verdict by class → module delivery + last-authenticated-inbound) and emits a
    `plane_health` warning event every 60 s when moved and at session end, carrying the WS
    transport counters via the new `RoleProviders::plane_stats` seam — surfaced in
    `vhc detail` recent events.
  - **Live churn regression**: `ws_live_do::live_heartbeat_cadence_and_churn_recovery`
    (relay + real 20 s heartbeat + client churn recovery) green against wrangler-dev with
    the fixed DO; the full 4-test live lane green.
  - Docs moved with the change: ABI §12.8 [LT-5] (relay-first + heartbeat + deafness
    verdict + plane-health counters), runbook §5.4 plane-liveness diagnosis. Known
    pre-existing: cloud `seat.test.ts` has 4 failures on clean HEAD (seat-lease v2 fixture
    drift, predates this phase; convergence work, Phase 8).
- **Pre-C2 hardening — Phase 3 LANDED (2026-08-05): incremental authenticated archive
  publication.**
  - **Wire contract** (`daemon-vhc-proto::archive`): `ArchiveHeadBody`/`ArchiveHeadRecord`
    (domain `daemon-vhc/archive-head/1.0.0`) — the signed per-seal chain-extension claim,
    chain-scoped `(run, role, base identity, chain_instance = founding incarnation)`, with
    successor linking (`predecessor` = the prior chain's terminal head by content address)
    and offline verification (certificate chain to the genesis-trusted bases).
    `ArchiveChainSlot::fold` is the normative structural fold: dense linked extension,
    byte-identical idempotent republish, typed non-extending refusal = fork evidence.
  - **Journal substrate** (`daemon-vhc-journal`): `RotatePolicy.max_open` (production 5 min
    — the recovery-point cadence; age only ever rolls a NON-empty segment, on append),
    `SealHook`/`SealedSegment` fired at every seal, series `founding_id` tracked across
    reopen and the live-upgrade seam, seam hook armed BEFORE the seam roll (the retiring
    span's final seal streams too), an empty segment re-headers instead of sealing
    content-free under a retired identity.
  - **Publisher** (`daemon-vhc-session::archive::spawn_archive_publisher`): spawned by
    `spawn_role` beside every session whose journal home is durable AND whose providers
    carry a head store; startup reconciliation (republish sealed-but-unpublished from
    disk; resolve the succession link), per-seal upload (bytes → content plane, then the
    attested head), capped-backoff retries on transient store faults, typed abort on fork
    evidence; heads attest under the SEALING span's certificate (the switch pushes the
    successor's `SignerBinding`); bounded end-of-session drain (not load-bearing —
    reconciliation covers a missed tail).
  - **Stores** (`daemon-vhc-net::archive_store`): `ArchiveHeadStore` trait;
    `HttpArchiveHeadStore` (registry routes, same credential as the WS plane);
    `FsArchiveHeadStore` (`<run state dir>/archive/heads/`, fold re-applied on open).
    Segment bytes ride the EXISTING content plane (R2/fs) — no new byte surface.
  - **Cloud surface** (`daemon-cloud` `apps/vhc`): `PUT /runs/:id/archive/head`,
    `GET /runs/:id/archive/heads` → `RunCoordinatorDO` archive slots; the TS fold is a
    faithful port (Rust authoritative), untrusted-storage posture identical to
    roster/seat. Vitest suites: shared fold parity + DO route/byte-echo/fork regressions.
  - **Regressions (Rust)**: proto fold + authorization unit tests; journal seal-hook /
    age-bound / founding-identity crash tests; publisher live-stream + crash-reconcile +
    seam-attestation integration tests; net fs-store persistence. Journal, net, session
    (108-test integration suite), observe: all green.
  - Docs moved with the change: ABI **§8.8** ([AR-1]..[AR-6]), runbook §3.4 "where the
    archive comes from", architecture-spec divergence note updated (§5.3 publication is
    now product).
- **Pre-C2 hardening — Phase 4 LANDED (2026-08-05): GREEN replay assembly from the
  product archive.**
  - **Assembler** (`daemon-vhc-observe::journal::assemble`): `assemble_archive` — envelope
    → run id + trusted bases (`envelope_trusted_bases`: `identities.coordinator` +
    `coordinator_set`); every published head record authorized ([AR-4]) + every chain
    re-folded (`verify_chains`); coordinator lineage ordered by succession links
    (`coordinator_lineage`); every content object fetched by address and RE-HASHED
    (untrusted store); committed payloads enumerated from the lineage's published
    `RoundRecord`s; per-peer digest transcripts extracted from its recorded signed
    `Digest` inputs; §3.4 layout written atomically (tmp+rename).
  - **Oracle entry** (`daemon-vhc-observe::consensus`): `recover_chain_from_verified_heads`
    / `replay_consensus_from_verified_archive` — the structural walk for heads whose
    authority the CALLER established through the §8.8 record scheme (the legacy
    `AttestedHead`/`AuthorityConfig` path is unchanged for existing drills).
  - **CLI**: `xtask vhc-archive-pull` (registry descriptor + envelope blake3-verify, head
    snapshot, presigned content GETs through the production `R2Store`/`ContentStore`
    path) → the layout; `xtask vhc-replay` reworked to the product `heads.cbor`
    (`Vec<ArchiveHeadRecord>`, reader-side authorization; multi-chain lineage refused
    typed until reconstruction lands — Phase 5).
  - **Gate** (`daemon-vhc-testkit/tests/archive_assembly.rs`): real sandboxed
    `coordinator_quorum` run (2 workers, 4 rounds, commitments + digests + receipts),
    journaled with the per-seal hook feeding the REAL `spawn_archive_publisher` into
    `FsArchiveHeadStore`/`FsContentStore`; a third party assembles from those stores
    alone and the consensus oracle re-derives GREEN. The actual `vhc-replay` CLI ran
    GREEN over the same assembled archive (5 segments, 4 rounds re-derived, 8 payload
    entries, per-peer digest agreement all rounds; `DVHC_KEEP_ARCHIVE` keeps the layout).
- **Pre-C2 hardening — Phase 5 LANDED (2026-08-05): sandboxed coordinator reconstruction
  in the join transaction (ABI §8.8 [AR-7]/[AR-8]).**
  - **Shared verification in the contract crate**: `verify_chains` / `coordinator_lineage` /
    `envelope_trusted_bases` (+ `VerifiedChain`, `ChainVerifyError`, `latest_round_claim`)
    moved from the observe assembler into `daemon-vhc-proto::archive` — node, worker, and
    oracle all run the SAME verification; no `daemon-vhc-observe` linkage in the production
    host.
  - **The head carries the freshness claim** ([AR-7]): `ArchiveHeadBody.round` (additive) —
    the committed-round watermark at seal time, maintained by a structural probe of published
    frames in the session's egress relay (no consensus schema linked) and stamped by the
    publisher.
  - **Node orchestrates** (`resolve_recovery`): fetch heads via the registry
    (`RunDiscovery::fetch_archive_heads` → `HttpArchiveHeadStore`), verify against the
    envelope's trusted bases — **missing/conflicting/broken lineage = typed join refusal**;
    a seat-role join with verified history carries `SessionCredentials.reconstruct`
    (`CoordinatorRecovery { heads }` — carriage is bootstrap, not trust). `CheckpointStale`
    is REWIRED: staleness judged against the latest VERIFIED committed-round claim across
    the seat lineage (fallback: the graceful-drain pointer), not registry metadata.
  - **Worker executes** (`daemon-vhc-session::reconstruct`): re-verify the carried heads
    against genesis trust; recover the record stream — attested segments (local journal file
    when it hash-matches the head, else the content plane; both re-hashed; `prev_blake3`
    cross-checked) plus the newest chain's local unsealed tail; replay the recorded signed
    frames VERBATIM through a throwaway sandbox instance (consensus never folds natively),
    drain behind a pre-quiesce barrier, export the §10.2 capture — which founds the real
    instance's migration with an anchoring snapshot (`MigrationInput.anchor`: a
    chain-founding migration journals its restore manifest as the new chain's tag-10, so
    every chain is self-contained). Checkpoint-anchored fast path only when the recovered
    stream's snapshot byte-matches the restore capture; otherwise full replay from genesis.
    `RoleReady` only after the rebuilt state is live.
  - **Regressions**: crash → product-path reconstruction → byte-identical resumed decisions
    (`daemon-vhc-testkit/tests/reconstruct_product.rs`); conflicting-heads and
    untrusted-attestor typed refusals (worker); stale-trainer-vs-verified-head and
    conflicting-heads typed refusals (node, `tests/service.rs`).
  - Docs moved with the change: ABI §8.8 [AR-7]/[AR-8]; runbook §5.5 coordinator crash
    drill.
- **Not claimed** (recorded): general archive backfill, in-session exact-frame
  loss repair, §5.3 standby failover, and cross-chain (restart-succession) replay in the
  OFFLINE oracle — `xtask vhc-replay` still certifies a single uninterrupted chain and
  refuses a multi-chain lineage typed (the product reconstruction executor walks the full
  lineage; teaching the offline oracle the seam semantics is follow-on work).
- C0: green and pinned. C1: **green on hardware** (this boundary). Freeze/C2: the only
  rung left — needs the Windows 5090 seat joined in and the pre-C2 workstream scoped.
- Program archive: frozen and locked read-only 2026-07-27.

## 8. Fleet roster and next actions

The roster is status this file MUST carry (an earlier revision dropped it into the archive and
a session then declared reachable boxes unreachable — access facts live here, verified by ssh):

| Box | Access | Hardware | Role | Backend |
|---|---|---|---|---|
| Strix Halo (build host) | local | AMD Strix Halo, 128 GiB UMA, RADV | trainer + coordinator seat + operator seat | wgpu/Vulkan |
| M4 Mac | `ssh m1@62.210.193.129` | Apple M4, 32 GiB unified (memory floor) | trainer | wgpu/Metal |
| Windows 5090 | `ssh usergpu356@37.230.134.194` (cmd.exe; build via sealed Nix cross-build, never on-box) | RTX 5090 32 GiB, Server 2022 | trainer | wgpu/DX12 |
| M1 mini | `ssh m1@51.159.120.241` | Apple M1, 8 GiB | iroh relay only | — |

All four answer ssh (verified 2026-08-04). Next actions (in order):

1. ~~Fit probes at the reduced geometry~~ DONE — all three seats GREEN, native lanes (§7).
2. ~~Run C1 on two boxes~~ DONE — run-j complete + zero mismatch, run-k churn drills
   passed (§7).
3. Pre-C2 workstream (plan `production-ready_ceremony_completion_47713b86`, strictly
   sequential): ~~Phase 1 cadence contract + typed storage taxonomy~~ DONE (§7).
   ~~Phase 2 deaf-path instrumentation + root cause~~ DONE (§7: relay-first DO, server
   Binary heartbeat, client deafness verdict, plane_health surface, live churn
   regression).    ~~Phase 3 incremental authenticated archive publication~~ DONE (§7:
   proto archive contract, seal-hook publisher, head stores, cloud archive slots,
   ABI §8.8). ~~Phase 4 GREEN replay assembly~~ DONE (§7: `assemble_archive` +
   `vhc-archive-pull` + product `vhc-replay`, end-to-end gate GREEN). ~~Phase 5
   sandboxed coordinator reconstruction~~ DONE (§7: node-verified head lineage +
   worker sandbox replay + rewired `CheckpointStale`, [AR-7]/[AR-8]). Next: Phase 6
   disk custody, then C1.5; plus the registry Byzantine posture owner decision (§6).
4. Bring the Windows 5090 seat into a three-box run (its worker lane builds with
   `vhc-net,wgpu-spirv`; native DX12 verdict is already green).
5. Freeze; memoized preflight; run C2; evidence closure; human-signed master merge.

## 9. Agent contract

- **Read this file and your task. Nothing else by default.** The archive is off-limits; do not
  reconstruct program history; do not re-derive decisions recorded here or in the specs.
- **One line of work on the critical path at a time.** No parallel waves on the same worktree;
  measurement/probe builds pin a committed revision, never a working tree.
- **A deliverable is a gate that runs green**, not a report. Reports are one page or less.
- **Never create a normative document, a rule family, or an append-only ledger.** Spec changes
  are commits to the tracked specs. Run evidence goes into a content-addressed directory under
  `~/experiments/ceremony-artifacts/<run-id>/` with a one-line pointer here — never inlined
  into markdown.
- **Timebox investigations.** If a probe or a test can answer the question, run it instead of
  investigating. Escalate only product-semantic questions (§6), never numbers.
- **Resource discipline is unchanged**: capped jobs, one build at a time, diff-scoped lint.
- **Boundary recording.** Every phase boundary rewrites §7/§8 of this file + commits, and
  appends a dated section to the implementation handoff. Plan todos are marked as they land —
  bookkeeping is part of done, never deferred.
- **Reference docs move WITH the change, not after.** A landed change updates its normative
  home in the same commit: trap/outcome changes → ABI §7.6/§12.6/§12.10; checkpoint/cadence →
  ABI §12.14 + runbook §4.7; archive publication/custody → runbook §3.4/§6.1 + architecture
  spec; disk custody → architecture spec resources + config reference.
- **Disk discipline (operator + agent).** Evidence copies never land on the live-run
  filesystem; check headroom before runs and before copies; collectors/logs live under the
  artifacts dir; dead run state is reclaimed via the reconciliation tooling, never ad-hoc `rm`.
