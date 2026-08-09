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

**Two boundaries, kept distinct.** (1) **c15m EXECUTED at `068873f2`** — the two-box
recovery-first workstream is closed on that commit: defects 15–22 found live and fixed with
regressions, the drill run completed 12/12 after a coordinator hard-kill, and the product
archive pull verified end to end. (2) The **cross-chain certification commits are the
SUBSEQUENT candidate boundary the C2 freeze takes**: `f1e43a2e` (certification kernel +
15-case gate), `46df0409` (wasmtime 46.0.2 security patch; the c15m verdict re-ran
byte-identical), plus the `vhc detail` round-telemetry fix (this boundary's commit). Nothing
after `068873f2` has run in a live ceremony; everything after it is offline-verifier,
telemetry, and docs surface.

- **Cross-chain replay certification (2026-08-09, `f1e43a2e`)** — the architecture claim
  "every coordinator decision is offline re-verifiable by anyone from the archive" is now
  executable across restart succession, no new trust authority, no wire change:
  - `daemon-vhc-journal::binding` — the head↔segment binding verifier (sealed + internally
    consistent + every signed identity field agrees, seal pre-seal count == head count),
    adopted at all three readers (session recovery, observe assembly, chain walker).
  - `daemon-vhc-session::reconstruct` — `EndPolicy::Certify` + `ClosureClass`: a completed
    run's replay rides its recorded stop into the guest's own return and must reproduce the
    recorded tag-9 kind-0 outcome (**`terminal` closure**) or remains an honest **verified
    `prefix`**; every reason-2 seam is bound (anchoring tag-10 == the predecessor replay's
    exported manifest; each kind-3 read-back == the section staged at that identity, by
    value or sidecar hash); typed SeamAnchor/Kind3 refusals.
  - `daemon-vhc-observe::journal::certify` — the lineage semantic fold: identical
    replay-forward duplicates deduplicate and count once; different `RoundRecord`s for one
    round = equivocation; deduplicated rounds dense from 0; conflicting per-peer digests
    refuse typed (the assembler's silent last-write-wins is gone).
  - `xtask vhc-replay` routes wire-form archives — single- and multi-chain — through the
    kernel; the verdict carries closure class, chain/span/seam facts, dedup counts, and the
    failing stage on RED. 15-case test gate green (fold + binding + seam units, product
    matrix over real lineages incl. a tampered-publish decision-divergence refusal).
  - **c15m is CERTIFIED**: GREEN over `archive-c15m-final` — 11 chains, 10 seams bound,
    121,570 records, 12 unique rounds (39,029 replay-forward duplicates deduplicated, zero
    equivocation), payload + set-commitment closure, per-peer digest agreement. Closure
    class **`prefix`, why recorded**: the terminal head never published — `RunTerminated`
    reaped the worker before the archive publisher drained on every COMPLETED run (fixed in
    `f1e43a2e`: drain precedes the event) — and c15m's journals were swept post-pull. The
    live completion (`Outcome(0)`, cross-box digest agreement) remains a live claim.
    Archive-borne completeness claims start with C2, whose evidence gate requires
    `terminal`. Verdict: `$C15/replay-verdict-c15m-certified.json` (sha256 `067a45c5…`);
    adjudication upgraded in `$C15/VERDICT.md`.
- **The c15 series (2026-08-06..09, closed at `068873f2`)**: four runs on one freeze
  (tiny-llama `640983cf…` lineage) — c15j/k/l clean 12-round runs (GREEN single-chain
  replay; c15l 19 segments / 11,689 records), c15m the DRILL run (coordinator + workers
  SIGKILL at round 3 → product reconstruction from the verified archive lineage → staged
  archive catch-up → live checkpoint → completion 12/12, final digest cross-box identical).
  Defects 15–22 fixed live with regressions (abandoned-tail adoption, supersession
  terminals, within-horizon catch-up, seat-replacement race, fence-inclusive extraction,
  guest-quiescence-paced delivery, back-pressure vs gap aging). Round-0 quorum digest
  bit-identical across all four runs. Full adjudication: `$C15/VERDICT.md`; forensic
  narrative: the handoff doc's dated 2026-08-08/09 sections.
- **`vhc detail` round telemetry fixed (this boundary)**: the durable run row's
  `last_round` now advances monotonically from every `RoundOutcome` (`VhcStore::
  advance_round`); previously only lifecycle-edge `RunPhase` snapshots wrote it, so a
  training run read `round=0` on the operator surface for its whole life. Regression:
  `lifecycle.rs::round_outcomes_advance_the_durable_round_head`. Verified live on the
  Windows smoke (the counter advanced 0→1→2→3 on the operator surface).
- **Windows 5090 full node GREEN (2026-08-09, Phase 3 of the closure plan)** — the last
  seat runs the full product node. Sealed cross-builds at `851a2ef6` (never on-box),
  SHA-256 verified both sides; `vhc hardware` through the running node names dx12; fit
  probe GREEN on native DX12 at the final worker revision (pinned c15 modules, peak
  43,319,296 B — bit-identical to Strix/M4); single-peer smoke `windows-dx12-smoke-k`
  (run `97512137…`) COMPLETED 4/4 rounds through the full product path (real corpus over
  R2, cadence-2 checkpoint pointers in the registry, steady-state round wall ≈ 8–10 min
  with the 12.6 GB checkpoint walks overlapped). **Its pulled archive is the program's
  FIRST `terminal(0)`-closure certification** — the `f1e43a2e` drain-before-`RunTerminated`
  fix proven live (node log: "archive publisher drained; chain is current" precedes the
  event on both roles), and the C2 evidence gate's required closure class demonstrated end
  to end. Box parked clean (run scope wiped via `vhc wipe`, 31 GB reclaimed, base.key
  `884d77e4…` + dx12 profile + sealed binaries retained; the box profile moved to
  `C:\Users\Administrator` — the old `usergpu356` identity is gone). Evidence:
  `~/experiments/ceremony-artifacts/windows-851a2ef6/VERDICT.md` (+ verdict JSON, sha256
  `1c990713…`).
- **Three-seat smoke GREEN (2026-08-09, Phase 4 of the closure plan)** — `three-seat-smoke-a`
  (run `bd85e26a…`, all three boxes at `851a2ef6`): Strix coordinator+trainer (Vulkan), M4
  trainer (Metal), Windows 5090 trainer (DX12); min=max=3, stop 6, cadence 4. **COMPLETED
  6/6 rounds, three-way byte-identical digests every round** (`bf057944…` → `12aac917…`),
  clean teardown order on all four role instances, and the pulled archive is the program's
  **first MULTI-SEAM `terminal(0)` certification**: coordinator lineage [217, 219, 222]
  (2 reconstruction seams, anchors + kind-3 bound), 58,417 records, 5,168 replay-forward
  duplicates deduplicated, per-peer fold conflict-free over 4 trainer keys. Two incidents,
  both root-caused FROM THE ARCHIVE (the offline verifier as the forensic instrument, not
  guesswork): (1) a transient R2 egress fault trapped the tiny-llama guest terminal
  (frozen-module misclassification, recovered by rejoin); (2) **the `RosterFull`
  crash-rejoin deadlock** — with a FULL fixed roster (min=max), a crashed trainer's zombie
  entry holds its slot for `k_absences` rounds, every rejoin Join rejects `RosterFull`
  (804 ingested, zero accepted), the round-3 finalize drops the zombie → floor breach →
  `WaitingForMembers` (no timeout) with empty pending, while the rejoined guest had stopped
  announcing on observing round traffic — a deadly embrace that idled the run ~2 h; plus a
  recovery corollary, the **trainer-only silent rejoin** (a leave+rejoin inside the seat
  lease TTL stands down from coordinator duty silently and the resident keeper never
  retries a run whose admitted role is trainer). Evidence + full mechanism:
  `~/experiments/ceremony-artifacts/three-seat-20260809/VERDICT.md` (verdict JSON sha256
  `9ce5a0b4…`). **Consequence bound into Phase 5: the C2 genesis carries churn headroom
  (`max_peers = min_peers + 1`)** — G-3's trainer kill+rejoin drill re-creates exactly this
  deadlock on a full roster.
- The pre-c15 hardening waves (One-Lifecycle/Two-Identities, checkpoint durability seam,
  typed storage taxonomy + disk custodian [AR-9], deaf-path relay-first DO + heartbeat +
  deafness verdict, incremental authenticated archive publication [AR-1..6], product
  replay assembly, sandboxed coordinator reconstruction [AR-7/8], wire v45 disk surface)
  are all landed, field-proven across c1 run-j/k and the c15 series, and recorded in the
  ABI/runbook sections named in git history (`7cf37625`..`6d8c0713`); this file no longer
  chronicles them individually.
- Doctrine, provisioning, probes, and geometry are unchanged and load-bearing: `[RC-15]`
  fit-verdict authority; PC-12 provisioning (`DAEMON_VHC_PROFILE_DIR`, dev authorities
  doubly-opt-in, integration evidence only); ceremony geometry `CEREMONY_SEQ_LEN = 512`;
  all three seats hold GREEN fit verdicts on their native lanes (bit-identical measured
  peak 43,319,296 B; Strix/Vulkan-SPIR-V, 5090/native DX12, M4/native Metal MSL). A worker
  rebuild changes the backend revision — re-provision the profile and re-run the fit probe
  (now a checked runbook preflight item, not a note), or the join refuses
  `EstimateNotComposable` (typed, correct).
- **Standing non-claims (explicit; each is deliberate and recorded, not an oversight):**
  1. R2 payload retention is authored (`payload_retention_rounds`) but NOT enforced by any
     production deletion; payload availability is verified, never assumed.
  2. Durable-watermark backpressure is absent (needed with payload GC + elastic
     membership).
  3. Stalled-record ordering in LIVE ring-replay bursts (defect 21's module-side half) is
     deferred: host-paced catch-up is the product mechanism; the module fix changes the
     module hash and waits for the compatibility-class work.
  4. No general archive backfill and no in-session exact-frame loss repair.
  5. Standby coordinator failover (§5.3) is uncertified; recovery is reconstruction.
  6. Certification completeness is relative to the SUPPLIED heads snapshot: a withheld
     fork is out of scope (fork evidence within the snapshot refuses typed).
  7. c15m's archive certifies as a verified sealed prefix (terminal head unpublished,
     unrecoverable post-sweep — see above); the `terminal` closure class is since
     demonstrated live (the Windows smoke `terminal(0)`, then the three-seat multi-seam
     `terminal(0)`), and C2's gate requires it.
  8. Frozen tiny-llama module (compatibility-class work, neutralized operationally for C2):
     (a) a transient IO fault during payload fetch traps the guest as a deterministic
     terminal (`FailedTerminal`), instead of a retryable straggle; (b) the Join/Heartbeat
     announce loop stops on OBSERVING round traffic (`admitted` flips on any `RoundOpen`),
     not on actual admission — an unadmitted rejoiner goes silent and cannot heal a later
     roster vacancy. Both mitigated by the C2 churn-headroom genesis (non-claim → recorded
     defect, three-seat smoke).
- C0: green and pinned. C1: green on hardware. C1.5 recovery-first: **closed at
  `068873f2`** (c15 series). Windows 5090 full node: **GREEN (above)**. Three-seat smoke:
  **GREEN (above)**. Remaining rungs: freeze → C2 → closure (plan
  `cross-chain_certification_and_c2_closure_82651370`).
- Program archive: frozen and locked read-only 2026-07-27.

<details>
<summary>Pre-c15 boundary chronicle (retired 2026-08-09; retained for provenance)</summary>

- Integration trunk: `vhc-integration` at `6d8c0713` (Phase 6 boundary), clean. The One-Lifecycle/Two-Identities
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
- **Pre-C2 hardening — Phase 6 LANDED (2026-08-06): central disk custody (ABI §8.8 [AR-9];
  wire v45).**
  - **The custodian** (`daemon-vhc-custody`, new crate): ONE per filesystem root
    (per-process registry, canonicalized), atomic reserve→write→commit with an OS
    free-space floor, a global quota, per-run scope quotas, and an emergency margin only
    `Critical` writes (seals, terminals, snapshot bodies, segment headers, archive heads)
    may draw on — sealing always lands. Ambient attachment via `DAEMON_VHC_CUSTODY_ROOT`
    + `DAEMON_VHC_DISK_{RESERVE,QUOTA,RUN_QUOTA,EMERGENCY}_MB` /
    `DAEMON_VHC_DISK_PRUNE_HORIZON_SEGMENTS` (exported to workers from `[vhc.storage]`;
    config reference regenerated). Custodied write paths: journal appends charge their
    exact framed size; spill, payload and archive-head stores reserve before writing.
    The node's resume gate asks the custodian's pressure state
    (`Nominal | Warn | RefuseNew`); the owner ledger carries `run_quota_mb`.
  - **Bounded local storage** ([AR-9]): the archive publisher records archive facts in
    the persisted per-chain custody ledger (`custody.cbor`) on every acknowledged head
    and prunes proven-archived segments outside the recovery horizon (default 4),
    sidecar dependency closure respected, contiguous-prefix only. A pruned chain stays
    re-openable through the **chain anchor** (`chain-anchor.cbor`, written atomically
    BEFORE each delete): recovery verifies the first retained segment against the
    archived predecessor's hash, skips crash-window debris, refuses a missing anchored
    segment. This un-broke the live module switch's seam re-open (found by the gate:
    the pruner ate the prefix, `open_continuation` refused its own chain) — and every
    restart over a pruned chain. The switch exchange also left the 30 s chatty watchdog
    for `deadline_ms + assess_timeout` (quiesce may legitimately consume the full drain
    ceiling; the migrate is assess-class silent compute).
  - **Reconciliation** (`daemon-vhc-node::reclaim`): at service start, superseded
    incarnation dirs are reclaimed IFF their ledger proves every segment archived (the
    newest incarnation is never touched — it is the [AR-8] reconstruction input;
    unknown scopes never reclaimed). Best-effort, loud, never blocks boot.
  - **Wire v45 operator surface**: `VhcDiskUsage` (free/used/quota/reserve/emergency,
    pressure, per-run rows split recoverable vs archived-evidence; orphans flagged) and
    `VhcDiskWipe` (identity-preserving safe wipe: refuses a live run and a standing
    joined intent typed; evidence planes only on explicit request; the identity
    keystore always survives). CLI: `daemon-cli vhc disk` / `vhc wipe`. CDDL +
    conformance + `WireVersion::CURRENT = 45`.
  - **Gate GREEN end to end** (det + t2 + node + acceptance incl. `module_switch_live`
    and `three_node_training`; 3776 s). Boundary commit `6d8c0713`.
- **Not claimed** (recorded): general archive backfill, in-session exact-frame
  loss repair, §5.3 standby failover, and cross-chain (restart-succession) replay in the
  OFFLINE oracle — `xtask vhc-replay` still certifies a single uninterrupted chain and
  refuses a multi-chain lineage typed (the product reconstruction executor walks the full
  lineage; teaching the offline oracle the seam semantics is follow-on work).
- C0: green and pinned. C1: **green on hardware** (this boundary). Freeze/C2: the only
  rung left — needs the Windows 5090 seat joined in and the pre-C2 workstream scoped.

</details>

## 8. Fleet roster and next actions

The roster is status this file MUST carry (an earlier revision dropped it into the archive and
a session then declared reachable boxes unreachable — access facts live here, verified by ssh):

| Box | Access | Hardware | Role | Backend |
|---|---|---|---|---|
| Strix Halo (build host) | local | AMD Strix Halo, 128 GiB UMA, RADV | trainer + coordinator seat + operator seat | wgpu/Vulkan |
| M4 Mac | `ssh m1@62.210.193.129` | Apple M4, 32 GiB unified (memory floor) | trainer | wgpu/Metal |
| Windows 5090 | `ssh usergpu356@37.230.134.194` (cmd.exe; the profile is `C:\Users\Administrator`; build via sealed Nix cross-build, never on-box) | RTX 5090 32 GiB, Server 2022 | trainer | wgpu/DX12 |
| M1 mini | `ssh m1@51.159.120.241` | Apple M1, 8 GiB | iroh relay only | — |

All four answer ssh (verified 2026-08-04). Next actions (in order; the active plan is
`cross-chain_certification_and_c2_closure_82651370`):

1. ~~Fit probes at the reduced geometry~~ DONE — all three seats GREEN, native lanes (§7).
2. ~~Run C1 on two boxes~~ DONE — run-j complete + zero mismatch, run-k churn drills
   passed.
3. ~~Pre-C2 hardening (plan `production-ready_ceremony_completion_47713b86`, Phases 1–6)~~
   DONE — chronicled in §7's provenance fold; boundary `6d8c0713`.
4. ~~C1.5 recovery-first workstream (plan `recovery-first_phase_7_gates_f1d60cb7`)~~
   DONE — the c15 series, defects 15–22 closed, drill run completed + pulled; boundary
   `068873f2` (§7).
5. ~~Cross-chain replay certification~~ DONE — kernel + 15-case gate + c15m certified
   (`prefix`, why recorded); commits `f1e43a2e` + `46df0409` (§7).
6. ~~Windows 5090 full node~~ DONE — sealed cross-builds at `851a2ef6` hash-verified
   on-box, full node + keystore (base identity `884d77e4…`), fit probe GREEN native DX12
   at the final worker revision, single-peer smoke COMPLETED 4/4 with the program's first
   `terminal(0)`-closure certification; round wall ≈ 8–10 min (checkpoint-overlapped) —
   timers finalized at Phase-5 authoring from the three-seat smoke (§7).
7. ~~Three-seat smoke~~ DONE — `three-seat-smoke-a` COMPLETED 6/6, three-way byte-identical
   digests every round, multi-seam `terminal(0)` certification GREEN; the `RosterFull`
   crash-rejoin deadlock found + root-caused from the archive, boxes parked (§7).
8. Freeze (human ratifies `authoring-report.txt` before seeding — runbook §4.7 gate); the
   C2 genesis MUST carry churn headroom (`max_peers = min_peers + 1` — the three-seat
   deadlock consequence) and the smoke-calibrated timers; memoized preflight; run C2 (G-2
   transcript, G-3 drills, G-4 restore, completion, archive pull, GREEN `terminal`-closure
   replay); evidence closure; human-signed master merge.

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
