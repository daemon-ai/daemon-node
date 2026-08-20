# VHC Architecture Specification

**Subsystem:** VHC (Virtual Heterogeneous Cluster) — the daemon training subsystem, design v2.
**Status:** design specification, pre-implementation. [`swarm-training-spec.md`](swarm-training-spec.md)
remains the authoritative description of the **shipping v1 system**; this document is the design
its successor implements. Nothing here changes v1 behavior; the v2.0 launch profile is defined to
reproduce it bit-for-bit (§1.4).

**Sources and citation keys.** This spec cites research the way the v1 specs cite code:

| Key | Document | Role |
|---|---|---|
| map §N | `~/experiments/decentralised-llm-training/decentralized_training_map.md` | the 2020–2026 research map; §10 is the seam/chassis analysis this design is reverse-engineered from |
| VHC ref §N | `~/experiments/decentralised-llm-training/vhc-architecture-spec-v1.md` | **external research-normative input** (an architecture study of the field, not of this system); requirements adopted from it are restated normatively here |
| v1 spec §N | [`swarm-training-spec.md`](swarm-training-spec.md) | the shipping system |
| ABI v1 §N | [`swarm-tensor-abi-spec.md`](swarm-tensor-abi-spec.md) | the frozen `tabi@1` contract |
| P3 ledger | `swarm-p3-ledger.md` | the CUDA/fleet program whose findings §9 normativizes; the ledger file lands on master with the in-flight P3 merge |
| SZK §N | `~/experiments/decentralised-llm-training/sentinel-zk-swarm-v1-complete-spec.md` | SENTINEL-ZK-SWARM-v1 — a worked per-profile instantiation of the §14 layer for the [AL-3] screen (design input only; nothing in it is normative here) |

**Companion documents:** [`vhc-module-abi-spec.md`](vhc-module-abi-spec.md) — the `tabi@2` module
ABI (interface groups, manifest schema, SDK slot crates, grant enforcement, the `plan@1`
derivation contract in full), drafted in parallel with this spec; `vhc-migration.md`
(forthcoming) — the current-code → v2 seam map, staged implementation waves, the complete litmus
decompositions, and the swarm→vhc rename manifest;
[`vhc-capability-reliability-spec.md`](vhc-capability-reliability-spec.md) — the post-C2
capability-seam reliability workstream (transient absorption at the environment boundary,
truthful failure naming, environmental trap attribution, node-level stall announcement,
guest typed run-ends; substantially LANDED as of 2026-08-11 — per-section status in that
spec). Its guest-contract rung assigns ABI minor 6 (`EnvStarved`, ABI §4.5): the trap
taxonomy itself is unchanged — environmental starvation leaves the trap lane entirely and
becomes a typed outcome, so `guest trap ⇒ examine as module defect` sharpens into a
conformance property rather than acquiring new trap codes.
[`vhc-sdk-primitives-spec.md`](vhc-sdk-primitives-spec.md) (2026-08-18) — the SDK training
primitives contract and refactor plan: it **embodies** this spec's reserved seams ([VP-12],
[PIR-11/12], [CO-6/8], [PL-1], [LC-2], [GR-6], §4.2 output vocabulary, §12 monitor, A.4)
into implementation clauses and waves; every clause of this spec it touches is dispositioned
in its Appendix B (embodied / narrowed / deferred / adopted-informative — notably: the
`plan@1` derivation slot of §4.1 is deferred in favor of authored genesis/run configuration;
[PL-1]'s binding reading is pinned there as a consensus-clock constraint, not a local-compute
constraint). Nothing in that document contradicts this one; where wording differs, this spec
remains the design authority and that document the implementation contract. Where this spec
defers a detail to a companion, the deferral is explicit.

**The archived program architecture doc is a TARGET where it diverges.** The frozen
`daemon-vhc-architecture.md` (archived under the program docs) reads as current in places where
it is aspirational; the tracked specs here are authoritative for what exists. Known divergences:
standby coordinator failover has no product caller (coordinator crash reconstruction is a named
pre-C2 workstream); in-session transport sequence-gap recovery is typed-retryable rejoin, not
archive backfill (transport repair, semantic catch-up, and module/journal reconstruction are
three distinct recovery invariants that share the durable archive substrate of ABI §8.8 as it
lands); and the seat lease is scheme v2 — a leadership term separate from the execution
incarnation (ABI §12.4 [SEAT-1]) — superseding any `token == incarnation` framing. Its §5.3
journal-archive publication IS now product: every role session with a durable journal home runs
the incremental per-seal publisher (ABI §8.8; runbook §3.4).

**Naming.** This spec uses the subsystem's new name, **vhc**; the shipping code is still named
`swarm` (`daemon-swarm-*` crates, `SwarmApi`, `daemon-train`). Shipping code is cited by its real
name; new vocabulary, ABI symbols, and future crate/API names use `vhc`. The rename is a separate
documented effort (§1.5).

**Normative language:** MUST / MUST NOT / SHOULD / MAY per RFC 2119. Sections marked
*(informative)* carry rationale and evidence. Rule identifiers (`[VP-n]`, `[PIR-n]`, …) are stable
for cross-reference. American English throughout.

---

## 1. Problem statement and relationship to v1

### 1.1 What v1 hard-codes

The v1 system (v1 spec, ABI v1) is a working, gated, full-replica data-parallel trainer with
properties nothing in the published field matches end-to-end: signed frozen envelopes, a
deterministic consensus lane, byte-identical cross-peer round digests, offline replay, and
churn drills that assert a rejoiner re-enters bit-identical. Four structural commitments are
baked into it:

| # | v1 commitment | Where it lives |
|---|---|---|
| 1 | **One global round clock.** A single coordinator-driven lockstep round paces every peer; round = version boundary = admission unit (v1 spec §6.2, §6.4). | coordinator `tick`, round protocol |
| 2 | **One traffic class.** The only inter-peer payload is the compressed parameter-delta update object; transport, budgets, records, and the det lane all assume it (v1 spec §6.4, §7). | round protocol, payload planes |
| 3 | **Full replica only.** Every peer holds the whole model; eligibility is one VRAM/RAM inequality for one implicit role (v1 spec §5.1, §6.5). | admission, autotune |
| 4 | **Monolithic experiment.** One WASM module owns model, loss, inner optimizer, compression, aggregation math, and the outer step as one opaque artifact (v1 spec §5.1). | `tabi@1`, guest SDK |

These commitments were correct for v1's scope and produced its strongest property. They also mean
that nearly every result in the 2020–2026 decentralized-training literature that is *not*
full-replica lockstep DP — Streaming DiLoCo's fragment-strided sync, AsyncMesh's asynchronous
axes, Factored Gossip's Mix1/Mix2, pipeline stages, subnetworks, modular paths, per-channel
compression — requires host surgery to express (map §1, §10). Today's system is a single
hardcoded point in the space the map describes: layout = replicate, mask = full, routing = none,
consistency = global lockstep round, with the message type named everywhere and the four
highest-churn slots (compressor, sync policy, outer update, aggregation math) fused inside one
opaque module.

### 1.2 What v2 lifts

v2 re-cuts the seams so that the v1 system becomes **one derived instantiation of a general
chassis**, and new research lands as pushed modules rather than fleet upgrades. "Chassis"
(map §10.4–10.5) names the fixed frame everything else plugs into. The four commitments lift as:

1. Global round clock → **group-scoped logical clocks** with membership epochs (§6, §8).
2. One traffic class → **typed channels** in three classes (`param_delta`, `activation`, `kv`),
   with plane, determinism, transport, and compression typed per channel (§4.2.2).
3. Full replica only → **groups** with per-role memory claims and replication targets, admitted
   by measured per-role eligibility (§4.2.1, §10).
4. Monolithic experiment → **one module, composable slots** (§3.3): the `tabi@2` ABI organizes
   guest exports into named interface groups (`model@2`, `compress@1`, `outer@1`, `plan@1`,
   `metric@1`); the SDK becomes slot crates (Rust traits) compiled together into the one module;
   the module manifest self-declares the composition.

The pivotal move is the **chassis split**. Plan *derivation* — deciding groups, channels, states,
and schedule from the envelope's plan inputs, the model graph, and the committed fleet snapshot —
runs **in the guest** (`plan@1`): deterministically, byte-identical across peers, digest-checked
as consensus state, re-derived per membership epoch. The host freezes only the **PlanIR
vocabulary** (§4.2) and all **enforcement**: transports per channel class, det-lane execution,
GPU backends, memory-budget verdicts, fuel, records at schedule commit points, replay,
checkpoint/resync, egress policy. The coordinator consumes PlanIR groups and epochs — it assigns
peers to groups, tracks per-group clocks, manages leases and availability evidence — and **never
sees a strategy name** (§4.1, §6).

*(informative)* Why derivation belongs in the guest: the three derivation passes are pure
functions coupled to the **model graph**, which the guest owns. Layer boundaries, mask structure,
and expert/path modules are experiment concepts; a host-side partitioner would have to understand
model internals, violating the seam in the other direction. Derivation in the guest is what makes
a new partitioning scheme SDK code pushed over the network, not a fleet upgrade — and it composes
with the determinism machinery: every peer derives a byte-identical plan from committed inputs,
the plan digest is checked like any round digest, and no single planner is trusted. Peers and the
coordinator verify the derived artifact, never the derivation — which is why untrusted-ish
experiment code can own the partitioner without owning the fleet.

### 1.3 What v2 preserves

Everything below carries over from v1 with its semantics intact — generalized, not replaced:

- **Signed frozen envelopes** — canonical CBOR (RFC 8949 §4.2), blake3 content addressing,
  author signature over the hash (v1 spec §6.1). v2 adds sections; conventions are unchanged (§5).
- **The det lane** — CPU fp32, fixed-order, bit-exact execution for consensus math (ABI v1 §5.9)
  — re-scoped from a global run property to the capability of `det` channels (§4.2.2, §11).
- **Round digests, signed records, committed sets, merkle roots, offline replay** — the v1 round
  protocol's consensus artifacts and invariants I1–I6 (v1 spec §6.4), instantiated per group
  (§6.1, §7.4).
- **Transports** — WS + iroh gossip control plane; R2/S3 presign and iroh-blobs payload planes
  (v1 spec §7.1) — selected per channel class rather than per run.
- **Coordinator machinery** — the pure `tick` state machine, signed-evidence commit rule,
  deterministic assignment, two-sided admission (v1 spec §6.2–§6.5) — generalized over groups.
- **Worker supervision, sandbox budgets, checkpoint machinery, observe/replay tooling.**

### 1.4 The v2.0 launch constraint (the degenerate plan)

v2.0 ships with the **degenerate plan** as its only enforced instantiation: one group; one
`param_delta`/`consensus`/`det` channel; schedule = per-round `sync` + `commit` (+ periodic
`checkpoint`); interval data assignment with one cursor — today's run shape, re-expressed
(Appendix A.1).

- **[LC-1]** The existing e2e gates — 20-round cross-peer digest equality, offline replay
  equivalence, and the churn drills (drop → park → checkpoint-resync rejoin, byte-identical) —
  MUST hold unchanged for the degenerate plan. They are the migration's regression floor.
- **[LC-2]** A v2 host MUST refuse plans that exercise vocabulary it cannot yet enforce (e.g.
  `activation` channels before an activation transport exists) with a typed admission error —
  never by silent degradation. The vocabulary freezes ahead of the transports on purpose, so
  research can target it (§4.3).
- **[LC-3]** `tabi@1` (`da_*` exports) stays frozen (P3 ledger: FROZEN FOREVER, additive
  `op@version` growth only); v1 modules keep running under v1 hosts during migration. v2 hosts
  speak `tabi@2` (`vhc_*` exports). Coexistence rules belong to the module ABI spec.

### 1.5 Naming: the swarm → vhc rename effort

The subsystem adopts the VHC name; `swarm` is retired by a **mechanical rename wave** that is
part of the future implementation program (first wave after the in-flight P3 program merges),
specified in full in `vhc-migration.md`. Scope summary, so this spec's naming is legible:

- **Renamed:** daemon-node crates (`daemon-swarm-{coordinator,net,node,observe,run}` →
  `daemon-vhc-*`); the wire contract (`SwarmApi`, `swarm.rs` DTOs, CDDL type names → `VhcApi` /
  `vhc-*` — a WireVersion bump + `just update-codec` + app-side vendored codec regen); daemon-cloud
  (`apps/swarm` coordinator app and deployed worker names); superproject justfile recipes, config
  keys, and `SWARM_*`/`DAEMON_SWARM*` env vars; doc/spec filenames for living documents.
- **Kept (documented, not renamed):** historical ledgers (`swarm-p1/p2/p3-*.md`), git history
  (renames land as `git mv`), the operational `swarm-dev` R2 bucket and its minted token (storage
  renaming is a separate opt-in migration), and `tabi@1` symbol names (frozen).
- **Sequencing:** one atomic commit per repo, no behavior change, gates green before and after.

---

## 2. Design principles

*(VP-1..VP-6 operationalize map §10.3's five pluggability decisions plus the composition-evidence
rule; VP-7..VP-12 adopt the VHC ref's P-series where it fits this system. Each principle is
normative for design review: a change that violates one needs an explicit adjudication in its
ledger entry.)*

**[VP-1] Seams live where the literature demonstrated composition.** A seam is an interface
across which one published method varied while its neighbors held fixed: SENTINEL ran with
Protocol-Models compression active; AsyncMesh bolted a pipeline staleness corrector onto sparse
DP averaging; Factored Gossip declares itself orthogonal to compression (map §10). We cut seams
where that evidence exists and refuse speculative seams where it does not.

**[VP-2] The anti-seam is typed away.** DP-gradient compressors on activation traffic are
forbidden **by type**: a compressor binds to a channel *class*, and a slot declared for
`param_delta` cannot be wired to an `activation` channel (Protocol Models, Statement 7.1 via map
§10.3: per-layer approximation error compounds with pipeline depth). What looks droppable-in but
silently diverges must be unrepresentable, not discouraged.

**[VP-3] Message types are derived, never declared.** The plan names groups, channels, states,
and schedule events; the wire carries channel ids, classes, and group rounds. No message, record,
or coordinator state ever carries a strategy name — no "PP"/"DP"/"DiLoCo" strings on the wire
(map §10.5: traffic is a function of placement mismatch; plugins bind to derived typed seams and
therefore survive partitioning schemes invented after they shipped).

**[VP-4] Leaks become declared capabilities.** Where a slot must reach across a seam — the
subspace compressor that requires a row-constant second-moment optimizer, the masking scheme that
biases gradients — the reach is a declared `requires`/`provides` contract checked at wiring time,
never a hidden convention (§13; map §10.3 calls this "the single most important thing to get
right").

**[VP-5] Outer optimizer and staleness corrector are one slot.** Nesterov look-ahead, EMA delay
correction, and Streaming DiLoCo's α-mixing are the same object: an outer update consuming staged
pseudo-gradients plus staleness evidence (map §10.3). `outer@1` is that one slot (§7.1). Building
two seams here would split exactly the place where asynchrony research concentrates next.

**[VP-6] One shared physical assumption, one service.** Every 2025–26 method leans on slowly
drifting trajectories — for staleness correction, for compression drift schedules, and for
anomaly baselines (map §4, §10.3). That assumption is surfaced as one cross-cutting **trajectory
monitor** (§12) rather than re-implemented privately inside each slot.

**[VP-7] Planes never merge.** Control/coordination, per-step execution, consensus exchange, and
assurance are separate communication planes with distinct latency classes and failure semantics
(VHC ref P-1, §5.3). Merging them recreates the synchronization bottleneck v2 exists to remove
(§3.1).

**[VP-8] Determinism is a per-channel capability.** The det lane is the enforcement mechanism of
`det` channels — today, the `param_delta` consensus class — not a global run property. `native`
channels are vendor-variant by declaration and verified statistically, never bit-exactly (§4.2.2,
§11). Conflating the two either cripples execution channels or silently weakens consensus ones
(VHC ref P-6: classify errors by channel).

**[VP-9] One state, one meaning.** Local optimizer state, replicated consensus state, and
transactional residual state are semantically distinct classes with distinct churn semantics and
MUST NOT be aliased by default (VHC ref P-4; §4.2.3). Aliasing is only ever a declared low-memory
profile.

**[VP-10] Churn is state logistics.** The binding cost of membership change is moving weights,
replicated state, and residuals — not lost FLOPs. Residual state carries exactly-once
transactional semantics across every failure point (VHC ref P-8, §9.4; §7.3).

**[VP-11] Provenance is not correctness.** Signatures, merkle roots, and receipts prove who
committed which bytes when; they never prove the bytes came from the required computation.
Correctness needs replay (det channels) or screening (native channels) (VHC ref P-10; §11).

**[VP-12] The group round is the universal primitive.** Version boundary, admission window,
schedule anchor, audit-sampling unit, and (future) billing unit are all the group-scoped logical
clock; no global scalar step exists in the v2 vocabulary (VHC ref P-11, AV-5; §6, §8). The
degenerate plan's single group makes its group round coincide with v1's global round — the
generalization costs v1 nothing.

---

## 3. The seam architecture

### 3.1 Four planes

*(Adapted from VHC ref §5.3 to this system's actors. The latency classes are normative for
transport selection.)*

| Plane | Traffic | Latency class | Failure semantics |
|---|---|---|---|
| **Control** | envelopes, joins/leases, round protocol messages, fleet snapshots, plan digests, checkpoint pointers | seconds–minutes; fully asynchronous | retried; append-only where committed; coordinator is the single writer |
| **Data — consensus channels** | committed payload objects on `consensus`-plane channels (compressed deltas), record-set objects | seconds–minutes; overlapped with training (§4.2.4 barrier rule) | committed sets tolerate absentees; late contributions fold into later rounds (§7) |
| **Data — execution channels** | `activation`/`kv` tensors on `execution`-plane channels (future: pipeline/context traffic) | milliseconds–seconds; latency-critical | reroute to a replica; bounded-staleness rejection; never blocks the control plane |
| **Assurance** | digests, replay verdicts, screen statistics, quarantine/adjudication evidence | asynchronous; never gates a round | divergence detection and quarantine, not round liveness (§11) |

The v1 system has the control plane, the consensus data plane, and the digest half of the
assurance plane. Execution channels are vocabulary-expressible from v2.0; their transports are
enforcement-deferred ([LC-2]).

- **[PL-1]** A schedule event, record, or consensus decision MUST NOT wait on an execution-plane
  transfer; the planes decouple by clock (VP-7; map §10.3 decision 5 — a new scheduler or
  incentive mechanism never touches the hot path).
- **[PL-2]** Every plane's messages carry the version tuple (§8); cross-plane correlation happens
  through the tuple, never through shared mutable state.

### 3.2 The slots, churn-ordered

*(Adapted from map §10.2. "Slot" = a guest-side interface group compiled into the experiment
module; "host" = fixed machinery. Interface stability is inversely correlated with churn — the
fastest-churning components have the most stable interfaces, which is what makes their churn safe
to host.)*

| Seam | v2 home | Interface stability | Expected churn | What varies across the literature |
|---|---|---|---|---|
| Inner optimizer | inside `model@2` (guest code) | fixed core | none | AdamW, everywhere, unchanged |
| Membership / coordination | coordinator (host) | very stable | low | one substrate from Learning@home through 2026 |
| Pseudo-gradient contract | `param_delta` channel class | very stable | low | Δ = θ_before − θ_after, the FedOpt currency |
| Compressor | `compress@1` slots, axis-typed per channel class | stable per class | **highest** | 8-bit, DeMo, 4-bit E3M0, SPARTA, Protocol Models, MoS |
| Sync policy | `plan@1` **schedule output** (declarative) | stable | high | every-step, every-H, fragment-strided, continuous-sparse, triggered |
| Outer update ⊕ staleness corrector | `outer@1` (one slot, VP-5) | emerging | rising | Nesterov look-ahead, EMA correction, α-mixing |
| Aggregation mechanics | host staging: committed sets, residual transactions (§7) | stable | medium | all-reduce → grouped → partial collectives → gossip |
| Aggregation math | `outer@1` via det ops | stable | medium | mean, weighted, trimmed, sign-family |
| Scheduler / placement | coordinator over PlanIR groups | semi-stable | medium | static-optimal ↔ dynamic-adaptive |
| Trajectory monitor | host service + `metric@1` (§12) | immature | low now, high later | drift EMAs, logit-JS disagreement |
| Integrity | host assurance plane (§11) | immature | low now, high later | only SENTINEL exists |
| Incentives | none (out of scope) | undefined | none | sketches only, zero implementations |
| **Partitioner (the chassis)** | `plan@1` derivation + host PlanIR vocabulary (§4) | changes message *types* | structural, **rare** | DP / PP / CP / MoE / paths / subnetworks |

- **[SL-1]** New research MUST land as an envelope edit or a pushed module wherever this table
  marks churn medium-or-higher; host changes are legitimate only for a new vocabulary primitive
  (§4.5).
- **[SL-2]** Compressor slots are axis-typed: a binding names the channel class it serves, and
  wiring a slot to a channel of another class is an admission-time refusal (VP-2, §13). One
  interface per traffic class; never a generic "compress before send".
- **[SL-3]** Sync policy is **plan output**: computed by composable SDK code inside `plan@1`,
  emitted as declarative schedule data (§4.2.4) that host and coordinator interpret. It is
  neither hand-authored envelope DSL nor runtime guest scheduling — same legibility as config,
  full composability of code ([CO-7]).

### 3.3 The artifact: one module, composable slot crates

*(The settled artifact-shape decision, recorded with its rationale.)*

An experiment ships as **one WASM module** whose exports are organized by `tabi@2` into named
**interface groups**: `model@2` (graph, loss, inner loop), `compress@1` (one or more, axis-typed),
`outer@1` (outer update ⊕ staleness correction), `plan@1` (plan derivation, §4.1), `metric@1`
(trajectory statistics). The guest SDK is restructured into **slot crates** — Rust traits
(`Model`, `Compressor`, `OuterUpdate`, `Partitioner`, `Metrics`) with first-party implementations
you compose at build time (`daemon-vhc-sdk-compress-demo`, `-outer-nesterov`, …) — so an
experiment is ordinary Rust: pick crates, or implement a trait, `cargo build` links one artifact.

- **[AR-1]** The module exports a **manifest** (`vhc_manifest`) that self-declares its
  composition: slot names, implementation identifiers and versions, per-slot config schema, and
  per-slot capability contracts (§13). Run records carry the manifest digest inside
  `composition_digest` ([VT-1]), so ablation arms and postmortems are machine-legible even though
  the artifact is a monolith — this recovers, at the metadata level, the tournament-identity
  value that per-slot artifact binding would give (VHC ref §13.2's naming discipline).
- **[AR-2]** Slot interfaces MUST be designed as **protocol objects** — stable, versioned,
  independently conformance-tested, meta-measurable per slot (per-group meta-mode and fuel
  accounting; ABI spec) — even while every slot compiles into one module. This is the one thing
  the single-module launch cannot defer: it is what keeps later promotion of a slot to a
  separately-bindable module a rollout decision instead of a rewrite (the chassis mistake the map
  warns about, §10.5).
- **[AR-3]** The envelope's slots table (§5.1) binds each slot to `(module hash, interface,
  config)` and supports **1..N distinct module hashes**. The launch profile binds every slot to
  ONE hash; multi-module composition is schema-ready and enforcement-deferred, the same posture
  as [LC-2]. Nothing in the schema forbids all slots resolving to the same module — the
  single-module case is the degenerate point of the general schema, not a different architecture.
- **[AR-4]** Issuance ladder under this shape: parameter changes are envelope edits (no build);
  algorithm changes are a slot-crate recomposition → one new module hash pushed per-run over the
  existing artifact plane (v1 spec §8: content-addressed, blake3-verified before instantiation);
  neither touches the fleet. This is rung 1 and rung 2 of §4.5.

*(informative)* Why one module rather than per-slot modules at launch: one frozen ABI surface to
conformance-test instead of three; one meta-mode measurement and one fuel envelope; the
determinism and replay story is character-for-character what v1 has (replay pins one hash); and
migration from `tabi@1` is a regrouping of exports plus an SDK restructuring rather than a host
instance-wiring/broker layer. What is given up — enforced (rather than declared) cross-slot
isolation, per-slot artifact caching, envelope-only recombination of published slots — is exactly
what [AR-2]+[AR-3] keep purchasable later. All tensor and persistent state already lives host-side
behind handles (ABI v1 §3.3), so cross-slot data flow costs nothing either way — a compressor
"owning the momentum buffer" is a grant (§4.2.3), not a memory-layout question.

### 3.4 The trajectory monitor is cross-cutting

The monitor (§12) is a host service, not a slot: it ingests per-peer statistics (`metric@1`) and
host-observed signals, and feeds the three consumers that would otherwise each grow a private
estimator — schedule triggers (§4.2.4), `outer@1` staleness inputs (§7.1), and the integrity
screens (§11). One observable, three consumers (VP-6).

---

## 4. The chassis: PlanIR

### 4.1 Where the plan comes from

A **plan** is the canonical-CBOR document that instantiates the chassis for one membership epoch.
It is produced by the guest's `plan@1` slot:

```
plan = vhc_plan_derive( envelope [plan] inputs × model graph × committed fleet snapshot )
```

*(informative)* Internally, `plan@1` implements the map's three derivation passes (map §10.5):
**memory** (per-role footprint from layout × instantiation — how a configuration is known to
fit), **typed communication** (emit typed channels from placement mismatches — the pass that
produces the buses compressors bind to), and **schedule** (order local steps, syncs, commits, and
checkpoints into events). The map's four-axis space (layout, mask/instantiation,
routing/composition, consistency schedule) is the natural *internal* representation of those
passes inside the SDK's partitioner crates. **PlanIR is deliberately coarser**: it is the
host-visible *output* vocabulary — groups, channels, states, schedule, data — the minimum the
host must understand to enforce. The four axes live in guest library code and churn with
research; the output vocabulary is frozen per `vocab_version` and does not (VP-3).

Derivation rules:

- **[PD-1]** `vhc_plan_derive` runs under det-lane discipline: CPU fp32, no wall clock, no I/O,
  no peer-local randomness, canonical CBOR out. Identical inputs MUST produce identical bytes on
  every peer.
- **[PD-2]** Every measured or floating-point input — fleet snapshot throughput, memory headroom,
  monitor signals — is **quantized by defined rules before entering derivation**: fleet-snapshot
  quantities are bucketed into integer classes at snapshot commit (bucketing rules: the `plan@1`
  contract in the module ABI spec), and schedule trigger thresholds are fixed-point ([PIR-17]).
  Raw floats from measurement never cross into consensus state.
- **[PD-3]** The plan's blake3 digest (`plan_digest`) is consensus state: peers cross-check it
  exactly like round digests; a mismatched peer MUST NOT participate in the epoch (§4.4).
- **[PD-4]** The plan is re-derived at every membership epoch from the new committed fleet
  snapshot; a plan never mutates within an epoch. The plan is derived state — never authored,
  never edited.

### 4.2 The PlanIR vocabulary (NORMATIVE)

PlanIR is a canonical CBOR map (RFC 8949 §4.2 deterministic encoding — the same rules as the v1
envelope, v1 spec §6.1), snake_case keys. Unknown keys are invalid: a plan is consensus state,
not a config file. Top level:

```
{ vocab_version: u32, groups, channels, states, schedule, data }
```

- **[PIR-1]** `vocab_version` names the vocabulary the plan is written in. A host MUST refuse a
  plan whose `vocab_version` it does not implement. v2.0 ships `vocab_version = 1`.

#### 4.2.1 `groups` — who exists

```
group = { group_id: u32, role: text, fragments: u32,
          replication: { min: u32, target: u32 },
          memory: { vram_mb: u32, host_mb: u32 },
          requirements: map }
```

- **[PIR-2]** `group_id`s are dense from 0 and unique. `role` is a plan-chosen label, **opaque to
  the coordinator** (VP-3): it exists for records, logs, and ablation legibility and MUST NOT be
  interpreted by host or coordinator logic.
- **[PIR-3]** `fragments ≥ 1` partitions the group's consensus state into F opaque fragments,
  indexed 0..F−1, referenced by schedule scopes ([PIR-14]) and payload objects. Fragment indices
  carry **no host-visible meaning** beyond scoping: which parameters a fragment covers is guest
  knowledge, deterministic across peers because derivation is ([PD-1]). `fragments = 1` is the
  unfragmented (v1) case.
- **[PIR-4]** `replication.min` is the group's liveness floor — the coordinator parks the run
  when a group falls below it, generalizing v1 `min_peers` (v1 spec §6.1) — and
  `replication.target` is the assignment goal. `min ≥ 1`, `target ≥ min`.
- **[PIR-5]** `memory` is the plan's **claim** of the per-member footprint for this role: weights,
  grads, optimizer state, activations, payload staging, and checkpoint-staging headroom
  ([EB-4]) for the states and channels this group touches. Admission compares the claim against
  the peer's measured autotune verdict (§9.5, §10); a peer whose measured budget cannot cover the
  claim MUST NOT be assigned the role.
- **[PIR-6]** `requirements` reuses the envelope eligibility vocabulary (v1 spec §6.1
  `[requirements]`: `throughput_floor`, `uplink_mbps_min`, `downlink_mbps_min`, `disk_gb_min`,
  capability set), scoped per group.

#### 4.2.2 `channels` — what flows

```
channel = { channel_id: u32,
            class: "param_delta" | "activation" | "kv",
            from_group: u32, to_group: u32,
            plane: "consensus" | "execution",
            determinism: "det" | "native",
            compress: { slot: text, config: map },
            budget: { bytes_per_event_max: u64 } }
```

- **[PIR-7]** `class` types the payload and therefore the legal compressor bindings (VP-2,
  [SL-2]) and the transport family. `from_group == to_group` denotes an intra-group
  (replica-consensus) channel — the v1 update exchange is exactly one such `param_delta` channel.
- **[PIR-8]** Plane and determinism are constrained jointly at `vocab_version 1`:
  `plane: "consensus"` requires `determinism: "det"` (consensus inputs must be bit-exact; the det
  lane is the enforcement), and `plane: "execution"` requires `determinism: "native"`
  (vendor-variant, statistically screened, §11). The other two combinations are invalid — there
  is no use case, and admitting them would blur the error-channel classification (VP-8).
- **[PIR-9]** `compress.slot` names a `compress.<name>` binding from the envelope (§5.1); the
  named slot's declared channel class MUST equal `class`. `compress.config` is that slot's
  plan-resolved configuration — opaque to the host beyond canonical encoding.
- **[PIR-10]** `budget.bytes_per_event_max` is receive-side enforced by the host per event (per
  committed payload on consensus channels, per transfer on execution channels), generalizing v1
  `update_mb_max` (v1 spec §6.1, §7.3). Overflow is a typed refusal, never truncation.
- **[PIR-11]** The exchange pattern on consensus channels at `vocab_version 1` is the **committed
  set** (§7): every group member ingests the identical deadline-frozen set. Pairwise/gossip
  exchange patterns — where members deliberately ingest different subsets (map §3.14) — are a
  future vocabulary axis on the channel object (§4.5), not a config option; they change the
  determinism story and MUST arrive with their own assurance treatment ([CO-8]).

#### 4.2.3 `states` — what persists

```
state = { state_id: u32, name: text,
          class: "local" | "replicated" | "residual",
          scope: { groups: [u32] },
          dims: [u64], dtype: "f32",
          grants: [text] }
```

- **[PIR-12]** `local` and `replicated` carry v1 semantics unchanged (v1 spec §5.1): `local` is
  droppable, never digested, never checkpointed fleet-wide (moments, caches — rebuilt in ≤H
  steps); `replicated` is consensus state — digested every commit, carried fp32-exact in
  checkpoints, bit-identical within each scoped group by construction. Shared protocol objects
  (e.g. a subspace basis) are `replicated` states scoped to every group that reads them; their
  version is the round of the commit that last updated them (§8).
- **[PIR-13]** `residual` is **new in v2**: per-peer state that is *neither* droppable *nor*
  replicated — untransmitted error-feedback whose loss or double-counting corrupts the
  trajectory. Residual state carries the transactional exactly-once protocol of §7.3 across
  rounds, checkpoints, and handoffs. v1 stores error feedback as `local`, which is sound only
  while the compressor treats residual loss as benign re-accumulation; declaring `residual` makes
  the stronger contract available and checkable (VP-9, VP-10).
- **[PIR-14]** `grants` lists the slot names (§5.1 keys) allowed to access the state. The host
  enforces grants at the ABI boundary (mechanics in the module ABI spec); an ungranted access is
  a typed trap. Grants are how the compressor-optimizer leak becomes a wiring-time fact instead
  of a hidden wire (VP-4) — e.g. a DeMo-family compressor is granted the momentum state it owns.
- **[PIR-15]** `dtype` is `"f32"` at `vocab_version 1` (consensus state is fp32 by det-lane rule,
  ABI v1 §5.9). Quantized state classes are a vocabulary extension (§4.5).

#### 4.2.4 `schedule` — when things happen

```
event = { kind: "local_step" | "sync" | "commit" | "checkpoint",
          scope: { group_id: u32, fragment: ?u32, channel_id: ?u32 },
          cadence: { every_steps: u32, phase: ?u32 }
        | trigger: { signal: text, threshold: q16.16 } }
```

- **[PIR-16]** Event kinds are closed at `vocab_version 1`: `local_step` (inner-step pacing),
  `sync` (produce + publish payloads on the scoped channel — all of the group's consensus
  channels when `channel_id` is absent), `commit` (freeze the committed set, ingest, record,
  digest — §7.4), `checkpoint` (host checkpoint of params + `replicated` + `residual` state).
  `scope.fragment` restricts a `sync` to one fragment's payload ([PIR-3]); `scope.channel_id`
  restricts it to one channel — which is how two sync policies with different cadences coexist in
  one group (Appendix A.3).
- **[PIR-17]** `cadence` and `trigger` are mutually exclusive. A cadence event fires at group
  steps s with `(s − phase) mod every_steps == 0` (`phase` defaults to 0). Phase is how
  derivation staggers fragment syncs — strided, sequential, or any pattern `plan@1` computes —
  without the host hardcoding a stride law (VP-3). `trigger.signal` names a trajectory-monitor
  signal (§12); `trigger.threshold` is fixed-point Q16.16 (the f32 multiplied by 2^16, rounded to
  nearest) so plans stay byte-exact ([PD-2]). Triggers are evaluated at group-round boundaries
  against the monitor's committed, identically-quantized signal values, so all members fire
  identically ([TM-4]).
- **[PIR-18] The barrier rule.** `sync` events never block training: a peer publishes and
  continues. Only `commit` gates: the first `local_step` after commit c happens-after the ingest
  of c (v1 invariant I2, per group). Overlap therefore needs no vocabulary: a plan that wants
  communication hidden under compute simply places the commit that folds a sync's payloads later
  than the sync that produced them, and the realized staleness is measured by the version plane
  ([VT-3]) and corrected by `outer@1` (VP-5). This is Streaming DiLoCo's overlap-then-merge and
  Factored Gossip's non-blocking Mix1, expressed as schedule geometry (Appendix A.2, A.3).
- **[PIR-19]** Every group MUST have at least one `commit` event, and every `sync` MUST have a
  reachable subsequent `commit` covering its channel — unrecorded consensus traffic is
  unauditable.

#### 4.2.5 `data` — what feeds it

```
data = { assignment: "interval", cursor_scope: "per_group" }
```

- **[PIR-20]** At `vocab_version 1` both fields have exactly one legal value: contiguous
  `BatchId` interval assignment (v1 spec §6.3 semantics — deterministic, throughput-weighted,
  small deliberate overlap) with one data cursor per group, advanced at that group's commits.
  Alternative assignment families — routed/path-sharded data (DiPaCo), per-fragment cursors —
  are vocabulary extensions (§4.5); the field exists so they extend the document instead of
  reshaping it.

#### 4.2.6 Cross-object validation

- **[PIR-21]** Every `channels[].from_group`/`to_group`, every `schedule[].scope` reference,
  every `states[].scope.groups` entry, and every grant target MUST resolve; fragment indices MUST
  be < the scoped group's `fragments`; dangling references invalidate the plan.
- **[PIR-22]** Validation is host-side and total: a plan that parses but violates any PIR rule is
  refused at admission with a typed error naming the rule. Peers validate independently; since
  validation is deterministic over identical bytes, a plan valid anywhere is valid everywhere —
  [PD-3] makes disagreement detectable, validation makes it unrepresentable.

### 4.3 What the host enforces (and what it refuses)

The host owns, per plan: transports per channel class; det-lane execution for `det` channels; GPU
backends and the memory-budget verdict against `memory` claims ([PIR-5], §9); WASM fuel and
sandbox budgets; records, digests, and replay at `commit` events; checkpoint/resync at
`checkpoint` events; payload budgets ([PIR-10]); and egress policy. The guest owns every decision
*expressed* in the plan and every piece of math the slots implement.

A v2.0 host implements enforcement for: multiple groups on the coordinator surface;
`param_delta`/`consensus`/`det` channels end-to-end; all state classes; cadence-driven schedules;
interval data assignment. `activation`/`kv` channels, `execution`-plane transports, and
trigger-driven schedules are **vocabulary-complete but enforcement-deferred**: [LC-2] requires
refusal, never degradation. This is deliberate — the vocabulary freezes now so modules and plans
can target it; transports land by evidence, without a spec change (Appendix A.4).

### 4.4 Plan lifecycle

1. **Inputs commit.** At epoch start the coordinator publishes the committed fleet snapshot
   (§6.2); the envelope's `[plan]` inputs and the module are already frozen (§5).
2. **Derivation.** Every member runs `vhc_plan_derive` ([PD-1]); the host validates ([PIR-22])
   and computes `plan_digest`.
3. **Digest consensus.** Members exchange `plan_digest` exactly as round digests; the coordinator
   records the quorum digest in the epoch record. A mismatched peer re-derives once, then leaves
   ([PD-3]); a quorum-level mismatch parks the run — it means nondeterminism inside `plan@1`,
   which is a bug, not an operational state.
4. **Enactment.** The coordinator assigns peers to groups (respecting [PIR-4]/[PIR-5]/[PIR-6] and
   current leases), opens per-group clocks at the epoch's base rounds, and the schedule runs.
5. **Epoch turnover.** Membership changes stage until an epoch boundary (§6.3). At the boundary:
   new snapshot → re-derivation → **plan diff → state logistics**. The host compares old and new
   plans and derives the migration set — which state moves where (group splits/merges,
   replication changes, fragment reassignment) — as content-addressed transfers verified before
   the new epoch's first round. Residual state moves under §7.3 commit point 5. A diff the host
   cannot enact (a group whose state has no live source) parks the run for operator action rather
   than silently reinitializing (VP-10). Churn thereby becomes a plan diff plus state logistics.

### 4.5 The frame-change ladder

*(map §10.5's level shift, made operational: the space of schemes is expressible, message types
are derived, and a frame change is needed only for a genuinely new axis.)*

| Rung | Change | Mechanism | Cost |
|---|---|---|---|
| 1 | **Config** (H, k, LR, fragment count, phases, replication targets) | envelope edit → new frozen envelope, same module | no build; per run |
| 2 | **New algorithm or partitioning scheme within the vocabulary** | new SDK composition and/or new `plan@1` → new module hash pushed per run | module build + signing; no fleet touch |
| 3 | **New vocabulary primitive** (new channel class or exchange pattern, new state class, new event kind, new assignment family — a genuinely new axis) | host release + `vocab_version` bump (+ WireVersion where wire types change) | rare by design |

- **[FC-1]** Rungs 1 and 2 MUST NOT require any host or coordinator deployment. This is the
  operational payoff of the chassis split and the property design review protects first.
- **[FC-2]** Rung 3 changes are additive: a host implementing `vocab_version` N SHOULD accept
  plans of version < N under the older rules; modules never re-target old vocabularies.

**The honest caveat** (map §10.5): the ladder pushes frame changes outward; it does not abolish
them. The four internal axes cover the DP/TP/PP/CP/MoE quadrant and the known structural escapes
(SDP subnetworks, DiPaCo paths) as *points*, but work that needs a genuinely new axis — a changed
loss decomposition, a new kind of state beyond parameters/activations/optimizer state, a module
library that grows mid-run, gossip exchange patterns ([PIR-11]) — will force rung 3 no matter how
clean the vocabulary is. The design target is few, orthogonal axes closed over the derivations
that matter, so most research is rung 1–2 and rung 3 stays rare — not "no frame changes ever".
Appendix B walks the seven litmus papers up this ladder.

---

## 5. Envelope v2

The envelope remains what v1 made it (v1 spec §6.1): an authored-then-frozen document, canonical
CBOR, blake3-hashed, author-signed, the only thing non-executing parties read. Freezing, hashing,
and signing conventions are **unchanged**, and the v1 seam rule (v1 spec §4.3 — a field belongs
in the envelope iff a party that never executes run code must read it) still decides every
field's home. v2 changes the *experiment half* of the document: the opaque
`[experiment]`/`[experiment.config]` pair becomes a **slots table** plus a **plan-inputs
section** — the envelope evolves from "opaque experiment config" into a **wiring manifest**.

### 5.1 The `[slots]` table

TOML authoring surface (frozen to canonical CBOR like everything else):

```toml
[slots.model]
module    = "blake3:…"       # launch profile: the same hash in every slot ([AR-3])
interface = "model@2"
config    = { d_model = 1024, n_layers = 24, n_heads = 16, seq_len = 2048 }

[slots."compress.delta"]
module    = "blake3:…"
interface = "compress@1"
config    = { top_k = 64, quant_bits = 2, ef_decay = 0.95 }
requires  = { state_access = ["error_feedback"], channel_class = "param_delta" }

[slots.outer]
module    = "blake3:…"
interface = "outer@1"
config    = { rule = "nesterov", lr = 0.7 }
provides  = { update_family = "linear" }

[slots.plan]
module    = "blake3:…"
interface = "plan@1"
config    = { }

[slots.metric]
module    = "blake3:…"
interface = "metric@1"
config    = { probe_every_rounds = 1 }
```

- **[EV2-1]** Slot names are `model`, `outer`, `plan`, `metric`, and one or more
  `compress.<name>` entries (referenced from PlanIR channels, [PIR-9]). Each binding is
  `{ module: blake3, interface: text, config: map }` plus optional `requires`/`provides` contract
  maps (§13).
- **[EV2-2]** `interface` pins the interface group and version the binding implements. Guest
  export naming is `vhc_<group>_<fn>` (`vhc_model_step`, `vhc_outer_apply`, `vhc_plan_derive`,
  …); the full export set per group belongs to the module ABI spec. At admission the host
  verifies each binding against the module manifest's self-declared composition ([AR-1]);
  envelope and manifest disagreeing is a refusal.
- **[EV2-3]** Slot `config` maps are canonical sub-encodings handed to the slot verbatim — v1's
  `[experiment.config]` discipline (v1 spec §6.1), now per slot. The system carries and hashes
  them, never interprets them.
- **[EV2-4]** A hyperparameter sweep is N envelopes differing only in slot configs and `[plan]`
  inputs, all pinning the same module hash — unchanged from v1's authoring economics, now with
  the varied slot visible in the record ([AR-1]).

### 5.2 Capability contracts on bindings

- **[EV2-5]** A binding's `requires` map declares what the slot needs from its neighbors and the
  plan: state access (checked against grants, [PIR-14]), optimizer invariants (e.g.
  `optimizer_invariant = "row_constant_second_moment"`), gradient assumptions, determinism
  capability (`det_capable = true` for slots the plan wires into `det` channels), and its
  `channel_class` for compress slots. A binding's `provides` map declares what it guarantees.
  The key/value registry lives in the module ABI spec and grows additively.
- **[EV2-6]** The host performs **wiring-time conflict detection at admission, before any
  execution** (§13): every `requires` satisfied by a `provides`, a grant, or a host guarantee;
  every conflict a typed refusal naming both parties.

### 5.3 `[plan]` inputs and coordination parameters

- **[EV2-7]** The envelope carries a `[plan]` section: the author-chosen inputs to
  `vhc_plan_derive` — sync period H, fragment counts, replication targets, group-count hints,
  compression budgets — plus the coordination parameters the coordinator needs before any plan
  exists (phase timeouts, epoch length, retention windows: v1 `[phases]` semantics carried
  forward, v1 spec §6.1). `[plan]` contents are *inputs* to derivation, never the plan itself
  ([PD-4]).
- **[EV2-8]** Read against the v1 seam rule: everything the v1 envelope gave the coordinator and
  transport keeps its home (`[run]`, `[artifacts]`, `[data]`, requirement floors); what changes
  is that experiment hyperparameters split per slot ([EV2-3]) and the cadence block generalizes
  into `[plan]` inputs consumed by derivation. v1's module-derived-then-verified convention for
  cadence (v1 spec §6.1, `steps_per_round`) generalizes to the whole plan: authored inputs,
  derived schedule, peers verify the derivation ([PD-3]).

### 5.4 Authoring invariants

*Ratified 2026-07-26 (normative amendment A2, the derivation invariant). Cross-references: §4.4
(plan lifecycle), §6.3 (admission).*

An envelope is **derived**, never assembled by hand, and the derivation has exactly one
implementation. Four times in this program's history a value was authored in one place, re-derived
in another, and the two disagreed — a corpus key, an artifact URL, a role's resource requirement, a
claim figure. Each was fixed as an incident. These rules make the class unrepresentable instead.

- **[DI-1]** Every value in a genesis envelope MUST be **derived from the run's configuration** by
  the authoring pipeline, or be an input a human supplies **exactly once** and the pipeline then
  propagates. No value is stated twice in two places, in any form, including "for clarity".
- **[DI-2]** Negative-path and conformance fixtures are **exempt** from [DI-1] and MUST say so at
  their construction site: a fixture whose purpose is to be malformed cannot be derived from a
  configuration that would make it well-formed.
- **[DI-3]** Where a value crosses a language boundary and cannot share the implementation, the
  contract takes one of exactly two forms, and the change MUST say which: **form 1**, one
  implementation both sides link; **form 2**, canonical conformance vectors both sides are gated
  against. A second hand-maintained implementation is neither form and is refused.
- **[DI-6]** A role's **execution requirements** are carried on the genesis **schema major**, not
  added compatibly to an older one: a reader that does not understand them MUST refuse the envelope
  rather than silently ignore a section that changes what the run costs. The envelope's outer
  framing carries the schema major, the required reader features, a bounded payload length and the
  payload digest, so **refusal precedes payload decoding**.
- **[DI-7]** Every shipping authoring path runs the same pipeline. A convenience builder that
  drives a whole run without passing through it is a second authority and MUST NOT exist; the live
  path is a **parameterization** of the ceremony path, not a sibling of it.
- **[DI-9]** The admitted tuple records the identity of everything the admission decided, and a
  renamed member is a **rename, never a repurpose**: `claim_hash` becoming
  `logical_resource_plan_hash` means historical `claim_hash` evidence keeps its old meaning and is
  never reinterpreted under the new one.
- **[DI-10]** The composition planner has **one implementation and four callers** — authoring
  validation, node admission, sealed-binary conformance, and ceremony preflight. A caller that
  cannot link it is bound by [DI-3] form 2 instead.

**What this forbids in practice.** Authoring MUST NOT contain a physical figure for a role: the
authored artifact carries the role's **module-derived** requirements, and the Physical Estimate is
composed at admission from those requirements plus the node's own profile and capability report
(§9.6). An authoring seat therefore has no constructor for stating a requirement, and no path by
which an operator's figure becomes one.

---

## 6. Groups, rounds, epochs, and fleet snapshots

### 6.1 Group-scoped rounds

The v1 round protocol (v1 spec §6.4: the seven signed messages `RoundOpen`, `Commitment`,
`Attestation`, `StorageReceipt`, `RoundRecord`, `Digest`, `Straggle`; committed sets as merkle
roots; a pure commit rule; invariants I1–I6) is **instantiated per group**. Time becomes
`(group_id, group_round)` plus the causal chain (§8); no global scalar step exists (VP-12).

- **[GR-1]** The coordinator maintains one logical clock per group. Every round-protocol message
  carries `(group_id, group_round)` plus the version tuple (§8); the commit rule is evaluated per
  group over that group's signed evidence, unchanged in shape.
- **[GR-2]** Groups advance independently: a straggling group never blocks another group's
  commit. Cross-group coupling exists only where the plan declares an inter-group channel, and
  then only on the data plane ([PL-1]).
- **[GR-3]** All v1 invariants hold per group: replayability (I1), the ingest barrier (I2, as
  scoped by [PIR-18]), exact committed sets ordered by node public-key bytes (I3), deadline
  liveness (I4), coordinator blindness (I5 — hashes, sizes, roots, receipts; never payload
  bytes), signed evidence only (I6).
- **[GR-4]** Record entries gain a per-contribution declared sample count `n` (additive), so
  contribution weighting is auditable guest math ([CO-4]).
- **[GR-5]** The coordinator consumes group ids, opaque role labels, replication floors,
  requirements, and channel budgets — nothing else from the plan (VP-3). It never fetches or
  executes modules (I5 carried forward).

### 6.2 Membership epochs and committed fleet snapshots

- **[GR-6]** A **membership epoch** is a span during which every group's roster is frozen; staged
  membership changes materialize only at its boundaries. `member_epoch` (u64, per run) increments
  at each boundary. Joins and leaves stage as pending until then — v1 §6.2 semantics, now also
  covering group (re)assignment.
- **[GR-7]** At each boundary the coordinator publishes the **committed fleet snapshot**: a
  signed canonical-CBOR document listing members, leases, prior group assignments, and their
  **quantized** measured eligibility (probe verdicts, throughput classes, memory budgets —
  [PD-2] rules). The snapshot is an input to plan derivation and is content-addressed into the
  epoch record, so replay can re-derive the plan — I1 extends to epochs.
- **[GR-8]** Epoch records chain: `(member_epoch, snapshot hash, plan_digest, per-group base
  rounds)`, coordinator-signed. A joiner entering at epoch E needs the epoch-E record, the
  snapshot, the envelope, and the checkpoint set — nothing older, unless it replays.

### 6.3 Admission, leases, availability evidence

- **[GR-9]** Admission stays two-sided (v1 spec §6.5): the coordinator authenticates and enforces
  floors; the peer self-assesses. The assess step now evaluates the **per-role memory
  inequality** against each group's `memory` claim ([PIR-5], §10) and reports which roles the
  peer could hold; assignment is the coordinator's choice among eligible peers ([PIR-4] targets,
  throughput weighting).
- **[GR-10]** A group slot is held under a **lease**: coordinator-granted, heartbeat-renewed,
  epoch-scoped. Lease expiry (missed heartbeats, repeated failed commits) marks the slot
  reassignable at the next boundary, the member quarantined-not-penalized meanwhile ([AL-2]).
  Leases generalize v1's roster entry and drop rule into the one revocation mechanism
  multi-group assignment needs.
- **[GR-11]** **Availability evidence generalizes beyond one store.** v1's commit rule accepts a
  signed `StorageReceipt` (coordinator-as-storage-client HEAD) or a witness-quorum `Attestation`
  (v1 spec §6.4). v2 keeps exactly this shape per group and treats each as one *evidence class*
  among potentially several signed classes (store receipts, witness quorums; future third-party
  availability attestations). The commit rule remains a pure function of signed messages — which
  is what keeps `tick` re-executable and auditable. Witness committees are drawn per group per
  round from `(round seed, group roster)` (v1 spec §6.3).

### 6.3.1 Certified per-run identity

Admission binds an execution to a cryptographic identity that is scoped to the run, not to the
machine. A peer does not sign plane traffic with its long-lived key; it signs with a per-run key
that a **base identity** certifies. This keeps the machine's durable key off the wire, makes a
compromised run key expirable without touching the machine, and lets a verifier authorize a frame
purely from the frozen genesis plus the certificate the frame carries.

- **[CI-1]** **Per-run keys are CSPRNG-minted and base-certified.** At admission each production
  role mints a fresh signing key from a CSPRNG — never a key derived from the run label, run id, or
  any other predictable material. A **base identity** (a long-lived machine key) issues a
  certificate vouching that this per-run key speaks for the role. Run-label-derived keys are not a
  permitted fallback: the mint path is the only path.
- **[CI-2]** **The certificate binds the full execution identity.** A run-key certificate binds the
  tuple `(run/genesis hash, epoch, role, incarnation, module hash)`. A verifier authorizes an
  inbound frame only when the sender's certificate binds exactly the identity the verifier expects
  for that frame; any bound-field mismatch refuses. The certificate is therefore a statement about
  *which execution* a key may speak for, not merely *that* a key is signed.
- **[CI-3]** **Trust roots are genesis/Authority, never ambient config.** The base identities a
  verifier trusts are named in the frozen genesis (the Authority topology, §7/§10). A per-run key is
  trusted only through a certificate that chains to one of those base identities. No host
  configuration, environment, or discovery path may introduce a trusted base identity out of band —
  a base identity absent from genesis cannot admit anyone.
- **[CI-4]** **Revocation is signed, sequenced, and replay-protected.** A run key is revoked by a
  signed revocation record carrying a strictly monotonic per-`(run, role)` sequence number; a
  verifier rejects any record whose sequence does not advance, so a captured record cannot be
  replayed to un-revoke or re-revoke. Records propagate on the control plane best-effort — delivery
  is not assumed synchronous or total.
- **[CI-5]** **Incarnation supersession is the partition-safe floor — per base identity.** Because
  revocation delivery is best-effort, correctness does not depend on it: a higher incarnation
  supersedes a lower one within the same `(run, role, base identity)` ladder, so a certificate for
  a stale incarnation is refused even where no revocation record has arrived. The base identity is
  load-bearing: a role names a duty, not a seat — a run whose roster carries two trainers has two
  independent trainer ladders, one per base, and each base may only supersede its OWN keys.
  Cross-base ordering exists in exactly one place, the coordinator seat's leadership term (ABI
  §12.4 [SEAT-1]), which is a separate ordinal fed only by verified seat grants — never by
  certificate incarnations. Supersession is the floor that holds under partition; revocation is
  the timely-but-best-effort layer above it.
- **[CI-6]** **Epoch rebinds, incarnation rotates.** An epoch change rebinds the *same* per-run key
  under a new certificate (the key is stable across epochs of one incarnation). Key rotation happens
  only on an incarnation change — a new incarnation mints a new key and obtains a new certificate.
  Epoch and incarnation are thus separated: one re-certifies, the other re-keys.
- **[CI-7]** **Key material and certificates are deleted on terminal completion.** When a run
  reaches a terminal state, the node deletes the per-run key material and its certificates; no run
  identity outlives the run it was minted for (crash-safe lifecycle, §10.3-class persistence).
- **[CI-8]** **Certificate verification is mandatory on every production path.** Inbound frame
  verification always performs a certified-sender check; there is no cert-optional verifier
  constructor on any production attach. The type that verifies frames cannot be built without a
  certificate-checking sender, so "authenticated but un-certified" is unrepresentable in production
  wiring.
- **[CI-9]** **Key custody never touches ordinary payloads or journals.** Per-run key material
  reaches a worker subprocess only by secret reference or an inherited protected descriptor — never
  embedded in an ordinary command payload, argument vector, or journal/log record. Journals and
  command channels carry references, not secrets.
- **[CI-10]** **Transport reachability is a signed, certificate-carried statement — the iroh
  roster record.** Each admitted node publishes to the registry one record per run binding its
  iroh endpoint id (the transport public key, a separate CSPRNG identity from every signing key)
  to its current direct addresses and/or relay URL, signed by its certified per-run key and
  carried with the certificate (the seat-lease distribution shape). Peers fetch the run's roster
  and verify every entry themselves — signature, certificate chain to a genesis-trusted base
  ([CI-3]), exact scope ([CI-2]) — before dialing an address. The registry stores records under a
  structural monotonic upsert only: untrusted storage that can withhold a record but never forge
  one, and never becomes a discovery authority.
- **[CI-11]** **Roster staleness is precedence, never wall clock.** A record's freshness key is
  `(incarnation, issued_at_ms)`, lexicographic: a rejoined node's higher incarnation supersedes
  every record of its prior incarnations ([CI-5] extended to reachability), and within one
  incarnation a later issue supersedes (the re-address republish). Readers group verified records
  by `(role, certificate base identity)` — the durable node key; the per-run key rotates with the
  incarnation — and keep only the freshest per group, so a registry serving stale state can delay
  but never roll back a reader that observed a newer record.

### 6.3.2 The admitted tuple

Admission's decision is captured as an immutable tuple that travels with the join. The peer that
was assessed and the node that persists the join both rederive that tuple from their own inputs and
compare it; a join is legitimate only when the two agree. This closes the gap between *what
assessment admitted* and *what the run actually executes* — a peer cannot be assessed under one
artifact, configuration, or grant and then join under another.

- **[AT-1]** **Assessment produces an immutable admitted tuple.** A successful assessment freezes
  the tuple `(module hash, config hash, grants hash, claim hash, genesis hash, role/incarnation,
  device-profile revision, owner-policy revision)`. The tuple is immutable once produced and is
  carried into the join intent; it is the canonical statement of the identity and terms under which
  the peer was admitted.
- **[AT-2]** **Join rederives and compares; mismatch is a typed refusal.** At join the tuple is
  rederived from the joining side's own inputs and compared field-by-field against the carried
  tuple. Any mismatch is a **typed refusal** that reruns assessment rather than proceeding — there
  is no silent-proceed or best-effort path. A join therefore executes only under a tuple both sides
  independently agree on.
- **[AT-3]** **The grants document is complete.** The grants hash in the tuple binds a *complete*
  grants document: every world the module links (compute and data), every channel, every artifact,
  every custom op, and the buffer limits and rates the module is authorized to use. Completeness is
  what makes the grants hash load-bearing — a join cannot widen the module's authority beyond what
  assessment admitted, because any additional world, channel, op, or raised limit changes the hash
  and trips [AT-2]. The same completeness argument covers a run's **genesis state contract**
  (the chunk-addressed canonical-state geometry + init pin): the seed-form init rides inside the
  genesis — hence inside the tuple's genesis hash — and the artifact-form init manifest and its
  family folds enter the granted artifact set exactly like corpus shards, so bulk initial state
  never needs to travel in the (hashed) role config to be admission-pinned.

### 6.4 The run-instance lifecycle at the node

The node is the single authority for a run instance's lifecycle, held as a durable **two-axis**
state machine: the owner-intent axis (`joined | paused | left`) records what the owner wants; the
observed axis (`running | completed | failed_retryable | failed_terminal | left`) records what the
last incarnation actually did. Clients render the six-state projection (`running | completed |
paused | failed_retryable | failed_terminal | left`; terminal observations win, a paused intent
masks recoverable states) and never re-derive it. Transitions on the observed axis are driven by
the worker's classified terminal events and by observed process/stream loss — never by inference
from silence.

- **[RL-1]** **Terminal handling is idempotent and generation-gated.** Every run-scoped worker
  event carries its instance's generation (the never-reused incarnation id); an event stamped with
  a generation other than the run's current one is discarded whole — a reaped instance's late
  events can never fold telemetry, transition state, release resources, or touch key custody for
  its replacement. Duplicate terminal delivery transitions nothing and cannot double-release.
- **[RL-2]** **Teardown is observed before the ledger releases.** A terminal transition follows a
  fixed, crash-repairable order: a durable release marker commits first (teardown observed — the
  terminal event arrived, or the worker's event stream closed / its process was reaped — with the
  terminal target recorded); only then does the instance leave supervision and its resource
  reservation release (a replacement is never admitted while the predecessor may hold devices);
  only then does the terminal state commit. A node crash inside that window is finished by the
  startup reconciliation pass — on a fresh start every child died with the node, so process
  absence makes teardown definitional and the recorded target simply commits.
- **[RL-3]** **A completed run never restarts.** The reconvergence set is standing `joined`
  intents whose observed state is non-terminal: a module-signaled run end and a terminal failure
  drop out permanently (the latter until explicit owner action). An owner rejoin of a settled run
  mints a fresh incarnation; identity retention across a node restart is legitimate exactly
  because no live predecessor can exist there.
- **[RL-4]** **Recoverable failures reconverge under a bounded budget.** A recoverable fault
  consumes one attempt of a config-bounded retry budget with exponential backoff; exhaustion
  escalates to the terminal failure with a typed reason. The budget resets only when an
  incarnation stays running past a configured minimum uptime — the coarse stability signal (the
  node never inspects rounds) — so a crash loop cannot launder its budget by restarting. Mid-run
  reconvergence always mints a new incarnation with freshly-authored credentials and certificate
  ([CI-1]/[CI-6]); the generation strictly advances, so the predecessor's stale events stay gated.
- **[RL-5]** **Pause is durable owner intent with release-on-pause.** Pause persists before it
  acts: a paused run survives node restart and is never reconverged until resumed. The pause
  lever is hard (memory, not just time), the run's resource reservation releases, and a held
  coordinator seat lease is released fenced (the floor persists). Resume re-admits against the
  owner's *current* ledgers before lifting the pause — a refusal is typed and loud, and the run
  stays paused with nothing half-claimed.
- **[RL-6]** **Coordinator-seat duty is resident, and fenced-out claimants never fight.** When
  the owner enables coordinator duty, a resident keeper covers each joined run whose admitted
  role is the seat role: it claims when a bid derives (standing by against a live incumbent),
  heartbeat-renews under the same fencing token, drops the lease when a renew is refused (the
  seat moved; supersession is the safety floor, [CI-5]-class), and releases signed on owner
  pause/leave and node shutdown so a successor takes over at floor + 1 without waiting out the
  lease TTL. The shutdown release is a bounded best-effort hook on the node's graceful-shutdown
  path (a hung registry must not stall shutdown; the TTL remains the safety net).
- **[RL-7]** **Module-upgrade records are consumed operator-driven and validated fail-closed.**
  A committed run-level module-upgrade record reaches the node through its product API (an
  operator submits the canonical record; idempotent via the operation id) — the node never acts
  on a record it did not validate at an operator's request. Validation is total: the frozen
  genesis is re-fetched and re-verified, the transition chain is rebuilt from genesis plus the
  node's durable mirror of previously-consumed records, and the presented record must append
  cleanly (authority threshold, hash link, strictly-monotone epoch, current-module binding); any
  failure is a typed refusal with every durable fact untouched. A validated record drives the
  live switch through the worker-control surface — target assessed where the module bytes live,
  a post-switch incarnation minted strictly above the running one, identity provisioned before
  the command — and the record mirror plus the advanced execution identity persist only on
  activation. A post-fence exit that leaves the run persists no advance: the run-level record
  stays committed; only this node's instance left.
- **[RL-8]** **Restore pointers are role- and kind-scoped, own-seat-first, with a periodic live
  cadence.** Published checkpoint pointers are keyed per `(role, kind)`: a joining instance
  restores only from its own role family's slots — a coordinator drain snapshot can never shadow
  a trainer restore source. Selection is **own-seat-first**: the seat's OWN pointers are
  preferred whenever any exists (freshest `live`, else `drain` — a drain snapshot exists only
  when an instance drains; a hard-crashed peer never drains), because a checkpoint document
  carries the producing seat's **replica-local (class-1) sections** — optimizer moments, error
  feedback — which are that seat's own training trajectory, and an own pointer's extra staleness
  is bridged by replay/catch-up, never by adopting foreign local state. A **sibling seat's**
  pointer in the same role family is a FALLBACK taken only when the seat itself has published
  nothing (e.g. alternating publisher election with a crash before the seat's first slot): the
  sibling's class-0 consensus-canonical sections are digest-identical by the deterministic-state
  contract, but the restore adopts the sibling's class-1 sections — consensus-safe, and always
  RECORDED (a persisted `sibling_restore_adopted` warning naming both seats and the round),
  never a silent equivalence. At restore the worker additionally fails closed on the document's
  state-manifest: an unreadable manifest header or a schema major this build has no defined
  restore semantics for is a typed refusal before the module ever sees the document
  (module-hash binding through the epoch transition chain is deferred work). The trainer exports
  its full restorable state on a configured ingested-round cadence as a live checkpoint, so a
  hard-crashed peer resumes from state that already folds the rounds it missed and its digests
  stay continuous with the survivors.
  The cadence separates **sealing from publishing**: sealing a checkpoint locally is cheap
  (state the host already holds) and may happen every cadence round, while **remote upload**
  obeys a byte budget — one deterministically designated publisher per cadence slot (derivable
  from roster + round, so every peer agrees without a message; R identical uploads per slot are
  waste), with a slot whose publisher died simply going unpublished and the next slot's rotation
  covering it. The remote cadence is bounded by payload retention (genesis-validated): a
  rejoiner replays forward from the freshest reachable checkpoint only across *retained*
  payloads, so the remote cadence plus one slot of publisher-churn slack must fit inside the
  payload-retention floor — a configuration that could strand a rejoiner past retention is
  refused at authoring, never discovered at rejoin.

  **Checkpoint carriage is by-reference (streaming det fold, [SF-6]).** A checkpoint document is
  `[manifest, sections]` where large families (`master`, `ef`, the AdamW moments) are carried
  **by reference** — a `{fold, byte_len, chunk_size, chunk_hashes}` descriptor of an
  already-sealed chunk-addressed family — and only small state (the round watermark) is inline. A
  live checkpoint therefore costs **zero extra local bytes**: the canonical families are already
  the round's sealed folds, so "sealing" the document is naming them. Restore is **streaming
  rehydration**: a rejoiner resolves its `(role, live)` pointer, fetches the small document,
  registers the referenced family folds as externally-sourced roots, and streams their windows on
  demand — reconstructing device weights and optimizer moments with guest memory bounded at
  O(windows in flight), never materializing a whole family. The published family chunks ride the
  content-addressed payload plane content-addressed, so a chunk unchanged since a prior slot
  uploads nothing; and consensus continuity holds because a restored instance folds forward from
  the checkpoint's canonical master exactly as its survivors did — the digest values are
  preserved bit-for-bit across the restore, not merely close.
  The one carriage that does **not** touch the content plane is a **live module switch** (§6.4):
  an in-process migrate on one node, not a rejoin. The switch transaction carries the draining
  instance's sealed families directly into the successor instance's state store and the successor
  inherits the run-pinned `state_chunk_size`, so it serves those folds **self-sealed** — the host
  keeps custody of canonical state across the fence, publishing nothing to and fetching nothing
  from the payload plane. Local switch ≠ content-plane restore.
- **[RL-9]** **A stale restore fence is bridged by staged archive catch-up, and the archive tip
  is paced to stay within reach.** A rejoiner's fence may trail the live head past the
  coordinator's bounded in-memory replay ring (`RETAINED_RECORD_HORIZON_ROUNDS` — authoring-time
  *sizing* for the ordinary rejoin, not the recoverability guarantee). When it does, the node
  compares the fence against the seat lineage's **verified** archive tip (the latest signed
  committed-round claim, certificate-chained to genesis trust — never registry metadata): if the
  tip reaches within a ring of the head, the join proceeds carrying a **catch-up directive**
  (the verified head records + the fence) in its internal session credentials — additive,
  node↔worker only, no consensus-wire change. The worker re-verifies the lineage, fetches the
  attested segments (local file when hash-matched, else the content plane), extracts the
  coordinator's historical round records from the archived publish stream, and the restored
  guest folds them **staged, before live attach** — authenticity is the verified lineage, so the
  historical frames deliberately bypass per-frame certificate liveness (a superseded incarnation's
  records would otherwise refuse `CertRevoked`); payload bytes fetch from the content plane as in
  live operation; the ordinary ring replay covers the unarchived tail and the dedup window
  absorbs the overlap. The **overlap invariant** is host-enforced by round-aware seal pacing: the
  session watches the committed-round watermark against the acknowledged archive tip and requests
  a journal recovery point (a segment seal at the next append — never churning an empty segment)
  once the lag crosses half the ring, so the archived stream and the ring always overlap under
  healthy publication. A gap that still forms (a publication outage outlasting the ring — the
  budget-free transport deferral keeps publication retrying) is a **typed refusal**
  (`CheckpointStale`, naming the fence/head/horizon shape), never a silent wedge.

### 6.5 Node-local role composition

*Ratified 2026-07-26 (normative amendment A3, node-local role composition). Extends [RL-1]/[RL-2]
above; cross-references [CI-10]/[CI-11] (the reachability record and roster fold) and [PC-8]
(timeout classes).*

One host may run more than one role instance, and the rules below are what the implementation
already does — written down so that a change to them is a decision rather than a drift.

- **[NC-1]** Admission is **per role instance**, and the node's occupancy delta is reserved
  **atomically**: two instances admitted concurrently on one device MUST NOT each pass a check the
  pair would fail. One sandbox is one role instance; there is no shared-instance mode.
- **[NC-2]** A role instance's identity includes its **incarnation**, and incarnations are never
  reused on a host. A re-admission after a release is a new incarnation, not a resumption of the
  old reservation's identity.
- **[NC-3]** A colocated pair is one peer to the roster fold: the reachability record and roster
  rules see the **node**, not each of its instances, so a two-instance host does not vote twice or
  count twice toward a quorum.
- **[NC-4]** A **non-accelerator** role charges **zero accelerator duty**. The consensus seat runs
  host-side and performs no training compute, so charging it duty exhausts the ledger and refuses
  the trainer it is colocated with — which is a resource decision nobody made.
- **[NC-5]** A failure **before any guest code runs** is a host-side bring-up or admission refusal
  with its own stage, recorded **outside** the guest-trap surface. Attributing it to the guest's
  initialization phase states a fact about a phase the guest never entered.
- **[NC-7]** Scoped terms are charged **once at their scope**: a per-process or per-device term is
  charged once per node, not once per instance, and a per-allocation term is a maximum to validate
  rather than occupancy to hold. Summing per-role is the double count this rule exists to name.
- **[NC-8]** A release returns the reservation only when the instance's **teardown is observed**,
  not when the leave is requested — otherwise a second admission can be granted against memory the
  first is still holding.

---

## 7. The consensus and aggregation seam

### 7.1 The split: mechanics host-side, math guest-side

The host does the **staging**: fetch and verify the committed set (exact, root-checked, ordered
by node public-key bytes), expose it to the guest with per-entry metadata, snapshot round bases,
and manage residual transactions. The guest's `outer@1` consumes the staging via det ops and
computes the update. Its semantic contract (the concrete `vhc_outer_apply` export shape belongs
to the module ABI spec):

```
( θ (round base),
  staged pseudo-gradients (committed set, ordered),
  staleness estimates (realized version gaps, monitor signals),
  quorum metadata { n_i, member_epoch, absence set } )
    → θ′ + residual updates
```

This one contract is the outer-optimizer ⊕ staleness-corrector slot (VP-5): Nesterov look-ahead,
EMA delay correction, α-mixing, trimmed/weighted/sign-family aggregation are all points inside
it. Staleness arrives measured, from the version plane ([VT-3]) and the monitor (§12) — never
from a side channel.

### 7.2 Consensus-operator requirements (adopted)

The VHC ref §7.4 defines eight requirements, C-1..C-8, for any replica-consensus operator. v2
adopts them as the **normative contract for the (host staging + `outer@1`) pair**: the host
provides the mechanics that make each satisfiable; the module's `outer@1` + `compress.<name>`
slots must satisfy the math. Ids are stable for cross-reference.

| Req | v2 normative reading | Host provides | Guest must satisfy |
|---|---|---|---|
| **[CO-1]** partial quorum | The committed set is whatever signed evidence admitted by deadline (I4); any set size ≥ the group floor is a *normal* round. | deadline-frozen exact sets | `outer@1` defined for every committed-set size, including empty |
| **[CO-2]** exactly-once residuals | Residual state MUST commit or roll back exactly once per round. | §7.3 transactions over `residual` states | route all error feedback through declared `residual` states |
| **[CO-3]** late-message path | A contribution absent from record r has a defined route into round r+k, never double-counted. | stall ladder + retention windows (v1 spec §6.4) | residual semantics: unsent information persists locally and retries |
| **[CO-4]** contribution weighting | Committed entries carry declared sample counts n ([GR-4]); weighting is guest math. | n in staging metadata; assignment makes n auditable | weight by n where the math requires it |
| **[CO-5]** join/leave semantics | Replacement policy per state class: `local` rebuilt; `replicated` from checkpoint, bit-exact; `residual` handed off transactionally (§7.3 point 5) or zero-initialized **as a declared perturbation** in the record. | state classes, checkpoint/resync, handoff | declare which perturbations its math tolerates (§13) |
| **[CO-6]** sublinear aggregate cost | Staging MUST NOT preclude hierarchical/tree aggregation (aggregation math that is linear before its nonlinearity commutes with `det_sum`). **v2.0 does not satisfy C-6**: the degenerate plan is all-download-all, honest at small rosters (v1 spec §18); the seam is shaped so the fix is staging-side. | a staging interface that admits partial aggregates as inputs (design constraint now, implementation later) | keep aggregation linear-before-nonlinearity where sublinear staging is wanted |
| **[CO-7]** adaptive schedule | Cadences, fragment phases, and compression budgets are plan outputs; adaptivity = trigger events on monitor signals ([PIR-17]) + re-derivation at epoch boundaries — never guest-side runtime scheduling. | trigger evaluation at round boundaries; epoch re-derivation | express adaptivity declaratively in `plan@1` ([SL-3]) |
| **[CO-8]** contraction | Inter-replica disagreement MUST contract: E[D_{t+1}] ≤ ρ·E[D_t] + σ with ρ < 1, **measured** via monitor drift signals (§12), never assumed. For the degenerate plan this is trivially the digest-equality gate (D = 0 exactly). For any plan whose consensus math leaves D > 0 by design, measured ρ̂ is an acceptance gate before scale. | monitor signals + observe tooling | choose consensus math with contraction evidence |

### 7.3 The transactional residual protocol

*(Adopts VHC ref §9.4 into the round protocol; realizes [CO-2] for the `residual` state class
[PIR-13].)*

The host maintains, per `residual` state, a committed copy and a working copy. Commit points — a
round either commits or rolls back residual state exactly once at each:

1. after local delta extraction (the compressor consumed the residual), before payload upload;
2. after upload, before the round record admits the payload;
3. after record admission, before the ingest's residual subtraction;
4. after subtraction, before the next checkpointable event covers it;
5. during any state handoff — join, group migration (§4.4 step 5), resync, warm-spare promotion.

- **[RES-1]** The working residual becomes committed only when the full round path (extract →
  upload → record → ingest) succeeds for this peer; any failure or abort reloads the committed
  copy. A peer absent from the record ([CO-3]) keeps its pre-round residual — information is
  retried, never dropped or double-subtracted.
- **[RES-2]** Checkpoints include `residual` state (unlike `local`); a resync that replays rounds
  MUST replay residual transactions with them, or declare the reset as a [CO-5] perturbation in
  the rejoin record.
- **[RES-3]** The churn drill for any plan using `residual` states injects failure at each of the
  five points and asserts exactly-once behavior — an extension of the existing drills, gated like
  [LC-1].

### 7.4 Records and committed sets (carried)

Committed sets remain exact sets committed by merkle root, ordered by node public-key bytes, with
scale-invariant signed roots and content-addressed record-set objects (v1 spec §6.4). v2 scopes
them by `(group_id, group_round)`, extends entries with n ([GR-4]), and writes records at
`commit` schedule events — the plan decides *when*, the host decides *what a record is*. Digest
coverage follows state classes: params + `replicated` + `residual` (committed copies) are
digested; `local` never is (v1 spec §5.6 machinery, per group).

Because full coverage is a sequential fold over the state image, the digest is computable as a
**streaming carry** (seeded streaming hasher + absolute byte offset, block-index frames injected
at each block boundary independent of update splits): the carry reproduces the identical value
bit-for-bit for any chunking of the state, so canonical state need never be materialized to be
digested. Coverage and formula are unchanged — existing pinned digests remain valid and are the
parity evidence for any streaming refactor. Alongside the comparison digest, chunk-addressed
canonical state defines a **per-round det-state manifest** (per-family blake3 chunk folds under
a dedicated derivation domain, canonical CBOR) whose hash is the **round state root**: a
collision-resistant, audit-grade agreement object every peer derives identically from its own
sealed chunks — a derivable object, not a wire message; it rides checkpoint documents, and
promotion to an explicit consensus voice is a separately-ratified later change.

Canonical chunk custody is **host-side**: the per-instance state store holds content-addressed
chunks the guest writes through the ABI's three-state-import stream (open a family stream, emit
verified chunks — the host hashes at the write, so bytes never leave host custody between emit
and fetch — seal to the family fold), and a self-sealed fold is fetchable by construction over
the ordinary content-addressed read path. Only the seal mints a durable artifact: an
opened-but-unsealed stream's chunks are garbage-collected at instance teardown and the store is
instance-scoped, so a crash mid-fold leaves nothing observable (the torn-fold rule). Retention
is grant-declared (`state_retain_roots` per family, plus checkpoint-pinned folds and the init
artifact), evicted oldest-first with chunks refcounted — content addressing dedups identical
chunks across rounds structurally. Sealing journals a normal-size cross-check record (the
journal stays O(records)); tier-1 replay re-executes emits over replay-reproduced guest memory
into a replay-side chunk store and cross-checks each seal's recorded fold — fold divergence is
detected at the seal in O(1), and no bulk state is ever archived.

---

## 8. The version plane and the message DAG

*(Adopts VHC ref §9.1–9.2. v1 already refuses a global step in its checkpoint design and keys
everything by round; v2 makes the tuple explicit and per-message.)*

- **[VT-1]** Every control-plane message, consensus payload commitment, record, digest, and audit
  artifact carries the version tuple

```
{ run_id, group_id, group_round, member_epoch,
  plan_digest, weight_version, composition_digest, schema_versions }
```

  plus `parent: blake3` — the hash of the causally preceding message in the sender's stream —
  forming the per-run message DAG (messages remain signed as in v1). `weight_version` is the
  group round whose commit produced the state the message was computed against.
  `composition_digest` binds *what code*: blake3 over the envelope's slot bindings and the module
  manifest digest ([AR-1]). `schema_versions` pins `{ proto, tabi, envelope_schema,
  vocab_version }`.
- **[VT-2]** Receivers MUST reject messages whose tuple they cannot resolve — unknown plan
  digest, future epoch, unresolvable weight version. An honest peer on a stale plan produces
  *unusable*, not merely noisy, contributions; tuple rejection is the firewall between error
  channels (VHC ref SV-1; VP-8). The VHC ref's `basis_version` needs no dedicated field here:
  shared transforms are `replicated` states ([PIR-12]) whose version *is* the committing round,
  covered by `weight_version`.
- **[VT-3]** Realized staleness is a measured input: a contribution's staleness is the gap
  between its `weight_version` and the consuming round, read off the tuple. This feeds
  `outer@1`'s staleness estimates (§7.1) — delay is consumed as data, not assumed (VHC ref EX-5).
- **[VT-4]** Mapping from v1: messages carry `(run_id, round)` and an implicit epoch today; v2
  adds `group_id`, `member_epoch`, `plan_digest`, `composition_digest`, and the explicit
  `parent`. v1 records already hash-chain through the next-round seed; `parent` makes the DAG
  explicit per message so replay, audit, and any future reward accounting key off one structure.
  Attestations and storage receipts slot in unchanged — they attest set membership and
  availability *at a tuple*, and the tuple grew.

---

## 9. The execution-backend contract

*(This section normativizes the P3 CUDA program's findings and generalizes them into the backend
seam — the execution half of fleet heterogeneity. Sources: P3 ledger Merge-1 frozen seams (the
D5 fat-worker and D6 NVRTC decisions, the `select_backend()` ladder, the CUDA `DeviceLimits`
source); the P3 Merge-2 threading and memory findings, whose code is on the P3 trunk
(`daemon-train/src/bin/daemon-train-worker/live.rs`, the pinned device thread) and whose ledger
entries land with that merge — cited here so the contract does not wait on the paperwork;
platform budget sources: [`swarm-uma-platform-findings.md`](swarm-uma-platform-findings.md),
[`swarm-windows-vram-design.md`](swarm-windows-vram-design.md),
[`swarm-macos-uma-findings.md`](swarm-macos-uma-findings.md).)*

### 9.1 Thread and stream discipline

**Finding (P3 Merge-2).** cubecl-cuda derives its CUDA stream and memory-pool registry from the
*calling thread* (`cubecl_common::StreamId::current()` is a thread-local); backends are `Send`
but **not `Sync`**. Driving a backend from multiple OS threads — which is exactly what happens
when an async task holding it migrates across executor threads at `.await` points — silently
splits pool bookkeeping across per-thread streams. At low memory pressure the split pools merely
waste VRAM; under pressure (160M-scale working sets with paging active) a handle allocated under
one thread's stream is looked up under another's and the worker dies inside cubecl (allocation /
"memory page" panics). Single-threaded drivers at the same scale are green, and low-pressure
multi-threaded drivers are green — the failure needs migration *and* pressure jointly, which is
why it survived every small-model gate.

- **[EB-1]** A non-`Sync` GPU backend MUST be owned by one dedicated device thread for the
  process lifetime: constructed on that thread, every call serialized onto it over a channel with
  per-call completion, callers blocking on the reply — the **pinned-device-thread pattern**
  (`DeviceThread`/`BackendHost::Pinned` on the P3 trunk, mirroring `daemon-infer`'s GPU-worker
  discipline). Call semantics are byte-identical to in-place calls; only the executing thread is
  fixed.
- **[EB-2]** The engine MUST NOT assume backends are `Sync`, and MUST NOT rely on "we only poll
  from one thread in practice" — async executors guarantee nothing at `.await` points. A backend
  that is internally synchronized MAY use direct mutex-guarded ownership where that is its proven
  configuration (wgpu today — do not churn proven lanes), but the portable contract is [EB-1].
- **[EB-3]** Device-thread failure is a typed, recoverable worker error surfacing through the
  existing respawn ladder — never a silent hang. The call channel's disconnect is the detection
  point.

### 9.2 Memory-pool behavior and headroom

**Finding (P3 Merge-2).** cubecl memory pools never shrink: the peak working set is permanent for
the process. At 160M-live scale the pool runs near device capacity on 24 GB-class cards
(compression intermediates and ingest staging on top of weights/grads/moments), and **checkpoint
staging is the peak allocation**: the device-readback staging buffer (order 1.5 GiB contiguous at
160M) lands on a pool already at capacity — the observed first-failure site was the first
checkpoint round, and disabling checkpointing isolated the trigger.

- **[EB-4]** The autotune/eligibility verdict MUST reserve **checkpoint-staging headroom**: the
  inequality a peer proves at admission includes the largest contiguous staging allocation the
  plan's `checkpoint` events imply (params + `replicated` + `residual` readback for its role's
  fragment share), not just steady-state training residency. `plan@1`'s memory pass MUST author
  the same term into `memory` claims ([PIR-5]) — claim and verdict measure the same quantity.
- **[EB-5]** Allocation failure on a full pool MUST surface as a recoverable typed error — churn
  the instance, retry smaller, or decline the role — never an abort. Today the failure path
  panics inside cubecl; that is an upstream defect tracked as such, [EB-4]'s headroom is the
  operative defense, and the worker respawn ladder is the containment.
- **[EB-6]** Never-shrink is a scheduling input: a worker that has run a large plan MUST NOT be
  assumed to have reclaimed VRAM for a later verdict without a process restart. The governor
  restarts workers between runs; keep it that way.

**Relationship to §9.6 (the residency contract).** [EB-4] and [EB-6] are the device-side siblings of
two terms in the composed estimate: [EB-4]'s checkpoint-staging headroom is [RC-4]'s staging term, and
[EB-6]'s never-shrink pool is its retained-pool term. Under §9.6 those terms are supplied by the
backend's **certified profile** rather than authored into a guest's claim — the quantity is the same
one, and the change is which authority states it.

### 9.3 The selection ladder and device limits

- **[EB-7]** One runtime probe ladder, fixed order **cuda → wgpu/Vulkan → CPU** (`select_backend()`,
  P3 ledger Merge-1): each rung requires its feature compiled *and* a passing runtime probe;
  failure falls through. A cuda-featured worker on a non-NVIDIA host MUST degrade cleanly — no
  panic, no link-time CUDA dependency (dlopen only; the Merge-1 hard gates: clean
  `probe_cuda() → None`, ELF `DT_NEEDED` free of `libcuda`/`libnvrtc`).
  `DAEMON_TRAIN_BACKEND=cpu` remains the operator escape hatch.
- **[EB-8]** The det lane is host fp32 on **every** rung; backend choice affects only the native
  (tolerance-class) lane. The standing tripwire — cpu-vs-GPU det digests byte-identical per
  round, native payloads tolerance-class — MUST stay in the gate set: it is what makes [VP-8]'s
  per-channel determinism real on a heterogeneous fleet (P3 ledger Merge-1, adjudication 1).
- **[EB-9]** Memory budgets come from the `DeviceLimits` machinery, per platform, and verdicts
  MUST use the platform-correct source, never a folded "dedicated + shared" figure: Linux
  discrete = driver total VRAM; Linux UMA = `vram_mb` (carve-out) + `shared_mb` (GTT) with
  **effective budget = vram + 0.9·shared** and the **joint pool check** (device + host draw one
  DRAM pool) per `swarm-uma-platform-findings.md`; Windows = DXGI static sizes +
  `QueryVideoMemoryInfo` dynamic budget + the D3D12 UMA flag as authoritative, discrete
  NON_LOCAL contributing 0, per `swarm-windows-vram-design.md`; macOS =
  `recommendedMaxWorkingSetSize` with the joint pool check, per `swarm-macos-uma-findings.md`;
  CUDA discrete = `cuDeviceTotalMem`, `shared_mb = 0`, `unified = false` (P3 ledger, Lane G).

### 9.4 Packaging

- **[EB-10]** **One fat worker binary** (P3 ledger D5): ndarray + wgpu + cuda feature-unioned,
  cuda target-gated to x86_64 Linux/Windows at packaging time, the runtime ladder selecting.
  No per-backend binaries and no dynamic backend plugins — the worker's final linkage shape must
  still permit `libloading` dlopen of the system driver, which a plugin ABI would multiply across
  every backend × platform × linkage combination (recorded packaging follow-on: verify the
  linkage shape when bundling lands).
- **[EB-11]** **NVRTC/CUDA runtime libraries are fetch-on-demand assets** (P3 ledger D6): keyed
  by the detected driver version, distributed like model assets (content-addressed,
  hash-verified), staged into `DAEMON_CUDA_RUNTIME_DIR` with the complete cudart JIT include
  tree; readiness is the two-leg gate (loadable `libnvrtc` AND complete include tree). Until
  staged, the probe **downgrades to the Vulkan rung**: CUDA is an upgrade the fleet applies
  itself, never a precondition.
- **[EB-12]** The shipped distribution MUST NOT depend on Nix at runtime: dlopen targets are the
  end-user system's standard paths (Windows `System32\nvcuda.dll`, Linux ldconfig
  `libcuda.so.1`). devShells and staged runtime dirs are developer/CI mechanisms, not the shipped
  contract.

### 9.5 Backends close the loop on fleet heterogeneity

The backend contract and the autotune verdict are what turn PlanIR `memory` claims into
admissible assignments: `plan@1`'s memory pass authors the claim ([PIR-5], [EB-4]); the probe
ladder + `DeviceLimits` produce the peer's measured budget ([EB-7], [EB-9]); admission is the
per-role inequality between them (§10). Heterogeneity splits into a declarative half (the plan
claims what a role costs) and a measured half (the peer proves what it has) — neither side trusts
the other's arithmetic.

### 9.6 The guest residency contract

*Ratified 2026-07-26 (normative amendment A1, the guest residency contract). The device-side
siblings of two of its terms are already in §9.2: [EB-4] is the checkpoint-staging headroom of
[RC-4] term 8, and [EB-6] is the never-shrink pool behavior of [RC-4] term 4.*

A module's memory footprint used to be one number the guest stated. It is now **three artifacts with
three owners and three independent lifecycles**, because the single number required the guest to
know things only the host can know, and a guest that guesses at device physics is a guest that
re-pins every time a driver moves.

| Artifact | Owner | States |
|---|---|---|
| **Logical Resource Plan** | the guest | what the algorithm needs, in logical units |
| **Backend Execution Profile** | the host backend implementation | what it physically costs to deliver that |
| **Device Capability Report** | the participating node | what the machine actually has |

- **[RC-1]** The plan is **backend-neutral and parametric**: shapes, dtypes, logical byte sizes,
  lifetimes, and bounded choice sets. It MUST NOT contain a physical constant, a backend name, an
  allocator term, or a measurement. Naming one in any identifier is a refusal, not a warning.
- **[RC-3]** Peak arithmetic is **persistent floor + the maximal concurrently-live transient set +
  fragmentation headroom**, not a sum over everything declared. The maximal-set form is the
  correction: summing transients over-states a peak no execution reaches.
- **[RC-4]** `PhysicalEstimate = compose(plan, authenticated certified profile)` — **one estimate
  per backend, never a maximum across backends**. Every profile term declares its **allocation
  scope** (`PerAllocation` / `PerRoleInstance` / `PerProcess` / `PerDevice`), a stable aggregation
  key and an associative composition rule; unknown sharing takes the conservative non-sharing rule.
  The Device Capability Report is **measured on the participating node** and is a statement of
  **supply**, never of demand, and never of instantaneous free memory.
- **[RC-6]** A divergence between estimated and observed residency MUST name a **root authority and
  the contributing authorities** among plan, profile, planner, probe and governor. "The memory was
  wrong" is not an attribution.
- **[RC-10]** The governor **intercepts and attributes** what it can, **pre-authorizes from the
  profile's derived worst case** what it cannot, and records the two enforcement classes
  **distinctly** in every estimate, conformance record and certification statement. A statement that
  reported one property over both would be asserting something about a driver's internals that
  nobody verified.
- **[RC-11]** The host composes the plan at a fixed binding and delivers **logical values only** as
  an immutable **Execution Grant**. A `UniformRun` grant is frozen in the signed role entry and
  every participant consumes those exact bytes and **verifies rather than reselects**;
  `PerParticipant` requires the module's stated normalization contract and peer-visible evidence.
  There is no estimate-driven geometry search on the admission path: which binding a grant carries
  is decided by fit-verdict evidence ([RC-15]), not by ascending the choice set against a
  composed figure.
- **[RC-13]** There is **one memory reservation authority**: the composed estimate and its
  node/device aggregates. Non-memory ledgers — duty, disk, bandwidth, instance ceiling — keep their
  existing grant and policy derivations and are never estimate-derived. The owner's cap **authorizes
  and never pays**: it is not a substitute for a figure that was supposed to be derived.
- **[RC-14]** The profiled hidden-overhead reserve is **visible and counted once**, and a
  ledger-versus-governor comparison is about **reservation identity and bounds**, not occupancy —
  measured usage below a reserved bound is normal and is not a divergence.
- **[RC-15]** **The composed figure is an estimate; the device is the oracle.** `compose()` output
  serves exactly two purposes: **cheap sound refusal** and **sizing the enforced budget** the
  governor holds the run to. Sound refusal keys on the bound that makes it sound: the estimate's
  **exactly-stated persistent floor** above measured supply refuses without a probe under every
  posture (no probe outcome can shrink what must reside), while a **conservative total** above
  supply is *unproven*, not disproven — a **join** refuses on it (nothing has yet proved the role
  fits, until verdict-store wiring hands the join a green verdict to defer to), but the **fit
  probe MUST be admitted past it**, because the probe is the instrument that answers exactly this
  question and an estimate that can veto its own audit is an authority again. Lane bounds,
  measured per-allocation ceilings, pool configuration and the owner's cap are posture-neutral. It is not a proof of fit and MUST NOT be refined toward one — no
  byte-exactness obligation attaches to it, and a divergence between estimate and residency inside
  the budget is not a defect. The authority that admits a geometry is the **Fit Verdict**: the
  actual module, on the actual backend, at the granted geometry, under the enforced budget, in the
  sandbox — recorded content-addressed and **memoized by
  `(module hash, backend implementation revision, plan, grant, budget)`**. A green verdict admits
  that exact key; a red verdict is a **contained, typed outcome** whose answer is a smaller
  geometry from the grant's declared space, not a forensic investigation; an absent verdict means
  the probe has not run — never that the estimate answers instead. Fleet feasibility for a frozen
  roster is set membership — every node holds a green verdict — and involves no roster search and
  no offer/matching protocol. An infeasibility surfaced by estimate or verdict refuses TYPED,
  naming the binding constraint, its scope, and the refusing authority. The only question this
  model ever escalates to the owner is a **product** question (a device whose declared space is
  empty at minimum geometry: in or out); byte, margin and allocator questions are answered by the
  probe or they are not questions.

**Supply is discovered, never supplied.** A certified platform adapter derives conservative usable
device supply from the platform's own facts, and there is no parameter through which a human figure
can enter: the node is the party that can measure the device and the operator is the party that
cannot. Where no trustworthy derivation exists the backend **fails certification or admission** —
that is the answer, not a prompt. The figure MUST be **stable rather than instantaneous**: a
platform budget that moves with co-tenant pressure is the governor's input, because a report cited by
digest cannot be a different report each time it is taken. An **optional owner cap is separate node
policy, outside the report**, and admission makes **two independent comparisons with two
attributions** — estimate against measured supply, and estimate against the cap where one is set. A single
`min()` of the two loses which refused, and an operator whose own policy is the binding constraint
must not be told their hardware is too small.

**Ceilings are measured or absent.** The per-allocation ceiling an estimate is validated against MUST be
a **measurement, taken by allocating**, carried with the method that obtained it. Every ceiling a
platform merely *states* — a framework constant, an advertised buffer limit — describes what an API
permits rather than what the device honors, and promoting one into a field whose contract says
measured is how a two-gigabyte supply was once reported for a thirty-gigabyte card. An absent
measurement refuses admission; it does not fall back.

### 9.7 Per-platform conformance

*Ratified 2026-07-26 (normative amendment A4, platform conformance). [PC-9]'s checklist form also
lands in [`vhc-fleet-ceremony-runbook.md`](vhc-fleet-ceremony-runbook.md), which is where an
operator reads it. Cross-references §9.4 (packaging).*

- **[PC-1]** The platform-sensitive registry is a **floor, not a ceiling**: process supervision,
  path and permission handling, secret storage and dynamic loading are all platform-shaped in this
  tree, and each entry inherits [PC-9]'s per-platform evidence obligation.
- **[PC-3]** Every probe returns a **measured value or a typed unavailability**. A zero resource
  reading is an admission refusal wearing a measurement's clothes: it refuses the machine rather than
  reporting the defect, and it sends whoever investigates to the wrong place. Absence MUST
  distinguish *not exposed by the platform*, *not exposed by the framework*, *the probe failed*,
  *requires a privilege this process lacks*, and *not applicable to this lane*.
- **[PC-9]** A platform is conformant when its evidence exists **per platform**, on **sealed
  binaries**, not inferred from another platform's run.
- **[PC-10]** Profile certification requires that the implementation identity is reported
  truthfully, only compatible profiles are accepted, workspace formulas bound observed allocation,
  pool retention matches, **max-allocation probing is accurate**, compilation and staging are
  included, measured peaks sit within claim plus headroom, **stale profiles fail closed**, and the
  digests enter admission evidence. Five separately-priced operation families cannot be certified by
  joint sampling: either isolated per-family evidence, or per-dispatch attribution, or a profile that
  deliberately collapses them into **one workspace group** whose claims are then group-level claims.
- **[PC-11]** Profiles are versioned **independently of any guest**. A profile revision does not
  re-pin a module, and a profile-attributed divergence costs a re-certified profile and re-composed
  claims with the guest hash unchanged.
- **[PC-12]** A profile is accepted only under the **intersection** of owner policy and run policy,
  and a refusal names the rejecting policy. The envelope binds schema version, compatible planner
  version, sealed backend binary identity, certification evidence digest, signer and release
  authority, validity and revocation, and the exact implementation plus permitted driver/API ranges.

  **Development authorities require explicit naming by BOTH policies** *(ratified 2026-07-26)*. A
  development-signed profile is accepted only when owner policy **and** run policy each name that
  development authority explicitly. Deferral-on-silence — an empty set deferring to the other side —
  remains correct for **release** authorities only; for a development authority silence is never
  consent, because a machine owner listing a dev key would otherwise admit a dev-signed profile into
  a run that never opted in, and a run listing one would push it onto a machine whose owner never
  agreed. Each side would be unilaterally lowering the other's bar. A development authority
  **authenticates and does not certify**: it satisfies integration evidence and never ceremony
  certification, and the class comes back from authentication so the fence survives a misconfigured
  policy and a caller who did not know it was there.

- **[PC-13]** A superseded, revoked or dangling certification record **fails a statement closed**.

**Authoring caution — a permitted range MUST name whose numbering it constrains.** Two revision
fields carry different numbering on different platforms, and a range that does not say which it means
is unevaluable:

- `os.build` carries the **kernel release** on Linux (for example `6.19.7`) and a wholly different
  **OS build identifier** on macOS (for example `25D2140`). They are not comparable, not ordered
  against each other, and not interchangeable.
- The same applies to driver revisions, where a vendor release and a framework-reported driver
  version are two numbering systems for one stack.

A permitted range therefore states the platform whose numbering it constrains, and a record from
another platform's numbering does not satisfy it. On a backend whose framework supplies no driver
revision at all — every Metal case — `os.build` is the implementation-revision signal, which is
precisely why its numbering must be named rather than assumed.

---

## 10. The fleet contract: tiers and eligibility

*(Adopts VHC ref §4.2 (D2) with one deliberate inversion: the reference declares hardware tiers
with trust semantics; v2 derives tier labels from measurement. Permissioned v2.0 runs need no
trust perimeter beyond org membership, and hardcoded tiers age badly against real fleets.)*

- **[FT-1]** **Tiers are labels derived from measured eligibility, never admission classes.** A
  peer's probe verdict (backend rung, `DeviceLimits` budget, throughput class, host RAM, uplink,
  disk) determines *which PlanIR roles it can hold* ([GR-9]); "anchor"/"body"/"edge" are display
  groupings over that role-set (anchor ≈ eligible for the largest-memory, highest-availability
  roles), never gates of their own.
- **[FT-2]** Role admission is an explicit per-role memory inequality (VHC ref D2-1,
  generalized): measured effective budget ≥ the group's `memory` claim, where the claim includes
  weights, grads, optimizer state, activations, payload staging, and checkpoint-staging headroom
  ([EB-4]) for that role's fragment share. A peer MUST NOT be promised a role generically ("can
  host stages", "can replay") — only roles whose inequality it satisfies.
- **[FT-3]** Eligibility inputs are probe- and execution-attested: declared classes seed
  assignment weighting only; measured round outcomes re-weight and eventually re-tier (v1 spec
  §6.5's trust-but-verify; VHC ref §10.9's "attested outcomes, never self-reported specs" is the
  direction of travel). Probe reports ride the existing signed hardware-report surface
  (`SwarmHardwareReport` lineage) and are quantized into the fleet snapshot ([GR-7]).
- **[FT-4]** An island (multi-GPU machine or co-located group) MAY register as one logical member
  (VHC ref D2-2); its internal parallelism is invisible to the protocol. v2.0 fleets are
  single-GPU-per-member in practice; the contract just refuses to bake that in.

---

## 11. Assurance layering

*(Adopts VHC ref §10.2's layered stack into v2's channel classes. Scoped honestly: v2.0 ships
det-channel assurance only — which is the strongest layer anyone has — and specifies the
statistical path so it is designed-for and dormant, not retrofitted.)*

| Layer | Covers | Status |
|---|---|---|
| Signed envelopes, records, receipts, version tuples (provenance) | who committed which bytes when | shipping (v1) |
| Round digests over consensus state | divergence *detection* on det channels | shipping (v1) |
| **Deterministic replay** | divergence *verdicts* + full audit on det channels | shipping (v1: observe/replay) |
| Statistical screens (SENTINEL-class) | plausibility on native channels | **specified, dormant** |
| Availability evidence | committed payloads are fetchable | shipping (v1: receipts/attestations) |
| Quarantine → adjudication → penalty | containing flagged members | quarantine shipping; penalties out of scope (permissioned) |

- **[AL-1]** **Det replay is the verifier for det channels.** For every `det` consensus channel,
  post-state is a pure function of (checkpoint, records, payloads) — replayable offline
  byte-exactly (I1). This is the subsystem's strongest asset and MUST survive every v2
  generalization: a proposed channel or state feature that would break det-channel replay is
  rejected at design review, not mitigated.
- **[AL-2]** Digest mismatch handling generalizes v1 (v1 spec §6.4): mismatch ⇒ quarantine
  (excluded from staging and routing, lease suspended, state preserved) ⇒ resync-by-replay ⇒
  readmission; repeated ⇒ leave + operator alert. **Quarantine always precedes penalty** (VHC ref
  §10.7); slow-but-honest is the canonical false-positive class, separable because realized
  staleness is observable ([VT-3]).
- **[AL-3]** **Statistical screens are the assurance path for native channels** (future
  `activation`/`kv`): EMA baselines of boundary statistics at verifier positions, IQR-calibrated
  thresholds tuned to a target false-positive rate, taint-and-substitute on flags so training
  never stalls, violation counters with forgiveness — SENTINEL's shape (map §3.13; VHC ref
  §10.3). Screens read the trajectory monitor's committed statistics (§12): the monitor is the
  shared statistical substrate, not a second pipeline. Dormant until an execution-plane transport
  exists ([LC-2]); specified now so `metric@1` and the monitor carry the right statistics from
  day one.
- **[AL-4]** Provenance is never accepted as correctness (VP-11): a signed, committed, available
  payload is *plausible* until replay (det) or screening (native) covers it. Records track which
  layer covered each committed entry, so audit coverage is itself auditable.
- **[AL-6]** *(forward reference, design-intent)* A future **verifiability/audit layer** (§14,
  reserved, dated 2026-07-17) sits **above** this stack, not inside it, with two composable
  pieces: a signed, merkleized **audit trail of inputs** enabling optimistic verification (commit
  cheaply; on dispute, bisect to one step and re-execute or check a single-step proof), and
  **procedure accountability** — validity proofs that any verifier-role decision procedure (the
  [AL-3] screens first) executed its own published transition function over committed inputs,
  closing the trusted-verifier gap. It alters no verdict in this table and does not weaken VP-11
  (a commitment still proves only provenance; even a transition proof proves procedure, not
  ground truth).
- **[AL-5]** Assurance-plane traffic never gates round liveness (§3.1): divergence parks or
  quarantines; deadlines commit.
- **[AL-7]** **Admission-layer assurances precede this stack.** Certified per-run identity (§6.3.1)
  and the immutable admitted tuple (§6.3.2) are the admission-time assurances the whole stack above
  presumes: the first establishes *who* may sign plane traffic for a given `(run, role,
  incarnation)` — a CSPRNG per-run key certified by a genesis-named base identity, mandatorily
  checked on every production frame — and the second establishes *that* the artifact, configuration,
  grants, and policy a peer joins under are byte-for-byte those admission assessed. Provenance
  ([AL-4]) only means something once the signer is an authenticated run identity and the execution
  identity is agreed; both are furnished here, before the first round opens. These are admission-
  layer guarantees, defined normatively in §6.3, not additional entries in the round-assurance table
  above.

---

## 12. The trajectory monitor

*(The one shared physical assumption — trajectories drift slowly — surfaced as one host service
(VP-6). v1 has fragments of this: loss metrics, digests, observe. v2 names the service and its
contract.)*

- **[TM-1]** **Signals.** Per group (and per channel where applicable): parameter drift d_θ
  (norms of round-over-round deltas of consensus state); update statistics (norms, clip rates);
  boundary-activation statistics (native channels, when they exist); and **functional
  disagreement** — Jensen–Shannon distance between members' per-token output distributions on
  deterministic probe batches — the leading instability indicator, ahead of L2 parameter
  distance (Factored Gossip via map §3.14; VHC ref §7.5's disagreement vector D).
- **[TM-2]** **Sources.** Guests contribute statistics through `metric@1` on schedule-declared
  probe batches (deterministically assigned, so every member measures the same thing); the host
  contributes what it observes for free: digest distances, payload norms, realized staleness,
  round timing. No signal that feeds a consensus decision trusts a single reporter — per-peer
  statistics are cross-checked like any consensus input.
- **[TM-3]** **Consumers.** Exactly three, by contract: schedule **triggers** ([PIR-17] — e.g.
  JS-triggered extra sync, Appendix A.3); **`outer@1` staleness inputs** (§7.1); **integrity
  screens** ([AL-3] — the same EMAs that correct staleness are the anomaly baselines, the map's
  "correction and verification are two faces of one bet", map §4). New consumers subscribe; they
  do not grow private estimators (VP-6).
- **[TM-4]** **Consensus discipline.** Any signal that feeds a consensus decision (triggers,
  outer math) is quantized ([PD-2]) and committed at round boundaries so all members act on
  identical values; free-running unquantized signals are observability only.
- **[TM-5]** **Storage.** Signals ride the observe/event-log machinery (v1 spec §14) keyed by the
  version tuple. Signals are not consensus state and are never digested; their *committed
  quantizations* inside records are.
- **[TM-6]** *(informative)* The VHC ref's drift-response surface ℛ (§10.5 there) — the measured
  map from disagreement and staleness to loss penalty and detection statistics — is the
  calibration object for trigger thresholds, screen floors, and [CO-8] acceptance. The degenerate
  plan pins D ≈ 0; measuring ℛ becomes meaningful with the first non-degenerate plan, and the
  monitor is specified now so those runs produce ℛ's axes from day one.

---

## 13. The assumption-contract layer

*(The field's largest open problem is composition-by-assumption: methods are validated in
isolation and composed on faith (map §10.4). v2 mechanizes the map's discipline — every plugin
publishes the invariants it assumes — as the `requires`/`provides` registry over slot bindings
([EV2-5]) plus PlanIR grants ([PIR-14]), enforced before a run, not discovered forty hours in.)*

- **[AC-1]** **The registry.** Contract keys and value vocabularies are registered in the module
  ABI spec and grow additively. Initial keys: `state_access: [state names]`,
  `optimizer_invariant` (e.g. `row_constant_second_moment`), `update_family`
  (`linear` | `sign`), `gradient_bias` (`unbiased` | `biased_regularizing`),
  `det_capable: bool`, `channel_class` (compress slots), `staleness_tolerance` (declared bound
  class).
- **[AC-2]** **Wiring-time conflict detection** runs at admission, before any execution, over
  (envelope slots table × module manifest × plan): every `requires` satisfied by a `provides`, a
  grant, or a host guarantee; compressor `channel_class` matching its channels ([SL-2]/[PIR-9]);
  every slot on a `det` channel path declaring `det_capable`; contradictory provides refused
  loudly with both parties named. Verdicts land in the run log; a refused wiring never reaches
  guest execution.
- **[AC-3]** Worked examples (normative as test vectors for [AC-2]):
  1. A **Protocol-Models-style subspace compressor** declares
     `requires { optimizer_invariant: "row_constant_second_moment", state_access: ["subspace_basis"] }`.
     Wiring it with a `model@2`/`outer@1` pair that does not provide the invariant refuses at
     admission — the compressor-optimizer leak as a checked capability instead of a silent
     divergence (map §10.3 decision 2).
  2. **Forward-masking SDP** declares `provides { gradient_bias: "biased_regularizing" }`; an
     `outer@1` requiring `unbiased` refuses; backward-masking SDP provides `unbiased` and wires.
     The mask axis's known optimization property becomes machine-checked (map §10.5).
  3. A **DeMo-style momentum compressor** declares
     `requires { state_access: ["momentum"], update_family: "sign" }` — it needs a grant on the
     momentum state it owns and only composes with a sign-family outer; a linear-family outer
     refuses.
- **[AC-4]** Contracts are legible in records: `composition_digest` ([VT-1]) plus the manifest's
  declared contracts reconstruct *what was promised* for any historical run — ablations and
  postmortems read contracts, not code.

---

## 14. Verifiable and auditable training runs (FUTURE — design-intent, 2026-07-17)

*(**Design-intent, NOT implemented.** This section reserves the shape of a verifiability/audit
layer for training runs; it builds nothing in v2.0 and lands nothing on the training hot path. Its
job is to make the journal (ABI §8), the Phase-D record archive (ABI §15), and the det-lane digests
(§7.4, §11) carry the structure two later programs — an optimistic-verification layer, and after
it a zkvm orchestrator — can consume without a re-cut. The layer is specified **generically**: it
applies to any run topology, any assurance screen, and any verifier-role decision procedure;
nothing in it binds to one detector, one paper, or one deployment profile. Worked instantiations
(e.g. the SENTINEL-ZK protocol companion for the [AL-3] screen profile, sources table) are design
input only; none of their machinery is normative here. Nothing in this section changes v2.0
behavior, adds a wire type, or introduces a crate; §14.7 states the non-goals and §14.5 fixes the
cost discipline and the off-by-default rule. Dated 2026-07-17.)*

The correctness stack today has two verifiers (§11): **deterministic replay** on `det` consensus
channels (bit-exact, the subsystem's strongest asset, [AL-1]) and **statistical screens** on
`native` execution channels (specified-and-dormant, [AL-3]). Both leave the same structural gap
the moment any *verifying role* is not organizationally trusted: nothing watches the watcher. Any
procedure that can flag, substitute, quarantine, or ban on grounds others cannot check — a
statistical screen, an admission gate, an arbitration policy — *relocates* trust rather than
removing it: a dishonest holder of that role can fabricate inputs, poison its own baselines, flag
honest members, or silently drop messages, and no journal entry proves otherwise. This section
closes that gap with two composable pieces, both strictly above VP-11's provenance base: **(1)** a
hashed + merkleized **audit trail of inputs** — signed commitments to what every party hands over,
which is what makes *optimistic* verification (commit cheaply, re-execute only on dispute)
possible at all (§14.1–§14.3); and **(2)** **procedure accountability** — validity proofs that a
decision procedure executed its own published transition function over committed inputs (§14.4).
A commitment still proves only *who committed which bytes when*, and even a transition proof
proves *procedure* (the published rules classified this input this way), never ground truth (the
member was malicious). Correctness verdicts still come from re-execution, screening, or a validity
proof — never from a signature alone.

### 14.1 The input audit trail: signed, merkleized commitments

The foundation of both pieces is an audit trail of **what every party handed over**, hashed and
merkleized so one item is provable without the whole history (O(log n) inclusion proofs — the base
journal's flat whole-file BLAKE3 chain, ABI §8.2, requires the entire segment to verify one
record). Commitment classes:

| Class | Source | Structure | Why |
|---|---|---|---|
| **Per-round training inputs** — staged committed-set payload bytes and `kind-0` module-wire frames the round branched on | journal event/publish records (ABI §8, tags 1/4); content-addressed by `blake3` already | **Merkle leaf** = the record's existing `blake3` | a disputed round's inputs must be provable-in without the whole segment |
| **Guest-published commitment / digest frames** (`tag-3`/`tag-4`) | det-lane digests the guest publishes at `sync`/`commit` (§7.4) | **Merkle leaf** = the published digest | the consensus-visible integrity anchor; as leaves they need no new hashing (§14.5) |
| **Stage-boundary tensors in pipeline-parallel runs** — inter-stage `activation`/`kv` transfers and their gradients (the signals the [AL-3] screens inspect) | `execution`-plane channel events (§3.1, §4.2.2), when those transports exist ([LC-2]) | **worker-signed proof-native commitment** per emitted tensor ([VC-2]); one leaf per boundary event | per-boundary granularity localizes a dispute to one stage; the signature stops a verifier proving against fabricated inputs |
| **Checkpoint manifests** | state-manifest sections, content-addressed per section (ABI §10.2) | **already a hash list**; the manifest digest enters as one leaf | internal merkleization buys nothing over a small flat vector ([VC-5], OQ-VC-4) |
| **Coordinator decisions** — committed-set roots, `RoundRecord`s, epoch records | coordinator journal / round protocol (§6.2, §7.4); committed sets are **already Merkle roots** (v1 §6.4) | **Merkle leaf** = the record digest | binds *which decisions* the run took to the same signed head, so a dispute can name a round unambiguously |

- **[VC-1]** The layer defines, per journal segment carrying any class above, a **record-digest
  Merkle tree**: leaves are the canonical-CBOR record digests (or the det-lane digests / content
  addresses those records carry), ordered by the journal's monotone record ordinal (ABI §8.2). The
  tree is additive metadata over the segment; it MUST NOT reorder, mutate, or replace any
  base-journal record.
- **[VC-2]** **Tensor commitments are worker-signed and proof-native.** On execution-plane channels
  the emitting worker MUST sign a commitment binding the exact canonical tensor bytes it sent
  (shape and payload digest included), in a commitment scheme the future proof layer can open
  *inside* a proof relation (a committed-witness polynomial commitment) — signatures are verified
  natively, outside any circuit, and the proof binds to the commitment. Without this, a lying
  verifier fabricates inputs and "proves" against fiction; with it, input fabrication is
  cryptographically excluded.
- **[VC-3]** **Receipts and availability.** Every handover is acknowledged by a **signed,
  sequence-numbered receipt** from the recipient, so silent message-dropping is itself provable
  misconduct rather than deniable weather; and committed tensors carry a **data-availability
  obligation** (bounded retention, erasure-coded or escrowed per policy) so a dispute can actually
  retrieve what was committed. Receipt and availability windows align with the run's retention
  windows ([CO-3]).
- **[VC-4]** Payload *bytes* are never copied into the commitment layer. Consistent with the
  journal (ABI §8.1) and I5 coordinator blindness ([GR-3]), only digests/commitments are
  committed; bytes live in the content-addressed payload plane (or the availability profile of
  [VC-3]) and are fetched only on dispute.
- **[VC-5]** Classes already carrying hash lists or Merkle roots (checkpoint manifests,
  committed-set roots) are **not** re-merkleized internally; their digest/root enters the segment
  tree as a single leaf. Merkle structure is added only where inclusion-without-the-whole is the
  actual need.

### 14.2 Where the layer sits, and the commitment primitive

- **[VC-6]** The audit trail is layered **above** the base journal, exactly as the Phase-D record
  archive already layers over ABI §8.2's chained segments and §8.6 evidence records. It does
  **not** replace the base discipline: per-record CRC32C and the whole-file BLAKE3 segment chain
  remain the unsigned base, and OQ-12's resolution (ABI §8.2) stands. This preserves VP-11: the
  base journal is a faithful transcript for replay; cryptographic authority is a distinct higher
  layer.
- **[VC-7]** The layer **extends the AttestedHead / record-archive authority from coordinator-only
  to worker training journals**: a worker journal's sealed segment is covered by an ed25519
  **AttestedHead** (`SingleKey` or threshold `Authority`; ABI §15) whose committed value is the
  **[VC-1] record-digest Merkle root** — not the whole-file hash the base chain uses — so
  inclusion proofs verify against a signed head. The head's `Authority` is the run's certified
  per-run key chain, scoped by the execution identity `(run_id, epoch, role, instance,
  module_hash)` (ABI §8.1). Cross-checking a worker's committed inputs against the coordinator's
  committed-set roots binds the two heads into one auditable run — neither head alone is trusted
  for correctness (VP-11).
- **[VC-8]** The commitment primitive SHOULD be a **BLAKE3 Merkle tree over canonical-CBOR record
  digests, signed by the existing ed25519 AttestedHead**, and SHOULD NOT adopt the daemon-host
  journal's Gordian Envelope / dCBOR sealing (daemon-host-spec §5.1) for this layer: the leaves
  are digests the journal and det lane already produce, the whole `tabi@2`/PlanIR/envelope/journal
  contract is canonical CBOR + BLAKE3 (wire-contract coherence; no new dependency), and the
  AttestedHead signer already exists — only *what is committed* changes (root vs whole-file hash).
  The daemon-host choice is right for its dCBOR-native transcript world and is not a precedent to
  copy; the divergence is recorded, not accidental (OQ-VC-1). The one addition [VC-2] forces:
  stage-boundary tensor commitments are **proof-native** (polynomial commitments, not plain
  hashes) where the §14.4 proof layer is in scope — a plain BLAKE3 leaf cannot be opened inside a
  proof relation.

### 14.3 Optimistic verification: commit cheaply, re-execute on dispute

"Optimistic commitments" is the happy-path discipline the audit trail enables: **post commitments,
decision bits, and signed state roots per step — one hash plus one signature, microseconds — and
re-execute only when someone disputes.**

- **[VC-9]** **Optimistic acceptance.** By default committed work is accepted without
  re-execution: the audit trail is *recorded*, not *checked*, on the hot path. Verification is
  triggered, never continuous — this is what distinguishes the layer from full computation
  duplication (the redundancy baseline that halves throughput).
- **[VC-10]** **Deterministic re-execution target.** Anyone can re-execute the **deterministic
  fixed-point specification** of the disputed computation against the committed inputs. For `det`
  consensus channels that substrate already ships: bit-exact input replay (ABI §8.7) and the det
  lane on every backend rung ([EB-8], [AL-1]). For verifier-role decisions, the normative
  fixed-point transition semantics of §14.4 ([VC-14]) are the re-execution target. Worker *stage
  computation* on `native` channels is NOT bit-reproducible across heterogeneous backends (VP-8)
  and stays outside the objective-re-execution perimeter — it remains covered by the statistical
  screen ([AL-3]); this boundary is explicit, and closing it is OQ-VC-3.
- **[VC-11]** **Dispute path: bisect, then referee.** A challenge names a step/round/stage
  boundary with a Merkle inclusion proof against the signed head ([VC-1], [VC-7]); a disagreement
  over a span of steps **bisects** over the committed per-step state roots to the single first
  step where committed and re-executed state diverge; that one step settles either by a **referee
  quorum** re-executing it (composition: OQ-VC-5) or by a **single-step validity proof** (a SNARK
  of that step alone) — so proof cost is paid **per dispute, not per step**. Adjudication reuses
  the assurance ladder: sustained divergence drives quarantine → resync-by-replay → readmission
  ([AL-2]); quarantine always precedes penalty; a false challenge is recorded against the
  challenger.
- **[VC-12]** **What optimism smuggles in (stated, not hidden).** The scheme is sound only with:
  **data availability** of committed tensors ([VC-3]) — you cannot re-execute what you cannot
  fetch; **staking/slashing or an equivalent cost** so misreporting and false challenges cost
  something (out of scope for permissioned v2, where the ladder's quarantine is the only teeth —
  OQ-VC-6); a **1-of-N honest-challenger assumption** — someone must actually check within the
  window; and the **signed sequence-numbered receipts** of [VC-3] so omission is provable. A
  deployment that cannot meet these has a weaker layer and MUST say so rather than imply
  optimistic security it does not have.
- **[VC-13]** **Challenge window.** A committed step/round is disputable for a bounded window in
  group rounds / membership epochs, aligned to retention ([CO-3], v1 §6.4); outside it, final.
  Window length and challenger set are policy (OQ-VC-6).

### 14.4 Procedure accountability: proofs of the decision transition function

Any role whose decisions carry consequences others cannot check — an assurance screen ([AL-3]),
an admission or arbitration gate, an aggregation-policy holder — is a **decision procedure**: a
published transition function over committed inputs and its own persistent state. Accountability
means proving the procedure executed that function, nothing more.

- **[VC-14]** **The statement proven.** Per decision event: *given the signed commitment to the
  input ([VC-2]), the previous procedure state root (baselines, histories, counters — whatever the
  procedure's published state comprises), the public randomness of [VC-15], and the published
  decision with new state root — the decision and the new root follow from correctly executing
  the procedure's published, normative transition function.* This proves **procedure**, not
  ground truth: it certifies that the public rules classified this input this way, never that the
  flagged member was malicious or that the member's own computation was correct. Behavior a
  screen is statistically blind to remains blind spots of that screen, proof or no proof.
- **[VC-15]** **Prerequisites.** (a) Signed, proof-native input commitments ([VC-2]) — signatures
  verified outside any circuit, the proof bound to the commitment. (b) **Post-commitment neutral
  randomness**: any randomized sub-procedure (sampling, projections, probe selection) MUST derive
  its randomness from a source fixed only *after* the input commitment is signed and irreversible,
  and controlled by neither the committing party nor the procedure holder — else either side
  grinds the randomness. Candidate sources are committee-held threshold randomness or an external
  beacon (OQ-VC-7).
- **[VC-16]** **Make it provable first.** A decision procedure becomes provable by
  **respecification**, never by compiling its floating-point implementation: (a) deterministic
  fixed-point semantics, with every rounding mode, tie-break, interpolation rule, inclusive bound,
  and escalation criterion stated normatively — anything left as prose is unprovable; (b) prefer
  **witness-plus-relation over in-circuit algorithms** — supply a claimed result as witness and
  prove the cheap relation that validates it (a sorted array proves by multiset-equality plus
  monotonicity, never by proving a sort; division and roots certify by quotient/remainder and
  range witnesses); (c) fuse streaming reductions over the same input into a single pass;
  (d) respecification changes the procedure's observable behavior, so its published performance
  MUST be re-validated on the normative semantics before adoption ([VC-23]).
- **[VC-17]** **Architecture: local lanes + one global policy proof.** Accountable state splits
  into independent **local lanes** along the run's natural units (per stage and direction, per
  channel, per group) — each lane proof certifying the chain commitment-opening → decision
  computation → decision branch → state update (or freeze) → output behavior → event emission →
  new lane root — plus one separate **global policy proof** consuming the ordered event stream
  for cross-lane policy (counters, escalation, bans). **Canonical event ordering is finalized by
  a coordinator/BFT log — the proof proves the transition against the finalized event root; it
  cannot pick the canonical order itself.** Any calibration or warm-up lifecycle is itself proved
  and closed by a finalized readiness record before the procedure's decisions carry weight.
- **[VC-18]** **THE key security condition: mandatory, bounded-lag proof coverage.** A
  procedure's state may advance **only** through transitions that are eventually — within a fixed
  deadline — covered by **mandatory validity proofs**. On-demand-only proving (prove a ban only
  when someone asks) is **unsound**: a dishonest procedure holder poisons its own
  baselines/histories/counters through unproven transitions, then honestly proves the ban against
  the corrupted state. **Folding/recursion is certificate *compression*, not the arithmetic
  accelerator: aggregation of already-proven proofs may be lazy; base proofs may not be.** Three
  valid finality modes: (a) synchronous proof-carrying output; (b) mandatory bounded-lag proving
  (proofs due within ≤K events or ≤Δ time; a missed deadline rolls back to the last proven
  epoch — **not** optimistic: no challenger is needed, absence of proof is itself protocol
  failure); (c) lazy folding of already-proven batch proofs. Batching discipline: bound the
  transitions per lane epoch, close an epoch early on violation, certify violations immediately,
  aggregate clean epochs in the background, and tree-accumulate across lanes rather than one
  serial fold.
- **[VC-19]** **Fault classes and what each mechanism buys** *(informative)*: invalid
  computation → no accepted proof; input fabrication → excluded by signature + commitment binding
  ([VC-2]); randomness grinding → excluded by post-commitment randomness ([VC-15]); state fork or
  omission → excluded by public state roots, proof chains, and the finalized event log; silent
  dropping → provable via [VC-3] receipts; ambiguous liveness (network weather) →
  abort/non-finality, honest attribution may be impossible; behavior that is malicious but
  statistically normal → out of scope by construction ([VC-14]).

Proof-system selection (prover families, commitment schemes, benchmark gates) is deliberately
**not** specified here — it belongs to per-profile protocol companions (sources table). The
architectural requirements are only: proofs are **publicly transferable** (a designated-verifier
proof is not an audit artifact); provers **stream** (memory bounded independently of run length);
and throughput satisfies [VC-21].

### 14.5 Cost model and the off-by-default rule

- **[VC-20]** **Happy-path cost is one hash + one signature per commitment.** [VC-1]'s tree leaves
  on `det` classes are digests the journal already computes; segment-tree building over 32-byte
  leaves plus one AttestedHead signature MAY be on by default for coordinator/consensus journals
  (no payload hashing is added). Stage-boundary tensor commitment ([VC-2]) IS new hot-path work
  (a proof-native commitment per emitted tensor) and MUST default off, be enabled per run as an
  envelope/plan-declared capability, and be bounded by the per-event payload budgets ([PIR-10]).
- **[VC-21]** **Prover throughput is a formula, not a promise.** For `n_events` accountable
  decision events per optimizer step of duration `T_opt`, the proof service MUST sustain
  `μ_proof ≥ (1.2–1.3) · n_events / T_opt` — stable queues with explicit headroom — under a p99
  proof-lag bound tied to the [VC-18] deadline. Proving runs as a service off the training hot
  path while the native procedure keeps running natively. Overhead is expressed and gated as a
  **multiple of the procedure's native work**: the acceptance bar is that the proof stack beats
  the redundant-computation baseline (re-running the work on a second node — the alternative any
  accountability layer must outcompete), and the artifact it buys is one redundancy never
  produces: a **transferable proof that a decision was justified**. Input compression serves
  proofs exactly as it serves bandwidth — commit to what the procedure actually consumes. No
  seconds-level or percent-of-bill figure is normative before a measurement campaign runs on the
  target profile.
- **[VC-22]** **Nothing lands on the hot path in v2.0.** This is design-for-later: the layer is
  reserved so journal, archive, and digests carry the right structure. Until scheduled, `det`
  replay and the dormant statistical screens remain the shipping assurance (§11).

### 14.6 Two-tier adoption and composition with the zkvm orchestrator

- **[VC-23]** **Two tiers, adopted as the recommendation.** **Tier 1 — optimistic commitments,
  always**: the near-free happy path of §14.1–§14.3 whenever the audit trail exists at all.
  **Tier 2 — mandatory bounded-lag validity proofs of decision transitions ([VC-18]) wherever the
  deciding roles are decentralized** — the moment a verifier-role holder is not organizationally
  trusted, tier 2 is what makes its decisions adoptable at all. **TEE attestation** is
  acknowledged as the pragmatic near-zero-overhead alternative for tier 2, with its trade-off
  named: it substitutes "trust the vendor's attestation chain" for cryptographic soundness.
  **Empirical caveat (blocking for adoption, not for design):** respecification ([VC-16]) changes
  a statistical screen's observable behavior — its published detection performance MUST be
  re-validated on the normative semantics (full attack suite, multiple seeds) before a deployment
  relies on it; witness-preserving formulations ([VC-16]b) are preferred precisely because they
  keep the original statistical definition intact.
- **[VC-24]** **One substrate, two consumers — composition with the zkvm orchestrator.**
  Optimistic re-execution ([VC-10]) and validity proofs commit to the **same** [VC-1]/[VC-2]
  roots: a future zkvm orchestrator produces succinct proofs whose **public inputs are the
  committed input and output roots** for a unit, so the zk proof *replaces or compresses* the
  optimistic re-execution (dispute settlement collapses from "re-execute one step" to "check one
  proof" — [VC-11]'s single-step SNARK is the first instance) **over commitments that do not
  change**. The migration is a ladder, not a frame change: audit trail first (near-free), disputes
  by re-execution next, mandatory transition proofs where verifier trust demands them, zk
  compression of whatever remains — each rung consuming the rung below's commitments. All rungs
  stay strictly above VP-11.

### 14.7 Non-goals (now)

No implementation; no wire-protocol change (no new `Command`/`Event`, DTO, or CDDL type; no
WireVersion bump); no new crate; no version bump. No change to the base journal (ABI §8.2 / OQ-12
stands), to VP-11, or to the §11 assurance verdicts. No adoption of any companion protocol's
normative machinery (committees, locks, escrow, certificate formats) into this spec — companions
are design input, not architecture. This section reserves a design; it authorizes no code.

### 14.8 Open questions

- **[OQ-VC-1]** Encoding unification. [VC-8] recommends canonical CBOR + BLAKE3 Merkle and
  declines the daemon-host Gordian/dCBOR primitive. If a future cross-subsystem audit tool wants
  one envelope format for both journals, teach the tool both primitives or reconcile the journals?
  (Recommendation stands until such a tool is scoped.)
- **[OQ-VC-2]** Proof-native commitment scheme. [VC-2]/[VC-8] require stage-boundary tensor
  commitments a proof relation can open (polynomial commitments), while the rest of the trail is
  plain BLAKE3. Which committed-witness PCS (hiding, transparent, streaming), and does the
  plain-hash trail ever need retrofitting to proof-native?
- **[OQ-VC-3]** Objective verdicts on native stage computation. Decision transitions become
  provable ([VC-14]) and `det` re-execution is bit-exact, but worker stage computation on `native`
  channels is tolerance-class (VP-8) and outside the objective perimeter ([VC-10]). Is an
  objective fraud proof ever achievable there (deterministic-kernel profiles? proof-friendly
  quantized inference?), or is the honest ceiling the statistical screen plus committed audit
  trail for forensics?
- **[OQ-VC-4]** Commitment granularity. Per-record vs batched per-round/per-stage leaves; do
  checkpoint manifests ([VC-5]) ever need internal merkleization? Proof size vs tree-build cost.
- **[OQ-VC-5]** Referee composition. [VC-11]'s bisection endgame needs a referee quorum (or a
  single-step validity proof): who sits on it on this fleet — coordinator, sampled peers,
  external auditors — and what does its honest-majority assumption cost?
- **[OQ-VC-6]** Challenge economics. Window length in rounds/epochs ([VC-13]); challenger set;
  staking/slashing mechanics vs quarantine-only teeth in permissioned deployments ([VC-12]); who
  pays for dispute re-execution.
- **[OQ-VC-7]** Randomness source for post-commitment neutrality ([VC-15]): committee-held
  threshold randomness vs an external randomness beacon on this fleet, where no standing
  committee exists today; and whether the coordinator can host the source without violating I5
  blindness.
- **[OQ-VC-8]** Worker-head authority under churn. A worker AttestedHead ([VC-7]) is scoped to an
  incarnation; how does a committed head survive resync/handoff (§4.4 step 5, §7.3 point 5) so a
  dispute can still verify inclusion against work a since-departed incarnation committed?
- **[OQ-VC-9]** Screen re-validation. Who runs the [VC-23] statistical-equivalence campaign for
  a respecified screen (normative fixed-point semantics vs the original floating-point detector,
  full attack suite, multiple seeds), and does it gate only tier-2 adoption or any [AL-3] screen
  that adopts the fixed-point semantics?

---

## Appendix A — worked plans *(informative)*

Canonical-CBOR-shaped pseudo-JSON (comments for exposition; hashes and most sizes elided). Every
example is a point in the same `vocab_version 1` vocabulary — that is the point.

### A.1 The degenerate DP plan (v2.0 launch: today's run, re-expressed)

The 160M-class preset shape (v1 spec §6.1's envelope, H = 30):

```jsonc
{
  "vocab_version": 1,
  "groups": [
    { "group_id": 0, "role": "replica", "fragments": 1,
      "replication": { "min": 4, "target": 16 },
      "memory": { "vram_mb": 8000, "host_mb": 16000 },
      "requirements": { "throughput_floor": "c2", "uplink_mbps_min": 15,
                        "downlink_mbps_min": 100, "disk_gb_min": 60 } }
  ],
  "channels": [
    { "channel_id": 0, "class": "param_delta", "from_group": 0, "to_group": 0,
      "plane": "consensus", "determinism": "det",
      "compress": { "slot": "compress.delta",
                    "config": { "top_k": 64, "quant_bits": 2, "ef_decay": 0.95 } },
      "budget": { "bytes_per_event_max": 41943040 } }
  ],
  "states": [
    { "state_id": 0, "name": "outer_momentum", "class": "replicated",
      "scope": { "groups": [0] }, "dims": [162000000], "dtype": "f32",
      "grants": ["outer"] },
    { "state_id": 1, "name": "error_feedback", "class": "residual",
      "scope": { "groups": [0] }, "dims": [162000000], "dtype": "f32",
      "grants": ["compress.delta"] }
  ],
  "schedule": [
    { "kind": "local_step", "scope": { "group_id": 0 }, "cadence": { "every_steps": 1 } },
    { "kind": "sync",       "scope": { "group_id": 0 }, "cadence": { "every_steps": 30 } },
    { "kind": "commit",     "scope": { "group_id": 0 }, "cadence": { "every_steps": 30 } },
    { "kind": "checkpoint", "scope": { "group_id": 0 }, "cadence": { "every_steps": 12000 } }
  ],
  "data": { "assignment": "interval", "cursor_scope": "per_group" }
}
```

One group, one det consensus channel, sync + commit every H — v1's round. (v1 stores error
feedback as `local`; declaring it `residual` is the v2-stronger contract, [PIR-13]. The [LC-1]
gates run against exactly this plan.)

### A.2 Streaming DiLoCo (fragment-strided sync, α-mixing outer, 4-bit compressor)

Paper-scale sketch (map §3.6): 4B params, F = 4 fragments, H = 100, strided offsets.

```jsonc
{
  "vocab_version": 1,
  "groups": [
    { "group_id": 0, "role": "replica", "fragments": 4,
      "replication": { "min": 8, "target": 32 },
      "memory": { "vram_mb": 22000, "host_mb": 48000 },
      "requirements": { "throughput_floor": "c3", "uplink_mbps_min": 25,
                        "downlink_mbps_min": 100, "disk_gb_min": 120 } }
  ],
  "channels": [
    { "channel_id": 0, "class": "param_delta", "from_group": 0, "to_group": 0,
      "plane": "consensus", "determinism": "det",
      "compress": { "slot": "compress.e3m0",
                    "config": { "bits": 4, "format": "e3m0", "accumulate": "f32" } },
      "budget": { "bytes_per_event_max": 536870912 } }   // one 4-bit fragment ≈ 0.5 GB at 4B/4
  ],
  "states": [
    { "state_id": 0, "name": "outer_momentum", "class": "replicated",
      "scope": { "groups": [0] }, "dims": [4000000000], "dtype": "f32",
      "grants": ["outer"] }
  ],
  "schedule": [
    { "kind": "local_step", "scope": { "group_id": 0 }, "cadence": { "every_steps": 1 } },
    // Strided fragment schedule: each fragment syncs every H=100, phases 0/25/50/75
    // — the stride is derivation output ([PIR-17] phase), not a host law.
    { "kind": "sync", "scope": { "group_id": 0, "fragment": 0 }, "cadence": { "every_steps": 100, "phase": 0  } },
    { "kind": "sync", "scope": { "group_id": 0, "fragment": 1 }, "cadence": { "every_steps": 100, "phase": 25 } },
    { "kind": "sync", "scope": { "group_id": 0, "fragment": 2 }, "cadence": { "every_steps": 100, "phase": 50 } },
    { "kind": "sync", "scope": { "group_id": 0, "fragment": 3 }, "cadence": { "every_steps": 100, "phase": 75 } },
    // Commits trail syncs by 25 steps: training continues meanwhile ([PIR-18]);
    // the arrived-late global fragment is merged by the outer slot's α-mix.
    { "kind": "commit",     "scope": { "group_id": 0 }, "cadence": { "every_steps": 25 } },
    { "kind": "checkpoint", "scope": { "group_id": 0 }, "cadence": { "every_steps": 20000 } }
  ],
  "data": { "assignment": "interval", "cursor_scope": "per_group" }
}
```

The three Streaming DiLoCo ideas land in three seams, no frame change: fragment streaming =
fragment-scoped syncs with derived phases; overlap = sync→commit lag under the barrier rule
([PIR-18]), with the staleness the paper's α-mix absorbs arriving as measured version gaps
([VT-3]); 4-bit E3M0 = a `compress@1` binding on the `param_delta` class. The α-mix itself is
`outer@1` config (`rule = "alpha_mix", alpha = 0.5`) — sync policy is plan output, mixing math is
slot code (VP-5, [SL-3]).

### A.3 Factored Gossip DiLoCo (overlapped Mix1, JS-triggered Mix2)

Mix1/Mix2 with the committed-set exchange ([PIR-11]): Mix1 as non-blocking mixing of the
*previous* round's parameters (overlap via commit lag — noisier, never stale), Mix2 as a blocking
consensus event fired only when functional disagreement demands it.

```jsonc
{
  "vocab_version": 1,
  "groups": [ { "group_id": 0, "role": "replica", "fragments": 1,
                "replication": { "min": 8, "target": 24 },
                "memory": { "vram_mb": 12000, "host_mb": 24000 },
                "requirements": { "throughput_floor": "c2" } } ],
  "channels": [
    { "channel_id": 0, "class": "param_delta", "from_group": 0, "to_group": 0,   // Mix1 lane
      "plane": "consensus", "determinism": "det",
      "compress": { "slot": "compress.mix1", "config": { "quant_bits": 8 } },
      "budget": { "bytes_per_event_max": 268435456 } },
    { "channel_id": 1, "class": "param_delta", "from_group": 0, "to_group": 0,   // Mix2 lane
      "plane": "consensus", "determinism": "det",
      "compress": { "slot": "compress.mix2", "config": { "blocks": "top_disagreeing" } },
      "budget": { "bytes_per_event_max": 67108864 } }
  ],
  "states": [
    { "state_id": 0, "name": "outer_state", "class": "replicated",
      "scope": { "groups": [0] }, "dims": [1000000000], "dtype": "f32",
      "grants": ["outer"] }
  ],
  "schedule": [
    { "kind": "local_step", "scope": { "group_id": 0 }, "cadence": { "every_steps": 1 } },
    // Mix1: publish previous-round params each H; fold at a commit 20 steps later —
    // fully overlapped with local compute, temporally current, "noisy never stale".
    { "kind": "sync",   "scope": { "group_id": 0, "channel_id": 0 }, "cadence": { "every_steps": 60, "phase": 0  } },
    { "kind": "commit", "scope": { "group_id": 0 },                  "cadence": { "every_steps": 60, "phase": 20 } },
    // Mix2: blocking consensus on the freshest outer gradients, fired by the monitor's
    // committed JS-disagreement signal crossing the threshold ([PIR-17], [TM-3]).
    { "kind": "sync",   "scope": { "group_id": 0, "channel_id": 1 },
      "trigger": { "signal": "logit_js", "threshold": 13107 } },      // 0.2 in Q16.16
    { "kind": "checkpoint", "scope": { "group_id": 0 }, "cadence": { "every_steps": 20000 } }
  ],
  "data": { "assignment": "interval", "cursor_scope": "per_group" }
}
```

This instantiates the paper's *overlapped-global-averaging* Mix1 (its own reference choice) and
block-selective Mix2 as two channels with two policies in one group — the JS trigger the paper
proposes as future work is [PIR-17] + [TM-3] machinery. The paper's *pairwise gossip* mix
operators are the one part outside `vocab_version 1`: a new exchange pattern ([PIR-11], rung 3),
anticipated as an axis, arriving with its own assurance treatment (gossip leaves replicas
divergent by design, so digest equality yields to measured contraction, [CO-8]).

### A.4 Four-stage asymmetric pipeline (expressible now, transport deferred)

```jsonc
{
  "vocab_version": 1,
  "groups": [
    { "group_id": 0, "role": "stage_head",   "fragments": 1, "replication": { "min": 1, "target": 2 },
      "memory": { "vram_mb": 22000, "host_mb": 64000 }, "requirements": { "throughput_floor": "c3" } },
    { "group_id": 1, "role": "stage_body_a", "fragments": 1, "replication": { "min": 2, "target": 4 },
      "memory": { "vram_mb": 11000, "host_mb": 32000 }, "requirements": { "throughput_floor": "c2" } },
    { "group_id": 2, "role": "stage_body_b", "fragments": 1, "replication": { "min": 2, "target": 4 },
      "memory": { "vram_mb": 11000, "host_mb": 32000 }, "requirements": { "throughput_floor": "c2" } },
    { "group_id": 3, "role": "stage_tail",   "fragments": 1, "replication": { "min": 1, "target": 2 },
      "memory": { "vram_mb": 24000, "host_mb": 80000 }, "requirements": { "throughput_floor": "c3" } }
  ],
  "channels": [
    // Forward activations, subspace-compressed; backward gradients mirror 3→2→1→0.
    { "channel_id": 0, "class": "activation", "from_group": 0, "to_group": 1,
      "plane": "execution", "determinism": "native",
      "compress": { "slot": "compress.subspace", "config": { "k": 40 } },
      "budget": { "bytes_per_event_max": 2097152 } },
    // channels 1–2: 1→2, 2→3 forward; 3–5: backward — same shape, elided.
    // Within-stage replica consensus: one det param_delta channel per stage group.
    { "channel_id": 6, "class": "param_delta", "from_group": 1, "to_group": 1,
      "plane": "consensus", "determinism": "det",
      "compress": { "slot": "compress.delta", "config": { "top_k": 64, "quant_bits": 2 } },
      "budget": { "bytes_per_event_max": 20971520 } }
    // channels 7–9: groups 0, 2, 3 — same shape, elided.
  ],
  "states": [
    { "state_id": 0, "name": "subspace_basis", "class": "replicated",
      "scope": { "groups": [0, 1, 2, 3] },       // a shared protocol object: every stage reads it
      "dims": [4096, 40], "dtype": "f32",
      "grants": ["compress.subspace", "outer"] }
  ],
  "schedule": [
    { "kind": "local_step", "scope": { "group_id": 1 }, "cadence": { "every_steps": 1 } },
    { "kind": "sync",   "scope": { "group_id": 1 }, "cadence": { "every_steps": 20 } },
    { "kind": "commit", "scope": { "group_id": 1 }, "cadence": { "every_steps": 20 } }
    // per-group local_step/sync/commit for groups 0, 2, 3 and checkpoints — elided.
  ],
  "data": { "assignment": "interval", "cursor_scope": "per_group" }
}
```

Everything validates under §4.2; a v2.0 host **refuses it at admission** ([LC-2]) because no
execution-plane transport ships yet. The coordinator half — four groups, four clocks, per-group
records, per-stage rounds (VP-12) — is the same machinery as A.1. Note what is absent: no "PP"
anywhere. A scheduler sees groups, channels, and budgets (VP-3); the basis is a `replicated`
state versioned by its committing round ([PIR-12], [VT-2]).

---

## Appendix B — litmus decompositions *(informative)*

The acceptance test for the seam design: each paper from the map decomposes into rungs of the
frame-change ladder (§4.5) without touching the frame — or names, in advance, the axis it would
add. Full migration-grade decompositions land in `vhc-migration.md`.

| Paper (map §) | Decomposition into v2 seams | Rung |
|---|---|---|
| **Streaming DiLoCo** (§3.6) | fragment-scoped syncs with derived phases + commit-lag overlap + α-mix `outer@1` + 4-bit `compress@1` — Appendix A.2 | 1–2 |
| **AsyncMesh** (§3.12) | continuous sparse `param_delta` syncs (SPARTA-style slices as compressor config); install-old-average + EMA staleness correction inside `outer@1`, staleness measured by the version plane ([VT-3]); the PP axis = per-stage-group outer updates (A.4 topology) | 2 |
| **Factored Gossip** (§3.14) | Mix1 = overlapped sync/commit lag; Mix2 = JS-triggered sync on a second channel — Appendix A.3; pairwise-gossip mix operators = a new exchange pattern | 2 (gossip operators: 3, anticipated [PIR-11]) |
| **SENTINEL** (§3.13) | statistical screens at verifier positions on `activation` channels — host assurance implementation ([AL-3]) consuming monitor baselines (§12); no vocabulary change | host work, dormant until execution transports |
| **Protocol Models** (§3.10) | subspace `compress@1` on the `activation` class + declared optimizer-invariant contract ([AC-3].1) + basis as shared `replicated` state with round-versioned drift updates (A.4) | 2 (once execution transports land) |
| **SDP** (§3.16) | mask axis internal to `plan@1`; overlapping subnetworks = groups with per-role memory claims + inter-group `param_delta` channels for masked averaging; forward/backward masking bias declared via `gradient_bias` ([AC-3].2) | 2 |
| **DiPaCo** (§3.6) | paths = groups; shared modules sync over inter-group `param_delta` channels (DiLoCo on shared modules); offline per-document routing = a routed data-assignment family | 2, except `data.assignment: "routed"` = 3 (anticipated [PIR-20]) |
| **CrossPipe** (§3.15) | pipeline micro-schedules are `plan@1` schedule-pass output over the derived transfer graph; lands with execution transports; may grow schedule-event vocabulary | 2–3, dormant |

Two readings of this table matter. First, the highest-churn literature (compressors, sync
policies, outer updates — the top of §3.2) lands entirely at rungs 1–2: pushed modules and
envelope edits, zero fleet touch ([FC-1]). Second, every rung-3 entry names a vocabulary axis
this spec has already reserved a field or a rule for ([PIR-11] exchange patterns, [PIR-20]
assignment families) — the frame anticipates its own extensions, which is the map's design
target stated in §4.5.

---

## Appendix C — grounding table *(informative)*

| Source | Claim / requirement | This spec |
|---|---|---|
| map §10.1 | latent architecture: control/data planes, partitioner between, cross-cutting monitor | §3.1, §3.4, §4 |
| map §10.2 | churn inversely correlated with interface stability | §3.2 |
| map §10.3 (1) | three typed buses; axis-typed compressors; the anti-seam (Protocol Models Stmt 7.1) | VP-2, [SL-2], [PIR-7]–[PIR-9] |
| map §10.3 (2) | compressor-optimizer leak as declared capability | VP-4, §13, [PIR-14] |
| map §10.3 (3) | outer optimizer ⊕ staleness corrector = one slot | VP-5, §7.1 |
| map §10.3 (4) | trajectory monitor as shared service | VP-6, §12 |
| map §10.3 (5) | control/data clock decoupling | VP-7, §3.1, [GR-2] |
| map §10.4 | partitioner is the chassis; composition evidence; assumption contracts | VP-1, §4, §13 |
| map §10.5 | four-axis space; three derivation passes; derived-not-declared; honest caveat | §4.1, §4.2, §4.5 |
| map §3.6 (Streaming DiLoCo) | fragments, overlap, α-mix, 4-bit E3M0 | [PIR-17], [PIR-18], A.2 |
| map §3.12 (AsyncMesh) | EMA staleness correction; realized-τ inputs | §7.1, [VT-3], App. B |
| map §3.14 (Factored Gossip) | Mix1/Mix2; JS disagreement beats L2 | A.3, [TM-1] |
| map §3.13 (SENTINEL) | EMA+IQR screens, taint-and-substitute, forgiveness | [AL-3] |
| SENTINEL (Mohaghegh Dolatabadi et al.) | statistical detector with **trusted** verifiers (its Def. 2.1); verifier accountability left by the paper to future work — the gap §14 closes generically | [AL-3], §14 (motivation) |
| SZK (`sentinel-zk-swarm-v1-complete-spec.md`) | worked instantiation of §14 for the [AL-3] screen profile: proof-native tensor commitments, randomness lock, fixed-point detector semantics, lane/policy proof split, bounded-lag coverage | §14 (FUTURE design input) |
| map §3.16 / §3.6 (SDP / DiPaCo) | mask and routing axes; structural escapes | §4.1, §4.5, App. B |
| map §3.8 (DeMo) | momentum ownership; sign-family updates | [AC-3].3, [PIR-14] |
| VHC ref §5.3, P-1 | planes with latency classes, never merged | VP-7, §3.1 |
| VHC ref P-4 | state separation, no default aliasing | VP-9, §4.2.3 |
| VHC ref P-6 | classify errors by channel | VP-8, [PIR-8], [VT-2] |
| VHC ref P-8 | churn is state logistics; transactional residuals | VP-10, §7.3, §4.4(5) |
| VHC ref P-10 | provenance ≠ correctness | VP-11, [AL-4] |
| VHC ref P-11, AV-5 | stage round as universal primitive; no global step | VP-12, §6.1, §8 |
| VHC ref §7.4 C-1..C-8 | consensus-operator requirements | §7.2 |
| VHC ref §9.1–9.2 | message envelope, version tuple, causal DAG | §8 |
| VHC ref §9.4 | five-commit-point transactional residual protocol | §7.3 |
| VHC ref EX-5 | delay consumed as measured input | [VT-3] |
| VHC ref §4.2 D2 | fleet tiers; per-role memory inequality; islands | §10 (tiers derived, not declared) |
| VHC ref §10.2–10.7 | layered assurance; screens; quarantine before penalty | §11 |
| VHC ref §10.5 (ℛ) | drift-response calibration surface | [TM-6] |
| v1 spec §4.3 | the envelope/experiment seam rule | §5, [EV2-8] |
| v1 spec §5.1 | state classes `local`/`replicated`; module monolith | §4.2.3, §1.1 |
| v1 spec §5.6 / ABI v1 §5.9 | det lane, digests, fp32 masters | VP-8, [EB-8], §11 |
| v1 spec §6.1 | envelope freeze/hash/sign; requirements vocabulary | §5, [PIR-6] |
| v1 spec §6.2–§6.5 | tick, assignment, round protocol, I1–I6, admission | §6 |
| v1 spec §7, §8 | transports, artifact plane | §1.3, [AR-4] |
| P3 ledger Merge-1 | D5 fat worker; D6 NVRTC on demand; `select_backend()`; CUDA `DeviceLimits`; det-digest tripwire | §9.3, §9.4, [EB-8] |
| P3 trunk `live.rs` (ledger entry pending) | pinned device thread; `StreamId` thread-locals; pool pressure; checkpoint-staging peak | §9.1, §9.2 |
| `swarm-uma-platform-findings.md` | UMA budgets: vram + 0.9·GTT; joint pool check | [EB-9] |
| `swarm-windows-vram-design.md` | DXGI/D3D12 budget sources; UMA flag | [EB-9] |
| `swarm-macos-uma-findings.md` | Metal working-set budget; joint pool check | [EB-9] |

*— End of specification —*
