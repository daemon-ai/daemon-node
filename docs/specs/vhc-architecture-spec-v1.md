# The Virtual Heterogeneous GPU Cluster (VHC)

## Canonical Architecture Specification — Version 1.0

**Date:** 2026-07-13
**Status:** Consolidated reference. This document is the canonical record of an extended multi-party technical review, grounded in the primary literature and the public production record through July 2026. It supersedes the individual exchange documents it consolidates.

**Normative language.** MUST / MUST NOT / SHOULD / MAY are used in the RFC-2119 sense. Sections marked *(informative)* carry evidence and rationale. Requirement lists are normative unless stated otherwise.

---

## Part I — Foundations

## 1. Problem statement and scope

### 1.1 Objective

Specify a training system that pools geographically distributed, heterogeneous, intermittently available, and partially untrusted GPUs into a single logical cluster capable of training neural networks **larger than any participant can hold**, at a quality-per-unit-compute cost approaching centralized training, with integrity guarantees compatible with open participation. This is the system the DisTrO preliminary report named as the goal of combining its optimizer with FSDP/SWARM-class sharding: *the first virtual and fully heterogeneous GPU cluster* [1].

### 1.2 The five heterogeneity axes

Prior systems each solved a subset; this specification addresses the cross-product.

| Axis | Range in practice | Primary mitigation |
|---|---|---|
| H1 — Memory | 16 GB consumer → 96 GB workstation → 80 GB+ datacenter | Tiered fleet (§4.2), asymmetric stages (§6.1), state tiering (§7.3) |
| H2 — Throughput | ~8 TFLOPs (T4-class) → 150+ TFLOPs | Asymmetric partition, throughput-weighted routing (§7.6) |
| H3 — Network | 20 Mbps residential uplink → 1 Gbps+; 10–250 ms RTT | Boundary compression (§6.2), plane separation (§5.3), bandwidth model (§8) |
| H4 — Availability | Minutes–hours sessions; continuous churn | Stage replication, partial collectives, transactional state (§7.4, §9.4) |
| H5 — Trust | Operator-run → staked → anonymous | Assurance plane (§10), tier-scoped permissionless perimeter (§4.2) |

### 1.3 Success metrics

A run MUST declare one primary metric before launch:

1. **Economic:** tokens per dollar versus a matched centralized control, under the viability inequality (§13.6).
2. **Capability:** final evaluations at equalized useful FLOPs versus a matched control.
3. **Sovereignty:** censorship-resistance and access to compute that markets cannot otherwise aggregate — accepting a declared efficiency tax.

### 1.4 Status at a glance

| Layer | Status |
|---|---|
| WAN pipeline sharding with compressed boundaries | **Validated** in production (8B, ~500B tokens, live churn) |
| Federated full-replica training with compressed optimizer sync | **Validated** in production (post-training at 36B; pretraining at 40B scale) |
| Asymmetric stages, partial collectives, drift + periodic healing | **Validated** |
| Fully asynchronous PP + sparse replica consensus | **Prototyped** (≤1B academically; deployed in one production run) |
| Statistical stage-integrity screening | **Prototyped** (Sentinel; not deployed) |
| Delay-aware optimization under *stochastic* delay | **Open** (P1) |
| Churn-native compressed stage consensus with contraction | **Open** (P2) |
| Subspace scaling law and basis governance | **Open** (P3) |
| Cheap permissionless verification of stage computation | **Open** (P4) |
| WAN-trainable coarse expert overlay | **Open** (P5) |

The one-line status: **the sharding architecture is settled; the open work is five algorithms at its interfaces** (§11), plus integration, scheduling, and economics.

### 1.5 Non-goals

This document does not specify an inference-serving system (touched only where ownership mode constrains it), a token or governance design (only the mechanism-design requirements the training protocol imposes), or any single organization's product. It is architecture-normative and implementation-agnostic.

---

## 2. Evidence base *(informative)*

### 2.1 Two validated regimes, one missing intersection

Production has validated **two complementary halves** of the eventual system:

```
Regime A (Pluralis: Node0, Agora)          Regime B (Nous: Psyche)
────────────────────────────────           ─────────────────────────────
WAN pipeline sharding                      complete logical replicas
subspace-compressed boundaries             local TP/FSDP inside replicas
local per-replica optimizers               compressed WAN optimizer sync
sparse periodic replica consensus          (DisTrO/DeMo family)
elastic membership, partial collectives    epoch membership, witnesses,
regional prosumer fleet                    on-chain coordination
```

The missing intersection — the subject of this specification's open problems — is:

> **WAN pipeline parallelism × compressed stage-replica consensus × stochastic delay × permissionless verification.**

The DisTrO report itself listed adapting FSDP/SWARM-class sharding to work with its optimizer as future work [1]; no published system has closed the intersection.

### 2.2 Production ledger

**Node0-7.5B (Pluralis, 2025→Jan 2026 write-up)** [10][17]: first public model-parallel pretraining over the internet; 7.5B OLMo-style, 36B tokens, 3 weeks, 303 participants, ~1.7k GPUs, 198 cities, 16 GB+ permissionless. Stack: Hivemind/SWARM foundation; SSN boundary compression; PowerSGD (64×) for in-stage gradient sync; SPARTA sparse parameter healing every 5 steps; fault-tolerant *partial* butterfly all-reduce (rounds observed succeeding with as little as ~10% of peers); joins restricted to the first 10% of an accumulation round, ≤5 rounds of staleness, ~2.4 GB of state per layer; per-stage timestamp checkpoints (no global step); stage-wise clipping at 1/√(num_stages); Moshpit-style multi-round averaging **tried and rejected** at realistic group sizes; QK-norm reordering broke SSN losslessness until RMSNorm scales were frozen; verification limited to code-integrity checks.

**Agora / Pluralis-8B (2026, complete)** [11][23]: first *asynchronous* production WAN pipeline. 7 asymmetric stages — head = embeddings + 6 layers, five body stages of 4 layers (~0.9B params each), tail = 6 layers + output projection (~2.7B params) — head and tail staffed exclusively by operator nodes on 96 GB-class cards. Admission: 24 GB GPU + **80 GB host RAM per GPU** + 200 Mbps + North America placement with a <80 ms latency gate; 104-node capacity cap with a join queue (~100 requests/min). Per-replica local optimizer steps; stage-local lockstep rounds via a progress tracker; **no per-step gradient all-reduce**; AsyncMesh sparse averaging — 5% rotating non-overlapping parameter slices every 20 local steps [11]. FP32 params + grads on GPU; FP32 Adam moments in host RAM. Run outcome: order-500B tokens over multiple weeks (final accounting pending a promised technical report — the public 170k tok/s steady-throughput claim and the 500B/7-week figures do not yet reconcile); 176 contributors; 669 GPU joins / 607 departures with throughput flat; ~20% MFU; self-reported ~**1.5× tokens-per-unit-compute tax** versus centralized and ~15× faster than Megatron-LM in the WAN setting. Incentive v0: PFLOPs-processed + uptime-credit leaderboard.

**Psyche (Nous)** [15][16]: Consilience 40B — dense MLA architecture sized to fit one large local node group, 20T-token corpus target, described by Nous as the largest distributed pretraining run (full-token completion not publicly confirmed). Hermes 4.3 (36B): first production model **post-trained entirely on Psyche** — run twice, once centralized (FSDP+AdamW) and once on Psyche (TP + DisTrO, 24 nodes, 144k tok/s, P2P communication fully overlapped), with the Psyche version outperforming the centralized twin on downstream evals. Practical verification explicitly open (floating-point nondeterminism, TP ordering, compressed-result variability).

### 2.3 Component ledger

Each component was validated in a specific regime; the third column is the assumption that breaks when components are composed. This table is the origin of most of Part III.

| Component | Validated regime | Assumption that breaks in composition |
|---|---|---|
| DeMo / DisTrO [1][2] | Synchronous DDP, identical replicas, IID shards, ≤1.2B, per-step sync; 85×/44×/857× comm reduction | Fresh gradients from identical weights; download scales with worker count (AllGather); residual survives membership; Theorem needs β=O(1/√T), practice needs β=0.999 |
| Protocol Models / SSN [3] | Synchronous PP, 8B/32 stages/4 regions at 60–350 Mbps ≈ centralized; k=40 at d=4096 (~100×); 13× slower uncompressed | Shared basis U_k is versioned global state (Grassmann-updated ~every 500 iters, broadcast); replicas weight-identical; architecture closure invariants (QK-norm episode) |
| Nesterov async PP [4] | Fixed per-stage delay; ≤1B; degradation visible by ~24 stages; weight-stash memory or the weaker no-stash variant (the only one SWARM-compatible) | Delay is a per-microbatch random variable under routed heterogeneous swarms; convex/smooth fixed-τ theory |
| Subspace context parallelism [5] | 800M, 132K context, 8 GPUs, 300 Mbps ≈ 100 Gbps; >95% KV compression; droppable heads | Orthogonal axis; a second learned-basis family to version |
| SWARM / Hivemind [6] | Stochastic wiring, adaptive rebalancing, preemptible T4s ≤200 Mbps | Square-cube feasibility favors *wide* stages; 16 GB cards force deep-narrow; replicas assumed fungible |
| Moshpit averaging [7] | Exponential contraction of disagreement, 512–1024 peers, groups ~32 | Multi-round + large-group assumptions; **empirically rejected in-stage** at real group sizes |
| Tasklet scheduling [8] | 4.8× from placement, 64 GPUs / 8 regions, static membership | Membership churns per minute; measurements adversarial |
| Sentinel [12] | EMA+IQR statistical screening at verifier-trainers; 128-worker SWARM, 37.5% malicious/stage at 15% collusion, integrity held | Trusted verifiers; first/last two stages assumed honest by construction; statistical (detection floor), not cryptographic |
| Factored Gossip DiLoCo [13] | Non-blocking mixing overlapped with compute + small blocking agreement step; ≤~1B, small groups; JS output-divergence tracks instability better than L2 parameter distance | Scale; partial-quorum/churn semantics unproven |
| UPM [14] | Time-varying invertible transforms → shards captured across epochs cannot be assembled; 0.5–1B | Constrains verification to online / version-scoped / committee modes |

### 2.4 Epistemic stance

Two composition principles govern this document:

1. **Parity does not compose.** Each component's "matches centralized" claim holds against its own baseline under its own assumptions. The composed system runs all approximations simultaneously on shifted distributions; the experimental unit is the stack (§13), not the parts.
2. **Theory bounds sanity; empirics carry the weight.** Every layer operates outside its formal coverage: DeMo's rate requires vanishing momentum while practice requires β=0.999 [2]; the delay-corrected pipeline result is convex/smooth/fixed-τ [4]; PowerSGD's proofs do not cover partial quorums; Sentinel's negligible-impact theorem holds under its stated assumptions [12]. Consequently this specification mandates pre-declared falsification criteria (§14) and measured calibration surfaces (§10.5) over proof-backed confidence.

---

## 3. Design principles (normative)

**P-1 · Three planes, never merged.** Critical-path execution, replica consensus, and control/assurance are separate communication planes with different latency classes and failure semantics (§5.3). Merging them into one collective or one global step recreates the synchronization bottleneck the architecture exists to remove.

**P-2 · Point-to-point beats collectives on the WAN.** The WAN critical path carries only stage-boundary tensors between adjacent islands. Latency-sensitive collectives (tensor parallelism, ZeRO-3 gathers, token-level expert all-to-all) are confined to fast local islands.

**P-3 · Compression is load-bearing, not an optimization.** Consumer uplinks fall short of the compute-bound boundary bandwidth by 10–100× (§8.1). The ~100× subspace compression is precisely the gap-closer; systems MUST budget as if its removal ends the run — because it does (13× measured slowdown uncompressed [3]).

**P-4 · One state, one meaning; storage tiered.** Local optimizer moments, consensus/error-feedback residuals, and staleness estimates are semantically distinct states and MUST NOT be aliased by default. The memory bill is paid by precision reduction and host-RAM/NVMe placement of slow-cadence states, and MUST be reported (VRAM + host RAM) alongside any optimizer claim (§7.3). Partial aliasing is permitted only as a declared low-memory profile.

**P-5 · Shared transforms are protocol objects.** The SSN basis, any consensus transform, and compression schemas are versioned, committed, governed protocol state — not implementation details. Mismatched versions produce garbage, not noise (§9.3).

**P-6 · Classify errors by channel.** (a) *Invariant violations* (basis mismatch, closure-breaking ops) → hard protocol failure; prevented by versioning and architecture linting. (b) *Trajectory divergence* (replica drift; correct transmission of a different function) → gradient variance; managed by the healing controller. (c) *Class restriction* (the constrained hypothesis space) → a possibly scale-dependent constant; measured by the scaling gate. Different machinery for each; never conflate.

**P-7 · Replica fungibility is a budget, not a property.** Same-stage replicas drift by design. The tolerable drift is a measured function (ℛ, §10.5) that jointly calibrates healing frequency, quarantine, routing, and verification tolerance.

**P-8 · Churn is state logistics.** The binding cost of membership change is moving weights, optimizer state, residuals, and version history — not lost FLOPs. Error-feedback state is optimizer-trajectory state and MUST have transactional (exactly-once) semantics across every failure point (§9.4). Useful fleet time = session − download − resync − failed-collective recovery.

**P-9 · Pipeline failures are structural, not additive.** In data parallelism a bad update is statistical noise; in pipeline parallelism a bad stage corrupts everything downstream and errors become path-dependent. Assurance therefore lives at stage boundaries (§10), not only at the update aggregate.

**P-10 · Provenance is not correctness.** Signatures and Merkle commitments prove who committed which bytes when; they never prove the bytes came from the required computation. Correctness requires screening, algebraic checks, or replay (§10.2–10.6).

**P-11 · The stage round is the universal primitive.** The stage-local lockstep optimizer round serves simultaneously as logical clock, version boundary, admission window, billing unit, healing-slice boundary, and audit-sampling unit. No global scalar step exists or is needed.

**P-12 · Declare the trust perimeter.** Anchor nodes currently carry the memory-heaviest stages, the stability-critical stages, and the security root simultaneously (§2.2; Sentinel's protected-boundary assumption). This is acceptable if declared and measured (anchor-criticality accounting, §13.6); "permissionless" describes exactly the tiers where it holds.

**P-13 · Measure the stack.** Component loss-curve overlap is inadmissible as evidence for the composed system. Acceptance metrics are tokens-to-target-loss per byte, per wall-clock second, and per dollar, under recorded churn/delay traces, against matched controls.

**P-14 · Falsify before you scale.** Every run pre-declares the observations that would kill its branch (§14), and the fallback architecture is specified in advance.

### 3.1 Prohibited on the WAN critical path (normative)

| ID | Prohibited | Reason / evidence |
|---|---|---|
| AV-1 | Tensor parallelism across WAN peers | Per-layer collectives multiply latency hundreds of times per step; confined to islands [8] |
| AV-2 | ZeRO-3 / FSDP across WAN peers | Parameter gathering on the execution critical path; churn breaks the process group; local-island use only |
| AV-3 | Token-level expert dispatch across WAN | Per-layer all-to-all makes every MoE layer a geographic sync point; route at document/path granularity (§6.4) |
| AV-4 | Remote parameter servers / demand paging over WAN | Puts transient connectivity on the critical path; offload to local RAM/NVMe instead |
| AV-5 | A global scalar training step | Impossible under asynchronous stages; use stage rounds + causal message DAG (§9.2) |
| AV-6 | Multi-round shuffled-group averaging within stages | Rejected in production: too slow, assumes 50–100-node groups where stages have far fewer [10] |
| AV-7 | Merging the three communication planes | P-1 |
| AV-8 | Treating replica-sync compression (DeMo/DisTrO) as memory sharding | It reduces sync traffic between full copies of a shard; it does not fit a model into smaller VRAM [1][2] |

---

## Part II — Architecture

## 4. Pre-run decisions

Four decisions MUST be made, in order, before system design begins. Everything downstream inherits from them.

### 4.1 D1 — Ownership mode

| Mode | Definition | Natural assurance model | Checkpoint semantics |
|---|---|---|---|
| **Canonical checkpoint** | An authorized process can assemble and publish full weights | Offline independent replay; full-model audits | Global assembly path required; per-stage snapshots exported and merged |
| **Distributed custody** | No participant holds full weights during operation; assembly is possible but authorized | Stage-local replay; authorized checkpoint assembly | Per-stage checkpoints; export path (e.g., to centralized frameworks) defined |
| **Unextractable (UPM)** | Time-varying invertible transforms prevent coherent assembly of shards captured across epochs [14] | Online, version-scoped, or committee verification only; detached indefinite offline replay is foreclosed | Protocol-resident state; no conventional checkpoint exists |

The mode fixes what contributors are paid for, who can verify, and who can serve inference. All three modes have real production or research artifacts; a run MUST pick one and design assurance for it — the UPM branch's verification constraint is a declared cost, not a discovered one.

### 4.2 D2 — Fleet contract (three tiers)

| Tier | Hardware class | Trust | Roles |
|---|---|---|---|
| **Anchor** | 48–96 GB+, stable, high host RAM; operator-run or heavily staked | Highest; explicitly inside the trust perimeter | Embedding+head stage, tail+projection stage, checkpoint custody, basis governance, verifier coordination, seeds |
| **Body** | 24–48 GB + ≥80 GB host RAM + ≥200 Mbps; staked; regional latency domain | Staked; the current permissionless frontier | Body pipeline stages, stage-replica groups, ordinary trainer traffic |
| **Edge** | 16 GB-class, global, intermittent | Lightly staked or reputational | Expert replicas that fit, audit replay of compact stages, transcript validation, redundant statistic computation, data services |

Requirements:
- **[D2-1]** Role admission is an explicit memory inequality per role: `M_weights + M_grads + M_optimizer + M_activations + M_stash + M_audit ≤ M_tier`. A 16 GB node MUST NOT be promised "expert hosting" or "replay" generically — only roles whose inequality it satisfies (it cannot replay a 2.7B-parameter tail).
- **[D2-2]** An island (multi-GPU machine, rack, or campus group) registers as **one logical worker**; TP/FSDP/ZeRO/CPU-offload inside it are local implementation details invisible to the WAN protocol.
- **[D2-3]** The run MUST publish **anchor-criticality accounting** (§13.6): the fraction of FLOPs, checkpoint custody, and assurance authority held by operator/anchor nodes. The permissionless perimeter is defined as the tiers where it does not apply, and MUST be stated as such.
- **[D2-4]** The edge tier is the membership on-ramp: nodes earn stake/reputation there and graduate to body-tier slots.

### 4.3 D3 — Threat model tier

Declare one: (a) *honest-but-unreliable* (crash/churn only), (b) *rational-cheating* (cost-minimizing shortcuts: fabricated cheap gradients, uptime farming, staleness gaming), (c) *Byzantine-colluding* (coordinated corruption, basis poisoning, verifier collusion). The assurance plane (§10) is dimensioned to the declared tier; (a) permits the v1 profile (§12); (c) requires P4 solved.

### 4.4 D4 — Primary success metric

One of §1.3, with the matched-control methodology fixed before launch (§13.5).

### 4.5 D5 — Trunk sizing

Chosen from the capacity worksheet (Appendix A): stage size from the tier memory inequality; stage count from the delay-tolerance frontier (currently degradation by ~24 stages [4]); first serious runs SHOULD target an 8–15B dense trunk, with 20–50B as the current engineering envelope (not a theorem — see §6.1.3) and capacity beyond it delegated to the expert overlay (§6.4).

---

## 5. System overview

### 5.1 Topology

```
                        CONTROL / ASSURANCE PLANE (§9, §10)
        membership · data leases · versions · basis governance · commitments
        audits · quarantine · rewards            (low-bandwidth, async, signed)
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                             DENSE SSN TRUNK                                │
│                                                                            │
│  ANCHOR              BODY STAGE GROUPS                       ANCHOR        │
│  head island   ┌────────────────────┐   ┌─────────┐        tail island    │
│  embed+layers  │ Stage s            │   │ Stage   │        layers+proj    │
│  ───────────►  │ replicas A B C ... │──►│  s+1    │──► ... ───────────►   │
│                │ (drift + healing)  │   │         │                        │
│                └────────────────────┘   └─────────┘                        │
│   WAN critical path: k-dim activation & activation-gradient coefficients   │
│   Within a stage: local optimizers + periodic partial consensus (§7.4)     │
└────────────────────────────────────────────────────────────────────────────┘
                    │                                   │
                    ▼                                   ▼
          Expert group A (island/edge)        Expert group B (island/edge)
          replicas, doc/path-routed           replicas, doc/path-routed
```

### 5.2 Roles

**Workers** (GPU): host one stage's layers (or one expert); compute forward/backward; participate in stage consensus. **Trainers/routers** (CPU): hold the network map; route microbatches by measured throughput/latency/queue depth; natural hosts for verifier duty [12]. **Seeds:** DHT bootstrap (routing only, no model data). **Authorizer:** admission per D2/D3. **Verifiers:** §10; may be trainer-colocated, anchor-hosted, or independent per ownership mode. **Health/metrics:** scrapes the DHT. DHT instances SHOULD be partitioned per stage to avoid cross-stage bottlenecks [10].

### 5.3 The three communication planes

| Plane | Traffic | Latency class | Failure semantics |
|---|---|---|---|
| **Critical-path execution** | Forward activation coefficients; backward activation-gradient coefficients; version tags | Milliseconds–seconds; cannot wait indefinitely | Reroute to another replica; bounded-staleness rejection |
| **Replica consensus** | Sparse parameter slices; compressed deltas; drift measurements | Seconds–minutes; overlapped with training | Partial quorums are normal; late contributions fold into later rounds |
| **Control / assurance** | Membership, leases, versions, manifests, commitments, audits, rewards | Seconds–hours; fully asynchronous | Retried; append-only where committed |

---

## 6. Capacity plane

### 6.1 Dense trunk

**6.1.1 Asymmetric partition.** Stages carry unequal layer counts sized to each island's memory inequality and measured throughput; fat head (embeddings + extra layers) and fat tail (layers + output projection) are pinned to anchors — the deployed reference layout is head = embed+6L, 5 × body = 4L (~0.9B each), tail = 6L+proj (~2.7B) at 8B total [11]. Input/output embeddings MUST be untied (tying creates a WAN weight-sharing dependency between the first and last stages). Placement avoids adjacent stages on poorly connected pairs; the placement/replication problem is solved **online** (§7.6).

**6.1.2 Capacity and replication.** Dense capacity grows with *distinct* stages: `P_dense = Σ_s P_s`, with distinct stages bounded by `≲ N/r` for replication factor r. Replicas add throughput and availability, not parameters. r per stage is set by the session-length distribution and state-transfer time (P-8), not by throughput alone; scarce/slow/unreliable stages get more replicas; r ≥ 2 for every stage on the critical path.

**6.1.3 The envelope.** With ~6 bytes/param optimistic persistent state (bf16 weights+grads, 8-bit moments) to ~8–16 bytes/param as deployed (fp32 + host-offloaded moments; ~10 B/param observed persistent at Node0), a 16–24 GB card hosts ~0.7–2.5B stage parameters; at 20–32 stages before staleness degradation, the dense envelope is **~15–50B** — an engineering envelope, movable by more stages (needs P1), bigger cards, islands, offload, lower-precision state, or bounded-staleness execution; not a theorem. Beyond it, sparse capacity (§6.4) is the economically dominant route.

**6.1.4 Architecture linting.** Any architectural change to the trunk MUST pass an SSN-closure lint: every operation at a block output must preserve the shared-subspace invariant (the QK-norm/RMSNorm-reordering episode is the canonical counterexample and its fix — freezing the norm's elementwise scale — the canonical remedy) [10].

### 6.2 Boundary compression (Subspace Networks)

**Mechanism (informative).** Projection-matrix row spaces are constrained to a shared low-dimensional subspace S = Col(U_k), U_k ∈ R^{d×k} orthonormal; block outputs then live in S, so boundaries transmit `X·U_k ∈ R^{b×n×k}` instead of `X ∈ R^{b×n×d}`; backward activation gradients compress into the same subspace losslessly; a modified AdamW keeps constrained weights in S; the high-rank embedding component is a fixed table replicated to all nodes once; U_k drifts via infrequent Grassmann-manifold steps (~every 500 iterations in the reference) and is rebroadcast [3]. Reference point: k = 40 at d = 4096 ≈ 100×.

**Normative:**
- **[SSN-1]** U_k and every compression schema are versioned protocol objects with the lifecycle of §9.3; a receiver MUST reject coefficients whose basis version it does not hold.
- **[SSN-2]** Reconstruction exactness is conditional on invariants (P-6a). Replica drift does **not** break reconstruction — the receiver exactly recovers whatever the drifted sender computed; drift is managed as trajectory divergence (P-6b), not transmission error.
- **[SSN-3]** The class-restriction cost (P-6c) MUST pass the scaling gate (§13.3, P3) before any compute-optimal budget is committed to the SSN class.
- **[SSN-4]** Long-context runs MAY add subspace context parallelism (mixture-of-subspaces KV compression, >95% at 100K+ context [5]); its bases are versioned under the same lifecycle.

### 6.3 What crosses each boundary

Per microbatch, each direction: k-dim coefficients + the message envelope of §9.1. Nothing else. Raw activations cross a WAN boundary only inside a transmit-on-challenge audit (§10.4).

### 6.4 Sparse capacity: the coarse expert overlay

Total capacity beyond the trunk comes from modular experts:

`P_total = P_trunk + Σ_e P_e    P_active(x) = P_trunk + Σ_{e∈A(x)} P_e`

- **[EXP-1]** Routing granularity is a document, sequence, task, or path — never per-token per-layer across the WAN (AV-3). A routed unit executes its expert(s) on one island end-to-end.
- **[EXP-2]** Within an expert's replica group, the data plane MUST perform deterministic, balanced, randomized assignment of routed samples across replicas. This restores the i.i.d.-per-conditional-distribution assumption the consensus operator requires; without it, replicas see different conditional streams and the consensus assumptions break. The conditional stream remains nonstationary while the router trains — a declared residual gap.
- **[EXP-3]** Expert replica groups synchronize with the same consensus operator as trunk stages (§7.4); this is the operator's most favorable regime (full-replica, conditional-IID).
- **[EXP-4]** Placement/replication follows popularity and island capability; migration is incremental and resumable (weights, optimizer state, residuals, router metadata, version history). Experts are the edge tier's primary training role and the network's onboarding path (D2-4).
- **[EXP-5]** Router training under WAN delay is open problem P5; until resolved, overlays SHOULD run with slow outer-loop or periodically frozen routers.

---

## 7. Execution plane

### 7.1 Stage rounds (the universal primitive)

A **stage round** is the stage-local lockstep event triggered when the stage's aggregate processed-sample count crosses its target: participating replicas take an optimizer step together; the round increments the stage's logical clock. Per P-11 the round is simultaneously: version boundary (weight version = round), admission window (joins land in the first ~10% of a round), billing unit (attested work per round), healing boundary (consensus slices keyed to rounds), and audit-sampling unit. Stages advance their rounds independently; inter-stage execution is fully asynchronous; there is no global step (AV-5).

### 7.2 Local optimizer (per replica)

- **[EX-1]** Delay-aware momentum optimizer of the NAG/NAdam family: gradient term discounted by (1−γ) so the look-ahead acts as delay correction; reference configuration NAdam with β₁ ≈ 0.99 [4].
- **[EX-2]** Stage-dependent schedules: learning rate discounted and momentum increased toward earlier stages (larger delay); reference: momentum 0.9→0.99 tail→head.
- **[EX-3]** Weight stashing (exact backprop against forward-time weights) costs τ_stage weight copies and is incompatible with stochastic replica routing; the no-stash variant with [EX-2] is the SWARM-compatible default [4]. Stash memory, if used, MUST be host-offloaded and counted in the D2 inequality.
- **[EX-4]** Gradient clipping is stage-wise; interim rule: per-stage threshold 1/√S with tail threshold 5/√S (empirically matched to global clipping at Node0 [10]) until P6 supplies a principled rule.
- **[EX-5]** Delay is consumed as a **measured input**: every gradient contribution carries the weight version it was computed against, so realized τ is known per contribution (feeds P1; enables staleness-weighted acceptance and bounded-staleness rejection as the v1 fallback).

### 7.3 Optimizer-state separation and tiering (P-4)

| State | Meaning | Precision | Residence | Cadence |
|---|---|---|---|---|
| m, v (local moments) | Local optimizer dynamics | FP32 (m hot; v MAY be host-resident, prefetched) | GPU / host | Every microbatch |
| r (consensus residual) | Untransmitted error-feedback for replica sync | 8–16-bit | Host RAM | Every consensus round (~20 steps) |
| ẑ (staleness estimate) | Replica/update delay EMA | Scalar or per-tensor | GPU | Continuous |
| Audit/version state | Manifests, commitments, logs | — | Host/disk | Per round |

VRAM + host-RAM footprints are first-class scheduling inputs and reporting requirements. Rationale: DeMo's aliasing of momentum-as-residual was a real memory optimization [2]; separation re-introduces that cost, which the deployed system paid via the 80 GB host-RAM floor [11]. A partially aliased buffer is a permitted declared low-memory profile for the edge tier, never the default.

### 7.4 The stage consensus operator (interface; algorithm = P2)

Signature, executed per consensus round over the responding quorum Q of a stage's replicas:

```
CONSENSUS( round ρ, quorum Q, { (θ_i, δ_i, r_i, n_i, v_i) : i ∈ Q } )
      →  { (θ_i′, r_i′) : i ∈ Q }   +   defined semantics for i ∉ Q
θ: parameters (or slice)   δ: compressed delta   r: residual
n: samples processed       v: weight version
```

Normative requirements (any candidate MUST satisfy all eight):

- **[C-1] Partial-quorum semantics.** A mathematically defined result for any Q above a minimum; ten of thirty responding is a normal round, not a failure.
- **[C-2] Exactly-once residuals.** A failed or ambiguous round MUST NOT both subtract and later retransmit the same residual; commit/rollback points are enumerated (§9.4). (Production precedent: the transactional PowerSGD error-buffer fix [10].)
- **[C-3] Late-message path.** A delayed contribution has a defined route into a subsequent round (error-feedback residuals give this natively: unsent information persists locally and retries).
- **[C-4] Contribution weighting.** Updates weighted by n_i; 10,000 processed tokens ≠ 1,000.
- **[C-5] Join/leave semantics.** Replacement policy for weights, moments, residuals, and drift state; a departed replica's residual is trajectory state — zero-initializing a successor is a declared perturbation; residual synthesis from peers is a P2 research item.
- **[C-6] Sublinear aggregate cost.** Download MUST NOT scale linearly with replica count (DeMo's flat AllGather does [2]); hierarchical/tree/regional aggregation is required at scale — the aggregation is linear before any update nonlinearity, so sum-trees are sound.
- **[C-7] Adaptive schedule.** Period H, slice fraction q, and compression budget k are set by the drift controller (§7.5), not fixed constants.
- **[C-8] Contraction.** Measured inter-replica disagreement MUST contract: `E[D_{t+1}] ≤ ρ·E[D_t] + σ_comp` with ρ < 1 under partial, weighted, compressed, rotating rounds. Requirements C-1..C-7 can all hold while drift random-walks upward; C-8 is what keeps long-horizon runs off the cliff, and the measured ρ̂ per unit bandwidth is a primary tournament metric (§13.2).

**v1 baseline instantiation:** rotating sparse slice averaging (reference: 5% every 20 local steps, non-overlapping slices, seed-fixed partitions with a slack window [10][11]) over fault-tolerant single-round partial collectives, healing interleaved with training. **Challenger candidates:** factored gossip (non-blocking mixing + small blocking agreement [13]); DeMo-derived compressed outer deltas with a separate transactional residual [2]; PowerSGD with transactional error feedback [18]. Optimizer moments are NOT averaged by default (slow-moving, drift-robust; saves bandwidth [10]).

### 7.5 Drift controller

`(H, q, k)_{t+1} = π( D_t, bandwidth, failure rate )`, where D is the multivariate disagreement vector (d_θ, d_activation, d_gradient, d_logit-JS) — output-distribution divergence is the leading indicator [13]. The controller sets healing frequency, triggers quarantine and full resynchronization, and bounds the drift downstream stages tolerate. Its calibration surface is ℛ (§10.5). Until ℛ is measured, fixed-schedule operation (v1) is permitted with conservative settings.

### 7.6 Routing, placement, replication (online)

- **[EX-6]** Trainers route microbatches by measured effective throughput (batch round-trip), queue depth, and reliability; new joiners enter at the **tail** of the priority queue (head-insertion measurably degrades system throughput under churn [10]).
- **[EX-7]** Placement/replication is a continuous controller: `min max_s ( T_s^compute + T_s^network + T_s^queue + T_s^failure )` subject to the D2 memory inequalities, reliability, and state-transfer cost — the tasklet problem [8] made online. Measurements feeding it MUST derive from attested execution outcomes, never self-reported specs (§10.9).
- **[EX-8]** Eviction on repeated failed collectives (reference: two consecutive) with quarantine before penalty (§10.7).

### 7.7 Membership and state transfer

- **[EX-9]** Join protocol: wait for round start (<10% progress) → download state (base snapshot + versioned deltas; chunked, resumable, multi-peer) → confirm round; staleness allowance ≤ L rounds (reference L = 5), else re-queue [10].
- **[EX-10]** Warm spares per scarce stage, fed by continuous delta streams; promotion MUST NOT depend on a single live peer (erasure-code stage state across the replica group).
- **[EX-11]** Departure: which optimizer state may be reset and what is preserved is part of the run contract (C-5); state-transfer traffic is scheduled off the consensus windows (state sharing disabled during all-reduce phases [10]).

---

## 8. Bandwidth model

### 8.1 The governing arithmetic *(informative)*

Per token per boundary, bf16 forward + backward activations cost ≈ 4d bytes; a stage of k_L layers computes ≈ 72·k_L·d² FLOPs per token (≈6 FLOPs per parameter, ≈12d² params/layer). The compute-to-communication ratio is therefore ≈ **18·k_L·d FLOP/byte**. At d = 4096, k_L = 4: ≈ 0.3 MFLOP/byte, so a card sustaining 25 effective TFLOPs needs ≈ 85 MB/s ≈ 0.7 Gbps of boundary bandwidth to stay compute-bound — 10–100× above residential uplinks. A ~100× subspace compression brings this to single-digit Mbps, which is why P-3 holds. (Deployed sanity check: a 67 MB fp32 activation ≈ 5 s per hop at 100 Mbps uncompressed, ≈ 0.05 s compressed [10].) The ratio *improves* with width and layers-per-stage — the square-cube effect — which is exactly what small-VRAM fleets work against by forcing deep-narrow partitions; this is the structural argument behind the tier floors in D2 and the envelope in §6.1.3. Full tables in Appendix A.

### 8.2 The per-edge budget

Three compressed streams share every WAN edge:

1. **Activation fidelity** — subspace dimension k (per boundary).
2. **Consensus freshness** — slice fraction q, period H, delta compression (per stage group).
3. **Assurance** — audit sampling rate and transcript transport (per stage).

- **[BW-1]** These draw on one budget and MUST be accounted jointly per edge; v1 fixes them statically; the online rate-distortion allocator that shifts bits among them as the binding constraint moves through the run is open problem P7. Sensitivity is not constant over training — late training near low loss tolerates less of everything.
- **[BW-2]** Boundary transport SHOULD support multipath and credit-based flow control; consumer links exhibit heavy-tailed jitter, and per-route p99 governs pipeline throughput.

---

## 9. State and version plane

### 9.1 The message envelope

Every boundary activation, activation gradient, consensus contribution, and audit artifact carries:

```
{ run_id, stage_id, stage_round, worker_id, batch_id, data_hash,
  weight_version, basis_version, optimizer_protocol_version,
  compression_schema, rng_seed, parent_hash, input_commit,
  output_commit, signature }
```

- **[SV-1]** Receivers MUST reject messages whose weight/basis/schema versions they cannot resolve (P-5, P-6a). An honest worker under a stale basis produces *unusable*, not *noisy*, coefficients; version rejection is the firewall between error channels.
- **[SV-2]** No global scalar step exists. Time is (stage_id, stage_round) plus the causal chain: `h(m) = H( h(parent m), weight_version, basis_version, batch_id, rng, output_commit )`, forming a per-run message DAG. Replay, audit, and reward accounting key off this DAG.

### 9.2 Checkpoints

Per-stage snapshots keyed by (stage_round, wall-timestamp); a "model checkpoint" is a *vector* of per-stage snapshots at causally consistent points, not a single step (production precedent [10]). Assembly and export semantics follow the D1 ownership mode. Restart semantics for stages at skewed logical times are part of P6.

### 9.3 Basis lifecycle (governance of U_k and all shared transforms)

```
PROPOSE ──► COMMIT ──► CHALLENGE WINDOW ──► TWO-PHASE ACTIVATE ──► (ROLLBACK)
```

- **[SV-3]** The update statistic (accumulated out-of-subspace gradient energy at the final compressed layer [3]) MUST be computed redundantly across tail-stage replicas and aggregated **Byzantine-robustly on the Grassmann manifold** (geodesic median / trimmed manifold mean — open algorithm, P3). A single aggregator is a single point of subspace control with global blast radius.
- **[SV-4]** The candidate basis is committed (content-addressed) before activation; a challenge window precedes adoption; activation occurs at a named stage round per stage, with an overlap grace window during which both basis versions are routable; reconstruction- or loss-diagnostic failure triggers rollback.
- **[SV-5]** Fixed/operator-controlled bases are a permitted v1 profile (they block very wide trunks, not initial deployments); basis poisoning is a mandatory design threat for any dynamic-basis deployment.

### 9.4 Transactional residual protocol (C-2 realized)

Enumerated commit points — a consensus round either commits or rolls back residual state exactly once at each:

1. after local delta extraction, before transmission;
2. after transmission, before aggregate acknowledgment;
3. after acknowledgment, before residual subtraction;
4. after subtraction, before the next checkpointable event;
5. during any state handoff (join, migration, warm-spare promotion).

- **[SV-6]** Implementations MUST keep a backup residual committed only when the full round (all phases of the collective) succeeds, reloaded otherwise — the production-validated pattern [10]. The churn drill (§13.4, E4) injects failures at each numbered point and verifies exactly-once behavior.

---

## 10. Assurance plane

### 10.1 Threat catalog

Fabricated coefficients; correct coefficients under a wrong (stale/forked) weight or basis version; replay of old valid messages; data substitution and duplicate-work claims; selective omission; sub-tolerance directed perturbation ("norm-capped steering"); staleness gaming (serving old-but-valid versions to earn while degrading); uptime farming and throughput-without-correctness against the reward function; manufactured stage scarcity; colluding stage replicas; colluding worker/verifier pairs; basis poisoning via the update statistic; slow-but-honest workers (the false-positive class).

### 10.2 The layered stack

```
continuous statistical screen (cheap, always on)          [10.3]
        ▼
signed, versioned, content-addressed transcripts           [9.1]
        ▼
Merkle commitments per stage round                         [SV-2]
        ▼
algebraic projection checks on linear ops (on challenge)   [10.4]
        ▼
sampled full replay within version windows                 [10.6]
        ▼
robust aggregation on the consensus axis                   [10.7]
        ▼
quarantine ──► adjudication ──► economic penalty           [10.7]
```

Each layer covers the one above's blind spot: screens have a detection floor; commitments prove provenance, never correctness (P-10); projection checks cover the linear ~99% of FLOPs cheaply; replay is exact but expensive; robust aggregation is the defense in the sub-tolerance gray zone; quarantine bounds false-positive harm before irreversible penalties.

### 10.3 Statistical screen (Sentinel-class)

EMA baselines of stage-boundary activations and activation gradients maintained at verifier positions (trainers are the natural hosts), with adaptively calibrated (IQR) divergence thresholds; validated to 128-worker SWARM scale with 37.5% malicious workers per stage at 15% collusion [12]. Declared limits: trusted verifier positions; boundary (first/last) stages assumed honest — which maps exactly onto the anchor tier and MUST be stated in D2-3 accounting; detection floor set by honest drift (the adversary's hiding room); statistical, not cryptographic.

### 10.4 Algebraic verification of stage computation (transmit-on-challenge)

For a claimed linear product Y = XW, a verifier holding the stage weights draws an unpredictable challenge vector r **after** the worker's commitment and checks `‖ Y·r − X·(W·r) ‖ ≤ ε_v`, caching W·r once per weight version; per-token cost is O(d) against O(d²) recomputation — Freivalds-style verification adapted to floating point and versioned state.

- **[AS-1] The guarantee is norm-capped, not absolute.** Floating-point tolerance ε_v (required by honest accumulation-order nondeterminism) converts "detects any corruption w.h.p." into "bounds the norm of undetectable corruption by B(ε_v, c) for c challenges." Whether norm-capped *directed* perturbation injected at every boundary can steer training is the decisive open question — answered empirically by the directed-perturbation extension of ℛ (§10.5), which sets ε_v.
- **[AS-2] Transmit-on-challenge protocol.** For a sampled microbatch (selected unpredictably post-commitment), the worker uploads its boundary tensors and designated intermediates; the verifier checks every linear map via projections (O(n·d) each) and recomputes elementwise/nonlinear ops outright (also O(n·d)); total verifier compute ≈ O(L·n·d) versus O(L·n·d²) full recomputation — a ~d-fold compute reduction, with the audit cost moved into **bandwidth** for the sampled fraction. Verification compute is thereby essentially solved on paper; the residual open items are the ε-versus-determinism-tax tradeoff, challenge freshness, sampled-bandwidth economics, collusion, and state availability (P4).
- **[AS-3]** Determinism requirements on senders (fixed kernel algorithms, deterministic reductions) cost throughput and MUST be measured and priced into the assurance budget.

### 10.5 The drift-response function ℛ (the shared calibration surface)

`ℛ( d_θ, d_activation, d_gradient, d_logit-JS, τ ) → ( Δloss / sample-efficiency penalty, detection statistics )`

measured on fixed **secret, rotating** probe batches (public probes are gameable), including a **directed-perturbation** variant (adversarially chosen, norm-bounded injections). One measured surface calibrates four consumers: the healing controller π (§7.5), quarantine thresholds, the statistical screen's floor (§10.3), and the verification tolerance ε_v (§10.4). Honest drift and dishonest perturbation live on the same axes; ℛ is simultaneously the optimization tolerance curve and the assurance ROC curve. It is measurable immediately under the v1 fixed-schedule operator and re-measured per consensus candidate (§13.2–13.3).

### 10.6 Sampled replay, version windows, and cost accounting

- **[AS-4]** Replay is unpredictable post-commitment, **risk-weighted** (concentrated where the screen flags, on scarce/anchor stages, and on new members), and executed **within version windows** while the required weight/basis versions are resident on co-replicas — minimizing verifier state transfer and remaining the only mode compatible with UPM ownership.
- **[AS-5]** Cost accounting distinguishes replay **compute** (∝ sampling rate; ~1–5% is affordable) from **state logistics, determinism tax, archival, cold-start, and ambiguous-tolerance re-runs**, which do not scale with the sampling rate and MUST be budgeted separately. Collusion resistance requires some verification by parties outside the replica group; its state-download cost couples the assurance budget to the churn plane (P-8) — design them as one ledger.

### 10.7 Robust aggregation, quarantine, adjudication

Consensus-axis updates pass robust aggregation (trimmed/median-of-means over weighted contributions) as the standing defense against sub-tolerance poisoning. Flagged workers are **quarantined** (removed from routing, state preserved) before any irreversible penalty; economic penalties require adjudication with published false-positive and false-negative bounds. Slow-but-honest is the canonical false-positive class and MUST be separable via τ-conditioning (the version plane makes realized staleness observable).

### 10.8 Data plane

Content-addressed sample IDs; deterministic assignment derivable from committed run state; leases with expiry; non-repeat proofs under retries and failure; duplicate-work accounting that distinguishes intentional verification duplication from accidental retraining; proof that a claimed batch belonged to the run. Deterministic balanced expert-group assignment (EXP-2) is implemented here.

### 10.9 Incentives

`reward = attested useful compute × stage scarcity × reliability × bandwidth/state-transfer contribution`, computed **only** from signed execution outcomes (never self-reported specs), with verification work itself compensated. Known-gameable v0 (raw PFLOPs + uptime credit [11]) is the baseline the mechanism must dominate. Scarcity pricing MUST be robust to manufactured scarcity (capacity withholding); mitigations include protocol-set posted-price curves and randomized assignment among qualified bidders. The allocation policy above the attestation layer (market vs. scheduler) is a tuning choice; **attestation is the security boundary** (open problem P8).

---

## Part III — Open problems, reference profile, validation

## 11. The open algorithms

Five algorithms gate the outcome (P1–P5); four determine how good it gets (P6–P9). "The sharding architecture is settled" means precisely: no new sharding primitive is required; the research is concentrated at these interfaces.

### P1 — Stochastic-delay pipeline optimization

**Problem.** The validated delay correction assumes a fixed, known per-stage delay, is proven for convex/smooth objectives, and degrades visibly by ~24 stages [4]. In the composed system, delay is a per-microbatch random variable — stage-dependent, heavy-tailed, replica-correlated, churn-shocked, and asymmetric between forward and backward — realized as `τ_{i,b} = v_i^{current} − v_i^{forward(b)}`, which the version plane makes directly observable (EX-5).

**Required.** An update rule `Δw = O( g_stale, τ, recent trajectory, route statistics )` that consumes realized delay: candidates include delay-conditioned NAG/NAdam discounting, staleness-weighted acceptance, delay-dependent clipping, exponential rejection of extreme staleness, trajectory extrapolation, and Kalman-style update-direction estimation. Stability MUST hold **with stage-wise data-parallel replicas present** — verbatim the named future work of the strongest existing result [4].

**Acceptance.** Tokens-to-target-loss under characterized (E[τ], Var[τ], P[τ>τ_max], drift, partial-backprop inconsistency) on recorded traces, beating bounded-staleness rejection. **Why it gates:** stage count — and therefore dense capacity — is bounded by delay tolerance; solving P1 moves the §6.1.3 envelope more than any memory technique.

### P2 — Churn-native compressed stage consensus

**Problem.** No existing operator satisfies C-1…C-8 (§7.4) simultaneously: rotating sparse averaging lacks proven contraction under partial weighted rounds; flat DeMo aggregation violates C-6 and its residual semantics under churn are undefined [2]; PowerSGD needs the transactional retrofit [10][18]; factored gossip is unproven under partial quorums at scale [13]. This is the main missing bridge between Regime B's compressed coordination and Regime A's sharded execution.

**Required.** The §7.4 operator, plus: residual synthesis for replacements (C-5), hierarchical aggregation topology (C-6), and the drift controller π (C-7) calibrated by ℛ. **Acceptance:** the tournament (§13.2) — tokens-to-loss per byte, per wall-clock, per GB of resident state, **and measured contraction ρ̂ per unit bandwidth** on recorded churn traces. A candidate that wins short benchmarks while ρ̂ ≥ 1 fails.

### P3 — Subspace scaling law and robust basis governance

**Problem (scientific).** Determine whether adequate quality requires k = O(1), O(√d), or O(d) as width grows, and whether the class restriction costs a constant multiplier or bends the L(N,D) exponent — via matched pairs `L_SSN(N, D, k)` vs `L_vanilla(N, D)` across width-at-fixed-depth, depth-at-fixed-width, several token budgets, several k/d ratios, seeds at small scales. If k/d stays ~1%, WAN advantage persists with width; if k → O(d), it erodes. The completed 8B run is the anchor datapoint; the gate (§13.3) is mandatory before compute-optimal commitment (SSN-3).

**Problem (algorithmic).** Byzantine-robust subspace tracking: redundant statistic computation, robust aggregation **on the Grassmann manifold** (geodesic median / trimmed manifold mean — a genuinely new algorithm class), committed candidates, challenge windows, two-phase activation with overlap, rollback (§9.3). Blocks very wide dynamic-basis trunks; not fixed-basis v1 deployments.

### P4 — Cheap verification of an intermediate stage

**Problem.** Distinguish natural drift, floating-point nondeterminism, honest staleness, hardware faults, and deliberate corruption — at a cost compatible with open membership. Provenance is solved (§9.1); correctness is not (P-10).

**Required.** The §10.2 stack with its two open cores: (a) the **separating statistic** — ℛ's discriminative axis between honest drift and directed perturbation (early evidence: output-distribution divergence [13]); (b) the **floating-point Freivalds bound** — whether norm-capped undetectable perturbation B(ε_v, c) at every boundary can steer training (AS-1), plus challenge freshness, worker/verifier collusion, state availability for outside verifiers, and operation under UPM ownership. Transmit-on-challenge (AS-2) reduces verifier compute ~d-fold, leaving economics dominated by sampled bandwidth and state logistics (AS-5).

**Why it gates:** this is the only open problem that changes what the system *is* — permissionless versus staked — rather than how well it runs. A staked system operates without it (§12); a permissionless one cannot. It is self-contained (no swarm required to develop) and MUST NOT be serialized behind the others in a multi-team effort.

### P5 — A trainable coarse-grained expert overlay

**Problem.** The only route past the dense envelope, and the least-developed component. Five coupled sub-problems: (1) **router credit assignment under delay** — the quality signal for a document-level routing decision returns much later; candidates: auxiliary router losses, distillation from an offline router, straight-through estimation, contextual-bandit routing, periodic frozen-router phases, slow outer-loop updates; (2) **balanced assignment** inside expert groups (EXP-2 — precondition for the consensus operator's assumptions); (3) **heterogeneous placement/replication** reacting to popularity, bandwidth, state-transfer cost, expected session length; (4) **incremental resumable migration** of the full expert state vector; (5) **nonstationarity** — every expert trains against a moving router-conditional distribution even with perfect balance. No mature WAN-scale algorithm jointly solves these. **Depends on:** P2 (expert groups run the consensus operator in its most favorable regime, EXP-3).

### Secondary problems

**P6 — Composition theory.** No analysis covers asynchronous pipeline SGD with intermittent, compressed, *partial* averaging under drift; even a simplified model (smooth nonconvex, bounded stochastic delay, periodic ε-approximate averaging) would give the first principled coupling of healing period H, delay distribution, and learning rate — subsuming principled stage-wise clipping and skewed-clock restart semantics. Until then all three are folk-tuned (§2.4).

**P7 — Unified bandwidth allocation.** The online rate-distortion controller of BW-1: is the next megabyte on an edge worth more as activation fidelity, consensus freshness, or audit rate? Unformulated; even a heuristic controller should dominate static settings.

**P8 — Incentive-compatible allocation.** Strategyproof stage-slot allocation from attested outcomes only; scarcity pricing without manufactured scarcity; compensation for verification, bandwidth, and state-transfer work; replica collusion treated as an economic problem as well as a cryptographic one (§10.9). Essentially untouched.

**P9 — Elastic-batch semantics and re-globalization.** Effective batch per stage round is a random variable under churn; schedules assume it fixed. Needed: contribution-weighted updates and batch-adaptive LR/warmup rules; note sign-family updates (Signum/Lion/DeMo-style) have batch-size-invariant step magnitude — a candidate selection criterion. Re-globalization — recovering the abandoned global fleet from the current single-region, <80 ms deployment — is P1's delay mathematics composed with placement: regional replica rings, few intercontinental pipeline cuts placed at the highest staleness-tolerance boundaries.

---

## 12. Reference implementation profile v1 (buildable now)

A credible first system requires **none** of P1–P5 solved:

| Dimension | v1 choice |
|---|---|
| Membership | Semi-permissioned anchors + staked body tier; regional (<80 ms domain); edge tier for audit/data roles only |
| Trunk | 8–15B dense SSN model; asymmetric stages; fixed operator-controlled basis (SV-5); untied embeddings |
| Local optimizer | NAdam no-stash with stage-dependent schedules (EX-1..4); stage-wise clipping |
| Delay handling | Bounded-staleness rejection on realized τ (EX-5) |
| Consensus | Fixed-schedule rotating sparse averaging over partial single-round collectives; transactional residuals (SV-6); moments not averaged |
| State plane | Full §9 envelope, per-stage checkpoints, resumable base+delta transfer |
| Assurance | Signed versioned transcripts + commitments; statistical screen; quarantine; **no slashing** (threat tier a/b) |
| Overlay | None initially |
| Ownership | Canonical checkpoint or distributed custody |

This trains a dense model larger than any body node can hold — approximately the system that has already run. What v1 explicitly does **not** provide: open global membership (needs P1 + P4), horizontal consensus width at hundreds of replicas (P2/C-6), very wide trunks (P3), participation-scaled total parameters (P5), or adversarial-tier integrity (P4).

**Upgrade map:** global permissionless consumer network ⇐ P1 + P2 + P3(governance) + P4. Total parameters beyond the dense envelope ⇐ P5 (after P2). Competitive economics ⇐ P6–P9 + MFU engineering (from ~20% deployed; every recovered point moves the viability inequality directly).

---

## 13. Validation program

### 13.1 E0 — Interface and version invariants (first; cheap; unblocks assurance)

Implement §9 as a real state system: causal message IDs, stage-local clocks, immutable weight/basis versions, durable code/data/RNG manifests, two-phase basis activation, exactly-once residual commit/rollback, base+delta snapshots. **Exit test:** an activation generated under version tuple (v_w, v_U, batch, seed) can be replayed and audited **after the producing worker has departed**.

### 13.2 E1 — The consensus tournament (P2)

1–3B scale on **recorded** WAN/churn traces (not synthetic distributions). Arms: (1) fixed-schedule rotating sparse averaging [production baseline]; (2) factored gossip; (3) DeMo-derived compressed outer deltas with separate transactional residual; (4) PowerSGD + transactional error feedback; (5) declared low-memory partially-aliased profile; (6) fully unified buffer, exploratory only. Metrics: tokens-to-target-loss per network byte, per wall-clock second; VRAM + host RAM + PCIe traffic; **contraction ρ̂ per unit bandwidth (C-8)**; coupling diagnostics (delay-alignment cosine, residual-norm trajectories). Naming discipline: a DCT top-k parameter-delta method is *DeMo-inspired* unless it preserves DeMo's momentum construction, subtraction semantics, and update rule.

### 13.3 E2 — Interaction study and the SSN scaling gate (P3)

Factorial over {SSN, PP asynchrony, replica drift, compression method, churn} — run as a fractional-factorial screen first (five factors at matched tokens is otherwise ruinous), then full grid on the significant two or three. The scaling gate runs in parallel: matched L(N,D,k) pairs per P3; **passing the gate is a prerequisite for any compute-optimal E6 commitment** (SSN-3). Judged on final-quality and tokens-to-target-loss, never visual overlap of short curves.

### 13.4 E3 / E4 — Drift response and churn drill

**E3:** measure ℛ (§10.5) under the v1 operator — downstream loss, gradient variance, and replay disagreement as functions of (d_θ, d_activation, d_gradient, d_logit-JS, τ), with secret rotating probes and the directed-perturbation variant. Deliverables: the healing controller's policy surface, quarantine thresholds, screen floor, ε_v. E3 is measurable immediately and is upstream of more consumers than any other experiment; re-measure per E1 candidate. **E4:** inject departures at each numbered commit point of §9.4 (before compression; during aggregation; after upload pre-ack; after subtraction pre-checkpoint; during handoff) and verify exactly-once residual behavior; replay session-length traces to choose replication factors r(stage) and state-streaming policy by measured useful-fleet-time.

### 13.5 E5 — Adversarial assurance test

Red-team the full §10.1 catalog — including basis poisoning, reconstruction-adversarial coefficients within ε_v, staleness gaming, and worker/verifier collusion — measuring detection probability, false-positive rate (slow-but-honest separability), verification compute, audit bandwidth, and time-to-quarantine. Economic penalties activate only after published FP/FN bounds exist (§10.7).

### 13.6 E6 — The merged controlled run, and metric definitions

8–15B dense SSN trunk + coarse expert overlay; **matched centralized control** (same architecture, data order, token budget); equalized useful FLOPs. Report: final benchmarks and held-out perplexity; tokens-to-target-loss; tokens per dollar; useful FLOPs / contributed FLOPs; state-transfer, failed-work, and audit overhead; full churn and stage-occupancy traces; TPS distribution (not one snapshot); **MFU with published methodology** (heterogeneous fleets make the denominator ambiguous: nominal peak vs measured BF16 peak vs kernel peak vs useful model FLOPs); **anchor-criticality accounting** (D2-3). 

**The viability inequality** (the economic verdict): the swarm is viable iff
`cost per device-hour (fleet) < cost per device-hour (centralized) / ( T_compute × R_MFU )`
where T_compute is the full-stack sample-efficiency tax (first self-reported estimate ≈ 1.5×, controls pending) and R_MFU the MFU ratio (≈ 0.35–0.45 / 0.20 ≈ 2× today) — i.e., roughly a 3× device-time handicap that idle consumer hardware plausibly clears and rented cloud GPUs plausibly do not.

**Evidence-package norm:** every production run publishes a postmortem containing the above **plus its churn/bandwidth/occupancy traces as a public benchmark dataset** — the field's substitute for a shared testbed, and the input that makes E1–E5 realistic. (The outstanding exemplar: the promised Agora-8B technical report, which must also reconcile its token-count/duration/throughput figures and state the basis of its 1.5× claim.)

---

## 14. Falsification criteria and fallback

Pre-declared, per P-14. Reconsider the corresponding branch if:

- **F1 (dense-WAN):** `(L_SSN − L_vanilla)/L_vanilla` grows materially with scale at matched compute, **or** required k grows enough that boundary compression loses its WAN advantage (P3 gate fails).
- **F2 (deep pipelines):** the tokens-to-loss penalty grows superlinearly with the route-delay distribution or stage count despite P1 candidates.
- **F3 (DeMo-lineage consensus):** it fails to beat the sparse-averaging or factored-gossip baselines on wall-clock tokens-to-loss even at smaller byte counts (E1).
- **F4 (permissionless membership):** combined verification, replay, and false-positive costs cannot be held to low single-digit percent of useful training cost (E5); membership then remains staked/permissioned — which is where deployed authorization layers already sit.
- **F5 (sparse overlay):** routed-expert quality at matched active parameters underperforms dense at matched total compute under WAN constraints.

**Fallback (coherent, not failure):** Regime B — federated large logical replicas with compressed optimizer coordination — carrying the coarse expert overlay: participation-scaled *total* capacity without globally sharded dense execution.

---

## 15. Dependency structure and sequencing

```
Agora postmortem / public traces ─┬─► E1 tournament ─► P2 operator ─► P5 expert overlay
                                  ├─► E3 ℛ-measurement (v1 operator) ─► π, quarantine,
                                  │                                     screen floor, ε_v
E0 version/state plane ───────────┴─► P1 delay-aware optimizer
P3 scaling gate (small-model pairs; 8B anchor) ── independent, parallel
P4 verification study (Freivalds/ℛ-directed)  ── independent, parallel, self-contained
P6–P9 ── sharpen everything, block nothing
```

Two hard edges only: **P5 ← P2** and **P1 ← version-plane instrumentation** (which v1 installs regardless; the audit sensor doubles as the delay sensor). E3 SHOULD NOT wait for the tournament — it runs under the baseline operator and is re-measured per candidate. P4 is the only item that changes what the system is and MUST be parallelized, not serialized, across teams. Single-team order: postmortem → E0 → E1+E3 → P3 gate → E5 → E6.

---

## Appendix A — Quantitative design tables *(informative)*

**A.1 Compute-to-communication ratio and required boundary bandwidth** (ratio = 18·k_L·d FLOP/byte; bandwidth at 25 effective TFLOPs, both directions combined; compressed = ÷100):

| d | k_L (layers/stage) | FLOP/byte | Uncompressed need | With SSN ~100× |
|---|---|---|---|---|
| 2048 | 4 | ≈147k | ≈1.4 Gbps | ≈14 Mbps |
| 4096 | 4 | ≈295k | ≈0.7 Gbps | ≈7 Mbps |
| 4096 | 8 | ≈590k | ≈0.34 Gbps | ≈3.4 Mbps |
| 8192 | 4 | ≈590k | ≈0.34 Gbps | ≈3.4 Mbps |
| 8192 | 8 | ≈1.18M | ≈0.17 Gbps | ≈1.7 Mbps |

**A.2 Persistent state per parameter:** optimistic 6 B (bf16 w+g, 8-bit moments) · Node0 observed ≈10 B (per-layer 2.4 GB / ≈234M params) · deployed Agora regime ≈8 B GPU (fp32 w+g) + 8 B host (fp32 moments) — hence the 80 GB host-RAM floor. Add activations, in-flight microbatches, stash, buffers before applying D2-1.

**A.3 Onboarding time** (per 2.4 GB layer of state): 100 Mbps ≈ 3.2 min · 50 Mbps ≈ 6.4 min · 20 Mbps ≈ 16 min; multiply by layers/stage; hence join windows, staleness allowances, warm spares, and r ≥ 2–3 (§7.7).

**A.4 Dense envelope grid** (stage params × distinct stages): 1B×20 = 20B · 1.5B×24 = 36B · 2B×32 = 64B (requires P1 beyond ~24 stages) — read jointly with A.1–A.3 and §6.1.3.

## Appendix B — Rejected alternatives ledger *(informative)*

WAN tensor parallelism (per-layer collectives × RTT; placement studies route around it [8]) · WAN ZeRO-3/FSDP (gathers on the critical path; churn breaks groups) · token-level WAN MoE (per-layer all-to-all sync points; DMoE-lineage evidence [9]) · remote parameter paging (connectivity on the critical path; local RAM/NVMe instead) · global scalar step (per-stage timestamp checkpoints adopted in production [10]) · Moshpit-in-stage (multi-round, large-group assumptions; empirically rejected [10]) · single merged communication plane (P-1) · DeMo/DisTrO as memory sharding (AV-8) · unified momentum buffer as default (semantic collisions among optimizer/residual/delay roles; retained only as declared low-memory profile) · ungoverned dynamic basis (poisoning blast radius) and permanent frozen basis (blocks width scaling) — superseded by the governed lifecycle (§9.3) · uniform audit sampling (risk-weighted instead) · markets on self-reported specs (attestation is the security boundary).

## Appendix C — Glossary *(informative)*

**Island:** co-located GPU group registering as one logical worker. **Stage round:** stage-local lockstep optimizer step; the universal primitive (P-11). **Anchor/body/edge:** fleet tiers (§4.2). **SSN / U_k:** subspace-constrained architecture and its shared basis (§6.2). **Consensus operator / residual / healing:** §7.4. **Drift D / ℛ:** disagreement vector and its measured response surface (§10.5). **Transmit-on-challenge:** audit mode of §10.4. **Warm spare:** pre-synchronized standby replica. **Useful vs contributed FLOPs:** work that advanced the model vs work performed. **Viability inequality:** §13.6. **UPM:** unextractable protocol model (§4.1). **Witness:** availability/observation attestor (provenance role; not a correctness verifier).

## Appendix D — References

[1] Nous Research, *A Preliminary Report on DisTrO* (2024). [2] Peng, Quesnelle, Chen, Su, Kingma, Liu, *DeMo: Decoupled Momentum Optimization*. [3] Ramasinghe et al., *Protocol Models*, arXiv:2506.01260 (NeurIPS 2025). [4] Ajanthan et al., *Nesterov Method for Asynchronous Pipeline Parallel Optimization* (Pluralis). [5] Ramasinghe et al., *Mixtures of Subspaces for Bandwidth-Efficient Context-Parallel Training* (Pluralis). [6] Ryabinin et al., *SWARM Parallelism*, arXiv:2301.11913. [7] Ryabinin et al., *Moshpit SGD*, arXiv:2103.03239. [8] Yuan et al., *Decentralized Training of Foundation Models in Heterogeneous Environments*, arXiv:2206.01288. [9] Ryabinin & Gusev, *Decentralized Mixture-of-Experts*, arXiv:2002.04013. [10] Pluralis, *Multi-party Training Stack* (Jan 2026). [11] Pluralis, *Agora documentation / Pluralis-8B*; *AsyncMesh*, arXiv:2601.22442. [12] Dolatabadi et al., *Sentinel: Stagewise Integrity Verification for Pipeline-Parallel Decentralized Training*, arXiv:2603.03592. [13] Pluralis, *Factored Gossip DiLoCo* (ICML 2026). [14] Pluralis, *Unextractable Protocol Models*. [15] Nous Research, *The Psyche Network Architecture*; *The Next Phase of Psyche* (Nov 2025). [16] Nous Research, *Introducing Hermes 4.3* (Dec 2025). [17] Pluralis, *Node0-7.5B dashboard and completion report*. [18] Vogels et al., *PowerSGD*, arXiv:1905.13727. [19] Beton et al., *Sparse Parameter Averaging (SPARTA)*, MCDC@ICLR 2025. [20] Douillard et al., *DiLoCo*, arXiv:2311.08105. [21] Epoch AI, *How far can decentralized training over the internet scale?* (Dec 2025). [22] *FlexDeMo*, arXiv:2502.06728. [23] Pluralis, Pluralis-8B completion announcement (2026).

## Appendix E — Provenance and change control

Derived from a five-round adversarial technical review (July 2026) over the primary sources above and the public production record; positions retained here survived verification against primary sources or are marked with their evidentiary status. Amendments to this specification require either (a) new primary evidence with citation, or (b) results from the §13 validation program; falsification events under §14 trigger a major-version revision including the fallback re-baseline.

*— End of specification —*
