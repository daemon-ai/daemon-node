# daemon-vhc — SDK training primitives specification

**Subsystem:** VHC (Virtual Heterogeneous Cluster) — the guest SDK primitive ladder: contract
vocabulary, typed async runtime, codecs, partial collectives, sync drivers, membership,
execution-plane transport, pipeline machinery, and the trajectory monitor, through the SWARM and
AsyncMesh compositions.
**Status:** design specification, pre-implementation. This document is the SDK primitive
contract that the post-C2 training waves implement. It **embodies** decisions the
[architecture spec](vhc-architecture-spec.md) already made and **defers** nothing that spec
decided — every clause of that spec this document touches is dispositioned in Appendix B, and the
architecture spec carries a matching cross-reference so no competing model survives in the
corpus. The [module ABI spec](vhc-module-abi-spec.md) remains the ABI authority; nothing here
changes a frozen ABI surface without the minor-versioning discipline that spec defines.

**Requirement grading.** This document is deliberately **graduated** — coverage is complete
through SWARM and AsyncMesh, but the epistemic commitment is not uniform:

- **Part A (§2–§9): normative.** Determinism model, contract vocabulary, async runtime,
  compute-lane levers, codecs, partial collectives, sync drivers, membership. RFC 2119/8174
  keywords bind here.
- **Part B (§10–§12): vocabulary-complete, enforcement-deferred** — the architecture spec's own
  [LC-2] device. Execution transport, pipeline machinery, and the trajectory monitor are
  specified so contracts and manifests can carry their vocabulary from day one, but their
  clauses do not bind a conforming implementation until the **promotion criterion** named in
  each section is met (a wave gate in §14). Until promotion, a host MUST refuse plans exercising
  this vocabulary ([LC-2]), exactly as the architecture spec requires.
- **Part C (§13): informative reference profiles.** The experiments. Infrastructure invariants
  they rely on are normative in Parts A/B; the algorithms themselves (DiffusionBlocks-LM,
  sparse-averaging disagreement behavior, swarm-routed pipeline convergence, AsyncMesh
  composition) are empirical research and are never normative-MUST. Each profile carries
  acceptance gates instead.

**§1.4 is the mechanism inventory** — every clause's provenance class (in-tree / ported /
paper-only / authored / reference-only), source, owner, and wave, in one table. Readers asking
"what is being ported from what" start there.

---

## 0. Conformance, terminology, and sources

### 0.1 Requirement levels

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHOULD**, **SHOULD NOT**,
**MAY**, and **OPTIONAL** are to be interpreted as in RFC 2119/8174, subject to the requirement
grading above: in Part B they bind only after the section's promotion criterion is met; in
Part C they do not appear (profiles use gates, not requirement keywords).

Clause ID families: **[DM-n]** determinism model (§2), **[GV-n]** contract vocabulary (§3),
**[RT-n]** async runtime (§4), **[ML-n]** compute-lane memory levers (§5), **[CX-n]** codecs
(§6), **[PC-n]** partial collectives (§7), **[DR-n]** sync drivers (§8), **[MB-n]** membership
(§9), **[XP-n]** execution-plane transport (§10), **[PP-n]** pipeline machinery (§11),
**[MO-n]** trajectory monitor (§12), **[EX-n]** reference profiles (§13), **[W-n]** refactor
waves (§14), **[OQ-n]** open questions (§15).

### 0.2 Citation keys and pinned sources

| Key | Document | Role |
|---|---|---|
| arch §N / [XX-n] | [`vhc-architecture-spec.md`](vhc-architecture-spec.md) | design authority; clauses cited by their IDs ([VP-n], [PIR-n], [CO-n], [LC-n], [PL-n], [PD-n], [GR-n], [AL-n]) |
| ABI §N | [`vhc-module-abi-spec.md`](vhc-module-abi-spec.md) | ABI authority (`da_abi` major 2) |
| v1 §N | [`swarm-training-spec.md`](swarm-training-spec.md) | the v1 spec lineage; §5.4/§5.5 carry the Phase-2/3 designs this document grounds in code |
| TDD §N | [`swarm-training-tdd.md`](swarm-training-tdd.md) | test-port plan; §4 gap register, §7 Phase-2/3 double debt |
| PS | [`../vhc-program-state.md`](../vhc-program-state.md) | program state; the standing non-claims (~lines 232–265) bound what is host work |
| REL §N | [`vhc-capability-reliability-spec.md`](vhc-capability-reliability-spec.md) | the post-C2 reliability workstream (substantially LANDED) |

Reference checkouts (all under `~/experiments/decentralised-llm-training/`), pinned at the
commits audited for this document. Line citations throughout are against these commits.

| Checkout | Commit | Date | Note |
|---|---|---|---|
| `hivemind` | `4bd43b7` | 2026-01-04 | |
| `agora` | `37d29a9` | 2026-07-04 | server + launcher CLI + docs; the trainer client (worker selection/routing) is not in the tree |
| `AsyncMesh` | `969c9a7` | 2026-02-02 | vendors `asyncpp/` (same code as the `AsyncPP` checkout) |
| `AsyncPP` | `584658e` | 2026-02-02 | |
| `DiffusionBlocks` | `ade0b08` | 2026-02-18 | image classification only; no LM experiment exists (the *method* is not vision-only — §1.3 correction 5) |
| `psyche` | `0bdb13d9` | 2026-03-20 | `nousnet` is the same commit under a second remote — one source, not two |
| `node0` | `32bd084` | 2025-09-23 | |
| `OpenDiloco` | `2d750e5` | 2025-01-13 | |
| `prime-diloco` | `de5b931` | 2025-04-10 | |
| `torchft` | `90d7f68` | 2026-07-16 | |
| `rl-swarm` | `9c95410` | 2026-01-05 | different product (RL swarms, v1 §5.5); no primitive is ported from it |
| `swarm` | `f66855f` | 2023-12-11 | official [SWARM-P] prototype; carries the routing balancer ([PP-6]) |
| `pipedream` | `7db6a1c` | 2021-07-22 | official [PDREAM] code; weight-stashing origin ([PP-4]) |
| `torchgpipe` | `a1b4ee2` | 2020-09-18 | reference [GPIPE] implementation (schedule + recompute witnesses) |
| `powersgd` | `f07be92` | 2024-10-29 | official [PSGD] code; upstream EF-loop witness ([CX-8]) |
| `AC-SGD` | `95a0f2e` | 2023-04-25 | official [AQSGD] code; per-sample delta compressor ([CX-10], [OQ-9]) |
| `CheckFree` | `8d7d547` | 2026-03-27 | official [CHECKFREE] code (EX-5 vacancy recovery option) |
| `DeMo` | `0e48145` | 2024-12-02 | official [DEMO-P] standalone optimizer + OLMo patch (second witness beside psyche) |
| `moshpit-sgd` | `949b60b` | 2025-02-04 | official [MOSHPIT] experiments; the averaging machinery itself lives upstream in hivemind |
| `asyncdiloco` | `e783a30` | 2024-01-18 | official [ALSGD] release — a toy-example notebook, NOT the paper's training stack |

The `p2p/` directory (librats, dhtnet/OpenDHT, libdatachannel, iroh) is **excluded as a porting
source**: it contains four transport stacks and no training collectives, and `daemon-vhc-net`
already runs iroh.

**Paper register.** Literature citations use the bracketed keys below; `[KEY]:N` is a line pin in
the named file under `~/experiments/decentralised-llm-training/research-papers/`. Code checkouts
and papers are cited separately on purpose: a mechanism's code source and its published source
frequently disagree (§1.3), and every port must know which one it is claiming fidelity to.

| Key | File |
|---|---|
| [DBLOCK] | `diffusionblocks__block_wise_neural_network_training_via_diffusion_interpretation.md` |
| [BTRAIN] | `decentralised_ai_training_and_inference_with_blocktrain.md` |
| [SPARTA] | `improving_the_efficiency_of_distributed_training_using_sparse_parameter_averaging.md` |
| [AMESH] | `asyncmesh__fully_asynchronous_optimization_for_data_and_pipeline_parallelism.md` |
| [NAG-PP] | `nesterov_method_for_asynchronous_pipeline_parallel_optimization.md` |
| [SDILOCO] | `streaming_diloco_with_overlapping_communication__towards_a_distributed_free_lunch.md` |
| [DILOCO-P] | `diloco__distributed_low_communication_training_of_language_models.md` |
| [SCALE-DL] | `communication_efficient_language_model_training_scales_reliably_and_robustly__scaling_laws_for_diloco.md` |
| [CCD] | `consensus_control_for_decentralized_deep_learning.md` |
| [MERGE1] | `on_the_surprising_effectiveness_of_a_single_global_merging_in_decentralized_learning.md` |
| [FG] | `factored_gossip_diloco__reducing_blocking_communication_in_diloco.md` |
| [DEMO-P] | `demo__decoupled_momentum_optimization.md` |
| [DISTRO-R] | `a_preliminary_report_on_distro.md` |
| [PM] | `protocol_models__scaling_decentralized_training_with_communication_efficient_model_parallelism.md` |
| [AGORA-P] | `agora__collective_and_permissionless_internet_scale_pretraining_of_large_language_models.md` |
| [MAPL] | `learned_subspace_compression_for_communication_efficient_pipeline_parallelism.md` |
| [MOS-P] | `mixtures_of_subspaces_for_bandwidth_efficient_context_parallel_training.md` |
| [RESBM] | `resbm__residual_bottleneck_models_for_low_bandwidth_pipeline_parallelism.md` |
| [AQSGD] | `fine_tuning_language_models_over_slow_networks_using_activation_compression_with_guarantees.md` |
| [CHECKFREE] | `all_is_not_lost__llm_recovery_without_checkpoints.md` |
| [BROT] | `mitigating_staleness_in_asynchronous_pipeline_parallelism_via_basis_rotation.md` |
| [SKIPPIPE] | `skippipe__partial_and_reordered_pipelining_framework_for_training_llms_in_heterogeneous_networks.md` |
| [SWARM-P] | `swarm_parallelism__training_large_models_can_be_surprisingly_communication_efficient.md` |
| [NOLOCO] | `noloco__no_all_reduce_low_communication_training_method_for_large_models.md` |
| [MOSHPIT] | `moshpit_sgd__communication_efficient_decentralized_training_on_heterogeneous_unreliable_devices.md` |
| [DESLOC] | `des_loc__desynced_low_communication_adaptive_optimizers_for_training_foundation_models.md` |
| [MTDAO] | `mt_dao__multi_timescale_distributed_adaptive_optimizers_with_local_updates.md` |
| [CROSSPIPE] | `crosspipe__towards_optimal_pipeline_schedules_for_cross_datacenter_training.md` |
| [LOSCAR] | `loscar_sgd__local_sgd_with_communication_computation_overlap_and_delay_corrected_sparse_model_averaging.md` |
| [BIRCH] | `birch_sgd__a_tree_graph_framework_for_local_and_asynchronous_sgd_methods.md` |
| [EF-SGD] | `error_feedback_fixes_signsgd_and_other_gradient_compression_schemes.md` |
| [EF21] | `ef21__a_new__simpler__theoretically_better__and_practically_faster_error_feedback.md` |
| [GPIPE] | `gpipe__efficient_training_of_giant_neural_networks_using_pipeline_parallelism.md` |
| [PDREAM] | `pipedream__fast_and_efficient_pipeline_parallel_dnn_training.md` |
| [PSGD] | `powersgd__practical_low_rank_gradient_compression_for_distributed_optimization.md` |
| [ALSGD] | `asynchronous_local_sgd_training_for_language_modeling.md` |

**Regime note (rev 5).** The target regime is the wild internet: untrusted, heterogeneous
(GPU class, VRAM, link speed), churning consumer clients — not a datacenter. Some registered
sources are datacenter-regime work admitted under a restricted citation discipline:
[GPIPE]/[PDREAM] (with the torchgpipe/pipedream checkouts) are cited **only as mechanism
origins** — the semantics a port claims fidelity to (re-materialization, fill/drain with
synchronous accumulation, same-version weight stashing) — because the slow-network pipeline
work this document actually targets ([SWARM-P], [AGORA-P], [PM], [NAG-PP], [CROSSPIPE],
[SKIPPIPE]) inherits and cites those mechanisms. Likewise [PSGD]/[EF-SGD]/[EF21] are
datacenter data-parallel work cited for **convergence theory only** (the math is
regime-independent; the systems evaluation is not). No clause may cite a datacenter-regime
source for feasibility, throughput, bubble economics, or failure behavior over the open
internet — that evidence must come from the decentralized corpus or from VHC's own gates
(§2, §12).

**Corpus gap, resolved (rev 4):** LOSCAR-SGD — cited by [AGORA-P]:1603 as the convergence
analysis for delay-corrected sparse averaging — was the register's one named gap; the paper is
now in the corpus as [LOSCAR]. It is the first theory combining local steps, sparse model
averaging, communication-computation overlap, and worker-specific step counts in one method
([LOSCAR]:48), and it directly anchors [DR-1]'s overlap claim and [DR-3]'s delta-application
convention (see those clauses for scope limits — it is round-structured, common-mask, and
data-homogeneous, not [AMESH]'s fully asynchronous regime).

### 0.3 No-porting-source register

The authoritative provenance map is §1.4 (one row per mechanism, all five classes); this
register is its nuance annex for the P2/P3 entries — the clauses with **no reference
implementation in any checkout**, authored from this specification (or the named paper) alone.
A reader looking for the porting source of any of these will not find one; that absence is
recorded here so it is not mistaken for a missing citation:

- **§3 [GV]** — the group/channel/state vocabulary and the per-group layout bindings. Nothing in
  any checkout is a manifest vocabulary; AsyncMesh's rank layout is a formula, not a contract.
- **§4 [RT]** — the typed async runtime. No checkout is completion-based over an op-ID ABI.
- **§5 [ML]** — all items are host compute-lane work with no portable external
  implementation. Behavioral witnesses and published origins exist (bf16 posture, blockwise
  storage format, the recompute pattern — §1.4 host table) but the lanes themselves are
  authored; nothing is ported.
- **§7 [PC-1]** — windows-as-reduction-parts is a VHC-specific invariant; hivemind supplies
  geometry only.
- **§10 [XP-1]** — the [PL-1]-conformant framing. All three pipeline reference repos gate the
  step boundary on drain, and the corpus-wide check confirms it: [SKIPPIPE]:60 skips stages
  deterministically *under an explicitly synchronous weight-update boundary* ("the weight update
  of an iteration is done after all the corresponding microbatches are processed"), [SWARM-P]
  reroutes and bans but never drops, and [CROSSPIPE] waits and accounts the wait as a bubble.
  The deadline-omission model has no precedent; it is authored.
- **§12 [MO]** — nothing in any checkout computes a contraction rate. The statistic
  *definitions* now anchor on the literature ([CCD]'s scalar consensus-distance estimator,
  [FG]'s fixed-batch logit-JS — §12), but the monitor computing them is authored.
- **§13/§15** — Grassmann basis refresh and row-constant-second-moment AdamW ([PM]) are
  paper-only, and AsyncMesh's published EMA-coefficient λ cosine schedule (0.5 → 0.01,
  [AMESH]:155) is absent from its checkout, which cosines momentum instead. TDD §7 flags all
  three as double debt: implement from paper AND author the only tests. (An earlier draft
  misattributed the λ schedule to SPARTA — §1.3 correction 9.)

### 0.4 Source-fidelity traps

Two upstream artifacts will corrupt a port that trusts names over bytes; they are pinned verbatim
so every port cites them:

- psyche's wire-integrity hash method is misspelled **`comptue_hash`**
  (`psyche/shared/network/src/serialized_distro.rs:33`). Any port that "fixes" the name while
  claiming wire parity with psyche fixtures is not bit-compatible with the fixture generator.
- torchft's `bucket_cap_mb` **default** is `1 * 1024 * 1024 * 1024` (bytes — 1 GiB) at
  `torchft/torchft/local_sgd.py:176`, while a **passed** value is scaled by `1024 * 1024` at
  `:225`. The parameter name and its default disagree by a factor of 1024²; port the semantics,
  not the name.

### 0.5 Changelog

- **Revision 6 (2026-08-18, review response — gate math and provenance consistency).** Response
  to two independent reviews of revision 5; adopted findings only. Substantive: the two `p`'s
  are named apart — `p_mix` ([CCD] Assumption 1's mixing parameter) vs `p_sparse` (§8's
  sparsity fraction) — with the identity between them stated and scoped ([DM-3]: exact for a
  random common mask over the full roster, [AMESH]:137; inapplicable to the shipped rotation
  and async-delayed policies, whose gate targets are therefore phase-local and empirical);
  [MO-2]'s Θ estimator restated in VHC terms (the committed-set fold is the averaging
  operator — no mixing matrix — and rotation samples per cycle); [DM-3] gains an evaluability
  clause (four profile-declared values: phase boundaries, bound, cadence, breach action) which
  EX-6 now carries. Literature fixes, registered as §1.3 corrections 11 and 12: SPARTA's
  exchange is asynchronous/overlapped (the synchronous element is the DiLoCo H-boundary), and
  AsyncMesh's correction is an EMA-estimated average, with "delta application" reserved for
  [LOSCAR]'s rule ([DR-3] reworded to "corrected merge instead of naive overwrite").
  Consistency: [DR-4] restored to W5/§8/EX-1 (composition + outer-step determinism gate);
  [CX-10] and [PC-9] given wave homes in W4's row and [MB-7] in W5's; the P2-concentration
  sentence now names its one early exception (E3M0, W4); the [ML] family's four provenance
  statements reconciled (authored host lanes with named witnesses; [ML-2] re-labeled P3);
  A.1 gains the "in-tree (P0)" relationship so P0 rows are no longer labeled "authored";
  [CX-7]'s inventory row separates the agora/node0 codec from the [PM] P2 promotion rung;
  EX-5's gates scoped (per-direction byte accounting; vacancy recovery per applicable rung);
  EX-2r's early runnability surfaced in §14.1; dangling "correction 12" fixed to 5; D8
  reworded to "specified W1 behavior". Rejected from the reviews (with reasons the reviewers
  themselves partly supplied): the PlanIR either/or demand (arch §0 already frames this
  document as the staged contract), W1 ABI schema demands (implementation prerequisite, not a
  spec defect), and payload retention as a blocker (a named dependency, §14.3, not a current
  unavailability).
- **Revision 5 (2026-08-18, corpus expansion — nine checkouts, seven papers).** The missing
  official repositories identified after revision 4 were cloned and pinned (§0.2): swarm,
  pipedream, torchgpipe, powersgd, AC-SGD, CheckFree, DeMo, moshpit-sgd, asyncdiloco; papers
  [BIRCH], [EF-SGD], [EF21], [GPIPE], [PDREAM], [PSGD], [ALSGD] registered. Material
  consequence: **[PP-6] is re-classed P2 → P1 port-variant** — the SWARM prototype's
  `ExpertBalancer` (`swarm/.../moe/client/balancer.py`) is the routing heuristic's lineage
  implementation, with one declared code-vs-paper divergence (new-worker entry at the
  least-loaded queue head vs [AGORA-P]'s most-loaded level); §1.3 correction 1 amended.
  Provenance strengthened without re-classing: [PP-4] gains its published origin
  ([PDREAM]:350 + `pipedream/runtime/optimizer.py:19`), [PP-2]/[PP-3] gain [GPIPE] +
  torchgpipe, [CX-8] a third EF witness (official powersgd), [CX-9] its theory anchors
  (divergence without the residual — [EF-SGD]:53, [EF21]:125), [CX-10] the AC-SGD
  `DeltaCompressor` shape reference, EX-5's CheckFree option its official code, [PC-9] the
  moshpit-sgd evidence, [DR-3] a third named staleness-correction option (Delayed Nesterov,
  [ALSGD] — paper-grade; the official release is a toy notebook) and [BIRCH] as the
  async-side analysis framework. A.0/A.1/A.2/A.3/A.4/A.6 extended accordingly. A **regime
  note** added to §0.2: the datacenter-regime sources ([GPIPE], [PDREAM], [PSGD],
  [EF-SGD]/[EF21] and their checkouts) are admitted for mechanism origin and convergence
  theory only — never for feasibility or performance claims in the wild-internet target
  regime. No invariant, gate, or wave changed.
- **Revision 4 (2026-08-18, corpus-gap resolution).** LOSCAR-SGD arrived in the corpus and
  the register's one named gap is closed: [LOSCAR] key added to §0.2; [DR-3]'s "theory gap"
  note replaced with the theory anchor it was waiting for (delta application validated —
  delay-corrected merge beats naive overwrite, [LOSCAR]:1093; staleness cost is O(η²) and
  grows as sparsity shrinks, [LOSCAR]:301, 305) with its scope limits stated (round-structured,
  common-mask, data-homogeneous — not [AMESH]'s regime; rotation still analyzed by no paper);
  [DR-1] gains the same higher-order-staleness anchor. No mechanism or gate changed.
- **Revision 3 (2026-08-18, provenance-structure pass).** Structural response to the review
  finding that provenance information was scattered across five non-cross-referencing places
  (§0.3, clause bodies, §1.2, §1.3, Appendix A) and that Appendix A's source→mechanism keying
  answered the inverse of the question readers ask. Changes: §1.4 added as the **authoritative
  mechanism inventory** — one row per mechanism with provenance class (P0 in-tree / P1
  reference-ported / P2 paper-port / P3 authored / P4 reference-only), source, owner, and
  wave — with host prerequisites split out from SDK primitives and the invariant/gate register
  separated from mechanisms; §0.3 re-scoped as §1.4's nuance annex; Appendix A re-keyed —
  A.0 per-source ledger (what each checkout contributes and what is rejected from it), A.1
  mechanism→source primary index (the former Appendix C table, moved), A.2–A.10 the
  source-keyed line-pin dossier as secondary index, with evidence-only rows (PowerSGD, random
  projection) marked so they no longer read as supported; per-section mechanism/invariant
  classification lines added to §3–§12; a bf16 wire scalar added to [CX-5] (EX-5 assumed a
  wire codec §6 did not list). No mechanism, gate, or profile changed meaning.
- **Revision 2 (2026-08-18, literature-fidelity pass).** A review against the full
  `research-papers/` corpus corrected claims the code-only audit had over-generalized, every
  correction re-verified against the papers directly. Substantive: DiffusionBlocks re-scoped
  from "image-classification-only" (true of the checkout, false of [DBLOCK], which evaluates
  five architecture families including autoregressive LM) with EX-2 rewritten around the
  paper's AR construction, [BTRAIN]'s decentralized prototype evidence, the masked-diffusion
  band rule, and a new recurrent-depth variant; EX-5 recomposed after [NAG-PP]:306 ("weight
  stashing is not applicable in SWARM") invalidated its [PP-4] inclusion, and relabeled a
  novel VHC composition; [PP-5] inverted — the paper's fixed-β₁ NAdam + weight-space
  look-ahead is primary, the AsyncPP `adaptive_momentum` formula demoted to an implementation
  heuristic the paper's own ablation finds slightly worse; [DR-3] split into its three
  non-conflatable sources (SPARTA random subsets / AsyncMesh EMA-corrected delayed averaging /
  Agora AsyncSPARTA with implementation-only β^k) with sparsity `p` made a declared parameter;
  [DM-3]'s gate re-grounded in [CCD]/[MERGE1] (contraction recursion, scalar Θ estimator,
  phase-local critical consensus distance) and relabeled engineering acceptance policy;
  [PC-5]'s wire-cost claim removed (delta return is a numerical-stability device). Structural:
  §0.2 gains the paper register (with the LOSCAR-SGD corpus gap named), §7 states its
  complexity change precisely (linear aggregate, not sublinear), and a mechanism inventory
  mapping every mechanism to clause, code source, paper source, and port relationship was
  added as Appendix C (moved to Appendix A.1 by revision 3). New material: [CX-10]
  (AQ-SGD's change-compression constraint on activation
  codecs), [MB-7] (DES-LOC optimizer-state cadence and churn), [OQ-8]/[OQ-9], E3M0 wire
  format, CheckFree vacancy recovery, basis rotation for EX-4, [SCALE-DL] hyperparameters for
  EX-1.
- **Initial version (2026-08-18).** Authored from the post-C2 SDK audit (verified baseline §1),
  the architecture spec's reserved seams (Appendix B dispositions), and the audited reference
  corpus (Appendix A). Companion amendment applied to `vhc-architecture-spec.md` in the same
  change (cross-reference block; see Appendix B).

---

## 1. The verified baseline

Everything in this section is a **fact about the tree at the time of writing**, verified against
the code, not a design statement. Pins are `path:lines` under `daemon-node/`.

### 1.1 What ships

The shipping system is the **C2-certified VHC** — `da_abi` major 2, the current `crates/vhc`
tree, certified `terminal(0)` on 2026-08-11 (PS). Its *behavior* retains the degenerate
composition inherited from the v1 lineage: full-model-replica data parallelism, a single
implicit group, barrier rounds, SparseLoco consensus. The baseline is NOT "v1" — v1 is the spec
lineage (`swarm-training-spec.md`), not the running system. The distinction matters because the
compatibility gates in §3 are stated against the certified artifact, not against a document.

### 1.2 Code facts

- `BarrierRound` is the only round driver; the `RoundExperiment` trait has exactly the hooks
  `train_step` / `inner_update` / `make_update` / `ingest` / `begin_ingest`
  (`crates/vhc/sdk/daemon-vhc-sdk-rounds/src/lib.rs:105-126, 197-198`). Asynchronous ingest
  exists *inside* the barrier via `IngestOutcome::Deferred` + `finish_ingest` (`lib.rs:81-99,
  209-216`) — the working precedent §8's overlap driver generalizes. No async, quorum,
  staleness-bounded, gossip, or continuous driver exists.
- The guest ABI exposes `publish`, `payload_put/get`, `stream_open/accept/write/read`,
  `data@2::fetch`, buffers, fences, timers, cancellation, and state folds
  (`crates/vhc/sdk/daemon-vhc-sdk/src/abi.rs`; registry
  `crates/vhc/contracts/daemon-vhc-abi/src/lib.rs`). **No collective/reduce-shaped import
  exists.**
- Compute lane: `OperationIr::Float(Quantize/Dequantize)` → `ComputeError::Reserved`, and
  `OperationIr::Custom` → `CustomOpUnsupported` — `flash_attn@1` is admitted at manifest level
  but refused at dispatch (`crates/vhc/host/daemon-vhc-host/src/compute.rs:135-146`; conformance
  pin `tests/compute_conformance.rs:275-301`). The guest backend is f32-typed:
  `HostBackend::FloatElem = f32` (`crates/vhc/sdk/daemon-vhc-sdk-compute/src/lib.rs:169`). Wire
  IR is CBOR(`burn_ir::OperationIr`) at Burn 0.21.0.
- The production det lane is the **chunk-addressed streaming engine**: the tiny-llama guest
  links `SparseLocoIngestWalk`/`SparseLocoUpdateWalk`/`IngestFetch` and holds no resident
  canonical state — canonical master and error-feedback are host-side chunk-addressed families,
  the AdamW moments are device-resident but sealed window-at-a-time, and **peak guest memory is
  one window, never one family** (`crates/vhc/guests/tiny-llama/src/lib.rs:286-306, 1532`; a
  resident family at ceremony geometry is ~2.93 GiB). Its five hand-rolled completion-driven
  state machines are `IngestWalkState`, `UpdateWalkState`, `MomentSealWalk`, `RestoreState`,
  `CkptWalk` — the concrete consumers of §4.
- The resident profiles `SparseLoco`/`DiLoCo`/`Demo`
  (`crates/vhc/sdk/daemon-vhc-sdk-profiles/src/lib.rs`) are **dev/test-quarantined parity
  oracles**, retained as the live bit-identity oracle for the streaming engine. Their
  constructors take a `numels: &[usize]` layout; **the current production caller supplies the
  whole-model layout — the API itself accepts arbitrary layouts.**
- **Zero occurrences** of `group_id`, `group_round`, `vocab_version`, `plan@1`, `compress@1`, or
  `outer@1` anywhere in `crates/vhc` Rust code. The proto crates carry `resource_plan` /
  `execution_grant` / `execution_requirements` — resource accounting and grants, not topology.
  The manifest input is the narrow `ModuleDecl` (name, version, abi_minor, flat channel IDs,
  four memory tiers — `crates/vhc/sdk/daemon-vhc-sdk/src/module.rs:21-52`).
- The det-state layout is bound to **one whole-model parameter list**: `LayoutBinding::of_numels`
  hashes the canonical-CBOR numels list
  (`crates/vhc/contracts/daemon-vhc-proto/src/det_state.rs:439-455`), and `family_chunk_count` /
  `family_byte_len` / `family_chunk_lens` (`det_state.rs:147, 156, 181`) plus `fold_walk::windows`
  (`crates/vhc/sdk/daemon-vhc-sdk-consensus/src/fold_walk.rs:65-88`) key off it — as do genesis
  `expected_root` and every checkpoint family.
- The data cursor is global: `advance_cursor(data_index, gb, round)`
  (`crates/vhc/sdk/daemon-vhc-sdk-consensus/src/assignment.rs:278`). Throughput classes
  C1–C4 exist in assignment weights; the ceremony ran all-C1.
- No activation rematerialization exists anywhere in the guest compute/autodiff path.
- The `pipeline-stage` guest is a stream/credit tensor doubler with **no backward pass**
  (`crates/vhc/guests/pipeline-stage/src/lib.rs:4-24`) — a transport proof, not a pipeline
  runtime.

### 1.3 Corrections register

Findings that circulated during the audit and were corrected against source; recorded so they are
not re-derived wrong:

1. **Agora's worker routing has no implementation in the checkout — but it is published.** The
   min-heap least-loaded selection is described in `agora/docs/agora-system/
   training-architecture.md` and, in full detail, in the Agora paper: the trainer's balancer
   re-reads each stage's worker set from the DHT about every 30 s ([AGORA-P]:332), workers sit
   in a min-heap keyed by *accumulated virtual runtime* with the estimated task cost charged
   up-front and corrected by measured throughput, a new worker enters level with the
   most-loaded replica, and the backward pass is NOT load-balanced — it follows the forward
   path ([AGORA-P]:334). The trainer client is absent from the checkout, so agora supplies no
   code to port; [PP-6] is authored from [AGORA-P] (primary) with [SWARM-P] as the
   stochastic-rerouting lineage. (An earlier draft attributed the heuristic to the SWARM
   paper.) **Rev-5 update:** the official SWARM prototype is now checked out, and its
   `ExpertBalancer` (`swarm/swarm/pipeline/src/moe/client/balancer.py`) IS the lineage
   implementation of this heuristic — [PP-6] gains a code source (see the clause for the one
   divergence between the balancer and [AGORA-P]'s description). Agora's own trainer remains
   unpublished.
2. **AsyncPP's `adaptive_momentum` flag is an implementation heuristic, not the paper's
   method.** The flag computes `momentum += (num_versions − 1) · (0.99 − momentum) /
   num_stages` (`AsyncPP/main_with_runtime.py:308-314, 328-336`). The paper's method
   ([NAG-PP]) is NAdam *as is* with a large fixed β₁ = 0.99 ([NAG-PP]:154) under a modified
   look-ahead that discounts the gradient term by (1 − γ_t) ([NAG-PP]:120); its stage-dependent
   schedule is linear 0.9 → 0.99 across stages ([NAG-PP]:168); and its own ablation finds the
   adaptive formula "slightly worse" than fixed 0.99 for the stashing method ([NAG-PP]:274).
   See [PP-5].
3. **prime-diloco's "up to 8×" for layer-bucketed transfer is README prose**, not a measured
   assertion in code; the mechanism itself (sequential per-bucket reduce over grouped tensors,
   `prime-diloco/src/zeroband/diloco.py:108-112, 194-199`) is real.
4. **torchft's anti-split-brain quorum is a majority of currently-heartbeating replicas**, not of
   all registered peers (`torchft/src/lighthouse.rs:218-240`).
5. **The DiffusionBlocks *checkout* is image-classification-only; the *method* is not.** The
   pinned repo is ViT on CIFAR-100/Tiny-ImageNet with single-process DDP +
   `find_unused_parameters` (`DiffusionBlocks/main.py:67`), and no LM experiment exists *in the
   repo*. The paper evaluates five architecture families — "vision, diffusion, autoregressive,
   recurrent-depth, and masked diffusion" ([DBLOCK]:13) — including 12-layer Llama-style AR
   transformers on LM1B/OpenWebText (§5.4), masked diffusion on text8 at 1.45 BPC vs MD4's
   1.56 with 3× less memory ([DBLOCK]:194), and recurrent-depth Huginn ([DBLOCK]:485). An
   earlier draft of this document generalized the checkout fact to the method; EX-2 is written
   against the paper.
6. **Agora "forward recomputation at boundaries"** is the backward path's re-forward under
   autocast (`agora/agora_server/src/agora_server/core/server/module_collab.py:88, 109-114,
   122-125`), not a separate forward-recompute API. Same conclusion (recompute is mandatory for
   cross-process backward), different shape.
7. **hivemind's Hagenbach-Bischoff apportionment is vestigial upstream**: spelled
   `hagenbach_bishoff` (`hivemind/hivemind/averaging/load_balancing.py:89-103`) and marked
   `TODO(jheuristic) we no longer need hagenbach-bishoff with new AllReduceRunner` (`:32`). §7
   ports the LP cost model as load-bearing and treats integer apportionment as an implementation
   detail.
8. **Delta return does not reduce wire bytes.** hivemind returns `(aggregate − contribution)`
   "in order to improve numerical stability" (its own docstring); delta and aggregate have the
   same shape and dtype. An earlier draft claimed it halved the return leg's wire cost; [PC-5]
   now carries only the stability rationale.
9. **The λ 0.5 → 0.01 cosine schedule is AsyncMesh's, not SPARTA's.** [SPARTA] has no λ at all
   (its schedule is a sparsity-fraction ramp); the λ is [AMESH]'s EMA staleness-correction
   coefficient, published at [AMESH]:155 and absent from the AsyncMesh checkout (which cosines
   momentum instead).
10. **`k = 40` at `d = 4096` is Protocol Models' configuration, not Agora's.** [PM]:225 sets
    k = 40 (100× compression) on a d = 4096, ~2B model ([PM]:217). Agora production reports
    rank 40 at d = 2048 (1B) and rank ≈ 51 at d = 5120 (Pluralis-8B), and explicitly declines
    to claim a scaling law from two configurations ([AGORA-P]:378). See EX-5.
11. **SPARTA's exchange is asynchronous, not synchronous.** An earlier draft of [DR-3] called
    the [SPARTA] exchange synchronous. The paper is explicit that sparse communications "can
    be carried out asynchronously, thus not blocking workers": a parameter from step t−1 is
    shared while step t computes, at no wall-clock cost ([SPARTA]:23). The synchronous element
    in that paper is the DiLoCo H-boundary it stretches ([SPARTA]:29). What distinguishes
    [AMESH] is therefore not asynchrony per se but multi-step staleness τ and its correction.
12. **AsyncMesh's correction is an EMA-estimated average, not delta application.** An earlier
    draft said [AMESH] applies the stale average "as a delta". Its published rule estimates
    the *current* average as the stale average plus an EMA of local weight drift and writes
    that estimate over the selected coordinates ([AMESH]:107; the checkout's `weight_update`/
    `ema` rules, `sparta.py:74-139`, comment the same equations). "Delta application" is
    reserved for [LOSCAR]'s merge rule (`m_j + (z − y)`); both are corrected merges, distinct
    from the `avg` rule's naive overwrite.

### 1.4 Mechanism inventory and provenance *(authoritative)*

This subsection is the one-lookup answer to "what is being ported from what": every mechanism
this document puts in the SDK, its provenance class, its source, its owner, and when it lands.
Where a clause body and this table disagree, the clause body governs and the table has a bug.
Appendix A remains the evidence dossier: A.0 is the per-source ledger (what each checkout
contributes and what is rejected from it), A.1 the detailed mechanism→source index, A.2–A.10
the line pins keyed by source. Presence of a source in Appendix A does NOT mean the mechanism
is supported — this table is the support list.

**Provenance classes.** Provenance is risk ordering: P0 has a live bit-identity oracle, P1 has
a reference implementation to diff against, P2 has neither, P3 has no external validation at
all because it is ours by construction.

- **P0 — in-tree.** Generalize, extract, or re-express what the certified artifact already
  does. The gate is bit-identity against the existing parity oracles. Roughly a third of the
  ladder is P0 — these clauses port *nothing*, and describing them in porting voice was this
  document's earlier structural defect.
- **P1 — reference-ported.** A checkout implements it; semantics are ported against pinned
  lines. Qualifiers: *port* (substantially as-is), *adapted* (VHC-specific semantics — e.g.
  DHT metadata moved onto the existing control plane), *composite* (assembled from multiple
  sources; not attributable to one paper).
- **P2 — paper-port.** No implementation in any checkout: implement from the paper AND author
  the only tests (TDD §7's double debt). The highest-risk class.
- **P3 — authored.** VHC-specific invariants, gates, and policies. Not implementable units —
  constraints on implementations. These are what make P0–P2 safe.
- **P4 — reference-only.** Informative or coordinator-side; not an SDK primitive.

Experiment profiles (§13) are **research policy, not SDK mechanisms** — they appear here only
where they carry P2 debt.

#### SDK mechanisms (P0–P2)

Owner names abbreviate `daemon-vhc-*` crates (`proto`, `sdk`, `consensus` = `sdk-consensus`,
`codec` = `sdk-codec`, `collective` = `sdk-collective`, `rounds` = `sdk-rounds`,
`membership` = `sdk-membership`, `mesh` = `sdk-mesh`, `observe`).

| Clause | Mechanism | Provenance | Source | Owner | Wave → first profile |
|---|---|---|---|---|---|
| [GV-4] | per-group det-state layout bindings | P0 generalize | `LayoutBinding::of_numels`, whole-model today (§1.2) | proto | W1 → EX-2 |
| [GV-5] | per-group data cursors + deterministic reassignment | P0 generalize | `advance_cursor`, global today (§1.2) | consensus | W1 → EX-1 |
| [RT-1..3] | typed op layer, completion demux, combinators | P0 extract | the five walkers' hand-rolled bookkeeping (§1.2) | sdk | W2 → all |
| [CX-4] | det kernels as codecs (SparseLoco wire form) | P0 re-express | `daemon-vhc-det` kernels + shipping payload format | codec | W4 → EX-1 |
| [CX-5] | scalar quantisers: fp16/scaled-fp16, uniform/quantile/blockwise 8-bit, uint8+LUT, bf16 wire | P1 port | hivemind `compression/*`; prime-diloco | codec | W4 → EX-1 |
| [CX-5] | E3M0 4-bit outer-gradient wire format | P2 paper | [SDILOCO]:156 | codec | W4 → EX-1 |
| [CX-6] | DCT + top-k + index packing + 1-bit sign | P1 port | psyche `distro.rs` (algorithm: [DEMO-P]; official witness DeMo `demo.py`) | codec | W4 → EX-1 |
| [CX-7] | frozen-subspace activation codec | P1 adapted | agora `experts.py`; node0 `layers.py` (published: [AGORA-P]; the [PM] additions — `T_fixed` genesis, Grassmann refresh, optimizer coupling — are EX-5's separate P2 promotion rung) | codec | W4 → EX-5 |
| [CX-8] | checkpointable error-feedback residual | P0 extract, P1 semantics | streaming engine `ef` family; psyche EF loop; [DEMO-P] partial-α (theory: [EF-SGD], [EF21]) | codec | W4 → EX-1 |
| [PC-4] | ~512 KiB byte-bounded parts over disjoint window sets | P1 port | hivemind `partition.py` | collective | W4 → EX-1 |
| [PC-5] | delta return (numerical stability) | P1 port | hivemind `partition.py`, `allreduce.py` | collective | W4 → EX-1 |
| [PC-6] | minimax-LP reducer assignment | P1 port | hivemind `load_balancing.py` | collective | W4 → EX-1 |
| [PC-7] | layer-bucketed + uint8 ring part transports (optional) | P1 port | prime-diloco | collective | W4 → EX-1 |
| [DR-1] | overlap driver: prepare/commit split, fragment rotation | P0 generalize, P1 | `IngestOutcome::Deferred` (§1.2); torchft `local_sgd.py`; [SDILOCO] defaults | rounds | W5 → EX-1 |
| [DR-2] | cohort-scoped rounds | P0 generalize | `BarrierRound`, single implicit group today | rounds | W5 → EX-1/EX-2 |
| [DR-3] | sparse asynchronous driver (selectable partition + correction policies) | P1 composite | AsyncMesh `sparta.py`; agora selector + β^k (papers: [SPARTA], [AMESH]) | rounds | W8 → EX-6 |
| [DR-3] | EMA λ 0.5 → 0.01 cosine staleness schedule | P2 paper | [AMESH]:155 (absent from its checkout) | rounds | W8 → EX-6 |
| [DR-3] | Delayed-Nesterov staleness correction (optional policy) | P2 paper | [ALSGD]:9, 42 (official `asyncdiloco` release is a toy notebook) | rounds | W8 → EX-6 |
| [DR-4] | fp32-shadow pseudo-gradient convention | P1 port | prime-diloco `diloco.py`; OpenDiloco ([DILOCO-P]) | rounds | W5 → EX-1 |
| [MB-2] | two-phase ghost join | P1 adapted | agora `optimizer_sparta_async_nstep.py` (schema; DHT encoding dropped) | membership | W5 → EX-1 |
| [MB-3] | stale-epoch bump window, scoped by consistency class | P1 adapted | node0 `optim.py` | membership | W5 → EX-1 |
| [MB-4] | sample-weighted progress + ETA | P1 adapted | hivemind + agora progress trackers (DHT keys dropped) | membership | W5 → EX-1 |
| [MB-5] | state handoff via checkpoint families (+ optional p2p streaming) | P0 existing, P1 optional | checkpoint plane; hivemind averager / prime-diloco p2p (OPTIONAL) | membership | W5 → EX-1 |
| [XP-2] | typed activation frames + credit flow control | P0 generalize | existing stream receiver credits (§1.2); frame contract authored | sdk + host | W7 → EX-3 |
| [XP-3] | remote forward/backward bridge over [ML-5] | P0, P1 adapted | `pipeline-stage` guest (transport proof); hivemind `expert.py` | sdk + host | W7 → EX-3 |
| [PP-2] | recompute-mandatory backward | P1 port | AsyncPP `runtime.py` (origin: [GPIPE]:77; witness torchgpipe `checkpoint.py`) | mesh | W7 → EX-3 |
| [PP-3] | static fill/drain schedule + warmup sizing | P1 adapted | AsyncPP `runtime.py` (origin: [GPIPE]:109; witness torchgpipe) | mesh | W7 → EX-3 |
| [PP-4] | weight-stash ring (fixed-route pipes only) | P1 port | AsyncPP `optim/optimizer.py` (origin: pipedream `runtime/optimizer.py:19`, [PDREAM]:350) | mesh | W9 → EX-4 |
| [PP-5] | Nesterov weight-space delay correction, incl. no-stash O(N) form | P2 paper | [NAG-PP]:120, 154, 196 | mesh | W9 → EX-4/EX-5 |
| [PP-5] | `adaptive_momentum` stage-count heuristic (demoted variant) | P1 port | AsyncPP `main_with_runtime.py` ([NAG-PP]:274 ablates it worse) | mesh | W9 → EX-4 |
| [PP-6] | virtual-runtime min-heap routing + periodic worker discovery | P1 port (rev 5; was P2) | swarm `balancer.py` (published: [AGORA-P]:332-334; declared divergence on new-worker entry level) | mesh | W9 → EX-5 |
| [MO-2] | Θ scalar consensus-distance estimator; fixed-batch logit-JS | P2 paper | [CCD]:316-318, 444; [FG]:206-210 | observe + `metric@1` | W8 → EX-6 |

P2 debt carried by profiles rather than SDK clauses: EX-2's AR construction and α(t)
band rule ([DBLOCK], W6); EX-5's Protocol-Models promotion pair — Grassmann refresh and
row-constant AdamW with the per-iteration `W_p1` projection ([PM], W9+). Both are double debt.

#### Host prerequisites (not SDK primitives)

Owned by the host/coordinator; the SDK consumes them. Listing these as "SDK primitives"
obscures ownership.

| Clause | Prerequisite | Provenance | Wave |
|---|---|---|---|
| [ML-1] | `flash_attn@1` dispatch (admitted today, refused at dispatch) | P0 unblock | W3 |
| [ML-2] | bf16 compute lane for `local` trajectories | P3 authored (posture witness: agora) | W3 |
| [ML-3] | QFloat quantize/dequantize (lift out of `Reserved`) | P0 unblock | W3 |
| [ML-4] | blockwise 8-bit optimizer-state storage | P1 port (hivemind) | W3 |
| [ML-5] | stage-boundary recompute API | P1 adapted (agora; AsyncPP) | W3 → EX-3 |
| §10 host side | execution-plane channel networking (chunking, credits) | P0 generalize | W7 |
| [MO-1] | trajectory-monitor host service + `metric@1` slot | P3 authored | W8 |
| [MB-1] (host half) | membership views: leases, heartbeats, epochs, checkpoint logistics | P3 authored ([GR-6]) | W5 |

#### Invariants, gates, and policies (P3)

Not mechanisms — constraints that make the mechanisms safe. §0.3 is this class's nuance annex.

- **§2 [DM-1..9]** — the determinism and assurance model: restated architecture decisions
  ([DM-1..5], incl. [DM-3]'s authored phase-local disagreement gate) and the three-axes rule
  ([DM-6..9]).
- **§3 [GV-1]** vocabulary types; **[GV-2]** capability/placement boundary; **[GV-3]** host
  opacity; **[GV-6]** admission refusal (existing arch rule [LC-2], newly enforced per
  surface); gates **[GV-4g]**, **[GV-7]**.
- **§4 [RT-4]** walker-rebuild parity gate.
- **§5 [ML-6]** dependency guidance (only [ML-5] is a hard prerequisite).
- **§6 [CX-1]** static trait composition; **[CX-2]** det-crate boundary; **[CX-3]**
  channel-class binding; **[CX-9]** residual declaration rule; **[CX-10]** activation-codec
  convergence constraint ([AQSGD] anchor).
- **§7 [PC-1]** windows-as-reduction-parts identity (P0-backed: the fold-walk oracle);
  **[PC-2]** frozen committed set — no reducer membership discretion; **[PC-3]** post-record
  sequencing; gate **[PC-8]**; dispositions **[PC-9]**.
- **§9 [MB-1]** authority split (guest policy half); **[MB-7]** momentum-cadence declaration
  rule ([DESLOC] anchor).
- **§10 [XP-1]** deadline omission under the pinned [PL-1] reading (authored — no precedent,
  §0.3); **[XP-5]** idempotent dedup/late discard.
- **§11 [PP-1]** stage topology over §3 groups.
- **§12 [MO-3]** committed-statistics quantization discipline.

#### Reference-only (P4)

**[MB-6]** coordinator quorum semantics (torchft reference); **[PP-7]** PP×DP rank-layout
formula (informative); **[XP-4]** failure-semantics contract (arch §3.1 verbatim — owned
there, restated here).

#### Explicitly unsupported or deferred

- General PlanIR derivation (`plan@1` slot) — deferred, disposition D7.
- Runtime codec registry — rejected for now ([CX-1], [OQ-4]).
- Per-layer in-stage activation checkpointing — [OQ-2].
- Tensor parallelism and expert/MoE parallelism — no clause reserves them; hivemind's MoE
  machinery is not ported.
- KV/context-parallel codecs — vocabulary reserved only ([OQ-5]).
- Gossip-class exchange beyond [DR-3]; NoLoCo/Moshpit — dispositioned ([PC-9], [OQ-6]).
- PowerSGD and agora's random-projection compression — evidence witnesses in Appendix A, not
  supported codecs.
- MAPL, ResBM, full Protocol Models — open basis axis ([OQ-8]); EX-5 names the promotion gate
  for the Protocol-Models rung only.
- Verification/security/incentive mechanisms — out of scope (D14).

**The shape this table exposes** (and the prose hides): P2 — the class with no implementation
and no tests anywhere — is concentrated almost entirely in W8/W9/W10 and the profiles; the
one early exception is [CX-5]'s E3M0 wire format (P2, W4, deliberately small). W1–W6 is
otherwise purely P0 and P1: the early ladder is oracle-gated generalization of shipping code
plus line-pinned ports; the risk arrives late, and it arrives labeled.

---

# Part A — normative

## 2. Determinism and assurance model [DM]

This section restates the architecture decisions this document builds on — they are **already
made** (Appendix B, dispositions D1–D5) — and then pins the one embodiment rule the audits got
wrong twice.

### 2.1 The made decisions (restated, binding here)

- **[DM-1]** The **group round is the universal primitive** (arch [VP-12]). Progress is
  `(group_id, group_round)` plus the causal chain; **no global scalar step exists** in the
  extended vocabulary. The degenerate single-group plan makes its group round coincide with the
  current global round, so the generalization costs the certified baseline nothing.
- **[DM-2]** Consensus channels at the initial vocabulary carry **committed-set exchange** (arch
  [PIR-11]): the exchange artifact is a record-frozen, order-pinned committed set. Gossip-class
  exchange patterns are a **new exchange pattern** (arch rung 3), and MUST arrive with their own
  assurance treatment ([DM-3]) — never by relaxing committed-set semantics in place.
- **[DM-3]** Where a driver's consensus math leaves inter-replica disagreement D > 0 **by
  design**, digest equality yields to **measured, bounded disagreement** (arch [CO-8]),
  observed via §12 monitor signals, never assumed. A driver in this class (§8 [DR-3]) MUST NOT
  ship without this gate. The gate's shape is literature-informed but is **VHC engineering
  acceptance policy, not a theorem**:
  - The citable recursion is the consensus-distance form ([MERGE1]:670, Lemma D.1, after
    [CCD]): `E[Ξ²_{t+1}] ≤ (1 − p_mix/2)·Ξ²_t + O((1−p_mix)/p_mix)·η²(φ²_t + σ²)` —
    contraction factor set by the **mixing parameter `p_mix`** ([CCD] Assumption 1:
    `E‖XW − X̄‖²_F ≤ (1−p_mix)‖X − X̄‖²_F`), additive term driven by gradient norms and noise
    (not a constant). Mixing-operator contraction factors ρ are defined per operator
    ([FG]:77). `p_mix` is a property of the averaging operator; it is NOT §8's sparsity
    fraction `p_sparse` — two different quantities that this document names apart.
  - **When the two `p`'s coincide, and when they don't.** For a **random common mask over the
    full committed roster**, each coordinate is fully averaged with probability `p_sparse` and
    untouched otherwise, so the expected consensus error shrinks by exactly `(1 − p_sparse)`
    per exchange — `p_mix = p_sparse` identically. [AMESH]:137 states this outright ("sparse
    averaging shrinks the consensus error by a factor of (1−p) on expectation"), and the
    identity is what makes [LOSCAR]'s common-mask analysis and [CCD]'s arithmetic line up.
    Both configurations [DR-3] actually ships fall **outside** the identity: under
    **deterministic rotation** there is no per-round contraction factor (the selected
    partition contracts fully, the rest not at all — contraction is a property of the
    ⌈1/p_sparse⌉-round cycle, so Θ must be sampled per cycle, not per round), and under
    **[AMESH]-regime async delayed application** there is no doubly-stochastic per-step
    mixing operator at all, so Assumption 1 and the bound above do not apply. For both, the
    gate's target is **phase-local and empirical — declared by the profile, not derived from
    the recursion**.
  - "Must always contract" is deliberately NOT the gate. [CCD] shows the critical consensus
    distance is strictly positive ("we do not need perfect consensus", [CCD]:113), that only
    the **initial training phase** is pivotal, and that "large consensus distance in later
    training phases can even be beneficial" ([CCD]:31-35). The gate is therefore:
    **disagreement bounded against a phase-local target, with contraction demonstrated in the
    early phase** — a ρ̂ < 1 demand at all times would reject configurations the literature
    finds fine or better.
  - The gate's statistic is cheap: [CCD]'s Θ_t bounds the consensus distance
    (Ξ_t ≤ (2/p_mix)·Θ_t, [CCD]:436-438), each term is locally computable after one exchange,
    and the average needs an all-reduce over **scalars only** ([CCD]:444) — a [MO-2]
    statistic that never ships parameter vectors.
  - **Evaluability — what a consuming profile MUST declare** for this gate to be checkable at
    all: the phase boundaries (in group rounds), the phase-local disagreement bound, the
    measurement window and cadence (per exchange round; per rotation *cycle* under rotation),
    and the action on breach. A [DM-3] gate without these four declarations is unevaluable —
    W8's gate row requires the monitor live before [DR-3] ships, and EX-6 carries the
    declared values.
- **[DM-4]** **Planes decouple by clock** (arch [PL-1]): a schedule event, record, or consensus
  decision MUST NOT wait on an execution-plane transfer. [XP-1] states what this does and does
  not imply for pipelines — [PL-1] constrains the *consensus clock*, not a stage's local
  compute.
- **[DM-5]** State carries a **consistency class**: `local` (peer-private, never digested),
  `replicated` (canonical, digest-covered, version = committing round, arch [PIR-12]), and
  `residual` (peer-local algorithm state that MUST survive checkpoint/restore and MUST be
  declared — the stronger-than-`local` contract for error-feedback state).

### 2.2 The three axes (do not conflate)

Precision, plane, and aggregation determinism are **independent axes**. Every prior audit pass
conflated at least two of them; the following is the rule:

- **[DM-6] Compute precision** (local forward/backward numerics) is already legitimately
  divergent cross-peer — data shards and GPU-vendor numerics differ (v1 §5.6). A bf16 local
  trajectory is in the same category as a different GPU vendor: it changes **nothing**
  structural. In particular, bf16 local compute does NOT move the resulting `param_delta`
  channel outside the det class — the committed payload is still folded by deterministic
  consensus math.
- **[DM-7] Channel plane and wire dtype** are codec concerns. Payloads are opaque by invariant;
  a codec MAY emit bf16/int8/1-bit wire forms on any channel whose declared codec produces them
  (§6). Wire dtype never touches the determinism obligation of the fold that consumes the
  *decoded* values.
- **[DM-8] Aggregation determinism** (the agree-path: decode → clip → aggregate → outer update
  over the committed set) stays **fp32 det**, fixed order, exactly as the streaming fold engine
  pins it today (fold order ascending, window math window-local, digest carry sequential). The
  det fold is defined over f32-le images (`STATE_ELEM_BYTES`); this is the axis a wire or
  compute dtype never bends.

- **[DM-9] The backend-lane test, restated correctly.** The comment at
  `daemon-vhc-sdk-compute/src/lib.rs:170-175` concerns **i32 index tensors**, and its argument
  is: bit-identity holds *because* "an index dtype changes no f32 value." That sentence is the
  **test**, not a licence. bf16 is the first proposed change that FAILS the test wherever bf16
  values enter canonical or `replicated` state. Consequence: a bf16 compute lane ([ML-2]) is
  admissible for `local` trajectories under [DM-6], and inadmissible for canonical state bytes
  without a det-state contract change (which this document does not make — [OQ-3] carries the
  open question of precision heterogeneity within a cohort).

## 3. Contract vocabulary [GV]

Wave W1. Extends `daemon-vhc-proto` and `daemon-vhc-abi`. No porting source (§0.3). This section
is the embodiment of arch §4.2's PlanIR **output vocabulary** — adopted per disposition D6
(Appendix B); the `plan@1` **derivation slot** (arch §4.1) is explicitly deferred, because the
near-term instantiation path is authored genesis/run configuration, not in-guest derivation.

*Mechanisms: [GV-4], [GV-5] (both P0 generalize). Invariants: [GV-1..3], [GV-6]. Gates:
[GV-4g], [GV-7].*

### 3.1 Types

- **[GV-1]** `daemon-vhc-proto` gains: `GroupId` (dense u32 from 0, unique per plan), `GroupRound`
  (u64, scoped to a group — the [DM-1] clock), channel classes
  (`param_delta | activation | activation_grad | kv | metric`), planes
  (`consensus | execution`), consistency classes (`local | replicated | residual`, [DM-5]), and
  version tuples: `weight_version` (the group round that committed the weights a computation
  used), `basis_version` (the committing round of a shared transform such as a subspace basis),
  and per-channel staleness bounds (max acceptable `GroupRound` lag, u32). All canonical CBOR
  under the ABI §0.3 encoding rules.

### 3.2 The manifest/genesis boundary

- **[GV-2]** The module manifest declares **capabilities**, never placements: the channel
  classes the module can serve, the state slots it requires (name, consistency class, dims,
  dtype), and the group-role *shapes* it can fill (e.g. "any contiguous transformer-layer range
  with tied-embedding exclusion"). Concrete groups, assignments, replication targets, and
  channel wiring live in **genesis/run configuration and the execution context**, because they
  depend on the committed fleet snapshot. `ModuleDecl` is NOT extended with concrete groups.
  *(This resolves the W0 boundary question: capability = module-static, placement =
  run-dynamic.)*
- **[GV-3]** **Host opacity.** `GroupId` MAY surface as transport/journal metadata (routing,
  retention, telemetry). Algorithmic `group_round` content — what a round means, what its
  payloads contain — stays guest consensus schema. Production host crates MUST NOT decode SDK
  consensus frames to support groups; if a host feature seems to need frame contents, the
  feature is mis-assigned (arch VP-3, the seam rule).

### 3.3 Per-group state and data

- **[GV-4]** **Per-group det-state layout bindings.** Today one `LayoutBinding` covers the whole
  model (§1.2), and genesis root, checkpoint families, and the fold-walk schedule all key off
  it. The binding becomes **per-group**: each group binds the numels list of the parameters it
  *owns* (registration order within the group), and genesis/checkpoint/fold-walk machinery
  resolves `(group_id, binding)` pairs. Disjoint ownership (DiffusionBlocks blocks, pipeline
  stages) is thereby expressible without any group holding the whole-model list.
  **Gate [GV-4g]:** a degenerate single-group binding over the whole-model numels reproduces
  today's genesis root and every family root **byte-for-byte**.
- **[GV-5]** **Per-group data cursors.** `advance_cursor` generalizes to
  `(group_id, cursor)` state with the scoping rule arch A.4 already declares
  (`cursor_scope: per_group`): DP cohorts within a stage advance **distinct** cursors; stages
  within one pipe consume the **same** microbatch sequence (the pipe's head cursor is
  authoritative). Includes deterministic reassignment on membership change (cursor hand-off is
  coordinator policy over committed snapshots, not guest improvisation).

### 3.4 Enforcement and compatibility

- **[GV-6]** [LC-2] becomes implemented behavior: a host MUST refuse at admission any
  manifest/genesis exercising vocabulary it cannot yet enforce (execution-plane channels before
  §10 promotion, kv channels before a kv transport, monitor slots before §12 promotion). The
  refusal is typed (ABI §1.5 discipline).
- **[GV-7]** Two standing compatibility gates, from W1 onward:
  - **(a) Degenerate reproduction (the arch [LC-1] gate):** the single-group plan reproduces the
    certified baseline's behavior and digests bit-for-bit.
  - **(b) Frozen-artifact forward compatibility:** the certified C2 module artifact — whose hash
    predates the REL-6 fix and cannot be re-authored (PS non-claim 8) — with its narrow
    `ModuleDecl` still **admits unchanged** under the extended vocabulary. Backward
    compatibility of new shapes is not enough; the certified artifact's admission is a
    regression test.

## 4. Typed async runtime [RT]

Wave W2. In `daemon-vhc-sdk`. No porting source (§0.3). No new imports — this is a library over
the existing OpId/Completion ABI.

*Mechanisms: [RT-1..3] (P0 extract). Gate: [RT-4]. torchft's prepare/perform split is a
behavioral precedent for [DR-1], not a runtime port — nothing here is ported.*

- **[RT-1]** The SDK gains a typed operation layer: `Op<T>` (a typed pending operation),
  completion demultiplexing (route `Event::Completion` frames to their pending ops), typed
  cancellation, deadlines/TTLs, and bounded in-flight windows (issue-ahead ≤ N, the fold-walk
  discipline generalized).
- **[RT-2]** Combinators for the established sequences: fence → export → stream-write;
  ranged-read → fold-contiguous → seal; register → window-read-ahead. These encode the patterns
  the five walkers hand-roll today.
- **[RT-3]** **Scope: plumbing reuse, not elimination.** The tiny-llama walkers
  (`IngestWalkState`, `UpdateWalkState`, `MomentSealWalk`, `RestoreState`, `CkptWalk`) keep
  their transactional domain state machines — what they stop hand-rolling is issue/complete
  bookkeeping, contiguity tracking, and cancellation. The runtime carries no domain semantics.
- **[RT-4] Gate:** the det-digest parity suites are bit-identical before and after the walkers
  are rebuilt on [RT-1]/[RT-2]. The parity oracles (§1.2) exist precisely to make this
  refactor safe; the gate is non-negotiable.

## 5. Compute-lane memory levers [ML]

Wave W3 — **host work, starts immediately** (no dependency on §3; compute-lane changes touch no
manifest vocabulary). Nothing here is ported code — the host lanes are authored — but several
items have **behavioral witnesses and published origins** that guide the implementation
without sourcing it ([ML-2] agora's bf16 posture; [ML-4] hivemind's blockwise storage via the
[CX-5] port; [ML-5] the recompute pattern, origin [GPIPE]:77, witnesses agora/AsyncPP/
torchgpipe). §0.3 records the authored status; §1.4's host table records the witness
relationships.

*All of §5 is a host prerequisite, not an SDK primitive (§1.4 host table): mechanisms
[ML-1..5]; guidance [ML-6].*

- **[ML-1]** **`flash_attn@1` dispatch.** The custom-op registry admits `flash_attn@1` at
  manifest level; dispatch is refused (§1.2). Implement the IR dispatch. Effect: removes the
  quadratic materialized attention-score working set — and only that; other activation memory is
  bounded by planning, microbatching, and [ML-5], not by this op.
- **[ML-2]** **bf16 compute lane.** A negotiated backend float element type for *local*
  trajectories, governed by [DM-6]/[DM-9]: admissible for `local` compute, inadmissible for
  canonical state bytes. **Separate from [ML-3] in every respect** — bf16 is a float format
  negotiation, not tensor quantization. Reference posture: Agora runs bf16 forward with fp32
  parameters and gradients.
- **[ML-3]** **QFloat quantized tensors** — lift `Float(Quantize/Dequantize)` out of `Reserved`.
- **[ML-4]** **8-bit optimizer state** — blockwise absmax at the TDD golden constant (4096-block).
- **[ML-5]** **Stage recompute API.** The SDK exposes stage-boundary recompute over the
  **existing guest-side tape** — no Burn tape hooks: retain the detached stage input; on
  receiving the output gradient, re-run the stage's forward; invoke local backward. Witnesses:
  agora `core/server/module_collab.py:88` (backward), `:109-114` (detach), `:122-125` (bf16
  autocast re-forward); AsyncPP `runtime/runtime.py:77-82` (recompute asserted mandatory),
  `:565-591` (re-forward inside backward — the comment notes this is for activation/version
  *correctness*, not primarily memory). The pattern's published origin is GPipe's
  re-materialization ([GPIPE]:77), with `torchgpipe/torchgpipe/checkpoint.py` as a third,
  framework-clean witness. Per-layer in-stage checkpointing is deliberately out of
  scope ([OQ-2]).
- **[ML-6] Dependency structure (normative guidance).** Only [ML-5] is a hard prerequisite — of
  §10/§13 EX-3. [ML-1]–[ML-4] gate **scale tiers**, not correctness: no wave in §14 before W7
  waits on them, and a correctness-scale static pipeline runs f32 without them.

## 6. Codec layer [CX]

Wave W4, co-designed with §7. New crate `crates/vhc/sdk/daemon-vhc-sdk-codec`.

*Mechanisms: [CX-4..8]. Invariants: [CX-1..3], [CX-9]. Constraint: [CX-10]. The interface
([CX-1..3]) is authored; the implementations split P0/P1/P2 per §1.4.*

### 6.1 Shape

- **[CX-1]** **Static trait composition, not a runtime registry.** Codecs are Rust traits
  compiled into the module (the architecture's SDK-as-slot-crates model, arch §3): a module's
  manifest names the codec each declared channel class uses ([GV-2]); the binding is
  module-static. A runtime-pluggable registry is recorded as a rejected-for-now alternative
  ([OQ-4]) — it buys nothing while modules are pinned by blake3, and it costs the ability to
  reason about a module's wire behavior from its manifest.
- **[CX-2]** **`daemon-vhc-det` is never absorbed.** The codec crate *calls and re-exports* the
  det kernels; the normative dual-compiled consensus math stays where the contract boundary
  protects it (`crates/vhc/contracts/daemon-vhc-det/src/lib.rs` — `topk_chunk` :597,
  `absmax_pack` :455, `det_absmax_unpack` :384, `dct2` :638, `idct2` :650,
  `pack_chunk_indices` :535, `unpack_chunk_indices` :558, `det_chunk_scatter_add` :299).
- **[CX-3]** A codec binds to a **channel class**, not to a strategy name: `param_delta` codecs
  produce/consume committed-set payload bytes under [DM-8]; `activation`/`activation_grad`
  codecs produce execution-plane frames under [DM-7]; `metric` codecs produce §12 statistics.
  Codec identity (name@version + config) is part of the channel declaration and thereby of the
  admission decision ([GV-6]).

### 6.2 Codec inventory

- **[CX-4]** **Det kernels as codecs** (the existing SparseLoco wire form, re-expressed):
  chunked top-k + absmax pack (1/2/4/8-bit) + packed indices + 2-D DCT. Bit-identical to the
  streaming engine's current payload format — the gate is the existing parity fixtures.
- **[CX-5]** **Scalar quantisers**, ported: fp16 / scaled-fp16 (hivemind
  `compression/floating.py:10-40, 43-74`); uniform 8-bit with 6σ range and bucket-mean codebook
  (`compression/quantization.py:60-74`, `average_buckets` `:88-91`); quantile 8-bit (`:77-84`);
  blockwise 8-bit (`:130-148`, dequant `:199`); role/size-adaptive selection *interface*
  (`compression/adaptive.py:25-56`) as a trait, with selection policy left to the module.
  uint8 + per-bin lookup-table dequant as the reduce-side form (prime-diloco
  `src/zeroband/compression.py:54-70`; C kernel `csrc/compression.cpp:5-45`). **bf16**
  (round-to-nearest-even truncation of f32) as a declared wire scalar — trivial, but it must
  be *listed* because EX-5's boundary budget assumes a bf16 wire and a profile may not assume
  a codec this section does not name. **4-bit E3M0** (1 sign, 3 exponent, 0 mantissa) as the
  outer-gradient wire format, with accumulation in FP32 after receipt — Streaming DiLoCo
  reports no regression at billion scale ([SDILOCO]:156); paper-port, no code source.
- **[CX-6]** **1-bit sign** (psyche `distro.rs:687-692`,
  `quantize_nozeros_tensor_to_boolean_sign`) and the DCT+top-k transform pipeline as a codec
  (psyche `distro.rs:330-345` compress, `:370` decompress, `:397` batch_decompress, `:412`
  compress_idx with u8/u16/u32 index widths at 256/65536 thresholds). **The paper anchor for
  this family is DeMo** ([DEMO-P]) — decoupled momentum, DCT + top-k, momentum subtraction as
  error feedback, default 64×64 DCT chunks for matrix tensors ([DEMO-P]:183), and a
  sign-shaped final update for Signum-class optimizers ([DEMO-P]:125). The DisTrO preliminary
  report is NOT an algorithm source: it states "we currently do not use any compression and
  transmit the tensors directly" ([DISTRO-R]:131); psyche's code is the DeMo-lineage
  implementation this clause ports, and the paper's own standalone optimizer (`DeMo/demo.py`,
  with the OLMo reproduction patch) is the reference second witness for golden fixtures.
- **[CX-7]** **Subspace activation codec** (the §10/§11 enabler): shared frozen projection
  `rcv ∈ ℝ^{d×k}`, `compressed = rcvᵀ(x − fixed_embed)` with the token index carried as an
  extra channel. Two code witnesses: agora
  `agora_server/src/agora_server/models/reparam_llama/experts.py:113-127` (compress),
  `:129-141` (decompress), `rcv` at `:74`, frozen embed `:103-104`, basis load `:145-151`; node0
  `src/node0/models/llama/layers.py:502-517, 519-531`, subspace weight projection `:550`. The
  basis is a `replicated` state whose version is its committing round ([DM-5], arch [PIR-12]) —
  loaded as a pinned genesis artifact in the frozen-basis profile (§13 EX-5). Couplings the
  Protocol-Models form adds, both load-bearing:
  - **`T_fixed` is a genesis artifact too**: "at the beginning of training, T_fixed is
    transmitted to all nodes and stored" ([PM]:105), and compression subtracts both the
    positional encoding and the token embedding (`X̂ = X − PE − TE`, [PM]:79) — the token-index
    side channel exists to index `T_fixed` on the receiving side.
  - **Row-constant AdamW covers only half the optimizer coupling**: it keeps `W_p2` in the
    subspace without projection, but "we still need to project W_p1 onto S due to the
    nonlinearity of activations" — every iteration ([PM]:143).
  Rank provenance (§1.3 correction 10): k = 40 at d = 4096 is [PM]'s configuration (100×,
  [PM]:225); Agora production used rank 40 at d = 2048 and ≈51 at d = 5120 with an explicit
  no-scaling-law disclaimer ([AGORA-P]:378). `k ≈ d/100` is a starting heuristic; rank is a
  per-run hyperparameter. The basis *strategy* itself is an open design axis ([OQ-8]), not a
  settled progression.

### 6.3 Error-feedback residual state

- **[CX-8]** The error-feedback accumulator is extracted from `SparseLoco` into a reusable
  **`residual`-class state slot** ([DM-5]): versioned, declared in the manifest, checkpointed
  with the state families, restored on rejoin. Code semantics per the psyche reference —
  decode peer transmissions and subtract from the residual, decay, add `lr·grad`, transform +
  top-k (`psyche/shared/modeling/src/distro.rs:505` generate, `:555-586` peer subtraction,
  `:618` apply, `:669` error_correction; container `DistroResult` `:442`). Paper semantics per
  DeMo: subtraction of communicated values is **partial**, coefficient α — the ablation finds
  α = 0.2 improves over full subtraction (α = 1) by letting the top-k set evolve gradually,
  and pairs with a large momentum decay (β = 0.999) when subtraction is on ([DEMO-P]:226,
  183).   α is a declared codec parameter. PowerSGD-style `m.add_(grad)` before /
  `m.sub_(new_m)` after factorization as the low-rank witness pair
  (agora `core/averaging/gradient_averager_powersgd.py:226-283`; hivemind
  `optim/power_sgd_averager.py:146-183`; upstream-official third witness
  `powersgd/powersgd/powersgd.py:149, 211, 221` — inputs are modified in place to hold the
  approximation error for feedback).
- **[CX-9]** Residuals are NEVER incidental optimizer buffers: a codec that carries residual
  state MUST declare the slot, and a checkpoint that omits a declared residual is invalid. (The
  streaming engine's `ef` family is the existing precedent — this clause generalizes it. The
  empirical basis is now also published: with a fixed basis, no subtraction (α = 0) repeatedly
  re-selects and re-sends the same top-k elements and "is detrimental" — [DEMO-P]:226.) The
  theory behind the severity is error-feedback theory: biased compressors (sign, top-k)
  **do not converge in general without the residual** — sign compression provably diverges on
  convex counter-examples and EF restores the full SGD rate ([EF-SGD]:31, 53), and Top-1
  without EF exhibits *exponential divergence* on a 3-quadratic example ([EF21]:125). A lost
  residual is therefore not a degraded state, it is a different (possibly non-convergent)
  algorithm — which is why the slot is checkpoint-mandatory, not best-effort.
- **[CX-10]** **Activation codecs carry a convergence constraint the delta family does not.**
  The only convergence guarantee for compressed pipeline activations in the corpus is AQ-SGD,
  and it compresses the **changes** of activations against a per-sample buffer — not the
  values: quantizing values directly relies on unbiasedness assumptions "that do not hold for
  deep learning models with non-linear activation functions", and in AQ-SGD's experiments
  direct quantization "fails to converge" where change-quantization trains at 2–4 bits
  ([AQSGD]:17, 71). Consequence: a lossy `activation` codec claiming a convergence story needs
  **per-sample residual state** (a `residual`-class slot keyed by sample identity) — a
  materially bigger contract than [CX-8]'s per-parameter residual, with data-shuffling and
  memory implications. The official AC-SGD code is now in the corpus and shows the shape of
  that contract: `DeltaCompressor` (`AC-SGD/compress/delta_modules.py:17`) keeps an activation
  cache read/written by `sample_ids` (`:86-125`) around compress/decompress (`:126, :140`).
  Recorded as a constraint on [XP-2] codec selection and as open question
  [OQ-9]; the frozen-subspace codec ([CX-7]) is *not* in this class (it is a fixed linear
  projection, not a stochastic quantizer).

## 7. Partial collectives [PC]

Wave W4, co-designed with §6. New crate `crates/vhc/sdk/daemon-vhc-sdk-collective`. Discharges
arch [CO-6] (v2.0's aggregation is all-download-all; the seam was shaped so the fix is
staging-side — this is that fix).

*Mechanisms: [PC-4..7] (all P1 ports). Invariants: [PC-1..3]. Gate: [PC-8]. Dispositions:
[PC-9].*

**What the fix buys, stated precisely.** For N contributors each committing a payload of m
bytes: all-download-all costs **O(N·m) per peer** and **O(N²·m) aggregate**; the partitioned
reduce costs each contributor O(m) up + O(m) down, each of R reducers O(N·m/R) fetch, so
**O(m·(1 + N/R)) per peer** (≈ O(m) at R ≈ N) and **O(N·m) aggregate**. That is a linear
aggregate cost in roster size — an N× reduction over the baseline, but NOT mathematically
sublinear; [CO-6]'s "sublinear" is discharged as *per-peer cost sublinear in roster size*.
The download-skew motivation is quantified in the field: at 32 nodes, DisTrO-style exchange
uploads 2.8 MB but downloads 86.8 MB per step per peer — an asymmetry the report calls
advantageous only because consumer links skew toward download ([DISTRO-R]:131); the
partitioned reduce removes the O(N) download instead of leaning on it.

### 7.1 Invariants

- **[PC-1]** **Reduction parts ARE fold-walk windows.** The part identity is the window ordinal,
  which by construction equals the family chunk ordinal
  (`fold_walk.rs:46-59` — "the identities coincide by construction"). Windows partition each
  parameter ascending; fold order is ascending always; per-window math is window-local. A
  collective that honours [PC-1] is bit-identical to the existing det fold over the same
  committed set — the property the parity suites protect. VHC-specific; no porting source
  (§0.3).
- **[PC-2]** **The collective is a transport optimization over a record-frozen committed set.**
  No reducer has membership discretion. hivemind's mid-flight sender/reducer exclusion
  (`averaging/allreduce.py:115, 198, 319-320`; `partition.py:128` `register_failed_reducer`,
  `:248` `on_sender_failed`) is **explicitly rejected**: different reducers excluding different
  senders produces a different committed set per window, which breaks [DM-8]. Sender failure is
  handled *before* the record (the peer's payload is not listed) or *after* it by the existing
  straggle/fetchability ladder — never by per-reducer improvisation.
- **[PC-3]** **Sequencing (decided): reduction runs post-record; reducers range-fetch their
  assigned windows from content-addressed payloads.** Consequences, each load-bearing:
  - the final contributor set is known (the record froze it);
  - contributors need not stay online (availability = payload fetchability + the straggle
    ladder, the machinery that already exists);
  - canonical contributor order within each window is **record order** (the order
    `Committed::mint` already pins);
  - reducer election and bandwidth inputs are deterministic functions of the **committed
    admission snapshot** — quantized classes, never raw floats (arch [PD-2] discipline).
  Direct pre-record peer reduction is recorded as a rejected alternative: it cannot know the
  final set, so it re-imports the [PC-2] problem.

### 7.2 Geometry and mechanics

- **[PC-4]** Byte-bounded parts: windows group into reducer assignments of ~512 KiB
  (hivemind's `DEFAULT_PART_SIZE_BYTES = 2**19`, `partition.py:17`; splitting mechanics
  `:21-90`). The window schedule is already pinned; a part is a **disjoint set of window
  ordinals** — contiguous runs are the common case, but strided/interleaved sets are equally
  admissible ([PC-1] needs the part sets to partition the family and the fold order to stay
  ascending, not contiguity). This matters because Streaming DiLoCo's default fragment pattern
  is **strided** — interleaved transformer blocks, chosen over sequential for compute
  utilization ([SDILOCO]:95-97) — and [DR-1]'s fragments are [PC-4] parts.
- **[PC-5]** **Delta return**: reducers return `(aggregate − contribution)` deltas, not full
  values (hivemind `partition.py:114-136`, `allreduce.py:201-210`, serialize `:353-356`). The
  rationale is **numerical stability of the apply side** (hivemind's own docstring); delta and
  aggregate have the same shape and dtype, so this saves no wire bytes (§1.3 correction 8).
- **[PC-6]** Bandwidth-proportional reducer assignment via the LP cost model (hivemind
  `load_balancing.py:13-33` entry, `:36-86` `optimize_parts_lp` — client + aggregator cost as a
  minimax LP). Integer apportionment is an implementation detail (§1.3 correction 7 — upstream
  marks `hagenbach_bishoff` vestigial). Inputs come from the committed snapshot per [PC-3].
- **[PC-7]** Options, not requirements: layer-bucketed parallel transfer (prime-diloco
  `diloco.py:194-199` grouping, `:108-112` per-bucket reduce) and a uint8 ring-reduce lane
  (`collectives.py:31-43` dispatch, `:88-146` double-buffered ring with lookup-table
  accumulate) — both admissible under [PC-1]–[PC-3] as alternative part transports.
- **[PC-8] Gate:** for every committed set, the collective's sealed result is bit-identical to
  the all-download-all det fold of the same set. This gate is permanent, not a bring-up
  artifact.
- **[PC-9]** **Alternatives to the whole premise, dispositioned rather than ignored.** Two
  published designs remove the partitioned collective instead of optimizing it: NoLoCo drops
  collectives entirely — pairwise averaging folded into a modified Nesterov update, sync
  latency 2t_c versus ~2t_c·log₂(n) for tree all-reduce ([NOLOCO]:217-219) — and Moshpit runs
  all-reduce inside randomly re-drawn capped groups with an averaging rate that is exponential
  and topology-independent ([MOSHPIT]:46, Theorem 3.2 at `:176`; official experiment code is
  the `moshpit-sgd` checkout — the group-matchmaking/averaging machinery itself lives upstream
  in hivemind, A.3). Both are **deferred, not rejected**: each is compatible with [PC-2]'s "no reducer has membership discretion" only if
  the pair/group assignment is a deterministic function of the committed snapshot rather than
  ad-hoc peer discovery, and both change the exchange pattern — which makes them [DM-2]
  rung-3 material with a [DM-3] obligation, not drop-in [PC] transports.

## 8. Sync drivers [DR]

Waves W5 ([DR-1], [DR-2], [DR-4]) and W8 ([DR-3]). Extends `daemon-vhc-sdk-rounds`.

*All four clauses are mechanisms; [DR-3]'s [DM-3] gate obligation is the section's one
invariant-grade demand.*

- **[DR-1]** **Overlap driver.** Generalizes the existing `IngestOutcome::Deferred` /
  `finish_ingest` precedent (§1.2) into a full prepare/commit split: communication for round
  H's update MAY begin at inner step H − δ (prepare), while the commit — the decision that the
  round's aggregate is the one applied — remains explicit and record-gated (perform). Reference:
  torchft `local_sgd.py:404` (`prepare_sync`), `:421` (`perform_sync`), fragment scheduler
  `:563-677` (staggered prepare at `sync_every − delay`), `:759-778` (trigger); commit ceremony
  `manager.py:855-943` (`should_commit` — an optimizer MUST NOT step on a failed round).
  Includes fragment/shard rotation (Streaming-DiLoCo shape): one round's bandwidth spread
  across the inner window, fragments identified by **disjoint window sets** under [PC-1]
  identities — sequential runs or strided sets; [SDILOCO]:95-97 defaults to **strided**
  (interleaved blocks) and holds fragment *size* fixed (3 layers, [SDILOCO]:278) so fragment
  count grows with depth. Paper defaults worth porting as the profile's starting values: a
  small overlap delay τ (τ = 1 in the main experiments, [SDILOCO]:330) and E3M0 outer-gradient
  wire quantization ([CX-5]). Theory anchor: [LOSCAR] proves (smooth non-convex) that
  overlap-induced staleness enters only as a higher-order O(η²) disagreement term
  ([LOSCAR]:293, 305) — the overlap window is not a leading-order convergence cost.
- **[DR-2]** **Cohort-scoped rounds.** `BarrierRound` parameterized by group: roster, record
  channel, assignment, and digests all scope to `(group_id)` ([DM-1] machinery). This is
  deliberately small — the round logic is unchanged; only the roster source and the digest
  scope move — and it is the composition point every model-parallel layout reuses (within-stage
  DP in §11 is [DR-2] on the stage shard under a [GV-4] per-group binding).
- **[DR-3]** **Sparse asynchronous driver (a VHC composition)** — the first rung-3 exchange
  pattern ([DM-2]), and the first driver in the [DM-3] class. It is deliberately NOT named
  after any single source, because it composes three that must not be conflated:
  - **[SPARTA] (paper):** subsets sampled **uniformly at random** each step ([SPARTA]:84) at
    p_sparse = 0.05% ([SPARTA]:82), exchange **asynchronous and overlapped** — a parameter
    from the previous step is shared while step t computes, costing no wall-clock
    ([SPARTA]:23; the synchronous element in that paper is the DiLoCo H-boundary it
    stretches, [SPARTA]:29 — §1.3 correction 11); probabilistic coverage 1 − (1−p)^n
    ([SPARTA]:159); its contribution is the correlation effect that lets DiLoCo's H stretch
    100× ([SPARTA]:171).
  - **[AMESH] (paper):** **asynchronous delayed** sparse averaging at 5% subsets with
    multi-step staleness τ, corrected by an **EMA-estimated average**: the new average is
    approximated as the stale average plus an EMA of local weight drift ([AMESH]:107), and
    that estimate replaces the selected coordinates — NOT a delta application in [LOSCAR]'s
    sense (§1.3 correction 12); EMA coefficient λ cosined 0.5 → 0.01 after 1k iterations
    ([AMESH]:155), convergence guarantees, and the claim of strictly generalizing eager
    DiLoCo ([AMESH]:119).
  - **Agora/AsyncMesh (code):** rotation-based partition selectors (AsyncMesh
    `sparta/sparta.py:142-179` — `PartitionedIndexSelector` at `:156-168`; agora variant
    `core/averaging/state_averager_sparta.py:47-60`), delayed buffer + `avg`/`weight_update`/
    `ema` rules (`sparta.py:74-139`), and β^k optimizer-state scaling for a k-epoch gap (agora
    `state_averager_sparta.py:70-94, 102-108`; `beta_k_correction` `:75, 87, 93`) — **β^k is
    implementation-only evidence**, established by no paper in the corpus.
  Two variant axes are therefore **declared plan parameters**, not baked-in choices:
  - **Partition policy** — random-per-step (the [SPARTA] mechanism) vs deterministic rotation
    (the checkouts' mechanism). VHC selects **rotation as a declared divergence**: it is
    det-lane-reproducible from `(group_round, seed)`, and coverage becomes exact in ⌈1/p⌉
    rounds instead of probabilistic.
  - **Staleness correction** — EMA λ (published, [AMESH]) vs β^k (implementation-only) vs
    Delayed Nesterov + dynamic local steps (published for async Local-SGD LM training,
    [ALSGD]:9, 42 — matches synchronous DiLoCo's perplexity-per-step at ≤150M scale; the
    official `asyncdiloco` release is a toy notebook, so a port is paper-grade).
  - **Sparsity `p_sparse` itself** — the sources differ by 100× (0.05% vs 5%), and since the
    contraction behavior scales with `p_sparse` (exactly, in the random-common-mask case
    where it equals the mixing parameter; only cycle-wise under rotation — [DM-3]), the
    [DM-3] gate is unevaluable for a profile that does not declare it.
  **[DR-3] MUST NOT ship without its [DM-3] gate consuming §12 monitor statistics; building
  measurement inside the driver is forbidden** (arch [AL-3]: the monitor is the shared
  statistical substrate, not a second pipeline). Theory anchor (the former corpus gap, §0.2):
  [LOSCAR] is the convergence analysis [AGORA-P]:1603 pointed at, and it validates the
  driver's two structural choices — **corrected merge instead of naive overwrite**
  ([LOSCAR]'s delay-corrected delta rule keeps local overlap progress and corrects only the
  synchronization disagreement, `m_j + (z − y)` on synced coordinates, [LOSCAR]:200, 247-249;
  [AMESH]'s EMA-estimated average is the other published corrected-merge form, and the
  checkouts' `avg` rule — naive overwrite — is empirically strictly worse, [LOSCAR]:1093,
  most under long delay, [LOSCAR]:339) and
  **bounded higher-order staleness cost** (all disagreement terms are O(η²), amplified as
  sparsity `p_sparse` shrinks through q = 1 − p_sparse — [LOSCAR]:293, 301, 305 — which is
  exactly why `p_sparse` must be declared for the [DM-3] gate). Scope limits, so the anchor
  is not over-claimed:
  [LOSCAR] is round-structured with a server message per round (between Local SGD and
  fully-async, [LOSCAR]:255-257), uses one **common random mask across all workers**
  ([LOSCAR]:215) — the [SPARTA]-style random policy, not the checkouts' rotation — and
  assumes data-homogeneity ([LOSCAR]:54). [AMESH]'s fully asynchronous regime keeps its own
  guarantees; rotation remains a declared divergence with no analysis in either paper. On the
  fully-asynchronous side, [BIRCH] now supplies the general framework (computation trees over
  local + asynchronous SGD variants, with optimal-complexity Async-Local SGD members,
  [BIRCH]:19, 53) — an analysis language for the driver's async regime, though it does not
  treat sparsification.
- **[DR-4]** Pseudo-gradient convention shared by [DR-1]/[DR-3]: `Δ = shadow − local` computed
  against an offloaded fp32 shadow, outer step on the shadow, copy-back (prime-diloco
  `diloco.py:78-99` pseudo-grad, `:204-209` outer step + `sync_inner_model`; OpenDiloco
  `hivemind_diloco.py:158-167`; overlap scheduling precedent `:121-132, 722-738`).

## 9. Membership and elasticity [MB]

Wave W5. New crate `crates/vhc/sdk/daemon-vhc-sdk-membership` (guest policy) + host/coordinator
mechanism. No DHT — semantics ride the existing control plane.

*Mechanisms: [MB-2..5]. Invariants: [MB-1] (authority split), [MB-7] (cadence declaration
rule). Reference-only: [MB-6].*

- **[MB-1]** **Authority split (normative).** The host/coordinator owns membership **views**:
  leases, heartbeats, group assignment, epoch boundaries ([GR-6] semantics), and
  checkpoint-handoff logistics. The guest SDK owns membership **policy**: ghost contribution
  weights, stale-update acceptance, warm-up behavior. A guest MUST NOT independently decide
  authoritative membership; a host MUST NOT interpret policy ([GV-3]).
- **[MB-2]** **Ghost-mode join** — the two-phase state machine: phase 1 receive-only (no
  batches, aggregates in, contribution weight 0), phase 2 contributing at weight 0, then full
  participation (agora `core/optimization/optimizer_sparta_async_nstep.py:50-57` docs, `:63-86`
  ghost parameters, `:82-85` join blocking, `:104-115` staleness→ghost re-entry). The
  membership metadata *schema* (phase tags on the peer record) ports; the DHT encoding
  (`core/server/dht_handler.py:39-50, 205-227`) does not — phase tags ride the existing
  control-plane peer records.
- **[MB-3]** **Stale-epoch bump, scoped by consistency class.** A peer ≤ `max_allowed_stale`
  epochs behind MAY bump its local epoch without full restore **only for `local` and
  `residual` state**; canonical `replicated` state whose missing rounds change the
  deterministic trajectory MUST take the existing `StaleRestore` path. Reference: node0
  `src/node0/server/optim.py:80-88` (bump window vs reload), `AutoStepOptimizer` `:36-56`.
- **[MB-4]** **Sample-weighted progress.** Group progress is measured in samples against a
  target batch size, not in fixed per-peer step counts — the mechanism that makes C1 and C4
  peers simultaneously useful (hivemind `optim/progress_tracker.py:44` schema, `:132` local
  report, `:195` epoch increment, `:235-237` DHT key shape (not ported), `:281-287` ETA; agora
  variant `core/optimization/progress_tracker.py:59-72, 182-214`). Assignment continues to use
  the committed C1–C4 classes; progress reporting becomes contribution-weighted.
- **[MB-5]** **State handoff: the content-addressed checkpoint plane is primary.** A joiner
  restores from checkpoint families (already chunk-addressed, already resumable); peer-to-peer
  state streaming (hivemind `averaging/averager.py:601` entry, `:628` `rpc_download_state`,
  `:653-689` donor priority + chunking; prime-diloco `checkpoint.py:443-510`
  `recv_ckpt_from_peer`/`send_ckpt_to_peer`) is an OPTIONAL optimization behind the same
  restore interface, not a requirement.
- **[MB-6]** Elastic-quorum reference semantics (for the coordinator side, not the SDK):
  participant majority is computed over currently-heartbeating replicas (torchft
  `src/lighthouse.rs:141-180`, majority rule `:218-240` — §1.3 correction 4), recovery sources
  assigned round-robin from up-to-date peers (`src/manager.rs:568-584`).
- **[MB-7]** **Optimizer-state sync cadence is a churn question, not only a bandwidth one.**
  DES-LOC's analysis decouples parameter sync (probability p_x) from momentum sync (p_u):
  momentum averaging "can be turned off entirely (p_u = 0) without affecting the asymptotic
  behavior of the rate", while vanishing parameter sync breaks it — though more momentum
  averaging admits larger step sizes ([DESLOC]:173). The same source warns that keeping
  optimizer states **purely local** "accumulates noisy small-batch gradients and does not
  provide a means of adding new workers", making it "unsuitable for environments prone to
  random system failures" ([DESLOC]:47). Consequences here: a joiner needs a declared momentum
  bootstrap path — [MB-5] restore of checkpointed optimizer families or a [MB-2] ghost warm-up
  long enough to rebuild moments — and a profile that never syncs optimizer state MUST say how
  its joiners bootstrap. Momentum sync cadence joins `p`/τ as a declared profile parameter
  (multi-timescale cadences per [MTDAO] are admissible under the same declaration rule).

---

# Part B — vocabulary-complete, enforcement-deferred

Sections 10–12 are specified now so the §3 vocabulary, manifests, and journals carry the right
shapes from day one (the same reasoning arch [AL-3] applies to the monitor's statistics). Their
clauses bind a conforming implementation only after the named **promotion criterion**; until
then a host refuses plans exercising them ([GV-6]).

## 10. Execution-plane transport [XP]

**Promotion criterion:** the W7 gate (§14) — EX-3's loss-parity result on real transport.
This is the arch [LC-2] lift: the arch spec's Appendix A.4 four-stage pipeline manifest becomes
admissible.

*Mechanisms: [XP-2], [XP-3]. Invariants: [XP-1] (authored, no precedent), [XP-5]. Contract
restated from arch §3.1: [XP-4].*

- **[XP-1]** **The [PL-1] reading, corrected and pinned.** [PL-1] requires that *schedule
  events, records, and consensus decisions* never wait on an execution-plane transfer. It does
  NOT prohibit a stage's local backward from waiting on an activation gradient — local compute
  is not consensus. The conformance shape is therefore: **independent group clocks plus
  deadline-based omission** — a late activation gradient leads to omission of that microbatch's
  contribution (or reroute per [XP-4]), never to a stalled group round. A structurally
  drain-free 1F1B is NOT required; a 1F1B whose *record commit* gates on drain is
  non-conformant. No precedent exists — all three pipeline reference repos gate the step
  boundary on drain (§0.3); this clause is authored, not ported.
- **[XP-2]** **Typed frames.** Execution-plane channels carry typed frames: tensor metadata +
  schema hash, `(microbatch_id, weight_version)` correlation ([GV-1] tuples), chunking and
  reassembly, receiver credits (the existing stream flow control), bounded in-flight bytes, and
  the channel's declared codec ([CX-3] — `activation`/`activation_grad` class). Frames carry
  **no round-commit obligations at the type level**: nothing in an execution frame can name a
  consensus decision ([XP-1] enforced structurally).
- **[XP-3]** **Backward path.** Forward: send activation (codec-compressed). Backward: receive
  activation gradient, re-forward the stage under [ML-5], local backward. Remote-autograd
  bridge reference: hivemind `moe/client/expert.py:194-221` (`_RemoteModuleCall`), with the
  unary-vs-streaming selection by payload size at `:155, 188`. The `pipeline-stage` guest
  (§1.2) is the credit/stream transport proof this section upgrades.
- **[XP-4]** **Failure semantics** (arch §3.1, verbatim contract): reroute to a replica;
  bounded-staleness rejection (a frame whose `weight_version` lag exceeds the channel's
  declared bound is dropped, counted, and reported to §12); **never blocks the control plane**.
- **[XP-5]** Duplicate and late frames are idempotent-discard: `(channel, microbatch_id,
  weight_version)` is a dedup key; a frame arriving after its deadline is dropped under [XP-1]
  omission accounting, not re-ordered into the past.

## 11. Pipeline machinery [PP]

**Promotion criterion:** [PP-1]–[PP-3] promote with W7 (static pipeline, EX-3); [PP-4]–[PP-6]
promote with W9 (async pipeline + SWARM, EX-4/EX-5). New crate
`crates/vhc/sdk/daemon-vhc-sdk-mesh`.

*Mechanisms: [PP-2..6]. Invariant: [PP-1]. Informative: [PP-7]. Note the provenance split
inside [PP-5]: the paper method is P2, the AsyncPP heuristic is P1 (§1.4).*

- **[PP-1]** **Stage topology over §3 groups.** A pipeline is groups + `activation`/
  `activation_grad` channels + per-stage `param_delta` channels — exactly arch A.4's shapes
  ("note what is absent: no 'PP' anywhere"). Stage ownership is a [GV-4] per-group layout
  binding; within-stage DP is [DR-2] on that binding.
- **[PP-2]** **Recompute-first backward** ([ML-5]): stage inputs are retained detached;
  activations between microbatches are not (AsyncPP asserts recompute mandatory —
  `runtime/runtime.py:77-82`).
- **[PP-3]** **Static schedule (W7 form):** GPipe-shaped fill/drain within a step, [XP-1]
  conformant at the commit boundary. The published origin is [GPIPE]:109 — micro-batches
  pipelined through the cells, gradients accumulated across all micro-batches and applied
  synchronously at mini-batch end (update consistency independent of partition count) — with
  `torchgpipe` as a clean reference implementation (`torchgpipe/gpipe.py`, schedule in
  `pipeline.py`). Warmup depth derives from pipeline depth (AsyncPP
  `runtime/runtime.py:168-177`).
- **[PP-4]** **Weight-stash ring (W9):** `num_versions = num_warmup_minibatches + 1`; backward
  runs on the stashed version its forward used (`load_old_params` → queue head), the optimizer
  steps on latest (`load_new_params`), rotate after step. **The mechanism's published origin
  is PipeDream**: weight stashing keeps one weight version per active minibatch so forward and
  backward of a minibatch use the *same version within a stage* ([PDREAM]:350), yields a
  per-stage delay gradient that is valid (stage 1 delayed n steps, stage 2 by n−1, …
  [PDREAM]:364), and is "critical for meaningful learning" ([PDREAM]:376); origin
  implementation `pipedream/runtime/optimizer.py:19` (`OptimizerWithWeightStashing`; deque
  `:59`, `load_old_params`/`load_new_params` `:110-114`). Port source: AsyncPP
  `optim/optimizer.py:21` (class), `:39-43` (`num_versions`, `stash_to_cpu`), `:103-104`
  (deque), `:154-158` (load old/new at `154`/`158`); CPU/disk stash options `:77`; ring sizing
  `main_with_runtime.py:308-314`. (Path note: `optim/`, not `runtime/` — §1.3.) Vendored
  second witness: `AsyncMesh/asyncpp/optim/optimizer.py`. **Scope: fixed-route pipes only.**
  Stashing is O(P·N) memory against O(N) for the no-stash alternative ([NAG-PP] Table 1,
  `:192-196`), and under stochastic routing there is no per-stage version chain to stash
  against — "weight stashing is not applicable in SWARM" ([NAG-PP]:306). A profile with
  [PP-6] routing takes the [PP-5] no-stash form instead.
- **[PP-5]** **Delay-corrected optimizer (W9).** The primary mechanism is the published one
  ([NAG-PP]): a modified Nesterov look-ahead in **weight space** — the update discounts the
  gradient term by (1 − γ_t) so the look-ahead step approximates and cancels the delay
  ([NAG-PP]:120-122) — realized in practice as **NAdam used as-is with a large fixed
  β₁ = 0.99** ([NAG-PP]:154), with sublinear convergence proven under fixed delay. Two forms:
  - **With stashing** (`Ours`): best perplexity, O(P·N) memory ([NAG-PP]:195).
  - **No-weight-stash** (`Ours-No-WS`): O(N) memory, beats every other asynchronous method and
    approaches GPipe (WT 29.90 vs 30.63; OWT degrades to 108.20 vs 65.17) ([NAG-PP]:196) —
    the only form available under [PP-6] routing, and the memory-honest default for this
    programme.
  Variants, in decreasing evidence order: the stage-dependent schedule (LR decreasing toward
  earlier stages, momentum linear 0.9 → 0.99 from last stage to first, [NAG-PP]:168); the
  AsyncPP `adaptive_momentum` stage-count formula — an **implementation heuristic** the
  paper's own ablation finds *slightly worse* than fixed 0.99 for the stashing method and
  helpful only for the no-stash variant ([NAG-PP]:274; §1.3 correction 2); the `lr_correction`
  discount (`main_with_runtime.py:116-118`). Integration pattern incl. SPARTA-mid-loop
  stash-head replacement: `AsyncMesh/examples/pp_diloco_async.py:103-112, 123-163`.
- **[PP-6]** **Stochastic routing (W9):** the *contract* is [XP-4] (reroute to replica,
  bounded-staleness rejection, control plane never blocks); the *selection heuristic* is
  published in the Agora paper (§1.3 correction 1): per-stage worker discovery by DHT re-read
  about every 30 s with expiry-based drop-out ([AGORA-P]:332); a min-heap keyed by
  **accumulated virtual runtime**, estimated task cost charged up-front and corrected by
  measured throughput so slower replicas are charged more; a new worker enters level with the
  most-loaded replica; failed workers are temporarily banned; and the **backward pass is not
  load-balanced — it follows the forward path** ([AGORA-P]:334), which is exactly the
  determinism [XP-3] wants. [SWARM-P] is the lineage for stochastic rerouting itself.
  **Code source (rev 5): the SWARM prototype's `ExpertBalancer`**
  (`swarm/swarm/pipeline/src/moe/client/balancer.py`) implements this heuristic family:
  min-heap keyed by accumulated expected runtime (`:30, :111, :126-127`), 30 s DHT refresh
  (`update_period` `:22`), up-front cost charge corrected by a throughput `PerformanceEMA`
  (`:122-131`), expiry-based temporary bans (`:90-98, :140-143`), and deadline-exceeded →
  re-route to the next expert (`:136-138`). One **declared divergence between code and
  paper**: the balancer seeds a *new* worker at the least-loaded queue head "to ensure it is
  evaluated" (`:82-84`), where [AGORA-P]:332 enters it at the most-loaded replica's level —
  the entry level is a plan parameter, not a fact to inherit silently. Agora's own trainer
  client remains unpublished; routing inputs are committed-snapshot classes plus local
  observation, never consensus state.
- **[PP-7]** PP×DP layout reference (informative): `pp_stage = rank % num_pp_stages`,
  `pp_id = rank // num_pp_stages` (AsyncMesh `sparta/setup.py:31-35, 53, 91-105`) — a formula,
  not a contract; the contract is [GV-2] capability shapes + genesis placement.

## 12. Trajectory monitor [MO]

**Promotion criterion:** the W8 gate — [DR-3] may not ship before [MO-1]/[MO-2] are live
(§8's hard dependency), so the monitor promotes with W8's first half.

*Mechanism: [MO-2] (P2 paper — nothing in any checkout computes either statistic). Host
service: [MO-1]. Invariant: [MO-3].*

- **[MO-1]** **Host service + `metric@1` slot.** The monitor is a host service (arch §12); the
  guest emits per-peer statistics through a `metric@1` slot bound to `metric`-class channels
  ([CX-3]). Today's `sys@2::emit_metric` (advisory, rate-limited `(name, f64)`) is not the
  monitor; it remains the ambient telemetry lane. Placement — extend `daemon-vhc-observe` vs a
  new host crate — is [OQ-1]; the recommendation is `daemon-vhc-observe` (it already owns the
  replay-oracle statistics machinery, and the monitor's committed statistics must survive
  replay).
- **[MO-2]** **Day-one statistics** (arch [AL-3] pre-specification — carried from the start so
  later screens consume the same baselines): parameter drift EMAs, logit-JS disagreement,
  staleness distributions, [XP-5] omission counts, and the [DM-3] disagreement statistics for
  every driver in that class. Definitions anchor on the literature:
  - **Scalar consensus-distance estimator** ([CCD]:316-318, 444): each peer locally computes
    `θ_i = ‖aggregate − x_i‖²` after an exchange round — the literature's form is
    `‖Σ_j w_ij·x_j − x_i‖²` over a gossip mixing matrix `W`, but VHC has no mixing matrix:
    the committed-set fold IS the averaging operator, so the sealed aggregate plays `W`'s
    row. `Θ² = avg(θ_i)` needs only a scalar all-reduce and upper-bounds the consensus
    distance (Ξ ≤ (2/p_mix)·Θ; see [DM-3] for `p_mix` vs `p_sparse` and for when the bound
    applies at all). This is the statistic that lets the monitor gate [DR-3] without shipping
    parameter vectors. Under [DR-3]'s rotation policy the estimator is sampled per rotation
    **cycle**, not per round ([DM-3]).
  - **Logit-JS disagreement** ([FG]:206-210): per-token Jensen–Shannon distance between each
    peer's output distribution and that of the averaged parameters, on a **held-out batch
    sampled once and reused at every measurement** — the fixed batch is what makes rounds
    comparable. [FG] reports it tracks instability better than parameter distance.
  Nothing in any checkout computes either (§0.3); the ambient metric *taxonomy* has a
  reference in agora's log-scrape monitor (`prometheus/monitor.py:728-777` — AR duration,
  yield stalls, barrier waits, queue depth, chunk-gap stats).
- **[MO-3]** Monitor statistics are **committed statistics**: quantized per [PD-2] discipline
  before entering any consensus-adjacent decision (a [DM-3] acceptance gate is a decision).
  Raw floats stay in the observability lane.

---

# Part C — informative reference profiles

## 13. Reference profiles [EX]

Nothing in this section is normative. Each profile names its prerequisites (Parts A/B), its
acceptance gates, and the honest state of the underlying research. Profiles are the *evidence
plan* for promoting Part B.

### EX-1 — Decentralized SparseLoco/DiLoCo data parallelism (wave W6)

- **Composition:** [DR-1] overlap + [DR-2] cohort scope + [DR-4] pseudo-gradient outer step
  (the mechanism that makes the "DiLoCo" in the name true) + [PC] collectives + [CX-4/5]
  codecs + [MB] elasticity, on heterogeneous C1–C4 peers.
- **Data assignment references:** psyche `shared/coordinator/src/data_selection.rs:7`
  (`assign_data_for_state`), `:36` (seeded shuffle), `:40-47` (largest-remainder split), `:110`
  (`get_data_index_for_step`); prime-diloco `data.py:509-516` (file-level rank split).
- **Reference hyperparameters** ([SCALE-DL], the corpus's only systematic sweep): outer LR
  swept over {0.2, 0.4, 0.6, 0.8, 1.0} with optima interior ([SCALE-DL]:165); fitted scaling
  laws give optimal batch size growing with both model size and replica count; H = 30 for the
  law-fitting runs ([SCALE-DL]:44); predictions validated at 4B/10B ([SCALE-DL]:48). Baseline
  sanity anchor: DiLoCo at M = 1 (an outer-Nesterov Lookahead variant) *outperforms*
  data-parallel at every scale tested ([SCALE-DL]:52) — so the single-replica configuration
  of this profile is itself expected to beat plain DP, and failing that is a red flag before
  any decentralization question arises.
- **Gates:** survives seeded churn (join/leave mid-round) without digest divergence; per-group
  digests bit-exact across peers every round ([PC-8] standing); the [DR-4] outer step is
  deterministic given the sealed fold — same committed pseudo-gradient in, same shadow out,
  digest-checked with the round; measured per-peer bytes vs the
  all-download-all baseline at the same geometry ([PC] complexity statement, §7); C1 and C4
  peers both contribute (sample-weighted, [MB-4]).

### EX-2 — Decentralized DiffusionBlocks (wave W6)

A **paper-backed, prototype-backed reproduction**, not a speculative spike: [DBLOCK] evaluates
five architecture families including autoregressive LM (§1.3 correction 5), and [BTRAIN] has
already run the mechanics decentralized on real text.

- **The property that makes it cheap:** a block's training forward consumes the raw input, the
  noised target embedding, and σ — never the previous block's hidden activations. The paper
  states it directly: each block trains "in an embarrassingly parallel manner … with
  absolutely no communication overhead" ([DBLOCK]:567). Code pins: block σ-bands by equal CDF
  mass under the log-normal prior (`DiffusionBlocks/dblock_modules.py:6-20`), per-block σ
  sampling (`model.py:157-177`), σ→block inversion (`:182-188`), EDM preconditioning
  (`:203-205`), layer-subset partial forward (`model.py:208-215`; `vit.py:351` layer skip,
  `:690-692` `forward_block`), σ-weighted loss (`model.py:222-231`, weights `:179`), inference
  sampler passing only the latent (`model.py:269`).
- **Scope of the zero-traffic claim:** it covers block-local *training* only. The shared
  embed/head cohort still synchronizes (its own group below); distributed *inference* hands
  latent state between block owners ([BTRAIN]'s serving path is exactly this) and does need a
  transport; and decentralized *validation* of an assembled model needs block collection. "No
  §10 transport" holds for the training loop, not the lifecycle.
- **The memory lever — stronger than zero traffic, and the reason this profile leads:**
  DiffusionBlocks reduces **all** memory components — parameters, gradients, optimizer state,
  activations — by the block factor B, where checkpointing reduces activations only:
  (4P + A)·(L/B) versus 4PL + A, and composing both "uses the least memory among these four
  patterns" ([DBLOCK]:561-563). Given the ceremony's binding constraint is device-side working
  set (§1.2), this composes with [ML-5] recompute and is the strongest memory mechanism in
  this document. Optimal B is task-dependent (B = 4 for AR text; B = 3 for text8 masked
  diffusion) ([DBLOCK]:194, and §5.4).
- **The AR construction** ([DBLOCK] §5.4, App. E.4): noise enters after the embedding
  (`z = f_in(x)`, `z_σ = z + σε`); the block denoiser recovers token *i*'s clean embedding
  conditioned on **clean past embeddings**, cross-entropy replacing L2. Causal consistency has
  two published implementations, and the choice is declared against the profile's memory
  tiers: sequence concatenation under a modified causal mask — single forward pass, **doubles
  sequence memory** (the paper's choice) — or separate clean/noisy KV computation — standard
  memory, two forward passes ([DBLOCK]:481). Interacts with [ML-1]/[ML-5].
- **A second band-assignment rule:** masked-diffusion text partitions the *masking schedule*
  α(t) by equal decrement — equal shares of demasking work — rather than σ-bands by CDF mass
  ([DBLOCK]:447, results at `:194`). The [GV-4] binding metadata carries the band rule as a
  declared enum, not a hard-coded prior.
- **Decentralized prototype evidence** ([BTRAIN], a litepaper — self-reported, not independent
  replication, so it narrows the gate without closing it): Sakana-style DiffusionBlocks
  mechanics on byte-level WikiText reach CE 1.359 vs ≈1.32 for the same-setup end-to-end
  reference; a six-worker shared run with same-block replica averaging reaches CE 1.385; an
  HTTP/TCP transport proof and public-IP runs complete with deadline-based update acceptance
  ([BTRAIN]:11, 35). Block-local ownership, replica averaging, and deadline acceptance are
  exactly this profile's composition.
- **Composition:** disjoint block ownership via [GV-4] bindings (each peer's group owns one
  block's parameters and optimizer state); [DR-2] cohort DP within a block; a small shared
  embed/head cohort (its own group + `param_delta` channel); block-local checkpoint families.
- **Named prerequisite gate — reproduction, not existence:** before the decentralized
  experiment is scheduled, a single-process run reproduces the paper's AR construction at
  reference scale against its published result ([DBLOCK] §5.4/Table 4 as the check), plus a
  [BTRAIN]-class decentralized smoke on the VHC substrate. A negative result now means the
  *reproduction* failed (a port bug or an unstated dependency) — it no longer re-scopes EX-2
  to vision, because the formulation question is answered in the literature.
- **Gates:** trained quality parity with the single-process oracle at equal steps; zero
  measured inter-block training traffic (the property, verified live); churn on a block cohort
  recovers from block-local checkpoints alone; the causal-consistency memory choice declared
  and measured against device tiers.

### EX-2r — Recurrent-depth (looping-transformer) variant

Structurally NOT EX-2: Huginn-class recurrent-depth models need **no block partitioning** —
the entire recurrent module trains as a single-pass denoiser with σ sampled per step, where
baseline Huginn runs ~32 recurrent iterations with 8-step truncated BPTT; the single pass
costs roughly 10× less total training compute ([DBLOCK]:485). That is full replication plus
σ-sampling: an **EX-1-shaped
data-parallel profile**, requiring no [GV-4] disjoint ownership, no per-group cursors, and no
shared embed/head cohort — it runs on the certified-baseline vocabulary of Part A as it stands
today. It is therefore the cheapest LM-objective validation of the diffusion interpretation
available, and a candidate to run *before* W6 commits to EX-2's group machinery.

### EX-3 — Static two-stage pipeline (wave W7; promotes [XP], [PP-1..3])

- **Composition:** [XP] transport + [ML-5] recompute + [PP-3] static schedule; f32 throughout
  ([ML-6] — no dependency on ML-1/2/3/4); [CX-7] optional (a first run MAY ship uncompressed
  boundaries at toy width).
- **Gates:** loss-curve parity with the single-process oracle (same seeds, same data order —
  exact for the f32 det-side states, tolerance-classed for local trajectories per [DM-6]);
  [XP-1] conformance demonstrated by injecting transfer delay and observing omission, never
  record stall.

### EX-4 — Async 1F1B (wave W9; promotes [PP-4..5])

- **Composition:** EX-3 + [PP-5] delay-corrected optimizer (both forms — [PP-4] stashing on
  fixed routes, no-stash where memory tiers demand it) + [XP-5] dedup/late handling.
- **Staleness escalation path:** if the [PP-5] look-ahead correction is insufficient at depth
  (staleness grows with stage count; [BROT] measures a 5.81× convergence slowdown at 32 stages
  for uncorrected Adam, [BROT]:19), the literature's strongest answer is **basis rotation**:
  aligning the Hessian eigenbasis with the coordinate basis so Adam's coordinate-wise
  adaptivity survives delay — same-loss with 81.6% fewer iterations in [BROT]'s headline
  result ([BROT]:47), evaluated with the [NAG-PP] look-ahead among its baselines, and it works
  without weight stashing. A complement to [PP-5], not a replacement; recorded here as the
  named option this profile escalates to.
- **Gates:** convergence within the staleness budget declared in the channel bounds;
  degradation vs EX-3 quantified (bubble fraction, staleness distribution from §12); stash
  memory within the declared device tiers (or the no-stash form declared instead).

### EX-5 — Swarm-routed subspace pipeline (wave W9; promotes [PP-6])

- **This is a novel VHC composition, and says so.** It is not "Agora's production
  configuration" (Agora runs backward-pinned-to-forward-worker, stage recomputation,
  independently drifting stage replicas, and AsyncSPARTA reconciliation — no [PP-5]-class
  optimizer correction), and it is not SWARM as published (which has none of the subspace
  machinery). EX-5 composes published parts — Agora's routing heuristic and frozen-basis
  codec, [NAG-PP]'s no-stash delay correction, SWARM's rerouting contract — into a
  configuration no paper reports. Its gates are correspondingly its own.
- **v1 codec is the frozen-basis subspace variant** — the frozen reparameterized-SSN design
  genuinely is Agora's production architecture: pinned pretrained basis as a genesis artifact
  and the [CX-7] codec. **Parameters are per-run hyperparameters, not Agora's:** k = 40 at
  d = 4096 is Protocol Models' configuration; Agora ran rank 40 at d = 2048 and ≈51 at
  d = 5120 with an explicit no-scaling-law disclaimer ([AGORA-P]:378; §1.3 correction 10), so
  k ≈ d/100 is a starting heuristic only. Boundary budget at the illustrative point (b = 8,
  n = 2048, k = 40, bf16 wire): 8·2048·40·2 B = **1.25 MiB — one activation tensor, one
  direction, before the token-index side channel; backward activation-gradient traffic is
  additional and comparable.** The **full Protocol-Models scheme** — Grassmann basis refresh
  (~every 500 steps, broadcast; [PM]:139) and row-constant-second-moment AdamW (plus the
  per-iteration `W_p1` projection it does NOT remove, [PM]:143) — is paper-only double debt
  (TDD §7, §0.3) and an explicit **promotion gate**: the profile may not claim the
  Protocol-Models name until both land with their authored tests. Until then the claim is
  "frozen-basis subspace pipeline". The basis strategy beyond these two rungs is an open axis
  ([OQ-8]).
- **Composition:** [PP-1..3] + [PP-5] in its **no-weight-stash form** + [PP-6] routing +
  [DR-2] within replicated stages + [MB] (ghost join for stage replicas). [PP-4] stashing is
  **excluded, not deferred**: under stochastic routing there is no stashed version chain to
  load — "weight stashing is not applicable in SWARM" ([NAG-PP]:306) — so its inclusion here
  in an earlier draft composed two incompatible mechanisms.
- **Vacancy recovery, two rungs:** [MB-5] restore of the stage's [GV-4] families
  (checkpoint-exact), and **CheckFree** neighbour-reconstruction as the cheap option — a lost
  intermediate stage reinitializes as the gradient-norm-weighted average of its two neighbour
  stages (ω = the stage's last ‖∇W‖², one scalar of overhead; [CHECKFREE]:132-136; official
  code now in the corpus — norm-weighted recovery at
  `CheckFree/simulate_training/convergence_training.py:544-545`), evaluated
  at 124M–1.5B under 5–16%/hour failure. Caveats carried with it: intermediate stages only
  (first/last need CheckFree+'s out-of-order variant), consecutive-stage loss unrecoverable,
  and it is lossy — a convergence cost accepted for wall-clock.
- **Gates:** a model larger than any single participating device trains across seeded churn
  including transient stage vacancy — recovered by whichever rung applies ([MB-5] generally;
  CheckFree only for intermediate, non-consecutive losses, per the caveats above); boundary
  bytes match the [CX-7] budget **accounted per direction** — forward activations, backward
  activation-gradients, and the token-index side channel measured separately (the illustrative
  1.25 MiB above is one tensor, one direction); routing never blocks the control plane under
  injected worker failure ([XP-4]).

### EX-6 — AsyncMesh composition (wave W10)

- **Composition:** [DR-3] within replicated stages × [PP] across stages — the DP×PP mesh.
  EMA delay correction on the sparse axis (AsyncMesh `sparta.py:119-137` — `ema` rule;
  delayed-buffer pop after `async_sparta_delay` `:88-110`); the published λ 0.5 → 0.01 cosine
  schedule ([AMESH]:155) is absent from the checkout, which cosines momentum instead — TDD §7
  flags the delta (§1.3 correction 9).
- **Declared parameters, with their evidence status:** staleness τ is **plan-declared and
  profile-measured**, not a portable bound — [AMESH]'s default is τ = 10, its ablation
  demonstrates tolerance up to 50 steps at 5% subsets ([AMESH]:231), and an effective delay of
  100 did not converge ([AMESH]:195); 10 is the starting value, 50 the demonstrated ceiling.
  Sparsity p_sparse is declared per [DR-3] (AsyncMesh's regime is 5%; SPARTA's is 0.05%).
  **This profile also carries the [DM-3] evaluability declarations** (§2): phase boundaries
  in group rounds, the phase-local disagreement bound, the measurement cadence (per rotation
  cycle under the rotation policy), and the action on breach — all four are run-configuration
  values this profile must state, because [DM-3]'s recursion does not derive them for the
  shipped policies.
- **Gates:** the limit-case identity — delay→0, averaging every K steps with full parameter
  exchange recovers eager-DiLoCo behavior, per [AMESH]'s own strict-generalization claim
  ([AMESH]:119); [DM-3] disagreement gate green on the sparse axis at mesh scale (against the
  declared phase-local targets above); composition adds no new digest class (DP-axis stays
  [PC-8]-exact per stage).

---

## 14. Refactor plan [W]

### 14.1 Wave dependency structure

```
W0 (spec amendments) ──► W1 (GV vocabulary) ──► W2 (RT runtime) ──► W4 (CX + PC)
                              │                                        │
W3 (ML levers — starts        │                                        ▼
    immediately, no W0/W1     └──────────────────────────► W5 (DR-1/2 + MB)
    dependency)                                                │
       │                                                       ├─► W6 (EX-1 + EX-2)
       │ ML-5 only                                             │
       └────────────────────────────────────────────┐          │
                                                    ▼          ▼
                                          W7 (XP + EX-3) ◄─────┘
                                                    │
                                W8 (MO + DR-3) ◄────┤ (W8 needs W5; runs beside W7)
                                                    ▼
                                          W9 (PP-4..6 + EX-4/EX-5)
                                                    │
                                          W10 (EX-6) ◄── W8
```

- **W3 has no W0/W1 dependency** — compute-lane work touches no manifest vocabulary and is the
  longest-lead host item; it starts first. Only [ML-5] feeds W7 ([ML-6]).
- **W6's critical path runs through [GV-4] and [GV-5]** — per-group layout bindings and cursors
  are W1-class *contract* work, not driver work; scheduling them late blocks both W6 profiles.
- **[GV-7]'s two compatibility gates apply from W1 onward, permanently** (degenerate
  reproduction; frozen-C2-artifact admission).
- **EX-2r is runnable before its wave.** It needs no [GV-4] bindings, no per-group cursors,
  and no shared embed/head cohort — it runs on Part A as the tree stands (§13 EX-2r). It is
  the cheapest LM-objective validation of the diffusion interpretation, and W6's EX-2
  reproduction gate can consume its result; scheduling it early is a free de-risking step.

### 14.2 Waves

| Wave | Delivers | Crates touched | Suites (TDD cross-ref) | Gate |
|---|---|---|---|---|
| **W0** | This document + the arch-spec amendment (Appendix B applied) | docs only | — | both documents land together; no dangling dispositions |
| **W1** | [GV-1..7]: vocabulary, per-group bindings, per-group cursors, LC-2 refusal | `daemon-vhc-proto`, `daemon-vhc-abi`, `daemon-vhc-sdk-consensus` | net-new (gap-register class 1/3 shape: contract + admission) | [GV-4g] byte-for-byte; [GV-7a/b] |
| **W2** | [RT-1..3] runtime; walkers rebuilt | `daemon-vhc-sdk`, `guests/tiny-llama` | existing parity suites | [RT-4] parity bit-identical |
| **W3** | [ML-1..5] | `daemon-vhc-host` (compute), `daemon-vhc-sdk-compute` | compute conformance extensions (HOST-class) | flash-attn conformance flips; [ML-5] used by a test guest |
| **W4** | [CX-1..10] + [PC-1..9], co-designed ([CX-10] lands as a recorded constraint that binds [XP-2] at W7; [PC-9] as a disposition — neither is code) | new `daemon-vhc-sdk-codec`, new `daemon-vhc-sdk-collective` | SDK-1..5 analogues + net-new collective suite | [PC-8] bit-identity vs det fold; codec golden fixtures (TDD §5 constants) |
| **W5** | [DR-1..2] + [DR-4], [MB-1..5] + the [MB-7] declaration rule | `daemon-vhc-sdk-rounds`, new `daemon-vhc-sdk-membership`, session/coordinator mechanism | net-new driver + membership suites | overlap driver commits correctly under injected delay; cohort digests scope-clean; [DR-4] outer step deterministic across peers |
| **W6** | EX-1 + EX-2 (the EX-2 reproduction gate — AR construction + [BTRAIN]-class smoke — before it; EX-2r runs earlier on Part A alone) | `daemon-vhc-sdk-profiles` (rebuilt on codec/collective), experiment guests | EX-1/EX-2 gates (§13) | both profiles' gates green |
| **W7** | [XP-1..5] + [PP-1..3] + EX-3; **promotes §10, [PP-1..3]** | `daemon-vhc-net`, new `daemon-vhc-sdk-mesh`, `guests/pipeline-stage` successor | net-new transport suite (frame typing, dedup, omission) | EX-3 gates; [XP-1] delay-injection test |
| **W8** | [MO-1..3] + [DR-3]; **promotes §12** | `daemon-vhc-observe` (or [OQ-1] outcome), `daemon-vhc-sdk-rounds` | net-new monitor + sparse-driver suites (TDD §7 Phase-3 items) | [DM-3] disagreement gate (phase-local form, §2) live before DR-3 ships |
| **W9** | [PP-4..6] + EX-4 + EX-5; **promotes §11 fully** | `daemon-vhc-sdk-mesh` | TDD §7 Phase-2 items (stash ring, NAdam parity, pipe composition on churn) | EX-4/EX-5 gates; frozen-basis labeling per EX-5; EX-5 built no-stash ([PP-4] excluded under routing) |
| **W10** | EX-6 | composition only | TDD §7 Phase-3 items (rank mapping, EMA correction, limit case) | EX-6 gates |

### 14.3 Standing constraints (every wave)

- **Host-bucket non-claims stay host work** (PS non-claims 1/2/4: payload-retention GC,
  durable-watermark backpressure, archive backfill/frame repair). No wave silently absorbs
  them; [PC-3]'s reliance on payload fetchability *names* the retention dependency, it does not
  discharge it.
- **The verification layer (SENTINEL-ZK) is out of scope** — design input to arch §14 only.
- **Resource discipline**: capped builds, diff-scoped `just lint`, one build at a time — the
  superproject rules apply to every wave without exception.
- **`daemon-vhc-det` boundary inviolable** ([CX-2]).
- **Existing parity suites are permanent regression gates**, never bring-up scaffolding to be
  retired.
- **No version bumps** as a side effect of any wave; wire/packaging-relevant changes are flagged
  for the human to decide.

## 15. Open questions [OQ]

Registered, not settled — recorded so no wave pretends otherwise:

- **[OQ-1]** Monitor placement: extend `daemon-vhc-observe` vs a new host crate. Recommendation
  in [MO-1] (extend observe); decide at W8 entry.
- **[OQ-2]** Per-layer in-stage activation checkpointing (needs tape hooks; [ML-5] deliberately
  avoids them). Revisit if EX-4 stash memory misses its tier.
- **[OQ-3]** Precision heterogeneity within a cohort: may bf16 and f32 peers share a group?
  Their Δ contributions differ in magnitude; the cross-backend tolerance class does not answer
  whether precision is a tolerance or a group-partitioning constraint ([DM-9]).
- **[OQ-4]** Runtime codec registry (rejected for now, [CX-1]) — revisit only if module-static
  binding demonstrably blocks an experiment.
- **[OQ-5]** KV/context-parallel codec (MoS lane, v1 §5.5) — vocabulary reserved (`kv` channel
  class), nothing else designed here. The published mechanism is now concrete enough to name:
  [MOS-P] caches a fixed orthonormal frame Ū per node and transmits only rotation parameters
  θ (`U(θ) = R(θ)·Ū`, [MOS-P]:83-99), with per-chunk rotations for dynamic subspace mixtures
  ([MOS-P]:107) — i.e. the `kv` codec's state is a genesis frame plus a small learned
  rotation, structurally akin to [CX-7]'s basis-as-genesis-artifact.
- **[OQ-6]** Gossip beyond the sparse driver (pairwise mix operators — arch [PIR-11] rung 3
  anticipated them; [DM-2] holds the door; [PC-9]'s NoLoCo/Moshpit dispositions live at the
  same door).
- **[OQ-7]** bf16 lane negotiation shape (how [ML-2] is declared and admitted: manifest
  capability vs compute-world minor).
- **[OQ-8]** **Boundary-compression basis strategy is an open design axis, not a settled
  progression.** The corpus offers at least five published positions: frozen reparameterized
  SSN (Agora production); global subspace + Grassmann refresh + modified AdamW ([PM]); learned
  per-boundary Stiefel-manifold projectors updated every step, with factorized anchor
  embeddings and streaming-codebook VQ ([MAPL] — which reports per-boundary subspaces are
  geometrically distinct and beats SSNs by ~5%, [MAPL]:41); learned residual bottlenecks
  ([RESBM]); and change-compression with convergence guarantees ([AQSGD], see [CX-10]).
  EX-5's frozen-basis → Protocol-Models naming gate orders the first two rungs only; it does
  not commit the programme to that path over the others.
- **[OQ-9]** **Per-sample residual state for lossy activation codecs** ([CX-10]): AQ-SGD's
  guarantee needs a buffer keyed by sample identity, which interacts with data assignment
  ([GV-5] cursors), checkpoint families, and memory tiers. Whether the `residual` class
  gains a per-sample keying mode or a new class is undecided.

---

## Appendix A — implementation evidence index *(informative)*

The audited porting corpus. **Presence of a source here does NOT mean the mechanism is
supported** — §1.4 is the support list; this appendix is the evidence dossier behind it. It is
keyed both ways: A.0 answers "what is each source for" (per-source ledger), A.1 answers "where
is each mechanism from" (the primary index), and A.2–A.10 carry the line pins, keyed by source
per spec section (the secondary index).

Vocabulary for the source-keyed tables: **portable** = algorithm/transport semantics reusable
without the source framework; **policy** = wiring tied to PyTorch process groups, hivemind
DHT/P2P, agora expert UIDs, Lightning, or Prometheus (the peel-off list); **evidence-only** =
a witness for a pattern, itself not a supported mechanism. All paths relative to
`~/experiments/decentralised-llm-training/`; commits per §0.2. Agora paths carry the `core/`
segment; AsyncPP optimizer paths use `optim/` (§1.3 corrections).

### A.0 Per-source ledger

What each checkout actually contributes, and what is explicitly rejected or not ported from it
— in one view per source.

| Source | Contributes | Explicitly rejected / not ported |
|---|---|---|
| hivemind | scalar quantiser family + adaptive-selection interface ([CX-5]); blockwise quantization ([ML-4]); part geometry ([PC-4]); delta return ([PC-5]); minimax-LP cost model ([PC-6]); sample-weighted progress ([MB-4]); optional state streaming ([MB-5]); remote-autograd bridge ([XP-3]); PowerSGD EF witness ([CX-8], evidence-only) | mid-flight sender/reducer exclusion (**rejected**, [PC-2]); DHT encodings ([MB-2/4]); `hagenbach_bishoff` (upstream-vestigial, §1.3); MoE machinery (no clause consumes it) |
| psyche | DCT/top-k/1-bit codec ([CX-6]); error-feedback loop semantics ([CX-8]); wire framing; deterministic data assignment (EX-1) | — (note the `comptue_hash` trap, §0.4) |
| agora | bf16 posture ([ML-2]); recompute witness ([ML-5]); frozen-subspace codec ([CX-7]); ghost join ([MB-2]); progress variant ([MB-4]); AsyncSPARTA selector + β^k ([DR-3]); PowerSGD EF witness ([CX-8], evidence-only); random-projection codec (evidence-only); ambient metric taxonomy (§12) | worker routing (its trainer client is unpublished; [PP-6]'s code source is swarm's `balancer.py`, its published description [AGORA-P]:332-334); expert-UID and DHT wiring |
| torchft | prepare/perform split + commit ceremony ([DR-1]); fragment stagger; quorum semantics ([MB-6], coordinator reference only) | `bucket_cap_mb` naming (1024² trap, §0.4); quorum as guest SDK policy |
| AsyncPP / AsyncMesh | recompute mandate ([PP-2]); warmup sizing ([PP-3]); weight-stash ring ([PP-4]); `adaptive_momentum` heuristic ([PP-5] demoted variant); SPARTA selector + delayed buffer ([DR-3]); PP×DP formula ([PP-7], informative) | `dist.broadcast`/process-group wiring; the adaptive-momentum flag as *the* delay correction ([NAG-PP]:274 ablates it worse) |
| prime-diloco | pseudo-gradient shadow ([DR-4]); layer buckets + uint8 ring ([PC-7]); uint8+LUT codec ([CX-5]); p2p checkpoint ([MB-5], optional); file-level data split (EX-1) | `ProcessGroup` lifecycle; the README "8×" claim class (§1.3 correction 3 pattern) |
| node0 | second subspace witness ([CX-7]); stale-bump window ([MB-3]) | — |
| OpenDiloco | overlap scheduling precedent; pseudo-gradient second witness ([DR-4]) | — |
| DiffusionBlocks | σ-band partition, per-block sampling, partial forward, σ-weighted loss (EX-2) | DDP `find_unused_parameters` artifact; the checkout's ViT-only scope is not the paper's ([DBLOCK] covers five families, §1.3 correction 5) |
| swarm | `ExpertBalancer` routing — min-heap virtual runtime, DHT refresh, throughput EMA, bans ([PP-6]) | its vendored hivemind-derived DHT/MoE stack (the VHC control plane replaces it); fairseq bottleneck experiments |
| pipedream | weight-stashing origin — `OptimizerWithWeightStashing` ([PP-4]) | its profiler/planner and parameter-server paths (VHC schedules via [PP-3]/[PP-7]); ALL datacenter performance evidence (§0.2 regime note) |
| torchgpipe | clean-room schedule + checkpoint witnesses ([PP-2/3], [ML-5]) | CUDA-stream copy machinery (host lane owns transfer); ALL datacenter performance evidence (§0.2 regime note) |
| powersgd | upstream-official EF loop witness ([CX-8]) | PowerSGD as a supported codec (evidence-only — §1.4 unsupported list) |
| AC-SGD | `DeltaCompressor` per-sample activation cache ([CX-10], [OQ-9] shape) | the codec itself (deferred; constraint recorded, not a supported mechanism) |
| CheckFree | neighbour-average stage recovery (EX-5 vacancy option) | its fixed simulation harness (`trainer.py` topology wiring) |
| DeMo | official standalone optimizer + OLMo patch — fixture witness ([CX-6], [CX-8]) | training-harness wiring (psyche remains the port source) |
| moshpit-sgd | official [MOSHPIT] experiments (disposition evidence, [PC-9]) | nothing ported; the averaging machinery lives upstream in hivemind |
| asyncdiloco | official [ALSGD] release — toy notebook documenting Delayed Nesterov ([DR-3] option) | not a training stack; any DN port is paper-grade |
| rl-swarm | nothing | whole checkout (different product, §0.2) |
| `p2p/` | nothing | whole directory (transport stacks, no collectives, §0.2) |

### A.1 Mechanism → source (primary index)

Every mechanism this document puts in the SDK (or explicitly keeps out), its owning clause,
its code source (A.2–A.10 carry the line pins), its paper source (§0.2 register), and the port
relationship. Relationship vocabulary (maps onto §1.4's classes: port/port-variant → P1,
paper-port → P2, authored → P3):

- **port** — semantics ported from the named code, paper agrees or is silent;
- **port-variant** — ported from code with a declared divergence from the published method;
- **paper-port** — implemented from the paper; no code source exists in any checkout;
- **in-tree (P0)** — no *external* source because the source is the certified tree itself;
  §1.4 classes these P0 (generalize/extract), never P3 — an "authored" label here would
  wrongly suggest no oracle exists;
- **authored** — VHC-specific; no code or paper source (§0.3 register);
- **parameter** — not a mechanism but a declared profile parameter this document forces into
  the open;
- **rejected / deferred** — documented anti-pattern, or dispositioned alternative.

| Mechanism | Clause | Code source | Paper source | Relationship |
|---|---|---|---|---|
| Group/channel/state manifest vocabulary | [GV-1..7] | — | — | authored |
| Per-group layout bindings | [GV-4] | `LayoutBinding::of_numels` (whole-model today) | — | in-tree (P0 generalize) |
| Per-group data cursors | [GV-5] | `advance_cursor` (global today); psyche `data_selection.rs` (assignment math only) | — | in-tree (P0 generalize; assignment math: port) |
| Typed async runtime (completion over op-ID ABI) | [RT-1..3] | the five walkers' hand-rolled bookkeeping | — | in-tree (P0 extract) |
| Rematerialization (stage recompute) | [ML-5] | agora `module_collab.py`; AsyncPP `runtime.py`; torchgpipe `checkpoint.py` | [GPIPE]:77 (origin) | port |
| bf16 compute lane | [ML-2] | — (posture witness: agora, §1.4) | — | authored (host lever) |
| Codec trait + role/size-adaptive selection | [CX-1/2] | hivemind `compression/*` | — | port (interface; the static-composition invariant [CX-1] itself is authored, P3) |
| fp16 / 8-bit scalar quantiser family; bf16 wire scalar | [CX-5] | hivemind, prime-diloco | — | port |
| E3M0 4-bit outer-gradient wire format | [CX-5] | — | [SDILOCO]:156 | paper-port |
| DCT + top-k + sign transform codec | [CX-6] | psyche `distro.rs`; witness DeMo `demo.py` | [DEMO-P] | port ([DEMO-P] is the anchor; [DISTRO-R] has no compression) |
| Frozen-subspace activation codec | [CX-7] | agora `experts.py`; node0 `layers.py` | [PM], [AGORA-P] | port |
| `T_fixed` genesis artifact + PE/TE subtraction | [CX-7] | agora (implicit in codec) | [PM]:105 | paper-port coupling |
| Row-constant AdamW + per-iteration `W_p1` projection | EX-5 gate | — | [PM]:143 | paper-port (double debt) |
| Grassmann basis refresh | EX-5 gate | — | [PM]:139 | paper-port (double debt) |
| Error-feedback `residual` state class | [CX-8/9], [DM-5] | psyche `distro.rs`; agora/hivemind/upstream powersgd | [DEMO-P]:226 (partial subtraction α); theory [EF-SGD]:53, [EF21]:125 | port + authored contract |
| Per-sample change-compression for activations | [CX-10], [OQ-9] | AC-SGD `delta_modules.py` (shape reference) | [AQSGD] | deferred (constraint recorded) |
| Byte-bounded partitioned reduce | [PC-1..4] | hivemind `partition.py`, `allreduce.py` | — | port (windows-as-parts: authored) |
| Delta return | [PC-5] | hivemind `allreduce.py` | — | port (stability rationale only) |
| Minimax-LP reducer load balancing | [PC-6] | hivemind `load_balancing.py` | — | port |
| Mid-flight membership exclusion | — | hivemind `allreduce.py:115...` | — | **rejected** ([PC-2]) |
| Pairwise/group-random averaging (NoLoCo, Moshpit) | [PC-9], [OQ-6] | moshpit-sgd (official experiments; machinery upstream in hivemind) | [NOLOCO], [MOSHPIT] | deferred (rung-3, snapshot-derived groups) |
| Overlap driver (prepare/commit split) | [DR-1] | torchft `local_sgd.py`, `manager.py` | [SDILOCO] | port |
| Strided/sequential fragment rotation | [DR-1], [PC-4] | torchft fragment scheduler | [SDILOCO]:95-97 | port (strided default: paper) |
| Sparse async driver — rotation partition policy | [DR-3] | AsyncMesh/agora selectors | [SPARTA]:84 samples randomly | **port-variant** (rotation ≠ paper's random; declared divergence) |
| Sparse async driver — EMA staleness correction | [DR-3] | AsyncMesh `sparta.py` (ema rule) | [AMESH]:103-155 | port (λ schedule itself: paper-port) |
| Sparse async driver — β^k state scaling | [DR-3] | agora `state_averager_sparta.py` | — | port (implementation-only evidence) |
| Sparse async driver — Delayed-Nesterov correction (option) | [DR-3] | asyncdiloco (toy notebook only) | [ALSGD]:9, 42 | paper-port (named option) |
| Sparsity `p_sparse`, staleness τ, momentum-sync cadence, [DM-3] phase/bound/cadence/breach declarations | [DR-3], [MB-7], EX-6 | — | [SPARTA]/[AMESH]/[DESLOC] disagree | parameter |
| DiLoCo outer step (pseudo-gradient sign) | [DR-4] | prime-diloco, OpenDiloco | [DILOCO-P] | port (sign confirmed) |
| Sample-weighted progress / ghost membership | [MB-1..4] | hivemind, agora, node0 | — | port |
| State handoff / p2p checkpoint restore | [MB-5] | hivemind averager, prime-diloco | — | port (OPTIONAL) |
| Momentum bootstrap on join | [MB-7] | — | [DESLOC]:47, 173 | paper-port (constraint) |
| Deadline-omission execution transport | [XP-1..5] | — | — (no precedent: [SKIPPIPE]:60 is synchronous) | authored |
| Remote-autograd boundary pattern | [XP-2/3] | hivemind `expert.py` | — | port (pattern) |
| 1F1B schedule + warmup sizing | [PP-2/3] | AsyncPP `runtime.py`; witness torchgpipe | [GPIPE]:77, 109 (recompute + fill/drain origin) | port |
| Weight-stash ring | [PP-4] | AsyncPP `optim/optimizer.py`; origin pipedream `runtime/optimizer.py:19` | [PDREAM]:350, 364; [NAG-PP]:306 scopes it (fixed routes only) | port (scoped) |
| Nesterov weight-space delay correction (β₁=0.99 NAdam) | [PP-5] | — | [NAG-PP]:120, 154 | paper-port (primary) |
| No-weight-stash variant | [PP-5], EX-5 | — | [NAG-PP]:196 | paper-port |
| `adaptive_momentum` stage-count formula | [PP-5] | AsyncPP `main_with_runtime.py` | [NAG-PP]:274 ablates it worse | port (demoted heuristic) |
| Min-heap virtual-runtime routing + DHT discovery | [PP-6] | swarm `moe/client/balancer.py` (rev 5) | [AGORA-P]:332-334; [SWARM-P] lineage | port-variant (new-worker entry level diverges from [AGORA-P]) |
| Basis rotation for staleness | EX-4 escalation | — | [BROT] | deferred (named option) |
| CheckFree neighbour-average stage recovery | EX-5 vacancy | CheckFree `convergence_training.py` (rev 5) | [CHECKFREE]:132-136 | port (option) |
| Scalar consensus-distance estimator Θ | [MO-2], [DM-3] | — | [CCD]:316-318, 444 | paper-port |
| Fixed-batch logit-JS disagreement | [MO-2] | — | [FG]:206-210 | paper-port |
| Phase-local bounded-disagreement gate | [DM-3] | — | informed by [CCD]/[MERGE1] | authored (acceptance policy) |
| σ-band block partition (equal CDF mass) | EX-2 | DiffusionBlocks `dblock_modules.py` | [DBLOCK] | port |
| α(t) masking-schedule partition | EX-2 | — | [DBLOCK]:447 | paper-port (band-rule enum) |
| AR clean-past conditioning (concat vs 2-pass KV) | EX-2 | — | [DBLOCK]:481 | paper-port (declared choice) |
| Recurrent-depth single-pass denoiser training | EX-2r | — | [DBLOCK]:485 | paper-port (runs on Part A) |
| Block-local objectives, decentralized (deadline acceptance) | EX-2 | — | [BTRAIN] | prototype evidence (litepaper) |
| KV codec: cached frame + rotation params | [OQ-5] | — | [MOS-P]:83-107 | deferred (vocabulary reserved) |
| Learned per-boundary Stiefel projectors / ResBM | [OQ-8] | — | [MAPL], [RESBM] | deferred (open axis) |

### A.2 Codecs / compression / error feedback (→ §6)

| Source | Mechanism | Class |
|---|---|---|
| `hivemind/hivemind/compression/base.py:22-25, 48, 79-105` | `TensorRole`, `CompressionBase` trait, `NoCompression` | portable |
| `hivemind/hivemind/compression/adaptive.py:25-56` | role/size-adaptive codec selection | portable interface |
| `hivemind/hivemind/compression/floating.py:10-40, 43-74` | fp16 / scaled-fp16 | portable |
| `hivemind/hivemind/compression/quantization.py:60-74, 88-91` | uniform 8-bit, bucket-mean codebook | portable |
| `hivemind/hivemind/compression/quantization.py:77-84` | quantile 8-bit | portable |
| `hivemind/hivemind/compression/quantization.py:130-148, 199` | blockwise 8-bit | portable |
| `prime-diloco/src/zeroband/compression.py:54-70` | uint8 quantize → indices + per-bin lookup table | portable |
| `prime-diloco/src/zeroband/C/csrc/compression.cpp:5-45` | multithreaded uint8 quantize + bucket averaging | portable kernel |
| `psyche/shared/modeling/src/distro.rs:6-12, 120, 250, 285, 330-345, 370, 397, 412` | DCT transform, top-k compress/decompress, index packing (u8/u16/u32 at 256/65536) | portable |
| `psyche/shared/modeling/src/distro.rs:687-692` | 1-bit sign quantization | portable |
| `psyche/shared/modeling/src/distro.rs:442, 462, 505, 555-586, 618, 669` | DisTrO error-feedback loop (generate / peer-subtract / apply / error_correction) | portable |
| `psyche/shared/network/src/serialized_distro.rs:17, 25, 33, 91` | wire framing + integrity hash (`comptue_hash` — §0.4 trap) | portable framing |
| `agora/agora_server/src/agora_server/core/averaging/gradient_averager_powersgd.py:67-90, 226-283` | PowerSGD P/Q phases + EF buffers (`_ms`, `m.sub_(new_m)`), swarm-error-norm clip | **evidence-only** — EF witness for [CX-8]; PowerSGD itself is not a supported codec |
| `hivemind/hivemind/optim/power_sgd_averager.py:146-183` | upstream PowerSGD EF loop | **evidence-only** — second EF witness |
| `powersgd/powersgd/powersgd.py:149, 211, 221` | official PowerSGD: inputs modified in place to hold the approximation error for feedback | **evidence-only** — third, upstream-official EF witness ([PSGD]) |
| `DeMo/demo.py` (+ `0001-DeMo.patch` OLMo reproduction) | official standalone DeMo optimizer: DCT/top-k/sign + EF loop | **evidence-only** — golden-fixture witness for the psyche port ([CX-6]/[CX-8]) |
| `AC-SGD/compress/delta_modules.py:17, 86-125, 126, 140` | `DeltaCompressor`: activation cache read/written by `sample_ids` around compress/decompress | **evidence-only** — shape reference for the deferred [CX-10]/[OQ-9] contract |
| `agora/agora_server/src/agora_server/core/averaging/gradient_averager_random_proj.py:36-48, 93-105` | random-projection grad compression; stage gate | **evidence-only** — no clause ports it ([OQ-8] axis); stage gate = policy |
| `agora/agora_server/src/agora_server/models/reparam_llama/experts.py:74, 103-104, 113-127, 129-141, 145-151` | subspace activation codec (compress/decompress, frozen basis) | portable |
| `node0/src/node0/models/llama/layers.py:465-466, 502-517, 519-531, 550` | same codec, second witness + subspace weight projection | portable |

### A.3 Partitioned collectives / load balancing (→ §7)

| Source | Mechanism | Class |
|---|---|---|
| `hivemind/hivemind/averaging/partition.py:17, 21-90` | 512 KiB byte-bounded parts, per-peer fractions | portable geometry |
| `hivemind/hivemind/averaging/partition.py:114-136` | delta return (`return_deltas`) | portable |
| `hivemind/hivemind/averaging/allreduce.py:26-30, 32, 201-210, 259, 353-356` | butterfly runner, roles, per-peer streams, delta serialize | portable pattern; P2P RPC = policy |
| `hivemind/hivemind/averaging/allreduce.py:115, 198, 319-320`; `partition.py:128, 248, 257` | mid-flight sender/reducer exclusion | **rejected by [PC-2]** — cited as anti-pattern |
| `hivemind/hivemind/averaging/load_balancing.py:13-33, 36-86` | minimax-LP bandwidth cost model | portable |
| `hivemind/hivemind/averaging/load_balancing.py:32, 89-103` | `hagenbach_bishoff` apportionment — upstream-vestigial | implementation detail only (§1.3) |
| `agora/agora_server/src/agora_server/core/all_reduce/ar_runner.py:87-97, 109-124` | butterfly over P2P, delta return, ordered roles; full- vs half-duplex note `:93-96` | portable pattern |
| `prime-diloco/src/zeroband/diloco.py:108-112, 194-199` | layer-bucketed sequential reduce over grouped tensors | portable option |
| `prime-diloco/src/zeroband/collectives.py:31-43, 53, 88-146` | compression-dispatched double-buffered uint8 ring reduce | portable option; `ProcessGroup` = policy |
| `moshpit-sgd/` (official [MOSHPIT] experiments; group machinery upstream in hivemind matchmaking/averager) | randomly re-drawn capped-group all-reduce | **evidence-only** — [PC-9] disposition evidence, nothing ported |

### A.4 Local-SGD / DiLoCo overlap and commit (→ §8)

| Source | Mechanism | Class |
|---|---|---|
| `torchft/torchft/local_sgd.py:45, 129-157` | LocalSGD + commit gate | portable |
| `torchft/torchft/local_sgd.py:175, 404, 421, 471, 551-559, 563-677, 687-691, 759-778` | streaming fragments; prepare/perform split; save/restore; staggered trigger; bucketized+quantized reduce (`:176/:225` — §0.4 trap) | portable |
| `torchft/torchft/manager.py:855-943` | `should_commit` ceremony; consecutive-failure tracking | portable semantics |
| `prime-diloco/src/zeroband/diloco.py:78-99, 204-209` | pseudo-gradient vs CPU shadow; outer step + `sync_inner_model` | portable |
| `OpenDiloco/open_diloco/hivemind_diloco.py:121-132, 158-167, 722-738` | outer-sync overlap scheduling; pseudo-grad second witness | portable |
| `agora/agora_server/src/agora_server/core/averaging/state_averager_sparta.py:47-60, 70-94, 102-108` (`beta_k_correction` `:75, 87, 93`) | sparse index selector; delayed delta-apply; β^k optimizer-state scaling | portable |
| `AsyncMesh/sparta/sparta.py:8-18, 74-139, 142-179` | `SparseSGD`; delayed buffer + avg/weight_update/ema; rotating partitions | portable; `dist.broadcast` = policy |
| `AsyncMesh/sparta/diloco.py:24-33, 41-49` | DiLoCo outer step beside SPARTA | portable |
| `asyncdiloco/AsyncLocalSGDToyExample.ipynb` | Delayed-Nesterov outer update, toy demonstration | **evidence-only** — the [DR-3] DN option ports from the paper ([ALSGD]), not from this notebook |

### A.5 Membership / progress / handoff (→ §9)

| Source | Mechanism | Class |
|---|---|---|
| `agora/agora_server/src/agora_server/core/optimization/optimizer_sparta_async_nstep.py:30-58, 63-86, 82-85, 104-115` | n-step async optimizer; two-phase ghost; staleness→ghost entry | portable state machine |
| `agora/agora_server/src/agora_server/core/server/dht_handler.py:39-50, 205-227` | ghost-phase membership metadata schema | schema portable; DHT encoding = policy |
| `node0/src/node0/server/optim.py:36-56, 80-88` | `AutoStepOptimizer`; stale-bump vs reload window | portable policy |
| `hivemind/hivemind/optim/progress_tracker.py:21-31, 44, 132, 153-186, 195, 235-237, 281-287` | sample-weighted progress, epoch increment, ETA | portable; DHT keys = policy |
| `agora/agora_server/src/agora_server/core/optimization/progress_tracker.py:59-72, 182-214` | swarm progress variant (`allow_progress_report`) | portable |
| `hivemind/hivemind/averaging/averager.py:601, 628, 653, 668-689` | state handoff: donor priority, metadata-first, chunked stream | portable — OPTIONAL per [MB-5] |
| `hivemind/hivemind/optim/optimizer.py:655-709` | `load_state_from_peers` + catch-up alignment | portable flow |
| `prime-diloco/src/zeroband/checkpoint.py:443-510` | resumable p2p checkpoint transfer | portable — OPTIONAL per [MB-5] |
| `prime-diloco/src/zeroband/comms.py:21-25, 32, 200, 393, 452` | elastic PG lifecycle, heartbeat constants | reference only — host owns views |
| `torchft/src/lighthouse.rs:141-180, 218-240`; `src/manager.rs:568-584` | heartbeating-majority quorum; round-robin recovery source | reference only — coordinator side |

### A.6 Pipeline transport / recompute / stashing (→ §10–§11)

| Source | Mechanism | Class |
|---|---|---|
| `hivemind/hivemind/moe/client/expert.py:32, 155, 188, 194-221` | remote-autograd bridge; unary-vs-stream by size | portable pattern |
| `agora/agora_server/src/agora_server/core/server/module_collab.py:88, 109-114, 122-125, 193` | backward re-forward under autocast; detach; no-grad forward | portable ([ML-5] witness) |
| `AsyncPP/runtime/runtime.py:71-72, 77-82, 168-177, 504-519, 565-591` | RuntimeStats; recompute asserted mandatory; warmup sizing; recv→fwd→send; re-forward in backward | portable |
| `AsyncPP/optim/optimizer.py:21, 39-43, 77, 103-104, 154-158` | weight-stash ring; CPU/disk stash; load old/new | portable (path: `optim/` — §1.3) |
| `AsyncPP/main_with_runtime.py:116-118, 308-314, 328-336, 447-455` | `lr_correction`; `num_versions = num_warmup+1`; adaptive-momentum formula; warmup fill | portable formulas |
| `AsyncMesh/asyncpp/…` | vendored second witness of the three AsyncPP rows above | — |
| `AsyncMesh/examples/pp_diloco_async.py:103-112, 123-163` | 1F1B loop + SPARTA mid-loop with stash-head replacement | portable integration pattern |
| `AsyncMesh/sparta/setup.py:31-35, 53, 91-105` | PP×DP rank layout formula | informative only ([PP-7]) |
| `swarm/swarm/pipeline/src/moe/client/balancer.py:22, 30, 82-84, 90-98, 111-131, 136-143` | `ExpertBalancer`: min-heap accumulated virtual runtime; 30 s DHT refresh; throughput-EMA cost correction; new-worker seeding at queue head; expiry bans; deadline reroute | portable heuristic ([PP-6]); DHT/gRPC wiring = policy |
| `pipedream/runtime/optimizer.py:19, 59, 110-114` | `OptimizerWithWeightStashing` (version deque; load old/new) — the [PP-4] origin | portable origin witness |
| `torchgpipe/torchgpipe/checkpoint.py:58-94`; `pipeline.py:49, 113-114` | clean-room checkpoint/recompute pair; clock-cycle fill/drain schedule | portable witnesses ([PP-2/3], [ML-5]) |
| `CheckFree/simulate_training/convergence_training.py:544-545, 659`; `communication/pp_protocol.py:369` | gradient-norm-weighted neighbour recovery (α/β from neighbour norms; norm tracking; weight-recovery path) | portable option (EX-5 vacancy); simulation harness = policy |
| `agora/docs/agora-system/training-architecture.md` ("Worker selection within a stage") | min-heap routing heuristic summary | **docs-only** — implementation pinned above (swarm `balancer.py`, rev 5); the full published source is [AGORA-P]:332-334 (§1.3 correction 1) |

### A.7 DiffusionBlocks (→ §13 EX-2)

| Source | Mechanism |
|---|---|
| `DiffusionBlocks/dblock_modules.py:6-20, 23-44` | equal-CDF-mass σ partition; inference σ schedule |
| `DiffusionBlocks/model.py:157-177, 179, 182-188, 203-205, 208-215, 222-231, 269` | per-block σ sampling; loss weights; σ→block inversion; EDM preconditioning; partial forward; σ-weighted CE; latent-only sampler |
| `DiffusionBlocks/vit.py:351, 690-722` | layer skip (zero inter-block traffic); `forward_block` hook |
| `DiffusionBlocks/main.py:46-49, 67, 120` | DDP-with-unused-params artifact; epoch scaling; `num_blocks` default |

### A.8 Data assignment (→ §13 EX-1)

| Source | Mechanism |
|---|---|
| `psyche/shared/coordinator/src/data_selection.rs:7, 36, 40-47, 110` | seeded shuffle, largest-remainder split, index-for-step |
| `prime-diloco/src/zeroband/data.py:509-516` | file-level `data_rank::data_world_size` split |

### A.9 Metrics (→ §12)

| Source | Mechanism | Class |
|---|---|---|
| `agora/agora_server/src/agora_server/prometheus/monitor.py:728-777` | trajectory metric taxonomy (AR duration, yield stalls, barrier waits, queue depth, chunk gaps) | taxonomy portable; log-scrape = policy |
| `AsyncMesh/asyncpp/runtime/runtime.py:71-72` | per-stage fwd/bwd micro-timing | portable hook |

### A.10 Docs-only / paper-only (non-executable)

`archive/program-docs/daemon-vhc-{architecture,refactor,execution-planning-draft,fleet-ceremony}.md`
(superseded program drafts); `cursor_decentralized_sdk_architecture-discussion.md`;
`decentralized_training_map.md` (the research map arch §0 cites);
`agora/docs/agora-system/training-architecture.md`; `research-papers/*.md` — cited throughout
via the §0.2 paper register, never as a generic source; `AsyncMesh/readme.md`;
`DiffusionBlocks/README.md` (the "8×" class of claims — §1.3 correction 3 pattern);
`sentinel-zk-swarm-v1-complete-spec.md` (verification-layer design input to arch §14 only; out
of scope here, §14.3).

---

## Appendix B — supersession and amendment register *(normative)*

Dispositions of every architecture-spec clause this document touches. **embodied** = this
document is the implementation contract for the clause, unchanged in meaning; **narrowed** = a
subset is made binding here, the rest stays design; **deferred** = explicitly not implemented by
these waves; **adopted-informative** = carried as reference shape. The architecture spec gains a
companion cross-reference to this table (applied with W0; see §0.5).

| # | Arch clause | Disposition | Where here |
|---|---|---|---|
| D1 | [VP-12] group round universal, no global scalar step | **embodied** | [DM-1], [GV-1], [DR-2] |
| D2 | [PIR-11] committed-set exchange; gossip = rung-3 new pattern | **embodied** | [DM-2], [PC-2/3], [DR-3] |
| D3 | [CO-8] measured contraction for D>0 drivers | **embodied** | [DM-3], [MO-2], W8 gate |
| D4 | [PL-1] planes decouple by clock | **embodied, reading pinned** | [DM-4], [XP-1] — the corrected reading is now the binding one |
| D5 | [PIR-12] `local`/`replicated` semantics; version = committing round | **embodied + extended** | [DM-5] adds `residual` as the stronger-than-`local` class ([CX-8/9]) |
| D6 | §4.2 PlanIR output vocabulary (groups/channels/states/schedule/data) | **embodied** (types), **narrowed** (schedule kinds arrive per Part-B promotion) | [GV-1], [GV-6] |
| D7 | §4.1 `plan@1` derivation slot ([PD-1..4]) | **deferred** — near-term instantiation is authored genesis/run config under [GV-2]; derivation discipline ([PD-2] quantization) is embodied where decisions are made ([PC-3], [PC-6], [MO-3]) | [GV-2] |
| D8 | [LC-2] refuse unenforceable vocabulary | **embodied as specified W1 behavior** | [GV-6]; lifted surface-by-surface at W7/W8 promotions |
| D9 | [CO-6] sublinear aggregation debt, staging-side fix | **embodied** | §7 entire; [PC-8] |
| D10 | [GR-6] membership epochs | **embodied** (as the [MB-1] host-view boundary) | [MB-1], [GV-5] reassignment |
| D11 | [AL-3] monitor as shared statistical substrate; statistics pre-specified | **embodied** | [MO-1/2], [DR-3]'s hard dependency |
| D12 | §12 trajectory monitor (host service + `metric@1`) | **narrowed** — promoted at W8 with the [MO] scope; screens themselves stay arch §11/§14 future | §12 |
| D13 | Appendix A.4 four-stage pipeline manifest | **adopted-informative** — becomes admissible at W7 ([GV-6] lift); its shapes are [PP-1]'s definition | [PP-1] |
| D14 | §14 verifiable-training layer (SZK input) | **deferred** — out of scope (§14.3) | — |

No clause of the architecture spec is contradicted. The one interpretive correction is D4: the
[PL-1] reading in [XP-1] (consensus-clock constraint, not a local-compute constraint) is made
binding, because the looser reading admits implementations that stall records on drain — the
failure the clause exists to prevent — while the stricter misreading forbids pipelines outright.

*End of specification.*
