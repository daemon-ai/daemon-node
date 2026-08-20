# daemon-vhc — wild-fleet training: the blockwise-denoising design and its experiment ladder

**Status: design, pre-evidence.** This document replaces the previous revision of this file (the
two-spine program spec) in full; nothing from it is inherited (§8 says why). It specifies one
design — a merge of three published systems — and the experiment ladder that prices it.

It is written under one discipline: **"measured" is a banned word** until a run in the current
tree earns it. Every number below is either (a) a published paper's claim, pinned to file and
line, carrying the scale the source actually tested, or (b) design arithmetic whose formula is
shown inline. There are no first-party performance claims, because no first-party performance
evidence exists (§0.1).

---

## 0. Ground truth

### 0.1 What exists in this repository, today

- **The VHC substrate** (`crates/vhc/`): deterministic committed-set rounds with a bit-exact
  consensus lane (dual-compiled det kernels; the wasm32 guest is a streaming fold function over
  chunk-addressed state), the SparseLoco / DiLoCo / Demo communication profiles with golden and
  streaming-parity suites (`sdk/daemon-vhc-sdk-profiles`), content-addressed corpus, state, and
  checkpoint planes, coordinator-owned membership, and a multi-process test battery
  (`host/daemon-vhc-testkit`). Design record: the streaming det fold program
  (`decentralised-llm-training/archive/program-docs/daemon-vhc-streaming-det-fold.md`).
- **One live-fleet artifact:** a three-box ceremony of the frozen TinyLlama geometry (0.787 B
  params, `testkit/src/ceremony.rs`). It stress-tested guest memory ceilings (the wasm32
  residency findings that motivated the streaming fold) and consensus — det digests agreeing
  byte-for-byte across peers under restart and churn. It was **not** a throughput test and
  produced no throughput or quality numbers.
- **What does not exist:** any implementation of pipeline (SWARM-class) model parallelism; any
  multi-box training run beyond that ceremony; **per-group parameter ownership or independent
  group clocks** — the tree binds one whole-model state layout and one global data cursor, and
  `group_id`/`group_round` appear nowhere in `crates/vhc` (the gap §7's XS rung exists to
  close); any first-party throughput, convergence, or quality number. Earlier documents' "measured baselines" cited a retired pre-VHC tree whose
  harness no longer exists in this repository; they are struck, and nothing below rests on them.

### 0.2 Sources

Three papers carry every external claim. `[KEY]:N` pins a file line under
`~/experiments/decentralised-llm-training/research-papers/`; every pin in this document was
re-verified against those files on 2026-08-19.

| Key      | Paper                                                                 | What it evidences                                                        |
| -------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [AGORA]  | `agora__collective_and_permissionless_internet_scale_pretraining...` | the control plane, at production scale (8.6B / 500B tokens / 330 nodes) |
| [PM]     | `protocol_models__scaling_decentralized_training...`                 | the shared-subspace representation layer (2B–8B over 80 Mbps links)     |
| [DBLOCK] | `diffusionblocks__block_wise_neural_network_training...`             | the block-local training objective (12–24 layers; AR at 12L/768d/32K)   |

Network composition statistics (Ookla Speedtest Global Index, retrieved 2026-08-18: global
median fixed broadband **14.68 Mbps up / 112.07 Mbps down / 23 ms**) inform fleet expectations
only; nothing binds on them.

Scale is always named next to a claim, because the most common error in reading this corpus is
crediting a 12-layer result at 8B, or a datacenter link as "consumer": the flagship consumer-GPU
run's fleet actually spanned **213 Mbit/s–7.7 Gbit/s download (median 703, median latency
28 ms)** ([AGORA]:913) — an order of magnitude above residential uplink.

---

## 1. The design, in one page

This is not a three-way composition of peers. Two structural facts reshape the merge:

1. **Agora ships a production descendant of Protocol Models.** The Pluralis-8B run's
   inter-stage compression is the shared-subspace representation in its **reparameterized,
   frozen-basis form** — every projection and the trainable embedding factored through one
   shared basis ([AGORA]:707-719; rank 40 at d = 2048, ≈51 at d = 5120, with an explicit
   no-scaling-law disclaimer, [AGORA]:378) — and its data-parallel axis is asynchronous sparse
   parameter averaging. The representation this design grafts is therefore
   production-validated (8.6B, 500B tokens, 40 days, 330 nodes, ~170k tok/s, 63 % of a
   centralized H100 baseline, 10:1 heterogeneity tolerated — [AGORA]:33) **in its frozen
   form**; the adaptive [PM] mechanisms this design also borrows — Grassmann refresh,
   row-constant AdamW — have no production run behind them and stay labeled paper evidence.
2. **DiffusionBlocks is a partitioner-class change.** It deletes the train-time pipeline bus
   entirely: each block trains as an independent denoiser on `(x, y + σε, σ)` with a local
   loss — blocks train "in an embarrassingly parallel manner … with absolutely no communication
   overhead" ([DBLOCK]:567) — and moves all per-microbatch inter-block signal to inference
   time. What remains between groups at train time is H-amortized weight/factor sync (§5):
   megabytes per event, no activations, ever.

So the design is: **Agora's control plane, DiffusionBlocks' data plane, Protocol Models'
representation layer.**

```
MODEL    L transformer layers → B contiguous blocks, each owning an equi-probability
         noise band (γ-overlap at the boundaries, [DBLOCK]:397); AdaLN σ-conditioning.
         Shared factors:
           T_fixed   frozen high-rank embedding — genesis artifact, broadcast once
                     at join ([PM]:105)
           E (v×k)   trainable low-rank embedding coefficients, in subspace S
           U_k       the shared basis (frozen between refreshes)

WORKER   one replica of one block + the shared factors.
         loop: pull data shard → sample (σ, ε) in band, seeded
               → z_σ = embed(y) + σε, locally → block forward → local denoising loss
               → local AdamW step.
         No inter-block RPC. Ever, at train time.

CLOCKS   every step     local AdamW                             (per worker)
         H_band steps   sparse-delta averaging of block weights  (within a band's
                        — a committed-set round + det digest       replica group)
         H_E steps      factor sync: E deltas block→cohort,
                        folded E broadcast cohort→fleet
         ~500 steps     Grassmann refresh of U_k ([PM]:139) —
                        ships as a versioned (E, U_k) artifact

CONTROL  admission; placement onto lagging bands (per-band loss-progress EMA);
         per-band replication; warm-up ladder for joiners; band-local checkpoints,
         publisher-rotated. A vacant band lags and catches up on its own clock;
         only losing the factor cohort parks the run.

INFER    the only activation-bearing traffic: the T ≈ B-step Euler chain, hopping one
         worker per band; latents ship n×d (default) or n×k (the V-k bet, §3.3).
         Runs at eval cadence, and on demand.
```

The rest of this document is that page, made precise: why it composes (§2), the model and the
shared factors (§3), the protocol (§4), the traffic and memory arithmetic (§5), the bets stated
as bets (§6), and the experiment that prices them (§7).

---

## 2. Why it composes

### 2.1 Agora's cost centers map onto what DiffusionBlocks removes

| Agora's cost center (as deployed)                                                            | Under block-local training                    |
| -------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Compressed activations + activation grads on every microbatch (rank 40–51, [AGORA]:378)     | **zero** per-microbatch traffic of any kind   |
| "Throughput was bound by latency and GPU class, with bandwidth flat above the admission floor" ([AGORA]:384) | no per-microbatch round-trip exists           |
| Backward pinned to the forward worker (held activations)                                     | no cross-node backward exists                 |
| Pipeline staleness and its delay-corrected optimizer apparatus                               | no pipeline to be stale                       |
| Corruption cascading downstream of a bad stage                                               | nothing propagates between blocks at train time |

One honesty note on the second row: its source goes on to argue that the latency constraint
*relaxes* with scale — per-stage compute grows while the round-trip stays fixed, "so latency is
easier to hide at larger scale" ([AGORA]:384-386). The cost center is real at the scales and
fleets in view here; the asymptotic counter-argument is Agora's own, and a reader following the
pin should find both halves.

### 2.2 What DiffusionBlocks needs is what the other two provide

- **Every block needs the embedding and readout locally** (`z_σ = embed(y) + σε`; the loss runs
  through the readout), and every block's loss produces embedding gradients. At LLM vocab that
  is hundreds of millions of parameters to keep globally consistent — naively, a new consensus
  problem as large as the one the partitioner removed. Protocol Models' embedding decomposition
  is exactly the missing tool ([PM]:101-105, §3.3): a frozen high-rank component broadcast once,
  plus a trainable low-rank component small enough to ride the ordinary sync cadence. This is
  the deepest genuine synergy in the merge.
- **Blocks need admission, replication, churn handling, and within-group consensus.** Agora's
  control plane carries over with "stage" relabeled "band" — and in this repository, the
  committed-set round + det digest is the same machinery the three-box ceremony already
  exercised (§0.1). The calibration imports: heterogeneity tolerated to 10:1, degradation
  beginning when averaging participation falls below ~85 %, recovery on pruning ([AGORA]:33);
  669 joins / 607 departures over 40 days as the churn shape ([AGORA]:917).
- **Inference still chains blocks sequentially** (the Euler denoising chain), so an inter-block
  latent bus persists at inference and evaluation time — and Protocol Models compresses
  precisely that link, if the latent lives in subspace coordinates (§3.3, V-k).

### 2.3 What the fleet physics buys

Consumer uplink is the scarce axis (14.68 Mbps global median, §0.2). Every published
wild-training result that carries per-microbatch traffic sits on ≥80 Mbps tested links
([PM]:9 at 80 Mbps is the floor of the corpus; [AGORA]:913 is far above it); none exists below.
This design does not compress that traffic class — it **deletes** it. What remains at train
time is data shards (downlink, cacheable), sparse weight averaging within a band, and a tiny
factor sync — all H-amortized, all sized in §5. By this arithmetic a 10–20 Mbps residential
peer is a first-class trainer — a claim no published pipeline design can make, and one that
stays arithmetic until X0 supplies the step time and X3 the wire accounting.

---

## 3. The model and the shared factors

### 3.1 Blocks and bands

A decoder-only transformer of L layers is split into B contiguous blocks. Each block is trained
as an independent denoiser for one noise band: bands partition the σ-distribution by
equi-probability mass (the paper's ablation shows this beats uniform partitioning,
[DBLOCK]:247), with **γ-overlap** extending each band in log-σ space (γ ∈ [0, 0.1], 0.05
default, 0.1 for text — [DBLOCK]:397). σ-conditioning via AdaLN. Overlap gets one job here
beyond its published quality role, stated as a design inference with no paper claim behind it:
a neighbour block is *trained* on the overlapped σ-range, so the inference-time Euler chain can
cross a vacant or stale band on the neighbour's overlapped competence. Train-time vacancy
tolerance needs no overlap at all — there is no train-time coupling to break (§4.3). X2
carries the on/off arm.

**Causal consistency** for AR training follows the paper's concatenation scheme: noisy tokens
attend to their clean past under a modified causal mask — one forward pass, doubled sequence
memory; the two-pass KV alternative trades compute for memory ([DBLOCK]:481). The attention-FLOP
overhead of concatenation is our arithmetic to bound, not the paper's claim: X2 reports both
variants in bytes and FLOPs.

### 3.2 What a block's step consumes

Raw input tokens x (clean past), the target y, a seeded (σ, ε) draw, and the shared factors.
Never another block's hidden state ([DBLOCK]:567). Memory per device scales as
(4P + A)·(L/B) versus (4P + A)·L for end-to-end — all components divided by B, the only
published mechanism that removes O(model) from a device without creating per-microbatch traffic
([DBLOCK]:561-563).

### 3.3 The shared factors — the Protocol Models graft

The embedding table is decomposed per [PM]:101-105:

- **`T_fixed`** — frozen, high-rank, "transmitted to all nodes and stored" ([PM]:105): a genesis
  artifact in this substrate's vocabulary, content-addressed and broadcast once at join. Its
  lookups are ephemeral (no gradient state; negligible peak-memory pressure, [PM]:313-315).
- **`T_S = E·U_kᵀ`** — the trainable low-rank component, parameterized directly as coefficients
  `E ∈ R^{v×k}`. This is Agora's own deployed form — the reparameterized SSN factors the
  embedding exactly this way ([AGORA]:719) — and the in-S-by-construction reading of [PM]'s
  `T_S = T_fixed·U_k·U_kᵀ` ([PM]:105), which becomes the initialization: E₀ = T_fixed·U_k.
  No optimizer modification is needed for E; [PM]'s row-constant AdamW and per-iteration
  projection ([PM]:143) enter only where a full projection weight must stay in S (see V-k
  below).
- **`U_k`** — the shared basis, frozen between infrequent Grassmann refreshes (every ~500
  iterations, [PM]:139). [PM] drives the refresh from leftover-gradient mass at the last
  *compressed transformer layer* ([PM]:131-133); this design has no such boundary, so the
  statistic is redefined — a design decision with no published witness: the cohort aggregates
  the out-of-S residual mass of the blocks' embedding-gradient contributions, the same quantity
  measured at the one surface every block shares. **A refresh changes E's coordinate system**,
  so it never ships alone: the cohort rebases `E' = E·(U_kᵀU_k')`, bumps `basis_version`, and
  broadcasts `(E', U_k')` as one artifact; blocks switch atomically on version, and
  mixed-version training is bounded by the declared staleness, never silent. Basis provenance
  is a declared run parameter: random init + refresh is [PM]'s published position; Agora
  deploys a fixed shared basis via reparameterization ([AGORA]:707-719) without documenting its
  provenance; frozen-random-without-refresh has no published witness and is not used here.

The readout is weight-tied to the embedding, so one factor pair covers both surfaces. [PM]:101's
caveat is carried honestly: restricting the **full** table to S "degrades network performance";
the fixed-plus-low-rank split is the published middle, and whether it holds under the denoising
objective is untested — it is X1's question, with a dense-table fallback that costs consensus
bytes, not the design.

**Two latent variants**, priced separately:

- **V-d (default):** the diffusion latent z lives in d dimensions as published; the factors
  compress only the *trainable consensus surface* of the embedding (v×d dense → v×k
  coefficients). Closest to both papers; the conservative graft.
- **V-k (bet B5):** sample ε inside S and run the whole denoising problem in k dimensions —
  latents become n×k coefficients end-to-end, which also compresses the inference chain ~d/k×.
  The latent's clean component is an embedding, and `z − T_fixed − PE` lies in S by construction
  ([PM]:107-111 is the analogous forward identity) — but whether a k-dimensional latent carries
  the denoising signal at scale is open theory with no published witness. It also extends the
  subspace constraint to each block's input/readout projections, which is where [PM]:143's
  modified optimizer would genuinely bind. An X1 arm, load-bearing nowhere.

**Factor gradient flow:** block owners accumulate low-rank factor gradients locally and ship
them block→cohort on the slow clock; the factor cohort folds contributions (a committed-set
reduction) and re-broadcasts. Both directions are megabytes at any scale in view (§5).

---

## 4. The protocol

### 4.1 Four clocks, no barriers

| Cadence      | What                                                                       | Consistency                                         |
| ------------ | --------------------------------------------------------------------------- | ---------------------------------------------------- |
| every step   | local AdamW on the block replica                                            | local trajectory                                    |
| H_band steps | sparse-delta averaging of block weights within the band's replica group     | committed-set round; det digest bit-identical       |
| H_E steps    | factor sync: E contributions up, folded E broadcast down                    | committed-set round on the factor cohort's clock    |
| ~500 steps   | Grassmann refresh of U_k → versioned (E, U_k) artifact (§3.3)               | atomic switch on `basis_version`                    |

Bands run on independent round clocks; a slow band lags, it does not stall anyone. There is no
global optimizer step to coordinate — the global batch-size machinery of a pipeline run has no
meaning here, and only the factor clocks are fleet-wide. H_band and H_E are genesis parameters;
[PM]'s ~500-step cadence is evidence for the *refresh* only, and no published result prices
staling a shared trainable E — so H_E starts at H_band (the sync is megabytes, §5) and X1's
stale-mirror arm prices stretching it.

The band clock runs the substrate's existing **SparseLoco** profile: top-k *delta* exchange
with an error-feedback residual, folded det-exact over the committed set (§0.1). That is a
different algorithm from what Agora deployed on its DP axis — AsyncSPARTA averages rotating
disjoint *parameter segments*, 5 % per round every 20 steps, covering every parameter once per
400 steps ([AGORA]:1094) — and their two sparsity fractions mean different things, so Agora's
operating point does not calibrate SparseLoco's. H_band and the top-k fraction are priced at
X3 from measurement; rotating-segment averaging stays registered as an alternative band
profile with its own port, not silently blended into this one.

**State classes, stated once:** block weights are consensus-canonical within their band (det
fold, digest-checked per round). The factor pair (E, U_k) is consensus-canonical **on the factor
cohort's clock**; block owners hold *mirrors* refreshed by the broadcast, staleness-bounded by
the declared factor cadence. Blocks training against a slightly stale embedding is a real
perturbation of the objective — X3 injects refresh delay deliberately and X2's ablations carry
a fresh-vs-stale arm. `T_fixed` is a frozen genesis artifact; optimizer moments are
replica-local, checkpointed with the seat; the error-feedback residual is `residual`-class —
checkpoint-mandatory, because a top-k codec that loses its residual is a different (possibly
non-convergent) algorithm, not a degraded one.

**The evaluation cut.** Independent clocks make "the model" ambiguous without a snapshot rule,
so one is declared: a model snapshot is each band's latest sealed checkpoint at or before the
cut, plus the factor artifact at exactly one `basis_version`, with every declared staleness
bound satisfied. An eval site assembles one snapshot and names it by its per-band roots and
basis version; band-round skew and factor-version skew are instrumented at X3/X4.

### 4.2 Determinism and verification

The det lane is the consensus bar: same committed inputs, same bytes, same digest — the property
the three-box ceremony exercised, inherited unchanged. Block-local objectives compose with it
into something no pipeline design has: **a training claim is self-contained**. The (σ, ε) draw
is seeded per sample, so replaying a claimed update needs one block's weights plus the shared
factors — no held activations, no upstream stages, no cascading blame. Stated with the
substrate's own precision: native forward/backward math is tolerance-class across backends, so
cross-vendor spot-checks are *tolerance-based* re-execution plus blame localization; bit-exact
recomputation holds same-backend, and what the det lane reproduces exactly is the aggregation
over committed payloads. That is weaker than "deterministically recompute any update" — and
still the cheapest verification story in the corpus, with the poisoning surface shrunk to the
factor consensus and the inference chain. Out of scope to build in v1; recorded because it is
the design's structural upside if the bets land.

### 4.3 Churn, placement, joins

- **Band vacancy is lag debt, not failure.** A band that loses all replicas restores from its
  band-local checkpoint family and catches up on its own clock while every other band keeps
  training. Only the factor cohort's vacancy parks the run — so it carries replication one above
  the band target (bands: min 2, target 3; cohort: target 4), and checkpoint restore latency is
  a first-class engineering target (it enters any coverage model at power R−1). One honesty
  note: lag-tolerant vacancy is **new substrate vocabulary** — today's contracts park a run
  when any group loses coverage, and nothing in the planned vocabulary distinguishes a
  lag-tolerant group from a required one. XS (§7) lands it as a declared per-group liveness
  class (`required | lag_tolerant`): bands are lag-tolerant, the factor cohort is required.
- **Placement balances band progress, not throughput.** The composed sampler is only as good as
  its worst band, so the admission/placement signal is per-band loss-progress EMA: joiners land
  on lagging bands; a persistently lagging band gets replicas before a fast band gets depth.
  Heterogeneity tolerance is structural — a slow GPU trains its band slower and the scheduler
  compensates — with Agora's 10:1 and ~85 %-participation numbers as the calibration to beat
  ([AGORA]:33).
- **Joins:** restore the band family (streamed; the checkpoint plane exists, §0.1) → receive-only
  warm-up → weight-zero contribution while moments warm → active. Every gate is a declared
  threshold in genesis, never a baked-in constant.
- **Checkpoints:** band-local families, single deterministic publisher per cadence slot,
  byte-budgeted remote cadence — the ratified pattern of the streaming det fold program
  (its D-SF3), reused verbatim.

---

## 5. Budgets — design arithmetic, no measurements

At the canonical per-block geometry (§7) — 3 layers of d = 768, v = 32K tied → ~21 M params
per block (the paper config at B = 4 and X2's L = 24 sweep at B = 8 share it); sequence
n = 1024. Formulas inline; every figure re-derivable; other arms scale by block size.

| Bus                       | Bytes per event                                        | Cadence            |
| ------------------------- | ------------------------------------------------------- | ------------------- |
| data shards (down)        | 2 B/token → ~2 KB per step at b=1, n=1024               | every step; cacheable |
| band averaging (up+down)  | ~0.06 B/param wire — the det-fold record's ceremony-tier figure (~50 MB @ 0.787 B params), an index-encoding-dominated format that record itself flags; a placeholder until X3's byte accounting → **~1.3 MB** per replica | per H_band-step round |
| factor sync               | E + U_k = 32K×64 + 768×64 ≈ 2.1 M params → **≈ 8.4 MB each way** f32 dense (contribution up, folded broadcast down; compression optional at this size) | per H_E steps |
| band checkpoint publish   | (master + EF + 2 moments) = 16 B/param → **~0.34 GB** per family | publisher-rotated, byte-budgeted |
| inference/eval chain      | V-d: n×d bf16 ≈ 1.6 MB per hop; V-k: n×k ≈ 0.13–0.26 MB | per Euler step, eval cadence only |
| **per-microbatch activation traffic** | **0 — the deleted class**                   | —                   |

Converting bytes-per-round to Mbps needs a step time this repository does not yet have — X0
produces it. The shape is already decisive without it: every training bus above is H-amortized
and megabyte-scale; the only per-microbatch cost in the whole design is local compute.

**At target scale** (an 8B-class model at d = 5120, 128K vocab, B = 16, ~0.5 B params/block):
band averaging ~32 MB/replica/round at the same wire rate; factor sync 128K×64 ≈ 8.4 M params
≈ 34 MB f32 dense each way per H_E steps; `T_fixed` ≈ 1.3 GB bf16 held per worker
(lookup-ephemeral, [PM]:313-315) plus a transient n×V logit tensor ≈ 0.5 GB bf16 at n = 2048 —
the vocab terms are material at exactly the consumer tier this design targets, so X2's
residency plot carries the vocab axis explicitly. Per-block optimizer state at f32-everything
is 16 B/param ≈ 8 GB, ~3 GB with bf16 weights/grads and 8-bit moments — the 8–16 GB consumer
tier is reachable at B ≥ 16. And an 8B model at d = 5120 is only ~25 layers deep, so B = 16
puts per-block depth at L/B ≈ 1.6 — below every published observation and below the crossover
§6's B2 locates. The target operating point is therefore an *output* of X2's floor-finding,
not this paragraph's assumption: if the floor lands at L/B ≥ 3, the target rebalances —
deeper-narrower at fixed parameters, or smaller B with larger per-block shares and an honestly
higher device floor. That tension is the experiment.

For contrast, the deleted bus: a pipeline at this width ships n×d ≈ 21 MB uncompressed (≈0.2 MB
at ~100× subspace compression) per boundary, per direction, per microbatch, forever — the traffic
class whose published feasibility floor is 80 Mbps (§2.3).

---

## 6. The bets, stated as bets

- **B1 — scale.** [DBLOCK]'s AR evidence is a 12-layer / 768-d / 32K-vocab model, B = 4, 10
  epochs ([DBLOCK]:477), evaluated by MAUVE and teacher perplexity because standard NLL is
  non-trivial for it ([DBLOCK]:479). The target is ~40× wider-deeper and ~1000× more tokens.
  Nobody knows the quality at that distance; this is the load-bearing bet.
- **B2 — band count vs block depth.** The published curve is non-monotone with an interior
  optimum: at L = 24 on ImageNet, B = 2–4 all *beat* end-to-end (FID 9.90 / 11.11 / 11.90 vs
  12.09 at B = 1), and the crossover to worse sits between B = 4 and B = 6 — between L/B = 6
  and L/B = 4 ([DBLOCK]:227-235); B = 4 was best for LM at L = 12 ([DBLOCK]:555). So the bet
  is not "how much quality does B cost" but **where the per-block-depth floor sits**: §5's
  memory story wants B ≥ 8–16 and puts the target-scale point at L/B ≈ 1.6, below every
  published observation. The published sweeps conflate band count with per-block depth; X2's
  factorial separates them and locates the floor.
- **B3 — no cross-block feature hierarchy.** Blocks condition on raw context, never on upstream
  representations. At 12 layers this costs little; at 32+ it may be exactly what caps quality.
  The registered fallback is **hybrid stitching**: rare, subspace-compressed end-to-end passes
  interleaved with block-local training — a two-timescale objective, unpublished, never
  scheduled without X2's verdict. The free variant needs no design: a deep block already
  backprops end-to-end internally on its owner.
- **B4 — the embedding split under the denoising objective.** [PM] validated the split under
  end-to-end backprop; carrying it under block-local denoising is new. Favorable structure (the
  latent's clean component *is* an embedding), no witness. X1's question; dense fallback.
- **B5 — subspace-coordinate latents (V-k).** Open theory, no published witness, load-bearing
  nowhere; an X1 arm because the payoff (~d/k× on the inference chain, and the cleanest possible
  factor story) justifies one cheap experiment.
- **B6 — the one global object.** The factor cohort is the single fleet-wide clock: every
  replica contributes to its fold and consumes its broadcast, and its vacancy is the one event
  that parks the run. The fan-in is a committed-set fold — contributors need not be
  simultaneous, and partitioned reduction is planned substrate machinery for when the roster
  grows — but this is structurally the design's one barrier-shaped survivor, the place where
  "delete synchronization" didn't. Priced in §5, replicated above the bands (§4.3), exercised
  by X3's kill-the-cohort gate; escalations if it binds at fleet scale: stretch H_E,
  hierarchical folding, or (V-d only) freeze the embedding after warm-up.
- **The artifact is different, and this is not a footnote.** The product is a blockwise-denoising
  LM whose inference is a T ≈ B-step Euler chain — a different generation cost and evaluation
  methodology from a next-token LM. The distance between this artifact and a standard AR model
  is priced once, at X0/X2, against a matched AR baseline; it is B1's price tag, not a per-rung
  gate.

---

## 7. The experiment — machinery first

**What this experiment is for:** landing and proving the substrate machinery — XS's
generalizations and the SDK primitives they exercise — over a real internet ceremony, with
model quality benchmarked against published claims. It is explicitly **not** a run at the §5
target scale: the 8B arithmetic there prices the design's ambition, and nothing in this ladder
depends on reaching it.

**The canonical model is [DBLOCK]'s own AR configuration** — 12-layer Llama-2-style, d = 768,
32K vocab, tied embeddings, ~110 M params ([DBLOCK]:477) — chosen precisely so the
decentralized runs have an external comparator: the paper's published MAUVE /
teacher-perplexity results ([DBLOCK]:479), bridged through our own X0 reproduction
(cross-codebase comparisons anchor on X0; the paper's numbers are the reference X0 must
approach). The L = 24 extension appears only inside X2, where floor-finding needs depth the
paper config lacks.

Two tracks, run in parallel, each rung naming its kill criterion or gate. The **research
track** (X0 → X1 → X2) is single-process research-harness work — any one GPU box, deliberately
not on the substrate. The **machinery track** (XS → X3 → X4) is the deliverable and the
successor to the TinyLlama ceremony (§0.1): X3's topology does not exist on the tree as it
stands — XS is the contract work that creates it, with the existing ceremony machinery
supplying consensus, checkpointing, and transport underneath. X3 consumes X0's model config
and XS's contracts; it does not wait on X2's verdict.

- **X0 — reproduce the paper.** [DBLOCK]'s AR configuration verbatim: 12-layer Llama-2-style,
  d = 768, 32K vocab, B = 4, γ = 0.1, seq 256 (LM1B), batch 256, AdamW 3e-4, 10 epochs
  ([DBLOCK]:477) — plus the matched standard-AR baseline, and the MAUVE / teacher-perplexity
  harness ([DBLOCK]:479). Produces the step time §5 needs and the comparator every later rung
  reuses. One eval decision made explicitly rather than inherited: the paper compares a
  top-p 0.95 AR baseline against its own 4-step greedy decoding ([DBLOCK]:479); X0's
  comparator matches sampling regimes. **Gate, not a kill: cannot approach the paper's own
  numbers** — treated as a port bug until proven otherwise (the config is small and public);
  the ladder's first true kill lives at X2. The recurrent-depth variant ([DBLOCK]:485) is an
  optional side-arm: it needs no block partitioning at all, so it is the cheapest on-substrate
  smoke of the objective if one is wanted early.
- **X1 — the graft, single process.** X0's config with the §3.3 factors. Arms: dense table
  (control) · V-d split · V-k latents · fresh-vs-stale factor mirrors (§4.1). **Kill for V-d:
  degradation beyond tolerance → dense-table fallback** (costs bytes, not the design). V-k
  failing is an acceptable outcome of a cheap arm; it retires B5.
- **X2 — the (B, L/B) factorial at L = 24, then the horizon extension.** The research track's
  one departure from the paper config: floor-finding needs depth the 12-layer model lacks
  (~170 M non-embedding params; 24 divides cleanly across the sweep). B ∈ {2, 4, 8, 12} at
  L = 24 crossed with B ∈ {2, 4} at L = 12: the equal-depth pairs (L/B = 3: 24/8 vs 12/4;
  L/B = 6: 24/4 vs 12/2) disentangle band count from per-block depth — the published ablations
  cannot (§6, B2) — and the B = 12 arm (L/B = 2) brackets the target-scale operating point
  (§5) from above, the region no published sweep touches. γ on/off; concat vs two-pass KV in
  bytes and FLOPs; residency recorded with vocab
  terms. Two comparators, kept separate: the B = 1 same-objective oracle (prices band count)
  and the X0 AR baseline (prices the artifact, B1). **The winning B gets a ≥10× token-horizon
  extension before anything downstream consumes it** — all published curves are 10-epoch-short,
  and curves crossing at 10–100× tokens is the failure mode a bare sweep cannot see. **Kill: no
  B ≥ 8 inside tolerance at the extended horizon** → the memory story caps at B = 4 (device
  floor rises to the 12–16 GB class), hybrid stitching (§6, B3) becomes the active
  investigation, and the wild-fleet ambition shrinks honestly.
- **XS — the substrate rung (contract work, priced as such).** X3's topology does not exist in
  the tree (§0.1): state layout binds one whole-model parameter list, the data cursor is
  global, rounds have one implicit group. XS lands, against the existing parity oracles:
  per-group state-layout bindings and per-group data cursors, with the degenerate gate that a
  single-group binding reproduces today's genesis and family roots byte-for-byte; group-scoped
  rounds (the barrier round parameterized by group — deliberately small); the group vocabulary
  including `basis_version` and per-channel staleness bounds; the **per-group liveness class**
  (`required | lag_tolerant`, §4.3) — the one vocabulary item this design adds to the plan;
  and the factor artifact path (`T_fixed` as a genesis artifact; the versioned (E, U_k)
  broadcast, §3.3). These are the SDK primitives spec's W1-class contract items
  ([GV-1]/[GV-4]/[GV-5]) plus its cohort-scoped rounds ([DR-2]), pulled onto this experiment's
  critical path; the rest of that plan stays off it. X3's digest gates are XS's acceptance
  criteria run on real boxes.
- **X3 — the band ceremony (the 3–4 hosted boxes).** The fleet is real WAN — hosted nodes over
  the internet with decent bandwidth — so latency is real but the consumer-uplink claim is not:
  at least one pass runs with links shaped to §0.2's consumer median (~15 Mbps up, ~25 ms), so
  §2.3's headline is exercised, not assumed. Quality is not this rung's question; systems
  behavior is. Two arms on the same boxes:
  - **X3-mesh — the minimal non-degenerate mesh.** X0's model at B = 2 (L/B = 6), two replicas
    per band, the factor cohort co-hosted on the strongest box (four seats on four boxes; on
    three, one box doubles seats — a functional arrangement, not a performance one). The
    smallest geometry where every axis is simultaneously real: two independent band clocks,
    within-band folds with ≥ 2 contributors and cross-peer digest equality, the cohort folding
    contributions from both bands — and every liveness case live: kill one replica → the band
    continues; kill both replicas of one band → lag, restore, catch-up; kill the cohort →
    park → resume.
  - **X3-paper — the external benchmark arm.** The paper's exact configuration, B = 4
    ([DBLOCK]:477), one replica per band: the published single-process result rerun under this
    protocol, evaluated with the paper's own harness against our X0 reproduction, the paper's
    numbers as the external anchor. One replica per band leaves the within-band axis idle by
    construction — the arm isolates what decentralization *adds*: block ownership over the
    substrate, factor/embed sync on the cohort clock, independent band clocks.
  Genesis declares the four clocks (§4.1). Gates, all systems-level: **every wire byte
  accounted against §5's formulas at each arm's block size, with zero activation-class
  traffic**; per-band det digests bit-identical across restart; the liveness cases above
  observed live; factor-refresh delay injected deliberately (the §4.1 staleness bound
  exercised, not assumed).
- **X4 — churn and placement campaign.** Seeded joins and departures against X3's fleet: the
  full join ladder (restore → warm-up → weight-0 → active), placement moving capacity to lagging
  bands on the loss-EMA signal, correlated whole-band loss, publisher-slot churn. Instruments:
  catch-up time, warm-up fraction of session life, band-balance spread — the statistics no
  paper publishes and this fleet lives or dies by. At this fleet size they are correctness
  smokes, not fleet claims: the gates are that the machinery behaves, never a throughput or
  scaling number.

**Standing gates at every rung:** the comparator discipline above; det digests never diverge;
traffic accounting matches §5 or the table is corrected in the same commit; and §0's language
rule — a number graduates from "design arithmetic" to "measured" only with a reproduce path in
the tree.

---

## 8. What this replaced, and the rule that survives

The previous revision of this file specified a two-spine program — a consumer block spine plus
an Agora-shaped subspace-pipeline anchor mesh — as normative policy over a clause vocabulary
spanning four other documents, anchored on "measured facts" from a pre-VHC tree whose harness no
longer exists. It is struck for two reasons. The evidence chain was circular: documents citing
documents citing deleted code, with no claim terminating at a runnable artifact. And the hedge
was idle: the pipeline spine had no implementation, no funded experiment, and no path to one
here — while everything of Agora's that this design actually needs (the control plane's
admission/replication/averaging patterns, and a routed transport for the inference chain)
survives the merge without a training pipeline ever being built.

One item from the struck revision is rescued rather than lost with it: the architecture and
SDK-primitives specs ship conflicting channel-class enumerations (`param_delta | activation |
kv` in `vhc-architecture-spec.md` vs the five-class set in `vhc-sdk-primitives-spec.md`'s
[GV-1]) — a real contract bug, verified against both files, that XS inherits and must resolve
before manifest shapes freeze.

If §6's bets fail, the honest fallback is not a pipeline this program never built; it is the
substrate's existing full-replica DP profile at device-fit scales, hybrid stitching if X2's miss
is marginal, and an honest re-scope.

The rule that survives the episode, and now heads this file: a claim is "measured" only if its
reproduce path executes in the current tree. Everything else is design, and says so.

*End of specification.*
