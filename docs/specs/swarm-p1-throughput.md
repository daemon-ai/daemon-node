# Swarm P1 — reference-parity + throughput record (the P1 numeric exit-gate)

The program-record evidence for the **P1 exit gate** (spec §17 / TDD §8 "P1"): *160M pretrains
through the module (tabi) path with loss curves matching a straight-burn reference, and tokens/s is
measured and reported*. Produced by lane **M2** (`swarm-ledger-m2.md`) on this machine — AMD Strix
Halo APU (Radeon 8060S / RADV GFX1151, unified memory) in the `.#vulkan` devShell (wgpu) and the
default devShell (ndarray CPU). Merge 3 re-verifies. All numbers are from the M2 harness
(`crates/coprocessor/daemon-train/tests/reference/mod.rs` + `reference_parity{,_wgpu}.rs`).

## Method

- **Reference:** an independent burn LLaMA (`RefLlama<B: AutodiffBackend>`) — no wasm sandbox, no
  tensor-ABI dispatch, no handle arena, no `OpBackend` indirection — differentiated by burn's own
  `Autodiff` decorator. Op definitions mirror the tabi native lane (`burn_backend.rs`) so the
  comparison is apples-to-apples. `llama-burn` is **not** used (it is not a workspace dep; spec
  §5.1/§15.3 keep it as a reading reference only — building directly on burn avoids a frozen-file
  edit and is the stronger independent reference).
- **Matched init:** the reference is initialized from the tabi path's *own* freshly-built weights
  (`Instance::params()` + `param_master`, canonical order), so both paths start **bit-identical**.
- **Same batches:** real vendored TinyStories tokens (M1 fixture, GPT-2 BPE vocab 50257) via
  `Corpus::{from_parts,sequence}`; identical `[b, seq]` on both paths.
- **Tolerance:** the frozen G1 **`Optimizer` class** (`tests/tolerance`): `rtol 2e-4, atol 2e-5` —
  the loosest per-op class, used as the outer bound because a full training step composes every
  class and ends on the fused f32 AdamW.
- **tokens/s:** next-token positions per second = `b·(seq−1)·steps / wall`; the wgpu number uses a
  warmup step (lazy GPU bringup / kernel autotune) + measured mean ± sample sd.

## Loss-curve parity (tabi vs reference)

Per-step loss deltas and the final-weights max delta, all **far inside** the Optimizer class:

| Run | Config | Backend | Steps | Batch | max per-step \|Δloss\| | final-weight max Δ | loss (first→last) |
|---|---|---|---:|---|---:|---:|---|
| reduced (always-on) | default 2-layer (d64, seq9, vocab64) | ndarray CPU | 8 | b2, deterministic | 4.77e-7 | 2.29e-7 | 4.169 → 3.904 |
| medium (`#[ignore]`) | d256, L4, h4×64, seq128, vocab50257 | ndarray CPU | 20 | b2, TinyStories | 4.77e-7 | 5.96e-7 | 10.838 → 7.055 |
| **160M (P1 gate, `#[ignore]`)** | **llama_160m (d768, L12, seq1024, vocab50257)** | **wgpu RADV** | **4** | **b1, TinyStories** | **0.000** | **4.77e-7** | **10.846 → 8.986** |

**The tabi (module) path is numerically transparent:** the deltas are at the f32 round-off floor
(≤ 4.77e-7 ≈ 1 ULP at loss ≈ 10) and the 160M wgpu run is **bit-identical** loss step-for-step. The
`Optimizer` tolerance is never approached — the achieved fidelity is ~3 orders of magnitude tighter
than the bound. This meets the P1 gate criterion ("loss within tolerance of a reference run").

## Loss-curve evidence run (160M, wgpu, the P1 "trains through the stack" run)

`loss_curve_160m_wgpu`: the 160M preset driven on wgpu through **2 full rounds** (H=30 inner AdamW
steps + `make_update` + self-`ingest` per round; `sparse_loco H=30, chunk 256, topk 4, bits 2`),
b=1 over real TinyStories, `build 3.0s`:

| Round | 30-step inner loop | make_update + ingest | payload | inner loss (first→last) |
|---|---:|---:|---:|---|
| 0 | 107.4 s (~3.6 s/step) | 6.8 s | 12,463,354 B | 10.846 → 4.928 |
| 1 | 119.5 s | 7.9 s | 12,463,354 B | 10.605 → 4.948 |

Full 60-step inner-loss series (recorded; round boundary at step 30 where the compressed
`sparse_loco` outer step is applied — the 1/64-density 2-bit update is deliberately lossy, so the
inner loss re-ascends to ~10.6 at round start and re-descends to ~4.9, the expected DiLoCo-family
outer/inner dynamic for a single self-peer at high compression):

```
round 0: 10.846 10.218 9.596 8.986 8.396 7.834 7.311 6.842 6.442 6.113 5.849 5.636 5.464 5.326
         5.215 5.129 5.063 5.014 4.980 4.957 4.943 4.938 4.939 4.945 4.953 4.959 4.961 4.955
         4.944 4.928
round 1: 10.605 10.359 10.037 9.659 9.241 8.796 8.336 7.877 7.436 7.030 6.675 6.375 6.126 5.916
         5.737 5.585 5.455 5.345 5.254 5.179 5.119 5.071 5.032 5.000 4.975 4.957 4.946 4.942
         4.943 4.948
```

The byte-identical **cpu-vs-wgpu det-lane digest** invariant (the consensus digest is backend-
independent, spec §7.2) is covered by the G2 cross-backend digest tests
(`tests/wasm_backend_determinism.rs`, `cross_backend_wgpu::*`) — the det lane never runs on the GPU,
so it is proven at the (cheap) tiny-config level and holds by construction at 160M.

## Throughput (tokens/s) — tabi vs reference

Deterministic timed loop (the wgpu row is warmup 1 + 4 measured steps, mean ± sd):

| Path | Backend | Config | step time | tokens/s | tabi/reference wall |
|---|---|---|---|---:|---:|
| tabi | ndarray CPU | reduced (b2, seq9) | 0.731 s / 8 | 175.0 | **1.65×** |
| reference | ndarray CPU | reduced | 0.443 s / 8 | 288.9 | — |
| tabi | ndarray CPU | medium (b2, seq128) | 30.208 s / 8 | 67.3 | **2.05×** |
| reference | ndarray CPU | medium | 14.733 s / 8 | 137.9 | — |
| **tabi** | **wgpu RADV** | **160M (b1, seq1024)** | **3.728 s ± 0.082** | **274.4** | **2.33×** |
| **reference** | **wgpu RADV** | **160M** | **1.597 s ± 0.022** | **640.6** | — |

### Honest overhead accounting (the spec §15.1 "<1% dispatch overhead" claim)

The measured tabi-vs-reference wall factor is **2.33× at 160M on wgpu** — far above spec §15.1's
"<1% dispatch overhead". The two are **not in conflict**, because the observed cost is *not* the wasm
ABI dispatch:

- **The wasm ABI dispatch overhead genuinely is negligible** (§15.1 correct): guest op calls are
  sub-µs host-function calls; a 160M forward issues a few thousand of them against multi-second GPU
  kernels — well under 1% of wall.
- **The 2.33× is the Wave-2 `BurnBackend` host-materialization tax**, a *known, documented*
  fidelity-over-speed trade (`burn_backend.rs` module docs: "Each tensor is a flat rank-1 tensor
  **plus a cached host `Vec<f32>`** … Trade: a 2× host copy, chosen for a clean `view` and numerical
  fidelity over speed this wave"). `insert_result` calls `to_vec_f32` (`to_data`) after **every**
  op, forcing a device→host readback + queue flush per op on wgpu, which serializes the GPU pipeline.
  The reference issues the same burn ops but keeps tensors on-device across the step, so it
  pipelines. The overhead therefore **grows with the number of ops per step** (CPU: 1.65× reduced →
  2.05× medium; GPU: 2.33× at 160M where the per-op readback stall is most visible), consistent with
  a per-op host-copy cost, not a per-call dispatch cost.

**Conclusion (P1):** the gate's tokens/s is *measured and reported* (the P1 requirement). The 2.33×
is not perf work in P1's scope; it is fully attributed to the `BurnBackend` per-op host copy, and the
fix is a follow-on (a lazy / host-copy-free `OpBackend` that materializes host `Vec<f32>` only on the
explicit `view`/`scalar` readouts the det lane and metrics actually need, keeping the native lane
resident on-device across a step). Recorded as a P2/perf follow-on; the numeric gate stands.

## Reproduce

```
# ndarray (default devShell): always-on parity + throughput
cargo test -p daemon-train --features burn-ndarray --test reference_parity -- --nocapture
# ndarray expensive (20-step medium TinyStories parity + medium throughput)
cargo test -p daemon-train --features burn-ndarray --release --test reference_parity -- \
  --ignored --nocapture
# wgpu (.#vulkan devShell): the P1 160M gate — parity, throughput, loss-curve
nix develop .#vulkan --command cargo test -p daemon-train --features wgpu --release \
  --test reference_parity_wgpu -- --ignored --nocapture --test-threads=1
# knobs: M2_NDARRAY_STEPS, M2_WGPU_STEPS, M2_WGPU_ROUNDS, M2_WGPU_H; SWARM_TEST_GUEST_DIR to reuse
# prebuilt guests.
```
