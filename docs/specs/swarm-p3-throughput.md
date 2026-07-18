# Swarm P3 — throughput record (Lane G: the CUDA engine arm at 160M)

> **Historical document.** Predates the `swarm`→`vhc` rename and the v1 retirement; current naming and behavior are in [`vhc-architecture-spec.md`](vhc-architecture-spec.md).

The P3 Lane-G exit-criterion evidence: the **160M single-host run on the RunPod RTX 4090** through
the new `BackendKind::Cuda` arm — reference parity inside the Optimizer tolerance class + the
tokens/s record vs the P2 wgpu figures. Extends [`swarm-p2-throughput.md`](swarm-p2-throughput.md)
(the B3 lazy-residency record; same harness, same method) with the CUDA column. Produced on the
RunPod 4090 (driver 550.127.05 / CUDA 12.4, nvrtc 12.4 staged at `/root/cuda-rt-124`,
`.#cuda-train` devShell), tree `swarm/g` @ `acc9969`, from the M2 harness
(`crates/coprocessor/daemon-train/tests/reference/` + `reference_parity_cuda.rs`).

## Method

Identical to `swarm-p2-throughput.md`: deterministic timed loop; `tokens/s = b·(seq−1)·steps / wall`;
release profile; **3 warmup + 10 measured** steps (warmup drops lazy device bringup + NVRTC kernel
JIT + cubecl autotune), mean ± sd reported; TinyStories batch `b=1, seq=1024`; matched init (the
reference consumes the tabi path's own initial state dict, bit-identical — asserted).

## 160M single-host throughput (tabi/reference wall factor — lower is better, 1.0 = parity)

| Config | Box / backend | tabi tok/s | reference tok/s | tabi step | ref step | ratio |
|---|---|---:|---:|---:|---:|---:|
| 160M (b1, seq1024) | Strix Halo, **wgpu RADV** (P2 record, lazy) | 383.9 | 753.4 | — | — | 1.96× |
| **160M (b1, seq1024)** | **RunPod 4090, CUDA** (this record) | **357.9** | **933.4** | 2.859 s ± 0.065 | 1.096 s ± 0.003 | **2.61×** |

**Headline:** the CUDA arm runs the 160M preset end-to-end on the 4090 with the **fastest reference
of any box in the program** (933.4 tok/s, +24% over RADV's 753.4 — the straight-burn ceiling of the
card) and a tabi rate of **357.9 tok/s**, in the same band as the P2 RADV tabi figure (383.9).

### Honest interpretation of the 2.61× (vs RADV's 1.96×)

The tabi/reference gap at 160M is the **host-side fp32 residency contract** (ABI §5.9 — per-param
gradient host-fold, master/storage sync, the det/compression boundary), retained by design for
det-lane exactness (see the P2 record's residual analysis). On the 4090 that cost is relatively
**larger**, not because the GPU is slower (the reference proves the opposite) but because:

- the 4090 is **discrete** — every §5.9 host materialization crosses PCIe, where the Strix Halo's
  UMA makes the same copies DRAM-local; a faster device shrinks the compute term while the
  host-boundary term stays constant, so the *ratio* grows even as absolute compute improves;
- the residency term is identical CPU-side work on both boxes, so tabi tok/s converges toward the
  host-boundary bound (~360–385 tok/s on both) regardless of GPU class.

This is the load-bearing residual the P2 record documented — not a CUDA-arm regression. The det
lane (the consensus bar) is unaffected: cross-backend det digests are byte-identical (below).

## Parity (the correctness bar — the Lane G exit criterion)

From `loss_parity_within_tolerance_160m_cuda` (release, 4 steps, TinyStories b=1, matched init):

- Per-step loss **byte-identical**: |Δ| = 0.000e0 at every step
  (10.846266 → 10.217550 → 9.595692 → 8.986223, tabi ≡ reference).
- Final-weight max Δ = **4.746e-6** — inside the Optimizer class (rtol 2e-4 / atol 2e-5), and the
  same order as the P2 wgpu 160M figure (4.768e-7 · seq-128 medium 5.960e-7).
- Loss strictly decreasing (the 160M preset trains on CUDA).

Supporting runs (same box, same tree):

- `preset_160m_trains_on_cuda` (dev profile): build 8.8 s, 4 overfit steps at 7.0 s/step,
  `make_update` 46.3 s → a 12,463,354-byte sparse_loco payload; loss 10.843 → 9.100. Green.
- `wasm_backend_determinism::cross_backend_cuda::*` (sparse_loco / diloco / demo, 6 rounds each):
  **cpu-vs-CUDA det digests byte-identical every round** while the native payloads diverge
  (tolerance-class native lane; the det lane is host fp32 by construction). Green — the consensus
  invariant with a CUDA-native contributor.
- Live plane: `fleet_heterogeneous_det_lane_agrees` — a local Linux peer + the 4090 worker
  (`--features swarm-net,cuda`, CUDA native lane **selected and active**) over the real Cloudflare
  dev coordinator, run `run-c3-fleet-1784023517`, 4 rounds, det digests byte-identical every round,
  Finished. (See `swarm-ledger-p3-g.md` for the JIT-staging findings this run surfaced.)

## Reproduce (on the CUDA box)

```
export DAEMON_CUDA_RUNTIME_DIR=/root/cuda-rt-124   # driver-matched nvrtc 12.4 + full include/ tree
# parity:
nix develop .#cuda-train --command cargo test -p daemon-train --features cuda --release \
  --test reference_parity_cuda loss_parity_within_tolerance_160m_cuda -- --ignored --nocapture --test-threads=1
# throughput (3 warmup + 10 measured):
M2_CUDA_WARMUP=3 M2_CUDA_MEASURED=10 nix develop .#cuda-train --command cargo test -p daemon-train \
  --features cuda --release --test reference_parity_cuda throughput_160m_cuda_documented -- \
  --ignored --nocapture --test-threads=1
```
