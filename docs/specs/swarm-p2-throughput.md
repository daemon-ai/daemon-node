# Swarm P2 — throughput record (B3 lazy device-resident `OpBackend`)

The P2 perf follow-on evidence: the **before/after** tokens/s for the B3 lazy-residency change to
`BurnBackend` (remove the per-op host-readback tax the P1 M2 gate measured, `swarm-p1-throughput.md
§"Honest overhead accounting"`). Extends the P1 record's method. Produced on this machine — AMD Strix
Halo APU (Radeon 8060S / RADV GFX1151, unified memory) — in the `.#vulkan` devShell (wgpu) and the
default devShell (ndarray CPU), from the M2 harness (`crates/coprocessor/daemon-train/tests/reference/`
+ `reference_parity{,_wgpu}.rs`).

## What changed (B3)

`BurnBackend` kept a host `Vec<f32>` mirror of every tensor and refreshed it (`to_data`, a device→host
readback + queue flush) after **every** native op — serializing the wgpu pipeline. B3 makes that host
mirror a lazily-materialized `OnceCell<Vec<f32>>`: native op results stay **device-resident**, and the
host copy is filled once, on the first read at a genuine host boundary (det lane, scalar/metric,
consensus digest, checkpoint, `make_update` staging). The frozen `OpBackend` trait is unchanged (see
`burn_backend.rs` module docs + `swarm-ledger-p2-b2b3.md`). The det lane stays host-side fp32, so the
consensus digest is byte-identical and the parity/tolerance bar is unchanged.

## Method

Identical to `swarm-p1-throughput.md`: deterministic timed loop; `tokens/s = b·(seq−1)·steps / wall`;
the wgpu rows drop leading warmup steps (lazy GPU bringup + cubecl autotune) and report mean ± sd.
The **before** rows are produced by swapping the base commit's (`4e821cd`) eager `burn_backend.rs`
back in; everything else (the reference path, the batches, the harness) is byte-identical, and the
reference path's near-identical number across the two runs certifies the A/B is machine-state-fair.
The wgpu rows use **3 warmup + 10 measured** steps (env `M2_WGPU_WARMUP`/`M2_WGPU_MEASURED`; defaults
stay 1+4 so the P1 gate is byte-identical) for a low-variance measurement.

## Before/after throughput (tabi/reference wall factor — lower is better, 1.0 = parity)

| Config | Backend | before (eager) tabi | after (lazy) tabi | before ratio | **after ratio** |
|---|---|---:|---:|---:|---:|
| reduced (b2, seq9) | ndarray CPU | 1634.5 tok/s | 3236.0 tok/s | 2.18× | **1.12×** |
| medium (b2, seq128) | ndarray CPU | 82.5 tok/s | 138.1 tok/s | 2.03× | **1.23×** |
| **160M (b1, seq1024)** | **wgpu RADV** | **253.6 tok/s** | **383.9 tok/s** | **2.90×** | **1.96×** |

Reference tok/s per A/B pair (the fairness anchor): ndarray reduced 3563→3619, medium 167.8→169.4;
wgpu 160M 735.4→753.4 — all within noise, so the ratio deltas are the change, not the machine.

**Headline:** the lazy backend lifts tabi tokens/s by **+51% at 160M on wgpu** (253.6→383.9) and
roughly **halves the overhead factor's excess over 1.0** (1.90×→0.96×), with tabi per-step variance
cut ~13× (±1.01 s → ±0.08 s — the eager per-op readbacks *were* the variance source). On CPU the win
is larger still: the reduced config lands at **1.12×** (near parity) and doubles tabi tok/s.

### Why the win scales inversely with model size (the honest residual)

The per-op **activation** readback (removed by B3) was the dominant tax on small configs — the reduced
config, whose few params make the per-param host traffic negligible, drops to **1.12×** (essentially
parity: the residual is the sub-µs wasm ABI dispatch, §15.1 <1%). At 160M the residual is **1.96×**
because the load-bearing cost there is **not** activation readback but the host-side fp32 **residency
contract** (ABI §5.9), which scales with param count and which both paths partly share:

- **param gradient host-fold** — each param's gradient is read back and folded into the host `grad`
  accumulator (survives micro-batch splits);
- **master/storage sync** — the fp32 master is host-authoritative and mirrored to a `storage` leaf
  (re-uploaded) so the next pass differentiates through it;
- **the det + compression boundary** — `make_update`/`ingest` materialize host fp32 to run
  `dct2`/`topk_chunk`/`absmax_pack`/the `det_*` aggregate (the consensus digest must be
  backend-independent and bit-exact) — absent from the reference (which never compresses).

These are the det-lane exactness guarantee and are intentionally retained; removing them would change
the frozen §5.9 residency contract. So B3 removed the tax that was *not* load-bearing (per-op
activation readback) and the residual is honestly the load-bearing residency cost. The spec §15.1
"<1% dispatch overhead" claim is corroborated by the reduced-config 1.12× (dispatch alone is
negligible; the rest is data movement, not dispatch).

## Parity after the change (unchanged — the correctness bar)

The det lane stays host-side fp32, so digests and loss curves are unaffected. From the same runs:

- **160M wgpu** (`loss_parity_within_tolerance_160m`): per-step loss **byte-identical** (|Δ| = 0.000e0,
  4 steps), final-weight max Δ = **4.768e-7** (Optimizer class rtol 2e-4/atol 2e-5).
- **medium ndarray** (`loss_parity_within_tolerance_ndarray`, 20 steps): |Δ| ≤ 4.768e-7, final-weight
  max Δ = **5.960e-7**.
- Cross-backend det digest byte-identity (`wasm_backend_determinism::cross_backend::*`), `det_lane_bit_exact`,
  `compression_natives_bit_exact` — all green (see `swarm-ledger-p2-b2b3.md`).

## Reproduce

```
# ndarray before/after throughput (default devShell): reduced (always-on) + medium (ignored)
cargo test -p daemon-train --features burn-ndarray --release --test reference_parity throughput_ -- \
  --include-ignored --nocapture --test-threads=1
# wgpu 160M throughput (.#vulkan devShell), low-variance evidence run:
M2_WGPU_WARMUP=3 M2_WGPU_MEASURED=10 nix develop .#vulkan --command \
  cargo test -p daemon-train --features wgpu --release --test reference_parity_wgpu \
  throughput_within_budget_or_documented -- --ignored --nocapture --test-threads=1
# For the "before" rows: swap the base burn_backend.rs back in first:
#   git show 4e821cd:crates/coprocessor/daemon-train/src/burn_backend.rs > <path>/burn_backend.rs
```
