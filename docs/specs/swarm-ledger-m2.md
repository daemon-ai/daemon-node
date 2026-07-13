# Swarm P1 — lane M2 ledger (reference parity + throughput, Wave 3)

Lane record for **M2** of the "Swarm P1 + Transport" program, Wave 3 — the **P1 numeric exit-gate
evidence lane** (program ledger: [`swarm-p1-ledger.md`](swarm-p1-ledger.md); the plan conventions
are the contract; predecessor lanes: [`swarm-ledger-m1.md`](swarm-ledger-m1.md) for the 160M preset
/ `canonical_param_layout` / safetensors surface, [`swarm-ledger-g2.md`](swarm-ledger-g2.md) for the
wgpu backend / autotune / GPU-skip convention). M2 proves the P1 gate (spec §17 / TDD §8 "P1"):
**160M pretrains through the module (tabi) path with loss curves matching a straight-burn reference,
and tokens/s is measured and reported.**

## Base + branch

- **Repo / worktree:** `daemon-node` @ `/home/j/experiments/daemon-worktree/swarm-engine`.
- **Base commit:** `2f1ce1f` (`mirror(merge-2): freeze Wave-2 interfaces`) on `integrations/swarm-p1`.
- **Branch:** `swarm/m2`.
- **Owns (this wave):** the reference-parity + throughput harness in
  `crates/coprocessor/daemon-train/tests/*`, additive `daemon-train-sdk`/`guests`/`data.rs` +
  `daemon-train-sdk` models goldens **where additive**, and the program docs
  `docs/specs/swarm-p1-throughput.md` (new) + this ledger + the (housekept) UMA findings docs.
- **Never touched:** the main checkout `/home/j/experiments/daemon` (read-only); FROZEN files (root
  `Cargo.toml`, `deny.toml`, `flake.nix`); other lanes' dirs (`daemon-swarm-net`/`daemon-swarm-run`
  engine/transport are B3's this wave; `daemon-api` is W1's). **`daemon-train/src/*` is NOT edited**
  — the harness drives the tabi path exclusively through the already-public `Instance` /
  `WasmBackend` surface (`params()`, `param_master(name)`, `register_batch`, `step`, `inner_update`,
  `metrics()`, `make_update`), so no single-writer collision with the G-lane's `src/` files.

## Frozen-file need? **NO.** (the `llama-burn` question, resolved)

The plan's reference-pack line names `llama-burn` (tracel-ai/models) as the parity oracle. It is
**not** a workspace dependency, and spec §5.1/§15.3 are explicit that `llama-burn` is "kept as a
numerics golden-test **reference**, not a runtime path". Adding it would be a root `Cargo.toml`
edit (a FROZEN file) and pull a large model-zoo tree. **Decision: build the reference model directly
on `burn` inside the M2 test harness** — burn is already a direct dependency of `daemon-train`
(`crates/coprocessor/daemon-train/Cargo.toml:30`, with `ndarray`+`autodiff` always on and `wgpu`
behind the feature), so the self-contained reference needs **no new dependency and no frozen-file
change**. This is also the stronger reading of the gate ("loss curves matching a straight-burn
reference"): a hand-written burn module is a genuinely independent implementation of the same
architecture, differentiated by burn's own `Autodiff` decorator, rather than a third-party port.
**No ledger note for the integration owner is required** — no frozen-file need arose.

## What "independent-of-the-tabi-host" means here (the reference design)

The tabi (module) path is: guest `tiny_llama.wasm` → wasm ABI dispatch → `WasmBackend` → the
`OpBackend` engine (`BurnBackend<Autodiff<B>>`, `burn_backend.rs`) → burn. The **reference** is a
plain Rust `burn` module (`RefLlama<B: AutodiffBackend>`) that issues burn tensor ops directly —
**no wasm sandbox, no ABI, no handle arena, no `OpBackend` indirection**. It is therefore independent
of the *tabi host* (the thing under test) while remaining a faithful burn implementation of the same
math, so any per-step divergence isolates to (a) burn kernel non-associativity across two different
op-issue orders and (b) f32 AdamW accumulation — both **tolerance-class** effects (spec §7.2,
program "Determinism story"), never the det lane.

**Op-definition grounding (mirrored from the tabi native lane so the comparison is apples-to-apples,
`burn_backend.rs`):**
- RMSNorm: `x · rsqrt(mean(x²,‑1)+eps) · w` (`burn_backend.rs:386-393`).
- RoPE: **half-split** (not interleaved; the preset calls `rope(…, false)`), `freq_j =
  θ^(−2j/hd)`, `out1=x1·cos−x2·sin, out2=x1·sin+x2·cos` (`burn_backend.rs:404-450`, model call
  `models.rs:370-371`).
- Attention: dense causal `softmax(QKᵀ·scale + mask)·V`, `mask = −1e30` above the diagonal,
  `scale = 1/√hd` (`burn_backend.rs:451-482`, model call `models.rs:372-373`).
- SwiGLU: `silu(x·Wgate) ⊙ (x·Wup) · Wdown` (`models.rs:380-384`; `activation::silu`).
- Cross-entropy: shifted log-softmax with a **detached** per-row max, mean over counted rows
  (`burn_backend.rs:347-375`).
- AdamW (fused, f32): `t=step`, `m←β1·m+(1−β1)g`, `v←β2·v+(1−β2)g²`, `m̂=m/(1−β1ᵗ)`,
  `v̂=v/(1−β2ᵗ)`, `w←w·(1−lr·wd) − lr·m̂/(√v̂+eps)` (`burn_backend.rs:556-594`); step index is
  `inner_step+1` (`models.rs:400-410`); hyperparams `lr 4e-4/β[0.9,0.95]/eps 1e-8/wd 0.1` (TDD §5).
- Loss scaling: `loss·(size/step_seqs)` before backward (`api.rs:685-692`); the harness drives both
  paths with `step_seqs = num_sequences` ⇒ scale `1.0` ⇒ plain mean-loss backward.
- burn API anchors: `Tensor::{matmul,swap_dims,select,reshape}`, `activation::{silu,softmax}`,
  `Tensor::backward`/`grad`, `burn::backend::{Autodiff,NdArray,Wgpu}` (burn 0.21).

**Matched init (the safetensors substrate, M1 seam 4).** The reference is initialized from the tabi
path's *own* freshly-built weights: build the `WasmBackend`/`Instance`, read every param via
`Instance::params()` + `param_master(name)` (canonical registration order), and load those exact
fp32 arrays into `RefLlama`. So both paths start from **bit-identical weights** — the only honest way
to compare loss curves. Final-weights parity is asserted by exporting both paths' masters to
`daemon_train_safetensors::StateDict` and comparing element-wise within the tolerance class.

**Same batches.** Token batches are the **real vendored TinyStories fixture**
(`daemon-swarm-run/tests/fixtures/tinystories/`, M1) loaded via
`daemon_swarm_run::data::Corpus::{from_parts,sequence}`; both paths consume the identical
`[B, seq_len]` `u32` tokens (GPT-2 BPE vocab 50257). The reduced always-on variant uses a
deterministic small-vocab batch (identical on both paths) since the tiny config's vocab is smaller
than GPT-2's.

## Tolerance methodology

Per the plan, the outer bound is the **`Optimizer` tolerance class** from the frozen G1 harness
(`tests/tolerance/mod.rs`: `OpClass::Optimizer` ⇒ `rtol 2e-4, atol 2e-5`, via `tol_for`) — the
loosest per-op class, appropriate because a full training step composes every class and finishes on
the fused AdamW. Per-step loss deltas and the final-weights max delta are **recorded** (this ledger +
`swarm-p1-throughput.md`), not just asserted, so the achieved fidelity is on the record. Losses are
compared with `assert_close(&[loss], &[loss_ref], OpClass::Optimizer, …)`.

## Throughput methodology

A deterministic timed loop (warmup + N measured steps, mean ± sample stddev) — preferred over a
criterion bench for wall-time honesty (plan item 3). tokens/s = `B·(seq_len−1)·steps / wall_secs`
(next-token positions per second). Measured for **tabi vs reference** on **(a) CPU ndarray** and
**(b) wgpu** at the reduced config (always-on) and at 160M (`#[ignore]`, run once in the wgpu lane).
The tabi-vs-reference overhead factor is reported **honestly** (P1 is a numeric gate, not perf work;
spec §15.1 claims <1% dispatch overhead — this lane measures whether that holds at 160M or explains
the gap).

## Gate tests (TDD IDs: P1 loss/throughput; HOST-9 full-model backward parity)

- **Always-on (fast, reduced):** `loss_parity_reduced_ndarray` (2-layer reduced config, 8 steps,
  ndarray) + `final_weights_parity_reduced_ndarray` + `throughput_reduced_ndarray` — wired into the
  normal `--features burn-ndarray` suite (build-guests required; the harness rebuilds guests via the
  G2 stale-guest guard / honors `SWARM_TEST_GUEST_DIR`).
- **Expensive (`#[ignore]`):** `loss_parity_within_tolerance_160m_ndarray` (≥20 steps, ndarray),
  `loss_parity_within_tolerance_160m` (≥4 steps, **wgpu**, `require_gpu!` skip),
  `throughput_within_budget_or_documented` (tabi vs reference tokens/s, both backends), and the
  160M loss-curve / det-lane evidence run. Run **once** in this session; results below + in the
  throughput doc (Merge 3 re-verifies).

## Seams M2 exports (freeze at Merge 3)

1. **Reference-parity harness API + tolerance evidence format** — `tests/reference/mod.rs`:
   `RefLlama<B: AutodiffBackend>` (`from_state_dict`, `step_loss`, `adamw`, `state_dict`),
   `drive_tabi(...)`/`drive_reference(...)` returning per-step losses + final `StateDict`,
   `assert_loss_parity(...)` (Optimizer class), `ParityReport { per_step_delta: Vec<f32>,
   final_weight_max_delta: f32, class }`. Reused verbatim across the ndarray + wgpu lanes (backend is
   the generic parameter, exactly like the G1/G2 tolerance harness).
2. **Throughput report format** — the tokens/s table shape in `swarm-p1-throughput.md`
   (path × backend × {build_s, step_s mean±sd, tokens/s, overhead×}).
3. **UMA findings docs** — `swarm-macos-uma-findings.md` + `swarm-uma-platform-findings.md` (now
   committed; the Merge-2 platform-matrix record).

## Planned slices (commits)

1. `mirror(M2): land UMA platform findings from Merge-2 investigation` — housekeeping (DONE).
2. `mirror(M2): ledger` — this file.
3. `test(train): burn reference-parity harness + reduced always-on parity (green)` — the harness
   module + reduced ndarray loss/weight parity + the fast throughput probe.
4. `test(train): 160M reference parity + throughput gate tests (green)` — the `#[ignore]` 160M
   ndarray/wgpu parity + throughput tests + `require_gpu!` skip.
5. `mirror(M2): P1 exit-gate evidence — parity deltas, tokens/s, loss series` — the throughput doc +
   this ledger's Evidence section, filled from the one-shot session runs.

## Final — base `2f1ce1f`, branch `swarm/m2`

### Commit list (oldest → newest)

| Commit | Subject |
|---|---|
| `mirror(M2): land UMA platform findings from Merge-2 investigation` | housekeeping (the two UMA docs) |
| `mirror(M2): ledger` | this file |
| `test(train): burn reference-parity harness + reduced always-on parity (green)` | harness + reduced ndarray + throughput probe + dev-dep |
| `test(train): 160M reference parity + throughput gate tests (green)` | 160M wgpu parity + throughput + loss-curve |
| `mirror(M2): P1 exit-gate evidence — parity deltas, tokens/s, loss series` | throughput doc + this Evidence section |

### Parity evidence (P1 gate — "loss within tolerance of a reference run") ✅

Tolerance bound: **`Optimizer` class** `rtol 2e-4 / atol 2e-5`. **Achieved fidelity is ~3 orders of
magnitude tighter** — the tabi (module) path is numerically transparent vs the independent burn
reference (matched init, identical TinyStories batches):

| Run | Backend | Steps | max per-step \|Δloss\| | final-weight max Δ | loss first→last |
|---|---|---:|---:|---:|---|
| reduced 2-layer (always-on) | ndarray CPU | 8 | 4.77e-7 | 2.29e-7 | 4.169 → 3.904 |
| medium d256/L4/seq128 (TinyStories, `#[ignore]`) | ndarray CPU | 20 | 4.77e-7 | 5.96e-7 | 10.838 → 7.055 |
| **160M preset (TinyStories, `#[ignore]`)** | **wgpu RADV** | **4** | **0.000 (bit-identical)** | **4.77e-7** | **10.846 → 8.986** |

The 160M wgpu run is **bit-identical loss step-for-step**; the ndarray deltas are at the f32
round-off floor (≤ 1 ULP at loss ≈ 10). The `Optimizer` tolerance is never approached.

### Throughput (tokens/s, tabi vs reference)

| Path | Backend | Config | step time | tokens/s | tabi/reference |
|---|---|---|---|---:|---:|
| tabi / reference | ndarray CPU | reduced (b2,seq9) | 0.731 / 0.443 s (8) | 175.0 / 288.9 | **1.65×** |
| tabi / reference | ndarray CPU | medium (b2,seq128) | 30.21 / 14.73 s (8) | 67.3 / 137.9 | **2.05×** |
| **tabi / reference** | **wgpu RADV** | **160M (b1,seq1024)** | **3.728±0.082 / 1.597±0.022 s** | **274.4 / 640.6** | **2.33×** |

**Overhead accounting (honest):** the 2.33× is **not** the wasm ABI dispatch (spec §15.1's "<1%" is
correct — sub-µs host calls vs multi-second kernels). It is the **`BurnBackend` per-op host-copy tax**
— `insert_result` calls `to_vec_f32`/`to_data` after every op (`burn_backend.rs` module docs: "a 2×
host copy, chosen for a clean `view` and numerical fidelity over speed this wave"), which forces a
device→host readback + queue flush per op on wgpu and serializes the GPU pipeline. It scales with
ops/step (1.65×→2.05× CPU, 2.33× GPU), consistent with a per-op copy, not a per-call dispatch. **P1
requires tokens/s be measured + reported — done; the fix (a lazy host-copy-free `OpBackend`) is a
P2/perf follow-on**, not P1 scope. Full detail: `swarm-p1-throughput.md`.

### Loss-curve evidence run (160M, wgpu, ≥2 full rounds)

`loss_curve_160m_wgpu`: 2 rounds × 30 inner AdamW steps + `make_update` + self-`ingest`, b=1
TinyStories, `build 3.0 s`. Round 0: 30 steps 107.4 s (~3.6 s/step), make_update+ingest 6.8 s →
12,463,354 B `sparse_loco` payload, inner loss **10.846 → 4.928**. Round 1: 119.5 s + 7.9 s, inner
loss **10.605 → 4.948** (the round boundary re-ascent is the expected DiLoCo-family dynamic — the
1/64-density 2-bit outer update is deliberately lossy for a single self-peer). Full 60-step series in
`swarm-p1-throughput.md`. The byte-identical **cpu-vs-wgpu det-lane digest** invariant is covered by
the G2 `wasm_backend_determinism.rs::cross_backend_wgpu` tests (the det lane never runs on the GPU).

### Frozen-file need? NO. Deviations / Merge-3 must-know

- **No frozen-file edit** (root `Cargo.toml`/`deny.toml`/`flake.nix` untouched); the one manifest
  edit is a **dev-dep** on `daemon-train-safetensors`, an *already-declared* `[workspace.dependencies]`
  entry (Merge 2) — a lane-owned change, no `cargo deny` impact. No `daemon-train/src/*` edit (the
  harness drives the tabi path through the already-public `Instance`/`WasmBackend` surface). **No
  integration-owner ledger note required.**
- **`llama-burn` was NOT added** — the reference is built directly on burn (spec §5.1/§15.3 sanction
  this; avoids a frozen `Cargo.toml` edit + a large model-zoo tree). This is the stronger independent
  reference and the P1-gate reading.
- **ndarray parity is at a medium config (not 160M)**: a 160M fp32 execute pass on CPU is
  impractically slow (Risk 3) — which is *why* 160M needs a GPU. The ndarray lane proves many-step
  (20) curve tracking cheaply; **the full-160M parity + loss-curve run is on wgpu** (≥4 parity steps
  + 2 full rounds), where the gate model belongs. Both use real TinyStories tokens.
- **`tabi@1` unspent by M2** (matches M1): the harness adds no host op / no `TABI_IMPORTS` /
  `phase.rs` change. The additive window closes at Merge 3 (spec §16) — nothing here blocks the freeze.
- **Merge 3 re-verify:** `nix develop .#vulkan --command cargo test -p daemon-train --features wgpu
  --release --test reference_parity_wgpu -- --ignored --nocapture --test-threads=1` (the P1 gate);
  the always-on `reference_parity_reduced_ndarray` rides the normal `--features burn-ndarray` suite.
  Known flake outside swarm: the `daemon-conformance` detached-delegation trio (pass-in-isolation).

## Seams exported (freeze at Merge 3) — as-built

Unchanged from the "Seams M2 exports" section above; the as-built shapes are in
`tests/reference/mod.rs` (`RefLlama::{from_state_dict,step,state_dict}`, `drive_tabi`,
`drive_reference`, `TokenBatch::{deterministic,tinystories,truncate_seq}`, `assert_parity →
ParityReport`, `throughput_stats`) and the `swarm-p1-throughput.md` table format.
